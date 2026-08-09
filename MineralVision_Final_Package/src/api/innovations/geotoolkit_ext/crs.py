"""Innovation 10 — geo-crs-service.

Batch coordinate transforms (pyproj), UTM zone lookup, CRS detection
heuristics, and grid-reference (UTM/MGRS-style) geocoding.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pyproj import CRS, Transformer

try:
    from src.api.innovations.geotoolkit_ext.geo_common import utm_zone, utm_epsg_for
except ImportError:  # pragma: no cover
    from api.innovations.geotoolkit_ext.geo_common import utm_zone, utm_epsg_for

router = APIRouter()

MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"  # south->north, 8 degrees each from -80


class TransformRequest(BaseModel):
    coords: List[List[float]]   # [[x, y], ...] i.e. [lon, lat] for geographic
    src_crs: str
    dst_crs: str


class DetectRequest(BaseModel):
    coords: List[List[float]]


class GridRefRequest(BaseModel):
    grid_ref: str               # e.g. "55H 250000 7000000"


@router.post("/crs/transform")
def crs_transform(req: TransformRequest) -> Dict[str, Any]:
    try:
        src, dst = CRS.from_user_input(req.src_crs), CRS.from_user_input(req.dst_crs)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"invalid CRS: {e}")
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    out = []
    for pair in req.coords:
        if len(pair) != 2:
            raise HTTPException(status_code=422, detail="each coord must be [x, y]")
        x, y = transformer.transform(pair[0], pair[1])
        out.append([float(x), float(y)])
    return {
        "src_crs": src.to_string(),
        "dst_crs": dst.to_string(),
        "coords": out,
        "count": len(out),
    }


@router.get("/crs/utm-zone")
def crs_utm_zone(lon: float = Query(...), lat: float = Query(...)) -> Dict[str, Any]:
    if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
        raise HTTPException(status_code=422, detail="lon/lat out of range")
    zone, north = utm_zone(lon, lat)
    return {
        "lon": lon, "lat": lat,
        "utm_zone": zone,
        "hemisphere": "north" if north else "south",
        "epsg": (32600 + zone) if north else (32700 + zone),
    }


def _ranges(coords: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), max(xs), min(ys), max(ys)


@router.post("/crs/detect")
def crs_detect(req: DetectRequest) -> Dict[str, Any]:
    if not req.coords:
        raise HTTPException(status_code=422, detail="no coordinates supplied")
    minx, maxx, miny, maxy = _ranges(req.coords)
    candidates: List[Dict[str, Any]] = []

    if -180 <= minx and maxx <= 180 and -90 <= miny and maxy <= 90:
        candidates.append({
            "epsg": 4326, "name": "WGS 84 geographic",
            "confidence": 0.9,
            "reasoning": "all |x|<=180 and |y|<=90 — consistent with lon/lat degrees",
        })
    # UTM-style ranges: easting 100k-900k, northing 0-10M
    if all(0 <= c[0] <= 1000000 for c in req.coords) and all(0 <= c[1] <= 10000000 for c in req.coords)             and not (maxx <= 180 and maxy <= 90):
        mid_lon_est = None
        candidates.append({
            "epsg": None,
            "name": "UTM (zone undetermined)",
            "confidence": 0.6,
            "reasoning": "eastings within 0-1,000,000 and northings within 0-10,000,000 — "
                         "consistent with UTM metres; zone cannot be determined from ranges alone",
        })
        # if a plausible zone can be guessed from longitude-looking spread, offer zones
    if all(0 <= c[0] <= 1000000 for c in req.coords) and all(0 <= c[1] <= 10000000 for c in req.coords):
        for hemi, epsg_base in (("north", 32600), ("south", 32700)):
            candidates.append({
                "epsg": f"{epsg_base}+zone",
                "name": f"UTM {hemi} hemisphere",
                "confidence": 0.3,
                "reasoning": f"northing values compatible with UTM {hemi} hemisphere metre ranges",
            })
    if any(abs(c[0]) > 180 or abs(c[1]) > 90 for c in req.coords):
        candidates.append({
            "epsg": 3857, "name": "Web Mercator",
            "confidence": 0.4 if (max(abs(c[0]) for c in req.coords) <= 20037508
                                  and max(abs(c[1]) for c in req.coords) <= 20037508) else 0.1,
            "reasoning": "magnitudes exceed degree ranges — projected CRS; Web Mercator "
                         "plausible if within +-20037508 m",
        })
    candidates.sort(key=lambda c: -c["confidence"])
    return {
        "count": len(req.coords),
        "ranges": {"x": [minx, maxx], "y": [miny, maxy]},
        "candidates": candidates,
        "best_guess": candidates[0] if candidates else None,
    }


_GRIDREF_RE = re.compile(
    r"^\s*(\d{1,2})\s*([C-HJ-NP-X])\s+(\d{2,7})\s+(\d{2,7})\s*$", re.IGNORECASE)


def _band_to_lat_range(band: str) -> Tuple[float, bool]:
    idx = MGRS_BANDS.index(band.upper())
    south_edge = -80.0 + idx * 8.0
    return south_edge, south_edge + 8.0 >= 0.0 and band.upper() >= "N" or band.upper() >= "N"


@router.post("/geocode/grid-ref")
def geocode_grid_ref(req: GridRefRequest) -> Dict[str, Any]:
    """Parse UTM/MGRS-style grid references like '55H 250000 7000000'.

    Zone number + latitude band letter determine the hemisphere; easting and
    northing are UTM metres. Returns lat/lon via pyproj.
    """
    m = _GRIDREF_RE.match(req.grid_ref)
    if not m:
        raise HTTPException(
            status_code=422,
            detail="unparseable grid reference; expected format like '55H 250000 7000000'")
    zone = int(m.group(1))
    band = m.group(2).upper()
    if not (1 <= zone <= 60):
        raise HTTPException(status_code=422, detail="UTM zone must be 1-60")
    if band in ("I", "O"):
        raise HTTPException(status_code=422, detail="invalid MGRS latitude band (I/O not used)")
    idx = MGRS_BANDS.index(band)
    band_south_lat = -80.0 + idx * 8.0
    northern = band_south_lat >= 0.0
    easting = float(m.group(3))
    northing = float(m.group(4))
    if not (0 <= easting <= 1000000):
        raise HTTPException(status_code=422, detail="easting out of UTM range")
    if not (0 <= northing <= 10000000):
        raise HTTPException(status_code=422, detail="northing out of UTM range")

    epsg = (32600 + zone) if northern else (32700 + zone)
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    # sanity: decoded latitude should fall inside the band (allow band edge tolerance)
    if not (band_south_lat - 1.0 <= lat <= band_south_lat + 9.0):
        raise HTTPException(
            status_code=422,
            detail=f"decoded latitude {lat:.3f} inconsistent with band {band} "
                   f"({band_south_lat}..{band_south_lat + 8})")
    return {
        "grid_ref": req.grid_ref,
        "zone": zone,
        "band": band,
        "hemisphere": "north" if northern else "south",
        "easting": easting,
        "northing": northing,
        "epsg": epsg,
        "lon": float(lon),
        "lat": float(lat),
        "lat_band_range": [band_south_lat, band_south_lat + 8.0],
    }
