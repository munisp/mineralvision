"""2. Collar snapping/validation: DTM bilinear elevation, slope constraint,
keep-out polygons (shapely)."""

import math
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from shapely.geometry import shape, Point

try:
    from src.api.innovations.drill_planner.common import LocalProjector
except ImportError:  # pragma: no cover
    from api.innovations.drill_planner.common import LocalProjector

router = APIRouter()


class CollarIn(BaseModel):
    id: Optional[str] = None
    lon: float
    lat: float


class SnapRequest(BaseModel):
    collars: List[CollarIn]
    dtm: Optional[List[List[float]]] = None          # rows north->south
    dtm_bounds: Optional[List[float]] = None         # [minlon,minlat,maxlon,maxlat]
    max_slope_deg: float = 25.0
    keepouts: Optional[Dict[str, Any]] = None        # GeoJSON FeatureCollection


def bilinear(dtm: np.ndarray, minx, miny, maxx, maxy, x, y) -> float:
    """Bilinear interpolation of grid value at (x, y); NaN outside."""
    rows, cols = dtm.shape
    gx = (x - minx) / (maxx - minx) * (cols - 1)
    gy = (maxy - y) / (maxy - miny) * (rows - 1)  # row 0 = north edge
    if gx < 0 or gy < 0 or gx > cols - 1 or gy > rows - 1:
        return float("nan")
    x0, y0 = int(math.floor(gx)), int(math.floor(gy))
    x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
    fx, fy = gx - x0, gy - y0
    return float((dtm[y0, x0] * (1 - fx) * (1 - fy)
                  + dtm[y0, x1] * fx * (1 - fy)
                  + dtm[y1, x0] * (1 - fx) * fy
                  + dtm[y1, x1] * fx * fy))


def slope_grid(dtm: np.ndarray, cell_x_m: float, cell_y_m: float) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(dtm, cell_y_m, cell_x_m)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))


@router.post("/collars/snap")
def collars_snap(req: SnapRequest) -> Dict[str, Any]:
    if not req.collars:
        raise HTTPException(status_code=422, detail="no collars supplied")
    keepout_geoms = []
    if req.keepouts:
        feats = req.keepouts.get("features", [])
        keepout_geoms = [(shape(f["geometry"]), f.get("properties") or {}) for f in feats]

    dtm = np.asarray(req.dtm, float) if req.dtm is not None else None
    slopes = None
    cell_x = cell_y = None
    proj = None
    if dtm is not None:
        if req.dtm_bounds is None or len(req.dtm_bounds) != 4:
            raise HTTPException(status_code=422, detail="dtm_bounds [minlon,minlat,maxlon,maxlat] required with dtm")
        b = req.dtm_bounds
        rows, cols = dtm.shape
        proj = LocalProjector((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        # cell size in metres via UTM edge lengths
        ex0, ey0 = proj.to_m(b[0], b[1])
        ex1, ey1 = proj.to_m(b[2], b[3])
        cell_x = abs(ex1 - ex0) / (cols - 1)
        cell_y = abs(ey1 - ey0) / (rows - 1)
        slopes = slope_grid(dtm, cell_x, cell_y)

    accepted, rejected = [], []
    for i, col in enumerate(req.collars):
        cid = col.id or f"C{i:04d}"
        reasons = []
        elev = slope = None
        if dtm is not None:
            b = req.dtm_bounds
            elev = bilinear(dtm, b[0], b[1], b[2], b[3], col.lon, col.lat)
            if math.isnan(elev):
                reasons.append("outside_dtm_extent")
            else:
                slope = bilinear(slopes, b[0], b[1], b[2], b[3], col.lon, col.lat)
                if slope > req.max_slope_deg:
                    reasons.append(f"slope {slope:.1f}deg > max {req.max_slope_deg}deg")
        pt = Point(col.lon, col.lat)
        for geom, props in keepout_geoms:
            if geom.covers(pt):
                reasons.append(f"inside keep-out '{props.get('name', 'unnamed')}'")
        rec = {"id": cid, "lon": col.lon, "lat": col.lat,
               "elevation": elev, "slope_deg": slope}
        if reasons:
            rec["reasons"] = reasons
            rejected.append(rec)
        else:
            accepted.append(rec)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "max_slope_deg": req.max_slope_deg,
        "cell_size_m": [cell_x, cell_y] if cell_x else None,
    }
