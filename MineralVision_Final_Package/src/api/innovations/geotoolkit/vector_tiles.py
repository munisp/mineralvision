"""Innovation 2 — vector-geojson-tiles: bbox-clipped GeoJSON tiles.

Features live in an in-memory store backed by best-effort loading from the
platform DB (drillhole collars / samples) when a session factory is available.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from shapely.geometry import Point, box, mapping, shape
from shapely.geometry.base import BaseGeometry

from .core import merc_to_lonlat, tile_bounds_merc, validate_tile

router = APIRouter(tags=["geotoolkit-vector-tiles"])

# In-memory feature store: layer -> list of {id, geometry(shapely, EPSG:3857), properties}
FEATURE_STORE: Dict[str, List[dict]] = {}
_id_counter = itertools.count(1)


class FeatureIn(BaseModel):
    geometry: dict = Field(..., description="GeoJSON geometry (Point or Polygon)")
    properties: dict = Field(default_factory=dict)
    crs: str = Field("EPSG:4326")


class FeatureRegisterRequest(BaseModel):
    layer: str
    features: List[FeatureIn]


def _to_merc_geom(geom: BaseGeometry, crs: str) -> BaseGeometry:
    c = crs.upper().replace(" ", "")
    if c in ("EPSG:3857", "3857"):
        return geom
    if c not in ("EPSG:4326", "4326", "WGS84"):
        raise HTTPException(422, "supported crs: EPSG:4326 or EPSG:3857")
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return shp_transform(t.transform, geom)


@router.post("/tiles/features/register")
def register_features(req: FeatureRegisterRequest):
    layer = FEATURE_STORE.setdefault(req.layer, [])
    n = 0
    for f in req.features:
        geom = shape(f.geometry)
        geom = _to_merc_geom(geom, f.crs)
        layer.append({"id": next(_id_counter),
                      "geometry": geom,
                      "properties": dict(f.properties)})
        n += 1
    return {"layer": req.layer, "registered": n,
            "total_in_layer": len(layer)}


def load_db_features() -> int:
    """Best-effort load of drillhole collars from the platform DB into the
    'drillholes' layer. Returns number of features loaded (0 when no DB)."""
    loaded = 0
    try:
        try:
            from src.api.core.database import SessionLocal  # type: ignore
        except ImportError:
            from api.core.database import SessionLocal  # type: ignore
        try:
            from src.api.drillholes import models as dh_models  # type: ignore
        except Exception:
            try:
                from api.drillholes import models as dh_models  # type: ignore
            except Exception:
                dh_models = None
        if dh_models is None:
            return 0
        session = SessionLocal()
        try:
            model = getattr(dh_models, "Drillhole", None) or getattr(dh_models, "Collar", None)
            if model is None:
                return 0
            rows = session.query(model).limit(5000).all()
            layer = FEATURE_STORE.setdefault("drillholes", [])
            existing = {f["properties"].get("hole_id") for f in layer}
            for row in rows:
                hid = getattr(row, "hole_id", None) or getattr(row, "name", None) or str(row.id)
                if hid in existing:
                    continue
                east = getattr(row, "easting", None)
                north = getattr(row, "northing", None)
                if east is None or north is None:
                    continue
                layer.append({
                    "id": next(_id_counter),
                    "geometry": Point(float(east), float(north)),
                    "properties": {"hole_id": hid,
                                   "elevation": getattr(row, "elevation", None),
                                   "source": "db"},
                })
                loaded += 1
        finally:
            session.close()
    except Exception:
        return 0
    return loaded


@router.get("/tiles/features/{z}/{x}/{y}")
def get_feature_tile(z: int, x: int, y: int,
                     layer: str = Query("drillholes"),
                     source: str = Query("all", description="all|memory|db")):
    if not validate_tile(z, x, y):
        raise HTTPException(400, "invalid z/x/y for slippy map tile scheme")
    if source in ("all", "db") and layer == "drillholes":
        load_db_features()
    minx, miny, maxx, maxy = tile_bounds_merc(z, x, y)
    tb = box(minx, miny, maxx, maxy)
    feats = []
    for rec in FEATURE_STORE.get(layer, []):
        geom = rec["geometry"]
        if not geom.intersects(tb):
            continue
        clipped = geom.intersection(tb)
        if clipped.is_empty:
            continue
        # Emit clipped geometry in EPSG:4326 for direct web-map consumption.
        from pyproj import Transformer
        from shapely.ops import transform as shp_transform
        t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        clipped_ll = shp_transform(t.transform, clipped)
        props = dict(rec["properties"])
        props["clipped"] = clipped.geom_type != geom.geom_type or not geom.equals(clipped)
        gj_geom = mapping(clipped_ll)
        feats.append({"type": "Feature", "geometry": gj_geom,
                      "properties": props})
    return {
        "type": "FeatureCollection",
        "tile": {"z": z, "x": x, "y": y},
        "layer": layer,
        "count": len(feats),
        "features": feats,
    }
