"""Innovation 6 — spatial-overlay service.

Layer overlay analysis (intersect | union | erase | clip) on GeoJSON
FeatureCollections (or registered layer refs) using shapely, with
area-weighted attribute transfer for intersect.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry

try:
    from src.api.innovations.geotoolkit_ext.geo_common import (
        fc_to_geometries,
        geom_to_feature,
        looks_geographic,
        area_m2,
        utm_epsg_for,
    )
except ImportError:  # pragma: no cover
    from api.innovations.geotoolkit_ext.geo_common import (
        fc_to_geometries,
        geom_to_feature,
        looks_geographic,
        area_m2,
        utm_epsg_for,
    )

router = APIRouter()

# Simple in-memory layer registry so requests may reference layers by name.
_LAYER_REGISTRY: Dict[str, Dict[str, Any]] = {}


class LayerRegisterRequest(BaseModel):
    name: str
    feature_collection: Dict[str, Any]


class OverlayRequest(BaseModel):
    layer_a: Optional[Dict[str, Any]] = None  # GeoJSON FeatureCollection
    layer_b: Optional[Dict[str, Any]] = None
    layer_a_ref: Optional[str] = None
    layer_b_ref: Optional[str] = None
    # properties of layer_b to transfer onto intersect output (area-weighted)
    transfer_properties: Optional[List[str]] = None


def _resolve_layer(fc: Optional[Dict[str, Any]], ref: Optional[str]) -> Dict[str, Any]:
    if fc is not None:
        return fc
    if ref is not None:
        if ref not in _LAYER_REGISTRY:
            raise HTTPException(status_code=404, detail=f"layer ref '{ref}' not registered")
        return _LAYER_REGISTRY[ref]
    raise HTTPException(status_code=422, detail="provide a FeatureCollection or a layer ref")


def _numeric_props(prop_dicts: List[Dict[str, Any]]) -> List[str]:
    """Numeric property keys across a list of properties dicts."""
    keys = set()
    for props in prop_dicts:
        for k, v in props.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    return sorted(keys)


@router.post("/overlay/layers/register")
def register_layer(req: LayerRegisterRequest) -> Dict[str, Any]:
    if req.feature_collection.get("type") != "FeatureCollection":
        raise HTTPException(status_code=422, detail="feature_collection must be a GeoJSON FeatureCollection")
    _LAYER_REGISTRY[req.name] = req.feature_collection
    return {"registered": req.name, "feature_count": len(req.feature_collection.get("features", []))}


@router.get("/overlay/layers")
def list_layers() -> Dict[str, Any]:
    return {name: len(fc.get("features", [])) for name, fc in _LAYER_REGISTRY.items()}


@router.post("/overlay/{operation}")
def overlay(operation: str, req: OverlayRequest) -> Dict[str, Any]:
    operation = operation.lower()
    if operation not in ("intersect", "union", "erase", "clip"):
        raise HTTPException(status_code=422, detail="operation must be one of intersect|union|erase|clip")

    fc_a = _resolve_layer(req.layer_a, req.layer_a_ref)
    fc_b = _resolve_layer(req.layer_b, req.layer_b_ref)
    feats_a = fc_to_geometries(fc_a)
    feats_b = fc_to_geometries(fc_b)
    if not feats_a or not feats_b:
        raise HTTPException(status_code=422, detail="both layers must contain at least one feature")

    geoms_a = [g for g, _ in feats_a]
    geographic = looks_geographic(geoms_a + [g for g, _ in feats_b])
    epsg = utm_epsg_for(geoms_a[0].centroid.x, geoms_a[0].centroid.y) if geographic else None

    out_features: List[Dict[str, Any]] = []
    transfer = req.transfer_properties
    if operation == "intersect" and transfer is None:
        transfer = _numeric_props([p for _, p in feats_b])

    for geom_a, props_a in feats_a:
        for geom_b, props_b in feats_b:
            if operation == "intersect":
                if not geom_a.intersects(geom_b):
                    continue
                inter = geom_a.intersection(geom_b)
                if inter.is_empty or inter.area <= 0:
                    continue
                props = dict(props_a)
                overlap = inter.area
                frac = overlap / geom_b.area if geom_b.area > 0 else 0.0
                # area-weighted attribute transfer from b
                for key in (transfer or []):
                    val = props_b.get(key)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        props[f"{key}_weighted"] = val * frac
                        props[f"{key}_source"] = val
                props["overlap_fraction_of_b"] = frac
                out_features.append((inter, props, overlap))
            elif operation == "union":
                if not geom_a.intersects(geom_b):
                    continue
                u = geom_a.union(geom_b)
                if u.is_empty:
                    continue
                props = {f"a_{k}": v for k, v in props_a.items()}
                props.update({f"b_{k}": v for k, v in props_b.items()})
                out_features.append((u, props, u.area))
            elif operation in ("erase", "clip"):
                # erase: a minus b ; clip: a clipped to b (== intersect of geometries,
                # but attributes come solely from a)
                if not geom_a.intersects(geom_b):
                    if operation == "erase":
                        out_features.append((geom_a, dict(props_a), geom_a.area))
                    continue
                res = geom_a.difference(geom_b) if operation == "erase" else geom_a.intersection(geom_b)
                if res.is_empty or res.area <= 0:
                    continue
                out_features.append((res, dict(props_a), res.area))

    # flatten multi-part results into individual features with per-feature area stats
    final_features: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []
    idx = 0
    for geom, props, _area in out_features:
        parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            if part.is_empty or part.area <= 0:
                continue
            props2 = dict(props)
            props2["area"] = part.area
            f = geom_to_feature(part, props2)
            f["id"] = idx
            final_features.append(f)
            st = {"feature_index": idx, "area": part.area,
                  "centroid": [part.centroid.x, part.centroid.y]}
            if epsg is not None:
                st["area_m2"] = area_m2(part, epsg)
            stats.append(st)
            idx += 1

    return {
        "operation": operation,
        "feature_count": len(final_features),
        "result": {"type": "FeatureCollection", "features": final_features},
        "stats": stats,
        "total_area": sum(s["area"] for s in stats),
        "area_units": "m2" if epsg is None else "input-crs-units (area_m2 given per feature)",
        "crs": {"geographic_input": geographic, "metric_epsg": epsg},
    }
