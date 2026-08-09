"""HTTP layer for hyperspectral alteration mapping (thin; see logic.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import logic

router = APIRouter(
    prefix="/innovations/hyperspectral_alteration",
    tags=["hyperspectral_alteration"])


class BandRatio(BaseModel):
    numerator: int = Field(ge=1)
    denominator: int = Field(ge=1)


class MapRequest(BaseModel):
    """Multi-band cube as nested lists: bands x rows x cols."""
    cube: List[List[List[float]]]
    index: str = Field(description="clay | iron_oxide | carbonate | ndvi")
    preset: str = "aster"
    threshold: Optional[float] = None
    band_map: Optional[Dict[str, BandRatio]] = None
    ndvi_mask_threshold: Optional[float] = 0.3
    min_pixels: int = Field(default=1, ge=1)
    pixel_size: float = Field(default=15.0, gt=0)


class GeoJsonRequest(MapRequest):
    origin_x: float = 0.0
    origin_y: float = 0.0
    crs: str = "EPSG:local"


def _run(req: MapRequest) -> Dict[str, Any]:
    cube = np.asarray(req.cube, dtype=float)
    band_map = None
    if req.band_map is not None:
        band_map = {k: (v.numerator, v.denominator)
                    for k, v in req.band_map.items()}
    try:
        return logic.map_alteration_zones(
            cube, req.index, preset=req.preset, threshold=req.threshold,
            band_map=band_map,
            ndvi_mask_threshold=req.ndvi_mask_threshold,
            min_pixels=req.min_pixels, pixel_size=req.pixel_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/presets")
def get_presets() -> Dict[str, Any]:
    """Available sensor band presets and default thresholds."""
    return {
        "presets": {k: {i: {"numerator": v[0], "denominator": v[1]}
                        for i, v in m.items()}
                    for k, m in logic.BAND_PRESETS.items()},
        "default_thresholds": logic.DEFAULT_THRESHOLDS,
        "convention": "1-based band numbers into the posted cube",
    }


@router.post("/indices")
def compute_indices(req: MapRequest) -> Dict[str, Any]:
    """Compute a single alteration index array (no zoning)."""
    cube = np.asarray(req.cube, dtype=float)
    band_map = None
    if req.band_map is not None:
        band_map = {k: (v.numerator, v.denominator)
                    for k, v in req.band_map.items()}
    elif req.preset in logic.BAND_PRESETS:
        band_map = logic.BAND_PRESETS[req.preset]
    else:
        raise HTTPException(status_code=422,
                            detail=f"unknown preset: {req.preset}")
    try:
        idx = logic.compute_index(cube, req.index, band_map)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"index": req.index, "shape": list(idx.shape),
            "min": float(idx.min()), "max": float(idx.max()),
            "mean": float(idx.mean()), "values": idx.tolist()}


@router.post("/map")
def map_zones(req: MapRequest) -> Dict[str, Any]:
    """Detect alteration zones (threshold + connected components)."""
    return _run(req)


@router.post("/map/geojson")
def map_zones_geojson(req: GeoJsonRequest) -> Dict[str, Any]:
    """Detect alteration zones and export as GeoJSON FeatureCollection."""
    result = _run(req)
    return logic.zones_to_geojson(result["zones"], origin_x=req.origin_x,
                                  origin_y=req.origin_y, crs=req.crs)
