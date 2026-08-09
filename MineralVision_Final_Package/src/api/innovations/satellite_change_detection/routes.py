"""HTTP layer for bi-temporal change detection (thin; see logic.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import logic

router = APIRouter(
    prefix="/innovations/satellite_change_detection",
    tags=["satellite_change_detection"])


class DetectRequest(BaseModel):
    """Two co-registered scene cubes (bands x rows x cols, identical shapes)."""
    scene_t1: List[List[List[float]]]
    scene_t2: List[List[List[float]]]
    index: str = Field(default="band", description="band | ndvi | ndmi")
    band: int = Field(default=1, ge=1)
    band_map: Optional[Dict[str, int]] = Field(
        default=None, description="logical -> 1-based band, e.g. "
        '{"nir": 4, "red": 3, "swir": 5}')
    abs_threshold: float = Field(default=0.2, ge=0)
    z_threshold: float = Field(default=2.0, ge=0)
    morph_open: int = Field(default=1, ge=0)
    morph_close: int = Field(default=1, ge=0)
    min_pixels: int = Field(default=1, ge=1)
    pixel_size: float = Field(default=30.0, gt=0)


class GeoJsonRequest(DetectRequest):
    origin_x: float = 0.0
    origin_y: float = 0.0
    crs: str = "EPSG:local"


def _run(req: DetectRequest) -> Dict[str, Any]:
    try:
        return logic.detect_changes(
            np.asarray(req.scene_t1, dtype=float),
            np.asarray(req.scene_t2, dtype=float),
            index=req.index, band=req.band, band_map=req.band_map,
            abs_threshold=req.abs_threshold, z_threshold=req.z_threshold,
            morph_open=req.morph_open, morph_close=req.morph_close,
            min_pixels=req.min_pixels, pixel_size=req.pixel_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/detect")
def detect(req: DetectRequest) -> Dict[str, Any]:
    """Detect change regions between two scene cubes."""
    return _run(req)


@router.post("/detect/geojson")
def detect_geojson(req: GeoJsonRequest) -> Dict[str, Any]:
    """Detect change regions and export as GeoJSON FeatureCollection."""
    result = _run(req)
    return logic.regions_to_geojson(result["regions"], origin_x=req.origin_x,
                                    origin_y=req.origin_y, crs=req.crs)
