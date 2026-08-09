"""HTTP layer for marine sonar / bathymetry (thin; see logic.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import logic

router = APIRouter(prefix="/innovations/marine-sonar", tags=["marine-sonar"])


def _err(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class BathymetryProcessRequest(BaseModel):
    pings: List[List[float]] = Field(min_length=4)
    grid_shape: Tuple[int, int] = (100, 100)
    median_size: int = Field(default=5, ge=1)
    spike_threshold: float = Field(default=5.0, gt=0)


class TerrainRequest(BaseModel):
    grid: List[List[float]]
    cell_size: float = Field(default=1.0, gt=0)
    bpi_inner: int = Field(default=2, ge=1)
    bpi_outer: int = Field(default=8, ge=2)


class BackscatterRequest(BaseModel):
    mosaic: List[List[float]]
    n_classes: int = Field(default=3, ge=2, le=8)
    window: int = Field(default=9, ge=3)
    slope: Optional[List[List[float]]] = None
    seed: int = 42


class FeatureDetectRequest(BaseModel):
    grid: List[List[float]]
    cell_size: float = Field(default=1.0, gt=0)
    relief_threshold: float = Field(default=1.0, gt=0)
    smooth_window: int = Field(default=15, ge=3)
    min_area_cells: int = Field(default=6, ge=1)
    min_elongation: float = Field(default=3.0, ge=1.0)


class TargetScoreRequest(BaseModel):
    grid: List[List[float]]
    rugosity: List[List[float]]
    slope: List[List[float]]
    class_map: Optional[List[List[int]]] = None
    model: str = "placer_gold"
    top_k: int = Field(default=5, ge=1, le=50)


@router.post("/bathymetry/process")
def bathymetry_process(req: BathymetryProcessRequest) -> Dict[str, Any]:
    """Grid a raw ping cloud into a DTM with spike filtering/artifact flags."""
    res = _err(
        logic.process_bathymetry,
        pings=req.pings,
        grid_shape=tuple(req.grid_shape),
        median_size=req.median_size,
        spike_threshold=req.spike_threshold,
    )
    return {
        "grid": res["grid"].tolist(),
        "artifact_mask": res["artifact_mask"].astype(int).tolist(),
        "intensity_grid": (
            res["intensity_grid"].tolist()
            if res["intensity_grid"] is not None
            else None
        ),
        "x_coords": res["x_coords"].tolist(),
        "y_coords": res["y_coords"].tolist(),
        "stats": res["stats"],
    }


@router.post("/bathymetry/terrain")
def bathymetry_terrain(req: TerrainRequest) -> Dict[str, Any]:
    """Terrain derivatives: slope, aspect, rugosity, BPI, hillshade."""
    grid = _err(logic.as_rectangular_grid, req.grid)
    res = _err(
        logic.terrain_derivatives,
        grid,
        cell_size=req.cell_size,
        bpi_inner=req.bpi_inner,
        bpi_outer=req.bpi_outer,
    )
    return {k: v.tolist() for k, v in res.items()}


@router.post("/backscatter/classify")
def backscatter_classify(req: BackscatterRequest) -> Dict[str, Any]:
    """Unsupervised seafloor classification from backscatter texture."""
    mosaic = _err(logic.as_rectangular_grid, req.mosaic, "mosaic")
    slope = None
    if req.slope is not None:
        slope = _err(logic.as_rectangular_grid, req.slope, "slope")
    res = _err(
        logic.classify_backscatter,
        mosaic,
        n_classes=req.n_classes,
        window=req.window,
        slope_grid=slope,
        seed=req.seed,
    )
    return {
        "class_map": res["class_map"].astype(int).tolist(),
        "class_stats": res["class_stats"],
    }


@router.post("/features/detect")
def features_detect(req: FeatureDetectRequest) -> Dict[str, Any]:
    """Detect relief maxima (pinnacles/reefs) and linear features (channels)."""
    grid = _err(logic.as_rectangular_grid, req.grid)
    res = _err(
        logic.detect_features,
        grid,
        cell_size=req.cell_size,
        relief_threshold=req.relief_threshold,
        smooth_window=req.smooth_window,
        min_area_cells=req.min_area_cells,
        min_elongation=req.min_elongation,
    )
    return {
        "features": res["features"],
        "n_features": res["n_features"],
        "dominant_lineament_orientation_deg": res[
            "dominant_lineament_orientation_deg"
        ],
        "residual_grid": res["residual_grid"].tolist(),
    }


@router.post("/targets/score")
def targets_score(req: TargetScoreRequest) -> Dict[str, Any]:
    """Composite marine prospectivity for a deposit-model preset."""
    grid = _err(logic.as_rectangular_grid, req.grid)
    rug = _err(logic.as_rectangular_grid, req.rugosity, "rugosity")
    slp = _err(logic.as_rectangular_grid, req.slope, "slope")
    cm = _err(logic.as_rectangular_grid, req.class_map, "class_map") if req.class_map else None
    res = _err(
        logic.score_targets,
        grid,
        rugosity=rug,
        slope=slp,
        class_map=cm,
        model=req.model,
        top_k=req.top_k,
    )
    return {
        "score_grid": res["score_grid"].tolist(),
        "component_grids": {
            k: (v.tolist() if v is not None else None)
            for k, v in res["component_grids"].items()
        },
        "top_zones": res["top_zones"],
        "model": res["model"],
    }


@router.get("/deposit-models")
def get_deposit_models() -> Dict[str, Any]:
    """Marine deposit model presets (placer gold/diamond/tin, SMS, nodules)."""
    return {"models": logic.deposit_models()}
