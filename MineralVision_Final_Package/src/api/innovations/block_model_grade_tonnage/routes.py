"""HTTP layer for block model + grade-tonnage (thin; see logic.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from . import logic
from ..resource_monte_carlo.logic import VariogramSpec
from ..resource_monte_carlo.routes import VariogramIn

router = APIRouter(
    prefix="/innovations/block_model_grade_tonnage",
    tags=["block_model_grade_tonnage"])


class SampleIn(BaseModel):
    x: float
    y: float
    z: float
    grade: float


class GeometryIn(BaseModel):
    origin: List[float] = Field(min_length=3, max_length=3)
    block_size: List[float] = Field(min_length=3, max_length=3)
    n_blocks: List[int] = Field(min_length=3, max_length=3)


class BuildRequest(BaseModel):
    samples: List[SampleIn]
    geometry: GeometryIn
    variogram: VariogramIn = VariogramIn()
    density: float = Field(default=2.7, gt=0)
    density_field: Optional[List[float]] = None
    max_samples: int = Field(default=32, ge=2)


class BlockIn(BaseModel):
    x: float
    y: float
    z: float
    grade: float
    density: float = Field(default=2.7, gt=0)


class GradeTonnageRequest(BaseModel):
    """Grade-tonnage over posted blocks (or a posted kriged grid)."""
    blocks: List[BlockIn]
    block_volume: float = Field(gt=0)
    n_steps: int = Field(default=20, ge=2, le=500)
    cutoff_min: float = 0.0
    cutoff_max: Optional[float] = None


class BuildAndSweepRequest(BaseModel):
    build: BuildRequest
    n_steps: int = Field(default=20, ge=2, le=500)
    cutoff_min: float = 0.0
    cutoff_max: Optional[float] = None


def _build_blocks(req: BuildRequest) -> List[logic.Block]:
    spec = VariogramSpec(**req.variogram.model_dump())
    coords = np.array([[s.x, s.y, s.z] for s in req.samples])
    values = np.array([s.grade for s in req.samples])
    return logic.build_block_model(
        coords, values,
        origin=req.geometry.origin,
        block_size=req.geometry.block_size,
        n_blocks=req.geometry.n_blocks,
        spec=spec,
        density=req.density,
        density_field=(np.array(req.density_field)
                       if req.density_field is not None else None),
        max_samples=req.max_samples,
    )


def _block_volume(g: GeometryIn) -> float:
    return float(g.block_size[0] * g.block_size[1] * g.block_size[2])


def _posted_blocks(blocks: List[BlockIn]) -> List[logic.Block]:
    return [logic.Block(x=b.x, y=b.y, z=b.z, grade=b.grade, density=b.density,
                        kriging_variance=0.0, n_samples=0) for b in blocks]


def _gt_response(gt: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {
        "n_steps": len(gt["cutoff"]),
        "cutoff": gt["cutoff"].tolist(),
        "tonnage": gt["tonnage"].tolist(),
        "avg_grade": gt["avg_grade"].tolist(),
        "metal": gt["metal"].tolist(),
    }


@router.post("/build")
def build(req: BuildRequest) -> Dict[str, Any]:
    """Build a regular block model by ordinary kriging of sample grades."""
    try:
        blocks = _build_blocks(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "n_blocks": len(blocks),
        "block_volume": _block_volume(req.geometry),
        "blocks": [b.to_dict() for b in blocks],
    }


@router.post("/grade-tonnage")
def grade_tonnage(req: GradeTonnageRequest) -> Dict[str, Any]:
    """Grade-tonnage sweep over posted blocks (JSON)."""
    try:
        gt = logic.cutoff_sweep(
            _posted_blocks(req.blocks), req.block_volume,
            n_steps=req.n_steps,
            cutoff_min=req.cutoff_min, cutoff_max=req.cutoff_max)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _gt_response(gt)


@router.post("/grade-tonnage/csv")
def grade_tonnage_csv(req: GradeTonnageRequest) -> Response:
    """Grade-tonnage sweep as CSV download."""
    try:
        gt = logic.cutoff_sweep(
            _posted_blocks(req.blocks), req.block_volume,
            n_steps=req.n_steps,
            cutoff_min=req.cutoff_min, cutoff_max=req.cutoff_max)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=logic.grade_tonnage_csv(gt),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=grade_tonnage.csv"},
    )


@router.post("/build-and-sweep")
def build_and_sweep(req: BuildAndSweepRequest) -> Dict[str, Any]:
    """Kriged block model + grade-tonnage sweep in one call."""
    try:
        blocks = _build_blocks(req.build)
        gt = logic.cutoff_sweep(
            blocks, _block_volume(req.build.geometry),
            n_steps=req.n_steps,
            cutoff_min=req.cutoff_min, cutoff_max=req.cutoff_max)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "n_blocks": len(blocks),
        "block_volume": _block_volume(req.build.geometry),
        "blocks": [b.to_dict() for b in blocks],
        **_gt_response(gt),
    }
