"""HTTP layer for resource Monte Carlo simulation (thin; see logic.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import logic

router = APIRouter(
    prefix="/innovations/resource_monte_carlo", tags=["resource_monte_carlo"])


class VariogramIn(BaseModel):
    model: str = Field(default="spherical",
                       pattern="^(spherical|exponential|gaussian)$")
    nugget: float = Field(default=0.0, ge=0)
    contribution: float = Field(default=1.0, gt=0)
    range: float = Field(default=100.0, gt=0)


class Point3(BaseModel):
    x: float
    y: float
    z: float


class DataPoint(Point3):
    value: float


class GridNode(Point3):
    tonnage: float = Field(gt=0, description="tonnage represented by the node "
                                             "(block volume x density)")


class SimulateRequest(BaseModel):
    data: List[DataPoint]
    grid: List[GridNode]
    variogram: VariogramIn = VariogramIn()
    n_realizations: int = Field(default=100, ge=1, le=2000)
    seed: Optional[int] = None
    cutoff: float = 0.0


def _run(req: SimulateRequest):
    spec = logic.VariogramSpec(**req.variogram.model_dump())
    data_coords = np.array([[d.x, d.y, d.z] for d in req.data])
    data_values = np.array([d.value for d in req.data])
    grid_coords = np.array([[g.x, g.y, g.z] for g in req.grid])
    block_tonnages = np.array([g.tonnage for g in req.grid])
    sim = logic.conditional_simulation(
        data_coords, data_values, grid_coords, spec,
        n_realizations=req.n_realizations, seed=req.seed)
    return sim, block_tonnages


@router.post("/simulate")
def simulate(req: SimulateRequest) -> Dict[str, Any]:
    """Conditional Gaussian simulation + P10/P50/P90 above cutoff."""
    try:
        sim, block_tonnages = _run(req)
        summary = logic.uncertainty_summary(sim, block_tonnages, req.cutoff)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        **summary,
        "conditional_mean": sim["conditional_mean"].tolist(),
        "conditional_std": sim["conditional_std"].tolist(),
    }


@router.post("/realizations")
def realizations(req: SimulateRequest) -> Dict[str, Any]:
    """Return the raw realization matrix (n_grid x n_realizations)."""
    try:
        sim, _ = _run(req)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "n_grid": int(sim["realizations"].shape[0]),
        "n_realizations": int(sim["realizations"].shape[1]),
        "seed": req.seed,
        "realizations": sim["realizations"].tolist(),
        "conditional_mean": sim["conditional_mean"].tolist(),
        "conditional_std": sim["conditional_std"].tolist(),
    }
