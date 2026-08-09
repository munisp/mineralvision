"""HTTP layer for geostatistical drift alerting (thin — see logic.py)."""

from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .logic import registry

router = APIRouter(prefix="/innovations/geostat_drift_alerting",
                   tags=["geostat_drift_alerting"])


class RegisterStreamRequest(BaseModel):
    stream_id: str = Field(..., min_length=1, max_length=200)
    baseline_values: List[float] = Field(..., min_length=8)
    baseline_coords: Optional[List[Tuple[float, float]]] = None
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    ewma_lam: float = 0.2
    ewma_L: float = 3.0
    window: int = 50
    mean_tol: float = 0.5
    sill_tol: float = 0.5


class PushBatchRequest(BaseModel):
    values: List[float] = Field(..., min_length=1)
    coords: Optional[List[Tuple[float, float]]] = None
    timestamps: Optional[List[float]] = None


@router.post("/streams")
def register_stream(req: RegisterStreamRequest):
    if registry.get(req.stream_id) is not None:
        raise HTTPException(status_code=409, detail="stream already registered")
    try:
        monitor = registry.register(
            req.stream_id, req.baseline_values,
            baseline_coords=req.baseline_coords,
            cusum_k=req.cusum_k, cusum_h=req.cusum_h,
            ewma_lam=req.ewma_lam, ewma_L=req.ewma_L,
            window=req.window, mean_tol=req.mean_tol, sill_tol=req.sill_tol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"stream_id": req.stream_id, "baseline_mean": monitor.base_mean,
            "baseline_std": monitor.base_std, "baseline_sill": monitor.base_sill}


@router.post("/streams/{stream_id}/batches")
def push_batch(stream_id: str, req: PushBatchRequest):
    monitor = registry.get(stream_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail="stream not found")
    try:
        alerts = monitor.push(req.values, coords=req.coords,
                              timestamps=req.timestamps)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"stream_id": stream_id, "n_samples": monitor.n_seen,
            "alerts": alerts}


@router.get("/streams/{stream_id}/alerts")
def list_alerts(stream_id: str):
    monitor = registry.get(stream_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail="stream not found")
    return {"stream_id": stream_id, "alerts": monitor.list_alerts()}


@router.delete("/streams/{stream_id}")
def delete_stream(stream_id: str):
    if not registry.remove(stream_id):
        raise HTTPException(status_code=404, detail="stream not found")
    return {"deleted": stream_id}
