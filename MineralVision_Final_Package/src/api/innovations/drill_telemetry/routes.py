"""HTTP layer for rig telemetry ingestion & auto-logging (thin)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import logic
from .models import TelemetryPoint, get_session

router = APIRouter(
    prefix="/innovations/drill_telemetry", tags=["drill_telemetry"])


class TelemetryPointIn(BaseModel):
    timestamp: str
    depth: float
    rop: float
    torque: float = 0.0
    rpm: float = 0.0
    vibration: float = 0.0


class IngestRequest(BaseModel):
    points: List[TelemetryPointIn]


class AutoLogRequest(BaseModel):
    collar_depth: float = 0.0
    planned_total_depth: Optional[float] = None
    deviation_tolerance: float = Field(default=0.5, ge=0)
    threshold_factor: float = Field(default=2.0, gt=0)
    min_segment: int = Field(default=8, ge=2)


@router.post("/rigs/{rig_id}/telemetry", status_code=201)
def ingest_batch(rig_id: str, req: IngestRequest,
                 session: Session = Depends(get_session)) -> Dict[str, Any]:
    """Ingest a batch of rig telemetry points (stored via SQLAlchemy)."""
    if not req.points:
        raise HTTPException(status_code=422, detail="empty batch")
    rows = [
        TelemetryPoint(rig_id=rig_id, timestamp=p.timestamp, depth=p.depth,
                       rop=p.rop, torque=p.torque, rpm=p.rpm,
                       vibration=p.vibration)
        for p in req.points
    ]
    session.add_all(rows)
    session.commit()
    return {"rig_id": rig_id, "n_ingested": len(rows)}


@router.get("/rigs/{rig_id}/telemetry")
def get_telemetry(rig_id: str,
                  session: Session = Depends(get_session)) -> Dict[str, Any]:
    """List stored telemetry for a rig (depth-ordered)."""
    rows = (session.query(TelemetryPoint)
            .filter(TelemetryPoint.rig_id == rig_id)
            .order_by(TelemetryPoint.depth, TelemetryPoint.id).all())
    return {"rig_id": rig_id, "n_points": len(rows),
            "points": [r.to_dict() for r in rows]}


@router.post("/rigs/{rig_id}/auto-log")
def auto_log(rig_id: str, req: AutoLogRequest,
             session: Session = Depends(get_session)) -> Dict[str, Any]:
    """CUSUM regime segmentation of the stored series -> interval table.

    Depths are aligned to the collar; the MWD-vs-collar deviation check
    compares aligned final depth with the planned total depth.
    """
    rows = (session.query(TelemetryPoint)
            .filter(TelemetryPoint.rig_id == rig_id)
            .order_by(TelemetryPoint.depth, TelemetryPoint.id).all())
    if len(rows) < 3:
        raise HTTPException(
            status_code=422,
            detail="need at least 3 telemetry points for segmentation")

    depth = np.array([r.depth for r in rows])
    rop = np.array([r.rop for r in rows])
    torque = np.array([r.torque for r in rows])
    rpm = np.array([r.rpm for r in rows])
    vibration = np.array([r.vibration for r in rows])

    try:
        intervals = logic.segment_intervals(
            depth, rop, torque, rpm, vibration,
            threshold_factor=req.threshold_factor,
            min_segment=req.min_segment)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = logic.align_to_collar(
        intervals,
        collar_depth=req.collar_depth,
        planned_total_depth=req.planned_total_depth,
        deviation_tolerance=req.deviation_tolerance,
        final_measured_depth=float(depth.max()),
    )
    return {
        "rig_id": rig_id,
        "n_points": len(rows),
        "n_intervals": len(result["intervals"]),
        **result,
    }
