"""HTTP layer for the QAQC analyzer (thin; see logic.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import logic

router = APIRouter(prefix="/innovations/qaqc_analyzer", tags=["qaqc_analyzer"])


class QaqcRow(BaseModel):
    sample_id: str
    row_type: str = Field(description="standard | blank | duplicate | sample")
    value: float
    crm_id: Optional[str] = None
    detection_limit: Optional[float] = None
    pair_id: Optional[str] = None
    # explicit duplicate-pair form:
    original_value: Optional[float] = None
    duplicate_value: Optional[float] = None
    original_id: Optional[str] = None
    duplicate_id: Optional[str] = None


class CrmSpec(BaseModel):
    mean: float
    sd: float = Field(gt=0)


class AnalyzeRequest(BaseModel):
    batch_id: str = "batch-1"
    rows: List[QaqcRow]
    crm_library: Dict[str, CrmSpec] = Field(default_factory=dict)
    detection_limit: float = Field(default=0.01, gt=0)
    blank_multiplier: float = Field(default=5.0, gt=0)


def _run_one(req: AnalyzeRequest) -> Dict[str, Any]:
    rows = [r.model_dump(exclude_none=True) for r in req.rows]
    crm_library = {k: v.model_dump() for k, v in req.crm_library.items()}
    try:
        result = logic.analyze_batch(
            rows,
            crm_library=crm_library,
            detection_limit=req.detection_limit,
            blank_multiplier=req.blank_multiplier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["batch_id"] = req.batch_id
    return result


@router.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    """Run full QAQC analysis on one batch of assay/QAQC rows."""
    return _run_one(req)


@router.post("/analyze-batches")
def analyze_batches(reqs: List[AnalyzeRequest]) -> Dict[str, Any]:
    """Run QAQC analysis over multiple batches; per-batch verdict."""
    out = [_run_one(req) for req in reqs]
    return {
        "n_batches": len(out),
        "batches": out,
        "verdicts": {b["batch_id"]: b["verdict"] for b in out},
    }
