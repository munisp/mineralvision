"""HTTP layer for the JORC reporter (thin; see logic.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import logic

router = APIRouter(prefix="/innovations/jorc_reporter", tags=["jorc_reporter"])


class Point(BaseModel):
    x: float
    y: float
    z: float


class BlockIn(Point):
    grade: float
    density: float = 2.7


class RuleIn(BaseModel):
    max_range_fraction: float = Field(gt=0)
    min_samples: int = Field(ge=1)


class RulesIn(BaseModel):
    measured: RuleIn = RuleIn(max_range_fraction=0.25, min_samples=4)
    indicated: RuleIn = RuleIn(max_range_fraction=0.50, min_samples=3)
    inferred: RuleIn = RuleIn(max_range_fraction=1.00, min_samples=2)


class EllipsoidIn(BaseModel):
    radius_major: float = Field(gt=0)
    radius_minor: float = Field(gt=0)
    radius_vertical: float = Field(gt=0)
    azimuth: float = 0.0
    dip: float = 0.0


class ClassifyRequest(BaseModel):
    blocks: List[BlockIn]
    samples: List[Point]
    variogram_range: float = Field(gt=0)
    block_volume: float = Field(gt=0)
    search_ellipsoid: Optional[EllipsoidIn] = None
    rules: Optional[RulesIn] = None


class ReportRequest(ClassifyRequest):
    project_name: str = "unnamed"
    element: str = "grade"
    qaqc_summary: Optional[Dict[str, Any]] = None


def _to_logic(req: ClassifyRequest):
    search = None
    if req.search_ellipsoid is not None:
        search = logic.SearchEllipsoid(**req.search_ellipsoid.model_dump())
    rules = None
    if req.rules is not None:
        rules = logic.ClassificationRules(
            measured=logic.ClassificationRule(**req.rules.measured.model_dump()),
            indicated=logic.ClassificationRule(**req.rules.indicated.model_dump()),
            inferred=logic.ClassificationRule(**req.rules.inferred.model_dump()),
        )
    return search, rules


@router.get("/rules/defaults")
def get_default_rules() -> Dict[str, Any]:
    """Return the default classification rules and their JORC 2012 rationale."""
    rules = logic.default_rules()
    return {
        "framework": "JORC 2012",
        "rationale": (
            "JORC 2012 is principles-based and mandates no fixed drill spacing; "
            "defaults reflect common industry practice for deposits with "
            "demonstrated geological and grade continuity: Measured within "
            "0.25x variogram range with >=4 informing samples, Indicated "
            "0.5x range / >=3 samples, Inferred 1.0x range / >=2 samples. "
            "All thresholds are configurable per request."
        ),
        "rules": {
            "measured": rules.measured.__dict__,
            "indicated": rules.indicated.__dict__,
            "inferred": rules.inferred.__dict__,
        },
    }


@router.post("/classify")
def classify(req: ClassifyRequest) -> Dict[str, Any]:
    """Classify blocks and return classified blocks + per-class summary."""
    search, rules = _to_logic(req)
    try:
        classified = logic.classify_blocks(
            [b.model_dump() for b in req.blocks],
            [s.model_dump() for s in req.samples],
            variogram_range=req.variogram_range,
            search=search,
            rules=rules,
        )
        summary = logic.summarize_by_class(classified, req.block_volume)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "n_blocks": len(classified),
        "classified_blocks": [c.to_dict() for c in classified],
        "summary_by_class": summary,
    }


@router.post("/report")
def report(req: ReportRequest) -> Dict[str, Any]:
    """Full structured JORC-style classification report."""
    search, rules = _to_logic(req)
    if search is None:
        search = logic.SearchEllipsoid(
            req.variogram_range, req.variogram_range, req.variogram_range)
    if rules is None:
        rules = logic.default_rules()
    try:
        classified = logic.classify_blocks(
            [b.model_dump() for b in req.blocks],
            [s.model_dump() for s in req.samples],
            variogram_range=req.variogram_range,
            search=search,
            rules=rules,
        )
        return logic.build_report(
            classified,
            [s.model_dump() for s in req.samples],
            variogram_range=req.variogram_range,
            search=search,
            rules=rules,
            block_volume=req.block_volume,
            qaqc_summary=req.qaqc_summary,
            project_name=req.project_name,
            element=req.element,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
