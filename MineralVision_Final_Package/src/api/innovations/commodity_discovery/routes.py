"""HTTP layer for commodity discovery (thin — see logic.py)."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .logic import (
    GoldDepositType,
    LithiumDepositType,
    analyze_brine_samples,
    compute_gold_alteration,
    compute_regolith,
    list_deposit_types,
    run_discovery_workflow,
    score_gold_samples,
    score_pegmatite_samples,
)

router = APIRouter(prefix="/innovations/commodity-discovery", tags=["commodity-discovery"])


class GeochemSampleIn(BaseModel):
    sample_id: str
    x: float
    y: float
    z: Optional[float] = None
    sample_type: str = "soil"
    elements: Dict[str, float] = Field(default_factory=dict)
    units: Dict[str, str] = Field(default_factory=dict)


class GoldScoreRequest(BaseModel):
    deposit_type: GoldDepositType
    samples: List[GeochemSampleIn] = Field(..., min_length=1)


@router.post("/gold/score-samples")
def gold_score_samples(req: GoldScoreRequest) -> Dict[str, Any]:
    """Batch-score geochem samples for a gold deposit type (real engine)."""
    results = score_gold_samples([s.model_dump() for s in req.samples], req.deposit_type)
    return {
        "deposit_type": req.deposit_type.value,
        "n_samples": len(results),
        "results": results,
    }


class GoldAlterationRequest(BaseModel):
    hyperspectral_data: Optional[List[List[List[float]]]] = None
    wavelengths: Optional[List[float]] = None
    spectral_indices: Optional[List[str]] = None
    geochem_samples: Optional[List[GeochemSampleIn]] = None
    geochem_indices: Optional[List[str]] = None


@router.post("/gold/alteration")
def gold_alteration(req: GoldAlterationRequest) -> Dict[str, Any]:
    """Alteration classification from hyperspectral array and/or geochem."""
    if req.hyperspectral_data is None and not req.geochem_samples:
        raise HTTPException(
            status_code=422,
            detail="provide hyperspectral_data and/or geochem_samples",
        )
    try:
        return compute_gold_alteration(
            hyperspectral_data=req.hyperspectral_data,
            wavelengths=req.wavelengths,
            spectral_indices=req.spectral_indices,
            geochem_samples=([s.model_dump() for s in req.geochem_samples]
                             if req.geochem_samples else None),
            geochem_indices=req.geochem_indices,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class GoldRegolithRequest(BaseModel):
    dem: List[List[float]]
    slope: Optional[List[List[float]]] = None
    curvature: Optional[List[List[float]]] = None
    rainfall: float = Field(500.0, gt=0)
    cell_size: float = Field(25.0, gt=0)
    drainage_distance: Optional[List[List[float]]] = None


@router.post("/gold/regolith")
def gold_regolith(req: GoldRegolithRequest) -> Dict[str, Any]:
    """Regolith thickness + classification from a DEM (real RegolithModel)."""
    try:
        return compute_regolith(
            dem=req.dem,
            slope=req.slope,
            curvature=req.curvature,
            rainfall=req.rainfall,
            cell_size=req.cell_size,
            drainage_distance=req.drainage_distance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class PegmatiteSampleIn(BaseModel):
    sample_id: str
    x: float
    y: float
    z: float = 0.0
    sample_type: str = "rock"
    li: float = 0.0
    cs: float = 0.0
    rb: float = 0.0
    ta: float = 0.0
    nb: float = 0.0
    sn: float = 0.0
    be: float = 0.0
    b: float = 0.0
    f: float = 0.0
    p: float = 0.0
    k: float = 0.0
    na: float = 0.0
    al: float = 0.0
    si: float = 0.0
    fe: float = 0.0
    minerals_identified: List[str] = Field(default_factory=list)
    li2o_percent: float = 0.0


class PegmatiteScoreRequest(BaseModel):
    deposit_type: LithiumDepositType = LithiumDepositType.PEGMATITE_LCT
    sample_medium: str = "rock"
    samples: List[PegmatiteSampleIn] = Field(..., min_length=1)


@router.post("/lithium/score-pegmatite")
def lithium_score_pegmatite(req: PegmatiteScoreRequest) -> Dict[str, Any]:
    """Pegmatite pathfinder score + fractionation index + zonation."""
    results = score_pegmatite_samples(
        [s.model_dump() for s in req.samples], req.deposit_type, req.sample_medium)
    return {
        "deposit_type": req.deposit_type.value,
        "n_samples": len(results),
        "results": results,
    }


class BrineSampleIn(BaseModel):
    sample_id: str
    x: float
    y: float
    z: float = 0.0
    brine_type: str = "continental_salar"
    lithium: float = 0.0
    sodium: float = 0.0
    potassium: float = 0.0
    magnesium: float = 0.0
    calcium: float = 0.0
    chloride: float = 0.0
    sulfate: float = 0.0
    bicarbonate: float = 0.0
    boron: float = 0.0
    tds: float = 0.0


class BrineRequest(BaseModel):
    samples: List[BrineSampleIn] = Field(..., min_length=1)


@router.post("/lithium/brine")
def lithium_brine(req: BrineRequest) -> Dict[str, Any]:
    """Brine type classification, Mg/Li interpretation, evaporation index."""
    try:
        results = analyze_brine_samples([s.model_dump() for s in req.samples])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"n_samples": len(results), "results": results}


class DiscoveryWorkflowRequest(BaseModel):
    commodity: str = Field(..., description="'gold' or 'lithium'")
    deposit_type: Optional[str] = None
    samples: List[Dict[str, Any]] = Field(..., min_length=1)
    cell_size: float = Field(500.0, gt=0,
                             description="zone clustering cell size (map units)")


@router.post("/discovery/workflow")
def discovery_workflow(req: DiscoveryWorkflowRequest) -> Dict[str, Any]:
    """Orchestrated pipeline: score -> normalize -> cluster zones -> rank."""
    try:
        return run_discovery_workflow(
            commodity=req.commodity,
            samples=req.samples,
            deposit_type=req.deposit_type,
            cell_size=req.cell_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/deposit-types")
def deposit_types() -> Dict[str, Any]:
    """Enumerate supported gold + lithium deposit models and diagnostics."""
    return list_deposit_types()
