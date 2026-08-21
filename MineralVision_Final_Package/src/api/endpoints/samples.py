"""Sample-data endpoints with owner-or-admin project authorization."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth_middleware import TokenPayload, require_auth
from ..authz import is_admin, require_project_access
from ..database import DrillholeModel, ProjectModel, SampleModel, get_db

router = APIRouter()


class SampleCreate(BaseModel):
    sampleId: str = Field(..., min_length=1, max_length=100)
    drillholeId: str
    fromDepth: float = Field(ge=0)
    toDepth: float = Field(ge=0)
    sampleType: str = Field(default="core")
    assays: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class SampleUpdate(BaseModel):
    sampleId: Optional[str] = Field(None, min_length=1, max_length=100)
    fromDepth: Optional[float] = Field(None, ge=0)
    toDepth: Optional[float] = Field(None, ge=0)
    sampleType: Optional[str] = None
    assays: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class Sample(BaseModel):
    id: str
    sampleId: str
    drillholeId: str
    fromDepth: float
    toDepth: float
    sampleType: str
    assays: Dict[str, float] = Field(default_factory=dict)
    createdAt: str
    updatedAt: str


def _to_response(sample: SampleModel) -> Sample:
    return Sample(
        id=sample.id,
        sampleId=sample.sample_id,
        drillholeId=sample.drillhole_id,
        fromDepth=sample.from_depth,
        toDepth=sample.to_depth,
        sampleType=sample.sample_type,
        assays=sample.assay_data or {},
        createdAt=sample.created_at.isoformat(),
        updatedAt=sample.updated_at.isoformat(),
    )


def _require_drillhole_access(db: Session, drillhole_id: str, user: TokenPayload) -> DrillholeModel:
    drillhole = db.query(DrillholeModel).filter(DrillholeModel.id == drillhole_id).first()
    if drillhole is None:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    require_project_access(db, drillhole.project_id, user)
    return drillhole


def _require_sample_access(db: Session, sample_id: str, user: TokenPayload) -> SampleModel:
    sample = db.query(SampleModel).filter(SampleModel.id == sample_id).first()
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")
    _require_drillhole_access(db, sample.drillhole_id, user)
    return sample


@router.get("", response_model=List[Sample])
async def list_samples(
    drillholeId: Optional[str] = Query(None, description="Filter by drillhole ID"),
    sampleType: Optional[str] = Query(None, description="Filter by sample type"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    """List samples only from projects accessible to the caller."""
    query = db.query(SampleModel).join(DrillholeModel, SampleModel.drillhole_id == DrillholeModel.id).join(
        ProjectModel, DrillholeModel.project_id == ProjectModel.id
    )
    if not is_admin(user):
        query = query.filter(ProjectModel.owner_id == user.user_id)
    if drillholeId:
        query = query.filter(SampleModel.drillhole_id == drillholeId)
    if sampleType:
        query = query.filter(SampleModel.sample_type == sampleType)
    return [_to_response(sample) for sample in query.offset(offset).limit(limit).all()]


@router.get("/{sample_id}", response_model=Sample)
async def get_sample(
    sample_id: str,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    return _to_response(_require_sample_access(db, sample_id, user))


@router.post("", response_model=Sample, status_code=201)
async def create_sample(
    sample: SampleCreate,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    if sample.toDepth <= sample.fromDepth:
        raise HTTPException(status_code=400, detail="toDepth must be greater than fromDepth")
    drillhole = _require_drillhole_access(db, sample.drillholeId, user)
    db_sample = SampleModel(
        id=str(uuid.uuid4()),
        sample_id=sample.sampleId,
        drillhole_id=sample.drillholeId,
        from_depth=sample.fromDepth,
        to_depth=sample.toDepth,
        sample_type=sample.sampleType,
        lithology=(sample.metadata or {}).get("lithology"),
        assay_data=sample.assays or {},
    )
    db.add(db_sample)
    drillhole.assay_count = (drillhole.assay_count or 0) + len(sample.assays or {})
    db.commit()
    db.refresh(db_sample)
    return _to_response(db_sample)


@router.put("/{sample_id}", response_model=Sample)
async def update_sample(
    sample_id: str,
    sample: SampleUpdate,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    db_sample = _require_sample_access(db, sample_id, user)
    update_data = sample.model_dump(exclude_unset=True)
    if update_data.get("sampleId"):
        db_sample.sample_id = update_data["sampleId"]
    if update_data.get("fromDepth") is not None:
        db_sample.from_depth = update_data["fromDepth"]
    if update_data.get("toDepth") is not None:
        db_sample.to_depth = update_data["toDepth"]
    if update_data.get("sampleType"):
        db_sample.sample_type = update_data["sampleType"]
    if update_data.get("assays") is not None:
        db_sample.assay_data = update_data["assays"]
    if db_sample.to_depth <= db_sample.from_depth:
        raise HTTPException(status_code=400, detail="toDepth must be greater than fromDepth")
    db_sample.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_sample)
    return _to_response(db_sample)


@router.delete("/{sample_id}", status_code=204)
async def delete_sample(
    sample_id: str,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    db_sample = _require_sample_access(db, sample_id, user)
    db.delete(db_sample)
    db.commit()
    return None


@router.post("/{sample_id}/assays")
async def add_assays(
    sample_id: str,
    assays: Dict[str, float],
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    db_sample = _require_sample_access(db, sample_id, user)
    merged = dict(db_sample.assay_data or {})
    merged.update(assays)
    db_sample.assay_data = merged
    drillhole = db.query(DrillholeModel).filter(DrillholeModel.id == db_sample.drillhole_id).first()
    if drillhole:
        drillhole.assay_count = (drillhole.assay_count or 0) + len(assays)
    db_sample.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_sample)
    return _to_response(db_sample)


@router.get("/by-drillhole/{drillhole_id}", response_model=List[Sample])
async def get_samples_by_drillhole(
    drillhole_id: str,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    _require_drillhole_access(db, drillhole_id, user)
    samples = db.query(SampleModel).filter(SampleModel.drillhole_id == drillhole_id).all()
    return [_to_response(sample) for sample in samples]
