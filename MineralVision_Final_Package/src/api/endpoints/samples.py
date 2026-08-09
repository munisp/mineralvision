"""
API endpoints for Sample management.

Database-backed CRUD operations for geological samples.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from sqlalchemy.orm import Session

from ..database import get_db, SampleModel, DrillholeModel

router = APIRouter()


class SampleCreate(BaseModel):
    """Schema for creating a sample."""
    sampleId: str = Field(..., min_length=1, max_length=100)
    drillholeId: str
    fromDepth: float = Field(ge=0)
    toDepth: float = Field(ge=0)
    sampleType: str = Field(default="core")
    assays: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class SampleUpdate(BaseModel):
    """Schema for updating a sample."""
    sampleId: Optional[str] = None
    fromDepth: Optional[float] = None
    toDepth: Optional[float] = None
    sampleType: Optional[str] = None
    assays: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class Sample(BaseModel):
    """Schema for sample response."""
    id: str
    sampleId: str
    drillholeId: str
    fromDepth: float
    toDepth: float
    sampleType: str
    assays: Dict[str, float] = Field(default_factory=dict)
    createdAt: str
    updatedAt: str


def _to_response(s: SampleModel) -> Sample:
    return Sample(
        id=s.id,
        sampleId=s.sample_id,
        drillholeId=s.drillhole_id,
        fromDepth=s.from_depth,
        toDepth=s.to_depth,
        sampleType=s.sample_type,
        assays=s.assay_data or {},
        createdAt=s.created_at.isoformat(),
        updatedAt=s.created_at.isoformat()
    )


@router.get("", response_model=List[Sample])
async def list_samples(
    drillholeId: Optional[str] = Query(None, description="Filter by drillhole ID"),
    sampleType: Optional[str] = Query(None, description="Filter by sample type"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List all samples with optional filtering."""
    query = db.query(SampleModel)
    if drillholeId:
        query = query.filter(SampleModel.drillhole_id == drillholeId)
    if sampleType:
        query = query.filter(SampleModel.sample_type == sampleType)
    samples = query.offset(offset).limit(limit).all()
    return [_to_response(s) for s in samples]


@router.get("/{sample_id}", response_model=Sample)
async def get_sample(sample_id: str, db: Session = Depends(get_db)):
    """Get a specific sample by ID."""
    sample = db.query(SampleModel).filter(SampleModel.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")
    return _to_response(sample)


@router.post("", response_model=Sample, status_code=201)
async def create_sample(sample: SampleCreate, db: Session = Depends(get_db)):
    """Create a new sample."""
    # Validate depth range
    if sample.toDepth <= sample.fromDepth:
        raise HTTPException(status_code=400, detail="toDepth must be greater than fromDepth")

    # Verify drillhole exists
    drillhole = db.query(DrillholeModel).filter(
        DrillholeModel.id == sample.drillholeId
    ).first()
    if not drillhole:
        raise HTTPException(status_code=404, detail=f"Drillhole {sample.drillholeId} not found")

    db_sample = SampleModel(
        id=str(uuid.uuid4()),
        sample_id=sample.sampleId,
        drillhole_id=sample.drillholeId,
        from_depth=sample.fromDepth,
        to_depth=sample.toDepth,
        sample_type=sample.sampleType,
        lithology=(sample.metadata or {}).get("lithology"),
        assay_data=sample.assays or {}
    )
    db.add(db_sample)

    # Update drillhole assay count
    drillhole.assay_count = (drillhole.assay_count or 0) + len(sample.assays or {})

    db.commit()
    db.refresh(db_sample)
    return _to_response(db_sample)


@router.put("/{sample_id}", response_model=Sample)
async def update_sample(sample_id: str, sample: SampleUpdate, db: Session = Depends(get_db)):
    """Update an existing sample."""
    db_sample = db.query(SampleModel).filter(SampleModel.id == sample_id).first()
    if not db_sample:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")

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

    db.commit()
    db.refresh(db_sample)
    return _to_response(db_sample)


@router.delete("/{sample_id}", status_code=204)
async def delete_sample(sample_id: str, db: Session = Depends(get_db)):
    """Delete a sample."""
    db_sample = db.query(SampleModel).filter(SampleModel.id == sample_id).first()
    if not db_sample:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")

    db.delete(db_sample)
    db.commit()
    return None


@router.post("/{sample_id}/assays")
async def add_assays(
    sample_id: str,
    assays: Dict[str, float],
    db: Session = Depends(get_db)
):
    """Add or update assay results for a sample."""
    db_sample = db.query(SampleModel).filter(SampleModel.id == sample_id).first()
    if not db_sample:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")

    merged = dict(db_sample.assay_data or {})
    merged.update(assays)
    db_sample.assay_data = merged

    drillhole = db.query(DrillholeModel).filter(
        DrillholeModel.id == db_sample.drillhole_id
    ).first()
    if drillhole:
        drillhole.assay_count = (drillhole.assay_count or 0) + len(assays)

    db.commit()
    db.refresh(db_sample)
    return _to_response(db_sample)


@router.get("/by-drillhole/{drillhole_id}", response_model=List[Sample])
async def get_samples_by_drillhole(drillhole_id: str, db: Session = Depends(get_db)):
    """Get all samples for a drillhole."""
    samples = db.query(SampleModel).filter(
        SampleModel.drillhole_id == drillhole_id
    ).all()
    return [_to_response(s) for s in samples]
