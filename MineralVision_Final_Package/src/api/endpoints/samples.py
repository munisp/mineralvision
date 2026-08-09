"""
API endpoints for Sample management.

This module provides CRUD operations for geological samples.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

router = APIRouter()

# Runtime storage for API operations
samples_db: Dict[str, dict] = {}


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


@router.get("", response_model=List[Sample])
async def list_samples(
    drillholeId: Optional[str] = Query(None, description="Filter by drillhole ID"),
    sampleType: Optional[str] = Query(None, description="Filter by sample type"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all samples with optional filtering."""
    samples = list(samples_db.values())
    
    # Apply filters
    if drillholeId:
        samples = [s for s in samples if s.get("drillholeId") == drillholeId]
    if sampleType:
        samples = [s for s in samples if s.get("sampleType") == sampleType]
    
    # Apply pagination
    samples = samples[offset:offset + limit]
    
    return [Sample(**s) for s in samples]


@router.get("/{sample_id}", response_model=Sample)
async def get_sample(sample_id: str):
    """Get a specific sample by ID."""
    if sample_id not in samples_db:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")
    return Sample(**samples_db[sample_id])


@router.post("", response_model=Sample, status_code=201)
async def create_sample(sample: SampleCreate):
    """Create a new sample."""
    # Validate depth range
    if sample.toDepth <= sample.fromDepth:
        raise HTTPException(status_code=400, detail="toDepth must be greater than fromDepth")
    
    sample_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    sample_data = {
        "id": sample_id,
        "sampleId": sample.sampleId,
        "drillholeId": sample.drillholeId,
        "fromDepth": sample.fromDepth,
        "toDepth": sample.toDepth,
        "sampleType": sample.sampleType,
        "assays": sample.assays or {},
        "createdAt": now,
        "updatedAt": now
    }
    
    samples_db[sample_id] = sample_data
    return Sample(**sample_data)


@router.put("/{sample_id}", response_model=Sample)
async def update_sample(sample_id: str, sample: SampleUpdate):
    """Update an existing sample."""
    if sample_id not in samples_db:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")
    
    existing = samples_db[sample_id]
    update_data = sample.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if value is not None:
            existing[key] = value
    
    # Validate depth range if updated
    if existing["toDepth"] <= existing["fromDepth"]:
        raise HTTPException(status_code=400, detail="toDepth must be greater than fromDepth")
    
    existing["updatedAt"] = datetime.utcnow().isoformat()
    samples_db[sample_id] = existing
    
    return Sample(**existing)


@router.delete("/{sample_id}", status_code=204)
async def delete_sample(sample_id: str):
    """Delete a sample."""
    if sample_id not in samples_db:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")
    
    del samples_db[sample_id]
    return None


@router.post("/{sample_id}/assays")
async def add_assays(sample_id: str, assays: Dict[str, float]):
    """Add or update assay results for a sample."""
    if sample_id not in samples_db:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")
    
    existing = samples_db[sample_id]
    existing_assays = existing.get("assays", {})
    existing_assays.update(assays)
    existing["assays"] = existing_assays
    existing["updatedAt"] = datetime.utcnow().isoformat()
    
    samples_db[sample_id] = existing
    return Sample(**existing)


@router.get("/by-drillhole/{drillhole_id}", response_model=List[Sample])
async def get_samples_by_drillhole(drillhole_id: str):
    """Get all samples for a specific drillhole."""
    samples = [s for s in samples_db.values() if s.get("drillholeId") == drillhole_id]
    return [Sample(**s) for s in samples]
