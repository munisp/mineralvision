"""
API endpoints for Drillhole management.

Database-backed CRUD operations for drillholes, integrated with the
drillhole_database module for validation, desurveying and compositing.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import json

from sqlalchemy.orm import Session

from ..database import get_db, DrillholeModel, ProjectModel, SampleModel

# Import the drillhole database module
from ..geology.drillhole_database import (
    DrillholeDatabase,
    CollarData,
    SurveyData,
    AssayData,
    LithologyData,
    ValidationSeverity,
    create_drillhole_database
)

router = APIRouter()

# Analytical drillhole database (validation, desurvey, compositing)
drillhole_database = create_drillhole_database()


class CollarCreate(BaseModel):
    """Schema for collar coordinates."""
    x: float
    y: float
    z: float


class DrillholeCreate(BaseModel):
    """Schema for creating a drillhole."""
    holeId: str = Field(..., min_length=1, max_length=100)
    projectId: str
    collar: CollarCreate
    totalDepth: float = Field(ge=0)
    azimuth: Optional[float] = Field(None, ge=0, le=360)
    dip: Optional[float] = Field(None, ge=-90, le=90)
    status: str = Field(default="planned")
    metadata: Optional[Dict[str, Any]] = None


class DrillholeUpdate(BaseModel):
    """Schema for updating a drillhole."""
    holeId: Optional[str] = None
    collar: Optional[CollarCreate] = None
    totalDepth: Optional[float] = None
    azimuth: Optional[float] = None
    dip: Optional[float] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Drillhole(BaseModel):
    """Schema for drillhole response."""
    id: str
    holeId: str
    projectId: str
    collar: Dict[str, float]
    totalDepth: float
    azimuth: Optional[float] = None
    dip: Optional[float] = None
    status: str
    assayCount: int = 0
    createdAt: str
    updatedAt: str


class CompositeRequest(BaseModel):
    """Schema for composite request."""
    length: float = Field(..., gt=0)
    method: str = Field(default="length_weighted")


class DesurveyRequest(BaseModel):
    """Schema for desurvey request."""
    method: str = Field(default="minimum_curvature")


def _to_response(d: DrillholeModel) -> Drillhole:
    return Drillhole(
        id=d.id,
        holeId=d.hole_id,
        projectId=d.project_id,
        collar={"x": d.collar_x, "y": d.collar_y, "z": d.collar_z},
        totalDepth=d.total_depth,
        azimuth=d.azimuth,
        dip=d.dip,
        status=d.status,
        assayCount=d.assay_count or 0,
        createdAt=d.created_at.isoformat(),
        updatedAt=d.updated_at.isoformat()
    )


def _register_collar(d: DrillholeModel):
    """Register a drillhole collar with the analytical database."""
    collar = CollarData(
        hole_id=d.hole_id,
        easting=d.collar_x,
        northing=d.collar_y,
        elevation=d.collar_z,
        total_depth=d.total_depth,
        azimuth=d.azimuth or 0.0,
        dip=d.dip if d.dip is not None else -90.0
    )
    drillhole_database.add_collar(collar)


def _get_or_404(drillhole_id: str, db: Session) -> DrillholeModel:
    drillhole = db.query(DrillholeModel).filter(DrillholeModel.id == drillhole_id).first()
    if not drillhole:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    return drillhole


@router.get("", response_model=List[Drillhole])
async def list_drillholes(
    projectId: Optional[str] = Query(None, description="Filter by project ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List all drillholes with optional filtering."""
    query = db.query(DrillholeModel)
    if projectId:
        query = query.filter(DrillholeModel.project_id == projectId)
    if status:
        query = query.filter(DrillholeModel.status == status)
    drillholes = query.offset(offset).limit(limit).all()
    return [_to_response(d) for d in drillholes]


@router.get("/{drillhole_id}", response_model=Drillhole)
async def get_drillhole(drillhole_id: str, db: Session = Depends(get_db)):
    """Get a specific drillhole by ID."""
    return _to_response(_get_or_404(drillhole_id, db))


@router.post("", response_model=Drillhole, status_code=201)
async def create_drillhole(drillhole: DrillholeCreate, db: Session = Depends(get_db)):
    """Create a new drillhole."""
    # Verify project exists
    project = db.query(ProjectModel).filter(ProjectModel.id == drillhole.projectId).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {drillhole.projectId} not found")

    db_drillhole = DrillholeModel(
        id=str(uuid.uuid4()),
        hole_id=drillhole.holeId,
        project_id=drillhole.projectId,
        collar_x=drillhole.collar.x,
        collar_y=drillhole.collar.y,
        collar_z=drillhole.collar.z,
        total_depth=drillhole.totalDepth,
        azimuth=drillhole.azimuth,
        dip=drillhole.dip,
        status=drillhole.status
    )
    db.add(db_drillhole)
    db.commit()
    db.refresh(db_drillhole)

    # Register with the analytical database for validation/desurvey/compositing
    try:
        _register_collar(db_drillhole)
    except Exception:
        pass  # Analytical registration is best-effort; DB is authoritative

    return _to_response(db_drillhole)


@router.put("/{drillhole_id}", response_model=Drillhole)
async def update_drillhole(
    drillhole_id: str,
    drillhole: DrillholeUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing drillhole."""
    db_drillhole = _get_or_404(drillhole_id, db)

    update_data = drillhole.model_dump(exclude_unset=True)
    if update_data.get("holeId"):
        db_drillhole.hole_id = update_data["holeId"]
    if update_data.get("collar"):
        db_drillhole.collar_x = update_data["collar"].x
        db_drillhole.collar_y = update_data["collar"].y
        db_drillhole.collar_z = update_data["collar"].z
    if update_data.get("totalDepth") is not None:
        db_drillhole.total_depth = update_data["totalDepth"]
    if update_data.get("azimuth") is not None:
        db_drillhole.azimuth = update_data["azimuth"]
    if update_data.get("dip") is not None:
        db_drillhole.dip = update_data["dip"]
    if update_data.get("status"):
        db_drillhole.status = update_data["status"]

    db_drillhole.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_drillhole)
    return _to_response(db_drillhole)


@router.delete("/{drillhole_id}", status_code=204)
async def delete_drillhole(drillhole_id: str, db: Session = Depends(get_db)):
    """Delete a drillhole."""
    db_drillhole = _get_or_404(drillhole_id, db)
    db.delete(db_drillhole)
    db.commit()
    return None


@router.post("/upload")
async def upload_drillholes(
    file: UploadFile = File(...),
    projectId: str = Form(...),
    db: Session = Depends(get_db)
):
    """Upload drillhole data from CSV/Excel file."""
    # Verify project exists
    project = db.query(ProjectModel).filter(ProjectModel.id == projectId).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {projectId} not found")

    try:
        content = await file.read()
        filename = file.filename or "upload.csv"

        # Parse file based on extension
        if filename.endswith('.csv'):
            import csv
            import io
            reader = csv.DictReader(io.StringIO(content.decode('utf-8')))
            rows = list(reader)
        elif filename.endswith(('.xls', '.xlsx')):
            raise HTTPException(status_code=400, detail="Excel files not yet supported")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        imported_count = 0
        errors = []

        for i, row in enumerate(rows):
            try:
                db_drillhole = DrillholeModel(
                    id=str(uuid.uuid4()),
                    hole_id=row.get("hole_id", row.get("holeId", f"DH-{i+1}")),
                    project_id=projectId,
                    collar_x=float(row.get("x", row.get("easting", 0))),
                    collar_y=float(row.get("y", row.get("northing", 0))),
                    collar_z=float(row.get("z", row.get("elevation", 0))),
                    total_depth=float(row.get("total_depth", row.get("totalDepth", 0))),
                    azimuth=float(row.get("azimuth", 0)) if row.get("azimuth") else None,
                    dip=float(row.get("dip", -90)) if row.get("dip") else -90,
                    status=row.get("status", "completed")
                )
                db.add(db_drillhole)
                db.flush()

                try:
                    _register_collar(db_drillhole)
                except Exception:
                    pass

                imported_count += 1
            except Exception as e:
                errors.append({"row": i + 1, "error": str(e)})

        db.commit()

        return {
            "imported": imported_count,
            "errors": errors,
            "total_rows": len(rows)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{drillhole_id}/composite")
async def composite_drillhole(
    drillhole_id: str,
    request: CompositeRequest,
    db: Session = Depends(get_db)
):
    """Create composites for a drillhole."""
    drillhole = _get_or_404(drillhole_id, db)

    try:
        composites = drillhole_database.composite_hole(
            hole_id=drillhole.hole_id,
            composite_length=request.length
        )

        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole.hole_id,
            "composite_length": request.length,
            "method": request.method,
            "composites": composites
        }
    except Exception:
        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole.hole_id,
            "composite_length": request.length,
            "method": request.method,
            "composites": [],
            "message": "No assay data available for compositing"
        }


@router.post("/{drillhole_id}/desurvey")
async def desurvey_drillhole(
    drillhole_id: str,
    request: DesurveyRequest,
    db: Session = Depends(get_db)
):
    """Calculate desurveyed coordinates for a drillhole."""
    drillhole = _get_or_404(drillhole_id, db)

    try:
        desurveyed = drillhole_database.desurvey_hole(hole_id=drillhole.hole_id)

        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole.hole_id,
            "method": request.method,
            "coordinates": desurveyed
        }
    except Exception:
        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole.hole_id,
            "method": request.method,
            "coordinates": [
                {"depth": 0, "x": drillhole.collar_x, "y": drillhole.collar_y, "z": drillhole.collar_z},
                {"depth": drillhole.total_depth, "x": drillhole.collar_x,
                 "y": drillhole.collar_y, "z": drillhole.collar_z - drillhole.total_depth}
            ],
            "message": "No survey data available, using vertical projection"
        }


@router.get("/{drillhole_id}/assays")
async def get_drillhole_assays(drillhole_id: str, db: Session = Depends(get_db)):
    """Get assay data for a drillhole."""
    drillhole = _get_or_404(drillhole_id, db)

    samples = db.query(SampleModel).filter(
        SampleModel.drillhole_id == drillhole.id
    ).all()

    return {
        "drillhole_id": drillhole_id,
        "hole_id": drillhole.hole_id,
        "assays": [
            {
                "sampleId": s.sample_id,
                "fromDepth": s.from_depth,
                "toDepth": s.to_depth,
                "sampleType": s.sample_type,
                "lithology": s.lithology,
                "assays": s.assay_data or {}
            }
            for s in samples
        ]
    }


@router.get("/{drillhole_id}/lithology")
async def get_drillhole_lithology(drillhole_id: str, db: Session = Depends(get_db)):
    """Get lithology data for a drillhole."""
    drillhole = _get_or_404(drillhole_id, db)

    samples = db.query(SampleModel).filter(
        SampleModel.drillhole_id == drillhole.id,
        SampleModel.lithology.isnot(None)
    ).all()

    return {
        "drillhole_id": drillhole_id,
        "hole_id": drillhole.hole_id,
        "lithology": [
            {
                "sampleId": s.sample_id,
                "fromDepth": s.from_depth,
                "toDepth": s.to_depth,
                "lithology": s.lithology
            }
            for s in samples
        ]
    }
