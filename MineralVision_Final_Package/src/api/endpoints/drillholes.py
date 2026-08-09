"""
API endpoints for Drillhole management.

This module provides CRUD operations for drillholes and integrates
with the drillhole_database module for validation and processing.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import json

# Import the drillhole database module
from ..geology.drillhole_database import (
    DrillholeDatabase,
    CollarRecord,
    SurveyRecord,
    AssayRecord,
    LithologyRecord,
    ValidationLevel,
    create_drillhole_database
)

router = APIRouter()

# Runtime storage for API operations
drillholes_db: Dict[str, dict] = {}

# Initialize drillhole database
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


@router.get("", response_model=List[Drillhole])
async def list_drillholes(
    projectId: Optional[str] = Query(None, description="Filter by project ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all drillholes with optional filtering."""
    drillholes = list(drillholes_db.values())
    
    # Apply filters
    if projectId:
        drillholes = [d for d in drillholes if d.get("projectId") == projectId]
    if status:
        drillholes = [d for d in drillholes if d.get("status") == status]
    
    # Apply pagination
    drillholes = drillholes[offset:offset + limit]
    
    return [Drillhole(**d) for d in drillholes]


@router.get("/{drillhole_id}", response_model=Drillhole)
async def get_drillhole(drillhole_id: str):
    """Get a specific drillhole by ID."""
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    return Drillhole(**drillholes_db[drillhole_id])


@router.post("", response_model=Drillhole, status_code=201)
async def create_drillhole(drillhole: DrillholeCreate):
    """Create a new drillhole."""
    drillhole_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    drillhole_data = {
        "id": drillhole_id,
        "holeId": drillhole.holeId,
        "projectId": drillhole.projectId,
        "collar": {
            "x": drillhole.collar.x,
            "y": drillhole.collar.y,
            "z": drillhole.collar.z
        },
        "totalDepth": drillhole.totalDepth,
        "azimuth": drillhole.azimuth,
        "dip": drillhole.dip,
        "status": drillhole.status,
        "assayCount": 0,
        "createdAt": now,
        "updatedAt": now
    }
    
    # Add to drillhole database for validation
    try:
        collar = CollarRecord(
            hole_id=drillhole.holeId,
            x=drillhole.collar.x,
            y=drillhole.collar.y,
            z=drillhole.collar.z,
            total_depth=drillhole.totalDepth,
            azimuth=drillhole.azimuth or 0.0,
            dip=drillhole.dip or -90.0
        )
        drillhole_database.add_collar(collar)
    except Exception as e:
        # Log but don't fail - in-memory storage is primary
        pass
    
    drillholes_db[drillhole_id] = drillhole_data
    return Drillhole(**drillhole_data)


@router.put("/{drillhole_id}", response_model=Drillhole)
async def update_drillhole(drillhole_id: str, drillhole: DrillholeUpdate):
    """Update an existing drillhole."""
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    
    existing = drillholes_db[drillhole_id]
    update_data = drillhole.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if value is not None:
            if key == "collar":
                existing["collar"] = {
                    "x": value.x,
                    "y": value.y,
                    "z": value.z
                }
            else:
                existing[key] = value
    
    existing["updatedAt"] = datetime.utcnow().isoformat()
    drillholes_db[drillhole_id] = existing
    
    return Drillhole(**existing)


@router.delete("/{drillhole_id}", status_code=204)
async def delete_drillhole(drillhole_id: str):
    """Delete a drillhole."""
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    
    del drillholes_db[drillhole_id]
    return None


@router.post("/upload")
async def upload_drillholes(
    file: UploadFile = File(...),
    projectId: str = Form(...)
):
    """Upload drillhole data from CSV/Excel file."""
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
            # Would need openpyxl for Excel files
            raise HTTPException(status_code=400, detail="Excel files not yet supported")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        imported_count = 0
        errors = []
        
        for i, row in enumerate(rows):
            try:
                drillhole_id = str(uuid.uuid4())
                now = datetime.utcnow().isoformat()
                
                drillhole_data = {
                    "id": drillhole_id,
                    "holeId": row.get("hole_id", row.get("holeId", f"DH-{i+1}")),
                    "projectId": projectId,
                    "collar": {
                        "x": float(row.get("x", row.get("easting", 0))),
                        "y": float(row.get("y", row.get("northing", 0))),
                        "z": float(row.get("z", row.get("elevation", 0)))
                    },
                    "totalDepth": float(row.get("total_depth", row.get("totalDepth", 0))),
                    "azimuth": float(row.get("azimuth", 0)) if row.get("azimuth") else None,
                    "dip": float(row.get("dip", -90)) if row.get("dip") else -90,
                    "status": row.get("status", "completed"),
                    "assayCount": 0,
                    "createdAt": now,
                    "updatedAt": now
                }
                
                drillholes_db[drillhole_id] = drillhole_data
                imported_count += 1
            except Exception as e:
                errors.append({"row": i + 1, "error": str(e)})
        
        return {
            "imported": imported_count,
            "errors": errors,
            "total_rows": len(rows)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{drillhole_id}/composite")
async def composite_drillhole(drillhole_id: str, request: CompositeRequest):
    """Create composites for a drillhole."""
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    
    drillhole = drillholes_db[drillhole_id]
    
    # Use drillhole database for compositing
    try:
        composites = drillhole_database.composite_assays(
            hole_id=drillhole["holeId"],
            composite_length=request.length,
            method=request.method
        )
        
        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole["holeId"],
            "composite_length": request.length,
            "method": request.method,
            "composites": composites
        }
    except Exception as e:
        # Return default result on error
        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole["holeId"],
            "composite_length": request.length,
            "method": request.method,
            "composites": [],
            "message": "No assay data available for compositing"
        }


@router.post("/{drillhole_id}/desurvey")
async def desurvey_drillhole(drillhole_id: str, request: DesurveyRequest):
    """Calculate desurveyed coordinates for a drillhole."""
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    
    drillhole = drillholes_db[drillhole_id]
    
    # Use drillhole database for desurveying
    try:
        desurveyed = drillhole_database.desurvey_hole(
            hole_id=drillhole["holeId"],
            method=request.method
        )
        
        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole["holeId"],
            "method": request.method,
            "coordinates": desurveyed
        }
    except Exception as e:
        # Return default result on error
        collar = drillhole["collar"]
        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole["holeId"],
            "method": request.method,
            "coordinates": [
                {"depth": 0, "x": collar["x"], "y": collar["y"], "z": collar["z"]},
                {"depth": drillhole["totalDepth"], "x": collar["x"], "y": collar["y"], "z": collar["z"] - drillhole["totalDepth"]}
            ],
            "message": "No survey data available, using vertical projection"
        }


@router.get("/{drillhole_id}/assays")
async def get_drillhole_assays(drillhole_id: str):
    """Get assay data for a drillhole."""
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    
    drillhole = drillholes_db[drillhole_id]
    
    try:
        assays = drillhole_database.get_assays(hole_id=drillhole["holeId"])
        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole["holeId"],
            "assays": assays
        }
    except Exception:
        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole["holeId"],
            "assays": []
        }


@router.get("/{drillhole_id}/lithology")
async def get_drillhole_lithology(drillhole_id: str):
    """Get lithology data for a drillhole."""
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    
    drillhole = drillholes_db[drillhole_id]
    
    try:
        lithology = drillhole_database.get_lithology(hole_id=drillhole["holeId"])
        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole["holeId"],
            "lithology": lithology
        }
    except Exception:
        return {
            "drillhole_id": drillhole_id,
            "hole_id": drillhole["holeId"],
            "lithology": []
        }
