"""
Simplified API endpoints for Drillhole management (no heavy dependencies).
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

router = APIRouter()

# Runtime storage for API operations
drillholes_db: Dict[str, dict] = {}


class CollarCreate(BaseModel):
    x: float
    y: float
    z: float


class DrillholeCreate(BaseModel):
    holeId: str = Field(..., min_length=1, max_length=100)
    projectId: str
    collar: CollarCreate
    totalDepth: float = Field(ge=0)
    azimuth: Optional[float] = Field(None, ge=0, le=360)
    dip: Optional[float] = Field(None, ge=-90, le=90)
    status: str = Field(default="planned")
    metadata: Optional[Dict[str, Any]] = None


class DrillholeUpdate(BaseModel):
    holeId: Optional[str] = None
    collar: Optional[CollarCreate] = None
    totalDepth: Optional[float] = None
    azimuth: Optional[float] = None
    dip: Optional[float] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Drillhole(BaseModel):
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


@router.get("", response_model=List[Drillhole])
async def list_drillholes(
    projectId: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    drillholes = list(drillholes_db.values())
    if projectId:
        drillholes = [d for d in drillholes if d.get("projectId") == projectId]
    if status:
        drillholes = [d for d in drillholes if d.get("status") == status]
    drillholes = drillholes[offset:offset + limit]
    return [Drillhole(**d) for d in drillholes]


@router.get("/{drillhole_id}", response_model=Drillhole)
async def get_drillhole(drillhole_id: str):
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    return Drillhole(**drillholes_db[drillhole_id])


@router.post("", response_model=Drillhole, status_code=201)
async def create_drillhole(drillhole: DrillholeCreate):
    drillhole_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    drillhole_data = {
        "id": drillhole_id,
        "holeId": drillhole.holeId,
        "projectId": drillhole.projectId,
        "collar": {"x": drillhole.collar.x, "y": drillhole.collar.y, "z": drillhole.collar.z},
        "totalDepth": drillhole.totalDepth,
        "azimuth": drillhole.azimuth,
        "dip": drillhole.dip,
        "status": drillhole.status,
        "assayCount": 0,
        "createdAt": now,
        "updatedAt": now
    }
    
    drillholes_db[drillhole_id] = drillhole_data
    return Drillhole(**drillhole_data)


@router.put("/{drillhole_id}", response_model=Drillhole)
async def update_drillhole(drillhole_id: str, drillhole: DrillholeUpdate):
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    
    existing = drillholes_db[drillhole_id]
    update_data = drillhole.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if value is not None:
            if key == "collar":
                existing["collar"] = {"x": value.x, "y": value.y, "z": value.z}
            else:
                existing[key] = value
    
    existing["updatedAt"] = datetime.utcnow().isoformat()
    drillholes_db[drillhole_id] = existing
    return Drillhole(**existing)


@router.delete("/{drillhole_id}", status_code=204)
async def delete_drillhole(drillhole_id: str):
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    del drillholes_db[drillhole_id]
    return None


@router.get("/{drillhole_id}/assays")
async def get_drillhole_assays(drillhole_id: str):
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    return {"drillhole_id": drillhole_id, "assays": []}


@router.get("/{drillhole_id}/lithology")
async def get_drillhole_lithology(drillhole_id: str):
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    return {"drillhole_id": drillhole_id, "lithology": []}
