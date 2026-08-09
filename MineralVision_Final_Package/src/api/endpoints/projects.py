"""
API endpoints for Project management.

This module provides CRUD operations for mining/exploration projects.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

router = APIRouter()

# Runtime storage for API operations
projects_db: Dict[str, dict] = {}


class ProjectCreate(BaseModel):
    """Schema for creating a project."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    location: Optional[str] = None
    commodities: List[str] = Field(default_factory=list)
    status: str = Field(default="active")
    metadata: Optional[Dict[str, Any]] = None


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    name: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    commodities: Optional[List[str]] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Project(BaseModel):
    """Schema for project response."""
    id: str
    name: str
    description: Optional[str] = None
    location: Optional[str] = None
    commodities: List[str] = Field(default_factory=list)
    status: str
    createdAt: str
    updatedAt: str
    metadata: Optional[Dict[str, Any]] = None


@router.get("", response_model=List[Project])
async def list_projects(
    status: Optional[str] = Query(None, description="Filter by status"),
    commodity: Optional[str] = Query(None, description="Filter by commodity"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all projects with optional filtering."""
    projects = list(projects_db.values())
    
    # Apply filters
    if status:
        projects = [p for p in projects if p.get("status") == status]
    if commodity:
        projects = [p for p in projects if commodity in p.get("commodities", [])]
    
    # Apply pagination
    projects = projects[offset:offset + limit]
    
    return [Project(**p) for p in projects]


@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Get a specific project by ID."""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return Project(**projects_db[project_id])


@router.post("", response_model=Project, status_code=201)
async def create_project(project: ProjectCreate):
    """Create a new project."""
    project_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    project_data = {
        "id": project_id,
        "name": project.name,
        "description": project.description,
        "location": project.location,
        "commodities": project.commodities,
        "status": project.status,
        "createdAt": now,
        "updatedAt": now,
        "metadata": project.metadata or {}
    }
    
    projects_db[project_id] = project_data
    return Project(**project_data)


@router.put("/{project_id}", response_model=Project)
async def update_project(project_id: str, project: ProjectUpdate):
    """Update an existing project."""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    existing = projects_db[project_id]
    update_data = project.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if value is not None:
            existing[key] = value
    
    existing["updatedAt"] = datetime.utcnow().isoformat()
    projects_db[project_id] = existing
    
    return Project(**existing)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str):
    """Delete a project."""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    del projects_db[project_id]
    return None


@router.get("/{project_id}/statistics")
async def get_project_statistics(project_id: str):
    """Get statistics for a project."""
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    # Return default result on error
    return {
        "project_id": project_id,
        "drillhole_count": 0,
        "sample_count": 0,
        "total_meters_drilled": 0.0,
        "assay_count": 0,
        "last_updated": datetime.utcnow().isoformat()
    }
