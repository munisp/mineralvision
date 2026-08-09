"""
API endpoints for Project management.

Database-backed CRUD operations for mining/exploration projects.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from sqlalchemy.orm import Session

from ..database import get_db, ProjectModel, DrillholeModel, SampleModel
from ..auth_middleware import TokenPayload, require_auth

router = APIRouter()


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


def _to_response(p: ProjectModel) -> Project:
    return Project(
        id=p.id,
        name=p.name,
        description=p.description,
        location=p.location,
        commodities=p.commodities or [],
        status=p.status,
        createdAt=p.created_at.isoformat(),
        updatedAt=p.updated_at.isoformat(),
        metadata=None
    )


@router.get("", response_model=List[Project])
async def list_projects(
    status: Optional[str] = Query(None, description="Filter by status"),
    commodity: Optional[str] = Query(None, description="Filter by commodity"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List all projects with optional filtering."""
    query = db.query(ProjectModel)
    if status:
        query = query.filter(ProjectModel.status == status)
    projects = query.offset(offset).limit(limit).all()
    if commodity:
        projects = [p for p in projects if commodity in (p.commodities or [])]
    return [_to_response(p) for p in projects]


@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str, db: Session = Depends(get_db)):
    """Get a specific project by ID."""
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return _to_response(project)


@router.post("", response_model=Project, status_code=201)
async def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth)
):
    """Create a new project."""
    db_project = ProjectModel(
        id=str(uuid.uuid4()),
        name=project.name,
        description=project.description,
        location=project.location,
        commodities=project.commodities,
        status=project.status,
        owner_id=user.user_id
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return _to_response(db_project)


@router.put("/{project_id}", response_model=Project)
async def update_project(
    project_id: str,
    project: ProjectUpdate,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth)
):
    """Update an existing project."""
    db_project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    update_data = project.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is None or key == "metadata":
            continue
        setattr(db_project, key, value)

    db_project.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_project)
    return _to_response(db_project)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth)
):
    """Delete a project."""
    db_project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    db.delete(db_project)
    db.commit()
    return None


@router.get("/{project_id}/statistics")
async def get_project_statistics(project_id: str, db: Session = Depends(get_db)):
    """Get statistics for a project."""
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    drillholes = db.query(DrillholeModel).filter(
        DrillholeModel.project_id == project_id
    ).all()
    drillhole_ids = [d.id for d in drillholes]
    sample_count = 0
    if drillhole_ids:
        sample_count = db.query(SampleModel).filter(
            SampleModel.drillhole_id.in_(drillhole_ids)
        ).count()

    return {
        "project_id": project_id,
        "drillhole_count": len(drillholes),
        "sample_count": sample_count,
        "total_meters_drilled": sum(d.total_depth for d in drillholes),
        "assay_count": sum(d.assay_count or 0 for d in drillholes),
        "last_updated": datetime.utcnow().isoformat()
    }
