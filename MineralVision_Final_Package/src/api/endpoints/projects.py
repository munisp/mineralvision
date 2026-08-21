"""Project CRUD endpoints with explicit owner-or-admin tenant isolation."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth_middleware import TokenPayload, require_auth
from ..authz import project_scope_query, require_project_access
from ..database import DrillholeModel, ProjectModel, SampleModel, get_db

router = APIRouter()


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    location: Optional[str] = None
    commodities: List[str] = Field(default_factory=list)
    status: str = Field(default="active")
    metadata: Optional[Dict[str, Any]] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    location: Optional[str] = None
    commodities: Optional[List[str]] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Project(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    location: Optional[str] = None
    commodities: List[str] = Field(default_factory=list)
    status: str
    createdAt: str
    updatedAt: str
    metadata: Optional[Dict[str, Any]] = None


def _to_response(project: ProjectModel) -> Project:
    return Project(
        id=project.id,
        name=project.name,
        description=project.description,
        location=project.location,
        commodities=project.commodities or [],
        status=project.status,
        createdAt=project.created_at.isoformat(),
        updatedAt=project.updated_at.isoformat(),
        metadata=None,
    )


@router.get("", response_model=List[Project])
async def list_projects(
    status: Optional[str] = Query(None, description="Filter by status"),
    commodity: Optional[str] = Query(None, description="Filter by commodity"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    """List only projects owned by the caller unless the caller is an admin."""
    query = project_scope_query(db.query(ProjectModel), user)
    if status:
        query = query.filter(ProjectModel.status == status)
    projects = query.offset(offset).limit(limit).all()
    if commodity:
        projects = [project for project in projects if commodity in (project.commodities or [])]
    return [_to_response(project) for project in projects]


@router.get("/{project_id}", response_model=Project)
async def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    return _to_response(require_project_access(db, project_id, user))


@router.post("", response_model=Project, status_code=201)
async def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    """Create a project owned by the authenticated caller, never by caller input."""
    db_project = ProjectModel(
        id=str(uuid.uuid4()),
        name=project.name,
        description=project.description,
        location=project.location,
        commodities=project.commodities,
        status=project.status,
        owner_id=user.user_id,
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
    user: TokenPayload = Depends(require_auth),
):
    db_project = require_project_access(db, project_id, user)
    for key, value in project.model_dump(exclude_unset=True).items():
        if value is not None and key != "metadata":
            setattr(db_project, key, value)
    db_project.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_project)
    return _to_response(db_project)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    db_project = require_project_access(db, project_id, user)
    db.delete(db_project)
    db.commit()
    return None


@router.get("/{project_id}/statistics")
async def get_project_statistics(
    project_id: str,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    require_project_access(db, project_id, user)
    drillholes = db.query(DrillholeModel).filter(DrillholeModel.project_id == project_id).all()
    drillhole_ids = [drillhole.id for drillhole in drillholes]
    sample_count = 0
    if drillhole_ids:
        sample_count = db.query(SampleModel).filter(SampleModel.drillhole_id.in_(drillhole_ids)).count()
    return {
        "project_id": project_id,
        "drillhole_count": len(drillholes),
        "sample_count": sample_count,
        "total_meters_drilled": sum(drillhole.total_depth for drillhole in drillholes),
        "assay_count": sum(drillhole.assay_count or 0 for drillhole in drillholes),
        "last_updated": datetime.utcnow().isoformat(),
    }
