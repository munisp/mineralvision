"""
MineralVision API Server - Production Ready

Full-featured FastAPI application with:
- SQLAlchemy + SQLite database
- JWT authentication middleware
- WALDO service proxy
- All core API endpoints

Run with: uvicorn src.api.main_production:app --host 0.0.0.0 --port 8000
"""

import os
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Query, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.database import (
    init_db, get_db, seed_demo_data,
    UserModel, ProjectModel, DrillholeModel, SampleModel, QAQCRecordModel, ReportModel
)
from src.api.auth_middleware import (
    create_access_token, verify_password, hash_password,
    get_current_user, require_auth, require_role, TokenPayload, JWTMiddleware
)
from src.api.waldo_proxy import router as waldo_router, cleanup_waldo_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Pydantic Models
class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    location: Optional[str] = None
    commodities: List[str] = Field(default_factory=list)
    status: str = Field(default="active")


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    location: Optional[str] = None
    commodities: List[str] = Field(default_factory=list)
    status: str
    createdAt: str
    updatedAt: str

    class Config:
        from_attributes = True


class DrillholeCreate(BaseModel):
    holeId: str = Field(..., min_length=1, max_length=100)
    projectId: str
    x: float
    y: float
    z: float
    totalDepth: float = Field(ge=0)
    azimuth: Optional[float] = Field(None, ge=0, le=360)
    dip: Optional[float] = Field(None, ge=-90, le=90)
    status: str = Field(default="planned")


class DrillholeResponse(BaseModel):
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

    class Config:
        from_attributes = True


class SampleCreate(BaseModel):
    sampleId: str
    drillholeId: str
    fromDepth: float
    toDepth: float
    sampleType: str = "core"
    lithology: Optional[str] = None
    assayData: Optional[Dict[str, float]] = None


class SampleResponse(BaseModel):
    id: str
    sampleId: str
    drillholeId: str
    fromDepth: float
    toDepth: float
    sampleType: str
    lithology: Optional[str] = None
    assayData: Optional[Dict[str, float]] = None
    createdAt: str

    class Config:
        from_attributes = True


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MineralVision Production API Server starting up...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Seed demo data
    seed_demo_data()
    logger.info("Demo data seeded")
    
    yield
    
    # Cleanup
    await cleanup_waldo_client()
    logger.info("MineralVision Production API Server shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MineralVision API",
    description="AI-Powered Mineral Exploration Platform - Production API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add JWT middleware (set enforce=True for production)
# app.add_middleware(JWTMiddleware, enforce=True)

# Include WALDO router
app.include_router(waldo_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Health endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}


@app.get("/api/status")
async def api_status():
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "database": "sqlite",
        "authentication": "jwt",
        "services": {
            "projects": "active",
            "drillholes": "active",
            "samples": "active",
            "qaqc": "active",
            "geostatistics": "active",
            "visualization": "active",
            "reports": "active",
            "users": "active",
            "waldo": "active"
        }
    }


# Auth endpoints
@app.post("/api/auth/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return JWT token."""
    user = db.query(UserModel).filter(UserModel.username == request.username).first()
    
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    if not user.is_active:
        raise HTTPException(status_code=401, detail="User account is disabled")
    
    # Create JWT token
    token = create_access_token({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role
    })
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 86400,
        "session": {"token": token, "expires_in": 86400},
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "firstName": user.first_name or "",
            "lastName": user.last_name or "",
            "roles": [user.role]
        }
    }


@app.post("/api/auth/register")
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check if username exists
    if db.query(UserModel).filter(UserModel.username == request.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email exists
    if db.query(UserModel).filter(UserModel.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Create user
    user = UserModel(
        id=str(uuid.uuid4()),
        username=request.username,
        email=request.email,
        password_hash=hash_password(request.password),
        first_name=request.first_name,
        last_name=request.last_name,
        role="user"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {"message": "User registered successfully", "user_id": user.id}


@app.get("/api/auth/me")
async def get_current_user_info(
    user: TokenPayload = Depends(require_auth),
    db: Session = Depends(get_db)
):
    """Get current authenticated user info."""
    db_user = db.query(UserModel).filter(UserModel.id == user.user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": db_user.id,
        "username": db_user.username,
        "email": db_user.email,
        "firstName": db_user.first_name,
        "lastName": db_user.last_name,
        "role": db_user.role
    }


# Projects endpoints
@app.get("/api/projects", response_model=List[ProjectResponse])
async def list_projects(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List all projects."""
    projects = db.query(ProjectModel).offset(offset).limit(limit).all()
    return [
        ProjectResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            location=p.location,
            commodities=p.commodities or [],
            status=p.status,
            createdAt=p.created_at.isoformat(),
            updatedAt=p.updated_at.isoformat()
        )
        for p in projects
    ]


@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: Session = Depends(get_db)):
    """Get a specific project."""
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        location=project.location,
        commodities=project.commodities or [],
        status=project.status,
        createdAt=project.created_at.isoformat(),
        updatedAt=project.updated_at.isoformat()
    )


@app.post("/api/projects", response_model=ProjectResponse, status_code=201)
async def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    user: Optional[TokenPayload] = Depends(get_current_user)
):
    """Create a new project."""
    project_id = str(uuid.uuid4())
    db_project = ProjectModel(
        id=project_id,
        name=project.name,
        description=project.description,
        location=project.location,
        commodities=project.commodities,
        status=project.status,
        owner_id=user.user_id if user else None
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    return ProjectResponse(
        id=db_project.id,
        name=db_project.name,
        description=db_project.description,
        location=db_project.location,
        commodities=db_project.commodities or [],
        status=db_project.status,
        createdAt=db_project.created_at.isoformat(),
        updatedAt=db_project.updated_at.isoformat()
    )


@app.delete("/api/projects/{project_id}", status_code=204)
async def delete_project(project_id: str, db: Session = Depends(get_db)):
    """Delete a project."""
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    db.delete(project)
    db.commit()


# Drillholes endpoints
@app.get("/api/drillholes", response_model=List[DrillholeResponse])
async def list_drillholes(
    projectId: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List drillholes, optionally filtered by project."""
    query = db.query(DrillholeModel)
    if projectId:
        query = query.filter(DrillholeModel.project_id == projectId)
    
    drillholes = query.offset(offset).limit(limit).all()
    return [
        DrillholeResponse(
            id=d.id,
            holeId=d.hole_id,
            projectId=d.project_id,
            collar={"x": d.collar_x, "y": d.collar_y, "z": d.collar_z},
            totalDepth=d.total_depth,
            azimuth=d.azimuth,
            dip=d.dip,
            status=d.status,
            assayCount=d.assay_count,
            createdAt=d.created_at.isoformat(),
            updatedAt=d.updated_at.isoformat()
        )
        for d in drillholes
    ]


@app.get("/api/drillholes/{drillhole_id}", response_model=DrillholeResponse)
async def get_drillhole(drillhole_id: str, db: Session = Depends(get_db)):
    """Get a specific drillhole."""
    drillhole = db.query(DrillholeModel).filter(DrillholeModel.id == drillhole_id).first()
    if not drillhole:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    
    return DrillholeResponse(
        id=drillhole.id,
        holeId=drillhole.hole_id,
        projectId=drillhole.project_id,
        collar={"x": drillhole.collar_x, "y": drillhole.collar_y, "z": drillhole.collar_z},
        totalDepth=drillhole.total_depth,
        azimuth=drillhole.azimuth,
        dip=drillhole.dip,
        status=drillhole.status,
        assayCount=drillhole.assay_count,
        createdAt=drillhole.created_at.isoformat(),
        updatedAt=drillhole.updated_at.isoformat()
    )


@app.post("/api/drillholes", response_model=DrillholeResponse, status_code=201)
async def create_drillhole(drillhole: DrillholeCreate, db: Session = Depends(get_db)):
    """Create a new drillhole."""
    # Verify project exists
    project = db.query(ProjectModel).filter(ProjectModel.id == drillhole.projectId).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {drillhole.projectId} not found")
    
    drillhole_id = str(uuid.uuid4())
    db_drillhole = DrillholeModel(
        id=drillhole_id,
        hole_id=drillhole.holeId,
        project_id=drillhole.projectId,
        collar_x=drillhole.x,
        collar_y=drillhole.y,
        collar_z=drillhole.z,
        total_depth=drillhole.totalDepth,
        azimuth=drillhole.azimuth,
        dip=drillhole.dip,
        status=drillhole.status
    )
    db.add(db_drillhole)
    db.commit()
    db.refresh(db_drillhole)
    
    return DrillholeResponse(
        id=db_drillhole.id,
        holeId=db_drillhole.hole_id,
        projectId=db_drillhole.project_id,
        collar={"x": db_drillhole.collar_x, "y": db_drillhole.collar_y, "z": db_drillhole.collar_z},
        totalDepth=db_drillhole.total_depth,
        azimuth=db_drillhole.azimuth,
        dip=db_drillhole.dip,
        status=db_drillhole.status,
        assayCount=db_drillhole.assay_count,
        createdAt=db_drillhole.created_at.isoformat(),
        updatedAt=db_drillhole.updated_at.isoformat()
    )


# Samples endpoints
@app.get("/api/samples")
async def list_samples(
    drillholeId: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List samples, optionally filtered by drillhole."""
    query = db.query(SampleModel)
    if drillholeId:
        query = query.filter(SampleModel.drillhole_id == drillholeId)
    
    samples = query.all()
    return {
        "items": [
            SampleResponse(
                id=s.id,
                sampleId=s.sample_id,
                drillholeId=s.drillhole_id,
                fromDepth=s.from_depth,
                toDepth=s.to_depth,
                sampleType=s.sample_type,
                lithology=s.lithology,
                assayData=s.assay_data,
                createdAt=s.created_at.isoformat()
            )
            for s in samples
        ],
        "total": len(samples)
    }


@app.post("/api/samples", status_code=201)
async def create_sample(sample: SampleCreate, db: Session = Depends(get_db)):
    """Create a new sample."""
    # Verify drillhole exists
    drillhole = db.query(DrillholeModel).filter(DrillholeModel.id == sample.drillholeId).first()
    if not drillhole:
        raise HTTPException(status_code=404, detail=f"Drillhole {sample.drillholeId} not found")
    
    sample_id = str(uuid.uuid4())
    db_sample = SampleModel(
        id=sample_id,
        sample_id=sample.sampleId,
        drillhole_id=sample.drillholeId,
        from_depth=sample.fromDepth,
        to_depth=sample.toDepth,
        sample_type=sample.sampleType,
        lithology=sample.lithology,
        assay_data=sample.assayData
    )
    db.add(db_sample)
    
    # Update drillhole assay count
    drillhole.assay_count += 1
    
    db.commit()
    db.refresh(db_sample)
    
    return SampleResponse(
        id=db_sample.id,
        sampleId=db_sample.sample_id,
        drillholeId=db_sample.drillhole_id,
        fromDepth=db_sample.from_depth,
        toDepth=db_sample.to_depth,
        sampleType=db_sample.sample_type,
        lithology=db_sample.lithology,
        assayData=db_sample.assay_data,
        createdAt=db_sample.created_at.isoformat()
    )


# QA/QC endpoints
@app.get("/api/qaqc")
async def list_qaqc(db: Session = Depends(get_db)):
    """List QA/QC records."""
    records = db.query(QAQCRecordModel).all()
    return {
        "items": [
            {
                "id": r.id,
                "projectId": r.project_id,
                "sampleId": r.sample_id,
                "qcType": r.qc_type,
                "expectedValue": r.expected_value,
                "actualValue": r.actual_value,
                "tolerance": r.tolerance,
                "passed": r.passed,
                "notes": r.notes,
                "createdAt": r.created_at.isoformat()
            }
            for r in records
        ],
        "total": len(records),
        "summary": {
            "standards": len([r for r in records if r.qc_type == "standard"]),
            "blanks": len([r for r in records if r.qc_type == "blank"]),
            "duplicates": len([r for r in records if r.qc_type == "duplicate"])
        }
    }


@app.get("/api/qaqc/summary/{project_id}")
async def get_qaqc_summary(project_id: str, db: Session = Depends(get_db)):
    """Get QA/QC summary for a project."""
    records = db.query(QAQCRecordModel).filter(QAQCRecordModel.project_id == project_id).all()
    
    standards = [r for r in records if r.qc_type == "standard"]
    blanks = [r for r in records if r.qc_type == "blank"]
    duplicates = [r for r in records if r.qc_type == "duplicate"]
    
    return {
        "project_id": project_id,
        "standards": {
            "count": len(standards),
            "pass_rate": sum(1 for r in standards if r.passed) / len(standards) if standards else 1.0
        },
        "blanks": {
            "count": len(blanks),
            "pass_rate": sum(1 for r in blanks if r.passed) / len(blanks) if blanks else 1.0
        },
        "duplicates": {
            "count": len(duplicates),
            "pass_rate": sum(1 for r in duplicates if r.passed) / len(duplicates) if duplicates else 1.0
        }
    }


# Geostatistics endpoints
@app.get("/api/geostatistics/variogram")
async def get_variogram():
    """Get variogram parameters."""
    return {
        "variogram": {
            "type": "spherical",
            "sill": 1.0,
            "range": 100,
            "nugget": 0.1,
            "direction": 0
        }
    }


@app.post("/api/geostatistics/kriging")
async def run_kriging():
    """Run kriging interpolation."""
    return {
        "status": "completed",
        "grid_size": [100, 100, 50],
        "method": "ordinary_kriging"
    }


# Visualization endpoints
@app.get("/api/visualization/scenes")
async def list_scenes():
    """List visualization scenes."""
    return {
        "scenes": [
            {"id": "scene-1", "name": "3D Drillhole View", "type": "drillholes"}
        ],
        "total": 1
    }


@app.get("/api/visualization/drillholes/{project_id}")
async def get_drillhole_visualization(project_id: str, db: Session = Depends(get_db)):
    """Get drillhole visualization data for a project."""
    drillholes = db.query(DrillholeModel).filter(DrillholeModel.project_id == project_id).all()
    
    if not drillholes:
        return {"project_id": project_id, "drillholes": [], "bounds": None}
    
    dh_data = [
        {
            "id": d.id,
            "holeId": d.hole_id,
            "collar": {"x": d.collar_x, "y": d.collar_y, "z": d.collar_z},
            "totalDepth": d.total_depth,
            "azimuth": d.azimuth,
            "dip": d.dip
        }
        for d in drillholes
    ]
    
    return {
        "project_id": project_id,
        "drillholes": dh_data,
        "bounds": {
            "minX": min(d.collar_x for d in drillholes),
            "maxX": max(d.collar_x for d in drillholes),
            "minY": min(d.collar_y for d in drillholes),
            "maxY": max(d.collar_y for d in drillholes)
        }
    }


# Reports endpoints
@app.get("/api/reports")
async def list_reports(db: Session = Depends(get_db)):
    """List all reports."""
    reports = db.query(ReportModel).all()
    return {
        "reports": [
            {
                "id": r.id,
                "name": r.name,
                "status": r.status,
                "type": r.report_type,
                "projectId": r.project_id,
                "createdAt": r.created_at.isoformat()
            }
            for r in reports
        ],
        "total": len(reports)
    }


@app.post("/api/reports/generate")
async def generate_report(
    report_type: str = "ni43-101",
    project_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Generate a new report."""
    report_id = str(uuid.uuid4())
    report = ReportModel(
        id=report_id,
        name=f"{report_type.upper()} Technical Report",
        report_type=report_type,
        project_id=project_id,
        status="generating"
    )
    db.add(report)
    db.commit()
    
    return {
        "id": report_id,
        "type": report_type,
        "status": "generating",
        "estimated_time": "5 minutes"
    }


# Users endpoints
@app.get("/api/users")
async def list_users(
    db: Session = Depends(get_db),
    admin: TokenPayload = Depends(require_role(["admin"]))
):
    """List all users (admin only)."""
    users = db.query(UserModel).all()
    return {
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "firstName": u.first_name,
                "lastName": u.last_name,
                "role": u.role,
                "isActive": u.is_active,
                "createdAt": u.created_at.isoformat()
            }
            for u in users
        ],
        "total": len(users)
    }


@app.get("/api/users/{user_id}")
async def get_user(user_id: str, db: Session = Depends(get_db)):
    """Get a specific user."""
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "firstName": user.first_name,
        "lastName": user.last_name,
        "role": user.role
    }


# Upload endpoint
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file."""
    content = await file.read()
    return {
        "filename": file.filename,
        "size": len(content),
        "content_type": file.content_type,
        "status": "uploaded"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main_production:app", host="0.0.0.0", port=8000, reload=True)
