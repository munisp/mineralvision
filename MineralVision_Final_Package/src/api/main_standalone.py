"""
MineralVision API Server - Standalone Demo

Completely standalone FastAPI application for PWA demonstration.
No external module dependencies.

Run with: uvicorn src.api.main_standalone:app --host 0.0.0.0 --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Runtime storage for API operations
projects_db: Dict[str, dict] = {}
drillholes_db: Dict[str, dict] = {}
samples_db: Dict[str, dict] = {}
users_db: Dict[str, dict] = {
    "1": {"id": "1", "email": "admin@mineralvision.com", "name": "Admin User", "role": "admin", "createdAt": datetime.utcnow().isoformat()}
}


# Pydantic Models
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    location: Optional[str] = None
    commodities: List[str] = Field(default_factory=list)
    status: str = Field(default="active")


class Project(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    location: Optional[str] = None
    commodities: List[str] = Field(default_factory=list)
    status: str
    createdAt: str
    updatedAt: str


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


class SampleCreate(BaseModel):
    sampleId: str
    drillholeId: str
    fromDepth: float
    toDepth: float
    sampleType: str = "core"


class Sample(BaseModel):
    id: str
    sampleId: str
    drillholeId: str
    fromDepth: float
    toDepth: float
    sampleType: str
    createdAt: str


class LoginRequest(BaseModel):
    username: str
    password: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MineralVision API Server starting up...")
    # Add demo data
    now = datetime.utcnow().isoformat()
    projects_db["demo-1"] = {
        "id": "demo-1", "name": "Gold Exploration Project", "description": "Demo gold exploration in West Africa",
        "location": "Nigeria", "commodities": ["Gold", "Copper"], "status": "active", "createdAt": now, "updatedAt": now
    }
    projects_db["demo-2"] = {
        "id": "demo-2", "name": "Lithium Assessment", "description": "Lithium brine assessment project",
        "location": "Chile", "commodities": ["Lithium"], "status": "active", "createdAt": now, "updatedAt": now
    }
    drillholes_db["dh-1"] = {
        "id": "dh-1", "holeId": "DH-001", "projectId": "demo-1", "collar": {"x": 500000, "y": 1000000, "z": 350},
        "totalDepth": 250, "azimuth": 45, "dip": -60, "status": "completed", "assayCount": 50, "createdAt": now, "updatedAt": now
    }
    drillholes_db["dh-2"] = {
        "id": "dh-2", "holeId": "DH-002", "projectId": "demo-1", "collar": {"x": 500100, "y": 1000050, "z": 345},
        "totalDepth": 180, "azimuth": 45, "dip": -55, "status": "in_progress", "assayCount": 30, "createdAt": now, "updatedAt": now
    }
    yield
    logger.info("MineralVision API Server shutting down...")


app = FastAPI(
    title="MineralVision API",
    description="AI-Powered Mineral Exploration Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "services": {"projects": "active", "drillholes": "active", "samples": "active", "qaqc": "active",
                     "geostatistics": "active", "visualization": "active", "reports": "active", "users": "active"}
    }


# Auth endpoints
@app.post("/api/auth/login")
async def login(request: LoginRequest):
    token = f"demo_token_{uuid.uuid4().hex[:16]}"
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 3600,
        "session": {"token": token, "expires_in": 3600},
        "user": {
            "id": "1",
            "username": request.username,
            "email": f"{request.username}@mineralvision.com",
            "firstName": "Demo",
            "lastName": "User",
            "roles": ["admin"]
        }
    }


@app.get("/api/auth/me")
async def get_current_user():
    return {"id": "1", "email": "admin@mineralvision.com", "name": "Admin User", "role": "admin"}


# Projects endpoints
@app.get("/api/projects", response_model=List[Project])
async def list_projects(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    projects = list(projects_db.values())[offset:offset + limit]
    return [Project(**p) for p in projects]


@app.get("/api/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return Project(**projects_db[project_id])


@app.post("/api/projects", response_model=Project, status_code=201)
async def create_project(project: ProjectCreate):
    project_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    project_data = {"id": project_id, **project.model_dump(), "createdAt": now, "updatedAt": now}
    projects_db[project_id] = project_data
    return Project(**project_data)


@app.delete("/api/projects/{project_id}", status_code=204)
async def delete_project(project_id: str):
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    del projects_db[project_id]


# Drillholes endpoints
@app.get("/api/drillholes", response_model=List[Drillhole])
async def list_drillholes(projectId: Optional[str] = None, limit: int = 100, offset: int = 0):
    drillholes = list(drillholes_db.values())
    if projectId:
        drillholes = [d for d in drillholes if d.get("projectId") == projectId]
    return [Drillhole(**d) for d in drillholes[offset:offset + limit]]


@app.get("/api/drillholes/{drillhole_id}", response_model=Drillhole)
async def get_drillhole(drillhole_id: str):
    if drillhole_id not in drillholes_db:
        raise HTTPException(status_code=404, detail=f"Drillhole {drillhole_id} not found")
    return Drillhole(**drillholes_db[drillhole_id])


@app.post("/api/drillholes", response_model=Drillhole, status_code=201)
async def create_drillhole(drillhole: DrillholeCreate):
    drillhole_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    drillhole_data = {
        "id": drillhole_id, "holeId": drillhole.holeId, "projectId": drillhole.projectId,
        "collar": {"x": drillhole.x, "y": drillhole.y, "z": drillhole.z},
        "totalDepth": drillhole.totalDepth, "azimuth": drillhole.azimuth, "dip": drillhole.dip,
        "status": drillhole.status, "assayCount": 0, "createdAt": now, "updatedAt": now
    }
    drillholes_db[drillhole_id] = drillhole_data
    return Drillhole(**drillhole_data)


# Samples endpoints
@app.get("/api/samples")
async def list_samples(drillholeId: Optional[str] = None):
    samples = list(samples_db.values())
    if drillholeId:
        samples = [s for s in samples if s.get("drillholeId") == drillholeId]
    return {"items": samples, "total": len(samples)}


@app.post("/api/samples", status_code=201)
async def create_sample(sample: SampleCreate):
    sample_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    sample_data = {"id": sample_id, **sample.model_dump(), "createdAt": now}
    samples_db[sample_id] = sample_data
    return sample_data


# QA/QC endpoints
@app.get("/api/qaqc")
async def list_qaqc():
    return {"items": [], "total": 0, "summary": {"standards": 0, "blanks": 0, "duplicates": 0}}


@app.get("/api/qaqc/summary/{project_id}")
async def get_qaqc_summary(project_id: str):
    return {"project_id": project_id, "standards": {"count": 12, "pass_rate": 0.95},
            "blanks": {"count": 8, "pass_rate": 1.0}, "duplicates": {"count": 15, "pass_rate": 0.93}}


# Geostatistics endpoints
@app.get("/api/geostatistics/variogram")
async def get_variogram():
    return {"variogram": {"type": "spherical", "sill": 1.0, "range": 100, "nugget": 0.1, "direction": 0}}


@app.post("/api/geostatistics/kriging")
async def run_kriging():
    return {"status": "completed", "grid_size": [100, 100, 50], "method": "ordinary_kriging"}


# Visualization endpoints
@app.get("/api/visualization/scenes")
async def list_scenes():
    return {"scenes": [{"id": "scene-1", "name": "3D Drillhole View", "type": "drillholes"}], "total": 1}


@app.get("/api/visualization/drillholes/{project_id}")
async def get_drillhole_visualization(project_id: str):
    drillholes = [d for d in drillholes_db.values() if d.get("projectId") == project_id]
    return {"project_id": project_id, "drillholes": drillholes, "bounds": {"minX": 499900, "maxX": 500200, "minY": 999900, "maxY": 1000100}}


# Reports endpoints
@app.get("/api/reports")
async def list_reports():
    return {"reports": [{"id": "report-1", "name": "NI 43-101 Technical Report", "status": "draft", "type": "ni43-101"}], "total": 1}


@app.post("/api/reports/generate")
async def generate_report(report_type: str = "ni43-101"):
    return {"id": str(uuid.uuid4()), "type": report_type, "status": "generating", "estimated_time": "5 minutes"}


# Users endpoints
@app.get("/api/users")
async def list_users():
    return {"users": list(users_db.values()), "total": len(users_db)}


@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return users_db[user_id]


# Upload endpoint
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "size": len(content), "content_type": file.content_type, "status": "uploaded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main_standalone:app", host="0.0.0.0", port=8000, reload=True)
