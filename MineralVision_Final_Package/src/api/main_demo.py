"""
MineralVision API Server - Demo Entry Point

Minimal FastAPI application for PWA demonstration.
Run with: uvicorn src.api.main_demo:app --reload --host 0.0.0.0 --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import only routers without heavy dependencies
from .endpoints.projects import router as projects_router
from .endpoints.samples import router as samples_router
from .endpoints.auth import router as auth_router
from .endpoints.upload import router as upload_router
from .endpoints.drillholes_simple import router as drillholes_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MineralVision API Server starting up...")
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
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)})


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}


@app.get("/api/status", tags=["health"])
async def api_status():
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "projects": "active",
            "drillholes": "active",
            "samples": "active",
            "qaqc": "demo",
            "geostatistics": "demo",
            "visualization": "demo",
            "reports": "demo",
            "users": "demo"
        }
    }


# Demo endpoints for features with heavy dependencies
@app.get("/api/qaqc", tags=["qaqc"])
async def list_qaqc():
    return {"items": [], "total": 0}


@app.get("/api/geostatistics/variogram", tags=["geostatistics"])
async def get_variogram():
    return {"variogram": {"type": "spherical", "sill": 1.0, "range": 100, "nugget": 0.1}}


@app.get("/api/visualization/scenes", tags=["visualization"])
async def list_scenes():
    return {"scenes": [], "total": 0}


@app.get("/api/reports", tags=["reports"])
async def list_reports():
    return {"reports": [], "total": 0}


@app.get("/api/users", tags=["users"])
async def list_users():
    return {"users": [{"id": "1", "email": "admin@mineralvision.com", "name": "Admin User", "role": "admin"}], "total": 1}


@app.get("/api/users/me", tags=["users"])
async def get_current_user():
    return {"id": "1", "email": "admin@mineralvision.com", "name": "Admin User", "role": "admin"}


# Mount routers
app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
app.include_router(projects_router, prefix="/api/projects", tags=["projects"])
app.include_router(drillholes_router, prefix="/api/drillholes", tags=["drillholes"])
app.include_router(samples_router, prefix="/api/samples", tags=["samples"])
app.include_router(upload_router, prefix="/api/upload", tags=["upload"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main_demo:app", host="0.0.0.0", port=8000, reload=True)
