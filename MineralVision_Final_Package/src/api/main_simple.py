"""
MineralVision API Server - Simplified Entry Point for Demo

This module provides a simplified FastAPI application that wires together
the core API routers without heavy dependencies for quick demo purposes.

Run with: uvicorn src.api.main_simple:app --reload --host 0.0.0.0 --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core routers (no heavy dependencies)
from .endpoints.projects import router as projects_router
from .endpoints.drillholes import router as drillholes_router
from .endpoints.samples import router as samples_router
from .endpoints.qaqc import router as qaqc_router
from .endpoints.geostatistics_api import router as geostatistics_router
from .endpoints.visualization import router as visualization_router
from .endpoints.inversion import router as inversion_router
from .endpoints.reports import router as reports_router
from .endpoints.users import router as users_router
from .endpoints.auth import router as auth_router
from .endpoints.upload import router as upload_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info("MineralVision API Server starting up...")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    yield
    logger.info("MineralVision API Server shutting down...")


# Create FastAPI application
app = FastAPI(
    title="MineralVision API",
    description="""
    MineralVision - AI-Powered Mineral Exploration Platform
    
    A comprehensive platform for mineral exploration integrating:
    - Geology (drillholes, samples, QA/QC)
    - Geostatistics (variography, kriging, block modeling)
    - Geophysics (inversion, forward modeling)
    - Visualization (3D scenes, cross-sections)
    - Reports (NI 43-101, JORC compliance)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# API status endpoint
@app.get("/api/status", tags=["health"])
async def api_status():
    """Get API server status and available services."""
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "projects": "active",
            "drillholes": "active",
            "samples": "active",
            "qaqc": "active",
            "geostatistics": "active",
            "geophysics": "active",
            "visualization": "active",
            "reports": "active",
            "users": "active"
        }
    }


# Mount routers with consistent /api prefix
app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
app.include_router(users_router, prefix="/api/users", tags=["users"])
app.include_router(projects_router, prefix="/api/projects", tags=["projects"])
app.include_router(drillholes_router, prefix="/api/drillholes", tags=["drillholes"])
app.include_router(samples_router, prefix="/api/samples", tags=["samples"])
app.include_router(upload_router, prefix="/api/upload", tags=["upload"])
app.include_router(qaqc_router, prefix="/api/qaqc", tags=["qaqc"])
app.include_router(geostatistics_router, prefix="/api/geostatistics", tags=["geostatistics"])
app.include_router(inversion_router, prefix="/api/inversion", tags=["geophysics"])
app.include_router(visualization_router, prefix="/api/visualization", tags=["visualization"])
app.include_router(reports_router, prefix="/api/reports", tags=["reports"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main_simple:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("ENVIRONMENT", "development") == "development"
    )
