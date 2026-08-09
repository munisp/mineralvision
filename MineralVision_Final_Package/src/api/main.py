"""
MineralVision API Server - Main Entry Point

This module provides the unified FastAPI application that wires together
all API routers for the MineralVision platform.

Run with: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
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

# Import existing routers
from .endpoints.sensor_fusion import router as sensor_fusion_router
from .endpoints.predictive_modeling import router as predictive_modeling_router
from .endpoints.blockchain import router as blockchain_router
from .endpoints.climate_resilience import router as climate_resilience_router
from .endpoints.digital_twin import router as digital_twin_router
from .endpoints.autonomous_exploration import router as autonomous_exploration_router
from .endpoints.indigenous_knowledge import router as indigenous_knowledge_router

# Import new routers (to be created)
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

# Import orchestration router
from .endpoints.journeys import router as journeys_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info("MineralVision API Server starting up...")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    yield
    # Shutdown
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
    - Sensor Fusion (magnetometry, radiometrics, LiDAR, GPR)
    - AI/ML (predictive modeling, prospectivity mapping)
    - Digital Twin (3D visualization, real-time streaming)
    - Climate Resilience Analysis
    - Autonomous Exploration
    - Indigenous Knowledge Integration
    - Blockchain Data Provenance
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
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
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
            "sensor_fusion": "active",
            "predictive_modeling": "active",
            "blockchain": "active",
            "climate_resilience": "active",
            "digital_twin": "active",
            "autonomous_exploration": "active",
                "indigenous_knowledge": "active",
                "reports": "active",
                "users": "active",
                "journeys": "active",
                "orchestration": "active"
            }
        }


# Mount routers with consistent /api prefix
# Core geology/mining routers
app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
app.include_router(users_router, prefix="/api/users", tags=["users"])
app.include_router(projects_router, prefix="/api/projects", tags=["projects"])
app.include_router(drillholes_router, prefix="/api/drillholes", tags=["drillholes"])
app.include_router(samples_router, prefix="/api/samples", tags=["samples"])
app.include_router(upload_router, prefix="/api/upload", tags=["upload"])
app.include_router(qaqc_router, prefix="/api/qaqc", tags=["qaqc"])

# Geostatistics and geophysics routers
app.include_router(geostatistics_router, prefix="/api/geostatistics", tags=["geostatistics"])
app.include_router(inversion_router, prefix="/api/inversion", tags=["geophysics"])
app.include_router(visualization_router, prefix="/api/visualization", tags=["visualization"])

# Reports router
app.include_router(reports_router, prefix="/api/reports", tags=["reports"])

# Sensor fusion router (normalize prefix - router has /sensor-fusion, we add /api)
app.include_router(sensor_fusion_router, prefix="/api", tags=["sensor-fusion"])

# AI/ML routers (predictive_modeling already has /api/predictive-modeling prefix)
app.include_router(predictive_modeling_router, tags=["predictive-modeling"])

# Blockchain router (already has /api/blockchain prefix)
app.include_router(blockchain_router, tags=["blockchain"])

# Climate resilience router (add /api prefix)
app.include_router(climate_resilience_router, prefix="/api", tags=["climate-resilience"])

# Digital twin router (add /api prefix)
app.include_router(digital_twin_router, prefix="/api", tags=["digital-twin"])

# Autonomous exploration router (add /api prefix)
app.include_router(autonomous_exploration_router, prefix="/api", tags=["autonomous-exploration"])

# Indigenous knowledge router (add /api prefix)
app.include_router(indigenous_knowledge_router, prefix="/api", tags=["indigenous-knowledge"])

# Orchestration/Journeys router (already has /api/journeys prefix)
app.include_router(journeys_router, tags=["journeys"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("ENVIRONMENT", "development") == "development"
    )
