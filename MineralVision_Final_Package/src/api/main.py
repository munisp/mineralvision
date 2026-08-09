"""
MineralVision API Server - Canonical Entry Point

This module provides the single unified FastAPI application that wires
together all API routers for the MineralVision platform, with the security
contracts from REMEDIATION_SPEC C2 enforced:

- PyJWT authentication enforced globally via JWTMiddleware (public paths:
  /auth/login, /auth/register, /health, /docs, /openapi.json, /redoc)
- bcrypt password hashing (work factor 12)
- CORS restricted via CORS_ORIGINS env (default localhost:3000/5173)
- Demo data seeded only when SEED_DEMO=true (ADMIN_INITIAL_PASSWORD required)
- Postgres via DATABASE_URL; SQLite is a dev-only fallback

Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database and security infrastructure
from .database import init_db, seed_demo_data
from .auth_middleware import JWTMiddleware

# Core routers
from .endpoints.auth import router as auth_router
from .endpoints.users import router as users_router
from .endpoints.projects import router as projects_router
from .endpoints.drillholes import router as drillholes_router
from .endpoints.drillholes_simple import router as drillholes_simple_router
from .endpoints.samples import router as samples_router
from .endpoints.qaqc import router as qaqc_router
from .endpoints.upload import router as upload_router
from .endpoints.reports import router as reports_router

# Geostatistics and geophysics routers
from .endpoints.geostatistics_api import router as geostatistics_router
from .endpoints.inversion import router as inversion_router
from .endpoints.visualization import router as visualization_router

# Platform routers
from .endpoints.sensor_fusion import router as sensor_fusion_router
from .endpoints.predictive_modeling import router as predictive_modeling_router
from .endpoints.blockchain import router as blockchain_router
from .endpoints.climate_resilience import router as climate_resilience_router
from .endpoints.digital_twin import router as digital_twin_router
from .endpoints.autonomous_exploration import router as autonomous_exploration_router
from .endpoints.indigenous_knowledge import router as indigenous_knowledge_router

# Orchestration router
from .endpoints.journeys import router as journeys_router

# WALDO proxy router
from .waldo_proxy import router as waldo_router, cleanup_waldo_client


def _cors_origins() -> list:
    """CORS origins from env; never '*' with credentials."""
    raw = os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info("MineralVision API Server starting up...")
    logger.info(f"Environment: {os.environ.get('ENV', 'development')}")

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Seed demo data only when explicitly requested
    if os.environ.get("SEED_DEMO", "").lower() == "true":
        admin_password = os.environ.get("ADMIN_INITIAL_PASSWORD")
        if not admin_password:
            raise RuntimeError(
                "SEED_DEMO=true requires ADMIN_INITIAL_PASSWORD to be set. "
                "Refusing to seed with default credentials."
            )
        seed_demo_data(admin_password=admin_password)
        logger.info("Demo data seeded (SEED_DEMO=true)")

    yield

    # Shutdown
    await cleanup_waldo_client()
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

# Enforce JWT authentication globally (public paths are the only exceptions)
app.add_middleware(JWTMiddleware, enforce=True)

# Configure CORS from environment (never '*' with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
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
        content={"detail": "Internal server error"}
    )


# Health check endpoint (public)
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
        "authentication": "jwt",
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
            "waldo": "active"
        }
    }


# Mount routers
# Auth router is mounted at both /auth (public contract paths) and /api/auth
# (consistent with the rest of the API surface).
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
app.include_router(users_router, prefix="/api/users", tags=["users"])
app.include_router(projects_router, prefix="/api/projects", tags=["projects"])
app.include_router(drillholes_router, prefix="/api/drillholes", tags=["drillholes"])
app.include_router(drillholes_simple_router, prefix="/api/drillholes-simple", tags=["drillholes"])
app.include_router(samples_router, prefix="/api/samples", tags=["samples"])
app.include_router(upload_router, prefix="/api/upload", tags=["upload"])
app.include_router(qaqc_router, prefix="/api/qaqc", tags=["qaqc"])

# Geostatistics and geophysics routers
app.include_router(geostatistics_router, prefix="/api/geostatistics", tags=["geostatistics"])
app.include_router(inversion_router, prefix="/api/inversion", tags=["geophysics"])
app.include_router(visualization_router, prefix="/api/visualization", tags=["visualization"])

# Reports router
app.include_router(reports_router, prefix="/api/reports", tags=["reports"])

# Sensor fusion router (router has /sensor-fusion prefix, we add /api)
app.include_router(sensor_fusion_router, prefix="/api", tags=["sensor-fusion"])

# AI/ML router (predictive_modeling already has /api/predictive-modeling prefix)
app.include_router(predictive_modeling_router, tags=["predictive-modeling"])

# Blockchain router (already has /api/blockchain prefix)
app.include_router(blockchain_router, tags=["blockchain"])

# Platform routers (add /api prefix; routers carry their own sub-prefix)
app.include_router(climate_resilience_router, prefix="/api", tags=["climate-resilience"])
app.include_router(digital_twin_router, prefix="/api", tags=["digital-twin"])
app.include_router(autonomous_exploration_router, prefix="/api", tags=["autonomous-exploration"])
app.include_router(indigenous_knowledge_router, prefix="/api", tags=["indigenous-knowledge"])

# Orchestration/Journeys router (already has /api/journeys prefix)
app.include_router(journeys_router, tags=["journeys"])

# WALDO proxy router
app.include_router(waldo_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("ENV", "development") == "development"
    )
