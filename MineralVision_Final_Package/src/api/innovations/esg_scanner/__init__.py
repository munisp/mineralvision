"""esg_scanner — ESG/environmental compliance gap scanner (innovation 14)."""

from fastapi import APIRouter

from .routes import router as _routes

router = APIRouter(prefix="/innovations/esg_scanner", tags=["esg_scanner"])
router.include_router(_routes)

__all__ = ["router"]
