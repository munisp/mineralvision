"""indigenous_governance — indigenous knowledge access governance (innovation 16)."""

from fastapi import APIRouter

from .routes import router as _routes

router = APIRouter(prefix="/innovations/indigenous_governance", tags=["indigenous_governance"])
router.include_router(_routes)

__all__ = ["router"]
