"""field_sync — offline field data synchronization (innovation 18)."""

from fastapi import APIRouter

from .routes import router as _routes

router = APIRouter(prefix="/innovations/field_sync", tags=["field_sync"])
router.include_router(_routes)

__all__ = ["router"]
