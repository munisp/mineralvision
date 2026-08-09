"""submission_packager — regulatory tenement submission bundle (innovation 15)."""

from fastapi import APIRouter

from .routes import router as _routes

router = APIRouter(prefix="/innovations/submission_packager", tags=["submission_packager"])
router.include_router(_routes)

__all__ = ["router"]
