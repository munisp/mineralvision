"""doc_intelligence — historical report extraction (innovation 17)."""

from fastapi import APIRouter

from .routes import router as _routes

router = APIRouter(prefix="/innovations/doc_intelligence", tags=["doc_intelligence"])
router.include_router(_routes)

__all__ = ["router"]
