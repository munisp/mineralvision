"""audit_collaboration — audit trail + comments + versioning (innovation 19)."""

from fastapi import APIRouter

from .routes import router as _routes

router = APIRouter(prefix="/innovations/audit_collaboration", tags=["audit_collaboration"])
router.include_router(_routes)

__all__ = ["router"]
