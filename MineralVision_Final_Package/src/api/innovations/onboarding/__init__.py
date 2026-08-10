"""onboarding — stakeholder onboarding: organizations, invitations,
role-scoped access, and a real password-reset token flow."""

from fastapi import APIRouter

from .routes import router as _routes

router = APIRouter(prefix="/innovations/onboarding", tags=["onboarding"])
router.include_router(_routes)

__all__ = ["router"]
