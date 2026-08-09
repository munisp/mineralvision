"""integration_hub — webhooks, API keys, SDK surface (innovation 20)."""

from fastapi import APIRouter

from .routes import router as _routes

router = APIRouter(prefix="/innovations/integration_hub", tags=["integration_hub"])
router.include_router(_routes)

__all__ = ["router"]
