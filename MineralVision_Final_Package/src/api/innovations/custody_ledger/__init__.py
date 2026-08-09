"""custody_ledger — assay chain-of-custody hash ledger (innovation 13)."""

from fastapi import APIRouter

from .routes import router as _routes

router = APIRouter(prefix="/innovations/custody_ledger", tags=["custody_ledger"])
router.include_router(_routes)

__all__ = ["router"]
