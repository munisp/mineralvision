"""Validate that the oil-spill API router is importable and exposes the required routes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ENTRYPOINT = ROOT / "MineralVision_Final_Package" / "src" / "api" / "main.py"
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package" / "src"))

from api.endpoints.oil_spill import router  # noqa: E402

REQUIRED_ROUTES = {
    "/api/oil-spill/analyze/mask",
    "/api/oil-spill/analyze/image",
    "/api/oil-spill/incidents",
    "/api/oil-spill/incidents/{incident_id}",
    "/api/oil-spill/incidents/{incident_id}/review",
    "/api/oil-spill/incidents/{incident_id}/coverage-plan",
}


def main() -> None:
    actual_routes = {route.path for route in router.routes}
    missing = REQUIRED_ROUTES.difference(actual_routes)
    if missing:
        raise RuntimeError(f"Oil-spill API routes missing: {sorted(missing)}")
    entrypoint_text = CANONICAL_ENTRYPOINT.read_text(encoding="utf-8")
    if "from .endpoints.oil_spill import router as oil_spill_router" not in entrypoint_text:
        raise RuntimeError("Canonical API entrypoint does not import the oil-spill router")
    if "app.include_router(oil_spill_router)" not in entrypoint_text:
        raise RuntimeError("Canonical API entrypoint does not mount the oil-spill router")
    print(f"Oil-spill router verified: {len(actual_routes)} routes available and mounted by src.api.main")


if __name__ == "__main__":
    main()
