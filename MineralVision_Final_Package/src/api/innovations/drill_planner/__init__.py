"""MineralVision drill_planner — competitive gap #11: drill campaign planning
with rig-execution integration (Leapfrog 2026.1 / Micromine feature parity).

1. POST /patterns/grid       — rotated drill-grid generation with real geodesy
2. POST /collars/snap        — collar validation: DTM slope + keep-out polygons
3. POST /traces/design       — hole design + deviation/uncertainty model
4. POST /campaign/optimize   — budget-constrained greedy weighted set-cover
5. POST /campaign/schedule   — multi-rig sequencing (NN + 2-opt)
   GET  /campaign/export     — rig-software compatible CSV/JSON export
"""

from fastapi import APIRouter

try:
    from src.api.innovations.drill_planner.patterns import router as patterns_router
    from src.api.innovations.drill_planner.collars import router as collars_router
    from src.api.innovations.drill_planner.traces import router as traces_router
    from src.api.innovations.drill_planner.campaign import router as campaign_router
except ImportError:  # pragma: no cover - dual-context import
    from api.innovations.drill_planner.patterns import router as patterns_router
    from api.innovations.drill_planner.collars import router as collars_router
    from api.innovations.drill_planner.traces import router as traces_router
    from api.innovations.drill_planner.campaign import router as campaign_router

router = APIRouter(prefix="/innovations/drill-planner", tags=["drill-planner"])
router.include_router(patterns_router)
router.include_router(collars_router)
router.include_router(traces_router)
router.include_router(campaign_router)

__all__ = ["router"]
