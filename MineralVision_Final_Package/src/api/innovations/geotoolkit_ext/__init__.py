"""MineralVision geotoolkit_ext — geospatial innovations 6-10.

6.  spatial-overlay    — /overlay/{operation}      (intersect|union|erase|clip)
7.  change-map-service — /change/map
8.  terrain-3d         — /terrain/mesh
9.  tenement-guard     — /tenements/check, /tenements/expiry-watch, /tenements/alerts
10. geo-crs-service    — /crs/transform, /crs/utm-zone, /crs/detect, /geocode/grid-ref
"""

from fastapi import APIRouter

try:
    from src.api.innovations.geotoolkit_ext.overlay import router as overlay_router
    from src.api.innovations.geotoolkit_ext.change_map import router as change_router
    from src.api.innovations.geotoolkit_ext.terrain3d import router as terrain_router
    from src.api.innovations.geotoolkit_ext.tenement import router as tenement_router
    from src.api.innovations.geotoolkit_ext.crs import router as crs_router
except ImportError:  # pragma: no cover - dual-context import
    from api.innovations.geotoolkit_ext.overlay import router as overlay_router
    from api.innovations.geotoolkit_ext.change_map import router as change_router
    from api.innovations.geotoolkit_ext.terrain3d import router as terrain_router
    from api.innovations.geotoolkit_ext.tenement import router as tenement_router
    from api.innovations.geotoolkit_ext.crs import router as crs_router

router = APIRouter(prefix="/innovations/geotoolkit-ext", tags=["geotoolkit-ext"])
router.include_router(overlay_router)
router.include_router(change_router)
router.include_router(terrain_router)
router.include_router(tenement_router)
router.include_router(crs_router)

__all__ = ["router"]
