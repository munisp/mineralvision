"""
MineralVision geotoolkit innovations (items 1-5 of GEOSPATIAL_SPEC):

1. raster-tiles        — XYZ PNG tiles from registered rasters
2. vector-geojson-tiles — bbox-clipped GeoJSON tiles from a feature store
3. drillhole-3d        — minimum-curvature desurvey + three.js scene JSON
4. terrain-profile     — DTM elevation profiles and cross-sections
5. targeting-tiles     — kriging/IDW prospectivity surface -> PNG tiles
"""

from fastapi import APIRouter

from .raster_tiles import router as raster_tiles_router
from .vector_tiles import router as vector_tiles_router
from .drillhole3d import router as drillhole3d_router
from .terrain import router as terrain_router
from .targeting import router as targeting_router

router = APIRouter(prefix="/innovations/geotoolkit", tags=["geotoolkit"])
router.include_router(raster_tiles_router)
router.include_router(vector_tiles_router)
router.include_router(drillhole3d_router)
router.include_router(terrain_router)
router.include_router(targeting_router)

__all__ = ["router"]
