"""GeoLibre integration module — /innovations/geolibre router.

Authors/serves ``.geolibre.json`` project documents from platform data for
the client-side GeoLibre GIS app (opengeos/GeoLibre).  Core JSON authoring is
pure stdlib; the ``geolibre`` wheel is used lazily when installed.
"""

try:  # dual-context import
    from src.api.innovations.geolibre.routes import router
except ImportError:  # pragma: no cover
    from api.innovations.geolibre.routes import router

__all__ = ["router"]
