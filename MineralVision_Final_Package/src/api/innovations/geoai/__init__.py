"""geoai integration module — /innovations/geoai router."""

try:  # dual-context import
    from src.api.innovations.geoai.router import router
except ImportError:  # pragma: no cover
    from api.innovations.geoai.router import router

__all__ = ["router"]
