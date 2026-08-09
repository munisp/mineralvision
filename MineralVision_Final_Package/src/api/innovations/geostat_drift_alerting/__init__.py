"""Geostatistical drift alerting: CUSUM + EWMA + rolling sill/mean monitors."""

from .routes import router

__all__ = ["router"]
