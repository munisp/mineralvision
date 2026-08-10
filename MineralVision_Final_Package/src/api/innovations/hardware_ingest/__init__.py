"""hardware_ingest module — /innovations/hardware-ingest router.

Exposes the existing real hardware parsers (LiDAR LAS/LAZ, pXRF CSV,
GNSS, downhole well-log LAS) plus Sentinel-1 InSAR coherence change
metrics over HTTP. No fabrication: missing backends produce 503 with
remediation.
"""

try:  # dual-context import
    from src.api.innovations.hardware_ingest.router import router
except ImportError:  # pragma: no cover
    from api.innovations.hardware_ingest.router import router

__all__ = ["router"]
