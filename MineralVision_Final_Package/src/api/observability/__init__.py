"""
MineralVision API Observability

Application-level observability primitives:
- metrics: Prometheus request metrics + /metrics exposition
- logging_config: structured JSON logging + request-id correlation
- health_checks: real dependency checks backing the /health endpoint
"""

from .metrics import MetricsMiddleware, metrics_endpoint, METRICS_CONTENT_TYPE
from .logging_config import RequestIDMiddleware, setup_logging, get_request_id
from .health_checks import run_health_checks

__all__ = [
    "MetricsMiddleware",
    "metrics_endpoint",
    "METRICS_CONTENT_TYPE",
    "RequestIDMiddleware",
    "setup_logging",
    "get_request_id",
    "run_health_checks",
]
