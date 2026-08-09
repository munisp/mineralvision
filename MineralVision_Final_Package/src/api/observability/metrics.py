"""
Prometheus request metrics for the MineralVision API.

Uses the ``prometheus_client`` library (pinned in requirements.txt) to expose:

- ``mineralvision_http_requests_total{method,path,status}``  (Counter)
- ``mineralvision_http_request_duration_seconds{method,path}`` (Histogram)
- ``mineralvision_http_requests_in_progress``                (Gauge)

A pure-ASGI middleware records every HTTP request, and ``metrics_endpoint``
serves the Prometheus text exposition format at ``/metrics``.

The endpoint is unauthenticated (added to JWTMiddleware.PUBLIC_PATHS in
main.py) on the assumption that /metrics is only reachable on the internal
network / by the Prometheus scraper. Do not expose it publicly without
adding auth.
"""

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST

# Dedicated registry so tests can build isolated app instances without
# duplicate-timeseries errors from the global default registry.
registry = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "mineralvision_http_requests_total",
    "Total HTTP requests handled by the API.",
    labelnames=("method", "path", "status"),
    registry=registry,
)

REQUEST_DURATION = Histogram(
    "mineralvision_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry,
)

REQUESTS_IN_PROGRESS = Gauge(
    "mineralvision_http_requests_in_progress",
    "HTTP requests currently being processed.",
    registry=registry,
)

# Paths that must never appear as metric labels (high-cardinality / noise).
_SKIP_PATHS = {"/metrics"}


class MetricsMiddleware:
    """Pure-ASGI middleware recording request count, latency and concurrency."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        REQUESTS_IN_PROGRESS.inc()
        start = time.perf_counter()
        status_code = 500  # default if the app raises before sending

        async def send_with_metrics(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_metrics)
        finally:
            elapsed = time.perf_counter() - start
            REQUESTS_IN_PROGRESS.dec()
            REQUESTS_TOTAL.labels(
                method=method, path=path, status=str(status_code)
            ).inc()
            REQUEST_DURATION.labels(method=method, path=path).observe(elapsed)


async def metrics_endpoint(request):
    """Serve the Prometheus text exposition of all registered metrics."""
    return Response(content=generate_latest(registry), media_type=METRICS_CONTENT_TYPE)
