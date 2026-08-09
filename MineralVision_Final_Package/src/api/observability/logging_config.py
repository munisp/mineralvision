"""
Structured JSON logging + request-id correlation for the MineralVision API.

Stdlib-only. ``setup_logging`` installs a JSON formatter emitting
``timestamp``, ``level``, ``logger``, ``message`` and ``request_id`` fields.

``RequestIDMiddleware`` (pure ASGI) generates a uuid4 request id per HTTP
request (or propagates an inbound ``X-Request-ID`` header), stores it in a
contextvar for the duration of the request, and returns it in the
``X-Request-ID`` response header.
"""

import contextvars
import json
import logging
import sys
import uuid
from datetime import datetime, timezone

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

REQUEST_ID_HEADER = b"x-request-id"


def get_request_id() -> str:
    """Return the current request id ('-' outside request context)."""
    return _request_id_var.get()


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for structured JSON output on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    # Replace any pre-existing handlers (e.g. basicConfig) so output is JSON-only.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)


class RequestIDMiddleware:
    """Pure-ASGI middleware assigning/propagating X-Request-ID per request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        incoming = headers.get(REQUEST_ID_HEADER)
        request_id = incoming.decode() if incoming else str(uuid.uuid4())

        token = _request_id_var.set(request_id)

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.append((REQUEST_ID_HEADER, request_id.encode()))
                message = {**message, "headers": response_headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            _request_id_var.reset(token)
