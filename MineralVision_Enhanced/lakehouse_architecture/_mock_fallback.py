"""
Shared real-client-first connection policy for MineralVision middleware.

Contract (REMEDIATION wave 2):
- Every middleware component attempts a REAL client connection first
  (short timeout).
- An in-memory mock is used ONLY when the environment variable
  ``MV_ALLOW_MOCK_FALLBACK=true`` is explicitly set (default: false).
- When the mock fallback triggers, a loud warning is logged and the
  component exposes ``degraded: true`` in its health/status response.
- When the fallback is not allowed, connect() raises RuntimeError with a
  clear message — the component never pretends success.
"""

import logging
import os
import socket
from typing import Optional

logger = logging.getLogger(__name__)

MOCK_FALLBACK_ENV = "MV_ALLOW_MOCK_FALLBACK"


def mock_fallback_allowed() -> bool:
    """True only when MV_ALLOW_MOCK_FALLBACK=true is explicitly set."""
    return os.environ.get(MOCK_FALLBACK_ENV, "").strip().lower() == "true"


def real_client_unavailable(component: str, reason: str,
                            error: Optional[BaseException] = None) -> bool:
    """
    Handle a failed/unavailable real client connection.

    Returns True when the in-memory mock fallback is explicitly allowed
    (after logging a loud warning). Raises RuntimeError otherwise.

    Usage::

        try:
            ... real connection ...
        except Exception as exc:
            if real_client_unavailable("Redis", "connection failed", exc):
                self._degraded = True
                self.client = MockClient(...)
    """
    if mock_fallback_allowed():
        logger.warning(
            "MV DEGRADED MODE: %s real client unavailable (%s%s). "
            "Using in-memory MOCK because %s=true. "
            "This component is degraded — do NOT use in production.",
            component,
            reason,
            f": {error}" if error else "",
            MOCK_FALLBACK_ENV,
        )
        return True
    raise RuntimeError(
        f"{component}: real client unavailable ({reason}"
        f"{f': {error}' if error else ''}). Install/start the real backend "
        f"or set {MOCK_FALLBACK_ENV}=true to explicitly enable the "
        f"in-memory mock fallback."
    ) from error


def probe_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    """Attempt a real TCP connection with a short timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_url(url: str, timeout: float = 2.0) -> bool:
    """Attempt a real TCP connection to the host:port of a URL."""
    from urllib.parse import urlparse
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return probe_tcp(host, port, timeout=timeout)
