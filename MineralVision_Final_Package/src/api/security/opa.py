"""Fail-closed OPA policy enforcement for protected MineralVision operations."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class OPAConfigurationError(RuntimeError):
    """Raised if a production deployment has no usable policy decision point."""


class OPAMiddleware:
    """Evaluate OPA policy for all oil-spill routes after authentication.

    The middleware intentionally sends no bearer token, request body, image bytes, or
    personally sensitive fields to OPA. It defaults to deny on policy errors when
    configured as a production control.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.enabled = os.getenv("OPA_ENABLED", "false").lower() == "true"
        self.fail_closed = os.getenv("OPA_FAIL_CLOSED", "true").lower() == "true"
        self.url = os.getenv("OPA_URL", "").rstrip("/")
        self.timeout_seconds = float(os.getenv("OPA_TIMEOUT_SECONDS", "1.5"))
        environment = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).lower()
        if environment == "production" and not self.enabled:
            raise OPAConfigurationError("OPA_ENABLED=true is required in production")
        if self.enabled and not self.url:
            raise OPAConfigurationError("OPA_URL is required when OPA_ENABLED=true")
        if self.enabled:
            parsed = urlparse(self.url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise OPAConfigurationError("OPA_URL must be a credential-free HTTP(S) URL")
            if environment == "production" and parsed.hostname not in {"opa", "localhost", "127.0.0.1", "::1"}:
                raise OPAConfigurationError("production OPA_URL must target the private OPA service")

    @staticmethod
    def _protected(path: str) -> bool:
        return path.startswith("/api/oil-spill/")

    def _evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.url}/v1/data/mineralvision/authz/decision",
            data=json.dumps({"input": payload}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - controlled OPA URL configuration
            document = json.loads(response.read().decode("utf-8"))
        result = document.get("result")
        if not isinstance(result, dict):
            raise ValueError("OPA response does not contain a decision object")
        return result

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self._protected(scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        if not self.enabled:
            await self.app(scope, receive, send)
            return

        user = scope.get("state", {}).get("user")
        if not isinstance(user, dict):
            # JWT middleware should already have returned 401. Do not permit a
            # route merely because a middleware ordering change removed identity.
            await self._deny(send, "identity_context_missing")
            return
        payload = {
            "subject": {
                "id": user.get("id", ""),
                "roles": user.get("roles", []),
                "mfa_verified": bool(user.get("mfa_verified", False)),
                "project_ids": user.get("project_ids", []),
            },
            "request": {"method": scope.get("method", ""), "path": scope.get("path", "")},
            "resource": {"project_id": ""},
        }
        try:
            decision = await asyncio.to_thread(self._evaluate, payload)
            if decision.get("allow") is True:
                await self.app(scope, receive, send)
                return
            await self._deny(send, str(decision.get("reason", "policy_denied")))
        except (URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as exc:
            logger.error("OPA policy evaluation failed: %s", exc)
            if self.fail_closed:
                await self._deny(send, "policy_unavailable")
            else:
                await self.app(scope, receive, send)

    @staticmethod
    async def _deny(send, reason: str) -> None:
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")],
        })
        await send({"type": "http.response.body", "body": json.dumps({"detail": "Access denied", "reason": reason}).encode("utf-8")})
