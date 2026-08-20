"""OIDC bearer-token validation for production identity-provider integration.

The module validates issuer, audience, signature, expiry, and required subject claims
against a configured JWKS endpoint. It performs no password handling and accepts no
unsigned or symmetric application-issued token when AUTH_MODE=oidc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import jwt


class OIDCConfigurationError(RuntimeError):
    """Raised when production OIDC identity validation is not configured safely."""


class OIDCTokenError(RuntimeError):
    """Raised for a rejected bearer token without exposing validation internals."""


@dataclass(frozen=True)
class OIDCIdentity:
    subject: str
    username: str
    email: str
    roles: list[str]
    project_ids: list[str]
    mfa_verified: bool
    expires_at: int
    token_id: str | None

    @property
    def role(self) -> str:
        """Compatibility role for legacy callers; PBAC must use roles instead."""
        return self.roles[0] if self.roles else "user"


class OIDCValidator:
    def __init__(self) -> None:
        self.issuer = os.getenv("OIDC_ISSUER", "").rstrip("/")
        self.audience = os.getenv("OIDC_AUDIENCE", "")
        self.jwks_url = os.getenv("OIDC_JWKS_URL", "")
        algorithms = os.getenv("OIDC_ALLOWED_ALGORITHMS", "RS256,ES256")
        self.algorithms = [item.strip() for item in algorithms.split(",") if item.strip()]
        if not self.issuer or not self.audience or not self.jwks_url:
            raise OIDCConfigurationError("OIDC_ISSUER, OIDC_AUDIENCE, and OIDC_JWKS_URL are required")
        if not self.algorithms or any(algorithm.startswith("HS") for algorithm in self.algorithms):
            raise OIDCConfigurationError("OIDC_ALLOWED_ALGORITHMS must contain only asymmetric algorithms")
        self.jwks_client = jwt.PyJWKClient(self.jwks_url, cache_keys=True, lifespan=300)

    @staticmethod
    def _roles(claims: dict[str, Any]) -> list[str]:
        roles: set[str] = set()
        realm_access = claims.get("realm_access", {})
        if isinstance(realm_access, dict):
            roles.update(str(role) for role in realm_access.get("roles", []) if role)
        resource_access = claims.get("resource_access", {})
        if isinstance(resource_access, dict):
            for client in resource_access.values():
                if isinstance(client, dict):
                    roles.update(str(role) for role in client.get("roles", []) if role)
        direct_roles = claims.get("roles", [])
        if isinstance(direct_roles, list):
            roles.update(str(role) for role in direct_roles if role)
        return sorted(roles)

    @staticmethod
    def _mfa_verified(claims: dict[str, Any]) -> bool:
        amr = claims.get("amr", [])
        if isinstance(amr, str):
            amr = [amr]
        normalized_amr = {str(value).lower() for value in amr}
        acr = str(claims.get("acr", "")).lower()
        return bool({"mfa", "otp", "webauthn", "hwk"} & normalized_amr) or acr in {"mfa", "aal2", "aal3", "gold"}

    @staticmethod
    def _project_ids(claims: dict[str, Any]) -> list[str]:
        value = claims.get("project_ids", [])
        if isinstance(value, str):
            value = [value]
        return sorted({str(item) for item in value if item}) if isinstance(value, list) else []

    def validate(self, token: str) -> OIDCIdentity:
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "sub"]},
                leeway=30,
            )
        except jwt.PyJWTError as exc:
            raise OIDCTokenError("Invalid OIDC bearer token") from exc
        except Exception as exc:
            raise OIDCTokenError("OIDC signing-key retrieval failed") from exc
        subject = str(claims.get("sub", ""))
        if not subject:
            raise OIDCTokenError("OIDC token has no subject")
        return OIDCIdentity(
            subject=subject,
            username=str(claims.get("preferred_username") or claims.get("email") or subject),
            email=str(claims.get("email") or ""),
            roles=self._roles(claims),
            project_ids=self._project_ids(claims),
            mfa_verified=self._mfa_verified(claims),
            expires_at=int(claims["exp"]),
            token_id=str(claims["jti"]) if claims.get("jti") else None,
        )
