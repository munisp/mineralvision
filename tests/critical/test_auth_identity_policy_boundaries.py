"""AUTH-01 through AUTH-07 tests for OIDC, MFA, OPA, and tenant boundaries.

The OIDC/JWKS and OPA services are represented by strict local contract doubles.  The
application code that maps identity claims and enforces policy results is real.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import jwt
import pytest
from fastapi import HTTPException

from api.auth_middleware import JWTMiddleware, TokenPayload
from api.authz import project_scope_query, require_project_access
from api.security.oidc import OIDCIdentity, OIDCTokenError, OIDCValidator
from api.security.opa import OPAMiddleware


def _user(*, identifier: str = "owner-a", roles: list[str] | None = None, mfa: bool = False) -> TokenPayload:
    roles = roles or ["user"]
    return TokenPayload(
        user_id=identifier,
        username=identifier,
        email=f"{identifier}@example.test",
        role=roles[0],
        roles=roles,
        project_ids=["project-a"],
        mfa_verified=mfa,
        exp=datetime.utcnow() + timedelta(hours=1),
    )


def _validator_with_claims(monkeypatch, claims: dict):
    validator = OIDCValidator.__new__(OIDCValidator)
    validator.algorithms = ["RS256"]
    validator.audience = "mineralvision-api"
    validator.issuer = "https://id.example.test/realms/mineralvision"
    validator.jwks_client = SimpleNamespace(
        get_signing_key_from_jwt=lambda _token: SimpleNamespace(key="public-key")
    )
    monkeypatch.setattr(jwt, "decode", lambda *_args, **_kwargs: claims)
    return validator


async def _asgi_call(app, *, path: str, method: str = "GET", state: dict | None = None) -> tuple[int, bytes, list[tuple[bytes, bytes]]]:
    messages: list[dict] = []
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [],
        "state": state or {},
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = next((message.get("body", b"") for message in messages if message["type"] == "http.response.body"), b"")
    return start["status"], body, start.get("headers", [])


async def _ok_app(_scope, _receive, send):
    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def test_auth_01_rejects_expired_wrong_issuer_wrong_audience_and_malformed_oidc_tokens(monkeypatch):
    validator = OIDCValidator.__new__(OIDCValidator)
    validator.algorithms = ["RS256"]
    validator.audience = "mineralvision-api"
    validator.issuer = "https://id.example.test/realms/mineralvision"
    validator.jwks_client = SimpleNamespace(get_signing_key_from_jwt=lambda _token: SimpleNamespace(key="public-key"))

    for error in (
        jwt.ExpiredSignatureError("expired"),
        jwt.InvalidIssuerError("wrong issuer"),
        jwt.InvalidAudienceError("wrong audience"),
        jwt.DecodeError("malformed"),
    ):
        monkeypatch.setattr(jwt, "decode", MagicMock(side_effect=error))
        with pytest.raises(OIDCTokenError, match="Invalid OIDC bearer token"):
            validator.validate("untrusted-token")


def test_auth_02_uses_only_asymmetric_jwks_validation_and_rejects_key_lookup_failure(monkeypatch):
    claims = {
        "sub": "user-1", "preferred_username": "operator", "email": "operator@example.test",
        "exp": 2_000_000_000, "iat": 1_900_000_000,
        "realm_access": {"roles": ["oil_spill_operator"]},
    }
    validator = _validator_with_claims(monkeypatch, claims)
    identity = validator.validate("valid-asymmetric-token")
    assert identity.subject == "user-1"
    assert identity.roles == ["oil_spill_operator"]

    validator.jwks_client = SimpleNamespace(get_signing_key_from_jwt=MagicMock(side_effect=RuntimeError("jwks offline")))
    with pytest.raises(OIDCTokenError, match="signing-key retrieval failed"):
        validator.validate("token-with-unknown-kid")


def test_auth_03_mfa_and_privilege_claims_are_explicit_not_inferred(monkeypatch):
    claims = {
        "sub": "reviewer-1", "email": "reviewer@example.test", "exp": 2_000_000_000, "iat": 1_900_000_000,
        "realm_access": {"roles": ["oil_spill_reviewer"]}, "project_ids": ["project-a", "project-a"],
        "amr": ["pwd"],
    }
    identity = _validator_with_claims(monkeypatch, claims).validate("password-only-token")
    assert identity.mfa_verified is False
    assert identity.project_ids == ["project-a"]
    assert identity.roles == ["oil_spill_reviewer"]

    claims["amr"] = ["pwd", "webauthn"]
    claims["resource_access"] = {"mineralvision-api": {"roles": ["oil_spill_approver"]}}
    step_up_identity = _validator_with_claims(monkeypatch, claims).validate("step-up-token")
    assert step_up_identity.mfa_verified is True
    assert step_up_identity.roles == ["oil_spill_approver", "oil_spill_reviewer"]


def test_auth_04_opa_denies_missing_identity_timeout_malformed_and_explicit_deny(monkeypatch):
    monkeypatch.setenv("OPA_ENABLED", "true")
    monkeypatch.setenv("OPA_FAIL_CLOSED", "true")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("ENV", "development")
    middleware = OPAMiddleware(_ok_app)

    missing_status, _body, _headers = asyncio.run(_asgi_call(middleware, path="/api/oil-spill/incidents"))
    assert missing_status == 403

    for failure in (TimeoutError("policy timeout"), ValueError("malformed policy JSON")):
        monkeypatch.setattr(middleware, "_evaluate", MagicMock(side_effect=failure))
        status, body, headers = asyncio.run(
            _asgi_call(middleware, path="/api/oil-spill/incidents", state={"user": {"id": "reviewer", "roles": ["oil_spill_reviewer"], "mfa_verified": True, "project_ids": []}})
        )
        assert status == 403 and b"policy_unavailable" in body
        assert (b"cache-control", b"no-store") in headers

    monkeypatch.setattr(middleware, "_evaluate", lambda _payload: {"allow": False, "reason": "mfa_required"})
    status, body, _headers = asyncio.run(
        _asgi_call(middleware, path="/api/oil-spill/incidents", state={"user": {"id": "reviewer", "roles": ["oil_spill_reviewer"], "mfa_verified": False, "project_ids": []}})
    )
    assert status == 403 and b"mfa_required" in body


def test_auth_05_project_bola_owner_admin_and_missing_resource_matrix():
    owner_project = SimpleNamespace(id="project-a", owner_id="owner-a")

    def fake_db_for(project):
        query = MagicMock()
        query.filter.return_value.first.return_value = project
        db = MagicMock()
        db.query.return_value = query
        return db

    assert require_project_access(fake_db_for(owner_project), "project-a", _user(identifier="owner-a")) is owner_project
    assert require_project_access(fake_db_for(owner_project), "project-a", _user(identifier="admin-1", roles=["security_admin"])) is owner_project
    with pytest.raises(HTTPException) as forbidden:
        require_project_access(fake_db_for(owner_project), "project-a", _user(identifier="owner-b"))
    assert forbidden.value.status_code == 403
    with pytest.raises(HTTPException) as missing:
        require_project_access(fake_db_for(None), "project-does-not-exist", _user())
    assert missing.value.status_code == 404


def test_auth_06_tenant_scope_query_never_uses_client_project_claim_as_owner_override():
    query = MagicMock()
    query.filter.return_value = "owner-scoped-query"
    assert project_scope_query(query, _user(identifier="owner-a")) == "owner-scoped-query"
    assert query.filter.called

    admin_query = MagicMock()
    assert project_scope_query(admin_query, _user(identifier="admin-a", roles=["admin"])) is admin_query
    assert admin_query.filter.called is False


def test_auth_07_jwt_middleware_requires_bearer_for_protected_routes_and_bypasses_only_explicit_public_paths(monkeypatch):
    middleware = JWTMiddleware(_ok_app, enforce=True)
    monkeypatch.setattr("api.auth_middleware.decode_token", lambda _token: None)
    denied, denied_body, denied_headers = asyncio.run(_asgi_call(middleware, path="/api/projects"))
    assert denied == 401 and b"Authentication required" in denied_body
    assert (b"www-authenticate", b"Bearer") in denied_headers

    public_status, _body, _headers = asyncio.run(_asgi_call(middleware, path="/health"))
    assert public_status == 204

    monkeypatch.setattr("api.auth_middleware.decode_token", lambda _token: _user(identifier="owner-a", roles=["oil_spill_reviewer"], mfa=True))
    # Invoke the raw ASGI app with an Authorization header to inspect state injection.
    events = []
    captured_state = {}

    async def state_app(scope, _receive, send):
        captured_state.update(scope["state"]["user"])
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    authenticated = JWTMiddleware(state_app, enforce=True)
    scope = {"type": "http", "method": "GET", "path": "/api/projects", "headers": [(b"authorization", b"Bearer valid-token")]}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        events.append(message)

    asyncio.run(authenticated(scope, receive, send))
    assert next(event for event in events if event["type"] == "http.response.start")["status"] == 204
    assert captured_state == {"id": "owner-a", "username": "owner-a", "email": "owner-a@example.test", "role": "oil_spill_reviewer", "roles": ["oil_spill_reviewer"], "project_ids": ["project-a"], "mfa_verified": True}
