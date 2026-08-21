"""Additional branch coverage for the critical FIN and AUTH module set.

The file deliberately exercises error handling and policy-deny branches that are
otherwise easy to leave untested.  It does not contact external payment or OIDC
systems and never moves real value.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MineralVision_Enhanced"))

from middleware.financial.tigerbeetle_ledger import (  # noqa: E402
    Account,
    AccountFlags,
    AccountManager,
    InMemoryTransferControlStore,
    MockTigerBeetleClient,
    TransferControlError,
    TransferIntent,
    TransferManager,
    TransferPolicy,
    TransferResult,
    TransferApproval,
    TigerBeetleConfig,
    _validate_controlled_intent,
    _validate_transfer_request,
)
from api import auth_middleware as auth  # noqa: E402
from api.security.oidc import OIDCConfigurationError, OIDCValidator  # noqa: E402
from api.security.opa import OPAConfigurationError, OPAMiddleware  # noqa: E402


async def _asgi(app, path: str, *, state: dict | None = None):
    messages = []
    scope = {"type": "http", "method": "GET", "path": path, "headers": [], "state": state or {}}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    return next(message["status"] for message in messages if message["type"] == "http.response.start")


async def _ok(_scope, _receive, send):
    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def test_fin_branch_validation_rejects_bad_value_transfer_inputs():
    for debit, credit, amount in ((0, 2, 1), (1, 0, 1), (1, 1, 1), (1, 2, 0), (1, 2, -1), (1, 2, True)):
        with pytest.raises(ValueError):
            _validate_transfer_request(debit, credit, amount)

    intent = TransferIntent("key", "maker", 1, 2, 1, "USD", 1, 1, "purpose", "reference")
    policy = TransferPolicy("USD", 10)
    _validate_controlled_intent(intent, policy)
    for changed, expected in (({"currency": "EUR"}, "currency"), ({"ledger": 0}, "ledger"), ({"purpose": ""}, "purpose")):
        altered = TransferIntent(**{**intent.__dict__, **changed})
        with pytest.raises(TransferControlError, match=expected):
            _validate_controlled_intent(altered, policy)


def test_fin_branch_mock_ledger_regular_pending_post_void_and_account_history():
    async def scenario():
        client = MockTigerBeetleClient(TigerBeetleConfig())
        accounts = AccountManager(client)
        debit = await accounts.create(1, 1000, AccountFlags.NONE)
        credit = await accounts.create(1, 4000)
        manager = TransferManager(client)

        regular = await manager.transfer(debit.id, credit.id, 15, code=1)
        assert regular.success and (await accounts.get_balance(credit.id))["credits_posted"] == 15
        pending = await manager.create_pending(debit.id, credit.id, 10, code=1)
        assert pending.success and (await accounts.get_balance(debit.id))["debits_pending"] == 10
        posted = await manager.post_pending(pending.transfer_id)
        assert posted.success and (await accounts.get_balance(debit.id))["debits_pending"] == 0
        second = await manager.create_pending(debit.id, credit.id, 7, code=1)
        voided = await manager.void_pending(second.transfer_id)
        assert voided.success and (await accounts.get_balance(credit.id))["credits_pending"] == 0
        history = await accounts.get_history(credit.id)
        assert {item.id for item in history} == {regular.transfer_id, posted.transfer_id}
        assert await manager.get(999999) is None
        assert await accounts.get(999999) is None

    asyncio.run(scenario())


def test_fin_branch_account_constraints_and_missing_account_errors():
    async def scenario():
        client = MockTigerBeetleClient(TigerBeetleConfig())
        await client.create_accounts([Account(id=10, ledger=1, code=1, flags=AccountFlags.CREDITS_MUST_NOT_EXCEED_DEBITS)])
        manager = TransferManager(client)
        missing = await manager.transfer(10, 999, 1, code=1)
        assert missing.success is False and missing.error_code == "credit_account_not_found"
        # A direct transfer to a valid account proves the debit balance constraint branch.
        await client.create_accounts([Account(id=20, ledger=1, code=2)])
        rejected = await manager.transfer(10, 20, 1, code=1)
        assert rejected.success is False and rejected.error_code == "exceeds_credits"

    asyncio.run(scenario())


def test_fin_branch_in_memory_control_store_conflict_and_completion_conflict():
    async def scenario():
        store = InMemoryTransferControlStore(b"a-test-audit-key-that-is-long-enough-for-a-test")
        intent = TransferIntent("in-memory-key", "maker", 1, 2, 2, "USD", 1, 1, "purpose", "reference")
        assert await store.reserve(intent) is None
        changed = TransferIntent("in-memory-key", "maker", 1, 2, 3, "USD", 1, 1, "purpose", "reference")
        with pytest.raises(Exception):
            await store.reserve(changed)
        event = await store.append_audit("requested", intent, "maker", {"safe": True})
        assert len(event) == 64
        receipt = type("Receipt", (), {"idempotency_key": "in-memory-key", "request_hash": intent.payload_hash()})()
        await store.complete(receipt)
        conflict = type("Receipt", (), {"idempotency_key": "in-memory-key", "request_hash": "different"})()
        with pytest.raises(Exception):
            await store.complete(conflict)

    asyncio.run(scenario())


def test_auth_branch_local_token_hash_password_blacklist_and_role_requirement(monkeypatch):
    if auth.AUTH_MODE != "local":
        pytest.skip("local-token test must run in local compatibility mode")
    hashed = auth.hash_password("correct horse battery staple")
    assert auth.verify_password("correct horse battery staple", hashed) is True
    assert auth.verify_password("wrong", hashed) is False
    assert auth.verify_password(None, hashed) is False

    token = auth.create_access_token({"id": "user-1", "username": "u", "email": "u@example.test", "role": "admin"})
    payload = auth.decode_token(token)
    assert payload and payload.user_id == "user-1" and payload.roles == ["admin"]
    auth.blacklist_token(payload.jti)
    assert auth.decode_token(token) is None
    with pytest.raises(HTTPException) as missing:
        asyncio.run(auth.require_auth(None))
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as invalid:
        asyncio.run(auth.require_auth(HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad")))
    assert invalid.value.status_code == 401

    checker = auth.require_role(["admin"])
    allowed = asyncio.run(checker(auth.TokenPayload("u", "u", "", "admin", datetime.utcnow() + timedelta(hours=1))))
    assert allowed.role == "admin"
    with pytest.raises(HTTPException) as forbidden:
        asyncio.run(checker(auth.TokenPayload("u", "u", "", "user", datetime.utcnow() + timedelta(hours=1))))
    assert forbidden.value.status_code == 403


def test_auth_branch_oidc_configuration_and_claim_shape_validation(monkeypatch):
    for key in ("OIDC_ISSUER", "OIDC_AUDIENCE", "OIDC_JWKS_URL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(OIDCConfigurationError, match="required"):
        OIDCValidator()

    monkeypatch.setenv("OIDC_ISSUER", "https://id.example.test/realm")
    monkeypatch.setenv("OIDC_AUDIENCE", "api")
    monkeypatch.setenv("OIDC_JWKS_URL", "https://id.example.test/jwks")
    monkeypatch.setenv("OIDC_ALLOWED_ALGORITHMS", "HS256")
    with pytest.raises(OIDCConfigurationError, match="asymmetric"):
        OIDCValidator()

    assert OIDCValidator._roles({"roles": "not-a-list"}) == []
    assert OIDCValidator._project_ids({"project_ids": {"wrong": "shape"}}) == []
    assert OIDCValidator._mfa_verified({"amr": "otp"}) is True
    assert OIDCValidator._mfa_verified({"acr": "gold"}) is True


def test_auth_branch_opa_configuration_disabled_bypass_and_explicit_allow(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("OPA_ENABLED", "false")
    with pytest.raises(OPAConfigurationError, match="required in production"):
        OPAMiddleware(_ok)

    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("OPA_ENABLED", "true")
    monkeypatch.delenv("OPA_URL", raising=False)
    with pytest.raises(OPAConfigurationError, match="OPA_URL"):
        OPAMiddleware(_ok)
    monkeypatch.setenv("OPA_URL", "ftp://opa")
    with pytest.raises(OPAConfigurationError, match="HTTP"):
        OPAMiddleware(_ok)

    monkeypatch.setenv("OPA_ENABLED", "false")
    disabled = OPAMiddleware(_ok)
    assert asyncio.run(_asgi(disabled, "/api/oil-spill/incidents")) == 204

    monkeypatch.setenv("OPA_ENABLED", "true")
    monkeypatch.setenv("OPA_URL", "http://opa")
    allowed = OPAMiddleware(_ok)
    allowed._evaluate = lambda payload: {"allow": payload["subject"]["mfa_verified"] is True}
    status = asyncio.run(_asgi(allowed, "/api/oil-spill/incidents", state={"user": {"id": "reviewer", "roles": ["oil_spill_reviewer"], "mfa_verified": True, "project_ids": []}}))
    assert status == 204


def test_auth_branch_optional_current_user_and_valid_bearer(monkeypatch):
    assert asyncio.run(auth.get_current_user(None)) is None
    valid_user = auth.TokenPayload("member", "member", "member@example.test", "user", datetime.utcnow() + timedelta(hours=1))
    monkeypatch.setattr(auth, "decode_token", lambda _token: valid_user)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid")
    assert asyncio.run(auth.get_current_user(credentials)) is valid_user


def test_auth_branch_opa_http_decision_and_malformed_document(monkeypatch):
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("OPA_ENABLED", "true")
    monkeypatch.setenv("OPA_URL", "http://opa")
    middleware = OPAMiddleware(_ok)

    class Response:
        def __init__(self, document):
            self.document = document

        def read(self):
            return self.document

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("api.security.opa.urlopen", lambda _request, timeout: Response(b'{"result": {"allow": true}}'))
    assert middleware._evaluate({"subject": {}}) == {"allow": True}
    monkeypatch.setattr("api.security.opa.urlopen", lambda _request, timeout: Response(b'{"not_result": true}'))
    with pytest.raises(ValueError, match="decision"):
        middleware._evaluate({"subject": {}})
