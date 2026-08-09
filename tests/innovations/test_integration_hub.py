"""Deterministic tests for the integration_hub innovation (B5-20)."""

import json

import bcrypt
import httpx
import pytest
from api.innovations.integration_hub import router
from api.innovations.integration_hub.db import Base, get_db
from api.innovations.integration_hub.logic import (
    WebhookRegistry,
    api_key_to_dict,
    authenticate_key,
    build_api_key_dependency,
    canonical_body,
    create_api_key,
    sign_payload,
    verify_signature,
)
from api.innovations.integration_hub.models import ApiKeyModel
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = factory()
    yield db
    db.close()
    engine.dispose()


class TestSigning:
    def test_signature_matches_manual_hmac(self):
        import hashlib
        import hmac

        body = canonical_body({"b": 2, "a": 1})
        assert body == b'{"a":1,"b":2}'
        expected = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        assert sign_payload("s3cret", body) == expected
        assert verify_signature("s3cret", body, expected) is True
        assert verify_signature("s3cret", body, "sha256=" + "0" * 64) is False
        assert verify_signature("wrong", body, expected) is False


class TestDelivery:
    def _registry(self, session, handler):
        transport = httpx.MockTransport(handler)
        return WebhookRegistry(session, transport=transport, sleep=lambda _s: None)

    def test_successful_delivery_signed_and_logged(self, session):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content
            captured["sig"] = request.headers.get("X-MV-Signature")
            captured["topic"] = request.headers.get("X-MV-Topic")
            return httpx.Response(200, json={"ok": True})

        registry = self._registry(session, handler)
        webhook = registry.subscribe("https://receiver.test/hook", ["assay.result"], name="lab")
        [delivery] = registry.publish("assay.result", {"batch": "B1", "au_ppm": 2.4})

        # receiver can verify the signature over the exact bytes received
        assert verify_signature(webhook.secret, captured["body"], captured["sig"]) is True
        assert json.loads(captured["body"]) == {"batch": "B1", "au_ppm": 2.4}
        assert captured["topic"] == "assay.result"
        assert delivery.status == "success" and delivery.attempts == 1
        assert delivery.last_status_code == 200 and delivery.error == ""
        assert delivery.signature == captured["sig"]

    def test_retry_on_500_then_success_recorded(self, session):
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500 if calls["n"] == 1 else 200)

        registry = self._registry(session, handler)
        registry.subscribe("https://receiver.test/hook", ["drill.complete"])
        [delivery] = registry.publish("drill.complete", {"hole": "RC001"})

        assert calls["n"] == 2  # one retry
        assert delivery.status == "success"
        assert delivery.attempts == 2
        assert delivery.last_status_code == 200

    def test_persistent_failure_logged_with_backoff_schedule(self, session):
        sleeps = []

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        transport = httpx.MockTransport(handler)
        registry = WebhookRegistry(session, transport=transport,
                                   sleep=sleeps.append, backoff_base=0.5)
        registry.subscribe("https://receiver.test/hook", ["t"])
        [delivery] = registry.publish("t", {})

        assert delivery.status == "failed"
        assert delivery.attempts == 3  # MAX_ATTEMPTS
        assert delivery.last_status_code == 503
        assert delivery.error == "HTTP 503"
        assert sleeps == [0.5, 1.0]  # exponential backoff base*2^n

    def test_topic_filtering_and_network_error(self, session):
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        registry = self._registry(session, handler)
        registry.subscribe("https://a.test/hook", ["geology"])
        registry.subscribe("https://b.test/hook", ["assay"])
        deliveries = registry.publish("geology", {"x": 1})
        assert len(deliveries) == 1  # only the geology subscriber
        assert deliveries[0].status == "failed"
        assert deliveries[0].last_status_code is None
        assert "connection refused" in deliveries[0].error


class TestApiKeys:
    def test_created_key_authenticates_and_hash_not_plaintext(self, session):
        created = create_api_key(session, "field-tablet", ["read"])
        assert created["key"].startswith(f"mvk_{created['key_id']}.")
        record = session.query(ApiKeyModel).filter_by(key_id=created["key_id"]).one()
        assert created["key"] not in record.key_hash  # never stored in plaintext
        assert bcrypt.checkpw(created["key"].encode(), record.key_hash.encode())
        assert authenticate_key(session, created["key"]).id == record.id

    def test_wrong_key_rejected(self, session):
        created = create_api_key(session, "k", ["read"])
        assert authenticate_key(session, created["key"] + "tampered") is None
        assert authenticate_key(session, "mvk_deadbeef.nope") is None
        assert authenticate_key(session, "not-a-key") is None

    def test_scope_validation(self, session):
        with pytest.raises(ValueError):
            create_api_key(session, "bad", ["admin"])


class TestApiKeyDependency:
    @pytest.fixture()
    def client(self, session):
        app = FastAPI()
        app.include_router(router)

        def override_db():
            yield session

        app.dependency_overrides[get_db] = override_db

        # probe routes protected by the real dependency factory
        read_dep = build_api_key_dependency(get_db, "read")
        write_dep = build_api_key_dependency(get_db, "write")

        @app.get("/probe/read")
        def probe_read(key: ApiKeyModel = Depends(read_dep)):  # noqa: B008
            return {"ok": True, "key": api_key_to_dict(key)}

        @app.post("/probe/write")
        def probe_write(key: ApiKeyModel = Depends(write_dep)):  # noqa: B008
            return {"ok": True}

        return TestClient(app)

    def test_scoped_enforcement_401_and_403(self, session, client):
        ro = create_api_key(session, "reader", ["read"])
        rw = create_api_key(session, "writer", ["read", "write"])

        assert client.get("/probe/read").status_code == 401  # no header
        assert client.get("/probe/read", headers={"X-API-Key": "mvk_00000000.bad"}).status_code == 401

        ok = client.get("/probe/read", headers={"X-API-Key": ro["key"]})
        assert ok.status_code == 200 and ok.json()["key"]["scopes"] == ["read"]

        # read-only key forbidden on write-protected route
        assert client.post("/probe/write", headers={"X-API-Key": ro["key"]}).status_code == 403
        assert client.post("/probe/write", headers={"X-API-Key": rw["key"]}).status_code == 200

    def test_deactivated_key_rejected(self, session, client):
        key = create_api_key(session, "temp", ["read"])
        record = session.query(ApiKeyModel).filter_by(key_id=key["key_id"]).one()
        record.active = False
        session.commit()
        assert client.get("/probe/read", headers={"X-API-Key": key["key"]}).status_code == 401


class TestHubAPI:
    @pytest.fixture()
    def client(self, session):
        app = FastAPI()
        app.include_router(router)

        def override_db():
            yield session

        app.dependency_overrides[get_db] = override_db
        return TestClient(app)

    def test_webhook_crud_requires_write_key(self, session, client):
        rw = create_api_key(session, "admin-integration", ["read", "write"])
        headers = {"X-API-Key": rw["key"]}

        assert client.post("/innovations/integration_hub/webhooks",
                           json={"url": "https://x.test/h", "topics": ["t"]}).status_code == 401

        resp = client.post("/innovations/integration_hub/webhooks",
                           json={"url": "https://x.test/h", "topics": ["t", "u"], "name": "hook"},
                           headers=headers)
        assert resp.status_code == 201
        assert "secret" in resp.json()  # returned once

        listed = client.get("/innovations/integration_hub/webhooks", headers=headers).json()
        assert listed["count"] == 1
        assert "secret" not in listed["webhooks"][0]  # masked thereafter

        wid = listed["webhooks"][0]["id"]
        assert client.delete(f"/innovations/integration_hub/webhooks/{wid}", headers=headers).json() == {"deleted": wid}
        assert client.delete(f"/innovations/integration_hub/webhooks/{wid}", headers=headers).status_code == 404
