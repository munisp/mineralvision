"""Webhook delivery + scoped API key logic.

Webhooks
  - payloads are serialized canonically (sorted keys) and signed with
    HMAC-SHA256 using the subscription secret; the signature travels in the
    ``X-MV-Signature: sha256=<hex>`` header;
  - delivery retries with exponential backoff (base * 2**attempt, capped)
    in-process; transport and sleep are injectable for deterministic tests;
  - every delivery is logged (status, attempts, last HTTP status, error).

API keys
  - generated as ``mvk_<key_id>.<secret>``; only a bcrypt hash of the full
    key is stored; lookup happens via the public ``key_id`` segment;
  - ``require_api_key(scope)`` is a FastAPI dependency factory (mirrors the
    platform's require_role pattern) enforcing 401 (unknown/inactive key)
    and 403 (valid key, missing scope).
"""

import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import bcrypt
import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from .models import ApiKeyModel, DeliveryModel, WebhookModel, _utcnow

SIGNATURE_HEADER = "X-MV-Signature"
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5
BACKOFF_CAP_SECONDS = 30.0

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------

def canonical_body(payload: Dict[str, Any]) -> bytes:
    """Canonical payload bytes — exactly what is sent and signed."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_payload(secret: str, body: bytes) -> str:
    """``sha256=<hmac-sha256 hex>`` signature string for the header."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(secret: str, body: bytes, signature: str) -> bool:
    """Receiver-side verification helper (constant-time compare)."""
    return hmac.compare_digest(sign_payload(secret, body), signature)


def _default_sleep(seconds: float) -> None:
    time.sleep(seconds)


class WebhookRegistry:
    def __init__(
        self,
        db: Session,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = _default_sleep,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_base: float = BACKOFF_BASE_SECONDS,
    ):
        self.db = db
        self.transport = transport
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base

    # ------------------------------------------------------------- registry
    def subscribe(self, url: str, topics: List[str], name: str = "") -> WebhookModel:
        if not topics:
            raise ValueError("at least one topic is required")
        webhook = WebhookModel(
            url=url, secret=secrets.token_hex(16), topics=sorted(set(topics)),
            name=name, active=True, created_at=_utcnow(),
        )
        self.db.add(webhook)
        self.db.commit()
        self.db.refresh(webhook)
        return webhook

    def list_webhooks(self) -> List[WebhookModel]:
        return self.db.query(WebhookModel).order_by(WebhookModel.id).all()

    def unsubscribe(self, webhook_id: int) -> bool:
        webhook = self.db.get(WebhookModel, webhook_id)
        if webhook is None:
            return False
        self.db.delete(webhook)
        self.db.commit()
        return True

    def subscribers(self, topic: str) -> List[WebhookModel]:
        return [
            w for w in self.db.query(WebhookModel).filter(WebhookModel.active.is_(True)).all()
            if topic in (w.topics or [])
        ]

    # ------------------------------------------------------------- delivery
    def deliver(self, webhook: WebhookModel, topic: str, payload: Dict[str, Any]) -> DeliveryModel:
        """Deliver with retry + exponential backoff; always logged."""
        body = canonical_body(payload)
        signature = sign_payload(webhook.secret, body)
        headers = {
            "Content-Type": "application/json",
            SIGNATURE_HEADER: signature,
            "X-MV-Topic": topic,
        }
        attempts = 0
        last_status: Optional[int] = None
        error = ""
        with httpx.Client(transport=self.transport, timeout=10.0) as client:
            while attempts < self.max_attempts:
                if attempts:
                    self.sleep(min(self.backoff_base * (2 ** (attempts - 1)), BACKOFF_CAP_SECONDS))
                attempts += 1
                try:
                    response = client.post(webhook.url, content=body, headers=headers)
                    last_status = response.status_code
                    if response.status_code < 400:
                        error = ""
                        break
                    error = f"HTTP {response.status_code}"
                except httpx.HTTPError as exc:  # network-level failure → retry
                    last_status = None
                    error = str(exc)
        succeeded = last_status is not None and last_status < 400
        delivery = DeliveryModel(
            webhook_id=webhook.id, topic=topic, payload=payload,
            status="success" if succeeded else "failed",
            attempts=attempts, last_status_code=last_status,
            signature=signature, error="" if succeeded else error,
            created_at=_utcnow(),
        )
        self.db.add(delivery)
        self.db.commit()
        self.db.refresh(delivery)
        return delivery

    def publish(self, topic: str, payload: Dict[str, Any]) -> List[DeliveryModel]:
        """Fan an event out to all active subscribers of the topic."""
        return [self.deliver(w, topic, payload) for w in self.subscribers(topic)]

    def list_deliveries(self, webhook_id: Optional[int] = None) -> List[DeliveryModel]:
        query = self.db.query(DeliveryModel)
        if webhook_id is not None:
            query = query.filter(DeliveryModel.webhook_id == webhook_id)
        return query.order_by(DeliveryModel.id).all()


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

def create_api_key(db: Session, name: str, scopes: List[str]) -> Dict[str, Any]:
    """Create a scoped key. The plaintext key is returned ONCE; only a
    bcrypt hash of the full key is persisted."""
    if not scopes:
        raise ValueError("at least one scope is required")
    invalid = set(scopes) - {"read", "write"}
    if invalid:
        raise ValueError(f"unknown scopes {sorted(invalid)}; allowed: read, write")
    key_id = secrets.token_hex(4)  # 8-char public id
    secret = secrets.token_urlsafe(24)
    full_key = f"mvk_{key_id}.{secret}"
    record = ApiKeyModel(
        key_id=key_id, key_hash=bcrypt.hashpw(full_key.encode(), bcrypt.gensalt()).decode(),
        name=name, scopes=sorted(set(scopes)), active=True, created_at=_utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"key": full_key, "key_id": key_id, "name": name, "scopes": record.scopes}


def authenticate_key(db: Session, presented_key: str) -> Optional[ApiKeyModel]:
    """Look up by key_id segment, then bcrypt-verify the full key."""
    if not presented_key.startswith("mvk_") or "." not in presented_key:
        return None
    key_id = presented_key[4:].split(".", 1)[0]
    record = (
        db.query(ApiKeyModel)
        .filter(ApiKeyModel.key_id == key_id, ApiKeyModel.active.is_(True))
        .first()
    )
    if record is None:
        return None
    try:
        if bcrypt.checkpw(presented_key.encode(), record.key_hash.encode()):
            return record
    except ValueError:
        return None
    return None


def build_api_key_dependency(get_db_dep, required_scope: str):
    """Construct the concrete dependency bound to a session provider.

    ``get_db_dep`` is this module's ``get_db`` so tests can override it via
    ``app.dependency_overrides`` like any other dependency.
    """

    async def key_checker(
        presented: Optional[str] = Depends(_api_key_header),
        db: Session = Depends(get_db_dep),
    ) -> ApiKeyModel:
        if not presented:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required (X-API-Key header)",
            )
        record = authenticate_key(db, presented)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or inactive API key",
            )
        if required_scope not in (record.scopes or []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key lacks required scope '{required_scope}'",
            )
        return record

    return key_checker


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def webhook_to_dict(w: WebhookModel, include_secret: bool = False) -> Dict[str, Any]:
    out = {
        "id": w.id, "url": w.url, "topics": w.topics, "name": w.name,
        "active": w.active, "created_at": w.created_at.isoformat(),
    }
    if include_secret:
        out["secret"] = w.secret
    return out


def delivery_to_dict(d: DeliveryModel) -> Dict[str, Any]:
    return {
        "id": d.id, "webhook_id": d.webhook_id, "topic": d.topic, "payload": d.payload,
        "status": d.status, "attempts": d.attempts, "last_status_code": d.last_status_code,
        "signature": d.signature, "error": d.error, "created_at": d.created_at.isoformat(),
    }


def api_key_to_dict(k: ApiKeyModel) -> Dict[str, Any]:
    return {
        "id": k.id, "key_id": k.key_id, "name": k.name, "scopes": k.scopes,
        "active": k.active, "created_at": k.created_at.isoformat(),
    }
