"""Tamper-evident audit streams and Ed25519-signed connector envelopes.

This module provides cryptographic integrity and origin authentication for a
configured signing key. It is deliberately independent of the application's
private key store: production callers obtain signing keys from an approved
KMS/HSM/workload-identity integration and pass only key material or signing
callbacks at the trust boundary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import AuditStreamModel, SignedAuditEventModel

SCHEMA_VERSION = 1
GENESIS_HASH = ""


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Canonical UTF-8 JSON bytes used for every hash and signature payload."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def sha256_hex(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def private_key_from_b64(encoded: str) -> Ed25519PrivateKey:
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != 32:
        raise ValueError("Ed25519 private keys must be 32 raw bytes")
    return Ed25519PrivateKey.from_private_bytes(raw)


def public_key_from_b64(encoded: str) -> Ed25519PublicKey:
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != 32:
        raise ValueError("Ed25519 public keys must be 32 raw bytes")
    return Ed25519PublicKey.from_public_bytes(raw)


def public_key_to_b64(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:
    if isinstance(key, Ed25519PrivateKey):
        key = key.public_key()
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def sign_b64(private_key: Ed25519PrivateKey, body: Mapping[str, Any]) -> str:
    return base64.b64encode(private_key.sign(canonical_json(body))).decode("ascii")


def verify_b64(public_key: Ed25519PublicKey, body: Mapping[str, Any], signature_b64: str) -> bool:
    try:
        public_key.verify(base64.b64decode(signature_b64, validate=True), canonical_json(body))
        return True
    except (InvalidSignature, ValueError):
        return False


def build_connector_envelope(
    *,
    tenant_id: str,
    connector_id: str,
    event_type: str,
    payload: Dict[str, Any],
    key_id: str,
    private_key: Ed25519PrivateKey,
    issued_at: Optional[datetime] = None,
    nonce: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an Ed25519-signed, replay-addressable connector envelope."""
    if not all([tenant_id, connector_id, event_type, key_id]):
        raise ValueError("tenant_id, connector_id, event_type, and key_id are required")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    issued_at = _as_utc_naive(issued_at or utcnow())
    body = {
        "schema_version": SCHEMA_VERSION,
        "tenant_id": tenant_id,
        "connector_id": connector_id,
        "event_type": event_type,
        "issued_at": issued_at.isoformat(timespec="microseconds") + "Z",
        "nonce": nonce or uuid.uuid4().hex,
        "key_id": key_id,
        "payload_hash": sha256_hex(payload),
        "payload": payload,
    }
    return {"envelope": body, "signature_b64": sign_b64(private_key, body)}


def verify_connector_envelope(
    signed: Mapping[str, Any],
    *,
    public_keys: Mapping[str, Ed25519PublicKey],
    expected_tenant_id: Optional[str] = None,
    max_age: Optional[timedelta] = None,
    now: Optional[datetime] = None,
    seen_nonces: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Verify signature, payload commitment, tenant, age, and optional replay set."""
    envelope = signed.get("envelope")
    signature = signed.get("signature_b64")
    if not isinstance(envelope, dict) or not isinstance(signature, str):
        raise ValueError("signed connector envelope must contain envelope and signature_b64")
    required = {
        "schema_version", "tenant_id", "connector_id", "event_type", "issued_at",
        "nonce", "key_id", "payload_hash", "payload",
    }
    missing = required - set(envelope)
    if missing:
        raise ValueError(f"connector envelope missing fields: {sorted(missing)}")
    if envelope["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported connector envelope schema version")
    if not isinstance(envelope["payload"], dict) or sha256_hex(envelope["payload"]) != envelope["payload_hash"]:
        raise ValueError("connector payload hash mismatch")
    if expected_tenant_id and envelope["tenant_id"] != expected_tenant_id:
        raise ValueError("connector envelope tenant mismatch")
    key = public_keys.get(envelope["key_id"])
    if key is None:
        raise ValueError("unknown connector signing key")
    if not verify_b64(key, envelope, signature):
        raise ValueError("connector envelope signature is invalid")
    if max_age is not None:
        try:
            issued = datetime.fromisoformat(envelope["issued_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("connector envelope issued_at is invalid") from exc
        if (_as_utc_naive(now or utcnow()) - _as_utc_naive(issued)) > max_age:
            raise ValueError("connector envelope has expired")
    if seen_nonces is not None:
        if envelope["nonce"] in seen_nonces:
            raise ValueError("connector envelope nonce was already accepted")
        seen_nonces.add(envelope["nonce"])
    return dict(envelope)


def _event_body(
    *,
    tenant_id: str,
    stream_id: str,
    sequence: int,
    previous_hash: str,
    event_type: str,
    actor_id: str,
    payload: Dict[str, Any],
    occurred_at: datetime,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tenant_id": tenant_id,
        "stream_id": stream_id,
        "sequence": sequence,
        "previous_hash": previous_hash,
        "event_type": event_type,
        "actor_id": actor_id,
        "payload": payload,
        "occurred_at": _as_utc_naive(occurred_at).isoformat(timespec="microseconds") + "Z",
    }


def _event_commitment(event_body: Mapping[str, Any], event_hash: str, key_id: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tenant_id": event_body["tenant_id"],
        "stream_id": event_body["stream_id"],
        "sequence": event_body["sequence"],
        "previous_hash": event_body["previous_hash"],
        "event_hash": event_hash,
        "key_id": key_id,
        "occurred_at": event_body["occurred_at"],
    }


def append_signed_audit_event(
    db: Session,
    *,
    tenant_id: str,
    stream_id: str,
    event_type: str,
    actor_id: str,
    payload: Dict[str, Any],
    key_id: str,
    private_key: Ed25519PrivateKey,
    occurred_at: Optional[datetime] = None,
) -> SignedAuditEventModel:
    """Append one signed event using a row lock on the stream anchor.

    Callers must never update/delete returned events. PostgreSQL deployments
    should use a database role that grants the application INSERT/SELECT but
    not UPDATE/DELETE on the signed-event table.
    """
    if not all([tenant_id, stream_id, event_type, actor_id, key_id]):
        raise ValueError("tenant_id, stream_id, event_type, actor_id, and key_id are required")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    occurred_at = _as_utc_naive(occurred_at or utcnow())

    stream = (
        db.query(AuditStreamModel)
        .filter(AuditStreamModel.tenant_id == tenant_id, AuditStreamModel.stream_id == stream_id)
        .with_for_update()
        .first()
    )
    if stream is None:
        stream = AuditStreamModel(
            tenant_id=tenant_id,
            stream_id=stream_id,
            last_sequence=0,
            last_event_hash=GENESIS_HASH,
            active_key_id=key_id,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        db.add(stream)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            stream = (
                db.query(AuditStreamModel)
                .filter(AuditStreamModel.tenant_id == tenant_id, AuditStreamModel.stream_id == stream_id)
                .with_for_update()
                .one()
            )
    if stream.active_key_id != key_id:
        raise ValueError("active audit stream key differs; rotate the stream key explicitly")

    sequence = stream.last_sequence + 1
    event_body = _event_body(
        tenant_id=tenant_id,
        stream_id=stream_id,
        sequence=sequence,
        previous_hash=stream.last_event_hash,
        event_type=event_type,
        actor_id=actor_id,
        payload=payload,
        occurred_at=occurred_at,
    )
    event_hash = sha256_hex(event_body)
    commitment = _event_commitment(event_body, event_hash, key_id)
    event = SignedAuditEventModel(
        event_id=f"ae_{uuid.uuid4().hex}",
        tenant_id=tenant_id,
        stream_id=stream_id,
        sequence=sequence,
        previous_hash=stream.last_event_hash,
        event_hash=event_hash,
        key_id=key_id,
        signature_b64=sign_b64(private_key, commitment),
        event_type=event_type,
        actor_id=actor_id,
        payload=payload,
        occurred_at=occurred_at,
        created_at=utcnow(),
    )
    db.add(event)
    stream.last_sequence = sequence
    stream.last_event_hash = event_hash
    stream.updated_at = utcnow()
    db.commit()
    db.refresh(event)
    return event


def event_to_export_record(event: SignedAuditEventModel) -> Dict[str, Any]:
    """Return a self-contained signed record suitable for JSON Lines export."""
    body = _event_body(
        tenant_id=event.tenant_id,
        stream_id=event.stream_id,
        sequence=event.sequence,
        previous_hash=event.previous_hash,
        event_type=event.event_type,
        actor_id=event.actor_id,
        payload=event.payload,
        occurred_at=event.occurred_at,
    )
    return {
        "event_id": event.event_id,
        "event": body,
        "event_hash": event.event_hash,
        "key_id": event.key_id,
        "signature_b64": event.signature_b64,
        "created_at": event.created_at.isoformat(timespec="microseconds") + "Z",
    }


@dataclass(frozen=True)
class ChainVerificationResult:
    valid: bool
    event_count: int
    last_sequence: int
    last_event_hash: str
    failures: Sequence[Dict[str, Any]]


def verify_audit_chain(
    records: Iterable[Mapping[str, Any]],
    *,
    public_keys: Mapping[str, Ed25519PublicKey],
    expected_tenant_id: Optional[str] = None,
    expected_stream_id: Optional[str] = None,
    expected_prior_hash: str = GENESIS_HASH,
    expected_start_sequence: int = 1,
) -> ChainVerificationResult:
    """Verify JSON Lines-style records independently of the source database."""
    failures: list[Dict[str, Any]] = []
    prior_hash = expected_prior_hash
    expected_sequence = expected_start_sequence
    last_sequence = expected_start_sequence - 1
    count = 0

    for record in records:
        count += 1
        event = record.get("event")
        event_id = record.get("event_id", f"index:{count}")
        if not isinstance(event, dict):
            failures.append({"event_id": event_id, "reason": "event payload missing"})
            continue
        if expected_tenant_id and event.get("tenant_id") != expected_tenant_id:
            failures.append({"event_id": event_id, "reason": "tenant mismatch"})
        if expected_stream_id and event.get("stream_id") != expected_stream_id:
            failures.append({"event_id": event_id, "reason": "stream mismatch"})
        if event.get("sequence") != expected_sequence:
            failures.append({"event_id": event_id, "reason": "sequence discontinuity"})
        if event.get("previous_hash") != prior_hash:
            failures.append({"event_id": event_id, "reason": "previous hash mismatch"})
        calculated = sha256_hex(event)
        if calculated != record.get("event_hash"):
            failures.append({"event_id": event_id, "reason": "event hash mismatch"})
        key_id = record.get("key_id")
        key = public_keys.get(key_id)
        if key is None:
            failures.append({"event_id": event_id, "reason": "unknown signing key"})
        else:
            commitment = _event_commitment(event, record.get("event_hash", ""), key_id)
            if not verify_b64(key, commitment, record.get("signature_b64", "")):
                failures.append({"event_id": event_id, "reason": "signature mismatch"})
        prior_hash = str(record.get("event_hash", ""))
        last_sequence = event.get("sequence", last_sequence)
        expected_sequence += 1

    return ChainVerificationResult(
        valid=not failures,
        event_count=count,
        last_sequence=last_sequence,
        last_event_hash=prior_hash,
        failures=tuple(failures),
    )


def jsonl_export(events: Iterable[SignedAuditEventModel]) -> str:
    """Serialize signed events as newline-delimited canonical JSON records."""
    return "\n".join(
        canonical_json(event_to_export_record(event)).decode("utf-8") for event in events
    ) + "\n"
