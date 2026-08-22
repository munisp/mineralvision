from datetime import timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.innovations.integration_hub.audit_crypto import (
    append_signed_audit_event,
    build_connector_envelope,
    event_to_export_record,
    jsonl_export,
    public_key_to_b64,
    verify_audit_chain,
    verify_connector_envelope,
)
from src.api.innovations.integration_hub.db import Base
from src.api.innovations.integration_hub.models import SignedAuditEventModel


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture()
def signing_key():
    return Ed25519PrivateKey.generate()


def test_signed_connector_envelope_detects_tenant_payload_and_replay_tampering(signing_key):
    signed = build_connector_envelope(
        tenant_id="tenant-a",
        connector_id="arcgis-prod-1",
        event_type="candidate.created",
        payload={"evidence_id": "ev_1", "grade_ppm": 1.2},
        key_id="connector-2026-q3",
        private_key=signing_key,
        nonce="replay-test-nonce",
    )
    keys = {"connector-2026-q3": signing_key.public_key()}
    seen = set()
    verified = verify_connector_envelope(
        signed,
        public_keys=keys,
        expected_tenant_id="tenant-a",
        max_age=timedelta(minutes=5),
        seen_nonces=seen,
    )
    assert verified["payload"]["evidence_id"] == "ev_1"
    assert public_key_to_b64(signing_key)

    with pytest.raises(ValueError, match="nonce"):
        verify_connector_envelope(
            signed,
            public_keys=keys,
            expected_tenant_id="tenant-a",
            seen_nonces=seen,
        )

    tampered = {"envelope": dict(signed["envelope"]), "signature_b64": signed["signature_b64"]}
    tampered["envelope"]["payload"] = {"evidence_id": "ev_1", "grade_ppm": 99.9}
    with pytest.raises(ValueError, match="payload hash mismatch"):
        verify_connector_envelope(tampered, public_keys=keys, expected_tenant_id="tenant-a")


def test_append_only_chain_exports_for_offhost_verification(db_session, signing_key):
    events = [
        append_signed_audit_event(
            db_session,
            tenant_id="tenant-a",
            stream_id="connector:arcgis-prod-1",
            event_type="connector.envelope_accepted",
            actor_id="svc-arcgis",
            payload={"evidence_id": "ev_1"},
            key_id="audit-2026-q3",
            private_key=signing_key,
        ),
        append_signed_audit_event(
            db_session,
            tenant_id="tenant-a",
            stream_id="connector:arcgis-prod-1",
            event_type="writeback.staged",
            actor_id="analyst-a",
            payload={"proposal_id": "wb_1"},
            key_id="audit-2026-q3",
            private_key=signing_key,
        ),
        append_signed_audit_event(
            db_session,
            tenant_id="tenant-a",
            stream_id="connector:arcgis-prod-1",
            event_type="writeback.approved",
            actor_id="reviewer-b",
            payload={"proposal_id": "wb_1", "mfa_verified": True},
            key_id="audit-2026-q3",
            private_key=signing_key,
        ),
    ]
    assert [event.sequence for event in events] == [1, 2, 3]
    assert events[1].previous_hash == events[0].event_hash

    records = [event_to_export_record(event) for event in events]
    result = verify_audit_chain(
        records,
        public_keys={"audit-2026-q3": signing_key.public_key()},
        expected_tenant_id="tenant-a",
        expected_stream_id="connector:arcgis-prod-1",
    )
    assert result.valid
    assert result.last_sequence == 3
    assert result.last_event_hash == events[-1].event_hash
    assert '"event_hash"' in jsonl_export(events)


def test_offhost_verifier_detects_payload_and_chain_link_tampering(db_session, signing_key):
    first = append_signed_audit_event(
        db_session,
        tenant_id="tenant-a",
        stream_id="stream-1",
        event_type="ingest.completed",
        actor_id="svc",
        payload={"source_ref": "object-1"},
        key_id="audit-key",
        private_key=signing_key,
    )
    second = append_signed_audit_event(
        db_session,
        tenant_id="tenant-a",
        stream_id="stream-1",
        event_type="review.completed",
        actor_id="reviewer",
        payload={"evidence_id": "ev_1"},
        key_id="audit-key",
        private_key=signing_key,
    )
    records = [event_to_export_record(first), event_to_export_record(second)]
    records[1]["event"]["payload"] = {"evidence_id": "ev_999"}
    records[1]["event"]["previous_hash"] = "0" * 64

    result = verify_audit_chain(records, public_keys={"audit-key": signing_key.public_key()})
    assert not result.valid
    assert {failure["reason"] for failure in result.failures} >= {
        "previous hash mismatch", "event hash mismatch", "signature mismatch"
    }


def test_audit_events_are_persisted_as_signed_rows(db_session, signing_key):
    event = append_signed_audit_event(
        db_session,
        tenant_id="tenant-a",
        stream_id="stream-1",
        event_type="rotation.checked",
        actor_id="auditor",
        payload={"key_id": "audit-key"},
        key_id="audit-key",
        private_key=signing_key,
    )
    stored = db_session.query(SignedAuditEventModel).filter_by(event_id=event.event_id).one()
    assert stored.signature_b64 == event.signature_b64
    assert stored.event_hash == event.event_hash
