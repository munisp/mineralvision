#!/usr/bin/env python3
"""Run a staging-only Ed25519 signing-key compromise and rotation drill.

The drill uses ephemeral in-memory keys and a unique synthetic tenant. It never
contacts a KMS/HSM, production collector, payment rail, or external connector.
It proves application-level containment and verification behavior only.

Required environment:
  ENV=staging
  MV_AUDIT_DRILL_CONFIRM=ROTATE_COMPROMISED_KEY
  MV_AUDIT_DRILL_DATABASE_URL=postgresql+psycopg2://...
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package"))

from src.api.innovations.integration_hub.audit_crypto import (  # noqa: E402
    append_signed_audit_event,
    build_connector_envelope,
    event_to_export_record,
    jsonl_export,
    verify_audit_chain,
    verify_connector_envelope,
)
from src.api.innovations.integration_hub.db import Base  # noqa: E402
from src.api.innovations.integration_hub.models import (  # noqa: E402
    AuditStreamModel,
    SignedAuditEventModel,
)


def require_staging() -> str:
    if os.getenv("ENV", "").lower() != "staging":
        raise RuntimeError("refusing key-compromise drill outside ENV=staging")
    if os.getenv("MV_AUDIT_DRILL_CONFIRM") != "ROTATE_COMPROMISED_KEY":
        raise RuntimeError("set MV_AUDIT_DRILL_CONFIRM=ROTATE_COMPROMISED_KEY to run")
    url = os.getenv("MV_AUDIT_DRILL_DATABASE_URL", "")
    if not url.startswith("postgresql"):
        raise RuntimeError("MV_AUDIT_DRILL_DATABASE_URL must be a staging PostgreSQL URL")
    return url


def cleanup(session_factory: Any, tenant_id: str) -> None:
    session = session_factory()
    try:
        session.query(SignedAuditEventModel).filter(SignedAuditEventModel.tenant_id == tenant_id).delete(
            synchronize_session=False
        )
        session.query(AuditStreamModel).filter(AuditStreamModel.tenant_id == tenant_id).delete(
            synchronize_session=False
        )
        session.commit()
    finally:
        session.close()


def main() -> int:
    database_url = require_staging()
    keep_data = os.getenv("MV_AUDIT_DRILL_KEEP_DATA", "false").lower() == "true"
    engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    tenant_id = f"staging-drill-{uuid.uuid4().hex}"
    stream_id = "connector:staging-drill"
    old_key_id = "staging-old-compromised"
    new_key_id = "staging-new-rotated"
    old_private = Ed25519PrivateKey.generate()
    new_private = Ed25519PrivateKey.generate()
    historical_keys = {old_key_id: old_private.public_key(), new_key_id: new_private.public_key()}
    active_keys = {old_key_id: old_private.public_key()}
    evidence: dict[str, Any] = {"tenant_id": tenant_id, "stream_id": stream_id, "steps": []}

    try:
        # 1. Establish known-good old-key evidence before simulated compromise.
        old_envelope = build_connector_envelope(
            tenant_id=tenant_id,
            connector_id="staging-connector",
            event_type="evidence.registered",
            payload={"evidence_id": "ev-before-compromise"},
            key_id=old_key_id,
            private_key=old_private,
            nonce="old-before",
        )
        verify_connector_envelope(
            old_envelope,
            public_keys=active_keys,
            expected_tenant_id=tenant_id,
            max_age=timedelta(minutes=5),
            seen_nonces=set(),
        )
        session = session_factory()
        try:
            before = append_signed_audit_event(
                session,
                tenant_id=tenant_id,
                stream_id=stream_id,
                event_type="connector.envelope_accepted",
                actor_id="svc-staging",
                payload={"phase": "pre_compromise", "key_id": old_key_id},
                key_id=old_key_id,
                private_key=old_private,
            )
        finally:
            session.close()
        evidence["steps"].append({"step": "old_key_pre_compromise_accepted", "event_id": before.event_id})

        # 2. Simulate containment. Active verification registry removes old key,
        # while historical registry retains it for evidence verification.
        active_keys.pop(old_key_id)
        try:
            verify_connector_envelope(
                old_envelope,
                public_keys=active_keys,
                expected_tenant_id=tenant_id,
            )
            raise AssertionError("compromised old key was unexpectedly accepted")
        except ValueError as exc:
            if "unknown connector signing key" not in str(exc):
                raise
        evidence["steps"].append({"step": "old_key_contained", "result": "rejected_for_new_envelopes"})

        # 3. Controlled active-key change by the privileged rotation job.
        session = session_factory()
        try:
            stream = (
                session.query(AuditStreamModel)
                .filter(AuditStreamModel.tenant_id == tenant_id, AuditStreamModel.stream_id == stream_id)
                .with_for_update()
                .one()
            )
            old_anchor = stream.last_event_hash
            stream.active_key_id = new_key_id
            session.commit()
        finally:
            session.close()
        active_keys[new_key_id] = new_private.public_key()
        evidence["steps"].append({"step": "stream_key_rotated", "old_anchor": old_anchor, "new_key_id": new_key_id})

        # 4. New key signs the activation event, chained to old-key evidence.
        session = session_factory()
        try:
            rotated = append_signed_audit_event(
                session,
                tenant_id=tenant_id,
                stream_id=stream_id,
                event_type="key.rotation.activated",
                actor_id="rotation-job",
                payload={
                    "incident_type": "simulated_compromise",
                    "old_key_id": old_key_id,
                    "new_key_id": new_key_id,
                    "old_anchor": old_anchor,
                },
                key_id=new_key_id,
                private_key=new_private,
            )
            accepted = append_signed_audit_event(
                session,
                tenant_id=tenant_id,
                stream_id=stream_id,
                event_type="connector.envelope_accepted",
                actor_id="svc-staging",
                payload={"phase": "post_rotation", "key_id": new_key_id},
                key_id=new_key_id,
                private_key=new_private,
            )
            events = (
                session.query(SignedAuditEventModel)
                .filter(SignedAuditEventModel.tenant_id == tenant_id, SignedAuditEventModel.stream_id == stream_id)
                .order_by(SignedAuditEventModel.sequence)
                .all()
            )
        finally:
            session.close()
        evidence["steps"].append({"step": "new_key_post_rotation_accepted", "event_id": accepted.event_id})

        # 5. Verify complete evidence with the historical public-key registry.
        records = [event_to_export_record(event) for event in events]
        verification = verify_audit_chain(
            records,
            public_keys=historical_keys,
            expected_tenant_id=tenant_id,
            expected_stream_id=stream_id,
        )
        if not verification.valid:
            raise AssertionError(f"audit chain verification failed: {verification.failures}")
        evidence["steps"].append(
            {"step": "full_chain_verified", "event_count": verification.event_count, "last_hash": verification.last_event_hash}
        )

        # 6. Preserve an off-host-style JSON Lines evidence export in a temporary
        # directory for drill inspection. Real staging must send this to its
        # independently governed collector through mTLS.
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as handle:
            handle.write(jsonl_export(events))
            export_path = handle.name
        evidence["offhost_bundle_path"] = export_path
        evidence["result"] = "passed"
        print(json.dumps(evidence, sort_keys=True, indent=2))
        return 0
    finally:
        if not keep_data:
            cleanup(session_factory, tenant_id)
        engine.dispose()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"result": "failed", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        raise SystemExit(2)
