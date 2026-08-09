"""Deterministic tests for the custody_ledger innovation (B4-13)."""

import hashlib
import hmac
import json
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.innovations.custody_ledger import router
from api.innovations.custody_ledger.db import Base, get_db
from api.innovations.custody_ledger.logic import (
    ENTITY_TYPES,
    GENESIS_HASH,
    CustodyLedger,
    canonical_json,
    compute_entry_hash,
    sign_entry,
)
from api.innovations.custody_ledger.models import CustodyLedgerEntry

TEST_KEY = b"test-ledger-hmac-key-0123456789abcdef"


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


@pytest.fixture()
def ledger(session):
    return CustodyLedger(session, hmac_key=TEST_KEY)


def _append_full_chain(ledger: CustodyLedger, entity_id: str = "BATCH-2025-001"):
    """Canonical custody lifecycle: batch → dispatch → receipt → results."""
    events = [
        ("sample_batch", "batch_created", "geologist.a", {"sample_count": 42, "project": "Ridge A"}),
        ("dispatch", "dispatch", "geologist.a", {"courier": "FastShip", "seal_numbers": ["S1", "S2"]}),
        ("lab_receipt", "lab_receipt", "labtech.b", {"received_count": 42, "seals_intact": True}),
        ("results", "results", "labtech.b", {"lab_job": "LJ-99", "mean_au_ppm": 1.87}),
    ]
    return [
        ledger.append(entity_id=entity_id, entity_type=et, event_type=ev, actor=actor, payload=pl)
        for et, ev, actor, pl in events
    ]


class TestHashMath:
    def test_hash_formula_exact(self):
        payload = {"b": 2, "a": 1}
        iso = "2025-01-01T00:00:00"
        expected_material = f"{GENESIS_HASH}{json.dumps(payload, sort_keys=True, separators=(',', ':'))}{iso}geologist.a"
        expected = hashlib.sha256(expected_material.encode()).hexdigest()
        assert compute_entry_hash(GENESIS_HASH, payload, iso, "geologist.a") == expected

    def test_canonical_json_key_order_invariant(self):
        assert canonical_json({"z": 1, "a": 2}) == canonical_json({"a": 2, "z": 1})

    def test_signature_is_hmac_sha256(self):
        entry_hash = compute_entry_hash(GENESIS_HASH, {"x": 1}, "2025-01-01T00:00:00", "actor")
        expected = hmac.new(TEST_KEY, entry_hash.encode(), hashlib.sha256).hexdigest()
        assert sign_entry(entry_hash, TEST_KEY) == expected


class TestAppendAndChain:
    def test_chain_links_and_monotonic_sequence(self, ledger):
        entries = _append_full_chain(ledger)
        assert [e.id for e in entries] == [1, 2, 3, 4]
        assert entries[0].prev_hash == GENESIS_HASH
        for prev, cur in zip(entries, entries[1:]):
            assert cur.prev_hash == prev.entry_hash

    def test_stored_hash_matches_formula(self, ledger):
        entry = ledger.append(
            entity_id="B1", entity_type="sample_batch", event_type="batch_created",
            actor="geo", payload={"n": 1}, timestamp=datetime(2025, 3, 1, 12, 0, 0),
        )
        recomputed = compute_entry_hash(GENESIS_HASH, {"n": 1}, "2025-03-01T12:00:00", "geo")
        assert entry.entry_hash == recomputed
        assert entry.signature == sign_entry(recomputed, TEST_KEY)

    def test_invalid_event_and_entity_types_rejected(self, ledger):
        with pytest.raises(ValueError):
            ledger.append("B1", "assay_batch", "batch_created", "geo", {})
        with pytest.raises(ValueError):
            ledger.append("B1", "sample_batch", "planted", "geo", {})
        with pytest.raises(ValueError):
            ledger.append("B1", "sample_batch", "batch_created", "", {})
        assert set(ENTITY_TYPES) == {"sample_batch", "dispatch", "lab_receipt", "results"}

    def test_separate_entities_have_independent_chains(self, ledger):
        _append_full_chain(ledger, "BATCH-A")
        other = ledger.append("BATCH-B", "sample_batch", "batch_created", "geo", {"n": 7})
        assert other.prev_hash == GENESIS_HASH  # not linked to BATCH-A's head
        assert len(ledger.get_chain("BATCH-A")) == 4
        assert len(ledger.get_chain("BATCH-B")) == 1


class TestVerification:
    def test_clean_chain_verifies(self, ledger):
        _append_full_chain(ledger)
        result = ledger.verify_chain("BATCH-2025-001")
        assert result.valid is True
        assert result.entries_checked == 4
        assert result.errors == []
        assert ledger.verify_all().valid is True

    def test_payload_tamper_breaks_verification(self, ledger, session):
        entries = _append_full_chain(ledger)
        victim = session.get(CustodyLedgerEntry, entries[2].id)
        victim.payload = {"received_count": 41, "seals_intact": False}  # tampered
        session.commit()
        result = ledger.verify_chain("BATCH-2025-001")
        assert result.valid is False
        assert result.first_invalid_id == entries[2].id
        assert any("hash mismatch" in e for e in result.errors)

    def test_flipped_hash_byte_breaks_linkage(self, ledger, session):
        entries = _append_full_chain(ledger)
        victim = session.get(CustodyLedgerEntry, entries[1].id)
        victim.entry_hash = ("f" if victim.entry_hash[0] != "f" else "e") + victim.entry_hash[1:]
        session.commit()
        result = ledger.verify_chain("BATCH-2025-001")
        assert result.valid is False
        assert result.first_invalid_id == entries[1].id
        assert any("hash mismatch" in e for e in result.errors)
        assert any("linkage broken" in e for e in result.errors)  # next entry's prev_hash

    def test_signature_forgery_detected(self, ledger, session):
        entries = _append_full_chain(ledger)
        victim = session.get(CustodyLedgerEntry, entries[0].id)
        victim.signature = "0" * 64  # forged without the server key
        session.commit()
        result = ledger.verify_chain("BATCH-2025-001")
        assert result.valid is False
        assert any("invalid HMAC signature" in e for e in result.errors)

    def test_wrong_hmac_key_fails_all_signatures(self, ledger):
        _append_full_chain(ledger)
        foreign = CustodyLedger(ledger.db, hmac_key=b"different-key")
        result = foreign.verify_chain("BATCH-2025-001")
        assert result.valid is False
        assert result.errors.count(result.errors[0]) >= 1
        assert all("signature" in e for e in result.errors)

    def test_row_deletion_detected_as_sequence_gap(self, ledger, session):
        entries = _append_full_chain(ledger)
        session.delete(session.get(CustodyLedgerEntry, entries[2].id))
        session.commit()
        result = ledger.verify_chain("BATCH-2025-001")
        assert result.valid is False
        assert any("linkage broken" in e or "sequence gap" in e for e in result.errors)


class TestAPI:
    @pytest.fixture()
    def client(self, session, monkeypatch):
        monkeypatch.setenv("MV_LEDGER_HMAC_KEY", TEST_KEY.decode())
        app = FastAPI()
        app.include_router(router)

        def override_db():
            yield session

        app.dependency_overrides[get_db] = override_db
        return TestClient(app)

    def test_full_lifecycle_via_api(self, client):
        posts = [
            {"entity_id": "BATCH-X", "entity_type": "sample_batch", "event_type": "batch_created",
             "actor": "geo", "payload": {"sample_count": 10}},
            {"entity_id": "BATCH-X", "entity_type": "dispatch", "event_type": "dispatch",
             "actor": "geo", "payload": {"courier": "FastShip"}},
            {"entity_id": "BATCH-X", "entity_type": "lab_receipt", "event_type": "lab_receipt",
             "actor": "lab", "payload": {"received_count": 10}},
            {"entity_id": "BATCH-X", "entity_type": "results", "event_type": "results",
             "actor": "lab", "payload": {"mean_au_ppm": 2.4}},
        ]
        resp = client.post("/innovations/custody_ledger/entries/batch", json={"entries": posts})
        assert resp.status_code == 201
        body = resp.json()
        assert body["appended"] == 4
        assert body["entries"][0]["prev_hash"] == GENESIS_HASH

        chain = client.get("/innovations/custody_ledger/chain/BATCH-X")
        assert chain.status_code == 200
        assert chain.json()["length"] == 4

        verify = client.get("/innovations/custody_ledger/verify/BATCH-X")
        assert verify.status_code == 200
        assert verify.json()["valid"] is True
        assert verify.json()["entries_checked"] == 4

        verify_all = client.get("/innovations/custody_ledger/verify")
        assert verify_all.json()["valid"] is True

    def test_invalid_event_type_422(self, client):
        resp = client.post("/innovations/custody_ledger/entries", json={
            "entity_id": "B1", "entity_type": "sample_batch", "event_type": "bogus",
            "actor": "geo", "payload": {},
        })
        assert resp.status_code == 422

    def test_missing_entity_chain_404(self, client):
        assert client.get("/innovations/custody_ledger/chain/NOPE").status_code == 404
        assert client.get("/innovations/custody_ledger/verify/NOPE").status_code == 404
