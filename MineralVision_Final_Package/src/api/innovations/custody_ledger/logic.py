"""Core hash-chain logic for the assay chain-of-custody ledger.

Append-only ledger: every entry commits to its predecessor via
    entry_hash = SHA256(prev_hash || canonical_json(payload) || iso_timestamp || actor)
and is HMAC-signed with a server key so that rows cannot be forged or
re-signed without the key.  Verification replays the chain and checks
hashes, signatures, linkage and sequence monotonicity.
"""

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .models import CustodyLedgerEntry

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64

# Custody event vocabulary for assay chains.
EVENT_TYPES = ("batch_created", "dispatch", "lab_receipt", "results")
ENTITY_TYPES = ("sample_batch", "dispatch", "lab_receipt", "results")

_ephemeral_key: Optional[bytes] = None


def get_hmac_key() -> bytes:
    """Resolve the ledger HMAC key.

    Mirrors the JWT-secret pattern: read from the environment; in
    development fall back to an ephemeral random key (generated once per
    process) with a loud warning.
    """
    env_key = os.getenv("MV_LEDGER_HMAC_KEY")
    if env_key:
        return env_key.encode("utf-8")
    global _ephemeral_key
    if _ephemeral_key is None:
        logger.warning(
            "MV_LEDGER_HMAC_KEY not set — using an ephemeral random key. "
            "Signatures will not survive a process restart (dev only)."
        )
        _ephemeral_key = os.urandom(32)
    return _ephemeral_key


def canonical_json(payload: Dict[str, Any]) -> str:
    """Deterministic JSON serialization for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_entry_hash(prev_hash: str, payload: Dict[str, Any], iso_timestamp: str, actor: str) -> str:
    """SHA256(prev_hash || canonical_json(payload) || iso_timestamp || actor)."""
    material = f"{prev_hash}{canonical_json(payload)}{iso_timestamp}{actor}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def sign_entry(entry_hash: str, key: Optional[bytes] = None) -> str:
    """HMAC-SHA256 signature over the entry hash."""
    return hmac.new(key or get_hmac_key(), entry_hash.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass
class VerificationResult:
    valid: bool
    entries_checked: int
    first_invalid_id: Optional[int] = None
    errors: List[str] = field(default_factory=list)


def _entity_chain(db: Session, entity_id: str) -> List[CustodyLedgerEntry]:
    return (
        db.query(CustodyLedgerEntry)
        .filter(CustodyLedgerEntry.entity_id == entity_id)
        .order_by(CustodyLedgerEntry.id.asc())
        .all()
    )


class CustodyLedger:
    """Append and verify operations over the ledger table."""

    def __init__(self, db: Session, hmac_key: Optional[bytes] = None):
        self.db = db
        self.key = hmac_key  # None → resolve from env/ephemeral per call

    # ------------------------------------------------------------------ append
    def append(
        self,
        entity_id: str,
        entity_type: str,
        event_type: str,
        actor: str,
        payload: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> CustodyLedgerEntry:
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"invalid entity_type {entity_type!r}; expected one of {ENTITY_TYPES}")
        if event_type not in EVENT_TYPES:
            raise ValueError(f"invalid event_type {event_type!r}; expected one of {EVENT_TYPES}")
        if not actor:
            raise ValueError("actor is required")

        ts = timestamp or datetime.now(timezone.utc).replace(tzinfo=None)
        iso_ts = ts.isoformat()

        chain = _entity_chain(self.db, entity_id)
        prev_hash = chain[-1].entry_hash if chain else GENESIS_HASH

        entry_hash = compute_entry_hash(prev_hash, payload, iso_ts, actor)
        signature = sign_entry(entry_hash, self.key)

        entry = CustodyLedgerEntry(
            entity_id=entity_id,
            entity_type=entity_type,
            event_type=event_type,
            actor=actor,
            payload=payload,
            timestamp=ts,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            signature=signature,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    # ------------------------------------------------------------------- read
    def get_chain(self, entity_id: str) -> List[CustodyLedgerEntry]:
        return _entity_chain(self.db, entity_id)

    def list_entities(self) -> List[str]:
        rows = self.db.query(CustodyLedgerEntry.entity_id).distinct().order_by(CustodyLedgerEntry.entity_id).all()
        return [r[0] for r in rows]

    # ----------------------------------------------------------------- verify
    def _check_entry(self, entry: CustodyLedgerEntry, expected_prev: str, expected_seq: int) -> List[str]:
        errors = []
        if entry.id != expected_seq:
            errors.append(f"entry {entry.id}: sequence gap (expected {expected_seq})")
        if entry.prev_hash != expected_prev:
            errors.append(f"entry {entry.id}: prev_hash linkage broken")
        iso_ts = entry.timestamp.isoformat()
        recomputed = compute_entry_hash(entry.prev_hash, entry.payload, iso_ts, entry.actor)
        if recomputed != entry.entry_hash:
            errors.append(f"entry {entry.id}: hash mismatch (payload/timestamp/actor tampered)")
        expected_sig = sign_entry(entry.entry_hash, self.key)
        if not hmac.compare_digest(expected_sig, entry.signature):
            errors.append(f"entry {entry.id}: invalid HMAC signature")
        return errors

    def verify_chain(self, entity_id: str) -> VerificationResult:
        """Replay one entity's chain, proving integrity end to end."""
        errors: List[str] = []
        prev_hash = GENESIS_HASH
        expected_seq: Optional[int] = None
        entries = _entity_chain(self.db, entity_id)
        first_invalid: Optional[int] = None
        for i, entry in enumerate(entries):
            if expected_seq is None:
                expected_seq = entry.id  # first row of this entity defines its base seq
            entry_errors = self._check_entry(entry, prev_hash, expected_seq)
            if entry_errors and first_invalid is None:
                first_invalid = entry.id
            errors.extend(entry_errors)
            prev_hash = entry.entry_hash
            expected_seq += 1
        return VerificationResult(
            valid=not errors,
            entries_checked=len(entries),
            first_invalid_id=first_invalid,
            errors=errors,
        )

    def verify_all(self) -> VerificationResult:
        """Verify every entity chain plus global sequence monotonicity."""
        errors: List[str] = []
        total = 0
        first_invalid: Optional[int] = None
        for entity_id in self.list_entities():
            result = self.verify_chain(entity_id)
            total += result.entries_checked
            if not result.valid:
                if first_invalid is None:
                    first_invalid = result.first_invalid_id
                errors.extend(f"[{entity_id}] {e}" for e in result.errors)
        # Global monotonic sequence check
        ids = [r[0] for r in self.db.query(CustodyLedgerEntry.id).order_by(CustodyLedgerEntry.id.asc()).all()]
        if ids != list(range(1, len(ids) + 1)):
            errors.append("global sequence is not strictly monotonic from 1")
        return VerificationResult(valid=not errors, entries_checked=total, first_invalid_id=first_invalid, errors=errors)


def entry_to_dict(entry: CustodyLedgerEntry) -> Dict[str, Any]:
    return {
        "id": entry.id,
        "entity_id": entry.entity_id,
        "entity_type": entry.entity_type,
        "event_type": entry.event_type,
        "actor": entry.actor,
        "payload": entry.payload,
        "timestamp": entry.timestamp.isoformat(),
        "prev_hash": entry.prev_hash,
        "entry_hash": entry.entry_hash,
        "signature": entry.signature,
    }
