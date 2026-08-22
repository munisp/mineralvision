"""Reference off-host collector for signed partitioned Merkle audit batches.

The collector is deliberately stateful and conservative. It maintains independent
anchors per (tenant, routing epoch, partition); it never invents global ordering
across partitions. This module is an in-process reference, not a networked or
HA collector service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .merkle_audit import MerkleProofStep, verify_inclusion, verify_signed_batch_commitment

ChainKey = Tuple[str, int, int]


@dataclass(frozen=True)
class BatchInclusion:
    commitment: Mapping[str, Any]
    proof: Sequence[MerkleProofStep]


@dataclass(frozen=True)
class CollectorDecision:
    status: str  # accepted, pending, duplicate, rejected
    reason: str
    chain_key: ChainKey | None
    accepted_sequences: Tuple[int, ...] = ()


@dataclass
class ChainAnchor:
    last_sequence: int = 0
    last_root_b64: str = ""


@dataclass
class _ReceivedBatch:
    signed_batch: Mapping[str, Any]
    inclusions: Sequence[BatchInclusion]


@dataclass
class MerkleAnchorCollector:
    public_keys: Mapping[str, Ed25519PublicKey]
    anchors: Dict[ChainKey, ChainAnchor] = field(default_factory=dict)
    accepted: Dict[Tuple[ChainKey, int, int], str] = field(default_factory=dict)
    pending: Dict[ChainKey, Dict[Tuple[int, int, str], _ReceivedBatch]] = field(default_factory=dict)

    def ingest(
        self,
        signed_batch: Mapping[str, Any],
        inclusions: Sequence[BatchInclusion],
    ) -> CollectorDecision:
        batch = signed_batch.get("batch") if isinstance(signed_batch, Mapping) else None
        if not isinstance(batch, Mapping):
            return CollectorDecision("rejected", "missing batch payload", None)
        try:
            tenant_id = str(batch["tenant_id"])
            epoch = int(batch["routing_epoch"])
            partition = int(batch["partition"])
            first = int(batch["first_sequence"])
            last = int(batch["last_sequence"])
            root = str(batch["merkle_root_b64"])
            previous_root = str(batch["previous_batch_root_b64"])
            event_count = int(batch["event_count"])
        except (KeyError, TypeError, ValueError):
            return CollectorDecision("rejected", "malformed batch metadata", None)
        chain_key = (tenant_id, epoch, partition)
        if not tenant_id or first < 1 or last < first or event_count != last - first + 1:
            return CollectorDecision("rejected", "invalid sequence range", chain_key)
        if len(inclusions) != event_count:
            return CollectorDecision("rejected", "inclusion count does not match event count", chain_key)
        if not verify_signed_batch_commitment(signed_batch, self.public_keys):
            return CollectorDecision("rejected", "invalid batch signature or untrusted key", chain_key)
        for inclusion in inclusions:
            if inclusion.commitment.get("tenant_id") != tenant_id:
                return CollectorDecision("rejected", "cross-tenant event commitment", chain_key)
            if not verify_inclusion(inclusion.commitment, inclusion.proof, root):
                return CollectorDecision("rejected", "invalid Merkle inclusion proof", chain_key)

        range_key = (chain_key, first, last)
        existing = self.accepted.get(range_key)
        if existing is not None:
            if existing == root:
                return CollectorDecision("duplicate", "idempotent replay", chain_key)
            return CollectorDecision("rejected", "conflicting root for accepted sequence range", chain_key)
        pending_key = (first, last, root)
        pending_chain = self.pending.setdefault(chain_key, {})
        if pending_key in pending_chain:
            return CollectorDecision("duplicate", "idempotent pending replay", chain_key)

        item = _ReceivedBatch(signed_batch=signed_batch, inclusions=inclusions)
        anchor = self.anchors.setdefault(chain_key, ChainAnchor())
        if first != anchor.last_sequence + 1 or previous_root != anchor.last_root_b64:
            pending_chain[pending_key] = item
            return CollectorDecision("pending", "awaiting predecessor anchor", chain_key)

        accepted_sequences = self._accept_and_drain(chain_key, item)
        return CollectorDecision("accepted", "anchor advanced", chain_key, accepted_sequences)

    def _accept_and_drain(self, chain_key: ChainKey, item: _ReceivedBatch) -> Tuple[int, ...]:
        accepted_sequences = []
        current = item
        while current is not None:
            batch = current.signed_batch["batch"]
            first, last = int(batch["first_sequence"]), int(batch["last_sequence"])
            root = str(batch["merkle_root_b64"])
            self.accepted[(chain_key, first, last)] = root
            anchor = self.anchors[chain_key]
            anchor.last_sequence = last
            anchor.last_root_b64 = root
            accepted_sequences.extend(range(first, last + 1))

            current = None
            for pending_key, candidate in list(self.pending.get(chain_key, {}).items()):
                candidate_batch = candidate.signed_batch["batch"]
                if (
                    int(candidate_batch["first_sequence"]) == anchor.last_sequence + 1
                    and str(candidate_batch["previous_batch_root_b64"]) == anchor.last_root_b64
                ):
                    del self.pending[chain_key][pending_key]
                    current = candidate
                    break
        return tuple(accepted_sequences)
