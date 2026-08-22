"""Partition routing and Merkle-batch primitives for scalable audit evidence.

These primitives are intentionally small and explicit. They do not silently
claim RFC 9162 wire compatibility; deployments must define their own batch,
outbox, and durability policy around the signed commitments they produce.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .audit_crypto import SCHEMA_VERSION, _as_utc_naive, canonical_json, sign_b64, verify_b64

LEAF_PREFIX = b"\x00"
BRANCH_PREFIX = b"\x01"


def route_partition(
    *, tenant_id: str, connector_id: str, entity_scope: str, partition_count: int
) -> int:
    """Route a tenant-scoped connector/entity stream to a stable partition."""
    if not all([tenant_id, connector_id, entity_scope]):
        raise ValueError("tenant_id, connector_id, and entity_scope are required")
    if partition_count < 1:
        raise ValueError("partition_count must be positive")
    material = canonical_json(
        {
            "tenant_id": tenant_id,
            "connector_id": connector_id,
            "entity_scope": entity_scope,
        }
    )
    return int.from_bytes(hashlib.sha256(material).digest(), "big") % partition_count


def leaf_hash(commitment: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + canonical_json(commitment)).digest()


def branch_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(BRANCH_PREFIX + left + right).digest()


@dataclass(frozen=True)
class MerkleProofStep:
    side: str  # "left" means the sibling is left of current hash.
    sibling_b64: str


@dataclass(frozen=True)
class MerkleBatch:
    root_b64: str
    leaf_hashes_b64: Sequence[str]
    proofs: Sequence[Sequence[MerkleProofStep]]


def build_merkle_batch(commitments: Sequence[Mapping[str, Any]]) -> MerkleBatch:
    """Build a binary Merkle root and a directed proof for each commitment.

    The final odd node is duplicated at that tree level. The duplication is
    represented as an ordinary right sibling in the proof, making the verifier
    deterministic and independent from the builder.
    """
    if not commitments:
        raise ValueError("at least one event commitment is required")
    current = [leaf_hash(item) for item in commitments]
    proofs: List[List[MerkleProofStep]] = [[] for _ in current]
    index_groups: List[List[int]] = [[index] for index in range(len(current))]

    while len(current) > 1:
        next_level: List[bytes] = []
        next_groups: List[List[int]] = []
        for index in range(0, len(current), 2):
            left, left_group = current[index], index_groups[index]
            if index + 1 == len(current):
                # Duplicate an odd final node, but append exactly one proof step
                # for its original leaf indices. Adding it as both left and right
                # would make an inclusion proof self-contradictory.
                right, right_group = left, []
                for leaf_index in left_group:
                    proofs[leaf_index].append(
                        MerkleProofStep("right", base64.b64encode(right).decode("ascii"))
                    )
            else:
                right, right_group = current[index + 1], index_groups[index + 1]
                for leaf_index in left_group:
                    proofs[leaf_index].append(
                        MerkleProofStep("right", base64.b64encode(right).decode("ascii"))
                    )
                for leaf_index in right_group:
                    proofs[leaf_index].append(
                        MerkleProofStep("left", base64.b64encode(left).decode("ascii"))
                    )
            next_level.append(branch_hash(left, right))
            next_groups.append(left_group + right_group)
        current, index_groups = next_level, next_groups

    return MerkleBatch(
        root_b64=base64.b64encode(current[0]).decode("ascii"),
        leaf_hashes_b64=tuple(base64.b64encode(leaf_hash(item)).decode("ascii") for item in commitments),
        proofs=tuple(tuple(proof) for proof in proofs),
    )


def verify_inclusion(
    commitment: Mapping[str, Any], proof: Iterable[MerkleProofStep], root_b64: str
) -> bool:
    """Verify one event commitment against a batch root."""
    current = leaf_hash(commitment)
    try:
        for step in proof:
            sibling = base64.b64decode(step.sibling_b64, validate=True)
            if step.side == "left":
                current = branch_hash(sibling, current)
            elif step.side == "right":
                current = branch_hash(current, sibling)
            else:
                return False
        return current == base64.b64decode(root_b64, validate=True)
    except (ValueError, TypeError):
        return False


def build_signed_batch_commitment(
    *,
    tenant_id: str,
    routing_epoch: int,
    partition_count: int,
    partition: int,
    first_sequence: int,
    last_sequence: int,
    previous_batch_root_b64: str,
    merkle_root_b64: str,
    event_count: int,
    key_id: str,
    private_key: Ed25519PrivateKey,
    sealed_at: datetime,
) -> Dict[str, Any]:
    """Create an Ed25519-signed commitment for one sealed Merkle batch."""
    if not tenant_id or not key_id:
        raise ValueError("tenant_id and key_id are required")
    if event_count < 1 or first_sequence < 1 or last_sequence < first_sequence:
        raise ValueError("invalid batch sequence range or event count")
    body = {
        "schema_version": SCHEMA_VERSION,
        "tenant_id": tenant_id,
        "routing_epoch": routing_epoch,
        "partition_count": partition_count,
        "partition": partition,
        "first_sequence": first_sequence,
        "last_sequence": last_sequence,
        "event_count": event_count,
        "previous_batch_root_b64": previous_batch_root_b64,
        "merkle_root_b64": merkle_root_b64,
        "key_id": key_id,
        "sealed_at": _as_utc_naive(sealed_at).isoformat(timespec="microseconds") + "Z",
    }
    return {"batch": body, "signature_b64": sign_b64(private_key, body)}


def verify_signed_batch_commitment(
    signed: Mapping[str, Any], public_keys: Mapping[str, Ed25519PublicKey]
) -> bool:
    batch = signed.get("batch")
    signature = signed.get("signature_b64")
    if not isinstance(batch, dict) or not isinstance(signature, str):
        return False
    key = public_keys.get(batch.get("key_id"))
    if key is None:
        return False
    return verify_b64(key, batch, signature)
