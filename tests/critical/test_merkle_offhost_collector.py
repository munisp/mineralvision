from datetime import datetime, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.api.innovations.integration_hub.merkle_audit import (
    build_merkle_batch,
    build_signed_batch_commitment,
)
from src.api.innovations.integration_hub.merkle_collector import (
    BatchInclusion,
    MerkleAnchorCollector,
)


def make_batch(key, tenant, epoch, partition, first, previous_root, labels):
    commitments = [
        {"tenant_id": tenant, "event_id": label, "payload_hash": f"hash-{label}"}
        for label in labels
    ]
    merkle = build_merkle_batch(commitments)
    signed = build_signed_batch_commitment(
        tenant_id=tenant,
        routing_epoch=epoch,
        partition_count=8,
        partition=partition,
        first_sequence=first,
        last_sequence=first + len(commitments) - 1,
        previous_batch_root_b64=previous_root,
        merkle_root_b64=merkle.root_b64,
        event_count=len(commitments),
        key_id="collector-test-key",
        private_key=key,
        sealed_at=datetime.now(timezone.utc),
    )
    inclusions = [
        BatchInclusion(commitment=commitment, proof=proof)
        for commitment, proof in zip(commitments, merkle.proofs)
    ]
    return signed, inclusions, merkle.root_b64


def test_collector_drains_out_of_order_partition_anchor_delivery():
    key = Ed25519PrivateKey.generate()
    collector = MerkleAnchorCollector({"collector-test-key": key.public_key()})
    first, first_inclusions, root_one = make_batch(key, "tenant-a", 1, 2, 1, "", ["a1", "a2"])
    second, second_inclusions, _ = make_batch(key, "tenant-a", 1, 2, 3, root_one, ["a3", "a4"])

    pending = collector.ingest(second, second_inclusions)
    assert pending.status == "pending"
    assert collector.anchors[("tenant-a", 1, 2)].last_sequence == 0

    accepted = collector.ingest(first, first_inclusions)
    assert accepted.status == "accepted"
    assert accepted.accepted_sequences == (1, 2, 3, 4)
    assert collector.anchors[("tenant-a", 1, 2)].last_sequence == 4
    assert collector.anchors[("tenant-a", 1, 2)].last_root_b64 == second["batch"]["merkle_root_b64"]


def test_collector_keeps_partitions_independent_and_rejects_cross_tenant_proof():
    key = Ed25519PrivateKey.generate()
    collector = MerkleAnchorCollector({"collector-test-key": key.public_key()})
    tenant_a, inclusions_a, _ = make_batch(key, "tenant-a", 1, 0, 1, "", ["a1"])
    tenant_b, inclusions_b, _ = make_batch(key, "tenant-b", 1, 0, 1, "", ["b1"])
    assert collector.ingest(tenant_a, inclusions_a).status == "accepted"
    assert collector.ingest(tenant_b, inclusions_b).status == "accepted"
    assert collector.anchors[("tenant-a", 1, 0)].last_sequence == 1
    assert collector.anchors[("tenant-b", 1, 0)].last_sequence == 1

    bad = [BatchInclusion(commitment={**inclusions_a[0].commitment, "tenant_id": "tenant-b"}, proof=inclusions_a[0].proof)]
    assert collector.ingest(tenant_a, bad).status == "rejected"


def test_collector_rejects_fork_and_allows_exact_replay():
    key = Ed25519PrivateKey.generate()
    collector = MerkleAnchorCollector({"collector-test-key": key.public_key()})
    original, original_inclusions, _ = make_batch(key, "tenant-a", 1, 3, 1, "", ["a1"])
    assert collector.ingest(original, original_inclusions).status == "accepted"
    assert collector.ingest(original, original_inclusions).status == "duplicate"

    conflicting, conflicting_inclusions, _ = make_batch(key, "tenant-a", 1, 3, 1, "", ["altered-a1"])
    rejected = collector.ingest(conflicting, conflicting_inclusions)
    assert rejected.status == "rejected"
    assert "conflicting root" in rejected.reason
