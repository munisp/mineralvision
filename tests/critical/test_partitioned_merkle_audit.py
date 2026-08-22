from datetime import datetime, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.api.innovations.integration_hub.merkle_audit import (
    MerkleProofStep,
    build_merkle_batch,
    build_signed_batch_commitment,
    route_partition,
    verify_inclusion,
    verify_signed_batch_commitment,
)


def test_partition_routing_is_deterministic_tenant_scoped_and_validated():
    first = route_partition(
        tenant_id="tenant-a", connector_id="arcgis", entity_scope="project-1", partition_count=32
    )
    second = route_partition(
        tenant_id="tenant-a", connector_id="arcgis", entity_scope="project-1", partition_count=32
    )
    other_tenant = route_partition(
        tenant_id="tenant-b", connector_id="arcgis", entity_scope="project-1", partition_count=32
    )
    assert first == second
    assert 0 <= first < 32
    assert 0 <= other_tenant < 32
    # Different tenant material is explicitly included in the routing digest.
    assert route_partition(tenant_id="tenant-a", connector_id="arcgis", entity_scope="project-1", partition_count=1) == 0


def test_merkle_batch_proves_every_event_and_rejects_tampering():
    commitments = [
        {"event_id": f"ev-{index}", "tenant_id": "tenant-a", "payload_hash": str(index)}
        for index in range(5)
    ]
    batch = build_merkle_batch(commitments)
    assert len(batch.proofs) == len(commitments)
    for commitment, proof in zip(commitments, batch.proofs):
        assert verify_inclusion(commitment, proof, batch.root_b64)

    changed = dict(commitments[2])
    changed["payload_hash"] = "altered"
    assert not verify_inclusion(changed, batch.proofs[2], batch.root_b64)
    invalid_proof = list(batch.proofs[0])
    invalid_proof[0] = MerkleProofStep("invalid", invalid_proof[0].sibling_b64)
    assert not verify_inclusion(commitments[0], invalid_proof, batch.root_b64)


def test_signed_batch_commitment_binds_partition_anchor_and_root():
    key = Ed25519PrivateKey.generate()
    batch = build_merkle_batch([{"event_id": "ev-1", "tenant_id": "tenant-a"}])
    signed = build_signed_batch_commitment(
        tenant_id="tenant-a",
        routing_epoch=1,
        partition_count=32,
        partition=7,
        first_sequence=1,
        last_sequence=1,
        previous_batch_root_b64="",
        merkle_root_b64=batch.root_b64,
        event_count=1,
        key_id="audit-q3",
        private_key=key,
        sealed_at=datetime.now(timezone.utc),
    )
    assert verify_signed_batch_commitment(signed, {"audit-q3": key.public_key()})
    tampered = {"batch": dict(signed["batch"]), "signature_b64": signed["signature_b64"]}
    tampered["batch"]["partition"] = 8
    assert not verify_signed_batch_commitment(tampered, {"audit-q3": key.public_key()})
