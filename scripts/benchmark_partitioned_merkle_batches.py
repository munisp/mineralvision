#!/usr/bin/env python3
"""Benchmark partitioned Merkle batching at a synthetic target arrival rate.

This benchmark measures in-process routing, batching, proof creation, Ed25519
batch signing, reference collector verification, and per-event time-to-anchor.
It does not measure durable queueing, PostgreSQL, KMS/HSM, mTLS, or off-host
network/export latency. It is not a production capacity test.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package"))

from src.api.innovations.integration_hub.merkle_audit import (  # noqa: E402
    build_merkle_batch,
    build_signed_batch_commitment,
    route_partition,
)
from src.api.innovations.integration_hub.merkle_collector import (  # noqa: E402
    BatchInclusion,
    MerkleAnchorCollector,
)


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=20_000)
    parser.add_argument("--target-rate", type=int, default=5_000)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.events, args.target_rate, args.partitions, args.batch_size) < 1:
        raise ValueError("events, target-rate, partitions, and batch-size must be positive")

    key = Ed25519PrivateKey.generate()
    key_id = "benchmark-merkle-ed25519"
    tenant_id = "benchmark-tenant"
    collector = MerkleAnchorCollector({key_id: key.public_key()})
    buffers: Dict[int, List[tuple[dict[str, Any], float]]] = defaultdict(list)
    previous_roots = defaultdict(str)
    next_sequences = defaultdict(lambda: 1)
    anchor_latencies_ms: List[float] = []
    batch_sizes: List[int] = []
    collector_failures: List[str] = []
    sealed_batches = 0

    def seal(partition: int) -> None:
        nonlocal sealed_batches
        buffered = buffers[partition]
        if not buffered:
            return
        commitments = [item[0] for item in buffered]
        merkle = build_merkle_batch(commitments)
        first = next_sequences[partition]
        signed = build_signed_batch_commitment(
            tenant_id=tenant_id,
            routing_epoch=1,
            partition_count=args.partitions,
            partition=partition,
            first_sequence=first,
            last_sequence=first + len(commitments) - 1,
            previous_batch_root_b64=previous_roots[partition],
            merkle_root_b64=merkle.root_b64,
            event_count=len(commitments),
            key_id=key_id,
            private_key=key,
            sealed_at=datetime.now(timezone.utc),
        )
        decision = collector.ingest(
            signed,
            [BatchInclusion(commitment=commitment, proof=proof) for commitment, proof in zip(commitments, merkle.proofs)],
        )
        if decision.status != "accepted":
            collector_failures.append(f"partition={partition}: {decision.status}: {decision.reason}")
            return
        completion = time.perf_counter()
        anchor_latencies_ms.extend((completion - received) * 1000.0 for _, received in buffered)
        batch_sizes.append(len(buffered))
        sealed_batches += 1
        previous_roots[partition] = merkle.root_b64
        next_sequences[partition] = first + len(commitments)
        buffers[partition] = []

    started = time.perf_counter()
    for index in range(args.events):
        # Pace synthetic arrival rate. The schedule avoids cumulative drift while
        # allowing a visibly measurable backlog if processing cannot keep up.
        due = started + (index / args.target_rate)
        while time.perf_counter() < due:
            time.sleep(0.00005)
        received = time.perf_counter()
        connector_id = f"connector-{index % 4}"
        entity_scope = f"project-{index % 64}"
        partition = route_partition(
            tenant_id=tenant_id,
            connector_id=connector_id,
            entity_scope=entity_scope,
            partition_count=args.partitions,
        )
        commitment = {
            "tenant_id": tenant_id,
            "event_id": f"benchmark-{index}",
            "connector_id": connector_id,
            "entity_scope": entity_scope,
            "payload_hash": f"payload-{index:08x}",
        }
        buffers[partition].append((commitment, received))
        if len(buffers[partition]) >= args.batch_size:
            seal(partition)
    for partition in list(buffers):
        seal(partition)
    finished = time.perf_counter()

    elapsed = finished - started
    anchor_state = {
        f"{tenant}/{epoch}/{partition}": {"last_sequence": anchor.last_sequence, "last_root_b64": anchor.last_root_b64}
        for (tenant, epoch, partition), anchor in collector.anchors.items()
    }
    report = {
        "benchmark": "MineralVision partitioned Merkle batch simulation",
        "scope": "in-process routing, Merkle proofs, Ed25519 signing, and reference collector; excludes durable storage, KMS/HSM, mTLS, and off-host transport",
        "events": args.events,
        "target_events_per_second": args.target_rate,
        "achieved_events_per_second": round(args.events / elapsed, 3),
        "elapsed_seconds": round(elapsed, 6),
        "partitions": args.partitions,
        "batch_size_limit": args.batch_size,
        "sealed_batches": sealed_batches,
        "batch_size": {
            "min": min(batch_sizes) if batch_sizes else 0,
            "mean": round(statistics.mean(batch_sizes), 3) if batch_sizes else 0.0,
            "max": max(batch_sizes) if batch_sizes else 0,
        },
        "time_to_signed_collector_anchor_ms": {
            "min": round(min(anchor_latencies_ms), 3) if anchor_latencies_ms else 0.0,
            "mean": round(statistics.mean(anchor_latencies_ms), 3) if anchor_latencies_ms else 0.0,
            "p50": round(percentile(anchor_latencies_ms, 0.50), 3),
            "p95": round(percentile(anchor_latencies_ms, 0.95), 3),
            "p99": round(percentile(anchor_latencies_ms, 0.99), 3),
            "max": round(max(anchor_latencies_ms), 3) if anchor_latencies_ms else 0.0,
        },
        "collector_failures": collector_failures,
        "collector_pending_batches": sum(len(value) for value in collector.pending.values()),
        "collector_anchors": anchor_state,
        "success": not collector_failures and len(anchor_latencies_ms) == args.events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
