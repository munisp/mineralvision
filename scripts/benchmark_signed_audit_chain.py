#!/usr/bin/env python3
"""Benchmark concurrent append-only signed audit events on PostgreSQL.

This is a controlled performance benchmark, not a production capacity promise.
It measures one tenant/stream, where row-lock serialization represents the
highest-contention chain design. Production sizing must use comparable database,
KMS/HSM signing latency, collector/export traffic, and retention settings.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package"))

from src.api.innovations.integration_hub.audit_crypto import (  # noqa: E402
    append_signed_audit_event,
    event_to_export_record,
    verify_audit_chain,
)
from src.api.innovations.integration_hub.db import Base  # noqa: E402
from src.api.innovations.integration_hub.models import (  # noqa: E402
    AuditStreamModel,
    SignedAuditEventModel,
)


def percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def run_case(session_factory: Any, tenant_id: str, stream_id: str, count: int, workers: int) -> dict[str, Any]:
    key = Ed25519PrivateKey.generate()

    def append_one(index: int) -> float:
        session = session_factory()
        started = time.perf_counter()
        try:
            append_signed_audit_event(
                session,
                tenant_id=tenant_id,
                stream_id=stream_id,
                event_type="benchmark.append",
                actor_id="benchmark-runner",
                payload={"index": index, "run_id": tenant_id},
                key_id="benchmark-ed25519",
                private_key=key,
            )
            return (time.perf_counter() - started) * 1000.0
        finally:
            session.close()

    latencies: list[float] = []
    errors: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(append_one, index) for index in range(count)]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as exc:  # Record benchmark failures rather than conceal them.
                errors.append(f"{type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - started

    session = session_factory()
    try:
        events = (
            session.query(SignedAuditEventModel)
            .filter(
                SignedAuditEventModel.tenant_id == tenant_id,
                SignedAuditEventModel.stream_id == stream_id,
            )
            .order_by(SignedAuditEventModel.sequence)
            .all()
        )
        verification = verify_audit_chain(
            [event_to_export_record(event) for event in events],
            public_keys={"benchmark-ed25519": key.public_key()},
            expected_tenant_id=tenant_id,
            expected_stream_id=stream_id,
        )
    finally:
        session.close()

    return {
        "workers": workers,
        "requested_events": count,
        "appended_events": len(latencies),
        "errors": errors,
        "wall_seconds": round(elapsed, 6),
        "throughput_events_per_second": round(len(latencies) / elapsed, 3) if elapsed else 0.0,
        "latency_ms": {
            "min": round(min(latencies), 3) if latencies else 0.0,
            "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "chain_verification": {
            "valid": verification.valid,
            "event_count": verification.event_count,
            "last_sequence": verification.last_sequence,
            "failure_count": len(verification.failures),
        },
    }


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv("MV_BENCHMARK_DATABASE_URL", ""))
    parser.add_argument("--events", type=int, default=200, help="Events for each worker-count case")
    parser.add_argument("--workers", default="1,4,16,32", help="Comma-separated thread counts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-data", action="store_true", help="Keep synthetic benchmark rows for inspection")
    args = parser.parse_args()

    if not args.database_url.startswith("postgresql"):
        raise RuntimeError("--database-url or MV_BENCHMARK_DATABASE_URL must be a PostgreSQL SQLAlchemy URL")
    if args.events < 1:
        raise ValueError("--events must be positive")
    worker_counts = [int(value) for value in args.workers.split(",")]
    if any(value < 1 for value in worker_counts):
        raise ValueError("worker counts must be positive")

    engine = create_engine(args.database_url, pool_size=max(worker_counts) + 4, max_overflow=4, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    tenant_id = f"benchmark-{uuid.uuid4().hex}"
    report: dict[str, Any] = {
        "benchmark": "MineralVision signed append-only audit chain",
        "database_dialect": engine.dialect.name,
        "events_per_case": args.events,
        "worker_counts": worker_counts,
        "scope": "single tenant/single stream row-lock contention; in-process Ed25519 signing; no KMS/HSM or off-host exporter latency",
        "cases": [],
    }
    try:
        for workers in worker_counts:
            report["cases"].append(
                run_case(session_factory, tenant_id, f"stream-{workers}", args.events, workers)
            )
    finally:
        if not args.keep_data:
            cleanup(session_factory, tenant_id)
        engine.dispose()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(not case["errors"] and case["chain_verification"]["valid"] for case in report["cases"]) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"benchmark_error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        raise SystemExit(2)
