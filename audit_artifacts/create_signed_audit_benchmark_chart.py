#!/usr/bin/env python3
"""Render throughput and p95 latency from the signed audit-chain benchmark JSON."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    data = json.loads(source.read_text(encoding="utf-8"))
    cases = data["cases"]
    workers = [case["workers"] for case in cases]
    throughput = [case["throughput_events_per_second"] for case in cases]
    p95 = [case["latency_ms"]["p95"] for case in cases]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    color = "#0d5c63"
    accent = "#ca6f1e"

    axes[0].plot(workers, throughput, marker="o", linewidth=2.5, color=color)
    axes[0].set_title("Append Throughput")
    axes[0].set_xlabel("Concurrent workers")
    axes[0].set_ylabel("Events per second")
    axes[0].set_xticks(workers)
    for x, y in zip(workers, throughput):
        axes[0].annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")

    axes[1].plot(workers, p95, marker="o", linewidth=2.5, color=accent)
    axes[1].set_title("p95 Append Latency")
    axes[1].set_xlabel("Concurrent workers")
    axes[1].set_ylabel("Milliseconds")
    axes[1].set_xticks(workers)
    for x, y in zip(workers, p95):
        axes[1].annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")

    figure.suptitle("MineralVision Signed Audit Chain: Single-Stream PostgreSQL Contention Benchmark", fontsize=12, weight="bold")
    figure.text(0.5, 0.01, "200 events/case; in-process Ed25519; excludes KMS/HSM and off-host export latency", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
