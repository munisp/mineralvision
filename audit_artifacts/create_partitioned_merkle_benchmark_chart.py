#!/usr/bin/env python3
"""Render key latency and batching metrics from a partitioned Merkle benchmark."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    report = json.loads(source.read_text(encoding="utf-8"))
    latency = report["time_to_signed_collector_anchor_ms"]
    labels = ["p50", "p95", "p99", "max"]
    values = [latency[label] for label in labels]
    colors = ["#0d5c63", "#2a9d8f", "#e9c46a", "#ca6f1e"]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    axes[0].bar(labels, values, color=colors)
    axes[0].set_title("Time to Signed Collector Anchor")
    axes[0].set_ylabel("Milliseconds")
    for index, value in enumerate(values):
        axes[0].text(index, value + max(values) * 0.02, f"{value:.1f}", ha="center", weight="bold")

    metrics = ["Target\nrate", "Achieved\nrate", "Events", "Batches"]
    values_two = [
        report["target_events_per_second"],
        report["achieved_events_per_second"],
        report["events"],
        report["sealed_batches"],
    ]
    axes[1].bar(metrics, values_two, color=["#64748b", "#0d5c63", "#2a9d8f", "#ca6f1e"])
    axes[1].set_title("Batch Processing Volume")
    axes[1].set_ylabel("Count / events per second")
    for index, value in enumerate(values_two):
        axes[1].text(index, value + max(values_two) * 0.02, f"{value:,.0f}", ha="center", weight="bold")

    figure.suptitle("MineralVision Partitioned Merkle Batch Simulation", weight="bold", fontsize=13)
    figure.text(0.5, 0.01, "8 partitions · 256-event maximum batch · in-process routing/signing/collector only", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
