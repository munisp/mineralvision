#!/usr/bin/env python3
"""Validate that a benchmark configuration matches the implemented promotion gate.

This does not evaluate a model and does not create synthetic samples. It protects against
configuration drift before an evaluator submits real sealed-holdout metrics to PostgreSQL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package" / "src"))
from api.oil_spill.governance import MIN_SEALED_HOLDOUT_SAMPLES, PROMOTION_THRESHOLDS  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()

with args.config.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

configured_gate = config.get("promotion_gate", {})
if configured_gate != PROMOTION_THRESHOLDS:
    raise SystemExit(
        f"Promotion-gate mismatch. Configured={configured_gate}; implemented={PROMOTION_THRESHOLDS}. "
        "Do not evaluate under a configuration that differs from the deployed gate."
    )
minimum = int(config.get("minimum_sealed_holdout_samples_per_domain", 0))
if minimum < MIN_SEALED_HOLDOUT_SAMPLES:
    raise SystemExit(
        f"Sealed-holdout minimum {minimum} is below implemented minimum {MIN_SEALED_HOLDOUT_SAMPLES}."
    )
if config.get("split", {}).get("group_key") != "incident_id":
    raise SystemExit("The reference benchmark requires incident_id grouping to prevent within-incident leakage.")
if not config.get("split", {}).get("sealed_holdout_locked"):
    raise SystemExit("The sealed holdout must be explicitly locked.")
if not config.get("intended_domains"):
    raise SystemExit("At least one intended domain is required.")

print(
    {
        "valid": True,
        "implemented_thresholds": PROMOTION_THRESHOLDS,
        "minimum_sealed_holdout_samples": minimum,
        "intended_domains": config["intended_domains"],
        "group_key": "incident_id",
    }
)
