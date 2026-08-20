#!/usr/bin/env python3
"""Evaluate oil-spill segmentation predictions with incident-disjoint sealed holdout.

The benchmark is intentionally prediction-format agnostic: it scores grayscale probability
masks emitted by an evaluated candidate model. It does not train a model, download a dataset,
or manufacture performance. Each incident_id is assigned to exactly one split, and the
resulting per-domain sealed-holdout metrics are checked by the platform's actual promotion
function.

Expected manifest columns are declared in configs/oil_spill_incident_disjoint_benchmark.yaml.
Paths are resolved relative to the manifest CSV. Reference masks use the configured positive
label values; prediction masks can be 8-bit probabilities (0..255) or normalized values.

Example:
  python benchmarks/evaluate_oil_spill_promotion.py \
    --config configs/oil_spill_incident_disjoint_benchmark.yaml \
    --model-id my-oil-model --model-version 1.2.0 --reviewer analyst@example.com
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package" / "src"))
from api.oil_spill.governance import evaluate_promotion_eligibility  # noqa: E402
from api.oil_spill.schemas import EvaluationSplit  # noqa: E402


METRIC_NAMES = ("oil_f1", "oil_iou", "oil_precision", "oil_recall", "pixel_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest override for a protected runner")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--jepa-backbone", default=None, help="Exact JEPA encoder/checkpoint identity, if used")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Benchmark configuration must contain a mapping")
    return config


def resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def load_manifest(config: dict[str, Any], config_path: Path) -> tuple[Path, list[dict[str, str]]]:
    manifest_path = resolve_path(str(config["manifest_csv"]), config_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest has no rows")
    missing = [column for column in config["required_columns"] if column not in rows[0]]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    for index, row in enumerate(rows, start=2):
        empty = [column for column in config["required_columns"] if not (row.get(column) or "").strip()]
        if empty:
            raise ValueError(f"Manifest row {index} has blank required values: {empty}")
    return manifest_path, rows


def assign_incident_disjoint_splits(rows: list[dict[str, str]], config: dict[str, Any]) -> dict[str, str]:
    """Assign whole incidents deterministically, stratified by domain.

    Each incident must belong to exactly one declared domain. Multi-domain incidents need a
    composite incident ID before benchmarking; silently splitting them would leak evidence.
    """
    split_config = config["split"]
    group_key = split_config["group_key"]
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_group[row[group_key]].add(row["domain"])
    multi_domain = {group: domains for group, domains in by_group.items() if len(domains) != 1}
    if multi_domain:
        example = next(iter(multi_domain.items()))
        raise ValueError(
            f"Incident-disjoint grouping requires one domain per {group_key}; found {example[0]} in {sorted(example[1])}"
        )

    by_domain: dict[str, list[str]] = defaultdict(list)
    for group, domains in by_group.items():
        by_domain[next(iter(domains))].append(group)

    seed = str(config["seed"])
    assignment: dict[str, str] = {}
    for domain, groups in by_domain.items():
        ordered = sorted(groups, key=lambda value: sha256_bytes(f"{seed}:{domain}:{value}".encode("utf-8")))
        total = len(ordered)
        if total < 3:
            raise ValueError(f"Domain '{domain}' has only {total} incidents; at least 3 are required for train/validation/sealed holdout")
        sealed_count = max(1, math.floor(total * float(split_config["sealed_holdout_fraction"])))
        validation_count = max(1, math.floor(total * float(split_config["validation_fraction"])))
        if sealed_count + validation_count >= total:
            validation_count = max(0, total - sealed_count - 1)
        for position, group in enumerate(ordered):
            if position < sealed_count:
                assignment[group] = split_config["sealed_holdout_name"]
            elif position < sealed_count + validation_count:
                assignment[group] = "validation"
            else:
                assignment[group] = "train"
    return assignment


def read_grayscale(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask file does not exist: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def confusion_for_sample(row: dict[str, str], manifest_path: Path, mask_config: dict[str, Any]) -> dict[str, int] | None:
    reference = read_grayscale(resolve_path(row["mask_path"], manifest_path))
    prediction = read_grayscale(resolve_path(row["prediction_path"], manifest_path))
    if reference.shape != prediction.shape:
        raise ValueError(f"Mask/prediction shape mismatch for sample {row['sample_id']}: {reference.shape} != {prediction.shape}")

    ignore_value = mask_config.get("ignore_value")
    valid = np.ones(reference.shape, dtype=bool) if ignore_value is None else reference != float(ignore_value)
    if int(valid.sum()) < int(mask_config["minimum_valid_pixels_per_sample"]):
        return None
    positive_values = np.asarray(mask_config["reference_positive_values"], dtype=np.float32)
    actual = np.isin(reference, positive_values) & valid
    probability = prediction / 255.0 if float(prediction.max(initial=0.0)) > 1.0 else prediction
    if not np.isfinite(probability).all() or probability.min(initial=0.0) < 0 or probability.max(initial=0.0) > 1:
        raise ValueError(f"Prediction for sample {row['sample_id']} is not a finite probability mask")
    predicted = (probability >= float(mask_config["threshold"])) & valid
    return {
        "tp": int(np.logical_and(predicted, actual).sum()),
        "fp": int(np.logical_and(predicted, np.logical_and(~actual, valid)).sum()),
        "fn": int(np.logical_and(~predicted, actual).sum()),
        "tn": int(np.logical_and(~predicted, np.logical_and(~actual, valid)).sum()),
        "valid_pixels": int(valid.sum()),
    }


def add_confusion(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {name: int(left.get(name, 0) + right.get(name, 0)) for name in ("tp", "fp", "fn", "tn", "valid_pixels")}


def empty_confusion() -> dict[str, int]:
    return {name: 0 for name in ("tp", "fp", "fn", "tn", "valid_pixels")}


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics_from_confusion(confusion: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (confusion[name] for name in ("tp", "fp", "fn", "tn"))
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    return {
        "oil_precision": precision,
        "oil_recall": recall,
        "oil_f1": safe_ratio(2 * precision * recall, precision + recall),
        "oil_iou": safe_ratio(tp, tp + fp + fn),
        # Diagnostic only. It is not part of promotion due to extreme background imbalance.
        "pixel_accuracy": safe_ratio(tp + tn, tp + fp + fn + tn),
    }


def bootstrap_intervals(
    group_confusions: list[dict[str, int]], seed_key: str, replicates: int, confidence_level: float
) -> dict[str, list[float]]:
    if not group_confusions:
        return {metric: [0.0, 0.0] for metric in METRIC_NAMES}
    rng_seed = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    values: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    count = len(group_confusions)
    for _ in range(replicates):
        aggregate = empty_confusion()
        for index in rng.integers(0, count, size=count):
            aggregate = add_confusion(aggregate, group_confusions[int(index)])
        for metric, value in metrics_from_confusion(aggregate).items():
            values[metric].append(value)
    lower = (1.0 - confidence_level) / 2.0
    upper = 1.0 - lower
    return {
        metric: [round(float(np.quantile(series, lower)), 6), round(float(np.quantile(series, upper)), 6)]
        for metric, series in values.items()
    }


def dataset_fingerprint(config: dict[str, Any], manifest_path: Path, rows: Iterable[dict[str, str]], assignment: dict[str, str]) -> str:
    """Hash config, manifest, split assignment, and actual reference/prediction files."""
    payload: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        payload.append({
            "sample_id": row["sample_id"],
            "incident_id": row["incident_id"],
            "split": assignment[row[config["split"]["group_key"]]],
            "domain": row["domain"],
            "mask_sha256": sha256_file(resolve_path(row["mask_path"], manifest_path)),
            "prediction_sha256": sha256_file(resolve_path(row["prediction_path"], manifest_path)),
            "annotation_version": row["annotation_version"],
        })
    canonical = json.dumps({"config": config, "samples": payload}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + sha256_bytes(canonical)


def write_split_manifest(rows: list[dict[str, str]], assignment: dict[str, str], config: dict[str, Any], output_dir: Path) -> None:
    group_key = config["split"]["group_key"]
    fields = list(rows[0].keys()) + ["benchmark_split"]
    with (output_dir / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "benchmark_split": assignment[row[group_key]]})


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.manifest is not None:
        # The sealed manifest may exist only on an approved self-hosted runner.
        # Its immutable path is intentionally not committed to the repository.
        config["manifest_csv"] = str(args.manifest.resolve())
    manifest_path, rows = load_manifest(config, config_path)
    assignment = assign_incident_disjoint_splits(rows, config)
    output_dir = (ROOT / config["output_directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(rows, assignment, config, output_dir)

    sealed_name = config["split"]["sealed_holdout_name"]
    sealed_rows = [row for row in rows if assignment[row[config["split"]["group_key"]]] == sealed_name]
    groups_by_domain: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(empty_confusion))
    samples_by_domain: dict[str, int] = defaultdict(int)
    skipped_samples: list[str] = []
    for row in sealed_rows:
        confusion = confusion_for_sample(row, manifest_path, config["mask_processing"])
        if confusion is None:
            skipped_samples.append(row["sample_id"])
            continue
        domain = row["domain"]
        group = row[config["split"]["group_key"]]
        groups_by_domain[domain][group] = add_confusion(groups_by_domain[domain][group], confusion)
        samples_by_domain[domain] += 1

    intended_domains = list(config["intended_domains"])
    per_domain: dict[str, dict[str, Any]] = {}
    evidence = []
    for domain in intended_domains:
        group_confusions = list(groups_by_domain[domain].values())
        aggregate = empty_confusion()
        for confusion in group_confusions:
            aggregate = add_confusion(aggregate, confusion)
        metrics = metrics_from_confusion(aggregate)
        intervals = bootstrap_intervals(
            group_confusions,
            seed_key=f"{config['seed']}:{domain}",
            replicates=int(config["bootstrap"]["replicates"]),
            confidence_level=float(config["bootstrap"]["confidence_level"]),
        )
        sample_count = samples_by_domain[domain]
        per_domain[domain] = {
            "sealed_sample_count": sample_count,
            "sealed_incident_count": len(group_confusions),
            "confusion": aggregate,
            "metrics": {name: round(value, 6) for name, value in metrics.items()},
            "bootstrap_confidence_intervals": intervals,
        }
        evidence.append((domain, EvaluationSplit.SEALED_HOLDOUT, sample_count, metrics))

    decision = evaluate_promotion_eligibility(evidence, intended_domains)
    fingerprint = dataset_fingerprint(config, manifest_path, sealed_rows, assignment)
    api_payloads = [
        {
            "dataset_fingerprint": fingerprint,
            "split": "sealed_holdout",
            "domain": domain,
            "sample_count": report["sealed_sample_count"],
            "metrics": {metric: report["metrics"][metric] for metric in ("oil_f1", "oil_iou", "oil_precision", "oil_recall")},
            "jepa_backbone": args.jepa_backbone,
            "reviewer": args.reviewer,
            "notes": args.notes or "Generated by incident-disjoint benchmark; review split manifest and confidence intervals before recording.",
        }
        for domain, report in per_domain.items()
    ]
    report = {
        "benchmark_name": config["benchmark_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "model_version": args.model_version,
        "reviewer": args.reviewer,
        "jepa_backbone": args.jepa_backbone,
        "dataset_fingerprint": fingerprint,
        "manifest": str(manifest_path),
        "split_policy": config["split"],
        "promotion_gate": config["promotion_gate"],
        "per_domain": per_domain,
        "promotion_eligible": decision.eligible,
        "promotion_reasons": decision.reasons,
        "skipped_samples": skipped_samples,
        "warning": "A report is evidence only. Do not register it if the dataset provenance, labels, model artifact, or sealed-split process has not been independently reviewed.",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "api_evaluation_payloads.json").write_text(json.dumps(api_payloads, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(output_dir / "report.json"),
        "split_manifest": str(output_dir / "split_manifest.csv"),
        "api_payloads": str(output_dir / "api_evaluation_payloads.json"),
        "dataset_fingerprint": fingerprint,
        "promotion_eligible": decision.eligible,
        "promotion_reasons": decision.reasons,
    }, indent=2))
    return 0 if decision.eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
