#!/usr/bin/env python3
"""Create a governed MineralVision admission record from local OctoSense manifests.

This tool does not download MCAP files, call Hugging Face, run FiftyOne, or
train a model. Its output is a reviewer-facing manifest and provenance record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.api.innovations.octosense.manifest import (
    APPROVED_PURPOSES,
    OctoSenseManifestError,
    admit_octosense_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path, help="local OctoSense samples.json")
    parser.add_argument("--metadata", required=True, type=Path, help="local OctoSense metadata.json")
    parser.add_argument(
        "--dataset-revision",
        required=True,
        help="approved immutable Hugging Face repository revision SHA",
    )
    parser.add_argument(
        "--purpose",
        required=True,
        choices=sorted(APPROVED_PURPOSES),
        help="approved use; domain-specific oil/mineral performance claims are intentionally unavailable",
    )
    parser.add_argument("--expected-samples-sha256", help="optional approved SHA-256 for samples.json")
    parser.add_argument("--expected-metadata-sha256", help="optional approved SHA-256 for metadata.json")
    parser.add_argument("--output", required=True, type=Path, help="destination JSON admission record")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        admission = admit_octosense_manifest(
            samples_path=args.samples,
            metadata_path=args.metadata,
            dataset_revision=args.dataset_revision,
            purpose=args.purpose,
            expected_samples_sha256=args.expected_samples_sha256,
            expected_metadata_sha256=args.expected_metadata_sha256,
        )
    except OctoSenseManifestError as exc:
        print(f"OctoSense manifest admission rejected: {exc}", file=sys.stderr)
        return 2
    output = args.output.resolve()
    if output.exists() and output.is_symlink():
        print("OctoSense manifest admission rejected: output must not be a symlink", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": admission["admission_state"],
                "episode_count": admission["quality_report"]["episode_count"],
                "lineage_hash": admission["lineage_hash"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
