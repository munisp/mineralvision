"""Governed OctoSense manifest admission and sensor-data quality assessment.

This package deliberately accepts only reviewer-provided local manifests. It does
not download MCAP assets, execute third-party dataset code, or provide a public
raw-data ingestion endpoint.
"""

from .manifest import (
    APPROVED_PURPOSES,
    DATASET_ID,
    DATASET_URL,
    OctoSenseManifestError,
    admit_octosense_manifest,
    build_governed_evidence_request,
)

__all__ = [
    "APPROVED_PURPOSES",
    "DATASET_ID",
    "DATASET_URL",
    "OctoSenseManifestError",
    "admit_octosense_manifest",
    "build_governed_evidence_request",
]
