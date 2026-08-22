"""Fail-closed manifest admission for the Voxel51/OctoSense dataset.

The adapter is intentionally manifest-first. Raw MCAP files remain external,
are never downloaded here, and are not interpreted as MineralVision training or
accuracy evidence for oil-spill or mineral-domain models.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

DATASET_ID = "Voxel51/OctoSense"
DATASET_URL = "https://huggingface.co/datasets/Voxel51/OctoSense"
DOCUMENTED_REVISION = "0ff6263bdaf0693ea039016398624501c4812804"
LICENSE_ASSERTION = "MIT"
SCHEMA_VERSION = "mineralvision.octosense.admission.v1"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024

APPROVED_PURPOSES = frozenset(
    {
        "sensor_ingestion_validation",
        "sensor_fusion_evaluation",
        "calibration_and_synchronization_validation",
        "multimodal_representation_pretraining",
    }
)
PROHIBITED_PURPOSE_TOKENS = frozenset(
    {
        "oil",
        "spill",
        "mineral",
        "prospect",
        "geolog",
        "hydrocarbon",
        "environmental",
    }
)
KNOWN_PLATFORMS = frozenset({"boat", "car", "unitree"})
KNOWN_SPLITS = frozenset({"train", "test", "validation", "unassigned"})
REQUIRED_SAMPLE_FIELD_NAMES = frozenset(
    {
        "filepath",
        "duration_s",
        "message_count",
        "channel_count",
        "topics",
        "schemas",
        "session",
        "start_time",
        "split",
        "n_lidar_frames",
        "n_rgb_frames",
        "n_imu_samples",
        "n_gps_fixes",
        "n_gps_valid",
        "gps_quality",
        "rgb_cal_id",
        "imu_cal_id",
        "lidar_cal_id",
        "n_events_left",
        "n_events_right",
        "sensor_dropout",
        "platform",
        "bag_id",
    }
)


class OctoSenseManifestError(ValueError):
    """Raised when an OctoSense manifest fails a governance or schema check."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _require_local_manifest(path: Path, label: str) -> tuple[Path, bytes]:
    if not path.exists() or not path.is_file():
        raise OctoSenseManifestError(f"{label} must be an existing local regular file")
    if path.is_symlink():
        raise OctoSenseManifestError(f"{label} must not be a symlink")
    size = path.stat().st_size
    if size <= 0:
        raise OctoSenseManifestError(f"{label} must not be empty")
    if size > MAX_MANIFEST_BYTES:
        raise OctoSenseManifestError(
            f"{label} is {size} bytes; manifest admission limit is {MAX_MANIFEST_BYTES} bytes"
        )
    return path.resolve(), path.read_bytes()


def _load_json_object(path: Path, label: str) -> tuple[Path, bytes, Dict[str, Any]]:
    resolved, raw = _require_local_manifest(path, label)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OctoSenseManifestError(f"{label} must contain UTF-8 JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise OctoSenseManifestError(f"{label} top level must be a JSON object")
    return resolved, raw, decoded


def _normalize_string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise OctoSenseManifestError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise OctoSenseManifestError(f"{field} must not be blank")
    return normalized


def _normalize_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OctoSenseManifestError(f"{field} must be a non-negative integer")
    return value


def _normalize_number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OctoSenseManifestError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise OctoSenseManifestError(f"{field} must be a finite number >= {minimum}")
    return number


def _normalize_optional_string(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    normalized = _normalize_string(value, field, allow_empty=True)
    return None if normalized.lower() in {"", "nan", "none", "null"} else normalized


def _normalize_split(value: Any) -> str:
    normalized = _normalize_optional_string(value, "split")
    split = "unassigned" if normalized is None else normalized.lower()
    if split not in KNOWN_SPLITS:
        raise OctoSenseManifestError(
            f"split {split!r} is unsupported; allowed: {sorted(KNOWN_SPLITS)}"
        )
    return split


def _validate_revision(revision: str) -> str:
    normalized = _normalize_string(revision, "dataset_revision")
    if not re.fullmatch(r"[0-9a-f]{7,64}", normalized):
        raise OctoSenseManifestError("dataset_revision must be a lowercase git SHA of 7-64 hex characters")
    return normalized


def _validate_expected_digest(value: Optional[str], observed: str, label: str) -> None:
    if value is None:
        return
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise OctoSenseManifestError(f"expected {label} SHA-256 must be 64 lowercase hex characters")
    if value != observed:
        raise OctoSenseManifestError(
            f"{label} SHA-256 mismatch: expected {value}, observed {observed}"
        )


def _validate_purpose(purpose: str) -> str:
    normalized = _normalize_string(purpose, "purpose").lower()
    if any(token in normalized for token in PROHIBITED_PURPOSE_TOKENS):
        raise OctoSenseManifestError(
            "OctoSense is not admitted for oil-spill, mineral, geological, hydrocarbon, or environmental model claims"
        )
    if normalized not in APPROVED_PURPOSES:
        raise OctoSenseManifestError(
            f"purpose {normalized!r} is not approved; allowed: {sorted(APPROVED_PURPOSES)}"
        )
    return normalized


def _validate_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    name = _normalize_string(metadata.get("name"), "metadata.name").lower()
    slug = _normalize_string(metadata.get("slug"), "metadata.slug").lower()
    media_type = _normalize_string(metadata.get("media_type"), "metadata.media_type").lower()
    if name != "octosense" or slug != "octosense":
        raise OctoSenseManifestError("metadata name and slug must both be 'octosense'")
    if media_type != "multimodal":
        raise OctoSenseManifestError("metadata.media_type must be 'multimodal'")
    raw_fields = metadata.get("sample_fields")
    if not isinstance(raw_fields, list):
        raise OctoSenseManifestError("metadata.sample_fields must be a list")
    field_names = {
        entry.get("name")
        for entry in raw_fields
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str)
    }
    missing = sorted(REQUIRED_SAMPLE_FIELD_NAMES - field_names)
    if missing:
        raise OctoSenseManifestError(f"metadata.sample_fields missing required names: {missing}")
    return {
        "name": name,
        "slug": slug,
        "media_type": media_type,
        "declared_field_count": len(field_names),
        "dataset_version": _normalize_optional_string(metadata.get("version"), "metadata.version"),
        "created_at": _nested_date(metadata.get("created_at")),
        "last_modified_at": _nested_date(metadata.get("last_modified_at")),
    }


def _nested_date(value: Any) -> Optional[str]:
    if not isinstance(value, Mapping):
        return None
    date = value.get("$date")
    return date.strip() if isinstance(date, str) and date.strip() else None


def _validate_topics(value: Any, field: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise OctoSenseManifestError(f"{field} must be a non-empty list")
    topics = [_normalize_string(item, f"{field}[]") for item in value]
    if len(set(topics)) != len(topics):
        raise OctoSenseManifestError(f"{field} must not contain duplicate topics")
    return sorted(topics)


def _validate_schema_names(value: Any, field: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise OctoSenseManifestError(f"{field} must be a non-empty list")
    schemas = [_normalize_string(item, f"{field}[]") for item in value]
    return sorted(set(schemas))


def _validate_relative_mcap_path(value: Any) -> str:
    filepath = _normalize_string(value, "filepath")
    path = PurePosixPath(filepath)
    if path.is_absolute() or ".." in path.parts or not filepath.startswith("data/"):
        raise OctoSenseManifestError("filepath must be a safe relative path under data/")
    if path.suffix.lower() != ".mcap":
        raise OctoSenseManifestError("filepath must reference an .mcap asset")
    return path.as_posix()


def _sensor_modalities(topics: Sequence[str]) -> List[str]:
    joined = "\n".join(topics).lower()
    modalities = []
    if "/lidar/" in joined:
        modalities.append("lidar")
    if "/camera/" in joined:
        modalities.append("camera")
    if "/camera/infrared/" in joined:
        modalities.append("thermal_or_infrared_camera")
    if "/event/" in joined:
        modalities.append("event_camera")
    if "/imu/" in joined:
        modalities.append("imu")
    if "/gps" in joined:
        modalities.append("gps")
    if "/tf" in joined:
        modalities.append("transform_frames")
    if "/odom" in joined:
        modalities.append("odometry")
    return modalities


def _normalize_episode(sample: Mapping[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise OctoSenseManifestError(f"samples[{index}] must be an object")
    filepath = _validate_relative_mcap_path(sample.get("filepath"))
    platform = _normalize_string(sample.get("platform"), f"samples[{index}].platform").lower()
    if platform not in KNOWN_PLATFORMS:
        raise OctoSenseManifestError(
            f"samples[{index}].platform {platform!r} is unsupported; allowed: {sorted(KNOWN_PLATFORMS)}"
        )
    bag_id = _normalize_string(sample.get("bag_id"), f"samples[{index}].bag_id")
    session = _normalize_string(sample.get("session"), f"samples[{index}].session")
    start_time = _normalize_string(sample.get("start_time"), f"samples[{index}].start_time")
    try:
        datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OctoSenseManifestError(f"samples[{index}].start_time must be ISO-8601") from exc
    topics = _validate_topics(sample.get("topics"), f"samples[{index}].topics")
    schemas = _validate_schema_names(sample.get("schemas"), f"samples[{index}].schemas")
    n_gps_fixes = _normalize_int(sample.get("n_gps_fixes"), f"samples[{index}].n_gps_fixes")
    n_gps_valid = _normalize_number(sample.get("n_gps_valid"), f"samples[{index}].n_gps_valid")
    if n_gps_valid > n_gps_fixes:
        raise OctoSenseManifestError(f"samples[{index}].n_gps_valid must not exceed n_gps_fixes")
    counts = {
        "lidar_frames": _normalize_int(sample.get("n_lidar_frames"), f"samples[{index}].n_lidar_frames"),
        "rgb_frames": _normalize_int(sample.get("n_rgb_frames"), f"samples[{index}].n_rgb_frames"),
        "imu_samples": _normalize_int(sample.get("n_imu_samples"), f"samples[{index}].n_imu_samples"),
        "gps_fixes": n_gps_fixes,
        "gps_valid": n_gps_valid,
        "event_left": _normalize_int(sample.get("n_events_left"), f"samples[{index}].n_events_left"),
        "event_right": _normalize_int(sample.get("n_events_right"), f"samples[{index}].n_events_right"),
    }
    calibration_ids = {
        "rgb": _normalize_string(sample.get("rgb_cal_id"), f"samples[{index}].rgb_cal_id"),
        "imu": _normalize_string(sample.get("imu_cal_id"), f"samples[{index}].imu_cal_id"),
        "lidar": _normalize_string(sample.get("lidar_cal_id"), f"samples[{index}].lidar_cal_id"),
    }
    optional_bounds = {}
    for key in ("gps_lat_min", "gps_lat_max", "gps_lon_min", "gps_lon_max"):
        value = sample.get(key)
        optional_bounds[key] = None if value is None else _normalize_number(value, f"samples[{index}].{key}", minimum=-1e9)
    if optional_bounds["gps_lat_min"] is not None and optional_bounds["gps_lat_min"] > optional_bounds["gps_lat_max"]:
        raise OctoSenseManifestError(f"samples[{index}] GPS latitude bounds are inverted")
    if optional_bounds["gps_lon_min"] is not None and optional_bounds["gps_lon_min"] > optional_bounds["gps_lon_max"]:
        raise OctoSenseManifestError(f"samples[{index}] GPS longitude bounds are inverted")
    if (optional_bounds["gps_lat_min"] is None) != (optional_bounds["gps_lat_max"] is None):
        raise OctoSenseManifestError(f"samples[{index}] GPS latitude bounds must be paired")
    if (optional_bounds["gps_lon_min"] is None) != (optional_bounds["gps_lon_max"] is None):
        raise OctoSenseManifestError(f"samples[{index}] GPS longitude bounds must be paired")

    sensor_dropout = _normalize_optional_string(sample.get("sensor_dropout"), f"samples[{index}].sensor_dropout")
    return {
        "episode_id": f"octosense:{bag_id}",
        "bag_id": bag_id,
        "asset": {
            "relative_path": filepath,
            "format": "mcap",
            "content_sha256": None,
            "integrity_status": "not_downloaded_or_verified_by_manifest_adapter",
        },
        "platform": platform,
        "session": session,
        "start_time": start_time,
        "split": _normalize_split(sample.get("split")),
        "duration_s": _normalize_number(sample.get("duration_s"), f"samples[{index}].duration_s", minimum=0.001),
        "message_count": _normalize_int(sample.get("message_count"), f"samples[{index}].message_count"),
        "channel_count": _normalize_int(sample.get("channel_count"), f"samples[{index}].channel_count"),
        "topics": topics,
        "schemas": schemas,
        "sensor_modalities": _sensor_modalities(topics),
        "counts": counts,
        "gps": {
            "quality": _normalize_string(sample.get("gps_quality"), f"samples[{index}].gps_quality"),
            "bounds": optional_bounds,
        },
        "calibration_ids": calibration_ids,
        "sensor_dropout": sensor_dropout,
        "has_segmentation_annotation": sample.get("has_seg") is True,
        "degraded": sample.get("degraded") is True,
    }


def _validate_unique(episodes: Iterable[Mapping[str, Any]], key: str) -> None:
    values = [episode[key] for episode in episodes]
    duplicates = sorted({value for value, count in Counter(values).items() if count > 1})
    if duplicates:
        raise OctoSenseManifestError(f"duplicate {key} values: {duplicates}")


def _quality_report(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    modality_coverage = Counter()
    platform_distribution = Counter()
    split_distribution = Counter()
    gps_quality_distribution = Counter()
    warnings = []
    for episode in episodes:
        platform_distribution[episode["platform"]] += 1
        split_distribution[episode["split"]] += 1
        gps_quality_distribution[episode["gps"]["quality"]] += 1
        modality_coverage.update(episode["sensor_modalities"])
        if episode["split"] == "unassigned":
            warnings.append({"episode_id": episode["episode_id"], "code": "unassigned_split"})
        if episode["counts"]["gps_fixes"] and not episode["counts"]["gps_valid"]:
            warnings.append({"episode_id": episode["episode_id"], "code": "no_valid_gps_fix"})
        if episode["sensor_dropout"] is not None:
            warnings.append({"episode_id": episode["episode_id"], "code": "sensor_dropout_declared"})
        if episode["degraded"]:
            warnings.append({"episode_id": episode["episode_id"], "code": "degraded_episode"})
    return {
        "episode_count": len(episodes),
        "platform_distribution": dict(sorted(platform_distribution.items())),
        "split_distribution": dict(sorted(split_distribution.items())),
        "gps_quality_distribution": dict(sorted(gps_quality_distribution.items())),
        "sensor_modality_coverage": dict(sorted(modality_coverage.items())),
        "calibration_complete_episode_count": sum(
            all(episode["calibration_ids"].values()) for episode in episodes
        ),
        "segmentation_annotation_episode_count": sum(
            episode["has_segmentation_annotation"] for episode in episodes
        ),
        "declared_issue_count": len(warnings),
        "declared_issues": warnings,
        "interpretation": (
            "Manifest-level quality metadata only; this report does not measure model accuracy, "
            "ground-truth quality, temporal synchronization error, calibration accuracy, or raw-asset integrity."
        ),
    }


def admit_octosense_manifest(
    *,
    samples_path: Path | str,
    metadata_path: Path | str,
    dataset_revision: str,
    purpose: str,
    expected_samples_sha256: Optional[str] = None,
    expected_metadata_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate local source manifests and create a normalized governed admission.

    The caller is responsible for separately approving any MCAP download and
    verifying raw-asset checksums. This function intentionally has no network
    capability and never loads a remote dataset library or MCAP payload.
    """
    revision = _validate_revision(dataset_revision)
    approved_purpose = _validate_purpose(purpose)
    samples_file, samples_raw, samples_document = _load_json_object(Path(samples_path), "samples manifest")
    metadata_file, metadata_raw, metadata_document = _load_json_object(Path(metadata_path), "metadata manifest")
    samples_sha256 = _sha256_bytes(samples_raw)
    metadata_sha256 = _sha256_bytes(metadata_raw)
    _validate_expected_digest(expected_samples_sha256, samples_sha256, "samples manifest")
    _validate_expected_digest(expected_metadata_sha256, metadata_sha256, "metadata manifest")
    metadata_summary = _validate_metadata(metadata_document)
    raw_samples = samples_document.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise OctoSenseManifestError("samples manifest must contain a non-empty 'samples' list")
    episodes = [_normalize_episode(sample, index) for index, sample in enumerate(raw_samples)]
    _validate_unique(episodes, "episode_id")
    _validate_unique([{"filepath": item["asset"]["relative_path"]} for item in episodes], "filepath")
    quality_report = _quality_report(episodes)
    manifest_lineage = {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "dataset_revision": revision,
        "license_assertion": LICENSE_ASSERTION,
        "samples_manifest_sha256": samples_sha256,
        "metadata_manifest_sha256": metadata_sha256,
        "purpose": approved_purpose,
        "episode_ids": [episode["episode_id"] for episode in episodes],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "admission_state": "manifest_validated_pending_data_governance_review",
        "dataset": {
            "id": DATASET_ID,
            "url": DATASET_URL,
            "revision": revision,
            "license_assertion": LICENSE_ASSERTION,
            "source_format": "fiftyone_multimodal_mcap_manifest",
            "source_viewer_status": "not_used_by_adapter; raw manifests parsed directly",
            "metadata": metadata_summary,
        },
        "source_manifests": {
            "samples": {
                "path": str(samples_file),
                "sha256": samples_sha256,
                "bytes": len(samples_raw),
            },
            "metadata": {
                "path": str(metadata_file),
                "sha256": metadata_sha256,
                "bytes": len(metadata_raw),
            },
        },
        "governance": {
            "intended_purpose": approved_purpose,
            "allowed_purposes": sorted(APPROVED_PURPOSES),
            "prohibited_validation_claims": [
                "oil_spill_segmentation_accuracy",
                "oil_spill_detection_accuracy",
                "mineral_prospectivity_accuracy",
                "mineral_resource_estimation_accuracy",
                "environmental_incident_detection_accuracy",
            ],
            "raw_asset_policy": "MCAP assets are not downloaded, executed, redistributed, or integrity-verified by this adapter.",
            "license_policy": "MIT license assertion is recorded from the source and must be revalidated before asset download or redistribution.",
            "review_required": True,
        },
        "lineage_hash": _sha256_json(manifest_lineage),
        "episodes": episodes,
        "quality_report": quality_report,
    }


def build_governed_evidence_request(
    admission: Mapping[str, Any],
    *,
    tenant_id: str,
    observed_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Create a minimal payload consumable by integration_hub.register_evidence.

    It contains source manifests and normalized quality information only, not
    MCAP data, geometry inferred from GPS, or any model performance claim.
    """
    normalized_tenant = _normalize_string(tenant_id, "tenant_id")
    if admission.get("schema_version") != SCHEMA_VERSION:
        raise OctoSenseManifestError("admission does not have the expected OctoSense schema version")
    dataset = admission.get("dataset")
    if not isinstance(dataset, Mapping):
        raise OctoSenseManifestError("admission.dataset must be an object")
    if admission.get("admission_state") != "manifest_validated_pending_data_governance_review":
        raise OctoSenseManifestError("admission has an unsupported state")
    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return {
        "tenant_id": normalized_tenant,
        "source_system": "huggingface_dataset",
        "source_ref": DATASET_URL,
        "source_version": str(dataset.get("revision")),
        "observed_at": timestamp.astimezone(timezone.utc),
        "geometry": {},
        "payload": {
            "dataset_admission_schema": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "license_assertion": dataset.get("license_assertion"),
            "lineage_hash": admission.get("lineage_hash"),
            "source_manifests": admission.get("source_manifests"),
            "governance": admission.get("governance"),
            "quality_report": admission.get("quality_report"),
            "episode_count": len(admission.get("episodes", [])),
        },
        "model_run": {
            "kind": "third_party_dataset_manifest_admission",
            "purpose": admission.get("governance", {}).get("intended_purpose"),
            "performance_evidence": False,
            "raw_assets_loaded": False,
        },
    }
