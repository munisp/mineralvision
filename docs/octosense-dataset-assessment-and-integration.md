# OctoSense Dataset Assessment and Governed MineralVision Integration

**Prepared by Manus AI**  
**Date:** 22 August 2026

## Executive Assessment

`Voxel51/OctoSense` is an openly accessible Hugging Face dataset package for a calibrated, time-synchronized, multimodal robotics sensor rig. The source describes representative MCAP episodes from car, boat, and Unitree quadruped platforms, with RGB/stereo, infrared, event-camera, LiDAR, IMU, GPS, calibration, and frame-transform information. The source repository metadata currently asserts the MIT license and records revision `0ff6263bdaf0693ea039016398624501c4812804`. [1] [2]

MineralVision now includes a **local, manifest-first, governed admission adapter** for the dataset. It can validate source metadata, produce a normalized sensor-episode inventory, calculate deterministic provenance, report manifest-level sensor/GPS/calibration conditions, and construct a payload for the existing tenant-bound evidence layer. It does **not** download raw MCAP files, execute third-party code, train a model, expose a raw-data endpoint, or make a performance claim.

> **Scope boundary:** OctoSense is appropriate for generic multimodal ingestion, calibration/synchronization, modality-dropout, and sensor-fusion pipeline validation. It is **not** valid evidence for oil-spill segmentation/detection accuracy, mineral prospectivity accuracy, resource-estimation accuracy, or environmental incident performance. The available package is a robotics perception sample set, not a domain-labelled oil-spill or geology benchmark.

## Source Review

The source dataset page identifies OctoSense as a FiftyOne multimodal MCAP dataset. Its manifest defines fields for episode duration, topic/schema lists, LiDAR/RGB/IMU/GPS/event counts, calibration identifiers, sensor dropout, platform, and optional operational attributes. The eight representative manifest entries include boat, car, and Unitree runs. Car episode metadata declares depth, flow, semantic, caption, radar, and odometry topics as well as a `has_seg` boolean; that fact alone does not establish any oil-spill, mineral, or environmental ground-truth label. [1] [3] [4]

The companion paper describes the broader robotics collection as time-synchronized multimodal data spanning differing sample rates, latencies, noise, nighttime, and degraded-sensor settings. It evaluates generic robotics tasks such as optical flow, depth, semantic segmentation, and ego-motion. [5] This supports sensor-pipeline robustness work, but it does not change the domain limitation above.

The Hugging Face dataset viewer currently reports a `StreamingRowsError` / `CastError` for the FiftyOne JSON structure. The adapter therefore reads approved local `samples.json` and `metadata.json` directly rather than relying on the viewer or `datasets.load_dataset`. The viewer failure is a packaging/compatibility issue and should not be treated as a raw-asset quality conclusion. [1]

| MineralVision use | Status | Basis and control |
|---|---|---|
| Sensor-ingestion contract validation | **Allowed** | Validates local manifests, declared modalities, MCAP paths, episode uniqueness, and data-quality metadata. |
| Calibration and synchronization pipeline validation | **Allowed** | Records calibration IDs, transform topics, and modality coverage; it does not measure calibration error without raw asset analysis. |
| Sensor-fusion and modality-dropout evaluation | **Allowed** | Supports non-domain-specific robustness evaluation after a reviewed asset acquisition process. |
| Multimodal representation pretraining | **Conditionally allowed** | Requires data-governance and license review, approved raw-asset acquisition, and a separate model-card/evaluation plan. |
| Oil-spill segmentation or detection accuracy | **Blocked** | No oil-spill imagery/labels or suitable domain holdout are established by the source manifests. |
| Mineral prospectivity or resource-estimation accuracy | **Blocked** | No geological/mineral target labels or ground-truth evaluation design are established by the source. |
| Production autonomous decisioning | **Blocked by this integration** | The adapter generates evidence/provenance only; no prediction or operational write-back is performed. |

## Implemented Components

| Component | Location | Function |
|---|---|---|
| Manifest adapter | `MineralVision_Final_Package/src/api/innovations/octosense/manifest.py` | Local-only manifest validation, normalization, provenance hashing, quality reporting, and governed evidence contract creation. |
| Package boundary | `MineralVision_Final_Package/src/api/innovations/octosense/__init__.py` | Exposes the restricted public Python interface. |
| Admission CLI | `scripts/ingest_octosense_manifest.py` | Writes a reviewed JSON admission record from local manifests only. |
| Evidence source allow-list | `src/api/innovations/integration_hub/governed.py` | Adds `huggingface_dataset` as a tenant-bound evidence source. |
| Regression suite | `tests/critical/test_octosense_manifest_integration.py` | Covers admission, deterministic lineage, purpose blocking, path/platform validation, GPS checks, and evidence persistence. |
| Fixtures | `tests/fixtures/octosense/` | Compact, non-raw-MCAP representative metadata/samples for deterministic tests. |

## Admission Controls

The adapter fails closed when required conditions are not met. It accepts only an existing local regular file for each manifest, rejects symbolic links, caps each manifest at 2 MiB, requires UTF-8 JSON objects, and never makes a network request. It validates that metadata identify the expected `octosense` multimodal collection and declares every field the adapter needs.

Each normalized episode must have a safe relative path under `data/` ending in `.mcap`, a known platform (`boat`, `car`, or `unitree`), a unique bag ID and asset path, ISO-8601 start time, non-negative counts, valid calibration IDs, and non-inverted paired GPS bounds. The string value `nan` in source split/dropout metadata is normalized to **`unassigned`** or an absent dropout declaration, respectively. This avoids accidentally treating an unspecified source split as training or test data.

The resulting admission captures the source URL, immutable revision selected by the reviewer, license assertion, SHA-256 of both local manifests, source paths, normalized episode IDs, approved purpose, and a deterministic SHA-256 lineage hash. Raw MCAP asset hashes remain explicitly unverified because the adapter does not fetch or inspect those assets.

| Control | Behavior |
|---|---|
| Revision pinning | Requires a 7–64 character lower-case git SHA; reviewers should use the documented source revision or another explicitly approved immutable revision. |
| Manifest integrity | Optional expected SHA-256 values fail the command when the locally supplied manifest differs. |
| Purpose allow-list | Permits only `sensor_ingestion_validation`, `sensor_fusion_evaluation`, `calibration_and_synchronization_validation`, or `multimodal_representation_pretraining`. |
| Domain guard | Rejects purposes containing oil, spill, mineral, prospect, geology, hydrocarbon, or environmental terms. |
| Source safety | No automatic download, no remote URLs accepted as file arguments, no dataset-library execution, and no MCAP parser invocation. |
| Tenant governance | `build_governed_evidence_request` emits the contract for tenant-bound `register_evidence`; persisting it remains subject to existing integration-hub controls. |
| Raw asset restrictions | A separate approved acquisition process must verify asset checksums, retention, access control, and any redistribution decision. |

## Runbook

First obtain `samples.json` and `metadata.json` through an approved review process, record the source revision and terms, and store them locally. Do not download or redistribute MCAP assets merely by running the command below.

```bash
cd /path/to/mineralvision-repo
export PYTHONPATH="$PWD/MineralVision_Final_Package:$PYTHONPATH"

python scripts/ingest_octosense_manifest.py \
  --samples /approved/octosense/samples.json \
  --metadata /approved/octosense/metadata.json \
  --dataset-revision 0ff6263bdaf0693ea039016398624501c4812804 \
  --purpose sensor_fusion_evaluation \
  --output /approved/octosense/mineralvision-admission.json
```

Where reviewers have recorded expected source-manifest content hashes, add them to prevent silent drift:

```bash
python scripts/ingest_octosense_manifest.py \
  --samples /approved/octosense/samples.json \
  --metadata /approved/octosense/metadata.json \
  --dataset-revision 0ff6263bdaf0693ea039016398624501c4812804 \
  --purpose calibration_and_synchronization_validation \
  --expected-samples-sha256 <64-lowercase-hex-characters> \
  --expected-metadata-sha256 <64-lowercase-hex-characters> \
  --output /approved/octosense/mineralvision-admission.json
```

The resulting admission is **pending data-governance review**, not an authorization to train, benchmark, distribute, or operationalize a model. A reviewer may translate it into a tenant-bound evidence record using `build_governed_evidence_request(...)` and the existing `register_evidence(...)` service. The generated evidence contains no raw MCAP content, inferred geometry, or performance result.

## Validation

Run the focused regression suite in the locked CI environment:

```bash
export PYTHONPATH="$PWD/MineralVision_Final_Package:$PYTHONPATH"
.venv-ci/bin/python -m pytest -q \
  tests/critical/test_octosense_manifest_integration.py
```

The suite verifies a three-platform representative manifest, deterministic provenance, expected-hash enforcement, prohibited oil-spill purpose rejection, duplicate path rejection, unknown-platform rejection, non-MCAP rejection, GPS count consistency, and tenant-bound evidence persistence. It is intentionally independent of raw MCAP download, FiftyOne, Hugging Face viewer health, and external network availability.

## Limitations and Required Next Gates

The adapter validates declared manifest metadata, not raw observation content. It does not prove calibration correctness, clock alignment, GPS correctness, label quality, data ownership beyond the source assertion, completeness of the eight representative episodes, or model generalization. The MIT value is recorded as a source **assertion** and must be revalidated when the source revision is acquired or redistributed.

Before any pretraining or raw MCAP processing, the operating team should complete a dataset intake review, verify the exact source revision and license, acquire files through an approved path, establish per-asset checksums, scan/parse MCAP using a constrained environment, classify GPS/location sensitivity, define retention/access controls, and create an evaluation protocol with appropriate task labels and domain holdouts. Any oil-spill or mineral performance promotion must use separately governed, domain-valid evaluation data.

## References

[1] [Voxel51/OctoSense dataset page and current viewer status](https://huggingface.co/datasets/Voxel51/OctoSense)

[2] [Hugging Face OctoSense dataset API metadata](https://huggingface.co/api/datasets/Voxel51/OctoSense)

[3] [Hugging Face OctoSense repository file listing](https://huggingface.co/api/datasets/Voxel51/OctoSense/tree/main)

[4] [OctoSense samples manifest](https://huggingface.co/datasets/Voxel51/OctoSense/raw/main/samples.json)

[5] [OctoSense paper abstract, arXiv:2606.27317](https://arxiv.org/abs/2606.27317)
