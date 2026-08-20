# PostgreSQL Provisioning and Oil-Spill Promotion Benchmark Runbook

This runbook provisions PostgreSQL, upgrades the MineralVision schema through the oil-spill Alembic revision, formats candidate segmentation predictions, and runs the incident-disjoint benchmark. It creates **evaluation evidence** only. A model must not be described as meeting the 97% target until the generated sealed-holdout evidence has been independently reviewed and the API promotion gate approves it.

> **Security boundary:** Run database provisioning from a privileged, controlled administration session. Do not paste production passwords into source code, shell history, shared notebooks, or CI logs. Use a secret manager for deployed database credentials.

## 1. Provision PostgreSQL Roles and Databases

The commands below target a self-managed Linux host with PostgreSQL installed. For managed PostgreSQL, create the equivalent login role and databases through the provider’s administration interface, then continue at [Alembic migration](#2-run-alembic-migrations).

```bash
# Generate a URL-safe, quote-free password. Store it immediately in your secret manager.
export MV_DB_PASSWORD="$(openssl rand -hex 32)"

# Create the application login and separate application/test databases.
sudo -u postgres psql -v ON_ERROR_STOP=1 \
  -c "CREATE ROLE mineralvision LOGIN PASSWORD '$MV_DB_PASSWORD';"
sudo -u postgres createdb -O mineralvision mineralvision
sudo -u postgres createdb -O mineralvision mineralvision_oil_spill_test

# Optional: verify the login without exposing the password in process arguments.
export PGPASSWORD="$MV_DB_PASSWORD"
psql -h localhost -U mineralvision -d mineralvision \
  -c 'SELECT current_database(), current_user, version();'
unset PGPASSWORD
```

If the role or database already exists, do not re-run the creation commands blindly. Instead, inspect it with `\du mineralvision` and `\l mineralvision`, rotate credentials through the approved process, and ensure the role owns or can migrate the intended schema.

Set a PostgreSQL URL for the application and a distinct one for integration tests. The repository accepts `postgresql://`, `postgresql+psycopg://`, or `postgresql+psycopg2://` URLs; the examples use the installed `psycopg2` driver.

```bash
export DATABASE_URL="postgresql+psycopg2://mineralvision:${MV_DB_PASSWORD}@localhost:5432/mineralvision"
export OIL_SPILL_TEST_DATABASE_URL="postgresql+psycopg2://mineralvision:${MV_DB_PASSWORD}@localhost:5432/mineralvision_oil_spill_test"
```

## 2. Run Alembic Migrations

The runtime does **not** create its own schema. Alembic is the only supported schema-provisioning route.

```bash
cd /path/to/mineralvision/MineralVision_Final_Package

# Install pinned database tooling if it is not already part of your environment.
pip install -r ../requirements.txt

# Inspect current state, apply 0001_initial and 0002_oil_spill_postgres, then confirm.
alembic current
alembic upgrade head
alembic current
```

The target revision must be `0002_oil_spill_postgres`. Verify both the revision and the four oil-spill tables directly.

```bash
export PGPASSWORD="$MV_DB_PASSWORD"
psql -h localhost -U mineralvision -d mineralvision -c 'SELECT version_num FROM alembic_version;'
psql -h localhost -U mineralvision -d mineralvision -c '\dt oil_spill_*'
psql -h localhost -U mineralvision -d mineralvision -c '
  SELECT table_name, column_name, data_type, udt_name
  FROM information_schema.columns
  WHERE table_name IN (
    '\''oil_spill_incidents'\'', '\''oil_spill_models'\'',
    '\''oil_spill_evaluation_runs'\'', '\''oil_spill_incident_events'\''
  )
  ORDER BY table_name, ordinal_position;
'
unset PGPASSWORD
```

Expected oil-spill tables are `oil_spill_incidents`, `oil_spill_models`, `oil_spill_evaluation_runs`, and `oil_spill_incident_events`. Their geospatial evidence, evaluation metrics, intended domains, and incident details use PostgreSQL JSONB fields.

Apply the same migration to the isolated test database before running persistence or governance integration tests.

```bash
export DATABASE_URL="$OIL_SPILL_TEST_DATABASE_URL"
cd /path/to/mineralvision/MineralVision_Final_Package
alembic upgrade head
cd ..
python scripts/verify_oil_spill_persistence.py
python scripts/verify_oil_spill_governance.py
```

## 3. Prepare Candidate Predictions and the Evaluation Manifest

The benchmark intentionally does not invoke the model. Generate an 8-bit grayscale **probability mask** for every image with the candidate artifact first, then list each reference/prediction pair in the manifest. A value of `0` means probability 0.0; `255` means probability 1.0; with the reference configuration, values ≥ `128` are positive predictions because the threshold is 0.50.

Use a directory layout such as the following. The paths in the manifest are resolved relative to the manifest CSV location.

```text
data/oil_spill/
├── benchmark_manifest.csv
├── images/
│   ├── INC-2025-001/frame_00001.jpg
│   └── INC-2025-001/frame_00002.jpg
├── reference_masks/
│   ├── INC-2025-001/frame_00001.png
│   └── INC-2025-001/frame_00002.png
└── candidate_predictions/
    ├── INC-2025-001/frame_00001.png
    └── INC-2025-001/frame_00002.png
```

The required CSV header is shown below. The data rows are **illustrative structure only**, not evaluation evidence. In a real benchmark, include every scored frame and ensure all frames from an incident/flight/video use the same `incident_id`.

```csv
sample_id,incident_id,sequence_id,frame_index,image_path,mask_path,prediction_path,domain,sensor,acquisition_date,sea_state,glint_level,annotation_version
INC-2025-001-F00001,INC-2025-001,FLIGHT-2025-001,1,images/INC-2025-001/frame_00001.jpg,reference_masks/INC-2025-001/frame_00001.png,candidate_predictions/INC-2025-001/frame_00001.png,offshore_rgb_daylight,drone_rgb,2025-05-14,calm,low,oil-mask-v3
INC-2025-001-F00002,INC-2025-001,FLIGHT-2025-001,2,images/INC-2025-001/frame_00002.jpg,reference_masks/INC-2025-001/frame_00002.png,candidate_predictions/INC-2025-001/frame_00002.png,offshore_rgb_daylight,drone_rgb,2025-05-14,calm,low,oil-mask-v3
INC-2025-017-F00001,INC-2025-017,FLIGHT-2025-017,1,images/INC-2025-017/frame_00001.jpg,reference_masks/INC-2025-017/frame_00001.png,candidate_predictions/INC-2025-017/frame_00001.png,offshore_rgb_glint,drone_rgb,2025-06-09,moderate,high,oil-mask-v3
INC-2025-041-F00001,INC-2025-041,PORT-2025-041,1,images/INC-2025-041/frame_00001.jpg,reference_masks/INC-2025-041/frame_00001.png,candidate_predictions/INC-2025-041/frame_00001.png,port_rgb_complex_background,drone_rgb,2025-04-22,calm,medium,oil-mask-v3
INC-2025-079-F00001,INC-2025-079,SAR-2025-079,1,images/INC-2025-079/frame_00001.tif,reference_masks/INC-2025-079/frame_00001.png,candidate_predictions/INC-2025-079/frame_00001.png,offshore_sar,sentinel_1,2025-03-18,rough,na,oil-mask-v3
```

| Manifest field | Requirement |
|---|---|
| `sample_id` | Globally unique, stable sample identifier. |
| `incident_id` | Mandatory leakage-control group. Never reuse an incident across train, validation, and sealed holdout. If a flight includes several incident states, use a composite group ID. |
| `sequence_id`, `frame_index` | Enable temporal audit and reveal whether a video sequence was split incorrectly. |
| `image_path`, `mask_path`, `prediction_path` | Existing files; reference and prediction masks must have identical height and width. |
| `domain` | One of the configured intended domains. Every domain is evaluated separately; the benchmark never pools a passing domain with a weak one. |
| `sensor`, `sea_state`, `glint_level` | Required shift/hard-negative strata. Record `na` where a field does not apply rather than leaving it blank. |
| `annotation_version` | Immutable annotation-release identifier. Change it when masks are corrected. |

The default reference-mask policy marks values `1` and `255` as oil. Update `reference_positive_values` in `configs/oil_spill_incident_disjoint_benchmark.yaml` to match the selected dataset’s label map **before** creating predictions. Do not use an image label map that encodes water or ignored pixels as one of the configured oil values.

## 4. Validate the Configuration and Run the Benchmark

The configuration assigns whole `incident_id` groups deterministically within each domain using SHA-256 of the configured seed and group ID. Its allocation is 60% train, 20% validation, and 20% locked sealed holdout. The sealed split must never be used for training, early stopping, calibration, threshold setting, or model selection.

```bash
cd /path/to/mineralvision

# The validator confirms that YAML thresholds match the actual promotion module.
python benchmarks/validate_oil_spill_benchmark_config.py \
  --config configs/oil_spill_incident_disjoint_benchmark.yaml

# Score an immutable candidate artifact's pre-generated prediction masks.
# --jepa-backbone is provenance only; it identifies the exact encoder/checkpoint used.
python benchmarks/evaluate_oil_spill_promotion.py \
  --config configs/oil_spill_incident_disjoint_benchmark.yaml \
  --model-id offshore-oil-segmentation-jepa \
  --model-version 2026.08.20 \
  --reviewer qualified_reviewer \
  --jepa-backbone vjepa2.1-vit-large-384 \
  --notes 'Candidate evaluation; incident-disjoint data freeze 2026-08-20.'
```

The evaluator writes three artifacts below `artifacts/oil_spill_benchmark/`.

| Artifact | Purpose |
|---|---|
| `split_manifest.csv` | Immutable record of the deterministic train/validation/sealed assignment for every manifest row. |
| `report.json` | Per-domain confusion counts, oil F1/IoU/precision/recall, incident-level bootstrap intervals, fingerprint, and promotion decision. |
| `api_evaluation_payloads.json` | Exact JSON payloads ready for the model-evaluation API after independent review. |

The process exits with code `0` when all domains clear the gate and `2` when the model is not promotion-eligible. A nonzero exit is expected for a candidate that does not meet the criteria; it is a safety result, not an evaluator failure.

## 5. Record Verified Evidence and Request Promotion

Only after an independent reviewer confirms the dataset fingerprint, frozen split manifest, candidate artifact SHA-256, labels, and report may the results be recorded. The API stores the supplied metrics and never recalculates or invents them.

```bash
export API_URL='https://mineralvision.example'
export TOKEN='REPLACE_WITH_SHORT_LIVED_JWT'
export AUTH="Authorization: Bearer $TOKEN"
export MODEL_ID='offshore-oil-segmentation-jepa'
export MODEL_VERSION='2026.08.20'

# Register the candidate first if it is not already in the model registry.
# The artifact SHA-256 must match the immutable locally deployed ONNX/TorchScript file.
export MODEL_SHA256='REPLACE_WITH_REAL_ARTIFACT_SHA256'
curl --fail-with-body -sS -X POST "$API_URL/api/oil-spill/models" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d "{
    \"model_id\": \"$MODEL_ID\",
    \"model_version\": \"$MODEL_VERSION\",
    \"engine\": \"onnx\",
    \"artifact_sha256\": \"$MODEL_SHA256\",
    \"intended_domains\": [\"offshore_rgb_daylight\", \"offshore_rgb_glint\", \"port_rgb_complex_background\", \"offshore_sar\"]
  }"

# Submit each independently reviewed per-domain payload generated by the evaluator.
jq -c '.[]' artifacts/oil_spill_benchmark/api_evaluation_payloads.json | while IFS= read -r payload; do
  curl --fail-with-body -sS -X POST \
    "$API_URL/api/oil-spill/models/$MODEL_ID/$MODEL_VERSION/evaluations" \
    -H "$AUTH" -H 'Content-Type: application/json' \
    -d "$payload"
done

# Inspect the deterministic decision before any approval request.
curl --fail-with-body -sS \
  "$API_URL/api/oil-spill/models/$MODEL_ID/$MODEL_VERSION/promotion" \
  -H "$AUTH" | jq

# Approval returns HTTP 409 unless every intended domain meets the sealed-holdout gate.
curl --fail-with-body -sS -X POST \
  "$API_URL/api/oil-spill/models/$MODEL_ID/$MODEL_VERSION/approve" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"reviewer":"qualified_reviewer","note":"Fingerprint, split manifest, labels, and sealed report independently reviewed."}' | jq
```

A successful approval is still not a permission to automate external response actions. Raw imagery results remain evidence that should be assessed within the responsible organization’s incident-response and regulatory procedures.
