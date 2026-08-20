# Raw Aerial Imagery Inference with a Custom ONNX Model

This is a copy-paste testing sequence for the secured MineralVision oil-spill API. It assumes that PostgreSQL has been migrated with `alembic upgrade head`, the API is reachable over HTTPS, and the caller has a valid short-lived JWT.

> **Important:** `POST /api/oil-spill/analyze/image` will return an error until the server’s local ONNX artifact is hash-verified, registered, and approved after sealed evaluation evidence is recorded for **every** intended domain. The endpoint creates reviewable evidence only; it does not notify authorities, launch drones, or authorize cleanup.

## 1. Set Client Variables

```bash
export API_URL='https://mineralvision.example'
export TOKEN='REPLACE_WITH_SHORT_LIVED_JWT'
export AUTH="Authorization: Bearer $TOKEN"

export MODEL_ID='oil-segmentation-aerial'
export MODEL_VERSION='2026.08.20'
export MODEL_PATH='/opt/mineralvision/models/oil-spill-segmentation-v1.onnx'
export MODEL_SHA256="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"

# Local aerial image to submit after approval.
export AERIAL_IMAGE='./aerial_frame.jpg'
```

The model file must be present on the **API server**, not merely on the curl client. The required ONNX input is normalized RGB `float32` in `[0,1]`, shape `[1,3,H,W]`. The output must be single-channel or multiclass logits as documented in the [model and API integration guide](model-and-api-integration.md).

## 2. Configure the API Server’s Local Model Environment

Run these commands in the API service environment, then restart or redeploy the API process using the normal deployment procedure.

```bash
export DATABASE_URL='postgresql://mineralvision:REPLACE_ME@db.example:5432/mineralvision'
export OIL_SPILL_MODEL_PATH='/opt/mineralvision/models/oil-spill-segmentation-v1.onnx'
export OIL_SPILL_MODEL_ENGINE='onnx'
export OIL_SPILL_MODEL_ID='oil-segmentation-aerial'
export OIL_SPILL_MODEL_VERSION='2026.08.20'
export OIL_SPILL_MODEL_SHA256='REPLACE_WITH_THE_SHA256_FROM_STEP_1'
export OIL_SPILL_MODEL_INPUT_SIZE='512'
export OIL_SPILL_OIL_CLASS_INDEX='1'
```

## 3. Register the Exact Artifact

```bash
curl --fail-with-body -sS -X POST "$API_URL/api/oil-spill/models" \
  -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model_id\": \"$MODEL_ID\",
    \"model_version\": \"$MODEL_VERSION\",
    \"engine\": \"onnx\",
    \"artifact_sha256\": \"$MODEL_SHA256\",
    \"intended_domains\": [\"drone_rgb_daylight\", \"port_glare\"],
    \"model_card_url\": \"https://registry.example/model-card/oil-segmentation-aerial\",
    \"notes\": \"ONNX aerial-RGB segmentation model; output channel 1 is oil.\"
  }" | jq
```

**Endpoint:** `POST /api/oil-spill/models`

Expected result: HTTP `201` with `"lifecycle_status": "candidate"`.

## 4. Record Measured Sealed-Holdout Evidence

The values below are examples only. Submit the **actual measured** metrics from an incident-disjoint, sealed holdout. The promotion gate requires each intended domain to have at least 100 samples, oil F1 ≥ 0.97, oil IoU ≥ 0.95, oil precision ≥ 0.97, and oil recall ≥ 0.97.

```bash
curl --fail-with-body -sS -X POST "$API_URL/api/oil-spill/models/$MODEL_ID/$MODEL_VERSION/evaluations" \
  -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_fingerprint": "sha256:sealed-drone-rgb-v1",
    "split": "sealed_holdout",
    "domain": "drone_rgb_daylight",
    "sample_count": 250,
    "metrics": {
      "oil_f1": 0.971,
      "oil_iou": 0.953,
      "oil_precision": 0.976,
      "oil_recall": 0.970,
      "expected_calibration_error": 0.021
    },
    "jepa_backbone": "vjepa2.1-vit-large-384",
    "reviewer": "qualified_reviewer",
    "notes": "Measured, sealed holdout; no location or incident leakage."
  }' | jq

curl --fail-with-body -sS -X POST "$API_URL/api/oil-spill/models/$MODEL_ID/$MODEL_VERSION/evaluations" \
  -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_fingerprint": "sha256:sealed-port-glare-v1",
    "split": "sealed_holdout",
    "domain": "port_glare",
    "sample_count": 180,
    "metrics": {
      "oil_f1": 0.973,
      "oil_iou": 0.956,
      "oil_precision": 0.974,
      "oil_recall": 0.972,
      "expected_calibration_error": 0.024
    },
    "jepa_backbone": "vjepa2.1-vit-large-384",
    "reviewer": "qualified_reviewer",
    "notes": "Measured, sealed holdout; glare-lookalikes included."
  }' | jq
```

**Endpoint:** `POST /api/oil-spill/models/{model_id}/{model_version}/evaluations`

## 5. Inspect and Approve the Promotion Gate

```bash
curl --fail-with-body -sS \
  "$API_URL/api/oil-spill/models/$MODEL_ID/$MODEL_VERSION/promotion" \
  -H "$AUTH" | jq

curl --fail-with-body -sS -X POST \
  "$API_URL/api/oil-spill/models/$MODEL_ID/$MODEL_VERSION/approve" \
  -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{
    "reviewer": "qualified_reviewer",
    "note": "Promotion gate evidence reviewed and accepted."
  }' | jq
```

**Endpoints:**

| Purpose | Method and path |
|---|---|
| Inspect deterministic eligibility | `GET /api/oil-spill/models/{model_id}/{model_version}/promotion` |
| Approve an eligible model | `POST /api/oil-spill/models/{model_id}/{model_version}/approve` |

A failing promotion gate returns HTTP `409` with the unmet requirements. Do not override it by changing only a label or environment variable.

## 6. Submit Raw Aerial Imagery for ONNX Inference

```bash
curl --fail-with-body -sS -X POST "$API_URL/api/oil-spill/analyze/image" \
  -H "$AUTH" \
  -F "image=@$AERIAL_IMAGE;type=image/jpeg" \
  -F 'source=drone_rgb' \
  -F "model_id=$MODEL_ID" \
  -F "model_version=$MODEL_VERSION" \
  -F 'image_id=mission-2026-08-20-frame-001' \
  -F 'observed_at=2026-08-20T14:30:00Z' \
  -F 'ground_sampling_distance_m=0.15' \
  -F 'probability_threshold=0.50' \
  -F 'min_component_area_px=25' \
  -F 'geographic_bounds_json={"west":4.000000,"south":51.000000,"east":4.002000,"north":51.001500}' \
  -F 'metadata_json={"mission_id":"mission-2026-08-20","platform":"drone-rgb","sensor":"RGB","operator":"field-team-a"}' | jq
```

**Endpoint:** `POST /api/oil-spill/analyze/image`

Expected result: HTTP `201` with an `incident_id`, predicted footprint metrics, quality flags, review state `pending_review`, and GeoJSON when a valid geographic footprint is supplied.

## 7. Inspect, Review, and Export the Result

```bash
export INCIDENT_ID='REPLACE_WITH_INCIDENT_ID_FROM_STEP_6'

curl --fail-with-body -sS "$API_URL/api/oil-spill/incidents/$INCIDENT_ID" \
  -H "$AUTH" | jq

curl --fail-with-body -sS -X PATCH "$API_URL/api/oil-spill/incidents/$INCIDENT_ID/review" \
  -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{
    "status": "confirmed",
    "reviewer": "qualified_reviewer",
    "note": "Aerial evidence reviewed under the incident-response procedure."
  }' | jq

curl --fail-with-body -sS -X POST "$API_URL/api/oil-spill/incidents/$INCIDENT_ID/coverage-plan" \
  -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{"cell_size_m": 50, "drone_count": 2, "buffer_m": 100}' | jq

curl --fail-with-body -sS "$API_URL/api/oil-spill/incidents/$INCIDENT_ID/export.geojson" \
  -H "$AUTH" > "oil-spill-$INCIDENT_ID.geojson"
```

## 8. Frequent Failure Responses

| HTTP response | Meaning | Correct action |
|---|---|---|
| `401` | Missing, invalid, or expired JWT. | Obtain a valid short-lived operator token. |
| `409` from approval | One or more sealed-evaluation requirements are not met. | Review the returned reasons; collect valid evidence, do not bypass the gate. |
| `409` from image analysis | The configured artifact is registered but not approved. | Complete the per-domain sealed evaluation and approval sequence. |
| `503` from image analysis | Missing server model configuration, file/hash mismatch, unregistered artifact, or unavailable ONNX runtime. | Check the API server environment and exact artifact SHA-256. |
| `422` from image analysis | Invalid image, malformed metadata/footprint JSON, or incompatible model output. | Validate file, JSON, model input/output contract, and channel index. |
