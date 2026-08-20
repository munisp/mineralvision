# Custom Model and API Integration Guide

This guide explains how to integrate a custom **PyTorch TorchScript** or **ONNX** segmentation model for controlled raw aerial-image inference, and how to test the complete oil-spill API. The platform is intentionally fail-closed: an image can be analyzed only when its local artifact hash matches the environment configuration and the matching model has passed the sealed-evaluation gate and been explicitly approved.

> **Operational boundary:** An approved model provides decision-support evidence. It does not automatically notify authorities, launch aircraft, authorize cleanup, estimate released volume, or make a final response decision.

## 1. Model Interface Contract

The local model adapter resizes RGB imagery to `OIL_SPILL_MODEL_INPUT_SIZE`, converts it to a `float32` tensor in the range `[0, 1]` with shape `[1, 3, H, W]`, and expects one of these outputs.

| Output form | Interpretation |
|---|---|
| `[1, 1, H, W]` or `[1, H, W]` | Single-channel **logits**. The platform applies sigmoid. |
| `[1, C, H, W]` or `[C, H, W]`, `C > 1` | Multiclass **logits**. The platform applies softmax and selects `OIL_SPILL_OIL_CLASS_INDEX`. |
| `[H, W]` | Single-channel **logits**. The platform applies sigmoid. |

The output is resized to the original image dimensions and checked for finite probabilities in `[0, 1]`. Do not deploy a model that expects a different color order, normalization, tensor layout, or output semantics without adapting it first.

### TorchScript Export

```python
# export_torchscript_oil_spill.py
import torch

model = ...  # Load your evaluated PyTorch segmentation model.
model.eval()
example = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
traced = torch.jit.trace(model, example)
traced.save("/opt/mineralvision/models/oil-spill-segmentation-v1.ts")
```

### ONNX Export

```python
# export_onnx_oil_spill.py
import torch

model = ...  # Load your evaluated PyTorch segmentation model.
model.eval()
example = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
torch.onnx.export(
    model,
    example,
    "/opt/mineralvision/models/oil-spill-segmentation-v1.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
```

## 2. Verify and Configure the Local Artifact

First calculate the artifact digest. This digest must match both the registry record and `OIL_SPILL_MODEL_SHA256` exactly.

```bash
MODEL_PATH=/opt/mineralvision/models/oil-spill-segmentation-v1.onnx
MODEL_SHA256=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
printf '%s\n' "$MODEL_SHA256"
```

Set the following deployment configuration. Do not commit the model file or its environment file to Git.

```bash
export DATABASE_URL='postgresql://mineralvision:REPLACE_ME@db.example:5432/mineralvision'
export OIL_SPILL_MODEL_PATH="$MODEL_PATH"
export OIL_SPILL_MODEL_ENGINE='onnx'                 # or: torchscript
export OIL_SPILL_MODEL_ID='oil-segmentation-aerial'
export OIL_SPILL_MODEL_VERSION='2026.08.20'
export OIL_SPILL_MODEL_SHA256="$MODEL_SHA256"
export OIL_SPILL_MODEL_INPUT_SIZE='512'
export OIL_SPILL_OIL_CLASS_INDEX='1'
```

The runtime verifies the local file’s SHA-256 before it loads the artifact. It then checks PostgreSQL for a matching, **approved** model registration. A missing registry entry, hash mismatch, candidate model, retired model, missing runtime, or malformed output keeps `POST /api/oil-spill/analyze/image` disabled.

## 3. Model-Governance Workflow

The following example assumes an API with JWT middleware, an operator token, and an evaluation run independently computed on a sealed holdout set. Replace every placeholder.

```bash
export API_URL='https://mineralvision.example'
export TOKEN='REPLACE_WITH_SHORT_LIVED_JWT'
export AUTH="Authorization: Bearer $TOKEN"
export MODEL_SHA256='REPLACE_WITH_REAL_64_CHARACTER_SHA256'

curl -sS -X POST "$API_URL/api/oil-spill/models" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d "{
    \"model_id\": \"oil-segmentation-aerial\",
    \"model_version\": \"2026.08.20\",
    \"engine\": \"onnx\",
    \"artifact_sha256\": \"$MODEL_SHA256\",
    \"intended_domains\": [\"drone_rgb_daylight\", \"port_glare\"],
    \"model_card_url\": \"https://registry.example/model-card\"
  }" | jq

# Store measured metrics only. The API never calculates or invents them.
curl -sS -X POST "$API_URL/api/oil-spill/models/oil-segmentation-aerial/2026.08.20/evaluations" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{
    "dataset_fingerprint": "sha256:sealed-holdout-manifest-v1",
    "split": "sealed_holdout",
    "domain": "drone_rgb_daylight",
    "sample_count": 250,
    "metrics": {"oil_f1": 0.971, "oil_iou": 0.953, "oil_precision": 0.976, "oil_recall": 0.970, "expected_calibration_error": 0.021},
    "jepa_backbone": "vjepa2.1-vit-large-384",
    "reviewer": "qualified_reviewer"
  }' | jq

# Repeat a sealed evaluation for every intended domain, then inspect the deterministic gate.
curl -sS "$API_URL/api/oil-spill/models/oil-segmentation-aerial/2026.08.20/promotion" -H "$AUTH" | jq

# Only an eligible model can be approved; a failing gate returns HTTP 409 with reasons.
curl -sS -X POST "$API_URL/api/oil-spill/models/oil-segmentation-aerial/2026.08.20/approve" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"reviewer": "qualified_reviewer", "note": "Sealed per-domain evidence reviewed."}' | jq
```

The current promotion gate requires **oil F1 ≥ 0.97, oil IoU ≥ 0.95, oil precision ≥ 0.97, and oil recall ≥ 0.97** for every intended domain on a sealed holdout with at least 100 samples. This is a deployment gate, not a claim that any model automatically meets the target.

## 4. Incident API Testing

### Assess a versioned probability mask

```bash
MASK_BASE64=$(base64 -w 0 sample_probability_mask.png)

curl -sS -X POST "$API_URL/api/oil-spill/analyze/mask" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d "{
    \"mask_base64\": \"$MASK_BASE64\",
    \"image_width_px\": 1024,
    \"image_height_px\": 768,
    \"source\": \"drone_rgb\",
    \"model_id\": \"oil-segmentation-aerial\",
    \"model_version\": \"2026.08.20\",
    \"ground_sampling_distance_m\": 0.15,
    \"geographic_bounds\": {\"west\": 4.0000, \"south\": 51.0000, \"east\": 4.0020, \"north\": 51.0015},
    \"metadata\": {\"mission_id\": \"demo-001\", \"sensor\": \"RGB\"}
  }" | jq
```

### Analyze raw imagery with the approved local model

```bash
curl -sS -X POST "$API_URL/api/oil-spill/analyze/image" \
  -H "$AUTH" \
  -F 'image=@aerial_frame.jpg;type=image/jpeg' \
  -F 'source=drone_rgb' \
  -F 'ground_sampling_distance_m=0.15' \
  -F 'geographic_bounds_json={"west":4.0,"south":51.0,"east":4.002,"north":51.0015}' \
  -F 'metadata_json={"mission_id":"demo-001","sensor":"RGB"}' | jq
```

### Fuse aligned sequential masks with optional real JEPA embeddings

```bash
curl -sS -X POST "$API_URL/api/oil-spill/temporal-consensus" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d "{
    \"masks_base64\": [\"$MASK_BASE64\", \"$MASK_BASE64\"],
    \"image_width_px\": 1024,
    \"image_height_px\": 768,
    \"jepa_embeddings\": [[0.12, -0.03, 0.91], [0.11, -0.02, 0.90]]
  }" | jq
```

The endpoint never invokes or fabricates JEPA embeddings. If embeddings are omitted it applies a median fusion and returns an explicit quality flag. If supplied, embeddings must be generated by a real, approved JEPA pipeline for the **same aligned frame sequence**.

### Review, cover, and export an incident

```bash
export INCIDENT_ID='REPLACE_WITH_INCIDENT_UUID'

curl -sS -X PATCH "$API_URL/api/oil-spill/incidents/$INCIDENT_ID/review" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"status":"confirmed","reviewer":"qualified_reviewer","note":"Verified by incident lead."}' | jq

curl -sS -X POST "$API_URL/api/oil-spill/incidents/$INCIDENT_ID/coverage-plan" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"cell_size_m":50,"drone_count":2,"buffer_m":100}' | jq

curl -sS "$API_URL/api/oil-spill/incidents/$INCIDENT_ID/export.geojson" -H "$AUTH" > incident.geojson
```

## 5. PostgreSQL and Migration Commands

```bash
export DATABASE_URL='postgresql://mineralvision:REPLACE_ME@localhost:5432/mineralvision'
cd MineralVision_Final_Package
alembic upgrade head
```

Run the repository validation against a dedicated, migrated PostgreSQL test database.

```bash
export OIL_SPILL_TEST_DATABASE_URL='postgresql://mineralvision:REPLACE_ME@localhost:5432/mineralvision_oil_spill_test'
export DATABASE_URL="$OIL_SPILL_TEST_DATABASE_URL"
python ../scripts/verify_oil_spill_persistence.py
```
