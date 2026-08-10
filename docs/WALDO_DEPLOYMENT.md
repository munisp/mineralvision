# WALDO Deployment Guide

Deployment notes for the MineralVision WALDO detection stack and its
optional vision backends.

## SAM Segmentation Backend (api/vision/sam3)

The `sam3` package exposes a **SAM3-ready interface** mounted at
`/api/v1/sam3`. The current loadable inference backend is an ultralytics
SAM/SAM2 checkpoint or a remote service; the native `sam3` package is
supported when installed. When no backend is available the `/segment/*`
endpoints answer **503 with remediation text** and `/health` reports
`sam3_available: false` — no silent empty results.

### Environment variables

| Variable | Purpose |
|---|---|
| `SAM3_SERVICE_URL` | URL of a remote SAM segmentation service. When set, deployments can route segmentation to that service instead of loading a local checkpoint. Informational in `/health` (`service_url_configured`). |
| `SAM3_CHECKPOINT_PATH` | Path to a SAM/SAM2 checkpoint file inside the container. Mount the model volume (see below) and point this variable at the checkpoint, e.g. `/models/sam2.pt`. Informational in `/health` (`checkpoint_configured`). |
| `MV_ALLOW_MOCK_FALLBACK` | **UI development only.** When `true`, `/training/start` runs a labelled no-op fallback (status + message only, never fabricated metrics) instead of answering 503, and `/segment/*` callers may pass `allow_empty_fallback=true` to receive a labelled empty result (`metadata.mock: true`). Leave unset in production. |

### Checkpoint volume

Checkpoints live outside the image. Mount a model volume into the API
container:

```yaml
services:
  api:
    volumes:
      - model-checkpoints:/models:ro
    environment:
      SAM3_CHECKPOINT_PATH: /models/sam2.pt
      # SAM3_SERVICE_URL: http://sam-service:9100

volumes:
  model-checkpoints:
```

### Optional Python backends

Heavy/optional dependencies are pinned (commented) in
`requirements-ml.txt`:

- `segment-geospatial` — SAM/SAM2 geospatial helpers (optional)
- `sam3` — native SAM3 backend (optional; install from vendor source)

With only the lean API image, `/api/v1/sam3/health` returns
`status: degraded` and `sam3_available: false`, and inference endpoints
return 503 — install the optional pins and provision a checkpoint (or set
`SAM3_SERVICE_URL`) to enable segmentation.
