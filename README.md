# MineralVision

[![CI](https://github.com/mineralvision/mineralvision/actions/workflows/ci.yml/badge.svg)](https://github.com/mineralvision/mineralvision/actions/workflows/ci.yml)

**The premiere platform for mineral exploration, discovery, and compliance.**

MineralVision unifies the full exploration lifecycle — satellite anomaly
detection, target ranking, drilling, resource estimation, and compliant
reporting — behind a single FastAPI backend and a single React web
application. Read [docs/VISION.md](docs/VISION.md) for the platform vision
and the personas it serves.

## Quick Start

### API (canonical entry point)

```bash
pip install -r requirements.txt
cd MineralVision_Final_Package
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs are then available at http://localhost:8000/docs
(Swagger UI) and http://localhost:8000/redoc (ReDoc).

> Note: `src.api.main:app` is the canonical application entry point. The
> historical `main_demo.py` / `main_production.py` / `main_simple.py` /
> `main_standalone.py` entry points are being consolidated into it.

### Web UI

```bash
cd MineralVision_Final_Package/src/ui/web/mineralvision-app
npm ci
npm run dev
```

### Docker

```bash
docker compose up --build
```

## Features

- **Drillhole database** — collars, surveys, assays, lithology, with 3D
  visualization and cross-sections
- **QAQC** — standards, blanks, and duplicates tracked from assay load to
  resource report
- **Geostatistics** — variography, kriging, block modeling, grade shells
  (`src/api/geostatistics/`)
- **Geophysical inversion** — gravity, magnetics, EM
  (`src/api/geophysics/`)
- **Prospectivity ML** — target ranking with spatial cross-validation and
  uncertainty quantification (`src/api/ml/`)
- **WALDO field detection** — YOLO11 + RF-DETR ensemble detection of
  outcrop, gossan, and alteration indicators in field imagery
- **Sensor fusion** — magnetometry, radiometrics, hyperspectral, LiDAR, GPR,
  and SEG-Y with Kalman and deep-learning fusion (`src/api/sensor_fusion/`)
- **Compliant reporting** — JORC- and NI 43-101-aligned report generation
  with end-to-end audit trails (`src/api/reporting/`)
- **Auth & multi-tenancy** — JWT authentication with role-based access
  control

## Project Structure

```
mineralvision/
├── MineralVision_Final_Package/     # Canonical API + UI
│   └── src/
│       ├── api/                     # FastAPI application (entry: src.api.main:app)
│       └── ui/web/mineralvision-app # React + TypeScript web app
├── MineralVision_Enhanced/          # Lakehouse + geospatial middleware
├── MineralVision_WALDO_Production_Package/  # WALDO detection services
├── infrastructure/                  # Kubernetes, Helm, Terraform
├── tests/                           # Pytest test suite
└── docs/                            # Documentation (start with VISION.md)
```

## Oil-Spill Intelligence Extension

MineralVision now supports **reviewable oil-spill assessment** from drone, satellite, fluorosensor, or manual-annotation evidence. The extension converts a versioned segmentation mask into a cleaned oil footprint, area estimate, severity screen, GeoJSON geometry (when a geographic image footprint is supplied), persistent incident record, human review state, and advisory drone-coverage grid. Raw-image inference is disabled by default and requires an explicitly configured, trusted local TorchScript or ONNX model.

Read the [research review](docs/oil-spill/research-review.md) and [architecture and operating constraints](docs/oil-spill/architecture.md). The API surface is available under `/api/oil-spill`.

## Development

```bash
# Backend tests
pytest tests/

# UI type-check / build (from the UI directory)
npm run build
```

## License

Proprietary
