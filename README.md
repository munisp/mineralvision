# MineralVision

AI-Powered Mineral Exploration Platform

## Overview

MineralVision is a comprehensive platform for mineral exploration integrating geology, geostatistics, geophysics, sensor fusion, AI/ML, digital twin, climate resilience, autonomous exploration, indigenous knowledge integration, and blockchain data provenance.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Run the API server
cd MineralVision_Final_Package/src
uvicorn api.main_production:app --host 0.0.0.0 --port 8000

# Run the web UI (separate terminal)
cd MineralVision_Final_Package/src/ui/web/mineralvision-app
npm install && npm start
```

## Project Structure

```
mineralvision/
├── MineralVision_Enhanced/          # Lakehouse + Middleware
│   ├── lakehouse_architecture/      # Delta Lake, Iceberg, Kafka streaming
│   └── middleware/                  # 16 middleware integrations
├── MineralVision_Final_Package/     # API + ML + UI
│   └── src/
│       ├── api/                     # FastAPI endpoints
│       ├── ml/                      # Machine learning models
│       └── ui/                      # Web (React) + Mobile (Flutter)
├── MineralVision_WALDO_Production_Package/  # Object detection
│   └── src/waldo_integration/       # YOLOv8 + RF-DETR detection
├── infrastructure/                  # Kubernetes, Helm, Terraform
├── tests/                          # Pytest test suite
└── docs/                           # Documentation
```

## Features

- Drillhole database with 3D visualization
- Geostatistics (variography, kriging, block modeling)
- Geophysical inversion (gravity, magnetics, EM)
- Sensor fusion (magnetometry, radiometrics, LiDAR, GPR)
- AI/ML prospectivity mapping
- WALDO object detection (YOLOv8 + RF-DETR)
- Digital twin with real-time streaming
- Climate resilience analysis
- Blockchain data provenance
- Multi-user RBAC with JWT authentication

## API Documentation

Once running, access the API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

See `.env.example` for all configuration options.

## License

Proprietary
