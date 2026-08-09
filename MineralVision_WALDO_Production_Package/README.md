# MineralVision WALDO Production Package

This package contains all the artifacts needed to deploy the MineralVision platform integrated with WALDO (Wide Area Large-scale Detection and Observation) capabilities.

## Package Contents

1. **Source Code**
   - WALDO Integration Modules
   - API Server
   - Web UI Components
   - Database Schema Extensions
   - ArcGIS Integration

2. **Deployment Configurations**
   - Cloud Deployment (Docker, Kubernetes)
   - On-Premise Deployment
   - Edge Device Deployment

3. **Documentation**
   - Installation Guides
   - User Manuals
   - API Documentation
   - Administrator Guides
   - Training Materials

4. **Sample Data**
   - Aerial Imagery
   - Drone Video
   - Historical Data

## Quick Start

1. Choose your deployment environment (cloud, on-premise, or edge)
2. Follow the appropriate installation guide in the `docs/installation` directory
3. Refer to the user manual for operating instructions

## Deployment Options

### Cloud Deployment

For cloud deployment, use the Docker Compose or Kubernetes configurations in the `deployment/cloud` directory:

```bash
# Docker Compose
cd deployment/cloud
docker-compose up -d

# Kubernetes
cd deployment/cloud/kubernetes
./deploy.sh
```

### On-Premise Deployment

For on-premise deployment, use the installation script:

```bash
cd deployment/on-premise
sudo ./install.sh
```

### Edge Device Deployment

For edge devices (Jetson, Raspberry Pi, etc.), use the edge installation script:

```bash
cd deployment/edge
sudo ./install.sh
```

## ArcGIS Integration

This package includes full integration with ArcGIS for spatial visualization and analysis. See the ArcGIS integration documentation in `docs/user_manual/user_manual.md` for details.

## Support

For support, contact support@mineralvision.com or visit https://support.mineralvision.com.
