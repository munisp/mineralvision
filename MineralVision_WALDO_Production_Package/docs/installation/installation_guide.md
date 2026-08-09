# MineralVision WALDO Integration - Installation Guide

This guide provides instructions for installing the MineralVision WALDO integration in different deployment environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Cloud Deployment](#cloud-deployment)
   - [Docker Compose Deployment](#docker-compose-deployment)
   - [Kubernetes Deployment](#kubernetes-deployment)
3. [On-Premise Deployment](#on-premise-deployment)
4. [Edge Device Deployment](#edge-device-deployment)
   - [Jetson Devices](#jetson-devices)
   - [Raspberry Pi Devices](#raspberry-pi-devices)
   - [Other Edge Devices](#other-edge-devices)
5. [Post-Installation Verification](#post-installation-verification)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### General Requirements

- Internet connection for downloading dependencies and models
- Administrator/root access to the target system
- Git for cloning the repository

### Cloud Deployment Requirements

- Docker and Docker Compose (for Docker Compose deployment)
- Kubernetes cluster with NVIDIA GPU support (for Kubernetes deployment)
- Access to container registry

### On-Premise Requirements

- Ubuntu 20.04 LTS or later
- Python 3.8 or later
- NVIDIA GPU with CUDA support (recommended)
- PostgreSQL database
- RabbitMQ message broker

### Edge Device Requirements

- Jetson device with JetPack 4.6 or later, or
- Raspberry Pi 4 with 4GB+ RAM, or
- Other edge device with ARM/x86 processor and 4GB+ RAM
- Python 3.7 or later

## Cloud Deployment

### Docker Compose Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/mineralvision/waldo-integration.git
   cd waldo-integration
   ```

2. Create a `.env` file with the required environment variables:
   ```bash
   DB_PASSWORD=your_secure_password
   RABBITMQ_PASSWORD=your_secure_password
   ARCGIS_URL=https://your-arcgis-server.com
   ARCGIS_USERNAME=your_username
   ARCGIS_PASSWORD=your_password
   ```

3. Start the services:
   ```bash
   docker-compose -f deployment/cloud/docker-compose.yml up -d
   ```

4. Verify the deployment:
   ```bash
   docker-compose -f deployment/cloud/docker-compose.yml ps
   ```

### Kubernetes Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/mineralvision/waldo-integration.git
   cd waldo-integration/deployment/cloud/kubernetes
   ```

2. Update the secrets in `storage-and-secrets.yaml` with your actual credentials (base64 encoded):
   ```bash
   # Generate base64 encoded values
   echo -n "postgresql://mineralvision:your_password@database:5432/waldo_detections" | base64
   echo -n "amqp://mineralvision:your_password@message-broker:5672/" | base64
   echo -n "https://your-arcgis-server.com" | base64
   echo -n "your_username" | base64
   echo -n "your_password" | base64
   ```

3. Deploy using the provided script:
   ```bash
   ./deploy.sh
   ```

4. Verify the deployment:
   ```bash
   kubectl get pods -n mineralvision
   ```

## On-Premise Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/mineralvision/waldo-integration.git
   cd waldo-integration
   ```

2. Run the installation script:
   ```bash
   sudo ./deployment/on-premise/install.sh
   ```

3. Configure the system by editing the configuration files:
   ```bash
   sudo nano /etc/mineralvision/waldo/config.yaml
   ```

4. Restart the services:
   ```bash
   sudo systemctl restart waldo-detection waldo-api waldo-arcgis
   ```

5. Verify the installation:
   ```bash
   sudo systemctl status waldo-detection
   sudo systemctl status waldo-api
   sudo systemctl status waldo-arcgis
   ```

## Edge Device Deployment

### Jetson Devices

1. Clone the repository:
   ```bash
   git clone https://github.com/mineralvision/waldo-integration.git
   cd waldo-integration
   ```

2. Run the installation script:
   ```bash
   sudo ./deployment/edge/install.sh
   ```

3. Configure the device by editing the configuration files:
   ```bash
   sudo nano /etc/mineralvision/waldo/edge_config.yaml
   ```

4. Set the central server URL for synchronization:
   ```bash
   sudo nano /etc/systemd/system/waldo-sync.service
   # Update the CENTRAL_SERVER_URL environment variable
   ```

5. Restart the services:
   ```bash
   sudo systemctl restart waldo-edge waldo-sync
   ```

### Raspberry Pi Devices

Follow the same steps as for Jetson devices, but note that performance will be optimized for Raspberry Pi hardware.

### Other Edge Devices

Follow the same steps as for Jetson devices. The installation script will detect the device type and install the appropriate dependencies and models.

## Post-Installation Verification

After installation, verify that the system is working correctly:

1. Access the web UI:
   - Cloud/Kubernetes: https://waldo.mineralvision.com
   - On-premise/Edge: http://localhost

2. Check the logs:
   - Cloud/Docker: `docker-compose logs -f`
   - Kubernetes: `kubectl logs -f deployment/waldo-detection -n mineralvision`
   - On-premise/Edge: `sudo journalctl -u waldo-detection`

3. Test the API:
   ```bash
   curl http://localhost:8080/api/status
   ```

## Troubleshooting

### Common Issues

1. **Services not starting**
   - Check the logs: `sudo journalctl -u waldo-detection`
   - Verify permissions: `sudo chown -R www-data:www-data /var/lib/mineralvision/waldo`

2. **Database connection errors**
   - Verify database is running: `sudo systemctl status postgresql`
   - Check connection string in configuration

3. **GPU not detected**
   - Verify NVIDIA drivers: `nvidia-smi`
   - Check CUDA installation: `nvcc --version`

4. **Model loading errors**
   - Verify model exists: `ls -la /var/lib/mineralvision/waldo/models`
   - Check model format matches device (ONNX for edge devices)

For additional help, contact MineralVision support at support@mineralvision.com.
