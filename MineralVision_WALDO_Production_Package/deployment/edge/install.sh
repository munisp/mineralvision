#!/bin/bash
# Edge device installation script for MineralVision WALDO integration

# Set default installation directory
INSTALL_DIR=${INSTALL_DIR:-"/opt/mineralvision/waldo"}
CONFIG_DIR=${CONFIG_DIR:-"/etc/mineralvision/waldo"}
DATA_DIR=${DATA_DIR:-"/var/lib/mineralvision/waldo"}
LOG_DIR=${LOG_DIR:-"/var/log/mineralvision/waldo"}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit 1
fi

# Detect device type
if [ -f /etc/nv_tegra_release ]; then
  DEVICE_TYPE="jetson"
  echo "Detected Jetson device"
elif [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
  DEVICE_TYPE="raspberry_pi"
  echo "Detected Raspberry Pi device"
else
  DEVICE_TYPE="generic"
  echo "Generic edge device detected"
fi

# Create directories
echo "Creating directories..."
mkdir -p $INSTALL_DIR
mkdir -p $CONFIG_DIR
mkdir -p $DATA_DIR/models
mkdir -p $DATA_DIR/database
mkdir -p $LOG_DIR

# Install system dependencies based on device type
echo "Installing system dependencies..."
apt-get update

if [ "$DEVICE_TYPE" = "jetson" ]; then
  # Jetson-specific dependencies
  apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    sqlite3 \
    nginx \
    supervisor
elif [ "$DEVICE_TYPE" = "raspberry_pi" ]; then
  # Raspberry Pi-specific dependencies
  apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libopenjp2-7 \
    libtiff5 \
    libatlas-base-dev \
    sqlite3 \
    nginx \
    supervisor
else
  # Generic dependencies
  apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    sqlite3 \
    nginx \
    supervisor
fi

# Copy application files
echo "Copying application files..."
cp -r src/* $INSTALL_DIR/
cp -r config/* $CONFIG_DIR/

# Install Python dependencies based on device type
echo "Installing Python dependencies..."
if [ "$DEVICE_TYPE" = "jetson" ]; then
  pip3 install -r requirements-jetson.txt
elif [ "$DEVICE_TYPE" = "raspberry_pi" ]; then
  pip3 install -r requirements-rpi.txt
else
  pip3 install -r requirements-edge.txt
fi

# Configure services
echo "Configuring services..."
cp deployment/edge/waldo-edge.service /etc/systemd/system/
cp deployment/edge/waldo-sync.service /etc/systemd/system/

# Configure nginx
echo "Configuring nginx..."
cp deployment/edge/nginx.conf /etc/nginx/sites-available/waldo
ln -sf /etc/nginx/sites-available/waldo /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Download optimized WALDO model for edge device
echo "Downloading optimized WALDO model for $DEVICE_TYPE..."
if [ "$DEVICE_TYPE" = "jetson" ]; then
  # Jetson-specific model (optimized for TensorRT)
  pip3 install gdown
  gdown --id 1_QS9GgFjgZ7Jh5TQH4K5Gl5E8QI8mwg_jetson -O $DATA_DIR/models/waldo_edge.onnx
elif [ "$DEVICE_TYPE" = "raspberry_pi" ]; then
  # Raspberry Pi-specific model (int8 quantized)
  pip3 install gdown
  gdown --id 1_QS9GgFjgZ7Jh5TQH4K5Gl5E8QI8mwg_rpi -O $DATA_DIR/models/waldo_edge.onnx
else
  # Generic edge model
  pip3 install gdown
  gdown --id 1_QS9GgFjgZ7Jh5TQH4K5Gl5E8QI8mwg_edge -O $DATA_DIR/models/waldo_edge.onnx
fi

# Set permissions
echo "Setting permissions..."
chown -R www-data:www-data $DATA_DIR
chown -R www-data:www-data $LOG_DIR

# Start services
echo "Starting services..."
systemctl daemon-reload
systemctl enable waldo-edge
systemctl enable waldo-sync
systemctl enable nginx

systemctl start waldo-edge
systemctl start waldo-sync
systemctl restart nginx

echo "MineralVision WALDO integration installed successfully on edge device!"
echo "Access the local web UI at: http://localhost"
echo "Data will be synchronized with the central server based on the configured schedule."
