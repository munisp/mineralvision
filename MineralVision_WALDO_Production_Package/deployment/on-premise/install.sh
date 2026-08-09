#!/bin/bash
# On-premise installation script for MineralVision WALDO integration

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

# Create directories
echo "Creating directories..."
mkdir -p $INSTALL_DIR
mkdir -p $CONFIG_DIR
mkdir -p $DATA_DIR/models
mkdir -p $DATA_DIR/database
mkdir -p $LOG_DIR

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    postgresql \
    postgresql-contrib \
    rabbitmq-server \
    nginx \
    supervisor

# Install NVIDIA drivers and CUDA if GPU is available
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected, installing drivers and CUDA..."
    apt-get install -y --no-install-recommends \
        nvidia-driver-525 \
        nvidia-cuda-toolkit
fi

# Copy application files
echo "Copying application files..."
cp -r src/* $INSTALL_DIR/
cp -r config/* $CONFIG_DIR/

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Set up database
echo "Setting up database..."
sudo -u postgres psql -c "CREATE USER mineralvision WITH PASSWORD 'password';"
sudo -u postgres psql -c "CREATE DATABASE waldo_detections OWNER mineralvision;"

# Configure services
echo "Configuring services..."
cp deployment/on-premise/waldo-detection.service /etc/systemd/system/
cp deployment/on-premise/waldo-api.service /etc/systemd/system/
cp deployment/on-premise/waldo-arcgis.service /etc/systemd/system/

# Configure nginx
echo "Configuring nginx..."
cp deployment/on-premise/nginx.conf /etc/nginx/sites-available/waldo
ln -sf /etc/nginx/sites-available/waldo /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Download WALDO model
echo "Downloading WALDO model..."
pip3 install gdown
gdown --id 1_QS9GgFjgZ7Jh5TQH4K5Gl5E8QI8mwg -O $DATA_DIR/models/waldo_v3.pt

# Set permissions
echo "Setting permissions..."
chown -R www-data:www-data $DATA_DIR
chown -R www-data:www-data $LOG_DIR

# Start services
echo "Starting services..."
systemctl daemon-reload
systemctl enable postgresql
systemctl enable rabbitmq-server
systemctl enable waldo-detection
systemctl enable waldo-api
systemctl enable waldo-arcgis
systemctl enable nginx

systemctl start postgresql
systemctl start rabbitmq-server
systemctl start waldo-detection
systemctl start waldo-api
systemctl start waldo-arcgis
systemctl restart nginx

echo "MineralVision WALDO integration installed successfully!"
echo "Access the web UI at: http://localhost"
