FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/

# Create model and data directories
RUN mkdir -p /app/models /app/data

# Download WALDO model (can be commented out if models are mounted as volume)
# gdown is pinned in requirements.txt
RUN mkdir -p /app/models && \
    gdown --id 1_QS9GgFjgZ7Jh5TQH4K5Gl5E8QI8mwg -O /app/models/waldo_v3.pt

# Set up entry point
COPY deployment/cloud/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
