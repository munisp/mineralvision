#!/bin/bash
# Entrypoint script for WALDO detection service

# Set default environment variables if not provided
export MODEL_PRECISION=${MODEL_PRECISION:-"fp16"}
export BATCH_SIZE=${BATCH_SIZE:-4}
export DATABASE_URI=${DATABASE_URI:-"sqlite:///app/data/waldo_detections.db"}

# Start the WALDO detection service (Flask API in src/api/server.py)
echo "Starting WALDO detection service with precision: $MODEL_PRECISION, batch size: $BATCH_SIZE"
export PYTHONPATH=/app/src
export PORT=${PORT:-8000}
exec python3 -m api.server
