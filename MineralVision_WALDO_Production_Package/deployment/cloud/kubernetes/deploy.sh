#!/bin/bash
# Kubernetes deployment script for MineralVision WALDO integration

# Set default namespace
NAMESPACE=${NAMESPACE:-"mineralvision"}

# Create namespace if it doesn't exist
kubectl get namespace $NAMESPACE > /dev/null 2>&1 || kubectl create namespace $NAMESPACE

# Apply storage and secrets
kubectl apply -f storage-and-secrets.yaml -n $NAMESPACE

# Apply deployments
kubectl apply -f waldo-detection.yaml -n $NAMESPACE
kubectl apply -f api-service.yaml -n $NAMESPACE
kubectl apply -f web-ui.yaml -n $NAMESPACE
kubectl apply -f arcgis-integration.yaml -n $NAMESPACE

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment/waldo-detection -n $NAMESPACE
kubectl rollout status deployment/api-service -n $NAMESPACE
kubectl rollout status deployment/web-ui -n $NAMESPACE
kubectl rollout status deployment/arcgis-integration -n $NAMESPACE

echo "MineralVision WALDO integration deployed successfully!"
echo "Access the web UI at: https://waldo.mineralvision.com"
