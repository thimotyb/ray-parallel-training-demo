#!/bin/bash
set -e

# Configuration
IMAGE_NAME="ray-training-demo"
IMAGE_TAG="latest"
REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"  # Default to local registry

echo "=========================================="
echo "Building Docker image for Ray Training"
echo "=========================================="

# Build the image
echo "Building image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile ..

# Tag for Minikube's local registry
echo "Tagging image for Minikube..."
docker tag ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:${IMAGE_TAG}

# Load image into Minikube
echo "Loading image into Minikube..."
minikube image load ${IMAGE_NAME}:${IMAGE_TAG}

echo ""
echo "=========================================="
echo "Image built and loaded into Minikube!"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "=========================================="
