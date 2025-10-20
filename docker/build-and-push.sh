#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Configuration
IMAGE_NAME="ray-training-demo"
IMAGE_TAG="latest"
REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"  # Default to local registry

echo "=========================================="
echo "Building Docker image for Ray Training"
echo "=========================================="

# Check if image already exists in Minikube
echo "Checking if image already exists in Minikube..."
if minikube image ls --format table | grep -q "${IMAGE_NAME}:${IMAGE_TAG}"; then
    echo "Image ${IMAGE_NAME}:${IMAGE_TAG} already exists in Minikube"
    echo "Skipping build and load (use 'minikube image rm ${IMAGE_NAME}:${IMAGE_TAG}' to force rebuild)"
    echo ""
    echo "=========================================="
    echo "Using existing image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "=========================================="
    exit 0
fi

# Build the image
echo "Building image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"

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
