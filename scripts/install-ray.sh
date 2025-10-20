#!/bin/bash
set -e

echo "=========================================="
echo "Installing Ray on Kubernetes"
echo "=========================================="

# Check if helm is installed
if ! command -v helm &> /dev/null; then
    echo "Error: helm is not installed"
    echo "Please install helm from: https://helm.sh/docs/intro/install/"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed"
    echo "Please install docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Build custom Ray image with TensorFlow
echo ""
echo "=========================================="
echo "Step 1: Building Custom Docker Image"
echo "=========================================="
echo "Building Ray image with TensorFlow and dependencies..."
echo "This may take 3-5 minutes on first build..."
echo ""

cd ../docker
./build-and-push.sh
cd ../scripts

echo ""
echo "Custom image built successfully!"
echo ""

# Add Ray Helm repository
echo "=========================================="
echo "Step 2: Setting up KubeRay"
echo "=========================================="
echo "Adding Ray Helm repository..."
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Create namespace for Ray
echo "Creating namespace 'ray-system'..."
kubectl create namespace ray-system --dry-run=client -o yaml | kubectl apply -f -

# Install KubeRay operator
echo "Installing KubeRay operator..."
helm upgrade --install kuberay-operator kuberay/kuberay-operator \
  --namespace ray-system \
  --wait

# Wait for operator to be ready
echo "Waiting for KubeRay operator to be ready..."
kubectl wait --for=condition=available --timeout=300s \
  deployment/kuberay-operator -n ray-system

# Deploy Ray cluster
echo ""
echo "=========================================="
echo "Step 3: Deploying Ray Cluster"
echo "=========================================="
echo "Deploying Ray cluster with custom image..."
kubectl apply -f ../k8s/ray-cluster.yaml

# Deploy Ray service (NodePort for dashboard access)
echo "Deploying Ray dashboard service..."
kubectl apply -f ../k8s/ray-service.yaml

# Wait for Ray cluster to be ready
echo "Waiting for Ray cluster to be ready..."
sleep 10
kubectl wait --for=condition=ready --timeout=300s \
  pod -l ray.io/node-type=head -n ray-system

echo ""
echo "=========================================="
echo "Ray cluster deployed successfully!"
echo "Cluster status:"
kubectl get raycluster -n ray-system
echo ""
echo "Ray pods:"
kubectl get pods -n ray-system
echo ""
echo "Ray dashboard service:"
kubectl get svc -n ray-system ray-cluster-head-svc
echo ""
echo "=========================================="
echo "Dashboard Access:"
echo "=========================================="
MINIKUBE_IP=$(minikube ip 2>/dev/null || echo "unavailable")
if [ "$MINIKUBE_IP" != "unavailable" ]; then
    echo "Dashboard URL: http://${MINIKUBE_IP}:30265"
    echo ""
    echo "Or run: ./scripts/open-dashboard.sh"
else
    echo "Run: ./scripts/open-dashboard.sh"
fi
echo "=========================================="
