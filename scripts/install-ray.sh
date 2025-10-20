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

# Deploy monitoring stack
echo ""
echo "=========================================="
echo "Step 4: Installing Monitoring Stack"
echo "=========================================="
echo "Installing Prometheus and Grafana for enhanced metrics..."
echo ""

# Deploy Prometheus
echo "Deploying Prometheus..."
kubectl apply -f ../k8s/prometheus-config.yaml

# Deploy Grafana
echo "Deploying Grafana..."
kubectl apply -f ../k8s/grafana-config.yaml

# Wait for monitoring stack
echo "Waiting for monitoring stack to be ready..."
echo "(This may take 1-2 minutes)"
sleep 10

# Wait for Prometheus
kubectl wait --for=condition=available --timeout=120s \
  deployment/prometheus -n ray-system 2>/dev/null || echo "Prometheus deployment pending..."

# Wait for Grafana
kubectl wait --for=condition=available --timeout=120s \
  deployment/grafana -n ray-system 2>/dev/null || echo "Grafana deployment pending..."

# Give pods a moment to fully start
sleep 5

echo ""
echo "Monitoring stack deployed!"
echo ""

# Show all services
echo "Deployed services:"
kubectl get svc -n ray-system
echo ""

echo "Deployed pods:"
kubectl get pods -n ray-system
echo ""

echo "=========================================="
echo "Access Information"
echo "=========================================="
MINIKUBE_IP=$(minikube ip 2>/dev/null || echo "unavailable")

if [ "$MINIKUBE_IP" != "unavailable" ]; then
    echo ""
    echo "Ray Dashboard:"
    echo "  URL: http://${MINIKUBE_IP}:30265"
    echo "  Features: Time-series charts enabled"
    echo ""
    echo "Grafana Dashboard:"
    echo "  URL: http://${MINIKUBE_IP}:30300"
    echo "  Username: admin"
    echo "  Password: admin"
    echo ""
    echo "Helper script:"
    echo "  ./scripts/open-dashboard.sh"
    echo ""
else
    echo ""
    echo "Ray Dashboard: Run ./scripts/open-dashboard.sh"
    echo "Grafana: kubectl port-forward -n ray-system svc/grafana 3000:3000"
    echo ""
fi

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "What's been installed:"
echo "  ✓ Custom Ray image with TensorFlow"
echo "  ✓ KubeRay operator"
echo "  ✓ Ray cluster (1 head + 2 workers)"
echo "  ✓ Prometheus (metrics collection)"
echo "  ✓ Grafana (visualization)"
echo ""
echo "Next steps:"
echo "  1. Access Ray dashboard (see URLs above)"
echo "  2. Run benchmark: ./scripts/run-benchmark.sh"
echo "  3. View Grafana dashboards (optional)"
echo ""
echo "Note: Refresh the Ray dashboard to see time-series charts"
echo "=========================================="
