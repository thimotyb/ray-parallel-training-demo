#!/bin/bash
set -e

echo "=========================================="
echo "Installing Ray Monitoring Stack"
echo "=========================================="
echo ""
echo "This will install:"
echo "  - Prometheus (metrics collection)"
echo "  - Grafana (visualization)"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed"
    exit 1
fi

# Check if ray-system namespace exists
if ! kubectl get namespace ray-system &> /dev/null; then
    echo "Error: ray-system namespace not found"
    echo "Please install Ray first with: ./scripts/install-ray.sh"
    exit 1
fi

# Deploy Prometheus
echo "=========================================="
echo "1. Deploying Prometheus"
echo "=========================================="
kubectl apply -f ../k8s/prometheus-config.yaml

echo "Waiting for Prometheus to be ready..."
kubectl wait --for=condition=available --timeout=120s \
  deployment/prometheus -n ray-system || true

# Deploy Grafana
echo ""
echo "=========================================="
echo "2. Deploying Grafana"
echo "=========================================="
kubectl apply -f ../k8s/grafana-config.yaml

echo "Waiting for Grafana to be ready..."
kubectl wait --for=condition=available --timeout=120s \
  deployment/grafana -n ray-system || true

# Wait a bit for pods to be fully ready
echo ""
echo "Waiting for pods to be fully operational..."
sleep 10

# Show status
echo ""
echo "=========================================="
echo "Monitoring Stack Deployed Successfully!"
echo "=========================================="
echo ""

echo "Deployed services:"
kubectl get svc -n ray-system | grep -E "NAME|prometheus|grafana"
echo ""

echo "Deployed pods:"
kubectl get pods -n ray-system | grep -E "NAME|prometheus|grafana"
echo ""

# Get Minikube IP
MINIKUBE_IP=$(minikube ip 2>/dev/null || echo "unavailable")

echo "=========================================="
echo "Access Information"
echo "=========================================="
echo ""

if [ "$MINIKUBE_IP" != "unavailable" ]; then
    echo "Prometheus:"
    echo "  URL: http://${MINIKUBE_IP}:9090"
    echo "  (Internal ClusterIP service - use port-forward for access)"
    echo ""

    echo "Grafana:"
    echo "  URL: http://${MINIKUBE_IP}:30300"
    echo "  Default Credentials:"
    echo "    Username: admin"
    echo "    Password: admin"
    echo ""

    echo "Ray Dashboard:"
    echo "  URL: http://${MINIKUBE_IP}:30265"
    echo "  (Refresh to see time-series charts)"
else
    echo "Grafana: Access via port-forward"
    echo "Prometheus: Access via port-forward"
fi

echo ""
echo "=========================================="
echo "Port Forwarding Commands (if needed)"
echo "=========================================="
echo ""
echo "Prometheus:"
echo "  kubectl port-forward -n ray-system svc/prometheus 9090:9090"
echo "  Then access: http://localhost:9090"
echo ""
echo "Grafana:"
echo "  kubectl port-forward -n ray-system svc/grafana 3000:3000"
echo "  Then access: http://localhost:3000"
echo ""

echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Refresh the Ray dashboard to see time-series charts"
echo "2. Access Grafana to create custom dashboards"
echo "3. View Prometheus metrics at /metrics endpoint"
echo ""
echo "Note: The Ray dashboard will automatically detect"
echo "      Prometheus and enable time-series graphs."
echo ""
echo "=========================================="
