#!/bin/bash
set -e

echo "=========================================="
echo "Ray Dashboard Access"
echo "=========================================="
echo ""

# Check if minikube is running
if ! minikube status &> /dev/null; then
    echo "Error: Minikube is not running"
    echo "Please start Minikube first with: ./scripts/setup-minikube.sh"
    exit 1
fi

# Check if ray-system namespace exists
if ! kubectl get namespace ray-system &> /dev/null; then
    echo "Error: Ray system namespace not found"
    echo "Please install Ray first with: ./scripts/install-ray.sh"
    exit 1
fi

# Check if Ray cluster is running
if ! kubectl get raycluster -n ray-system ray-training-cluster &> /dev/null; then
    echo "Error: Ray cluster not found"
    echo "Please install Ray first with: ./scripts/install-ray.sh"
    exit 1
fi

# Get the Minikube IP
MINIKUBE_IP=$(minikube ip)
DASHBOARD_PORT=30265

echo "Ray Dashboard Access Methods:"
echo ""
echo "=========================================="
echo "Method 1: Direct NodePort Access"
echo "=========================================="
echo ""
echo "The Ray dashboard is exposed via NodePort on port 30265"
echo ""
echo "Dashboard URL: http://${MINIKUBE_IP}:${DASHBOARD_PORT}"
echo ""
echo "Open this URL in your browser to access the dashboard."
echo ""

echo "=========================================="
echo "Method 2: Port Forwarding (Alternative)"
echo "=========================================="
echo ""
echo "If NodePort doesn't work, use kubectl port-forward:"
echo ""
echo "Run in a separate terminal:"
echo "  kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265"
echo ""
echo "Then access: http://localhost:8265"
echo ""

echo "=========================================="
echo "Dashboard Features"
echo "=========================================="
echo ""
echo "The Ray dashboard provides:"
echo "  • Cluster overview and resource usage"
echo "  • Job monitoring and management"
echo "  • Actor and task visualization"
echo "  • Logs and metrics"
echo "  • Node and worker status"
echo ""

echo "=========================================="
echo "Quick Status Check"
echo "=========================================="
echo ""
echo "Ray Cluster Status:"
kubectl get raycluster -n ray-system
echo ""
echo "Ray Pods:"
kubectl get pods -n ray-system -o wide
echo ""
echo "Ray Service:"
kubectl get svc -n ray-system ray-cluster-head-svc
echo ""

# Check if we can open browser automatically
if command -v xdg-open &> /dev/null; then
    read -p "Would you like to open the dashboard in your browser now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Opening dashboard in browser..."
        xdg-open "http://${MINIKUBE_IP}:${DASHBOARD_PORT}" &
    fi
elif command -v open &> /dev/null; then
    read -p "Would you like to open the dashboard in your browser now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Opening dashboard in browser..."
        open "http://${MINIKUBE_IP}:${DASHBOARD_PORT}" &
    fi
fi

echo ""
echo "=========================================="
echo "Note: Ray Dashboard does NOT require login"
echo "=========================================="
echo ""
echo "The Ray dashboard is accessible without authentication."
echo "If you need to secure it, consider:"
echo "  • Using kubectl port-forward (only accessible from localhost)"
echo "  • Setting up ingress with authentication"
echo "  • Using a VPN or SSH tunnel"
echo ""
echo "=========================================="
