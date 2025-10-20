#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=========================================="
echo "Cleaning up Ray and Minikube resources"
echo "=========================================="

# Delete Ray cluster
echo "Deleting Ray cluster..."
kubectl delete -f "$PROJECT_ROOT/k8s/ray-cluster.yaml" --ignore-not-found=true

# Uninstall KubeRay operator
echo "Uninstalling KubeRay operator..."
helm uninstall kuberay-operator -n ray-system --ignore-not-found

# Delete namespace
echo "Deleting ray-system namespace..."
kubectl delete namespace ray-system --ignore-not-found=true

# Stop Minikube (optional - uncomment if you want to stop minikube)
# echo "Stopping Minikube cluster..."
# minikube stop

# Delete Minikube (optional - uncomment if you want to delete the entire cluster)
# echo "Deleting Minikube cluster..."
# minikube delete

echo ""
echo "=========================================="
echo "Cleanup completed!"
echo "=========================================="
