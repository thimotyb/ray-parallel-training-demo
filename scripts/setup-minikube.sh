#!/bin/bash
set -e

echo "=========================================="
echo "Ray Parallel Training Demo - Minikube Setup"
echo "=========================================="

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "Error: minikube is not installed"
    echo "Please install minikube from: https://minikube.sigs.k8s.io/docs/start/"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed"
    echo "Please install kubectl from: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

# Start Minikube with 2 nodes
echo "Starting Minikube cluster with 2 nodes..."
minikube start --nodes 2 --cpus 4 --memory 8192 --driver=docker

# Enable required addons
echo "Enabling required addons..."
minikube addons enable metrics-server

# Verify cluster status
echo "Verifying cluster status..."
kubectl get nodes

echo ""
echo "=========================================="
echo "Minikube cluster is ready!"
echo "Nodes in cluster:"
kubectl get nodes -o wide
echo "=========================================="
