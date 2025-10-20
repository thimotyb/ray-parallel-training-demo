# Quick Reference Guide

## Common Commands

### Minikube

```bash
# Start cluster
minikube start --nodes 2 --cpus 4 --memory 8192

# Stop cluster
minikube stop

# Delete cluster
minikube delete

# Get cluster status
minikube status

# Access dashboard
minikube dashboard

# SSH into node
minikube ssh
minikube ssh -n minikube-m02  # For node 2

# View logs
minikube logs
```

### Kubectl

```bash
# Get nodes
kubectl get nodes

# Get all resources in namespace
kubectl get all -n ray-system

# Get pods with details
kubectl get pods -n ray-system -o wide

# Describe pod
kubectl describe pod -n ray-system <pod-name>

# Get pod logs
kubectl logs -n ray-system <pod-name>
kubectl logs -n ray-system <pod-name> -f  # Follow logs

# Execute command in pod
kubectl exec -it -n ray-system <pod-name> -- bash

# Copy files to/from pod
kubectl cp local-file.py ray-system/<pod-name>:/remote/path/
kubectl cp ray-system/<pod-name>:/remote/file.json ./local-file.json

# Port forward
kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265

# Delete resource
kubectl delete pod -n ray-system <pod-name>
kubectl delete raycluster -n ray-system ray-training-cluster

# Resource usage
kubectl top nodes
kubectl top pods -n ray-system
```

### Helm

```bash
# List repositories
helm repo list

# Update repositories
helm repo update

# List installed releases
helm list -n ray-system

# Get release status
helm status kuberay-operator -n ray-system

# Uninstall release
helm uninstall kuberay-operator -n ray-system
```

### Ray

```bash
# Get Ray head pod
RAY_HEAD_POD=$(kubectl get pods -n ray-system -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

# Check Ray cluster status
kubectl exec -it -n ray-system $RAY_HEAD_POD -- ray status

# View Ray logs
kubectl exec -it -n ray-system $RAY_HEAD_POD -- ray logs

# Python REPL in Ray head
kubectl exec -it -n ray-system $RAY_HEAD_POD -- python

# Interactive bash session
kubectl exec -it -n ray-system $RAY_HEAD_POD -- bash
```

## Project Scripts

```bash
# Setup Minikube cluster
./scripts/setup-minikube.sh

# Install Ray on Kubernetes
./scripts/install-ray.sh

# Run benchmark (both baseline + Ray)
./scripts/run-benchmark.sh both

# Baseline only
./scripts/run-benchmark.sh baseline

# Ray distributed only
./scripts/run-benchmark.sh ray

# Run with custom parameters
EPOCHS=20 BATCH_SIZE=256 ./scripts/run-benchmark.sh both

# Cleanup resources
./scripts/cleanup.sh
```

## Training Commands

### Baseline Training

```bash
RAY_HEAD_POD=$(kubectl get pods -n ray-system -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

# Copy script
kubectl cp training/baseline_training.py ray-system/$RAY_HEAD_POD:/home/ray/

# Run training
kubectl exec -it -n ray-system $RAY_HEAD_POD -- \
  python /home/ray/baseline_training.py --epochs 10 --batch-size 128

# Get results
kubectl cp ray-system/$RAY_HEAD_POD:/home/ray/baseline_results.json ./results/
```

### Ray Distributed Training

```bash
# Copy script
kubectl cp training/ray_training.py ray-system/$RAY_HEAD_POD:/home/ray/

# Run with 2 workers
kubectl exec -it -n ray-system $RAY_HEAD_POD -- \
  python /home/ray/ray_training.py \
  --ray-address auto \
  --num-workers 2 \
  --epochs 10 \
  --batch-size 128

# Run with 3 workers
kubectl exec -it -n ray-system $RAY_HEAD_POD -- \
  python /home/ray/ray_training.py \
  --ray-address auto \
  --num-workers 3 \
  --epochs 10

# Get results
kubectl cp ray-system/$RAY_HEAD_POD:/home/ray/ray_results_2workers.json ./results/
```

## Debugging

### Check Cluster Health

```bash
# Node status
kubectl get nodes
kubectl describe node <node-name>

# Pod status
kubectl get pods -n ray-system
kubectl describe pod -n ray-system <pod-name>

# Resource usage
kubectl top nodes
kubectl top pods -n ray-system

# Events
kubectl get events -n ray-system --sort-by='.lastTimestamp'

# Ray cluster status
kubectl get raycluster -n ray-system
kubectl describe raycluster -n ray-system ray-training-cluster
```

### View Logs

```bash
# Ray operator logs
kubectl logs -n ray-system deployment/kuberay-operator

# Ray head logs
kubectl logs -n ray-system <head-pod-name>

# Ray worker logs
kubectl logs -n ray-system <worker-pod-name>

# Follow logs in real-time
kubectl logs -n ray-system <pod-name> -f

# Previous container logs (after restart)
kubectl logs -n ray-system <pod-name> --previous
```

### Common Issues

#### Pods Stuck in Pending

```bash
# Check why pod is pending
kubectl describe pod -n ray-system <pod-name>

# Check available resources
kubectl describe nodes

# Check if image can be pulled
kubectl get events -n ray-system | grep -i pull
```

#### Pods Crashing

```bash
# View logs
kubectl logs -n ray-system <pod-name>

# Check resource limits
kubectl describe pod -n ray-system <pod-name> | grep -A 5 Limits

# Check liveness/readiness probes
kubectl describe pod -n ray-system <pod-name> | grep -A 10 Liveness
```

#### Ray Workers Not Connecting

```bash
# Check Ray head service
kubectl get svc -n ray-system

# Test connectivity from worker
kubectl exec -it -n ray-system <worker-pod-name> -- \
  nc -zv ray-training-cluster-head-svc 6379

# Check Ray status
kubectl exec -it -n ray-system $RAY_HEAD_POD -- ray status
```

## Useful Kubernetes Manifests

### Scale Workers

```bash
# Edit ray-cluster.yaml and change replicas
kubectl edit raycluster -n ray-system ray-training-cluster

# Or apply updated file
kubectl apply -f k8s/ray-cluster.yaml
```

### Restart Components

```bash
# Delete and recreate Ray cluster
kubectl delete raycluster -n ray-system ray-training-cluster
kubectl apply -f k8s/ray-cluster.yaml

# Restart operator
kubectl rollout restart deployment/kuberay-operator -n ray-system

# Delete specific pod (will be recreated)
kubectl delete pod -n ray-system <pod-name>
```

## Environment Variables

```bash
# Set custom epochs
export EPOCHS=20

# Set custom batch size
export BATCH_SIZE=256

# Use environment variables
EPOCHS=15 BATCH_SIZE=128 ./scripts/run-benchmark.sh both
```

## File Locations

```
Configuration Files:
  k8s/ray-cluster.yaml          - Ray cluster definition
  k8s/ray-service.yaml          - Service configuration
  docker/Dockerfile             - Custom Docker image

Training Scripts:
  training/baseline_training.py - Non-distributed training
  training/ray_training.py      - Ray distributed training

Setup Scripts:
  scripts/setup-minikube.sh     - Minikube setup
  scripts/install-ray.sh        - Ray installation
  scripts/run-benchmark.sh      - Run benchmark
  scripts/cleanup.sh            - Cleanup resources

Results:
  results/baseline_results.json      - Baseline metrics
  results/ray_results_2workers.json  - Ray metrics
  results/comparison.json            - Comparison report
```

## Ray Dashboard URLs

```bash
# Local port forward
kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265

# Access at:
http://localhost:8265

# Direct Minikube service
minikube service ray-cluster-head-svc -n ray-system --url
```

## Docker Commands (If Using Custom Image)

```bash
# Build image
cd docker
docker build -t ray-training-demo:latest -f Dockerfile ..

# Load into Minikube
minikube image load ray-training-demo:latest

# List images in Minikube
minikube ssh docker images

# Run build script
./docker/build-and-push.sh
```

## Monitoring

```bash
# Watch pods
watch kubectl get pods -n ray-system

# Watch resource usage
watch kubectl top pods -n ray-system

# Stream events
kubectl get events -n ray-system -w

# Monitor training logs
kubectl logs -n ray-system $RAY_HEAD_POD -f | grep -i "epoch\|accuracy"
```

## Cleanup Commands

```bash
# Quick cleanup (keep Minikube running)
./scripts/cleanup.sh

# Full cleanup
kubectl delete namespace ray-system
helm uninstall kuberay-operator -n ray-system
minikube stop
minikube delete

# Force delete stuck resources
kubectl delete raycluster -n ray-system ray-training-cluster --grace-period=0 --force
kubectl delete namespace ray-system --grace-period=0 --force
```

## Tips and Tricks

### Faster Iterations

```bash
# Use fewer epochs for testing
EPOCHS=5 ./scripts/run-benchmark.sh both

# Keep results from previous runs
mkdir -p results/run-$(date +%Y%m%d-%H%M%S)
mv results/*.json results/run-$(date +%Y%m%d-%H%M%S)/
```

### Debugging Training

```bash
# Interactive Python session in Ray head
kubectl exec -it -n ray-system $RAY_HEAD_POD -- python

# Inside Python:
import ray
ray.init(address='auto')
print(ray.cluster_resources())
print(ray.nodes())
```

### Performance Monitoring

```bash
# Monitor CPU/Memory during training
watch -n 2 'kubectl top pods -n ray-system'

# Save resource metrics
kubectl top pods -n ray-system > metrics-$(date +%Y%m%d-%H%M%S).txt
```

## Official Links

- Ray Documentation: https://docs.ray.io
- Ray Train: https://docs.ray.io/en/latest/train/train.html
- KubeRay: https://docs.ray.io/en/latest/cluster/kubernetes/index.html
- Minikube Docs: https://minikube.sigs.k8s.io/docs/
- Kubernetes Docs: https://kubernetes.io/docs/

---

**Save this for quick reference during development!**
