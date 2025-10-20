# Ray Parallel Training Demo

A comprehensive demonstration of Ray's distributed training capabilities using TensorFlow/Keras on a Minikube cluster. This project showcases the performance benefits of parallel training by comparing baseline single-node training with Ray's distributed training across multiple workers.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Running the Benchmark](#running-the-benchmark)
- [Understanding the Results](#understanding-the-results)
- [Project Structure](#project-structure)
- [Official Documentation](#official-documentation)
- [Troubleshooting](#troubleshooting)

## Overview

This project demonstrates:
- Setting up a 2-node Minikube Kubernetes cluster
- Deploying Ray cluster using KubeRay operator
- Training a CNN model on CIFAR-10 dataset
- Comparing training performance with and without Ray
- Measuring speedup and efficiency gains

### Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Dataset**: CIFAR-10 (60,000 32x32 color images in 10 classes)
- **Framework**: TensorFlow 2.15 / Keras
- **Layers**:
  - 3 convolutional blocks with batch normalization and dropout
  - Dense layers with 512 units
  - 10-class softmax output

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Minikube Cluster                      │
│                                                          │
│  ┌──────────────┐              ┌──────────────┐        │
│  │   Node 1     │              │   Node 2     │        │
│  │              │              │              │        │
│  │ ┌──────────┐ │              │ ┌──────────┐ │        │
│  │ │Ray Head  │ │              │ │Ray Worker│ │        │
│  │ │  Pod     │◄├──────────────┤►│  Pod 1   │ │        │
│  │ └──────────┘ │              │ └──────────┘ │        │
│  │              │              │              │        │
│  │              │              │ ┌──────────┐ │        │
│  │              │              │ │Ray Worker│ │        │
│  │              │◄─────────────┤►│  Pod 2   │ │        │
│  │              │              │ └──────────┘ │        │
│  └──────────────┘              └──────────────┘        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### Quick Install (Automated)

Run the automated installation script to install all prerequisites (except Docker):

```bash
./scripts/install-prerequisites.sh
```

This script will install:
- kubectl (Kubernetes CLI)
- Minikube (Local Kubernetes cluster)
- Helm (Kubernetes package manager)

**Note**: Docker must be installed manually before running the script.

### Manual Installation

Ensure you have the following installed:

- **Docker**: Container runtime (required - must be installed manually)
  - Install: https://docs.docker.com/get-docker/

- **Minikube**: Local Kubernetes cluster
  - Install: https://minikube.sigs.k8s.io/docs/start/
  - Version: 1.30.0 or later

- **kubectl**: Kubernetes CLI
  - Install: https://kubernetes.io/docs/tasks/tools/

- **Helm**: Kubernetes package manager
  - Install: https://helm.sh/docs/intro/install/
  - Version: 3.0 or later

- **Python 3.10+**: For local development (optional)
  - Install: https://www.python.org/downloads/

### System Requirements

- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM
- **Disk**: 20GB+ free space

## Quick Start

```bash
# 0. Install prerequisites (if not already installed)
./scripts/install-prerequisites.sh

# 1. Clone and navigate to the project
cd ray-parallel-training-demo

# 2. Setup Minikube cluster
./scripts/setup-minikube.sh

# 3. Install Ray on Kubernetes
./scripts/install-ray.sh

# 4. Run the benchmark
./scripts/run-benchmark.sh
```

That's it! The benchmark will run both baseline and Ray distributed training, then display a comparison report.

## Detailed Setup

### Step 1: Setup Minikube Cluster

```bash
cd scripts
./setup-minikube.sh
```

This script will:
- Start a 2-node Minikube cluster
- Allocate 4 CPUs and 8GB RAM
- Enable metrics-server addon
- Verify cluster status

Verify the setup:
```bash
kubectl get nodes
```

Expected output:
```
NAME           STATUS   ROLES           AGE   VERSION
minikube       Ready    control-plane   1m    v1.28.3
minikube-m02   Ready    <none>          1m    v1.28.3
```

### Step 2: Install Ray Cluster

```bash
./install-ray.sh
```

This script will:
- Add KubeRay Helm repository
- Install KubeRay operator
- Deploy Ray cluster with 1 head node and 2 worker nodes
- Wait for cluster to be ready

Verify Ray cluster:
```bash
kubectl get pods -n ray-system
```

Expected output:
```
NAME                                READY   STATUS    RESTARTS   AGE
kuberay-operator-...                1/1     Running   0          2m
ray-training-cluster-head-...       1/1     Running   0          1m
ray-training-cluster-worker-...     1/1     Running   0          1m
ray-training-cluster-worker-...     1/1     Running   0          1m
```

### Step 3: Access Ray Dashboard (Optional)

```bash
# Port forward to Ray dashboard
kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265

# Open browser to http://localhost:8265
```

## Running the Benchmark

### Full Benchmark

Run the complete benchmark comparing baseline vs. Ray training:

```bash
./scripts/run-benchmark.sh
```

You can customize the training parameters:

```bash
# Run with 20 epochs
EPOCHS=20 ./scripts/run-benchmark.sh

# Run with custom batch size
BATCH_SIZE=256 ./scripts/run-benchmark.sh

# Combine parameters
EPOCHS=15 BATCH_SIZE=256 ./scripts/run-benchmark.sh
```

### Manual Training

You can also run training scripts individually:

#### Baseline Training (No Ray)

```bash
# Get Ray head pod name
RAY_HEAD_POD=$(kubectl get pods -n ray-system -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

# Copy script to pod
kubectl cp training/baseline_training.py ray-system/$RAY_HEAD_POD:/home/ray/

# Run training
kubectl exec -it -n ray-system $RAY_HEAD_POD -- \
  python /home/ray/baseline_training.py --epochs 10
```

#### Ray Distributed Training

```bash
# Copy script to pod
kubectl cp training/ray_training.py ray-system/$RAY_HEAD_POD:/home/ray/

# Run distributed training
kubectl exec -it -n ray-system $RAY_HEAD_POD -- \
  python /home/ray/ray_training.py --ray-address auto --num-workers 2 --epochs 10
```

## Understanding the Results

### Benchmark Output

After running the benchmark, you'll see a comparison report:

```
======================================================================
                    BENCHMARK RESULTS COMPARISON
======================================================================

Training Time:
  Baseline (No Ray):        15.42 minutes (925.20s)
  Ray (2 Workers):          8.73 minutes (523.80s)
  Time Saved:               6.69 minutes (401.40s)
  Speedup:                  1.77x

Model Performance:
  Baseline Test Accuracy:   0.7234
  Ray Test Accuracy:        0.7198
  Accuracy Difference:      0.0036

Configuration:
  Epochs:                   10
  Batch Size:               128
  Ray Workers:              2
  Ray Cluster CPUs:         8.0
  Ray Cluster Nodes:        2

======================================================================
Speedup with Ray: 1.77x faster!
Time saved: 6.69 minutes
======================================================================
```

### Result Files

All results are saved in the `results/` directory:

1. **baseline_results.json**: Baseline training metrics
2. **ray_results_2workers.json**: Ray distributed training metrics
3. **comparison.json**: Side-by-side comparison with speedup calculations

Example `comparison.json`:
```json
{
  "baseline": {
    "training_type": "baseline",
    "training_time_seconds": 925.2,
    "test_accuracy": 0.7234
  },
  "ray_distributed": {
    "training_type": "ray_distributed",
    "num_workers": 2,
    "training_time_seconds": 523.8,
    "test_accuracy": 0.7198
  },
  "comparison": {
    "speedup": 1.77,
    "time_saved_minutes": 6.69
  }
}
```

### What to Expect

- **Speedup**: Typically 1.5x - 2x with 2 workers on small cluster
- **Accuracy**: Should be within ±1-2% between baseline and Ray training
- **Time**: Depends on hardware, but Ray should always be faster

## Project Structure

```
ray-parallel-training-demo/
├── README.md                          # This file
├── docs/
│   └── EXERCISE.md                    # Step-by-step exercise guide
├── scripts/
│   ├── setup-minikube.sh             # Minikube cluster setup
│   ├── install-ray.sh                # Ray installation script
│   ├── run-benchmark.sh              # Benchmark runner
│   └── cleanup.sh                    # Cleanup resources
├── k8s/
│   ├── ray-cluster.yaml              # Ray cluster configuration
│   └── ray-service.yaml              # Ray service configuration
├── training/
│   ├── baseline_training.py          # Baseline TF/Keras training
│   └── ray_training.py               # Ray distributed training
├── docker/
│   ├── Dockerfile                    # Custom Docker image
│   ├── requirements.txt              # Python dependencies
│   └── build-and-push.sh            # Docker build script
└── results/                          # Generated benchmark results
    ├── baseline_results.json
    ├── ray_results_2workers.json
    └── comparison.json
```

## Official Documentation

### Ray Documentation

- **Ray Train Overview**: https://docs.ray.io/en/latest/train/train.html
- **Distributed TensorFlow with Ray**: https://docs.ray.io/en/latest/train/distributed-tensorflow-keras.html
- **KubeRay Documentation**: https://docs.ray.io/en/latest/cluster/kubernetes/index.html
- **Ray Train API Reference**: https://docs.ray.io/en/latest/train/api.html
- **Ray Architecture**: https://docs.ray.io/en/latest/ray-core/key-concepts.html

### Kubernetes & Minikube

- **Minikube Documentation**: https://minikube.sigs.k8s.io/docs/
- **kubectl Cheat Sheet**: https://kubernetes.io/docs/reference/kubectl/cheatsheet/
- **Helm Documentation**: https://helm.sh/docs/

### TensorFlow

- **TensorFlow Distributed Training**: https://www.tensorflow.org/guide/distributed_training
- **Keras API Reference**: https://keras.io/api/

## Troubleshooting

See [docs/EXERCISE.md](docs/EXERCISE.md) for detailed troubleshooting guide.

### Common Issues

#### Minikube won't start
```bash
# Delete and restart
minikube delete
./scripts/setup-minikube.sh
```

#### Ray pods not starting
```bash
# Check pod status
kubectl describe pod -n ray-system <pod-name>

# Check operator logs
kubectl logs -n ray-system deployment/kuberay-operator
```

#### Out of memory errors
```bash
# Increase Minikube memory
minikube delete
minikube start --nodes 2 --cpus 4 --memory 16384
```

#### Script permission denied
```bash
chmod +x scripts/*.sh
```

### Getting Help

- **Ray Slack Community**: https://ray-distributed.slack.com
- **Ray GitHub Issues**: https://github.com/ray-project/ray/issues
- **KubeRay GitHub**: https://github.com/ray-project/kuberay

## Cleanup

To remove all resources:

```bash
./scripts/cleanup.sh

# To also stop Minikube (optional)
minikube stop

# To completely delete Minikube cluster (optional)
minikube delete
```

## Next Steps

1. **Experiment with more workers**: Modify `k8s/ray-cluster.yaml` to add more workers
2. **Try different models**: Replace the CNN with other architectures
3. **Use different datasets**: Switch from CIFAR-10 to other datasets
4. **Enable GPU support**: Modify configurations for GPU-accelerated training
5. **Add monitoring**: Integrate Prometheus and Grafana for metrics
6. **Try Ray Tune**: Add hyperparameter optimization with Ray Tune

## License

This project is provided as-is for educational and demonstration purposes.

## Contributing

Feel free to submit issues and enhancement requests!

---

**Happy Training with Ray!** 🚀
