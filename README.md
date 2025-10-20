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
- Building a custom Ray Docker image with TensorFlow
- Deploying Ray cluster using KubeRay operator
- Training a CNN model on MNIST dataset
- Comparing training performance with and without Ray
- Measuring speedup and efficiency gains

### Model Details

- **Architecture**: Lightweight Convolutional Neural Network (CNN)
- **Dataset**: MNIST (70,000 28x28 grayscale images of handwritten digits)
- **Framework**: TensorFlow 2.15 / Keras
- **Layers**:
  - 2 convolutional blocks (32 and 64 filters)
  - MaxPooling after each conv block
  - Dense layer with 128 units and dropout
  - 10-class softmax output
- **Training Time**: ~2-3 minutes per training run (baseline or Ray)

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

# 3. Install Ray on Kubernetes (includes monitoring)
./scripts/install-ray.sh

# 4. Run the benchmark
./scripts/run-benchmark.sh
```

That's it! The benchmark will run both baseline and Ray distributed training, then display a comparison report.

**Note:** The install-ray.sh script now automatically installs:
- Custom Ray image with TensorFlow
- KubeRay operator
- Ray cluster (1 head + 2 workers)
- Prometheus (metrics collection)
- Grafana (visualization with time-series charts)

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

### Step 2: Install Ray Cluster and Monitoring

```bash
./install-ray.sh
```

This script will:
1. **Build custom Docker image** with TensorFlow and dependencies (3-5 minutes first time)
2. Add KubeRay Helm repository and install operator
3. Deploy Ray cluster with 1 head node and 2 worker nodes
4. **Install monitoring stack** (Prometheus + Grafana)
5. Wait for all services to be ready

**What gets installed:**
- Custom Ray image with: TensorFlow 2.15.0, NumPy, Pandas, Matplotlib, scikit-learn
- Ray cluster (1 head + 2 workers)
- Prometheus (metrics collection)
- Grafana (visualization and dashboards)

**Time estimate:**
- First installation: 7-10 minutes (includes Docker build)
- Reinstallation: 3-5 minutes (Docker image cached)

**Docker Image Caching:**
The script automatically caches the built Docker image in Minikube. On subsequent runs, if the image already exists, the build step is skipped, saving 3-5 minutes. To force a rebuild:
```bash
minikube image rm ray-training-demo:latest
./scripts/install-ray.sh
```

Verify installation:
```bash
kubectl get pods -n ray-system
```

Expected output:
```
NAME                                READY   STATUS    RESTARTS   AGE
kuberay-operator-...                1/1     Running   0          3m
ray-training-cluster-head-...       1/1     Running   0          2m
ray-training-cluster-worker-...     1/1     Running   0          2m
ray-training-cluster-worker-...     1/1     Running   0          2m
prometheus-...                      1/1     Running   0          1m
grafana-...                         1/1     Running   0          1m
```

### Step 3: Access Ray Dashboard (Optional)

The Ray dashboard provides real-time monitoring of your cluster, jobs, and resources.

**Quick Access (Recommended):**

```bash
./scripts/open-dashboard.sh
```

This script will:
- Display the dashboard URL
- Show cluster status
- Optionally open the dashboard in your browser

**Manual Access Methods:**

**Method 1: NodePort (Direct Access)**

The dashboard is exposed via NodePort on port 30265:

```bash
# Get Minikube IP
minikube ip

# Open browser to: http://<minikube-ip>:30265
# Example: http://192.168.49.2:30265
```

**Method 2: Port Forwarding (Localhost)**

Access via localhost using port forwarding:

```bash
# In a separate terminal, run:
kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265

# Then open browser to: http://localhost:8265
```

**Dashboard Features:**

The Ray dashboard provides:
- **Cluster Overview**: Resource usage, nodes, and workers
- **Jobs**: Monitor running and completed jobs
- **Actors & Tasks**: Visualize distributed tasks
- **Logs**: View real-time logs from all nodes
- **Metrics**: Performance metrics and graphs

**Note:** The Ray dashboard does NOT require authentication by default. It's accessible without login credentials.

**Monitoring Features:**

Since the monitoring stack (Prometheus + Grafana) is automatically installed:
- **Time-series charts**: Powered by Prometheus (accessible internally)
- **Resource graphs**: CPU, memory, network usage over time
- **Job metrics**: Task duration, throughput trends

**Note about Grafana Integration:**
The Ray dashboard uses Prometheus directly for time-series charts. Grafana is available separately for creating custom dashboards and advanced visualization.

**Access Grafana** (for custom dashboards):
- URL: `http://<minikube-ip>:30300`
- Default credentials: `admin` / `admin`
- Create custom dashboards and set up alerts
- View Prometheus datasource metrics

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
  Baseline (No Ray):        2.35 minutes (141.20s)
  Ray (2 Workers):          1.42 minutes (85.30s)
  Time Saved:               0.93 minutes (55.90s)
  Speedup:                  1.66x

Model Performance:
  Baseline Test Accuracy:   0.9876
  Ray Test Accuracy:        0.9868
  Accuracy Difference:      0.0008

Configuration:
  Epochs:                   5
  Batch Size:               128
  Ray Workers:              2
  Ray Cluster CPUs:         8.0
  Ray Cluster Nodes:        2

======================================================================
Speedup with Ray: 1.66x faster!
Time saved: 0.93 minutes
Total benchmark time: ~4 minutes
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
    "training_time_seconds": 141.2,
    "test_accuracy": 0.9876,
    "epochs_completed": 5
  },
  "ray_distributed": {
    "training_type": "ray_distributed",
    "num_workers": 2,
    "training_time_seconds": 85.3,
    "test_accuracy": 0.9868,
    "epochs_completed": 5
  },
  "comparison": {
    "speedup": 1.66,
    "time_saved_minutes": 0.93,
    "time_saved_seconds": 55.9
  }
}
```

### What to Expect

- **Speedup**: Typically 1.5x - 2x with 2 workers on small cluster
- **Accuracy**: Should reach ~98-99% on MNIST test set (±1% between runs)
- **Time**: Baseline ~2-3 minutes, Ray ~1.5-2 minutes on typical hardware
- **Total Benchmark Time**: ~5-7 minutes for both baseline and Ray training combined

## Development & Testing Cycle

### Fast Iteration for Development

The project is optimized for rapid development and testing iterations with intelligent Docker image caching.

#### First Installation (Full Setup)
```bash
./scripts/install-ray.sh
# Time: 7-10 minutes (includes Docker build)
```

#### Rapid Reinstallation (Cached Image)

If you need to reinstall the Ray cluster (e.g., testing configuration changes), the Docker image is automatically reused:

```bash
# Delete cluster
kubectl delete raycluster -n ray-system ray-training-cluster

# Reinstall (skips Docker build!)
./scripts/install-ray.sh
# Time: 3-5 minutes (60% faster!)
```

**Why faster?** The Docker image (`ray-training-demo:latest`) is cached in Minikube and automatically reused if it already exists.

#### Development Workflow

**Typical iteration cycle:**

```bash
# 1. Make changes to Ray cluster configuration
vim k8s/ray-cluster.yaml

# 2. Delete current cluster
kubectl delete raycluster -n ray-system ray-training-cluster

# 3. Reinstall (fast with cached image)
./scripts/install-ray.sh
# ~3-5 minutes

# 4. Test changes
./scripts/run-benchmark.sh
```

#### When to Force Image Rebuild

Rebuild the Docker image when you change:
- Python dependencies (`docker/requirements.txt`)
- Dockerfile
- Training scripts that are baked into the image

```bash
# Force rebuild
minikube image rm ray-training-demo:latest
./scripts/install-ray.sh
# Will rebuild from scratch (~7-10 minutes)
```

#### Time Comparison

| Scenario | First Time | Subsequent Runs | Time Saved |
|----------|-----------|-----------------|------------|
| **Full installation** | 7-10 min | 3-5 min | **4-5 min** |
| **Docker build only** | 3-5 min | instant | **3-5 min** |
| **Cluster deploy only** | 3-5 min | 3-5 min | 0 min |

#### Quick Commands Reference

```bash
# Check if image is cached
minikube image ls | grep ray-training-demo

# Remove cached image (force rebuild next time)
minikube image rm ray-training-demo:latest

# Restart Ray cluster (keeps image)
kubectl delete raycluster -n ray-system ray-training-cluster
./scripts/install-ray.sh

# Full cleanup and reinstall
./scripts/cleanup.sh
./scripts/install-ray.sh
```

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
2. **Try different models**: Replace the CNN with other architectures (ResNet, VGG, etc.)
3. **Use different datasets**: Switch from MNIST to CIFAR-10, Fashion-MNIST, or custom datasets
4. **Enable GPU support**: Modify configurations for GPU-accelerated training
5. **Try Ray Tune**: Add hyperparameter optimization with Ray Tune
6. **Increase model complexity**: Try deeper networks or larger batch sizes to see more dramatic speedup

## License

This project is provided as-is for educational and demonstration purposes.

## Contributing

Feel free to submit issues and enhancement requests!

---

**Happy Training with Ray!** 🚀
