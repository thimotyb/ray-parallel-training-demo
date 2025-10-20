# Agent Instructions - Ray Parallel Training Demo

This document provides complete context for AI agents (or developers) picking up this project in a new session. It contains the original user requirements, implementation decisions, project structure, and instructions for continuing work.

## Original User Request

**Date**: 2025-10-20

**Request Summary**:
Create a sample project to demonstrate Ray's capability to start parallel training with the following specifications:

1. **Model**: Simple neural network (CNN) using Keras and TensorFlow
2. **Dataset**: Simple but large enough to show benefits of parallel training (CIFAR-10)
3. **Infrastructure**: Ray running in a small Minikube cluster
4. **Cluster**: 2-node Minikube cluster
5. **Comparison**: Time the benefits with and without Ray training
6. **Deliverables**:
   - Complete setup scripts
   - Exercise documentation
   - Relevant links to Ray's official documentation

## Implementation Decisions Made

### Technology Stack

- **Ray Version**: 2.9.0 (stable version with good TensorFlow support)
- **TensorFlow**: 2.15.0 (latest stable)
- **Python**: 3.10 (compatible with Ray 2.9.0)
- **Kubernetes**: Via Minikube with 2 nodes
- **Ray Deployment**: KubeRay operator with Helm

### Model & Dataset Choice

**Model**: Convolutional Neural Network (CNN)
- 3 convolutional blocks (64, 128, 256 filters)
- Batch normalization and dropout for regularization
- Dense layers with 512 units
- ~3M parameters total
- Complex enough to show distributed training benefits

**Why this architecture?**
- Not too small (would train too fast, minimal speedup)
- Not too large (would exceed Minikube resource limits)
- Representative of real-world image classification tasks

**Dataset**: CIFAR-10
- 60,000 32x32 color images (50K train, 10K test)
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Built into Keras (no manual download)
- Large enough to show parallelization benefits
- Small enough to run on limited resources

**Why CIFAR-10?**
- Standard ML benchmark
- Readily available
- Reasonable training time (10-20 minutes baseline)
- Shows clear speedup with 2 workers

### Cluster Configuration

**Minikube Setup**:
- 2 nodes (minikube + minikube-m02)
- 4 CPUs per node
- 8GB RAM per node
- Docker driver

**Ray Cluster Layout**:
- 1 Ray head node (pod on node 1)
- 2 Ray worker nodes (pods on node 2)
- Resources per pod: 1-2 CPUs, 2-4GB RAM

**Why this configuration?**
- Minimal resource requirements (runnable on laptops)
- Clear demonstration of distributed training
- Realistic multi-node setup
- Workers on separate node shows true distribution

### Training Approach

**Baseline Training** (`baseline_training.py`):
- Standard Keras `model.fit()`
- Single process, single node
- No parallelization
- Serves as performance baseline

**Ray Distributed Training** (`ray_training.py`):
- Ray Train with TensorflowTrainer
- MultiWorkerMirroredStrategy (data parallelism)
- 2 workers by default (configurable)
- Automatic gradient synchronization

**Why this comparison?**
- Shows Ray's value proposition clearly
- Minimal code changes between versions
- Realistic comparison (not contrived)
- Measurable speedup (~1.5-2x expected)

## Project Structure Created

```
ray-parallel-training-demo/
├── README.md                   # Main documentation (12KB)
├── GETTING_STARTED.md          # Step-by-step checklist (7KB)
├── AGENTS.md                   # This file - agent context
├── .gitignore                  # Standard Python/TF/Ray gitignore
│
├── scripts/                    # Executable automation scripts
│   ├── setup-minikube.sh      # Creates 2-node Minikube cluster
│   ├── install-ray.sh         # Deploys KubeRay operator + Ray cluster
│   ├── run-benchmark.sh       # Runs both trainings + comparison
│   └── cleanup.sh             # Removes all resources
│
├── k8s/                        # Kubernetes manifests
│   ├── ray-cluster.yaml       # RayCluster CRD (1 head + 2 workers)
│   └── ray-service.yaml       # NodePort service for dashboard
│
├── training/                   # Python training scripts
│   ├── baseline_training.py   # Single-node training (5.7KB)
│   └── ray_training.py        # Ray distributed training (9.4KB)
│
├── docker/                     # Docker configuration
│   ├── Dockerfile             # Custom image (Ray + TensorFlow)
│   ├── requirements.txt       # Python dependencies
│   └── build-and-push.sh     # Build and load into Minikube
│
└── docs/                       # Additional documentation
    ├── EXERCISE.md            # 60-90 min hands-on exercise (14KB)
    └── QUICK_REFERENCE.md    # Command cheat sheet (9KB)
```

## File-by-File Implementation Details

### Scripts (`scripts/`)

#### `setup-minikube.sh`
- Checks for minikube and kubectl installation
- Starts Minikube with `--nodes 2 --cpus 4 --memory 8192 --driver=docker`
- Enables metrics-server addon
- Verifies cluster status with `kubectl get nodes`

#### `install-ray.sh`
- Checks for Helm installation
- Adds KubeRay Helm repository
- Creates `ray-system` namespace
- Installs KubeRay operator via Helm
- Deploys Ray cluster from `k8s/ray-cluster.yaml`
- Waits for all pods to be ready

#### `run-benchmark.sh`
- Gets Ray head pod name dynamically
- Runs baseline training first (~10-15 min)
- Copies results back from pod
- Runs Ray distributed training (~6-8 min)
- Copies Ray results back
- Generates comparison report using Python script (embedded)
- Displays side-by-side comparison with speedup calculation
- Saves all results to `results/` directory

#### `cleanup.sh`
- Deletes Ray cluster
- Uninstalls KubeRay operator
- Deletes ray-system namespace
- Optional: stop/delete Minikube (commented out)

### Kubernetes Manifests (`k8s/`)

#### `ray-cluster.yaml`
- **API Version**: ray.io/v1
- **Kind**: RayCluster
- **Ray Version**: 2.9.0
- **Image**: rayproject/ray:2.9.0-py310
- **Head Group**:
  - 1 replica
  - 2 CPU / 4GB RAM limits
  - Dashboard on port 8265
  - GCS server on port 6379
  - Client port 10001
- **Worker Group**:
  - 2 replicas (min 1, max 3)
  - 2 CPU / 4GB RAM per worker
  - Autoscaling disabled (fixed size)

#### `ray-service.yaml`
- **Type**: NodePort
- **Ports**:
  - Dashboard: 8265 → NodePort 30265
  - Client: 10001 → NodePort 30001
  - GCS: 6379 (internal)
- **Selector**: Targets Ray head pod

### Training Scripts (`training/`)

#### `baseline_training.py`
**Key Functions**:
- `create_model()`: Builds CNN architecture
- `load_and_prepare_data()`: Loads CIFAR-10, normalizes, one-hot encodes
- `train_model()`: Main training loop with Keras

**Features**:
- Command-line arguments (epochs, batch-size, save-path)
- Callbacks: ReduceLROnPlateau, EarlyStopping
- Timing with `time.time()`
- Saves results to JSON with detailed metrics
- GPU memory growth configuration

**Output**: `baseline_results.json`
```json
{
  "training_type": "baseline",
  "training_time_seconds": 925.2,
  "epochs_completed": 10,
  "test_accuracy": 0.7234,
  ...
}
```

#### `ray_training.py`
**Key Functions**:
- `create_model()`: Same CNN as baseline
- `load_and_prepare_data()`: Same data loading
- `train_func_per_worker(config)`: Training function executed on each worker
- `train_with_ray()`: Main Ray Train orchestration

**Ray-Specific Code**:
```python
# Initialize Ray
ray.init(address='auto')

# Define per-worker training
def train_func_per_worker(config):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = create_model()
        model.compile(...)
    # Custom training loop with train.report()

# Configure distributed training
scaling_config = ScalingConfig(
    num_workers=2,
    use_gpu=False,
    resources_per_worker={"CPU": 1}
)

trainer = TensorflowTrainer(
    train_loop_per_worker=train_func_per_worker,
    scaling_config=scaling_config
)

result = trainer.fit()
```

**Features**:
- Command-line arguments including `--num-workers`
- Reports metrics to Ray Train each epoch
- Displays cluster info (CPUs, nodes, memory)
- Saves results with cluster metadata

**Output**: `ray_results_2workers.json`
```json
{
  "training_type": "ray_distributed",
  "num_workers": 2,
  "training_time_seconds": 523.8,
  "test_accuracy": 0.7198,
  "ray_cluster_cpus": 8.0,
  ...
}
```

### Docker Configuration (`docker/`)

#### `Dockerfile`
- **Base**: rayproject/ray:2.9.0-py310
- **Additions**: TensorFlow 2.15.0, numpy, pandas, matplotlib, scikit-learn
- **Copies**: Training scripts to /home/ray/training/
- **Environment**: PYTHONUNBUFFERED=1, TF_CPP_MIN_LOG_LEVEL=2

#### `requirements.txt`
```
tensorflow==2.15.0
ray[train]==2.9.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0
```

#### `build-and-push.sh`
- Builds Docker image
- Tags for Minikube
- Loads image into Minikube with `minikube image load`

### Documentation (`docs/`)

#### `EXERCISE.md` (14KB, ~60-90 min exercise)
**Structure**:
1. Part 1: Environment Setup (15-20 min)
2. Part 2: Understanding the Code (15 min)
3. Part 3: Running the Benchmark (20-30 min)
4. Part 4: Experiments and Analysis (15-20 min)
5. Part 5: Distributed Training Concepts (10 min)
6. Part 6: Troubleshooting Common Issues
7. Part 7: Cleanup
8. Part 8: Advanced Challenges (Optional)

**Includes**:
- Checkpoint questions throughout
- Code walkthroughs with explanations
- Hands-on experiments (vary workers, batch size, epochs)
- Conceptual explanations (data parallelism, speedup, efficiency)
- Detailed troubleshooting for 5 common issue categories

#### `QUICK_REFERENCE.md` (9KB)
**Sections**:
- Common kubectl, helm, minikube commands
- Ray-specific commands
- Debugging commands
- File locations
- Monitoring tips
- Cleanup commands

### Main Documentation

#### `README.md` (12KB)
**Sections**:
- Overview and architecture diagram
- Prerequisites with installation links
- Quick start (3 commands)
- Detailed setup instructions
- Running benchmarks
- Understanding results
- Project structure
- **Official Ray documentation links**:
  - Ray Train Overview
  - Distributed TensorFlow with Ray
  - KubeRay Documentation
  - Ray Train API Reference
- Troubleshooting guide
- Next steps and experiments

#### `GETTING_STARTED.md` (7KB)
Interactive checklist format:
- Pre-flight checklist (prerequisites)
- Setup checklist (step-by-step)
- Running benchmark checklist
- Verification checklist
- Troubleshooting checklist
- Cleanup checklist
- Estimated times for each section

## Expected Workflow & Results

### Standard Workflow

```bash
# 1. Initial setup (one-time, ~10 min)
./scripts/setup-minikube.sh
./scripts/install-ray.sh

# 2. Run benchmark (~20-30 min)
./scripts/run-benchmark.sh

# 3. Review results
cat results/comparison.json

# 4. Cleanup
./scripts/cleanup.sh
```

### Expected Results

**Timing**:
- Baseline training: 10-15 minutes (10 epochs)
- Ray training: 6-8 minutes (10 epochs)
- Speedup: 1.5x - 2.0x

**Accuracy**:
- Both should achieve ~70-75% test accuracy
- Difference should be < 2% between baseline and Ray
- Some variation is normal due to random initialization

**Speedup Calculation**:
```
Speedup = Baseline Time / Ray Time
Efficiency = Speedup / Num Workers

Example:
Baseline: 925s, Ray: 524s, Workers: 2
Speedup = 925 / 524 = 1.77x
Efficiency = 1.77 / 2 = 0.885 (88.5%)
```

**Why Not 2x Speedup?**
- Communication overhead (gradient synchronization)
- Data loading overhead
- Batch size effects
- Framework overhead
- Small dataset relative to model size

## Ray Concepts Demonstrated

### 1. Ray Cluster Architecture
- **Head Node**: Orchestration, scheduling, GCS (Global Control Store)
- **Worker Nodes**: Execute tasks, store objects
- **Dashboard**: Web UI for monitoring (port 8265)

### 2. Ray Train
- High-level API for distributed training
- Supports TensorFlow, PyTorch, XGBoost, etc.
- Automatic data sharding
- Checkpoint management
- Fault tolerance

### 3. Data Parallelism
- Each worker has full model copy
- Training data is partitioned
- Each worker processes different batches
- Gradients are synchronized and averaged
- Model updated with averaged gradients

### 4. TensorFlow Integration
- `MultiWorkerMirroredStrategy`: TF's distributed strategy
- Ray manages worker orchestration
- TensorFlow handles gradient synchronization
- Seamless integration via TensorflowTrainer

## Customization & Extension Points

### Changing Number of Workers

**Option 1**: Edit YAML and redeploy
```bash
# Edit k8s/ray-cluster.yaml
# Change: replicas: 3 (under workerGroupSpecs)
kubectl apply -f k8s/ray-cluster.yaml
```

**Option 2**: Command-line argument
```bash
kubectl exec -it -n ray-system $RAY_HEAD_POD -- \
  python /home/ray/ray_training.py --num-workers 3
```

### Changing Training Parameters

```bash
# Epochs
EPOCHS=20 ./scripts/run-benchmark.sh

# Batch size
BATCH_SIZE=256 ./scripts/run-benchmark.sh

# Both
EPOCHS=15 BATCH_SIZE=128 ./scripts/run-benchmark.sh
```

### Using Different Dataset

Modify `load_and_prepare_data()` in both scripts:
```python
def load_and_prepare_data():
    # Replace CIFAR-10 loading with custom dataset
    # Ensure same preprocessing for both scripts
    return (x_train, y_train), (x_test, y_test)
```

### Using Different Model

Modify `create_model()` in both scripts:
```python
def create_model():
    # Replace CNN with custom architecture
    # Keep input/output shapes compatible with dataset
    return model
```

### Adding GPU Support

1. Start Minikube with GPU:
```bash
minikube start --nodes 2 --gpus all
```

2. Edit `ray-cluster.yaml`:
```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
```

3. Update training scripts:
```python
scaling_config = ScalingConfig(
    num_workers=2,
    use_gpu=True,  # Changed from False
    resources_per_worker={"GPU": 1}
)
```

## Troubleshooting Guide for Agents

### Common Issues & Solutions

#### 1. Pods Stuck in Pending
**Symptoms**: `kubectl get pods -n ray-system` shows Pending status

**Diagnose**:
```bash
kubectl describe pod -n ray-system <pod-name>
# Look for: resource constraints, image pull errors
```

**Solutions**:
- Insufficient resources: Increase Minikube memory/CPU
- Image pull issues: Check internet connection, use `minikube cache`
- Node selector issues: Remove node affinity constraints

#### 2. Ray Workers Not Connecting
**Symptoms**: Workers can't connect to head node

**Diagnose**:
```bash
kubectl exec -it -n ray-system $RAY_HEAD_POD -- ray status
# Should show connected workers
```

**Solutions**:
- Service misconfigured: Check `kubectl get svc -n ray-system`
- Network policy: Ensure no restrictive policies
- Restart cluster: `kubectl delete raycluster` then reapply

#### 3. Out of Memory Errors
**Symptoms**: Training crashes with OOM

**Solutions**:
- Reduce batch size: `BATCH_SIZE=64`
- Reduce model size: Fewer layers or filters
- Increase Minikube memory: `--memory 16384`
- Reduce workers: Use 1 worker instead of 2

#### 4. Slow Training Performance
**Symptoms**: No speedup or slower than baseline

**Diagnose**:
```bash
kubectl top pods -n ray-system
# Check CPU/memory usage
```

**Solutions**:
- CPU throttling: Increase CPU allocation
- Too many workers: Try fewer workers (overhead dominates)
- Small batch size: Increase batch size
- Network bottleneck: Check pod placement across nodes

#### 5. Accuracy Divergence
**Symptoms**: Ray training accuracy significantly different from baseline

**Causes**:
- Different random seeds (normal, small variation OK)
- Batch size effects (larger effective batch with distributed)
- Learning rate needs adjustment for distributed training

**Solutions**:
- Set seeds: `tf.random.set_seed(42)`
- Adjust learning rate: Scale with number of workers
- Increase epochs: Distributed may need more iterations

## Official Ray Documentation References

These links were included throughout the documentation:

### Core Documentation
- **Ray Train Overview**: https://docs.ray.io/en/latest/train/train.html
  - High-level distributed training API
  - Framework integrations (TensorFlow, PyTorch, etc.)

- **Distributed TensorFlow with Ray**: https://docs.ray.io/en/latest/train/distributed-tensorflow-keras.html
  - TensorFlow-specific guide
  - MultiWorkerMirroredStrategy integration
  - Examples and best practices

- **KubeRay Documentation**: https://docs.ray.io/en/latest/cluster/kubernetes/index.html
  - Deploying Ray on Kubernetes
  - RayCluster CRD reference
  - Autoscaling and configuration

- **Ray Train API Reference**: https://docs.ray.io/en/latest/train/api.html
  - ScalingConfig parameters
  - TensorflowTrainer options
  - Checkpoint management

- **Ray Architecture**: https://docs.ray.io/en/latest/ray-core/key-concepts.html
  - Core concepts (tasks, actors, objects)
  - Cluster architecture
  - Scheduling and execution

### Additional Resources
- **Ray Slack Community**: https://ray-distributed.slack.com
- **Ray GitHub Issues**: https://github.com/ray-project/ray/issues
- **KubeRay GitHub**: https://github.com/ray-project/kuberay
- **KubeRay Examples**: https://github.com/ray-project/kuberay/tree/master/examples

## Instructions for Continuing Work

### For New Agent Sessions

When picking up this project:

1. **Read this file first** to understand the context and decisions
2. **Review user's new requirements** to understand what they want to add/change
3. **Check current state**:
   ```bash
   ls -la
   kubectl get pods -n ray-system 2>/dev/null
   minikube status 2>/dev/null
   ```
4. **Read relevant existing files** before modifying them
5. **Maintain consistency** with existing code style and structure

### For Adding New Features

**Examples of potential extensions**:

1. **Add Ray Tune for hyperparameter optimization**:
   - Create new script `training/ray_tune_training.py`
   - Add to benchmark script as optional step
   - Document in EXERCISE.md Part 8

2. **Add visualization/plotting**:
   - Create `scripts/plot_results.py`
   - Generate training curves, speedup graphs
   - Save to `results/plots/`

3. **Add model checkpointing**:
   - Modify training scripts to save checkpoints
   - Use Ray Train's checkpoint API
   - Document recovery process

4. **Add Prometheus/Grafana monitoring**:
   - Create `k8s/monitoring/` directory
   - Add Prometheus deployment
   - Create Grafana dashboards
   - Document in new `docs/MONITORING.md`

5. **Support for larger datasets**:
   - Add data loading from cloud storage (S3, GCS)
   - Implement data preprocessing pipeline
   - Document storage configuration

### Modifying Existing Code

**Guidelines**:
- Always read the entire file before editing
- Maintain the same logging/output format
- Keep command-line arguments consistent
- Update documentation if behavior changes
- Test both baseline and Ray paths if changing training code

### Testing Changes

**Minimal test**:
```bash
EPOCHS=2 ./scripts/run-benchmark.sh
```

**Full test**:
```bash
EPOCHS=10 ./scripts/run-benchmark.sh
```

**Verify**:
- Both trainings complete successfully
- Results files are generated
- Speedup is > 1.0
- No errors in pod logs

## Project Context & Constraints

### Design Constraints

1. **Resource Constraints**:
   - Must run on laptop (8-16GB RAM)
   - Total Minikube allocation: ~16GB RAM, 8 CPUs
   - Per-pod limits to prevent OOM

2. **Time Constraints**:
   - Setup should complete in 10-15 minutes
   - Full benchmark in 20-30 minutes
   - Quick test (5 epochs) in 8-12 minutes

3. **Complexity Constraints**:
   - Should be understandable by ML practitioners
   - Minimal Kubernetes knowledge required
   - Clear separation of concerns (scripts, configs, code)

4. **Educational Constraints**:
   - Must show clear value of Ray
   - Results must be reproducible
   - Code should be well-commented

### Non-Functional Requirements

1. **Reliability**:
   - Scripts should fail fast with clear errors
   - Automatic retries where appropriate
   - Cleanup should always work

2. **Usability**:
   - One-command execution where possible
   - Sensible defaults
   - Customizable via environment variables

3. **Documentation**:
   - Multiple formats for different audiences
   - Comprehensive troubleshooting
   - Links to official docs

## What Was NOT Implemented (Future Work)

The following were considered but not implemented:

1. **Custom Docker Image**:
   - Dockerfile is provided but not required
   - Uses official Ray image by default
   - Custom image would be needed for custom dependencies

2. **Persistent Storage**:
   - No PersistentVolumeClaims
   - Results are copied out of pods
   - Could add PVC for model checkpoints

3. **GPU Support**:
   - Configured for CPU-only
   - GPU config is documented but not default
   - Would need GPU-enabled Minikube

4. **Autoscaling**:
   - Ray cluster has fixed size (2 workers)
   - KubeRay supports autoscaling
   - Could enable for dynamic workloads

5. **Multi-Job Support**:
   - Single job at a time
   - Could add job queue system
   - Ray Job Submission API could be used

6. **Advanced Monitoring**:
   - Basic dashboard only
   - No Prometheus/Grafana
   - No custom metrics

7. **CI/CD**:
   - No GitHub Actions or pipelines
   - Manual execution only
   - Could add automated testing

8. **Security**:
   - No RBAC configuration
   - No network policies
   - No secrets management
   - Suitable for demo only, not production

## Summary for Agents

**What This Project Is**:
- A complete, working demonstration of Ray distributed training
- Comparison of baseline vs. Ray performance on Kubernetes
- Educational resource with comprehensive documentation
- Foundation for learning Ray Train with TensorFlow

**What This Project Is NOT**:
- Production-ready ML training infrastructure
- Enterprise-grade security implementation
- Optimized for large-scale deployments
- Framework-agnostic (TensorFlow-specific)

**Key Success Metrics**:
- ✅ Both training scripts work
- ✅ Ray shows measurable speedup (>1.5x)
- ✅ Accuracy is comparable (within 2%)
- ✅ Setup is automated and reliable
- ✅ Documentation is comprehensive
- ✅ User can understand and modify code

**When Modifying**:
- Preserve the educational value
- Maintain simplicity and clarity
- Keep resource requirements reasonable
- Update documentation alongside code
- Test end-to-end before considering done

---

**Last Updated**: 2025-10-20
**Created By**: AI Assistant (Claude)
**For User**: thimoty
**Project Status**: Complete and ready for use
