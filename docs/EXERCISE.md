# Ray Parallel Training Exercise Guide

A step-by-step hands-on exercise to learn Ray's distributed training capabilities by comparing single-node training with multi-worker distributed training on a Kubernetes cluster.

## Learning Objectives

By completing this exercise, you will:
- Understand how to deploy Ray on Kubernetes using KubeRay
- Learn to write distributed training code with Ray Train
- Compare performance between baseline and distributed training
- Measure speedup and efficiency of parallel training
- Understand distributed training concepts and best practices

## Exercise Duration

Estimated time: 60-90 minutes

## Prerequisites Check

Before starting, verify you have all required tools:

```bash
# Check Docker
docker --version
# Expected: Docker version 20.0+ or later

# Check Minikube
minikube version
# Expected: minikube version: v1.30.0 or later

# Check kubectl
kubectl version --client
# Expected: v1.25.0 or later

# Check Helm
helm version
# Expected: v3.0.0 or later

# Check Python (optional, for local testing)
python3 --version
# Expected: Python 3.10 or later
```

If any tool is missing, refer to the [Prerequisites](../README.md#prerequisites) section.

## Part 1: Environment Setup (15-20 minutes)

### Step 1.1: Start Minikube Cluster

Create a 2-node Kubernetes cluster using Minikube:

```bash
cd ray-parallel-training-demo/scripts
./setup-minikube.sh
```

**What's happening:**
- Minikube creates a 2-node cluster with 4 CPUs and 8GB RAM per node
- Metrics-server addon is enabled for resource monitoring
- Nodes are verified to be in Ready state

**Verify:**
```bash
kubectl get nodes
```

Expected output:
```
NAME           STATUS   ROLES           AGE   VERSION
minikube       Ready    control-plane   2m    v1.28.3
minikube-m02   Ready    <none>          2m    v1.28.3
```

**Checkpoint Question:** How many nodes do you see? What are their roles?

### Step 1.2: Install Ray Cluster

Deploy Ray using the KubeRay operator:

```bash
./install-ray.sh
```

**What's happening:**
- Helm installs the KubeRay operator
- Ray cluster is deployed with 1 head node and 2 worker nodes
- Pods are scheduled across the Minikube nodes

**Verify:**
```bash
kubectl get pods -n ray-system -o wide
```

Expected output:
```
NAME                                READY   STATUS    RESTARTS   AGE   NODE
kuberay-operator-...                1/1     Running   0          3m    minikube
ray-training-cluster-head-...       1/1     Running   0          2m    minikube
ray-training-cluster-worker-...     1/1     Running   0          2m    minikube-m02
ray-training-cluster-worker-...     1/1     Running   0          2m    minikube-m02
```

**Checkpoint Questions:**
1. How many worker pods are running?
2. On which nodes are the workers scheduled?

### Step 1.3: Explore Ray Dashboard (Optional)

Access the Ray dashboard to visualize the cluster:

```bash
# In a new terminal, port forward the dashboard
kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265
```

Open your browser to: http://localhost:8265

**Explore:**
- Navigate to "Cluster" tab to see nodes and resources
- Check "Jobs" tab (empty for now)
- Review "Metrics" for cluster health

## Part 2: Understanding the Code (15 minutes)

### Step 2.1: Review Baseline Training Script

Open and examine `training/baseline_training.py`:

```bash
# View the file
cat ../training/baseline_training.py
```

**Key points to understand:**
1. **Model Architecture**: CNN with 3 convolutional blocks
2. **Dataset**: CIFAR-10 (50,000 training images, 10,000 test images)
3. **Training Loop**: Standard Keras `model.fit()`
4. **No parallelization**: Runs on a single process

**Discussion Questions:**
- What is the model complexity? (Count the layers)
- How many parameters does the model have?
- What is the expected training time on a single node?

### Step 2.2: Review Ray Training Script

Open and examine `training/ray_training.py`:

```bash
cat ../training/ray_training.py
```

**Key differences from baseline:**

1. **Ray Initialization**:
   ```python
   ray.init(address='auto')  # Connect to Ray cluster
   ```

2. **Training Function**:
   ```python
   def train_func_per_worker(config):
       # This runs on each worker
       strategy = tf.distribute.MultiWorkerMirroredStrategy()
       # Training code...
   ```

3. **Ray Trainer Setup**:
   ```python
   trainer = TensorflowTrainer(
       train_loop_per_worker=train_func_per_worker,
       scaling_config=ScalingConfig(num_workers=2)
   )
   ```

4. **Distributed Training**:
   - Each worker gets a copy of the model
   - Data is automatically partitioned
   - Gradients are synchronized across workers

**Discussion Questions:**
- How does Ray distribute the training workload?
- What is the role of `MultiWorkerMirroredStrategy`?
- How are training metrics collected from workers?

### Step 2.3: Compare Training Approaches

| Aspect | Baseline Training | Ray Distributed Training |
|--------|------------------|-------------------------|
| **Parallelism** | None (single process) | Multiple workers |
| **Code Complexity** | Simple | Slightly more complex |
| **Setup Required** | None | Ray cluster needed |
| **Speedup** | 1x (baseline) | ~1.5-2x with 2 workers |
| **Scalability** | Limited | Scales to many nodes |

## Part 3: Running the Benchmark (20-30 minutes)

### Step 3.1: Run Full Benchmark

Execute both baseline and Ray training:

```bash
./run-benchmark.sh
```

This will:
1. Run baseline training (10 epochs, ~10-15 minutes)
2. Run Ray distributed training (10 epochs, ~6-8 minutes)
3. Generate comparison report

**Watch the output:**
- Observe training progress for both approaches
- Note the time taken for each epoch
- Compare final accuracies

### Step 3.2: Monitor Ray Dashboard

While training is running, check the Ray dashboard (http://localhost:8265):

**Observe:**
- "Jobs" tab shows the training job
- Worker utilization and task execution
- Resource consumption (CPU, memory)
- Task timeline and scheduling

### Step 3.3: Analyze Results

After the benchmark completes, examine the results:

```bash
cd ../results
cat comparison.json
```

**Key metrics to analyze:**

1. **Training Time**:
   ```json
   "baseline": {"training_time_seconds": 925.2},
   "ray_distributed": {"training_time_seconds": 523.8}
   ```

2. **Speedup**:
   ```json
   "comparison": {"speedup": 1.77}
   ```

3. **Accuracy**:
   Compare test accuracy between baseline and Ray training

**Checkpoint Questions:**
1. What speedup did you achieve?
2. Is the accuracy comparable between baseline and Ray?
3. What is the efficiency? (Speedup / Number of Workers)

## Part 4: Experiments and Analysis (15-20 minutes)

### Experiment 1: Different Number of Workers

Modify the Ray cluster to use different worker counts:

**Edit `k8s/ray-cluster.yaml`:**
```yaml
workerGroupSpecs:
- replicas: 3  # Change from 2 to 3
```

**Apply changes:**
```bash
kubectl apply -f ../k8s/ray-cluster.yaml
```

**Run training with 3 workers:**
```bash
RAY_HEAD_POD=$(kubectl get pods -n ray-system -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

kubectl cp ../training/ray_training.py ray-system/$RAY_HEAD_POD:/home/ray/

kubectl exec -it -n ray-system $RAY_HEAD_POD -- \
  python /home/ray/ray_training.py --ray-address auto --num-workers 3 --epochs 10
```

**Question:** Does speedup scale linearly with more workers?

### Experiment 2: Vary Batch Size

Test different batch sizes to see the impact:

```bash
# Small batch
BATCH_SIZE=64 ./run-benchmark.sh

# Large batch
BATCH_SIZE=256 ./run-benchmark.sh
```

**Question:** How does batch size affect training time and speedup?

### Experiment 3: Fewer Epochs

For faster iterations:

```bash
EPOCHS=5 ./run-benchmark.sh
```

**Question:** Does the speedup ratio remain consistent with fewer epochs?

## Part 5: Understanding Distributed Training Concepts (10 minutes)

### Data Parallelism

Ray Train uses **data parallelism**:
- Each worker has a full copy of the model
- Training data is partitioned across workers
- Each worker processes different batches
- Gradients are synchronized and averaged

**Visualization:**
```
Worker 1: Batch 1, 2, 3 → Compute gradients →
                                              ↓
Worker 2: Batch 4, 5, 6 → Compute gradients → Average → Update model
                                              ↑
Worker 3: Batch 7, 8, 9 → Compute gradients →
```

### Synchronous vs Asynchronous Training

Ray's `MultiWorkerMirroredStrategy` uses **synchronous training**:
- All workers train in lockstep
- Gradients are averaged at each step
- Ensures model consistency
- May wait for slower workers (straggler problem)

### Speedup and Efficiency

**Speedup**: `S = T_baseline / T_distributed`
- Measures how much faster distributed training is

**Efficiency**: `E = S / N` (where N = number of workers)
- Measures how well we utilize additional workers
- Perfect efficiency = 1.0 (linear speedup)
- Realistic efficiency = 0.7-0.9

**Why not perfect speedup?**
- Communication overhead (gradient synchronization)
- Load imbalance
- Framework overhead

## Part 6: Troubleshooting Common Issues

### Issue 1: Pods Not Starting

**Symptoms:**
```bash
kubectl get pods -n ray-system
# Shows pods in Pending or CrashLoopBackOff
```

**Solutions:**
```bash
# Check pod details
kubectl describe pod -n ray-system <pod-name>

# Check resource availability
kubectl top nodes

# Check operator logs
kubectl logs -n ray-system deployment/kuberay-operator

# Restart operator
kubectl rollout restart deployment/kuberay-operator -n ray-system
```

### Issue 2: Out of Memory

**Symptoms:**
- Training crashes with OOM errors
- Pods are evicted

**Solutions:**
```bash
# Reduce batch size
BATCH_SIZE=64 ./run-benchmark.sh

# Increase Minikube memory
minikube delete
minikube start --nodes 2 --cpus 4 --memory 16384

# Reduce model size or workers
```

### Issue 3: Slow Performance

**Symptoms:**
- Training is slower than expected
- Workers are idle

**Check:**
```bash
# CPU throttling
kubectl top pods -n ray-system

# Network issues
kubectl exec -it -n ray-system $RAY_HEAD_POD -- ray status

# Check Ray logs
kubectl logs -n ray-system <worker-pod-name>
```

### Issue 4: Connection Issues

**Symptoms:**
- Workers can't connect to head node
- "Ray cluster not found" errors

**Solutions:**
```bash
# Verify service
kubectl get svc -n ray-system

# Check network policies
kubectl get networkpolicies -n ray-system

# Restart Ray cluster
kubectl delete raycluster -n ray-system ray-training-cluster
kubectl apply -f ../k8s/ray-cluster.yaml
```

### Issue 5: Docker/Minikube Issues

**Symptoms:**
- "Cannot connect to Docker daemon"
- Minikube won't start

**Solutions:**
```bash
# Start Docker daemon
sudo systemctl start docker

# Reset Minikube
minikube delete
minikube start --nodes 2 --cpus 4 --memory 8192

# Check Minikube status
minikube status

# View Minikube logs
minikube logs
```

## Part 7: Cleanup

After completing the exercise, clean up resources:

```bash
# Delete Ray cluster and operator
./cleanup.sh

# Stop Minikube (optional)
minikube stop

# Delete Minikube cluster (optional)
minikube delete
```

## Part 8: Advanced Challenges (Optional)

### Challenge 1: Add GPU Support

Modify the configuration to use GPUs if available:

1. Start Minikube with GPU support
2. Update `ray-cluster.yaml` to request GPUs
3. Modify training scripts to use GPU

### Challenge 2: Implement Hyperparameter Tuning

Use Ray Tune to find optimal hyperparameters:

```python
from ray import tune

config = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([64, 128, 256]),
}

tuner = tune.Tuner(
    train_func,
    param_space=config,
    tune_config=tune.TuneConfig(num_samples=10)
)
```

### Challenge 3: Add Model Checkpointing

Implement checkpointing to save model progress:

```python
from ray.train import Checkpoint

train.report(
    metrics={"accuracy": acc},
    checkpoint=Checkpoint.from_dict({"model": model.get_weights()})
)
```

### Challenge 4: Scale to Larger Dataset

Try training on ImageNet or a custom larger dataset:
- Download and prepare the dataset
- Modify data loading code
- Adjust cluster resources

### Challenge 5: Implement Custom Metrics

Add custom training metrics and visualizations:
- Learning rate schedule
- Per-class accuracy
- Training/validation loss curves
- Resource utilization graphs

## Summary

In this exercise, you learned:

1. **Setup**: Deploy Kubernetes cluster and Ray using KubeRay
2. **Code**: Understand baseline vs. distributed training code
3. **Execution**: Run and benchmark both approaches
4. **Analysis**: Measure speedup and efficiency
5. **Optimization**: Experiment with different configurations
6. **Troubleshooting**: Diagnose and fix common issues

## Key Takeaways

- **Ray simplifies distributed training** without major code changes
- **Speedup is not linear** due to communication overhead
- **Efficiency matters** more than raw speedup
- **Kubernetes + Ray** provides scalable ML infrastructure
- **Trade-offs exist** between complexity and performance

## Further Reading

- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [Distributed Training Best Practices](https://docs.ray.io/en/latest/train/user-guides/index.html)
- [Scaling Ray Clusters](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html)
- [TensorFlow Distributed Training](https://www.tensorflow.org/guide/distributed_training)
- [KubeRay Examples](https://github.com/ray-project/kuberay/tree/master/examples)

## Feedback

If you encounter issues or have suggestions, please:
- Check the [main README troubleshooting section](../README.md#troubleshooting)
- Review [Ray documentation](https://docs.ray.io)
- Ask in [Ray Slack community](https://ray-distributed.slack.com)

---

**Congratulations on completing the Ray Parallel Training exercise!** 🎉
