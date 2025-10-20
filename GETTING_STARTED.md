# Getting Started Checklist

Follow this checklist to successfully run the Ray parallel training demo.

## Pre-Flight Checklist

### ✓ System Requirements

- [ ] CPU: 4+ cores available
- [ ] RAM: 8GB+ available
- [ ] Disk: 20GB+ free space
- [ ] OS: Linux, macOS, or WSL2 on Windows

### ✓ Required Software

**Option A: Automated Install (Recommended)**

- [ ] Run the prerequisites installation script
  ```bash
  ./scripts/install-prerequisites.sh
  ```
  This will install kubectl, Minikube, and Helm automatically.

**Option B: Manual Install**

- [ ] Docker installed and running
  ```bash
  docker --version
  docker ps  # Should not show error
  ```

- [ ] Minikube installed
  ```bash
  minikube version
  ```

- [ ] kubectl installed
  ```bash
  kubectl version --client
  ```

- [ ] Helm installed
  ```bash
  helm version
  ```

- [ ] Python 3.10+ (optional, for local testing)
  ```bash
  python3 --version
  ```

## Setup Checklist

### Step 1: Clone/Navigate to Project

- [ ] Navigate to project directory
  ```bash
  cd ray-parallel-training-demo
  ```

- [ ] Verify project structure
  ```bash
  ls -la
  # Should see: scripts/, k8s/, training/, docker/, docs/
  ```

### Step 2: Start Minikube

- [ ] Run setup script
  ```bash
  ./scripts/setup-minikube.sh
  ```

- [ ] Verify 2 nodes are running
  ```bash
  kubectl get nodes
  # Expected: 2 nodes in Ready state
  ```

- [ ] Check resource availability
  ```bash
  kubectl top nodes
  ```

**Time estimate:** 3-5 minutes

### Step 3: Install Ray

- [ ] Run installation script
  ```bash
  ./scripts/install-ray.sh
  ```

- [ ] Verify KubeRay operator is running
  ```bash
  kubectl get pods -n ray-system | grep operator
  # Expected: 1/1 Running
  ```

- [ ] Verify Ray cluster is deployed
  ```bash
  kubectl get raycluster -n ray-system
  # Expected: ray-training-cluster READY
  ```

- [ ] Check all pods are running
  ```bash
  kubectl get pods -n ray-system
  # Expected: 1 head + 2 workers, all Running
  ```

**Time estimate:** 3-5 minutes

### Step 4: Access Ray Dashboard (Optional)

- [ ] Port forward to dashboard
  ```bash
  kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265
  ```

- [ ] Open browser to http://localhost:8265

- [ ] Verify you can see the cluster info

**Time estimate:** 1 minute

## Running Benchmark Checklist

### Step 5: Run Full Benchmark

- [ ] Execute benchmark script
  ```bash
  ./scripts/run-benchmark.sh
  ```

- [ ] Monitor progress (this will take 15-25 minutes)
  - [ ] Baseline training completes (~10-15 min)
  - [ ] Ray distributed training completes (~6-8 min)
  - [ ] Comparison report is generated

- [ ] Review results
  ```bash
  cat results/comparison.json
  ```

**Time estimate:** 15-25 minutes

### Alternative: Quick Test (5 epochs)

- [ ] Run quick benchmark for faster testing
  ```bash
  EPOCHS=5 ./scripts/run-benchmark.sh
  ```

**Time estimate:** 8-12 minutes

## Verification Checklist

### ✓ Results Generated

- [ ] `results/baseline_results.json` exists
- [ ] `results/ray_results_2workers.json` exists
- [ ] `results/comparison.json` exists

### ✓ Performance Metrics

- [ ] Speedup is greater than 1.0x
  ```bash
  cat results/comparison.json | grep speedup
  ```

- [ ] Both training runs completed successfully
- [ ] Test accuracy is reasonable (> 0.65)

### ✓ Ray Cluster Health

- [ ] All Ray pods still running
  ```bash
  kubectl get pods -n ray-system
  ```

- [ ] No errors in Ray operator logs
  ```bash
  kubectl logs -n ray-system deployment/kuberay-operator | tail -20
  ```

## Troubleshooting Checklist

If something goes wrong, check:

### Minikube Issues

- [ ] Minikube is running
  ```bash
  minikube status
  ```

- [ ] Docker daemon is running
  ```bash
  docker ps
  ```

- [ ] Sufficient resources allocated
  ```bash
  minikube config get memory  # Should be >= 8192
  minikube config get cpus    # Should be >= 4
  ```

### Ray Cluster Issues

- [ ] All pods are in Running state
  ```bash
  kubectl get pods -n ray-system
  ```

- [ ] Check pod logs for errors
  ```bash
  kubectl logs -n ray-system <pod-name>
  ```

- [ ] Verify Ray cluster is accessible
  ```bash
  RAY_HEAD_POD=$(kubectl get pods -n ray-system -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
  kubectl exec -it -n ray-system $RAY_HEAD_POD -- ray status
  ```

### Training Issues

- [ ] Scripts were copied successfully
- [ ] Python dependencies are available
  ```bash
  kubectl exec -it -n ray-system $RAY_HEAD_POD -- pip list | grep tensorflow
  ```

- [ ] Check training logs
  ```bash
  kubectl logs -n ray-system $RAY_HEAD_POD
  ```

## Cleanup Checklist

### Step 6: Clean Up Resources

- [ ] Run cleanup script
  ```bash
  ./scripts/cleanup.sh
  ```

- [ ] Verify Ray resources are deleted
  ```bash
  kubectl get all -n ray-system
  # Should show minimal resources or namespace not found
  ```

- [ ] (Optional) Stop Minikube
  ```bash
  minikube stop
  ```

- [ ] (Optional) Delete Minikube cluster
  ```bash
  minikube delete
  ```

**Time estimate:** 2-3 minutes

## Next Steps Checklist

After successful completion:

- [ ] Read the [Exercise Guide](docs/EXERCISE.md) for deeper understanding
- [ ] Try the experiments in Part 4 of the exercise
- [ ] Experiment with different worker counts
- [ ] Modify the model architecture
- [ ] Try a different dataset
- [ ] Review [Ray Train documentation](https://docs.ray.io/en/latest/train/train.html)

## Common Gotchas

### Watch Out For:

- **Resource constraints**: Ensure Docker has enough resources allocated
- **Port conflicts**: Port 8265 may be in use (for Ray dashboard)
- **Image pull delays**: First run may take longer to download images
- **Python package versions**: Use exact versions from requirements.txt
- **File permissions**: Ensure scripts are executable (`chmod +x scripts/*.sh`)

## Estimated Total Time

| Task | Time |
|------|------|
| Prerequisites check | 5 min |
| Minikube setup | 5 min |
| Ray installation | 5 min |
| Full benchmark | 20 min |
| Review results | 5 min |
| **Total** | **~40 minutes** |

Quick test (5 epochs): **~25 minutes total**

## Success Criteria

You've successfully completed the demo if:

1. ✓ Both training runs complete without errors
2. ✓ Ray training is faster than baseline (speedup > 1.0)
3. ✓ Model accuracy is similar between both approaches (within ~2%)
4. ✓ Results files are generated in `results/` directory
5. ✓ You understand the speedup benefits of distributed training

## Help & Support

If you're stuck:

1. Check the [Troubleshooting section](README.md#troubleshooting) in README
2. Review the [Exercise Guide](docs/EXERCISE.md) Part 6
3. Consult the [Quick Reference](docs/QUICK_REFERENCE.md)
4. Check [Ray documentation](https://docs.ray.io)
5. Ask in [Ray Slack community](https://ray-distributed.slack.com)

## Quick Commands Reference

```bash
# Start everything
./scripts/setup-minikube.sh
./scripts/install-ray.sh

# Run benchmark
./scripts/run-benchmark.sh

# Check status
kubectl get pods -n ray-system
kubectl get nodes

# View results
cat results/comparison.json

# Clean up
./scripts/cleanup.sh
```

---

**Ready to start? Begin with Step 1!** 🚀

**Pro tip:** Open this checklist in a separate window and check off items as you go!
