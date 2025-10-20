#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=========================================="
echo "Ray Parallel Training Benchmark"
echo "=========================================="

# Configuration
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
RAY_HEAD_POD=$(kubectl get pods -n ray-system -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

if [ -z "$RAY_HEAD_POD" ]; then
    echo "Error: Could not find Ray head pod"
    echo "Make sure Ray cluster is deployed"
    exit 1
fi

echo "Ray head pod: $RAY_HEAD_POD"
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Create results directory in project root
mkdir -p "$PROJECT_ROOT/results"

# Run baseline training (without Ray)
echo "=========================================="
echo "Step 1: Running Baseline Training (No Ray)"
echo "=========================================="
echo "Copying training scripts to Ray head pod..."
kubectl cp "$PROJECT_ROOT/training/baseline_training.py" ray-system/$RAY_HEAD_POD:/home/ray/baseline_training.py

echo "Running baseline training..."
kubectl exec -it -n ray-system $RAY_HEAD_POD -- python /home/ray/baseline_training.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE

# Copy baseline results
echo "Copying baseline results..."
kubectl cp ray-system/$RAY_HEAD_POD:/home/ray/baseline_results.json "$PROJECT_ROOT/results/baseline_results.json"

echo ""
echo "Baseline training complete!"
echo ""

# Run Ray distributed training with 2 workers
echo "=========================================="
echo "Step 2: Running Ray Distributed Training (2 Workers)"
echo "=========================================="
echo "Copying Ray training script to Ray head pod..."
kubectl cp "$PROJECT_ROOT/training/ray_training.py" ray-system/$RAY_HEAD_POD:/home/ray/ray_training.py

echo "Running Ray distributed training with 2 workers..."
kubectl exec -it -n ray-system $RAY_HEAD_POD -- python /home/ray/ray_training.py \
    --ray-address auto \
    --num-workers 2 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE

# Copy Ray results
echo "Copying Ray training results..."
kubectl cp ray-system/$RAY_HEAD_POD:/home/ray/ray_results_2workers.json "$PROJECT_ROOT/results/ray_results_2workers.json"

echo ""
echo "Ray distributed training complete!"
echo ""

# Generate comparison report
echo "=========================================="
echo "Generating Comparison Report"
echo "=========================================="

python3 << EOF
import json
import os

# Load results
with open('$PROJECT_ROOT/results/baseline_results.json', 'r') as f:
    baseline = json.load(f)

with open('$PROJECT_ROOT/results/ray_results_2workers.json', 'r') as f:
    ray_results = json.load(f)

# Calculate speedup
speedup = baseline['training_time_seconds'] / ray_results['training_time_seconds']
time_saved = baseline['training_time_seconds'] - ray_results['training_time_seconds']

# Print comparison
print("\n" + "="*70)
print(" " * 20 + "BENCHMARK RESULTS COMPARISON")
print("="*70)
print("\nTraining Time:")
print(f"  Baseline (No Ray):        {baseline['training_time_minutes']:.2f} minutes ({baseline['training_time_seconds']:.2f}s)")
print(f"  Ray (2 Workers):          {ray_results['training_time_minutes']:.2f} minutes ({ray_results['training_time_seconds']:.2f}s)")
print(f"  Time Saved:               {time_saved/60:.2f} minutes ({time_saved:.2f}s)")
print(f"  Speedup:                  {speedup:.2f}x")

print("\nModel Performance:")
print(f"  Baseline Test Accuracy:   {baseline['test_accuracy']:.4f}")
print(f"  Ray Test Accuracy:        {ray_results['test_accuracy']:.4f}")
print(f"  Accuracy Difference:      {abs(baseline['test_accuracy'] - ray_results['test_accuracy']):.4f}")

print("\nConfiguration:")
print(f"  Epochs:                   {baseline['epochs_completed']}")
print(f"  Batch Size:               {baseline['batch_size']}")
print(f"  Ray Workers:              {ray_results['num_workers']}")
print(f"  Ray Cluster CPUs:         {ray_results.get('ray_cluster_cpus', 'N/A')}")
print(f"  Ray Cluster Nodes:        {ray_results.get('ray_cluster_nodes', 'N/A')}")

print("\n" + "="*70)
print(f"\nSpeedup with Ray: {speedup:.2f}x faster!")
print(f"Time saved: {time_saved/60:.2f} minutes")
print("="*70 + "\n")

# Save comparison
comparison = {
    'baseline': baseline,
    'ray_distributed': ray_results,
    'comparison': {
        'speedup': speedup,
        'time_saved_seconds': time_saved,
        'time_saved_minutes': time_saved / 60,
        'accuracy_difference': abs(baseline['test_accuracy'] - ray_results['test_accuracy'])
    }
}

with open('$PROJECT_ROOT/results/comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print("Detailed comparison saved to results/comparison.json\n")
EOF

echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results saved in ./results/ directory:"
echo "  - baseline_results.json"
echo "  - ray_results_2workers.json"
echo "  - comparison.json"
echo ""
