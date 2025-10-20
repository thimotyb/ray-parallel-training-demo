#!/bin/bash
set -e

RUN_TARGET_RAW=${1:-both}

if [[ "$RUN_TARGET_RAW" == "-h" || "$RUN_TARGET_RAW" == "--help" ]]; then
    cat <<'USAGE'
Usage: ./scripts/run-benchmark.sh [baseline|ray|both]

Run the MNIST training benchmark:
  baseline - run the baseline (no Ray) training only
  ray      - run the Ray distributed training only
  both     - run both baseline and Ray training (default)

Environment variables:
  EPOCHS      Number of training epochs (default: 10)
  BATCH_SIZE  Training batch size (default: 128)
USAGE
    exit 0
fi

RUN_TARGET=$(echo "$RUN_TARGET_RAW" | tr '[:upper:]' '[:lower:]')
case "$RUN_TARGET" in
    baseline|ray|both) ;;
    *)
        echo "Error: Invalid run target '$RUN_TARGET_RAW'."
        echo "Valid options are: baseline, ray, both."
        exit 1
        ;;
esac

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=========================================="
echo "Ray Parallel Training Benchmark"
echo "=========================================="
echo "Selected run target: $RUN_TARGET"
echo ""

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

# Track what we run during this invocation
baseline_run=false
ray_run=false

if [[ "$RUN_TARGET" == "baseline" || "$RUN_TARGET" == "both" ]]; then
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
    baseline_run=true
fi

if [[ "$RUN_TARGET" == "ray" || "$RUN_TARGET" == "both" ]]; then
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
    ray_run=true
fi

if [[ "$baseline_run" == true && "$ray_run" == true ]]; then
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

    comparison_generated=true
else
    echo "=========================================="
    echo "Skipping Comparison Report"
    echo "=========================================="
    echo "Comparison requires both baseline and Ray results from the current run."
    echo "Run with 'both' to generate the side-by-side report."
    echo ""
    comparison_generated=false
fi

echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Outputs generated:"
if [[ "$baseline_run" == true ]]; then
    echo "  - results/baseline_results.json"
fi
if [[ "$ray_run" == true ]]; then
    echo "  - results/ray_results_2workers.json"
fi
if [[ "${comparison_generated:-false}" == true ]]; then
    echo "  - results/comparison.json"
fi
echo ""
