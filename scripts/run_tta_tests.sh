#!/bin/bash
# run_tta_tests.sh
# Runs test-time adaptation experiments on all trained ASTRAL models

set -e  # Exit on error

cd "$(dirname "$0")/.."

# Activate venv (adjust path if needed)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "ASTRAL Test-Time Adaptation Tests"
echo "=============================================="

NUM_ADAPT=${1:-20}
NUM_EVAL=${2:-15}

echo "Adaptation episodes: $NUM_ADAPT"
echo "Evaluation episodes: $NUM_EVAL"
echo ""

# Find all ASTRAL runs with final_model.pt
RUNS=$(find results/runs -name "final_model.pt" -path "*astral*" 2>/dev/null || true)

if [ -z "$RUNS" ]; then
    echo "No trained ASTRAL models found!"
    echo "Run training first: ./scripts/run_core_experiments.sh"
    exit 1
fi

echo "Found models:"
echo "$RUNS" | while read -r model; do
    dirname "$model"
done
echo ""

# Run TTA on each model
echo "=== Running TTA Tests ==="
echo "$RUNS" | while read -r model; do
    run_dir=$(dirname "$model")
    run_name=$(basename "$run_dir")
    
    echo ""
    echo "Testing: $run_name"
    echo "---"
    
    python src/test_time_adapt.py \
        --checkpoint "$model" \
        --num_adapt_episodes $NUM_ADAPT \
        --num_eval_episodes $NUM_EVAL || echo "  [FAILED] $run_name"
done

echo ""
echo "=============================================="
echo "TTA Tests Complete!"
echo "Results saved to: results/tta/"
echo "=============================================="

