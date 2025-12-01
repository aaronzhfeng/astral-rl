#!/bin/bash
# run_interventions.sh
# Runs causal intervention experiments on all trained ASTRAL models

set -e  # Exit on error

cd "$(dirname "$0")/.."

# Activate venv (adjust path if needed)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "ASTRAL Causal Intervention Experiments"
echo "=============================================="

NUM_EPISODES=${1:-20}

echo "Episodes per experiment: $NUM_EPISODES"
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

# Run interventions on each model
echo "=== Running Intervention Experiments ==="
echo "$RUNS" | while read -r model; do
    run_dir=$(dirname "$model")
    run_name=$(basename "$run_dir")
    
    echo ""
    echo "Interventions: $run_name"
    echo "---"
    
    python src/interventions.py \
        --checkpoint "$model" \
        --num_episodes $NUM_EPISODES || echo "  [FAILED] $run_name"
done

echo ""
echo "=============================================="
echo "Intervention Experiments Complete!"
echo "Results saved to: results/interventions/"
echo "=============================================="

