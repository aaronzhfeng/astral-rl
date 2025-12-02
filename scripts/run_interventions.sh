#!/bin/bash
# run_interventions.sh
# Runs causal intervention experiments on all trained ASTRAL models

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
mapfile -t MODELS < <(find results/runs -name "final_model.pt" -path "*astral*" 2>/dev/null)

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No trained ASTRAL models found!"
    echo "Run training first: ./scripts/run_core_experiments.sh"
    exit 1
fi

echo "Found ${#MODELS[@]} models:"
for model in "${MODELS[@]}"; do
    dirname "$model"
done
echo ""

# Track failures
FAILURES=()
SUCCESSES=0

# Run interventions on each model
echo "=== Running Intervention Experiments ==="
for model in "${MODELS[@]}"; do
    run_dir=$(dirname "$model")
    run_name=$(basename "$run_dir")
    
    echo ""
    echo "Interventions: $run_name"
    echo "---"
    
    if python src/interventions.py \
        --checkpoint "$model" \
        --num_episodes $NUM_EPISODES; then
        ((SUCCESSES++))
    else
        FAILURES+=("$run_name")
        echo "  [FAILED] $run_name"
    fi
done

echo ""
echo "=============================================="
echo "Intervention Experiments Complete!"
echo "=============================================="
echo "  Successful: $SUCCESSES"
echo "  Failed: ${#FAILURES[@]}"

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo ""
    echo "FAILED MODELS:"
    for fail in "${FAILURES[@]}"; do
        echo "  - $fail"
    done
fi

echo ""
echo "Results saved to: results/interventions/"

# Exit with error if any failures
[ ${#FAILURES[@]} -eq 0 ]
