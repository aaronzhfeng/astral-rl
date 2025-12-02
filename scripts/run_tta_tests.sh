#!/bin/bash
# run_tta_tests.sh
# Runs test-time adaptation experiments on all trained ASTRAL models

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

# Run TTA on each model
echo "=== Running TTA Tests ==="
for model in "${MODELS[@]}"; do
    run_dir=$(dirname "$model")
    run_name=$(basename "$run_dir")
    
    echo ""
    echo "Testing: $run_name"
    echo "---"
    
    if python src/test_time_adapt.py \
        --checkpoint "$model" \
        --num_adapt_episodes $NUM_ADAPT \
        --num_eval_episodes $NUM_EVAL; then
        ((SUCCESSES++))
    else
        FAILURES+=("$run_name")
        echo "  [FAILED] $run_name"
    fi
done

echo ""
echo "=============================================="
echo "TTA Tests Complete!"
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
echo "Results saved to: results/tta/"

# Exit with error if any failures
[ ${#FAILURES[@]} -eq 0 ]
