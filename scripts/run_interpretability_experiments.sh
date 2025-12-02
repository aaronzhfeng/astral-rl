#!/bin/bash
# run_interpretability_experiments.sh
# Runs all interpretability improvement experiments

cd "$(dirname "$0")/.."

# Activate venv (adjust path if needed)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "ASTRAL Interpretability Experiments"
echo "=============================================="

TIMESTEPS=${1:-200000}
SEED=42

echo "Timesteps: $TIMESTEPS"
echo "Seed: $SEED"
echo ""

# Track failures
FAILURES=()
SUCCESSES=0

# Helper function to run experiment
run_experiment() {
    local name=$1
    shift
    echo "$name"
    if python src/train.py "$@"; then
        ((SUCCESSES++))
    else
        FAILURES+=("$name")
        echo "  [FAILED] $name"
    fi
}

# 1. Individual improvements
echo "=== Individual Improvements ==="

run_experiment "[1/7] Gumbel-Softmax only" \
    --exp_name interp_gumbel \
    --use_gumbel True \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

run_experiment "[2/7] Hard routing only" \
    --exp_name interp_hard \
    --hard_routing True \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

run_experiment "[3/7] Orthogonal init only" \
    --exp_name interp_orth \
    --orthogonal_init True \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

run_experiment "[4/7] Temperature annealing only" \
    --exp_name interp_temp_anneal \
    --temp_anneal True \
    --tau_start 5.0 \
    --tau_end 0.5 \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

run_experiment "[5/7] Contrastive loss only (λ=0.05)" \
    --exp_name interp_contrast \
    --lambda_contrast 0.05 \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

run_experiment "[6/7] Slot prediction only" \
    --exp_name interp_slot_pred \
    --slot_prediction True \
    --lambda_slot_pred 0.01 \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

# 2. All combined
echo ""
echo "=== All Improvements Combined ==="

run_experiment "[7/7] All improvements" \
    --exp_name interp_all \
    --use_gumbel True \
    --hard_routing True \
    --orthogonal_init True \
    --temp_anneal True \
    --tau_start 5.0 \
    --tau_end 0.5 \
    --lambda_contrast 0.01 \
    --slot_prediction True \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

echo ""
echo "=============================================="
echo "Interpretability Experiments Complete!"
echo "=============================================="
echo "  Successful: $SUCCESSES"
echo "  Failed: ${#FAILURES[@]}"

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo ""
    echo "FAILED RUNS:"
    for fail in "${FAILURES[@]}"; do
        echo "  - $fail"
    done
fi

echo ""
echo "View results: tensorboard --logdir results/runs"

# Exit with error if any failures
[ ${#FAILURES[@]} -eq 0 ]
