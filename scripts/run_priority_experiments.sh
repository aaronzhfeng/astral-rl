#!/bin/bash
# run_priority_experiments.sh
# Priority experiments to address slot collapse (no code changes needed)

cd "$(dirname "$0")/.."

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "Priority Experiments: Addressing Slot Collapse"
echo "=============================================="
echo ""
echo "These experiments test stronger regularization"
echo "to prevent slot collapse without code changes."
echo ""
echo "Estimated time: ~2 hours"
echo ""

TIMESTEPS=${1:-300000}
echo "Timesteps per run: $TIMESTEPS"
echo ""

# Track results
FAILURES=()
SUCCESSES=0

run_experiment() {
    local name=$1
    shift
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if python src/train.py "$@"; then
        ((SUCCESSES++))
        echo "✓ $name completed"
    else
        FAILURES+=("$name")
        echo "✗ $name FAILED"
    fi
}

echo ""
echo "=============================================="
echo "[1/4] Strong Contrastive Loss Sweep"
echo "=============================================="

run_experiment "strong_contrast_0.1" \
    --exp_name strong_contrast_0.1 \
    --lambda_contrast 0.1 \
    --total_timesteps $TIMESTEPS \
    --seed 42

run_experiment "strong_contrast_0.2" \
    --exp_name strong_contrast_0.2 \
    --lambda_contrast 0.2 \
    --total_timesteps $TIMESTEPS \
    --seed 42

run_experiment "strong_contrast_0.5" \
    --exp_name strong_contrast_0.5 \
    --lambda_contrast 0.5 \
    --total_timesteps $TIMESTEPS \
    --seed 42

echo ""
echo "=============================================="
echo "[2/4] Strong Load Balancing"
echo "=============================================="

run_experiment "strong_lb_0.05" \
    --exp_name strong_lb_0.05 \
    --lambda_lb 0.05 \
    --total_timesteps $TIMESTEPS \
    --seed 42

run_experiment "strong_lb_0.1" \
    --exp_name strong_lb_0.1 \
    --lambda_lb 0.1 \
    --total_timesteps $TIMESTEPS \
    --seed 42

echo ""
echo "=============================================="
echo "[3/4] Strong Weight Entropy"
echo "=============================================="

run_experiment "strong_w_ent_0.05" \
    --exp_name strong_w_ent_0.05 \
    --lambda_w_ent 0.05 \
    --total_timesteps $TIMESTEPS \
    --seed 42

run_experiment "strong_w_ent_0.1" \
    --exp_name strong_w_ent_0.1 \
    --lambda_w_ent 0.1 \
    --total_timesteps $TIMESTEPS \
    --seed 42

echo ""
echo "=============================================="
echo "[4/4] Combined Strong Regularization"
echo "=============================================="

run_experiment "strong_all_reg" \
    --exp_name strong_all_reg \
    --lambda_contrast 0.1 \
    --lambda_lb 0.05 \
    --lambda_w_ent 0.05 \
    --lambda_orth 0.01 \
    --total_timesteps $TIMESTEPS \
    --seed 42

# Best config with all improvements + strong regularization
run_experiment "best_config_strong" \
    --exp_name best_config_strong \
    --use_gumbel True \
    --temp_anneal True \
    --tau_start 5.0 \
    --tau_end 0.5 \
    --lambda_contrast 0.1 \
    --lambda_lb 0.05 \
    --slot_prediction True \
    --total_timesteps $TIMESTEPS \
    --seed 42

# Multiple seeds for best config
for seed in 123 456; do
    run_experiment "best_config_strong_seed${seed}" \
        --exp_name best_config_strong \
        --use_gumbel True \
        --temp_anneal True \
        --tau_start 5.0 \
        --tau_end 0.5 \
        --lambda_contrast 0.1 \
        --lambda_lb 0.05 \
        --slot_prediction True \
        --total_timesteps $TIMESTEPS \
        --seed $seed
done

echo ""
echo "=============================================="
echo "Priority Experiments Complete!"
echo "=============================================="
echo "  Successful: $SUCCESSES"
echo "  Failed: ${#FAILURES[@]}"

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo ""
    echo "FAILED EXPERIMENTS:"
    for fail in "${FAILURES[@]}"; do
        echo "  - $fail"
    done
fi

echo ""
echo "Next steps:"
echo "  1. View training curves: tensorboard --logdir results/runs"
echo "  2. Run TTA tests: ./scripts/run_tta_tests.sh"
echo "  3. Run interventions: ./scripts/run_interventions.sh"
echo ""
echo "Look for models with:"
echo "  - Weight entropy > 0.8"
echo "  - All slots receiving >10% weight"
echo "  - Good TTA improvement"
echo "=============================================="

[ ${#FAILURES[@]} -eq 0 ]

