#!/bin/bash
# run_core_experiments.sh
# Runs core ASTRAL vs Baseline experiments with multiple seeds

cd "$(dirname "$0")/.."

# Activate venv (adjust path if needed)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "ASTRAL Core Experiments"
echo "=============================================="

TIMESTEPS=${1:-500000}
SEEDS="42 123 456"

echo "Timesteps: $TIMESTEPS"
echo "Seeds: $SEEDS"
echo ""

# Track failures
FAILURES=()
SUCCESSES=0

# 1. ASTRAL experiments
echo "=== Training ASTRAL ==="
for seed in $SEEDS; do
    echo "[ASTRAL] Seed=$seed, Timesteps=$TIMESTEPS"
    if python src/train.py \
        --exp_name astral \
        --use_abstractions True \
        --total_timesteps $TIMESTEPS \
        --seed $seed; then
        ((SUCCESSES++))
    else
        FAILURES+=("ASTRAL seed=$seed")
        echo "  [FAILED] ASTRAL seed=$seed"
    fi
done

# 2. Baseline experiments
echo ""
echo "=== Training Baseline ==="
for seed in $SEEDS; do
    echo "[Baseline] Seed=$seed, Timesteps=$TIMESTEPS"
    if python src/train.py \
        --exp_name baseline \
        --use_abstractions False \
        --total_timesteps $TIMESTEPS \
        --seed $seed; then
        ((SUCCESSES++))
    else
        FAILURES+=("Baseline seed=$seed")
        echo "  [FAILED] Baseline seed=$seed"
    fi
done

echo ""
echo "=============================================="
echo "Core Experiments Complete!"
echo "=============================================="
echo "  Successful: $SUCCESSES"
echo "  Failed: ${#FAILURES[@]}"

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo ""
    echo "FAILED RUNS:"
    for fail in "${FAILURES[@]}"; do
        echo "  - $fail"
    done
    echo ""
    echo "View results: tensorboard --logdir results/runs"
    exit 1
else
    echo ""
    echo "View results: tensorboard --logdir results/runs"
    exit 0
fi
