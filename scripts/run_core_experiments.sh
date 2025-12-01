#!/bin/bash
# run_core_experiments.sh
# Runs core ASTRAL vs Baseline experiments with multiple seeds

set -e  # Exit on error

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

# 1. ASTRAL experiments
echo "=== Training ASTRAL ==="
for seed in $SEEDS; do
    echo "[ASTRAL] Seed=$seed, Timesteps=$TIMESTEPS"
    python src/train.py \
        --exp_name astral \
        --use_abstractions True \
        --total_timesteps $TIMESTEPS \
        --seed $seed
done

# 2. Baseline experiments
echo ""
echo "=== Training Baseline ==="
for seed in $SEEDS; do
    echo "[Baseline] Seed=$seed, Timesteps=$TIMESTEPS"
    python src/train.py \
        --exp_name baseline \
        --use_abstractions False \
        --total_timesteps $TIMESTEPS \
        --seed $seed
done

echo ""
echo "=============================================="
echo "Core Experiments Complete!"
echo "View results: tensorboard --logdir results/runs"
echo "=============================================="

