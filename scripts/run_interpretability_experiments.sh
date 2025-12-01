#!/bin/bash
# run_interpretability_experiments.sh
# Runs all interpretability improvement experiments

set -e  # Exit on error

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

# 1. Individual improvements
echo "=== Individual Improvements ==="

echo "[1/7] Gumbel-Softmax only..."
python src/train.py \
    --exp_name interp_gumbel \
    --use_gumbel True \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

echo "[2/7] Hard routing only..."
python src/train.py \
    --exp_name interp_hard \
    --hard_routing True \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

echo "[3/7] Orthogonal init only..."
python src/train.py \
    --exp_name interp_orth \
    --orthogonal_init True \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

echo "[4/7] Temperature annealing only..."
python src/train.py \
    --exp_name interp_temp_anneal \
    --temp_anneal True \
    --tau_start 5.0 \
    --tau_end 0.5 \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

echo "[5/7] Contrastive loss only (λ=0.05)..."
python src/train.py \
    --exp_name interp_contrast \
    --lambda_contrast 0.05 \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

echo "[6/7] Slot prediction only..."
python src/train.py \
    --exp_name interp_slot_pred \
    --slot_prediction True \
    --lambda_slot_pred 0.01 \
    --total_timesteps $TIMESTEPS \
    --seed $SEED

# 2. All combined
echo ""
echo "=== All Improvements Combined ==="

echo "[7/7] All improvements..."
python src/train.py \
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
echo "View results: tensorboard --logdir results/runs"
echo "=============================================="

