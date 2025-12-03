#!/bin/bash
# scripts/run_tta_comparison.sh
# Compare ASTRAL TTA (gating) vs Baseline fine-tuning (policy_head)
# This is the critical experiment to prove ASTRAL TTA provides value

set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "========================================"
echo "TTA Comparison: ASTRAL vs Baseline"
echo "========================================"

# Configuration
NUM_ADAPT=30
NUM_EVAL=20

# Find checkpoints
ASTRAL_CKPT=$(ls -d results/runs/best_config_strong_astral_42_* 2>/dev/null | head -1)
BASELINE_CKPT=$(ls -d results/runs/baseline_baseline_42_* 2>/dev/null | head -1)

if [ -z "$ASTRAL_CKPT" ]; then
    echo "ERROR: No best_config_strong_astral_42 checkpoint found!"
    echo "Please run priority experiments first."
    exit 1
fi

if [ -z "$BASELINE_CKPT" ]; then
    echo "ERROR: No baseline_baseline_42 checkpoint found!"
    echo "Please run core experiments first."
    exit 1
fi

echo ""
echo "Using checkpoints:"
echo "  ASTRAL:   $ASTRAL_CKPT"
echo "  Baseline: $BASELINE_CKPT"
echo ""

# Create output directories
mkdir -p results/tta_comparison/astral_gating
mkdir -p results/tta_comparison/baseline_policy
mkdir -p results/tta_comparison/astral_policy

# 1. ASTRAL with gating-only TTA (the ASTRAL approach)
echo "========================================"
echo "[1/3] ASTRAL with Gating TTA"
echo "========================================"
python src/test_time_adapt.py \
    --checkpoint "$ASTRAL_CKPT/final_model.pt" \
    --adapt_mode gating \
    --num_adapt_episodes $NUM_ADAPT \
    --num_eval_episodes $NUM_EVAL \
    --save_dir results/tta_comparison/astral_gating

# 2. Baseline with policy-head TTA (the control)
echo ""
echo "========================================"
echo "[2/3] Baseline with Policy-Head TTA"
echo "========================================"
python src/test_time_adapt.py \
    --checkpoint "$BASELINE_CKPT/final_model.pt" \
    --adapt_mode policy_head \
    --num_adapt_episodes $NUM_ADAPT \
    --num_eval_episodes $NUM_EVAL \
    --save_dir results/tta_comparison/baseline_policy

# 3. ASTRAL with policy-head TTA (ablation)
echo ""
echo "========================================"
echo "[3/3] ASTRAL with Policy-Head TTA (ablation)"
echo "========================================"
python src/test_time_adapt.py \
    --checkpoint "$ASTRAL_CKPT/final_model.pt" \
    --adapt_mode policy_head \
    --num_adapt_episodes $NUM_ADAPT \
    --num_eval_episodes $NUM_EVAL \
    --save_dir results/tta_comparison/astral_policy

echo ""
echo "========================================"
echo "TTA Comparison Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  results/tta_comparison/astral_gating/  (ASTRAL TTA)"
echo "  results/tta_comparison/baseline_policy/ (Baseline control)"
echo "  results/tta_comparison/astral_policy/  (ASTRAL ablation)"
echo ""
echo "To analyze results, compare improvements across conditions."

