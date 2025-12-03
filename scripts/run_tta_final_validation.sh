#!/bin/bash
# scripts/run_tta_final_validation.sh
# Final attempt to validate TTA: Test on diverse models vs collapsed models

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "========================================"
echo "TTA Final Validation Experiments"
echo "========================================"
echo ""

OUTPUT_DIR="results/tta_final_validation"
mkdir -p $OUTPUT_DIR

# ============================================
# PART 1: Train models with better diversity
# ============================================

echo "=== Part 1: Training Diverse Models ==="

# 1A: Slot dropout (forces all slots to be useful)
echo "[1/4] Training with slot dropout 0.3..."
python src/train.py \
    --exp_name slot_dropout_0.3 \
    --slot_dropout 0.3 \
    --total_timesteps 200000 \
    --seed 42

echo "[2/4] Training with slot dropout 0.5..."
python src/train.py \
    --exp_name slot_dropout_0.5 \
    --slot_dropout 0.5 \
    --total_timesteps 200000 \
    --seed 42

# 1B: Strong diversity regularization
echo "[3/4] Training with strong combined regularization..."
python src/train.py \
    --exp_name diverse_strong \
    --lambda_contrast 0.15 \
    --lambda_lb 0.08 \
    --lambda_w_ent 0.08 \
    --lambda_orth 0.02 \
    --total_timesteps 200000 \
    --seed 42

# 1C: Best config with more training
echo "[4/4] Training best_config longer..."
python src/train.py \
    --exp_name best_config_long \
    --use_gumbel True \
    --temp_anneal True \
    --lambda_contrast 0.1 \
    --slot_prediction True \
    --total_timesteps 300000 \
    --seed 42

echo ""
echo "=== Part 2: TTA on Diverse Models ==="

# ============================================
# PART 2: Test TTA on all models
# ============================================

# Function to run TTA and save results
run_tta() {
    local model_name=$1
    local checkpoint=$2
    local adapt_mode=$3
    local agent_type=$4
    
    echo "Testing TTA: $model_name ($adapt_mode mode)"
    python src/test_time_adapt.py \
        --checkpoint "$checkpoint" \
        --adapt_mode $adapt_mode \
        --agent_type $agent_type \
        --num_adapt_episodes 30 \
        --num_eval_episodes 20 \
        --save_dir "$OUTPUT_DIR/${model_name}_${adapt_mode}"
}

# 2A: Test slot dropout models
for dropout in 0.3 0.5; do
    CKPT=$(ls -d results/runs/slot_dropout_${dropout}_astral_42_* 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        run_tta "slot_dropout_${dropout}" "$CKPT/final_model.pt" "gating" "astral"
    fi
done

# 2B: Test diverse_strong model
CKPT=$(ls -d results/runs/diverse_strong_astral_42_* 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    run_tta "diverse_strong" "$CKPT/final_model.pt" "gating" "astral"
fi

# 2C: Test best_config_long model
CKPT=$(ls -d results/runs/best_config_long_astral_42_* 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    run_tta "best_config_long" "$CKPT/final_model.pt" "gating" "astral"
fi

# 2D: Test existing best_config_strong models (cross-seed)
for seed in 42 123 456; do
    CKPT=$(ls -d results/runs/best_config_strong_astral_${seed}_* 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        run_tta "best_config_strong_seed${seed}" "$CKPT/final_model.pt" "gating" "astral"
    fi
done

# 2E: Compare with collapsed default model
CKPT=$(ls -d results/runs/astral_astral_42_* 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    run_tta "collapsed_default" "$CKPT/final_model.pt" "gating" "astral"
fi

# 2F: Baseline comparison (policy head fine-tuning)
CKPT=$(ls -d results/runs/baseline_baseline_42_* 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    run_tta "baseline" "$CKPT/final_model.pt" "policy_head" "baseline"
fi

echo ""
echo "=== Part 3: Analysis ==="

# ============================================
# PART 3: Analyze and compare results
# ============================================

python << 'EOF'
import json
import glob
import os

output_dir = "results/tta_final_validation"
results = []

# Load all TTA results
for result_dir in glob.glob(f"{output_dir}/*_gating") + glob.glob(f"{output_dir}/*_policy_head"):
    json_file = os.path.join(result_dir, "tta_results.json")
    if os.path.exists(json_file):
        with open(json_file) as f:
            data = json.load(f)
        
        model_name = os.path.basename(result_dir).replace("_gating", "").replace("_policy_head", "")
        mode = "gating" if "_gating" in result_dir else "policy_head"
        
        # Calculate improvements
        before = data.get("before_tta", {})
        after = data.get("after_tta", {})
        
        if before and after:
            before_mean = sum(before.values()) / len(before) if before else 0
            after_mean = sum(after.values()) / len(after) if after else 0
            improvement = after_mean - before_mean
            pct_improvement = (improvement / before_mean * 100) if before_mean > 0 else 0
            
            results.append({
                "model": model_name,
                "mode": mode,
                "before": before_mean,
                "after": after_mean,
                "improvement": improvement,
                "pct_improvement": pct_improvement
            })

# Sort by improvement
results.sort(key=lambda x: x["improvement"], reverse=True)

# Print summary
print("=" * 80)
print("TTA VALIDATION RESULTS SUMMARY")
print("=" * 80)
print(f"{'Model':<30} {'Mode':<12} {'Before':>8} {'After':>8} {'Δ':>8} {'%Δ':>8}")
print("-" * 80)

for r in results:
    print(f"{r['model']:<30} {r['mode']:<12} {r['before']:>8.1f} {r['after']:>8.1f} {r['improvement']:>+8.1f} {r['pct_improvement']:>+7.1f}%")

print("-" * 80)

# Key findings
diverse_models = [r for r in results if "slot_dropout" in r["model"] or "diverse" in r["model"] or "best_config" in r["model"]]
collapsed_models = [r for r in results if "collapsed" in r["model"]]
baseline_models = [r for r in results if "baseline" in r["model"]]

if diverse_models:
    avg_diverse = sum(r["improvement"] for r in diverse_models) / len(diverse_models)
    print(f"\nDiverse models avg improvement: {avg_diverse:+.1f}")
    
if collapsed_models:
    avg_collapsed = sum(r["improvement"] for r in collapsed_models) / len(collapsed_models)
    print(f"Collapsed models avg improvement: {avg_collapsed:+.1f}")

if baseline_models:
    avg_baseline = sum(r["improvement"] for r in baseline_models) / len(baseline_models)
    print(f"Baseline avg improvement: {avg_baseline:+.1f}")

# Save summary
with open(f"{output_dir}/summary.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to {output_dir}/summary.json")
EOF

echo ""
echo "========================================"
echo "TTA Final Validation Complete!"
echo "========================================"
echo "Results in: $OUTPUT_DIR"

