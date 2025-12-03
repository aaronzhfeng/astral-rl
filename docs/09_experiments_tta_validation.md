# Experiments: Validating Test-Time Abstraction Learning

**Purpose:** Prove that TTA improvement is due to abstraction reweighting, not just generic fine-tuning.

**Status:** Experiment 1 implemented, ready to run

---

## Background

From our experiments, we observed:
- TTA improves performance when slots are diverse (+35% avg, up to +70%)
- TTA fails when slots are collapsed (nothing to reweight)
- `best_config_strong` achieves diverse slots [0.56, 0.41, 0.02] + high performance (~450)

**However, we haven't proven:**
1. Is TTA better than fine-tuning a baseline (no abstractions)?
2. Do weights actually shift toward the "correct" slot?
3. Is the improvement statistically significant?

---

## Experiment 1: Baseline Control Comparison (Critical)

**Status: ✅ IMPLEMENTED**

### Hypothesis
If fine-tuning a baseline model (no abstraction bank) achieves similar improvement to ASTRAL TTA, then the abstraction mechanism isn't providing value.

### Implementation (DONE)

Added to `test_time_adapt.py`:
- `--adapt_mode` flag: `gating` (default), `policy_head`, `all_params`
- `--agent_type` flag: `auto` (default), `astral`, `baseline`
- `load_agent()`: Auto-detects agent type from checkpoint path
- `test_time_adapt_policy_head()`: Adapts only policy head
- `test_time_adapt_all_params()`: Full fine-tuning

**Run with:**
```bash
./scripts/run_tta_comparison.sh
```

### Experiments

```bash
# 1A: ASTRAL with gating-only TTA (current approach)
python src/test_time_adapt.py \
    --checkpoint results/runs/best_config_strong_astral_42_*/final_model.pt \
    --adapt_mode gating \
    --num_adapt_episodes 30 \
    --num_eval_episodes 20

# 1B: Baseline with policy-head TTA (control)
python src/test_time_adapt.py \
    --checkpoint results/runs/baseline_baseline_42_*/final_model.pt \
    --adapt_mode policy_head \
    --num_adapt_episodes 30 \
    --num_eval_episodes 20

# 1C: ASTRAL with full fine-tuning (ablation)
python src/test_time_adapt.py \
    --checkpoint results/runs/best_config_strong_astral_42_*/final_model.pt \
    --adapt_mode all_params \
    --num_adapt_episodes 30 \
    --num_eval_episodes 20
```

### Success Criteria

| Comparison | Expected Result | Interpretation |
|------------|-----------------|----------------|
| ASTRAL TTA > Baseline TTA | +15% or more | Abstractions provide value |
| ASTRAL gating ≈ ASTRAL full | Within 5% | Gating is sufficient |
| ASTRAL TTA > No TTA | +20% or more | TTA works |

### Analysis

```python
# Compare improvements
astral_improvement = astral_after - astral_before
baseline_improvement = baseline_after - baseline_before
advantage = astral_improvement - baseline_improvement

# Statistical test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(astral_returns, baseline_returns)
print(f"ASTRAL advantage: {advantage:.1f}, p-value: {p_value:.4f}")
```

---

## Experiment 2: Weight Trajectory Analysis

### Hypothesis
During TTA, gating weights should shift toward the slot that's most appropriate for the current mode. If weights don't change meaningfully, TTA isn't learning abstractions.

### Required Code Changes

Modify `test_time_adapt.py` to:
1. Save weight snapshots at each episode
2. Plot weight evolution over adaptation
3. Calculate weight shift metrics

### Experiments

```bash
# Run TTA with trajectory tracking
for mode in 0 1 2; do
    python src/test_time_adapt.py \
        --checkpoint results/runs/best_config_strong_astral_42_*/final_model.pt \
        --modes $mode \
        --track_trajectory True \
        --save_trajectory results/tta/trajectory_mode${mode}.json \
        --num_adapt_episodes 50
done
```

### Success Criteria

1. **Weight Shift Magnitude**
   ```
   shift = ||weights_after - weights_before||
   Expected: shift > 0.1 (meaningful change)
   ```

2. **Mode-Specific Shift Direction**
   ```
   For each mode i:
   - weights should shift toward slot that's best for mode i
   - Different modes should shift in different directions
   ```

3. **Convergence**
   ```
   Weights should stabilize after ~20-30 episodes
   Variance in final weights < 0.05
   ```

### Visualization

```python
import matplotlib.pyplot as plt

# Plot weight trajectories for each mode
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for mode in [0, 1, 2]:
    ax = axes[mode]
    trajectory = load_trajectory(f"trajectory_mode{mode}.json")
    for slot in range(3):
        ax.plot(trajectory[:, slot], label=f"Slot {slot}")
    ax.set_title(f"Mode {mode} Adaptation")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Weight")
    ax.legend()
plt.savefig("results/tta/weight_trajectories.png")
```

---

## Experiment 3: Mode Identification from Weights

### Hypothesis
After TTA, the gating weights should encode information about which mode the agent adapted to. If we can predict the mode from weights, the abstraction bank learned mode-specific representations.

### Required Code Changes

Add mode prediction analysis to `test_time_adapt.py`:
1. After TTA on each mode, record final weights
2. Train simple classifier (or use nearest-centroid)
3. Evaluate mode prediction accuracy

### Experiments

```bash
# Collect weight signatures for each mode (multiple runs)
for seed in 42 123 456 789 1000; do
    for mode in 0 1 2; do
        python src/test_time_adapt.py \
            --checkpoint results/runs/best_config_strong_astral_42_*/final_model.pt \
            --modes $mode \
            --num_adapt_episodes 30 \
            --save_final_weights results/tta/weights_mode${mode}_seed${seed}.json \
            --seed $seed
    done
done

# Analyze mode separability
python src/analyze_mode_weights.py \
    --weights_dir results/tta/ \
    --output results/tta/mode_separability.png
```

### Success Criteria

1. **Mode Prediction Accuracy**
   ```
   accuracy = correct_predictions / total_predictions
   Expected: > 80% (significantly above 33% random)
   ```

2. **Weight Cluster Separation**
   ```
   Using k-means or PCA:
   - 3 distinct clusters should emerge
   - Each cluster corresponds to one mode
   ```

3. **Statistical Significance**
   ```
   Chi-squared test: p < 0.01 for mode-weight association
   ```

---

## Experiment 4: Adaptation Speed Comparison

### Hypothesis
ASTRAL should adapt faster than baseline because it only needs to reweight existing abstractions, not learn new features.

### Experiments

```bash
# Compare adaptation curves
for episodes in 5 10 15 20 25 30; do
    # ASTRAL
    python src/test_time_adapt.py \
        --checkpoint results/runs/best_config_strong_astral_42_*/final_model.pt \
        --num_adapt_episodes $episodes \
        --save_results results/tta/astral_adapt_${episodes}.json
    
    # Baseline
    python src/test_time_adapt.py \
        --checkpoint results/runs/baseline_baseline_42_*/final_model.pt \
        --adapt_mode policy_head \
        --num_adapt_episodes $episodes \
        --save_results results/tta/baseline_adapt_${episodes}.json
done
```

### Success Criteria

1. **Faster Initial Improvement**
   ```
   ASTRAL improvement at 10 episodes > Baseline improvement at 10 episodes
   ```

2. **Sample Efficiency**
   ```
   Episodes to reach 90% of final improvement:
   ASTRAL < Baseline
   ```

---

## Experiment 5: Cross-Seed Consistency

### Hypothesis
TTA improvement should be consistent across different random seeds, not a lucky artifact.

### Experiments

```bash
# Run TTA on all best_config_strong seeds
for seed in 42 123 456; do
    python src/test_time_adapt.py \
        --checkpoint results/runs/best_config_strong_astral_${seed}_*/final_model.pt \
        --num_adapt_episodes 30 \
        --num_eval_episodes 20 \
        --save_results results/tta/tta_seed${seed}.json
done

# Also run on multiple TTA random seeds
for tta_seed in 1 2 3 4 5; do
    python src/test_time_adapt.py \
        --checkpoint results/runs/best_config_strong_astral_42_*/final_model.pt \
        --num_adapt_episodes 30 \
        --seed $tta_seed \
        --save_results results/tta/tta_run${tta_seed}.json
done
```

### Success Criteria

1. **Consistent Direction**
   ```
   All seeds show positive TTA improvement
   ```

2. **Low Variance**
   ```
   std(improvements) / mean(improvements) < 0.5
   ```

3. **Statistical Significance**
   ```
   One-sample t-test: mean improvement > 0, p < 0.05
   ```

---

## Implementation Priority

| Experiment | Effort | Impact | Priority |
|------------|--------|--------|----------|
| 1. Baseline Control | Medium | **Critical** | 🔴 High |
| 2. Weight Trajectory | Low | High | 🟡 Medium |
| 3. Mode Identification | Medium | High | 🟡 Medium |
| 4. Adaptation Speed | Low | Medium | 🟢 Low |
| 5. Cross-Seed | Low | Medium | 🟢 Low |

---

## Summary: Evidence Required

To confidently claim "Test-Time Abstraction Learning Works":

| Evidence | Current Status | Required |
|----------|----------------|----------|
| TTA improves performance | ✅ Shown | ✅ |
| Improvement > baseline fine-tuning | ❌ Not tested | **Critical** |
| Weights shift meaningfully | ⚠️ Partially shown | Important |
| Mode-specific weight patterns | ❌ Not tested | Important |
| Consistent across seeds | ⚠️ Limited data | Nice to have |

**Bottom line:** We need **Experiment 1 (Baseline Control)** to make a strong scientific claim.

---

## Quick Start Script

```bash
#!/bin/bash
# run_tta_validation.sh

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== TTA Validation Experiments ==="

# 1. Critical: Baseline comparison
echo "[1/3] Baseline Control Comparison..."
# Requires code changes first - see Experiment 1

# 2. Weight trajectory (if implemented)
echo "[2/3] Weight Trajectory Analysis..."
# Requires code changes first - see Experiment 2

# 3. Cross-seed consistency (can run now)
echo "[3/3] Cross-Seed Consistency..."
for seed in 42 123 456; do
    checkpoint=$(ls -d results/runs/best_config_strong_astral_${seed}_* 2>/dev/null | head -1)
    if [ -n "$checkpoint" ]; then
        echo "Testing seed $seed..."
        python src/test_time_adapt.py \
            --checkpoint "$checkpoint/final_model.pt" \
            --num_adapt_episodes 30 \
            --num_eval_episodes 20
    fi
done

echo "=== TTA Validation Complete ==="
```

