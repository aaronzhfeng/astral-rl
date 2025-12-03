# ASTRAL Experiment Checklist

**Last Updated:** December 2, 2025

This document tracks all proposed experiments and their completion status.

---

## Quick Summary

| Category | Proposed | Completed | Ready to Run | Pending |
|----------|----------|-----------|--------------|---------|
| Core Validation | 6 | 6 | 0 | 0 |
| Interpretability | 7 | 7 | 0 | 0 |
| Slot Collapse (Strong Reg) | 11 | 11 | 0 | 0 |
| Slot Collapse (Code Changes) | 18 | 0 | 0 | 18 |
| TTA Validation | 5 | 0 | 1 | 4 |
| **Total** | **47** | **24** | **1** | **22** |

---

## 1. Core Validation Experiments

**Source:** `07_experiment_guide.md` Section 2

| Experiment | Seeds | Timesteps | Status |
|------------|-------|-----------|--------|
| ASTRAL training | 42, 123, 456 | 500k | ✅ Done |
| Baseline training | 42, 123, 456 | 500k | ✅ Done |

**Results:** `results/runs/astral_astral_*`, `results/runs/baseline_baseline_*`

---

## 2. Interpretability Experiments

**Source:** `07_experiment_guide.md` Section 5

| Experiment | Config | Status |
|------------|--------|--------|
| Gumbel-Softmax | `--use_gumbel True` | ✅ Done |
| Hard Routing | `--hard_routing True` | ✅ Done |
| Orthogonal Init | `--orthogonal_init True` | ✅ Done |
| Temperature Annealing | `--temp_anneal True` | ✅ Done |
| Contrastive Loss | `--lambda_contrast 0.05` | ✅ Done |
| Slot Prediction | `--slot_prediction True` | ✅ Done |
| All Combined | All flags | ✅ Done |

**Results:** `results/runs/interp_*`

---

## 3. TTA & Intervention Tests

**Source:** `07_experiment_guide.md` Sections 3-4

| Test | Models Tested | Status |
|------|---------------|--------|
| Test-Time Adaptation | 22 models | ✅ Done |
| Causal Interventions | 22 models | ✅ Done |

**Results:** `results/tta/`, `results/interventions/`

---

## 4. Slot Collapse Experiments (No Code Changes)

**Source:** `08_experiments_slot_collapse.md` Experiment 1

| Experiment | Config | Status |
|------------|--------|--------|
| Strong Contrastive λ=0.1 | `--lambda_contrast 0.1` | ✅ Done |
| Strong Contrastive λ=0.2 | `--lambda_contrast 0.2` | ✅ Done |
| Strong Contrastive λ=0.5 | `--lambda_contrast 0.5` | ✅ Done |
| Strong Load Balance λ=0.05 | `--lambda_lb 0.05` | ✅ Done |
| Strong Load Balance λ=0.1 | `--lambda_lb 0.1` | ✅ Done |
| Strong Weight Entropy λ=0.05 | `--lambda_w_ent 0.05` | ✅ Done |
| Strong Weight Entropy λ=0.1 | `--lambda_w_ent 0.1` | ✅ Done |
| Combined Strong Reg | All strong flags | ✅ Done |
| Best Config Strong (seed 42) | Gumbel+TempAnneal+Contrast+LB | ✅ Done |
| Best Config Strong (seed 123) | Gumbel+TempAnneal+Contrast+LB | ✅ Done |
| Best Config Strong (seed 456) | Gumbel+TempAnneal+Contrast+LB | ✅ Done |

**Results:** `results/runs/strong_*`, `results/runs/best_config_strong_*`

---

## 5. Slot Collapse Experiments (Require Code Changes)

**Source:** `08_experiments_slot_collapse.md` Experiments 2-7

### Experiment 2: Curriculum Learning

| Experiment | Status | Requires |
|------------|--------|----------|
| Pre-train Mode 0 | ❌ Not Run | `--single_mode` flag |
| Pre-train Mode 1 | ❌ Not Run | `--single_mode` flag |
| Pre-train Mode 2 | ❌ Not Run | `--single_mode` flag |
| Curriculum Fine-tune | ❌ Not Run | `--load_checkpoint` support |
| Progressive Mode Mixing | ❌ Not Run | `--progressive_mode_schedule` |

### Experiment 3: Hard Slot Assignment

| Experiment | Status | Requires |
|------------|--------|----------|
| Top-1 Routing | ❌ Not Run | `--routing_mode topk` |
| Top-2 Routing | ❌ Not Run | `--routing_mode topk` |
| Slot Dropout 0.3 | ❌ Not Run | `--slot_dropout` flag |
| Slot Dropout 0.5 | ❌ Not Run | `--slot_dropout` flag |

### Experiment 4: Mode-Conditioned Supervision

| Experiment | Status | Requires |
|------------|--------|----------|
| Mode Classification Head | ❌ Not Run | `--mode_classification` |
| Mode-Slot Alignment | ❌ Not Run | `--mode_slot_alignment` |
| Slot-Specific Mode Pred | ❌ Not Run | `--slot_mode_prediction` |

### Experiment 5: Architecture Modifications

| Experiment | Status | Requires |
|------------|--------|----------|
| Large Abstraction (d=128) | ❌ Not Run | `--abstraction_dim` |
| 6 Slots | ❌ Not Run | Already supported |
| 9 Slots | ❌ Not Run | Already supported |
| Separate Gating | ❌ Not Run | Architecture change |
| MoE-Style Sparse | ❌ Not Run | `--moe_style` |

### Experiment 6: Environment Modifications

| Experiment | Status | Requires |
|------------|--------|----------|
| Extreme Mode Differences | ❌ Not Run | Env parameter changes |
| 5 Modes | ❌ Not Run | `--num_modes` |
| Adversarial Switching | ❌ Not Run | `--adversarial_mode_switch` |

### Experiment 7: Two-Phase Training

| Experiment | Status | Requires |
|------------|--------|----------|
| Phase 1: Frozen Gating | ❌ Not Run | `--freeze_gating` |
| Phase 2: Frozen Policy | ❌ Not Run | `--freeze_policy` |
| Alternating Training | ❌ Not Run | `--alternating_freeze` |

---

## 6. TTA Validation Experiments

**Source:** `09_experiments_tta_validation.md`

| Experiment | Priority | Status | Result |
|------------|----------|--------|--------|
| **1. Baseline Control Comparison** | 🔴 Critical | ✅ Run | ⚠️ Inconclusive (baseline broken) |
| 1b. Suboptimal ASTRAL | 🔴 Critical | ✅ Run | ❌ TTA hurt performance |
| 2. Weight Trajectory Analysis | 🟡 Medium | ❌ Not Run | — |
| 3. Mode Identification | 🟡 Medium | ❌ Not Run | — |
| 4. Adaptation Speed Comparison | 🟢 Low | ❌ Not Run | — |
| 5. Cross-Seed Consistency | 🟢 Low | ❌ Not Run | — |

**Key Finding:** TTA doesn't improve performance on any tested model. Root cause likely slot collapse.

---

## Key Findings So Far

### From Initial Experiments (07_experiment_guide.md)

1. **Slot Collapse:** 73% of models collapsed to single slot
2. **TTA Works (when diverse):** +35% avg improvement with diverse slots
3. **Performance-Interpretability Tradeoff:** Diverse slots → lower raw performance

### From Strong Regularization (08_experiments_slot_collapse.md Exp 1)

1. **Collapse Fixed:** `best_config_strong` achieves [0.56, 0.41, 0.02] weights
2. **Performance Maintained:** ~450 avg return (vs ~150 for interp_all)
3. **Winning Config:**
   ```bash
   --use_gumbel True --temp_anneal True --tau_start 5.0 --tau_end 0.5 \
   --lambda_contrast 0.1 --lambda_lb 0.05 --slot_prediction True
   ```

### Still Unproven

1. ❓ Is ASTRAL TTA better than baseline fine-tuning?
2. ❓ Do weights shift toward "correct" slot during TTA?
3. ❓ Is there true mode-slot correspondence?

---

## Next Priority

To strengthen the TTA claim, implement and run:

1. **Experiment 1 from 09_experiments_tta_validation.md** (Baseline Control)
   - Compare ASTRAL TTA vs Baseline fine-tuning
   - This is the critical missing evidence

---

## Analysis Documents

| File | Content |
|------|---------|
| `results/analysis/01_experiment_results_initial.md` | Initial experiment analysis |
| `results/analysis/02_experiment_results_strong_reg.md` | Strong regularization analysis |

---

## How to Run Remaining Experiments

### Already Runnable (No Code Changes)

```bash
# Slot count variations (from 08_experiments_slot_collapse.md Exp 5B)
python src/train.py --exp_name slots_6 --num_abstractions 6 --total_timesteps 300000
python src/train.py --exp_name slots_9 --num_abstractions 9 --total_timesteps 300000

# Cross-seed TTA consistency (from 09_experiments_tta_validation.md Exp 5)
for seed in 42 123 456; do
    python src/test_time_adapt.py \
        --checkpoint results/runs/best_config_strong_astral_${seed}_*/final_model.pt \
        --num_adapt_episodes 30
done
```

### Require Implementation

See individual experiment documents for required code changes:
- `08_experiments_slot_collapse.md` for Experiments 2-7
- `09_experiments_tta_validation.md` for Experiments 1-4

