# ASTRAL Experiment Checklist

**Last Updated:** December 4, 2025

This document tracks all proposed experiments and their completion status.

---

## Quick Summary

| Category | Proposed | Completed | Pending |
|----------|----------|-----------|---------|
| Core Training | 6 | 6 | 0 |
| Interpretability | 7 | 7 | 0 |
| Slot Collapse (No Code) | 11 | 11 | 0 |
| Slot Dropout (Implemented) | 4 | 4 | 0 |
| TTA Validation | 5 | 3 | 2 |
| SB3 PPO Baseline | 2 | 2 | 0 |
| Fair Comparison (A-D) | 4 | 4 | 0 |
| Extreme Few-Shot | 1 | 1 | 0 |
| Slot Collapse (Code Required) | 14 | 0 | 14 |
| **Total** | **54** | **38** | **16** |

---

## ✅ COMPLETED EXPERIMENTS

---

### 1. Core Training Experiments

**Source:** `07_experiment_guide.md` Section 2

| Experiment | Seeds | Timesteps | Status |
|------------|-------|-----------|--------|
| ASTRAL training | 42, 123, 456 | 500k | ✅ Done |
| Baseline (custom) training | 42, 123, 456 | 500k | ✅ Done |

**Results:** `results/runs/astral_astral_*`, `results/runs/baseline_baseline_*`

**Note:** Custom baseline was later found to be broken (policy entropy stuck at max).

---

### 2. Interpretability Experiments

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

### 3. TTA & Intervention Tests (Initial)

**Source:** `07_experiment_guide.md` Sections 3-4

| Test | Models Tested | Status |
|------|---------------|--------|
| Test-Time Adaptation | 22 models | ✅ Done |
| Causal Interventions | 22 models | ✅ Done |

**Results:** `results/tta/`, `results/interventions/`

---

### 4. Slot Collapse - Strong Regularization

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

### 5. Slot Dropout Implementation & Training

**Source:** `08_experiments_slot_collapse.md` Experiment 3B (Implemented)

| Experiment | Config | Status | Result |
|------------|--------|--------|--------|
| Slot Dropout p=0.3 | `--slot_dropout 0.3` | ✅ Done | **Best TTA: +11.4** |
| Slot Dropout p=0.5 | `--slot_dropout 0.5` | ✅ Done | Over-regularized |
| Diverse Strong | Combined flags | ✅ Done | Good diversity |
| Best Config Long | Extended training | ✅ Done | Stable |

**Results:** `results/runs/slot_dropout_*`, `results/tta_final_validation/`

**Key Finding:** Slot dropout p=0.3 was the **only** configuration to achieve positive TTA improvement.

---

### 6. SB3 PPO Baseline (Replacement)

**Source:** New experiments after discovering custom baseline was broken

| Experiment | Config | Status | Mean Return |
|------------|--------|--------|-------------|
| SB3 PPO Training | 500k steps, 16 envs | ✅ Done | ~443 |
| SB3 PPO Fine-tuning Test | 10 episodes adapt | ✅ Done | +77 improvement |

**Results:** `results/sb3_baseline/`, `results/sb3_finetuning/`

**Script:** `scripts/train_sb3_baseline.py`, `scripts/test_sb3_finetuning.py`

---

### 7. Fair Comparison Experiments (A-D)

**Source:** New experiments designed to fairly compare ASTRAL vs fine-tuning

| Experiment | Description | Status | Key Finding |
|------------|-------------|--------|-------------|
| **A: Parameter-Matched** | ASTRAL gating (~4.3k) vs policy-head (~4.3k) | ✅ Done | Gating 13× less variance |
| **B: Catastrophic Forgetting** | Mode-specific adaptation → evaluate all modes | ✅ Done | Gating 10× less forgetting |
| **C: Few-Shot Speed** | 1-30 episodes adaptation | ✅ Done | Gating stable across budgets |
| **D: Extreme Modes** | gravity 5-25, length 0.3-0.8 | ✅ Done | Gating advantage increases |

**Results:** `results/fair_comparison/`

**Script:** `scripts/run_fair_comparison.py`

**Key Insight:** ASTRAL's advantage is **stability**, not raw performance.

---

### 8. Extreme Few-Shot Experiments

**Source:** Extended from Experiment C

| Episodes | Gating | Policy-Head | Full Fine-tune |
|----------|--------|-------------|----------------|
| 1 | -5.0 | -50.0 | -80.0 |
| 3 | +10.0 | -30.0 | -40.0 |
| 5 | +15.0 | -10.0 | -20.0 |
| 10 | +20.0 | +10.0 | +5.0 |
| 20 | +25.0 | +25.0 | +30.0 |
| 30 | +25.0 | +30.0 | +40.0 |
| 50 | +25.0 | +35.0 | +50.0 |

**Results:** `results/extreme_fewshot/`

**Script:** `scripts/run_extreme_fewshot.py`

**Key Insight:** Gating dominates at <10 episodes; full fine-tuning wins at >30 episodes.

---

### 9. TTA Comparison (Post-Fix)

**Source:** `09_experiments_tta_validation.md` Experiment 1

| Test | Status | Result |
|------|--------|--------|
| ASTRAL (gating) TTA | ✅ Done | Variable by model |
| ASTRAL (policy-head) TTA | ✅ Done | Often worse |
| Baseline fine-tuning | ✅ Done | Better raw improvement |

**Key Finding:** TTA benefit requires slot diversity (from slot dropout).

---

## ❌ NOT COMPLETED (Require Code Changes)

---

### Slot Collapse - Curriculum Learning (Exp 2)

| Experiment | Status | Requires |
|------------|--------|----------|
| Pre-train Mode 0 | ❌ Not Run | `--single_mode` flag |
| Pre-train Mode 1 | ❌ Not Run | `--single_mode` flag |
| Pre-train Mode 2 | ❌ Not Run | `--single_mode` flag |
| Curriculum Fine-tune | ❌ Not Run | `--load_checkpoint` support |
| Progressive Mode Mixing | ❌ Not Run | `--progressive_mode_schedule` |

---

### Slot Collapse - Mode-Conditioned Supervision (Exp 4)

| Experiment | Status | Requires |
|------------|--------|----------|
| Mode Classification Head | ❌ Not Run | `--mode_classification` |
| Mode-Slot Alignment | ❌ Not Run | `--mode_slot_alignment` |
| Slot-Specific Mode Pred | ❌ Not Run | `--slot_mode_prediction` |

---

### Slot Collapse - Architecture Modifications (Exp 5)

| Experiment | Status | Requires |
|------------|--------|----------|
| Large Abstraction (d=128) | ❌ Not Run | `--abstraction_dim` |
| Separate Gating Network | ❌ Not Run | Architecture change |
| MoE-Style Sparse | ❌ Not Run | `--moe_style` |

---

### Slot Collapse - Two-Phase Training (Exp 7)

| Experiment | Status | Requires |
|------------|--------|----------|
| Phase 1: Frozen Gating | ❌ Not Run | `--freeze_gating` |
| Phase 2: Frozen Policy | ❌ Not Run | `--freeze_policy` |
| Alternating Training | ❌ Not Run | `--alternating_freeze` |

---

### TTA Validation - Remaining

| Experiment | Status | Notes |
|------------|--------|-------|
| Weight Trajectory Analysis | ❌ Not Run | Visualize how weights shift during adaptation |
| Mode Identification from Weights | ❌ Not Run | Can we predict mode from learned weights? |

---

## Key Findings Summary

### Slot Collapse
- **73% of configurations collapse** to single dominant slot
- **Slot dropout (p=0.3)** is the only solution that enables positive TTA
- Strong regularization helps diversity but not functional diversity

### TTA Reality
- TTA **doesn't automatically help** - requires slot diversity
- Even with diversity, improvement is modest (+11 at best)
- Baseline fine-tuning often achieves higher raw improvement

### ASTRAL's True Value
- **Stability**: 13× less variance in worst-case scenarios
- **Forgetting resistance**: 10× less catastrophic forgetting
- **Few-shot**: Dominates when adaptation budget is ≤10 episodes
- **Extreme modes**: Advantage increases with distribution shift

---

## Analysis Documents

| File | Content |
|------|---------|
| `results/analysis/00_experiment_index.md` | Index of all analysis docs |
| `results/analysis/01_slot_collapse_experiment_results.md` | Initial experiments |
| `results/analysis/02_slot_collapse_experiment_results.md` | Strong regularization |
| `results/analysis/03_tta_comparison_analysis.md` | TTA comparison |
| `results/analysis/04_sb3_baseline_training.md` | SB3 PPO baseline |
| `results/analysis/05_sb3_finetuning_test.md` | Baseline fine-tuning |
| `results/analysis/06_tta_final_validation.md` | Final TTA validation |
| `results/analysis/07_fair_comparison_experiments.md` | Experiments A-D |
| `results/analysis/08_extreme_fewshot.md` | Extreme few-shot |

---

## Paper Status

- **Version 1:** `report/version_1/` - Original draft (archived)
- **Version 2:** `report/version_2/` - Current version (complete)
  - 20 pages with appendix
  - 9 figures, 10 tables, 54 references
  - Reframed narrative: stability over raw performance
