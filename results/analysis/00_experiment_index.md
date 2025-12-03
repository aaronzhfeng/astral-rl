# ASTRAL Experiment Index

**Project:** ASTRAL (Abstraction-based Slot-routed Test-time Reinforced Adaptation Layer)  
**Date Range:** December 1-3, 2025  
**Hardware:** NVIDIA GeForce RTX 4090

---

## Experiment Timeline

| # | File | Date | Focus | Key Finding |
|---|------|------|-------|-------------|
| 01 | `01_experiment_results_initial.md` | Dec 1 | Initial training & TTA | Slot collapse in 73% of models |
| 02 | `02_experiment_results_strong_reg.md` | Dec 1 | Strong regularization | `best_config_strong` fixes collapse |
| 03 | `03_tta_comparison_analysis.md` | Dec 2 | ASTRAL vs Baseline TTA | Custom baseline is **broken** |
| 04 | `04_sb3_baseline_training.md` | Dec 3 | SB3 PPO baseline | Functional baseline achieved (~443) |
| 05 | `05_sb3_finetuning_test.md` | Dec 3 | SB3 fine-tuning | Fine-tuning improves +77 avg |
| 06 | `06_tta_final_validation.md` | Dec 3 | Slot dropout TTA | `slot_dropout_0.3` enables positive TTA |
| 07 | `07_fair_comparison_experiments.md` | Dec 3 | Fair comparison (A,B,C,D) | Gating prevents forgetting |
| 08 | `08_extreme_fewshot.md` | Dec 3 | Extreme modes few-shot | Gating is safest strategy |

---

## Major Discoveries

### Phase 1: Initial Training (Dec 1)
- **Problem:** 73% of models exhibited **slot collapse** (single slot dominates)
- **Solution:** Combined regularization (contrast + load balancing + temp anneal + slot prediction)
- **Best Config:** `best_config_strong` achieves diversity AND high performance (~450 return)

### Phase 2: TTA Validation (Dec 2)
- **Problem:** Custom `BaselineAgent` returns ~25 (random policy level)
- **Discovery:** TTA doesn't help optimal models (ceiling effect)
- **Discovery:** Policy-head TTA causes catastrophic forgetting on ASTRAL

### Phase 3: Fair Comparison (Dec 3)
- **Fix:** Replaced broken baseline with SB3 PPO (~443 return)
- **Discovery:** SB3 fine-tuning (+77) >> ASTRAL gating TTA (+11 best)
- **Discovery:** Gating prevents forgetting (10x less than full fine-tuning)
- **Conclusion:** ASTRAL value is **stability**, not **maximum improvement**

---

## Key Results Summary

### Training Performance

| Agent | Mean Return | Learns? | Notes |
|-------|-------------|---------|-------|
| ASTRAL (best_config_strong) | ~490 | ✅ Yes | Near-optimal |
| SB3 PPO (100k steps) | ~443 | ✅ Yes | Functional baseline |
| Custom BaselineAgent | ~25 | ❌ No | **Broken** |

### TTA Effectiveness

| Configuration | TTA Improvement | Why |
|---------------|-----------------|-----|
| slot_dropout_0.3 | **+11.4** | Only positive TTA |
| best_config_strong | -3.0 | Ceiling effect |
| diverse_strong | -64.6 | Diversity without specialization |

### Adaptation Stability

| Method | Forgetting (Avg) | Variance | Safe? |
|--------|------------------|----------|-------|
| Gating-only | -25.8 | Low | ✅ Yes |
| Policy-head | -32.5 | High | ⚠️ Risky |
| Full fine-tune | -250.1 | Very High | ❌ No |

---

## Experiment Categories

### Category A: Training & Architecture
- `01_experiment_results_initial.md` — Core ASTRAL training
- `02_experiment_results_strong_reg.md` — Regularization experiments
- `04_sb3_baseline_training.md` — SB3 PPO baseline

### Category B: Test-Time Adaptation
- `03_tta_comparison_analysis.md` — Initial TTA comparison
- `05_sb3_finetuning_test.md` — SB3 fine-tuning capability
- `06_tta_final_validation.md` — Slot dropout TTA experiments

### Category C: Fair Comparison Experiments
- `07_fair_comparison_experiments.md` — A, B, C, D experiments
- `08_extreme_fewshot.md` — Extreme physics few-shot tests

---

## Final Conclusions

### What ASTRAL Achieves ✅
1. **Prevents catastrophic forgetting** during adaptation
2. **Provides stable, predictable adaptation** across budgets
3. **Enables interpretable slot routing** (when regularized)
4. **Matches baseline training performance** (~490 vs ~443)

### What ASTRAL Does NOT Achieve ❌
1. **Does not beat full fine-tuning** on raw improvement
2. **Does not guarantee mode-slot correspondence**
3. **TTA requires slot dropout** to be effective

### Recommended Use Cases
- **Multi-mode deployment** where forgetting is dangerous
- **Risk-averse adaptation** where stability > maximum gain
- **Interpretability requirements** where slot analysis is needed

---

## File Locations

```
results/
├── analysis/                          # This folder - all documentation
│   ├── 00_experiment_index.md         # This file
│   ├── 01_experiment_results_initial.md
│   ├── 02_experiment_results_strong_reg.md
│   ├── 03_tta_comparison_analysis.md
│   ├── 04_sb3_baseline_training.md
│   ├── 05_sb3_finetuning_test.md
│   ├── 06_tta_final_validation.md
│   ├── 07_fair_comparison_experiments.md
│   └── 08_extreme_fewshot.md
├── runs/                              # TensorBoard logs & checkpoints
├── sb3_baseline/                      # SB3 PPO model
├── sb3_finetuning/                    # Fine-tuning test results
├── fair_comparison/                   # Experiments A-D results
├── extreme_fewshot/                   # Extreme mode results
├── tta_final_validation/              # Slot dropout TTA results
├── tta_comparison/                    # Initial TTA comparison
├── interventions/                     # Causal intervention plots
└── tta/                               # Original TTA results
```

---

## Navigation

| If you want to... | Read this file |
|-------------------|----------------|
| Understand initial results | `01_experiment_results_initial.md` |
| See how slot collapse was fixed | `02_experiment_results_strong_reg.md` |
| Understand the baseline problem | `03_tta_comparison_analysis.md` |
| See the working baseline | `04_sb3_baseline_training.md` |
| Compare TTA methods | `07_fair_comparison_experiments.md` |
| See extreme mode results | `08_extreme_fewshot.md` |

