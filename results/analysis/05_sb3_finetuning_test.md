# SB3 PPO Fine-Tuning Test

**Date:** December 3, 2025  
**Experiment:** Testing fine-tuning capabilities of SB3 PPO baseline  
**Hardware:** NVIDIA GeForce RTX 4090

---

## Objective

Establish that SB3 PPO can be **fine-tuned at test time** to adapt to specific modes. This provides a strong baseline for comparison against ASTRAL's gating-only TTA.

---

## Method

1. **Load pre-trained SB3 PPO model** (100k timesteps training)
2. **Evaluate on each mode** (20 episodes) — "Before" performance
3. **Fine-tune for 1000 timesteps** on a fixed mode
4. **Evaluate again** on the same mode — "After" performance

### Fine-Tuning Configuration
- **Adaptation timesteps:** 1000 (equivalent to ~10 episodes)
- **Learning rate:** Same as training (0.0003)
- **Vectorized:** 16 parallel environments

---

## Results

| Mode | Before (Mean) | Before (Std) | After (Mean) | After (Std) | **Improvement** |
|------|---------------|--------------|--------------|-------------|-----------------|
| 0 | 426.05 | 91.9 | 488.3 | 25.4 | **+62.3** |
| 1 | 421.95 | 85.6 | 500.0 | 0.0 | **+78.1** |
| 2 | 410.10 | 102.2 | 500.0 | 0.0 | **+89.9** |

### Average Improvement: **+76.7** (≈ +18%)

---

## Key Observations

### 1. Fine-Tuning Works Extremely Well
- All modes reach **near-optimal performance** after fine-tuning
- Modes 1 and 2 reach **perfect 500 return** with **zero variance**
- Mode 0 reaches 488.3 with low variance (25.4)

### 2. Variance Dramatically Reduced
| Mode | Std Before | Std After | Reduction |
|------|------------|-----------|-----------|
| 0 | 91.9 | 25.4 | -72% |
| 1 | 85.6 | 0.0 | -100% |
| 2 | 102.2 | 0.0 | -100% |

Fine-tuning not only improves mean return but **stabilizes** performance.

### 3. Rapid Adaptation
- Only 1000 timesteps (~10 episodes) needed
- Entire fine-tuning takes < 1 second
- SB3 is highly efficient at adaptation

---

## Comparison: ASTRAL TTA vs SB3 Fine-Tuning

| Method | Agent | Avg Improvement | Notes |
|--------|-------|-----------------|-------|
| **SB3 Full Fine-Tune** | PPO | **+76.7** | All parameters adapted |
| ASTRAL Gating TTA | best_config_strong | -3.0 | Only gating adapted |
| ASTRAL Gating TTA | slot_dropout_0.3 | +11.4 | Only model with positive TTA |

### Interpretation
- SB3 fine-tuning is **dramatically more effective** than ASTRAL gating TTA
- This is expected: SB3 adapts **all parameters**, ASTRAL adapts only gating (~4k params)
- However, this comparison is **unfair** — need parameter-matched comparison

---

## Why SB3 Fine-Tuning is Effective

1. **Full network adaptation:** All 67k+ parameters can change
2. **Stable optimization:** SB3's PPO implementation is highly tuned
3. **Vectorized training:** 16 envs provide stable gradients
4. **Good starting point:** Pre-trained model already near-optimal

---

## Implications for ASTRAL

### The Challenge
- ASTRAL's gating TTA adapts ~4,300 parameters
- SB3 full fine-tuning adapts ~67,000+ parameters
- **Fair comparison requires parameter-matched adaptation**

### Next Steps
1. Compare ASTRAL gating vs adapting **only SB3's policy head** (~4k params)
2. Test for **catastrophic forgetting** — does SB3 forget other modes?
3. Test **few-shot efficiency** — who adapts faster with limited data?

---

## File Locations

```
results/sb3_finetuning/
└── results.json      # Fine-tuning results by mode
```

---

## Command Used

```bash
python scripts/test_sb3_finetuning.py
```

---

## Critical Note

This experiment revealed that **full network fine-tuning significantly outperforms gating-only TTA**. The ASTRAL narrative must shift from "TTA beats baseline" to "TTA provides **stability** and **forgetting resistance** under constrained adaptation."

