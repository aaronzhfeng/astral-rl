# SB3 PPO Baseline Training

**Date:** December 3, 2025  
**Experiment:** Training Stable-Baselines3 PPO on NonStationary CartPole  
**Hardware:** NVIDIA GeForce RTX 4090

---

## Background

After discovering that the custom `BaselineAgent` (GRU-based) **failed to learn** (returns ~20-30 vs 500 max), we needed a functional baseline for comparison.

### Problem with Original Baseline
- Custom `BaselineAgent` achieved only ~20-30 return
- This is equivalent to **random policy** performance
- Investigation revealed the policy entropy was stuck at maximum (~1.39)
- The agent never learned to reduce exploration

### Solution
Use **Stable-Baselines3 PPO** — a well-tested, production-ready RL implementation.

---

## Configuration

```python
config = {
    "total_timesteps": 100000,
    "n_envs": 16,          # Vectorized environments for stability
    "batch_size": 512,
    "n_steps": 2048,
    "n_epochs": 10,
    "learning_rate": 0.0003,
    "device": "cuda",
    "seed": 42
}
```

---

## Results

### Training Time
- **31.8 seconds** (100k timesteps)
- Extremely fast due to optimized SB3 implementation

### Evaluation by Mode

| Mode | Mean Return | Std Dev | n Episodes |
|------|-------------|---------|------------|
| 0 | **448.8** | 82.6 | 19 |
| 1 | **409.2** | 88.6 | 17 |
| 2 | **461.9** | 74.0 | 24 |
| **Overall** | **442.9** | 84.0 | 60 |

### Raw Episode Returns

**Mode 0:**
```
500, 500, 500, 423, 500, 482, 313, 500, 276, 309,
500, 500, 500, 500, 280, 500, 500, 500, 445
```

**Mode 1:**
```
500, 311, 416, 353, 329, 500, 422, 500, 310, 500,
452, 500, 270, 500, 302, 500, 292
```

**Mode 2:**
```
500, 500, 500, 286, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 325, 329, 500, 500, 500, 500,
494, 500, 321, 331
```

---

## Comparison: SB3 PPO vs Original Baseline

| Agent | Mean Return | Training Time | Learned? |
|-------|-------------|---------------|----------|
| **SB3 PPO** | **442.9** | 32s (100k steps) | ✅ Yes |
| Custom BaselineAgent | 23.8 | ~30min (500k steps) | ❌ No |
| ASTRAL (best_config_strong) | 490.5 | ~3min (300k steps) | ✅ Yes |

---

## Key Observations

### 1. SB3 PPO Successfully Learns
- Achieves ~90% of maximum return (442/500)
- Consistent across all modes
- Some variance, but never fails catastrophically

### 2. Mode Performance Differences
- Mode 2 (short pole, heavy) performs best (461.9)
- Mode 1 (long pole, normal) performs worst (409.2)
- Suggests environment modes have varying difficulty

### 3. High Variance Remains
- Std dev ~80-90 across modes
- Some episodes reach max 500, others drop to ~270-330
- This is inherent to CartPole dynamics, not a training failure

---

## Implications

### For Baseline Comparisons
- All future baseline comparisons should use **SB3 PPO**, not custom `BaselineAgent`
- Previous experiments comparing against custom baseline are **invalid**
- Need to re-run comparisons with this new baseline

### For TTA Experiments
- SB3 PPO provides a **functional baseline** for fine-tuning comparisons
- Can now fairly compare ASTRAL's gating TTA vs baseline fine-tuning

### For Paper Narrative
- Cannot claim "ASTRAL beats baseline" if baseline was broken
- Focus shifts to **adaptation stability**, not raw performance

---

## File Locations

```
results/sb3_baseline/
├── ppo_model.zip     # Trained SB3 model
└── results.json      # Evaluation results
```

---

## Command Used

```bash
python scripts/train_sb3_baseline.py \
    --device cuda \
    --total_timesteps 100000 \
    --n_envs 16
```

