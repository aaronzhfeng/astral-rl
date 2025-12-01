# ASTRAL Experiment Guide

Complete commands for validating the ASTRAL approach and running all experiments.

---

## Table of Contents

1. [Setup](#1-setup)
2. [Core Validation](#2-core-validation)
3. [Test-Time Adaptation](#3-test-time-adaptation)
4. [Causal Interventions](#4-causal-interventions)
5. [Interpretability Experiments](#5-interpretability-experiments)
6. [Ablation Studies](#6-ablation-studies)
7. [Analysis & Visualization](#7-analysis--visualization)

---

## 1. Setup

### 1.1 Environment Setup

```bash
# Navigate to project
cd astral-rl

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify dependencies
python -c "import torch; import gymnasium; print('PyTorch:', torch.__version__); print('Ready!')"
```

### 1.2 Test Components

```bash
# Test environment wrapper
python src/envs/nonstationary_cartpole.py

# Test abstraction bank (with all improvements)
python src/models/abstraction_bank.py

# Test losses
python src/losses.py

# Test full agent
python src/models/astral_agent.py
```

---

## 2. Core Validation

### 2.1 Train ASTRAL (Full Run)

```bash
# Standard ASTRAL training (500k steps, ~10 min on CPU)
python src/train.py \
    --exp_name astral_full \
    --use_abstractions True \
    --total_timesteps 500000 \
    --seed 42

# Multiple seeds for statistical significance
for seed in 42 123 456 789 1000; do
    python src/train.py \
        --exp_name astral_seed${seed} \
        --use_abstractions True \
        --total_timesteps 500000 \
        --seed $seed
done
```

### 2.2 Train Baseline (GRU-only)

```bash
# Baseline training (no abstraction bank)
python src/train.py \
    --exp_name baseline_full \
    --use_abstractions False \
    --total_timesteps 500000 \
    --seed 42

# Multiple seeds
for seed in 42 123 456 789 1000; do
    python src/train.py \
        --exp_name baseline_seed${seed} \
        --use_abstractions False \
        --total_timesteps 500000 \
        --seed $seed
done
```

### 2.3 Quick Validation (Sanity Check)

```bash
# Fast training to verify everything works (50k steps, ~1 min)
python src/train.py \
    --exp_name quick_test \
    --total_timesteps 50000 \
    --log_interval 5 \
    --seed 42
```

---

## 3. Test-Time Adaptation

### 3.1 Basic TTA Experiment

```bash
# Find your trained model checkpoint
ls results/runs/

# Run TTA experiment (replace <run_name> with actual folder)
python src/test_time_adapt.py \
    --checkpoint results/runs/<run_name>/final_model.pt \
    --num_adapt_episodes 30 \
    --num_eval_episodes 20

# Example with specific run:
python src/test_time_adapt.py \
    --checkpoint results/runs/astral_astral_42_1764466077/final_model.pt \
    --num_adapt_episodes 30 \
    --num_eval_episodes 20
```

### 3.2 TTA with Different Episode Counts

```bash
# Short adaptation (10 episodes)
python src/test_time_adapt.py \
    --checkpoint results/runs/<run_name>/final_model.pt \
    --num_adapt_episodes 10

# Medium adaptation (30 episodes)
python src/test_time_adapt.py \
    --checkpoint results/runs/<run_name>/final_model.pt \
    --num_adapt_episodes 30

# Long adaptation (50 episodes)
python src/test_time_adapt.py \
    --checkpoint results/runs/<run_name>/final_model.pt \
    --num_adapt_episodes 50
```

### 3.3 TTA Comparison Script

```bash
# Compare TTA across different training runs
for run in results/runs/astral_*; do
    echo "Testing: $run"
    python src/test_time_adapt.py \
        --checkpoint "$run/final_model.pt" \
        --num_adapt_episodes 20 \
        --num_eval_episodes 15
done
```

---

## 4. Causal Interventions

### 4.1 Full Intervention Experiment

```bash
# Run slot clamping and disabling experiments
python src/interventions.py \
    --checkpoint results/runs/<run_name>/final_model.pt \
    --num_episodes 20

# Example:
python src/interventions.py \
    --checkpoint results/runs/astral_astral_42_1764466077/final_model.pt \
    --num_episodes 20
```

### 4.2 Detailed Interventions

```bash
# More episodes for statistical significance
python src/interventions.py \
    --checkpoint results/runs/<run_name>/final_model.pt \
    --num_episodes 50
```

---

## 5. Interpretability Experiments

These experiments test whether the interpretability improvements help address slot collapse.

### 5.1 Gumbel-Softmax Only

```bash
python src/train.py \
    --exp_name gumbel_only \
    --use_gumbel True \
    --total_timesteps 200000 \
    --seed 42
```

### 5.2 Hard Routing Only

```bash
python src/train.py \
    --exp_name hard_routing_only \
    --hard_routing True \
    --total_timesteps 200000 \
    --seed 42
```

### 5.3 Gumbel + Hard Routing

```bash
python src/train.py \
    --exp_name gumbel_hard \
    --use_gumbel True \
    --hard_routing True \
    --total_timesteps 200000 \
    --seed 42
```

### 5.4 Orthogonal Initialization Only

```bash
python src/train.py \
    --exp_name orthogonal_init \
    --orthogonal_init True \
    --total_timesteps 200000 \
    --seed 42
```

### 5.5 Temperature Annealing Only

```bash
# Standard annealing (5.0 → 0.5)
python src/train.py \
    --exp_name temp_anneal_std \
    --temp_anneal True \
    --tau_start 5.0 \
    --tau_end 0.5 \
    --total_timesteps 200000 \
    --seed 42

# Aggressive annealing (10.0 → 0.1)
python src/train.py \
    --exp_name temp_anneal_aggressive \
    --temp_anneal True \
    --tau_start 10.0 \
    --tau_end 0.1 \
    --total_timesteps 200000 \
    --seed 42
```

### 5.6 Contrastive Loss Only

```bash
# Weak contrastive (λ=0.01)
python src/train.py \
    --exp_name contrast_weak \
    --lambda_contrast 0.01 \
    --total_timesteps 200000 \
    --seed 42

# Strong contrastive (λ=0.1)
python src/train.py \
    --exp_name contrast_strong \
    --lambda_contrast 0.1 \
    --total_timesteps 200000 \
    --seed 42
```

### 5.7 Slot Prediction Only

```bash
python src/train.py \
    --exp_name slot_pred \
    --slot_prediction True \
    --lambda_slot_pred 0.01 \
    --total_timesteps 200000 \
    --seed 42
```

### 5.8 All Improvements Combined

```bash
python src/train.py \
    --exp_name all_improvements \
    --use_gumbel True \
    --hard_routing True \
    --orthogonal_init True \
    --temp_anneal True \
    --tau_start 5.0 \
    --tau_end 0.5 \
    --lambda_contrast 0.01 \
    --slot_prediction True \
    --total_timesteps 200000 \
    --seed 42
```

### 5.9 Recommended Combinations

```bash
# Combo 1: Gumbel + Temp Anneal + Contrastive
python src/train.py \
    --exp_name combo1_gumbel_anneal_contrast \
    --use_gumbel True \
    --temp_anneal True \
    --lambda_contrast 0.05 \
    --total_timesteps 200000 \
    --seed 42

# Combo 2: Orthogonal + Strong Regularization
python src/train.py \
    --exp_name combo2_orth_strong_reg \
    --orthogonal_init True \
    --lambda_w_ent 0.05 \
    --lambda_lb 0.05 \
    --lambda_orth 0.01 \
    --total_timesteps 200000 \
    --seed 42

# Combo 3: Hard Routing + Contrastive + Slot Pred
python src/train.py \
    --exp_name combo3_hard_contrast_pred \
    --hard_routing True \
    --lambda_contrast 0.05 \
    --slot_prediction True \
    --total_timesteps 200000 \
    --seed 42
```

---

## 6. Ablation Studies

### 6.1 Regularization Strength Sweep

```bash
# No regularization
python src/train.py \
    --exp_name reg_none \
    --lambda_w_ent 0 \
    --lambda_lb 0 \
    --lambda_orth 0 \
    --total_timesteps 200000 \
    --seed 42

# Weak regularization (default)
python src/train.py \
    --exp_name reg_weak \
    --lambda_w_ent 0.001 \
    --lambda_lb 0.001 \
    --lambda_orth 0.0001 \
    --total_timesteps 200000 \
    --seed 42

# Medium regularization
python src/train.py \
    --exp_name reg_medium \
    --lambda_w_ent 0.01 \
    --lambda_lb 0.01 \
    --lambda_orth 0.001 \
    --total_timesteps 200000 \
    --seed 42

# Strong regularization
python src/train.py \
    --exp_name reg_strong \
    --lambda_w_ent 0.1 \
    --lambda_lb 0.1 \
    --lambda_orth 0.01 \
    --total_timesteps 200000 \
    --seed 42
```

### 6.2 Temperature Sweep (Fixed)

```bash
# Very cold (τ=0.1)
python src/train.py \
    --exp_name tau_0.1 \
    --tau 0.1 \
    --total_timesteps 200000 \
    --seed 42

# Cold (τ=0.5)
python src/train.py \
    --exp_name tau_0.5 \
    --tau 0.5 \
    --total_timesteps 200000 \
    --seed 42

# Standard (τ=1.0)
python src/train.py \
    --exp_name tau_1.0 \
    --tau 1.0 \
    --total_timesteps 200000 \
    --seed 42

# Hot (τ=5.0)
python src/train.py \
    --exp_name tau_5.0 \
    --tau 5.0 \
    --total_timesteps 200000 \
    --seed 42

# Very hot (τ=10.0)
python src/train.py \
    --exp_name tau_10.0 \
    --tau 10.0 \
    --total_timesteps 200000 \
    --seed 42
```

### 6.3 Number of Abstraction Slots

```bash
# 2 slots
python src/train.py \
    --exp_name slots_2 \
    --num_abstractions 2 \
    --total_timesteps 200000 \
    --seed 42

# 3 slots (default, matches 3 modes)
python src/train.py \
    --exp_name slots_3 \
    --num_abstractions 3 \
    --total_timesteps 200000 \
    --seed 42

# 5 slots (over-provisioned)
python src/train.py \
    --exp_name slots_5 \
    --num_abstractions 5 \
    --total_timesteps 200000 \
    --seed 42

# 8 slots (highly over-provisioned)
python src/train.py \
    --exp_name slots_8 \
    --num_abstractions 8 \
    --total_timesteps 200000 \
    --seed 42
```

### 6.4 Hidden Dimension Size

```bash
# Small (d=32)
python src/train.py \
    --exp_name dmodel_32 \
    --d_model 32 \
    --total_timesteps 200000 \
    --seed 42

# Medium (d=64, default)
python src/train.py \
    --exp_name dmodel_64 \
    --d_model 64 \
    --total_timesteps 200000 \
    --seed 42

# Large (d=128)
python src/train.py \
    --exp_name dmodel_128 \
    --d_model 128 \
    --total_timesteps 200000 \
    --seed 42
```

### 6.5 Number of Environments

```bash
# Few envs (slower but more stable)
python src/train.py \
    --exp_name envs_4 \
    --num_envs 4 \
    --total_timesteps 200000 \
    --seed 42

# Standard (8 envs)
python src/train.py \
    --exp_name envs_8 \
    --num_envs 8 \
    --total_timesteps 200000 \
    --seed 42

# Many envs (faster but noisier)
python src/train.py \
    --exp_name envs_16 \
    --num_envs 16 \
    --total_timesteps 200000 \
    --seed 42
```

---

## 7. Analysis & Visualization

### 7.1 TensorBoard

```bash
# View all runs
tensorboard --logdir results/runs

# View specific experiment
tensorboard --logdir results/runs/astral_full_*

# Compare ASTRAL vs Baseline
tensorboard --logdir results/runs --tag_filter "mean_return"
```

### 7.2 View Intervention Results

```bash
# Check generated plots
ls results/interventions/

# Open plots (macOS)
open results/interventions/*.png
```

### 7.3 View TTA Results

```bash
# Check TTA output
ls results/tta/

# Open plots (macOS)
open results/tta/*.png
```

---

## 8. Full Experiment Suite (Automated)

All automation scripts are located in `scripts/` and are ready to run.

### 8.0 Running the Automated Scripts

#### Option A: Run Everything (2-3 hours)

```bash
cd astral-rl
source venv/bin/activate

# Run ALL experiments automatically
./scripts/run_all.sh
```

This executes in order:
1. **Core experiments** — ASTRAL + Baseline × 3 seeds (~1.5 hours)
2. **Interpretability experiments** — 7 configurations (~40 min)
3. **TTA tests** — On all trained models (~10 min)
4. **Causal interventions** — On all trained models (~10 min)

#### Option B: Run Each Part Separately

```bash
# 1. Core: ASTRAL vs Baseline (3 seeds each)
./scripts/run_core_experiments.sh 500000

# 2. Interpretability improvements
./scripts/run_interpretability_experiments.sh 200000

# 3. Test-time adaptation
./scripts/run_tta_tests.sh 20 15

# 4. Causal interventions
./scripts/run_interventions.sh 20
```

#### Option C: Run in Background (Overnight)

```bash
# Run all experiments in background, log to file
nohup ./scripts/run_all.sh > experiment_log.txt 2>&1 &

# Check progress anytime
tail -f experiment_log.txt

# Check if still running
ps aux | grep run_all
```

---

### 8.1 Script Details: run_core_experiments.sh

```bash
#!/bin/bash
# scripts/run_core_experiments.sh

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Running Core Experiments ==="

# 1. ASTRAL vs Baseline (3 seeds)
for seed in 42 123 456; do
    echo "Training ASTRAL (seed=$seed)..."
    python src/train.py --exp_name astral --use_abstractions True \
        --total_timesteps 500000 --seed $seed
    
    echo "Training Baseline (seed=$seed)..."
    python src/train.py --exp_name baseline --use_abstractions False \
        --total_timesteps 500000 --seed $seed
done

# 2. Interpretability experiments
echo "Running interpretability experiments..."

python src/train.py --exp_name gumbel --use_gumbel True \
    --total_timesteps 200000 --seed 42

python src/train.py --exp_name temp_anneal --temp_anneal True \
    --total_timesteps 200000 --seed 42

python src/train.py --exp_name contrastive --lambda_contrast 0.05 \
    --total_timesteps 200000 --seed 42

python src/train.py --exp_name all_improve \
    --use_gumbel True --hard_routing True --orthogonal_init True \
    --temp_anneal True --lambda_contrast 0.01 --slot_prediction True \
    --total_timesteps 200000 --seed 42

echo "=== Experiments Complete ==="
```

### 8.2 Script Details: run_tta_tests.sh

```bash
#!/bin/bash
# scripts/run_tta_tests.sh

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Running TTA Experiments ==="

for run in results/runs/astral_*; do
    if [ -f "$run/final_model.pt" ]; then
        echo "Testing: $run"
        python src/test_time_adapt.py \
            --checkpoint "$run/final_model.pt" \
            --num_adapt_episodes 20 \
            --num_eval_episodes 15
    fi
done

echo "=== TTA Tests Complete ==="
```

### 8.3 Script Details: run_interventions.sh

```bash
#!/bin/bash
# scripts/run_interventions.sh

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Running Intervention Experiments ==="

for run in results/runs/astral_*; do
    if [ -f "$run/final_model.pt" ]; then
        echo "Interventions: $run"
        python src/interventions.py \
            --checkpoint "$run/final_model.pt" \
            --num_episodes 20
    fi
done

echo "=== Interventions Complete ==="
```

---

## 9. Expected Results

### 9.1 Training Performance

| Agent | Mode 0 | Mode 1 | Mode 2 | Avg |
|:------|:-------|:-------|:-------|:----|
| Baseline (GRU) | ~100-120 | ~140-160 | ~80-100 | ~110-130 |
| ASTRAL | ~100-130 | ~150-180 | ~90-120 | ~115-140 |

### 9.2 Test-Time Adaptation

| Condition | Before TTA | After TTA (20 eps) | Improvement |
|:----------|:-----------|:-------------------|:------------|
| Mode 0 | ~110 | ~120 | +10% |
| Mode 1 | ~150 | ~160 | +7% |
| Mode 2 | ~100 | ~105 | +5% |

### 9.3 Slot Usage (Expected with Improvements)

| Improvement | Slot 0 | Slot 1 | Slot 2 | Notes |
|:------------|:-------|:-------|:-------|:------|
| None (default) | ~0% | ~100% | ~0% | Full collapse |
| Temp Anneal | ~15% | ~70% | ~15% | Slight improvement |
| Contrastive | ~25% | ~50% | ~25% | Better balance |
| All combined | ~30% | ~40% | ~30% | Best balance |

---

## 10. Troubleshooting

### 10.1 Out of Memory

```bash
# Reduce batch sizes
python src/train.py --num_envs 4 --minibatch_size 128 --total_timesteps 200000
```

### 10.2 Slow Training

```bash
# Use fewer environments but more steps per rollout
python src/train.py --num_envs 4 --num_steps 256 --total_timesteps 200000
```

### 10.3 Missing Dependencies

```bash
# Install missing packages
pip install matplotlib seaborn pandas
```

### 10.4 Check GPU Usage

```bash
# Verify CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Quick Reference Card

```bash
# Setup
source cleanrl/venv/bin/activate

# Train ASTRAL
python src/train.py --total_timesteps 500000

# Train Baseline
python src/train.py --use_abstractions False --total_timesteps 500000

# TTA Test
python src/test_time_adapt.py --checkpoint results/runs/<name>/final_model.pt

# Interventions
python src/interventions.py --checkpoint results/runs/<name>/final_model.pt

# All improvements
python src/train.py --use_gumbel True --hard_routing True --orthogonal_init True \
    --temp_anneal True --lambda_contrast 0.01 --slot_prediction True

# View logs
tensorboard --logdir results/runs
```

