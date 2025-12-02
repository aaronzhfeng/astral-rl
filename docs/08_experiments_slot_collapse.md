# Proposed Experiments: Addressing Slot Collapse

Based on the findings from the initial experiments, this document outlines new experiments designed to:
1. **Prevent slot collapse** — Force the model to use multiple slots
2. **Achieve mode-slot correspondence** — Each mode should prefer a different slot
3. **Maintain performance** — Don't sacrifice too much return for interpretability

---

## Experiment 1: Stronger Diversity Pressure

### Hypothesis
The current regularization coefficients (λ=0.01-0.05) are too weak to overcome the policy gradient's tendency to reinforce a single slot.

### Experiments

```bash
# 1A: Strong contrastive loss
python src/train.py \
    --exp_name strong_contrast_0.1 \
    --lambda_contrast 0.1 \
    --total_timesteps 300000 \
    --seed 42

python src/train.py \
    --exp_name strong_contrast_0.2 \
    --lambda_contrast 0.2 \
    --total_timesteps 300000 \
    --seed 42

python src/train.py \
    --exp_name strong_contrast_0.5 \
    --lambda_contrast 0.5 \
    --total_timesteps 300000 \
    --seed 42

# 1B: Strong load balancing
python src/train.py \
    --exp_name strong_lb_0.05 \
    --lambda_lb 0.05 \
    --total_timesteps 300000 \
    --seed 42

python src/train.py \
    --exp_name strong_lb_0.1 \
    --lambda_lb 0.1 \
    --total_timesteps 300000 \
    --seed 42

# 1C: Strong weight entropy
python src/train.py \
    --exp_name strong_w_ent_0.05 \
    --lambda_w_ent 0.05 \
    --total_timesteps 300000 \
    --seed 42

python src/train.py \
    --exp_name strong_w_ent_0.1 \
    --lambda_w_ent 0.1 \
    --total_timesteps 300000 \
    --seed 42

# 1D: Combined strong regularization
python src/train.py \
    --exp_name strong_all_reg \
    --lambda_contrast 0.1 \
    --lambda_lb 0.05 \
    --lambda_w_ent 0.05 \
    --lambda_orth 0.01 \
    --total_timesteps 300000 \
    --seed 42
```

### Success Criteria
- Slot weight entropy > 0.8 (vs ~0.1 for collapsed models)
- All 3 slots receive >10% average weight
- TTA improvement maintained or improved

---

## Experiment 2: Curriculum Learning

### Hypothesis
Training on mixed modes from the start allows the model to find a "universal" solution. Training on single modes first forces slot specialization.

### Experiments

**Requires code modification:** Add `--single_mode` flag to train on one mode only.

```bash
# 2A: Pre-train on each mode separately, then fine-tune on mixed
# Phase 1: Single-mode pre-training (100k steps each)
python src/train.py \
    --exp_name pretrain_mode0 \
    --single_mode 0 \
    --total_timesteps 100000 \
    --seed 42

python src/train.py \
    --exp_name pretrain_mode1 \
    --single_mode 1 \
    --total_timesteps 100000 \
    --seed 42

python src/train.py \
    --exp_name pretrain_mode2 \
    --single_mode 2 \
    --total_timesteps 100000 \
    --seed 42

# Phase 2: Load best single-mode checkpoint and fine-tune on mixed
python src/train.py \
    --exp_name curriculum_finetune \
    --load_checkpoint results/runs/pretrain_mode1_*/final_model.pt \
    --total_timesteps 200000 \
    --seed 42

# 2B: Progressive mode mixing
# Start with 1 mode, add modes over time
python src/train.py \
    --exp_name progressive_modes \
    --progressive_mode_schedule "0:100000,0-1:200000,0-1-2:300000" \
    --total_timesteps 300000 \
    --seed 42
```

### Success Criteria
- Each slot becomes "expert" for one mode
- Clamping to matching slot improves performance
- Clear mode-slot mapping in weight distributions

---

## Experiment 3: Hard Slot Assignment

### Hypothesis
Soft attention (softmax) naturally converges to near-one-hot. Using truly discrete slot selection from the start may prevent collapse while maintaining differentiability.

### Experiments

**Requires code modification:** Implement top-k routing and slot dropout.

```bash
# 3A: Top-1 routing with straight-through gradient
python src/train.py \
    --exp_name topk_1 \
    --routing_mode topk \
    --topk 1 \
    --total_timesteps 300000 \
    --seed 42

# 3B: Top-2 routing (use 2 slots per forward pass)
python src/train.py \
    --exp_name topk_2 \
    --routing_mode topk \
    --topk 2 \
    --total_timesteps 300000 \
    --seed 42

# 3C: Slot dropout (randomly mask slots during training)
python src/train.py \
    --exp_name slot_dropout_0.3 \
    --slot_dropout 0.3 \
    --total_timesteps 300000 \
    --seed 42

python src/train.py \
    --exp_name slot_dropout_0.5 \
    --slot_dropout 0.5 \
    --total_timesteps 300000 \
    --seed 42
```

### Success Criteria
- Each slot used roughly equally over training
- Performance maintained despite discrete routing
- Clear slot selection visible in weight histograms

---

## Experiment 4: Mode-Conditioned Supervision

### Hypothesis
The model doesn't know which mode it's in. Providing explicit mode supervision during training may help slots specialize.

### Experiments

**Requires code modification:** Add mode prediction auxiliary task.

```bash
# 4A: Mode classification head (predict mode from hidden state)
python src/train.py \
    --exp_name mode_classify \
    --mode_classification True \
    --lambda_mode_cls 0.1 \
    --total_timesteps 300000 \
    --seed 42

# 4B: Mode-slot alignment loss (encourage slot i to activate for mode i)
python src/train.py \
    --exp_name mode_slot_align \
    --mode_slot_alignment True \
    --lambda_align 0.1 \
    --total_timesteps 300000 \
    --seed 42

# 4C: Slot-specific mode prediction (each slot predicts a different mode)
python src/train.py \
    --exp_name slot_mode_pred \
    --slot_mode_prediction True \
    --lambda_slot_mode 0.05 \
    --total_timesteps 300000 \
    --seed 42
```

### Success Criteria
- Mode classification accuracy > 90%
- Slot weights correlate with true mode
- Clamping to slot i improves mode i performance

---

## Experiment 5: Architecture Modifications

### Hypothesis
The current architecture may make collapse too easy. Structural changes can force diversity.

### Experiments

**Requires code modification:** Modify AbstractionBank architecture.

```bash
# 5A: Larger abstraction dimension (harder to compress into one slot)
python src/train.py \
    --exp_name large_abstraction_128 \
    --abstraction_dim 128 \
    --total_timesteps 300000 \
    --seed 42

# 5B: More slots than modes (over-provisioning)
python src/train.py \
    --exp_name slots_6 \
    --num_abstractions 6 \
    --total_timesteps 300000 \
    --seed 42

python src/train.py \
    --exp_name slots_9 \
    --num_abstractions 9 \
    --total_timesteps 300000 \
    --seed 42

# 5C: Separate gating networks (diagnostic - not deployable)
python src/train.py \
    --exp_name separate_gating \
    --separate_gating_per_mode True \
    --total_timesteps 300000 \
    --seed 42

# 5D: Mixture of Experts style (sparse gating)
python src/train.py \
    --exp_name moe_sparse \
    --moe_style True \
    --moe_capacity_factor 1.25 \
    --total_timesteps 300000 \
    --seed 42
```

### Success Criteria
- Multiple slots receive significant weight
- Performance comparable to collapsed models
- Clear specialization patterns emerge

---

## Experiment 6: Environment Modifications

### Hypothesis
The 3 modes in NonStationaryCartPole may be too similar, allowing a single solution to work for all.

### Experiments

**Requires code modification:** Modify environment parameters.

```bash
# 6A: More extreme mode differences
python src/train.py \
    --exp_name extreme_modes \
    --mode_gravity_range "5.0,15.0,25.0" \
    --mode_length_range "0.3,0.5,0.8" \
    --total_timesteps 300000 \
    --seed 42

# 6B: More modes (5 modes)
python src/train.py \
    --exp_name modes_5 \
    --num_modes 5 \
    --num_abstractions 5 \
    --total_timesteps 500000 \
    --seed 42

# 6C: Adversarial mode switching (switch when doing well)
python src/train.py \
    --exp_name adversarial_switch \
    --adversarial_mode_switch True \
    --switch_threshold 100 \
    --total_timesteps 300000 \
    --seed 42
```

### Success Criteria
- Model cannot achieve high performance with single slot
- Forced to develop mode-specific strategies
- TTA shows larger improvements

---

## Experiment 7: Two-Phase Training

### Hypothesis
Learning both the policy and slot specialization simultaneously is too hard. Separating these objectives may help.

### Experiments

```bash
# 7A: Phase 1 - Learn policy with frozen uniform weights
python src/train.py \
    --exp_name phase1_frozen_gating \
    --freeze_gating True \
    --uniform_weights True \
    --total_timesteps 200000 \
    --seed 42

# 7A: Phase 2 - Unfreeze gating, freeze policy, learn specialization
python src/train.py \
    --exp_name phase2_learn_gating \
    --load_checkpoint results/runs/phase1_frozen_gating_*/final_model.pt \
    --freeze_policy True \
    --lambda_contrast 0.2 \
    --total_timesteps 100000 \
    --seed 42

# 7B: Alternating training (switch what's frozen every N steps)
python src/train.py \
    --exp_name alternating_train \
    --alternating_freeze True \
    --alternate_every 10000 \
    --total_timesteps 300000 \
    --seed 42
```

### Success Criteria
- Phase 2 develops slot specialization
- Final model has both good policy and diverse slots
- TTA works effectively

---

## Implementation Priority

Based on effort vs. expected impact:

### Quick Wins (No Code Changes)
1. **Experiment 1A-D**: Stronger regularization coefficients
2. Run with multiple seeds for statistical significance

### Medium Effort (Minor Code Changes)
3. **Experiment 3C**: Slot dropout
4. **Experiment 4A**: Mode classification head
5. **Experiment 7A-B**: Two-phase training

### Higher Effort (Significant Code Changes)
6. **Experiment 2**: Curriculum learning
7. **Experiment 3A-B**: Top-k routing
8. **Experiment 5D**: MoE-style sparse gating
9. **Experiment 6**: Environment modifications

---

## Quick Start: Run Priority Experiments

```bash
#!/bin/bash
# run_priority_experiments.sh

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Priority Experiments to Address Slot Collapse ==="

# 1. Strong regularization sweep
for lambda in 0.1 0.2 0.5; do
    echo "Running: strong_contrast_${lambda}"
    python src/train.py \
        --exp_name strong_contrast_${lambda} \
        --lambda_contrast $lambda \
        --total_timesteps 300000 \
        --seed 42
done

# 2. Combined strong regularization
echo "Running: strong_all_reg"
python src/train.py \
    --exp_name strong_all_reg \
    --lambda_contrast 0.1 \
    --lambda_lb 0.05 \
    --lambda_w_ent 0.05 \
    --lambda_orth 0.01 \
    --total_timesteps 300000 \
    --seed 42

# 3. Best config with multiple seeds
for seed in 42 123 456; do
    echo "Running: best_config_seed${seed}"
    python src/train.py \
        --exp_name best_config \
        --use_gumbel True \
        --temp_anneal True \
        --lambda_contrast 0.1 \
        --slot_prediction True \
        --total_timesteps 300000 \
        --seed $seed
done

echo "=== Priority Experiments Complete ==="
echo "Run TTA and interventions on new models to evaluate"
```

---

## Expected Outcomes

| Experiment | Expected Effect | Risk |
|------------|-----------------|------|
| Strong contrastive (λ=0.1+) | ↑ Slot diversity | ↓ Performance |
| Load balancing (λ=0.05+) | ↑ Slot usage balance | ↓ Specialization |
| Slot dropout | ↑ Robustness | ↑ Training variance |
| Mode classification | ↑ Mode awareness | Minimal risk |
| Two-phase training | ↑ Both performance & diversity | ↑ Complexity |
| Curriculum learning | ↑ Mode-slot correspondence | ↑ Training time |

---

## Success Metrics

A successful experiment should achieve:

1. **Slot Diversity Score** > 0.7
   ```
   diversity = -sum(w * log(w)) / log(num_slots)  # Normalized entropy
   ```

2. **Mode-Slot Correspondence** > 0.5
   ```
   correspondence = max(correlation(mode_i, slot_j)) for each mode
   ```

3. **TTA Improvement** > 20% average across modes

4. **Performance Retention** > 70% of collapsed model's return

---

## Next Steps After Experiments

1. **If strong regularization works:** Tune coefficients for best tradeoff
2. **If curriculum learning works:** Develop automated curriculum
3. **If architecture changes work:** Consider for ASTRAL v2
4. **If nothing works:** Reconsider fundamental approach (maybe collapse is optimal?)

