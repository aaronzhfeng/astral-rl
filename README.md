# ASTRAL: Abstraction-Slot Test-time Reweighting for Adaptation in Latent RL

A small-scale proof-of-concept for **structured abstractions in in-context RL**.

ASTRAL tests whether discrete, learnable "abstraction slots" can provide interpretable, mode-specific adaptation in non-stationary environments, with efficient test-time adaptation by updating only the gating network.

---

## 🎯 Research Question

> Can we add structured abstractions to in-context RL that are:
> 1. **Interpretable** — different modes activate different abstraction slots
> 2. **Efficient** — test-time adaptation requires updating only the gating network
> 3. **Causal** — clamping/disabling slots produces predictable behavioral changes

---

## 🏗️ Architecture

```
Input: (s_t, a_{t-1}, r_{t-1})
           │
           ▼
    ┌─────────────┐
    │  Input MLP  │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    GRU      │  ← In-context adaptation via hidden state
    └──────┬──────┘
           │
           ▼
         h_t (context embedding)
           │
           ├──────────────────────────┐
           │                          │
           ▼                          ▼
    ┌─────────────┐          ┌────────────────┐
    │ Gating MLP  │          │  Abstraction   │
    │   g(h_t)    │          │  Bank A [K×d]  │
    └──────┬──────┘          └───────┬────────┘
           │                         │
           ▼                         │
    w_t = softmax(logits/τ)          │
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
              z_t = w_t^T · A  (combined abstraction)
                        │
                        ▼
                 ┌─────────────┐
                 │    FiLM     │  ← Forces dependency on abstraction
                 │  γ, β = f(z)│
                 └──────┬──────┘
                        │
                        ▼
              h'_t = γ ⊙ h_t + β  (modulated context)
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
     ┌─────────────┐         ┌─────────────┐
     │ Policy Head │         │ Value Head  │
     └─────────────┘         └─────────────┘
```

**Key Design Decisions:**
- **FiLM modulation** ensures the policy depends on abstractions (no bypass)
- **Soft attention** over K slots allows gradient-based learning
- **Test-time adaptation** updates only the gating network (4K parameters)

---

## 📁 Project Structure

```
test_time_RL/
├── README.md                    # This file
├── astral_implementation_plan.md # Detailed implementation guide
├── astral_proposal.md           # Research proposal
│
├── src/
│   ├── envs/
│   │   └── nonstationary_cartpole.py  # 3-mode CartPole environment
│   │
│   ├── models/
│   │   ├── abstraction_bank.py   # K learnable slots + gating
│   │   ├── film.py               # Feature-wise Linear Modulation
│   │   └── astral_agent.py       # Full agent + baseline
│   │
│   ├── losses.py                 # Regularization losses
│   ├── train.py                  # PPO training loop
│   ├── test_time_adapt.py        # TTA experiments
│   └── interventions.py          # Causal intervention experiments
│
├── cleanrl/                      # CleanRL reference (cloned)
│   └── venv/                     # Python virtual environment
│
├── results/
│   ├── runs/                     # Training runs + checkpoints
│   ├── tta/                      # TTA experiment results
│   └── interventions/            # Intervention experiment results
│
└── configs/                      # (Optional) Config files
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone and enter directory
cd test_time_RL

# Activate virtual environment
source cleanrl/venv/bin/activate

# Verify installation
python -c "import torch; import gymnasium; print('Ready!')"
```

### 2. Train ASTRAL

```bash
# Train ASTRAL agent (500k timesteps, ~10 min on CPU)
python src/train.py --total_timesteps 500000

# Train baseline (GRU-only, for comparison)
python src/train.py --use_abstractions False --exp_name baseline
```

### 3. Test-Time Adaptation

```bash
# Run TTA experiment on all modes
python src/test_time_adapt.py \
    --checkpoint results/runs/<run_name>/final_model.pt \
    --num_adapt_episodes 30
```

### 4. Causal Interventions

```bash
# Run clamping/disabling experiments
python src/interventions.py \
    --checkpoint results/runs/<run_name>/final_model.pt \
    --num_episodes 20
```

### 5. View Logs

```bash
tensorboard --logdir results/runs
```

---

## 🧪 Environment: NonStationaryCartPole

A CartPole variant with 3 hidden "modes" that change physical dynamics:

| Mode | Gravity | Pole Length | Difficulty |
|:-----|:--------|:------------|:-----------|
| 0 | 9.8 | 0.5 | Default |
| 1 | 7.5 | 0.7 | Easy (slower, longer) |
| 2 | 12.0 | 0.4 | Hard (faster, shorter) |

The agent does **not** observe the mode — it must infer it from dynamics.

---

## 📊 Key Results

### Training
- Both ASTRAL and baseline learn CartPole (~100-150 return)
- Mode 1 (easy) performs best, Mode 2 (hard) worst

### Test-Time Adaptation
- TTA improves Mode 0 by **+10.4%** with only 20 episodes
- Updates only **4,355 parameters** (8.5% of model)

### Slot Collapse (Known Issue)
All modes collapse to using **Slot 1** (~99.99%). This limits interpretability.

**Causal Evidence:**
- Clamping to Slot 1: Best performance
- Clamping to Slot 0/2: Severe drop (-75 to -115 points)
- Disabling Slot 1: Catastrophic failure

See `docs/interpretability_improvements.md` for solutions.

---

## 🔧 Configuration

Key hyperparameters in `src/train.py`:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `d_model` | 64 | Hidden dimension |
| `num_abstractions` | 3 | Number of slots (K) |
| `tau` | 1.0 | Softmax temperature |
| `learning_rate` | 3e-4 | PPO learning rate |
| `lambda_w_ent` | 0.001 | Weight entropy regularization |
| `lambda_lb` | 0.001 | Load balancing regularization |
| `lambda_orth` | 0.0001 | Orthogonality regularization |

### Interpretability Improvements (Optional)

All improvements are modular and disabled by default:

| Flag | Description |
|:-----|:------------|
| `--use_gumbel True` | Gumbel-Softmax for slot exploration |
| `--hard_routing True` | Discrete one-hot slot selection |
| `--orthogonal_init True` | Initialize slots orthogonally |
| `--temp_anneal True` | Anneal temperature from high to low |
| `--tau_start 5.0` | Starting temperature (if annealing) |
| `--tau_end 0.5` | Ending temperature (if annealing) |
| `--lambda_contrast 0.01` | Contrastive loss (mode→slot) |
| `--slot_prediction True` | Auxiliary slot prediction task |

---

## 📈 Experiments

### 1. Baseline Comparison
```bash
# Train both
python src/train.py --use_abstractions True --exp_name astral
python src/train.py --use_abstractions False --exp_name baseline

# Compare in tensorboard
tensorboard --logdir results/runs
```

### 2. Regularization Ablation
```bash
# Stronger regularization
python src/train.py --lambda_w_ent 0.01 --lambda_lb 0.01 --lambda_orth 0.001

# No regularization
python src/train.py --lambda_w_ent 0 --lambda_lb 0 --lambda_orth 0
```

### 3. Temperature Sweep
```bash
# Cold (peaked weights)
python src/train.py --tau 0.1

# Hot (uniform weights)
python src/train.py --tau 10.0
```

### 4. Interpretability Improvements
```bash
# All improvements (recommended for addressing slot collapse)
python src/train.py \
    --use_gumbel True \
    --hard_routing True \
    --orthogonal_init True \
    --temp_anneal True \
    --lambda_contrast 0.01 \
    --slot_prediction True

# Just temperature annealing
python src/train.py --temp_anneal True --tau_start 5.0 --tau_end 0.5

# Contrastive loss only
python src/train.py --lambda_contrast 0.05
```

---

## 📚 Documentation

| Document | Description |
|:---------|:------------|
| `CONTEXT.md` | **Quick onboarding for new environments** |
| `docs/baseline_vs_astral.md` | Train vs Test-time differences |
| `docs/abstraction_bank_vs_moe.md` | Comparison with Mixture of Experts |
| `docs/experiment_guide.md` | Complete commands for all experiments |
| `docs/interpretability_improvements.md` | Solutions for slot collapse |

---

## 🔬 Future Work

1. **Fix Slot Collapse** — See `docs/interpretability_improvements.md`
2. **Scale to TAG-AMAGO** — Transformer backbone, MuJoCo/Meta-World
3. **Mode-Conditioned Auxiliary Loss** — Encourage mode→slot correspondence
4. **Continual Learning** — Test on sequentially changing modes

---

## 📚 References

- **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
- **AMAGO**: Grigsby et al., "AMAGO: Scalable In-Context Reinforcement Learning"
- **CleanRL**: Huang et al., "CleanRL: High-quality Single-file Implementations of Deep RL Algorithms"

---

## 📝 Citation

```bibtex
@misc{astral2024,
  title={ASTRAL: Abstraction-Slot Test-time Reweighting for Adaptation in Latent RL},
  author={...},
  year={2024},
  note={Proof-of-concept implementation}
}
```

---

## 📄 License

MIT License

