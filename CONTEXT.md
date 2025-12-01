# ASTRAL Project Context

**Quick onboarding document for new environments or AI assistants.**

---

## What Is This Project?

**ASTRAL** = Abstraction-Slot Test-time Reweighting for Adaptation in Latent RL

A proof-of-concept for **interpretable test-time adaptation** in reinforcement learning.

### The Problem
Standard in-context RL agents (like GRU-based policies) adapt implicitly via hidden states. This is:
- **Opaque**: Can't see what strategy the agent is using
- **Inflexible**: At test-time, either update all parameters (risky) or none

### The Solution
ASTRAL adds:
1. **Abstraction Bank**: K learnable "strategy vectors" (slots)
2. **Gating Network**: Selects which slots to use based on context
3. **FiLM Modulation**: Forces policy to depend on selected slots
4. **Test-Time RL**: Update ONLY the gating network (2K params vs 50K total)

---

## Architecture (ASCII)

```
Input → GRU → h_t → [Gating] → weights → [Bank] → z_t → [FiLM] → h'_t → Policy/Value
                         ↑                   ↑
                    (updatable at          (K slots)
                     test-time)
```

---

## Benchmark

**NonStationaryCartPole**: CartPole with 3 hidden modes (agent can't observe mode)

| Mode | Gravity | Pole Length | Difficulty |
|:-----|:--------|:------------|:-----------|
| 0 | 9.8 | 0.5 | Default |
| 1 | 7.5 | 0.7 | Easy |
| 2 | 12.0 | 0.4 | Hard |

---

## Project Structure

```
test_time_RL/
├── src/
│   ├── envs/nonstationary_cartpole.py  # Environment
│   ├── models/
│   │   ├── abstraction_bank.py         # K slots + gating
│   │   ├── film.py                     # FiLM modulation
│   │   └── astral_agent.py             # Full agent + baseline
│   ├── losses.py                       # Regularization losses
│   ├── train.py                        # PPO training
│   ├── test_time_adapt.py              # TTA experiments
│   └── interventions.py                # Causal interventions
├── scripts/                            # Automation scripts
├── results/                            # Outputs
├── docs/                               # Documentation
└── cleanrl/venv/                       # Virtual environment
```

---

## Quick Start (New Machine)

```bash
# 1. Clone and navigate
git clone https://github.com/<username>/astral-rl.git
cd astral-rl

# 2. Create/activate venv
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python -c "from src.models import ASTRALAgent; print('Ready!')"

# 5. Train
python src/train.py --total_timesteps 500000

# 6. TTA test
python src/test_time_adapt.py --checkpoint results/runs/<name>/final_model.pt
```

---

## Key Commands

```bash
# Train ASTRAL
python src/train.py --use_abstractions True --total_timesteps 500000

# Train Baseline (comparison)
python src/train.py --use_abstractions False --total_timesteps 500000

# Test-time adaptation
python src/test_time_adapt.py --checkpoint <path>

# Causal interventions
python src/interventions.py --checkpoint <path>

# All experiments (automated)
./scripts/run_all.sh

# View logs
tensorboard --logdir results/runs
```

---

## Interpretability Improvements (Optional Flags)

All disabled by default, enable with flags:

```bash
python src/train.py \
    --use_gumbel True \        # Gumbel-Softmax
    --hard_routing True \      # Discrete slot selection
    --orthogonal_init True \   # Diverse slot initialization
    --temp_anneal True \       # Temperature annealing
    --lambda_contrast 0.01 \   # Contrastive loss
    --slot_prediction True     # Auxiliary prediction task
```

---

## Known Issue: Slot Collapse

During training, weights collapse to single slot (~99% slot 1). 

**Solutions documented in**: `docs/interpretability_improvements.md`

---

## Documentation Index

| File | Purpose |
|:-----|:--------|
| `README.md` | Project overview |
| `docs/baseline_vs_astral.md` | Train vs test-time differences |
| `docs/abstraction_bank_vs_moe.md` | MoE comparison |
| `docs/experiment_guide.md` | All experiment commands |
| `docs/interpretability_improvements.md` | Slot collapse solutions |
| `astral_proposal.md` | Research proposal |
| `astral_implementation_plan.md` | Implementation details |

---

## GPU Notes

For GPU training:
```bash
# PyTorch will auto-detect CUDA
python src/train.py --total_timesteps 500000

# Verify GPU usage
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

The code automatically uses GPU if available (no flags needed).

---

## TL;DR

1. **Goal**: Interpretable, efficient test-time adaptation in RL
2. **Method**: Learnable abstraction slots + gating + FiLM modulation
3. **Benchmark**: NonStationaryCartPole (3 hidden modes)
4. **Key advantage**: Update only 2K params at test-time (vs 50K total)
5. **Run experiments**: `./scripts/run_all.sh`

