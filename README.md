# ASTRAL: Abstraction-Structured Test-time Reinforcement Adaptation Layer

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](report/version_2/astral_paper.pdf)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **proof-of-concept** for structured abstractions in in-context RL, demonstrating stable test-time adaptation through gating-only updates.

**Authors:** Aaron Feng, Rita Yujia Wu, Bella Wang, Sophie Wang (UC San Diego)

---

## 🎯 Key Finding

> **Gating-only adaptation provides 10× less catastrophic forgetting than full fine-tuning**, at the cost of lower peak improvement. ASTRAL is suited for **risk-averse** deployment where stability matters more than maximum single-mode performance.

| Method | TTA Improvement | Forgetting | Variance |
|--------|----------------|------------|----------|
| Gating (ASTRAL) | +11 | -25.8 | Low |
| Full Fine-tune | +77 | -250.1 | High |

---

## 📊 Results Summary

Through **38 experiments** across 33+ model configurations:

- ✅ **Slot collapse** occurs in 73% of configurations → **Slot dropout (p=0.3)** mitigates it
- ✅ **10× less forgetting** with gating-only TTA vs full fine-tuning
- ✅ **Consistent performance** across all episode budgets (1-50)
- ⚠️ **Interpretability limited** — no clean mode→slot correspondence emerges
- ⚠️ **Toy environment** — CartPole is simple; scaling to complex envs needed

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

**Key Design:**
- **FiLM modulation** ensures policy depends on abstractions (no bypass)
- **Test-time adaptation** updates only gating network (~4.3k params, 8% of model)

---

## 📁 Project Structure

```
astral-rl/
├── README.md
├── requirements.txt
│
├── src/
│   ├── envs/
│   │   └── nonstationary_cartpole.py  # 3-mode CartPole
│   ├── models/
│   │   ├── abstraction_bank.py        # K learnable slots + gating
│   │   ├── film.py                    # Feature-wise Linear Modulation
│   │   └── astral_agent.py            # Full agent + baseline
│   ├── losses.py                      # Regularization losses
│   ├── train.py                       # PPO training loop
│   ├── test_time_adapt.py             # TTA experiments
│   └── interventions.py               # Causal interventions
│
├── scripts/
│   ├── generate_paper_plots_v2.py     # Figure generation
│   ├── run_fair_comparison.py         # Fair comparison experiments
│   └── ...
│
├── results/
│   ├── runs/                          # Training checkpoints
│   ├── analysis/                      # Experiment documentation
│   ├── fair_comparison/               # Experiments A-D results
│   └── tta_final_validation/          # TTA results
│
├── report/
│   └── version_2/
│       ├── astral_paper.pdf           # Full paper (20 pages)
│       └── astral_paper.tex
│
└── docs/
    ├── 00_experiment_checklist.md     # Experiment status tracker
    ├── 01_astral_proposal.md          # Original proposal
    └── ...
```

---

## 🚀 Quick Start

### Setup

```bash
git clone https://github.com/aaronzhfeng/astral-rl.git
cd astral-rl
pip install -r requirements.txt
```

### Train ASTRAL (Best Config)

```bash
python src/train.py \
    --use_gumbel True \
    --temp_anneal True \
    --lambda_contrast 0.1 \
    --lambda_lb 0.05 \
    --slot_dropout 0.3 \
    --exp_name best_config
```

### Test-Time Adaptation

```bash
python src/test_time_adapt.py \
    --checkpoint results/runs/best_config/final_model.pt \
    --num_adapt_episodes 20
```

### View Training Logs

```bash
tensorboard --logdir results/runs
```

---

## 🧪 Environment: NonStationaryCartPole

| Mode | Gravity | Pole Length | Difficulty |
|:-----|:--------|:------------|:-----------|
| 0 | 9.8 | 0.5 | Default |
| 1 | 7.5 | 0.7 | Easy |
| 2 | 12.0 | 0.4 | Hard |

Agent does **not** observe the mode — must infer from dynamics.

---

## ⚠️ Known Limitations

This is a **proof-of-concept** with significant limitations:

1. **Toy environment** — CartPole is trivial; solved in the 1980s
2. **Slot collapse** — 73% of configs collapse to single slot
3. **No interpretability** — slots don't map to semantic modes
4. **Modest TTA** — +11 improvement vs +77 for full fine-tuning
5. **No meta-RL comparison** — didn't compare to MAML, PEARL, etc.

See paper Section 7.3 for full discussion.

---

## 🔮 Future Work: TAG-AMAGO

ASTRAL validates core ideas on a toy testbed. The next step is **TAG-AMAGO**:

1. **Transformer backbone** — Replace GRU with AMAGO-style transformer
2. **Challenging benchmarks** — MuJoCo, Meta-World ML10/ML45
3. **More slots** — K=8-16 instead of K=3
4. **Rigorous interpretability** — Cross-task transfer, causal ablations
5. **Meta-RL comparison** — MAML, PEARL, VariBAD, Algorithm Distillation

See paper Section 8.4 and `docs/` for detailed roadmap.

---

## 📚 Documentation

| Document | Description |
|:---------|:------------|
| [Paper (PDF)](report/version_2/astral_paper.pdf) | Full 20-page paper |
| [docs/00_experiment_checklist.md](docs/00_experiment_checklist.md) | Experiment status |
| [results/analysis/](results/analysis/) | Detailed experiment logs |

---

## 📚 References

Key papers that influenced this work:

| Paper | Relevance |
|:------|:----------|
| [AMAGO](https://arxiv.org/abs/2310.09971) (Grigsby et al., 2024) | State-of-the-art in-context RL; TAG-AMAGO builds on this |
| [FiLM](https://arxiv.org/abs/1709.07871) (Perez et al., 2018) | Feature-wise modulation mechanism we use |
| [MoASE](https://arxiv.org/abs/2405.16486) (Zhang et al., 2024) | MoE for continual test-time adaptation (vision) |
| [Slot Attention](https://arxiv.org/abs/2006.15055) (Locatello et al., 2020) | Object-centric slots; inspired our abstraction bank |
| [EWC](https://arxiv.org/abs/1612.00796) (Kirkpatrick et al., 2017) | Catastrophic forgetting prevention |
| [MAML](https://arxiv.org/abs/1703.03400) (Finn et al., 2017) | Meta-learning for fast adaptation |
| [PEARL](https://arxiv.org/abs/1903.08254) (Rakelly et al., 2019) | Probabilistic context for meta-RL |
| [VariBAD](https://arxiv.org/abs/1910.08348) (Zintgraf et al., 2020) | Bayes-adaptive deep RL |
| [PPO](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) | RL algorithm we use |
| [Stable-Baselines3](https://jmlr.org/papers/v22/20-1364.html) (Raffin et al., 2021) | Baseline implementation |

See paper for full bibliography (54 references).

---

## 📄 License

MIT License
