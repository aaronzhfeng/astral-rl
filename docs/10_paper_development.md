# Paper Development Journey

## Overview

This document chronicles the development of the ASTRAL paper from initial experiments through to the final NeurIPS-style submission.

---

## Paper Versions

### Version 1 (Initial Draft)
- Location: `report/version_1/`
- Basic structure with preliminary results
- Focused on ASTRAL vs baseline comparison
- Discovered issues with original baseline (policy entropy stuck at max)

### Version 2 (Major Revision)
- Location: `report/version_2/`
- Complete rewrite with new narrative focus
- Reframed around **stability** rather than raw performance
- Incorporated SB3 PPO as robust baseline
- Added fair comparison experiments (A, B, C, D)

---

## Key Paper Improvements (Version 2)

### 1. Title Alignment
**Problem**: Title didn't match acronym definition.

**Before**: "ASTRAL: Stable Test-Time Adaptation via Abstraction-Structured Gating..."

**After**: "ASTRAL: Abstraction-Structured Test-time Reinforcement Adaptation Layer for Non-Stationary Environments"

**Acronym**: **A**bstraction-**S**tructured **T**est-time **R**einforcement **A**daptation **L**ayer

---

### 2. Architecture Diagram Overhaul

**Iterations**:
1. Initial vertical layout (too large, unclear)
2. Horizontal layout (better flow)
3. Added detailed Abstraction Bank internals:
   - Gating Network with `~4.3k` parameter annotation
   - Learnable Embeddings (E₀, E₁, E₂)
   - Weighted combination operator (⊗)
   - `Σ wₖEₖ` formula on top of combination node
   - Output `z` clearly shown
4. Standardized box sizes for consistency
5. Math expressions in dashed boxes
6. Proper arrow directions (slots → circle, not through)
7. Enlarged for readability (1.1× textwidth with centering)

**Final Files**:
- `architecture.png` (main, horizontal)
- `architecture_horizontal.png`
- `architecture_vertical.png`

**Generator**: `scripts/generate_architecture_plot.py`

---

### 3. Related Work Expansion

**Before**: 5 short paragraphs (~1 page)

**After**: 8 comprehensive subsections (~3 pages):

1. **Non-Stationary and Hidden-Mode MDPs**
   - Hidden-mode MDPs, context-conditioned policies
   - LILAC, VariBAD
   
2. **Abstraction and State Representation in RL**
   - State abstraction theory, bisimulation metrics
   - DeepMDP, CURL, Slot Attention, Entity Abstraction
   
3. **Catastrophic Forgetting and Continual Learning**
   - Three paradigms: regularization, replay, architecture
   - EWC, SI, MAS, Progressive Networks, PackNet
   
4. **Test-Time Adaptation**
   - Vision TTA: Tent, MEMO, TTT
   - RL adaptation: RMA, context-based
   
5. **Meta-Learning for Fast Adaptation**
   - MAML, Reptile, RL², ProMP, PEARL
   
6. **Mixture of Experts and Gating Mechanisms**
   - MoE history, sparsely-gated MoE
   - Attention mechanisms connection
   
7. **Feature Modulation (FiLM)**
   - Original FiLM, RL applications
   
8. **Dropout and Stochastic Regularization**
   - Dropout variants, connection to slot dropout

**Added ~40 new citations**

---

### 4. Citation Style Change

**Before**: Author-year style `(Kirkpatrick et al., 2017)`

**After**: Numbered style `[1]`, `[2]`, `[17]`

**Technical Implementation**:
```latex
\usepackage[preprint,nonatbib]{neurips_2025}
\usepackage[numbers,sort&compress]{natbib}
\bibliographystyle{unsrtnat}
```

**Note**: Had to use `nonatbib` option because `neurips_2025.sty` loads natbib by default.

---

### 5. Enhanced Experiment Interpretations

For each experiment (A, B, C, D), added:
- **Motivation**: Why this experiment matters
- **Protocol**: Detailed methodology
- **Results and Interpretation**: Deep analysis
- **Why does this happen?**: Mechanistic explanation
- **Implication**: Practical takeaway

**Key additions**:
- Experiment A: "13× worst-case reduction" analysis
- Experiment B: "Anchor effect" explanation, line-by-line Mode 1 preservation
- Experiment C: "Goldilocks problem" for policy-head, implicit regularization
- Experiment D: Counter-intuitive finding that gating advantage increases with mode diversity

**New Summary Table**: Consolidated all findings across experiments.

---

### 6. Slot Collapse Section Enhancement

Added interpretation for each configuration in Table 2:
- Slot Dropout (p=0.3): "The sweet spot"
- Slot Dropout (p=0.5): "Over-regularization"
- Strong Regularization: "Ceiling effect"
- Collapsed Default: "No diversity to exploit"
- Diverse (reg. only): "Numerical but not functional diversity"

**Key insight**: Numerical vs. functional diversity distinction.

---

## Final Paper Statistics

- **Pages**: 20 (including appendix)
- **Figures**: 9 (6 main + 3 appendix)
- **Tables**: 10
- **References**: 54
- **Main contributions**: 4

---

## File Structure

```
report/
├── version_1/           # Original draft (archived)
│   ├── astral_paper.tex
│   ├── astral_paper.pdf
│   └── asset/
│       ├── table/
│       └── figure/
│
└── version_2/           # Current version
    ├── astral_paper.tex # Main paper
    ├── astral_paper.pdf # Compiled PDF
    ├── references.bib   # 54 citations
    ├── neurips_2025.sty # NeurIPS style
    └── asset/
        └── figure/
            ├── architecture.png
            ├── architecture_horizontal.png
            ├── architecture_vertical.png
            ├── slot_collapse_comparison.png
            ├── forgetting_comparison.png
            ├── fewshot_adaptation.png
            ├── extreme_modes.png
            ├── training_comparison_v2.png
            ├── tta_by_model.png
            ├── causal_interventions.png
            └── sb3_finetuning.png
```

---

## Plot Generation Scripts

| Script | Purpose |
|--------|---------|
| `generate_architecture_plot.py` | ASTRAL architecture diagrams |
| `generate_paper_plots.py` | Initial plot generation |
| `generate_paper_plots_v2.py` | Improved plots with better aesthetics |
| `plot_fewshot_comparison.py` | Few-shot adaptation comparison |

---

## Lessons Learned

### 1. Baseline Matters
Our original custom `BaselineAgent` was fundamentally broken (policy entropy stuck at max). Switching to SB3 PPO provided a legitimate comparison.

### 2. Narrative Reframing
Initial focus on "ASTRAL beats baseline" was unsustainable. Reframing around **stability and forgetting resistance** led to a more honest and compelling story.

### 3. Fair Comparisons
Parameter-matched comparisons (Experiment A) revealed insights that raw comparisons missed. Always control for confounding variables.

### 4. Slot Collapse is Real
The slot collapse phenomenon was unexpected but became a key contribution. Documenting failures can be as valuable as successes.

### 5. Slot Dropout Works
Simple solution (30% dropout) to a complex problem. Sometimes the straightforward approach is best.

---

## Future Work (from paper)

1. Functional diversity mechanisms beyond slot dropout
2. Explicit mode-conditioned training
3. Extension to more complex environments
4. Integration with meta-learning approaches

---

## Timeline

1. **Initial experiments**: ASTRAL vs custom baseline
2. **Slot collapse discovery**: 73% of configurations collapsed
3. **Slot dropout solution**: p=0.3 enables TTA
4. **Baseline fix**: Switch to SB3 PPO
5. **Fair comparison experiments**: A, B, C, D designed
6. **Paper v1**: Initial draft
7. **Paper v2**: Major revision with stability focus
8. **Final polish**: Architecture diagram, citations, interpretations

