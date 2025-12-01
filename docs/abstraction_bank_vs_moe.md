# Abstraction Bank vs Mixture of Experts (MoE)

This document explains the conceptual relationship between ASTRAL's **Abstraction Bank** and **Mixture of Experts (MoE)** architectures.

---

## Overview

ASTRAL's abstraction bank shares conceptual similarities with MoE but serves different purposes. Understanding this relationship helps clarify the design choices in ASTRAL.

**Key insight**: ASTRAL is like **MoE's gating mechanism applied to a bank of learned embeddings** instead of expert networks.

---

## Similarities

| Aspect | MoE | ASTRAL Abstraction Bank |
|:-------|:----|:------------------------|
| **Multiple components** | K experts | K abstraction slots |
| **Gating mechanism** | Router network | Gating MLP |
| **Soft selection** | softmax over experts | softmax over slots |
| **Weighted combination** | Σ w_k × Expert_k(x) | Σ w_k × slot_k |
| **Sparsity encouraged** | Top-K routing | Temperature τ |

Both share the intuition: *"different inputs should activate different specialized components"*

---

## Key Differences

### 1. Experts vs Vectors

**MoE:**
```
Input x → Expert_k(x) → output_k   (Each expert is a full neural network)
          ↓
Final = Σ w_k × output_k
```

**Abstraction Bank:**
```
slot_k is just a learned vector (no computation)
          ↓
z = Σ w_k × slot_k   (weighted sum of vectors, not network outputs)
```

The abstraction slots are **static parameter vectors**, not networks that process input.

### 2. What Gets Mixed

**MoE:** Mixes the *outputs* of expert networks

```python
# Transformer MoE layer
outputs = [expert(x) for expert in experts]  # Each expert computes
final = sum(w[i] * outputs[i] for i in range(K))
```

**Abstraction Bank:** Mixes *vectors* that then *modulate* the context

```python
# ASTRAL
z = sum(w[i] * slots[i] for i in range(K))  # No computation in slots
h_modulated = gamma(z) * h + beta(z)        # z modulates h via FiLM
```

### 3. Purpose and Design Goals

| Aspect | MoE | Abstraction Bank |
|:-------|:----|:-----------------|
| **Primary goal** | Scale model capacity | Interpretability + Test-time adaptation |
| **Why K components?** | Computational efficiency | Each slot = one "strategy" |
| **Sparsity motivation** | Save compute (activate fewer experts) | Cleaner mode→slot mapping |
| **At test time** | Fixed routing | Can update gating with gradients |

---

## Visual Comparison

### MoE (Transformer Style)

```
         Input x
            │
            ▼
    ┌───────────────┐
    │    Router     │ ← Learned gating
    └───────┬───────┘
            │
     weights [w1, w2, w3]
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
┌──────┐ ┌──────┐ ┌──────┐
│Expert│ │Expert│ │Expert│   ← Each is a full FFN
│  1   │ │  2   │ │  3   │
└──┬───┘ └──┬───┘ └──┬───┘
   │        │        │
   ▼        ▼        ▼
  o_1      o_2      o_3      ← Expert outputs
   │        │        │
   └────────┼────────┘
            ▼
   output = Σ w_i × o_i
```

### ASTRAL Abstraction Bank

```
         Context h
            │
            ▼
    ┌───────────────┐
    │  Gating MLP   │ ← Learned gating
    └───────┬───────┘
            │
     weights [w1, w2, w3]
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
┌──────┐ ┌──────┐ ┌──────┐
│Slot 0│ │Slot 1│ │Slot 2│   ← Just vectors (no computation)
│ [d]  │ │ [d]  │ │ [d]  │
└──────┘ └──────┘ └──────┘
   │        │        │
   └────────┼────────┘
            ▼
    z = Σ w_i × slot_i       ← Weighted sum of vectors
            │
            ▼
    ┌───────────────┐
    │     FiLM      │       ← z modulates h
    └───────┬───────┘
            ▼
    h' = γ(z) ⊙ h + β(z)    ← Modulated context to policy/value
```

---

## The "Soft MoE over Embeddings" Analogy

If you squint, ASTRAL is like a **"Soft MoE over embeddings"**:

- Instead of K expert *networks*, you have K expert *embeddings*
- Instead of mixing *outputs*, you mix *latent vectors*
- The mixed vector then *conditions* the downstream computation via FiLM

This design pattern is related to:

| Related Concept | Similarity |
|:----------------|:-----------|
| **Prompt tuning** | Learnable prefix embeddings that condition behavior |
| **Task embeddings** | In multi-task learning, tasks are represented as vectors |
| **Concept bottleneck models** | Interpretable intermediate representations |
| **Hypernetworks** | Parameters conditioned on context |
| **Soft attention** | Weighted combination of value vectors |

---

## Why Not Just Use Full MoE?

For ASTRAL's goals, full MoE would be **overkill**:

| Requirement | Full MoE | Abstraction Vectors |
|:------------|:---------|:--------------------|
| **Interpretability** | Hard (experts are black boxes) | Easy (vectors are small, can visualize) |
| **Test-time adaptation** | Update which experts? All? | Update only gating (tiny!) |
| **Causal intervention** | How to "disable" an expert cleanly? | Just zero out one slot's weight |
| **Parameter count** | K × expert_size | K × d (much smaller) |
| **Analysis** | Complex (expert behavior) | Simple (vector similarity, weights) |

The simplicity of "slots = vectors" is the point — it makes ASTRAL **interpretable and efficient**.

---

## Detailed Comparison Table

| Feature | MoE | ASTRAL Abstraction Bank |
|:--------|:----|:------------------------|
| **Component type** | Neural network (FFN) | Learned vector |
| **Component size** | d_model × d_ff × 2 (thousands of params) | d_model (64 params) |
| **Gating input** | Token/input embedding | GRU context embedding h_t |
| **Gating output** | Top-K sparse weights | Soft weights (softmax with τ) |
| **Combination** | Mix expert outputs | Mix vectors, then FiLM modulate |
| **Computation** | Experts compute in parallel | No computation in slots |
| **Training** | End-to-end with load balancing | End-to-end with regularization |
| **Test-time update** | Not typical | Core feature (update gating only) |
| **Interpretability** | Low | High (designed goal) |
| **Causal intervention** | Difficult | Easy (clamp/disable slots) |

---

## Regularization Comparison

Both architectures need regularization to prevent collapse:

### MoE Load Balancing

```python
# Encourage uniform usage across experts
aux_loss = sum((fraction_routed_to_k - 1/K)^2 for k in range(K))
```

### ASTRAL Abstraction Regularization

```python
# 1. Per-sample entropy: encourage non-peaked weights
L_w_ent = -mean(entropy(w))

# 2. Load balancing: uniform usage across batch
L_lb = -entropy(mean(w, dim=batch))

# 3. Orthogonality: diverse slot vectors
L_orth = ||A @ A.T - I||^2
```

ASTRAL adds **orthogonality regularization** (not in MoE) because slots being different vectors matters for interpretability.

---

## Code Comparison

### MoE Routing (Simplified)

```python
class MoELayer(nn.Module):
    def __init__(self, num_experts, d_model, d_ff):
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        # Route
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)
        
        # Compute expert outputs
        outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(outputs, dim=-1)  # [batch, d_model, K]
        
        # Weighted sum
        return torch.einsum('bdk,bk->bd', outputs, weights)
```

### ASTRAL Abstraction Bank

```python
class AbstractionBank(nn.Module):
    def __init__(self, d_model, num_slots):
        # Slots are just learned vectors, NOT networks
        self.slots = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.gating = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_slots)
        )
    
    def forward(self, h):
        # Route
        logits = self.gating(h)
        weights = F.softmax(logits / self.tau, dim=-1)
        
        # Weighted sum of slot VECTORS (no computation)
        z = torch.einsum('bk,kd->bd', weights, self.slots)
        
        return z, weights
```

**Key difference**: MoE iterates through `self.experts` (networks), ASTRAL just indexes into `self.slots` (vectors).

---

## When to Use Which

| Use Case | Best Choice |
|:---------|:------------|
| Scaling language models | MoE (proven at scale) |
| Interpretable RL | Abstraction Bank (ASTRAL) |
| Multi-task learning | Either (depends on goals) |
| Test-time adaptation | Abstraction Bank (can update gating) |
| Computational efficiency | MoE with Top-K |
| Causal analysis | Abstraction Bank (clean interventions) |

---

## Summary

> **ASTRAL's abstraction bank is like MoE's gating mechanism applied to a bank of learned embeddings instead of expert networks.**

Same spirit:
- Specialized components
- Learned routing
- Weighted combination

Different implementation:
- Vectors vs networks
- Modulation vs output mixing
- Interpretability vs capacity scaling

The simplicity of ASTRAL's design (slots = vectors) is intentional — it enables the interpretability and efficient test-time adaptation that are the core goals of the approach.

---

## References

- **MoE**: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- **Switch Transformer**: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models" (2021)
- **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
- **Hypernetworks**: Ha et al., "HyperNetworks" (2016)

