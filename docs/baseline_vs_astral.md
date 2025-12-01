# Baseline vs ASTRAL: Training and Test-Time Differences

A detailed comparison of the **GRU-only Baseline** and **ASTRAL** during training and test-time deployment.

---

## Quick Summary

|  | Baseline (GRU-only) | ASTRAL |
|:-|:--------------------|:-------|
| **Training** | Standard PPO | PPO + Abstraction Regularization |
| **Test (no adaptation)** | GRU hidden state adapts | GRU hidden state + slot weights adapt |
| **Test (with adaptation)** | Update all params (expensive) or none | Update only gating (cheap, targeted) |
| **Interpretability** | None | Can inspect slot weights per mode |

---

## Architecture Comparison

### Baseline (GRU-only)

```
    Input: (s_t, a_{t-1}, r_{t-1})
              │
              ▼
    ┌─────────────────┐
    │   Input MLP     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │      GRU        │ ──── h_t (context embedding)
    └────────┬────────┘
             │
             ▼
           h_t ─────────────────┐
             │                  │
             ▼                  ▼
    ┌─────────────┐     ┌─────────────┐
    │ Policy Head │     │ Value Head  │
    └─────────────┘     └─────────────┘
```

**Parameters**: ~47K
- Input MLP: ~450
- GRU: ~25K  
- Policy Head: ~8.5K
- Value Head: ~8.5K

### ASTRAL

```
    Input: (s_t, a_{t-1}, r_{t-1})
              │
              ▼
    ┌─────────────────┐
    │   Input MLP     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │      GRU        │ ──── h_t (raw context)
    └────────┬────────┘
             │
     ┌───────┴───────┐
     │               │
     ▼               ▼
┌─────────┐   ┌─────────────────┐
│ Gating  │   │ Abstraction     │
│  MLP    │   │ Bank [K × d]    │
└────┬────┘   └────────┬────────┘
     │                 │
     ▼                 │
  weights w_t          │
     │                 │
     └────────┬────────┘
              ▼
        z_t = Σ w_k × slot_k
              │
              ▼
    ┌─────────────────┐
    │      FiLM       │ ──── h'_t = γ(z) ⊙ h_t + β(z)
    └────────┬────────┘
             │
             ▼
           h'_t ────────────────┐
             │                  │
             ▼                  ▼
    ┌─────────────┐     ┌─────────────┐
    │ Policy Head │     │ Value Head  │
    └─────────────┘     └─────────────┘
```

**Parameters**: ~51K
- Input MLP: ~450
- GRU: ~25K
- Gating MLP: ~2.2K ← **Only this updated at test-time**
- Abstraction Bank: ~192 (K=3, d=64)
- FiLM: ~8.3K
- Policy Head: ~8.5K
- Value Head: ~8.5K

---

## Training Phase

### What's the Same

| Aspect | Both |
|:-------|:-----|
| **Algorithm** | PPO (Proximal Policy Optimization) |
| **Environment** | NonStationaryCartPole (3 modes, random per episode) |
| **Optimizer** | Adam, lr=3e-4 |
| **Rollout** | Collect 128 steps × 8 envs, then update |
| **Losses** | Policy loss + Value loss + Entropy bonus |

### What's Different

| Aspect | Baseline | ASTRAL |
|:-------|:---------|:-------|
| **Forward pass** | h_t → Heads | h_t → Abstraction Bank → FiLM → h'_t → Heads |
| **Extra losses** | None | L_w_ent + L_lb + L_orth (regularization) |
| **Logged metrics** | Return, length | Return, length, **slot weights**, **entropy** |
| **What learns** | All params | All params |

### Training Forward Pass

**Baseline:**
```python
def forward(obs, prev_action, prev_reward, hidden):
    x = input_mlp(concat(obs, prev_action, prev_reward))
    h = gru(x, hidden)
    
    logits = policy_head(h)   # Direct from h
    value = value_head(h)     # Direct from h
    
    return logits, value, h
```

**ASTRAL:**
```python
def forward(obs, prev_action, prev_reward, hidden):
    x = input_mlp(concat(obs, prev_action, prev_reward))
    h = gru(x, hidden)
    
    # Abstraction bank
    z, weights = abstraction_bank(h)  # z = Σ w_k × slot_k
    
    # FiLM modulation
    h_mod = film(h, z)  # h' = γ(z) ⊙ h + β(z)
    
    logits = policy_head(h_mod)   # From modulated h'
    value = value_head(h_mod)     # From modulated h'
    
    return logits, value, h, weights
```

### Training Loss

**Baseline:**
```python
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

**ASTRAL:**
```python
# Standard PPO losses
ppo_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

# Abstraction regularization
L_w_ent = -λ1 * mean_entropy(weights)        # Encourage non-peaked
L_lb = λ2 * negative_entropy(mean(weights))  # Encourage balanced usage
L_orth = λ3 * ||A @ A.T - I||²               # Encourage diverse slots

loss = ppo_loss + L_w_ent + L_lb + L_orth
```

---

## Test Phase (No Adaptation)

In this mode, we deploy the trained model **without any gradient updates**.

### What's the Same

| Aspect | Both |
|:-------|:-----|
| **Weights** | Frozen (no updates) |
| **Environment** | Can be fixed to specific mode (e.g., Mode 2) |
| **Evaluation** | Run N episodes, measure return |

### What's Different

| Aspect | Baseline | ASTRAL |
|:-------|:---------|:-------|
| **In-context adaptation** | Only via GRU h_t | GRU h_t + slot weights w_t |
| **Observability** | Can't see internal state | Can log which slots are active |
| **Adaptation speed** | Depends on GRU memory | Can shift weights quickly |

### How In-Context Adaptation Works

**Baseline:**
```
Episode starts (Mode 2 - hard physics)
   ↓
Step 1: h_1 = GRU(x_1, h_0)
   - h_1 starts learning "physics feel fast"
   ↓
Step 5: h_5 = GRU(x_5, h_4)
   - h_5 has encoded "I'm in hard mode"
   - Policy behavior changes implicitly
   ↓
All adaptation is INSIDE the opaque h vector
```

**ASTRAL:**
```
Episode starts (Mode 2 - hard physics)
   ↓
Step 1: h_1 = GRU(x_1, h_0)
         w_1 = gating(h_1) = [0.3, 0.3, 0.4]
   - Weights start shifting toward slot 2
   ↓
Step 5: h_5 = GRU(x_5, h_4)
         w_5 = gating(h_5) = [0.1, 0.1, 0.8]
   - Now strongly using slot 2 (hard mode strategy)
   - We can SEE this shift in the weights!
   ↓
Adaptation is VISIBLE in slot weights w
```

### Visualization During Test

**Baseline**: Only see returns
```
Episode 1: Return = 95
Episode 2: Return = 102
Episode 3: Return = 98
```

**ASTRAL**: See returns AND slot usage
```
Episode 1: Return = 95,  Avg weights = [0.2, 0.2, 0.6]
Episode 2: Return = 102, Avg weights = [0.1, 0.1, 0.8]
Episode 3: Return = 98,  Avg weights = [0.1, 0.1, 0.8]
                                        ↑ Mode 2 → Slot 2
```

---

## Test Phase (With Adaptation / Test-Time RL)

In this mode, we allow **gradient updates at test time** to adapt to a new/shifted environment.

### What's the Same

| Aspect | Both |
|:-------|:-----|
| **Goal** | Improve performance on target mode |
| **Data** | Collect rollouts in target environment |

### What's Different — THIS IS THE KEY DISTINCTION

| Aspect | Baseline | ASTRAL |
|:-------|:---------|:-------|
| **What to update?** | All params or none | **Only gating network** |
| **Parameters updated** | ~47,000 | **~2,200** (4.7% of model) |
| **Risk** | Catastrophic forgetting | Minimal (backbone frozen) |
| **Speed** | Slow | Fast |
| **Stability** | Can diverge | Stable (constrained update) |

### Test-Time Adaptation Code

**Baseline (Full Fine-tuning):**
```python
# Option 1: Update everything (risky)
optimizer = Adam(agent.parameters(), lr=1e-4)

for episode in range(num_adapt_episodes):
    rollout = collect_episode(env, agent)
    loss = compute_ppo_loss(rollout)
    
    optimizer.zero_grad()
    loss.backward()  # Gradients flow to ALL parameters
    optimizer.step()

# Problems:
# - Can forget how to handle other modes
# - 47K parameters is a lot to update with few samples
# - Easy to overfit to current mode
```

**Baseline (No Adaptation):**
```python
# Option 2: Don't update at all
# Just rely on GRU in-context learning
# Limited adaptation capability
```

**ASTRAL (Targeted Gating Update):**
```python
# Freeze everything except gating
agent.freeze_except_gating()  # Only gating.requires_grad = True

optimizer = Adam(agent.get_gating_parameters(), lr=3e-4)

for episode in range(num_adapt_episodes):
    rollout = collect_episode(env, agent)
    loss = compute_ppo_loss(rollout)
    
    optimizer.zero_grad()
    loss.backward()  # Gradients ONLY flow to gating (~2.2K params)
    optimizer.step()

# Benefits:
# - Backbone (GRU, FiLM, heads) stays intact
# - Only learning "which slots to use"
# - Fast convergence (few parameters)
# - Preserves knowledge of all modes
```

### What Each Approach Learns at Test Time

**Baseline (full fine-tune):**
- GRU weights change → changes how context is built
- Policy head changes → changes action mapping
- **Risk**: Entire behavior can shift unpredictably

**ASTRAL:**
- Gating weights change → changes slot mixing
- **Everything else fixed**
- **Result**: Just learns "use more of slot X for this mode"

### Analogy

| | Baseline | ASTRAL |
|:-|:---------|:-------|
| **Full fine-tune** | Rewriting the whole book | Choosing a different chapter |
| **What changes** | All behaviors | Only slot selection |
| **Recovery** | Hard (weights overwritten) | Easy (just reset gating) |

---

## Test-Time Adaptation Results (Typical)

### Setup
- Train on all 3 modes (random per episode)
- Test on fixed Mode 0 with 20 adaptation episodes

| Metric | Baseline (no adapt) | Baseline (full adapt) | ASTRAL (gating only) |
|:-------|:--------------------|:----------------------|:---------------------|
| **Initial return** | ~115 | ~115 | ~115 |
| **After 20 eps** | ~115 (no change) | ~125 (improved) | ~127 (improved) |
| **Params updated** | 0 | 47,000 | 2,200 |
| **Forgetting risk** | None | High | Low |
| **Compute** | - | High | Low |

---

## Causal Interventions (Test Time)

Only ASTRAL supports clean causal interventions.

### Slot Clamping

```python
# Force specific slot usage
def forward_with_clamp(h, clamp_slot):
    # Override gating
    weights = torch.zeros(K)
    weights[clamp_slot] = 1.0  # 100% slot X
    
    z = weights @ abstraction_bank
    h_mod = film(h, z)
    return policy_head(h_mod)
```

**Use case**: "What if agent ONLY uses the hard-mode strategy?"

### Slot Disabling

```python
# Prevent using a specific slot
def forward_with_disable(h, disable_slot):
    logits = gating(h)
    logits[disable_slot] = -1000  # Mask out
    weights = softmax(logits)
    
    z = weights @ abstraction_bank
    h_mod = film(h, z)
    return policy_head(h_mod)
```

**Use case**: "How important is slot 2 for Mode 2 performance?"

### Baseline Cannot Do This

The baseline has no discrete components to intervene on. The GRU hidden state is a dense, entangled vector — you can't cleanly "disable" part of it.

---

## Summary: When to Use Which

| Scenario | Recommendation |
|:---------|:---------------|
| **Just need good performance** | Either (similar) |
| **Need interpretability** | ASTRAL |
| **Need test-time adaptation** | ASTRAL (much better) |
| **Limited compute at test time** | ASTRAL |
| **Want to analyze mode→behavior** | ASTRAL |
| **Simplicity is priority** | Baseline |

---

## Complete Comparison Table

| Aspect | Baseline | ASTRAL |
|:-------|:---------|:-------|
| **Architecture** | GRU → Heads | GRU → Abstraction → FiLM → Heads |
| **Parameters** | ~47K | ~51K |
| **Training algorithm** | PPO | PPO + regularization |
| **Training losses** | Policy + Value + Entropy | + L_w_ent + L_lb + L_orth |
| **In-context adaptation** | Via h_t only | Via h_t and w_t |
| **Adaptation visibility** | None | Slot weights logged |
| **Test-time RL** | All-or-nothing | Gating only (2.2K params) |
| **Forgetting risk** | High (if fine-tune) | Low |
| **Intervention** | Not possible | Clamp/disable slots |
| **Interpretability** | Low | High |

---

## Code Reference

```python
# Training
python src/train.py --use_abstractions True   # ASTRAL
python src/train.py --use_abstractions False  # Baseline

# Test-time adaptation (ASTRAL only)
python src/test_time_adapt.py --checkpoint <model.pt>

# Causal interventions (ASTRAL only)
python src/interventions.py --checkpoint <model.pt>
```

