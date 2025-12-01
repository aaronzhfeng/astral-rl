# Interpretability Improvements for ASTRAL

This document discusses the **slot collapse** problem observed in ASTRAL and proposes solutions to improve interpretability.

**All improvements are now implemented and available via command-line flags.**

---

## Problem: Slot Collapse

### Observed Behavior

During training, the abstraction weights collapse to a single slot:

```
Update 5:   Avg weights: [0.007, 0.985, 0.008]  ← Already peaked
Update 10:  Avg weights: [0.000, 0.999, 0.000]  ← Collapsed
Update 95:  Avg weights: [0.000, 0.999, 0.000]  ← Stays collapsed
```

All three modes (default, easy, hard) use **Slot 1** exclusively (~99.999%).

### Why This Happens

1. **Gradient Flow**: Early in training, one slot may receive slightly stronger gradients, causing it to become more useful, which attracts more weight, creating a feedback loop.

2. **FiLM Identity Initialization**: FiLM starts as near-identity (γ≈1, β≈0), so different abstractions initially produce similar outputs. The model doesn't "need" multiple slots.

3. **Regularization Too Weak**: The default λ values (0.001) are insufficient to counteract the natural tendency toward collapse.

4. **Soft Attention Dynamics**: Softmax naturally pushes weights toward extremes when the model is confident.

### Consequences

| Aspect | Impact |
|:-------|:-------|
| **Learning** | ✅ Model still learns (GRU handles adaptation) |
| **Interpretability** | ❌ No mode→slot correspondence |
| **TTA Value** | ⚠️ Limited (gating has little to adjust) |
| **Causal Interventions** | ❌ All slots equivalent except Slot 1 |

---

## Solution 1: Stronger Regularization

### Approach

Increase the regularization coefficients significantly.

### Implementation

```python
# In train.py, increase lambdas:
python src/train.py \
    --lambda_w_ent 0.1 \    # 100x increase (maximize entropy)
    --lambda_lb 0.1 \       # 100x increase (balance load)
    --lambda_orth 0.01      # 100x increase (diverse slots)
```

### Expected Effect

- Higher `lambda_w_ent`: Penalizes peaked weight distributions → more uniform weights
- Higher `lambda_lb`: Penalizes uneven slot usage across batch → better load balancing
- Higher `lambda_orth`: Penalizes similar abstraction vectors → diverse slots

### Tradeoff

Too much regularization may hurt task performance. Need to find the right balance.

---

## Solution 2: Temperature Annealing

### Approach

Start with high temperature (uniform weights), gradually decrease to allow specialization.

### Implementation

```python
# In src/train.py, add temperature schedule:

class TemperatureScheduler:
    def __init__(self, tau_start=10.0, tau_end=0.5, warmup_steps=100000):
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.warmup_steps = warmup_steps
    
    def get_tau(self, step):
        if step >= self.warmup_steps:
            return self.tau_end
        progress = step / self.warmup_steps
        return self.tau_start + (self.tau_end - self.tau_start) * progress

# During training:
tau_scheduler = TemperatureScheduler()
for update in range(num_updates):
    current_tau = tau_scheduler.get_tau(global_step)
    agent.abstraction_bank.set_temperature(current_tau)
    # ... training loop
```

### Expected Effect

- **Early training** (τ=10): Uniform weights → all slots get gradients → all slots learn useful features
- **Late training** (τ=0.5): Peaked weights → slots specialize to different modes

---

## Solution 3: Gumbel-Softmax with Hard Routing

### Approach

Use Gumbel-Softmax to enable discrete slot selection during forward pass while maintaining gradient flow.

### Implementation

```python
# In src/models/abstraction_bank.py:

import torch.nn.functional as F

class AbstractionBankGumbel(nn.Module):
    def __init__(self, d_model, num_abstractions, tau=1.0, hard=False):
        super().__init__()
        self.hard = hard  # Whether to use hard (one-hot) routing
        # ... same as before
    
    def forward(self, h):
        logits = self.gating(h)
        
        if self.training:
            # Gumbel-Softmax for differentiable discrete selection
            w = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard)
        else:
            # At test time, use hard selection
            w = F.one_hot(logits.argmax(dim=-1), num_classes=self.num_abstractions).float()
        
        z = torch.einsum('bk,kd->bd', w, self.abstractions)
        return z, w
```

### Expected Effect

- **Hard routing** forces the model to commit to a single slot per sample
- **Gumbel noise** during training encourages exploration of different slots
- **Discrete at test time** makes interpretability clear (exactly one slot per sample)

---

## Solution 4: Contrastive Auxiliary Loss

### Approach

Add a contrastive loss that encourages different modes to use different slots.

### Implementation

```python
# In src/losses.py:

def compute_contrastive_slot_loss(weights, modes, lambda_contrast=0.01):
    """
    Encourage different modes to have different weight distributions.
    
    Args:
        weights: [batch, K] mixture weights
        modes: [batch] mode labels (0, 1, 2)
        lambda_contrast: regularization strength
    """
    unique_modes = torch.unique(modes)
    if len(unique_modes) < 2:
        return torch.tensor(0.0)
    
    # Compute mean weight per mode
    mode_weights = {}
    for mode in unique_modes:
        mask = (modes == mode)
        mode_weights[mode.item()] = weights[mask].mean(dim=0)
    
    # Contrastive loss: encourage different modes to have different weights
    loss = 0.0
    count = 0
    for i, m1 in enumerate(unique_modes):
        for m2 in unique_modes[i+1:]:
            # Maximize distance between mode weight distributions
            similarity = F.cosine_similarity(
                mode_weights[m1.item()].unsqueeze(0),
                mode_weights[m2.item()].unsqueeze(0)
            )
            loss += similarity  # We want to minimize similarity
            count += 1
    
    return lambda_contrast * loss / max(count, 1)
```

### Expected Effect

- Explicitly encourages different modes to activate different slot patterns
- Requires mode labels during training (available in our environment)

### Note

This is **supervised** in the sense that it uses mode labels. A fully unsupervised version would use clustering or other techniques.

---

## Solution 5: Slot-Specific Initialization

### Approach

Initialize abstraction vectors to be maximally different from each other.

### Implementation

```python
# In src/models/abstraction_bank.py:

def _init_orthogonal_abstractions(self):
    """Initialize abstractions to be orthogonal."""
    # Use SVD to get orthogonal vectors
    random_matrix = torch.randn(self.num_abstractions, self.d_model)
    u, s, v = torch.svd(random_matrix)
    
    # Use first K columns of V as orthogonal abstractions
    orthogonal = v[:, :self.num_abstractions].T
    
    self.abstractions.data = orthogonal * 0.1  # Scale down
```

### Expected Effect

- Abstractions start maximally different
- May help early training distinguish between slots

---

## Solution 6: Auxiliary Slot Prediction Loss

### Approach

Add a secondary loss that predicts which slot was used, encouraging slot diversity.

### Implementation

```python
# Add slot predictor to agent:

class SlotPredictor(nn.Module):
    """Predict which slot was used from the observation."""
    def __init__(self, obs_dim, num_slots):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_slots)
        )
    
    def forward(self, obs):
        return self.net(obs)

# In training loop:
def compute_slot_prediction_loss(slot_predictor, obs, weights, lambda_pred=0.01):
    """
    Auxiliary loss: predict used slot from observation.
    
    This encourages different observations to use different slots,
    since the predictor needs to distinguish them.
    """
    slot_logits = slot_predictor(obs)
    target_slot = weights.argmax(dim=-1)
    
    # Cross-entropy loss
    loss = F.cross_entropy(slot_logits, target_slot)
    
    return lambda_pred * loss
```

### Expected Effect

- Forces model to use slots in a way that correlates with observations
- Implicitly encourages mode→slot correspondence (if modes differ in observations)

---

## Implemented Solutions (Command-Line Flags)

All solutions are now implemented as modular command-line options:

| Solution | Flag | Default |
|:---------|:-----|:--------|
| Temperature Annealing | `--temp_anneal True` | Off |
| Gumbel-Softmax | `--use_gumbel True` | Off |
| Hard Routing | `--hard_routing True` | Off |
| Orthogonal Init | `--orthogonal_init True` | Off |
| Contrastive Loss | `--lambda_contrast 0.01` | 0.0 (off) |
| Slot Prediction | `--slot_prediction True` | Off |
| Stronger Regularization | `--lambda_w_ent 0.1` | 0.001 |

---

## Quick Experiments

### Experiment 1: All Improvements Combined

```bash
python src/train.py \
    --total_timesteps 200000 \
    --use_gumbel True \
    --hard_routing True \
    --orthogonal_init True \
    --temp_anneal True \
    --tau_start 5.0 \
    --tau_end 0.5 \
    --lambda_contrast 0.01 \
    --slot_prediction True
```

### Experiment 2: Just Temperature Annealing

```bash
python src/train.py \
    --total_timesteps 200000 \
    --temp_anneal True \
    --tau_start 10.0 \
    --tau_end 0.5
```

### Experiment 3: Strong Regularization + Contrastive

```bash
python src/train.py \
    --total_timesteps 200000 \
    --lambda_w_ent 0.1 \
    --lambda_lb 0.1 \
    --lambda_contrast 0.05
```

### Experiment 4: Gumbel-Softmax with Hard Routing

```bash
python src/train.py \
    --total_timesteps 200000 \
    --use_gumbel True \
    --hard_routing True
```

### Experiment 5: Orthogonal Init + Slot Prediction

```bash
python src/train.py \
    --total_timesteps 200000 \
    --orthogonal_init True \
    --slot_prediction True \
    --lambda_slot_pred 0.05
```

---

## Evaluation Metrics for Interpretability

After implementing improvements, evaluate with:

### 1. Mode-Slot Correspondence
```python
# For each mode, what's the dominant slot?
# Ideal: Mode 0 → Slot 0, Mode 1 → Slot 1, Mode 2 → Slot 2
```

### 2. Weight Entropy
```python
# Higher = more uniform usage
entropy = -sum(w * log(w))
# Ideal: Entropy close to log(K) during training, lower at test time
```

### 3. Slot Usage Balance
```python
# Each slot should be used ~33% of the time
usage_per_slot = weights.mean(dim=0)
# Ideal: [0.33, 0.33, 0.33]
```

### 4. Intervention Selectivity
```python
# Clamping to the "right" slot should help, "wrong" slot should hurt
# Ideal: Clear diagonal pattern in clamp experiment heatmap
```

---

## Conclusion

Slot collapse is a common challenge in mixture models. The solutions above range from simple (regularization tweaks) to more involved (contrastive losses, Gumbel-softmax).

**Recommended first step**: Try temperature annealing (Solution 2) combined with stronger regularization (Solution 1). This requires minimal code changes and may be sufficient for the CartPole scale.

For more robust interpretability, implement Gumbel-Softmax (Solution 3) with contrastive loss (Solution 4).

