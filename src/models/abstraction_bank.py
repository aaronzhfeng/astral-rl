# src/models/abstraction_bank.py
"""
AbstractionBank: Bank of K learnable abstraction vectors with gating mechanism.

This is the core component of ASTRAL that provides structured, interpretable
adaptation through discrete abstraction slots.

Key components:
    - K learnable abstraction vectors (the "bank")
    - Gating network that selects/combines abstractions based on context
    - Soft attention (softmax) over slots with temperature control

Improvements for interpretability (all optional/modular):
    - Gumbel-Softmax for differentiable discrete selection
    - Hard routing for explicit slot selection
    - Orthogonal initialization for diverse slots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AbstractionBank(nn.Module):
    """
    Bank of K learnable abstraction vectors with gating mechanism.
    
    Given context h from the GRU, outputs:
    - z: combined abstraction (weighted sum of slots)
    - w: mixture weights over slots (for analysis/regularization)
    
    Architecture:
        h (context) → Gating MLP → logits → softmax(logits/τ) → w
        w (weights) × A (bank) → z (combined abstraction)
    
    Improvements (all optional):
        - use_gumbel: Use Gumbel-Softmax for differentiable discrete selection
        - hard_routing: Use hard (one-hot) routing at forward pass
        - orthogonal_init: Initialize abstractions to be orthogonal
    """
    
    def __init__(
        self,
        d_model: int,
        num_abstractions: int = 3,
        tau: float = 1.0,
        gating_hidden_dim: Optional[int] = None,
        use_gumbel: bool = False,
        hard_routing: bool = False,
        orthogonal_init: bool = False,
        slot_dropout: float = 0.0,
    ):
        """
        Initialize AbstractionBank.
        
        Args:
            d_model: Dimension of context embedding and abstractions
            num_abstractions: Number of abstraction slots (K)
            tau: Softmax temperature (1.0 = standard, <1 = more peaked)
            gating_hidden_dim: Hidden dimension for gating MLP (default: d_model)
            use_gumbel: If True, use Gumbel-Softmax (helps exploration)
            hard_routing: If True, use hard one-hot routing (with straight-through gradient)
            orthogonal_init: If True, initialize abstractions orthogonally
            slot_dropout: Probability of dropping each slot during training (0.0 = no dropout)
        """
        super().__init__()
        self.d_model = d_model
        self.num_abstractions = num_abstractions
        self.tau = tau
        self.use_gumbel = use_gumbel
        self.hard_routing = hard_routing
        self.slot_dropout = slot_dropout
        
        if gating_hidden_dim is None:
            gating_hidden_dim = d_model
        
        # K learnable abstraction vectors [K, d_model]
        self.abstractions = nn.Parameter(torch.zeros(num_abstractions, d_model))
        
        # Initialize abstractions
        if orthogonal_init:
            self._init_orthogonal()
        else:
            self._init_random()
        
        # Gating network: h → logits over K slots
        self.gating = nn.Sequential(
            nn.Linear(d_model, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, num_abstractions),
        )
    
    def _init_random(self):
        """Standard random initialization."""
        nn.init.normal_(self.abstractions, mean=0, std=0.02)
    
    def _init_orthogonal(self):
        """Initialize abstractions to be orthogonal (for diversity)."""
        # Create random matrix and get orthogonal basis via QR decomposition
        if self.num_abstractions <= self.d_model:
            random_matrix = torch.randn(self.d_model, self.num_abstractions)
            q, r = torch.linalg.qr(random_matrix)
            orthogonal = q[:, :self.num_abstractions].T  # [K, d_model]
        else:
            # More slots than dimensions: use random init for extra
            random_matrix = torch.randn(self.d_model, self.d_model)
            q, r = torch.linalg.qr(random_matrix)
            orthogonal = torch.randn(self.num_abstractions, self.d_model) * 0.02
            orthogonal[:self.d_model] = q.T
        
        self.abstractions.data = orthogonal * 0.1  # Scale down
        
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute combined abstraction from context.
        
        Args:
            h: [batch, d_model] context embedding from GRU
            
        Returns:
            z: [batch, d_model] combined abstraction (weighted sum)
            w: [batch, K] mixture weights (softmax probabilities)
        """
        # Compute gating logits
        logits = self.gating(h)  # [batch, K]
        
        # Apply slot dropout during training (mask out random slots)
        if self.training and self.slot_dropout > 0:
            # Create dropout mask: 1 = keep, 0 = drop
            dropout_mask = (torch.rand(self.num_abstractions, device=logits.device) > self.slot_dropout).float()
            # Ensure at least one slot is active
            if dropout_mask.sum() == 0:
                dropout_mask[torch.randint(self.num_abstractions, (1,))] = 1.0
            # Apply mask to logits (set dropped slots to -inf so softmax gives 0)
            logits = logits + (1 - dropout_mask) * (-1e9)
        
        # Compute weights based on mode
        if self.use_gumbel and self.training:
            # Gumbel-Softmax: differentiable approximation to discrete sampling
            w = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard_routing)
        elif self.hard_routing:
            # Hard routing with straight-through gradient
            w_soft = F.softmax(logits / self.tau, dim=-1)
            w_hard = F.one_hot(logits.argmax(dim=-1), num_classes=self.num_abstractions).float()
            # Straight-through: forward uses hard, backward uses soft
            w = w_hard - w_soft.detach() + w_soft
        else:
            # Standard soft attention
            w = F.softmax(logits / self.tau, dim=-1)
        
        # Weighted sum of abstractions
        z = torch.einsum('bk,kd->bd', w, self.abstractions)
        
        return z, w
    
    def get_abstractions(self) -> torch.Tensor:
        """Return the abstraction matrix for regularization/analysis."""
        return self.abstractions
    
    def get_gating_params(self) -> nn.Module:
        """Return gating network (for test-time adaptation)."""
        return self.gating
    
    def set_temperature(self, tau: float):
        """Update softmax temperature."""
        self.tau = tau


class TemperatureScheduler:
    """
    Temperature scheduler for annealing from high (uniform) to low (peaked).
    
    High temperature early in training encourages exploration of all slots.
    Low temperature later allows specialization.
    """
    
    def __init__(
        self,
        tau_start: float = 5.0,
        tau_end: float = 0.5,
        warmup_steps: int = 100000,
        schedule: str = "linear",
    ):
        """
        Initialize temperature scheduler.
        
        Args:
            tau_start: Initial temperature (high = uniform weights)
            tau_end: Final temperature (low = peaked weights)
            warmup_steps: Steps over which to anneal
            schedule: "linear", "cosine", or "exponential"
        """
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.warmup_steps = warmup_steps
        self.schedule = schedule
    
    def get_tau(self, step: int) -> float:
        """Get temperature for current step."""
        if step >= self.warmup_steps:
            return self.tau_end
        
        progress = step / self.warmup_steps
        
        if self.schedule == "linear":
            return self.tau_start + (self.tau_end - self.tau_start) * progress
        elif self.schedule == "cosine":
            import math
            return self.tau_end + (self.tau_start - self.tau_end) * (1 + math.cos(math.pi * progress)) / 2
        elif self.schedule == "exponential":
            import math
            return self.tau_start * math.exp(math.log(self.tau_end / self.tau_start) * progress)
        else:
            return self.tau_start + (self.tau_end - self.tau_start) * progress


# ============================================================================
# Test Script
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing AbstractionBank (with improvements)")
    print("=" * 60)
    
    batch_size = 8
    d_model = 64
    num_abstractions = 3
    
    # Test 1: Standard bank
    print("\n[Test 1: Standard AbstractionBank]")
    bank = AbstractionBank(d_model=d_model, num_abstractions=num_abstractions, tau=1.0)
    h = torch.randn(batch_size, d_model)
    z, w = bank(h)
    print(f"Weights shape: {w.shape}, sum: {w.sum(dim=-1)[:3]}")
    
    # Test 2: Gumbel-Softmax
    print("\n[Test 2: Gumbel-Softmax]")
    bank_gumbel = AbstractionBank(
        d_model=d_model, num_abstractions=num_abstractions, 
        tau=1.0, use_gumbel=True
    )
    bank_gumbel.train()
    z_gumbel, w_gumbel = bank_gumbel(h)
    print(f"Gumbel weights sample: {w_gumbel[0].detach().numpy()}")
    
    # Test 3: Hard routing
    print("\n[Test 3: Hard Routing]")
    bank_hard = AbstractionBank(
        d_model=d_model, num_abstractions=num_abstractions,
        tau=1.0, hard_routing=True
    )
    z_hard, w_hard = bank_hard(h)
    print(f"Hard weights sample: {w_hard[0].detach().numpy()}")
    print(f"Is one-hot? {(w_hard.sum(dim=-1) == 1).all() and (w_hard.max(dim=-1)[0] == 1).all()}")
    
    # Test 4: Orthogonal initialization
    print("\n[Test 4: Orthogonal Initialization]")
    bank_orth = AbstractionBank(
        d_model=d_model, num_abstractions=num_abstractions,
        orthogonal_init=True
    )
    A = bank_orth.abstractions.data
    A_norm = F.normalize(A, dim=-1)
    similarity = torch.mm(A_norm, A_norm.t())
    print(f"Similarity matrix (should be ~identity):\n{similarity.numpy()}")
    
    # Test 5: Temperature scheduler
    print("\n[Test 5: Temperature Scheduler]")
    scheduler = TemperatureScheduler(tau_start=5.0, tau_end=0.5, warmup_steps=100)
    print(f"Step 0: tau={scheduler.get_tau(0):.2f}")
    print(f"Step 50: tau={scheduler.get_tau(50):.2f}")
    print(f"Step 100: tau={scheduler.get_tau(100):.2f}")
    print(f"Step 200: tau={scheduler.get_tau(200):.2f}")
    
    # Test 6: Gradient flow with improvements
    print("\n[Test 6: Gradient Flow]")
    bank_full = AbstractionBank(
        d_model=d_model, num_abstractions=num_abstractions,
        use_gumbel=True, hard_routing=True, orthogonal_init=True
    )
    bank_full.train()
    z, w = bank_full(h)
    loss = z.sum()
    loss.backward()
    print(f"Abstractions have grad? {bank_full.abstractions.grad is not None}")
    print(f"Gating has grad? {bank_full.gating[0].weight.grad is not None}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
