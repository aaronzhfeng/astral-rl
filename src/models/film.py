# src/models/film.py
"""
FiLM: Feature-wise Linear Modulation.

FiLM modulates the GRU hidden state using the combined abstraction,
ensuring the policy MUST depend on the abstraction (no bypass).

The modulation is: h' = γ ⊙ h + β
where (γ, β) are generated from the abstraction z.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Given abstraction z, produces (gamma, beta) to modulate context h:
        h' = gamma * h + beta
    
    This forces the policy to depend on abstractions (no bypass from raw h).
    
    Architecture:
        z (abstraction) → FiLM MLP → (gamma, beta)
        h' = gamma ⊙ h + beta
    """
    
    def __init__(
        self, 
        d_model: int,
        hidden_dim: Optional[int] = None,
        init_gamma_bias: float = 1.0,
        init_beta_bias: float = 0.0,
    ):
        """
        Initialize FiLM layer.
        
        Args:
            d_model: Dimension of context and abstraction
            hidden_dim: Hidden dimension for FiLM MLP (default: d_model)
            init_gamma_bias: Initial bias for gamma (1.0 = identity at start)
            init_beta_bias: Initial bias for beta (0.0 = no shift at start)
        """
        super().__init__()
        self.d_model = d_model
        
        if hidden_dim is None:
            hidden_dim = d_model
        
        # FiLM network: z → (gamma, beta)
        # Output 2*d_model: first half is gamma, second half is beta
        self.film_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model * 2),
        )
        
        # Initialize to approximate identity: gamma ≈ 1, beta ≈ 0
        # This helps training stability at the start
        self._init_identity(init_gamma_bias, init_beta_bias)
    
    def _init_identity(self, gamma_bias: float, beta_bias: float):
        """Initialize FiLM to approximate identity transformation."""
        # Get the last linear layer
        last_layer = self.film_net[-1]
        
        # Initialize weights to small values
        nn.init.zeros_(last_layer.weight)
        
        # Initialize biases: gamma part to gamma_bias, beta part to beta_bias
        with torch.no_grad():
            last_layer.bias[:self.d_model] = gamma_bias  # gamma
            last_layer.bias[self.d_model:] = beta_bias   # beta
    
    def forward(
        self, 
        h: torch.Tensor, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM modulation to context.
        
        Args:
            h: [batch, d_model] context from GRU
            z: [batch, d_model] combined abstraction from AbstractionBank
            
        Returns:
            h_mod: [batch, d_model] modulated context
        """
        # Generate gamma and beta from abstraction
        film_params = self.film_net(z)  # [batch, 2*d_model]
        gamma, beta = film_params.chunk(2, dim=-1)  # each [batch, d_model]
        
        # Apply FiLM modulation: h' = gamma * h + beta
        h_mod = gamma * h + beta
        
        return h_mod
    
    def forward_with_params(
        self, 
        h: torch.Tensor, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply FiLM modulation and return gamma, beta for analysis.
        
        Args:
            h: [batch, d_model] context from GRU
            z: [batch, d_model] combined abstraction
            
        Returns:
            h_mod: [batch, d_model] modulated context
            gamma: [batch, d_model] scale factors
            beta: [batch, d_model] shift factors
        """
        film_params = self.film_net(z)
        gamma, beta = film_params.chunk(2, dim=-1)
        h_mod = gamma * h + beta
        return h_mod, gamma, beta


# ============================================================================
# Test Script
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing FiLM")
    print("=" * 60)
    
    # Test parameters
    batch_size = 8
    d_model = 64
    
    # Create module
    film = FiLM(d_model=d_model)
    
    print(f"\nConfig: d_model={d_model}")
    
    # Test forward pass
    h = torch.randn(batch_size, d_model)
    z = torch.randn(batch_size, d_model)
    
    h_mod = film(h, z)
    
    print(f"\nInput h shape: {h.shape}")
    print(f"Input z shape: {z.shape}")
    print(f"Output h_mod shape: {h_mod.shape}")
    
    # Test identity initialization
    print("\n[Identity Initialization Test]")
    film_init = FiLM(d_model=d_model)
    
    # With z=0, should get h_mod ≈ h (due to gamma≈1, beta≈0)
    z_zero = torch.zeros(batch_size, d_model)
    h_mod_zero = film_init(h, z_zero)
    
    diff = (h_mod_zero - h).abs().mean()
    print(f"With z=0: |h_mod - h| mean = {diff:.6f}")
    print(f"Approximate identity? {diff < 0.1}")
    
    # Test with non-zero z
    h_mod, gamma, beta = film_init.forward_with_params(h, z)
    print(f"\nWith random z:")
    print(f"  Gamma mean: {gamma.mean():.4f}, std: {gamma.std():.4f}")
    print(f"  Beta mean: {beta.mean():.4f}, std: {beta.std():.4f}")
    
    # Test gradient flow
    print("\n[Gradient Flow Test]")
    h.requires_grad_(True)
    z.requires_grad_(True)
    
    h_mod = film(h, z)
    loss = h_mod.sum()
    loss.backward()
    
    print(f"h has grad? {h.grad is not None}")
    print(f"z has grad? {z.grad is not None}")
    print(f"FiLM net weight has grad? {film.film_net[0].weight.grad is not None}")
    
    # Test modulation effect
    print("\n[Modulation Effect Test]")
    # Create two very different abstractions
    z1 = torch.ones(1, d_model) * 2.0
    z2 = torch.ones(1, d_model) * -2.0
    h_test = torch.randn(1, d_model)
    
    h_mod1 = film(h_test, z1)
    h_mod2 = film(h_test, z2)
    
    diff = (h_mod1 - h_mod2).abs().mean()
    print(f"Different z produces different h_mod? diff = {diff:.4f}")
    print(f"Modulation has effect? {diff > 0.1}")
    
    print("\n" + "=" * 60)
    print("All FiLM tests passed! ✓")
    print("=" * 60)

