# src/models/astral_agent.py
"""
ASTRAL Agent: GRU backbone + Abstraction Bank + FiLM + Policy/Value heads.

ASTRAL = Abstraction-Slot Test-time Reweighting for Adaptation in Latent RL

Key architectural decisions:
1. GRU produces context h_t from (s, a, r) history (in-context adaptation)
2. AbstractionBank produces z_t and weights w_t from h_t
3. FiLM modulates h_t with z_t to get h'_t (forces abstraction dependency)
4. Policy and Value heads see ONLY h'_t (no bypass from raw h_t)

This file also includes BaselineAgent (GRU-only) for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict, Any

from .abstraction_bank import AbstractionBank, TemperatureScheduler
from .film import FiLM


class ASTRALAgent(nn.Module):
    """
    ASTRAL Agent with abstraction bank and FiLM modulation.
    
    Architecture:
        (s_t, a_{t-1}, r_{t-1}) → Input MLP → GRU → h_t
        h_t → AbstractionBank → (z_t, w_t)
        (h_t, z_t) → FiLM → h'_t
        h'_t → Policy Head → action logits
        h'_t → Value Head → value
    
    Key: Only h'_t (modulated) reaches the heads, not raw h_t.
    
    Interpretability improvements (all optional):
        - use_gumbel: Gumbel-Softmax for exploration
        - hard_routing: Discrete slot selection
        - orthogonal_init: Diverse slot initialization
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 64,
        num_abstractions: int = 3,
        tau: float = 1.0,
        # Interpretability improvements (all optional)
        use_gumbel: bool = False,
        hard_routing: bool = False,
        orthogonal_init: bool = False,
        slot_dropout: float = 0.0,
    ):
        """
        Initialize ASTRAL agent.
        
        Args:
            obs_dim: Dimension of observation (4 for CartPole)
            action_dim: Number of actions (2 for CartPole)
            d_model: Hidden dimension for GRU and abstractions
            num_abstractions: Number of abstraction slots (K)
            tau: Softmax temperature for abstraction gating
            use_gumbel: Use Gumbel-Softmax for differentiable discrete selection
            hard_routing: Use hard (one-hot) routing
            orthogonal_init: Initialize abstractions orthogonally
            slot_dropout: Probability of dropping each slot during training
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_abstractions = num_abstractions
        
        # Input: (obs, prev_action_onehot, prev_reward)
        input_dim = obs_dim + action_dim + 1
        
        # Input projection: raw input → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
        )
        
        # GRU backbone (in-context adaptation via hidden state)
        self.gru = nn.GRUCell(d_model, d_model)
        
        # Abstraction bank with optional improvements
        self.abstraction_bank = AbstractionBank(
            d_model=d_model,
            num_abstractions=num_abstractions,
            tau=tau,
            use_gumbel=use_gumbel,
            hard_routing=hard_routing,
            orthogonal_init=orthogonal_init,
            slot_dropout=slot_dropout,
        )
        
        # FiLM modulation (forces dependency on abstraction)
        self.film = FiLM(d_model=d_model)
        
        # Policy head: h'_t → action logits
        # Note: Takes h'_t (modulated), NOT raw h_t
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )
        
        # Value head: h'_t → scalar value
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step forward pass.
        
        Args:
            obs: [batch, obs_dim] current observation
            prev_action: [batch, action_dim] one-hot previous action
            prev_reward: [batch, 1] previous reward
            hidden: [batch, d_model] GRU hidden state
            
        Returns:
            logits: [batch, action_dim] action logits
            value: [batch] state value
            new_hidden: [batch, d_model] updated GRU hidden
            weights: [batch, K] abstraction mixture weights
        """
        # Concatenate inputs
        x = torch.cat([obs, prev_action, prev_reward], dim=-1)
        x = self.input_proj(x)
        
        # GRU update
        h = self.gru(x, hidden)
        
        # Abstraction bank: get combined abstraction and weights
        z, w = self.abstraction_bank(h)
        
        # FiLM modulation: h' = gamma * h + beta
        # This is the key anti-bypass mechanism
        h_mod = self.film(h, z)
        
        # Policy and value from MODULATED context only
        logits = self.policy_head(h_mod)
        value = self.value_head(h_mod).squeeze(-1)
        
        return logits, value, h, w
    
    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize GRU hidden state to zeros."""
        return torch.zeros(batch_size, self.d_model, device=device)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log prob, entropy, value, and new hidden.
        Used during rollout and PPO update.
        
        Args:
            obs: [batch, obs_dim]
            prev_action: [batch, action_dim] one-hot
            prev_reward: [batch, 1]
            hidden: [batch, d_model]
            action: [batch] optional action (for computing log_prob during update)
            
        Returns:
            action: [batch] sampled or provided action
            log_prob: [batch] log probability of action
            entropy: [batch] policy entropy
            value: [batch] state value
            new_hidden: [batch, d_model] updated hidden state
            weights: [batch, K] abstraction weights
        """
        logits, value, new_hidden, weights = self.forward(
            obs, prev_action, prev_reward, hidden
        )
        
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            value,
            new_hidden,
            weights,
        )
    
    def get_gating_parameters(self):
        """Return only the gating network parameters (for test-time adaptation)."""
        return self.abstraction_bank.gating.parameters()
    
    def freeze_except_gating(self):
        """Freeze all parameters except the gating network."""
        for name, param in self.named_parameters():
            if 'gating' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class BaselineAgent(nn.Module):
    """
    Baseline: GRU-only agent without abstraction bank.
    
    Same architecture as ASTRAL but h_t goes directly to heads.
    This serves as the comparison baseline to show abstraction bank value.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 64,
    ):
        """
        Initialize baseline GRU agent.
        
        Args:
            obs_dim: Dimension of observation
            action_dim: Number of actions
            d_model: Hidden dimension
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        
        input_dim = obs_dim + action_dim + 1
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
        )
        
        self.gru = nn.GRUCell(d_model, d_model)
        
        # Note: Heads take raw h, not modulated h'
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass without abstraction bank."""
        x = torch.cat([obs, prev_action, prev_reward], dim=-1)
        x = self.input_proj(x)
        h = self.gru(x, hidden)
        
        # Direct path: h → heads (no abstraction, no FiLM)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        
        return logits, value, h
    
    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_model, device=device)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Get action and value (weights is None for baseline)."""
        logits, value, new_hidden = self.forward(
            obs, prev_action, prev_reward, hidden
        )
        
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            value,
            new_hidden,
            None,  # No weights for baseline
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Test Script
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing ASTRAL Agent")
    print("=" * 60)
    
    # Test parameters (CartPole)
    obs_dim = 4
    action_dim = 2
    d_model = 64
    num_abstractions = 3
    batch_size = 8
    
    device = torch.device("cpu")
    
    # Create agents
    astral = ASTRALAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_model=d_model,
        num_abstractions=num_abstractions,
    ).to(device)
    
    baseline = BaselineAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_model=d_model,
    ).to(device)
    
    print(f"\nASTRAL parameters: {count_parameters(astral):,}")
    print(f"Baseline parameters: {count_parameters(baseline):,}")
    print(f"Overhead: {count_parameters(astral) - count_parameters(baseline):,}")
    
    # Test inputs
    obs = torch.randn(batch_size, obs_dim, device=device)
    prev_action = torch.zeros(batch_size, action_dim, device=device)
    prev_action[:, 0] = 1  # One-hot
    prev_reward = torch.zeros(batch_size, 1, device=device)
    
    # Test ASTRAL
    print("\n[ASTRAL Agent Test]")
    hidden = astral.get_initial_hidden(batch_size, device)
    print(f"Initial hidden shape: {hidden.shape}")
    
    action, log_prob, entropy, value, new_hidden, weights = astral.get_action_and_value(
        obs, prev_action, prev_reward, hidden
    )
    
    print(f"Action shape: {action.shape}, values: {action[:3].tolist()}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Entropy shape: {entropy.shape}, mean: {entropy.mean():.4f}")
    print(f"Value shape: {value.shape}")
    print(f"New hidden shape: {new_hidden.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights sample: {weights[0].detach().numpy()}")
    print(f"Weights sum to 1? {torch.allclose(weights.sum(dim=-1), torch.ones(batch_size))}")
    
    # Test Baseline
    print("\n[Baseline Agent Test]")
    hidden = baseline.get_initial_hidden(batch_size, device)
    
    action, log_prob, entropy, value, new_hidden, weights = baseline.get_action_and_value(
        obs, prev_action, prev_reward, hidden
    )
    
    print(f"Action shape: {action.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Weights: {weights}")  # Should be None
    
    # Test gradient flow
    print("\n[Gradient Flow Test - ASTRAL]")
    hidden = astral.get_initial_hidden(batch_size, device)
    
    logits, value, new_hidden, weights = astral.forward(
        obs, prev_action, prev_reward, hidden
    )
    
    loss = logits.sum() + value.sum()
    loss.backward()
    
    print(f"GRU weight has grad? {astral.gru.weight_hh.grad is not None}")
    print(f"Abstraction bank has grad? {astral.abstraction_bank.abstractions.grad is not None}")
    print(f"FiLM has grad? {astral.film.film_net[0].weight.grad is not None}")
    print(f"Policy head has grad? {astral.policy_head[0].weight.grad is not None}")
    
    # Test freeze_except_gating
    print("\n[Test-Time Adaptation Setup Test]")
    astral.zero_grad()
    astral.freeze_except_gating()
    
    trainable = sum(1 for p in astral.parameters() if p.requires_grad)
    total = sum(1 for p in astral.parameters())
    print(f"Trainable after freeze: {trainable}/{total}")
    
    # Check which params are trainable
    for name, param in astral.named_parameters():
        if param.requires_grad:
            print(f"  Trainable: {name}")
    
    astral.unfreeze_all()
    trainable = sum(1 for p in astral.parameters() if p.requires_grad)
    print(f"Trainable after unfreeze: {trainable}/{total}")
    
    # Test multi-step rollout
    print("\n[Multi-Step Rollout Test]")
    astral.zero_grad()
    hidden = astral.get_initial_hidden(batch_size, device)
    prev_action = torch.zeros(batch_size, action_dim, device=device)
    prev_reward = torch.zeros(batch_size, 1, device=device)
    
    all_weights = []
    for step in range(10):
        obs = torch.randn(batch_size, obs_dim, device=device)
        action, log_prob, entropy, value, hidden, weights = astral.get_action_and_value(
            obs, prev_action, prev_reward, hidden
        )
        all_weights.append(weights)
        
        # Update for next step
        prev_action = torch.zeros(batch_size, action_dim, device=device)
        prev_action.scatter_(1, action.unsqueeze(1), 1.0)
        prev_reward = torch.randn(batch_size, 1, device=device)
    
    all_weights = torch.stack(all_weights)  # [steps, batch, K]
    print(f"Weights over 10 steps shape: {all_weights.shape}")
    print(f"Weight variance across steps: {all_weights.var(dim=0).mean():.4f}")
    
    print("\n" + "=" * 60)
    print("All Agent tests passed! ✓")
    print("=" * 60)

