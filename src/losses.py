# src/losses.py
"""
Regularization losses for ASTRAL's abstraction bank.

These losses encourage:
1. Non-peaked weight distributions (per-sample entropy)
2. Balanced slot usage across batch (load balancing)
3. Diverse abstraction vectors (orthogonality)

Additional losses for interpretability (optional):
4. Contrastive loss: encourage different modes to use different slots
5. Slot prediction loss: auxiliary task to predict slot from observation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


def compute_abstraction_losses(
    weights: torch.Tensor,
    abstractions: torch.Tensor,
    lambda_w_ent: float = 0.001,
    lambda_lb: float = 0.001,
    lambda_orth: float = 0.0001,
) -> Dict[str, torch.Tensor]:
    """
    Compute regularization losses for abstraction bank.
    
    Args:
        weights: [batch, K] mixture weights from gating network
        abstractions: [K, d_model] abstraction vectors
        lambda_w_ent: Weight for per-sample entropy loss
        lambda_lb: Weight for load-balancing loss
        lambda_orth: Weight for orthogonality loss
        
    Returns:
        Dictionary containing:
            - L_w_ent: Per-sample entropy loss (negative, we maximize entropy)
            - L_lb: Load-balancing loss
            - L_orth: Orthogonality loss
            - total: Sum of all losses
            - entropy_mean: Mean entropy (for logging)
            - avg_weights: Average weight per slot (for logging)
    """
    K = weights.shape[-1]
    device = weights.device
    
    # 1. Per-sample entropy: encourage non-peaked distributions
    # H(w) = -sum(w * log(w))
    # We want HIGH entropy (uniform), so we minimize -entropy
    log_weights = torch.log(weights + 1e-8)
    entropy = -torch.sum(weights * log_weights, dim=-1)  # [batch]
    L_w_ent = -lambda_w_ent * entropy.mean()  # Negative to maximize entropy
    
    # 2. Load balancing: encourage uniform usage across batch
    # If all samples use the same slot, avg_weights will be peaked
    # We want high entropy of avg_weights
    avg_weights = weights.mean(dim=0)  # [K]
    log_avg_weights = torch.log(avg_weights + 1e-8)
    L_lb = lambda_lb * torch.sum(avg_weights * log_avg_weights)  # Minimize negative entropy
    
    # 3. Orthogonality: encourage diverse abstractions
    # We want A @ A.T to be close to identity (orthogonal vectors)
    A_norm = F.normalize(abstractions, dim=-1)  # [K, d]
    similarity = torch.mm(A_norm, A_norm.t())  # [K, K]
    identity = torch.eye(K, device=device)
    L_orth = lambda_orth * torch.norm(similarity - identity, p='fro') ** 2
    
    # Total loss
    total = L_w_ent + L_lb + L_orth
    
    return {
        'L_w_ent': L_w_ent,
        'L_lb': L_lb,
        'L_orth': L_orth,
        'total': total,
        'entropy_mean': entropy.mean().detach(),
        'avg_weights': avg_weights.detach().cpu().numpy(),
    }


def compute_contrastive_slot_loss(
    weights: torch.Tensor,
    modes: torch.Tensor,
    lambda_contrast: float = 0.01,
) -> torch.Tensor:
    """
    Contrastive loss: encourage different modes to use different slots.
    
    This loss minimizes the cosine similarity between weight distributions
    of different modes, encouraging mode-specific slot usage.
    
    Args:
        weights: [batch, K] mixture weights
        modes: [batch] mode labels (0, 1, 2, ...)
        lambda_contrast: regularization strength
        
    Returns:
        Contrastive loss (scalar tensor)
    """
    device = weights.device
    unique_modes = torch.unique(modes)
    
    if len(unique_modes) < 2:
        return torch.tensor(0.0, device=device)
    
    # Compute mean weight distribution per mode
    mode_weights = {}
    for mode in unique_modes:
        mask = (modes == mode)
        if mask.sum() > 0:
            mode_weights[mode.item()] = weights[mask].mean(dim=0)
    
    # Contrastive loss: minimize similarity between different modes
    loss = torch.tensor(0.0, device=device)
    count = 0
    
    mode_list = list(mode_weights.keys())
    for i in range(len(mode_list)):
        for j in range(i + 1, len(mode_list)):
            m1, m2 = mode_list[i], mode_list[j]
            w1, w2 = mode_weights[m1], mode_weights[m2]
            
            # Cosine similarity (we want this to be low = different)
            similarity = F.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0))
            loss = loss + similarity
            count += 1
    
    if count > 0:
        loss = loss / count
    
    return lambda_contrast * loss


class SlotPredictor(nn.Module):
    """
    Auxiliary module to predict which slot was used from observation.
    
    This encourages the model to use slots in a way that correlates
    with observable features, improving interpretability.
    
    If the predictor can guess the slot from the observation alone,
    it means different observations use different slots (good for interpretability).
    """
    
    def __init__(self, obs_dim: int, num_slots: int, hidden_dim: int = 64):
        """
        Initialize slot predictor.
        
        Args:
            obs_dim: Dimension of observation
            num_slots: Number of abstraction slots (K)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_slots),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict slot logits from observation.
        
        Args:
            obs: [batch, obs_dim] observations
            
        Returns:
            logits: [batch, K] slot prediction logits
        """
        return self.net(obs)


def compute_slot_prediction_loss(
    slot_predictor: SlotPredictor,
    obs: torch.Tensor,
    weights: torch.Tensor,
    lambda_pred: float = 0.01,
) -> torch.Tensor:
    """
    Auxiliary loss: predict used slot from observation.
    
    This encourages different observations to use different slots,
    since the predictor needs to distinguish them.
    
    Args:
        slot_predictor: SlotPredictor module
        obs: [batch, obs_dim] observations
        weights: [batch, K] mixture weights
        lambda_pred: regularization strength
        
    Returns:
        Slot prediction loss (scalar tensor)
    """
    # Predict slot from observation
    slot_logits = slot_predictor(obs)  # [batch, K]
    
    # Target: which slot had highest weight
    target_slot = weights.argmax(dim=-1)  # [batch]
    
    # Cross-entropy loss
    loss = F.cross_entropy(slot_logits, target_slot)
    
    return lambda_pred * loss


def compute_slot_prediction_accuracy(
    slot_predictor: SlotPredictor,
    obs: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """
    Compute slot prediction accuracy (for logging).
    
    Args:
        slot_predictor: SlotPredictor module
        obs: [batch, obs_dim] observations
        weights: [batch, K] mixture weights
        
    Returns:
        Accuracy (float between 0 and 1)
    """
    with torch.no_grad():
        slot_logits = slot_predictor(obs)
        predicted_slot = slot_logits.argmax(dim=-1)
        target_slot = weights.argmax(dim=-1)
        accuracy = (predicted_slot == target_slot).float().mean().item()
    return accuracy


# ============================================================================
# Test Script
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Abstraction Losses (with improvements)")
    print("=" * 60)
    
    batch_size = 32
    K = 3
    d_model = 64
    obs_dim = 4
    
    # Test 1: Standard losses (uniform weights)
    print("\n[Test 1: Standard Losses - Uniform weights]")
    weights_uniform = torch.ones(batch_size, K) / K
    abstractions = torch.randn(K, d_model)
    
    losses = compute_abstraction_losses(weights_uniform, abstractions)
    print(f"Entropy: {losses['entropy_mean']:.4f} (max: {np.log(K):.4f})")
    print(f"L_w_ent: {losses['L_w_ent']:.6f}")
    print(f"L_lb: {losses['L_lb']:.6f}")
    print(f"L_orth: {losses['L_orth']:.6f}")
    
    # Test 2: Contrastive loss
    print("\n[Test 2: Contrastive Loss]")
    # Create weights that are mode-specific (good)
    weights_diverse = torch.zeros(batch_size, K)
    modes = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        mode = i % 3
        modes[i] = mode
        weights_diverse[i, mode] = 1.0  # Each mode uses its own slot
    
    loss_diverse = compute_contrastive_slot_loss(weights_diverse, modes, lambda_contrast=1.0)
    print(f"Contrastive loss (diverse): {loss_diverse.item():.4f} (should be low)")
    
    # Create weights that are same for all modes (bad)
    weights_same = torch.ones(batch_size, K) / K
    loss_same = compute_contrastive_slot_loss(weights_same, modes, lambda_contrast=1.0)
    print(f"Contrastive loss (same): {loss_same.item():.4f} (should be high)")
    
    # Test 3: Slot predictor
    print("\n[Test 3: Slot Predictor]")
    predictor = SlotPredictor(obs_dim=obs_dim, num_slots=K)
    obs = torch.randn(batch_size, obs_dim)
    weights = torch.softmax(torch.randn(batch_size, K), dim=-1)
    
    pred_loss = compute_slot_prediction_loss(predictor, obs, weights, lambda_pred=1.0)
    accuracy = compute_slot_prediction_accuracy(predictor, obs, weights)
    print(f"Prediction loss: {pred_loss:.4f}")
    print(f"Accuracy: {accuracy:.2%} (random baseline: {1/K:.2%})")
    
    # Test 4: Gradient flow
    print("\n[Test 4: Gradient Flow]")
    weights = torch.softmax(torch.randn(batch_size, K, requires_grad=True), dim=-1)
    modes = torch.randint(0, 3, (batch_size,))
    
    contrast_loss = compute_contrastive_slot_loss(weights, modes)
    contrast_loss.backward()
    print(f"Contrastive loss has grad? {weights.grad is not None}")
    
    print("\n" + "=" * 60)
    print("All loss tests passed! ✓")
    print("=" * 60)
