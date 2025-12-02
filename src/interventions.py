#!/usr/bin/env python3
# src/interventions.py
"""
Causal Intervention Experiments for ASTRAL.

These experiments test interpretability by:
1. Clamping weights to a single slot → Does each slot correspond to a mode?
2. Disabling slots → How much does performance drop?
3. Cross-mode transfer → Does a slot trained on one mode help another?

If abstractions are meaningful:
- Clamping to the "right" slot should improve performance on the matching mode
- Clamping to the "wrong" slot should hurt performance
- Disabling a slot should affect some modes more than others

Usage:
    python src/interventions.py --checkpoint results/runs/.../final_model.pt
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.nonstationary_cartpole import make_nonstationary_cartpole
from src.models.astral_agent import ASTRALAgent


class IntervenedAgent:
    """
    Wrapper that allows interventions on ASTRAL's abstraction mechanism.
    
    Supports:
    - clamp_slot: Force all weight to a single slot
    - disable_slot: Zero out a specific slot
    - random_weights: Use random (uniform) weights
    """
    
    def __init__(self, agent: ASTRALAgent, device: torch.device):
        self.agent = agent
        self.device = device
        self.intervention_mode = None
        self.intervention_param = None
    
    def set_intervention(self, mode: str, param: Optional[int] = None):
        """
        Set intervention mode.
        
        Args:
            mode: 'none', 'clamp_slot', 'disable_slot', 'random_weights'
            param: Slot index for clamp/disable interventions
        """
        self.intervention_mode = mode
        self.intervention_param = param
    
    def clear_intervention(self):
        """Remove any intervention."""
        self.intervention_mode = None
        self.intervention_param = None
    
    def forward_with_intervention(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: torch.Tensor,
    ):
        """Forward pass with intervention applied."""
        # Get context from GRU
        x = torch.cat([obs, prev_action, prev_reward], dim=-1)
        x = self.agent.input_proj(x)
        h = self.agent.gru(x, hidden)
        
        # Get abstraction weights (but we may override them)
        z_natural, w_natural = self.agent.abstraction_bank(h)
        
        # Apply intervention
        if self.intervention_mode == 'clamp_slot':
            # Force all weight to a single slot
            slot_idx = self.intervention_param
            w = torch.zeros_like(w_natural)
            w[:, slot_idx] = 1.0
            z = self.agent.abstraction_bank.abstractions[slot_idx].unsqueeze(0).expand(h.shape[0], -1)
        
        elif self.intervention_mode == 'disable_slot':
            # Zero out one slot and renormalize
            slot_idx = self.intervention_param
            w = w_natural.clone()
            w[:, slot_idx] = 0.0
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize
            z = torch.einsum('bk,kd->bd', w, self.agent.abstraction_bank.abstractions)
        
        elif self.intervention_mode == 'random_weights':
            # Use uniform random weights
            w = torch.ones_like(w_natural) / w_natural.shape[-1]
            z = torch.einsum('bk,kd->bd', w, self.agent.abstraction_bank.abstractions)
        
        else:
            # No intervention
            w = w_natural
            z = z_natural
        
        # FiLM modulation
        h_mod = self.agent.film(h, z)
        
        # Policy and value
        logits = self.agent.policy_head(h_mod)
        value = self.agent.value_head(h_mod).squeeze(-1)
        
        return logits, value, h, w


def evaluate_with_intervention(
    intervened_agent: IntervenedAgent,
    mode: int,
    intervention_type: str,
    intervention_param: Optional[int] = None,
    num_episodes: int = 20,
) -> Dict:
    """
    Evaluate agent with a specific intervention on a specific mode.
    
    Args:
        intervened_agent: IntervenedAgent wrapper
        mode: Environment mode (0, 1, 2)
        intervention_type: 'none', 'clamp_slot', 'disable_slot', 'random_weights'
        intervention_param: Slot index for clamp/disable
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary with returns, lengths, and weights
    """
    device = intervened_agent.device
    env = make_nonstationary_cartpole(mode=mode)
    
    intervened_agent.set_intervention(intervention_type, intervention_param)
    intervened_agent.agent.eval()
    
    returns = []
    lengths = []
    weight_histories = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        hidden = intervened_agent.agent.get_initial_hidden(1, device)
        prev_action = torch.zeros((1, 2), device=device)
        prev_reward = torch.zeros((1, 1), device=device)
        
        episode_return = 0
        episode_weights = []
        done = False
        steps = 0
        
        while not done:
            with torch.no_grad():
                logits, value, new_hidden, weights = intervened_agent.forward_with_intervention(
                    obs, prev_action, prev_reward, hidden
                )
                
                # Sample action
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
            
            episode_weights.append(weights[0].cpu().numpy())
            
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            episode_return += reward
            steps += 1
            
            obs = torch.tensor(next_obs, device=device, dtype=torch.float32).unsqueeze(0)
            prev_action = torch.zeros((1, 2), device=device)
            prev_action[0, action.item()] = 1.0
            prev_reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
            hidden = new_hidden
        
        returns.append(episode_return)
        lengths.append(steps)
        weight_histories.append(np.array(episode_weights))
    
    env.close()
    intervened_agent.clear_intervention()
    
    return {
        'returns': returns,
        'lengths': lengths,
        'weight_histories': weight_histories,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
    }


def run_clamp_experiment(
    checkpoint_path: str,
    modes: List[int] = [0, 1, 2],
    num_slots: int = 3,
    num_episodes: int = 20,
    save_dir: str = "results/interventions",
):
    """
    Run slot clamping experiment: test each slot on each mode.
    
    Creates a heatmap showing performance when clamping to each slot.
    If slots are meaningful, we expect:
    - High performance when clamping to the "right" slot for each mode
    - Lower performance when clamping to the "wrong" slot
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    agent = ASTRALAgent(obs_dim=4, action_dim=2, d_model=64, num_abstractions=num_slots).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        agent.load_state_dict(checkpoint)
    
    intervened = IntervenedAgent(agent, device)
    
    # Results matrix: [mode, slot] -> mean return
    results_matrix = np.zeros((len(modes), num_slots))
    baseline_returns = []
    
    print("\n" + "=" * 60)
    print("SLOT CLAMPING EXPERIMENT")
    print("=" * 60)
    
    # First, get baseline (no intervention)
    print("\n[Baseline - No Intervention]")
    for i, mode in enumerate(modes):
        result = evaluate_with_intervention(intervened, mode, 'none', num_episodes=num_episodes)
        baseline_returns.append(result['mean_return'])
        print(f"  Mode {mode}: {result['mean_return']:.2f} ± {result['std_return']:.2f}")
    
    # Test each slot on each mode
    print("\n[Clamping to Each Slot]")
    for i, mode in enumerate(modes):
        print(f"\nMode {mode}:")
        for slot in range(num_slots):
            result = evaluate_with_intervention(
                intervened, mode, 'clamp_slot', slot, num_episodes=num_episodes
            )
            results_matrix[i, slot] = result['mean_return']
            diff = result['mean_return'] - baseline_returns[i]
            print(f"  Slot {slot}: {result['mean_return']:.2f} (diff: {diff:+.2f})")
    
    # Plot heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap of absolute performance
    ax1 = axes[0]
    sns.heatmap(results_matrix, annot=True, fmt='.1f', cmap='viridis',
                xticklabels=[f'Slot {s}' for s in range(num_slots)],
                yticklabels=[f'Mode {m}' for m in modes],
                ax=ax1)
    ax1.set_title('Mean Return (Clamped to Single Slot)')
    ax1.set_xlabel('Clamped Slot')
    ax1.set_ylabel('Environment Mode')
    
    # Heatmap of difference from baseline
    ax2 = axes[1]
    diff_matrix = results_matrix - np.array(baseline_returns).reshape(-1, 1)
    sns.heatmap(diff_matrix, annot=True, fmt='+.1f', cmap='RdYlGn', center=0,
                xticklabels=[f'Slot {s}' for s in range(num_slots)],
                yticklabels=[f'Mode {m}' for m in modes],
                ax=ax2)
    ax2.set_title('Difference from Baseline (No Intervention)')
    ax2.set_xlabel('Clamped Slot')
    ax2.set_ylabel('Environment Mode')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'clamp_experiment.png'), dpi=150)
    plt.close()
    
    print(f"\nPlot saved to: {os.path.join(save_dir, 'clamp_experiment.png')}")
    
    return results_matrix, baseline_returns


def run_disable_experiment(
    checkpoint_path: str,
    modes: List[int] = [0, 1, 2],
    num_slots: int = 3,
    num_episodes: int = 20,
    save_dir: str = "results/interventions",
):
    """
    Run slot disabling experiment: test disabling each slot.
    
    If slots are specialized, disabling a slot should hurt performance
    on the modes that rely on it.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    agent = ASTRALAgent(obs_dim=4, action_dim=2, d_model=64, num_abstractions=num_slots).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        agent.load_state_dict(checkpoint)
    
    intervened = IntervenedAgent(agent, device)
    
    results_matrix = np.zeros((len(modes), num_slots))
    baseline_returns = []
    
    print("\n" + "=" * 60)
    print("SLOT DISABLING EXPERIMENT")
    print("=" * 60)
    
    # Baseline
    print("\n[Baseline - No Intervention]")
    for i, mode in enumerate(modes):
        result = evaluate_with_intervention(intervened, mode, 'none', num_episodes=num_episodes)
        baseline_returns.append(result['mean_return'])
        print(f"  Mode {mode}: {result['mean_return']:.2f}")
    
    # Disable each slot
    print("\n[Disabling Each Slot]")
    for i, mode in enumerate(modes):
        print(f"\nMode {mode}:")
        for slot in range(num_slots):
            result = evaluate_with_intervention(
                intervened, mode, 'disable_slot', slot, num_episodes=num_episodes
            )
            results_matrix[i, slot] = result['mean_return']
            diff = result['mean_return'] - baseline_returns[i]
            print(f"  Disable Slot {slot}: {result['mean_return']:.2f} (diff: {diff:+.2f})")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    diff_matrix = results_matrix - np.array(baseline_returns).reshape(-1, 1)
    sns.heatmap(diff_matrix, annot=True, fmt='+.1f', cmap='RdYlGn', center=0,
                xticklabels=[f'Disable Slot {s}' for s in range(num_slots)],
                yticklabels=[f'Mode {m}' for m in modes],
                ax=ax)
    ax.set_title('Performance Drop When Disabling Slot')
    ax.set_xlabel('Disabled Slot')
    ax.set_ylabel('Environment Mode')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'disable_experiment.png'), dpi=150)
    plt.close()
    
    print(f"\nPlot saved to: {os.path.join(save_dir, 'disable_experiment.png')}")
    
    return results_matrix, baseline_returns


def run_natural_weights_analysis(
    checkpoint_path: str,
    modes: List[int] = [0, 1, 2],
    num_slots: int = 3,
    num_episodes: int = 20,
    save_dir: str = "results/interventions",
):
    """
    Analyze natural weight distributions for each mode.
    
    This shows which slots the agent naturally uses for each mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    agent = ASTRALAgent(obs_dim=4, action_dim=2, d_model=64, num_abstractions=num_slots).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        agent.load_state_dict(checkpoint)
    
    intervened = IntervenedAgent(agent, device)
    
    print("\n" + "=" * 60)
    print("NATURAL WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    mode_weights = {}
    
    for mode in modes:
        print(f"\nMode {mode}:")
        result = evaluate_with_intervention(intervened, mode, 'none', num_episodes=num_episodes)
        
        # Collect all weights
        all_weights = np.concatenate([w for w in result['weight_histories']], axis=0)
        mean_weights = all_weights.mean(axis=0)
        std_weights = all_weights.std(axis=0)
        
        mode_weights[mode] = {
            'all_weights': all_weights,
            'mean': mean_weights,
            'std': std_weights,
        }
        
        print(f"  Mean weights: {mean_weights}")
        print(f"  Std weights: {std_weights}")
        print(f"  Return: {result['mean_return']:.2f}")
    
    # Plot weight distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, mode in enumerate(modes):
        ax = axes[i]
        weights = mode_weights[mode]['all_weights']
        
        for slot in range(num_slots):
            ax.hist(weights[:, slot], bins=50, alpha=0.7, label=f'Slot {slot}')
        
        ax.set_title(f'Mode {mode} Weight Distribution')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_distributions.png'), dpi=150)
    plt.close()
    
    # Summary bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(modes))
    width = 0.25
    
    for slot in range(num_slots):
        means = [mode_weights[m]['mean'][slot] for m in modes]
        ax.bar(x + slot * width, means, width, label=f'Slot {slot}')
    
    ax.set_xlabel('Mode')
    ax.set_ylabel('Mean Weight')
    ax.set_title('Mean Slot Weights per Mode')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Mode {m}' for m in modes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mean_weights_per_mode.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to: {save_dir}/")
    
    return mode_weights


def main():
    parser = argparse.ArgumentParser(description="Causal Interventions for ASTRAL")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--modes", type=int, nargs='+', default=[0, 1, 2],
                        help="Modes to test")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Episodes per condition")
    parser.add_argument("--save_dir", type=str, default="results/interventions",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Run all experiments
    print("\n" + "=" * 70)
    print("ASTRAL CAUSAL INTERVENTION EXPERIMENTS")
    print("=" * 70)
    
    # 1. Natural weight analysis
    mode_weights = run_natural_weights_analysis(
        args.checkpoint, args.modes, num_episodes=args.num_episodes, save_dir=args.save_dir
    )
    
    # 2. Clamp experiment
    clamp_results, clamp_baseline = run_clamp_experiment(
        args.checkpoint, args.modes, num_episodes=args.num_episodes, save_dir=args.save_dir
    )
    
    # 3. Disable experiment
    disable_results, disable_baseline = run_disable_experiment(
        args.checkpoint, args.modes, num_episodes=args.num_episodes, save_dir=args.save_dir
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n[Natural Weight Usage]")
    for mode in args.modes:
        print(f"  Mode {mode}: {mode_weights[mode]['mean']}")
    
    print("\n[Best Slot per Mode (Clamping)]")
    for i, mode in enumerate(args.modes):
        best_slot = np.argmax(clamp_results[i])
        best_return = clamp_results[i, best_slot]
        print(f"  Mode {mode}: Slot {best_slot} ({best_return:.2f})")
    
    print("\n[Most Important Slot per Mode (Disabling)]")
    for i, mode in enumerate(args.modes):
        # Most important = biggest drop when disabled
        drops = disable_baseline[i] - disable_results[i]
        most_important = np.argmax(drops)
        drop_amount = drops[most_important]
        print(f"  Mode {mode}: Slot {most_important} (drop: {drop_amount:.2f})")
    
    # Save raw data for regenerating plots
    data_to_save = {
        'modes': args.modes,
        'natural_weights': {
            str(mode): {
                'mean': mode_weights[mode]['mean'].tolist(),
                'std': mode_weights[mode]['std'].tolist(),
            }
            for mode in args.modes
        },
        'clamp_experiment': {
            'results_matrix': clamp_results.tolist(),
            'baseline_returns': clamp_baseline,
        },
        'disable_experiment': {
            'results_matrix': disable_results.tolist(),
            'baseline_returns': disable_baseline,
        },
    }
    
    data_path = os.path.join(args.save_dir, 'intervention_results.json')
    with open(data_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    print(f"\nData saved to: {data_path}")


if __name__ == "__main__":
    main()

