#!/usr/bin/env python3
# src/test_time_adapt.py
"""
Test-Time Adaptation (TTA) for ASTRAL.

At test time, we freeze all parameters except the gating network,
then adapt only the gating network on a fixed environment mode.

This tests whether ASTRAL can quickly adapt to a specific mode
by adjusting which abstraction slot it uses.

Supports multiple adaptation modes for comparison:
- gating: Only adapt gating network (ASTRAL only)
- policy_head: Only adapt policy head (works for both ASTRAL and Baseline)
- all_params: Adapt all parameters (full fine-tuning)

Usage:
    python src/test_time_adapt.py --checkpoint results/runs/.../final_model.pt --mode 2
    python src/test_time_adapt.py --checkpoint results/runs/.../final_model.pt --adapt_mode policy_head
"""

import os
import sys
import argparse
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Union

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.nonstationary_cartpole import make_nonstationary_cartpole
from src.models.astral_agent import ASTRALAgent, BaselineAgent


def load_agent(
    checkpoint_path: str,
    agent_type: str = "auto",
    device: torch.device = None,
) -> Tuple[Union[ASTRALAgent, BaselineAgent], str]:
    """
    Load agent from checkpoint with auto-detection of agent type.
    
    Args:
        checkpoint_path: Path to model checkpoint
        agent_type: "auto", "astral", or "baseline"
        device: Torch device
        
    Returns:
        Tuple of (agent, detected_type)
    """
    if device is None:
        device = torch.device("cpu")
    
    # Auto-detect from path
    if agent_type == "auto":
        if "baseline" in checkpoint_path.lower():
            agent_type = "baseline"
        else:
            agent_type = "astral"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if agent_type == "baseline":
        agent = BaselineAgent(obs_dim=4, action_dim=2, d_model=64).to(device)
    else:
        agent = ASTRALAgent(
            obs_dim=4, action_dim=2, d_model=64, num_abstractions=3
        ).to(device)
    
    # Load weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        agent.load_state_dict(checkpoint)
    
    return agent, agent_type


def evaluate_agent(
    agent: ASTRALAgent,
    mode: int,
    num_episodes: int = 20,
    device: torch.device = None,
    render: bool = False,
) -> Dict[str, List[float]]:
    """
    Evaluate agent on a fixed mode without adaptation.
    
    Args:
        agent: ASTRAL or Baseline agent
        mode: Fixed environment mode (0, 1, or 2)
        num_episodes: Number of evaluation episodes
        device: Torch device
        render: Whether to render (not supported for CartPole)
        
    Returns:
        Dictionary with returns, lengths, and weight histories
    """
    if device is None:
        device = torch.device("cpu")
    
    env = make_nonstationary_cartpole(mode=mode)
    
    agent.eval()
    
    returns = []
    lengths = []
    weight_histories = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros((1, 2), device=device)  # 2 actions for CartPole
        prev_reward = torch.zeros((1, 1), device=device)
        
        episode_return = 0
        episode_weights = []
        done = False
        steps = 0
        
        while not done:
            with torch.no_grad():
                action, _, _, _, new_hidden, weights = agent.get_action_and_value(
                    obs, prev_action, prev_reward, hidden
                )
            
            if weights is not None:
                episode_weights.append(weights[0].cpu().numpy())
            
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            episode_return += reward
            steps += 1
            
            # Update for next step
            obs = torch.tensor(next_obs, device=device, dtype=torch.float32).unsqueeze(0)
            prev_action = torch.zeros((1, 2), device=device)
            prev_action[0, action.item()] = 1.0
            prev_reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
            hidden = new_hidden
        
        returns.append(episode_return)
        lengths.append(steps)
        if episode_weights:
            weight_histories.append(np.array(episode_weights))
    
    env.close()
    
    return {
        'returns': returns,
        'lengths': lengths,
        'weight_histories': weight_histories,
    }


def test_time_adapt(
    agent: ASTRALAgent,
    target_mode: int,
    num_adapt_episodes: int = 20,
    adapt_lr: float = 1e-3,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Perform test-time adaptation on the gating network only.
    
    Uses REINFORCE (policy gradient) to adapt the gating network
    on a fixed environment mode.
    
    Args:
        agent: Trained ASTRAL agent
        target_mode: Fixed mode to adapt to
        num_adapt_episodes: Number of adaptation episodes
        adapt_lr: Learning rate for gating network
        device: Torch device
        verbose: Print progress
        
    Returns:
        Dictionary with returns during adaptation and weight histories
    """
    if device is None:
        device = torch.device("cpu")
    
    # Freeze everything except gating network
    agent.freeze_except_gating()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    if verbose:
        print(f"Test-time adaptation: {trainable_params} trainable parameters (gating only)")
    
    # Optimizer for gating network only
    gating_params = list(agent.get_gating_parameters())
    optimizer = optim.Adam(gating_params, lr=adapt_lr)
    
    # Create environment with fixed mode
    env = make_nonstationary_cartpole(mode=target_mode)
    
    returns = []
    weight_histories = []
    
    for episode in range(num_adapt_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros((1, 2), device=device)
        prev_reward = torch.zeros((1, 1), device=device)
        
        episode_return = 0
        episode_log_probs = []
        episode_rewards = []
        episode_weights = []
        
        done = False
        
        while not done:
            # Need gradients for adaptation
            action, log_prob, _, _, new_hidden, weights = agent.get_action_and_value(
                obs, prev_action, prev_reward, hidden
            )
            
            # Store for REINFORCE
            episode_log_probs.append(log_prob)
            if weights is not None:
                episode_weights.append(weights[0].detach().cpu().numpy())
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_return += reward
            
            # Update for next step
            obs = torch.tensor(next_obs, device=device, dtype=torch.float32).unsqueeze(0)
            prev_action = torch.zeros((1, 2), device=device)
            prev_action[0, action.item()] = 1.0
            prev_reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
            hidden = new_hidden.detach()  # Detach hidden to avoid backprop through time
        
        returns.append(episode_return)
        if episode_weights:
            weight_histories.append(np.array(episode_weights))
        
        # REINFORCE update on gating network
        # Compute discounted returns
        G = 0
        discounted_returns = []
        for r in reversed(episode_rewards):
            G = r + 0.99 * G
            discounted_returns.insert(0, G)
        
        discounted_returns = torch.tensor(discounted_returns, device=device, dtype=torch.float32)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)
        
        # Policy gradient loss
        policy_loss = []
        for log_prob, G in zip(episode_log_probs, discounted_returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        if verbose and (episode + 1) % 5 == 0:
            mean_weight = np.mean([w.mean(axis=0) for w in weight_histories[-5:]], axis=0)
            print(f"  Episode {episode + 1}: Return = {episode_return:.0f}, Avg weights = {mean_weight}")
    
    env.close()
    
    # Restore all parameters to trainable
    agent.unfreeze_all()
    
    return {
        'returns': returns,
        'weight_histories': weight_histories,
    }


def test_time_adapt_policy_head(
    agent: nn.Module,
    target_mode: int,
    num_adapt_episodes: int = 20,
    adapt_lr: float = 1e-3,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Adapt policy head only (for baseline comparison).
    Works with both ASTRAL and Baseline agents.
    
    Args:
        agent: ASTRAL or Baseline agent
        target_mode: Fixed mode to adapt to
        num_adapt_episodes: Number of adaptation episodes
        adapt_lr: Learning rate for policy head
        device: Torch device
        verbose: Print progress
        
    Returns:
        Dictionary with returns during adaptation
    """
    if device is None:
        device = torch.device("cpu")
    
    # Freeze everything except policy head
    for param in agent.parameters():
        param.requires_grad = False
    for param in agent.policy_head.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    if verbose:
        print(f"Test-time adaptation: {trainable_params} trainable parameters (policy_head only)")
    
    optimizer = optim.Adam(agent.policy_head.parameters(), lr=adapt_lr)
    
    env = make_nonstationary_cartpole(mode=target_mode)
    
    returns = []
    
    for episode in range(num_adapt_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros((1, 2), device=device)
        prev_reward = torch.zeros((1, 1), device=device)
        
        episode_return = 0
        episode_log_probs = []
        episode_rewards = []
        
        done = False
        
        while not done:
            action, log_prob, _, _, new_hidden, _ = agent.get_action_and_value(
                obs, prev_action, prev_reward, hidden
            )
            
            episode_log_probs.append(log_prob)
            
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_return += reward
            
            obs = torch.tensor(next_obs, device=device, dtype=torch.float32).unsqueeze(0)
            prev_action = torch.zeros((1, 2), device=device)
            prev_action[0, action.item()] = 1.0
            prev_reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
            hidden = new_hidden.detach()
        
        returns.append(episode_return)
        
        # REINFORCE update
        G = 0
        discounted_returns = []
        for r in reversed(episode_rewards):
            G = r + 0.99 * G
            discounted_returns.insert(0, G)
        
        discounted_returns = torch.tensor(discounted_returns, device=device, dtype=torch.float32)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, G in zip(episode_log_probs, discounted_returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        if verbose and (episode + 1) % 5 == 0:
            print(f"  Episode {episode + 1}: Return = {episode_return:.0f}")
    
    env.close()
    
    # Restore all parameters to trainable
    for param in agent.parameters():
        param.requires_grad = True
    
    return {
        'returns': returns,
        'weight_histories': [],
    }


def test_time_adapt_all_params(
    agent: nn.Module,
    target_mode: int,
    num_adapt_episodes: int = 20,
    adapt_lr: float = 1e-4,  # Lower LR for full fine-tuning
    device: torch.device = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Adapt all parameters (full fine-tuning).
    
    Args:
        agent: ASTRAL or Baseline agent
        target_mode: Fixed mode to adapt to
        num_adapt_episodes: Number of adaptation episodes
        adapt_lr: Learning rate (lower for full fine-tuning)
        device: Torch device
        verbose: Print progress
        
    Returns:
        Dictionary with returns during adaptation
    """
    if device is None:
        device = torch.device("cpu")
    
    # All parameters trainable
    for param in agent.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    if verbose:
        print(f"Test-time adaptation: {trainable_params} trainable parameters (all_params)")
    
    optimizer = optim.Adam(agent.parameters(), lr=adapt_lr)
    
    env = make_nonstationary_cartpole(mode=target_mode)
    
    returns = []
    weight_histories = []
    
    for episode in range(num_adapt_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros((1, 2), device=device)
        prev_reward = torch.zeros((1, 1), device=device)
        
        episode_return = 0
        episode_log_probs = []
        episode_rewards = []
        episode_weights = []
        
        done = False
        
        while not done:
            action, log_prob, _, _, new_hidden, weights = agent.get_action_and_value(
                obs, prev_action, prev_reward, hidden
            )
            
            episode_log_probs.append(log_prob)
            if weights is not None:
                episode_weights.append(weights[0].detach().cpu().numpy())
            
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_return += reward
            
            obs = torch.tensor(next_obs, device=device, dtype=torch.float32).unsqueeze(0)
            prev_action = torch.zeros((1, 2), device=device)
            prev_action[0, action.item()] = 1.0
            prev_reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
            hidden = new_hidden.detach()
        
        returns.append(episode_return)
        if episode_weights:
            weight_histories.append(np.array(episode_weights))
        
        # REINFORCE update
        G = 0
        discounted_returns = []
        for r in reversed(episode_rewards):
            G = r + 0.99 * G
            discounted_returns.insert(0, G)
        
        discounted_returns = torch.tensor(discounted_returns, device=device, dtype=torch.float32)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, G in zip(episode_log_probs, discounted_returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        if verbose and (episode + 1) % 5 == 0:
            if episode_weights:
                mean_weight = np.mean([w.mean(axis=0) for w in weight_histories[-5:]], axis=0)
                print(f"  Episode {episode + 1}: Return = {episode_return:.0f}, Avg weights = {mean_weight}")
            else:
                print(f"  Episode {episode + 1}: Return = {episode_return:.0f}")
    
    env.close()
    
    return {
        'returns': returns,
        'weight_histories': weight_histories,
    }


def run_tta_experiment(
    checkpoint_path: str,
    modes: List[int] = [0, 1, 2],
    num_eval_episodes: int = 20,
    num_adapt_episodes: int = 30,
    adapt_lr: float = 1e-3,
    adapt_mode: str = "gating",
    agent_type: str = "auto",
    save_dir: str = "results/tta",
):
    """
    Run full TTA experiment: evaluate before/after adaptation on each mode.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        modes: Modes to test
        num_eval_episodes: Episodes for evaluation
        num_adapt_episodes: Episodes for adaptation
        adapt_lr: Learning rate for TTA
        adapt_mode: What to adapt ("gating", "policy_head", "all_params")
        agent_type: Agent type ("auto", "astral", "baseline")
        save_dir: Directory to save results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    agent, detected_type = load_agent(checkpoint_path, agent_type, device)
    print(f"Model loaded successfully (type: {detected_type})")
    
    # Validate adapt_mode
    if adapt_mode == "gating" and detected_type == "baseline":
        raise ValueError("Cannot use adapt_mode='gating' with baseline agent (no gating network)")
    
    # Choose adaptation function
    if adapt_mode == "gating":
        adapt_fn = test_time_adapt
        adapt_fn_name = "gating network"
    elif adapt_mode == "policy_head":
        adapt_fn = test_time_adapt_policy_head
        adapt_fn_name = "policy head"
    else:  # all_params
        adapt_fn = test_time_adapt_all_params
        adapt_fn_name = "all parameters"
        # Use lower learning rate for full fine-tuning
        if adapt_lr == 1e-3:
            adapt_lr = 1e-4
    
    print(f"Adaptation mode: {adapt_mode} ({adapt_fn_name})")
    
    results = {}
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Mode {mode}")
        print(f"{'='*60}")
        
        # Save original weights for reset
        original_state = {k: v.clone() for k, v in agent.state_dict().items()}
        
        # 1. Evaluate WITHOUT adaptation
        print(f"\n[1] Evaluating WITHOUT adaptation...")
        before_results = evaluate_agent(agent, mode, num_eval_episodes, device)
        before_mean = np.mean(before_results['returns'])
        before_std = np.std(before_results['returns'])
        print(f"    Mean return: {before_mean:.2f} ± {before_std:.2f}")
        
        # 2. Perform TTA
        print(f"\n[2] Performing test-time adaptation ({num_adapt_episodes} episodes, {adapt_mode})...")
        adapt_results = adapt_fn(
            agent, mode, num_adapt_episodes, adapt_lr, device, verbose=True
        )
        
        # 3. Evaluate AFTER adaptation
        print(f"\n[3] Evaluating AFTER adaptation...")
        after_results = evaluate_agent(agent, mode, num_eval_episodes, device)
        after_mean = np.mean(after_results['returns'])
        after_std = np.std(after_results['returns'])
        print(f"    Mean return: {after_mean:.2f} ± {after_std:.2f}")
        
        # Compute improvement
        improvement = after_mean - before_mean
        improvement_pct = 100 * improvement / (before_mean + 1e-8)
        print(f"\n    Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")
        
        results[mode] = {
            'before': before_results,
            'adapt': adapt_results,
            'after': after_results,
            'before_mean': before_mean,
            'after_mean': after_mean,
            'improvement': improvement,
        }
        
        # Reset agent to original weights for next mode
        agent.load_state_dict(original_state)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<10} {'Before':<15} {'After':<15} {'Improvement':<15}")
    print("-" * 55)
    for mode in modes:
        r = results[mode]
        print(f"{mode:<10} {r['before_mean']:<15.2f} {r['after_mean']:<15.2f} {r['improvement']:+.2f}")
    
    # Plot results
    plot_tta_results(results, save_dir, adapt_mode, detected_type)
    
    return results


def plot_tta_results(results: Dict, save_dir: str, adapt_mode: str = "gating", agent_type: str = "astral"):
    """Plot TTA results."""
    modes = sorted(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Title suffix
    title_suffix = f" ({agent_type}, {adapt_mode})"
    
    # Plot 1: Before vs After comparison
    ax1 = axes[0]
    x = np.arange(len(modes))
    width = 0.35
    
    before_means = [results[m]['before_mean'] for m in modes]
    after_means = [results[m]['after_mean'] for m in modes]
    
    ax1.bar(x - width/2, before_means, width, label='Before TTA', color='steelblue')
    ax1.bar(x + width/2, after_means, width, label='After TTA', color='coral')
    ax1.set_xlabel('Mode')
    ax1.set_ylabel('Mean Return')
    ax1.set_title('Before vs After TTA' + title_suffix)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Mode {m}' for m in modes])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Adaptation curves
    ax2 = axes[1]
    for mode in modes:
        returns = results[mode]['adapt']['returns']
        ax2.plot(returns, label=f'Mode {mode}', marker='o', markersize=3)
    ax2.set_xlabel('Adaptation Episode')
    ax2.set_ylabel('Return')
    ax2.set_title('Adaptation Progress')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Improvement
    ax3 = axes[2]
    improvements = [results[m]['improvement'] for m in modes]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax3.bar(x, improvements, color=colors)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Mode')
    ax3.set_ylabel('Improvement (After - Before)')
    ax3.set_title('TTA Improvement')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Mode {m}' for m in modes])
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tta_results.png'), dpi=150)
    plt.close()
    
    print(f"\nPlot saved to: {os.path.join(save_dir, 'tta_results.png')}")
    
    # Save raw data for regenerating plots
    data_to_save = {
        'metadata': {
            'adapt_mode': adapt_mode,
            'agent_type': agent_type,
        },
        'results': {}
    }
    for mode in modes:
        data_to_save['results'][str(mode)] = {
            'before_mean': results[mode]['before_mean'],
            'after_mean': results[mode]['after_mean'],
            'improvement': results[mode]['improvement'],
            'before_returns': results[mode]['before']['returns'],
            'after_returns': results[mode]['after']['returns'],
            'adapt_returns': results[mode]['adapt']['returns'],
        }
    
    data_path = os.path.join(save_dir, 'tta_results.json')
    with open(data_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Data saved to: {data_path}")


def main():
    parser = argparse.ArgumentParser(description="Test-Time Adaptation for ASTRAL")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--modes", type=int, nargs='+', default=[0, 1, 2],
                        help="Modes to test")
    parser.add_argument("--num_eval_episodes", type=int, default=20,
                        help="Episodes for evaluation")
    parser.add_argument("--num_adapt_episodes", type=int, default=30,
                        help="Episodes for adaptation")
    parser.add_argument("--adapt_lr", type=float, default=1e-3,
                        help="Learning rate for TTA")
    parser.add_argument("--adapt_mode", type=str, default="gating",
                        choices=["gating", "policy_head", "all_params"],
                        help="What to adapt: gating (ASTRAL only), policy_head, or all_params")
    parser.add_argument("--agent_type", type=str, default="auto",
                        choices=["auto", "astral", "baseline"],
                        help="Agent type (auto-detects from checkpoint path)")
    parser.add_argument("--save_dir", type=str, default="results/tta",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    run_tta_experiment(
        checkpoint_path=args.checkpoint,
        modes=args.modes,
        num_eval_episodes=args.num_eval_episodes,
        num_adapt_episodes=args.num_adapt_episodes,
        adapt_lr=args.adapt_lr,
        adapt_mode=args.adapt_mode,
        agent_type=args.agent_type,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()

