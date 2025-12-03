#!/usr/bin/env python3
"""
Fair Comparison Experiments: ASTRAL TTA vs Constrained Baseline

Experiments:
A. Parameter-matched: Same # params updated (4k)
B. Catastrophic forgetting: Test preservation of other modes
C. Few-shot: Limited adaptation budget (1, 3, 5, 10 episodes)
D. Extreme modes: Harder environment with more diverse physics

Usage:
    python scripts/run_fair_comparison.py --experiment all
    python scripts/run_fair_comparison.py --experiment A
    python scripts/run_fair_comparison.py --experiment B
"""

import argparse
import sys
import os
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.astral_agent import ASTRALAgent
from src.envs.nonstationary_cartpole import NonStationaryCartPole
import gymnasium as gym


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def evaluate_agent(agent, env, n_episodes=20, device='cuda'):
    """Evaluate agent on environment."""
    returns = []
    agent.eval()
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros(1, env.action_space.n, device=device)
        prev_reward = torch.zeros(1, 1, device=device)
        
        total_reward = 0
        for _ in range(500):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, _, _, _, hidden, _ = agent.get_action_and_value(
                    obs_t, prev_action, prev_reward, hidden
                )
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            
            prev_action = torch.zeros(1, env.action_space.n, device=device)
            prev_action[0, action.item()] = 1
            prev_reward = torch.FloatTensor([[reward]]).to(device)
            
            if terminated or truncated:
                break
        
        returns.append(total_reward)
    
    agent.train()
    return np.mean(returns), np.std(returns)


def adapt_gating_only(agent, env, n_episodes, lr=1e-3, device='cuda'):
    """Adapt only the gating network (ASTRAL TTA)."""
    # Freeze everything except gating
    for name, param in agent.named_parameters():
        param.requires_grad = 'abstraction_bank.gating' in name
    
    trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"  Gating-only: {trainable} trainable params")
    
    optimizer = torch.optim.Adam(
        [p for p in agent.parameters() if p.requires_grad], 
        lr=lr
    )
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros(1, env.action_space.n, device=device)
        prev_reward = torch.zeros(1, 1, device=device)
        
        log_probs = []
        rewards = []
        
        for _ in range(500):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            action, log_prob, _, _, hidden, _ = agent.get_action_and_value(
                obs_t, prev_action, prev_reward, hidden
            )
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            prev_action = torch.zeros(1, env.action_space.n, device=device)
            prev_action[0, action.item()] = 1
            prev_reward = torch.FloatTensor([[reward]]).to(device)
            
            if terminated or truncated:
                break
        
        # REINFORCE update
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Unfreeze all
    for param in agent.parameters():
        param.requires_grad = True
    
    return agent


def adapt_policy_head_only(agent, env, n_episodes, lr=1e-3, device='cuda'):
    """Adapt only the policy head (parameter-matched baseline ~4.3k params)."""
    # Freeze everything except policy_head (all layers)
    for name, param in agent.named_parameters():
        param.requires_grad = 'policy_head' in name  # All policy head layers
    
    trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"  Policy-head-only: {trainable} trainable params")
    
    optimizer = torch.optim.Adam(
        [p for p in agent.parameters() if p.requires_grad], 
        lr=lr
    )
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros(1, env.action_space.n, device=device)
        prev_reward = torch.zeros(1, 1, device=device)
        
        log_probs = []
        rewards = []
        
        for _ in range(500):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            action, log_prob, _, _, hidden, _ = agent.get_action_and_value(
                obs_t, prev_action, prev_reward, hidden
            )
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            prev_action = torch.zeros(1, env.action_space.n, device=device)
            prev_action[0, action.item()] = 1
            prev_reward = torch.FloatTensor([[reward]]).to(device)
            
            if terminated or truncated:
                break
        
        # REINFORCE update
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Unfreeze all
    for param in agent.parameters():
        param.requires_grad = True
    
    return agent


def adapt_full(agent, env, n_episodes, lr=1e-3, device='cuda'):
    """Adapt all parameters (full fine-tuning)."""
    trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"  Full fine-tuning: {trainable} trainable params")
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros(1, env.action_space.n, device=device)
        prev_reward = torch.zeros(1, 1, device=device)
        
        log_probs = []
        rewards = []
        
        for _ in range(500):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            action, log_prob, _, _, hidden, _ = agent.get_action_and_value(
                obs_t, prev_action, prev_reward, hidden
            )
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            prev_action = torch.zeros(1, env.action_space.n, device=device)
            prev_action[0, action.item()] = 1
            prev_reward = torch.FloatTensor([[reward]]).to(device)
            
            if terminated or truncated:
                break
        
        # REINFORCE update
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return agent


def load_astral_checkpoint(checkpoint_path, device='cuda'):
    """Load ASTRAL checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    agent = ASTRALAgent(
        obs_dim=4,
        action_dim=2,
        d_model=64,
        num_abstractions=3,
        slot_dropout=0.0,
    ).to(device)
    
    agent.load_state_dict(checkpoint)
    return agent


# =============================================================================
# EXPERIMENT A: Parameter-Matched Comparison
# =============================================================================

def experiment_A_parameter_matched(checkpoint_path, device='cuda'):
    """Compare gating adaptation vs last-layer adaptation (same param count)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Parameter-Matched Comparison")
    print("=" * 70)
    print("Question: Is gating a better thing to adapt than the last layer?")
    print()
    
    results = {"gating": {}, "policy_head": {}}
    n_adapt_episodes = 30
    
    for mode in [0, 1, 2]:
        print(f"\n--- Mode {mode} ---")
        
        # Create environment
        env = NonStationaryCartPole(gym.make("CartPole-v1"), mode=mode)
        
        # Test gating adaptation
        agent_gating = load_astral_checkpoint(checkpoint_path, device)
        before_gating, _ = evaluate_agent(agent_gating, env, device=device)
        print(f"Gating - Before: {before_gating:.1f}")
        
        agent_gating = adapt_gating_only(agent_gating, env, n_adapt_episodes, device=device)
        after_gating, _ = evaluate_agent(agent_gating, env, device=device)
        print(f"Gating - After: {after_gating:.1f}, Δ: {after_gating - before_gating:+.1f}")
        
        results["gating"][mode] = {
            "before": before_gating,
            "after": after_gating,
            "improvement": after_gating - before_gating
        }
        
        # Test policy-head adaptation
        agent_policy = load_astral_checkpoint(checkpoint_path, device)
        before_policy, _ = evaluate_agent(agent_policy, env, device=device)
        print(f"Policy-head - Before: {before_policy:.1f}")
        
        agent_policy = adapt_policy_head_only(agent_policy, env, n_adapt_episodes, device=device)
        after_policy, _ = evaluate_agent(agent_policy, env, device=device)
        print(f"Policy-head - After: {after_policy:.1f}, Δ: {after_policy - before_policy:+.1f}")
        
        results["policy_head"][mode] = {
            "before": before_policy,
            "after": after_policy,
            "improvement": after_policy - before_policy
        }
        
        env.close()
    
    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY - Experiment A")
    print("-" * 50)
    gating_avg = np.mean([results["gating"][m]["improvement"] for m in [0, 1, 2]])
    policy_avg = np.mean([results["policy_head"][m]["improvement"] for m in [0, 1, 2]])
    print(f"Gating adaptation avg improvement:      {gating_avg:+.1f}")
    print(f"Policy-head adaptation avg improvement: {policy_avg:+.1f}")
    print(f"Winner: {'GATING' if gating_avg > policy_avg else 'POLICY-HEAD'}")
    
    return results


# =============================================================================
# EXPERIMENT B: Catastrophic Forgetting
# =============================================================================

def experiment_B_forgetting(checkpoint_path, device='cuda'):
    """Test if ASTRAL preserves knowledge of other modes after adaptation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Catastrophic Forgetting Test")
    print("=" * 70)
    print("Question: Does ASTRAL preserve Mode 1 & 2 after adapting to Mode 0?")
    print()
    
    results = {"gating": {}, "full": {}}
    n_adapt_episodes = 30
    adapt_mode = 0
    
    # Create environments for all modes
    envs = {m: NonStationaryCartPole(gym.make("CartPole-v1"), mode=m) for m in [0, 1, 2]}
    
    # === GATING-ONLY ADAPTATION ===
    print("\n--- Gating-Only Adaptation ---")
    agent_gating = load_astral_checkpoint(checkpoint_path, device)
    
    # Evaluate on all modes BEFORE adaptation
    print("Before adaptation:")
    before = {}
    for m in [0, 1, 2]:
        score, _ = evaluate_agent(agent_gating, envs[m], device=device)
        before[m] = score
        print(f"  Mode {m}: {score:.1f}")
    
    # Adapt to Mode 0 only
    print(f"\nAdapting to Mode {adapt_mode} for {n_adapt_episodes} episodes...")
    agent_gating = adapt_gating_only(agent_gating, envs[adapt_mode], n_adapt_episodes, device=device)
    
    # Evaluate on all modes AFTER adaptation
    print("After adaptation:")
    after_gating = {}
    for m in [0, 1, 2]:
        score, _ = evaluate_agent(agent_gating, envs[m], device=device)
        after_gating[m] = score
        delta = score - before[m]
        print(f"  Mode {m}: {score:.1f} (Δ: {delta:+.1f})")
    
    results["gating"] = {
        "before": before.copy(),
        "after": after_gating,
        "adapted_mode": adapt_mode,
        "forgetting": {m: after_gating[m] - before[m] for m in [1, 2]}
    }
    
    # === FULL FINE-TUNING ===
    print("\n--- Full Fine-Tuning ---")
    agent_full = load_astral_checkpoint(checkpoint_path, device)
    
    # Before is same
    print("Before adaptation: (same as above)")
    
    # Adapt to Mode 0 with full fine-tuning
    print(f"\nAdapting to Mode {adapt_mode} for {n_adapt_episodes} episodes...")
    agent_full = adapt_full(agent_full, envs[adapt_mode], n_adapt_episodes, device=device)
    
    # Evaluate on all modes AFTER adaptation
    print("After adaptation:")
    after_full = {}
    for m in [0, 1, 2]:
        score, _ = evaluate_agent(agent_full, envs[m], device=device)
        after_full[m] = score
        delta = score - before[m]
        print(f"  Mode {m}: {score:.1f} (Δ: {delta:+.1f})")
    
    results["full"] = {
        "before": before.copy(),
        "after": after_full,
        "adapted_mode": adapt_mode,
        "forgetting": {m: after_full[m] - before[m] for m in [1, 2]}
    }
    
    for env in envs.values():
        env.close()
    
    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY - Experiment B")
    print("-" * 50)
    print(f"Adapted to Mode {adapt_mode}")
    print(f"\nMode {adapt_mode} improvement:")
    print(f"  Gating: {after_gating[0] - before[0]:+.1f}")
    print(f"  Full:   {after_full[0] - before[0]:+.1f}")
    print(f"\nForgetting on other modes (Mode 1 + Mode 2):")
    gating_forget = sum(results["gating"]["forgetting"].values())
    full_forget = sum(results["full"]["forgetting"].values())
    print(f"  Gating: {gating_forget:+.1f}")
    print(f"  Full:   {full_forget:+.1f}")
    print(f"\nLess forgetting: {'GATING' if gating_forget > full_forget else 'FULL'}")
    
    return results


# =============================================================================
# EXPERIMENT C: Few-Shot Adaptation
# =============================================================================

def experiment_C_fewshot(checkpoint_path, device='cuda'):
    """Compare adaptation speed with limited episodes."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Few-Shot Adaptation Speed")
    print("=" * 70)
    print("Question: Who learns faster with limited data?")
    print()
    
    episode_budgets = [1, 3, 5, 10, 20, 30]
    results = {"gating": {}, "policy_head": {}, "full": {}}
    
    test_mode = 0  # Test on one mode for simplicity
    env = NonStationaryCartPole(gym.make("CartPole-v1"), mode=test_mode)
    
    # Get baseline performance
    agent_base = load_astral_checkpoint(checkpoint_path, device)
    baseline_score, _ = evaluate_agent(agent_base, env, device=device)
    print(f"Baseline (no adaptation): {baseline_score:.1f}")
    
    for method_name, adapt_fn in [
        ("gating", adapt_gating_only),
        ("policy_head", adapt_policy_head_only),
        ("full", adapt_full)
    ]:
        print(f"\n--- {method_name.upper()} ---")
        results[method_name] = {"baseline": baseline_score, "by_budget": {}}
        
        for n_eps in episode_budgets:
            agent = load_astral_checkpoint(checkpoint_path, device)
            agent = adapt_fn(agent, env, n_eps, device=device)
            score, _ = evaluate_agent(agent, env, device=device)
            improvement = score - baseline_score
            results[method_name]["by_budget"][n_eps] = {
                "score": score,
                "improvement": improvement
            }
            print(f"  {n_eps:2d} episodes: {score:.1f} (Δ: {improvement:+.1f})")
    
    env.close()
    
    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY - Experiment C")
    print("-" * 50)
    print(f"{'Episodes':<10} {'Gating':>10} {'Policy-Head':>12} {'Full':>10}")
    print("-" * 50)
    for n_eps in episode_budgets:
        g = results["gating"]["by_budget"][n_eps]["improvement"]
        p = results["policy_head"]["by_budget"][n_eps]["improvement"]
        f = results["full"]["by_budget"][n_eps]["improvement"]
        print(f"{n_eps:<10} {g:>+10.1f} {p:>+12.1f} {f:>+10.1f}")
    
    return results


# =============================================================================
# EXPERIMENT D: Extreme Modes
# =============================================================================

class ExtremeNonStationaryCartPole(gym.Wrapper):
    """CartPole with EXTREME mode differences."""
    
    EXTREME_MODES = {
        0: {"gravity": 5.0, "masscart": 0.5, "masspole": 0.05, "length": 0.3},   # Very easy
        1: {"gravity": 15.0, "masscart": 1.0, "masspole": 0.1, "length": 0.5},   # Normal
        2: {"gravity": 25.0, "masscart": 2.0, "masspole": 0.2, "length": 0.8},   # Very hard
    }
    
    def __init__(self, env, mode=None):
        super().__init__(env)
        self._mode = mode
        self._set_mode(mode if mode is not None else np.random.randint(3))
    
    def _set_mode(self, mode):
        self._current_mode = mode
        params = self.EXTREME_MODES[mode]
        self.unwrapped.gravity = params["gravity"]
        self.unwrapped.masscart = params["masscart"]
        self.unwrapped.masspole = params["masspole"]
        self.unwrapped.length = params["length"]
        self.unwrapped.total_mass = params["masscart"] + params["masspole"]
        self.unwrapped.polemass_length = params["masspole"] * params["length"]
    
    def reset(self, **kwargs):
        if self._mode is None:
            self._set_mode(np.random.randint(3))
        obs, info = self.env.reset(**kwargs)
        info["mode"] = self._current_mode
        return obs, info


def experiment_D_extreme_modes(checkpoint_path, device='cuda'):
    """Test adaptation on extreme mode differences."""
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Extreme Mode Differences")
    print("=" * 70)
    print("Question: Do extreme modes force slot specialization?")
    print()
    print("Extreme mode settings:")
    for mode, params in ExtremeNonStationaryCartPole.EXTREME_MODES.items():
        print(f"  Mode {mode}: gravity={params['gravity']}, length={params['length']}")
    print()
    
    results = {"gating": {}, "full": {}}
    n_adapt_episodes = 30
    
    for mode in [0, 1, 2]:
        print(f"\n--- Extreme Mode {mode} ---")
        
        env = ExtremeNonStationaryCartPole(gym.make("CartPole-v1"), mode=mode)
        
        # Gating adaptation
        agent_gating = load_astral_checkpoint(checkpoint_path, device)
        before, _ = evaluate_agent(agent_gating, env, device=device)
        print(f"Gating - Before: {before:.1f}")
        
        agent_gating = adapt_gating_only(agent_gating, env, n_adapt_episodes, device=device)
        after_gating, _ = evaluate_agent(agent_gating, env, device=device)
        print(f"Gating - After: {after_gating:.1f}, Δ: {after_gating - before:+.1f}")
        
        results["gating"][mode] = {
            "before": before,
            "after": after_gating,
            "improvement": after_gating - before
        }
        
        # Full adaptation
        agent_full = load_astral_checkpoint(checkpoint_path, device)
        before_full, _ = evaluate_agent(agent_full, env, device=device)
        
        agent_full = adapt_full(agent_full, env, n_adapt_episodes, device=device)
        after_full, _ = evaluate_agent(agent_full, env, device=device)
        print(f"Full - After: {after_full:.1f}, Δ: {after_full - before_full:+.1f}")
        
        results["full"][mode] = {
            "before": before_full,
            "after": after_full,
            "improvement": after_full - before_full
        }
        
        env.close()
    
    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY - Experiment D")
    print("-" * 50)
    gating_avg = np.mean([results["gating"][m]["improvement"] for m in [0, 1, 2]])
    full_avg = np.mean([results["full"][m]["improvement"] for m in [0, 1, 2]])
    print(f"Gating avg improvement: {gating_avg:+.1f}")
    print(f"Full avg improvement:   {full_avg:+.1f}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "A", "B", "C", "D"],
                        help="Which experiment to run")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to ASTRAL checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Try to find slot_dropout model first (best for TTA)
        import glob
        candidates = glob.glob("results/runs/slot_dropout_0.3_astral_*/final_model.pt")
        if not candidates:
            candidates = glob.glob("results/runs/best_config_strong_astral_*/final_model.pt")
        if not candidates:
            candidates = glob.glob("results/runs/astral_astral_*/final_model.pt")
        
        if not candidates:
            print("ERROR: No ASTRAL checkpoint found. Train one first or specify --checkpoint")
            return
        
        checkpoint_path = candidates[0]
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    all_results = {}
    
    if args.experiment in ["all", "A"]:
        all_results["A"] = experiment_A_parameter_matched(checkpoint_path, device)
    
    if args.experiment in ["all", "B"]:
        all_results["B"] = experiment_B_forgetting(checkpoint_path, device)
    
    if args.experiment in ["all", "C"]:
        all_results["C"] = experiment_C_fewshot(checkpoint_path, device)
    
    if args.experiment in ["all", "D"]:
        all_results["D"] = experiment_D_extreme_modes(checkpoint_path, device)
    
    # Save results
    os.makedirs("results/fair_comparison", exist_ok=True)
    with open("results/fair_comparison/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print("Results saved to results/fair_comparison/results.json")


if __name__ == "__main__":
    main()

