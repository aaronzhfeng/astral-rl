#!/usr/bin/env python3
"""
Experiment D+C: Extreme Modes with Few-Shot Adaptation

Test how gating vs full fine-tuning perform on extreme mode differences
with varying adaptation budgets.
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.astral_agent import ASTRALAgent
import gymnasium as gym


class ExtremeNonStationaryCartPole(gym.Wrapper):
    """CartPole with EXTREME mode differences."""
    
    EXTREME_MODES = {
        0: {"gravity": 5.0, "masscart": 0.5, "masspole": 0.05, "length": 0.3},
        1: {"gravity": 15.0, "masscart": 1.0, "masspole": 0.1, "length": 0.5},
        2: {"gravity": 25.0, "masscart": 2.0, "masspole": 0.2, "length": 0.8},
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


def evaluate_agent(agent, env, n_episodes=20, device='cuda'):
    returns = []
    agent.eval()
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros(1, 2, device=device)
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
            prev_action = torch.zeros(1, 2, device=device)
            prev_action[0, action.item()] = 1
            prev_reward = torch.FloatTensor([[reward]]).to(device)
            if terminated or truncated:
                break
        returns.append(total_reward)
    
    agent.train()
    return np.mean(returns), np.std(returns)


def adapt_agent(agent, env, n_episodes, mode='gating', lr=1e-3, device='cuda'):
    """Adapt agent with specified mode."""
    # Set requires_grad based on mode
    for name, param in agent.named_parameters():
        if mode == 'gating':
            param.requires_grad = 'abstraction_bank.gating' in name
        elif mode == 'policy_head':
            param.requires_grad = 'policy_head' in name
        else:  # full
            param.requires_grad = True
    
    optimizer = torch.optim.Adam(
        [p for p in agent.parameters() if p.requires_grad], lr=lr
    )
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros(1, 2, device=device)
        prev_reward = torch.zeros(1, 1, device=device)
        
        log_probs, rewards = [], []
        
        for _ in range(500):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, log_prob, _, _, hidden, _ = agent.get_action_and_value(
                obs_t, prev_action, prev_reward, hidden
            )
            obs, reward, terminated, truncated, _ = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            prev_action = torch.zeros(1, 2, device=device)
            prev_action[0, action.item()] = 1
            prev_reward = torch.FloatTensor([[reward]]).to(device)
            if terminated or truncated:
                break
        
        # REINFORCE
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = -(torch.stack(log_probs) * returns).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Unfreeze all
    for param in agent.parameters():
        param.requires_grad = True
    
    return agent


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    agent = ASTRALAgent(obs_dim=4, action_dim=2, d_model=64, num_abstractions=3).to(device)
    agent.load_state_dict(checkpoint)
    return agent


def main():
    import glob
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Find checkpoint
    candidates = glob.glob("results/runs/slot_dropout_0.3_astral_*/final_model.pt")
    if not candidates:
        candidates = glob.glob("results/runs/astral_astral_*/final_model.pt")
    checkpoint_path = candidates[0]
    print(f"Checkpoint: {checkpoint_path}")
    
    episode_budgets = [1, 3, 5, 10, 20, 30, 50]
    modes_to_test = [0, 1, 2]
    methods = ['gating', 'policy_head', 'full']
    
    results = {m: {method: {} for method in methods} for m in modes_to_test}
    
    print("\n" + "=" * 80)
    print("EXTREME MODES + FEW-SHOT ADAPTATION")
    print("=" * 80)
    print("Extreme mode settings:")
    for mode, params in ExtremeNonStationaryCartPole.EXTREME_MODES.items():
        print(f"  Mode {mode}: gravity={params['gravity']}, length={params['length']}")
    
    for mode in modes_to_test:
        print(f"\n{'='*80}")
        print(f"EXTREME MODE {mode}")
        print("=" * 80)
        
        env = ExtremeNonStationaryCartPole(gym.make("CartPole-v1"), mode=mode)
        
        # Get baseline
        agent_base = load_checkpoint(checkpoint_path, device)
        baseline, _ = evaluate_agent(agent_base, env, device=device)
        print(f"Baseline (no adapt): {baseline:.1f}")
        
        for method in methods:
            print(f"\n--- {method.upper()} ---")
            results[mode][method]['baseline'] = baseline
            results[mode][method]['by_budget'] = {}
            
            for n_eps in episode_budgets:
                agent = load_checkpoint(checkpoint_path, device)
                agent = adapt_agent(agent, env, n_eps, mode=method, device=device)
                score, _ = evaluate_agent(agent, env, device=device)
                improvement = score - baseline
                results[mode][method]['by_budget'][n_eps] = {
                    'score': score, 'improvement': improvement
                }
                print(f"  {n_eps:2d} eps: {score:.1f} (Δ: {improvement:+.1f})")
        
        env.close()
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Average Improvement Across All Extreme Modes")
    print("=" * 80)
    print(f"{'Episodes':<10} {'Gating':>12} {'Policy-Head':>14} {'Full':>12}")
    print("-" * 50)
    
    for n_eps in episode_budgets:
        g_avg = np.mean([results[m]['gating']['by_budget'][n_eps]['improvement'] for m in modes_to_test])
        p_avg = np.mean([results[m]['policy_head']['by_budget'][n_eps]['improvement'] for m in modes_to_test])
        f_avg = np.mean([results[m]['full']['by_budget'][n_eps]['improvement'] for m in modes_to_test])
        print(f"{n_eps:<10} {g_avg:>+12.1f} {p_avg:>+14.1f} {f_avg:>+12.1f}")
    
    # Save
    os.makedirs("results/extreme_fewshot", exist_ok=True)
    with open("results/extreme_fewshot/results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to results/extreme_fewshot/results.json")


if __name__ == "__main__":
    main()

