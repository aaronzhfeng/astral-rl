#!/usr/bin/env python3
"""
Train SB3 PPO baseline on NonStationary CartPole.

GPU Utilization Notes:
- PPO is CPU-bound because env.step() runs on CPU
- To improve GPU utilization:
  1. Use more parallel environments (n_envs)
  2. Use larger batch sizes
  3. Use more epochs per update (n_epochs)
  
For MLP policies, CPU is often faster than GPU for small networks.
GPU benefits more for CNN policies or very large networks.

Usage:
    python scripts/train_sb3_baseline.py --n_envs 8 --total_timesteps 100000
    python scripts/train_sb3_baseline.py --device cpu  # Often faster for MLP!
"""

import argparse
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def make_env(mode=None):
    """Create a NonStationary CartPole environment."""
    import gymnasium as gym
    from src.envs.nonstationary_cartpole import NonStationaryCartPole
    
    def _init():
        base_env = gym.make("CartPole-v1")
        env = NonStationaryCartPole(base_env, mode=mode)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train SB3 PPO on NonStationary CartPole")
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--n_envs", type=int, default=8, 
                        help="Number of parallel environments (higher = better GPU util)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Minibatch size (higher = better GPU util)")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Steps per rollout per env")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Epochs per update")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device (cpu often faster for MLP!)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=60)
    parser.add_argument("--save_dir", type=str, default="results/sb3_baseline")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no_progress_bar", action="store_true",
                        help="Disable progress bar (avoids tqdm/rich dependency)")
    args = parser.parse_args()
    
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    
    # Set seed
    set_random_seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("SB3 PPO Baseline Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"N_envs: {args.n_envs}")
    print(f"Batch size: {args.batch_size}")
    print(f"N_steps: {args.n_steps}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Effective batch: {args.n_envs * args.n_steps}")
    print("=" * 60)
    
    # Create vectorized environment
    # Use SubprocVecEnv for parallel CPU execution (better for many envs)
    # Use DummyVecEnv for single-threaded (simpler, good for debugging)
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env()])
    
    # Callback to track episode returns
    class TrackReturnsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_returns = []
            self.episode_lengths = []
            
        def _on_step(self):
            # Check for completed episodes
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self.episode_returns.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
            return True
    
    callback = TrackReturnsCallback()
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=args.verbose,
        device=device,
        seed=args.seed,
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=not args.no_progress_bar,
    )
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"SPS: {args.total_timesteps / train_time:.0f}")
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model.save(os.path.join(args.save_dir, "ppo_model"))
    
    # Evaluate per mode
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    
    # Create single env for evaluation
    import gymnasium as gym
    from src.envs.nonstationary_cartpole import NonStationaryCartPole
    
    eval_env = NonStationaryCartPole(gym.make("CartPole-v1"))
    returns_by_mode = {0: [], 1: [], 2: []}
    
    for ep in range(args.eval_episodes):
        obs, info = eval_env.reset()
        mode = info.get('mode', 0)
        total = 0
        
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            if terminated or truncated:
                break
        
        returns_by_mode[mode].append(total)
        
        if (ep + 1) % 20 == 0:
            print(f"  Evaluated {ep + 1}/{args.eval_episodes} episodes")
    
    # Results
    results = {
        "config": vars(args),
        "training_time": train_time,
        "training_returns": callback.episode_returns,
        "eval_by_mode": {str(k): v for k, v in returns_by_mode.items()},
        "eval_summary": {}
    }
    
    print("\nResults by mode:")
    for mode in [0, 1, 2]:
        if returns_by_mode[mode]:
            mean = np.mean(returns_by_mode[mode])
            std = np.std(returns_by_mode[mode])
            results["eval_summary"][str(mode)] = {
                "mean": float(mean), 
                "std": float(std), 
                "n": len(returns_by_mode[mode])
            }
            print(f"  Mode {mode}: {mean:.1f} ± {std:.1f} ({len(returns_by_mode[mode])} episodes)")
    
    all_returns = [r for returns in returns_by_mode.values() for r in returns]
    overall_mean = np.mean(all_returns)
    overall_std = np.std(all_returns)
    results["eval_summary"]["overall"] = {
        "mean": float(overall_mean), 
        "std": float(overall_std)
    }
    print(f"\n  Overall: {overall_mean:.1f} ± {overall_std:.1f}")
    
    # Save results
    results_path = os.path.join(args.save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {args.save_dir}/")
    print("=" * 60)
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

