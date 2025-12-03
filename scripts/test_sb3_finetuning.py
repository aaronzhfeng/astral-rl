#!/usr/bin/env python3
"""
Test if fine-tuning SB3 PPO helps on specific modes.
This is the baseline equivalent of ASTRAL's TTA.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from src.envs.nonstationary_cartpole import NonStationaryCartPole
import json

def evaluate_model(model, env, n_episodes=20):
    """Evaluate model on environment."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    return np.mean(returns), np.std(returns)


def main():
    print("=" * 60)
    print("SB3 PPO Fine-tuning Test (Baseline TTA Equivalent)")
    print("=" * 60)
    
    # Load pre-trained model
    model_path = "results/sb3_baseline/ppo_model.zip"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Run: python scripts/train_sb3_baseline.py first")
        return
    
    results = {"modes": {}}
    
    for mode in [0, 1, 2]:
        print(f"\n{'='*60}")
        print(f"Mode {mode}")
        print("=" * 60)
        
        # Create mode-specific environment
        def make_env():
            base_env = gym.make("CartPole-v1")
            return NonStationaryCartPole(base_env, mode=mode)
        
        env = make_env()  # Single env for evaluation
        
        # Load fresh model with new env (handles n_envs mismatch)
        model = PPO.load(model_path, env=env, device="cuda")
        
        # Evaluate BEFORE fine-tuning
        before_mean, before_std = evaluate_model(model, env, n_episodes=20)
        print(f"Before fine-tuning: {before_mean:.1f} ± {before_std:.1f}")
        
        # Fine-tune on this specific mode (30 episodes worth of steps)
        print("Fine-tuning for 15,000 steps...")
        model.learn(total_timesteps=15000, reset_num_timesteps=False)
        
        # Evaluate AFTER fine-tuning
        after_mean, after_std = evaluate_model(model, env, n_episodes=20)
        print(f"After fine-tuning:  {after_mean:.1f} ± {after_std:.1f}")
        
        improvement = after_mean - before_mean
        print(f"Improvement: {improvement:+.1f}")
        
        results["modes"][str(mode)] = {
            "before": {"mean": before_mean, "std": before_std},
            "after": {"mean": after_mean, "std": after_std},
            "improvement": improvement
        }
        
        env.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<10} {'Before':>12} {'After':>12} {'Δ':>10}")
    print("-" * 50)
    
    improvements = []
    for mode in ["0", "1", "2"]:
        r = results["modes"][mode]
        print(f"{mode:<10} {r['before']['mean']:>12.1f} {r['after']['mean']:>12.1f} {r['improvement']:>+10.1f}")
        improvements.append(r["improvement"])
    
    avg_improvement = np.mean(improvements)
    print("-" * 50)
    print(f"{'Average':<10} {'':<12} {'':<12} {avg_improvement:>+10.1f}")
    
    results["avg_improvement"] = avg_improvement
    
    # Save results
    os.makedirs("results/sb3_finetuning", exist_ok=True)
    with open("results/sb3_finetuning/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/sb3_finetuning/results.json")
    
    # Comparison with ASTRAL TTA
    print("\n" + "=" * 60)
    print("COMPARISON WITH ASTRAL TTA")
    print("=" * 60)
    print(f"SB3 PPO fine-tuning avg improvement: {avg_improvement:+.1f}")
    print(f"ASTRAL slot_dropout TTA improvement: +11.4")
    print(f"ASTRAL collapsed TTA improvement:    -4.8")


if __name__ == "__main__":
    main()

