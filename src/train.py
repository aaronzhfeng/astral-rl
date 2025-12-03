#!/usr/bin/env python3
# src/train.py
"""
ASTRAL Training Script.

Trains ASTRAL agent (or baseline) on NonStationaryCartPole using PPO.

Key features:
- PPO algorithm with GAE
- Abstraction bank regularization losses
- Per-mode performance tracking
- Tensorboard logging
- Checkpoint saving

Interpretability improvements (all optional, enabled via flags):
- --use_gumbel: Gumbel-Softmax for exploration
- --hard_routing: Discrete slot selection
- --orthogonal_init: Diverse slot initialization
- --temp_anneal: Temperature annealing from high to low
- --lambda_contrast: Contrastive loss for mode-slot correspondence
- --slot_prediction: Auxiliary slot prediction task

Usage:
    # Basic training
    python src/train.py --use_abstractions True --total_timesteps 500000
    
    # With interpretability improvements
    python src/train.py --use_gumbel True --hard_routing True --orthogonal_init True
    python src/train.py --temp_anneal True --tau_start 5.0 --tau_end 0.5
    python src/train.py --lambda_contrast 0.01 --slot_prediction True
    
    # Baseline (no abstractions)
    python src/train.py --use_abstractions False
"""

import os
import sys
import time
import random
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.nonstationary_cartpole import make_vectorized_nonstationary_cartpole
from src.models.astral_agent import ASTRALAgent, BaselineAgent, count_parameters
from src.models.abstraction_bank import TemperatureScheduler
from src.losses import (
    compute_abstraction_losses,
    compute_contrastive_slot_loss,
    SlotPredictor,
    compute_slot_prediction_loss,
    compute_slot_prediction_accuracy,
)


@dataclass
class Config:
    """Training configuration."""
    # Experiment
    exp_name: str = "astral"
    seed: int = 42
    use_abstractions: bool = True
    
    # Environment
    num_envs: int = 8
    
    # Architecture
    d_model: int = 64
    num_abstractions: int = 3
    tau: float = 1.0
    
    # Interpretability improvements (all disabled by default)
    use_gumbel: bool = False          # Gumbel-Softmax for exploration
    hard_routing: bool = False        # Hard one-hot routing
    orthogonal_init: bool = False     # Orthogonal abstraction initialization
    slot_dropout: float = 0.0         # Slot dropout probability (0 = disabled)
    temp_anneal: bool = False         # Temperature annealing
    tau_start: float = 5.0            # Starting temperature (if temp_anneal)
    tau_end: float = 0.5              # Ending temperature (if temp_anneal)
    temp_anneal_steps: int = 200000   # Steps to anneal temperature
    lambda_contrast: float = 0.0      # Contrastive loss weight (0 = disabled)
    slot_prediction: bool = False     # Auxiliary slot prediction task
    lambda_slot_pred: float = 0.01    # Slot prediction loss weight
    
    # PPO Hyperparameters
    total_timesteps: int = 500_000
    num_steps: int = 128  # Steps per rollout per env
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    
    # Abstraction regularization
    lambda_w_ent: float = 0.001
    lambda_lb: float = 0.001
    lambda_orth: float = 0.0001
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    log_dir: str = "results/runs"


def parse_args() -> Config:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ASTRAL agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interpretability Improvements (all optional):
  --use_gumbel       Use Gumbel-Softmax for differentiable discrete selection
  --hard_routing     Use hard (one-hot) routing with straight-through gradient
  --orthogonal_init  Initialize abstractions to be orthogonal
  --temp_anneal      Anneal temperature from tau_start to tau_end
  --lambda_contrast  Add contrastive loss (encourage different modes to use different slots)
  --slot_prediction  Add auxiliary task to predict slot from observation

Example:
  python src/train.py --use_gumbel True --temp_anneal True --lambda_contrast 0.01
        """
    )
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="astral")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_abstractions", type=lambda x: x.lower() == 'true', default=True)
    
    # Architecture
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_abstractions", type=int, default=3)
    parser.add_argument("--tau", type=float, default=1.0)
    
    # Interpretability improvements
    parser.add_argument("--use_gumbel", type=lambda x: x.lower() == 'true', default=False,
                        help="Use Gumbel-Softmax for exploration")
    parser.add_argument("--hard_routing", type=lambda x: x.lower() == 'true', default=False,
                        help="Use hard one-hot routing")
    parser.add_argument("--orthogonal_init", type=lambda x: x.lower() == 'true', default=False,
                        help="Initialize abstractions orthogonally")
    parser.add_argument("--slot_dropout", type=float, default=0.0,
                        help="Slot dropout probability during training (0 = disabled)")
    parser.add_argument("--temp_anneal", type=lambda x: x.lower() == 'true', default=False,
                        help="Enable temperature annealing")
    parser.add_argument("--tau_start", type=float, default=5.0,
                        help="Starting temperature for annealing")
    parser.add_argument("--tau_end", type=float, default=0.5,
                        help="Ending temperature for annealing")
    parser.add_argument("--temp_anneal_steps", type=int, default=200000,
                        help="Steps to anneal temperature")
    parser.add_argument("--lambda_contrast", type=float, default=0.0,
                        help="Contrastive loss weight (0 = disabled)")
    parser.add_argument("--slot_prediction", type=lambda x: x.lower() == 'true', default=False,
                        help="Enable auxiliary slot prediction task")
    parser.add_argument("--lambda_slot_pred", type=float, default=0.01,
                        help="Slot prediction loss weight")
    
    # PPO
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    
    # Regularization
    parser.add_argument("--lambda_w_ent", type=float, default=0.001)
    parser.add_argument("--lambda_lb", type=float, default=0.001)
    parser.add_argument("--lambda_orth", type=float, default=0.0001)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default="results/runs")
    
    args = parser.parse_args()
    
    config = Config()
    for key, value in vars(args).items():
        setattr(config, key, value)
    
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(config: Config):
    """
    Main training loop.
    
    Args:
        config: Training configuration
    """
    # Setup
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create run directory
    run_name = f"{config.exp_name}_{'astral' if config.use_abstractions else 'baseline'}_{config.seed}_{int(time.time())}"
    run_dir = os.path.join(config.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Tensorboard
    writer = SummaryWriter(run_dir)
    
    # Create vectorized environments
    envs = make_vectorized_nonstationary_cartpole(num_envs=config.num_envs)
    
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create agent
    if config.use_abstractions:
        # Print enabled improvements
        improvements = []
        if config.use_gumbel:
            improvements.append("Gumbel-Softmax")
        if config.hard_routing:
            improvements.append("Hard-Routing")
        if config.orthogonal_init:
            improvements.append("Orthogonal-Init")
        if config.slot_dropout > 0:
            improvements.append(f"Slot-Dropout({config.slot_dropout})")
        if config.temp_anneal:
            improvements.append(f"Temp-Anneal({config.tau_start}→{config.tau_end})")
        if config.lambda_contrast > 0:
            improvements.append(f"Contrastive(λ={config.lambda_contrast})")
        if config.slot_prediction:
            improvements.append(f"SlotPred(λ={config.lambda_slot_pred})")
        
        agent = ASTRALAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=config.d_model,
            num_abstractions=config.num_abstractions,
            tau=config.tau_start if config.temp_anneal else config.tau,
            use_gumbel=config.use_gumbel,
            hard_routing=config.hard_routing,
            orthogonal_init=config.orthogonal_init,
            slot_dropout=config.slot_dropout,
        ).to(device)
        
        print(f"Training ASTRAL agent ({count_parameters(agent):,} parameters)")
        if improvements:
            print(f"  Improvements: {', '.join(improvements)}")
        else:
            print(f"  No interpretability improvements enabled (use --help to see options)")
    else:
        agent = BaselineAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=config.d_model,
        ).to(device)
        print(f"Training Baseline agent ({count_parameters(agent):,} parameters)")
    
    # Temperature scheduler (if enabled)
    temp_scheduler = None
    if config.use_abstractions and config.temp_anneal:
        temp_scheduler = TemperatureScheduler(
            tau_start=config.tau_start,
            tau_end=config.tau_end,
            warmup_steps=config.temp_anneal_steps,
        )
    
    # Slot predictor (if enabled)
    slot_predictor = None
    if config.use_abstractions and config.slot_prediction:
        slot_predictor = SlotPredictor(
            obs_dim=obs_dim,
            num_slots=config.num_abstractions,
        ).to(device)
        print(f"  Slot predictor: {sum(p.numel() for p in slot_predictor.parameters()):,} parameters")
    
    # Optimizer (include slot predictor if present)
    params_to_optimize = list(agent.parameters())
    if slot_predictor is not None:
        params_to_optimize += list(slot_predictor.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=config.learning_rate)
    
    # Storage for rollout
    obs_buffer = torch.zeros((config.num_steps, config.num_envs, obs_dim), device=device)
    actions_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    values_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    hiddens_buffer = torch.zeros((config.num_steps, config.num_envs, config.d_model), device=device)
    prev_actions_buffer = torch.zeros((config.num_steps, config.num_envs, action_dim), device=device)
    prev_rewards_buffer = torch.zeros((config.num_steps, config.num_envs, 1), device=device)
    
    if config.use_abstractions:
        weights_buffer = torch.zeros(
            (config.num_steps, config.num_envs, config.num_abstractions), 
            device=device
        )
    
    # Mode buffer for contrastive loss
    modes_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=device)
    current_modes = torch.zeros(config.num_envs, dtype=torch.long, device=device)
    
    # Initialize
    obs, info = envs.reset(seed=config.seed)
    obs = torch.tensor(obs, device=device, dtype=torch.float32)
    hidden = agent.get_initial_hidden(config.num_envs, device)
    
    # Get initial modes from info
    if "mode" in info:
        current_modes = torch.tensor(info["mode"], dtype=torch.long, device=device)
    
    # For tracking previous action/reward
    prev_action = torch.zeros((config.num_envs, action_dim), device=device)
    prev_reward = torch.zeros((config.num_envs, 1), device=device)
    
    # Training metrics
    global_step = 0
    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    
    episode_returns = []
    episode_lengths = []
    mode_returns = defaultdict(list)
    mode_weights = defaultdict(list)  # Track weights per mode
    
    start_time = time.time()
    
    print(f"\nStarting training for {config.total_timesteps:,} timesteps ({num_updates} updates)")
    print("=" * 60)
    
    for update in range(num_updates):
        # Update temperature if annealing enabled
        if temp_scheduler is not None:
            current_tau = temp_scheduler.get_tau(global_step)
            agent.abstraction_bank.set_temperature(current_tau)
        
        # Rollout
        for step in range(config.num_steps):
            global_step += config.num_envs
            
            # Store current state
            obs_buffer[step] = obs
            hiddens_buffer[step] = hidden
            prev_actions_buffer[step] = prev_action
            prev_rewards_buffer[step] = prev_reward
            
            with torch.no_grad():
                action, logprob, entropy, value, new_hidden, weights = agent.get_action_and_value(
                    obs, prev_action, prev_reward, hidden
                )
            
            # Store action info
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob
            values_buffer[step] = value
            modes_buffer[step] = current_modes  # Store current modes for contrastive loss
            if config.use_abstractions and weights is not None:
                weights_buffer[step] = weights
            
            # Step environment
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards_buffer[step] = torch.tensor(reward, device=device, dtype=torch.float32)
            dones_buffer[step] = torch.tensor(done, device=device, dtype=torch.float32)
            
            # Update current modes from info
            if "mode" in infos:
                current_modes = torch.tensor(infos["mode"], dtype=torch.long, device=device)
            
            # Track episode stats (gymnasium vectorized env format)
            if "episode" in infos:
                # _episode is a boolean mask indicating which envs finished
                finished_mask = infos.get("_episode", np.zeros(config.num_envs, dtype=bool))
                
                for i in range(config.num_envs):
                    if finished_mask[i]:
                        ep_return = infos["episode"]["r"][i]
                        ep_length = infos["episode"]["l"][i]
                        mode = infos.get("mode", np.full(config.num_envs, -1))[i]
                        
                        episode_returns.append(float(ep_return))
                        episode_lengths.append(int(ep_length))
                        mode_returns[int(mode)].append(float(ep_return))
                        
                        # Track weights for this mode
                        if config.use_abstractions and weights is not None:
                            mode_weights[int(mode)].append(weights[i].cpu().numpy())
            
            # Update for next step
            obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
            
            # Update prev_action (one-hot)
            prev_action = torch.zeros((config.num_envs, action_dim), device=device)
            prev_action.scatter_(1, action.unsqueeze(1), 1.0)
            prev_reward = torch.tensor(reward, device=device, dtype=torch.float32).unsqueeze(1)
            
            # Update hidden, reset for done episodes
            hidden = new_hidden.clone()
            for i, d in enumerate(done):
                if d:
                    hidden[i] = torch.zeros(config.d_model, device=device)
                    prev_action[i] = torch.zeros(action_dim, device=device)
                    prev_reward[i] = torch.zeros(1, device=device)
        
        # Compute advantages (GAE)
        with torch.no_grad():
            _, _, _, next_value, _, _ = agent.get_action_and_value(
                obs, prev_action, prev_reward, hidden
            )
            advantages = torch.zeros_like(rewards_buffer, device=device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                delta = rewards_buffer[t] + config.gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buffer
        
        # Flatten for PPO update
        b_obs = obs_buffer.reshape(-1, obs_dim)
        b_actions = actions_buffer.reshape(-1)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)
        b_hiddens = hiddens_buffer.reshape(-1, config.d_model)
        b_prev_actions = prev_actions_buffer.reshape(-1, action_dim)
        b_prev_rewards = prev_rewards_buffer.reshape(-1, 1)
        b_modes = modes_buffer.reshape(-1)  # For contrastive loss
        
        if config.use_abstractions:
            b_weights = weights_buffer.reshape(-1, config.num_abstractions)
        
        # PPO update
        batch_size = config.num_envs * config.num_steps
        inds = np.arange(batch_size)
        
        clipfracs = []
        
        for epoch in range(config.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = inds[start:end]
                
                _, newlogprob, entropy, newvalue, _, mb_weights = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_prev_actions[mb_inds],
                    b_prev_rewards[mb_inds],
                    b_hiddens[mb_inds],
                    b_actions[mb_inds],
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Clipping stats
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss (clipped)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Base loss
                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                
                # Abstraction regularization
                abs_loss_info = {}
                contrast_loss_val = 0.0
                slot_pred_loss_val = 0.0
                slot_pred_acc = 0.0
                
                if config.use_abstractions and mb_weights is not None:
                    # Standard abstraction losses
                    abs_losses = compute_abstraction_losses(
                        mb_weights,
                        agent.abstraction_bank.get_abstractions(),
                        config.lambda_w_ent,
                        config.lambda_lb,
                        config.lambda_orth,
                    )
                    loss = loss + abs_losses['total']
                    abs_loss_info = abs_losses
                    
                    # Contrastive loss (if enabled)
                    if config.lambda_contrast > 0:
                        mb_modes = b_modes[mb_inds]
                        contrast_loss = compute_contrastive_slot_loss(
                            mb_weights, mb_modes, config.lambda_contrast
                        )
                        loss = loss + contrast_loss
                        contrast_loss_val = contrast_loss.item()
                    
                    # Slot prediction loss (if enabled)
                    if slot_predictor is not None:
                        slot_pred_loss = compute_slot_prediction_loss(
                            slot_predictor, b_obs[mb_inds], mb_weights, config.lambda_slot_pred
                        )
                        loss = loss + slot_pred_loss
                        slot_pred_loss_val = slot_pred_loss.item()
                        slot_pred_acc = compute_slot_prediction_accuracy(
                            slot_predictor, b_obs[mb_inds], mb_weights
                        )
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                if slot_predictor is not None:
                    nn.utils.clip_grad_norm_(slot_predictor.parameters(), config.max_grad_norm)
                optimizer.step()
        
        # Logging
        if update % config.log_interval == 0 and len(episode_returns) > 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            
            mean_return = np.mean(episode_returns[-100:])
            mean_length = np.mean(episode_lengths[-100:])
            
            print(f"\nUpdate {update}/{num_updates} | Step {global_step:,}")
            print(f"  SPS: {sps:.0f} | Time: {elapsed:.0f}s")
            print(f"  Mean return (last 100): {mean_return:.2f}")
            print(f"  Mean length (last 100): {mean_length:.2f}")
            
            # Log to tensorboard
            writer.add_scalar("charts/mean_return", mean_return, global_step)
            writer.add_scalar("charts/mean_length", mean_length, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            
            # Per-mode returns
            for mode in sorted(mode_returns.keys()):
                if len(mode_returns[mode]) > 0:
                    mode_mean = np.mean(mode_returns[mode][-20:])
                    print(f"  Mode {mode} return: {mode_mean:.2f}")
                    writer.add_scalar(f"charts/mode_{mode}_return", mode_mean, global_step)
            
            # Abstraction stats
            if config.use_abstractions:
                if abs_loss_info:
                    print(f"  Abstraction entropy: {abs_loss_info['entropy_mean']:.4f}")
                    print(f"  Avg weights: {abs_loss_info['avg_weights']}")
                    writer.add_scalar("abstractions/entropy", abs_loss_info['entropy_mean'], global_step)
                    writer.add_scalar("abstractions/L_w_ent", abs_loss_info['L_w_ent'].item(), global_step)
                    writer.add_scalar("abstractions/L_lb", abs_loss_info['L_lb'].item(), global_step)
                    writer.add_scalar("abstractions/L_orth", abs_loss_info['L_orth'].item(), global_step)
                    
                    for k in range(config.num_abstractions):
                        writer.add_scalar(f"abstractions/slot_{k}_usage", abs_loss_info['avg_weights'][k], global_step)
                
                # Temperature (if annealing)
                if temp_scheduler is not None:
                    current_tau = temp_scheduler.get_tau(global_step)
                    print(f"  Temperature: {current_tau:.3f}")
                    writer.add_scalar("abstractions/temperature", current_tau, global_step)
                
                # Contrastive loss (if enabled)
                if config.lambda_contrast > 0:
                    print(f"  Contrastive loss: {contrast_loss_val:.6f}")
                    writer.add_scalar("abstractions/L_contrast", contrast_loss_val, global_step)
                
                # Slot prediction (if enabled)
                if slot_predictor is not None:
                    print(f"  Slot pred loss: {slot_pred_loss_val:.6f}, acc: {slot_pred_acc:.2%}")
                    writer.add_scalar("abstractions/L_slot_pred", slot_pred_loss_val, global_step)
                    writer.add_scalar("abstractions/slot_pred_accuracy", slot_pred_acc, global_step)
        
        # Save checkpoint
        if update % config.save_interval == 0 or update == num_updates - 1:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_{update}.pt")
            torch.save({
                'update': update,
                'global_step': global_step,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Final save
    final_path = os.path.join(run_dir, "final_model.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    # Cleanup
    writer.close()
    envs.close()
    
    return agent


if __name__ == "__main__":
    config = parse_args()
    agent = train(config)

