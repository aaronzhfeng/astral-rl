# ASTRAL Implementation Plan

**ASTRAL: Abstraction-Slot Test-time Reweighting for Adaptation in Latent RL**

A small-scale proof-of-concept for structured abstractions in in-context RL.

---

## Overview

| Component | Choice |
|:----------|:-------|
| **Base Algorithm** | PPO (on-policy RL) |
| **Backbone** | GRU (in-context via hidden state) |
| **Environment** | NonStationaryCartPole (3 modes) |
| **Codebase** | CleanRL (modify `ppo_atari_lstm.py`) |
| **Hardware** | Single GPU or CPU (CartPole is fast) |
| **Timeline** | ~7 days |

---

## Architecture Summary

```
Input: (s_t, a_{t-1}, r_{t-1})
           │
           ▼
    ┌─────────────┐
    │  Input MLP  │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    GRU      │ ← In-context adaptation via hidden state
    └──────┬──────┘
           │
           ▼
         h_t (context embedding)
           │
           ├─────────────────────────┐
           │                         │
           ▼                         ▼
    ┌─────────────┐          ┌───────────────┐
    │ Gating MLP  │          │ Abstraction   │
    │  g(h_t)     │          │ Bank A [K×d]  │
    └──────┬──────┘          └───────┬───────┘
           │                         │
           ▼                         │
    w_t = softmax(logits/τ)          │
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
              z_t = w_t^T · A  (combined abstraction)
                        │
                        ▼
                 ┌─────────────┐
                 │    FiLM     │
                 │  γ, β = f(z)│
                 └──────┬──────┘
                        │
                        ▼
              h'_t = γ ⊙ h_t + β  (modulated context)
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
     ┌─────────────┐         ┌─────────────┐
     │ Policy Head │         │ Value Head  │
     │   (MLP)     │         │   (MLP)     │
     └──────┬──────┘         └──────┬──────┘
            │                       │
            ▼                       ▼
      action logits              value
```

**Key Design**: Only h'_t (modulated by abstraction) reaches the heads. No bypass from raw h_t.

---

## Phase 1: Setup (Day 1)

### 1.1 Clone and Setup CleanRL

```bash
# Create project directory
mkdir astral && cd astral

# Clone CleanRL
git clone https://github.com/vwxyzjn/cleanrl.git
cd cleanrl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 1.2 Install Dependencies

> **⚠️ Known Issue**: CleanRL's `requirements.txt` requires Python <3.11.
> If you have Python 3.11+, you'll get: `ERROR: Package 'cleanrl' requires a different Python: 3.11.x not in '<3.11,>=3.8'`

**Option A: If requirements.txt works (Python 3.8-3.10)**
```bash
pip install -r requirements/requirements.txt
```

**Option B: Manual install (Python 3.11+) — RECOMMENDED**
```bash
# Core dependencies for ASTRAL
pip install torch gymnasium numpy tensorboard wandb

# tyro is required for CleanRL's argument parsing
pip install tyro
```

**Verify installation:**
```bash
python -c "import torch; import gymnasium; import tyro; print('torch:', torch.__version__); print('gymnasium:', gymnasium.__version__); print('All imports OK!')"
```

### 1.3 Verify Base PPO Works

```bash
# Test basic PPO on CartPole
python cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 50000

# Should solve CartPole (reward ~500) in a few minutes
```

### 1.4 Verify Recurrent PPO Works

```bash
# Test LSTM version (we'll modify this)
python cleanrl/ppo_atari_lstm.py --env-id CartPole-v1 --total-timesteps 50000
```

### 1.5 Project Structure

```
astral/
├── cleanrl/                    # Cloned CleanRL repo
├── src/
│   ├── envs/
│   │   └── nonstationary_cartpole.py
│   ├── models/
│   │   ├── abstraction_bank.py
│   │   ├── film.py
│   │   └── astral_agent.py
│   ├── train_baseline.py       # GRU-only baseline
│   ├── train_astral.py         # ASTRAL training
│   ├── test_time_adapt.py      # TTA experiments
│   └── interventions.py        # Clamp/disable experiments
├── configs/
│   └── default.yaml
├── scripts/
│   ├── run_baseline.sh
│   ├── run_astral.sh
│   └── run_tta.sh
└── results/
    └── (logs, checkpoints, plots)
```

---

## Phase 2: Environment (Day 1-2)

### 2.1 NonStationaryCartPole Implementation

```python
# src/envs/nonstationary_cartpole.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NonStationaryCartPole(gym.Wrapper):
    """
    CartPole with hidden modes that change dynamics.
    
    Mode 0: gravity=9.8,  length=0.5  (default)
    Mode 1: gravity=7.5,  length=0.7  (easier - slower, longer pole)
    Mode 2: gravity=12.0, length=0.4  (harder - faster, shorter pole)
    
    The agent does NOT observe the mode - only (s, a, r).
    Mode is revealed in info["mode"] for analysis only.
    """
    
    MODES = {
        0: {"gravity": 9.8,  "length": 0.5, "name": "default"},
        1: {"gravity": 7.5,  "length": 0.7, "name": "easy"},
        2: {"gravity": 12.0, "length": 0.4, "name": "hard"},
    }
    
    def __init__(self, env, mode=None):
        """
        Args:
            env: Base CartPole environment
            mode: If None, sample random mode each episode.
                  If int, fix to that mode (for testing).
        """
        super().__init__(env)
        self.fixed_mode = mode
        self.current_mode = None
        self.num_modes = len(self.MODES)
    
    def reset(self, **kwargs):
        # Sample or fix mode
        if self.fixed_mode is not None:
            self.current_mode = self.fixed_mode
        else:
            self.current_mode = np.random.randint(0, self.num_modes)
        
        # Apply dynamics parameters
        params = self.MODES[self.current_mode]
        self.env.unwrapped.gravity = params["gravity"]
        self.env.unwrapped.length = params["length"]
        # Also update masspole and polemass_length for consistency
        self.env.unwrapped.polemass_length = (
            self.env.unwrapped.masspole * params["length"]
        )
        
        # Reset environment
        obs, info = self.env.reset(**kwargs)
        info["mode"] = self.current_mode
        info["mode_name"] = params["name"]
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["mode"] = self.current_mode
        info["mode_name"] = self.MODES[self.current_mode]["name"]
        return obs, reward, terminated, truncated, info


def make_nonstationary_cartpole(mode=None):
    """Factory function for creating the environment."""
    env = gym.make("CartPole-v1")
    env = NonStationaryCartPole(env, mode=mode)
    return env


# Test the environment
if __name__ == "__main__":
    env = make_nonstationary_cartpole()
    
    for episode in range(5):
        obs, info = env.reset()
        print(f"Episode {episode}: Mode {info['mode']} ({info['mode_name']})")
        
        total_reward = 0
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"  Total reward: {total_reward}, Steps: {step+1}")
    
    env.close()
```

### 2.2 Verify Environment

```bash
python src/envs/nonstationary_cartpole.py
# Should print episodes with different modes and varying difficulty
```

---

## Phase 3: Models (Day 2-3)

### 3.1 Abstraction Bank

```python
# src/models/abstraction_bank.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AbstractionBank(nn.Module):
    """
    Bank of K learnable abstraction vectors with gating mechanism.
    
    Given context h, outputs:
    - z: combined abstraction (weighted sum of slots)
    - w: mixture weights over slots (for analysis/regularization)
    """
    
    def __init__(
        self,
        d_model: int,
        num_abstractions: int = 3,
        tau: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_abstractions = num_abstractions
        self.tau = tau
        
        # K learnable abstraction vectors [K, d]
        self.abstractions = nn.Parameter(
            torch.randn(num_abstractions, d_model) * 0.02
        )
        
        # Gating network: h -> logits over K slots
        self.gating = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_abstractions),
        )
    
    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [batch, d_model] context embedding from GRU
            
        Returns:
            z: [batch, d_model] combined abstraction
            w: [batch, K] mixture weights (softmax)
        """
        # Compute gating logits
        logits = self.gating(h)  # [batch, K]
        
        # Softmax with temperature
        w = F.softmax(logits / self.tau, dim=-1)  # [batch, K]
        
        # Weighted sum of abstractions
        z = torch.einsum('bk,kd->bd', w, self.abstractions)  # [batch, d]
        
        return z, w
    
    def get_abstractions(self) -> torch.Tensor:
        """Return the abstraction matrix for regularization."""
        return self.abstractions
```

### 3.2 FiLM Modulation

```python
# src/models/film.py

import torch
import torch.nn as nn

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    
    Given abstraction z, produces (gamma, beta) to modulate context h:
        h' = gamma * h + beta
    
    This forces the policy to depend on abstractions (no bypass).
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # z -> (gamma, beta)
        self.film_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2),  # gamma and beta
        )
    
    def forward(
        self, 
        h: torch.Tensor, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: [batch, d_model] context from GRU
            z: [batch, d_model] combined abstraction
            
        Returns:
            h_mod: [batch, d_model] modulated context
        """
        film_params = self.film_net(z)  # [batch, 2*d]
        gamma, beta = film_params.chunk(2, dim=-1)  # each [batch, d]
        
        # Modulate: h' = gamma * h + beta
        h_mod = gamma * h + beta
        
        return h_mod
```

### 3.3 ASTRAL Agent

```python
# src/models/astral_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .abstraction_bank import AbstractionBank
from .film import FiLM


class ASTRALAgent(nn.Module):
    """
    ASTRAL Agent: GRU backbone + Abstraction Bank + FiLM + Policy/Value heads.
    
    Key design:
    - GRU produces context h_t from (s, a, r) history
    - AbstractionBank produces z_t and weights w_t from h_t
    - FiLM modulates h_t with z_t to get h'_t
    - Policy and Value heads see ONLY h'_t (no bypass from h_t)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 64,
        num_abstractions: int = 3,
        tau: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_abstractions = num_abstractions
        
        # Input: (obs, prev_action_onehot, prev_reward)
        input_dim = obs_dim + action_dim + 1
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
        )
        
        # GRU backbone (in-context via hidden state)
        self.gru = nn.GRUCell(d_model, d_model)
        
        # Abstraction bank
        self.abstraction_bank = AbstractionBank(
            d_model=d_model,
            num_abstractions=num_abstractions,
            tau=tau,
        )
        
        # FiLM modulation
        self.film = FiLM(d_model)
        
        # Policy head: h'_t -> action logits
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )
        
        # Value head: h'_t -> scalar value
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step forward pass.
        
        Args:
            obs: [batch, obs_dim]
            prev_action: [batch, action_dim] one-hot
            prev_reward: [batch, 1]
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
        
        # Abstraction bank
        z, w = self.abstraction_bank(h)
        
        # FiLM modulation (key: no bypass!)
        h_mod = self.film(h, z)
        
        # Policy and value from modulated context only
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
        action: torch.Tensor = None,
    ):
        """
        Get action, log prob, entropy, value, and new hidden.
        Used during rollout and PPO update.
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


class BaselineAgent(nn.Module):
    """
    Baseline: GRU-only agent without abstraction bank.
    Same architecture but h_t goes directly to heads.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 64,
    ):
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
    
    def forward(self, obs, prev_action, prev_reward, hidden):
        x = torch.cat([obs, prev_action, prev_reward], dim=-1)
        x = self.input_proj(x)
        h = self.gru(x, hidden)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value, h
    
    def get_initial_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.d_model, device=device)
    
    def get_action_and_value(self, obs, prev_action, prev_reward, hidden, action=None):
        logits, value, new_hidden = self.forward(obs, prev_action, prev_reward, hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, new_hidden, None
```

---

## Phase 4: Training (Day 3-5)

### 4.1 Regularization Losses

```python
# src/losses.py

import torch
import torch.nn.functional as F


def compute_abstraction_losses(
    weights: torch.Tensor,
    abstractions: torch.Tensor,
    lambda_w_ent: float = 0.001,
    lambda_lb: float = 0.001,
    lambda_orth: float = 0.0001,
) -> dict:
    """
    Compute regularization losses for abstraction bank.
    
    Args:
        weights: [batch, K] mixture weights from all timesteps
        abstractions: [K, d] abstraction matrix
        lambda_*: regularization coefficients
        
    Returns:
        Dictionary of losses
    """
    K = weights.shape[-1]
    device = weights.device
    
    # 1. Per-sample entropy: encourage non-peaked distributions
    # H(w) = -sum(w * log(w))
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
    L_w_ent = -lambda_w_ent * entropy.mean()  # Negative because we maximize entropy
    
    # 2. Load balancing: encourage uniform usage across batch
    avg_weights = weights.mean(dim=0)  # [K]
    # Negative entropy of average weights (want it high = uniform)
    L_lb = lambda_lb * torch.sum(avg_weights * torch.log(avg_weights + 1e-8))
    
    # 3. Orthogonality: encourage diverse abstractions
    A_norm = F.normalize(abstractions, dim=-1)  # [K, d]
    similarity = torch.mm(A_norm, A_norm.t())  # [K, K]
    identity = torch.eye(K, device=device)
    L_orth = lambda_orth * torch.norm(similarity - identity, p='fro') ** 2
    
    return {
        'L_w_ent': L_w_ent,
        'L_lb': L_lb,
        'L_orth': L_orth,
        'entropy_mean': entropy.mean().item(),
        'avg_weights': avg_weights.detach().cpu().numpy(),
    }
```

### 4.2 Training Script (Simplified PPO)

```python
# src/train_astral.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import gymnasium as gym

from envs.nonstationary_cartpole import make_nonstationary_cartpole
from models.astral_agent import ASTRALAgent, BaselineAgent
from losses import compute_abstraction_losses


class Config:
    # Environment
    num_envs = 8
    
    # Architecture
    d_model = 64
    num_abstractions = 3
    tau = 1.0
    
    # PPO
    total_timesteps = 500_000
    num_steps = 128  # steps per rollout per env
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    update_epochs = 4
    minibatch_size = 256
    
    # Abstraction regularization
    lambda_w_ent = 0.001
    lambda_lb = 0.001
    lambda_orth = 0.0001
    
    # Logging
    log_interval = 10


def make_env():
    """Create a single environment instance."""
    return make_nonstationary_cartpole(mode=None)


def train(config: Config, use_abstractions: bool = True):
    """
    Train ASTRAL or baseline agent.
    
    Args:
        config: Training configuration
        use_abstractions: If True, train ASTRAL. If False, train baseline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create vectorized environments
    envs = gym.vector.SyncVectorEnv([make_env for _ in range(config.num_envs)])
    
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    
    # Create agent
    if use_abstractions:
        agent = ASTRALAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=config.d_model,
            num_abstractions=config.num_abstractions,
            tau=config.tau,
        ).to(device)
        print("Training ASTRAL agent")
    else:
        agent = BaselineAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=config.d_model,
        ).to(device)
        print("Training Baseline agent")
    
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate)
    
    # Storage for rollout
    obs_buffer = torch.zeros((config.num_steps, config.num_envs, obs_dim), device=device)
    actions_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    values_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    
    if use_abstractions:
        weights_buffer = torch.zeros(
            (config.num_steps, config.num_envs, config.num_abstractions), 
            device=device
        )
    
    # Initialize
    obs, info = envs.reset()
    obs = torch.tensor(obs, device=device, dtype=torch.float32)
    hidden = agent.get_initial_hidden(config.num_envs, device)
    
    # For tracking previous action/reward
    prev_action = torch.zeros((config.num_envs, action_dim), device=device)
    prev_reward = torch.zeros((config.num_envs, 1), device=device)
    
    # Training loop
    global_step = 0
    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    
    episode_returns = []
    episode_lengths = []
    mode_returns = defaultdict(list)  # Track returns per mode
    
    for update in range(num_updates):
        # Rollout
        for step in range(config.num_steps):
            global_step += config.num_envs
            
            with torch.no_grad():
                action, logprob, entropy, value, new_hidden, weights = agent.get_action_and_value(
                    obs, prev_action, prev_reward, hidden
                )
            
            # Store
            obs_buffer[step] = obs
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob
            values_buffer[step] = value
            if use_abstractions and weights is not None:
                weights_buffer[step] = weights
            
            # Step environment
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards_buffer[step] = torch.tensor(reward, device=device, dtype=torch.float32)
            dones_buffer[step] = torch.tensor(done, device=device, dtype=torch.float32)
            
            # Track episode stats
            if "final_info" in infos:
                for i, info_i in enumerate(infos["final_info"]):
                    if info_i is not None:
                        ep_return = info_i.get("episode", {}).get("r", 0)
                        ep_length = info_i.get("episode", {}).get("l", 0)
                        mode = info_i.get("mode", -1)
                        episode_returns.append(ep_return)
                        episode_lengths.append(ep_length)
                        mode_returns[mode].append(ep_return)
            
            # Update for next step
            obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
            
            # Update prev_action and prev_reward
            prev_action = torch.zeros((config.num_envs, action_dim), device=device)
            prev_action.scatter_(1, action.unsqueeze(1), 1.0)
            prev_reward = torch.tensor(reward, device=device, dtype=torch.float32).unsqueeze(1)
            
            # Reset hidden for done episodes
            hidden = new_hidden
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
        
        if use_abstractions:
            b_weights = weights_buffer.reshape(-1, config.num_abstractions)
        
        # PPO update
        batch_size = config.num_envs * config.num_steps
        inds = np.arange(batch_size)
        
        for epoch in range(config.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = inds[start:end]
                
                # Note: For simplicity, we're not tracking hidden states per minibatch
                # In a full implementation, you'd need to handle this properly
                mb_hidden = agent.get_initial_hidden(len(mb_inds), device)
                mb_prev_action = torch.zeros((len(mb_inds), action_dim), device=device)
                mb_prev_reward = torch.zeros((len(mb_inds), 1), device=device)
                
                _, newlogprob, entropy, newvalue, _, mb_weights = agent.get_action_and_value(
                    b_obs[mb_inds],
                    mb_prev_action,
                    mb_prev_reward,
                    mb_hidden,
                    b_actions[mb_inds],
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                
                # Abstraction regularization
                if use_abstractions and mb_weights is not None:
                    abs_losses = compute_abstraction_losses(
                        mb_weights,
                        agent.abstraction_bank.get_abstractions(),
                        config.lambda_w_ent,
                        config.lambda_lb,
                        config.lambda_orth,
                    )
                    loss = loss + abs_losses['L_w_ent'] + abs_losses['L_lb'] + abs_losses['L_orth']
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()
        
        # Logging
        if update % config.log_interval == 0 and len(episode_returns) > 0:
            mean_return = np.mean(episode_returns[-100:])
            mean_length = np.mean(episode_lengths[-100:])
            
            print(f"Update {update}/{num_updates}")
            print(f"  Global step: {global_step}")
            print(f"  Mean return (last 100): {mean_return:.2f}")
            print(f"  Mean length (last 100): {mean_length:.2f}")
            
            if use_abstractions:
                print(f"  Abstraction weights (avg): {abs_losses['avg_weights']}")
                print(f"  Weight entropy (avg): {abs_losses['entropy_mean']:.4f}")
            
            # Per-mode returns
            for mode in sorted(mode_returns.keys()):
                if len(mode_returns[mode]) > 0:
                    print(f"  Mode {mode} return: {np.mean(mode_returns[mode][-20:]):.2f}")
    
    # Save model
    torch.save(agent.state_dict(), f"results/{'astral' if use_abstractions else 'baseline'}_agent.pt")
    print("Training complete!")
    
    envs.close()
    return agent


if __name__ == "__main__":
    config = Config()
    
    # Train baseline first
    print("=" * 50)
    print("Training Baseline (GRU-only)")
    print("=" * 50)
    train(config, use_abstractions=False)
    
    # Train ASTRAL
    print("\n" + "=" * 50)
    print("Training ASTRAL")
    print("=" * 50)
    train(config, use_abstractions=True)
```

---

## Phase 5: Test-Time Adaptation (Day 5-6)

### 5.1 TTA Script

```python
# src/test_time_adapt.py

import torch
import torch.optim as optim
import numpy as np
from collections import defaultdict

from envs.nonstationary_cartpole import make_nonstationary_cartpole
from models.astral_agent import ASTRALAgent


def test_time_adapt(
    agent: ASTRALAgent,
    target_mode: int,
    num_episodes: int = 20,
    adapt_lr: float = 1e-4,
    device: torch.device = None,
):
    """
    Test-time adaptation: only update the gating network.
    
    Args:
        agent: Trained ASTRAL agent
        target_mode: Fixed mode to adapt to
        num_episodes: Number of adaptation episodes
        adapt_lr: Learning rate for gating network
        device: Torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Freeze everything except gating network
    for name, param in agent.named_parameters():
        if 'gating' in name:
            param.requires_grad = True
            print(f"Trainable: {name}")
        else:
            param.requires_grad = False
    
    # Optimizer for gating only
    gating_params = [p for n, p in agent.named_parameters() if 'gating' in n]
    optimizer = optim.Adam(gating_params, lr=adapt_lr)
    
    # Create environment with fixed mode
    env = make_nonstationary_cartpole(mode=target_mode)
    
    episode_returns = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros((1, env.action_space.n), device=device)
        prev_reward = torch.zeros((1, 1), device=device)
        
        episode_return = 0
        episode_log_probs = []
        episode_rewards = []
        
        done = False
        while not done:
            with torch.no_grad():
                action, log_prob, _, _, new_hidden, weights = agent.get_action_and_value(
                    obs, prev_action, prev_reward, hidden
                )
            
            # Store for REINFORCE update
            episode_log_probs.append(log_prob)
            
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_return += reward
            
            # Update
            obs = torch.tensor(next_obs, device=device, dtype=torch.float32).unsqueeze(0)
            prev_action = torch.zeros((1, env.action_space.n), device=device)
            prev_action[0, action.item()] = 1.0
            prev_reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
            hidden = new_hidden
        
        episode_returns.append(episode_return)
        
        # Simple REINFORCE update on gating network
        returns = []
        R = 0
        for r in reversed(episode_rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, R in zip(episode_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        print(f"Episode {episode}: Return = {episode_return:.0f}")
    
    env.close()
    
    return episode_returns


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained ASTRAL agent
    agent = ASTRALAgent(
        obs_dim=4,
        action_dim=2,
        d_model=64,
        num_abstractions=3,
    ).to(device)
    agent.load_state_dict(torch.load("results/astral_agent.pt"))
    
    # Test on each mode
    for mode in [0, 1, 2]:
        print(f"\n{'='*50}")
        print(f"Adapting to Mode {mode}")
        print(f"{'='*50}")
        
        # Reset agent to original weights
        agent.load_state_dict(torch.load("results/astral_agent.pt"))
        
        returns = test_time_adapt(agent, target_mode=mode, num_episodes=20)
        
        print(f"Mode {mode}: Mean return = {np.mean(returns):.2f}")
```

---

## Phase 6: Interpretability (Day 6-7)

### 6.1 Causal Interventions

```python
# src/interventions.py

import torch
import numpy as np
import matplotlib.pyplot as plt

from envs.nonstationary_cartpole import make_nonstationary_cartpole
from models.astral_agent import ASTRALAgent


def clamp_slot_experiment(
    agent: ASTRALAgent,
    slot_idx: int,
    mode: int,
    num_episodes: int = 10,
    device: torch.device = None,
):
    """
    Clamp mixture weights to a single slot and measure behavior.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = make_nonstationary_cartpole(mode=mode)
    
    returns = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        hidden = agent.get_initial_hidden(1, device)
        prev_action = torch.zeros((1, env.action_space.n), device=device)
        prev_reward = torch.zeros((1, 1), device=device)
        
        episode_return = 0
        done = False
        
        while not done:
            with torch.no_grad():
                # Forward through GRU
                x = torch.cat([obs, prev_action, prev_reward], dim=-1)
                x = agent.input_proj(x)
                h = agent.gru(x, hidden)
                
                # Override abstraction: clamp to single slot
                z = agent.abstraction_bank.abstractions[slot_idx].unsqueeze(0)
                
                # FiLM and heads
                h_mod = agent.film(h, z)
                logits = agent.policy_head(h_mod)
                
                action = torch.argmax(logits, dim=-1)
            
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            episode_return += reward
            
            obs = torch.tensor(next_obs, device=device, dtype=torch.float32).unsqueeze(0)
            prev_action = torch.zeros((1, env.action_space.n), device=device)
            prev_action[0, action.item()] = 1.0
            prev_reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
            hidden = h
        
        returns.append(episode_return)
    
    env.close()
    return np.mean(returns), np.std(returns)


def run_intervention_experiments(agent, device):
    """Run full intervention experiment matrix."""
    results = {}
    
    for mode in [0, 1, 2]:
        results[mode] = {}
        for slot in range(agent.num_abstractions):
            mean_ret, std_ret = clamp_slot_experiment(agent, slot, mode, device=device)
            results[mode][slot] = (mean_ret, std_ret)
            print(f"Mode {mode}, Slot {slot}: {mean_ret:.1f} ± {std_ret:.1f}")
    
    return results


def plot_intervention_heatmap(results, save_path="results/intervention_heatmap.png"):
    """Plot intervention results as heatmap."""
    modes = sorted(results.keys())
    slots = sorted(results[modes[0]].keys())
    
    matrix = np.array([[results[m][s][0] for s in slots] for m in modes])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Mean Return')
    plt.xlabel('Slot Index')
    plt.ylabel('Mode')
    plt.xticks(range(len(slots)), [f'Slot {s}' for s in slots])
    plt.yticks(range(len(modes)), [f'Mode {m}' for m in modes])
    plt.title('Intervention Experiment: Clamp Single Slot')
    
    for i in range(len(modes)):
        for j in range(len(slots)):
            plt.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap to {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = ASTRALAgent(obs_dim=4, action_dim=2, d_model=64, num_abstractions=3).to(device)
    agent.load_state_dict(torch.load("results/astral_agent.pt"))
    agent.eval()
    
    print("Running intervention experiments...")
    results = run_intervention_experiments(agent, device)
    
    plot_intervention_heatmap(results)
```

---

## Phase 7: Run All Experiments (Day 7)

### 7.1 Experiment Scripts

```bash
# scripts/run_all.sh

#!/bin/bash

# Create results directory
mkdir -p results

# 1. Train baseline
echo "Training Baseline..."
python src/train_astral.py --use_abstractions False

# 2. Train ASTRAL
echo "Training ASTRAL..."
python src/train_astral.py --use_abstractions True

# 3. Test-time adaptation
echo "Running TTA experiments..."
python src/test_time_adapt.py

# 4. Intervention experiments
echo "Running intervention experiments..."
python src/interventions.py

echo "All experiments complete!"
```

---

## Summary Checklist

| Day | Task | Status |
|:----|:-----|:-------|
| **1** | Setup CleanRL, verify base PPO works | ⬜ |
| **1-2** | Implement NonStationaryCartPole | ⬜ |
| **2-3** | Implement AbstractionBank, FiLM | ⬜ |
| **3** | Implement ASTRALAgent, BaselineAgent | ⬜ |
| **3-4** | Implement training loop with regularizers | ⬜ |
| **4-5** | Train baseline and ASTRAL, debug | ⬜ |
| **5-6** | Implement and run TTA experiments | ⬜ |
| **6-7** | Implement and run intervention experiments | ⬜ |
| **7** | Analysis, plots, documentation | ⬜ |

---

## Expected Results

If ASTRAL works correctly:

1. **Training**: Both baseline and ASTRAL solve CartPole (~500 reward)
2. **Slot usage**: Different modes activate different slots
3. **TTA**: Adapting gating network improves performance on fixed mode
4. **Interventions**: Clamping slots produces consistent behavioral changes

If these don't happen:
- Check regularizers (are all slots being used?)
- Check FiLM (is it actually being used?)
- Check hidden state reset (is it proper per episode?)

