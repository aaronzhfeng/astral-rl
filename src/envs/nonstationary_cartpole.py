# src/envs/nonstationary_cartpole.py
"""
NonStationaryCartPole: CartPole with hidden modes that change dynamics.

This environment wraps the standard CartPole-v1 and introduces non-stationarity
by changing physical parameters (gravity, pole length) based on a hidden "mode".

The agent does NOT observe the mode — it must infer it from the dynamics.
Mode is revealed in info["mode"] for analysis/logging only.

Modes:
    0: Default (gravity=9.8, length=0.5)
    1: Easy (gravity=7.5, length=0.7) - slower fall, longer pole
    2: Hard (gravity=12.0, length=0.4) - faster fall, shorter pole
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple


class NonStationaryCartPole(gym.Wrapper):
    """
    CartPole with hidden modes that change dynamics per episode.
    
    The agent must adapt its behavior based on observed dynamics,
    not explicit mode information.
    
    Attributes:
        MODES: Dictionary mapping mode index to physical parameters
        fixed_mode: If set, always use this mode (for testing)
        current_mode: The mode for the current episode
        num_modes: Total number of modes (K=3)
    """
    
    MODES: Dict[int, Dict[str, Any]] = {
        0: {"gravity": 9.8,  "length": 0.5, "name": "default"},
        1: {"gravity": 7.5,  "length": 0.7, "name": "easy"},
        2: {"gravity": 12.0, "length": 0.4, "name": "hard"},
    }
    
    def __init__(self, env: gym.Env, mode: Optional[int] = None):
        """
        Initialize NonStationaryCartPole wrapper.
        
        Args:
            env: Base CartPole environment (CartPole-v1)
            mode: If None, sample random mode each episode.
                  If int (0, 1, or 2), fix to that mode (for testing/TTA).
        """
        super().__init__(env)
        self.fixed_mode = mode
        self.current_mode: Optional[int] = None
        self.num_modes = len(self.MODES)
        
        # Store original parameters for reference
        self._original_gravity = self.env.unwrapped.gravity
        self._original_length = self.env.unwrapped.length
        
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment and sample/set mode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (can include "mode" to override)
            
        Returns:
            observation: Initial state
            info: Contains "mode" and "mode_name" for logging
        """
        # Handle seed
        if seed is not None:
            np.random.seed(seed)
        
        # Determine mode for this episode
        if options is not None and "mode" in options:
            # Allow override via options
            self.current_mode = options["mode"]
        elif self.fixed_mode is not None:
            # Use fixed mode if set
            self.current_mode = self.fixed_mode
        else:
            # Sample random mode
            self.current_mode = np.random.randint(0, self.num_modes)
        
        # Apply dynamics parameters
        params = self.MODES[self.current_mode]
        self.env.unwrapped.gravity = params["gravity"]
        self.env.unwrapped.length = params["length"]
        
        # Update derived parameters
        # polemass_length = masspole * length (used in physics calculations)
        self.env.unwrapped.polemass_length = (
            self.env.unwrapped.masspole * params["length"]
        )
        # total_mass stays the same
        # Note: We don't modify masspole or masscart
        
        # Reset the base environment
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Add mode info for logging (agent should NOT use this!)
        info["mode"] = self.current_mode
        info["mode_name"] = params["name"]
        info["gravity"] = params["gravity"]
        info["pole_length"] = params["length"]
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: 0 (push left) or 1 (push right)
            
        Returns:
            observation: Next state
            reward: +1 for each step survived
            terminated: True if pole fell or cart out of bounds
            truncated: True if episode length exceeded
            info: Contains mode information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add mode info
        info["mode"] = self.current_mode
        info["mode_name"] = self.MODES[self.current_mode]["name"]
        
        return obs, reward, terminated, truncated, info
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Return current mode information (for debugging/analysis)."""
        if self.current_mode is None:
            return {"mode": None, "mode_name": "not_set"}
        return {
            "mode": self.current_mode,
            **self.MODES[self.current_mode]
        }


def make_nonstationary_cartpole(
    mode: Optional[int] = None,
    record_episode_stats: bool = False,
) -> NonStationaryCartPole:
    """
    Factory function to create NonStationaryCartPole environment.
    
    Args:
        mode: If None, sample random mode each episode.
              If int (0, 1, or 2), fix to that mode.
        record_episode_stats: If True, wrap with RecordEpisodeStatistics.
              
    Returns:
        NonStationaryCartPole environment instance
        
    Example:
        # Random mode each episode (for training)
        env = make_nonstationary_cartpole()
        
        # Fixed mode (for testing/TTA)
        env = make_nonstationary_cartpole(mode=2)  # Always hard mode
    """
    base_env = gym.make("CartPole-v1")
    env = NonStationaryCartPole(base_env, mode=mode)
    
    if record_episode_stats:
        from gymnasium.wrappers import RecordEpisodeStatistics
        env = RecordEpisodeStatistics(env)
    
    return env


def make_vectorized_nonstationary_cartpole(
    num_envs: int, 
    mode: Optional[int] = None,
) -> gym.vector.SyncVectorEnv:
    """
    Create vectorized NonStationaryCartPole environments with episode tracking.
    
    Args:
        num_envs: Number of parallel environments
        mode: If None, each env samples random mode per episode.
              If int, all envs use that fixed mode.
              
    Returns:
        SyncVectorEnv with num_envs NonStationaryCartPole instances
        (wrapped with RecordEpisodeStatistics for tracking)
    """
    def make_env():
        return make_nonstationary_cartpole(mode=mode, record_episode_stats=True)
    
    return gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])


# ============================================================================
# Test Script
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing NonStationaryCartPole")
    print("=" * 60)
    
    # Test 1: Random mode sampling
    print("\n[Test 1] Random mode sampling (5 episodes)")
    print("-" * 40)
    env = make_nonstationary_cartpole()
    
    for episode in range(5):
        obs, info = env.reset()
        print(f"Episode {episode}: Mode {info['mode']} ({info['mode_name']})")
        print(f"  Gravity: {info['gravity']}, Pole Length: {info['pole_length']}")
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        print(f"  Total reward: {total_reward}, Steps: {steps}")
    
    env.close()
    
    # Test 2: Fixed mode
    print("\n[Test 2] Fixed mode (mode=2, hard)")
    print("-" * 40)
    env = make_nonstationary_cartpole(mode=2)
    
    for episode in range(3):
        obs, info = env.reset()
        print(f"Episode {episode}: Mode {info['mode']} ({info['mode_name']})")
        assert info['mode'] == 2, "Mode should be fixed to 2!"
    
    env.close()
    print("Fixed mode test passed!")
    
    # Test 3: Mode override via options
    print("\n[Test 3] Mode override via reset options")
    print("-" * 40)
    env = make_nonstationary_cartpole()  # Random mode by default
    
    for override_mode in [0, 1, 2]:
        obs, info = env.reset(options={"mode": override_mode})
        print(f"Requested mode {override_mode}: Got mode {info['mode']} ({info['mode_name']})")
        assert info['mode'] == override_mode, "Mode override failed!"
    
    env.close()
    print("Mode override test passed!")
    
    # Test 4: Vectorized environment
    print("\n[Test 4] Vectorized environment (4 envs)")
    print("-" * 40)
    vec_env = make_vectorized_nonstationary_cartpole(num_envs=4)
    
    obs, infos = vec_env.reset()
    print(f"Observation shape: {obs.shape}")
    # In vectorized envs, infos is a dict with arrays
    if "mode" in infos:
        print(f"Modes: {infos['mode']}")
    else:
        print("Mode info: (vectorized env info structure varies)")
    
    # Take a few steps
    for step in range(10):
        actions = vec_env.action_space.sample()
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    
    vec_env.close()
    print("Vectorized environment test passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

