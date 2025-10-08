"""
Vectorized Flappy Bird Environment for Multi-Bird Training.
Allows training multiple birds in parallel with a shared DQN and replay buffer.
"""

import numpy as np
from typing import List, Tuple, Any, Dict
from flappy_bird_env import FlappyBirdEnv


class VectorFlappyBirdEnv:
    """
    Simple single-process vectorized wrapper for multiple FlappyBirdEnv instances.
    
    This wrapper manages multiple independent Flappy Bird environments simultaneously,
    allowing for population-based training where multiple birds learn together.
    
    Key Features:
    - All birds share the same DQN network and replay buffer
    - Each bird explores independently, contributing diverse experiences
    - Automatic reset of individual environments when they terminate
    - Optional rendering of the first bird to monitor training
    
    Args:
        num_envs: Number of birds/environments to run in parallel (default: 8)
        render_mode: If True, render the first bird's environment (default: False)
        initial_flap: If True, automatically flap on reset (default: True)
    
    Methods:
        reset() -> np.ndarray: Reset all environments, returns (num_envs, state_dim)
        step(actions) -> Tuple: Execute actions, returns (states, rewards, dones, infos)
        render(): Render the first environment (if render_mode=True)
        close(): Close all environments
    """
    
    def __init__(self, num_envs: int = 8, render_mode: bool = False, initial_flap: bool = True):
        self.num_envs = num_envs
        # Render only the first env if render_mode True (to avoid performance issues)
        self.envs = [
            FlappyBirdEnv(render_mode=(render_mode and i == 0), initial_flap=initial_flap)
            for i in range(num_envs)
        ]
        self.state_dim = self.envs[0].STATE_COUNT
        self.action_dim = self.envs[0].ACTION_COUNT

    def reset(self) -> np.ndarray:
        """
        Reset all environments to their initial state.
        
        Returns:
            np.ndarray: Stacked initial states, shape (num_envs, state_dim)
        """
        states = [env.reset() for env in self.envs]
        return np.stack(states, axis=0)  # shape (num_envs, state_dim)

    def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Execute one step in each environment with the given actions.
        
        For any environment that terminates (done=True), it will be automatically
        reset so the next step continues with a fresh episode.
        
        Args:
            actions: Array-like of length num_envs, each element is 0 or 1
        
        Returns:
            next_states: np.ndarray of shape (num_envs, state_dim)
            rewards: np.ndarray of shape (num_envs,) - reward for each bird
            dones: np.ndarray of shape (num_envs,) - done flags (bool)
            infos: List of dicts containing additional info per environment
        """
        next_states = []
        rewards = []
        dones = []
        infos = []
        
        for env, a in zip(self.envs, actions):
            ns, r, d, info = env.step(int(a))
            rewards.append(r)
            dones.append(d)
            infos.append(info or {})
            # Store the terminal state as next_state before resetting
            next_states.append(ns)
            
            if d:
                # Auto-reset this specific env so next call continues training
                _ = env.reset()
                
        return (
            np.stack(next_states, axis=0),
            np.array(rewards, dtype=float),
            np.array(dones, dtype=bool),
            infos
        )

    def render(self):
        """
        Render the first environment (if it was created with render_mode=True).
        """
        if self.envs:
            self.envs[0].render()

    def close(self):
        """
        Close all environments and cleanup resources.
        """
        for env in self.envs:
            env.close()
