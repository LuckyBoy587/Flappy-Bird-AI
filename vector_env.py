"""
Vectorized Flappy Bird Environment for Multi-Bird Training.
Allows training multiple birds in parallel with a shared DQN and replay buffer.
"""

import numpy as np
from typing import List, Tuple, Any, Dict
from flappy_bird_env import FlappyBirdEnv
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Iterable


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


class ParallelVectorFlappyBirdEnv:
    """
    Parallel vectorized wrapper that runs each `FlappyBirdEnv` in its own process.

    Notes / trade-offs:
    - Uses multiprocessing.Process + Pipe to avoid GIL limits and run CPU-bound
      environment logic in parallel.
    - Rendering is NOT supported in parallel mode (pygame and many display
      backends require the main process). Keep `render_mode=False` for workers.
    - On Windows the spawn start method is used by default; the worker function
      is defined at module scope so it can be pickled.

    API mirrors `VectorFlappyBirdEnv`:
      reset() -> np.ndarray
      step(actions) -> (next_states, rewards, dones, infos)
      close()

    """

    def __init__(self, num_envs: int = 8, initial_flap: bool = True):
        self.num_envs = int(num_envs)
        self.initial_flap = bool(initial_flap)
        self._parent_conns: List[Connection] = []
        self._processes: List[mp.Process] = []

        # Start worker processes
        for i in range(self.num_envs):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=_env_worker, args=(child_conn, initial_flap), daemon=True)
            p.start()
            child_conn.close()  # close child end in parent
            self._parent_conns.append(parent_conn)
            self._processes.append(p)

        # Probe one worker to extract dims
        # Send a reset to each and collect states
        states = self.reset()
        # derive dims
        self.state_dim = states.shape[1]
        # ask one worker for action count via a small info request
        # Not all envs expose ACTION_COUNT via pipe; infer from FlappyBirdEnv constant
        try:
            self.action_dim = FlappyBirdEnv.ACTION_COUNT
        except Exception:
            self.action_dim = 2

    def reset(self) -> np.ndarray:
        """Reset all remote environments and return stacked states."""
        for conn in self._parent_conns:
            conn.send(("reset", None))

        states = [conn.recv() for conn in self._parent_conns]
        return np.stack(states, axis=0)

    def step(self, actions: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all envs in parallel. Actions should be an iterable of length num_envs.

        Returns the same shapes as `VectorFlappyBirdEnv.step`.
        """
        actions = list(actions)
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")

        # Send commands to all workers
        for conn, a in zip(self._parent_conns, actions):
            conn.send(("step", int(a)))

        # Collect results
        next_states = []
        rewards = []
        dones = []
        infos = []

        for conn in self._parent_conns:
            ns, r, d, info = conn.recv()
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
            infos.append(info or {})

        return (
            np.stack(next_states, axis=0),
            np.array(rewards, dtype=float),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self):
        """Tell all workers to close and join them."""
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass

        for p in self._processes:
            p.join(timeout=1.0)


def _env_worker(conn: Connection, initial_flap: bool):
    """Worker that runs a single FlappyBirdEnv and speaks over a Pipe.

    Commands (tuples) received from parent:
      ("reset", None) -> sends back state (np.ndarray)
      ("step", action:int) -> performs step, auto-reset if done, sends back (ns, r, d, info)
      ("close", None) -> closes env and exits

    The worker does NOT attempt to render; creating a window from a subprocess
    can fail on many platforms.
    """
    try:
        env = FlappyBirdEnv(render_mode=False, initial_flap=initial_flap)
    except Exception:
        # If env construction fails, send failures back until closed
        env = None

    while True:
        try:
            cmd, payload = conn.recv()
        except EOFError:
            break

        if cmd == "reset":
            if env is None:
                conn.send(np.zeros((env.STATE_COUNT if env is not None else 1,), dtype=float))
                continue
            s = env.reset()
            conn.send(s)
        elif cmd == "step":
            if env is None:
                # Return dummy values
                conn.send((np.zeros((1,), dtype=float), 0.0, True, {}))
                continue
            ns, r, d, info = env.step(int(payload))
            if d:
                # auto-reset so next step starts a fresh episode
                try:
                    _ = env.reset()
                except Exception:
                    pass
            conn.send((ns, r, d, info))
        elif cmd == "close":
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            break
        else:
            # unknown command, reply None
            conn.send(None)
