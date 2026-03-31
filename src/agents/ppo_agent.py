"""PPO agent wrapping Stable-Baselines3."""
from __future__ import annotations

import numpy as np

from .base import BaseAgent

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimisation agent backed by Stable-Baselines3.

    Parameters
    ----------
    env:    A Gymnasium-compatible environment.
    config: Dict that may contain keys:
              learning_rate  (float, default 3e-4)
              gamma          (float, default 0.99)
              batch_size     (int,   default 64)
    """

    def __init__(self, env, config: dict | None = None) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for PPOAgent. "
                "Install with: pip install stable-baselines3"
            )
        config = config or {}
        self.env = env
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get("learning_rate", 3e-4),
            gamma=config.get("gamma", 0.99),
            batch_size=config.get("batch_size", 64),
            verbose=0,
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return the action predicted by the policy (deterministic)."""
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def learn(self, env=None, total_timesteps: int = 100_000) -> None:
        """Train the agent.  ``env`` is ignored (uses the one from constructor)."""
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str) -> None:
        """Save model weights to *path* (SB3 appends ``.zip`` automatically)."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load model weights from *path*."""
        self.model = PPO.load(path, env=self.env)
