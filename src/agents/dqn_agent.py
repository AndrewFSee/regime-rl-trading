"""DQN agent wrapping Stable-Baselines3 with a discrete strategy-selection wrapper."""
from __future__ import annotations

import numpy as np

from .base import BaseAgent

try:
    from stable_baselines3 import DQN
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False


class _DiscreteStrategyWrapper(gym.Wrapper):
    """
    Wraps a continuous Box-action TradingEnv so that DQN can operate over it.

    The discrete action *i* sets the *i*-th strategy weight to 1.0 and all
    others to 0.0 (pure strategy selection rather than blending).
    """

    def __init__(self, env, n_strategies: int = 4) -> None:
        super().__init__(env)
        self.n_strategies = n_strategies
        self.action_space = spaces.Discrete(n_strategies)

    def step(self, action: int):
        weights = np.zeros(self.n_strategies, dtype=np.float32)
        weights[int(action)] = 1.0
        return self.env.step(weights)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent backed by Stable-Baselines3.

    Because DQN requires a ``Discrete`` action space and :class:`TradingEnv`
    exposes a ``Box`` action space, this agent automatically wraps the
    environment with :class:`_DiscreteStrategyWrapper` to perform pure
    strategy selection (each action selects one of the four strategies).

    Parameters
    ----------
    env:    A Gymnasium-compatible TradingEnv instance.
    config: Dict that may contain keys:
              learning_rate  (float, default 1e-4)
              gamma          (float, default 0.99)
              batch_size     (int,   default 64)
              epsilon_start  (float, default 1.0)
              epsilon_end    (float, default 0.05)
    """

    def __init__(self, env, config: dict | None = None) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for DQNAgent. "
                "Install with: pip install stable-baselines3"
            )
        config = config or {}

        # Wrap env if it has a continuous action space
        wrapped_env = env
        if hasattr(env, "action_space") and not isinstance(env.action_space, spaces.Discrete):
            n_strategies = env.action_space.shape[0]
            wrapped_env = _DiscreteStrategyWrapper(env, n_strategies=n_strategies)

        self.env = wrapped_env
        self._n_strategies = wrapped_env.action_space.n

        self.model = DQN(
            "MlpPolicy",
            wrapped_env,
            learning_rate=config.get("learning_rate", 1e-4),
            gamma=config.get("gamma", 0.99),
            batch_size=config.get("batch_size", 64),
            exploration_initial_eps=config.get("epsilon_start", 1.0),
            exploration_final_eps=config.get("epsilon_end", 0.05),
            verbose=0,
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict the best strategy index and convert to a one-hot weight vector.
        """
        action_idx, _ = self.model.predict(observation, deterministic=True)
        weights = np.zeros(self._n_strategies, dtype=np.float32)
        weights[int(action_idx)] = 1.0
        return weights

    def learn(self, env=None, total_timesteps: int = 100_000) -> None:
        """Train the agent."""
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = DQN.load(path, env=self.env)
