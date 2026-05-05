"""PPO agent wrapping Stable-Baselines3."""
from __future__ import annotations

import os

import numpy as np

from .base import BaseAgent
from ._env_io import _save_env_normalizer, _load_env_normalizer

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
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
              learning_rate      (float, default 3e-4)
              gamma              (float, default 0.99)
              batch_size         (int,   default 64)
              checkpoint_freq    (int,   default 0 — disabled)
              checkpoint_dir     (str,   default "results/checkpoints")
              eval_freq          (int,   default 0 — disabled)
              early_stop_patience(int,   default 5)
    """

    def __init__(self, env, config: dict | None = None) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for PPOAgent. "
                "Install with: pip install stable-baselines3"
            )
        config = config or {}
        self.env = env
        self._config = config
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get("learning_rate", 3e-4),
            gamma=config.get("gamma", 0.99),
            batch_size=config.get("batch_size", 64),
            tensorboard_log=config.get("tensorboard_log"),
            verbose=int(config.get("verbose", 0)),
        )

    def _build_callbacks(self):
        """Build SB3 callbacks from config (checkpoint + eval/early-stop)."""
        callbacks = []
        ckpt_freq = self._config.get("checkpoint_freq", 0)
        if ckpt_freq > 0:
            ckpt_dir = self._config.get("checkpoint_dir", os.path.join("results", "checkpoints"))
            callbacks.append(
                CheckpointCallback(save_freq=ckpt_freq, save_path=ckpt_dir, name_prefix="ppo")
            )

        eval_freq = self._config.get("eval_freq", 0)
        eval_env = self._config.get("eval_env")
        if eval_freq > 0 and eval_env is not None:
            best_dir = self._config.get("checkpoint_dir", os.path.join("results", "checkpoints"))
            callbacks.append(
                EvalCallback(
                    eval_env,
                    eval_freq=eval_freq,
                    best_model_save_path=best_dir,
                    n_eval_episodes=3,
                    deterministic=True,
                    verbose=0,
                )
            )
        elif eval_freq > 0 and eval_env is None:
            # Skip silently rather than evaluating on the training env, which
            # would produce a meaningless "best model" signal.
            pass
        return CallbackList(callbacks) if callbacks else None

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return the action predicted by the policy (deterministic)."""
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def learn(self, total_timesteps: int = 100_000) -> None:
        """Train the agent with optional checkpointing and evaluation callbacks."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self._build_callbacks(),
            tb_log_name=self._config.get("tb_log_name", "PPO"),
        )

    def save(self, path: str) -> None:
        """Save model weights to *path* (SB3 appends ``.zip`` automatically).
        Also persists the env's observation-normalizer state to ``<path>.norm.npz``
        so evaluation can restore the same statistics."""
        self.model.save(path)
        _save_env_normalizer(self.env, path)

    def load(self, path: str) -> None:
        """Load model weights from *path* and, if present, restore the
        normalizer state from the sidecar produced by ``save``. The env's
        normalizer is then frozen (training=False) for evaluation."""
        self.model = PPO.load(path, env=self.env)
        _load_env_normalizer(self.env, path, freeze=True)
