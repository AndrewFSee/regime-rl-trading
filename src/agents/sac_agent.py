"""SAC agent wrapping Stable-Baselines3 for continuous strategy-weight allocation."""
from __future__ import annotations

import os

import numpy as np

from .base import BaseAgent
from ._env_io import _save_env_normalizer, _load_env_normalizer

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic agent backed by Stable-Baselines3.

    SAC is an off-policy, maximum-entropy RL algorithm that natively supports
    continuous action spaces.  This makes it a natural fit for the TradingEnv's
    ``Box(0, 1, (n_strategies,))`` action space — no discretisation wrapper is
    needed, and the agent can learn smooth weight blends across strategies.

    Parameters
    ----------
    env:    A Gymnasium-compatible TradingEnv instance.
    config: Dict that may contain keys:
              learning_rate  (float, default 3e-4)
              gamma          (float, default 0.99)
              batch_size     (int,   default 64)
              tau             (float, default 0.005)
              ent_coef       (str|float, default "auto")
              buffer_size    (int,   default 10_000)
              train_freq     (int,   default 1)  — env steps between gradient updates
              gradient_steps (int,   default 1)  — gradient steps per update
    """

    def __init__(self, env, config: dict | None = None) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for SACAgent. "
                "Install with: pip install stable-baselines3"
            )
        config = config or {}
        self.env = env
        self._config = config

        # Optional MLP architecture override (for capacity / overfit experiments).
        net_arch = config.get("net_arch")
        policy_kwargs = config.get("policy_kwargs")
        if net_arch is not None and policy_kwargs is None:
            policy_kwargs = {"net_arch": list(net_arch)}

        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=config.get("learning_rate", 3e-4),
            gamma=config.get("gamma", 0.99),
            batch_size=config.get("batch_size", 64),
            tau=config.get("tau", 0.005),
            ent_coef=config.get("ent_coef", "auto"),
            buffer_size=config.get("buffer_size", 10_000),
            train_freq=config.get("train_freq", 1),
            gradient_steps=config.get("gradient_steps", 1),
            seed=config.get("seed", None),
            tensorboard_log=config.get("tensorboard_log"),
            policy_kwargs=policy_kwargs,
            verbose=int(config.get("verbose", 0)),
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return the action predicted by the policy (deterministic)."""
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def learn(self, total_timesteps: int = 100_000, callback=None) -> None:
        """Train the agent with optional checkpointing and evaluation callbacks."""
        built = self._build_callbacks()
        if callback is not None:
            from stable_baselines3.common.callbacks import CallbackList
            all_cbs = [callback] + ([built] if built else [])
            built = CallbackList(all_cbs)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=built,
            tb_log_name=self._config.get("tb_log_name", "SAC"),
        )

    def _build_callbacks(self):
        """Build SB3 callbacks from config (checkpoint + eval)."""
        callbacks = []
        ckpt_freq = self._config.get("checkpoint_freq", 0)
        if ckpt_freq > 0:
            ckpt_dir = self._config.get("checkpoint_dir", os.path.join("results", "checkpoints"))
            callbacks.append(
                CheckpointCallback(save_freq=ckpt_freq, save_path=ckpt_dir, name_prefix="sac")
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
        return CallbackList(callbacks) if callbacks else None

    def save(self, path: str) -> None:
        """Save model weights to *path* (SB3 appends ``.zip`` automatically).
        Also persists the env's observation-normalizer state to ``<path>.norm.npz``."""
        self.model.save(path)
        _save_env_normalizer(self.env, path)

    def load(self, path: str) -> None:
        """Load model weights from *path* and, if present, restore the
        normalizer state from the sidecar produced by ``save``; the env's
        normalizer is then frozen for evaluation."""
        self.model = SAC.load(path, env=self.env)
        _load_env_normalizer(self.env, path, freeze=True)
