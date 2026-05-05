"""RecurrentPPO agent wrapping sb3-contrib for LSTM-based policy."""
from __future__ import annotations

import os

import numpy as np

from .base import BaseAgent

try:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    _SB3_CONTRIB_AVAILABLE = True
except ImportError:
    _SB3_CONTRIB_AVAILABLE = False


class RecurrentPPOAgent(BaseAgent):
    """
    Recurrent PPO agent with LSTM policy, backed by sb3-contrib.

    The LSTM policy maintains hidden state across timesteps, allowing the
    agent to learn temporal patterns without relying solely on the lookback
    window in the observation vector.

    Parameters
    ----------
    env:    A Gymnasium-compatible TradingEnv instance.
    config: Dict that may contain keys:
              learning_rate      (float, default 3e-4)
              gamma              (float, default 0.99)
              batch_size         (int,   default 64)
              n_steps            (int,   default 256)  — steps per rollout
              n_epochs           (int,   default 10)   — PPO epochs per update
              lstm_hidden_size   (int,   default 128)  — LSTM hidden units
              n_lstm_layers      (int,   default 1)    — number of LSTM layers
              checkpoint_freq    (int,   default 0 — disabled)
              checkpoint_dir     (str,   default "results/checkpoints")
              eval_freq          (int,   default 0 — disabled)
    """

    def __init__(self, env, config: dict | None = None) -> None:
        if not _SB3_CONTRIB_AVAILABLE:
            raise ImportError(
                "sb3-contrib is required for RecurrentPPOAgent. "
                "Install with: pip install sb3-contrib"
            )
        config = config or {}
        self.env = env
        self._config = config

        lstm_hidden_size = config.get("lstm_hidden_size", 128)
        n_lstm_layers = config.get("n_lstm_layers", 1)

        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=config.get("learning_rate", 3e-4),
            gamma=config.get("gamma", 0.99),
            batch_size=config.get("batch_size", 64),
            n_steps=config.get("n_steps", 256),
            n_epochs=config.get("n_epochs", 10),
            policy_kwargs=dict(
                lstm_hidden_size=lstm_hidden_size,
                n_lstm_layers=n_lstm_layers,
            ),
            verbose=0,
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return the action predicted by the policy (deterministic)."""
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def learn(self, total_timesteps: int = 100_000, callback=None) -> None:
        """Train the agent with optional callbacks."""
        built = self._build_callbacks()
        if callback is not None:
            all_cbs = [callback] + ([built] if built else [])
            built = CallbackList(all_cbs)
        self.model.learn(total_timesteps=total_timesteps, callback=built)

    def _build_callbacks(self):
        """Build SB3 callbacks from config (checkpoint + eval)."""
        callbacks = []
        ckpt_freq = self._config.get("checkpoint_freq", 0)
        if ckpt_freq > 0:
            ckpt_dir = self._config.get("checkpoint_dir", os.path.join("results", "checkpoints"))
            callbacks.append(
                CheckpointCallback(save_freq=ckpt_freq, save_path=ckpt_dir, name_prefix="rppo")
            )
        eval_freq = self._config.get("eval_freq", 0)
        if eval_freq > 0:
            best_dir = self._config.get("checkpoint_dir", os.path.join("results", "checkpoints"))
            callbacks.append(
                EvalCallback(
                    self.env,
                    eval_freq=eval_freq,
                    best_model_save_path=best_dir,
                    n_eval_episodes=3,
                    deterministic=True,
                    verbose=0,
                )
            )
        return CallbackList(callbacks) if callbacks else None

    def save(self, path: str) -> None:
        """Save model weights to *path*."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load model weights from *path*."""
        self.model = RecurrentPPO.load(path, env=self.env)
