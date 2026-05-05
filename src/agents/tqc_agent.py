"""Distributional / risk-averse SAC variant using TQC (Truncated Quantile Critics).

TQC replaces SAC's scalar Q with a quantile-regression critic that learns the
full return distribution. By dropping the top-K quantiles before computing the
target, the agent optimises a Conditional-Value-at-Risk-style objective —
the more quantiles dropped, the more risk-averse the policy.

This wraps :class:`sb3_contrib.TQC` with the same surface as
:class:`SACAgent`, including normalizer persistence and TensorBoard logging.

Config keys (in addition to those accepted by ``SACAgent``):
    top_quantiles_to_drop_per_net (int, default 2)
        Higher = more pessimistic value estimate -> more risk-averse policy.
        Set to 0 for a vanilla quantile critic, or higher (e.g. 5-10) for
        explicit CVaR-style behaviour.
    n_quantiles (int, default 25)
        Number of quantiles per critic head.
    n_critics (int, default 2)
"""
from __future__ import annotations

import numpy as np

from .base import BaseAgent
from ._env_io import _save_env_normalizer, _load_env_normalizer

try:
    from sb3_contrib import TQC
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    _TQC_AVAILABLE = True
except ImportError:
    _TQC_AVAILABLE = False

import os


class TQCAgent(BaseAgent):
    """Risk-averse, distributional SAC backed by ``sb3_contrib.TQC``."""

    def __init__(self, env, config: dict | None = None) -> None:
        if not _TQC_AVAILABLE:
            raise ImportError(
                "sb3-contrib is required for TQCAgent. "
                "Install with: pip install sb3-contrib"
            )
        config = config or {}
        self.env = env
        self._config = config

        n_quantiles = int(config.get("n_quantiles", 25))
        n_critics = int(config.get("n_critics", 2))
        top_drop = int(config.get("top_quantiles_to_drop_per_net", 2))
        if top_drop < 0 or top_drop >= n_quantiles:
            raise ValueError(
                f"top_quantiles_to_drop_per_net must satisfy "
                f"0 <= drop < n_quantiles ({n_quantiles}); got {top_drop}"
            )

        self.model = TQC(
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
            top_quantiles_to_drop_per_net=top_drop,
            policy_kwargs={"n_quantiles": n_quantiles, "n_critics": n_critics},
            seed=config.get("seed", None),
            tensorboard_log=config.get("tensorboard_log"),
            verbose=int(config.get("verbose", 0)),
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def learn(self, total_timesteps: int = 100_000, callback=None) -> None:
        built = self._build_callbacks()
        if callback is not None:
            all_cbs = [callback] + ([built] if built else [])
            built = CallbackList(all_cbs)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=built,
            tb_log_name=self._config.get("tb_log_name", "TQC"),
        )

    def save(self, path: str) -> None:
        self.model.save(path)
        _save_env_normalizer(self.env, path)

    def load(self, path: str) -> None:
        self.model = TQC.load(path, env=self.env)
        _load_env_normalizer(self.env, path, freeze=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_callbacks(self):
        callbacks = []
        ckpt_freq = self._config.get("checkpoint_freq", 0)
        if ckpt_freq > 0:
            ckpt_dir = self._config.get(
                "checkpoint_dir", os.path.join("results", "checkpoints")
            )
            callbacks.append(
                CheckpointCallback(
                    save_freq=ckpt_freq, save_path=ckpt_dir, name_prefix="tqc"
                )
            )
        eval_freq = self._config.get("eval_freq", 0)
        eval_env = self._config.get("eval_env")
        if eval_freq > 0 and eval_env is not None:
            best_dir = self._config.get(
                "checkpoint_dir", os.path.join("results", "checkpoints")
            )
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
