"""Early-stopping callback for Stable-Baselines3 agents.

Periodically evaluates the agent on a held-out validation environment
and keeps the best model weights (by total return).  Training is stopped
early if no improvement is seen for ``patience`` consecutive evaluations.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


if _SB3_AVAILABLE:
    class EarlyStoppingCallback(BaseCallback):
        """
        Parameters
        ----------
        val_env:        Validation TradingEnv (separate data from training).
        eval_freq:      Evaluate every *eval_freq* training timesteps.
        patience:       Stop after this many evaluations with no improvement.
        min_timesteps:  Never stop before this many total timesteps.
        verbose:        Print eval results if > 0.
        """

        def __init__(
            self,
            val_env,
            eval_freq: int = 10_000,
            patience: int = 10,
            min_timesteps: int = 50_000,
            verbose: int = 0,
        ) -> None:
            super().__init__(verbose)
            self.val_env = val_env
            self.eval_freq = eval_freq
            self.patience = patience
            self.min_timesteps = min_timesteps

            self.best_return: float = -np.inf
            self.best_params: dict[str, Any] | None = None
            self.no_improve_count: int = 0
            self.eval_count: int = 0
            self.eval_history: list[dict] = []

        def _on_step(self) -> bool:
            if self.num_timesteps % self.eval_freq != 0:
                return True

            # Run one episode on validation env
            val_return = self._evaluate()
            self.eval_count += 1
            self.eval_history.append({
                "timestep": self.num_timesteps,
                "return": val_return,
            })

            if val_return > self.best_return:
                self.best_return = val_return
                self.best_params = copy.deepcopy(self.model.policy.state_dict())
                self.no_improve_count = 0
                if self.verbose > 0:
                    print(f"  [ES] step {self.num_timesteps}: new best val return {val_return:+.2%}")
            else:
                self.no_improve_count += 1
                if self.verbose > 0:
                    print(f"  [ES] step {self.num_timesteps}: val return {val_return:+.2%} "
                          f"(no improve {self.no_improve_count}/{self.patience})")

            # Don't stop before min_timesteps
            if self.num_timesteps < self.min_timesteps:
                return True

            # Stop if patience exhausted
            if self.no_improve_count >= self.patience:
                if self.verbose > 0:
                    print(f"  [ES] Early stopping at step {self.num_timesteps} "
                          f"(best return {self.best_return:+.2%} at eval #{self.eval_count - self.no_improve_count})")
                return False

            return True

        def _evaluate(self) -> float:
            """Run one deterministic episode on the validation env and return total return."""
            obs, _ = self.val_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.val_env.step(action)
                done = terminated or truncated
            # Total return from the environment
            final_value = info.get("portfolio_value", self.val_env.initial_cash)
            return (final_value / self.val_env.initial_cash) - 1.0

        def restore_best(self) -> None:
            """Load the best policy weights back into the model."""
            if self.best_params is not None:
                self.model.policy.load_state_dict(self.best_params)
