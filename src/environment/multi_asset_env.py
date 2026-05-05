"""Multi-asset trading environment.

A minimal portfolio environment that holds long-only positions across M
tickers. Action space is a simplex of asset weights plus a global
exposure scalar, both in ``[0, 1]``::

    raw_action = [w_1, ..., w_M, e]
    weights    = softmax(raw_action[:M])
    exposure   = sigmoid-clipped(e)
    target_pos_i = exposure * weights[i] * max_position

Reward = portfolio return − transaction cost − slippage. There are no
sub-strategies and no regime guardrail in this MVP — the focus is on
clean multi-asset bookkeeping. Use the single-asset :class:`TradingEnv`
when you need the strategy/regime stack.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False
    gym = object  # type: ignore

from .features import FeatureEngineer


_REQUIRED_COLS = ("Open", "High", "Low", "Close", "Volume")


class MultiAssetTradingEnv(gym.Env if _GYM_AVAILABLE else object):
    """Long-only portfolio env over M synchronised OHLCV frames."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: Mapping[str, pd.DataFrame],
        lookback_window: int = 20,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        slippage_bps: float = 1.0,
        max_position: float = 1.0,
        reward_scaling: float = 1.0,
    ) -> None:
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium is required for MultiAssetTradingEnv.")
        if len(data) < 2:
            raise ValueError("MultiAssetTradingEnv needs at least 2 tickers.")

        self.tickers: list[str] = list(data.keys())
        self.M = len(self.tickers)
        self.lookback_window = int(lookback_window)
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.slippage_bps = float(max(slippage_bps, 0.0))
        self.max_position = float(max_position)
        self.reward_scaling = float(reward_scaling)

        # Validate frames and align by row position (assumes pre-aligned input).
        self._frames: Dict[str, pd.DataFrame] = {}
        T_ref = None
        for tkr in self.tickers:
            df = data[tkr].reset_index(drop=True)
            for col in _REQUIRED_COLS:
                if col not in df.columns:
                    raise ValueError(
                        f"Ticker '{tkr}' missing required column '{col}'."
                    )
            if T_ref is None:
                T_ref = len(df)
            elif len(df) != T_ref:
                raise ValueError(
                    f"All tickers must have the same length; "
                    f"'{tkr}' has {len(df)} rows, expected {T_ref}."
                )
            self._frames[tkr] = df

        # Pre-compute per-asset features (T, 10) each.
        eng = FeatureEngineer()
        self._features: Dict[str, np.ndarray] = {
            tkr: eng.compute(self._frames[tkr]).values for tkr in self.tickers
        }
        self._closes: Dict[str, np.ndarray] = {
            tkr: self._frames[tkr]["Close"].astype(float).values for tkr in self.tickers
        }
        self._n_features = next(iter(self._features.values())).shape[1]
        self.T = T_ref

        if self.T <= self.lookback_window + 1:
            raise ValueError(
                f"Not enough rows ({self.T}) for lookback_window={self.lookback_window}."
            )

        # --- Spaces ---------------------------------------------------
        # Action: M asset logits + 1 exposure scalar, all in [0, 1].
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.M + 1,), dtype=np.float32
        )
        # Obs: lookback * M * n_features  +  M position ratios  +  cash deviation
        obs_dim = self.lookback_window * self.M * self._n_features + self.M + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Internal state ------------------------------------------
        self._step_idx: int = 0
        self._positions: np.ndarray = np.zeros(self.M, dtype=np.float64)
        self._portfolio_value: float = self.initial_cash

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._step_idx = self.lookback_window
        self._positions = np.zeros(self.M, dtype=np.float64)
        self._portfolio_value = self.initial_cash
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).ravel()
        if action.shape != (self.M + 1,):
            raise ValueError(
                f"Expected action shape ({self.M + 1},); got {action.shape}."
            )

        # Decode: softmax over M asset logits, exposure = clipped scalar.
        weights = self._softmax(action[: self.M])
        exposure = float(np.clip(action[self.M], 0.0, 1.0))
        target_positions = exposure * weights * self.max_position  # (M,)

        # Realised per-asset returns from prev close to current close.
        prev_idx = self._step_idx - 1
        curr_idx = self._step_idx
        per_asset_ret = np.array(
            [
                (self._closes[tkr][curr_idx] - self._closes[tkr][prev_idx])
                / self._closes[tkr][prev_idx]
                for tkr in self.tickers
            ],
            dtype=np.float64,
        )

        # Transaction cost: L1 turnover * cost_rate, plus bps slippage on turnover.
        turnover = float(np.sum(np.abs(target_positions - self._positions)))
        tc = self.transaction_cost * turnover
        slip = (self.slippage_bps / 1e4) * turnover

        # Portfolio return = dot(positions_at_open_of_period, per_asset_ret)
        # We use target_positions as the position held over the bar.
        gross_return = float(np.dot(target_positions, per_asset_ret))
        net_return = gross_return - tc - slip

        self._portfolio_value *= 1.0 + net_return
        self._positions = target_positions

        self._step_idx += 1
        terminated = self._step_idx >= self.T
        truncated = False

        reward = float(net_return * self.reward_scaling)
        info = {
            "portfolio_value": self._portfolio_value,
            "weights": weights.copy(),
            "exposure": exposure,
            "positions": self._positions.copy(),
            "per_asset_return": per_asset_ret,
            "gross_return": gross_return,
            "transaction_cost": tc,
            "slippage": slip,
            "net_return": net_return,
            "turnover": turnover,
        }
        return self._get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        # Window of features per asset, flattened: (lookback, M, F) -> (lookback*M*F,)
        start = self._step_idx - self.lookback_window
        windows = [
            self._features[tkr][start : self._step_idx]  # (lookback, F)
            for tkr in self.tickers
        ]
        feat = np.stack(windows, axis=1).reshape(-1)  # (lookback*M*F,)

        # Position ratios + cash deviation
        pos_feat = self._positions.astype(np.float32)
        cash_feat = np.array(
            [self._portfolio_value / self.initial_cash - 1.0], dtype=np.float32
        )
        return np.concatenate([feat.astype(np.float32), pos_feat, cash_feat])

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.max()
        e = np.exp(x)
        s = e.sum()
        if s <= 0 or not np.isfinite(s):
            return np.ones_like(x) / len(x)
        return e / s
