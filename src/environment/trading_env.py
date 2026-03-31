"""
Gymnasium-compatible trading environment for regime-RL strategy selection.

Observation space  : Box(-inf, inf, (lookback_window * n_features + 4 + 4,))
  - lookback_window * n_features : rolling feature history
  - 4 portfolio features          : [cash_ratio, position, unrealised_pnl, drawdown]
  - 4 regime one-hot              : [BULL, BEAR, SIDEWAYS, VOLATILE]

Action space       : Box(0, 1, (n_strategies,))
  Actions are interpreted as raw allocation weights and softmax-normalised
  internally so they always sum to 1.

Reward             : daily_return
                     - 0.1 * rolling_volatility
                     - 0.1 * current_drawdown
                     - transaction_cost * turnover
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required. Install with: pip install gymnasium"
    ) from exc

from .features import FeatureEngineer
from ..regime_detection.feature_detector import FeatureRegimeDetector
from ..regime_detection.base import MarketRegime
from ..strategies.momentum import MomentumStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.breakout import BreakoutStrategy
from ..strategies.defensive import DefensiveStrategy


_N_FEATURES = 10
_N_STRATEGIES = 4
_N_REGIME = 4
_N_PORTFOLIO = 4


class TradingEnv(gym.Env):
    """
    Custom trading environment for regime-aware RL strategy selection.

    Parameters
    ----------
    data:               OHLCV DataFrame (Open, High, Low, Close, Volume).
    lookback_window:    Number of past time-steps in the observation (default 20).
    initial_cash:       Starting portfolio value in dollars (default 100 000).
    transaction_cost:   Fractional round-trip cost per trade (default 0.001).
    reward_scaling:     Scalar multiplier applied to the reward (default 1.0).
    max_position:       Maximum absolute position fraction (default 1.0).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        lookback_window: int = 20,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        reward_scaling: float = 1.0,
        max_position: float = 1.0,
    ) -> None:
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.lookback_window = lookback_window
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.max_position = max_position

        # Pre-compute features once
        self._feature_eng = FeatureEngineer()
        self._features: np.ndarray = self._feature_eng.compute(self.data).values  # (T, 10)

        # Regime detector
        self._regime_detector = FeatureRegimeDetector()

        # Strategies
        self._strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
            DefensiveStrategy(),
        ]

        # Spaces
        obs_dim = lookback_window * _N_FEATURES + _N_PORTFOLIO + _N_REGIME
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(_N_STRATEGIES,), dtype=np.float32
        )

        # Episode state (initialised in reset)
        self._step_idx: int = lookback_window
        self._portfolio_value: float = initial_cash
        self._peak_value: float = initial_cash
        self._position: float = 0.0           # current allocation fraction
        self._return_history: list[float] = []
        self._regime_history: list[MarketRegime] = []
        self._action_history: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _get_current_regime(self) -> MarketRegime:
        window = self.data.iloc[max(0, self._step_idx - 20): self._step_idx + 1]
        try:
            regimes = self._regime_detector.predict(window)
            return regimes[-1]
        except Exception:
            return MarketRegime.SIDEWAYS

    def _regime_one_hot(self, regime: MarketRegime) -> np.ndarray:
        vec = np.zeros(_N_REGIME, dtype=np.float32)
        vec[regime.value] = 1.0
        return vec

    def _get_observation(self) -> np.ndarray:
        start = self._step_idx - self.lookback_window
        window_feats = self._features[start: self._step_idx].ravel()  # (lookback*10,)

        portfolio_feats = np.array([
            self._portfolio_value / self.initial_cash - 1.0,   # cash ratio deviation
            self._position,
            (self._portfolio_value - self.initial_cash) / self.initial_cash,  # unrealised PnL
            self._current_drawdown(),
        ], dtype=np.float32)

        regime = self._get_current_regime()
        regime_feats = self._regime_one_hot(regime)

        obs = np.concatenate([window_feats.astype(np.float32), portfolio_feats, regime_feats])
        return obs

    def _current_drawdown(self) -> float:
        self._peak_value = max(self._peak_value, self._portfolio_value)
        return (self._peak_value - self._portfolio_value) / self._peak_value

    def _rolling_volatility(self) -> float:
        if len(self._return_history) < 2:
            return 0.0
        recent = self._return_history[-20:]
        return float(np.std(recent))

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._step_idx = self.lookback_window
        self._portfolio_value = self.initial_cash
        self._peak_value = self.initial_cash
        self._position = 0.0
        self._return_history = []
        self._regime_history = []
        self._action_history = []

        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Normalise action weights via softmax
        weights = self._softmax(np.asarray(action, dtype=np.float64).ravel())

        # Gather strategy signals using the current feature vector
        current_features = self._features[self._step_idx]
        signals = np.array(
            [s.generate_signal(current_features).action for s in self._strategies],
            dtype=np.float64,
        )
        combined_signal = float(np.clip(np.dot(weights, signals), -self.max_position, self.max_position))

        # Compute daily return from close prices
        prev_close = float(self.data["Close"].iloc[self._step_idx - 1])
        curr_close = float(self.data["Close"].iloc[self._step_idx])
        market_return = (curr_close - prev_close) / prev_close

        # Portfolio return = position × market_return
        portfolio_return = self._position * market_return

        # Transaction cost based on turnover
        turnover = abs(combined_signal - self._position)
        tc_penalty = self.transaction_cost * turnover

        # Update position
        self._position = combined_signal

        # Reward components
        vol_penalty = 0.1 * self._rolling_volatility()
        dd_penalty  = 0.1 * self._current_drawdown()
        reward = float((portfolio_return - vol_penalty - dd_penalty - tc_penalty) * self.reward_scaling)

        # Update portfolio value
        self._portfolio_value *= (1.0 + portfolio_return - tc_penalty)
        self._return_history.append(portfolio_return)

        # Track history
        regime = self._get_current_regime()
        self._regime_history.append(regime)
        self._action_history.append(weights.copy())

        self._step_idx += 1
        terminated = self._step_idx >= len(self.data)
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "portfolio_value": self._portfolio_value,
            "position": self._position,
            "market_return": market_return,
            "portfolio_return": portfolio_return,
            "regime": regime,
            "drawdown": self._current_drawdown(),
        }
        return obs, reward, terminated, truncated, info
