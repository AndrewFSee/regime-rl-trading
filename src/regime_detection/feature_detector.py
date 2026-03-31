"""Feature-threshold based market regime detector."""
from typing import Optional

import numpy as np
import pandas as pd

from .base import MarketRegime, RegimeDetector


class FeatureRegimeDetector(RegimeDetector):
    """
    Classifies market regimes using rolling volatility, trend, and momentum
    thresholds.  No model fitting is required – ``fit`` is a no-op that returns
    ``self`` so the class satisfies the ``RegimeDetector`` interface.

    Classification rules (evaluated in priority order):
        1. volatility > volatility_high  → VOLATILE
        2. trend > trend_threshold       → BULL
        3. trend < -trend_threshold      → BEAR
        4. (otherwise)                   → SIDEWAYS
    """

    def __init__(
        self,
        volatility_low: float = 0.01,
        volatility_high: float = 0.025,
        trend_threshold: float = 0.02,
        momentum_threshold: float = 0.05,
        lookback: int = 20,
        momentum_window: int = 10,
    ) -> None:
        self.volatility_low = volatility_low
        self.volatility_high = volatility_high
        self.trend_threshold = trend_threshold
        self.momentum_threshold = momentum_threshold
        self.lookback = lookback
        self.momentum_window = momentum_window

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_close(data: pd.DataFrame) -> pd.Series:
        if "Close" in data.columns:
            return data["Close"]
        if "close" in data.columns:
            return data["close"]
        # Assume 4th column is close (OHLCV order)
        return data.iloc[:, 3]

    def _compute_signals(self, data: pd.DataFrame):
        """Return (volatility, trend, momentum) Series aligned to data index."""
        close = self._get_close(data).astype(float)
        log_ret = np.log(close / close.shift(1))

        volatility = log_ret.rolling(self.lookback, min_periods=1).std()
        trend = log_ret.rolling(self.lookback, min_periods=1).mean() * self.lookback
        momentum = (close / close.shift(self.momentum_window) - 1).fillna(0.0)

        return volatility.fillna(0.0), trend.fillna(0.0), momentum.fillna(0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "FeatureRegimeDetector":
        """No-op – feature detector requires no fitting."""
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return an array of :class:`MarketRegime` values."""
        volatility, trend, momentum = self._compute_signals(data)
        regimes: list[MarketRegime] = []
        for vol, tr, mom in zip(volatility, trend, momentum):
            if vol > self.volatility_high:
                regimes.append(MarketRegime.VOLATILE)
            elif tr > self.trend_threshold:
                regimes.append(MarketRegime.BULL)
            elif tr < -self.trend_threshold:
                regimes.append(MarketRegime.BEAR)
            else:
                regimes.append(MarketRegime.SIDEWAYS)
        return np.array(regimes)
