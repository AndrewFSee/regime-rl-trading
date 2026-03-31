"""Technical feature engineering for OHLCV market data."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "returns",
    "volatility",
    "short_ma",
    "long_ma",
    "bb_upper",
    "bb_lower",
    "momentum",
    "rsi",
    "atr",
    "volume_ratio",
]


class FeatureEngineer:
    """
    Computes a fixed set of 10 technical features from OHLCV data.

    All features are designed to be roughly stationary and unit-free so they
    can be fed directly into an RL observation vector.

    Feature index mapping (matches ``FEATURE_NAMES``):
        0  returns      – daily log return
        1  volatility   – 20-day rolling std of log returns
        2  short_ma     – 10-day SMA / close - 1
        3  long_ma      – 30-day SMA / close - 1
        4  bb_upper     – upper Bollinger Band (20d, 2σ) / close - 1
        5  bb_lower     – lower Bollinger Band (20d, 2σ) / close - 1
        6  momentum     – 10-day price return
        7  rsi          – RSI(14) normalised to [-1, 1]  (rsi/50 - 1)
        8  atr          – ATR(14) / close  (unitless ratio: absolute ATR divided by close price)
        9  volume_ratio – volume / 20-day avg volume
    """

    # ------------------------------------------------------------------
    # Low-level indicator helpers (all return pd.Series)
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(period, min_periods=1).mean()
        loss = (-delta.clip(upper=0.0)).rolling(period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all 10 features.

        Parameters
        ----------
        data:
            DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``,
            ``Volume`` (case-insensitive lookup attempted).

        Returns
        -------
        pd.DataFrame with columns matching ``FEATURE_NAMES``, same index as
        *data*.  Any remaining NaN values are filled with 0.
        """
        cols = {c.lower(): c for c in data.columns}

        def _get(name: str) -> pd.Series:
            for candidate in (name, name.capitalize(), name.upper()):
                if candidate in data.columns:
                    return data[candidate].astype(float)
            if name in cols:
                return data[cols[name]].astype(float)
            raise KeyError(f"Column '{name}' not found in DataFrame. Columns: {list(data.columns)}")

        close  = _get("close")
        high   = _get("high")
        low    = _get("low")
        volume = _get("volume")

        log_ret = np.log(close / close.shift(1))

        # 0 – returns
        returns = log_ret

        # 1 – volatility
        volatility = log_ret.rolling(20, min_periods=1).std()

        # 2 – short_ma  (10d SMA / price - 1)
        short_ma = close.rolling(10, min_periods=1).mean() / close - 1

        # 3 – long_ma  (30d SMA / price - 1)
        long_ma = close.rolling(30, min_periods=1).mean() / close - 1

        # 4 & 5 – Bollinger Bands  (20d, 2σ)
        bb_mid = close.rolling(20, min_periods=1).mean()
        bb_std = close.rolling(20, min_periods=1).std().fillna(0.0)
        bb_upper = (bb_mid + 2 * bb_std) / close - 1
        bb_lower = (bb_mid - 2 * bb_std) / close - 1

        # 6 – momentum  (10-day return)
        momentum = (close / close.shift(10) - 1).fillna(0.0)

        # 7 – RSI(14) normalised to [-1, 1]
        rsi_raw = self._rsi(close, period=14)
        rsi = rsi_raw / 50.0 - 1.0

        # 8 – ATR(14) / close
        atr_raw = self._atr(high, low, close, period=14)
        atr = atr_raw / close

        # 9 – volume ratio  (volume / 20d avg volume)
        avg_vol = volume.rolling(20, min_periods=1).mean().replace(0, np.nan)
        volume_ratio = (volume / avg_vol).fillna(1.0)

        features = pd.DataFrame(
            {
                "returns":      returns,
                "volatility":   volatility,
                "short_ma":     short_ma,
                "long_ma":      long_ma,
                "bb_upper":     bb_upper,
                "bb_lower":     bb_lower,
                "momentum":     momentum,
                "rsi":          rsi,
                "atr":          atr,
                "volume_ratio": volume_ratio,
            },
            index=data.index,
        )
        return features.ffill().fillna(0.0)
