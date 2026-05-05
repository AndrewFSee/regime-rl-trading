"""Trend-following trading strategy.

Long-only filter that goes long when price is sustainably above its long-term
moving average and flat otherwise. Designed as a regime-conditional "ride the
bull market" component in the strategy router — fills the gap that the prior
``DefensiveStrategy`` (constant +0.2) left in the action menu.
"""
from __future__ import annotations

import numpy as np

from .base import TradeSignal, TradingStrategy


class TrendFollowingStrategy(TradingStrategy):
    """
    Trend-following / risk-on filter.

    Expected feature layout (matches ``FEATURE_NAMES``):
        long_ma_idx   – 30d SMA / close - 1                      [idx 3]
                        (negative ⇒ price above 30d MA  ⇒ uptrend)
        momentum_idx  – 10-day price return                      [idx 6]
        volatility_idx – 20-day rolling std of log returns       [idx 1]

    Decision rules:
        long_ma <= -BULL_THRESH AND momentum >= 0  → strong long  (+1.0)
        long_ma <= -BULL_THRESH                    → moderate long (+0.5)
        long_ma >=  BEAR_THRESH                    → flat / no-trade (0.0)
        otherwise                                  → small long bias (+0.1)

    Notes
    -----
    * ``long_ma`` is the 30d SMA divided by current close, minus 1. Values
      below zero mean price > MA (bullish trend).
    * The strategy is long-biased; it never goes short. Mean-reversion handles
      bearish counter-trends in the existing strategy menu.
    """

    BULL_THRESH = 0.005   # price > 0.5% above 30d MA → uptrend confirmed
    BEAR_THRESH = 0.01    # price > 1% below 30d MA   → downtrend, stand aside
    HIGH_VOL = 0.025      # disable trend-on bias above this vol regime

    def __init__(
        self,
        long_ma_idx: int = 3,
        momentum_idx: int = 6,
        volatility_idx: int = 1,
    ) -> None:
        self.long_ma_idx = long_ma_idx
        self.momentum_idx = momentum_idx
        self.volatility_idx = volatility_idx

    @property
    def name(self) -> str:
        return "trend_following"

    def generate_signal(self, state: np.ndarray) -> TradeSignal:
        state = np.asarray(state, dtype=float).ravel()
        n = len(state)
        long_ma  = state[self.long_ma_idx]    if self.long_ma_idx    < n else 0.0
        momentum = state[self.momentum_idx]   if self.momentum_idx   < n else 0.0
        vol      = abs(state[self.volatility_idx]) if self.volatility_idx < n else 0.0

        # In high-vol regimes, suppress trend-on bias (prevents Q1-2020-style
        # late-cycle long entries right before a crash).
        if vol > self.HIGH_VOL:
            return TradeSignal(action=0.0, confidence=0.6, strategy_name=self.name)

        # long_ma < 0  ⇒  price above 30d MA  ⇒  uptrend.
        if long_ma <= -self.BULL_THRESH and momentum >= 0:
            return TradeSignal(action=1.0, confidence=0.85, strategy_name=self.name)
        if long_ma <= -self.BULL_THRESH:
            return TradeSignal(action=0.5, confidence=0.6, strategy_name=self.name)
        if long_ma >= self.BEAR_THRESH:
            return TradeSignal(action=0.0, confidence=0.7, strategy_name=self.name)
        return TradeSignal(action=0.1, confidence=0.4, strategy_name=self.name)
