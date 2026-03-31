"""Volatility breakout trading strategy."""
import numpy as np

from .base import TradeSignal, TradingStrategy


class BreakoutStrategy(TradingStrategy):
    """
    Breakout strategy that triggers on high ATR relative to recent average,
    combined with proximity to recent high or low.

    Expected feature layout (indices are configurable):
        atr_idx   – ATR(14) / price  (normalised ATR)
        high_idx  – recent high / price - 1
        low_idx   – recent low  / price - 1
        close_idx – log return or returns feature (index 0 by default)

    Decision rules:
        1. Compute an ATR z-score relative to its typical value (use 0.01 as baseline).
        2. If ATR is elevated (> 1.5× baseline) AND price is near recent high  → long.
        3. If ATR is elevated (> 1.5× baseline) AND price is near recent low   → short.
        4. Otherwise → flat.
    """

    ATR_BASELINE = 0.01
    ATR_MULTIPLIER = 1.5
    PROXIMITY_THRESHOLD = 0.005  # within 0.5 % of recent high/low

    def __init__(
        self,
        atr_idx: int = 8,
        high_idx: int = 1,
        low_idx: int = 2,
        close_idx: int = 0,
    ) -> None:
        self.atr_idx = atr_idx
        self.high_idx = high_idx
        self.low_idx = low_idx
        self.close_idx = close_idx

    @property
    def name(self) -> str:
        return "breakout"

    def generate_signal(self, state: np.ndarray) -> TradeSignal:
        state = np.asarray(state, dtype=float).ravel()
        n = len(state)

        atr  = state[self.atr_idx]  if self.atr_idx  < n else 0.0
        high = state[self.high_idx] if self.high_idx  < n else 0.0
        low  = state[self.low_idx]  if self.low_idx   < n else 0.0

        elevated = atr > self.ATR_BASELINE * self.ATR_MULTIPLIER
        if not elevated:
            return TradeSignal(action=0.0, confidence=0.3, strategy_name=self.name)

        near_high = abs(high) < self.PROXIMITY_THRESHOLD
        near_low  = abs(low)  < self.PROXIMITY_THRESHOLD

        if near_high and not near_low:
            return TradeSignal(action=1.0, confidence=0.75, strategy_name=self.name)
        if near_low and not near_high:
            return TradeSignal(action=-1.0, confidence=0.75, strategy_name=self.name)

        # Elevated ATR but not clearly near high or low → cautious long bias
        return TradeSignal(action=0.2, confidence=0.4, strategy_name=self.name)
