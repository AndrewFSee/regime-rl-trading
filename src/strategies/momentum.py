"""Momentum trading strategy."""
import numpy as np

from .base import TradeSignal, TradingStrategy


class MomentumStrategy(TradingStrategy):
    """
    Trend-following strategy based on short/long moving-average crossover
    and price momentum.

    Expected feature layout (indices are configurable):
        short_ma_idx  – normalised short (10d) MA deviation from price
        long_ma_idx   – normalised long  (30d) MA deviation from price
        momentum_idx  – 10-day price return

    Decision rules:
        short_ma > long_ma AND momentum > 0  → strong buy  (+1.0, conf 0.9)
        short_ma > long_ma                   → moderate buy (+0.5, conf 0.6)
        short_ma < long_ma AND momentum < 0  → strong sell (-1.0, conf 0.9)
        short_ma < long_ma                   → moderate sell(-0.5, conf 0.6)
        otherwise                            → neutral      ( 0.0, conf 0.5)
    """

    def __init__(
        self,
        short_ma_idx: int = 2,
        long_ma_idx: int = 3,
        momentum_idx: int = 6,
    ) -> None:
        self.short_ma_idx = short_ma_idx
        self.long_ma_idx = long_ma_idx
        self.momentum_idx = momentum_idx

    @property
    def name(self) -> str:
        return "momentum"

    def generate_signal(self, state: np.ndarray) -> TradeSignal:
        state = np.asarray(state, dtype=float).ravel()
        # Guard against insufficient feature vector length
        n = len(state)
        short_ma  = state[self.short_ma_idx]  if self.short_ma_idx  < n else 0.0
        long_ma   = state[self.long_ma_idx]   if self.long_ma_idx   < n else 0.0
        momentum  = state[self.momentum_idx]  if self.momentum_idx  < n else 0.0

        if short_ma > long_ma and momentum > 0:
            return TradeSignal(action=1.0, confidence=0.9, strategy_name=self.name)
        if short_ma > long_ma:
            return TradeSignal(action=0.5, confidence=0.6, strategy_name=self.name)
        if short_ma < long_ma and momentum < 0:
            return TradeSignal(action=-1.0, confidence=0.9, strategy_name=self.name)
        if short_ma < long_ma:
            return TradeSignal(action=-0.5, confidence=0.6, strategy_name=self.name)
        return TradeSignal(action=0.0, confidence=0.5, strategy_name=self.name)
