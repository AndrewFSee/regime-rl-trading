"""Defensive (low-volatility / capital-preservation) trading strategy."""
import numpy as np

from .base import TradeSignal, TradingStrategy


class DefensiveStrategy(TradingStrategy):
    """
    Reduces market exposure when volatility is elevated and maintains a small
    positive bias during calm periods.

    Expected feature layout:
        volatility_idx – 20-day rolling std of log-returns

    Thresholds:
        vol > HIGH_VOL  → near-zero exposure (0.0, signal neutral)
        vol > MED_VOL   → reduced exposure   (0.1)
        otherwise       → small long bias     (0.2)
    """

    HIGH_VOL = 0.025
    MED_VOL = 0.015

    def __init__(self, volatility_idx: int = 1) -> None:
        self.volatility_idx = volatility_idx

    @property
    def name(self) -> str:
        return "defensive"

    def generate_signal(self, state: np.ndarray) -> TradeSignal:
        state = np.asarray(state, dtype=float).ravel()
        n = len(state)
        vol = state[self.volatility_idx] if self.volatility_idx < n else 0.0
        # Use absolute value in case the feature is signed/normalised
        vol = abs(vol)

        if vol > self.HIGH_VOL:
            return TradeSignal(action=0.0, confidence=0.8, strategy_name=self.name)
        if vol > self.MED_VOL:
            return TradeSignal(action=0.1, confidence=0.7, strategy_name=self.name)
        return TradeSignal(action=0.2, confidence=0.6, strategy_name=self.name)
