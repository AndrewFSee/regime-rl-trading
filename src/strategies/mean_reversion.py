"""Mean-reversion trading strategy."""
import numpy as np

from .base import TradeSignal, TradingStrategy


class MeanReversionStrategy(TradingStrategy):
    """
    Counter-trend strategy based on RSI and Bollinger Band deviation.

    Expected feature layout (indices are configurable):
        rsi_idx      – RSI normalised to [-1, 1]  (rsi/50 - 1)
        bb_upper_idx – upper BB / price - 1
        bb_lower_idx – lower BB / price - 1
        price_idx    – not directly used (placeholder, kept for compatibility)

    Decision rules (each condition contributes ±1 vote):
        rsi < -0.4  (≈ RSI<30) → buy  vote
        rsi > +0.4  (≈ RSI>70) → sell vote
        bb_lower < 0            → price above lower band (buy if deeply below)
        The raw signal is the mean of all individual votes, clipped to [-1, 1].
    """

    def __init__(
        self,
        rsi_idx: int = 7,
        bb_upper_idx: int = 4,
        bb_lower_idx: int = 5,
        price_idx: int = 0,
    ) -> None:
        self.rsi_idx = rsi_idx
        self.bb_upper_idx = bb_upper_idx
        self.bb_lower_idx = bb_lower_idx
        self.price_idx = price_idx

    @property
    def name(self) -> str:
        return "mean_reversion"

    def generate_signal(self, state: np.ndarray) -> TradeSignal:
        state = np.asarray(state, dtype=float).ravel()
        n = len(state)

        rsi      = state[self.rsi_idx]      if self.rsi_idx      < n else 0.0
        bb_upper = state[self.bb_upper_idx] if self.bb_upper_idx < n else 0.0
        bb_lower = state[self.bb_lower_idx] if self.bb_lower_idx < n else 0.0

        votes: list[float] = []

        # RSI signals  (normalised: -1 = RSI 0, 0 = RSI 50, +1 = RSI 100)
        if rsi < -0.4:    # RSI < 30 → oversold → buy
            votes.append(1.0)
        elif rsi > 0.4:   # RSI > 70 → overbought → sell
            votes.append(-1.0)

        # Bollinger Band signals  (bb_X = band/price - 1, so negative means price above band)
        if bb_lower > 0:   # lower band > price → price below lower band → buy
            votes.append(1.0)
        elif bb_upper < 0: # upper band < price → price above upper band → sell
            votes.append(-1.0)

        if not votes:
            return TradeSignal(action=0.0, confidence=0.3, strategy_name=self.name)

        action = float(np.clip(np.mean(votes), -1.0, 1.0))
        confidence = min(0.5 + 0.2 * len(votes), 0.95)
        return TradeSignal(action=action, confidence=confidence, strategy_name=self.name)
