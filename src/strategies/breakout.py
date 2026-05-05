"""Volatility breakout trading strategy."""
import numpy as np

from .base import TradeSignal, TradingStrategy


class BreakoutStrategy(TradingStrategy):
    """
    Breakout strategy that triggers on high ATR relative to recent average,
    combined with proximity to recent high or low.

    Expected feature layout (indices match ``FEATURE_NAMES``):
        atr_idx       – ATR(14) / price  (normalised ATR)               [idx 8]
        bb_upper_idx  – upper Bollinger Band / price - 1                 [idx 4]
                        (≈ 0 when price touches the upper band)
        bb_lower_idx  – lower Bollinger Band / price - 1                 [idx 5]
                        (≈ 0 when price touches the lower band)
        close_idx     – returns feature (kept for compatibility)         [idx 0]

    Decision rules:
        1. ATR elevated (> 1.5× baseline)  AND  price near upper band → long.
        2. ATR elevated (> 1.5× baseline)  AND  price near lower band → short.
        3. Otherwise → flat (or weak long when ATR is elevated alone).

    Note
    ----
    The previous defaults read ``state[1]`` (volatility) and ``state[2]``
    (short_ma) as proxies for "distance from recent high/low". That mismatch
    made ``near_high`` / ``near_low`` almost never trigger; the strategy fired
    on only ~25% of days and never produced a strong (+1) signal. The fixed
    defaults below use bb_upper / bb_lower, which by construction are zero
    when price is at the corresponding band.
    """

    ATR_BASELINE = 0.01
    ATR_MULTIPLIER = 1.5
    PROXIMITY_THRESHOLD = 0.005  # within 0.5 % of the band

    def __init__(
        self,
        atr_idx: int = 8,
        bb_upper_idx: int = 4,
        bb_lower_idx: int = 5,
        close_idx: int = 0,
        # Back-compat aliases (deprecated names).
        high_idx: int | None = None,
        low_idx: int | None = None,
    ) -> None:
        self.atr_idx = atr_idx
        # Accept legacy ``high_idx`` / ``low_idx`` kwargs but map them onto the
        # new band-relative semantics so old call sites keep working.
        self.bb_upper_idx = high_idx if high_idx is not None else bb_upper_idx
        self.bb_lower_idx = low_idx if low_idx is not None else bb_lower_idx
        self.close_idx = close_idx

    @property
    def name(self) -> str:
        return "breakout"

    def generate_signal(self, state: np.ndarray) -> TradeSignal:
        state = np.asarray(state, dtype=float).ravel()
        n = len(state)

        atr      = state[self.atr_idx]      if self.atr_idx      < n else 0.0
        bb_upper = state[self.bb_upper_idx] if self.bb_upper_idx < n else 0.0
        bb_lower = state[self.bb_lower_idx] if self.bb_lower_idx < n else 0.0

        elevated = atr > self.ATR_BASELINE * self.ATR_MULTIPLIER
        if not elevated:
            return TradeSignal(action=0.0, confidence=0.3, strategy_name=self.name)

        # bb_X = band/price - 1; |bb_upper| ≈ 0 means price touches upper band.
        near_high = abs(bb_upper) < self.PROXIMITY_THRESHOLD
        near_low  = abs(bb_lower) < self.PROXIMITY_THRESHOLD

        if near_high and not near_low:
            return TradeSignal(action=1.0, confidence=0.75, strategy_name=self.name)
        if near_low and not near_high:
            return TradeSignal(action=-1.0, confidence=0.75, strategy_name=self.name)

        # Elevated ATR but not clearly near a band → cautious long bias
        return TradeSignal(action=0.2, confidence=0.4, strategy_name=self.name)
