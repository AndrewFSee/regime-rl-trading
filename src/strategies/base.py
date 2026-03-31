"""Abstract base class for trading strategies."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TradeSignal:
    """Represents a trading signal."""
    action: float       # -1 (full short), 0 (neutral), 1 (full long), or fractional
    confidence: float   # 0-1 confidence in the signal
    strategy_name: str


class TradingStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def generate_signal(self, state: np.ndarray) -> TradeSignal:
        """Generate a trading signal given the current market state."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        ...
