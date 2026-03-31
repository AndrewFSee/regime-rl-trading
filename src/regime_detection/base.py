"""Abstract base class for market regime detectors."""
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime types."""
    BULL = 0
    BEAR = 1
    SIDEWAYS = 2
    VOLATILE = 3


class RegimeDetector(ABC):
    """Abstract base class for all regime detectors."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "RegimeDetector":
        """Fit the detector on historical data."""
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime labels for the given data."""
        ...

    def fit_predict(self, data: pd.DataFrame) -> np.ndarray:
        """Fit and predict in one step."""
        return self.fit(data).predict(data)
