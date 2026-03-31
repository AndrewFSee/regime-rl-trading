"""HMM-based market regime detector using hmmlearn."""
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base import MarketRegime, RegimeDetector

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


class HMMRegimeDetector(RegimeDetector):
    """
    Gaussian HMM regime detector.

    Extracts log-return, rolling volatility, and rolling mean-return features
    then maps each hidden state to a ``MarketRegime`` label by analysing the
    per-state mean return and volatility.
    """

    def __init__(
        self,
        n_components: int = 4,
        n_iter: int = 100,
        covariance_type: str = "full",
        random_state: Optional[int] = 42,
    ) -> None:
        if not _HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for HMMRegimeDetector. "
                "Install it with: pip install hmmlearn"
            )
        self.n_components = n_components
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self._state_to_regime: dict[int, MarketRegime] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(data: pd.DataFrame) -> np.ndarray:
        """Return (n_samples, 3) array: [log_return, rolling_vol, rolling_mean]."""
        close = data["Close"] if "Close" in data.columns else data.iloc[:, 3]
        log_ret = np.log(close / close.shift(1))
        rolling_vol = log_ret.rolling(20, min_periods=1).std()
        rolling_mean = log_ret.rolling(20, min_periods=1).mean()

        features = pd.DataFrame({
            "log_return": log_ret,
            "rolling_vol": rolling_vol,
            "rolling_mean": rolling_mean,
        })
        # Forward-fill then fill remaining NaN with 0
        features = features.ffill().fillna(0.0)
        return features.values

    def _map_states_to_regimes(self, features: np.ndarray, states: np.ndarray) -> None:
        """Assign a MarketRegime to each hidden state index."""
        self._state_to_regime = {}
        for state_idx in range(self.n_components):
            mask = states == state_idx
            if mask.sum() == 0:
                self._state_to_regime[state_idx] = MarketRegime.SIDEWAYS
                continue

            mean_ret = features[mask, 2].mean()   # rolling_mean column
            mean_vol = features[mask, 1].mean()   # rolling_vol column

            # Compute global thresholds from all data
            global_vol_75 = np.percentile(features[:, 1], 75)

            if mean_vol >= global_vol_75:
                self._state_to_regime[state_idx] = MarketRegime.VOLATILE
            elif mean_ret > 0.0005:
                self._state_to_regime[state_idx] = MarketRegime.BULL
            elif mean_ret < -0.0005:
                self._state_to_regime[state_idx] = MarketRegime.BEAR
            else:
                self._state_to_regime[state_idx] = MarketRegime.SIDEWAYS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "HMMRegimeDetector":
        """Fit the HMM on historical OHLCV data."""
        features = self._extract_features(data)
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(features)

        states = self.model.predict(features)
        self._map_states_to_regimes(features, states)
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return an array of :class:`MarketRegime` values."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before predict().")
        features = self._extract_features(data)
        states = self.model.predict(features)
        return np.array([self._state_to_regime[s] for s in states])
