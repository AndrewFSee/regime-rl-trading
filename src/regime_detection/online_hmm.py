"""Online (streaming) HMM regime detector.

Wraps an already-fit :class:`HMMRegimeDetector` and exposes an incremental
``update(bar)`` method that runs a single forward-algorithm step per new
OHLCV bar, maintaining the posterior over hidden states ``alpha_t``.

This is the standard filtering recursion::

    alpha_t(s) ∝ p(o_t | s) * sum_{s'} A[s', s] * alpha_{t-1}(s')

Emission and transition parameters are taken from the fitted offline model
and held FIXED — this gives O(K^2) per-bar cost without the numerical
hazards of online Baum-Welch on a single trajectory. If the user wants
parameter adaptation they can periodically call ``refit(window)`` on the
underlying detector.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from .base import MarketRegime, RegimeDetector
from .hmm_detector import HMMRegimeDetector, _HMM_AVAILABLE


class OnlineHMMRegimeDetector(RegimeDetector):
    """Streaming wrapper around :class:`HMMRegimeDetector`.

    Parameters
    ----------
    base:
        A fitted (or fittable) :class:`HMMRegimeDetector`. If not yet fit
        when :meth:`fit` is called here, it will be fit on the supplied
        history.
    window:
        Length of the rolling OHLCV buffer used to recompute per-bar
        features (log-return + 20-bar rolling vol/mean). Must be at least
        20 to match the offline feature extractor.
    """

    def __init__(self, base: Optional[HMMRegimeDetector] = None, window: int = 64) -> None:
        if not _HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for OnlineHMMRegimeDetector."
            )
        if window < 20:
            raise ValueError("window must be >= 20 to match feature extractor.")
        self.base = base if base is not None else HMMRegimeDetector()
        self.window = int(window)

        self._buffer: deque[dict] = deque(maxlen=self.window)
        self._alpha: Optional[np.ndarray] = None  # posterior over states

    # ------------------------------------------------------------------
    # RegimeDetector interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "OnlineHMMRegimeDetector":
        """Fit the underlying HMM and prime the streaming buffer/posterior."""
        if not self.base._is_fitted:
            self.base.fit(data)

        # Seed buffer with the most recent ``window`` bars so that subsequent
        # ``update`` calls have valid rolling features from bar 1.
        tail = data.tail(self.window)
        self._buffer.clear()
        for _, row in tail.iterrows():
            self._buffer.append(row.to_dict())

        # Initial posterior = stationary distribution of the transition matrix.
        self._alpha = self._stationary_distribution(self.base.model.transmat_)
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Filter ``data`` bar-by-bar and return the MAP regime per row.

        This RESETS the streaming state so the result is deterministic for
        the supplied frame. For true streaming use :meth:`update` directly.
        """
        if not self.base._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        # Reset filter
        self._buffer.clear()
        self._alpha = self._stationary_distribution(self.base.model.transmat_)

        out: list[MarketRegime] = []
        for _, row in data.iterrows():
            out.append(self.update(row.to_dict()))
        return np.array(out)

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def update(self, bar: dict) -> MarketRegime:
        """Ingest one new OHLCV bar and return the current MAP regime."""
        if not self.base._is_fitted or self._alpha is None:
            raise RuntimeError("Call fit() before update().")

        self._buffer.append(bar)

        feat = self._extract_latest_feature()
        if feat is None:
            # Not enough history yet — fall back to MAP of prior alpha.
            return self._map_regime(self._alpha)

        # Forward step: alpha_t(s) ∝ B(o_t | s) * A^T @ alpha_{t-1}
        A = self.base.model.transmat_                 # (K, K)
        log_b = self.base.model._compute_log_likelihood(feat[None, :])[0]  # (K,)
        # Numerical stability: subtract max before exp
        log_b = log_b - log_b.max()
        b = np.exp(log_b)

        new_alpha = b * (A.T @ self._alpha)
        s = new_alpha.sum()
        if s <= 0 or not np.isfinite(s):
            # Likelihood underflow — re-seed with stationary prior
            new_alpha = self._stationary_distribution(A)
        else:
            new_alpha = new_alpha / s
        self._alpha = new_alpha

        return self._map_regime(self._alpha)

    @property
    def posterior(self) -> Optional[np.ndarray]:
        """Current posterior over hidden states (or None before fit)."""
        return None if self._alpha is None else self._alpha.copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_latest_feature(self) -> Optional[np.ndarray]:
        """Compute (log_return, rolling_vol, rolling_mean) for the newest bar."""
        if len(self._buffer) < 2:
            return None
        df = pd.DataFrame(list(self._buffer))
        if "Close" not in df.columns:
            return None
        close = df["Close"].astype(float)
        log_ret = np.log(close / close.shift(1))
        rolling_vol = log_ret.rolling(20, min_periods=1).std()
        rolling_mean = log_ret.rolling(20, min_periods=1).mean()
        feat = np.array([
            log_ret.iloc[-1],
            rolling_vol.iloc[-1],
            rolling_mean.iloc[-1],
        ], dtype=float)
        # ffill/0-fill semantics matching offline detector
        feat = np.nan_to_num(feat, nan=0.0)
        return feat

    def _map_regime(self, alpha: np.ndarray) -> MarketRegime:
        state = int(np.argmax(alpha))
        return self.base._state_to_regime.get(state, MarketRegime.SIDEWAYS)

    @staticmethod
    def _stationary_distribution(A: np.ndarray) -> np.ndarray:
        """Solve π A = π via dominant eigenvector of A^T."""
        K = A.shape[0]
        try:
            eigvals, eigvecs = np.linalg.eig(A.T)
            # Pick the eigenvector closest to eigenvalue 1
            idx = int(np.argmin(np.abs(eigvals - 1.0)))
            pi = np.real(eigvecs[:, idx])
            pi = np.abs(pi)
            s = pi.sum()
            if s > 0 and np.isfinite(s):
                return pi / s
        except np.linalg.LinAlgError:
            pass
        return np.full(K, 1.0 / K)
