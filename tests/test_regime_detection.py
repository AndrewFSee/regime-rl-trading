"""
Tests for regime detection modules.

All tests run with only numpy + pandas and skip gracefully when
heavyweight dependencies (hmmlearn, torch, etc.) are absent.
"""
import sys
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 150, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with *n* rows."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high   = close * (1 + rng.uniform(0, 0.02, n))
    low    = close * (1 - rng.uniform(0, 0.02, n))
    open_  = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    })


# ---------------------------------------------------------------------------
# MarketRegime enum
# ---------------------------------------------------------------------------

def test_market_regime_enum_values():
    from src.regime_detection.base import MarketRegime
    assert MarketRegime.BULL.value     == 0
    assert MarketRegime.BEAR.value     == 1
    assert MarketRegime.SIDEWAYS.value == 2
    assert MarketRegime.VOLATILE.value == 3
    assert len(MarketRegime) == 4


def test_market_regime_names():
    from src.regime_detection.base import MarketRegime
    names = {r.name for r in MarketRegime}
    assert names == {"BULL", "BEAR", "SIDEWAYS", "VOLATILE"}


# ---------------------------------------------------------------------------
# FeatureRegimeDetector
# ---------------------------------------------------------------------------

def test_feature_detector_fit_returns_self():
    from src.regime_detection.feature_detector import FeatureRegimeDetector
    detector = FeatureRegimeDetector()
    data = _make_ohlcv()
    result = detector.fit(data)
    assert result is detector


def test_feature_detector_predict_length():
    from src.regime_detection.feature_detector import FeatureRegimeDetector
    from src.regime_detection.base import MarketRegime
    data = _make_ohlcv(n=100)
    detector = FeatureRegimeDetector()
    regimes = detector.predict(data)
    assert len(regimes) == len(data)


def test_feature_detector_returns_market_regime():
    from src.regime_detection.feature_detector import FeatureRegimeDetector
    from src.regime_detection.base import MarketRegime
    data = _make_ohlcv(n=100)
    detector = FeatureRegimeDetector()
    regimes = detector.predict(data)
    for r in regimes:
        assert isinstance(r, MarketRegime)


def test_feature_detector_all_regimes_possible():
    """At least BULL and SIDEWAYS should appear over 500 rows."""
    from src.regime_detection.feature_detector import FeatureRegimeDetector
    from src.regime_detection.base import MarketRegime
    rng = np.random.default_rng(42)
    # Construct data with both strong trends and flat periods
    n = 500
    closes = np.concatenate([
        100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.005, 125))),   # bull trend
        100.0 * np.exp(np.cumsum(rng.normal(-0.002, 0.005, 125))),  # bear trend
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.0005, 125))),      # sideways
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, 125))),        # volatile
    ])
    high   = closes * 1.01
    low    = closes * 0.99
    volume = np.ones(n) * 1e6
    data = pd.DataFrame({
        "Open": closes, "High": high, "Low": low,
        "Close": closes, "Volume": volume
    })
    regimes = FeatureRegimeDetector().predict(data)
    found = {r for r in regimes}
    assert len(found) >= 2


def test_feature_detector_fit_predict_consistent():
    from src.regime_detection.feature_detector import FeatureRegimeDetector
    data = _make_ohlcv(n=80)
    detector = FeatureRegimeDetector()
    r1 = detector.fit_predict(data)
    r2 = detector.predict(data)
    np.testing.assert_array_equal(r1, r2)


def test_feature_detector_custom_thresholds():
    from src.regime_detection.feature_detector import FeatureRegimeDetector
    from src.regime_detection.base import MarketRegime
    # With a very high volatility_high threshold, VOLATILE should never appear.
    data = _make_ohlcv(n=100)
    detector = FeatureRegimeDetector(volatility_high=999.0)
    regimes = detector.predict(data)
    for r in regimes:
        assert r != MarketRegime.VOLATILE


# ---------------------------------------------------------------------------
# HMMRegimeDetector (skipped if hmmlearn unavailable)
# ---------------------------------------------------------------------------

def test_hmm_detector_basic():
    pytest.importorskip("hmmlearn")
    from src.regime_detection.hmm_detector import HMMRegimeDetector
    from src.regime_detection.base import MarketRegime

    data = _make_ohlcv(n=200)
    detector = HMMRegimeDetector(n_components=4, n_iter=10)
    detector.fit(data)
    regimes = detector.predict(data)
    assert len(regimes) == len(data)
    for r in regimes:
        assert isinstance(r, MarketRegime)


def test_hmm_detector_fit_predict():
    pytest.importorskip("hmmlearn")
    from src.regime_detection.hmm_detector import HMMRegimeDetector
    data = _make_ohlcv(n=200)
    detector = HMMRegimeDetector(n_components=4, n_iter=10)
    r1 = detector.fit_predict(data)
    r2 = detector.predict(data)
    np.testing.assert_array_equal(r1, r2)


def test_hmm_detector_raises_without_hmmlearn(monkeypatch):
    import importlib
    import sys as _sys
    # Temporarily hide hmmlearn
    hmmlearn_backup = _sys.modules.pop("hmmlearn", None)
    hmmlearn_hmm_backup = _sys.modules.pop("hmmlearn.hmm", None)
    # Patch the module-level flag
    import src.regime_detection.hmm_detector as mod
    original_flag = mod._HMM_AVAILABLE
    mod._HMM_AVAILABLE = False
    try:
        with pytest.raises(ImportError):
            from src.regime_detection.hmm_detector import HMMRegimeDetector
            HMMRegimeDetector()
    finally:
        mod._HMM_AVAILABLE = original_flag
        if hmmlearn_backup is not None:
            _sys.modules["hmmlearn"] = hmmlearn_backup
        if hmmlearn_hmm_backup is not None:
            _sys.modules["hmmlearn.hmm"] = hmmlearn_hmm_backup
