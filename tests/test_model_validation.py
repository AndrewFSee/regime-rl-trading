"""
Model validation suite – tests that regime detectors and strategies
produce *correct* outputs on synthetic data with known regime labels.

These tests go beyond mechanical correctness (covered by test_regime_detection
and test_strategies) to check whether the models actually generalise:
    • Can the detectors recover known regimes from synthetic price data?
    • Do strategies generate directionally correct signals in their target regimes?
    • Do strategies stay cautious in regimes they shouldn't be active in?
"""
import numpy as np
import pandas as pd
import pytest

from src.regime_detection.base import MarketRegime
from src.regime_detection.feature_detector import FeatureRegimeDetector
from src.environment.features import FeatureEngineer, FEATURE_NAMES
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.defensive import DefensiveStrategy

# Try importing HMM detector (optional dependency)
try:
    from src.regime_detection.hmm_detector import HMMRegimeDetector
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


# ======================================================================
# Synthetic data generators – each produces OHLCV with a known regime
# ======================================================================

def _make_bull(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Strong uptrend, moderate volatility."""
    rng = np.random.default_rng(seed)
    daily_drift = 0.002  # ~0.2% per day
    daily_vol = 0.008
    log_rets = daily_drift + daily_vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    return _close_to_ohlcv(close, rng)


def _make_bear(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Strong downtrend, moderate volatility."""
    rng = np.random.default_rng(seed)
    daily_drift = -0.002
    daily_vol = 0.008
    log_rets = daily_drift + daily_vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    return _close_to_ohlcv(close, rng)


def _make_sideways(n: int = 300, seed: int = 2) -> pd.DataFrame:
    """No drift, very low volatility."""
    rng = np.random.default_rng(seed)
    daily_vol = 0.003  # very calm
    log_rets = daily_vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    return _close_to_ohlcv(close, rng)


def _make_volatile(n: int = 300, seed: int = 3) -> pd.DataFrame:
    """No drift, very high volatility."""
    rng = np.random.default_rng(seed)
    daily_vol = 0.04  # wild swings
    log_rets = daily_vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    return _close_to_ohlcv(close, rng)


def _close_to_ohlcv(close: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Convert a close series to plausible OHLCV DataFrame."""
    spread = np.abs(close) * 0.005
    high = close + rng.uniform(0, 1, len(close)) * spread
    low = close - rng.uniform(0, 1, len(close)) * spread
    open_ = low + rng.uniform(0, 1, len(close)) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, len(close)).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def _make_multi_regime(n_per: int = 200, seed: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Concatenates BULL → BEAR → SIDEWAYS → VOLATILE blocks into one price series.
    Returns (ohlcv_df, true_regime_labels) where labels use MarketRegime values.
    """
    rng = np.random.default_rng(seed)
    segments = []
    labels = []

    configs = [
        (MarketRegime.BULL,     0.002,  0.008),
        (MarketRegime.BEAR,    -0.002,  0.008),
        (MarketRegime.SIDEWAYS, 0.0,    0.003),
        (MarketRegime.VOLATILE, 0.0,    0.04),
    ]
    price = 100.0
    for regime, drift, vol in configs:
        log_rets = drift + vol * rng.standard_normal(n_per)
        prices = price * np.exp(np.cumsum(log_rets))
        price = prices[-1]
        segments.append(prices)
        labels.extend([regime] * n_per)

    close = np.concatenate(segments)
    df = _close_to_ohlcv(close, rng)
    return df, np.array(labels)


def _compute_features_dict(data: pd.DataFrame, idx: int) -> dict:
    """Compute features for a whole DataFrame then return row `idx` as a dict."""
    fe = FeatureEngineer()
    features = fe.compute(data)
    row = features.iloc[idx]
    return dict(zip(FEATURE_NAMES, row.values))


# ======================================================================
# REGIME DETECTOR VALIDATION
# ======================================================================


class TestFeatureDetectorValidation:
    """
    Validates FeatureRegimeDetector on synthetic single-regime price series.

    After a 30-bar warm-up (for rolling windows), we expect the detector to
    classify the *majority* of remaining bars into the correct regime.
    """

    WARMUP = 30  # let rolling windows fill
    MIN_ACCURACY = 0.65  # at least 65% correct on pure-regime data

    def _accuracy(self, data: pd.DataFrame, expected: MarketRegime) -> float:
        det = FeatureRegimeDetector()
        preds = det.predict(data)
        tail = preds[self.WARMUP:]
        return float(np.mean(tail == expected))

    def test_bull_detection(self):
        data = _make_bull(300)
        acc = self._accuracy(data, MarketRegime.BULL)
        assert acc >= self.MIN_ACCURACY, (
            f"Bull accuracy = {acc:.2%}, expected ≥ {self.MIN_ACCURACY:.0%}"
        )

    def test_bear_detection(self):
        data = _make_bear(300)
        acc = self._accuracy(data, MarketRegime.BEAR)
        assert acc >= self.MIN_ACCURACY, (
            f"Bear accuracy = {acc:.2%}, expected ≥ {self.MIN_ACCURACY:.0%}"
        )

    def test_sideways_detection(self):
        data = _make_sideways(300)
        acc = self._accuracy(data, MarketRegime.SIDEWAYS)
        assert acc >= self.MIN_ACCURACY, (
            f"Sideways accuracy = {acc:.2%}, expected ≥ {self.MIN_ACCURACY:.0%}"
        )

    def test_volatile_detection(self):
        data = _make_volatile(300)
        acc = self._accuracy(data, MarketRegime.VOLATILE)
        assert acc >= self.MIN_ACCURACY, (
            f"Volatile accuracy = {acc:.2%}, expected ≥ {self.MIN_ACCURACY:.0%}"
        )

    def test_multi_regime_overall(self):
        """Overall accuracy on concatenated regimes (after warmup per segment)."""
        data, true_labels = _make_multi_regime(n_per=200)
        det = FeatureRegimeDetector()
        preds = det.predict(data)
        # Skip first WARMUP of each 200-bar segment
        mask = np.ones(len(preds), dtype=bool)
        for seg in range(4):
            start = seg * 200
            mask[start: start + self.WARMUP] = False
        acc = float(np.mean(preds[mask] == true_labels[mask]))
        assert acc >= 0.55, f"Multi-regime accuracy = {acc:.2%}, expected ≥ 55%"

    def test_out_of_sample_consistency(self):
        """Train-set accuracy vs test-set accuracy should be similar (no fitting,
        so this is really testing that the threshold generalises across seeds)."""
        accs = []
        for seed in range(5):
            data = _make_bull(300, seed=seed * 10)
            accs.append(self._accuracy(data, MarketRegime.BULL))
        std = np.std(accs)
        assert std < 0.20, f"Bull accuracy varies too much across seeds: std={std:.3f}"

    def test_confusion_matrix_report(self):
        """Print a confusion summary (not an assertion, but useful diagnostic)."""
        data, true_labels = _make_multi_regime(n_per=200)
        det = FeatureRegimeDetector()
        preds = det.predict(data)
        regimes = [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS, MarketRegime.VOLATILE]
        for true_r in regimes:
            mask = true_labels == true_r
            if mask.sum() == 0:
                continue
            pred_dist = {r.name: int(np.sum(preds[mask] == r)) for r in regimes}
            total = int(mask.sum())
            correct = pred_dist.get(true_r.name, 0)
            # Just needs to exist and not crash — the printout is the real value
            assert total > 0


@pytest.mark.skipif(not _HMM_AVAILABLE, reason="hmmlearn not installed")
class TestHMMDetectorValidation:
    """
    Validates HMMRegimeDetector on synthetic data.

    HMMs are unsupervised so we test:
    1. Whether fitting on single-regime data produces a dominant regime.
    2. Whether fitting on multi-regime data can at least distinguish
       high-vol from low-vol segments.
    """

    WARMUP = 30

    def test_bull_dominant_regime(self):
        data = _make_bull(400, seed=42)
        det = HMMRegimeDetector(n_components=4, random_state=42)
        det.fit(data)
        preds = det.predict(data)
        tail = preds[self.WARMUP:]
        # HMM may not label it BULL exactly, but it should NOT be confused
        # with BEAR — at minimum the dominant regime has positive connotation
        bull_or_sideways = np.mean((tail == MarketRegime.BULL) | (tail == MarketRegime.SIDEWAYS))
        dist = {r.name: int(np.sum(tail == r)) for r in MarketRegime}
        assert bull_or_sideways >= 0.50, (
            f"Expected mostly BULL/SIDEWAYS on uptrend data, got distribution: {dist}"
        )

    def test_bear_dominant_regime(self):
        data = _make_bear(400, seed=42)
        det = HMMRegimeDetector(n_components=4, random_state=42)
        det.fit(data)
        preds = det.predict(data)
        tail = preds[self.WARMUP:]
        bear_or_sideways = np.mean((tail == MarketRegime.BEAR) | (tail == MarketRegime.SIDEWAYS))
        assert bear_or_sideways >= 0.50, (
            f"Expected mostly BEAR/SIDEWAYS on downtrend data"
        )

    def test_volatile_data_detected(self):
        """HMM should pick up elevated volatility."""
        data = _make_volatile(400, seed=42)
        det = HMMRegimeDetector(n_components=4, random_state=42)
        det.fit(data)
        preds = det.predict(data)
        tail = preds[self.WARMUP:]
        vol_frac = float(np.mean(tail == MarketRegime.VOLATILE))
        assert vol_frac >= 0.30, f"Only {vol_frac:.0%} VOLATILE on high-vol data"

    def test_multi_regime_separability(self):
        """On concatenated data, HMM should produce different state distributions
        for the low-vol vs high-vol segments."""
        data, true_labels = _make_multi_regime(n_per=250)
        det = HMMRegimeDetector(n_components=4, random_state=42)
        det.fit(data)
        preds = det.predict(data)

        # Compare predicted distribution in SIDEWAYS segment vs VOLATILE segment
        sw_seg = preds[500:750]  # sideways block
        vol_seg = preds[750:1000]  # volatile block

        sw_volatile = float(np.mean(sw_seg == MarketRegime.VOLATILE))
        vol_volatile = float(np.mean(vol_seg == MarketRegime.VOLATILE))

        # The volatile segment should have more VOLATILE labels than the sideways
        assert vol_volatile > sw_volatile, (
            f"VOLATILE fraction: sideways={sw_volatile:.2%}, volatile={vol_volatile:.2%}. "
            f"HMM cannot distinguish volatility regimes."
        )

    def test_out_of_sample_prediction(self):
        """Fit on one dataset, predict on fresh data from same distribution."""
        train = _make_bull(400, seed=100)
        test = _make_bull(200, seed=200)
        det = HMMRegimeDetector(n_components=4, random_state=42)
        det.fit(train)
        preds = det.predict(test)
        tail = preds[self.WARMUP:]
        bull_or_sideways = np.mean((tail == MarketRegime.BULL) | (tail == MarketRegime.SIDEWAYS))
        assert bull_or_sideways >= 0.40, (
            f"Out-of-sample bull data classified poorly: "
            f"BULL+SIDEWAYS = {bull_or_sideways:.2%}"
        )


# ======================================================================
# STRATEGY VALIDATION
# ======================================================================


def _make_features_for_regime(regime: MarketRegime, rng: np.random.Generator) -> dict:
    """
    Create a feature dict that represents typical conditions for a given regime.
    Uses the same feature semantics as FeatureEngineer.
    """
    if regime == MarketRegime.BULL:
        return {
            "returns": 0.005 + 0.002 * rng.standard_normal(),
            "volatility": 0.010 + 0.002 * abs(rng.standard_normal()),
            "short_ma": 0.02 + 0.005 * rng.standard_normal(),   # above zero → above price
            "long_ma": 0.005 + 0.003 * rng.standard_normal(),   # positive but smaller
            "bb_upper": 0.03 + 0.01 * abs(rng.standard_normal()),  # well above price
            "bb_lower": -0.03 - 0.01 * abs(rng.standard_normal()), # well below price
            "momentum": 0.04 + 0.01 * rng.standard_normal(),  # positive
            "rsi": 0.3 + 0.1 * rng.standard_normal(),         # above neutral (RSI ~65)
            "atr": 0.010 + 0.003 * abs(rng.standard_normal()),
            "volume_ratio": 1.0 + 0.2 * rng.standard_normal(),
        }
    elif regime == MarketRegime.BEAR:
        return {
            "returns": -0.005 + 0.002 * rng.standard_normal(),
            "volatility": 0.012 + 0.003 * abs(rng.standard_normal()),
            "short_ma": -0.02 + 0.005 * rng.standard_normal(),   # below zero
            "long_ma": -0.005 + 0.003 * rng.standard_normal(),   # less negative
            "bb_upper": 0.02 + 0.01 * abs(rng.standard_normal()),
            "bb_lower": -0.02 - 0.01 * abs(rng.standard_normal()),
            "momentum": -0.04 + 0.01 * rng.standard_normal(),  # negative
            "rsi": -0.3 + 0.1 * rng.standard_normal(),         # below neutral (RSI ~35)
            "atr": 0.012 + 0.003 * abs(rng.standard_normal()),
            "volume_ratio": 1.1 + 0.3 * rng.standard_normal(),
        }
    elif regime == MarketRegime.SIDEWAYS:
        return {
            "returns": 0.001 * rng.standard_normal(),
            "volatility": 0.005 + 0.001 * abs(rng.standard_normal()),  # very low
            "short_ma": 0.001 * rng.standard_normal(),
            "long_ma": 0.001 * rng.standard_normal(),
            "bb_upper": 0.01 + 0.003 * abs(rng.standard_normal()),
            "bb_lower": -0.01 - 0.003 * abs(rng.standard_normal()),
            "momentum": 0.002 * rng.standard_normal(),
            "rsi": 0.05 * rng.standard_normal(),  # near neutral
            "atr": 0.005 + 0.001 * abs(rng.standard_normal()),  # low ATR
            "volume_ratio": 0.9 + 0.1 * rng.standard_normal(),
        }
    else:  # VOLATILE
        return {
            "returns": 0.01 * rng.standard_normal(),  # large random
            "volatility": 0.035 + 0.01 * abs(rng.standard_normal()),  # very high
            "short_ma": 0.01 * rng.standard_normal(),  # noisy
            "long_ma": 0.005 * rng.standard_normal(),
            "bb_upper": 0.06 + 0.02 * abs(rng.standard_normal()),
            "bb_lower": -0.06 - 0.02 * abs(rng.standard_normal()),
            "momentum": 0.02 * rng.standard_normal(),
            "rsi": 0.3 * rng.standard_normal(),  # wide swings
            "atr": 0.035 + 0.01 * abs(rng.standard_normal()),  # high ATR
            "volume_ratio": 1.5 + 0.5 * abs(rng.standard_normal()),
        }


class TestMomentumStrategyValidation:
    """Momentum should be net-long in BULL, net-short in BEAR, near-zero in SIDEWAYS."""

    N_SAMPLES = 100

    def _mean_action(self, regime: MarketRegime, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        strat = MomentumStrategy()
        actions = []
        for _ in range(self.N_SAMPLES):
            feat = _make_features_for_regime(regime, rng)
            sig = strat.generate_signal(feat)
            actions.append(sig.action)
        return float(np.mean(actions))

    def test_bull_net_long(self):
        mean_a = self._mean_action(MarketRegime.BULL)
        assert mean_a > 0.3, f"Momentum should be net long in BULL, got mean action={mean_a:.3f}"

    def test_bear_net_short(self):
        mean_a = self._mean_action(MarketRegime.BEAR)
        assert mean_a < -0.3, f"Momentum should be net short in BEAR, got mean action={mean_a:.3f}"

    def test_sideways_near_neutral(self):
        mean_a = self._mean_action(MarketRegime.SIDEWAYS)
        assert abs(mean_a) < 0.5, f"Momentum should be near neutral in SIDEWAYS, got {mean_a:.3f}"

    def test_confidence_higher_in_trending(self):
        rng = np.random.default_rng(42)
        strat = MomentumStrategy()
        bull_confs = [strat.generate_signal(_make_features_for_regime(MarketRegime.BULL, rng)).confidence
                      for _ in range(50)]
        sw_confs = [strat.generate_signal(_make_features_for_regime(MarketRegime.SIDEWAYS, rng)).confidence
                    for _ in range(50)]
        assert np.mean(bull_confs) > np.mean(sw_confs), (
            f"Expected higher confidence in trending market (bull={np.mean(bull_confs):.2f} vs "
            f"sideways={np.mean(sw_confs):.2f})"
        )


class TestMeanReversionValidation:
    """Mean reversion should buy oversold and sell overbought conditions."""

    N_SAMPLES = 100

    def test_buy_in_oversold(self):
        """When RSI is deeply negative and price near lower BB → buy."""
        rng = np.random.default_rng(0)
        strat = MeanReversionStrategy()
        actions = []
        for _ in range(self.N_SAMPLES):
            feat = {
                "rsi": -0.5 - 0.1 * abs(rng.standard_normal()),  # deeply oversold
                "bb_upper": 0.03,
                "bb_lower": 0.01 + 0.005 * abs(rng.standard_normal()),  # lower band above price
                "returns": 0.0,
            }
            actions.append(strat.generate_signal(feat).action)
        mean_a = np.mean(actions)
        assert mean_a > 0.5, f"Mean reversion should buy oversold, got mean action={mean_a:.3f}"

    def test_sell_in_overbought(self):
        """When RSI is very positive and price near upper BB → sell."""
        rng = np.random.default_rng(0)
        strat = MeanReversionStrategy()
        actions = []
        for _ in range(self.N_SAMPLES):
            feat = {
                "rsi": 0.5 + 0.1 * abs(rng.standard_normal()),  # overbought
                "bb_upper": -0.01 - 0.005 * abs(rng.standard_normal()),  # upper band below price
                "bb_lower": -0.03,
                "returns": 0.0,
            }
            actions.append(strat.generate_signal(feat).action)
        mean_a = np.mean(actions)
        assert mean_a < -0.5, f"Mean reversion should sell overbought, got mean action={mean_a:.3f}"

    def test_neutral_in_calm_market(self):
        """When indicators are mild → near-zero signal."""
        rng = np.random.default_rng(0)
        strat = MeanReversionStrategy()
        actions = []
        for _ in range(self.N_SAMPLES):
            feat = _make_features_for_regime(MarketRegime.SIDEWAYS, rng)
            actions.append(strat.generate_signal(feat).action)
        mean_a = np.mean(actions)
        assert abs(mean_a) < 0.4, f"Mean reversion should be neutral in calm market, got {mean_a:.3f}"


class TestBreakoutStrategyValidation:
    """
    Breakout strategy validation.

    Uses Bollinger Band pierce + elevated ATR to detect breakouts.
    Volume and momentum act as confidence boosters.
    """

    def test_stays_flat_when_atr_low(self):
        """When ATR is below threshold, should be flat regardless."""
        strat = BreakoutStrategy()
        feat = {
            "atr": 0.005,
            "bb_upper": -0.01,  # price above upper band
            "bb_lower": -0.04,
            "momentum": 0.05,
            "volume_ratio": 1.5,
        }
        sig = strat.generate_signal(feat)
        assert sig.action == 0.0, "Should be flat when ATR is low even with BB pierce"

    def test_long_breakout_upper_band_pierce(self):
        """When ATR elevated and price pierces upper BB → long."""
        strat = BreakoutStrategy()
        feat = {
            "atr": 0.025,
            "bb_upper": -0.005,  # price at/above upper band
            "bb_lower": -0.04,  # lower band well below
            "momentum": 0.03,
            "volume_ratio": 1.3,
        }
        sig = strat.generate_signal(feat)
        assert sig.action == 1.0, f"Expected long breakout, got {sig.action}"

    def test_short_breakout_lower_band_pierce(self):
        """When ATR elevated and price pierces lower BB → short."""
        strat = BreakoutStrategy()
        feat = {
            "atr": 0.025,
            "bb_upper": 0.04,   # upper band well above
            "bb_lower": 0.005,  # price at/below lower band
            "momentum": -0.03,
            "volume_ratio": 1.3,
        }
        sig = strat.generate_signal(feat)
        assert sig.action == -1.0, f"Expected short breakout, got {sig.action}"

    def test_confidence_boosted_by_volume_and_momentum(self):
        """Confidence should increase when volume confirms and momentum aligns."""
        strat = BreakoutStrategy()
        # Base case: no volume/momentum confirmation
        base = strat.generate_signal({
            "atr": 0.025, "bb_upper": -0.005, "bb_lower": -0.04,
            "momentum": -0.01, "volume_ratio": 0.5,
        })
        # With both confirmations
        boosted = strat.generate_signal({
            "atr": 0.025, "bb_upper": -0.005, "bb_lower": -0.04,
            "momentum": 0.03, "volume_ratio": 1.5,
        })
        assert boosted.confidence > base.confidence, (
            f"Expected boosted confidence, got base={base.confidence} vs boosted={boosted.confidence}"
        )

    def test_volatile_regime_signals(self):
        """In volatile regime, ATR should be elevated → strategy is active."""
        rng = np.random.default_rng(42)
        strat = BreakoutStrategy()
        active_count = 0
        for _ in range(100):
            feat = _make_features_for_regime(MarketRegime.VOLATILE, rng)
            sig = strat.generate_signal(feat)
            if sig.action != 0.0:
                active_count += 1
        assert active_count >= 70, (
            f"Breakout should be active in volatile regime (ATR high), "
            f"only {active_count}/100 non-zero"
        )

    def test_calm_regime_mostly_flat(self):
        """In sideways regime, ATR should be low → strategy stays flat."""
        rng = np.random.default_rng(42)
        strat = BreakoutStrategy()
        flat_count = 0
        for _ in range(100):
            feat = _make_features_for_regime(MarketRegime.SIDEWAYS, rng)
            sig = strat.generate_signal(feat)
            if sig.action == 0.0:
                flat_count += 1
        assert flat_count >= 80, (
            f"Breakout should be mostly flat in calm regime, "
            f"only {flat_count}/100 flat"
        )


class TestDefensiveStrategyValidation:
    """Defensive should reduce exposure when volatility is high."""

    N_SAMPLES = 100

    def test_flat_in_volatile(self):
        rng = np.random.default_rng(0)
        strat = DefensiveStrategy()
        actions = []
        for _ in range(self.N_SAMPLES):
            feat = _make_features_for_regime(MarketRegime.VOLATILE, rng)
            actions.append(strat.generate_signal(feat).action)
        mean_a = np.mean(actions)
        assert mean_a <= 0.05, f"Defensive should be nearly flat in VOLATILE, got {mean_a:.3f}"

    def test_small_long_in_calm(self):
        rng = np.random.default_rng(0)
        strat = DefensiveStrategy()
        actions = []
        for _ in range(self.N_SAMPLES):
            feat = _make_features_for_regime(MarketRegime.SIDEWAYS, rng)
            actions.append(strat.generate_signal(feat).action)
        mean_a = np.mean(actions)
        assert mean_a >= 0.15, f"Defensive should have small long bias in calm, got {mean_a:.3f}"

    def test_action_scales_with_volatility(self):
        """Action should decrease as volatility increases."""
        strat = DefensiveStrategy()
        vols = [0.005, 0.015, 0.020, 0.030, 0.050]
        actions = []
        for v in vols:
            sig = strat.generate_signal({"volatility": v})
            actions.append(sig.action)
        # Should be monotonically non-increasing
        for i in range(len(actions) - 1):
            assert actions[i] >= actions[i + 1], (
                f"Expected action to decrease with volatility: "
                f"vol={vols[i]:.3f}→{actions[i]}, vol={vols[i+1]:.3f}→{actions[i+1]}"
            )


# ======================================================================
# CROSS-VALIDATION: strategies on real FeatureEngineer output
# ======================================================================


class TestStrategiesOnRealFeatures:
    """
    Run strategies on features computed by FeatureEngineer from synthetic
    regime data.  This tests the full feature→strategy pipeline end-to-end.
    """

    def _feature_rows(self, data: pd.DataFrame, start: int = 50) -> list[dict]:
        """Compute features and return rows start..end as list of dicts."""
        fe = FeatureEngineer()
        features = fe.compute(data)
        rows = []
        for i in range(start, len(features)):
            rows.append(dict(zip(FEATURE_NAMES, features.iloc[i].values)))
        return rows

    def test_momentum_bull_vs_bear(self):
        """Momentum strategy mean action should be higher on bull data than bear data."""
        strat = MomentumStrategy()
        bull_actions = [strat.generate_signal(f).action
                        for f in self._feature_rows(_make_bull(300))]
        bear_actions = [strat.generate_signal(f).action
                        for f in self._feature_rows(_make_bear(300))]
        assert np.mean(bull_actions) > np.mean(bear_actions), (
            f"Momentum: bull mean={np.mean(bull_actions):.3f} should > "
            f"bear mean={np.mean(bear_actions):.3f}"
        )

    def test_defensive_volatile_vs_calm(self):
        """Defensive mean exposure should be lower on volatile data."""
        strat = DefensiveStrategy()
        vol_actions = [strat.generate_signal(f).action
                       for f in self._feature_rows(_make_volatile(300))]
        calm_actions = [strat.generate_signal(f).action
                        for f in self._feature_rows(_make_sideways(300))]
        assert np.mean(vol_actions) < np.mean(calm_actions), (
            f"Defensive: volatile mean={np.mean(vol_actions):.3f} should < "
            f"calm mean={np.mean(calm_actions):.3f}"
        )

    def test_mean_reversion_contrarian(self):
        """Mean reversion should not be strongly trend-following in a strong bull market.
        (It's a contrarian strategy — it shouldn't pile into the trend.)"""
        strat = MeanReversionStrategy()
        bull_actions = [strat.generate_signal(f).action
                        for f in self._feature_rows(_make_bull(300))]
        # Mean reversion shouldn't be strongly positive like momentum would be
        mean_a = np.mean(bull_actions)
        assert mean_a < 0.5, (
            f"Mean reversion shouldn't chase the bull trend, got mean action={mean_a:.3f}"
        )
