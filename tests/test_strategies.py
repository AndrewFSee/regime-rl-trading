"""
Tests for trading strategy modules.

All tests run with only numpy and do not require any ML framework.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_state(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.1, 0.1, n).astype(np.float32)


def _assert_valid_signal(signal, name: str):
    from src.strategies.base import TradeSignal
    assert isinstance(signal, TradeSignal), f"{name}: expected TradeSignal, got {type(signal)}"
    assert -1.0 <= signal.action <= 1.0, f"{name}: action {signal.action} out of [-1, 1]"
    assert 0.0 <= signal.confidence <= 1.0, f"{name}: confidence {signal.confidence} out of [0, 1]"
    assert isinstance(signal.strategy_name, str), f"{name}: strategy_name must be str"


# ---------------------------------------------------------------------------
# TradeSignal dataclass
# ---------------------------------------------------------------------------

def test_trade_signal_creation():
    from src.strategies.base import TradeSignal
    sig = TradeSignal(action=0.5, confidence=0.8, strategy_name="test")
    assert sig.action == 0.5
    assert sig.confidence == 0.8
    assert sig.strategy_name == "test"


# ---------------------------------------------------------------------------
# MomentumStrategy
# ---------------------------------------------------------------------------

class TestMomentumStrategy:
    def test_instantiation(self):
        from src.strategies.momentum import MomentumStrategy
        s = MomentumStrategy()
        assert s.name == "momentum"

    def test_signal_valid(self):
        from src.strategies.momentum import MomentumStrategy
        s = MomentumStrategy()
        sig = s.generate_signal(_random_state(10))
        _assert_valid_signal(sig, "MomentumStrategy")

    def test_strong_buy(self):
        from src.strategies.momentum import MomentumStrategy
        s = MomentumStrategy(short_ma_idx=0, long_ma_idx=1, momentum_idx=2)
        # short_ma > long_ma, momentum > 0  → strong buy
        state = np.array([0.05, 0.01, 0.03])
        sig = s.generate_signal(state)
        assert sig.action == 1.0

    def test_strong_sell(self):
        from src.strategies.momentum import MomentumStrategy
        s = MomentumStrategy(short_ma_idx=0, long_ma_idx=1, momentum_idx=2)
        # short_ma < long_ma, momentum < 0  → strong sell
        state = np.array([-0.05, 0.01, -0.03])
        sig = s.generate_signal(state)
        assert sig.action == -1.0

    def test_moderate_buy(self):
        from src.strategies.momentum import MomentumStrategy
        s = MomentumStrategy(short_ma_idx=0, long_ma_idx=1, momentum_idx=2)
        # short_ma > long_ma, momentum == 0 → moderate buy
        state = np.array([0.05, 0.01, 0.0])
        sig = s.generate_signal(state)
        assert sig.action == 0.5

    def test_moderate_sell(self):
        from src.strategies.momentum import MomentumStrategy
        s = MomentumStrategy(short_ma_idx=0, long_ma_idx=1, momentum_idx=2)
        # short_ma < long_ma, momentum == 0 → moderate sell
        state = np.array([-0.05, 0.01, 0.0])
        sig = s.generate_signal(state)
        assert sig.action == -0.5

    def test_short_state_doesnt_crash(self):
        from src.strategies.momentum import MomentumStrategy
        s = MomentumStrategy()
        sig = s.generate_signal(np.array([0.01]))  # much shorter than expected indices
        _assert_valid_signal(sig, "MomentumStrategy short state")


# ---------------------------------------------------------------------------
# MeanReversionStrategy
# ---------------------------------------------------------------------------

class TestMeanReversionStrategy:
    def test_instantiation(self):
        from src.strategies.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        assert s.name == "mean_reversion"

    def test_signal_valid(self):
        from src.strategies.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        sig = s.generate_signal(_random_state(10))
        _assert_valid_signal(sig, "MeanReversionStrategy")

    def test_oversold_buy(self):
        from src.strategies.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy(rsi_idx=0, bb_upper_idx=1, bb_lower_idx=2)
        # rsi < -0.4 (≈ RSI < 30) → buy
        state = np.array([-0.5, 0.02, -0.02])
        sig = s.generate_signal(state)
        assert sig.action > 0

    def test_overbought_sell(self):
        from src.strategies.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy(rsi_idx=0, bb_upper_idx=1, bb_lower_idx=2)
        # rsi > 0.4 (≈ RSI > 70) → sell
        state = np.array([0.6, 0.02, -0.02])
        sig = s.generate_signal(state)
        assert sig.action < 0

    def test_no_signal_neutral(self):
        from src.strategies.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy(rsi_idx=0, bb_upper_idx=1, bb_lower_idx=2)
        state = np.array([0.0, 0.02, -0.02])
        sig = s.generate_signal(state)
        assert sig.action == 0.0


# ---------------------------------------------------------------------------
# BreakoutStrategy
# ---------------------------------------------------------------------------

class TestBreakoutStrategy:
    def test_instantiation(self):
        from src.strategies.breakout import BreakoutStrategy
        s = BreakoutStrategy()
        assert s.name == "breakout"

    def test_signal_valid(self):
        from src.strategies.breakout import BreakoutStrategy
        s = BreakoutStrategy()
        sig = s.generate_signal(_random_state(10))
        _assert_valid_signal(sig, "BreakoutStrategy")

    def test_no_breakout_when_atr_low(self):
        from src.strategies.breakout import BreakoutStrategy
        s = BreakoutStrategy(atr_idx=0, high_idx=1, low_idx=2)
        state = np.array([0.005, 0.001, -0.001])  # ATR below baseline×1.5
        sig = s.generate_signal(state)
        assert sig.action == 0.0

    def test_long_breakout_near_high(self):
        from src.strategies.breakout import BreakoutStrategy
        s = BreakoutStrategy(atr_idx=0, high_idx=1, low_idx=2)
        # elevated ATR, price near recent high (high≈0 means price==high)
        state = np.array([0.02, 0.001, -0.05])
        sig = s.generate_signal(state)
        assert sig.action > 0

    def test_short_breakout_near_low(self):
        from src.strategies.breakout import BreakoutStrategy
        s = BreakoutStrategy(atr_idx=0, high_idx=1, low_idx=2)
        state = np.array([0.02, 0.05, -0.001])
        sig = s.generate_signal(state)
        assert sig.action < 0


# ---------------------------------------------------------------------------
# DefensiveStrategy
# ---------------------------------------------------------------------------

class TestDefensiveStrategy:
    def test_instantiation(self):
        from src.strategies.defensive import DefensiveStrategy
        s = DefensiveStrategy()
        assert s.name == "defensive"

    def test_signal_valid(self):
        from src.strategies.defensive import DefensiveStrategy
        s = DefensiveStrategy()
        sig = s.generate_signal(_random_state(10))
        _assert_valid_signal(sig, "DefensiveStrategy")

    def test_high_vol_neutral(self):
        from src.strategies.defensive import DefensiveStrategy
        s = DefensiveStrategy(volatility_idx=0)
        state = np.array([0.05])   # well above HIGH_VOL=0.025
        sig = s.generate_signal(state)
        assert sig.action == 0.0

    def test_low_vol_small_long(self):
        from src.strategies.defensive import DefensiveStrategy
        s = DefensiveStrategy(volatility_idx=0)
        state = np.array([0.005])  # below MED_VOL=0.015
        sig = s.generate_signal(state)
        assert sig.action == 0.2

    def test_med_vol_reduced(self):
        from src.strategies.defensive import DefensiveStrategy
        s = DefensiveStrategy(volatility_idx=0)
        state = np.array([0.020])  # between MED_VOL and HIGH_VOL
        sig = s.generate_signal(state)
        assert sig.action == 0.1
