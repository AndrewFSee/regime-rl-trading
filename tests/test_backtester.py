"""
Unit tests for Backtester metric helpers.

These are pure numerical functions and require only numpy.
"""
import math

import numpy as np
import pytest

from src.evaluation.backtester import Backtester


# ---------------------------------------------------------------------------
# _annualised_return
# ---------------------------------------------------------------------------

class TestAnnualisedReturn:
    def test_zero_days(self):
        assert Backtester._annualised_return(0.10, 0) == 0.0

    def test_one_year(self):
        result = Backtester._annualised_return(0.10, 252)
        assert result == pytest.approx(0.10, abs=1e-6)

    def test_half_year(self):
        # 5% in 126 days → annualised ≈ (1.05)^2 - 1 ≈ 10.25%
        result = Backtester._annualised_return(0.05, 126)
        assert result == pytest.approx((1.05 ** 2) - 1, abs=1e-6)

    def test_negative_return(self):
        result = Backtester._annualised_return(-0.10, 252)
        assert result == pytest.approx(-0.10, abs=1e-6)


# ---------------------------------------------------------------------------
# _sharpe
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_constant_returns(self):
        # Zero std → Sharpe = 0
        assert Backtester._sharpe([0.01, 0.01, 0.01]) == 0.0

    def test_too_few_returns(self):
        assert Backtester._sharpe([0.01]) == 0.0

    def test_positive_sharpe(self):
        returns = [0.01, 0.02, 0.015, 0.005, 0.012]
        result = Backtester._sharpe(returns)
        assert result > 0

    def test_negative_sharpe(self):
        returns = [-0.01, -0.02, -0.015, -0.005, -0.012]
        result = Backtester._sharpe(returns)
        assert result < 0


# ---------------------------------------------------------------------------
# _sortino
# ---------------------------------------------------------------------------

class TestSortino:
    def test_no_downside(self):
        # All positive → Sortino is undefined (no downside variance). The
        # backtester returns NaN so downstream aggregations flag it instead
        # of silently averaging in a sentinel.
        import math
        result = Backtester._sortino([0.01, 0.02, 0.005])
        assert math.isnan(result)

    def test_all_negative(self):
        result = Backtester._sortino([-0.01, -0.02, -0.005])
        assert result < 0

    def test_mixed_returns(self):
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        result = Backtester._sortino(returns)
        assert result > 0


# ---------------------------------------------------------------------------
# _max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_monotonically_increasing(self):
        values = [100, 110, 120, 130]
        assert Backtester._max_drawdown(values) == 0.0

    def test_simple_drawdown(self):
        values = [100, 110, 90, 95]
        # Peak = 110, trough = 90 → dd = 20/110 ≈ 0.18182
        assert Backtester._max_drawdown(values) == pytest.approx(20 / 110, abs=1e-6)

    def test_empty(self):
        assert Backtester._max_drawdown([]) == 0.0

    def test_single_value(self):
        assert Backtester._max_drawdown([100]) == 0.0


# ---------------------------------------------------------------------------
# _calmar
# ---------------------------------------------------------------------------

class TestCalmar:
    def test_no_drawdown_positive_return(self):
        result = Backtester._calmar(0.15, 0.0)
        assert result == float("inf")

    def test_no_drawdown_zero_return(self):
        assert Backtester._calmar(0.0, 0.0) == 0.0

    def test_normal_case(self):
        assert Backtester._calmar(0.20, 0.10) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _win_rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_wins(self):
        assert Backtester._win_rate([0.01, 0.02, 0.005]) == pytest.approx(1.0)

    def test_all_losses(self):
        assert Backtester._win_rate([-0.01, -0.02, -0.005]) == pytest.approx(0.0)

    def test_half_and_half(self):
        assert Backtester._win_rate([0.01, -0.01]) == pytest.approx(0.5)

    def test_empty(self):
        assert Backtester._win_rate([]) == 0.0


# ---------------------------------------------------------------------------
# _profit_factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_no_losses(self):
        result = Backtester._profit_factor([0.01, 0.02])
        assert result == float("inf")

    def test_no_gains(self):
        result = Backtester._profit_factor([-0.01, -0.02])
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_balanced(self):
        # Gains = 0.03, losses = 0.03 → PF = 1.0
        result = Backtester._profit_factor([0.01, 0.02, -0.01, -0.02])
        assert result == pytest.approx(1.0)
