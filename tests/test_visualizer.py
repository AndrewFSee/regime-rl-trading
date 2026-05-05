"""
Smoke tests for Visualizer.

Verifies that all plot methods run without errors and return Figure objects.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")

from src.evaluation.visualizer import Visualizer
from src.regime_detection.base import MarketRegime


@pytest.fixture
def sample_results():
    """Minimal backtest-like results dict."""
    n = 50
    rng = np.random.default_rng(42)
    values = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    regimes = [MarketRegime(rng.integers(0, 4)) for _ in range(n - 1)]
    actions = [rng.dirichlet([1, 1, 1, 1]) for _ in range(n - 1)]
    return {
        "portfolio_values": values.tolist(),
        "regime_history": regimes,
        "action_history": actions,
    }


class TestVisualizer:
    def test_plot_equity_curve(self, sample_results):
        viz = Visualizer()
        fig = viz.plot_equity_curve(
            sample_results["portfolio_values"],
            sample_results["regime_history"],
        )
        assert fig is not None
        matplotlib.pyplot.close(fig)

    def test_plot_strategy_allocation(self, sample_results):
        viz = Visualizer()
        fig = viz.plot_strategy_allocation(sample_results["action_history"])
        assert fig is not None
        matplotlib.pyplot.close(fig)

    def test_plot_drawdown(self, sample_results):
        viz = Visualizer()
        fig = viz.plot_drawdown(sample_results["portfolio_values"])
        assert fig is not None
        matplotlib.pyplot.close(fig)

    def test_plot_regime_distribution(self, sample_results):
        viz = Visualizer()
        fig = viz.plot_regime_distribution(sample_results["regime_history"])
        assert fig is not None
        matplotlib.pyplot.close(fig)

    def test_plot_all_returns_four_figures(self, sample_results):
        viz = Visualizer()
        figs = viz.plot_all(sample_results)
        assert len(figs) == 4
        for f in figs:
            matplotlib.pyplot.close(f)

    def test_plot_all_save(self, sample_results):
        import tempfile, os
        viz = Visualizer()
        with tempfile.TemporaryDirectory() as td:
            figs = viz.plot_all(sample_results, save_dir=td)
            saved = os.listdir(td)
            assert len(saved) == 4
            assert all(f.endswith(".png") for f in saved)
            for f in figs:
                matplotlib.pyplot.close(f)

    def test_empty_action_history(self):
        viz = Visualizer()
        fig = viz.plot_strategy_allocation([])
        assert fig is not None
        matplotlib.pyplot.close(fig)
