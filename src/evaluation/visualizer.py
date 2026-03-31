"""
Visualisation utilities for backtesting results.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    _SEABORN_AVAILABLE = True
except ImportError:
    _SEABORN_AVAILABLE = False

from ..regime_detection.base import MarketRegime

_REGIME_COLORS = {
    MarketRegime.BULL:     "#2ca02c",
    MarketRegime.BEAR:     "#d62728",
    MarketRegime.SIDEWAYS: "#ff7f0e",
    MarketRegime.VOLATILE: "#9467bd",
}
_STRATEGY_NAMES = ["Momentum", "MeanReversion", "Breakout", "Defensive"]


def _require_matplotlib():
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualisation. pip install matplotlib")


class Visualizer:
    """
    Plotting utilities for regime-RL backtest results.

    All methods return a :class:`matplotlib.figure.Figure` so they can be
    saved, embedded in notebooks, or displayed interactively.
    """

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_equity_curve(
        self,
        portfolio_values: list[float],
        regime_history: list | None = None,
        title: str = "Equity Curve",
    ):
        _require_matplotlib()
        fig, ax = plt.subplots(figsize=(12, 5))
        xs = np.arange(len(portfolio_values))
        ax.plot(xs, portfolio_values, linewidth=1.5, color="#1f77b4", label="Portfolio")

        # Shade background by regime
        if regime_history:
            for i, regime in enumerate(regime_history):
                if i < len(portfolio_values) - 1:
                    color = _REGIME_COLORS.get(regime, "white")
                    ax.axvspan(i, i + 1, alpha=0.15, color=color, linewidth=0)

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio Value ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        if regime_history:
            patches = [
                mpatches.Patch(color=c, alpha=0.4, label=r.name)
                for r, c in _REGIME_COLORS.items()
            ]
            ax.legend(handles=patches, loc="upper left", fontsize=8)

        fig.tight_layout()
        return fig

    def plot_strategy_allocation(
        self,
        action_history: list,
        title: str = "Strategy Allocation",
    ):
        _require_matplotlib()
        if not action_history:
            fig, ax = plt.subplots()
            ax.set_title(title)
            return fig

        arr = np.array(action_history, dtype=float)
        n_strategies = arr.shape[1] if arr.ndim == 2 else 4
        names = _STRATEGY_NAMES[:n_strategies]

        fig, ax = plt.subplots(figsize=(12, 4))
        bottom = np.zeros(len(arr))
        colors = plt.cm.tab10(np.linspace(0, 1, n_strategies))
        for i, (name, color) in enumerate(zip(names, colors)):
            ax.bar(np.arange(len(arr)), arr[:, i], bottom=bottom, label=name, color=color, width=1.0)
            bottom += arr[:, i]

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Weight")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        return fig

    def plot_drawdown(
        self,
        portfolio_values: list[float],
        title: str = "Drawdown",
    ):
        _require_matplotlib()
        arr = np.array(portfolio_values, dtype=float)
        running_max = np.maximum.accumulate(arr)
        drawdown = (running_max - arr) / np.where(running_max == 0, 1, running_max)

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(np.arange(len(drawdown)), -drawdown, 0, color="#d62728", alpha=0.6)
        ax.plot(np.arange(len(drawdown)), -drawdown, color="#d62728", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        fig.tight_layout()
        return fig

    def plot_regime_distribution(
        self,
        regime_history: list,
        title: str = "Regime Distribution",
    ):
        _require_matplotlib()
        from collections import Counter
        counts = Counter(regime_history)
        regimes = list(MarketRegime)
        values  = [counts.get(r, 0) for r in regimes]
        colors  = [_REGIME_COLORS[r] for r in regimes]
        labels  = [r.name for r in regimes]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Count")
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(val),
                    ha="center", va="bottom", fontsize=9,
                )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Composite plot
    # ------------------------------------------------------------------

    def plot_all(
        self,
        results: dict,
        save_dir: Optional[str] = None,
    ) -> list:
        """
        Generate all four plots and optionally save them to *save_dir*.

        Returns a list of :class:`matplotlib.figure.Figure` objects.
        """
        figs = []
        portfolio_values = results.get("portfolio_values", [])
        regime_history   = results.get("regime_history", [])
        action_history   = results.get("action_history", [])

        figs.append(self.plot_equity_curve(portfolio_values, regime_history))
        figs.append(self.plot_strategy_allocation(action_history))
        figs.append(self.plot_drawdown(portfolio_values))
        figs.append(self.plot_regime_distribution(regime_history))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            names = ["equity_curve", "strategy_allocation", "drawdown", "regime_distribution"]
            for fig, name in zip(figs, names):
                fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")

        return figs
