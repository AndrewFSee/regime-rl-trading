"""
Backtester: runs an agent through a TradingEnv and computes performance metrics.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class Backtester:
    """
    Evaluates an agent on a :class:`~src.environment.trading_env.TradingEnv`.

    Parameters
    ----------
    env:   A :class:`~src.environment.trading_env.TradingEnv` instance.
    agent: Any object with an ``act(obs) -> action`` method.
    """

    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent

    # ------------------------------------------------------------------
    # Metric helpers (static so they can be unit-tested independently)
    # ------------------------------------------------------------------

    @staticmethod
    def _annualised_return(total_return: float, n_days: int) -> float:
        if n_days <= 0:
            return 0.0
        return (1.0 + total_return) ** (252.0 / n_days) - 1.0

    @staticmethod
    def _sharpe(returns: list[float], risk_free: float = 0.0) -> float:
        arr = np.array(returns, dtype=float)
        if len(arr) < 2 or arr.std() == 0:
            return 0.0
        excess = arr - risk_free / 252.0
        return float(excess.mean() / arr.std() * np.sqrt(252))

    @staticmethod
    def _sortino(returns: list[float], risk_free: float = 0.0) -> float:
        arr = np.array(returns, dtype=float)
        if len(arr) < 2:
            return 0.0
        excess = arr - risk_free / 252.0
        downside = arr[arr < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf") if excess.mean() > 0 else 0.0
        return float(excess.mean() / downside.std() * np.sqrt(252))

    @staticmethod
    def _max_drawdown(portfolio_values: list[float]) -> float:
        arr = np.array(portfolio_values, dtype=float)
        if len(arr) == 0:
            return 0.0
        running_max = np.maximum.accumulate(arr)
        drawdowns = (running_max - arr) / np.where(running_max == 0, 1, running_max)
        return float(drawdowns.max())

    @staticmethod
    def _calmar(annualised_return: float, max_dd: float) -> float:
        if max_dd == 0:
            return float("inf") if annualised_return > 0 else 0.0
        return annualised_return / max_dd

    @staticmethod
    def _win_rate(returns: list[float]) -> float:
        arr = np.array(returns, dtype=float)
        if len(arr) == 0:
            return 0.0
        return float((arr > 0).mean())

    @staticmethod
    def _profit_factor(returns: list[float]) -> float:
        arr = np.array(returns, dtype=float)
        gross_profit = arr[arr > 0].sum()
        gross_loss   = abs(arr[arr < 0].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 1.0
        return float(gross_profit / gross_loss)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, n_episodes: int = 1) -> dict[str, Any]:
        """
        Run *n_episodes* episodes and aggregate performance metrics.

        Returns
        -------
        dict with keys:
            total_return, annualized_return, sharpe_ratio, sortino_ratio,
            max_drawdown, calmar_ratio, win_rate, profit_factor,
            portfolio_values, regime_history, action_history,
            benchmark_return
        """
        all_portfolio_values: list[float] = []
        all_returns: list[float] = []
        all_regimes: list = []
        all_actions: list = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_values = [self.env.initial_cash]
            episode_returns: list[float] = []

            while not done:
                action = self.agent.act(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_values.append(info.get("portfolio_value", episode_values[-1]))
                episode_returns.append(info.get("portfolio_return", 0.0))

                if "regime" in info:
                    all_regimes.append(info["regime"])
                all_actions.append(action)

            all_portfolio_values.extend(episode_values)
            all_returns.extend(episode_returns)

        # Benchmark: buy-and-hold the underlying asset
        close = self.env.data["Close"].values
        benchmark_return = float((close[-1] - close[0]) / close[0]) if len(close) > 1 else 0.0

        # Aggregate metrics
        initial_value = all_portfolio_values[0]
        final_value   = all_portfolio_values[-1]
        total_return  = (final_value - initial_value) / initial_value
        n_days        = len(all_returns)

        ann_return  = self._annualised_return(total_return, n_days)
        sharpe      = self._sharpe(all_returns)
        sortino     = self._sortino(all_returns)
        max_dd      = self._max_drawdown(all_portfolio_values)
        calmar      = self._calmar(ann_return, max_dd)
        win_rate    = self._win_rate(all_returns)
        profit_fac  = self._profit_factor(all_returns)

        return {
            "total_return":      total_return,
            "annualized_return": ann_return,
            "sharpe_ratio":      sharpe,
            "sortino_ratio":     sortino,
            "max_drawdown":      max_dd,
            "calmar_ratio":      calmar,
            "win_rate":          win_rate,
            "profit_factor":     profit_fac,
            "portfolio_values":  all_portfolio_values,
            "regime_history":    all_regimes,
            "action_history":    all_actions,
            "benchmark_return":  benchmark_return,
        }
