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
            # No downside variance: Sortino is mathematically undefined. Return
            # NaN so downstream aggregations (mean across seeds) flag the issue
            # rather than silently averaging in a sentinel like 100.0.
            return float("nan")
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
                # Prefer the post-cost net return so that Sharpe / Sortino /
                # win-rate reflect the strategy actually realised. Fall back to
                # the gross return for envs that have not been updated yet.
                episode_returns.append(
                    info.get("net_return", info.get("portfolio_return", 0.0))
                )

                if "regime" in info:
                    all_regimes.append(info["regime"])
                all_actions.append(action)

            all_portfolio_values.extend(episode_values)
            all_returns.extend(episode_returns)

        # Benchmark: buy-and-hold the underlying asset
        close = np.asarray(self.env.data["Close"].values, dtype=float)
        if len(close) > 1:
            benchmark_return = float((close[-1] - close[0]) / close[0])
            bh_daily_returns = (close[1:] / close[:-1] - 1.0).tolist()
            # B&H portfolio path normalised to the same starting value as the
            # agent so MaxDD / Sharpe are computed on a comparable series.
            bh_path = (close / close[0]).tolist()
            bh_sharpe = self._sharpe(bh_daily_returns)
            bh_sortino = self._sortino(bh_daily_returns)
            bh_max_dd = self._max_drawdown(bh_path)
            bh_ann_return = self._annualised_return(benchmark_return, len(bh_daily_returns))
            bh_calmar = self._calmar(bh_ann_return, bh_max_dd)

            # Additional baselines computed on the same close series
            extra_baselines = self._compute_extra_baselines(close)
        else:
            benchmark_return = 0.0
            bh_sharpe = 0.0
            bh_sortino = 0.0
            bh_max_dd = 0.0
            bh_ann_return = 0.0
            bh_calmar = 0.0
            extra_baselines = {}

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
            "benchmark_annualized_return": bh_ann_return,
            "benchmark_sharpe_ratio":      bh_sharpe,
            "benchmark_sortino_ratio":     bh_sortino,
            "benchmark_max_drawdown":      bh_max_dd,
            "benchmark_calmar_ratio":      bh_calmar,
            "baselines":                   extra_baselines,
        }

    # ------------------------------------------------------------------
    # Additional rule-based baselines
    # ------------------------------------------------------------------

    def _compute_extra_baselines(self, close: np.ndarray) -> dict:
        """Compute Sharpe/MaxDD/Calmar/Total for several rule-based baselines
        on the same close series:

        * ``sixty_forty``      : 60% market / 40% cash (cash earns 0).
        * ``vol_target_10``    : SPY scaled to a 10%% annualised vol target
                                 using a 60-day trailing realised-vol estimate
                                 (lagged by 1 bar to avoid lookahead).
        * ``ma_filter_200``    : 100% market when close > 200d SMA, else cash.
        """
        results: dict[str, dict] = {}
        n = len(close)
        if n < 3:
            return results

        daily = close[1:] / close[:-1] - 1.0  # length n-1

        def _summary(weights: np.ndarray) -> dict:
            """Given per-bar exposure ``weights`` (length n-1, lagged), build
            the equity path and compute summary stats."""
            port_returns = weights * daily
            path = np.concatenate(([1.0], np.cumprod(1.0 + port_returns)))
            total = float(path[-1] - 1.0)
            sharpe = self._sharpe(port_returns.tolist())
            sortino = self._sortino(port_returns.tolist())
            max_dd = self._max_drawdown(path.tolist())
            ann = self._annualised_return(total, len(port_returns))
            calmar = self._calmar(ann, max_dd)
            return {
                "total_return": total,
                "annualized_return": ann,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown": max_dd,
                "calmar_ratio": calmar,
            }

        # 60/40 — fixed 0.6 weight on the asset
        results["sixty_forty"] = _summary(np.full(n - 1, 0.6))

        # Vol-target 10% on a 60-day trailing realised-vol estimate
        target_vol = 0.10
        window = 60
        # Compute realised vol up to bar i-1, applied to return on bar i.
        rv = np.zeros(n - 1, dtype=float)
        for i in range(1, n - 1):
            start = max(0, i - window)
            seg = daily[start:i]
            if len(seg) >= 5:
                rv[i] = float(np.std(seg) * np.sqrt(252))
            else:
                rv[i] = target_vol  # neutral until enough history
        # Avoid divide-by-zero
        rv = np.where(rv > 1e-6, rv, target_vol)
        weights_vt = np.clip(target_vol / rv, 0.0, 2.0)
        results["vol_target_10"] = _summary(weights_vt)

        # 200-day MA filter
        ma = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i - 199)
            ma[i] = float(np.mean(close[start: i + 1]))
        # Lag the signal by 1 bar (use yesterday's close vs MA to set today's exposure)
        signal = (close[:-1] > ma[:-1]).astype(float)  # length n-1
        results["ma_filter_200"] = _summary(signal)

        return results
