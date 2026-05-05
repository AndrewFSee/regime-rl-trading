"""
Real market data experiment.

Downloads historical OHLCV data via yfinance and runs the same agent
comparison as test_agent_convergence, but on real price history.

Usage:
    python -m tests.test_real_data                          # SPY, default settings
    python -m tests.test_real_data --ticker QQQ             # different ticker
    python -m tests.test_real_data --timesteps 100000       # more training
    python -m tests.test_real_data --start 2018-01-01       # custom date range
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.environment.data_loader import DataLoader
from src.environment.trading_env import TradingEnv
from src.evaluation.backtester import Backtester
from src.regime_detection.base import MarketRegime


# ======================================================================
# Baseline agents (same as convergence test)
# ======================================================================

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.name = "Random"

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.env.action_space.sample()


class EqualWeightAgent:
    def __init__(self, n_strategies: int = 4):
        self.name = "EqualWeight"
        weights = np.ones(n_strategies, dtype=np.float32) / n_strategies
        self._action = np.append(weights, 1.0)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._action.copy()


class BuyAndHoldMomentumAgent:
    def __init__(self, n_strategies: int = 4):
        self.name = "BuyHoldMomentum"
        weights = np.zeros(n_strategies, dtype=np.float32)
        weights[0] = 1.0
        self._action = np.append(weights, 1.0)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._action.copy()


class DefensiveOnlyAgent:
    def __init__(self, n_strategies: int = 4):
        self.name = "DefensiveOnly"
        weights = np.zeros(n_strategies, dtype=np.float32)
        weights[3] = 1.0
        self._action = np.append(weights, 1.0)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._action.copy()


# ======================================================================
# Result container
# ======================================================================

@dataclass
class AgentResult:
    name: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    train_time: float


# ======================================================================
# Helpers
# ======================================================================

def _evaluate_agent(agent, env: TradingEnv, name: str, train_time: float = 0.0) -> AgentResult:
    bt = Backtester(env, agent)
    results = bt.run(n_episodes=1)
    return AgentResult(
        name=name,
        total_return=results["total_return"],
        sharpe_ratio=results["sharpe_ratio"],
        sortino_ratio=results["sortino_ratio"],
        max_drawdown=results["max_drawdown"],
        win_rate=results["win_rate"],
        train_time=train_time,
    )


def _fetch_data(
    ticker: str, start: str, end: str, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download data and return (full, train, test)."""
    loader = DataLoader(
        tickers=[ticker], start_date=start, end_date=end, train_ratio=train_ratio
    )
    loader.fetch_data()
    train, test = loader.get_train_test_split(ticker)
    full = loader.get_data(ticker)
    return full, train, test


# ======================================================================
# Main experiment
# ======================================================================

def run_real_data_experiment(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    train_ratio: float = 0.7,
    timesteps: int = 50_000,
) -> list[AgentResult]:
    print("=" * 76)
    print(f"REAL DATA EXPERIMENT — {ticker}")
    print("=" * 76)

    # --- Fetch data ---
    print(f"\nFetching {ticker} ({start} → {end}) via yfinance ...")
    full, train, test = _fetch_data(ticker, start, end, train_ratio)
    print(f"Total: {len(full)} bars | Train: {len(train)} bars | Test: {len(test)} bars")

    train_start = train.index[0] if hasattr(train.index[0], 'strftime') else 0
    train_end = train.index[-1] if hasattr(train.index[-1], 'strftime') else len(train) - 1
    test_start = test.index[0] if hasattr(test.index[0], 'strftime') else 0
    test_end = test.index[-1] if hasattr(test.index[-1], 'strftime') else len(test) - 1
    print(f"Train period: {train_start} → {train_end}")
    print(f"Test  period: {test_start} → {test_end}")

    # Buy-and-hold benchmark
    bnh_return = (test["Close"].iloc[-1] / test["Close"].iloc[0]) - 1
    print(f"\nBuy-and-hold {ticker} return on test set: {bnh_return:+.1%}")

    # --- Build environments ---
    train_env = TradingEnv(train.reset_index(drop=True), lookback_window=20, normalize_obs=True)
    test_env = TradingEnv(test.reset_index(drop=True), lookback_window=20, normalize_obs=True)

    results: list[AgentResult] = []

    # --- Baselines ---
    print("\n--- Baselines ---")
    baselines = [
        (RandomAgent(test_env), "Random"),
        (EqualWeightAgent(), "EqualWeight"),
        (BuyAndHoldMomentumAgent(), "BuyHoldMomentum"),
        (DefensiveOnlyAgent(), "DefensiveOnly"),
    ]
    for agent, name in baselines:
        r = _evaluate_agent(agent, test_env, name)
        results.append(r)
        print(f"  {name}: return={r.total_return:+.3f}, sharpe={r.sharpe_ratio:.3f}, dd={r.max_drawdown:.3f}")

    # --- Trained agents ---
    sac_train_freq = 4 if timesteps > 10_000 else 1
    print(f"\n--- Trained Agents ({timesteps} timesteps) ---")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # PPO
        from src.agents.ppo_agent import PPOAgent
        ppo_env = TradingEnv(train.reset_index(drop=True), lookback_window=20, normalize_obs=True)
        ppo = PPOAgent(ppo_env, config={"learning_rate": 3e-4, "batch_size": 64})
        t0 = time.perf_counter()
        ppo.learn(total_timesteps=timesteps)
        ppo_time = time.perf_counter() - t0

        ppo_test = PPOAgent(test_env, config={})
        ppo_test.model = ppo.model
        ppo_test.model.set_env(test_env)
        r = _evaluate_agent(ppo_test, test_env, "PPO", ppo_time)
        results.append(r)
        print(f"  PPO: return={r.total_return:+.3f}, sharpe={r.sharpe_ratio:.3f}, "
              f"dd={r.max_drawdown:.3f}, time={r.train_time:.1f}s")

        # SAC
        from src.agents.sac_agent import SACAgent
        sac_env = TradingEnv(train.reset_index(drop=True), lookback_window=20, normalize_obs=True)
        sac = SACAgent(sac_env, config={"learning_rate": 3e-4, "train_freq": sac_train_freq})
        t0 = time.perf_counter()
        sac.learn(total_timesteps=timesteps)
        sac_time = time.perf_counter() - t0

        sac_test = SACAgent(test_env, config={})
        sac_test.model = sac.model
        sac_test.model.set_env(test_env)
        r = _evaluate_agent(sac_test, test_env, "SAC", sac_time)
        results.append(r)
        print(f"  SAC: return={r.total_return:+.3f}, sharpe={r.sharpe_ratio:.3f}, "
              f"dd={r.max_drawdown:.3f}, time={r.train_time:.1f}s")

        # MetaAgent
        from src.agents.meta_agent import MetaAgent
        meta_env = TradingEnv(train.reset_index(drop=True), lookback_window=20, normalize_obs=True)
        meta = MetaAgent(meta_env, config={"learning_rate": 3e-4, "batch_size": 64})
        t0 = time.perf_counter()
        meta.learn(total_timesteps=timesteps)
        meta_time = time.perf_counter() - t0

        meta_test = MetaAgent(test_env, config={})
        for regime in MarketRegime:
            meta_test.agents[regime].model = meta.agents[regime].model
            meta_test.agents[regime].model.set_env(test_env)
        r = _evaluate_agent(meta_test, test_env, "MetaAgent", meta_time)
        results.append(r)
        print(f"  Meta: return={r.total_return:+.3f}, sharpe={r.sharpe_ratio:.3f}, "
              f"dd={r.max_drawdown:.3f}, time={r.train_time:.1f}s")

    # --- Summary table ---
    print("\n" + "=" * 76)
    print(f"{'Agent':<20} {'Return':>10} {'Sharpe':>10} {'Sortino':>10} "
          f"{'MaxDD':>10} {'WinRate':>10} {'Time(s)':>10}")
    print("-" * 76)
    for r in results:
        print(f"{r.name:<20} {r.total_return:>+10.3f} {r.sharpe_ratio:>10.3f} "
              f"{r.sortino_ratio:>10.3f} {r.max_drawdown:>10.3f} "
              f"{r.win_rate:>10.3f} {r.train_time:>10.1f}")
    print("=" * 76)
    print(f"\n  Buy-and-hold {ticker}: {bnh_return:+.1%}")

    # --- Diagnosis ---
    ew = next(r for r in results if r.name == "EqualWeight")
    trained = [r for r in results if r.name in ("PPO", "SAC", "MetaAgent")]
    best_trained = max(trained, key=lambda r: r.sharpe_ratio)

    print("\n--- DIAGNOSIS ---")
    if best_trained.sharpe_ratio > ew.sharpe_ratio:
        print(f"PASS: {best_trained.name} (Sharpe={best_trained.sharpe_ratio:.3f}) "
              f"beats EqualWeight (Sharpe={ew.sharpe_ratio:.3f}).")
    else:
        print(f"WARNING: No trained agent beats EqualWeight (Sharpe={ew.sharpe_ratio:.3f}).")
        print(f"Best was {best_trained.name} (Sharpe={best_trained.sharpe_ratio:.3f}).")

    for t in trained:
        if t.total_return < -0.20:
            print(f"\nRED FLAG: {t.name} lost {abs(t.total_return):.0%} on real data!")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real market data experiment")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (default: 2015-01-01)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date (default: 2025-01-01)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train/test split ratio (default: 0.7)")
    parser.add_argument("--timesteps", "-t", type=int, default=50_000, help="Training timesteps (default: 50000)")
    args = parser.parse_args()

    run_real_data_experiment(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        train_ratio=args.train_ratio,
        timesteps=args.timesteps,
    )
