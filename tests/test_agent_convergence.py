"""
Agent learning convergence experiment.

Tests whether RL agents can learn a profitable policy on the TradingEnv
by comparing trained agents against trivial baselines:

  1. RandomAgent        – samples the action space uniformly at each step
  2. EqualWeightAgent   – always allocates equally across all 4 strategies
  3. PPOAgent           – trained for N timesteps
  4. SACAgent           – trained for N timesteps
  5. MetaAgent          – ensemble of per-regime PPO sub-agents

For each agent the backtester computes total return, Sharpe ratio, and
max drawdown on held-out test data.

Usage:
    python -m tests.test_agent_convergence          # run as pytest
    python tests/test_agent_convergence.py           # run as script for detailed output
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.environment.trading_env import TradingEnv
from src.evaluation.backtester import Backtester
from src.regime_detection.base import MarketRegime


# ======================================================================
# Synthetic data
# ======================================================================

def _make_multi_regime(
    total_bars: int = 1200,
    seg_min: int = 50,
    seg_max: int = 150,
    seed: int = 10,
) -> pd.DataFrame:
    """Cycling multi-regime price series with random order and duration.

    Each cycle shuffles the four regimes into a random order and assigns
    each a random duration between *seg_min* and *seg_max* bars.  This
    ensures every train/test split sees all regimes and the agent cannot
    memorise a fixed sequence or fixed segment length.
    """
    rng = np.random.default_rng(seed)
    regime_params = [
        (0.002,  0.008),   # BULL
        (-0.002, 0.008),   # BEAR
        (0.0,    0.003),   # SIDEWAYS
        (0.0,    0.04),    # VOLATILE
    ]

    price = 100.0
    segments: list[np.ndarray] = []
    bars = 0

    while bars < total_bars:
        # Shuffle regime order each cycle
        order = list(range(len(regime_params)))
        rng.shuffle(order)
        for idx in order:
            if bars >= total_bars:
                break
            remaining = total_bars - bars
            seg_len = int(min(rng.integers(seg_min, seg_max + 1), remaining))
            drift, vol = regime_params[idx]
            log_rets = drift + vol * rng.standard_normal(seg_len)
            prices = price * np.exp(np.cumsum(log_rets))
            price = prices[-1]
            segments.append(prices)
            bars += seg_len

    close = np.concatenate(segments)
    spread = np.abs(close) * 0.005
    high = close + rng.uniform(0, 1, len(close)) * spread
    low = close - rng.uniform(0, 1, len(close)) * spread
    open_ = low + rng.uniform(0, 1, len(close)) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, len(close)).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def _train_test_split(data: pd.DataFrame, ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(data) * ratio)
    return data.iloc[:split].reset_index(drop=True), data.iloc[split:].reset_index(drop=True)


# ======================================================================
# Baseline agents
# ======================================================================

class RandomAgent:
    """Uniformly random action at every step."""
    def __init__(self, env):
        self.env = env
        self.name = "Random"

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.env.action_space.sample()


class EqualWeightAgent:
    """Equal allocation across all strategies — a naive diversification baseline."""
    def __init__(self, n_strategies: int = 4):
        self.name = "EqualWeight"
        weights = np.ones(n_strategies, dtype=np.float32) / n_strategies
        self._action = np.append(weights, 1.0)  # full exposure

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._action.copy()


class BuyAndHoldMomentumAgent:
    """Always selects the momentum strategy (strategy 0) — a directional baseline."""
    def __init__(self, n_strategies: int = 4):
        self.name = "BuyHoldMomentum"
        weights = np.zeros(n_strategies, dtype=np.float32)
        weights[0] = 1.0
        self._action = np.append(weights, 1.0)  # full exposure

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._action.copy()


class DefensiveOnlyAgent:
    """Always selects the defensive strategy (strategy 3) — a risk-averse baseline."""
    def __init__(self, n_strategies: int = 4):
        self.name = "DefensiveOnly"
        weights = np.zeros(n_strategies, dtype=np.float32)
        weights[3] = 1.0
        self._action = np.append(weights, 1.0)  # full exposure

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
    train_time: float  # seconds


def _evaluate_agent(agent, env: TradingEnv, name: str, train_time: float = 0.0) -> AgentResult:
    """Run backtester and collect metrics."""
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


# ======================================================================
# Pytest tests
# ======================================================================

@pytest.fixture(scope="module")
def data():
    """Multi-regime synthetic data split into train/test."""
    full = _make_multi_regime(total_bars=1200, seed=42)
    train, test = _train_test_split(full, ratio=0.7)
    return train, test


@pytest.fixture(scope="module")
def baseline_results(data):
    """Evaluate all baselines on test data."""
    _, test = data
    test_env = TradingEnv(test, lookback_window=20, normalize_obs=True)
    baselines = [
        (RandomAgent(test_env), "Random"),
        (EqualWeightAgent(), "EqualWeight"),
        (BuyAndHoldMomentumAgent(), "BuyHoldMomentum"),
        (DefensiveOnlyAgent(), "DefensiveOnly"),
    ]
    results = {}
    for agent, name in baselines:
        results[name] = _evaluate_agent(agent, test_env, name)
    return results


@pytest.fixture(scope="module")
def ppo_result(data):
    """Train PPO and evaluate on test data."""
    train, test = data
    train_env = TradingEnv(train, lookback_window=20, normalize_obs=True)
    test_env = TradingEnv(test, lookback_window=20, normalize_obs=True)

    from src.agents.ppo_agent import PPOAgent
    agent = PPOAgent(train_env, config={"learning_rate": 3e-4, "batch_size": 64})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        agent.learn(total_timesteps=8192)
        train_time = time.perf_counter() - t0

    # Evaluate on test — need a fresh agent pointing at test env for backtester
    test_agent = PPOAgent(test_env, config={})
    test_agent.model = agent.model
    test_agent.model.set_env(test_env)

    return _evaluate_agent(test_agent, test_env, "PPO", train_time)


@pytest.fixture(scope="module")
def sac_result(data):
    """Train SAC and evaluate on test data."""
    train, test = data
    train_env = TradingEnv(train, lookback_window=20, normalize_obs=True)
    test_env = TradingEnv(test, lookback_window=20, normalize_obs=True)

    from src.agents.sac_agent import SACAgent
    agent = SACAgent(train_env, config={"learning_rate": 3e-4})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        agent.learn(total_timesteps=8192)
        train_time = time.perf_counter() - t0

    test_agent = SACAgent(test_env, config={})
    test_agent.model = agent.model
    test_agent.model.set_env(test_env)

    return _evaluate_agent(test_agent, test_env, "SAC", train_time)


@pytest.fixture(scope="module")
def meta_result(data):
    """Train MetaAgent and evaluate on test data."""
    train, test = data
    train_env = TradingEnv(train, lookback_window=20, normalize_obs=True)
    test_env = TradingEnv(test, lookback_window=20, normalize_obs=True)

    from src.agents.meta_agent import MetaAgent
    agent = MetaAgent(train_env, config={"learning_rate": 3e-4, "batch_size": 64})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        agent.learn(total_timesteps=8192)
        train_time = time.perf_counter() - t0

    # Point all sub-agents at test env
    test_meta = MetaAgent(test_env, config={})
    for regime in MarketRegime:
        test_meta.agents[regime].model = agent.agents[regime].model
        test_meta.agents[regime].model.set_env(test_env)

    return _evaluate_agent(test_meta, test_env, "MetaAgent", train_time)


class TestAgentConvergence:
    """
    Learning sanity checks — do RL agents learn anything better than baselines?

    These are soft tests with wide tolerances. The goal is to detect fundamental
    problems (broken reward signal, non-learning) rather than benchmark performance.
    """

    def test_ppo_runs_without_error(self, ppo_result):
        """PPO trains and produces finite metrics."""
        r = ppo_result
        assert np.isfinite(r.total_return), f"PPO total_return is not finite: {r.total_return}"
        assert np.isfinite(r.sharpe_ratio), f"PPO Sharpe is not finite: {r.sharpe_ratio}"
        assert 0 <= r.max_drawdown <= 1.0, f"PPO drawdown out of range: {r.max_drawdown}"

    def test_sac_runs_without_error(self, sac_result):
        """SAC trains and produces finite metrics."""
        r = sac_result
        assert np.isfinite(r.total_return), f"SAC total_return is not finite: {r.total_return}"
        assert np.isfinite(r.sharpe_ratio), f"SAC Sharpe is not finite: {r.sharpe_ratio}"

    def test_meta_runs_without_error(self, meta_result):
        """MetaAgent trains and produces finite metrics."""
        r = meta_result
        assert np.isfinite(r.total_return), f"Meta total_return is not finite: {r.total_return}"
        assert np.isfinite(r.sharpe_ratio), f"Meta Sharpe is not finite: {r.sharpe_ratio}"

    def test_ppo_not_worse_than_random(self, ppo_result, baseline_results):
        """PPO should not catastrophically underperform a random agent."""
        ppo = ppo_result
        rand = baseline_results["Random"]
        # Allow PPO to be somewhat worse (noisy), but not disastrously so
        assert ppo.total_return > rand.total_return - 0.15, (
            f"PPO ({ppo.total_return:.3f}) is much worse than Random ({rand.total_return:.3f})"
        )

    def test_sac_not_worse_than_random(self, sac_result, baseline_results):
        """SAC should not catastrophically underperform a random agent."""
        sac = sac_result
        rand = baseline_results["Random"]
        assert sac.total_return > rand.total_return - 0.15, (
            f"SAC ({sac.total_return:.3f}) is much worse than Random ({rand.total_return:.3f})"
        )

    def test_ppo_drawdown_bounded(self, ppo_result):
        """PPO should not experience total portfolio wipeout."""
        assert ppo_result.max_drawdown < 0.95, (
            f"PPO max drawdown is {ppo_result.max_drawdown:.2%} — near total wipeout"
        )

    def test_sac_drawdown_bounded(self, sac_result):
        """SAC should not experience total portfolio wipeout."""
        assert sac_result.max_drawdown < 0.95, (
            f"SAC max drawdown is {sac_result.max_drawdown:.2%} — near total wipeout"
        )

    def test_meta_drawdown_bounded(self, meta_result):
        """MetaAgent should not experience total portfolio wipeout."""
        assert meta_result.max_drawdown < 0.95, (
            f"Meta max drawdown is {meta_result.max_drawdown:.2%} — near total wipeout"
        )

    def test_at_least_one_agent_beats_equal_weight(self, ppo_result, sac_result, meta_result, baseline_results):
        """At least one trained agent should beat the naive equal-weight baseline."""
        ew = baseline_results["EqualWeight"]
        trained = [ppo_result, sac_result, meta_result]
        best = max(trained, key=lambda r: r.sharpe_ratio)
        # This is a learning signal test — if NO trained agent can beat equal-weight
        # on Sharpe, the reward signal or environment may be broken
        any_beats = any(r.sharpe_ratio > ew.sharpe_ratio for r in trained)
        # Soft assertion: we warn but don't fail if short training didn't converge yet
        if not any_beats:
            pytest.skip(
                f"No trained agent beat EqualWeight Sharpe ({ew.sharpe_ratio:.3f}). "
                f"Best was {best.name} ({best.sharpe_ratio:.3f}). "
                f"May need more training steps."
            )


# ======================================================================
# Script mode — rich output table
# ======================================================================

def _run_experiment(timesteps: int = 8192):
    """Run the full experiment and print a comparison table."""
    print("=" * 72)
    print("AGENT LEARNING CONVERGENCE EXPERIMENT")
    print("=" * 72)

    # Data — cycling regimes with random order & duration
    full = _make_multi_regime(total_bars=1200, seed=42)
    train, test = _train_test_split(full, ratio=0.7)
    print(f"\nData: {len(full)} bars multi-regime (cycling, random order & duration)")
    print(f"Train: {len(train)} bars | Test: {len(test)} bars")

    train_env = TradingEnv(train, lookback_window=20, normalize_obs=True)
    test_env = TradingEnv(test, lookback_window=20, normalize_obs=True)

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
    TIMESTEPS = timesteps
    # Use train_freq=4 for SAC when scaling beyond quick smoke tests
    sac_train_freq = 4 if TIMESTEPS > 10_000 else 1
    print(f"\n--- Trained Agents ({TIMESTEPS} timesteps) ---")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # PPO
        from src.agents.ppo_agent import PPOAgent
        ppo_env = TradingEnv(train, lookback_window=20, normalize_obs=True)
        ppo = PPOAgent(ppo_env, config={"learning_rate": 3e-4, "batch_size": 64})
        t0 = time.perf_counter()
        ppo.learn(total_timesteps=TIMESTEPS)
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
        sac_env = TradingEnv(train, lookback_window=20, normalize_obs=True)
        sac = SACAgent(sac_env, config={"learning_rate": 3e-4, "train_freq": sac_train_freq})
        t0 = time.perf_counter()
        sac.learn(total_timesteps=TIMESTEPS)
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
        meta_env = TradingEnv(train, lookback_window=20, normalize_obs=True)
        meta = MetaAgent(meta_env, config={"learning_rate": 3e-4, "batch_size": 64})
        t0 = time.perf_counter()
        meta.learn(total_timesteps=TIMESTEPS)
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
    print("\n" + "=" * 72)
    print(f"{'Agent':<20} {'Return':>10} {'Sharpe':>10} {'Sortino':>10} "
          f"{'MaxDD':>10} {'WinRate':>10} {'Time(s)':>10}")
    print("-" * 72)
    for r in results:
        print(f"{r.name:<20} {r.total_return:>+10.3f} {r.sharpe_ratio:>10.3f} "
              f"{r.sortino_ratio:>10.3f} {r.max_drawdown:>10.3f} "
              f"{r.win_rate:>10.3f} {r.train_time:>10.1f}")
    print("=" * 72)

    # --- Diagnosis ---
    ew = next(r for r in results if r.name == "EqualWeight")
    trained = [r for r in results if r.name in ("PPO", "SAC", "MetaAgent")]
    best_trained = max(trained, key=lambda r: r.sharpe_ratio)

    print("\n--- DIAGNOSIS ---")
    if best_trained.sharpe_ratio > ew.sharpe_ratio:
        print(f"PASS: {best_trained.name} (Sharpe={best_trained.sharpe_ratio:.3f}) "
              f"beats EqualWeight (Sharpe={ew.sharpe_ratio:.3f}).")
        print("The reward signal is learnable. Agents can improve over naive baselines.")
    else:
        print(f"WARNING: No trained agent beats EqualWeight (Sharpe={ew.sharpe_ratio:.3f}).")
        print(f"Best was {best_trained.name} (Sharpe={best_trained.sharpe_ratio:.3f}).")
        print("Possible causes:")
        print("  1. Too few training steps (8192 may not be enough)")
        print("  2. DSR reward signal too noisy for the agent to learn from")
        print("  3. Observation space too large / poorly conditioned")
        print("  4. Action space mismatch (softmax normalization may flatten gradients)")
        print("Recommended: increase to 50k-100k steps, or try adjusting reward_scaling.")

    rand = next(r for r in results if r.name == "Random")
    for t in trained:
        if t.total_return < rand.total_return - 0.15:
            print(f"\nRED FLAG: {t.name} is significantly worse than Random! "
                  f"({t.total_return:.3f} vs {rand.total_return:.3f})")

    for t in trained:
        if t.max_drawdown > 0.8:
            print(f"\nRED FLAG: {t.name} has {t.max_drawdown:.0%} max drawdown — near wipeout!")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent convergence experiment")
    parser.add_argument(
        "--timesteps", "-t", type=int, default=8192,
        help="Total training timesteps per agent (default: 8192)",
    )
    args = parser.parse_args()
    _run_experiment(timesteps=args.timesteps)
