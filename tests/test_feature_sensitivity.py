"""
Feature sensitivity analysis for trained RL trading agents.

Trains an SAC agent, then measures how much each observation feature
group influences the agent's decisions by zeroing out each group
and computing the mean absolute action deviation.

Usage:
    python -m tests.test_feature_sensitivity
    python -m tests.test_feature_sensitivity --timesteps 200000
"""
from __future__ import annotations

import argparse
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.environment.data_loader import DataLoader
from src.environment.features import FEATURE_NAMES
from src.environment.trading_env import TradingEnv
from src.evaluation.backtester import Backtester


# ──────────────────────────────────────────────────────────────
# Feature groups: name → indices within the flat observation
#
# Observation layout (for lookback=20, n_features=10):
#   [0..199]   = 20 timesteps × 10 features  (row-major)
#   [200..203] = portfolio features (cash_ratio, position, pnl, drawdown)
#   [204..207] = regime one-hot (BULL, BEAR, SIDEWAYS, VOLATILE)
# ──────────────────────────────────────────────────────────────

LOOKBACK = 20
N_FEATURES = 10
N_PORTFOLIO = 4
N_REGIME = 4


def _feature_indices(feature_idx: int) -> list[int]:
    """Indices for a single technical feature across all lookback steps."""
    return [step * N_FEATURES + feature_idx for step in range(LOOKBACK)]


def _build_feature_groups() -> OrderedDict[str, list[int]]:
    """Map group names to their indices in the observation vector."""
    groups = OrderedDict()

    # Individual technical features (spread across lookback window)
    for i, name in enumerate(FEATURE_NAMES):
        groups[name] = _feature_indices(i)

    # Group related features together for summary
    groups["ALL_MA (short+long)"] = _feature_indices(2) + _feature_indices(3)
    groups["ALL_BB (upper+lower)"] = _feature_indices(4) + _feature_indices(5)
    groups["ALL_technical"] = list(range(LOOKBACK * N_FEATURES))

    # Portfolio state
    port_start = LOOKBACK * N_FEATURES
    groups["portfolio_state"] = list(range(port_start, port_start + N_PORTFOLIO))

    # Regime
    regime_start = port_start + N_PORTFOLIO
    groups["regime_one_hot"] = list(range(regime_start, regime_start + N_REGIME))

    return groups


def _fetch_data(
    ticker: str, start: str, end: str, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(
        tickers=[ticker], start_date=start, end_date=end, train_ratio=train_ratio
    )
    loader.fetch_data()
    return loader.get_train_test_split(ticker)


def _collect_observations_and_actions(agent, env) -> tuple[np.ndarray, np.ndarray]:
    """Run one episode and collect all (obs, action) pairs."""
    obs_list, act_list = [], []
    obs, _ = env.reset()
    done = False
    while not done:
        obs_list.append(obs.copy())
        action = agent.act(obs)
        act_list.append(action.copy())
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return np.array(obs_list), np.array(act_list)


def _perturbed_actions(agent, observations: np.ndarray, zero_indices: list[int]) -> np.ndarray:
    """Run the agent on observations with specific indices zeroed out."""
    actions = []
    for obs in observations:
        perturbed = obs.copy()
        perturbed[zero_indices] = 0.0
        action = agent.act(perturbed)
        actions.append(action)
    return np.array(actions)


def run_sensitivity(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    train_ratio: float = 0.7,
    timesteps: int = 200_000,
    return_weight: float = 0.0,
    exposure_penalty: float = 0.0,
) -> pd.DataFrame:
    print("=" * 80)
    print(f"FEATURE SENSITIVITY ANALYSIS — {ticker} — {timesteps} timesteps")
    print("=" * 80)

    # Fetch data
    print(f"\nFetching {ticker} ({start} → {end}) ...")
    train, test = _fetch_data(ticker, start, end, train_ratio)
    print(f"Train: {len(train)} bars | Test: {len(test)} bars")

    # Train SAC
    from src.agents.sac_agent import SACAgent

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        train_env = TradingEnv(
            train.reset_index(drop=True), lookback_window=LOOKBACK,
            normalize_obs=True, return_weight=return_weight,
            exposure_penalty=exposure_penalty,
            obs_features=FEATURE_NAMES,  # use all features for sensitivity analysis
        )
        sac = SACAgent(train_env, config={
            "learning_rate": 3e-4,
            "train_freq": 4,
        })
        print(f"\nTraining SAC for {timesteps} steps ...")
        t0 = time.perf_counter()
        sac.learn(total_timesteps=timesteps)
        print(f"  Training done in {time.perf_counter() - t0:.0f}s")

        # Evaluate baseline performance on test
        test_env = TradingEnv(
            test.reset_index(drop=True), lookback_window=LOOKBACK,
            normalize_obs=True, return_weight=return_weight,
            exposure_penalty=exposure_penalty,
            obs_features=FEATURE_NAMES,
        )
        sac_test = SACAgent(test_env, config={})
        sac_test.model = sac.model
        sac_test.model.set_env(test_env)

        bt = Backtester(test_env, sac_test)
        base_results = bt.run(n_episodes=1)
        print(f"\nBaseline test performance:")
        print(f"  Return: {base_results['total_return']:+.2%}")
        print(f"  Sharpe: {base_results['sharpe_ratio']:.3f}")

    # Collect observations and baseline actions on test set
    # Need a fresh env for clean collection
    test_env2 = TradingEnv(
        test.reset_index(drop=True), lookback_window=LOOKBACK,
        normalize_obs=True, return_weight=return_weight,
        exposure_penalty=exposure_penalty,
        obs_features=FEATURE_NAMES,
    )
    sac_test2 = SACAgent(test_env2, config={})
    sac_test2.model = sac.model
    sac_test2.model.set_env(test_env2)

    print("\nCollecting observations and actions ...")
    observations, baseline_actions = _collect_observations_and_actions(sac_test2, test_env2)
    print(f"  Collected {len(observations)} observation-action pairs")
    print(f"  Observation dim: {observations.shape[1]}")
    print(f"  Action dim: {baseline_actions.shape[1]}")

    # Compute mean absolute action for reference
    mean_abs_action = np.mean(np.abs(baseline_actions), axis=0)
    print(f"\n  Mean |action| per dim: {np.array2string(mean_abs_action, precision=3)}")
    print(f"  Mean exposure (dim 4): {np.mean(baseline_actions[:, 4]):.3f}")

    # Sensitivity analysis: zero each feature group and measure action change
    feature_groups = _build_feature_groups()
    print(f"\nRunning sensitivity analysis on {len(feature_groups)} feature groups ...\n")

    results = []
    for group_name, indices in feature_groups.items():
        perturbed = _perturbed_actions(sac_test2, observations, indices)
        # Mean absolute deviation in actions
        mad = np.mean(np.abs(perturbed - baseline_actions))
        # Max deviation
        max_dev = np.max(np.abs(perturbed - baseline_actions))
        # Mean deviation per action dimension
        mad_per_dim = np.mean(np.abs(perturbed - baseline_actions), axis=0)
        # Exposure change specifically
        exposure_change = np.mean(np.abs(perturbed[:, 4] - baseline_actions[:, 4]))

        results.append({
            "feature_group": group_name,
            "n_indices": len(indices),
            "mean_action_deviation": mad,
            "max_action_deviation": max_dev,
            "exposure_deviation": exposure_change,
            "weight_deviation": np.mean(mad_per_dim[:4]),
        })

        print(f"  {group_name:<25} ({len(indices):>3} dims): "
              f"MAD={mad:.4f}, max={max_dev:.4f}, "
              f"exposure_Δ={exposure_change:.4f}")

    df = pd.DataFrame(results).sort_values("mean_action_deviation", ascending=False)

    # Summary
    print("\n" + "=" * 80)
    print("SENSITIVITY RANKING (by mean action deviation when feature zeroed):")
    print("=" * 80)
    print(f"\n{'Feature Group':<25} {'Dims':>5} {'MAD':>10} {'Max Dev':>10} {'Exposure Δ':>12} {'Weight Δ':>10}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['feature_group']:<25} {row['n_indices']:>5} "
              f"{row['mean_action_deviation']:>10.4f} {row['max_action_deviation']:>10.4f} "
              f"{row['exposure_deviation']:>12.4f} {row['weight_deviation']:>10.4f}")
    print("-" * 80)

    # Identify low-impact features
    individual = df[~df["feature_group"].str.startswith("ALL_") &
                    ~df["feature_group"].isin(["portfolio_state", "regime_one_hot"])]
    threshold = individual["mean_action_deviation"].median() * 0.5
    low_impact = individual[individual["mean_action_deviation"] < threshold]
    if not low_impact.empty:
        print(f"\nLOW-IMPACT features (MAD < {threshold:.4f}, median/2):")
        for _, row in low_impact.iterrows():
            print(f"  - {row['feature_group']} (MAD={row['mean_action_deviation']:.4f})")

    high_impact = individual.nlargest(5, "mean_action_deviation")
    print(f"\nMOST INFLUENTIAL features:")
    for _, row in high_impact.iterrows():
        print(f"  + {row['feature_group']} (MAD={row['mean_action_deviation']:.4f})")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature sensitivity analysis")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--timesteps", "-t", type=int, default=200_000)
    parser.add_argument("--return-weight", type=float, default=0.0,
                        help="Reward return_weight (default: 0.0 = pure DSR)")
    parser.add_argument("--exposure-penalty", type=float, default=0.0,
                        help="Reward exposure_penalty (default: 0.0)")
    args = parser.parse_args()

    run_sensitivity(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        train_ratio=args.train_ratio,
        timesteps=args.timesteps,
        return_weight=args.return_weight,
        exposure_penalty=args.exposure_penalty,
    )
