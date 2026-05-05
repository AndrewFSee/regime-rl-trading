"""
Feature importance analysis via permutation importance.

Trains a single Config F (Sortino DSR) agent, then measures the impact
of shuffling each feature on test-set Sharpe ratio.  Features where
shuffling causes a large Sharpe drop are important; features where
shuffling has no impact (or improves performance) are noise.

Also computes statistical signal metrics (correlation, mutual information)
with next-day returns for a quick pre-filter.

Usage:
    python -m tests.test_feature_importance                     # defaults
    python -m tests.test_feature_importance --timesteps 100000  # faster
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.environment.data_loader import DataLoader
from src.environment.features import FeatureEngineer, FEATURE_NAMES
from src.environment.trading_env import TradingEnv
from src.evaluation.backtester import Backtester


# ======================================================================
# Statistical signal analysis (no training required)
# ======================================================================

def _statistical_signal(train_data: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation and mutual-information of each feature with
    next-day returns.  Returns a DataFrame sorted by |correlation|."""
    fe = FeatureEngineer()
    feats = fe.compute(train_data)
    close = train_data["Close"].astype(float)
    fwd_ret = close.pct_change().shift(-1).iloc[:-1]  # next-day return
    feats = feats.iloc[:-1]  # align

    rows = []
    for col in FEATURE_NAMES:
        x = feats[col].values
        y = fwd_ret.values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 30:
            rows.append({"feature": col, "corr": 0.0, "abs_corr": 0.0, "mi": 0.0})
            continue

        corr = float(np.corrcoef(x, y)[0, 1])

        # Mutual information via discretisation (10 bins)
        try:
            from sklearn.metrics import mutual_info_score
            x_bin = pd.qcut(x, q=10, labels=False, duplicates="drop")
            y_bin = pd.qcut(y, q=10, labels=False, duplicates="drop")
            mi = mutual_info_score(x_bin, y_bin)
        except Exception:
            mi = 0.0

        rows.append({
            "feature": col,
            "corr": round(corr, 4),
            "abs_corr": round(abs(corr), 4),
            "mi": round(mi, 4),
        })

    df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return df


# ======================================================================
# Permutation importance
# ======================================================================

@dataclass
class PermutationResult:
    feature: str
    baseline_sharpe: float
    shuffled_sharpe: float
    sharpe_drop: float            # positive = feature is important
    baseline_return: float
    shuffled_return: float
    return_drop: float


def _evaluate_agent(agent, env: TradingEnv) -> dict:
    bt = Backtester(env, agent)
    return bt.run(n_episodes=1)


def _permutation_importance(
    agent,
    test_data: pd.DataFrame,
    n_repeats: int = 5,
) -> list[PermutationResult]:
    """Measure feature importance by shuffling each feature column in the
    pre-computed observation features and evaluating performance drop.

    Shuffles are done on the raw feature matrix inside TradingEnv, so the
    agent sees scrambled values for exactly one feature at a time.
    """
    # Baseline evaluation (all features intact)
    env_base = TradingEnv(
        test_data.reset_index(drop=True),
        lookback_window=20,
        normalize_obs=True,
        downside_only=True,
    )
    base_result = _evaluate_agent(agent, env_base)
    base_sharpe = base_result["sharpe_ratio"]
    base_return = base_result["total_return"]
    print(f"\n  Baseline: Return={base_return:+.1%}, Sharpe={base_sharpe:.3f}")

    results = []
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        shuffled_sharpes = []
        shuffled_returns = []

        for rep in range(n_repeats):
            env = TradingEnv(
                test_data.reset_index(drop=True),
                lookback_window=20,
                normalize_obs=True,
                downside_only=True,
            )
            # Shuffle this feature column in both _features and _obs_features
            rng = np.random.RandomState(42 + rep)
            perm = rng.permutation(len(env._features))
            # Make writable copies before mutating
            env._features = env._features.copy()
            env._obs_features = env._obs_features.copy()
            env._features[:, feat_idx] = env._features[perm, feat_idx]
            # Also shuffle in the obs subset if this feature is in it
            if feat_idx < env._obs_features.shape[1]:
                obs_idx = None
                for oi, fi in enumerate(env._obs_feature_indices):
                    if fi == feat_idx:
                        obs_idx = oi
                        break
                if obs_idx is not None:
                    env._obs_features[:, obs_idx] = env._obs_features[perm, obs_idx]

            r = _evaluate_agent(agent, env)
            shuffled_sharpes.append(r["sharpe_ratio"])
            shuffled_returns.append(r["total_return"])

        avg_sharpe = float(np.mean(shuffled_sharpes))
        avg_return = float(np.mean(shuffled_returns))

        results.append(PermutationResult(
            feature=feat_name,
            baseline_sharpe=base_sharpe,
            shuffled_sharpe=avg_sharpe,
            sharpe_drop=base_sharpe - avg_sharpe,
            baseline_return=base_return,
            shuffled_return=avg_return,
            return_drop=base_return - avg_return,
        ))

        direction = "▼" if results[-1].sharpe_drop > 0.01 else ("▲" if results[-1].sharpe_drop < -0.01 else "—")
        print(f"  [{feat_idx:2d}] {feat_name:<18s}  Sharpe: {avg_sharpe:.3f} (Δ={results[-1].sharpe_drop:+.3f}) {direction}  "
              f"Return: {avg_return:+.1%} (Δ={results[-1].return_drop:+.1%})")

    return results


# ======================================================================
# Main
# ======================================================================

def run_feature_importance(
    ticker: str = "SPY",
    start: str = "2007-01-01",
    end: str = "2025-01-01",
    train_ratio: float = 0.7,
    timesteps: int = 200_000,
    n_repeats: int = 5,
) -> None:
    print("=" * 90)
    print(f"FEATURE IMPORTANCE ANALYSIS — {ticker} — {timesteps} timesteps")
    print("=" * 90)

    # Fetch data
    print(f"\nFetching {ticker} ({start} → {end}) ...")
    loader = DataLoader(tickers=[ticker], start_date=start, end_date=end, train_ratio=train_ratio)
    loader.fetch_data()
    train, test = loader.get_train_test_split(ticker)
    print(f"Train: {len(train)} bars | Test: {len(test)} bars")
    print(f"Features: {len(FEATURE_NAMES)} — {FEATURE_NAMES}")

    # ------------------------------------------------------------------
    # Phase 1: Statistical signal
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 1: Statistical signal (correlation & MI with next-day returns)")
    print(f"{'='*70}")
    stat_df = _statistical_signal(train)
    print(stat_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Phase 2: Train agent
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 2: Training Config F (Sortino DSR) agent ...")
    print(f"{'='*70}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        train_env = TradingEnv(
            train.reset_index(drop=True),
            lookback_window=20,
            normalize_obs=True,
            downside_only=True,
        )

        from src.agents.sac_agent import SACAgent
        agent = SACAgent(train_env, config={
            "learning_rate": 3e-4,
            "train_freq": 4,
        })
        t0 = time.perf_counter()
        agent.learn(total_timesteps=timesteps)
        train_time = time.perf_counter() - t0
        print(f"  Training complete in {train_time:.0f}s")

    # ------------------------------------------------------------------
    # Phase 3: Permutation importance
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"PHASE 3: Permutation importance ({n_repeats} repeats per feature)")
    print(f"{'='*70}")

    perm_results = _permutation_importance(agent, test, n_repeats=n_repeats)

    # Sort by Sharpe drop (most important first)
    perm_results.sort(key=lambda r: r.sharpe_drop, reverse=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*90}")
    print("FEATURE RANKING (sorted by Sharpe drop when shuffled)")
    print(f"{'='*90}")
    print(f"{'Rank':<6} {'Feature':<20} {'Sharpe Drop':>12} {'Return Drop':>12} {'Verdict':>10}")
    print("-" * 62)

    important = []
    noise = []
    for i, r in enumerate(perm_results):
        if r.sharpe_drop > 0.02:
            verdict = "SIGNAL"
            important.append(r.feature)
        elif r.sharpe_drop < -0.02:
            verdict = "HARMFUL"
            noise.append(r.feature)
        else:
            verdict = "noise"
            noise.append(r.feature)
        print(f"{i+1:<6} {r.feature:<20} {r.sharpe_drop:>+12.3f} {r.return_drop:>+12.1%} {verdict:>10}")

    print(f"\n{'='*90}")
    print(f"SIGNAL features ({len(important)}): {important}")
    print(f"Noise/harmful features ({len(noise)}): {noise}")
    print(f"\nRecommended obs_features list:")
    print(f"  {important}")
    print(f"{'='*90}")

    # Also merge with statistical signal for a combined view
    print(f"\n{'='*90}")
    print("COMBINED RANKING (permutation importance + statistical signal)")
    print(f"{'='*90}")
    stat_dict = {row["feature"]: row for _, row in stat_df.iterrows()}
    print(f"{'Feature':<20} {'Sharpe Drop':>12} {'|Corr|':>8} {'MI':>8}")
    print("-" * 50)
    for r in perm_results:
        s = stat_dict.get(r.feature, {"abs_corr": 0, "mi": 0})
        print(f"{r.feature:<20} {r.sharpe_drop:>+12.3f} {s['abs_corr']:>8.4f} {s['mi']:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature importance analysis")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--start", type=str, default="2007-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--timesteps", "-t", type=int, default=200_000)
    parser.add_argument("--repeats", "-r", type=int, default=5,
                        help="Number of shuffle repeats per feature (default: 5)")
    args = parser.parse_args()

    run_feature_importance(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        train_ratio=args.train_ratio,
        timesteps=args.timesteps,
        n_repeats=args.repeats,
    )
