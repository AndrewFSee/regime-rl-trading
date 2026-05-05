"""
Train one SAC seed and inspect its policy on the test split.

Reports:
  * Headline metrics (total return, sharpe, max DD).
  * Action / position distribution (mean, std, percentiles, % near-zero, % long, % short).
  * Per-decile bucketed counts.
  * Largest single-step action change.

Used to test the "stuck flat" / "stuck long" / "thrashing" hypotheses about
why the trained policy underperforms buy-and-hold.

Usage
-----
    python scripts/inspect_policy.py --episodes 50 --return-weight 0.0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np

from scripts.run_experiment import (
    _build_agent,
    _build_data,
    _load_config,
    _make_env,
    _set_global_seeds,
)


def _summarise_actions(actions: np.ndarray, positions: np.ndarray) -> None:
    """Print distribution of agent actions and resulting portfolio positions."""
    print("\n  Action distribution (raw policy output)")
    print("  " + "-" * 60)
    a = actions.flatten()
    print(f"    n               : {len(a)}")
    print(f"    mean            : {a.mean():+.4f}")
    print(f"    std             : {a.std():.4f}")
    print(f"    min / max       : {a.min():+.4f} / {a.max():+.4f}")
    pcts = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    print(f"    percentiles 1/5/25/50/75/95/99:")
    print(f"      {pcts[0]:+.3f} {pcts[1]:+.3f} {pcts[2]:+.3f} "
          f"{pcts[3]:+.3f} {pcts[4]:+.3f} {pcts[5]:+.3f} {pcts[6]:+.3f}")

    print("\n  Position distribution (effective exposure on test data)")
    print("  " + "-" * 60)
    p = positions.flatten()
    print(f"    n               : {len(p)}")
    print(f"    mean            : {p.mean():+.4f}")
    print(f"    std             : {p.std():.4f}")
    print(f"    min / max       : {p.min():+.4f} / {p.max():+.4f}")

    near_zero = np.mean(np.abs(p) < 0.05) * 100
    long_pct = np.mean(p > 0.05) * 100
    short_pct = np.mean(p < -0.05) * 100
    full_long = np.mean(p > 0.95) * 100
    full_short = np.mean(p < -0.95) * 100
    print(f"    %% near flat (|p|<0.05) : {near_zero:5.1f}%")
    print(f"    %% long  (p > 0.05)     : {long_pct:5.1f}%")
    print(f"    %% short (p < -0.05)    : {short_pct:5.1f}%")
    print(f"    %% near full long       : {full_long:5.1f}%")
    print(f"    %% near full short      : {full_short:5.1f}%")

    bucket_edges = np.linspace(-1.0, 1.0, 11)
    counts, _ = np.histogram(p, bins=bucket_edges)
    print("\n  Position histogram (decile buckets, % of steps)")
    print("  " + "-" * 60)
    for i, c in enumerate(counts):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        pct = c / len(p) * 100
        bar = "#" * int(pct / 2)
        print(f"    [{lo:+.1f},{hi:+.1f}) {pct:5.1f}%  {bar}")

    if len(p) > 1:
        deltas = np.diff(p)
        print("\n  Position change between steps")
        print("  " + "-" * 60)
        print(f"    mean |delta_p|  : {np.mean(np.abs(deltas)):.4f}")
        print(f"    max  |delta_p|  : {np.max(np.abs(deltas)):.4f}")
        print(f"    %% no-change      : {np.mean(np.abs(deltas) < 1e-4) * 100:5.1f}%")


def main(
    config_path: str = "config/default.yaml",
    ticker: str = "SPY",
    seed: int = 0,
    episodes: int = 50,
    agent_type: str = "sac",
    return_weight: float | None = None,
) -> None:
    cfg = _load_config(config_path)
    cfg["agent"]["training_episodes"] = episodes
    cfg["agent"]["type"] = agent_type
    cfg["agent"]["verbose"] = 0
    if return_weight is not None:
        cfg.setdefault("environment", {})["return_weight"] = float(return_weight)
    total_timesteps = episodes * 1000

    print(f"[inspect] Agent          : {agent_type.upper()}")
    print(f"[inspect] Seed           : {seed}")
    print(f"[inspect] Timesteps      : {total_timesteps:,}")
    print(f"[inspect] return_weight  : {cfg.get('environment', {}).get('return_weight', 0.0)}")
    print(f"[inspect] Ticker         : {ticker}")

    print("[inspect] Fetching data ...")
    train_data, test_data, _ = _build_data(cfg, multi_asset=False, ticker=ticker)

    _set_global_seeds(seed)
    train_env = _make_env(cfg, train_data, multi_asset=False)
    agent, _ = _build_agent(train_env, cfg["agent"], seed=seed)

    print("[inspect] Training ...")
    t0 = time.perf_counter()
    agent.learn(total_timesteps=total_timesteps)
    print(f"[inspect] Trained in {time.perf_counter() - t0:.1f}s")

    test_env = _make_env(cfg, test_data, multi_asset=False)
    agent.env = test_env

    # Deterministic rollout collecting actions and positions.
    obs, _ = test_env.reset()
    actions: list = []
    positions: list = []
    pv = [test_env.initial_cash]
    rets: list = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(obs)
        actions.append(np.asarray(action, dtype=float).flatten())
        obs, _r, terminated, truncated, info = test_env.step(action)
        pv.append(float(info.get("portfolio_value", pv[-1])))
        rets.append(float(info.get("net_return", 0.0)))
        # Position recorded by the env after the action is applied.
        if hasattr(test_env, "_position"):
            positions.append(float(test_env._position))
        elif hasattr(test_env, "position"):
            positions.append(float(test_env.position))
        else:
            positions.append(float(np.asarray(action, dtype=float).flatten()[0]))

    actions_arr = np.stack(actions, axis=0)
    positions_arr = np.asarray(positions, dtype=float)

    total_return = pv[-1] / pv[0] - 1.0
    rets_np = np.asarray(rets, dtype=float)
    if rets_np.std() > 1e-12:
        sharpe = float(rets_np.mean() / rets_np.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    peak = np.maximum.accumulate(pv)
    max_dd = float(((np.asarray(pv) - peak) / peak).min())

    print("\n  Headline (test split)")
    print("  " + "-" * 60)
    print(f"    total_return    : {total_return * 100:+.2f}%")
    print(f"    sharpe          : {sharpe:+.3f}")
    print(f"    max_drawdown    : {max_dd * 100:+.2f}%")

    _summarise_actions(actions_arr, positions_arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a trained policy's actions on test data.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50,
                        help="training_episodes (1 episode = 1000 steps).")
    parser.add_argument("--agent", default="sac")
    parser.add_argument("--return-weight", type=float, default=None)
    args = parser.parse_args()
    main(
        config_path=args.config,
        ticker=args.ticker,
        seed=args.seed,
        episodes=args.episodes,
        agent_type=args.agent,
        return_weight=args.return_weight,
    )
