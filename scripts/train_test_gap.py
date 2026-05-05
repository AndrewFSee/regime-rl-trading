"""
Train one agent, then evaluate on BOTH train and test splits.

Reveals overfit: if train metrics >> test metrics, the policy memorised
patterns that don't generalise. If both are bad, the policy never learned
anything useful (under-fit / wrong features / wrong reward).

Usage
-----
    python scripts/train_test_gap.py --agent sac --episodes 50 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scripts.run_experiment import (
    _build_agent,
    _build_data,
    _evaluate_single_asset,
    _load_config,
    _make_env,
    _set_global_seeds,
)


def _fmt(r: dict) -> str:
    return (
        f"return={r['total_return'] * 100:+7.2f}%  "
        f"sharpe={r['sharpe_ratio']:+5.2f}  "
        f"sortino={r['sortino_ratio']:+5.2f}  "
        f"maxDD={r['max_drawdown'] * 100:5.2f}%"
    )


def main(
    config_path: str = "config/default.yaml",
    ticker: str = "SPY",
    seed: int = 0,
    episodes: int = 50,
    agent_type: str = "sac",
    return_weight: float | None = 0.0,
    net_arch: list[int] | None = None,
    use_macro: bool = False,
) -> None:
    cfg = _load_config(config_path)
    cfg["agent"]["training_episodes"] = episodes
    cfg["agent"]["type"] = agent_type
    cfg["agent"]["verbose"] = 0
    if return_weight is not None:
        cfg.setdefault("environment", {})["return_weight"] = float(return_weight)
    if net_arch is not None:
        cfg["agent"]["net_arch"] = list(net_arch)
    cfg["_action_repeat"] = 1
    total_timesteps = episodes * 1000

    print(f"[gap] Agent          : {agent_type.upper()}")
    print(f"[gap] Seed           : {seed}")
    print(f"[gap] Timesteps      : {total_timesteps:,}")
    print(f"[gap] return_weight  : {cfg['environment'].get('return_weight', 0.0)}")
    print(f"[gap] net_arch       : {cfg['agent'].get('net_arch', '[256, 256] (default)')}")
    print(f"[gap] macro features : {use_macro}")
    print(f"[gap] Ticker         : {ticker}")

    print("[gap] Fetching data ...")
    train_data, test_data, _ = _build_data(cfg, multi_asset=False, ticker=ticker)
    print(f"[gap] train rows: {len(train_data)}, test rows: {len(test_data)}")

    if use_macro:
        from src.environment.data_loader import DataLoader
        from src.environment.macro_loader import MacroLoader
        print("[gap] Fetching macro panel ...")
        macro_dl = DataLoader(
            tickers=[ticker],
            start_date=cfg["data"]["start_date"],
            end_date=cfg["data"]["end_date"],
            train_ratio=cfg["data"]["train_ratio"],
            interval=cfg["data"]["interval"],
        )
        # Need OHLCV first so we know the date index to align macro to.
        macro_dl.fetch_data()
        macro_dl.fetch_macro()
        macro_raw = macro_dl.get_macro_aligned(ticker)
        macro_feats = MacroLoader.compute_features(macro_raw)
        cfg["_macro_panel"] = macro_feats
        print(f"[gap] macro feats    : shape={macro_feats.shape}")

    _set_global_seeds(seed)
    train_env = _make_env(cfg, train_data, multi_asset=False)
    agent, _ = _build_agent(train_env, cfg["agent"], seed=seed)

    print("[gap] Training ...")
    t0 = time.perf_counter()
    agent.learn(total_timesteps=total_timesteps)
    print(f"[gap] Trained in {time.perf_counter() - t0:.1f}s")

    # ----- evaluate on train split -----
    eval_train_env = _make_env(cfg, train_data, multi_asset=False)
    agent.env = eval_train_env
    train_r = _evaluate_single_asset(eval_train_env, agent)
    train_bh = train_r.get("benchmark_return", float("nan"))
    train_bh_sharpe = train_r.get("benchmark_sharpe_ratio", float("nan"))

    # ----- evaluate on test split -----
    eval_test_env = _make_env(cfg, test_data, multi_asset=False)
    agent.env = eval_test_env
    test_r = _evaluate_single_asset(eval_test_env, agent)
    test_bh = test_r.get("benchmark_return", float("nan"))
    test_bh_sharpe = test_r.get("benchmark_sharpe_ratio", float("nan"))

    print("\n" + "=" * 70)
    print(f"  TRAIN/TEST GAP REPORT  (seed={seed})")
    print("=" * 70)
    print(f"  TRAIN agent : {_fmt(train_r)}")
    print(f"  TRAIN B&H   : return={train_bh * 100:+7.2f}%  sharpe={train_bh_sharpe:+5.2f}")
    print(f"  TEST  agent : {_fmt(test_r)}")
    print(f"  TEST  B&H   : return={test_bh * 100:+7.2f}%  sharpe={test_bh_sharpe:+5.2f}")
    print()
    print(f"  Sharpe gap  : train {train_r['sharpe_ratio']:+.2f}  ->  "
          f"test {test_r['sharpe_ratio']:+.2f}   "
          f"(delta {train_r['sharpe_ratio'] - test_r['sharpe_ratio']:+.2f})")
    print(f"  Return gap  : train {train_r['total_return'] * 100:+.2f}%  ->  "
          f"test {test_r['total_return'] * 100:+.2f}%   "
          f"(delta {(train_r['total_return'] - test_r['total_return']) * 100:+.2f} pp)")
    print()
    print("  Diagnosis:")
    if train_r["sharpe_ratio"] > 0.5 and test_r["sharpe_ratio"] < 0.0:
        print("    >>> OVERFIT: policy works on train, fails on test. "
              "Reduce capacity / regularize / less training / walk-forward CV.")
    elif train_r["sharpe_ratio"] < 0.2 and test_r["sharpe_ratio"] < 0.2:
        print("    >>> UNDER-FIT or WRONG SIGNAL: policy fails on train too. "
              "Check features, reward shaping, or train longer.")
    else:
        print("    >>> MIXED. Inspect manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate on both splits.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--agent", default="sac")
    parser.add_argument("--return-weight", type=float, default=0.0)
    parser.add_argument("--net-arch", type=str, default=None,
                        help="Comma-separated MLP widths, e.g. '64,64'. Default: SB3's [256,256].")
    parser.add_argument("--macro", action="store_true",
                        help="Add macro / cross-asset features to the obs vector.")
    args = parser.parse_args()
    net_arch = None
    if args.net_arch:
        net_arch = [int(x.strip()) for x in args.net_arch.split(",") if x.strip()]
    main(
        config_path=args.config,
        ticker=args.ticker,
        seed=args.seed,
        episodes=args.episodes,
        agent_type=args.agent,
        return_weight=args.return_weight,
        net_arch=net_arch,
        use_macro=args.macro,
    )
