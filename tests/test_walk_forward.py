"""Walk-forward evaluation harness.

Trains an agent on rolling training windows and evaluates on the immediately
following out-of-sample window. Produces per-fold metrics plus aggregate stats
(mean / std across folds), giving statistically more credible numbers than a
single contiguous 70/30 split.

Usage
-----
    python -m tests.test_walk_forward \
        --start 2007-01-01 --end 2025-01-01 \
        --train-size 1260 --test-size 252 --step 252 --embargo 5 \
        --timesteps 50000 --agent sac

Notes
-----
* ``train-size`` / ``test-size`` are in trading-day bars (~252/yr).
* The script uses the ``DataLoader.get_walk_forward_splits`` helper.
* Aggregates Sharpe / total return / MaxDD across folds with bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    benchmark_return: float
    benchmark_sharpe: float
    benchmark_max_drawdown: float


def _train_and_eval_fold(
    fold: int,
    train_df,
    test_df,
    timesteps: int,
    agent_type: str,
    seed: int,
    env_kwargs: dict,
    macro_panel=None,
) -> FoldResult:
    from src.environment.trading_env import TradingEnv
    from src.evaluation.backtester import Backtester

    train_kw = dict(env_kwargs)
    test_kw = dict(env_kwargs)
    if macro_panel is not None:
        train_kw["macro_features"] = macro_panel.reindex(train_df.index).ffill().bfill().fillna(0.0)
        test_kw["macro_features"] = macro_panel.reindex(test_df.index).ffill().bfill().fillna(0.0)

    train_env = TradingEnv(data=train_df, **train_kw)
    test_env = TradingEnv(data=test_df, **test_kw)

    if agent_type == "sac":
        from src.agents.sac_agent import SACAgent
        agent = SACAgent(train_env, config={"seed": seed})
    elif agent_type == "ppo":
        from src.agents.ppo_agent import PPOAgent
        agent = PPOAgent(train_env, config={"seed": seed})
    else:
        raise ValueError(f"Unsupported agent: {agent_type}")

    agent.learn(total_timesteps=timesteps)

    # Transfer normalizer state so eval matches the training distribution and
    # freeze updates (so test stats don't drift mid-episode).
    train_state = train_env.get_normalizer_state()
    if train_state is not None:
        test_env.load_normalizer_state(train_state)
        test_env.set_training_mode(False)

    # Re-bind agent to the test env for backtesting
    agent.env = test_env
    if hasattr(agent, "model"):
        agent.model.set_env(test_env)

    bt = Backtester(test_env, agent)
    res = bt.run(n_episodes=1)

    return FoldResult(
        fold=fold,
        train_start=str(train_df.index[0]) if hasattr(train_df, "index") else str(0),
        train_end=str(train_df.index[-1]) if hasattr(train_df, "index") else str(len(train_df)),
        test_start=str(test_df.index[0]) if hasattr(test_df, "index") else str(0),
        test_end=str(test_df.index[-1]) if hasattr(test_df, "index") else str(len(test_df)),
        total_return=res["total_return"],
        sharpe_ratio=res["sharpe_ratio"],
        sortino_ratio=res["sortino_ratio"],
        max_drawdown=res["max_drawdown"],
        benchmark_return=res["benchmark_return"],
        benchmark_sharpe=res["benchmark_sharpe_ratio"],
        benchmark_max_drawdown=res["benchmark_max_drawdown"],
    )


def _summarise(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation")
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--train-size", type=int, default=1260, help="bars per train window")
    parser.add_argument("--test-size", type=int, default=252, help="bars per test window")
    parser.add_argument("--step", type=int, default=252, help="stride between fold starts")
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--agent", default="sac", choices=("sac", "ppo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    parser.add_argument("--return-weight", type=float, default=0.0,
                        help="DSR/return blend (0=pure DSR, 1=pure net return).")
    parser.add_argument("--macro", action="store_true",
                        help="Add macro / cross-asset features to the obs vector.")
    parser.add_argument("--selective-macro", action="store_true",
                        help="Only expose macro features in BEAR/VOLATILE regimes; "
                             "zero them out otherwise. Requires --macro.")
    parser.add_argument("--hard-selection", action="store_true",
                        help="Force argmax single-strategy selection per step "
                             "instead of the default softmax blend.")
    parser.add_argument("--interval", default="1d",
                        help="yfinance bar interval (e.g. '1d', '1wk').")
    args = parser.parse_args()

    from src.environment.data_loader import DataLoader

    loader = DataLoader(
        tickers=[args.ticker],
        start_date=args.start,
        end_date=args.end,
        train_ratio=0.7,
        interval=args.interval,
    )
    print(f"[walk-forward] Fetching {args.ticker} {args.start} -> {args.end} ...")
    loader.fetch_data()
    splits = loader.get_walk_forward_splits(
        args.ticker,
        train_size=args.train_size,
        test_size=args.test_size,
        step=args.step,
        embargo=args.embargo,
    )
    if args.max_folds:
        splits = splits[: args.max_folds]
    print(f"[walk-forward] {len(splits)} folds  "
          f"(train={args.train_size}, test={args.test_size}, step={args.step})")

    macro_panel = None
    if args.macro:
        from src.environment.macro_loader import MacroLoader
        print("[walk-forward] Fetching macro panel ...")
        loader.fetch_macro()
        macro_raw = loader.get_macro_aligned(args.ticker)
        macro_panel = MacroLoader.compute_features(macro_raw)
        print(f"[walk-forward] macro feats: shape={macro_panel.shape}")

    env_kwargs: dict = {
        "lookback_window": 20,
        "transaction_cost": 0.001,
        "slippage_bps": 1.0,
        "normalize_obs": True,
        "return_weight": float(args.return_weight),
        "hard_selection": bool(args.hard_selection),
        "selective_macro": bool(args.selective_macro),
    }

    results: list[FoldResult] = []
    t0 = time.perf_counter()
    for i, (train_df, test_df) in enumerate(splits):
        fold_t0 = time.perf_counter()
        res = _train_and_eval_fold(
            fold=i,
            train_df=train_df,
            test_df=test_df,
            timesteps=args.timesteps,
            agent_type=args.agent,
            seed=args.seed + i,
            env_kwargs=env_kwargs,
            macro_panel=macro_panel,
        )
        results.append(res)
        dt = time.perf_counter() - fold_t0
        print(
            f"  Fold {i:2d}: ret={res.total_return*100:+6.2f}% "
            f"(B&H {res.benchmark_return*100:+6.2f}%)  "
            f"Sharpe={res.sharpe_ratio:5.2f} (B&H {res.benchmark_sharpe:5.2f})  "
            f"MaxDD={res.max_drawdown*100:5.2f}%  "
            f"[{dt:5.1f}s]"
        )

    elapsed = time.perf_counter() - t0
    print(f"\n[walk-forward] {len(results)} folds completed in {elapsed:.1f}s\n")

    print("Aggregate (across folds)")
    print("-" * 50)
    for label, key in [
        ("Total Return    ", "total_return"),
        ("Sharpe Ratio    ", "sharpe_ratio"),
        ("Sortino Ratio   ", "sortino_ratio"),
        ("Max Drawdown    ", "max_drawdown"),
        ("Benchmark Return", "benchmark_return"),
        ("Benchmark Sharpe", "benchmark_sharpe"),
    ]:
        s = _summarise([getattr(r, key) for r in results])
        print(f"  {label} : mean={s['mean']:+8.4f}  std={s['std']:7.4f}  "
              f"median={s['median']:+8.4f}  n={s['n']}")
    print("-" * 50)

    # Excess return / Sharpe vs B&H per fold
    excess_returns = [r.total_return - r.benchmark_return for r in results]
    excess_sharpes = [r.sharpe_ratio - r.benchmark_sharpe for r in results]
    s_er = _summarise(excess_returns)
    s_es = _summarise(excess_sharpes)
    print(f"  Agent − B&H Ret  : mean={s_er['mean']:+8.4f}  std={s_er['std']:7.4f}")
    print(f"  Agent − B&H Sharpe: mean={s_es['mean']:+8.4f}  std={s_es['std']:7.4f}")
    win_rate = float(np.mean([1.0 if x > 0 else 0.0 for x in excess_returns]))
    print(f"  Folds beating B&H: {win_rate*100:.1f}% ({sum(1 for x in excess_returns if x > 0)}/{len(excess_returns)})")
    print("=" * 50)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(
                {
                    "args": vars(args),
                    "folds": [vars(r) for r in results],
                    "aggregate": {
                        k: _summarise([getattr(r, k) for r in results])
                        for k in (
                            "total_return", "sharpe_ratio", "sortino_ratio",
                            "max_drawdown", "benchmark_return", "benchmark_sharpe",
                        )
                    },
                    "excess_return": s_er,
                    "excess_sharpe": s_es,
                    "fold_win_rate": win_rate,
                },
                fh,
                indent=2,
                default=str,
            )
        print(f"[walk-forward] wrote {args.output}")


if __name__ == "__main__":
    main()
