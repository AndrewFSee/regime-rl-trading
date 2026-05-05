"""Diagnostic: trace agent positions vs market for selected walk-forward folds.

Trains SAC on the train slice of one fold, then runs a deterministic rollout
over the test slice, capturing per-bar:
    date, close, market_return, position, regime, chosen_strategy, exposure

Prints a per-fold summary of:
    - average position
    - % of bars flat (|position| < 0.05)
    - lag (in bars) from start of test window to first non-flat position
    - lag from first +ve market week to first long position
    - regime histogram

Outputs full traces to ``results/diag_fold_<n>_<label>.csv``.

Defaults match the weekly walk-forward run on SPY 2000-04 → 2024-04 with
train_size=156, test_size=52, step=52, embargo=2.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def _trace_fold(
    fold_idx: int,
    train_df,
    test_df,
    timesteps: int,
    seed: int,
    out_csv: str,
) -> dict:
    from src.environment.trading_env import TradingEnv
    from src.agents.sac_agent import SACAgent

    env_kwargs = dict(
        lookback_window=20,
        transaction_cost=0.001,
        slippage_bps=1.0,
        normalize_obs=True,
        return_weight=0.0,
    )
    train_env = TradingEnv(data=train_df, **env_kwargs)
    test_env = TradingEnv(data=test_df, **env_kwargs)

    agent = SACAgent(train_env, config={"seed": seed})
    agent.learn(total_timesteps=timesteps)

    # Transfer normalizer (matches the harness)
    state = train_env.get_normalizer_state()
    if state is not None:
        test_env.load_normalizer_state(state)
        test_env.set_training_mode(False)
    agent.env = test_env
    if hasattr(agent, "model"):
        agent.model.set_env(test_env)

    obs, _ = test_env.reset()
    rows = []
    done = False
    while not done:
        action, _ = agent.model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = test_env.step(action)
        done = term or trunc
        # ``test_env._step_idx`` was incremented inside step() — use it to look up
        # the bar that just closed.
        bar_idx = test_env._step_idx - 1
        if bar_idx >= len(test_df):
            break
        ts = test_df.index[bar_idx]
        rows.append({
            "date": ts,
            "close": float(test_df["Close"].iloc[bar_idx]),
            "market_return": info["market_return"],
            "position": info["position"],
            "regime": info["regime"].name if hasattr(info["regime"], "name") else str(info["regime"]),
            "chosen_strategy": int(info["chosen_strategy"]),
            "exposure": info["effective_exposure"],
            "drawdown": info["drawdown"],
        })

    trace = pd.DataFrame(rows).set_index("date")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    trace.to_csv(out_csv)

    # Summary stats
    n = len(trace)
    flat_mask = trace["position"].abs() < 0.05
    avg_pos = float(trace["position"].mean())
    pct_flat = float(flat_mask.mean()) * 100
    pct_long = float((trace["position"] > 0.05).mean()) * 100
    pct_short = float((trace["position"] < -0.05).mean()) * 100

    # Lag from start to first non-flat bar
    nonflat = np.where(~flat_mask.values)[0]
    lag_to_first_nonflat = int(nonflat[0]) if len(nonflat) else n

    # Lag from first +ve week to first long bar
    first_up = trace["market_return"].gt(0).idxmax() if (trace["market_return"] > 0).any() else None
    long_dates = trace.index[trace["position"] > 0.05]
    if first_up is not None and len(long_dates):
        # bars between first up week and first long bar (inclusive)
        idx_first_up = trace.index.get_loc(first_up)
        idx_first_long = trace.index.get_loc(long_dates[0])
        lag_first_up_to_long = int(idx_first_long - idx_first_up)
    else:
        lag_first_up_to_long = None

    regime_hist = trace["regime"].value_counts(normalize=True).mul(100).round(1).to_dict()
    strat_hist = trace["chosen_strategy"].value_counts(normalize=True).mul(100).round(1).to_dict()

    bnh_ret = (1 + trace["market_return"]).prod() - 1
    agent_ret = (trace["position"].shift(1).fillna(0.0) * trace["market_return"]).add(1).prod() - 1

    return {
        "fold": fold_idx,
        "test_start": str(trace.index[0].date()),
        "test_end": str(trace.index[-1].date()),
        "n_bars": n,
        "agent_return": float(agent_ret),
        "bnh_return": float(bnh_ret),
        "avg_position": avg_pos,
        "pct_long": pct_long,
        "pct_flat": pct_flat,
        "pct_short": pct_short,
        "lag_to_first_nonflat": lag_to_first_nonflat,
        "lag_first_up_to_long": lag_first_up_to_long,
        "regime_pct": regime_hist,
        "strategy_pct": strat_hist,
        "trace_csv": out_csv,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2024-04-01")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--interval", default="1wk")
    p.add_argument("--train-size", type=int, default=156)
    p.add_argument("--test-size", type=int, default=52)
    p.add_argument("--step", type=int, default=52)
    p.add_argument("--embargo", type=int, default=2)
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    # Default folds: 5 (2008 GFC — agent worked), 0 (2003), 6 (2009), 7 (2010), 18 (2020 Covid)
    p.add_argument("--folds", default="0,5,6,7,18",
                   help="Comma-separated fold indices to diagnose")
    args = p.parse_args()

    from src.environment.data_loader import DataLoader
    loader = DataLoader(
        tickers=[args.ticker],
        start_date=args.start,
        end_date=args.end,
        train_ratio=0.7,
        interval=args.interval,
    )
    print(f"[diag] Fetching {args.ticker} {args.start} -> {args.end} interval={args.interval}")
    loader.fetch_data()
    splits = loader.get_walk_forward_splits(
        args.ticker,
        train_size=args.train_size,
        test_size=args.test_size,
        step=args.step,
        embargo=args.embargo,
    )
    print(f"[diag] {len(splits)} folds available")

    target = [int(x) for x in args.folds.split(",")]
    summaries = []
    for fold_idx in target:
        if fold_idx >= len(splits):
            print(f"[diag] skipping fold {fold_idx}: only {len(splits)} available")
            continue
        train_df, test_df = splits[fold_idx]
        print(f"\n[diag] fold {fold_idx}: train {train_df.index[0].date()}..{train_df.index[-1].date()}  "
              f"test {test_df.index[0].date()}..{test_df.index[-1].date()}")
        out_csv = f"results/diag_fold_{fold_idx:02d}_{args.interval}.csv"
        s = _trace_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            test_df=test_df,
            timesteps=args.timesteps,
            seed=args.seed + fold_idx,
            out_csv=out_csv,
        )
        summaries.append(s)
        print(
            f"  agent {s['agent_return']*100:+6.2f}%  vs B&H {s['bnh_return']*100:+6.2f}%  "
            f"avg_pos={s['avg_position']:+.2f}  long%={s['pct_long']:5.1f}  flat%={s['pct_flat']:5.1f}  short%={s['pct_short']:5.1f}"
        )
        print(f"  lag-to-first-nonflat: {s['lag_to_first_nonflat']} bars   "
              f"lag-first-up-to-long: {s['lag_first_up_to_long']} bars")
        print(f"  regime%: {s['regime_pct']}")
        print(f"  strategy% (0=mom 1=mr 2=brk 3=trend): {s['strategy_pct']}")
        print(f"  trace -> {out_csv}")

    print("\n[diag] summary across folds")
    print(f"{'fold':>4} {'period':<23} {'agent%':>7} {'B&H%':>7} {'avg_pos':>8} {'long%':>6} {'flat%':>6} {'short%':>7} {'lag1st':>7}")
    for s in summaries:
        print(
            f"{s['fold']:>4} {s['test_start']}..{s['test_end']} "
            f"{s['agent_return']*100:+7.2f} {s['bnh_return']*100:+7.2f} "
            f"{s['avg_position']:+8.2f} {s['pct_long']:6.1f} {s['pct_flat']:6.1f} {s['pct_short']:7.1f} "
            f"{s['lag_to_first_nonflat']:>7}"
        )


if __name__ == "__main__":
    main()
