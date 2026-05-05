"""Diagnostic: distribution of each strategy's raw signal over a date range.

Runs the 4 hand-coded strategies on the base 10 features (the same vector the
TradingEnv passes to them) and prints summary stats: mean, std, fraction of
days each action bucket fires, and correlation across strategies.

Usage
-----
    python -u scripts/inspect_strategy_signals.py --ticker SPY \
        --start 2014-01-01 --end 2024-04-01
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.environment.data_loader import DataLoader
from src.environment.features import FeatureEngineer, FEATURE_NAMES
from src.strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    TrendFollowingStrategy,
)


def _bucket(a: float) -> str:
    if a > 0.5:
        return "long"
    if a < -0.5:
        return "short"
    if a > 0.05:
        return "weak_long"
    if a < -0.05:
        return "weak_short"
    return "flat"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end", default="2024-04-01")
    args = p.parse_args()

    dl = DataLoader([args.ticker], args.start, args.end)
    dl.fetch_data()
    data = dl.get_data(args.ticker).reset_index(drop=True)
    feats_df = FeatureEngineer().compute(data)
    feats = feats_df.values  # (T, 10)
    print(f"[inspect] data: {len(data)} rows; features: {feats.shape}")
    print(f"[inspect] FEATURE_NAMES = {FEATURE_NAMES}\n")

    strategies = [
        MomentumStrategy(),
        MeanReversionStrategy(),
        BreakoutStrategy(),
        TrendFollowingStrategy(),
    ]

    # Skip the first 30 rows (warmup for long_ma) to avoid NaN-like artifacts.
    signal_rows = []
    for i in range(30, len(feats)):
        row = feats[i]
        signal_rows.append([s.generate_signal(row).action for s in strategies])
    sig = np.array(signal_rows)  # (T-30, 4)
    names = [s.name for s in strategies]

    print("=== Per-strategy signal distribution ===")
    print(f"{'strategy':<16} {'mean':>7} {'std':>6} {'min':>6} {'max':>6}   "
          "long%  wL%   flat%  wS%  short%")
    for j, n in enumerate(names):
        col = sig[:, j]
        buckets = pd.Series([_bucket(v) for v in col]).value_counts(normalize=True)
        print(
            f"{n:<16} {col.mean():+7.3f} {col.std():6.3f} "
            f"{col.min():+6.2f} {col.max():+6.2f}   "
            f"{100*buckets.get('long',0):5.1f} "
            f"{100*buckets.get('weak_long',0):5.1f} "
            f"{100*buckets.get('flat',0):5.1f} "
            f"{100*buckets.get('weak_short',0):5.1f} "
            f"{100*buckets.get('short',0):5.1f}"
        )

    print("\n=== Pairwise signal correlation ===")
    df = pd.DataFrame(sig, columns=names)
    print(df.corr().round(2).to_string())

    # Useful: how often does Breakout actually do something?
    nz = (np.abs(sig) > 0.05).mean(axis=0) * 100
    print("\n=== Active-day % (|signal|>0.05) ===")
    for n, p in zip(names, nz):
        print(f"  {n:<16} {p:5.1f}%")

    # Forward 1-day return correlation: does each strategy's signal predict
    # tomorrow's return? (Quick edge check — needs to be positive to be useful.)
    fwd_ret = data["Close"].pct_change().shift(-1).iloc[30:].values
    fwd_ret = fwd_ret[: len(sig)]
    mask = np.isfinite(fwd_ret)
    print("\n=== Corr(signal_t, return_{t+1}) — needs to be >0 to add value ===")
    for j, n in enumerate(names):
        c = np.corrcoef(sig[mask, j], fwd_ret[mask])[0, 1]
        print(f"  {n:<16} corr = {c:+.4f}")


if __name__ == "__main__":
    main()
