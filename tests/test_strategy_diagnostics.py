"""
Strategy diagnostics: trains an agent and analyzes its behavior.

Shows:
    1. What strategy signals the 4 strategies actually output over the test period
    2. What strategy weights or hard picks the agent uses
    3. What exposure level the agent chooses after any risk-off overrides
    4. How the final position compares to a simple "always long" approach

Usage:
        python -m tests.test_strategy_diagnostics
        python -m tests.test_strategy_diagnostics --timesteps 500000 --hard-selection
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.environment.data_loader import DataLoader
from src.environment.trading_env import TradingEnv
from src.environment.features import FeatureEngineer, FEATURE_NAMES
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.defensive import DefensiveStrategy


def _load_checkpoint_metadata(
    checkpoint_path: str,
    checkpoint_metadata: str | None = None,
) -> dict:
    metadata_path = Path(checkpoint_metadata) if checkpoint_metadata else Path(checkpoint_path).with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Checkpoint metadata not found: {metadata_path}. "
            "Pass --checkpoint-metadata or rerun the benchmark with --save-checkpoints-dir."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def run_diagnostics(
    ticker: str = "SPY",
    start: str = "2007-01-01",
    end: str = "2025-01-01",
    train_ratio: float = 0.7,
    timesteps: int = 500_000,
    seed: int = 0,
    hard_selection: bool = False,
    regime_guardrail: bool = False,
    risk_off_exposure_cap: float | None = None,
    risk_off_strategy_penalty: float = 0.0,
    checkpoint_path: str | None = None,
    checkpoint_metadata: str | None = None,
):
    metadata = None
    if checkpoint_path:
        metadata = _load_checkpoint_metadata(checkpoint_path, checkpoint_metadata)
        ticker = metadata.get("ticker", ticker)
        start = metadata.get("start", start)
        end = metadata.get("end", end)
        train_ratio = metadata.get("train_ratio", train_ratio)
        timesteps = metadata.get("timesteps", timesteps)
        seed = metadata.get("seed", seed)
        saved_env_kwargs = metadata.get("env_kwargs", {})
        hard_selection = saved_env_kwargs.get("hard_selection", hard_selection)
        regime_guardrail = saved_env_kwargs.get("regime_guardrail", regime_guardrail)
        risk_off_exposure_cap = saved_env_kwargs.get("risk_off_exposure_cap", risk_off_exposure_cap)
        risk_off_strategy_penalty = saved_env_kwargs.get("risk_off_strategy_penalty", risk_off_strategy_penalty)

    print("=" * 90)
    print(f"STRATEGY DIAGNOSTICS - {ticker} - {timesteps} steps - seed={seed}")
    print("=" * 90)

    # Fetch data
    loader = DataLoader(tickers=[ticker], start_date=start, end_date=end, train_ratio=train_ratio)
    loader.fetch_data()
    train, test = loader.get_train_test_split(ticker)
    print(f"Train: {len(train)} bars | Test: {len(test)} bars")

    # =====================================================================
    # PART 1: Raw strategy signals over the test period (no agent needed)
    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 1: RAW STRATEGY SIGNALS ON TEST DATA")
    print("=" * 90)

    fe = FeatureEngineer()
    test_reset = test.reset_index(drop=True)
    features_df = fe.compute(test_reset)
    features_arr = features_df.values

    strategies = [
        MomentumStrategy(),
        MeanReversionStrategy(),
        BreakoutStrategy(),
        DefensiveStrategy(),
    ]

    n_steps = len(features_arr)
    signals = np.zeros((n_steps, 4))
    for t in range(n_steps):
        feat_dict = dict(zip(FEATURE_NAMES, features_arr[t]))
        for s_idx, strat in enumerate(strategies):
            sig = strat.generate_signal(feat_dict)
            signals[t, s_idx] = sig.action

    strat_names = ["Momentum", "MeanRevert", "Breakout", "Defensive"]

    # Summary stats
    print(f"\n{'Strategy':<15} {'Mean':>8} {'Std':>8} {'%Long':>8} {'%Short':>8} {'%Flat':>8}")
    print("-" * 55)
    for i, name in enumerate(strat_names):
        s = signals[:, i]
        pct_long = (s > 0.01).mean() * 100
        pct_short = (s < -0.01).mean() * 100
        pct_flat = ((s >= -0.01) & (s <= 0.01)).mean() * 100
        print(f"{name:<15} {s.mean():>8.3f} {s.std():>8.3f} {pct_long:>7.1f}% {pct_short:>7.1f}% {pct_flat:>7.1f}%")

    # Equal-weight blend (what EqualWeight agent sees)
    ew_blend = signals.mean(axis=1)
    ew_direction = np.tanh(ew_blend * 5.0)
    print(f"\n{'EW Blend':<15} {ew_blend.mean():>8.3f} {ew_blend.std():>8.3f} "
          f"{(ew_blend > 0.01).mean()*100:>7.1f}% {(ew_blend < -0.01).mean()*100:>7.1f}% "
          f"{((ew_blend >= -0.01) & (ew_blend <= 0.01)).mean()*100:>7.1f}%")
    print(f"{'EW Direction':<15} {ew_direction.mean():>8.3f} {ew_direction.std():>8.3f}")

    # Sub-period analysis of strategy signals
    test_dates = test_reset["Close"].index  # integer index
    test_close = test_reset["Close"].values.astype(float)

    # Define sub-periods by approximate bar index using dates from original test
    # We need dates - let's use the original test DataFrame
    if "Date" in test.columns:
        dates = pd.to_datetime(test["Date"]).values
    else:
        dates = pd.to_datetime(test.index).values

    periods = [
        ("COVID crash",      "2020-02-19", "2020-03-23"),
        ("COVID recovery",   "2020-03-23", "2020-08-18"),
        ("2022 bear",        "2022-01-03", "2022-10-12"),
        ("Late-2022 rebound","2022-10-12", "2022-12-30"),
        ("2023-24 bull",     "2023-01-03", "2024-12-31"),
    ]

    print(f"\n{'Period':<20} {'Momentum':>10} {'MeanRev':>10} {'Breakout':>10} {'Defensive':>10} {'EW Blend':>10}")
    print("-" * 70)
    for pname, pstart, pend in periods:
        mask = (dates >= np.datetime64(pstart)) & (dates <= np.datetime64(pend))
        if mask.sum() == 0:
            continue
        # Map to reset_index positions
        idx = np.where(mask)[0]
        valid_idx = idx[idx < n_steps]
        if len(valid_idx) == 0:
            continue
        period_signals = signals[valid_idx]
        period_ew = ew_blend[valid_idx]
        print(f"{pname:<20} {period_signals[:,0].mean():>10.3f} {period_signals[:,1].mean():>10.3f} "
              f"{period_signals[:,2].mean():>10.3f} {period_signals[:,3].mean():>10.3f} "
              f"{period_ew.mean():>10.3f}")

    # =====================================================================
    # PART 2: Train agent and analyze its behavior
    # =====================================================================
    config_bits = ["floor50", "trend0.1"]
    if hard_selection:
        config_bits.append("hard")
    if regime_guardrail:
        config_bits.append("guardrail")
    if risk_off_exposure_cap is not None:
        config_bits.append(f"cap{risk_off_exposure_cap:.2f}")
    if risk_off_strategy_penalty > 0:
        config_bits.append(f"pen{risk_off_strategy_penalty:.2f}")
    config_label = "+".join(config_bits)

    print("\n" + "=" * 90)
    print(f"PART 2: TRAINED AGENT BEHAVIOR ({config_label}, {timesteps} steps)")
    print("=" * 90)

    train_data = train.reset_index(drop=True)
    env_kwargs = dict(
        lookback_window=20,
        normalize_obs=True,
        downside_only=True,
        exposure_floor=0.5,
        trend_bonus=0.1,
        hard_selection=hard_selection,
        regime_guardrail=regime_guardrail,
        risk_off_exposure_cap=risk_off_exposure_cap,
        risk_off_strategy_penalty=risk_off_strategy_penalty,
    )
    if metadata is not None:
        env_kwargs.update(metadata.get("env_kwargs", {}))

    from src.agents.sac_agent import SACAgent

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_env = TradingEnv(train_data.copy(), **env_kwargs)
        agent = SACAgent(train_env, config={
            "learning_rate": 3e-4,
            "train_freq": 4,
            "seed": seed,
        })
        if checkpoint_path:
            agent.load(checkpoint_path)
            train_time = 0.0
            print(f"  Loaded checkpoint: {checkpoint_path}")
        else:
            t0 = time.perf_counter()
            agent.learn(total_timesteps=timesteps)
            train_time = time.perf_counter() - t0
            print(f"  Training took {train_time:.0f}s")

    # Run on test set, recording detailed action info
    test_env = TradingEnv(test_reset.copy(), **env_kwargs)
    test_agent = SACAgent(test_env, config={})
    test_agent.model = agent.model
    test_agent.model.set_env(test_env)

    obs, _ = test_env.reset()
    done = False

    agent_weights_history = []
    agent_exposure_history = []
    agent_position_history = []
    strategy_signals_during_test = []
    requested_strategy_history = []
    chosen_strategy_history = []
    guardrail_history = []

    step_count = 0
    while not done:
        raw_action = test_agent.act(obs)

        # Get strategy signals at this timestep
        feat_arr = test_env._features[test_env._step_idx]
        feat_dict = dict(zip(FEATURE_NAMES, feat_arr))
        sigs = [s.generate_signal(feat_dict).action for s in strategies]
        strategy_signals_during_test.append(sigs)

        obs, reward, terminated, truncated, info = test_env.step(raw_action)
        done = terminated or truncated
        agent_weights_history.append(np.asarray(info["strategy_weights"], dtype=np.float64))
        agent_exposure_history.append(float(info["effective_exposure"]))
        agent_position_history.append(info["position"])
        requested_strategy_history.append(int(info["requested_strategy"]))
        chosen_strategy_history.append(int(info["chosen_strategy"]))
        guardrail_history.append(bool(info["guardrail_active"]))
        step_count += 1

    agent_weights = np.array(agent_weights_history)
    agent_exposure = np.array(agent_exposure_history)
    agent_position = np.array(agent_position_history)
    test_signals = np.array(strategy_signals_during_test)
    requested_strategy = np.array(requested_strategy_history, dtype=int)
    chosen_strategy = np.array(chosen_strategy_history, dtype=int)
    guardrail_used = np.array(guardrail_history, dtype=bool)

    print(f"\n  Test steps: {step_count}")
    print(f"  Final portfolio value: ${test_env._portfolio_value:,.0f} (initial $100,000)")
    total_ret = test_env._portfolio_value / test_env.initial_cash - 1
    print(f"  Total return: {total_ret:+.1%}")

    # Strategy weight analysis
    print(f"\n--- Agent Strategy Allocation (mean over test) ---")
    print(f"{'Strategy':<15} {'Mean Wt':>8} {'Std Wt':>8} {'Min':>8} {'Max':>8}")
    print("-" * 47)
    for i, name in enumerate(strat_names):
        w = agent_weights[:, i]
        print(f"{name:<15} {w.mean():>8.1%} {w.std():>8.1%} {w.min():>8.1%} {w.max():>8.1%}")

    if hard_selection:
        print(f"\n--- Hard Strategy Picks ---")
        print(f"{'Strategy':<15} {'Chosen':>8} {'Requested':>10}")
        print("-" * 37)
        for i, name in enumerate(strat_names):
            chosen_pct = (chosen_strategy == i).mean() * 100
            requested_pct = (requested_strategy == i).mean() * 100
            print(f"{name:<15} {chosen_pct:>7.1f}% {requested_pct:>9.1f}%")
        if regime_guardrail:
            overrides = np.mean(requested_strategy != chosen_strategy) * 100
            print(f"  Guardrail active on {guardrail_used.mean()*100:.1f}% of steps")
            print(f"  Guardrail changed the requested pick on {overrides:.1f}% of steps")

    # Exposure analysis
    print(f"\n--- Agent Exposure Level ---")
    print(f"  Mean exposure: {agent_exposure.mean():.3f}")
    print(f"  Std exposure:  {agent_exposure.std():.3f}")
    print(f"  Min exposure:  {agent_exposure.min():.3f}")
    print(f"  Max exposure:  {agent_exposure.max():.3f}")
    print(f"  % at floor (0.50-0.55): {((agent_exposure >= 0.50) & (agent_exposure <= 0.55)).mean()*100:.1f}%")
    print(f"  % at max (0.95-1.00):   {((agent_exposure >= 0.95) & (agent_exposure <= 1.00)).mean()*100:.1f}%")

    # Position analysis
    print(f"\n--- Agent Position (direction x exposure) ---")
    print(f"  Mean |position|: {np.abs(agent_position).mean():.3f}")
    print(f"  Mean position:   {agent_position.mean():.3f}")
    print(f"  % long (>0.1):   {(agent_position > 0.1).mean()*100:.1f}%")
    print(f"  % short (<-0.1): {(agent_position < -0.1).mean()*100:.1f}%")
    print(f"  % flat (-0.1 to 0.1): {((agent_position >= -0.1) & (agent_position <= 0.1)).mean()*100:.1f}%")

    # What the agent position would be if it just used momentum strategy at full exposure
    momentum_only_pos = np.tanh(test_signals[:, 0] * 5.0)
    print(f"\n--- Comparison: Momentum-only at full exposure ---")
    print(f"  Mean |position|: {np.abs(momentum_only_pos).mean():.3f}")
    print(f"  % long (>0.1):   {(momentum_only_pos > 0.1).mean()*100:.1f}%")

    # Sub-period analysis of agent behavior
    period_header = f"{'Period':<20} {'Avg Pos':>8} {'Avg Exp':>8} {'Momentum':>10} {'MeanRev':>10} {'Breakout':>10} {'Defens':>10}"
    print(f"\n--- Agent Behavior by Market Period ---")
    print(period_header)
    print("-" * 88)

    valid_steps = min(len(dates), step_count)
    for pname, pstart, pend in periods:
        mask = (dates[:valid_steps] >= np.datetime64(pstart)) & (dates[:valid_steps] <= np.datetime64(pend))
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0]
        # Offset by lookback_window since agent starts at step lookback_window
        agent_idx = idx - test_env.lookback_window
        agent_idx = agent_idx[(agent_idx >= 0) & (agent_idx < step_count)]
        if len(agent_idx) == 0:
            continue
        p_pos = agent_position[agent_idx]
        p_exp = agent_exposure[agent_idx]
        p_wt = agent_weights[agent_idx]
        print(f"{pname:<20} {p_pos.mean():>+8.3f} {p_exp.mean():>8.3f} "
              f"{p_wt[:,0].mean():>10.1%} {p_wt[:,1].mean():>10.1%} "
              f"{p_wt[:,2].mean():>10.1%} {p_wt[:,3].mean():>10.1%}")

    # =====================================================================
    # PART 3: Strategy signal quality - correlation with next-day returns
    # =====================================================================
    print("\n" + "=" * 90)
    print("PART 3: STRATEGY SIGNAL PREDICTIVE QUALITY")
    print("=" * 90)

    # Next-day returns
    next_returns = np.diff(test_close) / test_close[:-1]
    # Align: signal[t] should predict return[t] (close[t]-close[t-1])
    # signals are computed at bar t, returns are computed at bar t (close[t]/close[t-1])
    # So signal[t] aligns with next_returns[t] for t from lookback to end-1
    max_t = min(n_steps - 1, len(next_returns))
    sig_aligned = signals[:max_t]
    ret_aligned = next_returns[:max_t]

    print(f"\n{'Strategy':<15} {'Corr w/ ret':>12} {'Hit Rate':>10} {'Avg ret|long':>13} {'Avg ret|short':>14}")
    print("-" * 64)
    for i, name in enumerate(strat_names):
        s = sig_aligned[:, i]
        corr = np.corrcoef(s, ret_aligned)[0, 1] if s.std() > 0 else 0.0
        # Hit rate: when signal > 0, did return > 0?
        long_mask = s > 0.01
        short_mask = s < -0.01
        if long_mask.sum() > 0:
            long_ret = ret_aligned[long_mask].mean() * 100
            long_hit = (ret_aligned[long_mask] > 0).mean() * 100
        else:
            long_ret = 0.0
            long_hit = 0.0
        if short_mask.sum() > 0:
            short_ret = ret_aligned[short_mask].mean() * 100
        else:
            short_ret = 0.0
        hit_rate = long_hit if long_mask.sum() > short_mask.sum() else (ret_aligned[short_mask] < 0).mean() * 100 if short_mask.sum() > 0 else 0.0
        print(f"{name:<15} {corr:>12.4f} {hit_rate:>9.1f}% {long_ret:>12.4f}% {short_ret:>13.4f}%")

    # Equal-weight blend
    ew = sig_aligned.mean(axis=1)
    corr_ew = np.corrcoef(ew, ret_aligned)[0, 1] if ew.std() > 0 else 0.0
    ew_long = ew > 0.01
    ew_long_ret = ret_aligned[ew_long].mean() * 100 if ew_long.sum() > 0 else 0.0
    print(f"{'EW Blend':<15} {corr_ew:>12.4f}")

    if hard_selection:
        print(f"\n--- Hard Picks by Market Period ---")
        print(f"{'Period':<20} {'Momentum':>10} {'MeanRev':>10} {'Breakout':>10} {'Defensive':>10}")
        print("-" * 72)
        valid_steps = min(len(dates), step_count)
        for pname, pstart, pend in periods:
            mask = (dates[:valid_steps] >= np.datetime64(pstart)) & (dates[:valid_steps] <= np.datetime64(pend))
            if mask.sum() == 0:
                continue
            idx = np.where(mask)[0]
            agent_idx = idx - test_env.lookback_window
            agent_idx = agent_idx[(agent_idx >= 0) & (agent_idx < step_count)]
            if len(agent_idx) == 0:
                continue
            picks = chosen_strategy[agent_idx]
            row = [f"{(picks == i).mean():>9.1%}" for i in range(len(strat_names))]
            print(f"{pname:<20} {row[0]} {row[1]} {row[2]} {row[3]}")

    print("\n" + "=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy diagnostics")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--start", type=str, default="2007-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--timesteps", "-t", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard-selection", action="store_true")
    parser.add_argument("--regime-guardrail", action="store_true")
    parser.add_argument("--risk-off-cap", type=float, default=None)
    parser.add_argument("--risk-off-strategy-penalty", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional saved model checkpoint (.zip) to load instead of retraining")
    parser.add_argument("--checkpoint-metadata", type=str, default=None,
                        help="Optional metadata JSON for a saved checkpoint; defaults to sibling .json")
    args = parser.parse_args()

    run_diagnostics(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        train_ratio=args.train_ratio,
        timesteps=args.timesteps,
        seed=args.seed,
        hard_selection=args.hard_selection,
        regime_guardrail=args.regime_guardrail,
        risk_off_exposure_cap=args.risk_off_cap,
        risk_off_strategy_penalty=args.risk_off_strategy_penalty,
        checkpoint_path=args.checkpoint,
        checkpoint_metadata=args.checkpoint_metadata,
    )
