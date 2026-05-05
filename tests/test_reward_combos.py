"""
Reward-shaping combo experiment.

Tests different reward configurations to find the best blend of:
  - DSR (Differential Sharpe Ratio)
  - Raw return blending (return_weight)
  - Exposure penalty (exposure_penalty)

Each combo trains SAC for --timesteps steps on real SPY data, then
evaluates on the held-out test set.

Usage:
    python -m tests.test_reward_combos                        # defaults
    python -m tests.test_reward_combos --timesteps 200000     # more training
    python -m tests.test_reward_combos --ticker QQQ           # different ticker
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.environment.data_loader import DataLoader
from src.environment.trading_env import TradingEnv
from src.evaluation.backtester import Backtester
from src.agents.early_stopping import EarlyStoppingCallback


# ======================================================================
# Reward configurations to test
# ======================================================================

COMBOS: dict[str, dict] = {
    "A: DSR-only (baseline)":     {"return_weight": 0.0, "exposure_penalty": 0.0},
    "B: +return_blend":           {"return_weight": 0.3, "exposure_penalty": 0.0},
    "C: +exposure_penalty":       {"return_weight": 0.0, "exposure_penalty": 0.5},
    "D: return+exposure":         {"return_weight": 0.3, "exposure_penalty": 0.5},
    "E: heavy_return+exposure":   {"return_weight": 0.6, "exposure_penalty": 0.3},
    "F: Sortino DSR":             {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True},
    "G: Sortino+return":          {"return_weight": 0.2, "exposure_penalty": 0.0, "downside_only": True},
    "H: Sortino+trend":           {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "trend_bonus": 0.1},
    "I: Sortino+benchrel":         {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "benchmark_relative": True},
    "J: Sortino+benchrel+trend":   {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "benchmark_relative": True, "trend_bonus": 0.1},
    "K: Sortino+bull_bench":        {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "bull_benchmark": True},
    "L: Sortino+floor50":           {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5},
    "M: Sortino+bull+floor50":      {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "bull_benchmark": True, "exposure_floor": 0.5},
    "N: Sortino+floor70":           {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.7},
    "O: Sortino+floor50+trend":     {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.1},
    "P: Sortino+floor70+trend":     {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.7, "trend_bonus": 0.1},
    "Q: Sortino+floor80":           {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.8},
    "R: Sortino+f60+t10":           {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.6, "trend_bonus": 0.1},
    "S: Sortino+f60+t15":           {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.6, "trend_bonus": 0.15},
    "T: Sortino+f50+t20":           {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.2},
    "U: Sortino+f60+t20":           {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.6, "trend_bonus": 0.2},
    "V: hard+f50+t10":              {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.1, "hard_selection": True},
    "W: hard+f60+t10":              {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.6, "trend_bonus": 0.1, "hard_selection": True},
    "X: hard+guard+f50+t10":        {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.1, "hard_selection": True, "regime_guardrail": True},
    "Y: hard+cap35+f50+t10":        {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.1, "hard_selection": True, "risk_off_exposure_cap": 0.35},
    "YP2: hard+cap35+pen02+f50+t10": {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.1, "hard_selection": True, "risk_off_exposure_cap": 0.35, "risk_off_strategy_penalty": 0.02},
    "YP3: hard+cap35+pen03+f50+t10": {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.1, "hard_selection": True, "risk_off_exposure_cap": 0.35, "risk_off_strategy_penalty": 0.03},
    "YP: hard+cap35+pen05+f50+t10": {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.1, "hard_selection": True, "risk_off_exposure_cap": 0.35, "risk_off_strategy_penalty": 0.05},
    "Z: hard+guard+cap35+t10":      {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.1, "hard_selection": True, "regime_guardrail": True, "risk_off_exposure_cap": 0.35},
    "Y25: hard+cap25+f50+t10":      {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.1, "hard_selection": True, "risk_off_exposure_cap": 0.25},
    "Y45: hard+cap45+f50+t10":      {"return_weight": 0.0, "exposure_penalty": 0.0, "downside_only": True, "exposure_floor": 0.5, "trend_bonus": 0.1, "hard_selection": True, "risk_off_exposure_cap": 0.45},
}


# ======================================================================
# Baseline for comparison
# ======================================================================

class EqualWeightAgent:
    def __init__(self):
        self.name = "EqualWeight"
        weights = np.ones(4, dtype=np.float32) / 4
        self._action = np.append(weights, 1.0)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._action.copy()


# ======================================================================
# Result container
# ======================================================================

@dataclass
class ComboResult:
    name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_position: float
    train_time: float
    portfolio_values: list = None  # stored for sub-period analysis
    stopped_at: int = 0


# ======================================================================
# Helpers
# ======================================================================

def _evaluate(agent, env: TradingEnv, name: str, train_time: float = 0.0) -> ComboResult:
    bt = Backtester(env, agent)
    results = bt.run(n_episodes=1)

    # Compute average absolute position from action history
    positions = [info.get("position", 0) for info in [{}]]  # placeholder
    # Get it from the env internals after backtesting
    avg_pos = float(np.mean([abs(p) for p in (results.get("position_history", [0.5]))]))

    return ComboResult(
        name=name,
        total_return=results["total_return"],
        annualized_return=results["annualized_return"],
        sharpe_ratio=results["sharpe_ratio"],
        sortino_ratio=results["sortino_ratio"],
        max_drawdown=results["max_drawdown"],
        calmar_ratio=results["calmar_ratio"],
        win_rate=results["win_rate"],
        avg_position=avg_pos,
        train_time=train_time,
        portfolio_values=results["portfolio_values"],
    )


# ======================================================================
# Sub-period analysis
# ======================================================================

# Key market periods to analyse (start, end, label)
_SUBPERIODS = [
    ("2020-02-19", "2020-03-23", "COVID crash"),
    ("2020-03-23", "2020-08-18", "COVID recovery"),
    ("2022-01-03", "2022-10-12", "2022 bear market"),
    ("2022-10-12", "2023-01-01", "Late-2022 rebound"),
    ("2023-01-01", "2024-12-31", "2023-24 bull run"),
]

# Lookback window used by TradingEnv (first N rows consumed, not traded)
_LOOKBACK = 20


def _subperiod_return(portfolio_values: list[float], dates: pd.DatetimeIndex,
                      start: str, end: str) -> float | None:
    """Compute return of portfolio values within [start, end] date range.

    portfolio_values has len = (len(dates) - lookback + 1):
      pv[0]  = initial cash (before first tradeable bar)
      pv[i]  = value after processing dates[lookback - 1 + i - 1]
    So pv index for date index j is:  pv_idx = j - lookback + 1
    """
    mask = (dates >= start) & (dates <= end)
    indices = np.where(mask)[0]
    if len(indices) < 2:
        return None
    i_start, i_end = int(indices[0]), int(indices[-1])
    # Map date indices to portfolio_values indices
    pv_start = i_start - _LOOKBACK + 1
    pv_end = i_end - _LOOKBACK + 1
    pv = np.array(portfolio_values, dtype=float)
    # Clamp to valid range
    pv_start = max(0, pv_start)
    pv_end = min(len(pv) - 1, pv_end)
    if pv_start >= pv_end or pv[pv_start] == 0:
        return None
    return float((pv[pv_end] - pv[pv_start]) / pv[pv_start])


def _print_subperiod_analysis(
    test_df: pd.DataFrame,
    results: list[ComboResult],
    ticker: str,
) -> None:
    """Print per-regime sub-period returns for each agent vs buy-and-hold."""
    if not isinstance(test_df.index, pd.DatetimeIndex):
        print("\n(Sub-period analysis skipped: test data has no datetime index)")
        return

    dates = test_df.index
    test_close = test_df["Close"].values.astype(float)
    test_start = dates[0]
    test_end = dates[-1]

    # Filter to sub-periods that fall within the test range
    active_periods = []
    for sp_start, sp_end, label in _SUBPERIODS:
        sp_s = pd.Timestamp(sp_start)
        sp_e = pd.Timestamp(sp_end)
        if sp_s >= test_start and sp_e <= test_end:
            active_periods.append((sp_start, sp_end, label))

    if not active_periods:
        print("\n(No predefined sub-periods fall within the test range)")
        return

    # Build short names for columns
    short_names = []
    for r in results:
        # Use first letter + key part: "A: DSR-only..." -> "A:DSR-only"
        name = r.name.split("(")[0].strip()
        if len(name) > 14:
            name = name[:14]
        short_names.append(name)

    col_w = max(14, max(len(n) for n in short_names) + 2)
    print(f"\n{'='*110}")
    print("SUB-PERIOD ANALYSIS (return within each market phase)")
    print(f"{'='*110}")

    header = f"{'Period':<22} {'Dates':>25} {'B&H':>10}"
    for sn in short_names:
        header += f" {sn:>{col_w}}"
    print(header)
    print("-" * len(header))

    for sp_start, sp_end, label in active_periods:
        mask = (dates >= sp_start) & (dates <= sp_end)
        indices = np.where(mask)[0]
        if len(indices) < 2:
            continue
        i0, i1 = int(indices[0]), int(indices[-1])
        bnh_sub = float((test_close[i1] / test_close[i0]) - 1)
        date_range = f"{dates[i0].strftime('%Y-%m-%d')}->{dates[i1].strftime('%Y-%m-%d')}"

        row = f"{label:<22} {date_range:>25} {bnh_sub:>+10.1%}"
        for r in results:
            if r.portfolio_values is not None:
                agent_ret = _subperiod_return(r.portfolio_values, dates, sp_start, sp_end)
                if agent_ret is not None:
                    row += f" {agent_ret:>+{col_w}.1%}"
                else:
                    row += f" {'N/A':>{col_w}}"
            else:
                row += f" {'N/A':>{col_w}}"
        print(row)

    print(f"{'='*110}")


def _fetch_data(
    ticker: str, start: str, end: str, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(
        tickers=[ticker], start_date=start, end_date=end, train_ratio=train_ratio
    )
    loader.fetch_data()
    train, test = loader.get_train_test_split(ticker)
    return train, test


def _config_code(name: str) -> str:
    return name.split(":", 1)[0].strip().upper()


def _checkpoint_stem(
    save_dir: str,
    ticker: str,
    combo_name: str,
    timesteps: int,
    seed_val: int | None,
) -> Path:
    stem_parts = [ticker.upper(), _config_code(combo_name), f"steps{timesteps}"]
    if seed_val is not None:
        stem_parts.append(f"seed{seed_val}")
    return Path(save_dir) / "_".join(stem_parts)


def _save_checkpoint_bundle(agent, stem: Path, metadata: dict) -> Path:
    stem.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(stem))
    checkpoint_path = stem.with_suffix(".zip")
    metadata_path = stem.with_suffix(".json")
    payload = dict(metadata)
    payload["checkpoint_path"] = str(checkpoint_path)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return checkpoint_path


# ======================================================================
# Main experiment
# ======================================================================

def run_combo_experiment(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    train_ratio: float = 0.7,
    timesteps: int = 200_000,
    config_filter: list[str] | None = None,
    use_early_stopping: bool = True,
    use_lstm: bool = False,
    obs_features: list[str] | None = None,
    seeds: list[int] | None = None,
    save_checkpoints_dir: str | None = None,
) -> list[ComboResult]:
    agent_label = "RecurrentPPO (LSTM)" if use_lstm else "SAC"
    print("=" * 90)
    print(f"REWARD COMBO EXPERIMENT - {ticker} - {timesteps} timesteps - {agent_label}")
    print("=" * 90)
    if obs_features:
        print(f"Obs features ({len(obs_features)}): {obs_features}")

    print(f"\nFetching {ticker} ({start} -> {end}) via yfinance ...")
    train, test = _fetch_data(ticker, start, end, train_ratio)
    print(f"Train: {len(train)} bars | Test: {len(test)} bars")

    if use_early_stopping:
        # Split train into train/val for early stopping (80/20)
        val_split = int(len(train) * 0.8)
        train_data = train.iloc[:val_split].reset_index(drop=True)
        val_data = train.iloc[val_split:].reset_index(drop=True)
        print(f"  Train split: {len(train_data)} train / {len(val_data)} val")
    else:
        train_data = train.reset_index(drop=True)
        val_data = None
        print(f"  No early stopping - using all {len(train_data)} train bars")

    bnh_return = float(test["Close"].iloc[-1] / test["Close"].iloc[0] - 1)
    print(f"Buy-and-hold {ticker} return on test set: {bnh_return:+.1%}")

    results: list[ComboResult] = []

    # Baseline: EqualWeight with full exposure (no training)
    test_env_base = TradingEnv(test.reset_index(drop=True), lookback_window=20, normalize_obs=True, obs_features=obs_features)
    ew_result = _evaluate(EqualWeightAgent(), test_env_base, "EqualWeight (baseline)")
    results.append(ew_result)
    print(f"\n  EqualWeight: return={ew_result.total_return:+.1%}, sharpe={ew_result.sharpe_ratio:.3f}")

    # Test each reward combo
    from src.agents.sac_agent import SACAgent
    if use_lstm:
        from src.agents.recurrent_ppo_agent import RecurrentPPOAgent

    sac_train_freq = 4 if timesteps > 10_000 else 1

    # Filter combos if requested (match on first letter, e.g. "A", "E")
    combos_to_run = COMBOS
    if config_filter:
        keys_upper = [k.upper() for k in config_filter]
        combos_to_run = {
            name: params for name, params in COMBOS.items()
            if _config_code(name) in keys_upper
        }

    multi_seed = seeds is not None and len(seeds) > 1

    for combo_name, params in combos_to_run.items():
        print(f"\n--- {combo_name} ---")
        extra_tags = []
        if params.get("downside_only"):
            extra_tags.append("downside_only")
        if params.get("benchmark_relative"):
            extra_tags.append("benchmark_relative")
        if params.get("bull_benchmark"):
            extra_tags.append("bull_benchmark")
        if params.get("exposure_floor", 0) > 0:
            extra_tags.append(f"exposure_floor={params['exposure_floor']}")
        if params.get("trend_bonus", 0) > 0:
            extra_tags.append(f"trend_bonus={params['trend_bonus']}")
        if params.get("hard_selection"):
            extra_tags.append("hard_selection")
        if params.get("regime_guardrail"):
            extra_tags.append("regime_guardrail")
        if params.get("risk_off_exposure_cap") is not None:
            extra_tags.append(f"risk_off_exposure_cap={params['risk_off_exposure_cap']}")
        if params.get("risk_off_strategy_penalty", 0) > 0:
            extra_tags.append(f"risk_off_strategy_penalty={params['risk_off_strategy_penalty']}")
        extra_str = (', ' + ', '.join(extra_tags)) if extra_tags else ''
        print(f"  return_weight={params['return_weight']}, exposure_penalty={params['exposure_penalty']}{extra_str}")

        env_kwargs = dict(
            lookback_window=20,
            normalize_obs=True,
            return_weight=params["return_weight"],
            exposure_penalty=params["exposure_penalty"],
            downside_only=params.get("downside_only", False),
            trend_bonus=params.get("trend_bonus", 0.0),
            benchmark_relative=params.get("benchmark_relative", False),
            bull_benchmark=params.get("bull_benchmark", False),
            exposure_floor=params.get("exposure_floor", 0.0),
            obs_features=obs_features,
            hard_selection=params.get("hard_selection", False),
            regime_guardrail=params.get("regime_guardrail", False),
            risk_off_exposure_cap=params.get("risk_off_exposure_cap"),
            risk_off_strategy_penalty=params.get("risk_off_strategy_penalty", 0.0),
        )

        seed_list = seeds if seeds is not None else [None]
        seed_results: list[ComboResult] = []

        for seed_val in seed_list:
            seed_label = f" [seed={seed_val}]" if seed_val is not None else ""

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Build train env with this reward config
                train_env = TradingEnv(train_data.copy(), **env_kwargs)

                es_callback = None
                if use_early_stopping and val_data is not None:
                    val_env = TradingEnv(val_data.copy(), **env_kwargs)
                    es_callback = EarlyStoppingCallback(
                        val_env=val_env,
                        eval_freq=10_000,
                        patience=10,
                        min_timesteps=50_000,
                        verbose=1,
                    )

                # Train agent
                agent_config = {
                    "learning_rate": 3e-4,
                    "train_freq": sac_train_freq,
                }
                if seed_val is not None:
                    agent_config["seed"] = seed_val

                if use_lstm:
                    agent = RecurrentPPOAgent(train_env, config={
                        "learning_rate": 3e-4,
                        "n_steps": 256,
                        "batch_size": 64,
                        "lstm_hidden_size": 128,
                    })
                else:
                    agent = SACAgent(train_env, config=agent_config)
                t0 = time.perf_counter()
                agent.learn(total_timesteps=timesteps, callback=es_callback)
                train_time = time.perf_counter() - t0

            if es_callback is not None:
                es_callback.restore_best()
                stopped_at = es_callback.num_timesteps
            else:
                stopped_at = timesteps
            if es_callback is not None:
                print(f"  {seed_label} Stopped at {stopped_at} steps (best val return: {es_callback.best_return:+.2%})")
            else:
                print(f"  {seed_label} Trained full {stopped_at} steps (no early stopping)")

            if save_checkpoints_dir:
                stem = _checkpoint_stem(
                    save_dir=save_checkpoints_dir,
                    ticker=ticker,
                    combo_name=combo_name,
                    timesteps=timesteps,
                    seed_val=seed_val,
                )
                metadata = {
                    "combo_name": combo_name,
                    "config_code": _config_code(combo_name),
                    "ticker": ticker,
                    "start": start,
                    "end": end,
                    "train_ratio": train_ratio,
                    "timesteps": timesteps,
                    "stopped_at": stopped_at,
                    "seed": seed_val,
                    "use_lstm": use_lstm,
                    "obs_features": obs_features,
                    "env_kwargs": env_kwargs,
                }
                checkpoint_path = _save_checkpoint_bundle(agent, stem, metadata)
                print(f"  {seed_label} Saved checkpoint: {checkpoint_path}")

            # Evaluate on test (test env uses same reward config for consistency,
            # though backtester only looks at portfolio values)
            test_env = TradingEnv(test.reset_index(drop=True), **env_kwargs)
            if use_lstm:
                test_agent = RecurrentPPOAgent(test_env, config={})
                test_agent.model = agent.model
                test_agent.model.set_env(test_env)
            else:
                test_agent = SACAgent(test_env, config={})
                test_agent.model = agent.model
                test_agent.model.set_env(test_env)

            r = _evaluate(test_agent, test_env, combo_name, train_time)
            r.stopped_at = stopped_at
            seed_results.append(r)

            print(f"  {seed_label} Return={r.total_return:+.1%}, Sharpe={r.sharpe_ratio:.3f}, "
                  f"Calmar={r.calmar_ratio:.3f}, MaxDD={r.max_drawdown:.1%}, Time={r.train_time:.0f}s")

        # Aggregate seed results
        if multi_seed:
            returns = [r.total_return for r in seed_results]
            sharpes = [r.sharpe_ratio for r in seed_results]
            sortinos = [r.sortino_ratio for r in seed_results]
            max_dds = [r.max_drawdown for r in seed_results]
            calmars = [r.calmar_ratio for r in seed_results]
            print(f"  >>> MEAN +/- STD over {len(seeds)} seeds:")
            print(f"      Return: {np.mean(returns):+.1%} +/- {np.std(returns):.1%}")
            print(f"      Sharpe: {np.mean(sharpes):.3f} +/- {np.std(sharpes):.3f}")
            print(f"      Sortino: {np.mean(sortinos):.3f} +/- {np.std(sortinos):.3f}")
            print(f"      MaxDD: {np.mean(max_dds):.1%} +/- {np.std(max_dds):.1%}")
            print(f"      Calmar: {np.mean(calmars):.3f} +/- {np.std(calmars):.3f}")
            # Use median result for summary table
            median_idx = int(np.argsort(returns)[len(returns) // 2])
            best_seed = seed_results[median_idx]
            best_seed.name = f"{combo_name} (median of {len(seeds)})"
            results.append(best_seed)
        else:
            results.append(seed_results[0])

    # ------------------------------------------------------------------
    # Buy-and-hold risk metrics for the test period
    # ------------------------------------------------------------------
    test_close = test["Close"].values.astype(float)
    bnh_returns = np.diff(test_close) / test_close[:-1]
    bnh_ann_ret = (1 + bnh_return) ** (252 / len(bnh_returns)) - 1
    bnh_sharpe = float(bnh_returns.mean() / bnh_returns.std() * np.sqrt(252)) if bnh_returns.std() > 0 else 0.0
    bnh_peak = np.maximum.accumulate(test_close)
    bnh_dd = (bnh_peak - test_close) / np.where(bnh_peak == 0, 1, bnh_peak)
    bnh_max_dd = float(bnh_dd.max())
    bnh_calmar = bnh_ann_ret / bnh_max_dd if bnh_max_dd > 0 else 0.0

    # ------------------------------------------------------------------
    # Summary table (with risk metrics)
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print(f"{'Configuration':<30} {'Return':>10} {'Ann.Ret':>10} {'Sharpe':>10} "
          f"{'Sortino':>10} {'MaxDD':>10} {'Calmar':>10} {'Steps':>10} {'Time(s)':>10}")
    print("-" * 110)
    for r in results:
        steps_str = f"{r.stopped_at:>10}" if r.stopped_at > 0 else f"{'-':>10}"
        print(f"{r.name:<30} {r.total_return:>+10.1%} {r.annualized_return:>+10.1%} "
              f"{r.sharpe_ratio:>10.3f} {r.sortino_ratio:>10.3f} {r.max_drawdown:>10.1%} "
              f"{r.calmar_ratio:>10.3f} {steps_str} {r.train_time:>10.0f}")
    print("-" * 110)
    print(f"{'Buy-and-hold ' + ticker:<30} {bnh_return:>+10.1%} {bnh_ann_ret:>+10.1%} "
          f"{bnh_sharpe:>10.3f} {'-':>10} {bnh_max_dd:>10.1%} {bnh_calmar:>10.3f}")
    print("=" * 110)

    # ------------------------------------------------------------------
    # Per-regime sub-period analysis
    # ------------------------------------------------------------------
    _print_subperiod_analysis(test, results, ticker)

    # Find best
    trained = [r for r in results if r.name != "EqualWeight (baseline)"]
    best = max(trained, key=lambda r: r.total_return)
    print(f"\nBest return: {best.name} at {best.total_return:+.1%} (Sharpe={best.sharpe_ratio:.3f})")

    best_sharpe = max(trained, key=lambda r: r.sharpe_ratio)
    if best_sharpe.name != best.name:
        print(f"Best Sharpe: {best_sharpe.name} at Sharpe={best_sharpe.sharpe_ratio:.3f} (Return={best_sharpe.total_return:+.1%})")

    best_calmar = max(trained, key=lambda r: r.calmar_ratio)
    if best_calmar.name != best.name and best_calmar.name != best_sharpe.name:
        print(f"Best Calmar: {best_calmar.name} at Calmar={best_calmar.calmar_ratio:.3f} (Return={best_calmar.total_return:+.1%})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward combo experiment")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--start", type=str, default="2007-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--timesteps", "-t", type=int, default=200_000)
    parser.add_argument("--configs", "-c", nargs="*", default=None,
                        help="Config keys to run (A, B, C, D, E). Default: all")
    parser.add_argument("--no-es", action="store_true", default=False,
                        help="Disable early stopping (use full training data)")
    parser.add_argument("--lstm", action="store_true", default=False,
                        help="Use RecurrentPPO (LSTM) instead of SAC")
    parser.add_argument("--features", nargs="*", default=None,
                        help="Obs feature subset (e.g. short_ma long_ma momentum). Default: all")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Random seeds for multi-seed runs (e.g. --seeds 0 1 2 3 4)")
    parser.add_argument("--save-checkpoints-dir", type=str, default=None,
                        help="Optional directory to save one checkpoint+metadata bundle per trained seed")
    args = parser.parse_args()

    run_combo_experiment(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        train_ratio=args.train_ratio,
        timesteps=args.timesteps,
        config_filter=args.configs,
        use_early_stopping=not args.no_es,
        use_lstm=args.lstm,
        obs_features=args.features,
        seeds=args.seeds,
        save_checkpoints_dir=args.save_checkpoints_dir,
    )
