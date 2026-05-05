"""
Evaluation script for regime-RL trading.

Usage
-----
    python evaluate.py [--config config/default.yaml] [--ticker SPY]

The script:
1. Loads configuration from YAML.
2. Downloads (or loads) OHLCV data and takes the test split.
3. Creates a TradingEnv on the test split.
4. Loads the trained model from ``results/model/``.
5. Runs the Backtester for one episode.
6. Prints all performance metrics.
7. Saves plots to ``results/plots/``.
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import yaml


def _load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _fmt(value: float, pct: bool = False) -> str:
    if pct:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def main(
    config_path: str = "config/default.yaml",
    ticker: str | None = None,
    multi_asset: bool = False,
) -> None:
    cfg = _load_config(config_path)

    print(f"[evaluate] Config : {config_path}")
    if multi_asset:
        tickers = list(cfg["data"]["tickers"])
        if len(tickers) < 2:
            raise ValueError("--multi-asset requires >=2 tickers in config.data.tickers.")
        print(f"[evaluate] Mode   : MULTI-ASSET ({len(tickers)} tickers)")
    else:
        ticker = ticker or cfg["data"]["tickers"][0]
        print(f"[evaluate] Ticker : {ticker}")

    # ------------------------------------------------------------------
    # Data – test split
    # ------------------------------------------------------------------
    from src.environment.data_loader import DataLoader

    loader = DataLoader(
        tickers=tickers if multi_asset else [ticker],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        train_ratio=cfg["data"]["train_ratio"],
        interval=cfg["data"]["interval"],
    )
    print("[evaluate] Fetching data …")
    loader.fetch_data()
    if multi_asset:
        test_data: dict = {}
        min_len = None
        for tk in tickers:
            _, te = loader.get_train_test_split(tk)
            test_data[tk] = te
            min_len = len(te) if min_len is None else min(min_len, len(te))
        for tk in tickers:
            test_data[tk] = test_data[tk].iloc[:min_len].reset_index(drop=True)
        print(f"[evaluate] Test rows per ticker: {min_len}")
    else:
        _, test_df = loader.get_train_test_split(ticker)
        print(f"[evaluate] Test rows: {len(test_df)}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env_cfg = cfg["environment"]
    if multi_asset:
        from src.environment.multi_asset_env import MultiAssetTradingEnv

        ma_kwargs = {
            "data": test_data,
            "lookback_window": env_cfg["lookback_window"],
            "initial_cash": env_cfg["initial_cash"],
            "transaction_cost": env_cfg["transaction_cost"],
            "reward_scaling": env_cfg["reward_scaling"],
            "max_position": env_cfg["max_position"],
        }
        if "slippage_bps" in env_cfg:
            ma_kwargs["slippage_bps"] = env_cfg["slippage_bps"]
        env = MultiAssetTradingEnv(**ma_kwargs)
    else:
        from src.environment.trading_env import TradingEnv

        optional_env_keys = (
            "slippage_factor", "slippage_bps", "slippage_vol_coef", "borrow_cost_bps_yr",
            "normalize_obs", "return_weight", "exposure_penalty", "downside_only",
            "trend_bonus", "benchmark_relative", "bull_benchmark", "exposure_floor",
            "obs_features", "hard_selection", "regime_guardrail",
            "risk_off_exposure_cap", "risk_off_strategy_penalty",
            "drawdown_penalty", "drawdown_threshold",
        )
        env_kwargs = {
            "data": test_df,
            "lookback_window": env_cfg["lookback_window"],
            "initial_cash": env_cfg["initial_cash"],
            "transaction_cost": env_cfg["transaction_cost"],
            "reward_scaling": env_cfg["reward_scaling"],
            "max_position": env_cfg["max_position"],
        }
        for k in optional_env_keys:
            if k in env_cfg:
                env_kwargs[k] = env_cfg[k]
        env = TradingEnv(**env_kwargs)

    # ------------------------------------------------------------------
    # Load agent
    # ------------------------------------------------------------------
    agent_type = cfg["agent"].get("type", "ppo").lower()
    output_dir = cfg.get("evaluation", {}).get("output_dir", "results")
    tag = "multi" if multi_asset else ticker
    model_path = os.path.join(output_dir, "model", f"{agent_type}_{tag}")

    if agent_type == "ppo":
        from src.agents.ppo_agent import PPOAgent
        agent = PPOAgent(env, config=cfg["agent"])
    elif agent_type == "sac":
        from src.agents.sac_agent import SACAgent
        agent = SACAgent(env, config=cfg["agent"])
    elif agent_type == "tqc":
        from src.agents.tqc_agent import TQCAgent
        agent = TQCAgent(env, config=cfg["agent"])
    else:
        raise ValueError(f"Unknown agent type '{agent_type}'.")
    agent.load(model_path)

    print(f"[evaluate] Model loaded from {model_path}.zip")

    # ------------------------------------------------------------------
    # Multi-asset: lightweight rollout (Backtester is single-asset only)
    # ------------------------------------------------------------------
    if multi_asset:
        import numpy as np
        from src.evaluation.backtester import Backtester as _BT  # static metrics

        obs, _ = env.reset()
        portfolio_values = [env.initial_cash]
        step_returns: list[float] = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            portfolio_values.append(float(info.get("portfolio_value", portfolio_values[-1])))
            step_returns.append(float(info.get("net_return", 0.0)))

        total_return = portfolio_values[-1] / portfolio_values[0] - 1.0
        n_days = len(step_returns)
        ann_ret = _BT._annualised_return(total_return, n_days)
        sharpe = _BT._sharpe(step_returns)
        sortino = _BT._sortino(step_returns)
        mdd = _BT._max_drawdown(portfolio_values)
        calmar = _BT._calmar(ann_ret, mdd)

        # Equal-weight buy-and-hold benchmark.
        eq_returns = np.zeros(n_days, dtype=float)
        for tk in env.tickers:
            close = env._frames[tk]["Close"].to_numpy()
            start = env.lookback_window
            seg = close[start : start + n_days + 1]
            if len(seg) >= 2:
                seg_ret = np.diff(seg) / seg[:-1]
                eq_returns[: len(seg_ret)] += seg_ret / env.M
        eq_pv = np.concatenate([[1.0], np.cumprod(1.0 + eq_returns)])
        bench_total = float(eq_pv[-1] - 1.0)
        bench_ann = _BT._annualised_return(bench_total, n_days)
        bench_sharpe = _BT._sharpe(eq_returns.tolist())
        bench_mdd = _BT._max_drawdown(eq_pv.tolist())

        print("\n" + "=" * 50)
        print("  MULTI-ASSET BACKTEST RESULTS")
        print("=" * 50)
        print(f"  Total Return        : {_fmt(total_return, pct=True)}")
        print(f"  Annualised Return   : {_fmt(ann_ret, pct=True)}")
        print(f"  Sharpe Ratio        : {_fmt(sharpe)}")
        print(f"  Sortino Ratio       : {_fmt(sortino)}")
        print(f"  Max Drawdown        : {_fmt(mdd, pct=True)}")
        print(f"  Calmar Ratio        : {_fmt(calmar)}")
        print("-" * 50)
        print("  EQUAL-WEIGHT BUY-AND-HOLD")
        print("-" * 50)
        print(f"  Total Return        : {_fmt(bench_total, pct=True)}")
        print(f"  Annualised Return   : {_fmt(bench_ann, pct=True)}")
        print(f"  Sharpe Ratio        : {_fmt(bench_sharpe)}")
        print(f"  Max Drawdown        : {_fmt(bench_mdd, pct=True)}")
        print("=" * 50)
        return

    # ------------------------------------------------------------------
    # Single-asset Backtest
    # ------------------------------------------------------------------
    from src.evaluation.backtester import Backtester

    backtester = Backtester(env, agent)
    results = backtester.run(n_episodes=1)

    # ------------------------------------------------------------------
    # Print metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Total Return        : {_fmt(results['total_return'], pct=True)}")
    print(f"  Annualised Return   : {_fmt(results['annualized_return'], pct=True)}")
    print(f"  Sharpe Ratio        : {_fmt(results['sharpe_ratio'])}")
    print(f"  Sortino Ratio       : {_fmt(results['sortino_ratio'])}")
    print(f"  Max Drawdown        : {_fmt(results['max_drawdown'], pct=True)}")
    print(f"  Calmar Ratio        : {_fmt(results['calmar_ratio'])}")
    print(f"  Win Rate            : {_fmt(results['win_rate'], pct=True)}")
    print(f"  Profit Factor       : {_fmt(results['profit_factor'])}")
    print("-" * 50)
    print("  BUY-AND-HOLD BASELINE")
    print("-" * 50)
    print(f"  Total Return        : {_fmt(results['benchmark_return'], pct=True)}")
    print(f"  Annualised Return   : {_fmt(results['benchmark_annualized_return'], pct=True)}")
    print(f"  Sharpe Ratio        : {_fmt(results['benchmark_sharpe_ratio'])}")
    print(f"  Sortino Ratio       : {_fmt(results['benchmark_sortino_ratio'])}")
    print(f"  Max Drawdown        : {_fmt(results['benchmark_max_drawdown'], pct=True)}")
    print(f"  Calmar Ratio        : {_fmt(results['benchmark_calmar_ratio'])}")
    print("-" * 50)
    excess_return = results["total_return"] - results["benchmark_return"]
    excess_sharpe = results["sharpe_ratio"] - results["benchmark_sharpe_ratio"]
    print(f"  Agent \u2212 B&H Return  : {_fmt(excess_return, pct=True)}")
    print(f"  Agent \u2212 B&H Sharpe  : {_fmt(excess_sharpe)}")
    print("=" * 50)

    # Additional rule-based baselines (60/40, vol-target, 200d MA filter)
    baselines = results.get("baselines") or {}
    if baselines:
        print("\n  RULE-BASED BASELINES")
        print("-" * 50)
        print(f"  {'Baseline':<18}{'Return':>10}{'Sharpe':>10}{'MaxDD':>10}{'Calmar':>10}")
        labels = {
            "sixty_forty":   "60/40 SPY/cash",
            "vol_target_10": "Vol-target 10%",
            "ma_filter_200": "200d MA filter",
        }
        for key, label in labels.items():
            row = baselines.get(key)
            if not row:
                continue
            print(
                f"  {label:<18}"
                f"{_fmt(row['total_return'], pct=True):>10}"
                f"{_fmt(row['sharpe_ratio']):>10}"
                f"{_fmt(row['max_drawdown'], pct=True):>10}"
                f"{_fmt(row['calmar_ratio']):>10}"
            )
        print("=" * 50)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    save_plots = cfg.get("evaluation", {}).get("save_plots", True)
    if save_plots:
        try:
            from src.evaluation.visualizer import Visualizer
            plots_dir = os.path.join(output_dir, "plots")
            viz = Visualizer()
            viz.plot_all(results, save_dir=plots_dir)
            print(f"[evaluate] Plots saved → {plots_dir}/")
        except ImportError:
            print("[evaluate] matplotlib not available – skipping plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate regime-RL trading agent")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--ticker", default=None,
        help="Ticker symbol (overrides config; ignored with --multi-asset)",
    )
    parser.add_argument(
        "--multi-asset", action="store_true",
        help="Evaluate the multi-asset portfolio model (MultiAssetTradingEnv).",
    )
    args = parser.parse_args()
    main(config_path=args.config, ticker=args.ticker, multi_asset=args.multi_asset)
