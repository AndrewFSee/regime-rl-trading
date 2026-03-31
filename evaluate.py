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


def main(config_path: str = "config/default.yaml", ticker: str | None = None) -> None:
    cfg = _load_config(config_path)
    ticker = ticker or cfg["data"]["tickers"][0]

    print(f"[evaluate] Config : {config_path}")
    print(f"[evaluate] Ticker : {ticker}")

    # ------------------------------------------------------------------
    # Data – test split
    # ------------------------------------------------------------------
    from src.environment.data_loader import DataLoader

    loader = DataLoader(
        tickers=[ticker],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        train_ratio=cfg["data"]["train_ratio"],
        interval=cfg["data"]["interval"],
    )
    print("[evaluate] Fetching data …")
    loader.fetch_data()
    _, test_df = loader.get_train_test_split(ticker)
    print(f"[evaluate] Test rows: {len(test_df)}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    from src.environment.trading_env import TradingEnv

    env_cfg = cfg["environment"]
    env = TradingEnv(
        data=test_df,
        lookback_window=env_cfg["lookback_window"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        reward_scaling=env_cfg["reward_scaling"],
        max_position=env_cfg["max_position"],
    )

    # ------------------------------------------------------------------
    # Load agent
    # ------------------------------------------------------------------
    agent_type = cfg["agent"].get("type", "ppo").lower()
    output_dir = cfg.get("evaluation", {}).get("output_dir", "results")
    model_path = os.path.join(output_dir, "model", f"{agent_type}_{ticker}")

    if agent_type == "ppo":
        from src.agents.ppo_agent import PPOAgent
        agent = PPOAgent(env, config=cfg["agent"])
        agent.load(model_path)
    elif agent_type == "dqn":
        from src.agents.dqn_agent import DQNAgent
        agent = DQNAgent(env, config=cfg["agent"])
        agent.load(model_path)
    else:
        raise ValueError(f"Unknown agent type '{agent_type}'.")

    print(f"[evaluate] Model loaded from {model_path}.zip")

    # ------------------------------------------------------------------
    # Backtest
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
    print(f"  Benchmark (B&H)     : {_fmt(results['benchmark_return'], pct=True)}")
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
        help="Ticker symbol (overrides config)",
    )
    args = parser.parse_args()
    main(config_path=args.config, ticker=args.ticker)
