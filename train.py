"""
Main training script for regime-RL trading.

Usage
-----
    python train.py [--config config/default.yaml] [--ticker SPY]

The script:
1. Loads configuration from YAML.
2. Downloads (or loads) OHLCV data via DataLoader.
3. Creates a TradingEnv on the training split.
4. Instantiates a PPO or DQN agent according to ``config.agent.type``.
5. Trains the agent.
6. Saves the model to ``results/model/``.
7. Prints a brief training summary.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure the project src/ directory is on the path when running as a script.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import yaml


def _load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def main(config_path: str = "config/default.yaml", ticker: str | None = None) -> None:
    cfg = _load_config(config_path)
    ticker = ticker or cfg["data"]["tickers"][0]

    print(f"[train] Config : {config_path}")
    print(f"[train] Ticker : {ticker}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    from src.environment.data_loader import DataLoader

    loader = DataLoader(
        tickers=[ticker],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        train_ratio=cfg["data"]["train_ratio"],
        interval=cfg["data"]["interval"],
    )
    print("[train] Fetching data …")
    loader.fetch_data()
    train_df, _ = loader.get_train_test_split(ticker)
    print(f"[train] Training rows: {len(train_df)}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    from src.environment.trading_env import TradingEnv

    env_cfg = cfg["environment"]
    env = TradingEnv(
        data=train_df,
        lookback_window=env_cfg["lookback_window"],
        initial_cash=env_cfg["initial_cash"],
        transaction_cost=env_cfg["transaction_cost"],
        reward_scaling=env_cfg["reward_scaling"],
        max_position=env_cfg["max_position"],
    )

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    agent_cfg = cfg["agent"]
    agent_type = agent_cfg.get("type", "ppo").lower()
    # timesteps = episodes × steps_per_episode (1 000 is a reasonable episode length)
    total_timesteps = agent_cfg["training_episodes"] * 1000

    if agent_type == "ppo":
        from src.agents.ppo_agent import PPOAgent
        agent = PPOAgent(env, config=agent_cfg)
    elif agent_type == "dqn":
        from src.agents.dqn_agent import DQNAgent
        agent = DQNAgent(env, config=agent_cfg)
    else:
        raise ValueError(f"Unknown agent type '{agent_type}'. Choose 'ppo' or 'dqn'.")

    print(f"[train] Agent  : {agent_type.upper()}  ({total_timesteps:,} timesteps)")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    agent.learn(total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_dir = cfg.get("evaluation", {}).get("output_dir", "results")
    model_dir  = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{agent_type}_{ticker}")
    agent.save(model_path)

    print(f"[train] Training complete in {elapsed:.1f}s")
    print(f"[train] Model saved → {model_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regime-RL trading agent")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--ticker", default=None,
        help="Ticker symbol to trade (overrides config)",
    )
    args = parser.parse_args()
    main(config_path=args.config, ticker=args.ticker)
