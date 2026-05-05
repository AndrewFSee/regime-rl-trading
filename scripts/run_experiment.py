"""
Multi-seed train+evaluate experiment harness.

Runs the same agent+config across N random seeds, evaluates each on the test
split, and reports mean +/- std for headline metrics. Saves a per-seed CSV to
``results/experiment_<ticker>_<agent>.csv``.

Usage
-----
    python scripts/run_experiment.py --seeds 5
    python scripts/run_experiment.py --seeds 3 --ticker QQQ --episodes 200
    python scripts/run_experiment.py --seeds 5 --multi-asset
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import yaml

try:
    import gymnasium as _gym
except ImportError:  # pragma: no cover - gymnasium is a hard dep of SB3
    import gym as _gym  # type: ignore


class _ActionRepeatWrapper(_gym.Wrapper):
    """Repeat the agent's action for ``k`` underlying env steps.

    Reduces decision frequency, which (a) cuts turnover by ~k and (b) gives the
    policy fewer noisy reward signals to fit. Sums the rewards across the k
    inner steps and returns the final observation. Terminates early if the
    inner env terminates or truncates mid-burst.
    """

    def __init__(self, env, k: int):
        super().__init__(env)
        if k < 1:
            raise ValueError(f"action_repeat must be >= 1 (got {k})")
        self._k = int(k)

    def __getattr__(self, name):
        # Proxy attribute access (e.g. initial_cash, _position) to the wrapped
        # env so callers like Backtester can introspect env state without
        # caring about wrapper layers.
        if name.startswith("_") and name not in ("_position", "_portfolio_value", "_peak_value"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def step(self, action):  # type: ignore[override]
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        obs = None
        for _ in range(self._k):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def _load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _build_data(cfg: dict, multi_asset: bool, ticker: str | None):
    from src.environment.data_loader import DataLoader

    tickers = list(cfg["data"]["tickers"]) if multi_asset else [ticker or cfg["data"]["tickers"][0]]
    loader = DataLoader(
        tickers=tickers,
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        train_ratio=cfg["data"]["train_ratio"],
        interval=cfg["data"]["interval"],
    )
    loader.fetch_data()

    if multi_asset:
        train_d, test_d = {}, {}
        min_tr = min_te = None
        for tk in tickers:
            tr, te = loader.get_train_test_split(tk)
            train_d[tk] = tr
            test_d[tk] = te
            min_tr = len(tr) if min_tr is None else min(min_tr, len(tr))
            min_te = len(te) if min_te is None else min(min_te, len(te))
        for tk in tickers:
            train_d[tk] = train_d[tk].iloc[:min_tr].reset_index(drop=True)
            test_d[tk] = test_d[tk].iloc[:min_te].reset_index(drop=True)
        return train_d, test_d, tickers
    else:
        tr, te = loader.get_train_test_split(tickers[0])
        return tr, te, tickers


def _make_env(cfg: dict, data: Any, multi_asset: bool):
    env_cfg = cfg["environment"]
    action_repeat = int(cfg.get("_action_repeat", 1) or 1)
    if multi_asset:
        from src.environment.multi_asset_env import MultiAssetTradingEnv
        kw = {
            "data": data,
            "lookback_window": env_cfg["lookback_window"],
            "initial_cash": env_cfg["initial_cash"],
            "transaction_cost": env_cfg["transaction_cost"],
            "reward_scaling": env_cfg["reward_scaling"],
            "max_position": env_cfg["max_position"],
        }
        if "slippage_bps" in env_cfg:
            kw["slippage_bps"] = env_cfg["slippage_bps"]
        env = MultiAssetTradingEnv(**kw)
        if action_repeat > 1:
            env = _ActionRepeatWrapper(env, action_repeat)
        return env

    from src.environment.trading_env import TradingEnv
    optional = (
        "slippage_factor", "slippage_bps", "slippage_vol_coef", "borrow_cost_bps_yr",
        "normalize_obs", "return_weight", "exposure_penalty", "downside_only",
        "trend_bonus", "benchmark_relative", "bull_benchmark", "exposure_floor",
        "obs_features", "hard_selection", "regime_guardrail",
        "risk_off_exposure_cap", "risk_off_strategy_penalty",
        "drawdown_penalty", "drawdown_threshold",
    )
    kw = {
        "data": data,
        "lookback_window": env_cfg["lookback_window"],
        "initial_cash": env_cfg["initial_cash"],
        "transaction_cost": env_cfg["transaction_cost"],
        "reward_scaling": env_cfg["reward_scaling"],
        "max_position": env_cfg["max_position"],
    }
    for k in optional:
        if k in env_cfg:
            kw[k] = env_cfg[k]

    # Optional: inject macro features. cfg["_macro_panel"] is a Date-indexed
    # DataFrame of pre-transformed (stationary) macro features covering at
    # least ``data``'s date range. We reindex it to ``data``'s index before
    # passing to TradingEnv, which resets the index internally.
    macro_panel = cfg.get("_macro_panel")
    if macro_panel is not None:
        macro_aligned = macro_panel.reindex(data.index).ffill().bfill().fillna(0.0)
        kw["macro_features"] = macro_aligned

    env = TradingEnv(**kw)
    if action_repeat > 1:
        env = _ActionRepeatWrapper(env, action_repeat)
    return env


def _build_agent(env, agent_cfg: dict, seed: int):
    agent_type = agent_cfg.get("type", "ppo").lower()
    cfg = dict(agent_cfg)
    cfg["seed"] = seed  # threaded into SB3 ctor via kwargs in agents that support it
    if agent_type == "ppo":
        from src.agents.ppo_agent import PPOAgent
        return PPOAgent(env, config=cfg), agent_type
    if agent_type == "sac":
        from src.agents.sac_agent import SACAgent
        return SACAgent(env, config=cfg), agent_type
    if agent_type == "tqc":
        from src.agents.tqc_agent import TQCAgent
        return TQCAgent(env, config=cfg), agent_type
    if agent_type == "dqn":
        from src.agents.dqn_agent import DQNAgent
        return DQNAgent(env, config=cfg), agent_type
    raise ValueError(f"Unknown agent type '{agent_type}'.")


def _evaluate_single_asset(env, agent) -> dict:
    from src.evaluation.backtester import Backtester
    bt = Backtester(env, agent)
    return bt.run(n_episodes=1)


def _evaluate_multi_asset(env, agent) -> dict:
    """Lightweight portfolio rollout matching evaluate.py --multi-asset."""
    from src.evaluation.backtester import Backtester as _BT
    obs, _ = env.reset()
    pv = [env.initial_cash]
    rets: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(obs)
        obs, _r, terminated, truncated, info = env.step(action)
        pv.append(float(info.get("portfolio_value", pv[-1])))
        rets.append(float(info.get("net_return", 0.0)))
    total = pv[-1] / pv[0] - 1.0
    n = len(rets)
    ann = _BT._annualised_return(total, n)
    return {
        "total_return": total,
        "annualized_return": ann,
        "sharpe_ratio": _BT._sharpe(rets),
        "sortino_ratio": _BT._sortino(rets),
        "max_drawdown": _BT._max_drawdown(pv),
        "calmar_ratio": _BT._calmar(ann, _BT._max_drawdown(pv)),
    }


HEADLINE_KEYS = (
    "total_return", "annualized_return", "sharpe_ratio",
    "sortino_ratio", "max_drawdown", "calmar_ratio",
)


def _print_summary(per_seed: list[dict], label: str) -> None:
    print(f"\n  {label}")
    print("  " + "-" * 60)
    print(f"  {'metric':<22}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}")
    for k in HEADLINE_KEYS:
        vals = np.array([float(r[k]) for r in per_seed if not np.isnan(float(r[k]))], dtype=float)
        if len(vals) == 0:
            print(f"  {k:<22}{'nan':>12}")
            continue
        print(
            f"  {k:<22}"
            f"{vals.mean():>12.4f}"
            f"{vals.std(ddof=1) if len(vals) > 1 else 0.0:>12.4f}"
            f"{vals.min():>12.4f}"
            f"{vals.max():>12.4f}"
        )


def main(
    config_path: str = "config/default.yaml",
    ticker: str | None = None,
    seeds: int = 3,
    episodes: int | None = None,
    multi_asset: bool = False,
    agent_type: str | None = None,
    verbose: int = 0,
    return_weight: float | None = None,
    transaction_cost: float | None = None,
    action_repeat: int = 1,
) -> None:
    cfg = _load_config(config_path)
    if episodes is not None:
        cfg["agent"]["training_episodes"] = episodes
    if agent_type is not None:
        cfg["agent"]["type"] = agent_type
    if return_weight is not None:
        cfg.setdefault("environment", {})["return_weight"] = float(return_weight)
    if transaction_cost is not None:
        cfg.setdefault("environment", {})["transaction_cost"] = float(transaction_cost)
    cfg["_action_repeat"] = max(1, int(action_repeat))
    cfg["agent"]["verbose"] = int(verbose)
    total_timesteps = cfg["agent"]["training_episodes"] * 1000

    print(f"[experiment] Config        : {config_path}")
    print(f"[experiment] Mode          : {'MULTI-ASSET' if multi_asset else 'SINGLE-ASSET'}")
    print(f"[experiment] Agent         : {cfg['agent'].get('type', 'ppo').upper()}")
    print(f"[experiment] Seeds         : {seeds}")
    print(f"[experiment] Timesteps/run : {total_timesteps:,}")
    print(f"[experiment] return_weight : {cfg.get('environment', {}).get('return_weight', 0.0)}")
    print(f"[experiment] txn_cost      : {cfg.get('environment', {}).get('transaction_cost', 0.0)}")
    print(f"[experiment] action_repeat : {cfg['_action_repeat']}")

    print("[experiment] Fetching data ...")
    train_data, test_data, tickers = _build_data(cfg, multi_asset, ticker)

    per_seed: list[dict] = []
    seed_list = list(range(seeds))
    t_start = time.perf_counter()

    for s in seed_list:
        print(f"\n[experiment] === seed {s} ({s + 1}/{seeds}) ===")
        _set_global_seeds(s)

        train_env = _make_env(cfg, train_data, multi_asset)
        agent, agent_type = _build_agent(train_env, cfg["agent"], seed=s)

        t0 = time.perf_counter()
        agent.learn(total_timesteps=total_timesteps)
        train_t = time.perf_counter() - t0
        print(f"[experiment] seed {s}: trained in {train_t:.1f}s")

        test_env = _make_env(cfg, test_data, multi_asset)
        # Re-bind agent's env to the test env for evaluation.
        agent.env = test_env
        if multi_asset:
            r = _evaluate_multi_asset(test_env, agent)
        else:
            r = _evaluate_single_asset(test_env, agent)
        r["seed"] = s
        r["train_seconds"] = round(train_t, 2)
        per_seed.append(r)
        print(
            f"[experiment] seed {s}: "
            f"return={r['total_return']*100:6.2f}%  "
            f"sharpe={r['sharpe_ratio']:5.2f}  "
            f"maxDD={r['max_drawdown']*100:5.2f}%"
        )

    elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Aggregate + report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT RESULTS ({seeds} seeds, {elapsed:.1f}s total)")
    print("=" * 70)
    _print_summary(per_seed, "Agent metrics across seeds")

    # Single-asset path also returns benchmark metrics; aggregate if present.
    if not multi_asset and "benchmark_return" in per_seed[0]:
        bench = []
        for r in per_seed:
            bench.append({
                "total_return": r["benchmark_return"],
                "annualized_return": r.get("benchmark_annualized_return", float("nan")),
                "sharpe_ratio": r.get("benchmark_sharpe_ratio", float("nan")),
                "sortino_ratio": r.get("benchmark_sortino_ratio", float("nan")),
                "max_drawdown": r.get("benchmark_max_drawdown", float("nan")),
                "calmar_ratio": r.get("benchmark_calmar_ratio", float("nan")),
            })
        # Benchmark is deterministic across seeds, but report once for clarity.
        _print_summary(bench, "Buy-and-hold benchmark")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    out_dir = cfg.get("evaluation", {}).get("output_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = "multi" if multi_asset else (ticker or cfg["data"]["tickers"][0])
    csv_path = os.path.join(out_dir, f"experiment_{tag}_{cfg['agent'].get('type','ppo')}.csv")
    fieldnames = ["seed", "train_seconds"] + list(HEADLINE_KEYS)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in per_seed:
            writer.writerow(r)
    print(f"\n[experiment] Per-seed CSV saved -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed regime-RL experiment runner")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds (default: 3)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override config.agent.training_episodes (1 episode = 1000 steps)")
    parser.add_argument("--multi-asset", action="store_true")
    parser.add_argument("--agent", default=None,
                        help="Override config.agent.type (ppo, sac, tqc, dqn).")
    parser.add_argument("--verbose", type=int, default=0,
                        help="SB3 verbosity (0=quiet, 1=progress, 2=debug).")
    parser.add_argument("--return-weight", type=float, default=None,
                        help="Override config.environment.return_weight (P&L weight in reward).")
    parser.add_argument("--transaction-cost", type=float, default=None,
                        help="Override config.environment.transaction_cost (per-unit turnover cost).")
    parser.add_argument("--action-repeat", type=int, default=1,
                        help="Hold each action for K env steps (turnover reduction). Default 1.")
    args = parser.parse_args()
    main(
        config_path=args.config,
        ticker=args.ticker,
        seeds=args.seeds,
        episodes=args.episodes,
        multi_asset=args.multi_asset,
        agent_type=args.agent,
        verbose=args.verbose,
        return_weight=args.return_weight,
        transaction_cost=args.transaction_cost,
        action_repeat=args.action_repeat,
    )
