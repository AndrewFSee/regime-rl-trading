"""
Integration smoke test: verifies the full pipeline (data → env → agent → backtest)
runs end-to-end without errors using synthetic data and minimal timesteps.

Requires stable-baselines3 and gymnasium.
"""
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    })


@pytest.mark.skipif(
    not pytest.importorskip("stable_baselines3", reason="stable_baselines3 not installed"),
    reason="stable_baselines3 not installed",
)
class TestIntegration:
    def test_ppo_train_and_backtest(self):
        """Full pipeline: create env → PPO agent → train 256 steps → backtest."""
        from src.environment.trading_env import TradingEnv
        from src.agents.ppo_agent import PPOAgent
        from src.evaluation.backtester import Backtester

        data = _make_ohlcv(200)
        env = TradingEnv(data, lookback_window=20, normalize_obs=True)
        agent = PPOAgent(env, config={"learning_rate": 3e-4, "batch_size": 64})

        # Train for a tiny number of steps (just verifies no crash)
        agent.learn(total_timesteps=256)

        backtester = Backtester(env, agent)
        results = backtester.run(n_episodes=1)

        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "portfolio_values" in results
        assert len(results["portfolio_values"]) > 1
        assert np.isfinite(results["total_return"])
        assert np.isfinite(results["sharpe_ratio"])

    def test_sac_train_and_backtest(self):
        """Full pipeline with SAC agent."""
        from src.environment.trading_env import TradingEnv
        from src.agents.sac_agent import SACAgent
        from src.evaluation.backtester import Backtester

        data = _make_ohlcv(200)
        env = TradingEnv(data, lookback_window=20, normalize_obs=True)
        agent = SACAgent(env, config={"learning_rate": 3e-4})

        agent.learn(total_timesteps=256)

        backtester = Backtester(env, agent)
        results = backtester.run(n_episodes=1)

        assert "total_return" in results
        assert np.isfinite(results["total_return"])

    def test_data_loader_roundtrip(self):
        """DataLoader → split → env → one step."""
        from src.environment.data_loader import DataLoader
        from src.environment.trading_env import TradingEnv

        loader = DataLoader(["SPY"], "2020-01-01", "2023-01-01")
        data = _make_ohlcv(300)
        loader.load_from_dataframe(data, ticker="SYNTH")
        train, test = loader.get_train_test_split("SYNTH")

        env = TradingEnv(train, lookback_window=10)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape

        action = env.action_space.sample()
        obs2, reward, term, trunc, info = env.step(action)
        assert np.isfinite(reward)
