"""Tests for MultiAssetTradingEnv (minimal viable)."""
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    })


@pytest.fixture
def basket():
    pytest.importorskip("gymnasium")
    return {
        "AAA": _make_ohlcv(seed=1, drift=0.0005),
        "BBB": _make_ohlcv(seed=2, drift=-0.0002),
        "CCC": _make_ohlcv(seed=3, drift=0.0001),
    }


def test_action_and_observation_space(basket):
    from src.environment.multi_asset_env import MultiAssetTradingEnv
    env = MultiAssetTradingEnv(basket, lookback_window=20)

    M = len(basket)
    assert env.action_space.shape == (M + 1,)
    # 20 lookback * 3 assets * 10 features + 3 positions + 1 cash dev = 604
    assert env.observation_space.shape == (20 * M * 10 + M + 1,)


def test_reset_and_step_shapes(basket):
    from src.environment.multi_asset_env import MultiAssetTradingEnv
    env = MultiAssetTradingEnv(basket, lookback_window=20)
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert info == {}

    action = np.array([0.5, 0.3, 0.2, 1.0], dtype=np.float32)  # M+1
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert info["weights"].shape == (3,)
    assert info["positions"].shape == (3,)
    # Weights are softmaxed -> sum to 1
    assert info["weights"].sum() == pytest.approx(1.0)
    # Positions = exposure * weights * max_position(=1)
    np.testing.assert_allclose(info["positions"], info["exposure"] * info["weights"])


def test_zero_exposure_zero_return(basket):
    """Exposure=0 must produce zero gross return regardless of asset moves."""
    from src.environment.multi_asset_env import MultiAssetTradingEnv
    env = MultiAssetTradingEnv(basket, lookback_window=20, transaction_cost=0.0, slippage_bps=0.0)
    env.reset(seed=0)
    action = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)  # exposure = 0
    _, reward, _, _, info = env.step(action)
    assert info["positions"].sum() == pytest.approx(0.0)
    assert info["gross_return"] == pytest.approx(0.0)
    assert reward == pytest.approx(0.0)


def test_misaligned_lengths_raise():
    from src.environment.multi_asset_env import MultiAssetTradingEnv
    data = {
        "AAA": _make_ohlcv(n=100),
        "BBB": _make_ohlcv(n=99),
    }
    with pytest.raises(ValueError, match="same length"):
        MultiAssetTradingEnv(data, lookback_window=20)


def test_missing_column_raises():
    from src.environment.multi_asset_env import MultiAssetTradingEnv
    bad = {
        "AAA": _make_ohlcv(),
        "BBB": pd.DataFrame({"Close": [1, 2, 3]}),
    }
    with pytest.raises(ValueError, match="missing required column"):
        MultiAssetTradingEnv(bad, lookback_window=20)


def test_terminates_at_end(basket):
    from src.environment.multi_asset_env import MultiAssetTradingEnv
    env = MultiAssetTradingEnv(basket, lookback_window=20)
    env.reset()
    action = np.array([0.5, 0.3, 0.2, 0.5], dtype=np.float32)
    terminated = False
    steps = 0
    while not terminated and steps < 1000:
        _, _, terminated, _, _ = env.step(action)
        steps += 1
    assert terminated
    assert env._step_idx == env.T


def test_ppo_agent_end_to_end(basket):
    """Smoke test: PPOAgent can train + act on MultiAssetTradingEnv."""
    pytest.importorskip("stable_baselines3")
    from src.environment.multi_asset_env import MultiAssetTradingEnv
    from src.agents.ppo_agent import PPOAgent

    env = MultiAssetTradingEnv(basket, lookback_window=20)
    agent = PPOAgent(env, config={"n_steps": 64, "batch_size": 32, "n_epochs": 1})
    agent.learn(total_timesteps=128)
    obs, _ = env.reset()
    action = agent.act(obs)
    assert action.shape == env.action_space.shape
    assert np.all(action >= env.action_space.low - 1e-6)
    assert np.all(action <= env.action_space.high + 1e-6)
