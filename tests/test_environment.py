"""
Tests for TradingEnv, FeatureEngineer, and DataLoader.

All tests generate synthetic data locally and do not require network access
or heavy ML dependencies.
"""
import sys
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close  = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high   = close * (1 + rng.uniform(0, 0.02, n))
    low    = close * (1 - rng.uniform(0, 0.02, n))
    open_  = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    })


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

class TestFeatureEngineer:
    def test_compute_returns_dataframe(self):
        from src.environment.features import FeatureEngineer, FEATURE_NAMES
        eng = FeatureEngineer()
        data = _make_ohlcv(100)
        feats = eng.compute(data)
        assert isinstance(feats, pd.DataFrame)
        assert len(feats) == len(data)
        assert list(feats.columns) == FEATURE_NAMES

    def test_no_nan(self):
        from src.environment.features import FeatureEngineer
        eng = FeatureEngineer()
        data = _make_ohlcv(100)
        feats = eng.compute(data)
        assert not feats.isnull().any().any(), "Features contain NaN values"

    def test_feature_count(self):
        from src.environment.features import FeatureEngineer, FEATURE_NAMES
        assert len(FEATURE_NAMES) == 10

    def test_short_data(self):
        """Should not crash on very short data."""
        from src.environment.features import FeatureEngineer
        eng = FeatureEngineer()
        data = _make_ohlcv(5)
        feats = eng.compute(data)
        assert len(feats) == 5

    def test_rsi_in_range(self):
        from src.environment.features import FeatureEngineer
        eng = FeatureEngineer()
        data = _make_ohlcv(100)
        feats = eng.compute(data)
        # rsi column is normalised: rsi/50 - 1 → range [-1, 1]
        assert feats["rsi"].between(-1.0, 1.0).all()

    def test_volume_ratio_positive(self):
        from src.environment.features import FeatureEngineer
        eng = FeatureEngineer()
        data = _make_ohlcv(100)
        feats = eng.compute(data)
        assert (feats["volume_ratio"] >= 0).all()


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_load_from_dataframe(self):
        from src.environment.data_loader import DataLoader
        loader = DataLoader(["SPY"], "2015-01-01", "2023-12-31")
        data = _make_ohlcv()
        result = loader.load_from_dataframe(data, ticker="TEST")
        assert "TEST" in result
        pd.testing.assert_frame_equal(result["TEST"], data)

    def test_train_test_split_ratio(self):
        from src.environment.data_loader import DataLoader
        loader = DataLoader(["SPY"], "2015-01-01", "2023-12-31", train_ratio=0.8)
        data = _make_ohlcv(100)
        loader.load_from_dataframe(data, ticker="TEST")
        train, test = loader.get_train_test_split("TEST")
        assert len(train) == 80
        assert len(test)  == 20

    def test_train_test_split_no_overlap(self):
        from src.environment.data_loader import DataLoader
        loader = DataLoader(["SPY"], "2015-01-01", "2023-12-31")
        data = _make_ohlcv(100)
        loader.load_from_dataframe(data, ticker="TEST")
        train, test = loader.get_train_test_split("TEST")
        assert set(train.index).isdisjoint(set(test.index))

    def test_missing_ticker_raises(self):
        from src.environment.data_loader import DataLoader
        loader = DataLoader(["SPY"], "2015-01-01", "2023-12-31")
        with pytest.raises(KeyError):
            loader.get_train_test_split("NONEXISTENT")

    def test_missing_column_raises(self):
        from src.environment.data_loader import DataLoader
        loader = DataLoader(["SPY"], "2015-01-01", "2023-12-31")
        bad_df = pd.DataFrame({"Close": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing columns"):
            loader.load_from_dataframe(bad_df, ticker="BAD")


# ---------------------------------------------------------------------------
# TradingEnv  (requires gymnasium)
# ---------------------------------------------------------------------------

gymnasium = pytest.importorskip("gymnasium", reason="gymnasium not installed")


class TestTradingEnv:
    @pytest.fixture
    def env(self):
        from src.environment.trading_env import TradingEnv
        return TradingEnv(_make_ohlcv(200), lookback_window=20)

    def test_observation_space_shape(self, env):
        # lookback_window * n_features + 4 portfolio + 4 regime = 20*10 + 4 + 4 = 208
        assert env.observation_space.shape == (208,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (4,)

    def test_reset_returns_observation(self, env):
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_reset_observation_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_step_returns_valid_tuple(self, env):
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_shape(self, env):
        env.reset()
        action = env.action_space.sample()
        obs, *_ = env.step(action)
        assert obs.shape == env.observation_space.shape

    def test_episode_terminates(self, env):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if steps > 10_000:
                pytest.fail("Episode did not terminate within 10k steps")
        assert done

    def test_portfolio_value_in_info(self, env):
        env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "portfolio_value" in info
        assert info["portfolio_value"] > 0

    def test_action_softmax_normalisation(self, env):
        """All-zero action should not crash and produce a valid step."""
        env.reset()
        action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)

    def test_seed_reproducibility(self):
        from src.environment.trading_env import TradingEnv
        data = _make_ohlcv(200)
        env1 = TradingEnv(data)
        env2 = TradingEnv(data)
        obs1, _ = env1.reset(seed=0)
        obs2, _ = env2.reset(seed=0)
        np.testing.assert_array_equal(obs1, obs2)

    def test_custom_initial_cash(self):
        from src.environment.trading_env import TradingEnv
        data = _make_ohlcv(200)
        env = TradingEnv(data, initial_cash=50_000)
        assert env.initial_cash == 50_000

    def test_custom_lookback_window(self):
        from src.environment.trading_env import TradingEnv
        data = _make_ohlcv(200)
        env = TradingEnv(data, lookback_window=10)
        # obs_dim = 10*10 + 4 + 4 = 108
        assert env.observation_space.shape == (108,)
