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

    def test_walk_forward_splits_shape(self):
        from src.environment.data_loader import DataLoader
        loader = DataLoader(["SPY"], "2015-01-01", "2023-12-31")
        data = _make_ohlcv(300)
        loader.load_from_dataframe(data, ticker="WF")
        splits = loader.get_walk_forward_splits(
            "WF", train_size=100, test_size=50, step=50, embargo=0
        )
        # (300 - 100) / 50 = 4 folds
        assert len(splits) == 4
        for tr, te in splits:
            assert len(tr) == 100
            assert len(te) == 50
            assert set(tr.index).isdisjoint(set(te.index))

    def test_walk_forward_splits_embargo_drops_bars(self):
        from src.environment.data_loader import DataLoader
        loader = DataLoader(["SPY"], "2015-01-01", "2023-12-31")
        data = _make_ohlcv(200)
        loader.load_from_dataframe(data, ticker="WF2")
        splits = loader.get_walk_forward_splits(
            "WF2", train_size=100, test_size=20, step=20, embargo=10
        )
        for tr, te in splits:
            # Test window starts AFTER the embargo gap
            assert te.index[0] - tr.index[-1] >= 10


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
        # lookback_window * n_obs_features + 4 portfolio + 4 regime = 20*10 + 4 + 4 = 208
        assert env.observation_space.shape == (208,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (5,)

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
        action = np.zeros(5, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)

    def test_hard_selection_guardrail_restricts_bear_choices(self):
        from src.environment.trading_env import TradingEnv
        from src.regime_detection.base import MarketRegime

        env = TradingEnv(
            _make_ohlcv(200),
            lookback_window=20,
            hard_selection=True,
            regime_guardrail=True,
        )
        env.reset()
        env._get_current_regime = lambda: MarketRegime.BEAR

        action = np.array([0.1, 0.95, 0.2, 0.3, 1.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert info["requested_strategy"] == 1
        assert info["chosen_strategy"] == 3
        assert info["guardrail_active"] is True
        np.testing.assert_array_equal(info["strategy_weights"], np.array([0.0, 0.0, 0.0, 1.0]))

    def test_soft_mode_guardrail_masks_forbidden_strategies(self):
        """Soft-blend mode must zero out forbidden-strategy weights when the
        guardrail is active (action masking, not reward shaping)."""
        from src.environment.trading_env import TradingEnv
        from src.regime_detection.base import MarketRegime

        env = TradingEnv(
            _make_ohlcv(200),
            lookback_window=20,
            hard_selection=False,
            regime_guardrail=True,
        )
        env.reset()
        env._get_current_regime = lambda: MarketRegime.VOLATILE

        # Logits favour MeanReversion (idx 1) and Breakout (idx 2), both
        # forbidden in VOLATILE. After masking only Momentum (0) and
        # Defensive (3) should receive any weight.
        action = np.array([0.0, 5.0, 5.0, 0.0, 1.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        weights = info["strategy_weights"]
        assert info["guardrail_active"] is True
        assert weights[1] == 0.0  # MeanReversion masked
        assert weights[2] == 0.0  # Breakout masked
        assert weights[0] + weights[3] == pytest.approx(1.0)
        assert weights[0] == pytest.approx(weights[3])  # equal logits → equal mass

    def test_risk_off_exposure_cap_overrides_floor(self):
        from src.environment.trading_env import TradingEnv
        from src.regime_detection.base import MarketRegime

        env = TradingEnv(
            _make_ohlcv(200),
            lookback_window=20,
            hard_selection=True,
            exposure_floor=0.5,
            risk_off_exposure_cap=0.25,
        )
        env.reset()
        env._get_current_regime = lambda: MarketRegime.VOLATILE

        action = np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert info["effective_exposure"] == pytest.approx(0.25)
        assert info["raw_exposure"] == pytest.approx(1.0)

    def test_risk_off_strategy_penalty_hits_long_momentum_in_bear(self):
        from src.environment.trading_env import TradingEnv
        from src.regime_detection.base import MarketRegime

        class _DummyStrategy:
            def __init__(self, action: float):
                self._action = action

            def generate_signal(self, _features):
                return type("Signal", (), {"action": self._action})()

        env = TradingEnv(
            _make_ohlcv(200),
            lookback_window=20,
            hard_selection=True,
            risk_off_strategy_penalty=0.2,
        )
        env._strategies = [
            _DummyStrategy(1.0),
            _DummyStrategy(-1.0),
            _DummyStrategy(1.0),
            _DummyStrategy(0.2),
        ]
        env.reset()
        env._get_current_regime = lambda: MarketRegime.BEAR

        action = np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert info["chosen_strategy"] == 0
        assert info["position"] == pytest.approx(np.tanh(5.0), rel=1e-6)
        assert info["risk_off_strategy_penalty_applied"] == pytest.approx(0.2 * np.tanh(5.0), rel=1e-6)

    def test_risk_off_strategy_penalty_skips_defensive_long_in_bear(self):
        from src.environment.trading_env import TradingEnv
        from src.regime_detection.base import MarketRegime

        class _DummyStrategy:
            def __init__(self, action: float):
                self._action = action

            def generate_signal(self, _features):
                return type("Signal", (), {"action": self._action})()

        env = TradingEnv(
            _make_ohlcv(200),
            lookback_window=20,
            hard_selection=True,
            risk_off_strategy_penalty=0.2,
        )
        env._strategies = [
            _DummyStrategy(1.0),
            _DummyStrategy(-1.0),
            _DummyStrategy(1.0),
            _DummyStrategy(0.2),
        ]
        env.reset()
        env._get_current_regime = lambda: MarketRegime.BEAR

        action = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert info["chosen_strategy"] == 3
        assert info["position"] > 0.0
        assert info["risk_off_strategy_penalty_applied"] == pytest.approx(0.0)

    def test_drawdown_penalty_applies_only_above_threshold(self):
        """drawdown_penalty should only fire once drawdown exceeds the
        threshold; below threshold the reward must equal the no-penalty case."""
        from src.environment.trading_env import TradingEnv
        data = _make_ohlcv(200)

        common = dict(
            lookback_window=20,
            return_weight=1.0,  # use raw return so DSR EMA noise is gone
            normalize_obs=False,
            reward_scaling=1.0,
        )
        env_off = TradingEnv(data, **common, drawdown_penalty=0.0)
        env_on = TradingEnv(
            data, **common, drawdown_penalty=10.0, drawdown_threshold=0.0
        )
        env_off.reset(seed=42)
        env_on.reset(seed=42)

        # Force a losing position: full long while market falls.
        action = np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        rewards_off, rewards_on, drawdowns = [], [], []
        for _ in range(20):
            _, r_off, _, _, _ = env_off.step(action)
            _, r_on, _, _, info_on = env_on.step(action)
            rewards_off.append(r_off)
            rewards_on.append(r_on)
            drawdowns.append(info_on["drawdown"])

        # On every step where drawdown > 0, the on-env must have a STRICTLY
        # smaller reward than the off-env. When dd == 0, rewards must match.
        for r_off, r_on, dd in zip(rewards_off, rewards_on, drawdowns):
            if dd > 0:
                assert r_on < r_off
            else:
                assert r_on == pytest.approx(r_off)

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

    def test_strategy_signals_use_only_past_features(self):
        """Regression test for the lookahead leak fix.

        Strategies must compute their signal from features through
        ``close[step_idx - 1]`` only. If a future bar is mutated, the signal
        emitted on the step that earns that bar's return must be unchanged.
        """
        from src.environment.trading_env import TradingEnv

        data_a = _make_ohlcv(200, seed=1)
        data_b = data_a.copy()
        # Mutate the bar at index `target_idx` -- this is the bar whose return
        # will be earned at step_idx == target_idx. Its features must NOT be
        # visible to the strategy on that step.
        target_idx = 50
        data_b.loc[target_idx, "Close"] *= 1.10
        data_b.loc[target_idx, "High"] *= 1.10
        data_b.loc[target_idx, "Low"] *= 1.10
        data_b.loc[target_idx, "Open"] *= 1.10

        env_a = TradingEnv(data_a, lookback_window=20, normalize_obs=False)
        env_b = TradingEnv(data_b, lookback_window=20, normalize_obs=False)
        env_a.reset()
        env_b.reset()
        # Drive both envs to step_idx == target_idx.
        action = np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(target_idx - env_a._step_idx):
            env_a.step(action)
            env_b.step(action)
        assert env_a._step_idx == target_idx
        _, _, _, _, info_a = env_a.step(action)
        _, _, _, _, info_b = env_b.step(action)
        # Signals must be identical despite the future bar being different.
        np.testing.assert_array_equal(
            info_a["strategy_signals"], info_b["strategy_signals"]
        )

    def test_step_exposes_net_return(self):
        from src.environment.trading_env import TradingEnv
        env = TradingEnv(_make_ohlcv(200), transaction_cost=0.001, slippage_factor=0.0)
        env.reset()
        # Force a turnover so transaction cost fires.
        action = np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "net_return" in info
        assert "transaction_cost" in info
        # net = gross - tc - slippage
        expected = (
            info["portfolio_return"] - info["transaction_cost"] - info["slippage"]
        )
        assert info["net_return"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
