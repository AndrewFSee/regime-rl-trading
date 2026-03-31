"""
Tests for RL agent modules.

Heavy dependencies (torch, stable-baselines3) are skipped gracefully when
not installed.
"""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, seed: int = 7) -> "pd.DataFrame":
    import pandas as pd
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


def _make_env():
    gymnasium = pytest.importorskip("gymnasium", reason="gymnasium not installed")
    from src.environment.trading_env import TradingEnv
    return TradingEnv(_make_ohlcv(200), lookback_window=20)


# ---------------------------------------------------------------------------
# BaseAgent interface
# ---------------------------------------------------------------------------

def test_base_agent_is_abstract():
    from src.agents.base import BaseAgent
    import inspect
    assert inspect.isabstract(BaseAgent)


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.importorskip("stable_baselines3", reason="stable_baselines3 not installed"),
    reason="stable_baselines3 not installed",
)
class TestPPOAgent:
    @pytest.fixture
    def agent_and_env(self):
        pytest.importorskip("stable_baselines3")
        from src.agents.ppo_agent import PPOAgent
        env = _make_env()
        agent = PPOAgent(env, config={"learning_rate": 3e-4, "gamma": 0.99, "batch_size": 64})
        return agent, env

    def test_instantiation(self, agent_and_env):
        agent, _ = agent_and_env
        from src.agents.ppo_agent import PPOAgent
        assert isinstance(agent, PPOAgent)

    def test_act_returns_valid_action(self, agent_and_env):
        agent, env = agent_and_env
        obs, _ = env.reset()
        action = agent.act(obs)
        assert action.shape == env.action_space.shape

    def test_act_deterministic(self, agent_and_env):
        agent, env = agent_and_env
        obs, _ = env.reset()
        a1 = agent.act(obs)
        a2 = agent.act(obs)
        np.testing.assert_array_equal(a1, a2)

    def test_save_load(self, agent_and_env, tmp_path):
        agent, env = agent_and_env
        save_path = str(tmp_path / "ppo_model")
        agent.save(save_path)
        # load into a new agent instance
        from src.agents.ppo_agent import PPOAgent
        new_agent = PPOAgent(env, config={})
        new_agent.load(save_path)
        obs, _ = env.reset()
        action = new_agent.act(obs)
        assert action.shape == env.action_space.shape


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.importorskip("stable_baselines3", reason="stable_baselines3 not installed"),
    reason="stable_baselines3 not installed",
)
class TestDQNAgent:
    @pytest.fixture
    def agent_and_env(self):
        pytest.importorskip("stable_baselines3")
        from src.agents.dqn_agent import DQNAgent
        env = _make_env()
        agent = DQNAgent(env, config={"learning_rate": 1e-4, "gamma": 0.99})
        return agent, env

    def test_instantiation(self, agent_and_env):
        agent, _ = agent_and_env
        from src.agents.dqn_agent import DQNAgent
        assert isinstance(agent, DQNAgent)

    def test_act_returns_weight_vector(self, agent_and_env):
        agent, env = agent_and_env
        obs, _ = env.reset()
        action = agent.act(obs)
        # DQN returns one-hot weight vector of length n_strategies=4
        assert action.shape == (4,)
        assert action.sum() == pytest.approx(1.0)
        assert (action >= 0).all()

    def test_act_is_one_hot(self, agent_and_env):
        """Exactly one strategy should be selected."""
        agent, env = agent_and_env
        obs, _ = env.reset()
        action = agent.act(obs)
        assert int((action == 1.0).sum()) == 1


# ---------------------------------------------------------------------------
# MetaAgent (skipped unless SB3 available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.importorskip("stable_baselines3", reason="stable_baselines3 not installed"),
    reason="stable_baselines3 not installed",
)
class TestMetaAgent:
    @pytest.fixture
    def meta_agent_and_env(self):
        pytest.importorskip("stable_baselines3")
        from src.agents.meta_agent import MetaAgent
        env = _make_env()
        agent = MetaAgent(env, config={})
        return agent, env

    def test_instantiation(self, meta_agent_and_env):
        from src.agents.meta_agent import MetaAgent
        agent, _ = meta_agent_and_env
        assert isinstance(agent, MetaAgent)

    def test_has_one_agent_per_regime(self, meta_agent_and_env):
        from src.regime_detection.base import MarketRegime
        agent, _ = meta_agent_and_env
        assert set(agent.agents.keys()) == set(MarketRegime)

    def test_act_returns_valid_action(self, meta_agent_and_env):
        agent, env = meta_agent_and_env
        obs, _ = env.reset()
        action = agent.act(obs)
        assert action.shape == env.action_space.shape
