"""
Tests for RL agent modules.

Heavy dependencies (torch, stable-baselines3) are skipped gracefully when
not installed.
"""
import os
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

    def test_tensorboard_log_threaded_to_sb3(self):
        """``tensorboard_log`` from agent config must reach SB3's PPO model."""
        pytest.importorskip("stable_baselines3")
        from src.agents.ppo_agent import PPOAgent
        env = _make_env()
        tb_dir = "/tmp/regime-rl-tb-test-marker"
        agent = PPOAgent(env, config={"tensorboard_log": tb_dir})
        assert agent.model.tensorboard_log == tb_dir


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


# ---------------------------------------------------------------------------
# TQCAgent (distributional / risk-averse SAC)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.importorskip("sb3_contrib", reason="sb3_contrib not installed"),
    reason="sb3_contrib not installed",
)
class TestTQCAgent:
    def test_instantiation_and_act(self):
        from src.agents.tqc_agent import TQCAgent
        env = _make_env()
        agent = TQCAgent(
            env,
            config={"top_quantiles_to_drop_per_net": 4, "n_quantiles": 25},
        )
        obs, _ = env.reset()
        action = agent.act(obs)
        assert action.shape == env.action_space.shape

    def test_invalid_top_drop_raises(self):
        from src.agents.tqc_agent import TQCAgent
        env = _make_env()
        with pytest.raises(ValueError, match="top_quantiles_to_drop_per_net"):
            TQCAgent(env, config={"n_quantiles": 25, "top_quantiles_to_drop_per_net": 25})

    def test_top_drop_threaded_to_model(self):
        """The CVaR knob must reach sb3-contrib's TQC instance."""
        from src.agents.tqc_agent import TQCAgent
        env = _make_env()
        agent = TQCAgent(env, config={"top_quantiles_to_drop_per_net": 5})
        assert agent.model.top_quantiles_to_drop_per_net == 5


# ---------------------------------------------------------------------------
# HierarchicalAgent (manager + worker ensemble)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed"),
    reason="torch not installed",
)
class TestHierarchicalAgent:
    """Lightweight sanity tests using mock workers (no SB3 dependency)."""

    class _MockWorker:
        """Returns a fixed action; satisfies BaseAgent surface needed."""
        def __init__(self, action):
            self._action = np.asarray(action, dtype=np.float32)

        def act(self, _obs):
            return self._action.copy()

        def save(self, path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.save(path + ".npy", self._action)

        def load(self, path):
            self._action = np.load(path + ".npy")

    def test_act_routes_to_chosen_worker(self):
        from src.agents.hierarchical_agent import HierarchicalAgent
        env = _make_env()
        action_dim = env.action_space.shape[0]
        # Two workers with distinct actions
        w0 = self._MockWorker(np.zeros(action_dim, dtype=np.float32))
        w1 = self._MockWorker(np.ones(action_dim, dtype=np.float32))
        agent = HierarchicalAgent(env, workers=[w0, w1], config={"seed": 0})

        obs, _ = env.reset()
        chosen = agent.select_worker(obs)
        action = agent.act(obs)
        np.testing.assert_array_equal(action, [w0, w1][chosen]._action)

    def test_manager_updates_change_routing(self, tmp_path=None):
        """After REINFORCE updates with one worker forced to be 'good',
        the manager should converge to picking that worker more often."""
        from src.agents.hierarchical_agent import HierarchicalAgent
        import torch

        env = _make_env()
        action_dim = env.action_space.shape[0]
        # One worker outputs full long, the other full flat.
        long_action = np.array([1.0] + [0.0] * (action_dim - 1), dtype=np.float32)
        flat_action = np.zeros(action_dim, dtype=np.float32)
        w_long = self._MockWorker(long_action)
        w_flat = self._MockWorker(flat_action)

        agent = HierarchicalAgent(
            env,
            workers=[w_long, w_flat],
            config={"seed": 0, "manager_lr": 1e-2, "entropy_coef": 0.0},
        )

        # Manually feed positive reward to worker 0 picks, negative to worker 1.
        # We simulate one full update cycle.
        for _ in range(50):
            obs, _ = env.reset()
            log_p, idx = agent._sample_worker(obs)
            agent._log_probs.append(log_p)
            agent._rewards.append(1.0 if idx == 0 else -1.0)
        agent._update_manager()

        # After biased training, sampling distribution should favour worker 0.
        with torch.no_grad():
            obs, _ = env.reset()
            logits = agent.manager(agent._to_tensor(obs))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        assert probs[0] > probs[1], f"manager did not learn preference: {probs}"
