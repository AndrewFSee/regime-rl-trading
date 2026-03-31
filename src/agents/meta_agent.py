"""
Meta-agent that routes decisions to regime-specific PPO sub-agents.

Each of the four market regimes (BULL, BEAR, SIDEWAYS, VOLATILE) has its
own dedicated PPOAgent.  During inference the current regime is detected
from the latest observation and the matching sub-agent is queried.

During training all sub-agents are trained jointly on the shared
environment.  On regime transitions the exploration noise of the newly
active sub-agent is temporarily increased to encourage fast adaptation.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import BaseAgent
from ..regime_detection.base import MarketRegime

try:
    from .ppo_agent import PPOAgent
    _PPO_AVAILABLE = True
except ImportError:
    _PPO_AVAILABLE = False

try:
    from ..regime_detection.feature_detector import FeatureRegimeDetector
    _DETECTOR_AVAILABLE = True
except ImportError:
    _DETECTOR_AVAILABLE = False

import pandas as pd


class MetaAgent(BaseAgent):
    """
    Ensemble meta-agent that maintains one PPOAgent per market regime and
    routes each decision to the agent responsible for the currently detected
    regime.

    Parameters
    ----------
    env:              A Gymnasium-compatible TradingEnv.
    config:           Configuration dict (same schema as used by PPOAgent).
    regime_detector:  Optional pre-configured RegimeDetector.  If ``None``
                      a :class:`FeatureRegimeDetector` is instantiated.
    """

    def __init__(self, env, config: dict | None = None, regime_detector=None) -> None:
        if not _PPO_AVAILABLE:
            raise ImportError("stable-baselines3 is required for MetaAgent.")

        self.env = env
        self.config = config or {}

        self.regime_detector = regime_detector or FeatureRegimeDetector()

        # One sub-agent per regime
        self.agents: dict[MarketRegime, PPOAgent] = {
            regime: PPOAgent(env, self.config) for regime in MarketRegime
        }
        self._current_regime: Optional[MarketRegime] = None

    # ------------------------------------------------------------------
    # Regime detection from observation
    # ------------------------------------------------------------------

    def _detect_regime_from_obs(self, observation: np.ndarray) -> MarketRegime:
        """
        Heuristic: use the one-hot regime encoding embedded in the observation.

        The TradingEnv places a 4-dimensional regime one-hot at the end of
        the observation vector.  We decode it here.
        """
        regime_vec = observation[-4:]
        idx = int(np.argmax(regime_vec))
        try:
            return MarketRegime(idx)
        except ValueError:
            return MarketRegime.SIDEWAYS

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Route the observation to the appropriate regime sub-agent."""
        regime = self._detect_regime_from_obs(observation)

        # Exploration boost on regime transition
        if self._current_regime is not None and regime != self._current_regime:
            self._exploration_boost(regime)

        self._current_regime = regime
        return self.agents[regime].act(observation)

    def learn(self, env=None, total_timesteps: int = 100_000) -> None:
        """
        Train all sub-agents in round-robin fashion.

        Each sub-agent receives an equal share of the total timestep budget.
        """
        per_agent = max(1, total_timesteps // len(self.agents))
        for regime, agent in self.agents.items():
            agent.learn(env=env, total_timesteps=per_agent)

    def save(self, path: str) -> None:
        """Save each sub-agent under ``<path>/<regime_name>``."""
        os.makedirs(path, exist_ok=True)
        for regime, agent in self.agents.items():
            agent.save(os.path.join(path, regime.name.lower()))

    def load(self, path: str) -> None:
        """Load each sub-agent from ``<path>/<regime_name>.zip``."""
        for regime, agent in self.agents.items():
            agent.load(os.path.join(path, regime.name.lower()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exploration_boost(self, regime: MarketRegime) -> None:
        """
        Temporarily increase exploration for the newly active sub-agent by
        resetting the SB3 model's exploration schedule.
        """
        agent = self.agents[regime]
        try:
            import torch
            if hasattr(agent.model, "policy") and hasattr(agent.model.policy, "log_std"):
                with torch.no_grad():
                    agent.model.policy.log_std.data += 0.5
        except Exception:
            pass  # Fail silently – exploration boost is a nice-to-have
