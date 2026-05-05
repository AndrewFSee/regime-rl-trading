"""Hierarchical policy: a learned manager over K continuous workers.

Architecturally distinct from :class:`MetaAgent`:

* **MetaAgent** routes each step to the worker matching the *env-detected*
  regime (BULL/BEAR/SIDEWAYS/VOLATILE). The router is a hard, hand-coded
  function of the obs's regime one-hot.
* **HierarchicalAgent** uses a small *trainable* manager network that maps
  the full observation to a categorical distribution over the K workers.
  Manager is updated with REINFORCE on the env reward; each worker is a
  fully-fledged continuous policy (SAC by default) that only sees
  trajectories on which it was selected.

This gives the agent flexibility to discover alternative routings the
hand-coded regime classifier misses. Kept deliberately minimal — workers
do not currently train online here; train them separately first via the
flat SAC/TQC pipeline, then drop them in via :meth:`set_workers`. The
manager learning loop is intentionally tiny (single REINFORCE update per
batch) to keep dependencies small and the implementation auditable.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from .base import BaseAgent

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _ManagerNet(nn.Module if _TORCH_AVAILABLE else object):
    """Small MLP outputting unnormalised logits over K workers."""

    def __init__(self, obs_dim: int, n_workers: int, hidden: int = 64) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for HierarchicalAgent.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_workers),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class HierarchicalAgent(BaseAgent):
    """Manager + worker ensemble with a REINFORCE-trained router.

    Parameters
    ----------
    env:        Gymnasium-compatible environment.
    workers:    Sequence of K already-instantiated worker BaseAgents
                (e.g. SACAgent / TQCAgent / PPOAgent). All must share the
                env's action space.
    config:     Optional dict with keys:
                  manager_lr     (float, default 1e-3)
                  hidden         (int,   default 64)
                  entropy_coef   (float, default 0.01)
                  device         (str,   default "cpu")
                  seed           (int|None, default None)
    """

    def __init__(
        self,
        env,
        workers: Sequence[BaseAgent],
        config: Optional[dict] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for HierarchicalAgent.")
        if len(workers) < 2:
            raise ValueError("HierarchicalAgent needs at least 2 workers.")

        self.env = env
        self.workers = list(workers)
        self.n_workers = len(self.workers)
        self._config = dict(config or {})

        seed = self._config.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        obs_dim = int(np.prod(env.observation_space.shape))
        self.device = torch.device(self._config.get("device", "cpu"))
        self.manager = _ManagerNet(
            obs_dim=obs_dim,
            n_workers=self.n_workers,
            hidden=int(self._config.get("hidden", 64)),
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.manager.parameters(),
            lr=float(self._config.get("manager_lr", 1e-3)),
        )
        self.entropy_coef = float(self._config.get("entropy_coef", 0.01))

        # Rolling REINFORCE buffer (per-step log-probs and rewards)
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Manager picks worker (argmax under eval); worker chooses action."""
        worker_idx = self._select_worker(observation, deterministic=True)
        return self.workers[worker_idx].act(observation)

    def select_worker(self, observation: np.ndarray) -> int:
        """Public helper exposing the manager's choice (for analysis/tests)."""
        return self._select_worker(observation, deterministic=True)

    def learn(self, total_timesteps: int = 10_000) -> None:
        """Roll out the env, training ONLY the manager via REINFORCE.

        Workers are assumed pre-trained (or being updated externally).
        Manager update happens once per episode end.
        """
        obs, _ = self.env.reset()
        steps = 0
        while steps < total_timesteps:
            log_p, idx = self._sample_worker(obs)
            action = self.workers[idx].act(obs)
            obs, reward, terminated, truncated, _info = self.env.step(action)

            self._log_probs.append(log_p)
            self._rewards.append(float(reward))
            steps += 1

            if terminated or truncated:
                self._update_manager()
                obs, _ = self.env.reset()

        # Final flush
        if self._rewards:
            self._update_manager()

    def save(self, path: str) -> None:
        """Save manager weights + each worker under a subdir."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.manager.state_dict(), os.path.join(path, "manager.pt"))
        for i, w in enumerate(self.workers):
            w.save(os.path.join(path, f"worker_{i}"))

    def load(self, path: str) -> None:
        self.manager.load_state_dict(
            torch.load(os.path.join(path, "manager.pt"), map_location=self.device)
        )
        for i, w in enumerate(self.workers):
            w.load(os.path.join(path, f"worker_{i}"))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_worker(self, observation: np.ndarray, deterministic: bool) -> int:
        with torch.no_grad():
            logits = self.manager(self._to_tensor(observation))
        if deterministic:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _sample_worker(self, observation: np.ndarray):
        """Stochastic sample with grad-tracking log-prob for REINFORCE."""
        logits = self.manager(self._to_tensor(observation))
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        return dist.log_prob(idx), int(idx.item())

    def _update_manager(self) -> None:
        if not self._rewards:
            return
        # Compute discounted-return baseline (simple mean baseline keeps
        # variance manageable without a learned critic).
        returns = torch.tensor(self._rewards, device=self.device, dtype=torch.float32)
        baseline = returns.mean()
        advantages = returns - baseline

        log_probs = torch.stack(self._log_probs)
        loss = -(log_probs * advantages).mean()

        # Entropy regularisation pulled from a fresh forward to stay simple.
        # (We could cache distributions; this is fine for MVP scale.)
        if self.entropy_coef > 0.0:
            # Approximate entropy bonus from the empirical action distribution
            with torch.no_grad():
                pass  # entropy bonus folded into the policy gradient term below
            # Use stored log_probs directly: H ≈ -E[log π(a|s)]
            entropy = -log_probs.mean()
            loss = loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._log_probs.clear()
        self._rewards.clear()

    def _to_tensor(self, observation: np.ndarray) -> "torch.Tensor":
        return torch.as_tensor(
            np.asarray(observation, dtype=np.float32), device=self.device
        )
