"""Abstract base class for RL agents."""
from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """Base interface for all RL trading agents."""

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Select an action given an observation."""
        ...

    @abstractmethod
    def learn(self, env, total_timesteps: int) -> None:
        """Train the agent on the given environment."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the agent to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the agent from disk."""
        ...
