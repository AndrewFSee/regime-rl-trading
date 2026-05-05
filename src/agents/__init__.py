"""Agents sub-package."""
from .base import BaseAgent
from .ppo_agent import PPOAgent
from .dqn_agent import DQNAgent
from .meta_agent import MetaAgent

try:
    from .sac_agent import SACAgent  # noqa: F401
except ImportError:
    SACAgent = None  # type: ignore[assignment]

try:
    from .tqc_agent import TQCAgent  # noqa: F401
except ImportError:
    TQCAgent = None  # type: ignore[assignment]

try:
    from .hierarchical_agent import HierarchicalAgent  # noqa: F401
except ImportError:
    HierarchicalAgent = None  # type: ignore[assignment]

__all__ = [
    "BaseAgent", "PPOAgent", "DQNAgent", "MetaAgent",
    "SACAgent", "TQCAgent", "HierarchicalAgent",
]
