"""Environment sub-package."""
from .features import FeatureEngineer, FEATURE_NAMES
from .data_loader import DataLoader
from .trading_env import TradingEnv

try:
    from .multi_asset_env import MultiAssetTradingEnv  # noqa: F401
except ImportError:
    MultiAssetTradingEnv = None  # type: ignore[assignment]

__all__ = [
    "FeatureEngineer", "FEATURE_NAMES", "DataLoader",
    "TradingEnv", "MultiAssetTradingEnv",
]
