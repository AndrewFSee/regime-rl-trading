"""Helpers for persisting TradingEnv normalizer state alongside SB3 models."""
from __future__ import annotations

import os
from typing import Any

import numpy as np


def _normalizer_path(model_path: str) -> str:
    # SB3 may or may not have appended .zip; treat the base path consistently.
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    return base + ".norm.npz"


def _unwrap(env: Any) -> Any:
    """Best-effort unwrapping for VecEnv / Gymnasium wrappers down to TradingEnv."""
    # SB3 VecEnv exposes envs[0]
    inner = env
    if hasattr(inner, "envs") and len(getattr(inner, "envs", [])) > 0:
        inner = inner.envs[0]
    # Gymnasium wrappers expose .unwrapped
    if hasattr(inner, "unwrapped"):
        inner = inner.unwrapped
    return inner


def _save_env_normalizer(env: Any, model_path: str) -> None:
    inner = _unwrap(env)
    getter = getattr(inner, "get_normalizer_state", None)
    if getter is None:
        return
    state = getter()
    if state is None:
        return
    path = _normalizer_path(model_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        mean=state["mean"],
        var=state["var"],
        count=np.float64(state["count"]),
        clip=np.float64(state["clip"]),
        training=np.bool_(state["training"]),
    )


def _load_env_normalizer(env: Any, model_path: str, freeze: bool = True) -> bool:
    inner = _unwrap(env)
    loader = getattr(inner, "load_normalizer_state", None)
    if loader is None:
        return False
    path = _normalizer_path(model_path)
    if not os.path.exists(path):
        return False
    data = np.load(path, allow_pickle=False)
    state = {
        "mean": np.asarray(data["mean"]),
        "var": np.asarray(data["var"]),
        "count": float(data["count"]),
        "clip": float(data["clip"]),
        "training": bool(data["training"]),
    }
    loader(state)
    if freeze:
        setter = getattr(inner, "set_training_mode", None)
        if setter is not None:
            setter(False)
    return True
