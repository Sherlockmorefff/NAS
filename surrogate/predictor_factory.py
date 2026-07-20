"""Central construction and type-safe loading for Phase4 accuracy surrogates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from surrogate.accuracy_gp import AccuracyGPPredictor
from surrogate.checkpoint_io import torch_load_compat
from surrogate.dkl_accuracy_gp import DKLAccuracyGPPredictor


SURROGATE_TYPES = ("exact_gp", "dkl_gp")


def fit_accuracy_predictor(
    train_X_raw: Any,
    train_Y: Any,
    *,
    surrogate_type: str = "exact_gp",
    **kwargs: Any,
) -> AccuracyGPPredictor:
    """Fit the selected predictor without changing the exact-GP defaults."""

    if surrogate_type == "exact_gp":
        return AccuracyGPPredictor.fit_offline(train_X_raw, train_Y, **kwargs)
    if surrogate_type == "dkl_gp":
        return DKLAccuracyGPPredictor.fit_offline(train_X_raw, train_Y, **kwargs)
    raise ValueError(f"surrogate_type must be one of {SURROGATE_TYPES}, got {surrogate_type!r}")


def checkpoint_surrogate_type(path: str | Path) -> str:
    """Return checkpoint type, treating legacy exact checkpoints as exact_gp."""

    payload = torch_load_compat(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"surrogate checkpoint must contain a mapping: {path}")
    return str(payload.get("surrogate_type", "exact_gp"))


def load_accuracy_predictor(
    path: str | Path,
    *,
    surrogate_type: str = "exact_gp",
    **kwargs: Any,
) -> AccuracyGPPredictor:
    """Load only a checkpoint whose declared type matches the requested type."""

    if surrogate_type not in SURROGATE_TYPES:
        raise ValueError(f"surrogate_type must be one of {SURROGATE_TYPES}, got {surrogate_type!r}")
    actual_type = checkpoint_surrogate_type(path)
    if actual_type != surrogate_type:
        raise ValueError(
            f"surrogate checkpoint type mismatch: expected {surrogate_type!r}, actual {actual_type!r}"
        )
    loader = AccuracyGPPredictor if surrogate_type == "exact_gp" else DKLAccuracyGPPredictor
    return loader.load(path, **kwargs)
