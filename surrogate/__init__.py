"""Offline and online accuracy surrogate support for Phase4.

Imports are lazy so metric-only tooling remains usable without the GP stack.
"""

from typing import Any

__all__ = [
    "AccuracyGPPredictor",
    "DKLAccuracyGPPredictor",
    "DeepFeatureKernel",
    "SmallFeatureExtractor",
    "GPConvergenceMonitor",
    "denormalize_search_vector",
    "normalize_search_vector",
    "prediction_metrics",
]


def __getattr__(name: str) -> Any:
    if name in {"AccuracyGPPredictor", "denormalize_search_vector", "normalize_search_vector"}:
        from . import accuracy_gp

        return getattr(accuracy_gp, name)
    if name in {"DKLAccuracyGPPredictor", "DeepFeatureKernel", "SmallFeatureExtractor"}:
        from . import dkl_accuracy_gp

        return getattr(dkl_accuracy_gp, name)
    if name in {"GPConvergenceMonitor", "prediction_metrics"}:
        from . import metrics

        return getattr(metrics, name)
    raise AttributeError(name)
