"""Pure CPU utilities for NAS-adapted WGMM-clustered TED initialization.

The implementation is independent of BOOM-Explorer source code.  It follows
the sequential transductive experimental design equations in Yu, Bi & Tresp
(ICML 2006) and uses latent clusters as a NAS-specific substitute for the
hand-authored microarchitecture categories in BOOM-Explorer.  It is not a
literal reproduction of BOOM-Explorer MicroAL.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from surrogate.accuracy_gp import normalize_search_vector
from weighted_diag_gmm_init import WeightedDiagonalGMM, standardize_apply, standardize_fit


WGMM_TED_FORMAT_VERSION = 1
WGMM_ASSIGNMENT_SOURCES = ("checkpoint", "gmm_fit_pool")
QUOTA_MODES = ("equal", "proportional", "hybrid")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def candidate_fingerprint(z_search: Any) -> str:
    return fingerprint_array(np.asarray(z_search, dtype=np.float32).reshape(1, -1))


def canonical_json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json_dump(value: Any, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class DiagonalGMMParameters:
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    source: str
    source_fingerprint: str
    standardize_mean: np.ndarray | None = None
    standardize_std: np.ndarray | None = None

    def validate(self, arch_dim: int) -> "DiagonalGMMParameters":
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        means = np.asarray(self.means, dtype=np.float64)
        variances = np.asarray(self.variances, dtype=np.float64)
        if means.ndim != 2 or means.shape[1] != int(arch_dim):
            raise ValueError(
                f"WGMM means must have shape (components, {arch_dim}), got {means.shape}"
            )
        if weights.shape != (means.shape[0],):
            raise ValueError("WGMM weights/component count mismatch")
        if variances.shape != means.shape:
            raise ValueError(
                "only diagonal WGMM covariance is supported; variances must match means"
            )
        if not np.isfinite(weights).all() or not np.isfinite(means).all() or not np.isfinite(variances).all():
            raise ValueError("WGMM parameters contain NaN or Inf")
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("WGMM weights must be non-negative with positive sum")
        if np.any(variances <= 0.0):
            raise ValueError("WGMM variances must be positive")
        if self.standardize_mean is not None or self.standardize_std is not None:
            if self.standardize_mean is None or self.standardize_std is None:
                raise ValueError("WGMM standardization mean and std must be provided together")
            mean = np.asarray(self.standardize_mean, dtype=np.float64).reshape(-1)
            std = np.asarray(self.standardize_std, dtype=np.float64).reshape(-1)
            if mean.shape != (arch_dim,) or std.shape != (arch_dim,):
                raise ValueError("WGMM standardization vectors have the wrong dimension")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
                raise ValueError("WGMM standardization vectors are invalid")
        return self

    @property
    def n_components(self) -> int:
        return int(np.asarray(self.weights).size)

    @property
    def estimator_semantics(self) -> str:
        return (
            "ordinary_diagonal_gmm_uniform_candidate_weights"
            if self.source == "gmm_fit_pool"
            else "serialized_checkpoint_diagonal_mixture"
        )

    @property
    def parameter_fingerprint(self) -> str:
        """Fingerprint the effective mixture parameters, independent of their source file."""

        digest = hashlib.sha256()
        values = (
            ("weights", self.weights),
            ("means", self.means),
            ("variances", self.variances),
            ("standardize_mean", self.standardize_mean),
            ("standardize_std", self.standardize_std),
        )
        for name, value in values:
            digest.update(name.encode("ascii"))
            if value is None:
                digest.update(b"none")
                continue
            array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": WGMM_TED_FORMAT_VERSION,
            "source": self.source,
            "source_fingerprint": self.source_fingerprint,
            "parameter_fingerprint": self.parameter_fingerprint,
            "estimator_semantics": self.estimator_semantics,
            "n_components": self.n_components,
            "weights": np.asarray(self.weights, dtype=np.float64).tolist(),
            "means": np.asarray(self.means, dtype=np.float64).tolist(),
            "variances": np.asarray(self.variances, dtype=np.float64).tolist(),
            "standardize_mean": (
                None if self.standardize_mean is None
                else np.asarray(self.standardize_mean, dtype=np.float64).tolist()
            ),
            "standardize_std": (
                None if self.standardize_std is None
                else np.asarray(self.standardize_std, dtype=np.float64).tolist()
            ),
        }


def _torch_load_cpu(path: str | os.PathLike[str]) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _mapping_candidates(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    candidates = [payload]
    for key in ("wgmm", "gmm", "mixture", "mixture_model", "metadata"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    return candidates


def _first(mapping: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def load_checkpoint_wgmm(
    path: str | os.PathLike[str],
    *,
    arch_dim: int,
    covariance_regularization: float = 0.0,
) -> DiagonalGMMParameters:
    """Load an explicitly serialized diagonal WGMM or fail without fallback."""

    payload = _torch_load_cpu(path)
    for mapping in _mapping_candidates(payload):
        weights = _first(mapping, ("weights", "weights_", "mixture_weights", "wgmm_weights"))
        means = _first(mapping, ("means", "means_", "mixture_means", "wgmm_means"))
        variances = _first(
            mapping,
            ("variances", "vars", "vars_", "covariances", "covariances_", "wgmm_variances"),
        )
        if weights is None or means is None or variances is None:
            continue
        weights_array = np.asarray(torch.as_tensor(weights).cpu(), dtype=np.float64)
        means_array = np.asarray(torch.as_tensor(means).cpu(), dtype=np.float64)
        variances_array = np.asarray(torch.as_tensor(variances).cpu(), dtype=np.float64)
        if variances_array.ndim == 3:
            off_diagonal = variances_array - np.asarray(
                [np.diag(np.diag(row)) for row in variances_array], dtype=np.float64
            )
            if not np.allclose(off_diagonal, 0.0, rtol=0.0, atol=1e-12):
                raise ValueError("checkpoint WGMM has full covariance; diagonal WGMM is required")
            variances_array = np.stack([np.diag(row) for row in variances_array])
        variances_array = variances_array + float(covariance_regularization)
        standardize_mean = _first(mapping, ("standardize_mean", "x_mean", "feature_mean"))
        standardize_std = _first(mapping, ("standardize_std", "x_std", "feature_std"))
        parameters = DiagonalGMMParameters(
            weights=weights_array,
            means=means_array,
            variances=variances_array,
            source="checkpoint",
            source_fingerprint=sha256_file(path),
            standardize_mean=(
                None if standardize_mean is None
                else np.asarray(torch.as_tensor(standardize_mean).cpu(), dtype=np.float64)
            ),
            standardize_std=(
                None if standardize_std is None
                else np.asarray(torch.as_tensor(standardize_std).cpu(), dtype=np.float64)
            ),
        )
        return parameters.validate(arch_dim)
    raise ValueError(
        f"checkpoint {os.fspath(path)!r} contains no recoverable WGMM weights/means/variances; "
        "use --wgmm_source gmm_fit_pool explicitly to fit an unlabeled pool GMM"
    )


def fit_pool_gmm(
    z_arch: Any,
    *,
    n_components: int,
    random_state: int,
    covariance_regularization: float,
    var_floor: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> DiagonalGMMParameters:
    """Fit an ordinary diagonal GMM to the unlabeled pool using uniform weights."""

    arch = np.asarray(z_arch, dtype=np.float64)
    if arch.ndim != 2 or arch.shape[0] == 0:
        raise ValueError("z_arch must be a non-empty 2D array")
    if not np.isfinite(arch).all():
        raise ValueError("z_arch contains NaN or Inf")
    standardized, mean, std = standardize_fit(arch)
    model = WeightedDiagonalGMM(
        n_components=int(n_components),
        max_iter=int(max_iter),
        tol=float(tol),
        var_floor=float(var_floor),
        reg_covar=float(covariance_regularization),
        random_state=int(random_state),
    ).fit(standardized, sample_weight=np.ones(arch.shape[0], dtype=np.float64))
    assert model.weights_ is not None and model.means_ is not None and model.vars_ is not None
    source_config = {
        "source": "gmm_fit_pool",
        "pool_fingerprint": fingerprint_array(arch),
        "n_components": int(n_components),
        "random_state": int(random_state),
        "covariance_regularization": float(covariance_regularization),
        "var_floor": float(var_floor),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "candidate_sample_weights": "uniform_ones",
        "estimator_semantics": "ordinary_diagonal_gmm",
    }
    return DiagonalGMMParameters(
        weights=model.weights_.copy(),
        means=model.means_.copy(),
        variances=model.vars_.copy(),
        source="gmm_fit_pool",
        source_fingerprint=canonical_json_fingerprint(source_config),
        standardize_mean=mean,
        standardize_std=std,
    ).validate(arch.shape[1])


def gmm_responsibilities(z_arch: Any, parameters: DiagonalGMMParameters) -> np.ndarray:
    arch = np.asarray(z_arch, dtype=np.float64)
    if arch.ndim != 2:
        raise ValueError("z_arch must be 2D")
    parameters.validate(arch.shape[1])
    features = arch
    if parameters.standardize_mean is not None:
        features = standardize_apply(
            features,
            np.asarray(parameters.standardize_mean),
            np.asarray(parameters.standardize_std),
        )
    model = WeightedDiagonalGMM(n_components=parameters.n_components)
    model.weights_ = np.asarray(parameters.weights, dtype=np.float64).copy()
    model.means_ = np.asarray(parameters.means, dtype=np.float64).copy()
    model.vars_ = np.asarray(parameters.variances, dtype=np.float64).copy()
    return model.predict_proba(features)


def responsibility_entropy(responsibilities: Any) -> np.ndarray:
    resp = np.asarray(responsibilities, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("responsibilities must be 2D")
    safe = np.clip(resp, 1e-300, 1.0)
    return -np.sum(resp * np.log(safe), axis=1)


def assign_clusters(z_arch: Any, parameters: DiagonalGMMParameters) -> dict[str, np.ndarray]:
    resp = gmm_responsibilities(z_arch, parameters)
    cluster_ids = np.argmax(resp, axis=1).astype(np.int64)
    return {
        "responsibilities": resp,
        "cluster_ids": cluster_ids,
        "entropy": responsibility_entropy(resp),
        "max_responsibility": np.max(resp, axis=1),
    }


def _largest_remainder(raw: np.ndarray, total: int, cluster_ids: np.ndarray) -> np.ndarray:
    floors = np.floor(raw).astype(np.int64)
    remaining = int(total) - int(floors.sum())
    fractions = raw - floors
    order = sorted(range(raw.size), key=lambda pos: (-float(fractions[pos]), int(cluster_ids[pos])))
    for pos in order[:remaining]:
        floors[pos] += 1
    return floors


def allocate_cluster_quotas(
    capacities: dict[int, int],
    budget: int,
    *,
    mode: str,
    component_weights: dict[int, float] | None = None,
    equal_weight: float = 0.5,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Allocate exact deterministic quotas with capacity redistribution."""

    if mode not in QUOTA_MODES:
        raise ValueError(f"quota mode must be one of {QUOTA_MODES}, got {mode!r}")
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    active = np.asarray(sorted(key for key, value in capacities.items() if int(value) > 0), dtype=np.int64)
    total_capacity = int(sum(max(0, int(capacities[int(key)])) for key in active))
    if budget > total_capacity:
        raise ValueError(f"budget {budget} exceeds active cluster capacity {total_capacity}")
    if active.size == 0:
        if budget != 0:
            raise ValueError("cannot allocate positive budget to empty clusters")
        return {}, {"raw": {}, "initial": {}, "final": {}, "redistribution": []}
    alpha = float(equal_weight)
    if not 0.0 <= alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("equal_weight must be finite and in [0, 1]")
    weights = np.asarray(
        [0.0 if component_weights is None else float(component_weights.get(int(key), 0.0)) for key in active],
        dtype=np.float64,
    )
    weights = np.maximum(np.where(np.isfinite(weights), weights, 0.0), 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.asarray([int(capacities[int(key)]) for key in active], dtype=np.float64)
    weights /= weights.sum()
    equal = np.full(active.size, 1.0 / active.size, dtype=np.float64)
    if mode == "equal":
        mass = equal
    elif mode == "proportional":
        mass = weights
    else:
        mass = alpha * equal + (1.0 - alpha) * weights
    raw = mass * budget
    initial = _largest_remainder(raw, budget, active)
    if budget >= active.size:
        zero_positions = [pos for pos, value in enumerate(initial) if value == 0]
        for pos in zero_positions:
            donors = sorted(
                [idx for idx, value in enumerate(initial) if value > 1],
                key=lambda idx: (-int(initial[idx]), int(active[idx])),
            )
            if donors:
                initial[donors[0]] -= 1
                initial[pos] += 1
    final = initial.copy()
    redistribution: list[dict[str, int]] = []
    overflow = 0
    for pos, cluster_id in enumerate(active):
        capacity = int(capacities[int(cluster_id)])
        if final[pos] > capacity:
            overflow += int(final[pos] - capacity)
            redistribution.append(
                {"cluster_id": int(cluster_id), "shortfall": int(final[pos] - capacity)}
            )
            final[pos] = capacity
    while overflow > 0:
        receivers = [
            pos for pos, cluster_id in enumerate(active)
            if int(final[pos]) < int(capacities[int(cluster_id)])
        ]
        if not receivers:
            raise RuntimeError("quota redistribution exhausted cluster capacity")
        receivers.sort(
            key=lambda pos: (
                -float(raw[pos] - final[pos]),
                int(active[pos]),
            )
        )
        receiver = receivers[0]
        final[receiver] += 1
        overflow -= 1
        redistribution.append({"cluster_id": int(active[receiver]), "added": 1})
    result = {int(cluster_id): int(final[pos]) for pos, cluster_id in enumerate(active)}
    if sum(result.values()) != budget:
        raise RuntimeError("cluster quota allocation did not preserve the budget")
    diagnostics = {
        "mode": mode,
        "equal_weight": alpha,
        "budget": budget,
        "capacities": {str(key): int(value) for key, value in sorted(capacities.items())},
        "raw": {str(key): float(raw[pos]) for pos, key in enumerate(active)},
        "initial": {str(key): int(initial[pos]) for pos, key in enumerate(active)},
        "final": {str(key): int(final[pos]) for pos, key in enumerate(active)},
        "redistribution": redistribution,
    }
    return result, diagnostics


def ted_features(
    z_search: Any,
    *,
    arch_nz: int,
    hp_mode: str,
    z_bound: float,
    condition_masks: Any | None = None,
) -> np.ndarray:
    """Use the Exact-GP fixed normalization and suppress inactive dimensions."""

    normalized = np.asarray(
        normalize_search_vector(
            np.asarray(z_search, dtype=np.float64),
            arch_nz=int(arch_nz),
            hp_mode=hp_mode,
            z_bound=float(z_bound),
        ),
        dtype=np.float64,
    )
    if condition_masks is not None:
        masks = np.asarray(condition_masks, dtype=np.float64)
        if masks.shape != normalized.shape:
            raise ValueError(
                f"condition_masks must match normalized search shape {normalized.shape}, got {masks.shape}"
            )
        if not np.isfinite(masks).all() or np.any(masks < 0.0) or np.any(masks > 1.0):
            raise ValueError("condition masks must be finite and in [0, 1]")
        normalized = normalized * masks
    return normalized


def rbf_kernel(features: Any, *, lengthscale: float) -> np.ndarray:
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array")
    lengthscale = float(lengthscale)
    if not math.isfinite(lengthscale) or lengthscale <= 0.0:
        raise ValueError("TED kernel lengthscale must be positive and finite")
    differences = X[:, None, :] - X[None, :, :]
    squared = np.sum(differences * differences, axis=-1)
    kernel = np.exp(-0.5 * squared / (lengthscale * lengthscale))
    kernel = 0.5 * (kernel + kernel.T)
    if not np.isfinite(kernel).all():
        raise RuntimeError("TED kernel contains NaN or Inf")
    return kernel


def _cholesky_solve(matrix: np.ndarray, rhs: np.ndarray, jitter: float) -> tuple[np.ndarray, float]:
    used = max(0.0, float(jitter))
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    for _ in range(8):
        try:
            factor = np.linalg.cholesky(0.5 * (matrix + matrix.T) + used * identity)
            solution = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
            return solution, used
        except np.linalg.LinAlgError:
            used = 1e-12 if used == 0.0 else used * 10.0
    raise np.linalg.LinAlgError("TED conditioning matrix is not positive definite")


def conditional_covariance(
    kernel: Any,
    remaining_positions: Sequence[int],
    conditioned_positions: Sequence[int],
    *,
    regularization: float,
    jitter: float,
) -> tuple[np.ndarray, float]:
    K = np.asarray(kernel, dtype=np.float64)
    remaining = np.asarray(list(remaining_positions), dtype=np.int64)
    conditioned = np.asarray(list(conditioned_positions), dtype=np.int64)
    covariance = K[np.ix_(remaining, remaining)].copy()
    used_jitter = float(jitter)
    if conditioned.size:
        K_ss = K[np.ix_(conditioned, conditioned)] + np.eye(conditioned.size) * float(regularization)
        K_us = K[np.ix_(remaining, conditioned)]
        solved, used_jitter = _cholesky_solve(K_ss, K_us.T, float(jitter))
        covariance -= K_us @ solved
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance).copy()
    if np.min(diagonal) < -1e-7:
        raise RuntimeError(f"TED conditional covariance has materially negative variance {np.min(diagonal)}")
    np.fill_diagonal(covariance, np.maximum(diagonal, 0.0))
    if not np.isfinite(covariance).all():
        raise RuntimeError("TED conditional covariance contains NaN or Inf")
    return covariance, used_jitter


def greedy_ted_select(
    kernel: Any,
    candidate_indices: Sequence[int],
    budget: int,
    *,
    conditioned_indices: Sequence[int] = (),
    regularization: float = 0.1,
    jitter: float = 1e-8,
    cluster_id: int | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Sequentially maximize full-pool trace reduction."""

    K = np.asarray(kernel, dtype=np.float64)
    original_indices = np.asarray(list(candidate_indices), dtype=np.int64)
    conditioned_original = np.asarray(list(conditioned_indices), dtype=np.int64)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] != original_indices.size:
        raise ValueError("kernel shape must match candidate_indices")
    if len(set(original_indices.tolist())) != original_indices.size:
        raise ValueError("candidate_indices must be unique")
    if int(budget) < 0 or int(budget) > original_indices.size - conditioned_original.size:
        raise ValueError("TED budget is outside available candidate capacity")
    if float(regularization) <= 0.0 or not math.isfinite(float(regularization)):
        raise ValueError("TED regularization must be finite and positive")
    position_by_original = {int(value): pos for pos, value in enumerate(original_indices)}
    try:
        conditioned_positions = [position_by_original[int(value)] for value in conditioned_original]
    except KeyError as exc:
        raise ValueError(f"conditioned candidate {exc.args[0]} is not in the kernel") from exc
    conditioned_set = set(conditioned_positions)
    remaining_positions = [pos for pos in range(original_indices.size) if pos not in conditioned_set]
    covariance, used_jitter = conditional_covariance(
        K,
        remaining_positions,
        conditioned_positions,
        regularization=float(regularization),
        jitter=float(jitter),
    )
    covariance_original = [int(original_indices[pos]) for pos in remaining_positions]
    active_positions = list(range(len(covariance_original)))
    selected: list[int] = []
    trace: list[dict[str, Any]] = []
    for rank in range(int(budget)):
        diagonal = np.maximum(np.diag(covariance), 0.0)
        denominators = diagonal + float(regularization)
        if np.any(denominators <= 0.0):
            raise RuntimeError("TED score denominator is non-positive")
        scores = np.sum(covariance[:, active_positions] ** 2, axis=0) / denominators[active_positions]
        if not np.isfinite(scores).all():
            raise RuntimeError("TED scores contain NaN or Inf")
        best_value = float(np.max(scores))
        tied = np.flatnonzero(np.isclose(scores, best_value, rtol=1e-12, atol=1e-15))
        active_pick = min(
            tied.tolist(), key=lambda pos: covariance_original[active_positions[pos]]
        )
        pick = active_positions[active_pick]
        chosen_original = int(covariance_original[pick])
        trace_before = float(np.trace(covariance))
        residual_variance = float(diagonal[pick])
        update = np.outer(covariance[:, pick], covariance[pick, :]) / denominators[pick]
        updated = covariance - update
        updated = 0.5 * (updated + updated.T)
        if updated.size:
            updated_diagonal = np.diag(updated).copy()
            if np.min(updated_diagonal) < -1e-7:
                raise RuntimeError("TED rank-one update created a materially negative variance")
            np.fill_diagonal(updated, np.maximum(updated_diagonal, 0.0))
        trace_after = float(np.trace(updated)) if updated.size else 0.0
        reduction = trace_before - trace_after
        if reduction < -1e-8:
            raise RuntimeError("TED trace increased after a selection")
        trace.append(
            {
                "candidate_index": chosen_original,
                "ted_score": best_value,
                "residual_variance": residual_variance,
                "trace_before": trace_before,
                "trace_after": trace_after,
                "trace_reduction": reduction,
                "cluster_id": None if cluster_id is None else int(cluster_id),
                "selection_rank": rank,
                "jitter_used": float(used_jitter),
            }
        )
        selected.append(chosen_original)
        active_positions.pop(active_pick)
        covariance = updated
    return selected, trace


def percentile_ranks(
    values: Sequence[float],
    candidate_indices: Sequence[int],
    *,
    valid: Sequence[bool] | None = None,
) -> np.ndarray:
    """Return deterministic [0, 1] ranks; higher values are better."""

    values_array = np.asarray(values, dtype=np.float64).reshape(-1)
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if values_array.shape != indices.shape:
        raise ValueError("rank values and candidate_indices must have equal length")
    valid_array = np.ones(values_array.size, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid_array.shape != values_array.shape:
        raise ValueError("rank valid flags have the wrong length")
    finite_valid = valid_array & np.isfinite(values_array)
    order = sorted(
        range(values_array.size),
        key=lambda pos: (
            0 if finite_valid[pos] else 1,
            -float(values_array[pos]) if finite_valid[pos] else 0.0,
            int(indices[pos]),
        ),
    )
    ranks = np.zeros(values_array.size, dtype=np.float64)
    count_valid = int(finite_valid.sum())
    if count_valid:
        denominator = max(1, count_valid - 1)
        for ordinal, pos in enumerate(order[:count_valid]):
            ranks[pos] = 1.0 - ordinal / denominator if count_valid > 1 else 1.0
    return ranks


def combined_expansion_scores(
    candidate_indices: Sequence[int],
    low_fidelity: Sequence[float],
    gp_mean: Sequence[float],
    gp_std: Sequence[float],
    *,
    low_fidelity_valid: Sequence[bool],
    weights: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    score_weights = np.asarray(weights, dtype=np.float64)
    if score_weights.shape != (3,) or not np.isfinite(score_weights).all() or np.any(score_weights < 0.0):
        raise ValueError("expansion score weights must be three finite non-negative values")
    if not np.isclose(float(score_weights.sum()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("expansion score weights must sum to 1")
    indices = np.asarray(candidate_indices, dtype=np.int64)
    lf_rank = percentile_ranks(low_fidelity, indices, valid=low_fidelity_valid)
    mean_rank = percentile_ranks(gp_mean, indices)
    std_rank = percentile_ranks(gp_std, indices)
    combined = score_weights[0] * lf_rank + score_weights[1] * mean_rank + score_weights[2] * std_rank
    valid_array = np.asarray(low_fidelity_valid, dtype=bool)
    combined = np.where(valid_array, combined, -1.0)
    return {
        "low_fidelity_rank": lf_rank,
        "gp_mean_rank": mean_rank,
        "gp_std_rank": std_rank,
        "combined_score": combined,
    }


def select_by_cluster_scores(
    candidate_indices: Sequence[int],
    cluster_ids: Sequence[int],
    scores: Sequence[float],
    quotas: dict[int, int],
) -> list[int]:
    indices = np.asarray(candidate_indices, dtype=np.int64)
    clusters = np.asarray(cluster_ids, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if indices.shape != clusters.shape or indices.shape != score_array.shape:
        raise ValueError("score selection arrays must have equal shape")
    selected: list[int] = []
    for cluster_id in sorted(quotas):
        positions = np.flatnonzero(clusters == int(cluster_id)).tolist()
        positions.sort(
            key=lambda pos: (
                -float(score_array[pos]) if np.isfinite(score_array[pos]) else math.inf,
                int(indices[pos]),
            )
        )
        quota = int(quotas[cluster_id])
        if quota > len(positions):
            raise ValueError(f"cluster {cluster_id} score quota exceeds shortlist capacity")
        selected.extend(int(indices[pos]) for pos in positions[:quota])
    if len(selected) != sum(int(value) for value in quotas.values()):
        raise RuntimeError("score-based cluster selection did not fill the requested budget")
    return selected
