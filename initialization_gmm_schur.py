"""Cluster-local Schur conditional-variance selection on a shared global kernel."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from eval_utils import stable_seed
from initialization_wgmm_ted import (
    DiagonalGMMParameters,
    _fit_standardized_pool_gmm,
    assign_clusters,
    canonical_json_fingerprint,
)
from weighted_diag_gmm_init import standardize_fit


SCHUR_SELECTION_CRITERION = "schur_conditional_variance"
GMM_FIXED_VERSION = "gmm_fixed_v1"


def _fingerprint_arrays(*values: Any, dtype: str) -> str:
    digest = hashlib.sha256()
    for position, value in enumerate(values):
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        digest.update(str(position).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_assignment_fingerprint(cluster_ids: Any) -> str:
    assignments = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    return _fingerprint_arrays(assignments, dtype="<i8")


def canonicalize_diagonal_gmm_parameters(
    parameters: DiagonalGMMParameters,
) -> tuple[DiagonalGMMParameters, list[int]]:
    """Canonicalize component labels by standardized means and original index."""

    means = np.asarray(parameters.means, dtype=np.float64)
    parameters.validate(means.shape[1])
    order = sorted(
        range(parameters.n_components),
        key=lambda component: (
            *tuple(float(value) for value in means[component].tolist()),
            int(component),
        ),
    )
    canonical = DiagonalGMMParameters(
        weights=np.asarray(parameters.weights, dtype=np.float64)[order].copy(),
        means=means[order].copy(),
        variances=np.asarray(parameters.variances, dtype=np.float64)[order].copy(),
        source=parameters.source,
        source_fingerprint=parameters.source_fingerprint,
        standardize_mean=(
            None
            if parameters.standardize_mean is None
            else np.asarray(parameters.standardize_mean, dtype=np.float64).copy()
        ),
        standardize_std=(
            None
            if parameters.standardize_std is None
            else np.asarray(parameters.standardize_std, dtype=np.float64).copy()
        ),
    ).validate(means.shape[1])
    return canonical, [int(component) for component in order]


def assert_exact_cluster_quotas(
    quotas: dict[int, int],
    capacities: dict[int, int],
    requested_budget: int,
    *,
    stage: str,
) -> None:
    requested_budget = int(requested_budget)
    normalized_capacities = {
        int(cluster_id): int(capacity)
        for cluster_id, capacity in capacities.items()
    }
    normalized_quotas = {
        int(cluster_id): int(quota) for cluster_id, quota in quotas.items()
    }
    available = sum(max(0, capacity) for capacity in normalized_capacities.values())
    if available < requested_budget:
        raise ValueError(
            f"{stage} quota capacity is insufficient: requested={requested_budget} "
            f"available_capacity={available} "
            f"per_cluster_capacity={normalized_capacities} "
            f"unfilled_count={requested_budget - available}"
        )
    if sum(normalized_quotas.values()) != requested_budget:
        raise RuntimeError(
            f"{stage} quotas do not fill the requested budget: "
            f"requested={requested_budget} quotas={normalized_quotas}"
        )
    for cluster_id, quota in normalized_quotas.items():
        capacity = normalized_capacities.get(cluster_id, 0)
        if quota < 0 or quota > capacity:
            raise RuntimeError(
                f"{stage} quota violates cluster capacity: cluster={cluster_id} "
                f"quota={quota} capacity={capacity}"
            )


def fit_fixed_pool_gmm_restarts(
    z_arch: Any,
    *,
    search_seed: int,
    candidate_pool_fingerprint: str,
    n_components: int = 6,
    restarts: int = 5,
    covariance_regularization: float,
    var_floor: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[
    DiagonalGMMParameters,
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Fit fixed-K ordinary diagonal-GMM restarts and require non-empty hard clusters."""

    arch = np.asarray(z_arch, dtype=np.float64)
    if arch.ndim != 2 or arch.shape[0] == 0:
        raise ValueError("z_arch must be a non-empty 2D array")
    if not np.isfinite(arch).all():
        raise ValueError("z_arch contains NaN or Inf")
    if not isinstance(candidate_pool_fingerprint, str) or not candidate_pool_fingerprint:
        raise ValueError("candidate_pool_fingerprint must be a non-empty string")
    n_components = int(n_components)
    restarts = int(restarts)
    if n_components <= 0:
        raise ValueError("fixed GMM component count must be positive")
    if n_components > arch.shape[0]:
        raise ValueError(
            f"fixed GMM component count {n_components} exceeds sample count {arch.shape[0]}"
        )
    if restarts <= 0:
        raise ValueError("fixed GMM restarts must be positive")

    standardized, standardize_mean, standardize_std = standardize_fit(arch)
    preprocessing_fingerprint = _fingerprint_arrays(
        standardize_mean,
        standardize_std,
        dtype="<f8",
    )
    trace: list[dict[str, Any]] = []
    eligible: list[
        tuple[
            int,
            float,
            DiagonalGMMParameters,
            dict[str, np.ndarray],
            list[int],
        ]
    ] = []
    for restart_id in range(restarts):
        restart_seed = stable_seed(
            int(search_seed),
            GMM_FIXED_VERSION,
            candidate_pool_fingerprint,
            n_components,
            restart_id,
        )
        record: dict[str, Any] = {
            "gmm_version": GMM_FIXED_VERSION,
            "candidate_pool_fingerprint": candidate_pool_fingerprint,
            "gmm_preprocessing_fingerprint": preprocessing_fingerprint,
            "n_components": n_components,
            "restart_id": restart_id,
            "restart_seed": int(restart_seed),
            "fit_success": False,
            "converged": False,
            "n_iter": None,
            "average_lower_bound": None,
            "log_likelihood": None,
            "parameters_finite": False,
            "all_hard_clusters_nonempty": False,
            "hard_cluster_count": 0,
            "cluster_capacities": {},
            "canonical_component_order": [],
            "gmm_parameter_fingerprint": None,
            "assignment_fingerprint": None,
            "selection_eligible": False,
            "selected_restart": False,
            "failure_reason": "",
        }
        try:
            model = _fit_standardized_pool_gmm(
                standardized,
                n_components=n_components,
                random_state=int(restart_seed),
                covariance_regularization=float(covariance_regularization),
                var_floor=float(var_floor),
                max_iter=int(max_iter),
                tol=float(tol),
            )
            if model.n_components != n_components:
                raise RuntimeError(
                    f"ordinary GMM fitted {model.n_components} components, "
                    f"expected fixed K={n_components}"
                )
            assert model.weights_ is not None
            assert model.means_ is not None
            assert model.vars_ is not None
            raw_parameters = DiagonalGMMParameters(
                weights=model.weights_.copy(),
                means=model.means_.copy(),
                variances=model.vars_.copy(),
                source="gmm_fit_pool",
                source_fingerprint=canonical_json_fingerprint(
                    {
                        "gmm_version": GMM_FIXED_VERSION,
                        "candidate_pool_fingerprint": candidate_pool_fingerprint,
                        "search_seed": int(search_seed),
                        "n_components": n_components,
                        "restart_id": restart_id,
                        "restart_seed": int(restart_seed),
                        "covariance_regularization": float(
                            covariance_regularization
                        ),
                        "var_floor": float(var_floor),
                        "max_iter": int(max_iter),
                        "tol": float(tol),
                    }
                ),
                standardize_mean=standardize_mean,
                standardize_std=standardize_std,
            ).validate(arch.shape[1])
            parameters, component_order = canonicalize_diagonal_gmm_parameters(
                raw_parameters
            )
            assignments = assign_clusters(arch, parameters)
            cluster_ids = np.asarray(assignments["cluster_ids"], dtype=np.int64)
            capacities = {
                component: int(np.sum(cluster_ids == component))
                for component in range(n_components)
            }
            hard_cluster_count = sum(capacity > 0 for capacity in capacities.values())
            all_nonempty = hard_cluster_count == n_components
            average_lower_bound = float(model.lower_bound_)
            log_likelihood = float(arch.shape[0]) * average_lower_bound
            parameters_finite = all(
                np.isfinite(value).all()
                for value in (
                    parameters.weights,
                    parameters.means,
                    parameters.variances,
                    parameters.standardize_mean,
                    parameters.standardize_std,
                )
            )
            converged = bool(model.converged_)
            eligible_restart = (
                converged
                and parameters_finite
                and math.isfinite(log_likelihood)
                and all_nonempty
            )
            failure_reasons: list[str] = []
            if not converged:
                failure_reasons.append("not_converged")
            if not parameters_finite or not math.isfinite(log_likelihood):
                failure_reasons.append("non_finite_parameters_or_likelihood")
            if not all_nonempty:
                failure_reasons.append("empty_hard_cluster")
            assignment_fingerprint = canonical_assignment_fingerprint(cluster_ids)
            record.update(
                {
                    "fit_success": True,
                    "converged": converged,
                    "n_iter": int(model.n_iter_),
                    "average_lower_bound": average_lower_bound,
                    "log_likelihood": log_likelihood,
                    "parameters_finite": parameters_finite,
                    "all_hard_clusters_nonempty": all_nonempty,
                    "hard_cluster_count": hard_cluster_count,
                    "cluster_capacities": json.dumps(
                        capacities, sort_keys=True, separators=(",", ":")
                    ),
                    "canonical_component_order": json.dumps(
                        component_order, separators=(",", ":")
                    ),
                    "gmm_parameter_fingerprint": parameters.parameter_fingerprint,
                    "assignment_fingerprint": assignment_fingerprint,
                    "selection_eligible": eligible_restart,
                    "failure_reason": ",".join(failure_reasons),
                }
            )
            if eligible_restart:
                eligible.append(
                    (
                        restart_id,
                        log_likelihood,
                        parameters,
                        assignments,
                        component_order,
                    )
                )
        except Exception as exc:
            record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        trace.append(record)

    if not eligible:
        failures = {
            int(record["restart_id"]): str(record["failure_reason"])
            for record in trace
        }
        raise RuntimeError(
            f"no eligible fixed K={n_components} ordinary diagonal-GMM restart "
            f"produced {n_components} non-empty hard clusters; failures={failures}"
        )
    best_log_likelihood = max(item[1] for item in eligible)
    tied = [
        item
        for item in eligible
        if math.isclose(
            item[1],
            best_log_likelihood,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ]
    (
        selected_restart_id,
        _selected_log_likelihood,
        selected_parameters,
        selected_assignments,
        selected_order,
    ) = min(tied, key=lambda item: item[0])
    for record in trace:
        record["selected_restart"] = (
            int(record["restart_id"]) == selected_restart_id
        )

    selected_cluster_ids = np.asarray(
        selected_assignments["cluster_ids"], dtype=np.int64
    )
    selected_capacities = {
        component: int(np.sum(selected_cluster_ids == component))
        for component in range(n_components)
    }
    metadata = {
        "gmm_version": GMM_FIXED_VERSION,
        "n_components": n_components,
        "restarts": restarts,
        "selected_restart_id": selected_restart_id,
        "selected_restart_seed": int(
            trace[selected_restart_id]["restart_seed"]
        ),
        "gmm_preprocessing_fingerprint": preprocessing_fingerprint,
        "canonical_component_order": selected_order,
        "canonical_centers_fingerprint": _fingerprint_arrays(
            selected_parameters.means, dtype="<f8"
        ),
        "gmm_parameter_fingerprint": selected_parameters.parameter_fingerprint,
        "assignment_fingerprint": canonical_assignment_fingerprint(
            selected_cluster_ids
        ),
        "cluster_capacities": selected_capacities,
    }
    return selected_parameters, selected_assignments, trace, metadata


def _indices_fingerprint(indices: Sequence[int]) -> str:
    payload = json.dumps(
        [int(value) for value in indices],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _validate_kernel(kernel: Any) -> np.ndarray:
    matrix = np.asarray(kernel, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kernel must be a square matrix")
    if matrix.shape[0] == 0:
        raise ValueError("kernel must be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError("kernel contains NaN or Inf")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("kernel must be symmetric")
    return 0.5 * (matrix + matrix.T)


def _cholesky_factor(
    matrix: np.ndarray,
    *,
    jitter: float,
    jitter_multiplier: float,
    max_jitter_tries: int,
) -> tuple[np.ndarray, float]:
    jitter = float(jitter)
    jitter_multiplier = float(jitter_multiplier)
    max_jitter_tries = int(max_jitter_tries)
    if not math.isfinite(jitter) or jitter < 0.0:
        raise ValueError("Schur jitter must be finite and non-negative")
    if not math.isfinite(jitter_multiplier) or jitter_multiplier <= 1.0:
        raise ValueError("Schur jitter multiplier must be finite and greater than 1")
    if max_jitter_tries <= 0:
        raise ValueError("Schur max_jitter_tries must be positive")

    identity = np.eye(matrix.shape[0], dtype=np.float64)
    used_jitter = jitter
    failures: list[str] = []
    for _ in range(max_jitter_tries):
        try:
            factor = np.linalg.cholesky(matrix + used_jitter * identity)
            return factor, used_jitter
        except np.linalg.LinAlgError as exc:
            failures.append(str(exc))
            used_jitter = 1e-12 if used_jitter == 0.0 else used_jitter * jitter_multiplier
    raise RuntimeError(
        "Schur conditioning Cholesky failed after "
        f"{max_jitter_tries} jitter attempts; last_error={failures[-1] if failures else 'unknown'}"
    )


def conditional_schur_variances(
    kernel: Any,
    candidate_indices: Sequence[int],
    *,
    conditioned_indices: Sequence[int] = (),
    jitter: float = 1e-8,
    jitter_multiplier: float = 10.0,
    max_jitter_tries: int = 8,
    negative_tolerance: float = 1e-7,
) -> tuple[np.ndarray, float]:
    """Return exact kernel conditional variances for global candidate indices."""

    matrix = _validate_kernel(kernel)
    candidates = np.asarray([int(value) for value in candidate_indices], dtype=np.int64)
    conditioned = np.asarray([int(value) for value in conditioned_indices], dtype=np.int64)
    if candidates.ndim != 1 or conditioned.ndim != 1:
        raise ValueError("candidate and conditioned indices must be one-dimensional")
    if len(set(candidates.tolist())) != candidates.size:
        raise ValueError("candidate_indices must be unique")
    if len(set(conditioned.tolist())) != conditioned.size:
        raise ValueError("conditioned_indices must be unique")
    for label, values in (("candidate", candidates), ("conditioned", conditioned)):
        if values.size and (int(values.min()) < 0 or int(values.max()) >= matrix.shape[0]):
            raise ValueError(f"{label} index is outside the global kernel")
    if candidates.size == 0:
        return np.empty((0,), dtype=np.float64), 0.0

    scores = np.diag(matrix)[candidates].astype(np.float64, copy=True)
    used_jitter = 0.0
    if conditioned.size:
        conditioning_matrix = matrix[np.ix_(conditioned, conditioned)]
        factor, used_jitter = _cholesky_factor(
            conditioning_matrix,
            jitter=float(jitter),
            jitter_multiplier=float(jitter_multiplier),
            max_jitter_tries=int(max_jitter_tries),
        )
        cross = matrix[np.ix_(conditioned, candidates)]
        solved = np.linalg.solve(factor, cross)
        scores -= np.sum(solved * solved, axis=0)
    if not np.isfinite(scores).all():
        raise RuntimeError("Schur conditional variances contain NaN or Inf")
    negative_tolerance = float(negative_tolerance)
    if not math.isfinite(negative_tolerance) or negative_tolerance < 0.0:
        raise ValueError("negative_tolerance must be finite and non-negative")
    minimum = float(np.min(scores))
    if minimum < -negative_tolerance:
        offending = int(candidates[int(np.argmin(scores))])
        raise RuntimeError(
            "Schur conditional variance is materially negative: "
            f"candidate_index={offending} variance={minimum}"
        )
    return np.maximum(scores, 0.0), float(used_jitter)


def greedy_conditional_schur_select(
    kernel: Any,
    candidate_indices: Sequence[int],
    budget: int,
    *,
    conditioned_indices: Sequence[int] = (),
    jitter: float = 1e-8,
    jitter_multiplier: float = 10.0,
    max_jitter_tries: int = 8,
    negative_tolerance: float = 1e-7,
    cluster_id: int | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Greedily maximize exact conditional variance on a precomputed global kernel."""

    matrix = _validate_kernel(kernel)
    candidates = [int(value) for value in candidate_indices]
    conditioned = [int(value) for value in conditioned_indices]
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_indices must be unique")
    if len(set(conditioned)) != len(conditioned):
        raise ValueError("conditioned_indices must be unique")
    conditioned_set = set(conditioned)
    remaining = [index for index in candidates if index not in conditioned_set]
    budget = int(budget)
    if budget < 0 or budget > len(remaining):
        raise ValueError("Schur budget is outside available candidate capacity")

    selected: list[int] = []
    trace: list[dict[str, Any]] = []
    current_conditioned = list(conditioned)
    for selection_rank in range(budget):
        scores, used_jitter = conditional_schur_variances(
            matrix,
            remaining,
            conditioned_indices=current_conditioned,
            jitter=float(jitter),
            jitter_multiplier=float(jitter_multiplier),
            max_jitter_tries=int(max_jitter_tries),
            negative_tolerance=float(negative_tolerance),
        )
        best_score = float(np.max(scores))
        tied_positions = np.flatnonzero(scores == best_score).tolist()
        chosen_position = min(tied_positions, key=lambda position: remaining[position])
        chosen_index = int(remaining[chosen_position])
        conditioned_before = list(current_conditioned)
        trace.append(
            {
                "candidate_index": chosen_index,
                "cluster_id": None if cluster_id is None else int(cluster_id),
                "selection_rank": int(selection_rank),
                "selection_criterion": SCHUR_SELECTION_CRITERION,
                "selection_score": best_score,
                "schur_conditional_variance": best_score,
                "conditioned_count": len(conditioned_before),
                "conditioned_indices_fingerprint": _indices_fingerprint(
                    conditioned_before
                ),
                "jitter_used": float(used_jitter),
            }
        )
        selected.append(chosen_index)
        current_conditioned.append(chosen_index)
        remaining.pop(chosen_position)
    return selected, trace
