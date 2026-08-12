from __future__ import annotations

import copy
import csv
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bo_phase4
from eval_utils import stable_seed
from initialization_gmm_schur import (
    GMM_FIXED_VERSION,
    assert_exact_cluster_quotas,
    canonical_assignment_fingerprint,
    canonicalize_diagonal_gmm_parameters,
    conditional_schur_variances,
    fit_fixed_pool_gmm_restarts,
    greedy_conditional_schur_select,
)
from initialization_wgmm_ted import (
    DiagonalGMMParameters,
    allocate_cluster_quotas,
    assign_clusters,
    diagonal_gmm_bic,
    diagonal_gmm_parameter_count,
    fit_pool_gmm_bic,
    rbf_kernel,
    select_bic_model_records,
    ted_features,
)


def _brute_variance(
    kernel: np.ndarray,
    candidate: int,
    conditioned: list[int],
    jitter: float,
) -> float:
    if not conditioned:
        return float(kernel[candidate, candidate])
    conditioning = kernel[np.ix_(conditioned, conditioned)]
    cross = kernel[np.ix_([candidate], conditioned)]
    solved = np.linalg.solve(
        conditioning + jitter * np.eye(len(conditioned)),
        cross.T,
    )
    return float(kernel[candidate, candidate] - (cross @ solved)[0, 0])


def test_schur_conditional_variance_matches_stepwise_brute_force() -> None:
    features = np.asarray([[0.0], [0.2], [0.8], [1.0]], dtype=np.float64)
    kernel = rbf_kernel(features, lengthscale=0.35)
    jitter = 1e-8
    selected, trace = greedy_conditional_schur_select(
        kernel,
        [3, 0, 2, 1],
        3,
        jitter=jitter,
    )
    conditioned_positions: list[int] = []
    for rank, chosen in enumerate(selected):
        available = [
            index
            for index in [3, 0, 2, 1]
            if index not in conditioned_positions
        ]
        brute = {
            index: _brute_variance(
                kernel,
                index,
                conditioned_positions,
                jitter,
            )
            for index in available
        }
        best = max(brute.values())
        expected = min(
            index
            for index, value in brute.items()
            if np.isclose(value, best, rtol=1e-12, atol=1e-15)
        )
        assert chosen == expected
        assert trace[rank]["selection_score"] == pytest.approx(best)
        assert trace[rank]["schur_conditional_variance"] == pytest.approx(best)
        assert trace[rank]["conditioned_count"] == rank
        conditioned_positions.append(chosen)


def test_seed_and_shortlist_condition_on_all_prior_same_cluster_points() -> None:
    features = np.asarray([[0.0], [0.1], [0.4], [0.8], [1.0]], dtype=np.float64)
    kernel = rbf_kernel(features, lengthscale=0.3)
    seeds, seed_trace = greedy_conditional_schur_select(
        kernel, [0, 1, 2, 3, 4], 2, cluster_id=7,
    )
    shortlist, shortlist_trace = greedy_conditional_schur_select(
        kernel,
        [index for index in range(5) if index not in seeds],
        2,
        conditioned_indices=seeds,
        cluster_id=7,
    )
    assert [row["conditioned_count"] for row in seed_trace] == [0, 1]
    assert [row["conditioned_count"] for row in shortlist_trace] == [2, 3]
    assert not set(seeds).intersection(shortlist)
    assert all(row["cluster_id"] == 7 for row in seed_trace + shortlist_trace)
    assert all(
        row["selection_criterion"] == "schur_conditional_variance"
        for row in seed_trace + shortlist_trace
    )


def test_conditioned_candidates_are_excluded_and_ties_use_global_index() -> None:
    selected, trace = greedy_conditional_schur_select(
        np.eye(4),
        [3, 0, 2, 1],
        2,
        conditioned_indices=[3],
    )
    assert selected == [0, 1]
    assert 3 not in selected
    assert [row["candidate_index"] for row in trace] == [0, 1]


def test_schur_near_tie_still_selects_the_strictly_larger_variance() -> None:
    kernel = np.diag(np.asarray([1.0, 1.0 + 1e-10], dtype=np.float64))
    selected, trace = greedy_conditional_schur_select(
        kernel,
        [0, 1],
        1,
    )
    assert selected == [1]
    assert trace[0]["selection_score"] == pytest.approx(1.0 + 1e-10)


def test_schur_numerical_failure_is_explicit_without_fallback() -> None:
    kernel = np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="Cholesky failed"):
        conditional_schur_variances(
            kernel,
            [1],
            conditioned_indices=[0],
            jitter=0.0,
            max_jitter_tries=2,
        )


def test_k1_selector_matches_direct_global_kernel_schur() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(8, 3))
    kernel = rbf_kernel(features, lengthscale=1.2)
    selected, _ = greedy_conditional_schur_select(
        kernel, list(range(8)), 5, jitter=1e-8,
    )
    direct: list[int] = []
    for _ in range(5):
        remaining = [index for index in range(8) if index not in direct]
        scores = {
            index: _brute_variance(kernel, index, direct, 1e-8)
            for index in remaining
        }
        best = max(scores.values())
        direct.append(
            min(
                index
                for index, value in scores.items()
                if np.isclose(value, best, rtol=1e-12, atol=1e-15)
            )
        )
    assert selected == direct


def test_clustered_schur_consumes_only_precomputed_global_kernel(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        bo_phase4,
        "ted_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("features must not be recomputed inside a cluster")
        ),
    )
    monkeypatch.setattr(
        bo_phase4,
        "rbf_kernel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("kernel must not be rebuilt inside a cluster")
        ),
    )
    kernel = np.eye(6, dtype=np.float64)
    selected, _ = bo_phase4._select_clustered_schur(
        kernel=kernel,
        cluster_ids=np.asarray([0, 0, 0, 1, 1, 1]),
        quotas={0: 2, 1: 2},
        jitter=1e-8,
        jitter_multiplier=10.0,
        max_jitter_tries=8,
    )
    assert selected == [0, 1, 3, 4]


def test_schur_and_ted_share_design_features_and_rbf_implementation() -> None:
    assert bo_phase4.ted_features is ted_features
    assert bo_phase4.rbf_kernel is rbf_kernel
    rows = np.zeros((2, 19), dtype=np.float64)
    rows[1, 12:] = 1.0
    masks = np.ones_like(rows)
    masks[:, 12:] = 0.0
    features = ted_features(
        rows,
        arch_nz=12,
        hp_mode="hybrid_cond7",
        z_bound=2.5,
        condition_masks=masks,
    )
    kernel = rbf_kernel(features, lengthscale=1.0)
    assert kernel[0, 1] == pytest.approx(1.0)


def _six_cluster_arch(seed: int = 19, points_per_cluster: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.linspace(-5.0, 5.0, 6)
    return np.vstack(
        [
            rng.normal(center, 0.08, size=(points_per_cluster, 12))
            for center in centers
        ]
    )


def test_fixed_k6_gmm_restarts_are_deterministic_and_nonempty() -> None:
    arch = _six_cluster_arch()
    kwargs = {
        "search_seed": 7,
        "candidate_pool_fingerprint": "a" * 64,
        "n_components": 6,
        "restarts": 5,
        "covariance_regularization": 1e-6,
        "max_iter": 100,
    }
    first, first_assignments, first_trace, first_metadata = (
        fit_fixed_pool_gmm_restarts(arch, **kwargs)
    )
    second, second_assignments, second_trace, second_metadata = (
        fit_fixed_pool_gmm_restarts(arch.copy(), **kwargs)
    )
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.variances, second.variances)
    np.testing.assert_array_equal(
        first_assignments["cluster_ids"],
        second_assignments["cluster_ids"],
    )
    assert first_trace == second_trace
    assert first_metadata == second_metadata
    assert first.n_components == 6
    assert first_metadata["n_components"] == 6
    assert first_metadata["restarts"] == 5
    assert set(first_metadata["cluster_capacities"]) == set(range(6))
    assert all(
        capacity > 0
        for capacity in first_metadata["cluster_capacities"].values()
    )
    assert [
        row["restart_seed"] for row in first_trace
    ] == [
        stable_seed(
            7,
            GMM_FIXED_VERSION,
            "a" * 64,
            6,
            restart_id,
        )
        for restart_id in range(5)
    ]
    eligible = [row for row in first_trace if row["selection_eligible"]]
    selected = [row for row in first_trace if row["selected_restart"]]
    assert len(selected) == 1
    assert selected[0]["selection_eligible"]
    assert selected[0]["log_likelihood"] == max(
        row["log_likelihood"] for row in eligible
    )
    assert first_metadata["assignment_fingerprint"] == (
        canonical_assignment_fingerprint(first_assignments["cluster_ids"])
    )
    assert all(
        row["candidate_pool_fingerprint"] == "a" * 64
        for row in first_trace
    )
    assert all(
        row["gmm_preprocessing_fingerprint"]
        == first_metadata["gmm_preprocessing_fingerprint"]
        for row in first_trace
    )


def test_fixed_gmm_assignments_read_z_arch_only_not_hp() -> None:
    arch = _six_cluster_arch(seed=23)
    hp_a = np.zeros((arch.shape[0], 4), dtype=np.float64)
    hp_b = np.random.default_rng(9).uniform(size=(arch.shape[0], 4))
    first = fit_fixed_pool_gmm_restarts(
        np.concatenate((arch, hp_a), axis=1)[:, :12],
        search_seed=3,
        candidate_pool_fingerprint="b" * 64,
        n_components=6,
        restarts=5,
        covariance_regularization=1e-6,
    )
    second = fit_fixed_pool_gmm_restarts(
        np.concatenate((arch, hp_b), axis=1)[:, :12],
        search_seed=3,
        candidate_pool_fingerprint="b" * 64,
        n_components=6,
        restarts=5,
        covariance_regularization=1e-6,
    )
    np.testing.assert_array_equal(
        first[1]["cluster_ids"],
        second[1]["cluster_ids"],
    )
    assert first[3]["assignment_fingerprint"] == second[3][
        "assignment_fingerprint"
    ]


def test_component_permutation_has_canonical_parameter_and_assignment_identity() -> None:
    means = np.asarray(
        [[-2.0, -1.0], [0.0, 0.5], [2.0, 1.0]],
        dtype=np.float64,
    )
    parameters = DiagonalGMMParameters(
        weights=np.asarray([0.2, 0.3, 0.5]),
        means=means,
        variances=np.ones_like(means) * 0.2,
        source="gmm_fit_pool",
        source_fingerprint="source",
        standardize_mean=np.zeros(2),
        standardize_std=np.ones(2),
    )
    permutation = [2, 0, 1]
    permuted = DiagonalGMMParameters(
        weights=parameters.weights[permutation],
        means=parameters.means[permutation],
        variances=parameters.variances[permutation],
        source=parameters.source,
        source_fingerprint=parameters.source_fingerprint,
        standardize_mean=parameters.standardize_mean,
        standardize_std=parameters.standardize_std,
    )
    canonical, _ = canonicalize_diagonal_gmm_parameters(parameters)
    canonical_permuted, _ = canonicalize_diagonal_gmm_parameters(permuted)
    assert canonical.parameter_fingerprint == (
        canonical_permuted.parameter_fingerprint
    )
    points = np.asarray([[-2.1, -1.0], [0.1, 0.4], [2.1, 1.1]])
    first_ids = assign_clusters(points, canonical)["cluster_ids"]
    second_ids = assign_clusters(points, canonical_permuted)["cluster_ids"]
    np.testing.assert_array_equal(first_ids, second_ids)
    assert canonical_assignment_fingerprint(first_ids) == (
        canonical_assignment_fingerprint(second_ids)
    )


def test_fixed_k6_all_restart_failures_are_explicit(monkeypatch) -> None:
    monkeypatch.setattr(
        "initialization_gmm_schur._fit_standardized_pool_gmm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("forced failure")
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="no eligible fixed K=6.*six|no eligible fixed K=6",
    ):
        fit_fixed_pool_gmm_restarts(
            _six_cluster_arch(),
            search_seed=3,
            candidate_pool_fingerprint="c" * 64,
            n_components=6,
            restarts=5,
            covariance_regularization=1e-6,
        )


def test_quota_capacity_failure_reports_requested_available_and_unfilled() -> None:
    with pytest.raises(
        ValueError,
        match=r"requested=200.*available_capacity=199.*unfilled_count=1",
    ):
        allocate_cluster_quotas(
            {0: 99, 1: 100},
            200,
            mode="hybrid",
            component_weights={0: 0.5, 1: 0.5},
            equal_weight=0.5,
        )
    with pytest.raises(ValueError, match="unfilled_count=1"):
        assert_exact_cluster_quotas(
            {0: 99, 1: 100},
            {0: 99, 1: 100},
            200,
            stage="shortlist",
        )


def test_schur_negative_variance_tolerance_clips_only_roundoff() -> None:
    nearly_psd = np.asarray(
        [[1.0, 1.0 + 1e-10], [1.0 + 1e-10, 1.0]],
        dtype=np.float64,
    )
    scores, _ = conditional_schur_variances(
        nearly_psd,
        [1],
        conditioned_indices=[0],
        jitter=0.0,
        negative_tolerance=1e-7,
    )
    assert scores.tolist() == [0.0]
    with pytest.raises(RuntimeError, match="materially negative"):
        conditional_schur_variances(
            np.asarray([[1.0, 1.1], [1.1, 1.0]], dtype=np.float64),
            [1],
            conditioned_indices=[0],
            jitter=0.0,
            negative_tolerance=1e-7,
        )


def test_bic_formula_parameter_count_and_tie_breaks() -> None:
    assert diagonal_gmm_parameter_count(3, 12) == 3 * (2 * 12 + 1) - 1
    expected = -2.0 * -120.0 + diagonal_gmm_parameter_count(3, 12) * np.log(50)
    assert diagonal_gmm_bic(
        -120.0, n_samples=50, n_components=3, n_features=12,
    ) == pytest.approx(expected)

    records = [
        {
            "n_components": 1,
            "restart_id": 1,
            "fit_success": True,
            "converged": True,
            "parameters_finite": True,
            "log_likelihood": -10.0,
            "bic": 100.0,
        },
        {
            "n_components": 1,
            "restart_id": 0,
            "fit_success": True,
            "converged": True,
            "parameters_finite": True,
            "log_likelihood": -10.0,
            "bic": 100.0,
        },
        {
            "n_components": 2,
            "restart_id": 0,
            "fit_success": True,
            "converged": True,
            "parameters_finite": True,
            "log_likelihood": -9.0,
            "bic": 100.0,
        },
        {
            "n_components": 3,
            "restart_id": 0,
            "fit_success": False,
            "converged": False,
            "parameters_finite": False,
            "log_likelihood": None,
            "bic": None,
        },
    ]
    selected_k, selected_restart, marked = select_bic_model_records(records)
    assert (selected_k, selected_restart) == (1, 0)
    assert next(
        row for row in marked
        if row["n_components"] == 1 and row["restart_id"] == 0
    )["is_final_model"]


def test_nonconverged_finite_restart_cannot_win_bic_selection() -> None:
    records = [
        {
            "n_components": 1,
            "restart_id": 0,
            "fit_success": True,
            "converged": False,
            "parameters_finite": True,
            "log_likelihood": -1.0,
            "bic": 1.0,
        },
        {
            "n_components": 1,
            "restart_id": 1,
            "fit_success": True,
            "converged": True,
            "parameters_finite": True,
            "log_likelihood": -10.0,
            "bic": 20.0,
        },
    ]
    selected_k, selected_restart, marked = select_bic_model_records(records)
    assert (selected_k, selected_restart) == (1, 1)
    excluded = next(row for row in marked if row["restart_id"] == 0)
    assert excluded["selection_eligible"] is False
    assert excluded["selection_exclusion_reason"] == "not_converged"
    assert excluded["k_available"] is True


def test_bic_skips_component_count_when_all_restarts_are_nonconverged() -> None:
    records = [
        {
            "n_components": 1,
            "restart_id": 0,
            "fit_success": True,
            "converged": True,
            "parameters_finite": True,
            "log_likelihood": -10.0,
            "bic": 50.0,
        },
        {
            "n_components": 2,
            "restart_id": 0,
            "fit_success": True,
            "converged": False,
            "parameters_finite": True,
            "log_likelihood": -1.0,
            "bic": 1.0,
        },
        {
            "n_components": 2,
            "restart_id": 1,
            "fit_success": True,
            "converged": False,
            "parameters_finite": True,
            "log_likelihood": -2.0,
            "bic": 2.0,
        },
    ]
    selected_k, selected_restart, marked = select_bic_model_records(records)
    assert (selected_k, selected_restart) == (1, 0)
    unavailable = [row for row in marked if row["n_components"] == 2]
    assert all(row["k_available"] is False for row in unavailable)
    assert all(
        row["k_selection_status"] == "unavailable" for row in unavailable
    )
    assert not any(row["is_best_restart_for_k"] for row in unavailable)


def test_bic_all_component_counts_unavailable_raises_explicit_error() -> None:
    records = [
        {
            "n_components": 1,
            "restart_id": 0,
            "fit_success": True,
            "converged": False,
            "parameters_finite": True,
            "log_likelihood": -10.0,
            "bic": 50.0,
        },
        {
            "n_components": 2,
            "restart_id": 0,
            "fit_success": False,
            "converged": False,
            "parameters_finite": False,
            "log_likelihood": None,
            "bic": None,
        },
    ]
    with pytest.raises(
        RuntimeError,
        match="all BIC-GMM component counts are unavailable",
    ):
        select_bic_model_records(records)


def test_bic_range_includes_k1_and_fit_is_deterministic() -> None:
    rng = np.random.default_rng(19)
    arch = np.vstack(
        (
            rng.normal(-1.0, 0.2, size=(16, 3)),
            rng.normal(1.0, 0.2, size=(16, 3)),
        )
    )
    fingerprint = "a" * 64
    kwargs = {
        "search_seed": 7,
        "candidate_pool_fingerprint": fingerprint,
        "min_components": 1,
        "max_components": 3,
        "restarts": 2,
        "covariance_regularization": 1e-6,
        "max_iter": 40,
    }
    first, first_trace = fit_pool_gmm_bic(arch, **kwargs)
    second, second_trace = fit_pool_gmm_bic(arch.copy(), **kwargs)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.variances, second.variances)
    assert first_trace == second_trace
    assert {row["n_components"] for row in first_trace} == {1, 2, 3}
    assert first_trace[0]["restart_seed"] == stable_seed(
        7, "gmm_bic", fingerprint, 1, 0,
    )
    with pytest.raises(ValueError, match="include K=1"):
        fit_pool_gmm_bic(
            arch,
            **{**kwargs, "min_components": 2},
        )


def test_global_shortlist_and_promotion_budgets_are_exact_across_k() -> None:
    for capacities in ({0: 260}, {0: 70, 1: 80, 2: 110}):
        shortlist, _ = allocate_cluster_quotas(
            capacities,
            200,
            mode="hybrid",
            component_weights={key: value for key, value in capacities.items()},
            equal_weight=0.5,
        )
        assert sum(shortlist.values()) == 200
        assert all(shortlist[key] <= capacities[key] for key in shortlist)
        promotion, _ = allocate_cluster_quotas(
            shortlist,
            100,
            mode="hybrid",
            component_weights={key: value for key, value in capacities.items()},
            equal_weight=0.5,
        )
        assert sum(promotion.values()) == 100
        assert all(promotion[key] <= shortlist[key] for key in promotion)


def _config_args(checkpoint, output, strategy: str, **overrides):
    values = {
        "n_init": 2,
        "initial_seed_evals": 2,
        "initial_shortlist_evals": 4,
        "initial_expand_evals": 2,
        "n_iter": 1,
        "max_total_full_evals": 5,
        "initial_selection_strategy": strategy,
        "surrogate_type": "exact_gp",
        "gp_init_mode": "scratch",
        "gp_checkpoint": None,
        "warm_start": "",
        "frozen_init_history": None,
        "gmm_init_history": None,
        "gmm_init_trials": 0,
        "n_lhs_candidates": 12,
        "wgmm_source": "gmm_fit_pool",
        "wgmm_checkpoint": None,
        "wgmm_n_components": None,
        "gmm_component_selection": "bic",
        "gmm_min_components": 1,
        "gmm_max_components": 2,
        "gmm_restarts": 2,
        "wgmm_assignment": "hard",
        "wgmm_quota_mode": "hybrid",
        "wgmm_covariance_regularization": 1e-6,
        "wgmm_equal_weight": 0.5,
        "ted_regularization": 0.1,
        "ted_jitter": 1e-8,
        "ted_kernel_lengthscale": None,
        "ted_shortlist_per_cluster": 100,
        "schur_jitter": 1e-8,
        "schur_jitter_multiplier": 10.0,
        "schur_jitter_max_tries": 8,
        "low_fidelity_epochs": 20,
        "low_fidelity_patience": 0,
        "low_fidelity_score_weight": 0.5,
        "gp_mean_score_weight": 0.25,
        "gp_std_score_weight": 0.25,
        "adaptive_sampling": False,
        "min_bo_samples": 1,
        "max_bo_samples": 2,
        "convergence_check_every": 1,
        "convergence_patience": 2,
        "prequential_window": 2,
        "mae_relative_tol": 0.01,
        "mae_absolute_tol": 0.002,
        "std_relative_tol": 0.02,
        "spearman_tol": 0.01,
        "degradation_tolerance": 0.01,
        "best_acc_patience": 2,
        "best_acc_min_delta": 0.001,
        "max_wall_time_hours": 1.0,
        "probe_pool_size": 8,
        "probe_pool_seed": None,
        "output": str(output),
        "checkpoint": str(checkpoint),
        "hp_mode": "global4",
        "z_bound": 2.5,
        "seed": 3,
        "sigma_arch": 0.8,
        "gmm_var_floor": 1e-4,
        "gmm_max_iter": 30,
        "gmm_tol": 1e-4,
        "scratch_gp_min_points": 2,
        "use_conditional_kernel": False,
        "eval_epochs": 100,
        "patience": 20,
        "version": "test_gmm_schur",
        "gcnii_alpha": 0.1,
        "gcnii_theta": 0.5,
        "gp_update_mode": "warm_refit",
        "gp_refit_steps": 2,
        "gp_refit_every": 1,
        "online_candidate_strategy": "qlogei",
        "num_restarts": 2,
        "raw_samples": 8,
        "n_extra": 2,
        "resume_initialization": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _k6_config_args(checkpoint, output, **overrides):
    args = _config_args(
        checkpoint,
        output,
        "gmm_schur_lowfid",
        n_init=50,
        initial_seed_evals=50,
        initial_shortlist_evals=200,
        initial_expand_evals=100,
        n_iter=150,
        max_total_full_evals=300,
        n_lhs_candidates=768,
        wgmm_n_components=6,
        gmm_component_selection="fixed",
        gmm_restarts=5,
        eval_epochs=150,
        patience=40,
        low_fidelity_epochs=20,
        low_fidelity_patience=0,
        convergence_check_every=1000,
        max_bo_samples=150,
        probe_pool_size=8,
        gp_refit_every=1000,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_mock_evaluator(monkeypatch, calls):
    def fake_eval(_vae, z, _data, _in, _out, args, _device, **kwargs):
        stage = str(kwargs.get("evaluation_stage"))
        fidelity = str(kwargs.get("evaluation_fidelity"))
        calls.append((args.initial_selection_strategy, stage, fidelity))
        z_tensor = torch.as_tensor(z).float()
        hp = {
            "lr": 0.01,
            "dropout": 0.2,
            "hidden_dim": 64,
            "weight_decay": 1e-4,
            "condition_mask_vector": [1.0] * 4,
            "condition_mask": {},
        }
        config = {
            "operations": ["GCNConv", "GATConv"],
            "edges": [[0, 1], [1, 2]],
        }
        decoder_seed = stable_seed(
            int(args.seed), "decoder", z_tensor[: bo_phase4.ARCH_NZ],
        )
        return z_tensor, {
            "val_acc": 0.55 + float(z_tensor[0]) * 1e-3,
            "valid": True,
            "hp": hp,
            "config": config,
            "epochs_ran": kwargs.get("max_epochs_override", args.eval_epochs),
            "candidate_eval_seed": kwargs.get("evaluation_seed"),
            "evaluation_seed": kwargs.get("evaluation_seed"),
            "decoder_seed": decoder_seed,
            "architecture_fingerprint": (
                bo_phase4._decoded_architecture_fingerprint(config)
            ),
            "hp_fingerprint": bo_phase4._hp_configuration_fingerprint(
                {"hp": hp}
            ),
            "search_seed": args.seed,
        }

    monkeypatch.setattr(bo_phase4, "eval_candidate", fake_eval)
    monkeypatch.setattr(
        bo_phase4, "_score_logei", lambda *_args, **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        bo_phase4,
        "schur_greedy_select",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("new clustered Schur must not call legacy Schur")
        ),
    )


def _formal_k6_pool() -> list[torch.Tensor]:
    rng = np.random.default_rng(20260728)
    centers = np.linspace(-2.1, 2.1, 6)
    arch = np.vstack(
        [
            rng.normal(center, 0.025, size=(128, bo_phase4.ARCH_NZ))
            for center in centers
        ]
    )
    hp = rng.uniform(-0.25, 0.25, size=(arch.shape[0], 4))
    pool = np.concatenate((arch, hp), axis=1).astype(np.float32)
    return [
        torch.tensor(
            bo_phase4.clip_z_search_by_mode(
                row, "global4", bo_phase4.ARCH_NZ, 2.5,
            ),
            dtype=torch.float32,
        )
        for row in pool
    ]


class _MockExactPredictor:
    def __init__(self, train_size: int):
        self.train_size = int(train_size)
        self.offline_train_size = 0
        self.use_conditional_kernel = False
        self.metadata: dict[str, object] = {}
        self.train_observation_counts = torch.ones(
            self.train_size, dtype=torch.long,
        )
        self.holdout_X_raw = None
        self.holdout_Y = None
        self.holdout_condition_masks = None

    def predict_batch(self, rows, condition_masks=None):
        del condition_masks
        values = torch.as_tensor(rows, dtype=torch.float32)
        return [
            {
                "mean": 0.60 + float(row[0]) * 1e-3,
                "std": 0.10 + abs(float(row[1])) * 1e-4,
                "lower_95": 0.40,
                "upper_95": 0.80,
            }
            for row in values
        ]

    def predict(self, row, condition_mask=None):
        del condition_mask
        return self.predict_batch(torch.as_tensor(row).reshape(1, -1))[0]

    def is_holdout_point(self, _row):
        return False

    def append_observation(self, _row, _value, condition_mask=None):
        del condition_mask
        self.train_size += 1
        self.train_observation_counts = torch.cat(
            (self.train_observation_counts, torch.ones(1, dtype=torch.long))
        )

    def refit(self, *, optimize=True, steps=None):
        del optimize, steps

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"train_size": self.train_size}, destination)


def _install_formal_k6_runtime_mocks(monkeypatch, calls):
    _install_mock_evaluator(monkeypatch, calls)
    monkeypatch.setattr(bo_phase4, "make_lhs_pool", lambda _args: _formal_k6_pool())
    fit_sizes: list[int] = []

    def fake_fit(init_valid, _args, _device, _logger):
        fit_sizes.append(len(init_valid))
        return _MockExactPredictor(len(init_valid))

    monkeypatch.setattr(bo_phase4, "_fit_scratch_predictor", fake_fit)
    return fit_sizes


def _read_cluster_ids(path) -> list[int]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [int(row["cluster_id"]) for row in csv.DictReader(handle)]


def test_mock_k6_lowfid_end_to_end_has_exact_budgets_and_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "vae.pt"
    checkpoint.write_bytes(b"mock-vae-checkpoint")
    calls: list[tuple[str, str, str]] = []
    fit_sizes = _install_formal_k6_runtime_mocks(monkeypatch, calls)
    feature_calls = 0
    kernel_calls = 0
    original_ted_features = bo_phase4.ted_features
    original_rbf_kernel = bo_phase4.rbf_kernel

    def counted_features(*args, **kwargs):
        nonlocal feature_calls
        feature_calls += 1
        return original_ted_features(*args, **kwargs)

    def counted_kernel(*args, **kwargs):
        nonlocal kernel_calls
        kernel_calls += 1
        return original_rbf_kernel(*args, **kwargs)

    monkeypatch.setattr(bo_phase4, "ted_features", counted_features)
    monkeypatch.setattr(bo_phase4, "rbf_kernel", counted_kernel)
    monkeypatch.setattr(
        bo_phase4,
        "fit_pool_gmm_bic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixed K=6 strategy must not call BIC selection")
        ),
    )
    online_counter = 0

    def fake_optimize_acq(*_args, **_kwargs):
        nonlocal online_counter
        z_search = torch.zeros(bo_phase4.ARCH_NZ + 4, dtype=torch.float32)
        z_search[0] = 2.30 + online_counter * 5e-4
        z_search[1] = 2.45
        online_counter += 1
        return z_search, 0.0, [1.0] * (bo_phase4.ARCH_NZ + 4)

    monkeypatch.setattr(bo_phase4, "optimize_acq", fake_optimize_acq)
    monkeypatch.setattr(
        bo_phase4,
        "decode_arch",
        lambda *_args, **_kwargs: {
            "operations": ["GCNConv", "GATConv"],
            "edges": [[0, 1], [1, 2]],
        },
    )
    monkeypatch.setattr(bo_phase4, "_plot", lambda *_args, **_kwargs: None)

    args = _k6_config_args(checkpoint, tmp_path / "k6")
    logger = logging.getLogger("test_gmm_schur_k6_mock")
    bo_phase4.run_wgmm_bo(
        object(), object(), 3, 2, args, torch.device("cpu"), logger,
    )
    run_dir = tmp_path / "k6"
    config = json.loads(
        (run_dir / "gmm_schur_lowfid_initialization_config.json").read_text()
    )
    budget = json.loads((run_dir / "budget_summary.json").read_text())
    low_history = json.loads((run_dir / "low_fidelity_history.json").read_text())
    full_history = json.loads((run_dir / "history_final.json").read_text())
    seed_indices = json.loads(
        (run_dir / "selected_seed_indices.json").read_text()
    )
    shortlist_indices = json.loads(
        (run_dir / "selected_shortlist_indices.json").read_text()
    )
    promoted_indices = json.loads(
        (run_dir / "selected_expand_indices.json").read_text()
    )

    assert feature_calls == 1
    assert kernel_calls == 1
    assert fit_sizes == [50, 150]
    assert config["gmm_fixed_components"] == 6
    assert config["selected_gmm_n_components"] == 6
    assert len(config["cluster_capacities"]) == 6
    assert all(int(value) > 0 for value in config["cluster_capacities"].values())
    assert len(seed_indices) == 50
    assert len(shortlist_indices) == 200
    assert len(promoted_indices) == 100
    assert set(seed_indices).isdisjoint(shortlist_indices)
    assert set(promoted_indices).issubset(shortlist_indices)
    assert len(set(promoted_indices)) == 100
    assert len(low_history) == 200
    assert len(full_history) == 300
    assert all(row["evaluation_fidelity"] == "low" for row in low_history)
    assert all(row["evaluation_fidelity"] == "full" for row in full_history)
    assert sum(fidelity == "full" for _, _, fidelity in calls) == 300
    assert sum(fidelity == "low" for _, _, fidelity in calls) == 200
    assert budget["initial_seed_full_evals"] == 50
    assert budget["initial_expand_full_evals"] == 100
    assert budget["completed_online_full_evals"] == 150
    assert budget["completed_total_full_evals"] == 300
    assert budget["completed_low_fidelity_count"] == 200
    assert budget["low_fidelity_counted_in_full_budget"] is False
    assert budget["low_fidelity_equivalent_full_evaluations"] == pytest.approx(
        200 * 20 / 150,
    )
    assert budget["planned_low_fidelity_equivalent_full_evaluations"] == (
        pytest.approx(200 * 20 / 150)
    )
    assert {
        stage: {
            "requested": budget[f"{stage}_requested_count"],
            "completed": budget[
                "promotion_full_completed_count"
                if stage == "promotion"
                else f"{stage}_completed_count"
            ],
            "invalid": budget[f"{stage}_invalid_count"],
            "replenished": budget[f"{stage}_replenished_count"],
        }
        for stage in ("seed", "shortlist", "promotion")
    } == {
        "seed": {
            "requested": 50,
            "completed": 50,
            "invalid": 0,
            "replenished": 0,
        },
        "shortlist": {
            "requested": 200,
            "completed": 200,
            "invalid": 0,
            "replenished": 0,
        },
        "promotion": {
            "requested": 100,
            "completed": 100,
            "invalid": 0,
            "replenished": 0,
        },
    }
    assert all("ted_score" not in row for row in full_history)

    low_by_index = {
        int(row["candidate_pool_index"]): row for row in low_history
    }
    promoted_full = [
        row
        for row in full_history
        if row["evaluation_stage"] == "initial_expand"
    ]
    assert len(promoted_full) == 100
    for full in promoted_full:
        if full["evaluation_stage"] != "initial_expand":
            continue
        low = low_by_index[int(full["candidate_pool_index"])]
        assert low["candidate_fingerprint"] == full["candidate_fingerprint"]
        assert low["decoder_seed"] == full["decoder_seed"]
        assert low["architecture_fingerprint"] == full["architecture_fingerprint"]
        assert low["hp_fingerprint"] == full["hp_fingerprint"]
        assert low["evaluation_seed"] != full["evaluation_seed"]
    assert all(
        bo_phase4.candidate_evaluation_seed(
            3, row["candidate_fingerprint"], "full",
        )
        == row["evaluation_seed"]
        for row in full_history
    )

    for required in (
        "gmm_fixed_restarts.csv",
        "gmm_cluster_assignments.csv",
        "gmm_cluster_quotas.json",
        "schur_seed_trace.csv",
        "schur_shortlist_trace.csv",
        "gmm_schur_lowfid_promotion.csv",
        "gmm_schur_lowfid_initialization_config.json",
    ):
        assert (run_dir / required).exists()
    with open(
        run_dir / "schur_shortlist_trace.csv",
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        trace = list(csv.DictReader(handle))
    seed_counts = {
        int(cluster): int(quota)
        for cluster, quota in config["seed_quotas"].items()
    }
    for cluster in sorted({int(row["cluster_id"]) for row in trace}):
        rows = [
            row for row in trace if int(row["cluster_id"]) == cluster
        ]
        assert int(rows[0]["conditioned_count"]) == seed_counts[cluster]
        assert [
            int(row["conditioned_count"]) for row in rows
        ] == list(
            range(seed_counts[cluster], seed_counts[cluster] + len(rows))
        )
    assert all(
        row["selection_criterion"] == "schur_conditional_variance"
        for row in trace
    )


def test_low_fidelity_gpu_seconds_are_summed_and_validated() -> None:
    assert bo_phase4._sum_low_fidelity_gpu_seconds([]) == pytest.approx(0.0)
    assert bo_phase4._sum_low_fidelity_gpu_seconds(
        [
            {"gpu_seconds": 1.25},
            {"gpu_seconds": None},
            {"gpu_seconds": 2.75},
        ]
    ) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        bo_phase4._sum_low_fidelity_gpu_seconds(
            [{"gpu_seconds": float("nan")}]
        )


def test_k6_resume_rejects_kernel_or_jitter_fingerprint_changes(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "vae.pt"
    checkpoint.write_bytes(b"mock-vae-checkpoint")
    calls: list[tuple[str, str, str]] = []
    _install_formal_k6_runtime_mocks(monkeypatch, calls)
    args = _k6_config_args(checkpoint, tmp_path / "run")
    logger = logging.getLogger("test_gmm_schur_resume")
    bo_phase4.run_wgmm_two_stage_initialization(
        object(), object(), 3, 2, args, torch.device("cpu"), logger,
    )
    for field, value in (
        ("ted_kernel_lengthscale", 0.75),
        ("schur_jitter", 1e-6),
    ):
        changed = copy.deepcopy(args)
        changed.resume_initialization = True
        setattr(changed, field, value)
        with pytest.raises(ValueError, match="fingerprint"):
            bo_phase4.run_wgmm_two_stage_initialization(
                object(), object(), 3, 2, changed, torch.device("cpu"), logger,
            )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gmm_restarts", 3, "gmm_restarts 5"),
        ("wgmm_quota_mode", "equal", "quota_mode hybrid"),
        ("wgmm_equal_weight", 0.7, "equal_weight 0.5"),
        ("initial_shortlist_evals", 199, "fixed K6-LF protocol"),
        ("low_fidelity_epochs", 19, "fixed K6-LF protocol"),
    ],
)
def test_k6_resume_rejects_protocol_changes_before_artifact_reuse(
    tmp_path,
    field,
    value,
    message,
) -> None:
    checkpoint = tmp_path / "vae.pt"
    checkpoint.write_bytes(b"mock-vae-checkpoint")
    args = _k6_config_args(checkpoint, tmp_path / "run")
    args.resume_initialization = True
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        bo_phase4.validate_two_stage_initialization_config(args)


def test_candidate_evaluation_seed_ignores_method_cluster_stage_and_rank() -> None:
    fingerprint = "d" * 64
    expected = bo_phase4.candidate_evaluation_seed(9, fingerprint, "full")
    metadata_variants = [
        {
            "initialization_strategy": "gmm_schur_lowfid",
            "cluster_id": 0,
            "evaluation_stage": "initial_seed",
            "selection_rank": 0,
        },
        {
            "initialization_strategy": "wgmm_ted_lowfid",
            "cluster_id": 5,
            "evaluation_stage": "initial_expand",
            "selection_rank": 99,
        },
    ]
    assert metadata_variants[0] != metadata_variants[1]
    assert all(
        bo_phase4.candidate_evaluation_seed(9, fingerprint, "full") == expected
        for _metadata in metadata_variants
    )


def test_cli_defaults_keep_legacy_schur_and_fixed_components(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bo_phase4.py"])
    args = bo_phase4.parse_args()
    assert args.initial_selection_strategy == "schur"
    assert args.gmm_component_selection == "fixed"
    assert args.initial_shortlist_evals is None
