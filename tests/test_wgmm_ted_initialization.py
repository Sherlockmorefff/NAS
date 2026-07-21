from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bo_phase4
from eval_utils import stable_seed
from initialization_wgmm_ted import (
    DiagonalGMMParameters,
    allocate_cluster_quotas,
    assign_clusters,
    combined_expansion_scores,
    fit_pool_gmm,
    greedy_ted_select,
    load_checkpoint_wgmm,
    percentile_ranks,
    rbf_kernel,
    select_by_cluster_scores,
    ted_features,
)
from weighted_diag_gmm_init import WeightedDiagonalGMM, standardize_fit


def test_greedy_ted_matches_brute_force_trace_reduction_and_ties() -> None:
    kernel = np.asarray(
        [
            [1.0, 0.8, 0.1],
            [0.8, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    regularization = 0.1
    brute = np.sum(kernel * kernel, axis=0) / (np.diag(kernel) + regularization)
    selected, trace = greedy_ted_select(
        kernel, [7, 3, 9], 3, regularization=regularization, jitter=1e-12,
    )
    assert selected[0] == [7, 3, 9][int(np.argmax(brute))]
    assert trace[0]["ted_score"] == pytest.approx(float(np.max(brute)))
    covariance = kernel.copy()
    original_indices = [7, 3, 9]
    active = list(range(3))
    for rank, chosen_index in enumerate(selected):
        trial_updates = {}
        trial_reductions = {}
        for position in active:
            denominator = covariance[position, position] + regularization
            updated = covariance - np.outer(
                covariance[:, position], covariance[position, :],
            ) / denominator
            trial_updates[position] = updated
            trial_reductions[position] = float(np.trace(covariance) - np.trace(updated))
        best_reduction = max(trial_reductions.values())
        tied = [
            position for position, reduction in trial_reductions.items()
            if np.isclose(reduction, best_reduction, rtol=1e-12, atol=1e-15)
        ]
        expected_position = min(tied, key=lambda position: original_indices[position])
        assert chosen_index == original_indices[expected_position]
        assert trace[rank]["ted_score"] == pytest.approx(best_reduction)
        assert trace[rank]["trace_reduction"] == pytest.approx(best_reduction)
        covariance = trial_updates[expected_position]
        active.remove(expected_position)
    assert all(row["trace_after"] <= row["trace_before"] + 1e-12 for row in trace)
    assert np.isfinite([row["ted_score"] for row in trace]).all()

    identity_selected, _ = greedy_ted_select(
        np.eye(3), [8, 2, 5], 1, regularization=regularization,
    )
    assert identity_selected == [2]


def test_schur_diagonal_and_ted_trace_can_select_different_points() -> None:
    covariance = np.asarray(
        [
            [1.9191596527, 1.1465964143, 0.3812663229, 0.8661784561],
            [1.1465964143, 2.7084088162, -0.0822847056, 0.2261629402],
            [0.3812663229, -0.0822847056, 2.4945594774, 1.6181117688],
            [0.8661784561, 0.2261629402, 1.6181117688, 2.2683107497],
        ]
    )
    schur_pick = int(np.argmax(np.diag(covariance)))
    ted_pick = greedy_ted_select(
        covariance, range(4), 1, regularization=0.1,
    )[0][0]
    assert schur_pick == 1
    assert ted_pick == 3


def test_wgmm_uses_only_arch_latent_and_responsibilities_are_deterministic() -> None:
    arch = np.vstack((np.full((8, 3), -2.0), np.full((8, 3), 2.0)))
    params = fit_pool_gmm(
        arch, n_components=2, random_state=19,
        covariance_regularization=1e-6, max_iter=30,
    )
    first = assign_clusters(arch, params)
    second = assign_clusters(arch.copy(), params)
    np.testing.assert_array_equal(first["cluster_ids"], second["cluster_ids"])
    np.testing.assert_allclose(first["responsibilities"].sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        first["cluster_ids"], np.argmax(first["responsibilities"], axis=1),
    )

    # HP is deliberately changed but never passed to the assignment API.
    full_a = np.column_stack((arch, np.zeros((16, 4))))
    full_b = np.column_stack((arch, np.ones((16, 4))))
    np.testing.assert_array_equal(
        assign_clusters(full_a[:, :3], params)["cluster_ids"],
        assign_clusters(full_b[:, :3], params)["cluster_ids"],
    )
    assert len(params.parameter_fingerprint) == 64
    assert params.to_json()["parameter_fingerprint"] == params.parameter_fingerprint


def test_hp_does_not_change_wgmm_assignment_but_does_change_joint_ted_distance() -> None:
    arch = np.vstack((np.full((8, 3), -1.5), np.full((8, 3), 1.5)))
    params = fit_pool_gmm(
        arch,
        n_components=2,
        random_state=23,
        covariance_regularization=1e-6,
        max_iter=30,
    )
    same_arch = np.repeat(arch[:1], 2, axis=0)
    rows = np.column_stack((same_arch, np.asarray([[0.0] * 4, [1.0] * 4])))
    assignments = assign_clusters(same_arch, params)
    assert assignments["cluster_ids"][0] == assignments["cluster_ids"][1]

    features = ted_features(
        rows,
        arch_nz=3,
        hp_mode="global4",
        z_bound=2.5,
        condition_masks=np.ones_like(rows),
    )
    kernel = rbf_kernel(features, lengthscale=1.0)
    assert kernel[0, 1] < 1.0


def test_wgmm_parameter_fingerprint_changes_with_effective_parameters() -> None:
    base = DiagonalGMMParameters(
        weights=np.asarray([0.4, 0.6]),
        means=np.zeros((2, 3)),
        variances=np.ones((2, 3)),
        source="checkpoint",
        source_fingerprint="a" * 64,
    ).validate(3)
    changed = DiagonalGMMParameters(
        weights=base.weights.copy(),
        means=base.means.copy(),
        variances=base.variances.copy(),
        source=base.source,
        source_fingerprint=base.source_fingerprint,
    )
    changed.variances[0, 0] = 1.5
    assert base.parameter_fingerprint != changed.parameter_fingerprint


def test_fit_pool_uniform_weights_are_ordinary_diagonal_gmm() -> None:
    rng = np.random.default_rng(31)
    arch = rng.normal(size=(24, 3))
    params = fit_pool_gmm(
        arch,
        n_components=3,
        random_state=17,
        covariance_regularization=1e-6,
        max_iter=40,
    )
    standardized, _, _ = standardize_fit(arch)
    ordinary = WeightedDiagonalGMM(
        n_components=3,
        random_state=17,
        reg_covar=1e-6,
        max_iter=40,
    ).fit(standardized, sample_weight=None)
    np.testing.assert_allclose(params.weights, ordinary.weights_)
    np.testing.assert_allclose(params.means, ordinary.means_)
    np.testing.assert_allclose(params.variances, ordinary.vars_)
    assert params.estimator_semantics == "ordinary_diagonal_gmm_uniform_candidate_weights"


def test_checkpoint_wgmm_load_and_incompatible_checkpoint_error(tmp_path) -> None:
    good = tmp_path / "wgmm.pt"
    torch.save(
        {
            "wgmm": {
                "weights": torch.tensor([0.4, 0.6]),
                "means": torch.zeros(2, 3),
                "variances": torch.ones(2, 3),
            }
        },
        good,
    )
    loaded = load_checkpoint_wgmm(good, arch_dim=3)
    assert loaded.n_components == 2
    assert loaded.source == "checkpoint"
    assert loaded.estimator_semantics == "serialized_checkpoint_diagonal_mixture"
    assert bo_phase4._initialization_method_id("wgmm_ted", loaded.source) == (
        "wgmm_ted__checkpoint"
    )
    assert bo_phase4._initialization_method_id("wgmm_ted", "gmm_fit_pool") == (
        "wgmm_ted__gmm_fit_pool"
    )
    bad = tmp_path / "vae.pt"
    torch.save({"model": torch.ones(1)}, bad)
    with pytest.raises(ValueError, match="no recoverable WGMM"):
        load_checkpoint_wgmm(bad, arch_dim=3)


def test_quota_modes_capacity_redistribution_and_exact_budget() -> None:
    capacities = {0: 1, 1: 7, 2: 0, 3: 4}
    weights = {0: 0.7, 1: 0.2, 3: 0.1}
    for mode in ("equal", "proportional", "hybrid"):
        first, diagnostics = allocate_cluster_quotas(
            capacities, 8, mode=mode, component_weights=weights, equal_weight=0.5,
        )
        second, _ = allocate_cluster_quotas(
            capacities, 8, mode=mode, component_weights=weights, equal_weight=0.5,
        )
        assert first == second
        assert sum(first.values()) == 8
        assert all(first[key] <= capacities[key] for key in first)
        assert 2 not in first
        assert diagnostics["budget"] == 8

    small, _ = allocate_cluster_quotas(
        {0: 3, 1: 3, 2: 3}, 2, mode="equal",
    )
    assert small == {0: 1, 1: 1, 2: 0}


def test_conditional_ted_features_suppress_inactive_hp_dimensions() -> None:
    rows = np.zeros((2, 19), dtype=np.float64)
    rows[:, :12] = 0.25
    rows[1, 12:] = 1.0
    masks = np.ones_like(rows)
    masks[:, 12:] = 0.0
    transformed = ted_features(
        rows, arch_nz=12, hp_mode="hybrid_cond7", z_bound=2.5,
        condition_masks=masks,
    )
    np.testing.assert_allclose(transformed[0], transformed[1])
    assert np.linalg.norm(transformed[0] - transformed[1]) == pytest.approx(0.0)
    kernel = rbf_kernel(transformed, lengthscale=1.0)
    assert kernel[0, 1] == pytest.approx(1.0)


def test_scoring_invalids_ties_and_cluster_quota_selection() -> None:
    ranks = percentile_ranks([0.7, 0.7, 0.9], [4, 2, 8])
    assert ranks[1] > ranks[0]
    scores = combined_expansion_scores(
        [2, 4, 6, 8],
        [0.8, np.nan, 0.7, 0.9],
        [0.5, 0.6, 0.4, 0.7],
        [0.2, 0.3, 0.1, 0.4],
        low_fidelity_valid=[True, False, True, True],
        weights=(0.5, 0.25, 0.25),
    )
    assert scores["combined_score"][1] == -1.0
    selected = select_by_cluster_scores(
        [2, 4, 6, 8], [0, 0, 1, 1], scores["combined_score"], {0: 1, 1: 1},
    )
    assert len(selected) == 2 and 4 not in selected
    with pytest.raises(ValueError, match="sum to 1"):
        combined_expansion_scores(
            [1], [1.0], [1.0], [1.0], low_fidelity_valid=[True],
            weights=(0.5, 0.4, 0.4),
        )


def _config_args(**overrides):
    values = {
        "n_init": 50,
        "initial_seed_evals": 50,
        "initial_expand_evals": 150,
        "n_iter": 100,
        "max_total_full_evals": 300,
        "initial_selection_strategy": "wgmm_ted_lowfid",
        "surrogate_type": "exact_gp",
        "gp_init_mode": "scratch",
        "gp_checkpoint": None,
        "warm_start": "",
        "frozen_init_history": None,
        "gmm_init_history": None,
        "gmm_init_trials": 0,
        "n_lhs_candidates": 500,
        "wgmm_source": "gmm_fit_pool",
        "wgmm_checkpoint": None,
        "wgmm_n_components": 4,
        "wgmm_assignment": "hard",
        "wgmm_quota_mode": "hybrid",
        "wgmm_covariance_regularization": 1e-6,
        "wgmm_equal_weight": 0.5,
        "ted_regularization": 0.1,
        "ted_jitter": 1e-8,
        "ted_kernel_lengthscale": None,
        "ted_shortlist_per_cluster": 100,
        "low_fidelity_epochs": 20,
        "low_fidelity_patience": 0,
        "low_fidelity_score_weight": 0.5,
        "gp_mean_score_weight": 0.25,
        "gp_std_score_weight": 0.25,
        "adaptive_sampling": False,
        "min_bo_samples": 20,
        "max_bo_samples": 60,
        "convergence_check_every": 5,
        "convergence_patience": 3,
        "prequential_window": 10,
        "mae_relative_tol": 0.01,
        "mae_absolute_tol": 0.002,
        "std_relative_tol": 0.02,
        "spearman_tol": 0.01,
        "degradation_tolerance": 0.01,
        "best_acc_patience": 20,
        "best_acc_min_delta": 0.001,
        "max_wall_time_hours": 4.0,
        "probe_pool_size": 128,
        "probe_pool_seed": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_budget_validation_and_legacy_default_compatibility() -> None:
    bo_phase4.validate_two_stage_initialization_config(_config_args())
    with pytest.raises(ValueError, match="budget mismatch"):
        bo_phase4.validate_two_stage_initialization_config(
            _config_args(max_total_full_evals=299),
        )
    legacy = _config_args(
        initial_selection_strategy="schur", initial_seed_evals=None,
        initial_expand_evals=0, n_init=20, n_iter=60,
        max_total_full_evals=None, warm_start="some-existing-default",
        gp_init_mode="checkpoint", surrogate_type="exact_gp",
    )
    bo_phase4.validate_two_stage_initialization_config(legacy)


def test_cli_default_is_legacy_schur_and_run_bo_dispatches_only_when_selected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["bo_phase4.py"])
    assert bo_phase4.parse_args().initial_selection_strategy == "schur"
    called = []
    monkeypatch.setattr(
        bo_phase4,
        "run_wgmm_bo",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )
    args = SimpleNamespace(initial_selection_strategy="wgmm_ted")
    bo_phase4.run_bo(
        object(), object(), 1, 1, args, torch.device("cpu"),
        logging.getLogger("dispatch"), None,
    )
    assert len(called) == 1


def test_fidelity_seed_is_order_independent_and_stage_isolated() -> None:
    fingerprints = ["a" * 64, "b" * 64, "c" * 64]
    forward = {
        value: stable_seed(7, "candidate_evaluation", value, "full", "initial_expand")
        for value in fingerprints
    }
    reverse = {
        value: stable_seed(7, "candidate_evaluation", value, "full", "initial_expand")
        for value in reversed(fingerprints)
    }
    assert forward == reverse
    assert forward[fingerprints[0]] != stable_seed(
        7, "candidate_evaluation", fingerprints[0], "low", "initial_shortlist",
    )


def test_low_full_eval_reuses_decode_and_hp_but_isolates_training_seed(monkeypatch) -> None:
    hp = {
        "lr": 0.01,
        "dropout": 0.2,
        "hidden_dim": 64,
        "weight_decay": 1e-4,
        "condition_mask_vector": [1.0] * 4,
    }
    config = {"operations": ["GCNConv", "GATConv"], "edges": [[0, 1], [1, 2]]}
    decoder_calls = []

    def fake_eval_z_search(_vae, _z, _data, _in, _out, **kwargs):
        decoder_calls.append(
            (kwargs["decoder_seed"], kwargs["candidate_eval_seed"], kwargs["max_epochs"])
        )
        return {
            "val_acc": 0.75,
            "valid": True,
            "hp": hp,
            "config": config,
            "epochs_ran": kwargs["max_epochs"],
        }

    monkeypatch.setattr(bo_phase4, "eval_z_search", fake_eval_z_search)
    args = SimpleNamespace(
        seed=5,
        hp_mode="global4",
        z_bound=2.5,
        log_lr_min=-4.0,
        log_lr_max=-1.5,
        dropout_min=0.1,
        dropout_max=0.6,
        gcnii_alpha=0.1,
        gcnii_theta=0.5,
        eval_epochs=150,
        patience=40,
    )
    z = torch.linspace(-0.4, 0.4, 16)
    fingerprint = bo_phase4.make_candidate_fingerprint(z.numpy())
    low_z, low = bo_phase4.eval_candidate(
        object(), z, object(), 3, 2, args, torch.device("cpu"),
        evaluation_stage="initial_shortlist",
        evaluation_fidelity="low",
        candidate_fingerprint=fingerprint,
        max_epochs_override=20,
        patience_override=21,
    )
    full_z, full = bo_phase4.eval_candidate(
        object(), z, object(), 3, 2, args, torch.device("cpu"),
        evaluation_stage="initial_expand",
        evaluation_fidelity="full",
        candidate_fingerprint=fingerprint,
    )
    torch.testing.assert_close(low_z, full_z, rtol=0.0, atol=0.0)
    assert low["candidate_fingerprint"] == full["candidate_fingerprint"]
    assert low["decoder_seed"] == full["decoder_seed"] == decoder_calls[0][0]
    assert low["architecture_fingerprint"] == full["architecture_fingerprint"]
    assert low["hp_fingerprint"] == full["hp_fingerprint"]
    assert low["config"] == full["config"] == config
    assert low["hp"] == full["hp"] == hp
    assert low["candidate_eval_seed"] != full["candidate_eval_seed"]
    assert decoder_calls[0][1] != decoder_calls[1][1]
    assert decoder_calls[0][2] == 20
    assert decoder_calls[1][2] == 150


def test_adaptive_resume_requires_clean_committed_boundary() -> None:
    bo_phase4._validate_online_resume_boundary(
        raw_history_count=7,
        completed_count=7,
        adaptive_sampling=True,
        n_iter=120,
        max_bo_samples=60,
    )
    with pytest.raises(ValueError, match="clean committed iteration boundary"):
        bo_phase4._validate_online_resume_boundary(
            raw_history_count=8,
            completed_count=7,
            adaptive_sampling=True,
            n_iter=120,
            max_bo_samples=60,
        )
    bo_phase4._validate_online_resume_boundary(
        raw_history_count=8,
        completed_count=7,
        adaptive_sampling=False,
        n_iter=120,
        max_bo_samples=60,
    )
    with pytest.raises(ValueError, match="more evaluations than the online budget"):
        bo_phase4._validate_online_resume_boundary(
            raw_history_count=61,
            completed_count=61,
            adaptive_sampling=True,
            n_iter=120,
            max_bo_samples=60,
        )


class _FakePredictor:
    use_conditional_kernel = False
    metadata: dict = {}
    offline_train_size = 0

    def __init__(self, train_size: int):
        self.train_size = train_size
        self.train_observation_counts = torch.ones(train_size, dtype=torch.long)

    def predict(self, z, condition_mask=None):
        mean = float(torch.as_tensor(z).sum()) / 100.0
        return {"mean": mean, "std": 0.1, "lower_95": mean - 0.196, "upper_95": mean + 0.196}

    def predict_batch(self, rows, condition_masks=None):
        return [self.predict(row) for row in rows]

    def is_holdout_point(self, _z):
        return False

    def append_observation(self, _z, _value, condition_mask=None):
        self.train_size += 1
        self.train_observation_counts = torch.ones(self.train_size, dtype=torch.long)

    def refit(self, *, optimize=True, steps=None):
        return None


def test_online_full_record_is_persisted_before_gp_mutation(tmp_path, monkeypatch) -> None:
    args = SimpleNamespace(
        seed=9,
        hp_mode="global4",
        z_bound=2.5,
        log_lr_min=-4.0,
        log_lr_max=-1.5,
        dropout_min=0.1,
        dropout_max=0.6,
        eval_epochs=100,
        patience=20,
        version="test_wgmm_ted",
        gcnii_alpha=0.1,
        gcnii_theta=0.5,
        gp_update_mode="warm_refit",
        gp_refit_every=1,
        gp_refit_steps=2,
        gp_init_mode="scratch",
        gp_checkpoint=None,
        surrogate_type="exact_gp",
        online_candidate_strategy="qlogei",
        output=str(tmp_path),
    )
    z = torch.linspace(-0.5, 0.5, 16)
    fingerprint = bo_phase4.make_candidate_fingerprint(z.numpy())

    def fake_eval(*_args, **kwargs):
        return z.clone(), {
            "val_acc": 0.75,
            "valid": True,
            "hp": {
                "lr": 0.01,
                "dropout": 0.2,
                "hidden_dim": 64,
                "weight_decay": 1e-4,
                "condition_mask_vector": [1.0] * 4,
                "condition_mask": {},
            },
            "config": {"operations": ["GCNConv"], "edges": [[0, 1]]},
            "epochs_ran": 100,
            "candidate_eval_seed": kwargs["evaluation_seed"],
            "search_seed": args.seed,
        }

    monkeypatch.setattr(bo_phase4, "eval_candidate", fake_eval)
    predictor = _FakePredictor(2)
    history, X_obs, Y_obs, predictions = [], [], [], []
    callback_state = []

    def persist_before_update():
        callback_state.append(
            (predictor.train_size, len(history), history[-1]["gp_update_performed"])
        )

    bo_phase4._append_eval(
        object(), z, object(), 3, 2, args, torch.device("cpu"),
        history, X_obs, Y_obs, 2, "bo", logging.getLogger("persist-order"),
        predictor, predictions, "online_bo", 0.0, True, 0, [1.0] * 16,
        best_acc=0.7,
        record_metadata={
            "initialization_strategy": "wgmm_ted",
            "evaluation_stage": "online_bo",
            "evaluation_fidelity": "full",
            "candidate_fingerprint": fingerprint,
            "full_evaluation_index": 2,
            "online_iteration": 0,
        },
        pre_update_persist=persist_before_update,
    )
    assert callback_state == [(2, 1, False)]
    assert predictor.train_size == 3
    assert history[0]["gp_update_performed"] is True


def test_mock_two_stage_low_fidelity_and_resume_do_not_duplicate(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "vae.pt"
    checkpoint.write_bytes(b"mock-vae-checkpoint")
    args = _config_args(
        n_init=2,
        initial_seed_evals=2,
        initial_expand_evals=2,
        n_iter=1,
        max_total_full_evals=5,
        n_lhs_candidates=12,
        wgmm_n_components=2,
        ted_shortlist_per_cluster=3,
        output=str(tmp_path / "run"),
        checkpoint=str(checkpoint),
        hp_mode="global4",
        z_bound=2.5,
        seed=0,
        sigma_arch=0.8,
        gmm_var_floor=1e-4,
        gmm_max_iter=30,
        gmm_tol=1e-4,
        scratch_gp_min_points=2,
        use_conditional_kernel=False,
        eval_epochs=100,
        patience=20,
        version="test_wgmm_ted",
        gcnii_alpha=0.1,
        gcnii_theta=0.5,
        gp_update_mode="warm_refit",
        gp_refit_steps=2,
        gp_refit_every=1,
        online_candidate_strategy="qlogei",
        num_restarts=2,
        raw_samples=8,
        n_extra=2,
        resume_initialization=False,
    )
    calls: list[tuple[str, str, int]] = []
    expansion_selections_seen: list[list[int]] = []

    def fake_eval(_vae, z, _data, _in, _out, _args, _device, **kwargs):
        stage = str(kwargs.get("evaluation_stage"))
        fidelity = str(kwargs.get("evaluation_fidelity"))
        calls.append((stage, fidelity, int(kwargs.get("evaluation_seed") or -1)))
        if stage == "initial_expand" and fidelity == "full":
            selected_path = tmp_path / "run" / "selected_expand_indices.json"
            assert selected_path.exists()
            expansion_selections_seen.append(json.loads(selected_path.read_text()))
        hp = {
            "lr": 0.01,
            "dropout": 0.2,
            "hidden_dim": 64,
            "weight_decay": 1e-4,
            "condition_mask_vector": [1.0] * 4,
            "condition_mask": {},
        }
        config = {"operations": ["GCNConv"], "edges": [[0, 1]]}
        decoder_seed = stable_seed(
            int(_args.seed), "decoder", torch.as_tensor(z).float()[: bo_phase4.ARCH_NZ],
        )
        return torch.as_tensor(z).float(), {
            "val_acc": 0.5 + float(torch.as_tensor(z)[0]) * 1e-3,
            "valid": True,
            "hp": hp,
            "config": config,
            "epochs_ran": kwargs.get("max_epochs_override", 100),
            "candidate_eval_seed": kwargs.get("evaluation_seed"),
            "evaluation_seed": kwargs.get("evaluation_seed"),
            "decoder_seed": decoder_seed,
            "architecture_fingerprint": bo_phase4._decoded_architecture_fingerprint(config),
            "hp_fingerprint": bo_phase4._hp_configuration_fingerprint({"hp": hp}),
            "search_seed": args.seed,
        }

    monkeypatch.setattr(bo_phase4, "eval_candidate", fake_eval)
    monkeypatch.setattr(bo_phase4, "_score_logei", lambda *_args, **_kwargs: 0.0)
    logger = logging.getLogger("test_wgmm_ted_mock")
    first = bo_phase4.run_wgmm_two_stage_initialization(
        object(), object(), 3, 2, args, torch.device("cpu"), logger,
    )
    history, _, _, _, valid_rows, predictor, _, _ = first
    assert len(history) == 4
    initialization_config = json.loads(
        (tmp_path / "run" / "initialization_config.json").read_text()
    )
    candidate_metadata = json.loads(
        (tmp_path / "run" / "candidate_pool_metadata.json").read_text()
    )
    assert len(initialization_config["wgmm_parameter_fingerprint"]) == 64
    assert initialization_config["initialization_method_id"] == (
        "wgmm_ted_lowfid__gmm_fit_pool"
    )
    assert (
        candidate_metadata["wgmm"]["parameter_fingerprint"]
        == initialization_config["wgmm_parameter_fingerprint"]
    )
    assert [row["evaluation_stage"] for row in history] == [
        "initial_seed", "initial_seed", "initial_expand", "initial_expand",
    ]
    assert len(valid_rows) == predictor.train_size == 4
    low_history = json.loads((tmp_path / "run" / "low_fidelity_history.json").read_text())
    assert low_history
    assert all(row["evaluation_fidelity"] == "full" for row in history)
    assert all("gp_train_size_before" not in row for row in low_history)
    low_by_index = {int(row["candidate_pool_index"]): row for row in low_history}
    for full_row in history:
        if full_row["evaluation_stage"] != "initial_expand":
            continue
        low_row = low_by_index[int(full_row["candidate_pool_index"])]
        assert low_row["candidate_fingerprint"] == full_row["candidate_fingerprint"]
        assert low_row["z_search"] == full_row["z_search"]
        assert low_row["decoder_seed"] == full_row["decoder_seed"]
        assert low_row["architecture_fingerprint"] == full_row["architecture_fingerprint"]
        assert low_row["hp_fingerprint"] == full_row["hp_fingerprint"]
        assert low_row["operations"] == full_row["operations"]
        assert low_row["edges"] == full_row["edges"]
        assert low_row["hp"]["lr"] == full_row["lr"]
        assert low_row["hp"]["dropout"] == full_row["dropout"]
        assert low_row["hp"]["hidden_dim"] == full_row["hidden_dim"]
        assert low_row["hp"]["weight_decay"] == full_row["l2"]
        assert low_row["hp"]["condition_mask_vector"] == full_row["condition_mask_vector"]
        assert low_row["evaluation_seed"] != full_row["evaluation_seed"]
    assert expansion_selections_seen
    assert all(row == expansion_selections_seen[0] for row in expansion_selections_seen)
    call_count = len(calls)

    args.resume_initialization = True
    second = bo_phase4.run_wgmm_two_stage_initialization(
        object(), object(), 3, 2, args, torch.device("cpu"), logger,
    )
    assert len(second[0]) == 4
    assert len(calls) == call_count
