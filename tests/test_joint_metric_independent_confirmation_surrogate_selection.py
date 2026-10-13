from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import inspect
import json
import math
from pathlib import Path
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "analyse/joint_metric_independent_confirmation_surrogate_selection.py"
SPEC = importlib.util.spec_from_file_location(
    "joint_metric_independent_confirmation_surrogate_selection_for_tests",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


METHODS = ("M1", "M2", "B1", "B3", "B4", "R")
SCORERS = ("M1", "M2", "B1", "B3", "B4")


def _fingerprint(namespace: str, index: int) -> str:
    return hashlib.sha256(f"{namespace}:{index}".encode("utf-8")).hexdigest()


def _previous_pilot_candidates() -> list[dict[str, object]]:
    return [
        {
            "pool_id": index // 4,
            "candidate_index": index % 4,
            "architecture_fingerprint": _fingerprint("pilot-architecture", index),
            "joint_fingerprint": _fingerprint("pilot-joint", index),
            "validation_mean": 0.70 + 0.0001 * index,
        }
        for index in range(32)
    ]


def _candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pool in range(16):
        for slot in range(4):
            index = pool * 4 + slot
            architecture = _fingerprint("confirmation-architecture", index)
            hp = _fingerprint("confirmation-hp", index)
            joint = _fingerprint("confirmation-joint", index)
            lr = 10.0 ** (-3.0 + 0.01 * slot)
            dropout = 0.20 + 0.02 * slot
            hidden = (32, 64, 128, 256)[slot]
            l2 = 10.0 ** (-5.0 + 0.01 * pool)
            rows.append({
                "pool_id": pool,
                "candidate_index": slot,
                "generator_seed": 202608281601 + pool,
                "architecture_fingerprint": architecture,
                "hp_fingerprint": hp,
                "joint_fingerprint": joint,
                "joint_config_fingerprint": joint,
                "operations": ["GCNConv"],
                "edges": [[0, 1], [1, 2]],
                "effective_layers": 1,
                "raw_hp_vector": [0.81, 0.82, 0.83, 0.84],
                "raw_hp": {
                    "lr": lr,
                    "dropout": dropout,
                    "hidden_dim": hidden,
                    "l2": l2,
                },
                "lr": lr,
                "dropout": dropout,
                "hidden_dim": hidden,
                "l2": l2,
                "hp_regime": "hp_high",
                "proposal_distribution": "HIGH_REGIME_PRIOR",
            })
    return rows


def _generator_audit() -> dict[str, object]:
    return {
        "master_seed": 202608281601,
        "pool_seeds": [202608281601 + pool for pool in range(16)],
        "candidate_count": 64,
        "pool_count": 16,
        "candidates_per_pool": 4,
        "proposal_distribution": "HIGH_REGIME_PRIOR",
        "model_blind": True,
        "predictor_accessed": False,
        "validation_labels_accessed": False,
        "historical_top_architectures_used": False,
        "metric_distance_used": False,
        "mutation_or_local_search_used": False,
    }


def _historical_architectures() -> set[str]:
    return {_fingerprint("historical-architecture", index) for index in range(192)}


def _historical_joints() -> set[str]:
    return {_fingerprint("historical-joint", index) for index in range(336)}


def _previous_architectures() -> set[str]:
    return {str(row["architecture_fingerprint"]) for row in _previous_pilot_candidates()}


def _previous_joints() -> set[str]:
    return {str(row["joint_fingerprint"]) for row in _previous_pilot_candidates()}


def _validate_candidates(rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    return gate.validate_candidate_manifest(
        rows if rows is not None else _candidates(),
        historical_architecture_fingerprints=_historical_architectures(),
        historical_joint_fingerprints=_historical_joints(),
        previous_architecture_fingerprints=_previous_architectures(),
        previous_joint_fingerprints=_previous_joints(),
        expected_pool_count=16,
        expected_candidates_per_pool=4,
        expected_master_seed=202608281601,
        expected_distribution="HIGH_REGIME_PRIOR",
        generator_audit=_generator_audit(),
    )


def _prediction_rows(candidates: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in candidates if candidates is not None else _candidates():
        pool = int(row["pool_id"])
        slot = int(row["candidate_index"])
        output.append({
            "pool_id": pool,
            "joint_fingerprint": str(row["joint_fingerprint"]),
            "architecture_fingerprint": str(row["architecture_fingerprint"]),
            "M1": 1.0 if slot == (1 if pool < 6 else 0) else 0.0,
            "M2": 1.0 if slot == (2 if pool < 8 else 0) else 0.0,
            "B1": 1.0 if slot == 2 else 0.0,
            "B3": 1.0 if slot == 0 else 0.0,
            "B4": 1.0 if slot == 3 else 0.0,
        })
    return output


def _joint_for(candidates: list[dict[str, object]], pool: int, slot: int) -> str:
    return str(next(
        row["joint_fingerprint"]
        for row in candidates
        if int(row["pool_id"]) == pool and int(row["candidate_index"]) == slot
    ))


def _selections(
    candidates: list[dict[str, object]] | None = None,
    *, m1_disagreements: int = 6,
    m2_disagreements: int = 8,
) -> list[dict[str, object]]:
    candidates = candidates if candidates is not None else _candidates()
    random_indices = gate.frozen_random_indices(
        pool_count=16,
        choices_per_pool=4,
        seed=202608281602,
    )
    rows: list[dict[str, object]] = []
    for pool in range(16):
        slots = {
            "M1": 1 if pool < m1_disagreements else 0,
            "M2": 2 if pool < m2_disagreements else 0,
            "B1": 2,
            "B3": 0,
            "B4": 3,
            "R": int(random_indices[pool]),
        }
        row: dict[str, object] = {"pool_id": pool, "random_index": int(random_indices[pool])}
        for method, slot in slots.items():
            row[f"{method}_selected_joint_fingerprint"] = _joint_for(candidates, pool, slot)
        row["B_STAR"] = "B3"
        row["B_STAR_selected_joint_fingerprint"] = row["B3_selected_joint_fingerprint"]
        rows.append(row)
    return rows


def _scorer_identity() -> dict[str, object]:
    return {
        "status": "IDENTICAL",
        "outer_folds": 5,
        "metric_seeds": [0, 1, 2],
        "metric_best_epochs": list(range(11, 26)),
        "distance_gap_spearman": [0.10 + 0.01 * index for index in range(15)],
        "gap_at_5": 0.06259007936507938,
        "top5_historical_neighbors": {
            "query": [_fingerprint("historical-neighbor", index) for index in range(5)],
        },
        "M1_oof_predictions": {"query": 0.71},
        "M2_oof_predictions": {"query": 0.72},
        "prospective_scorer_identity": "IDENTICAL",
        "previous_pilot_labels_used": False,
        "test_data_accessed": False,
    }


def _baseline_selection() -> dict[str, object]:
    return {
        "B_STAR": "B3",
        "source": "historical_soft_metric_v2_strict_oof_only",
        "prospective_labels_used": False,
    }


def _previous_pilot_exclusion_audit() -> dict[str, object]:
    return {
        "status": "PASSED",
        "previous_candidate_count": 32,
        "current_candidate_count": 64,
        "architecture_overlap_count": 0,
        "joint_overlap_count": 0,
        "previous_labels_used_for_training": False,
    }


def _freeze(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    pre_eval = tmp_path / "pre_eval"
    audit = gate.freeze_pre_eval(
        pre_eval,
        candidate_manifest=_candidates(),
        previous_pilot_exclusion_audit=_previous_pilot_exclusion_audit(),
        predictions=_prediction_rows(),
        selections=_selections(),
        baseline_selection=_baseline_selection(),
        scorer_identity=_scorer_identity(),
        historical_input_hashes={"soft_archive": "a" * 64, "pilot_archive": "b" * 64},
        generator_identity={
            "source_sha256": "c" * 64,
            "master_seed": 202608281601,
            "random_seed": 202608281602,
        },
    )
    return pre_eval, audit


def _evaluation_rows(
    candidates: list[dict[str, object]] | None = None,
    *, omit: tuple[str, int] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates if candidates is not None else _candidates():
        joint = str(candidate["joint_fingerprint"])
        pool = int(candidate["pool_id"])
        slot = int(candidate["candidate_index"])
        for seed in (0, 1, 2):
            if omit == (joint, seed):
                continue
            rows.append({
                "pool_id": pool,
                "joint_fingerprint": joint,
                "architecture_fingerprint": str(candidate["architecture_fingerprint"]),
                "training_seed": seed,
                "validation_accuracy": 0.70 + 0.01 * slot + 0.001 * seed,
                "valid": True,
                "terminal_status": "success",
                "track_test": False,
            })
    return rows


def _method_metrics(*, direction: str = "gp") -> dict[str, object]:
    methods: dict[str, dict[str, object]] = {
        method: {
            "selected_validation_mean": 0.80,
            "mean_simple_regret": 0.010,
            "median_simple_regret": 0.010,
            "true_best_hits": 2,
            "true_best_hit_rate": 0.125,
            "within_0_006_of_oracle_rate": 0.25,
        }
        for method in METHODS
    }
    if direction == "gp":
        methods["M2"].update({"mean_simple_regret": 0.008, "true_best_hits": 3})
    elif direction == "knn":
        methods["M1"].update({"mean_simple_regret": 0.008, "true_best_hits": 3})
    return {"methods": methods, "statistical_unit": "pool", "pool_count": 16}


def _paired_rows(
    challenger: str,
    comparator: str,
    *,
    direction: str = "challenger",
    disagreements: int = 8,
    challenger_wins: int = 6,
    winning_gain: float = 0.010,
    losing_gap: float = 0.001,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pool in range(16):
        comparator_value = 0.800
        if pool >= disagreements:
            difference = 0.0
            challenger_fp = comparator_fp = f"same-{pool}"
        else:
            challenger_fp = f"{challenger}-{pool}"
            comparator_fp = f"{comparator}-{pool}"
            challenger_is_winner = pool < challenger_wins
            signed = winning_gain if challenger_is_winner else -losing_gap
            difference = signed if direction == "challenger" else -signed
        challenger_value = comparator_value + difference
        row: dict[str, object] = {
            "pool_id": pool,
            "observed_pool_oracle": 0.820,
            f"{challenger}_selected_joint_fingerprint": challenger_fp,
            f"{comparator}_selected_joint_fingerprint": comparator_fp,
            f"{challenger}_selected_validation_mean": challenger_value,
            f"{comparator}_selected_validation_mean": comparator_value,
            f"{challenger}_simple_regret": 0.820 - challenger_value,
            f"{comparator}_simple_regret": 0.820 - comparator_value,
            f"{challenger}_true_best_hit": int(math.isclose(challenger_value, 0.820)),
            f"{comparator}_true_best_hit": 0,
        }
        rows.append(row)
    return rows


def _confirmation_rows(
    method: str = "M1",
    *,
    disagreements: int = 8,
    wins: int = 6,
    winning_gain: float = 0.010,
    losing_gap: float = 0.001,
) -> list[dict[str, object]]:
    return _paired_rows(
        method,
        "B3",
        disagreements=disagreements,
        challenger_wins=wins,
        winning_gain=winning_gain,
        losing_gap=losing_gap,
    )


def _confirmation_metrics(method: str = "M1") -> dict[str, object]:
    metrics = _method_metrics()
    methods = metrics["methods"]
    assert isinstance(methods, dict)
    methods["B3"].update({"mean_simple_regret": 0.010, "true_best_hits": 2})
    methods[method].update({"mean_simple_regret": 0.007, "true_best_hits": 3})
    return metrics


def _family_confirmation(confirmed: bool = True) -> dict[str, object]:
    return {
        "M1": {"confirmed": confirmed},
        "M2": {"confirmed": False},
        "family_confirmed": confirmed,
    }


def _surrogate_rows(
    *,
    direction: str,
    disagreements: int = 8,
    wins: int = 6,
    winning_gain: float = 0.010,
    losing_gap: float = 0.001,
) -> list[dict[str, object]]:
    challenger, comparator = ("M2", "M1") if direction == "gp" else ("M1", "M2")
    return _paired_rows(
        challenger,
        comparator,
        disagreements=disagreements,
        challenger_wins=wins,
        winning_gain=winning_gain,
        losing_gap=losing_gap,
    )


def _fake_bootstrap(lower: float):
    def run(
        values: list[float],
        *,
        replicates: int = 10_000,
        seed: int = 0,
        unit: str = "pool",
    ) -> dict[str, object]:
        return {
            "bootstrap_unit": unit,
            "pool_count": len(values),
            "replicates": replicates,
            "seed": seed,
            "confidence_interval_95": [lower, max(lower, 0.01)],
        }

    return run


def test_frozen_historical_identity_is_336_192_24_1008() -> None:
    audit = gate.validate_historical_identity({
        "unique_joint_configurations": 336,
        "unique_architectures": 192,
        "unique_pools": 24,
        "seed_record_count": 1008,
        "training_seeds": [0, 1, 2],
    })
    assert audit["passed"] is True


def test_previous_prospective_32_candidates_are_recoverable_without_labels() -> None:
    recovered = gate.recover_previous_pilot_fingerprints({
        "classification": "METRIC_GP_PROSPECTIVE_SIGNAL",
        "candidates": _previous_pilot_candidates(),
    })
    assert recovered["candidate_count"] == 32
    assert recovered["architecture_fingerprints"] == _previous_architectures()
    assert recovered["joint_fingerprints"] == _previous_joints()
    assert "validation_mean" not in recovered


@pytest.mark.parametrize("overlap", ["previous_architecture", "previous_joint", "historical_architecture", "historical_joint"])
def test_current_candidates_reject_all_historical_and_previous_pilot_overlaps(overlap: str) -> None:
    rows = _candidates()
    if overlap == "previous_architecture":
        rows[0]["architecture_fingerprint"] = next(iter(_previous_architectures()))
    elif overlap == "previous_joint":
        rows[0]["joint_fingerprint"] = next(iter(_previous_joints()))
    elif overlap == "historical_architecture":
        rows[0]["architecture_fingerprint"] = next(iter(_historical_architectures()))
    else:
        rows[0]["joint_fingerprint"] = next(iter(_historical_joints()))
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows)


@pytest.mark.parametrize("fingerprint_field", ["architecture_fingerprint", "joint_fingerprint"])
def test_all_64_current_architecture_and_joint_fingerprints_are_internally_unique(
    fingerprint_field: str,
) -> None:
    assert _validate_candidates()["passed"] is True
    rows = _candidates()
    rows[1][fingerprint_field] = rows[0][fingerprint_field]
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows)


def test_candidate_manifest_is_exactly_16_pools_by_four() -> None:
    audit = _validate_candidates()
    assert audit["pool_count"] == 16
    assert audit["candidate_count"] == 64
    rows = _candidates()[:-1]
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows)


def test_candidate_pool_seeds_are_frozen_to_master_plus_pool() -> None:
    assert gate.MASTER_CANDIDATE_SEED == 202608281601
    rows = _candidates()
    rows[9]["generator_seed"] = 202608281999
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows)


@pytest.mark.parametrize(
    "field",
    [
        "predictor_accessed",
        "validation_labels_accessed",
        "historical_top_architectures_used",
        "metric_distance_used",
        "mutation_or_local_search_used",
    ],
)
def test_candidate_generation_is_model_blind(field: str) -> None:
    audit = _generator_audit()
    audit[field] = True
    with pytest.raises(gate.CandidateGenerationError):
        gate.validate_candidate_manifest(
            _candidates(),
            historical_architecture_fingerprints=_historical_architectures(),
            historical_joint_fingerprints=_historical_joints(),
            previous_architecture_fingerprints=_previous_architectures(),
            previous_joint_fingerprints=_previous_joints(),
            generator_audit=audit,
        )


def test_candidate_hp_sampling_reuses_high_regime_prior() -> None:
    rows = _candidates()
    rows[0]["proposal_distribution"] = "predictor_guided"
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows)


def test_soft_metric_scorer_replay_requires_exact_identity() -> None:
    archived = _scorer_identity()
    audit = gate.assert_scorer_replay_identity(copy.deepcopy(archived), archived, atol=1.0e-8)
    assert audit["status"] == "IDENTICAL"
    changed = copy.deepcopy(archived)
    changed["M2_oof_predictions"]["query"] = 0.720001
    with pytest.raises(gate.ScorerReplayIdentityMismatch):
        gate.assert_scorer_replay_identity(changed, archived, atol=1.0e-8)


@pytest.mark.parametrize("method", SCORERS)
def test_previous_prospective_labels_never_enter_any_scorer(method: str) -> None:
    audit = {
        "historical_training_joint_count": 336,
        "historical_training_architecture_count": 192,
        "historical_training_seed_record_count": 1008,
        "training_seeds": [0, 1, 2],
        "previous_pilot_labels_used": {name: False for name in SCORERS},
        "current_confirmation_labels_used": {name: False for name in SCORERS},
    }
    assert gate.assert_no_prospective_label_training(audit)["passed"] is True
    audit["previous_pilot_labels_used"][method] = True
    with pytest.raises(gate.ProtocolError):
        gate.assert_no_prospective_label_training(audit)


def _frozen_scorer_protocol() -> dict[str, object]:
    return {
        "M1": {
            "k": 5,
            "distance": "Euclidean",
            "weight": "inverse_distance_plus_1e-8",
            "protocol_reused_without_tuning": True,
        },
        "M2": {
            "input_dim": 8,
            "input_l2_normalized": True,
            "mean": "ConstantMean",
            "kernel": "Matern-5/2",
            "ard": False,
            "lengthscale_count": 1,
            "input_standardized": False,
            "historical_observation_se": True,
            "learned_noise_floor": True,
            "fixed_observation_noise_rule": "validation_se_squared",
            "fitted_noise_floor_rule": "validation_se_squared_plus_floor_squared",
            "metric_seeds": [0, 1, 2],
            "fold_count": 5,
            "previous_pilot_labels_used": False,
            "current_confirmation_labels_used": False,
            "candidate_labels_used": False,
            "posterior_variance_used": False,
            "protocol_reused_without_tuning": True,
        },
        "B1": {
            "features": "raw_transformed_hp",
            "hp_features": ["log10_lr", "dropout", "log2_hidden_dim", "log10_l2"],
            "kernel": "Matern-5/2",
            "ard": True,
            "standardization": "outer_train_only",
            "input_standardized": True,
            "posterior_variance_used": False,
            "previous_pilot_labels_used": False,
            "current_confirmation_labels_used": False,
            "protocol_reused_without_tuning": True,
        },
        "B3": {
            "features": "explicit_canonical_graph_plus_raw_transformed_hp",
            "kernel": "Matern-5/2",
            "ard": True,
            "standardization": "outer_train_only",
            "input_standardized": True,
            "posterior_variance_used": False,
            "previous_pilot_labels_used": False,
            "current_confirmation_labels_used": False,
            "protocol_reused_without_tuning": True,
        },
        "B4": {
            "architecture_branch": "frozen_soft_metric_v2",
            "hp_branch": "frozen_soft_metric_v2",
            "interaction_fusion": "frozen_soft_metric_v2",
            "loss": "Huber",
            "model_seeds": [0, 1, 2],
            "early_stopping": "frozen_soft_metric_v2",
            "outer_split": "frozen_five_fold",
            "prediction_aggregation": "arithmetic_mean_model_seeds",
            "final_refit_used_complete_outer_train": True,
            "previous_pilot_labels_used": False,
            "current_confirmation_labels_used": False,
            "protocol_reused_without_tuning": True,
        },
    }


@pytest.mark.parametrize(
    ("method", "audit_key"),
    [("B1", "B1_frozen_protocol"), ("B3", "B3_frozen_protocol"), ("B4", "B4_frozen_protocol")],
)
def test_non_metric_baselines_reuse_their_frozen_protocols(method: str, audit_key: str) -> None:
    protocol = {
        name: copy.deepcopy(_frozen_scorer_protocol()[name])
        for name in ("B1", "B3", "B4")
    }
    audit = gate.validate_nonmetric_baseline_protocol(protocol)
    assert audit[audit_key] is True
    protocol[method]["protocol_reused_without_tuning"] = False
    with pytest.raises(gate.ProtocolError):
        gate.validate_nonmetric_baseline_protocol(protocol)


def _historical_baseline_metrics() -> dict[str, dict[str, float]]:
    return {
        "B1": {"mean_best_of_4_simple_regret": 0.0100000000000, "mae": 0.020},
        "B3": {"mean_best_of_4_simple_regret": 0.0100000000005, "mae": 0.019},
        "B4": {"mean_best_of_4_simple_regret": 0.0100000000005, "mae": 0.019},
    }


def test_b_star_uses_historical_oof_regret_then_mae_then_lexicographic_ties() -> None:
    selected = gate.select_historical_baseline(_historical_baseline_metrics())
    assert selected["B_STAR"] == "B3"
    assert selected["selection_source"] == "historical_oof_only"
    assert selected["regret_tie_tolerance"] == 1.0e-12


def test_prospective_labels_cannot_change_historical_b_star() -> None:
    historical = _historical_baseline_metrics()
    first = gate.select_historical_baseline(
        historical,
        prospective_labels=[{"B1": 1.0, "B3": 0.0, "B4": 0.0}],
    )
    second = gate.select_historical_baseline(
        historical,
        prospective_labels=[{"B1": 0.0, "B3": 1.0, "B4": 1.0}],
    )
    assert first["B_STAR"] == second["B_STAR"] == "B3"
    assert first["prospective_labels_used"] is second["prospective_labels_used"] is False


def test_all_16_pools_are_frozen_together(tmp_path: Path) -> None:
    pre_eval, audit = _freeze(tmp_path)
    assert audit["passed"] is True
    assert audit["pool_count"] == 16
    required = {
        "candidate_manifest.json",
        "previous_pilot_exclusion_audit.json",
        "predictions.csv",
        "selections.json",
        "baseline_selection.json",
        "scorer_identity.json",
        "freeze_manifest.json",
        "pre_eval_root_sha256.txt",
        "PRE_EVAL_SELECTION_FROZEN",
    }
    assert required <= {path.name for path in pre_eval.iterdir()}
    assert gate.verify_pre_eval_freeze(pre_eval)["passed"] is True

    with pytest.raises(gate.PreEvalFreezeError):
        gate.freeze_pre_eval(
            tmp_path / "incomplete_pre_eval",
            candidate_manifest=_candidates(),
            previous_pilot_exclusion_audit=_previous_pilot_exclusion_audit(),
            predictions=_prediction_rows(),
            selections=_selections()[:-1],
            baseline_selection=_baseline_selection(),
            scorer_identity=_scorer_identity(),
        )


def test_pre_eval_freeze_must_precede_first_gnn_task() -> None:
    with pytest.raises(gate.PreEvalFreezeError):
        gate.build_evaluation_tasks(
            _candidates(),
            pre_eval_verified=False,
            information_gate_passed=True,
            training_seeds=[0, 1, 2],
            track_test=False,
        )


def test_information_gate_uses_max_m1_or_m2_disagreement_against_b_star() -> None:
    selections = _selections(m1_disagreements=6, m2_disagreements=8)
    counts = gate.disagreement_counts(selections, b_star="B3")
    assert counts == {"d1": 6, "d2": 8, "d12": 8}
    decision = gate.information_gate(selections, b_star="B3", minimum_disagreement_pools=6)
    assert decision["authorized"] is True
    assert decision["max_disagreement"] == 8
    assert decision["total_gnn_runs"] == 192


def test_low_information_authorizes_exactly_zero_gnn_runs() -> None:
    selections = _selections(m1_disagreements=5, m2_disagreements=4)
    decision = gate.information_gate(selections, b_star="B3", minimum_disagreement_pools=6)
    assert decision["classification"] == "CONFIRMATION_LOW_INFORMATION"
    assert decision["authorized"] is False
    assert decision["total_gnn_runs"] == 0
    assert gate.build_evaluation_tasks(
        _candidates(),
        pre_eval_verified=True,
        information_gate_passed=False,
        training_seeds=[0, 1, 2],
        track_test=False,
    ) == []


def test_training_seeds_are_exactly_zero_one_two_and_matched_within_pool() -> None:
    tasks = gate.build_evaluation_tasks(
        _candidates(),
        pre_eval_verified=True,
        information_gate_passed=True,
        training_seeds=[0, 1, 2],
        track_test=False,
    )
    assert sorted({int(row["training_seed"]) for row in tasks}) == [0, 1, 2]
    for pool in range(16):
        pool_rows = [row for row in tasks if int(row["pool_id"]) == pool]
        by_joint: dict[str, set[int]] = {}
        for row in pool_rows:
            by_joint.setdefault(str(row["joint_fingerprint"]), set()).add(int(row["training_seed"]))
        assert set(map(frozenset, by_joint.values())) == {frozenset({0, 1, 2})}


def test_test_loader_firewall_and_track_test_false_are_enforced() -> None:
    with pytest.raises(gate.ProtocolError):
        gate.reject_test_fields({"validation_accuracy": 0.8, "test_accuracy": 0.9})
    assert gate.FROZEN_TRAINING_PROTOCOL["track_test"] is False
    with pytest.raises(gate.ProtocolError):
        gate.build_evaluation_tasks(
            _candidates(),
            pre_eval_verified=True,
            information_gate_passed=True,
            training_seeds=[0, 1, 2],
            track_test=True,
        )


def test_complete_task_manifest_has_exactly_192_unique_tasks() -> None:
    tasks = gate.build_evaluation_tasks(
        _candidates(),
        pre_eval_verified=True,
        information_gate_passed=True,
        training_seeds=[0, 1, 2],
        track_test=False,
    )
    assert len(tasks) == 192
    assert len({str(row["task_key"]) for row in tasks}) == 192


def test_incomplete_pool_fails_closed() -> None:
    candidates = _candidates()
    missing = (str(candidates[0]["joint_fingerprint"]), 2)
    with pytest.raises(gate.EvaluationIncompleteError):
        gate.aggregate_evaluation_results(candidates, _evaluation_rows(candidates, omit=missing))


def test_failed_candidate_cannot_be_replaced_after_freeze() -> None:
    before = _candidates()
    after = copy.deepcopy(before)
    after[0]["joint_fingerprint"] = _fingerprint("replacement", 0)
    with pytest.raises(gate.PreEvalFreezeError):
        gate.assert_candidate_manifest_immutable(before, after)


@pytest.mark.parametrize("method", SCORERS)
def test_no_scorer_can_online_update_from_confirmation_labels(method: str) -> None:
    audit = {
        "candidate_labels_used_for_update": {name: False for name in SCORERS},
        "pool_sequential_updates": False,
        "all_predictions_frozen_before_first_evaluation": True,
        "candidate_replacement": False,
    }
    assert gate.assert_no_online_update(audit)["passed"] is True
    audit["candidate_labels_used_for_update"][method] = True
    with pytest.raises(gate.ProtocolError):
        gate.assert_no_online_update(audit)


def test_observed_three_seed_pool_oracle_simple_regret_and_hits_are_correct() -> None:
    statistics = gate.aggregate_evaluation_results(_candidates(), _evaluation_rows())
    pool_zero = [row for row in statistics if int(row["pool_id"]) == 0]
    expected_oracle = max(float(row["validation_mean"]) for row in pool_zero)
    assert all(math.isclose(float(row["observed_pool_oracle"]), expected_oracle) for row in pool_zero)

    metrics = gate.compute_method_metrics(statistics, _selections())
    assert set(metrics["methods"]) == set(METHODS)
    assert metrics["statistical_unit"] == "pool"
    assert metrics["pool_count"] == 16
    for row in metrics["per_pool"]:
        for method in METHODS:
            expected = float(row["observed_pool_oracle"]) - float(
                row[f"{method}_selected_validation_mean"]
            )
            assert math.isclose(float(row[f"{method}_simple_regret"]), expected, abs_tol=1.0e-15)
            assert row[f"{method}_true_best_hit"] == int(abs(expected) <= 1.0e-12)


def test_pool_bootstrap_has_frozen_unit_replicates_and_seed() -> None:
    result = gate.pool_bootstrap(
        [0.001 * index for index in range(16)],
        replicates=10_000,
        seed=0,
        unit="pool",
    )
    assert result["bootstrap_unit"] == "pool"
    assert result["pool_count"] == 16
    assert result["replicates"] == 10_000
    assert result["seed"] == 0
    with pytest.raises(gate.ProtocolError):
        gate.pool_bootstrap([0.1] * 16, replicates=10_000, seed=0, unit="training_seed")
    with pytest.raises(gate.ProtocolError):
        gate.pool_bootstrap([0.1] * 16, replicates=10_000, seed=1, unit="pool")


def test_metric_confirmation_base_case_passes_all_c1_through_c6() -> None:
    result = gate.evaluate_metric_confirmation(
        "M1", _confirmation_rows(), _confirmation_metrics(), "B3"
    )
    assert result["confirmed"] is True
    assert all(result[name] is True for name in ("C1", "C2", "C3", "C4", "C5", "C6"))


@pytest.mark.parametrize("condition", ["C1", "C2", "C3", "C4", "C5"])
def test_each_non_bootstrap_metric_confirmation_condition_is_individually_required(
    condition: str,
) -> None:
    rows = _confirmation_rows()
    metrics = _confirmation_metrics()
    if condition == "C1":
        rows = _confirmation_rows(disagreements=5, wins=4, winning_gain=0.030)
    elif condition == "C2":
        rows = _confirmation_rows(disagreements=8, wins=5, winning_gain=0.020)
    elif condition == "C3":
        rows = _confirmation_rows(disagreements=8, wins=6, winning_gain=0.005)
    elif condition == "C4":
        metrics["methods"]["M1"]["mean_simple_regret"] = 0.008000000001
    else:
        metrics["methods"]["M1"]["true_best_hits"] = 1
    result = gate.evaluate_metric_confirmation("M1", rows, metrics, "B3")
    assert result[condition] is False
    assert result["confirmed"] is False


def test_c4_uses_near_zero_comparator_fallback() -> None:
    metrics = _confirmation_metrics()
    metrics["methods"]["B3"]["mean_simple_regret"] = 0.5e-6
    metrics["methods"]["M1"]["mean_simple_regret"] = 0.5e-6
    assert gate.evaluate_metric_confirmation("M1", _confirmation_rows(), metrics, "B3")["C4"] is True
    metrics["methods"]["M1"]["mean_simple_regret"] = 0.5000001e-6
    assert gate.evaluate_metric_confirmation("M1", _confirmation_rows(), metrics, "B3")["C4"] is False


def test_c6_requires_strictly_positive_bootstrap_lower_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "pool_bootstrap", _fake_bootstrap(0.0))
    at_zero = gate.evaluate_metric_confirmation("M1", _confirmation_rows(), _confirmation_metrics(), "B3")
    assert at_zero["C6"] is False
    assert at_zero["confirmed"] is False

    monkeypatch.setattr(gate, "pool_bootstrap", _fake_bootstrap(1.0e-15))
    above_zero = gate.evaluate_metric_confirmation("M1", _confirmation_rows(), _confirmation_metrics(), "B3")
    assert above_zero["C6"] is True
    assert above_zero["confirmed"] is True


@pytest.mark.parametrize(("m1", "m2", "expected"), [(True, False, True), (False, True, True), (True, True, True), (False, False, False)])
def test_either_metric_method_confirmation_confirms_the_family(
    m1: bool,
    m2: bool,
    expected: bool,
) -> None:
    family = gate.combine_metric_family_confirmation(
        {"method": "M1", "confirmed": m1},
        {"method": "M2", "confirmed": m2},
    )
    assert family["family_confirmed"] is expected


def test_all_gp_increment_conditions_select_m2() -> None:
    decision = gate.select_final_surrogate(
        _surrogate_rows(direction="gp"),
        _method_metrics(direction="gp"),
        _family_confirmation(),
        "B3",
    )
    assert all(decision["gp_conditions"][name] for name in (
        "S_GP_1", "S_GP_2", "S_GP_3", "S_GP_4", "S_GP_5"
    ))
    assert decision["classification"] == "METRIC_FAMILY_CONFIRMED_GP_SELECTED"
    assert decision["final_surrogate"] == "M2"
    assert decision["selection_basis"] == "CLEAR_GP_INCREMENT"


@pytest.mark.parametrize("condition", ["S_GP_1", "S_GP_2", "S_GP_3", "S_GP_4"])
def test_each_non_bootstrap_gp_increment_condition_is_required(condition: str) -> None:
    rows = _surrogate_rows(direction="gp")
    metrics = _method_metrics(direction="gp")
    if condition == "S_GP_1":
        rows = _surrogate_rows(direction="gp", disagreements=5, wins=4, winning_gain=0.030)
    elif condition == "S_GP_2":
        rows = _surrogate_rows(direction="gp", disagreements=8, wins=5, winning_gain=0.020)
    elif condition == "S_GP_3":
        metrics["methods"]["M2"]["mean_simple_regret"] = 0.009000000001
    else:
        metrics["methods"]["M2"]["true_best_hits"] = 1
    decision = gate.select_final_surrogate(rows, metrics, _family_confirmation(), "B3")
    assert decision["gp_conditions"][condition] is False
    assert decision["final_surrogate"] == "M1"
    assert decision["selection_basis"] == "PARSIMONY_NO_CLEAR_GP_INCREMENT"


def test_all_symmetric_knn_advantage_conditions_select_m1_clearly() -> None:
    decision = gate.select_final_surrogate(
        _surrogate_rows(direction="knn"),
        _method_metrics(direction="knn"),
        _family_confirmation(),
        "B3",
    )
    assert all(decision["knn_conditions"][name] for name in (
        "S_KNN_1", "S_KNN_2", "S_KNN_3", "S_KNN_4", "S_KNN_5"
    ))
    assert decision["classification"] == "METRIC_FAMILY_CONFIRMED_KNN_SELECTED"
    assert decision["final_surrogate"] == "M1"
    assert decision["selection_basis"] == "CLEAR_KNN_ADVANTAGE"


@pytest.mark.parametrize("condition", ["S_KNN_1", "S_KNN_2", "S_KNN_3", "S_KNN_4"])
def test_each_non_bootstrap_symmetric_knn_condition_is_required(condition: str) -> None:
    rows = _surrogate_rows(direction="knn")
    metrics = _method_metrics(direction="knn")
    if condition == "S_KNN_1":
        rows = _surrogate_rows(direction="knn", disagreements=5, wins=4, winning_gain=0.030)
    elif condition == "S_KNN_2":
        rows = _surrogate_rows(direction="knn", disagreements=8, wins=5, winning_gain=0.020)
    elif condition == "S_KNN_3":
        metrics["methods"]["M1"]["mean_simple_regret"] = 0.009000000001
    else:
        metrics["methods"]["M1"]["true_best_hits"] = 1
    decision = gate.select_final_surrogate(rows, metrics, _family_confirmation(), "B3")
    assert decision["knn_conditions"][condition] is False
    assert decision["final_surrogate"] == "M1"
    assert decision["selection_basis"] == "PARSIMONY_NO_CLEAR_GP_INCREMENT"


@pytest.mark.parametrize(
    ("direction", "condition", "positive_basis"),
    [("gp", "S_GP_5", "CLEAR_GP_INCREMENT"), ("knn", "S_KNN_5", "CLEAR_KNN_ADVANTAGE")],
)
def test_surrogate_s5_bootstrap_lower_bound_is_strict(
    direction: str,
    condition: str,
    positive_basis: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _surrogate_rows(direction=direction)
    metrics = _method_metrics(direction=direction)
    condition_group = "gp_conditions" if direction == "gp" else "knn_conditions"

    monkeypatch.setattr(gate, "pool_bootstrap", _fake_bootstrap(0.0))
    at_zero = gate.select_final_surrogate(rows, metrics, _family_confirmation(), "B3")
    assert at_zero[condition_group][condition] is False
    assert at_zero["selection_basis"] == "PARSIMONY_NO_CLEAR_GP_INCREMENT"

    monkeypatch.setattr(gate, "pool_bootstrap", _fake_bootstrap(1.0e-15))
    positive = gate.select_final_surrogate(rows, metrics, _family_confirmation(), "B3")
    assert positive[condition_group][condition] is True
    assert positive["selection_basis"] == positive_basis


def test_no_clear_gp_increment_defaults_to_m1_by_parsimony() -> None:
    decision = gate.select_final_surrogate(
        _surrogate_rows(direction="gp", disagreements=5, wins=4),
        _method_metrics(direction="gp"),
        _family_confirmation(),
        "B3",
    )
    assert decision["classification"] == "METRIC_FAMILY_CONFIRMED_KNN_SELECTED"
    assert decision["final_surrogate"] == "M1"
    assert decision["selection_basis"] == "PARSIMONY_NO_CLEAR_GP_INCREMENT"
    assert decision["m1_statistically_proven_superior"] is False


def test_previous_pilot_and_random_are_invariant_to_both_primary_gates() -> None:
    rows = _surrogate_rows(direction="gp")
    metrics = _method_metrics(direction="gp")
    first = gate.select_final_surrogate(
        rows,
        metrics,
        _family_confirmation(),
        "B3",
        pilot_context={"pilot_direction": "M1", "attempt_to_merge_into_gate": True},
        random_metrics={"mean_simple_regret": -999.0, "true_best_hits": 16},
    )
    second = gate.select_final_surrogate(
        rows,
        metrics,
        _family_confirmation(),
        "B3",
        pilot_context={"pilot_direction": "M2", "attempt_to_merge_into_gate": False},
        random_metrics={"mean_simple_regret": 999.0, "true_best_hits": 0},
    )
    keys = ("classification", "final_surrogate", "selection_basis")
    assert tuple(first[key] for key in keys) == tuple(second[key] for key in keys)
    assert first["pilot_used_in_gate"] is second["pilot_used_in_gate"] is False
    assert first["random_used_in_primary_gate"] is second["random_used_in_primary_gate"] is False


def test_unconfirmed_family_stops_before_surrogate_selection() -> None:
    decision = gate.select_final_surrogate(
        _surrogate_rows(direction="gp"),
        _method_metrics(direction="gp"),
        _family_confirmation(False),
        "B3",
    )
    assert decision["classification"] == "METRIC_FAMILY_NOT_CONFIRMED"
    assert decision["final_surrogate"] is None


def test_final_classification_set_is_exact_and_exhaustive() -> None:
    assert gate.ALLOWED_CLASSIFICATIONS == {
        "CONFIRMATION_INPUT_INVALID",
        "SCORER_REPLAY_IDENTITY_MISMATCH",
        "CONFIRMATION_CANDIDATE_GENERATION_FAILED",
        "CONFIRMATION_LOW_INFORMATION",
        "CONFIRMATION_EVALUATION_INCOMPLETE",
        "METRIC_FAMILY_NOT_CONFIRMED",
        "METRIC_FAMILY_CONFIRMED_KNN_SELECTED",
        "METRIC_FAMILY_CONFIRMED_GP_SELECTED",
        "RUN_FAILED",
    }


def test_source_snapshot_requires_new_and_frozen_runtime_sources(tmp_path: Path) -> None:
    required = set(gate.REQUIRED_SOURCE_SNAPSHOT_PATHS)
    assert {
        "configs/joint_metric_independent_confirmation_surrogate_selection_v1.yaml",
        "analyse/joint_metric_independent_confirmation_surrogate_selection.py",
        "scripts/run_joint_metric_independent_confirmation_surrogate_selection.sh",
        "tests/test_joint_metric_independent_confirmation_surrogate_selection.py",
        "configs/joint_configuration_soft_metric_gate_v2.yaml",
        "analyse/joint_configuration_soft_metric_gate.py",
        "analyse/joint_configuration_metric_gate.py",
        "analyse/performance_metric_quick_gate.py",
        "analyse/high_regime_prior_model_blind_validation.py",
        "analyse/run_high_regime_prior_model_blind_validation_labels.py",
        "eval_utils.py",
        "architecture_diagnostics.py",
    } <= required
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frozen:{relative}".encode("utf-8"))
    (tmp_path / "imported_local_modules.json").write_text("[]\n", encoding="utf-8")
    entries = [
        {
            "repository_relative_path": relative,
            "snapshot_relative_path": relative,
            "size_bytes": (tmp_path / relative).stat().st_size,
            "sha256": hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest(),
        }
        for relative in sorted(required)
    ]
    manifest = {
        "entries": entries,
        "manifest_root_sha256": gate.stable_hash([
            {
                "repository_relative_path": entry["repository_relative_path"],
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
            }
            for entry in entries
        ]),
    }
    (tmp_path / "sha256_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    audit = gate.validate_source_snapshot(tmp_path, sorted(required))
    assert audit["passed"] is True
    assert audit["all_required_sources_copied_as_bytes"] is True


def test_final_report_requires_all_26_preregistered_sections() -> None:
    expected = (
        "1. Executive conclusion",
        "2. Historical evidence boundary",
        "3. Independence from previous 8-pool pilot",
        "4. Historical scorer replay",
        "5. Historical baseline replay",
        "6. Frozen B_STAR selection",
        "7. Candidate generation",
        "8. Candidate novelty audit",
        "9. Pre-evaluation freeze",
        "10. Predictor scores",
        "11. Method disagreement counts",
        "12. GNN evaluation completeness",
        "13. Per-pool actual validation table",
        "14. All-method selection metrics",
        "15. M1 vs B_STAR",
        "16. M2 vs B_STAR",
        "17. Metric-family confirmation",
        "18. M1 vs M2 comparison",
        "19. Final surrogate selection",
        "20. Random baseline",
        "21. Pool bootstrap/sign summary",
        "22. Frozen classification",
        "23. Interpretation",
        "24. What this does NOT prove",
        "25. Next authorized experiment",
        "26. Engineering audit",
    )
    assert tuple(gate.REPORT_SECTIONS) == expected
    report = "\n\n".join(f"## {heading}\n\nSynthetic evidence." for heading in expected)
    gate.assert_complete_final_report(report)
    with pytest.raises(gate.ProtocolError):
        gate.assert_complete_final_report(report.replace(expected[17], "18. omitted", 1))


def test_dry_run_does_not_create_formal_result_directory(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    result = gate.execute_dry_run(gate.DEFAULT_CONFIG, output)
    assert result == 0
    assert not output.exists()


def test_nonempty_formal_result_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    (output / "existing.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(gate.ProtocolError):
        gate.ensure_output_policy(output)


def test_failure_bundle_is_packaged_with_sha256(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    archive = tmp_path / "formal.tar.gz"
    gate.write_failure_bundle(
        output,
        gate.ProtocolError("synthetic failure"),
        archive_path=archive,
    )
    assert (output / "FAILED").is_file()
    assert (output / "error.json").is_file()
    assert json.loads((output / "engineering_fix.json").read_text(encoding="utf-8")) == gate.ENGINEERING_FIX
    assert archive.is_file()
    assert Path(str(archive) + ".sha256").is_file()
    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    assert any(name.endswith("FAILED") for name in names)
    assert any(name.endswith("error.json") for name in names)
    assert any(name.endswith("engineering_fix.json") for name in names)


def test_engineering_fix_uses_rerun1_without_changing_scientific_protocol() -> None:
    config = gate.load_config(gate.DEFAULT_CONFIG)
    assert gate.DEFAULT_OUTPUT.name.endswith("-rerun1")
    assert config["output"]["result_root"].endswith("-rerun1")
    assert config["engineering_fix"] == gate.ENGINEERING_FIX
    assert gate.ENGINEERING_FIX["scientific_protocol_changed"] is False


def test_synthetic_smoke_exercises_every_non_failure_terminal_route() -> None:
    result = gate.synthetic_smoke()
    assert result["passed"] is True
    for classification in gate.ALLOWED_CLASSIFICATIONS - {"RUN_FAILED"}:
        assert result[classification] is True


def _regression_metric_protocol() -> dict[str, object]:
    protocol = _frozen_scorer_protocol()
    return {method: copy.deepcopy(protocol[method]) for method in ("M1", "M2")}


def _regression_nonmetric_protocol() -> dict[str, object]:
    protocol = _frozen_scorer_protocol()
    return {method: copy.deepcopy(protocol[method]) for method in ("B1", "B3", "B4")}


def test_regression_01_baseline_validator_accepts_b1_b3_b4_without_m2() -> None:
    audit = gate.validate_nonmetric_baseline_protocol(_regression_nonmetric_protocol())
    assert audit["passed"] is True


def test_regression_02_baseline_validator_rejects_missing_b3() -> None:
    protocol = _regression_nonmetric_protocol()
    del protocol["B3"]
    with pytest.raises(gate.ProtocolError):
        gate.validate_nonmetric_baseline_protocol(protocol)


def test_regression_03_metric_validator_accepts_complete_m1_m2_audit() -> None:
    audit = gate.validate_metric_scorer_protocol(_regression_metric_protocol())
    assert audit["passed"] is True


def test_regression_04_metric_validator_rejects_missing_m2() -> None:
    protocol = _regression_metric_protocol()
    del protocol["M2"]
    with pytest.raises(gate.ProtocolError):
        gate.validate_metric_scorer_protocol(protocol)


def test_regression_05_metric_validator_rejects_m2_ard() -> None:
    protocol = _regression_metric_protocol()
    protocol["M2"]["ard"] = True
    with pytest.raises(gate.ProtocolError):
        gate.validate_metric_scorer_protocol(protocol)


def test_regression_06_metric_validator_rejects_m2_input_standardization() -> None:
    protocol = _regression_metric_protocol()
    protocol["M2"]["input_standardized"] = True
    with pytest.raises(gate.ProtocolError):
        gate.validate_metric_scorer_protocol(protocol)


def test_regression_07_m2_historical_replay_numeric_mismatch_fails_closed() -> None:
    archived = {
        "M1_oof_predictions": {"historical-query": 0.71},
        "M2_oof_predictions": {"historical-query": 0.72},
    }
    replayed = copy.deepcopy(archived)
    replayed["M2_oof_predictions"]["historical-query"] += 1.0e-6
    with pytest.raises(gate.ScorerReplayIdentityMismatch):
        gate.assert_scorer_replay_identity(replayed, archived, atol=1.0e-8)


def test_regression_08_formal_static_call_sites_use_the_matching_validators() -> None:
    formal_source = inspect.getsource(gate.execute_formal)
    metric_replay = formal_source.index("replay_and_retain_scorers")
    metric_validation = formal_source.index("validate_metric_scorer_protocol")
    baseline_replay = formal_source.index("replay_historical_baselines")
    assert metric_replay < metric_validation < baseline_replay

    baseline_source = inspect.getsource(gate.replay_historical_baselines)
    baseline_validation = baseline_source.index("validate_nonmetric_baseline_protocol")
    baseline_return = baseline_source.rindex("return audit, selection")
    assert baseline_validation < baseline_return
    assert "validate_metric_scorer_protocol" not in baseline_source


def test_regression_08_formal_metric_stage_calls_metric_validator_before_baseline_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopAtBaselineReplay(RuntimeError):
        pass

    events: list[str] = []

    monkeypatch.setattr(gate, "ensure_output_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(gate, "sha256_file", lambda path: "config-sha")
    monkeypatch.setattr(gate, "aggregate_source_sha256", lambda *args, **kwargs: "source-sha")
    monkeypatch.setattr(gate, "load_config", lambda path: {})
    monkeypatch.setattr(gate, "validate_frozen_protocol", lambda config: {"passed": True})
    monkeypatch.setattr(gate, "write_environment_and_git", lambda *args, **kwargs: None)
    monkeypatch.setattr(gate, "_logger", lambda path: (lambda message: None))
    monkeypatch.setattr(gate, "validate_authoritative_inputs", lambda config: {
        "previous_pilot_identity": {},
        "previous_pilot_exclusion": {
            "candidate_count": 32,
            "architecture_fingerprints": set(),
            "joint_fingerprints": set(),
        },
    })
    monkeypatch.setattr(gate.pilot_gate, "load_frozen_soft_runtime", lambda: {"soft": object()})
    monkeypatch.setattr(
        gate.pilot_gate,
        "replay_and_retain_scorers",
        lambda *args, **kwargs: ([], [object()], {"status": "IDENTICAL"}),
    )

    def metric_validator(audit: object) -> dict[str, object]:
        events.append("metric_validator")
        return {"passed": True}

    def wrong_baseline_validator(audit: object) -> dict[str, object]:
        events.append("baseline_validator_too_early")
        return {"passed": True}

    def stop_at_baseline(*args: object, **kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        events.append("baseline_replay")
        raise StopAtBaselineReplay

    monkeypatch.setattr(gate, "validate_metric_scorer_protocol", metric_validator)
    monkeypatch.setattr(gate, "validate_nonmetric_baseline_protocol", wrong_baseline_validator)
    monkeypatch.setattr(gate, "replay_historical_baselines", stop_at_baseline)

    with pytest.raises(StopAtBaselineReplay):
        gate.execute_formal(
            tmp_path / "synthetic-config.yaml",
            tmp_path / "formal",
            "config-sha",
            "source-sha",
            preflight_ready=False,
        )
    assert events == ["metric_validator", "baseline_replay"]


def test_regression_09_dry_run_never_calls_candidate_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_candidate_generation(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("dry-run reached formal candidate generation")

    monkeypatch.setattr(gate, "generate_candidates", forbidden_candidate_generation)
    output = tmp_path / "formal"
    assert gate.execute_dry_run(gate.DEFAULT_CONFIG, output) == 0
    assert calls == 0
    assert not output.exists()


def test_regression_10_dry_run_reports_zero_generated_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "formal"
    assert gate.execute_dry_run(gate.DEFAULT_CONFIG, output) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_generation_started"] is False
    assert payload["generated_candidate_count"] == 0
    assert not output.exists()
