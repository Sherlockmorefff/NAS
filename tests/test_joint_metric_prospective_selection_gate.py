from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tarfile

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "analyse/joint_metric_prospective_selection_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "joint_metric_prospective_selection_gate_for_tests", MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _fingerprint(namespace: str, index: int) -> str:
    return hashlib.sha256(f"{namespace}:{index}".encode("utf-8")).hexdigest()


def _candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pool in range(8):
        for slot in range(4):
            index = pool * 4 + slot
            architecture = _fingerprint("prospective-architecture", index)
            hp = _fingerprint("prospective-hp", index)
            joint = gate.stable_hash({
                "architecture_fingerprint": architecture,
                "hp_fingerprint": hp,
            })
            lr = 10.0 ** (-3.0 + 0.01 * slot)
            dropout = 0.20 + 0.02 * slot
            hidden = (32, 64, 128, 256)[slot]
            l2 = 10.0 ** (-5.0 + 0.01 * pool)
            rows.append({
                "pool_id": pool,
                "candidate_index": slot,
                "generator_seed": 2026082801 + pool,
                "architecture_fingerprint": architecture,
                "hp_fingerprint": hp,
                "joint_fingerprint": joint,
                "joint_config_fingerprint": joint,
                "operations": ["GCNConv"],
                "edges": [[0, 1], [1, 2]],
                "effective_layers": 1,
                "raw_hp_vector": [0.81, 0.82, 0.83, 0.84],
                "raw_hp": {
                    "lr": lr, "dropout": dropout,
                    "hidden_dim": hidden, "l2": l2,
                },
                "lr": lr, "dropout": dropout,
                "hidden_dim": hidden, "l2": l2,
                "hp_regime": "hp_high",
                "proposal_distribution": "HIGH_REGIME_PRIOR",
            })
    return rows


def _generator_audit() -> dict[str, object]:
    return {
        "master_seed": 2026082801,
        "pool_seeds": [2026082801 + pool for pool in range(8)],
        "allowed_rejection_reasons": [
            "illegal_architecture", "fingerprint_duplicate",
            "historical_metric_training_overlap",
        ],
        "predictor_accessed": False,
        "validation_labels_accessed": False,
        "historical_top_architectures_used": False,
        "mutation_or_local_search_used": False,
        "candidate_count": 32,
        "pool_count": 8,
        "candidates_per_pool": 4,
    }


def _validate_candidates(
    rows: list[dict[str, object]] | None = None,
    *, historical_architectures: set[str] | None = None,
    historical_joints: set[str] | None = None,
) -> dict[str, object]:
    return gate.validate_candidate_manifest(
        rows or _candidates(),
        historical_architecture_fingerprints=historical_architectures or set(),
        historical_joint_fingerprints=historical_joints or set(),
        expected_pool_count=8,
        expected_candidates_per_pool=4,
        expected_master_seed=2026082801,
        expected_distribution="HIGH_REGIME_PRIOR",
        generator_audit=_generator_audit(),
    )


def _prediction_rows(candidates: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in candidates or _candidates():
        pool = int(row["pool_id"])
        slot = int(row["candidate_index"])
        output.append({
            "pool_id": pool,
            "joint_fingerprint": str(row["joint_fingerprint"]),
            "M1_score": 0.70 + 0.001 * ((slot + 1) % 4),
            "M2_score": 0.70 + 0.001 * slot,
            "B3_score": 0.70 + 0.001 * ((slot + 2) % 4),
        })
    return output


def _selections(
    candidates: list[dict[str, object]] | None = None,
    predictions: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    candidates = candidates or _candidates()
    predictions = predictions or _prediction_rows(candidates)
    random_slots = gate.frozen_random_indices(pool_count=8, choices_per_pool=4, seed=2026082802)
    rows: list[dict[str, object]] = []
    for pool in range(8):
        pool_predictions = [row for row in predictions if int(row["pool_id"]) == pool]
        by_joint = {
            str(row["joint_fingerprint"]): row
            for row in candidates if int(row["pool_id"]) == pool
        }
        selected = {
            method: gate.select_pool_argmax(pool_predictions, score_field=f"{method}_score")
            for method in ("M1", "M2", "B3")
        }
        ordered = sorted(by_joint)
        rows.append({
            "pool_id": pool,
            "M1_selected_joint_fingerprint": selected["M1"],
            "M2_selected_joint_fingerprint": selected["M2"],
            "B3_selected_joint_fingerprint": selected["B3"],
            "R_selected_joint_fingerprint": ordered[int(random_slots[pool])],
            "random_index": int(random_slots[pool]),
        })
    return rows


def _scorer_identity() -> dict[str, object]:
    return {
        "status": "REPRODUCED",
        "folds": 5,
        "metric_seeds": [0, 1, 2],
        "distance_gap_spearman_reproduced": True,
        "gap_at_5_reproduced": True,
        "top5_fingerprints_reproduced": True,
        "M1_oof_predictions_reproduced": True,
        "M2_oof_predictions_reproduced": True,
        "absolute_tolerance": 1.0e-8,
        "test_data_accessed": False,
    }


def _freeze(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    pre_eval = tmp_path / "pre_eval"
    audit = gate.freeze_pre_eval(
        pre_eval,
        candidate_manifest=_candidates(),
        predictions=_prediction_rows(),
        selections=_selections(),
        scorer_identity=_scorer_identity(),
        historical_input_hashes={"soft_archive": "a" * 64},
        generator_identity={"source_sha256": "b" * 64},
    )
    return pre_eval, audit


def _evaluation_rows(
    candidates: list[dict[str, object]] | None = None,
    *, omit: tuple[str, int] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates or _candidates():
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
                "hp_fingerprint": str(candidate["hp_fingerprint"]),
                "seed": seed,
                "val_acc": 0.70 + 0.01 * slot + 0.001 * seed,
                "valid": True,
                "terminal_status": "success",
                "track_test": False,
            })
    return rows


def _completed_evidence(
    *, m2_pass: bool = True, m1_pass: bool = False,
    disagreement: int = 8,
) -> dict[str, object]:
    b3_regret = 0.010
    evidence: dict[str, object] = {
        "input_valid": True,
        "scorer_replay_valid": True,
        "candidate_generation_valid": True,
        "evaluation_complete": True,
        "n_disagreement": disagreement,
        "B3": {"mean_regret": b3_regret, "true_best_hits": 2},
        "R": {"mean_regret": 0.020, "true_best_hits": 1},
        "M2": {
            "wins_vs_B3": 6 if m2_pass else 5,
            "ties_vs_B3": 0,
            "mean_selected_gain_vs_B3": 0.004 if m2_pass else 0.002,
            "mean_regret": 0.007 if m2_pass else 0.009,
            "true_best_hits": 3,
        },
        "M1": {
            "wins_vs_B3": 6 if m1_pass else 4,
            "ties_vs_B3": 0,
            "mean_selected_gain_vs_B3": 0.004 if m1_pass else 0.001,
            "mean_regret": 0.007 if m1_pass else 0.009,
            "true_best_hits": 3,
        },
    }
    return evidence


def _classification(value: object) -> str:
    if isinstance(value, str):
        return value
    assert isinstance(value, dict)
    return str(value["classification"])


def test_frozen_historical_identity_is_336_192_24_1008() -> None:
    payload = {
        **gate.FROZEN_HISTORICAL_IDENTITY,
        "unique_joint_configurations": 336,
        "unique_architectures": 192,
        "unique_pools": 24,
        "seed_record_count": 1008,
        "training_seeds": [0, 1, 2],
    }
    audit = gate.validate_historical_identity(payload)
    assert audit["passed"] is True


def test_soft_metric_frozen_hashes_are_exact() -> None:
    identity = gate.FROZEN_SOFT_METRIC_IDENTITY
    assert identity["classification"] == "SOFT_JOINT_METRIC_REJECTED"
    assert identity["archive_sha256"] == "f5a0a73cb45098994afa3d60de44eb4c54c010d6814d963e3d4c3ba69c219f74"
    assert identity["config_sha256"] == "0b97af8a13d270e80fe8cc7b9a4beaf2616ff624c01faf5832fdebbeae25aadd"
    assert identity["source_sha256"] == "d042a158a2c006baecc7c9d1d50dd42683f7b4f4c4dbdfed0c2f2c8bcc3a3aea"
    assert identity["split_sha256"] == "b80b1a211a05285121cafe4dabd4c67baddea93606c1913e777fcd34dd79a2ec"
    assert identity["source_snapshot_manifest_root"] == "7125d9d7eececca53fcd00c2d578fb685c1b0d2156de62f4de08b362ffe10273"


def test_scorer_replay_identity_checks_oof_gap_and_neighbors() -> None:
    archived = {
        "distance_gap_spearman": [0.1 + index / 100.0 for index in range(15)],
        "gap_at_5": 0.06259007936507938,
        "top5_fingerprints": {"query": [_fingerprint("neighbor", i) for i in range(5)]},
        "M1_oof_predictions": {"query": 0.71},
        "M2_oof_predictions": {"query": 0.72},
    }
    audit = gate.assert_scorer_replay_identity(copy.deepcopy(archived), archived, atol=1.0e-8)
    assert audit["passed"] is True
    changed = copy.deepcopy(archived)
    changed["M2_oof_predictions"]["query"] = 0.720001
    with pytest.raises(gate.ScorerReplayIdentityMismatch):
        gate.assert_scorer_replay_identity(changed, archived, atol=1.0e-8)


def test_candidate_generator_firewall_excludes_predictor_and_labels() -> None:
    audit = _validate_candidates()
    assert audit["predictor_accessed"] is False
    assert audit["validation_labels_accessed"] is False


def test_all_candidate_architectures_are_historical_unseen() -> None:
    rows = _candidates()
    historical = {str(rows[0]["architecture_fingerprint"])}
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows, historical_architectures=historical)


def test_all_joint_fingerprints_are_historical_unseen() -> None:
    rows = _candidates()
    historical = {str(rows[0]["joint_fingerprint"])}
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows, historical_joints=historical)


def test_candidate_architectures_are_internally_unique() -> None:
    rows = _candidates()
    rows[1]["architecture_fingerprint"] = rows[0]["architecture_fingerprint"]
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows)


def test_candidate_pool_count_is_exactly_eight() -> None:
    rows = [row for row in _candidates() if int(row["pool_id"]) != 7]
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows)


def test_each_pool_has_exactly_four_candidates() -> None:
    rows = _candidates()[:-1]
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows)


def test_candidate_hp_distribution_is_high_regime_prior() -> None:
    rows = _candidates()
    rows[0]["proposal_distribution"] = "predictor_guided"
    with pytest.raises(gate.CandidateGenerationError):
        _validate_candidates(rows)


def test_pre_eval_selection_is_frozen_before_evaluation(tmp_path: Path) -> None:
    pre_eval, audit = _freeze(tmp_path)
    assert audit["passed"] is True
    assert (pre_eval / "PRE_EVAL_SELECTION_FROZEN").is_file()
    assert gate.verify_pre_eval_freeze(pre_eval)["passed"] is True


def test_prediction_manifest_is_immutable_after_freeze(tmp_path: Path) -> None:
    pre_eval, _audit = _freeze(tmp_path)
    predictions = pre_eval / "predictions.csv"
    predictions.write_text(predictions.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(gate.PreEvalFreezeError):
        gate.verify_pre_eval_freeze(pre_eval)


def test_pre_eval_root_is_recomputable(tmp_path: Path) -> None:
    pre_eval, audit = _freeze(tmp_path)
    verified = gate.verify_pre_eval_freeze(pre_eval)
    stored = (pre_eval / "pre_eval_root_sha256.txt").read_text(encoding="utf-8").strip()
    assert verified["root_sha256"] == audit["root_sha256"] == stored


def test_cross_fitted_score_aggregation_is_seed_then_fold_mean() -> None:
    values = {
        (fold, seed): np.asarray([fold + seed, 10.0 + fold + seed], dtype=np.float64)
        for fold in range(5) for seed in (0, 1, 2)
    }
    observed = gate.aggregate_cross_fitted_scores(
        values, folds=[0, 1, 2, 3, 4], metric_seeds=[0, 1, 2],
    )
    expected = np.mean(np.stack(list(values.values()), axis=0), axis=0)
    assert np.allclose(observed, expected, atol=0.0, rtol=0.0)


def test_m2_is_isotropic_unstandardized_and_mean_only() -> None:
    audit = gate.validate_scorer_protocol({
        "M2": {
            "input_dim": 8, "input_l2_normalized": True,
            "kernel": "Matern-5/2", "ard": False,
            "lengthscale_count": 1, "input_standardized": False,
            "posterior_variance_used": False,
        },
        "B3": {
            "features": "explicit_graph_plus_transformed_hp",
            "kernel": "Matern-5/2", "ard": True,
            "input_standardized": True, "posterior_variance_used": False,
        },
    })
    assert audit["M2_isotropic_unstandardized"] is True
    assert audit["posterior_mean_only"] is True


def test_b3_uses_frozen_explicit_joint_gp_protocol() -> None:
    audit = gate.validate_scorer_protocol({
        "M2": {
            "input_dim": 8, "input_l2_normalized": True,
            "kernel": "Matern-5/2", "ard": False,
            "lengthscale_count": 1, "input_standardized": False,
            "posterior_variance_used": False,
        },
        "B3": {
            "features": "explicit_graph_plus_transformed_hp",
            "kernel": "Matern-5/2", "ard": True,
            "input_standardized": True, "posterior_variance_used": False,
            "protocol_reused_without_tuning": True,
        },
    })
    assert audit["B3_frozen_protocol"] is True


def test_random_selection_seed_is_frozen() -> None:
    first = gate.frozen_random_indices(pool_count=8, choices_per_pool=4, seed=2026082802)
    second = gate.frozen_random_indices(pool_count=8, choices_per_pool=4, seed=2026082802)
    assert first == second
    assert len(first) == 8 and all(0 <= int(value) < 4 for value in first)


def test_score_ties_break_by_joint_fingerprint() -> None:
    rows = [
        {"joint_fingerprint": "b" * 64, "M2_score": 0.8},
        {"joint_fingerprint": "a" * 64, "M2_score": 0.8},
    ]
    assert gate.select_pool_argmax(rows, score_field="M2_score") == "a" * 64


def test_disagreement_count_is_pool_level() -> None:
    selections = _selections()
    expected = sum(
        row["M2_selected_joint_fingerprint"] != row["B3_selected_joint_fingerprint"]
        for row in selections
    )
    assert gate.disagreement_count(selections) == expected == 8


def test_low_disagreement_authorizes_zero_gnn_runs() -> None:
    tasks = gate.build_evaluation_tasks(
        _candidates(), pre_eval_verified=True, disagreement_pool_count=4,
        minimum_disagreement_pools=5, training_seeds=[0, 1, 2], track_test=False,
    )
    assert tasks == []
    assert _classification(gate.classify_prospective({
        **_completed_evidence(disagreement=4), "evaluation_complete": False,
        "gnn_runs_started": 0,
    })) == "PROSPECTIVE_LOW_DISAGREEMENT"


def test_candidate_manifest_cannot_change_after_selection() -> None:
    before = _candidates()
    after = copy.deepcopy(before)
    after[0]["joint_fingerprint"] = _fingerprint("replacement", 0)
    with pytest.raises(gate.PreEvalFreezeError):
        gate.assert_candidate_manifest_immutable(before, after)


def test_training_seeds_are_exactly_zero_one_two() -> None:
    tasks = gate.build_evaluation_tasks(
        _candidates(), pre_eval_verified=True, disagreement_pool_count=8,
        minimum_disagreement_pools=5, training_seeds=[0, 1, 2], track_test=False,
    )
    assert sorted({int(row["seed"]) for row in tasks}) == [0, 1, 2]


def test_every_pool_uses_matched_training_seeds() -> None:
    tasks = gate.build_evaluation_tasks(
        _candidates(), pre_eval_verified=True, disagreement_pool_count=8,
        minimum_disagreement_pools=5, training_seeds=[0, 1, 2], track_test=False,
    )
    for pool in range(8):
        pool_rows = [row for row in tasks if int(row["pool_id"]) == pool]
        by_joint: dict[str, set[int]] = {}
        for row in pool_rows:
            by_joint.setdefault(str(row["joint_fingerprint"]), set()).add(int(row["seed"]))
        assert set(map(frozenset, by_joint.values())) == {frozenset({0, 1, 2})}


def test_loader_rejects_any_test_field() -> None:
    with pytest.raises(gate.ProtocolError):
        gate.reject_test_fields({"validation": 0.8, "test_accuracy": 0.9})


def test_track_test_is_frozen_false() -> None:
    assert gate.FROZEN_TRAINING_PROTOCOL["track_test"] is False
    with pytest.raises(gate.ProtocolError):
        gate.build_evaluation_tasks(
            _candidates(), pre_eval_verified=True, disagreement_pool_count=8,
            minimum_disagreement_pools=5, training_seeds=[0, 1, 2], track_test=True,
        )


def test_complete_task_manifest_has_exactly_96_tasks() -> None:
    tasks = gate.build_evaluation_tasks(
        _candidates(), pre_eval_verified=True, disagreement_pool_count=8,
        minimum_disagreement_pools=5, training_seeds=[0, 1, 2], track_test=False,
    )
    assert len(tasks) == 96
    assert len({str(row["task_key"]) for row in tasks}) == 96


def test_incomplete_pool_fails_closed() -> None:
    candidates = _candidates()
    missing = (str(candidates[0]["joint_fingerprint"]), 2)
    with pytest.raises(gate.EvaluationIncompleteError):
        gate.aggregate_evaluation_results(candidates, _evaluation_rows(candidates, omit=missing))


def test_observed_oracle_is_maximum_three_seed_mean() -> None:
    statistics = gate.aggregate_evaluation_results(_candidates(), _evaluation_rows())
    pool_zero = [row for row in statistics if int(row["pool_id"]) == 0]
    oracle = max(float(row["validation_mean"]) for row in pool_zero)
    assert all(math.isclose(float(row["observed_pool_oracle"]), oracle) for row in pool_zero)


def test_simple_regret_is_oracle_minus_selected_mean() -> None:
    statistics = gate.aggregate_evaluation_results(_candidates(), _evaluation_rows())
    metrics = gate.compute_prospective_metrics(statistics, _selections())
    for row in metrics["per_pool"]:
        for method in ("M1", "M2", "B3", "R"):
            expected = float(row["observed_pool_oracle"]) - float(row[f"{method}_selected_validation_mean"])
            assert math.isclose(float(row[f"{method}_simple_regret"]), expected, abs_tol=1.0e-15)


def test_true_best_hit_accepts_oracle_ties_at_one_e_minus_twelve() -> None:
    assert gate.oracle_hit(0.8, 0.8 + 5.0e-13, tolerance=1.0e-12) == 1
    assert gate.oracle_hit(0.8, 0.8 + 2.0e-12, tolerance=1.0e-12) == 0


def test_training_seeds_are_not_counted_as_independent_pools() -> None:
    statistics = gate.aggregate_evaluation_results(_candidates(), _evaluation_rows())
    metrics = gate.compute_prospective_metrics(statistics, _selections())
    assert metrics["statistical_unit"] == "pool"
    assert metrics["pool_count"] == 8
    assert metrics["training_seed_count_is_not_independent_unit"] is True


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"n_disagreement": 4}, "METRIC_SELECTION_PROSPECTIVE_REJECTED"),
        ({"M2.wins_vs_B3": 5}, "METRIC_SELECTION_PROSPECTIVE_REJECTED"),
        ({"M2.mean_selected_gain_vs_B3": 0.0029}, "METRIC_SELECTION_PROSPECTIVE_REJECTED"),
        ({"M2.mean_regret": 0.0081}, "METRIC_SELECTION_PROSPECTIVE_REJECTED"),
        ({"M2.true_best_hits": 1}, "METRIC_SELECTION_PROSPECTIVE_REJECTED"),
    ],
)
def test_p1_through_p5_are_all_required(mutation: dict[str, object], expected: str) -> None:
    evidence = _completed_evidence(m2_pass=True)
    for dotted, value in mutation.items():
        if "." in dotted:
            model, field = dotted.split(".", 1)
            evidence[model][field] = value
        else:
            evidence[dotted] = value
    assert _classification(gate.classify_prospective(evidence)) == expected


def test_m2_pass_has_precedence_over_m1() -> None:
    assert _classification(gate.classify_prospective(
        _completed_evidence(m2_pass=True, m1_pass=True)
    )) == "METRIC_GP_PROSPECTIVE_SIGNAL"


def test_m1_cannot_replace_m2_as_gp_signal() -> None:
    assert _classification(gate.classify_prospective(
        _completed_evidence(m2_pass=False, m1_pass=True)
    )) == "METRIC_KNN_PROSPECTIVE_SIGNAL_ONLY"


def test_random_baseline_does_not_enter_primary_gate() -> None:
    first = _completed_evidence(m2_pass=True)
    second = copy.deepcopy(first)
    second["R"] = {"mean_regret": 0.0, "true_best_hits": 8}
    assert _classification(gate.classify_prospective(first)) == _classification(
        gate.classify_prospective(second)
    )


def test_pool_bootstrap_uses_exactly_eight_pool_units() -> None:
    values = [0.001 * index for index in range(8)]
    result = gate.pool_bootstrap(values, replicates=10_000, seed=0, unit="pool")
    assert result["bootstrap_unit"] == "pool"
    assert result["pool_count"] == 8
    assert result["replicates"] == 10_000
    assert len(result["confidence_interval_95"]) == 2


def test_new_labels_cannot_update_any_scorer() -> None:
    audit = {
        "candidate_labels_used_for_metric_update": False,
        "candidate_labels_used_for_gp_update": False,
        "pool_sequential_updates": False,
        "all_predictions_frozen_before_first_evaluation": True,
    }
    assert gate.assert_no_online_update(audit)["passed"] is True
    bad = {**audit, "candidate_labels_used_for_gp_update": True}
    with pytest.raises(gate.ProtocolError):
        gate.assert_no_online_update(bad)


def test_source_snapshot_requires_new_and_frozen_runtime_sources(tmp_path: Path) -> None:
    required = list(gate.REQUIRED_SOURCE_SNAPSHOT_PATHS)
    assert "analyse/joint_metric_prospective_selection_gate.py" in required
    assert "analyse/joint_configuration_soft_metric_gate.py" in required
    assert "analyse/joint_configuration_metric_gate.py" in required
    assert "analyse/performance_metric_quick_gate.py" in required
    assert "analyse/high_regime_prior_model_blind_validation.py" in required
    assert "analyse/run_high_regime_prior_model_blind_validation_labels.py" in required
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frozen:{relative}".encode("utf-8"))
    audit = gate.validate_source_snapshot(tmp_path, required)
    assert audit["passed"] is True
    assert audit["all_required_sources_copied_as_bytes"] is True


def test_formal_result_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    (output / "existing.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(gate.ProtocolError):
        gate.ensure_output_policy(output)


def test_dry_run_does_not_create_formal_result_directory(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    result = gate.execute_dry_run(gate.DEFAULT_CONFIG, output)
    assert result == 0
    assert not output.exists()


_FROZEN_HIGH_IMPORT_PROBE_CACHE: dict[bool, dict[str, object]] = {}


def _frozen_high_import_probe(preseed_wrong_order: bool = False) -> dict[str, object]:
    """Import the frozen high-regime stack in a fresh interpreter, without candidates."""

    cached = _FROZEN_HIGH_IMPORT_PROBE_CACHE.get(preseed_wrong_order)
    if cached is not None:
        return cached

    import json
    import subprocess
    import sys

    probe_source = f"""
import json
import sys
from pathlib import Path

import analyse

namespace_path_type = type(analyse.__path__).__name__
from analyse import joint_metric_prospective_selection_gate as gate

frozen_analyse = (
    gate.REPO_ROOT
    / "results/search/high-regime-prior-model-blind-validation-global4-v1"
    / "source_snapshot/analyse"
).resolve()
live_analyse = (gate.REPO_ROOT / "analyse").resolve()

if {preseed_wrong_order!r}:
    analyse.__path__ = [
        str(live_analyse),
        str(frozen_analyse),
        str(frozen_analyse),
    ]

high = gate._load_frozen_high_module()
dependency_names = [
    "analyse.original_head_confirmatory_attribution_v2",
    "analyse.original_head_deterministic_batch_pilot",
]
dependency_files = {{
    name: str(Path(sys.modules[name].__file__).resolve())
    for name in dependency_names
}}
required_v1 = [
    "load_joint_vae",
    "config_from_igraph",
    "validate_static_config",
    "architecture_fingerprint",
    "realize_global4_hp",
    "hp_fingerprint",
]
smoke = gate.frozen_module_import_smoke()
resolved_paths = [str(Path(path).resolve()) for path in analyse.__path__]
payload = {{
    "namespace_path_type": namespace_path_type,
    "high_file": str(Path(high.__file__).resolve()),
    "dependency_files": dependency_files,
    "frozen_analyse": str(frozen_analyse),
    "live_analyse": str(live_analyse),
    "import_paths": resolved_paths,
    "frozen_path_count": resolved_paths.count(str(frozen_analyse)),
    "required_top_callable": callable(getattr(high, "joint_config_fingerprint", None)),
    "required_v1_callables": {{
        name: callable(getattr(high.v1, name, None)) for name in required_v1
    }},
    "smoke_passed": bool(smoke.get("passed")),
}}
print("FROZEN_HIGH_IMPORT_PROBE=" + json.dumps(payload, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe_source],
        cwd=gate.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    marker = "FROZEN_HIGH_IMPORT_PROBE="
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    assert len(lines) == 1, completed.stdout
    payload = json.loads(lines[0][len(marker) :])
    _FROZEN_HIGH_IMPORT_PROBE_CACHE[preseed_wrong_order] = payload
    return payload


def test_frozen_high_loader_accepts_real_namespace_package_path() -> None:
    probe = _frozen_high_import_probe()
    assert probe["namespace_path_type"] == "_NamespacePath"


def test_frozen_high_main_module_origin_is_snapshot() -> None:
    from pathlib import Path

    probe = _frozen_high_import_probe()
    high_file = Path(str(probe["high_file"]))
    frozen_analyse = Path(str(probe["frozen_analyse"]))
    live_analyse = Path(str(probe["live_analyse"]))
    assert high_file.is_relative_to(frozen_analyse)
    assert not high_file.is_relative_to(live_analyse)


def test_frozen_high_dependency_origins_are_snapshot() -> None:
    from pathlib import Path

    probe = _frozen_high_import_probe()
    frozen_analyse = Path(str(probe["frozen_analyse"]))
    live_analyse = Path(str(probe["live_analyse"]))
    dependency_files = dict(probe["dependency_files"])
    assert set(dependency_files) == {
        "analyse.original_head_confirmatory_attribution_v2",
        "analyse.original_head_deterministic_batch_pilot",
    }
    for source_file in dependency_files.values():
        resolved = Path(str(source_file))
        assert resolved.is_relative_to(frozen_analyse)
        assert not resolved.is_relative_to(live_analyse)


def test_frozen_high_path_is_first_and_not_duplicated() -> None:
    probe = _frozen_high_import_probe(preseed_wrong_order=True)
    assert probe["import_paths"][0] == probe["frozen_analyse"]
    assert probe["frozen_path_count"] == 1


def test_frozen_high_candidate_import_surface_smoke_without_generation() -> None:
    probe = _frozen_high_import_probe()
    assert probe["required_top_callable"] is True
    assert all(dict(probe["required_v1_callables"]).values())
    assert probe["smoke_passed"] is True


def test_engineering_fix_payload_and_rerun_defaults() -> None:
    payload = gate.engineering_fix_payload()
    assert payload["parent_failed_archive_sha256"] == (
        "a09123b997c4cef6ac4cd02d4261711002e3a9d28c2431ed0321d2d03f81f955"
    )
    assert payload["parent_failure"] == (
        "AttributeError: '_NamespacePath' object has no attribute 'insert'"
    )
    assert payload["scientific_protocol_changed"] is False
    assert payload["candidate_generation_changed"] is False
    assert payload["candidate_seeds_changed"] is False
    assert payload["scorer_changed"] is False
    assert payload["selection_gate_changed"] is False
    assert gate.PARENT_FAILED_ARCHIVE_SHA256 == payload["parent_failed_archive_sha256"]
    assert gate.PARENT_FAILURE == payload["parent_failure"]
    assert gate.DEFAULT_OUTPUT.name == (
        "joint-metric-prospective-selection-gate-global4-v1-rerun1"
    )
    assert gate.DEFAULT_ARCHIVE.name == (
        "joint-metric-prospective-selection-gate-global4-v1-rerun1.tar.gz"
    )


def test_failure_bundle_is_written_and_archived(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    archive = tmp_path / "formal.tar.gz"
    gate.write_failure_bundle(output, gate.ProtocolError("synthetic failure"), archive_path=archive)
    assert (output / "FAILED").is_file()
    assert (output / "error.json").is_file()
    assert archive.is_file()
    assert archive.with_suffix(archive.suffix + ".sha256").is_file()
    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    assert any(name.endswith("FAILED") for name in names)
    assert any(name.endswith("error.json") for name in names)


def test_synthetic_smoke_exercises_all_terminal_gate_routes() -> None:
    result = gate.synthetic_smoke()
    assert result["passed"] is True
    for classification in (
        "PROSPECTIVE_INPUT_INVALID",
        "SCORER_REPLAY_IDENTITY_MISMATCH",
        "CANDIDATE_GENERATION_FAILED",
        "PROSPECTIVE_LOW_DISAGREEMENT",
        "PROSPECTIVE_EVALUATION_INCOMPLETE",
        "METRIC_SELECTION_PROSPECTIVE_REJECTED",
        "METRIC_KNN_PROSPECTIVE_SIGNAL_ONLY",
        "METRIC_GP_PROSPECTIVE_SIGNAL",
    ):
        assert result[classification] is True


def test_frozen_protocol_rejects_threshold_or_model_changes() -> None:
    config = gate.load_config(gate.DEFAULT_CONFIG)
    gate.validate_frozen_protocol(config)
    changed = copy.deepcopy(config)
    changed["prospective_gate"]["P3"]["mean_selected_validation_advantage_all_8_at_least"] = 0.002
    with pytest.raises(gate.ProtocolError):
        gate.validate_frozen_protocol(changed)
