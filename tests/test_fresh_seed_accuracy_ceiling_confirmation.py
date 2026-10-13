from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import fresh_seed_accuracy_ceiling_confirmation as gate


@pytest.fixture(scope="session")
def protocol() -> dict:
    value = gate.load_config()
    assert gate.validate_frozen_protocol(value) == {
        "status": "VALID",
        "protocol_version": gate.PROTOCOL_VERSION,
        "test_accessed": False,
    }
    return value


def _candidate(index: int, value: float, *, provenance: str = "METHOD") -> dict:
    row = {
        "operations": ["input", f"op-{index}", "output"],
        "edges": [[0, 1], [1, 2]],
        "raw_hp_vector": [0.1 + index * 0.001, 0.2, 0.3, 0.4],
        "lr": 0.01 + index * 0.0001,
        "dropout": 0.25,
        "hidden_dim": 64,
        "l2": 0.0005,
        "historical_seed_0": value - 0.001,
        "historical_seed_1": value,
        "historical_seed_2": value + 0.001,
        "historical_validation_mean": value,
        "historical_validation_sd": 0.001,
        "source_experiments": ["synthetic"],
        "originally_selected_by": [provenance],
    }
    architecture, hp, joint = gate.recompute_fingerprints(row)
    row.update({
        "architecture_fingerprint": architecture,
        "hp_fingerprint": hp,
        "joint_fingerprint": joint,
        "historical_score": round(value, 12),
    })
    return row


def _raw_results(
    candidate_joints: list[str], seeds: tuple[int, ...], *,
    p_base: float = 0.80, deltas: dict[str, float] | None = None,
) -> list[dict]:
    deltas = deltas or {joint: 0.001 * (index + 1) for index, joint in enumerate(candidate_joints)}
    rows: list[dict] = []
    for seed_index, seed in enumerate(seeds):
        p_value = p_base + 0.0001 * seed_index
        rows.append({
            "joint_fingerprint": gate.P_JOINT,
            "training_seed": seed,
            "validation_accuracy": p_value,
            "terminal_status": "success",
            "valid": True,
            "attempt_count": 1,
        })
        for joint in candidate_joints:
            rows.append({
                "joint_fingerprint": joint,
                "training_seed": seed,
                "validation_accuracy": p_value + deltas[joint],
                "terminal_status": "success",
                "valid": True,
                "attempt_count": 1,
            })
    return rows


# 01-06: authoritative source hashes, classifications, test firewall, and
# compatible evaluator identity are immutable protocol fields.
def test_01_to_06_source_identities_and_protocol(protocol: dict) -> None:
    expected = {
        "A": (
            "afb4bcf6dcc490ef884203dfa22414bbc1e203178b54f623fdb2ebdf1bf56ee9",
            "B4_SEARCH_NOT_CONFIRMED",
        ),
        "B": (
            "49e7b91956c433597f6f5e076ccaf2cf8d110b703388c9c81ad4dd5a8a141597",
            "METRIC_SEARCH_NOT_CONFIRMED",
        ),
        "C": (
            "aaa471d6e97b88c70d7f11b0ded1b3833895f1e24ed5e9489615e2d1296af1be",
            "DIVERSITY_SEARCH_NOT_CONFIRMED",
        ),
    }
    for source_id, (digest, classification) in expected.items():
        source = protocol["authoritative_sources"][source_id]
        assert source["archive_sha256"] == digest
        assert source["required_classification"] == classification
        assert source["required_test_accessed"] is False
    evaluator = protocol["evaluation_protocol"]
    assert evaluator["compatibility_required_across_sources"] is True
    assert evaluator["dataset"] == "Cora"
    assert evaluator["split"] == "public"
    assert evaluator["max_epochs"] == 300
    assert evaluator["patience"] == 80
    assert evaluator["validation_metric"] == "accuracy"
    assert evaluator["track_test"] is False


# 07-09: only complete real seed-0/1/2 rows may enter the exact-joint union.
def test_07_to_09_candidate_universe_membership_and_exact_dedup(
    protocol: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    rows = [_candidate(index, 0.78 + index * 0.001) for index in range(3)]
    roots = {source_id: tmp_path / source_id for source_id in gate.SOURCE_IDS}
    stats_by_source: dict[str, list[dict]] = {}
    manifests_by_source: dict[str, list[dict]] = {}
    for index, source_id in enumerate(gate.SOURCE_IDS):
        candidate = copy.deepcopy(rows[index])
        (roots[source_id] / "evaluation").mkdir(parents=True)
        (roots[source_id] / "evaluation/task_manifest.json").write_text(json.dumps({
            "tasks": [{
                "joint_config_fingerprint": candidate["joint_fingerprint"],
                "seed": seed,
                "task_key": f"synthetic::{source_id}::{seed}",
            } for seed in gate.HISTORICAL_SEEDS],
        }), encoding="utf-8")
        stats_by_source[source_id] = [{
            "joint_fingerprint": candidate["joint_fingerprint"],
            "validation_seed_0": candidate["historical_seed_0"],
            "validation_seed_1": candidate["historical_seed_1"],
            "validation_seed_2": candidate["historical_seed_2"],
            "validation_mean": candidate["historical_validation_mean"],
            "validation_sd": candidate["historical_validation_sd"],
            "selected_by_B4": "true",
            "selected_by_M1": "true",
            "selected_by_pure_b4": "true",
        }]
        manifests_by_source[source_id] = [candidate]
    local = copy.deepcopy(protocol)
    for source_id in gate.SOURCE_IDS:
        local["authoritative_sources"][source_id]["evaluated_config_count"] = 1

    def source_for(path: Path) -> str:
        return path.parent.parent.name

    monkeypatch.setattr(gate, "SOURCE_ROOTS", roots)
    monkeypatch.setattr(gate, "EXPECTED_UNIVERSE", 3)
    monkeypatch.setattr(gate, "read_csv", lambda path: stats_by_source[source_for(path)])
    monkeypatch.setattr(gate, "_parse_candidate_manifest", lambda path: manifests_by_source[source_for(path)])
    universe, audit = gate.build_candidate_universe(local)
    assert len(universe) == 3
    assert len({row["joint_fingerprint"] for row in universe}) == 3
    assert audit["only_complete_real_seed_0_1_2_evaluations"] is True
    assert audit["predictor_only_candidates_included"] is False

    # An absent real seed is not silently admitted.
    stats_by_source["A"][0]["validation_seed_2"] = "nan"
    with pytest.raises(gate.InputIdentityInvalid, match="incomplete seed0/1/2"):
        gate.build_candidate_universe(local)


# 10-18: P is a reference, historical scores are rounded to 12 decimals,
# fingerprint order breaks ties, full configuration identities are recovered,
# and provenance never influences the ranking.
def test_10_to_18_reference_ranking_recovery_and_provenance() -> None:
    assert gate.P_JOINT == "30e0bb108c1fdbe8db186152a09c3f2fc6e812be189db920fe4d0d2b0d3cb1a2"
    candidates = [_candidate(index, 0.79 + index * 0.001) for index in range(8)]
    # Repeating the exact same joint in another source contributes one ranking
    # record, never another challenger slot.
    duplicate_ranked, _ = gate.rank_historical_candidates(
        [candidates[0], {**candidates[0], "source_experiments": ["second-source"]}],
        p_joint=gate.P_JOINT,
    )
    assert len(duplicate_ranked) == 1
    candidates.append({**_candidate(99, 0.999), "joint_fingerprint": gate.P_JOINT})
    ranked, top = gate.rank_historical_candidates(
        candidates, p_joint=gate.P_JOINT, top_n=6,
    )
    assert all(row["joint_fingerprint"] != gate.P_JOINT for row in ranked)
    assert len(top) == 6
    assert [row["historical_validation_mean"] for row in top] == sorted(
        [row["historical_validation_mean"] for row in top], reverse=True,
    )

    # Values identical after 12-decimal rounding use fingerprint order, and
    # changing descriptive provenance cannot change the order.
    tied_a = _candidate(200, 0.80800000000031, provenance="RANDOM")
    tied_b = _candidate(201, 0.80800000000039, provenance="B4")
    tied_a["joint_fingerprint"], tied_b["joint_fingerprint"] = "a", "b"
    first = gate.rank_historical_candidates([tied_b, tied_a], p_joint=gate.P_JOINT, top_n=2)[1]
    tied_a["originally_selected_by"], tied_b["originally_selected_by"] = ["M2"], ["P2"]
    second = gate.rank_historical_candidates([tied_a, tied_b], p_joint=gate.P_JOINT, top_n=2)[1]
    assert [row["joint_fingerprint"] for row in first] == ["a", "b"]
    assert [row["joint_fingerprint"] for row in second] == ["a", "b"]

    recovered = _candidate(4, 0.8)
    architecture, hp, joint = gate.recompute_fingerprints(recovered)
    assert recovered["operations"] and recovered["edges"]
    assert all(key in recovered for key in ("lr", "dropout", "hidden_dim", "l2", "raw_hp_vector"))
    assert (architecture, hp, joint) == (
        recovered["architecture_fingerprint"], recovered["hp_fingerprint"],
        recovered["joint_fingerprint"],
    )


def test_14_expected_top6_identity_is_frozen(protocol: dict) -> None:
    assert gate.EXPECTED_UNIVERSE == 1121
    assert protocol["candidate_universe"]["expected_unique_joint_config_count"] == 1121
    assert tuple(protocol["candidate_universe"]["expected_top6_joint_fingerprints"]) == gate.EXPECTED_TOP6
    assert len(gate.EXPECTED_TOP6) == len(set(gate.EXPECTED_TOP6)) == 6


# 19-21: exact, disjoint stage seed schedules and budgets.
def test_19_to_21_stage_seed_schedules(protocol: dict) -> None:
    assert gate.STAGE1_SEEDS == tuple(range(1000, 1015))
    assert gate.STAGE2_SEEDS == tuple(range(2000, 2030))
    assert not set(gate.STAGE1_SEEDS) & set(gate.STAGE2_SEEDS)
    assert protocol["budget"] == {
        "stage1_GNN_runs": 105,
        "stage2_GNN_runs": 90,
        "total_GNN_runs": 195,
        "stage2_may_start_only_after_complete_stage1_and_frozen_selection": True,
        "cross_stage_run_reuse": "forbidden",
    }


# 22: an exact config x fresh-seed collision fails closed; another config or
# another seed is not a collision.
def test_22_fresh_seed_collision_audit() -> None:
    target_joints = {"candidate-a", gate.P_JOINT}
    target_seeds = {1000, 2000}
    historical = [
        {"joint_fingerprint": "candidate-a", "training_seed": 999, "terminal_status": "success", "valid": True},
        {"joint_fingerprint": "candidate-b", "training_seed": 1000, "terminal_status": "success", "valid": True},
    ]
    assert gate.collision_pairs_from_rows(historical, target_joints, target_seeds) == set()
    historical.append({
        "joint_fingerprint": "candidate-a", "training_seed": 1000,
        "terminal_status": "success", "valid": True,
    })
    assert gate.collision_pairs_from_rows(historical, target_joints, target_seeds) == {("candidate-a", 1000)}


# 23-24: the protocol freezes all Stage-1 rules before fresh labels and keeps
# the test firewall false at configuration and mutable-state boundaries.
def test_23_to_24_pre_stage1_rules_and_test_firewall(protocol: dict, tmp_path: Path) -> None:
    assert protocol["stage1"]["role"] == "independent_fresh_seed_screening_only"
    assert protocol["stage2"]["role"] == "sole_confirmatory_dataset"
    assert protocol["stage1"]["final_classification_use"] == "forbidden"
    assert protocol["inference"]["practical_gain"]["minimum_mean_delta"] == 0.003
    state = gate.initial_state("c" * 64, "s" * 64)
    gate.dump_json(tmp_path / "state.json", state)
    assert gate.update_state(tmp_path, test_accessed=False)["test_accessed"] is False
    with pytest.raises(gate.ProtocolError, match="test_accessed"):
        gate.update_state(tmp_path, test_accessed=True)


def test_23_pre_stage1_freeze_is_complete_and_immutable(protocol: dict, tmp_path: Path) -> None:
    universe = [_candidate(index, 0.80 + index * 0.001) for index in range(6)]
    ranked, top6 = gate.rank_historical_candidates(universe, top_n=6)
    for rank, row in enumerate(top6, 1):
        row["historical_rank"] = rank
    p = _candidate(100, gate.P_OLD_MEAN)
    p.update({
        "joint_fingerprint": gate.P_JOINT,
        "architecture_fingerprint": gate.P_ARCH,
        "hp_fingerprint": gate.P_HP,
        "role": "REFERENCE_CONFIGURATION",
        "test_accessed": False,
    })
    root = tmp_path / "pre_stage1"
    gate.freeze_pre_stage1(
        root, ranked, top6, p,
        {"status": "COMPATIBLE", "track_test": False, "test_accessed": False},
        protocol,
    )
    verified = gate.verify_pre_stage1(root)
    assert verified["passed"] is True
    assert verified["file_count"] == 9
    assert (root / "STAGE1_FROZEN").is_file()
    selection_rule = json.loads((root / "stage2_selection_rule.json").read_text(encoding="utf-8"))
    assert selection_rule["challenger_count"] == 2
    assert selection_rule["stage1_may_not_classify_final_accuracy"] is True
    assert selection_rule["final_gate_uses_stage2_only"] is True
    assert selection_rule["statistical_gates"]["paired_bootstrap"]["replicates"] == 20_000
    assert "validation_accuracy" not in "\n".join(
        path.read_text(encoding="utf-8") for path in root.iterdir() if path.is_file()
    )
    (root / "stage1_seed_spec.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(gate.ProtocolError, match="immutable file changed"):
        gate.verify_pre_stage1(root)


# 25-29: Stage 1 is a complete 105-run matched-seed screen; paired deltas and
# deterministic top-two selection are correct, and it makes no final claim.
def test_25_to_29_stage1_complete_paired_screening() -> None:
    candidates = [_candidate(index, 0.79 + index * 0.001) for index in range(6)]
    joints = [row["joint_fingerprint"] for row in candidates]
    deltas = {joint: 0.001 * (index + 1) for index, joint in enumerate(joints)}
    raw = _raw_results(joints, gate.STAGE1_SEEDS, deltas=deltas)
    audit = gate.validate_stage_results(
        raw, joints + [gate.P_JOINT], gate.STAGE1_SEEDS, "stage1",
    )
    assert audit["expected_runs"] == audit["successful_runs"] == 105
    assert audit["complete"] is True

    paired, stats = gate.paired_candidate_statistics(raw, joints, gate.P_JOINT, gate.STAGE1_SEEDS)
    assert len(paired) == 6 * 15
    by_joint = {row["joint_fingerprint"]: row for row in stats}
    for joint in joints:
        assert by_joint[joint]["seed_count"] == 15
        assert by_joint[joint]["mean_paired_difference"] == pytest.approx(deltas[joint])
        assert by_joint[joint]["wins"] == 15
        assert by_joint[joint]["ties"] == by_joint[joint]["losses"] == 0
    ranking = gate.rank_stage1_candidates(stats)
    assert [row["joint_fingerprint"] for row in ranking[:2]] == list(reversed(joints[-2:]))
    assert sum(row["advanced_to_stage2"] for row in ranking) == 2
    assert all("classification" not in row for row in ranking)


def test_28_stage1_top2_fingerprint_tie_break() -> None:
    stats = [
        {"joint_fingerprint": "b", "mean_paired_difference": 0.002},
        {"joint_fingerprint": "a", "mean_paired_difference": 0.002},
        {"joint_fingerprint": "c", "mean_paired_difference": 0.001},
    ]
    ranking = gate.rank_stage1_candidates(stats)
    assert [row["joint_fingerprint"] for row in ranking[:2]] == ["a", "b"]


# 30: Stage-2 identities are immutable before any Stage-2 task is built.
def test_30_stage2_selection_freeze_detects_mutation(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    for name in ("raw_seed_results.csv", "paired_differences.csv", "candidate_statistics.csv", "stage1_ranking.csv"):
        (stage1 / name).write_text("field\nvalue\n", encoding="utf-8")
    ranking = [
        {"joint_fingerprint": "a", "mean_paired_difference": 0.002},
        {"joint_fingerprint": "b", "mean_paired_difference": 0.001},
    ]
    gate.freeze_stage2_selection(stage1, ranking)
    assert gate.verify_stage2_selection(stage1)["passed"] is True
    selection = json.loads((stage1 / "stage2_selection.json").read_text(encoding="utf-8"))
    assert selection["selection_uses_stage1_only"] is True
    assert selection["final_classification_made"] is False
    (stage1 / "stage2_selection.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(gate.ProtocolError, match="immutable file changed"):
        gate.verify_stage2_selection(stage1)


# 31-32: Stage 2 is exactly two challengers plus freshly retrained P on all 30
# matched seeds and fails closed on any missing or failed run.
def test_31_to_32_stage2_requires_exact_90_run_top2_plus_p() -> None:
    candidates = [_candidate(index, 0.80 + index * 0.001) for index in range(2)]
    p = _candidate(50, gate.P_OLD_MEAN)
    p["joint_fingerprint"] = gate.P_JOINT
    normalized, tasks = gate.build_stage_tasks(candidates + [p], gate.STAGE2_SEEDS, "stage2")
    assert len(normalized) == 3
    assert {row["joint_fingerprint"] for row in normalized} == {
        candidates[0]["joint_fingerprint"], candidates[1]["joint_fingerprint"], gate.P_JOINT,
    }
    assert len(tasks) == 90
    assert {(row["joint_fingerprint"], row["training_seed"]) for row in tasks} == {
        (row["joint_fingerprint"], seed) for row in normalized for seed in gate.STAGE2_SEEDS
    }

    raw = _raw_results([row["joint_fingerprint"] for row in candidates], gate.STAGE2_SEEDS)
    audit = gate.validate_stage_results(
        raw, [row["joint_fingerprint"] for row in candidates] + [gate.P_JOINT],
        gate.STAGE2_SEEDS, "stage2",
    )
    assert audit["expected_runs"] == audit["successful_runs"] == 90
    with pytest.raises(gate.StageEvaluationIncomplete, match="incomplete"):
        gate.validate_stage_results(
            raw[:-1], [row["joint_fingerprint"] for row in candidates] + [gate.P_JOINT],
            gate.STAGE2_SEEDS, "stage2",
        )


# 54-55 are exercised at the task/evaluator boundary without starting CUDA:
# track_test remains false and retries cannot alter the exact task identity.
def test_54_to_55_task_identity_and_retry_contract(protocol: dict) -> None:
    candidate = _candidate(0, 0.8)
    normalized, tasks = gate.build_stage_tasks([candidate], gate.STAGE1_SEEDS, "stage1")
    assert len(normalized) == 1 and len(tasks) == 15
    assert len({row["task_key"] for row in tasks}) == 15
    assert all(row["candidate"]["joint_fingerprint"] == candidate["joint_fingerprint"] for row in tasks)
    assert all(row["training_seed"] in gate.STAGE1_SEEDS for row in tasks)
    assert protocol["evaluation_protocol"]["retry_preserves_exact_config_and_seed"] is True
    assert protocol["evaluation_protocol"]["candidate_replacement_after_failure"] == "forbidden"
    assert protocol["evaluation_protocol"]["track_test"] is False


# 56-57: failures and duplicates in either stage are never analyzed as a
# successful subset.
@pytest.mark.parametrize(
    ("stage", "seeds"), (("stage1", gate.STAGE1_SEEDS), ("stage2", gate.STAGE2_SEEDS)),
)
def test_56_to_57_incomplete_stages_fail_closed(stage: str, seeds: tuple[int, ...]) -> None:
    candidate = _candidate(0, 0.8)["joint_fingerprint"]
    raw = _raw_results([candidate], seeds)
    raw[0]["terminal_status"] = "failed"
    raw[0]["valid"] = False
    with pytest.raises(gate.StageEvaluationIncomplete, match="incomplete"):
        gate.validate_stage_results(raw, [candidate, gate.P_JOINT], seeds, stage)
    raw = _raw_results([candidate], seeds)
    raw.append(dict(raw[0]))
    with pytest.raises(gate.StageEvaluationIncomplete, match="duplicate"):
        gate.validate_stage_results(raw, [candidate, gate.P_JOINT], seeds, stage)


# 33-35: final inference consumes Stage-2 paired labels only; Stage-1 labels,
# historical seed-0/1/2 estimates, and method provenance are inert metadata.
def test_33_to_35_final_gate_ignores_nonconfirmatory_inputs() -> None:
    basic = [{
        "joint_fingerprint": "challenger",
        "seed_count": 30,
        "candidate_validation_mean": 0.804,
        "candidate_validation_sd": 0.005,
        "P_validation_mean": 0.800,
        "P_validation_sd": 0.005,
        "mean_paired_difference": 0.004,
        "median_paired_difference": 0.004,
        "paired_difference_sd": 0.0,
        "wins": 30,
        "losses": 0,
        "ties": 0,
        "paired_differences": [0.004] * 30,
    }]
    first = gate.analyze_stage2(basic)[2]
    altered = copy.deepcopy(basic)
    altered[0].update({
        "stage1_mean": -100.0,
        "historical_validation_mean": -100.0,
        "originally_selected_by": ["different-method"],
        "predictor_score": -100.0,
    })
    second = gate.analyze_stage2(altered)[2]
    assert first == second
    assert first["stage1_labels_used_in_final_gate"] is False
    assert first["historical_seed_0_1_2_used_in_final_gate"] is False
    assert first["method_provenance_used_in_final_gate"] is False


# 36-40: bootstrap resamples the complete paired seed vector, uses 20,000
# draws from master seed 0, and reports the preregistered 2.5th percentile.
def test_36_to_40_paired_bootstrap_identity_and_percentile() -> None:
    deltas = np.linspace(-0.002, 0.006, 30)
    observed = gate.paired_bootstrap(deltas)
    generator = np.random.default_rng(0)
    indices = generator.integers(0, 30, size=(20_000, 30))
    means = deltas[indices].mean(axis=1)
    assert observed["sampling_unit"] == "matched_training_seed"
    assert observed["seed_count"] == 30
    assert observed["replicates"] == 20_000
    assert observed["bonferroni_one_sided_alpha"] == 0.025
    assert observed["bonferroni_lower_bound_percentile"] == 2.5
    assert observed["bonferroni_lower_bound"] == pytest.approx(np.percentile(means, 2.5))
    assert observed["descriptive_ci95"] == pytest.approx(
        [np.percentile(means, 2.5), np.percentile(means, 97.5)],
    )
    assert observed == gate.paired_bootstrap(deltas)


def _stage2_stat(joint: str, deltas: list[float]) -> dict:
    candidate_values = [0.8 + value for value in deltas]
    return {
        "joint_fingerprint": joint,
        "seed_count": len(deltas),
        "candidate_validation_mean": float(np.mean(candidate_values)),
        "candidate_validation_sd": float(np.std(candidate_values, ddof=1)),
        "P_validation_mean": 0.8,
        "P_validation_sd": 0.0,
        "mean_paired_difference": float(np.mean(deltas)),
        "median_paired_difference": float(np.median(deltas)),
        "paired_difference_sd": float(np.std(deltas, ddof=1)),
        "wins": sum(value > gate.NUMERICAL_TOLERANCE for value in deltas),
        "losses": sum(value < -gate.NUMERICAL_TOLERANCE for value in deltas),
        "ties": sum(abs(value) <= gate.NUMERICAL_TOLERANCE for value in deltas),
        "paired_differences": deltas,
    }


# 41-44: stable and practical gates implement the strict-positive tests,
# 0.003 margin, and the frozen numerical tolerance.
def test_41_to_44_stable_and_practical_gate_boundaries() -> None:
    practical = _stage2_stat("practical", [0.003] * 30)
    small = _stage2_stat("small", [0.002] * 30)
    unstable = _stage2_stat("unstable", [-0.009] * 15 + [0.010] * 15)
    tolerance = _stage2_stat("tolerance", [gate.NUMERICAL_TOLERANCE] * 30)
    rows, _bootstraps, decision = gate.analyze_stage2([practical, small, unstable, tolerance])
    by_joint = {row["joint_fingerprint"]: row for row in rows}
    assert by_joint["practical"]["SP1_mean_delta_positive"] is True
    assert by_joint["practical"]["SP2_bonferroni_lower_positive"] is True
    assert by_joint["practical"]["PRACTICAL_GAIN"] is True
    assert by_joint["small"]["STABLE_POSITIVE_GAIN"] is True
    assert by_joint["small"]["PRACTICAL_GAIN"] is False
    assert by_joint["unstable"]["SP1_mean_delta_positive"] is True
    assert by_joint["unstable"]["SP2_bonferroni_lower_positive"] is False
    assert by_joint["tolerance"]["SP1_mean_delta_positive"] is False
    assert decision["classification"] == "PRACTICAL_ACCURACY_IMPROVEMENT_CONFIRMED"


# 45: the terminal scientific classifications are exhaustive and each branch
# of the frozen Stage-2 rule is reachable.
@pytest.mark.parametrize(
    ("flags", "expected"),
    (
        ((True, True), "PRACTICAL_ACCURACY_IMPROVEMENT_CONFIRMED"),
        ((True, False), "SMALL_STABLE_ACCURACY_GAIN"),
        ((False, False), "NO_STABLE_ACCURACY_GAIN"),
    ),
)
def test_45_classification_branches(flags: tuple[bool, bool], expected: str) -> None:
    stable, practical = flags
    decision = gate.classify_stage2([{
        "joint_fingerprint": "candidate",
        "STABLE_POSITIVE_GAIN": stable,
        "PRACTICAL_GAIN": practical,
    }])
    assert decision["classification"] == expected
    assert expected in gate.ALLOWED_CLASSIFICATIONS
    assert set(gate.ALLOWED_CLASSIFICATIONS) == {
        "INPUT_IDENTITY_INVALID", "INPUT_PROTOCOL_MISMATCH", "CANDIDATE_RECOVERY_FAILED",
        "FRESH_SEED_COLLISION", "STAGE1_EVALUATION_INCOMPLETE",
        "STAGE2_EVALUATION_INCOMPLETE", "PRACTICAL_ACCURACY_IMPROVEMENT_CONFIRMED",
        "SMALL_STABLE_ACCURACY_GAIN", "NO_STABLE_ACCURACY_GAIN", "RUN_FAILED",
    }


# 46: practical qualification has priority; within a level mean delta wins and
# a <=1e-12 tie is resolved lexicographically.
def test_46_two_qualified_candidate_selection() -> None:
    common = {"STABLE_POSITIVE_GAIN": True, "candidate_validation_mean": 0.81,
              "P_validation_mean": 0.80, "bonferroni_lower_bound": 0.001}
    small = {**common, "joint_fingerprint": "small", "PRACTICAL_GAIN": False,
             "mean_paired_difference": 0.02}
    practical_b = {**common, "joint_fingerprint": "b", "PRACTICAL_GAIN": True,
                   "mean_paired_difference": 0.004}
    practical_a = {**common, "joint_fingerprint": "a", "PRACTICAL_GAIN": True,
                   "mean_paired_difference": 0.004 + gate.NUMERICAL_TOLERANCE / 2}
    winner = gate.select_new_stable_best([small, practical_b, practical_a])
    assert winner is not None
    assert winner["joint_fingerprint"] == "a"
    assert winner["qualification_level"] == "PRACTICAL_GAIN"


# 47: a positive classification freezes complete candidate identity and paired
# seed evidence; a negative classification creates no frozen-best marker.
def test_47_winner_freeze(tmp_path: Path) -> None:
    candidate = _candidate(0, 0.81)
    joint = candidate["joint_fingerprint"]
    winner = {
        "joint_fingerprint": joint, "qualification_level": "PRACTICAL_GAIN",
        "candidate_validation_mean": 0.805, "P_validation_mean": 0.8,
        "mean_paired_difference": 0.005, "bonferroni_lower_bound": 0.001,
    }
    stage1 = [{"joint_fingerprint": joint, "candidate_validation_mean": 0.806}]
    raw = _raw_results([joint], gate.STAGE2_SEEDS, deltas={joint: 0.005})
    paired, _stats = gate.paired_candidate_statistics(raw, [joint], gate.P_JOINT, gate.STAGE2_SEEDS)
    frozen = gate.write_final_candidate(
        tmp_path, "PRACTICAL_ACCURACY_IMPROVEMENT_CONFIRMED", winner,
        {joint: candidate}, stage1, raw, paired,
    )
    assert frozen["status"] == "FROZEN"
    assert (tmp_path / "final_candidate/NEW_STABLE_BEST_FROZEN").is_file()
    payload = json.loads((tmp_path / "final_candidate/new_stable_best.json").read_text(encoding="utf-8"))
    assert payload["joint_fingerprint"] == joint
    assert payload["candidate_level_claim_only"] is True
    assert payload["method_superiority_claim"] is False

    negative = tmp_path / "negative"
    result = gate.write_final_candidate(
        negative, "NO_STABLE_ACCURACY_GAIN", None, {joint: candidate}, stage1, raw, paired,
    )
    assert result["status"] == "NOT_FROZEN"
    assert not (negative / "final_candidate/NEW_STABLE_BEST_FROZEN").exists()


# 48-50: winner's-curse shifts and old-P statistics are descriptive only, and
# the method-attribution boundary permits only a candidate-level claim.
def test_48_to_50_winner_curse_P_and_attribution_boundary(protocol: dict) -> None:
    universe = {"a": {"historical_validation_mean": 0.81}}
    stage1 = [{"joint_fingerprint": "a", "candidate_validation_mean": 0.805}]
    stage2 = [{"joint_fingerprint": "a", "candidate_validation_mean": 0.803}]
    result = gate.winner_curse_analysis(["a"], universe, stage1, stage2)
    row = result["candidates"][0]
    assert row["original_to_stage1_shift"] == pytest.approx(-0.005)
    assert row["original_to_stage2_shift"] == pytest.approx(-0.007)
    assert row["final_gate_use"] is False
    assert protocol["historical_P"]["historical_validation_mean"] == gate.P_OLD_MEAN
    assert protocol["historical_P"]["historical_statistics_role"] == "descriptive_only"
    boundary = protocol["method_attribution_boundary"]
    assert boundary["randomized_method_comparison"] is False
    assert boundary["candidate_level_accuracy_claim_only"] is True
    assert boundary["method_level_superiority_claim"] == "forbidden"


# 51-53: this module contains neither surrogate fitting nor candidate search;
# CUDA is mentioned only in the evaluator launch/execution boundary.
def test_51_to_53_no_surrogate_search_or_non_evaluator_GPU_path() -> None:
    source = inspect.getsource(gate)
    assert "train_direct_joint_regressor" not in source
    assert "score_candidates_with_frozen_b4" not in source
    assert "generate_all_candidates" not in source
    assert "replay_frozen_b4" not in source
    for function in (gate.synthetic_smoke, gate.execute_dry_run):
        body = inspect.getsource(function)
        assert "subprocess.Popen" not in body
        assert "cuda:" not in body.lower()


# 58: source snapshots contain actual bytes for all declared implementation
# files, plus a recomputable SHA-256 inventory.
def test_58_source_snapshot_saves_actual_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    source = repo / "pkg/source.py"
    source.write_text("VALUE = 7\n", encoding="utf-8")
    output = tmp_path / "bundle"
    monkeypatch.setattr(gate, "REPO_ROOT", repo)
    monkeypatch.setattr(gate, "IMPLEMENTATION_FILES", ("pkg/source.py",))
    monkeypatch.setattr(gate, "_local_python_modules", lambda: [])
    manifest = gate.snapshot_sources(output, config=None)
    copied = output / "source_snapshot/implementation/pkg/source.py"
    assert copied.read_bytes() == source.read_bytes()
    assert manifest["actual_bytes_saved"] is True
    assert manifest["files"]
    pairs = [(row["relative_path"], row["sha256"]) for row in manifest["files"]]
    assert manifest["snapshot_root_sha256"] == gate.stable_hash(pairs)


# 59: the formal directory manifest validates file bytes, sizes, and its root,
# then rejects mutation.
def test_59_manifest_integrity(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "evidence.txt").write_text("frozen\n", encoding="utf-8")
    written = gate.write_manifest(output)
    verified = gate.verify_manifest(output)
    assert verified["file_count"] == written["file_count"] == 1
    assert verified["all_file_hashes_valid"] is True
    (output / "evidence.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(gate.InputIdentityInvalid, match="manifest entry mismatch"):
        gate.verify_manifest(output)


# 60: any existing formal directory, archive, or SHA sidecar is immutable.
@pytest.mark.parametrize("existing", ("directory", "archive", "sidecar"))
def test_60_formal_output_refuses_overwrite(
    existing: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    posthoc = repo / "results/posthoc"
    posthoc.mkdir(parents=True)
    output = posthoc / "fresh"
    archive = Path(str(output) + ".tar.gz")
    sidecar = Path(str(archive) + ".sha256")
    targets = {"directory": output, "archive": archive, "sidecar": sidecar}
    target = targets[existing]
    if existing == "directory":
        target.mkdir()
    else:
        target.write_text("occupied\n", encoding="utf-8")
    monkeypatch.setattr(gate, "REPO_ROOT", repo)
    with pytest.raises(gate.ProtocolError, match="already exists"):
        gate.ensure_output_policy(output, archive)


# 61: dry-run validates identities and the frozen plan but creates neither the
# formal directory nor an evaluator/GPU process.
def test_61_dry_run_creates_no_formal_result_directory(
    protocol: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    formal = tmp_path / "formal-result"
    monkeypatch.setattr(gate, "load_config", lambda _path=gate.DEFAULT_CONFIG: protocol)
    monkeypatch.setattr(
        gate, "validate_authoritative_inputs",
        lambda _config: (
            {"status": "VALID", "test_accessed": False},
            {"status": "COMPATIBLE", "test_accessed": False},
        ),
    )
    monkeypatch.setattr(
        gate, "build_candidate_universe",
        lambda _config: ([], {"unique_joint_config_count_before_P_exclusion": 1121}),
    )
    monkeypatch.setattr(
        gate, "select_top6",
        lambda _universe: [{"joint_fingerprint": joint} for joint in gate.EXPECTED_TOP6],
    )
    monkeypatch.setattr(
        gate, "recover_historical_p",
        lambda _config: {"joint_fingerprint": gate.P_JOINT},
    )
    monkeypatch.setattr(
        gate.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("dry-run started an evaluator subprocess"),
    )
    assert gate.execute_dry_run(tmp_path / "synthetic.yaml", formal) == 0
    assert not formal.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN_READY"
    assert payload["formal_output_created"] is False
    assert payload["GNN_training_started"] is False
    assert payload["evaluator_subprocess_started"] is False
    assert payload["CUDA_initialized"] is False
    assert payload["test_accessed"] is False
