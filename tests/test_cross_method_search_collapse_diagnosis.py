from __future__ import annotations

from collections import Counter
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

from analyse import cross_method_search_collapse_diagnosis as diagnosis


@pytest.fixture(scope="session")
def config() -> dict:
    value = diagnosis.load_config()
    diagnosis.validate_protocol(value)
    return value


@pytest.fixture(scope="session")
def frozen(config: dict):
    return diagnosis.load_frozen_inputs(config)


@pytest.fixture(scope="session")
def selection_stats(frozen):
    methods, _integrity = frozen
    return diagnosis.selection_diversity(methods)


@pytest.fixture(scope="session")
def counterfactual(frozen):
    methods, _integrity = frozen
    return diagnosis.counterfactual_diagnosis(methods)


def _records(method: str = "X") -> list[dict]:
    rows = []
    architectures = {
        0: ["a", "a", "b", "c"],
        1: ["a", "d", "e", "f"],
        2: ["b", "g", "h", "i"],
        3: ["a", "j", "k", "l"],
    }
    for replicate in range(8):
        for round_id in range(4):
            for slot, architecture in enumerate(architectures[round_id]):
                rows.append({
                    "universe": "U", "method": method, "replicate_id": replicate,
                    "round_id": round_id, "selected_position": slot,
                    "joint_fingerprint": f"{method}-{replicate}-{round_id}-{slot}",
                    "architecture_fingerprint": architecture,
                    "validation_mean": 0.70 + 0.01 * round_id + 0.001 * slot,
                })
    return rows


def _synthetic_policy_methods() -> dict:
    candidates = []
    for replicate in range(8):
        for round_id in range(4):
            for slot in range(128):
                candidates.append({
                    "replicate_id": replicate, "round_id": round_id,
                    "candidate_slot": slot, "joint_fingerprint": f"j-{replicate}-{round_id}-{slot:03d}",
                    "architecture_fingerprint": f"arch-{slot % 80:03d}",
                    "score": float(128 - slot),
                })
    return {"X": {"universe": "U", "score_field": "score", "candidates": candidates}}


def test_01_02_no_gnn_or_evaluator_path() -> None:
    source = Path(diagnosis.__file__).read_text(encoding="utf-8")
    assert "train_and_eval_arch" not in source
    assert "execute_evaluator" not in source
    assert "subprocess.Popen" not in source
    assert "new_GNN_runs\": 0" in source


def test_03_no_cuda_initialization() -> None:
    source = Path(diagnosis.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert ".cuda(" not in source and "torch.cuda" not in source
    assert diagnosis.synthetic_smoke()["CUDA_initialized"] is False


def test_04_no_test_access(config: dict) -> None:
    assert config["test_access"] == "forbidden"
    smoke = diagnosis.synthetic_smoke()
    assert smoke["test_accessed"] is False


def test_05_06_input_classifications(frozen) -> None:
    _methods, integrity = frozen
    assert integrity["b4"]["classification"] == "B4_SEARCH_NOT_CONFIRMED"
    assert integrity["metric"]["classification"] == "METRIC_SEARCH_NOT_CONFIRMED"
    assert integrity["test_accessed"] is False


def test_07_manifest_validation(frozen) -> None:
    _methods, integrity = frozen
    assert integrity["b4"]["manifest"]["all_file_hashes_valid"] is True
    assert integrity["metric"]["manifest"]["all_file_hashes_valid"] is True
    assert integrity["b4"]["pre_search"]["root_sha256"]
    assert integrity["metric"]["pre_search"]["root_sha256"]


def test_08_09_paired_randoms_are_experiment_local() -> None:
    assert diagnosis.PAIR_RANDOM == {"B4": "B4_RANDOM", "M1": "M12_RANDOM", "M2": "M12_RANDOM"}


def test_10_candidate_universes_never_mixed(frozen) -> None:
    methods, integrity = frozen
    assert methods["B4"]["universe"] == methods["B4_RANDOM"]["universe"] == "B4"
    assert methods["M1"]["universe"] == methods["M2"]["universe"] == methods["M12_RANDOM"]["universe"] == "M12"
    assert integrity["candidate_universes_kept_separate"] is True


def test_11_to_14_concentration_math() -> None:
    stats = diagnosis.concentration(["a", "a", "b", "c"])
    expected_entropy = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
    assert stats["shannon_entropy"] == pytest.approx(expected_entropy)
    assert stats["effective_architectures"] == pytest.approx(math.exp(expected_entropy))
    assert stats["HHI"] == pytest.approx(0.375)
    assert stats["top2_share"] == pytest.approx(0.75)


def test_15_16_per_replicate_and_round_stats(selection_stats) -> None:
    _overall, per_replicate, per_round, _lookup = selection_stats
    assert len(per_replicate) == 5 * 8
    assert len(per_round) == 5 * 8 * 4
    assert all(row["selected_slot_count"] == 16 for row in per_replicate)
    assert all(row["selected_slot_count"] == 4 for row in per_round)


def test_17_repeat_excludes_same_round_duplicate() -> None:
    result = diagnosis.repeat_diagnosis("X", _records())
    first_round_a = [
        row for row in result["slot_rows"]
        if row["replicate_id"] == 0 and row["round_id"] == 0 and row["architecture_fingerprint"] == "a"
    ]
    assert len(first_round_a) == 2
    assert all(row["within_round_duplicate"] for row in first_round_a)
    assert all(not row["repeat_architecture_slot"] for row in first_round_a)


def test_18_within_round_duplicate_calculation(selection_stats) -> None:
    _overall, _replicate, rounds, _lookup = selection_stats
    assert all(row["within_round_duplicate_slots"] == 4 - row["unique_architecture_count"] for row in rounds)


def test_19_to_21_first_later_and_refinement_gain() -> None:
    result = diagnosis.repeat_diagnosis("X", _records())
    row = next(row for row in result["refinement_rows"] if row["replicate_id"] == 0 and row["architecture_fingerprint"] == "a")
    assert row["FIRST_BEST"] == pytest.approx(0.701)
    assert row["LATER_REPEAT_BEST"] == pytest.approx(0.73)
    assert row["REFINEMENT_GAIN"] == pytest.approx(0.029)


def test_22_repeat_winner_classification() -> None:
    result = diagnosis.repeat_diagnosis("X", _records())
    assert all(row["final_winner_origin"] in {"FIRST_APPEARANCE", "LATER_REPEAT"} for row in result["winner_rows"])
    assert all(row["REPEAT_WINNER_GAIN"] >= 0 for row in result["winner_rows"])


def test_23_candidate_pool_richness(frozen) -> None:
    methods, _integrity = frozen
    rows, summary = diagnosis.candidate_pool_diversity(methods)
    assert len(rows) == 64
    assert summary["POOL_RICH"] == (summary["B4"]["median"] >= 64 and summary["M12"]["median"] >= 64)


def test_24_ranking_top4_8_16_32(frozen) -> None:
    methods, _integrity = frozen
    rows = diagnosis.ranking_concentration(methods)
    assert len(rows) == 3 * 32 * 4
    assert {row["top_k"] for row in rows} == {4, 8, 16, 32}


def test_25_to_27_collapse_rules() -> None:
    overall = {
        "B4": {"effective_architectures": 10.0, "top2_share": 0.60},
        "B4_RANDOM": {"effective_architectures": 40.0, "top2_share": 0.10},
        "M1": {"effective_architectures": 10.0, "top2_share": 0.60},
        "M2": {"effective_architectures": 10.0, "top2_share": 0.60},
        "M12_RANDOM": {"effective_architectures": 40.0, "top2_share": 0.10},
    }
    per = []
    for method, unique in (("B4", 5), ("B4_RANDOM", 12), ("M1", 5), ("M2", 5), ("M12_RANDOM", 12)):
        per.extend({"method": method, "replicate_id": r, "unique_architecture_count": unique} for r in range(8))
    result = diagnosis.collapse_gates(overall, per)
    assert all(result["methods"][method]["indicators"]["C1"]["passed"] for method in diagnosis.METHODS)
    assert all(result["methods"][method]["METHOD_COLLAPSE_STRONG"] for method in diagnosis.METHODS)
    assert result["SYSTEMATIC_COLLAPSE"] is True


def test_28_exploitation_saturation(frozen) -> None:
    methods, _integrity = frozen
    result = diagnosis.anytime_saturation(methods)
    expected = sum(result["methods"][method]["EARLY_SEARCH_SIGNAL"] for method in diagnosis.METHODS) >= 2
    assert result["SATURATION_PATTERN"] == expected


def test_29_30_p1_p2_round_unique() -> None:
    methods = _synthetic_policy_methods()
    for policy in ("P1", "P2"):
        rows = diagnosis.build_counterfactual_policy("X", methods, policy)
        assert len(rows) == 128
        for replicate in range(8):
            for round_id in range(4):
                architectures = {
                    row["architecture_fingerprint"] for row in rows
                    if row["replicate_id"] == replicate and row["round_id"] == round_id
                }
                assert len(architectures) == 4


def test_31_p2_maximum_two_slots_per_architecture() -> None:
    rows = diagnosis.build_counterfactual_policy("X", _synthetic_policy_methods(), "P2")
    for replicate in range(8):
        counts = Counter(row["architecture_fingerprint"] for row in rows if row["replicate_id"] == replicate)
        assert max(counts.values()) <= 2


def test_32_policies_never_use_validation() -> None:
    source = inspect.getsource(diagnosis.build_counterfactual_policy)
    assert "validation_mean" not in source
    rows = diagnosis.build_counterfactual_policy("X", _synthetic_policy_methods(), "P1")
    assert all(row["validation_used_for_selection"] is False for row in rows)


def test_33_deterministic_tie_break() -> None:
    methods = _synthetic_policy_methods()
    for row in methods["X"]["candidates"]:
        row["score"] = 1.0
    first = diagnosis.build_counterfactual_policy("X", methods, "P1")
    second = diagnosis.build_counterfactual_policy("X", methods, "P1")
    assert first == second
    first_round = [row["joint_fingerprint"] for row in first if row["replicate_id"] == 0 and row["round_id"] == 0]
    assert first_round == sorted(first_round)


def test_34_rank_sacrifice(counterfactual) -> None:
    for method in diagnosis.METHODS:
        assert counterfactual["rank_sacrifice"][method]["ORIGINAL"]["mean_rank"] == pytest.approx(2.5)
        assert counterfactual["rank_sacrifice"][method]["P1"]["max_rank"] >= 4


def test_35_36_score_sacrifice_and_degenerate_pool() -> None:
    methods = _synthetic_policy_methods()
    for row in methods["X"]["candidates"]:
        row["score"] = 1.0
    p1 = diagnosis.build_counterfactual_policy("X", methods, "P1")
    p2 = diagnosis.build_counterfactual_policy("X", methods, "P2")
    original = []
    for ranking in diagnosis.ranked_candidates("X", methods).values():
        original.extend({**row, "policy": "ORIGINAL", "frozen_score": row["score"]} for row in ranking[:4])
    result = diagnosis._score_sacrifice("X", methods, {"ORIGINAL": original, "P1": p1, "P2": p2})
    assert result["summary"]["P1"]["mean_z_score_sacrifice"] == 0
    assert result["summary"]["P2"]["mean_z_score_sacrifice"] == 0
    assert result["summary"]["P1"]["degenerate_score_pool_count"] == 32


def test_37_38_policy_feasibility_and_preference(counterfactual) -> None:
    for method in diagnosis.METHODS:
        for policy in ("P1", "P2"):
            gates = counterfactual["preferred_policy"][method][policy]["gates"]
            assert set(gates) == {"F1", "F2", "F3", "F4", "F5"}
            assert counterfactual["preferred_policy"][method][policy]["feasible"] == all(gates.values())
        expected = "P2" if counterfactual["preferred_policy"][method]["P2"]["feasible"] else "P1" if counterfactual["preferred_policy"][method]["P1"]["feasible"] else "NONE"
        assert counterfactual["preferred_policy"][method]["PREFERRED_POLICY"] == expected


def test_39_40_scorer_selection_and_tie_break() -> None:
    counter = {"preferred_policy": {m: {"PREFERRED_POLICY": "P1"} for m in diagnosis.METHODS}}
    anytime = {"methods": {
        "B4": {"ANYTIME_AUC_GAIN": 0.01, "FINAL_BEST_GAIN": 0.0},
        "M1": {"ANYTIME_AUC_GAIN": 0.02, "FINAL_BEST_GAIN": 0.0},
        "M2": {"ANYTIME_AUC_GAIN": 0.015, "FINAL_BEST_GAIN": 0.1},
    }}
    assert diagnosis.select_next_scorer(anytime, counter) == "M1"
    for method in diagnosis.METHODS:
        anytime["methods"][method] = {"ANYTIME_AUC_GAIN": 0.01, "FINAL_BEST_GAIN": 0.02}
    assert diagnosis.select_next_scorer(anytime, counter) == "B4"


def test_41_existing_label_coverage_does_not_impute(counterfactual) -> None:
    assert all(
        counterfactual["coverage"][method][policy]["unobserved_labels_imputed"] == 0
        for method in diagnosis.METHODS for policy in ("P1", "P2")
    )


def test_42_observed_subset_cannot_affect_classification() -> None:
    assert "observed_subset" not in inspect.signature(diagnosis.authorization_decision).parameters
    assert "coverage" not in inspect.signature(diagnosis.authorization_decision).parameters


def test_43_44_authorization_and_exhaustive_classification() -> None:
    pool = {"POOL_RICH": True}
    collapse = {"SYSTEMATIC_COLLAPSE": True}
    anytime = {"SATURATION_PATTERN": True, "methods": {
        m: {"ANYTIME_AUC_GAIN": 0.01, "FINAL_BEST_GAIN": 0.0} for m in diagnosis.METHODS
    }}
    counter = {"preferred_policy": {m: {"PREFERRED_POLICY": "P1"} for m in diagnosis.METHODS}}
    result = diagnosis.authorization_decision(pool, collapse, anytime, counter)
    assert result["classification"] == "DIVERSITY_TEST_AUTHORIZED"
    collapse["SYSTEMATIC_COLLAPSE"] = False
    assert diagnosis.authorization_decision(pool, collapse, anytime, counter)["classification"] == "NO_SYSTEMATIC_SEARCH_COLLAPSE"
    collapse["SYSTEMATIC_COLLAPSE"] = True; anytime["SATURATION_PATTERN"] = False
    assert diagnosis.authorization_decision(pool, collapse, anytime, counter)["classification"] == "COLLAPSE_PRESENT_BUT_DIVERSITY_TEST_NOT_JUSTIFIED"
    assert {
        "DIAGNOSIS_INPUT_INVALID", "NO_SYSTEMATIC_SEARCH_COLLAPSE",
        "COLLAPSE_PRESENT_BUT_DIVERSITY_TEST_NOT_JUSTIFIED",
        "DIVERSITY_TEST_AUTHORIZED", "RUN_FAILED",
    } == diagnosis.ALLOWED_CLASSIFICATIONS


def test_45_46_authorized_spec_but_no_next_gnn(tmp_path: Path) -> None:
    decision = {
        "classification": "DIVERSITY_TEST_AUTHORIZED",
        "SCORER_FOR_NEXT_DIVERSITY_TEST": "M1", "SELECTED_DIVERSITY_POLICY": "P2",
    }
    diagnosis.write_next_experiment(tmp_path, decision)
    spec = json.loads((tmp_path / "next_experiment/prospective_diversity_test_spec.json").read_text())
    assert spec["methods"] == [
        "PURE_FROZEN_SCORER_TOP4", "FROZEN_SCORER_PLUS_ARCH_DIVERSITY",
        "MODEL_BLIND_RANDOM_HIGH_REGIME",
    ]
    assert spec["GNN_training_started"] is False and spec["new_GNN_runs"] == 0


def test_47_source_snapshot(tmp_path: Path) -> None:
    manifest = diagnosis.snapshot_sources(tmp_path)
    assert manifest["source_bytes_copied"] is True
    assert len(manifest["files"]) == 4
    assert all((tmp_path / "source_snapshot" / row["relative_path"]).is_file() for row in manifest["files"])


def test_48_manifest_integrity(tmp_path: Path) -> None:
    diagnosis.dump_text(tmp_path / "a.txt", "a\n")
    manifest = diagnosis.write_manifest(tmp_path)
    assert manifest["manifest_root_sha256"] == diagnosis.stable_hash([
        (row["relative_path"], row["sha256"]) for row in manifest["files"]
    ])


def test_49_rerun_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(diagnosis.DiagnosisError, match="already exists"):
        diagnosis.ensure_output_absent(output)


def test_synthetic_smoke() -> None:
    result = diagnosis.synthetic_smoke()
    assert result["passed"] is True
    assert result["new_GNN_runs"] == 0 and result["GPU_used"] is False
