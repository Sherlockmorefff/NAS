from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import b4_guided_search_accuracy_gate as gate


@pytest.fixture(scope="session")
def protocol() -> dict:
    return gate.load_config()


@pytest.fixture(scope="session")
def candidates() -> list[dict]:
    rows = []
    index = 0
    for replicate in range(gate.REPLICATES):
        for round_id in range(gate.ROUNDS):
            for slot in range(gate.CANDIDATES_PER_ROUND):
                joint = f"{index:064x}"
                rows.append({
                    "replicate_id": replicate,
                    "round_id": round_id,
                    "candidate_slot": slot,
                    "pool_id": replicate,
                    "batch_id": round_id,
                    "architecture_slot": slot,
                    "config_id": f"synthetic-{replicate}-{round_id}-{slot}",
                    "joint_fingerprint": joint,
                    "joint_config_fingerprint": joint,
                    "architecture_fingerprint": f"arch-{index % 101:03d}",
                    "hp_fingerprint": f"hp-{index:064d}",
                    "operations": ["input", "gcn", "output"],
                    "edges": [[0, 1], [1, 2]],
                    "effective_layers": 1,
                    "raw_hp_vector": [0.81, 0.82, 0.83, 0.84],
                    "raw_hp": {"lr": 0.01, "dropout": 0.51, "hidden_dim": 256, "l2": 0.001},
                    "hp_regime": "hp_high",
                    "lr": 0.01,
                    "dropout": 0.51,
                    "hidden_dim": 256,
                    "l2": 0.001,
                    "gat_heads": 8,
                    "sage_aggr": "mean",
                    "gin_eps": 0.0,
                    "gat_heads_by_layer": [8],
                    "sage_aggr_by_layer": ["mean"],
                    "gin_eps_by_layer": [0.0],
                    "generation_seed": gate.CANDIDATE_MASTER_SEED + replicate,
                    "generation_draw_index": index,
                    "proposal_distribution": "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR",
                })
                index += 1
    return rows


@pytest.fixture(scope="session")
def predictions(candidates: list[dict]) -> list[dict]:
    return [{
        "replicate_id": row["replicate_id"],
        "round_id": row["round_id"],
        "candidate_slot": row["candidate_slot"],
        "joint_fingerprint": row["joint_fingerprint"],
        "architecture_fingerprint": row["architecture_fingerprint"],
        "b4_prediction": float(row["candidate_slot"]),
    } for row in candidates]


@pytest.fixture(scope="session")
def plans(candidates: list[dict], predictions: list[dict]):
    return gate.select_search_plans(candidates, predictions)


@pytest.fixture
def promoted_analysis():
    b4, random, information, stats = gate._synthetic_analysis_case(promoted=True)
    training = {"selected_unique_config_count": 256, "physical_unique_GNN_runs": 768}
    return gate.compute_search_analysis(stats, b4, random, information, training)


def test_01_frozen_b4_source_and_protocol_identity(protocol: dict) -> None:
    frozen = protocol["frozen_B4"]
    assert frozen["name"] == "B4_DIRECT_JOINT_REGRESSOR"
    assert frozen["model_seeds"] == [0, 1, 2]
    assert frozen["outer_folds"] == 5
    assert frozen["loss"] == "Huber"
    source = gate.SOFT_SNAPSHOT / "analyse/joint_configuration_metric_gate.py"
    assert gate.sha256_file(source) == "e7c883e63b7a6133f5be6d029cb8d8f2f5cf755d8366cf9cf715e0c214906a13"
    assert gate.validate_frozen_protocol(protocol)["status"] == "VALID"


def test_02_previous_labels_never_enter_b4_training(protocol: dict) -> None:
    frozen = protocol["frozen_B4"]
    assert frozen["supervision"] == "historical_336_only"
    assert frozen["previous_8_pool_labels_in_training"] == "forbidden"
    assert frozen["previous_16_pool_labels_in_training"] == "forbidden"
    assert frozen["current_labels_in_training"] == "forbidden"
    source = inspect.getsource(gate.replay_frozen_b4)
    assert "evaluation/raw_seed_results" not in source
    assert "evaluation/config_3seed_statistics" not in source


def test_03_04_exact_historical_and_previous_joints_are_excluded(candidates: list[dict]) -> None:
    base = dict(
        historical_joint=set(), previous_joint=set(),
        historical_architecture=set(), previous_architecture=set(),
    )
    historical = {candidates[0]["joint_fingerprint"]}
    with pytest.raises(gate.CandidateGenerationFailed, match="historical exact joint"):
        gate.validate_candidate_set(candidates, **{**base, "historical_joint": historical})
    previous = {candidates[1]["joint_fingerprint"]}
    with pytest.raises(gate.CandidateGenerationFailed, match="previous prospective exact joint"):
        gate.validate_candidate_set(candidates, **{**base, "previous_joint": previous})


def test_05_06_all_4096_joints_unique_and_architecture_repetition_reported(candidates: list[dict]) -> None:
    audit = gate.validate_candidate_set(
        candidates, historical_joint=set(), previous_joint=set(),
        historical_architecture={"arch-000"}, previous_architecture={"arch-001"},
    )
    assert audit["candidate_count"] == 4096
    assert audit["unique_joint_count"] == 4096
    assert audit["unique_architecture_count"] == 101
    assert audit["architecture_repetition_allowed"] is True
    assert audit["current_repeated_architecture_fraction"] > 0
    duplicate = [dict(row) for row in candidates]
    duplicate[-1]["joint_fingerprint"] = duplicate[0]["joint_fingerprint"]
    with pytest.raises(gate.CandidateGenerationFailed, match="not all unique"):
        gate.validate_candidate_set(
            duplicate, historical_joint=set(), previous_joint=set(),
            historical_architecture=set(), previous_architecture=set(),
        )


def test_07_08_candidate_generation_is_model_blind_and_high_regime(protocol: dict) -> None:
    generation = protocol["candidate_generation"]
    assert generation["proposal"] == "frozen_model_blind_architecture_decode_plus_HIGH_REGIME_PRIOR"
    assert generation["hp_raw_unit_interval_range"] == [0.80, 0.95]
    assert generation["scorer_access"] == "forbidden"
    assert generation["validation_access"] == "forbidden"
    source = inspect.getsource(gate.generate_all_candidates)
    assert "uniform(0.80, 0.95" in source
    assert "score" not in inspect.signature(gate.generate_all_candidates).parameters


def test_09_10_11_frozen_search_grid(protocol: dict, candidates: list[dict]) -> None:
    generation = protocol["candidate_generation"]
    assert generation["replicates"] == 8
    assert generation["rounds_per_replicate"] == 4
    assert generation["candidates_per_round"] == 128
    counts = {}
    for row in candidates:
        key = (row["replicate_id"], row["round_id"])
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 32
    assert set(counts.values()) == {128}


def test_12_13_14_all_candidates_scores_and_random_plan_precede_gnn() -> None:
    source = inspect.getsource(gate.execute_formal)
    generated = source.index("generate_all_candidates")
    scored = source.index("score_all_candidates_with_b4")
    selected = source.index("select_search_plans")
    frozen = source.index("freeze_pre_search")
    tasks = source.index("build_selected_union_tasks")
    gnn = source.index("run_frozen_gnn_evaluation")
    assert generated < scored < selected < frozen < tasks < gnn


def test_15_b4_top4_and_17_lexicographic_tie_break(candidates: list[dict], predictions: list[dict]) -> None:
    b4, _random, _selected, _information = gate.select_search_plans(candidates, predictions)
    first = b4["rounds"][0]
    assert first["selected_candidate_slots"] == [127, 126, 125, 124]
    tied = [{**row, "b4_prediction": 1.0} for row in predictions]
    tied_b4, _random2, _selected2, _information2 = gate.select_search_plans(candidates, tied)
    expected = sorted(row["joint_fingerprint"] for row in candidates[:128])[:4]
    assert tied_b4["rounds"][0]["selected_joint_fingerprints"] == expected


def test_16_random_uniform_without_replacement_is_seeded(candidates: list[dict], predictions: list[dict]) -> None:
    first = gate.select_search_plans(candidates, predictions)[1]
    second = gate.select_search_plans(candidates, predictions)[1]
    assert first == second
    assert first["master_seed"] == 202608290002
    for row in first["rounds"]:
        assert len(row["selected_joint_fingerprints"]) == 4
        assert len(set(row["selected_joint_fingerprints"])) == 4
        assert row["sampling"] == "uniform_without_replacement"


def test_18_pre_search_root_recomputes_and_19_freeze_is_immutable(
    tmp_path: Path, candidates: list[dict], predictions: list[dict], plans,
) -> None:
    b4, random, _selected, information = plans
    freeze = gate.freeze_pre_search(
        tmp_path / "pre_search", candidates, predictions, b4, random,
        {"status": "IDENTICAL"}, {"status": "IDENTICAL"}, information,
    )
    verified = gate.verify_pre_search_freeze(tmp_path / "pre_search")
    assert verified["root_sha256"] == freeze["root_sha256"]
    assert [row["relative_path"] for row in freeze["files"]] == sorted(gate.PRE_SEARCH_FILES)
    path = tmp_path / "pre_search/b4_predictions.csv"
    path.write_text(path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    with pytest.raises(gate.PreSearchFreezeError):
        gate.verify_pre_search_freeze(tmp_path / "pre_search")
    before = [{"joint": "a"}]
    after = [{"joint": "b"}]
    with pytest.raises(gate.PreSearchFreezeError):
        gate.assert_candidate_manifest_immutable(before, after)


def test_20_overlap_trains_once_21_method_budget_remains_128(
    candidates: list[dict], plans,
) -> None:
    b4, random, selected, information = plans
    configs, tasks = gate.build_selected_union_tasks(candidates, b4, random, pre_search_verified=True)
    union = selected["B4_GUIDED"] | selected["MODEL_BLIND_RANDOM_HIGH_REGIME"]
    assert len(configs) == len(union)
    assert len(tasks) == 3 * len(union)
    assert len({(row["joint_fingerprint"], row["training_seed"]) for row in tasks}) == len(tasks)
    assert information["B4_selected_slots"] == 128
    assert information["random_selected_slots"] == 128
    assert information["method_equivalent_GNN_runs"] == 768


def test_22_training_seeds_fixed_23_same_config_results_shared(candidates: list[dict], plans) -> None:
    b4, random, _selected, _information = plans
    configs, tasks = gate.build_selected_union_tasks(candidates, b4, random, pre_search_verified=True)
    assert {row["training_seed"] for row in tasks} == {0, 1, 2}
    overlap = next((row for row in configs if row["selected_by_B4"] and row["selected_by_random"]), None)
    if overlap is not None:
        matching = [row for row in tasks if row["joint_fingerprint"] == overlap["joint_fingerprint"]]
        assert len(matching) == 3
        assert {row["training_seed"] for row in matching} == {0, 1, 2}


def test_24_test_firewall_and_validation_only_view(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(gate.ProtocolError, match="test-named field"):
        gate.reject_test_named_fields({"test_accuracy": 0.8})
    gate.reject_test_named_fields({"track_test": False, "test_accessed": False}, allow_false_controls=True)

    class FakeData:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def keys(self):
            return list(self.__dict__)

    fake_module = types.ModuleType("torch_geometric.data")
    fake_module.Data = FakeData
    fake_parent = types.ModuleType("torch_geometric")
    fake_parent.data = fake_module
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_parent)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", fake_module)
    source = types.SimpleNamespace(
        x=1, edge_index=2, y=3, train_mask=4, val_mask=5,
        test_mask=object(),
    )
    view = gate.validation_only_data_view(source)
    assert set(view.keys()) == {"x", "edge_index", "y", "train_mask", "val_mask"}
    assert not hasattr(view, "test_mask")
    bundle = types.SimpleNamespace(
        data=source, manifest={"identity": "synthetic"},
        num_features=7, num_classes=3,
    )
    loaded, features, classes, context, audit = gate.load_validation_only_dataset(
        object(), lambda _request: bundle, lambda manifest: manifest,
    )
    assert set(loaded.keys()) == {"x", "edge_index", "y", "train_mask", "val_mask"}
    assert (features, classes, context) == (7, 3, {"identity": "synthetic"})
    assert audit["raw_loader_bundle_exposed"] is False
    assert audit["test_named_fields_exposed"] == []


def test_25_track_test_false_26_retry_keeps_task_identity(protocol: dict) -> None:
    evaluation = protocol["evaluation"]
    assert evaluation["track_test"] is False
    assert evaluation["maximum_system_retries"] == 2
    assert evaluation["candidate_replacement_after_failure"] == "forbidden"
    source = inspect.getsource(gate.execute_evaluator_shard)
    assert "while True" in source
    assert "attempt <= maximum_retries" in source
    assert "task[\"seed\"]" in source
    assert "task[\"operations\"]" in source


def test_27_incomplete_config_fails_closed_28_mean_and_sample_sd() -> None:
    candidate = {
        "joint_fingerprint": "j", "architecture_fingerprint": "a",
        "selected_by_B4": True, "selected_by_random": True,
    }
    prediction = [{"joint_fingerprint": "j", "b4_prediction": 0.9}]
    complete_rows = [{
        "joint_fingerprint": "j", "training_seed": seed,
        "validation_accuracy": value, "terminal_status": "success", "valid": True,
    } for seed, value in enumerate((0.80, 0.81, 0.82))]
    stats, complete = gate.aggregate_config_statistics([candidate], complete_rows, prediction)
    assert complete is True
    assert stats[0]["validation_mean"] == pytest.approx(0.81)
    assert stats[0]["validation_sd"] == pytest.approx(0.01)
    incomplete, complete_flag = gate.aggregate_config_statistics(
        [candidate], complete_rows[:-1], prediction, allow_incomplete=True,
    )
    assert complete_flag is False
    assert incomplete[0]["validation_mean"] is None
    with pytest.raises(gate.EvaluationIncomplete):
        gate.aggregate_config_statistics([candidate], complete_rows[:-1], prediction)


def test_29_30_31_32_round_final_anytime_and_auc(promoted_analysis) -> None:
    assert len(promoted_analysis["per_round"]) == 8 * 4 * 2
    assert len(promoted_analysis["per_replicate"]) == 8
    assert len(promoted_analysis["anytime"]) == 8 * 2 * 4
    for replicate in promoted_analysis["per_replicate"]:
        assert replicate["best_B4"] > replicate["best_random"]
        assert replicate["B4_anytime_AUC"] == pytest.approx(np.mean([
            row["best_so_far"] for row in promoted_analysis["anytime"]
            if row["replicate_id"] == replicate["replicate_id"] and row["method"] == "B4_GUIDED"
        ]))
    for method in ("B4_GUIDED", "MODEL_BLIND_RANDOM_HIGH_REGIME"):
        for replicate in range(8):
            curve = [
                row["best_so_far"] for row in promoted_analysis["anytime"]
                if row["replicate_id"] == replicate and row["method"] == method
            ]
            assert curve == sorted(curve)


def test_33_34_bootstrap_unit_and_seed_are_frozen() -> None:
    first = gate.bootstrap_paired_mean([0.01] * 8)
    second = gate.bootstrap_paired_mean([0.01] * 8)
    assert first == second
    assert first["unit"] == "search_replicate"
    assert first["replicate_count"] == 8
    assert first["bootstrap_replicates"] == 10000
    assert first["seed"] == 0
    assert first["ci95_lower"] == pytest.approx(0.01)


def test_35_s1_to_s5_classification(promoted_analysis) -> None:
    decision = promoted_analysis["decision"]
    assert decision["classification"] == "B4_SEARCH_PROMOTED"
    assert all(value["passed"] for value in decision["gates"].values())
    b4, random, information, stats = gate._synthetic_analysis_case(promoted=False)
    not_confirmed = gate.compute_search_analysis(
        stats, b4, random, information,
        {"selected_unique_config_count": 256, "physical_unique_GNN_runs": 768},
    )
    assert not_confirmed["decision"]["classification"] == "B4_SEARCH_NOT_CONFIRMED"
    assert not all(value["passed"] for value in not_confirmed["decision"]["gates"].values())


def test_36_historical_p_is_not_in_primary_gate(promoted_analysis) -> None:
    source = inspect.signature(gate.classify_primary_gate)
    assert "historical_P" not in source.parameters
    assert promoted_analysis["decision"]["historical_P_used_in_primary_gate"] is False
    assert promoted_analysis["final_comparison"]["historical_P"]["primary_gate_use"] is False


def test_37_promoted_only_winner_marker_38_top3_dedupe_and_tie_break(
    tmp_path: Path, promoted_analysis,
) -> None:
    promoted_root = tmp_path / "promoted"
    result = gate.write_next_stage(
        promoted_root, "B4_SEARCH_PROMOTED",
        promoted_analysis["b4_winners"], promoted_analysis["random_winners"],
    )
    assert result["promoted"] is True
    assert (promoted_root / "next_stage/NEXT_STAGE_WINNERS_FROZEN").is_file()
    assert len(json.loads((promoted_root / "next_stage/b4_top3.json").read_text())["winners"]) == 3
    not_root = tmp_path / "not"
    gate.write_next_stage(
        not_root, "B4_SEARCH_NOT_CONFIRMED",
        promoted_analysis["b4_winners"], promoted_analysis["random_winners"],
    )
    assert not (not_root / "next_stage/NEXT_STAGE_WINNERS_FROZEN").exists()
    tied = [
        {"replicate_id": 0, "joint_fingerprint": "b", "architecture_fingerprint": "a", "validation_mean": 0.8, "validation_sd": 0.1, "b4_prediction": 0.8},
        {"replicate_id": 1, "joint_fingerprint": "a", "architecture_fingerprint": "a", "validation_mean": 0.8, "validation_sd": 0.1, "b4_prediction": 0.8},
        {"replicate_id": 2, "joint_fingerprint": "a", "architecture_fingerprint": "a", "validation_mean": 0.8, "validation_sd": 0.1, "b4_prediction": 0.8},
        {"replicate_id": 3, "joint_fingerprint": "c", "architecture_fingerprint": "a", "validation_mean": 0.7, "validation_sd": 0.1, "b4_prediction": 0.7},
        {"replicate_id": 4, "joint_fingerprint": "d", "architecture_fingerprint": "a", "validation_mean": 0.6, "validation_sd": 0.1, "b4_prediction": 0.6},
    ]
    assert [row["joint_fingerprint"] for row in gate.select_top3_winners(tied)] == ["a", "b", "c"]


def test_39_fresh_seeds_are_not_run_in_this_stage(protocol: dict) -> None:
    assert protocol["next_stage"]["fresh_seed_training_in_this_run"] == "forbidden"
    assert protocol["evaluation"]["training_seeds"] == [0, 1, 2]
    assert set(gate.TRAINING_SEEDS) == {0, 1, 2}


def test_40_source_snapshot_contract_is_complete() -> None:
    paths = gate._source_identity_paths()
    required = {
        "configs/b4_guided_search_accuracy_gate_v1.yaml",
        "analyse/b4_guided_search_accuracy_gate.py",
        "scripts/run_b4_guided_search_accuracy_gate.sh",
        "tests/test_b4_guided_search_accuracy_gate.py",
        "frozen/analyse/joint_configuration_metric_gate.py",
        "frozen/analyse/high_regime_prior_model_blind_validation.py",
        "frozen/analyse/run_high_regime_prior_model_blind_validation_labels.py",
        "frozen/analyse/original_head_deterministic_batch_pilot.py",
        "frozen/analyse/structure_native_surrogate_pilot.py",
        "dataset_utils.py",
    }
    assert required <= set(paths)
    assert all(path.is_file() for path in paths.values())
    source = inspect.getsource(gate.snapshot_sources)
    assert "sys.modules.items()" in source
    assert "shutil.copy2" in source
    git_source = inspect.getsource(gate._git_diff_patch)
    assert "--no-index" in git_source
    assert "IMPLEMENTATION_FILES" in git_source


def test_41_final_report_has_all_22_sections(promoted_analysis) -> None:
    artifacts = {
        **promoted_analysis,
        "input_integrity": {"status": "VALID", "test_accessed": False},
        "replay": {"status": "IDENTICAL", "historical_OOF_count": 336, "previous_16_pool_prediction_count": 64},
        "generation": {"coverage": {"candidate_count": 4096, "unique_joint_count": 4096}},
        "information": {"selected_union_unique_configurations": 256},
        "training_audit": {"evaluation_complete": True, "task_count": 768, "completed_gnn_runs": 768, "failed_gnn_runs": 0, "loader_firewall_passed": True},
        "pre_search_root": "abc",
        "next_stage": {"promoted": True, "fresh_seed_training_started": False},
    }
    report = gate.render_final_report(artifacts)
    gate.assert_complete_final_report(report)
    assert sum(f"## {section}" in report for section in gate.REPORT_SECTIONS) == 22


def test_42_dry_run_does_not_create_formal_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "formal-result"
    assert gate.execute_dry_run(gate.DEFAULT_CONFIG, output) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["formal_output_created"] is False
    assert payload["formal_candidate_generation_started"] is False
    assert payload["formal_search_plan_frozen"] is False
    assert payload["GNN_training_started"] is False
    assert not output.exists()


def test_43_formal_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(gate.ProtocolError, match="already exists"):
        gate.ensure_output_policy(output, Path(str(output) + ".tar.gz"))


def test_44_failure_bundle_can_be_generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "failed-bundle"
    monkeypatch.setattr(
        gate, "_frozen_dependency_entries",
        lambda: (_ for _ in ()).throw(gate.InputIdentityInvalid("synthetic missing dependency")),
    )
    gate.write_failure_bundle(output, "synthetic failure")
    assert (output / "FAILED").is_file()
    assert (output / "error.json").is_file()
    assert not (output / "COMPLETE").exists()
    assert json.loads((output / "decision.json").read_text())["classification"] == "RUN_FAILED"
    archive = Path(str(output) + ".tar.gz")
    assert archive.is_file()
    assert Path(str(archive) + ".sha256").is_file()
    snapshot_manifest = json.loads((output / "source_snapshot/manifest.json").read_text())
    assert snapshot_manifest["best_effort"] is True
    assert snapshot_manifest["snapshot_error_count"] == 1


def test_synthetic_smoke_covers_terminal_logic() -> None:
    result = gate.synthetic_smoke()
    assert result["passed"] is True
    assert result["formal_candidate_generation_started"] is False
    assert result["formal_search_plan_frozen"] is False
    assert result["GNN_training_started"] is False
    assert result["test_accessed"] is False
