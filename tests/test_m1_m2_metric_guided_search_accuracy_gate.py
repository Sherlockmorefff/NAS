from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import m1_m2_metric_guided_search_accuracy_gate as gate


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
                    "replicate_id": replicate, "round_id": round_id,
                    "candidate_slot": slot, "pool_id": replicate, "batch_id": round_id,
                    "architecture_slot": slot, "config_id": f"synthetic-{replicate}-{round_id}-{slot}",
                    "joint_fingerprint": joint, "joint_config_fingerprint": joint,
                    "architecture_fingerprint": f"arch-{index % 101:03d}",
                    "hp_fingerprint": f"hp-{index:064d}",
                    "operations": ["input", "gcn", "output"], "edges": [[0, 1], [1, 2]],
                    "effective_layers": 1, "raw_hp_vector": [0.81, 0.82, 0.83, 0.84],
                    "raw_hp": {"lr": 0.01, "dropout": 0.51, "hidden_dim": 256, "l2": 0.001},
                    "hp_regime": "hp_high", "lr": 0.01, "dropout": 0.51,
                    "hidden_dim": 256, "l2": 0.001, "gat_heads": 8,
                    "sage_aggr": "mean", "gin_eps": 0.0, "gat_heads_by_layer": [8],
                    "sage_aggr_by_layer": ["mean"], "gin_eps_by_layer": [0.0],
                    "generation_seed": gate.CANDIDATE_MASTER_SEED + replicate,
                    "generation_draw_index": index,
                    "proposal_distribution": "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR",
                })
                index += 1
    return rows


@pytest.fixture(scope="session")
def predictions(candidates: list[dict]) -> list[dict]:
    return [{
        "replicate_id": row["replicate_id"], "round_id": row["round_id"],
        "candidate_slot": row["candidate_slot"], "joint_fingerprint": row["joint_fingerprint"],
        "architecture_fingerprint": row["architecture_fingerprint"],
        "m1_prediction": float(row["candidate_slot"]),
        "m2_prediction": float(-row["candidate_slot"]),
    } for row in candidates]


@pytest.fixture(scope="session")
def plans(candidates: list[dict], predictions: list[dict]):
    return gate.select_search_plans(candidates, predictions)


@pytest.fixture(scope="session")
def promoted_analysis():
    m1, m2, random_plan, information, stats = gate._synthetic_metric_case("both_gp")
    training = {"selected_unique_config_count": len(stats), "physical_unique_GNN_runs": 3 * len(stats)}
    return gate.compute_search_analysis(
        stats, m1, m2, random_plan, information, training,
        {"selection_modified": False, "methods": {}},
    )


def test_01_historical_scorer_identity_and_replay_contract(protocol: dict) -> None:
    frozen = protocol["frozen_metric"]
    assert frozen["supervision"] == "historical_336_only"
    assert frozen["metric_seeds"] == [0, 1, 2]
    assert (frozen["outer_folds"], frozen["encoder_count"]) == (5, 15)
    assert frozen["required_replay"] == [
        "all_15_metric_best_epochs", "M1_historical_OOF_predictions",
        "M2_historical_OOF_predictions", "pooled_metric_Gap_at_5",
        "all_15_distance_gap_Spearman", "all_336_top5_neighbor_fingerprint_sequences",
    ]
    assert gate.sha256_file(gate.SOFT_SNAPSHOT / "analyse/joint_configuration_metric_gate.py") == \
        "e7c883e63b7a6133f5be6d029cb8d8f2f5cf755d8366cf9cf715e0c214906a13"
    assert gate.validate_frozen_protocol(protocol)["status"] == "VALID"


def test_02_previous_and_current_labels_never_train_metric(protocol: dict) -> None:
    frozen = protocol["frozen_metric"]
    assert frozen["previous_labels_in_training"] == "forbidden"
    assert frozen["B4_search_labels_in_training"] == "forbidden"
    assert frozen["current_labels_in_training"] == "forbidden"
    source = inspect.getsource(gate.replay_frozen_metric)
    assert "evaluation/raw_seed_results" not in source
    assert "evaluation/config_3seed_statistics" not in source


def test_03_to_06_all_exact_joint_exclusions(candidates: list[dict]) -> None:
    base = dict(
        historical_joint=set(), previous_joint=set(), b4_search_joint=set(),
        historical_architecture=set(), previous_architecture=set(),
    )
    for key, message, index in (
        ("historical_joint", "historical exact joint", 0),
        ("previous_joint", "previous prospective exact joint", 1),
        ("b4_search_joint", "B4-search evaluated exact joint", 2),
    ):
        with pytest.raises(gate.CandidateGenerationFailed, match=message):
            gate.validate_candidate_set(
                candidates, **{**base, key: {candidates[index]["joint_fingerprint"]}},
            )


def test_07_08_all_4096_unique_and_architecture_repetition_audited(candidates: list[dict]) -> None:
    audit = gate.validate_candidate_set(
        candidates, historical_joint=set(), previous_joint=set(), b4_search_joint=set(),
        historical_architecture={"arch-000"}, previous_architecture={"arch-001"},
    )
    assert (audit["candidate_count"], audit["unique_joint_count"]) == (4096, 4096)
    assert audit["unique_architecture_count"] == 101
    assert audit["architecture_repetition_allowed"] is True
    assert audit["current_repeated_architecture_fraction"] > 0
    assert len(audit["architecture_frequency_distribution"]) == 101
    duplicate = [dict(row) for row in candidates]
    duplicate[-1]["joint_fingerprint"] = duplicate[0]["joint_fingerprint"]
    with pytest.raises(gate.CandidateGenerationFailed, match="not all unique"):
        gate.validate_candidate_set(
            duplicate, historical_joint=set(), previous_joint=set(), b4_search_joint=set(),
            historical_architecture=set(), previous_architecture=set(),
        )


def test_09_10_candidate_generation_model_blind_and_high_regime(protocol: dict) -> None:
    generation = protocol["candidate_generation"]
    assert generation["proposal"] == "frozen_model_blind_architecture_decode_plus_HIGH_REGIME_PRIOR"
    assert generation["hp_raw_unit_interval_range"] == [0.80, 0.95]
    assert generation["predictor_validation_metric_distance_head_test_access"] == "forbidden"
    source = inspect.getsource(gate.generate_all_candidates)
    assert "uniform(0.80, 0.95" in source
    assert "score" not in inspect.signature(gate.generate_all_candidates).parameters


def test_11_to_13_frozen_search_grid(protocol: dict, candidates: list[dict]) -> None:
    generation = protocol["candidate_generation"]
    assert (generation["replicates"], generation["rounds_per_replicate"], generation["candidates_per_round"]) == (8, 4, 128)
    counts = {}
    for row in candidates:
        key = (row["replicate_id"], row["round_id"])
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 32 and set(counts.values()) == {128}


def test_14_to_17_everything_is_frozen_before_gnn() -> None:
    source = inspect.getsource(gate.execute_formal)
    positions = [source.index(name) for name in (
        "generate_all_candidates", "score_all_candidates_with_metric",
        "select_search_plans", "freeze_pre_search", "build_selected_union_tasks",
        "run_frozen_gnn_evaluation",
    )]
    assert positions == sorted(positions)


def test_18_19_m1_m2_top4_and_tie_break(candidates: list[dict], predictions: list[dict]) -> None:
    m1, m2, _random, _selected, _information, _collapse = gate.select_search_plans(candidates, predictions)
    assert m1["rounds"][0]["selected_candidate_slots"] == [127, 126, 125, 124]
    assert m2["rounds"][0]["selected_candidate_slots"] == [0, 1, 2, 3]
    tied = [{**row, "m1_prediction": 1.0, "m2_prediction": 1.0} for row in predictions]
    tied_m1, tied_m2, *_ = gate.select_search_plans(candidates, tied)
    expected = sorted(row["joint_fingerprint"] for row in candidates[:128])[:4]
    assert tied_m1["rounds"][0]["selected_joint_fingerprints"] == expected
    assert tied_m2["rounds"][0]["selected_joint_fingerprints"] == expected


def test_20_21_random_sampling_and_no_postprocessing(candidates: list[dict], predictions: list[dict]) -> None:
    first = gate.select_search_plans(candidates, predictions)[2]
    second = gate.select_search_plans(candidates, predictions)[2]
    assert first == second and first["master_seed"] == 202608300002
    assert first["diversity_postprocessing"] is False
    for row in first["rounds"]:
        assert row["sampling"] == "uniform_without_replacement"
        assert len(row["selected_joint_fingerprints"]) == len(set(row["selected_joint_fingerprints"])) == 4


def test_22_collapse_statistics_are_exact(candidates: list[dict], plans) -> None:
    m1, m2, random_plan, _selected, _information, collapse = plans
    recomputed = gate.selection_collapse_audit(candidates, {"M1": m1, "M2": m2, "RANDOM": random_plan})
    assert collapse == recomputed and collapse["selection_modified"] is False
    for method in ("M1", "M2", "RANDOM"):
        total = collapse["methods"][method]["all_selected_slots"]
        assert total["selected_slots_count"] == 128
        assert total["effective_number_of_architectures"] == pytest.approx(
            np.exp(total["architecture_selection_entropy"])
        )


def test_23_24_overlap_is_trained_once_and_method_budget_is_fixed(candidates: list[dict], plans) -> None:
    m1, m2, random_plan, selected, information, _collapse = plans
    configs, tasks = gate.build_selected_union_tasks(candidates, m1, m2, random_plan, pre_search_verified=True)
    union = selected["M1"] | selected["M2"] | selected["RANDOM"]
    assert len(configs) == len(union) and len(tasks) == 3 * len(union)
    assert len({(row["joint_fingerprint"], row["training_seed"]) for row in tasks}) == len(tasks)
    assert information["method_equivalent_config_slots"] == 384
    assert information["method_equivalent_GNN_runs"] == 1152


def test_25_26_seeds_and_same_config_results_are_shared(candidates: list[dict], plans) -> None:
    m1, m2, random_plan, _selected, _information, _collapse = plans
    configs, tasks = gate.build_selected_union_tasks(candidates, m1, m2, random_plan, pre_search_verified=True)
    assert {row["training_seed"] for row in tasks} == {0, 1, 2}
    for config in configs:
        matching = [row for row in tasks if row["joint_fingerprint"] == config["joint_fingerprint"]]
        assert len(matching) == 3


def test_27_28_track_test_false_and_loader_firewall(protocol: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    assert protocol["evaluation"]["track_test"] is False
    with pytest.raises(gate.ProtocolError, match="test-named field"):
        gate.reject_test_named_fields({"test_accuracy": 0.8})

    class FakeData:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def keys(self): return list(self.__dict__)

    fake_module = types.ModuleType("torch_geometric.data")
    fake_module.Data = FakeData
    fake_parent = types.ModuleType("torch_geometric")
    fake_parent.data = fake_module
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_parent)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", fake_module)
    source = types.SimpleNamespace(x=1, edge_index=2, y=3, train_mask=4, val_mask=5, test_mask=object())
    bundle = types.SimpleNamespace(data=source, manifest={"identity": "synthetic"}, num_features=7, num_classes=3)
    loaded, features, classes, context, audit = gate.load_validation_only_dataset(
        object(), lambda _request: bundle, lambda manifest: manifest,
    )
    assert set(loaded.keys()) == {"x", "edge_index", "y", "train_mask", "val_mask"}
    assert not hasattr(loaded, "test_mask")
    assert (features, classes, context) == (7, 3, {"identity": "synthetic"})
    assert audit["raw_loader_bundle_exposed"] is False and audit["test_named_fields_exposed"] == []


def test_29_retry_preserves_task_identity(protocol: dict) -> None:
    assert protocol["evaluation"]["maximum_system_retries"] == 2
    assert protocol["evaluation"]["candidate_replacement_after_failure"] == "forbidden"
    source = inspect.getsource(gate.execute_evaluator_shard)
    assert "while True" in source and "attempt <= maximum_retries" in source
    assert 'task["seed"]' in source and 'task["operations"]' in source


def test_30_31_incomplete_fails_closed_and_three_seed_mean() -> None:
    candidate = {
        "joint_fingerprint": "j", "architecture_fingerprint": "a",
        "selected_by_M1": True, "selected_by_M2": True, "selected_by_random": True,
    }
    prediction = [{"joint_fingerprint": "j", "m1_prediction": 0.9, "m2_prediction": 0.8}]
    rows = [{
        "joint_fingerprint": "j", "training_seed": seed, "validation_accuracy": value,
        "terminal_status": "success", "valid": True,
    } for seed, value in enumerate((0.80, 0.81, 0.82))]
    stats, complete = gate.aggregate_config_statistics([candidate], rows, prediction)
    assert complete is True and stats[0]["validation_mean"] == pytest.approx(0.81)
    assert stats[0]["validation_sd"] == pytest.approx(0.01)
    incomplete, complete = gate.aggregate_config_statistics([candidate], rows[:-1], prediction, allow_incomplete=True)
    assert complete is False and incomplete[0]["validation_mean"] is None
    with pytest.raises(gate.EvaluationIncomplete):
        gate.aggregate_config_statistics([candidate], rows[:-1], prediction)


def test_32_to_34_round_final_anytime_and_auc(promoted_analysis) -> None:
    assert len(promoted_analysis["per_round"]) == 8 * 4 * 3
    assert len(promoted_analysis["per_replicate"]) == 8
    assert len(promoted_analysis["anytime"]) == 8 * 3 * 4
    for row in promoted_analysis["per_replicate"]:
        for method in ("M1", "M2", "RANDOM"):
            curve = [item["best_so_far"] for item in promoted_analysis["anytime"]
                     if item["replicate_id"] == row["replicate_id"] and item["method"] == method]
            assert curve == sorted(curve)
            key = "random_anytime_AUC" if method == "RANDOM" else f"{method}_anytime_AUC"
            assert row[key] == pytest.approx(np.mean(curve))


def test_35_36_bootstrap_unit_and_seed() -> None:
    first = gate.bootstrap_paired_mean([0.01] * 8)
    assert first == gate.bootstrap_paired_mean([0.01] * 8)
    assert first["unit"] == "search_replicate" and first["replicate_count"] == 8
    assert first["bootstrap_replicates"] == 10000 and first["seed"] == 0


def test_37_38_k_and_g_gate_logic(promoted_analysis) -> None:
    decision = promoted_analysis["decision"]
    assert decision["M1_SEARCH_PROMOTED"] is True and decision["M2_SEARCH_PROMOTED"] is True
    assert all(row["passed"] for row in decision["M1_gate"]["gates"].values())
    assert all(row["passed"] for row in decision["M2_gate"]["gates"].values())
    m1, m2, random_plan, information, stats = gate._synthetic_metric_case("none")
    analysis = gate.compute_search_analysis(
        stats, m1, m2, random_plan, information,
        {"selected_unique_config_count": len(stats), "physical_unique_GNN_runs": 3 * len(stats)},
        {"selection_modified": False, "methods": {}},
    )
    assert analysis["decision"]["classification"] == "METRIC_SEARCH_NOT_CONFIRMED"


def test_39_point003_tolerance() -> None:
    def rows(delta: float) -> list[dict]:
        return [{
            "M1_minus_random": delta, "M1_anytime_AUC": 0.81,
            "random_anytime_AUC": 0.80,
        } for _ in range(8)]
    anytime = [
        {"method": method, "budget": budget, "best_so_far": 0.81 if method == "M1" else 0.80}
        for _replicate in range(8) for budget in (4, 8, 12, 16) for method in ("M1", "RANDOM")
    ]
    bootstrap = {"ci95_lower": 0.001}
    assert gate.classify_metric_gate("M1", rows(0.003 - 0.5e-12), anytime, bootstrap)["gates"]["K2"]["passed"] is True
    assert gate.classify_metric_gate("M1", rows(0.003 - 2e-12), anytime, bootstrap)["gates"]["K2"]["passed"] is False


def test_40_41_surrogate_rule_observed_union_oracle_and_parsimony(promoted_analysis) -> None:
    decision = promoted_analysis["decision"]
    assert decision["classification"] == "M1_M2_SEARCH_PROMOTED_GP_SELECTED"
    selection = decision["M1_vs_M2_surrogate_selection"]
    assert selection["regret_oracle_name"] == "OBSERVED_EVALUATED_UNION_ORACLE_r"
    assert selection["is_complete_128_candidate_pool_oracle"] is False
    zero_rows = [{"M2_minus_M1": 0.0, "REGRET_M1": 0.0, "REGRET_M2": 0.0} for _ in range(8)]
    zero = gate.classify_surrogate_selection(zero_rows, {"ci95_lower": 1.0})
    assert zero["gates"]["S2"]["passed"] is False and zero["NEXT_SURROGATE"] == "M1"
    assert zero["gates"]["S2"]["tolerance"] == 1e-12


def test_42_historical_p_is_descriptive_only(promoted_analysis) -> None:
    assert "historical_P" not in inspect.signature(gate.classify_metric_gate).parameters
    assert promoted_analysis["decision"]["historical_P_used_in_primary_gate"] is False
    assert promoted_analysis["auxiliary"]["historical_P"]["primary_gate_use"] is False


def test_43_44_promoted_winner_freeze_and_no_fresh_seeds(tmp_path: Path, promoted_analysis, protocol: dict) -> None:
    promoted = tmp_path / "promoted"
    result = gate.write_next_stage(
        promoted, promoted_analysis["decision"]["classification"], "M2",
        promoted_analysis["m1_winners"], promoted_analysis["m2_winners"], promoted_analysis["random_winners"],
    )
    assert result["promoted"] is True
    assert (promoted / "next_stage/NEXT_STAGE_WINNERS_FROZEN").is_file()
    assert len(json.loads((promoted / "next_stage/metric_top3.json").read_text())["winners"]) == 3
    not_promoted = tmp_path / "not-promoted"
    gate.write_next_stage(
        not_promoted, "METRIC_SEARCH_NOT_CONFIRMED", None,
        promoted_analysis["m1_winners"], promoted_analysis["m2_winners"], promoted_analysis["random_winners"],
    )
    assert not (not_promoted / "next_stage/NEXT_STAGE_WINNERS_FROZEN").exists()
    assert protocol["next_stage"]["fresh_seed_training_in_this_run"] == "forbidden"


def test_45_source_snapshot_contract() -> None:
    paths = gate._source_identity_paths()
    required = {
        "configs/m1_m2_metric_guided_search_accuracy_gate_v1.yaml",
        "analyse/m1_m2_metric_guided_search_accuracy_gate.py",
        "scripts/run_m1_m2_metric_guided_search_accuracy_gate.sh",
        "tests/test_m1_m2_metric_guided_search_accuracy_gate.py",
        "frozen/analyse/joint_configuration_soft_metric_gate.py",
        "frozen/analyse/joint_configuration_metric_gate.py",
        "frozen/analyse/performance_metric_quick_gate.py",
        "frozen/analyse/high_regime_prior_model_blind_validation.py",
        "frozen/analyse/run_high_regime_prior_model_blind_validation_labels.py",
        "dataset_utils.py", "hp_modes.py",
    }
    assert required <= set(paths) and all(path.is_file() for path in paths.values())
    source = inspect.getsource(gate.snapshot_sources)
    assert "sys.modules.items()" in source and "shutil.copy2" in source


def test_46_final_report_has_27_sections(promoted_analysis) -> None:
    artifacts = {
        **promoted_analysis, "replay": {"status": "IDENTICAL", "model_count": 15},
        "generation": {"coverage": {"candidate_count": 4096, "unique_joint_count": 4096}},
        "information": {"selected_union_unique_configurations": 384, "replicates": []},
        "training_audit": {"evaluation_complete": True, "selected_unique_config_count": 384,
                           "physical_unique_GNN_runs": 1152, "completed_gnn_runs": 1152,
                           "failed_gnn_runs": 0, "loader_firewall_passed": True},
        "pre_search_root": "abc", "next_stage": {"promoted": True, "metric_method": "M2"},
    }
    report = gate.render_final_report(artifacts)
    gate.assert_complete_final_report(report)
    assert sum(f"## {section}" in report for section in gate.REPORT_SECTIONS) == 27
    assert "not the complete 128-candidate pool" in report


def test_47_dry_run_does_not_create_formal_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "formal-result"
    assert gate.execute_dry_run(gate.DEFAULT_CONFIG, output) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["formal_output_created"] is False
    assert payload["formal_candidate_generation_started"] is False
    assert payload["formal_search_plan_frozen"] is False
    assert payload["GNN_training_started"] is False and not output.exists()


def test_48_formal_result_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(gate.ProtocolError, match="already exists"):
        gate.ensure_output_policy(output, Path(str(output) + ".tar.gz"))


def test_49_failure_bundle_can_be_generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "failed-bundle"
    monkeypatch.setattr(
        gate, "_frozen_dependency_entries",
        lambda: (_ for _ in ()).throw(gate.InputIdentityInvalid("synthetic missing dependency")),
    )
    gate.write_failure_bundle(output, "synthetic failure")
    assert (output / "FAILED").is_file() and not (output / "COMPLETE").exists()
    assert json.loads((output / "decision.json").read_text())["classification"] == "RUN_FAILED"
    assert Path(str(output) + ".tar.gz").is_file()
    assert Path(str(output) + ".tar.gz.sha256").is_file()


def test_synthetic_smoke_covers_terminal_logic() -> None:
    result = gate.synthetic_smoke()
    assert result["passed"] is True
    assert result["observed_union_oracle_rule_checked"] is True
    assert result["GNN_training_started"] is False and result["test_accessed"] is False
