from __future__ import annotations

from collections import Counter
import inspect
import json
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import b4_arch_diversity_prospective_search_accuracy_gate as gate


@pytest.fixture(scope="session")
def protocol() -> dict:
    value = gate.load_config()
    assert gate.validate_frozen_protocol(value)["status"] == "VALID"
    return value


@pytest.fixture(scope="session")
def candidates() -> list[dict]:
    # This is the analysis module's CPU-only synthetic grid.  It never invokes
    # the VAE decoder or any formal candidate-generation path.
    return gate._synthetic_candidates()


@pytest.fixture(scope="session")
def predictions(candidates: list[dict]) -> list[dict]:
    return [
        {
            "replicate_id": row["replicate_id"],
            "round_id": row["round_id"],
            "candidate_slot": row["candidate_slot"],
            "joint_fingerprint": row["joint_fingerprint"],
            "architecture_fingerprint": row["architecture_fingerprint"],
            "b4_prediction": float(gate.CANDIDATES_PER_ROUND - row["candidate_slot"]),
        }
        for row in candidates
    ]


@pytest.fixture(scope="session")
def plans(candidates: list[dict], predictions: list[dict]):
    return gate.select_search_plans(candidates, predictions)


@pytest.fixture(scope="session")
def exclusion_identities(protocol: dict):
    # Use synthetic historical rows plus the authenticated, label-free frozen
    # candidate manifests.  This only parses files and hashes freeze roots.
    historical = [
        {
            "joint_fingerprint": f"unit-historical-joint-{index:03d}",
            "architecture_fingerprint": f"unit-historical-arch-{index:03d}",
        }
        for index in range(336)
    ]
    return gate.load_exclusion_identities(historical, protocol)


def _analysis_inputs(
    *, pure_base: float, p2_base: float, random_base: float,
) -> tuple[list[dict], dict[str, dict], list[dict], dict, dict, dict]:
    """Small selected-only search fixture; no 128-pool or evaluator is used."""
    bases = {gate.PURE: pure_base, gate.P2: p2_base, gate.RANDOM: random_base}
    candidate_rows: list[dict] = []
    stats: list[dict] = []
    plans: dict[str, dict] = {}
    for method in gate.METHODS:
        rounds = []
        for replicate in range(gate.REPLICATES):
            for round_id in range(gate.ROUNDS):
                joints = []
                for slot in range(gate.SELECTED_PER_ROUND):
                    joint = f"analysis::{method}::{replicate}::{round_id}::{slot}"
                    architecture = f"analysis-arch::{method}::{replicate}::{round_id // 2}::{slot}"
                    value = bases[method] + 0.001 * round_id + 0.00001 * slot
                    joints.append(joint)
                    candidate_rows.append({
                        "replicate_id": replicate,
                        "round_id": round_id,
                        "candidate_slot": slot,
                        "joint_fingerprint": joint,
                        "architecture_fingerprint": architecture,
                    })
                    stats.append({
                        "joint_fingerprint": joint,
                        "architecture_fingerprint": architecture,
                        "validation_mean": value,
                        "validation_sd": 0.005,
                        "b4_prediction": value,
                    })
                rounds.append({
                    "replicate_id": replicate,
                    "round_id": round_id,
                    "selected_joint_fingerprints": joints,
                })
        plans[method] = gate._plan(method, rounds)
    manipulation = gate.diversity_manipulation_audit(plans, candidate_rows)
    information = {"passed": True}
    training = {
        "evaluation_complete": True,
        "physical_unique_configurations": len(candidate_rows),
        "physical_unique_GNN_runs": len(candidate_rows) * 3,
        "completed_gnn_runs": len(candidate_rows) * 3,
        "failed_gnn_runs": 0,
        "method_equivalent_config_slots": 576,
        "method_equivalent_GNN_runs": 1728,
        "track_test": False,
        "test_accessed": False,
    }
    return stats, plans, candidate_rows, manipulation, information, training


def _analysis(*, pure_base: float, p2_base: float, random_base: float) -> dict:
    return gate.compute_search_analysis(
        *_analysis_inputs(
            pure_base=pure_base, p2_base=p2_base, random_base=random_base,
        )
    )


@pytest.fixture(scope="session")
def promoted_analysis() -> dict:
    return _analysis(pure_base=0.780, p2_base=0.785, random_base=0.776)


# 01-03: diagnosis authorization, selected scorer, and selected policy.
def test_01_to_03_authorization_is_diversity_b4_p2(protocol: dict) -> None:
    decision = json.loads((gate.DIAGNOSIS_ROOT / "decision.json").read_text(encoding="utf-8"))
    state = json.loads((gate.DIAGNOSIS_ROOT / "state.json").read_text(encoding="utf-8"))
    prospective = json.loads(
        (gate.DIAGNOSIS_ROOT / "next_experiment/prospective_diversity_test_spec.json").read_text(encoding="utf-8")
    )
    authority = protocol["authoritative_inputs"]["collapse_diagnosis"]
    assert (gate.DIAGNOSIS_ROOT / "COMPLETE").is_file()
    assert decision["classification"] == authority["required_classification"] == "DIVERSITY_TEST_AUTHORIZED"
    assert state["classification"] == "DIVERSITY_TEST_AUTHORIZED"
    assert decision["SCORER_FOR_NEXT_DIVERSITY_TEST"] == authority["required_scorer"] == "B4"
    assert decision["SELECTED_DIVERSITY_POLICY"] == authority["required_policy"] == "P2"
    assert prospective["selected_scorer"] == "B4"
    assert prospective["selected_diversity_policy"] == "P2"
    assert decision["test_accessed"] is state["test_accessed"] is False


# 04: frozen B4 must replay both the upstream identity and all 4,096 old-search predictions.
def test_04_b4_scorer_identity_replay_and_mismatch_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_candidates = [
        {
            "replicate_id": index // 512,
            "round_id": (index // 128) % 4,
            "candidate_slot": index % 128,
            "joint_fingerprint": f"old-joint-{index:04d}",
            "architecture_fingerprint": f"old-arch-{index % 200:03d}",
        }
        for index in range(4096)
    ]
    reference = [
        {"joint_fingerprint": row["joint_fingerprint"], "b4_prediction": f"{index / 4096:.12f}"}
        for index, row in enumerate(old_candidates)
    ]
    monkeypatch.setattr(gate, "_parse_candidate_csv", lambda _path: old_candidates)
    monkeypatch.setattr(gate, "read_csv", lambda _path: reference)
    monkeypatch.setattr(
        gate.base,
        "replay_frozen_b4",
        lambda *_args, **_kwargs: {
            "status": "IDENTICAL", "maximum_absolute_error": 0.0,
            "anchors": {"historical_OOF_sha256": "synthetic"},
        },
    )
    monkeypatch.setattr(
        gate,
        "score_candidates_with_frozen_b4",
        lambda *_args, **_kwargs: ([
            {
                "joint_fingerprint": row["joint_fingerprint"],
                "b4_prediction": float(reference[index]["b4_prediction"]),
            }
            for index, row in enumerate(old_candidates)
        ], {"status": "SCORED"}),
    )
    monkeypatch.setattr(gate, "verify_historical_pre_search", lambda _root: {"root_sha256": "synthetic-root"})
    replay = gate.replay_frozen_b4({}, [], lambda _message: None)
    assert replay["status"] == "IDENTICAL"
    assert replay["previous_B4_search_prediction_count"] == 4096
    assert replay["B4_retrained_or_updated_scientifically"] is False

    def mismatching_score(*_args, **_kwargs):
        rows = [
            {"joint_fingerprint": row["joint_fingerprint"], "b4_prediction": float(reference[index]["b4_prediction"])}
            for index, row in enumerate(old_candidates)
        ]
        rows[0]["b4_prediction"] += gate.REPLAY_ATOL * 2
        return rows, {"status": "SCORED"}

    monkeypatch.setattr(gate, "score_candidates_with_frozen_b4", mismatching_score)
    with pytest.raises(gate.B4ScorerReplayMismatch, match="prediction mismatch"):
        gate.replay_frozen_b4({}, [], lambda _message: None)


# 05: only historical supervision may reach deterministic scorer reconstruction.
def test_05_prior_search_labels_never_enter_b4_training() -> None:
    expected_epochs = gate.base._expected_best_epochs(gate.base.load_config())

    def train(_train, query, seed, _config, standardizer):
        fold = int(standardizer["fold"])
        return np.full(len(query), 0.8 + seed * 0.001), {
            "best_epoch": expected_epochs[(fold, seed)],
        }

    joint = SimpleNamespace(
        train_direct_joint_regressor=train,
        average_metric_seed_predictions=lambda values: np.mean(
            np.stack([values[seed] for seed in gate.B4_MODEL_SEEDS]), axis=0,
        ),
    )
    runtime = {
        "soft": SimpleNamespace(load_config=lambda _path: {}, joint_v1=joint),
        "soft_config_path": Path("synthetic.yaml"),
    }
    folds = [
        SimpleNamespace(fold=fold, train_rows=(), hp_standardizer={"fold": fold})
        for fold in range(5)
    ]
    rows, audit = gate.score_candidates_with_frozen_b4(
        runtime,
        folds,
        [{
            "replicate_id": 0, "round_id": 0, "candidate_slot": 0,
            "joint_fingerprint": "query", "architecture_fingerprint": "arch",
        }],
        lambda _message: None,
        label="unit",
    )
    assert len(rows) == 1
    assert audit["historical_supervision"] == "historical_336_only"
    assert audit["previous_prospective_labels_used"] is False
    assert audit["old_search_labels_used"] is False
    assert audit["current_labels_used"] is False
    assert audit["online_update_used"] is audit["round_refit_used"] is False


# 06-07: generator remains model blind and preserves HIGH_REGIME_PRIOR identity.
def test_06_07_generator_model_blind_and_high_regime_identity(protocol: dict) -> None:
    generation = protocol["candidate_generation"]
    assert generation["proposal"] == "frozen_model_blind_architecture_decode_plus_HIGH_REGIME_PRIOR"
    assert generation["hp_raw_unit_interval_range"] == [0.80, 0.95]
    assert generation["scorer_validation_test_or_search_feedback_access"] == "forbidden"
    signature = inspect.signature(gate.generate_all_candidates)
    assert set(signature.parameters) == {"runtime", "exclusions", "log"}
    source = inspect.getsource(gate.generate_all_candidates)
    for assertion in (
        '"predictor_accessed": False', '"validation_accessed": False',
        '"historical_best_accessed": False', '"metric_accessed": False',
        '"mutation_or_local_search_used": False', '"acquisition_used": False',
    ):
        assert assertion in source


# 08-12 and 18: exact grid/cardinality/uniqueness while architecture repetition remains legal.
def test_08_to_12_and_18_candidate_grid_uniqueness_and_architecture_reuse(
    candidates: list[dict],
) -> None:
    assert gate.REPLICATES == 12
    assert gate.ROUNDS == 4
    assert gate.CANDIDATES_PER_ROUND == 128
    assert gate.TOTAL_CANDIDATES == 6144 == len(candidates)
    assert len({row["joint_fingerprint"] for row in candidates}) == 6144
    assert len({row["architecture_fingerprint"] for row in candidates}) < 6144
    coverage = gate.validate_candidate_set(
        candidates,
        {
            "joint": set(), "historical_architecture": set(),
            "prior_search_architecture": set(),
        },
    )
    assert coverage["passed"] is True
    assert coverage["unique_joint_count"] == 6144
    assert coverage["architecture_reuse_allowed"] is True
    assert coverage["architecture_repetition_rate"] > 0


# 13-17: every preregistered historical/prior exact-joint universe is excluded.
@pytest.mark.parametrize(
    "source_key, expected_count",
    [
        ("joint_historical_336", 336),
        ("joint_previous_8_pool_32", 32),
        ("joint_previous_16_pool_64", 64),
        ("joint_old_b4_search_all_4096", 4096),
        ("joint_old_m1_m2_search_all_4096", 4096),
    ],
)
def test_13_to_17_each_exact_joint_exclusion_is_enforced(
    exclusion_identities, source_key: str, expected_count: int,
) -> None:
    exclusions, manifest = exclusion_identities
    assert len(exclusions[source_key]) == expected_count
    leaked = next(iter(exclusions[source_key]))
    with pytest.raises(gate.CandidateGenerationFailed, match="excluded exact joint"):
        gate.validate_candidate_set(
            [{
                "replicate_id": 0, "round_id": 0, "candidate_slot": 0,
                "joint_fingerprint": leaked, "architecture_fingerprint": "new-arch",
                "proposal_distribution": "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR",
            }],
            exclusions,
            expected_count=1,
        )
    assert manifest["joint_union_count"] == 8624
    assert all(row["joint_overlap"] == 0 for row in manifest["pairwise_overlaps"])


# 19-23: candidates, scores, and all three selections precede task creation/GNN.
def test_19_to_23_all_search_material_is_frozen_before_any_gnn(
    candidates: list[dict], predictions: list[dict], plans, tmp_path: Path,
) -> None:
    search_plans, _selected, feasibility, audits = plans
    manipulation = gate.diversity_manipulation_audit(search_plans, candidates)
    freeze = gate.freeze_pre_search(
        tmp_path / "pre_search", candidates, predictions, search_plans,
        audits["selection_overlap"], manipulation, audits["information_gate"],
        feasibility, {"status": "IDENTICAL"}, {"status": "IDENTICAL"},
        {"status": "LOADED"},
    )
    plan_payload = json.loads((tmp_path / "pre_search/search_plan.json").read_text(encoding="utf-8"))
    assert plan_payload["all_candidates_generated_before_first_GNN"] is True
    assert plan_payload["all_B4_predictions_completed_before_first_GNN"] is True
    assert plan_payload["all_PURE_selections_completed_before_first_GNN"] is True
    assert plan_payload["all_P2_selections_completed_before_first_GNN"] is True
    assert plan_payload["all_random_selections_completed_before_first_GNN"] is True
    assert plan_payload["GNN_training_started"] is False
    assert gate.verify_pre_search_freeze(tmp_path / "pre_search")["root_sha256"] == freeze["root_sha256"]
    with pytest.raises(gate.ProtocolError, match="forbidden before pre-search"):
        gate.build_selected_union_tasks(candidates, search_plans, pre_search_verified=False)
    formal_source = inspect.getsource(gate.execute_formal)
    ordered_tokens = (
        "generate_all_candidates", "score_candidates_with_frozen_b4",
        "select_search_plans", "freeze_pre_search", "build_selected_union_tasks",
        "run_frozen_gnn_evaluation",
    )
    positions = [formal_source.index(token) for token in ordered_tokens]
    assert positions == sorted(positions)


# 24-29: PURE ranking, P2 traversal, uniqueness, cap, tie-break, and label firewall.
def test_24_to_29_pure_and_p2_selection_contract(
    candidates: list[dict], predictions: list[dict], plans,
) -> None:
    search_plans, _selected, feasibility, _audits = plans
    first_pure = search_plans[gate.PURE]["rounds"][0]
    first_p2 = search_plans[gate.P2]["rounds"][0]
    assert first_pure["selected_candidate_slots"] == [0, 1, 2, 3]
    assert first_p2["selected_original_ranks"] == [1, 9, 10, 11]
    assert len(set(first_p2["selected_architecture_fingerprints"])) == 4
    assert feasibility["legal_rounds"] == 48
    assert feasibility["maximum_observed_architecture_slots_per_replicate"] <= 2
    counts = Counter(
        (int(row["replicate_id"]), architecture)
        for row in search_plans[gate.P2]["rounds"]
        for architecture in row["selected_architecture_fingerprints"]
    )
    assert max(counts.values()) <= 2

    tied = [{**row, "b4_prediction": 1.0} for row in predictions]
    tied_plans, _tied_selected, _tied_feasibility, _tied_audits = gate.select_search_plans(candidates, tied)
    first_pool_joints = sorted(row["joint_fingerprint"] for row in candidates[:128])
    assert tied_plans[gate.PURE]["rounds"][0]["selected_joint_fingerprints"] == first_pool_joints[:4]
    assert tied_plans[gate.P2]["rounds"][0]["tie_break"] == "joint_fingerprint_lexicographic_ascending"

    labeled = [dict(row) for row in candidates]
    labeled[0]["validation_mean"] = 1.0
    with pytest.raises(gate.ProtocolError, match="validation labels"):
        gate.select_search_plans(labeled, predictions)


# 30-31: seeded uniform-without-replacement Random never redraws overlap.
def test_30_31_random_is_seeded_without_replacement_and_never_redraws_overlap(
    candidates: list[dict], predictions: list[dict], plans,
) -> None:
    search_plans, _selected, _feasibility, audits = plans
    random_plan = search_plans[gate.RANDOM]
    rng = np.random.default_rng(gate.RANDOM_MASTER_SEED)
    expected_first = sorted(rng.choice(128, size=4, replace=False).tolist())
    assert random_plan["rounds"][0]["selected_candidate_slots"] == expected_first
    assert random_plan["sampling"] == "numpy_default_rng_uniform_without_replacement"
    assert random_plan["overlap_redraw"] is False
    assert audits["selection_overlap"]["overlap_redraw_performed"] is False
    repeated = gate.select_search_plans(candidates, predictions)[0][gate.RANDOM]
    assert repeated == random_plan
    assert all(len(set(row["selected_joint_fingerprints"])) == 4 for row in random_plan["rounds"])


# 32: immutable root is reproducible and detects tampering.
def test_32_pre_search_root_reproducible_and_tamper_evident(
    candidates: list[dict], predictions: list[dict], plans, tmp_path: Path,
) -> None:
    search_plans, _selected, feasibility, audits = plans
    manipulation = gate.diversity_manipulation_audit(search_plans, candidates)
    freeze = gate.freeze_pre_search(
        tmp_path / "pre", candidates, predictions, search_plans,
        audits["selection_overlap"], manipulation, audits["information_gate"],
        feasibility, {"status": "IDENTICAL"}, {"status": "IDENTICAL"},
        {"status": "LOADED"},
    )
    assert [row["relative_path"] for row in freeze["files"]] == sorted(gate.PRE_SEARCH_FILES)
    assert gate.verify_pre_search_freeze(tmp_path / "pre")["root_sha256"] == freeze["root_sha256"]
    prediction_path = tmp_path / "pre/b4_predictions.csv"
    prediction_path.write_text(prediction_path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    with pytest.raises(gate.ProtocolError, match="changed"):
        gate.verify_pre_search_freeze(tmp_path / "pre")


# 33-34: exact divergence and frozen information-gate rule.
def test_33_34_information_divergence_and_gate_rule(
    candidates: list[dict], predictions: list[dict], plans,
) -> None:
    search_plans, _selected, _feasibility, audits = plans
    info = audits["information_gate"]
    for row in info["per_replicate"]:
        replicate = int(row["replicate_id"])
        pure = {
            joint for round_row in search_plans[gate.PURE]["rounds"]
            if int(round_row["replicate_id"]) == replicate
            for joint in round_row["selected_joint_fingerprints"]
        }
        p2 = {
            joint for round_row in search_plans[gate.P2]["rounds"]
            if int(round_row["replicate_id"]) == replicate
            for joint in round_row["selected_joint_fingerprints"]
        }
        assert row["DIVERGENCE_r"] == 16 - len(pure & p2)
    assert info["passed"] == (
        info["replicates_with_divergence_at_least_4"] >= 8
        and info["median_divergence"] >= 4
    )

    low_information_candidates = [
        {
            **row,
            "architecture_fingerprint": (
                f"round-unique-{row['replicate_id']}-{row['round_id']}-{row['candidate_slot']}"
            ),
        }
        for row in candidates
    ]
    low_plans, _low_selected, _low_feasibility, low_audits = gate.select_search_plans(
        low_information_candidates, predictions,
    )
    assert all(
        round_row["selected_joint_fingerprints"]
        == low_plans[gate.P2]["rounds"][index]["selected_joint_fingerprints"]
        for index, round_row in enumerate(low_plans[gate.PURE]["rounds"])
    )
    assert low_audits["information_gate"]["passed"] is False
    assert low_audits["information_gate"]["replicates_with_divergence_at_least_4"] == 0


# 35-37: overlap configs train once, budgets remain method-equivalent, seeds fixed.
def test_35_to_37_union_dedup_budget_and_training_seeds(
    candidates: list[dict], plans,
) -> None:
    search_plans, selected, _feasibility, audits = plans
    configs, tasks = gate.build_selected_union_tasks(candidates, search_plans, pre_search_verified=True)
    union = set().union(*selected.values())
    assert len(configs) == len(union)
    assert len(tasks) == len(union) * 3
    assert len({(row["joint_fingerprint"], row["training_seed"]) for row in tasks}) == len(tasks)
    assert {row["training_seed"] for row in tasks} == {0, 1, 2}
    assert audits["selection_overlap"]["overall"]["method_equivalent_config_slots"] == 576
    assert audits["selection_overlap"]["overall"]["method_equivalent_GNN_runs"] == 1728
    overlap = next(
        (row for row in configs if row["selected_by_pure_b4"] and row["selected_by_p2"]),
        None,
    )
    assert overlap is not None
    shared = [row for row in tasks if row["joint_fingerprint"] == overlap["joint_fingerprint"]]
    assert len(shared) == 3


# 38-40: track_test false, loader firewall, and retry identity preservation.
def test_38_to_40_test_firewall_and_retry_identity(
    protocol: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert protocol["evaluation"]["track_test"] is False
    with pytest.raises(gate.base.ProtocolError, match="test-named field"):
        gate.base.reject_test_named_fields({"test_accuracy": 0.9})
    gate.base.reject_test_named_fields(
        {"track_test": False, "test_accessed": False}, allow_false_controls=True,
    )

    class FakeData:
        def __init__(self, **values):
            self.__dict__.update(values)

        def keys(self):
            return list(self.__dict__)

    data_module = types.ModuleType("torch_geometric.data")
    data_module.Data = FakeData
    parent_module = types.ModuleType("torch_geometric")
    parent_module.data = data_module
    monkeypatch.setitem(sys.modules, "torch_geometric", parent_module)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)
    source = SimpleNamespace(
        x=1, edge_index=2, y=3, train_mask=4, val_mask=5, test_mask=object(),
    )
    view = gate.base.validation_only_data_view(source)
    assert set(view.keys()) == {"x", "edge_index", "y", "train_mask", "val_mask"}
    assert not hasattr(view, "test_mask")

    retry_source = inspect.getsource(gate.base.execute_evaluator_shard)
    assert "while True" in retry_source
    assert 'task["seed"]' in retry_source
    assert 'task["operations"]' in retry_source
    assert "candidate replacement" not in retry_source.lower()


# 41-42: incomplete evaluation fails closed; complete triplet mean/SD is exact.
def test_41_42_incomplete_fail_closed_and_three_seed_aggregation() -> None:
    selected = [{
        "joint_fingerprint": "joint", "architecture_fingerprint": "arch",
        "selected_by_pure_b4": True, "selected_by_p2": True,
        "selected_by_random": False,
    }]
    prediction = [{"joint_fingerprint": "joint", "b4_prediction": 0.9}]
    raw = [
        {
            "joint_fingerprint": "joint", "training_seed": seed,
            "validation_accuracy": value, "terminal_status": "success", "valid": True,
        }
        for seed, value in enumerate((0.80, 0.81, 0.82))
    ]
    stats, complete = gate.aggregate_config_statistics(selected, raw, prediction)
    assert complete is True
    assert stats[0]["validation_mean"] == pytest.approx(0.81)
    assert stats[0]["validation_sd"] == pytest.approx(0.01)
    partial, partial_complete = gate.aggregate_config_statistics(
        selected, raw[:-1], prediction, allow_incomplete=True,
    )
    assert partial_complete is False
    assert partial[0]["validation_mean"] is None
    with pytest.raises(gate.EvaluationIncomplete, match="lacks seeds"):
        gate.aggregate_config_statistics(selected, raw[:-1], prediction)


# 43-45: final best, best@4/8/12/16, and discrete anytime AUC.
def test_43_to_45_final_best_anytime_curve_and_auc(promoted_analysis: dict) -> None:
    assert len(promoted_analysis["per_round"]) == 12 * 4 * 3
    assert len(promoted_analysis["per_replicate"]) == 12
    assert len(promoted_analysis["anytime_curves"]) == 12 * 3 * 4
    for row in promoted_analysis["per_replicate"]:
        replicate = int(row["replicate_id"])
        for method in gate.METHODS:
            curve = [
                value["best_so_far"] for value in promoted_analysis["anytime_curves"]
                if int(value["replicate_id"]) == replicate and value["method"] == method
            ]
            assert len(curve) == 4
            assert curve == sorted(curve)
            assert row[f"best_{method}"] == pytest.approx(curve[-1])
            assert row[f"{method}_anytime_AUC"] == pytest.approx(float(np.mean(curve)))
    assert {row["budget"] for row in promoted_analysis["anytime_curves"]} == {4, 8, 12, 16}


# 46-50: bootstrap unit/seeds and all five A/B gates.
def test_46_to_50_bootstrap_and_primary_A_B_gates() -> None:
    per_replicate, curves = gate._synthetic_comparison_case()
    comparison_a = gate.classify_comparison(
        per_replicate, curves, left=gate.P2, right=gate.PURE,
        bootstrap_seed=0, primary=True,
    )
    comparison_b = gate.classify_comparison(
        per_replicate, curves, left=gate.P2, right=gate.RANDOM,
        bootstrap_seed=1, primary=True,
    )
    assert comparison_a["bootstrap"]["unit"] == "search_replicate"
    assert comparison_a["bootstrap"]["input_replicates"] == 12
    assert comparison_a["bootstrap"]["bootstrap_replicates"] == 10_000
    assert comparison_a["bootstrap"]["seed"] == 0
    assert comparison_b["bootstrap"]["seed"] == 1
    assert comparison_a["all_five_gates_passed"] is True
    assert comparison_b["all_five_gates_passed"] is True
    assert all(row["passed"] for row in comparison_a["gates"].values())
    assert all(row["passed"] for row in comparison_b["gates"].values())


# 51: the 0.003 comparisons and strict wins use the frozen 1e-12 tolerance.
def test_51_numerical_tolerance_is_applied_to_gain_and_ties() -> None:
    per_replicate, curves = gate._synthetic_comparison_case()
    for row in per_replicate:
        row[f"best_{gate.PURE}"] = 0.8
        row[f"best_{gate.P2}"] = 0.8 + gate.MINIMUM_GAIN - 0.5 * gate.TOLERANCE
    boundary = gate.classify_comparison(
        per_replicate, curves, left=gate.P2, right=gate.PURE,
        bootstrap_seed=0, primary=True,
    )
    assert boundary["gates"]["2"]["passed"] is True
    assert boundary["gates"]["2"]["tolerance"] == 1e-12

    for row in per_replicate:
        row[f"best_{gate.P2}"] = 0.8 + 0.5 * gate.TOLERANCE
    tied = gate.classify_comparison(
        per_replicate, curves, left=gate.P2, right=gate.PURE,
        bootstrap_seed=0, primary=True,
    )
    assert tied["wins"] == 0
    assert tied["ties"] == 12


# 52: the outcome mapping is exhaustive for all three analytic terminal states.
def test_52_final_classification_mapping_is_exhaustive() -> None:
    not_confirmed = _analysis(pure_base=0.780, p2_base=0.781, random_base=0.776)
    mechanism_only = _analysis(pure_base=0.780, p2_base=0.785, random_base=0.790)
    promoted = _analysis(pure_base=0.780, p2_base=0.785, random_base=0.776)
    assert not_confirmed["decision"]["classification"] == "DIVERSITY_SEARCH_NOT_CONFIRMED"
    assert mechanism_only["decision"]["classification"] == "DIVERSITY_MECHANISM_CONFIRMED_NO_SEARCH_PROMOTION"
    assert promoted["decision"]["classification"] == "DIVERSITY_SEARCH_PROMOTED"
    expected = {
        "INPUT_IDENTITY_INVALID", "B4_SCORER_REPLAY_MISMATCH",
        "CANDIDATE_GENERATION_FAILED", "P2_POLICY_INFEASIBLE",
        "DIVERSITY_INTERVENTION_LOW_INFORMATION", "DIVERSITY_EVALUATION_INCOMPLETE",
        "DIVERSITY_SEARCH_NOT_CONFIRMED",
        "DIVERSITY_MECHANISM_CONFIRMED_NO_SEARCH_PROMOTION",
        "DIVERSITY_SEARCH_PROMOTED", "RUN_FAILED",
    }
    assert gate.ALLOWED_CLASSIFICATIONS == expected


# 53-54: observed-union regret and historical P are descriptive and never gate inputs.
def test_53_54_observed_union_regret_and_historical_p_do_not_gate(
    promoted_analysis: dict,
) -> None:
    decision = promoted_analysis["decision"]
    observed = promoted_analysis["observed_union_regret"]
    assert observed["oracle_name"] == "OBSERVED_EVALUATED_UNION_ORACLE_r"
    assert observed["is_complete_128_candidate_pool_oracle"] is False
    assert observed["primary_gate_use"] is False
    assert decision["observed_union_regret_used_in_primary_gates"] is False
    assert decision["historical_P_used_in_primary_gates"] is False
    assert promoted_analysis["historical_P"]["formal_inference_made"] is False
    comparison_signature = inspect.signature(gate.classify_comparison)
    assert "historical_P" not in comparison_signature.parameters
    assert "observed_union_regret" not in comparison_signature.parameters


def _winner_fixture() -> dict[str, list[dict]]:
    output = {}
    for method in gate.METHODS:
        output[method] = [
            {
                "replicate_id": replicate,
                "method": method,
                "joint_fingerprint": f"winner::{method}::{replicate:02d}",
                "architecture_fingerprint": f"winner-arch::{method}::{replicate:02d}",
                "validation_mean": 0.80 + replicate * 0.001,
                "validation_sd": 0.005,
                "b4_prediction": 0.82,
            }
            for replicate in range(12)
        ]
    return output


# 55-57: promoted-only freeze, deterministic top-3 unique rule, fresh seeds only specified.
def test_55_to_57_next_stage_freeze_top3_and_fresh_seeds(
    tmp_path: Path,
) -> None:
    winners = _winner_fixture()
    not_promoted = gate.write_next_stage(
        tmp_path / "not-promoted", "DIVERSITY_SEARCH_NOT_CONFIRMED", winners,
    )
    assert not_promoted["frozen"] is False
    assert not (tmp_path / "not-promoted/next_stage/NEXT_STAGE_WINNERS_FROZEN").exists()

    promoted = gate.write_next_stage(
        tmp_path / "promoted", "DIVERSITY_SEARCH_PROMOTED", winners,
    )
    assert promoted["frozen"] is True
    assert (tmp_path / "promoted/next_stage/NEXT_STAGE_WINNERS_FROZEN").is_file()
    assert promoted["fresh_training_seeds"] == list(range(1000, 1030))
    assert promoted["fresh_seed_confirmation_started"] is False
    seed_spec = json.loads((tmp_path / "promoted/next_stage/fresh_seed_spec.json").read_text())
    assert seed_spec["executed_in_this_run"] is False
    assert seed_spec["GNN_runs_started"] == 0
    assert not set(seed_spec["training_seeds"]) & set(gate.TRAINING_SEEDS)
    for filename in ("p2_top3.json", "pure_b4_top3.json", "random_top3.json"):
        payload = json.loads((tmp_path / "promoted/next_stage" / filename).read_text())
        assert len(payload["winners"]) == 3
        assert len({row["joint_fingerprint"] for row in payload["winners"]}) == 3

    tied = [
        {"joint_fingerprint": "b", "validation_mean": 0.9},
        {"joint_fingerprint": "a", "validation_mean": 0.9},
        {"joint_fingerprint": "a", "validation_mean": 0.9},
        {"joint_fingerprint": "c", "validation_mean": 0.8},
        {"joint_fingerprint": "d", "validation_mean": 0.7},
    ]
    assert [row["joint_fingerprint"] for row in gate.select_top3_winners(tied)] == ["a", "b", "c"]


# 58: declared and imported local source bytes are actually snapshotted.
def test_58_source_snapshot_contains_declared_and_imported_bytes(tmp_path: Path) -> None:
    required = {
        "configs/b4_arch_diversity_prospective_search_accuracy_gate_v1.yaml",
        "analyse/b4_arch_diversity_prospective_search_accuracy_gate.py",
        "scripts/run_b4_arch_diversity_prospective_search_accuracy_gate.sh",
        "tests/test_b4_arch_diversity_prospective_search_accuracy_gate.py",
        "frozen/direct_joint_regressor_source.py",
        "frozen/high_regime_generator.py",
        "frozen/high_regime_evaluator.py",
        "frozen/architecture_diagnostics.py",
        "hp_modes.py", "dataset_utils.py",
    }
    paths = gate._source_identity_paths()
    assert required <= set(paths)
    assert all(path.is_file() for path in paths.values())
    snapshot = gate.snapshot_sources(tmp_path)
    assert snapshot["actual_bytes_saved"] is True
    assert snapshot["imported_local_module_count"] > 0
    assert snapshot["manifest_root_sha256"]
    assert (tmp_path / "source_snapshot/imported_local_modules.json").is_file()
    assert (tmp_path / "source_snapshot/sha256_manifest.json").is_file()
    assert all((tmp_path / "source_snapshot" / row["relative_path"]).is_file() for row in snapshot["files"])


# 59: result manifest hashes every included file and recomputes exactly.
def test_59_manifest_integrity_recomputes(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    gate.dump_text(output / "a.txt", "a\n")
    gate.dump_json(output / "nested/b.json", {"b": 2})
    written = gate.write_manifest(output)
    verified = gate.verify_directory_manifest(output)
    assert verified["all_file_hashes_valid"] is True
    assert verified["manifest_root_sha256"] == written["manifest_root_sha256"]
    (output / "a.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(gate.InputIdentityInvalid, match="manifest entry mismatch"):
        gate.verify_directory_manifest(output)


# 60: formal result directory/archive targets are never overwritten.
def test_60_formal_result_directory_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(gate.ProtocolError, match="already exists"):
        gate.ensure_output_policy(output, Path(str(output) + ".tar.gz"))


# 61: a RUN_FAILED bundle remains complete, archived, and test-free.
def test_61_failure_bundle_can_be_generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_environment(output: Path, *, before: bool) -> None:
        if before:
            gate.dump_json(output / "environment.json", {"test_accessed": False})
            gate.dump_text(output / "git_status_before.txt", "synthetic\n")
        else:
            gate.dump_text(output / "git_status_after.txt", "synthetic\n")
            gate.dump_text(output / "git_diff.patch", "")

    def fake_snapshot(output: Path, *, best_effort: bool = False) -> dict:
        root = output / "source_snapshot"
        gate.dump_json(root / "imported_local_modules.json", {"modules": []})
        payload = {
            "manifest_root_sha256": "synthetic", "files": [],
            "actual_bytes_saved": False, "best_effort": best_effort,
        }
        gate.dump_json(root / "sha256_manifest.json", payload)
        return payload

    monkeypatch.setattr(gate, "write_environment_and_git", fake_environment)
    monkeypatch.setattr(gate, "snapshot_sources", fake_snapshot)
    output = tmp_path / "failed"
    gate.write_failure_bundle(output, "synthetic failure")
    assert (output / "FAILED").is_file()
    assert not (output / "COMPLETE").exists()
    assert json.loads((output / "decision.json").read_text())["classification"] == "RUN_FAILED"
    assert json.loads((output / "error.json").read_text())["test_accessed"] is False
    assert Path(str(output) + ".tar.gz").is_file()
    assert Path(str(output) + ".tar.gz.sha256").is_file()
    assert (output / "manifest.json").is_file()


# 62: dry-run performs only protocol/static identity checks and creates no formal result.
def test_62_dry_run_creates_no_formal_result_or_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "formal"
    monkeypatch.setattr(gate, "static_input_audit", lambda _config: {"status": "VALID"})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run crossed into a formal/GPU path")

    monkeypatch.setattr(gate, "generate_all_candidates", forbidden)
    monkeypatch.setattr(gate, "score_candidates_with_frozen_b4", forbidden)
    monkeypatch.setattr(gate, "run_frozen_gnn_evaluation", forbidden)
    assert gate.execute_dry_run(gate.DEFAULT_CONFIG, output) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN_READY"
    assert payload["formal_candidate_generation_started"] is False
    assert payload["formal_search_plan_frozen"] is False
    assert payload["GNN_training_started"] is False
    assert payload["formal_output_created"] is False
    assert payload["test_accessed"] is False
    assert not output.exists()


def test_synthetic_smoke_is_cpu_only_and_non_formal() -> None:
    result = gate.synthetic_smoke()
    assert result["passed"] is True
    assert result["formal_candidate_generation_started"] is False
    assert result["formal_search_plan_frozen"] is False
    assert result["GNN_training_started"] is False
    assert result["CUDA_initialized"] is False
    assert result["formal_output_created"] is False
    assert result["test_accessed"] is False
