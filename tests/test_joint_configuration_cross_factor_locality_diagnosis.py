from __future__ import annotations

import json
import math
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSE_ROOT = REPO_ROOT / "analyse"
if str(ANALYSE_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSE_ROOT))

import joint_configuration_cross_factor_locality_diagnosis as diagnosis  # noqa: E402


EXPECTED_INPUT_IDENTITY = {
    "classification": "SOFT_JOINT_METRIC_REJECTED",
    "archive_sha256": "f5a0a73cb45098994afa3d60de44eb4c54c010d6814d963e3d4c3ba69c219f74",
    "config_sha256": "0b97af8a13d270e80fe8cc7b9a4beaf2616ff624c01faf5832fdebbeae25aadd",
    "source_sha256": "d042a158a2c006baecc7c9d1d50dd42683f7b4f4c4dbdfed0c2f2c8bcc3a3aea",
    "source_snapshot_manifest_root": "7125d9d7eececca53fcd00c2d578fb685c1b0d2156de62f4de08b362ffe10273",
    "split_sha256": "b80b1a211a05285121cafe4dabd4c67baddea93606c1913e777fcd34dd79a2ec",
    "unique_joint_configurations": 336,
    "unique_architectures": 192,
    "pool_count": 24,
    "seed_record_count": 1008,
    "training_seeds": [0, 1, 2],
}


def _replay_fixture() -> dict:
    return {
        "distance_gap_spearman": {
            "fold_0": {"seed_0": 0.21, "seed_1": 0.22, "seed_2": 0.23},
        },
        "cross_factor_locality_ratio": {"fold_0": 0.96262},
        "metric_top5_fingerprints": {
            "query-a": ["j-01", "j-02", "j-03", "j-04", "j-05"],
        },
        "pooled_metric_gap_at_5": 0.06259,
    }


def _retrieval_fixture(delta: float = 0.006) -> dict:
    return diagnosis.retrieval_metrics_for_query(
        distances=np.asarray([0.4, 0.1, 0.3, 0.2]),
        reference_fingerprints=["d", "a", "c", "b"],
        performance_gaps=np.asarray([0.080, 0.005, 0.004, 0.020]),
        hp_far=np.asarray([True, True, False, True]),
        architecture_different=np.asarray([True, True, True, True]),
        delta_close=delta,
        k=2,
    )


# 1. Frozen soft-v2 archive/config/source/split identity.
def test_frozen_soft_v2_identity_constants_and_validation() -> None:
    assert diagnosis.FROZEN_INPUT_IDENTITY == EXPECTED_INPUT_IDENTITY
    checked = diagnosis.validate_input_identity_payload(dict(EXPECTED_INPUT_IDENTITY))
    assert checked["classification"] == "SOFT_JOINT_METRIC_REJECTED"
    bad = dict(EXPECTED_INPUT_IDENTITY, split_sha256="0" * 64)
    with pytest.raises(diagnosis.ReplayInputIdentityMismatch):
        diagnosis.validate_input_identity_payload(bad)


# 2. Frozen 336/192/24/1008 identity.
def test_data_cardinality_identity_is_fail_closed() -> None:
    checked = diagnosis.validate_input_identity_payload(dict(EXPECTED_INPUT_IDENTITY))
    assert (
        checked["unique_joint_configurations"],
        checked["unique_architectures"],
        checked["pool_count"],
        checked["seed_record_count"],
    ) == (336, 192, 24, 1008)
    with pytest.raises(diagnosis.ReplayInputIdentityMismatch):
        diagnosis.validate_input_identity_payload(
            dict(EXPECTED_INPUT_IDENTITY, unique_joint_configurations=335)
        )


# 3. Test fields are rejected at the loader firewall.
@pytest.mark.parametrize(
    "forbidden_key",
    ["test_accuracy", "test_mask", "test_labels", "test_predictions"],
)
def test_validation_loader_rejects_test_fields(forbidden_key: str) -> None:
    record = {"validation": 0.81, forbidden_key: "must-not-be-read"}
    with pytest.raises(diagnosis.ProtocolError):
        diagnosis.assert_no_test_fields(record)


# 4. Replay consumes exactly seeds 0, 1 and 2.
def test_replay_seed_vector_is_exactly_zero_one_two() -> None:
    values = {0: 0.80, 1: 0.81, 2: 0.82}
    assert diagnosis.extract_replay_seed_values(values) == [0.80, 0.81, 0.82]
    with pytest.raises(diagnosis.ProtocolError):
        diagnosis.extract_replay_seed_values({0: 0.80, 1: 0.81})


# 5. Stabilization seeds 3..29 cannot be consumed by replay extraction.
def test_replay_extractor_does_not_read_stabilization_seeds() -> None:
    class GuardedSeedMap(Mapping):
        def __init__(self) -> None:
            self.accessed: list[int] = []

        def __getitem__(self, key: object) -> float:
            seed = int(key)
            self.accessed.append(seed)
            if seed > 2:
                raise AssertionError("stabilization seed was accessed")
            return 0.8 + seed / 100.0

        def __iter__(self):
            return iter((0, 1, 2, 3, 29))

        def __len__(self) -> int:
            return 5

    guarded = GuardedSeedMap()
    assert diagnosis.extract_replay_seed_values(guarded) == pytest.approx([0.8, 0.81, 0.82])
    assert guarded.accessed == [0, 1, 2]


# 6. Metric initializations are frozen; no best-seed selection is available.
def test_metric_seeds_are_frozen_and_averaged() -> None:
    assert diagnosis.METRIC_SEEDS == (0, 1, 2)
    matrices = [np.full((2, 3), value) for value in (1.0, 2.0, 6.0)]
    averaged = diagnosis.average_metric_seed_distance_matrices(matrices)
    np.testing.assert_array_equal(averaged, np.full((2, 3), 3.0))


# 7. Formal replay contract is deterministic and CPU-only.
def test_replay_runtime_contract_is_cpu_deterministic() -> None:
    contract = diagnosis.replay_runtime_contract()
    assert contract["device"] == "cpu"
    assert contract["cuda_visible_devices"] == ""
    assert contract["deterministic"] is True
    assert contract["torch_num_threads"] == 1


# 8. Archived top-5 fingerprint order is exact, not set-equivalent.
def test_replay_identity_requires_exact_top5_fingerprint_order() -> None:
    archived = _replay_fixture()
    replayed = _replay_fixture()
    assert diagnosis.assert_replay_identity(replayed, archived, atol=1e-8)["passed"]
    replayed["metric_top5_fingerprints"]["query-a"][0:2] = ["j-02", "j-01"]
    with pytest.raises(diagnosis.ReplayIdentityMismatch):
        diagnosis.assert_replay_identity(replayed, archived, atol=1e-8)


# 9. Archived pooled metric Gap@5 must reproduce within 1e-8.
def test_replay_identity_requires_archived_gap_at_5() -> None:
    archived = _replay_fixture()
    replayed = _replay_fixture()
    replayed["pooled_metric_gap_at_5"] += 0.9e-8
    assert diagnosis.assert_replay_identity(replayed, archived, atol=1e-8)["passed"]
    replayed["pooled_metric_gap_at_5"] += 0.2e-8
    with pytest.raises(diagnosis.ReplayIdentityMismatch):
        diagnosis.assert_replay_identity(replayed, archived, atol=1e-8)


# 10. Archived per-fold G3 ratio (and all Spearman values) are replay gates.
def test_replay_identity_requires_archived_g3_and_spearman() -> None:
    archived = _replay_fixture()
    bad_ratio = _replay_fixture()
    bad_ratio["cross_factor_locality_ratio"]["fold_0"] += 1.1e-8
    with pytest.raises(diagnosis.ReplayIdentityMismatch):
        diagnosis.assert_replay_identity(bad_ratio, archived, atol=1e-8)
    bad_spearman = _replay_fixture()
    bad_spearman["distance_gap_spearman"]["fold_0"]["seed_2"] -= 1.1e-8
    with pytest.raises(diagnosis.ReplayIdentityMismatch):
        diagnosis.assert_replay_identity(bad_spearman, archived, atol=1e-8)


# 11. Architecture-different rate is one because architecture groups are held out.
def test_architecture_difference_rate_is_one_for_grouped_split() -> None:
    assert diagnosis.architecture_difference_rate(
        ["heldout-a", "heldout-b"], ["train-a", "train-b", "train-c"]
    ) == 1.0
    with pytest.raises(diagnosis.ProtocolError):
        diagnosis.assert_architecture_disjoint(
            ["heldout-a", "shared"], ["train-a", "shared"]
        )


# 12. The HP-far scaler and pair median use outer-train HP only.
def test_hp_far_context_uses_only_outer_train() -> None:
    train_hp = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.2, 1.0, 0.5],
            [2.0, 0.4, 2.0, 1.0],
            [3.0, 0.6, 3.0, 1.5],
        ]
    )
    context = diagnosis.fit_hp_far_context(train_hp)
    threshold = context["train_pair_median_distance"]
    query_a = np.asarray([[0.5, 0.1, 0.5, 0.25]])
    query_b = np.asarray([[500.0, 500.0, 500.0, 500.0]])
    diagnosis.hp_far_matrix(query_a, train_hp, context)
    diagnosis.hp_far_matrix(query_b, train_hp, context)
    assert context["train_pair_median_distance"] == threshold


# 13. Absolute performance-gap bins have the frozen inclusive boundaries.
@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (0.0, "gap_le_0.006"),
        (0.006, "gap_le_0.006"),
        (0.0060000001, "gap_0.006_0.012"),
        (0.012, "gap_0.006_0.012"),
        (0.0120000001, "gap_0.012_0.025"),
        (0.025, "gap_0.012_0.025"),
        (0.050, "gap_0.025_0.050"),
        (0.0500000001, "gap_gt_0.050"),
    ],
)
def test_absolute_gap_bin_boundaries(gap: float, expected: str) -> None:
    assert diagnosis.assign_absolute_gap_bin(gap) == expected


# 14. Top-k distance ties are resolved lexicographically by joint fingerprint.
def test_topk_tie_break_is_joint_fingerprint() -> None:
    ranked = diagnosis.stable_distance_order(
        np.asarray([0.2, 0.1, 0.1, 0.3]), ["z", "b", "a", "c"]
    )
    assert ranked.tolist() == [2, 1, 0, 3]
    assert diagnosis.stable_topk_fingerprints(
        np.asarray([0.2, 0.1, 0.1, 0.3]), ["z", "b", "a", "c"], k=2
    ) == ["a", "b"]


# 15. Relevance is architecture-different AND HP-far AND gap <= delta.
def test_relevant_cross_factor_definition() -> None:
    relevant = diagnosis.cross_factor_relevant_mask(
        performance_gaps=np.asarray([0.005, 0.005, 0.013, 0.012]),
        hp_far=np.asarray([True, False, True, True]),
        architecture_different=np.asarray([True, True, True, False]),
        delta_close=0.012,
    )
    np.testing.assert_array_equal(relevant, [True, False, False, False])


# 16. Queries with no opportunity do not enter conditional recall/hit denominators.
def test_no_opportunity_excluded_from_conditional_denominator() -> None:
    eligible = _retrieval_fixture()
    no_opportunity = dict(eligible)
    no_opportunity.update(
        eligible=False,
        relevant_count=0,
        recall=None,
        hit=None,
        conditional_hit=None,
    )
    summary = diagnosis.summarize_retrieval_queries([eligible, no_opportunity])
    assert summary["eligible_query_count"] == 1
    assert summary["no_opportunity_query_count"] == 1
    assert summary["conditional_recall_mean"] == pytest.approx(eligible["recall"])
    assert summary["conditional_hit_rate"] == pytest.approx(eligible["hit"])


# 17. Precision denominator is always k, never the number of relevant items.
def test_precision_denominator_is_k() -> None:
    result = _retrieval_fixture()
    assert result["relevant_retrieved"] == 1
    assert result["k"] == 2
    assert result["precision"] == pytest.approx(0.5)


# 18. Recall denominator is all relevant outer-train references.
def test_recall_denominator_is_all_relevant_references() -> None:
    result = _retrieval_fixture()
    assert result["relevant_count"] == 1
    assert result["relevant_retrieved"] == 1
    assert result["recall"] == pytest.approx(1.0)


# 19. Hit@k is one iff at least one relevant item is retrieved.
def test_hit_at_k_definition() -> None:
    result = _retrieval_fixture()
    assert result["hit"] == 1
    missed = diagnosis.retrieval_metrics_for_query(
        distances=np.asarray([0.1, 0.2, 0.3]),
        reference_fingerprints=["a", "b", "c"],
        performance_gaps=np.asarray([0.02, 0.03, 0.005]),
        hp_far=np.asarray([True, True, True]),
        architecture_different=np.asarray([True, True, True]),
        delta_close=0.006,
        k=2,
    )
    assert missed["eligible"] is True
    assert missed["hit"] == 0


# 20. Relevant base prevalence and precision lift use all train references.
def test_base_prevalence_and_precision_lift() -> None:
    result = _retrieval_fixture()
    assert result["base_prevalence"] == pytest.approx(1 / 4)
    assert result["precision_lift"] == pytest.approx(2.0)


# 21. Relevant ranks are one-based.
def test_relevant_ranks_are_one_based() -> None:
    result = diagnosis.relevant_rank_metrics_for_query(
        distances=np.asarray([0.4, 0.1, 0.2]),
        reference_fingerprints=["c", "a", "b"],
        relevant=np.asarray([True, False, True]),
    )
    assert result["relevant_ranks"] == [2, 3]
    assert result["best_rank"] == 2
    assert result["median_rank"] == pytest.approx(2.5)


# 22. MRR is mean(1 / best_rank) across eligible queries.
def test_mrr_uses_best_rank_per_eligible_query() -> None:
    summary = diagnosis.summarize_relevant_rank_queries(
        [
            {"eligible": True, "best_rank": 1, "median_rank": 3.0},
            {"eligible": True, "best_rank": 4, "median_rank": 6.0},
            {"eligible": False, "best_rank": None, "median_rank": None},
        ]
    )
    assert summary["mrr"] == pytest.approx((1.0 + 0.25) / 2.0)


# 23. Query-level AUC is P(distance_far > distance_close), with half credit ties.
def test_query_level_auc_direction_and_ties() -> None:
    separated = diagnosis.query_distance_discrimination(
        close_distances=np.asarray([0.1, 0.2]),
        far_distances=np.asarray([0.8, 0.9]),
    )
    assert separated["auc"] == pytest.approx(1.0)
    tied = diagnosis.query_distance_discrimination(
        close_distances=np.asarray([0.2]), far_distances=np.asarray([0.2])
    )
    assert tied["auc"] == pytest.approx(0.5)


# 24. Cliff's delta is P(far > close) - P(far < close).
def test_cliffs_delta_has_frozen_direction() -> None:
    good = diagnosis.query_distance_discrimination(
        close_distances=np.asarray([0.1, 0.2]),
        far_distances=np.asarray([0.8, 0.9]),
    )
    reversed_signal = diagnosis.query_distance_discrimination(
        close_distances=np.asarray([0.8, 0.9]),
        far_distances=np.asarray([0.1, 0.2]),
    )
    assert good["cliffs_delta"] == pytest.approx(1.0)
    assert reversed_signal["cliffs_delta"] == pytest.approx(-1.0)


# 25. Bootstrap resamples query summaries, with frozen replicates and seed.
def test_bootstrap_unit_is_heldout_query() -> None:
    result = diagnosis.query_bootstrap_summary([0.1, 0.2, 0.3, 0.4])
    assert result["bootstrap_unit"] == "heldout_query"
    assert result["bootstrap_replicates"] == 2000
    assert result["bootstrap_seed"] == 0
    assert result["query_count"] == 4
    again = diagnosis.query_bootstrap_summary([0.1, 0.2, 0.3, 0.4])
    assert result == again


# 26. Metric and explicit baselines must share identical query/reference cells.
def test_explicit_baseline_must_share_query_reference_cells() -> None:
    metric_cells = [("q1", "r1"), ("q1", "r2"), ("q2", "r3")]
    baseline_cells = [("q1", "r1"), ("q1", "r2"), ("q2", "r3")]
    diagnosis.assert_shared_query_reference_cells(metric_cells, baseline_cells)
    with pytest.raises(diagnosis.ProtocolError):
        diagnosis.assert_shared_query_reference_cells(
            metric_cells, [("q1", "r1"), ("q2", "r3")]
        )


# 27. Replay scope admits encoders and rejects GP/regressor/acquisition training.
@pytest.mark.parametrize(
    "component",
    ["B0", "B1", "B2", "B3", "B4", "M2", "M3", "M4", "direct_regressor", "acquisition"],
)
def test_replay_scope_rejects_forbidden_models(component: str) -> None:
    diagnosis.assert_replay_scope(["metric_encoder"])
    with pytest.raises(diagnosis.ProtocolError):
        diagnosis.assert_replay_scope(["metric_encoder", component])


# 28. Scientific soft-v2 parameters cannot be changed by this diagnosis.
def test_soft_metric_parameters_are_frozen() -> None:
    assert diagnosis.FROZEN_SOFT_METRIC_PARAMETERS == {
        "delta": 0.012,
        "embedding_dim": 8,
        "metric_seeds": [0, 1, 2],
        "outer_folds": 5,
        "hp_features": ["log10(lr)", "dropout", "log2(hidden_dim)", "log10(l2)"],
    }
    diagnosis.validate_frozen_soft_metric_parameters(
        dict(diagnosis.FROZEN_SOFT_METRIC_PARAMETERS)
    )
    with pytest.raises(diagnosis.ProtocolError):
        diagnosis.validate_frozen_soft_metric_parameters(
            dict(diagnosis.FROZEN_SOFT_METRIC_PARAMETERS, delta=0.013)
        )


# 29. Historical REJECTED classification is immutable and no readiness class exists.
def test_decision_preserves_historical_rejected_classification() -> None:
    decision = diagnosis.build_decision("CROSS_FACTOR_DIAGNOSIS_COMPLETE")
    assert decision["classification"] == "CROSS_FACTOR_DIAGNOSIS_COMPLETE"
    assert decision["historical_soft_metric_classification"] == "SOFT_JOINT_METRIC_REJECTED"
    assert decision["historical_classification_modified"] is False
    for forbidden in ("GP_READY", "LOCALITY_READY", "BO_READY"):
        with pytest.raises(diagnosis.ProtocolError):
            diagnosis.build_decision(forbidden)


# 30. A complete report has all sixteen prescribed, non-empty sections.
def test_final_report_requires_all_sixteen_sections() -> None:
    assert len(diagnosis.REPORT_SECTIONS) == 16
    report = "\n\n".join(
        f"## {index}. {title}\n\nNon-empty diagnostic table.\n\n| key | value |\n|---|---:|\n| x | 1 |"
        for index, title in enumerate(diagnosis.REPORT_SECTIONS, start=1)
    )
    diagnosis.assert_complete_final_report(report)
    missing = report.replace("## 11. HP-near vs HP-far locality", "## missing")
    with pytest.raises(diagnosis.ProtocolError):
        diagnosis.assert_complete_final_report(missing)


# 31. Dry-run validates paths/contracts without creating the formal result root.
def test_dry_run_does_not_create_formal_result_directory(tmp_path: Path) -> None:
    formal_root = tmp_path / "formal-result"
    audit = diagnosis.execute_dry_run(formal_output=formal_root)
    assert audit["dry_run"] is True
    assert audit["formal_replay_executed"] is False
    assert not formal_root.exists()


# 32. A non-empty formal directory is never overwritten.
def test_formal_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    formal_root = tmp_path / "formal-result"
    formal_root.mkdir()
    (formal_root / "existing.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(diagnosis.ProtocolError):
        diagnosis.ensure_formal_output_available(formal_root)
    assert (formal_root / "existing.json").read_text(encoding="utf-8") == "{}\n"


# 33. A failure bundle is self-contained and archived without deleting evidence.
def test_failure_bundle_is_generated_and_archived(tmp_path: Path) -> None:
    output_root = tmp_path / "failed-run"
    archive_path = tmp_path / "failed-run.tar.gz"
    result = diagnosis.write_failure_bundle(
        output_root=output_root,
        error=RuntimeError("synthetic failure"),
        archive_path=archive_path,
    )
    assert (output_root / "FAILED").is_file()
    error = json.loads((output_root / "error.json").read_text(encoding="utf-8"))
    decision = json.loads((output_root / "decision.json").read_text(encoding="utf-8"))
    assert error["error_type"] == "RuntimeError"
    assert decision["classification"] == "RUN_FAILED"
    assert decision["historical_soft_metric_classification"] == "SOFT_JOINT_METRIC_REJECTED"
    assert archive_path.is_file()
    assert Path(result["archive_path"]) == archive_path
    with tarfile.open(archive_path, "r:gz") as bundle:
        names = set(bundle.getnames())
    assert any(name.endswith("/FAILED") for name in names)
    assert any(name.endswith("/error.json") for name in names)


def test_query_discrimination_requires_both_close_and_far() -> None:
    with pytest.raises(diagnosis.ProtocolError):
        diagnosis.query_distance_discrimination(
            close_distances=np.asarray([]), far_distances=np.asarray([0.8])
        )


def test_gap_bin_monotonicity_is_diagnostic_not_a_gate() -> None:
    summary = diagnosis.adjacent_bin_median_increases([0.1, 0.2, 0.15, 0.3, 0.4])
    assert summary == {"increasing_transitions": 3, "total_transitions": 4}
    assert "passed" not in summary


def test_source_snapshot_contract_includes_frozen_v2_sources() -> None:
    required = set(diagnosis.REQUIRED_SOURCE_SNAPSHOT_PATHS)
    assert {
        "analyse/joint_configuration_soft_metric_gate.py",
        "analyse/joint_configuration_metric_gate.py",
        "analyse/performance_metric_quick_gate.py",
    }.issubset(required)
