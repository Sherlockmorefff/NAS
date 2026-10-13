from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata

import analyse.direct_head_chronological_offline_replay_v2 as replay


def _rows() -> list[dict]:
    values = [0.9, 0.85, 0.7, 0.6, 0.5]
    predictions = [0.8, 0.8, 0.7, 0.2, 0.1]
    return [
        {
            "replay_id": f"r{index}",
            "search_seed": 0,
            "step": index,
            "architecture_fingerprint": f"a{index}",
            "hp_fingerprint": f"h{index}",
            "val_acc": values[index],
            "prediction": predictions[index],
        }
        for index in range(5)
    ]


def test_actual_v1_artifacts_and_expected_counts_are_intact():
    audit = replay.verify_v1(replay.REPO_ROOT / replay.V1_RELATIVE)
    assert audit["passed"], audit["differences"]
    assert audit["observed_counts"] == audit["expected_counts"]
    assert set(audit["coverage_by_mandatory_predictor"]) == set(replay.MANDATORY_PREDICTORS)
    assert all(value == 1.0 for value in audit["coverage_by_mandatory_predictor"].values())
    assert audit["historical_validation_connected"] is False


def test_v2_amendment_makes_zs_optional_without_coverage_failure():
    amendment = replay.protocol_amendment()
    zs = amendment["optional_models"]["original_zs_fixed_surrogate"]
    assert zs["status"] == "NOT_AVAILABLE_LEGALLY"
    assert zs["included_in_common_coverage"] is False
    assert zs["triggers_insufficient_coverage"] is False
    assert "original_zs_fixed_surrogate" not in replay.MANDATORY_PREDICTORS


def test_random_rankings_are_complete_deterministic_and_label_free(tmp_path):
    projection = []
    for seed in range(5):
        for step in (50, 150, 250, 350, 425, 475):
            projection.append({
                "replay_id": f"s{seed}-{step}", "search_seed": seed, "step": step,
                "architecture_fingerprint": f"a{seed}-{step}", "hp_fingerprint": "h",
            })
    first = replay.freeze_random_rankings(projection, tmp_path / "first")
    second = replay.freeze_random_rankings(projection, tmp_path / "second")
    first_arrays = np.load(first["arrays_path"], allow_pickle=False)
    second_arrays = np.load(second["arrays_path"], allow_pickle=False)
    assert set(first_arrays.files) == set(second_arrays.files)
    assert len(first_arrays.files) == 35
    for key in first_arrays.files:
        assert first_arrays[key].shape[0] == 1000
        assert np.array_equal(first_arrays[key], second_arrays[key])
    index = replay.load_json(Path(first["index_path"]))
    assert replay.v1.recursively_named(index, replay.FORBIDDEN_PROJECTION_PARTS) == []


def test_history_join_reads_val_acc_and_not_value(monkeypatch, tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text("placeholder", encoding="utf-8")
    source = {
        "search_seed": 0, "step": 7, "architecture_fingerprint": "a", "hp_fingerprint": "h",
        "val_acc": 0.71, "value": 0.99,
    }
    projection = [{
        "replay_id": "r", "search_seed": 0, "step": 7,
        "architecture_fingerprint": "a", "hp_fingerprint": "h",
        "source_history_path": str(history_path), "source_index": 0,
    }]
    joined, audit = replay.join_historical_validation(projection, history_loader=lambda path: [source])
    assert audit["passed"]
    assert audit["label_field_read"] == "val_acc"
    assert audit["value_alias_read"] is False
    assert joined[0]["val_acc"] == 0.71


def test_time_windows_use_unfiltered_raw_step_clock():
    rows = [
        {"step": step, "architecture_fingerprint": f"a{step}", "hp_fingerprint": "h"}
        for step in (99, 100, 399, 400, 449, 450, 499)
    ]
    assert [row["step"] for row in replay.rows_for_window(rows, "prefix_20")] == [99]
    assert [row["step"] for row in replay.rows_for_window(rows, "late_100")] == [400, 449, 450, 499]
    assert [row["step"] for row in replay.rows_for_window(rows, "late_50")] == [450, 499]


def test_average_rank_ties_and_complete_metrics():
    rows = _rows()
    order = replay._stable_order(rows, "prediction")
    average = rankdata(-np.asarray([row["prediction"] for row in rows]), method="average")
    assert average[0] == average[1] == 1.5
    assert order[:2] == [0, 1]
    metrics = replay.ranking_metrics(rows)
    assert metrics["ranking_tie_method"] == "average"
    for key in (
        "kendall_tau_b", "top_region_kendall_tau_b", "ndcg_at_10",
        "top_1_overlap_rate", "top_3_any_hit", "top_3_any_hit_rate", "regret_at_10",
        "false_positive_rate", "false_negative_rate",
        "gap_filtered_pairwise_accuracy_0_006",
        "gap_filtered_pairwise_accuracy_0_012",
        "gap_filtered_pairwise_accuracy_0_020",
    ):
        assert key in metrics
    assert metrics["top_3_any_hit_rate"] in (0.0, 1.0)


def test_bootstrap_metric_registry_covers_every_required_family():
    metrics = set(replay.BOOTSTRAP_METRICS)
    assert {f"top_{k}_overlap_rate" for k in replay.TOP_K_VALUES} <= metrics
    assert {f"top_{k}_any_hit_rate" for k in replay.TOP_K_VALUES} <= metrics
    assert {f"regret_at_{k}" for k in replay.TOP_K_VALUES} <= metrics
    assert {"false_positive_rate", "false_negative_rate", "ndcg_at_10"} <= metrics
    assert {
        "gap_filtered_pairwise_accuracy_0_006",
        "gap_filtered_pairwise_accuracy_0_012",
        "gap_filtered_pairwise_accuracy_0_020",
    } <= metrics


def test_same_fingerprint_hp_rows_form_one_cluster():
    rows = [
        {"replay_id": "a1", "architecture_fingerprint": "a", "hp_fingerprint": "h1"},
        {"replay_id": "a2", "architecture_fingerprint": "a", "hp_fingerprint": "h2"},
        {"replay_id": "b1", "architecture_fingerprint": "b", "hp_fingerprint": "h1"},
    ]
    sampled = replay._cluster_sample(rows, ["a", "a", "b"])
    assert len(sampled) == 5
    assert sum(row["architecture_fingerprint"] == "a" for row in sampled) == 4


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"input_error": True}, "INPUT_OR_IMPLEMENTATION_ERROR"),
        ({"reproduction_pass": False}, "FROZEN_MODEL_REPRODUCTION_FAILED"),
        ({"enough_unseen": False}, "INSUFFICIENT_UNSEEN_HISTORY"),
        ({"protocol_compatible": False}, "REPLAY_LABEL_PROTOCOL_INCOMPATIBLE"),
        ({"enough_coverage": False}, "INSUFFICIENT_REPLAY_COVERAGE"),
        ({
            "refit_kendall_low": 0.01, "refit_kendall_high": 0.3,
            "refit_top_low": 0.01, "delta": 0.06, "delta_low": 0.01,
            "positive_seed_count": 4, "stable_prefixes": True,
            "ndcg_not_lower": True, "regret_not_higher": True,
        }, "DIRECT_HEAD_VALIDATED_FOR_RANKING_LOSS_PILOT"),
        ({"refit_kendall_high": 0.0}, "DIRECT_HEAD_REPLAY_REJECTED"),
        ({"refit_kendall_low": 0.01, "refit_kendall_high": 0.2}, "DIRECT_HEAD_SIGNAL_ONLY"),
        ({"refit_kendall_low": -0.1, "refit_kendall_high": 0.2, "delta": 0.02, "delta_low": -0.1, "delta_high": 0.1}, "DIRECT_HEAD_REPLAY_UNDERPOWERED"),
    ],
)
def test_all_classification_branches(kwargs, expected):
    classification, _ = replay.classify(**kwargs)
    assert classification == expected


def test_source_has_no_training_gp_or_bo_execution_path():
    source = inspect.getsource(replay)
    assert "GaussianProcessRegressor(" not in source
    assert "_fixed_gp_predict(" not in source
    assert ".fit(" not in source
    assert "train_and_eval_arch" not in source
    assert "import bo_phase4" not in source
