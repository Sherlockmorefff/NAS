from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from architecture_diagnostics import architecture_fingerprint
from analyse.performance_aware_vae_strict_regression import StrictPerformanceHead
import analyse.direct_head_chronological_offline_replay as replay


def _history_source(*, seed: int = 0, step: int = 0, hp: str = "hp0") -> dict:
    row = {
        "search_seed": seed,
        "step": step,
        "operations": ["GCNConv", "SAGEConv"],
        "edges": [[0, 1], [1, 2]],
        "z_search": [float(index) / 10.0 for index in range(16)],
        "z_arch": [float(index) / 10.0 for index in range(12)],
        "hp_fingerprint": hp,
        "condition_mask": {"gat_heads": False},
        "condition_mask_vector": [1.0, 0.0, 1.0, 0.0],
        "lr": 0.01,
        "dropout": 0.2,
        "hidden_dim": 64,
        "l2": 0.0005,
        "gat_heads": 1,
        "sage_aggr": "mean",
        "gin_eps": 0.0,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": ["mean"],
        "gin_eps_by_layer": None,
        "normalization": "none",
        "val_acc": 0.75,
        "value": 0.75,
        "gp_pred_mean": 0.8,
        "performance_rank": 1,
    }
    row["architecture_fingerprint"] = architecture_fingerprint(row)
    return row


def _ranking_rows() -> list[dict]:
    return [
        {
            "replay_id": f"r{index}",
            "search_seed": 0,
            "step": index,
            "architecture_fingerprint": f"a{index}",
            "hp_fingerprint": f"h{index}",
            "validation": value,
            "prediction": prediction,
            "baseline": baseline,
        }
        for index, (value, prediction, baseline) in enumerate(
            [
                (0.90, 0.88, 0.10),
                (0.85, 0.80, 0.20),
                (0.75, 0.70, 0.30),
                (0.60, 0.20, 0.80),
                (0.50, 0.10, 0.90),
            ]
        )
    ]


def test_refit_head_clean24_reproduction_with_frozen_values(monkeypatch, tmp_path):
    clean = [
        {
            "config_id": f"c{index}",
            "architecture_fingerprint": f"a{index}",
            "raw_hp_vector": [0.0] * 4,
            "mu_arch": [0.0] * 12,
        }
        for index in range(24)
    ]
    monkeypatch.setattr(replay, "_load_clean24", lambda: clean)
    monkeypatch.setattr(replay, "encode_selected_mu", lambda rows: list(rows))
    monkeypatch.setattr(
        replay,
        "frozen_refit_predictions",
        lambda rows: (np.asarray([0.70 + index / 1000 for index in range(24)]), []),
    )
    monkeypatch.setattr(
        replay,
        "frozen_head_h_predictions",
        lambda rows, root: [
            {
                "config_id": row["config_id"],
                "head_prediction": 0.60 + index / 1000,
                "h_prediction": 0.65 + index / 1000,
            }
            for index, row in enumerate(rows)
        ],
    )
    expected = []
    for index, row in enumerate(clean):
        expected.extend(
            [
                {"config_id": row["config_id"], "model": "refit_head", "prediction": 0.70 + index / 1000},
                {"config_id": row["config_id"], "model": "original_frozen_head", "prediction": 0.60 + index / 1000},
                {"config_id": row["config_id"], "model": "original_frozen_h", "prediction": 0.65 + index / 1000},
            ]
        )
    monkeypatch.setattr(replay, "load_json", lambda path: expected)
    decision = replay.reproduce_clean24_predictions(tmp_path)
    assert decision["classification"] == "PHASE_1_PASSED"
    assert decision["maximum_absolute_error"] == 0.0


def test_refit_head_inference_preserves_checkpoint_hash(tmp_path):
    checkpoint = tmp_path / "refit.pth"
    head = StrictPerformanceHead(16)
    payload = {
        "model_name": "refit_head",
        "feature_schema": ["mu_arch", "raw_hp_vector"],
        "vae_trained": False,
        "latent_distance_gp_used": False,
        "head_input_dim": 16,
        "head_state": head.state_dict(),
        "scaling": {
            "feature_mean": [0.0] * 16,
            "feature_std": [1.0] * 16,
            "y_mean": 0.75,
            "y_std": 0.05,
        },
    }
    torch.save(payload, checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    tasks = [
        {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": before,
            "split_id": "group_000",
            "training_seed": 0,
        }
    ]
    rows = [{"mu_arch": [0.0] * 12, "raw_hp_vector": [0.0] * 4}]
    predicted, identities = replay.frozen_refit_predictions(rows, tasks)
    after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert predicted.shape == (1,)
    assert identities[0]["feature_schema"] == ["mu_arch", "raw_hp_vector"]
    assert before == after


def test_exclusion_sources_cover_development_clean_and_quarantine():
    sources = replay.exclusion_sources()
    assert len(sources["development_80"]) == 80
    assert len(sources["clean_prospective_24"]) == 24
    assert len(sources["quarantined_24"]) == 24
    assert sources["refit_training_tuning_calibration"] == sources["development_80"]
    assert not (sources["development_80"] & sources["clean_prospective_24"])
    assert not (sources["development_80"] & sources["quarantined_24"])
    assert not (sources["clean_prospective_24"] & sources["quarantined_24"])


def test_projection_masks_labels_and_uses_historical_architecture(tmp_path):
    source = _history_source()
    path = tmp_path / "history.json"
    path.write_text(json.dumps([source]), encoding="utf-8")
    projected = replay._project_history_record(source, source_path=path, source_index=0)
    replay.assert_projection_is_label_free([projected])
    assert projected["operations"] == source["operations"]
    assert projected["edges"] == source["edges"]
    assert projected["architecture_source"] == "historical_operations_and_edges_without_decode"
    assert projected["raw_hp_vector"] == source["z_search"][12:16]
    assert "val_acc" not in projected
    assert "gp_pred_mean" not in projected


def test_projection_rejects_label_field():
    with pytest.raises(ValueError, match="leaked"):
        replay.assert_projection_is_label_free([{"architecture_fingerprint": "a", "val_acc": 0.8}])


def test_actual_histories_keep_first_architecture_hp_configuration_only():
    projection, audits, _excluded = replay.project_histories()
    keys = [(row["search_seed"], row["architecture_fingerprint"], row["hp_fingerprint"]) for row in projection]
    assert len(keys) == len(set(keys))
    assert set(audits) == set(range(5))
    assert all(row["retained_architecture_count"] >= 100 for row in audits.values())


def test_same_architecture_different_hp_is_one_bootstrap_cluster():
    rows = [
        {"replay_id": "a-h1", "architecture_fingerprint": "a", "hp_fingerprint": "h1"},
        {"replay_id": "a-h2", "architecture_fingerprint": "a", "hp_fingerprint": "h2"},
        {"replay_id": "b-h1", "architecture_fingerprint": "b", "hp_fingerprint": "h1"},
    ]
    sampled = replay._bootstrap_rows(rows, ["a", "a", "b"])
    assert len(sampled) == 5
    assert sum(row["architecture_fingerprint"] == "a" for row in sampled) == 4


def test_time_prefixes_use_original_clock_and_no_future_rows():
    rows = [
        {"step": step, "architecture_fingerprint": f"a{step}", "hp_fingerprint": "h"}
        for step in (5, 99, 100, 399, 400, 449, 450, 499)
    ]
    assert [row["step"] for row in replay.rows_for_window(rows, "prefix_20")] == [5, 99]
    assert max(row["step"] for row in replay.rows_for_window(rows, "prefix_80")) < 400
    assert [row["step"] for row in replay.rows_for_window(rows, "late_50")] == [450, 499]
    assert [row["step"] for row in replay.rows_for_window(rows, "late_100")] == [400, 449, 450, 499]


def test_top_k_ndcg_regret_and_false_rates():
    metrics = replay.ranking_metric_bundle(_ranking_rows())
    assert metrics["kendall"] > 0
    assert metrics["ndcg_at_5"] > 0.9
    assert metrics["recall_at_1"] == 1.0
    assert metrics["precision_at_3"] == 1.0
    assert metrics["regret_at_1"] == 0.0
    assert metrics["regret_at_5"] == 0.0
    assert 0.0 <= metrics["false_positive_rate"] <= 1.0
    assert 0.0 <= metrics["false_negative_rate"] <= 1.0


def test_three_gap_threshold_pairwise_metrics_are_present():
    metrics = replay.ranking_metric_bundle(_ranking_rows())
    for suffix in ("0_006", "0_012", "0_020"):
        assert f"pairwise_accuracy_gap_{suffix}" in metrics
        assert f"pair_count_gap_{suffix}" in metrics


def test_fixed_random_baseline_is_reproducible():
    rows = _ranking_rows()
    first = replay.fixed_random_scores(rows, seed=123)
    second = replay.fixed_random_scores(rows, seed=123)
    third = replay.fixed_random_scores(rows, seed=124)
    assert first == second
    assert first != third


def test_shared_bootstrap_keeps_five_search_seeds_equal_weight():
    rows_by_seed = {}
    for seed in range(5):
        rows = []
        for row in _ranking_rows():
            rows.append({**row, "search_seed": seed, "replay_id": f"s{seed}-{row['replay_id']}"})
        if seed == 4:
            rows.extend(
                {
                    **row,
                    "search_seed": seed,
                    "replay_id": f"s{seed}-extra-{row['replay_id']}",
                    "architecture_fingerprint": f"extra-{row['architecture_fingerprint']}",
                }
                for row in _ranking_rows()
            )
        rows_by_seed[seed] = rows
    audit = replay.shared_cluster_bootstrap(
        rows_by_seed,
        candidate_key="prediction",
        baseline_key="baseline",
        metric="kendall",
        replicates=20,
        seed=7,
    )
    assert audit["bootstrap_valid_replicates"] == 20
    assert audit["bootstrap_unit"].startswith("search-seed-stratified")
    assert audit["difference_ci95_low"] > 0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"all_success_gates": True}, "DIRECT_HEAD_VALIDATED_FOR_RANKING_LOSS_PILOT"),
        ({"signal_only": True}, "DIRECT_HEAD_SIGNAL_ONLY"),
        ({}, "DIRECT_HEAD_REPLAY_REJECTED"),
        ({"underpowered": True}, "DIRECT_HEAD_REPLAY_UNDERPOWERED"),
        ({"protocol_incompatible": True}, "REPLAY_LABEL_PROTOCOL_INCOMPATIBLE"),
        ({"insufficient_unseen": True}, "INSUFFICIENT_UNSEEN_HISTORY"),
        ({"insufficient_coverage": True}, "INSUFFICIENT_REPLAY_COVERAGE"),
        ({"reproduction_failed": True}, "FROZEN_MODEL_REPRODUCTION_FAILED"),
        ({"input_or_implementation_error": True}, "INPUT_OR_IMPLEMENTATION_ERROR"),
    ],
)
def test_all_nine_final_classification_branches(kwargs, expected):
    assert replay.classify_replay(**kwargs) == expected


def test_source_has_no_training_gp_or_bo_path():
    source = inspect.getsource(replay)
    assert "GaussianProcessRegressor(" not in source
    assert "_fixed_gp_predict(" not in source
    assert ".fit(" not in source
    assert "train_and_eval_arch" not in source
    assert "import bo_phase4" not in source


def test_evaluate_refuses_terminal_prelabel_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(
        replay,
        "verify_frozen_plan",
        lambda path: {"terminal_pre_label_classification": "INSUFFICIENT_REPLAY_COVERAGE"},
    )
    with pytest.raises(RuntimeError, match="firewall remains closed"):
        replay.evaluate(tmp_path / "plan.json", tmp_path)
