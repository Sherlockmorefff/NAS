from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

import analyse.original_head_uncertainty_selective_risk_audit as audit


def _row(
    config_id: str,
    fingerprint: str,
    stratum: str,
    sigma: float,
    prediction: float,
    seeds: list[float],
) -> dict:
    return {
        "config_id": config_id,
        "architecture_fingerprint": fingerprint,
        "hp_fingerprint": f"hp-{config_id}",
        "structure_stratum": stratum,
        "member_prediction_sample_std_ddof1": sigma,
        "prediction": prediction,
        "seed_values": seeds,
        "mean_val_acc": float(np.mean(seeds)),
    }


def test_finite_sample_conformal_ranks_for_96_configurations():
    scores = list(range(1, 97))
    q90, rank90 = audit.finite_conformal_quantile(scores, 0.90)
    q95, rank95 = audit.finite_conformal_quantile(scores, 0.95)
    assert (q90, rank90) == (88.0, 88)
    assert (q95, rank95) == (93.0, 93)


def test_member_uncertainty_is_ddof1_sample_standard_deviation():
    values = [0.70, 0.72, 0.76]
    expected = float(np.std(values, ddof=1))
    assert audit.member_prediction_sample_std(values) == expected
    legacy_total = float(np.sqrt(np.mean(np.asarray(values) ** 2 + 0.1**2) - np.mean(values) ** 2))
    assert legacy_total != expected


def test_external_firewall_rejects_label_and_test_named_fields():
    safe = [{"config_id": "c", "prediction": 0.7, "member_prediction_sample_std_ddof1": 0.02}]
    assert audit.contains_forbidden_external_key(safe) == []
    hits = audit.contains_forbidden_external_key({"val_acc": 0.7, "test_score": 0.8})
    assert len(hits) == 2


def test_random_rejection_permutations_are_seeded_and_complete():
    left = audit.frozen_random_permutations(12, 20, 20260832)
    right = audit.frozen_random_permutations(12, 20, 20260832)
    assert np.array_equal(left, right)
    assert left.shape == (20, 12)
    assert all(sorted(map(int, row)) == list(range(12)) for row in left)


def test_new48_bootstrap_keeps_dual_hp_cluster_and_structure_counts():
    rows = [
        _row("a-0", "a", "sparse", 0.02, 0.7, [0.70, 0.71, 0.72]),
        _row("a-1", "a", "sparse", 0.03, 0.71, [0.69, 0.70, 0.71]),
        _row("b-0", "b", "sparse", 0.04, 0.72, [0.72, 0.73, 0.74]),
        _row("c-0", "c", "dense", 0.05, 0.73, [0.73, 0.74, 0.75]),
    ]
    boot = audit._new48_bootstrap_rows(rows, np.random.default_rng(7))
    copied = {}
    for row in boot:
        source = row["config_id"].split("#bootstrap", 1)[0]
        copied[source] = copied.get(source, 0) + 1
    assert copied.get("a-0", 0) == copied.get("a-1", 0)
    sparse_fps = {
        row["architecture_fingerprint"] for row in boot if row["structure_stratum"] == "sparse"
    }
    dense_fps = {
        row["architecture_fingerprint"] for row in boot if row["structure_stratum"] == "dense"
    }
    assert len(sparse_fps) == 2
    assert len(dense_fps) == 1
    assert all("#bootstrap" in row["architecture_fingerprint"] for row in boot)


def test_interval_score_penalizes_misses_without_clipping():
    y = np.asarray([0.5, 0.9])
    low = np.asarray([0.4, 0.6])
    high = np.asarray([0.6, 0.8])
    scores = audit.interval_score(y, low, high, alpha=0.10)
    assert scores[0] == pytest.approx(0.2)
    assert scores[1] == pytest.approx(2.2)


def test_stable_boundary_order_uses_fingerprints_after_prediction_tie():
    rows = [
        _row("z", "z-fp", "s", 0.1, 0.8, [0.7, 0.7, 0.7]),
        _row("a", "a-fp", "s", 0.1, 0.8, [0.7, 0.7, 0.7]),
    ]
    assert audit._stable_order(rows, "prediction") == [1, 0]


def test_classification_success_and_rejection_priority():
    new_metrics = {
        "std_abs_error_spearman": 0.4,
        "uncertain_top25_minus_remaining75_mae": 0.02,
        "hetero_90_coverage": 0.90,
        "hetero_95_coverage": 0.95,
        "hetero_90_interval_score": 0.1,
        "fixed_90_interval_score": 0.2,
        "hetero_95_interval_score": 0.1,
        "fixed_95_interval_score": 0.2,
        "exclude_25_mae": 0.03,
        "full_mae": 0.04,
        "exclude_25_kendall_tau_b": 0.5,
        "full_kendall_tau_b": 0.4,
        "exclude_25_false_positive_rate": 0.0,
        "full_false_positive_rate": 0.2,
    }
    clean = {"std_abs_error_spearman": 0.1}
    intervals = {
        ("new48", "std_abs_error_spearman"): {"ci95_low": 0.1, "ci95_high": 0.7},
        ("new48", "uncertain_top25_minus_remaining75_mae"): {"ci95_low": 0.01, "ci95_high": 0.03},
    }
    random = [{"exclusion_rate": 0.25, "metric": "mae", "q05": 0.035}]
    classification, gates = audit.classify(new_metrics, clean, intervals, random, True)
    assert classification == "HEAD_UNCERTAINTY_VALIDATED_FOR_ACQUISITION_PILOT"
    assert all(gates.values())
    rejected_metrics = {**new_metrics, "hetero_90_coverage": 0.70}
    classification, _ = audit.classify(rejected_metrics, clean, intervals, random, True)
    assert classification == "HEAD_UNCERTAINTY_REJECTED"


def test_prepare_source_cannot_connect_external_labels_or_train_models():
    source = inspect.getsource(audit.prepare)
    assert "connect_external_labels" not in source
    module_source = inspect.getsource(audit)
    assert ".fit(" not in module_source
    assert "train_and_eval_arch" not in module_source
    assert "GaussianProcessRegressor" not in module_source


def test_main_records_frozen_identity_mismatch_before_label_join(tmp_path, monkeypatch):
    output = tmp_path / "audit"

    def fail_identity(_plan):
        raise audit.InputIdentityMismatch("deliberate frozen-input drift")

    monkeypatch.setattr(audit, "verify_plan", fail_identity)
    exit_code = audit.main([
        "evaluate", "--plan", str(tmp_path / "plan.json"), "--output", str(output),
    ])
    final = json.loads((output / "final_classification.json").read_text(encoding="utf-8"))
    assert exit_code == 2
    assert final["classification"] == "INPUT_IDENTITY_MISMATCH"
    assert final["external_validation_labels_connected"] is False


def test_main_records_unhandled_implementation_error(tmp_path, monkeypatch):
    output = tmp_path / "audit"

    def fail_prepare(_output):
        raise ValueError("deliberate implementation failure")

    monkeypatch.setattr(audit, "prepare", fail_prepare)
    exit_code = audit.main(["prepare", "--output", str(output)])
    final = json.loads((output / "final_classification.json").read_text(encoding="utf-8"))
    assert exit_code == 2
    assert final["classification"] == "INPUT_OR_IMPLEMENTATION_ERROR"
    assert final["evidence"]["error_type"] == "ValueError"
