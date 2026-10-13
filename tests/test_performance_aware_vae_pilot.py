from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analyse import performance_aware_vae_pilot as pilot
from analyse import evaluate_performance_aware_vae_pilot as evaluation


REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = REPO_ROOT / "results/posthoc/vae-label-expansion-global4-v1/unified_training_data_no_test.json"
CHECKPOINT = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"


@pytest.fixture(scope="module")
def labels():
    return pilot.load_validation_labels(LABELS)


def test_architecture_group_folds_have_zero_leakage_and_full_coverage(labels):
    folds = pilot.build_nested_folds(labels)
    observed = []
    for fold in folds:
        outer_test = set(fold["outer_test_architectures"])
        outer_train = set(fold["outer_train_architectures"])
        inner_train = set(fold["inner_train_architectures"])
        inner_val = set(fold["inner_validation_architectures"])
        assert not outer_test & outer_train
        assert not inner_train & inner_val
        assert inner_train | inner_val == outer_train
        observed.extend(outer_test)
    assert len(observed) == len(set(observed)) == 60


def test_same_architecture_multiple_hps_have_balanced_total_weight(labels):
    weights = pilot.training_weights(labels, use_noise_weight=False)
    totals = {}
    for row, weight in zip(labels, weights):
        totals.setdefault(row["architecture_fingerprint"], 0.0)
        totals[row["architecture_fingerprint"]] += weight
    assert max(totals.values()) - min(totals.values()) < 1e-10
    assert weights.mean() == pytest.approx(1.0)


def test_performance_head_requires_and_uses_four_hyperparameters():
    pilot.set_all_seeds(123)
    head = pilot.PerformanceHead()
    mu = torch.zeros((2, 12))
    hp_a = torch.zeros((2, 4))
    hp_b = hp_a.clone()
    hp_b[1, 0] = 1.0
    with torch.no_grad():
        observed = head(mu, hp_b)
    assert observed.shape == (2,)
    assert observed[0] != observed[1]
    with pytest.raises(ValueError, match="four standardized global4"):
        head(mu, torch.zeros((2, 3)))


def test_noise_weight_is_bounded_and_monotone():
    assert pilot.noise_factor(0.001) == pytest.approx(1.0)
    assert pilot.noise_factor(0.006) == pytest.approx(1.0)
    assert 0.1 <= pilot.noise_factor(0.02) < 1.0
    assert pilot.noise_factor(100.0) == pytest.approx(0.1)


def test_reliable_pairs_exclude_same_arch_small_gap_and_overlapping_ci():
    base = {
        "validation_ci95_low": 0.70,
        "validation_ci95_high": 0.71,
    }
    rows = [
        {**base, "architecture_fingerprint": "a", "validation_mean": 0.705},
        {"architecture_fingerprint": "b", "validation_mean": 0.73,
         "validation_ci95_low": 0.72, "validation_ci95_high": 0.74},
        {"architecture_fingerprint": "a", "validation_mean": 0.75,
         "validation_ci95_low": 0.74, "validation_ci95_high": 0.76},
        {"architecture_fingerprint": "c", "validation_mean": 0.708,
         "validation_ci95_low": 0.702, "validation_ci95_high": 0.715},
    ]
    pairs = pilot.reliable_pairs(rows)
    assert (0, 1, -1) in pairs
    assert not any({left, right} == {0, 2} for left, right, _ in pairs)
    assert not any({left, right} == {0, 3} for left, right, _ in pairs)


def test_test_named_fields_are_rejected(tmp_path):
    assert pilot.contains_test_field([{"validation_mean": 0.8}]) is False
    assert pilot.contains_test_field([{"nested": {"test_acc": 0.8}}]) is True
    contaminated = json.loads(LABELS.read_text())
    contaminated[0]["test_metric"] = 0.9
    path = tmp_path / "contaminated.json"
    path.write_text(json.dumps(contaminated), encoding="utf-8")
    with pytest.raises(ValueError, match="test-named"):
        pilot.load_validation_labels(path)


def test_original_checkpoint_load_is_read_only():
    before = pilot.sha256_file(CHECKPOINT)
    model = pilot.load_joint_vae(CHECKPOINT, "global4", device="cpu")
    del model
    assert pilot.sha256_file(CHECKPOINT) == before == pilot.PROTECTED_CHECKPOINT_SHA256


def test_tasks_are_unique_and_evenly_sharded():
    tasks = pilot._make_tasks()
    assert len(tasks) == 150
    assert len({task["task_id"] for task in tasks}) == len(tasks)
    assert sum(task["assigned_shard"] == 0 for task in tasks) == 75
    assert sum(task["assigned_shard"] == 1 for task in tasks) == 75


def test_inner_strength_selection_is_aggregated_across_three_seeds():
    rows = []
    for task in pilot._make_tasks():
        strength = float(task["performance_strength"])
        pairwise = 0.5
        if task["variant"] != "original_checkpoint_probe":
            pairwise = 0.8 if strength == 0.3 else 0.6
            if task["training_seed"] == pilot.TRAINING_SEEDS[0] and strength == 3.0:
                pairwise = 0.9
        rows.append(
            {
                **task,
                "inner_metrics": {
                    "reliable_pairwise_accuracy": pairwise,
                    "kendall": pairwise,
                    "rmse": 1.0 - pairwise,
                },
            }
        )
    selected = evaluation.select_inner_candidates(rows)
    assert len(selected) == 90
    fine_tuned = [row for row in selected if row["variant"] != "original_checkpoint_probe"]
    assert {row["performance_strength"] for row in fine_tuned} == {0.3}


def test_primary_metrics_weight_architectures_equally():
    rows = []
    for index in range(10):
        rows.append(
            {
                "architecture_fingerprint": "many-hp", "config_id": f"a{index}",
                "validation_mean": 1.0, "prediction": 0.0,
                "validation_ci95_low": 0.99, "validation_ci95_high": 1.0,
                "prediction_interval_low": -1.0, "prediction_interval_high": 1.0,
                "time_index": index,
            }
        )
    rows.append(
        {
            "architecture_fingerprint": "one-hp", "config_id": "b",
            "validation_mean": 0.0, "prediction": 0.0,
            "validation_ci95_low": 0.0, "validation_ci95_high": 0.01,
            "prediction_interval_low": -1.0, "prediction_interval_high": 1.0,
            "time_index": 20,
        }
    )
    metrics = evaluation.metric_bundle(rows)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))


def test_frozen_plan_verification_detects_input_drift(tmp_path):
    source = tmp_path / "input.txt"
    code = tmp_path / "code.py"
    folds = tmp_path / "folds.json"
    protected = tmp_path / "protected.bin"
    for path, content in ((source, "a"), (code, "b"), (folds, "[]"), (protected, "c")):
        path.write_text(content, encoding="utf-8")
    plan_path = tmp_path / "task_plan.json"
    plan = {
        "input_hashes": {str(source): pilot.sha256_file(source)},
        "code_hashes": {str(code): pilot.sha256_file(code)},
        "folds_file": str(folds), "folds_sha256": pilot.sha256_file(folds),
        "protected_file_hashes_before": {str(protected): pilot.sha256_file(protected)},
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_path.with_suffix(".sha256").write_text(
        f"{pilot.sha256_file(plan_path)}  task_plan.json\n", encoding="utf-8"
    )
    pilot.verify_frozen_plan(plan_path, plan)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input hash changed"):
        pilot.verify_frozen_plan(plan_path, plan)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"preflight_blocked": True, "integrity_ok": True, "safety_ok": True}, "BLOCKED_PREFLIGHT"),
        ({"preflight_blocked": False, "integrity_ok": False, "safety_ok": True}, "PILOT_FAILED_SAFETY"),
        ({"preflight_blocked": False, "integrity_ok": True, "safety_ok": True,
          "improvement_ci_low": -0.01}, "PILOT_INCONCLUSIVE"),
        ({"preflight_blocked": False, "integrity_ok": True, "safety_ok": True}, "PILOT_SUCCESS"),
    ],
)
def test_all_four_conclusion_branches(kwargs, expected):
    defaults = {
        "preflight_blocked": False, "integrity_ok": True, "safety_ok": True,
        "improvement_ci_low": 0.01, "reliable_pair_estimate": 0.7,
        "reliable_pair_ci_low": 0.55, "top20_ci_high": 0.1,
        "reproducing_seed_count": 2, "positive_fold_count": 2,
        "reliable_pair_count": 10, "reliable_pair_architecture_count": 6,
        "minimum_reliable_pair_count": 10,
        "minimum_reliable_pair_architecture_count": 6,
    }
    defaults.update(kwargs)
    assert evaluation.classify_pilot(**defaults) == expected


def test_too_few_reliable_pairs_is_inconclusive():
    assert evaluation.classify_pilot(
        preflight_blocked=False, integrity_ok=True, safety_ok=True,
        improvement_ci_low=0.01, reliable_pair_estimate=0.8,
        reliable_pair_ci_low=0.6, top20_ci_high=0.1,
        reproducing_seed_count=3, positive_fold_count=4,
        reliable_pair_count=9, reliable_pair_architecture_count=6,
        minimum_reliable_pair_count=10,
        minimum_reliable_pair_architecture_count=6,
    ) == "PILOT_INCONCLUSIVE"


def test_fixed_seed_reproduces_folds_and_head_initialization(labels):
    assert pilot.build_nested_folds(labels) == pilot.build_nested_folds(labels)
    pilot.set_all_seeds(77)
    state_a = {key: value.clone() for key, value in pilot.PerformanceHead().state_dict().items()}
    pilot.set_all_seeds(77)
    state_b = pilot.PerformanceHead().state_dict()
    assert all(torch.equal(state_a[key], state_b[key]) for key in state_a)
