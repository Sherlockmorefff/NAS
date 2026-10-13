from __future__ import annotations

import copy

import pytest

from analyse import h_targeted_reliable_pair_validation as targeted
from analyse import evaluate_h_targeted_reliable_pair_validation as evaluator


def _row(config_id, fingerprint, mean, low, high, prediction=0.0):
    return {
        "config_id": config_id,
        "architecture_fingerprint": fingerprint,
        "validation_mean": mean,
        "validation_ci95_low": low,
        "validation_ci95_high": high,
        "time_index": 0,
        "prediction": prediction,
        "prediction_interval_low": prediction - 0.1,
        "prediction_interval_high": prediction + 0.1,
        "split_id": "s0",
        "training_seed": 0,
        "model": "frozen_head",
    }


def test_reliable_pair_direction_uses_higher_validation_mean():
    high = _row("high", "a", 0.82, 0.81, 0.83)
    low = _row("low", "b", 0.80, 0.79, 0.805)
    pairs, raw = targeted.reliable_config_pairs([low, high], {"a", "b"})
    assert raw == 1
    assert [(left["config_id"], right["config_id"]) for left, right in pairs] == [
        ("high", "low")
    ]


def test_symmetric_pair_is_emitted_once():
    left = _row("left", "a", 0.82, 0.81, 0.83)
    right = _row("right", "b", 0.80, 0.79, 0.805)
    pairs, raw = targeted.reliable_config_pairs([left, right], {"a", "b"})
    assert raw == 1
    assert len(pairs) == 1


def test_same_architecture_hp_pair_is_excluded():
    left = _row("left_hp", "same", 0.82, 0.81, 0.83)
    right = _row("right_hp", "same", 0.80, 0.79, 0.805)
    pairs, raw = targeted.reliable_config_pairs([left, right], {"same"})
    assert raw == 0
    assert pairs == []


@pytest.mark.parametrize(
    "gap,low_high,high_low,expected",
    [
        (0.006, 0.803, 0.804, True),
        (0.0059, 0.803, 0.804, False),
        (0.006, 0.805, 0.804, False),
    ],
)
def test_reliable_interval_rule(gap, low_high, high_low, expected):
    low_mean = 0.80
    high_mean = low_mean + gap
    high = _row("high", "a", high_mean, high_low, high_mean + 0.002)
    low = _row("low", "b", low_mean, low_mean - 0.002, low_high)
    pairs, _ = targeted.reliable_config_pairs([high, low], {"a", "b"})
    assert bool(pairs) is expected


def test_architecture_level_top_selection_groups_hp_rows(monkeypatch):
    monkeypatch.setattr(targeted, "TOP_FRACTION", 0.5)
    rows = [
        _row("a1", "a", 0.90, 0.89, 0.91),
        _row("a2", "a", 0.70, 0.69, 0.71),
        _row("b1", "b", 0.79, 0.78, 0.80),
        _row("c1", "c", 0.78, 0.77, 0.79),
        _row("d1", "d", 0.77, 0.76, 0.78),
    ]
    assert set(targeted.select_top_architectures(rows)) == {"a", "b"}


def test_aggregate_predictions_pools_seeds_and_splits():
    base = _row("a", "fp", 0.8, 0.79, 0.81, prediction=0.7)
    second = copy.deepcopy(base)
    second.update(prediction=0.9, split_id="s1", training_seed=1)
    pooled = targeted.aggregate_predictions([base, second])
    assert len(pooled) == 1
    assert pooled[0]["prediction"] == pytest.approx(0.8)
    assert pooled[0]["prediction_source_count"] == 2
    assert pooled[0]["split_id_count"] == 2
    assert pooled[0]["training_seed_count"] == 2


def test_history_projection_masks_label_fields(tmp_path):
    source = {
        "operations": ["GCNConv"],
        "edges": [[0, 1], [1, 2]],
        "z_arch": [0.0] * 12,
        "z_search": [0.0] * 12 + [0.1, 0.2, 0.3, 0.4],
        "search_seed": 3,
        "step": 7,
        "val_acc": 0.99,
        "test_acc": 1.0,
        "value": 0.99,
    }
    projected, masked = targeted.project_history_candidate(
        source, source_path=tmp_path / "history.json", source_index=0
    )
    assert set(masked) == {"test_acc", "val_acc", "value"}
    assert not targeted.contains_forbidden_candidate_field(projected)
    assert "val_acc" not in projected and "test_acc" not in projected


def test_one_hp_per_architecture_and_existing_overlap_removed():
    base = {
        "architecture_fingerprint": "a",
        "source_path": "/tmp/a",
        "source_index": 0,
        "z_arch": [0.0] * 12,
        "raw_hp_vector": [0.1] * 4,
    }
    duplicate = {**base, "source_index": 1, "raw_hp_vector": [0.9] * 4}
    excluded = {**base, "architecture_fingerprint": "existing", "source_index": 2}
    unique = {**base, "architecture_fingerprint": "b", "source_index": 3}
    pool, audit = targeted.deduplicate_candidates(
        [base, duplicate, excluded, unique], {"existing"}
    )
    assert {row["architecture_fingerprint"] for row in pool} == {"a", "b"}
    assert len(pool) == len({row["architecture_fingerprint"] for row in pool})
    assert audit["excluded_existing_unique_architecture_count"] == 1


def test_adaptive_seed_rule_is_only_ci_width_or_missing_valid_seeds():
    rows = [
        {"config_id": "stable", "seed_count": 15, "validation_ci95_half_width": 0.006},
        {"config_id": "wide", "seed_count": 15, "validation_ci95_half_width": 0.0061},
        {"config_id": "missing", "seed_count": 14, "validation_ci95_half_width": 0.001},
    ]
    assert evaluator.adaptive_config_ids(rows) == ["missing", "wide"]


def _classification_metrics(reliable_pairs=20):
    head = {
        "ndcg_at_10": 0.9,
        "top_10_regret": 0.02,
    }
    h = {
        "reliable_pair_count": reliable_pairs,
        "reliable_unique_architecture_count": 12,
        "reliable_pairwise_accuracy_ci95_low": 0.6,
        "kendall_ci95_low": 0.1,
        "pairwise_accuracy_ci95_low": 0.6,
        "ndcg_at_10": 0.91,
        "top_10_regret": 0.01,
        "prediction_interval_coverage": 0.9,
    }
    difference = {"estimate": 0.05, "ci95_low": 0.01}
    return head, h, difference


def test_all_four_final_classification_branches():
    head, h, difference = _classification_metrics()
    assert evaluator.classify_final(
        phase_a_passed=False, h=h, head=head, kendall_difference=difference,
        sensitivity_direction_consistent=True,
    )[0] == "PAIR_AUDIT_ERROR_FOUND"
    assert evaluator.classify_final(
        phase_a_passed=True, h=h, head=head, kendall_difference=difference,
        sensitivity_direction_consistent=True,
    )[0] == "H_VALIDATED_FOR_STRUCTURE_SURROGATE_PILOT"
    rejected = dict(h, kendall_ci95_low=-0.1)
    assert evaluator.classify_final(
        phase_a_passed=True, h=rejected, head=head, kendall_difference=difference,
        sensitivity_direction_consistent=True,
    )[0] == "H_REJECTED_WITH_SUFFICIENT_EVIDENCE"
    underpowered = dict(h, reliable_pair_count=19)
    assert evaluator.classify_final(
        phase_a_passed=True, h=underpowered, head=head, kendall_difference=difference,
        sensitivity_direction_consistent=True,
    )[0] == "H_STILL_UNDERPOWERED"


def test_shard_assignment_is_ten_ten_without_duplicates():
    selection = [
        {"config_id": f"c{i}", "architecture_fingerprint": f"fp{i}", "assigned_gpu": i % 2}
        for i in range(20)
    ]
    assert sum(row["assigned_gpu"] == 0 for row in selection) == 10
    assert sum(row["assigned_gpu"] == 1 for row in selection) == 10
    assert len({row["architecture_fingerprint"] for row in selection}) == 20
