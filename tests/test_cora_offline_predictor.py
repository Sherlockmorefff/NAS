from __future__ import annotations

import copy

import numpy as np
import pytest

from analyse import cora_offline_predictor as predictor


def _record(index: int = 50) -> dict:
    return {
        "architecture_fingerprint": "arch-a",
        "hp_fingerprint": "hp-a",
        "candidate_fingerprint": "candidate-a",
        "operations": ["GCNConv", "Identity", "GCNII", "SAGEConv", "GINConv"],
        "edges": [[0, 1], [1, 2], [0, 2], [2, 3], [3, 4], [2, 4], [4, 5], [5, 6]],
        "lr": 1e-3,
        "dropout": 0.4,
        "hidden_dim": 256,
        "l2": 1e-4,
        "val_acc": 0.8,
        "full_evaluation_index": index,
        "evaluation_stage": "online_bo",
    }


def test_features_are_independent_of_test_accuracy():
    left = _record()
    right = copy.deepcopy(left)
    left["test_acc"] = 0.1
    right["test_acc"] = 0.99
    left_arch, left_names = predictor.architecture_features(left)
    right_arch, right_names = predictor.architecture_features(right)
    left_hp, left_hp_names = predictor.hp_features(left)
    right_hp, right_hp_names = predictor.hp_features(right)
    np.testing.assert_array_equal(left_arch, right_arch)
    np.testing.assert_array_equal(left_hp, right_hp)
    assert left_names == right_names
    assert left_hp_names == right_hp_names
    assert "has_identity" in left_names
    assert "has_gcnii" in left_names
    assert "mixed_width_predecessor_events" in left_names
    assert any(name.startswith("path_transition_") for name in left_names)


def test_replay_cutoff_never_uses_current_or_future_label():
    frozen = _record(149)
    frozen["evaluation_stage"] = "initial_expand"
    assert predictor.replay_train_cutoff(frozen, 10) == 50
    online = _record(157)
    assert predictor.replay_train_cutoff(online, 10) == 150
    assert predictor.replay_train_cutoff(online, 10) < online["full_evaluation_index"]


def test_configuration_deduplication_uses_architecture_and_hp():
    first = _record(0)
    same_configuration = copy.deepcopy(first)
    same_configuration["candidate_fingerprint"] = "different-latent-candidate"
    same_configuration["full_evaluation_index"] = 1
    new_hp = copy.deepcopy(first)
    new_hp["hp_fingerprint"] = "hp-b"
    new_hp["full_evaluation_index"] = 2
    assert predictor.first_configuration_indices([first, same_configuration, new_hp]) == [0, 2]


def test_ranking_metrics_reward_correct_top_order():
    actual = np.linspace(0.6, 0.9, 60)
    correct = predictor.ranking_metrics(actual, actual)
    reversed_metrics = predictor.ranking_metrics(actual, actual[::-1])
    assert correct["top20_kendall"] == pytest.approx(1.0)
    assert correct["top_region_pairwise_accuracy"] == pytest.approx(1.0)
    assert correct["top10_recall"] == pytest.approx(1.0)
    assert correct["best_of_10_regret"] == pytest.approx(0.0)
    assert reversed_metrics["top20_kendall"] < 0
    assert reversed_metrics["top10_recall"] == pytest.approx(0.0)
