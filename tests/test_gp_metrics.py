from __future__ import annotations

import math
import random
import json

import pytest

from surrogate.metrics import prequential_ranking_metrics


def _records(actual, predicted, predicted_std=0.02):
    return [
        {
            "valid": True,
            "val_acc": actual_value,
            "gp_pred_mean": predicted_value,
            "gp_pred_std": predicted_std,
        }
        for actual_value, predicted_value in zip(actual, predicted)
    ]


def test_perfect_prequential_ranking_metrics():
    actual = [0.70 + 0.01 * index for index in range(12)]
    metrics = prequential_ranking_metrics(_records(actual, actual), 12)

    assert metrics["prequential_spearman"] == pytest.approx(1.0)
    assert metrics["prequential_pairwise_accuracy"] == pytest.approx(1.0)
    assert metrics["prequential_ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["prequential_top10_recall"] == pytest.approx(1.0)
    assert metrics["prequential_top10_regret"] == pytest.approx(0.0)
    assert metrics["prequential_prediction_bias"] == pytest.approx(0.0)
    assert metrics["prequential_top_region_mae"] == pytest.approx(0.0)


def test_reverse_prequential_ranking_metrics():
    actual = [0.70 + 0.01 * index for index in range(12)]
    reversed_prediction = list(reversed(actual))
    perfect = prequential_ranking_metrics(_records(actual, actual), 12)
    reverse = prequential_ranking_metrics(_records(actual, reversed_prediction), 12)

    assert reverse["prequential_spearman"] == pytest.approx(-1.0)
    assert reverse["prequential_pairwise_accuracy"] == pytest.approx(0.0)
    assert reverse["prequential_ndcg_at_10"] < perfect["prequential_ndcg_at_10"]
    assert reverse["prequential_top10_regret"] >= 0.0
    assert reverse["prequential_top10_regret"] > 0.0


def test_nonfinite_missing_constant_and_small_inputs_are_safe():
    noisy_records = [
        {"valid": True, "val_acc": 0.7, "gp_pred_mean": 0.71, "gp_pred_std": 0.02},
        {"valid": True, "val_acc": math.nan, "gp_pred_mean": 0.72, "gp_pred_std": 0.02},
        {"valid": True, "val_acc": 0.73, "gp_pred_mean": math.inf, "gp_pred_std": 0.02},
        {"valid": True, "val_acc": 0.74, "gp_pred_std": 0.02},
        {"valid": False, "val_acc": 0.99, "gp_pred_mean": 0.99, "gp_pred_std": 0.02},
    ]
    filtered = prequential_ranking_metrics(noisy_records, 10)
    assert filtered["prequential_window_size"] == 1
    assert filtered["prequential_spearman"] is None
    assert filtered["prequential_pairwise_accuracy"] is None

    constant_prediction = prequential_ranking_metrics(
        _records([0.70, 0.71, 0.72], [0.5, 0.5, 0.5]), 3
    )
    assert constant_prediction["prequential_spearman"] is None

    constant_actual = prequential_ranking_metrics(
        _records([0.7, 0.7, 0.7], [0.6, 0.7, 0.8]), 3
    )
    assert constant_actual["prequential_spearman"] is None
    assert constant_actual["prequential_ndcg_at_10"] is None
    assert constant_actual["prequential_pairwise_accuracy"] is None

    close_actual = prequential_ranking_metrics(
        _records([0.7000, 0.7005, 0.7010], [0.9, 0.8, 0.7]), 3
    )
    assert close_actual["prequential_pairwise_pairs"] == 0
    assert close_actual["prequential_pairwise_accuracy"] is None
    json.dumps(constant_actual, allow_nan=False)


def test_rolling_window_uses_latest_valid_pre_update_predictions_only():
    records = _records(
        [0.60, 0.61, 0.70, 0.71, 0.72],
        [0.10, 0.11, 0.70, 0.71, 0.72],
    )
    records.extend(
        [
            {"valid": True, "val_acc": 0.99, "gp_pred_mean": None, "gp_pred_std": None},
            {"valid": False, "val_acc": 1.0, "gp_pred_mean": 0.0, "gp_pred_std": 0.02},
        ]
    )

    metrics = prequential_ranking_metrics(records, 3)

    assert metrics["prequential_window_size"] == 3
    assert metrics["prequential_window_mae"] == pytest.approx(0.0)
    assert metrics["prequential_prediction_bias"] == pytest.approx(0.0)
    assert metrics["prequential_spearman"] == pytest.approx(1.0)


def test_uncertainty_metrics_only_use_finite_nonnegative_std():
    records = _records([0.70, 0.71, 0.72], [0.70, 0.71, 0.72])
    records[1]["gp_pred_std"] = math.inf
    records[2]["gp_pred_std"] = -1.0

    metrics = prequential_ranking_metrics(records, 3)

    assert metrics["prequential_window_size"] == 3
    assert metrics["prequential_window_mean_std"] == pytest.approx(0.02)
    assert metrics["prequential_window_coverage_95"] == pytest.approx(1.0)


def test_metrics_do_not_consume_rng_state():
    records = _records([0.70, 0.72, 0.71], [0.69, 0.73, 0.70])
    python_state = random.getstate()

    numpy_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except ImportError:
        np = None

    torch_state = None
    try:
        import torch

        torch_state = torch.random.get_rng_state().clone()
    except ImportError:
        torch = None

    prequential_ranking_metrics(records, 3)

    assert random.getstate() == python_state
    if numpy_state is not None:
        after = np.random.get_state()
        assert after[0] == numpy_state[0]
        assert (after[1] == numpy_state[1]).all()
        assert after[2:] == numpy_state[2:]
    if torch_state is not None:
        assert torch.equal(torch.random.get_rng_state(), torch_state)
