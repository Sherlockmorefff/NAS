from __future__ import annotations

import hashlib
import math

import numpy as np

from first_formal_top10_test30 import (
    DETAILED_WINNER_FIELDS,
    STUDENT_T_975_DF29,
    _build_detailed_winner_rows,
    _candidate_stats,
    _dataset_split_seed_from_context,
    _initialize_database,
    candidate_fingerprint,
    claim_task,
    evaluation_cache_key,
    final_evaluation_seed,
    queue_counts,
    representative_method_origin,
    representative_origin,
    select_method_winner,
    select_top10_records,
)


def _fingerprint(index: int) -> str:
    return hashlib.sha256(f"candidate-{index}".encode()).hexdigest()


def _record(index: int, validation: float, **updates):
    record = {
        "evaluation_fidelity": "full",
        "valid": True,
        "val_acc": validation,
        "candidate_fingerprint": _fingerprint(index),
        "full_evaluation_index": index,
    }
    record.update(updates)
    return record


def test_candidate_fingerprint_matches_frozen_float32_layout():
    z = np.asarray([1.25, -2.5, 0.0], dtype=np.float64)
    array = np.ascontiguousarray(z.astype(np.float32).reshape(1, -1))
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    assert candidate_fingerprint(z) == digest.hexdigest()


def test_final_seed_matches_frozen_seed_fair_protocol():
    assert final_evaluation_seed(0, "0" * 64, 0) == 430542660
    assert final_evaluation_seed(0, "a" * 64, 29) == 97268938
    assert final_evaluation_seed(0, "a" * 64, 29) != final_evaluation_seed(
        0, "a" * 64, 28
    )


def test_top10_excludes_ineligible_deduplicates_and_uses_tie_order():
    records = [_record(index, 0.9 - index / 100) for index in range(12)]
    records.extend(
        [
            _record(50, 1.0, valid=False),
            _record(51, 1.0, evaluation_fidelity="low"),
            _record(52, float("nan")),
            _record(53, 0.9, candidate_fingerprint=_fingerprint(0)),
            _record(54, 0.9),
        ]
    )
    selected = select_top10_records(records)
    assert len(selected) == 10
    # Index 0 wins the 0.9 tie, and the duplicate fingerprint at index 53 is removed.
    assert [int(row["full_evaluation_index"]) for row in selected[:2]] == [0, 54]
    assert len({row["candidate_fingerprint"] for row in selected}) == 10
    assert all(row["valid"] is True for row in selected)


def test_representative_origin_and_winner_tie_breaks():
    base = {
        "candidate_fingerprint": _fingerprint(1),
        "search_validation_accuracy": 0.8,
        "global_full_evaluation_index": 100,
    }
    origins = [
        {**base, "search_seed": 7},
        {**base, "search_seed": 5, "global_full_evaluation_index": 110},
        {**base, "search_seed": 5, "global_full_evaluation_index": 90},
        {**base, "search_seed": 9, "search_validation_accuracy": 0.81},
    ]
    assert representative_origin(origins) is origins[3]

    candidates = [
        {
            "candidate_fingerprint": _fingerprint(2),
            "final_validation_mean": 0.7,
            "final_validation_sample_std": 0.02,
            "test_mean": 0.7,
            "test_sample_std": 0.02,
            "search_validation_accuracy": 0.9,
        },
        {
            "candidate_fingerprint": _fingerprint(3),
            "final_validation_mean": 0.7,
            "final_validation_sample_std": 0.01,
            "test_mean": 0.7,
            "test_sample_std": 0.01,
            "search_validation_accuracy": 0.8,
        },
    ]
    assert select_method_winner(candidates) is candidates[1]


def test_representative_method_origin_never_uses_another_method():
    origins = [
        {
            "method": "S0",
            "candidate_fingerprint": _fingerprint(40),
            "search_validation_accuracy": 0.99,
            "search_seed": 5,
            "global_full_evaluation_index": 10,
        },
        {
            "method": "G100",
            "candidate_fingerprint": _fingerprint(40),
            "search_validation_accuracy": 0.80,
            "search_seed": 7,
            "global_full_evaluation_index": 20,
        },
    ]
    assert representative_origin(origins) is origins[0]
    assert representative_method_origin(origins, "G100") is origins[1]


def test_method_winner_uses_final_validation_without_test_leakage():
    higher_final_validation = {
        "candidate_fingerprint": _fingerprint(4),
        "final_validation_mean": 0.81,
        "final_validation_sample_std": 0.03,
        "test_mean": 0.4,
        "test_sample_std": 0.2,
        "search_validation_accuracy": 0.7,
    }
    higher_test = {
        "candidate_fingerprint": _fingerprint(5),
        "final_validation_mean": 0.80,
        "final_validation_sample_std": 0.01,
        "test_mean": 0.99,
        "test_sample_std": 0.001,
        "search_validation_accuracy": 0.9,
    }
    assert select_method_winner([higher_test, higher_final_validation]) is higher_final_validation

    tied_validation_a = {
        **higher_final_validation,
        "final_validation_mean": 0.8,
        "final_validation_sample_std": 0.01,
        "search_validation_accuracy": 0.7,
        "test_mean": 0.0,
    }
    tied_validation_b = {
        **higher_test,
        "final_validation_mean": 0.8,
        "final_validation_sample_std": 0.01,
        "search_validation_accuracy": 0.7,
        "test_mean": 1.0,
    }
    expected = min(
        (tied_validation_a, tied_validation_b),
        key=lambda row: row["candidate_fingerprint"],
    )
    assert select_method_winner([tied_validation_b, tied_validation_a]) is expected


def test_sample_std_uses_ddof_one():
    test_values = [float(index) / 100 for index in range(30)]
    validation_values = [float(index) / 200 for index in range(30)]
    stats = _candidate_stats(test_values, validation_values)
    assert stats["test_mean"] == float(
        np.mean(np.asarray(test_values, dtype=np.float64))
    )
    assert stats["test_sample_std"] == float(
        np.std(np.asarray(test_values, dtype=np.float64), ddof=1)
    )
    expected_margin = (
        STUDENT_T_975_DF29 * stats["test_sample_std"] / math.sqrt(len(test_values))
    )
    assert math.isclose(
        stats["test_mean_ci95_low"], stats["test_mean"] - expected_margin
    )
    assert math.isclose(
        stats["test_mean_ci95_high"], stats["test_mean"] + expected_margin
    )
    assert stats["final_validation_mean"] == float(
        np.mean(np.asarray(validation_values, dtype=np.float64))
    )
    assert stats["final_validation_sample_std"] == float(
        np.std(np.asarray(validation_values, dtype=np.float64), ddof=1)
    )


def test_detailed_winner_row_contains_provenance_and_all_test_replicates():
    fingerprint = _fingerprint(41)
    winner = {
        "dataset": "pubmed",
        "method": "G150",
        "candidate_fingerprint": fingerprint,
        "architecture": {"operations": ["GCNConv"], "edges": [[0, 1]]},
        "hyperparameters": {
            "lr": 0.01,
            "dropout": 0.2,
            "hidden_dim": 128,
            "l2": 0.0001,
        },
        "lr": 0.01,
        "dropout": 0.2,
        "hidden_dim": 128,
        "l2": 0.0001,
        "final_validation_mean": 0.82,
        "final_validation_sample_std": 0.01,
        "test_mean": 0.81,
        "test_sample_std": 0.02,
        "test_mean_ci95_low": 0.8025,
        "test_mean_ci95_high": 0.8175,
        "search_seed": 9,
        "search_stage": "online_bo",
        "global_full_evaluation_position": 287,
        "within_stage_full_position": 87,
        "validation_top10_rank": 2,
    }
    results = [
        {
            "replicate_id": replicate_id,
            "test_accuracy": 0.8 + replicate_id / 1000,
        }
        for replicate_id in range(30)
    ]
    rows = _build_detailed_winner_rows(
        [winner], {("pubmed", fingerprint): results}
    )
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(DETAILED_WINNER_FIELDS)
    assert row["search_seed"] == 9
    assert row["global_evaluation_order"] == 287
    assert row["stage_local_order"] == 87
    assert row["test_accuracy_0"] == 0.8
    assert math.isclose(row["test_accuracy_29"], 0.829)


def test_worker_preserves_absent_split_seed_for_non_dblp():
    assert _dataset_split_seed_from_context("citeseer", {"split_seed": None}) is None
    assert _dataset_split_seed_from_context("pubmed", {"split_seed": None}) is None
    assert _dataset_split_seed_from_context("flickr", {"split_seed": None}) is None
    assert _dataset_split_seed_from_context("dblp", {"split_seed": 0}) == 0


def test_cache_key_ignores_candidate_origin_metadata():
    context = {
        "dataset_content_fingerprint": _fingerprint(20),
        "split_protocol": "official",
        "split_seed": None,
        "split_fingerprint": _fingerprint(21),
        "graph_transform": "none",
        "training_mode": "full_batch",
        "metric": "accuracy",
    }
    common = dict(
        dataset="flickr",
        fingerprint=_fingerprint(22),
        replicate_id=3,
        protocol_fingerprint=_fingerprint(23),
        dataset_context=context,
    )
    key_from_s0 = evaluation_cache_key(**common)
    # There is intentionally nowhere to pass method, search seed, stage, or rank.
    key_from_g150 = evaluation_cache_key(**common)
    assert key_from_s0 == key_from_g150


def test_sqlite_claims_are_unique_and_resumable(tmp_path):
    pipeline_root = tmp_path
    context = {
        "dataset_content_fingerprint": _fingerprint(30),
        "split_protocol": "public",
        "split_seed": None,
        "split_fingerprint": _fingerprint(31),
        "graph_transform": "normalize",
        "training_mode": "full_batch",
        "metric": "accuracy",
    }
    unique_rows = [
        {
            "dataset": "citeseer",
            "candidate_fingerprint": _fingerprint(32),
            "dataset_context": context,
        }
    ]
    assert _initialize_database(
        pipeline_root / "queue.sqlite3", unique_rows, _fingerprint(33)
    ) == 30
    first = claim_task(pipeline_root, "worker-a", 0)
    second = claim_task(pipeline_root, "worker-b", 1)
    assert first is not None and second is not None
    assert first["task_id"] != second["task_id"]
    assert first["replicate_id"] == 0
    assert second["replicate_id"] == 1
    assert queue_counts(pipeline_root) == {
        "pending": 28,
        "running": 2,
        "completed": 0,
        "failed": 0,
        "total": 30,
    }
