from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analyse.original_head_deterministic_batch_pilot import (
    CLASSIFICATIONS,
    _config_fingerprint_pairs_from_text,
    _reference_config_ids_from_text,
    deterministic_kmedoids,
    explicit_feature_blocks,
    head_only_selection,
    head_partition_selection,
    make_union_tasks,
    mixed_distance,
    random_partition_selection,
    strategy_selections,
)
from analyse.analyse_original_head_batch_pilot import (
    aggregate_configurations,
    build_pool_metrics,
    classify,
    paired_differences,
    paired_pool_bootstrap,
    validate_complete_seed_results,
)
from analyse.run_original_head_batch_pilot_labels import load_jsonl


def candidate(config_id: str, pool: int = 0, fp: str | None = None, hp: str | None = None) -> dict:
    fp = fp or (f"{pool:02x}" + config_id.encode().hex()).ljust(64, "0")[:64]
    hp = hp or ("f" + config_id.encode().hex()).ljust(64, "1")[:64]
    return {
        "pool_id": pool,
        "config_id": config_id,
        "architecture_fingerprint": fp,
        "hp_fingerprint": hp,
        "operations": ["GCNConv", "GATConv"],
        "edges": [[0, 1], [1, 2], [2, 3]],
        "effective_layers": 2,
        "raw_hp_vector": [0.2, 0.3, 0.4, 0.5],
        "lr": 10 ** -3.5,
        "dropout": 0.25,
        "hidden_dim": 64,
        "l2": 1e-4,
        "gat_heads": 1,
        "sage_aggr": "mean",
        "gin_eps": 0.0,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
        "hp_regime": "hp_low",
    }


def test_distance_is_equal_weighted_and_uses_fixed_bounds() -> None:
    left = candidate("left")
    right = candidate("right")
    right["operations"] = ["GATConv", "GATConv"]
    right["edges"] = [[0, 1], [0, 2], [2, 3]]
    right.update({"lr": 10 ** -1.5, "dropout": 0.6, "hidden_dim": 512, "l2": 1e-3})
    total, blocks = mixed_distance(left, right)
    assert total == pytest.approx(sum(blocks.values()) / 3.0)
    assert blocks["operation_hamming"] == pytest.approx(1 / 5)
    assert blocks["edge_hamming"] == pytest.approx(2 / 21)
    assert 0 < blocks["hp_gower_l1"] <= 1
    _, _, hp = explicit_feature_blocks(right)
    assert hp.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_config_id_only_exposure_manifest_resolves_to_frozen_record(tmp_path: Path) -> None:
    fingerprint = "a" * 64
    targeted = tmp_path / "targeted.json"
    targeted.write_text(json.dumps([
        {"architecture_fingerprint": fingerprint, "config_id": "targeted_1", "h_prediction": 0.8}
    ], indent=2, sort_keys=True), encoding="utf-8")
    adaptive = tmp_path / "adaptive.json"
    adaptive.write_text(json.dumps({"config_ids": ["targeted_1"]}, indent=2, sort_keys=True), encoding="utf-8")
    assert _config_fingerprint_pairs_from_text(targeted) == [("targeted_1", fingerprint)]
    assert _reference_config_ids_from_text(adaptive) == ["targeted_1"]


@pytest.mark.parametrize("field", ["mu_arch", "latent_distance", "val_acc", "test_accuracy", "head_prediction"])
def test_distance_rejects_mu_prediction_and_label_fields(field: str) -> None:
    row = candidate("bad")
    row[field] = 0.0
    with pytest.raises(ValueError, match="forbidden distance field"):
        mixed_distance(row, candidate("good"))


def test_kmedoids_is_deterministic_nonempty_and_tie_breaks_by_id() -> None:
    points = np.asarray([0.0, 0.1, 4.9, 5.0, 9.9, 10.0, 14.9, 15.0])
    distances = np.abs(points[:, None] - points[None, :])
    ids = [f"c{index}" for index in range(len(points))]
    first = deterministic_kmedoids(distances, ids, k=4, seed=123)
    second = deterministic_kmedoids(distances, ids, k=4, seed=123)
    assert np.array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert set(first[0]) == {0, 1, 2, 3}
    assert all(np.sum(first[0] == cluster) > 0 for cluster in range(4))


def test_head_and_random_strategy_budgets_and_order() -> None:
    assignments = [
        {"pool_id": 0, "config_id": f"c{cluster}_{index}", "cluster_id": cluster}
        for cluster in range(4) for index in range(2)
    ]
    means = {row["config_id"]: float(index) for index, row in enumerate(assignments)}
    a = head_partition_selection(assignments, means)
    b = head_only_selection(list(means), means)
    assert len(a) == len(b) == 4
    assert [means[item] for item in a] == sorted([means[item] for item in a], reverse=True)
    assert [means[item] for item in b] == sorted([means[item] for item in b], reverse=True)
    blind = [
        {"pool_id": 0, "config_id": row["config_id"], "architecture_fingerprint": "a" * 64,
         "hp_fingerprint": "b" * 64, "cluster_id": row["cluster_id"]}
        for row in assignments
    ]
    assert random_partition_selection(blind, 99) == random_partition_selection(blind, 99)
    contaminated = [dict(row, head_prediction=0.5) for row in blind]
    with pytest.raises(ValueError, match="blind schema"):
        random_partition_selection(contaminated, 99)


def test_strategy_overlap_deduplicates_tasks_and_keeps_three_seeds_on_one_gpu() -> None:
    candidates = []
    assignments = []
    means = {}
    for pool in range(8):
        for index in range(8):
            row = candidate(f"p{pool}c{index}", pool=pool)
            candidates.append(row)
            assignments.append({"pool_id": pool, "config_id": row["config_id"], "cluster_id": index % 4})
            means[row["config_id"]] = float(index)
    selections, audit = strategy_selections(candidates, assignments, means)
    assert audit["passed"]
    union, tasks = make_union_tasks(candidates, selections)
    assert len(tasks) == len(union) * 3
    assert len({task["task_key"] for task in tasks}) == len(tasks)
    for config in union:
        matching = [task for task in tasks if task["config_id"] == config["config_id"]]
        assert {task["seed"] for task in matching} == {0, 1, 2}
        assert len({task["assigned_gpu"] for task in matching}) == 1


def synthetic_tasks_and_results() -> tuple[list[dict], list[dict]]:
    tasks = []
    results = []
    for seed in (0, 1, 2):
        base = {
            "pool_id": 0, "config_id": "c", "architecture_fingerprint": "a" * 64,
            "hp_fingerprint": "b" * 64, "seed": seed, "assigned_gpu": 0,
        }
        tasks.append(dict(base))
        results.append({
            **base, "dataset_content_fingerprint": "data", "split_fingerprint": "split",
            "hp_mode": "global4", "eval_epochs": 300, "patience": 80,
            "terminal_status": "success", "valid": True, "val_acc": 0.7 + seed * 0.01,
        })
    return tasks, results


def test_seed_completeness_has_no_pseudoreplication_and_rejects_test() -> None:
    tasks, results = synthetic_tasks_and_results()
    assert validate_complete_seed_results(tasks, results)["passed"]
    aggregate = aggregate_configurations(results)
    assert len(aggregate) == 1
    assert aggregate[0]["mean_validation"] == pytest.approx(0.71)
    assert aggregate[0]["seed_count"] == 3
    assert not validate_complete_seed_results(tasks, results[:-1])["passed"]
    contaminated = [dict(results[0], test_accuracy=0.9), *results[1:]]
    with pytest.raises(ValueError, match="test-named"):
        validate_complete_seed_results(tasks, contaminated)


def test_resume_loader_rejects_duplicates_and_test_fields(tmp_path: Path) -> None:
    _, results = synthetic_tasks_and_results()
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in results) + "\n", encoding="utf-8")
    assert len(load_jsonl(path)) == 3
    path.write_text(json.dumps(results[0]) + "\n" + json.dumps(results[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(path)
    bad = dict(results[0], test_acc=0.8)
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test-named"):
        load_jsonl(path)


def _difference_rows(ac: float, ab: float, bc: float) -> list[dict]:
    return [{"pool_id": pool, "A_minus_C": ac, "A_minus_B": ab, "B_minus_C": bc} for pool in range(8)]


def test_paired_bootstrap_resamples_pools_and_classification_branches() -> None:
    rows = _difference_rows(0.01, 0.0, 0.01)
    samples, summary = paired_pool_bootstrap(rows, 20000, 42)
    assert len(samples) == 20000
    assert summary["outer_unit"] == "pool"
    assert summary["comparisons"]["A_minus_C"]["ci95_lower"] == pytest.approx(0.01)
    assert classify(rows, summary)[0] == "HEAD_DIVERSITY_PROMOTED"

    rows = _difference_rows(0.0, -0.01, 0.01)
    _, summary = paired_pool_bootstrap(rows, 20000, 42)
    assert classify(rows, summary)[0] == "HEAD_ONLY_PROMOTED"

    rows = _difference_rows(-0.01, 0.0, -0.01)
    _, summary = paired_pool_bootstrap(rows, 20000, 42)
    assert classify(rows, summary)[0] == "RANDOM_DIVERSITY_RETAINED"

    rows = _difference_rows(0.003, 0.0, 0.003)
    _, summary = paired_pool_bootstrap(rows, 20000, 42)
    assert classify(rows, summary)[0] == "PILOT_INCONCLUSIVE"
    assert set(CLASSIFICATIONS) == {
        "HEAD_DIVERSITY_PROMOTED", "HEAD_ONLY_PROMOTED", "RANDOM_DIVERSITY_RETAINED",
        "PILOT_INCONCLUSIVE", "INSUFFICIENT_FRESH_CANDIDATES", "EXECUTION_OR_PROTOCOL_FAILURE",
    }


def test_observed_union_regret_uses_only_evaluated_union() -> None:
    candidates = [candidate(f"c{index}") for index in range(6)]
    selections = []
    for strategy, ids in {
        "HEAD_PARTITION_DIVERSITY": ["c0", "c1", "c2", "c3"],
        "HEAD_ONLY": ["c0", "c2", "c4", "c5"],
        "RANDOM_PARTITION_DIVERSITY": ["c1", "c2", "c4", "c5"],
    }.items():
        selections.extend({"pool_id": 0, "strategy": strategy, "budget_order": order, "config_id": config_id}
                          for order, config_id in enumerate(ids, 1))
    # Repeat the same construction for all pools to satisfy the fixed 8-pool API.
    all_candidates = []
    all_selections = []
    values = {}
    for pool in range(8):
        for row in candidates:
            copied = dict(row, pool_id=pool, config_id=f"p{pool}_{row['config_id']}")
            all_candidates.append(copied)
            values[copied["config_id"]] = int(row["config_id"][1:]) / 10
        for row in selections:
            all_selections.append(dict(row, pool_id=pool, config_id=f"p{pool}_{row['config_id']}"))
    _, _, regrets = build_pool_metrics(all_selections, all_candidates, values)
    assert all(row["not_full_candidate_pool_regret"] for row in regrets)
    assert all(row["observed_union_regret"] >= 0 for row in regrets)
