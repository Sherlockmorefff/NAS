from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from analyse import run_unified_label_expansion as runner
from analyse import vae_label_expansion as expansion
from dataset_utils import DATASET_CONTEXT_FIELDS


def _history_row(index: int):
    return {
        "architecture_fingerprint": f"arch-{index:03d}",
        "candidate_fingerprint": f"candidate-{index:03d}",
        "val_acc": index / 100.0,
        "_search_seed": index % 5,
        "_step": index,
    }


def _selected_config(index: int, gpu: int = 0):
    return {
        "config_id": f"config-{index:02d}",
        "source": "search_history_expansion",
        "architecture_fingerprint": f"arch-{index:02d}",
        "hp_fingerprint": f"hp-{index:02d}",
        "operations": ["GCNConv"],
        "edges": [[0, 1], [1, 2]],
        "z_search": [0.0] * 16,
        "z_arch": [0.0] * 12,
        "raw_hp_vector": [0.0] * 4,
        "lr": 0.01,
        "dropout": 0.5,
        "hidden_dim": 16,
        "l2": 0.0005,
        "gat_heads": 1,
        "sage_aggr": "mean",
        "gin_eps": 0.0,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
        "normalization": "none",
        "assigned_gpu": gpu,
    }


def test_history_regions_are_mutually_exclusive_and_complete():
    rows = expansion._assign_complete_regions([_history_row(index) for index in range(100)])
    counts = {
        region: sum(row["exclusive_region"] == region for row in rows)
        for region in ("lower", "middle", "upper")
    }
    assert counts == {"lower": 35, "middle": 30, "upper": 35}
    assert sum(counts.values()) == len(rows)


def test_rank_segments_select_ten_and_balance_search_seeds():
    selected = expansion._segment_select(
        [_history_row(index) for index in range(100)], "middle"
    )
    assert len(selected) == 10
    assert {row["selection_segment"] for row in selected} == set(range(10))
    assert {
        seed: sum(row["_search_seed"] == seed for row in selected)
        for seed in range(5)
    } == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}


def test_hp_fingerprint_is_stable_and_sensitive_to_realized_hp():
    config = _selected_config(0)
    reordered = dict(reversed(list(config.items())))
    changed = {**config, "dropout": 0.4}
    assert expansion.hp_fingerprint(config) == expansion.hp_fingerprint(reordered)
    assert expansion.hp_fingerprint(config) != expansion.hp_fingerprint(changed)


def test_adaptive_plan_only_adds_configs_above_preregistered_half_width(tmp_path):
    selected = [_selected_config(index, index % 2) for index in range(26)]
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected), encoding="utf-8")
    rows_by_gpu = {0: [], 1: []}
    for config in selected:
        for seed in expansion.INITIAL_SEEDS:
            value = (
                (0.6 if seed % 2 == 0 else 0.9)
                if config["config_id"] == "config-00"
                else 0.8 + seed * 0.00001
            )
            rows_by_gpu[config["assigned_gpu"]].append(
                {
                    "source": config["source"],
                    "config_id": config["config_id"],
                    "architecture_fingerprint": config["architecture_fingerprint"],
                    "hp_fingerprint": config["hp_fingerprint"],
                    "seed": seed,
                    "val_acc": value,
                    "terminal_status": (
                        "failed"
                        if config["config_id"] == "config-01" and seed == 14
                        else "success"
                    ),
                }
            )
    paths = []
    for gpu in (0, 1):
        path = tmp_path / f"gpu{gpu}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows_by_gpu[gpu]),
            encoding="utf-8",
        )
        paths.append(path)
    plan = expansion.adaptive_plan(selected_path, paths, tmp_path / "out")
    assert plan["adaptive_config_ids"] == ["config-00"]
    assert plan["additional_training_count"] == 15
    assert plan["by_gpu"] == {"0": ["config-00"], "1": []}
    incomplete = next(
        row for row in plan["initial_statistics"]
        if row["config_id"] == "config-01"
    )
    assert incomplete["successful_seed_count"] == 14
    assert incomplete["adaptive_seeds_required"] is False
    assert incomplete["adaptive_decision"] == "NOT_TRIGGERED_INCOMPLETE_INITIAL_LABELS"


def test_runner_dry_run_does_not_load_dataset_or_train(tmp_path, monkeypatch):
    selection = [_selected_config(0, 0)]
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "load_dataset_from_request",
        lambda *_args, **_kwargs: pytest.fail("dry-run loaded Cora"),
    )
    monkeypatch.setattr(
        runner,
        "train_and_eval_arch",
        lambda *_args, **_kwargs: pytest.fail("dry-run trained a model"),
    )
    output = tmp_path / "out"
    assert runner.main(
        [
            "--selection", str(path), "--shard", "0", "--seeds", "0", "1",
            "--output", str(output), "--dry_run",
        ]
    ) == 0
    plan = json.loads((output / runner.RUN_PLAN_JSON).read_text())
    assert plan["expected_training_count"] == 2
    assert not (output / runner.RESULTS_JSONL).exists()


def test_runner_records_three_failed_attempts_at_two_retry_cap(
    tmp_path, monkeypatch
):
    selection = [_selected_config(0, 0)]
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    manifest = {
        "canonical_name": "cora",
        "source_class": "torch_geometric.datasets.Planetoid",
        "source_version": "2.7.0",
        "torch_version": "2.5.1+cu121",
        "pyg_version": "2.7.0",
        "ogb_version": "1.3.6",
        "dataset_content_fingerprint": expansion.EXPECTED_CONTEXT["dataset_content_fingerprint"],
        "split_protocol": "planetoid_public",
        "split_seed": None,
        "split_fingerprint": expansion.EXPECTED_CONTEXT["split_fingerprint"],
        "graph_transform": "NormalizeFeatures",
        "training_mode": "full_batch",
        "metric": "accuracy",
    }
    assert set(DATASET_CONTEXT_FIELDS) <= set(manifest)
    data = SimpleNamespace(to=lambda _device: object())
    bundle = SimpleNamespace(
        manifest=manifest, data=data, num_features=8, num_classes=3
    )
    monkeypatch.setattr(runner, "resolve_dataset_request", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "load_dataset_from_request", lambda *_args: bundle)
    monkeypatch.setattr(
        runner,
        "train_and_eval_arch",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    output = tmp_path / "out"
    assert runner.main(
        [
            "--selection", str(path), "--shard", "0", "--seeds", "0",
            "--output", str(output), "--device", "cpu", "--max_retries", "2",
        ]
    ) == 0
    row = json.loads((output / runner.RESULTS_JSONL).read_text().strip())
    assert row["terminal_status"] == "failed"
    assert row["attempt_count"] == 3
    assert row["retry_count"] == 2
    assert len(row["attempt_failures"]) == 3


def test_unified_training_payload_rejects_any_test_named_key():
    assert expansion._contains_test_key([{"validation_mean": 0.8}]) is False
    assert expansion._contains_test_key([{"test_acc": 0.8}]) is True
    assert expansion._contains_test_key([{"nested": {"sealed_test": 0.8}}]) is True


def test_representation_increments_are_paired_by_split():
    rows = []
    for split, baseline, current in (("s0", 0.2, 0.4), ("s1", 0.3, 0.5)):
        rows.extend(
            [
                {
                    "protocol": "group_repeated", "split_id": split,
                    "representation": "Zs", "metric": "late_50_kendall",
                    "value": baseline,
                },
                {
                    "protocol": "group_repeated", "split_id": split,
                    "representation": "Ze", "metric": "late_50_kendall",
                    "value": current,
                },
            ]
        )
    increments = expansion._paired_representation_increments(rows)
    result = next(
        row for row in increments
        if row["representation"] == "Ze" and row["metric"] == "late_50_kendall"
    )
    assert result["paired_split_count"] == 2
    assert result["estimate"] == pytest.approx(0.2)
    assert result["ci95_low"] == pytest.approx(0.2)
