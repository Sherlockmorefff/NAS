from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

import determinism_smoke
import deterministic_three_strategy_pipeline as pipeline


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_runtime_paths_are_portable_and_repository_relative() -> None:
    assert pipeline.PYTHON_BIN == Path(sys.executable).resolve()
    assert pipeline.resolve_repository_path("server_validation/gate.json") == (
        pipeline.REPO_ROOT / "server_validation" / "gate.json"
    ).resolve()
    args = pipeline.parse_args(
        [
            "--protocol-id",
            "deterministic-three-strategy_v1_20260813_64392499",
            "--python-executable",
            sys.executable,
            "--training-mode-decisions",
            "fixtures/formal_decisions.json",
            "monitor",
        ]
    )
    assert args.python_executable == sys.executable
    assert args.training_mode_decisions == "fixtures/formal_decisions.json"


def test_determinism_smoke_requires_explicit_legacy_inputs() -> None:
    with pytest.raises(SystemExit):
        determinism_smoke.parse_args(["strict", "--gpu-id", "0"])
    args = determinism_smoke.parse_args(
        [
            "strict",
            "--gpu-id",
            "0",
            "--legacy-matrix",
            "fixtures/formal_matrix_60.json",
            "--legacy-repro-root",
            "fixtures/reproducibility",
        ]
    )
    assert args.legacy_matrix == "fixtures/formal_matrix_60.json"
    assert args.legacy_repro_root == "fixtures/reproducibility"


def _origin(
    dataset: str,
    method: str,
    search_seed: int,
    rank: int,
    fingerprint: str,
    source_id: str,
) -> dict:
    index = rank - 1
    raw = "initial_seed" if index < 5 else "online_bo"
    normalized, stage_index = pipeline.normalized_stage(raw, method, index if raw == "initial_seed" else (50 if method == "S0" else 150 if method == "G100" else 200) + index)
    full_index = index if raw == "initial_seed" else (50 if method == "S0" else 150 if method == "G100" else 200) + index
    return {
        "dataset": dataset,
        "method": method,
        "search_seed": search_seed,
        "validation_rank": rank,
        "candidate_fingerprint": fingerprint,
        "architecture": {"operations": ["GCNConv"], "edges": [[0, 1], [1, 2]]},
        "hyperparameters": {
            "lr": 0.01,
            "dropout": 0.3,
            "hidden_dim": 64,
            "l2": 0.0005,
            "gat_heads": 1,
            "sage_aggr": "mean",
            "gin_eps": 0.0,
            "gat_heads_by_layer": None,
            "sage_aggr_by_layer": None,
            "gin_eps_by_layer": None,
        },
        "validation_accuracy": 0.9 - rank / 1000,
        "raw_stage": raw,
        "normalized_stage": normalized,
        "full_evaluation_index_0based": full_index,
        "budget_1based": full_index + 1,
        "stage_index_1based": stage_index,
        "history_path": "/mock/history.json",
        "decoder_seed": 123,
        "source_id": source_id,
    }


def _prepare_mock_top10(root: Path) -> tuple[list[dict], list[dict]]:
    identity = pipeline.current_source_identity()
    source_rows = []
    origins_by_candidate = {}
    for dataset in pipeline.DATASETS:
        for method in pipeline.METHODS:
            for search_seed in pipeline.SEARCH_SEEDS:
                for rank in range(1, 11):
                    fingerprint = _fingerprint(f"{dataset}/candidate/{rank}")
                    row = _origin(
                        dataset, method, search_seed, rank, fingerprint, identity["source_id"]
                    )
                    source_rows.append(row)
                    origins_by_candidate.setdefault((dataset, fingerprint), []).append(row)
    unique_rows = []
    for (dataset, fingerprint), origins in sorted(origins_by_candidate.items()):
        canonical = min(
            origins,
            key=lambda row: (
                -row["validation_accuracy"], row["search_seed"], row["budget_1based"], row["method"]
            ),
        )
        unique_rows.append(
            {
                "dataset": dataset,
                "candidate_fingerprint": fingerprint,
                "architecture": canonical["architecture"],
                "hyperparameters": canonical["hyperparameters"],
                "decoder_seed": canonical["decoder_seed"],
                "canonical_method": canonical["method"],
                "canonical_search_seed": canonical["search_seed"],
                "canonical_validation_rank": canonical["validation_rank"],
                "canonical_validation_accuracy": canonical["validation_accuracy"],
                "canonical_raw_stage": canonical["raw_stage"],
                "canonical_normalized_stage": canonical["normalized_stage"],
                "canonical_full_evaluation_index_0based": canonical["full_evaluation_index_0based"],
                "canonical_budget_1based": canonical["budget_1based"],
                "canonical_stage_index_1based": canonical["stage_index_1based"],
                "all_origins": origins,
                "source_id": identity["source_id"],
            }
        )
    pipeline.write_csv(root / "top10" / "top10_source_candidates.csv", source_rows, pipeline.SOURCE_FIELDS)
    pipeline.write_csv(root / "top10" / "top10_unique_candidates.csv", unique_rows, pipeline.UNIQUE_FIELDS)
    pipeline.write_gate(
        root / "TOP10_EXTRACTION_AUDIT_PASSED",
        source_id=identity["source_id"],
        input_sha256="1" * 64,
        output_sha256="2" * 64,
        details={"source_rows": 600},
    )
    return source_rows, unique_rows


def test_top10_reads_only_full_validation_and_has_deterministic_ties() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "evaluation_fidelity": "full",
                "full_evaluation_index": index,
                "valid": True,
                "val_acc": 0.8 if index in (0, 1) else 0.79 - index / 100,
                "test_acc": 1.0 if index == 11 else 0.0,
                "candidate_fingerprint": f"{11-index:064x}",
            }
        )
    rows.append(
        {
            "evaluation_fidelity": "low",
            "full_evaluation_index": None,
            "valid": True,
            "val_acc": 1.0,
            "test_acc": 1.0,
            "candidate_fingerprint": "f" * 64,
        }
    )
    selected = pipeline.select_top10_full_records(rows)
    assert len(selected) == 10
    assert all(row["evaluation_fidelity"] == "full" for row in selected)
    assert [row["full_evaluation_index"] for row in selected[:2]] == [0, 1]
    changed_test = [{**row, "test_acc": 1.0 - float(row.get("test_acc", 0.0))} for row in rows]
    assert [row["candidate_fingerprint"] for row in pipeline.select_top10_full_records(changed_test)] == [row["candidate_fingerprint"] for row in selected]


def test_stage_mapping_uses_real_history_boundaries() -> None:
    assert pipeline.normalized_stage("initial_seed", "S0", 49) == ("seed_full", 50)
    assert pipeline.normalized_stage("initial_expand", "G100", 50) == ("promoted_full", 1)
    assert pipeline.normalized_stage("initial_expand", "G150", 199) == ("promoted_full", 150)
    assert pipeline.normalized_stage("online_bo", "S0", 50) == ("online_bo", 1)
    assert pipeline.normalized_stage("online_bo", "G100", 150) == ("online_bo", 1)
    assert pipeline.normalized_stage("online_bo", "G150", 200) == ("online_bo", 1)


def test_two_worker_partition_is_disjoint_complete_and_modulo_based() -> None:
    manifest = {
        "tasks": [
            {"dataset": "citeseer", "candidate_fingerprint": "a" * 64, "test_seed": seed}
            for seed in range(30)
        ]
    }
    worker0 = pipeline.assigned_tasks(manifest, 0, 2)
    worker1 = pipeline.assigned_tasks(manifest, 1, 2)
    keys0 = {(row["candidate_fingerprint"], row["test_seed"]) for row in worker0}
    keys1 = {(row["candidate_fingerprint"], row["test_seed"]) for row in worker1}
    assert not keys0 & keys1
    assert keys0 | keys1 == {("a" * 64, seed) for seed in range(30)}
    assert [row["test_seed"] for row in worker0] == list(range(0, 30, 2))
    assert [row["test_seed"] for row in worker1] == list(range(1, 30, 2))
    assert len(worker0) == len(worker1) == 15


def test_task_result_validation_rejects_corrupt_source_and_seed() -> None:
    task = {
        "dataset": "citeseer",
        "candidate_fingerprint": "a" * 64,
        "test_seed": 0,
        "source_id": "b" * 64,
        "config_fingerprint": "c" * 64,
    }
    row = {field: None for field in pipeline.TEST_LONG_FIELDS}
    row.update(
        {
            **task,
            "test_accuracy": 0.5,
            "training_seed": pipeline.expected_test_training_seed("a" * 64, 0),
            "worker_id": 0,
            "gpu_name": "GPU",
            "gpu_uuid": "GPU-UUID",
            "deterministic_provenance": {
                "deterministic_algorithms_enabled": True,
                "deterministic_warn_only": False,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
            },
            "completed": True,
        }
    )
    pipeline.validate_task_result(row, task, expected_worker=0)
    with pytest.raises(ValueError, match="source_id mismatch"):
        pipeline.validate_task_result({**row, "source_id": "d" * 64}, task)
    with pytest.raises(ValueError, match="training_seed mismatch"):
        pipeline.validate_task_result({**row, "training_seed": 1}, task)


def test_oracle_tie_breaks_by_sample_std_then_fingerprint() -> None:
    rows = [
        {"candidate_fingerprint": "c" * 64, "mean_test_accuracy": 0.8, "sample_std_test_accuracy": 0.02},
        {"candidate_fingerprint": "b" * 64, "mean_test_accuracy": 0.8, "sample_std_test_accuracy": 0.01},
        {"candidate_fingerprint": "a" * 64, "mean_test_accuracy": 0.8, "sample_std_test_accuracy": 0.01},
    ]
    assert pipeline.select_test_oracle_winner(rows)["candidate_fingerprint"] == "a" * 64


def test_atomic_json_rename_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "dataset" / "candidate" / "seed_0.json"
    pipeline.atomic_json(path, {"completed": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"completed": True}
    assert not list(path.parent.glob("*.tmp"))
    assert not list(path.parent.glob(".*.tmp"))


def test_mock_two_worker_end_to_end_produces_12_winners_and_360_seed_rows(tmp_path: Path) -> None:
    source_rows, unique_rows = _prepare_mock_top10(tmp_path)
    assert len(source_rows) == 600
    assert len(unique_rows) == 40
    manifest = pipeline.build_test_task_manifest(tmp_path)
    assert manifest["task_count"] == 1200
    assert manifest["maximum_task_count"] == 18000
    worker0 = pipeline.assigned_tasks(manifest, 0, 2)
    worker1 = pipeline.assigned_tasks(manifest, 1, 2)
    assert len(worker0) == len(worker1) == 600
    pipeline.write_mock_task_results(tmp_path)
    audit = pipeline.audit_test30(tmp_path)
    assert audit["task_count"] == 1200
    result = pipeline.summarize_test30(tmp_path)
    assert result["winner_rows"] == 12
    assert result["winner_seed_rows"] == 360
    with (tmp_path / "final_test" / "method_test_oracle_winners.csv").open(newline="", encoding="utf-8") as handle:
        winners = list(csv.DictReader(handle))
    with (tmp_path / "final_test" / "method_test_oracle_winner_seed_accuracies.csv").open(newline="", encoding="utf-8") as handle:
        seed_rows = list(csv.DictReader(handle))
    assert len(winners) == 12
    assert len(seed_rows) == 360
    for dataset in pipeline.DATASETS:
        for method in pipeline.METHODS:
            seeds = [int(row["test_seed"]) for row in seed_rows if row["dataset"] == dataset and row["method"] == method]
            assert seeds == list(range(30))
    report = (tmp_path / "final_test" / "report.md").read_text(encoding="utf-8")
    assert "selection_used_test = true" in report
    assert "result_type = test_oracle_diagnostic" in report
    assert "paper_claim_eligible = false" in report
