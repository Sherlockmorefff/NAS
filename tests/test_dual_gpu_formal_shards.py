from __future__ import annotations

import copy
from pathlib import Path

import pytest

import deterministic_three_strategy_pipeline as pipeline
import formal_matrix


def _parent_manifest() -> dict:
    budgets = {"S0": (50, 0, 250), "G100": (50, 100, 150), "G150": (50, 150, 100)}
    tasks = []
    for dataset in pipeline.DATASETS:
        for method in pipeline.METHODS:
            seed_full, expansion_full, online_full = budgets[method]
            for seed in pipeline.SEARCH_SEEDS:
                task_id = f"{dataset}__{method}__search_seed{seed}"
                tasks.append(
                    {
                        "task_id": task_id,
                        "dataset": dataset,
                        "method": method,
                        "search_seed": seed,
                        "output_directory": f"/new/results/{task_id}",
                        "log_directory": f"/new/logs/{task_id}",
                        "candidate_budget": 300,
                        "seed_full_count": seed_full,
                        "expansion_full_count": expansion_full,
                        "online_full_count": online_full,
                        "n_lhs_candidates": 768,
                        "exact_argv": ["python", "cross_dataset_runner.py", "--execute"],
                        "status": "not_started",
                    }
                )
    return {
        "source_id": "a" * 64,
        "configuration_fingerprint": "b" * 64,
        "tasks": tasks,
    }


def _assignments() -> dict:
    return {
        0: {"gpu_uuid": "GPU-ZERO"},
        1: {"gpu_uuid": "GPU-ONE"},
    }


def _build():
    parent = _parent_manifest()
    shards, audit = formal_matrix.build_formal_shards(
        parent,
        parent_matrix_sha256="c" * 64,
        gpu_assignments=_assignments(),
    )
    return parent, shards, audit


def test_dual_shards_are_exact_disjoint_ordered_partition() -> None:
    parent, shards, audit = _build()
    gpu0 = shards["gpu0_citeseer_pubmed"]
    gpu1 = shards["gpu1_dblp_flickr"]
    ids0 = [task["task_id"] for task in gpu0["tasks"]]
    ids1 = [task["task_id"] for task in gpu1["tasks"]]
    parent_ids = [task["task_id"] for task in parent["tasks"]]
    assert len(parent_ids) == 60
    assert len(ids0) == len(ids1) == 30
    assert not set(ids0) & set(ids1)
    assert set(ids0) | set(ids1) == set(parent_ids)
    assert ids0 == [task["task_id"] for task in parent["tasks"] if task["dataset"] in {"citeseer", "pubmed"}]
    assert ids1 == [task["task_id"] for task in parent["tasks"] if task["dataset"] in {"dblp", "flickr"}]
    assert audit["intersection_task_count"] == 0
    assert audit["union_task_count"] == 60
    assert audit["total_requested_full_evaluations"] == 18_000


def test_shards_preserve_scientific_task_records_and_budgets() -> None:
    parent, shards, _audit = _build()
    parent_by_id = {task["task_id"]: task for task in parent["tasks"]}
    for shard in shards.values():
        assert sum(task["candidate_budget"] for task in shard["tasks"]) == 9_000
        for task in shard["tasks"]:
            assert task == parent_by_id[task["task_id"]]
            assert task["seed_full_count"] + task["expansion_full_count"] + task["online_full_count"] == 300


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("parent_matrix_sha256", "parent matrix"),
        ("source_id", "source ID"),
        ("configuration_fingerprint", "method config"),
        ("shard_payload_sha256", "payload SHA-256"),
    ],
)
def test_shard_rejects_identity_tampering(field: str, message: str) -> None:
    parent, shards, _audit = _build()
    shard = copy.deepcopy(shards["gpu0_citeseer_pubmed"])
    shard[field] = "d" * 64
    with pytest.raises(ValueError, match=message):
        formal_matrix.audit_formal_shard(
            parent,
            shard,
            expected_parent_matrix_sha256="c" * 64,
        )


def test_shard_rejects_reordered_or_regenerated_task_record() -> None:
    parent, shards, _audit = _build()
    reordered = copy.deepcopy(shards["gpu0_citeseer_pubmed"])
    reordered["tasks"][0], reordered["tasks"][1] = reordered["tasks"][1], reordered["tasks"][0]
    reordered["shard_payload_sha256"] = formal_matrix.shard_payload_sha256(reordered)
    with pytest.raises(ValueError, match="ordered parent task"):
        formal_matrix.audit_formal_shard(parent, reordered, expected_parent_matrix_sha256="c" * 64)


def test_shard_files_and_fixed_worker_paths_are_isolated(tmp_path: Path) -> None:
    _parent, shards, _audit = _build()
    hashes = formal_matrix.write_formal_shards(shards, output_root=tmp_path)
    assert set(hashes) == {"gpu0_citeseer_pubmed", "gpu1_dblp_flickr"}
    paths0 = pipeline.dual_worker_paths(tmp_path, 0)
    paths1 = pipeline.dual_worker_paths(tmp_path, 1)
    assert set(paths0.values()).isdisjoint(set(paths1.values()))
    assert paths0["manifest"].is_file() and paths1["manifest"].is_file()


def test_worker_manifest_loader_enforces_own_worker_and_sidecar(tmp_path: Path) -> None:
    parent = _parent_manifest()
    pipeline.atomic_json(tmp_path / "formal_matrix_60.json", parent)
    shards, _audit = formal_matrix.build_formal_shards(
        parent,
        parent_matrix_sha256=pipeline.sha256_file(tmp_path / "formal_matrix_60.json"),
        gpu_assignments=_assignments(),
    )
    formal_matrix.write_formal_shards(shards, output_root=tmp_path)
    tasks, sharded = pipeline._load_search_tasks(
        root=tmp_path,
        task_manifest=pipeline.dual_worker_paths(tmp_path, 0)["manifest"],
        worker_id=0,
        expected_gpu_uuid="GPU-ZERO",
    )
    assert sharded is True and len(tasks) == 30
    with pytest.raises(RuntimeError, match="fixed worker assignment|worker ID"):
        pipeline._load_search_tasks(
            root=tmp_path,
            task_manifest=pipeline.dual_worker_paths(tmp_path, 0)["manifest"],
            worker_id=1,
            expected_gpu_uuid="GPU-ZERO",
        )


def test_queue_and_task_locks_reject_duplicate_but_allow_distinct(tmp_path: Path) -> None:
    lock0 = tmp_path / "locks" / "worker0.lock"
    lock1 = tmp_path / "locks" / "worker1.lock"
    with pipeline.exclusive_file_lock(lock0, owner={"worker": 0}):
        with pytest.raises(RuntimeError, match="already held"):
            with pipeline.exclusive_file_lock(lock0, owner={"worker": 0}):
                pass
        with pipeline.exclusive_file_lock(lock1, owner={"worker": 1}):
            assert lock0.read_text() and lock1.read_text()


def test_incomplete_output_is_not_completed_and_complete_is_skipped(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "partial.json").write_text("{}", encoding="utf-8")
    task = {"task_id": "partial", "output_directory": str(output)}
    with pytest.raises(FileNotFoundError, match="incomplete search task"):
        pipeline._complete_search_task(task, "a" * 64)
    monkeypatch.setattr(pipeline, "audit_search_history", lambda *_args: {"status": "passed"})
    assert pipeline._complete_search_task(task, "a" * 64) is True


def test_gpu_uuid_and_worker_assignment_are_enforced(monkeypatch) -> None:
    acceptance = {
        "details": {
            "mode": "fixed_dataset_dual_gpu",
            "cross_gpu_bitwise_exact": True,
            "gpu_assignments": {
                "0": {"gpu_index": 0, "gpu_uuid": "GPU-ZERO", "datasets": ["citeseer", "pubmed"], "software_hardware": {"gpu_name": "GPU", "driver_version": "1"}},
                "1": {"gpu_index": 1, "gpu_uuid": "GPU-ONE", "datasets": ["dblp", "flickr"], "software_hardware": {"gpu_name": "GPU", "driver_version": "1"}},
            },
        }
    }
    monkeypatch.setattr(
        "deterministic_runtime._nvidia_rows",
        lambda: ([
            {"index": 0, "gpu_uuid": "GPU-ZERO", "gpu_name": "GPU", "driver_version": "1"},
            {"index": 1, "gpu_uuid": "GPU-ONE", "gpu_name": "GPU", "driver_version": "1"},
        ], None),
    )
    assert pipeline.verify_search_gpu_matches_acceptance("0", acceptance, expected_gpu_uuid="GPU-ZERO", worker_id=0)["gpu_uuid"] == "GPU-ZERO"
    with pytest.raises(RuntimeError, match="UUID"):
        pipeline.verify_search_gpu_matches_acceptance("0", acceptance, expected_gpu_uuid="GPU-ONE", worker_id=0)


def test_single_gpu_cli_requires_protocol_identity() -> None:
    args = pipeline.parse_args(
        [
            "--protocol-id",
            "deterministic-three-strategy_v1_20260813_64392499",
            "run-search-queue",
            "--gpu-id",
            "0",
        ]
    )
    assert args.gpu_id == "0"
    assert args.task_manifest is None
    assert args.worker_id is None
