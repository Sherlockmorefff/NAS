#!/usr/bin/env python3
"""Audit validation-best checkpoints and freeze one-time reveal readiness."""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from analyse import freeze_final_validation_checkpoints as plan
from analyse import run_final_validation_checkpoints as runner
from analyse import reveal_frozen_checkpoints_once as reveal_protocol


def _load_raw_results(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    identities = []
    for shard in (0, 1):
        path = root / "training" / f"gpu{shard}" / "seed_results.jsonl"
        identities.append(plan.file_identity(path))
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if plan.seed_inventory.has_forbidden_json_key(line):
                    raise RuntimeError(f"forbidden metric key in training result {path}:{line_number}")
                row = json.loads(line)
                if plan._contains_forbidden_metric_key(row):
                    raise RuntimeError(f"forbidden metric field in training result {path}:{line_number}")
                rows.append(row)
    return rows, identities


def _checkpoint_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in payload.items()
        if key not in {"state_dict", "checkpoint_payload_identity_sha256"}
    }


def protected_after(root: Path) -> dict[str, Any]:
    before = plan.load_json(root / "protected_hashes_before.json")
    mismatches = []
    current = []
    for row in before["objects"]:
        path = Path(row["path"])
        if not path.is_file():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        identity = plan.file_identity(path)
        current.append(identity)
        if identity["size_bytes"] != row["size_bytes"] or identity["sha256"] != row["sha256"]:
            mismatches.append({"path": str(path), "reason": "identity_mismatch"})
    return {
        "format_version": plan.FORMAT_VERSION,
        "created_at": plan.utc_now(),
        "object_count": len(before["objects"]),
        "current_objects": current,
        "mismatches": mismatches,
        "passed": not mismatches and len(current) == len(before["objects"]),
    }


def audit_training(root: Path, *, write_outputs: bool = True) -> dict[str, Any]:
    freeze = plan.verify_manifest(root, root / "pretraining_immutable_manifest.json")
    if not freeze["passed"]:
        raise RuntimeError("pretraining immutable manifest drift")
    tasks = plan.load_json(root / "training_tasks.json")
    if len(tasks) != 60 or len({row["task_key"] for row in tasks}) != 60:
        raise RuntimeError("training task manifest is incomplete or duplicated")
    rows, raw_identities = _load_raw_results(root)
    if len(rows) != 60 or len({row["task_key"] for row in rows}) != 60:
        raise RuntimeError("raw training result count/key mismatch")
    task_by_key = {row["task_key"]: row for row in tasks}
    result_by_key = {row["task_key"]: row for row in rows}
    if set(task_by_key) != set(result_by_key):
        raise RuntimeError("training task/result key sets differ")
    if any(row.get("terminal_status") != "success" or not row.get("valid") for row in rows):
        raise RuntimeError("non-success validation training result")
    if {int(row["seed"]) for row in rows} != set(plan.FRESH_SEEDS):
        raise RuntimeError("fresh paired seed set mismatch")
    if any(sum(int(row["seed"]) == seed for row in rows) != 2 for seed in plan.FRESH_SEEDS):
        raise RuntimeError("P/Q seed pairing mismatch")
    if any(set(row) != runner.RESULT_KEYS for row in rows):
        raise RuntimeError("training result exact-key allowlist mismatch")
    expected_record_paths = {
        runner._record_path(root, int(task["assigned_physical_gpu_index"]), task).resolve()
        for task in tasks
    }
    actual_record_paths = {
        path.resolve() for shard in (0, 1)
        for path in (root / "training" / f"gpu{shard}" / "records").glob("*.json")
    }
    if actual_record_paths != expected_record_paths:
        raise RuntimeError("per-task record file set differs from frozen task manifest")
    for shard in (0, 1):
        shard_tasks = sorted(
            (row for row in tasks if int(row["assigned_physical_gpu_index"]) == shard),
            key=lambda row: int(row["ordinal_on_gpu"]),
        )
        if len(shard_tasks) != 30 or [int(row["ordinal_on_gpu"]) for row in shard_tasks] != list(range(30)):
            raise RuntimeError(f"GPU{shard} frozen schedule mismatch")
        for index in range(0, 30, 2):
            pair = shard_tasks[index:index + 2]
            if len({int(row["pair_id"]) for row in pair}) != 1 or [int(row["within_pair_order"]) for row in pair] != [0, 1]:
                raise RuntimeError(f"GPU{shard} pair adjacency mismatch")

    sealed_cpu = runner.load_sealed_worker_view(root / "sealed/train_val_sealed_view.pt")
    sealed_identity = plan.load_json(root / "sealed/train_val_sealed_view_identity.json")
    plan.verify_sealed_identity(
        sealed_cpu, root / "sealed/train_val_sealed_view.pt", sealed_identity,
    )
    candidates = {
        "P": plan.load_json(root / "frozen_candidate_P.json"),
        "Q": plan.load_json(root / "frozen_candidate_Q.json"),
    }
    pretraining_root = plan.load_json(root / "pretraining_immutable_manifest.json")["manifest_root_hash"]
    checkpoint_rows = []
    total_retries = 0
    for task_key in sorted(task_by_key):
        task = task_by_key[task_key]
        result = result_by_key[task_key]
        for key in (
            "seed", "role", "strategy", "joint_config_fingerprint",
            "assigned_physical_gpu_index", "assigned_gpu_uuid", "ordinal_on_gpu",
        ):
            if result[key] != task[key]:
                raise RuntimeError(f"task/result field mismatch {task_key}: {key}")
        total_retries += int(result["retry_count"])
        record_path = runner._record_path(
            root, int(task["assigned_physical_gpu_index"]), task,
        )
        if plan.stable_hash(plan.load_json(record_path)) != plan.stable_hash(result):
            raise RuntimeError(f"JSONL/per-task record mismatch: {task_key}")
        actual_gpu = result.get("actual_gpu")
        if not isinstance(actual_gpu, Mapping) or (
            int(actual_gpu.get("physical_index", -1)) != int(task["assigned_physical_gpu_index"])
            or actual_gpu.get("uuid") != task["assigned_gpu_uuid"]
        ):
            raise RuntimeError(f"actual physical GPU mismatch: {task_key}")
        relative_checkpoint = Path(str(result["checkpoint_relative_path"]))
        if relative_checkpoint.is_absolute() or ".." in relative_checkpoint.parts:
            raise RuntimeError(f"checkpoint path escapes result root: {task_key}")
        checkpoint = (root / relative_checkpoint).resolve()
        if checkpoint != runner.checkpoint_path_for(root, task).resolve():
            raise RuntimeError(f"checkpoint path differs from frozen task path: {task_key}")
        if not checkpoint.is_file():
            raise RuntimeError(f"checkpoint missing: {task_key}")
        if checkpoint.stat().st_size != int(result["checkpoint_size_bytes"]):
            raise RuntimeError(f"checkpoint size mismatch: {task_key}")
        if plan.sha256_file(checkpoint) != result["checkpoint_file_sha256"]:
            raise RuntimeError(f"checkpoint file hash mismatch: {task_key}")
        payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if not isinstance(payload, Mapping) or set(payload) != runner.CHECKPOINT_PAYLOAD_KEYS:
            raise RuntimeError(f"checkpoint payload exact-key mismatch: {task_key}")
        if plan._contains_forbidden_metric_key({key: value for key, value in payload.items() if key != "state_dict"}):
            raise RuntimeError(f"checkpoint metadata contains forbidden metric field: {task_key}")
        if payload["pretraining_immutable_root"] != pretraining_root:
            raise RuntimeError(f"checkpoint embeds wrong pretraining root: {task_key}")
        expected_payload_fields = {
            "role": result["role"],
            "strategy": result["strategy"],
            "pair_id": int(result["pair_id"]),
            "seed": int(result["seed"]),
            "joint_config_fingerprint": result["joint_config_fingerprint"],
            "protocol_sha256": plan.file_identity(root / "protocol.json")["sha256"],
            "dataset_content_identity": plan.DATASET_CONTENT_ID,
            "public_split_identity": plan.PUBLIC_SPLIT_ID,
            "sealed_view_combined_tensor_root": sealed_identity["combined_tensor_root"],
            "sealed_view_file_sha256": sealed_identity["sealed_file"]["sha256"],
            "best_epoch": int(result["best_epoch"]),
            "best_validation": float(result["best_validation"]),
        }
        if any(payload.get(key) != value for key, value in expected_payload_fields.items()):
            raise RuntimeError(f"checkpoint metadata/task mismatch: {task_key}")
        if payload["checkpoint_payload_identity_sha256"] != plan.stable_hash(_checkpoint_metadata(payload)):
            raise RuntimeError(f"checkpoint payload identity mismatch: {task_key}")
        state_hash = runner.tensor_state_hash(payload["state_dict"])
        if state_hash != result["best_state_tensor_sha256"] or state_hash != payload["best_state_tensor_sha256"]:
            raise RuntimeError(f"checkpoint tensor-state mismatch: {task_key}")
        trajectory = result["trajectory"]
        if not trajectory or any(set(event) != {"epoch_1based", "loss", "validation_accuracy"} for event in trajectory):
            raise RuntimeError(f"trajectory schema mismatch: {task_key}")
        if [int(event["epoch_1based"]) for event in trajectory] != list(range(1, len(trajectory) + 1)):
            raise RuntimeError(f"trajectory epoch sequence mismatch: {task_key}")
        if runner.trajectory_hash(trajectory) != result.get("trajectory_sha256") or len(trajectory) != int(result["epochs_ran"]):
            raise RuntimeError(f"trajectory identity mismatch: {task_key}")
        if (
            payload.get("trajectory_sha256") != result.get("trajectory_sha256")
            or int(payload.get("epochs_ran", -1)) != int(result["epochs_ran"])
            or int(payload.get("stopped_epoch", -1)) != int(result["epochs_ran"])
        ):
            raise RuntimeError(f"checkpoint/result trajectory binding mismatch: {task_key}")
        values = [float(event["validation_accuracy"]) for event in trajectory]
        best_value = max(values)
        earliest_epoch = values.index(best_value) + 1
        if earliest_epoch != int(result["best_epoch"]) or best_value != float(result["best_validation"]):
            raise RuntimeError(f"validation-best epoch rule mismatch: {task_key}")
        if int(result["best_epoch"]) != int(result["epochs_ran"]) and result["best_state_tensor_sha256"] == result["last_state_tensor_sha256"]:
            raise RuntimeError(f"non-final best checkpoint equals last state: {task_key}")
        model = runner.build_model(candidates[result["role"]], sealed_cpu, torch.device("cpu"))
        model.load_state_dict(payload["state_dict"], strict=True)
        disk_validation = runner.validation_accuracy(model, sealed_cpu)
        if disk_validation != float(result["best_validation"]) or disk_validation != float(result["disk_reload_validation"]):
            raise RuntimeError(f"validation recomputation mismatch: {task_key}")
        checkpoint_rows.append({
            "task_key": task_key,
            "pair_id": int(result["pair_id"]),
            "seed": int(result["seed"]),
            "role": result["role"],
            "strategy": result["strategy"],
            "architecture_fingerprint": candidates[result["role"]]["candidate_config"]["architecture_fingerprint"],
            "hp_fingerprint": candidates[result["role"]]["candidate_config"]["hp_fingerprint"],
            "joint_config_fingerprint": result["joint_config_fingerprint"],
            "best_epoch": int(result["best_epoch"]),
            "best_validation": float(result["best_validation"]),
            "checkpoint_relative_path": result["checkpoint_relative_path"],
            "checkpoint_size_bytes": int(result["checkpoint_size_bytes"]),
            "checkpoint_file_sha256": result["checkpoint_file_sha256"],
            "best_state_tensor_sha256": state_hash,
            "checkpoint_payload_identity_sha256": result["checkpoint_payload_identity_sha256"],
            "assigned_physical_gpu_index": int(result["assigned_physical_gpu_index"]),
            "assigned_gpu_uuid": result["assigned_gpu_uuid"],
            "started_at_utc": result["started_at_utc"],
            "completed_at_utc": result["completed_at_utc"],
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "retry_count": int(result["retry_count"]),
        })

    for seed in plan.FRESH_SEEDS:
        pair = [row for row in checkpoint_rows if row["seed"] == seed]
        if {row["role"] for row in pair} != {"P", "Q"}:
            raise RuntimeError(f"paired role mismatch for seed {seed}")
        if len({row["assigned_gpu_uuid"] for row in pair}) != 1:
            raise RuntimeError(f"paired physical GPU mismatch for seed {seed}")

    protected = protected_after(root)
    if not protected["passed"]:
        raise RuntimeError("protected object drift")
    claim_dir = root / "reveal_claim"
    claim_entries = sorted(str(path.relative_to(root)) for path in claim_dir.rglob("*") if path.is_file())
    if claim_entries:
        raise RuntimeError("reveal claim directory is not empty")

    validation_by_role = {}
    for role in ("P", "Q"):
        values = [row["best_validation"] for row in checkpoint_rows if row["role"] == role]
        validation_by_role[role] = {
            "count": len(values), "mean": float(sum(values) / len(values)),
            "minimum": float(min(values)), "maximum": float(max(values)),
        }
    started_at = min(row["started_at_utc"] for row in checkpoint_rows)
    completed_at = max(row["completed_at_utc"] for row in checkpoint_rows)
    wall_seconds = (
        datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)
    ).total_seconds()
    audit = {
        "format_version": plan.FORMAT_VERSION,
        "created_at": plan.utc_now(),
        "pretraining_immutable_audit": freeze,
        "task_count": len(tasks), "result_count": len(rows),
        "checkpoint_count": len(checkpoint_rows),
        "success_count": len(rows), "failure_count": 0,
        "retry_count": total_retries,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "wall_seconds": wall_seconds,
        "fresh_seed_min": min(plan.FRESH_SEEDS),
        "fresh_seed_max": max(plan.FRESH_SEEDS),
        "paired_seed_count": len(plan.FRESH_SEEDS),
        "raw_result_files": raw_identities,
        "validation_summary": validation_by_role,
        "sealed_view_combined_tensor_root": sealed_identity["combined_tensor_root"],
        "checkpoint_rows": checkpoint_rows,
        "protected_hashes_after_passed": protected["passed"],
        "reveal_claim_directory_empty": True,
        "real_benchmark_access_count": 0,
        "passed": True,
    }
    if write_outputs:
        plan.dump_json(root / "training_completion_audit.json", audit)
        plan.dump_json(root / "protected_hashes_after.json", protected)
        plan.dump_json(root / "checkpoint_inventory.json", checkpoint_rows)
        plan.dump_csv(root / "checkpoint_inventory.csv", checkpoint_rows)
    return audit


def _ready_manifest(root: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    source_manifest = plan.load_json(root / "source_dependency_manifest.json")
    source_by_relative = {row["relative_path"]: row for row in source_manifest["files"]}
    frozen_source_rows = []
    for relative in (
        "analyse/reveal_frozen_checkpoints_once.py",
        "analyse/analyse_final_checkpoint_reveal.py",
        "analyse/frozen_cora_benchmark_reveal_adapter.py",
        "tests/test_final_checkpoint_reveal.py",
    ):
        row = source_by_relative.get(relative)
        if row is None:
            raise RuntimeError(f"future reveal source missing from dependency manifest: {relative}")
        if plan.sha256_file(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"future reveal source drift: {relative}")
        frozen_source_rows.append({"relative_path": relative, "sha256": row["sha256"], "size_bytes": row["size_bytes"]})
    base_files = [
        "protocol.json", "preregistration.json", "frozen_candidate_P.json",
        "frozen_candidate_Q.json", "paired_seed_registry.json", "training_tasks.json",
        "sealed/train_val_sealed_view.pt", "sealed/train_val_sealed_view_identity.json",
        "pretraining_immutable_manifest.json", "historical_training_seed_inventory.json",
        "input_evidence_audit.json", "source_dependency_manifest.json",
        "training_completion_audit.json", "protected_hashes_after.json",
        "checkpoint_inventory.json", "independent_data_execution_audit.md",
        "independent_checkpoint_readiness_audit.md",
    ]
    files = []
    for relative in base_files:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"ready input missing: {relative}")
        files.append({"relative_path": relative, **plan.file_identity(path)})
    for row in audit["checkpoint_rows"]:
        path = root / row["checkpoint_relative_path"]
        files.append({"relative_path": row["checkpoint_relative_path"], **plan.file_identity(path)})
    compact = [
        {"relative_path": row["relative_path"], "size_bytes": row["size_bytes"], "sha256": row["sha256"]}
        for row in sorted(files, key=lambda row: row["relative_path"])
    ]
    root_hash = plan.stable_hash({"files": compact, "frozen_future_sources": frozen_source_rows})
    return {
        "format_version": plan.FORMAT_VERSION,
        "created_at": plan.utc_now(),
        "files": sorted(files, key=lambda row: row["relative_path"]),
        "frozen_future_sources": frozen_source_rows,
        "checkpoint_count": len(audit["checkpoint_rows"]),
        "manifest_root_hash": root_hash,
    }


def finalize_ready(root: Path) -> dict[str, Any]:
    audit = audit_training(root, write_outputs=True)
    manifest = _ready_manifest(root, audit)
    plan.dump_json(root / "ready_immutable_manifest.json", manifest)
    checkpoint_records = [
        {
            "method_role": row["role"],
            "seed": int(row["seed"]),
            "joint_fingerprint": row["joint_config_fingerprint"],
            "relative_path": row["checkpoint_relative_path"],
            "checkpoint_sha256": row["checkpoint_file_sha256"],
            "checkpoint_size": int(row["checkpoint_size_bytes"]),
            "best_state_tensor_sha256": row["best_state_tensor_sha256"],
            "best_epoch": int(row["best_epoch"]),
            "best_validation": float(row["best_validation"]),
            "assigned_gpu_uuid": row["assigned_gpu_uuid"],
        }
        for row in sorted(audit["checkpoint_rows"], key=lambda row: (int(row["seed"]), row["role"]))
    ]
    candidate_documents = {
        role: {
            "joint_fingerprint": expected,
            "strategy": plan.load_json(root / f"frozen_candidate_{role}.json")["strategy"],
        }
        for role, expected in (("P", plan.EXPECTED_P), ("Q", plan.EXPECTED_Q))
    }
    frozen_source_by_relative = {
        row["relative_path"]: row for row in manifest["frozen_future_sources"]
    }
    adapter_source = frozen_source_by_relative["analyse/frozen_cora_benchmark_reveal_adapter.py"]
    ready = reveal_protocol.seal_ready_document({
        "format_version": plan.FORMAT_VERSION,
        "created_at": plan.utc_now(),
        "ready_schema": reveal_protocol.READY_SCHEMA,
        "ready_root_scheme": reveal_protocol.READY_ROOT_SCHEME,
        "status": "TEST_REVEAL_READY",
        "classification": "TEST_REVEAL_READY",
        "method_level_state": reveal_protocol.METHOD_LEVEL_STATE,
        "method_level_status": reveal_protocol.METHOD_LEVEL_STATE,
        "method_name": plan.METHOD_NAME,
        "P_joint_config_fingerprint": plan.EXPECTED_P,
        "Q_joint_config_fingerprint": plan.EXPECTED_Q,
        "candidates": candidate_documents,
        "final_candidates_identical": plan.EXPECTED_P == plan.EXPECTED_Q,
        "fresh_seeds": list(plan.FRESH_SEEDS),
        "fresh_paired_seeds": list(plan.FRESH_SEEDS),
        "checkpoint_count": len(audit["checkpoint_rows"]),
        "checkpoint_records": checkpoint_records,
        "fixed_reveal_paths": {
            "claim": "reveal_claim/ONE_TIME_REVEAL_CLAIM.json",
            "result": "reveal_claim/ONE_TIME_REVEAL_RESULT.json",
            "failure": "reveal_claim/ONE_TIME_REVEAL_TERMINAL_FAILURE.json",
        },
        "frozen_benchmark_adapter": {
            "relative_path": adapter_source["relative_path"],
            "sha256": adapter_source["sha256"],
            "size_bytes": int(adapter_source["size_bytes"]),
        },
        "frozen_future_sources": manifest["frozen_future_sources"],
        "all_checkpoint_file_and_tensor_hashes_verified": True,
        "ready_manifest_root": manifest["manifest_root_hash"],
        "real_benchmark_access_count": 0,
        "reveal_claim_directory_empty": True,
        "future_scope": (
            "prospective-after-freeze benchmark reveal of two frozen winner configurations; "
            "project-level benchmark data are not untouched; not a retest of the search method"
        ),
        "next_action": "a separately dispatched one-time reveal task may verify this root and then claim access exactly once",
    })
    plan.dump_json(root / "TEST_REVEAL_READY.json", ready)
    report = (
        "# Final validation checkpoint freeze\n\n"
        "Classification: **TEST_REVEAL_READY**\n\n"
        f"P={plan.EXPECTED_P}; Q={plan.EXPECTED_Q}.\n\n"
        f"Sixty validation-only best checkpoints were frozen for paired seeds {plan.FRESH_SEEDS[0]}..{plan.FRESH_SEEDS[-1]}. "
        "All file and tensor-state hashes passed, and the real benchmark access count remains zero.\n\n"
        "Any future reveal is a prospective-after-freeze benchmark reveal of two frozen winner configurations; "
        "the project-level benchmark data are not untouched and the reveal does not retest the search method.\n"
    )
    (root / "validation_report.md").write_text(report, encoding="utf-8")
    (root / "final_goal_report.md").write_text(report, encoding="utf-8")
    return ready


def package(root: Path, archive_name: str) -> Path:
    ready = plan.load_json(root / "TEST_REVEAL_READY.json")
    if (
        ready.get("status") != "TEST_REVEAL_READY"
        or ready.get("classification") != "TEST_REVEAL_READY"
        or ready.get("real_benchmark_access_count") != 0
    ):
        raise RuntimeError("package requires TEST_REVEAL_READY with zero real benchmark access")
    audit = audit_training(root, write_outputs=False)
    ready_manifest = plan.load_json(root / "ready_immutable_manifest.json")
    compact = []
    for row in ready_manifest["files"]:
        path = root / row["relative_path"]
        current = plan.file_identity(path)
        if current["size_bytes"] != row["size_bytes"] or current["sha256"] != row["sha256"]:
            raise RuntimeError(f"ready manifest file drift: {row['relative_path']}")
        compact.append({
            "relative_path": row["relative_path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        })
    frozen_future_sources = ready_manifest["frozen_future_sources"]
    recomputed_manifest_root = plan.stable_hash({
        "files": sorted(compact, key=lambda row: row["relative_path"]),
        "frozen_future_sources": frozen_future_sources,
    })
    if (
        recomputed_manifest_root != ready_manifest["manifest_root_hash"]
        or recomputed_manifest_root != ready["ready_manifest_root"]
    ):
        raise RuntimeError("ready manifest root mismatch")
    if reveal_protocol.compute_ready_root(ready) != ready.get("immutable_ready_root"):
        raise RuntimeError("ready document root mismatch")
    reveal_protocol.validate_ready_bundle(
        root / "TEST_REVEAL_READY.json",
        expected_ready_root=ready["immutable_ready_root"],
    )
    dependency = plan.load_json(root / "source_dependency_manifest.json")
    snapshot_root = root / "source_snapshot"
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    snapshot_rows = []
    for row in dependency["files"]:
        source = Path(row["path"])
        if plan.sha256_file(source) != row["sha256"]:
            raise RuntimeError(f"source dependency drift: {source}")
        if not row["requires_delivery_snapshot"]:
            continue
        target = snapshot_root / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshot_rows.append({
            "relative_path": row["relative_path"],
            "source_sha256": row["sha256"],
            "snapshot_sha256": plan.sha256_file(target),
            "three_way_match": row["sha256"] == plan.sha256_file(source) == plan.sha256_file(target),
        })
    snapshot_audit = {
        "format_version": plan.FORMAT_VERSION,
        "created_at": plan.utc_now(),
        "required_count": dependency["snapshot_required_count"],
        "snapshot_count": len(snapshot_rows),
        "files": snapshot_rows,
        "passed": len(snapshot_rows) == dependency["snapshot_required_count"] and all(row["three_way_match"] for row in snapshot_rows),
    }
    plan.dump_json(root / "source_snapshot_audit.json", snapshot_audit)
    if not snapshot_audit["passed"]:
        raise RuntimeError("source snapshot audit failed")
    archive = root / archive_name
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    plan.dump_json(root / "delivery_package.json", {
        "format_version": plan.FORMAT_VERSION,
        "created_at": plan.utc_now(),
        "archive_name": archive.name,
        "includes_60_validation_best_checkpoints": len(audit["checkpoint_rows"]) == 60,
        "includes_sealed_train_validation_view": True,
        "includes_all_non_head_recoverable_source_dependencies": True,
        "omitted_clean_tracked_sources_require_frozen_git_head": True,
        "standalone_source_reproduction_claimed": False,
        "real_benchmark_access_count": 0,
        "immutable_ready_root": ready["immutable_ready_root"],
        "ready_manifest_root": ready["ready_manifest_root"],
    })
    exclusions = {archive.resolve(), sidecar.resolve(), (root / "artifact_manifest.json").resolve()}
    payloads = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() not in exclusions
        and not path.name.endswith(".tar.gz") and not path.name.endswith(".tar.gz.sha256")
    )
    manifest_rows = [
        {"relative_path": str(path.relative_to(root)), "sha256": plan.sha256_file(path), "size_bytes": path.stat().st_size}
        for path in payloads
    ]
    plan.dump_json(root / "artifact_manifest.json", {
        "format_version": plan.FORMAT_VERSION,
        "created_at": plan.utc_now(),
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    })
    payloads.append(root / "artifact_manifest.json")
    if archive.exists():
        archive.unlink()
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in sorted(payloads):
                    info = tar.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    archive_hash = plan.sha256_file(archive)
    sidecar.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        expected = {str(path.relative_to(root)) for path in payloads}
        if any(not member.isfile() for member in members) or {member.name for member in members} != expected:
            raise RuntimeError("archive member audit failed")
        for member in members:
            content = tar.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != plan.sha256_file(root / member.name):
                raise RuntimeError(f"archive payload mismatch: {member.name}")
    print(json.dumps({
        "archive": str(archive), "sha256": archive_hash,
        "member_count": len(payloads), "checkpoint_count": len(audit["checkpoint_rows"]),
        "source_snapshot_count": len(snapshot_rows),
    }, indent=2, sort_keys=True))
    return archive


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-training")
    audit.add_argument("--root", type=Path, required=True)
    ready = sub.add_parser("finalize-ready")
    ready.add_argument("--root", type=Path, required=True)
    package_parser = sub.add_parser("package")
    package_parser.add_argument("--root", type=Path, required=True)
    package_parser.add_argument("--archive-name", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.resolve()
    if args.command == "audit-training":
        print(json.dumps(audit_training(root), indent=2, sort_keys=True))
    elif args.command == "finalize-ready":
        print(json.dumps(finalize_ready(root), indent=2, sort_keys=True))
    elif args.command == "package":
        package(root, args.archive_name)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
