#!/usr/bin/env python3
"""Run one immutable GPU shard for the Original Head attribution v2 labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse.original_head_confirmatory_attribution_v2 import (
    TRAINING_SEEDS,
    load_json,
    stable_hash,
    strict_test_key_hits,
    verify_freeze,
)
from analyse.vae_label_expansion import EXPECTED_DATASET_CONTEXT
from dataset_utils import (
    assert_dataset_context_matches,
    dataset_context_from_manifest,
    load_dataset_from_request,
    resolve_dataset_request,
)
from eval_utils import train_and_eval_arch


RESULTS_FILENAME = "seed_results.jsonl"


class ProtocolFirewallViolation(RuntimeError):
    """A non-retryable label/test/trajectory protocol violation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    payload = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def result_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["pool_id"]),
        str(row["config_id"]),
        str(row["architecture_fingerprint"]),
        str(row["hp_fingerprint"]),
        int(row["seed"]),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    keys: set[tuple[Any, ...]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            hits = strict_test_key_hits(row)
            if hits:
                raise ValueError(f"test-named result fields at {path}:{line_number}: {hits}")
            key = result_key(row)
            if key in keys:
                raise ValueError(f"duplicate seed result at {path}:{line_number}: {key}")
            keys.add(key)
            rows.append(row)
    return rows


def _nonnull_test_values(value: Any, path: str = "root") -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_path = f"{path}.{key}"
            if "test" in str(key).lower() and nested is not None:
                hits.append({"path": nested_path, "value_type": type(nested).__name__})
            hits.extend(_nonnull_test_values(nested, nested_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            hits.extend(_nonnull_test_values(nested, f"{path}[{index}]"))
    return hits


def remove_null_test_keys(value: Any, path: str = "root") -> Any:
    """Delete test-named keys only after proving every such value is None."""
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, nested in value.items():
            nested_path = f"{path}.{key}"
            if "test" in str(key).lower():
                disabled_control = str(key).lower() in {
                    "track_test", "test_tracking", "test_tracking_enabled",
                } and nested is False
                if nested is not None and not disabled_control:
                    raise ProtocolFirewallViolation(
                        f"non-null test value encountered at {nested_path}: {type(nested).__name__}"
                    )
                continue
            cleaned[str(key)] = remove_null_test_keys(nested, nested_path)
        return cleaned
    if isinstance(value, list):
        return [remove_null_test_keys(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return [remove_null_test_keys(item, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def sanitize_training_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the null test trajectory field and hash only the saved trajectory."""
    raw = dict(metadata)
    raw_trajectory = raw.pop("trajectory", None)
    raw.pop("trajectory_hash", None)
    if not isinstance(raw_trajectory, list) or not raw_trajectory:
        raise ProtocolFirewallViolation("return_trajectory=True did not yield a non-empty trajectory")
    nonnull = _nonnull_test_values(raw_trajectory, "metadata.trajectory")
    if nonnull:
        raise ProtocolFirewallViolation(f"trajectory exposed a non-null test value: {nonnull[:3]}")
    sanitized_trajectory: list[dict[str, Any]] = []
    required = {"epoch_1based", "loss", "validation_accuracy"}
    allowed = required | {"test_accuracy"}
    for index, event in enumerate(raw_trajectory):
        if not isinstance(event, Mapping):
            raise ProtocolFirewallViolation(f"trajectory row {index} is not a mapping")
        missing = required - set(event)
        extra = set(event) - allowed
        if missing or extra:
            raise ProtocolFirewallViolation(
                f"trajectory schema mismatch at {index}: missing={missing}, extra={extra}"
            )
        if event.get("test_accuracy") is not None:
            raise ProtocolFirewallViolation(f"trajectory test_accuracy is non-null at row {index}")
        sanitized_trajectory.append({
            "epoch_1based": int(event["epoch_1based"]),
            "loss": float(event["loss"]),
            "validation_accuracy": float(event["validation_accuracy"]),
        })
    cleaned = remove_null_test_keys(raw, "metadata")
    cleaned["trajectory"] = sanitized_trajectory
    cleaned["sanitized_trajectory_hash"] = stable_hash(sanitized_trajectory)
    if strict_test_key_hits(cleaned):
        raise ProtocolFirewallViolation("sanitized training metadata still contains a test-named key")
    return cleaned


def physical_gpu_identity(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("this prospective experiment requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_index = int(visible.split(",")[0]) if visible else int(device.index or 0)
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    for line in result.stdout.splitlines():
        parts = [piece.strip() for piece in line.split(",", 2)]
        if len(parts) == 3 and int(parts[0]) == physical_index:
            return {
                "device": str(device),
                "physical_index": physical_index,
                "uuid": parts[1],
                "name": parts[2],
                "cuda_visible_devices": visible,
            }
    raise RuntimeError(f"could not resolve physical GPU {physical_index}")


def base_result(
    task: Mapping[str, Any], context: Mapping[str, Any], gpu: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "pool_id": int(task["pool_id"]),
        "config_id": str(task["config_id"]),
        "architecture_fingerprint": str(task["architecture_fingerprint"]),
        "hp_fingerprint": str(task["hp_fingerprint"]),
        "joint_config_fingerprint": str(task["joint_config_fingerprint"]),
        "seed": int(task["seed"]),
        "task_key": str(task["task_key"]),
        "assigned_gpu": int(task["assigned_gpu"]),
        "selected_by": list(task["selected_by"]),
        "operations": list(task["operations"]),
        "edges": [list(map(int, edge)) for edge in task["edges"]],
        "effective_layers": int(task["effective_layers"]),
        "raw_hp_vector": [float(value) for value in task["raw_hp_vector"]],
        "lr": float(task["lr"]),
        "dropout": float(task["dropout"]),
        "hidden_dim": int(task["hidden_dim"]),
        "l2": float(task["l2"]),
        "gat_heads": int(task["gat_heads"]),
        "sage_aggr": str(task["sage_aggr"]),
        "gin_eps": float(task["gin_eps"]),
        "gat_heads_by_layer": task["gat_heads_by_layer"],
        "sage_aggr_by_layer": task["sage_aggr_by_layer"],
        "gin_eps_by_layer": task["gin_eps_by_layer"],
        "dataset": context["canonical_name"],
        "dataset_content_fingerprint": context["dataset_content_fingerprint"],
        "split_fingerprint": context["split_fingerprint"],
        "graph_transform": context["graph_transform"],
        "training_mode": context["training_mode"],
        "metric": context["metric"],
        "hp_mode": "global4",
        "normalization": "none",
        "eval_epochs": 300,
        "patience": 80,
        "training_entry": "eval_utils.train_and_eval_arch",
        "actual_gpu": dict(gpu),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    freeze_audit = verify_freeze(root)
    if not freeze_audit["passed"]:
        raise RuntimeError(f"immutable pre-label manifest failed: {freeze_audit}")
    firewall = load_json(root / "pre_label_firewall_audit.json")
    if not firewall.get("passed"):
        raise RuntimeError("pre-label firewall is not passed")
    preregistration = load_json(root / "preregistration.json")
    all_tasks = load_json(root / "selected_union_tasks.json")
    hits = strict_test_key_hits(all_tasks)
    if hits:
        raise RuntimeError(f"frozen task plan contains test-named fields: {hits}")
    tasks = sorted(
        [dict(row) for row in all_tasks if int(row["assigned_gpu"]) == args.shard],
        key=result_key,
    )
    if not tasks:
        raise ValueError(f"shard {args.shard} is empty")
    config_gpus: dict[tuple[int, str], set[int]] = {}
    for task in all_tasks:
        key = (int(task["pool_id"]), str(task["joint_config_fingerprint"]))
        config_gpus.setdefault(key, set()).add(int(task["assigned_gpu"]))
    if any(len(gpus) != 1 for gpus in config_gpus.values()):
        raise RuntimeError("a configuration was split across GPUs")
    if any(int(task["seed"]) not in TRAINING_SEEDS for task in tasks):
        raise RuntimeError("task uses a non-frozen seed")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULTS_FILENAME
    completed = load_jsonl(result_path)
    completed_keys = {result_key(row) for row in completed}
    expected_keys = {result_key(task) for task in tasks}
    foreign = completed_keys - expected_keys
    if foreign:
        raise RuntimeError(f"result shard contains foreign task keys: {sorted(foreign)[:3]}")
    pending = [task for task in tasks if result_key(task) not in completed_keys]
    run_plan = {
        "shard": args.shard,
        "dry_run": bool(args.dry_run),
        "configuration_count": len({task["joint_config_fingerprint"] for task in tasks}),
        "seeds": sorted({int(task["seed"]) for task in tasks}),
        "expected_task_count": len(expected_keys),
        "already_completed": len(completed_keys),
        "pending_task_count": len(pending),
        "pre_label_manifest_root_hash": freeze_audit["recomputed_root_hash"],
        "models_loaded": False,
        "dataset_loaded": False,
        "training_started": False,
    }
    dump_json(output / "run_plan.json", run_plan)
    if args.dry_run:
        print(json.dumps(run_plan, indent=2, sort_keys=True))
        return 0

    device = torch.device(args.device)
    gpu = physical_gpu_identity(device)
    expected_uuid = str(preregistration["training"]["shard_to_gpu_uuid"][str(args.shard)])
    if gpu["physical_index"] != args.shard or gpu["uuid"] != expected_uuid:
        raise RuntimeError(
            f"shard/GPU mismatch: shard={args.shard}, actual={gpu}, expected_uuid={expected_uuid}"
        )
    request = resolve_dataset_request("cora", None, cora_root=args.cora_root)
    bundle = load_dataset_from_request(request)
    context = dataset_context_from_manifest(bundle.manifest)
    assert_dataset_context_matches(
        EXPECTED_DATASET_CONTEXT, context, context="Original Head attribution v2",
    )
    training = preregistration["training"]
    if (
        context["dataset_content_fingerprint"] != training["dataset_content_fingerprint"]
        or context["split_fingerprint"] != training["split_fingerprint"]
    ):
        raise RuntimeError("dataset or split fingerprint differs from frozen plan")
    data = bundle.data.to(device)
    started_at = utc_now()

    def write_progress() -> None:
        rows = load_jsonl(result_path)
        dump_json(output / "progress.json", {
            "shard": args.shard,
            "started_at": started_at,
            "updated_at": utc_now(),
            "expected": len(expected_keys),
            "completed": len(rows),
            "remaining": len(expected_keys) - len(rows),
            "success": sum(row["terminal_status"] == "success" for row in rows),
            "failed": sum(row["terminal_status"] == "failed" for row in rows),
            "invalid": sum(row["terminal_status"] == "invalid" for row in rows),
            "retry_count": sum(int(row["retry_count"]) for row in rows),
            "actual_gpu": gpu,
            "command": [sys.executable, *sys.argv],
        })

    write_progress()
    for task in pending:
        failures: list[dict[str, Any]] = []
        attempt = 0
        task_started = utc_now()
        monotonic = time.monotonic()
        while True:
            attempt += 1
            try:
                val_acc, valid, metadata = train_and_eval_arch(
                    config={
                        "operations": list(task["operations"]),
                        "edges": [tuple(edge) for edge in task["edges"]],
                        "effective_layers": int(task["effective_layers"]),
                    },
                    data=data,
                    in_ch=bundle.num_features,
                    out_ch=bundle.num_classes,
                    lr=float(task["lr"]),
                    dropout=float(task["dropout"]),
                    hidden_dim=int(task["hidden_dim"]),
                    weight_decay=float(task["l2"]),
                    gcnii_alpha=0.1,
                    gcnii_theta=0.5,
                    gat_heads=int(task["gat_heads"]),
                    sage_aggr=str(task["sage_aggr"]),
                    gin_eps=float(task["gin_eps"]),
                    gat_heads_by_layer=task["gat_heads_by_layer"],
                    sage_aggr_by_layer=task["sage_aggr_by_layer"],
                    gin_eps_by_layer=task["gin_eps_by_layer"],
                    device=device,
                    max_epochs=300,
                    patience=80,
                    seed=int(task["seed"]),
                    track_test=False,
                    return_metadata=True,
                    return_reproducibility_metadata=True,
                    return_trajectory=True,
                    component_variant="baseline",
                )
                sanitized_metadata = sanitize_training_metadata(metadata)
                row = {
                    **base_result(task, context, gpu),
                    "val_acc": float(val_acc),
                    "valid": bool(valid),
                    "terminal_status": "success" if valid else "invalid",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": failures,
                    "started_at": task_started,
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - monotonic,
                    **sanitized_metadata,
                }
                break
            except Exception as error:
                if isinstance(error, ProtocolFirewallViolation):
                    raise
                failures.append({
                    "attempt": attempt,
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(
                        traceback.format_exception(type(error), error, error.__traceback__)
                    ),
                })
                if attempt <= int(training["maximum_system_retries"]):
                    continue
                row = {
                    **base_result(task, context, gpu),
                    "val_acc": None,
                    "valid": False,
                    "terminal_status": "failed",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": failures,
                    "started_at": task_started,
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - monotonic,
                }
                break
        hits = strict_test_key_hits(row)
        if hits:
            raise RuntimeError(f"test-named field would be written: {hits}")
        append_jsonl(result_path, row)
        write_progress()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
