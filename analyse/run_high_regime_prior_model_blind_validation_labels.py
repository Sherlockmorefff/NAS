#!/usr/bin/env python3
"""Run one immutable GPU shard for the high-regime validation experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import high_regime_prior_model_blind_validation as plan
from analyse import run_original_head_confirmatory_attribution_v2_labels as v2_runner
from analyse.vae_label_expansion import EXPECTED_DATASET_CONTEXT
from dataset_utils import (
    assert_dataset_context_matches,
    dataset_context_from_manifest,
    load_dataset_from_request,
    resolve_dataset_request,
)
from eval_utils import train_and_eval_arch


ProtocolFirewallViolation = v2_runner.ProtocolFirewallViolation
utc_now = v2_runner.utc_now
dump_json = v2_runner.dump_json
append_jsonl = v2_runner.append_jsonl
sanitize_training_metadata = v2_runner.sanitize_training_metadata
physical_gpu_identity = v2_runner.physical_gpu_identity
strict_test_key_hits = plan.strict_test_key_hits
load_json = plan.load_json


def result_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("task_stage", "initial")), int(row["pool_id"]),
        str(row["joint_config_fingerprint"]), str(row["architecture_fingerprint"]),
        str(row["hp_fingerprint"]), int(row["seed"]),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    keys = set()
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


def verify_stabilization_freeze(root: Path) -> dict[str, Any]:
    path = root / "stabilization/stabilization_immutable_manifest.json"
    if not path.is_file():
        return {"passed": False, "reason": "missing stabilization immutable manifest"}
    manifest = load_json(path)
    mismatches = []
    for row in manifest["files"]:
        target = root / row["relative_path"]
        actual = plan.sha256_file(target) if target.is_file() else None
        if actual != row["sha256"]:
            mismatches.append({"relative_path": row["relative_path"], "expected": row["sha256"], "actual": actual})
    root_hash = plan.stable_hash([(row["relative_path"], row["sha256"]) for row in manifest["files"]])
    return {
        "passed": not mismatches and root_hash == manifest["manifest_root_hash"],
        "mismatches": mismatches, "stored_root_hash": manifest["manifest_root_hash"],
        "recomputed_root_hash": root_hash,
    }


def base_result(task: Mapping[str, Any], context: Mapping[str, Any], gpu: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_stage": str(task["task_stage"]), "pool_id": int(task["pool_id"]),
        "batch_id": int(task["batch_id"]), "architecture_slot": int(task["architecture_slot"]),
        "config_id": str(task["config_id"]),
        "architecture_fingerprint": str(task["architecture_fingerprint"]),
        "hp_fingerprint": str(task["hp_fingerprint"]),
        "joint_config_fingerprint": str(task["joint_config_fingerprint"]),
        "hp_regime": str(task["hp_regime"]), "seed": int(task["seed"]),
        "task_key": str(task["task_key"]), "assigned_gpu": int(task["assigned_gpu"]),
        "selected_by": list(task["selected_by"]),
        "operations": list(task["operations"]),
        "edges": [list(map(int, edge)) for edge in task["edges"]],
        "effective_layers": int(task["effective_layers"]),
        "raw_hp_vector": [float(value) for value in task["raw_hp_vector"]],
        "lr": float(task["lr"]), "dropout": float(task["dropout"]),
        "hidden_dim": int(task["hidden_dim"]), "l2": float(task["l2"]),
        "gat_heads": int(task["gat_heads"]), "sage_aggr": str(task["sage_aggr"]),
        "gin_eps": float(task["gin_eps"]),
        "gat_heads_by_layer": task["gat_heads_by_layer"],
        "sage_aggr_by_layer": task["sage_aggr_by_layer"],
        "gin_eps_by_layer": task["gin_eps_by_layer"],
        "dataset": context["canonical_name"],
        "dataset_content_fingerprint": context["dataset_content_fingerprint"],
        "split_fingerprint": context["split_fingerprint"],
        "graph_transform": context["graph_transform"], "training_mode": context["training_mode"],
        "metric": context["metric"], "hp_mode": "global4", "normalization": "none",
        "eval_epochs": 300, "patience": 80,
        "training_entry": "eval_utils.train_and_eval_arch", "actual_gpu": dict(gpu),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("initial", "stabilization"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    initial_freeze = plan.verify_freeze(root)
    if not initial_freeze["passed"]:
        raise RuntimeError(f"initial immutable manifest failed: {initial_freeze}")
    firewall = load_json(root / "pre_label_firewall_audit.json")
    if not firewall.get("passed"):
        raise RuntimeError("pre-label firewall is not passed")
    if args.stage == "initial":
        task_path = root / "selected_union_tasks.json"
        allowed_seeds = set(plan.INITIAL_SEEDS)
        stage_freeze = None
    else:
        stage_freeze = verify_stabilization_freeze(root)
        if not stage_freeze["passed"]:
            raise RuntimeError(f"stabilization freeze failed: {stage_freeze}")
        task_path = root / "stabilization/stabilization_tasks.json"
        allowed_seeds = set(plan.STABILIZATION_SEEDS)
    all_tasks = load_json(task_path)
    if strict_test_key_hits(all_tasks):
        raise RuntimeError("frozen task plan contains a test-named field")
    if any(str(task["task_stage"]) != args.stage for task in all_tasks):
        raise RuntimeError("task stage differs from runner stage")
    if any(int(task["seed"]) not in allowed_seeds for task in all_tasks):
        raise RuntimeError("task plan contains a non-frozen seed")
    tasks = sorted([dict(row) for row in all_tasks if int(row["assigned_gpu"]) == args.shard], key=result_key)
    if not tasks:
        raise ValueError(f"empty {args.stage} shard {args.shard}")
    config_gpus: dict[tuple[int, str], set[int]] = defaultdict(set)
    for task in all_tasks:
        config_gpus[(int(task["pool_id"]), str(task["joint_config_fingerprint"]))].add(int(task["assigned_gpu"]))
    if any(len(gpus) != 1 for gpus in config_gpus.values()):
        raise RuntimeError("one complete configuration is split across GPUs")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "seed_results.jsonl"
    completed = load_jsonl(result_path)
    completed_keys = {result_key(row) for row in completed}
    expected_keys = {result_key(task) for task in tasks}
    if completed_keys - expected_keys:
        raise RuntimeError("result shard contains foreign task keys")
    pending = [task for task in tasks if result_key(task) not in completed_keys]
    run_plan = {
        "stage": args.stage, "shard": args.shard, "dry_run": bool(args.dry_run),
        "configuration_count": len({task["joint_config_fingerprint"] for task in tasks}),
        "seeds": sorted({int(task["seed"]) for task in tasks}),
        "expected_task_count": len(expected_keys), "already_completed": len(completed_keys),
        "pending_task_count": len(pending),
        "pre_label_manifest_root_hash": initial_freeze["recomputed_root_hash"],
        "stabilization_manifest_root_hash": stage_freeze["recomputed_root_hash"] if stage_freeze else None,
        "dataset_loaded": False, "training_started": False,
    }
    dump_json(output / "run_plan.json", run_plan)
    if args.dry_run:
        print(json.dumps(run_plan, indent=2, sort_keys=True))
        return 0

    prereg = load_json(root / "preregistration.json")
    device = torch.device(args.device)
    gpu = physical_gpu_identity(device)
    expected_uuid = str(prereg["training"]["shard_to_gpu_uuid"][str(args.shard)])
    if gpu["physical_index"] != args.shard or gpu["uuid"] != expected_uuid:
        raise RuntimeError(f"shard/GPU mismatch: shard={args.shard}, actual={gpu}, expected={expected_uuid}")
    request = resolve_dataset_request("cora", None, cora_root=args.cora_root)
    bundle = load_dataset_from_request(request)
    context = dataset_context_from_manifest(bundle.manifest)
    assert_dataset_context_matches(EXPECTED_DATASET_CONTEXT, context, context="high-regime prior validation")
    if (
        context["dataset_content_fingerprint"] != prereg["training"]["dataset_content_fingerprint"]
        or context["split_fingerprint"] != prereg["training"]["split_fingerprint"]
    ):
        raise RuntimeError("dataset or split fingerprint differs from frozen plan")
    data = bundle.data.to(device)
    started_at = utc_now()

    def write_progress() -> None:
        rows = load_jsonl(result_path)
        statuses = Counter(str(row["terminal_status"]) for row in rows)
        dump_json(output / "progress.json", {
            "stage": args.stage, "shard": args.shard, "started_at": started_at,
            "updated_at": utc_now(), "expected": len(expected_keys), "completed": len(rows),
            "remaining": len(expected_keys) - len(rows), "terminal_status_counts": dict(statuses),
            "retry_count": sum(int(row["retry_count"]) for row in rows),
            "actual_gpu": gpu, "command": [sys.executable, *sys.argv],
        })

    write_progress()
    for task in pending:
        failures = []
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
                    data=data, in_ch=bundle.num_features, out_ch=bundle.num_classes,
                    lr=float(task["lr"]), dropout=float(task["dropout"]),
                    hidden_dim=int(task["hidden_dim"]), weight_decay=float(task["l2"]),
                    gcnii_alpha=0.1, gcnii_theta=0.5, gat_heads=int(task["gat_heads"]),
                    sage_aggr=str(task["sage_aggr"]), gin_eps=float(task["gin_eps"]),
                    gat_heads_by_layer=task["gat_heads_by_layer"],
                    sage_aggr_by_layer=task["sage_aggr_by_layer"],
                    gin_eps_by_layer=task["gin_eps_by_layer"],
                    device=device, max_epochs=300, patience=80, seed=int(task["seed"]),
                    track_test=False, return_metadata=True,
                    return_reproducibility_metadata=True, return_trajectory=True,
                    component_variant="baseline",
                )
                cleaned = sanitize_training_metadata(metadata)
                row = {
                    **base_result(task, context, gpu), "val_acc": float(val_acc),
                    "valid": bool(valid), "terminal_status": "success" if valid else "invalid",
                    "attempt_count": attempt, "retry_count": attempt - 1,
                    "attempt_failures": failures, "started_at": task_started,
                    "finished_at": utc_now(), "elapsed_seconds": time.monotonic() - monotonic,
                    **cleaned,
                }
                break
            except Exception as error:
                if isinstance(error, ProtocolFirewallViolation):
                    raise
                failures.append({
                    "attempt": attempt, "type": type(error).__name__, "message": str(error),
                    "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                })
                if attempt <= int(prereg["training"]["maximum_system_retries"]):
                    continue
                row = {
                    **base_result(task, context, gpu), "val_acc": None, "valid": False,
                    "terminal_status": "failed", "attempt_count": attempt,
                    "retry_count": attempt - 1, "attempt_failures": failures,
                    "started_at": task_started, "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - monotonic,
                }
                break
        if strict_test_key_hits(row):
            raise RuntimeError("test-named field would be written")
        append_jsonl(result_path, row)
        write_progress()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
