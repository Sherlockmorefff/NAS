#!/usr/bin/env python3
"""Run one frozen GPU shard of the H-vs-Head prospective GNN labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from analyse.h_vs_head_model_blind_prospective import load_json, verify_plan
from analyse.vae_label_expansion import EXPECTED_DATASET_CONTEXT
from dataset_utils import (
    assert_dataset_context_matches,
    dataset_context_from_manifest,
    load_dataset_from_request,
    resolve_dataset_request,
)
from eval_utils import train_and_eval_arch


RESULTS_FILENAME = "seed_results.jsonl"


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


def contains_test_named_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "test" in str(key).lower() or contains_test_named_field(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_test_named_field(child) for child in value)
    return False


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
            if contains_test_named_field(row):
                raise ValueError(f"test-named seed result field at {path}:{line_number}")
            key = (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
            if key in keys:
                raise ValueError(f"duplicate seed result at {path}:{line_number}: {key}")
            keys.add(key)
            rows.append(row)
    return rows


def physical_gpu_identity(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("prospective validation requires CUDA")
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
                "device": str(device), "physical_index": physical_index,
                "uuid": parts[1], "name": parts[2], "cuda_visible_devices": visible,
            }
    raise RuntimeError(f"could not resolve physical GPU {physical_index}")


def base_result(
    config: Mapping[str, Any], seed: int, context: Mapping[str, Any], gpu: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "config_id": str(config["config_id"]),
        "architecture_fingerprint": str(config["architecture_fingerprint"]),
        "hp_fingerprint": str(config["hp_fingerprint"]),
        "selection_stratum": str(config["selection_stratum"]),
        "structure_stratum": str(config["structure_stratum"]),
        "assigned_gpu": int(config["assigned_gpu"]), "seed": int(seed),
        "operations": list(config["operations"]), "edges": [list(edge) for edge in config["edges"]],
        "raw_hp_vector": [float(value) for value in config["raw_hp_vector"]],
        "lr": float(config["lr"]), "dropout": float(config["dropout"]),
        "hidden_dim": int(config["hidden_dim"]), "l2": float(config["l2"]),
        "gat_heads": int(config["gat_heads"]), "sage_aggr": str(config["sage_aggr"]),
        "gin_eps": float(config["gin_eps"]),
        "gat_heads_by_layer": config["gat_heads_by_layer"],
        "sage_aggr_by_layer": config["sage_aggr_by_layer"],
        "gin_eps_by_layer": config["gin_eps_by_layer"],
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = verify_plan(args.plan.resolve())
    selection = load_json(Path(plan["selection_path"]))
    configs = sorted(
        [dict(row) for row in selection if int(row["assigned_gpu"]) == args.shard],
        key=lambda row: (str(row["architecture_fingerprint"]), str(row["hp_fingerprint"])),
    )
    if len(configs) != 24:
        raise ValueError(f"shard {args.shard} must contain exactly 24 configurations")
    seeds = [int(seed) for seed in plan["training_seeds"]]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULTS_FILENAME
    completed = load_jsonl(result_path)
    completed_keys = {
        (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
        for row in completed
    }
    expected_keys = {
        (str(config["config_id"]), str(config["architecture_fingerprint"]), seed)
        for config in configs for seed in seeds
    }
    pending = [
        (config, seed) for config in configs for seed in seeds
        if (str(config["config_id"]), str(config["architecture_fingerprint"]), seed) not in completed_keys
    ]
    run_plan = {
        "shard": args.shard, "dry_run": bool(args.dry_run),
        "configuration_count": len(configs), "seeds": seeds,
        "expected_task_count": len(expected_keys),
        "already_completed": len(completed_keys & expected_keys),
        "pending_task_count": len(pending),
        "models_loaded": False, "dataset_loaded": False, "training_started": False,
    }
    dump_json(output / "run_plan.json", run_plan)
    if args.dry_run:
        print(json.dumps(run_plan, indent=2, sort_keys=True))
        return 0

    device = torch.device(args.device)
    gpu = physical_gpu_identity(device)
    expected_uuid = str(plan["shard_to_gpu_uuid"][str(args.shard)])
    if gpu["physical_index"] != args.shard or gpu["uuid"] != expected_uuid:
        raise RuntimeError(f"shard/GPU mismatch: shard={args.shard}, actual={gpu}, expected={expected_uuid}")
    request = resolve_dataset_request("cora", None, cora_root=args.cora_root)
    bundle = load_dataset_from_request(request)
    context = dataset_context_from_manifest(bundle.manifest)
    assert_dataset_context_matches(EXPECTED_DATASET_CONTEXT, context, context="H-vs-Head prospective validation")
    protocol = plan["labeling_protocol"]
    if (
        context["dataset_content_fingerprint"] != protocol["dataset_content_fingerprint"]
        or context["split_fingerprint"] != protocol["split_fingerprint"]
    ):
        raise RuntimeError("dataset or split fingerprint differs from frozen plan")
    data = bundle.data.to(device)
    started_at = utc_now()

    def write_progress() -> None:
        rows = load_jsonl(result_path)
        relevant = [
            row for row in rows
            if (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"])) in expected_keys
        ]
        dump_json(output / "progress.json", {
            "shard": args.shard, "started_at": started_at, "updated_at": utc_now(),
            "expected": len(expected_keys), "completed": len(relevant),
            "remaining": len(expected_keys) - len(relevant),
            "success": sum(row["terminal_status"] == "success" for row in relevant),
            "failed": sum(row["terminal_status"] == "failed" for row in relevant),
            "invalid": sum(row["terminal_status"] == "invalid" for row in relevant),
            "retry_count": sum(int(row["retry_count"]) for row in relevant),
            "actual_gpu": gpu, "command": [sys.executable, *sys.argv],
        })

    write_progress()
    for config, seed in pending:
        failures = []
        attempt = 0
        task_started = utc_now()
        monotonic = time.monotonic()
        while True:
            attempt += 1
            try:
                val_acc, valid, metadata = train_and_eval_arch(
                    config={
                        "operations": list(config["operations"]),
                        "edges": [tuple(edge) for edge in config["edges"]],
                        "effective_layers": int(config["effective_layers"]),
                    },
                    data=data, in_ch=bundle.num_features, out_ch=bundle.num_classes,
                    lr=float(config["lr"]), dropout=float(config["dropout"]),
                    hidden_dim=int(config["hidden_dim"]), weight_decay=float(config["l2"]),
                    gcnii_alpha=0.1, gcnii_theta=0.5,
                    gat_heads=int(config["gat_heads"]), sage_aggr=str(config["sage_aggr"]),
                    gin_eps=float(config["gin_eps"]),
                    gat_heads_by_layer=config["gat_heads_by_layer"],
                    sage_aggr_by_layer=config["sage_aggr_by_layer"],
                    gin_eps_by_layer=config["gin_eps_by_layer"],
                    device=device, max_epochs=300, patience=80, seed=int(seed),
                    track_test=False, return_metadata=True,
                    return_reproducibility_metadata=True, component_variant="baseline",
                )
                row = {
                    **base_result(config, seed, context, gpu),
                    "val_acc": float(val_acc), "valid": bool(valid),
                    "terminal_status": "success" if valid else "invalid",
                    "attempt_count": attempt, "retry_count": attempt - 1,
                    "attempt_failures": failures, "started_at": task_started,
                    "finished_at": utc_now(), "elapsed_seconds": time.monotonic() - monotonic,
                    **metadata,
                }
                break
            except Exception as error:
                failures.append({
                    "attempt": attempt, "type": type(error).__name__, "message": str(error),
                    "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                })
                if attempt <= int(plan["maximum_retries"]):
                    continue
                row = {
                    **base_result(config, seed, context, gpu),
                    "val_acc": None, "valid": False, "terminal_status": "failed",
                    "attempt_count": attempt, "retry_count": attempt - 1,
                    "attempt_failures": failures, "started_at": task_started,
                    "finished_at": utc_now(), "elapsed_seconds": time.monotonic() - monotonic,
                }
                break
        if contains_test_named_field(row):
            raise RuntimeError("test-named field would be written to prospective result")
        append_jsonl(result_path, row)
        write_progress()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
