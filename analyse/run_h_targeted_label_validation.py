#!/usr/bin/env python3
"""Run frozen targeted Cora validation labels without reading test accuracy."""

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

from analyse.h_targeted_reliable_pair_validation import sha256_file
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


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    payload = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_frozen_plan(plan_path: Path) -> dict[str, Any]:
    sidecar = plan_path.with_suffix(".sha256")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != sha256_file(plan_path) or fields[1] != plan_path.name:
        raise RuntimeError("targeted task-plan sidecar mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for section in ("code_hashes", "input_hashes", "protected_hashes"):
        for raw_path, expected in plan[section].items():
            path = Path(raw_path)
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"frozen {section} mismatch: {path}")
    selection = Path(plan["selection_path"])
    if sha256_file(selection) != plan["selection_sha256"]:
        raise RuntimeError("frozen targeted selection changed")
    return plan


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
            if key in keys:
                raise ValueError(f"duplicate seed result at line {line_number}: {key}")
            if any("test" in str(field).lower() for field in row):
                raise ValueError(f"test-named result field at line {line_number}")
            keys.add(key)
            rows.append(row)
    return rows


def _physical_gpu_identity(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("formal targeted labeling requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_index = int(visible.split(",")[0]) if visible else int(device.index or 0)
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    matched = None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3 and int(parts[0]) == physical_index:
            matched = parts
            break
    if matched is None:
        raise RuntimeError(f"could not resolve physical GPU {physical_index}")
    return {
        "device": str(device),
        "physical_index": physical_index,
        "uuid": matched[1],
        "name": matched[2],
        "cuda_visible_devices": visible,
    }


def _base_result(
    config: Mapping[str, Any], seed: int, context: Mapping[str, Any], gpu: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "config_id": str(config["config_id"]),
        "architecture_fingerprint": str(config["architecture_fingerprint"]),
        "hp_fingerprint": str(config["hp_fingerprint"]),
        "selection_stratum": str(config["selection_stratum"]),
        "assigned_gpu": int(config["assigned_gpu"]),
        "seed": int(seed),
        "operations": list(config["operations"]),
        "edges": [list(edge) for edge in config["edges"]],
        "raw_hp_vector": list(map(float, config["raw_hp_vector"])),
        "lr": float(config["lr"]),
        "dropout": float(config["dropout"]),
        "hidden_dim": int(config["hidden_dim"]),
        "l2": float(config["l2"]),
        "gat_heads": int(config["gat_heads"]),
        "sage_aggr": str(config["sage_aggr"]),
        "gin_eps": float(config["gin_eps"]),
        "gat_heads_by_layer": config["gat_heads_by_layer"],
        "sage_aggr_by_layer": config["sage_aggr_by_layer"],
        "gin_eps_by_layer": config["gin_eps_by_layer"],
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


def active_configs_and_seeds(
    plan: Mapping[str, Any], selection: Sequence[Mapping[str, Any]], shard: int, stage: str,
    adaptive_path: Path | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    configs = [dict(row) for row in selection if int(row["assigned_gpu"]) == shard]
    if len(configs) != 10:
        raise ValueError(f"shard {shard} does not contain exactly 10 configurations")
    if stage == "initial":
        seeds = list(map(int, plan["initial_seeds"]))
    elif stage == "adaptive":
        if adaptive_path is None or not adaptive_path.is_file():
            raise ValueError("adaptive stage requires --adaptive_selection")
        adaptive = json.loads(adaptive_path.read_text(encoding="utf-8"))
        requested = set(map(str, adaptive["config_ids"]))
        configs = [row for row in configs if str(row["config_id"]) in requested]
        seeds = list(map(int, plan["adaptive_seeds"]))
    else:
        raise ValueError(stage)
    return sorted(configs, key=lambda row: row["architecture_fingerprint"]), seeds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--stage", choices=("initial", "adaptive"), required=True)
    parser.add_argument("--adaptive_selection", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cora_root", default="Cora")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_retries", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = verify_frozen_plan(args.plan.resolve())
    selection = json.loads(Path(plan["selection_path"]).read_text(encoding="utf-8"))
    configs, seeds = active_configs_and_seeds(
        plan, selection, args.shard, args.stage, args.adaptive_selection
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULTS_FILENAME
    completed = _load_jsonl(result_path)
    completed_keys = {
        (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
        for row in completed
    }
    expected_keys = {
        (str(config["config_id"]), str(config["architecture_fingerprint"]), seed)
        for config in configs
        for seed in seeds
    }
    relevant_completed = completed_keys & expected_keys
    pending = [
        (config, seed)
        for config in configs
        for seed in seeds
        if (str(config["config_id"]), str(config["architecture_fingerprint"]), seed)
        not in completed_keys
    ]
    run_plan = {
        "stage": args.stage,
        "shard": args.shard,
        "dry_run": bool(args.dry_run),
        "task_plan": str(args.plan.resolve()),
        "task_plan_sha256": sha256_file(args.plan),
        "configuration_count": len(configs),
        "config_ids": [row["config_id"] for row in configs],
        "seeds": seeds,
        "expected_task_count": len(expected_keys),
        "already_completed_for_stage": len(relevant_completed),
        "pending_task_count": len(pending),
        "models_loaded": False,
        "dataset_loaded": False,
        "training_started": False,
    }
    _dump_json(output / f"run_plan_{args.stage}.json", run_plan)
    if args.dry_run:
        print(json.dumps(run_plan, indent=2, sort_keys=True))
        return 0

    device = torch.device(args.device)
    gpu = _physical_gpu_identity(device)
    expected_uuid = plan["shard_to_gpu_uuid"][str(args.shard)]
    if gpu["physical_index"] != args.shard or gpu["uuid"] != expected_uuid:
        raise RuntimeError(
            f"shard/GPU mismatch: shard={args.shard}, actual={gpu}, expected_uuid={expected_uuid}"
        )
    request = resolve_dataset_request("cora", None, cora_root=args.cora_root)
    bundle = load_dataset_from_request(request)
    context = dataset_context_from_manifest(bundle.manifest)
    assert_dataset_context_matches(
        EXPECTED_DATASET_CONTEXT, context, context="targeted H validation"
    )
    data = bundle.data.to(device)
    started_at = utc_now()

    def progress() -> None:
        all_rows = _load_jsonl(result_path)
        relevant = [
            row
            for row in all_rows
            if (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
            in expected_keys
        ]
        _dump_json(
            output / f"progress_{args.stage}.json",
            {
                "stage": args.stage,
                "shard": args.shard,
                "started_at": started_at,
                "updated_at": utc_now(),
                "expected": len(expected_keys),
                "completed": len(relevant),
                "remaining": len(expected_keys) - len(relevant),
                "success": sum(row["terminal_status"] == "success" for row in relevant),
                "invalid": sum(row["terminal_status"] == "invalid" for row in relevant),
                "failed": sum(row["terminal_status"] == "failed" for row in relevant),
                "retry_count": sum(int(row["retry_count"]) for row in relevant),
                "actual_gpu": gpu,
                "command": [sys.executable, *sys.argv],
            },
        )

    progress()
    for index, (config, seed) in enumerate(pending, 1):
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
                        "effective_layers": sum(op != "Identity" for op in config["operations"]),
                    },
                    data=data,
                    in_ch=bundle.num_features,
                    out_ch=bundle.num_classes,
                    lr=float(config["lr"]),
                    dropout=float(config["dropout"]),
                    hidden_dim=int(config["hidden_dim"]),
                    weight_decay=float(config["l2"]),
                    gcnii_alpha=0.1,
                    gcnii_theta=0.5,
                    gat_heads=int(config["gat_heads"]),
                    sage_aggr=str(config["sage_aggr"]),
                    gin_eps=float(config["gin_eps"]),
                    gat_heads_by_layer=config["gat_heads_by_layer"],
                    sage_aggr_by_layer=config["sage_aggr_by_layer"],
                    gin_eps_by_layer=config["gin_eps_by_layer"],
                    device=device,
                    max_epochs=300,
                    patience=80,
                    seed=seed,
                    track_test=False,
                    return_metadata=True,
                    return_reproducibility_metadata=True,
                    component_variant="baseline",
                )
                row = {
                    **_base_result(config, seed, context, gpu),
                    "val_acc": float(val_acc),
                    "valid": bool(valid),
                    "terminal_status": "success" if valid else "invalid",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": failures,
                    "started_at": task_started,
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - monotonic,
                    **metadata,
                }
                break
            except Exception as exc:
                failures.append(
                    {
                        "attempt": attempt,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        ),
                    }
                )
                if attempt <= args.max_retries:
                    continue
                row = {
                    **_base_result(config, seed, context, gpu),
                    "val_acc": None,
                    "valid": False,
                    "terminal_status": "failed",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": failures,
                    "failure": failures[-1],
                    "started_at": task_started,
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - monotonic,
                }
                break
        if any("test" in str(key).lower() for key in row):
            raise AssertionError("targeted seed result contains a test-named field")
        _append_jsonl(result_path, row)
        if index % 5 == 0 or index == len(pending):
            progress()
            print(
                f"PROGRESS stage={args.stage} shard={args.shard} "
                f"completed_this_run={index}/{len(pending)}",
                flush=True,
            )
    progress()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
