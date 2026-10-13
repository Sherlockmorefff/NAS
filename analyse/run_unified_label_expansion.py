#!/usr/bin/env python3
"""Run frozen 300/80 Cora validation-label tasks with per-seed resume."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse.vae_label_expansion import (
    EXPECTED_CONTEXT,
    EXPECTED_DATASET_CONTEXT,
    PROTOCOL,
    sha256_file,
)
from dataset_utils import (
    assert_dataset_context_matches,
    dataset_context_from_manifest,
    load_dataset_from_request,
    resolve_dataset_request,
)
from eval_utils import train_and_eval_arch


RESULTS_JSONL = "seed_results.jsonl"
PROGRESS_JSON = "progress.json"
RUN_PLAN_JSON = "run_plan.json"


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _append_jsonl_atomic(path: Path, row: Mapping[str, Any]) -> None:
    payload = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["source"]), str(row["architecture_fingerprint"]),
        str(row["hp_fingerprint"]), int(row["seed"]),
    )


def _load_completed(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, int]]]:
    rows, keys = [], set()
    if not path.is_file():
        return rows, keys
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("terminal_status") not in ("success", "invalid", "failed"):
                raise ValueError(f"invalid terminal status at {path}:{line_number}")
            key = _key(row)
            if key in keys:
                raise ValueError(f"duplicate completed key at {path}:{line_number}")
            keys.add(key)
            rows.append(row)
    return rows, keys


def _context_from_bundle(bundle: Any) -> dict[str, Any]:
    context = dataset_context_from_manifest(bundle.manifest)
    return {
        "dataset": context["canonical_name"],
        "dataset_content_fingerprint": context["dataset_content_fingerprint"],
        "split_fingerprint": context["split_fingerprint"],
        "graph_transform": context["graph_transform"],
        "training_mode": context["training_mode"],
        "metric": context["metric"],
        "hp_mode": "global4",
        "normalization": "none",
    }


def _base_row(
    config: Mapping[str, Any],
    seed: int,
    context: Mapping[str, Any],
    runner_sha256: str,
    actual_gpu: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source": config["source"],
        "config_id": config["config_id"],
        "architecture_fingerprint": config["architecture_fingerprint"],
        "hp_fingerprint": config["hp_fingerprint"],
        "seed": int(seed),
        "operations": list(config["operations"]),
        "edges": [list(edge) for edge in config["edges"]],
        "z_search": list(config["z_search"]),
        "z_arch": list(config["z_arch"]),
        "raw_hp_vector": list(config["raw_hp_vector"]),
        "lr": float(config["lr"]),
        "dropout": float(config["dropout"]),
        "hidden_dim": int(config["hidden_dim"]),
        "l2": float(config["l2"]),
        "gat_heads": int(config.get("gat_heads", 1)),
        "sage_aggr": str(config.get("sage_aggr", "mean")),
        "gin_eps": float(config.get("gin_eps", 0.0)),
        "gat_heads_by_layer": config.get("gat_heads_by_layer"),
        "sage_aggr_by_layer": config.get("sage_aggr_by_layer"),
        "gin_eps_by_layer": config.get("gin_eps_by_layer"),
        **context,
        "eval_epochs": int(PROTOCOL["eval_epochs"]),
        "patience": int(PROTOCOL["patience"]),
        "seed_semantics": "common_confirmation_seed_manifest_v1",
        "training_entry": "eval_utils.train_and_eval_arch",
        "runner_sha256": runner_sha256,
        "actual_gpu": dict(actual_gpu),
    }


def _actual_gpu_identity(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "physical_index": None, "uuid": None, "name": "cpu"}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = int(visible.split(",")[0]) if visible else int(device.index or 0)
    result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    matched = None
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) == 3 and int(parts[0]) == physical:
                matched = parts
                break
    return {
        "device": str(device),
        "physical_index": physical,
        "uuid": None if matched is None else matched[1],
        "name": torch.cuda.get_device_name(device) if matched is None else matched[2],
        "cuda_visible_devices": visible,
        "nvidia_smi_error": result.stderr.strip() or None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--config_ids_file", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cora_root", default="Cora")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_retries", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = sorted(set(map(int, args.seeds)))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must be unique non-negative integers")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    configs = [row for row in selection if int(row["assigned_gpu"]) == args.shard]
    if args.config_ids_file is not None:
        ids = json.loads(args.config_ids_file.read_text(encoding="utf-8"))
        if not isinstance(ids, list) or len(ids) != len(set(map(str, ids))):
            raise ValueError("config IDs file must be a unique JSON list")
        requested = set(map(str, ids))
        configs = [row for row in configs if str(row["config_id"]) in requested]
        missing = requested - {str(row["config_id"]) for row in configs}
        if missing:
            raise ValueError(f"config IDs are not assigned to shard {args.shard}: {sorted(missing)}")
    configs.sort(key=lambda row: str(row["architecture_fingerprint"]))
    if len({row["architecture_fingerprint"] for row in configs}) != len(configs):
        raise ValueError("runner selection contains repeated architectures")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    completed_rows, completed_keys = _load_completed(output / RESULTS_JSONL)
    expected_keys = {
        (str(config["source"]), str(config["architecture_fingerprint"]), str(config["hp_fingerprint"]), seed)
        for config in configs for seed in seeds
    }
    if not completed_keys <= expected_keys:
        raise ValueError("resume output contains keys outside the current frozen shard plan")
    pending = [
        (config, seed) for config in configs for seed in seeds
        if (str(config["source"]), str(config["architecture_fingerprint"]), str(config["hp_fingerprint"]), seed) not in completed_keys
    ]
    plan = {
        "dry_run": bool(args.dry_run),
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256_file(args.selection),
        "shard": args.shard,
        "config_ids_file": None if args.config_ids_file is None else str(args.config_ids_file.resolve()),
        "config_ids_file_sha256": None if args.config_ids_file is None else sha256_file(args.config_ids_file),
        "configuration_count": len(configs),
        "config_ids": [row["config_id"] for row in configs],
        "seeds": seeds,
        "expected_training_count": len(expected_keys),
        "already_completed": len(completed_keys),
        "pending_training_count": len(pending),
        "protocol": {**PROTOCOL, **EXPECTED_CONTEXT},
        "max_retries": args.max_retries,
        "merge_key": ["source", "architecture_fingerprint", "hp_fingerprint", "seed"],
    }
    _dump_json(output / RUN_PLAN_JSON, plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    request = resolve_dataset_request("cora", None, cora_root=args.cora_root)
    bundle = load_dataset_from_request(request)
    context = _context_from_bundle(bundle)
    assert_dataset_context_matches(
        EXPECTED_DATASET_CONTEXT,
        dataset_context_from_manifest(bundle.manifest),
        context="unified label expansion",
    )
    data = bundle.data.to(device)
    runner_hash = sha256_file(Path(__file__).resolve())
    actual_gpu = _actual_gpu_identity(device)
    started_at = datetime.now(timezone.utc).isoformat()

    def write_progress() -> None:
        counts = {
            status: sum(row.get("terminal_status") == status for row in completed_rows)
            for status in ("success", "invalid", "failed")
        }
        _dump_json(
            output / PROGRESS_JSON,
            {
                "pid": os.getpid(),
                "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "expected_training_count": len(expected_keys),
                "completed": len(completed_rows),
                "remaining": len(expected_keys) - len(completed_rows),
                **counts,
                "retries": sum(int(row.get("retry_count", 0)) for row in completed_rows),
                "command": [sys.executable, *sys.argv],
                "runner_sha256": runner_hash,
                "actual_gpu": actual_gpu,
            },
        )

    write_progress()
    for config, seed in pending:
        attempt_failures = []
        attempt = 0
        row_started = datetime.now(timezone.utc).isoformat()
        monotonic_started = time.monotonic()
        while True:
            attempt += 1
            try:
                result = train_and_eval_arch(
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
                    gat_heads=int(config.get("gat_heads", 1)),
                    sage_aggr=str(config.get("sage_aggr", "mean")),
                    gin_eps=float(config.get("gin_eps", 0.0)),
                    gat_heads_by_layer=config.get("gat_heads_by_layer"),
                    sage_aggr_by_layer=config.get("sage_aggr_by_layer"),
                    gin_eps_by_layer=config.get("gin_eps_by_layer"),
                    device=device,
                    max_epochs=300,
                    patience=80,
                    seed=seed,
                    track_test=True,
                    return_metadata=True,
                    return_reproducibility_metadata=True,
                    component_variant="baseline",
                )
                val_acc, valid, test_acc, metadata = result
                row = {
                    **_base_row(config, seed, context, runner_hash, actual_gpu),
                    "val_acc": float(val_acc),
                    "test_acc": float(test_acc),
                    "valid": bool(valid),
                    "terminal_status": "success" if valid else "invalid",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": attempt_failures,
                    "started_at": row_started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.monotonic() - monotonic_started,
                    "actual_device": str(device),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    **metadata,
                }
                break
            except Exception as exc:
                attempt_failures.append(
                    {
                        "attempt": attempt,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    }
                )
                if attempt <= args.max_retries:
                    continue
                row = {
                    **_base_row(config, seed, context, runner_hash, actual_gpu),
                    "val_acc": None,
                    "test_acc": None,
                    "valid": False,
                    "terminal_status": "failed",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": attempt_failures,
                    "failure": attempt_failures[-1],
                    "started_at": row_started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.monotonic() - monotonic_started,
                    "actual_device": str(device),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
                break
        _append_jsonl_atomic(output / RESULTS_JSONL, row)
        completed_rows.append(row)
        if len(completed_rows) % 10 == 0 or len(completed_rows) == len(expected_keys):
            write_progress()
            print(
                f"PROGRESS shard={args.shard} completed={len(completed_rows)} "
                f"remaining={len(expected_keys)-len(completed_rows)} "
                f"failed={sum(r.get('terminal_status') == 'failed' for r in completed_rows)}",
                flush=True,
            )
    write_progress()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
