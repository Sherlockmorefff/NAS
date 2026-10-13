#!/usr/bin/env python3
"""Matched relabel diagnostic for historical and greedy architectures.

Dry-run performs checkpoint-only architecture selection.  It never loads a
dataset and never invokes the GNN training evaluator.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import (
    ARCH_NZ,
    architecture_fingerprint,
    cycle_distances,
    encode_architecture_mu,
    load_joint_vae,
    normalize_architecture_config,
)
from dataset_utils import load_dataset_from_request, resolve_dataset_request
from eval_utils import _decode_arch, decode_hp_by_mode, eval_z_search
from hp_modes import hp_dim_from_mode, validate_hp_mode


RESULTS_JSONL = "matched_relabel_results.jsonl"
RESULTS_JSON = "matched_relabel_results.json"
RESULTS_CSV = "matched_relabel_results.csv"
SUMMARY_JSON = "matched_relabel_summary.json"
SUMMARY_CSV = "matched_relabel_summary.csv"
PAIRED_CSV = "matched_relabel_paired_differences.csv"
SUMMARY_MD = "matched_relabel_summary.md"
SELECTION_JSON = "matched_relabel_selection.json"
SELECTION_CSV = "matched_relabel_selection.csv"
RUN_PLAN_JSON = "matched_relabel_run_plan.json"
PROGRESS_JSON = "matched_relabel_progress.json"


def _json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _load_history_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        for key in ("history", "records", "evaluations"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of Phase 4 records")
    records = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} record {index} is not an object")
        record = dict(raw)
        record["_history_path"] = str(path.resolve())
        record["_history_index"] = index
        records.append(record)
    return records


def load_history_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_history_file(path))
    for time_index, record in enumerate(records):
        record["_time_index"] = time_index
    return records


def load_fixed_selection(path: Path, hp_mode: str) -> list[dict[str, Any]]:
    """Load a frozen diagnostic selection without invoking either decoder."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("fixed selection must be a non-empty JSON list")
    expected_length = ARCH_NZ + hp_dim_from_mode(hp_mode)
    required = (
        "point_id",
        "z_search",
        "z_arch",
        "raw_hp_vector",
        "historical_config",
        "greedy_config",
        "historical_fingerprint",
        "greedy_fingerprint",
        "historical_mu_arch",
        "greedy_mu_arch",
        "cycle_distance_l2_historical",
        "cycle_distance_normalized_historical",
        "cycle_distance_l2_greedy",
        "cycle_distance_normalized_greedy",
        "historical_val_acc",
        "time_index",
        "history_path",
        "history_index",
    )
    selection: list[dict[str, Any]] = []
    point_ids: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"fixed selection row {index} is not an object")
        missing = [field for field in required if field not in raw]
        if missing:
            raise ValueError(
                f"fixed selection row {index} is missing: {', '.join(missing)}"
            )
        row = dict(raw)
        point_id = str(row["point_id"])
        if point_id in point_ids:
            raise ValueError(f"fixed selection contains duplicate point_id {point_id}")
        point_ids.add(point_id)
        if len(row["z_search"]) != expected_length:
            raise ValueError(
                f"fixed selection point {point_id} has z_search length "
                f"{len(row['z_search'])}, expected {expected_length}"
            )
        if len(row["z_arch"]) != ARCH_NZ:
            raise ValueError(f"fixed selection point {point_id} has invalid z_arch length")
        if len(row["raw_hp_vector"]) != hp_dim_from_mode(hp_mode):
            raise ValueError(f"fixed selection point {point_id} has invalid HP length")
        for version in ("historical", "greedy"):
            config = normalize_architecture_config(row[f"{version}_config"])
            fingerprint = architecture_fingerprint(config)
            if fingerprint != row[f"{version}_fingerprint"]:
                raise ValueError(
                    f"fixed selection point {point_id} {version} fingerprint mismatch"
                )
            row[f"{version}_config"] = config
        selection.append(row)
    return selection


def _base_search_vector(record: Mapping[str, Any], hp_mode: str) -> list[float]:
    expected = ARCH_NZ + hp_dim_from_mode(hp_mode)
    source = record.get("z_base", record.get("z_search"))
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise ValueError("history record is missing z_base/z_search")
    values = [float(value) for value in source]
    if len(values) < expected:
        raise ValueError(
            f"history vector has length {len(values)}, expected at least {expected} for {hp_mode}"
        )
    return values[:expected]


def _candidate_fingerprint(vector: Sequence[float]) -> str:
    array = np.ascontiguousarray(np.asarray(vector, dtype=np.float32))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def recover_historical_config(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the evaluated architecture without invoking any decoder."""

    if "operations" not in record or "edges" not in record:
        raise ValueError(
            "historical relabel requires operations and edges in every history record"
        )
    return normalize_architecture_config(
        {"operations": record["operations"], "edges": record["edges"]}
    )


def _spread_select(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["cycle_distance_normalized_greedy"]),
            str(row["point_id"]),
        ),
    )
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    return [ordered[int(index)] for index in indices]


def build_selection(
    records: Sequence[Mapping[str, Any]],
    model: Any,
    hp_mode: str,
    sample_count: int,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        base = _base_search_vector(record, hp_mode)
        point_id = str(record.get("candidate_fingerprint") or _candidate_fingerprint(base))
        if point_id in seen:
            continue
        seen.add(point_id)
        historical = recover_historical_config(record)
        z_arch = base[:ARCH_NZ]
        greedy = _decode_arch(
            model,
            torch.tensor(z_arch, dtype=torch.float32, device=device),
            device,
            decoder_policy="deterministic_greedy",
        )
        if greedy is None:
            continue
        historical_fp = architecture_fingerprint(historical)
        greedy_fp = architecture_fingerprint(greedy)
        if historical_fp == greedy_fp:
            continue
        historical_mu = encode_architecture_mu(model, historical)
        greedy_mu = encode_architecture_mu(model, greedy)
        historical_cycle = cycle_distances(z_arch, historical_mu)
        greedy_cycle = cycle_distances(z_arch, greedy_mu)
        accuracy = float(record.get("val_acc", record.get("value", 0.0)))
        candidates.append(
            {
                "point_id": point_id,
                "time_index": int(record.get("_time_index", record.get("step", 0))),
                "history_path": str(record.get("_history_path", "")),
                "history_index": int(record.get("_history_index", -1)),
                "historical_val_acc": accuracy,
                "z_arch": z_arch,
                "raw_hp_vector": base[ARCH_NZ:],
                "z_search": base,
                "historical_config": historical,
                "historical_fingerprint": historical_fp,
                "historical_mu_arch": historical_mu,
                "cycle_distance_l2_historical": historical_cycle[0],
                "cycle_distance_normalized_historical": historical_cycle[1],
                "greedy_config": greedy,
                "greedy_fingerprint": greedy_fp,
                "greedy_mu_arch": greedy_mu,
                "cycle_distance_l2_greedy": greedy_cycle[0],
                "cycle_distance_normalized_greedy": greedy_cycle[1],
            }
        )
    if not candidates:
        raise ValueError("no history points have different historical and greedy fingerprints")

    accuracies = np.asarray([row["historical_val_acc"] for row in candidates])
    top_threshold = float(np.quantile(accuracies, 0.8))
    middle_low = float(np.quantile(accuracies, 0.35))
    middle_high = float(np.quantile(accuracies, 0.65))
    top = [row for row in candidates if row["historical_val_acc"] >= top_threshold]
    middle = [
        row
        for row in candidates
        if middle_low <= row["historical_val_acc"] <= middle_high
    ]
    top_count = min(len(top), math.ceil(sample_count / 2))
    selected = _spread_select(top, top_count)
    selected_ids = {row["point_id"] for row in selected}
    middle_pool = [row for row in middle if row["point_id"] not in selected_ids]
    selected.extend(_spread_select(middle_pool, sample_count - len(selected)))
    selected_ids = {row["point_id"] for row in selected}
    if len(selected) < sample_count:
        remainder = [row for row in candidates if row["point_id"] not in selected_ids]
        selected.extend(_spread_select(remainder, sample_count - len(selected)))
    for row in selected:
        if row["historical_val_acc"] >= top_threshold:
            row["selection_region"] = "top20"
        elif middle_low <= row["historical_val_acc"] <= middle_high:
            row["selection_region"] = "middle"
        else:
            row["selection_region"] = "coverage_fill"
        row["selection_thresholds"] = {
            "top20_min": top_threshold,
            "middle_min": middle_low,
            "middle_max": middle_high,
        }
    return sorted(selected, key=lambda row: (row["time_index"], row["point_id"]))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _load_completed(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, int]]]:
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if not path.exists():
        return rows, completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            terminal_status = row.get("terminal_status")
            if terminal_status not in ("success", "invalid", "failed"):
                raise ValueError(
                    f"completed relabel row at line {line_number} has invalid "
                    f"terminal_status={terminal_status!r}"
                )
            key = (
                str(row["point_id"]),
                str(row["architecture_version"]),
                int(row["seed"]),
            )
            if key in completed:
                raise ValueError(f"duplicate completed relabel key at line {line_number}: {key}")
            completed.add(key)
            rows.append(row)
    return rows, completed


def pending_evaluations(
    selection: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    completed: set[tuple[str, str, int]],
) -> list[tuple[Mapping[str, Any], str, int]]:
    pending = []
    for point in selection:
        for architecture_version in ("historical", "greedy"):
            for seed in seeds:
                key = (str(point["point_id"]), architecture_version, int(seed))
                if key not in completed:
                    pending.append((point, architecture_version, int(seed)))
    return pending


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_version: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_version.setdefault(
            (str(row["point_id"]), str(row["architecture_version"])), []
        ).append(row)
    summaries = []
    for (point_id, version), group in sorted(by_version.items()):
        valid_values = [
            float(row["val_acc"])
            for row in group
            if row.get("terminal_status") == "success"
        ]
        summaries.append(
            {
                "point_id": point_id,
                "architecture_version": version,
                "architecture_fingerprint": group[0]["architecture_fingerprint"],
                "seed_count": len(group),
                "valid_seed_count": len(valid_values),
                "mean_val_acc": statistics.fmean(valid_values) if valid_values else None,
                "std_val_acc": statistics.stdev(valid_values) if len(valid_values) > 1 else 0.0 if valid_values else None,
            }
        )
    lookup = {
        (str(row["point_id"]), str(row["architecture_version"]), int(row["seed"])): row
        for row in rows
    }
    paired = []
    for point_id, _version, seed in sorted(lookup):
        if _version != "historical":
            continue
        historical = lookup[(point_id, "historical", seed)]
        greedy = lookup.get((point_id, "greedy", seed))
        if greedy is None:
            continue
        both_valid = bool(
            historical.get("terminal_status") == "success"
            and greedy.get("terminal_status") == "success"
        )
        paired.append(
            {
                "point_id": point_id,
                "seed": seed,
                "historical_val_acc": historical.get("val_acc"),
                "greedy_val_acc": greedy.get("val_acc"),
                "greedy_minus_historical": (
                    float(greedy["val_acc"]) - float(historical["val_acc"])
                    if both_valid
                    else None
                ),
                "both_valid": both_valid,
            }
        )
    paired_by_point = []
    for point_id in sorted({row["point_id"] for row in paired}):
        values = [
            float(row["greedy_minus_historical"])
            for row in paired
            if row["point_id"] == point_id and row["both_valid"]
        ]
        paired_by_point.append(
            {
                "point_id": point_id,
                "paired_seed_count": len(values),
                "mean_greedy_minus_historical": (
                    statistics.fmean(values) if values else None
                ),
                "std_greedy_minus_historical": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0 if values else None
                ),
            }
        )
    return {
        "terminal_counts": _terminal_counts(rows),
        "by_point_and_version": summaries,
        "paired_differences": paired,
        "paired_by_point": paired_by_point,
    }


def _write_summary(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    summary = summarize_results(rows)
    _json_dump(output / RESULTS_JSON, list(rows))
    _write_csv(output / RESULTS_CSV, rows)
    _json_dump(output / SUMMARY_JSON, summary)
    _write_csv(output / SUMMARY_CSV, summary["by_point_and_version"])
    _write_csv(output / PAIRED_CSV, summary["paired_differences"])
    lines = [
        "# Matched relabel summary",
        "",
        f"Completed evaluations: {len(rows)}",
        "",
        "| Point | Version | Seeds | Valid | Mean val acc | Std |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["by_point_and_version"]:
        mean = "NA" if row["mean_val_acc"] is None else f"{row['mean_val_acc']:.6f}"
        std = "NA" if row["std_val_acc"] is None else f"{row['std_val_acc']:.6f}"
        lines.append(
            f"| {row['point_id'][:12]} | {row['architecture_version']} | "
            f"{row['seed_count']} | {row['valid_seed_count']} | {mean} | {std} |"
        )
    (output / SUMMARY_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _terminal_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"success": 0, "invalid": 0, "failed": 0}
    for row in rows:
        status = row.get("terminal_status")
        if status in counts:
            counts[str(status)] += 1
    counts["completed"] = sum(counts.values())
    counts["retries"] = sum(int(row.get("retry_count", 0)) for row in rows)
    return counts


def _is_clearly_temporary_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "temporarily unavailable",
            "resource temporarily unavailable",
            "connection reset",
            "connection aborted",
        )
    )


def _failure_row(
    point: Mapping[str, Any],
    architecture_version: str,
    decoder_policy: str,
    seed: int,
    hp_mode: str,
    args: argparse.Namespace,
    exc: BaseException,
    *,
    attempt_count: int,
    attempt_failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    config = point[f"{architecture_version}_config"]
    realized_hp = None
    condition_mask_vector = None
    hp_error = None
    try:
        realized_hp = decode_hp_by_mode(
            z_search=torch.tensor(point["z_search"], dtype=torch.float32),
            arch_nz=ARCH_NZ,
            config=config,
            hp_mode=hp_mode,
            log_lr_min=args.log_lr_min,
            log_lr_max=args.log_lr_max,
            dropout_min=args.dropout_min,
            dropout_max=args.dropout_max,
        )
        condition_mask_vector = realized_hp["condition_mask_vector"]
    except Exception as hp_exc:
        hp_error = {"type": type(hp_exc).__name__, "message": str(hp_exc)}
    return {
        "point_id": point["point_id"],
        "time_index": point["time_index"],
        "architecture_version": architecture_version,
        "decoder_policy": decoder_policy,
        "architecture_fingerprint": point[f"{architecture_version}_fingerprint"],
        "z_arch": point["z_arch"],
        "mu_arch": point[f"{architecture_version}_mu_arch"],
        "cycle_distance_l2": point[f"cycle_distance_l2_{architecture_version}"],
        "cycle_distance_normalized": point[
            f"cycle_distance_normalized_{architecture_version}"
        ],
        "seed": int(seed),
        "operations": list(config["operations"]),
        "edges": [list(edge) for edge in config["edges"]],
        "raw_hp_vector": point["raw_hp_vector"],
        "realized_hp": realized_hp,
        "condition_mask_vector": condition_mask_vector,
        "val_acc": None,
        "valid": False,
        "terminal_status": "failed",
        "attempt_count": int(attempt_count),
        "retry_count": int(attempt_count - 1),
        "historical_val_acc": point["historical_val_acc"],
        "history_path": point["history_path"],
        "history_index": point["history_index"],
        "failure": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            "hp_decode_failure": hp_error,
        },
        "attempt_failures": list(attempt_failures),
    }


def _write_progress(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    expected_total: int,
    *,
    started_at: str,
) -> dict[str, Any]:
    counts = _terminal_counts(rows)
    progress = {
        **counts,
        "expected_total": int(expected_total),
        "remaining": int(expected_total - counts["completed"]),
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
    }
    _json_dump(output / PROGRESS_JSON, progress)
    return progress


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched historical/greedy relabel diagnostic")
    parser.add_argument("--history", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--selection_file",
        type=Path,
        default=None,
        help="frozen selection JSON; bypasses history sampling and all architecture decoding",
    )
    parser.add_argument("--hp_mode", default="global4")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sample_count", type=int, default=18)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--cora_root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log_lr_min", type=float, default=-4.0)
    parser.add_argument("--log_lr_max", type=float, default=-1.5)
    parser.add_argument("--dropout_min", type=float, default=0.1)
    parser.add_argument("--dropout_max", type=float, default=0.6)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--max_retries",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="maximum retries after the initial attempt (hard-capped at two)",
    )
    parser.add_argument("--retry_all_failures", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    args = parse_args(argv)
    hp_mode = validate_hp_mode(args.hp_mode)
    seeds = sorted(set(map(int, args.seeds)))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain non-negative integers")
    if args.max_retries < 0:
        raise ValueError("--max_retries must be non-negative")
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() and not args.dry_run else "cpu")
    )
    model = load_joint_vae(args.checkpoint, hp_mode, device=device)
    selection = (
        load_fixed_selection(args.selection_file, hp_mode)
        if args.selection_file is not None
        else build_selection(
            load_history_records(args.history),
            model,
            hp_mode,
            args.sample_count,
            device=device,
        )
    )
    if len(selection) != int(args.sample_count):
        raise ValueError(
            f"selection contains {len(selection)} points, expected --sample_count={args.sample_count}"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _json_dump(output / SELECTION_JSON, selection)
    _write_csv(output / SELECTION_CSV, selection)
    target_files = [
        SELECTION_JSON,
        SELECTION_CSV,
        RUN_PLAN_JSON,
        RESULTS_JSONL,
        RESULTS_JSON,
        RESULTS_CSV,
        SUMMARY_JSON,
        SUMMARY_CSV,
        PAIRED_CSV,
        SUMMARY_MD,
        PROGRESS_JSON,
    ]
    completed_rows, completed = _load_completed(output / RESULTS_JSONL)
    selected_by_id = {str(point["point_id"]): point for point in selection}
    for row in completed_rows:
        point_id = str(row["point_id"])
        if point_id not in selected_by_id:
            raise ValueError(
                f"resume output contains point {point_id!r} outside the current selection"
            )
        version = str(row["architecture_version"])
        expected_fingerprint = selected_by_id[point_id].get(
            f"{version}_fingerprint"
        )
        if (
            expected_fingerprint is not None
            and row.get("architecture_fingerprint") != expected_fingerprint
        ):
            raise ValueError(
                f"resume architecture fingerprint mismatch for {(point_id, version)}"
            )
    pending = pending_evaluations(selection, seeds, completed)
    run_keys = [
        f"{point['point_id']}:{version}:{seed}"
        for point in selection
        for version in ("historical", "greedy")
        for seed in seeds
    ]
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("dry-run plan contains duplicate point/version/seed keys")
    plan = {
        "dry_run": bool(args.dry_run),
        "selection_source": (
            None if args.selection_file is None else str(args.selection_file.resolve())
        ),
        "selection_sha256": (
            None
            if args.selection_file is None
            else hashlib.sha256(args.selection_file.read_bytes()).hexdigest()
        ),
        "hp_mode": hp_mode,
        "selected_points": len(selection),
        "architecture_versions": ["historical", "greedy"],
        "seeds": seeds,
        "expected_total_trainings": len(selection) * 2 * len(seeds),
        "unique_run_key_count": len(set(run_keys)),
        "run_keys": run_keys,
        "already_completed": len(completed),
        "pending_trainings": len(pending),
        "target_files": [str(output / name) for name in target_files],
    }
    _json_dump(output / RUN_PLAN_JSON, plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    request = resolve_dataset_request(
        args.dataset,
        args.data_root,
        cora_root=args.cora_root,
    )
    bundle = load_dataset_from_request(request)
    data = bundle.data.to(device)
    expected_total = len(selection) * 2 * len(seeds)
    _write_progress(output, completed_rows, expected_total, started_at=started_at)
    for point, architecture_version, seed in pending:
        config = point[f"{architecture_version}_config"]
        decoder_policy = (
            "historical_modal3"
            if architecture_version == "historical"
            else "deterministic_greedy"
        )
        attempt_count = 0
        attempt_failures: list[dict[str, Any]] = []
        while True:
            attempt_count += 1
            try:
                result = eval_z_search(
                    model,
                    torch.tensor(point["z_search"], dtype=torch.float32, device=device),
                    data,
                    bundle.num_features,
                    bundle.num_classes,
                    arch_nz=ARCH_NZ,
                    log_lr_min=args.log_lr_min,
                    log_lr_max=args.log_lr_max,
                    dropout_min=args.dropout_min,
                    dropout_max=args.dropout_max,
                    device=device,
                    hp_mode=hp_mode,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    candidate_eval_seed=seed,
                    decoder_policy=decoder_policy,
                    architecture_config=config,
                    return_hp=True,
                    return_diagnostics=True,
                )
                row = {
                    **result,
                    "point_id": point["point_id"],
                    "time_index": point["time_index"],
                    "architecture_version": architecture_version,
                    "historical_val_acc": point["historical_val_acc"],
                    "history_path": point["history_path"],
                    "history_index": point["history_index"],
                    "seed": seed,
                    "terminal_status": (
                        "success" if bool(result.get("valid", False)) else "invalid"
                    ),
                    "attempt_count": attempt_count,
                    "retry_count": attempt_count - 1,
                    "attempt_failures": attempt_failures,
                }
                break
            except Exception as exc:
                attempt_failures.append(
                    {
                        "attempt": attempt_count,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        ),
                    }
                )
                should_retry = attempt_count <= int(args.max_retries) and (
                    bool(args.retry_all_failures)
                    or _is_clearly_temporary_failure(exc)
                )
                if should_retry:
                    continue
                row = _failure_row(
                    point,
                    architecture_version,
                    decoder_policy,
                    seed,
                    hp_mode,
                    args,
                    exc,
                    attempt_count=attempt_count,
                    attempt_failures=attempt_failures,
                )
                break
        _append_jsonl(output / RESULTS_JSONL, row)
        completed_rows.append(row)
        if len(completed_rows) % 12 == 0 or len(completed_rows) == expected_total:
            progress = _write_progress(
                output, completed_rows, expected_total, started_at=started_at
            )
            print(
                "PROGRESS "
                f"completed={progress['completed']} success={progress['success']} "
                f"invalid={progress['invalid']} failed={progress['failed']} "
                f"remaining={progress['remaining']}",
                flush=True,
            )
    _write_summary(output, completed_rows)
    final_progress = _write_progress(
        output, completed_rows, expected_total, started_at=started_at
    )
    print(json.dumps({**plan, "pending_trainings": 0}, indent=2, sort_keys=True))
    return 0 if final_progress["completed"] == expected_total else 2


if __name__ == "__main__":
    raise SystemExit(main())
