#!/usr/bin/env python3
"""Plan and analyze the pre-registered Stage 3 multi-seed calibration run.

This module never trains a model.  ``plan`` freezes inputs and task keys;
``analyze`` performs an exact merge, validates seed-invariant fields, applies
the noise gate, and writes the diagnostic-only evidence bundle.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, t, trim_mean

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import architecture_fingerprint


POLICIES = ("historical_modal3", "deterministic_greedy")
VERSIONS = {"historical_modal3": "historical", "deterministic_greedy": "greedy"}
ALL_SEEDS = tuple(range(15))
NEW_SEEDS = tuple(range(3, 15))
GPU0_SEEDS = (3, 5, 7, 9, 11, 13)
GPU1_SEEDS = (4, 6, 8, 10, 12, 14)
ANOMALY_PREFIXES = ("84ea3e9ccc66", "2993434c986a")
INVARIANT_FIELDS = (
    "architecture_fingerprint",
    "operations",
    "edges",
    "raw_hp_vector",
    "realized_hp",
    "condition_mask_vector",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(dict(value))
    return rows


def _dump_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() or None


def _task_key(point_id: str, policy: str, seed: int) -> tuple[str, str, int]:
    return str(point_id), str(policy), int(seed)


def validate_selection(selection: Any, expected_points: int = 18) -> list[dict[str, Any]]:
    if not isinstance(selection, list) or len(selection) != expected_points:
        raise ValueError(
            f"selection must contain exactly {expected_points} points, got "
            f"{len(selection) if isinstance(selection, list) else 'non-list'}"
        )
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(selection):
        if not isinstance(raw, Mapping):
            raise ValueError(f"selection row {index} is not an object")
        row = dict(raw)
        point_id = str(row.get("point_id", ""))
        if not point_id or point_id in seen:
            raise ValueError(f"invalid or duplicate point_id at selection row {index}")
        seen.add(point_id)
        if row.get("selection_region") not in ("top20", "middle"):
            raise ValueError(f"point {point_id} has unexpected selection_region")
        for version in ("historical", "greedy"):
            config = row.get(f"{version}_config")
            if not isinstance(config, Mapping):
                raise ValueError(f"point {point_id} is missing {version}_config")
            if architecture_fingerprint(config) != row.get(f"{version}_fingerprint"):
                raise ValueError(f"point {point_id} has mismatched {version} fingerprint")
        if row["historical_fingerprint"] == row["greedy_fingerprint"]:
            raise ValueError(f"point {point_id} does not differ across policies")
        output.append(row)
    regions = {name: sum(row["selection_region"] == name for row in output) for name in ("top20", "middle")}
    if regions != {"top20": 9, "middle": 9}:
        raise ValueError(f"frozen selection region counts changed: {regions}")
    top_historical_fingerprints = {
        row["historical_fingerprint"]
        for row in output if row["selection_region"] == "top20"
    }
    if len(top_historical_fingerprints) != 8:
        raise ValueError(
            "frozen top20 selection must contain exactly 8 unique historical "
            f"fingerprints, got {len(top_historical_fingerprints)}"
        )
    return output


def create_plan(
    selection_path: Path,
    stage2_results: Path,
    history_path: Path,
    checkpoint_path: Path,
    output: Path,
    code_paths: Sequence[Path],
) -> dict[str, Any]:
    selection = validate_selection(_load_json(selection_path))
    output.mkdir(parents=True, exist_ok=True)
    frozen_selection = output / "matched_relabel_selection.json"
    if frozen_selection.exists() and sha256_file(frozen_selection) != sha256_file(selection_path):
        raise ValueError("output already contains a different frozen selection")
    if not frozen_selection.exists():
        shutil.copyfile(selection_path, frozen_selection)
    task_rows = []
    for shard, seeds in (("gpu0_odd", GPU0_SEEDS), ("gpu1_even", GPU1_SEEDS)):
        for point in selection:
            for policy in POLICIES:
                for seed in seeds:
                    task_rows.append(
                        {
                            "shard": shard,
                            "point_id": point["point_id"],
                            "decoder_policy": policy,
                            "architecture_version": VERSIONS[policy],
                            "seed": seed,
                        }
                    )
    keys = [_task_key(row["point_id"], row["decoder_policy"], row["seed"]) for row in task_rows]
    if len(task_rows) != 432 or len(set(keys)) != 432:
        raise AssertionError("Stage 3 task plan must have 432 unique new evaluations")
    inputs = [selection_path, stage2_results, history_path, checkpoint_path, *code_paths]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage 3 inputs: {missing}")
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "hp_mode": "global4",
        "fixed_point_count": len(selection),
        "policies": list(POLICIES),
        "existing_seeds": [0, 1, 2],
        "new_seeds": list(NEW_SEEDS),
        "gpu_seed_assignment": {"gpu0_odd": list(GPU0_SEEDS), "gpu1_even": list(GPU1_SEEDS)},
        "new_training_count": len(task_rows),
        "merged_expected_count": len(selection) * len(POLICIES) * len(ALL_SEEDS),
        "max_retries_per_evaluation": 2,
        "selection_regions": {name: sum(row["selection_region"] == name for row in selection) for name in ("top20", "middle")},
        "merge_key": ["point_id", "decoder_policy", "seed"],
        "seed_invariant_fields": list(INVARIANT_FIELDS),
        "noise_gate": {
            "policy": "historical_modal3",
            "population": "the frozen selection's 9 top20 point-HP configurations",
            "median_t_ci95_half_width_max": 0.006,
            "point_fraction_with_half_width_le_0_010_min": 0.75,
            "require_no_missing_invalid_failed_seeds": True,
        },
        "expansion_gate": {
            "minimum_unique_historical_fingerprints": 60,
            "target_regions": {"top20": 24, "middle": 24, "lower": 12},
            "may_not_lower_minimum": True,
        },
        "input_sha256": {str(path.resolve()): sha256_file(path) for path in inputs},
        "git_commit": _git_commit(),
        "tasks": task_rows,
    }
    plan_path = output / "stage3_task_plan.json"
    _dump_json(plan_path, plan)
    (output / "stage3_task_plan.sha256").write_text(
        f"{sha256_file(plan_path)}  {plan_path.name}\n", encoding="utf-8"
    )
    return plan


def _normalize_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    policy = str(row.get("decoder_policy", ""))
    if policy not in POLICIES:
        raise ValueError(f"unexpected decoder_policy {policy!r}")
    if str(row.get("architecture_version")) != VERSIONS[policy]:
        raise ValueError(f"policy/version mismatch for point {row.get('point_id')}")
    row["seed"] = int(row["seed"])
    return row


def exact_merge(
    selection: Sequence[Mapping[str, Any]], sources: Sequence[Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {
        _task_key(point["point_id"], policy, seed)
        for point in selection for policy in POLICIES for seed in ALL_SEEDS
    }
    rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    duplicates = []
    for source in sources:
        rows = _load_jsonl(source)
        source_counts[str(source.resolve())] = len(rows)
        for raw in rows:
            row = _normalize_row(raw)
            key = _task_key(row["point_id"], row["decoder_policy"], row["seed"])
            if key in rows_by_key:
                duplicates.append(key)
            else:
                rows_by_key[key] = row
    unexpected = sorted(set(rows_by_key) - expected)
    missing = sorted(expected - set(rows_by_key))
    if duplicates or unexpected or missing:
        raise ValueError(
            f"exact merge failed: duplicates={len(duplicates)}, "
            f"unexpected={len(unexpected)}, missing={len(missing)}"
        )
    selection_by_id = {str(row["point_id"]): row for row in selection}
    invariant_checks = 0
    for point_id, point in selection_by_id.items():
        for policy in POLICIES:
            group = [rows_by_key[_task_key(point_id, policy, seed)] for seed in ALL_SEEDS]
            for field in INVARIANT_FIELDS:
                values = {_canonical(row.get(field)) for row in group}
                if len(values) != 1:
                    raise ValueError(f"seed invariant {field} changed for {(point_id, policy)}")
                invariant_checks += 1
            version = VERSIONS[policy]
            if group[0]["architecture_fingerprint"] != point[f"{version}_fingerprint"]:
                raise ValueError(f"selection fingerprint changed for {(point_id, policy)}")
            config = {"operations": group[0]["operations"], "edges": group[0]["edges"]}
            if architecture_fingerprint(config) != group[0]["architecture_fingerprint"]:
                raise ValueError(f"recorded config fingerprint mismatch for {(point_id, policy)}")
    ordered = [rows_by_key[key] for key in sorted(rows_by_key)]
    audit = {
        "merge_key": ["point_id", "decoder_policy", "seed"],
        "source_counts": source_counts,
        "expected_count": len(expected),
        "merged_count": len(ordered),
        "duplicate_count": 0,
        "unexpected_count": 0,
        "missing_count": 0,
        "seed_invariant_group_field_checks": invariant_checks,
        "selection_fingerprint_checks": len(selection) * len(POLICIES),
        "seeds": list(ALL_SEEDS),
        "passed": True,
    }
    return ordered, audit


def _successful_values(group: Sequence[Mapping[str, Any]]) -> list[float]:
    return [
        float(row["val_acc"]) for row in group
        if row.get("terminal_status", "success") == "success"
        and bool(row.get("valid", False)) and row.get("val_acc") is not None
    ]


def _mean_ci(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, None
    half = float(t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values)))
    return mean - half, mean + half, half


def point_policy_statistics(
    rows: Sequence[Mapping[str, Any]], selection: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    region = {str(row["point_id"]): row["selection_region"] for row in selection}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["point_id"]), str(row["decoder_policy"])), []).append(row)
    output = []
    for (point_id, policy), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["seed"]))
        values = _successful_values(group)
        ci_low, ci_high, half = _mean_ci(values)
        median = float(np.median(values)) if values else None
        statuses = {name: sum(row.get("terminal_status", "success") == name for row in group) for name in ("success", "invalid", "failed")}
        output.append(
            {
                "point_id": point_id,
                "decoder_policy": policy,
                "architecture_version": VERSIONS[policy],
                "architecture_fingerprint": group[0]["architecture_fingerprint"],
                "time_index": min(int(row.get("time_index", 0)) for row in group),
                "selection_region": region[point_id],
                "expected_seed_count": len(ALL_SEEDS),
                "successful_seed_count": len(values),
                **{f"terminal_{name}": count for name, count in statuses.items()},
                "retry_count": sum(int(row.get("retry_count", 0)) for row in group),
                "mean": statistics.fmean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "median": median,
                "mad": float(np.median(np.abs(np.asarray(values) - median))) if values else None,
                "trimmed_mean_10pct": float(trim_mean(values, 0.1)) if values else None,
                "t_ci95_low": ci_low,
                "t_ci95_high": ci_high,
                "t_ci95_half_width": half,
                "historical_single_label": float(group[0]["historical_val_acc"]),
                "mean_minus_historical_single_label": (
                    statistics.fmean(values) - float(group[0]["historical_val_acc"])
                    if values else None
                ),
                "odd_seed_mean": statistics.fmean([float(row["val_acc"]) for row in group if int(row["seed"]) % 2 and row.get("terminal_status", "success") == "success" and bool(row.get("valid", False))]) if any(int(row["seed"]) % 2 and row.get("terminal_status", "success") == "success" and bool(row.get("valid", False)) for row in group) else None,
                "even_seed_mean": statistics.fmean([float(row["val_acc"]) for row in group if int(row["seed"]) % 2 == 0 and row.get("terminal_status", "success") == "success" and bool(row.get("valid", False))]) if any(int(row["seed"]) % 2 == 0 and row.get("terminal_status", "success") == "success" and bool(row.get("valid", False)) for row in group) else None,
            }
        )
    return output


def _bootstrap_kendall(
    odd: np.ndarray, even: np.ndarray, *, seed: int = 20260821, replicates: int = 2000
) -> dict[str, Any]:
    if odd.size < 2:
        return {
            "estimate": None,
            "bootstrap_ci95_low": None,
            "bootstrap_ci95_high": None,
            "bootstrap_valid_replicates": 0,
        }
    estimate = kendalltau(odd, even).statistic
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        indices = rng.integers(0, odd.size, odd.size)
        value = kendalltau(odd[indices], even[indices]).statistic
        if np.isfinite(value):
            samples.append(float(value))
    return {
        "estimate": float(estimate) if np.isfinite(estimate) else None,
        "bootstrap_ci95_low": float(np.quantile(samples, 0.025)) if samples else None,
        "bootstrap_ci95_high": float(np.quantile(samples, 0.975)) if samples else None,
        "bootstrap_valid_replicates": len(samples),
    }


def odd_even_ordering(stats: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for policy in POLICIES:
        for region in ("all", "top20"):
            policy_rows = [
                row for row in stats
                if row["decoder_policy"] == policy
                and (region == "all" or row["selection_region"] == region)
            ]
            by_fingerprint: dict[str, Mapping[str, Any]] = {}
            for row in sorted(
                policy_rows,
                key=lambda value: (int(value["time_index"]), str(value["point_id"])),
            ):
                by_fingerprint.setdefault(str(row["architecture_fingerprint"]), row)
            odd, even = [], []
            for row in by_fingerprint.values():
                if row["odd_seed_mean"] is None or row["even_seed_mean"] is None:
                    continue
                odd.append(float(row["odd_seed_mean"]))
                even.append(float(row["even_seed_mean"]))
            output.append(
                {
                    "decoder_policy": policy,
                    "selection_region": region,
                    "deduplication_key": "architecture_fingerprint",
                    "duplicate_resolution": "first by (time_index, point_id)",
                    "unique_architecture_count": len(by_fingerprint),
                    "complete_odd_even_architecture_count": len(odd),
                    **_bootstrap_kendall(np.asarray(odd), np.asarray(even)),
                }
            )
    return output


def region_statistics(stats: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for policy in POLICIES:
        for region in ("top20", "middle", "all"):
            subset = [
                row for row in stats
                if row["decoder_policy"] == policy
                and (region == "all" or row["selection_region"] == region)
            ]
            means = [float(row["mean"]) for row in subset if row["mean"] is not None]
            stds = [float(row["sample_std"]) for row in subset if row["sample_std"] is not None]
            halfs = [float(row["t_ci95_half_width"]) for row in subset if row["t_ci95_half_width"] is not None]
            output.append(
                {
                    "decoder_policy": policy,
                    "selection_region": region,
                    "point_hp_count": len(subset),
                    "unique_architecture_count": len({row["architecture_fingerprint"] for row in subset}),
                    "mean_of_15seed_means": statistics.fmean(means) if means else None,
                    "median_sample_std": float(np.median(stds)) if stds else None,
                    "median_t_ci95_half_width": float(np.median(halfs)) if halfs else None,
                    "invalid_count": sum(int(row["terminal_invalid"]) for row in subset),
                    "failed_count": sum(int(row["terminal_failed"]) for row in subset),
                    "retry_count": sum(int(row["retry_count"]) for row in subset),
                }
            )
    return output


def run_metadata(
    output: Path, sources: Sequence[Path], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torch-geometric", "numpy", "scipy", "scikit-learn", "python-igraph"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    progress = []
    for source in sources:
        progress_path = source.parent / "matched_relabel_progress.json"
        if progress_path.is_file():
            value = _load_json(progress_path)
            progress.append(
                {
                    "source": str(source.resolve()),
                    "command": value.get("command"),
                    "started_at": value.get("started_at"),
                    "updated_at": value.get("updated_at"),
                    "terminal_counts": {
                        key: value.get(key)
                        for key in ("completed", "success", "invalid", "failed", "retries")
                    },
                }
            )
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpu_query": gpu.stdout.strip().splitlines() if gpu.returncode == 0 else [],
        "gpu_query_error": gpu.stderr.strip() or None,
        "git_commit": _git_commit(),
        "code_sha256": {
            str(path.resolve()): sha256_file(path)
            for path in (
                REPO_ROOT / "architecture_diagnostics.py",
                REPO_ROOT / "eval_utils.py",
                REPO_ROOT / "analyse/run_matched_relabel.py",
                REPO_ROOT / "analyse/compare_matched_representations.py",
                REPO_ROOT / "analyse/stage3_noise_labels.py",
            )
        },
        "source_sha256": {str(path.resolve()): sha256_file(path) for path in sources},
        "runs": progress,
        "failures": [row for row in rows if row.get("terminal_status") == "failed"],
        "invalids": [row for row in rows if row.get("terminal_status") == "invalid"],
        "retried_evaluations": [row for row in rows if int(row.get("retry_count", 0)) > 0],
    }
    _dump_json(output / "stage3_run_metadata.json", metadata)
    return metadata


def paired_policy_statistics(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = {_task_key(row["point_id"], row["decoder_policy"], row["seed"]): row for row in rows}
    paired = []
    for point_id in sorted({str(row["point_id"]) for row in rows}):
        common_seeds = sorted(
            {
                int(row["seed"])
                for row in rows
                if str(row["point_id"]) == point_id
                and row["decoder_policy"] == POLICIES[0]
            }
            & {
                int(row["seed"])
                for row in rows
                if str(row["point_id"]) == point_id
                and row["decoder_policy"] == POLICIES[1]
            }
        )
        for seed in common_seeds:
            historical = lookup[_task_key(point_id, POLICIES[0], seed)]
            greedy = lookup[_task_key(point_id, POLICIES[1], seed)]
            valid = bool(_successful_values([historical]) and _successful_values([greedy]))
            paired.append(
                {
                    "point_id": point_id,
                    "seed": seed,
                    "both_successful": valid,
                    "historical_val_acc": historical.get("val_acc"),
                    "greedy_val_acc": greedy.get("val_acc"),
                    "greedy_minus_historical": float(greedy["val_acc"]) - float(historical["val_acc"]) if valid else None,
                }
            )
    point_rows = []
    for point_id in sorted({row["point_id"] for row in paired}):
        values = [
            float(row["greedy_minus_historical"])
            for row in paired
            if row["point_id"] == point_id and row["both_successful"]
        ]
        low, high, half = _mean_ci(values)
        point_rows.append(
            {
                "point_id": point_id,
                "paired_seed_count": len(values),
                "mean_greedy_minus_historical": (
                    statistics.fmean(values) if values else None
                ),
                "sample_std_across_seeds": (
                    statistics.stdev(values) if len(values) > 1 else None
                ),
                "t_ci95_low_across_seeds": low,
                "t_ci95_high_across_seeds": high,
                "t_ci95_half_width_across_seeds": half,
            }
        )
    point_values = [
        float(row["mean_greedy_minus_historical"])
        for row in point_rows
        if row["mean_greedy_minus_historical"] is not None
    ]
    low, high, half = _mean_ci(point_values)
    rng = np.random.default_rng(20260821)
    bootstrap = []
    if point_values:
        array = np.asarray(point_values, dtype=np.float64)
        bootstrap = rng.choice(
            array, size=(2000, len(array)), replace=True
        ).mean(axis=1).tolist()
    return paired, {
        "inference_unit": "point_mean_of_seed_paired_differences",
        "paired_point_count": len(point_values),
        "paired_seed_row_count": sum(row["paired_seed_count"] for row in point_rows),
        "mean_greedy_minus_historical": (
            statistics.fmean(point_values) if point_values else None
        ),
        "sample_std_across_points": (
            statistics.stdev(point_values) if len(point_values) > 1 else None
        ),
        "t_ci95_low": low,
        "t_ci95_high": high,
        "t_ci95_half_width": half,
        "point_cluster_bootstrap_ci95_low": (
            float(np.quantile(bootstrap, 0.025)) if bootstrap else None
        ),
        "point_cluster_bootstrap_ci95_high": (
            float(np.quantile(bootstrap, 0.975)) if bootstrap else None
        ),
        "per_point": point_rows,
    }


def classify_anomaly(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(map(float, values))
    if not ordered:
        return {"classification": "missing", "largest_gap": None}
    median = float(np.median(ordered))
    collapse = min(ordered) < 0.5 or min(ordered) <= median - 0.1
    gaps = np.diff(ordered)
    split = int(np.argmax(gaps)) + 1 if gaps.size else 0
    largest_gap = float(gaps[split - 1]) if split else 0.0
    bimodal = largest_gap >= 0.05 and split >= 2 and len(ordered) - split >= 2
    unstable = len(ordered) > 1 and statistics.stdev(ordered) > 0.03
    labels = [name for name, flag in (("collapse", collapse), ("bimodality", bimodal), ("instability", unstable)) if flag]
    return {
        "classification": "+".join(labels) if labels else "stable",
        "collapse": collapse,
        "bimodality": bimodal,
        "instability": unstable,
        "minimum": min(ordered),
        "maximum": max(ordered),
        "median": median,
        "sample_std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "largest_gap": largest_gap,
        "largest_gap_split": split,
        "values": ordered,
    }


def anomaly_sensitivity(
    rows: Sequence[Mapping[str, Any]], stats: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    matched: dict[str, str] = {}
    for prefix in ANOMALY_PREFIXES:
        ids = sorted({str(row["point_id"]) for row in rows if str(row["point_id"]).startswith(prefix)})
        if len(ids) != 1:
            raise ValueError(f"anomaly prefix {prefix} matched {len(ids)} points")
        matched[prefix] = ids[0]
    anomaly_ids = set(matched.values())
    details = []
    for prefix, point_id in matched.items():
        for policy in POLICIES:
            values = _successful_values([row for row in rows if row["point_id"] == point_id and row["decoder_policy"] == policy])
            details.append({"prefix": prefix, "point_id": point_id, "decoder_policy": policy, **classify_anomaly(values)})
    sensitivity = []
    for policy in POLICIES:
        for excluded in (False, True):
            subset = [row for row in stats if row["decoder_policy"] == policy and (not excluded or row["point_id"] not in anomaly_ids)]
            means = [float(row["mean"]) for row in subset if row["mean"] is not None]
            halfs = [float(row["t_ci95_half_width"]) for row in subset if row["t_ci95_half_width"] is not None]
            sensitivity.append(
                {
                    "decoder_policy": policy,
                    "anomalies_excluded": excluded,
                    "point_count": len(subset),
                    "mean_of_point_means": statistics.fmean(means) if means else None,
                    "median_t_ci95_half_width": float(np.median(halfs)) if halfs else None,
                }
            )
    historical_top = [
        row for row in stats
        if row["decoder_policy"] == POLICIES[0]
        and row["selection_region"] == "top20"
    ]
    gate_sensitivity = []
    for excluded in (False, True):
        subset = [
            row for row in historical_top
            if not excluded or row["point_id"] not in anomaly_ids
        ]
        gate_sensitivity.append(
            {
                "anomalies_excluded": excluded,
                **noise_gate(subset, expected_point_count=len(subset)),
            }
        )
    greedy_unstable = any(
        row["decoder_policy"] == "deterministic_greedy"
        and (row.get("collapse") or row.get("bimodality") or row.get("instability"))
        for row in details
    )
    return {
        "anomaly_points": details,
        "included_excluded_sensitivity": sensitivity,
        "historical_top20_noise_gate_sensitivity": gate_sensitivity,
        "greedy_policy_instability": greedy_unstable,
    }


def noise_gate(
    stats: Sequence[Mapping[str, Any]], *, expected_point_count: int = 9
) -> dict[str, Any]:
    top = [
        row for row in stats
        if row["decoder_policy"] == POLICIES[0]
        and row["selection_region"] == "top20"
    ]
    if len(top) != expected_point_count:
        raise ValueError(
            f"noise gate expected {expected_point_count} historical top20 "
            f"point-HP rows, got {len(top)}"
        )
    complete = [row for row in top if row["successful_seed_count"] == len(ALL_SEEDS) and row["terminal_invalid"] == 0 and row["terminal_failed"] == 0]
    halfs = [float(row["t_ci95_half_width"]) for row in complete if row["t_ci95_half_width"] is not None]
    median_half = float(np.median(halfs)) if len(halfs) == len(top) else None
    fraction = sum(value <= 0.010 for value in halfs) / len(top)
    no_failures = len(complete) == len(top)
    passed = bool(median_half is not None and median_half <= 0.006 and fraction >= 0.75 and no_failures)
    return {
        "passed": passed,
        "population": f"{len(top)} frozen top20 historical point-HP configurations",
        "point_count": len(top),
        "complete_point_count": len(complete),
        "median_t_ci95_half_width": median_half,
        "median_threshold": 0.006,
        "fraction_half_width_le_0_010": fraction,
        "fraction_threshold": 0.75,
        "no_missing_invalid_failed_seeds": no_failures,
    }


def independent_history_summary(history_path: Path) -> dict[str, Any]:
    payload = _load_json(history_path)
    if isinstance(payload, Mapping):
        payload = payload.get("history", payload.get("records"))
    if not isinstance(payload, list):
        raise ValueError("history must be a list")
    first_by_fingerprint: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping) or "operations" not in raw or "edges" not in raw:
            continue
        fingerprint = architecture_fingerprint({"operations": raw["operations"], "edges": raw["edges"]})
        if fingerprint not in first_by_fingerprint:
            first_by_fingerprint[fingerprint] = {
                "architecture_fingerprint": fingerprint,
                "history_index": index,
                "step": int(raw.get("step", index)),
                "val_acc": float(raw.get("val_acc", raw.get("value", 0.0))),
                "operations": raw["operations"],
                "edges": raw["edges"],
            }
    rows = sorted(first_by_fingerprint.values(), key=lambda row: (row["step"], row["history_index"], row["architecture_fingerprint"]))
    ranked = sorted(rows, key=lambda row: (-row["val_acc"], row["step"], row["architecture_fingerprint"]))
    n = len(ranked)
    middle_start, middle_end = math.floor(n * 0.35), math.ceil(n * 0.65)
    regions = {
        "lower": ranked[middle_end:],
        "middle": ranked[middle_start:middle_end],
        "upper": ranked[:middle_start],
    }
    if sum(len(values) for values in regions.values()) != n:
        raise AssertionError("exclusive historical regions do not cover every fingerprint")
    return {
        "history_record_count": len(payload),
        "independent_fingerprint_count": n,
        "deduplication": "first occurrence by (step, history_index); no graph-isomorphism merge",
        "region_counts": {name: len(values) for name, values in regions.items()},
        "region_definition": "exclusive rank bands: upper 35%, middle 30%, lower 35%",
        "minimum_required": 60,
        "minimum_met": n >= 60,
        "records": rows,
    }


def compute_vae_finetune_ready(
    *,
    noise_gate_passed: bool,
    independent_architecture_count: int,
    minimum_independent_architectures: int,
    group_leakage_count: int,
    test_label_used: bool,
    ordering_gate_passed: bool,
) -> bool:
    """Compute label-pool admission independently from representation class."""

    return bool(
        noise_gate_passed
        and independent_architecture_count >= minimum_independent_architectures
        and group_leakage_count == 0
        and not test_label_used
        and ordering_gate_passed
    )


def analyze(
    selection_path: Path,
    stage2_results: Path,
    shard_results: Sequence[Path],
    history_path: Path,
    output: Path,
    representation_result: Path | None = None,
    result_audit_status: str | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    selection = validate_selection(_load_json(selection_path))
    rows, merge_audit = exact_merge(selection, [stage2_results, *shard_results])
    _dump_jsonl(output / "stage3_merged_results.jsonl", rows)
    _dump_json(output / "stage3_merged_results.json", rows)
    _dump_csv(output / "stage3_merged_results.csv", rows)
    _dump_json(output / "stage3_merge_audit.json", merge_audit)

    stats = point_policy_statistics(rows, selection)
    _dump_json(output / "point_policy_statistics.json", stats)
    _dump_csv(output / "point_policy_statistics.csv", stats)
    regions = region_statistics(stats)
    _dump_json(output / "region_statistics.json", regions)
    _dump_csv(output / "region_statistics.csv", regions)
    paired, paired_summary = paired_policy_statistics(rows)
    _dump_csv(output / "paired_policy_differences.csv", paired)
    ordering = odd_even_ordering(stats)
    _dump_json(output / "odd_even_ordering.json", ordering)
    anomaly = anomaly_sensitivity(rows, stats)
    _dump_json(output / "anomaly_sensitivity.json", anomaly)
    history = independent_history_summary(history_path)
    _dump_json(output / "independent_history_pool.json", history)
    _dump_json(output / "reliable_labels.json", stats)
    _dump_csv(output / "reliable_labels.csv", stats)
    gate = noise_gate(stats)
    if not gate["passed"]:
        classification = "NOISE_PERSISTS"
        recommendation = "improve_label_reliability_before_representation_or_vae_changes"
    elif not history["minimum_met"]:
        classification = "INSUFFICIENT_INDEPENDENT_ARCHITECTURES"
        recommendation = "expand_independent_architecture_label_pool_without_vae_finetuning"
    else:
        classification = "LABELS_RELIABLE_REPRESENTATION_UNRESOLVED"
        recommendation = "run_preregistered_independent_architecture_representation_comparison"
    terminal_counts = {
        status: sum(row.get("terminal_status", "success") == status for row in rows)
        for status in ("success", "invalid", "failed")
    }
    top20_ordering = next(
        row for row in ordering
        if row["decoder_policy"] == POLICIES[0]
        and row["selection_region"] == "top20"
    )
    ordering_ready = bool(
        top20_ordering["estimate"] is not None
        and top20_ordering["estimate"] >= 0.30
        and top20_ordering["bootstrap_ci95_low"] is not None
        and top20_ordering["bootstrap_ci95_low"] > 0.0
    )
    representation_evidence = None
    if representation_result is not None:
        payload = _load_json(representation_result)
        if not isinstance(payload, Mapping) or "metrics" not in payload:
            raise ValueError("representation result is missing metrics")
        representation_evidence = {
            "path": str(representation_result.resolve()),
            "sha256": sha256_file(representation_result),
            "diagnostic_conclusion": payload.get("conclusion"),
            "metric_row_count": len(payload["metrics"]),
            "note": "18-point sensitivity only; it cannot override the independent-architecture stop gate",
        }
    vae_finetune_ready = compute_vae_finetune_ready(
        noise_gate_passed=bool(gate["passed"]),
        independent_architecture_count=int(history["independent_fingerprint_count"]),
        minimum_independent_architectures=int(history["minimum_required"]),
        group_leakage_count=0,
        test_label_used=False,
        ordering_gate_passed=ordering_ready,
    )
    conclusion = {
        "classification": classification,
        "diagnostic_only": True,
        "formal_bo_modified": False,
        "noise_gate_passed": bool(gate["passed"]),
        "greedy_policy_instability": bool(anomaly["greedy_policy_instability"]),
        "independent_fingerprint_count": int(history["independent_fingerprint_count"]),
        "vae_finetune_ready": vae_finetune_ready,
        "recommended_next_branch": recommendation,
        "result_audit_status": result_audit_status,
        "numeric_evidence": {
            "merged_evaluations": len(rows),
            "terminal_counts": terminal_counts,
            "total_retries": sum(int(row.get("retry_count", 0)) for row in rows),
            "historical_top20_noise_gate": gate,
            "paired_policy": paired_summary,
            "odd_even_ordering": ordering,
            "historical_top20_ordering_gate_passed": ordering_ready,
            "region_statistics": regions,
            "unique_fingerprints_by_policy": {
                policy: len({row["architecture_fingerprint"] for row in stats if row["decoder_policy"] == policy})
                for policy in POLICIES
            },
            "representation_sensitivity": representation_evidence,
        },
    }
    _dump_json(output / "stage3_conclusion.json", conclusion)
    run_metadata(output, [stage2_results, *shard_results], rows)
    lines = [
        "# Stage 3 multi-seed calibration report", "",
        f"Conclusion: `{classification}`", "",
        "This is a diagnostic-only result. Formal BO and the VAE checkpoint were not modified.", "",
        "## Execution and merge", "",
        f"- New evaluations: `432/432` (GPU0 216, GPU1 216)",
        f"- Merged evaluations: `{len(rows)}/540`",
        f"- Terminal counts: `{terminal_counts}`",
        f"- Duplicate / missing / unexpected keys: `0 / 0 / 0`",
        f"- Seed-invariant group-field checks: `{merge_audit['seed_invariant_group_field_checks']}`",
        f"- Result audit: `{result_audit_status or 'PENDING'}`", "",
        "## Noise gate", "",
        f"- Passed: `{gate['passed']}`",
        f"- Historical top-20 point-HP median t-CI half-width: `{gate['median_t_ci95_half_width']}` (limit 0.006)",
        f"- Fraction with half-width <= 0.010: `{gate['fraction_half_width_le_0_010']}` (minimum 0.75)",
        f"- All required seeds successful: `{gate['no_missing_invalid_failed_seeds']}`", "",
        "## Architecture independence", "",
        f"- Unique historical fingerprints: `{history['independent_fingerprint_count']}`",
        f"- Pre-registered minimum: `{history['minimum_required']}`",
        f"- VAE fine-tune ready: `{conclusion['vae_finetune_ready']}`", "",
        "## Odd/even ordering", "",
        f"- Historical top20 Kendall tau: `{top20_ordering['estimate']}`",
        f"- Historical top20 bootstrap 95% CI: `[{top20_ordering['bootstrap_ci95_low']}, {top20_ordering['bootstrap_ci95_high']}]`",
        f"- Ordering gate passed: `{ordering_ready}`", "",
        "## Policy/anomaly diagnostics", "",
        f"- Greedy policy instability flagged: `{conclusion['greedy_policy_instability']}`",
        f"- Successful / invalid / failed evaluations: `{terminal_counts}`",
        f"- Total retries: `{conclusion['numeric_evidence']['total_retries']}`", "",
        *[
            f"- `{row['point_id'][:12]}...` / `{row['decoder_policy']}`: `{row['classification']}` "
            f"(min={row.get('minimum')}, std={row.get('sample_std')}, largest_gap={row.get('largest_gap')})"
            for row in anomaly["anomaly_points"]
        ], "",
        "## Representation sensitivity", "",
        (
            "- Updated 15-seed, 18-point sensitivity classification: "
            f"`{representation_evidence['diagnostic_conclusion']['classification']}`; "
            f"metric rows: `{representation_evidence['metric_row_count']}`. "
            "This cannot override the independent-architecture stop gate."
            if representation_evidence is not None
            else "- Not attached yet; rerun analyze with --representation_result after comparison."
        ), "",
        f"Recommended next branch: `{recommendation}`", "",
    ]
    report = "\n".join(lines)
    (output / "stage3_final_report.md").write_text(report, encoding="utf-8")
    (output / "final_goal_report.md").write_text(report, encoding="utf-8")
    return conclusion


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="freeze and hash the Stage 3 task plan")
    plan.add_argument("--selection", type=Path, required=True)
    plan.add_argument("--stage2_results", type=Path, required=True)
    plan.add_argument("--history", type=Path, required=True)
    plan.add_argument("--checkpoint", type=Path, required=True)
    plan.add_argument("--code", type=Path, nargs="+", required=True)
    plan.add_argument("--output", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze", help="exactly merge and apply the pre-registered gates")
    analyze_parser.add_argument("--selection", type=Path, required=True)
    analyze_parser.add_argument("--stage2_results", type=Path, required=True)
    analyze_parser.add_argument("--shard_results", type=Path, nargs=2, required=True)
    analyze_parser.add_argument("--history", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--representation_result", type=Path, default=None)
    analyze_parser.add_argument(
        "--result_audit_status", choices=("PASS", "FAIL"), default=None
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        plan = create_plan(args.selection, args.stage2_results, args.history, args.checkpoint, args.output, args.code)
        print(json.dumps({key: plan[key] for key in ("fixed_point_count", "new_training_count", "merged_expected_count", "gpu_seed_assignment")}, indent=2, sort_keys=True))
    else:
        conclusion = analyze(
            args.selection,
            args.stage2_results,
            args.shard_results,
            args.history,
            args.output,
            args.representation_result,
            args.result_audit_status,
        )
        print(json.dumps(conclusion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
