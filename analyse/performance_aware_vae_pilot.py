#!/usr/bin/env python3
"""Minimal, isolated performance-aware VAE pilot.

The module never runs BO or GNN label training.  It consumes validation-only
labels, replays the original JointSpaceVAE reconstruction dataset, and writes
all new checkpoints below a caller-provided posthoc output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import igraph
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import (
    ArchitectureArgs,
    architecture_fingerprint,
    config_from_igraph,
    config_to_igraph,
    load_joint_vae,
    structure_feature_vector,
)
from eval_utils import _decode_arch


FORMAT_VERSION = "performance_aware_vae_pilot_global4_v1"
VARIANTS = (
    "original_checkpoint_probe",
    "encoder_regression",
    "encoder_regression_ranking",
)
NOISE_MODES = ("bounded_noise_weight", "no_noise_weight")
PERFORMANCE_STRENGTHS = (0.3, 3.0)
OUTER_FOLDS = 5
TRAINING_SEEDS = (20260831, 20260832, 20260833)
PROTECTED_CHECKPOINT_SHA256 = (
    "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4"
)
PROTECTED_STAGE3_SHA256 = (
    "9d5a779ac33e8656ac665d5c60fa61c335e8924da8f4dd3258a7ccede4eed93e"
)
CI_NOISE_REFERENCE = 0.006
RELIABLE_PAIR_DELTA = 0.006
RESULTS_JSONL = "training_results.jsonl"
PROGRESS_JSON = "progress.json"
SHARD_PLAN_JSON = "shard_plan.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_plan(plan_path: Path, plan: Mapping[str, Any]) -> None:
    sidecar = plan_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise RuntimeError("frozen task plan SHA256 sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) < 2 or fields[0] != sha256_file(plan_path):
        raise RuntimeError("frozen task plan SHA256 sidecar mismatch")
    for path_string, expected in plan["input_hashes"].items():
        path = Path(path_string)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen input hash changed: {path_string}")
    for path_string, expected in plan["code_hashes"].items():
        path = Path(path_string)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen code hash changed: {path_string}")
    folds_path = Path(plan["folds_file"])
    if not folds_path.is_file() or sha256_file(folds_path) != plan["folds_sha256"]:
        raise RuntimeError("frozen architecture folds hash changed")
    for path_string, expected in plan["protected_file_hashes_before"].items():
        path = Path(path_string)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"protected file changed: {path_string}")


def stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(stable_hash(*parts)[:8], 16) & 0x7FFFFFFF


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def contains_test_field(value: Any) -> bool:
    """Reject any recursively nested field whose key mentions test."""

    if isinstance(value, Mapping):
        return any(
            "test" in str(key).lower() or contains_test_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_test_field(item) for item in value)
    return False


def load_validation_labels(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("validation label file must be a non-empty JSON list")
    if contains_test_field(rows):
        raise ValueError("validation label file contains a test-named field")
    required = {
        "architecture_fingerprint", "operations", "edges", "raw_hp_vector",
        "validation_mean", "validation_ci95_low", "validation_ci95_high",
        "validation_ci95_half_width", "source", "time_index",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"label row {index} missing {sorted(missing)}")
        if len(row["raw_hp_vector"]) != 4:
            raise ValueError(f"label row {index} is not global4")
        observed = architecture_fingerprint(row)
        if observed != row["architecture_fingerprint"]:
            raise ValueError(f"label row {index} architecture fingerprint drift")
    return rows


def load_replay_dataset(path: Path) -> list[tuple[igraph.Graph, torch.Tensor]]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError("VAE replay dataset must be a non-empty list")
    processed = []
    for index, item in enumerate(raw):
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(f"replay row {index} has invalid schema")
        types, adjacency, hp = item
        if len(types) != 7 or np.asarray(adjacency).shape != (7, 7) or len(hp) != 4:
            raise ValueError(f"replay row {index} is incompatible with global4/max_n=7")
        graph = igraph.Graph(directed=True)
        graph.add_vertices(7)
        graph.vs["type"] = [int(value) for value in types]
        edges = [
            (source, destination)
            for source in range(7)
            for destination in range(7)
            if int(adjacency[source][destination]) == 1
        ]
        graph.add_edges(edges)
        if not graph.is_dag():
            raise ValueError(f"replay row {index} is not a DAG")
        processed.append((graph, torch.tensor(hp, dtype=torch.float32)))
    return processed


def _architecture_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(row)
    means = {
        fingerprint: statistics.fmean(float(row["validation_mean"]) for row in values)
        for fingerprint, values in grouped.items()
    }
    ordered = sorted(means, key=lambda fingerprint: (means[fingerprint], fingerprint))
    rank = {fingerprint: index for index, fingerprint in enumerate(ordered)}
    records = []
    for fingerprint, values in grouped.items():
        sources = sorted({str(row["source"]) for row in values})
        perf_bin = min(4, int(5 * rank[fingerprint] / max(len(ordered), 1)))
        records.append(
            {
                "architecture_fingerprint": fingerprint,
                "configuration_count": len(values),
                "validation_mean": means[fingerprint],
                "performance_bin": perf_bin,
                "source_group": "+".join(sources),
            }
        )
    return records


def balanced_group_folds(
    architecture_records: Sequence[Mapping[str, Any]],
    *,
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    """Deterministically balance performance/source strata across folds."""

    if len(architecture_records) < n_splits:
        raise ValueError("not enough architecture groups for requested folds")
    strata_totals = Counter(
        (str(row["source_group"]), int(row["performance_bin"]))
        for row in architecture_records
    )
    ordered = sorted(
        architecture_records,
        key=lambda row: (
            strata_totals[(str(row["source_group"]), int(row["performance_bin"]))],
            str(row["source_group"]),
            int(row["performance_bin"]),
            stable_hash(seed, row["architecture_fingerprint"]),
        ),
    )
    fold_counts = [0] * n_splits
    stratum_counts = [Counter() for _ in range(n_splits)]
    assignment: dict[str, int] = {}
    for row in ordered:
        stratum = (str(row["source_group"]), int(row["performance_bin"]))
        fold = min(
            range(n_splits),
            key=lambda candidate: (
                stratum_counts[candidate][stratum],
                fold_counts[candidate],
                stable_hash(seed, stratum, candidate),
            ),
        )
        fingerprint = str(row["architecture_fingerprint"])
        assignment[fingerprint] = fold
        fold_counts[fold] += 1
        stratum_counts[fold][stratum] += 1
    return assignment


def build_nested_folds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    architecture_records = _architecture_records(rows)
    outer_assignment = balanced_group_folds(
        architecture_records, n_splits=OUTER_FOLDS, seed=2026083101
    )
    all_architectures = {str(row["architecture_fingerprint"]) for row in architecture_records}
    folds = []
    for outer_fold in range(OUTER_FOLDS):
        outer_test = {fp for fp, fold in outer_assignment.items() if fold == outer_fold}
        outer_train = all_architectures - outer_test
        inner_records = [
            row for row in architecture_records
            if str(row["architecture_fingerprint"]) in outer_train
        ]
        inner_assignment = balanced_group_folds(
            inner_records, n_splits=OUTER_FOLDS, seed=2026083200 + outer_fold
        )
        inner_val_bucket = outer_fold % OUTER_FOLDS
        inner_val = {fp for fp, fold in inner_assignment.items() if fold == inner_val_bucket}
        inner_train = outer_train - inner_val
        if (outer_test & outer_train) or (inner_train & inner_val) or (outer_test & inner_val):
            raise AssertionError("architecture group leakage while constructing folds")
        if inner_train | inner_val != outer_train:
            raise AssertionError("inner split does not cover outer training groups")
        folds.append(
            {
                "outer_fold": outer_fold,
                "outer_test_architectures": sorted(outer_test),
                "outer_train_architectures": sorted(outer_train),
                "inner_train_architectures": sorted(inner_train),
                "inner_validation_architectures": sorted(inner_val),
                "leakage_count": 0,
            }
        )
    appearances = Counter(
        fingerprint
        for fold in folds
        for fingerprint in fold["outer_test_architectures"]
    )
    if set(appearances) != all_architectures or set(appearances.values()) != {1}:
        raise AssertionError("outer folds do not test every architecture exactly once")
    return folds


def noise_factor(half_width: float) -> float:
    denominator = max(float(half_width), CI_NOISE_REFERENCE)
    return float(np.clip((CI_NOISE_REFERENCE / denominator) ** 2, 0.1, 1.0))


def training_weights(
    rows: Sequence[Mapping[str, Any]], *, use_noise_weight: bool
) -> np.ndarray:
    counts = Counter(str(row["architecture_fingerprint"]) for row in rows)
    raw = []
    for row in rows:
        base = 1.0 / counts[str(row["architecture_fingerprint"])]
        factor = noise_factor(float(row["validation_ci95_half_width"])) if use_noise_weight else 1.0
        raw.append(base * factor)
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("invalid training weights")
    return values * (len(values) / values.sum())


def reliable_pairs(
    rows: Sequence[Mapping[str, Any]], *, high_region_only: bool = False
) -> list[tuple[int, int, int]]:
    if high_region_only:
        architecture_means = defaultdict(list)
        for row in rows:
            architecture_means[str(row["architecture_fingerprint"])].append(
                float(row["validation_mean"])
            )
        ranked = sorted(
            architecture_means,
            key=lambda fp: statistics.fmean(architecture_means[fp]),
            reverse=True,
        )
        keep = set(ranked[: max(1, math.ceil(0.2 * len(ranked)))])
    else:
        keep = None
    pairs = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            a, b = rows[left], rows[right]
            if a["architecture_fingerprint"] == b["architecture_fingerprint"]:
                continue
            if keep is not None and (
                str(a["architecture_fingerprint"]) not in keep
                or str(b["architecture_fingerprint"]) not in keep
            ):
                continue
            delta = float(a["validation_mean"]) - float(b["validation_mean"])
            if abs(delta) < RELIABLE_PAIR_DELTA:
                continue
            disjoint = (
                float(a["validation_ci95_low"]) > float(b["validation_ci95_high"])
                or float(b["validation_ci95_low"]) > float(a["validation_ci95_high"])
            )
            if disjoint:
                pairs.append((left, right, 1 if delta > 0 else -1))
    return pairs


def _rows_for_architectures(
    rows: Sequence[Mapping[str, Any]], fingerprints: Iterable[str]
) -> list[dict[str, Any]]:
    wanted = set(map(str, fingerprints))
    return [dict(row) for row in rows if str(row["architecture_fingerprint"]) in wanted]


def _make_tasks() -> list[dict[str, Any]]:
    tasks = []
    for fold in range(OUTER_FOLDS):
        for seed in TRAINING_SEEDS:
            for variant in VARIANTS:
                strengths = (0.0,) if variant == "original_checkpoint_probe" else PERFORMANCE_STRENGTHS
                for strength in strengths:
                    for noise_mode in NOISE_MODES:
                        task_id = (
                            f"fold{fold}_seed{seed}_{variant}_lambda{strength:g}_{noise_mode}"
                        )
                        tasks.append(
                            {
                                "task_id": task_id,
                                "outer_fold": fold,
                                "training_seed": seed,
                                "variant": variant,
                                "performance_strength": strength,
                                "noise_mode": noise_mode,
                            }
                        )
    tasks.sort(
        key=lambda row: (
            row["outer_fold"], row["training_seed"], row["variant"],
            row["performance_strength"], row["noise_mode"],
        )
    )
    for index, task in enumerate(tasks):
        task["assigned_shard"] = index % 2
    return tasks


def _git_output(*args: str) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return result.returncode, result.stdout + result.stderr


def prepare_plan(
    *,
    labels_path: Path,
    replay_path: Path,
    checkpoint_path: Path,
    stage3_path: Path,
    output: Path,
    code_paths: Sequence[Path],
    test_log: Path,
    test_command: str,
    test_exit_code: int,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"pilot output already exists and is not empty: {output}")
    if test_exit_code != 0:
        raise RuntimeError("BLOCKED_PREFLIGHT: required preflight tests did not pass")
    if sha256_file(checkpoint_path) != PROTECTED_CHECKPOINT_SHA256:
        raise RuntimeError("BLOCKED_PREFLIGHT: original VAE checkpoint hash mismatch")
    if sha256_file(stage3_path) != PROTECTED_STAGE3_SHA256:
        raise RuntimeError("BLOCKED_PREFLIGHT: protected Stage3 result hash mismatch")
    labels = load_validation_labels(labels_path)
    replay = load_replay_dataset(replay_path)
    if len(labels) != 76 or len({row["architecture_fingerprint"] for row in labels}) != 60:
        raise RuntimeError("BLOCKED_PREFLIGHT: expected 76 configurations and 60 architectures")
    if len(replay) != 3000:
        raise RuntimeError("BLOCKED_PREFLIGHT: expected 3000 original replay records")
    output.mkdir(parents=True, exist_ok=False)
    copied_test_log = output / "logs" / "preflight_tests.log"
    copied_test_log.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_log, copied_test_log)
    folds = build_nested_folds(labels)
    _dump_json(output / "architecture_group_folds.json", folds)

    diff_check_code, diff_check_output = _git_output("diff", "--check")
    status_code, status_output = _git_output("status", "--short")
    names_code, names_output = _git_output("diff", "--name-only")
    commit_code, commit_output = _git_output("rev-parse", "HEAD")
    if any(code != 0 for code in (status_code, names_code, commit_code)) or diff_check_code != 0:
        raise RuntimeError("BLOCKED_PREFLIGHT: git evidence command failed")
    bo_hash = sha256_file(REPO_ROOT / "bo_phase4.py")
    gpu_rows = []
    if torch.cuda.is_available():
        smi = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=index,name,uuid",
                "--format=csv,noheader,nounits",
            ], text=True, capture_output=True, check=True,
        )
        for line in smi.stdout.strip().splitlines():
            index, name, uuid = [part.strip() for part in line.split(",", 2)]
            gpu_rows.append({"index": int(index), "name": name, "uuid": uuid})
    if len(gpu_rows) < 2 or len({row["uuid"] for row in gpu_rows[:2]}) != 2:
        raise RuntimeError("BLOCKED_PREFLIGHT: two distinct GPUs are required for frozen plan")

    protected = {
        str((REPO_ROOT / "bo_phase4.py").resolve()): bo_hash,
        str(checkpoint_path.resolve()): sha256_file(checkpoint_path),
        str(stage3_path.resolve()): sha256_file(stage3_path),
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpu_rows,
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "scikit-learn", "python-igraph")
        },
    }
    preflight = {
        "classification": "PREFLIGHT_SEALED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit_output.strip(),
        "git_status_short": status_output.splitlines(),
        "git_diff_name_only": names_output.splitlines(),
        "git_diff_check_exit_code": diff_check_code,
        "git_diff_check_output": diff_check_output,
        "protected_file_hashes": protected,
        "input_hashes": {
            str(labels_path.resolve()): sha256_file(labels_path),
            str(replay_path.resolve()): sha256_file(replay_path),
            str((REPO_ROOT / "results/posthoc/vae-label-expansion-global4-v1/task_plan.json").resolve()): sha256_file(REPO_ROOT / "results/posthoc/vae-label-expansion-global4-v1/task_plan.json"),
            **{str(path.resolve()): sha256_file(path) for path in code_paths},
        },
        "label_recursive_test_field_scan_passed": not contains_test_field(labels),
        "label_configuration_count": len(labels),
        "label_architecture_count": len({row["architecture_fingerprint"] for row in labels}),
        "replay_record_count": len(replay),
        "test_command": test_command,
        "test_exit_code": int(test_exit_code),
        "test_command_log": str(copied_test_log.resolve()),
        "test_command_log_sha256": sha256_file(copied_test_log),
        "environment": environment,
    }
    _dump_json(output / "preflight_seal.json", preflight)
    (output / "preflight_seal.md").write_text(
        "# Performance-aware VAE pilot preflight\n\n"
        "Result: `PREFLIGHT_SEALED`\n\n"
        f"- Git commit: `{preflight['git_commit']}`\n"
        f"- Labels: `{len(labels)}` configurations / `60` architectures; recursive test scan passed.\n"
        f"- Replay: `{len(replay)}` original global4 records.\n"
        f"- Original checkpoint: `{protected[str(checkpoint_path.resolve())]}`\n"
        f"- Stage3 result: `{protected[str(stage3_path.resolve())]}`\n"
        f"- bo_phase4.py baseline: `{bo_hash}`\n"
        f"- GPUs: `{gpu_rows}`\n",
        encoding="utf-8",
    )

    tasks = _make_tasks()
    fixed_samples = sorted(
        range(len(replay)), key=lambda index: stable_hash("reconstruction", index)
    )[:60]
    plan = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "formal_bo_must_remain_unchanged": True,
        "inputs": {
            "labels": str(labels_path.resolve()),
            "replay": str(replay_path.resolve()),
            "original_checkpoint": str(checkpoint_path.resolve()),
            "stage3_result": str(stage3_path.resolve()),
        },
        "input_hashes": preflight["input_hashes"],
        "protected_file_hashes_before": protected,
        "folds_file": str((output / "architecture_group_folds.json").resolve()),
        "folds_sha256": sha256_file(output / "architecture_group_folds.json"),
        "outer_folds": OUTER_FOLDS,
        "num_shards": 2,
        "shard_gpu_uuid": {"0": gpu_rows[0]["uuid"], "1": gpu_rows[1]["uuid"]},
        "training_seeds": list(TRAINING_SEEDS),
        "variants": list(VARIANTS),
        "performance_strength_candidates": list(PERFORMANCE_STRENGTHS),
        "noise_modes": list(NOISE_MODES),
        "training": {
            "epochs": 24,
            "patience": 7,
            "head_learning_rate": 0.003,
            "encoder_learning_rate": 0.00002,
            "replay_batch_size": 24,
            "beta_kl": 0.005,
            "ranking_coefficient": 0.2,
            "max_reliable_pairs_per_epoch": 512,
            "gradient_clip": 5.0,
            "trainable_encoder_modules": ["arch_vae.hg_unify", "arch_vae.fc1", "arch_vae.fc2"],
            "decoder_frozen": True,
            "inner_selection": "aggregate three seeds, then maximize reliable_pairwise_accuracy, then kendall, then minimize rmse, then smaller strength",
        },
        "weighting": {
            "architecture_base": "1 / configurations_in_architecture",
            "bounded_noise": "clip((0.006 / max(ci95_half_width, 0.006))^2, 0.1, 1.0)",
            "normalize": "mean weight equals 1 within each inner training fold",
            "high_noise_records_retained": True,
        },
        "reliable_pair_rule": {
            "different_architectures": True,
            "minimum_validation_mean_gap": RELIABLE_PAIR_DELTA,
            "confidence_intervals_must_not_overlap": True,
        },
        "fixed_reconstruction_sample_indices": fixed_samples,
        "decoder_policies": ["historical_modal3", "deterministic_greedy"],
        "thresholds": {
            "improvement_ci95_low_strictly_positive": True,
            "reliable_top_pairwise_estimate_min": 0.60,
            "reliable_top_pairwise_ci95_low_min": 0.50,
            "top20_kendall_must_not_be_significantly_negative": True,
            "seed_reproduction_minimum": 2,
            "single_fold_may_not_drive_result": True,
            "decode_validity_drop_max": 0.02,
            "exact_reconstruction_drop_max": 0.05,
            "cycle_distance_ratio_max": 1.10,
            "kl_active_dimension_drop_max_fraction": 0.20,
            "effective_rank_drop_max_fraction": 0.20,
        },
        "bootstrap_unit": "architecture_fingerprint",
        "primary_candidate_model": "encoder_regression_ranking_head",
        "primary_estimator_weighting": "architecture-equal; configuration comparisons are averaged within architecture pairs",
        "minimum_reliable_top_architecture_pairs": 10,
        "minimum_reliable_top_participating_architectures": 6,
        "bootstrap_replicates": 2000,
        "tasks": tasks,
        "task_count": len(tasks),
        "shard_task_counts": {
            str(shard): sum(task["assigned_shard"] == shard for task in tasks)
            for shard in (0, 1)
        },
        "maximum_retries": 2,
        "code_hashes": {str(path.resolve()): sha256_file(path) for path in code_paths},
    }
    _dump_json(output / "task_plan.json", plan)
    (output / "task_plan.sha256").write_text(
        f"{sha256_file(output / 'task_plan.json')}  task_plan.json\n", encoding="utf-8"
    )
    return plan


class PerformanceHead(nn.Module):
    def __init__(self, input_dim: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, mu_arch: torch.Tensor, standardized_hp: torch.Tensor) -> torch.Tensor:
        if standardized_hp.ndim != 2 or standardized_hp.shape[1] != 4:
            raise ValueError("performance head requires four standardized global4 HP inputs")
        if mu_arch.shape[0] != standardized_hp.shape[0]:
            raise ValueError("architecture latent and HP batch sizes differ")
        return self.network(torch.cat([mu_arch, standardized_hp], dim=1)).squeeze(1)


def _standardization(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hp = np.asarray([row["raw_hp_vector"] for row in train_rows], dtype=np.float64)
    y = np.asarray([row["validation_mean"] for row in train_rows], dtype=np.float64)
    hp_mean, hp_std = hp.mean(0), hp.std(0)
    hp_std[hp_std < 1e-8] = 1.0
    y_std = float(y.std())
    if y_std < 1e-8:
        y_std = 1.0
    return {
        "hp_mean": hp_mean.tolist(), "hp_std": hp_std.tolist(),
        "y_mean": float(y.mean()), "y_std": y_std,
    }


def _tensor_inputs(
    rows: Sequence[Mapping[str, Any]], scaling: Mapping[str, Any], device: torch.device
) -> tuple[list[igraph.Graph], torch.Tensor, torch.Tensor]:
    graphs = [config_to_igraph(row) for row in rows]
    hp = np.asarray([row["raw_hp_vector"] for row in rows], dtype=np.float32)
    hp = (hp - np.asarray(scaling["hp_mean"], dtype=np.float32)) / np.asarray(
        scaling["hp_std"], dtype=np.float32
    )
    y = (
        np.asarray([row["validation_mean"] for row in rows], dtype=np.float32)
        - float(scaling["y_mean"])
    ) / float(scaling["y_std"])
    return graphs, torch.tensor(hp, device=device), torch.tensor(y, device=device)


def _encoder_override_state(model: nn.Module) -> dict[str, torch.Tensor]:
    prefixes = ("arch_vae.hg_unify.", "arch_vae.fc1.", "arch_vae.fc2.")
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def load_task_checkpoint(
    path: Path, original_checkpoint: Path, device: torch.device
) -> tuple[nn.Module, PerformanceHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = load_joint_vae(original_checkpoint, "global4", device=device)
    state = model.state_dict()
    state.update(payload["encoder_override_state"])
    model.load_state_dict(state)
    model.eval()
    head = PerformanceHead().to(device)
    head.load_state_dict(payload["performance_head_state"])
    head.eval()
    return model, head, payload


def _predict(
    model: nn.Module,
    head: PerformanceHead,
    rows: Sequence[Mapping[str, Any]],
    scaling: Mapping[str, Any],
    device: torch.device,
) -> np.ndarray:
    graphs, hp, _ = _tensor_inputs(rows, scaling, device)
    model.eval()
    head.eval()
    with torch.no_grad():
        mu, _ = model.arch_vae.encode(graphs)
        standardized = head(mu, hp)
    return (
        standardized.detach().cpu().numpy() * float(scaling["y_std"])
        + float(scaling["y_mean"])
    )


def _validation_metrics(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[float]
) -> dict[str, float | int | None]:
    from scipy.stats import kendalltau

    y = np.asarray([row["validation_mean"] for row in rows], dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    pairs = reliable_pairs(rows)
    pair_accuracy = (
        statistics.fmean(
            float((pred[left] - pred[right]) * direction > 0)
            for left, right, direction in pairs
        )
        if pairs else None
    )
    tau = kendalltau(y, pred, nan_policy="omit").statistic if len(y) > 1 else None
    return {
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "kendall": None if tau is None or not np.isfinite(tau) else float(tau),
        "reliable_pair_count": len(pairs),
        "reliable_pairwise_accuracy": pair_accuracy,
    }


def _actual_gpu(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "uuid": None, "name": "cpu"}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical = int(visible.split(",")[0]) if visible.split(",")[0].isdigit() else None
    uuid = None
    if physical is not None:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
        for line in result.stdout.splitlines():
            index, candidate = [part.strip() for part in line.split(",", 1)]
            if int(index) == physical:
                uuid = candidate
    return {
        "device": str(device), "physical_index": physical, "uuid": uuid,
        "name": torch.cuda.get_device_name(device), "cuda_visible_devices": visible,
    }


def train_task(
    task: Mapping[str, Any],
    plan: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    replay: Sequence[tuple[igraph.Graph, torch.Tensor]],
    output: Path,
    device: torch.device,
    *,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    set_all_seeds(int(task["training_seed"]))
    fold = folds[int(task["outer_fold"])]
    train_rows = _rows_for_architectures(labels, fold["inner_train_architectures"])
    validation_rows = _rows_for_architectures(labels, fold["inner_validation_architectures"])
    test_rows = _rows_for_architectures(labels, fold["outer_test_architectures"])
    scaling = _standardization(train_rows)
    use_noise = task["noise_mode"] == "bounded_noise_weight"
    weights = torch.tensor(
        training_weights(train_rows, use_noise_weight=use_noise),
        dtype=torch.float32, device=device,
    )
    train_graphs, train_hp, train_y = _tensor_inputs(train_rows, scaling, device)
    validation_graphs, validation_hp, _ = _tensor_inputs(validation_rows, scaling, device)

    original_checkpoint = Path(plan["inputs"]["original_checkpoint"])
    model = load_joint_vae(original_checkpoint, "global4", device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if task["variant"] != "original_checkpoint_probe":
        for module in (model.arch_vae.hg_unify, model.arch_vae.fc1, model.arch_vae.fc2):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    head = PerformanceHead().to(device)
    encoder_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_groups = [{"params": head.parameters(), "lr": plan["training"]["head_learning_rate"]}]
    if encoder_parameters:
        optimizer_groups.append(
            {"params": encoder_parameters, "lr": plan["training"]["encoder_learning_rate"]}
        )
    optimizer = torch.optim.Adam(optimizer_groups)
    pair_rows = reliable_pairs(train_rows)
    epochs = int(smoke_epochs or plan["training"]["epochs"])
    patience = int(plan["training"]["patience"])
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    trace = []

    for epoch in range(epochs):
        model.train(task["variant"] != "original_checkpoint_probe")
        head.train()
        optimizer.zero_grad()
        mu, logvar = model.arch_vae.encode(train_graphs)
        predictions = head(mu, train_hp)
        regression = torch.sum(weights * (predictions - train_y) ** 2) / torch.sum(weights)
        ranking = torch.zeros((), device=device)
        if task["variant"] == "encoder_regression_ranking" and pair_rows:
            selected_pairs = sorted(
                pair_rows,
                key=lambda pair: stable_hash(task["task_id"], epoch, pair),
            )[: int(plan["training"]["max_reliable_pairs_per_epoch"])]
            left = torch.tensor([pair[0] for pair in selected_pairs], device=device)
            right = torch.tensor([pair[1] for pair in selected_pairs], device=device)
            direction = torch.tensor(
                [pair[2] for pair in selected_pairs], dtype=torch.float32, device=device
            )
            ranking = F.softplus(-direction * (predictions[left] - predictions[right])).mean()
        reconstruction = torch.zeros((), device=device)
        kl = torch.zeros((), device=device)
        if task["variant"] != "original_checkpoint_probe":
            batch_size = int(plan["training"]["replay_batch_size"])
            replay_indices = sorted(
                range(len(replay)),
                key=lambda index: stable_hash(task["task_id"], epoch, index),
            )[:batch_size]
            replay_graphs = [replay[index][0].copy() for index in replay_indices]
            replay_mu, replay_logvar = model.arch_vae.encode(replay_graphs)
            _, reconstruction_sum, kl_sum = model.arch_vae.loss(
                replay_mu, replay_logvar, replay_graphs,
                beta=float(plan["training"]["beta_kl"]),
            )
            reconstruction = reconstruction_sum / batch_size
            kl = kl_sum / batch_size
        performance = regression + float(plan["training"]["ranking_coefficient"]) * ranking
        loss = reconstruction + float(plan["training"]["beta_kl"]) * kl
        loss = loss + float(task["performance_strength"]) * performance
        if task["variant"] == "original_checkpoint_probe":
            loss = regression
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite pilot training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(
            [*head.parameters(), *encoder_parameters],
            float(plan["training"]["gradient_clip"]),
        )
        optimizer.step()

        validation_predictions = _predict(model, head, validation_rows, scaling, device)
        metrics = _validation_metrics(validation_rows, validation_predictions)
        score = float(metrics["rmse"])
        trace.append(
            {
                "epoch": epoch + 1, "loss": float(loss.detach().cpu()),
                "regression": float(regression.detach().cpu()),
                "ranking": float(ranking.detach().cpu()),
                "reconstruction": float(reconstruction.detach().cpu()),
                "kl": float(kl.detach().cpu()), "inner_rmse": score,
            }
        )
        if score < best_score - 1e-10:
            best_score = score
            best_epoch = epoch + 1
            no_improve = 0
            best_state = {
                "encoder": _encoder_override_state(model),
                "head": {name: value.detach().cpu() for name, value in head.state_dict().items()},
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is None:
        raise RuntimeError("pilot task produced no checkpoint")
    current = model.state_dict()
    current.update(best_state["encoder"])
    model.load_state_dict(current)
    head.load_state_dict(best_state["head"])
    inner_predictions = _predict(model, head, validation_rows, scaling, device)
    outer_predictions = _predict(model, head, test_rows, scaling, device)
    inner_metrics = _validation_metrics(validation_rows, inner_predictions)
    checkpoint_path = output / "candidate_checkpoints" / f"{task['task_id']}.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "format_version": FORMAT_VERSION,
        "task": dict(task),
        "original_checkpoint_sha256": sha256_file(original_checkpoint),
        "encoder_override_state": best_state["encoder"],
        "performance_head_state": best_state["head"],
        "scaling": scaling,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    return {
        "inner_metrics": inner_metrics,
        "inner_predictions": [
            {
                "config_id": row["config_id"],
                "architecture_fingerprint": row["architecture_fingerprint"],
                "validation_mean": row["validation_mean"],
                "prediction": float(prediction),
            }
            for row, prediction in zip(validation_rows, inner_predictions)
        ],
        "outer_predictions": [
            {
                "config_id": row["config_id"],
                "architecture_fingerprint": row["architecture_fingerprint"],
                "validation_mean": row["validation_mean"],
                "validation_ci95_low": row["validation_ci95_low"],
                "validation_ci95_high": row["validation_ci95_high"],
                "validation_ci95_half_width": row["validation_ci95_half_width"],
                "prediction": float(prediction),
            }
            for row, prediction in zip(test_rows, outer_predictions)
        ],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": best_epoch,
        "epochs_ran": len(trace),
        "training_trace": trace,
        "train_architecture_count": len(set(fold["inner_train_architectures"])),
        "validation_architecture_count": len(set(fold["inner_validation_architectures"])),
        "outer_architecture_count": len(set(fold["outer_test_architectures"])),
        "training_weight_sum": float(weights.sum().detach().cpu()),
        "reliable_training_pair_count": len(pair_rows),
        "scaling": scaling,
    }


def run_shard(args: argparse.Namespace) -> int:
    plan_path = args.task_plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported task plan format")
    verify_frozen_plan(plan_path, plan)
    if args.num_shards != int(plan["num_shards"]):
        raise RuntimeError("runtime shard count differs from frozen plan")
    tasks = [task for task in plan["tasks"] if int(task["assigned_shard"]) == args.shard_index]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    completed_rows = _load_jsonl(output / RESULTS_JSONL)
    completed_ids = {str(row["task_id"]) for row in completed_rows}
    if len(completed_ids) != len(completed_rows):
        raise ValueError("resume shard contains duplicate task IDs")
    expected_ids = {str(task["task_id"]) for task in tasks}
    if not completed_ids <= expected_ids:
        raise ValueError("resume shard contains tasks outside frozen shard")
    shard_plan = {
        "dry_run": bool(args.dry_run), "smoke_epochs": args.smoke_epochs,
        "task_plan": str(plan_path), "task_plan_sha256": sha256_file(plan_path),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "task_count": len(tasks), "task_ids": [task["task_id"] for task in tasks],
        "already_completed": len(completed_ids),
        "pending": len(expected_ids - completed_ids),
    }
    _dump_json(output / SHARD_PLAN_JSON, shard_plan)
    if args.dry_run:
        print(json.dumps(shard_plan, indent=2, sort_keys=True))
        return 0
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    labels = load_validation_labels(Path(plan["inputs"]["labels"]))
    replay = load_replay_dataset(Path(plan["inputs"]["replay"]))
    folds = json.loads(Path(plan["folds_file"]).read_text(encoding="utf-8"))
    gpu = _actual_gpu(device)
    expected_uuid = plan["shard_gpu_uuid"][str(args.shard_index)]
    if gpu.get("uuid") != expected_uuid:
        raise RuntimeError(
            f"shard {args.shard_index} expected GPU UUID {expected_uuid}, got {gpu.get('uuid')}"
        )
    started = datetime.now(timezone.utc).isoformat()

    def write_progress() -> None:
        _dump_json(
            output / PROGRESS_JSON,
            {
                "started_at": started,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "expected": len(tasks), "completed": len(completed_rows),
                "remaining": len(tasks) - len(completed_rows),
                "success": sum(row.get("terminal_status") == "success" for row in completed_rows),
                "failed": sum(row.get("terminal_status") == "failed" for row in completed_rows),
                "retries": sum(int(row.get("retry_count", 0)) for row in completed_rows),
                "actual_gpu": gpu,
            },
        )

    write_progress()
    for task in tasks:
        if task["task_id"] in completed_ids:
            continue
        attempt_failures = []
        row_started = datetime.now(timezone.utc).isoformat()
        monotonic_started = time.monotonic()
        for attempt in range(1, int(plan["maximum_retries"]) + 2):
            try:
                payload = train_task(
                    task, plan, folds, labels, replay, output, device,
                    smoke_epochs=args.smoke_epochs,
                )
                row = {
                    "task_id": task["task_id"], **dict(task), **payload,
                    "terminal_status": "success", "attempt_count": attempt,
                    "retry_count": attempt - 1, "attempt_failures": attempt_failures,
                    "started_at": row_started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.monotonic() - monotonic_started,
                    "actual_gpu": gpu,
                    "input_hashes": plan["input_hashes"],
                    "original_checkpoint_sha256": sha256_file(plan["inputs"]["original_checkpoint"]),
                }
                break
            except Exception as exc:
                attempt_failures.append(
                    {
                        "attempt": attempt, "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    }
                )
                if attempt <= int(plan["maximum_retries"]):
                    continue
                row = {
                    "task_id": task["task_id"], **dict(task),
                    "terminal_status": "failed", "attempt_count": attempt,
                    "retry_count": attempt - 1, "attempt_failures": attempt_failures,
                    "started_at": row_started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.monotonic() - monotonic_started,
                    "actual_gpu": gpu, "input_hashes": plan["input_hashes"],
                    "original_checkpoint_sha256": sha256_file(plan["inputs"]["original_checkpoint"]),
                }
                break
        _append_jsonl(output / RESULTS_JSONL, row)
        completed_rows.append(row)
        write_progress()
        print(
            f"PROGRESS shard={args.shard_index} completed={len(completed_rows)}/{len(tasks)} "
            f"failed={sum(item.get('terminal_status') == 'failed' for item in completed_rows)}",
            flush=True,
        )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--labels", type=Path, required=True)
    prepare.add_argument("--replay", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--stage3", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--code", type=Path, nargs="+", required=True)
    prepare.add_argument("--test_log", type=Path, required=True)
    prepare.add_argument("--test_command", required=True)
    prepare.add_argument("--test_exit_code", type=int, required=True)
    run = commands.add_parser("run")
    run.add_argument("--task_plan", type=Path, required=True)
    run.add_argument("--num_shards", type=int, choices=(1, 2), default=2)
    run.add_argument("--shard_index", type=int, choices=(0, 1), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default=None)
    run.add_argument("--dry_run", action="store_true")
    run.add_argument("--smoke_epochs", type=int, default=None)
    run.add_argument("--max_tasks", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        plan = prepare_plan(
            labels_path=args.labels.resolve(), replay_path=args.replay.resolve(),
            checkpoint_path=args.checkpoint.resolve(), stage3_path=args.stage3.resolve(),
            output=args.output.resolve(), code_paths=[path.resolve() for path in args.code],
            test_log=args.test_log.resolve(), test_command=args.test_command,
            test_exit_code=args.test_exit_code,
        )
        print(
            json.dumps(
                {"task_count": plan["task_count"], "shard_task_counts": plan["shard_task_counts"]},
                indent=2, sort_keys=True,
            )
        )
        return 0
    if args.num_shards == 1 and args.shard_index != 0:
        raise ValueError("single-shard execution requires --shard_index 0")
    return run_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
