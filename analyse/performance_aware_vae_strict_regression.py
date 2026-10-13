#!/usr/bin/env python3
"""Strict validation-only performance-aware VAE regression experiment.

This diagnostic trains frozen baselines and fully trainable architecture VAEs on
the fixed representation-comparison splits.  It never imports a GNN training
entry point and never invokes BO.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from scipy.stats import t

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import (
    architecture_fingerprint,
    config_from_igraph,
    config_to_igraph,
    load_joint_vae,
    structure_feature_vector,
)
from analyse.compare_matched_representations import (
    evaluate_protocol,
    grouped_repeated_splits,
    time_prefix_splits,
)
from analyse.performance_aware_vae_pilot import (
    PerformanceHead as LegacyPerformanceHead,
    contains_test_field,
    load_replay_dataset,
    load_validation_labels,
    set_all_seeds,
    sha256_file,
    stable_hash,
)
from analyse.vae_label_expansion import hp_fingerprint


FORMAT_VERSION = "performance_aware_vae_strict_regression_global4_v2"
LAMBDAS = (0.1, 0.5, 0.9)
TRAINING_SEEDS = (0, 1, 2)
DATASET_VARIANTS = ("full", "exclude_high_noise")
BASELINE_MODELS = ("frozen_head", "frozen_hybrid_head")
TASK_KINDS = ("baseline", "strict_ft")
MAXIMUM_RETRIES = 2
RESULTS_DIRNAME = "task_results"
PROGRESS_FILENAME = "training_progress.json"
EXPECTED_BASE_CHECKPOINT_SHA256 = (
    "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _write_sha256_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def validate_lambda(value: float) -> float:
    observed = float(value)
    if observed not in LAMBDAS:
        raise ValueError(f"lambda must be one of {LAMBDAS}, got {observed}")
    return observed


def convex_combination_loss(
    vae_loss: torch.Tensor, mse_loss: torch.Tensor, lambda_value: float
) -> torch.Tensor:
    """Return the preregistered positive-sign convex objective."""

    value = float(lambda_value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("convex-combination lambda must be in [0, 1]")
    return (1.0 - value) * vae_loss + value * mse_loss


def module_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_delta_norm(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    names: Iterable[str] | None = None,
) -> float:
    selected = set(before) if names is None else set(names)
    total = 0.0
    for name in selected:
        delta = after[name].detach().cpu().double() - before[name].detach().cpu().double()
        total += float(torch.sum(delta * delta))
    return math.sqrt(total)


def gradient_norm(
    loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter], *, retain_graph: bool
) -> float:
    if not parameters or not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total = 0.0
    for gradient in gradients:
        if gradient is not None:
            total += float(torch.sum(gradient.detach().double() ** 2))
    return math.sqrt(total)


def _record_order(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (int(row["time_index"]), str(row["config_id"])),
    )


def build_fixed_splits(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = _record_order(rows)
    records = [
        {
            "point_id": row["config_id"],
            "architecture_fingerprint": row["architecture_fingerprint"],
            "time_index": row["time_index"],
        }
        for row in ordered
    ]
    protocols = {
        "time_prefix": time_prefix_splits(
            records, min_train=20, step=10, validation_window=15
        ),
        "group_repeated": grouped_repeated_splits(
            records,
            repeats=50,
            seed=20260821,
            validation_fraction=0.30,
        ),
    }
    output: list[dict[str, Any]] = []
    for protocol, splits in protocols.items():
        for split in splits:
            train_rows = [ordered[index] for index in split["train"]]
            validation_rows = [ordered[index] for index in split["validation"]]
            train_arch = sorted({str(row["architecture_fingerprint"]) for row in train_rows})
            validation_arch = sorted(
                {str(row["architecture_fingerprint"]) for row in validation_rows}
            )
            overlap = sorted(set(train_arch) & set(validation_arch))
            if overlap:
                raise ValueError(f"architecture leakage in {protocol}/{split['split_id']}")
            output.append(
                {
                    "protocol": protocol,
                    "split_id": str(split["split_id"]),
                    "train_config_ids": [str(row["config_id"]) for row in train_rows],
                    "validation_config_ids": [
                        str(row["config_id"]) for row in validation_rows
                    ],
                    "train_architecture_fingerprints": train_arch,
                    "validation_architecture_fingerprints": validation_arch,
                    "train_observation_count": len(train_rows),
                    "validation_observation_count": len(validation_rows),
                    "train_architecture_count": len(train_arch),
                    "validation_architecture_count": len(validation_arch),
                    "leakage_count": 0,
                }
            )
    return output


def filter_split_manifest(
    full_splits: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    available = {str(row["config_id"]): row for row in rows}
    output = []
    for split in full_splits:
        train_ids = [item for item in split["train_config_ids"] if item in available]
        validation_ids = [
            item for item in split["validation_config_ids"] if item in available
        ]
        train_arch = sorted(
            {str(available[item]["architecture_fingerprint"]) for item in train_ids}
        )
        validation_arch = sorted(
            {str(available[item]["architecture_fingerprint"]) for item in validation_ids}
        )
        overlap = sorted(set(train_arch) & set(validation_arch))
        if overlap:
            raise ValueError(f"filtered split leakage: {split['split_id']}")
        if len(train_arch) < 4 or len(validation_arch) < 2:
            raise ValueError(f"filtered split is too small: {split['split_id']}")
        output.append(
            {
                **{key: split[key] for key in ("protocol", "split_id")},
                "train_config_ids": train_ids,
                "validation_config_ids": validation_ids,
                "train_architecture_fingerprints": train_arch,
                "validation_architecture_fingerprints": validation_arch,
                "train_observation_count": len(train_ids),
                "validation_observation_count": len(validation_ids),
                "train_architecture_count": len(train_arch),
                "validation_architecture_count": len(validation_arch),
                "leakage_count": 0,
            }
        )
    return output


def assert_same_architecture_side(
    split: Mapping[str, Any], rows_by_id: Mapping[str, Mapping[str, Any]]
) -> None:
    train = {
        str(rows_by_id[item]["architecture_fingerprint"])
        for item in split["train_config_ids"]
    }
    validation = {
        str(rows_by_id[item]["architecture_fingerprint"])
        for item in split["validation_config_ids"]
    }
    if train & validation:
        raise ValueError("same architecture appears on both split sides")


def _input_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in paths}


def verify_frozen_plan(plan_path: Path, plan: Mapping[str, Any]) -> None:
    if plan.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported strict task plan")
    sidecar = plan_path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != sha256_file(plan_path):
        raise RuntimeError("frozen task plan sidecar mismatch")
    for section in ("input_hashes", "code_hashes", "protected_hashes"):
        for name, expected in plan[section].items():
            path = Path(name)
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"frozen {section} drift: {path}")
    manifest = Path(plan["fixed_split_manifest"])
    if sha256_file(manifest) != plan["fixed_split_manifest_sha256"]:
        raise RuntimeError("fixed split manifest changed")


def _old_pilot_audit(old_root: Path) -> dict[str, Any]:
    plan = json.loads((old_root / "task_plan.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (old_root / "merged_training_results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    unique_ids = {str(row["task_id"]) for row in rows}
    ranking_tasks = sum(row["variant"] == "encoder_regression_ranking" for row in rows)
    archived_metrics = []
    with (old_root / "performance_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if (
                row["scope"] == "seed_mean"
                and row["noise_mode"] == "bounded_noise_weight"
                and row["metric"]
                in {
                    "late_kendall",
                    "top20_kendall",
                    "top_pairwise_accuracy",
                    "reliable_top_pairwise_accuracy",
                }
            ):
                archived_metrics.append(
                    {
                        "model": row["model"],
                        "metric": row["metric"],
                        "estimate": float(row["estimate"]),
                    }
                )
    heldout_config_ids = set()
    heldout_fields = []
    with (old_root / "heldout_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        heldout_fields = list(reader.fieldnames or [])
        for row in reader:
            heldout_config_ids.add(str(row["config_id"]))
    return {
        "actual_path": str(old_root.resolve()),
        "planned_task_count": int(plan["task_count"]),
        "observed_task_count": len(rows),
        "unique_task_count": len(unique_ids),
        "all_success": all(row.get("terminal_status") == "success" for row in rows),
        "old_split_file": plan["folds_file"],
        "old_outer_fold_count": len(json.loads(Path(plan["folds_file"]).read_text())),
        "predictor_input": "raw mu_arch + train-standardized global4 HP",
        "predictor_structure": "Linear(input,32)-ReLU-Linear(32,1)",
        "optimizer": "Adam",
        "training": plan["training"],
        "decoder_frozen": bool(plan["training"]["decoder_frozen"]),
        "performance_strengths": plan["performance_strength_candidates"],
        "ranking_task_count": ranking_tasks,
        "ranking_coefficient": plan["training"]["ranking_coefficient"],
        "heldout_prediction_file": str((old_root / "heldout_predictions.csv").resolve()),
        "heldout_configuration_count": len(heldout_config_ids),
        "heldout_contains_test_field": any(
            "test" in field.lower() for field in heldout_fields
        ),
        "archived_metric_snapshot": archived_metrics,
    }


def _recompute_label_provenance(
    labels: Sequence[Mapping[str, Any]], seed_audit_path: Path
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with seed_audit_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            # Sealed test values are deliberately not copied into the allowed view.
            allowed = {
                key: raw.get(key)
                for key in (
                    "config_id",
                    "architecture_fingerprint",
                    "operations",
                    "edges",
                    "hp_fingerprint",
                    "lr",
                    "dropout",
                    "hidden_dim",
                    "l2",
                    "gat_heads",
                    "sage_aggr",
                    "gin_eps",
                    "gat_heads_by_layer",
                    "sage_aggr_by_layer",
                    "gin_eps_by_layer",
                    "normalization",
                    "val_acc",
                    "terminal_status",
                )
            }
            grouped[str(allowed["config_id"])].append(allowed)
    label_lookup = {str(row["config_id"]): row for row in labels}
    mismatches = []
    recomputed = []
    for config_id, label in sorted(label_lookup.items()):
        successful = [
            row for row in grouped.get(config_id, []) if row["terminal_status"] == "success"
        ]
        values = [float(row["val_acc"]) for row in successful]
        if not values:
            mismatches.append({"config_id": config_id, "reason": "no validation rows"})
            continue
        first = {
            **dict(label),
            "gat_heads": 1,
            "sage_aggr": "mean",
            "gin_eps": 0.0,
            "gat_heads_by_layer": None,
            "sage_aggr_by_layer": None,
            "gin_eps_by_layer": None,
            **{
                key: value
                for key, value in successful[0].items()
                if value is not None
            },
        }
        arch = architecture_fingerprint(label)
        hp = stable_hash(*[float(value) for value in label["raw_hp_vector"]])
        legacy_hp_recomputed = hp_fingerprint(first)
        mean = statistics.fmean(values)
        sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
        half_width = (
            float(t.ppf(0.975, len(values) - 1) * sample_std / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
        checks = {
            "architecture_fingerprint": arch == label["architecture_fingerprint"],
            "seed_rows_hp_fingerprint": all(
                row["hp_fingerprint"] == label["hp_fingerprint"]
                for row in successful
            ),
            "seed_rows_architecture_fingerprint": all(
                row["architecture_fingerprint"] == label["architecture_fingerprint"]
                for row in successful
            ),
            "validation_mean": math.isclose(
                mean, float(label["validation_mean"]), rel_tol=0.0, abs_tol=1e-12
            ),
            "validation_sample_std": math.isclose(
                sample_std,
                float(label["validation_sample_std"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "validation_ci95_half_width": math.isclose(
                half_width,
                float(label["validation_ci95_half_width"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
        }
        if not all(checks.values()):
            mismatches.append({"config_id": config_id, "checks": checks})
        recomputed.append(
            {
                "config_id": config_id,
                "architecture_fingerprint": arch,
                "strict_raw_hp_fingerprint": hp,
                "legacy_archived_hp_fingerprint": label["hp_fingerprint"],
                "legacy_helper_recomputed_hp_fingerprint": legacy_hp_recomputed,
                "legacy_helper_matches_archived": legacy_hp_recomputed
                == label["hp_fingerprint"],
                "validation_mean": mean,
                "validation_sample_std": sample_std,
                "validation_ci95_half_width": half_width,
                "validation_seed_count": len(values),
            }
        )
    return {
        "seed_level_audit_count": sum(len(rows) for rows in grouped.values()),
        "configuration_count": len(recomputed),
        "mismatches": mismatches,
        "legacy_helper_match_count": sum(
            bool(row["legacy_helper_matches_archived"]) for row in recomputed
        ),
        "strict_hp_fingerprint_rule": "SHA256 canonical hash of ordered raw_hp_vector",
        "legacy_hp_fingerprint_note": (
            "Archived fingerprints are consistent across seed rows. Some historical "
            "records predate the current helper's canonical payload, so the strict audit "
            "also recomputes a stable fingerprint directly from raw_hp_vector."
        ),
        "recomputed_rows_sha256": stable_hash(*recomputed),
        "sealed_test_values_used": False,
        "passed": not mismatches,
    }


def _split_reproduction(
    splits: Sequence[Mapping[str, Any]],
    unified_root: Path,
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_audit = json.loads(
        (unified_root / "representation_split_audit.json").read_text(encoding="utf-8")
    )
    expected = {(row["protocol"], row["split_id"]): row for row in expected_audit}
    observed = {(row["protocol"], row["split_id"]): row for row in splits}
    mismatches = []
    for key in sorted(set(expected) | set(observed)):
        if key not in expected or key not in observed:
            mismatches.append({"split": key, "reason": "missing"})
            continue
        left, right = expected[key], observed[key]
        mapping = {
            "train_observation_count": "train_observation_count",
            "validation_observation_count": "validation_observation_count",
            "train_architecture_group_count": "train_architecture_count",
            "validation_architecture_group_count": "validation_architecture_count",
            "leakage_count": "leakage_count",
        }
        for expected_name, observed_name in mapping.items():
            if int(left[expected_name]) != int(right[observed_name]):
                mismatches.append(
                    {
                        "protocol": key[0],
                        "split_id": key[1],
                        "field": expected_name,
                        "expected": left[expected_name],
                        "observed": right[observed_name],
                    }
                )
    prediction_members: dict[tuple[str, str], set[str]] = {}
    with (unified_root / "representation_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            prediction_members.setdefault((row["protocol"], row["split_id"]), set()).add(
                row["point_id"]
            )
    membership_mismatches = []
    for key, split in observed.items():
        actual = set(split["validation_config_ids"])
        if prediction_members.get(key, set()) != actual:
            membership_mismatches.append(
                {
                    "protocol": key[0],
                    "split_id": key[1],
                    "expected_validation_count": len(prediction_members.get(key, set())),
                    "observed_validation_count": len(actual),
                }
            )
    representation_records = [
        {
            "point_id": row["config_id"],
            "decoder_policy": "unified_300_80",
            "architecture_fingerprint": row["architecture_fingerprint"],
            "operations": row["operations"],
            "edges": row["edges"],
            "z_arch": row["z_arch"],
            "mu_arch": row["mu_arch"],
            "raw_hp_vector": row["raw_hp_vector"],
            "condition_mask_vector": [1.0, 1.0, 1.0, 1.0],
            "time_index": row["time_index"],
            "label": row["validation_mean"],
        }
        for row in _record_order(labels)
    ]
    protocol_indices = {
        "time_prefix": time_prefix_splits(
            representation_records, min_train=20, step=10, validation_window=15
        ),
        "group_repeated": grouped_repeated_splits(
            representation_records,
            repeats=50,
            seed=20260821,
            validation_fraction=0.30,
        ),
    }
    reproduced_metrics = []
    for protocol, protocol_splits in protocol_indices.items():
        metrics, _predictions = evaluate_protocol(
            representation_records,
            protocol_splits,
            protocol,
            "unified_300_80",
            "validation_seed_mean",
            20260821,
            2000,
        )
        reproduced_metrics.extend(metrics)
    archived_metrics = {}
    with (unified_root / "representation_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            archived_metrics[(row["protocol"], row["representation"], row["metric"])] = row
    metric_mismatches = []
    for row in reproduced_metrics:
        key = (row["protocol"], row["representation"], row["metric"])
        archived = archived_metrics.get(key)
        if archived is None:
            metric_mismatches.append({"key": key, "reason": "missing archived metric"})
            continue
        for field in ("estimate", "ci95_low", "ci95_high"):
            if not math.isclose(
                float(row[field]), float(archived[field]), rel_tol=0.0, abs_tol=1e-12
            ):
                metric_mismatches.append(
                    {
                        "key": key,
                        "field": field,
                        "expected": float(archived[field]),
                        "observed": float(row[field]),
                    }
                )
    return {
        "fixed_seed": 20260821,
        "time_prefix_parameters": {
            "min_train": 20,
            "step": 10,
            "validation_window": 15,
        },
        "grouped_repeated_parameters": {
            "repeats": 50,
            "validation_fraction": 0.30,
        },
        "expected_split_count": len(expected),
        "observed_split_count": len(observed),
        "count_mismatches": mismatches,
        "validation_membership_mismatches": membership_mismatches,
        "representation_metric_count": len(reproduced_metrics),
        "representation_metric_mismatches": metric_mismatches,
        "group_leakage_count": sum(int(row["leakage_count"]) for row in splits),
        "passed": not mismatches and not membership_mismatches and not metric_mismatches,
        "historical_estimator_note": (
            "The unified Zs/Ze/D/H comparison used sklearn BayesianRidge, not an "
            "RBF GP. Its archived predictions are retained for exact split reproduction; "
            "the strict Euclidean experiment evaluates a preregistered fixed RBF kernel "
            "for both original and fine-tuned latents."
        ),
    }


def _make_tasks(splits_by_variant: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for dataset_variant in DATASET_VARIANTS:
        for split in splits_by_variant[dataset_variant]:
            common = {
                "dataset_variant": dataset_variant,
                "protocol": split["protocol"],
                "split_id": split["split_id"],
            }
            for seed in TRAINING_SEEDS:
                for model in BASELINE_MODELS:
                    tasks.append(
                        {
                            **common,
                            "task_kind": "baseline",
                            "model": model,
                            "training_seed": seed,
                            "lambda": None,
                        }
                    )
                for value in LAMBDAS:
                    tasks.append(
                        {
                            **common,
                            "task_kind": "strict_ft",
                            "model": "strict_ft_predictor",
                            "training_seed": seed,
                            "lambda": value,
                        }
                    )
    for task in tasks:
        suffix = task["model"] if task["lambda"] is None else f"lambda{task['lambda']:g}"
        task["task_id"] = (
            f"{task['dataset_variant']}__{task['protocol']}__{task['split_id']}__"
            f"seed{task['training_seed']}__{suffix}"
        )
    tasks.sort(key=lambda row: row["task_id"])
    for index, task in enumerate(tasks):
        task["assigned_shard"] = index % 2
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise AssertionError("task IDs are not unique")
    return tasks


def prepare_experiment(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"strict output is not empty: {output}")
    if args.test_exit_code != 0:
        raise RuntimeError("preflight CPU-safe tests did not pass")
    unified_root = args.unified_root.resolve()
    old_root = args.old_pilot_root.resolve()
    labels_path = unified_root / "unified_training_data_no_test.json"
    labels = load_validation_labels(labels_path)
    label_provenance = _recompute_label_provenance(
        labels, unified_root / "unified_seed_level_audit.jsonl"
    )
    seed_audit_count = int(label_provenance["seed_level_audit_count"])
    architectures = {str(row["architecture_fingerprint"]) for row in labels}
    readiness = json.loads((unified_root / "vae_pilot_readiness.json").read_text())
    if (
        len(labels) != 76
        or len(architectures) != 60
        or seed_audit_count != 2130
        or bool(readiness["test_label_used"])
        or int(readiness["group_leakage_count"]) != 0
        or not bool(readiness["vae_finetune_ready"])
        or contains_test_field(labels)
        or not label_provenance["passed"]
    ):
        raise RuntimeError("DATA_OR_SPLIT_INCONSISTENCY: unified label gate failed")
    if sha256_file(args.base_checkpoint) != EXPECTED_BASE_CHECKPOINT_SHA256:
        raise RuntimeError("DATA_OR_SPLIT_INCONSISTENCY: base checkpoint mismatch")
    replay = load_replay_dataset(args.replay)
    if len(replay) != 3000:
        raise RuntimeError("DATA_OR_SPLIT_INCONSISTENCY: replay dataset count mismatch")
    replay_model = load_joint_vae(args.base_checkpoint, "global4", device="cpu")
    replay_architecture_count = len(
        {
            architecture_fingerprint(config_from_igraph(replay_model, graph))
            for graph, _hp in replay
        }
    )
    del replay_model, replay
    stats = json.loads(
        (unified_root / "unified_76_configuration_statistics.json").read_text()
    )
    high_noise_ids = sorted(str(row["config_id"]) for row in stats if row["high_noise"])
    label_high_noise_ids = sorted(str(row["config_id"]) for row in labels if row["high_noise"])
    if len(high_noise_ids) != 12 or high_noise_ids != label_high_noise_ids:
        raise RuntimeError("DATA_OR_SPLIT_INCONSISTENCY: high-noise IDs changed")
    full_splits = build_fixed_splits(labels)
    split_reproduction = _split_reproduction(full_splits, unified_root, labels)
    if not split_reproduction["passed"]:
        raise RuntimeError("DATA_OR_SPLIT_INCONSISTENCY: fixed splits do not reproduce")
    reduced_labels = [dict(row) for row in labels if not bool(row["high_noise"])]
    reduced_splits = filter_split_manifest(full_splits, reduced_labels)
    rows_by_id = {str(row["config_id"]): row for row in labels}
    for split in [*full_splits, *reduced_splits]:
        assert_same_architecture_side(split, rows_by_id)
    old_audit = _old_pilot_audit(old_root)
    if not (
        old_audit["planned_task_count"] == 150
        and old_audit["observed_task_count"] == 150
        and old_audit["unique_task_count"] == 150
        and old_audit["all_success"]
    ):
        raise RuntimeError("DATA_OR_SPLIT_INCONSISTENCY: old pilot is incomplete")

    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir(parents=True)
    shutil.copy2(args.test_log, output / "logs" / "preflight_tests.log")
    split_manifest = []
    for dataset_variant, splits in (
        ("full", full_splits),
        ("exclude_high_noise", reduced_splits),
    ):
        split_manifest.extend(
            [{"dataset_variant": dataset_variant, **dict(split)} for split in splits]
        )
    _dump_json(output / "fixed_split_manifest.json", split_manifest)
    _dump_csv(output / "fixed_split_manifest.csv", split_manifest)
    (output / "fixed_split_manifest.sha256").write_text(
        f"{sha256_file(output / 'fixed_split_manifest.json')}  fixed_split_manifest.json\n",
        encoding="utf-8",
    )
    _dump_json(output / "split_reproduction_audit.json", split_reproduction)

    old_folds = json.loads(Path(old_audit["old_split_file"]).read_text())
    old_memberships = {
        frozenset(fold["outer_test_architectures"]) for fold in old_folds
    }
    current_memberships = {
        frozenset(split["validation_architecture_fingerprints"])
        for split in full_splits
    }
    compatible = old_memberships == current_memberships
    baseline_task_count = len(full_splits) * len(DATASET_VARIANTS) * len(TRAINING_SEEDS) * len(BASELINE_MODELS)
    baseline_reuse = {
        "old_pilot": old_audit,
        "same_76_configurations": True,
        "same_60_architecture_fingerprints": True,
        "same_global4_hp_representation": True,
        "base_vae_frozen_for_old_head": True,
        "test_label_used": False,
        "prediction_mappable_to_fixed_splits": True,
        "training_membership_compatible": compatible,
        "reused_task_count": 0 if not compatible else baseline_task_count,
        "rerun_task_count": baseline_task_count if not compatible else 0,
        "incompatible_reason": None
        if compatible
        else "old pilot used five new outer folds; strict experiment uses six fixed time-prefix and fifty fixed grouped splits",
    }
    _dump_json(output / "baseline_reuse_audit.json", baseline_reuse)
    (output / "baseline_reuse_audit.md").write_text(
        "# Baseline reuse audit\n\n"
        f"Compatible: `{compatible}`\n\n"
        f"- Reused tasks: `{baseline_reuse['reused_task_count']}`\n"
        f"- Rerun tasks: `{baseline_reuse['rerun_task_count']}`\n"
        f"- Reason: `{baseline_reuse['incompatible_reason']}`\n",
        encoding="utf-8",
    )

    specified_unified_files = [
        "unified_training_data_no_test.json",
        "unified_76_configuration_statistics.json",
        "unified_60_architecture_group_statistics.json",
        "representation_comparison.json",
        "representation_metrics.csv",
        "representation_predictions.csv",
        "representation_split_audit.json",
        "representation_split_audit.csv",
        "vae_pilot_readiness.json",
        "final_goal_report.md",
        "result_audit.md",
        "unified_seed_level_audit.jsonl",
        "high_noise_configurations.json",
    ]
    input_paths = [unified_root / name for name in specified_unified_files]
    input_paths.extend(
        [
            args.base_checkpoint.resolve(),
            args.replay.resolve(),
            old_root / "task_plan.json",
            old_root / "merged_training_results.jsonl",
            old_root / "heldout_predictions.csv",
            old_root / "final_goal_report.md",
        ]
    )
    code_paths = [path.resolve() for path in args.code_paths]
    protected_paths = [
        args.base_checkpoint.resolve(),
        (REPO_ROOT / "bo_phase4.py").resolve(),
        (REPO_ROOT / "eval_utils.py").resolve(),
    ]
    git_status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    input_audit = {
        "classification_on_failure": "DATA_OR_SPLIT_INCONSISTENCY",
        "passed": True,
        "configuration_count": len(labels),
        "independent_architecture_count": len(architectures),
        "seed_level_audit_count": seed_audit_count,
        "label_provenance_recomputation": label_provenance,
        "group_leakage_count": 0,
        "test_label_used": False,
        "vae_finetune_ready": True,
        "high_noise_configuration_count": len(high_noise_ids),
        "high_noise_config_ids": high_noise_ids,
        "replay_record_count": 3000,
        "replay_independent_architecture_count": replay_architecture_count,
        "exclude_high_noise_configuration_count": len(reduced_labels),
        "exclude_high_noise_architecture_count": len(
            {row["architecture_fingerprint"] for row in reduced_labels}
        ),
        "architecture_fingerprints": sorted(architectures),
        "hp_fingerprints": sorted({str(row["hp_fingerprint"]) for row in labels}),
        "validation_means_sha256": stable_hash(
            *[(row["config_id"], row["validation_mean"]) for row in labels]
        ),
        "label_uncertainty_sha256": stable_hash(
            *[
                (row["config_id"], row["validation_ci95_half_width"])
                for row in labels
            ]
        ),
        "input_hashes": _input_hashes(input_paths),
        "protected_hashes_before": _input_hashes(protected_paths),
        "old_pilot": old_audit,
        "git_status_before": git_status,
        "generated_at": utc_now(),
    }
    _dump_json(output / "input_audit.json", input_audit)
    (output / "input_audit.md").write_text(
        "# Strict experiment input audit\n\nResult: `PASS`\n\n"
        f"- Configurations: `{len(labels)}`\n"
        f"- Architectures: `{len(architectures)}`\n"
        f"- Seed-level rows: `{seed_audit_count}`\n"
        f"- Fixed splits: `{len(full_splits)}`\n"
        f"- High-noise IDs: `{len(high_noise_ids)}`\n"
        "- Test label used: `false`\n"
        "- Formal BO run: `false`\n",
        encoding="utf-8",
    )
    _write_sha256_sidecar(output / "input_audit.json")
    _write_sha256_sidecar(output / "baseline_reuse_audit.json")
    (output / "base_checkpoint.sha256").write_text(
        f"{sha256_file(args.base_checkpoint)}  {args.base_checkpoint.resolve()}\n",
        encoding="utf-8",
    )
    (output / "formal_bo_precheck.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path}\n"
            for path in protected_paths[1:]
        ),
        encoding="utf-8",
    )
    splits_by_variant = {"full": full_splits, "exclude_high_noise": reduced_splits}
    tasks = _make_tasks(splits_by_variant)
    gpu_rows = []
    if torch.cuda.is_available():
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            index, name, uuid = [piece.strip() for piece in line.split(",", 2)]
            gpu_rows.append({"index": int(index), "name": name, "uuid": uuid})
    plan = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "diagnostic_only": True,
        "formal_bo_allowed": False,
        "gnn_training_allowed": False,
        "test_labels_allowed": False,
        "inputs": {
            "labels": str(labels_path.resolve()),
            "replay": str(args.replay.resolve()),
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "unified_root": str(unified_root),
            "old_pilot_root": str(old_root),
        },
        "input_hashes": _input_hashes(input_paths),
        "code_hashes": _input_hashes(code_paths),
        "protected_hashes": _input_hashes(protected_paths),
        "fixed_split_manifest": str((output / "fixed_split_manifest.json").resolve()),
        "fixed_split_manifest_sha256": sha256_file(output / "fixed_split_manifest.json"),
        "split_count_per_dataset": len(full_splits),
        "dataset_variants": list(DATASET_VARIANTS),
        "training_seeds": list(TRAINING_SEEDS),
        "lambdas": list(LAMBDAS),
        "baseline_models": list(BASELINE_MODELS),
        "loss": {
            "formula": "(1-lambda)*(node_edge_reconstruction + 0.005*KL) + lambda*MSE",
            "mse_sign": "positive_minimized",
            "ranking_weight": 0.0,
            "contrastive_weight": 0.0,
            "pseudo_label_weight": 0.0,
            "icnn_weight": 0.0,
        },
        "training": {
            "maximum_epochs": 24,
            "early_stopping_patience": 7,
            "early_stopping_fraction_by_architecture": 0.20,
            "final_fit": "reload base checkpoint and train all fixed-split training rows for selected epoch count",
            "head_learning_rate": 0.003,
            "vae_learning_rate": 0.00002,
            "reconstruction_batch_architectures": 24,
            "beta_kl": 0.005,
            "gradient_clip": 5.0,
            "performance_batch_unit": "architecture_plus_hp_configuration",
            "reconstruction_batch_unit": "unique_architecture",
        },
        "fixed_gp": {
            "kernel": "RBF(length_scale=1.0, fixed)",
            "observation_alpha": 1e-4,
            "feature_standardization": "training split only",
            "label_standardization": "training split only",
            "optimizer": None,
            "random_seed": 20260821,
            "historical_estimator_mismatch_recorded": True,
        },
        "bootstrap": {
            "unit": "architecture_fingerprint",
            "replicates": 2000,
            "seed": 20260821,
            "shared_indices": True,
        },
        "reliable_pair": {
            "minimum_validation_gap": 0.006,
            "confidence_intervals_disjoint": True,
            "advancement_minimum_pairs": 20,
            "top_minimum_architectures": 8,
        },
        "reconstruction_sample_indices": json.loads(
            (old_root / "task_plan.json").read_text()
        )["fixed_reconstruction_sample_indices"],
        "gpus_at_plan_time": gpu_rows,
        "num_shards": 2,
        "maximum_retries": MAXIMUM_RETRIES,
        "tasks": tasks,
        "task_count": len(tasks),
        "task_counts": {
            kind: sum(task["task_kind"] == kind for task in tasks)
            for kind in TASK_KINDS
        },
        "shard_task_counts": {
            str(index): sum(task["assigned_shard"] == index for task in tasks)
            for index in range(2)
        },
    }
    _dump_json(output / "task_plan.json", plan)
    _write_sha256_sidecar(output / "task_plan.json")
    environment = {
        "generated_at": utc_now(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpus": gpu_rows,
        "git_status": git_status,
    }
    _dump_json(output / "run_environment.json", environment)
    return plan


class StrictPerformanceHead(nn.Module):
    """The legacy 32-unit predictor with variable explicit input features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("predictor features must be a matrix")
        return self.network(features).squeeze(1)


def _scaling(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hp = np.asarray([row["raw_hp_vector"] for row in rows], dtype=np.float64)
    y = np.asarray([row["validation_mean"] for row in rows], dtype=np.float64)
    hp_std = hp.std(axis=0)
    hp_std[hp_std < 1e-8] = 1.0
    y_std = float(y.std())
    return {
        "hp_mean": hp.mean(axis=0).tolist(),
        "hp_std": hp_std.tolist(),
        "y_mean": float(y.mean()),
        "y_std": y_std if y_std >= 1e-8 else 1.0,
    }


def _graphs(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    return [config_to_igraph(row) for row in rows]


def _features(
    model: nn.Module,
    rows: Sequence[Mapping[str, Any]],
    scaling: Mapping[str, Any],
    model_name: str,
    device: torch.device,
    *,
    grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    graphs = _graphs(rows)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        mu, _ = model.arch_vae.encode(graphs)
    hp = np.asarray([row["raw_hp_vector"] for row in rows], dtype=np.float32)
    hp = (hp - np.asarray(scaling["hp_mean"], dtype=np.float32)) / np.asarray(
        scaling["hp_std"], dtype=np.float32
    )
    parts = [mu, torch.tensor(hp, dtype=torch.float32, device=device)]
    if model_name == "frozen_hybrid_head":
        structure = np.stack(
            [structure_feature_vector(row, [1.0] * 4)[0] for row in rows]
        ).astype(np.float32)
        parts.insert(1, torch.tensor(structure, dtype=torch.float32, device=device))
    return torch.cat(parts, dim=1), mu


def _targets(
    rows: Sequence[Mapping[str, Any]], scaling: Mapping[str, Any], device: torch.device
) -> torch.Tensor:
    values = (
        np.asarray([row["validation_mean"] for row in rows], dtype=np.float32)
        - float(scaling["y_mean"])
    ) / float(scaling["y_std"])
    return torch.tensor(values, dtype=torch.float32, device=device)


def _architecture_inner_split(
    rows: Sequence[Mapping[str, Any]], task_id: str, fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    architectures = sorted(
        {str(row["architecture_fingerprint"]) for row in rows},
        key=lambda item: stable_hash("strict-inner", task_id, item),
    )
    count = max(2, min(len(architectures) - 2, round(len(architectures) * fraction)))
    validation = set(architectures[:count])
    fit = [dict(row) for row in rows if row["architecture_fingerprint"] not in validation]
    stop = [dict(row) for row in rows if row["architecture_fingerprint"] in validation]
    if {row["architecture_fingerprint"] for row in fit} & {
        row["architecture_fingerprint"] for row in stop
    }:
        raise AssertionError("inner architecture leakage")
    return fit, stop


def _unique_replay_graphs(
    model: nn.Module, replay: Sequence[tuple[Any, torch.Tensor]]
) -> list[Any]:
    unique: dict[str, Any] = {}
    for graph, _hp in replay:
        fingerprint = architecture_fingerprint(config_from_igraph(model, graph))
        unique.setdefault(fingerprint, graph.copy())
    return [unique[key] for key in sorted(unique)]


def _parameter_groups(model: nn.Module) -> tuple[list[str], list[str]]:
    encoder_prefixes = (
        "grue_forward.",
        "grue_backward.",
        "fc1.",
        "fc2.",
        "hg_unify.",
    )
    decoder_prefixes = ("grud.", "fc3.", "add_vertex.", "add_edge.")
    names = [name for name, _ in model.arch_vae.named_parameters()]
    encoder = [name for name in names if name.startswith(encoder_prefixes)]
    decoder = [name for name in names if name.startswith(decoder_prefixes)]
    if not encoder or not decoder or set(encoder) & set(decoder):
        raise RuntimeError("could not identify disjoint encoder/decoder parameter groups")
    return encoder, decoder


def _prediction_values(
    model: nn.Module,
    head: StrictPerformanceHead,
    rows: Sequence[Mapping[str, Any]],
    scaling: Mapping[str, Any],
    model_name: str,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    head.eval()
    features, _ = _features(
        model, rows, scaling, model_name, device, grad=False
    )
    with torch.no_grad():
        values = head(features).cpu().numpy()
    return values * float(scaling["y_std"]) + float(scaling["y_mean"])


def _train_phase(
    *,
    task: Mapping[str, Any],
    plan: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    replay: Sequence[tuple[Any, torch.Tensor]],
    device: torch.device,
    epochs: int,
    phase: str,
    stop_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[
    nn.Module,
    StrictPerformanceHead,
    dict[str, Any],
    list[dict[str, Any]],
    float | None,
    dict[str, Any],
]:
    phase_seed = int(stable_hash(task["task_id"], phase, task["training_seed"])[:8], 16)
    set_all_seeds(phase_seed)
    model = load_joint_vae(plan["inputs"]["base_checkpoint"], "global4", device=device)
    strict_ft = task["task_kind"] == "strict_ft"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if strict_ft:
        for parameter in model.arch_vae.parameters():
            parameter.requires_grad_(True)
    model_name = str(task["model"])
    scaling = _scaling(train_rows)
    probe_features, _ = _features(
        model, train_rows[:1], scaling, model_name, device, grad=False
    )
    head = StrictPerformanceHead(probe_features.shape[1]).to(device)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    optimizer_groups = [{"params": head.parameters(), "lr": plan["training"]["head_learning_rate"]}]
    if strict_ft:
        optimizer_groups.append(
            {
                "params": model.arch_vae.parameters(),
                "lr": plan["training"]["vae_learning_rate"],
            }
        )
    optimizer = torch.optim.Adam(optimizer_groups)
    base_arch_state = {
        name: value.detach().cpu().clone()
        for name, value in model.arch_vae.state_dict().items()
    }
    head_initial_state = {
        name: value.detach().cpu().clone() for name, value in head.state_dict().items()
    }
    encoder_names, decoder_names = _parameter_groups(model)
    named_parameters = dict(model.arch_vae.named_parameters())
    encoder_parameters = [named_parameters[name] for name in encoder_names]
    decoder_parameters = [named_parameters[name] for name in decoder_names]
    unique_replay = _unique_replay_graphs(model, replay) if strict_ft else []
    trace: list[dict[str, Any]] = []
    best_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    no_improvement = 0
    calibration_std: float | None = None
    for epoch in range(1, int(epochs) + 1):
        model.arch_vae.train(strict_ft)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        features, _mu = _features(
            model, train_rows, scaling, model_name, device, grad=strict_ft
        )
        targets = _targets(train_rows, scaling, device)
        predictions = head(features)
        mse = torch.mean((predictions - targets) ** 2)
        reconstruction = torch.zeros((), device=device)
        kl = torch.zeros((), device=device)
        vae = torch.zeros((), device=device)
        encoder_performance_gradient = 0.0
        decoder_reconstruction_gradient = 0.0
        if strict_ft:
            batch_size = min(
                int(plan["training"]["reconstruction_batch_architectures"]),
                len(unique_replay),
            )
            indices = sorted(
                range(len(unique_replay)),
                key=lambda index: stable_hash(task["task_id"], phase, epoch, index),
            )[:batch_size]
            reconstruction_graphs = [unique_replay[index].copy() for index in indices]
            mu_replay, logvar_replay = model.arch_vae.encode(reconstruction_graphs)
            _combined, reconstruction_sum, kl_sum = model.arch_vae.loss(
                mu_replay,
                logvar_replay,
                reconstruction_graphs,
                beta=float(plan["training"]["beta_kl"]),
            )
            reconstruction = reconstruction_sum / batch_size
            kl = kl_sum / batch_size
            vae = reconstruction + float(plan["training"]["beta_kl"]) * kl
            encoder_performance_gradient = gradient_norm(
                mse, encoder_parameters, retain_graph=True
            )
            decoder_reconstruction_gradient = gradient_norm(
                vae, decoder_parameters, retain_graph=True
            )
            loss = convex_combination_loss(vae, mse, validate_lambda(task["lambda"]))
        else:
            loss = mse
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite strict training loss")
        loss.backward()
        parameters = [*head.parameters()]
        if strict_ft:
            parameters.extend(model.arch_vae.parameters())
        nn.utils.clip_grad_norm_(parameters, float(plan["training"]["gradient_clip"]))
        optimizer.step()
        row = {
            "epoch": epoch,
            "phase": phase,
            "raw_reconstruction_loss": float(reconstruction.detach().cpu()),
            "raw_kl_loss": float(kl.detach().cpu()),
            "raw_mse": float(mse.detach().cpu()),
            "weighted_reconstruction_and_kl": float(
                ((1.0 - float(task["lambda"])) * vae).detach().cpu()
            )
            if strict_ft
            else 0.0,
            "weighted_mse": float((float(task["lambda"]) * mse).detach().cpu())
            if strict_ft
            else float(mse.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
            "encoder_performance_gradient_norm": encoder_performance_gradient,
            "decoder_reconstruction_gradient_norm": decoder_reconstruction_gradient,
            "ranking_loss": 0.0,
            "contrastive_loss": 0.0,
            "pseudo_label_loss": 0.0,
            "icnn_loss": 0.0,
        }
        trace.append(row)
        if stop_rows is not None:
            stop_predictions = _prediction_values(
                model, head, stop_rows, scaling, model_name, device
            )
            stop_actual = np.asarray(
                [item["validation_mean"] for item in stop_rows], dtype=np.float64
            )
            stop_rmse = float(np.sqrt(np.mean((stop_predictions - stop_actual) ** 2)))
            row["inner_stop_rmse"] = stop_rmse
            if stop_rmse < best_rmse - 1e-10:
                best_rmse = stop_rmse
                best_epoch = epoch
                no_improvement = 0
                best_state = {
                    "arch": {
                        name: value.detach().cpu().clone()
                        for name, value in model.arch_vae.state_dict().items()
                    },
                    "head": {
                        name: value.detach().cpu().clone()
                        for name, value in head.state_dict().items()
                    },
                }
                calibration_std = max(
                    float(np.sqrt(np.mean((stop_predictions - stop_actual) ** 2))),
                    1e-6,
                )
            else:
                no_improvement += 1
                if no_improvement >= int(plan["training"]["early_stopping_patience"]):
                    break
    if stop_rows is not None:
        if best_state is None:
            raise RuntimeError("early-stopping phase produced no state")
        model.arch_vae.load_state_dict(best_state["arch"])
        head.load_state_dict(best_state["head"])
    else:
        best_epoch = int(epochs)
    after_arch_state = {
        name: value.detach().cpu().clone()
        for name, value in model.arch_vae.state_dict().items()
    }
    after_head_state = {
        name: value.detach().cpu().clone() for name, value in head.state_dict().items()
    }
    diagnostics = {
        "selected_epoch": best_epoch,
        "calibration_std": calibration_std,
        "encoder_parameter_delta_norm": parameter_delta_norm(
            base_arch_state, after_arch_state, encoder_names
        ),
        "decoder_parameter_delta_norm": parameter_delta_norm(
            base_arch_state, after_arch_state, decoder_names
        ),
        "all_arch_vae_parameter_delta_norm": parameter_delta_norm(
            base_arch_state, after_arch_state
        ),
        "predictor_parameter_delta_norm": parameter_delta_norm(
            head_initial_state, after_head_state
        ),
        "arch_state_sha256_before": module_state_sha256(base_arch_state),
        "arch_state_sha256_after": module_state_sha256(after_arch_state),
        "head_state_sha256_before": module_state_sha256(head_initial_state),
        "head_state_sha256_after": module_state_sha256(after_head_state),
        "encoder_requires_grad": all(parameter.requires_grad for parameter in encoder_parameters),
        "decoder_requires_grad": all(parameter.requires_grad for parameter in decoder_parameters),
        "predictor_requires_grad": all(parameter.requires_grad for parameter in head.parameters()),
        "ranking_loss_weight": 0.0,
    }
    return model, head, scaling, trace, calibration_std, diagnostics


def train_task(
    task: Mapping[str, Any],
    plan: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    replay: Sequence[tuple[Any, torch.Tensor]],
    output: Path,
    device: torch.device,
    *,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    split = next(
        row
        for row in manifest
        if row["dataset_variant"] == task["dataset_variant"]
        and row["protocol"] == task["protocol"]
        and row["split_id"] == task["split_id"]
    )
    label_lookup = {str(row["config_id"]): row for row in labels}
    train_rows = [label_lookup[item] for item in split["train_config_ids"]]
    validation_rows = [label_lookup[item] for item in split["validation_config_ids"]]
    if task["dataset_variant"] == "exclude_high_noise":
        if any(row["high_noise"] for row in [*train_rows, *validation_rows]):
            raise AssertionError("high-noise sensitivity task retained excluded row")
    inner_fit, inner_stop = _architecture_inner_split(
        train_rows,
        str(task["task_id"]),
        float(plan["training"]["early_stopping_fraction_by_architecture"]),
    )
    maximum_epochs = int(smoke_epochs or plan["training"]["maximum_epochs"])
    (
        _selection_model,
        _selection_head,
        _selection_scaling,
        selection_trace,
        calibration,
        _selection_diagnostics,
    ) = _train_phase(
        task=task,
        plan=plan,
        train_rows=inner_fit,
        replay=replay,
        device=device,
        epochs=maximum_epochs,
        phase="epoch_selection",
        stop_rows=inner_stop,
    )
    selected_epoch = max(1, int(selection_trace[-1].get("epoch", 1)))
    best_rows = [row for row in selection_trace if row.get("inner_stop_rmse") is not None]
    if best_rows:
        selected_epoch = int(min(best_rows, key=lambda row: row["inner_stop_rmse"])["epoch"])
    model, head, scaling, final_trace, _unused, final_diagnostics = _train_phase(
        task=task,
        plan=plan,
        train_rows=train_rows,
        replay=replay,
        device=device,
        epochs=selected_epoch,
        phase="final_all_training_rows",
        stop_rows=None,
    )
    predictions = _prediction_values(
        model, head, validation_rows, scaling, str(task["model"]), device
    )
    if calibration is None:
        calibration = max(
            float(
                np.sqrt(
                    np.mean(
                        (
                            _prediction_values(
                                model,
                                head,
                                inner_stop,
                                scaling,
                                str(task["model"]),
                                device,
                            )
                            - np.asarray(
                                [row["validation_mean"] for row in inner_stop]
                            )
                        )
                        ** 2
                    )
                )
            ),
            1e-6,
        )
    task_path_parts = [
        task["dataset_variant"],
        task["protocol"],
        task["split_id"],
        f"seed_{task['training_seed']}",
    ]
    if task["task_kind"] == "strict_ft":
        task_path_parts.append(f"lambda_{task['lambda']:g}")
    else:
        task_path_parts.append(str(task["model"]))
    checkpoint = output / "checkpoints" / Path(*task_path_parts) / "checkpoint.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    arch_state = {
        name: value.detach().cpu() for name, value in model.arch_vae.state_dict().items()
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "task": dict(task),
        "base_checkpoint_sha256": sha256_file(plan["inputs"]["base_checkpoint"]),
        "arch_vae_state": arch_state if task["task_kind"] == "strict_ft" else None,
        "head_state": {name: value.detach().cpu() for name, value in head.state_dict().items()},
        "head_input_dim": int(head.network[0].in_features),
        "scaling": scaling,
        "selected_epoch": selected_epoch,
        "calibration_std": calibration,
    }
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    base_model = load_joint_vae(plan["inputs"]["base_checkpoint"], "global4", device="cpu")
    base_state = {name: value.detach().cpu() for name, value in base_model.arch_vae.state_dict().items()}
    encoder_names, decoder_names = _parameter_groups(base_model)
    diagnostics = {
        "encoder_parameter_delta_norm": parameter_delta_norm(base_state, arch_state, encoder_names),
        "decoder_parameter_delta_norm": parameter_delta_norm(base_state, arch_state, decoder_names),
        "predictor_parameter_delta_norm": final_diagnostics[
            "predictor_parameter_delta_norm"
        ],
        "encoder_parameter_hash_before": module_state_sha256(
            {name: base_state[name] for name in encoder_names}
        ),
        "encoder_parameter_hash_after": module_state_sha256(
            {name: arch_state[name] for name in encoder_names}
        ),
        "decoder_parameter_hash_before": module_state_sha256(
            {name: base_state[name] for name in decoder_names}
        ),
        "decoder_parameter_hash_after": module_state_sha256(
            {name: arch_state[name] for name in decoder_names}
        ),
        "encoder_performance_gradient_norm_max": max(
            (row["encoder_performance_gradient_norm"] for row in final_trace), default=0.0
        ),
        "decoder_reconstruction_gradient_norm_max": max(
            (row["decoder_reconstruction_gradient_norm"] for row in final_trace), default=0.0
        ),
        "encoder_requires_grad": task["task_kind"] == "strict_ft",
        "decoder_requires_grad": task["task_kind"] == "strict_ft",
        "predictor_requires_grad": True,
        "all_arch_vae_parameter_delta_norm": final_diagnostics[
            "all_arch_vae_parameter_delta_norm"
        ],
    }
    del base_model
    return {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_epoch": selected_epoch,
        "calibration_std": calibration,
        "scaling": scaling,
        "module_diagnostics": diagnostics,
        "training_trace": final_trace,
        "epoch_selection_trace": selection_trace,
        "train_observation_count": len(train_rows),
        "train_architecture_count": len(
            {row["architecture_fingerprint"] for row in train_rows}
        ),
        "validation_observation_count": len(validation_rows),
        "validation_architecture_count": len(
            {row["architecture_fingerprint"] for row in validation_rows}
        ),
        "inner_fit_architecture_count": len(
            {row["architecture_fingerprint"] for row in inner_fit}
        ),
        "inner_stop_architecture_count": len(
            {row["architecture_fingerprint"] for row in inner_stop}
        ),
        "validation_predictions": [
            {
                "config_id": row["config_id"],
                "architecture_fingerprint": row["architecture_fingerprint"],
                "validation_mean": row["validation_mean"],
                "validation_ci95_low": row["validation_ci95_low"],
                "validation_ci95_high": row["validation_ci95_high"],
                "time_index": row["time_index"],
                "prediction": float(value),
                "prediction_interval_low": float(value - 1.96 * calibration),
                "prediction_interval_high": float(value + 1.96 * calibration),
            }
            for row, value in zip(validation_rows, predictions)
        ],
    }


def _load_task_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_result_path(output: Path, task_id: str) -> Path:
    return output / RESULTS_DIRNAME / f"{task_id}.json"


def _progress(output: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = failed = retries = 0
    for task in tasks:
        path = _task_result_path(output, str(task["task_id"]))
        if not path.is_file():
            continue
        row = _load_task_result(path)
        if row.get("terminal_status") == "success":
            completed += 1
        elif row.get("terminal_status") == "failed":
            failed += 1
        retries += int(row.get("retry_count", 0))
    return {
        "updated_at": utc_now(),
        "expected": len(tasks),
        "completed_success": completed,
        "completed_failed": failed,
        "remaining": len(tasks) - completed - failed,
        "retries": retries,
    }


def run_shard(args: argparse.Namespace) -> int:
    plan_path = args.task_plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    verify_frozen_plan(plan_path, plan)
    if args.num_shards != int(plan["num_shards"]):
        raise ValueError("runtime shard count differs from frozen plan")
    tasks = [
        task
        for task in plan["tasks"]
        if int(task["assigned_shard"]) == int(args.shard_index)
    ]
    if args.max_tasks is not None:
        tasks = tasks[: int(args.max_tasks)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dry_report = {
        "dry_run": bool(args.dry_run),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "task_count": len(tasks),
        "task_ids": [task["task_id"] for task in tasks],
        "models_loaded": False,
        "training_started": False,
    }
    if args.dry_run:
        report_path = output / "dry_run_report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = {
                "format_version": FORMAT_VERSION,
                "task_plan": str(plan_path),
                "task_plan_sha256": sha256_file(plan_path),
                "models_loaded": False,
                "training_started": False,
                "shards": {},
            }
        report["shards"][str(args.shard_index)] = dry_report
        report["total_dry_run_tasks"] = sum(
            int(row["task_count"]) for row in report["shards"].values()
        )
        _dump_json(report_path, report)
        print(json.dumps(dry_report, indent=2, sort_keys=True))
        return 0
    labels = load_validation_labels(Path(plan["inputs"]["labels"]))
    replay = load_replay_dataset(Path(plan["inputs"]["replay"]))
    manifest = json.loads(Path(plan["fixed_split_manifest"]).read_text())
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    gpu = {
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    for task in tasks:
        result_path = _task_result_path(output, str(task["task_id"]))
        if result_path.is_file():
            existing = _load_task_result(result_path)
            if existing.get("terminal_status") == "success":
                drift = {
                    key: (existing.get(key), value)
                    for key, value in task.items()
                    if existing.get(key) != value
                }
                if drift:
                    raise RuntimeError(
                        f"successful resume task metadata drift: {task['task_id']} {drift}"
                    )
                continue
            raise RuntimeError(f"failed terminal task requires explicit review: {task['task_id']}")
        failures = []
        started = utc_now()
        monotonic = time.monotonic()
        for attempt in range(1, int(plan["maximum_retries"]) + 2):
            try:
                details = train_task(
                    task,
                    plan,
                    manifest,
                    labels,
                    replay,
                    output,
                    device,
                    smoke_epochs=args.smoke_epochs,
                )
                row = {
                    **dict(task),
                    **details,
                    "terminal_status": "success",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": failures,
                    "started_at": started,
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - monotonic,
                    "actual_gpu": gpu,
                    "task_plan_sha256": sha256_file(plan_path),
                    "code_hashes": plan["code_hashes"],
                }
                _dump_json(result_path, row)
                break
            except Exception as exc:  # retries are recorded, never silent
                failures.append(
                    {
                        "attempt": attempt,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                if attempt > int(plan["maximum_retries"]):
                    _dump_json(
                        result_path,
                        {
                            **dict(task),
                            "terminal_status": "failed",
                            "attempt_count": attempt,
                            "retry_count": attempt - 1,
                            "attempt_failures": failures,
                            "started_at": started,
                            "finished_at": utc_now(),
                            "elapsed_seconds": time.monotonic() - monotonic,
                            "actual_gpu": gpu,
                        },
                    )
                    break
        progress = _progress(output, tasks)
        progress["shard_index"] = args.shard_index
        progress["actual_gpu"] = gpu
        _dump_json(output / f"shard_{args.shard_index}_{PROGRESS_FILENAME}", progress)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--unified_root", type=Path, required=True)
    prepare.add_argument("--old_pilot_root", type=Path, required=True)
    prepare.add_argument("--base_checkpoint", type=Path, required=True)
    prepare.add_argument("--replay", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--code_paths", type=Path, nargs="+", required=True)
    prepare.add_argument("--test_log", type=Path, required=True)
    prepare.add_argument("--test_exit_code", type=int, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--task_plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--shard_index", type=int, choices=(0, 1), required=True)
    run.add_argument("--num_shards", type=int, default=2)
    run.add_argument("--device")
    run.add_argument("--dry_run", action="store_true")
    run.add_argument("--max_tasks", type=int)
    run.add_argument("--smoke_epochs", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        plan = prepare_experiment(args)
        print(json.dumps({"task_count": plan["task_count"], "task_counts": plan["task_counts"]}, indent=2))
        return 0
    return run_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
