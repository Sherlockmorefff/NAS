#!/usr/bin/env python3
"""One-shot validation-only gate for a joint architecture+HP performance metric.

The formal path reads two explicitly frozen first-stage JSONL shards below the
configured source root.  It never reads test-labelled fields, trains candidate
GNNs, invokes a VAE/decoder, or runs an acquisition function.  Heavy numerical
dependencies are imported lazily so protocol tests and ``--dry-run`` remain
CPU-safe and data-free.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse only the already-tested canonical graph implementation and small pure
# utilities from performance metric v2.  None of its residual target, pair
# loss, training, evaluation, or decision code is called here.
from analyse import performance_metric_quick_gate as v2  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/joint_configuration_metric_gate_v1.yaml"
FORMAT_VERSION = "joint_configuration_metric_gate_global4_v1"
REQUIRED_SEEDS = (0, 1, 2)
PAIRED_T_CRITICAL_90_DF2 = 2.919985580353725
ALLOWED_DECISIONS = {
    "INSUFFICIENT_JOINT_CONFIG_DATA",
    "JOINT_METRIC_UNDERPOWERED",
    "JOINT_METRIC_REJECTED",
    "JOINT_METRIC_NOT_JOINT",
    "JOINT_METRIC_FEATURE_ONLY",
    "JOINT_METRIC_LOCALITY_ONLY",
    "JOINT_METRIC_GP_READY",
    "RUN_FAILED",
}
MODEL_NAMES = (
    "B0", "B1", "B2", "B3", "EJ_KNN", "M1", "M2", "M3", "M4",
    "M2_architecture_shuffled", "M2_hp_shuffled",
)
REQUIRED_OUTPUTS = (
    "state.json", "frozen_protocol.json", "data_inventory.json",
    "data_provenance.json", "split_manifest.json", "pair_inventory.json",
    "metric_training_audit.json", "geometry_metrics.json",
    "metrics_per_fold.csv", "oof_predictions.csv",
    "best_of_4_batches.json", "best_of_4_results.json",
    "branch_ablation.json", "shuffle_controls.json", "decision.json",
    "final_report.md", "environment.json", "git_status.txt",
    "git_diff.patch", "manifest.json", "logs/py_compile.log",
    "logs/pytest_focused.log", "logs/synthetic_smoke.log", "logs/run.log",
)
RAW_REQUIRED_FIELDS = {
    "pool_id", "batch_id", "architecture_slot", "architecture_fingerprint",
    "hp_fingerprint", "joint_config_fingerprint", "seed", "operations",
    "edges", "raw_hp_vector", "lr", "dropout", "hidden_dim", "l2",
    "dataset", "dataset_content_fingerprint", "split_fingerprint",
    "graph_transform", "training_mode", "metric", "hp_mode",
    "normalization", "eval_epochs", "patience", "training_entry",
    "val_acc", "valid", "terminal_status",
}
PROTOCOL_IDENTITY_FIELDS = (
    "dataset", "dataset_content_fingerprint", "split_fingerprint",
    "graph_transform", "training_mode", "metric", "hp_mode",
    "normalization", "eval_epochs", "patience", "training_entry",
)


class ProtocolError(RuntimeError):
    """Fail-closed protocol, provenance, or leakage violation."""


class UnderpoweredError(ProtocolError):
    """Raised only for the frozen per-outer-fold pair-power gate."""


class GPProtocolUnavailable(ProtocolError):
    """Required exact fixed-noise GP implementation is unavailable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def dump_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n",
    )


def dump_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def dump_csv(
    path: Path, rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    materialized = list(rows)
    fieldnames = list(fields or sorted({str(key) for row in materialized for key in row}))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in materialized:
                writer.writerow({
                    field: json.dumps(row.get(field), sort_keys=True, separators=(",", ":"))
                    if isinstance(row.get(field), (dict, list, tuple)) else row.get(field)
                    for field in fieldnames
                })
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ProtocolError(message)


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ProtocolError("PyYAML is required to read the frozen protocol") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "config must be a mapping")
    _require(value.get("protocol_version") == FORMAT_VERSION, "config protocol_version mismatch")
    _require(tuple(value["data"]["training_seeds"]) == REQUIRED_SEEDS, "training seeds must be [0,1,2]")
    _require(tuple(value["training"]["metric_model_seeds"]) == REQUIRED_SEEDS, "metric seeds must be [0,1,2]")
    _require(value["data"].get("hp_residualization") == "forbidden", "HP residualization must be forbidden")
    _require(value["data"].get("result_shards") == [
        "phase_b/initial/gpu0/seed_results.jsonl",
        "phase_b/initial/gpu1/seed_results.jsonl",
    ], "formal source shards differ from the frozen first-stage inputs")
    _require(value["data"].get("preregistration") == "preregistration.json", "preregistration path mismatch")
    _require(value["data"].get("artifact_manifest") == "artifact_manifest.json", "artifact manifest path mismatch")
    _require(value["fusion"]["components"] == [
        "architecture_projection", "hp_representation", "elementwise_interaction",
    ], "joint fusion components mismatch")
    _require(value["fusion"]["layers"] == [48, 32, 8], "joint fusion dimensions mismatch")
    _require(value["metric_methods"]["M2"]["standardize_input_dimensions"] is False, "M2 input standardization is forbidden")
    _require(value["metric_methods"]["M2"]["ard"] is False, "M2 must be isotropic")
    _require(value["metric_methods"]["M4"]["may_replace_M2_for_decision"] is False, "M4 cannot replace M2")
    _require(set(value["decision"]["allowed_classifications"]) == ALLOWED_DECISIONS, "classification set mismatch")
    return value


# Canonical graph functions are explicit aliases so tests and provenance can
# identify the exact reused implementation.
normalize_architecture = v2.normalize_architecture
architecture_fingerprint = v2.architecture_fingerprint
hp_fingerprint = v2.hp_fingerprint
explicit_graph_features = v2.explicit_graph_features
node_feature_rows = v2.node_feature_rows
OPERATION_ORDER = v2.OPERATION_ORDER


def nested_test_key_hits(value: Any, path: str = "root") -> list[str]:
    return v2.nested_test_key_hits(value, path)


def hp_feature_vector(row: Mapping[str, Any]) -> list[float]:
    raw = row["raw_hp"]
    lr = float(raw["lr"])
    dropout = float(raw["dropout"])
    hidden = int(raw["hidden_dim"])
    l2 = float(raw["l2"])
    if lr <= 0.0 or l2 <= 0.0 or hidden <= 0 or not 0.0 <= dropout <= 1.0:
        raise ProtocolError("raw HP is outside the frozen transform domain")
    return [math.log10(lr), dropout, math.log2(hidden), math.log10(l2)]


def _read_preregistration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    training = value.get("training")
    if not isinstance(training, Mapping):
        raise ProtocolError(f"missing training preregistration: {path}")
    if training.get("track_test") is not False:
        raise ProtocolError(f"track_test is not explicitly false: {path}")
    forbidden = [
        hit for hit in nested_test_key_hits(training, "root.training")
        if not hit.endswith(".track_test")
    ]
    if forbidden:
        raise ProtocolError(f"unexpected test-named training protocol fields: {forbidden[:5]}")
    return dict(value)


def _protocol_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in PROTOCOL_IDENTITY_FIELDS}


def _read_result_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one explicitly named shard and reject test data at the loader."""
    if not path.is_file():
        raise ProtocolError(f"missing frozen result shard: {path}")
    output: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ProtocolError(f"result row is not a mapping: {path}:{line_number}")
            hits = nested_test_key_hits(row)
            if hits:
                raise ProtocolError(
                    f"test field rejected at loader layer {path}:{line_number}: {hits[:5]}"
                )
            missing = RAW_REQUIRED_FIELDS - set(row)
            if missing:
                raise ProtocolError(f"result schema missing {sorted(missing)} at {path}:{line_number}")
            normalized = dict(row)
            normalized["source_file"] = str(path.resolve())
            normalized["source_line"] = line_number
            output.append(normalized)
    return output


def _validate_source_identity(
    rows: Sequence[Mapping[str, Any]], preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ProtocolError("frozen source shards are empty")
    identities: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = _protocol_identity(row)
        identities[stable_hash(identity)] = identity
    if len(identities) != 1:
        raise ProtocolError("training protocol identity is not unique")
    identity = next(iter(identities.values()))
    training = preregistration["training"]
    checks = {
        "dataset": str(identity["dataset"]).lower() == "cora",
        "dataset_content_fingerprint": identity["dataset_content_fingerprint"] == training.get("dataset_content_fingerprint"),
        "split_fingerprint": identity["split_fingerprint"] == training.get("split_fingerprint"),
        "eval_epochs": int(identity["eval_epochs"]) == int(training.get("max_epochs")),
        "patience": int(identity["patience"]) == int(training.get("patience")),
        "track_test": training.get("track_test") is False,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ProtocolError(f"result/preregistration identity mismatch: {failed}")
    return identity


def load_and_aggregate_source(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load only phase-B initial shards and aggregate exactly seeds 0,1,2."""
    root = (REPO_ROOT / str(config["data"]["source_root"])).resolve()
    prereg_path = root / str(config["data"]["preregistration"])
    artifact_manifest_path = root / str(config["data"]["artifact_manifest"])
    shard_paths = tuple(root / str(relative) for relative in config["data"]["result_shards"])
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    manifest_entries = {
        str(row["relative_path"]): row for row in artifact_manifest.get("files", [])
        if isinstance(row, Mapping) and "relative_path" in row
    }
    for relative, shard_path in zip(config["data"]["result_shards"], shard_paths):
        entry = manifest_entries.get(str(relative))
        if entry is None:
            raise ProtocolError(f"frozen shard is absent from artifact_manifest.json: {relative}")
        if str(entry.get("sha256")) != sha256_file(shard_path):
            raise ProtocolError(f"frozen shard hash differs from artifact manifest: {relative}")
    preregistration = _read_preregistration(prereg_path)
    rows = [row for path in shard_paths for row in _read_result_jsonl(path)]
    identity = _validate_source_identity(rows, preregistration)
    identity_hash = stable_hash(identity)

    unique_records: dict[tuple[str, int, str], dict[str, Any]] = {}
    excluded_non_success = 0
    for raw in rows:
        if str(raw["dataset"]).lower() != "cora":
            raise ProtocolError("non-Cora record in frozen source")
        canonical = normalize_architecture(raw)
        if architecture_fingerprint(canonical) != str(raw["architecture_fingerprint"]):
            raise ProtocolError(f"architecture fingerprint mismatch at {raw['source_file']}:{raw['source_line']}")
        raw_hp_vector = [float(value) for value in raw["raw_hp_vector"]]
        if hp_fingerprint(raw_hp_vector) != str(raw["hp_fingerprint"]):
            raise ProtocolError(f"HP fingerprint mismatch at {raw['source_file']}:{raw['source_line']}")
        expected_joint = stable_hash({
            "architecture_fingerprint": str(raw["architecture_fingerprint"]),
            "hp_fingerprint": str(raw["hp_fingerprint"]),
        })
        if expected_joint != str(raw["joint_config_fingerprint"]):
            raise ProtocolError(f"joint fingerprint mismatch at {raw['source_file']}:{raw['source_line']}")
        if raw.get("terminal_status") != "success" or raw.get("valid") is not True:
            excluded_non_success += 1
            continue
        validation = float(raw["val_acc"])
        if not math.isfinite(validation):
            raise ProtocolError("non-finite validation accuracy")
        seed = int(raw["seed"])
        if seed not in REQUIRED_SEEDS:
            continue
        key = (str(raw["joint_config_fingerprint"]), seed, identity_hash)
        normalized = dict(raw)
        normalized["operations"] = list(canonical["operations"])
        normalized["edges"] = [list(edge) for edge in canonical["edges"]]
        if key in unique_records:
            prior = unique_records[key]
            if float(prior["val_acc"]) != validation:
                raise ProtocolError(f"conflicting duplicate joint/seed/protocol record: {key}")
            raise ProtocolError(f"duplicate joint/seed/protocol record is not uniquely traceable: {key}")
        unique_records[key] = normalized

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in unique_records.values():
        grouped[str(record["joint_config_fingerprint"])].append(record)
    configurations: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for joint_fingerprint, values in sorted(grouped.items()):
        by_seed = {int(row["seed"]): row for row in values}
        if set(by_seed) != set(REQUIRED_SEEDS):
            incomplete.append({
                "joint_fingerprint": joint_fingerprint,
                "observed_seeds": sorted(by_seed),
            })
            continue
        ordered = [by_seed[seed] for seed in REQUIRED_SEEDS]
        invariant_fields = (
            "architecture_fingerprint", "hp_fingerprint", "operations", "edges",
            "raw_hp_vector", "lr", "dropout", "hidden_dim", "l2", "pool_id",
            "batch_id", "architecture_slot",
            "source_file",
        )
        for field in invariant_fields:
            if len({stable_hash(row[field]) for row in ordered}) != 1:
                raise ProtocolError(f"seed-invariant field mismatch for {joint_fingerprint}: {field}")
        validations = [float(row["val_acc"]) for row in ordered]
        mean = sum(validations) / 3.0
        variance = sum((value - mean) ** 2 for value in validations) / 2.0
        sd = math.sqrt(max(variance, 0.0))
        first = ordered[0]
        configurations.append({
            "joint_fingerprint": joint_fingerprint,
            "architecture_fingerprint": str(first["architecture_fingerprint"]),
            "hp_fingerprint": str(first["hp_fingerprint"]),
            "operations": list(first["operations"]),
            "edges": [list(edge) for edge in first["edges"]],
            "lr": float(first["lr"]),
            "dropout": float(first["dropout"]),
            "hidden_dim": int(first["hidden_dim"]),
            "l2": float(first["l2"]),
            "raw_hp": {
                "lr": float(first["lr"]), "dropout": float(first["dropout"]),
                "hidden_dim": int(first["hidden_dim"]), "l2": float(first["l2"]),
            },
            "pool_id": int(first["pool_id"]),
            "batch_id": int(first["batch_id"]),
            "slot_id": int(first["architecture_slot"]),
            "validation_seed_0": validations[0],
            "validation_seed_1": validations[1],
            "validation_seed_2": validations[2],
            "validation_mean": mean,
            "validation_sd": sd,
            "validation_se": sd / math.sqrt(3.0),
            "source_file": str(first["source_file"]),
            "source_lines": [int(row["source_line"]) for row in ordered],
            "training_protocol_identity": identity_hash,
        })

    inventory = {
        "raw_row_count": len(rows),
        "successful_required_seed_row_count": len(unique_records),
        "excluded_non_success_row_count": excluded_non_success,
        "incomplete_joint_configurations": incomplete,
        "unique_joint_configurations": len(configurations),
        "unique_architectures": len({row["architecture_fingerprint"] for row in configurations}),
        "unique_hp_fingerprints": len({row["hp_fingerprint"] for row in configurations}),
        "unique_pools": len({row["pool_id"] for row in configurations}),
        "complete_training_seeds": list(REQUIRED_SEEDS),
        "hp_residualization_used": False,
        "target": "raw_validation_accuracy_seed_mean",
    }
    provenance = {
        "source_root": str(root),
        "recursive_scan_used": False,
        "shards": [
            {"path": str(path), "sha256": sha256_file(path)} for path in shard_paths
        ],
        "preregistration": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
        "artifact_manifest": {
            "path": str(artifact_manifest_path), "sha256": sha256_file(artifact_manifest_path),
            "configured_shards_verified": True,
        },
        "training_protocol_identity": identity,
        "training_protocol_identity_sha256": identity_hash,
        "architecture_canonicalization": "v2 ordered operations plus sorted canonical edges; position-sensitive",
        "test_fields_in_result_rows": 0,
        "test_data_accessed": False,
        "phase_c_stabilization_read": False,
    }
    return configurations, inventory, provenance


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            keep, attach = sorted((left_root, right_root))
            self.parent[attach] = keep


def architecture_pool_components(
    configurations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], str]:
    architecture_nodes = {f"architecture:{row['architecture_fingerprint']}" for row in configurations}
    pool_nodes = {f"pool:{row['pool_id']}" for row in configurations}
    union_find = _UnionFind(architecture_nodes | pool_nodes)
    architecture_pools: dict[str, set[int]] = defaultdict(set)
    for row in configurations:
        architecture = str(row["architecture_fingerprint"])
        pool = int(row["pool_id"])
        architecture_pools[architecture].add(pool)
        union_find.union(f"architecture:{architecture}", f"pool:{pool}")
    crossing = any(len(pools) > 1 for pools in architecture_pools.values())
    mapping: dict[str, str] = {}
    for row in configurations:
        joint = str(row["joint_fingerprint"])
        if crossing:
            architecture_node = f"architecture:{row['architecture_fingerprint']}"
            group = f"component:{union_find.find(architecture_node)}"
        else:
            group = f"pool:{int(row['pool_id'])}"
        prior = mapping.get(joint)
        if prior is not None and prior != group:
            raise ProtocolError("joint configuration maps to multiple split groups")
        mapping[joint] = group
    return mapping, "architecture_pool_connected_component" if crossing else "pool_id"


def build_split_manifest(
    configurations: Sequence[Mapping[str, Any]], fold_count: int = 5,
) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.model_selection import GroupKFold
    except ImportError as error:
        raise ProtocolError("NumPy and scikit-learn GroupKFold are required for outer splits") from error
    if not configurations:
        raise ProtocolError("cannot split an empty configuration set")
    group_by_joint, split_kind = architecture_pool_components(configurations)
    groups = sorted(set(group_by_joint.values()))
    if len(groups) < int(fold_count):
        raise ProtocolError(f"only {len(groups)} architecture-pool groups exist for {fold_count} folds")
    ordered = sorted(configurations, key=lambda row: str(row["joint_fingerprint"]))
    group_vector = np.asarray([group_by_joint[str(row["joint_fingerprint"])] for row in ordered], dtype=object)
    splitter = GroupKFold(n_splits=int(fold_count))
    folds: list[dict[str, Any]] = []
    seen_joint: set[str] = set()
    for fold, (train_indices, heldout_indices) in enumerate(
        splitter.split(np.zeros((len(ordered), 1)), groups=group_vector)
    ):
        train_rows = [ordered[int(index)] for index in train_indices]
        heldout_rows = [ordered[int(index)] for index in heldout_indices]
        train_joint = sorted(str(row["joint_fingerprint"]) for row in train_rows)
        heldout_joint = sorted(str(row["joint_fingerprint"]) for row in heldout_rows)
        train_arch = sorted({str(row["architecture_fingerprint"]) for row in train_rows})
        heldout_arch = sorted({str(row["architecture_fingerprint"]) for row in heldout_rows})
        train_pools = sorted({int(row["pool_id"]) for row in train_rows})
        heldout_pools = sorted({int(row["pool_id"]) for row in heldout_rows})
        if set(train_joint) & set(heldout_joint):
            raise ProtocolError("joint fingerprint leakage across outer fold")
        if set(train_arch) & set(heldout_arch):
            raise ProtocolError("architecture leakage across outer fold")
        if set(train_pools) & set(heldout_pools):
            raise ProtocolError("pool leakage across outer fold")
        if seen_joint & set(heldout_joint):
            raise ProtocolError("joint fingerprint held out more than once")
        seen_joint.update(heldout_joint)
        folds.append({
            "fold": fold,
            "group_field": split_kind,
            "train_groups": sorted(set(str(group_vector[int(index)]) for index in train_indices)),
            "heldout_groups": sorted(set(str(group_vector[int(index)]) for index in heldout_indices)),
            "train_joint_fingerprints": train_joint,
            "heldout_joint_fingerprints": heldout_joint,
            "train_architecture_fingerprints": train_arch,
            "heldout_architecture_fingerprints": heldout_arch,
            "train_pool_ids": train_pools,
            "heldout_pool_ids": heldout_pools,
        })
    all_joint = {str(row["joint_fingerprint"]) for row in ordered}
    if seen_joint != all_joint:
        raise ProtocolError("outer folds do not cover every joint fingerprint exactly once")
    manifest = {
        "format_version": FORMAT_VERSION,
        "fold_count": int(fold_count),
        "split_kind": split_kind,
        "implementation": "architecture-pool components plus sklearn GroupKFold",
        "groups": groups,
        "folds": folds,
    }
    manifest["split_sha256"] = stable_hash(manifest)
    assert_split_integrity(configurations, manifest)
    return manifest


def assert_split_integrity(
    configurations: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any],
) -> None:
    joint_assignment: dict[str, int] = {}
    architecture_assignment: dict[str, int] = {}
    pool_assignment: dict[int, int] = {}
    for fold in manifest["folds"]:
        fold_id = int(fold["fold"])
        for joint in fold["heldout_joint_fingerprints"]:
            if str(joint) in joint_assignment:
                raise ProtocolError("joint fingerprint appears in multiple heldout folds")
            joint_assignment[str(joint)] = fold_id
        for architecture in fold["heldout_architecture_fingerprints"]:
            if str(architecture) in architecture_assignment:
                raise ProtocolError("architecture appears in multiple heldout folds")
            architecture_assignment[str(architecture)] = fold_id
        for pool in fold["heldout_pool_ids"]:
            if int(pool) in pool_assignment:
                raise ProtocolError("pool appears in multiple heldout folds")
            pool_assignment[int(pool)] = fold_id
    for row in configurations:
        joint = str(row["joint_fingerprint"])
        architecture = str(row["architecture_fingerprint"])
        pool = int(row["pool_id"])
        if joint not in joint_assignment or architecture not in architecture_assignment or pool not in pool_assignment:
            raise ProtocolError("configuration is absent from split manifest")
        if len({joint_assignment[joint], architecture_assignment[architecture], pool_assignment[pool]}) != 1:
            raise ProtocolError("configuration identities disagree on outer fold")


def fit_hp_standardizer(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as error:
        raise ProtocolError("NumPy is required for HP standardization") from error
    if not rows:
        raise ProtocolError("cannot fit HP standardizer on an empty outer train")
    values = np.asarray([hp_feature_vector(row) for row in rows], dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return {
        "mean": mean.tolist(), "scale": scale.tolist(),
        "fit_joint_fingerprints": sorted(str(row["joint_fingerprint"]) for row in rows),
        "fit_scope": "outer_train_only",
    }


def transform_hp(rows: Sequence[Mapping[str, Any]], standardizer: Mapping[str, Any]) -> Any:
    import numpy as np
    values = np.asarray([hp_feature_vector(row) for row in rows], dtype=np.float64)
    return (
        values - np.asarray(standardizer["mean"], dtype=np.float64)[None, :]
    ) / np.asarray(standardizer["scale"], dtype=np.float64)[None, :]


def _training_seed_values(row: Mapping[str, Any]) -> list[float]:
    return [float(row[f"validation_seed_{seed}"]) for seed in REQUIRED_SEEDS]


def paired_difference_interval(
    left: Mapping[str, Any], right: Mapping[str, Any], confidence: float = 0.90,
) -> dict[str, float]:
    if float(confidence) != 0.90:
        raise ProtocolError("only the frozen paired 90% t interval is supported")
    differences = [a - b for a, b in zip(_training_seed_values(left), _training_seed_values(right))]
    mean = sum(differences) / 3.0
    variance = sum((value - mean) ** 2 for value in differences) / 2.0
    standard_error = math.sqrt(max(variance, 0.0)) / math.sqrt(3.0)
    radius = PAIRED_T_CRITICAL_90_DF2 * standard_error
    return {
        "mean_difference": mean,
        "standard_error": standard_error,
        "ci_lower": mean - radius,
        "ci_upper": mean + radius,
    }


def classify_pair(
    left: Mapping[str, Any], right: Mapping[str, Any],
    equivalence_margin: float = 0.006, far_margin: float = 0.012,
) -> dict[str, Any]:
    interval = paired_difference_interval(left, right)
    if interval["ci_lower"] >= -float(equivalence_margin) and interval["ci_upper"] <= float(equivalence_margin):
        label = "positive"
    elif interval["ci_upper"] < -float(far_margin) or interval["ci_lower"] > float(far_margin):
        label = "negative"
    else:
        label = "ambiguous"
    return {**interval, "label": label}


def _pairwise_hp_median(rows: Sequence[Mapping[str, Any]], standardizer: Mapping[str, Any]) -> float:
    import numpy as np
    transformed = transform_hp(rows, standardizer)
    distances = [
        float(np.linalg.norm(transformed[left] - transformed[right]))
        for left, right in itertools.combinations(range(len(rows)), 2)
    ]
    if not distances:
        raise ProtocolError("at least two rows are needed for an HP-distance median")
    return float(np.median(np.asarray(distances, dtype=np.float64)))


def construct_statistical_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_joint_fingerprints: set[str],
    hp_standardizer: Mapping[str, Any] | None = None,
    hp_median_fit_rows: Sequence[Mapping[str, Any]] | None = None,
    anchor_joint_fingerprints: set[str] | None = None,
    equivalence_margin: float = 0.006,
    far_margin: float = 0.012,
) -> dict[str, Any]:
    """Build matched-seed pair labels without admitting any held-out label."""
    materialized = [dict(row) for row in rows]
    observed = {str(row["joint_fingerprint"]) for row in materialized}
    if not observed <= set(map(str, allowed_joint_fingerprints)):
        forbidden = sorted(observed - set(map(str, allowed_joint_fingerprints)))
        raise ProtocolError(f"outer-held-out label cannot enter triplet construction: {forbidden[:3]}")
    standardizer = dict(hp_standardizer or fit_hp_standardizer(hp_median_fit_rows or materialized))
    median_rows = list(hp_median_fit_rows or materialized)
    median_ids = {str(row["joint_fingerprint"]) for row in median_rows}
    if not median_ids <= set(map(str, allowed_joint_fingerprints)):
        raise ProtocolError("outer-held-out label cannot enter HP pair threshold")
    hp_median = _pairwise_hp_median(median_rows, standardizer)
    transformed = transform_hp(materialized, standardizer)
    anchors = set(map(str, anchor_joint_fingerprints or observed))
    if not anchors <= observed:
        raise ProtocolError("pair anchor set is not contained in the supplied rows")
    pairs: list[dict[str, Any]] = []
    positive_by_anchor: dict[str, list[str]] = defaultdict(list)
    negative_by_anchor: dict[str, list[str]] = defaultdict(list)
    cross_factor_by_anchor: dict[str, list[str]] = defaultdict(list)
    for left_index, right_index in itertools.combinations(range(len(materialized)), 2):
        left = materialized[left_index]
        right = materialized[right_index]
        left_id = str(left["joint_fingerprint"])
        right_id = str(right["joint_fingerprint"])
        if left_id not in anchors and right_id not in anchors:
            continue
        classification = classify_pair(left, right, equivalence_margin, far_margin)
        hp_distance = float(__import__("numpy").linalg.norm(transformed[left_index] - transformed[right_index]))
        cross_architecture = str(left["architecture_fingerprint"]) != str(right["architecture_fingerprint"])
        cross_factor = cross_architecture and hp_distance > hp_median
        pair = {
            "left": left_id, "right": right_id,
            "label": classification["label"],
            "mean_difference": classification["mean_difference"],
            "paired_standard_error": classification["standard_error"],
            "ci_lower": classification["ci_lower"],
            "ci_upper": classification["ci_upper"],
            "hp_distance": hp_distance,
            "cross_architecture": cross_architecture,
            "cross_factor": cross_factor,
        }
        pairs.append(pair)
        if classification["label"] == "positive":
            positive_by_anchor[left_id].append(right_id)
            positive_by_anchor[right_id].append(left_id)
            if cross_factor:
                cross_factor_by_anchor[left_id].append(right_id)
                cross_factor_by_anchor[right_id].append(left_id)
        elif classification["label"] == "negative":
            negative_by_anchor[left_id].append(right_id)
            negative_by_anchor[right_id].append(left_id)
    eligible_anchors = sorted(
        anchor for anchor in anchors
        if positive_by_anchor.get(anchor) and negative_by_anchor.get(anchor)
    )
    cross_factor_anchors = sorted(
        anchor for anchor in anchors
        if cross_factor_by_anchor.get(anchor) and negative_by_anchor.get(anchor)
    )
    positive_pairs = [pair for pair in pairs if pair["label"] == "positive"]
    negative_pairs = [pair for pair in pairs if pair["label"] == "negative"]
    ambiguous_pairs = [pair for pair in pairs if pair["label"] == "ambiguous"]
    return {
        "rows": materialized,
        "pairs": pairs,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "ambiguous_pairs": ambiguous_pairs,
        "positive_by_anchor": dict(positive_by_anchor),
        "negative_by_anchor": dict(negative_by_anchor),
        "cross_factor_by_anchor": dict(cross_factor_by_anchor),
        "eligible_anchors": eligible_anchors,
        "cross_factor_anchors": cross_factor_anchors,
        "hp_distance_median": hp_median,
        "hp_median_fit_joint_fingerprints": sorted(median_ids),
        "allowed_joint_fingerprints": sorted(allowed_joint_fingerprints),
        "anchor_joint_fingerprints": sorted(anchors),
        "outer_heldout_labels_used": False,
        "ambiguous_pairs_enter_loss": False,
        "counts": {
            "positive_pairs": len(positive_pairs),
            "negative_pairs": len(negative_pairs),
            "ambiguous_pairs": len(ambiguous_pairs),
            "anchors_with_positive_and_negative": len(eligible_anchors),
            "cross_architecture_positive_pairs": sum(bool(pair["cross_architecture"]) for pair in positive_pairs),
            "cross_factor_positive_pairs": sum(bool(pair["cross_factor"]) for pair in positive_pairs),
            "reliable_negative_pairs": len(negative_pairs),
        },
    }


def assert_pair_power(pair_inventory: Mapping[str, Any], minimums: Mapping[str, Any]) -> None:
    counts = pair_inventory["counts"]
    required = {
        "anchors_with_positive_and_negative": int(minimums["anchors_with_positive_and_negative"]),
        "cross_architecture_positive_pairs": int(minimums["cross_architecture_positive_pairs"]),
        "cross_factor_positive_pairs": int(minimums["cross_factor_positive_pairs"]),
        "reliable_negative_pairs": int(minimums["reliable_negative_pairs"]),
    }
    failed = {
        key: {"observed": int(counts[key]), "required": threshold}
        for key, threshold in required.items() if int(counts[key]) < threshold
    }
    if failed:
        raise UnderpoweredError(f"outer-fold reliable pair inventory below frozen minimums: {failed}")


def sample_triplets(
    pair_inventory: Mapping[str, Any], batch_anchors: int, seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(int(seed))
    eligible = list(pair_inventory["eligible_anchors"])
    cross_eligible = list(pair_inventory["cross_factor_anchors"])
    if not eligible or not cross_eligible:
        raise ProtocolError("reliable triplets or cross-factor triplets are unavailable")
    cross_count = int(math.ceil(int(batch_anchors) * 0.50))
    triplets: list[dict[str, str]] = []
    for position in range(int(batch_anchors)):
        require_cross = position < cross_count
        anchor = rng.choice(cross_eligible if require_cross else eligible)
        positive_options = (
            pair_inventory["cross_factor_by_anchor"][anchor]
            if require_cross else pair_inventory["positive_by_anchor"][anchor]
        )
        positive = rng.choice(list(positive_options))
        negative = rng.choice(list(pair_inventory["negative_by_anchor"][anchor]))
        triplets.append({
            "anchor": str(anchor), "positive": str(positive), "negative": str(negative),
            "cross_factor_positive": bool(require_cross),
        })
    if sum(bool(row["cross_factor_positive"]) for row in triplets) / len(triplets) < 0.50:
        raise ProtocolError("cross-factor positives are below 50% of epoch sampling")
    ambiguous_keys = {
        frozenset((str(pair["left"]), str(pair["right"])))
        for pair in pair_inventory["ambiguous_pairs"]
    }
    if any(
        frozenset((row["anchor"], row["positive"])) in ambiguous_keys
        or frozenset((row["anchor"], row["negative"])) in ambiguous_keys
        for row in triplets
    ):
        raise ProtocolError("ambiguous pair entered sampled triplet loss")
    return triplets


def inner_architecture_split(
    outer_train: Sequence[Mapping[str, Any]], seed: int, train_fraction: float = 0.80,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    architectures = sorted({str(row["architecture_fingerprint"]) for row in outer_train})
    rng = random.Random(int(seed))
    rng.shuffle(architectures)
    validation_count = max(1, int(math.ceil((1.0 - float(train_fraction)) * len(architectures))))
    if len(architectures) - validation_count < 2:
        raise ProtocolError("inner architecture split is too small")
    validation_architectures = set(architectures[:validation_count])
    inner_validation = [dict(row) for row in outer_train if str(row["architecture_fingerprint"]) in validation_architectures]
    inner_train = [dict(row) for row in outer_train if str(row["architecture_fingerprint"]) not in validation_architectures]
    if {row["architecture_fingerprint"] for row in inner_train} & {
        row["architecture_fingerprint"] for row in inner_validation
    }:
        raise ProtocolError("inner architecture split leakage")
    return inner_train, inner_validation


def _seed_everything(seed: int, torch: Any | None = None) -> None:
    random.seed(int(seed))
    try:
        import numpy as np
        np.random.seed(int(seed))
    except ImportError:
        np = None
    if torch is not None:
        torch.manual_seed(int(seed))
        torch.use_deterministic_algorithms(True)


def _graph_tensors(rows: Sequence[Mapping[str, Any]], torch: Any) -> list[dict[str, Any]]:
    return v2._metric_graph_tensors(rows, torch)


def _joint_model_classes(torch: Any) -> tuple[Any, Any]:
    nn = torch.nn
    functional = torch.nn.functional
    gin_layer, _unused_v2_model = v2._metric_model_classes(torch)

    class JointBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_one = gin_layer(10, 32)
            self.forward_two = gin_layer(32, 32)
            self.reverse_one = gin_layer(10, 32)
            self.reverse_two = gin_layer(32, 32)
            self.architecture_output = nn.Sequential(nn.Linear(128, 32), nn.ReLU())
            self.architecture_projection = nn.Linear(32, 16)
            self.hp_encoder = nn.Sequential(
                nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(),
            )

        def architecture_representation(self, graph: Mapping[str, Any]) -> Any:
            x = graph["x"]
            edges = graph["edges"]
            forward = functional.relu(self.forward_one(x, edges, False))
            forward = self.forward_two(forward, edges, False)
            reverse = functional.relu(self.reverse_one(x, edges, True))
            reverse = self.reverse_two(reverse, edges, True)
            nodes = torch.cat((forward, reverse), dim=-1)
            pooled = torch.cat((nodes.sum(dim=0), nodes.mean(dim=0)), dim=-1)
            return self.architecture_output(pooled)

        def fusion_components(
            self, graphs: Sequence[Mapping[str, Any]], hp_features: Any,
            branch_mode: str,
        ) -> tuple[Any, Any, Any]:
            architecture = torch.stack(
                [self.architecture_projection(self.architecture_representation(graph)) for graph in graphs],
                dim=0,
            )
            hp = self.hp_encoder(hp_features)
            if branch_mode == "architecture_only":
                hp = torch.zeros_like(hp)
            elif branch_mode == "hp_only":
                architecture = torch.zeros_like(architecture)
            elif branch_mode != "full_joint":
                raise ProtocolError(f"unknown branch mode: {branch_mode}")
            interaction = architecture * hp
            return architecture, hp, interaction

        def joint_input(
            self, graphs: Sequence[Mapping[str, Any]], hp_features: Any,
            branch_mode: str,
        ) -> tuple[Any, tuple[Any, Any, Any]]:
            components = self.fusion_components(graphs, hp_features, branch_mode)
            return torch.cat(components, dim=1), components

    class JointConfigurationEncoder(JointBackbone):
        def __init__(self) -> None:
            super().__init__()
            self.fusion = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 8))

        def forward(
            self, graphs: Sequence[Mapping[str, Any]], hp_features: Any,
            branch_mode: str = "full_joint", return_components: bool = False,
        ) -> Any:
            joint, components = self.joint_input(graphs, hp_features, branch_mode)
            embedding = functional.normalize(self.fusion(joint), p=2, dim=1)
            return (embedding, components) if return_components else embedding

    class DirectJointRegressor(JointBackbone):
        def __init__(self) -> None:
            super().__init__()
            self.regression = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, graphs: Sequence[Mapping[str, Any]], hp_features: Any) -> Any:
            joint, _components = self.joint_input(graphs, hp_features, "full_joint")
            return self.regression(joint).reshape(-1)

    return JointConfigurationEncoder, DirectJointRegressor


def fusion_contract_probe(
    architecture_projection: Any, hp_representation: Any,
) -> Any:
    """Pure 48-D fusion contract used by focused tests."""
    import numpy as np
    architecture = np.asarray(architecture_projection, dtype=np.float64)
    hp = np.asarray(hp_representation, dtype=np.float64)
    if architecture.shape != hp.shape or architecture.shape[-1] != 16:
        raise ProtocolError("fusion inputs must have matching trailing dimension 16")
    return np.concatenate((architecture, hp, architecture * hp), axis=-1)


def _triplet_loss(
    embeddings: Any, row_ids: Sequence[str], triplets: Sequence[Mapping[str, Any]],
    margin: float, torch: Any,
) -> Any:
    lookup = {str(fingerprint): index for index, fingerprint in enumerate(row_ids)}
    try:
        anchors = torch.tensor([lookup[str(row["anchor"])] for row in triplets], dtype=torch.long)
        positives = torch.tensor([lookup[str(row["positive"])] for row in triplets], dtype=torch.long)
        negatives = torch.tensor([lookup[str(row["negative"])] for row in triplets], dtype=torch.long)
    except KeyError as error:
        raise ProtocolError(f"triplet references an unencoded configuration: {error}") from error
    positive_distance = torch.linalg.vector_norm(embeddings[anchors] - embeddings[positives], dim=1)
    negative_distance = torch.linalg.vector_norm(embeddings[anchors] - embeddings[negatives], dim=1)
    return torch.mean(torch.clamp(positive_distance - negative_distance + float(margin), min=0.0))


def _state_tensor_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fresh_full_outer_refit(
    early_model: Any,
    outer_train: Sequence[Mapping[str, Any]],
    best_epoch: int,
    model_factory: Callable[[], Any],
    train_epoch: Callable[[Any, Sequence[Mapping[str, Any]], int], None],
) -> tuple[Any, dict[str, Any]]:
    """Execute the frozen fresh-model/full-data/exact-best-epoch contract."""
    if int(best_epoch) <= 0:
        raise ProtocolError("best epoch must be positive before final refit")
    final_model = model_factory()
    if final_model is early_model:
        raise ProtocolError("final refit did not create a fresh model")
    observed_epochs: list[int] = []
    for epoch in range(1, int(best_epoch) + 1):
        train_epoch(final_model, outer_train, epoch)
        observed_epochs.append(epoch)
    return final_model, {
        "fresh_model_for_full_outer_refit": final_model is not early_model,
        "final_refit_joint_fingerprints": sorted(
            str(row["joint_fingerprint"]) for row in outer_train
        ),
        "final_refit_epochs": len(observed_epochs),
        "observed_epoch_numbers": observed_epochs,
    }


def train_joint_metric_encoder(
    outer_train: Sequence[Mapping[str, Any]],
    outer_heldout: Sequence[Mapping[str, Any]],
    seed: int,
    config: Mapping[str, Any],
    hp_standardizer: Mapping[str, Any],
    full_pair_inventory: Mapping[str, Any],
    *,
    branch_mode: str = "full_joint",
    stage_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ProtocolError("PyTorch and NumPy are required for the joint metric encoder") from error
    train_ids = {str(row["joint_fingerprint"]) for row in outer_train}
    heldout_ids = {str(row["joint_fingerprint"]) for row in outer_heldout}
    if train_ids & heldout_ids:
        raise ProtocolError("metric encoder outer split leakage")
    training = config["training"]
    triplet_config = config["triplets"]
    inner_train, inner_validation = inner_architecture_split(
        outer_train, int(seed), float(config["split"]["inner_train_fraction"]),
    )
    inner_train_ids = {str(row["joint_fingerprint"]) for row in inner_train}
    inner_validation_ids = {str(row["joint_fingerprint"]) for row in inner_validation}
    fit_pairs = construct_statistical_pairs(
        inner_train,
        allowed_joint_fingerprints=train_ids,
        hp_standardizer=hp_standardizer,
        hp_median_fit_rows=inner_train,
        equivalence_margin=float(triplet_config["equivalence_margin"]),
        far_margin=float(triplet_config["far_margin"]),
    )
    # Validation anchors are disjoint architecture groups.  References may be
    # any outer-train configuration, but these labels never enter an optimizer
    # step and the cross-factor HP median remains fit on inner-train only.
    validation_pairs = construct_statistical_pairs(
        outer_train,
        allowed_joint_fingerprints=train_ids,
        hp_standardizer=hp_standardizer,
        hp_median_fit_rows=inner_train,
        anchor_joint_fingerprints=inner_validation_ids,
        equivalence_margin=float(triplet_config["equivalence_margin"]),
        far_margin=float(triplet_config["far_margin"]),
    )
    validation_triplets = sample_triplets(
        validation_pairs, int(training["batch_anchors"]), int(seed) * 1_000_003 + 991,
    )

    _seed_everything(seed, torch)
    model_class, _regressor_class = _joint_model_classes(torch)
    early_model = model_class()
    early_initial_hash = _state_tensor_hash(early_model)
    optimizer = torch.optim.Adam(
        early_model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    fit_graphs = _graph_tensors(inner_train, torch)
    validation_graphs = _graph_tensors(outer_train, torch)
    fit_hp = torch.tensor(transform_hp(inner_train, hp_standardizer), dtype=torch.float32)
    validation_hp = torch.tensor(transform_hp(outer_train, hp_standardizer), dtype=torch.float32)
    fit_row_ids = [str(row["joint_fingerprint"]) for row in inner_train]
    validation_row_ids = [str(row["joint_fingerprint"]) for row in outer_train]
    best_epoch = 0
    best_validation_loss = math.inf
    stale = 0
    epochs_run = 0
    epoch_cross_fractions: list[float] = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        epochs_run = epoch
        triplets = sample_triplets(
            fit_pairs, int(training["batch_anchors"]), int(seed) * 1_000_003 + epoch,
        )
        epoch_cross_fractions.append(
            sum(bool(row["cross_factor_positive"]) for row in triplets) / len(triplets)
        )
        early_model.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = early_model(fit_graphs, fit_hp, branch_mode)
        loss = _triplet_loss(
            embeddings, fit_row_ids, triplets, float(triplet_config["triplet_margin"]), torch,
        )
        if not torch.isfinite(loss):
            raise ProtocolError("non-finite triplet training loss")
        loss.backward()
        optimizer.step()
        early_model.eval()
        with torch.no_grad():
            validation_embeddings = early_model(validation_graphs, validation_hp, branch_mode)
            validation_loss = float(_triplet_loss(
                validation_embeddings, validation_row_ids, validation_triplets,
                float(triplet_config["triplet_margin"]), torch,
            ))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    if best_epoch <= 0:
        raise ProtocolError("inner validation did not select a best epoch")
    if stage_callback is not None:
        stage_callback({
            "stage": "inner_early_stop", "model_identity": id(early_model),
            "fit_joint_fingerprints": sorted(inner_train_ids),
            "validation_joint_fingerprints": sorted(inner_validation_ids),
            "epochs": epochs_run, "best_epoch": best_epoch,
        })

    # Required fresh initialization and full-outer-train refit.
    _seed_everything(seed, torch)
    full_graphs = _graph_tensors(outer_train, torch)
    full_hp = torch.tensor(transform_hp(outer_train, hp_standardizer), dtype=torch.float32)
    full_row_ids = [str(row["joint_fingerprint"]) for row in outer_train]
    refit_holder: dict[str, Any] = {}

    def final_model_factory() -> Any:
        model = model_class()
        refit_holder["initial_hash"] = _state_tensor_hash(model)
        return model

    def refit_epoch(model: Any, rows: Sequence[Mapping[str, Any]], epoch: int) -> None:
        if len(rows) != len(outer_train):
            raise ProtocolError("full outer refit callback received incomplete data")
        if "optimizer" not in refit_holder:
            refit_holder["optimizer"] = torch.optim.Adam(
                model.parameters(), lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
        final_optimizer = refit_holder["optimizer"]
        triplets = sample_triplets(
            full_pair_inventory, int(training["batch_anchors"]),
            int(seed) * 1_000_003 + epoch,
        )
        model.train()
        final_optimizer.zero_grad(set_to_none=True)
        loss = _triplet_loss(
            model(full_graphs, full_hp, branch_mode), full_row_ids, triplets,
            float(triplet_config["triplet_margin"]), torch,
        )
        if not torch.isfinite(loss):
            raise ProtocolError("non-finite full-outer-train triplet loss")
        loss.backward()
        final_optimizer.step()
    final_model, refit_contract = fresh_full_outer_refit(
        early_model, outer_train, best_epoch, final_model_factory, refit_epoch,
    )
    final_initial_hash = str(refit_holder["initial_hash"])
    if stage_callback is not None:
        stage_callback({
            "stage": "full_outer_refit", "model_identity": id(final_model),
            "fit_joint_fingerprints": sorted(train_ids),
            "epochs": best_epoch, "best_epoch": best_epoch,
        })
    final_model.eval()
    heldout_graphs = _graph_tensors(outer_heldout, torch)
    heldout_hp = torch.tensor(transform_hp(outer_heldout, hp_standardizer), dtype=torch.float32)
    with torch.no_grad():
        train_embeddings = final_model(full_graphs, full_hp, branch_mode).cpu().numpy().astype("float64")
        heldout_embeddings = final_model(heldout_graphs, heldout_hp, branch_mode).cpu().numpy().astype("float64")
    if train_embeddings.shape != (len(outer_train), 8) or heldout_embeddings.shape != (len(outer_heldout), 8):
        raise ProtocolError("joint metric embedding shape mismatch")
    if not np.allclose(np.linalg.norm(train_embeddings, axis=1), 1.0, atol=1e-5):
        raise ProtocolError("joint metric train embeddings are not L2 normalized")
    audit = {
        "seed": int(seed), "branch_mode": branch_mode,
        "inner_train_joint_fingerprints": sorted(inner_train_ids),
        "inner_validation_joint_fingerprints": sorted(inner_validation_ids),
        "inner_train_pair_threshold_joint_fingerprints": fit_pairs["hp_median_fit_joint_fingerprints"],
        "inner_validation_used_for_inner_train_pair_threshold": False,
        "outer_heldout_labels_used_for_pairs": False,
        "best_epoch": best_epoch, "early_stop_epochs_run": epochs_run,
        "best_inner_validation_triplet_loss": best_validation_loss,
        "early_model_identity": id(early_model), "final_model_identity": id(final_model),
        "fresh_model_for_full_outer_refit": refit_contract["fresh_model_for_full_outer_refit"],
        "initial_weights_reproduced_from_same_seed": early_initial_hash == final_initial_hash,
        "final_refit_joint_fingerprints": refit_contract["final_refit_joint_fingerprints"],
        "final_refit_used_complete_outer_train": sorted(train_ids) == sorted(full_row_ids),
        "final_refit_epochs": refit_contract["final_refit_epochs"],
        "epoch_cross_factor_positive_fraction_minimum": min(epoch_cross_fractions),
        "metric_loss": "triplet_margin_only",
        "ambiguous_pairs_entered_loss": False,
        "hp_residualization_used": False,
    }
    return train_embeddings, heldout_embeddings, final_model, audit


def train_direct_joint_regressor(
    outer_train: Sequence[Mapping[str, Any]], outer_heldout: Sequence[Mapping[str, Any]],
    seed: int, config: Mapping[str, Any], hp_standardizer: Mapping[str, Any],
    *, stage_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ProtocolError("PyTorch and NumPy are required for the direct joint regressor") from error
    training = config["training"]
    inner_train, inner_validation = inner_architecture_split(
        outer_train, int(seed), float(config["split"]["inner_train_fraction"]),
    )
    _seed_everything(seed, torch)
    _metric_class, regressor_class = _joint_model_classes(torch)
    early_model = regressor_class()
    optimizer = torch.optim.Adam(
        early_model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    fit_graphs = _graph_tensors(inner_train, torch)
    validation_graphs = _graph_tensors(inner_validation, torch)
    fit_hp = torch.tensor(transform_hp(inner_train, hp_standardizer), dtype=torch.float32)
    validation_hp = torch.tensor(transform_hp(inner_validation, hp_standardizer), dtype=torch.float32)
    fit_y = torch.tensor([float(row["validation_mean"]) for row in inner_train], dtype=torch.float32)
    validation_y = torch.tensor([float(row["validation_mean"]) for row in inner_validation], dtype=torch.float32)
    best_epoch = 0
    best_validation_loss = math.inf
    stale = 0
    epochs_run = 0
    for epoch in range(1, int(training["max_epochs"]) + 1):
        epochs_run = epoch
        early_model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.huber_loss(early_model(fit_graphs, fit_hp), fit_y)
        if not torch.isfinite(loss):
            raise ProtocolError("non-finite direct-regressor Huber loss")
        loss.backward()
        optimizer.step()
        early_model.eval()
        with torch.no_grad():
            validation_loss = float(torch.nn.functional.huber_loss(
                early_model(validation_graphs, validation_hp), validation_y,
            ))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    if best_epoch <= 0:
        raise ProtocolError("direct regressor early stopping selected no epoch")
    if stage_callback is not None:
        stage_callback({"stage": "direct_inner_early_stop", "model_identity": id(early_model), "best_epoch": best_epoch})
    _seed_everything(seed, torch)
    final_model = regressor_class()
    if id(final_model) == id(early_model):
        raise ProtocolError("direct regressor final refit model is not fresh")
    final_optimizer = torch.optim.Adam(
        final_model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    full_graphs = _graph_tensors(outer_train, torch)
    full_hp = torch.tensor(transform_hp(outer_train, hp_standardizer), dtype=torch.float32)
    full_y = torch.tensor([float(row["validation_mean"]) for row in outer_train], dtype=torch.float32)
    for _epoch in range(1, best_epoch + 1):
        final_model.train()
        final_optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.huber_loss(final_model(full_graphs, full_hp), full_y)
        if not torch.isfinite(loss):
            raise ProtocolError("non-finite direct-regressor full-refit loss")
        loss.backward()
        final_optimizer.step()
    final_model.eval()
    with torch.no_grad():
        prediction = final_model(
            _graph_tensors(outer_heldout, torch),
            torch.tensor(transform_hp(outer_heldout, hp_standardizer), dtype=torch.float32),
        ).cpu().numpy().astype("float64")
    if stage_callback is not None:
        stage_callback({
            "stage": "direct_full_outer_refit", "model_identity": id(final_model),
            "fit_joint_fingerprints": sorted(str(row["joint_fingerprint"]) for row in outer_train),
            "epochs": best_epoch,
        })
    return prediction, {
        "seed": int(seed), "loss": "Huber", "best_epoch": best_epoch,
        "early_stop_epochs_run": epochs_run,
        "best_inner_validation_huber_loss": best_validation_loss,
        "fresh_model_for_full_outer_refit": id(early_model) != id(final_model),
        "final_refit_used_complete_outer_train": True,
        "final_refit_epochs": best_epoch,
        "prediction_seed_aggregation": "arithmetic_mean_all_seeds_0_1_2",
    }


def average_metric_seed_predictions(predictions: Mapping[int, Any]) -> Any:
    import numpy as np
    if set(predictions) != set(REQUIRED_SEEDS):
        raise ProtocolError("predictions must contain exactly metric seeds 0, 1, and 2")
    return np.mean(
        np.stack([np.asarray(predictions[seed], dtype=np.float64) for seed in REQUIRED_SEEDS], axis=0),
        axis=0,
    )


def weighted_knn_predict(
    train_x: Any, train_y: Any, query_x: Any, k: int = 5,
    *, train_fingerprints: Sequence[str] | None = None,
    query_fingerprints: Sequence[str] | None = None,
) -> Any:
    try:
        return v2.weighted_knn_predict(
            train_x, train_y, query_x, k,
            train_fingerprints=train_fingerprints,
            query_fingerprints=query_fingerprints,
        )
    except v2.ProtocolError as error:
        raise ProtocolError(str(error)) from error


GP_NUMERICAL_PROTOCOL = {
    "optimizer": "Adam",
    "learning_rate": 0.05,
    "optimization_steps": 20,
    "restart_seed": 0,
    "initial_noise_floor_original_target_units": 1.0e-4,
    "minimum_noise_floor_original_target_units": 1.0e-8,
    "cholesky_jitter": 1.0e-6,
    "source": "unchanged numerical optimizer settings reused from performance metric v2",
}


def standardize_train_query(train_x: Any, query_x: Any) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    train = np.asarray(train_x, dtype=np.float64)
    query = np.asarray(query_x, dtype=np.float64)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (train - mean) / scale, (query - mean) / scale, {
        "mean": mean.tolist(), "scale": scale.tolist(), "fit_scope": "outer_train_only",
    }


def exact_matern_gp_predict(
    train_x: Any, train_y: Any, train_se: Any, query_x: Any,
    *, standardize_inputs: bool, ard: bool, protocol_name: str,
) -> tuple[Any, dict[str, Any]]:
    """Exact Matérn-5/2 GP with SE² plus a fitted noise-floor²."""
    try:
        import numpy as np
        import torch
        import gpytorch
        from gpytorch.constraints import GreaterThan
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
        from gpytorch.means import ConstantMean
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as error:
        raise GPProtocolUnavailable("GPyTorch fixed-noise Exact GP is unavailable") from error
    train = np.asarray(train_x, dtype=np.float64)
    query = np.asarray(query_x, dtype=np.float64)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]:
        raise ProtocolError("GP train/query feature dimensions are invalid")
    if bool(standardize_inputs):
        train_used, query_used, transform = standardize_train_query(train, query)
    else:
        train_used, query_used = train.copy(), query.copy()
        transform = {"fit_scope": "none", "mean": None, "scale": None}
        if protocol_name in {"M2", "M3"} and not np.array_equal(train_used, train):
            raise ProtocolError("M2/M3 input embedding was altered")
    y = np.asarray(train_y, dtype=np.float64)
    se = np.asarray(train_se, dtype=np.float64)
    if len(y) != len(train_used) or len(se) != len(train_used):
        raise ProtocolError("GP target/noise length mismatch")
    if not np.isfinite(train_used).all() or not np.isfinite(query_used).all():
        raise ProtocolError("GP features contain non-finite values")
    y_mean = float(y.mean())
    y_scale = max(float(y.std()), 1e-8)
    y_standardized = (y - y_mean) / y_scale
    fixed_variance = np.square(se / y_scale)
    initial_floor = float(GP_NUMERICAL_PROTOCOL["initial_noise_floor_original_target_units"])
    minimum_floor = float(GP_NUMERICAL_PROTOCOL["minimum_noise_floor_original_target_units"])
    initial_floor_variance = (initial_floor / y_scale) ** 2
    minimum_floor_variance = (minimum_floor / y_scale) ** 2
    _seed_everything(int(GP_NUMERICAL_PROTOCOL["restart_seed"]), torch)
    tx = torch.as_tensor(train_used, dtype=torch.double)
    ty = torch.as_tensor(y_standardized, dtype=torch.double)
    tq = torch.as_tensor(query_used, dtype=torch.double)
    tnoise = torch.as_tensor(fixed_variance, dtype=torch.double)
    with gpytorch.settings.min_fixed_noise(float_value=0.0, double_value=0.0, half_value=0.0):
        likelihood = FixedNoiseGaussianLikelihood(
            noise=tnoise,
            learn_additional_noise=True,
            noise_constraint=GreaterThan(minimum_floor_variance),
        ).double()
    if not torch.equal(likelihood.noise_covar.noise, tnoise):
        raise GPProtocolUnavailable("fixed validation SE-squared noise was altered")
    likelihood.second_noise = initial_floor_variance
    ard_dimensions = int(tx.shape[1]) if bool(ard) else None

    class FrozenExactGP(gpytorch.models.ExactGP):
        def __init__(self) -> None:
            super().__init__(tx, ty, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_dimensions))

        def forward(self, x: Any) -> Any:
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x),
            )

    model = FrozenExactGP().double()
    if protocol_name in {"M2", "M3"}:
        base_kernel = model.covar_module.base_kernel
        if base_kernel.ard_num_dims is not None or base_kernel.lengthscale.numel() != 1:
            raise ProtocolError("M2/M3 kernel is not isotropic single-lengthscale Matérn")
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(GP_NUMERICAL_PROTOCOL["learning_rate"]),
    )
    marginal_likelihood = ExactMarginalLogLikelihood(likelihood, model)
    with gpytorch.settings.cholesky_jitter(float(GP_NUMERICAL_PROTOCOL["cholesky_jitter"])):
        for _step in range(int(GP_NUMERICAL_PROTOCOL["optimization_steps"])):
            optimizer.zero_grad(set_to_none=True)
            loss = -marginal_likelihood(model(tx), ty)
            if not torch.isfinite(loss):
                raise ProtocolError(f"non-finite {protocol_name} exact GP marginal likelihood")
            loss.backward()
            optimizer.step()
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(float(GP_NUMERICAL_PROTOCOL["cholesky_jitter"])):
        prediction = model(tq).mean.detach().cpu().numpy() * y_scale + y_mean
    fitted_variance_standardized = float(likelihood.second_noise.detach().cpu().reshape(-1)[0])
    fitted_floor = math.sqrt(max(fitted_variance_standardized, 0.0)) * y_scale
    audit = {
        "protocol_name": protocol_name,
        "mean": "ConstantMean", "kernel": "Matern-5/2",
        "ard": bool(ard), "ard_num_dims": ard_dimensions,
        "lengthscale_count": int(tx.shape[1]) if ard else 1,
        "input_standardized": bool(standardize_inputs),
        "input_transform": transform,
        "input_exactly_preserved": bool(not standardize_inputs),
        "target_standardized_from_train": True,
        "target_mean": y_mean, "target_scale": y_scale,
        "fixed_observation_noise_rule": "validation_se_squared",
        "fitted_noise_floor_rule": "validation_se_squared_plus_floor_squared",
        "fitted_noise_floor_original_target_units": fitted_floor,
        "noise_floor_fit_scope": "supplied_outer_train_only",
        "posterior_variance_used": False,
        "numerical_protocol": GP_NUMERICAL_PROTOCOL,
    }
    return prediction.astype(np.float64), audit


def local_gp_predict(
    train_embedding: Any, train_y: Any, train_se: Any, query_embedding: Any,
    neighbors: int = 32,
) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np
    train_x = np.asarray(train_embedding, dtype=np.float64)
    query_x = np.asarray(query_embedding, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    se = np.asarray(train_se, dtype=np.float64)
    if len(train_x) < int(neighbors):
        raise ProtocolError(f"M3 requires at least {neighbors} outer-train configurations")
    predictions: list[float] = []
    audits: list[dict[str, Any]] = []
    for query_index, query in enumerate(query_x):
        distances = np.linalg.norm(train_x - query[None, :], axis=1)
        indices = np.argsort(distances, kind="stable")[: int(neighbors)]
        prediction, audit = exact_matern_gp_predict(
            train_x[indices], y[indices], se[indices], query[None, :],
            standardize_inputs=False, ard=False, protocol_name="M3",
        )
        predictions.append(float(prediction[0]))
        audits.append({
            "query_index": query_index,
            "neighbor_indices": [int(index) for index in indices],
            "gp": audit,
        })
    return np.asarray(predictions, dtype=np.float64), audits


def encode_rows_with_model(
    model: Any, rows: Sequence[Mapping[str, Any]], hp_standardizer: Mapping[str, Any],
    branch_mode: str = "full_joint",
) -> Any:
    try:
        import torch
    except ImportError as error:
        raise ProtocolError("PyTorch is required to encode control rows") from error
    model.eval()
    with torch.no_grad():
        return model(
            _graph_tensors(rows, torch),
            torch.tensor(transform_hp(rows, hp_standardizer), dtype=torch.float32),
            branch_mode,
        ).cpu().numpy().astype("float64")


def shuffle_rows_within_fold(
    rows: Sequence[Mapping[str, Any]], branch: str, seed: int,
) -> list[dict[str, Any]]:
    """Shuffle exactly one branch independently inside every OOF fold."""
    if branch not in {"architecture", "hp"}:
        raise ProtocolError("shuffle branch must be architecture or hp")
    by_fold: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if "fold" not in row:
            raise ProtocolError("fold-local shuffle row is missing fold")
        by_fold[int(row["fold"])].append((index, row))
    output = [dict(row) for row in rows]
    fields = (
        ("architecture_fingerprint", "operations", "edges")
        if branch == "architecture" else
        ("hp_fingerprint", "lr", "dropout", "hidden_dim", "l2", "raw_hp")
    )
    for fold, indexed_rows in sorted(by_fold.items()):
        donors = [dict(row) for _index, row in indexed_rows]
        rng = random.Random(int(seed) + fold * 1_000_003)
        rng.shuffle(donors)
        for (target_index, _target), donor in zip(indexed_rows, donors):
            for field in fields:
                output[target_index][field] = copy.deepcopy(donor[field])
    return output


def _spearman(left: Any, right: Any) -> float:
    import numpy as np
    from scipy.stats import spearmanr
    value = float(spearmanr(np.asarray(left), np.asarray(right)).statistic)
    return value if math.isfinite(value) else 0.0


def geometry_for_fold(
    train_rows: Sequence[Mapping[str, Any]], heldout_rows: Sequence[Mapping[str, Any]],
    seed_distance_matrices: Mapping[int, Any], explicit_distance_matrix: Any,
    hp_standardizer: Mapping[str, Any], equivalence_margin: float = 0.006,
    far_margin: float = 0.012,
) -> dict[str, Any]:
    import numpy as np
    if set(seed_distance_matrices) != set(REQUIRED_SEEDS):
        raise ProtocolError("geometry requires all three metric-seed distance matrices")
    train_y = np.asarray([float(row["validation_mean"]) for row in train_rows], dtype=np.float64)
    heldout_y = np.asarray([float(row["validation_mean"]) for row in heldout_rows], dtype=np.float64)
    performance_gaps = np.abs(heldout_y[:, None] - train_y[None, :])
    seed_spearman = {
        str(seed): _spearman(np.asarray(seed_distance_matrices[seed]).reshape(-1), performance_gaps.reshape(-1))
        for seed in REQUIRED_SEEDS
    }
    metric_distances = np.mean(
        np.stack([np.asarray(seed_distance_matrices[seed], dtype=np.float64) for seed in REQUIRED_SEEDS], axis=0),
        axis=0,
    )
    hp_train = transform_hp(train_rows, hp_standardizer)
    hp_heldout = transform_hp(heldout_rows, hp_standardizer)
    train_hp_distances = [
        float(np.linalg.norm(hp_train[left] - hp_train[right]))
        for left, right in itertools.combinations(range(len(train_rows)), 2)
    ]
    hp_median = float(np.median(np.asarray(train_hp_distances, dtype=np.float64)))
    cross_equivalent: list[float] = []
    cross_far: list[float] = []
    for query_index, query in enumerate(heldout_rows):
        for reference_index, reference in enumerate(train_rows):
            architecture_different = str(query["architecture_fingerprint"]) != str(reference["architecture_fingerprint"])
            hp_far = float(np.linalg.norm(hp_heldout[query_index] - hp_train[reference_index])) > hp_median
            if not architecture_different or not hp_far:
                continue
            pair_label = classify_pair(query, reference, equivalence_margin, far_margin)["label"]
            if pair_label == "positive":
                cross_equivalent.append(float(metric_distances[query_index, reference_index]))
            elif pair_label == "negative":
                cross_far.append(float(metric_distances[query_index, reference_index]))
    if not cross_equivalent or not cross_far:
        raise ProtocolError("cross-factor locality ratio has an empty heldout/train cell")
    ratio = float(
        np.median(np.asarray(cross_equivalent))
        / max(float(np.median(np.asarray(cross_far))), 1e-12)
    )

    def neighbor_precision(distance_matrix: Any) -> dict[str, Any]:
        distances = np.asarray(distance_matrix, dtype=np.float64)
        equivalent = 0
        total = 0
        cross_arch_equivalent = 0
        cross_arch_total = 0
        cross_factor_equivalent = 0
        cross_factor_total = 0
        neighbor_records: list[dict[str, Any]] = []
        for query_index, query in enumerate(heldout_rows):
            indices = np.argsort(distances[query_index], kind="stable")[:5]
            for reference_index in indices:
                reference = train_rows[int(reference_index)]
                mean_equivalent = abs(
                    float(query["validation_mean"]) - float(reference["validation_mean"])
                ) <= float(equivalence_margin)
                architecture_different = str(query["architecture_fingerprint"]) != str(reference["architecture_fingerprint"])
                hp_far = float(np.linalg.norm(hp_heldout[query_index] - hp_train[int(reference_index)])) > hp_median
                total += 1
                equivalent += int(mean_equivalent)
                if architecture_different:
                    cross_arch_total += 1
                    cross_arch_equivalent += int(mean_equivalent)
                if architecture_different and hp_far:
                    cross_factor_total += 1
                    cross_factor_equivalent += int(mean_equivalent)
                neighbor_records.append({
                    "query_joint_fingerprint": str(query["joint_fingerprint"]),
                    "reference_joint_fingerprint": str(reference["joint_fingerprint"]),
                    "equivalent": bool(mean_equivalent),
                    "cross_architecture": bool(architecture_different),
                    "cross_factor": bool(architecture_different and hp_far),
                })
        return {
            "equivalent_neighbor_precision_at_5": equivalent / max(total, 1),
            "cross_architecture_equivalent_neighbor_precision_at_5": cross_arch_equivalent / max(cross_arch_total, 1),
            "cross_factor_equivalent_neighbor_precision_at_5": cross_factor_equivalent / max(cross_factor_total, 1),
            "neighbor_count": total,
            "cross_architecture_neighbor_count": cross_arch_total,
            "cross_factor_neighbor_count": cross_factor_total,
            "records": neighbor_records,
        }

    return {
        "seed_distance_gap_spearman": seed_spearman,
        "cross_factor_hp_distance_train_median": hp_median,
        "cross_factor_equivalent_metric_distances": cross_equivalent,
        "cross_factor_far_metric_distances": cross_far,
        "cross_factor_locality_ratio": ratio,
        "joint_metric_neighbors": neighbor_precision(metric_distances),
        "explicit_joint_feature_neighbors": neighbor_precision(explicit_distance_matrix),
        "pair_scope": "heldout_query_to_outer_train_reference_only",
        "training_set_internal_geometry_used_as_main_evidence": False,
    }


def _raw_hp_matrix(rows: Sequence[Mapping[str, Any]]) -> Any:
    import numpy as np
    return np.asarray([hp_feature_vector(row) for row in rows], dtype=np.float64)


def _explicit_graph_matrix(rows: Sequence[Mapping[str, Any]]) -> Any:
    import numpy as np
    return np.asarray([explicit_graph_features(row) for row in rows], dtype=np.float64)


def evaluate_outer_fold(
    fold: Mapping[str, Any], configurations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any], full_pair_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    train_ids = set(map(str, fold["train_joint_fingerprints"]))
    heldout_ids = set(map(str, fold["heldout_joint_fingerprints"]))
    train_rows = sorted(
        (dict(row) for row in configurations if str(row["joint_fingerprint"]) in train_ids),
        key=lambda row: str(row["joint_fingerprint"]),
    )
    heldout_rows = sorted(
        (dict(row) for row in configurations if str(row["joint_fingerprint"]) in heldout_ids),
        key=lambda row: str(row["joint_fingerprint"]),
    )
    if len(train_rows) != len(train_ids) or len(heldout_rows) != len(heldout_ids):
        raise ProtocolError("outer fold configuration lookup is incomplete")
    hp_standardizer = fit_hp_standardizer(train_rows)
    train_y = np.asarray([float(row["validation_mean"]) for row in train_rows], dtype=np.float64)
    heldout_y = np.asarray([float(row["validation_mean"]) for row in heldout_rows], dtype=np.float64)
    train_se = np.asarray([float(row["validation_se"]) for row in train_rows], dtype=np.float64)
    train_hp_raw = _raw_hp_matrix(train_rows)
    heldout_hp_raw = _raw_hp_matrix(heldout_rows)
    train_graph = _explicit_graph_matrix(train_rows)
    heldout_graph = _explicit_graph_matrix(heldout_rows)
    train_explicit_joint = np.concatenate((train_graph, train_hp_raw), axis=1)
    heldout_explicit_joint = np.concatenate((heldout_graph, heldout_hp_raw), axis=1)
    standardized_explicit_train, standardized_explicit_heldout, explicit_transform = standardize_train_query(
        train_explicit_joint, heldout_explicit_joint,
    )
    explicit_distances = np.linalg.norm(
        standardized_explicit_heldout[:, None, :] - standardized_explicit_train[None, :, :], axis=2,
    )
    train_fingerprints = [str(row["joint_fingerprint"]) for row in train_rows]
    heldout_fingerprints = [str(row["joint_fingerprint"]) for row in heldout_rows]

    predictions: dict[str, Any] = {}
    gp_audits: dict[str, Any] = {}
    predictions["B0"], gp_audits["B0"] = exact_matern_gp_predict(
        train_hp_raw, train_y, train_se, heldout_hp_raw,
        standardize_inputs=True, ard=True, protocol_name="B0",
    )
    predictions["B1"], gp_audits["B1"] = exact_matern_gp_predict(
        train_graph, train_y, train_se, heldout_graph,
        standardize_inputs=True, ard=True, protocol_name="B1",
    )
    predictions["B2"], gp_audits["B2"] = exact_matern_gp_predict(
        train_explicit_joint, train_y, train_se, heldout_explicit_joint,
        standardize_inputs=True, ard=True, protocol_name="B2",
    )
    predictions["EJ_KNN"] = weighted_knn_predict(
        standardized_explicit_train, train_y, standardized_explicit_heldout, 5,
        train_fingerprints=train_fingerprints, query_fingerprints=heldout_fingerprints,
    )

    direct_seed_predictions: dict[int, Any] = {}
    direct_audits: list[dict[str, Any]] = []
    for seed in REQUIRED_SEEDS:
        prediction, audit = train_direct_joint_regressor(
            train_rows, heldout_rows, seed, config, hp_standardizer,
        )
        direct_seed_predictions[seed] = prediction
        direct_audits.append(audit)
    predictions["B3"] = average_metric_seed_predictions(direct_seed_predictions)

    metric_audits: list[dict[str, Any]] = []
    branch_predictions: dict[str, dict[str, Any]] = {}
    full_seed_train_embeddings: dict[int, Any] = {}
    full_seed_heldout_embeddings: dict[int, Any] = {}
    full_seed_models: dict[int, Any] = {}
    full_seed_knn: dict[int, Any] = {}
    full_seed_gp: dict[int, Any] = {}
    full_seed_local_gp: dict[int, Any] = {}
    full_seed_ard_gp: dict[int, Any] = {}
    full_seed_architecture_shuffle_gp: dict[int, Any] = {}
    full_seed_hp_shuffle_gp: dict[int, Any] = {}
    full_seed_distances: dict[int, Any] = {}
    local_gp_audits: dict[str, Any] = {}
    shuffle_seed = int(config["controls"]["shuffle_seed"])
    heldout_with_fold = [{**row, "fold": int(fold["fold"])} for row in heldout_rows]
    architecture_shuffled_rows = shuffle_rows_within_fold(heldout_with_fold, "architecture", shuffle_seed)
    hp_shuffled_rows = shuffle_rows_within_fold(heldout_with_fold, "hp", shuffle_seed)

    for branch_mode in ("architecture_only", "hp_only", "full_joint"):
        seed_knn: dict[int, Any] = {}
        seed_m2: dict[int, Any] = {}
        for seed in REQUIRED_SEEDS:
            train_embedding, heldout_embedding, model, audit = train_joint_metric_encoder(
                train_rows, heldout_rows, seed, config, hp_standardizer,
                full_pair_inventory, branch_mode=branch_mode,
            )
            audit["fold"] = int(fold["fold"])
            metric_audits.append(audit)
            seed_knn[seed] = weighted_knn_predict(
                train_embedding, train_y, heldout_embedding, 5,
                train_fingerprints=train_fingerprints, query_fingerprints=heldout_fingerprints,
            )
            seed_m2[seed], m2_audit = exact_matern_gp_predict(
                train_embedding, train_y, train_se, heldout_embedding,
                standardize_inputs=False, ard=False, protocol_name="M2",
            )
            audit["M2_gp"] = m2_audit
            if branch_mode == "full_joint":
                full_seed_train_embeddings[seed] = train_embedding
                full_seed_heldout_embeddings[seed] = heldout_embedding
                full_seed_models[seed] = model
                full_seed_knn[seed] = seed_knn[seed]
                full_seed_gp[seed] = seed_m2[seed]
                local_prediction, local_audit = local_gp_predict(
                    train_embedding, train_y, train_se, heldout_embedding,
                    int(config["metric_methods"]["M3"]["neighbors"]),
                )
                full_seed_local_gp[seed] = local_prediction
                local_gp_audits[str(seed)] = local_audit
                full_seed_ard_gp[seed], audit["M4_gp"] = exact_matern_gp_predict(
                    train_embedding, train_y, train_se, heldout_embedding,
                    standardize_inputs=True, ard=True, protocol_name="M4",
                )
                architecture_shuffled_embedding = encode_rows_with_model(
                    model, architecture_shuffled_rows, hp_standardizer, "full_joint",
                )
                hp_shuffled_embedding = encode_rows_with_model(
                    model, hp_shuffled_rows, hp_standardizer, "full_joint",
                )
                full_seed_architecture_shuffle_gp[seed], audit["architecture_shuffle_M2_gp"] = exact_matern_gp_predict(
                    train_embedding, train_y, train_se, architecture_shuffled_embedding,
                    standardize_inputs=False, ard=False, protocol_name="M2",
                )
                full_seed_hp_shuffle_gp[seed], audit["hp_shuffle_M2_gp"] = exact_matern_gp_predict(
                    train_embedding, train_y, train_se, hp_shuffled_embedding,
                    standardize_inputs=False, ard=False, protocol_name="M2",
                )
                full_seed_distances[seed] = np.linalg.norm(
                    heldout_embedding[:, None, :] - train_embedding[None, :, :], axis=2,
                )
        branch_predictions[branch_mode] = {
            "M1": average_metric_seed_predictions(seed_knn),
            "M2": average_metric_seed_predictions(seed_m2),
            "seed_aggregation": "arithmetic_mean_all_seeds_0_1_2",
        }

    predictions["M1"] = average_metric_seed_predictions(full_seed_knn)
    predictions["M2"] = average_metric_seed_predictions(full_seed_gp)
    predictions["M3"] = average_metric_seed_predictions(full_seed_local_gp)
    predictions["M4"] = average_metric_seed_predictions(full_seed_ard_gp)
    predictions["M2_architecture_shuffled"] = average_metric_seed_predictions(full_seed_architecture_shuffle_gp)
    predictions["M2_hp_shuffled"] = average_metric_seed_predictions(full_seed_hp_shuffle_gp)
    predictions["ABL_ARCH_M1"] = branch_predictions["architecture_only"]["M1"]
    predictions["ABL_ARCH_M2"] = branch_predictions["architecture_only"]["M2"]
    predictions["ABL_HP_M1"] = branch_predictions["hp_only"]["M1"]
    predictions["ABL_HP_M2"] = branch_predictions["hp_only"]["M2"]

    geometry = geometry_for_fold(
        train_rows, heldout_rows, full_seed_distances, explicit_distances,
        hp_standardizer,
        float(config["triplets"]["equivalence_margin"]),
        float(config["triplets"]["far_margin"]),
    )
    oof_rows: list[dict[str, Any]] = []
    for index, row in enumerate(heldout_rows):
        output_row: dict[str, Any] = {
            "fold": int(fold["fold"]),
            "joint_fingerprint": str(row["joint_fingerprint"]),
            "architecture_fingerprint": str(row["architecture_fingerprint"]),
            "hp_fingerprint": str(row["hp_fingerprint"]),
            "pool_id": int(row["pool_id"]), "batch_id": int(row["batch_id"]),
            "slot_id": int(row["slot_id"]),
            "raw_validation_mean": float(heldout_y[index]),
            "validation_se": float(row["validation_se"]),
        }
        for model, values in predictions.items():
            output_row[f"{model}_prediction"] = float(values[index])
        oof_rows.append(output_row)
    return {
        "fold": int(fold["fold"]),
        "oof_rows": oof_rows,
        "geometry": geometry,
        "pair_inventory": {
            "fold": int(fold["fold"]),
            "counts": dict(full_pair_inventory["counts"]),
            "hp_distance_median": float(full_pair_inventory["hp_distance_median"]),
            "allowed_joint_fingerprints": list(full_pair_inventory["allowed_joint_fingerprints"]),
            "outer_heldout_labels_used": False,
            "ambiguous_pairs_enter_loss": False,
        },
        "metric_training_audits": metric_audits,
        "direct_regressor_audits": direct_audits,
        "gp_audits": gp_audits,
        "local_gp_audits": local_gp_audits,
        "hp_standardizer": hp_standardizer,
        "explicit_joint_feature_transform": explicit_transform,
        "branch_prediction_protocol": {
            "architecture_only": "HP branch and interaction zeroed before frozen fusion",
            "hp_only": "architecture branch and interaction zeroed before frozen fusion",
            "full_joint": "architecture, HP, and elementwise interaction",
        },
        "shuffle_protocol": {
            "seed": shuffle_seed, "scope": "within_outer_fold",
            "architecture_multiset_preserved": True, "hp_multiset_preserved": True,
        },
        "metric_seed_aggregation": "arithmetic_mean_all_seeds_0_1_2",
        "best_metric_seed_selected": False,
        "old_z_search_status": "OLD_Z_SEARCH_UNAVAILABLE",
        "old_z_search_reason": "allowed frozen source rows contain no unambiguous z_search representation",
    }


def prediction_metrics(
    rows: Sequence[Mapping[str, Any]], model: str, top_fraction: float = 0.20,
) -> dict[str, float]:
    try:
        import numpy as np
        from scipy.stats import kendalltau, pearsonr, spearmanr
        from sklearn.metrics import ndcg_score
    except ImportError as error:
        raise ProtocolError("NumPy, SciPy, and scikit-learn are required for prediction metrics") from error
    if not rows:
        raise ProtocolError("prediction metrics require OOF rows")
    field = f"{model}_prediction"
    if any(field not in row for row in rows):
        raise ProtocolError(f"OOF rows are missing {field}")
    y = np.asarray([float(row["raw_validation_mean"]) for row in rows], dtype=np.float64)
    prediction = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ProtocolError(f"non-finite target or prediction for {model}")
    top_count = max(1, int(math.ceil(float(top_fraction) * len(rows))))
    identity = [str(row["joint_fingerprint"]) for row in rows]
    true_order = sorted(range(len(rows)), key=lambda index: (-float(y[index]), identity[index]))
    prediction_order = sorted(range(len(rows)), key=lambda index: (-float(prediction[index]), identity[index]))
    true_top = set(true_order[:top_count])
    predicted_top = set(prediction_order[:top_count])

    def finite_statistic(value: float) -> float:
        return float(value) if math.isfinite(float(value)) else 0.0

    pearson = finite_statistic(float(pearsonr(y, prediction).statistic))
    spearman = finite_statistic(float(spearmanr(y, prediction).statistic))
    kendall = finite_statistic(float(kendalltau(y, prediction).statistic))
    top_indices = np.asarray(true_order[:top_count], dtype=int)
    if len(top_indices) >= 2:
        top_kendall = finite_statistic(float(kendalltau(y[top_indices], prediction[top_indices]).statistic))
    else:
        top_kendall = 0.0
    ndcg = finite_statistic(float(ndcg_score(y[None, :], prediction[None, :], k=top_count)))
    errors = prediction - y
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "top20_precision": len(true_top & predicted_top) / top_count,
        "top_region_kendall": top_kendall,
        "ndcg_at_20_percent": ndcg,
        "top_count": float(top_count),
        "top_region_definition": "true top-20-percent configurations",
        "ndcg_relevance": "raw validation mean; sklearn ndcg_score",
    }


def _batch_manifest_for_fold(
    fold_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> list[list[str]]:
    fingerprints = sorted(str(row["joint_fingerprint"]) for row in fold_rows)
    batch_size = int(config["best_of_4"]["batch_size"])
    if len(fingerprints) < batch_size:
        raise ProtocolError("heldout fold contains fewer than four configurations")
    total = math.comb(len(fingerprints), batch_size)
    if total <= int(config["best_of_4"]["exhaustive_combination_limit"]):
        return [list(batch) for batch in itertools.combinations(fingerprints, batch_size)]
    count = int(config["best_of_4"]["sampled_batch_count"])
    rng = random.Random(int(config["best_of_4"]["sampled_batch_seed"]))
    batches: set[tuple[str, ...]] = set()
    while len(batches) < count:
        batches.add(tuple(sorted(rng.sample(fingerprints, batch_size))))
    return [list(batch) for batch in sorted(batches)]


def best_of_four_evaluation(
    oof_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    models: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    by_fold: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in oof_rows:
        by_fold[int(row["fold"])].append(row)
    manifest_folds: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for fold, rows in sorted(by_fold.items()):
        lookup = {str(row["joint_fingerprint"]): row for row in rows}
        batches = _batch_manifest_for_fold(rows, config)
        exhaustive_count = math.comb(len(rows), int(config["best_of_4"]["batch_size"]))
        manifest_folds.append({
            "fold": fold,
            "batch_count": len(batches),
            "mode": "exhaustive" if len(batches) == exhaustive_count else "sampled_seed_0",
            "batches": batches,
        })
        for batch_index, batch in enumerate(batches):
            candidates = [lookup[fingerprint] for fingerprint in batch]
            true_best = max(float(row["raw_validation_mean"]) for row in candidates)
            for model in models:
                selected = min(
                    candidates,
                    key=lambda row: (
                        -float(row[f"{model}_prediction"]), str(row["joint_fingerprint"]),
                    ),
                )
                selected_value = float(selected["raw_validation_mean"])
                result_rows.append({
                    "fold": fold, "batch_index": batch_index, "model": model,
                    "selected_joint_fingerprint": str(selected["joint_fingerprint"]),
                    "selected_raw_validation_mean": selected_value,
                    "true_best_raw_validation_mean": true_best,
                    "simple_regret": true_best - selected_value,
                    "selected_true_best": bool(abs(true_best - selected_value) <= 1e-15),
                })
    summaries: dict[str, Any] = {}
    for model in models:
        model_rows = [row for row in result_rows if row["model"] == model]
        regrets = np.asarray([float(row["simple_regret"]) for row in model_rows], dtype=np.float64)
        selected = np.asarray([float(row["selected_raw_validation_mean"]) for row in model_rows], dtype=np.float64)
        per_fold: dict[str, float] = {}
        for fold in sorted(by_fold):
            fold_values = [float(row["simple_regret"]) for row in model_rows if int(row["fold"]) == fold]
            per_fold[str(fold)] = float(np.mean(np.asarray(fold_values, dtype=np.float64)))
        summaries[model] = {
            "mean_simple_regret": float(np.mean(regrets)),
            "median_simple_regret": float(np.median(regrets)),
            "probability_selecting_true_best": sum(bool(row["selected_true_best"]) for row in model_rows) / len(model_rows),
            "selected_raw_validation_mean": float(np.mean(selected)),
            "per_fold_mean_simple_regret": per_fold,
            "batch_count": len(model_rows),
        }
    return {
        "format_version": FORMAT_VERSION,
        "batch_size": int(config["best_of_4"]["batch_size"]),
        "folds": manifest_folds,
        "shared_by_all_models": True,
    }, {
        "format_version": FORMAT_VERSION,
        "models": summaries,
        "rows": result_rows,
    }


def _relative_improvement(baseline: float, challenger: float) -> float:
    return (float(baseline) - float(challenger)) / max(abs(float(baseline)), 1e-12)


def _model_composite_gate(
    challenger: str, baseline: str,
    predictions: Mapping[str, Mapping[str, float]],
    best_of_four: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *, require_fold_consistency: bool,
) -> dict[str, Any]:
    challenger_metrics = predictions[challenger]
    baseline_metrics = predictions[baseline]
    challenger_regret = best_of_four["models"][challenger]
    baseline_regret = best_of_four["models"][baseline]
    mae_gain = _relative_improvement(baseline_metrics["mae"], challenger_metrics["mae"])
    precision_gain = float(challenger_metrics["top20_precision"]) - float(baseline_metrics["top20_precision"])
    regret_gain = _relative_improvement(
        baseline_regret["mean_simple_regret"], challenger_regret["mean_simple_regret"],
    )
    lower_regret_folds = sum(
        float(challenger_regret["per_fold_mean_simple_regret"][str(fold)])
        < float(baseline_regret["per_fold_mean_simple_regret"][str(fold)])
        for fold in range(5)
    )
    passed = (
        mae_gain >= float(thresholds["mae_relative_improvement_at_least"])
        and precision_gain >= float(thresholds["top20_precision_absolute_improvement_at_least"])
        and regret_gain >= float(thresholds["best_of_4_regret_relative_improvement_at_least"])
    )
    if require_fold_consistency:
        passed = passed and lower_regret_folds >= int(thresholds["lower_regret_outer_folds_at_least"])
    return {
        "challenger": challenger, "single_baseline_model": baseline,
        "mae_relative_improvement": mae_gain,
        "top20_precision_absolute_improvement": precision_gain,
        "best_of_4_regret_relative_improvement": regret_gain,
        "lower_regret_outer_folds": lower_regret_folds,
        "require_fold_consistency": require_fold_consistency,
        "passed": bool(passed),
        "cross_model_metric_splicing_used": False,
    }


def classify_joint_metric_gate(
    unique_joint_configurations: int,
    unique_architectures: int,
    pair_power_passed: bool | None,
    geometry: Mapping[str, Any] | None,
    predictions: Mapping[str, Mapping[str, float]] | None,
    best_of_four: Mapping[str, Any] | None,
    decision_config: Mapping[str, Any],
) -> dict[str, Any]:
    data_gate = decision_config["data_gate"]
    if (
        int(unique_joint_configurations) < int(data_gate["minimum_unique_joint_configurations"])
        or int(unique_architectures) < int(data_gate["minimum_unique_architectures"])
    ):
        return {
            "classification": "INSUFFICIENT_JOINT_CONFIG_DATA",
            "unique_joint_configurations": int(unique_joint_configurations),
            "unique_architectures": int(unique_architectures),
            "bo_authorized": False, "uncertainty_calibration_claimed": False,
        }
    if pair_power_passed is False:
        return {
            "classification": "JOINT_METRIC_UNDERPOWERED",
            "unique_joint_configurations": int(unique_joint_configurations),
            "unique_architectures": int(unique_architectures),
            "thresholds_relaxed": False,
            "bo_authorized": False, "uncertainty_calibration_claimed": False,
        }
    if pair_power_passed is not True or geometry is None or predictions is None or best_of_four is None:
        raise ProtocolError("complete decision inputs are required after data and pair-power gates")
    geometry_config = decision_config["geometry_gate"]
    seed_fold_values = {
        str(seed): [
            float(fold["seed_distance_gap_spearman"][str(seed)]) for fold in geometry["folds"]
        ] for seed in REQUIRED_SEEDS
    }
    all_spearman = [value for values in seed_fold_values.values() for value in values]
    import numpy as np
    median_spearman = float(np.median(np.asarray(all_spearman, dtype=np.float64)))
    positive_folds = {
        seed: sum(value > 0.0 for value in values) for seed, values in seed_fold_values.items()
    }
    positive_seed_count = sum(
        count >= int(geometry_config["positive_folds_per_seed_at_least"])
        for count in positive_folds.values()
    )
    neighbor_gain = (
        float(geometry["joint_metric_neighbors"]["equivalent_neighbor_precision_at_5"])
        - float(geometry["explicit_joint_feature_neighbors"]["equivalent_neighbor_precision_at_5"])
    )
    geometry_passed = (
        median_spearman > float(geometry_config["median_fold_spearman_across_metric_seeds_strictly_greater_than"])
        and positive_seed_count >= int(geometry_config["positive_seed_count_at_least"])
        and float(geometry["cross_factor_locality_ratio"]) < float(geometry_config["cross_factor_locality_ratio_strictly_less_than"])
        and neighbor_gain >= float(geometry_config["equivalent_neighbor_precision5_improvement_over_explicit_knn_at_least"])
    )
    geometry_evidence = {
        "median_fold_spearman_across_all_metric_seeds": median_spearman,
        "positive_folds_by_seed": positive_folds,
        "positive_seed_count": positive_seed_count,
        "cross_factor_locality_ratio": float(geometry["cross_factor_locality_ratio"]),
        "equivalent_neighbor_precision5_improvement_over_explicit_knn": neighbor_gain,
        "passed": bool(geometry_passed),
    }
    if not geometry_passed:
        return {
            "classification": "JOINT_METRIC_REJECTED",
            "reason": "geometry_gate_failed", "geometry_gate": geometry_evidence,
            "bo_authorized": False, "uncertainty_calibration_claimed": False,
        }
    strongest_non_metric = min(("B0", "B1", "B2"), key=lambda model: (float(predictions[model]["mae"]), model))
    m1_gate = _model_composite_gate(
        "M1", "EJ_KNN", predictions, best_of_four,
        decision_config["M1_gate_over_explicit_joint_feature_knn"],
        require_fold_consistency=False,
    )
    m2_gate = _model_composite_gate(
        "M2", strongest_non_metric, predictions, best_of_four,
        decision_config["M2_gate_over_strongest_non_metric_gp"],
        require_fold_consistency=True,
    )
    comparison_metrics = {
        "mae": (
            float(predictions["M2"]["mae"]) < float(predictions["B0"]["mae"])
            and float(predictions["M2"]["mae"]) < float(predictions["B1"]["mae"])
        ),
        "top20_precision": (
            float(predictions["M2"]["top20_precision"]) > float(predictions["B0"]["top20_precision"])
            and float(predictions["M2"]["top20_precision"]) > float(predictions["B1"]["top20_precision"])
        ),
        "best_of_4_regret": (
            float(best_of_four["models"]["M2"]["mean_simple_regret"])
            < float(best_of_four["models"]["B0"]["mean_simple_regret"])
            and float(best_of_four["models"]["M2"]["mean_simple_regret"])
            < float(best_of_four["models"]["B1"]["mean_simple_regret"])
        ),
    }
    required_better = int(decision_config["joint_contribution_gate"]["required_better_metric_count"])
    b3_degradation = (
        float(predictions["M2"]["mae"]) - float(predictions["B3"]["mae"])
    ) / max(float(predictions["B3"]["mae"]), 1e-12)
    branch_baseline_passed = (
        sum(bool(value) for value in comparison_metrics.values()) >= required_better
        and b3_degradation <= float(decision_config["joint_contribution_gate"]["maximum_mae_degradation_vs_B3"])
    )
    architecture_shuffle_comparison: dict[str, bool] | None = None
    architecture_shuffle_passed = True
    if "M2_architecture_shuffled" in predictions and "M2_architecture_shuffled" in best_of_four["models"]:
        architecture_shuffle_comparison = {
            "mae": float(predictions["M2"]["mae"]) < float(predictions["M2_architecture_shuffled"]["mae"]),
            "top20_precision": float(predictions["M2"]["top20_precision"]) > float(predictions["M2_architecture_shuffled"]["top20_precision"]),
            "best_of_4_regret": (
                float(best_of_four["models"]["M2"]["mean_simple_regret"])
                < float(best_of_four["models"]["M2_architecture_shuffled"]["mean_simple_regret"])
            ),
        }
        # Reuse the already-frozen two-of-three joint-contribution rule; no new
        # shuffle-effect magnitude is introduced.
        architecture_shuffle_passed = (
            sum(bool(value) for value in architecture_shuffle_comparison.values()) >= required_better
        )
    joint_contribution_passed = branch_baseline_passed and architecture_shuffle_passed
    joint_evidence = {
        "metrics_better_than_both_B0_and_B1": comparison_metrics,
        "better_metric_count": sum(bool(value) for value in comparison_metrics.values()),
        "mae_degradation_vs_B3": b3_degradation,
        "branch_baseline_and_B3_passed": bool(branch_baseline_passed),
        "architecture_shuffle_metrics_better": architecture_shuffle_comparison,
        "architecture_shuffle_two_of_three_passed": bool(architecture_shuffle_passed),
        "passed": bool(joint_contribution_passed),
    }
    feature_gates = {
        model: _model_composite_gate(
            model, strongest_non_metric, predictions, best_of_four,
            decision_config["M2_gate_over_strongest_non_metric_gp"],
            require_fold_consistency=True,
        ) for model in ("M4", "B3")
    }
    if m2_gate["passed"] and not joint_contribution_passed:
        classification = "JOINT_METRIC_NOT_JOINT"
    elif m1_gate["passed"] and m2_gate["passed"] and joint_contribution_passed:
        classification = "JOINT_METRIC_GP_READY"
    elif m1_gate["passed"] and not m2_gate["passed"]:
        classification = "JOINT_METRIC_LOCALITY_ONLY"
    elif not m1_gate["passed"] and not m2_gate["passed"] and any(
        gate["passed"] for gate in feature_gates.values()
    ):
        classification = "JOINT_METRIC_FEATURE_ONLY"
    else:
        classification = "JOINT_METRIC_REJECTED"
    return {
        "classification": classification,
        "geometry_gate": geometry_evidence,
        "strongest_non_metric_gp": strongest_non_metric,
        "strongest_non_metric_gp_selection_rule": "lowest pooled OOF MAE; one model used for every M2 threshold",
        "M1_gate": m1_gate, "M2_gate": m2_gate,
        "joint_contribution_gate": joint_evidence,
        "feature_only_diagnostics": feature_gates,
        "M4_substituted_for_M2": False,
        "cross_model_metric_splicing_used": False,
        "metric_seed_selection_used": False,
        "bo_authorized": False,
        "posterior_mean_claim_only": classification == "JOINT_METRIC_GP_READY",
        "uncertainty_calibration_claimed": False,
        "acquisition_validated": False,
    }


def data_gate_decision(
    unique_joint_configurations: int,
    unique_architectures: int,
    decision_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the frozen early classification, or None when data pass."""
    data_gate = decision_config["data_gate"]
    insufficient = (
        int(unique_joint_configurations) < int(data_gate["minimum_unique_joint_configurations"])
        or int(unique_architectures) < int(data_gate["minimum_unique_architectures"])
    )
    if not insufficient:
        return None
    return classify_joint_metric_gate(
        unique_joint_configurations, unique_architectures,
        None, None, None, None, decision_config,
    )


def aggregate_geometry(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np
    equivalent_distances = [
        float(value) for result in fold_results
        for value in result["geometry"]["cross_factor_equivalent_metric_distances"]
    ]
    far_distances = [
        float(value) for result in fold_results
        for value in result["geometry"]["cross_factor_far_metric_distances"]
    ]
    if not equivalent_distances or not far_distances:
        raise ProtocolError("pooled cross-factor geometry cell is empty")

    def aggregate_neighbors(key: str) -> dict[str, Any]:
        records = [
            record for result in fold_results
            for record in result["geometry"][key]["records"]
        ]
        total = len(records)
        cross_arch = [record for record in records if bool(record["cross_architecture"])]
        cross_factor = [record for record in records if bool(record["cross_factor"])]
        return {
            "equivalent_neighbor_precision_at_5": sum(bool(row["equivalent"]) for row in records) / max(total, 1),
            "cross_architecture_equivalent_neighbor_precision_at_5": sum(bool(row["equivalent"]) for row in cross_arch) / max(len(cross_arch), 1),
            "cross_factor_equivalent_neighbor_precision_at_5": sum(bool(row["equivalent"]) for row in cross_factor) / max(len(cross_factor), 1),
            "neighbor_count": total,
            "cross_architecture_neighbor_count": len(cross_arch),
            "cross_factor_neighbor_count": len(cross_factor),
        }

    return {
        "format_version": FORMAT_VERSION,
        "folds": [
            {
                "fold": int(result["fold"]),
                "seed_distance_gap_spearman": result["geometry"]["seed_distance_gap_spearman"],
                "cross_factor_locality_ratio": result["geometry"]["cross_factor_locality_ratio"],
                "joint_metric_neighbors": {
                    key: value for key, value in result["geometry"]["joint_metric_neighbors"].items()
                    if key != "records"
                },
                "explicit_joint_feature_neighbors": {
                    key: value for key, value in result["geometry"]["explicit_joint_feature_neighbors"].items()
                    if key != "records"
                },
            }
            for result in fold_results
        ],
        "cross_factor_locality_ratio": float(
            np.median(np.asarray(equivalent_distances, dtype=np.float64))
            / max(float(np.median(np.asarray(far_distances, dtype=np.float64))), 1e-12)
        ),
        "cross_factor_equivalent_pair_count": len(equivalent_distances),
        "cross_factor_far_pair_count": len(far_distances),
        "joint_metric_neighbors": aggregate_neighbors("joint_metric_neighbors"),
        "explicit_joint_feature_neighbors": aggregate_neighbors("explicit_joint_feature_neighbors"),
        "aggregation": "pooled heldout-query to outer-train reference pairs",
        "training_set_internal_geometry_used_as_main_evidence": False,
    }


def branch_ablation_summary(
    prediction_summary: Mapping[str, Any], best_results: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "architecture_branch_only": {
            "M1_prediction_metrics": prediction_summary["ABL_ARCH_M1"],
            "M2_prediction_metrics": prediction_summary["ABL_ARCH_M2"],
            "M1_best_of_4": best_results["models"]["ABL_ARCH_M1"],
            "M2_best_of_4": best_results["models"]["ABL_ARCH_M2"],
            "implementation": "HP representation and interaction are zeroed before fusion",
        },
        "hp_branch_only": {
            "M1_prediction_metrics": prediction_summary["ABL_HP_M1"],
            "M2_prediction_metrics": prediction_summary["ABL_HP_M2"],
            "M1_best_of_4": best_results["models"]["ABL_HP_M1"],
            "M2_best_of_4": best_results["models"]["ABL_HP_M2"],
            "implementation": "architecture representation and interaction are zeroed before fusion",
        },
        "full_joint": {
            "M1_prediction_metrics": prediction_summary["M1"],
            "M2_prediction_metrics": prediction_summary["M2"],
            "M1_best_of_4": best_results["models"]["M1"],
            "M2_best_of_4": best_results["models"]["M2"],
            "implementation": "architecture, HP, and elementwise interaction are active",
        },
        "target": "raw_validation_mean",
        "metric_seed_aggregation": "arithmetic_mean_all_seeds_0_1_2",
    }


def shuffle_control_summary(
    prediction_summary: Mapping[str, Any], best_results: Mapping[str, Any], seed: int,
) -> dict[str, Any]:
    baseline_metrics = prediction_summary["M2"]
    baseline_regret = best_results["models"]["M2"]["mean_simple_regret"]
    output: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "seed": int(seed), "scope": "within_outer_fold",
        "base_method": "M2",
    }
    for label, model in (
        ("architecture_shuffled", "M2_architecture_shuffled"),
        ("hp_shuffled", "M2_hp_shuffled"),
    ):
        metrics = prediction_summary[model]
        regret = best_results["models"][model]["mean_simple_regret"]
        output[label] = {
            "prediction_metrics": metrics,
            "best_of_4": best_results["models"][model],
            "mae_change": float(metrics["mae"]) - float(baseline_metrics["mae"]),
            "top20_precision_change": float(metrics["top20_precision"]) - float(baseline_metrics["top20_precision"]),
            "best_of_4_regret_change": float(regret) - float(baseline_regret),
        }
    output["shuffle_never_crossed_fold"] = True
    return output


def _run_git(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise ProtocolError(f"git {' '.join(arguments)} failed: {result.stdout.strip()}")
    return result.stdout


def write_git_and_environment(output: Path) -> None:
    branch = _run_git(("branch", "--show-current")).strip()
    head = _run_git(("rev-parse", "HEAD")).strip()
    status = _run_git(("status", "--short", "--branch"))
    dump_text(output / "git_status.txt", f"branch={branch}\nHEAD={head}\n{status}")
    tracked_diff = _run_git(("diff", "--binary", "HEAD"))
    untracked_patch_parts: list[str] = []
    for relative in (
        "configs/joint_configuration_metric_gate_v1.yaml",
        "analyse/joint_configuration_metric_gate.py",
        "scripts/run_joint_configuration_metric_gate.sh",
        "tests/test_joint_configuration_metric_gate.py",
    ):
        path = REPO_ROOT / relative
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if path.is_file() and not tracked:
            result = subprocess.run(
                ["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative],
                cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if result.returncode not in (0, 1):
                raise ProtocolError(f"git no-index diff failed for {relative}")
            untracked_patch_parts.append(result.stdout)
    dump_text(output / "git_diff.patch", tracked_diff + "".join(untracked_patch_parts))
    packages: dict[str, str | None] = {}
    for distribution in ("PyYAML", "numpy", "scipy", "scikit-learn", "torch", "gpytorch"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    dump_json(output / "environment.json", {
        "timestamp_utc": utc_now(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": packages,
        "branch": branch, "head": head,
        "dirty_tree": bool(status.splitlines()[1:]),
        "candidate_gnn_training_started": False,
        "cuda_required": False,
        "metric_models_run_on_cpu": True,
        "network_access_used": False,
    })


def dependency_inventory() -> dict[str, bool]:
    output: dict[str, bool] = {}
    for name in ("yaml", "numpy", "scipy", "sklearn", "torch", "gpytorch"):
        try:
            __import__(name)
            output[name] = True
        except ImportError:
            output[name] = False
    return output


def initial_state(config_hash: str, source_hash: str) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "status": "running", "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "config_sha256": config_hash, "source_sha256": source_hash,
        "split_sha256": None, "completed_folds": [], "decision": None,
        "formal_data_read": False, "candidate_gnn_training_started": False,
        "test_data_accessed": False, "decoder_used": False,
        "acquisition_used": False,
    }


def update_state(output: Path, state: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    updated = {**state, **changes, "updated_at_utc": utc_now()}
    dump_json(output / "state.json", updated)
    return updated


def ensure_output_policy(output: Path, preflight_ready: bool = False) -> None:
    if output.exists() and not output.is_dir():
        raise ProtocolError(f"formal result path is not a directory: {output}")
    if not output.exists():
        return
    entries = [path.relative_to(output).as_posix() for path in output.rglob("*")]
    if not entries:
        return
    allowed_preflight = {
        "logs",
        "logs/py_compile.log", "logs/pytest_focused.log",
        "logs/synthetic_smoke.log", "logs/run.log",
    }
    if preflight_ready and set(entries) <= allowed_preflight:
        return
    raise ProtocolError(f"formal result directory exists and is non-empty: {output}")


def _analysis_placeholders(output: Path, reason: str) -> None:
    if not (output / "split_manifest.json").exists():
        dump_json(output / "split_manifest.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": reason})
    if not (output / "pair_inventory.json").exists():
        dump_json(output / "pair_inventory.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": reason})
    if not (output / "metric_training_audit.json").exists():
        dump_json(output / "metric_training_audit.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": reason})
    if not (output / "geometry_metrics.json").exists():
        dump_json(output / "geometry_metrics.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": reason})
    if not (output / "metrics_per_fold.csv").exists():
        dump_csv(output / "metrics_per_fold.csv", [], fields=("fold", "model", "mae"))
    if not (output / "oof_predictions.csv").exists():
        dump_csv(output / "oof_predictions.csv", [], fields=("fold", "joint_fingerprint", "raw_validation_mean"))
    if not (output / "best_of_4_batches.json").exists():
        dump_json(output / "best_of_4_batches.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": reason})
    if not (output / "best_of_4_results.json").exists():
        dump_json(output / "best_of_4_results.json", {"format_version": FORMAT_VERSION, "models": {}, "reason": reason})
    if not (output / "branch_ablation.json").exists():
        dump_json(output / "branch_ablation.json", {"format_version": FORMAT_VERSION, "reason": reason})
    if not (output / "shuffle_controls.json").exists():
        dump_json(output / "shuffle_controls.json", {"format_version": FORMAT_VERSION, "reason": reason})


def write_final_report(output: Path, decision: Mapping[str, Any], inventory: Mapping[str, Any] | None = None) -> None:
    classification = str(decision["classification"])
    lines = [
        "# Joint configuration metric gate v1",
        "",
        f"- Frozen classification: `{classification}`",
        f"- Unique joint configurations: {int((inventory or {}).get('unique_joint_configurations', 0))}",
        f"- Unique architectures: {int((inventory or {}).get('unique_architectures', 0))}",
        "- Target: raw validation accuracy mean over training seeds 0, 1, and 2.",
        "- Scope: joint architecture+HP metric geometry and posterior mean only.",
        "- Test data, decoder, candidate training, and acquisition functions were not used.",
        "- GP uncertainty calibration and formal BO readiness are not claimed.",
    ]
    if classification == "JOINT_METRIC_GP_READY":
        lines.append("- Meaning: posterior mean warrants a prospective discrete-candidate-pool validation only.")
    if decision.get("reason"):
        lines.append(f"- Reason: `{decision['reason']}`")
    dump_text(output / "final_report.md", "\n".join(lines) + "\n")


def write_manifest(output: Path) -> None:
    if not output.is_dir():
        raise ProtocolError(f"cannot write manifest for missing result directory: {output}")
    files: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(output).as_posix()
        if relative.startswith("cache/") or path.suffix in {".pt", ".pth", ".ckpt"}:
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump_json(output / "manifest.json", {
        "format_version": FORMAT_VERSION, "generated_at_utc": utc_now(),
        "files": files, "file_count": len(files),
    })


def write_failure_bundle(
    output: Path, message: str, config_hash: str, source_hash: str,
    error_type: str = "RUN_FAILED",
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    for name in ("py_compile.log", "pytest_focused.log", "synthetic_smoke.log", "run.log"):
        path = output / "logs" / name
        if not path.exists():
            dump_text(path, "")
    state_path = output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else initial_state(config_hash, source_hash)
    state = update_state(output, state, status="failed", error=message)
    if not (output / "frozen_protocol.json").exists():
        dump_json(output / "frozen_protocol.json", {
            "format_version": FORMAT_VERSION, "config_sha256": config_hash,
            "source_sha256": source_hash, "status": "failure_before_protocol_freeze",
        })
    if not (output / "data_inventory.json").exists():
        dump_json(output / "data_inventory.json", {"status": "failed_before_inventory", "error": message})
    if not (output / "data_provenance.json").exists():
        dump_json(output / "data_provenance.json", {"status": "failed_before_provenance", "error": message})
    _analysis_placeholders(output, message)
    if not (output / "environment.json").exists() or not (output / "git_status.txt").exists() or not (output / "git_diff.patch").exists():
        try:
            write_git_and_environment(output)
        except Exception as git_error:
            if not (output / "environment.json").exists():
                dump_json(output / "environment.json", {"status": "unavailable", "error": str(git_error)})
            if not (output / "git_status.txt").exists():
                dump_text(output / "git_status.txt", f"unavailable: {git_error}\n")
            if not (output / "git_diff.patch").exists():
                dump_text(output / "git_diff.patch", "")
    decision = {
        "classification": "RUN_FAILED", "reason": message,
        "failure_code": error_type,
        "bo_authorized": False, "uncertainty_calibration_claimed": False,
    }
    dump_json(output / "decision.json", decision)
    write_final_report(output, decision)
    update_state(output, state, status="failed", decision="RUN_FAILED", error=message)
    write_manifest(output)


EVALUATED_MODELS = (
    "B0", "B1", "B2", "B3", "EJ_KNN", "M1", "M2", "M3", "M4",
    "ABL_ARCH_M1", "ABL_ARCH_M2", "ABL_HP_M1", "ABL_HP_M2",
    "M2_architecture_shuffled", "M2_hp_shuffled",
)


def _pair_artifact_record(fold: int, inventory: Mapping[str, Any]) -> dict[str, Any]:
    compact_pairs = [
        {
            "left": pair["left"], "right": pair["right"], "label": pair["label"],
            "cross_architecture": pair["cross_architecture"],
            "cross_factor": pair["cross_factor"],
        }
        for pair in inventory["pairs"]
    ]
    return {
        "fold": int(fold), "counts": dict(inventory["counts"]),
        "hp_distance_median": float(inventory["hp_distance_median"]),
        "hp_median_fit_joint_fingerprints": list(inventory["hp_median_fit_joint_fingerprints"]),
        "eligible_anchors": list(inventory["eligible_anchors"]),
        "cross_factor_anchors": list(inventory["cross_factor_anchors"]),
        "pair_labels_sha256": stable_hash(compact_pairs),
        "positive_pairs": [
            {key: pair[key] for key in ("left", "right", "cross_architecture", "cross_factor")}
            for pair in inventory["positive_pairs"]
        ],
        "negative_pair_count": len(inventory["negative_pairs"]),
        "negative_pairs_sha256": stable_hash([
            [pair["left"], pair["right"]] for pair in inventory["negative_pairs"]
        ]),
        "ambiguous_pair_count": len(inventory["ambiguous_pairs"]),
        "ambiguous_pairs_sha256": stable_hash([
            [pair["left"], pair["right"]] for pair in inventory["ambiguous_pairs"]
        ]),
        "ambiguous_pairs_enter_loss": False,
        "outer_heldout_labels_used": False,
    }


def execute_formal(
    config_path: Path, output: Path, config_hash: str, source_hash: str,
    *, preflight_ready: bool,
) -> int:
    config = load_config(config_path)
    ensure_output_policy(output, preflight_ready=preflight_ready)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    state = initial_state(config_hash, source_hash)
    dump_json(output / "state.json", state)
    inventory: dict[str, Any] = {}
    try:
        dump_json(output / "frozen_protocol.json", {
            "format_version": FORMAT_VERSION,
            "config": config, "config_sha256": config_hash,
            "source_sha256": source_hash,
            "reused_repository_module": {
                "path": str((REPO_ROOT / "analyse/performance_metric_quick_gate.py").resolve()),
                "sha256": sha256_file(REPO_ROOT / "analyse/performance_metric_quick_gate.py"),
                "scope": [
                    "canonical graph parser", "canonical fingerprints",
                    "explicit graph features", "node feature rows", "GIN layer",
                    "weighted kNN primitive",
                ],
                "v2_hp_residualization_or_metric_loss_reused": False,
            },
            "gp_numerical_protocol": GP_NUMERICAL_PROTOCOL,
            "scope_limit": "joint metric geometry and posterior mean only",
            "bo_authorized": False,
            "uncertainty_or_acquisition_validated": False,
        })
        write_git_and_environment(output)
        configurations, inventory, provenance = load_and_aggregate_source(config)
        dump_json(output / "data_inventory.json", inventory)
        dump_json(output / "data_provenance.json", provenance)
        state = update_state(output, state, formal_data_read=True)
        joint_count = int(inventory["unique_joint_configurations"])
        architecture_count = int(inventory["unique_architectures"])
        data_decision = data_gate_decision(
            joint_count, architecture_count, config["decision"],
        )
        if data_decision is not None:
            reason = data_decision["classification"]
            _analysis_placeholders(output, reason)
            dump_json(output / "decision.json", data_decision)
            write_final_report(output, data_decision, inventory)
            update_state(output, state, status="complete", decision=reason)
            write_manifest(output)
            return 0

        split_manifest = build_split_manifest(configurations, int(config["split"]["outer_folds"]))
        dump_json(output / "split_manifest.json", split_manifest)
        state = update_state(output, state, split_sha256=split_manifest["split_sha256"])

        pair_inventories: dict[int, dict[str, Any]] = {}
        pair_artifacts: list[dict[str, Any]] = []
        power_failures: list[dict[str, Any]] = []
        for fold in split_manifest["folds"]:
            fold_id = int(fold["fold"])
            train_ids = set(map(str, fold["train_joint_fingerprints"]))
            train_rows = sorted(
                (dict(row) for row in configurations if str(row["joint_fingerprint"]) in train_ids),
                key=lambda row: str(row["joint_fingerprint"]),
            )
            hp_standardizer = fit_hp_standardizer(train_rows)
            pair_inventory = construct_statistical_pairs(
                train_rows,
                allowed_joint_fingerprints=train_ids,
                hp_standardizer=hp_standardizer,
                hp_median_fit_rows=train_rows,
                equivalence_margin=float(config["triplets"]["equivalence_margin"]),
                far_margin=float(config["triplets"]["far_margin"]),
            )
            pair_inventories[fold_id] = pair_inventory
            artifact = _pair_artifact_record(fold_id, pair_inventory)
            try:
                assert_pair_power(
                    pair_inventory, config["triplets"]["minimum_power_per_outer_fold"],
                )
                artifact["power_gate_passed"] = True
            except UnderpoweredError as error:
                artifact["power_gate_passed"] = False
                artifact["power_gate_error"] = str(error)
                power_failures.append({"fold": fold_id, "error": str(error)})
            pair_artifacts.append(artifact)
        dump_json(output / "pair_inventory.json", {
            "format_version": FORMAT_VERSION,
            "folds": pair_artifacts,
            "power_failures": power_failures,
            "thresholds_relaxed": False,
            "pair_definition": "matched-seed paired 90% t intervals",
            "hp_residualization_used": False,
        })
        if power_failures:
            decision = classify_joint_metric_gate(
                joint_count, architecture_count, False, None, None, None, config["decision"],
            )
            decision["power_failures"] = power_failures
            _analysis_placeholders(output, decision["classification"])
            dump_json(output / "decision.json", decision)
            write_final_report(output, decision, inventory)
            update_state(output, state, status="complete", decision=decision["classification"])
            write_manifest(output)
            return 0

        fold_results: list[dict[str, Any]] = []
        for fold in split_manifest["folds"]:
            fold_id = int(fold["fold"])
            result = evaluate_outer_fold(
                fold, configurations, config, pair_inventories[fold_id],
            )
            fold_results.append(result)
            state = update_state(
                output, state,
                completed_folds=sorted({*state.get("completed_folds", []), fold_id}),
            )
        oof_rows = [row for result in fold_results for row in result["oof_rows"]]
        if len(oof_rows) != joint_count or len({row["joint_fingerprint"] for row in oof_rows}) != joint_count:
            raise ProtocolError("OOF predictions do not cover every joint configuration exactly once")
        geometry = aggregate_geometry(fold_results)
        prediction_summary = {
            model: prediction_metrics(oof_rows, model, 0.20) for model in EVALUATED_MODELS
        }
        batch_manifest, batch_results = best_of_four_evaluation(
            oof_rows, config, EVALUATED_MODELS,
        )
        decision = classify_joint_metric_gate(
            joint_count, architecture_count, True, geometry,
            prediction_summary, batch_results, config["decision"],
        )
        decision["prediction_metrics"] = prediction_summary
        decision["gp_numerical_protocol"] = GP_NUMERICAL_PROTOCOL
        decision["old_z_search_status"] = "OLD_Z_SEARCH_UNAVAILABLE"
        decision["shuffle_controls_are_diagnostic"] = True

        fold_metric_rows: list[dict[str, Any]] = []
        for result in fold_results:
            fold_id = int(result["fold"])
            for model in EVALUATED_MODELS:
                metrics = prediction_metrics(result["oof_rows"], model, 0.20)
                fold_metric_rows.append({
                    "fold": fold_id, "model": model,
                    **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
                    "mean_simple_regret": batch_results["models"][model]["per_fold_mean_simple_regret"][str(fold_id)],
                })
        dump_json(output / "metric_training_audit.json", {
            "format_version": FORMAT_VERSION,
            "folds": [
                {
                    "fold": result["fold"],
                    "metric_models": result["metric_training_audits"],
                    "direct_regressor_models": result["direct_regressor_audits"],
                    "baseline_gp": result["gp_audits"],
                    "local_gp": result["local_gp_audits"],
                    "hp_standardizer": result["hp_standardizer"],
                    "metric_seed_aggregation": result["metric_seed_aggregation"],
                    "best_metric_seed_selected": result["best_metric_seed_selected"],
                }
                for result in fold_results
            ],
            "outer_heldout_labels_used_for_training": False,
            "metric_seed_aggregation": "arithmetic_mean_all_seeds_0_1_2",
            "best_metric_seed_selected": False,
            "hp_residualization_used": False,
        })
        dump_json(output / "geometry_metrics.json", geometry)
        dump_csv(output / "metrics_per_fold.csv", fold_metric_rows)
        dump_csv(output / "oof_predictions.csv", oof_rows)
        dump_json(output / "best_of_4_batches.json", batch_manifest)
        dump_json(output / "best_of_4_results.json", batch_results)
        dump_json(output / "branch_ablation.json", branch_ablation_summary(prediction_summary, batch_results))
        dump_json(output / "shuffle_controls.json", shuffle_control_summary(
            prediction_summary, batch_results, int(config["controls"]["shuffle_seed"]),
        ))
        dump_json(output / "decision.json", decision)
        write_final_report(output, decision, inventory)
        update_state(output, state, status="complete", decision=decision["classification"])
        write_manifest(output)
        return 0
    except BaseException as error:
        message = f"{type(error).__name__}: {error}"
        write_failure_bundle(output, message, config_hash, source_hash, type(error).__name__)
        raise


def dry_run(config_path: Path, output: Path, config_hash: str, source_hash: str) -> int:
    config = load_config(config_path)
    report = {
        "status": "DRY_RUN_OK",
        "config_sha256": config_hash, "source_sha256": source_hash,
        "formal_result_root": str(output),
        "formal_result_root_created": False,
        "formal_data_read": False,
        "planned_source_root": str(config["data"]["source_root"]),
        "planned_result_shards": [
            "phase_b/initial/gpu0/seed_results.jsonl",
            "phase_b/initial/gpu1/seed_results.jsonl",
        ],
        "recursive_result_scan": False,
        "formal_gnn_training_started": False,
        "test_data_accessed": False, "decoder_used": False,
        "acquisition_used": False,
        "dependencies": dependency_inventory(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _synthetic_geometry(passed: bool = True) -> dict[str, Any]:
    coefficient = 0.50 if passed else 0.0
    return {
        "folds": [
            {
                "fold": fold,
                "seed_distance_gap_spearman": {
                    str(seed): coefficient for seed in REQUIRED_SEEDS
                },
            }
            for fold in range(5)
        ],
        "cross_factor_locality_ratio": 0.60 if passed else 1.0,
        "joint_metric_neighbors": {"equivalent_neighbor_precision_at_5": 0.70 if passed else 0.40},
        "explicit_joint_feature_neighbors": {"equivalent_neighbor_precision_at_5": 0.50},
    }


def _synthetic_predictions(not_joint: bool = False, feature_only: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions: dict[str, dict[str, float]] = {
        "B0": {"mae": 1.00, "top20_precision": 0.50},
        "B1": {"mae": 1.05, "top20_precision": 0.48},
        "B2": {"mae": 0.95, "top20_precision": 0.50},
        "B3": {"mae": 0.79, "top20_precision": 0.61},
        "EJ_KNN": {"mae": 1.00, "top20_precision": 0.50},
        "M1": {"mae": 0.85, "top20_precision": 0.60},
        "M2": {"mae": 0.80, "top20_precision": 0.60},
        "M3": {"mae": 0.82, "top20_precision": 0.58},
        "M4": {"mae": 0.80, "top20_precision": 0.60},
    }
    regret_values = {
        "B0": 1.05, "B1": 1.10, "B2": 1.00, "B3": 0.74,
        "EJ_KNN": 1.00, "M1": 0.80, "M2": 0.75, "M3": 0.78, "M4": 0.75,
    }
    if not_joint:
        predictions["B0"]["top20_precision"] = 0.70
        predictions["B1"]["top20_precision"] = 0.68
        regret_values["B0"] = 0.65
        regret_values["B1"] = 0.68
    if feature_only:
        predictions["M1"] = {"mae": 0.99, "top20_precision": 0.51}
        predictions["M2"] = {"mae": 0.98, "top20_precision": 0.51}
        regret_values["M1"] = 0.99
        regret_values["M2"] = 0.98
        predictions["M4"] = {"mae": 0.80, "top20_precision": 0.60}
        regret_values["M4"] = 0.75
    best = {"models": {}}
    for model, regret in regret_values.items():
        best["models"][model] = {
            "mean_simple_regret": regret,
            "per_fold_mean_simple_regret": {str(fold): regret for fold in range(5)},
        }
    return predictions, best


def synthetic_smoke(config_path: Path) -> int:
    config = load_config(config_path)
    predictions, best = _synthetic_predictions()
    pass_decision = classify_joint_metric_gate(
        336, 192, True, _synthetic_geometry(True), predictions, best, config["decision"],
    )
    reject_decision = classify_joint_metric_gate(
        336, 192, True, _synthetic_geometry(False), predictions, best, config["decision"],
    )
    underpowered_decision = classify_joint_metric_gate(
        336, 192, False, None, None, None, config["decision"],
    )
    not_joint_predictions, not_joint_best = _synthetic_predictions(not_joint=True)
    not_joint_decision = classify_joint_metric_gate(
        336, 192, True, _synthetic_geometry(True),
        not_joint_predictions, not_joint_best, config["decision"],
    )
    feature_predictions, feature_best = _synthetic_predictions(feature_only=True)
    feature_decision = classify_joint_metric_gate(
        336, 192, True, _synthetic_geometry(True),
        feature_predictions, feature_best, config["decision"],
    )
    positive_left = {
        "validation_seed_0": 0.801, "validation_seed_1": 0.811, "validation_seed_2": 0.821,
    }
    positive_right = {
        "validation_seed_0": 0.800, "validation_seed_1": 0.810, "validation_seed_2": 0.820,
    }
    negative_right = {
        "validation_seed_0": 0.780, "validation_seed_1": 0.790, "validation_seed_2": 0.800,
    }
    ambiguous_right = {
        "validation_seed_0": 0.793, "validation_seed_1": 0.803, "validation_seed_2": 0.813,
    }
    gp_protocol_ok = False
    model_protocol_ok = False
    try:
        import numpy as np
        import torch
        _seed_everything(0, torch)
        metric_class, _regressor_class = _joint_model_classes(torch)
        model = metric_class()
        rows = []
        for index in range(4):
            architecture = {
                "operations": [OPERATION_ORDER[index], OPERATION_ORDER[(index + 1) % len(OPERATION_ORDER)]],
                "edges": [[0, 1], [1, 2], [2, 3], [0, 2]],
            }
            rows.append({
                **architecture,
                "joint_fingerprint": f"synthetic-{index}",
                "raw_hp": {"lr": 0.001 * (index + 1), "dropout": 0.1 * index, "hidden_dim": 16 * (2 ** index), "l2": 0.0001 * (index + 1)},
            })
        standardizer = fit_hp_standardizer(rows)
        with torch.no_grad():
            embedding, components = model(
                _graph_tensors(rows, torch),
                torch.tensor(transform_hp(rows, standardizer), dtype=torch.float32),
                "full_joint", True,
            )
        model_protocol_ok = bool(
            embedding.shape == (4, 8)
            and all(component.shape == (4, 16) for component in components)
            and np.allclose(np.linalg.norm(embedding.numpy(), axis=1), 1.0, atol=1e-5)
            and fusion_contract_probe(np.ones((1, 16)), np.full((1, 16), 2.0)).shape == (1, 48)
        )
        x = np.asarray([
            np.roll(np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), index % 8)
            for index in range(12)
        ], dtype=np.float64)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        y = np.linspace(0.70, 0.81, 12)
        gp_prediction, gp_audit = exact_matern_gp_predict(
            x, y, np.full(12, 0.002), x[:2],
            standardize_inputs=False, ard=False, protocol_name="M2",
        )
        gp_protocol_ok = bool(
            np.isfinite(gp_prediction).all()
            and gp_audit["input_standardized"] is False
            and gp_audit["ard"] is False
            and gp_audit["lengthscale_count"] == 1
        )
    except (ImportError, ProtocolError, GPProtocolUnavailable):
        gp_protocol_ok = False
        model_protocol_ok = False
    result = {
        "synthetic_pass_case": pass_decision["classification"] == "JOINT_METRIC_GP_READY",
        "synthetic_reject_case": reject_decision["classification"] == "JOINT_METRIC_REJECTED",
        "synthetic_underpowered_case": underpowered_decision["classification"] == "JOINT_METRIC_UNDERPOWERED",
        "synthetic_not_joint_case": not_joint_decision["classification"] == "JOINT_METRIC_NOT_JOINT",
        "M4_cannot_replace_M2": feature_decision["classification"] == "JOINT_METRIC_FEATURE_ONLY",
        "matched_seed_positive": classify_pair(positive_left, positive_right)["label"] == "positive",
        "matched_seed_negative": classify_pair(positive_left, negative_right)["label"] == "negative",
        "matched_seed_ambiguous": classify_pair(positive_left, ambiguous_right)["label"] == "ambiguous",
        "joint_model_contract": model_protocol_ok,
        "M2_isotropic_unstandardized_gp": gp_protocol_ok,
    }
    result["passed"] = all(result.values())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-sha256")
    parser.add_argument("--source-sha256")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--preflight-ready", action="store_true")
    parser.add_argument("--write-failure-bundle")
    parser.add_argument("--refresh-manifest", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config_hash = args.config_sha256 or sha256_file(config_path)
    source_hash = args.source_sha256 or "standalone"
    if args.write_failure_bundle is not None:
        output = (
            args.output
            or (REPO_ROOT / "results/posthoc/joint-configuration-metric-gate-global4-v1")
        ).resolve()
        write_failure_bundle(
            output, str(args.write_failure_bundle), config_hash, source_hash,
            "LAUNCHER_FAILURE",
        )
        return 0
    if args.refresh_manifest:
        output = (
            args.output
            or (REPO_ROOT / "results/posthoc/joint-configuration-metric-gate-global4-v1")
        ).resolve()
        write_manifest(output)
        return 0
    if args.synthetic_smoke:
        return synthetic_smoke(config_path)
    config = load_config(config_path)
    output = (args.output or (REPO_ROOT / str(config["output"]["result_root"]))).resolve()
    if args.dry_run:
        return dry_run(config_path, output, config_hash, source_hash)
    return execute_formal(
        config_path, output, config_hash, source_hash,
        preflight_ready=bool(args.preflight_ready),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
