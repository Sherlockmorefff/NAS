#!/usr/bin/env python3
"""Offline performance-metric quick gate over validation-only NAS records.

The formal entry point reads only the two explicitly frozen result roots in the
v2 protocol.  Heavy scientific dependencies are imported lazily so that
``--dry-run`` and protocol-focused tests remain CPU-safe and data-free.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/performance_metric_quick_gate_v2.yaml"
FORMAT_VERSION = "performance_metric_quick_gate_global4_v2"
ALLOWED_DECISIONS = {
    "INSUFFICIENT_DATA",
    "QUICK_GATE_UNDERPOWERED",
    "QUICK_GATE_REJECTED",
    "QUICK_GATE_METRIC_ONLY",
    "QUICK_GATE_INCONCLUSIVE",
    "QUICK_GATE_GP_PASSED",
    "RUN_FAILED",
}
OPERATION_ORDER = (
    "GCNConv", "GATConv", "SAGEConv", "GINConv", "GCNII", "Identity",
)
MAX_INTERNAL_NODES = 5
MAX_NODES = 7
REQUIRED_SEEDS = (0, 1, 2)
REQUIRED_RESULT_FIELDS = {
    "pool_id", "architecture_fingerprint", "hp_fingerprint", "seed",
    "joint_config_fingerprint", "raw_hp_vector", "valid", "terminal_status",
    "operations", "edges", "lr", "dropout", "hidden_dim", "l2",
    "dataset", "dataset_content_fingerprint", "split_fingerprint",
    "metric", "eval_epochs", "patience", "training_entry", "val_acc",
    "hp_mode", "normalization", "training_mode", "graph_transform",
}
REQUIRED_OUTPUTS = (
    "state.json", "frozen_protocol.json", "config_sha256.txt",
    "data_inventory.json", "data_provenance.json", "hp_control_report.json",
    "split_manifest.json", "metrics_per_fold.csv", "geometry_metrics.json",
    "oof_predictions.csv", "best_of_4_batches.json", "best_of_4_results.json",
    "decision.json", "final_report.md", "environment.json", "git_status.txt",
    "git_diff.patch", "manifest.json", "logs/run.log",
)


class ProtocolError(RuntimeError):
    """Fail-closed protocol or data violation."""


class GPNoiseProtocolUnavailable(ProtocolError):
    """The required fixed SE-squared plus learned-floor likelihood is absent."""


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
    atomic_write_bytes(path, json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
    ).encode("utf-8") + b"\n")


def dump_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    fieldnames = list(fields or sorted({str(key) for row in rows for key in row}))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    field: json.dumps(row.get(field), sort_keys=True, separators=(",", ":"))
                    if isinstance(row.get(field), (dict, list, tuple)) else row.get(field)
                    for field in fieldnames
                })
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ProtocolError("PyYAML is required to read the frozen protocol") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol_version") != FORMAT_VERSION:
        raise ProtocolError("config protocol_version mismatch")
    if tuple(value["data"]["training_seeds"]) != REQUIRED_SEEDS:
        raise ProtocolError("training seeds must be exactly [0, 1, 2]")
    if tuple(value["metric_encoder"]["seeds"]) != REQUIRED_SEEDS:
        raise ProtocolError("metric model seeds must be exactly [0, 1, 2]")
    if value["metric_encoder"]["output_dim"] != 8 or value["metric_encoder"]["hidden_dim"] != 32:
        raise ProtocolError("metric encoder dimensions differ from the frozen protocol")
    if value["knn"]["k"] != 5 or value["gp"]["kernel"] != "Matern-5/2":
        raise ProtocolError("comparison model protocol mismatch")
    return value


def nested_test_key_hits(value: Any, path: str = "root") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_path = f"{path}.{key}"
            if "test" in str(key).lower():
                hits.append(nested_path)
            hits.extend(nested_test_key_hits(nested, nested_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            hits.extend(nested_test_key_hits(nested, f"{path}[{index}]") )
    return hits


def architecture_fingerprint(config: Mapping[str, Any]) -> str:
    normalized = normalize_architecture(config)
    return stable_hash({
        "operations": normalized["operations"],
        "edges": [list(edge) for edge in sorted(normalized["edges"])],
    })


def hp_fingerprint(raw_hp: Sequence[float]) -> str:
    return stable_hash([float(value) for value in raw_hp])


def normalize_architecture(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ProtocolError("architecture must be a mapping")
    operations = [str(value) for value in config.get("operations", [])]
    if not 1 <= len(operations) <= MAX_INTERNAL_NODES:
        raise ProtocolError(f"architecture operation count outside [1, 5]: {len(operations)}")
    unknown = sorted(set(operations) - set(OPERATION_ORDER))
    if unknown:
        raise ProtocolError(f"unknown operations: {unknown}")
    node_count = len(operations) + 2
    edges = [tuple(int(value) for value in edge) for edge in config.get("edges", [])]
    if not edges or len(edges) != len(set(edges)):
        raise ProtocolError("architecture edges are empty or duplicated")
    if any(len(edge) != 2 or not (0 <= edge[0] < edge[1] < node_count) for edge in edges):
        raise ProtocolError("architecture is not a canonical ordered DAG")
    if not any(source == 0 for source, _ in edges):
        raise ProtocolError("architecture has no input edge")
    if not any(destination == node_count - 1 for _, destination in edges):
        raise ProtocolError("architecture has no output edge")
    return {"operations": operations, "edges": sorted(edges), "node_count": node_count}


def _path_statistics(config: Mapping[str, Any]) -> dict[str, float]:
    normalized = normalize_architecture(config)
    node_count = normalized["node_count"]
    output = node_count - 1
    successors: dict[int, list[int]] = defaultdict(list)
    predecessors: dict[int, list[int]] = defaultdict(list)
    for source, destination in normalized["edges"]:
        successors[source].append(destination)
        predecessors[destination].append(source)
    path_counts = [0] * node_count
    shortest = [math.inf] * node_count
    longest = [-math.inf] * node_count
    path_counts[0] = 1
    shortest[0] = 0
    longest[0] = 0
    topological_depth = [0] * node_count
    for node in range(1, node_count):
        incoming = predecessors.get(node, [])
        if incoming:
            topological_depth[node] = 1 + max(topological_depth[parent] for parent in incoming)
        for parent in incoming:
            path_counts[node] += path_counts[parent]
            if path_counts[parent] > 0:
                shortest[node] = min(shortest[node], shortest[parent] + 1)
                longest[node] = max(longest[node], longest[parent] + 1)
    if path_counts[output] <= 0:
        raise ProtocolError("architecture has no input-output path")
    all_dag_depth = 0
    for node in range(node_count):
        for destination in successors.get(node, []):
            all_dag_depth = max(all_dag_depth, topological_depth[node] + 1)
    in_degrees = [len(predecessors.get(node, [])) for node in range(node_count)]
    out_degrees = [len(successors.get(node, [])) for node in range(node_count)]
    return {
        "path_count": float(path_counts[output]),
        "shortest_path": float(shortest[output]),
        "longest_path": float(longest[output]),
        "dag_depth": float(all_dag_depth),
        "in_degree_mean": sum(in_degrees) / node_count,
        "in_degree_max": float(max(in_degrees)),
        "out_degree_mean": sum(out_degrees) / node_count,
        "out_degree_max": float(max(out_degrees)),
    }


def explicit_graph_features(config: Mapping[str, Any]) -> list[float]:
    normalized = normalize_architecture(config)
    operations = normalized["operations"]
    edge_set = set(normalized["edges"])
    values: list[float] = []
    for source in range(MAX_NODES):
        for destination in range(source + 1, MAX_NODES):
            values.append(float((source, destination) in edge_set))
    padded = operations + [None] * (MAX_INTERNAL_NODES - len(operations))
    for observed in padded:
        values.extend(float(observed == operation) for operation in OPERATION_ORDER)
    counts = Counter(operations)
    values.extend(float(counts[operation]) for operation in OPERATION_ORDER)
    path = _path_statistics(normalized)
    values.extend([
        float(len(edge_set)), path["dag_depth"],
        float(sum(destination - source > 1 for source, destination in edge_set)),
        path["path_count"], path["shortest_path"], path["longest_path"],
        path["in_degree_mean"], path["in_degree_max"],
        path["out_degree_mean"], path["out_degree_max"],
    ])
    return values


def structural_hamming(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = normalize_architecture(left)
    b = normalize_architecture(right)
    a_edges, b_edges = set(a["edges"]), set(b["edges"])
    edge_universe = [
        (source, destination) for source in range(MAX_NODES)
        for destination in range(source + 1, MAX_NODES)
    ]
    edge_mismatch = sum((edge in a_edges) != (edge in b_edges) for edge in edge_universe)
    a_ops = a["operations"] + [None] * (MAX_INTERNAL_NODES - len(a["operations"]))
    b_ops = b["operations"] + [None] * (MAX_INTERNAL_NODES - len(b["operations"]))
    return float(edge_mismatch + sum(x != y for x, y in zip(a_ops, b_ops)))


def node_feature_rows(config: Mapping[str, Any]) -> list[list[float]]:
    normalized = normalize_architecture(config)
    node_count = normalized["node_count"]
    predecessors: dict[int, list[int]] = defaultdict(list)
    for source, destination in normalized["edges"]:
        predecessors[destination].append(source)
    depths = [0] * node_count
    for node in range(1, node_count):
        incoming = predecessors.get(node, [])
        depths[node] = 1 + max((depths[parent] for parent in incoming), default=-1)
    denominator = max(max(depths), 1)
    rows: list[list[float]] = []
    for node in range(node_count):
        operation = normalized["operations"][node - 1] if 1 <= node < node_count - 1 else None
        operation_one_hot = [float(operation == name) for name in OPERATION_ORDER]
        role = [float(node == 0), float(node == node_count - 1), float(0 < node < node_count - 1)]
        rows.append([*operation_one_hot, *role, depths[node] / denominator])
    return rows


def _read_preregistration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    training = value.get("training")
    if not isinstance(training, Mapping):
        raise ProtocolError(f"missing training preregistration: {path}")
    if training.get("track_test") is not False:
        raise ProtocolError(f"track_test is not explicitly false: {path}")
    # Source-code paths (for example tests/test_*.py) and statistical terms
    # such as sign_test are provenance metadata, not held-out test metrics.
    # Restrict this schema firewall to the training protocol; result rows are
    # still scanned recursively and reject every test-named field.
    forbidden = [
        hit for hit in nested_test_key_hits(training, "root.training")
        if not hit.endswith(".track_test")
    ]
    if forbidden:
        raise ProtocolError(f"unexpected test-named training protocol fields: {forbidden[:5]}")
    return dict(value)


def _read_result_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProtocolError(f"missing frozen result file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            hits = nested_test_key_hits(row)
            if hits:
                raise ProtocolError(f"test field rejected at loader layer {path}:{line_number}: {hits[:5]}")
            missing = REQUIRED_RESULT_FIELDS - set(row)
            if missing:
                raise ProtocolError(f"result schema missing {sorted(missing)} at {path}:{line_number}")
            rows.append(dict(row))
    return rows


def _source_protocol_identity(rows: Sequence[Mapping[str, Any]], prereg: Mapping[str, Any]) -> dict[str, Any]:
    training = prereg["training"]
    keys = (
        "dataset_content_fingerprint", "split_fingerprint", "metric", "eval_epochs",
        "patience", "training_entry", "hp_mode", "normalization",
        "training_mode", "graph_transform",
    )
    identity: dict[str, Any] = {
        "dataset": str(training.get("dataset", "")),
        "dataset_content_fingerprint": training.get("dataset_content_fingerprint"),
        "split_fingerprint": training.get("split_fingerprint"),
        "eval_epochs": training.get("max_epochs"),
        "patience": training.get("patience"),
        "track_test": training.get("track_test"),
    }
    for key in keys:
        observed: dict[str, Any] = {}
        for row in rows:
            value = row.get(key)
            observed[stable_hash(value)] = value
        if len(observed) != 1:
            raise ProtocolError(f"source protocol field is inconsistent: {key}")
        identity[key] = next(iter(observed.values()))
    if str(identity["dataset"]).lower() != "cora":
        raise ProtocolError("formal source dataset is not Cora")
    if identity["dataset_content_fingerprint"] != training.get("dataset_content_fingerprint"):
        raise ProtocolError("result/preregistration dataset identity mismatch")
    if identity["split_fingerprint"] != training.get("split_fingerprint"):
        raise ProtocolError("result/preregistration split identity mismatch")
    if int(identity["eval_epochs"]) != int(training.get("max_epochs")):
        raise ProtocolError("result/preregistration epoch mismatch")
    if int(identity["patience"]) != int(training.get("patience")):
        raise ProtocolError("result/preregistration patience mismatch")
    return identity


def load_frozen_source(root: Path, result_relative: str, prereg_relative: str, source_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prereg_path = root / prereg_relative
    result_path = root / result_relative
    prereg = _read_preregistration(prereg_path)
    rows = _read_result_jsonl(result_path)
    identity = _source_protocol_identity(rows, prereg)
    output: list[dict[str, Any]] = []
    duplicate_values: dict[tuple[str, str, int], float] = {}
    for row in rows:
        if str(row["dataset"]).lower() != "cora":
            raise ProtocolError("a result row is not Cora")
        config = normalize_architecture(row)
        computed = architecture_fingerprint(config)
        if computed != str(row["architecture_fingerprint"]):
            raise ProtocolError("architecture canonicalization/fingerprint mismatch")
        if "raw_hp_vector" in row and hp_fingerprint(row["raw_hp_vector"]) != str(row["hp_fingerprint"]):
            raise ProtocolError("HP fingerprint mismatch")
        expected_joint = stable_hash({
            "architecture_fingerprint": str(row["architecture_fingerprint"]),
            "hp_fingerprint": str(row["hp_fingerprint"]),
        })
        if str(row["joint_config_fingerprint"]) != expected_joint:
            raise ProtocolError("joint configuration fingerprint mismatch")
        if row.get("terminal_status") != "success" or row.get("valid") is not True:
            continue
        key = (
            str(row["architecture_fingerprint"]), str(row["hp_fingerprint"]), int(row["seed"]),
        )
        value = float(row["val_acc"])
        if key in duplicate_values:
            if duplicate_values[key] != value:
                raise ProtocolError(f"conflicting duplicate configuration/seed: {key}")
            continue
        duplicate_values[key] = value
        normalized = dict(row)
        normalized["operations"] = config["operations"]
        normalized["edges"] = [list(edge) for edge in config["edges"]]
        normalized["source_id"] = source_id
        normalized["pool_identity"] = f"{source_id}:{row['pool_id']}"
        output.append(normalized)
    provenance = {
        "source_id": source_id,
        "root": str(root.resolve()),
        "result_file": {"path": str(result_path.resolve()), "sha256": sha256_file(result_path)},
        "preregistration_file": {"path": str(prereg_path.resolve()), "sha256": sha256_file(prereg_path)},
        "protocol_identity": identity,
        "loaded_successful_rows": len(output),
    }
    return output, provenance


def complete_seed_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["architecture_fingerprint"]), str(row["hp_fingerprint"]))].append(row)
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        seeds = {int(row["seed"]) for row in values}
        if not set(REQUIRED_SEEDS).issubset(seeds):
            excluded.append({"architecture_fingerprint": key[0], "hp_fingerprint": key[1], "seeds": sorted(seeds)})
            continue
        retained.extend(dict(row) for row in values if int(row["seed"]) in REQUIRED_SEEDS)
    return retained, {
        "required_seeds": list(REQUIRED_SEEDS),
        "retained_joint_configurations": len(grouped) - len(excluded),
        "excluded_joint_configurations": len(excluded),
        "excluded": excluded,
    }


def aggregate_joint_configurations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["architecture_fingerprint"]), str(row["hp_fingerprint"]))].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        by_seed = {int(row["seed"]): row for row in values}
        if set(by_seed) != set(REQUIRED_SEEDS):
            raise ProtocolError(f"aggregation received incomplete seeds for {key}")
        ordered = [by_seed[seed] for seed in REQUIRED_SEEDS]
        invariant_fields = (
            "operations", "edges", "raw_hp_vector", "lr", "dropout",
            "hidden_dim", "l2", "pool_identity", "source_id",
        )
        for field in invariant_fields:
            if len({stable_hash(row[field]) for row in ordered}) != 1:
                raise ProtocolError(f"joint configuration seed invariant mismatch: {field} for {key}")
        labels = [float(row["val_acc"]) for row in ordered]
        mean = sum(labels) / len(labels)
        variance = sum((value - mean) ** 2 for value in labels) / (len(labels) - 1)
        std = math.sqrt(max(variance, 0.0))
        first = ordered[0]
        output.append({
            "architecture_fingerprint": key[0], "hp_fingerprint": key[1],
            "validation_mean": mean, "validation_std": std,
            "standard_error": std / math.sqrt(len(labels)),
            "training_seeds": list(REQUIRED_SEEDS),
            "raw_hp": {
                "lr": float(first["lr"]), "dropout": float(first["dropout"]),
                "hidden_dim": int(first["hidden_dim"]), "l2": float(first["l2"]),
                "normalized_coordinate_vector": [float(value) for value in first["raw_hp_vector"]],
            },
            "operations": list(first["operations"]),
            "edges": [list(edge) for edge in first["edges"]],
            "pool_identity": str(first["pool_identity"]),
            "source_id": str(first["source_id"]),
        })
    return output


def choose_hp_control_route(rows: Sequence[Mapping[str, Any]], minimum: int = 80) -> dict[str, Any]:
    """Choose route A from identity only; this function never reads validation values."""
    coverage: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        coverage[str(row["hp_fingerprint"])].add(str(row["architecture_fingerprint"]))
    eligible = [(len(architectures), hp) for hp, architectures in coverage.items() if len(architectures) >= minimum]
    if eligible:
        best_count = max(count for count, _ in eligible)
        selected = min(hp for count, hp in eligible if count == best_count)
        return {
            "route": "A_FIXED_HP", "selected_hp_fingerprint": selected,
            "selected_coverage": best_count,
            "selection_used_validation": False,
            "coverage": {hp: len(value) for hp, value in sorted(coverage.items())},
        }
    return {
        "route": "B_FOLD_LOCAL_HP_RESIDUALIZATION", "selected_hp_fingerprint": None,
        "selected_coverage": max((len(value) for value in coverage.values()), default=0),
        "selection_used_validation": False,
        "coverage": {hp: len(value) for hp, value in sorted(coverage.items())},
    }


def build_split_manifest(configurations: Sequence[Mapping[str, Any]], fold_count: int = 5) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.model_selection import GroupKFold
    except ImportError as error:
        raise ProtocolError("NumPy and scikit-learn GroupKFold are required for outer splits") from error
    architectures: dict[str, Mapping[str, Any]] = {}
    for row in configurations:
        fingerprint = str(row["architecture_fingerprint"])
        prior = architectures.get(fingerprint)
        if prior is not None and prior["pool_identity"] != row["pool_identity"]:
            raise ProtocolError("one architecture is assigned to multiple pools")
        architectures[fingerprint] = row
    pool_counts = Counter(str(row["pool_identity"]) for row in architectures.values())
    eligible_pool_split = len(pool_counts) >= 5 and all(count >= 8 for count in pool_counts.values())
    split_kind = "pool_identity" if eligible_pool_split else "architecture_fingerprint"
    groups: dict[str, list[str]] = defaultdict(list)
    for fingerprint, row in architectures.items():
        group = str(row["pool_identity"]) if eligible_pool_split else fingerprint
        groups[group].append(fingerprint)
    if len(groups) < fold_count:
        raise ProtocolError(f"only {len(groups)} split groups are available for {fold_count} folds")
    all_fingerprints = sorted(architectures)
    group_vector = np.asarray([
        str(architectures[fingerprint]["pool_identity"]) if eligible_pool_split else fingerprint
        for fingerprint in all_fingerprints
    ], dtype=object)
    splitter = GroupKFold(n_splits=fold_count)
    folds = []
    seen_heldout: set[str] = set()
    for fold, (train_indices, heldout_indices) in enumerate(
        splitter.split(np.zeros((len(all_fingerprints), 1)), groups=group_vector)
    ):
        train = sorted(all_fingerprints[int(index)] for index in train_indices)
        heldout = sorted(all_fingerprints[int(index)] for index in heldout_indices)
        heldout_groups = sorted(set(str(group_vector[int(index)]) for index in heldout_indices))
        if set(train) & set(heldout):
            raise ProtocolError("architecture leakage across outer split")
        seen_heldout.update(heldout)
        folds.append({
            "fold": fold, "group_field": split_kind,
            "heldout_groups": heldout_groups,
            "train_architecture_fingerprints": train,
            "heldout_architecture_fingerprints": heldout,
        })
    if seen_heldout != set(all_fingerprints):
        raise ProtocolError("outer split does not cover every architecture exactly once")
    manifest = {
        "format_version": FORMAT_VERSION,
        "split_kind": split_kind,
        "implementation": "sklearn.model_selection.GroupKFold",
        "fold_count": fold_count,
        "pool_counts": dict(sorted(pool_counts.items())),
        "folds": folds,
    }
    manifest["split_sha256"] = stable_hash(manifest)
    return manifest


def assert_configuration_split_integrity(
    configurations: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    assignment: dict[str, int] = {}
    for fold in manifest["folds"]:
        for fingerprint in fold["heldout_architecture_fingerprints"]:
            if fingerprint in assignment:
                raise ProtocolError("architecture appears in multiple held-out folds")
            assignment[str(fingerprint)] = int(fold["fold"])
    for row in configurations:
        fingerprint = str(row["architecture_fingerprint"])
        if fingerprint not in assignment:
            raise ProtocolError("configuration architecture is absent from split manifest")


def hp_feature_vector(row: Mapping[str, Any]) -> list[float]:
    raw = row["raw_hp"]
    lr, dropout = float(raw["lr"]), float(raw["dropout"])
    hidden, l2 = int(raw["hidden_dim"]), float(raw["l2"])
    if lr <= 0 or l2 <= 0 or hidden <= 0:
        raise ProtocolError("HP residualization received a non-positive log-domain value")
    return [math.log10(lr), dropout, math.log2(hidden), math.log10(l2)]


def _inverse_variance_aggregate(rows: Sequence[Mapping[str, Any]], target_field: str) -> dict[str, Any]:
    if not rows:
        raise ProtocolError("cannot aggregate an empty architecture")
    weights = [1.0 / (float(row["standard_error"]) ** 2 + 1e-8) for row in rows]
    total = sum(weights)
    target = sum(weight * float(row[target_field]) for weight, row in zip(weights, rows)) / total
    raw_mean = sum(weight * float(row["validation_mean"]) for weight, row in zip(weights, rows)) / total
    first = rows[0]
    return {
        "architecture_fingerprint": str(first["architecture_fingerprint"]),
        "target": target,
        "standard_error": math.sqrt(1.0 / total),
        "raw_validation_mean": raw_mean,
        "operations": list(first["operations"]),
        "edges": [list(edge) for edge in first["edges"]],
        "pool_identity": str(first["pool_identity"]),
        "configuration_count": len(rows),
    }


def fixed_hp_architecture_targets(configurations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in configurations:
        grouped[str(row["architecture_fingerprint"])].append(row)
    output = []
    for _fingerprint, rows in sorted(grouped.items()):
        if len(rows) != 1:
            raise ProtocolError("fixed HP route has duplicate architecture configurations")
        row = rows[0]
        output.append({
            "architecture_fingerprint": str(row["architecture_fingerprint"]),
            "target": float(row["validation_mean"]),
            "standard_error": float(row["standard_error"]),
            "raw_validation_mean": float(row["validation_mean"]),
            "operations": list(row["operations"]),
            "edges": [list(edge) for edge in row["edges"]],
            "pool_identity": str(row["pool_identity"]),
            "configuration_count": 1,
        })
    return output


def fold_local_hp_residuals(
    configurations: Sequence[Mapping[str, Any]],
    train_fingerprints: set[str],
    heldout_fingerprints: set[str],
    alphas: Sequence[float],
    group_folds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    try:
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ProtocolError("NumPy and scikit-learn are required for fold-local HP residualization") from error
    if train_fingerprints & heldout_fingerprints:
        raise ProtocolError("HP residualization train/held-out architecture overlap")
    train_rows = [row for row in configurations if row["architecture_fingerprint"] in train_fingerprints]
    heldout_rows = [row for row in configurations if row["architecture_fingerprint"] in heldout_fingerprints]
    if {row["architecture_fingerprint"] for row in train_rows} - train_fingerprints:
        raise ProtocolError("HP Ridge received a foreign training architecture")
    x = np.asarray([hp_feature_vector(row) for row in train_rows], dtype=np.float64)
    y = np.asarray([float(row["validation_mean"]) for row in train_rows], dtype=np.float64)
    groups = np.asarray([str(row["architecture_fingerprint"]) for row in train_rows], dtype=object)
    unique_groups = sorted(set(groups.tolist()))
    folds = min(int(group_folds), len(unique_groups))
    if folds < 2:
        raise ProtocolError("HP Ridge needs at least two architecture groups")
    splitter = GroupKFold(n_splits=folds)
    alpha_rows: list[dict[str, Any]] = []
    predictions_by_alpha: dict[float, Any] = {}
    for alpha in [float(value) for value in alphas]:
        predictions = np.full(len(train_rows), np.nan, dtype=np.float64)
        fold_losses = []
        for fit_indices, stop_indices in splitter.split(x, y, groups):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[fit_indices], y[fit_indices])
            fold_prediction = model.predict(x[stop_indices])
            predictions[stop_indices] = fold_prediction
            fold_losses.append(float(np.mean(np.square(fold_prediction - y[stop_indices]))))
        if not np.isfinite(predictions).all():
            raise ProtocolError("cross-fitted Ridge prediction is incomplete")
        score = float(np.mean(fold_losses))
        predictions_by_alpha[alpha] = predictions
        alpha_rows.append({"alpha": alpha, "mean_group_fold_mse": score})
    selected = min(alpha_rows, key=lambda row: (row["mean_group_fold_mse"], row["alpha"]))
    alpha = float(selected["alpha"])
    cross_fitted = predictions_by_alpha[alpha]
    final_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    final_model.fit(x, y)
    heldout_x = np.asarray([hp_feature_vector(row) for row in heldout_rows], dtype=np.float64)
    heldout_predictions = final_model.predict(heldout_x)
    residual_train = []
    for row, prediction in zip(train_rows, cross_fitted):
        residual_train.append({**dict(row), "residual_target": float(row["validation_mean"]) - float(prediction)})
    residual_heldout = []
    for row, prediction in zip(heldout_rows, heldout_predictions):
        residual_heldout.append({**dict(row), "residual_target": float(row["validation_mean"]) - float(prediction)})
    train_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    heldout_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in residual_train:
        train_grouped[str(row["architecture_fingerprint"])].append(row)
    for row in residual_heldout:
        heldout_grouped[str(row["architecture_fingerprint"])].append(row)
    train_targets = [_inverse_variance_aggregate(rows, "residual_target") for _, rows in sorted(train_grouped.items())]
    heldout_targets = [_inverse_variance_aggregate(rows, "residual_target") for _, rows in sorted(heldout_grouped.items())]
    audit = {
        "selected_alpha": alpha,
        "alpha_selection": alpha_rows,
        "ridge_fit_architecture_fingerprints": sorted(train_fingerprints),
        "outer_heldout_architecture_fingerprints": sorted(heldout_fingerprints),
        "train_prediction_mode": "cross_fitted_group_kfold",
        "heldout_prediction_mode": "final_outer_train_ridge",
        "outer_heldout_used_for_ridge_fit": False,
        "hp_feature_standardization": "fit_within_each_group_fold_then_refit_on_all_outer_train",
        "group_folds": folds,
    }
    return train_targets, heldout_targets, audit


def fold_targets(
    configurations: Sequence[Mapping[str, Any]], hp_report: Mapping[str, Any],
    fold: Mapping[str, Any], config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train = set(fold["train_architecture_fingerprints"])
    heldout = set(fold["heldout_architecture_fingerprints"])
    if hp_report["route"] == "A_FIXED_HP":
        selected = str(hp_report["selected_hp_fingerprint"])
        subset = [row for row in configurations if str(row["hp_fingerprint"]) == selected]
        targets = {row["architecture_fingerprint"]: row for row in fixed_hp_architecture_targets(subset)}
        return (
            [targets[fingerprint] for fingerprint in sorted(train)],
            [targets[fingerprint] for fingerprint in sorted(heldout)],
            {"route": "A_FIXED_HP", "selected_hp_fingerprint": selected, "ridge_used": False},
        )
    return fold_local_hp_residuals(
        configurations, train, heldout, config["hp_control"]["ridge_alphas"],
        int(config["hp_control"]["ridge_group_folds"]),
    )


def construct_metric_pairs(
    architectures: Sequence[Mapping[str, Any]],
    *, tau: float, weight_low: float, weight_high: float,
    weight_normalizer: float | None = None,
    allowed_fingerprints: set[str],
) -> list[dict[str, float | int]]:
    observed = {str(row["architecture_fingerprint"]) for row in architectures}
    if not observed <= allowed_fingerprints:
        raise ProtocolError("outer-held-out label entered metric pair construction")
    output: list[dict[str, float | int]] = []
    for left, right in itertools.combinations(range(len(architectures)), 2):
        a, b = architectures[left], architectures[right]
        gap = abs(float(a["target"]) - float(b["target"]))
        target_distance = min(gap / max(float(tau), 1e-8), 1.0)
        raw_weight = 1.0 / (
            float(a["standard_error"]) ** 2 + float(b["standard_error"]) ** 2 + 1e-8
        )
        clipped = min(max(raw_weight, weight_low), weight_high)
        output.append({"left": left, "right": right, "target_distance": target_distance, "weight": clipped})
    if not output:
        raise ProtocolError("metric pair set is empty")
    mean_weight = (
        sum(float(row["weight"]) for row in output) / len(output)
        if weight_normalizer is None else float(weight_normalizer)
    )
    for row in output:
        row["weight"] = float(row["weight"]) / mean_weight
    return output


def _all_pair_calibration(architectures: Sequence[Mapping[str, Any]]) -> tuple[float, float, float, float]:
    try:
        import numpy as np
    except ImportError as error:
        raise ProtocolError("NumPy is required for metric pair calibration") from error
    gaps: list[float] = []
    weights: list[float] = []
    for left, right in itertools.combinations(range(len(architectures)), 2):
        a, b = architectures[left], architectures[right]
        gaps.append(abs(float(a["target"]) - float(b["target"])))
        weights.append(1.0 / (
            float(a["standard_error"]) ** 2 + float(b["standard_error"]) ** 2 + 1e-8
        ))
    if not gaps:
        raise ProtocolError("outer-train needs at least two architectures")
    low = float(np.quantile(np.asarray(weights), 0.05))
    high = float(np.quantile(np.asarray(weights), 0.95))
    clipped_mean = float(np.mean(np.clip(np.asarray(weights), low, high)))
    return (float(np.quantile(np.asarray(gaps), 0.90)), low, high, clipped_mean)


def _inner_architecture_split(
    architectures: Sequence[Mapping[str, Any]], seed: int, train_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted((dict(row) for row in architectures), key=lambda row: row["architecture_fingerprint"])
    rng = random.Random(int(seed))
    rng.shuffle(ordered)
    validation_count = max(2, int(math.ceil((1.0 - float(train_fraction)) * len(ordered))))
    if len(ordered) - validation_count < 2:
        raise ProtocolError("inner metric split is too small")
    validation = sorted(ordered[:validation_count], key=lambda row: row["architecture_fingerprint"])
    training = sorted(ordered[validation_count:], key=lambda row: row["architecture_fingerprint"])
    if {row["architecture_fingerprint"] for row in training} & {
        row["architecture_fingerprint"] for row in validation
    }:
        raise ProtocolError("inner architecture split leakage")
    return training, validation


def _metric_graph_tensors(rows: Sequence[Mapping[str, Any]], torch: Any) -> list[dict[str, Any]]:
    graphs = []
    for row in rows:
        config = normalize_architecture(row)
        graphs.append({
            "x": torch.tensor(node_feature_rows(config), dtype=torch.float32),
            "edges": torch.tensor(config["edges"], dtype=torch.long),
        })
    return graphs


def _metric_model_classes(torch: Any) -> tuple[Any, Any]:
    nn = torch.nn
    functional = torch.nn.functional

    class GINLayer(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, x: Any, edges: Any, reverse: bool) -> Any:
            source = edges[:, 1] if reverse else edges[:, 0]
            destination = edges[:, 0] if reverse else edges[:, 1]
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, destination, x[source])
            return self.mlp(x + aggregated)

    class DirectedGINMetric(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            input_dim, hidden = 10, 32
            self.forward_one = GINLayer(input_dim, hidden)
            self.forward_two = GINLayer(hidden, hidden)
            self.reverse_one = GINLayer(input_dim, hidden)
            self.reverse_two = GINLayer(hidden, hidden)
            self.output = nn.Sequential(
                nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 8),
            )

        def encode_one(self, graph: Mapping[str, Any]) -> Any:
            x, edges = graph["x"], graph["edges"]
            forward = functional.relu(self.forward_one(x, edges, False))
            forward = self.forward_two(forward, edges, False)
            reverse = functional.relu(self.reverse_one(x, edges, True))
            reverse = self.reverse_two(reverse, edges, True)
            nodes = torch.cat((forward, reverse), dim=-1)
            pooled = torch.cat((nodes.sum(dim=0), nodes.mean(dim=0)), dim=-1)
            return functional.normalize(self.output(pooled), p=2, dim=0)

        def forward(self, graphs: Sequence[Mapping[str, Any]]) -> Any:
            return torch.stack([self.encode_one(graph) for graph in graphs], dim=0)

    return GINLayer, DirectedGINMetric


def _pair_loss(embeddings: Any, pairs: Sequence[Mapping[str, Any]], torch: Any) -> Any:
    functional = torch.nn.functional
    left = torch.tensor([int(row["left"]) for row in pairs], dtype=torch.long)
    right = torch.tensor([int(row["right"]) for row in pairs], dtype=torch.long)
    target = torch.tensor([float(row["target_distance"]) for row in pairs], dtype=torch.float32)
    weights = torch.tensor([float(row["weight"]) for row in pairs], dtype=torch.float32)
    predicted = torch.linalg.vector_norm(embeddings[left] - embeddings[right], dim=1) / 2.0
    losses = functional.smooth_l1_loss(predicted, target, reduction="none", beta=1.0)
    return torch.mean(weights * losses)


def train_metric_encoder(
    outer_train: Sequence[Mapping[str, Any]], outer_heldout: Sequence[Mapping[str, Any]],
    seed: int, metric_config: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ProtocolError("PyTorch and NumPy are required for the directed GIN metric encoder") from error
    if len(outer_train) > 500:
        raise ProtocolError("SPEC_UNDERDEFINED: pair sampling above 500 architectures is not frozen")
    train_fingerprints = {str(row["architecture_fingerprint"]) for row in outer_train}
    heldout_fingerprints = {str(row["architecture_fingerprint"]) for row in outer_heldout}
    if train_fingerprints & heldout_fingerprints:
        raise ProtocolError("metric encoder outer split leakage")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    inner_train, inner_validation = _inner_architecture_split(
        outer_train, int(seed), float(metric_config["inner_train_fraction"]),
    )
    tau, low, high, weight_normalizer = _all_pair_calibration(outer_train)
    fit_pairs = construct_metric_pairs(
        inner_train, tau=tau, weight_low=low, weight_high=high,
        allowed_fingerprints=train_fingerprints, weight_normalizer=weight_normalizer,
    )
    stop_pairs = construct_metric_pairs(
        inner_validation, tau=tau, weight_low=low, weight_high=high,
        allowed_fingerprints=train_fingerprints, weight_normalizer=weight_normalizer,
    )
    _unused, model_class = _metric_model_classes(torch)
    model = model_class()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(metric_config["learning_rate"]),
        weight_decay=float(metric_config["weight_decay"]),
    )
    fit_graphs = _metric_graph_tensors(inner_train, torch)
    stop_graphs = _metric_graph_tensors(inner_validation, torch)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale = 0
    epochs_run = 0
    for epoch in range(1, int(metric_config["max_epochs"]) + 1):
        epochs_run = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _pair_loss(model(fit_graphs), fit_pairs, torch)
        if not torch.isfinite(loss):
            raise ProtocolError("non-finite metric training loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(_pair_loss(model(stop_graphs), stop_pairs, torch))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(metric_config["patience"]):
            break
    if best_state is None:
        raise ProtocolError("metric early stopping produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_embeddings = model(_metric_graph_tensors(outer_train, torch)).cpu().numpy().astype("float64")
        heldout_embeddings = model(_metric_graph_tensors(outer_heldout, torch)).cpu().numpy().astype("float64")
    audit = {
        "seed": int(seed), "tau_outer_train_p90": tau,
        "reliability_weight_p05": low, "reliability_weight_p95": high,
        "reliability_weight_outer_train_clipped_mean": weight_normalizer,
        "outer_train_architecture_fingerprints": sorted(train_fingerprints),
        "outer_heldout_architecture_fingerprints": sorted(heldout_fingerprints),
        "inner_train_architecture_fingerprints": sorted(row["architecture_fingerprint"] for row in inner_train),
        "inner_validation_architecture_fingerprints": sorted(row["architecture_fingerprint"] for row in inner_validation),
        "metric_pair_architecture_fingerprints": sorted(train_fingerprints),
        "outer_heldout_labels_used_for_pairs": False,
        "outer_heldout_labels_used_for_early_stopping": False,
        "fit_pair_count": len(fit_pairs), "early_stop_pair_count": len(stop_pairs),
        "best_epoch": best_epoch, "epochs_run": epochs_run,
        "best_inner_validation_pair_loss": best_loss,
    }
    return train_embeddings, heldout_embeddings, audit


def weighted_knn_predict(
    train_x: Any, train_y: Any, query_x: Any, k: int = 5,
    *, train_fingerprints: Sequence[str] | None = None,
    query_fingerprints: Sequence[str] | None = None,
) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise ProtocolError("NumPy is required for weighted kNN") from error
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    query_x = np.asarray(query_x, dtype=np.float64)
    if train_fingerprints is not None and query_fingerprints is not None:
        if set(map(str, train_fingerprints)) & set(map(str, query_fingerprints)):
            raise ProtocolError("weighted kNN query self-match is forbidden")
    if len(train_x) < int(k):
        raise ProtocolError(f"weighted kNN needs at least {k} references")
    output = []
    for query in query_x:
        distances = np.linalg.norm(train_x - query[None, :], axis=1)
        indices = np.argsort(distances, kind="stable")[: int(k)]
        weights = 1.0 / (distances[indices] + 1e-8)
        output.append(float(np.dot(weights, train_y[indices]) / weights.sum()))
    return np.asarray(output, dtype=np.float64)


def average_metric_seed_predictions(predictions: Mapping[int, Any]) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise ProtocolError("NumPy is required for metric-seed aggregation") from error
    if set(predictions) != set(REQUIRED_SEEDS):
        raise ProtocolError("metric predictions must contain exactly seeds 0, 1, and 2")
    return np.mean(np.stack([np.asarray(predictions[seed]) for seed in REQUIRED_SEEDS], axis=0), axis=0)


def _standardize_train_query(train_x: Any, query_x: Any) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    train = np.asarray(train_x, dtype=np.float64)
    query = np.asarray(query_x, dtype=np.float64)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (train - mean) / scale, (query - mean) / scale, {
        "mean": mean.tolist(), "scale": scale.tolist(), "fit_scope": "outer_train_only",
    }


def gp_protocol_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    gp = config["gp"]
    return {
        key: gp[key] for key in (
            "mean", "kernel", "ard", "likelihood", "marginal_likelihood",
            "optimizer", "learning_rate", "optimization_steps", "restart_seed",
            "learned_noise_floor_initial", "learned_noise_floor_lower_bound",
            "cholesky_jitter", "standardize_features_from_outer_train",
            "standardize_target_from_outer_train",
        )
    }


def exact_matern_gp_predict(
    train_x: Any, train_y: Any, train_se: Any, query_x: Any,
    gp_config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
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
        raise GPNoiseProtocolUnavailable(
            "GP_NOISE_PROTOCOL_UNAVAILABLE: GPyTorch fixed-noise Exact GP is unavailable"
        ) from error
    required = {
        "likelihood": "fixed_se_squared_plus_learned_noise_floor",
        "kernel": "Matern-5/2", "mean": "ConstantMean",
    }
    if any(gp_config.get(key) != value for key, value in required.items()):
        raise GPNoiseProtocolUnavailable("GP_NOISE_PROTOCOL_UNAVAILABLE: frozen GP protocol mismatch")
    train_x_scaled, query_x_scaled, x_transform = _standardize_train_query(train_x, query_x)
    y = np.asarray(train_y, dtype=np.float64)
    se = np.asarray(train_se, dtype=np.float64)
    y_mean = float(y.mean())
    y_scale = max(float(y.std()), 1e-8)
    y_scaled = (y - y_mean) / y_scale
    fixed_noise = np.square(se / y_scale)
    torch.manual_seed(int(gp_config["restart_seed"]))
    tx = torch.as_tensor(train_x_scaled, dtype=torch.double)
    ty = torch.as_tensor(y_scaled, dtype=torch.double)
    tq = torch.as_tensor(query_x_scaled, dtype=torch.double)
    tnoise = torch.as_tensor(fixed_noise, dtype=torch.double)
    # GPyTorch otherwise rounds double fixed noise below 1e-6.  The learned
    # positive floor already supplies the numerical diagonal, so retain SE^2
    # exactly and fail if the installed likelihood cannot do so.
    with gpytorch.settings.min_fixed_noise(
        float_value=0.0, double_value=0.0, half_value=0.0,
    ):
        likelihood = FixedNoiseGaussianLikelihood(
            noise=tnoise,
            learn_additional_noise=True,
            noise_constraint=GreaterThan(float(gp_config["learned_noise_floor_lower_bound"])),
        ).double()
    if not torch.equal(likelihood.noise_covar.noise, tnoise):
        raise GPNoiseProtocolUnavailable(
            "GP_NOISE_PROTOCOL_UNAVAILABLE: fixed SE-squared noise was altered"
        )
    likelihood.second_noise = float(gp_config["learned_noise_floor_initial"])

    class FrozenExactGP(gpytorch.models.ExactGP):
        def __init__(self) -> None:
            super().__init__(tx, ty, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=tx.shape[1]))

        def forward(self, x: Any) -> Any:
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x),
            )

    model = FrozenExactGP().double()
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(gp_config["learning_rate"]))
    mll = ExactMarginalLogLikelihood(likelihood, model)
    with gpytorch.settings.cholesky_jitter(float(gp_config["cholesky_jitter"])):
        for _step in range(int(gp_config["optimization_steps"])):
            optimizer.zero_grad(set_to_none=True)
            loss = -mll(model(tx), ty)
            if not torch.isfinite(loss):
                raise ProtocolError("non-finite Exact GP marginal likelihood")
            loss.backward()
            optimizer.step()
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(float(gp_config["cholesky_jitter"])):
        prediction = model(tq).mean.detach().cpu().numpy() * y_scale + y_mean
    floor = float(likelihood.second_noise.detach().cpu().reshape(-1)[0])
    return prediction.astype(np.float64), {
        "protocol": gp_protocol_signature({"gp": gp_config}),
        "feature_transform": x_transform,
        "target_mean": y_mean, "target_scale": y_scale,
        "fixed_observation_noise_rule": "standard_error_squared",
        "gpytorch_min_fixed_noise_override": 0.0,
        "learned_noise_floor_standardized_target_units": floor,
        "learned_noise_floor_original_target_units": floor * y_scale * y_scale,
        "posterior_variance_used": False,
    }


def _spearman(values_a: Any, values_b: Any) -> float:
    try:
        import numpy as np
        from scipy.stats import spearmanr
    except ImportError as error:
        raise ProtocolError("SciPy and NumPy are required for Spearman geometry metrics") from error
    result = float(spearmanr(np.asarray(values_a), np.asarray(values_b)).statistic)
    return result if math.isfinite(result) else 0.0


def evaluate_outer_fold(
    fold: Mapping[str, Any], configurations: Sequence[Mapping[str, Any]],
    hp_report: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    train_rows, heldout_rows, hp_audit = fold_targets(configurations, hp_report, fold, config)
    train_rows = sorted(train_rows, key=lambda row: row["architecture_fingerprint"])
    heldout_rows = sorted(heldout_rows, key=lambda row: row["architecture_fingerprint"])
    train_y = np.asarray([row["target"] for row in train_rows], dtype=np.float64)
    heldout_y = np.asarray([row["target"] for row in heldout_rows], dtype=np.float64)
    train_se = np.asarray([row["standard_error"] for row in train_rows], dtype=np.float64)
    graph_train = np.asarray([explicit_graph_features(row) for row in train_rows], dtype=np.float64)
    graph_heldout = np.asarray([explicit_graph_features(row) for row in heldout_rows], dtype=np.float64)
    expected_gp_protocol = gp_protocol_signature(config)
    if expected_gp_protocol != gp_protocol_signature(config):
        raise ProtocolError("graph GP and metric GP protocols differ")
    b1_prediction, b1_audit = exact_matern_gp_predict(
        graph_train, train_y, train_se, graph_heldout, config["gp"],
    )
    seed_knn: dict[int, Any] = {}
    seed_gp: dict[int, Any] = {}
    seed_distances: list[Any] = []
    metric_audits: list[dict[str, Any]] = []
    seed_spearman: dict[str, float] = {}
    gaps = np.abs(heldout_y[:, None] - train_y[None, :])
    for seed in REQUIRED_SEEDS:
        train_embedding, heldout_embedding, metric_audit = train_metric_encoder(
            train_rows, heldout_rows, seed, config["metric_encoder"],
        )
        distances = np.linalg.norm(
            heldout_embedding[:, None, :] - train_embedding[None, :, :], axis=2,
        )
        seed_distances.append(distances)
        seed_spearman[str(seed)] = _spearman(distances.reshape(-1), gaps.reshape(-1))
        seed_knn[seed] = weighted_knn_predict(
            train_embedding, train_y, heldout_embedding, int(config["knn"]["k"]),
            train_fingerprints=[row["architecture_fingerprint"] for row in train_rows],
            query_fingerprints=[row["architecture_fingerprint"] for row in heldout_rows],
        )
        gp_prediction, gp_audit = exact_matern_gp_predict(
            train_embedding, train_y, train_se, heldout_embedding, config["gp"],
        )
        if gp_audit["protocol"] != b1_audit["protocol"]:
            raise ProtocolError("graph GP and metric GP use different GP protocols")
        seed_gp[seed] = gp_prediction
        metric_audits.append({**metric_audit, "gp": gp_audit})
    # Arithmetic means across all frozen metric seeds; no seed is selected.
    m1_prediction = average_metric_seed_predictions(seed_knn)
    m2_prediction = average_metric_seed_predictions(seed_gp)
    metric_distance = np.mean(np.stack(seed_distances, axis=0), axis=0)
    train_structure = []
    train_performance_gap = []
    for left, right in itertools.combinations(range(len(train_rows)), 2):
        train_structure.append(structural_hamming(train_rows[left], train_rows[right]))
        train_performance_gap.append(abs(train_y[left] - train_y[right]))
    structure_threshold = float(np.quantile(np.asarray(train_structure), 0.50))
    close_threshold = float(np.quantile(np.asarray(train_performance_gap), 0.20))
    far_threshold = float(np.quantile(np.asarray(train_performance_gap), 0.80))
    cross_close: list[float] = []
    cross_far: list[float] = []
    for query_index, query in enumerate(heldout_rows):
        for reference_index, reference in enumerate(train_rows):
            structure = structural_hamming(query, reference)
            gap = abs(heldout_y[query_index] - train_y[reference_index])
            if structure > structure_threshold and gap < close_threshold:
                cross_close.append(float(metric_distance[query_index, reference_index]))
            if structure > structure_threshold and gap > far_threshold:
                cross_far.append(float(metric_distance[query_index, reference_index]))
    if not cross_close or not cross_far:
        raise ProtocolError("cross-structure metric ratio has an empty heldout/train cell")
    ratio = float(np.median(np.asarray(cross_close)) / max(float(np.median(np.asarray(cross_far))), 1e-12))
    oof_rows = []
    for index, row in enumerate(heldout_rows):
        oof_rows.append({
            "fold": int(fold["fold"]),
            "architecture_fingerprint": row["architecture_fingerprint"],
            "pool_identity": row["pool_identity"],
            "target": float(heldout_y[index]),
            "raw_validation_mean": float(row["raw_validation_mean"]),
            "standard_error": float(row["standard_error"]),
            "B1_prediction": float(b1_prediction[index]),
            "M1_prediction": float(m1_prediction[index]),
            "M2_prediction": float(m2_prediction[index]),
        })
    return {
        "fold": int(fold["fold"]),
        "oof_rows": oof_rows,
        "geometry": {
            "seed_spearman": seed_spearman,
            "structure_far_threshold_train_median": structure_threshold,
            "performance_close_threshold_train_p20": close_threshold,
            "performance_far_threshold_train_p80": far_threshold,
            "cross_close_metric_distances": cross_close,
            "cross_far_metric_distances": cross_far,
            "cross_structure_metric_ratio": ratio,
        },
        "hp_audit": hp_audit,
        "metric_audits": metric_audits,
        "b1_gp_audit": b1_audit,
        "metric_seed_aggregation": "arithmetic_mean_all_seeds_0_1_2",
        "best_metric_seed_selected": False,
    }


def prediction_metrics(rows: Sequence[Mapping[str, Any]], model: str, top_fraction: float = 0.20) -> dict[str, float]:
    if not rows:
        raise ProtocolError("prediction metrics need OOF rows")
    n = len(rows)
    top_count = int(math.ceil(float(top_fraction) * n))
    true_top = {
        row["architecture_fingerprint"] for row in sorted(
            rows, key=lambda row: (-float(row["target"]), str(row["architecture_fingerprint"])),
        )[:top_count]
    }
    predicted_top = {
        row["architecture_fingerprint"] for row in sorted(
            rows, key=lambda row: (-float(row[f"{model}_prediction"]), str(row["architecture_fingerprint"])),
        )[:top_count]
    }
    return {
        "mae": sum(abs(float(row[f"{model}_prediction"]) - float(row["target"])) for row in rows) / n,
        "top20_precision": len(true_top & predicted_top) / top_count,
        "top_count": float(top_count),
    }


def _batch_manifest_for_fold(fold_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[list[str]]:
    fingerprints = sorted(str(row["architecture_fingerprint"]) for row in fold_rows)
    batch_size = int(config["metrics"]["best_of_batch_size"])
    total = math.comb(len(fingerprints), batch_size)
    if total <= int(config["metrics"]["exhaustive_batch_limit"]):
        return [list(batch) for batch in itertools.combinations(fingerprints, batch_size)]
    rng = random.Random(int(config["metrics"]["sampled_batch_seed"]))
    count = int(config["metrics"]["sampled_batch_count"])
    batches: set[tuple[str, ...]] = set()
    while len(batches) < count:
        batches.add(tuple(sorted(rng.sample(fingerprints, batch_size))))
    return [list(batch) for batch in sorted(batches)]


def best_of_four_evaluation(
    oof_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_fold: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in oof_rows:
        by_fold[int(row["fold"])].append(row)
    manifest_folds = []
    result_rows = []
    for fold, rows in sorted(by_fold.items()):
        lookup = {str(row["architecture_fingerprint"]): row for row in rows}
        batches = _batch_manifest_for_fold(rows, config)
        manifest_folds.append({
            "fold": fold, "batch_count": len(batches),
            "mode": "exhaustive" if len(batches) == math.comb(len(rows), 4) else "sampled_seed_0",
            "batches": batches,
        })
        for batch_index, batch in enumerate(batches):
            candidates = [lookup[fingerprint] for fingerprint in batch]
            best_target = max(float(row["target"]) for row in candidates)
            for model in ("B1", "M1", "M2"):
                selected = min(
                    candidates,
                    key=lambda row: (-float(row[f"{model}_prediction"]), str(row["architecture_fingerprint"])),
                )
                result_rows.append({
                    "fold": fold, "batch_index": batch_index, "model": model,
                    "selected_architecture_fingerprint": selected["architecture_fingerprint"],
                    "simple_regret": best_target - float(selected["target"]),
                })
    summary: dict[str, Any] = {"rows": result_rows, "models": {}}
    for model in ("B1", "M1", "M2"):
        model_rows = [row for row in result_rows if row["model"] == model]
        per_fold = {}
        for fold in sorted(by_fold):
            values = [float(row["simple_regret"]) for row in model_rows if int(row["fold"]) == fold]
            per_fold[str(fold)] = sum(values) / len(values)
        summary["models"][model] = {
            "mean_simple_regret": sum(float(row["simple_regret"]) for row in model_rows) / len(model_rows),
            "per_fold_mean_simple_regret": per_fold,
        }
    return {"format_version": FORMAT_VERSION, "folds": manifest_folds}, summary


def _relative_improvement(baseline: float, challenger: float) -> float:
    if baseline <= 0.0:
        return 0.0 if challenger >= baseline else math.inf
    return (baseline - challenger) / baseline


def classify_quick_gate(
    architecture_count: int,
    geometry: Mapping[str, Any] | None,
    predictions: Mapping[str, Mapping[str, float]] | None,
    regret: Mapping[str, Any] | None,
    decision_config: Mapping[str, Any],
) -> dict[str, Any]:
    if architecture_count < int(decision_config["insufficient_below"]):
        return {"classification": "INSUFFICIENT_DATA", "architecture_count": architecture_count}
    if architecture_count < int(decision_config["underpowered_below"]):
        return {"classification": "QUICK_GATE_UNDERPOWERED", "architecture_count": architecture_count}
    if geometry is None or predictions is None or regret is None:
        raise ProtocolError("complete quick-gate metrics are required for N >= 80")
    spearman = [
        float(value) for fold in geometry["folds"]
        for value in fold["seed_spearman"].values()
    ]
    ordered_spearman = sorted(spearman)
    middle = len(ordered_spearman) // 2
    median_spearman = ordered_spearman[middle] if len(ordered_spearman) % 2 else (
        ordered_spearman[middle - 1] + ordered_spearman[middle]
    ) / 2.0
    positive_folds_by_seed = {
        str(seed): sum(float(fold["seed_spearman"][str(seed)]) > 0.0 for fold in geometry["folds"])
        for seed in REQUIRED_SEEDS
    }
    geometry_pass = (
        median_spearman > float(decision_config["geometry_median_spearman_strictly_greater_than"])
        and sum(count >= int(decision_config["geometry_positive_fold_count_per_seed"]) for count in positive_folds_by_seed.values())
        >= int(decision_config["geometry_positive_seed_count"])
        and float(geometry["cross_structure_metric_ratio"]) < float(decision_config["cross_structure_ratio_strictly_less_than"])
    )
    b1 = predictions["B1"]
    regret_models = regret["models"]

    def model_gates(model: str) -> dict[str, Any]:
        point = predictions[model]
        mae_gain = _relative_improvement(float(b1["mae"]), float(point["mae"]))
        precision_gain = float(point["top20_precision"]) - float(b1["top20_precision"])
        regret_gain = _relative_improvement(
            float(regret_models["B1"]["mean_simple_regret"]),
            float(regret_models[model]["mean_simple_regret"]),
        )
        return {
            "mae_relative_improvement": mae_gain,
            "top20_precision_absolute_improvement": precision_gain,
            "best_of_4_regret_relative_improvement": regret_gain,
            "pooled_point_thresholds_passed": (
                mae_gain >= float(decision_config["mae_relative_improvement_at_least"])
                and precision_gain >= float(decision_config["top20_precision_absolute_improvement_at_least"])
                and regret_gain >= float(decision_config["best_of_4_regret_relative_improvement_at_least"])
            ),
        }

    m1 = model_gates("M1")
    m2 = model_gates("M2")
    m2_better_folds = sum(
        float(regret_models["M2"]["per_fold_mean_simple_regret"][str(fold)])
        < float(regret_models["B1"]["per_fold_mean_simple_regret"][str(fold)])
        for fold in range(5)
    )
    m2["better_simple_regret_fold_count"] = m2_better_folds
    m2["fold_consistency_passed"] = m2_better_folds >= int(decision_config["gp_better_fold_count_at_least"])
    m2["all_thresholds_passed"] = bool(m2["pooled_point_thresholds_passed"] and m2["fold_consistency_passed"])
    if not geometry_pass:
        classification = "QUICK_GATE_REJECTED"
    elif m2["all_thresholds_passed"]:
        classification = "QUICK_GATE_GP_PASSED"
    elif m2["pooled_point_thresholds_passed"] and not m2["fold_consistency_passed"]:
        classification = "QUICK_GATE_INCONCLUSIVE"
    elif m1["pooled_point_thresholds_passed"]:
        classification = "QUICK_GATE_METRIC_ONLY"
    else:
        classification = "QUICK_GATE_REJECTED"
    return {
        "classification": classification,
        "architecture_count": architecture_count,
        "geometry": {
            "median_seed_fold_spearman": median_spearman,
            "positive_folds_by_seed": positive_folds_by_seed,
            "cross_structure_metric_ratio": float(geometry["cross_structure_metric_ratio"]),
            "passed": geometry_pass,
        },
        "M1_gates": m1, "M2_gates": m2,
        "metric_seed_aggregation": "arithmetic_mean_all_seeds_0_1_2",
        "cross_model_metric_splicing_used": False,
        "bo_authorized": False,
        "gp_uncertainty_calibration_claimed": False,
    }


def _run_git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise ProtocolError(f"git {' '.join(args)} failed: {result.stdout.strip()}")
    return result.stdout


def write_git_and_environment(output: Path) -> None:
    branch = _run_git(["branch", "--show-current"]).strip()
    head = _run_git(["rev-parse", "HEAD"]).strip()
    status = _run_git(["status", "--short"])
    dump_text(output / "git_status.txt", f"branch: {branch}\nHEAD: {head}\n{status}")
    patch = _run_git(["diff", "--binary", "HEAD"])
    experiment_files = (
        "configs/performance_metric_quick_gate_v2.yaml",
        "analyse/performance_metric_quick_gate.py",
        "scripts/run_performance_metric_quick_gate.sh",
        "tests/test_performance_metric_quick_gate.py",
    )
    for relative in experiment_files:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if tracked:
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "/dev/null", relative],
            cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode not in (0, 1):
            raise ProtocolError(f"cannot capture untracked source patch for {relative}")
        patch += result.stdout
    dump_text(output / "git_diff.patch", patch)
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "gpytorch", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    dump_json(output / "environment.json", {
        "created_at": utc_now(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "packages": packages, "cuda_training_started": False,
        "formal_gnn_training_started": False,
    })


def initial_state(
    config_hash: str, source_hash: str, *, status: str = "running",
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION, "status": status,
        "created_at": utc_now(), "updated_at": utc_now(),
        "config_sha256": config_hash, "source_sha256": source_hash,
        "split_sha256": None, "completed_folds": [],
        "formal_gnn_training_started": False,
        "vae_loaded_or_modified": False, "decoder_used": False,
        "acquisition_used": False, "test_metric_accessed": False,
    }


def update_state(output: Path, state: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    updated = {**dict(state), **changes, "updated_at": utc_now()}
    dump_json(output / "state.json", updated)
    return updated


def _empty_required_analysis_outputs(output: Path, reason: str) -> None:
    dump_json(output / "split_manifest.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": reason})
    dump_csv(output / "metrics_per_fold.csv", [], fields=("fold", "model", "mae"))
    dump_json(output / "geometry_metrics.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": reason})
    dump_csv(output / "oof_predictions.csv", [], fields=(
        "fold", "architecture_fingerprint", "target", "B1_prediction", "M1_prediction", "M2_prediction",
    ))
    dump_json(output / "best_of_4_batches.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": reason})
    dump_json(output / "best_of_4_results.json", {"format_version": FORMAT_VERSION, "models": {}, "reason": reason})


def write_final_report(output: Path, decision: Mapping[str, Any], architecture_count: int) -> None:
    classification = str(decision["classification"])
    report = [
        "# Performance metric quick gate v2",
        "",
        f"- Classification: **{classification}**",
        f"- Usable architectures: {architecture_count}",
        "- Scope: validation-supervised performance geometry and GP posterior mean only.",
        "- GP uncertainty calibration is not claimed.",
        "- Bayesian optimization and acquisition are not authorized.",
        "- No test label or test metric was loaded.",
        "- No candidate GNN or VAE training was run by this program.",
    ]
    dump_text(output / "final_report.md", "\n".join(report) + "\n")


def write_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(output))
        if relative == "manifest.json" or relative.startswith("cache/"):
            continue
        if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            continue
        rows.append({
            "relative_path": relative, "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    dump_json(output / "manifest.json", {
        "format_version": FORMAT_VERSION, "created_at": utc_now(),
        "excludes": ["manifest.json", "cache/", "model weights"],
        "file_count": len(rows), "files": rows,
        "manifest_root_sha256": stable_hash([(row["relative_path"], row["sha256"]) for row in rows]),
    })


def _check_resume_state(
    output: Path, config_hash: str, source_hash: str,
) -> dict[str, Any]:
    path = output / "state.json"
    if not path.is_file():
        raise ProtocolError("--resume requires an existing state.json")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("config_sha256") != config_hash:
        raise ProtocolError("--resume config hash mismatch")
    if state.get("source_sha256") != source_hash:
        raise ProtocolError("--resume source hash mismatch")
    return state


def ensure_output_policy(output: Path, resume: bool) -> None:
    if output.is_dir() and not resume:
        files = [path for path in output.rglob("*") if path.is_file()]
        launcher_log_only = (
            all(str(path.relative_to(output)) == "logs/run.log" for path in files)
            and all(path.stat().st_size == 0 for path in files)
            and not any(path.is_dir() and path.name != "logs" for path in output.iterdir())
        )
        if (files or any(output.iterdir())) and not launcher_log_only:
            raise ProtocolError(f"formal result directory exists and is non-empty: {output}")
    if output.exists() and not output.is_dir():
        raise ProtocolError(f"formal result path is not a directory: {output}")


def _merge_source_rows(sources: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, int], dict[str, Any]] = {}
    for rows in sources:
        for row in rows:
            key = (
                str(row["architecture_fingerprint"]), str(row["hp_fingerprint"]), int(row["seed"]),
            )
            if key in seen:
                if float(seen[key]["val_acc"]) != float(row["val_acc"]):
                    raise ProtocolError(f"conflicting duplicate across formal sources: {key}")
                continue
            seen[key] = dict(row)
    return [seen[key] for key in sorted(seen)]


def _protocols_match(primary: Mapping[str, Any], supplement: Mapping[str, Any]) -> None:
    keys = (
        "dataset", "dataset_content_fingerprint", "split_fingerprint", "metric",
        "eval_epochs", "patience", "training_entry", "track_test", "hp_mode",
        "normalization", "training_mode", "graph_transform",
    )
    mismatches = {key: [primary.get(key), supplement.get(key)] for key in keys if primary.get(key) != supplement.get(key)}
    if mismatches:
        raise ProtocolError(f"supplement protocol identity mismatch: {mismatches}")


def _write_failure_artifacts(
    output: Path, state: Mapping[str, Any], error: BaseException,
    *, architecture_count: int = 0,
) -> None:
    message = f"{type(error).__name__}: {error}"
    if not (output / "data_inventory.json").exists():
        dump_json(output / "data_inventory.json", {"status": "failed_before_inventory", "error": message})
    if not (output / "data_provenance.json").exists():
        dump_json(output / "data_provenance.json", {"status": "failed_before_provenance", "error": message})
    if not (output / "hp_control_report.json").exists():
        dump_json(output / "hp_control_report.json", {"status": "failed_before_hp_control", "error": message})
    if not (output / "split_manifest.json").exists():
        dump_json(output / "split_manifest.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": message})
    if not (output / "metrics_per_fold.csv").exists():
        dump_csv(output / "metrics_per_fold.csv", [], fields=("fold", "model", "mae"))
    if not (output / "geometry_metrics.json").exists():
        dump_json(output / "geometry_metrics.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": message})
    if not (output / "oof_predictions.csv").exists():
        dump_csv(output / "oof_predictions.csv", [], fields=(
            "fold", "architecture_fingerprint", "target", "B1_prediction", "M1_prediction", "M2_prediction",
        ))
    if not (output / "best_of_4_batches.json").exists():
        dump_json(output / "best_of_4_batches.json", {"format_version": FORMAT_VERSION, "folds": [], "reason": message})
    if not (output / "best_of_4_results.json").exists():
        dump_json(output / "best_of_4_results.json", {"format_version": FORMAT_VERSION, "models": {}, "reason": message})
    decision = {
        "classification": "RUN_FAILED", "reason": message,
        "failure_code": "GP_NOISE_PROTOCOL_UNAVAILABLE" if isinstance(error, GPNoiseProtocolUnavailable) else type(error).__name__,
        "bo_authorized": False, "gp_uncertainty_calibration_claimed": False,
    }
    dump_json(output / "decision.json", decision)
    write_final_report(output, decision, architecture_count)
    update_state(output, state, status="failed", error=message)


def execute_formal(
    config_path: Path, output: Path, config_hash: str, source_hash: str, resume: bool,
) -> int:
    config = load_config(config_path)
    ensure_output_policy(output, resume)
    output.mkdir(parents=True, exist_ok=True)
    state = _check_resume_state(output, config_hash, source_hash) if resume else initial_state(config_hash, source_hash)
    if not resume:
        dump_json(output / "state.json", state)
    architecture_count = 0
    try:
        dump_text(output / "config_sha256.txt", config_hash + "\n")
        dump_json(output / "frozen_protocol.json", {
            "config": config,
            "config_sha256": config_hash, "source_sha256": source_hash,
            "scope_limit": "performance_geometry_and_gp_posterior_mean_only",
            "bo_authorized": False, "gp_uncertainty_calibration_claimed": False,
        })
        write_git_and_environment(output)
        data = config["data"]
        primary_root = REPO_ROOT / str(data["primary_root"])
        primary_rows, primary_provenance = load_frozen_source(
            primary_root, str(data["primary_results"]), str(data["primary_preregistration"]), "primary",
        )
        primary_complete, primary_seed_audit = complete_seed_rows(primary_rows)
        primary_unique = len({row["architecture_fingerprint"] for row in primary_complete})
        supplement_used = primary_unique < int(data["supplement_only_below_unique_architectures"])
        source_rows: list[Sequence[Mapping[str, Any]]] = [primary_complete]
        provenances = [primary_provenance]
        supplement_seed_audit = None
        if supplement_used:
            supplement_root = REPO_ROOT / str(data["supplement_root"])
            supplement_rows, supplement_provenance = load_frozen_source(
                supplement_root, str(data["supplement_results"]),
                str(data["supplement_preregistration"]), "supplement",
            )
            _protocols_match(
                primary_provenance["protocol_identity"], supplement_provenance["protocol_identity"],
            )
            supplement_complete, supplement_seed_audit = complete_seed_rows(supplement_rows)
            source_rows.append(supplement_complete)
            provenances.append(supplement_provenance)
        merged = _merge_source_rows(source_rows)
        configurations = aggregate_joint_configurations(merged)
        hp_report = choose_hp_control_route(
            configurations, int(config["hp_control"]["fixed_hp_minimum_unique_architectures"]),
        )
        if hp_report["route"] == "A_FIXED_HP":
            usable_configurations = [
                row for row in configurations
                if row["hp_fingerprint"] == hp_report["selected_hp_fingerprint"]
            ]
        else:
            usable_configurations = configurations
        architecture_count = len({row["architecture_fingerprint"] for row in usable_configurations})
        inventory = {
            "primary_successful_rows": len(primary_rows),
            "primary_complete_seed_rows": len(primary_complete),
            "primary_unique_architectures_after_strict_filter": primary_unique,
            "supplement_used": supplement_used,
            "supplement_seed_audit": supplement_seed_audit,
            "joint_configuration_count": len(configurations),
            "usable_joint_configuration_count": len(usable_configurations),
            "usable_unique_architecture_count": architecture_count,
            "primary_seed_audit": primary_seed_audit,
        }
        dump_json(output / "data_inventory.json", inventory)
        dump_json(output / "data_provenance.json", {
            "sources": provenances,
            "supplement_admission_rule_satisfied": (not supplement_used or primary_unique < 80),
            "architecture_canonicalization": "ordered operations plus sorted edges; position-sensitive",
            "test_fields_in_result_rows": 0, "test_metric_accessed": False,
        })
        dump_json(output / "hp_control_report.json", hp_report)
        if architecture_count < int(config["decision"]["underpowered_below"]):
            early = classify_quick_gate(architecture_count, None, None, None, config["decision"])
            reason = early["classification"]
            _empty_required_analysis_outputs(output, reason)
            dump_json(output / "decision.json", early)
            write_final_report(output, early, architecture_count)
            state = update_state(output, state, status="complete", decision=early["classification"])
            write_manifest(output)
            return 0
        split_manifest = build_split_manifest(usable_configurations, int(config["split"]["outer_folds"]))
        assert_configuration_split_integrity(usable_configurations, split_manifest)
        split_hash = split_manifest["split_sha256"]
        if resume and state.get("split_sha256") not in (None, split_hash):
            raise ProtocolError("--resume split hash mismatch")
        dump_json(output / "split_manifest.json", split_manifest)
        state = update_state(output, state, split_sha256=split_hash, status="running")
        cache_root = output / str(config["output"]["cache_directory"])
        cache_root.mkdir(parents=True, exist_ok=True)
        fold_results = []
        for fold in split_manifest["folds"]:
            cache_path = cache_root / f"fold_{int(fold['fold'])}.json"
            if resume and cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_identity") != {
                    "config_sha256": config_hash, "source_sha256": source_hash,
                    "split_sha256": split_hash, "fold": int(fold["fold"]),
                }:
                    raise ProtocolError("--resume fold cache identity mismatch")
                result = cached["result"]
            else:
                result = evaluate_outer_fold(fold, usable_configurations, hp_report, config)
                dump_json(cache_path, {
                    "cache_identity": {
                        "config_sha256": config_hash, "source_sha256": source_hash,
                        "split_sha256": split_hash, "fold": int(fold["fold"]),
                    },
                    "result": result,
                })
            fold_results.append(result)
            completed = sorted({*state.get("completed_folds", []), int(fold["fold"])})
            state = update_state(output, state, completed_folds=completed)
        oof_rows = [row for result in fold_results for row in result["oof_rows"]]
        if len(oof_rows) != architecture_count or len({row["architecture_fingerprint"] for row in oof_rows}) != architecture_count:
            raise ProtocolError("pooled OOF predictions are incomplete or duplicated")
        close_values = [value for result in fold_results for value in result["geometry"]["cross_close_metric_distances"]]
        far_values = [value for result in fold_results for value in result["geometry"]["cross_far_metric_distances"]]
        import numpy as np
        pooled_ratio = float(np.median(np.asarray(close_values)) / max(float(np.median(np.asarray(far_values))), 1e-12))
        geometry = {
            "folds": [
                {
                    "fold": result["fold"],
                    "seed_spearman": result["geometry"]["seed_spearman"],
                    "cross_structure_metric_ratio": result["geometry"]["cross_structure_metric_ratio"],
                }
                for result in fold_results
            ],
            "cross_structure_metric_ratio": pooled_ratio,
            "cross_structure_aggregation": "pooled_heldout_query_to_outer_train_pair_medians",
        }
        prediction_summary = {
            model: prediction_metrics(oof_rows, model, float(config["metrics"]["top_fraction"]))
            for model in ("B1", "M1", "M2")
        }
        batch_manifest, batch_results = best_of_four_evaluation(oof_rows, config)
        decision = classify_quick_gate(
            architecture_count, geometry, prediction_summary, batch_results, config["decision"],
        )
        fold_metric_rows = []
        for result in fold_results:
            fold = int(result["fold"])
            rows = result["oof_rows"]
            for model in ("B1", "M1", "M2"):
                fold_prediction = prediction_metrics(rows, model)
                fold_metric_rows.append({
                    "fold": fold, "model": model,
                    "mae": fold_prediction["mae"],
                    "top20_precision": fold_prediction["top20_precision"],
                    "simple_regret": batch_results["models"][model]["per_fold_mean_simple_regret"][str(fold)],
                    "cross_structure_metric_ratio": result["geometry"]["cross_structure_metric_ratio"],
                    "seed_0_spearman": result["geometry"]["seed_spearman"]["0"],
                    "seed_1_spearman": result["geometry"]["seed_spearman"]["1"],
                    "seed_2_spearman": result["geometry"]["seed_spearman"]["2"],
                })
        hp_report = {**hp_report, "fold_audits": [result["hp_audit"] for result in fold_results]}
        dump_json(output / "hp_control_report.json", hp_report)
        dump_csv(output / "metrics_per_fold.csv", fold_metric_rows)
        dump_json(output / "geometry_metrics.json", geometry)
        dump_csv(output / "oof_predictions.csv", oof_rows)
        dump_json(output / "best_of_4_batches.json", batch_manifest)
        dump_json(output / "best_of_4_results.json", batch_results)
        dump_json(output / "decision.json", {
            **decision, "prediction_metrics": prediction_summary,
            "gp_protocol": gp_protocol_signature(config),
        })
        write_final_report(output, decision, architecture_count)
        state = update_state(output, state, status="complete", decision=decision["classification"])
        write_manifest(output)
        return 0
    except BaseException as error:
        _write_failure_artifacts(output, state, error, architecture_count=architecture_count)
        write_manifest(output)
        raise


def dependency_inventory() -> dict[str, Any]:
    output = {}
    for name in ("yaml", "numpy", "scipy", "sklearn", "torch", "gpytorch"):
        try:
            __import__(name)
            output[name] = True
        except ImportError:
            output[name] = False
    return output


def dry_run(config_path: Path, output: Path, config_hash: str, source_hash: str) -> int:
    config = load_config(config_path)
    if output.exists():
        created = False
    else:
        created = False
    report = {
        "status": "DRY_RUN_OK", "config_sha256": config_hash,
        "source_sha256": source_hash, "formal_result_root": str(output),
        "formal_result_root_created": created,
        "formal_data_read": False, "formal_gnn_training_started": False,
        "decoder_used": False, "acquisition_used": False,
        "dependencies": dependency_inventory(),
        "planned_primary_result": config["data"]["primary_results"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _synthetic_gate_inputs(geometry_pass: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    coefficient = 0.50 if geometry_pass else 0.0
    geometry = {
        "folds": [
            {"fold": fold, "seed_spearman": {str(seed): coefficient for seed in REQUIRED_SEEDS}}
            for fold in range(5)
        ],
        "cross_structure_metric_ratio": 0.60 if geometry_pass else 1.0,
    }
    predictions = {
        "B1": {"mae": 1.0, "top20_precision": 0.50},
        "M1": {"mae": 0.90, "top20_precision": 0.60},
        "M2": {"mae": 0.85, "top20_precision": 0.60},
    }
    regret = {"models": {
        "B1": {"mean_simple_regret": 1.0, "per_fold_mean_simple_regret": {str(i): 1.0 for i in range(5)}},
        "M1": {"mean_simple_regret": 0.80, "per_fold_mean_simple_regret": {str(i): 0.8 for i in range(5)}},
        "M2": {"mean_simple_regret": 0.75, "per_fold_mean_simple_regret": {str(i): 0.75 for i in range(5)}},
    }}
    return geometry, predictions, regret


def synthetic_smoke(config_path: Path) -> int:
    config = load_config(config_path)
    geometry, predictions, regret = _synthetic_gate_inputs(True)
    passed = classify_quick_gate(80, geometry, predictions, regret, config["decision"])
    rejected = classify_quick_gate(80, *_synthetic_gate_inputs(False), config["decision"])
    leakage_closed = False
    rows = [
        {"architecture_fingerprint": "a", "target": 0.0, "standard_error": 0.1},
        {"architecture_fingerprint": "heldout", "target": 1.0, "standard_error": 0.1},
    ]
    try:
        construct_metric_pairs(
            rows, tau=1.0, weight_low=1.0, weight_high=100.0,
            allowed_fingerprints={"a"},
        )
    except ProtocolError:
        leakage_closed = True
    try:
        import numpy as np
        synthetic_rows = []
        optional_edges = ((0, 2), (0, 3), (1, 3), (1, 4), (2, 4))
        for index in range(20):
            architecture = {
                "operations": [
                    OPERATION_ORDER[index % 6],
                    OPERATION_ORDER[(index // 6) % 6],
                    OPERATION_ORDER[(index // 12) % 6],
                ],
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4]] + [
                    list(edge) for bit, edge in enumerate(optional_edges) if (index >> bit) & 1
                ],
            }
            synthetic_rows.append({
                **architecture,
                "architecture_fingerprint": architecture_fingerprint(architecture),
                "target": float(np.sin(index / 5.0)), "standard_error": 0.01,
                "raw_validation_mean": 0.70, "pool_identity": "synthetic",
            })
        train_embedding, query_embedding, metric_audit = train_metric_encoder(
            synthetic_rows[:16], synthetic_rows[16:], 0, config["metric_encoder"],
        )
        x = np.stack((np.linspace(0.0, 1.0, 12), np.linspace(0.0, 1.0, 12) ** 2), axis=1)
        y = np.sin(x[:, 0] * 3.0)
        gp_prediction, gp_audit = exact_matern_gp_predict(
            x, y, np.full(12, 0.02),
            np.asarray([[0.1, 0.01], [0.5, 0.25], [0.9, 0.81]]), config["gp"],
        )
        model_protocol_ok = bool(
            train_embedding.shape == (16, 8)
            and query_embedding.shape == (4, 8)
            and np.isfinite(train_embedding).all()
            and np.allclose(np.linalg.norm(train_embedding, axis=1), 1.0, atol=1e-5)
            and np.isfinite(gp_prediction).all()
            and metric_audit["outer_heldout_labels_used_for_pairs"] is False
            and gp_audit["posterior_variance_used"] is False
        )
    except (ImportError, GPNoiseProtocolUnavailable, ProtocolError):
        model_protocol_ok = False
    result = {
        "synthetic_pass_case": passed["classification"] == "QUICK_GATE_GP_PASSED",
        "synthetic_reject_case": rejected["classification"] == "QUICK_GATE_REJECTED",
        "synthetic_leakage_case_failed_closed": leakage_closed,
        "synthetic_metric_encoder_and_fixed_noise_gp": model_protocol_ok,
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--mark-failed")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config_hash = args.config_sha256 or sha256_file(config_path)
    source_hash = args.source_sha256 or "standalone"
    if args.synthetic_smoke:
        return synthetic_smoke(config_path)
    config = load_config(config_path)
    output = (args.output or (REPO_ROOT / config["output"]["result_root"])).resolve()
    if args.refresh_manifest:
        write_manifest(output)
        return 0
    if args.mark_failed is not None:
        state_path = output / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else initial_state(config_hash, source_hash)
        update_state(output, state, status="failed", launcher_error=str(args.mark_failed))
        return 0
    if args.dry_run:
        return dry_run(config_path, output, config_hash, source_hash)
    return execute_formal(config_path, output, config_hash, source_hash, bool(args.resume))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
