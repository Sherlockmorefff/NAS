#!/usr/bin/env python3
"""Frozen Phase A boundary and training-ceiling audit (validation only).

This program consumes only five pinned historical result archives.  It never
imports the dataset, evaluator, model, or training modules and has no code path
that starts a GNN or reads a Cora mask/label tensor.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "boundary-training-ceiling-audit-global4-v1"
RESULT_RELATIVE = f"results/posthoc/{EXPERIMENT}"
CONFIG_RELATIVE = "configs/boundary_training_ceiling_audit_global4_v1.json"
RUNNER_RELATIVE = "scripts/run_boundary_training_ceiling_audit.sh"
TMUX_SESSION = "boundary_training_audit_v1"
FROZEN_STAGE_KEY_ABSENT_POLICY = "matching_strategy_rows_require_task_stage_key_absent"
PRIMARY_NAMES = ("OH_BLIND_JOINT_RANDOM", "HR_BALANCED_RANDOM_HP")
SECONDARY_NAMES = (
    "B4_GUIDED_SEARCH_COMPLETE",
    "M1_M2_SEARCH_COMPLETE",
    "B4_ARCH_DIVERSITY_SEARCH_COMPLETE",
)
EXPECTED_ARCHIVE_HASHES = {
    "OH_BLIND_JOINT_RANDOM": "95a58f74985e202c2f6524b8d3daef91d25b89887bae9b50d66eedbd8da47819",
    "HR_BALANCED_RANDOM_HP": "050a64b0cde5bdb17ababa9833e5ee578b448c9b19da31fbb00f0251ec82491c",
    "B4_GUIDED_SEARCH_COMPLETE": "afb4bcf6dcc490ef884203dfa22414bbc1e203178b54f623fdb2ebdf1bf56ee9",
    "M1_M2_SEARCH_COMPLETE": "49e7b91956c433597f6f5e076ccaf2cf8d110b703388c9c81ad4dd5a8a141597",
    "B4_ARCH_DIVERSITY_SEARCH_COMPLETE": "aaa471d6e97b88c70d7f11b0ded1b3833895f1e24ed5e9489615e2d1296af1be",
}
EXPECTED_SOURCE_COUNTS = {
    "OH_BLIND_JOINT_RANDOM": 192,
    "HR_BALANCED_RANDOM_HP": 192,
    "B4_GUIDED_SEARCH_COMPLETE": 254,
    "M1_M2_SEARCH_COMPLETE": 364,
    "B4_ARCH_DIVERSITY_SEARCH_COMPLETE": 503,
}
EXPECTED_CONFIG_CANONICAL_SHA256 = "b2b12d41b0cfc761e20c6faa6d01f5704a60a055c33e5f60084aecde104845ec"
FROZEN_HIDDEN = (16, 32, 64, 128, 256, 512)
FROZEN_L2 = (1e-5, 5e-5, 1e-4, 5e-4, 1e-3)
FROZEN_LR = (1e-4, 10 ** -1.5)
FROZEN_DROPOUT = (0.1, 0.6)
FROZEN_SEEDS = (0, 1, 2)
REQUIRED_TRAINING_FIELDS = (
    "best_epoch",
    "stopped_epoch",
    "epochs_ran",
    "eval_epochs",
    "patience",
)


class ProtocolError(RuntimeError):
    """A frozen input or protocol requirement failed closed."""


class EngineeringError(RuntimeError):
    """An output or execution mechanism failed after scientific admission."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(json_bytes(value))
    os.replace(temporary, path)


def dump_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row}) if rows else []
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in rows:
                encoded = {}
                for key in fields:
                    value = row.get(key)
                    if isinstance(value, (dict, list, tuple)):
                        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
                    encoded[key] = value
                writer.writerow(encoded)
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def require_engineering(condition: bool, message: str) -> None:
    if not condition:
        raise EngineeringError(message)


def repo_path(relative: str) -> Path:
    require(isinstance(relative, str) and relative, "path must be a non-empty string")
    candidate = (REPO_ROOT / relative).resolve()
    require(candidate == REPO_ROOT or REPO_ROOT in candidate.parents, f"path escapes repository: {relative}")
    return candidate


def command_output(arguments: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(arguments),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"command": list(arguments), "returncode": completed.returncode, "output": completed.stdout}


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    require(stable_hash(config) == EXPECTED_CONFIG_CANONICAL_SHA256, "frozen config canonical SHA mismatch")
    require(config.get("protocol_version") == "phase-a-e-master-global4-v1", "protocol version mismatch")
    require(config.get("experiment") == EXPERIMENT, "experiment name mismatch")
    require(config.get("output") == RESULT_RELATIVE, "result path mismatch")
    require(config.get("tmux_session") == TMUX_SESSION, "tmux name mismatch")
    require(config.get("scientific_budget") == {"gpu": 0, "new_gnn": 0, "test_access": 0}, "Phase A budget mismatch")
    require(int(config.get("bootstrap_replicates", -1)) == 50000, "bootstrap count must be 50,000")
    require(
        config.get("bootstrap_specification")
        == {
            "confidence_interval": "percentile_2.5_97.5",
            "gate_role": "robustness_only_not_gate",
            "metric_order": ["hidden", "l2", "lr_u", "dropout_u", "depth", "hidden_l2_interaction"],
            "quantile_method": "linear",
            "resampling": "resample whole configurations with replacement within each source stratum, then recompute source Q90 gaps and equal-weight the two source gaps",
            "rng_seed": "unsigned big-endian integer from the first 8 bytes of frozen_protocol.json SHA256",
            "unit": "configuration",
        },
        "bootstrap specification mismatch",
    )
    require(tuple(config.get("top_tail_proportions", [])) == (0.25, 0.1, 0.05), "top-tail proportions mismatch")
    decoder = config.get("decoder", {})
    require(tuple(decoder.get("hidden_levels", [])) == FROZEN_HIDDEN, "hidden decoder levels mismatch")
    require(tuple(float(v) for v in decoder.get("l2_levels", [])) == FROZEN_L2, "L2 decoder levels mismatch")
    require(np.allclose(decoder.get("lr_bounds", []), FROZEN_LR, rtol=0.0, atol=1e-15), "LR bounds mismatch")
    require(tuple(float(v) for v in decoder.get("dropout_bounds", [])) == FROZEN_DROPOUT, "dropout bounds mismatch")
    require(int(decoder.get("max_effective_depth", -1)) == 5, "maximum effective depth mismatch")
    compatibility = config.get("compatibility", {})
    require(tuple(compatibility.get("seeds", [])) == FROZEN_SEEDS, "training seeds mismatch")
    require(compatibility.get("dataset") == "Cora", "dataset mismatch")
    require(compatibility.get("split") == "public", "split mismatch")
    require(compatibility.get("graph_transform") == "NormalizeFeatures", "graph transform mismatch")
    require(compatibility.get("metric") == "accuracy", "metric mismatch")
    require(compatibility.get("hp_mode") == "global4", "HP mode mismatch")
    require(int(compatibility.get("eval_epochs", -1)) == 300, "max epochs mismatch")
    require(int(compatibility.get("patience", -1)) == 80, "patience mismatch")
    require(compatibility.get("normalization") == "none", "LayerNorm must be off")
    require(compatibility.get("component_variant") == "baseline", "component variant must be baseline")
    require(compatibility.get("test_accessed") is False, "test_accessed must be false")
    require(compatibility.get("track_test") is False, "test tracking must be false")
    require(config.get("runner") == RUNNER_RELATIVE, "runner path mismatch")
    require(config.get("runner_sha256") == sha256_file(REPO_ROOT / RUNNER_RELATIVE), "runner SHA mismatch")
    gates = config.get("gates", {})
    frozen_gates = {
        "boundary_upper_main": 0.9,
        "boundary_upper_secondary": 0.8,
        "enrichment_min": 1.5,
        "interaction_gap_min": 0.003,
        "matched_gain_min": 0.003,
        "matched_min_pairs": 20,
        "matched_win_rate_min": 0.6,
        "q90_gap_min": 0.003,
        "training_late_best_fraction_min": 0.4,
        "training_late_best_ratio_min": 1.5,
        "training_ran_to_max_fraction_min": 0.5,
    }
    require(gates == frozen_gates, "frozen gate definitions mismatch")
    sources = list(config.get("input_sources", []))
    require([row.get("name") for row in sources] == [*PRIMARY_NAMES, *SECONDARY_NAMES], "source whitelist/order mismatch")
    for source in sources:
        name = str(source.get("name"))
        require(source.get("archive_sha256") == EXPECTED_ARCHIVE_HASHES[name], f"archive SHA protocol mismatch: {name}")
        require(int(source.get("expected_configurations", -1)) == EXPECTED_SOURCE_COUNTS[name], f"source count mismatch: {name}")
        if name == "HR_BALANCED_RANDOM_HP":
            require(source.get("task_stage") == "initial", "HR selected task stage mismatch")
            require(
                source.get("frozen_selection_task_stage_policy") == FROZEN_STAGE_KEY_ABSENT_POLICY,
                "HR frozen selection stage schema mismatch",
            )
        else:
            require(
                "frozen_selection_task_stage_policy" not in source,
                f"frozen stage-key-absent adapter is forbidden for {name}",
            )
    fresh = config.get("fresh_ceiling_identity", {})
    require(int(fresh.get("expected_configurations", -1)) == 1121, "fresh ceiling universe count mismatch")


def static_zero_compute_firewall(config: Mapping[str, Any]) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    forbidden_modules = {"torch", "torch_geometric", "dataset_utils", "eval_utils", "bo_phase4"}
    forbidden_calls = {
        "train_and_eval_arch",
        "train_arch",
        "load_dataset",
        "load_cora",
        "cuda",
    }
    forbidden_identifiers = {"test_" + "mask", "test_" + "labels", "test_" + "y"}
    import_hits: list[str] = []
    call_hits: list[str] = []
    identifier_hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden_modules:
                    import_hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in forbidden_modules:
                import_hits.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            if name in forbidden_calls:
                call_hits.append(name)
        elif isinstance(node, ast.Name) and node.id in forbidden_identifiers:
            identifier_hits.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in forbidden_identifiers:
            identifier_hits.append(node.attr)
    configured_paths = canonical_json_bytes(config).decode("utf-8").lower()
    path_hits = [token for token in ("/cora/", "data.pt") if token in configured_paths]
    runner_path = repo_path(str(config["runner"]))
    require(runner_path.is_file() and not runner_path.is_symlink(), "frozen runner is missing or a symlink")
    runner_text = runner_path.read_text(encoding="utf-8")
    runner_sha = sha256_file(runner_path)
    require(runner_sha == config["runner_sha256"], "frozen runner SHA mismatch")
    expected_runner = """#!/usr/bin/env bash
set -euo pipefail

cd /home/fangxy/D-VAE
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

python analyse/boundary_training_ceiling_audit.py \\
  --config configs/boundary_training_ceiling_audit_global4_v1.json \\
  --run
"""
    require(runner_text == expected_runner, "runner command is not the frozen zero-work Phase A entrypoint")
    require(not import_hits, f"zero-GNN firewall import hits: {sorted(set(import_hits))}")
    require(not call_hits, f"zero-GNN firewall call hits: {sorted(set(call_hits))}")
    require(not identifier_hits, f"test firewall identifier hits: {sorted(set(identifier_hits))}")
    require(not path_hits, f"test/data path firewall hits: {path_hits}")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"), "CUDA_VISIBLE_DEVICES must be empty or -1")
    return {
        "checked_at": utc_now(),
        "gpu_compute_calls": 0,
        "gnn_training_calls": 0,
        "test_data_reads": 0,
        "forbidden_call_hits": call_hits,
        "forbidden_identifier_hits": identifier_hits,
        "forbidden_import_hits": import_hits,
        "forbidden_path_hits": path_hits,
        "runner": str(runner_path.relative_to(REPO_ROOT)),
        "runner_sha256": runner_sha,
        "runner_exact_command_verified": True,
        "passed": True,
        "scientific_budget": dict(config["scientific_budget"]),
    }


def _manifest_root(files: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash([(str(row["relative_path"]), str(row["sha256"])) for row in files])


def validate_source_archive(source: Mapping[str, Any]) -> dict[str, Any]:
    name = str(source["name"])
    root = repo_path(str(source["root"]))
    archive = repo_path(str(source["archive"]))
    sidecar = repo_path(str(source["archive_sidecar"]))
    manifest_path = root / str(source["manifest"])
    for path in (root, archive, sidecar, manifest_path):
        require(path.exists(), f"missing pinned input for {name}: {path}")
    expected_archive = str(source["archive_sha256"])
    actual_archive = sha256_file(archive)
    require(actual_archive == expected_archive, f"archive SHA mismatch for {name}")
    sidecar_parts = sidecar.read_text(encoding="utf-8").strip().split()
    require(
        len(sidecar_parts) == 2
        and sidecar_parts[0] == expected_archive
        and sidecar_parts[1] == archive.name,
        f"archive sidecar mismatch for {name}",
    )
    require(sha256_file(manifest_path) == source["manifest_sha256"], f"live manifest SHA mismatch for {name}")
    with tarfile.open(archive, "r:gz") as handle:
        expected_member = str(source["archive_manifest_member"])
        candidates = [member for member in handle.getmembers() if member.name == expected_member]
        require(len(candidates) == 1 and candidates[0].isfile(), f"archive must contain exactly one regular frozen manifest member for {name}")
        extracted = handle.extractfile(candidates[0])
        require(extracted is not None, f"cannot extract manifest for {name}")
        archived_manifest_bytes = extracted.read()
    require(archived_manifest_bytes == manifest_path.read_bytes(), f"archive/live manifest mismatch for {name}")
    manifest = load_json(manifest_path)
    files = manifest.get("files")
    require(isinstance(files, list), f"manifest files missing for {name}")
    if "file_count" in manifest:
        require(int(manifest["file_count"]) == len(files), f"manifest file_count mismatch for {name}")
    verified_bytes = 0
    for entry in files:
        relative = str(entry.get("relative_path", ""))
        require(relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts, f"unsafe manifest path for {name}")
        path = (root / relative).resolve()
        require(root in path.parents, f"manifest path escapes root for {name}: {relative}")
        require(path.is_file(), f"manifest file missing for {name}: {relative}")
        size = path.stat().st_size
        require(size == int(entry["size_bytes"]), f"manifest size mismatch for {name}: {relative}")
        require(sha256_file(path) == entry["sha256"], f"manifest SHA mismatch for {name}: {relative}")
        verified_bytes += size
    if "manifest_root_sha256" in manifest:
        require(_manifest_root(files) == manifest["manifest_root_sha256"], f"manifest root mismatch for {name}")
    return {
        "name": name,
        "archive": str(archive.relative_to(REPO_ROOT)),
        "archive_sha256": actual_archive,
        "archive_size_bytes": archive.stat().st_size,
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_file_count": len(files),
        "manifest_root_sha256": manifest.get("manifest_root_sha256"),
        "manifest_verified_bytes": verified_bytes,
        "archive_manifest_matches_live": True,
    }


def _dependency_rows(path: Path) -> list[Mapping[str, Any]]:
    value = load_json(path)
    rows = value.get("files") if isinstance(value, dict) else None
    require(isinstance(rows, list), f"dependency manifest has no files list: {path}")
    return rows


def validate_dependency_identity(source: Mapping[str, Any], required: Mapping[str, str]) -> dict[str, str]:
    root = repo_path(str(source["root"]))
    relative = source.get("dependency_manifest", "source_dependency_manifest.json")
    path = root / str(relative)
    require(path.is_file(), f"missing dependency manifest for {source['name']}")
    by_basename: dict[str, set[str]] = defaultdict(set)
    for row in _dependency_rows(path):
        raw_path = str(row.get("relative_path") or row.get("path") or "")
        if raw_path:
            by_basename[Path(raw_path).name].add(str(row.get("sha256")))
    result = {}
    for basename, expected in required.items():
        hashes = by_basename.get(basename, set())
        require(hashes == {expected}, f"dependency mismatch for {source['name']}: {basename}={sorted(hashes)}")
        result[basename] = expected
    return result


def verify_decoder_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    decoder = config["decoder"]
    hp_modes = repo_path(str(decoder["hp_modes_snapshot"]))
    nas_space = repo_path(str(decoder["nas_space_snapshot"]))
    require(sha256_file(hp_modes) == decoder["hp_modes_sha256"], "frozen hp_modes.py SHA mismatch")
    require(sha256_file(nas_space) == decoder["nas_space_sha256"], "frozen nas_space.py SHA mismatch")
    hp_text = re.sub(r"\s+", "", hp_modes.read_text(encoding="utf-8"))
    nas_text = re.sub(r"\s+", "", nas_space.read_text(encoding="utf-8"))
    require("HIDDEN_DIM_OPTIONS=[16,32,64,128,256,512]" in hp_text, "hidden decoder source mismatch")
    require("L2_OPTIONS=[1e-5,5e-5,1e-4,5e-4,1e-3]" in hp_text, "L2 decoder source mismatch")
    require("MAX_OP_NODES=5" in hp_text, "maximum decoder depth mismatch")
    for token in ("LOG_LR_MIN=-4.0", "LOG_LR_MAX=-1.5", "DROPOUT_MIN=0.1", "DROPOUT_MAX=0.6"):
        require(token in nas_text, f"continuous decoder source mismatch: {token}")
    return {
        "hidden_levels": list(FROZEN_HIDDEN),
        "l2_levels": list(FROZEN_L2),
        "lr_bounds": list(FROZEN_LR),
        "dropout_bounds": list(FROZEN_DROPOUT),
        "max_effective_depth": 5,
        "hp_modes_sha256": sha256_file(hp_modes),
        "nas_space_sha256": sha256_file(nas_space),
        "matched_frozen_decoder": True,
    }


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ProtocolError(f"cannot parse boolean: {value!r}")


def parse_json_field(value: Any, field: str) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot parse JSON field {field}: {value!r}") from exc


def architecture_fingerprint(operations: Sequence[str], edges: Sequence[Sequence[int]]) -> str:
    payload = {
        "operations": [str(value) for value in operations],
        "edges": sorted([list(map(int, edge)) for edge in edges]),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hp_fingerprint(raw_hp: Sequence[float]) -> str:
    return stable_hash([float(value) for value in raw_hp])


def joint_fingerprint(architecture_fp: str, hp_fp: str) -> str:
    return stable_hash({"architecture_fingerprint": architecture_fp, "hp_fingerprint": hp_fp})


def normalize_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    joint = raw.get("joint_config_fingerprint", raw.get("joint_fingerprint"))
    seed = raw.get("seed", raw.get("training_seed"))
    validation = raw.get("val_acc", raw.get("validation_accuracy"))
    operations = parse_json_field(raw.get("operations"), "operations")
    edges = parse_json_field(raw.get("edges"), "edges")
    raw_hp = parse_json_field(raw.get("raw_hp_vector"), "raw_hp_vector")
    record = {
        "joint_fingerprint": str(joint),
        "architecture_fingerprint": str(raw.get("architecture_fingerprint")),
        "hp_fingerprint": str(raw.get("hp_fingerprint")),
        "operations": [str(value) for value in operations],
        "edges": sorted([list(map(int, edge)) for edge in edges]),
        "raw_hp_vector": [float(value) for value in raw_hp],
        "effective_layers": int(raw.get("effective_layers")),
        "lr": float(raw.get("lr")),
        "dropout": float(raw.get("dropout")),
        "hidden": int(raw.get("hidden_dim")),
        "l2": float(raw.get("l2")),
        "seed": None if seed is None else int(seed),
        "val_acc": None if validation is None else float(validation),
        "terminal_status": str(raw.get("terminal_status", "")),
        "valid": parse_bool(raw.get("valid")),
        "dataset": str(raw.get("dataset", "")),
        "dataset_content_fingerprint": str(raw.get("dataset_content_fingerprint", "")),
        "split_fingerprint": str(raw.get("split_fingerprint", "")),
        "graph_transform": str(raw.get("graph_transform", "")),
        "metric": str(raw.get("metric", "")),
        "hp_mode": str(raw.get("hp_mode", "")),
        "normalization": str(raw.get("normalization", "")),
        "training_entry": str(raw.get("training_entry", "")),
        "training_mode": str(raw.get("training_mode", "")),
        "best_epoch": raw.get("best_epoch"),
        "stopped_epoch": raw.get("stopped_epoch"),
        "epochs_ran": raw.get("epochs_ran"),
        "eval_epochs": raw.get("eval_epochs"),
        "patience": raw.get("patience"),
        "task_stage": raw.get("task_stage"),
    }
    for field in REQUIRED_TRAINING_FIELDS:
        if record[field] in (None, ""):
            record[field] = None
        else:
            record[field] = int(record[field])
    return record


def decoded_from_raw(raw_hp: Sequence[float]) -> dict[str, Any]:
    require(len(raw_hp) == 4, "global4 raw HP vector must have four coordinates")
    raw = np.asarray(raw_hp, dtype=np.float32).astype(np.float64)
    require(bool(np.all(np.isfinite(raw))) and bool(np.all((raw >= 0.0) & (raw <= 1.0))), "raw HP coordinate outside [0,1]")
    lr = float(10 ** (raw[0] * (-1.5 + 4.0) - 4.0))
    dropout = float(raw[1] * 0.5 + 0.1)
    hidden_index = min(int(min(raw[2], 1.0 - 1e-12) * len(FROZEN_HIDDEN)), len(FROZEN_HIDDEN) - 1)
    l2_index = min(int(min(raw[3], 1.0 - 1e-12) * len(FROZEN_L2)), len(FROZEN_L2) - 1)
    return {
        "lr": lr,
        "dropout": dropout,
        "hidden": FROZEN_HIDDEN[hidden_index],
        "l2": FROZEN_L2[l2_index],
    }


def validate_record_identity(record: Mapping[str, Any]) -> None:
    require(len(record["operations"]) <= 5, "architecture exceeds frozen depth")
    effective = sum(operation != "Identity" for operation in record["operations"])
    require(effective == int(record["effective_layers"]), f"effective depth mismatch: {record['joint_fingerprint']}")
    arch = architecture_fingerprint(record["operations"], record["edges"])
    hp = hp_fingerprint(record["raw_hp_vector"])
    joint = joint_fingerprint(arch, hp)
    require(arch == record["architecture_fingerprint"], f"architecture fingerprint mismatch: {record['joint_fingerprint']}")
    require(hp == record["hp_fingerprint"], f"HP fingerprint mismatch: {record['joint_fingerprint']}")
    require(joint == record["joint_fingerprint"], f"joint fingerprint mismatch: {record['joint_fingerprint']}")
    decoded = decoded_from_raw(record["raw_hp_vector"])
    require(decoded["hidden"] == record["hidden"], f"decoded hidden mismatch: {joint}")
    require(math.isclose(decoded["l2"], record["l2"], rel_tol=0.0, abs_tol=1e-15), f"decoded L2 mismatch: {joint}")
    require(math.isclose(decoded["lr"], record["lr"], rel_tol=2e-6, abs_tol=1e-12), f"decoded LR mismatch: {joint}")
    require(math.isclose(decoded["dropout"], record["dropout"], rel_tol=2e-6, abs_tol=1e-9), f"decoded dropout mismatch: {joint}")


def validate_record_compatibility(record: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    require(str(record["dataset"]).lower() == str(expected["dataset"]).lower(), "dataset record mismatch")
    for field in ("dataset_content_fingerprint", "split_fingerprint", "graph_transform", "metric", "hp_mode", "normalization", "training_entry", "training_mode"):
        require(record[field] == expected[field], f"record compatibility mismatch for {field}: {record['joint_fingerprint']}")
    if record["eval_epochs"] is not None:
        require(record["eval_epochs"] == int(expected["eval_epochs"]), "eval_epochs mismatch")
    if record["patience"] is not None:
        require(record["patience"] == int(expected["patience"]), "patience mismatch")
    require(record["terminal_status"] == "success" and record["valid"] is True, "source contains unsuccessful/invalid record")
    require(record["seed"] in FROZEN_SEEDS, "unexpected training seed")
    require(record["val_acc"] is not None and math.isfinite(float(record["val_acc"])), "missing validation accuracy")


def _row_has_strategy(row: Mapping[str, Any], strategy: str) -> bool:
    if row.get("strategy") == strategy:
        return True
    selected = row.get("selected_by", [])
    return isinstance(selected, list) and any(isinstance(value, dict) and value.get("strategy") == strategy for value in selected)


def _selection_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(raw)
    copied.setdefault("terminal_status", "success")
    copied.setdefault("valid", True)
    copied.setdefault("dataset", "cora")
    copied.setdefault("dataset_content_fingerprint", "")
    copied.setdefault("split_fingerprint", "")
    copied.setdefault("graph_transform", "")
    copied.setdefault("metric", "")
    copied.setdefault("hp_mode", "global4")
    copied.setdefault("normalization", "none")
    copied.setdefault("training_entry", "")
    copied.setdefault("training_mode", "")
    copied.setdefault("seed", 0)
    copied.setdefault("val_acc", 0.0)
    copied.setdefault("best_epoch", 0)
    copied.setdefault("stopped_epoch", 0)
    copied.setdefault("epochs_ran", 0)
    copied.setdefault("eval_epochs", 300)
    copied.setdefault("patience", 80)
    return normalize_record(copied)


def _identity_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "joint_fingerprint": record["joint_fingerprint"],
        "architecture_fingerprint": record["architecture_fingerprint"],
        "hp_fingerprint": record["hp_fingerprint"],
        "operations": record["operations"],
        "edges": record["edges"],
        "raw_hp_vector": record["raw_hp_vector"],
        "effective_layers": record["effective_layers"],
        "lr": record["lr"],
        "dropout": record["dropout"],
        "hidden": record["hidden"],
        "l2": record["l2"],
    }


def _metadata_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["joint_fingerprint"] != right["joint_fingerprint"]:
        return False
    exact_fields = ("architecture_fingerprint", "hp_fingerprint", "operations", "edges", "effective_layers", "hidden")
    if any(left[field] != right[field] for field in exact_fields):
        return False
    if not np.allclose(left["raw_hp_vector"], right["raw_hp_vector"], rtol=0.0, atol=1e-12):
        return False
    return all(math.isclose(float(left[field]), float(right[field]), rel_tol=2e-6, abs_tol=1e-12) for field in ("lr", "dropout", "l2"))


def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _manifest_fingerprint_set(
    rows: Sequence[Mapping[str, Any]], label: str, expected_count: int
) -> set[str]:
    fingerprints = []
    for row in rows:
        fingerprint = row.get("joint_config_fingerprint")
        require(
            isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None,
            f"missing or invalid joint fingerprint in {label}",
        )
        fingerprints.append(fingerprint)
    require(len(rows) == expected_count, f"{label} count mismatch: expected {expected_count}, got {len(rows)}")
    result = set(fingerprints)
    require(len(result) == expected_count, f"duplicate joint fingerprint in {label}")
    return result


def select_primary_manifest_rows(
    source: Mapping[str, Any],
    selection_raw: Sequence[Mapping[str, Any]],
    frozen_raw: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any]]:
    """Apply the frozen, source-specific pre-label manifest schemas."""
    require(
        isinstance(selection_raw, list)
        and isinstance(frozen_raw, list)
        and all(isinstance(row, dict) for row in [*selection_raw, *frozen_raw]),
        "primary selection files must be arrays of objects",
    )
    name = str(source["name"])
    strategy = str(source["strategy"])
    expected_count = int(source["expected_configurations"])
    task_stage = source.get("task_stage")
    frozen_stage_policy = source.get("frozen_selection_task_stage_policy")

    selected_strategy_rows = [row for row in selection_raw if _row_has_strategy(row, strategy)]
    if task_stage is None:
        require(frozen_stage_policy is None, f"unexpected frozen stage policy for {name}")
        selected_rows = selected_strategy_rows
    else:
        selected_rows = [
            row
            for row in selected_strategy_rows
            if "task_stage" in row and row["task_stage"] == task_stage
        ]

    frozen_strategy_rows = [row for row in frozen_raw if _row_has_strategy(row, strategy)]
    if frozen_stage_policy is None:
        require(task_stage is None, f"explicit frozen stage schema policy is required for {name}")
        frozen_rows = frozen_strategy_rows
        frozen_stage_schema = "NOT_APPLICABLE"
    else:
        require(
            name == "HR_BALANCED_RANDOM_HP"
            and task_stage == "initial"
            and frozen_stage_policy == FROZEN_STAGE_KEY_ABSENT_POLICY,
            f"frozen stage-key-absent adapter is not allowlisted for {name}",
        )
        require(
            all("task_stage" not in row for row in frozen_strategy_rows),
            f"matching frozen strategy rows must omit task_stage for {name}",
        )
        frozen_rows = frozen_strategy_rows
        frozen_stage_schema = frozen_stage_policy

    selected_set = _manifest_fingerprint_set(selected_rows, f"selected primary strategy {name}", expected_count)
    frozen_set = _manifest_fingerprint_set(frozen_rows, f"frozen primary strategy {name}", expected_count)
    missing = sorted(selected_set - frozen_set)
    extra = sorted(frozen_set - selected_set)
    require(
        not missing and not extra,
        f"pre-label selection manifests disagree for {name}: missing={len(missing)}, extra={len(extra)}",
    )
    return selected_rows, frozen_rows, {
        "selected_manifest_total_rows": len(selection_raw),
        "frozen_manifest_total_rows": len(frozen_raw),
        "selected_strategy_rows": len(selected_strategy_rows),
        "selected_rows_after_stage_filter": len(selected_rows),
        "frozen_strategy_rows": len(frozen_rows),
        "selected_task_stage": task_stage,
        "frozen_task_stage_schema": frozen_stage_schema,
        "selected_frozen_missing_count": 0,
        "selected_frozen_extra_count": 0,
    }


def validate_primary_metadata(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = repo_path(str(source["root"]))
    strategy = str(source["strategy"])
    selection_raw = load_json(root / str(source["selection"]))
    frozen_raw = load_json(root / str(source["frozen_selection"]))
    selected_rows, frozen_rows, manifest_schema_audit = select_primary_manifest_rows(
        source, selection_raw, frozen_raw
    )
    forbidden_selection_fields = {
        key
        for row in [*selected_rows, *frozen_rows]
        for key in row
        if key.lower().startswith(("val", "test")) or "accuracy" in key.lower()
    }
    require(
        not forbidden_selection_fields,
        f"primary pre-label selection contains outcome fields for {source['name']}: {sorted(forbidden_selection_fields)}",
    )
    expected_count = int(source["expected_configurations"])
    records = []
    for raw in selected_rows:
        record = _selection_record(raw)
        validate_record_identity(record)
        record["source"] = str(source["name"])
        record["hp_regime"] = raw.get("hp_regime")
        records.append(record)
    hidden = Counter(row["hidden"] for row in records)
    l2 = Counter(row["l2"] for row in records)
    regimes = Counter(str(row.get("hp_regime")) for row in records)
    hidden_upper = hidden[512]
    l2_upper = l2[1e-3]
    require(len(hidden) >= 4 and set(hidden).issubset(FROZEN_HIDDEN), f"hidden coverage insufficient for {source['name']}")
    require(len(l2) >= 4 and set(l2).issubset(FROZEN_L2), f"L2 coverage insufficient for {source['name']}")
    require(hidden_upper >= 20 and expected_count - hidden_upper >= 20, f"hidden upper/non-upper coverage insufficient for {source['name']}")
    require(l2_upper >= 20 and expected_count - l2_upper >= 20, f"L2 upper/non-upper coverage insufficient for {source['name']}")
    require(
        set(regimes) == {"hp_low", "hp_mid_low", "hp_mid_high", "hp_high"}
        and all(regimes[value] > 0 for value in regimes),
        f"raw HP strata do not establish non-truncated sampling for {source['name']}",
    )
    require(any(not all(0.8 <= value <= 0.95 for value in row["raw_hp_vector"]) for row in records), f"primary source is high-regime truncated: {source['name']}")
    audit = {
        "source": source["name"],
        "strategy": strategy,
        "configuration_count": len(records),
        "selection_manifest_agreement": True,
        "selection_used_validation": False,
        "selection_outcome_fields": [],
        "manifest_schema": manifest_schema_audit,
        "hidden_counts": {str(key): hidden[key] for key in sorted(hidden)},
        "l2_counts": {format(key, ".12g"): l2[key] for key in sorted(l2)},
        "hp_regime_counts": dict(sorted(regimes.items())),
        "hidden_upper_count": hidden_upper,
        "hidden_non_upper_count": expected_count - hidden_upper,
        "l2_upper_count": l2_upper,
        "l2_non_upper_count": expected_count - l2_upper,
        "raw_hp_high_regime_truncated": False,
        "passed": True,
    }
    return records, audit


def validate_primary_protocol_metadata(source: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    root = repo_path(str(source["root"]))
    prereg = load_json(root / str(source["preregistration"]))
    training = prereg.get("training", {})
    require(training.get("dataset") == expected["dataset"], f"preregistered dataset mismatch for {source['name']}")
    require(training.get("dataset_content_fingerprint") == expected["dataset_content_fingerprint"], "dataset fingerprint mismatch")
    require(training.get("split") == expected["split"], "split mismatch")
    require(training.get("split_fingerprint") == expected["split_fingerprint"], "split fingerprint mismatch")
    require(training.get("hp_mode") == expected["hp_mode"], "HP mode mismatch")
    require(int(training.get("max_epochs", -1)) == expected["eval_epochs"], "max epochs mismatch")
    require(int(training.get("patience", -1)) == expected["patience"], "patience mismatch")
    seeds = training.get("seeds", prereg.get("initial_seeds"))
    require(tuple(seeds or []) == FROZEN_SEEDS, "preregistered seeds mismatch")
    require(training.get("track_test") is False, "preregistered test tracking is not false")
    firewall = load_json(root / str(source["pre_label_firewall"]))
    require(firewall.get("passed") is True and firewall.get("track_test") is False, "source test firewall failed")
    for key in ("selection_validation_or_test_fields", "task_test_fields", "selection_test_field_hits", "task_test_field_hits"):
        if key in firewall:
            require(not firewall[key], f"source test-field hits for {source['name']}: {key}")
    evaluator = root / str(source["layernorm_evaluator"])
    text = re.sub(r"\s+", "", evaluator.read_text(encoding="utf-8"))
    require('component_variant="baseline"' in text, f"LayerNorm baseline evidence missing for {source['name']}")
    require("track_test=False" in text, f"track_test=False evaluator evidence missing for {source['name']}")
    final = load_json(root / str(source["test_evidence"]))
    if source["name"] == "OH_BLIND_JOINT_RANDOM":
        require(final.get("test_evaluation_authorized") is False, "OH test authorization mismatch")
    else:
        require(final.get("test_evaluation_performed") is False, "HR test evaluation was performed")
        require(final.get("one_time_test_reveal_performed") is False, "HR test reveal was performed")
    return {
        "dataset": training.get("dataset"),
        "split": training.get("split"),
        "dataset_content_fingerprint": training.get("dataset_content_fingerprint"),
        "split_fingerprint": training.get("split_fingerprint"),
        "max_epochs": training.get("max_epochs"),
        "patience": training.get("patience"),
        "seeds": list(seeds),
        "hp_mode": training.get("hp_mode"),
        "graph_transform": expected["graph_transform"],
        "metric": expected["metric"],
        "component_variant": "baseline",
        "track_test": False,
        "test_accessed": False,
        "passed": True,
    }


def load_primary_results(
    source: Mapping[str, Any],
    selection_records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = repo_path(str(source["root"]))
    selected = {row["joint_fingerprint"]: row for row in selection_records}
    records: list[dict[str, Any]] = []
    for relative in source["result_records"]:
        for raw in _read_csv(root / str(relative)):
            joint = str(raw.get("joint_config_fingerprint"))
            if joint not in selected:
                continue
            record = normalize_record(raw)
            validate_record_identity(record)
            validate_record_compatibility(record, expected)
            if source.get("task_stage") is not None:
                require(record["task_stage"] == source["task_stage"], "primary task stage mismatch")
            require(_metadata_matches(record, selected[joint]), f"selection/result identity mismatch: {joint}")
            record["source"] = str(source["name"])
            records.append(record)
    configurations = collapse_configurations(records, expected_count=int(source["expected_configurations"]))
    return configurations, records


def collapse_configurations(records: Sequence[Mapping[str, Any]], expected_count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["joint_fingerprint"])].append(row)
    require(len(grouped) == expected_count, f"configuration count mismatch: expected {expected_count}, got {len(grouped)}")
    configurations = []
    for joint in sorted(grouped):
        rows = grouped[joint]
        require(sorted(row["seed"] for row in rows) == list(FROZEN_SEEDS), f"seed coverage mismatch: {joint}")
        require(len({(row["joint_fingerprint"], row["seed"]) for row in rows}) == 3, f"duplicate seed record: {joint}")
        first = rows[0]
        require(all(_metadata_matches(first, row) for row in rows[1:]), f"seed metadata conflict: {joint}")
        values = [float(next(row["val_acc"] for row in rows if row["seed"] == seed)) for seed in FROZEN_SEEDS]
        configurations.append(
            {
                **_identity_signature(first),
                "source": first["source"],
                "seed_values": values,
                "mean_val": float(sum(values) / 3.0),
            }
        )
    return configurations


def validate_secondary_protocol_metadata(source: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    root = repo_path(str(source["root"]))
    frozen = load_json(root / str(source["frozen_protocol"]))
    evaluation = frozen.get("evaluation", {})
    require(frozen.get("dataset") == expected["dataset"], f"secondary dataset mismatch for {source['name']}")
    require(frozen.get("hp_mode") == expected["hp_mode"], "secondary HP mode mismatch")
    require(int(evaluation.get("max_epochs", -1)) == expected["eval_epochs"], "secondary max epochs mismatch")
    require(int(evaluation.get("patience", -1)) == expected["patience"], "secondary patience mismatch")
    require(tuple(evaluation.get("training_seeds", [])) == FROZEN_SEEDS, "secondary seed mismatch")
    require(evaluation.get("track_test") is False, "secondary track_test mismatch")
    require(evaluation.get("validation_only_data_view") == "required_before_evaluator_entry", "secondary validation-only contract missing")
    decision = load_json(root / str(source["decision"]))
    test_values = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "test_accessed":
                    test_values.append(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(decision)
    require(test_values and all(value is False for value in test_values), f"secondary test_accessed evidence missing/true for {source['name']}")
    return {
        "dataset": frozen.get("dataset"),
        "split": expected["split"],
        "dataset_content_fingerprint": expected["dataset_content_fingerprint"],
        "split_fingerprint": expected["split_fingerprint"],
        "max_epochs": evaluation.get("max_epochs"),
        "patience": evaluation.get("patience"),
        "seeds": evaluation.get("training_seeds"),
        "hp_mode": frozen.get("hp_mode"),
        "graph_transform": expected["graph_transform"],
        "metric": expected["metric"],
        "component_variant": "baseline",
        "track_test": False,
        "test_accessed": False,
        "passed": True,
    }


def validate_high_regime_generator(source: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    root = repo_path(str(source["root"]))
    path = root / str(source["generator_manifest"])
    count = 0
    joints = set()
    for raw in _read_csv(path):
        count += 1
        require(raw.get("proposal_distribution") == "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR", f"generator arm mismatch for {source['name']}")
        raw_hp = [float(value) for value in parse_json_field(raw.get("raw_hp_vector"), "raw_hp_vector")]
        require(len(raw_hp) == 4 and all(0.8 <= value <= 0.95 for value in raw_hp), f"high-regime raw HP mismatch for {source['name']}")
        require(math.isclose(float(raw["l2"]), 1e-3, rel_tol=0.0, abs_tol=1e-15), "secondary L2 is not fixed at 1e-3")
        require(int(raw["hidden_dim"]) in (256, 512), "secondary hidden is outside high regime")
        operations = parse_json_field(raw["operations"], "operations")
        edges = parse_json_field(raw["edges"], "edges")
        arch = architecture_fingerprint(operations, edges)
        hp = hp_fingerprint(raw_hp)
        joint = joint_fingerprint(arch, hp)
        require(arch == raw["architecture_fingerprint"] and hp == raw["hp_fingerprint"] and joint == raw["joint_fingerprint"], "generator fingerprint mismatch")
        joints.add(joint)
    require(count == int(source["expected_generator_candidates"]), f"generator candidate count mismatch for {source['name']}")
    require(len(joints) == count, f"duplicate generator candidates for {source['name']}")
    return {
        "source": source["name"],
        "candidate_count": count,
        "proposal_distribution": "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR",
        "raw_hp_bounds": [0.8, 0.95],
        "l2_values": [1e-3],
        "hidden_values": [256, 512],
        "boundary_authorization_role": "FORBIDDEN_SECONDARY_ONLY",
        "passed": True,
    }, joints


def load_secondary_results(
    source: Mapping[str, Any], expected: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = repo_path(str(source["root"]))
    records: list[dict[str, Any]] = []
    for relative in source["result_records"]:
        path = root / str(relative)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProtocolError(f"invalid JSONL at {path}:{line_number}") from exc
                trajectory = raw.pop("trajectory", None)
                if trajectory is not None:
                    require(isinstance(trajectory, list), "trajectory must be a list")
                    require(len(trajectory) == int(raw.get("epochs_ran")), "trajectory/epochs_ran mismatch")
                record = normalize_record(raw)
                validate_record_identity(record)
                validate_record_compatibility(record, expected)
                require(str(raw.get("hp_regime")) == "hp_high", "secondary evaluated record is not hp_high")
                require(all(0.8 <= value <= 0.95 for value in record["raw_hp_vector"]), "secondary evaluated raw HP outside high regime")
                require(math.isclose(record["l2"], 1e-3, rel_tol=0.0, abs_tol=1e-15), "secondary evaluated L2 mismatch")
                require(record["hidden"] in (256, 512), "secondary evaluated hidden mismatch")
                record["source"] = str(source["name"])
                records.append(record)
    configurations = collapse_configurations(records, expected_count=int(source["expected_configurations"]))
    require(len(records) == 3 * int(source["expected_configurations"]), "secondary seed-run count mismatch")
    return configurations, records


def validate_evaluated_generator_membership(
    evaluated_joints: set[str], generator_joints: set[str], source_name: str
) -> None:
    require(
        evaluated_joints <= generator_joints,
        f"evaluated secondary configuration is outside the frozen generator arm: {source_name}",
    )


def validate_fresh_identity(
    config: Mapping[str, Any], secondary_configurations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    spec = config["fresh_ceiling_identity"]
    path = repo_path(str(spec["candidate_universe"]))
    require(sha256_file(path) == spec["candidate_universe_sha256"], "fresh-ceiling candidate universe SHA mismatch")
    fresh = {}
    for row in _read_csv(path):
        joint = str(row["joint_fingerprint"])
        require(joint not in fresh, "duplicate fresh-ceiling identity")
        fresh[joint] = (str(row["architecture_fingerprint"]), str(row["hp_fingerprint"]))
    secondary = {
        str(row["joint_fingerprint"]): (str(row["architecture_fingerprint"]), str(row["hp_fingerprint"]))
        for row in secondary_configurations
    }
    require(len(fresh) == int(spec["expected_configurations"]), "fresh-ceiling identity count mismatch")
    require(fresh == secondary, "fresh-ceiling identity/coverage mismatch")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "configuration_count": len(fresh),
        "identity_matches_secondary_union": True,
        "validation_columns_used": False,
        "scientific_role": "IDENTITY_COVERAGE_CROSS_CHECK_ONLY",
        "passed": True,
    }


def u_log(value: float, lower: float, upper: float) -> float:
    result = (math.log10(value) - math.log10(lower)) / (math.log10(upper) - math.log10(lower))
    require(-1e-12 <= result <= 1.0 + 1e-12, f"log boundary coordinate outside [0,1]: {value}")
    return min(1.0, max(0.0, result))


def u_linear(value: float, lower: float, upper: float) -> float:
    result = (value - lower) / (upper - lower)
    require(-1e-12 <= result <= 1.0 + 1e-12, f"linear boundary coordinate outside [0,1]: {value}")
    return min(1.0, max(0.0, result))


def enrich_configuration(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "u_lr": u_log(float(record["lr"]), *FROZEN_LR),
        "u_dropout": u_linear(float(record["dropout"]), *FROZEN_DROPOUT),
        "u_l2": u_log(float(record["l2"]), FROZEN_L2[0], FROZEN_L2[-1]),
    }


def deduplicate_primary(
    primary: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    seen: dict[str, Mapping[str, Any]] = {}
    retained: dict[str, list[dict[str, Any]]] = {name: [] for name in PRIMARY_NAMES}
    duplicates = []
    premerge = sum(len(primary[name]) for name in PRIMARY_NAMES)
    require(premerge == 384, f"primary pre-merge count must be 384, got {premerge}")
    for name in PRIMARY_NAMES:
        for raw in primary[name]:
            row = enrich_configuration(raw)
            joint = str(row["joint_fingerprint"])
            if joint in seen:
                prior = seen[joint]
                require(_metadata_matches(row, prior), f"cross-stratum duplicate metadata conflict: {joint}")
                require(np.allclose(row["seed_values"], prior["seed_values"], rtol=0.0, atol=1e-15), f"cross-stratum duplicate seed values conflict: {joint}")
                duplicates.append({"joint_fingerprint": joint, "dropped_from": name, "retained_from": prior["source"]})
                continue
            seen[joint] = row
            retained[name].append(row)
    return retained, {
        "premerge_configuration_count": premerge,
        "post_dedup_configuration_count": len(seen),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "conflict_count": 0,
        "passed": True,
    }


def top_tail_analysis(
    primary: Mapping[str, Sequence[Mapping[str, Any]]],
    proportions: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indicators: dict[str, Callable[[Mapping[str, Any]], bool]] = {
        "hidden_512": lambda row: int(row["hidden"]) == 512,
        "l2_u_ge_0.9": lambda row: float(row["u_l2"]) >= 0.9,
        "l2_u_ge_0.8": lambda row: float(row["u_l2"]) >= 0.8,
        "lr_u_ge_0.9": lambda row: float(row["u_lr"]) >= 0.9,
        "lr_u_ge_0.8": lambda row: float(row["u_lr"]) >= 0.8,
        "dropout_u_ge_0.9": lambda row: float(row["u_dropout"]) >= 0.9,
        "dropout_u_ge_0.8": lambda row: float(row["u_dropout"]) >= 0.8,
        "depth_5": lambda row: int(row["effective_layers"]) == 5,
    }
    rows: list[dict[str, Any]] = []
    values: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for name in PRIMARY_NAMES:
        ordered = sorted(primary[name], key=lambda row: (-float(row["mean_val"]), str(row["joint_fingerprint"])))
        n = len(ordered)
        require(n > 0, f"empty primary stratum: {name}")
        selections: dict[str, Sequence[Mapping[str, Any]]] = {"All": ordered}
        for proportion in proportions:
            selections[f"Top{int(round(proportion * 100))}"] = ordered[: math.ceil(n * proportion)]
        for indicator, function in indicators.items():
            for subset_name, subset in selections.items():
                proportion_value = float(sum(bool(function(row)) for row in subset) / len(subset))
                values[indicator][name][subset_name] = proportion_value
                rows.append(
                    {
                        "indicator": indicator,
                        "source": name,
                        "subset": subset_name,
                        "n": len(subset),
                        "upper_count": sum(bool(function(row)) for row in subset),
                        "proportion": proportion_value,
                        "weighting": "within_source",
                    }
                )
    combined: dict[str, dict[str, Any]] = {}
    for indicator in indicators:
        subset_names = ("All", "Top25", "Top10", "Top5")
        combined[indicator] = {}
        all_prop = float(np.mean([values[indicator][name]["All"] for name in PRIMARY_NAMES]))
        require(all_prop > 0.0, f"zero enrichment denominator: {indicator}")
        for subset_name in subset_names:
            proportion_value = float(np.mean([values[indicator][name][subset_name] for name in PRIMARY_NAMES]))
            enrichment = proportion_value / all_prop
            combined[indicator][subset_name] = {
                "equal_weight_proportion": proportion_value,
                "enrichment_vs_all": enrichment,
                "source_proportions": {name: values[indicator][name][subset_name] for name in PRIMARY_NAMES},
            }
            rows.append(
                {
                    "indicator": indicator,
                    "source": "EQUAL_WEIGHT_PRIMARY",
                    "subset": subset_name,
                    "n": sum(math.ceil(len(primary[name]) * ({"Top25": 0.25, "Top10": 0.1, "Top5": 0.05}.get(subset_name, 1.0))) for name in PRIMARY_NAMES),
                    "upper_count": "",
                    "proportion": proportion_value,
                    "enrichment": enrichment,
                    "weighting": "equal_source_weight",
                }
            )
    gates = {}
    for label, indicator in (("H-H1", "hidden_512"), ("L-H1", "l2_u_ge_0.9")):
        top10 = combined[indicator]["Top10"]
        top5_prop = combined[indicator]["Top5"]["equal_weight_proportion"]
        top10_prop = top10["equal_weight_proportion"]
        all_prop = combined[indicator]["All"]["equal_weight_proportion"]
        source_enrichment = {
            name: values[indicator][name]["Top10"] / values[indicator][name]["All"]
            for name in PRIMARY_NAMES
        }
        passed = (
            top10["enrichment_vs_all"] >= 1.5
            and top5_prop >= top10_prop >= all_prop
            and all(value > 1.0 for value in source_enrichment.values())
        )
        gates[label] = {
            "valid": True,
            "passed": bool(passed),
            "equal_weight_top10_enrichment": top10["enrichment_vs_all"],
            "equal_weight_all_proportion": all_prop,
            "equal_weight_top10_proportion": top10_prop,
            "equal_weight_top5_proportion": top5_prop,
            "source_top10_enrichment": source_enrichment,
        }
    return rows, {"indicators": combined, "gates": gates}


def q90(values: Sequence[float]) -> float:
    require(len(values) > 0, "Q90 requested for empty values")
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.9, method="linear"))


def _continuous_bin(value: float) -> int:
    require(0.0 <= value <= 1.0, "continuous bin value outside [0,1]")
    return min(int(value * 5.0), 4)


def q90_specifications() -> dict[str, tuple[Callable[[Mapping[str, Any]], Any], Any, Any, list[Any]]]:
    return {
        "hidden": (lambda row: int(row["hidden"]), 512, 256, list(FROZEN_HIDDEN)),
        "l2": (lambda row: float(row["l2"]), 1e-3, 5e-4, list(FROZEN_L2)),
        "lr_u": (lambda row: _continuous_bin(float(row["u_lr"])), 4, 3, list(range(5))),
        "dropout_u": (lambda row: _continuous_bin(float(row["u_dropout"])), 4, 3, list(range(5))),
        "depth": (lambda row: int(row["effective_layers"]), 5, 4, list(range(1, 6))),
    }


def bootstrap_equal_weight_q90_gap(
    primary: Mapping[str, Sequence[Mapping[str, Any]]],
    bin_function: Callable[[Mapping[str, Any]], Any],
    high_bin: Any,
    low_bin: Any,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    samples = []
    for _ in range(replicates):
        gaps = []
        valid = True
        for name in PRIMARY_NAMES:
            corpus = primary[name]
            indices = rng.integers(0, len(corpus), size=len(corpus))
            high = [float(corpus[index]["mean_val"]) for index in indices if bin_function(corpus[index]) == high_bin]
            low = [float(corpus[index]["mean_val"]) for index in indices if bin_function(corpus[index]) == low_bin]
            if not high or not low:
                valid = False
                break
            gaps.append(q90(high) - q90(low))
        if valid:
            samples.append(float(np.mean(gaps)))
    require(samples, "all Q90 bootstrap draws were invalid")
    return {
        "draws": replicates,
        "valid_draws": len(samples),
        "ci95_lower": float(np.quantile(samples, 0.025, method="linear")),
        "ci95_upper": float(np.quantile(samples, 0.975, method="linear")),
        "unit": "configuration",
        "role": "robustness_only_not_gate",
    }


def q90_analysis(
    primary: Mapping[str, Sequence[Mapping[str, Any]]],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for dimension, (bin_function, high_bin, low_bin, bins) in q90_specifications().items():
        source_gaps = {}
        combined_counts = {high_bin: 0, low_bin: 0}
        for name in PRIMARY_NAMES:
            corpus = primary[name]
            by_bin = {bin_value: [float(row["mean_val"]) for row in corpus if bin_function(row) == bin_value] for bin_value in bins}
            for bin_value in bins:
                values = by_bin[bin_value]
                output_rows.append(
                    {
                        "dimension": dimension,
                        "source": name,
                        "bin": bin_value,
                        "n": len(values),
                        "q90": "" if not values else q90(values),
                    }
                )
            combined_counts[high_bin] += len(by_bin[high_bin])
            combined_counts[low_bin] += len(by_bin[low_bin])
            source_gaps[name] = None if not by_bin[high_bin] or not by_bin[low_bin] else q90(by_bin[high_bin]) - q90(by_bin[low_bin])
        valid = combined_counts[high_bin] >= 20 and combined_counts[low_bin] >= 20 and all(value is not None for value in source_gaps.values())
        equal_gap = None if not valid else float(np.mean(list(source_gaps.values())))
        bootstrap = None
        if valid and replicates > 0:
            bootstrap = bootstrap_equal_weight_q90_gap(primary, bin_function, high_bin, low_bin, replicates, rng)
        summaries[dimension] = {
            "high_bin": high_bin,
            "penultimate_bin": low_bin,
            "combined_high_n": combined_counts[high_bin],
            "combined_penultimate_n": combined_counts[low_bin],
            "source_gaps": source_gaps,
            "equal_weight_gap": equal_gap,
            "valid": valid,
            "bootstrap": bootstrap,
        }
    gates = {}
    for label, dimension in (("H-H2", "hidden"), ("L-H2", "l2")):
        summary = summaries[dimension]
        passed = bool(
            summary["valid"]
            and summary["equal_weight_gap"] >= 0.003
            and all(value > 0.0 for value in summary["source_gaps"].values())
        )
        gates[label] = {**summary, "passed": passed}
    return output_rows, {"dimensions": summaries, "gates": gates}


def greedy_pairs(
    corpus: Sequence[Mapping[str, Any]], target: str
) -> list[dict[str, Any]]:
    require(target in ("hidden", "l2"), "unsupported matching target")
    candidates = []
    by_architecture: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in corpus:
        by_architecture[str(row["architecture_fingerprint"])].append(row)
    for architecture, rows in by_architecture.items():
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                left, right = rows[left_index], rows[right_index]
                if target == "hidden":
                    if left["hidden"] == right["hidden"]:
                        continue
                    coordinates = ("u_lr", "u_dropout", "u_l2")
                    left_target, right_target = float(left["hidden"]), float(right["hidden"])
                else:
                    if math.isclose(float(left["l2"]), float(right["l2"]), rel_tol=0.0, abs_tol=1e-15):
                        continue
                    if left["hidden"] != right["hidden"]:
                        continue
                    coordinates = ("u_lr", "u_dropout")
                    left_target, right_target = float(left["l2"]), float(right["l2"])
                differences = [abs(float(left[key]) - float(right[key])) for key in coordinates]
                exact = all(value <= 1e-12 for value in differences)
                if not exact and not all(value <= 0.10 for value in differences):
                    continue
                distance = float(math.sqrt(sum(value * value for value in differences)))
                smaller, larger = (left, right) if left_target < right_target else (right, left)
                candidates.append(
                    {
                        "priority": 0 if exact else 1,
                        "distance": distance,
                        "architecture_fingerprint": architecture,
                        "smaller": smaller,
                        "larger": larger,
                        "smaller_fingerprint": str(smaller["joint_fingerprint"]),
                        "larger_fingerprint": str(larger["joint_fingerprint"]),
                        "match_type": "exact" if exact else "nearest",
                    }
                )
    candidates.sort(
        key=lambda row: (
            row["priority"],
            row["distance"],
            row["smaller_fingerprint"],
            row["larger_fingerprint"],
        )
    )
    used = set()
    selected = []
    for candidate in candidates:
        pair_ids = {candidate["smaller_fingerprint"], candidate["larger_fingerprint"]}
        if pair_ids & used:
            continue
        used.update(pair_ids)
        gain = float(candidate["larger"]["mean_val"] - candidate["smaller"]["mean_val"])
        selected.append(
            {
                "target": target,
                "source": candidate["smaller"]["source"],
                "architecture_fingerprint": candidate["architecture_fingerprint"],
                "match_type": candidate["match_type"],
                "distance": candidate["distance"],
                "smaller_fingerprint": candidate["smaller_fingerprint"],
                "larger_fingerprint": candidate["larger_fingerprint"],
                "smaller_target": candidate["smaller"][target],
                "larger_target": candidate["larger"][target],
                "smaller_mean_val": candidate["smaller"]["mean_val"],
                "larger_mean_val": candidate["larger"]["mean_val"],
                "gain": gain,
            }
        )
    return selected


def matched_analysis(
    primary: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_pairs = []
    target_summaries = {}
    for target, gate_name in (("hidden", "H-H3"), ("l2", "L-H3")):
        per_source = {}
        target_pairs = []
        for name in PRIMARY_NAMES:
            pairs = greedy_pairs(primary[name], target)
            target_pairs.extend(pairs)
            gains = [float(row["gain"]) for row in pairs]
            wins = sum(value > 0.0 for value in gains)
            ties = sum(value == 0.0 for value in gains)
            per_source[name] = {
                "pair_count": len(gains),
                "mean_gain": None if not gains else float(np.mean(gains)),
                "win_rate": None if not gains else float((wins + 0.5 * ties) / len(gains)),
                "wins": wins,
                "ties": ties,
                "losses": len(gains) - wins - ties,
            }
        all_pairs.extend(target_pairs)
        total_pairs = sum(value["pair_count"] for value in per_source.values())
        valid = total_pairs >= 20 and all(value["pair_count"] > 0 for value in per_source.values())
        equal_gain = None if not valid else float(np.mean([value["mean_gain"] for value in per_source.values()]))
        equal_win = None if not valid else float(np.mean([value["win_rate"] for value in per_source.values()]))
        no_reversal = all(
            value["pair_count"] < 10 or (value["mean_gain"] is not None and value["mean_gain"] > 0.0)
            for value in per_source.values()
        )
        passed = bool(valid and equal_win >= 0.60 and equal_gain >= 0.003 and no_reversal)
        target_summaries[target] = {
            "status": "VALID" if valid else "INSUFFICIENT_MATCHED_PAIRS",
            "valid": valid,
            "pair_count": total_pairs,
            "equal_weight_mean_gain": equal_gain,
            "equal_weight_win_rate": equal_win,
            "source_results": per_source,
            "source_reversal_guard": no_reversal,
            "gate": gate_name,
            "passed": passed,
        }
    return all_pairs, target_summaries


def interaction_analysis(
    primary: Mapping[str, Sequence[Mapping[str, Any]]],
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    cells = ((512, 1e-3), (256, 1e-3), (512, 5e-4))
    labels = ("high_high", "hidden_adjacent", "l2_adjacent")
    source_cells = {}
    combined_counts = Counter()
    source_gaps = {}
    for name in PRIMARY_NAMES:
        by_cell = {}
        for label, cell in zip(labels, cells):
            values = [float(row["mean_val"]) for row in primary[name] if (row["hidden"], row["l2"]) == cell]
            by_cell[label] = {"n": len(values), "q90": None if not values else q90(values)}
            combined_counts[label] += len(values)
        source_cells[name] = by_cell
        if all(by_cell[label]["n"] > 0 for label in labels):
            source_gaps[name] = {
                "vs_hidden_adjacent": by_cell["high_high"]["q90"] - by_cell["hidden_adjacent"]["q90"],
                "vs_l2_adjacent": by_cell["high_high"]["q90"] - by_cell["l2_adjacent"]["q90"],
            }
        else:
            source_gaps[name] = None
    valid = all(combined_counts[label] >= 20 for label in labels) and all(source_gaps[name] is not None for name in PRIMARY_NAMES)
    equal_gaps = None
    if valid:
        equal_gaps = {
            direction: float(np.mean([source_gaps[name][direction] for name in PRIMARY_NAMES]))
            for direction in ("vs_hidden_adjacent", "vs_l2_adjacent")
        }
    no_reversal = all(
        value is None or all(gap >= 0.0 for gap in value.values())
        for value in source_gaps.values()
    )
    interaction = bool(valid and all(gap >= 0.003 for gap in equal_gaps.values()) and no_reversal)
    bootstrap = None
    if valid and replicates > 0:
        samples = {"vs_hidden_adjacent": [], "vs_l2_adjacent": []}
        for _ in range(replicates):
            draw_gaps = defaultdict(list)
            draw_valid = True
            for name in PRIMARY_NAMES:
                corpus = primary[name]
                indices = rng.integers(0, len(corpus), size=len(corpus))
                by_cell_values = {}
                for label, cell in zip(labels, cells):
                    values = [float(corpus[index]["mean_val"]) for index in indices if (corpus[index]["hidden"], corpus[index]["l2"]) == cell]
                    if not values:
                        draw_valid = False
                        break
                    by_cell_values[label] = values
                if not draw_valid:
                    break
                draw_gaps["vs_hidden_adjacent"].append(q90(by_cell_values["high_high"]) - q90(by_cell_values["hidden_adjacent"]))
                draw_gaps["vs_l2_adjacent"].append(q90(by_cell_values["high_high"]) - q90(by_cell_values["l2_adjacent"]))
            if draw_valid:
                for direction in samples:
                    samples[direction].append(float(np.mean(draw_gaps[direction])))
        bootstrap = {
            direction: {
                "draws": replicates,
                "valid_draws": len(values),
                "ci95_lower": float(np.quantile(values, 0.025, method="linear")),
                "ci95_upper": float(np.quantile(values, 0.975, method="linear")),
                "role": "robustness_only_not_gate",
            }
            for direction, values in samples.items()
            if values
        }
    return {
        "cells": source_cells,
        "combined_counts": dict(combined_counts),
        "source_q90_gaps": source_gaps,
        "equal_weight_q90_gaps": equal_gaps,
        "valid": valid,
        "status": "VALID" if valid else "INSUFFICIENT_INTERACTION_CELLS",
        "source_reversal_guard": no_reversal,
        "interaction": interaction,
        "bootstrap": bootstrap,
    }


def secondary_analysis(
    secondary: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    sources = {}
    for name in SECONDARY_NAMES:
        corpus = sorted(secondary[name], key=lambda row: (-float(row["mean_val"]), str(row["joint_fingerprint"])))
        top = corpus[: math.ceil(0.10 * len(corpus))]
        all_upper = sum(row["hidden"] == 512 for row in corpus) / len(corpus)
        top_upper = sum(row["hidden"] == 512 for row in top) / len(top)
        hidden_512 = [float(row["mean_val"]) for row in corpus if row["hidden"] == 512]
        hidden_256 = [float(row["mean_val"]) for row in corpus if row["hidden"] == 256]
        sources[name] = {
            "configuration_count": len(corpus),
            "hidden_counts": dict(Counter(str(row["hidden"]) for row in corpus)),
            "l2_counts": dict(Counter(format(row["l2"], ".12g") for row in corpus)),
            "top10_hidden_512_proportion": top_upper,
            "all_hidden_512_proportion": all_upper,
            "top10_hidden_enrichment": None if all_upper == 0.0 else top_upper / all_upper,
            "hidden_512_minus_256_q90": None if not hidden_512 or not hidden_256 else q90(hidden_512) - q90(hidden_256),
        }
    return {
        "scientific_role": "HIGH_REGIME_SECONDARY_NON_AUTHORIZING",
        "boundary_gate_contribution": False,
        "configuration_count": sum(len(secondary[name]) for name in SECONDARY_NAMES),
        "sources": sources,
        "warning": "All configurations were generated by the high-regime prior; L2 is mechanically fixed at 1e-3.",
    }


def training_audit(
    records_by_source: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detailed_rows = []
    source_results = {}
    for name in (*PRIMARY_NAMES, *SECONDARY_NAMES):
        records = list(records_by_source[name])
        successful = [row for row in records if row["terminal_status"] == "success" and row["valid"] is True]
        complete = [row for row in successful if all(row.get(field) is not None for field in REQUIRED_TRAINING_FIELDS)]
        coverage = len(complete) / len(successful) if successful else 0.0
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in successful:
            groups[str(row["joint_fingerprint"])].append(row)
        config_means = []
        for joint, rows in groups.items():
            require(sorted(row["seed"] for row in rows) == list(FROZEN_SEEDS), f"training source incomplete config: {name}/{joint}")
            config_means.append((float(np.mean([row["val_acc"] for row in rows])), joint))
        config_means.sort(key=lambda value: (-value[0], value[1]))
        top_count = math.ceil(0.10 * len(config_means))
        top_joints = {joint for _, joint in config_means[:top_count]}
        source_detail = []
        for row in complete:
            late_best = int(row["best_epoch"]) >= 0.9 * int(row["eval_epochs"])
            ran_to_max = int(row["stopped_epoch"]) >= int(row["eval_epochs"]) or int(row["epochs_ran"]) >= int(row["eval_epochs"])
            if ran_to_max:
                reason = "max_epoch"
            elif int(row["stopped_epoch"]) - int(row["best_epoch"]) >= int(row["patience"]):
                reason = "patience"
            else:
                reason = "other/unknown"
            output = {
                "source": name,
                "joint_fingerprint": row["joint_fingerprint"],
                "seed": row["seed"],
                "top10": row["joint_fingerprint"] in top_joints,
                "best_epoch": row["best_epoch"],
                "stopped_epoch": row["stopped_epoch"],
                "epochs_ran": row["epochs_ran"],
                "eval_epochs": row["eval_epochs"],
                "patience": row["patience"],
                "late_best": late_best,
                "ran_to_max": ran_to_max,
                "early_stop_reason": reason,
            }
            source_detail.append(output)
            detailed_rows.append(output)
        all_rows = source_detail
        top_rows = [row for row in source_detail if row["top10"]]
        def fraction(rows: Sequence[Mapping[str, Any]], field: str) -> float:
            return float(sum(bool(row[field]) for row in rows) / len(rows))

        def cluster_fraction(rows: Sequence[Mapping[str, Any]], field: str) -> float:
            clustered: dict[str, list[bool]] = defaultdict(list)
            for row in rows:
                clustered[str(row["joint_fingerprint"])].append(bool(row[field]))
            return float(np.mean([np.mean(values) for values in clustered.values()]))

        all_available = bool(all_rows)
        top_available = bool(top_rows)
        source_results[name] = {
            "configuration_count": len(groups),
            "successful_seed_runs": len(successful),
            "complete_seed_runs": len(complete),
            "complete_field_coverage": coverage,
            "top10_configuration_count": top_count,
            "top10_complete_seed_runs": len(top_rows),
            "all_late_best_fraction": fraction(all_rows, "late_best") if all_available else None,
            "top10_late_best_fraction": fraction(top_rows, "late_best") if top_available else None,
            "all_ran_to_max_fraction": fraction(all_rows, "ran_to_max") if all_available else None,
            "top10_ran_to_max_fraction": fraction(top_rows, "ran_to_max") if top_available else None,
            "configuration_cluster_sensitivity": {
                "all_late_best_fraction": cluster_fraction(all_rows, "late_best") if all_available else None,
                "top10_late_best_fraction": cluster_fraction(top_rows, "late_best") if top_available else None,
                "all_ran_to_max_fraction": cluster_fraction(all_rows, "ran_to_max") if all_available else None,
                "top10_ran_to_max_fraction": cluster_fraction(top_rows, "ran_to_max") if top_available else None,
            },
            "early_stop_reason_counts": dict(Counter(row["early_stop_reason"] for row in all_rows)),
        }
    sufficient = all(
        result["complete_field_coverage"] >= 0.95 and result["top10_complete_seed_runs"] >= 30
        for result in source_results.values()
    )
    aggregate = {}
    for field in (
        "all_late_best_fraction",
        "top10_late_best_fraction",
        "all_ran_to_max_fraction",
        "top10_ran_to_max_fraction",
    ):
        values = [source_results[name][field] for name in (*PRIMARY_NAMES, *SECONDARY_NAMES)]
        aggregate[field] = None if any(value is None for value in values) else float(np.mean(values))
    aggregate["configuration_cluster_sensitivity"] = {
        field: (
            None
            if any(source_results[name]["configuration_cluster_sensitivity"][field] is None for name in (*PRIMARY_NAMES, *SECONDARY_NAMES))
            else float(np.mean([source_results[name]["configuration_cluster_sensitivity"][field] for name in (*PRIMARY_NAMES, *SECONDARY_NAMES)]))
        )
        for field in (
            "all_late_best_fraction",
            "top10_late_best_fraction",
            "all_ran_to_max_fraction",
            "top10_ran_to_max_fraction",
        )
    }
    all_late = aggregate["all_late_best_fraction"]
    top_late = aggregate["top10_late_best_fraction"]
    top_max = aggregate["top10_ran_to_max_fraction"]
    ratio = None if all_late in (None, 0.0) or top_late is None else top_late / all_late
    t1 = top_late is not None and top_late >= 0.40
    t2 = ratio is not None and ratio >= 1.5
    t3 = top_max is not None and top_max >= 0.50
    direction_count = sum(
        source_results[name]["top10_late_best_fraction"] is not None
        and source_results[name]["top10_late_best_fraction"] >= source_results[name]["all_late_best_fraction"]
        for name in (*PRIMARY_NAMES, *SECONDARY_NAMES)
    )
    raw_signal = bool(t1 and (t2 or t3))
    signal = bool(sufficient and raw_signal and direction_count >= 3)
    result = {
        "status": "TRAINING_AUDIT_SUFFICIENT" if sufficient else "TRAINING_AUDIT_INSUFFICIENT",
        "sufficient": sufficient,
        "source_results": source_results,
        "equal_weight_aggregate": aggregate,
        "T1": {"passed": t1, "top10_late_best_fraction": top_late, "threshold": ">=0.40"},
        "T2": {"passed": t2, "top10_to_all_late_best_ratio": ratio, "threshold": ">=1.5"},
        "T3": {"passed": t3, "top10_ran_to_max_fraction": top_max, "threshold": ">=0.50"},
        "raw_training_truncation_signal": raw_signal,
        "source_direction_count": direction_count,
        "source_direction_guard_passed": direction_count >= 3,
        "training_truncation_signal": signal,
    }
    return detailed_rows, result


def classify_decision(
    top: Mapping[str, Any],
    q90_result: Mapping[str, Any],
    matched: Mapping[str, Any],
    interaction: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    gates = {
        "H-H1": top["gates"]["H-H1"],
        "H-H2": q90_result["gates"]["H-H2"],
        "H-H3": matched["hidden"],
        "L-H1": top["gates"]["L-H1"],
        "L-H2": q90_result["gates"]["L-H2"],
        "L-H3": matched["l2"],
    }
    hidden_passes = sum(bool(gates[name]["passed"]) for name in ("H-H1", "H-H2", "H-H3"))
    l2_passes = sum(bool(gates[name]["passed"]) for name in ("L-H1", "L-H2", "L-H3"))
    hidden_strong = hidden_passes >= 2 and (gates["H-H1"]["passed"] or gates["H-H2"]["passed"])
    l2_supportive = l2_passes >= 2 or interaction["interaction"] is True
    boundary = bool(hidden_strong and l2_supportive)
    training_signal = bool(training["training_truncation_signal"])
    sufficient = bool(training["sufficient"])
    if boundary and training_signal and sufficient:
        classification = "BOUNDARY_AND_TRAINING_BOTH_PLAUSIBLE"
    elif boundary:
        classification = "HP_BOUNDARY_EXPANSION_AUTHORIZED"
    elif training_signal and sufficient:
        classification = "TRAINING_CEILING_PRIORITY"
    elif sufficient:
        classification = "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL"
    else:
        classification = "AUDIT_INPUT_INCOMPATIBLE"
    return {
        "classification": classification,
        "boundary": boundary,
        "training": training_signal,
        "training_sufficient": sufficient,
        "training_status": training["status"],
        "HiddenStrong": hidden_strong,
        "L2Supportive": l2_supportive,
        "hidden_gate_pass_count": hidden_passes,
        "l2_gate_pass_count": l2_passes,
        "interaction": interaction["interaction"],
        "depth_6_7_authorized": False,
        "gates": gates,
        "practical_margin": 0.003,
        "phase_b_started": False,
        "test_accessed": False,
        "new_gnn_runs": 0,
        "gpu_compute_runs": 0,
    }


def validate_all_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_frozen_config(config)
    firewall = static_zero_compute_firewall(config)
    decoder = verify_decoder_sources(config)
    archive_audits = []
    dependency_audits = {}
    protocol_audits = {}
    generator_audits = {}
    generator_sets = {}
    primary_metadata_audits = {}
    primary_configurations = {}
    secondary_configurations = {}
    records_by_source = {}
    required_dependencies = config["required_dependency_hashes"]
    expected = config["compatibility"]
    for source in config["input_sources"]:
        archive_audits.append(validate_source_archive(source))
        dependency_audits[source["name"]] = validate_dependency_identity(source, required_dependencies)
        if source["role"] == "primary":
            protocol_audits[source["name"]] = validate_primary_protocol_metadata(source, expected)
            metadata, metadata_audit = validate_primary_metadata(source)
            primary_metadata_audits[source["name"]] = metadata_audit
            configurations, records = load_primary_results(source, metadata, expected)
            primary_configurations[source["name"]] = configurations
            records_by_source[source["name"]] = records
        else:
            protocol_audits[source["name"]] = validate_secondary_protocol_metadata(source, expected)
            generator_audits[source["name"]], generator_sets[source["name"]] = validate_high_regime_generator(source)
            configurations, records = load_secondary_results(source, expected)
            evaluated_set = {row["joint_fingerprint"] for row in configurations}
            validate_evaluated_generator_membership(
                evaluated_set, generator_sets[source["name"]], str(source["name"])
            )
            generator_audits[source["name"]]["evaluated_configuration_count"] = len(evaluated_set)
            generator_audits[source["name"]]["evaluated_subset_of_generator"] = True
            generator_audits[source["name"]]["evaluated_joint_set_sha256"] = stable_hash(sorted(evaluated_set))
            secondary_configurations[source["name"]] = [enrich_configuration(row) for row in configurations]
            records_by_source[source["name"]] = records
    dependency_values = list(dependency_audits.values())
    require(all(value == dependency_values[0] for value in dependency_values[1:]), "model/evaluator dependency identities differ")
    primary_deduped, duplicate_audit = deduplicate_primary(primary_configurations)
    secondary_sets = {name: {row["joint_fingerprint"] for row in secondary_configurations[name]} for name in SECONDARY_NAMES}
    for left_index, left in enumerate(SECONDARY_NAMES):
        for right in SECONDARY_NAMES[left_index + 1 :]:
            require(not (secondary_sets[left] & secondary_sets[right]), f"secondary source overlap: {left}/{right}")
    secondary_union = [row for name in SECONDARY_NAMES for row in secondary_configurations[name]]
    require(len(secondary_union) == 1121, "secondary union must contain exactly 1121 configurations")
    fresh_audit = validate_fresh_identity(config, secondary_union)
    return {
        "archive_audits": archive_audits,
        "dependency_audits": dependency_audits,
        "protocol_audits": protocol_audits,
        "decoder_audit": decoder,
        "firewall": firewall,
        "generator_audits": generator_audits,
        "primary_metadata_audits": primary_metadata_audits,
        "primary_duplicate_audit": duplicate_audit,
        "fresh_identity_audit": fresh_audit,
        "primary": primary_deduped,
        "secondary": secondary_configurations,
        "records_by_source": records_by_source,
    }


def protocol_bytes_and_sha(config: Mapping[str, Any]) -> tuple[bytes, str]:
    payload = json_bytes(config)
    return payload, hashlib.sha256(payload).hexdigest()


def bootstrap_seed_from_protocol(protocol_sha256: str) -> int:
    require(re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is not None, "invalid protocol SHA")
    return int.from_bytes(bytes.fromhex(protocol_sha256)[:8], byteorder="big", signed=False)


def synthetic_bootstrap_benchmark(replicates: int = 250) -> float:
    synthetic = {}
    for source_index, name in enumerate(PRIMARY_NAMES):
        rows = []
        for index in range(192):
            hidden = FROZEN_HIDDEN[index % len(FROZEN_HIDDEN)]
            l2 = FROZEN_L2[index % len(FROZEN_L2)]
            rows.append(
                {
                    "source": name,
                    "joint_fingerprint": f"{source_index:02x}{index:062x}",
                    "architecture_fingerprint": f"{source_index + 2:02x}{index:062x}",
                    "hidden": hidden,
                    "l2": l2,
                    "effective_layers": 4 + (index % 2),
                    "u_lr": (index % 100) / 99.0,
                    "u_dropout": ((index * 7) % 100) / 99.0,
                    "u_l2": u_log(l2, FROZEN_L2[0], FROZEN_L2[-1]),
                    "mean_val": 0.70 + 0.0001 * index,
                }
            )
        synthetic[name] = rows
    started = time.perf_counter()
    rng = np.random.default_rng(20260830)
    q90_analysis(synthetic, replicates, rng)
    interaction_analysis(synthetic, replicates, rng)
    return time.perf_counter() - started


def synthetic_smoke() -> dict[str, Any]:
    primary = {}
    training_sources = {}
    for source_index, name in enumerate((*PRIMARY_NAMES, *SECONDARY_NAMES)):
        configs = []
        records = []
        n = 120 if name in PRIMARY_NAMES else 60
        for index in range(n):
            hidden = FROZEN_HIDDEN[index % 6]
            l2 = FROZEN_L2[index % 5]
            architecture = f"{source_index + 1:02x}{index // 6:062x}"
            joint = f"{source_index + 10:02x}{index:062x}"
            config = {
                "source": name,
                "joint_fingerprint": joint,
                "architecture_fingerprint": architecture,
                "hp_fingerprint": f"{source_index + 20:02x}{index:062x}",
                "operations": ["GCNConv"] * 5,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
                "raw_hp_vector": [index % 100 / 99.0, index % 97 / 96.0, (index % 6 + 0.1) / 6.0, (index % 5 + 0.1) / 5.0],
                "effective_layers": 5,
                "lr": 1e-4,
                "dropout": 0.1,
                "hidden": hidden,
                "l2": l2,
                "seed_values": [0.75, 0.76, 0.77],
                "mean_val": 0.75 + 0.0002 * index + (0.004 if hidden == 512 else 0.0),
                "u_lr": index % 100 / 99.0,
                "u_dropout": index % 97 / 96.0,
                "u_l2": u_log(l2, FROZEN_L2[0], FROZEN_L2[-1]),
            }
            configs.append(config)
            for seed in FROZEN_SEEDS:
                best = 285 if index >= math.ceil(0.9 * n) else 100
                records.append(
                    {
                        "source": name,
                        "joint_fingerprint": joint,
                        "seed": seed,
                        "val_acc": config["mean_val"],
                        "terminal_status": "success",
                        "valid": True,
                        "best_epoch": best,
                        "stopped_epoch": 300 if best == 285 else 180,
                        "epochs_ran": 300 if best == 285 else 180,
                        "eval_epochs": 300,
                        "patience": 80,
                    }
                )
        if name in PRIMARY_NAMES:
            primary[name] = configs
        training_sources[name] = records
    rng = np.random.default_rng(12345)
    top_rows, top = top_tail_analysis(primary, (0.25, 0.10, 0.05))
    q_rows, q_result = q90_analysis(primary, 25, rng)
    pairs, matched = matched_analysis(primary)
    interaction = interaction_analysis(primary, 25, rng)
    training_rows, training = training_audit(training_sources)
    decision = classify_decision(top, q_result, matched, interaction, training)
    require(top_rows and q_rows and training_rows, "synthetic smoke generated empty outputs")
    require(decision["classification"] in {
        "BOUNDARY_AND_TRAINING_BOTH_PLAUSIBLE",
        "HP_BOUNDARY_EXPANSION_AUTHORIZED",
        "TRAINING_CEILING_PRIORITY",
        "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL",
        "AUDIT_INPUT_INCOMPATIBLE",
    }, "synthetic classification invalid")
    return {
        "passed": True,
        "top_tail_rows": len(top_rows),
        "q90_rows": len(q_rows),
        "matched_pairs": len(pairs),
        "training_rows": len(training_rows),
        "classification": decision["classification"],
        "gpu": 0,
        "new_gnn": 0,
        "test_access": 0,
    }


def collect_environment() -> dict[str, Any]:
    packages = {}
    for package in ("torch", "torch-geometric", "ogb", "numpy", "scipy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "recorded_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_inventory_read_only": command_output([
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ]),
        "tmux_sessions_read_only": command_output(["tmux", "list-sessions", "-F", "#{session_name}"]),
        "disk_bytes": {"total": disk.total, "used": disk.used, "free": disk.free},
        "gpu_compute_used": False,
    }


def git_snapshot() -> dict[str, Any]:
    return {
        "branch": command_output(["git", "branch", "--show-current"]),
        "head": command_output(["git", "rev-parse", "HEAD"]),
        "status_porcelain": command_output(["git", "status", "--porcelain=v1", "--untracked-files=all"]),
    }


def git_snapshot_text(snapshot: Mapping[str, Any]) -> str:
    return (
        f"branch={snapshot['branch']['output'].strip()}\n"
        f"head={snapshot['head']['output'].strip()}\n"
        "status_porcelain:\n"
        f"{snapshot['status_porcelain']['output']}"
    )


def _input_bytes(validation: Mapping[str, Any], config_path: Path) -> int:
    total = config_path.stat().st_size + Path(__file__).stat().st_size
    total += (REPO_ROOT / RUNNER_RELATIVE).stat().st_size
    for row in validation["archive_audits"]:
        total += int(row["archive_size_bytes"])
        total += int(row["manifest_verified_bytes"])
    fresh = repo_path(str(validation["fresh_identity_audit"]["path"]))
    total += fresh.stat().st_size
    return total


def prepare_dry_run(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    output = repo_path(str(config["output"]))
    require_engineering(not os.path.lexists(output), f"result directory or symlink already exists: {output}")
    require_engineering(output.parent.is_dir() and not output.parent.is_symlink(), "result parent is missing or a symlink")
    require_engineering(os.access(output.parent, os.W_OK | os.X_OK), "result parent is not writable")
    for target_name in (f"{EXPERIMENT}.tar.gz", f"{EXPERIMENT}.tar.gz.sha256"):
        require_engineering(not os.path.lexists(output / target_name), f"output target already exists: {target_name}")
    started = time.perf_counter()
    validation = validate_all_inputs(config)
    benchmark_replicates = 250
    benchmark_seconds = synthetic_bootstrap_benchmark(benchmark_replicates)
    wall_seconds = time.perf_counter() - started
    total_bytes = _input_bytes(validation, config_path)
    scaled_bootstrap = benchmark_seconds * (int(config["bootstrap_replicates"]) / benchmark_replicates)
    projected_seconds = wall_seconds + scaled_bootstrap + 5.0
    best_minutes = max(1.0, round(projected_seconds * 0.75 / 60.0, 1))
    expected_minutes = max(best_minutes, round(projected_seconds * 1.5 / 60.0, 1))
    worst_minutes = max(expected_minutes, round(projected_seconds * 3.0 / 60.0, 1))
    protocol_payload, protocol_sha = protocol_bytes_and_sha(config)
    estimate = {
        "experiment": EXPERIMENT,
        "created_at": utc_now(),
        "best_minutes": best_minutes,
        "expected_minutes": expected_minutes,
        "worst_minutes": worst_minutes,
        "basis": {
            "actual_input_bytes_verified": total_bytes,
            "actual_input_gib_verified": total_bytes / (1024 ** 3),
            "actual_input_size_detail": {
                "config_bytes": config_path.stat().st_size,
                "script_bytes": Path(__file__).stat().st_size,
                "runner_bytes": (REPO_ROOT / RUNNER_RELATIVE).stat().st_size,
                "fresh_identity_bytes": repo_path(str(validation["fresh_identity_audit"]["path"])).stat().st_size,
                "sources": {
                    row["name"]: {
                        "archive_bytes": row["archive_size_bytes"],
                        "verified_live_manifest_payload_bytes": row["manifest_verified_bytes"],
                    }
                    for row in validation["archive_audits"]
                },
            },
            "dry_run_wall_seconds": wall_seconds,
            "synthetic_bootstrap_replicates": benchmark_replicates,
            "synthetic_bootstrap_wall_seconds": benchmark_seconds,
            "formal_bootstrap_replicates": int(config["bootstrap_replicates"]),
            "projection": "dry-run validation wall + linearly scaled full analysis bootstrap + 5s packaging; best/expected/worst multipliers 0.75/1.5/3.0",
        },
        "dry_run_passed": True,
        "all_inputs_and_output_targets_validated": True,
        "zero_gnn_test_gpu_firewall": validation["firewall"],
        "source_archive_hashes": {row["name"]: row["archive_sha256"] for row in validation["archive_audits"]},
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(REPO_ROOT / RUNNER_RELATIVE),
        "frozen_protocol_sha256": protocol_sha,
        "frozen_protocol_size_bytes": len(protocol_payload),
        "result_directory_was_absent": True,
        "result_directory_initialized_by_this_dry_run": True,
        "output_parent_writable": True,
        "output_targets_absent": True,
        "formal_run_started": False,
        "phase_b_started": False,
    }
    output.mkdir(parents=False, exist_ok=False)
    dump_json(output / "pre_execution_estimate.json", estimate)
    return estimate


def _validate_formal_start(output: Path, estimate: Mapping[str, Any], config_path: Path) -> None:
    require_engineering(output.is_dir() and not output.is_symlink(), "formal output directory must have been initialized by dry-run")
    entries = sorted(path.name for path in output.iterdir())
    require_engineering(entries == ["pre_execution_estimate.json"], f"formal output directory is not pristine: {entries}")
    require_engineering(estimate.get("dry_run_passed") is True, "dry-run did not pass")
    require_engineering(estimate.get("script_sha256") == sha256_file(Path(__file__).resolve()), "script changed after dry-run")
    require_engineering(estimate.get("config_sha256") == sha256_file(config_path), "config changed after dry-run")
    require_engineering(estimate.get("runner_sha256") == sha256_file(REPO_ROOT / RUNNER_RELATIVE), "runner changed after dry-run")


def _write_source_snapshot(output: Path, config_path: Path) -> None:
    sources = (
        Path(__file__).resolve(),
        config_path,
        REPO_ROOT / "tests/test_boundary_training_ceiling_audit.py",
        REPO_ROOT / "scripts/run_boundary_training_ceiling_audit.sh",
    )
    for source in sources:
        require(source.is_file(), f"source snapshot input missing: {source}")
        relative = source.relative_to(REPO_ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _artifact_manifest(output: Path) -> dict[str, Any]:
    excluded = {"manifest.json", f"{EXPERIMENT}.tar.gz", f"{EXPERIMENT}.tar.gz.sha256"}
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        files.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "created_at": utc_now(),
        "experiment": EXPERIMENT,
        "file_count": len(files),
        "files": files,
        "manifest_root_sha256": _manifest_root(files),
        "archive_excluded_from_manifest": True,
        "archive_self_inclusion": False,
    }


def _create_archive(output: Path) -> tuple[Path, str]:
    archive = output / f"{EXPERIMENT}.tar.gz"
    sidecar = output / f"{EXPERIMENT}.tar.gz.sha256"
    require_engineering(not os.path.lexists(archive) and not os.path.lexists(sidecar), "archive target already exists")
    files = [path for path in sorted(output.rglob("*")) if path.is_file() and path not in (archive, sidecar)]
    temporary = output / f".{EXPERIMENT}.tar.gz.tmp-{os.getpid()}"
    require_engineering(not os.path.lexists(temporary), "temporary archive target already exists")
    expected_members = []
    with tarfile.open(temporary, "w:gz") as handle:
        for path in files:
            arcname = PurePosixPath(EXPERIMENT) / path.relative_to(output).as_posix()
            expected_members.append(str(arcname))
            handle.add(path, arcname=str(arcname), recursive=False)
    with tarfile.open(temporary, "r:gz") as handle:
        actual_members = handle.getnames()
        require_engineering(actual_members == expected_members, "archive member verification failed")
        require_engineering(not any(name.endswith(".tar.gz") or name.endswith(".tar.gz.sha256") for name in actual_members), "archive recursively contains itself")
        for path, member_name in zip(files, expected_members):
            member = handle.getmember(member_name)
            require_engineering(member.isfile() and member.size == path.stat().st_size, f"archive member size mismatch: {member_name}")
            extracted = handle.extractfile(member)
            require_engineering(extracted is not None, f"archive member cannot be read: {member_name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
            require_engineering(digest.hexdigest() == sha256_file(path), f"archive member SHA mismatch: {member_name}")
    os.replace(temporary, archive)
    digest = sha256_file(archive)
    dump_text(sidecar, f"{digest}  {archive.name}\n")
    return archive, digest


def _final_report(
    decision: Mapping[str, Any],
    validation: Mapping[str, Any],
    training: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> str:
    gate_lines = []
    for name in ("H-H1", "H-H2", "H-H3", "L-H1", "L-H2", "L-H3"):
        gate = decision["gates"][name]
        gate_lines.append(f"| {name} | {gate.get('valid')} | {gate.get('passed')} |")
    return f"""# Phase A Boundary + Training Ceiling Audit

Experiment: `{EXPERIMENT}`  
Protocol: `phase-a-e-master-global4-v1`  
Completed: {utc_now()}

## Decision

**{decision['classification']}**

- Boundary: `{decision['boundary']}`
- HiddenStrong: `{decision['HiddenStrong']}`
- L2Supportive: `{decision['L2Supportive']}`
- Training truncation signal: `{decision['training']}`
- Training status: `{decision['training_status']}`
- Phase B started: `false`
- Depth 6/7 authorized: `false`

| Gate | Valid | Passed |
|---|---:|---:|
{chr(10).join(gate_lines)}

## Frozen corpora

- Primary: 192 OH blind-joint-random + 192 HR balanced-random-HP before cross-source deduplication.
- Secondary: 1121 high-regime configurations used only for non-authorizing secondary and training analysis.
- Fresh-ceiling candidate universe was used only for identity/coverage cross-checking; none of its validation columns were used as a new label source.
- All five pinned archives and every live manifest payload entry passed SHA256 and size validation.

## Training audit

- Field coverage sufficient: `{training['sufficient']}`
- T1: `{training['T1']['passed']}`
- T2: `{training['T2']['passed']}`
- T3: `{training['T3']['passed']}`
- Source-direction guard: `{training['source_direction_guard_passed']}` ({training['source_direction_count']}/5)

## Execution firewall and budget

- New GNN runs: `0`
- GPU compute runs: `0`
- Test reads: `0`
- Cora test mask/labels were never loaded.
- `bo_phase4.py`, `eval_utils.py`, old results, and all other historical files were read-only.

## Pre-execution estimate

- Best: {estimate['best_minutes']} minutes
- Expected: {estimate['expected_minutes']} minutes
- Worst: {estimate['worst_minutes']} minutes

The Q90 bootstrap used 50,000 configuration-level draws. Its confidence intervals are robustness reports only and did not alter any frozen gate.
"""


def _run_formal_admitted(
    config: Mapping[str, Any], config_path: Path, output: Path, estimate: Mapping[str, Any]
) -> dict[str, Any]:
    (output / "logs").mkdir()
    log_path = output / "logs/execution.log"
    dump_text(log_path, f"{utc_now()} formal Phase A started; zero GNN/GPU/test budget\n")
    git_before = git_snapshot()
    environment = collect_environment()
    validation = validate_all_inputs(config)
    protocol_payload, protocol_sha = protocol_bytes_and_sha(config)
    require(protocol_sha == estimate["frozen_protocol_sha256"], "frozen protocol changed after dry-run")
    (output / "frozen_protocol.json").write_bytes(protocol_payload)
    rng_seed = bootstrap_seed_from_protocol(protocol_sha)
    rng = np.random.default_rng(rng_seed)
    top_rows, top = top_tail_analysis(validation["primary"], config["top_tail_proportions"])
    q90_rows, q90_result = q90_analysis(validation["primary"], int(config["bootstrap_replicates"]), rng)
    matched_rows, matched = matched_analysis(validation["primary"])
    interaction = interaction_analysis(validation["primary"], int(config["bootstrap_replicates"]), rng)
    secondary = secondary_analysis(validation["secondary"])
    training_rows, training = training_audit(validation["records_by_source"])
    decision = classify_decision(top, q90_result, matched, interaction, training)
    primary_rows = []
    for name in PRIMARY_NAMES:
        for row in validation["primary"][name]:
            primary_rows.append(
                {
                    "source": name,
                    "joint_fingerprint": row["joint_fingerprint"],
                    "architecture_fingerprint": row["architecture_fingerprint"],
                    "hp_fingerprint": row["hp_fingerprint"],
                    "hidden": row["hidden"],
                    "l2": row["l2"],
                    "lr": row["lr"],
                    "dropout": row["dropout"],
                    "effective_layers": row["effective_layers"],
                    "u_lr": row["u_lr"],
                    "u_dropout": row["u_dropout"],
                    "u_l2": row["u_l2"],
                    "seed_0": row["seed_values"][0],
                    "seed_1": row["seed_values"][1],
                    "seed_2": row["seed_values"][2],
                    "mean_val": row["mean_val"],
                }
            )
    input_hashes = {
        "created_at": utc_now(),
        "archives": validation["archive_audits"],
        "decoder": validation["decoder_audit"],
        "dependencies": validation["dependency_audits"],
        "protocol_compatibility": validation["protocol_audits"],
        "fresh_ceiling_identity": validation["fresh_identity_audit"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "frozen_protocol_sha256": protocol_sha,
    }
    corpus_audit = {
        "primary_metadata": validation["primary_metadata_audits"],
        "primary_deduplication": validation["primary_duplicate_audit"],
        "secondary_generators": validation["generator_audits"],
        "fresh_identity": validation["fresh_identity_audit"],
        "primary_authorization_configurations": sum(len(validation["primary"][name]) for name in PRIMARY_NAMES),
        "secondary_non_authorizing_configurations": sum(len(validation["secondary"][name]) for name in SECONDARY_NAMES),
    }
    seed_collision = {
        "status": "NOT_APPLICABLE_ZERO_NEW_GNN",
        "new_seeds_requested": [],
        "new_seeds_selected": [],
        "collision_possible": False,
        "historical_analysis_seeds": list(FROZEN_SEEDS),
        "passed": True,
    }
    attempt_ledger = {
        "planned_scientific_tasks": 0,
        "completed_scientific_tasks": 0,
        "failed_scientific_tasks": 0,
        "retry_tasks": 0,
        "attempts": [],
        "new_gnn_runs": 0,
    }
    dump_json(output / "input_hashes.json", input_hashes)
    dump_json(output / "test_firewall.json", validation["firewall"])
    dump_json(output / "seed_collision_audit.json", seed_collision)
    dump_json(output / "attempt_ledger.json", attempt_ledger)
    dump_json(output / "corpus_audit.json", corpus_audit)
    dump_csv(output / "primary_configurations.csv", primary_rows)
    dump_csv(output / "top_tail_analysis.csv", top_rows)
    dump_json(output / "top_tail_analysis.json", top)
    dump_csv(output / "q90_analysis.csv", q90_rows)
    dump_json(output / "q90_analysis.json", {**q90_result, "bootstrap_seed": rng_seed, "bootstrap_replicates": int(config["bootstrap_replicates"])})
    dump_csv(output / "matched_pairs.csv", matched_rows)
    dump_json(output / "matched_analysis.json", matched)
    dump_json(output / "interaction_analysis.json", interaction)
    dump_json(output / "high_regime_secondary_analysis.json", secondary)
    dump_csv(output / "training_seed_runs.csv", training_rows)
    dump_json(output / "training_audit.json", training)
    dump_json(output / "environment.json", environment)
    dump_text(output / "git_before.txt", git_snapshot_text(git_before))
    dump_json(output / "decision.json", decision)
    state = {
        "experiment": EXPERIMENT,
        "status": "COMPLETE",
        "classification": decision["classification"],
        "started_at": environment["recorded_at"],
        "completed_at": utc_now(),
        "planned": {"analysis_steps": 6, "scientific_tasks": 0, "new_gnn": 0},
        "completed": {"analysis_steps": 6, "scientific_tasks": 0, "new_gnn": 0},
        "failed": {"analysis_steps": 0, "scientific_tasks": 0, "new_gnn": 0},
        "phase_b_started": False,
        "test_accessed": False,
        "gpu_compute_used": False,
    }
    dump_json(output / "state.json", state)
    dump_text(output / "final_report.md", _final_report(decision, validation, training, estimate))
    _write_source_snapshot(output, config_path)
    git_after = git_snapshot()
    dump_text(output / "git_after.txt", git_snapshot_text(git_after))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} analysis complete; classification={decision['classification']}\n")
    manifest = _artifact_manifest(output)
    dump_json(output / "manifest.json", manifest)
    archive, archive_sha = _create_archive(output)
    return {
        "classification": decision["classification"],
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "result": str(output),
        "new_gnn": 0,
        "gpu": 0,
        "test_access": 0,
        "phase_b_started": False,
    }


def _write_failure_bundle(output: Path, classification: str, exc: BaseException) -> None:
    """Best-effort terminal evidence for an admitted formal run; never reruns work."""
    try:
        (output / "logs").mkdir(exist_ok=True)
        dump_text(output / "logs/failure.log", traceback.format_exc())
        state = {
            "experiment": EXPERIMENT,
            "status": classification,
            "classification": classification,
            "failed_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "planned": {"scientific_tasks": 0, "new_gnn": 0},
            "completed": {"scientific_tasks": 0, "new_gnn": 0},
            "failed": {"scientific_tasks": 0, "new_gnn": 0},
            "phase_b_started": False,
            "test_accessed": False,
            "gpu_compute_used": False,
            "automatic_retry": False,
        }
        dump_json(output / "state.json", state)
        dump_json(
            output / "decision.json",
            {
                "classification": classification,
                "scientific_conclusion": None,
                "error": str(exc),
                "phase_b_started": False,
                "test_accessed": False,
                "new_gnn_runs": 0,
                "gpu_compute_runs": 0,
            },
        )
        dump_text(
            output / "final_report.md",
            f"# Phase A Boundary + Training Ceiling Audit\n\n**{classification}**\n\n"
            f"The admitted formal run stopped fail-closed: `{type(exc).__name__}: {exc}`\n\n"
            "No scientific conclusion was issued, no retry was attempted, Phase B was not started, "
            "and the GNN/GPU/test-access counts remain zero.\n",
        )
    except Exception:
        return


def run_formal(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    output = repo_path(str(config["output"]))
    estimate_path = output / "pre_execution_estimate.json"
    require_engineering(estimate_path.is_file(), "pre_execution_estimate.json is missing")
    estimate = load_json(estimate_path)
    _validate_formal_start(output, estimate, config_path)
    try:
        return _run_formal_admitted(config, config_path, output, estimate)
    except ProtocolError as exc:
        _write_failure_bundle(output, "AUDIT_INPUT_INCOMPATIBLE", exc)
        raise
    except BaseException as exc:
        _write_failure_bundle(output, "RUN_FAILED", exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise EngineeringError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_RELATIVE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.synthetic_smoke:
        print(json.dumps(synthetic_smoke(), indent=2, sort_keys=True))
        return 0
    config_path = repo_path(str(args.config))
    require(config_path == REPO_ROOT / CONFIG_RELATIVE, "only the frozen Phase A config is accepted")
    config = load_json(config_path)
    if args.dry_run:
        print(json.dumps(prepare_dry_run(config, config_path), indent=2, sort_keys=True))
        return 0
    print(json.dumps(run_formal(config, config_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as exc:
        print(f"AUDIT_INPUT_INCOMPATIBLE: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except EngineeringError as exc:
        print(f"RUN_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(3)
    except Exception as exc:
        print(f"RUN_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(3)
