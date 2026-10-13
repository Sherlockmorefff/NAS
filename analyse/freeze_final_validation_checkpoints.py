#!/usr/bin/env python3
"""Prepare frozen validation-only checkpoints without accessing held-out data."""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import final_checkpoint_seed_inventory as seed_inventory
from analyse import original_head_confirmatory_attribution_v2 as v2
from analyse import original_head_deterministic_batch_pilot as v1
from analyse.structure_native_surrogate_pilot import realize_global4_hp
from architecture_diagnostics import architecture_fingerprint


FORMAT_VERSION = "frozen_validation_checkpoints_global4_v1"
OUTPUT_RELATIVE = "results/final/frozen-validation-checkpoints-global4-v1"
INPUT_RELATIVE = "results/search/high-regime-prior-model-blind-validation-global4-v1"
INPUT_ARCHIVE_NAME = "high-regime-prior-model-blind-validation-global4-v1-20260826.tar.gz"
INPUT_ARCHIVE_SHA256 = "050a64b0cde5bdb17ababa9833e5ee578b448c9b19da31fbb00f0251ec82491c"
EXPECTED_CLASSIFICATION = "HIGH_REGIME_PRIOR_CONFIRMED_HEAD_REDUNDANT"
EXPECTED_P = "30e0bb108c1fdbe8db186152a09c3f2fc6e812be189db920fe4d0d2b0d3cb1a2"
EXPECTED_Q = "c624ab1284e71535f63dbe2bf02f1aa1ee4bdd1d722331162416cbf49a22c39d"
METHOD_NAME = "model-blind architecture search with a high-regime HP prior"
FRESH_SEEDS = tuple(range(100000, 100030))
MASTER_SEED = 20260827
GPU_UUIDS = (
    "GPU-5496be78-3b71-4978-3960-b6ed926ebe49",
    "GPU-197ae2b9-74cd-6afc-446d-7bc9e98176f9",
)
DATASET_CONTENT_ID = "373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7"
PUBLIC_SPLIT_ID = "bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09"
EXPECTED_CORA_NUM_NODES = 2708
EXPECTED_CORA_NUM_FEATURES = 1433
EXPECTED_CORA_EDGE_ENTRIES = 10556
EXPECTED_CORA_TRAIN_COUNT = 140
EXPECTED_CORA_VALIDATION_COUNT = 500
SEALED_KEYS = {
    "x", "edge_index", "train_mask", "val_mask", "y_sealed",
    "num_features", "num_classes",
}
SOURCE_ENTRY_FILES = (
    "analyse/final_checkpoint_seed_inventory.py",
    "analyse/freeze_final_validation_checkpoints.py",
    "analyse/run_final_validation_checkpoints.py",
    "analyse/audit_final_reveal_ready.py",
    "analyse/reveal_frozen_checkpoints_once.py",
    "analyse/analyse_final_checkpoint_reveal.py",
    "analyse/frozen_cora_benchmark_reveal_adapter.py",
    "tests/test_final_checkpoint_reveal.py",
)
FINGERPRINT_SOURCE_FILES = (
    "architecture_diagnostics.py",
    "analyse/original_head_deterministic_batch_pilot.py",
    "analyse/original_head_confirmatory_attribution_v2.py",
    "analyse/structure_native_surrogate_pilot.py",
    "eval_utils.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_input_json_without_test_values(path: Path) -> tuple[Any, dict[str, Any]]:
    """Parse frozen evidence while lexically skipping any test-token value."""
    text = path.read_text(encoding="utf-8")
    value, skipped_path_hashes = seed_inventory.safe_load_json(text, str(path))
    return value, {
        "file": file_identity(path),
        "skipped_test_value_count": len(skipped_path_hashes),
        "skipped_test_field_path_hashes": sorted(skipped_path_hashes),
        "test_values_deserialized": False,
    }


def _run_text(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    return result.stdout.strip()


def _contains_forbidden_metric_key(value: Any) -> list[str]:
    hits: list[str] = []
    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                next_path = f"{path}.{key}"
                if "test" in str(key).lower():
                    hits.append(next_path)
                walk(nested, next_path)
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                walk(nested, f"{path}[{index}]")
    walk(value, "$")
    return hits


def _safe_raw_result_rows(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for path in paths:
        identities.append(file_identity(path))
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                # Prior validation artifacts are required to contain no such key.
                # Reject before the JSON decoder can materialize its value.
                if seed_inventory.has_forbidden_json_key(line):
                    raise RuntimeError(
                        f"forbidden metric key in canonical validation shard: {path}:{line_number}"
                    )
                row = json.loads(line)
                if _contains_forbidden_metric_key(row):
                    raise RuntimeError(f"forbidden metric field in canonical shard: {path}:{line_number}")
                rows.append(row)
    return rows, identities


def _verify_archive(input_root: Path) -> dict[str, Any]:
    archive = input_root / INPUT_ARCHIVE_NAME
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if not archive.is_file() or not sidecar.is_file():
        raise RuntimeError("input archive or sidecar is missing")
    archive_hash = sha256_file(archive)
    sidecar_hash = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if archive_hash != INPUT_ARCHIVE_SHA256 or sidecar_hash != archive_hash:
        raise RuntimeError("input archive identity mismatch")
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        if any(not member.isfile() for member in members):
            raise RuntimeError("input archive contains a non-regular member")
        names = {member.name for member in members}
        if "artifact_manifest.json" not in names:
            raise RuntimeError("input archive lacks artifact_manifest.json")
        manifest_bytes = tar.extractfile("artifact_manifest.json").read()
        manifest = json.loads(manifest_bytes)
        expected_names = {row["relative_path"] for row in manifest["files"]}
        expected_names.add("artifact_manifest.json")
        if names != expected_names:
            raise RuntimeError("input archive member set differs from its manifest")
        bad = []
        for row in manifest["files"]:
            payload = tar.extractfile(row["relative_path"]).read()
            if len(payload) != int(row["size_bytes"]):
                bad.append(row["relative_path"])
            elif hashlib.sha256(payload).hexdigest() != row["sha256"]:
                bad.append(row["relative_path"])
        if bad:
            raise RuntimeError(f"input archive manifest mismatch: {bad[:5]}")
    live_manifest_path = input_root / "artifact_manifest.json"
    if not live_manifest_path.is_file() or live_manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError("input live artifact manifest differs from verified archive")
    live_bad = []
    for row in manifest["files"]:
        live = input_root / row["relative_path"]
        if not live.is_file():
            live_bad.append({"relative_path": row["relative_path"], "reason": "missing"})
            continue
        identity = file_identity(live)
        if identity["size_bytes"] != int(row["size_bytes"]) or identity["sha256"] != row["sha256"]:
            live_bad.append({"relative_path": row["relative_path"], "reason": "identity_mismatch"})
    if live_bad:
        raise RuntimeError(f"input live files differ from verified archive: {live_bad[:5]}")
    return {
        "archive": file_identity(archive),
        "sidecar": file_identity(sidecar),
        "member_count": len(members),
        "manifest_file_count": len(manifest["files"]),
        "live_files_matching_archive_manifest": len(manifest["files"]),
        "live_manifest_bytes_match_archive": True,
        "passed": True,
    }


def _verify_snapshot_implementation(input_root: Path) -> dict[str, Any]:
    audit_path = input_root / "source_snapshot_audit.json"
    audit, safe_input_audit = _load_input_json_without_test_values(audit_path)
    snapshot_by_relative = {
        row["relative_path"]: row for row in audit["files"]
    }
    rows = []
    for relative in FINGERPRINT_SOURCE_FILES:
        source = REPO_ROOT / relative
        snapshot = input_root / "source_snapshot" / relative
        row = snapshot_by_relative.get(relative)
        if row is None or not source.is_file() or not snapshot.is_file():
            raise RuntimeError(f"fingerprint implementation missing from input snapshot: {relative}")
        current_hash = sha256_file(source)
        snapshot_hash = sha256_file(snapshot)
        frozen_hash = str(row["frozen_sha256"])
        if not (current_hash == snapshot_hash == frozen_hash):
            raise RuntimeError(f"fingerprint implementation SHA mismatch: {relative}")
        rows.append({
            "relative_path": relative,
            "current_sha256": current_hash,
            "snapshot_sha256": snapshot_hash,
            "frozen_sha256": frozen_hash,
            "three_way_match": True,
        })
    return {
        "input_snapshot_audit": file_identity(audit_path),
        "safe_input_firewall": safe_input_audit,
        "files": rows,
        "passed": True,
    }


def _rows_by_joint(rows: Sequence[Mapping[str, Any]], joint: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row.get("joint_config_fingerprint")) == joint]


def _assert_overlap_equal(left: Mapping[str, Any], right: Mapping[str, Any], *, context: str) -> None:
    ignored = {"assigned_gpu", "task_stage", "selected_by", "architecture_source", "generation_index", "reservoir_order"}
    shared = (set(left) & set(right)) - ignored
    mismatches = [key for key in shared if left[key] != right[key]]
    if mismatches:
        raise RuntimeError(f"candidate field conflict ({context}): {mismatches}")


def _recompute_candidate_fingerprints(config: Mapping[str, Any]) -> dict[str, Any]:
    architecture_fp = architecture_fingerprint(config)
    hp_fp = v1.hp_fingerprint(config["raw_hp_vector"])
    joint_fp = v2.joint_config_fingerprint(architecture_fp, hp_fp)
    realized = realize_global4_hp(config)
    decoded_fields = (
        "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
        "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer",
    )
    decoded_mismatch = []
    for key in decoded_fields:
        expected = realized[key]
        observed = config.get(key)
        if isinstance(expected, float):
            if observed is None or abs(float(observed) - expected) > 1e-12:
                decoded_mismatch.append(key)
        elif observed != expected:
            decoded_mismatch.append(key)
    return {
        "architecture_fingerprint": architecture_fp,
        "hp_fingerprint": hp_fp,
        "joint_config_fingerprint": joint_fp,
        "realized_hp": realized,
        "decoded_field_mismatches": decoded_mismatch,
    }


def verify_input_evidence() -> dict[str, Any]:
    input_root = REPO_ROOT / INPUT_RELATIVE
    archive_audit = _verify_archive(input_root)
    snapshot_audit = _verify_snapshot_implementation(input_root)
    safe_summary_inputs = []
    final, safe_row = _load_input_json_without_test_values(input_root / "final_classification.json")
    safe_summary_inputs.append(safe_row)
    if final.get("classification") != EXPECTED_CLASSIFICATION:
        raise RuntimeError("INPUT_EVIDENCE_MISMATCH: classification")
    if not final.get("first_stage_classification_unchanged"):
        raise RuntimeError("INPUT_EVIDENCE_MISMATCH: first-stage classification drift")

    union_path = input_root / "stabilization/stabilization_union_configs.json"
    frozen_path = input_root / "frozen_candidate_configs.json"
    top_stats_path = input_root / "stabilization/strategy_top8_30seed_statistics.json"
    top_freeze_path = input_root / "stabilization/frozen_top8_selections.json"
    selections_path = input_root / "frozen_strategy_selections.json"
    union_rows, safe_row = _load_input_json_without_test_values(union_path); safe_summary_inputs.append(safe_row)
    frozen_rows, safe_row = _load_input_json_without_test_values(frozen_path); safe_summary_inputs.append(safe_row)
    top_stats, safe_row = _load_input_json_without_test_values(top_stats_path); safe_summary_inputs.append(safe_row)
    top_freeze, safe_row = _load_input_json_without_test_values(top_freeze_path); safe_summary_inputs.append(safe_row)
    selections, safe_row = _load_input_json_without_test_values(selections_path); safe_summary_inputs.append(safe_row)

    initial_paths = [
        input_root / "phase_b/initial/gpu0/seed_results.jsonl",
        input_root / "phase_b/initial/gpu1/seed_results.jsonl",
    ]
    stabilization_paths = [
        input_root / "phase_c/stabilization/gpu0/seed_results.jsonl",
        input_root / "phase_c/stabilization/gpu1/seed_results.jsonl",
    ]
    initial_rows, initial_identities = _safe_raw_result_rows(initial_paths)
    stabilization_rows, stabilization_identities = _safe_raw_result_rows(stabilization_paths)
    if len(initial_rows) != 1008 or len(stabilization_rows) != 378:
        raise RuntimeError("INPUT_EVIDENCE_MISMATCH: raw validation task counts")

    def result_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
        return (int(row["pool_id"]), str(row["joint_config_fingerprint"]), int(row["seed"]))
    all_raw = initial_rows + stabilization_rows
    raw_keys = [result_key(row) for row in all_raw]
    if len(raw_keys) != len(set(raw_keys)):
        raise RuntimeError("INPUT_EVIDENCE_MISMATCH: duplicate canonical raw result")
    if any(row.get("terminal_status") != "success" or not row.get("valid") for row in all_raw):
        raise RuntimeError("INPUT_EVIDENCE_MISMATCH: non-success canonical raw result")

    merged_audits = []
    for stage, raw_stage, merged_path in (
        ("initial", initial_rows, input_root / "phase_b/initial/seed_results_merged.jsonl"),
        ("stabilization", stabilization_rows, input_root / "phase_c/stabilization/seed_results_merged.jsonl"),
    ):
        merged_rows, ids = _safe_raw_result_rows([merged_path])
        raw_map = {result_key(row): stable_hash(row) for row in raw_stage}
        merged_map = {result_key(row): stable_hash(row) for row in merged_rows}
        if len(merged_map) != len(merged_rows) or raw_map != merged_map:
            raise RuntimeError(f"INPUT_EVIDENCE_MISMATCH: merged exact union ({stage})")
        merged_audits.append({
            "stage": stage, "row_count": len(merged_rows), "exact_union": True,
            "file": ids[0], "not_counted_twice": True,
        })

    initial_label = {
        (int(row["pool_id"]), str(row["joint_config_fingerprint"]), int(row["seed"])): float(row["val_acc"])
        for row in initial_rows
    }
    stabilization_label = {
        (int(row["pool_id"]), str(row["joint_config_fingerprint"]), int(row["seed"])): float(row["val_acc"])
        for row in stabilization_rows
    }

    recomputed_top: dict[str, list[dict[str, Any]]] = {}
    for strategy in ("HIGH_REGIME_PRIOR", "BALANCED_RANDOM_HP"):
        strategy_rows = [row for row in selections if row.get("strategy") == strategy]
        unique = {}
        for row in strategy_rows:
            unique[(int(row["pool_id"]), str(row["joint_config_fingerprint"]))] = row
        if len(unique) != 192:
            raise RuntimeError(f"INPUT_EVIDENCE_MISMATCH: strategy config count {strategy}")
        ranked = []
        for (pool_id, joint), row in unique.items():
            values = [initial_label[(pool_id, joint, seed)] for seed in (0, 1, 2)]
            ranked.append({
                "strategy": strategy, "pool_id": pool_id,
                "joint_config_fingerprint": joint,
                "initial_three_seed_mean_validation": float(np.mean(values)),
            })
        ranked.sort(key=lambda row: (-row["initial_three_seed_mean_validation"], row["joint_config_fingerprint"]))
        recomputed_top[strategy] = ranked[:8]
        frozen_strategy = [row for row in top_freeze if row.get("strategy") == strategy]
        frozen_order = [str(row["joint_config_fingerprint"]) for row in sorted(frozen_strategy, key=lambda row: int(row["initial_rank"]))]
        if frozen_order != [row["joint_config_fingerprint"] for row in ranked[:8]]:
            raise RuntimeError(f"INPUT_EVIDENCE_MISMATCH: top8 {strategy}")

    candidates = {}
    for role, strategy, expected_joint, final_name in (
        ("P", "HIGH_REGIME_PRIOR", EXPECTED_P, "F_final_candidate.json"),
        ("Q", "BALANCED_RANDOM_HP", EXPECTED_Q, "R_final_candidate.json"),
    ):
        final_path = input_root / "stabilization" / final_name
        final_row, safe_row = _load_input_json_without_test_values(final_path)
        safe_summary_inputs.append(safe_row)
        if final_row.get("strategy") != strategy or final_row.get("joint_config_fingerprint") != expected_joint:
            raise RuntimeError(f"INPUT_EVIDENCE_MISMATCH: final role {role}")
        union_match = _rows_by_joint(union_rows, expected_joint)
        frozen_match = _rows_by_joint(frozen_rows, expected_joint)
        stats_match = [row for row in top_stats if row.get("strategy") == strategy and row.get("joint_config_fingerprint") == expected_joint]
        if len(union_match) != 1 or len(frozen_match) != 1 or len(stats_match) != 1:
            raise RuntimeError(f"INPUT_EVIDENCE_MISMATCH: non-unique canonical join {role}")
        _assert_overlap_equal(union_match[0], frozen_match[0], context=f"{role} union/frozen")
        _assert_overlap_equal(final_row, stats_match[0], context=f"{role} final/stats")
        _assert_overlap_equal(final_row, union_match[0], context=f"{role} final/config")
        recomputed = _recompute_candidate_fingerprints(union_match[0])
        if recomputed["decoded_field_mismatches"]:
            raise RuntimeError(f"INPUT_EVIDENCE_MISMATCH: decoded HP {role}")
        if (
            recomputed["architecture_fingerprint"] != union_match[0]["architecture_fingerprint"]
            or recomputed["hp_fingerprint"] != union_match[0]["hp_fingerprint"]
            or recomputed["joint_config_fingerprint"] != expected_joint
        ):
            raise RuntimeError(f"INPUT_EVIDENCE_MISMATCH: fingerprint recompute {role}")
        pool_id = int(union_match[0]["pool_id"])
        all30 = [
            initial_label[(pool_id, expected_joint, seed)] if seed <= 2
            else stabilization_label[(pool_id, expected_joint, seed)]
            for seed in range(30)
        ]
        if all30 != [float(value) for value in final_row["all30_seed_values"]]:
            raise RuntimeError(f"INPUT_EVIDENCE_MISMATCH: all30 raw values {role}")
        if abs(float(np.mean(all30)) - float(final_row["all30_mean_validation"])) > 1e-15:
            raise RuntimeError(f"INPUT_EVIDENCE_MISMATCH: all30 mean {role}")
        ranked_all30 = []
        for top in recomputed_top[strategy]:
            top_pool = int(top["pool_id"]); top_joint = str(top["joint_config_fingerprint"])
            values = [
                initial_label[(top_pool, top_joint, seed)] if seed <= 2
                else stabilization_label[(top_pool, top_joint, seed)]
                for seed in range(30)
            ]
            ranked_all30.append((float(np.mean(values)), top_joint))
        ranked_all30.sort(key=lambda pair: (-pair[0], pair[1]))
        if ranked_all30[0][1] != expected_joint:
            raise RuntimeError(f"INPUT_EVIDENCE_MISMATCH: all30 winner {role}")

        canonical = dict(union_match[0])
        search_only = {
            key: canonical.pop(key) for key in list(canonical)
            if key in {"assigned_gpu", "task_stage", "selected_by"}
        }
        candidates[role] = {
            "role": role,
            "strategy": strategy,
            "method_identity": METHOD_NAME if role == "P" else "preregistered frozen comparison winner",
            "joint_config_fingerprint": expected_joint,
            "candidate_config": canonical,
            "search_provenance_only": search_only,
            "search_validation_statistics": dict(final_row),
            "recomputed_fingerprints": recomputed,
            "source_files": {
                "final_candidate": file_identity(final_path),
                "stabilization_union_configs": file_identity(union_path),
                "frozen_candidate_configs": file_identity(frozen_path),
                "strategy_top8_statistics": file_identity(top_stats_path),
                "frozen_top8": file_identity(top_freeze_path),
                "frozen_strategy_selections": file_identity(selections_path),
            },
        }

    if candidates["P"]["candidate_config"]["hp_regime"] != "hp_high" or candidates["Q"]["candidate_config"]["hp_regime"] != "hp_high":
        raise RuntimeError("INPUT_EVIDENCE_MISMATCH: both winners must be hp_high")
    q_in_f_top = any(row["joint_config_fingerprint"] == EXPECTED_Q for row in recomputed_top["HIGH_REGIME_PRIOR"])
    if not q_in_f_top:
        raise RuntimeError("INPUT_EVIDENCE_MISMATCH: Q absent from F top8")

    return {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "classification": EXPECTED_CLASSIFICATION,
        "method_name": METHOD_NAME,
        "archive_audit": archive_audit,
        "fingerprint_snapshot_implementation_audit": snapshot_audit,
        "candidates": candidates,
        "raw_validation_files": initial_identities + stabilization_identities,
        "merged_exact_union_audit": merged_audits,
        "safe_summary_input_firewall": {
            "files": safe_summary_inputs,
            "file_count": len(safe_summary_inputs),
            "test_values_deserialized": False,
        },
        "initial_record_count": len(initial_rows),
        "stabilization_record_count": len(stabilization_rows),
        "P_and_Q_identical": EXPECTED_P == EXPECTED_Q,
        "P_and_Q_both_high_regime": True,
        "Q_also_in_F_frozen_top8": True,
        "future_evidence_scope": (
            "prospective-after-freeze benchmark reveal of two frozen winner configurations; "
            "project-level benchmark data are not untouched and this does not retest the search method"
        ),
        "passed": True,
    }


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    for component in (
        str(value.dtype).encode("ascii"),
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"),
        value.numpy().tobytes(order="C"),
    ):
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return digest.hexdigest()


def validate_sealed_view(view: Mapping[str, Any]) -> dict[str, Any]:
    if set(view) != SEALED_KEYS:
        raise RuntimeError(f"sealed view key mismatch: {sorted(set(view) ^ SEALED_KEYS)}")
    tensors = {key: view[key] for key in ("x", "edge_index", "train_mask", "val_mask", "y_sealed")}
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("sealed view tensor field has non-tensor value")
    x = tensors["x"]; edge_index = tensors["edge_index"]
    train_mask = tensors["train_mask"]; val_mask = tensors["val_mask"]
    y_sealed = tensors["y_sealed"]
    if x.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("sealed graph tensor shape mismatch")
    if train_mask.dtype != torch.bool or val_mask.dtype != torch.bool:
        raise TypeError("sealed masks must be boolean")
    if len(train_mask) != len(x) or len(val_mask) != len(x) or len(y_sealed) != len(x):
        raise ValueError("sealed node tensor length mismatch")
    if bool((train_mask & val_mask).any().item()):
        raise ValueError("train and validation masks overlap")
    white = train_mask | val_mask
    if int(train_mask.sum().item()) <= 0 or int(val_mask.sum().item()) <= 0:
        raise ValueError("sealed train or validation partition is empty")
    if bool((y_sealed[~white] >= 0).any().item()):
        raise ValueError("non-whitelisted labels were not poisoned")
    labels = y_sealed[white]
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= 7:
        raise ValueError("whitelisted Cora labels violate protocol constant")
    if int(view["num_features"]) != int(x.shape[1]) or int(view["num_classes"]) != 7:
        raise ValueError("sealed feature/class identity mismatch")
    if (
        int(x.shape[0]) != EXPECTED_CORA_NUM_NODES
        or int(x.shape[1]) != EXPECTED_CORA_NUM_FEATURES
        or int(edge_index.shape[1]) != EXPECTED_CORA_EDGE_ENTRIES
        or int(train_mask.sum().item()) != EXPECTED_CORA_TRAIN_COUNT
        or int(val_mask.sum().item()) != EXPECTED_CORA_VALIDATION_COUNT
    ):
        raise ValueError("sealed Cora/public-split cardinality mismatch")
    hashes = {key: _tensor_hash(value) for key, value in tensors.items()}
    return {
        "schema_keys": sorted(SEALED_KEYS),
        "schema_sha256": stable_hash(sorted(SEALED_KEYS)),
        "tensor_sha256": hashes,
        "combined_tensor_root": stable_hash(hashes),
        "num_nodes": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "num_classes": 7,
        "train_count": int(train_mask.sum().item()),
        "validation_count": int(val_mask.sum().item()),
        "outside_nonnegative_label_count": int((y_sealed[~white] >= 0).sum().item()),
        "passed": True,
    }


def seal_train_validation_view(cora_root: Path, output_path: Path, identity_path: Path) -> None:
    # This is the only preparation worker allowed to instantiate the standard dataset.
    # It deliberately never resolves any held-out attribute.
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    data = Planetoid(root=str(cora_root), name="Cora", transform=NormalizeFeatures())[0]
    x = data.x.detach().cpu().contiguous()
    edge_index = data.edge_index.detach().cpu().contiguous()
    train_mask = data.train_mask.detach().cpu().to(dtype=torch.bool).contiguous()
    val_mask = data.val_mask.detach().cpu().to(dtype=torch.bool).contiguous()
    original_y = data.y
    white = train_mask | val_mask
    y_sealed = torch.full_like(original_y, -1, device="cpu")
    y_sealed[white] = original_y[white].detach().cpu()
    view = {
        "x": x,
        "edge_index": edge_index,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "y_sealed": y_sealed.contiguous(),
        "num_features": int(x.shape[1]),
        "num_classes": 7,
    }
    identity = validate_sealed_view(view)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(view, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    identity.update({
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "sealed_file": file_identity(output_path),
        "dataset_content_identity_reused": DATASET_CONTENT_ID,
        "public_split_identity_reused": PUBLIC_SPLIT_ID,
        "normalization": "NormalizeFeatures",
        "dataset_object_or_complete_label_hash_created": False,
        "full_label_clone_created": False,
        "sealer_access_allowlist": ["x", "edge_index", "train_mask", "val_mask", "y[train_mask|val_mask]"],
    })
    dump_json(identity_path, identity)


def load_sealed_view_safely(path: Path) -> Mapping[str, Any]:
    view = torch.load(path, weights_only=True, map_location="cpu")
    validate_sealed_view(view)
    return view


def verify_sealed_identity(
    view: Mapping[str, Any], path: Path, identity: Mapping[str, Any],
) -> dict[str, Any]:
    observed = validate_sealed_view(view)
    for key in (
        "schema_keys", "schema_sha256", "tensor_sha256", "combined_tensor_root",
        "num_nodes", "num_features", "num_classes", "train_count",
        "validation_count", "outside_nonnegative_label_count",
    ):
        if identity.get(key) != observed.get(key):
            raise RuntimeError(f"sealed identity mismatch: {key}")
    current_file = file_identity(path)
    if identity.get("sealed_file") != current_file:
        raise RuntimeError("sealed file identity mismatch")
    if (
        identity.get("dataset_content_identity_reused") != DATASET_CONTENT_ID
        or identity.get("public_split_identity_reused") != PUBLIC_SPLIT_ID
        or identity.get("num_classes") != 7
    ):
        raise RuntimeError("sealed declared dataset/split identity mismatch")
    return {**observed, "sealed_file": current_file, "passed": True}


def _local_source_closure(entries: Sequence[Path]) -> list[Path]:
    return v2.discover_local_source_closure(entries)


def _head_blob_identity(relative: str) -> tuple[bool, str | None]:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative], cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not tracked:
        return False, None
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return False, None
    return True, hashlib.sha256(result.stdout).hexdigest()


def source_dependency_manifest() -> dict[str, Any]:
    entries = [REPO_ROOT / relative for relative in SOURCE_ENTRY_FILES]
    missing = [str(path) for path in entries if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing source entry files: {missing}")
    rows = []
    for path in _local_source_closure(entries):
        relative = str(path.relative_to(REPO_ROOT))
        tracked, head_hash = _head_blob_identity(relative)
        current_hash = sha256_file(path)
        recoverable = bool(tracked and head_hash == current_hash)
        rows.append({
            **file_identity(path), "relative_path": relative,
            "tracked_by_git": tracked, "head_blob_sha256": head_hash,
            "recoverable_from_clean_head": recoverable,
            "requires_delivery_snapshot": not recoverable,
        })
    return {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "entry_files": list(SOURCE_ENTRY_FILES),
        "dependency_count": len(rows),
        "snapshot_required_count": sum(row["requires_delivery_snapshot"] for row in rows),
        "files": rows,
    }


def _source_identity(output: Path, dependency: Mapping[str, Any]) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=REPO_ROOT, check=True,
        stdout=subprocess.PIPE,
    ).stdout
    (output / "tracked_diff.binary.patch").write_bytes(diff)
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "pip_freeze": _run_text([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi": _run_text(["nvidia-smi"]),
    }
    (output / "environment_lock.txt").write_text(
        "\n\n".join(f"[{key}]\n{value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )
    return {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "branch": _run_text(["git", "branch", "--show-current"]),
        "head": _run_text(["git", "rev-parse", "HEAD"]),
        "status_short": _run_text(["git", "status", "--short"]).splitlines(),
        "untracked_files": _run_text(["git", "ls-files", "--others", "--exclude-standard"]).splitlines(),
        "tracked_binary_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dependency_manifest_sha256": stable_hash(dependency),
        "gpu_uuid_contract": list(GPU_UUIDS),
        "worktree_cleaned": False,
        "commit_created": False,
        "push_performed": False,
    }


def _protected_paths(input_root: Path, dependency: Mapping[str, Any]) -> list[Path]:
    paths = [REPO_ROOT / "bo_phase4.py", REPO_ROOT / "eval_utils.py"]
    artifact = load_json(input_root / "artifact_manifest.json")
    paths.extend(input_root / row["relative_path"] for row in artifact["files"])
    paths.extend(Path(row["path"]) for row in dependency["files"])
    return sorted({path.resolve() for path in paths if path.is_file()})


def _pair_plan(candidates: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows = []
    ordered_for_assignment = sorted(
        FRESH_SEEDS,
        key=lambda seed: stable_hash([MASTER_SEED, "pair_gpu_assignment", seed]),
    )
    gpu_by_seed = {seed: 0 if index < 15 else 1 for index, seed in enumerate(ordered_for_assignment)}
    local_by_gpu = {
        gpu: sorted(
            [seed for seed in FRESH_SEEDS if gpu_by_seed[seed] == gpu],
            key=lambda seed: stable_hash([MASTER_SEED, "local_pair_order", gpu, seed]),
        )
        for gpu in (0, 1)
    }
    tasks = []
    for gpu in (0, 1):
        for local_index, seed in enumerate(local_by_gpu[gpu]):
            pair_id = seed - FRESH_SEEDS[0]
            p_first = (local_index % 2 == 0) if gpu == 0 else (local_index % 2 == 1)
            order = ["P", "Q"] if p_first else ["Q", "P"]
            pair_rows.append({
                "pair_id": pair_id, "seed": seed,
                "assigned_physical_gpu_index": gpu,
                "assigned_gpu_uuid": GPU_UUIDS[gpu],
                "local_pair_index": local_index,
                "pair_execution_order": order,
            })
            for within_pair, role in enumerate(order):
                candidate = candidates[role]
                task_key = f"pair_{pair_id:02d}__{role}__{candidate['joint_config_fingerprint']}__seed_{seed}"
                partner_role = "Q" if role == "P" else "P"
                partner = candidates[partner_role]
                tasks.append({
                    "task_key": task_key,
                    "partner_task_key": f"pair_{pair_id:02d}__{partner_role}__{partner['joint_config_fingerprint']}__seed_{seed}",
                    "pair_id": pair_id, "seed": seed, "role": role,
                    "strategy": candidate["strategy"],
                    "joint_config_fingerprint": candidate["joint_config_fingerprint"],
                    "assigned_physical_gpu_index": gpu,
                    "assigned_gpu_uuid": GPU_UUIDS[gpu],
                    "local_pair_index": local_index,
                    "within_pair_order": within_pair,
                    "ordinal_on_gpu": 2 * local_index + within_pair,
                })
    tasks.sort(key=lambda row: (row["assigned_physical_gpu_index"], row["ordinal_on_gpu"]))
    pair_rows.sort(key=lambda row: row["pair_id"])
    if len(pair_rows) != 30 or len(tasks) != 60:
        raise RuntimeError("pair plan cardinality mismatch")
    for pair in pair_rows:
        pair_tasks = [row for row in tasks if row["pair_id"] == pair["pair_id"]]
        if len(pair_tasks) != 2 or len({row["seed"] for row in pair_tasks}) != 1 or len({row["assigned_gpu_uuid"] for row in pair_tasks}) != 1:
            raise RuntimeError("pair identity mismatch")
    return pair_rows, tasks


def _protocol() -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "method_level_status": "VALIDATION_CONFIRMED_HIGH_REGIME_PRIOR",
        "method_name": METHOD_NAME,
        "candidate_scope": "two frozen winner configurations only; not a new strategy comparison",
        "future_evidence_wording": (
            "prospective-after-freeze benchmark reveal; project-level benchmark data are not untouched"
        ),
        "training": {
            "dataset_identity": DATASET_CONTENT_ID,
            "public_split_identity": PUBLIC_SPLIT_ID,
            "normalization": "NormalizeFeatures",
            "full_batch": True,
            "epochs": 300,
            "patience": 80,
            "loss": "cross_entropy",
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "scheduler_T_max": 300,
            "scheduler_eta_min_multiplier": 0.01,
            "component_variant": "baseline",
            "activation_variant": "relu",
            "gcnii_alpha": 0.1,
            "gcnii_theta": 0.5,
            "best_rule": "strict validation improvement only; ties retain earliest epoch",
            "max_system_retries": 2,
            "protocol_errors_retryable": False,
            "fresh_paired_seeds": list(FRESH_SEEDS),
        },
        "checkpoint": {
            "full_parameters_and_buffers": True,
            "best_state_cloned_to_cpu_at_improvement": True,
            "atomic_file_replace": True,
            "best_state_tensor_hash_separate_from_file_sha256": True,
            "embeds_pretraining_immutable_root_only": True,
            "self_referential_final_manifest_embedded": False,
            "best_last_hash_equality_allowed_only_when_best_epoch_is_last_epoch": True,
        },
        "future_reveal_not_executed_in_this_goal": True,
        "future_primary_statistics": {
            "unit": "30 common paired training seeds",
            "delta": "benchmark(P,seed)-benchmark(Q,seed)",
            "primary_estimate": "mean paired difference",
            "primary_interval": "paired t interval, df=29, 95%",
            "paired_bootstrap_sensitivity_replicates": 50000,
            "report": ["P mean", "P SD", "P 95% CI", "Q mean", "Q SD", "Q 95% CI", "delta median", "W/L/T", "paired dz"],
        },
        "future_candidate_classification_order": [
            {
                "classification": "FROZEN_CANDIDATE_P_ADVANTAGE_ON_CORA_BENCHMARK",
                "condition": "mean_delta >= 0.006 and paired_t_lower > 0",
            },
            {
                "classification": "FROZEN_CANDIDATE_P_NONINFERIOR_ON_CORA_BENCHMARK",
                "condition": "not class 1 or 3 and paired_t_lower > -0.006",
            },
            {
                "classification": "FROZEN_CANDIDATE_Q_ADVANTAGE_ON_CORA_BENCHMARK",
                "condition": "mean_delta <= -0.006 and paired_t_upper < 0",
            },
            {
                "classification": "FROZEN_CANDIDATE_BENCHMARK_INCONCLUSIVE",
                "condition": "remaining complete outcomes",
            },
            {
                "classification": "EXECUTION_OR_PROTOCOL_FAILURE",
                "condition": "any protocol, access, or completeness error",
            },
        ],
        "identical_candidate_rule": (
            "FINAL_CANDIDATES_IDENTICAL: one training per seed; future difference fixed at zero and no superiority test"
        ),
        "future_reveal_cannot_rewrite_method_status": True,
    }


def _manifest_for_files(root: Path, relative_paths: Sequence[str], *, format_name: str) -> dict[str, Any]:
    files = []
    for relative in sorted(set(relative_paths)):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"immutable file missing: {relative}")
        files.append({"relative_path": relative, **file_identity(path)})
    root_hash = stable_hash([
        {"relative_path": row["relative_path"], "size_bytes": row["size_bytes"], "sha256": row["sha256"]}
        for row in files
    ])
    return {
        "format_version": format_name,
        "created_at": utc_now(),
        "files": files,
        "manifest_root_hash": root_hash,
    }


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    bad = []
    compact = []
    for row in manifest["files"]:
        path = root / row["relative_path"]
        if not path.is_file():
            bad.append({"relative_path": row["relative_path"], "reason": "missing"})
            continue
        current = file_identity(path)
        if current["size_bytes"] != row["size_bytes"] or current["sha256"] != row["sha256"]:
            bad.append({"relative_path": row["relative_path"], "reason": "identity_mismatch"})
        compact.append({"relative_path": row["relative_path"], "size_bytes": current["size_bytes"], "sha256": current["sha256"]})
    recomputed = stable_hash(compact)
    return {
        "passed": not bad and recomputed == manifest["manifest_root_hash"],
        "bad": bad,
        "expected_root": manifest["manifest_root_hash"],
        "recomputed_root": recomputed,
        "file_count": len(manifest["files"]),
    }


def _existing_output_resume_gate(output: Path) -> dict[str, Any]:
    if not output.exists():
        return {"exists": False, "resume_allowed": False}
    manifest_path = output / "pretraining_immutable_manifest.json"
    prereg_path = output / "preregistration.json"
    if not manifest_path.is_file() or not prereg_path.is_file():
        raise RuntimeError("OUTPUT_IDENTITY_MISMATCH: existing output lacks immutable identity")
    verification = verify_manifest(output, manifest_path)
    prereg = load_json(prereg_path)
    if not verification["passed"]:
        raise RuntimeError("OUTPUT_IDENTITY_MISMATCH: existing immutable manifest drift")
    expected = {
        "format_version": FORMAT_VERSION,
        "input_archive_sha256": INPUT_ARCHIVE_SHA256,
        "fresh_seeds": list(FRESH_SEEDS),
        "expected_P": EXPECTED_P,
        "expected_Q": EXPECTED_Q,
    }
    observed = {key: prereg.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError("OUTPUT_IDENTITY_MISMATCH: existing preregistration differs")
    return {"exists": True, "resume_allowed": True, "manifest": verification}


def run_preflight_scan(target: Path, inventory_output: Path) -> None:
    resume = _existing_output_resume_gate(target)
    wrapper = seed_inventory.scan_repository_training_seeds(
        repo_root=REPO_ROOT,
        excluded_target=target.resolve(),
        proposed_seeds=FRESH_SEEDS,
    )
    if not wrapper.get("passed"):
        raise RuntimeError(f"historical seed inventory failed: {wrapper.get('failure_reason')}")
    seed_inventory.verify_inventory_wrapper(wrapper, excluded_target=target)
    if wrapper.get("fresh_seed_collisions"):
        raise RuntimeError("FRESH_SEED_COLLISION")
    preflight = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "inventory_wrapper": wrapper,
        "target_resume_gate": resume,
        "scan_completed_before_output_creation": not target.exists(),
    }
    preflight["preflight_root_sha256"] = stable_hash(preflight)
    dump_json(inventory_output, preflight)
    print(json.dumps({
        "inventory_output": str(inventory_output),
        "inventory_root": wrapper["inventory_root"],
        "used_training_seed_count": len(wrapper["used_training_seeds"]),
        "fresh_seed_collisions": wrapper["fresh_seed_collisions"],
        "passed": True,
    }, indent=2, sort_keys=True))


def prepare(output: Path, inventory_path: Path, cora_root: Path) -> None:
    preflight = load_json(inventory_path)
    recorded_preflight_root = preflight.get("preflight_root_sha256")
    unhashed_preflight = dict(preflight)
    unhashed_preflight.pop("preflight_root_sha256", None)
    if recorded_preflight_root != stable_hash(unhashed_preflight):
        raise RuntimeError("historical inventory preflight root mismatch")
    inventory = preflight.get("inventory_wrapper")
    if not isinstance(inventory, Mapping):
        raise RuntimeError("historical inventory wrapper missing")
    seed_inventory.verify_inventory_wrapper(inventory, excluded_target=output)
    if not inventory.get("passed") or inventory.get("fresh_seed_collisions"):
        raise RuntimeError("FRESH_SEED_COLLISION or incomplete historical inventory")
    resume = _existing_output_resume_gate(output)
    if resume["exists"]:
        print(json.dumps({"resume_allowed": True, "output": str(output)}, indent=2))
        return

    evidence = verify_input_evidence()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(inventory_path, output / "historical_training_seed_inventory.json")
    dump_json(output / "input_evidence_audit.json", evidence)
    dump_json(output / "frozen_candidate_P.json", evidence["candidates"]["P"])
    dump_json(output / "frozen_candidate_Q.json", evidence["candidates"]["Q"])

    sealed_path = output / "sealed/train_val_sealed_view.pt"
    sealed_identity_path = output / "sealed/train_val_sealed_view_identity.json"
    seal_command = [
        sys.executable, str(Path(__file__).resolve()), "seal-view",
        "--cora-root", str(cora_root), "--output", str(sealed_path),
        "--identity-output", str(sealed_identity_path),
    ]
    subprocess.run(seal_command, cwd=REPO_ROOT, check=True)
    sealed_view = load_sealed_view_safely(sealed_path)
    verify_sealed_identity(sealed_view, sealed_path, load_json(sealed_identity_path))

    dependency = source_dependency_manifest()
    dump_json(output / "source_dependency_manifest.json", dependency)
    dump_json(output / "source_identity.json", _source_identity(output, dependency))
    input_root = REPO_ROOT / INPUT_RELATIVE
    protected_rows = [file_identity(path) for path in _protected_paths(input_root, dependency)]
    dump_json(output / "protected_hashes_before.json", {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "objects": protected_rows,
        "object_count": len(protected_rows),
    })
    protocol = _protocol()
    dump_json(output / "protocol.json", protocol)
    pair_rows, tasks = _pair_plan(evidence["candidates"])
    dump_json(output / "paired_seed_registry.json", {
        "format_version": FORMAT_VERSION,
        "fresh_seeds": list(FRESH_SEEDS),
        "historical_inventory_root": inventory["inventory_root"],
        "historical_collision_count": 0,
        "pairs": pair_rows,
    })
    dump_csv(output / "paired_seed_registry.csv", pair_rows)
    dump_json(output / "training_tasks.json", tasks)
    dump_csv(output / "training_tasks.csv", tasks)
    preregistration = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "input_archive_sha256": INPUT_ARCHIVE_SHA256,
        "input_classification": EXPECTED_CLASSIFICATION,
        "fresh_seeds": list(FRESH_SEEDS),
        "expected_P": EXPECTED_P,
        "expected_Q": EXPECTED_Q,
        "method_name": METHOD_NAME,
        "task_count": len(tasks),
        "pair_count": len(pair_rows),
        "P_Q_same_seed_and_same_physical_gpu": True,
        "gpu_pair_counts": {str(gpu): sum(row["assigned_physical_gpu_index"] == gpu for row in pair_rows) for gpu in (0, 1)},
        "global_order_counts": {
            "P_then_Q": sum(row["pair_execution_order"] == ["P", "Q"] for row in pair_rows),
            "Q_then_P": sum(row["pair_execution_order"] == ["Q", "P"] for row in pair_rows),
        },
        "per_gpu_order_balance": "15 pairs/GPU makes exact per-GPU 1:1 impossible; frozen complementary 8:7 and 7:8, globally 15:15",
        "sealed_view_identity_sha256": file_identity(sealed_identity_path)["sha256"],
        "protocol_sha256": file_identity(output / "protocol.json")["sha256"],
        "source_dependency_manifest_sha256": file_identity(output / "source_dependency_manifest.json")["sha256"],
        "real_benchmark_access_count": 0,
        "future_reveal_claim_directory_must_be_empty": True,
    }
    dump_json(output / "preregistration.json", preregistration)
    claim_dir = output / "reveal_claim"
    claim_dir.mkdir()

    immutable_relatives = [
        "historical_training_seed_inventory.json",
        "input_evidence_audit.json",
        "frozen_candidate_P.json", "frozen_candidate_Q.json",
        "sealed/train_val_sealed_view.pt",
        "sealed/train_val_sealed_view_identity.json",
        "source_dependency_manifest.json", "source_identity.json",
        "tracked_diff.binary.patch", "environment_lock.txt",
        "protected_hashes_before.json", "protocol.json",
        "paired_seed_registry.json", "paired_seed_registry.csv",
        "training_tasks.json", "training_tasks.csv", "preregistration.json",
    ]
    manifest = _manifest_for_files(output, immutable_relatives, format_name=FORMAT_VERSION)
    dump_json(output / "pretraining_immutable_manifest.json", manifest)
    audit = verify_manifest(output, output / "pretraining_immutable_manifest.json")
    if not audit["passed"]:
        raise RuntimeError("pretraining immutable manifest self-audit failed")
    dump_json(output / "pretraining_firewall_audit.json", {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "input_evidence_passed": evidence["passed"],
        "historical_seed_inventory_passed": inventory["passed"],
        "fresh_seed_collisions": inventory["fresh_seed_collisions"],
        "sealed_view_passed": load_json(sealed_identity_path)["passed"],
        "sealed_exact_allowlist": sorted(SEALED_KEYS),
        "task_count": len(tasks),
        "pair_count": len(pair_rows),
        "P_Q_same_seed_same_gpu": True,
        "physical_gpu_uuid_contract": list(GPU_UUIDS),
        "real_benchmark_access_count": 0,
        "reveal_claim_directory_empty": not any(claim_dir.iterdir()),
        "pretraining_immutable_root": manifest["manifest_root_hash"],
        "passed": True,
    })
    print(json.dumps({
        "output": str(output),
        "P": EXPECTED_P,
        "Q": EXPECTED_Q,
        "fresh_seeds": [FRESH_SEEDS[0], FRESH_SEEDS[-1]],
        "task_count": len(tasks),
        "pretraining_immutable_root": manifest["manifest_root_hash"],
        "passed": True,
    }, indent=2, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("preflight-scan")
    scan.add_argument("--target", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE)
    scan.add_argument("--inventory-output", type=Path, required=True)
    seal = sub.add_parser("seal-view")
    seal.add_argument("--cora-root", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--identity-output", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE)
    prep.add_argument("--inventory", type=Path, required=True)
    prep.add_argument("--cora-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "preflight-scan":
        run_preflight_scan(args.target.resolve(), args.inventory_output.resolve())
    elif args.command == "seal-view":
        seal_train_validation_view(args.cora_root.resolve(), args.output.resolve(), args.identity_output.resolve())
    elif args.command == "prepare":
        prepare(args.output.resolve(), args.inventory.resolve(), args.cora_root.resolve())
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
