#!/usr/bin/env python3
"""First-formal validation Top-10, 30-replicate independent evaluation pipeline.

The preparation path is deliberately read-only with respect to all historical
search artifacts.  GPU workers train persisted discrete architectures directly;
they never load the VAE checkpoint or decode a latent candidate.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_ROOT = REPOSITORY_ROOT / (
    "server_validation/formal_all4_validation_analysis_20260804_024807_EDT"
)
DEFAULT_SEARCH_SUMMARY = (
    REPOSITORY_ROOT / "reports/project_state_20260806_074942/06_search_results_summary.csv"
)
FROZEN_SOURCE_ID = "8831cd0a533721aa8733ff436058e8da3809861edfb98686edd8ce7fcf902388"
DEFAULT_FROZEN_SNAPSHOT = REPOSITORY_ROOT / (
    "server_validation/launch_gate_20260731/freeze/"
    f"frozen_source_snapshot_{FROZEN_SOURCE_ID}.tar.gz"
)
EXCLUDED_DETERMINISTIC_ROOT = REPOSITORY_ROOT / (
    "server_validation/"
    "deterministic_three_strategy_dualgpu_20260804_20260804T1326Z"
)

DATASETS = ("citeseer", "pubmed", "dblp", "flickr")
DATASET_DISPLAY = {
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "dblp": "DBLP",
    "flickr": "Flickr",
}
METHODS = ("S0", "G100", "G150")
SEARCH_SEEDS = (5, 6, 7, 8, 9)
METHOD_LABELS = {
    "S0": "global_schur",
    "G100": "gmm_exp100",
    "G150": "gmm_exp150",
}
EXPECTED_HISTORY_COUNT = 60
EXPECTED_SOURCE_SLOTS = 600
REPLICATE_IDS = tuple(range(30))
EVAL_EPOCHS = 150
PATIENCE = 40
FINAL_BASE_SEED = 0
FINAL_SEED_SCHEME = "final_base_candidate_fingerprint_replicate_v1"
ANALYSIS_LABEL = "final-validation-selected Top-10 x 30 independent evaluation"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STUDENT_T_975_DF29 = 2.045229642132703

ORIGIN_FIELDS = (
    "dataset",
    "method",
    "search_seed",
    "task_id",
    "source_history_path",
    "source_history_sha256",
    "history_metadata_path",
    "source_id",
    "candidate_fingerprint",
    "validation_top10_rank",
    "search_validation_accuracy",
    "search_stage",
    "within_stage_full_position",
    "stage_position_reconstructed",
    "global_full_evaluation_index",
    "global_full_evaluation_position",
    "low_fidelity_position",
    "promotion_position",
    "low_fidelity_record_id",
    "architecture",
    "hyperparameters",
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
    "gcnii_alpha",
    "gcnii_theta",
    "hp_mode",
    "canonical_z_search",
    "dataset_context",
    "candidate_config_fingerprint",
)

LONG_FIELDS = (
    "dataset",
    "candidate_fingerprint",
    "replicate_id",
    "effective_training_seed",
    "test_accuracy",
    "validation_accuracy",
    "epochs_trained",
    "best_epoch",
    "status",
    "wall_time",
    "gpu_id",
    "cache_key",
    "final_evaluation_protocol_fingerprint",
    "dataset_content_fingerprint",
    "split_fingerprint",
    "completed_at",
)

DETAILED_WINNER_FIELDS = (
    "dataset",
    "method",
    "candidate_fingerprint",
    "architecture",
    "hyperparameters",
    "lr",
    "dropout",
    "hidden_dim",
    "l2",
    "final_validation_mean",
    "final_validation_sample_std",
    "test_mean",
    "test_sample_std",
    "test_mean_ci95_low",
    "test_mean_ci95_high",
    "search_seed",
    "search_stage",
    "global_evaluation_order",
    "stage_local_order",
    "validation_top10_rank",
    *(f"test_accuracy_{replicate_id}" for replicate_id in REPLICATE_IDS),
)


class PipelineAuditError(RuntimeError):
    """Raised when a frozen-input or completion invariant is violated."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def json_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_fingerprint(z_search: Any) -> str:
    """Exact first-formal candidate_fingerprint implementation."""

    array = np.ascontiguousarray(
        np.asarray(z_search, dtype=np.float32).reshape(1, -1)
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def csv_text(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    delimiter: str = ",",
) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="ignore",
        delimiter=delimiter,
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return buffer.getvalue()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return value


def atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    atomic_write_text(path, csv_text(rows, fields))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineAuditError(message)


def _finite_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineAuditError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise PipelineAuditError(f"{context} is not finite: {value!r}")
    return result


def _parse_bool(value: Any, context: str) -> bool:
    if value is True or str(value).strip().lower() == "true":
        return True
    if value is False or str(value).strip().lower() == "false":
        return False
    raise PipelineAuditError(f"{context} is not boolean: {value!r}")


def _path_is_excluded(path: Path) -> bool:
    try:
        path.resolve().relative_to(EXCLUDED_DETERMINISTIC_ROOT.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_snapshot(snapshot: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(snapshot, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            try:
                member_path.relative_to(root)
            except ValueError as exc:
                raise PipelineAuditError(
                    f"unsafe path in frozen source snapshot: {member.name!r}"
                ) from exc
            if member.issym() or member.islnk():
                raise PipelineAuditError(
                    f"links are forbidden in frozen source snapshot: {member.name!r}"
                )
        archive.extractall(destination, members=members)


def _verify_runtime_snapshot(snapshot: Path, runtime_root: Path) -> dict[str, str]:
    checksum_path = Path(str(snapshot) + ".sha256")
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        actual = sha256_file(snapshot)
        _require(actual == expected, "frozen source snapshot SHA-256 mismatch")
    internal_manifest_path = runtime_root / ".source_freeze/source_manifest.json"
    _require(internal_manifest_path.is_file(), "snapshot source manifest is missing")
    internal_manifest = load_json(internal_manifest_path)
    _require(
        internal_manifest.get("source_id") == FROZEN_SOURCE_ID,
        "snapshot source_id does not match the first formal search",
    )
    manifest_files = {
        row["path"]: row["sha256"] for row in internal_manifest.get("files", [])
    }
    required = (
        "eval_utils.py",
        "final_eval.py",
        "dataset_utils.py",
        "evaluation_errors.py",
        "hp_modes.py",
    )
    hashes: dict[str, str] = {}
    for relative in required:
        path = runtime_root / relative
        _require(path.is_file(), f"snapshot runtime file missing: {relative}")
        actual = sha256_file(path)
        _require(
            manifest_files.get(relative) == actual,
            f"snapshot runtime hash mismatch: {relative}",
        )
        hashes[relative] = actual
    eval_text = (runtime_root / "eval_utils.py").read_text(encoding="utf-8")
    _require(
        "assert_strict_torch_determinism" not in eval_text
        and "configure_torch_determinism" not in eval_text,
        "first-formal runtime unexpectedly enables strict deterministic algorithms",
    )
    return hashes


def _history_sha_map(analysis_manifest: Mapping[str, Any]) -> dict[str, str]:
    raw = analysis_manifest.get("input_files_sha256", {})
    if isinstance(raw, dict):
        return {str(path): str(digest) for path, digest in raw.items()}
    result: dict[str, str] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "path" in item and "sha256" in item:
                result[str(item["path"])] = str(item["sha256"])
    return result


def _stage_expectation(method: str, full_index: int) -> tuple[str, int]:
    if method == "S0":
        return ("initial_seed", full_index + 1) if full_index < 50 else (
            "online_bo",
            full_index - 49,
        )
    expansion_end = 150 if method == "G100" else 200
    if full_index < 50:
        return "initial_seed", full_index + 1
    if full_index < expansion_end:
        return "initial_expand", full_index - 49
    return "online_bo", full_index - expansion_end + 1


def _history_hp(record: Mapping[str, Any]) -> dict[str, Any]:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    return {
        "lr": _finite_float(record.get("lr"), "history lr"),
        "dropout": _finite_float(record.get("dropout"), "history dropout"),
        "hidden_dim": int(record.get("hidden_dim")),
        "l2": _finite_float(
            record.get("l2", record.get("weight_decay")), "history l2"
        ),
        "gat_heads": int(record.get("gat_heads", 1)),
        "sage_aggr": str(record.get("sage_aggr", "mean")),
        "gin_eps": _finite_float(record.get("gin_eps", 0.0), "history gin_eps"),
        "gat_heads_by_layer": record.get("gat_heads_by_layer"),
        "sage_aggr_by_layer": record.get("sage_aggr_by_layer"),
        "gin_eps_by_layer": record.get("gin_eps_by_layer"),
        "gcnii_alpha": _finite_float(params.get("gcnii_alpha", 0.1), "gcnii_alpha"),
        "gcnii_theta": _finite_float(params.get("gcnii_theta", 0.5), "gcnii_theta"),
        "hp_mode": str(record.get("hp_mode", "global4")),
    }


def _architecture(record: Mapping[str, Any]) -> dict[str, Any]:
    operations = record.get("operations")
    edges = record.get("edges")
    _require(isinstance(operations, list), "history operations must be a list")
    _require(isinstance(edges, list), "history edges must be a list")
    normalized_edges: list[list[int]] = []
    for edge in edges:
        _require(
            isinstance(edge, (list, tuple)) and len(edge) == 2,
            f"invalid history edge: {edge!r}",
        )
        normalized_edges.append([int(edge[0]), int(edge[1])])
    return {
        "operations": [str(operation) for operation in operations],
        "edges": normalized_edges,
    }


def select_top10_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    eligible: list[Mapping[str, Any]] = []
    for record in records:
        if record.get("evaluation_fidelity") != "full":
            continue
        if record.get("valid") is not True:
            continue
        try:
            validation = float(record.get("val_acc"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(validation):
            continue
        fingerprint = record.get("candidate_fingerprint")
        if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
            continue
        try:
            full_index = int(record.get("full_evaluation_index"))
        except (TypeError, ValueError):
            continue
        normalized = dict(record)
        normalized["_selection_validation"] = validation
        normalized["_selection_full_index"] = full_index
        eligible.append(normalized)
    eligible.sort(
        key=lambda row: (
            -float(row["_selection_validation"]),
            int(row["_selection_full_index"]),
            str(row["candidate_fingerprint"]),
        )
    )
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in eligible:
        fingerprint = str(record["candidate_fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(record)
        if len(selected) == 10:
            break
    return selected


def representative_origin(origins: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    _require(bool(origins), "cannot select a representative from no origins")
    return min(
        origins,
        key=lambda row: (
            -float(row["search_validation_accuracy"]),
            int(row["search_seed"]),
            int(row["global_full_evaluation_index"]),
            str(row["candidate_fingerprint"]),
        ),
    )


def representative_method_origin(
    origins: Sequence[Mapping[str, Any]], method: str
) -> Mapping[str, Any]:
    """Choose provenance for the method whose winner row is being reported."""

    _require(method in METHODS, f"unknown method for winner origin: {method}")
    method_origins = [row for row in origins if row.get("method") == method]
    _require(bool(method_origins), f"winner candidate has no {method} origin")
    return representative_origin(method_origins)


def final_evaluation_seed(
    final_base_seed: int, fingerprint: str, replicate_id: int
) -> int:
    """Mirror frozen final_eval.final_evaluation_seed for manifest preflight."""

    _require(bool(SHA256_RE.fullmatch(fingerprint)), "invalid candidate fingerprint")
    digest = hashlib.sha256()
    fixed_parts = (
        (b"version", b"sha256_v1"),
        (b"base_seed", str(int(final_base_seed)).encode("ascii")),
        (b"namespace", b"final_evaluation"),
    )
    components = (
        (b"str", fingerprint.encode("utf-8")),
        (b"int", str(int(replicate_id)).encode("ascii")),
    )
    for tag, payload in (*fixed_parts, *components):
        digest.update(len(tag).to_bytes(4, "big"))
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return int.from_bytes(digest.digest()[:8], "big") % (2**32)


def evaluation_cache_key(
    *,
    dataset: str,
    fingerprint: str,
    replicate_id: int,
    protocol_fingerprint: str,
    dataset_context: Mapping[str, Any],
) -> str:
    return json_fingerprint(
        {
            "dataset": dataset,
            "candidate_fingerprint": fingerprint,
            "replicate_id": int(replicate_id),
            "final_evaluation_protocol_fingerprint": protocol_fingerprint,
            "dataset_identity": {
                "dataset_content_fingerprint": dataset_context.get(
                    "dataset_content_fingerprint"
                ),
                "split_protocol": dataset_context.get("split_protocol"),
                "split_seed": dataset_context.get("split_seed"),
                "split_fingerprint": dataset_context.get("split_fingerprint"),
                "graph_transform": dataset_context.get("graph_transform"),
                "training_mode": dataset_context.get("training_mode"),
                "metric": dataset_context.get("metric"),
            },
        }
    )


def _stable_seed_self_check(runtime_root: Path) -> None:
    script = (
        "import json,sys;sys.path.insert(0,sys.argv[1]);"
        "from final_eval import final_evaluation_seed;"
        "print(final_evaluation_seed(0,'0'*64,0))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(runtime_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = final_evaluation_seed(0, "0" * 64, 0)
    _require(
        int(completed.stdout.strip()) == expected,
        "local seed derivation does not match frozen final_eval.py",
    )


def _history_identity_sources(
    analysis_root: Path, summary_path: Path
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    dict[tuple[str, str, int], Mapping[str, Any]],
    dict[tuple[str, str, int], list[dict[str, str]]],
    dict[tuple[str, str, int], Mapping[str, str]],
]:
    audit = load_json(analysis_root / "audit_report.json")
    manifest = load_json(analysis_root / "analysis_manifest.json")
    _require(audit.get("completion_audit_passed") is True, "first-round audit failed")
    _require(audit.get("unique_task_count") == EXPECTED_HISTORY_COUNT, "audit task count")
    _require(
        audit.get("full_validation_record_count") == 18000,
        "audit full-validation count is not 18,000",
    )
    _require(audit.get("errors") == [], "first-round audit contains errors")
    _require(audit.get("source_id") == FROZEN_SOURCE_ID, "audit source_id mismatch")
    _require(manifest.get("source_id") == FROZEN_SOURCE_ID, "manifest source_id mismatch")
    _require(manifest.get("test_data_used") is False, "first analysis used test data")
    _require(manifest.get("gpu_training_started") is False, "analysis started training")
    histories = [Path(path).resolve() for path in manifest.get("history_final_paths", [])]
    _require(len(histories) == EXPECTED_HISTORY_COUNT, "manifest history count is not 60")
    _require(len(set(histories)) == EXPECTED_HISTORY_COUNT, "duplicate audited histories")
    _require(not any(_path_is_excluded(path) for path in histories), "excluded history found")

    task_map: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for task in manifest.get("tasks", []):
        key = (str(task["dataset"]).lower(), str(task["method"]), int(task["search_seed"]))
        _require(key not in task_map, f"duplicate analysis task: {key}")
        task_map[key] = task
    expected_keys = {
        (dataset, method, seed)
        for dataset in DATASETS
        for method in METHODS
        for seed in SEARCH_SEEDS
    }
    _require(set(task_map) == expected_keys, "analysis task matrix is not exact")

    evaluation_rows = load_csv(analysis_root / "evaluation_long.csv")
    evaluation_map: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in evaluation_rows:
        key = (row["dataset"].lower(), row["method"], int(row["search_seed"]))
        evaluation_map.setdefault(key, []).append(row)
    _require(set(evaluation_map) == expected_keys, "evaluation_long task matrix mismatch")
    _require(
        all(len(rows) == 300 for rows in evaluation_map.values()),
        "evaluation_long does not contain 300 rows per task",
    )

    summary_rows = load_csv(summary_path)
    formal_rows = [
        row
        for row in summary_rows
        if row.get("experiment_family") == "previous_nondeterministic_formal"
    ]
    deterministic_rows = [
        row for row in summary_rows if row.get("experiment_family") == "deterministic_rerun"
    ]
    _require(len(formal_rows) == 60, "06 summary does not have 60 first-formal rows")
    _require(len(deterministic_rows) == 60, "06 summary deterministic exclusion is incomplete")
    summary_map: dict[tuple[str, str, int], Mapping[str, str]] = {}
    for row in formal_rows:
        key = (row["dataset"].lower(), row["method"], int(row["search_seed"]))
        _require(key not in summary_map, f"duplicate first-formal summary row: {key}")
        _require(
            not _path_is_excluded(Path(row["source_history"])),
            f"formal summary points into excluded deterministic root: {key}",
        )
        summary_map[key] = row
    _require(set(summary_map) == expected_keys, "06 first-formal task matrix mismatch")
    return audit, manifest, task_map, evaluation_map, summary_map


def _resolve_audited_history(
    task: Mapping[str, Any],
    audited_histories: set[Path],
    summary_row: Mapping[str, str],
) -> Path:
    output_directory = Path(str(task["output_directory"])).resolve()
    candidates = [path for path in audited_histories if path.parent == output_directory]
    _require(
        len(candidates) == 1,
        f"analysis audit did not resolve exactly one history for {task['task_id']}",
    )
    history = candidates[0]
    _require(
        history == Path(summary_row["source_history"]).resolve(),
        f"06 summary history disagrees with analysis audit for {task['task_id']}",
    )
    _require(history.is_file(), f"audited history is missing: {history}")
    _require(not _path_is_excluded(history), f"excluded deterministic history: {history}")
    return history


def _validate_history_against_analysis(
    *,
    dataset: str,
    method: str,
    search_seed: int,
    task: Mapping[str, Any],
    history: Path,
    records: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, str]],
) -> tuple[Mapping[str, Any], Path, list[Mapping[str, Any]]]:
    metadata_path = history.with_name("history_metadata.json")
    _require(metadata_path.is_file(), f"history metadata missing: {metadata_path}")
    metadata = load_json(metadata_path)
    _require(int(metadata.get("search_seed")) == search_seed, "metadata search seed")
    _require(metadata.get("method_label") == METHOD_LABELS[method], "metadata method")
    context = metadata.get("dataset_context")
    _require(isinstance(context, dict), "metadata dataset_context is missing")
    _require(context.get("canonical_name") == dataset, "metadata dataset mismatch")
    _require(int(metadata.get("max_total_full_evals")) == 300, "metadata full budget")
    _require(int(metadata.get("eval_epochs")) == EVAL_EPOCHS, "metadata eval_epochs")
    _require(int(metadata.get("patience")) == PATIENCE, "metadata patience")
    _require(metadata.get("training_mode") == "full_batch", "metadata training mode")
    _require(len(records) == 300, f"history does not have 300 full records: {history}")

    rows_by_index = {int(row["full_evaluation_index"]): row for row in evaluation_rows}
    _require(set(rows_by_index) == set(range(300)), "evaluation_long indices are not 0..299")
    records_by_index: dict[int, Mapping[str, Any]] = {}
    for record in records:
        _require(record.get("evaluation_fidelity") == "full", "non-full record in history")
        index = int(record.get("full_evaluation_index"))
        _require(index not in records_by_index, f"duplicate full index {index}: {history}")
        records_by_index[index] = record
    _require(set(records_by_index) == set(range(300)), "history indices are not 0..299")

    ordered: list[Mapping[str, Any]] = []
    for index in range(300):
        record = records_by_index[index]
        row = rows_by_index[index]
        fingerprint = str(record.get("candidate_fingerprint"))
        _require(bool(SHA256_RE.fullmatch(fingerprint)), "invalid persisted fingerprint")
        _require(record.get("z_search") is not None, "history z_search is missing")
        _require(
            candidate_fingerprint(record["z_search"]) == fingerprint,
            f"candidate fingerprint mismatch at {history}:{index}",
        )
        _architecture(record)
        _history_hp(record)
        _require(int(record.get("search_seed")) == search_seed, "record search seed")
        _require(record.get("method_label") == METHOD_LABELS[method], "record method")
        record_context = record.get("dataset_context")
        _require(record_context == context, "record dataset context differs from metadata")
        expected_stage, _ = _stage_expectation(method, index)
        _require(
            record.get("evaluation_stage") == expected_stage,
            f"history stage mismatch at {history}:{index}",
        )
        _require(row["task_id"] == task["task_id"], "evaluation_long task_id mismatch")
        _require(row["output_directory"] == task["output_directory"], "output mismatch")
        _require(row["candidate_fingerprint"] == fingerprint, "evaluation fingerprint")
        _require(row["stage"] == record["evaluation_stage"], "evaluation stage")
        _require(
            _parse_bool(row["valid"], "evaluation_long valid")
            == (record.get("valid") is True),
            "evaluation validity mismatch",
        )
        _require(
            _finite_float(row["validation_accuracy"], "evaluation validation")
            == _finite_float(record.get("val_acc"), "history validation"),
            "evaluation validation mismatch",
        )
        _require(
            json.loads(row["architecture"]) == _architecture(record),
            "evaluation architecture mismatch",
        )
        ordered.append(record)
    return metadata, metadata_path, ordered


def _low_fidelity_map(history: Path) -> dict[str, tuple[int, Mapping[str, Any]]]:
    low_path = history.with_name("low_fidelity_history.json")
    if not low_path.exists():
        return {}
    rows = load_json(low_path)
    _require(isinstance(rows, list), f"low-fidelity history is not a list: {low_path}")
    result: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for position, row in enumerate(rows, start=1):
        record_id = row.get("record_id")
        _require(isinstance(record_id, str) and record_id, "low-fidelity record_id missing")
        _require(record_id not in result, f"duplicate low-fidelity record_id: {record_id}")
        result[record_id] = (position, row)
    return result


def _origin_row(
    *,
    dataset: str,
    method: str,
    search_seed: int,
    task_id: str,
    history: Path,
    history_sha256: str,
    metadata_path: Path,
    record: Mapping[str, Any],
    rank: int,
    low_map: Mapping[str, tuple[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    full_index = int(record["full_evaluation_index"])
    search_stage = str(record["evaluation_stage"])
    expected_stage, reconstructed_position = _stage_expectation(method, full_index)
    _require(search_stage == expected_stage, "selected candidate stage mismatch")
    explicit_fields = (
        "within_stage_full_position",
        "stage_full_position",
        "evaluation_stage_position",
    )
    explicit = next((record.get(field) for field in explicit_fields if field in record), None)
    if explicit is None:
        within_position = reconstructed_position
        reconstructed = True
    else:
        within_position = int(explicit)
        reconstructed = False

    low_id = record.get("low_fidelity_record_id")
    low_position: int | None = None
    promotion_position: int | None = None
    if low_id is not None:
        _require(str(low_id) in low_map, f"low-fidelity origin is missing: {low_id}")
        low_position, low_record = low_map[str(low_id)]
        _require(
            low_record.get("candidate_fingerprint") == record.get("candidate_fingerprint"),
            f"low/full candidate fingerprint mismatch: {low_id}",
        )
        raw_promotion = low_record.get("promotion_rank")
        promotion_position = None if raw_promotion is None else int(raw_promotion)
    elif record.get("low_fidelity_used") is True:
        raise PipelineAuditError("low_fidelity_used candidate lacks low_fidelity_record_id")

    architecture = _architecture(record)
    hp = _history_hp(record)
    config_identity = {
        "architecture": architecture,
        "lr": hp["lr"],
        "dropout": hp["dropout"],
        "hidden_dim": hp["hidden_dim"],
        "l2": hp["l2"],
    }
    context = record["dataset_context"]
    return {
        "dataset": dataset,
        "method": method,
        "search_seed": search_seed,
        "task_id": task_id,
        "source_history_path": str(history),
        "source_history_sha256": history_sha256,
        "history_metadata_path": str(metadata_path),
        "source_id": FROZEN_SOURCE_ID,
        "candidate_fingerprint": str(record["candidate_fingerprint"]),
        "validation_top10_rank": rank,
        "search_validation_accuracy": float(record["val_acc"]),
        "search_stage": search_stage,
        "within_stage_full_position": within_position,
        "stage_position_reconstructed": reconstructed,
        "global_full_evaluation_index": full_index,
        "global_full_evaluation_position": full_index + 1,
        "low_fidelity_position": low_position,
        "promotion_position": promotion_position,
        "low_fidelity_record_id": low_id,
        "architecture": architecture,
        "hyperparameters": hp,
        **hp,
        "canonical_z_search": record["z_search"],
        "dataset_context": context,
        "candidate_config_fingerprint": json_fingerprint(config_identity),
    }


def _validate_existing_top10(
    analysis_root: Path, origins: Sequence[Mapping[str, Any]]
) -> None:
    existing = load_csv(analysis_root / "validation_top10_candidates.csv")
    _require(len(existing) == EXPECTED_SOURCE_SLOTS, "existing Top-10 is not 600 rows")
    expected = {
        (row["dataset"], row["method"], int(row["search_seed"]), int(row["validation_rank"])): row
        for row in existing
    }
    _require(len(expected) == EXPECTED_SOURCE_SLOTS, "existing Top-10 keys duplicate")
    for origin in origins:
        key = (
            str(origin["dataset"]),
            str(origin["method"]),
            int(origin["search_seed"]),
            int(origin["validation_top10_rank"]),
        )
        row = expected.get(key)
        _require(row is not None, f"existing Top-10 row missing: {key}")
        _require(
            row["candidate_fingerprint"] == origin["candidate_fingerprint"],
            f"Top-10 fingerprint disagreement: {key}",
        )
        _require(
            float(row["validation_accuracy"])
            == float(origin["search_validation_accuracy"]),
            f"Top-10 validation disagreement: {key}",
        )


def _compact_origin(origin: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": origin["method"],
        "search_seed": int(origin["search_seed"]),
        "stage": origin["search_stage"],
        "stage_position": int(origin["within_stage_full_position"]),
        "global_full_evaluation_index": int(origin["global_full_evaluation_index"]),
        "global_position": int(origin["global_full_evaluation_position"]),
        "validation_top10_rank": int(origin["validation_top10_rank"]),
        "search_validation_accuracy": float(origin["search_validation_accuracy"]),
        "source_history_path": origin["source_history_path"],
    }


def _build_unique_manifest(
    origins: Sequence[Mapping[str, Any]], protocol_fingerprint: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for origin in origins:
        key = (str(origin["dataset"]), str(origin["candidate_fingerprint"]))
        grouped.setdefault(key, []).append(origin)
    unique_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    for (dataset, fingerprint), group in sorted(grouped.items()):
        config_fingerprints = {row["candidate_config_fingerprint"] for row in group}
        _require(
            len(config_fingerprints) == 1,
            f"same persisted candidate has inconsistent architecture/HP: {dataset}/{fingerprint}",
        )
        architectures = {canonical_json(row["architecture"]) for row in group}
        hps = {canonical_json(row["hyperparameters"]) for row in group}
        contexts = {canonical_json(row["dataset_context"]) for row in group}
        _require(len(architectures) == len(hps) == len(contexts) == 1, "origin mismatch")
        representative = representative_origin(group)
        compact_origins = [_compact_origin(row) for row in sorted(
            group,
            key=lambda item: (
                METHODS.index(str(item["method"])),
                int(item["search_seed"]),
                int(item["validation_top10_rank"]),
            ),
        )]
        row = {
            "dataset": dataset,
            "candidate_fingerprint": fingerprint,
            "candidate_config_fingerprint": representative["candidate_config_fingerprint"],
            "architecture": representative["architecture"],
            "hyperparameters": representative["hyperparameters"],
            "lr": representative["lr"],
            "dropout": representative["dropout"],
            "hidden_dim": representative["hidden_dim"],
            "l2": representative["l2"],
            "gat_heads": representative["gat_heads"],
            "sage_aggr": representative["sage_aggr"],
            "gin_eps": representative["gin_eps"],
            "gat_heads_by_layer": representative["gat_heads_by_layer"],
            "sage_aggr_by_layer": representative["sage_aggr_by_layer"],
            "gin_eps_by_layer": representative["gin_eps_by_layer"],
            "gcnii_alpha": representative["gcnii_alpha"],
            "gcnii_theta": representative["gcnii_theta"],
            "hp_mode": representative["hp_mode"],
            "dataset_context": representative["dataset_context"],
            "dataset_content_fingerprint": representative["dataset_context"].get(
                "dataset_content_fingerprint"
            ),
            "split_fingerprint": representative["dataset_context"].get(
                "split_fingerprint"
            ),
            "origin_count": len(group),
            "origin_methods": sorted({str(item["method"]) for item in group}, key=METHODS.index),
            "origin_search_seeds": sorted({int(item["search_seed"]) for item in group}),
            "representative_method": representative["method"],
            "representative_search_seed": representative["search_seed"],
            "representative_search_validation_accuracy": representative[
                "search_validation_accuracy"
            ],
            "all_origins": compact_origins,
            "final_evaluation_protocol_fingerprint": protocol_fingerprint,
        }
        unique_rows.append(row)
        representative_key = (
            representative["method"],
            representative["search_seed"],
            representative["global_full_evaluation_index"],
        )
        for origin in group:
            enriched = dict(origin)
            enriched["is_representative_origin"] = (
                origin["method"],
                origin["search_seed"],
                origin["global_full_evaluation_index"],
            ) == representative_key
            origin_rows.append(enriched)
    origin_rows.sort(
        key=lambda row: (
            DATASETS.index(str(row["dataset"])),
            METHODS.index(str(row["method"])),
            int(row["search_seed"]),
            int(row["validation_top10_rank"]),
        )
    )
    return unique_rows, origin_rows


def _derive_data_root(task_map: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> str:
    roots: set[str] = set()
    for task in task_map.values():
        argv = [str(item) for item in task.get("exact_argv", [])]
        _require("--data-root" in argv, f"task has no --data-root: {task['task_id']}")
        index = argv.index("--data-root")
        _require(index + 1 < len(argv), f"task has empty --data-root: {task['task_id']}")
        roots.add(str(Path(argv[index + 1]).resolve()))
    _require(len(roots) == 1, f"formal tasks disagree on data root: {sorted(roots)}")
    return next(iter(roots))


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def _initialize_database(
    database_path: Path,
    unique_rows: Sequence[Mapping[str, Any]],
    protocol_fingerprint: str,
) -> int:
    connection = _open_database(database_path)
    try:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                queue_order INTEGER NOT NULL UNIQUE,
                dataset TEXT NOT NULL,
                candidate_fingerprint TEXT NOT NULL,
                replicate_id INTEGER NOT NULL,
                effective_training_seed INTEGER NOT NULL,
                cache_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                claimed_by TEXT,
                worker_pid INTEGER,
                worker_host TEXT,
                gpu_id INTEGER,
                started_at TEXT,
                completed_at TEXT,
                result_path TEXT NOT NULL,
                error_path TEXT,
                UNIQUE(dataset, candidate_fingerprint, replicate_id)
            );
            CREATE INDEX tasks_status_order ON tasks(status, queue_order);
            """
        )
        connection.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?)",
            ("protocol_fingerprint", protocol_fingerprint),
        )
        order = 0
        for candidate in unique_rows:
            dataset = str(candidate["dataset"])
            fingerprint = str(candidate["candidate_fingerprint"])
            context = candidate["dataset_context"]
            for replicate_id in REPLICATE_IDS:
                seed = final_evaluation_seed(FINAL_BASE_SEED, fingerprint, replicate_id)
                cache_key = evaluation_cache_key(
                    dataset=dataset,
                    fingerprint=fingerprint,
                    replicate_id=replicate_id,
                    protocol_fingerprint=protocol_fingerprint,
                    dataset_context=context,
                )
                result_path = (
                    Path("replicates")
                    / dataset
                    / fingerprint
                    / f"replicate_{replicate_id:02d}.json"
                )
                task_id = f"{dataset}:{fingerprint}:{replicate_id}"
                connection.execute(
                    """
                    INSERT INTO tasks(
                        task_id,queue_order,dataset,candidate_fingerprint,replicate_id,
                        effective_training_seed,cache_key,status,result_path
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        task_id,
                        order,
                        dataset,
                        fingerprint,
                        replicate_id,
                        seed,
                        cache_key,
                        "pending",
                        str(result_path),
                    ),
                )
                order += 1
        connection.commit()
        return order
    finally:
        connection.close()


def _status_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(
        """
        SELECT queue_order,dataset,candidate_fingerprint,replicate_id,
               effective_training_seed,status,attempts,claimed_by,worker_pid,
               worker_host,gpu_id,started_at,completed_at,result_path,error_path
        FROM tasks ORDER BY queue_order
        """
    )]


def write_status_tsv(pipeline_root: Path) -> None:
    lock_path = pipeline_root / "queue_status.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        connection = _open_database(pipeline_root / "queue.sqlite3")
        try:
            rows = _status_rows(connection)
        finally:
            connection.close()
        fields = list(rows[0]) if rows else []
        atomic_write_text(
            pipeline_root / "queue_status.tsv",
            csv_text(rows, fields, delimiter="\t"),
        )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _append_event(pipeline_root: Path, event: Mapping[str, Any]) -> None:
    path = pipeline_root / "queue_events.jsonl"
    lock_path = pipeline_root / "queue_events.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(dict(event)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _append_progress_tsv(
    pipeline_root: Path,
    *,
    event: str,
    task: Mapping[str, Any],
    worker_id: str,
    gpu_id: int,
    artifact_path: str,
) -> None:
    path = pipeline_root / "queue_progress.tsv"
    lock_path = pipeline_root / "queue_progress.lock"
    lock_path.touch(exist_ok=True)
    fields = (
        "timestamp",
        "event",
        "task_id",
        "dataset",
        "candidate_fingerprint",
        "replicate_id",
        "status",
        "worker_id",
        "gpu_id",
        "attempt",
        "artifact_path",
    )
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        "task_id": task["task_id"],
        "dataset": task["dataset"],
        "candidate_fingerprint": task["candidate_fingerprint"],
        "replicate_id": task["replicate_id"],
        "status": "completed" if event == "replicate_completed" else "failed",
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "attempt": task["attempts"],
        "artifact_path": artifact_path,
    }
    with lock_path.open("r+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        new_file = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def prepare_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    from experiment_paths import validate_protocol_id

    protocol_id = validate_protocol_id(args.protocol_id)
    pipeline_root = (
        REPOSITORY_ROOT
        / "results"
        / "final_eval"
        / protocol_id
        / "first_formal_pipeline"
        if args.pipeline_root is None
        else Path(args.pipeline_root)
    ).resolve()
    _require(not pipeline_root.exists(), f"pipeline root already exists: {pipeline_root}")
    analysis_root = Path(args.analysis_root).resolve()
    summary_path = Path(args.search_summary).resolve()
    snapshot = Path(args.frozen_snapshot).resolve()
    _require(analysis_root == DEFAULT_ANALYSIS_ROOT.resolve(), "unexpected analysis root")
    _require(summary_path.is_file(), f"06 summary is missing: {summary_path}")
    _require(snapshot.is_file(), f"frozen source snapshot is missing: {snapshot}")
    pipeline_root.mkdir(parents=True)
    try:
        runtime_root = pipeline_root / "runtime_first_formal"
        _safe_extract_snapshot(snapshot, runtime_root)
        runtime_hashes = _verify_runtime_snapshot(snapshot, runtime_root)
        _stable_seed_self_check(runtime_root)

        audit, manifest, task_map, evaluation_map, summary_map = _history_identity_sources(
            analysis_root, summary_path
        )
        audited_histories = {
            Path(path).resolve() for path in manifest["history_final_paths"]
        }
        recorded_hashes = _history_sha_map(manifest)
        origins: list[dict[str, Any]] = []
        history_audit_rows: list[dict[str, Any]] = []
        invalid_count = 0
        for dataset in DATASETS:
            for method in METHODS:
                for search_seed in SEARCH_SEEDS:
                    key = (dataset, method, search_seed)
                    task = task_map[key]
                    history = _resolve_audited_history(
                        task, audited_histories, summary_map[key]
                    )
                    digest = sha256_file(history)
                    recorded = recorded_hashes.get(str(history))
                    if recorded is not None:
                        _require(recorded == digest, f"analysis input hash mismatch: {history}")
                    records = load_json(history)
                    _require(isinstance(records, list), f"history is not a list: {history}")
                    metadata, metadata_path, ordered = _validate_history_against_analysis(
                        dataset=dataset,
                        method=method,
                        search_seed=search_seed,
                        task=task,
                        history=history,
                        records=records,
                        evaluation_rows=evaluation_map[key],
                    )
                    invalid_count += sum(record.get("valid") is not True for record in ordered)
                    low_map = _low_fidelity_map(history)
                    selected = select_top10_records(ordered)
                    _require(len(selected) == 10, f"Top-10 extraction failed: {key}")
                    for rank, record in enumerate(selected, start=1):
                        origins.append(
                            _origin_row(
                                dataset=dataset,
                                method=method,
                                search_seed=search_seed,
                                task_id=str(task["task_id"]),
                                history=history,
                                history_sha256=digest,
                                metadata_path=metadata_path,
                                record=record,
                                rank=rank,
                                low_map=low_map,
                            )
                        )
                    history_audit_rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "search_seed": search_seed,
                            "task_id": task["task_id"],
                            "history_path": str(history),
                            "history_sha256": digest,
                            "record_count": len(ordered),
                            "valid_count": sum(record.get("valid") is True for record in ordered),
                            "invalid_count": sum(record.get("valid") is not True for record in ordered),
                            "top10_count": len(selected),
                            "metadata": metadata,
                        }
                    )

        _require(len(history_audit_rows) == EXPECTED_HISTORY_COUNT, "history audit count")
        _require(len(origins) == EXPECTED_SOURCE_SLOTS, "source slot count is not 600")
        _require(invalid_count == int(audit["invalid_full_record_count"]), "invalid count")
        group_counts: dict[tuple[str, str, int], int] = {}
        for row in origins:
            key = (row["dataset"], row["method"], int(row["search_seed"]))
            group_counts[key] = group_counts.get(key, 0) + 1
        _require(
            len(group_counts) == 60 and set(group_counts.values()) == {10},
            "each history must contribute exactly 10 source slots",
        )
        _validate_existing_top10(analysis_root, origins)

        protocol = {
            "analysis_label": ANALYSIS_LABEL,
            "source_id": FROZEN_SOURCE_ID,
            "runtime_hashes": runtime_hashes,
            "eval_epochs": EVAL_EPOCHS,
            "patience": PATIENCE,
            "training_mode": "full_batch",
            "metric": "accuracy",
            "test_value_semantics": "test accuracy at best validation epoch",
            "final_base_seed": FINAL_BASE_SEED,
            "final_evaluation_seed_scheme": FINAL_SEED_SCHEME,
            "replicate_ids": list(REPLICATE_IDS),
            "strict_deterministic_algorithms": False,
            "candidate_source": "persisted discrete architecture and hyperparameters",
            "vae_decode": False,
            "final_validation_selection": True,
            "winner_selection": (
                "highest arithmetic mean of 30 final-validation accuracies; "
                "test metrics are report-only"
            ),
        }
        protocol_fingerprint = json_fingerprint(protocol)
        unique_rows, origin_rows = _build_unique_manifest(origins, protocol_fingerprint)
        _require(
            len({(row["dataset"], row["candidate_fingerprint"]) for row in unique_rows})
            == len(unique_rows),
            "unique manifest key duplication",
        )
        _require(
            len({(row["dataset"], row["method"]) for row in origins}) == 12,
            "dataset-method group count is not 12",
        )
        data_root = _derive_data_root(task_map)
        protocol["data_root_from_formal_argv"] = data_root
        # Data root is an operational locator, not part of the training/cache seed.
        atomic_write_json(pipeline_root / "protocol.json", protocol)
        atomic_write_csv(pipeline_root / "top10_source_slots.csv", origins, ORIGIN_FIELDS)
        unique_fields = list(unique_rows[0])
        atomic_write_csv(
            pipeline_root / "unique_candidate_manifest.csv", unique_rows, unique_fields
        )
        atomic_write_csv(
            pipeline_root / "candidate_origin_map.csv",
            origin_rows,
            (*ORIGIN_FIELDS, "is_representative_origin"),
        )
        atomic_write_json(pipeline_root / "history_audit.json", history_audit_rows)
        total_tasks = _initialize_database(
            pipeline_root / "queue.sqlite3", unique_rows, protocol_fingerprint
        )
        theoretical = EXPECTED_SOURCE_SLOTS * len(REPLICATE_IDS)
        tmux_session = re.sub(r"[^A-Za-z0-9_-]", "_", pipeline_root.name)[:80]
        config = {
            "format_version": 1,
            "analysis_label": ANALYSIS_LABEL,
            "repository_root": str(REPOSITORY_ROOT),
            "pipeline_root": str(pipeline_root),
            "analysis_root": str(analysis_root),
            "search_summary": str(summary_path),
            "excluded_deterministic_root": str(EXCLUDED_DETERMINISTIC_ROOT),
            "frozen_snapshot": str(snapshot),
            "frozen_runtime_root": str(runtime_root),
            "source_id": FROZEN_SOURCE_ID,
            "data_root": data_root,
            "eval_epochs": EVAL_EPOCHS,
            "patience": PATIENCE,
            "final_base_seed": FINAL_BASE_SEED,
            "final_evaluation_seed_scheme": FINAL_SEED_SCHEME,
            "final_evaluation_protocol_fingerprint": protocol_fingerprint,
            "history_count": len(history_audit_rows),
            "source_slot_count": len(origins),
            "dataset_method_group_count": 12,
            "unique_candidate_count": len(unique_rows),
            "replicate_task_count": total_tasks,
            "theoretical_training_limit": theoretical,
            "dedup_training_savings": theoretical - total_tasks,
            "tmux_session": tmux_session,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_write_json(pipeline_root / "pipeline_config.json", config)
        manifest_audit = {
            "status": "manifest_audit_passed",
            "analysis_label": ANALYSIS_LABEL,
            "completion_audit_passed": True,
            "analysis_source_id": FROZEN_SOURCE_ID,
            "excluded_deterministic_root_used": False,
            "search_histories": len(history_audit_rows),
            "full_evaluation_records": sum(row["record_count"] for row in history_audit_rows),
            "invalid_full_records_excluded": invalid_count,
            "source_slots": len(origins),
            "top10_per_history": True,
            "dataset_method_groups": 12,
            "unique_dataset_candidates": len(unique_rows),
            "replicate_ids": list(REPLICATE_IDS),
            "replicate_tasks": total_tasks,
            "dedup_training_savings": theoretical - total_tasks,
            "history_vs_evaluation_long_match": True,
            "history_vs_06_summary_match": True,
            "history_vs_existing_validation_top10_match": True,
            "frozen_runtime_hashes": runtime_hashes,
            "strict_deterministic_algorithms": False,
            "gpu_training_started": False,
            "final_winners_available": False,
        }
        atomic_write_json(pipeline_root / "manifest_audit_report.json", manifest_audit)
        write_status_tsv(pipeline_root)
        _append_event(
            pipeline_root,
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": "manifest_prepared",
                "unique_candidates": len(unique_rows),
                "replicate_tasks": total_tasks,
            },
        )
        return config
    except BaseException:
        atomic_write_json(
            pipeline_root / "preparation_error.json",
            {
                "status": "preparation_failed",
                "traceback": traceback.format_exc(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        raise


def _process_alive(pid: int, host: str | None) -> bool:
    if host and host != socket.gethostname():
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def recover_stale_tasks(pipeline_root: Path) -> int:
    connection = _open_database(pipeline_root / "queue.sqlite3")
    recovered: list[dict[str, Any]] = []
    try:
        connection.execute("BEGIN IMMEDIATE")
        for row in connection.execute("SELECT * FROM tasks WHERE status='running'"):
            pid = int(row["worker_pid"] or -1)
            if pid > 0 and _process_alive(pid, row["worker_host"]):
                continue
            recovered.append(dict(row))
            connection.execute(
                """
                UPDATE tasks SET status='pending',claimed_by=NULL,worker_pid=NULL,
                    worker_host=NULL,gpu_id=NULL,started_at=NULL
                WHERE task_id=? AND status='running'
                """,
                (row["task_id"],),
            )
        connection.commit()
    finally:
        connection.close()
    for row in recovered:
        _append_event(
            pipeline_root,
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": "stale_task_recovered",
                "task_id": row["task_id"],
                "previous_worker_pid": row["worker_pid"],
                "previous_worker_host": row["worker_host"],
            },
        )
    if recovered:
        write_status_tsv(pipeline_root)
    return len(recovered)


def _validate_result_artifact(task: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    _require(result.get("status") == "completed", "result artifact status")
    for field in ("dataset", "candidate_fingerprint", "replicate_id", "cache_key"):
        _require(str(result.get(field)) == str(task[field]), f"result {field} mismatch")
    _require(
        int(result.get("effective_training_seed")) == int(task["effective_training_seed"]),
        "result seed mismatch",
    )
    _finite_float(result.get("test_accuracy"), "result test accuracy")
    _finite_float(result.get("validation_accuracy"), "result validation accuracy")


def reconcile_result_artifacts(pipeline_root: Path) -> int:
    connection = _open_database(pipeline_root / "queue.sqlite3")
    reconciled = 0
    try:
        rows = [dict(row) for row in connection.execute("SELECT * FROM tasks")]
        for task in rows:
            result_path = pipeline_root / task["result_path"]
            if not result_path.is_file():
                continue
            result = load_json(result_path)
            _validate_result_artifact(task, result)
            if task["status"] != "completed":
                connection.execute(
                    """
                    UPDATE tasks SET status='completed',completed_at=?,error_path=NULL
                    WHERE task_id=?
                    """,
                    (result["completed_at"], task["task_id"]),
                )
                reconciled += 1
        connection.commit()
    finally:
        connection.close()
    if reconciled:
        _append_event(
            pipeline_root,
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": "result_artifacts_reconciled",
                "count": reconciled,
            },
        )
        write_status_tsv(pipeline_root)
    return reconciled


def claim_task(
    pipeline_root: Path, worker_id: str, gpu_id: int
) -> dict[str, Any] | None:
    connection = _open_database(pipeline_root / "queue.sqlite3")
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM tasks WHERE status='pending' ORDER BY queue_order LIMIT 1"
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        updated = connection.execute(
            """
            UPDATE tasks SET status='running',attempts=attempts+1,claimed_by=?,
                worker_pid=?,worker_host=?,gpu_id=?,started_at=?
            WHERE task_id=? AND status='pending'
            """,
            (
                worker_id,
                os.getpid(),
                socket.gethostname(),
                gpu_id,
                timestamp,
                row["task_id"],
            ),
        )
        _require(updated.rowcount == 1, "atomic task claim failed")
        connection.commit()
        claimed = dict(row)
        claimed.update(
            {
                "status": "running",
                "claimed_by": worker_id,
                "worker_pid": os.getpid(),
                "worker_host": socket.gethostname(),
                "gpu_id": gpu_id,
                "started_at": timestamp,
                "attempts": int(row["attempts"]) + 1,
            }
        )
        return claimed
    finally:
        connection.close()


def _complete_task(pipeline_root: Path, task: Mapping[str, Any], completed_at: str) -> None:
    connection = _open_database(pipeline_root / "queue.sqlite3")
    try:
        updated = connection.execute(
            """
            UPDATE tasks SET status='completed',completed_at=?,error_path=NULL
            WHERE task_id=? AND status='running' AND worker_pid=?
            """,
            (completed_at, task["task_id"], os.getpid()),
        )
        _require(updated.rowcount == 1, "could not commit completed queue task")
        connection.commit()
    finally:
        connection.close()


def _fail_task(pipeline_root: Path, task: Mapping[str, Any], error_path: Path) -> None:
    connection = _open_database(pipeline_root / "queue.sqlite3")
    try:
        updated = connection.execute(
            """
            UPDATE tasks SET status='failed',completed_at=?,error_path=?
            WHERE task_id=? AND status='running' AND worker_pid=?
            """,
            (
                time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                str(error_path.relative_to(pipeline_root)),
                task["task_id"],
                os.getpid(),
            ),
        )
        _require(updated.rowcount == 1, "could not commit failed queue task")
        connection.commit()
    finally:
        connection.close()


def _load_unique_candidates(pipeline_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = load_csv(pipeline_root / "unique_candidate_manifest.csv")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    json_fields = (
        "architecture",
        "hyperparameters",
        "dataset_context",
        "gat_heads_by_layer",
        "sage_aggr_by_layer",
        "gin_eps_by_layer",
    )
    for row in rows:
        normalized: dict[str, Any] = dict(row)
        for field in json_fields:
            raw = row.get(field, "")
            normalized[field] = None if raw == "" else json.loads(raw)
        for field in ("lr", "dropout", "l2", "gin_eps", "gcnii_alpha", "gcnii_theta"):
            normalized[field] = float(row[field])
        for field in ("hidden_dim", "gat_heads"):
            normalized[field] = int(row[field])
        key = (row["dataset"], row["candidate_fingerprint"])
        _require(key not in result, f"duplicate unique candidate: {key}")
        result[key] = normalized
    return result


def _load_frozen_runtime(runtime_root: Path):
    sys.path.insert(0, str(runtime_root))
    from dataset_utils import (  # type: ignore
        assert_dataset_context_matches,
        load_dataset_from_request,
        resolve_dataset_request,
    )
    from evaluation_errors import InfrastructureEvaluationError  # type: ignore
    from eval_utils import train_and_eval_arch  # type: ignore
    from final_eval import final_evaluation_seed as frozen_seed  # type: ignore

    return {
        "assert_dataset_context_matches": assert_dataset_context_matches,
        "load_dataset_from_request": load_dataset_from_request,
        "resolve_dataset_request": resolve_dataset_request,
        "InfrastructureEvaluationError": InfrastructureEvaluationError,
        "train_and_eval_arch": train_and_eval_arch,
        "final_evaluation_seed": frozen_seed,
    }


def _current_task_path(pipeline_root: Path, gpu_id: int) -> Path:
    return pipeline_root / f"current_task_gpu{gpu_id}.json"


def _worker_log(log_path: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S%z')}] {message}\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _dataset_split_seed_from_context(
    dataset: str, context: Mapping[str, Any]
) -> int | None:
    raw_split_seed = context.get("split_seed")
    if dataset == "dblp":
        _require(raw_split_seed is not None, "DBLP dataset context lacks split_seed")
        return int(raw_split_seed)
    _require(
        raw_split_seed is None,
        f"non-DBLP dataset context unexpectedly defines split_seed: {dataset}",
    )
    return None


def worker(args: argparse.Namespace) -> int:
    pipeline_root = Path(args.pipeline_root).resolve()
    gpu_id = int(args.gpu_id)
    config = load_json(pipeline_root / "pipeline_config.json")
    log_path = pipeline_root / f"queue_gpu{gpu_id}.log"
    pid_path = pipeline_root / f"worker_gpu{gpu_id}.pid"
    atomic_write_text(pid_path, f"{os.getpid()}\n")
    worker_id = f"gpu{gpu_id}:{socket.gethostname()}:{os.getpid()}"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(gpu_id):
        raise PipelineAuditError(
            f"worker GPU binding mismatch: expected CUDA_VISIBLE_DEVICES={gpu_id}, got {visible!r}"
        )
    runtime = _load_frozen_runtime(Path(config["frozen_runtime_root"]))
    import torch

    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    _require(not torch.are_deterministic_algorithms_enabled(), "strict determinism is on")
    _require(torch.cuda.is_available(), "CUDA is unavailable in worker")
    _require(torch.cuda.device_count() == 1, "worker must see exactly one bound CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    candidates = _load_unique_candidates(pipeline_root)
    recover_stale_tasks(pipeline_root)
    reconcile_result_artifacts(pipeline_root)
    _append_event(
        pipeline_root,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": "worker_started",
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "pid": os.getpid(),
        },
    )
    _worker_log(log_path, f"worker started id={worker_id} physical_gpu={gpu_id}")
    bundle = None
    loaded_dataset: str | None = None
    data = None
    in_ch = out_ch = 0
    while True:
        task = claim_task(pipeline_root, worker_id, gpu_id)
        if task is None:
            break
        atomic_write_json(
            _current_task_path(pipeline_root, gpu_id),
            {
                "status": "running",
                "worker_id": worker_id,
                "pid": os.getpid(),
                **task,
            },
        )
        key = (task["dataset"], task["candidate_fingerprint"])
        candidate = candidates[key]
        started = time.monotonic()
        _worker_log(
            log_path,
            f"claim {task['task_id']} seed={task['effective_training_seed']}",
        )
        try:
            expected_seed = runtime["final_evaluation_seed"](
                int(config["final_base_seed"]),
                task["candidate_fingerprint"],
                int(task["replicate_id"]),
            )
            _require(expected_seed == task["effective_training_seed"], "worker seed mismatch")
            if loaded_dataset != task["dataset"]:
                if bundle is not None:
                    del data, bundle
                    bundle = None
                    data = None
                    gc.collect()
                    torch.cuda.empty_cache()
                context = candidate["dataset_context"]
                split_seed = _dataset_split_seed_from_context(task["dataset"], context)
                request = runtime["resolve_dataset_request"](
                    task["dataset"], config["data_root"], split_seed=split_seed
                )
                bundle = runtime["load_dataset_from_request"](request)
                runtime["assert_dataset_context_matches"](
                    context, bundle.context, context="test30 loaded dataset"
                )
                bundle.to(device)
                data = bundle.data
                in_ch = int(bundle.num_features)
                out_ch = int(bundle.num_classes)
                loaded_dataset = task["dataset"]
                _worker_log(
                    log_path,
                    f"loaded dataset={loaded_dataset} features={in_ch} classes={out_ch}",
                )
            architecture = candidate["architecture"]
            train_config = {
                "operations": list(architecture["operations"]),
                "edges": [tuple(edge) for edge in architecture["edges"]],
                "effective_layers": sum(
                    operation != "Identity" for operation in architecture["operations"]
                ),
            }
            result = runtime["train_and_eval_arch"](
                config=train_config,
                data=data,
                in_ch=in_ch,
                out_ch=out_ch,
                lr=float(candidate["lr"]),
                dropout=float(candidate["dropout"]),
                hidden_dim=int(candidate["hidden_dim"]),
                weight_decay=float(candidate["l2"]),
                gcnii_alpha=float(candidate["gcnii_alpha"]),
                gcnii_theta=float(candidate["gcnii_theta"]),
                gat_heads=int(candidate["gat_heads"]),
                sage_aggr=str(candidate["sage_aggr"]),
                gin_eps=float(candidate["gin_eps"]),
                gat_heads_by_layer=candidate["gat_heads_by_layer"],
                sage_aggr_by_layer=candidate["sage_aggr_by_layer"],
                gin_eps_by_layer=candidate["gin_eps_by_layer"],
                device=device,
                max_epochs=int(config["eval_epochs"]),
                patience=int(config["patience"]),
                seed=int(task["effective_training_seed"]),
                track_test=True,
                return_metadata=True,
            )
            _require(isinstance(result, tuple) and len(result) == 4, "training result shape")
            validation_accuracy, valid, test_accuracy, metadata = result
            _require(valid is True, "independent replicate returned valid=False")
            validation_accuracy = _finite_float(validation_accuracy, "replicate validation")
            test_accuracy = _finite_float(test_accuracy, "replicate test")
            completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            result_row = {
                "dataset": task["dataset"],
                "candidate_fingerprint": task["candidate_fingerprint"],
                "replicate_id": int(task["replicate_id"]),
                "effective_training_seed": int(task["effective_training_seed"]),
                "test_accuracy": test_accuracy,
                "validation_accuracy": validation_accuracy,
                "epochs_trained": int(metadata.get("epochs_ran", 0)),
                "best_epoch": metadata.get("best_epoch"),
                "status": "completed",
                "wall_time": float(time.monotonic() - started),
                "gpu_id": gpu_id,
                "cache_key": task["cache_key"],
                "final_evaluation_protocol_fingerprint": config[
                    "final_evaluation_protocol_fingerprint"
                ],
                "dataset_content_fingerprint": candidate["dataset_context"].get(
                    "dataset_content_fingerprint"
                ),
                "split_fingerprint": candidate["dataset_context"].get("split_fingerprint"),
                "completed_at": completed_at,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(device),
                "worker_id": worker_id,
            }
            result_path = pipeline_root / task["result_path"]
            atomic_write_json(result_path, result_row)
            _complete_task(pipeline_root, task, completed_at)
            _append_event(
                pipeline_root,
                {
                    "timestamp": completed_at,
                    "event": "replicate_completed",
                    "task_id": task["task_id"],
                    "gpu_id": gpu_id,
                    "wall_time": result_row["wall_time"],
                },
            )
            _append_progress_tsv(
                pipeline_root,
                event="replicate_completed",
                task=task,
                worker_id=worker_id,
                gpu_id=gpu_id,
                artifact_path=str(result_path),
            )
            _worker_log(
                log_path,
                f"complete {task['task_id']} test={test_accuracy:.12g} "
                f"wall={result_row['wall_time']:.3f}s",
            )
        except BaseException as exc:
            error_path = (
                pipeline_root
                / "infrastructure_errors"
                / f"{task['dataset']}_{task['candidate_fingerprint']}_"
                f"replicate_{int(task['replicate_id']):02d}_attempt_{int(task['attempts']):02d}.json"
            )
            atomic_write_json(
                error_path,
                {
                    "status": "infrastructure_error",
                    "task": task,
                    "candidate": candidate,
                    "gpu_id": gpu_id,
                    "worker_id": worker_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "wall_time": float(time.monotonic() - started),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            _fail_task(pipeline_root, task, error_path)
            _append_event(
                pipeline_root,
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "event": "replicate_failed",
                    "task_id": task["task_id"],
                    "gpu_id": gpu_id,
                    "error_path": str(error_path),
                },
            )
            _append_progress_tsv(
                pipeline_root,
                event="replicate_failed",
                task=task,
                worker_id=worker_id,
                gpu_id=gpu_id,
                artifact_path=str(error_path),
            )
            atomic_write_json(
                _current_task_path(pipeline_root, gpu_id),
                {
                    "status": "failed",
                    "worker_id": worker_id,
                    "task_id": task["task_id"],
                    "error_path": str(error_path),
                },
            )
            write_status_tsv(pipeline_root)
            _worker_log(log_path, f"FAILED {task['task_id']} error={error_path}")
            return 1
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if bundle is not None:
        del data, bundle
    gc.collect()
    torch.cuda.empty_cache()
    atomic_write_json(
        _current_task_path(pipeline_root, gpu_id),
        {
            "status": "idle_queue_exhausted",
            "worker_id": worker_id,
            "pid": os.getpid(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    _append_event(
        pipeline_root,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": "worker_queue_exhausted",
            "worker_id": worker_id,
            "gpu_id": gpu_id,
        },
    )
    _worker_log(log_path, "queue exhausted")
    write_status_tsv(pipeline_root)
    maybe_finalize(pipeline_root)
    return 0


def queue_counts(pipeline_root: Path) -> dict[str, int]:
    connection = _open_database(pipeline_root / "queue.sqlite3")
    try:
        counts = {row["status"]: int(row["count"]) for row in connection.execute(
            "SELECT status,COUNT(*) AS count FROM tasks GROUP BY status"
        )}
        total = int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
    finally:
        connection.close()
    return {status: counts.get(status, 0) for status in ("pending", "running", "completed", "failed")} | {"total": total}


def audit_dataset_storage(pipeline_root: Path) -> dict[str, Any]:
    """Verify first-formal dataset manifests and on-disk source files read-only."""

    config = load_json(pipeline_root / "pipeline_config.json")
    history_rows = load_json(pipeline_root / "history_audit.json")
    context_fields = (
        "canonical_name",
        "source_class",
        "source_version",
        "dataset_content_fingerprint",
        "split_protocol",
        "split_seed",
        "split_fingerprint",
        "graph_transform",
        "training_mode",
        "metric",
    )
    manifest_groups: dict[str, list[dict[str, Any]]] = {}
    checked_files: dict[str, dict[str, Any]] = {}
    for history_row in history_rows:
        history_path = Path(history_row["history_path"])
        manifest_path = history_path.with_name("dataset_manifest.json")
        _require(manifest_path.is_file(), f"dataset manifest is missing: {manifest_path}")
        dataset_manifest = load_json(manifest_path)
        history_context = history_row["metadata"]["dataset_context"]
        for field in context_fields:
            _require(
                dataset_manifest.get(field) == history_context.get(field),
                f"dataset manifest/history context mismatch for {field}: {manifest_path}",
            )
        dataset = str(history_row["dataset"])
        manifest_groups.setdefault(dataset, []).append(
            {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "dataset_content_fingerprint": dataset_manifest[
                    "dataset_content_fingerprint"
                ],
                "split_fingerprint": dataset_manifest["split_fingerprint"],
            }
        )
        storage_root = Path(dataset_manifest["storage_root"]).resolve()
        try:
            storage_root.relative_to(Path(config["data_root"]).resolve())
        except ValueError as exc:
            raise PipelineAuditError(
                f"dataset storage is outside formal --data-root: {storage_root}"
            ) from exc
        for source_file in dataset_manifest.get("source_files", []):
            source_path = storage_root / source_file["relative_path"]
            key = str(source_path)
            if key in checked_files:
                previous = checked_files[key]
                _require(
                    previous["expected_sha256"] == source_file["sha256"]
                    and previous["expected_size_bytes"] == int(source_file["size_bytes"]),
                    f"formal manifests disagree for source file: {source_path}",
                )
                continue
            _require(source_path.is_file(), f"dataset source file is missing: {source_path}")
            actual_size = source_path.stat().st_size
            _require(
                actual_size == int(source_file["size_bytes"]),
                f"dataset source file size mismatch: {source_path}",
            )
            actual_sha = sha256_file(source_path)
            _require(
                actual_sha == source_file["sha256"],
                f"dataset source file SHA-256 mismatch: {source_path}",
            )
            checked_files[key] = {
                "path": key,
                "expected_size_bytes": int(source_file["size_bytes"]),
                "expected_sha256": source_file["sha256"],
                "actual_size_bytes": actual_size,
                "actual_sha256": actual_sha,
            }
    _require(len(history_rows) == 60, "dataset storage audit history count is not 60")
    for dataset, rows in manifest_groups.items():
        _require(len(rows) == 15, f"dataset manifest count is not 15: {dataset}")
        _require(
            len({row["dataset_content_fingerprint"] for row in rows}) == 1,
            f"dataset content identity differs across histories: {dataset}",
        )
        _require(
            len({row["split_fingerprint"] for row in rows}) == 1,
            f"dataset split identity differs across histories: {dataset}",
        )
    report = {
        "status": "dataset_storage_audit_passed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "history_dataset_manifests": len(history_rows),
        "datasets": {
            dataset: {
                "manifest_count": len(rows),
                "dataset_content_fingerprint": rows[0]["dataset_content_fingerprint"],
                "split_fingerprint": rows[0]["split_fingerprint"],
                "manifest_paths": [row["path"] for row in rows],
            }
            for dataset, rows in sorted(manifest_groups.items())
        },
        "unique_source_files_checked": len(checked_files),
        "source_files": list(checked_files.values()),
        "data_root": config["data_root"],
        "read_only": True,
    }
    atomic_write_json(pipeline_root / "dataset_storage_audit.json", report)
    return report


def _nvidia_idle_check() -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    _require(query.returncode == 0, f"nvidia-smi failed: {query.stderr.strip()}")
    gpus: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        _require(len(parts) == 4, f"unexpected nvidia-smi row: {line!r}")
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "utilization_percent": int(parts[3]),
            }
        )
    _require(len(gpus) >= 2, "fewer than two NVIDIA GPUs are visible")
    selected = [gpu for gpu in gpus if gpu["index"] in (0, 1)]
    _require(len(selected) == 2, "GPU 0/1 are not both visible")
    for gpu in selected:
        _require(
            gpu["memory_used_mib"] <= 256 and gpu["utilization_percent"] <= 5,
            f"GPU {gpu['index']} is not idle: {gpu}",
        )
    torch_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch,sys;sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>=2 else 1)",
        ],
        capture_output=True,
        text=True,
    )
    _require(torch_check.returncode == 0, "PyTorch cannot access two CUDA GPUs")
    return selected


def _bound_gpu_ready(gpu_id: int) -> tuple[bool, str]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if query.returncode != 0:
        return False, f"nvidia-smi unavailable: {query.stderr.strip()}"
    parts = [part.strip() for part in query.stdout.strip().split(",")]
    if len(parts) != 4:
        return False, f"unexpected nvidia-smi output: {query.stdout.strip()!r}"
    memory_used = int(parts[2])
    utilization = int(parts[3])
    if memory_used > 256 or utilization > 5:
        return False, (
            f"GPU busy: memory_used_mib={memory_used} "
            f"utilization_percent={utilization}"
        )
    torch_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch,sys;sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()==1 else 1)",
        ],
        capture_output=True,
        text=True,
    )
    if torch_check.returncode != 0:
        return False, "bound PyTorch process cannot access exactly one CUDA GPU"
    return True, (
        f"GPU ready: index={parts[0]} name={parts[1]} "
        f"memory_used_mib={memory_used} utilization_percent={utilization}"
    )


def guarded_worker(args: argparse.Namespace) -> int:
    """Wait without claiming work until a bound idle CUDA device is usable."""

    pipeline_root = Path(args.pipeline_root).resolve()
    gpu_id = int(args.gpu_id)
    poll_seconds = int(args.poll_seconds)
    _require(5 <= poll_seconds <= 60, "guarded worker poll interval must be 5..60s")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _require(
        visible == str(gpu_id),
        f"guarded worker expected CUDA_VISIBLE_DEVICES={gpu_id}, got {visible!r}",
    )
    pid_path = pipeline_root / f"worker_gpu{gpu_id}.pid"
    log_path = pipeline_root / f"queue_gpu{gpu_id}.log"
    atomic_write_text(pid_path, f"{os.getpid()}\n")
    waiter_id = f"guarded-gpu{gpu_id}:{socket.gethostname()}:{os.getpid()}"
    checks = 0
    previous_reason: str | None = None
    _append_event(
        pipeline_root,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": "guarded_worker_started",
            "worker_id": waiter_id,
            "gpu_id": gpu_id,
            "pid": os.getpid(),
        },
    )
    while True:
        ready, reason = _bound_gpu_ready(gpu_id)
        checks += 1
        atomic_write_json(
            _current_task_path(pipeline_root, gpu_id),
            {
                "status": "waiting_for_cuda" if not ready else "cuda_gate_passed",
                "worker_id": waiter_id,
                "pid": os.getpid(),
                "gpu_id": gpu_id,
                "checks": checks,
                "last_check": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "reason": reason,
                "poll_seconds": poll_seconds,
            },
        )
        if ready:
            _worker_log(log_path, f"CUDA gate passed after {checks} checks: {reason}")
            _append_event(
                pipeline_root,
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "event": "guarded_worker_cuda_gate_passed",
                    "worker_id": waiter_id,
                    "gpu_id": gpu_id,
                    "checks": checks,
                    "reason": reason,
                },
            )
            return worker(args)
        if reason != previous_reason or checks % 20 == 0:
            _worker_log(
                log_path,
                f"waiting for idle CUDA physical_gpu={gpu_id} check={checks}: {reason}",
            )
            previous_reason = reason
        time.sleep(poll_seconds)


def launch_workers(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_root = Path(args.pipeline_root).resolve()
    config = load_json(pipeline_root / "pipeline_config.json")
    manifest_audit = load_json(pipeline_root / "manifest_audit_report.json")
    _require(manifest_audit.get("status") == "manifest_audit_passed", "manifest gate")
    storage_audit = audit_dataset_storage(pipeline_root)
    _require(
        storage_audit.get("status") == "dataset_storage_audit_passed",
        "dataset storage gate",
    )
    reconcile_result_artifacts(pipeline_root)
    recover_stale_tasks(pipeline_root)
    counts = queue_counts(pipeline_root)
    _require(counts["failed"] == 0, "failed tasks must be reset explicitly before launch")
    _require(counts["pending"] > 0, "there are no pending tasks to launch")
    gpus = _nvidia_idle_check()
    _require(shutil_which("tmux") is not None, "tmux is not installed")
    session = str(config["tmux_session"])
    tmux_socket = str(
        Path("/tmp") / f"dvae_test30_{hashlib.sha256(session.encode()).hexdigest()[:16]}.sock"
    )
    tmux = ["tmux", "-S", tmux_socket]
    existing = subprocess.run(
        [*tmux, "has-session", "-t", session], capture_output=True
    )
    _require(existing.returncode != 0, f"tmux session already exists: {session}")
    script = str(Path(__file__).resolve())
    commands: list[list[str]] = []
    for gpu_id in (0, 1):
        commands.append(
            [
                "env",
                f"CUDA_VISIBLE_DEVICES={gpu_id}",
                sys.executable,
                script,
                "worker",
                "--pipeline-root",
                str(pipeline_root),
                "--gpu-id",
                str(gpu_id),
            ]
        )
    subprocess.run(
        [*tmux, "new-session", "-d", "-s", session, "-n", "gpu0", shlex.join(commands[0])],
        check=True,
    )
    subprocess.run(
        [*tmux, "new-window", "-d", "-t", session, "-n", "gpu1", shlex.join(commands[1])],
        check=True,
    )
    subprocess.run(
        [*tmux, "set-window-option", "-t", session, "remain-on-exit", "on"],
        check=True,
    )
    deadline = time.monotonic() + 15.0
    pid_paths = [pipeline_root / f"worker_gpu{gpu_id}.pid" for gpu_id in (0, 1)]
    while time.monotonic() < deadline and not all(path.is_file() for path in pid_paths):
        time.sleep(0.1)
    _require(all(path.is_file() for path in pid_paths), "workers did not write PID files")
    pids = [int(path.read_text(encoding="utf-8").strip()) for path in pid_paths]
    _require(all(_process_alive(pid, socket.gethostname()) for pid in pids), "worker exited")
    launch = {
        "status": "workers_started",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pipeline_root": str(pipeline_root),
        "tmux_session": session,
        "tmux_socket": tmux_socket,
        "gpus": gpus,
        "worker_pids": {"gpu0": pids[0], "gpu1": pids[1]},
        "queue_logs": {
            "gpu0": str(pipeline_root / "queue_gpu0.log"),
            "gpu1": str(pipeline_root / "queue_gpu1.log"),
        },
        "current_task_files": {
            "gpu0": str(_current_task_path(pipeline_root, 0)),
            "gpu1": str(_current_task_path(pipeline_root, 1)),
        },
    }
    atomic_write_json(pipeline_root / "launch_report.json", launch)
    _append_event(pipeline_root, {"event": "workers_launched", **launch})
    return launch


def launch_guarded_workers(args: argparse.Namespace) -> dict[str, Any]:
    """Start two tmux runners that enter workers only after CUDA is ready."""

    pipeline_root = Path(args.pipeline_root).resolve()
    config = load_json(pipeline_root / "pipeline_config.json")
    manifest_audit = load_json(pipeline_root / "manifest_audit_report.json")
    _require(manifest_audit.get("status") == "manifest_audit_passed", "manifest gate")
    storage_audit = audit_dataset_storage(pipeline_root)
    _require(storage_audit.get("status") == "dataset_storage_audit_passed", "storage gate")
    reconcile_result_artifacts(pipeline_root)
    recover_stale_tasks(pipeline_root)
    counts = queue_counts(pipeline_root)
    _require(counts["failed"] == 0, "failed tasks must be reset before guarded launch")
    _require(counts["pending"] > 0, "there are no pending tasks for guarded launch")
    _require(shutil_which("tmux") is not None, "tmux is not installed")
    session = str(config["tmux_session"])
    tmux_socket = str(
        Path("/tmp") / f"dvae_test30_{hashlib.sha256(session.encode()).hexdigest()[:16]}.sock"
    )
    tmux = ["tmux", "-S", tmux_socket]
    existing = subprocess.run(
        [*tmux, "has-session", "-t", session], capture_output=True
    )
    _require(existing.returncode != 0, f"tmux session already exists: {session}")
    script = str(Path(__file__).resolve())
    commands: list[list[str]] = []
    for gpu_id in (0, 1):
        commands.append(
            [
                "env",
                f"CUDA_VISIBLE_DEVICES={gpu_id}",
                sys.executable,
                script,
                "guarded-worker",
                "--pipeline-root",
                str(pipeline_root),
                "--gpu-id",
                str(gpu_id),
                "--poll-seconds",
                str(args.poll_seconds),
            ]
        )
    subprocess.run(
        [
            *tmux,
            "new-session",
            "-d",
            "-s",
            session,
            "-n",
            "gpu0",
            shlex.join(commands[0]),
        ],
        check=True,
    )
    subprocess.run(
        [
            *tmux,
            "new-window",
            "-d",
            "-t",
            session,
            "-n",
            "gpu1",
            shlex.join(commands[1]),
        ],
        check=True,
    )
    subprocess.run(
        [*tmux, "set-window-option", "-t", session, "remain-on-exit", "on"],
        check=True,
    )
    deadline = time.monotonic() + 15.0
    pid_paths = [pipeline_root / f"worker_gpu{gpu_id}.pid" for gpu_id in (0, 1)]
    current_paths = [_current_task_path(pipeline_root, gpu_id) for gpu_id in (0, 1)]
    while time.monotonic() < deadline and not (
        all(path.is_file() for path in pid_paths)
        and all(path.is_file() for path in current_paths)
    ):
        time.sleep(0.1)
    _require(all(path.is_file() for path in pid_paths), "guarded runners lack PID files")
    _require(
        all(path.is_file() for path in current_paths),
        "guarded runners lack current-task files",
    )
    pids = [int(path.read_text(encoding="utf-8").strip()) for path in pid_paths]
    _require(all(_process_alive(pid, socket.gethostname()) for pid in pids), "runner exited")
    report = {
        "status": "guarded_workers_started_waiting_for_cuda",
        "gpu_training_started": False,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pipeline_root": str(pipeline_root),
        "tmux_session": session,
        "tmux_socket": tmux_socket,
        "poll_seconds": int(args.poll_seconds),
        "worker_pids": {"gpu0": pids[0], "gpu1": pids[1]},
        "queue_logs": {
            "gpu0": str(pipeline_root / "queue_gpu0.log"),
            "gpu1": str(pipeline_root / "queue_gpu1.log"),
        },
        "current_task_files": {
            "gpu0": str(current_paths[0]),
            "gpu1": str(current_paths[1]),
        },
        "queue": counts,
    }
    atomic_write_json(pipeline_root / "guarded_launch_report.json", report)
    _append_event(pipeline_root, {"event": "guarded_workers_launched", **report})
    return report


def launch_background_guarded_workers(args: argparse.Namespace) -> dict[str, Any]:
    """Fallback detached runner for environments where tmux sockets are denied."""

    pipeline_root = Path(args.pipeline_root).resolve()
    config = load_json(pipeline_root / "pipeline_config.json")
    manifest_audit = load_json(pipeline_root / "manifest_audit_report.json")
    _require(manifest_audit.get("status") == "manifest_audit_passed", "manifest gate")
    storage_audit = audit_dataset_storage(pipeline_root)
    _require(storage_audit.get("status") == "dataset_storage_audit_passed", "storage gate")
    reconcile_result_artifacts(pipeline_root)
    recover_stale_tasks(pipeline_root)
    counts = queue_counts(pipeline_root)
    _require(counts["failed"] == 0, "failed tasks must be reset before background launch")
    _require(counts["pending"] > 0, "there are no pending tasks for background launch")
    for gpu_id in (0, 1):
        pid_path = pipeline_root / f"worker_gpu{gpu_id}.pid"
        if pid_path.exists():
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            _require(
                not _process_alive(pid, socket.gethostname()),
                f"GPU {gpu_id} runner is already alive with PID {pid}",
            )
    script = str(Path(__file__).resolve())
    processes: list[subprocess.Popen[Any]] = []
    handles = []
    try:
        for gpu_id in (0, 1):
            command = [
                sys.executable,
                script,
                "guarded-worker",
                "--pipeline-root",
                str(pipeline_root),
                "--gpu-id",
                str(gpu_id),
                "--poll-seconds",
                str(args.poll_seconds),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            bootstrap_log = pipeline_root / f"runner_gpu{gpu_id}.stdout.log"
            handle = bootstrap_log.open("a", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            processes.append(process)
        deadline = time.monotonic() + 15.0
        pid_paths = [pipeline_root / f"worker_gpu{gpu_id}.pid" for gpu_id in (0, 1)]
        current_paths = [_current_task_path(pipeline_root, gpu_id) for gpu_id in (0, 1)]
        while time.monotonic() < deadline and not (
            all(path.is_file() for path in pid_paths)
            and all(path.is_file() for path in current_paths)
        ):
            time.sleep(0.1)
        _require(all(path.is_file() for path in pid_paths), "background runners lack PID files")
        _require(
            all(path.is_file() for path in current_paths),
            "background runners lack current-task files",
        )
        pids = [int(path.read_text(encoding="utf-8").strip()) for path in pid_paths]
        _require(pids == [process.pid for process in processes], "background PID mismatch")
        _require(
            all(_process_alive(pid, socket.gethostname()) for pid in pids),
            "background runner exited during launch",
        )
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for handle in handles:
            handle.close()
    report = {
        "status": "background_guarded_workers_started_waiting_for_cuda",
        "gpu_training_started": False,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pipeline_root": str(pipeline_root),
        "launch_mode": "detached_start_new_session",
        "tmux_unavailable_reason": "tmux socket operations denied with EPERM",
        "poll_seconds": int(args.poll_seconds),
        "worker_pids": {"gpu0": pids[0], "gpu1": pids[1]},
        "queue_logs": {
            "gpu0": str(pipeline_root / "queue_gpu0.log"),
            "gpu1": str(pipeline_root / "queue_gpu1.log"),
        },
        "bootstrap_logs": {
            "gpu0": str(pipeline_root / "runner_gpu0.stdout.log"),
            "gpu1": str(pipeline_root / "runner_gpu1.stdout.log"),
        },
        "current_task_files": {
            "gpu0": str(_current_task_path(pipeline_root, 0)),
            "gpu1": str(_current_task_path(pipeline_root, 1)),
        },
        "queue": counts,
    }
    atomic_write_json(pipeline_root / "background_launch_report.json", report)
    _append_event(pipeline_root, {"event": "background_guarded_workers_launched", **report})
    return report


def supervise_guarded_workers(args: argparse.Namespace) -> int:
    """Keep both guarded runners attached to one long-lived supervisor."""

    pipeline_root = Path(args.pipeline_root).resolve()
    manifest_audit = load_json(pipeline_root / "manifest_audit_report.json")
    _require(manifest_audit.get("status") == "manifest_audit_passed", "manifest gate")
    storage_audit = audit_dataset_storage(pipeline_root)
    _require(storage_audit.get("status") == "dataset_storage_audit_passed", "storage gate")
    reconcile_result_artifacts(pipeline_root)
    recover_stale_tasks(pipeline_root)
    counts = queue_counts(pipeline_root)
    _require(counts["failed"] == 0, "failed tasks must be reset before supervision")
    _require(counts["pending"] > 0, "there are no pending tasks for supervision")
    atomic_write_text(pipeline_root / "supervisor.pid", f"{os.getpid()}\n")
    script = str(Path(__file__).resolve())
    processes: list[subprocess.Popen[Any]] = []
    handles = []
    try:
        for gpu_id in (0, 1):
            command = [
                sys.executable,
                script,
                "guarded-worker",
                "--pipeline-root",
                str(pipeline_root),
                "--gpu-id",
                str(gpu_id),
                "--poll-seconds",
                str(args.poll_seconds),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            bootstrap_log = pipeline_root / f"runner_gpu{gpu_id}.stdout.log"
            handle = bootstrap_log.open("a", encoding="utf-8")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                )
            )
        report = {
            "status": "supervisor_running_guarded_workers_waiting_for_cuda",
            "gpu_training_started": False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pipeline_root": str(pipeline_root),
            "supervisor_pid": os.getpid(),
            "poll_seconds": int(args.poll_seconds),
            "worker_pids": {"gpu0": processes[0].pid, "gpu1": processes[1].pid},
            "queue_logs": {
                "gpu0": str(pipeline_root / "queue_gpu0.log"),
                "gpu1": str(pipeline_root / "queue_gpu1.log"),
            },
            "current_task_files": {
                "gpu0": str(_current_task_path(pipeline_root, 0)),
                "gpu1": str(_current_task_path(pipeline_root, 1)),
            },
            "queue": counts,
        }
        atomic_write_json(pipeline_root / "supervisor_report.json", report)
        _append_event(pipeline_root, {"event": "guarded_supervisor_started", **report})
        print(json.dumps(report, sort_keys=True), flush=True)
        while any(process.poll() is None for process in processes):
            time.sleep(5)
        return 0 if all(process.returncode == 0 for process in processes) else 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for handle in handles:
            handle.close()


def shutil_which(command: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def retry_failed(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_root = Path(args.pipeline_root).resolve()
    connection = _open_database(pipeline_root / "queue.sqlite3")
    try:
        connection.execute("BEGIN IMMEDIATE")
        failed = [dict(row) for row in connection.execute("SELECT * FROM tasks WHERE status='failed'")]
        for row in failed:
            connection.execute(
                """
                UPDATE tasks SET status='pending',claimed_by=NULL,worker_pid=NULL,
                    worker_host=NULL,gpu_id=NULL,started_at=NULL,completed_at=NULL,
                    error_path=NULL WHERE task_id=?
                """,
                (row["task_id"],),
            )
        connection.commit()
    finally:
        connection.close()
    for row in failed:
        _append_event(
            pipeline_root,
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": "failed_task_reset_for_retry",
                "task_id": row["task_id"],
                "previous_error_path": row["error_path"],
            },
        )
    write_status_tsv(pipeline_root)
    return {"reset_failed_tasks": len(failed), **queue_counts(pipeline_root)}


def _read_events(pipeline_root: Path) -> list[dict[str, Any]]:
    path = pipeline_root / "queue_events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def _load_completed_results(pipeline_root: Path) -> list[dict[str, Any]]:
    connection = _open_database(pipeline_root / "queue.sqlite3")
    try:
        tasks = [dict(row) for row in connection.execute("SELECT * FROM tasks ORDER BY queue_order")]
    finally:
        connection.close()
    results: list[dict[str, Any]] = []
    for task in tasks:
        _require(task["status"] == "completed", f"task is not completed: {task['task_id']}")
        path = pipeline_root / task["result_path"]
        _require(path.is_file(), f"completed result artifact is missing: {path}")
        result = load_json(path)
        _validate_result_artifact(task, result)
        results.append(result)
    return results


def _parse_origin_csv(pipeline_root: Path) -> list[dict[str, Any]]:
    rows = load_csv(pipeline_root / "candidate_origin_map.csv")
    parsed: list[dict[str, Any]] = []
    json_fields = (
        "architecture",
        "hyperparameters",
        "dataset_context",
        "canonical_z_search",
        "gat_heads_by_layer",
        "sage_aggr_by_layer",
        "gin_eps_by_layer",
    )
    int_fields = (
        "search_seed",
        "validation_top10_rank",
        "within_stage_full_position",
        "global_full_evaluation_index",
        "global_full_evaluation_position",
        "hidden_dim",
        "gat_heads",
    )
    float_fields = (
        "search_validation_accuracy",
        "lr",
        "dropout",
        "l2",
        "gin_eps",
        "gcnii_alpha",
        "gcnii_theta",
    )
    for row in rows:
        item: dict[str, Any] = dict(row)
        for field in json_fields:
            raw = row.get(field, "")
            item[field] = None if raw == "" else json.loads(raw)
        for field in int_fields:
            item[field] = int(row[field])
        for field in float_fields:
            item[field] = float(row[field])
        for field in ("low_fidelity_position", "promotion_position"):
            item[field] = None if row.get(field, "") == "" else int(row[field])
        parsed.append(item)
    return parsed


def _candidate_stats(
    test_values: Sequence[float],
    final_validation_values: Sequence[float] | None = None,
) -> dict[str, float]:
    _require(len(test_values) == 30, "candidate does not have 30 test values")
    test_array = np.asarray(test_values, dtype=np.float64)
    _require(bool(np.isfinite(test_array).all()), "candidate test accuracy contains NaN/Inf")
    test_mean = float(np.mean(test_array))
    test_sample_std = float(np.std(test_array, ddof=1))
    test_ci95_margin = STUDENT_T_975_DF29 * test_sample_std / math.sqrt(len(test_array))
    stats = {
        "test_mean": test_mean,
        "test_sample_std": test_sample_std,
        "test_mean_ci95_low": test_mean - test_ci95_margin,
        "test_mean_ci95_high": test_mean + test_ci95_margin,
        "test_median": float(np.median(test_array)),
        "test_min": float(np.min(test_array)),
        "test_max": float(np.max(test_array)),
    }
    if final_validation_values is None:
        return stats
    _require(
        len(final_validation_values) == 30,
        "candidate does not have 30 final-validation values",
    )
    validation_array = np.asarray(final_validation_values, dtype=np.float64)
    _require(
        bool(np.isfinite(validation_array).all()),
        "candidate final-validation accuracy contains NaN/Inf",
    )
    stats.update(
        {
            "final_validation_mean": float(np.mean(validation_array)),
            "final_validation_sample_std": float(np.std(validation_array, ddof=1)),
            "final_validation_median": float(np.median(validation_array)),
            "final_validation_min": float(np.min(validation_array)),
            "final_validation_max": float(np.max(validation_array)),
        }
    )
    return stats


def select_method_winner(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select only by final validation; test metrics never enter the ordering."""

    _require(bool(candidates), "method has no candidate summaries")
    return min(
        candidates,
        key=lambda row: (
            -float(row["final_validation_mean"]),
            float(row["final_validation_sample_std"]),
            -float(row["search_validation_accuracy"]),
            str(row["candidate_fingerprint"]),
        ),
    )


def _build_detailed_winner_rows(
    winners: Sequence[Mapping[str, Any]],
    result_groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Build one self-contained row per dataset/method winner."""

    detailed_rows: list[dict[str, Any]] = []
    for winner in winners:
        key = (str(winner["dataset"]), str(winner["candidate_fingerprint"]))
        _require(key in result_groups, f"winner has no result group: {key}")
        replicate_rows = sorted(
            result_groups[key], key=lambda row: int(row["replicate_id"])
        )
        replicate_ids = [int(row["replicate_id"]) for row in replicate_rows]
        _require(
            replicate_ids == list(REPLICATE_IDS),
            f"winner replicate IDs are not exactly 0..29: {key}",
        )
        detailed = {
            "dataset": winner["dataset"],
            "method": winner["method"],
            "candidate_fingerprint": winner["candidate_fingerprint"],
            "architecture": winner["architecture"],
            "hyperparameters": winner["hyperparameters"],
            "lr": winner["lr"],
            "dropout": winner["dropout"],
            "hidden_dim": winner["hidden_dim"],
            "l2": winner["l2"],
            "final_validation_mean": winner["final_validation_mean"],
            "final_validation_sample_std": winner[
                "final_validation_sample_std"
            ],
            "test_mean": winner["test_mean"],
            "test_sample_std": winner["test_sample_std"],
            "test_mean_ci95_low": winner["test_mean_ci95_low"],
            "test_mean_ci95_high": winner["test_mean_ci95_high"],
            "search_seed": winner["search_seed"],
            "search_stage": winner["search_stage"],
            "global_evaluation_order": winner[
                "global_full_evaluation_position"
            ],
            "stage_local_order": winner["within_stage_full_position"],
            "validation_top10_rank": winner["validation_top10_rank"],
        }
        detailed.update(
            {
                f"test_accuracy_{int(row['replicate_id'])}": float(
                    row["test_accuracy"]
                )
                for row in replicate_rows
            }
        )
        detailed_rows.append(detailed)
    return detailed_rows


def finalize_pipeline(pipeline_root: Path) -> dict[str, Any]:
    config = load_json(pipeline_root / "pipeline_config.json")
    reconcile_result_artifacts(pipeline_root)
    counts = queue_counts(pipeline_root)
    _require(counts["pending"] == 0, "cannot finalize with pending replicates")
    _require(counts["running"] == 0, "cannot finalize with running replicates")
    _require(counts["failed"] == 0, "cannot finalize with failed replicates")
    _require(counts["completed"] == counts["total"], "completion count mismatch")
    origins = _parse_origin_csv(pipeline_root)
    _require(len(origins) == EXPECTED_SOURCE_SLOTS, "final source slot count")
    results = _load_completed_results(pipeline_root)
    _require(len(results) == counts["total"], "result artifact count mismatch")
    result_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_replicates: set[tuple[str, str, int]] = set()
    for result in results:
        key3 = (
            result["dataset"],
            result["candidate_fingerprint"],
            int(result["replicate_id"]),
        )
        _require(key3 not in seen_replicates, f"duplicate replicate: {key3}")
        seen_replicates.add(key3)
        result_groups.setdefault(key3[:2], []).append(result)
    expected_unique = int(config["unique_candidate_count"])
    _require(len(result_groups) == expected_unique, "unique result candidate count mismatch")
    for key, rows in result_groups.items():
        _require(
            {int(row["replicate_id"]) for row in rows} == set(REPLICATE_IDS),
            f"replicate IDs are incomplete for {key}",
        )

    origin_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for origin in origins:
        origin_groups.setdefault(
            (origin["dataset"], origin["candidate_fingerprint"]), []
        ).append(origin)
    _require(set(origin_groups) == set(result_groups), "origin/result candidate keys differ")
    candidate_summaries: list[dict[str, Any]] = []
    for key in sorted(result_groups):
        replicate_rows = sorted(result_groups[key], key=lambda row: int(row["replicate_id"]))
        test_values = [float(row["test_accuracy"]) for row in replicate_rows]
        final_validation_values = [
            float(row["validation_accuracy"]) for row in replicate_rows
        ]
        stats = _candidate_stats(test_values, final_validation_values)
        group_origins = origin_groups[key]
        representative = representative_origin(group_origins)
        summary = {
            "dataset": key[0],
            "candidate_fingerprint": key[1],
            "architecture": representative["architecture"],
            "hyperparameters": representative["hyperparameters"],
            "lr": representative["lr"],
            "dropout": representative["dropout"],
            "hidden_dim": representative["hidden_dim"],
            "l2": representative["l2"],
            **stats,
            "n_test_seeds": len(test_values),
            "representative_method": representative["method"],
            "representative_search_seed": representative["search_seed"],
            "search_stage": representative["search_stage"],
            "within_stage_full_position": representative["within_stage_full_position"],
            "global_full_evaluation_index": representative[
                "global_full_evaluation_index"
            ],
            "global_full_evaluation_position": representative[
                "global_full_evaluation_position"
            ],
            "validation_top10_rank": representative["validation_top10_rank"],
            "search_validation_accuracy": representative[
                "search_validation_accuracy"
            ],
            "all_origins": [_compact_origin(row) for row in group_origins],
            "test_accuracies": test_values,
            "final_validation_accuracies": final_validation_values,
            "effective_training_seeds": [
                int(row["effective_training_seed"]) for row in replicate_rows
            ],
        }
        candidate_summaries.append(summary)

    summary_by_key = {
        (row["dataset"], row["candidate_fingerprint"]): row
        for row in candidate_summaries
    }
    winners: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            fingerprints = {
                origin["candidate_fingerprint"]
                for origin in origins
                if origin["dataset"] == dataset and origin["method"] == method
            }
            candidates = [summary_by_key[(dataset, fingerprint)] for fingerprint in fingerprints]
            chosen = dict(select_method_winner(candidates))
            chosen_key = (dataset, str(chosen["candidate_fingerprint"]))
            method_origin = representative_method_origin(
                origin_groups[chosen_key], method
            )
            chosen["method"] = method
            chosen["representative_method"] = method
            chosen["representative_search_seed"] = method_origin["search_seed"]
            chosen["search_seed"] = method_origin["search_seed"]
            chosen["search_stage"] = method_origin["search_stage"]
            chosen["within_stage_full_position"] = method_origin[
                "within_stage_full_position"
            ]
            chosen["global_full_evaluation_index"] = method_origin[
                "global_full_evaluation_index"
            ]
            chosen["global_full_evaluation_position"] = method_origin[
                "global_full_evaluation_position"
            ]
            chosen["validation_top10_rank"] = method_origin[
                "validation_top10_rank"
            ]
            chosen["search_validation_accuracy"] = method_origin[
                "search_validation_accuracy"
            ]
            winners.append(chosen)
    _require(len(winners) == 12, "winner count is not 12")

    long_rows = sorted(
        results,
        key=lambda row: (
            DATASETS.index(row["dataset"]),
            row["candidate_fingerprint"],
            int(row["replicate_id"]),
        ),
    )
    candidate_fields = list(candidate_summaries[0])
    winner_fields = (
        "dataset",
        "method",
        "candidate_fingerprint",
        "architecture",
        "hyperparameters",
        "lr",
        "dropout",
        "hidden_dim",
        "l2",
        "final_validation_mean",
        "final_validation_sample_std",
        "final_validation_median",
        "final_validation_min",
        "final_validation_max",
        "test_mean",
        "test_sample_std",
        "test_median",
        "test_min",
        "test_max",
        "n_test_seeds",
        "representative_method",
        "representative_search_seed",
        "search_stage",
        "within_stage_full_position",
        "global_full_evaluation_index",
        "global_full_evaluation_position",
        "validation_top10_rank",
        "search_validation_accuracy",
        "all_origins",
    )
    winner_by_seed: list[dict[str, Any]] = []
    winner_wide: list[dict[str, Any]] = []
    for winner in winners:
        rows = sorted(
            result_groups[(winner["dataset"], winner["candidate_fingerprint"])],
            key=lambda row: int(row["replicate_id"]),
        )
        for row in rows:
            winner_by_seed.append(
                {
                    "dataset": winner["dataset"],
                    "method": winner["method"],
                    "candidate_fingerprint": winner["candidate_fingerprint"],
                    "replicate_id": int(row["replicate_id"]),
                    "effective_training_seed": int(row["effective_training_seed"]),
                    "test_accuracy": float(row["test_accuracy"]),
                }
            )
        wide = {
            "dataset": winner["dataset"],
            "method": winner["method"],
            "candidate_fingerprint": winner["candidate_fingerprint"],
        }
        wide.update({f"seed_{int(row['replicate_id'])}": row["test_accuracy"] for row in rows})
        winner_wide.append(wide)
    _require(len(winner_by_seed) == 360, "winner by-seed row count is not 360")
    detailed_winners = _build_detailed_winner_rows(winners, result_groups)
    _require(len(detailed_winners) == 12, "detailed winner count is not 12")

    atomic_write_csv(pipeline_root / "test30_long.csv", long_rows, LONG_FIELDS)
    atomic_write_csv(
        pipeline_root / "test30_candidate_summary.csv",
        candidate_summaries,
        candidate_fields,
    )
    atomic_write_csv(
        pipeline_root / "test30_method_winners.csv", winners, winner_fields
    )
    atomic_write_csv(
        pipeline_root / "winner_test30_by_seed.csv",
        winner_by_seed,
        (
            "dataset",
            "method",
            "candidate_fingerprint",
            "replicate_id",
            "effective_training_seed",
            "test_accuracy",
        ),
    )
    atomic_write_csv(
        pipeline_root / "winner_test30_wide.csv",
        winner_wide,
        ("dataset", "method", "candidate_fingerprint", *(f"seed_{i}" for i in REPLICATE_IDS)),
    )
    atomic_write_csv(
        pipeline_root / "test30_method_winners_detailed.csv",
        detailed_winners,
        DETAILED_WINNER_FIELDS,
    )

    shared = [
        (key, group)
        for key, group in origin_groups.items()
        if len({origin["method"] for origin in group}) > 1
    ]
    events = _read_events(pipeline_root)
    error_artifacts = sorted(
        str(path) for path in (pipeline_root / "infrastructure_errors").glob("*.json")
    ) if (pipeline_root / "infrastructure_errors").exists() else []
    completed_wall = sum(float(row["wall_time"]) for row in results)
    gpu_wall = {
        str(gpu_id): sum(
            float(row["wall_time"]) for row in results if int(row["gpu_id"]) == gpu_id
        )
        for gpu_id in (0, 1)
    }
    audit_report = {
        "status": "complete",
        "analysis_label": ANALYSIS_LABEL,
        "winner_selection": (
            "highest arithmetic mean of 30 final-validation accuracies; "
            "test metrics were report-only"
        ),
        "test_used_for_selection": False,
        "source_id": FROZEN_SOURCE_ID,
        "excluded_deterministic_root_used": False,
        "search_histories": 60,
        "source_slots": len(origins),
        "top10_per_history": True,
        "dataset_method_groups": 12,
        "unique_dataset_candidates": len(result_groups),
        "replicate_tasks_expected": counts["total"],
        "replicate_tasks_completed": counts["completed"],
        "replicate_ids_exact_0_29": True,
        "missing_replicates": 0,
        "duplicate_replicates": 0,
        "current_failed_tasks": 0,
        "current_infrastructure_errors": 0,
        "nan_accuracy_count": 0,
        "cross_method_shared_candidates": len(shared),
        "cross_method_seed_and_accuracy_records_reused_identically": True,
        "theoretical_training_limit": EXPECTED_SOURCE_SLOTS * 30,
        "actual_completed_trainings": len(results),
        "dedup_training_savings": EXPECTED_SOURCE_SLOTS * 30 - len(results),
        "total_gpu_training_wall_time_seconds_sum": completed_wall,
        "gpu_training_wall_time_seconds_sum_by_gpu": gpu_wall,
        "historical_error_artifacts": error_artifacts,
        "failure_event_count": sum(event.get("event") == "replicate_failed" for event in events),
        "recovery_event_count": sum(
            event.get("event") in {"stale_task_recovered", "failed_task_reset_for_retry", "result_artifacts_reconciled"}
            for event in events
        ),
        "winner_rows": len(winners),
        "detailed_winner_rows": len(detailed_winners),
        "winner_by_seed_rows": len(winner_by_seed),
        "final_evaluation_protocol_fingerprint": config[
            "final_evaluation_protocol_fingerprint"
        ],
    }
    atomic_write_json(pipeline_root / "audit_report.json", audit_report)
    atomic_write_text(
        pipeline_root / "report.md",
        _render_report(
            winners=winners,
            winner_by_seed=winner_by_seed,
            origin_groups=origin_groups,
            audit=audit_report,
        ),
    )
    return audit_report


def _render_report(
    *,
    winners: Sequence[Mapping[str, Any]],
    winner_by_seed: Sequence[Mapping[str, Any]],
    origin_groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    audit: Mapping[str, Any],
) -> str:
    lines = [
        "# First-formal validation Top-10 test30 results",
        "",
        f"**Analysis label:** `{ANALYSIS_LABEL}`",
        "",
        "> Winners below were selected by the arithmetic mean of 30 independent "
        "final-validation replicates. Test metrics were used only to report the "
        "selected winner.",
        "",
        "> Test mean 95% CIs are two-sided Student-t intervals with 29 degrees "
        "of freedom.",
        "",
    ]
    by_seed = {
        (row["dataset"], row["method"], int(row["replicate_id"])): row
        for row in winner_by_seed
    }
    for dataset in DATASETS:
        lines.extend(
            [
                f"## {DATASET_DISPLAY[dataset]}",
                "",
                "| Method | Final-validation mean ± std | Test mean ± std | Test mean 95% CI | Architecture | HP | Search seed | Stage | Stage position | Global position | Top-10 rank | Search val | Fingerprint |",
                "| ------ | --------------------------: | --------------: | ----------------: | ------------ | -- | ----------: | ----- | -------------: | --------------: | ----------: | ---------: | ----------- |",
            ]
        )
        dataset_winners = [row for row in winners if row["dataset"] == dataset]
        for winner in dataset_winners:
            architecture = canonical_json(winner["architecture"])
            hp = (
                f"lr={winner['lr']:.12g}; dropout={winner['dropout']:.12g}; "
                f"hidden={winner['hidden_dim']}; l2={winner['l2']:.12g}"
            )
            lines.append(
                f"| {winner['method']} | {winner['final_validation_mean']:.12g} ± "
                f"{winner['final_validation_sample_std']:.12g} | "
                f"{winner['test_mean']:.12g} ± "
                f"{winner['test_sample_std']:.12g} | "
                f"[{winner['test_mean_ci95_low']:.12g}, "
                f"{winner['test_mean_ci95_high']:.12g}] | "
                f"`{architecture}` | {hp} | "
                f"{winner['representative_search_seed']} | {winner['search_stage']} | "
                f"{winner['within_stage_full_position']} | "
                f"{winner['global_full_evaluation_position']} | "
                f"{winner['validation_top10_rank']} | "
                f"{winner['search_validation_accuracy']:.12g} | "
                f"`{winner['candidate_fingerprint']}` |"
            )
        lines.append("")
        for winner in dataset_winners:
            lines.extend(
                [
                    f"### {winner['method']} winner test accuracies, replicates 0–29",
                    "",
                    f"Fingerprint: `{winner['candidate_fingerprint']}`",
                    "",
                ]
            )
            for replicate_id in REPLICATE_IDS:
                row = by_seed[(dataset, winner["method"], replicate_id)]
                lines.append(
                    f"replicate {replicate_id}: {float(row['test_accuracy']):.17g} "
                    f"(effective training seed {row['effective_training_seed']})"
                )
            all_origins = origin_groups[(dataset, winner["candidate_fingerprint"])]
            if len(all_origins) > 1:
                lines.extend(
                    [
                        "",
                        "All origins:",
                        "",
                        "| Method | Search seed | Stage | Stage position | Global position | Validation Top-10 rank | Search validation |",
                        "| ------ | ----------: | ----- | -------------: | --------------: | ---------------------: | ----------------: |",
                    ]
                )
                for origin in sorted(
                    all_origins,
                    key=lambda row: (
                        METHODS.index(str(row["method"])),
                        int(row["search_seed"]),
                    ),
                ):
                    lines.append(
                        f"| {origin['method']} | {origin['search_seed']} | "
                        f"{origin['search_stage']} | {origin['within_stage_full_position']} | "
                        f"{origin['global_full_evaluation_position']} | "
                        f"{origin['validation_top10_rank']} | "
                        f"{origin['search_validation_accuracy']:.12g} |"
                    )
            lines.append("")

    lines.extend(
        [
            "## Completion audit",
            "",
            f"- Search histories: {audit['search_histories']}.",
            f"- Validation Top-10 source slots: {audit['source_slots']}.",
            f"- Unique dataset-candidates: {audit['unique_dataset_candidates']}.",
            f"- Completed independent trainings: {audit['actual_completed_trainings']}.",
            f"- Deduplication savings: {audit['dedup_training_savings']} trainings.",
            f"- Self-contained detailed winner rows: {audit['detailed_winner_rows']}.",
            "- Detailed winner CSV: `test30_method_winners_detailed.csv`.",
            f"- Sum of GPU replicate wall times: {audit['total_gpu_training_wall_time_seconds_sum']:.3f} seconds.",
            f"- Missing / duplicate / failed replicates: {audit['missing_replicates']} / "
            f"{audit['duplicate_replicates']} / {audit['current_failed_tasks']}.",
            f"- Historical failure artifacts: {len(audit['historical_error_artifacts'])}; "
            f"failure events: {audit['failure_event_count']}; recovery events: "
            f"{audit['recovery_event_count']}.",
            "",
            f"Final declaration: **{ANALYSIS_LABEL}**. Selection used only the "
            "30-replicate final-validation mean; test metrics were report-only.",
            "",
        ]
    )
    return "\n".join(lines)


def maybe_finalize(pipeline_root: Path) -> bool:
    lock_path = pipeline_root / "finalize.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        counts = queue_counts(pipeline_root)
        if counts["completed"] != counts["total"] or any(
            counts[status] for status in ("pending", "running", "failed")
        ):
            return False
        if (pipeline_root / "audit_report.json").is_file():
            return True
        finalize_pipeline(pipeline_root)
        return True


def status_report(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_root = Path(args.pipeline_root).resolve()
    reconcile_result_artifacts(pipeline_root)
    counts = queue_counts(pipeline_root)
    config = load_json(pipeline_root / "pipeline_config.json")
    workers: dict[str, Any] = {}
    for gpu_id in (0, 1):
        pid_path = pipeline_root / f"worker_gpu{gpu_id}.pid"
        current_path = _current_task_path(pipeline_root, gpu_id)
        pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
        current_task = load_json(current_path) if current_path.exists() else None
        heartbeat_recent = bool(
            current_path.exists()
            and time.time() - current_path.stat().st_mtime < 600
        )
        pid_visible_alive = bool(pid and _process_alive(pid, socket.gethostname()))
        workers[f"gpu{gpu_id}"] = {
            "pid": pid,
            "alive": pid_visible_alive or heartbeat_recent,
            "pid_visible_alive": pid_visible_alive,
            "heartbeat_recent": heartbeat_recent,
            "current_task": current_task,
            "log": str(pipeline_root / f"queue_gpu{gpu_id}.log"),
        }
    report = {
        "analysis_label": ANALYSIS_LABEL,
        "pipeline_root": str(pipeline_root),
        "tmux_session": config["tmux_session"],
        "queue": counts,
        "unique_candidates": config["unique_candidate_count"],
        "replicate_tasks": config["replicate_task_count"],
        "final_results_available": (pipeline_root / "audit_report.json").is_file(),
        "workers": workers,
    }
    atomic_write_json(pipeline_root / "status_report.json", report)
    write_status_tsv(pipeline_root)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="audit histories and create queue")
    prepare.add_argument("--protocol-id", required=True)
    prepare.add_argument(
        "--pipeline-root",
        default=None,
        help=(
            "explicit output override; defaults to "
            "results/final_eval/<protocol-id>/first_formal_pipeline"
        ),
    )
    prepare.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT))
    prepare.add_argument("--search-summary", default=str(DEFAULT_SEARCH_SUMMARY))
    prepare.add_argument("--frozen-snapshot", default=str(DEFAULT_FROZEN_SNAPSHOT))

    for name in (
        "audit-storage",
        "launch",
        "launch-guarded",
        "launch-background",
        "status",
        "supervise-guarded",
        "retry-failed",
        "finalize",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--pipeline-root", required=True)
        if name in {"launch-guarded", "launch-background", "supervise-guarded"}:
            command.add_argument("--poll-seconds", type=int, default=30)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--pipeline-root", required=True)
    worker_parser.add_argument("--gpu-id", required=True, type=int, choices=(0, 1))
    guarded_parser = subparsers.add_parser("guarded-worker")
    guarded_parser.add_argument("--pipeline-root", required=True)
    guarded_parser.add_argument("--gpu-id", required=True, type=int, choices=(0, 1))
    guarded_parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_pipeline(args)
    elif args.command == "audit-storage":
        result = audit_dataset_storage(Path(args.pipeline_root).resolve())
    elif args.command == "launch":
        try:
            result = launch_workers(args)
        except BaseException:
            pipeline_root = Path(args.pipeline_root).resolve()
            error = {
                "status": "launch_gate_failed",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "traceback": traceback.format_exc(),
                "gpu_training_started": False,
            }
            atomic_write_json(pipeline_root / "launch_error.json", error)
            _append_event(pipeline_root, {"event": "launch_gate_failed", **error})
            raise
    elif args.command == "launch-guarded":
        try:
            result = launch_guarded_workers(args)
        except BaseException:
            pipeline_root = Path(args.pipeline_root).resolve()
            error = {
                "status": "guarded_launch_failed",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "traceback": traceback.format_exc(),
                "gpu_training_started": False,
            }
            atomic_write_json(pipeline_root / "guarded_launch_error.json", error)
            _append_event(pipeline_root, {"event": "guarded_launch_failed", **error})
            raise
    elif args.command == "launch-background":
        try:
            result = launch_background_guarded_workers(args)
        except BaseException:
            pipeline_root = Path(args.pipeline_root).resolve()
            error = {
                "status": "background_launch_failed",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "traceback": traceback.format_exc(),
                "gpu_training_started": False,
            }
            atomic_write_json(pipeline_root / "background_launch_error.json", error)
            _append_event(pipeline_root, {"event": "background_launch_failed", **error})
            raise
    elif args.command == "supervise-guarded":
        return supervise_guarded_workers(args)
    elif args.command == "worker":
        return worker(args)
    elif args.command == "guarded-worker":
        return guarded_worker(args)
    elif args.command == "status":
        result = status_report(args)
    elif args.command == "retry-failed":
        result = retry_failed(args)
    elif args.command == "finalize":
        result = finalize_pipeline(Path(args.pipeline_root).resolve())
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
