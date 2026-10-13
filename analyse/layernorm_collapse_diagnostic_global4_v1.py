#!/usr/bin/env python3
"""Freeze, run, and verify the LayerNorm collapse diagnostic v1.

This experiment is validation-only.  Dry-run freezes every scientific input
before any GNN is trained.  Formal execution makes exactly one attempt per
frozen task and never adapts settings, replaces tasks, or reads test data.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence
import uuid


EXPERIMENT = "layernorm-collapse-diagnostic-global4-v1"
IMPLEMENTATION_REL = "analyse/layernorm_collapse_diagnostic_global4_v1.py"
CONFIG_REL = "configs/layernorm_collapse_diagnostic_global4_v1.json"
RUNNER_REL = "scripts/run_layernorm_collapse_diagnostic_global4_v1.sh"
TEST_REL = "tests/test_layernorm_collapse_diagnostic_global4_v1.py"
SOURCE_RELS = (IMPLEMENTATION_REL, CONFIG_REL, RUNNER_REL, TEST_REL)
RESULT_REL = "results/posthoc/layernorm-collapse-diagnostic-global4-v1"
TMUX_NAME = "layernorm_collapse_diagnostic_v1"
CONDITION_IDS = (
    "no_ln",
    "ln_original",
    "ln_lr_x0_5",
    "ln_lr_x0_25",
    "ln_dropout_x0_5",
    "ln_weight_decay_x0_1",
)
LN_CONDITION_IDS = CONDITION_IDS[1:]
WILLIAMS_BASE = (0, 1, 5, 2, 4, 3)
DIAGNOSTIC_SEEDS = (2030, 2031, 2032, 2033, 2034)
RESERVED_SEEDS = tuple(range(2035, 2065))
ALLOWED_DATA_FIELDS = ("x", "edge_index", "train_mask", "val_mask", "y")
FORBIDDEN_METRIC_KEYS = {
    "test_accuracy",
    "test_acc",
    "best_test",
    "test_metric",
    "test_score",
    "test_loss",
    "test_result",
    "test_results",
    "test_outcome",
}
ALLOWED_TEST_CONTROL_KEYS = {
    "formal_track_test_literal",
    "skipped_test_field_count",
    "skipped_test_field_path_commitment_sha256",
    "test_access",
    "test_accessed",
    "test_mask_attribute_count_in_analyser",
    "test_value_policy",
    "track_test",
}


def _default_repo_root() -> Path:
    configured = os.environ.get("D_VAE_REPO_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _default_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ProtocolError(RuntimeError):
    """A fail-closed protocol or verification error."""

    def __init__(self, stage: str, code: str, detail: str | None = None):
        self.stage = stage
        self.code = code
        self.detail = detail
        message = f"{stage}:{code}"
        if detail:
            message += f":{detail}"
        super().__init__(message)


class NonFiniteTrainingError(RuntimeError):
    def __init__(self, stage: str, history: Sequence[Mapping[str, Any]]):
        self.stage = stage
        self.history = [dict(row) for row in history]
        self.best_validation_accuracy: float | None = None
        self.best_epoch: int | None = None
        self.best_validation_prediction_counts: list[int] | None = None
        self.validation_sample_count: int | None = None
        super().__init__(f"non-finite value detected at {stage}")


class TestAccessViolation(RuntimeError):
    """Raised after the validation-only data proxy records a forbidden access."""


def new_data_access_ledger() -> dict[str, Any]:
    return {
        "allowed_request_count": 0,
        "forbidden_request_count": 0,
        "allowed_attribute_counts": {
            field: 0 for field in sorted(ALLOWED_DATA_FIELDS)
        },
        "forbidden_attribute_counts": {},
        "test_accessed": False,
    }


class ValidationOnlyDataProxy:
    """Runtime firewall exposing only the five validation-training tensors."""

    def __init__(self, data: Any, ledger: dict[str, Any] | None = None):
        object.__setattr__(self, "_data", data)
        object.__setattr__(
            self, "_ledger", ledger if ledger is not None else new_data_access_ledger()
        )

    def __getattribute__(self, name: str) -> Any:
        ledger = object.__getattribute__(self, "_ledger")
        lowered = str(name).casefold()
        if "test" in lowered or name not in ALLOWED_DATA_FIELDS:
            ledger["forbidden_request_count"] += 1
            ledger["forbidden_attribute_counts"][str(name)] = (
                ledger["forbidden_attribute_counts"].get(str(name), 0) + 1
            )
            if "test" in lowered:
                ledger["test_accessed"] = True
            raise TestAccessViolation("validation-only data attribute denied")
        ledger["allowed_request_count"] += 1
        ledger["allowed_attribute_counts"][name] += 1
        return getattr(object.__getattribute__(self, "_data"), name)


def data_access_ledger(proxy: ValidationOnlyDataProxy) -> dict[str, Any]:
    """Return a defensive ledger copy without passing through the proxy."""

    return copy.deepcopy(object.__getattribute__(proxy, "_ledger"))


def fail(stage: str, code: str, detail: str | None = None) -> None:
    raise ProtocolError(stage, code, detail)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def commitment(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_embedded_commitment(
    document: Mapping[str, Any], field: str, stage: str
) -> str:
    payload = copy.deepcopy(dict(document))
    observed = payload.pop(field, None)
    if not isinstance(observed, str) or commitment(payload) != observed:
        fail(stage, "EMBEDDED_COMMITMENT_MISMATCH", field)
    return observed


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, relative_path: str | None = None) -> dict[str, Any]:
    return {
        "path": relative_path if relative_path is not None else str(path),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _assert_no_test_metric_keys_text(text: str, path: Path) -> None:
    keys = set(re.findall(r'"([^"\\]+)"\s*:', text))
    forbidden = sorted(
        key
        for key in keys
        if key.lower() in FORBIDDEN_METRIC_KEYS
        or (
            re.search(r"(?:^|_)test(?:$|_)", key.lower()) is not None
            and key.lower() not in ALLOWED_TEST_CONTROL_KEYS
        )
    )
    if forbidden:
        fail("TEST_FIREWALL", "FORBIDDEN_TEST_METRIC_KEY", f"{path}:{forbidden}")


def read_json(path: Path, *, reject_test_metrics: bool = False) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProtocolError("INPUT", "JSON_READ_FAILED", str(path)) from exc
    if reject_test_metrics:
        _assert_no_test_metric_keys_text(text, path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError("INPUT", "JSON_PARSE_FAILED", str(path)) from exc


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise ProtocolError("WRITE", "OVERWRITE_FORBIDDEN", str(path)) from exc


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path, reject_test_metrics=True)
    if not isinstance(config, dict):
        fail("CONFIG", "ROOT_NOT_OBJECT")
    if config.get("schema_version") != EXPERIMENT:
        fail("CONFIG", "SCHEMA_MISMATCH")
    if config.get("repository_root") != "/home/fangxy/D-VAE":
        fail("CONFIG", "REPOSITORY_ROOT_MISMATCH")
    if config.get("result_root") != RESULT_REL:
        fail("CONFIG", "RESULT_ROOT_MISMATCH")
    if config.get("tmux_name") != TMUX_NAME:
        fail("CONFIG", "TMUX_NAME_MISMATCH")
    return config


def resolve_result_root(config: Mapping[str, Any], override: str | None = None) -> Path:
    expected = (REPO_ROOT / str(config["result_root"])).resolve()
    if override is None:
        return expected
    observed = Path(override)
    if not observed.is_absolute():
        observed = (REPO_ROOT / observed).resolve()
    else:
        observed = observed.resolve()
    if observed != expected:
        fail("CONFIG", "RESULT_ROOT_OVERRIDE_MISMATCH", str(observed))
    return observed


def assert_host(config: Mapping[str, Any]) -> dict[str, Any]:
    hostname = socket.gethostname()
    short = hostname.split(".", 1)[0]
    if short != config.get("required_host"):
        fail("PREFLIGHT", "HOST_MISMATCH", short)
    if REPO_ROOT != Path(str(config["repository_root"])).resolve():
        fail("PREFLIGHT", "LIVE_REPOSITORY_ROOT_MISMATCH", str(REPO_ROOT))
    return {"hostname": hostname, "short_hostname": short}


def tmux_exists(name: str = TMUX_NAME) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def formal_process_exists(run_id: str | None = None) -> bool:
    pattern = "layernorm_collapse_diagnostic_global4_v1.py.*formal-run"
    if run_id:
        pattern += f".*{re.escape(run_id)}"
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def query_gpus(config: Mapping[str, Any], *, require_idle: bool) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        fail("GPU", "NVIDIA_SMI_QUERY_FAILED")
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            fail("GPU", "GPU_QUERY_SCHEMA_MISMATCH", line)
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": int(parts[3]),
                "memory_used_mib": int(parts[4]),
                "utilization_percent": int(parts[5]),
            }
        )
    expected = {
        int(row["gpu_index"]): str(row["gpu_uuid"])
        for row in config["workers"]["mapping"]
    }
    if {row["index"]: row["uuid"] for row in rows} != expected:
        fail("GPU", "GPU_IDENTITY_MISMATCH")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if processes.returncode != 0:
        fail("GPU", "COMPUTE_PROCESS_QUERY_FAILED")
    compute_lines = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    if require_idle:
        if compute_lines:
            fail("GPU", "COMPUTE_PROCESS_PRESENT")
        if any(row["memory_used_mib"] > 512 for row in rows):
            fail("GPU", "GPU_MEMORY_NOT_IDLE")
        if any(row["utilization_percent"] > 5 for row in rows):
            fail("GPU", "GPU_UTILIZATION_NOT_IDLE")
    return {"gpus": rows, "compute_process_count": len(compute_lines), "idle_required": require_idle}


def verify_pinned_files(config: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    sources = [
        config["phase_d"]["candidate_source"],
        config["phase_d"]["validation_result_source"],
        *config["seed_freeze"]["ledger_sources"],
    ]
    for source in sources:
        relative = str(source["path"])
        path = REPO_ROOT / relative
        observed = sha256_path(path)
        if observed != source["sha256"]:
            fail("INPUT", "PINNED_INPUT_HASH_MISMATCH", relative)
        records[relative] = file_record(path, relative)
    for relative, expected in config["dependency_hashes"].items():
        path = REPO_ROOT / relative
        observed = sha256_path(path)
        if observed != expected:
            fail("SOURCE", "DEPENDENCY_HASH_MISMATCH", relative)
        records[relative] = file_record(path, relative)
    return records


def _config_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "joint_config_fingerprint": row["joint_config_fingerprint"],
        "control_mean": row["control_mean"],
        "layernorm_mean": row["layernorm_mean"],
        "delta": row["delta"],
    }


def select_configuration_groups(
    complete: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, int]]:
    collapse_all = [
        row
        for row in complete
        if row["control_mean"] >= 0.75
        and all(
            item["validation_accuracy"] == 0.316
            for item in row["layernorm_by_seed"]
        )
    ]
    collapse = sorted(
        collapse_all,
        key=lambda row: (-row["control_mean"], row["joint_config_fingerprint"]),
    )[:4]
    used = {row["joint_config_fingerprint"] for row in collapse}
    improved_all = [
        row
        for row in complete
        if row["joint_config_fingerprint"] not in used and row["delta"] > 0
    ]
    improved = sorted(
        improved_all,
        key=lambda row: (-row["delta"], row["joint_config_fingerprint"]),
    )[:4]
    used.update(row["joint_config_fingerprint"] for row in improved)
    decline_all = [
        row
        for row in complete
        if row["joint_config_fingerprint"] not in used
        and row["control_mean"] >= 0.70
        and row["layernorm_mean"] >= 0.70
        and row["delta"] < 0
    ]
    decline = sorted(
        decline_all,
        key=lambda row: (row["delta"], row["joint_config_fingerprint"]),
    )[:4]
    groups: dict[str, list[Mapping[str, Any]]] = {
        "collapse": collapse,
        "improved": improved,
        "normal_decline": decline,
    }
    counts = {
        "collapse": len(collapse_all),
        "positive_improvement_remaining": len(improved_all),
        "normal_decline_remaining": len(decline_all),
    }
    return groups, counts


def freeze_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    candidate_path = REPO_ROOT / config["phase_d"]["candidate_source"]["path"]
    results_path = REPO_ROOT / config["phase_d"]["validation_result_source"]["path"]
    candidate_doc = read_json(candidate_path, reject_test_metrics=True)
    results_doc = read_json(results_path, reject_test_metrics=True)
    candidates = candidate_doc.get("candidates")
    results = results_doc.get("results")
    if not isinstance(candidates, list) or not isinstance(results, list):
        fail("SELECTION", "SOURCE_SCHEMA_MISMATCH")
    if len(candidates) != 192:
        fail("SELECTION", "CANDIDATE_COUNT_MISMATCH")
    candidate_by_fp: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        fp = str(candidate.get("joint_config_fingerprint", ""))
        if not fp or fp in candidate_by_fp:
            fail("SELECTION", "CANDIDATE_FINGERPRINT_INVALID")
        candidate_by_fp[fp] = copy.deepcopy(candidate)

    values: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    control_arm = str(config["phase_d"]["control_arm"])
    ln_arm = str(config["phase_d"]["layernorm_arm"])
    for row in results:
        if row.get("test_accessed") is not False:
            fail("TEST_FIREWALL", "PHASE_D_RESULT_TEST_ACCESS_NOT_FALSE")
        arm = row.get("arm")
        fp = str(row.get("joint_config_fingerprint", ""))
        if arm not in {control_arm, ln_arm} or fp not in candidate_by_fp:
            continue
        validation = float(row["validation_accuracy"])
        if not math.isfinite(validation):
            fail("SELECTION", "NONFINITE_PHASE_D_VALIDATION")
        values[fp][str(arm)].append((int(row["seed"]), validation))

    complete: list[dict[str, Any]] = []
    expected_seeds = list(config["phase_d"]["source_seeds"])
    for fp, candidate in candidate_by_fp.items():
        control = sorted(values.get(fp, {}).get(control_arm, []))
        layernorm = sorted(values.get(fp, {}).get(ln_arm, []))
        if [seed for seed, _ in control] != expected_seeds:
            continue
        if [seed for seed, _ in layernorm] != expected_seeds:
            continue
        control_mean = math.fsum(value for _, value in control) / len(control)
        layernorm_mean = math.fsum(value for _, value in layernorm) / len(layernorm)
        complete.append(
            {
                "joint_config_fingerprint": fp,
                "config_id": candidate["config_id"],
                "control_by_seed": [
                    {"seed": seed, "validation_accuracy": value}
                    for seed, value in control
                ],
                "layernorm_by_seed": [
                    {"seed": seed, "validation_accuracy": value}
                    for seed, value in layernorm
                ],
                "control_mean": control_mean,
                "layernorm_mean": layernorm_mean,
                "delta": layernorm_mean - control_mean,
                "candidate": candidate,
            }
        )
    required_count = int(config["phase_d"]["required_complete_paired_config_count"])
    if len(complete) != required_count:
        fail("SELECTION", "COMPLETE_PAIRED_CONFIG_COUNT_MISMATCH", str(len(complete)))

    groups, qualifying_counts = select_configuration_groups(complete)
    if any(len(rows) != 4 for rows in groups.values()):
        fail("SELECTION", "INSUFFICIENT_SELECTION_GROUP_ZERO_GNN")
    fingerprints = [
        row["joint_config_fingerprint"]
        for name in ("collapse", "improved", "normal_decline")
        for row in groups[name]
    ]
    if len(set(fingerprints)) != 12:
        fail("SELECTION", "SELECTION_GROUP_OVERLAP")

    configured = {
        group["name"]: group["selected"] for group in config["selection"]["groups"]
    }
    for name, rows in groups.items():
        observed = [_config_summary(row) for row in rows]
        if canonical_json_bytes(observed) != canonical_json_bytes(configured.get(name)):
            fail("SELECTION", "CONFIGURED_SELECTION_MISMATCH", name)

    record: dict[str, Any] = {
        "schema_version": "layernorm-collapse-selection-freeze-v1",
        "source_records": [
            dict(config["phase_d"]["candidate_source"]),
            dict(config["phase_d"]["validation_result_source"]),
        ],
        "complete_paired_config_count": len(complete),
        "qualifying_counts": {
            **qualifying_counts,
        },
        "selection_rules": {
            "collapse": "Control mean >= 0.75; every LN seed == 0.316; Control mean desc then fingerprint asc",
            "improved": "remaining positive LN-Control; delta desc then fingerprint asc",
            "normal_decline": "remaining means >= 0.70 and LN-Control < 0; delta asc then fingerprint asc",
        },
        "groups": [
            {"name": name, "configs": groups[name]}
            for name in ("collapse", "improved", "normal_decline")
        ],
    }
    record["commitment_sha256"] = commitment(record)
    return record


def ascending_unused_seeds(used: set[int], start: int, count: int) -> list[int]:
    selected: list[int] = []
    value = int(start)
    while len(selected) < int(count):
        if value not in used:
            selected.append(value)
        value += 1
    return selected


def retain_unresolved_seed_finding(finding: Mapping[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(dict(finding))
    if row.get("semantic") != "ambiguous":
        return row
    row["semantic"] = "unresolved_seed_semantics_for_window_audit"
    row["reason"] = (
        "unresolved numeric seed semantics retained for the exact first-35 "
        "selection-window proof"
    )
    return row


def build_candidate_window_gnn_seed_inventory(
    repo_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an exact inventory for the bounded window needed by this freeze.

    The pinned repository scanner normally aborts on any unresolved numeric
    seed semantics anywhere in history.  This wrapper retains those rows under
    a committed audit semantic, derives the window ending at the provisional
    35th unused recognized seed, and fails closed if any unresolved value lies
    inside that exact selection window.
    """

    import analyse.final_checkpoint_seed_inventory as inventory_module

    start = int(config["seed_freeze"]["scan_start"])
    scan_roots = tuple(str(value) for value in config["seed_freeze"]["scan_roots"])
    original_walker = inventory_module._walk_seed_fields
    original_structured_kind = inventory_module._structured_kind
    original_normalize_relpath = inventory_module._normalize_relpath
    repository_prefix = original_normalize_relpath(Path(repo_root).resolve())

    def window_normalize_relpath(value: str | PurePosixPath) -> str:
        normalized = original_normalize_relpath(value)
        prefix = f"{repository_prefix}/"
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
        return normalized

    def window_structured_kind(name: str) -> str | None:
        normalized_name = PurePosixPath(name).as_posix()
        if normalized_name in {"raw/class_map.json", "raw/role.json"}:
            return None
        return original_structured_kind(name)

    def window_guarded_walker(
        value: Any, *, source: str, path: str = "$"
    ) -> Iterator[dict[str, Any]]:
        for finding in original_walker(value, source=source, path=path):
            yield retain_unresolved_seed_finding(finding)

    inventory_module._normalize_relpath = window_normalize_relpath
    inventory_module._structured_kind = window_structured_kind
    inventory_module._walk_seed_fields = window_guarded_walker
    try:
        inventory = inventory_module.build_historical_gnn_seed_inventory(
            repo_root,
            scan_roots=scan_roots,
            excluded_target=str(config["result_root"]),
        )
    finally:
        inventory_module._walk_seed_fields = original_walker
        inventory_module._structured_kind = original_structured_kind
        inventory_module._normalize_relpath = original_normalize_relpath
    without_root = dict(inventory)
    observed_root = without_root.pop("inventory_root_sha256", None)
    expected_root = inventory_module._sha256_bytes(
        inventory_module._canonical_json(without_root)
    )
    if observed_root != expected_root:
        fail("SEEDS", "GLOBAL_INVENTORY_ROOT_MISMATCH")
    unresolved_values = sorted(
        {
            int(value)
            for value in inventory.get("semantic_seed_sets", {}).get(
                "unresolved_seed_semantics_for_window_audit", []
            )
        }
    )
    unresolved_groups = [
        row
        for row in inventory.get("seed_field_matches", [])
        if row.get("semantic") == "unresolved_seed_semantics_for_window_audit"
    ]
    recognized_used = {
        int(value) for value in inventory.get("executed_gnn_training_seeds", [])
    }
    provisional_selected = ascending_unused_seeds(recognized_used, start, 35)
    proof_end = provisional_selected[-1]
    all_unresolved_in_proof_window = sorted(
        value for value in unresolved_values if start <= value <= proof_end
    )
    independently_recognized_used_unresolved = sorted(
        set(all_unresolved_in_proof_window) & recognized_used
    )
    blocking_unresolved_candidate_values = sorted(
        set(all_unresolved_in_proof_window) & set(provisional_selected)
    )
    audit: dict[str, Any] = {
        "schema_version": "candidate-window-seed-semantics-audit-v1",
        "window_start_inclusive": start,
        "window_end_inclusive": proof_end,
        "all_unresolved_value_count_in_proof_window": len(
            all_unresolved_in_proof_window
        ),
        "all_unresolved_values_in_proof_window": all_unresolved_in_proof_window,
        "independently_recognized_used_unresolved_value_count_in_proof_window": len(
            independently_recognized_used_unresolved
        ),
        "independently_recognized_used_unresolved_values_in_proof_window": (
            independently_recognized_used_unresolved
        ),
        "blocking_unresolved_candidate_value_count": len(
            blocking_unresolved_candidate_values
        ),
        "blocking_unresolved_candidate_values": blocking_unresolved_candidate_values,
        "all_unresolved_group_count": len(unresolved_groups),
        "all_unresolved_occurrence_count": sum(
            int(row.get("occurrence_count", 0)) for row in unresolved_groups
        ),
        "all_unresolved_value_count": len(unresolved_values),
        "all_unresolved_values_commitment_sha256": commitment(unresolved_values),
        "provisional_first_35": provisional_selected,
        "exact_first_35_selection_provable": not blocking_unresolved_candidate_values,
    }
    audit["commitment_sha256"] = commitment(audit)
    if blocking_unresolved_candidate_values:
        fail("SEEDS", "UNRESOLVED_SEED_SEMANTICS_IN_SELECTION_WINDOW")
    return inventory, audit


def freeze_seeds(
    config: Mapping[str, Any], global_inventory: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    used: set[int] = set()
    details: list[dict[str, Any]] = []
    for source in config["seed_freeze"]["ledger_sources"]:
        path = REPO_ROOT / source["path"]
        document = read_json(path, reject_test_metrics=True)
        attempts = document.get("attempts")
        if not isinstance(attempts, list):
            fail("SEEDS", "ATTEMPT_LEDGER_SCHEMA_MISMATCH", str(path))
        seeds = sorted({int(row["seed"]) for row in attempts})
        used.update(seeds)
        details.append(
            {
                "path": str(source["path"]),
                "sha256": str(source["sha256"]),
                "attempt_count": len(attempts),
                "used_seeds": seeds,
                "statuses": sorted({str(row["status"]) for row in attempts}),
            }
        )
    if global_inventory is None:
        global_inventory, candidate_window_audit = (
            build_candidate_window_gnn_seed_inventory(REPO_ROOT, config)
        )
    else:
        candidate_window_audit = copy.deepcopy(
            global_inventory.get("candidate_window_audit")
        )
    if not isinstance(candidate_window_audit, dict):
        fail("SEEDS", "CANDIDATE_WINDOW_AUDIT_MISSING")
    global_used = {
        int(value) for value in global_inventory.get("executed_gnn_training_seeds", [])
    }
    if not used.issubset(global_used):
        fail("SEEDS", "PHASE_CD_ATTEMPT_SEEDS_MISSING_FROM_GLOBAL_INVENTORY")
    selected = ascending_unused_seeds(
        global_used, int(config["seed_freeze"]["scan_start"]), 35
    )
    if (
        candidate_window_audit.get("blocking_unresolved_candidate_values") != []
        or candidate_window_audit.get("exact_first_35_selection_provable") is not True
        or candidate_window_audit.get("provisional_first_35") != selected
    ):
        fail("SEEDS", "EXACT_FIRST_35_NOT_PROVEN")
    commitment_payload: dict[str, Any] = {
        "scan_start": int(config["seed_freeze"]["scan_start"]),
        "algorithm": str(config["seed_freeze"]["algorithm"]),
        "ledger_sources": details,
        "diagnostic_seeds": selected[:5],
        "reserved_confirmation_seeds": selected[5:],
        "all_selected_seeds": selected,
    }
    observed_commitment = commitment(commitment_payload)
    if selected[:5] != list(DIAGNOSTIC_SEEDS):
        fail("SEEDS", "DIAGNOSTIC_SEED_FREEZE_MISMATCH")
    if selected[5:] != list(RESERVED_SEEDS):
        fail("SEEDS", "RESERVED_SEED_FREEZE_MISMATCH")
    if observed_commitment != config["seed_freeze"]["commitment_sha256"]:
        fail("SEEDS", "SEED_COMMITMENT_MISMATCH", observed_commitment)
    if set(selected) & global_used or set(selected[:5]) & set(selected[5:]):
        fail("SEEDS", "SEED_COLLISION")
    global_summary = {
        "schema_version": global_inventory.get("schema_version"),
        "scan_roots": copy.deepcopy(global_inventory.get("scan_roots")),
        "excluded_target": global_inventory.get("excluded_target"),
        "source_count": global_inventory.get("source_count"),
        "seed_field_match_count": global_inventory.get("seed_field_match_count"),
        "executed_gnn_training_seeds": sorted(global_used),
        "executed_seed_set_commitment_sha256": commitment(sorted(global_used)),
        "skipped_test_field_count": global_inventory.get("skipped_test_field_count"),
        "skipped_test_field_path_commitment_sha256": global_inventory.get(
            "skipped_test_field_path_commitment_sha256"
        ),
        "test_value_policy": global_inventory.get("test_value_policy"),
        "inventory_root_sha256": global_inventory.get("inventory_root_sha256"),
        "candidate_window_audit": candidate_window_audit,
    }
    record: dict[str, Any] = {
        "schema_version": "layernorm-collapse-seed-freeze-v1",
        **copy.deepcopy(commitment_payload),
        "commitment_payload": commitment_payload,
        "commitment_sha256": observed_commitment,
        "commitment_scope": (
            "commitment_payload: all selected seeds, diagnostic/reserved split, "
            "algorithm, and Phase C/D actual ledger evidence"
        ),
        "global_inventory": global_summary,
        "global_inventory_attestation_sha256": commitment(global_summary),
    }
    validate_frozen_seed_record(record, config)
    return record


def validate_frozen_seed_record(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    if record.get("schema_version") != "layernorm-collapse-seed-freeze-v1":
        fail("SEEDS", "FROZEN_SEED_SCHEMA_MISMATCH")
    payload = {
        key: copy.deepcopy(record.get(key))
        for key in (
            "scan_start",
            "algorithm",
            "ledger_sources",
            "diagnostic_seeds",
            "reserved_confirmation_seeds",
            "all_selected_seeds",
        )
    }
    if canonical_json_bytes(record.get("commitment_payload")) != canonical_json_bytes(
        payload
    ):
        fail("SEEDS", "COMMITMENT_PAYLOAD_MISMATCH")
    observed_commitment = commitment(payload)
    if (
        record.get("commitment_sha256") != observed_commitment
        or config["seed_freeze"]["commitment_sha256"] != observed_commitment
    ):
        fail("SEEDS", "FROZEN_SEED_COMMITMENT_MISMATCH")
    global_summary = record.get("global_inventory")
    if not isinstance(global_summary, dict):
        fail("SEEDS", "GLOBAL_INVENTORY_SUMMARY_MISSING")
    if commitment(global_summary) != record.get("global_inventory_attestation_sha256"):
        fail("SEEDS", "GLOBAL_INVENTORY_ATTESTATION_MISMATCH")
    window_audit = global_summary.get("candidate_window_audit")
    if not isinstance(window_audit, dict):
        fail("SEEDS", "CANDIDATE_WINDOW_AUDIT_MISSING")
    verify_embedded_commitment(
        window_audit, "commitment_sha256", "CANDIDATE_WINDOW"
    )
    global_used = {
        int(value)
        for value in global_summary.get("executed_gnn_training_seeds", [])
    }
    if global_summary.get("executed_seed_set_commitment_sha256") != commitment(
        sorted(global_used)
    ):
        fail("SEEDS", "EXECUTED_SEED_SET_COMMITMENT_MISMATCH")
    expected = ascending_unused_seeds(
        global_used, int(config["seed_freeze"]["scan_start"]), 35
    )
    all_unresolved_in_proof_window = window_audit.get(
        "all_unresolved_values_in_proof_window"
    )
    if not isinstance(all_unresolved_in_proof_window, list) or any(
        type(value) is not int for value in all_unresolved_in_proof_window
    ):
        fail("SEEDS", "CANDIDATE_WINDOW_UNRESOLVED_SCHEMA_MISMATCH")
    normalized_unresolved = sorted(set(all_unresolved_in_proof_window))
    independently_recognized_used_unresolved = sorted(
        set(normalized_unresolved) & global_used
    )
    blocking_unresolved_candidate_values = sorted(
        set(normalized_unresolved) & set(expected)
    )
    if (
        window_audit.get("window_start_inclusive")
        != int(config["seed_freeze"]["scan_start"])
        or window_audit.get("window_end_inclusive")
        != expected[-1]
        or normalized_unresolved != all_unresolved_in_proof_window
        or any(
            not int(config["seed_freeze"]["scan_start"])
            <= value
            <= expected[-1]
            for value in normalized_unresolved
        )
        or window_audit.get("all_unresolved_value_count_in_proof_window")
        != len(normalized_unresolved)
        or window_audit.get(
            "independently_recognized_used_unresolved_values_in_proof_window"
        )
        != independently_recognized_used_unresolved
        or window_audit.get(
            "independently_recognized_used_unresolved_value_count_in_proof_window"
        )
        != len(independently_recognized_used_unresolved)
        or window_audit.get("blocking_unresolved_candidate_values")
        != blocking_unresolved_candidate_values
        or window_audit.get("blocking_unresolved_candidate_value_count")
        != len(blocking_unresolved_candidate_values)
        or blocking_unresolved_candidate_values
        or window_audit.get("exact_first_35_selection_provable") is not True
        or window_audit.get("provisional_first_35") != expected
    ):
        fail("SEEDS", "CANDIDATE_WINDOW_PROOF_MISMATCH")
    if payload["all_selected_seeds"] != expected:
        fail("SEEDS", "GLOBAL_UNUSED_SELECTION_MISMATCH")
    if (
        payload["diagnostic_seeds"] != expected[:5]
        or payload["reserved_confirmation_seeds"] != expected[5:]
        or set(expected) & global_used
        or set(expected[:5]) & set(expected[5:])
    ):
        fail("SEEDS", "FROZEN_SEED_PARTITION_MISMATCH")
    ledger_used = {
        int(seed)
        for source in payload["ledger_sources"]
        for seed in source["used_seeds"]
    }
    if not ledger_used.issubset(global_used) or set(expected) & ledger_used:
        fail("SEEDS", "PHASE_CD_FROZEN_SEED_COLLISION")
    return {
        "passed": True,
        "commitment_sha256": observed_commitment,
        "global_inventory_attestation_sha256": record[
            "global_inventory_attestation_sha256"
        ],
        "selected_seed_count": len(expected),
    }


def freeze_conditions(config: Mapping[str, Any]) -> dict[str, Any]:
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or [row.get("id") for row in conditions] != list(CONDITION_IDS):
        fail("CONDITIONS", "CONDITION_ORDER_MISMATCH")
    expected = {
        "no_ln": ("baseline", 1.0, 1.0, 1.0, None),
        "ln_original": ("layer_norm", 1.0, 1.0, 1.0, 0),
        "ln_lr_x0_5": ("layer_norm", 0.5, 1.0, 1.0, 1),
        "ln_lr_x0_25": ("layer_norm", 0.25, 1.0, 1.0, 2),
        "ln_dropout_x0_5": ("layer_norm", 1.0, 0.5, 1.0, 3),
        "ln_weight_decay_x0_1": ("layer_norm", 1.0, 1.0, 0.1, 4),
    }
    for row in conditions:
        observed = (
            row.get("component_variant"),
            float(row.get("lr_multiplier")),
            float(row.get("dropout_multiplier")),
            float(row.get("weight_decay_multiplier")),
            row.get("stable_recipe_priority"),
        )
        if observed != expected[row["id"]]:
            fail("CONDITIONS", "CONDITION_CONTRACT_MISMATCH", row["id"])
    record: dict[str, Any] = {
        "schema_version": "layernorm-collapse-condition-freeze-v1",
        "conditions": copy.deepcopy(conditions),
        "collapse_definition": copy.deepcopy(config["collapse_definition"]),
        "stable_recipe_gate": copy.deepcopy(config["stable_recipe_gate"]),
        "no_combined_or_additional_changes": True,
    }
    record["commitment_sha256"] = commitment(record)
    return record


def _effective_hp(base: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    hp = copy.deepcopy(dict(base))
    hp["lr"] = float(base["lr"]) * float(condition["lr_multiplier"])
    hp["dropout"] = float(base["dropout"]) * float(condition["dropout_multiplier"])
    hp["l2"] = float(base["l2"]) * float(condition["weight_decay_multiplier"])
    return hp


def build_tasks(
    selection: Mapping[str, Any],
    seeds: Mapping[str, Any],
    conditions: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    selected: list[tuple[str, Mapping[str, Any]]] = []
    for group in selection["groups"]:
        for row in group["configs"]:
            selected.append((str(group["name"]), row))
    condition_rows = [dict(row) for row in conditions["conditions"]]
    mapping = {int(row["worker_id"]): dict(row) for row in config["workers"]["mapping"]}
    tasks: list[dict[str, Any]] = []
    for config_index, (group, selected_row) in enumerate(selected):
        candidate = selected_row["candidate"]
        architecture = {
            "operations": copy.deepcopy(candidate["operations"]),
            "edges": copy.deepcopy(candidate["edges"]),
            "effective_layers": int(candidate["effective_layers"]),
        }
        for seed_index, seed in enumerate(seeds["diagnostic_seeds"]):
            block_index = config_index * 5 + seed_index
            worker_id = block_index % 4
            worker_local_index = block_index // 4
            worker = mapping[worker_id]
            row_index = (worker_local_index + int(worker["rotation_offset"])) % 6
            order = [((value + row_index) % 6) for value in WILLIAMS_BASE]
            for position, condition_index in enumerate(order):
                condition = condition_rows[condition_index]
                task_id = f"c{config_index:02d}_s{int(seed)}_{condition['id']}"
                task = {
                    "task_id": task_id,
                    "block_id": f"c{config_index:02d}_s{int(seed)}",
                    "block_index": block_index,
                    "config_index": config_index,
                    "selection_group": group,
                    "config_id": candidate["config_id"],
                    "joint_config_fingerprint": candidate["joint_config_fingerprint"],
                    "architecture_fingerprint": candidate["architecture_fingerprint"],
                    "hp_fingerprint": candidate["hp_fingerprint"],
                    "architecture": architecture,
                    "base_hp": copy.deepcopy(candidate["decoded_hp"]),
                    "effective_hp": _effective_hp(candidate["decoded_hp"], condition),
                    "seed": int(seed),
                    "condition_id": condition["id"],
                    "condition_label": condition["label"],
                    "component_variant": condition["component_variant"],
                    "execution_position": position,
                    "williams_row": row_index,
                    "worker_sequence": worker_local_index * 6 + position,
                    "worker_id": worker_id,
                    "gpu_index": int(worker["gpu_index"]),
                    "gpu_uuid": worker["gpu_uuid"],
                    "max_epochs": int(config["training"]["max_epochs"]),
                    "patience": int(config["training"]["patience"]),
                    "track_test": False,
                }
                tasks.append(task)
    record: dict[str, Any] = {
        "schema_version": "layernorm-collapse-task-freeze-v1",
        "planned_task_count": len(tasks),
        "paired_block_count": len(tasks) // 6,
        "tasks": tasks,
    }
    record["validation"] = validate_tasks(record, config)
    record["commitment_sha256"] = commitment(record)
    return record


def validate_tasks(record: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    tasks = record.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 360:
        fail("TASKS", "TASK_COUNT_MISMATCH")
    if len({row["task_id"] for row in tasks}) != 360:
        fail("TASKS", "DUPLICATE_TASK_ID")
    blocks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task.get("track_test") is not False:
            fail("TEST_FIREWALL", "TASK_TRACK_TEST_NOT_FALSE")
        blocks[str(task["block_id"])].append(task)
    if len(blocks) != 60:
        fail("TASKS", "BLOCK_COUNT_MISMATCH")
    conditions_by_id = {row["id"]: row for row in config["conditions"]}
    for block_id, rows in blocks.items():
        if {row["condition_id"] for row in rows} != set(CONDITION_IDS):
            fail("TASKS", "BLOCK_CONDITION_SET_MISMATCH", block_id)
        if len({row["worker_id"] for row in rows}) != 1:
            fail("TASKS", "BLOCK_WORKER_MISMATCH", block_id)
        if len({(row["gpu_index"], row["gpu_uuid"]) for row in rows}) != 1:
            fail("TASKS", "BLOCK_GPU_MISMATCH", block_id)
        invariant_fields = (
            "seed",
            "config_id",
            "joint_config_fingerprint",
            "architecture_fingerprint",
            "hp_fingerprint",
            "architecture",
            "base_hp",
            "max_epochs",
            "patience",
        )
        for field in invariant_fields:
            if len({canonical_json_bytes(row[field]) for row in rows}) != 1:
                fail("TASKS", "BLOCK_INVARIANT_MISMATCH", f"{block_id}:{field}")
        base = rows[0]["base_hp"]
        for row in rows:
            condition = conditions_by_id[row["condition_id"]]
            expected_hp = _effective_hp(base, condition)
            if canonical_json_bytes(row["effective_hp"]) != canonical_json_bytes(expected_hp):
                fail("TASKS", "EFFECTIVE_HP_MISMATCH", row["task_id"])
            if row["component_variant"] != condition["component_variant"]:
                fail("TASKS", "COMPONENT_VARIANT_MISMATCH", row["task_id"])
    worker_counts = Counter(int(row["worker_id"]) for row in tasks)
    gpu_counts = Counter(int(row["gpu_index"]) for row in tasks)
    if worker_counts != Counter({0: 90, 1: 90, 2: 90, 3: 90}):
        fail("TASKS", "WORKER_BALANCE_MISMATCH")
    if gpu_counts != Counter({0: 180, 1: 180}):
        fail("TASKS", "GPU_BALANCE_MISMATCH")
    global_positions = Counter(
        (str(row["condition_id"]), int(row["execution_position"])) for row in tasks
    )
    expected_global = Counter(
        {(condition, position): 10 for condition in CONDITION_IDS for position in range(6)}
    )
    if global_positions != expected_global:
        fail("TASKS", "GLOBAL_POSITION_BALANCE_MISMATCH")
    gpu_positions = Counter(
        (int(row["gpu_index"]), str(row["condition_id"]), int(row["execution_position"]))
        for row in tasks
    )
    expected_gpu = Counter(
        {
            (gpu, condition, position): 5
            for gpu in (0, 1)
            for condition in CONDITION_IDS
            for position in range(6)
        }
    )
    if gpu_positions != expected_gpu:
        fail("TASKS", "GPU_POSITION_BALANCE_MISMATCH")
    first_counts = Counter(
        row["condition_id"] for row in tasks if row["execution_position"] == 0
    )
    if first_counts != Counter({condition: 10 for condition in CONDITION_IDS}):
        fail("TASKS", "FIRST_POSITION_BALANCE_MISMATCH")
    return {
        "passed": True,
        "task_count": 360,
        "block_count": 60,
        "worker_task_counts": {str(key): value for key, value in sorted(worker_counts.items())},
        "gpu_task_counts": {str(key): value for key, value in sorted(gpu_counts.items())},
        "each_condition_each_position_global": 10,
        "each_condition_each_position_per_gpu": 5,
        "no_ln_first_block_count": first_counts["no_ln"],
    }


def audit_test_firewall(tasks: Mapping[str, Any]) -> dict[str, Any]:
    source = (REPO_ROOT / IMPLEMENTATION_REL).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_attributes = [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "test_mask"
    ]
    if forbidden_attributes:
        fail("TEST_FIREWALL", "TEST_MASK_ATTRIBUTE_PRESENT_IN_ANALYSER")
    worker = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "formal_worker"
        ),
        None,
    )
    if worker is None:
        fail("TEST_FIREWALL", "FORMAL_WORKER_MISSING")
    calls = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train_one_task"
    ]
    if len(calls) != 1:
        fail("TEST_FIREWALL", "TRAINING_CALL_COUNT_MISMATCH")
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    value = keywords.get("track_test")
    if not isinstance(value, ast.Constant) or value.value is not False:
        fail("TEST_FIREWALL", "TRACK_TEST_NOT_LITERAL_FALSE")
    if any(row.get("track_test") is not False for row in tasks["tasks"]):
        fail("TEST_FIREWALL", "FROZEN_TASK_TRACK_TEST_NOT_FALSE")
    proxy_calls = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ValidationOnlyDataProxy"
    ]
    if len(proxy_calls) != 2:
        fail("TEST_FIREWALL", "RUNTIME_PROXY_CALL_COUNT_MISMATCH")
    return {
        "schema_version": "layernorm-collapse-test-firewall-v1",
        "passed": True,
        "allowed_data_fields": list(ALLOWED_DATA_FIELDS),
        "test_mask_attribute_count_in_analyser": 0,
        "formal_training_call_count": 1,
        "runtime_proxy_call_count": 2,
        "runtime_proxy_enabled": True,
        "formal_track_test_literal": False,
        "planned_task_count": 360,
        "test_accessed": False,
    }


def _prelaunch_manifest(root: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    records = [file_record(root / relative, relative) for relative in sorted(relative_paths)]
    document: dict[str, Any] = {
        "schema_version": "layernorm-collapse-prelaunch-manifest-v1",
        "experiment": EXPERIMENT,
        "files": records,
    }
    document["manifest_root_sha256"] = commitment(records)
    return document


def verify_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    records = manifest.get("files")
    if not isinstance(records, list):
        fail("MANIFEST", "FILES_MISSING")
    for record in records:
        path = root / record["path"]
        if not path.is_file():
            fail("MANIFEST", "FILE_MISSING", record["path"])
        if sha256_path(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            fail("MANIFEST", "FILE_IDENTITY_MISMATCH", record["path"])
    if commitment(records) != manifest.get("manifest_root_sha256"):
        fail("MANIFEST", "ROOT_COMMITMENT_MISMATCH")


def verify_prelaunch(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(root / "manifest.json", reject_test_metrics=True)
    verify_manifest(root, manifest)
    config = load_config(root / f"source_snapshot/{CONFIG_REL}")
    if resolve_result_root(config) != root.resolve():
        fail("PREFLIGHT", "SNAPSHOT_RESULT_ROOT_MISMATCH")
    source_hashes = read_json(root / "source_hashes.json", reject_test_metrics=True)
    verify_embedded_commitment(source_hashes, "commitment_sha256", "SOURCE")
    if set(source_hashes.get("source_snapshot", {})) != set(SOURCE_RELS):
        fail("SOURCE", "SNAPSHOT_SOURCE_SET_MISMATCH")
    if set(source_hashes.get("dependencies", {})) != set(config["dependency_hashes"]):
        fail("SOURCE", "DEPENDENCY_SET_MISMATCH")
    for relative, record in source_hashes["source_snapshot"].items():
        snapshot = root / "source_snapshot" / relative
        if sha256_path(snapshot) != record["sha256"]:
            fail("SOURCE", "SNAPSHOT_HASH_MISMATCH", relative)
    for relative, expected in source_hashes["dependencies"].items():
        if expected.get("sha256") != config["dependency_hashes"][relative]:
            fail("SOURCE", "FROZEN_DEPENDENCY_HASH_MISMATCH", relative)
        if sha256_path(REPO_ROOT / relative) != expected["sha256"]:
            fail("SOURCE", "LIVE_DEPENDENCY_CHANGED", relative)
    return config, source_hashes


def dry_run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = resolve_result_root(config)
    assert_host(config)
    if root.exists():
        fail("DRY_RUN", "RESULT_ROOT_ALREADY_EXISTS", str(root))
    if tmux_exists():
        fail("DRY_RUN", "TMUX_ALREADY_EXISTS", TMUX_NAME)
    pinned = verify_pinned_files(config)
    selection = freeze_selection(config)
    seeds = freeze_seeds(config)
    conditions = freeze_conditions(config)
    tasks = build_tasks(selection, seeds, conditions, config)
    firewall = audit_test_firewall(tasks)
    gpu = query_gpus(config, require_idle=True)

    source_live: dict[str, Any] = {}
    for relative in SOURCE_RELS:
        path = REPO_ROOT / relative
        if not path.is_file():
            fail("SOURCE", "REQUIRED_SOURCE_MISSING", relative)
        source_live[relative] = file_record(path, relative)
    dependencies = {
        relative: pinned[relative] for relative in config["dependency_hashes"]
    }
    root.mkdir(parents=True, exist_ok=False)
    (root / "formal/task_results").mkdir(parents=True, exist_ok=False)
    try:
        for relative in SOURCE_RELS:
            destination = root / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (REPO_ROOT / relative).open("rb") as source_handle:
                with destination.open("xb") as destination_handle:
                    shutil.copyfileobj(source_handle, destination_handle)
        write_json_exclusive(root / "frozen_selection.json", selection)
        write_json_exclusive(root / "frozen_seeds.json", seeds)
        write_json_exclusive(root / "frozen_conditions.json", conditions)
        write_json_exclusive(root / "frozen_tasks.json", tasks)
        write_json_exclusive(
            root / "assignments.json",
            {
                "schema_version": "layernorm-collapse-assignments-v1",
                "workers": copy.deepcopy(config["workers"]),
                "validation": tasks["validation"],
            },
        )
        write_json_exclusive(root / "test_firewall.json", firewall)
        source_hashes = {
            "schema_version": "layernorm-collapse-source-hashes-v1",
            "source_snapshot": source_live,
            "dependencies": dependencies,
        }
        source_hashes["commitment_sha256"] = commitment(source_hashes)
        write_json_exclusive(root / "source_hashes.json", source_hashes)
        write_json_exclusive(
            root / "preflight.json",
            {
                "schema_version": "layernorm-collapse-preflight-v1",
                "created_at": utc_now(),
                "initial_result_root_absent": True,
                "initial_tmux_absent": True,
                "host": assert_host(config),
                "gpu": gpu,
                "gnn_training_count": 0,
                "test_accessed": False,
            },
        )
        versions = {}
        for package in ("torch", "torch-geometric", "numpy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        write_json_exclusive(
            root / "environment.json",
            {
                "schema_version": "layernorm-collapse-environment-v1",
                "created_at": utc_now(),
                "python": sys.version,
                "platform": platform.platform(),
                "package_versions": versions,
            },
        )
        write_json_exclusive(
            root / "state.json",
            {
                "schema_version": "layernorm-collapse-prelaunch-state-v1",
                "status": "FROZEN_READY",
                "planned_task_count": 360,
                "gnn_training_count": 0,
                "formal_started": False,
                "test_accessed": False,
            },
        )
        prelaunch_paths = [
            "frozen_selection.json",
            "frozen_seeds.json",
            "frozen_conditions.json",
            "frozen_tasks.json",
            "assignments.json",
            "test_firewall.json",
            "source_hashes.json",
            "preflight.json",
            "environment.json",
            "state.json",
            *[f"source_snapshot/{relative}" for relative in SOURCE_RELS],
        ]
        manifest = _prelaunch_manifest(root, prelaunch_paths)
        write_json_exclusive(root / "manifest.json", manifest)
    except BaseException:
        raise
    return {
        "status": "DRY_RUN_VALID_PASS",
        "result_root": str(root),
        "planned_task_count": 360,
        "selection_commitment_sha256": selection["commitment_sha256"],
        "seed_commitment_sha256": seeds["commitment_sha256"],
        "task_commitment_sha256": tasks["commitment_sha256"],
        "manifest_root_sha256": manifest["manifest_root_sha256"],
        "gnn_training_count": 0,
        "test_accessed": False,
    }


def preflight(result_root: Path) -> dict[str, Any]:
    config, source_hashes = verify_prelaunch(result_root)
    verify_pinned_files(config)
    host = assert_host(config)
    if tmux_exists():
        fail("PREFLIGHT", "TMUX_ALREADY_EXISTS", TMUX_NAME)
    formal = result_root / "formal"
    if (formal / "start.json").exists() or (formal / "terminal.json").exists():
        fail("PREFLIGHT", "FORMAL_ARTIFACT_ALREADY_EXISTS")
    task_result_dir = formal / "task_results"
    if not task_result_dir.is_dir() or any(task_result_dir.iterdir()):
        fail("PREFLIGHT", "TASK_RESULT_DIRECTORY_NOT_EMPTY")
    for relative, record in source_hashes["source_snapshot"].items():
        if sha256_path(REPO_ROOT / relative) != record["sha256"]:
            fail("SOURCE", "LIVE_EXPERIMENT_SOURCE_CHANGED", relative)
    tasks = read_json(result_root / "frozen_tasks.json", reject_test_metrics=True)
    verify_embedded_commitment(tasks, "commitment_sha256", "FROZEN_TASKS")
    validate_tasks(tasks, config)
    selection = read_json(
        result_root / "frozen_selection.json", reject_test_metrics=True
    )
    conditions = read_json(
        result_root / "frozen_conditions.json", reject_test_metrics=True
    )
    seeds = read_json(result_root / "frozen_seeds.json", reject_test_metrics=True)
    verify_embedded_commitment(selection, "commitment_sha256", "SELECTION")
    verify_embedded_commitment(conditions, "commitment_sha256", "CONDITIONS")
    validate_frozen_seed_record(seeds, config)
    if canonical_json_bytes(selection) != canonical_json_bytes(freeze_selection(config)):
        fail("PREFLIGHT", "SELECTION_FREEZE_RECOMPUTATION_MISMATCH")
    if canonical_json_bytes(conditions) != canonical_json_bytes(
        freeze_conditions(config)
    ):
        fail("PREFLIGHT", "CONDITION_FREEZE_RECOMPUTATION_MISMATCH")
    if canonical_json_bytes(tasks) != canonical_json_bytes(
        build_tasks(selection, seeds, conditions, config)
    ):
        fail("PREFLIGHT", "TASK_FREEZE_RECOMPUTATION_MISMATCH")
    firewall = audit_test_firewall(tasks)
    if firewall.get("passed") is not True:
        fail("TEST_FIREWALL", "PREFLIGHT_FIREWALL_FAILED")
    gpu = query_gpus(config, require_idle=True)
    return {
        "status": "FORMAL_ADMISSION_PASS",
        "host": host,
        "gpu": gpu,
        "result_root": str(result_root),
        "tmux": TMUX_NAME,
        "planned_task_count": 360,
        "test_accessed": False,
    }


def _ensure_finite_tensor(torch_module: Any, tensor: Any, stage: str, history: list[dict[str, Any]]) -> None:
    if not bool(torch_module.isfinite(tensor).all().detach().cpu().item()):
        raise NonFiniteTrainingError(stage, history)


def train_one_task(
    task: Mapping[str, Any],
    data: Any,
    in_ch: int,
    out_ch: int,
    device: Any,
    *,
    track_test: bool,
) -> dict[str, Any]:
    """Run one validation-only attempt using Phase D's training semantics."""

    if track_test is not False or task.get("track_test") is not False:
        fail("TEST_FIREWALL", "TRAIN_ONE_TASK_TRACK_TEST_NOT_FALSE")
    import torch
    import torch.nn.functional as functional

    from deterministic_runtime import assert_strict_torch_determinism
    from eval_utils import DynamicGNN, isolated_rng

    assert_strict_torch_determinism(torch)
    architecture = copy.deepcopy(task["architecture"])
    hp = copy.deepcopy(task["effective_hp"])
    history: list[dict[str, Any]] = []
    best_validation: float | None = None
    early_stop_best_validation = 0.0
    best_epoch: int | None = None
    best_counts: list[int] | None = None
    no_improve = 0
    validation_count = int(data.val_mask.sum().item())
    if validation_count <= 0:
        fail("TRAIN", "VALIDATION_MASK_EMPTY")
    try:
        with isolated_rng(int(task["seed"]), device):
            model = DynamicGNN(
                architecture,
                in_ch,
                out_ch,
                dropout=float(hp["dropout"]),
                hidden_dim=int(hp["hidden_dim"]),
                gcnii_alpha=0.1,
                gcnii_theta=0.5,
                gat_heads=int(hp.get("gat_heads", 1)),
                sage_aggr=str(hp.get("sage_aggr", "mean")),
                gin_eps=float(hp.get("gin_eps", 0.0)),
                gat_heads_by_layer=hp.get("gat_heads_by_layer"),
                sage_aggr_by_layer=hp.get("sage_aggr_by_layer"),
                gin_eps_by_layer=hp.get("gin_eps_by_layer"),
                component_variant=str(task["component_variant"]),
                activation_variant="relu",
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(hp["lr"]),
                weight_decay=float(hp["l2"]),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(task["max_epochs"]),
                eta_min=float(hp["lr"]) * 0.01,
            )
            for epoch in range(int(task["max_epochs"])):
                model.train()
                optimizer.zero_grad()
                output = model(data.x, data.edge_index)
                _ensure_finite_tensor(torch, output, "training_output", history)
                loss = functional.cross_entropy(
                    output[data.train_mask], data.y[data.train_mask]
                )
                _ensure_finite_tensor(torch, loss, "training_loss", history)
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        _ensure_finite_tensor(
                            torch, parameter.grad, "training_gradient", history
                        )
                optimizer.step()
                for parameter in model.parameters():
                    _ensure_finite_tensor(
                        torch, parameter, "model_parameter", history
                    )
                scheduler.step()

                model.eval()
                with torch.no_grad():
                    validation_output = model(data.x, data.edge_index)
                    _ensure_finite_tensor(
                        torch, validation_output, "validation_output", history
                    )
                    prediction = validation_output.argmax(dim=1)
                    validation_accuracy = float(
                        prediction[data.val_mask]
                        .eq(data.y[data.val_mask])
                        .sum()
                        .item()
                        / validation_count
                    )
                epoch_1based = epoch + 1
                history.append(
                    {
                        "epoch_1based": epoch_1based,
                        "training_loss": float(loss.detach().cpu().item()),
                        "validation_accuracy": validation_accuracy,
                    }
                )
                if best_epoch is None or validation_accuracy > best_validation:
                    # Preserve a diagnostic best even in the all-zero edge case.
                    best_validation = validation_accuracy
                    best_epoch = epoch_1based
                    best_counts = [
                        int(value)
                        for value in torch.bincount(
                            prediction[data.val_mask], minlength=out_ch
                        )
                        .detach()
                        .cpu()
                        .tolist()
                    ]
                if validation_accuracy > early_stop_best_validation:
                    # This separate state exactly mirrors Phase D early stopping.
                    early_stop_best_validation = validation_accuracy
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= int(task["patience"]):
                        break
    except NonFiniteTrainingError as exc:
        exc.best_validation_accuracy = best_validation
        exc.best_epoch = best_epoch
        exc.best_validation_prediction_counts = best_counts
        exc.validation_sample_count = validation_count
        raise
    if best_validation is None or best_epoch is None or best_counts is None:
        fail("TRAIN", "NO_BEST_VALIDATION_MODEL")
    if sum(best_counts) != validation_count:
        fail("TRAIN", "BEST_PREDICTION_COUNT_MISMATCH")
    return {
        "best_validation_accuracy": best_validation,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "history": history,
        "best_validation_prediction_counts": best_counts,
        "validation_sample_count": validation_count,
        "nan_or_inf_detected": False,
        "nan_or_inf_stage": None,
    }


def _task_result_path(root: Path, task_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", task_id):
        fail("FORMAL", "UNSAFE_TASK_ID", task_id)
    return root / "formal/task_results" / f"{task_id}.json"


def _result_identity(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(task[key])
        for key in (
            "task_id",
            "block_id",
            "block_index",
            "config_index",
            "selection_group",
            "config_id",
            "joint_config_fingerprint",
            "architecture_fingerprint",
            "hp_fingerprint",
            "seed",
            "condition_id",
            "condition_label",
            "component_variant",
            "execution_position",
            "williams_row",
            "worker_sequence",
            "worker_id",
            "gpu_index",
            "gpu_uuid",
            "effective_hp",
            "max_epochs",
            "patience",
            "track_test",
        )
    }


def formal_worker(payload: Mapping[str, Any]) -> None:
    worker_id = int(payload["worker_id"])
    gpu_index = int(payload["gpu_index"])
    gpu_uuid = str(payload["gpu_uuid"])
    root = Path(payload["result_root"])
    abort_event = payload["abort_event"]
    tasks = sorted(
        [dict(row) for row in payload["tasks"]], key=lambda row: int(row["worker_sequence"])
    )
    shard_path = root / f"formal/shards/worker_{worker_id}.json"
    shard: dict[str, Any] = {
        "schema_version": "layernorm-collapse-worker-shard-v1",
        "worker_id": worker_id,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "status": "STARTING",
        "planned_task_count": len(tasks),
        "completed_task_count": 0,
        "failed_task_count": 0,
        "current_task_id": None,
        "last_completed_task_id": None,
        "allowed_data_request_count": 0,
        "forbidden_data_request_count": 0,
        "test_accessed": False,
    }
    try:
        import torch
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures

        gpu_evidence = query_gpus(payload["config"], require_idle=False)
        match = next((row for row in gpu_evidence["gpus"] if row["index"] == gpu_index), None)
        if match is None or match["uuid"] != gpu_uuid:
            fail("WORKER", "GPU_BINDING_MISMATCH")
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
        dataset = Planetoid(
            root=str(REPO_ROOT / "Cora"),
            name="Cora",
            split="public",
            transform=NormalizeFeatures(),
        )
        loaded = dataset[0]
        setup_ledger = new_data_access_ledger()
        setup_proxy = ValidationOnlyDataProxy(loaded, setup_ledger)
        tensors = {
            field: getattr(setup_proxy, field).to(device)
            for field in ALLOWED_DATA_FIELDS
        }
        setup_access = data_access_ledger(setup_proxy)
        if setup_access["forbidden_request_count"] or setup_access["test_accessed"]:
            fail("TEST_FIREWALL", "SETUP_DATA_ACCESS_VIOLATION")
        data_bundle = SimpleNamespace(**tensors)
        shard["setup_data_access_ledger"] = setup_access
        shard["allowed_data_request_count"] += setup_access["allowed_request_count"]
        shard["forbidden_data_request_count"] += setup_access[
            "forbidden_request_count"
        ]
        shard["test_accessed"] = setup_access["test_accessed"]
        del setup_proxy
        del loaded
        del dataset
        in_ch = int(tensors["x"].shape[1])
        out_ch = int(tensors["y"].max().item()) + 1
        validation_sample_count = int(tensors["val_mask"].sum().item())
        shard["status"] = "RUNNING"
        write_json_atomic(shard_path, shard)
        for task in tasks:
            if abort_event.is_set():
                break
            if (
                int(task["worker_id"]) != worker_id
                or int(task["gpu_index"]) != gpu_index
                or str(task["gpu_uuid"]) != gpu_uuid
                or task.get("track_test") is not False
            ):
                fail("WORKER", "TASK_BINDING_MISMATCH", task["task_id"])
            shard["current_task_id"] = task["task_id"]
            write_json_atomic(shard_path, shard)
            started = time.monotonic()
            identity = _result_identity(task)
            task_ledger = new_data_access_ledger()
            data_proxy = ValidationOnlyDataProxy(data_bundle, task_ledger)
            try:
                diagnostic = train_one_task(
                    task,
                    data_proxy,
                    in_ch,
                    out_ch,
                    device,
                    track_test=False,
                )
                task_access = data_access_ledger(data_proxy)
                if (
                    task_access["forbidden_request_count"]
                    or task_access["test_accessed"]
                ):
                    fail("TEST_FIREWALL", "TASK_DATA_ACCESS_VIOLATION")
                result_record = {
                    "schema_version": "layernorm-collapse-task-result-v1",
                    **identity,
                    **diagnostic,
                    "status": "COMPLETE",
                    "attempt_count": 1,
                    "elapsed_seconds": time.monotonic() - started,
                    "data_access_ledger": task_access,
                    "test_accessed": task_access["test_accessed"],
                }
                write_json_exclusive(_task_result_path(root, task["task_id"]), result_record)
                shard["completed_task_count"] += 1
                shard["last_completed_task_id"] = task["task_id"]
            except BaseException as exc:
                task_access = data_access_ledger(data_proxy)
                partial_history = (
                    [dict(row) for row in exc.history]
                    if isinstance(exc, NonFiniteTrainingError)
                    else []
                )
                failure_record = {
                    "schema_version": "layernorm-collapse-task-result-v1",
                    **identity,
                    "status": "FAILED",
                    "attempt_count": 1,
                    "elapsed_seconds": time.monotonic() - started,
                    "history": partial_history,
                    "best_validation_accuracy": (
                        exc.best_validation_accuracy
                        if isinstance(exc, NonFiniteTrainingError)
                        else None
                    ),
                    "best_epoch": (
                        exc.best_epoch if isinstance(exc, NonFiniteTrainingError) else None
                    ),
                    "epochs_ran": len(partial_history),
                    "best_validation_prediction_counts": (
                        exc.best_validation_prediction_counts
                        if isinstance(exc, NonFiniteTrainingError)
                        else None
                    ),
                    "validation_sample_count": (
                        exc.validation_sample_count
                        if isinstance(exc, NonFiniteTrainingError)
                        else validation_sample_count
                    ),
                    "nan_or_inf_detected": isinstance(exc, NonFiniteTrainingError),
                    "nan_or_inf_stage": exc.stage if isinstance(exc, NonFiniteTrainingError) else None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "data_access_ledger": task_access,
                    "test_accessed": task_access["test_accessed"],
                }
                write_json_exclusive(_task_result_path(root, task["task_id"]), failure_record)
                shard["failed_task_count"] += 1
                shard["first_error"] = {
                    "task_id": task["task_id"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                shard["status"] = "FAILED"
                abort_event.set()
                break
            finally:
                task_access = data_access_ledger(data_proxy)
                shard["allowed_data_request_count"] += task_access[
                    "allowed_request_count"
                ]
                shard["forbidden_data_request_count"] += task_access[
                    "forbidden_request_count"
                ]
                shard["test_accessed"] = bool(
                    shard["test_accessed"] or task_access["test_accessed"]
                )
                shard["current_task_id"] = None
                write_json_atomic(shard_path, shard)
        if shard["status"] == "RUNNING":
            shard["status"] = (
                "COMPLETE"
                if shard["completed_task_count"] == shard["planned_task_count"]
                else "STOPPED_BY_PEER_FAILURE"
            )
    except BaseException as exc:
        shard["status"] = "FAILED"
        shard["first_error"] = {
            "task_id": shard.get("current_task_id"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        shard["failed_task_count"] = max(1, int(shard["failed_task_count"]))
        abort_event.set()
    finally:
        shard["current_task_id"] = None
        try:
            write_json_atomic(shard_path, shard)
        except BaseException:
            os._exit(3)


def _load_task_results(root: Path) -> list[dict[str, Any]]:
    result_dir = root / "formal/task_results"
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*.json")):
        row = read_json(path, reject_test_metrics=True)
        if not isinstance(row, dict):
            fail("FORMAL", "TASK_RESULT_SCHEMA_MISMATCH", path.name)
        rows.append(row)
    return rows


def is_collapse_result(row: Mapping[str, Any]) -> bool:
    counts = row.get("best_validation_prediction_counts")
    if not isinstance(counts, list) or not counts:
        return False
    total = sum(int(value) for value in counts)
    return (
        total == int(row.get("validation_sample_count", -1))
        and max(int(value) for value in counts) == total
        and row.get("best_validation_accuracy") == 0.316
    )


def compute_statistics(
    results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    if len(results) != 360 or any(row.get("status") != "COMPLETE" for row in results):
        fail("STATISTICS", "COMPLETE_RESULT_COUNT_MISMATCH")
    by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in results:
        key = (
            str(row["joint_config_fingerprint"]),
            int(row["seed"]),
            str(row["condition_id"]),
        )
        if key in by_key:
            fail("STATISTICS", "DUPLICATE_PAIRED_RESULT")
        by_key[key] = row
    fingerprints = sorted({str(row["joint_config_fingerprint"]) for row in results})
    seeds = list(DIAGNOSTIC_SEEDS)
    baseline_rows = [row for row in results if row["condition_id"] == "no_ln"]
    if len(baseline_rows) != 60:
        fail("STATISTICS", "BASELINE_COUNT_MISMATCH")
    baseline_collapse = sum(is_collapse_result(row) for row in baseline_rows)
    condition_config = {row["id"]: row for row in config["conditions"]}
    summaries: list[dict[str, Any]] = []
    for condition_id in LN_CONDITION_IDS:
        deltas: list[float] = []
        collapse_count = 0
        config_noninferior_count = 0
        config_rows: list[dict[str, Any]] = []
        for fingerprint in fingerprints:
            baseline_values: list[float] = []
            condition_values: list[float] = []
            for seed in seeds:
                baseline = by_key.get((fingerprint, seed, "no_ln"))
                condition = by_key.get((fingerprint, seed, condition_id))
                if baseline is None or condition is None:
                    fail("STATISTICS", "PAIR_MISSING", f"{fingerprint}:{seed}:{condition_id}")
                baseline_value = float(baseline["best_validation_accuracy"])
                condition_value = float(condition["best_validation_accuracy"])
                baseline_values.append(baseline_value)
                condition_values.append(condition_value)
                deltas.append(condition_value - baseline_value)
                collapse_count += int(is_collapse_result(condition))
            baseline_mean = math.fsum(baseline_values) / 5
            condition_mean = math.fsum(condition_values) / 5
            noninferior = condition_mean >= baseline_mean
            config_noninferior_count += int(noninferior)
            config_rows.append(
                {
                    "joint_config_fingerprint": fingerprint,
                    "no_ln_mean": baseline_mean,
                    "condition_mean": condition_mean,
                    "delta": condition_mean - baseline_mean,
                    "noninferior": noninferior,
                }
            )
        mean_delta = math.fsum(deltas) / 60
        eligible = (
            len(deltas) == 60
            and collapse_count <= 3
            and collapse_count <= baseline_collapse + 1
            and mean_delta >= 0.0
            and config_noninferior_count >= 8
        )
        summaries.append(
            {
                "condition_id": condition_id,
                "completed_pair_count": len(deltas),
                "collapse_count": collapse_count,
                "no_ln_collapse_count": baseline_collapse,
                "collapse_excess_over_no_ln": collapse_count - baseline_collapse,
                "mean_paired_delta": mean_delta,
                "config_mean_noninferior_count": config_noninferior_count,
                "config_summaries": config_rows,
                "stable_recipe_priority": condition_config[condition_id]["stable_recipe_priority"],
                "eligible": eligible,
            }
        )
    eligible = [row for row in summaries if row["eligible"]]
    winner = None
    if eligible:
        winner = sorted(
            eligible,
            key=lambda row: (
                -row["mean_paired_delta"],
                row["collapse_count"],
                row["stable_recipe_priority"],
            ),
        )[0]["condition_id"]
    document: dict[str, Any] = {
        "schema_version": "layernorm-collapse-statistics-v1",
        "result_count": 360,
        "paired_block_count": 60,
        "no_ln_collapse_count": baseline_collapse,
        "collapse_definition": copy.deepcopy(config["collapse_definition"]),
        "stable_recipe_gate": copy.deepcopy(config["stable_recipe_gate"]),
        "condition_summaries": summaries,
        "eligible_condition_ids": [row["condition_id"] for row in eligible],
        "selected_condition_id": winner,
    }
    document["commitment_sha256"] = commitment(document)
    return document


def build_decision(
    statistics: Mapping[str, Any],
    conditions: Mapping[str, Any],
    seeds: Mapping[str, Any],
    source_binding_sha256: str,
) -> dict[str, Any]:
    selected_id = statistics.get("selected_condition_id")
    classification = (
        "STABLE_LN_RECIPE_SELECTED" if selected_id is not None else "NO_STABLE_LN_RECIPE"
    )
    condition_by_id = {row["id"]: row for row in conditions["conditions"]}
    selected_recipe = copy.deepcopy(condition_by_id[selected_id]) if selected_id else None
    document: dict[str, Any] = {
        "schema_version": "layernorm-collapse-decision-v1",
        "state": "COMPLETE",
        "classification": classification,
        "selected_recipe": selected_recipe,
        "selected_condition_id": selected_id,
        "eligible_condition_ids": list(statistics["eligible_condition_ids"]),
        "statistics_commitment_sha256": statistics["commitment_sha256"],
        "reserved_confirmation_seeds": list(seeds["reserved_confirmation_seeds"]),
        "seed_commitment_sha256": seeds["commitment_sha256"],
        "seed_global_inventory_attestation_sha256": seeds[
            "global_inventory_attestation_sha256"
        ],
        "source_binding_sha256": source_binding_sha256,
        "next_phase_started": False,
        "test_accessed": False,
    }
    document["commitment_sha256"] = commitment(document)
    return document


def _attempt_ledger(
    frozen_tasks: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result_by_id = {str(row["task_id"]): row for row in results}
    attempts = []
    for task in frozen_tasks:
        result = result_by_id.get(str(task["task_id"]))
        attempts.append(
            {
                "task_id": task["task_id"],
                "seed": task["seed"],
                "condition_id": task["condition_id"],
                "worker_id": task["worker_id"],
                "gpu_index": task["gpu_index"],
                "gpu_uuid": task["gpu_uuid"],
                "attempt": 1 if result is not None else 0,
                "status": result.get("status") if result is not None else "NOT_RUN",
                "nan_or_inf_detected": result.get("nan_or_inf_detected") if result is not None else None,
                "test_accessed": (
                    result.get("test_accessed") if result is not None else False
                ),
            }
        )
    return {
        "schema_version": "layernorm-collapse-attempt-ledger-v1",
        "planned_task_count": 360,
        "physical_attempt_count": sum(row["attempt"] for row in attempts),
        "retry_count": 0,
        "attempts": attempts,
        "test_accessed": any(row["test_accessed"] is not False for row in attempts),
    }


def _task_ledger(
    frozen_tasks: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result_by_id = {str(row["task_id"]): row for row in results}
    rows = []
    for task in frozen_tasks:
        result = result_by_id.get(str(task["task_id"]))
        rows.append(
            {
                "task_id": task["task_id"],
                "status": result.get("status") if result is not None else "NOT_RUN",
                "seed": task["seed"],
                "condition_id": task["condition_id"],
                "worker_id": task["worker_id"],
                "gpu_index": task["gpu_index"],
                "test_accessed": (
                    result.get("test_accessed") if result is not None else False
                ),
            }
        )
    counts = Counter(row["status"] for row in rows)
    return {
        "schema_version": "layernorm-collapse-task-ledger-v1",
        "planned_task_count": 360,
        "completed_task_count": counts["COMPLETE"],
        "failed_task_count": counts["FAILED"],
        "not_run_task_count": counts["NOT_RUN"],
        "tasks": rows,
        "test_accessed": any(row["test_accessed"] is not False for row in rows),
    }


def _test_access_ledger(
    results: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    violations = [row["task_id"] for row in results if row.get("test_accessed") is not False]
    worker_violations = [
        int(row["worker_id"])
        for row in shards
        if row.get("test_accessed") is not False
    ]
    forbidden_requests = (
        sum(int(row.get("forbidden_data_request_count", 0)) for row in shards)
        if shards
        else sum(
            int(row.get("data_access_ledger", {}).get("forbidden_request_count", 0))
            for row in results
        )
    )
    return {
        "schema_version": "layernorm-collapse-test-access-ledger-v1",
        "observed_task_result_count": len(results),
        "observed_worker_shard_count": len(shards),
        "violation_task_ids": violations,
        "violation_worker_ids": worker_violations,
        "formal_track_test_literal": False,
        "runtime_proxy_enabled": True,
        "forbidden_request_count": forbidden_requests,
        "test_accessed": bool(violations or worker_violations),
    }


def _formal_manifest(root: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    records = [file_record(root / relative, relative) for relative in sorted(relative_paths)]
    task_records = [
        file_record(path, str(path.relative_to(root)))
        for path in sorted((root / "formal/task_results").glob("*.json"))
    ]
    document: dict[str, Any] = {
        "schema_version": "layernorm-collapse-formal-manifest-v1",
        "experiment": EXPERIMENT,
        "files": records,
        "task_result_file_count": len(task_records),
        "task_result_files_commitment_sha256": commitment(task_records),
    }
    document["manifest_root_sha256"] = commitment(
        {"files": records, "task_result_files": task_records}
    )
    return document


def _write_failure_terminal(
    root: Path,
    run_id: str,
    frozen_tasks: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    first_error: Mapping[str, Any],
) -> None:
    formal = root / "formal"
    attempt_ledger = _attempt_ledger(frozen_tasks, results)
    task_ledger = _task_ledger(frozen_tasks, results)
    shards = []
    for path in sorted((formal / "shards").glob("worker_*.json")):
        try:
            shard = read_json(path, reject_test_metrics=True)
        except BaseException:
            continue
        if isinstance(shard, dict):
            shards.append(shard)
    test_ledger = _test_access_ledger(results, shards)
    for name, document in (
        ("attempt_ledger.json", attempt_ledger),
        ("task_ledger.json", task_ledger),
        ("test_access_ledger.json", test_ledger),
    ):
        if not (formal / name).exists():
            write_json_exclusive(formal / name, document)
    write_json_atomic(
        formal / "state.json",
        {
            "status": "FAILED",
            "classification": "TERMINAL_FAILURE",
            "run_id": run_id,
            "test_accessed": test_ledger["test_accessed"],
        },
    )
    if not (formal / "terminal.json").exists():
        write_json_exclusive(
            formal / "terminal.json",
            {
                "schema_version": "layernorm-collapse-terminal-v1",
                "state": "FAILED",
                "classification": "TERMINAL_FAILURE",
                "run_id": run_id,
                "completed_task_count": task_ledger["completed_task_count"],
                "failed_task_count": task_ledger["failed_task_count"],
                "not_run_task_count": task_ledger["not_run_task_count"],
                "first_error": dict(first_error),
                "retry_count": 0,
                "test_accessed": test_ledger["test_accessed"],
                "finished_at": utc_now(),
            },
        )


def formal_run(result_root: Path, run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        fail("FORMAL", "INVALID_RUN_ID")
    config, source_hashes = verify_prelaunch(result_root)
    verify_pinned_files(config)
    assert_host(config)
    formal = result_root / "formal"
    if (formal / "start.json").exists() or (formal / "terminal.json").exists():
        fail("FORMAL", "FORMAL_ALREADY_STARTED")
    if any((formal / "task_results").iterdir()):
        fail("FORMAL", "TASK_RESULT_DIRECTORY_NOT_EMPTY")
    gpu = query_gpus(config, require_idle=True)
    frozen_task_doc = read_json(result_root / "frozen_tasks.json", reject_test_metrics=True)
    validate_tasks(frozen_task_doc, config)
    frozen_tasks = frozen_task_doc["tasks"]
    frozen_seed_doc = read_json(
        result_root / "frozen_seeds.json", reject_test_metrics=True
    )
    validate_frozen_seed_record(frozen_seed_doc, config)
    for relative, record in source_hashes["source_snapshot"].items():
        if sha256_path(REPO_ROOT / relative) != record["sha256"]:
            fail("SOURCE", "LIVE_EXPERIMENT_SOURCE_CHANGED", relative)
    start = {
        "schema_version": "layernorm-collapse-formal-start-v1",
        "run_id": run_id,
        "started_at": utc_now(),
        "result_root": str(result_root),
        "tmux_name": TMUX_NAME,
        "planned_task_count": 360,
        "worker_count": 4,
        "gpu": gpu,
        "task_commitment_sha256": frozen_task_doc["commitment_sha256"],
        "seed_commitment_sha256": frozen_seed_doc["commitment_sha256"],
        "seed_global_inventory_attestation_sha256": frozen_seed_doc[
            "global_inventory_attestation_sha256"
        ],
        "source_hashes_commitment_sha256": source_hashes["commitment_sha256"],
        "track_test": False,
        "test_accessed": False,
    }
    write_json_exclusive(formal / "start.json", start)
    source_binding = {
        "schema_version": "layernorm-collapse-formal-source-binding-v1",
        "run_id": run_id,
        "source_hashes_commitment_sha256": source_hashes["commitment_sha256"],
        "executed_implementation": file_record(Path(__file__).resolve(), IMPLEMENTATION_REL),
        "source_snapshot": copy.deepcopy(source_hashes["source_snapshot"]),
        "dependencies": copy.deepcopy(source_hashes["dependencies"]),
        "test_accessed": False,
    }
    source_binding["commitment_sha256"] = commitment(source_binding)
    write_json_exclusive(formal / "source_binding.json", source_binding)
    write_json_exclusive(
        formal / "state.json",
        {
            "status": "RUNNING",
            "run_id": run_id,
            "planned_task_count": 360,
            "test_accessed": False,
        },
    )
    context = multiprocessing.get_context("spawn")
    abort_event = context.Event()
    processes = []
    try:
        for mapping in config["workers"]["mapping"]:
            worker_id = int(mapping["worker_id"])
            worker_tasks = [row for row in frozen_tasks if int(row["worker_id"]) == worker_id]
            process = context.Process(
                target=formal_worker,
                args=(
                    {
                        "worker_id": worker_id,
                        "gpu_index": int(mapping["gpu_index"]),
                        "gpu_uuid": str(mapping["gpu_uuid"]),
                        "result_root": str(result_root),
                        "tasks": worker_tasks,
                        "config": copy.deepcopy(config),
                        "abort_event": abort_event,
                    },
                ),
                name=f"layernorm_diag_worker_{worker_id}",
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        results = _load_task_results(result_root)
        shard_docs = []
        for worker_id in range(4):
            path = formal / f"shards/worker_{worker_id}.json"
            if path.exists():
                shard_docs.append(read_json(path, reject_test_metrics=True))
        failed_shards = [
            row
            for row in shard_docs
            if row.get("status") != "COMPLETE"
            or row.get("test_accessed") is not False
            or row.get("forbidden_data_request_count") != 0
        ]
        nonzero = [process for process in processes if process.exitcode != 0]
        incomplete = len(results) != 360 or any(row.get("status") != "COMPLETE" for row in results)
        if nonzero or failed_shards or incomplete:
            first_error: dict[str, Any] = {
                "error_type": "INCOMPLETE_FORMAL_EXECUTION",
                "error_message": "formal workers did not produce 360 complete one-attempt results",
            }
            for shard in sorted(shard_docs, key=lambda row: int(row.get("worker_id", 999))):
                if shard.get("first_error"):
                    first_error = dict(shard["first_error"])
                    break
            _write_failure_terminal(result_root, run_id, frozen_tasks, results, first_error)
            fail("FORMAL", "TERMINAL_FAILURE", first_error.get("error_message"))

        results = sorted(results, key=lambda row: str(row["task_id"]))
        attempt_ledger = _attempt_ledger(frozen_tasks, results)
        task_ledger = _task_ledger(frozen_tasks, results)
        test_ledger = _test_access_ledger(results, shard_docs)
        if (
            attempt_ledger["physical_attempt_count"] != 360
            or attempt_ledger["retry_count"] != 0
            or task_ledger["completed_task_count"] != 360
            or task_ledger["failed_task_count"] != 0
            or task_ledger["not_run_task_count"] != 0
            or test_ledger["test_accessed"] is not False
        ):
            fail("FORMAL", "FINAL_LEDGER_GATE_FAILED")
        statistics = compute_statistics(results, config)
        seeds = read_json(result_root / "frozen_seeds.json", reject_test_metrics=True)
        conditions = read_json(result_root / "frozen_conditions.json", reject_test_metrics=True)
        decision = build_decision(
            statistics,
            conditions,
            seeds,
            source_binding["commitment_sha256"],
        )
        write_json_exclusive(
            formal / "results.json",
            {
                "schema_version": "layernorm-collapse-results-v1",
                "run_id": run_id,
                "result_count": 360,
                "results": results,
                "test_accessed": False,
            },
        )
        write_json_exclusive(formal / "attempt_ledger.json", attempt_ledger)
        write_json_exclusive(formal / "task_ledger.json", task_ledger)
        write_json_exclusive(formal / "test_access_ledger.json", test_ledger)
        write_json_exclusive(formal / "statistics.json", statistics)
        write_json_exclusive(formal / "decision.json", decision)
        write_json_atomic(
            formal / "state.json",
            {
                "status": "COMPLETE",
                "classification": decision["classification"],
                "run_id": run_id,
                "completed_task_count": 360,
                "test_accessed": False,
            },
        )
        formal_paths = [
            "formal/start.json",
            "formal/source_binding.json",
            "formal/results.json",
            "formal/attempt_ledger.json",
            "formal/task_ledger.json",
            "formal/test_access_ledger.json",
            "formal/statistics.json",
            "formal/decision.json",
            "formal/state.json",
            *[f"formal/shards/worker_{worker_id}.json" for worker_id in range(4)],
        ]
        manifest = _formal_manifest(result_root, formal_paths)
        write_json_exclusive(formal / "manifest.json", manifest)
        terminal = {
            "schema_version": "layernorm-collapse-terminal-v1",
            "state": "COMPLETE",
            "classification": decision["classification"],
            "run_id": run_id,
            "planned_task_count": 360,
            "completed_task_count": 360,
            "failed_task_count": 0,
            "physical_attempt_count": 360,
            "retry_count": 0,
            "selected_condition_id": decision["selected_condition_id"],
            "decision_commitment_sha256": decision["commitment_sha256"],
            "source_binding_sha256": source_binding["commitment_sha256"],
            "seed_commitment_sha256": seeds["commitment_sha256"],
            "seed_global_inventory_attestation_sha256": seeds[
                "global_inventory_attestation_sha256"
            ],
            "formal_manifest_root_sha256": manifest["manifest_root_sha256"],
            "next_phase_started": False,
            "test_accessed": False,
            "finished_at": utc_now(),
        }
        write_json_exclusive(formal / "terminal.json", terminal)
        return terminal
    except BaseException as exc:
        abort_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        if not (formal / "terminal.json").exists():
            results = _load_task_results(result_root)
            first_error = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            _write_failure_terminal(result_root, run_id, frozen_tasks, results, first_error)
        raise


def verify_formal_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    records = manifest.get("files")
    if not isinstance(records, list):
        fail("VERIFY", "FORMAL_MANIFEST_FILES_MISSING")
    for record in records:
        path = root / record["path"]
        if not path.is_file() or sha256_path(path) != record["sha256"]:
            fail("VERIFY", "FORMAL_MANIFEST_FILE_MISMATCH", record["path"])
        if path.stat().st_size != record["size_bytes"]:
            fail("VERIFY", "FORMAL_MANIFEST_SIZE_MISMATCH", record["path"])
    task_records = [
        file_record(path, str(path.relative_to(root)))
        for path in sorted((root / "formal/task_results").glob("*.json"))
    ]
    if len(task_records) != manifest.get("task_result_file_count"):
        fail("VERIFY", "TASK_RESULT_FILE_COUNT_MISMATCH")
    if commitment(task_records) != manifest.get("task_result_files_commitment_sha256"):
        fail("VERIFY", "TASK_RESULT_FILE_COMMITMENT_MISMATCH")
    if commitment({"files": records, "task_result_files": task_records}) != manifest.get(
        "manifest_root_sha256"
    ):
        fail("VERIFY", "FORMAL_MANIFEST_ROOT_MISMATCH")


def _validate_complete_result(row: Mapping[str, Any], task: Mapping[str, Any]) -> None:
    if row.get("status") != "COMPLETE" or row.get("attempt_count") != 1:
        fail("VERIFY", "TASK_RESULT_NOT_SINGLE_COMPLETE", task["task_id"])
    if row.get("track_test") is not False or row.get("test_accessed") is not False:
        fail("TEST_FIREWALL", "COMPLETE_RESULT_TEST_FLAG_MISMATCH", task["task_id"])
    if row.get("nan_or_inf_detected") is not False or row.get("nan_or_inf_stage") is not None:
        fail("VERIFY", "NONFINITE_FLAG_ON_COMPLETE_RESULT", task["task_id"])
    access = row.get("data_access_ledger")
    if (
        not isinstance(access, dict)
        or access.get("forbidden_request_count") != 0
        or access.get("test_accessed") is not False
        or int(access.get("allowed_request_count", 0)) <= 0
    ):
        fail("TEST_FIREWALL", "COMPLETE_RESULT_DATA_ACCESS_VIOLATION", task["task_id"])
    for field, value in _result_identity(task).items():
        if canonical_json_bytes(row.get(field)) != canonical_json_bytes(value):
            fail("VERIFY", "TASK_RESULT_IDENTITY_MISMATCH", f"{task['task_id']}:{field}")
    history = row.get("history")
    if not isinstance(history, list) or len(history) != row.get("epochs_ran") or not history:
        fail("VERIFY", "TRAJECTORY_LENGTH_MISMATCH", task["task_id"])
    for index, epoch in enumerate(history, start=1):
        if set(epoch) != {"epoch_1based", "training_loss", "validation_accuracy"}:
            fail("VERIFY", "TRAJECTORY_SCHEMA_MISMATCH", task["task_id"])
        if epoch["epoch_1based"] != index:
            fail("VERIFY", "TRAJECTORY_EPOCH_MISMATCH", task["task_id"])
        if not math.isfinite(float(epoch["training_loss"])) or not math.isfinite(
            float(epoch["validation_accuracy"])
        ):
            fail("VERIFY", "NONFINITE_TRAJECTORY", task["task_id"])
    best_epoch = int(row["best_epoch"])
    if not 1 <= best_epoch <= len(history):
        fail("VERIFY", "BEST_EPOCH_OUT_OF_RANGE", task["task_id"])
    if float(row["best_validation_accuracy"]) != float(history[best_epoch - 1]["validation_accuracy"]):
        fail("VERIFY", "BEST_ACCURACY_EPOCH_MISMATCH", task["task_id"])
    if any(
        float(epoch["validation_accuracy"]) > float(row["best_validation_accuracy"])
        for epoch in history
    ):
        fail("VERIFY", "BEST_ACCURACY_NOT_MAXIMUM", task["task_id"])
    counts = row.get("best_validation_prediction_counts")
    if not isinstance(counts, list) or any(type(value) is not int or value < 0 for value in counts):
        fail("VERIFY", "PREDICTION_COUNT_SCHEMA_MISMATCH", task["task_id"])
    if sum(counts) != row.get("validation_sample_count"):
        fail("VERIFY", "PREDICTION_COUNT_TOTAL_MISMATCH", task["task_id"])


def verify_complete(root: Path) -> dict[str, Any]:
    config, source_hashes = verify_prelaunch(root)
    verify_pinned_files(config)
    terminal = read_json(root / "formal/terminal.json", reject_test_metrics=True)
    if terminal.get("state") != "COMPLETE":
        fail("VERIFY", "TERMINAL_NOT_COMPLETE")
    frozen_task_doc = read_json(root / "frozen_tasks.json", reject_test_metrics=True)
    verify_embedded_commitment(
        frozen_task_doc, "commitment_sha256", "FROZEN_TASKS"
    )
    validate_tasks(frozen_task_doc, config)
    tasks = frozen_task_doc["tasks"]
    task_by_id = {str(row["task_id"]): row for row in tasks}
    results_doc = read_json(root / "formal/results.json", reject_test_metrics=True)
    results = results_doc.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 360
        or results_doc.get("result_count") != 360
        or results_doc.get("test_accessed") is not False
    ):
        fail("VERIFY", "RESULT_COUNT_MISMATCH")
    if {str(row["task_id"]) for row in results} != set(task_by_id):
        fail("VERIFY", "RESULT_TASK_SET_MISMATCH")
    for row in results:
        _validate_complete_result(row, task_by_id[str(row["task_id"])])
    task_files = _load_task_results(root)
    if canonical_json_bytes(sorted(task_files, key=lambda row: row["task_id"])) != canonical_json_bytes(
        sorted(results, key=lambda row: row["task_id"])
    ):
        fail("VERIFY", "AGGREGATED_RESULTS_MISMATCH_TASK_FILES")
    attempt_ledger = read_json(root / "formal/attempt_ledger.json", reject_test_metrics=True)
    task_ledger = read_json(root / "formal/task_ledger.json", reject_test_metrics=True)
    test_ledger = read_json(root / "formal/test_access_ledger.json", reject_test_metrics=True)
    shard_docs = [
        read_json(
            root / f"formal/shards/worker_{worker_id}.json",
            reject_test_metrics=True,
        )
        for worker_id in range(4)
    ]
    if any(
        row.get("status") != "COMPLETE"
        or row.get("test_accessed") is not False
        or row.get("forbidden_data_request_count") != 0
        or row.get("setup_data_access_ledger", {}).get("test_accessed") is not False
        or row.get("setup_data_access_ledger", {}).get("forbidden_request_count") != 0
        for row in shard_docs
    ):
        fail("TEST_FIREWALL", "WORKER_DATA_ACCESS_LEDGER_GATE_FAILED")
    expected_test_ledger = _test_access_ledger(results, shard_docs)
    expected_attempt_ledger = _attempt_ledger(tasks, results)
    expected_task_ledger = _task_ledger(tasks, results)
    if (
        attempt_ledger.get("physical_attempt_count") != 360
        or attempt_ledger.get("retry_count") != 0
        or len(attempt_ledger.get("attempts", [])) != 360
        or any(row.get("attempt") != 1 for row in attempt_ledger["attempts"])
        or task_ledger.get("planned_task_count") != 360
        or task_ledger.get("completed_task_count") != 360
        or task_ledger.get("failed_task_count") != 0
        or task_ledger.get("not_run_task_count") != 0
        or test_ledger.get("test_accessed") is not False
        or test_ledger.get("violation_task_ids") != []
        or canonical_json_bytes(attempt_ledger)
        != canonical_json_bytes(expected_attempt_ledger)
        or canonical_json_bytes(task_ledger) != canonical_json_bytes(expected_task_ledger)
        or canonical_json_bytes(test_ledger) != canonical_json_bytes(expected_test_ledger)
    ):
        fail("VERIFY", "LEDGER_GATE_FAILED")
    expected_statistics = compute_statistics(results, config)
    statistics = read_json(root / "formal/statistics.json", reject_test_metrics=True)
    if canonical_json_bytes(statistics) != canonical_json_bytes(expected_statistics):
        fail("VERIFY", "STATISTICS_RECOMPUTATION_MISMATCH")
    seeds = read_json(root / "frozen_seeds.json", reject_test_metrics=True)
    validate_frozen_seed_record(seeds, config)
    conditions = read_json(root / "frozen_conditions.json", reject_test_metrics=True)
    verify_embedded_commitment(conditions, "commitment_sha256", "CONDITIONS")
    selection = read_json(root / "frozen_selection.json", reject_test_metrics=True)
    verify_embedded_commitment(selection, "commitment_sha256", "SELECTION")
    expected_conditions = freeze_conditions(config)
    if canonical_json_bytes(conditions) != canonical_json_bytes(expected_conditions):
        fail("VERIFY", "CONDITION_FREEZE_RECOMPUTATION_MISMATCH")
    expected_selection = freeze_selection(config)
    if canonical_json_bytes(selection) != canonical_json_bytes(expected_selection):
        fail("VERIFY", "SELECTION_FREEZE_RECOMPUTATION_MISMATCH")
    expected_tasks = build_tasks(selection, seeds, conditions, config)
    if canonical_json_bytes(frozen_task_doc) != canonical_json_bytes(expected_tasks):
        fail("VERIFY", "TASK_FREEZE_RECOMPUTATION_MISMATCH")
    source_binding = read_json(root / "formal/source_binding.json", reject_test_metrics=True)
    start = read_json(root / "formal/start.json", reject_test_metrics=True)
    run_ids = {
        terminal.get("run_id"),
        results_doc.get("run_id"),
        source_binding.get("run_id"),
        start.get("run_id"),
    }
    if len(run_ids) != 1 or None in run_ids:
        fail("VERIFY", "RUN_ID_BINDING_MISMATCH")
    if (
        start.get("planned_task_count") != 360
        or start.get("track_test") is not False
        or start.get("test_accessed") is not False
        or start.get("task_commitment_sha256")
        != frozen_task_doc["commitment_sha256"]
        or start.get("seed_commitment_sha256") != seeds["commitment_sha256"]
        or start.get("seed_global_inventory_attestation_sha256")
        != seeds["global_inventory_attestation_sha256"]
        or start.get("source_hashes_commitment_sha256")
        != source_hashes["commitment_sha256"]
    ):
        fail("VERIFY", "FORMAL_START_BINDING_MISMATCH")
    if source_binding.get("source_hashes_commitment_sha256") != source_hashes["commitment_sha256"]:
        fail("VERIFY", "SOURCE_BINDING_MISMATCH")
    if canonical_json_bytes(
        source_binding.get("executed_implementation")
    ) != canonical_json_bytes(source_hashes["source_snapshot"][IMPLEMENTATION_REL]):
        fail("VERIFY", "EXECUTED_IMPLEMENTATION_BINDING_MISMATCH")
    if (
        canonical_json_bytes(source_binding.get("source_snapshot"))
        != canonical_json_bytes(source_hashes["source_snapshot"])
        or canonical_json_bytes(source_binding.get("dependencies"))
        != canonical_json_bytes(source_hashes["dependencies"])
    ):
        fail("VERIFY", "FORMAL_SOURCE_SET_BINDING_MISMATCH")
    binding_without_commitment = dict(source_binding)
    observed_binding_commitment = binding_without_commitment.pop("commitment_sha256", None)
    if commitment(binding_without_commitment) != observed_binding_commitment:
        fail("VERIFY", "SOURCE_BINDING_COMMITMENT_MISMATCH")
    expected_decision = build_decision(
        statistics,
        conditions,
        seeds,
        source_binding["commitment_sha256"],
    )
    decision = read_json(root / "formal/decision.json", reject_test_metrics=True)
    if canonical_json_bytes(decision) != canonical_json_bytes(expected_decision):
        fail("VERIFY", "DECISION_RECOMPUTATION_MISMATCH")
    if decision["classification"] not in {
        "STABLE_LN_RECIPE_SELECTED",
        "NO_STABLE_LN_RECIPE",
    }:
        fail("VERIFY", "INVALID_SCIENTIFIC_TERMINAL")
    if decision["reserved_confirmation_seeds"] != list(RESERVED_SEEDS):
        fail("VERIFY", "RESERVED_SEEDS_NOT_SEALED")
    manifest = read_json(root / "formal/manifest.json", reject_test_metrics=True)
    verify_formal_manifest(root, manifest)
    state = read_json(root / "formal/state.json", reject_test_metrics=True)
    if (
        terminal.get("classification") != decision["classification"]
        or terminal.get("selected_condition_id") != decision["selected_condition_id"]
        or terminal.get("decision_commitment_sha256") != decision["commitment_sha256"]
        or terminal.get("planned_task_count") != 360
        or terminal.get("completed_task_count") != 360
        or terminal.get("failed_task_count") != 0
        or terminal.get("physical_attempt_count") != 360
        or terminal.get("retry_count") != 0
        or terminal.get("formal_manifest_root_sha256") != manifest["manifest_root_sha256"]
        or terminal.get("source_binding_sha256") != source_binding["commitment_sha256"]
        or terminal.get("seed_commitment_sha256") != seeds["commitment_sha256"]
        or terminal.get("seed_global_inventory_attestation_sha256")
        != seeds["global_inventory_attestation_sha256"]
        or terminal.get("next_phase_started") is not False
        or terminal.get("test_accessed") is not False
        or state.get("status") != "COMPLETE"
        or state.get("classification") != decision["classification"]
        or state.get("run_id") != terminal.get("run_id")
        or state.get("completed_task_count") != 360
        or state.get("test_accessed") is not False
        or source_binding.get("test_accessed") is not False
    ):
        fail("VERIFY", "TERMINAL_GATE_FAILED")
    return {
        "status": "COMPLETE_VALID_PASS",
        "classification": decision["classification"],
        "selected_condition_id": decision["selected_condition_id"],
        "completed_task_count": 360,
        "physical_attempt_count": 360,
        "paired_block_count": 60,
        "source_binding_sha256": source_binding["commitment_sha256"],
        "test_accessed": False,
    }


def read_snapshot(root: Path, run_id: str | None = None) -> dict[str, Any]:
    terminal_path = root / "formal/terminal.json"
    if terminal_path.exists():
        terminal = read_json(terminal_path, reject_test_metrics=True)
        if terminal.get("state") == "COMPLETE":
            try:
                return verify_complete(root)
            except BaseException as exc:
                return {
                    "status": "COMPLETE_VALID_GATE_FAIL",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "test_accessed": terminal.get("test_accessed"),
                }
        return {
            "status": "TERMINAL_FAILURE",
            "run_id": terminal.get("run_id"),
            "first_error": terminal.get("first_error"),
            "completed_task_count": terminal.get("completed_task_count"),
            "failed_task_count": terminal.get("failed_task_count"),
            "not_run_task_count": terminal.get("not_run_task_count"),
            "test_accessed": terminal.get("test_accessed"),
        }
    task_paths = sorted((root / "formal/task_results").glob("*.json"))
    completed = 0
    failed = 0
    test_accessed = False
    for path in task_paths:
        row = read_json(path, reject_test_metrics=True)
        completed += int(row.get("status") == "COMPLETE")
        failed += int(row.get("status") == "FAILED")
        test_accessed = bool(test_accessed or row.get("test_accessed") is not False)
    process = formal_process_exists(run_id)
    session = tmux_exists()
    start_path = root / "formal/start.json"
    started_at = None
    if start_path.exists():
        started_at = read_json(start_path, reject_test_metrics=True).get("started_at")
    last_progress_at = None
    if task_paths:
        last_progress_at = datetime.fromtimestamp(
            max(path.stat().st_mtime for path in task_paths), tz=timezone.utc
        ).isoformat()
    if failed:
        status = "RUNNING_UNCERTAIN" if process else "INDETERMINATE_STOP"
    elif process and completed > 0:
        status = "RUNNING_HEALTHY"
    elif process or session:
        status = "RUNNING_UNCERTAIN"
    else:
        status = "INDETERMINATE_STOP"
    return {
        "status": status,
        "run_id": run_id,
        "tmux_exists": session,
        "formal_process_exists": process,
        "formal_start_exists": start_path.exists(),
        "started_at": started_at,
        "completed_task_count": completed,
        "failed_task_count": failed,
        "remaining_task_count": 360 - completed - failed,
        "last_progress_at": last_progress_at,
        "test_accessed": test_accessed,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("dry-run", "preflight", "formal-run", "verify", "snapshot"),
    )
    parser.add_argument("--config", default=str(REPO_ROOT / CONFIG_REL))
    parser.add_argument("--result-root")
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.mode == "dry-run":
            result = dry_run(Path(args.config).resolve())
        else:
            provisional_config = (
                load_config(Path(args.config).resolve())
                if args.result_root is None
                else None
            )
            if args.result_root is None:
                root = resolve_result_root(provisional_config)
            else:
                root = Path(args.result_root)
                if not root.is_absolute():
                    root = (REPO_ROOT / root).resolve()
                else:
                    root = root.resolve()
            if root != (REPO_ROOT / RESULT_REL).resolve():
                fail("CONFIG", "RESULT_ROOT_MISMATCH", str(root))
            if args.mode == "preflight":
                result = preflight(root)
            elif args.mode == "formal-run":
                if not args.run_id:
                    fail("FORMAL", "RUN_ID_REQUIRED")
                result = formal_run(root, args.run_id)
            elif args.mode == "verify":
                result = verify_complete(root)
            else:
                result = read_snapshot(root, args.run_id)
        print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
        return 0
    except BaseException as exc:
        if args.mode == "verify":
            print(
                json.dumps(
                    {
                        "status": "COMPLETE_VALID_GATE_FAIL",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "test_accessed": False,
                    },
                    sort_keys=True,
                )
            )
            return 3
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "test_accessed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
