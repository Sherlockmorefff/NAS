"""Gated deterministic S0/G100/G150 search and Top-10 test30 pipeline.

This module is deliberately an orchestration layer.  Search commands come
from ``formal_matrix.py`` and candidate training is delegated to the existing
``final_eval.train_and_eval_arch`` evaluator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from deterministic_runtime import prepare_deterministic_environment

prepare_deterministic_environment()


REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = (
    REPO_ROOT
    / "server_validation"
    / "deterministic_three_strategy_20260804"
)
PYTHON_BIN = Path(sys.executable).resolve()
CHECKPOINT = (
    REPO_ROOT
    / "results-wgmm1"
    / "joint_search"
    / "joint_model_pipeline_global4_best.pth"
)
METHOD_CONFIG = REPO_ROOT / "configs" / "cross_dataset_methods.json"
MANIFEST_ROOT = REPO_ROOT / "results" / "cross_dataset_v1_preflight" / "manifests"
TRAINING_MODE_DECISIONS = (
    REPO_ROOT
    / "server_validation"
    / "training_mode_decisions_formal.json"
)
DATA_ROOT = Path("/tmp/dvae_cross_dataset_data")
DATASETS = ("citeseer", "pubmed", "dblp", "flickr")
METHODS = ("S0", "G100", "G150")
SEARCH_SEEDS = (5, 6, 7, 8, 9)
TEST_SEEDS = tuple(range(30))
FINAL_BASE_SEED = 0
SHA256_RE_LENGTH = 64
GATE_STATUS = "passed"

SOURCE_FIELDS = (
    "dataset",
    "method",
    "search_seed",
    "validation_rank",
    "candidate_fingerprint",
    "architecture",
    "hyperparameters",
    "validation_accuracy",
    "raw_stage",
    "normalized_stage",
    "full_evaluation_index_0based",
    "budget_1based",
    "stage_index_1based",
    "history_path",
    "decoder_seed",
    "source_id",
)
UNIQUE_FIELDS = (
    "dataset",
    "candidate_fingerprint",
    "architecture",
    "hyperparameters",
    "decoder_seed",
    "canonical_method",
    "canonical_search_seed",
    "canonical_validation_rank",
    "canonical_validation_accuracy",
    "canonical_raw_stage",
    "canonical_normalized_stage",
    "canonical_full_evaluation_index_0based",
    "canonical_budget_1based",
    "canonical_stage_index_1based",
    "all_origins",
    "source_id",
)
TEST_LONG_FIELDS = (
    "dataset",
    "candidate_fingerprint",
    "test_seed",
    "test_accuracy",
    "training_seed",
    "decoder_seed",
    "architecture",
    "hyperparameters",
    "epochs_ran",
    "best_epoch",
    "trajectory_hash",
    "model_final_state_hash",
    "wall_time",
    "worker_id",
    "gpu_name",
    "gpu_uuid",
    "source_id",
    "config_fingerprint",
)


def resolve_repository_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a CLI artifact path, interpreting relative paths from the repo."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _resolve_python_executable(value: str | os.PathLike[str]) -> Path:
    """Resolve an explicit Python path or an executable available on PATH."""

    raw = os.fspath(value)
    path = Path(raw).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return resolve_repository_path(path)
    discovered = shutil.which(raw)
    if discovered is None:
        raise ValueError(f"Python executable is not available on PATH: {raw!r}")
    return Path(discovered).resolve()


def configure_runtime_paths(
    *,
    python_executable: str | os.PathLike[str],
    training_mode_decisions: str | os.PathLike[str],
) -> None:
    """Apply portable CLI-selected runtime resources for this process."""

    global PYTHON_BIN, TRAINING_MODE_DECISIONS
    PYTHON_BIN = _resolve_python_executable(python_executable)
    TRAINING_MODE_DECISIONS = resolve_repository_path(training_mode_decisions)


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


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, canonical_json_bytes(value) + b"\n")


def atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for raw in rows:
                row = {}
                for field in fields:
                    value = raw.get(field)
                    row[field] = (
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list))
                        else value
                    )
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_read_only(command: Sequence[str], timeout: float = 60.0) -> str:
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def _source_gate_rows(
    operational_root: Path = PIPELINE_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from source_freeze import SOURCE_GATE_DISCOVERY_RULES
    from source_verification import discover_source_gate_paths

    formal_rows = []
    for relative in discover_source_gate_paths(REPO_ROOT, SOURCE_GATE_DISCOVERY_RULES):
        path = REPO_ROOT / relative
        formal_rows.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "formal_run_required": True,
            }
        )
    operational_rows = []
    scripts = operational_root / "scripts"
    if scripts.is_dir():
        for path in sorted(scripts.glob("*.sh")):
            operational_rows.append(
                {
                    "path": path.relative_to(REPO_ROOT).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    commands = operational_root / "COMMANDS.md"
    if commands.is_file():
        operational_rows.append(
            {
                "path": commands.relative_to(REPO_ROOT).as_posix(),
                "size_bytes": int(commands.stat().st_size),
                "sha256": sha256_file(commands),
            }
        )
    return formal_rows, operational_rows


def current_source_identity(root: Path = PIPELINE_ROOT) -> dict[str, Any]:
    from source_verification import source_id_from_rows

    formal_rows, operational_rows = _source_gate_rows(root)
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    dirty_payload = {
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "formal_source_rows": formal_rows,
        "operational_rows": operational_rows,
    }
    return {
        "source_id": source_id_from_rows(formal_rows),
        "git_head": _run_read_only(["git", "rev-parse", "HEAD"]).strip(),
        "git_branch": _run_read_only(["git", "branch", "--show-current"]).strip(),
        "dirty_diff_sha256": canonical_json_sha256(dirty_payload),
        "formal_source_rows": formal_rows,
        "operational_rows": operational_rows,
    }


def validate_build_directory(root: Path = PIPELINE_ROOT) -> dict[str, Any] | None:
    if not root.exists():
        return None
    build_path = root / "BUILD_IN_PROGRESS.json"
    entries = [path for path in root.iterdir() if path.name != "BUILD_IN_PROGRESS.json"]
    if not build_path.is_file():
        if entries:
            raise RuntimeError(
                f"pipeline directory exists without BUILD_IN_PROGRESS.json: {root}"
            )
        return None
    build = read_json(build_path)
    identity = current_source_identity(root)
    for field in ("source_id", "git_head", "dirty_diff_sha256"):
        if build.get(field) != identity[field]:
            raise RuntimeError(
                f"BUILD_IN_PROGRESS {field} mismatch: "
                f"recorded={build.get(field)!r} current={identity[field]!r}"
            )
    return build


def write_build_in_progress(root: Path = PIPELINE_ROOT, *, status: str) -> dict[str, Any]:
    identity = current_source_identity(root)
    payload = {
        "format_version": 1,
        "status": status,
        "created_or_refreshed_at": utc_now(),
        "pipeline_root": str(root.resolve()),
        **{key: identity[key] for key in ("source_id", "git_head", "git_branch", "dirty_diff_sha256")},
    }
    atomic_json(root / "BUILD_IN_PROGRESS.json", payload)
    return payload


def initialize_dual_pipeline(root: Path) -> dict[str, Any]:
    """Create a fresh dual-GPU operational scaffold before source identity."""

    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to initialize non-empty pipeline root: {root}")
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    root_arg = shlex.quote(str(root))
    repo_arg = shlex.quote(str(REPO_ROOT))
    python_arg = shlex.quote(str(PYTHON_BIN))
    program_arg = shlex.quote(str(REPO_ROOT / "deterministic_three_strategy_pipeline.py"))
    runtime_args = (
        f"--python-executable {python_arg} "
        f"--training-mode-decisions {shlex.quote(str(TRAINING_MODE_DECISIONS))}"
    )
    script_payloads = {
        "preflight.sh": f"""#!/usr/bin/env bash
set -Eeuo pipefail
cd {repo_arg}
exec {python_arg} {program_arg} --root {root_arg} {runtime_args} preflight
""",
        "run_formal_search_shard.sh": f"""#!/usr/bin/env bash
set -Eeuo pipefail
GPU_ID=${{1:?GPU_ID required}}
EXPECTED_GPU_UUID=${{2:?EXPECTED_GPU_UUID required}}
WORKER_ID=${{3:?WORKER_ID required}}
TASK_MANIFEST=${{4:?TASK_MANIFEST required}}
STATUS_PATH=${{5:?STATUS_PATH required}}
CURRENT_PATH=${{6:?CURRENT_PATH required}}
LOCK_PATH=${{7:?LOCK_PATH required}}
cd {repo_arg}
exec {python_arg} {program_arg} --root {root_arg} {runtime_args} \
  run-search-queue --gpu-id "$GPU_ID" --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  --worker-id "$WORKER_ID" --task-manifest "$TASK_MANIFEST" \
  --status-path "$STATUS_PATH" --current-path "$CURRENT_PATH" --lock-path "$LOCK_PATH"
""",
        "run_formal_search_queue.sh": f"""#!/usr/bin/env bash
set -Eeuo pipefail
GPU_ID=${{1:?GPU_ID required}}
cd {repo_arg}
exec {python_arg} {program_arg} --root {root_arg} {runtime_args} \
  run-search-queue --gpu-id "$GPU_ID"
""",
        "monitor.sh": f"""#!/usr/bin/env bash
set -Eeuo pipefail
cd {repo_arg}
exec {python_arg} {program_arg} --root {root_arg} {runtime_args} monitor
""",
        "audit_formal_searches.sh": f"""#!/usr/bin/env bash
set -Eeuo pipefail
cd {repo_arg}
exec {python_arg} {program_arg} --root {root_arg} {runtime_args} audit-searches
""",
    }
    for name, payload in script_payloads.items():
        path = scripts / name
        atomic_text(path, payload)
        path.chmod(0o755)
    atomic_text(
        root / "COMMANDS.md",
        "# Deterministic dual-GPU formal search\n\n"
        "This root uses one global 60-task matrix and two fixed 30-task dataset shards.\n",
    )
    build = write_build_in_progress(root, status="dual_gpu_implementation_pending_acceptance")
    return {"status": "initialized", "pipeline_root": str(root), "build": build}


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_RE_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def load_gate(
    path: Path,
    *,
    expected_source_id: str | None = None,
    expected_input_hash: str | None = None,
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"required gate is missing or empty: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("status") != GATE_STATUS:
        raise RuntimeError(f"gate is not passed: {path}")
    if expected_source_id is not None and payload.get("source_id") != expected_source_id:
        raise RuntimeError(f"gate source_id mismatch: {path}")
    if expected_input_hash is not None and payload.get("input_sha256") != expected_input_hash:
        raise RuntimeError(f"gate input hash mismatch: {path}")
    if not _valid_sha256(payload.get("output_sha256")):
        raise RuntimeError(f"gate has invalid output_sha256: {path}")
    return payload


def write_gate(
    path: Path,
    *,
    source_id: str,
    input_sha256: str,
    output_sha256: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if path.exists():
        existing = load_gate(
            path,
            expected_source_id=source_id,
            expected_input_hash=input_sha256,
        )
        if existing.get("output_sha256") != output_sha256:
            raise RuntimeError(f"refusing to overwrite mismatched gate: {path}")
        return existing
    payload = {
        "format_version": 1,
        "status": GATE_STATUS,
        "source_id": source_id,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "accepted_at": utc_now(),
        "details": dict(details or {}),
    }
    atomic_json(path, payload)
    return payload


def _validate_method_semantics(config: Mapping[str, Any]) -> None:
    common = config["common"]
    methods = config["methods"]
    required_common = {
        "hp_mode": "global4",
        "n_lhs_candidates": 768,
        "gp_init_mode": "scratch",
        "warm_start": "",
        "surrogate_type": "exact_gp",
        "online_candidate_strategy": "qlogei",
        "eval_epochs": 150,
        "patience": 40,
        "adaptive_sampling": False,
    }
    for key, expected in required_common.items():
        if common.get(key) != expected:
            raise RuntimeError(
                f"method config {key} mismatch: expected {expected!r}, got {common.get(key)!r}"
            )
    budgets = {"S0": (50, 0, 250), "G100": (50, 100, 150), "G150": (50, 150, 100)}
    if set(methods) != set(METHODS):
        raise RuntimeError("method config must contain exactly S0, G100 and G150")
    for method, expected in budgets.items():
        row = methods[method]
        actual = (
            int(row["initial_seed_evals"]),
            int(row["initial_expand_evals"]),
            int(row["n_iter"]),
        )
        if actual != expected or sum(actual) != 300:
            raise RuntimeError(f"{method} budget mismatch: expected {expected}, got {actual}")
    for method in ("G100", "G150"):
        row = methods[method]
        if row["gmm"]["wgmm_n_components"] != 4:
            raise RuntimeError(f"{method} must use fixed K=4")
        if row["initial_selection_strategy"] != "wgmm_ted_lowfid":
            raise RuntimeError(f"{method} initialization semantics changed")


def dual_gpu_assignments(acceptance: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    details = acceptance.get("details", {})
    if details.get("mode") != "fixed_dataset_dual_gpu":
        return {}
    raw = details.get("gpu_assignments")
    if not isinstance(raw, dict):
        raise RuntimeError("dual-GPU acceptance has no gpu_assignments object")
    assignments: dict[int, dict[str, Any]] = {}
    expected_datasets = {0: ["citeseer", "pubmed"], 1: ["dblp", "flickr"]}
    for worker_id in (0, 1):
        row = raw.get(str(worker_id))
        if not isinstance(row, dict):
            raise RuntimeError(f"dual-GPU acceptance is missing worker {worker_id}")
        if int(row.get("gpu_index", -1)) != worker_id:
            raise RuntimeError(f"dual-GPU acceptance GPU index mismatch for worker {worker_id}")
        if row.get("datasets") != expected_datasets[worker_id]:
            raise RuntimeError(f"dual-GPU acceptance dataset assignment mismatch for worker {worker_id}")
        if not isinstance(row.get("gpu_uuid"), str) or not row["gpu_uuid"].startswith("GPU-"):
            raise RuntimeError(f"dual-GPU acceptance UUID is invalid for worker {worker_id}")
        assignments[worker_id] = dict(row)
    if assignments[0]["gpu_uuid"] == assignments[1]["gpu_uuid"]:
        raise RuntimeError("dual-GPU acceptance must contain two distinct GPU UUIDs")
    if details.get("cross_gpu_bitwise_exact") is not True:
        raise RuntimeError("same-model dual-GPU smoke comparison is not bitwise exact")
    return assignments


def dual_worker_paths(root: Path, worker_id: int) -> dict[str, Path]:
    suffixes = {0: "gpu0_citeseer_pubmed", 1: "gpu1_dblp_flickr"}
    shard_names = {
        0: "formal_shard_gpu0_citeseer_pubmed_30.json",
        1: "formal_shard_gpu1_dblp_flickr_30.json",
    }
    if worker_id not in suffixes:
        raise ValueError("dual-GPU worker ID must be 0 or 1")
    suffix = suffixes[worker_id]
    root = root.resolve()
    return {
        "manifest": root / shard_names[worker_id],
        "status": root / f"formal_search_status_{suffix}.tsv",
        "current": root / f"formal_search_current_task_{suffix}.txt",
        "lock": root / "locks" / f"formal_{suffix}.lock",
        "runner_log": root / "logs" / f"formal_queue_{suffix}.log",
        "exit_code": root / "logs" / f"formal_queue_{suffix}.exit_code",
    }


def preflight(*, require_acceptance: bool = True, root: Path = PIPELINE_ROOT) -> dict[str, Any]:
    identity = current_source_identity(root)
    if identity["git_branch"] != "feature/gp-offline-online-update":
        raise RuntimeError(f"wrong branch: {identity['git_branch']}")
    _run_read_only(["git", "diff", "--check"])
    for path in (PYTHON_BIN, CHECKPOINT, METHOD_CONFIG, TRAINING_MODE_DECISIONS):
        if not path.is_file():
            raise FileNotFoundError(path)
    for dataset in DATASETS:
        path = MANIFEST_ROOT / dataset / "dataset_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
    from cross_dataset_runner import load_and_validate_method_config

    config = load_and_validate_method_config(METHOD_CONFIG)
    _validate_method_semantics(config)
    build = validate_build_directory(root)
    acceptance = None
    if require_acceptance:
        acceptance = load_gate(
            root / "DETERMINISM_ACCEPTED",
            expected_source_id=identity["source_id"],
        )
        if acceptance.get("details", {}).get("git_head") != identity["git_head"]:
            raise RuntimeError("DETERMINISM_ACCEPTED git HEAD mismatch")
        if acceptance.get("details", {}).get("dirty_diff_sha256") != identity["dirty_diff_sha256"]:
            raise RuntimeError("DETERMINISM_ACCEPTED dirty diff hash mismatch")
        import torch
        from eval_utils import deterministic_provenance

        current_runtime = deterministic_provenance(torch)
        assignments = dual_gpu_assignments(acceptance)
        accepted_runtimes = (
            [assignments[index]["software_hardware"] for index in (0, 1)]
            if assignments
            else [acceptance.get("details", {}).get("software_hardware", {})]
        )
        for field in (
            "deterministic_algorithms_enabled",
            "deterministic_warn_only",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "CUBLAS_WORKSPACE_CONFIG",
            "torch_version",
            "cuda_version",
            "cudnn_version",
            "pyg_version",
        ):
            for accepted_runtime in accepted_runtimes:
                if current_runtime.get(field) != accepted_runtime.get(field):
                    raise RuntimeError(f"deterministic software environment changed: {field}")
        if assignments:
            from deterministic_runtime import _nvidia_rows

            inventory, error = _nvidia_rows()
            if error is not None:
                raise RuntimeError(f"cannot verify accepted GPU inventory: {error}")
            for worker_id, assignment in assignments.items():
                selected = next(
                    (row for row in inventory if int(row["index"]) == int(assignment["gpu_index"])),
                    None,
                )
                if selected is None or selected.get("gpu_uuid") != assignment["gpu_uuid"]:
                    raise RuntimeError(f"accepted GPU identity changed for worker {worker_id}")
                for field in ("gpu_name", "driver_version"):
                    if selected.get(field) != assignment["software_hardware"].get(field):
                        raise RuntimeError(f"accepted GPU {field} changed for worker {worker_id}")
            resource_gate = load_gate(
                root / "DUAL_GPU_RESOURCE_ACCEPTED",
                expected_source_id=identity["source_id"],
            )
            if resource_gate.get("details", {}).get("gpu_uuids") != [
                assignments[0]["gpu_uuid"],
                assignments[1]["gpu_uuid"],
            ]:
                raise RuntimeError("dual-GPU resource gate GPU UUIDs mismatch")
    environment_path = root / "freeze" / "environment_manifest.json"
    if environment_path.is_file():
        environment = read_json(environment_path)
        if environment.get("source_id") != identity["source_id"]:
            raise RuntimeError("frozen environment source ID mismatch")
        if environment.get("checkpoint", {}).get("sha256") != sha256_file(CHECKPOINT):
            raise RuntimeError("frozen checkpoint changed")
        if environment.get("method_config_sha256") != sha256_file(METHOD_CONFIG):
            raise RuntimeError("frozen method config changed")
        if environment.get("training_mode_decisions_sha256") != sha256_file(TRAINING_MODE_DECISIONS):
            raise RuntimeError("frozen training-mode decision changed")
        for dataset in DATASETS:
            manifest_path = MANIFEST_ROOT / dataset / "dataset_manifest.json"
            if environment.get("dataset_manifests", {}).get(dataset, {}).get("sha256") != sha256_file(manifest_path):
                raise RuntimeError(f"frozen dataset manifest changed for {dataset}")
        from source_verification import verify_frozen_source

        verify_frozen_source(
            root / "freeze" / "source_manifest.json",
            repo_root=REPO_ROOT,
            expected_source_id=identity["source_id"],
            checkpoint_path=CHECKPOINT,
        )
        if acceptance is not None and dual_gpu_assignments(acceptance):
            from formal_matrix import FORMAL_SHARD_SPECS, audit_formal_shards

            matrix_path = root / "formal_matrix_60.json"
            if not matrix_path.is_file():
                raise RuntimeError("frozen dual-GPU pipeline is missing global matrix")
            matrix = read_json(matrix_path)
            matrix_sha256 = sha256_file(matrix_path)
            shards = {
                str(spec["shard_id"]): read_json(root / str(spec["filename"]))
                for spec in FORMAL_SHARD_SPECS
            }
            audit_formal_shards(
                matrix,
                shards,
                expected_parent_matrix_sha256=matrix_sha256,
            )
    return {
        "status": "passed",
        "source_id": identity["source_id"],
        "git_head": identity["git_head"],
        "dirty_diff_sha256": identity["dirty_diff_sha256"],
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "method_config_fingerprint": canonical_json_sha256(config),
        "build": build,
        "determinism_acceptance": acceptance,
    }


def _compute_gpu_process_rows() -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot query GPU compute processes: {completed.stderr.strip()}")
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"invalid GPU compute-process row: {line!r}")
        rows.append(dict(zip(("gpu_uuid", "pid", "process_name", "used_memory_mib"), fields)))
    return rows


def run_dual_gpu_resource_smoke(root: Path) -> dict[str, Any]:
    """Run PubMed/Flickr GPU smokes concurrently and generate their gate."""

    root = root.resolve()
    identity = current_source_identity(root)
    acceptance = load_gate(root / "DETERMINISM_ACCEPTED", expected_source_id=identity["source_id"])
    assignments = dual_gpu_assignments(acceptance)
    if not assignments:
        raise RuntimeError("dual-GPU resource smoke requires dual-GPU acceptance")
    busy = [row for row in _compute_gpu_process_rows() if row["gpu_uuid"] in {assignments[0]["gpu_uuid"], assignments[1]["gpu_uuid"]}]
    if busy:
        raise RuntimeError(f"dual-GPU resource smoke requires both GPUs idle: {busy}")
    smoke_root = root / "resource_smoke"
    outputs = {0: smoke_root / "gpu0_pubmed", 1: smoke_root / "gpu1_flickr"}
    logs = {0: root / "logs" / "resource_smoke_gpu0_pubmed.log", 1: root / "logs" / "resource_smoke_gpu1_flickr.log"}
    for path in [*outputs.values(), *logs.values(), root / "dual_gpu_resource_smoke_audit.json", root / "DUAL_GPU_RESOURCE_ACCEPTED"]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite dual-GPU resource smoke artifact: {path}")
    datasets = {0: "pubmed", 1: "flickr"}
    commands: dict[int, list[str]] = {}
    for worker_id in (0, 1):
        dataset = datasets[worker_id]
        commands[worker_id] = [
            str(PYTHON_BIN),
            str(REPO_ROOT / "gpu_dataset_smoke.py"),
            "--dataset", dataset,
            "--data_root", str(DATA_ROOT),
            "--checkpoint", str(CHECKPOINT),
            "--expected_dataset_manifest", str(MANIFEST_ROOT / dataset / "dataset_manifest.json"),
            "--output", str(outputs[worker_id]),
            "--seed", "0",
            "--pool_size", "768",
            "--max_epochs", "3",
            "--patience", "2",
        ]
    memory_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = next((line for line in meminfo.read_text().splitlines() if line.startswith("MemAvailable:")), None)
        if match is not None:
            memory_kib = int(match.split()[1])
    host = {
        "cpu_count": os.cpu_count(),
        "memory_available_kib_before": memory_kib,
        "disk_free_bytes_before": shutil.disk_usage(root.parent).free,
        "dataset_cache": str(DATA_ROOT),
        "dataset_cache_exists": DATA_ROOT.is_dir(),
    }
    if not host["dataset_cache_exists"] or int(host["cpu_count"] or 0) < 2:
        raise RuntimeError("host resource preflight failed: dataset cache or CPU count")
    started_at = utc_now()
    processes: dict[int, subprocess.Popen[bytes]] = {}
    handles = {}
    try:
        for worker_id in (0, 1):
            logs[worker_id].parent.mkdir(parents=True, exist_ok=True)
            handles[worker_id] = logs[worker_id].open("xb")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(assignments[worker_id]["gpu_index"])
            env["PYTHONHASHSEED"] = "0"
            env["CUBLAS_WORKSPACE_CONFIG"] = prepare_deterministic_environment()
            processes[worker_id] = subprocess.Popen(
                commands[worker_id],
                cwd=REPO_ROOT,
                env=env,
                stdout=handles[worker_id],
                stderr=subprocess.STDOUT,
            )
        exit_codes = {worker_id: processes[worker_id].wait() for worker_id in (0, 1)}
    finally:
        for handle in handles.values():
            handle.close()
    if exit_codes != {0: 0, 1: 0}:
        raise RuntimeError(f"concurrent dual-GPU resource smoke failed: {exit_codes}")
    results = {worker_id: read_json(outputs[worker_id] / "smoke_result.json") for worker_id in (0, 1)}
    for worker_id in (0, 1):
        result = results[worker_id]
        if result.get("status") != "passed" or result.get("dataset") != datasets[worker_id]:
            raise RuntimeError(f"resource smoke result invalid for worker {worker_id}")
        reported_uuid = str(result.get("gpu", {}).get("gpu_uuid", ""))
        normalized_uuid = reported_uuid if reported_uuid.startswith("GPU-") else f"GPU-{reported_uuid}"
        if normalized_uuid != assignments[worker_id]["gpu_uuid"]:
            raise RuntimeError(f"resource smoke GPU UUID mismatch for worker {worker_id}")
    audit = {
        "status": "passed",
        "source_id": identity["source_id"],
        "started_at": started_at,
        "completed_at": utc_now(),
        "concurrent": True,
        "host": host,
        "workers": {
            str(worker_id): {
                "dataset": datasets[worker_id],
                "gpu_index": assignments[worker_id]["gpu_index"],
                "gpu_uuid": assignments[worker_id]["gpu_uuid"],
                "pid": processes[worker_id].pid,
                "exit_code": exit_codes[worker_id],
                "output": str(outputs[worker_id]),
                "log": str(logs[worker_id]),
                "result_sha256": sha256_file(outputs[worker_id] / "smoke_result.json"),
                "peak_reserved_bytes": results[worker_id]["peak_reserved_bytes"],
            }
            for worker_id in (0, 1)
        },
    }
    audit_path = root / "dual_gpu_resource_smoke_audit.json"
    atomic_json(audit_path, audit)
    write_gate(
        root / "DUAL_GPU_RESOURCE_ACCEPTED",
        source_id=identity["source_id"],
        input_sha256=canonical_json_sha256({str(i): commands[i] for i in (0, 1)}),
        output_sha256=sha256_file(audit_path),
        details={
            "gpu_uuids": [assignments[0]["gpu_uuid"], assignments[1]["gpu_uuid"]],
            "audit_path": str(audit_path),
            "concurrent": True,
        },
    )
    return audit


def verify_search_gpu_matches_acceptance(
    gpu_id: str,
    acceptance: Mapping[str, Any],
    *,
    expected_gpu_uuid: str | None = None,
    worker_id: int | None = None,
) -> dict[str, Any]:
    from deterministic_runtime import _nvidia_rows

    rows, error = _nvidia_rows()
    if error is not None:
        raise RuntimeError(f"cannot verify formal search GPU: {error}")
    try:
        numeric_id = int(gpu_id)
    except ValueError as exc:
        raise ValueError("formal search GPU_ID must be a numeric physical index") from exc
    selected = next((row for row in rows if int(row["index"]) == numeric_id), None)
    if selected is None:
        raise RuntimeError(f"formal search GPU does not exist: {gpu_id}")
    assignments = dual_gpu_assignments(acceptance)
    if assignments:
        if worker_id not in assignments:
            raise RuntimeError("dual-GPU worker ID is missing or invalid")
        assignment = assignments[int(worker_id)]
        if numeric_id != int(assignment["gpu_index"]):
            raise RuntimeError("formal worker physical GPU index differs from acceptance")
        accepted_uuid = str(assignment["gpu_uuid"])
        software = assignment["software_hardware"]
    else:
        accepted_uuid = str(acceptance.get("details", {}).get("gpu_uuid"))
        software = acceptance.get("details", {}).get("software_hardware", {})
    if expected_gpu_uuid is not None and expected_gpu_uuid != accepted_uuid:
        raise RuntimeError("requested expected GPU UUID differs from acceptance")
    expected_uuid = accepted_uuid
    if selected.get("gpu_uuid") != expected_uuid:
        raise RuntimeError(
            "formal search GPU UUID differs from the determinism-smoke GPU: "
            f"expected={expected_uuid!r} selected={selected.get('gpu_uuid')!r}"
        )
    for field in ("gpu_name", "driver_version"):
        if selected.get(field) != software.get(field):
            raise RuntimeError(f"formal search GPU {field} changed")
    return selected


def freeze_pipeline(root: Path = PIPELINE_ROOT) -> dict[str, Any]:
    state = preflight(require_acceptance=True, root=root)
    freeze = root / "freeze"
    formal_paths = (
        freeze / "source_manifest.json",
        freeze / "checkpoint_manifest.json",
        freeze / "environment_manifest.json",
        root / "formal_matrix_60.json",
        root / "formal_matrix_60.csv",
        root / "formal_matrix_audit.json",
        root / "formal_shard_gpu0_citeseer_pubmed_30.json",
        root / "formal_shard_gpu1_dblp_flickr_30.json",
        root / "formal_shards_audit.json",
    )
    existing = [path for path in formal_paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite freeze artifacts: " + ", ".join(map(str, existing)))
    identity = current_source_identity(root)
    acceptance = load_gate(root / "DETERMINISM_ACCEPTED", expected_source_id=identity["source_id"])
    checkpoint_row = {
        "path": str(CHECKPOINT.resolve()),
        "size_bytes": int(CHECKPOINT.stat().st_size),
        "sha256": sha256_file(CHECKPOINT),
    }
    from source_freeze import SOURCE_GATE_DISCOVERY_RULES

    source_manifest = {
        "format_version": 1,
        "source_id": identity["source_id"],
        "created_at": utc_now(),
        "repository_root": str(REPO_ROOT),
        "branch": identity["git_branch"],
        "head_commit": identity["git_head"],
        "dirty_diff_sha256": identity["dirty_diff_sha256"],
        "source_gate": {"discovery_rules": SOURCE_GATE_DISCOVERY_RULES},
        "checkpoint": checkpoint_row,
        "files": identity["formal_source_rows"],
        "operational_files": identity["operational_rows"],
    }
    atomic_json(freeze / "source_manifest.json", source_manifest)
    atomic_json(
        freeze / "checkpoint_manifest.json",
        {"format_version": 1, "source_id": identity["source_id"], "checkpoint": checkpoint_row},
    )
    environment = {
        "format_version": 1,
        "source_id": identity["source_id"],
        "created_at": utc_now(),
        "python_executable": str(PYTHON_BIN),
        "determinism_acceptance": acceptance,
        "checkpoint": checkpoint_row,
        "method_config_path": str(METHOD_CONFIG.resolve()),
        "method_config_sha256": sha256_file(METHOD_CONFIG),
        "training_mode_decisions_path": str(TRAINING_MODE_DECISIONS.resolve()),
        "training_mode_decisions_sha256": sha256_file(TRAINING_MODE_DECISIONS),
        "dataset_manifests": {
            dataset: {
                "path": str((MANIFEST_ROOT / dataset / "dataset_manifest.json").resolve()),
                "sha256": sha256_file(MANIFEST_ROOT / dataset / "dataset_manifest.json"),
            }
            for dataset in DATASETS
        },
        "test30": {
            "final_base_seed": FINAL_BASE_SEED,
            "test_seeds": list(TEST_SEEDS),
            "eval_epochs": 150,
            "patience": 40,
            "sharding": "test_seed % n_workers",
        },
    }
    atomic_json(freeze / "environment_manifest.json", environment)
    from formal_matrix import (
        build_formal_manifest,
        build_formal_shards,
        write_formal_manifest,
        write_formal_shards,
    )

    run_tag = f"deterministic_three_strategy_{identity['source_id'][:12]}"
    manifest, audit = build_formal_manifest(
        repo_root=REPO_ROOT,
        source_manifest_path=freeze / "source_manifest.json",
        frozen_source_id=identity["source_id"],
        checkpoint=CHECKPOINT,
        data_root=DATA_ROOT,
        manifest_root=MANIFEST_ROOT,
        training_mode_decisions=TRAINING_MODE_DECISIONS,
        method_config=METHOD_CONFIG,
        python_executable=str(PYTHON_BIN),
        run_tag=run_tag,
        artifact_root=root,
    )
    digest = write_formal_manifest(
        manifest,
        audit,
        output_json=root / "formal_matrix_60.json",
        output_csv=root / "formal_matrix_60.csv",
        output_audit=root / "formal_matrix_audit.json",
    )
    assignments = dual_gpu_assignments(acceptance)
    shard_hashes: dict[str, str] = {}
    shard_audit = None
    if assignments:
        shards, shard_audit = build_formal_shards(
            manifest,
            parent_matrix_sha256=digest,
            gpu_assignments=assignments,
        )
        shard_hashes = write_formal_shards(shards, output_root=root)
        atomic_json(root / "formal_shards_audit.json", shard_audit)
    return {
        **state,
        **audit,
        "formal_matrix_sha256": digest,
        "formal_shard_sha256": shard_hashes,
        "formal_shards_audit": shard_audit,
    }


def _history_deterministic_state(row: Mapping[str, Any]) -> Mapping[str, Any]:
    state = row.get("deterministic_provenance")
    if not isinstance(state, dict):
        raise ValueError("history record is missing deterministic_provenance")
    required = {
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    for key, expected in required.items():
        if state.get(key) is not expected:
            raise ValueError(f"deterministic provenance {key} is not {expected}")
    if state.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("invalid CUBLAS_WORKSPACE_CONFIG in history")
    for key in ("torch_version", "cuda_version", "cudnn_version", "pyg_version", "gpu_name", "gpu_uuid", "driver_version"):
        if state.get(key) in (None, ""):
            raise ValueError(f"deterministic provenance is missing {key}")
    return state


def audit_search_history(task: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    history_path = Path(str(task["output_directory"])) / "history_final.json"
    metadata_path = Path(str(task["output_directory"])) / "history_metadata.json"
    if not history_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"incomplete search task {task['task_id']}")
    rows = read_json(history_path)
    metadata = read_json(metadata_path)
    if not isinstance(rows, list) or len(rows) != 300:
        raise ValueError(f"{task['task_id']} must have exactly 300 history records")
    indices = [row.get("full_evaluation_index") for row in rows]
    if indices != list(range(300)):
        raise ValueError(f"{task['task_id']} full_evaluation_index is not 0..299")
    fingerprints = [row.get("candidate_fingerprint") for row in rows]
    if any(not _valid_sha256(value) for value in fingerprints) or len(set(fingerprints)) != 300:
        raise ValueError(f"{task['task_id']} candidate fingerprints are invalid or duplicated")
    for row in rows:
        if row.get("evaluation_fidelity") != "full":
            raise ValueError(f"{task['task_id']} history contains non-full record")
        value = row.get("val_acc")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{task['task_id']} has non-finite validation accuracy")
        context = row.get("dataset_context", {})
        if context.get("canonical_name") != task["dataset"]:
            raise ValueError(f"{task['task_id']} dataset mismatch")
        if row.get("method_label") != task["method_label"] or int(row.get("search_seed", -1)) != int(task["search_seed"]):
            raise ValueError(f"{task['task_id']} method/search seed mismatch")
        if row.get("formal_source_id") != source_id:
            raise ValueError(f"{task['task_id']} source ID mismatch")
        if row.get("formal_config_fingerprint") != task["configuration_fingerprint"]:
            raise ValueError(f"{task['task_id']} config fingerprint mismatch")
        _history_deterministic_state(row)
    if metadata.get("formal_source_id") != source_id:
        raise ValueError(f"{task['task_id']} metadata source ID mismatch")
    return {
        "task_id": task["task_id"],
        "history_path": str(history_path.resolve()),
        "history_sha256": sha256_file(history_path),
        "full_evaluation_count": 300,
        "unique_candidate_count": 300,
    }


def audit_formal_searches(root: Path = PIPELINE_ROOT, *, write_marker: bool = True) -> dict[str, Any]:
    state = preflight(require_acceptance=True, root=root)
    matrix_path = root / "formal_matrix_60.json"
    matrix = read_json(matrix_path)
    if matrix.get("source_id") != state["source_id"] or len(matrix.get("tasks", [])) != 60:
        raise RuntimeError("formal matrix identity/count mismatch")
    audits = [audit_search_history(task, state["source_id"]) for task in matrix["tasks"]]
    payload = {
        "status": "passed",
        "source_id": state["source_id"],
        "task_count": len(audits),
        "full_evaluation_count": sum(row["full_evaluation_count"] for row in audits),
        "tasks": audits,
    }
    atomic_json(root / "formal_search_completion_audit.json", payload)
    if write_marker:
        write_gate(
            root / "FORMAL_SEARCH_AUDIT_PASSED",
            source_id=state["source_id"],
            input_sha256=sha256_file(matrix_path),
            output_sha256=sha256_file(root / "formal_search_completion_audit.json"),
            details={"task_count": 60, "full_evaluation_count": 18_000},
        )
    return payload


def _complete_search_task(task: Mapping[str, Any], source_id: str) -> bool:
    output = Path(str(task["output_directory"]))
    if not output.exists():
        return False
    if not output.is_dir() or not any(output.iterdir()):
        return False
    audit_search_history(task, source_id)
    return True


@contextmanager
def exclusive_file_lock(path: Path, *, owner: Mapping[str, Any]):
    """Hold a non-blocking advisory lock and record its current owner."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"formal worker lock is already held: {path}") from exc
        payload = canonical_json_bytes({"pid": os.getpid(), "acquired_at": utc_now(), **dict(owner)}) + b"\n"
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, payload)
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _load_search_tasks(
    *,
    root: Path,
    task_manifest: Path | None,
    worker_id: int | None,
    expected_gpu_uuid: str | None,
) -> tuple[list[dict[str, Any]], bool]:
    matrix_path = root / "formal_matrix_60.json"
    matrix = read_json(matrix_path)
    if task_manifest is None:
        if worker_id is not None or expected_gpu_uuid is not None:
            raise ValueError("worker/GPU UUID arguments require --task-manifest")
        return list(matrix["tasks"]), False
    if worker_id is None or expected_gpu_uuid is None:
        raise ValueError("sharded search requires worker ID and expected GPU UUID")
    manifest_path = task_manifest.resolve()
    if manifest_path.parent != root.resolve():
        raise RuntimeError("formal task manifest must be directly under the pipeline root")
    if manifest_path != dual_worker_paths(root, int(worker_id))["manifest"]:
        raise RuntimeError("formal task manifest path does not match fixed worker assignment")
    from formal_matrix import audit_formal_shard

    shard = read_json(manifest_path)
    audit_formal_shard(
        matrix,
        shard,
        expected_parent_matrix_sha256=sha256_file(matrix_path),
    )
    if int(shard["worker_id"]) != int(worker_id):
        raise RuntimeError("formal shard worker ID mismatch")
    if shard["expected_gpu_uuid"] != expected_gpu_uuid:
        raise RuntimeError("formal shard expected GPU UUID mismatch")
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0] != sha256_file(manifest_path):
        raise RuntimeError("formal shard file SHA-256 sidecar mismatch")
    return [dict(task) for task in shard["tasks"]], True


def run_search_queue(
    gpu_id: str,
    root: Path = PIPELINE_ROOT,
    *,
    task_manifest: Path | None = None,
    worker_id: int | None = None,
    expected_gpu_uuid: str | None = None,
    status_path: Path | None = None,
    current_path: Path | None = None,
    lock_path: Path | None = None,
) -> int:
    state = preflight(require_acceptance=True, root=root)
    verify_search_gpu_matches_acceptance(
        gpu_id,
        state["determinism_acceptance"] or {},
        expected_gpu_uuid=expected_gpu_uuid,
        worker_id=worker_id,
    )
    tasks, sharded = _load_search_tasks(
        root=root,
        task_manifest=task_manifest,
        worker_id=worker_id,
        expected_gpu_uuid=expected_gpu_uuid,
    )
    status_path = (root / "formal_search_status.tsv") if status_path is None else status_path.resolve()
    current_path = (root / "formal_search_current_task.txt") if current_path is None else current_path.resolve()
    lock_path = (root / "locks" / "formal_queue.lock") if lock_path is None else lock_path.resolve()
    if sharded:
        expected_paths = dual_worker_paths(root, int(worker_id))
        for role, actual in (("status", status_path), ("current", current_path), ("lock", lock_path)):
            if actual != expected_paths[role]:
                raise RuntimeError(f"dual-GPU {role} path differs from fixed worker path")
    for role, path in (("status", status_path), ("current", current_path), ("lock", lock_path)):
        if not path.is_relative_to(root.resolve()):
            raise RuntimeError(f"formal worker {role} path must stay under pipeline root")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if not status_path.exists():
        atomic_text(status_path, "timestamp\ttask_id\tstate\texit_code\n")
    owner = {
        "worker_id": worker_id,
        "gpu_id": str(gpu_id),
        "expected_gpu_uuid": expected_gpu_uuid,
        "task_manifest": None if task_manifest is None else str(task_manifest.resolve()),
    }
    with exclusive_file_lock(lock_path, owner=owner):
        for task in tasks:
            task_lock = root / "locks" / "tasks" / f"{task['task_id']}.lock"
            with exclusive_file_lock(task_lock, owner={**owner, "task_id": task["task_id"]}):
                if _complete_search_task(task, state["source_id"]):
                    with status_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"{utc_now()}\t{task['task_id']}\tskipped_complete\t0\n")
                    continue
                output = Path(str(task["output_directory"]))
                log_dir = Path(str(task["log_directory"]))
                for role, path in (("output", output), ("log", log_dir)):
                    if path.exists() and (not path.is_dir() or any(path.iterdir())):
                        raise RuntimeError(f"refusing inconsistent nonempty {role}: {path}")
                atomic_text(current_path, str(task["task_id"]) + "\n")
                log_path = root / "logs" / f"formal_{task['task_id']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["PYTHONHASHSEED"] = "0"
                env["CUBLAS_WORKSPACE_CONFIG"] = prepare_deterministic_environment()
                with status_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{utc_now()}\t{task['task_id']}\trunning\t-\n")
                with log_path.open("ab") as log_handle:
                    completed = subprocess.run(
                        [str(value) for value in task["exact_argv"]],
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                with status_path.open("a", encoding="utf-8") as handle:
                    state_text = "completed" if completed.returncode == 0 else "failed"
                    handle.write(f"{utc_now()}\t{task['task_id']}\t{state_text}\t{completed.returncode}\n")
                if completed.returncode != 0:
                    return int(completed.returncode)
                audit_search_history(task, state["source_id"])
        atomic_text(current_path, "DONE\n")
    if not sharded:
        audit_formal_searches(root)
    return 0


def normalized_stage(raw_stage: str, method: str, full_index: int) -> tuple[str, int]:
    if raw_stage == "initial_seed":
        return "seed_full", full_index + 1
    if raw_stage == "initial_expand" and method in ("G100", "G150"):
        return "promoted_full", full_index - 50 + 1
    if raw_stage == "online_bo":
        expansion = 0 if method == "S0" else 100 if method == "G100" else 150
        return "online_bo", full_index - 50 - expansion + 1
    raise ValueError(
        f"unrecognized real history stage: method={method} raw_stage={raw_stage!r} index={full_index}"
    )


def select_top10_full_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select validation Top-10 from full, valid records without reading test fields."""

    full_records = [
        dict(row)
        for row in records
        if row.get("evaluation_fidelity") == "full"
        and isinstance(row.get("full_evaluation_index"), int)
        and row.get("valid") is True
        and isinstance(row.get("val_acc"), (int, float))
        and math.isfinite(float(row["val_acc"]))
    ]
    return sorted(
        full_records,
        key=lambda row: (
            -float(row["val_acc"]),
            int(row["full_evaluation_index"]),
            str(row["candidate_fingerprint"]),
        ),
    )[:10]


def _architecture(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"operations": list(row["operations"]), "edges": [list(edge) for edge in row["edges"]]}


def _hyperparameters(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lr": float(row["lr"]),
        "dropout": float(row["dropout"]),
        "hidden_dim": int(row["hidden_dim"]),
        "l2": float(row["l2"]),
        "gat_heads": int(row.get("gat_heads", 1)),
        "sage_aggr": str(row.get("sage_aggr", "mean")),
        "gin_eps": float(row.get("gin_eps", 0.0)),
        "gat_heads_by_layer": row.get("gat_heads_by_layer"),
        "sage_aggr_by_layer": row.get("sage_aggr_by_layer"),
        "gin_eps_by_layer": row.get("gin_eps_by_layer"),
    }


def extract_validation_top10(root: Path = PIPELINE_ROOT) -> dict[str, Any]:
    identity = current_source_identity(root)
    matrix_path = root / "formal_matrix_60.json"
    load_gate(
        root / "FORMAL_SEARCH_AUDIT_PASSED",
        expected_source_id=identity["source_id"],
        expected_input_hash=sha256_file(matrix_path),
    )
    matrix = read_json(matrix_path)
    source_rows: list[dict[str, Any]] = []
    for task in matrix["tasks"]:
        audit_search_history(task, identity["source_id"])
        history_path = Path(task["output_directory"]) / "history_final.json"
        records = read_json(history_path)
        full_records = select_top10_full_records(records)
        all_full_count = sum(
            row.get("evaluation_fidelity") == "full"
            and isinstance(row.get("full_evaluation_index"), int)
            and isinstance(row.get("val_acc"), (int, float))
            and math.isfinite(float(row["val_acc"]))
            for row in records
        )
        if all_full_count != 300:
            raise ValueError(f"{task['task_id']} does not expose exactly 300 finite full records")
        selected = full_records
        if len(selected) != 10:
            raise ValueError(f"{task['task_id']} has fewer than 10 valid full records")
        for rank, row in enumerate(selected, 1):
            index = int(row["full_evaluation_index"])
            raw = str(row["evaluation_stage"])
            norm, stage_index = normalized_stage(raw, str(task["method"]), index)
            source_rows.append(
                {
                    "dataset": task["dataset"],
                    "method": task["method"],
                    "search_seed": int(task["search_seed"]),
                    "validation_rank": rank,
                    "candidate_fingerprint": row["candidate_fingerprint"],
                    "architecture": _architecture(row),
                    "hyperparameters": _hyperparameters(row),
                    "validation_accuracy": float(row["val_acc"]),
                    "raw_stage": raw,
                    "normalized_stage": norm,
                    "full_evaluation_index_0based": index,
                    "budget_1based": index + 1,
                    "stage_index_1based": stage_index,
                    "history_path": str(history_path.resolve()),
                    "decoder_seed": int(row["decoder_seed"]),
                    "source_id": identity["source_id"],
                }
            )
    if len(source_rows) != 600:
        raise AssertionError(f"expected 600 Top-10 source rows, got {len(source_rows)}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[(row["dataset"], row["candidate_fingerprint"])].append(row)
    unique_rows = []
    for (dataset, fingerprint), origins in sorted(grouped.items()):
        architectures = {canonical_json_sha256(row["architecture"]) for row in origins}
        hyperparameters = {canonical_json_sha256(row["hyperparameters"]) for row in origins}
        if len(architectures) != 1 or len(hyperparameters) != 1:
            raise ValueError(f"candidate identity collision for {dataset}/{fingerprint}")
        canonical = min(
            origins,
            key=lambda row: (
                -float(row["validation_accuracy"]),
                int(row["search_seed"]),
                int(row["budget_1based"]),
                str(row["method"]),
            ),
        )
        unique_rows.append(
            {
                "dataset": dataset,
                "candidate_fingerprint": fingerprint,
                "architecture": canonical["architecture"],
                "hyperparameters": canonical["hyperparameters"],
                "decoder_seed": canonical["decoder_seed"],
                "canonical_method": canonical["method"],
                "canonical_search_seed": canonical["search_seed"],
                "canonical_validation_rank": canonical["validation_rank"],
                "canonical_validation_accuracy": canonical["validation_accuracy"],
                "canonical_raw_stage": canonical["raw_stage"],
                "canonical_normalized_stage": canonical["normalized_stage"],
                "canonical_full_evaluation_index_0based": canonical["full_evaluation_index_0based"],
                "canonical_budget_1based": canonical["budget_1based"],
                "canonical_stage_index_1based": canonical["stage_index_1based"],
                "all_origins": sorted(
                    origins,
                    key=lambda row: (row["method"], row["search_seed"], row["validation_rank"]),
                ),
                "source_id": identity["source_id"],
            }
        )
    top = root / "top10"
    source_path = top / "top10_source_candidates.csv"
    unique_path = top / "top10_unique_candidates.csv"
    report_path = top / "top10_extraction_report.json"
    write_csv(source_path, source_rows, SOURCE_FIELDS)
    write_csv(unique_path, unique_rows, UNIQUE_FIELDS)
    report = {
        "status": "passed",
        "source_id": identity["source_id"],
        "source_row_count": len(source_rows),
        "unique_candidate_count": len(unique_rows),
        "unique_candidates_by_dataset": dict(Counter(row["dataset"] for row in unique_rows)),
        "selection_field": "val_acc",
        "test_field_used": False,
        "low_fidelity_used": False,
        "sort": ["validation_accuracy desc", "full_evaluation_index asc", "fingerprint asc"],
        "source_csv_sha256": sha256_file(source_path),
        "unique_csv_sha256": sha256_file(unique_path),
    }
    atomic_json(report_path, report)
    input_hash = sha256_file(root / "FORMAL_SEARCH_AUDIT_PASSED")
    output_hash = canonical_json_sha256(
        {"source": sha256_file(source_path), "unique": sha256_file(unique_path), "report": sha256_file(report_path)}
    )
    write_gate(
        root / "TOP10_EXTRACTION_AUDIT_PASSED",
        source_id=identity["source_id"],
        input_sha256=input_hash,
        output_sha256=output_hash,
        details={"source_rows": 600, "unique_candidates": len(unique_rows)},
    )
    build_test_task_manifest(root)
    return report


def _decode_json_cell(value: str, field: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in CSV field {field}") from exc


def load_unique_candidates(root: Path = PIPELINE_ROOT) -> list[dict[str, Any]]:
    rows = read_csv(root / "top10" / "top10_unique_candidates.csv")
    output = []
    for row in rows:
        output.append(
            {
                **row,
                "architecture": _decode_json_cell(row["architecture"], "architecture"),
                "hyperparameters": _decode_json_cell(row["hyperparameters"], "hyperparameters"),
                "all_origins": _decode_json_cell(row["all_origins"], "all_origins"),
                "decoder_seed": int(row["decoder_seed"]),
            }
        )
    return output


def test30_config_fingerprint(source_id: str) -> str:
    return canonical_json_sha256(
        {
            "source_id": source_id,
            "final_base_seed": FINAL_BASE_SEED,
            "test_seeds": list(TEST_SEEDS),
            "eval_epochs": 150,
            "patience": 40,
            "training_seed_scheme": "final_base_candidate_fingerprint_replicate_v1",
            "worker_shard": "test_seed % n_workers",
        }
    )


def build_test_task_manifest(root: Path = PIPELINE_ROOT) -> dict[str, Any]:
    identity = current_source_identity(root)
    load_gate(root / "TOP10_EXTRACTION_AUDIT_PASSED", expected_source_id=identity["source_id"])
    candidates = load_unique_candidates(root)
    config_fingerprint = test30_config_fingerprint(identity["source_id"])
    tasks = [
        {
            "dataset": candidate["dataset"],
            "candidate_fingerprint": candidate["candidate_fingerprint"],
            "test_seed": seed,
            "worker_id_n2": seed % 2,
            "source_id": identity["source_id"],
            "config_fingerprint": config_fingerprint,
        }
        for candidate in sorted(candidates, key=lambda row: (row["dataset"], row["candidate_fingerprint"]))
        for seed in TEST_SEEDS
    ]
    keys = {(row["dataset"], row["candidate_fingerprint"], row["test_seed"]) for row in tasks}
    if len(keys) != len(tasks):
        raise AssertionError("canonical test task manifest contains duplicates")
    payload = {
        "format_version": 1,
        "source_id": identity["source_id"],
        "config_fingerprint": config_fingerprint,
        "unique_candidate_count": len(candidates),
        "unique_candidates_by_dataset": dict(Counter(row["dataset"] for row in candidates)),
        "task_count": len(tasks),
        "task_count_formula": "30 * sum(unique_candidate_count_by_dataset)",
        "maximum_task_count": 18_000,
        "test_seeds": list(TEST_SEEDS),
        "tasks": tasks,
    }
    path = root / "final_test" / "canonical_task_manifest.json"
    atomic_json(path, payload)
    return payload


def assigned_tasks(manifest: Mapping[str, Any], worker_id: int, n_workers: int) -> list[dict[str, Any]]:
    if n_workers <= 0 or not 0 <= worker_id < n_workers:
        raise ValueError("worker_id must satisfy 0 <= worker_id < n_workers")
    return [
        dict(task)
        for task in manifest["tasks"]
        if int(task["test_seed"]) % int(n_workers) == int(worker_id)
    ]


def task_result_path(root: Path, dataset: str, fingerprint: str, test_seed: int) -> Path:
    return root / "final_test" / "task_results" / dataset / fingerprint / f"seed_{int(test_seed)}.json"


def _finite_accuracy(value: Any) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("accuracy must be finite")
    accuracy = float(value)
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy is outside [0,1]: {accuracy}")
    return accuracy


def expected_test_training_seed(candidate_fingerprint: str, test_seed: int) -> int:
    from final_eval import final_evaluation_seed

    return final_evaluation_seed(FINAL_BASE_SEED, candidate_fingerprint, int(test_seed))


def validate_task_result(
    row: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    expected_worker: int | None = None,
) -> dict[str, Any]:
    required = set(TEST_LONG_FIELDS) | {"deterministic_provenance", "completed"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError("task result missing fields: " + ", ".join(missing))
    if row.get("completed") is not True:
        raise ValueError("task result is incomplete")
    for key in ("dataset", "candidate_fingerprint", "source_id", "config_fingerprint"):
        if row.get(key) != task.get(key):
            raise ValueError(f"task result {key} mismatch")
    if int(row.get("test_seed", -1)) != int(task["test_seed"]):
        raise ValueError("task result test_seed mismatch")
    if int(row.get("training_seed", -1)) != expected_test_training_seed(
        str(task["candidate_fingerprint"]), int(task["test_seed"])
    ):
        raise ValueError("task result training_seed mismatch")
    if expected_worker is not None and int(row.get("worker_id", -1)) != expected_worker:
        raise ValueError("task result worker_id mismatch")
    _finite_accuracy(row.get("test_accuracy"))
    state = row.get("deterministic_provenance")
    if not isinstance(state, dict):
        raise ValueError("task result deterministic_provenance is missing")
    for field, expected in (
        ("deterministic_algorithms_enabled", True),
        ("deterministic_warn_only", False),
        ("cudnn_deterministic", True),
        ("cudnn_benchmark", False),
    ):
        if state.get(field) is not expected:
            raise ValueError(f"task deterministic state {field} mismatch")
    if row.get("gpu_uuid") in (None, "") or row.get("gpu_name") in (None, ""):
        raise ValueError("task result lacks GPU identity")
    return dict(row)


def _candidate_map(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["dataset"], row["candidate_fingerprint"]): row
        for row in load_unique_candidates(root)
    }


def run_test30_worker(gpu_id: str, worker_id: int, n_workers: int, root: Path = PIPELINE_ROOT) -> int:
    identity = current_source_identity(root)
    load_gate(root / "TOP10_EXTRACTION_AUDIT_PASSED", expected_source_id=identity["source_id"])
    manifest_path = root / "final_test" / "canonical_task_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("source_id") != identity["source_id"]:
        raise RuntimeError("canonical task manifest source mismatch")
    tasks = assigned_tasks(manifest, worker_id, n_workers)
    worker_root = root / "final_test" / "workers" / f"worker_{worker_id}"
    worker_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        worker_root / "assignment_manifest.json",
        {
            "source_id": identity["source_id"],
            "worker_id": worker_id,
            "n_workers": n_workers,
            "gpu_id": str(gpu_id),
            "task_count": len(tasks),
            "task_keys_sha256": canonical_json_sha256(tasks),
        },
    )
    import torch
    from dataset_utils import load_dataset_from_request, resolve_dataset_request
    from deterministic_runtime import assert_strict_torch_determinism, deterministic_provenance
    import final_eval

    assert_strict_torch_determinism(torch)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; test30 worker refuses CPU fallback")
    device = torch.device("cuda:0")
    runtime_state = deterministic_provenance(torch, device=device)
    if runtime_state.get("gpu_uuid") in (None, ""):
        raise RuntimeError("cannot resolve worker GPU UUID")
    candidates = _candidate_map(root)
    bundles: dict[str, Any] = {}
    completed_count = 0
    status_path = worker_root / "status.tsv"
    if not status_path.exists():
        atomic_text(status_path, "timestamp\tdataset\tcandidate_fingerprint\ttest_seed\tstate\n")
    for task in tasks:
        path = task_result_path(root, task["dataset"], task["candidate_fingerprint"], task["test_seed"])
        temporary_matches = list(path.parent.glob(f".{path.name}.*.tmp")) if path.parent.exists() else []
        if temporary_matches:
            raise RuntimeError(f"stale temporary task files block resume: {temporary_matches}")
        if path.exists():
            validate_task_result(read_json(path), task, expected_worker=worker_id)
            completed_count += 1
            continue
        candidate = candidates[(task["dataset"], task["candidate_fingerprint"])]
        dataset = task["dataset"]
        if dataset not in bundles:
            request = resolve_dataset_request(
                dataset,
                str(DATA_ROOT),
                split_seed=0 if dataset == "dblp" else None,
            )
            bundle = load_dataset_from_request(request)
            bundle.to(device)
            bundles[dataset] = bundle
        bundle = bundles[dataset]
        architecture = candidate["architecture"]
        hp = candidate["hyperparameters"]
        config = {
            "operations": list(architecture["operations"]),
            "edges": [tuple(edge) for edge in architecture["edges"]],
            "effective_layers": sum(op != "Identity" for op in architecture["operations"]),
        }
        training_seed = expected_test_training_seed(task["candidate_fingerprint"], task["test_seed"])
        started = time.monotonic()
        result = final_eval.train_and_eval_arch(
            config=config,
            data=bundle.data,
            in_ch=bundle.num_features,
            out_ch=bundle.num_classes,
            lr=float(hp["lr"]),
            dropout=float(hp["dropout"]),
            hidden_dim=int(hp["hidden_dim"]),
            weight_decay=float(hp["l2"]),
            gcnii_alpha=0.1,
            gcnii_theta=0.5,
            gat_heads=int(hp.get("gat_heads", 1)),
            sage_aggr=str(hp.get("sage_aggr", "mean")),
            gin_eps=float(hp.get("gin_eps", 0.0)),
            gat_heads_by_layer=hp.get("gat_heads_by_layer"),
            sage_aggr_by_layer=hp.get("sage_aggr_by_layer"),
            gin_eps_by_layer=hp.get("gin_eps_by_layer"),
            device=device,
            max_epochs=150,
            patience=40,
            seed=training_seed,
            track_test=True,
            return_metadata=True,
            return_reproducibility_metadata=True,
        )
        val_acc, valid, test_acc, metadata = final_eval._unpack_training_result(result)
        if not valid:
            raise RuntimeError(f"test30 candidate training was invalid: {task}")
        row = {
            **task,
            "test_accuracy": _finite_accuracy(test_acc),
            "validation_accuracy": _finite_accuracy(val_acc),
            "training_seed": training_seed,
            "decoder_seed": int(candidate["decoder_seed"]),
            "architecture": architecture,
            "hyperparameters": hp,
            "epochs_ran": int(metadata.get("epochs_ran", 0)),
            "best_epoch": metadata.get("best_epoch"),
            "trajectory_hash": metadata.get("trajectory_hash"),
            "model_final_state_hash": metadata.get("model_final_state_hash"),
            "wall_time": float(time.monotonic() - started),
            "worker_id": worker_id,
            "gpu_name": runtime_state["gpu_name"],
            "gpu_uuid": runtime_state["gpu_uuid"],
            "deterministic_provenance": runtime_state,
            "completed": True,
        }
        validate_task_result(row, task, expected_worker=worker_id)
        atomic_json(path, row)
        completed_count += 1
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()}\t{task['dataset']}\t{task['candidate_fingerprint']}\t{task['test_seed']}\tcompleted\n")
    atomic_json(
        worker_root / "completion.json",
        {
            "source_id": identity["source_id"],
            "worker_id": worker_id,
            "n_workers": n_workers,
            "gpu_name": runtime_state["gpu_name"],
            "gpu_uuid": runtime_state["gpu_uuid"],
            "assigned_task_count": len(tasks),
            "completed_task_count": completed_count,
            "completed_at": utc_now(),
        },
    )
    return 0


def audit_test30(root: Path = PIPELINE_ROOT, *, write_marker: bool = True) -> dict[str, Any]:
    identity = current_source_identity(root)
    manifest_path = root / "final_test" / "canonical_task_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("source_id") != identity["source_id"]:
        raise RuntimeError("test manifest source ID mismatch")
    worker_sets = [
        {(row["dataset"], row["candidate_fingerprint"], int(row["test_seed"])) for row in assigned_tasks(manifest, worker, 2)}
        for worker in (0, 1)
    ]
    canonical_set = {
        (row["dataset"], row["candidate_fingerprint"], int(row["test_seed"]))
        for row in manifest["tasks"]
    }
    if worker_sets[0] & worker_sets[1] or worker_sets[0] | worker_sets[1] != canonical_set:
        raise AssertionError("two-worker assignment is not a disjoint exact partition")
    rows = []
    gpu_by_worker: dict[int, set[str]] = defaultdict(set)
    seeds_by_candidate_worker: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    for task in manifest["tasks"]:
        path = task_result_path(root, task["dataset"], task["candidate_fingerprint"], task["test_seed"])
        if not path.is_file():
            raise FileNotFoundError(f"missing task result: {path}")
        expected_worker = int(task["test_seed"]) % 2
        row = validate_task_result(read_json(path), task, expected_worker=expected_worker)
        rows.append(row)
        gpu_by_worker[expected_worker].add(str(row["gpu_uuid"]))
        seeds_by_candidate_worker[(task["dataset"], task["candidate_fingerprint"], expected_worker)].add(int(task["test_seed"]))
    for (dataset, fingerprint), candidate_rows in _group(rows, "dataset", "candidate_fingerprint").items():
        seeds = sorted(int(row["test_seed"]) for row in candidate_rows)
        if seeds != list(TEST_SEEDS):
            raise ValueError(f"{dataset}/{fingerprint} does not have test seeds 0..29")
        for worker in (0, 1):
            if len(seeds_by_candidate_worker[(dataset, fingerprint, worker)]) != 15:
                raise ValueError(f"{dataset}/{fingerprint} worker {worker} does not have 15 seeds")
    for worker in (0, 1):
        if len(gpu_by_worker[worker]) != 1:
            raise ValueError(f"worker {worker} used multiple or missing GPU UUIDs")
        completion = read_json(root / "final_test" / "workers" / f"worker_{worker}" / "completion.json")
        if completion.get("gpu_uuid") not in gpu_by_worker[worker]:
            raise ValueError(f"worker {worker} GPU manifest mismatch")
    payload = {
        "status": "passed",
        "source_id": identity["source_id"],
        "task_count": len(rows),
        "unique_candidate_count": int(manifest["unique_candidate_count"]),
        "worker_task_counts": {"0": len(worker_sets[0]), "1": len(worker_sets[1])},
        "worker_gpu_uuids": {str(worker): sorted(values) for worker, values in gpu_by_worker.items()},
        "partition_rule": "test_seed % 2",
        "task_results_sha256": canonical_json_sha256(
            [{"key": [row["dataset"], row["candidate_fingerprint"], row["test_seed"]], "sha256": sha256_file(task_result_path(root, row["dataset"], row["candidate_fingerprint"], row["test_seed"]))} for row in rows]
        ),
    }
    audit_path = root / "final_test" / "completion_audit.json"
    atomic_json(audit_path, payload)
    if write_marker:
        write_gate(
            root / "FINAL_TEST_AUDIT_PASSED",
            source_id=identity["source_id"],
            input_sha256=sha256_file(manifest_path),
            output_sha256=sha256_file(audit_path),
            details={"task_count": len(rows), "partition_rule": "test_seed % 2"},
        )
    return payload


def _group(rows: Iterable[Mapping[str, Any]], *fields: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    output: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[tuple(row[field] for field in fields)].append(dict(row))
    return output


def _sample_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def select_test_oracle_winner(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("cannot select a test-oracle winner from an empty set")
    return min(
        candidates,
        key=lambda row: (
            -float(row["mean_test_accuracy"]),
            float(row["sample_std_test_accuracy"]),
            str(row["candidate_fingerprint"]),
        ),
    )


def summarize_test30(root: Path = PIPELINE_ROOT) -> dict[str, Any]:
    identity = current_source_identity(root)
    manifest_path = root / "final_test" / "canonical_task_manifest.json"
    load_gate(
        root / "FINAL_TEST_AUDIT_PASSED",
        expected_source_id=identity["source_id"],
        expected_input_hash=sha256_file(manifest_path),
    )
    audit_test30(root, write_marker=False)
    manifest = read_json(manifest_path)
    task_by_key = {
        (row["dataset"], row["candidate_fingerprint"], int(row["test_seed"])): row
        for row in manifest["tasks"]
    }
    rows = [
        validate_task_result(
            read_json(task_result_path(root, dataset, fingerprint, seed)),
            task,
            expected_worker=seed % 2,
        )
        for (dataset, fingerprint, seed), task in sorted(task_by_key.items())
    ]
    test_long_path = root / "final_test" / "test_long.csv"
    write_csv(test_long_path, rows, TEST_LONG_FIELDS)
    unique_candidates = _candidate_map(root)
    grouped_results = _group(rows, "dataset", "candidate_fingerprint")
    candidate_summaries = []
    candidate_origins = []
    source_rows = read_csv(root / "top10" / "top10_source_candidates.csv")
    for origin in source_rows:
        candidate_origins.append(dict(origin))
    for key, candidate_rows in sorted(grouped_results.items()):
        candidate = unique_candidates[key]
        accuracies = [float(row["test_accuracy"]) for row in sorted(candidate_rows, key=lambda row: int(row["test_seed"]))]
        if len(accuracies) != 30:
            raise ValueError(f"candidate {key} does not have 30 results")
        candidate_summaries.append(
            {
                "dataset": key[0],
                "candidate_fingerprint": key[1],
                "architecture": candidate["architecture"],
                "hyperparameters": candidate["hyperparameters"],
                "mean_test_accuracy": float(statistics.fmean(accuracies)),
                "sample_std_test_accuracy": _sample_std(accuracies),
                "median_test_accuracy": float(statistics.median(accuracies)),
                "min_test_accuracy": min(accuracies),
                "max_test_accuracy": max(accuracies),
                "n_test_seeds": 30,
            }
        )
    summary_fields = (
        "dataset", "candidate_fingerprint", "architecture", "hyperparameters",
        "mean_test_accuracy", "sample_std_test_accuracy", "median_test_accuracy",
        "min_test_accuracy", "max_test_accuracy", "n_test_seeds",
    )
    write_csv(root / "final_test" / "candidate_test_summary.csv", candidate_summaries, summary_fields)
    write_csv(root / "final_test" / "candidate_origins.csv", candidate_origins, SOURCE_FIELDS)
    summary_by_key = {(row["dataset"], row["candidate_fingerprint"]): row for row in candidate_summaries}
    origins_by_dataset_method: dict[tuple[str, str], set[str]] = defaultdict(set)
    for origin in source_rows:
        origins_by_dataset_method[(origin["dataset"], origin["method"])].add(origin["candidate_fingerprint"])
    winner_rows = []
    winner_seed_rows = []
    winner_origin_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            fingerprints = origins_by_dataset_method[(dataset, method)]
            if not fingerprints:
                raise ValueError(f"no Top-10 union for {dataset}/{method}")
            winner_summary = select_test_oracle_winner(
                summary_by_key[(dataset, fingerprint)] for fingerprint in fingerprints
            )
            fingerprint = winner_summary["candidate_fingerprint"]
            candidate = unique_candidates[(dataset, fingerprint)]
            method_origins = [
                origin
                for origin in candidate["all_origins"]
                if origin["method"] == method
            ]
            canonical = min(
                method_origins,
                key=lambda row: (
                    -float(row["validation_accuracy"]),
                    int(row["search_seed"]),
                    int(row["budget_1based"]),
                    str(row["method"]),
                ),
            )
            all_origins = candidate["all_origins"]
            winner_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "candidate_fingerprint": fingerprint,
                    "architecture": winner_summary["architecture"],
                    "hyperparameters": winner_summary["hyperparameters"],
                    "mean_test_accuracy": winner_summary["mean_test_accuracy"],
                    "sample_std_test_accuracy": winner_summary["sample_std_test_accuracy"],
                    "median_test_accuracy": winner_summary["median_test_accuracy"],
                    "min_test_accuracy": winner_summary["min_test_accuracy"],
                    "max_test_accuracy": winner_summary["max_test_accuracy"],
                    "canonical_source_search_seed": canonical["search_seed"],
                    "canonical_source_validation_rank": canonical["validation_rank"],
                    "canonical_source_validation_accuracy": canonical["validation_accuracy"],
                    "canonical_source_raw_stage": canonical["raw_stage"],
                    "canonical_source_normalized_stage": canonical["normalized_stage"],
                    "canonical_source_full_evaluation_index_0based": canonical["full_evaluation_index_0based"],
                    "canonical_source_budget_1based": canonical["budget_1based"],
                    "canonical_source_stage_index_1based": canonical["stage_index_1based"],
                    "all_source_search_seeds": sorted({int(row["search_seed"]) for row in all_origins}),
                    "all_source_methods": sorted({str(row["method"]) for row in all_origins}),
                    "all_source_budgets": sorted({int(row["budget_1based"]) for row in all_origins}),
                    "n_test_seeds": 30,
                }
            )
            winner_origin_rows.extend(
                {**origin, "winner_method": method}
                for origin in method_origins
            )
            seed_rows = sorted(grouped_results[(dataset, fingerprint)], key=lambda row: int(row["test_seed"]))
            for row in seed_rows:
                winner_seed_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "candidate_fingerprint": fingerprint,
                        "test_seed": int(row["test_seed"]),
                        "test_accuracy": float(row["test_accuracy"]),
                    }
                )
    if len(winner_rows) != 12 or len(winner_seed_rows) != 360:
        raise AssertionError("method oracle summary must contain 12 winners and 360 seed rows")
    winner_fields = (
        "dataset", "method", "candidate_fingerprint", "architecture", "hyperparameters",
        "mean_test_accuracy", "sample_std_test_accuracy", "median_test_accuracy",
        "min_test_accuracy", "max_test_accuracy", "canonical_source_search_seed",
        "canonical_source_validation_rank", "canonical_source_validation_accuracy",
        "canonical_source_raw_stage", "canonical_source_normalized_stage",
        "canonical_source_full_evaluation_index_0based", "canonical_source_budget_1based",
        "canonical_source_stage_index_1based", "all_source_search_seeds",
        "all_source_methods", "all_source_budgets", "n_test_seeds",
    )
    winners_path = root / "final_test" / "method_test_oracle_winners.csv"
    winner_seeds_path = root / "final_test" / "method_test_oracle_winner_seed_accuracies.csv"
    write_csv(winners_path, winner_rows, winner_fields)
    write_csv(
        winner_seeds_path,
        winner_seed_rows,
        ("dataset", "method", "candidate_fingerprint", "test_seed", "test_accuracy"),
    )
    origin_fields = ("winner_method", *SOURCE_FIELDS)
    write_csv(root / "final_test" / "method_test_oracle_winner_origins.csv", winner_origin_rows, origin_fields)
    report_lines = [
        "# Method-level test-oracle diagnostic",
        "",
        "selection_used_test = true  ",
        "result_type = test_oracle_diagnostic  ",
        "paper_claim_eligible = false",
        "",
        "Winners are selected by the mean of 30 test results. They are not an unbiased final-test estimate and are not eligible for SOTA or paper generalization claims.",
        "",
        "| dataset | method | candidate | test mean ± sample std | source seed | validation rank | raw / normalized stage | full budget / stage index |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    result_lookup = _group(rows, "dataset", "candidate_fingerprint")
    for winner in winner_rows:
        report_lines.append(
            f"| {winner['dataset']} | {winner['method']} | `{winner['candidate_fingerprint']}` | "
            f"{float(winner['mean_test_accuracy']):.12g} ± {float(winner['sample_std_test_accuracy']):.12g} | "
            f"{winner['canonical_source_search_seed']} | {winner['canonical_source_validation_rank']} | "
            f"{winner['canonical_source_raw_stage']} / {winner['canonical_source_normalized_stage']} | "
            f"{winner['canonical_source_budget_1based']} / {winner['canonical_source_stage_index_1based']} |"
        )
    for winner in winner_rows:
        key = (winner["dataset"], winner["candidate_fingerprint"])
        seed_rows = sorted(result_lookup[key], key=lambda row: int(row["test_seed"]))
        report_lines.extend(
            [
                "",
                f"## {winner['dataset']} / {winner['method']}",
                "",
                f"- Candidate: `{winner['candidate_fingerprint']}`",
                f"- Architecture: `{json.dumps(winner['architecture'], sort_keys=True)}`",
                f"- Hyperparameters: `{json.dumps(winner['hyperparameters'], sort_keys=True)}`",
                f"- Source: search seed {winner['canonical_source_search_seed']}, validation rank {winner['canonical_source_validation_rank']}, raw stage {winner['canonical_source_raw_stage']}, normalized stage {winner['canonical_source_normalized_stage']}, full evaluation {winner['canonical_source_budget_1based']}, stage index {winner['canonical_source_stage_index_1based']}",
                "- Test accuracies (seed: accuracy): " + ", ".join(f"{int(row['test_seed'])}: {float(row['test_accuracy']):.12g}" for row in seed_rows),
                "- GPU UUIDs (seed: UUID): " + ", ".join(f"{int(row['test_seed'])}: {row['gpu_uuid']}" for row in seed_rows),
            ]
        )
    report_path = root / "final_test" / "report.md"
    atomic_text(report_path, "\n".join(report_lines) + "\n")
    reproducibility = {
        "source_id": identity["source_id"],
        "config_fingerprint": manifest["config_fingerprint"],
        "canonical_task_manifest_sha256": sha256_file(manifest_path),
        "test_long_sha256": sha256_file(test_long_path),
        "selection_used_test": True,
        "result_type": "test_oracle_diagnostic",
        "paper_claim_eligible": False,
    }
    atomic_json(root / "final_test" / "reproducibility_manifest.json", reproducibility)
    output_hash = canonical_json_sha256(
        {path.name: sha256_file(path) for path in (test_long_path, winners_path, winner_seeds_path, report_path)}
    )
    write_gate(
        root / "FINAL_SUMMARY_PASSED",
        source_id=identity["source_id"],
        input_sha256=sha256_file(root / "FINAL_TEST_AUDIT_PASSED"),
        output_sha256=output_hash,
        details={"winner_rows": 12, "winner_seed_rows": 360},
    )
    return {"status": "passed", "winner_rows": 12, "winner_seed_rows": 360, "output_sha256": output_hash}


def monitor(root: Path = PIPELINE_ROOT) -> dict[str, Any]:
    matrix_tasks = 0
    matrix = root / "formal_matrix_60.json"
    if matrix.is_file():
        matrix_tasks = len(read_json(matrix).get("tasks", []))
    manifest_tasks = 0
    task_results = 0
    manifest = root / "final_test" / "canonical_task_manifest.json"
    if manifest.is_file():
        manifest_tasks = int(read_json(manifest).get("task_count", 0))
    task_root = root / "final_test" / "task_results"
    if task_root.is_dir():
        task_results = sum(1 for path in task_root.rglob("seed_*.json") if path.is_file())
    return {
        "formal_matrix_tasks": matrix_tasks,
        "formal_search_gate": (root / "FORMAL_SEARCH_AUDIT_PASSED").is_file(),
        "top10_gate": (root / "TOP10_EXTRACTION_AUDIT_PASSED").is_file(),
        "test_tasks_total": manifest_tasks,
        "test_task_results": task_results,
        "final_test_gate": (root / "FINAL_TEST_AUDIT_PASSED").is_file(),
        "summary_gate": (root / "FINAL_SUMMARY_PASSED").is_file(),
    }


def package_results(root: Path = PIPELINE_ROOT) -> dict[str, Any]:
    identity = current_source_identity(root)
    load_gate(root / "FINAL_SUMMARY_PASSED", expected_source_id=identity["source_id"])
    required = [
        root / "freeze",
        root / "formal_matrix_60.json",
        root / "formal_matrix_60.csv",
        root / "formal_matrix_audit.json",
        root / "determinism_smoke_runs.csv",
        root / "determinism_smoke_report.json",
        root / "determinism_smoke_report.md",
        root / "DETERMINISM_ACCEPTED",
        root / "top10",
        root / "final_test" / "canonical_task_manifest.json",
        root / "final_test" / "completion_audit.json",
        root / "final_test" / "test_long.csv",
        root / "final_test" / "candidate_test_summary.csv",
        root / "final_test" / "candidate_origins.csv",
        root / "final_test" / "method_test_oracle_winners.csv",
        root / "final_test" / "method_test_oracle_winner_seed_accuracies.csv",
        root / "final_test" / "method_test_oracle_winner_origins.csv",
        root / "final_test" / "report.md",
        root / "scripts",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    included: set[Path] = set()
    for path in required:
        if path.is_file():
            included.add(path)
        else:
            included.update(candidate for candidate in path.rglob("*") if candidate.is_file())
    matrix = read_json(root / "formal_matrix_60.json")
    for task in matrix["tasks"]:
        for name in ("history_final.json", "history_metadata.json", "run_provenance.json"):
            path = Path(task["output_directory"]) / name
            if not path.is_file():
                raise FileNotFoundError(path)
            included.add(path)
    for path in (root / "logs").rglob("*"):
        if path.is_file():
            included.add(path)
    manifest_rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(included)
    ]
    manifest_path = root / "package_sha256_manifest.tsv"
    atomic_text(
        manifest_path,
        "path\tsize_bytes\tsha256\n"
        + "".join(
            f"{row['path']}\t{row['size_bytes']}\t{row['sha256']}\n"
            for row in manifest_rows
        ),
    )
    included.add(manifest_path)
    archive = root / f"deterministic_three_strategy_{identity['source_id'][:12]}.tar.gz"
    digest_path = archive.with_name(archive.name + ".sha256")
    if archive.exists() or digest_path.exists():
        raise FileExistsError("refusing to overwrite existing package")
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(included):
            handle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    digest = sha256_file(archive)
    atomic_text(digest_path, f"{digest}  {archive.name}\n")
    return {"archive": str(archive), "sha256": digest, "file_count": len(included)}


def write_mock_task_results(root: Path) -> None:
    manifest = read_json(root / "final_test" / "canonical_task_manifest.json")
    candidate_map = _candidate_map(root)
    for task in manifest["tasks"]:
        candidate = candidate_map[(task["dataset"], task["candidate_fingerprint"])]
        seed = int(task["test_seed"])
        worker = seed % 2
        accuracy = 0.5 + (int(task["candidate_fingerprint"][:8], 16) % 3000) / 10_000 + seed / 1_000_000
        state = {
            "deterministic_algorithms_enabled": True,
            "deterministic_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "torch_version": "mock",
            "cuda_version": "mock",
            "cudnn_version": 1,
            "pyg_version": "mock",
            "gpu_name": f"Mock GPU {worker}",
            "gpu_uuid": f"GPU-MOCK-{worker}",
            "driver_version": "mock",
        }
        row = {
            **task,
            "test_accuracy": min(accuracy, 0.999),
            "training_seed": expected_test_training_seed(task["candidate_fingerprint"], seed),
            "decoder_seed": candidate["decoder_seed"],
            "architecture": candidate["architecture"],
            "hyperparameters": candidate["hyperparameters"],
            "epochs_ran": 10,
            "best_epoch": 5,
            "trajectory_hash": hashlib.sha256(f"trajectory-{task}".encode()).hexdigest(),
            "model_final_state_hash": hashlib.sha256(f"model-{task}".encode()).hexdigest(),
            "wall_time": 0.01,
            "worker_id": worker,
            "gpu_name": state["gpu_name"],
            "gpu_uuid": state["gpu_uuid"],
            "deterministic_provenance": state,
            "completed": True,
        }
        atomic_json(task_result_path(root, task["dataset"], task["candidate_fingerprint"], seed), row)
    for worker in (0, 1):
        tasks = assigned_tasks(manifest, worker, 2)
        worker_root = root / "final_test" / "workers" / f"worker_{worker}"
        atomic_json(
            worker_root / "completion.json",
            {
                "source_id": manifest["source_id"],
                "worker_id": worker,
                "n_workers": 2,
                "gpu_name": f"Mock GPU {worker}",
                "gpu_uuid": f"GPU-MOCK-{worker}",
                "assigned_task_count": len(tasks),
                "completed_task_count": len(tasks),
            },
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PIPELINE_ROOT))
    parser.add_argument(
        "--python-executable",
        default=str(PYTHON_BIN),
        help="Python executable used by generated launchers and subprocesses",
    )
    parser.add_argument(
        "--training-mode-decisions",
        default=str(TRAINING_MODE_DECISIONS),
        help=(
            "formal training-mode decision artifact; relative paths are "
            "resolved from the repository root"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("initialize-dual")
    pre = sub.add_parser("preflight")
    pre.add_argument("--allow-missing-acceptance", action="store_true")
    sub.add_parser("freeze")
    sub.add_parser("run-dual-resource-smoke")
    search = sub.add_parser("run-search-queue")
    search.add_argument("--gpu-id", required=True)
    search.add_argument("--expected-gpu-uuid")
    search.add_argument("--worker-id", type=int)
    search.add_argument("--task-manifest")
    search.add_argument("--status-path")
    search.add_argument("--current-path")
    search.add_argument("--lock-path")
    sub.add_parser("audit-searches")
    sub.add_parser("extract-top10")
    test = sub.add_parser("run-test30-worker")
    test.add_argument("--gpu-id", required=True)
    test.add_argument("--worker-id", required=True, type=int)
    test.add_argument("--n-workers", required=True, type=int)
    sub.add_parser("audit-test30")
    sub.add_parser("summarize")
    sub.add_parser("monitor")
    sub.add_parser("package")
    sub.add_parser("build-test-manifest")
    sub.add_parser("write-mock-results")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_runtime_paths(
        python_executable=args.python_executable,
        training_mode_decisions=args.training_mode_decisions,
    )
    root = Path(args.root).resolve()
    if args.command == "initialize-dual":
        result = initialize_dual_pipeline(root)
    elif args.command == "preflight":
        result = preflight(require_acceptance=not args.allow_missing_acceptance, root=root)
    elif args.command == "freeze":
        result = freeze_pipeline(root)
    elif args.command == "run-dual-resource-smoke":
        result = run_dual_gpu_resource_smoke(root)
    elif args.command == "run-search-queue":
        return run_search_queue(
            args.gpu_id,
            root,
            task_manifest=None if args.task_manifest is None else Path(args.task_manifest),
            worker_id=args.worker_id,
            expected_gpu_uuid=args.expected_gpu_uuid,
            status_path=None if args.status_path is None else Path(args.status_path),
            current_path=None if args.current_path is None else Path(args.current_path),
            lock_path=None if args.lock_path is None else Path(args.lock_path),
        )
    elif args.command == "audit-searches":
        result = audit_formal_searches(root)
    elif args.command == "extract-top10":
        result = extract_validation_top10(root)
    elif args.command == "run-test30-worker":
        return run_test30_worker(args.gpu_id, args.worker_id, args.n_workers, root)
    elif args.command == "audit-test30":
        result = audit_test30(root)
    elif args.command == "summarize":
        result = summarize_test30(root)
    elif args.command == "monitor":
        result = monitor(root)
    elif args.command == "package":
        result = package_results(root)
    elif args.command == "build-test-manifest":
        result = build_test_task_manifest(root)
    else:
        write_mock_task_results(root)
        result = {"status": "passed"}
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
