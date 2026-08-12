"""Runtime provenance collection for Phase4 search and diagnostics.

The collector is intentionally observational: it reads process/backend state,
hashes source/checkpoint bytes, and invokes read-only Git/NVIDIA queries.  It
does not seed or sample Python, NumPy, Torch, or CUDA random number generators.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence


RUN_PROVENANCE_FORMAT_VERSION = 1
DEFAULT_PHASE4_SOURCE_PATHS = (
    "analyse/plot_gp_prediction_records.py",
    "bo_phase4.py",
    "configs/cross_dataset_methods.json",
    "cross_dataset_runner.py",
    "dataset_utils.py",
    "deterministic_runtime.py",
    "eval_utils.py",
    "evaluation_errors.py",
    "hp_modes.py",
    "initialization_gmm_schur.py",
    "initialization_wgmm_ted.py",
    "models.py",
    "nas_space.py",
    "run_provenance.py",
    "weighted_diag_gmm_init.py",
    "surrogate/accuracy_gp.py",
    "surrogate/checkpoint_io.py",
    "surrogate/dkl_accuracy_gp.py",
    "surrogate/history_dataset.py",
    "surrogate/metrics.py",
)
DEPENDENCY_DISTRIBUTIONS = {
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "ogb": "ogb",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit_learn": "scikit-learn",
    "gpytorch": "gpytorch",
    "botorch": "botorch",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def _run_read_only_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float = 15.0,
) -> tuple[bytes | None, str | None]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_seconds),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        return (
            None,
            f"exit_code={completed.returncode}: {stderr or 'no stderr'}",
        )
    return completed.stdout, None


def _git_text(
    arguments: Sequence[str],
    *,
    repo_root: Path,
    errors: list[str],
    field: str,
) -> str | None:
    payload, error = _run_read_only_command(
        ["git", *arguments],
        cwd=repo_root,
    )
    if error is not None:
        errors.append(f"{field}: {error}")
        return None
    assert payload is not None
    return payload.decode("utf-8", errors="replace").strip()


def _git_diff_sha256(
    arguments: Sequence[str],
    *,
    repo_root: Path,
    errors: list[str],
    field: str,
) -> str | None:
    payload, error = _run_read_only_command(
        ["git", *arguments],
        cwd=repo_root,
        timeout_seconds=60.0,
    )
    if error is not None:
        errors.append(f"{field}: {error}")
        return None
    assert payload is not None
    return sha256_bytes(payload)


def _dependency_versions(errors: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for field, distribution in DEPENDENCY_DISTRIBUTIONS.items():
        try:
            versions[field] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[field] = None
            errors.append(
                f"dependency_versions.{field}: distribution "
                f"{distribution!r} is not installed"
            )
        except Exception as exc:
            versions[field] = None
            errors.append(
                f"dependency_versions.{field}: {type(exc).__name__}: {exc}"
            )
    return versions


def _torch_backend_state(errors: list[str]) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        errors.append(f"torch_backend: {type(exc).__name__}: {exc}")
        return {
            "torch_version": None,
            "torch_cuda_version": None,
            "cudnn_version": None,
            "deterministic_algorithms_enabled": None,
            "deterministic_algorithms_warn_only": None,
            "cudnn_deterministic": None,
            "cudnn_benchmark": None,
            "cuda_matmul_allow_tf32": None,
            "cudnn_allow_tf32": None,
            "cuda_available": None,
        }

    warn_only_getter = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        None,
    )
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception as exc:
        cudnn_version = None
        errors.append(f"torch_backend.cudnn_version: {type(exc).__name__}: {exc}")
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": (
            None if cudnn_version is None else int(cudnn_version)
        ),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            None if warn_only_getter is None else bool(warn_only_getter())
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _nvidia_state(
    *,
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    payload, error = _run_read_only_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo_root,
    )
    if error is not None:
        errors.append(f"nvidia_smi: {error}")
        return {
            "available": False,
            "driver_version": None,
            "gpus": [],
            "error": error,
        }
    assert payload is not None
    rows: list[dict[str, Any]] = []
    driver_versions: set[str] = set()
    for raw_line in payload.decode("utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in raw_line.split(",", maxsplit=3)]
        if len(parts) != 4:
            errors.append(f"nvidia_smi: unparseable row {raw_line!r}")
            continue
        index_text, gpu_uuid, name, driver_version = parts
        try:
            index = int(index_text)
        except ValueError:
            errors.append(f"nvidia_smi: invalid GPU index {index_text!r}")
            continue
        rows.append(
            {
                "index": index,
                "uuid": gpu_uuid,
                "name": name,
                "driver_version": driver_version,
            }
        )
        driver_versions.add(driver_version)
    return {
        "available": bool(rows),
        "driver_version": (
            next(iter(driver_versions))
            if len(driver_versions) == 1
            else sorted(driver_versions)
        ),
        "gpus": rows,
        "error": None,
    }


def _resolve_source_paths(
    repo_root: Path,
    source_paths: Iterable[str | os.PathLike[str]] | None,
) -> list[Path]:
    requested = (
        DEFAULT_PHASE4_SOURCE_PATHS
        if source_paths is None
        else tuple(source_paths)
    )
    resolved: dict[str, Path] = {}
    for raw_path in requested:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"runtime-critical source {path} is outside repository {repo_root}"
            ) from exc
        resolved[relative] = path
    return [resolved[key] for key in sorted(resolved)]


def _source_manifest(
    *,
    repo_root: Path,
    source_paths: Iterable[str | os.PathLike[str]] | None,
    errors: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in _resolve_source_paths(repo_root, source_paths):
        relative = path.relative_to(repo_root).as_posix()
        if not path.is_file():
            rows.append(
                {
                    "path": relative,
                    "sha256": None,
                    "size_bytes": None,
                    "error": "missing_file",
                }
            )
            errors.append(
                f"runtime_critical_source_manifest.{relative}: missing file"
            )
            continue
        try:
            rows.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                    "error": None,
                }
            )
        except OSError as exc:
            rows.append(
                {
                    "path": relative,
                    "sha256": None,
                    "size_bytes": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            errors.append(
                f"runtime_critical_source_manifest.{relative}: "
                f"{type(exc).__name__}: {exc}"
            )
    return {
        "files": rows,
        "manifest_sha256": _canonical_json_sha256(rows),
    }


def _checkpoint_state(
    checkpoint_path: str | os.PathLike[str] | None,
    *,
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    if checkpoint_path is None or not str(checkpoint_path).strip():
        error = "checkpoint path is empty"
        errors.append(f"checkpoint: {error}")
        return {"path": None, "sha256": None, "size_bytes": None, "error": error}
    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        error = f"checkpoint does not exist: {path}"
        errors.append(f"checkpoint: {error}")
        return {
            "path": str(path),
            "sha256": None,
            "size_bytes": None,
            "error": error,
        }
    try:
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "error": None,
        }
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
        errors.append(f"checkpoint: {error}")
        return {
            "path": str(path),
            "sha256": None,
            "size_bytes": None,
            "error": error,
        }


def collect_run_provenance(
    *,
    repo_root: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str] | None,
    argv: Sequence[str] | None = None,
    seed_derivation_version: str,
    source_paths: Iterable[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """Collect JSON-safe runtime provenance without changing runtime state."""

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError(f"repository root does not exist: {root}")
    errors: list[str] = []
    git_status = _git_text(
        ["status", "--porcelain=v1", "--untracked-files=normal"],
        repo_root=root,
        errors=errors,
        field="git_status",
    )
    torch_backend = _torch_backend_state(errors)
    return {
        "format_version": RUN_PROVENANCE_FORMAT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "argv": list(sys.argv if argv is None else argv),
        "repository_root": str(root),
        "git": {
            "branch": _git_text(
                ["branch", "--show-current"],
                repo_root=root,
                errors=errors,
                field="git_branch",
            ),
            "commit": _git_text(
                ["rev-parse", "HEAD"],
                repo_root=root,
                errors=errors,
                field="git_commit",
            ),
            "dirty_worktree": (
                None if git_status is None else bool(git_status)
            ),
            "status_porcelain_sha256": (
                None
                if git_status is None
                else sha256_bytes(git_status.encode("utf-8"))
            ),
            "staged_diff_sha256": _git_diff_sha256(
                ["diff", "--cached", "--binary", "--no-ext-diff"],
                repo_root=root,
                errors=errors,
                field="staged_diff_sha256",
            ),
            "unstaged_diff_sha256": _git_diff_sha256(
                ["diff", "--binary", "--no-ext-diff"],
                repo_root=root,
                errors=errors,
                field="unstaged_diff_sha256",
            ),
        },
        "runtime_critical_source_manifest": _source_manifest(
            repo_root=root,
            source_paths=source_paths,
            errors=errors,
        ),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "dependency_versions": _dependency_versions(errors),
        "torch_backend": torch_backend,
        "nvidia": _nvidia_state(repo_root=root, errors=errors),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
        "checkpoint": _checkpoint_state(
            checkpoint_path,
            repo_root=root,
            errors=errors,
        ),
        "seed_derivation_version": str(seed_derivation_version),
        "errors": errors,
    }


def atomic_json_dump(
    value: Any,
    path: str | os.PathLike[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def persist_run_provenance(
    output_directory: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str] | None,
    argv: Sequence[str] | None,
    seed_derivation_version: str,
    source_paths: Iterable[str | os.PathLike[str]] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Write ``run_provenance.json`` once, preserving it on resume.

    Returns ``(payload, created)``.  An existing artifact is read and returned
    unchanged so a resume cannot rewrite the original run's provenance.
    """

    path = Path(output_directory) / "run_provenance.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, dict):
            raise ValueError(f"existing provenance is not a JSON object: {path}")
        return existing, False
    payload = collect_run_provenance(
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        argv=argv,
        seed_derivation_version=seed_derivation_version,
        source_paths=source_paths,
    )
    atomic_json_dump(payload, path)
    return payload, True
