"""Strict PyTorch determinism configuration and runtime provenance.

The cuBLAS workspace variable is configured without importing PyTorch so entry
points can call :func:`prepare_deterministic_environment` before the first
possible CUDA initialization.  PyTorch backend flags are then enabled by
``configure_torch_determinism`` immediately after importing torch.
"""

from __future__ import annotations

import csv
import os
import subprocess
from typing import Any


SUPPORTED_CUBLAS_WORKSPACE_CONFIGS = (":4096:8", ":16:8")
DEFAULT_CUBLAS_WORKSPACE_CONFIG = SUPPORTED_CUBLAS_WORKSPACE_CONFIGS[0]


def prepare_deterministic_environment() -> str:
    """Set or validate the cuBLAS deterministic workspace configuration."""

    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is None:
        configured = DEFAULT_CUBLAS_WORKSPACE_CONFIG
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = configured
    if configured not in SUPPORTED_CUBLAS_WORKSPACE_CONFIGS:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be one of "
            f"{SUPPORTED_CUBLAS_WORKSPACE_CONFIGS}, got {configured!r}"
        )
    return configured


def configure_torch_determinism(torch_module: Any) -> None:
    """Enable strict deterministic algorithms without changing TF32 or AMP."""

    prepare_deterministic_environment()
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    assert_strict_torch_determinism(torch_module)


def assert_strict_torch_determinism(torch_module: Any) -> None:
    """Raise a clear error if any required PyTorch flag is not strict."""

    if not bool(torch_module.are_deterministic_algorithms_enabled()):
        raise RuntimeError("torch deterministic algorithms are not enabled")
    warn_only_getter = getattr(
        torch_module, "is_deterministic_algorithms_warn_only_enabled", None
    )
    if warn_only_getter is not None and bool(warn_only_getter()):
        raise RuntimeError("torch deterministic algorithms are in warn-only mode")
    if not bool(torch_module.backends.cudnn.deterministic):
        raise RuntimeError("torch.backends.cudnn.deterministic is not enabled")
    if bool(torch_module.backends.cudnn.benchmark):
        raise RuntimeError("torch.backends.cudnn.benchmark must be disabled")
    prepare_deterministic_environment()


def _nvidia_rows() -> tuple[list[dict[str, Any]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return [], (
            f"exit_code={completed.returncode}: "
            f"{completed.stderr.strip() or 'no stderr'}"
        )
    rows: list[dict[str, Any]] = []
    for parsed in csv.reader(completed.stdout.splitlines()):
        if len(parsed) != 4:
            return [], f"unparseable nvidia-smi row: {parsed!r}"
        index, uuid, name, driver = (value.strip() for value in parsed)
        try:
            numeric_index = int(index)
        except ValueError:
            return [], f"invalid nvidia-smi GPU index: {index!r}"
        rows.append(
            {
                "index": numeric_index,
                "gpu_uuid": uuid,
                "gpu_name": name,
                "driver_version": driver,
            }
        )
    return rows, None


def _visible_physical_gpu_index(logical_index: int) -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return int(logical_index)
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if not 0 <= int(logical_index) < len(tokens):
        return None
    token = tokens[int(logical_index)]
    try:
        return int(token)
    except ValueError:
        return None


def deterministic_provenance(
    torch_module: Any,
    *,
    device: Any | None = None,
) -> dict[str, Any]:
    """Return the required strict-state and software/hardware provenance."""

    assert_strict_torch_determinism(torch_module)
    warn_only_getter = getattr(
        torch_module, "is_deterministic_algorithms_warn_only_enabled", None
    )
    cudnn_version = torch_module.backends.cudnn.version()
    result: dict[str, Any] = {
        "deterministic_algorithms_enabled": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": (
            False if warn_only_getter is None else bool(warn_only_getter())
        ),
        "cudnn_deterministic": bool(torch_module.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "CUBLAS_WORKSPACE_CONFIG": prepare_deterministic_environment(),
        "torch_version": str(torch_module.__version__),
        "cuda_version": (
            None
            if torch_module.version.cuda is None
            else str(torch_module.version.cuda)
        ),
        "cudnn_version": None if cudnn_version is None else int(cudnn_version),
        "pyg_version": None,
        "gpu_name": None,
        "gpu_uuid": None,
        "driver_version": None,
        "nvidia_smi_error": None,
    }
    try:
        import torch_geometric

        result["pyg_version"] = str(torch_geometric.__version__)
    except Exception as exc:  # provenance must retain the exact collection error
        result["pyg_version_error"] = f"{type(exc).__name__}: {exc}"

    rows, error = _nvidia_rows()
    result["nvidia_smi_error"] = error
    if device is None or getattr(device, "type", None) != "cuda":
        return result
    logical_index = 0 if getattr(device, "index", None) is None else int(device.index)
    physical_index = _visible_physical_gpu_index(logical_index)
    selected = None
    if physical_index is not None:
        selected = next(
            (row for row in rows if int(row["index"]) == physical_index), None
        )
    if selected is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        token = tokens[logical_index] if logical_index < len(tokens) else None
        if token is not None:
            selected = next(
                (row for row in rows if row["gpu_uuid"] == token), None
            )
    if selected is not None:
        result.update(
            {
                "gpu_name": selected["gpu_name"],
                "gpu_uuid": selected["gpu_uuid"],
                "driver_version": selected["driver_version"],
            }
        )
    else:
        # This call occurs only after the device has already been selected.
        result["gpu_name"] = str(torch_module.cuda.get_device_name(logical_index))
    return result
