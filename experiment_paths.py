"""Validated, repository-relative locations for experiment outputs.

This module deliberately handles storage identity only.  Protocol IDs and paths
must never participate in candidate fingerprints, random seeds, or budgets.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re


_PROTOCOL_RE = re.compile(
    r"^(?P<experiment>[a-z0-9][a-z0-9-]*)_v(?P<version>[1-9][0-9]*)_"
    r"(?P<date>[0-9]{8})_(?P<source>[0-9a-f]{8})$"
)
_COMPONENT_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_STAGES = {"search", "final_eval", "posthoc"}


def validate_protocol_id(protocol_id: str) -> str:
    """Return a valid protocol ID or raise ``ValueError``.

    The fixed form is ``<experiment>_v<version>_<YYYYMMDD>_<source8>``.
    """

    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("protocol_id must be a non-empty string")
    if Path(protocol_id).is_absolute() or ".." in Path(protocol_id).parts:
        raise ValueError("protocol_id must be a single relative component")
    match = _PROTOCOL_RE.fullmatch(protocol_id)
    if match is None:
        raise ValueError(
            "protocol_id must match "
            "<experiment>_v<version>_<YYYYMMDD>_<source-id first8>"
        )
    try:
        datetime.strptime(match.group("date"), "%Y%m%d")
    except ValueError as exc:
        raise ValueError("protocol_id contains an invalid calendar date") from exc
    return protocol_id


def _component(value: str, role: str) -> str:
    normalized = str(value).lower()
    if not _COMPONENT_RE.fullmatch(normalized):
        raise ValueError(
            f"{role} must be one relative component containing only "
            "lowercase letters, numbers, '.', '_' and '-'"
        )
    return normalized


def _explicit_relative(value: str | os.PathLike[str], role: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{role} must be a repository-relative path without '..'")
    return path


def _select(
    repo_root: str | os.PathLike[str],
    default: Path,
    *,
    output: str | os.PathLike[str] | None,
    resume: bool,
) -> Path:
    root = Path(repo_root).resolve()
    selected = root / (default if output is None else _explicit_relative(output, "output"))
    selected = selected.resolve()
    if not selected.is_relative_to(root):
        raise ValueError("experiment output escapes the repository root")
    if selected.exists() and (not selected.is_dir() or any(selected.iterdir())):
        if not resume:
            raise FileExistsError(
                f"refusing to use non-empty experiment directory: {selected}"
            )
    return selected


def search_dir(
    repo_root: str | os.PathLike[str],
    protocol_id: str,
    dataset: str,
    method: str,
    search_seed: int,
    *,
    output: str | os.PathLike[str] | None = None,
    resume: bool = False,
) -> Path:
    protocol = validate_protocol_id(protocol_id)
    relative = (
        Path("results")
        / "search"
        / protocol
        / _component(dataset, "dataset")
        / _component(method, "method")
        / f"search_seed{int(search_seed)}"
    )
    return _select(repo_root, relative, output=output, resume=resume)


def final_eval_dir(
    repo_root: str | os.PathLike[str],
    protocol_id: str,
    dataset: str,
    method: str,
    *,
    output: str | os.PathLike[str] | None = None,
    resume: bool = False,
) -> Path:
    protocol = validate_protocol_id(protocol_id)
    relative = (
        Path("results")
        / "final_eval"
        / protocol
        / _component(dataset, "dataset")
        / _component(method, "method")
    )
    return _select(repo_root, relative, output=output, resume=resume)


def posthoc_dir(
    repo_root: str | os.PathLike[str],
    protocol_id: str,
    analysis_name: str,
    *,
    output: str | os.PathLike[str] | None = None,
    resume: bool = False,
) -> Path:
    protocol = validate_protocol_id(protocol_id)
    relative = (
        Path("results")
        / "posthoc"
        / protocol
        / _component(analysis_name, "analysis_name")
    )
    return _select(repo_root, relative, output=output, resume=resume)


def log_dir(
    repo_root: str | os.PathLike[str],
    protocol_id: str,
    stage: str,
    *,
    dataset: str | None = None,
    method: str | None = None,
    search_seed: int | None = None,
    output: str | os.PathLike[str] | None = None,
    resume: bool = False,
) -> Path:
    protocol = validate_protocol_id(protocol_id)
    normalized_stage = _component(stage, "stage")
    if normalized_stage not in _STAGES:
        raise ValueError(f"stage must be one of {sorted(_STAGES)}")
    relative = Path("logs") / protocol / normalized_stage
    if dataset is not None:
        relative /= _component(dataset, "dataset")
    if method is not None:
        if dataset is None:
            raise ValueError("method log paths require dataset")
        relative /= _component(method, "method")
    if search_seed is not None:
        if method is None:
            raise ValueError("search_seed log paths require method")
        relative /= f"search_seed{int(search_seed)}"
    return _select(repo_root, relative, output=output, resume=resume)
