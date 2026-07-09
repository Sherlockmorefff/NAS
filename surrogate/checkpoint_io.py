"""Atomic checkpoint and JSON helpers used by the GP predictor."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def torch_load_compat(path: str | os.PathLike[str], map_location: Any = "cpu") -> Any:
    """Load checkpoints containing tensors and metadata across torch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> None:
    """Atomically replace *path* with a complete torch checkpoint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json_dump(payload: Any, path: str | os.PathLike[str]) -> None:
    """Write JSON using an atomic same-directory replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
