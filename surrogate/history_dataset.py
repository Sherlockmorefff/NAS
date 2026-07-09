"""Load, validate, aggregate, and group-split Phase history for GP training."""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hp_modes import condition_mask_vector_from_ops, hp_dim_from_mode, validate_hp_mode


@dataclass
class HistoryDataset:
    X: np.ndarray
    y: np.ndarray
    condition_masks: np.ndarray
    architecture_keys: list[str]
    records: list[dict[str, Any]]
    filter_counts: dict[str, int]
    warnings: list[str]


def _records_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("history", "records", "trials", "results"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
        if "z_search" in value:
            return [value]
    raise ValueError("history JSON must be a record list or contain history/records/trials/results")


def _finite_accuracy(row: dict[str, Any]) -> float | None:
    for key in ("val_acc", "value", "accuracy", "best_val_acc", "best_value"):
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def architecture_key_from_record(row: dict[str, Any], z: np.ndarray) -> tuple[str, bool]:
    """Return the architecture-group key used for train/holdout separation.

    The primary key is the decoded discrete architecture: operations plus
    sorted edges.  Histories from older runs may not have that information; for
    those rows, fall back to a rounded raw vector key and report the fallback to
    the caller.
    """

    operations = row.get("operations")
    edges = row.get("edges")
    if isinstance(operations, (list, tuple)) and operations and isinstance(edges, (list, tuple)):
        normalized_edges = []
        edges_valid = True
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                edges_valid = False
                break
            try:
                normalized_edges.append((int(edge[0]), int(edge[1])))
            except (TypeError, ValueError):
                edges_valid = False
                break
        if edges_valid:
            key = (tuple(str(op) for op in operations), tuple(sorted(normalized_edges)))
            return repr(key), False
    rounded = tuple(float(value) for value in np.round(z, decimals=8))
    return f"z_search:{rounded!r}", True


def _stable_architecture_key(row: dict[str, Any], z: np.ndarray) -> tuple[str, bool]:
    return architecture_key_from_record(row, z)


def _condition_mask(row: dict[str, Any], hp_mode: str, arch_nz: int) -> np.ndarray:
    hp_dim = hp_dim_from_mode(hp_mode)
    search_dim = arch_nz + hp_dim
    value = row.get("condition_mask_vector")
    if value is None and isinstance(row.get("condition_mask"), dict):
        value = row["condition_mask"].get("vector", row["condition_mask"].get("mask"))
    if value is not None:
        try:
            mask = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            mask = np.empty(0, dtype=np.float64)
        if mask.size == hp_dim:
            mask = np.concatenate((np.ones(arch_nz), mask))
        if mask.size == search_dim and np.isfinite(mask).all():
            return np.clip(mask, 0.0, 1.0)
    operations = [str(op) for op in row.get("operations", [])]
    hp_mask = condition_mask_vector_from_ops(operations, hp_mode)
    return np.asarray([1.0] * arch_nz + [float(value) for value in hp_mask], dtype=np.float64)


def _semantic_value(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    params = row.get("params")
    return params.get(key) if isinstance(params, dict) else None


def load_history_dataset(
    paths: Iterable[str | Path],
    *,
    hp_mode: str,
    arch_nz: int,
    expected_metadata: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> HistoryDataset:
    """Load histories and aggregate duplicate architecture/HP evaluations."""

    hp_mode = validate_hp_mode(hp_mode)
    search_dim = int(arch_nz) + hp_dim_from_mode(hp_mode)
    counts: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    warning_messages: list[str] = []
    expected_metadata = dict(expected_metadata or {})
    for path_value in paths:
        path = Path(path_value)
        try:
            rows = _records_from_json(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise ValueError(f"failed to load history {path}: {exc}") from exc
        counts["input_records"] += len(rows)
        for row in rows:
            if row.get("valid") is not True:
                counts["invalid"] += 1
                continue
            accuracy = _finite_accuracy(row)
            if accuracy is None:
                counts["missing_or_nonfinite_val_acc"] += 1
                continue
            if row.get("hp_mode") != hp_mode:
                counts["hp_mode_mismatch"] += 1
                continue
            try:
                z = np.asarray(row.get("z_search"), dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                counts["missing_or_invalid_z_search"] += 1
                continue
            if z.size != search_dim:
                counts["search_dim_mismatch"] += 1
                continue
            if not np.isfinite(z).all():
                counts["nonfinite_z_search"] += 1
                continue
            mismatch = False
            for key, wanted in expected_metadata.items():
                if wanted is None:
                    continue
                actual = _semantic_value(row, key)
                if actual is not None and actual != wanted:
                    counts[f"{key}_mismatch"] += 1
                    mismatch = True
                    break
            if mismatch:
                continue
            architecture_key, fallback = _stable_architecture_key(row, z)
            if fallback:
                counts["architecture_key_z_fallback"] += 1
            accepted.append(
                {
                    "row": row,
                    "z": z,
                    "accuracy": accuracy,
                    "architecture_key": architecture_key,
                    "mask": _condition_mask(row, hp_mode, int(arch_nz)),
                }
            )
    if counts["architecture_key_z_fallback"]:
        message = (
            f"{counts['architecture_key_z_fallback']} accepted records lacked usable operations/edges; "
            "stable rounded z_search keys were used"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        warning_messages.append(message)

    grouped: dict[tuple[str, tuple[float, ...]], list[dict[str, Any]]] = defaultdict(list)
    for item in accepted:
        hp_key = tuple(float(value) for value in np.round(item["z"][int(arch_nz) :], 8))
        grouped[(item["architecture_key"], hp_key)].append(item)
    aggregates: list[dict[str, Any]] = []
    for items in grouped.values():
        values = np.asarray([item["accuracy"] for item in items], dtype=np.float64)
        representative = dict(items[0]["row"])
        representative["val_acc"] = float(values.mean())
        representative["repeat_count"] = int(values.size)
        representative["repeat_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        aggregates.append(
            {
                "z": items[0]["z"],
                "accuracy": float(values.mean()),
                "mask": items[0]["mask"],
                "architecture_key": items[0]["architecture_key"],
                "record": representative,
            }
        )
    counts["accepted_records"] = len(accepted)
    counts["aggregated_samples"] = len(aggregates)
    counts["duplicates_aggregated"] = len(accepted) - len(aggregates)
    if logger is not None:
        logger.info("History filtering summary: %s", dict(sorted(counts.items())))
        for message in warning_messages:
            logger.warning(message)
    if not aggregates:
        raise ValueError(f"no valid history samples remain after filtering: {dict(counts)}")
    return HistoryDataset(
        X=np.stack([item["z"] for item in aggregates]),
        y=np.asarray([item["accuracy"] for item in aggregates], dtype=np.float64),
        condition_masks=np.stack([item["mask"] for item in aggregates]),
        architecture_keys=[item["architecture_key"] for item in aggregates],
        records=[item["record"] for item in aggregates],
        filter_counts=dict(counts),
        warnings=warning_messages,
    )


def grouped_train_holdout_split(
    dataset: HistoryDataset,
    *,
    holdout_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by discrete architecture key to prevent equivalent-vector leakage."""

    if not 0.0 < float(holdout_frac) < 1.0:
        raise ValueError("holdout_frac must be strictly between 0 and 1")
    groups = sorted(set(dataset.architecture_keys))
    if len(groups) < 2 or dataset.y.size < 4:
        raise ValueError("at least four samples across two architecture groups are required")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    holdout_group_count = min(len(groups) - 1, max(1, int(round(len(groups) * float(holdout_frac)))))
    holdout_groups = set(groups[:holdout_group_count])
    holdout = np.asarray([i for i, key in enumerate(dataset.architecture_keys) if key in holdout_groups], dtype=np.int64)
    train = np.asarray([i for i, key in enumerate(dataset.architecture_keys) if key not in holdout_groups], dtype=np.int64)
    if train.size < 2 or holdout.size < 1:
        raise ValueError(f"grouped split produced train={train.size}, holdout={holdout.size}")
    return train, holdout
