"""Build a read-only inventory of previously executed GNN training seeds.

The scanner is deliberately conservative:

* only seed fields attached to an executed GNN result count as collisions;
* planning, search, bootstrap, and proxy-model seeds are classified but do not
  count as executed GNN seeds;
* a numeric seed-like field with unknown semantics aborts the inventory;
* JSON/JSONL values under a key containing ``test`` are lexically skipped before
  deserialization; CSV test columns are removed before their cells are decoded.

No function in this module creates a directory or writes a file.  Callers must
finish this inventory before creating the checkpoint-freeze result directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "historical-gnn-training-seed-inventory-v1"
DEFAULT_SCAN_ROOTS = (
    "results", "logs", "output_results", "artifacts", "packages",
    "legacy", "legacy_artifacts", "raw",
)
EXCLUDED_TARGET = "results/final/frozen-validation-checkpoints-global4-v1"
FRESH_SEEDS = tuple(range(100000, 100030))
STRUCTURED_SUFFIXES = (".json", ".jsonl", ".ndjson", ".csv")
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz")
SQLITE_SUFFIXES = (".sqlite", ".sqlite3", ".db")
SQLITE_SIDECAR_SUFFIXES = tuple(
    sqlite_suffix + sidecar_suffix
    for sqlite_suffix in SQLITE_SUFFIXES
    for sidecar_suffix in ("-wal", "-shm", "-journal")
)
UNSUPPORTED_STRUCTURED_SUFFIXES = SQLITE_SUFFIXES + (".parquet", ".zip")

# SQLite is fail-closed by default.  This one immutable legacy Optuna database
# was manually audited after the first full scan stopped on it.  The audit
# inspected every table/column identifier, every distinct Optuna attribute and
# parameter name, and the key structure of all 2,102 JSON cells using the same
# lexical test-value firewall as this module.  No seed-bearing identifier or
# nested seed key was present.  Only these exact bytes may be treated as a
# non-seed source; any changed or additional SQLite file still aborts.
AUDITED_NON_SEED_SQLITE: Mapping[str, Mapping[str, Any]] = {
    "legacy_artifacts/pre_20260813/results-1-LHS/"
    "bo_phase4_condhp_v6_layerwise_gin_tpe/optuna_tpe.db": {
        "sha256": "318d02512f0ee38a57680eb57a799f1709bdcffdb4ec17e5abebc7711b7527fb",
        "size": 454656,
        "format": "optuna_sqlite",
        "table_count": 14,
        "distinct_outer_identifier_count": 39,
        "json_cells_key_scanned": 2102,
        "seed_identifier_hits": 0,
        "nested_seed_key_hits": 0,
        "skipped_test_value_count": 0,
        "audit_policy": (
            "exact-byte manual schema audit; no SQLite row value is treated "
            "as an executed GNN seed"
        ),
    },
}
MAX_STRUCTURED_BYTES = 512 * 1024 * 1024

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_SEED_ID_KEY_RE = re.compile(
    r"(?:^seed$|^seeds$|(?:^|_)seed(?:_id)?s?$|(?:^|_)training_seed(?:_id)?s?$|"
    r"(?:^|_)gnn_seed(?:_id)?s?$|(?:^|_)candidate_eval_seed(?:_id)?s?$)"
)
_NON_ID_SEED_KEY_PARTS = (
    "seed_count", "seed_shape", "seed_dtype", "seed_error", "seed_value",
    "seed_result", "seed_index", "seed_indices", "seed_sample", "seed_sha",
    "seed_hash", "seed_inventory", "seed_collision", "seed_fresh",
    "best_epoch_by_seed", "stop_epoch_by_seed", "epochs_by_seed",
)
_NON_ID_SEED_KEYS = {"n_seed", "n_seeds", "num_seed", "num_seeds"}
_EXPLICIT_NON_TRAINING_FIELD_PARTS = (
    "bootstrap", "master", "random", "rng", "draw", "permutation", "shuffle",
    "decoder", "selection", "sampler", "search", "source", "split", "fold",
    "pool", "shard", "gpu", "tie_break", "candidate_generation", "prng",
    "restart", "analysis",
)
_NON_TRAINING_CONTEXT_PARTS = (
    "bootstrap", "preregistration", "statistics", "randomization", "candidate_generation",
    "draw_audit", "permutation", "sampling", "resampling", "search_identity",
)
_GNN_ARCH_KEYS = {
    "architecture_fingerprint", "architecture", "operations", "ops", "edges",
    "joint_config_fingerprint", "joint_fingerprint", "config_fingerprint",
    "hp_fingerprint", "config_id", "raw_hp", "raw_hp_vector", "decoded_hp",
    "realized_hp", "global4_hp", "hyperparameters",
}
_EXECUTION_KEYS = {
    "val_acc", "validation_accuracy", "best_validation", "best_val_acc",
    "mean_validation", "seed_validation_values", "validation_values", "accuracy",
    "best_epoch", "epochs_ran", "stopped_epoch", "stop_epoch", "trajectory",
    "checkpoint_sha256", "checkpoint_path", "tensor_state_hash", "valid",
    "failure_reason", "retry_count", "elapsed_seconds", "wall_seconds", "status", "success",
}
_TASK_KEYS = {
    "task_key", "task_id", "assigned_gpu", "config_id", "joint_config_fingerprint",
    "architecture_fingerprint", "hp_fingerprint", "operations", "edges",
}
_PROXY_KEYS = {
    "model", "model_name", "surrogate", "surrogate_type", "split_id", "weight_mode", "fold_id",
    "head_member", "lambda", "predictor", "estimator",
}

# Exact-byte, source-specific semantic audits for two legacy pipeline argument
# records.  General ``seed`` fields remain fail-closed; these exceptions are
# accepted only after their complete file identity is checked in the scan loop.
AUDITED_STRUCTURED_SEED_SEMANTICS: Mapping[str, Mapping[str, Any]] = {
    "legacy_artifacts/pre_20260813/logs-LHS/bo_phase4_tpe/"
    "train_condhp_v6_layerwise_gin_tpe_0609_0356.json": {
        "sha256": "94f3fb04f86fbdffd0ba0ff7fb01d84ba33bb6cee25b4ac7f3256aa468ddbcbb",
        "size": 968,
        "field_path": "$/seed",
        "expected_seeds": [42],
        "semantic": "executed_gnn_run_rng_initialization_seed",
        "executed_gnn": True,
        "scope": "one 80-trial continuous RNG stream; not 80 independent seed-42 repeats",
        "dual_purpose": "Torch/NumPy run initialization and Optuna TPE sampler seed",
    },
    "legacy_artifacts/pre_20260813/logs-LHS/final_eval/"
    "train_condhp_v6_layerwise_gin_tpe_final_0609_0357.json": {
        "sha256": "9fa0255519373e7da5f3620fcbff315531e28fa73786b888772f7848f3bc661d",
        "size": 706,
        "field_path": "$/seed",
        "expected_seeds": [42],
        "semantic": "non_training_final_eval_orchestration_seed",
        "executed_gnn": False,
        "scope": "global orchestration initialization; per-GNN repeats use range(n_seeds)",
    },
}

# This result does not store explicit seed IDs, but its exact bytes contain
# seven validation lists of length ten and the exact historical implementation
# (commit 45ff0a68, final_eval.py SHA below) passes range(n_seeds) directly to
# every GNN training call.  The contract therefore unambiguously establishes
# executed training seeds 0..9 without reading any test value.
AUDITED_IMPLICIT_GNN_SEED_SOURCES: Mapping[str, Mapping[str, Any]] = {
    "legacy_artifacts/pre_20260813/results-1-LHS/"
    "final_eval_condhp_v6_layerwise_gin_tpe/final_results_v9.json": {
        "sha256": "c379e71636f0d86dca95ebef52c9b82025bd2628577b25def2e0a3c510e87c26",
        "size": 13159,
        "field_path": "$/manual_contract/val_list_index_is_training_seed",
        "seeds": list(range(10)),
        "semantic": "executed_gnn_training_seed_from_frozen_code_contract",
        "executed_gnn": True,
        "candidate_count": 7,
        "validation_list_length": 10,
        "historical_git_commit": "45ff0a68",
        "historical_final_eval_py_sha256": (
            "db42fa75ee524da7b1f8becb00f10db70b8bad37f5a1fc34e0b7fdfee1f58a89"
        ),
    },
}


class InventoryError(RuntimeError):
    """Base class for inventory failures."""


class AmbiguousSeedSemanticsError(InventoryError):
    """Raised when at least one numeric seed-like field cannot be classified."""

    def __init__(self, findings: Sequence[Mapping[str, Any]]):
        self.findings = [dict(item) for item in findings]
        summary = "; ".join(
            f"{item['source']}:{item['field_path']}" for item in self.findings[:8]
        )
        if len(self.findings) > 8:
            summary += f"; ... ({len(self.findings)} total)"
        super().__init__(f"ambiguous numeric seed semantics: {summary}")


class FreshSeedCollisionError(InventoryError):
    """Raised when a fixed fresh seed has an executed historical GNN collision."""

    def __init__(self, collisions: Sequence[int]):
        self.collisions = tuple(sorted({int(value) for value in collisions}))
        super().__init__(f"fixed fresh GNN seeds collide with history: {list(self.collisions)}")


class UnsafeStructuredInputError(InventoryError):
    """Raised for malformed, oversized, or otherwise unsafe structured input."""


@dataclass(frozen=True)
class _SkippedTestField:
    path_hash: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_record(path: Path, relative: str, kind: str) -> dict[str, Any]:
    before = path.stat()
    digest = _sha256_path(path)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise UnsafeStructuredInputError(f"source changed while hashing: {relative}")
    return {"source": relative, "kind": kind, "size": after.st_size, "sha256": digest}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _normalize_relpath(value: str | PurePosixPath) -> str:
    raw = str(value).replace("\\", "/")
    parts = []
    for part in PurePosixPath(raw).parts:
        if part in ("", ".", "/"):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _is_excluded(relative: str, excluded_target: str) -> bool:
    relative = _normalize_relpath(relative)
    excluded = _normalize_relpath(excluded_target)
    return relative == excluded or relative.startswith(excluded + "/")


def _is_excluded_archive_member(member: str, excluded_target: str) -> bool:
    normalized = _normalize_relpath(member)
    excluded = _normalize_relpath(excluded_target)
    target_name = PurePosixPath(excluded).name
    return (
        _is_excluded(normalized, excluded)
        or normalized == target_name
        or normalized.startswith(target_name + "/")
        or f"/{target_name}/" in f"/{normalized}/"
    )


def _is_test_key(key: str) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9]+", str(key).casefold()) if token]
    return "test" in tokens


class _SafeJsonParser:
    """Minimal JSON parser that skips test-key values without decoding them."""

    def __init__(self, text: str, source: str, root_path: str = "$"):
        self.text = text
        self.source = source
        self.root_path = root_path
        self.position = 0
        self.length = len(text)
        self.skipped: list[_SkippedTestField] = []
        self._decoder = json.JSONDecoder()

    def parse(self) -> Any:
        value = self._parse_value(self.root_path)
        self._skip_ws()
        if self.position != self.length:
            raise UnsafeStructuredInputError(
                f"trailing JSON data in {self.source} at character {self.position}"
            )
        return value

    def _skip_ws(self) -> None:
        while self.position < self.length and self.text[self.position] in " \t\r\n":
            self.position += 1

    def _expect(self, character: str) -> None:
        self._skip_ws()
        if self.position >= self.length or self.text[self.position] != character:
            raise UnsafeStructuredInputError(
                f"expected {character!r} in {self.source} at character {self.position}"
            )
        self.position += 1

    def _parse_string(self) -> str:
        self._skip_ws()
        try:
            value, end = self._decoder.raw_decode(self.text, self.position)
        except json.JSONDecodeError as exc:
            raise UnsafeStructuredInputError(f"invalid JSON string in {self.source}: {exc}") from exc
        if not isinstance(value, str):
            raise UnsafeStructuredInputError(
                f"expected JSON string in {self.source} at character {self.position}"
            )
        self.position = end
        return value

    @staticmethod
    def _child_path(parent: str, key: str) -> str:
        escaped = key.replace("~", "~0").replace("/", "~1")
        return f"{parent}/{escaped}"

    def _parse_value(self, path: str) -> Any:
        self._skip_ws()
        if self.position >= self.length:
            raise UnsafeStructuredInputError(f"unexpected end of JSON in {self.source}")
        character = self.text[self.position]
        if character == "{":
            return self._parse_object(path)
        if character == "[":
            return self._parse_array(path)
        try:
            value, end = self._decoder.raw_decode(self.text, self.position)
        except json.JSONDecodeError as exc:
            raise UnsafeStructuredInputError(f"invalid JSON in {self.source}: {exc}") from exc
        self.position = end
        return value

    def _parse_object(self, path: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self._expect("{")
        self._skip_ws()
        if self.position < self.length and self.text[self.position] == "}":
            self.position += 1
            return result
        while True:
            key = self._parse_string()
            self._expect(":")
            child_path = self._child_path(path, key)
            if _is_test_key(key):
                self._skip_json_value_lexically()
                self.skipped.append(
                    _SkippedTestField(_sha256_bytes(f"{self.source}\0{child_path}".encode("utf-8")))
                )
            else:
                result[key] = self._parse_value(child_path)
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == ",":
                self.position += 1
                continue
            if self.position < self.length and self.text[self.position] == "}":
                self.position += 1
                return result
            raise UnsafeStructuredInputError(
                f"invalid JSON object in {self.source} at character {self.position}"
            )

    def _parse_array(self, path: str) -> list[Any]:
        result: list[Any] = []
        self._expect("[")
        self._skip_ws()
        if self.position < self.length and self.text[self.position] == "]":
            self.position += 1
            return result
        index = 0
        while True:
            result.append(self._parse_value(f"{path}/{index}"))
            index += 1
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == ",":
                self.position += 1
                continue
            if self.position < self.length and self.text[self.position] == "]":
                self.position += 1
                return result
            raise UnsafeStructuredInputError(
                f"invalid JSON array in {self.source} at character {self.position}"
            )

    def _skip_json_value_lexically(self) -> None:
        """Advance over one JSON value without converting any part of it."""
        self._skip_ws()
        if self.position >= self.length:
            raise UnsafeStructuredInputError(f"missing skipped JSON value in {self.source}")
        start = self.position
        first = self.text[self.position]
        if first == '"':
            self.position = self._scan_string_end(self.position)
            return
        if first in "[{":
            stack = [first]
            self.position += 1
            while self.position < self.length and stack:
                character = self.text[self.position]
                if character == '"':
                    self.position = self._scan_string_end(self.position)
                    continue
                if character in "[{":
                    stack.append(character)
                elif character in "]}":
                    expected = "[" if character == "]" else "{"
                    if not stack or stack[-1] != expected:
                        raise UnsafeStructuredInputError(
                            f"malformed skipped JSON value in {self.source} at {self.position}"
                        )
                    stack.pop()
                self.position += 1
            if stack:
                raise UnsafeStructuredInputError(f"unterminated skipped JSON value in {self.source}")
            return
        while self.position < self.length and self.text[self.position] not in ",}]\r\n\t ":
            self.position += 1
        if self.position == start:
            raise UnsafeStructuredInputError(f"empty skipped JSON value in {self.source}")

    def _scan_string_end(self, start: int) -> int:
        position = start + 1
        while position < self.length:
            character = self.text[position]
            if character == "\\":
                position += 2
                continue
            if character == '"':
                return position + 1
            position += 1
        raise UnsafeStructuredInputError(f"unterminated JSON string in {self.source}")


def safe_load_json(text: str, source: str, root_path: str = "$") -> tuple[Any, list[str]]:
    parser = _SafeJsonParser(text, source, root_path=root_path)
    value = parser.parse()
    return value, [item.path_hash for item in parser.skipped]


def _raw_csv_records(text: str, source: str) -> Iterator[str]:
    start = 0
    position = 0
    quoted = False
    while position < len(text):
        character = text[position]
        if character == '"':
            if quoted and position + 1 < len(text) and text[position + 1] == '"':
                position += 2
                continue
            quoted = not quoted
        if character in "\r\n" and not quoted:
            yield text[start:position]
            if character == "\r" and position + 1 < len(text) and text[position + 1] == "\n":
                position += 1
            start = position + 1
        position += 1
    if quoted:
        raise UnsafeStructuredInputError(f"unterminated quoted CSV field in {source}")
    if start < len(text):
        yield text[start:]


def _raw_csv_fields(record: str, source: str) -> list[str]:
    fields: list[str] = []
    start = 0
    position = 0
    quoted = False
    while position < len(record):
        character = record[position]
        if character == '"':
            if quoted and position + 1 < len(record) and record[position + 1] == '"':
                position += 2
                continue
            quoted = not quoted
        elif character == "," and not quoted:
            fields.append(record[start:position])
            start = position + 1
        position += 1
    if quoted:
        raise UnsafeStructuredInputError(f"unterminated quoted CSV field in {source}")
    fields.append(record[start:])
    return fields


def _decode_csv_field(raw: str, source: str) -> str:
    if raw == "":
        return ""
    try:
        row = next(csv.reader([raw], strict=True))
    except (csv.Error, StopIteration) as exc:
        raise UnsafeStructuredInputError(f"invalid CSV field in {source}") from exc
    if len(row) != 1:
        raise UnsafeStructuredInputError(f"invalid isolated CSV field in {source}")
    return row[0]


def safe_load_csv(text: str, source: str) -> tuple[list[dict[str, str]], list[str]]:
    records = list(_raw_csv_records(text, source))
    while records and not records[-1].strip():
        records.pop()
    if not records:
        return [], []
    header_raw = _raw_csv_fields(records[0], source)
    headers = [_decode_csv_field(raw, source).lstrip("\ufeff") for raw in header_raw]
    if len(headers) != len(set(headers)):
        raise UnsafeStructuredInputError(f"duplicate CSV headers in {source}")
    test_columns = {index for index, name in enumerate(headers) if _is_test_key(name)}
    rows: list[dict[str, str]] = []
    skipped_hashes: list[str] = []
    for row_number, raw_record in enumerate(records[1:], start=2):
        if not raw_record.strip():
            continue
        raw_fields = _raw_csv_fields(raw_record, source)
        if len(raw_fields) != len(headers):
            raise UnsafeStructuredInputError(
                f"CSV column count mismatch in {source} row {row_number}"
            )
        row: dict[str, str] = {}
        for index, (header, raw_value) in enumerate(zip(headers, raw_fields)):
            if index in test_columns:
                skipped_hashes.append(
                    _sha256_bytes(f"{source}\0$row/{row_number}/{header}".encode("utf-8"))
                )
                continue
            row[header] = _decode_csv_field(raw_value, source)
        rows.append(row)
    return rows, skipped_hashes


def _coerce_seed_values(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, float):
        return (int(value),) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if _INTEGER_RE.fullmatch(stripped):
            return (int(stripped),)
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            return _coerce_seed_values(parsed)
        return None
    if isinstance(value, (list, tuple)):
        flattened: list[int] = []
        for item in value:
            converted = _coerce_seed_values(item)
            if converted is None:
                return None
            flattened.extend(converted)
        return tuple(flattened)
    return None


def _nested_integer_values(value: Any) -> tuple[int, ...]:
    """Collect integer-looking leaves solely to reject unsupported seed schemas."""
    direct = _coerce_seed_values(value)
    if direct is not None:
        return direct
    found: list[int] = []
    if isinstance(value, Mapping):
        for child in value.values():
            found.extend(_nested_integer_values(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_nested_integer_values(child))
    return tuple(found)


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _source_basename(source: str) -> str:
    """Return a stable member/file basename, excluding tar duplicate suffixes."""
    name = PurePosixPath(source.split("!", 1)[-1]).name.casefold()
    return re.sub(r"#\d+$", "", name)


def _is_seed_id_key(key: str) -> bool:
    normalized = _normalized_key(key)
    if normalized in _NON_ID_SEED_KEYS or re.fullmatch(
        r"(?:n|num)(?:_[a-z0-9]+)*_seeds?", normalized,
    ):
        return False
    if any(part in normalized for part in _NON_ID_SEED_KEY_PARTS):
        return False
    return bool(_SEED_ID_KEY_RE.search(normalized))


def _flatten_key_context(record: Mapping[str, Any]) -> set[str]:
    keys = {_normalized_key(str(key)) for key in record}
    for container_key in ("config", "architecture", "metadata", "task", "result", "protocol"):
        nested = record.get(container_key)
        if isinstance(nested, Mapping):
            keys.update(_normalized_key(str(key)) for key in nested)
    return keys


def _looks_like_gnn_identity(keys: set[str], source: str) -> bool:
    if keys & _GNN_ARCH_KEYS:
        return True
    source_name = _source_basename(source)
    return "seed_results" in source_name and bool(keys & {"task_key", "config_id", "pool_id"})


def _looks_executed(keys: set[str], source: str) -> bool:
    if keys & _EXECUTION_KEYS:
        return True
    source_name = _source_basename(source)
    return "seed_results" in source_name or "label_results" in source_name


def _classify_seed_field(
    *, key: str, field_path: str, record: Mapping[str, Any], source: str,
) -> tuple[str, bool] | None:
    normalized = _normalized_key(key)
    if not _is_seed_id_key(normalized):
        return None
    audited = AUDITED_STRUCTURED_SEED_SEMANTICS.get(source)
    if audited is not None and field_path == audited["field_path"]:
        return (str(audited["semantic"]), bool(audited["executed_gnn"]))
    keys = _flatten_key_context(record)
    path_context = _normalized_key(field_path)
    source_context = source.casefold().replace("\\", "/")

    if any(part in normalized for part in _EXPLICIT_NON_TRAINING_FIELD_PARTS):
        return ("non_training_random_or_statistical_seed", False)

    if any(part in path_context for part in _EXPLICIT_NON_TRAINING_FIELD_PARTS):
        return ("non_training_random_or_statistical_seed", False)

    source_name = _source_basename(source)

    # A protocol limit whose name ends in ``seed`` is a retry count, not a
    # seed identifier.  Require the surrounding frozen GNN protocol schema so
    # a similarly named field elsewhere remains fail-closed.
    if normalized == "maximum_system_retry_per_config_seed" and {
        "dataset", "max_epochs", "patience", "training_seeds",
        "maximum_system_retry_per_config_seed",
    }.issubset(keys):
        return ("non_seed_retry_limit_count", False)

    # Candidate construction and clustering use RNG seeds, but do not start a
    # GNN training repeat.  These rules intentionally require the complete
    # audit-record signature rather than trusting a bare ``seed`` or filename.
    if normalized in {"generation_seed", "hp_seed"} and {
        "generator", "generation_draws", "hp_regimes", "invalid_counts",
        "generation_seed", "hp_seed",
    }.issubset(keys):
        return ("non_training_candidate_generation_seed", False)

    if normalized == "seed" and source_name == "candidate_generation_audit.json" and {
        "pool_id", "draw_schedule", "draws_used", "fresh_unique_rate",
        "selected_architecture_count", "seed",
    }.issubset(keys):
        return ("non_training_candidate_generation_seed", False)

    rng_provenance_schema = {
        "seed_derivation", "search_seed", "lhs_seed", "wgmm_fit_seed",
        "full_fidelity_seed_context", "low_fidelity_seed_context", "python_hash_used",
    }
    if source_name == "rng_provenance.json" and rng_provenance_schema.issubset(keys):
        if normalized == "lhs_seed":
            return ("non_training_candidate_generation_seed", False)
        if normalized == "wgmm_fit_seed":
            return ("non_gnn_proxy_or_clustering_seed", False)

    if normalized == "seed" and source_name == "partition_audit.json" and {
        "pool_id", "cluster_sizes", "distance_definition", "medoid_config_ids",
        "silhouette", "seed",
    }.issubset(keys):
        return ("non_training_partition_seed", False)

    if normalized == "seed" and source_name == "random_rejection_manifest.json" and {
        "draws", "permutations_path", "permutations_sha256", "shape", "seed",
    }.issubset(keys):
        return ("non_training_random_or_statistical_seed", False)

    if normalized == "fixed_seed" and source_name == "split_reproduction_audit.json" and {
        "expected_split_count", "grouped_repeated_parameters", "observed_split_count",
        "validation_membership_mismatches", "fixed_seed",
    }.issubset(keys):
        return ("non_gnn_proxy_split_seed", False)

    # A plan can mention seeds after some tasks have completed, but it is still
    # not execution evidence.  Actual completion is established independently
    # by the seed-level result records.  Each accepted plan shape has a narrow
    # schema guard to prevent a result file named similarly from being ignored.
    run_plan_shape = (
        {"shard", "seeds"}.issubset(keys)
        and (
            {"expected_task_count", "pending_task_count"}.issubset(keys)
            or {"expected_training_count", "pending_training_count"}.issubset(keys)
        )
    )
    if normalized == "seeds" and (
        (source_name.startswith("run_plan") and source_name.endswith(".json") and run_plan_shape)
        or (
            re.fullmatch(r"gpu\d+\.stdout\.json", source_name) is not None
            and run_plan_shape
            and "dry_run" in keys
        )
        or (
            source_name.endswith("_initial_shard_plan.json")
            and {"architecture_fingerprints", "config_ids", "expected_training_count", "gpu"}
            .issubset(keys)
        )
    ):
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if normalized == "seeds" and source_name == "preregistration.json" and (
        path_context.endswith("training_seeds")
        and {"dataset", "max_epochs", "patience", "split", "seeds"}.issubset(keys)
    ):
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if normalized == "seeds" and source_name == "adaptive_selection.json" and {
        "config_ids", "configuration_count", "prediction_correctness_read", "rule", "seeds",
    }.issubset(keys):
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if normalized == "adaptive_seeds" and (
        (
            source_name in {"label_task_plan.json", "targeted_task_plan.json"}
            and (
                {"initial_seeds", "maximum_retries", "max_epochs", "patience"}.issubset(keys)
                or {"code_hashes", "merge_key", "selection_sha256"}.issubset(keys)
            )
        )
        or {"clean24_initial_seeds", "core_dataset_split_fidelity_and_early_stopping_unchanged"}
        .issubset(keys)
    ):
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if normalized == "clean24_adaptive_seeds" and {
        "clean24_initial_seeds", "core_dataset_split_fidelity_and_early_stopping_unchanged",
        "clean24_adaptive_seeds",
    }.issubset(keys):
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if normalized == "additional_seed_ids" and source_name == "stabilization_firewall_audit.json" and {
        "additional_task_count", "maximum_additional_tasks", "top8_selection_read_seed_ids",
        "union_configuration_count", "additional_seed_ids",
    }.issubset(keys):
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    # These records are executions, but they train a performance-aware VAE or
    # its prediction head rather than a Cora GNN candidate.  Guard on both the
    # experiment family and the task/result schema before excluding them.
    if normalized in {"training_seed", "training_seeds"} and (
        "performance-aware-vae" in source_context
        and (
            {"task_id", "variant", "outer_fold", "noise_mode", "performance_strength"}
            .issubset(keys)
            or {
                "variant", "outer_fold", "inner_metrics", "outer_predictions",
                "original_checkpoint_sha256",
            }.issubset(keys)
            or (
                source_name == "reconstruction_audit.json"
                and {"checkpoint", "checkpoint_sha256", "metrics", "outer_fold"}.issubset(keys)
            )
        )
    ):
        return ("non_gnn_representation_or_proxy_training_seed", False)
    if normalized == "cell_seed" and "random_ranking" in source_context:
        return ("non_training_random_or_statistical_seed", False)

    if normalized == "seed" and source_name == "representation_comparison.json" and (
        "metadata" in path_context
    ):
        return ("non_training_analysis_seed", False)

    if normalized == "seed" and source_name == "stage3_task_plan.json" and (
        "tasks" in path_context
    ):
        return ("planned_gnn_training_seed_not_executed", False)

    if normalized == "seed" and source_name in {
        "matched_relabel_summary.json", "matched_relabel_paired_differences.csv",
        "paired_policy_differences.csv",
    }:
        return ("derived_gnn_seed_reference_not_execution_evidence", False)

    if normalized in {"seeds", "existing_seeds", "new_seeds", "expected_seeds"} and (
        source_name.endswith("_run_plan.json")
        or source_name.endswith("_task_plan.json")
        or source_name.endswith("_merge_audit.json")
        or source_name in {"diagnostic_goal_progress.json", "representation_comparison.json"}
    ):
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if normalized == "final_base_seed":
        return ("declared_gnn_seed_base_not_execution_evidence", False)

    if normalized == "final_evaluation_seeds":
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if normalized in {"confirmation_seeds", "candidate_training_seeds"}:
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if normalized == "seeds" and "confirmation" in source_context:
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if normalized == "validation_positive_seeds":
        return ("non_training_validation_subset_index", False)

    if normalized == "final_eval_seed" and "replicates" in path_context and (
        "selected_history_records" in source_context
    ):
        return ("executed_gnn_final_evaluation_seed", True)

    if normalized == "training_seed" and "per_seed" in path_context and "valid" in keys and any(
        item == "validation" or item.endswith("_validation")
        or item == "validation_delta" or item.endswith("_best_epoch")
        for item in keys
    ):
        return ("executed_gnn_training_seed", True)

    if normalized == "seed" and (
        keys & {"cluster_plan_fingerprint", "promotion", "shortlist", "expand", "quotas", "selection_indices"}
        or "cluster_quota" in source_context
        or "gmm_cluster_quotas" in source_context
        or "/quotas.json" in source_context
        or "/selection_indices.json" in source_context
        or "/quota_diagnostics.json" in source_context
    ):
        return ("non_training_cluster_or_selection_bucket", False)

    if normalized == "seed" and "/precheck/" in source_context and not (
        _looks_like_gnn_identity(keys, source) and _looks_executed(keys, source)
    ):
        return ("non_training_precheck_or_determinism_seed", False)

    if normalized == "seed" and any(
        stage in source_context
        for stage in ("/bo_phase2/", "/bo_phase3/", "/bo_phase4/", "/bo_phase4_tpe/")
    ) and {"cora_root", "output", "eval_epochs"}.issubset(keys) and (
        "patience" in keys or {"n_init", "n_iter", "n_eval_seeds"}.issubset(keys)
    ):
        return ("executed_gnn_run_rng_initialization_seed", True)

    if normalized == "seed" and "/final_eval/" in source_context and (
        "n_seeds" in keys or "n_replicates" in keys
    ) and {"output", "eval_epochs", "patience"}.issubset(keys):
        return ("non_training_final_eval_orchestration_seed", False)

    if normalized == "seed" and "/train_joint/" in source_context and (
        {"epochs", "data", "checkpoint_dir"}.issubset(keys)
        or {"epochs", "data", "checkpoint"}.issubset(keys)
    ):
        return ("non_gnn_representation_training_seed", False)

    if normalized in {"candidate_eval_seed", "candidate_evaluation_seed"} and (
        keys & {"val_acc", "validation_accuracy", "best_validation"}
    ) and (keys & {"best_epoch", "epochs_ran", "stopped_epoch"}) and (
        keys & {"valid", "status", "record_type"}
    ):
        return ("executed_gnn_candidate_evaluation_seed", True)

    if normalized in {"training_seed", "training_seeds"} and keys & _PROXY_KEYS and not _looks_like_gnn_identity(keys, source):
        return ("non_gnn_proxy_model_training_seed", False)

    if _looks_like_gnn_identity(keys, source) and _looks_executed(keys, source):
        return ("executed_gnn_training_seed", True)

    if _looks_like_gnn_identity(keys, source) and keys & _TASK_KEYS:
        return ("planned_gnn_training_seed_not_executed", False)

    if keys & _PROXY_KEYS and not _looks_like_gnn_identity(keys, source):
        return ("non_gnn_proxy_model_training_seed", False)

    if keys & {"replicates", "n_bootstrap", "prng", "outer_unit", "confidence_interval"}:
        return ("non_training_statistical_seed", False)

    if normalized in {"seeds", "seed_ids", "training_seeds", "gnn_seeds"} and (
        "protocol" in path_context or "preregistration" in path_context or "task" in path_context
    ):
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if normalized.endswith("_seeds") and any(
        token in normalized
        for token in ("initial", "stabilization", "additional", "fixed", "common", "fresh", "gnn", "training")
    ):
        return ("declared_gnn_seed_list_not_execution_evidence", False)

    if any(part in path_context for part in _NON_TRAINING_CONTEXT_PARTS):
        return ("non_training_context_seed", False)

    return ("ambiguous", False)


def _walk_seed_fields(
    value: Any,
    *,
    source: str,
    path: str = "$",
) -> Iterator[dict[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_string = str(key)
            escaped = key_string.replace("~", "~0").replace("/", "~1")
            child_path = f"{path}/{escaped}"
            seed_values = _coerce_seed_values(child)
            classification = _classify_seed_field(
                key=key_string, field_path=child_path, record=value, source=source,
            )
            if classification is not None and seed_values is not None:
                semantic, executed = classification
                recorded_seeds = (
                    [] if semantic == "non_training_cluster_or_selection_bucket"
                    else sorted(set(seed_values))
                )
                yield {
                    "source": source,
                    "field_path": child_path,
                    "semantic": semantic,
                    "executed_gnn": executed,
                    "seeds": recorded_seeds,
                    **({
                        "reason": "seed-named cluster bucket contains member indices, not RNG seed IDs",
                    } if semantic == "non_training_cluster_or_selection_bucket" else {}),
                }
            elif classification is not None and _normalized_key(key_string) not in {
                "per_seed", "by_seed", "results_by_seed", "seed_results",
            }:
                semantic, executed = classification
                if semantic == "non_training_cluster_or_selection_bucket":
                    yield {
                        "source": source,
                        "field_path": child_path,
                        "semantic": semantic,
                        "executed_gnn": executed,
                        "seeds": [],
                        "reason": "seed-named cluster bucket contains member indices, not RNG seed IDs",
                    }
                elif semantic == "declared_gnn_seed_list_not_execution_evidence":
                    declared_values = _nested_integer_values(child)
                    yield {
                        "source": source,
                        "field_path": child_path,
                        "semantic": semantic,
                        "executed_gnn": executed,
                        "seeds": sorted(set(declared_values)),
                        "reason": "nested declared seed registry; not execution evidence by itself",
                    }
                else:
                    nested_values = _nested_integer_values(child)
                    if nested_values:
                        yield {
                            "source": source,
                            "field_path": child_path,
                            "semantic": "ambiguous",
                            "executed_gnn": False,
                            "seeds": sorted(set(nested_values)),
                            "reason": "numeric seed-like field uses an unsupported value schema",
                        }
            if isinstance(child, (Mapping, list, tuple)):
                yield from _walk_seed_fields(child, source=source, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_seed_fields(child, source=source, path=f"{path}/{index}")


def _extract_manifest_entries(value: Any, source: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            sha = node.get("sha256") or node.get("sha_256") or node.get("hash")
            declared_path = (
                node.get("path") or node.get("relative_path") or node.get("archive_path")
                or node.get("file") or node.get("name")
            )
            if isinstance(sha, str) and re.fullmatch(r"[0-9a-fA-F]{64}", sha):
                if isinstance(declared_path, str):
                    entries.append({
                        "manifest_source": source,
                        "manifest_field_path": path,
                        "declared_path": _normalize_relpath(declared_path),
                        "declared_sha256": sha.casefold(),
                        "declared_size": node.get("size") if isinstance(node.get("size"), int) else None,
                    })
                for key, child in node.items():
                    if isinstance(child, Mapping) and child is node:
                        continue
            for key, child in node.items():
                if isinstance(child, Mapping):
                    child_sha = child.get("sha256") or child.get("sha_256")
                    if (
                        isinstance(key, str) and "/" in key and isinstance(child_sha, str)
                        and re.fullmatch(r"[0-9a-fA-F]{64}", child_sha)
                    ):
                        entries.append({
                            "manifest_source": source,
                            "manifest_field_path": f"{path}/{key}",
                            "declared_path": _normalize_relpath(key),
                            "declared_sha256": child_sha.casefold(),
                            "declared_size": child.get("size") if isinstance(child.get("size"), int) else None,
                        })
                walk(child, f"{path}/{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}/{index}")

    walk(value, "$")
    unique = {_canonical_json(entry): entry for entry in entries}
    return [unique[key] for key in sorted(unique)]


def _structured_kind(name: str) -> str | None:
    lowered = name.casefold()
    if lowered.endswith(".json"):
        return "json"
    if lowered.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if lowered.endswith(".csv"):
        return "csv"
    return None


def _archive_kind(name: str) -> bool:
    lowered = name.casefold()
    return any(lowered.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _decode_utf8(data: bytes, source: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnsafeStructuredInputError(f"structured input is not UTF-8: {source}") from exc


def _scan_structured_bytes(
    data: bytes,
    *,
    source: str,
    kind: str,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    if len(data) > MAX_STRUCTURED_BYTES:
        raise UnsafeStructuredInputError(
            f"structured input exceeds {MAX_STRUCTURED_BYTES} bytes: {source}"
        )
    text = _decode_utf8(data, source)
    values: list[tuple[Any, str]] = []
    test_hashes: list[str] = []
    if kind == "json":
        value, skipped = safe_load_json(text, source)
        values.append((value, "$"))
        test_hashes.extend(skipped)
    elif kind == "jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value, skipped = safe_load_json(line, source, root_path=f"$line/{line_number}")
            values.append((value, f"$line/{line_number}"))
            test_hashes.extend(skipped)
    elif kind == "csv":
        rows, skipped = safe_load_csv(text, source)
        values.extend((row, f"$row/{index}") for index, row in enumerate(rows, start=2))
        test_hashes.extend(skipped)
    else:
        raise ValueError(f"unsupported structured kind: {kind}")

    matches: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    manifest_hint = "manifest" in PurePosixPath(source.split("!", 1)[-1]).name.casefold()
    for value, path in values:
        matches.extend(_walk_seed_fields(value, source=source, path=path))
        if manifest_hint:
            manifest_entries.extend(_extract_manifest_entries(value, source))
    return matches, test_hashes, manifest_entries


def _read_capped(handle: Any, source: str) -> bytes:
    data = handle.read(MAX_STRUCTURED_BYTES + 1)
    if len(data) > MAX_STRUCTURED_BYTES:
        raise UnsafeStructuredInputError(
            f"structured input exceeds {MAX_STRUCTURED_BYTES} bytes: {source}"
        )
    return data


def _iter_scan_files(repo_root: Path, scan_roots: Sequence[str], excluded_target: str) -> Iterator[Path]:
    for relative_root in scan_roots:
        scan_root = repo_root / relative_root
        if not scan_root.exists():
            continue
        if scan_root.is_symlink():
            raise UnsafeStructuredInputError(f"scan root must not be a symlink: {scan_root}")
        if not scan_root.is_dir():
            raise UnsafeStructuredInputError(f"scan root is not a directory: {scan_root}")
        for directory, directory_names, file_names in os.walk(scan_root, followlinks=False):
            current = Path(directory)
            retained_directories = []
            for name in sorted(directory_names):
                candidate = current / name
                relative = candidate.relative_to(repo_root).as_posix()
                if _is_excluded(relative, excluded_target):
                    continue
                if candidate.is_symlink():
                    raise UnsafeStructuredInputError(
                        f"symlinked directory prevents complete historical scan: {relative}"
                    )
                retained_directories.append(name)
            directory_names[:] = retained_directories
            for file_name in sorted(file_names):
                path = current / file_name
                relative = path.relative_to(repo_root).as_posix()
                if _is_excluded(relative, excluded_target):
                    continue
                lowered = file_name.casefold()
                if lowered.endswith(SQLITE_SIDECAR_SUFFIXES):
                    raise UnsafeStructuredInputError(
                        f"SQLite sidecar prevents immutable historical audit: {relative}"
                    )
                if lowered.endswith(SQLITE_SUFFIXES):
                    if relative not in AUDITED_NON_SEED_SQLITE:
                        raise UnsafeStructuredInputError(
                            f"unsupported structured historical format requires audit: {relative}"
                        )
                    if path.is_symlink():
                        raise UnsafeStructuredInputError(
                            f"symlinked structured file prevents identity freeze: {relative}"
                        )
                    yield path
                    continue
                if lowered.endswith((".parquet", ".zip")):
                    raise UnsafeStructuredInputError(
                        f"unsupported structured historical format requires audit: {relative}"
                    )
                if path.is_symlink() and (
                    _structured_kind(file_name) is not None or _archive_kind(file_name)
                ):
                    raise UnsafeStructuredInputError(
                        f"symlinked structured file prevents identity freeze: {relative}"
                    )
                target_name = PurePosixPath(excluded_target).name
                if _archive_kind(file_name) and (
                    file_name == target_name + ".tar"
                    or file_name == target_name + ".tar.gz"
                    or file_name == target_name + ".tgz"
                    or file_name.startswith(target_name + "-")
                ):
                    continue
                if _structured_kind(file_name) is not None or _archive_kind(file_name):
                    yield path


def _scan_archive(
    path: Path,
    *,
    repo_root: Path,
    excluded_target: str,
    match_sink: Any | None = None,
    test_sink: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    archive_relative = path.relative_to(repo_root).as_posix()
    sources: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    test_hashes: list[str] = []
    manifest_entries: list[dict[str, Any]] = []
    occurrence: defaultdict[str, int] = defaultdict(int)
    try:
        archive = tarfile.open(path, mode="r:*")
    except tarfile.TarError as exc:
        raise UnsafeStructuredInputError(f"invalid tar archive: {archive_relative}") from exc
    with archive:
        for member in archive:
            if not member.isfile():
                continue
            member_name = _normalize_relpath(member.name)
            if _is_excluded_archive_member(member_name, excluded_target):
                continue
            lowered_member = member_name.casefold()
            if lowered_member.endswith(SQLITE_SIDECAR_SUFFIXES):
                raise UnsafeStructuredInputError(
                    f"SQLite sidecar inside tar archive requires audit: "
                    f"{archive_relative}!{member_name}"
                )
            if lowered_member.endswith(UNSUPPORTED_STRUCTURED_SUFFIXES) or _archive_kind(
                member_name
            ):
                raise UnsafeStructuredInputError(
                    f"unsupported structured historical format inside tar archive requires audit: "
                    f"{archive_relative}!{member_name}"
                )
            kind = _structured_kind(member_name)
            if kind is None:
                continue
            occurrence[member_name] += 1
            suffix = f"#{occurrence[member_name]}" if occurrence[member_name] > 1 else ""
            locator = f"{archive_relative}!{member_name}{suffix}"
            extracted = archive.extractfile(member)
            if extracted is None:
                raise UnsafeStructuredInputError(f"cannot read tar member: {locator}")
            with extracted:
                data = _read_capped(extracted, locator)
            sources.append({
                "source": locator,
                "kind": f"tar_member_{kind}",
                "size": len(data),
                "sha256": _sha256_bytes(data),
            })
            found, skipped, manifests = _scan_structured_bytes(
                data, source=locator, kind=kind,
            )
            if match_sink is None:
                matches.extend(found)
            else:
                match_sink(found)
            if test_sink is None:
                test_hashes.extend(skipped)
            else:
                test_sink(skipped)
            manifest_entries.extend(manifests)
    return sources, matches, test_hashes, manifest_entries


def build_historical_gnn_seed_inventory(
    repo_root: str | Path,
    *,
    scan_roots: Sequence[str] = DEFAULT_SCAN_ROOTS,
    excluded_target: str = EXCLUDED_TARGET,
    fresh_seeds: Sequence[int] = FRESH_SEEDS,
) -> dict[str, Any]:
    """Scan historical artifacts without writing and return a frozen inventory.

    ``AmbiguousSeedSemanticsError`` is raised after the complete scan when any
    numeric seed-like field cannot be assigned a safe semantic.  The returned
    inventory is deterministic for identical bytes and paths.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    normalized_scan_roots = tuple(_normalize_relpath(item) for item in scan_roots)
    normalized_exclusion = _normalize_relpath(excluded_target)
    requested_fresh = tuple(int(value) for value in fresh_seeds)
    if requested_fresh != FRESH_SEEDS:
        raise ValueError(f"fresh seed registry is fixed at {list(FRESH_SEEDS)}")

    sources: list[dict[str, Any]] = []
    match_groups: dict[tuple[str, str, str, bool, str], dict[str, Any]] = {}
    total_match_count = 0
    test_path_hash_digest = hashlib.sha256()
    test_path_hash_count = 0
    manifest_entries: list[dict[str, Any]] = []

    def merge_matches(found: Sequence[Mapping[str, Any]]) -> None:
        nonlocal total_match_count
        for row in found:
            total_match_count += 1
            path = str(row["field_path"])
            pattern = re.sub(r"/\d+(?=/|$)", "/*", path)
            reason = str(row.get("reason", ""))
            key = (
                str(row["source"]), pattern, str(row["semantic"]),
                bool(row["executed_gnn"]), reason,
            )
            group = match_groups.setdefault(key, {
                "source": key[0], "field_path": pattern,
                "field_path_pattern": pattern, "semantic": key[2],
                "executed_gnn": key[3], "reason": reason or None,
                "occurrence_count": 0, "seeds": set(),
                "path_commitment": hashlib.sha256(),
            })
            group["occurrence_count"] += 1
            group["seeds"].update(int(seed) for seed in row["seeds"])
            encoded = path.encode("utf-8")
            group["path_commitment"].update(len(encoded).to_bytes(8, "big"))
            group["path_commitment"].update(encoded)

    def merge_test_hashes(values: Sequence[str]) -> None:
        nonlocal test_path_hash_count
        for value in values:
            encoded = str(value).encode("ascii")
            test_path_hash_digest.update(len(encoded).to_bytes(8, "big"))
            test_path_hash_digest.update(encoded)
            test_path_hash_count += 1

    for path in _iter_scan_files(root, normalized_scan_roots, normalized_exclusion):
        relative = path.relative_to(root).as_posix()
        audited_sqlite = AUDITED_NON_SEED_SQLITE.get(relative)
        if audited_sqlite is not None:
            file_record = _stable_file_record(path, relative, "audited_non_seed_sqlite")
            if (
                file_record["sha256"] != audited_sqlite["sha256"]
                or file_record["size"] != audited_sqlite["size"]
            ):
                raise UnsafeStructuredInputError(
                    f"audited SQLite identity changed and requires re-audit: {relative}"
                )
            file_record["manual_schema_audit"] = dict(audited_sqlite)
            sources.append(file_record)
            continue
        file_record = _stable_file_record(
            path,
            relative,
            "tar_archive" if _archive_kind(path.name) else f"file_{_structured_kind(path.name)}",
        )
        structured_audit = AUDITED_STRUCTURED_SEED_SEMANTICS.get(relative)
        implicit_seed_audit = AUDITED_IMPLICIT_GNN_SEED_SOURCES.get(relative)
        for audit_name, audit in (
            ("structured seed semantic", structured_audit),
            ("implicit GNN seed contract", implicit_seed_audit),
        ):
            if audit is not None and (
                file_record["sha256"] != audit["sha256"]
                or file_record["size"] != audit["size"]
            ):
                raise UnsafeStructuredInputError(
                    f"audited {audit_name} identity changed and requires re-audit: {relative}"
                )
        if structured_audit is not None:
            file_record["manual_seed_semantics_audit"] = dict(structured_audit)
        if implicit_seed_audit is not None:
            file_record["manual_implicit_seed_contract"] = dict(implicit_seed_audit)
        sources.append(file_record)
        if _archive_kind(path.name):
            member_sources, found, skipped, manifests = _scan_archive(
                path, repo_root=root, excluded_target=normalized_exclusion,
                match_sink=merge_matches, test_sink=merge_test_hashes,
            )
            sources.extend(member_sources)
            merge_matches(found)
            merge_test_hashes(skipped)
            manifest_entries.extend(manifests)
            if _stable_file_record(path, relative, "tar_archive") != file_record:
                raise UnsafeStructuredInputError(f"archive changed while scanning: {relative}")
            continue
        kind = _structured_kind(path.name)
        assert kind is not None
        with path.open("rb") as handle:
            data = _read_capped(handle, relative)
        if len(data) != file_record["size"] or _sha256_bytes(data) != file_record["sha256"]:
            raise UnsafeStructuredInputError(f"structured source changed while scanning: {relative}")
        found, skipped, manifests = _scan_structured_bytes(data, source=relative, kind=kind)
        if structured_audit is not None:
            audited_rows = [
                row for row in found if row["field_path"] == structured_audit["field_path"]
            ]
            if len(audited_rows) != 1 or audited_rows[0]["seeds"] != structured_audit["expected_seeds"]:
                raise UnsafeStructuredInputError(
                    f"audited seed semantic field changed and requires re-audit: {relative}"
                )
        if implicit_seed_audit is not None:
            found.append({
                "source": relative,
                "field_path": implicit_seed_audit["field_path"],
                "semantic": implicit_seed_audit["semantic"],
                "executed_gnn": bool(implicit_seed_audit["executed_gnn"]),
                "seeds": list(implicit_seed_audit["seeds"]),
                "reason": "exact-byte result plus frozen historical training-loop contract",
            })
        merge_matches(found)
        merge_test_hashes(skipped)
        manifest_entries.extend(manifests)

    sources.sort(key=lambda row: row["source"])
    matches = []
    for group in match_groups.values():
        row = {
            key: value for key, value in group.items()
            if key not in {"path_commitment", "seeds"} and value is not None
        }
        row["seeds"] = sorted(group["seeds"])
        row["field_path_commitment_sha256"] = group["path_commitment"].hexdigest()
        matches.append(row)
    matches.sort(key=lambda row: (row["source"], row["field_path"], row["semantic"], row["seeds"]))
    manifest_entries.sort(key=lambda row: (
        row["manifest_source"], row["declared_path"], row["declared_sha256"], row["manifest_field_path"],
    ))
    ambiguous = [row for row in matches if row["semantic"] == "ambiguous"]
    if ambiguous:
        raise AmbiguousSeedSemanticsError(ambiguous)

    semantic_values: defaultdict[str, set[int]] = defaultdict(set)
    executed: set[int] = set()
    for row in matches:
        semantic_values[row["semantic"]].update(int(value) for value in row["seeds"])
        if row["executed_gnn"]:
            executed.update(int(value) for value in row["seeds"])

    source_sha_index = {row["source"]: row["sha256"] for row in sources}
    suffix_sha_index: defaultdict[str, set[str]] = defaultdict(set)
    for source, sha in source_sha_index.items():
        locator = source.split("!", 1)[1] if "!" in source else source
        locator = re.sub(r"#\d+$", "", locator)
        parts = PurePosixPath(locator).parts
        for index in range(len(parts)):
            suffix_sha_index["/".join(parts[index:])].add(sha)
    for entry in manifest_entries:
        declared = entry["declared_path"]
        exact = source_sha_index.get(declared)
        observed = [exact] if exact is not None else sorted(suffix_sha_index.get(declared, set()))
        entry["observed_matching_sha256"] = observed
        entry["sha256_covered"] = entry["declared_sha256"] in observed
    uncovered_structured = [
        entry for entry in manifest_entries
        if _structured_kind(entry["declared_path"]) is not None and not entry["sha256_covered"]
    ]
    if uncovered_structured:
        raise UnsafeStructuredInputError(
            "structured manifest entries lack matching scanned bytes: "
            + ", ".join(entry["declared_path"] for entry in uncovered_structured[:8])
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scan_roots": list(normalized_scan_roots),
        "excluded_target": normalized_exclusion,
        "fixed_fresh_seeds": list(requested_fresh),
        "source_count": len(sources),
        "sources": sources,
        "seed_field_match_count": total_match_count,
        "seed_field_group_count": len(matches),
        "seed_field_matches": matches,
        "semantic_seed_sets": {
            key: sorted(values) for key, values in sorted(semantic_values.items())
        },
        "executed_gnn_training_seeds": sorted(executed),
        "fresh_seed_collisions": sorted(executed.intersection(requested_fresh)),
        "fresh_seed_collision_free": not executed.intersection(requested_fresh),
        "skipped_test_field_count": test_path_hash_count,
        "skipped_test_field_path_commitment_sha256": test_path_hash_digest.hexdigest(),
        "test_value_policy": (
            "values under JSON keys or CSV columns containing 'test' were lexically skipped "
            "before deserialization; only SHA-256 field-path commitments and counts are retained"
        ),
        "manifest_entry_count": len(manifest_entries),
        "manifest_coverage": manifest_entries,
    }
    payload["inventory_root_sha256"] = _sha256_bytes(_canonical_json(payload))
    return payload


def verify_fixed_fresh_seeds(inventory: Mapping[str, Any]) -> None:
    """Assert the fixed seed registry and collision-free result."""
    if tuple(inventory.get("fixed_fresh_seeds", ())) != FRESH_SEEDS:
        raise InventoryError("inventory does not contain the fixed 100000..100029 registry")
    collisions = sorted(
        set(int(value) for value in inventory.get("executed_gnn_training_seeds", ()))
        & set(FRESH_SEEDS)
    )
    if collisions:
        raise FreshSeedCollisionError(collisions)
    if inventory.get("fresh_seed_collisions") not in ([], ()):
        raise InventoryError("inventory collision fields are internally inconsistent")
    expected_root = inventory.get("inventory_root_sha256")
    without_root = dict(inventory)
    without_root.pop("inventory_root_sha256", None)
    actual_root = _sha256_bytes(_canonical_json(without_root))
    if expected_root != actual_root:
        raise InventoryError("inventory root SHA-256 mismatch")


def has_forbidden_json_key(text: str) -> bool:
    """Return true when a JSON object key names a forbidden metric namespace.

    This lexical check does not decode the associated value.  It is used for
    canonical new result shards, where any such key is a protocol violation.
    """
    for match in re.finditer(r'"((?:[^"\\]|\\.)*)"\s*:', text):
        try:
            key = json.loads('"' + match.group(1) + '"')
        except json.JSONDecodeError:
            continue
        if _is_test_key(str(key)):
            return True
    return False


def scan_repository_training_seeds(
    *, repo_root: Path, excluded_target: Path, proposed_seeds: Sequence[int],
) -> dict[str, Any]:
    """Compatibility wrapper used by the checkpoint-freeze preflight."""
    root = Path(repo_root).resolve()
    excluded = Path(excluded_target).resolve()
    try:
        relative_exclusion = excluded.relative_to(root).as_posix()
    except ValueError as exc:
        raise InventoryError("excluded target must be inside the repository") from exc
    if tuple(int(seed) for seed in proposed_seeds) != FRESH_SEEDS:
        raise InventoryError("fresh seeds are fixed at 100000..100029")
    try:
        inventory = build_historical_gnn_seed_inventory(
            root, excluded_target=relative_exclusion, fresh_seeds=proposed_seeds,
        )
        verify_fixed_fresh_seeds(inventory)
    except (InventoryError, OSError, tarfile.TarError, ValueError) as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "excluded_target": str(excluded),
            "fixed_fresh_seeds": list(FRESH_SEEDS),
            "passed": False,
            "failure_reason": f"{type(error).__name__}: {error}",
            "fresh_seed_collisions": list(getattr(error, "collisions", ())),
        }
    wrapper = {
        "schema_version": SCHEMA_VERSION,
        "absolute_excluded_target": str(excluded),
        "inventory": inventory,
        "inventory_root": inventory["inventory_root_sha256"],
        "used_training_seeds": inventory["executed_gnn_training_seeds"],
        "fresh_seed_collisions": inventory["fresh_seed_collisions"],
        "passed": True,
    }
    wrapper["wrapper_root_sha256"] = _sha256_bytes(_canonical_json(wrapper))
    return wrapper


def verify_inventory_wrapper(wrapper: Mapping[str, Any], *, excluded_target: Path) -> None:
    payload = dict(wrapper)
    recorded = payload.pop("wrapper_root_sha256", None)
    if recorded != _sha256_bytes(_canonical_json(payload)):
        raise InventoryError("inventory wrapper root SHA-256 mismatch")
    resolved_excluded = Path(excluded_target).resolve()
    fixed_target_parts = PurePosixPath(EXCLUDED_TARGET).parts
    if tuple(resolved_excluded.parts[-len(fixed_target_parts):]) != fixed_target_parts:
        raise InventoryError("inventory exclusion is not the fixed checkpoint-freeze target")
    if wrapper.get("absolute_excluded_target") != str(resolved_excluded):
        raise InventoryError("inventory excluded a different target")
    inventory = wrapper.get("inventory")
    if not isinstance(inventory, Mapping):
        raise InventoryError("inventory wrapper payload is missing")
    if inventory.get("excluded_target") != EXCLUDED_TARGET:
        raise InventoryError("inventory payload excluded a non-fixed target")
    verify_fixed_fresh_seeds(inventory)
    if wrapper.get("inventory_root") != inventory.get("inventory_root_sha256"):
        raise InventoryError("inventory root alias mismatch")
    if wrapper.get("used_training_seeds") != inventory.get("executed_gnn_training_seeds"):
        raise InventoryError("executed training seed alias mismatch")
    if wrapper.get("fresh_seed_collisions") != inventory.get("fresh_seed_collisions"):
        raise InventoryError("fresh collision alias mismatch")
    if wrapper.get("passed") is not True:
        raise InventoryError("inventory wrapper is not passed")


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument(
        "--print-json", action="store_true",
        help="print the inventory to stdout; this module never writes files",
    )
    args = parser.parse_args(argv)
    inventory = build_historical_gnn_seed_inventory(args.repo_root)
    verify_fixed_fresh_seeds(inventory)
    if args.print_json:
        print(json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(inventory["inventory_root_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


# Phase-C-v2 conservative seed-collision API (ordinal 5).  Everything above
# this line is the byte-identical ordinal-4 scanner.
import math as _conservative_math

CONSERVATIVE_API_SCHEMA_VERSION = "phase-c-v2-conservative-seed-collision-superset-v1"


def _conservative_integer_leaves(value: Any) -> list[int]:
    """Return every completely enumerable explicit integer leaf.

    Boolean values and numeric-looking values which are not full base-10
    integers are rejected.  Ordinary non-numeric metadata is ignored.
    """
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, float):
        if not _conservative_math.isfinite(value):
            raise UnsafeStructuredInputError("non-integral seed-bearing number")
        if value.is_integer():
            return [int(value)]
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)", stripped):
            return [int(stripped, 10)]
        return [int(token, 10) for token in re.findall(r"[+-]?[0-9]+", stripped)]
    if isinstance(value, (list, tuple)):
        output: list[int] = []
        for item in value:
            output.extend(_conservative_integer_leaves(item))
        return output
    if isinstance(value, Mapping):
        output = []
        for item in value.values():
            output.extend(_conservative_integer_leaves(item))
        return output
    if value is None:
        return []
    raise UnsafeStructuredInputError("unsupported seed-bearing value")


def _conservative_walk_seed_fields(
    value: Any, *, source: str, path: str = "$",
) -> Iterator[dict[str, Any]]:
    if isinstance(value, Mapping):
        parent_name = _normalized_key(PurePosixPath(path).name)
        seed_indexed = parent_name in {
            "per_seed", "by_seed", "results_by_seed", "seed_results",
        } or parent_name.endswith("_by_seed")
        if seed_indexed:
            for raw_key in value:
                key = str(raw_key)
                if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", key):
                    yield {
                        "source": source, "field_path": f"{path}/{key}",
                        "semantic": "conservative_explicit_seed_index_key",
                        "executed_gnn": False, "seeds": [int(key)],
                    }
                elif any(character.isdigit() for character in key):
                    raise UnsafeStructuredInputError("malformed seed-index mapping key")
        for key, child in value.items():
            key_string = str(key)
            escaped = key_string.replace("~", "~0").replace("/", "~1")
            child_path = f"{path}/{escaped}"
            classification = _classify_seed_field(
                key=key_string, field_path=child_path, record=value, source=source,
            )
            if classification is not None:
                semantic, executed = classification
                leaves = _conservative_integer_leaves(child)
                if leaves:
                    yield {
                        "source": source, "field_path": child_path,
                        "semantic": "conservative_explicit_occupied",
                        "old_semantic": semantic, "executed_gnn": executed,
                        "seeds": sorted(set(leaves)),
                    }
            if isinstance(child, (Mapping, list, tuple)):
                yield from _conservative_walk_seed_fields(
                    child, source=source, path=child_path,
                )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _conservative_walk_seed_fields(
                child, source=source, path=f"{path}/{index}",
            )


def build_conservative_seed_collision_superset(
    repo_root: str | Path,
    *,
    scan_roots: Sequence[str] = DEFAULT_SCAN_ROOTS,
    excluded_target: str = EXCLUDED_TARGET,
) -> dict[str, Any]:
    """Build the ordinal-5 conservative occupied set without changing old APIs."""
    original_walker = globals()["_walk_seed_fields"]
    original_structured_kind = globals()["_structured_kind"]
    original_normalize_relpath = globals()["_normalize_relpath"]
    repo_prefix = original_normalize_relpath(Path(repo_root).resolve())
    explicit_rows: list[dict[str, Any]] = []

    def merged_walker(value: Any, *, source: str, path: str = "$") -> Iterator[dict[str, Any]]:
        conservative = list(_conservative_walk_seed_fields(value, source=source, path=path))
        explicit_rows.extend(conservative)
        yield from conservative

    def structured_kind(name: str) -> str | None:
        if PurePosixPath(name).as_posix() in {
            "raw/class_map.json", "raw/role.json",
        }:
            return None
        return original_structured_kind(name)

    def normalize_relpath(value: str | PurePosixPath) -> str:
        normalized = original_normalize_relpath(value)
        prefix = f"{repo_prefix}/"
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
        return normalized

    globals()["_walk_seed_fields"] = merged_walker
    globals()["_structured_kind"] = structured_kind
    globals()["_normalize_relpath"] = normalize_relpath
    try:
        base = build_historical_gnn_seed_inventory(
            repo_root, scan_roots=scan_roots, excluded_target=excluded_target,
        )
    finally:
        globals()["_walk_seed_fields"] = original_walker
        globals()["_structured_kind"] = original_structured_kind
        globals()["_normalize_relpath"] = original_normalize_relpath
    a_values = sorted({int(seed) for row in explicit_rows for seed in row["seeds"]})
    p_values = sorted({
        int(seed) for row in explicit_rows if row.get("executed_gnn")
        for seed in row["seeds"]
    })
    k_values = sorted(set(base["executed_gnn_training_seeds"]) - set(a_values))
    s_values = sorted(set(a_values) | set(k_values))
    payload = {
        "schema_version": CONSERVATIVE_API_SCHEMA_VERSION,
        "scan_roots": list(scan_roots),
        "sources": base["sources"],
        "source_count": base["source_count"],
        "test_skip": {
            "count": base["skipped_test_field_count"],
            "path_commitment_sha256": base["skipped_test_field_path_commitment_sha256"],
        },
        "set_commitments": {
            name: {"count": len(values), "root_sha256": _sha256_bytes(_canonical_json(values))}
            for name, values in {
                "P": p_values, "A": a_values, "K": k_values,
                "K_positive": k_values, "E_recognized": sorted(set(p_values) | set(k_values)),
                "S": s_values,
            }.items()
        },
        "set_proofs": {
            "s_equals_union_a_k": s_values == sorted(set(a_values) | set(k_values)),
            "p_subset_or_equal_a": set(p_values) <= set(a_values),
            "k_positive_subset_or_equal_k": True,
            "e_recognized_subset_or_equal_s": set(p_values) | set(k_values) <= set(s_values),
        },
        "occupied_seed_values": s_values,
        "old_inventory_root_sha256": base["inventory_root_sha256"],
    }
    verify_conservative_seed_collision_superset(payload)
    return payload


def verify_conservative_seed_collision_superset(inventory: Mapping[str, Any]) -> None:
    if inventory.get("schema_version") != CONSERVATIVE_API_SCHEMA_VERSION:
        raise InventoryError("conservative inventory schema mismatch")
    values = inventory.get("occupied_seed_values")
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise InventoryError("conservative occupied values are not native integers")
    if values != sorted(set(values)):
        raise InventoryError("conservative occupied values are not sorted unique")
    commitment = inventory.get("set_commitments", {}).get("S", {})
    if commitment != {
        "count": len(values), "root_sha256": _sha256_bytes(_canonical_json(values)),
    }:
        raise InventoryError("conservative occupied-set commitment mismatch")
    proofs = inventory.get("set_proofs", {})
    if not proofs or not all(proofs.values()):
        raise InventoryError("conservative occupied-set proof failed")
