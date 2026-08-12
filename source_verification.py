"""Content-addressed source and checkpoint verification for formal launches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCE_MANIFEST_FORMAT_VERSION = 1


class SourceVerificationError(RuntimeError):
    """Raised when a frozen formal-launch input no longer matches its bytes."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_id_from_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    """Return the stable ID of the formal-run-required source rows."""

    identity_rows = []
    for row in rows:
        if not bool(row.get("formal_run_required", False)):
            continue
        identity_rows.append(
            {
                "path": str(row["path"]),
                "size_bytes": int(row["size_bytes"]),
                "sha256": str(row["sha256"]),
            }
        )
    identity_rows.sort(key=lambda row: row["path"])
    if not identity_rows:
        raise ValueError("source manifest has no formal-run-required files")
    return canonical_json_sha256(identity_rows)


def _safe_relative_path(raw_path: Any) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SourceVerificationError("source manifest path must be non-empty")
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise SourceVerificationError(
            f"source manifest path must stay repository-relative: {raw_path!r}"
        )
    return path.as_posix()


def discover_source_gate_paths(
    repo_root: str | os.PathLike[str],
    rules: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Discover the exact source namespace guarded against unexpected files."""

    root = Path(repo_root).resolve()
    discovered: set[str] = set()
    for rule in rules:
        rule_root_text = _safe_relative_path(rule.get("root", "."))
        rule_root = root if rule_root_text == "." else root / rule_root_text
        recursive = bool(rule.get("recursive", False))
        suffixes = tuple(str(value) for value in rule.get("suffixes", ()))
        if not suffixes:
            raise SourceVerificationError(
                f"source discovery rule {rule_root_text!r} has no suffixes"
            )
        if not rule_root.is_dir():
            raise SourceVerificationError(
                f"source discovery root is missing: {rule_root_text}"
            )
        iterator = rule_root.rglob("*") if recursive else rule_root.iterdir()
        for path in iterator:
            if path.is_file() and path.suffix in suffixes:
                discovered.add(path.relative_to(root).as_posix())
    return sorted(discovered)


def load_source_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SourceVerificationError(
            f"expected source manifest is missing: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceVerificationError(
            f"cannot read expected source manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SourceVerificationError("expected source manifest must be a JSON object")
    if payload.get("format_version") != SOURCE_MANIFEST_FORMAT_VERSION:
        raise SourceVerificationError(
            "unsupported expected source manifest format: "
            f"{payload.get('format_version')!r}"
        )
    return payload


def verify_frozen_source(
    manifest_path: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str],
    expected_source_id: str,
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify source membership/content and checkpoint bytes before launch."""

    payload = load_source_manifest(manifest_path)
    root = Path(repo_root).resolve()
    manifest_id = payload.get("source_id")
    if not isinstance(expected_source_id, str) or not expected_source_id.strip():
        raise SourceVerificationError("--frozen-source-id must be non-empty")
    if manifest_id != expected_source_id:
        raise SourceVerificationError(
            "frozen source ID mismatch: "
            f"expected {expected_source_id}, manifest records {manifest_id}"
        )
    raw_rows = payload.get("files")
    if not isinstance(raw_rows, list):
        raise SourceVerificationError("source manifest files must be a list")
    calculated_id = source_id_from_rows(raw_rows)
    if calculated_id != manifest_id:
        raise SourceVerificationError(
            "source manifest rows do not reproduce frozen source ID: "
            f"manifest={manifest_id} calculated={calculated_id}"
        )

    required_rows: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict) or not bool(
            row.get("formal_run_required", False)
        ):
            continue
        relative = _safe_relative_path(row.get("path"))
        if relative in required_rows:
            raise SourceVerificationError(
                f"duplicate formal-run-required source path: {relative}"
            )
        required_rows[relative] = row

    rules = payload.get("source_gate", {}).get("discovery_rules")
    if not isinstance(rules, list) or not rules:
        raise SourceVerificationError(
            "source manifest must define source_gate.discovery_rules"
        )
    discovered = set(discover_source_gate_paths(root, rules))
    expected_paths = set(required_rows)
    missing_members = sorted(expected_paths - discovered)
    unexpected_members = sorted(discovered - expected_paths)
    if missing_members:
        raise SourceVerificationError(
            "missing frozen source file(s): " + ", ".join(missing_members)
        )
    if unexpected_members:
        raise SourceVerificationError(
            "unexpected source file(s) in frozen namespace: "
            + ", ".join(unexpected_members)
        )

    verified_files = []
    for relative in sorted(required_rows):
        row = required_rows[relative]
        path = root / relative
        if not path.is_file():
            raise SourceVerificationError(f"missing frozen source file: {relative}")
        actual_size = int(path.stat().st_size)
        expected_size = int(row.get("size_bytes", -1))
        if actual_size != expected_size:
            raise SourceVerificationError(
                f"source size mismatch for {relative}: "
                f"expected {expected_size}, got {actual_size}"
            )
        actual_sha256 = sha256_file(path)
        expected_sha256 = str(row.get("sha256"))
        if actual_sha256 != expected_sha256:
            raise SourceVerificationError(
                f"source SHA-256 mismatch for {relative}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        verified_files.append(relative)

    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise SourceVerificationError("source manifest checkpoint must be an object")
    requested_checkpoint = Path(checkpoint_path).expanduser()
    if not requested_checkpoint.is_absolute():
        requested_checkpoint = root / requested_checkpoint
    requested_checkpoint = requested_checkpoint.resolve()
    recorded_checkpoint = Path(str(checkpoint.get("path", ""))).expanduser()
    if not recorded_checkpoint.is_absolute():
        recorded_checkpoint = root / recorded_checkpoint
    recorded_checkpoint = recorded_checkpoint.resolve()
    if requested_checkpoint != recorded_checkpoint:
        raise SourceVerificationError(
            "checkpoint path mismatch: "
            f"manifest={recorded_checkpoint} requested={requested_checkpoint}"
        )
    if not requested_checkpoint.is_file():
        raise SourceVerificationError(
            f"frozen checkpoint is missing: {requested_checkpoint}"
        )
    actual_checkpoint_size = int(requested_checkpoint.stat().st_size)
    expected_checkpoint_size = int(checkpoint.get("size_bytes", -1))
    if actual_checkpoint_size != expected_checkpoint_size:
        raise SourceVerificationError(
            "checkpoint size mismatch: "
            f"expected {expected_checkpoint_size}, got {actual_checkpoint_size}"
        )
    actual_checkpoint_sha256 = sha256_file(requested_checkpoint)
    expected_checkpoint_sha256 = str(checkpoint.get("sha256"))
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise SourceVerificationError(
            "checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_sha256}, got {actual_checkpoint_sha256}"
        )

    return {
        "status": "completed",
        "source_id": manifest_id,
        "verified_source_file_count": len(verified_files),
        "verified_source_files": verified_files,
        "checkpoint": {
            "path": str(requested_checkpoint),
            "size_bytes": actual_checkpoint_size,
            "sha256": actual_checkpoint_sha256,
        },
    }
