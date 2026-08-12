"""Build a metadata-only inventory of repository experiment artifacts.

The scanner never opens artifact contents: it walks selected result, log, data,
checkpoint, archive, and cache paths and records only lstat metadata plus Git
classification.  It does not follow symbolic links.  The only writes are the
three reports created below an explicit ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence


REPORT_FIELDS = (
    "path",
    "type",
    "git_status",
    "file_count",
    "total_size_bytes",
    "latest_modified_at",
    "likely_generator",
    "known_reader",
    "rebuildable",
    "contains_reproducibility_state",
    "retention_policy",
)
ARTIFACT_SUFFIXES = (
    ".db",
    ".mat",
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".sqlite",
    ".sqlite3",
    ".tar.gz",
    ".tgz",
    ".zip",
)
ROOT_NAMES = {
    ".pytest_cache",
    "Cora",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "data",
    "raw",
}


def _git_output(root: Path, args: Sequence[str], *, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        input=input_bytes,
        check=False,
        capture_output=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"git {' '.join(args)} failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout


def _tracked_paths(root: Path) -> set[str]:
    output = _git_output(root, ["ls-files", "-z"])
    return {
        value.decode("utf-8", errors="surrogateescape")
        for value in output.split(b"\0")
        if value
    }


def _ignored_paths(root: Path, relative_paths: Iterable[str]) -> set[str]:
    values = sorted(set(relative_paths))
    if not values:
        return set()
    payload = b"\0".join(
        value.encode("utf-8", errors="surrogateescape") for value in values
    ) + b"\0"
    output = _git_output(
        root,
        ["check-ignore", "--stdin", "-z", "--no-index"],
        input_bytes=payload,
    )
    return {
        value.decode("utf-8", errors="surrogateescape")
        for value in output.split(b"\0")
        if value
    }


def _is_artifact_root(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    return (
        name in ROOT_NAMES
        or lower.startswith("result")
        or lower.startswith("log")
        or lower.startswith("checkpoint")
        or any(lower.endswith(suffix) for suffix in ARTIFACT_SUFFIXES)
    )


def _walk_files(path: Path) -> tuple[list[Path], bool]:
    if path.is_symlink() or path.is_file():
        return [path], False
    files: list[Path] = []
    saw_directory = False

    def on_error(error: OSError) -> None:
        raise error

    for directory, names, filenames in os.walk(
        path, topdown=True, followlinks=False, onerror=on_error
    ):
        saw_directory = True
        directory_path = Path(directory)
        for name in list(names):
            candidate = directory_path / name
            if candidate.is_symlink():
                files.append(candidate)
                names.remove(name)
        files.extend(directory_path / name for name in filenames)
    return files, saw_directory and not files


def _scope_for(root: Path, top_level: Path, path: Path) -> str:
    relative = path.relative_to(root)
    if top_level.name.lower().startswith(("result", "log")) or top_level.name == "artifacts":
        parts = relative.parts
        if len(parts) >= 2:
            return Path(parts[0], parts[1]).as_posix()
    return top_level.relative_to(root).as_posix()


def classify_artifact(relative: str) -> str:
    """Classify an artifact using path/name evidence without reading it."""

    lower = relative.lower()
    name = Path(lower).name
    suffixes = Path(lower).suffixes
    if "__pycache__" in lower or ".pytest_cache" in lower or name.endswith(".pyc"):
        return "cache/resume"
    if lower.startswith("logs") or name.endswith(".log"):
        return "log"
    if name.endswith((".tar.gz", ".tgz", ".zip", ".sha256")):
        return "archive"
    if name.endswith((".db", ".sqlite", ".sqlite3")):
        return "database"
    if any(token in name for token in ("manifest", "provenance", "source_id")):
        return "manifest/provenance"
    if name.endswith((".pt", ".pth", ".pkl")):
        if any(token in name for token in ("resume", "rng", "gp_", "state")):
            return "cache/resume"
        return "checkpoint"
    if lower.startswith(("data/", "cora/", "raw/")):
        return "data"
    if any(token in lower for token in ("preflight", "smoke", "validation")):
        return "preflight"
    if any(token in lower for token in ("final_eval", "final-eval", "top10", "test30")):
        return "final evaluation"
    if any(
        token in lower
        for token in ("posthoc", "diagnostic", "analysis", "analyse", "report")
    ):
        return "posthoc/diagnostics"
    if lower.startswith("result"):
        return "search"
    if suffixes and suffixes[-1] in {".npy", ".npz", ".mat"}:
        return "data"
    return "unknown"


def _is_reproducibility_state(relative: str, artifact_type: str) -> bool:
    lower = relative.lower()
    if artifact_type in {
        "checkpoint",
        "database",
        "manifest/provenance",
        "search",
        "final evaluation",
    }:
        return True
    return any(
        token in lower
        for token in ("history", "resume", "fingerprint", "gp_state", "rng_state")
    )


def _descriptions(artifact_type: str, critical: bool) -> tuple[str, str, bool, str]:
    descriptions = {
        "search": ("bo_phase4.py / cross_dataset_runner.py", "final_eval.py; analysis tools"),
        "final evaluation": ("final_eval.py / evaluation launchers", "posthoc analysis tools"),
        "posthoc/diagnostics": ("analyse/ and scripts/analysis/", "reporting tools / users"),
        "preflight": ("scripts/validation/", "formal_matrix.py / operator review"),
        "checkpoint": ("train.py / NAS evaluation", "models.py / search and evaluation entrypoints"),
        "log": ("search, evaluation, and analysis launchers", "operators / diagnostics"),
        "cache/resume": ("search or Python/pytest runtime", "resume logic or runtime"),
        "manifest/provenance": ("source_freeze.py / run orchestration", "source_verification.py / runners"),
        "database": ("experiment or analysis runtime", "resume / analysis tools"),
        "archive": ("manual or maintenance archival", "manual restore / verification"),
        "data": ("dataset loaders or downloads", "dataset_utils.py / training"),
        "unknown": ("unknown", "unknown"),
    }
    generator, reader = descriptions[artifact_type]
    rebuildable = artifact_type == "posthoc/diagnostics" and not critical
    if artifact_type == "cache/resume" and not critical:
        rebuildable = True
    if critical:
        policy = "retain with its owning run; do not delete or relocate"
    elif rebuildable:
        policy = "retain until inputs are verified; may be regenerated explicitly"
    elif artifact_type == "log":
        policy = "retain as operational evidence"
    elif artifact_type == "unknown":
        policy = "keep in place pending user classification"
    else:
        policy = "retain in place"
    return generator, reader, rebuildable, policy


def build_inventory(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if not (root / ".git").is_dir():
        raise ValueError(f"repository root has no .git directory: {root}")
    top_levels = sorted(
        (path for path in root.iterdir() if path.name != ".git" and _is_artifact_root(path)),
        key=lambda path: path.name,
    )
    records: list[tuple[Path, str, str, os.stat_result]] = []
    empty_scopes: set[str] = set()
    for top_level in top_levels:
        files, empty = _walk_files(top_level)
        if empty:
            empty_scopes.add(top_level.relative_to(root).as_posix())
        for path in files:
            relative = path.relative_to(root).as_posix()
            records.append(
                (path, _scope_for(root, top_level, path), relative, path.lstat())
            )

    tracked = _tracked_paths(root)
    ignored = _ignored_paths(root, (record[2] for record in records))
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for _path, scope, relative, stat in records:
        artifact_type = classify_artifact(relative)
        key = (scope, artifact_type)
        row = grouped.setdefault(
            key,
            {
                "path": scope,
                "type": artifact_type,
                "statuses": set(),
                "file_count": 0,
                "total_size_bytes": 0,
                "latest_mtime": 0.0,
                "contains_reproducibility_state": False,
            },
        )
        status = "tracked" if relative in tracked else "ignored" if relative in ignored else "untracked"
        row["statuses"].add(status)
        row["file_count"] += 1
        row["total_size_bytes"] += int(stat.st_size)
        row["latest_mtime"] = max(row["latest_mtime"], float(stat.st_mtime))
        row["contains_reproducibility_state"] = bool(
            row["contains_reproducibility_state"]
            or _is_reproducibility_state(relative, artifact_type)
        )
    for scope in empty_scopes:
        grouped[(scope, "unknown")] = {
            "path": scope,
            "type": "unknown",
            "statuses": {"ignored" if scope in ignored else "untracked"},
            "file_count": 0,
            "total_size_bytes": 0,
            "latest_mtime": (root / scope).lstat().st_mtime,
            "contains_reproducibility_state": False,
        }

    entries = []
    for key in sorted(grouped):
        row = grouped[key]
        statuses = sorted(row.pop("statuses"))
        critical = bool(row["contains_reproducibility_state"])
        generator, reader, rebuildable, policy = _descriptions(row["type"], critical)
        entries.append(
            {
                **{name: row[name] for name in ("path", "type")},
                "git_status": statuses[0] if len(statuses) == 1 else "mixed:" + "+".join(statuses),
                "file_count": row["file_count"],
                "total_size_bytes": row["total_size_bytes"],
                "latest_modified_at": datetime.fromtimestamp(
                    row["latest_mtime"], tz=timezone.utc
                ).isoformat(),
                "likely_generator": generator,
                "known_reader": reader,
                "rebuildable": rebuildable,
                "contains_reproducibility_state": critical,
                "retention_policy": policy,
            }
        )
    by_type: dict[str, dict[str, int]] = {}
    for entry in entries:
        summary = by_type.setdefault(entry["type"], {"file_count": 0, "total_size_bytes": 0})
        summary["file_count"] += int(entry["file_count"])
        summary["total_size_bytes"] += int(entry["total_size_bytes"])
    return {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "scan_policy": "metadata only; artifact file contents not opened; symlinks not followed",
        "summary": {
            "entry_count": len(entries),
            "file_count": sum(int(entry["file_count"]) for entry in entries),
            "total_size_bytes": sum(int(entry["total_size_bytes"]) for entry in entries),
            "by_type": by_type,
        },
        "entries": entries,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Repository artifact inventory",
        "",
        f"Generated: `{payload['created_at']}`",
        "",
        "The inventory is metadata-only. Rebuildability is advisory; ignored does not mean disposable.",
        "",
        "| Path | Type | Git status | Files | Bytes | Latest modified | Rebuildable | Critical state | Retention |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in payload["entries"]:
        values = (
            row["path"],
            row["type"],
            row["git_status"],
            row["file_count"],
            row["total_size_bytes"],
            row["latest_modified_at"],
            row["rebuildable"],
            row["contains_reproducibility_state"],
            row["retention_policy"],
        )
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines) + "\n"


def write_reports(payload: dict[str, Any], output_dir: str | os.PathLike[str]) -> dict[str, str]:
    output = Path(output_dir).resolve()
    targets = {
        "json": output / "repository_inventory.json",
        "csv": output / "repository_inventory.csv",
        "markdown": output / "repository_inventory.md",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite inventory report(s): " + ", ".join(existing))
    output.mkdir(parents=True, exist_ok=True)
    targets["json"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with targets["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(payload["entries"])
    targets["markdown"].write_text(_markdown(payload), encoding="utf-8")
    return {name: str(path) for name, path in targets.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_inventory(args.repo_root)
    reports = write_reports(payload, args.output_dir)
    print(json.dumps({"summary": payload["summary"], "reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
