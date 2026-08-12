"""Create and verify a no-commit, content-addressed formal source freeze."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Sequence

import torch

import cross_dataset_runner
import formal_matrix
from initialization_wgmm_ted import canonical_json_fingerprint
from source_verification import (
    SOURCE_MANIFEST_FORMAT_VERSION,
    discover_source_gate_paths,
    sha256_file,
    source_id_from_rows,
    verify_frozen_source,
)


SOURCE_GATE_DISCOVERY_RULES = [
    {"root": ".", "recursive": False, "suffixes": [".py"]},
    {"root": "analyse", "recursive": True, "suffixes": [".py"]},
    {"root": "configs", "recursive": True, "suffixes": [".json"]},
    {"root": "scripts", "recursive": True, "suffixes": [".py", ".sh"]},
    {"root": "surrogate", "recursive": True, "suffixes": [".py"]},
    {
        "root": "legacy/phased_cora_pipeline",
        "recursive": True,
        "suffixes": [".py"],
    },
]
SNAPSHOT_EXCLUDED_PREFIXES = (
    ".git/",
    "results/",
    "logs/",
    "results-wgmm1/",
    "results-1-LHS/",
    "server_validation/",
    "legacy_artifacts/",
    "checkpoints/",
    "artifacts/reference/",
    "phase4_seedfair_parts_",
    "Cora/",
    "software/",
)


def _run(
    command: Sequence[str], *, root: Path, timeout: float = 120.0
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout


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


def _atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _git_file_sets(root: Path) -> tuple[set[str], set[str]]:
    tracked = {
        value
        for value in _run(["git", "ls-files"], root=root).splitlines()
        if value
    }
    untracked = {
        value
        for value in _run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            root=root,
        ).splitlines()
        if value
    }
    return tracked, untracked


def _is_excluded(relative: str) -> bool:
    return any(
        relative == prefix.rstrip("/") or relative.startswith(prefix)
        for prefix in SNAPSHOT_EXCLUDED_PREFIXES
    )


def snapshot_source_paths(root: Path, gate_paths: Iterable[str]) -> list[str]:
    selected = set(gate_paths)
    patterns = (
        "tests/**/*.py",
        "docs/**/*.md",
        "*.md",
        "*.sh",
    )
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if not _is_excluded(relative):
                    selected.add(relative)
    for special in ("AGENTS.md", "requirements.txt"):
        if (root / special).is_file():
            selected.add(special)
    output = []
    for relative in sorted(selected):
        path = root / relative
        if path.is_file() and not _is_excluded(relative):
            output.append(relative)
    return output


def _category(relative: str, formal_required: bool) -> str:
    if formal_required:
        if relative.startswith("configs/"):
            return "formal_configuration"
        if relative.startswith("surrogate/"):
            return "formal_surrogate_runtime"
        if relative.startswith("analyse/"):
            return "formal_analysis_runtime"
        if relative.startswith("scripts/"):
            return "formal_operational_utility"
        if relative.startswith("legacy/phased_cora_pipeline/"):
            return "legacy_compatibility_runtime"
        return "formal_search_runtime"
    if relative.startswith("tests/"):
        return "maintained_test"
    if relative.startswith("docs/") or relative.endswith(".md"):
        return "documentation_or_instruction"
    if relative.endswith(".sh"):
        return "launch_script"
    return "supporting_source"


def _dependency_versions() -> dict[str, str | None]:
    distributions = (
        "torch",
        "torch-geometric",
        "numpy",
        "scipy",
        "scikit-learn",
        "gpytorch",
        "botorch",
        "ogb",
    )
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def generate_source_freeze_materials(
    *,
    repo_root: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    method_config: str | os.PathLike[str],
    previous_acceptance_artifact: str | os.PathLike[str],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"formal checkpoint is missing: {checkpoint}")
    config = cross_dataset_runner.load_and_validate_method_config(method_config)
    gate_paths = discover_source_gate_paths(
        root, SOURCE_GATE_DISCOVERY_RULES
    )
    all_paths = snapshot_source_paths(root, gate_paths)
    tracked, untracked = _git_file_sets(root)
    gate_set = set(gate_paths)
    rows: list[dict[str, Any]] = []
    for relative in all_paths:
        path = root / relative
        formal_required = relative in gate_set
        rows.append(
            {
                "path": relative,
                "tracked_status": "tracked" if relative in tracked else "untracked",
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "formal_run_required": formal_required,
                "functional_category": _category(relative, formal_required),
            }
        )
    source_id = source_id_from_rows(rows)
    checkpoint_row = {
        "path": str(checkpoint),
        "size_bytes": int(checkpoint.stat().st_size),
        "sha256": sha256_file(checkpoint),
    }
    previous = Path(previous_acceptance_artifact).resolve()
    if not previous.is_file():
        raise FileNotFoundError(
            f"previous acceptance artifact is missing: {previous}"
        )
    nvidia = _run(["nvidia-smi"], root=root, timeout=30.0)
    manifest = {
        "format_version": SOURCE_MANIFEST_FORMAT_VERSION,
        "source_id": source_id,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "repository_root": str(root),
        "branch": _run(["git", "branch", "--show-current"], root=root).strip(),
        "head_commit": _run(["git", "rev-parse", "HEAD"], root=root).strip(),
        "head_commit_oneline": _run(
            ["git", "log", "-1", "--oneline"], root=root
        ).strip(),
        "git_status_sb": _run(["git", "status", "-sb"], root=root),
        "git_diff_check": _run(["git", "diff", "--check"], root=root),
        "git_diff_stat": _run(["git", "diff", "--stat"], root=root),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "dependencies": _dependency_versions(),
        },
        "cuda": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
            "cuda_available": bool(torch.cuda.is_available()),
            "nvidia_smi": nvidia,
        },
        "previous_acceptance_artifact": {
            "path": str(previous),
            "size_bytes": int(previous.stat().st_size),
            "sha256": sha256_file(previous),
        },
        "formal_configuration": {
            "path": str(Path(method_config).resolve()),
            "fingerprint": canonical_json_fingerprint(config),
            "development_search_seeds": list(
                cross_dataset_runner.DEVELOPMENT_SEARCH_SEEDS
            ),
            "formal_search_seeds": list(
                cross_dataset_runner.FORMAL_SEARCH_SEEDS
            ),
        },
        "candidate_pool_fingerprints": formal_matrix.candidate_pool_fingerprints(
            config
        ),
        "candidate_pool_fingerprint_source": (
            "bo_phase4.make_lhs_pool using frozen method config, 768 points, "
            "and each formal search seed"
        ),
        "source_gate": {
            "discovery_rules": SOURCE_GATE_DISCOVERY_RULES,
            "excludes_results_logs_datasets_checkpoints": True,
        },
        "checkpoint": checkpoint_row,
        "files": rows,
    }
    source_json = output / "source_manifest.json"
    source_tsv = output / "source_manifest.tsv"
    source_sha = output / "source_manifest.sha256"
    checkpoint_json = output / "checkpoint_manifest.json"
    patch_path = output / "working_tree_tracked.patch"
    untracked_path = output / "untracked_source_files.txt"
    _atomic_json(source_json, manifest)
    headings = (
        "path\ttracked_status\tsize_bytes\tsha256\t"
        "formal_run_required\tfunctional_category\n"
    )
    lines = [
        "\t".join(
            [
                str(row["path"]),
                str(row["tracked_status"]),
                str(row["size_bytes"]),
                str(row["sha256"]),
                str(row["formal_run_required"]).lower(),
                str(row["functional_category"]),
            ]
        )
        for row in rows
    ]
    _atomic_text(source_tsv, headings + "\n".join(lines) + "\n")
    manifest_sha256 = sha256_file(source_json)
    _atomic_text(
        source_sha, f"{manifest_sha256}  {source_json.name}\n"
    )
    _atomic_json(
        checkpoint_json,
        {
            "format_version": 1,
            "source_id": source_id,
            "checkpoint": checkpoint_row,
        },
    )
    tracked_patch = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    _atomic_bytes(patch_path, tracked_patch)
    source_untracked = sorted(
        relative for relative in all_paths if relative in untracked
    )
    _atomic_text(
        untracked_path,
        "".join(f"{relative}\n" for relative in source_untracked),
    )
    verification = verify_frozen_source(
        source_json,
        repo_root=root,
        expected_source_id=source_id,
        checkpoint_path=checkpoint,
    )
    return {
        "status": "completed",
        "source_id": source_id,
        "source_manifest": str(source_json),
        "source_manifest_sha256": manifest_sha256,
        "source_manifest_tsv": str(source_tsv),
        "checkpoint_manifest": str(checkpoint_json),
        "working_tree_tracked_patch": str(patch_path),
        "untracked_source_files": str(untracked_path),
        "source_file_count": len(rows),
        "formal_run_required_file_count": len(gate_paths),
        "verification": verification,
    }


def create_source_snapshot(
    *,
    repo_root: str | os.PathLike[str],
    source_manifest_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    extra_artifacts: Sequence[str | os.PathLike[str]] = (),
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_path = Path(source_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_id = str(manifest["source_id"])
    archive = (
        Path(output_directory).resolve()
        / f"frozen_source_snapshot_{source_id}.tar.gz"
    )
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        for row in manifest["files"]:
            relative = str(row["path"])
            handle.add(root / relative, arcname=relative, recursive=False)
        handle.add(
            manifest_path,
            arcname=".source_freeze/source_manifest.json",
            recursive=False,
        )
        for raw_path in extra_artifacts:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"snapshot extra artifact is missing: {path}")
            handle.add(
                path,
                arcname=f".source_freeze/{path.name}",
                recursive=False,
            )
    digest = sha256_file(archive)
    digest_path = archive.with_name(archive.name + ".sha256")
    _atomic_text(digest_path, f"{digest}  {archive.name}\n")
    return {
        "status": "completed",
        "source_id": source_id,
        "archive": str(archive),
        "archive_size_bytes": int(archive.stat().st_size),
        "archive_sha256": digest,
        "archive_sha256_path": str(digest_path),
    }


def verify_restored_snapshot(
    *,
    restore_root: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    root = Path(restore_root).resolve()
    manifest_path = root / ".source_freeze" / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file():
            mismatches.append(f"missing:{row['path']}")
        elif int(path.stat().st_size) != int(row["size_bytes"]):
            mismatches.append(f"size:{row['path']}")
        elif sha256_file(path) != row["sha256"]:
            mismatches.append(f"sha256:{row['path']}")
    if mismatches:
        raise RuntimeError(
            "restored snapshot source mismatch: " + ", ".join(mismatches)
        )
    gate = verify_frozen_source(
        manifest_path,
        repo_root=root,
        expected_source_id=manifest["source_id"],
        checkpoint_path=checkpoint_path,
    )
    compiled = []
    for row in manifest["files"]:
        if bool(row["formal_run_required"]) and str(row["path"]).endswith(
            ".py"
        ):
            py_compile.compile(str(root / row["path"]), doraise=True)
            compiled.append(str(row["path"]))
    return {
        "status": "completed",
        "source_id": manifest["source_id"],
        "manifest_file_count": len(manifest["files"]),
        "mismatch_count": 0,
        "source_gate": gate,
        "compiled_python_file_count": len(compiled),
    }


def set_freeze_artifacts_read_only(paths: Iterable[str | os.PathLike[str]]) -> None:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"freeze artifact is missing: {path}")
        path.chmod(0o444)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-directory", required=True)
    prepare.add_argument("--checkpoint", required=True)
    prepare.add_argument(
        "--method-config", default=str(cross_dataset_runner.DEFAULT_METHOD_CONFIG)
    )
    prepare.add_argument("--previous-acceptance-artifact", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--source-manifest", required=True)
    snapshot.add_argument("--output-directory", required=True)
    snapshot.add_argument("--extra-artifact", action="append", default=[])
    verify = subparsers.add_parser("verify-restored")
    verify.add_argument("--restore-root", required=True)
    verify.add_argument("--checkpoint", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = generate_source_freeze_materials(
            repo_root=cross_dataset_runner.ROOT,
            output_directory=args.output_directory,
            checkpoint_path=args.checkpoint,
            method_config=args.method_config,
            previous_acceptance_artifact=args.previous_acceptance_artifact,
        )
    elif args.command == "snapshot":
        result = create_source_snapshot(
            repo_root=cross_dataset_runner.ROOT,
            source_manifest_path=args.source_manifest,
            output_directory=args.output_directory,
            extra_artifacts=args.extra_artifact,
        )
    else:
        result = verify_restored_snapshot(
            restore_root=args.restore_root,
            checkpoint_path=args.checkpoint,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
