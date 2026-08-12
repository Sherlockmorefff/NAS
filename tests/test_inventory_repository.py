from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

from scripts.maintenance.inventory_repository import build_inventory, write_reports


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_inventory_classifies_metadata_without_reading_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    (repo / ".gitignore").write_text(
        "results/\nlogs/\ndata/\n*.pth\n", encoding="utf-8"
    )
    tracked = repo / "results" / "run-a" / "source_manifest.json"
    tracked.parent.mkdir(parents=True)
    tracked.write_text('{"format_version": 1}\n', encoding="utf-8")
    _run_git(repo, "add", ".gitignore")
    _run_git(repo, "add", "-f", "results/run-a/source_manifest.json")
    (repo / "results" / "run-a" / "history_final.json").write_text(
        "[]\n", encoding="utf-8"
    )
    diagnostic = repo / "results" / "diagnostics" / "summary.csv"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text("metric,value\n", encoding="utf-8")
    log = repo / "logs" / "run-a" / "train.log"
    log.parent.mkdir(parents=True)
    log.write_text("complete\n", encoding="utf-8")
    data = repo / "data" / "dataset.bin"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"data")
    (repo / "model.pth").write_bytes(b"checkpoint")

    payload = build_inventory(repo)
    entries = payload["entries"]
    types = {entry["type"] for entry in entries}

    assert {"search", "posthoc/diagnostics", "log", "data", "checkpoint"} <= types
    manifest = next(entry for entry in entries if entry["type"] == "manifest/provenance")
    assert manifest["git_status"] == "tracked"
    assert manifest["contains_reproducibility_state"] is True
    assert payload["summary"]["file_count"] == 6

    output = tmp_path / "reports"
    reports = write_reports(payload, output)
    assert json.loads(Path(reports["json"]).read_text())["format_version"] == 1
    with Path(reports["csv"]).open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(entries)
    assert "ignored does not mean disposable" in Path(reports["markdown"]).read_text()


def test_inventory_does_not_follow_external_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "large.bin").write_bytes(b"outside")
    results = repo / "results"
    results.mkdir()
    (results / "external").symlink_to(outside, target_is_directory=True)

    payload = build_inventory(repo)

    assert payload["summary"]["file_count"] == 1
    assert payload["summary"]["total_size_bytes"] != len(b"outside")
