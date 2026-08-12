from __future__ import annotations

from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from run_provenance import collect_run_provenance, persist_run_provenance


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _numpy_states_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _make_git_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "source.py"
    checkpoint = repo / "checkpoint.pt"
    staged = repo / "staged.txt"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint-v1")
    _git(repo, "init")
    _git(repo, "add", "source.py")
    _git(
        repo,
        "-c",
        "user.name=Codex Test",
        "-c",
        "user.email=codex@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    staged.write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    return repo, source, checkpoint


def test_collect_run_provenance_records_required_state_without_rng_consumption(
    tmp_path,
) -> None:
    repo, source, checkpoint = _make_git_repo(tmp_path)
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    payload = collect_run_provenance(
        repo_root=repo,
        checkpoint_path=checkpoint,
        argv=["bo_phase4.py", "--seed", "0"],
        seed_derivation_version="sha256_v1",
        source_paths=[source],
    )

    assert random.getstate() == python_state
    assert _numpy_states_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert payload["argv"] == ["bo_phase4.py", "--seed", "0"]
    assert payload["git"]["commit"]
    assert payload["git"]["dirty_worktree"] is True
    assert len(payload["git"]["staged_diff_sha256"]) == 64
    assert len(payload["git"]["unstaged_diff_sha256"]) == 64
    manifest = payload["runtime_critical_source_manifest"]
    assert manifest["files"][0]["path"] == "source.py"
    assert len(manifest["files"][0]["sha256"]) == 64
    assert len(manifest["manifest_sha256"]) == 64
    assert payload["checkpoint"]["path"] == str(checkpoint.resolve())
    assert len(payload["checkpoint"]["sha256"]) == 64
    assert payload["python"]["executable"]
    assert payload["dependency_versions"]["torch"] == torch.__version__
    assert payload["torch_backend"]["torch_version"] == torch.__version__
    assert payload["seed_derivation_version"] == "sha256_v1"


def test_provenance_records_missing_checkpoint_and_source_errors(tmp_path) -> None:
    repo, _source, _checkpoint = _make_git_repo(tmp_path)
    payload = collect_run_provenance(
        repo_root=repo,
        checkpoint_path=repo / "missing.pt",
        argv=["bo_phase4.py"],
        seed_derivation_version="sha256_v1",
        source_paths=["missing.py"],
    )
    assert payload["checkpoint"]["sha256"] is None
    assert "does not exist" in payload["checkpoint"]["error"]
    source_row = payload["runtime_critical_source_manifest"]["files"][0]
    assert source_row["sha256"] is None
    assert source_row["error"] == "missing_file"
    assert payload["errors"]


def test_persist_run_provenance_is_immutable_on_resume(tmp_path) -> None:
    repo, source, checkpoint = _make_git_repo(tmp_path)
    output = repo / "results" / "run"
    first, created = persist_run_provenance(
        output,
        repo_root=repo,
        checkpoint_path=checkpoint,
        argv=["bo_phase4.py", "--seed", "0"],
        seed_derivation_version="sha256_v1",
        source_paths=[source],
    )
    assert created is True
    checkpoint.write_bytes(b"checkpoint-v2")
    second, created = persist_run_provenance(
        output,
        repo_root=repo,
        checkpoint_path=checkpoint,
        argv=["bo_phase4.py", "--resume_initialization"],
        seed_derivation_version="sha256_v1",
        source_paths=[source],
    )
    assert created is False
    assert second == first
