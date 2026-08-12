from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = REPO_ROOT / "legacy/phased_cora_pipeline"
ENTRYPOINTS = (
    "generate_mini_data.py",
    "train_joint.py",
    "eval_joint.py",
    "bo_phase2.py",
    "bo_phase3.py",
    "bo_phase4_tpe.py",
)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_phased_cora_compatibility_help_from_other_cwd(
    entrypoint: str,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / entrypoint), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_phased_cora_wrapper_targets_one_archived_implementation(
    entrypoint: str,
) -> None:
    wrapper = (REPO_ROOT / entrypoint).read_text(encoding="utf-8")

    assert f'legacy/phased_cora_pipeline/{entrypoint}' in wrapper
    assert "runpy.run_path" in wrapper
    assert (LEGACY_ROOT / entrypoint).is_file()
