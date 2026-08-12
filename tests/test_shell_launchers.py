from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "launcher",
    (
        "scripts/analysis/run_collect_results.sh",
        "scripts/analysis/run_diagnostics_suite.sh",
        "scripts/evaluation/run_final_eval_topk_seedfair.sh",
        "analyse/run_collect_results.sh",
        "analyse/run_diagnostics_suite.sh",
        "run_final_eval_topk_seedfair.sh",
    ),
)
def test_launcher_resolves_repository_root_from_other_cwd(
    launcher: str,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        ["bash", str(REPO_ROOT / launcher), "--print-repo-root"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == str(REPO_ROOT)


def test_collect_wrapper_forwards_arguments_and_uses_repository_cwd(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python with spaces"
    args_record = tmp_path / "args.txt"
    cwd_record = tmp_path / "cwd.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$PWD\" > \"$TRACE_CWD\"\n"
        "printf '%s\\n' \"$@\" > \"$TRACE_ARGS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(fake_python),
            "TRACE_ARGS": str(args_record),
            "TRACE_CWD": str(cwd_record),
        }
    )

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "analyse/run_collect_results.sh"),
            "--results_root",
            "results with spaces",
            "--output",
            "output with spaces",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )

    assert cwd_record.read_text(encoding="utf-8").strip() == str(REPO_ROOT)
    assert args_record.read_text(encoding="utf-8").splitlines() == [
        str(REPO_ROOT / "analyse/collect_experiment_results.py"),
        "--results_root",
        "results with spaces",
        "--output",
        "output with spaces",
    ]


def test_diagnostics_wrapper_dry_run_is_cwd_independent_and_does_not_write(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python with spaces"
    fake_python.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    fake_python.chmod(0o755)
    empty_results = tmp_path / "empty results"
    bundle = tmp_path / "bundle output"
    diagnostics = tmp_path / "diagnostic output"
    empty_results.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "RESULTS_ROOT": str(empty_results),
            "BUNDLE_DIR": str(bundle),
            "DIAG_DIR": str(diagnostics),
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "analyse/run_diagnostics_suite.sh"),
            "--dry-run",
            "--python",
            str(fake_python),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"REPO_ROOT  : {REPO_ROOT}" in completed.stdout
    assert str(REPO_ROOT / "analyse/collect_experiment_results.py") in completed.stdout
    assert not bundle.exists()
    assert not diagnostics.exists()


def test_final_eval_compatibility_wrapper_help_is_safe_from_other_cwd(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        ["bash", str(REPO_ROOT / "run_final_eval_topk_seedfair.sh"), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--final-base-seed" in completed.stdout
    assert "Without --train" in completed.stdout
    assert not list(tmp_path.iterdir())
