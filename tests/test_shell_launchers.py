from __future__ import annotations

import os
import json
from pathlib import Path
import shlex
import subprocess
import sys

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


def _formal_matrix(tmp_path: Path, protocol: str) -> Path:
    tasks = []
    for dataset in ("citeseer", "pubmed", "dblp", "flickr"):
        for method in ("S0", "G100", "G150"):
            method_dir = {"S0": "s0", "G100": "g100", "G150": "g150"}[method]
            for seed in range(5, 10):
                output = (
                    REPO_ROOT
                    / "results"
                    / "search"
                    / protocol
                    / dataset
                    / method_dir
                    / f"search_seed{seed}"
                )
                tasks.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "search_seed": seed,
                        "output_directory": str(output),
                    }
                )
    path = tmp_path / "formal_matrix.json"
    path.write_text(
        json.dumps({"protocol_id": protocol, "tasks": tasks}),
        encoding="utf-8",
    )
    return path


def _fake_final_eval_python(tmp_path: Path) -> Path:
    path = tmp_path / "fake-python"
    real_python = shlex.quote(sys.executable)
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        "if [[ \"${1:-}\" == \"-c\" ]]; then exit 0; fi\n"
        f"if [[ \"${{1:-}}\" == \"-\" ]]; then exec {real_python} \"$@\"; fi\n"
        "printf '%s\\n' \"$*\" >> \"$TRACE\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_final_eval_formal_launcher_reads_search_seeds_and_history_paths_from_matrix(
    tmp_path: Path,
) -> None:
    protocol = "accuracy-diag_v1_20260813_bc2112d5"
    matrix = _formal_matrix(tmp_path, protocol)
    fake_python = _fake_final_eval_python(tmp_path)
    trace = tmp_path / "trace.txt"
    run_dir = tmp_path / "final"
    env = os.environ.copy()
    env.update({"TRACE": str(trace), "LOG_DIR": str(tmp_path / "logs")})
    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/evaluation/run_final_eval_topk_seedfair.sh"),
            "--protocol-id",
            protocol,
            "--search-protocol-id",
            protocol,
            "--formal-matrix",
            str(matrix),
            "--search-seeds",
            "5,6,7,8,9",
            "--dataset",
            "citeseer",
            "--final-base-seed",
            "0",
            "--run-dir",
            str(run_dir),
            "--python",
            str(fake_python),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "All 15 history preflights passed." in completed.stdout
    rows = trace.read_text(encoding="utf-8").splitlines()
    final_rows = [row for row in rows if "final_eval.py" in row]
    assert len(final_rows) == 15
    assert {int(row.split("--search_seed ", 1)[1].split()[0]) for row in final_rows} == set(range(5, 10))
    assert all("results/search/" in row for row in final_rows)
    assert all("--n_replicates 30" in row for row in final_rows)


def test_final_eval_formal_launcher_rejects_explicit_seed_mismatch(
    tmp_path: Path,
) -> None:
    protocol = "accuracy-diag_v1_20260813_bc2112d5"
    matrix = _formal_matrix(tmp_path, protocol)
    fake_python = _fake_final_eval_python(tmp_path)
    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/evaluation/run_final_eval_topk_seedfair.sh"),
            "--protocol-id",
            protocol,
            "--search-protocol-id",
            protocol,
            "--formal-matrix",
            str(matrix),
            "--search-seeds",
            "0,1,2,3,4",
            "--dataset",
            "citeseer",
            "--final-base-seed",
            "0",
            "--run-dir",
            str(tmp_path / "final"),
            "--python",
            str(fake_python),
        ],
        cwd=tmp_path,
        env={**os.environ, "TRACE": str(tmp_path / "trace.txt")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "does not match formal matrix seeds" in completed.stderr
    assert not (tmp_path / "final").exists()


def test_final_eval_legacy_mode_keeps_search_seeds_zero_through_four(
    tmp_path: Path,
) -> None:
    fake_python = _fake_final_eval_python(tmp_path)
    trace = tmp_path / "trace.txt"
    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/evaluation/run_final_eval_topk_seedfair.sh"),
            "--dataset",
            "cora",
            "--final-base-seed",
            "0",
            "--run-dir",
            str(tmp_path / "legacy-final"),
            "--python",
            str(fake_python),
        ],
        cwd=tmp_path,
        env={**os.environ, "TRACE": str(trace)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "All 15 history preflights passed." in completed.stdout
    rows = [row for row in trace.read_text(encoding="utf-8").splitlines() if "final_eval.py" in row]
    assert len(rows) == 15
    assert {int(row.split("--search_seed ", 1)[1].split()[0]) for row in rows} == set(range(5))
    assert all("--n_replicates 10" in row for row in rows)
