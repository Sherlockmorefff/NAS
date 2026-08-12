"""Run one isolated, reduced-budget GPU smoke of a formal method path.

The formal cross-dataset configuration remains the source of all semantic
parameters.  Only the explicitly recorded smoke budgets below are reduced.
This runner never creates a formal run tag and refuses to reuse a non-empty
output unless ``--resume-initialization`` is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cross_dataset_runner import (
    DEFAULT_METHOD_CONFIG,
    build_search_command,
    load_and_validate_method_config,
)
from dataset_utils import canonicalize_dataset_name
from surrogate.checkpoint_io import atomic_json_dump


SMOKE_OVERRIDES: dict[str, Any] = {
    "n_lhs_candidates": 64,
    "eval_epochs": 3,
    "patience": 2,
    "num_restarts": 2,
    "raw_samples": 16,
    "n_extra": 8,
    "gp_refit_every": 1,
    "gp_refit_steps": 2,
    "gp_save_every": 1,
    "probe_pool_size": 32,
}

METHOD_SMOKE_BUDGETS: dict[str, dict[str, Any]] = {
    "S0": {
        "n_init": 4,
        "initial_seed_evals": 4,
        "initial_expand_evals": 0,
        "n_iter": 1,
        "max_total_full_evals": 5,
    },
    "G100": {
        "n_init": 4,
        "initial_seed_evals": 4,
        "initial_expand_evals": 1,
        "initial_shortlist_evals": 8,
        "n_iter": 1,
        "max_total_full_evals": 6,
        "ted_shortlist_per_cluster": 2,
        "low_fidelity_epochs": 1,
        "low_fidelity_patience": 0,
    },
    "G150": {
        "n_init": 4,
        "initial_seed_evals": 4,
        "initial_expand_evals": 1,
        "initial_shortlist_evals": 8,
        "n_iter": 1,
        "max_total_full_evals": 6,
        "ted_shortlist_per_cluster": 2,
        "low_fidelity_epochs": 1,
        "low_fidelity_patience": 0,
    },
}


def _set_option(command: list[str], name: str, value: Any) -> None:
    option = f"--{name}"
    if option in command:
        index = command.index(option)
        command[index + 1] = str(value)
    else:
        command.extend([option, str(value)])


def build_smoke_command(
    *,
    python_executable: str,
    checkpoint: str,
    data_root: str,
    dataset: str,
    method: str,
    search_seed: int,
    expected_dataset_manifest: str,
    output: Path,
    log_dir: Path,
    method_config: str,
    resume_initialization: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    config = load_and_validate_method_config(method_config)
    canonical = canonicalize_dataset_name(dataset)
    key = method.upper()
    command, _, _ = build_search_command(
        python_executable=python_executable,
        checkpoint=checkpoint,
        data_root=data_root,
        run_tag="cross_dataset_v1_preflight_builder_only",
        dataset=canonical,
        method_key=key,
        search_seed=int(search_seed),
        config=config,
        expected_dataset_manifest=expected_dataset_manifest,
        search_seed_role="development",
    )
    version = f"cross_dataset_v1_preflight_{canonical}_{key.lower()}_seed{int(search_seed)}"
    recorded_overrides = {
        **SMOKE_OVERRIDES,
        **METHOD_SMOKE_BUDGETS[key],
    }
    for name, value in recorded_overrides.items():
        _set_option(command, name, value)
    _set_option(command, "output", output.resolve())
    _set_option(command, "log_dir", log_dir.resolve())
    _set_option(command, "version", version)
    _set_option(command, "method_label", f"preflight_smoke_{key.lower()}")
    if resume_initialization:
        if key == "S0":
            raise ValueError("S0 has no clustered initialization to resume")
        command.append("--resume_initialization")
    metadata = {
        "dataset": canonical,
        "method": key,
        "search_seed": int(search_seed),
        "formal_method_config": str(Path(method_config).resolve()),
        "formal_method_config_sha256": hashlib.sha256(
            Path(method_config).read_bytes()
        ).hexdigest(),
        "formal_full_budget_unchanged": {
            name: int(config["methods"][key][name])
            for name in (
                "initial_seed_evals",
                "initial_expand_evals",
                "n_iter",
                "max_total_full_evals",
            )
        },
        "smoke_overrides": recorded_overrides,
        "resume_initialization": bool(resume_initialization),
        "expected_dataset_manifest": str(
            Path(expected_dataset_manifest).resolve()
        ),
        "output": str(output.resolve()),
        "log_dir": str(log_dir.resolve()),
        "command": command,
    }
    return command, metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one reduced-budget cross-dataset method-path smoke"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, choices=("S0", "G100", "G150"))
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expected-dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--method-config", default=str(DEFAULT_METHOD_CONFIG)
    )
    parser.add_argument("--resume-initialization", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    log_dir = Path(args.log_dir)
    if args.resume_initialization:
        if not (output / "initialization_config.json").is_file():
            raise FileNotFoundError(
                "resume requires an existing initialization_config.json in output"
            )
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty smoke output: {output}")
    if not args.resume_initialization and log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty smoke log directory: {log_dir}")
    output.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    command, metadata = build_smoke_command(
        python_executable=args.python,
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        dataset=args.dataset,
        method=args.method,
        search_seed=args.search_seed,
        expected_dataset_manifest=args.expected_dataset_manifest,
        output=output,
        log_dir=log_dir,
        method_config=args.method_config,
        resume_initialization=args.resume_initialization,
    )
    invocation_name = (
        "method_path_resume_invocation.json"
        if args.resume_initialization
        else "method_path_initial_invocation.json"
    )
    invocation_path = output / invocation_name
    atomic_json_dump({**metadata, "status": "running"}, invocation_path)
    started = time.monotonic()
    try:
        completed = subprocess.run(command, check=False)
        exit_code = int(completed.returncode)
    except Exception as exc:
        payload = {
            **metadata,
            "status": "launcher_failed",
            "wall_seconds": float(time.monotonic() - started),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_json_dump(payload, invocation_path)
        raise
    payload = {
        **metadata,
        "status": "passed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "wall_seconds": float(time.monotonic() - started),
        "history_final_exists": (output / "history_final.json").is_file(),
        "run_summary_exists": (output / "run_summary.json").is_file(),
        "initialization_config_exists": (
            output / "initialization_config.json"
        ).is_file(),
        "low_fidelity_history_exists": (
            output / "low_fidelity_history.json"
        ).is_file(),
    }
    atomic_json_dump(payload, invocation_path)
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, command)
    if not payload["history_final_exists"] or not payload["run_summary_exists"]:
        raise RuntimeError("method smoke exited zero without final history/summary")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
