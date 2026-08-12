"""Build or execute one isolated cross-dataset Phase4 search command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

from deterministic_runtime import prepare_deterministic_environment

prepare_deterministic_environment()

from dataset_utils import canonicalize_dataset_name
from source_verification import canonical_json_sha256, verify_frozen_source


ROOT = Path(__file__).resolve().parent
DEFAULT_METHOD_CONFIG = ROOT / "configs" / "cross_dataset_methods.json"
DEFAULT_MANIFEST_ROOT = (
    ROOT / "results" / "cross_dataset_v1_preflight" / "manifests"
)
DEFAULT_TRAINING_MODE_DECISIONS = (
    ROOT
    / "results"
    / "cross_dataset_v1_preflight"
    / "training_mode_decisions.json"
)
METHOD_KEYS = ("S0", "G100", "G150")
FORMAL_DATASETS = ("citeseer", "pubmed", "dblp", "flickr")
DEVELOPMENT_SEARCH_SEEDS = (0, 1, 2, 3, 4)
FORMAL_SEARCH_SEEDS = (5, 6, 7, 8, 9)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_and_validate_method_config(
    path: str | os.PathLike[str] = DEFAULT_METHOD_CONFIG,
) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported cross-dataset method config format")
    if tuple(payload.get("formal_datasets", ())) != FORMAL_DATASETS:
        raise ValueError(
            "formal_datasets must be exactly " + repr(FORMAL_DATASETS)
        )
    common = payload.get("common")
    methods = payload.get("methods")
    if not isinstance(common, dict) or not isinstance(methods, dict):
        raise ValueError("method config must contain common and methods objects")
    if set(methods) != set(METHOD_KEYS):
        raise ValueError(f"method config must define exactly {METHOD_KEYS}")
    for key, method in methods.items():
        seed_full = int(method["initial_seed_evals"])
        expand_full = int(method["initial_expand_evals"])
        online_full = int(method["n_iter"])
        total = seed_full + expand_full + online_full
        if total != 300:
            raise ValueError(
                f"{key} requests {total} full evaluations; exactly 300 required"
            )
        if int(method["max_total_full_evals"]) != 300:
            raise ValueError(f"{key} max_total_full_evals must equal 300")
        if int(method["planned_unique_full_evals"]) != 300:
            raise ValueError(f"{key} planned_unique_full_evals must equal 300")
    for key in ("G100", "G150"):
        method = methods[key]
        if method["initial_selection_strategy"] != "wgmm_ted_lowfid":
            raise ValueError(f"{key} must use wgmm_ted_lowfid")
        if method["gmm"]["wgmm_source"] != "gmm_fit_pool":
            raise ValueError(f"{key} must use gmm_fit_pool")
    for section in ("gmm", "ted", "low_fidelity", "promotion"):
        if methods["G100"][section] != methods["G150"][section]:
            raise ValueError(
                f"G100/G150 must share identical {section} semantics"
            )
    if common.get("adaptive_sampling") is not False:
        raise ValueError("formal cross-dataset runs must use a fixed online budget")
    if common.get("use_conditional_kernel") is not False:
        raise ValueError("recovered formal runs used the unmasked Exact-GP kernel")
    development_seeds = tuple(common.get("development_search_seeds", ()))
    legacy_development_seeds = tuple(common.get("search_seeds", ()))
    formal_seeds = tuple(common.get("formal_search_seeds", ()))
    if development_seeds != DEVELOPMENT_SEARCH_SEEDS:
        raise ValueError(
            "development_search_seeds must remain exactly "
            f"{DEVELOPMENT_SEARCH_SEEDS}"
        )
    if legacy_development_seeds != DEVELOPMENT_SEARCH_SEEDS:
        raise ValueError(
            "legacy search_seeds must remain the development seeds "
            f"{DEVELOPMENT_SEARCH_SEEDS}"
        )
    if formal_seeds != FORMAL_SEARCH_SEEDS:
        raise ValueError(
            "formal_search_seeds must be exactly " + repr(FORMAL_SEARCH_SEEDS)
        )
    if len(set(formal_seeds)) != len(formal_seeds):
        raise ValueError("formal_search_seeds contains duplicates")
    if set(formal_seeds) & set(development_seeds):
        raise ValueError("formal and development search seeds must be disjoint")
    return payload


def validate_training_mode_decision(
    path: str | os.PathLike[str],
    dataset: str,
    *,
    search_seed: int | None = None,
) -> dict[str, Any]:
    decision_path = Path(path)
    if not decision_path.is_file():
        raise FileNotFoundError(
            f"training-mode decision artifact not found: {decision_path}"
        )
    with decision_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    canonical = canonicalize_dataset_name(dataset)
    row = payload.get("datasets", {}).get(canonical)
    if not isinstance(row, dict):
        raise ValueError(
            f"training-mode decision missing dataset {canonical!r}"
        )
    if row.get("status") != "frozen" or row.get("training_mode") != "full_batch":
        raise RuntimeError(
            f"dataset {canonical} training mode is not frozen: "
            f"status={row.get('status')!r} "
            f"training_mode={row.get('training_mode')!r} "
            f"reason={row.get('reason')!r}"
        )
    if payload.get("format_version") == 3 and canonical == "flickr":
        continuous = row.get("continuous_extreme_resource_preflight")
        if not isinstance(continuous, dict) or continuous.get("status") != "completed":
            raise RuntimeError(
                "Flickr formal launch requires a completed continuous extreme "
                "resource preflight"
            )
    formal_seeds = payload.get("formal_matrix", {}).get("search_seeds")
    if search_seed is not None and isinstance(formal_seeds, list):
        if int(search_seed) not in {int(seed) for seed in formal_seeds}:
            raise RuntimeError(
                f"training-mode artifact does not authorize formal "
                f"search_seed={int(search_seed)}; formal seeds are {formal_seeds}"
            )
    else:
        validated_seeds = row.get("validated_search_seeds")
        if search_seed is not None and isinstance(validated_seeds, list):
            if int(search_seed) not in {int(seed) for seed in validated_seeds}:
                raise RuntimeError(
                    f"dataset {canonical} full_batch was not resource-validated "
                    f"for search_seed={int(search_seed)}; validated seeds are "
                    f"{validated_seeds}"
                )
    return row


def isolated_run_paths(
    run_tag: str,
    dataset: str,
    method_label: str,
    search_seed: int,
    *,
    repo_root: str | os.PathLike[str] = ROOT,
    artifact_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    if not _SAFE_COMPONENT.fullmatch(run_tag):
        raise ValueError(
            "run_tag must contain only letters, numbers, '.', '_' and '-'"
        )
    canonical = canonicalize_dataset_name(dataset)
    base = Path(repo_root) if artifact_root is None else Path(artifact_root).resolve()
    relative = Path(run_tag) / canonical / method_label / f"search_seed{int(search_seed)}"
    return base / "results" / relative, base / "logs" / relative


def _append_option(command: list[str], name: str, value: Any) -> None:
    command.extend([f"--{name}", str(value)])


def build_search_command(
    *,
    python_executable: str,
    checkpoint: str,
    data_root: str,
    run_tag: str,
    dataset: str,
    method_key: str,
    search_seed: int,
    config: dict[str, Any],
    expected_dataset_manifest: str | os.PathLike[str] | None = None,
    repo_root: str | os.PathLike[str] = ROOT,
    search_seed_role: str = "formal",
    artifact_root: str | os.PathLike[str] | None = None,
    formal_source_id: str | None = None,
    formal_config_fingerprint: str | None = None,
) -> tuple[list[str], Path, Path]:
    canonical = canonicalize_dataset_name(dataset)
    if canonical not in FORMAL_DATASETS:
        raise ValueError(
            f"dataset {canonical!r} is excluded from the formal matrix; "
            f"expected one of {FORMAL_DATASETS}"
        )
    key = method_key.upper()
    if key not in METHOD_KEYS:
        raise ValueError(f"method must be one of {METHOD_KEYS}")
    if search_seed_role == "formal":
        allowed_seeds = tuple(config["common"]["formal_search_seeds"])
    elif search_seed_role == "development":
        allowed_seeds = tuple(config["common"]["development_search_seeds"])
    else:
        raise ValueError(
            "search_seed_role must be exactly 'formal' or 'development'"
        )
    if int(search_seed) not in allowed_seeds:
        allowed = ",".join(str(seed) for seed in allowed_seeds)
        raise ValueError(
            f"{search_seed_role} search_seed must be one of {allowed}"
        )
    common = config["common"]
    method = config["methods"][key]
    output, log_dir = isolated_run_paths(
        run_tag,
        canonical,
        method["label"],
        int(search_seed),
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    version = (
        f"{run_tag}_{canonical}_{method['label']}_seed{int(search_seed)}"
    )
    command = [
        str(python_executable),
        str(Path(repo_root) / "bo_phase4.py"),
        "--checkpoint",
        os.path.abspath(os.path.expanduser(checkpoint)),
        "--dataset",
        canonical,
        "--data_root",
        os.path.abspath(os.path.expanduser(data_root)),
        "--output",
        str(output),
        "--log_dir",
        str(log_dir),
        "--version",
        version,
        "--method_label",
        method["label"],
        "--require_cuda",
    ]
    if formal_source_id is not None:
        _append_option(command, "formal_source_id", formal_source_id)
    if formal_config_fingerprint is not None:
        _append_option(
            command,
            "formal_config_fingerprint",
            formal_config_fingerprint,
        )
    if expected_dataset_manifest is not None:
        command.extend(
            [
                "--expected_dataset_manifest",
                os.path.abspath(os.path.expanduser(expected_dataset_manifest)),
            ]
        )
    for name in (
        "hp_mode",
        "n_lhs_candidates",
        "sigma_arch",
        "z_bound",
        "log_lr_min",
        "log_lr_max",
        "dropout_min",
        "dropout_max",
        "warm_start",
        "warm_start_repeats",
        "warm_start_noise",
        "eval_epochs",
        "patience",
        "gcnii_alpha",
        "gcnii_theta",
        "gp_init_mode",
        "surrogate_type",
        "scratch_gp_min_points",
        "online_candidate_strategy",
        "num_restarts",
        "raw_samples",
        "n_extra",
        "max_online_proposal_attempts",
        "gp_update_mode",
        "gp_refit_every",
        "gp_refit_steps",
        "gp_save_every",
        "min_bo_samples",
        "max_bo_samples",
        "convergence_check_every",
        "convergence_patience",
        "prequential_window",
        "mae_relative_tol",
        "mae_absolute_tol",
        "std_relative_tol",
        "spearman_tol",
        "degradation_tolerance",
        "best_acc_patience",
        "best_acc_min_delta",
        "max_wall_time_hours",
        "probe_pool_size",
    ):
        _append_option(command, name, common[name])
    for name in (
        "initial_selection_strategy",
        "n_init",
        "initial_seed_evals",
        "initial_expand_evals",
        "n_iter",
        "max_total_full_evals",
    ):
        _append_option(command, name, method[name])
    _append_option(command, "seed", int(search_seed))
    if canonical == "dblp":
        _append_option(command, "split_seed", 0)
    if canonical == "ogbn-arxiv":
        _append_option(
            command,
            "ogbn_arxiv_edge_mode",
            common["ogbn_arxiv_edge_mode"],
        )
    if method["gmm"] is not None:
        for name, value in method["gmm"].items():
            _append_option(command, name, value)
        for name, value in method["ted"].items():
            if name in ("ted_kernel", "ted_feature_transform"):
                continue
            _append_option(command, name, value)
        _append_option(
            command, "low_fidelity_epochs", method["low_fidelity"]["epochs"]
        )
        _append_option(
            command,
            "low_fidelity_patience",
            method["low_fidelity"]["patience"],
        )
        for name, value in method["promotion"].items():
            _append_option(command, name, value)
    return command, output, log_dir


def validate_phase4_command(command: Sequence[str]) -> argparse.Namespace:
    """Parse a built command through the real Phase4 parser without training."""

    if len(command) < 2:
        raise ValueError("Phase4 command must contain Python and script argv")
    import bo_phase4

    previous = list(sys.argv)
    try:
        sys.argv = [str(command[1]), *[str(value) for value in command[2:]]]
        parsed = bo_phase4.parse_args()
    finally:
        sys.argv = previous
    bo_phase4.validate_frozen_init_configuration(parsed)
    bo_phase4.validate_two_stage_initialization_config(parsed)
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or execute one cross-dataset S0/G100/G150 search"
    )
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, choices=METHOD_KEYS)
    parser.add_argument("--search-seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="/tmp/gnn_datasets")
    parser.add_argument(
        "--manifest-root",
        default=str(DEFAULT_MANIFEST_ROOT),
        help="root containing <dataset>/dataset_manifest.json from preflight",
    )
    parser.add_argument(
        "--training-mode-decisions",
        default=str(DEFAULT_TRAINING_MODE_DECISIONS),
        help="preflight decision artifact that must freeze full_batch",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="optional isolated root containing results/ and logs/",
    )
    parser.add_argument(
        "--method-config", default=str(DEFAULT_METHOD_CONFIG)
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="execute only after all formal launch gates pass",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="run every launch gate and real parser validation without training",
    )
    parser.add_argument(
        "--expected-source-manifest",
        default=None,
        help="content-addressed frozen source manifest required for launch/dry-run",
    )
    parser.add_argument(
        "--frozen-source-id",
        default=None,
        help="expected source ID required for launch/dry-run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_and_validate_method_config(args.method_config)
    canonical = canonicalize_dataset_name(args.dataset)
    source_verification = None
    if args.execute or args.dry_run:
        if not args.expected_source_manifest or not args.frozen_source_id:
            raise ValueError(
                "--expected-source-manifest and --frozen-source-id are required "
                "for --execute and --dry-run"
            )
        source_verification = verify_frozen_source(
            args.expected_source_manifest,
            repo_root=ROOT,
            expected_source_id=args.frozen_source_id,
            checkpoint_path=args.checkpoint,
        )
    training_mode_decision = validate_training_mode_decision(
        args.training_mode_decisions,
        canonical,
        search_seed=int(args.search_seed),
    )
    expected_manifest = (
        Path(args.manifest_root) / canonical / "dataset_manifest.json"
    )
    if not expected_manifest.is_file():
        raise FileNotFoundError(
            f"preflight dataset manifest not found: {expected_manifest}"
        )
    command, output, log_dir = build_search_command(
        python_executable=args.python,
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        run_tag=args.run_tag,
        dataset=args.dataset,
        method_key=args.method,
        search_seed=args.search_seed,
        config=config,
        expected_dataset_manifest=expected_manifest,
        repo_root=ROOT,
        artifact_root=args.artifact_root,
        formal_source_id=args.frozen_source_id,
        formal_config_fingerprint=canonical_json_sha256(config),
    )
    parsed_phase4 = validate_phase4_command(command)
    for role, path in (("output", output), ("log", log_dir)):
        if path.exists() and (
            not path.is_dir() or any(path.iterdir())
        ):
            raise FileExistsError(
                f"refusing to start in non-empty {role} directory: {path}"
            )
    payload = {
        "method": args.method,
        "dataset": canonical,
        "expected_dataset_manifest": str(expected_manifest.resolve()),
        "training_mode_decision": training_mode_decision,
        "search_seed": int(args.search_seed),
        "output": str(output),
        "log_dir": str(log_dir),
        "command": command,
        "executed": bool(args.execute),
        "dry_run": bool(args.dry_run),
        "status": (
            "not_started" if args.execute else "completed" if args.dry_run else "not_started"
        ),
        "phase4_parser_validation": {
            "dataset": parsed_phase4.dataset,
            "seed": int(parsed_phase4.seed),
            "max_total_full_evals": int(parsed_phase4.max_total_full_evals),
        },
        "source_verification": source_verification,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not args.execute:
        return 0
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
