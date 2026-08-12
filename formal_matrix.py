"""Build and audit the frozen 60-task cross-dataset formal matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any, Mapping, Sequence

from deterministic_runtime import prepare_deterministic_environment

prepare_deterministic_environment()

import torch

import bo_phase4
import cross_dataset_runner
from initialization_wgmm_ted import canonical_json_fingerprint, fingerprint_array
from source_verification import load_source_manifest, verify_frozen_source


FORMAT_VERSION = 1
STATUS_VALUES = ("completed", "failed", "blocked", "skipped", "not_started")
FORMAL_SHARD_SPECS = (
    {
        "shard_id": "gpu0_citeseer_pubmed",
        "worker_id": 0,
        "expected_gpu_index": 0,
        "datasets": ("citeseer", "pubmed"),
        "filename": "formal_shard_gpu0_citeseer_pubmed_30.json",
    },
    {
        "shard_id": "gpu1_dblp_flickr",
        "worker_id": 1,
        "expected_gpu_index": 1,
        "datasets": ("dblp", "flickr"),
        "filename": "formal_shard_gpu1_dblp_flickr_30.json",
    },
)


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_formal_training_mode_decisions(
    accepted_path: str | os.PathLike[str],
    *,
    flickr_continuous_preflight_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Preserve accepted evidence while authorizing only formal seeds 5--9."""

    with open(accepted_path, "r", encoding="utf-8") as handle:
        accepted = json.load(handle)
    if not isinstance(accepted, dict):
        raise ValueError("accepted training-mode decisions must be a JSON object")
    datasets = accepted.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("accepted training-mode decisions have no datasets object")
    output = json.loads(json.dumps(accepted))
    output["format_version"] = 3
    output["development_search_seeds"] = list(
        cross_dataset_runner.DEVELOPMENT_SEARCH_SEEDS
    )
    output["formal_matrix"] = {
        "datasets": list(cross_dataset_runner.FORMAL_DATASETS),
        "methods": list(cross_dataset_runner.METHOD_KEYS),
        "search_seeds": list(cross_dataset_runner.FORMAL_SEARCH_SEEDS),
        "planned_search_runs": 60,
    }
    output["seed_semantics"] = {
        "development_search_seeds": "preserved historical and development evidence",
        "formal_search_seeds": "new independent formal search runs",
        "final_evaluation_replicate_seeds": "unchanged independent semantics",
    }
    for dataset in cross_dataset_runner.FORMAL_DATASETS:
        row = datasets.get(dataset)
        if not isinstance(row, dict):
            raise ValueError(f"accepted decisions are missing {dataset!r}")
        if row.get("status") != "frozen" or row.get("training_mode") != "full_batch":
            raise ValueError(f"accepted {dataset!r} decision is not frozen full_batch")
    ogbn = datasets.get("ogbn-arxiv")
    if not isinstance(ogbn, dict) or ogbn.get("status") != (
        "blocked_resource_preflight_oom"
    ):
        raise ValueError("ogbn-arxiv must remain blocked_resource_preflight_oom")
    if flickr_continuous_preflight_path is not None:
        preflight_path = Path(flickr_continuous_preflight_path).resolve()
        with preflight_path.open("r", encoding="utf-8") as handle:
            preflight = json.load(handle)
        if (
            preflight.get("status") != "completed"
            or preflight.get("dataset") != "flickr"
            or preflight.get("oom") is not False
        ):
            raise ValueError(
                "Flickr continuous extreme preflight is not completed without OOM"
            )
        output["datasets"]["flickr"][
            "continuous_extreme_resource_preflight"
        ] = {
            "status": "completed",
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
        }
    return output


def candidate_pool_fingerprints(config: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    common = config["common"]
    for seed in common["formal_search_seeds"]:
        args = argparse.Namespace(
            hp_mode=common["hp_mode"],
            n_lhs_candidates=int(common["n_lhs_candidates"]),
            n_init=50,
            seed=int(seed),
            sigma_arch=float(common["sigma_arch"]),
            z_bound=float(common["z_bound"]),
        )
        pool = torch.stack(bo_phase4.make_lhs_pool(args)).numpy()
        output[str(int(seed))] = fingerprint_array(pool)
    return output


def _option(argv: Sequence[str], name: str) -> str:
    flag = f"--{name}"
    try:
        return str(argv[argv.index(flag) + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"built Phase4 command is missing {flag}") from exc


def build_formal_manifest(
    *,
    repo_root: str | os.PathLike[str],
    source_manifest_path: str | os.PathLike[str],
    frozen_source_id: str,
    checkpoint: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
    manifest_root: str | os.PathLike[str],
    training_mode_decisions: str | os.PathLike[str],
    method_config: str | os.PathLike[str],
    python_executable: str,
    protocol_id: str | None = None,
    run_tag: str | None = None,
    artifact_root: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root).resolve()
    config = cross_dataset_runner.load_and_validate_method_config(method_config)
    source_manifest = load_source_manifest(source_manifest_path)
    if source_manifest.get("source_id") != frozen_source_id:
        raise ValueError("requested frozen source ID does not match source manifest")
    source_check = verify_frozen_source(
        source_manifest_path,
        repo_root=root,
        expected_source_id=frozen_source_id,
        checkpoint_path=checkpoint,
    )
    pool_fingerprints = candidate_pool_fingerprints(config)
    config_fingerprint = canonical_json_fingerprint(config)
    tasks: list[dict[str, Any]] = []
    outputs: set[str] = set()
    logs: set[str] = set()
    task_ids: set[str] = set()
    for dataset in cross_dataset_runner.FORMAL_DATASETS:
        decision = cross_dataset_runner.validate_training_mode_decision(
            training_mode_decisions,
            dataset,
        )
        dataset_manifest = Path(manifest_root) / dataset / "dataset_manifest.json"
        if not dataset_manifest.is_file():
            raise FileNotFoundError(
                f"preflight dataset manifest not found: {dataset_manifest}"
            )
        for method_key in cross_dataset_runner.METHOD_KEYS:
            method = config["methods"][method_key]
            for search_seed in config["common"]["formal_search_seeds"]:
                phase4_argv, output, log_dir = (
                    cross_dataset_runner.build_search_command(
                        python_executable=python_executable,
                        checkpoint=str(checkpoint),
                        data_root=str(data_root),
                        run_tag=run_tag,
                        protocol_id=protocol_id,
                        dataset=dataset,
                        method_key=method_key,
                        search_seed=int(search_seed),
                        config=config,
                        expected_dataset_manifest=dataset_manifest,
                        repo_root=root,
                        artifact_root=artifact_root,
                        formal_source_id=frozen_source_id,
                        formal_config_fingerprint=config_fingerprint,
                    )
                )
                parsed = cross_dataset_runner.validate_phase4_command(phase4_argv)
                task_id = f"{dataset}__{method_key}__search_seed{int(search_seed)}"
                if task_id in task_ids:
                    raise ValueError(f"duplicate formal task_id: {task_id}")
                if str(output) in outputs or str(log_dir) in logs:
                    raise ValueError(f"duplicate formal output/log path for {task_id}")
                for role, path in (("output", output), ("log", log_dir)):
                    if path.exists():
                        raise FileExistsError(
                            f"formal {role} path already exists for {task_id}: {path}"
                        )
                identity_argv = (
                    ["--protocol-id", protocol_id]
                    if protocol_id is not None
                    else ["--run-tag", str(run_tag)]
                )
                wrapper_argv = [
                    str(python_executable),
                    str(root / "cross_dataset_runner.py"),
                    *identity_argv,
                    "--dataset",
                    dataset,
                    "--method",
                    method_key,
                    "--search-seed",
                    str(int(search_seed)),
                    "--checkpoint",
                    str(Path(checkpoint).resolve()),
                    "--data-root",
                    str(Path(data_root).resolve()),
                    "--manifest-root",
                    str(Path(manifest_root).resolve()),
                    "--training-mode-decisions",
                    str(Path(training_mode_decisions).resolve()),
                    "--python",
                    str(python_executable),
                    "--method-config",
                    str(Path(method_config).resolve()),
                    *(
                        []
                        if artifact_root is None
                        else ["--artifact-root", str(Path(artifact_root).resolve())]
                    ),
                    "--expected-source-manifest",
                    str(Path(source_manifest_path).resolve()),
                    "--frozen-source-id",
                    frozen_source_id,
                    "--execute",
                ]
                parsed_wrapper = cross_dataset_runner.parse_args(wrapper_argv[2:])
                if not parsed_wrapper.execute or parsed_wrapper.dry_run:
                    raise AssertionError("formal wrapper argv is not executable")
                task_ids.add(task_id)
                outputs.add(str(output))
                logs.add(str(log_dir))
                low_fidelity = method["low_fidelity"]
                tasks.append(
                    {
                        "task_id": task_id,
                        "dataset": dataset,
                        "method": method_key,
                        "method_label": method["label"],
                        "search_seed": int(search_seed),
                        "exact_argv": wrapper_argv,
                        "shell_command": shlex.join(wrapper_argv),
                        "dry_run_argv": [
                            *(wrapper_argv[:-1]),
                            "--dry-run",
                        ],
                        "checkpoint": source_check["checkpoint"],
                        "dataset_manifest": str(dataset_manifest.resolve()),
                        "dataset_training_mode": decision["training_mode"],
                        "output_directory": str(output),
                        "log_directory": str(log_dir),
                        "candidate_budget": int(method["planned_unique_full_evals"]),
                        "seed_full_count": int(method["initial_seed_evals"]),
                        "expansion_full_count": int(method["initial_expand_evals"]),
                        "online_full_count": int(method["n_iter"]),
                        "low_fidelity": low_fidelity,
                        "eval_epochs": int(parsed.eval_epochs),
                        "patience": int(parsed.patience),
                        "n_lhs_candidates": int(parsed.n_lhs_candidates),
                        "gp": {
                            "surrogate_type": parsed.surrogate_type,
                            "gp_init_mode": parsed.gp_init_mode,
                            "online_candidate_strategy": parsed.online_candidate_strategy,
                            "gp_update_mode": parsed.gp_update_mode,
                            "gp_refit_every": int(parsed.gp_refit_every),
                            "gp_refit_steps": int(parsed.gp_refit_steps),
                            "use_conditional_kernel": bool(
                                parsed.use_conditional_kernel
                            ),
                            "num_restarts": int(parsed.num_restarts),
                            "raw_samples": int(parsed.raw_samples),
                            "n_extra": int(parsed.n_extra),
                            "max_online_proposal_attempts": int(
                                parsed.max_online_proposal_attempts
                            ),
                        },
                        "configuration_fingerprint": config_fingerprint,
                        "candidate_pool_fingerprint": pool_fingerprints[
                            str(int(search_seed))
                        ],
                        "candidate_pool_fingerprint_source": (
                            "bo_phase4.make_lhs_pool with frozen config and search_seed"
                        ),
                        "expected_source_manifest": str(
                            Path(source_manifest_path).resolve()
                        ),
                        "expected_source_manifest_id": frozen_source_id,
                        "resume_strategy": {
                            "mode": "refuse_nonempty_output",
                            "automatic_history_reuse": False,
                            "description": (
                                "a formal task starts only in absent output/log paths; "
                                "resume requires a separately audited invocation"
                            ),
                        },
                        "status": "not_started",
                    }
                )

    manifest = {
        "format_version": FORMAT_VERSION,
        "scope": "formal_search_only_no_final_evaluation",
        "formal_search_started": False,
        "datasets": list(cross_dataset_runner.FORMAL_DATASETS),
        "methods": list(cross_dataset_runner.METHOD_KEYS),
        "development_search_seeds": list(
            cross_dataset_runner.DEVELOPMENT_SEARCH_SEEDS
        ),
        "formal_search_seeds": list(cross_dataset_runner.FORMAL_SEARCH_SEEDS),
        "task_count": len(tasks),
        "protocol_id": protocol_id,
        "legacy_run_tag": run_tag,
        "source_id": frozen_source_id,
        "configuration_fingerprint": config_fingerprint,
        "candidate_pool_fingerprints": pool_fingerprints,
        "tasks": tasks,
    }
    audit = audit_formal_manifest(manifest)
    return manifest, audit


def audit_formal_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("formal manifest tasks must be a list")
    task_ids = [str(row["task_id"]) for row in tasks]
    outputs = [str(row["output_directory"]) for row in tasks]
    logs = [str(row["log_directory"]) for row in tasks]
    seeds = [int(row["search_seed"]) for row in tasks]
    if len(tasks) != 60:
        raise ValueError(f"formal manifest must contain exactly 60 tasks, got {len(tasks)}")
    if len(set(task_ids)) != 60:
        raise ValueError("formal manifest task_id values are not unique")
    if len(set(outputs)) != 60 or len(set(logs)) != 60:
        raise ValueError("formal output/log directories are not unique")
    if set(seeds) != set(cross_dataset_runner.FORMAL_SEARCH_SEEDS):
        raise ValueError("formal manifest does not contain exactly seeds 5--9")
    if set(seeds) & set(cross_dataset_runner.DEVELOPMENT_SEARCH_SEEDS):
        raise ValueError("development seed leaked into formal manifest")
    dataset_counts = Counter(str(row["dataset"]) for row in tasks)
    method_counts = Counter(str(row["method"]) for row in tasks)
    seed_counts = Counter(int(row["search_seed"]) for row in tasks)
    combination_counts = Counter(
        (str(row["dataset"]), str(row["method"])) for row in tasks
    )
    if set(dataset_counts.values()) != {15}:
        raise ValueError(f"dataset task counts are invalid: {dict(dataset_counts)}")
    if set(method_counts.values()) != {20}:
        raise ValueError(f"method task counts are invalid: {dict(method_counts)}")
    if set(seed_counts.values()) != {12}:
        raise ValueError(f"seed task counts are invalid: {dict(seed_counts)}")
    if len(combination_counts) != 12 or set(combination_counts.values()) != {5}:
        raise ValueError("each dataset/method combination must contain five tasks")
    budget_fields_present = ["candidate_budget" in row for row in tasks]
    if any(budget_fields_present) and not all(budget_fields_present):
        raise ValueError("formal manifest candidate_budget fields are incomplete")
    total_full_evaluations = (
        sum(int(row["candidate_budget"]) for row in tasks)
        if all(budget_fields_present)
        else 300 * len(tasks)
    )
    if total_full_evaluations != 18_000:
        raise ValueError(
            "formal matrix must request exactly 18,000 full evaluations, got "
            f"{total_full_evaluations}"
        )
    for row in tasks:
        if "candidate_budget" in row and int(row["candidate_budget"]) != 300:
            raise ValueError(f"{row['task_id']} does not request 300 candidates")
        if "n_lhs_candidates" in row and int(row["n_lhs_candidates"]) != 768:
            raise ValueError(f"{row['task_id']} does not use the frozen pool of 768")
        argv = [str(value) for value in row.get("exact_argv", [])]
        if argv and ("--execute" not in argv or "--dry-run" in argv):
            raise ValueError(f"{row['task_id']} exact_argv is not executable")
    invalid_status = sorted(
        {
            str(row.get("status"))
            for row in tasks
            if row.get("status") not in STATUS_VALUES
        }
    )
    if invalid_status:
        raise ValueError(f"formal manifest has invalid status values: {invalid_status}")
    return {
        "status": "completed",
        "formal_search_started": False,
        "task_count": len(tasks),
        "unique_task_id_count": len(set(task_ids)),
        "unique_output_count": len(set(outputs)),
        "unique_log_count": len(set(logs)),
        "by_dataset": dict(sorted(dataset_counts.items())),
        "by_method": dict(sorted(method_counts.items())),
        "by_seed": {str(key): seed_counts[key] for key in sorted(seed_counts)},
        "by_dataset_method": {
            f"{dataset}__{method}": combination_counts[(dataset, method)]
            for dataset, method in sorted(combination_counts)
        },
        "development_seed_overlap": [],
        "ogbn_arxiv_included": False,
        "all_initial_status_not_started": all(
            row["status"] == "not_started" for row in tasks
        ),
        "total_requested_full_evaluations": total_full_evaluations,
    }


def shard_payload_sha256(shard: Mapping[str, Any]) -> str:
    """Hash the immutable shard payload without its embedded content hash."""

    payload = {key: value for key, value in shard.items() if key != "shard_payload_sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_formal_shards(
    parent_manifest: Mapping[str, Any],
    *,
    parent_matrix_sha256: str,
    gpu_assignments: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Derive two immutable, order-preserving dataset shards from the matrix."""

    audit_formal_manifest(dict(parent_manifest))
    if len(parent_matrix_sha256) != 64:
        raise ValueError("parent matrix SHA-256 must contain 64 hexadecimal characters")
    tasks = parent_manifest["tasks"]
    shards: dict[str, dict[str, Any]] = {}
    for spec in FORMAL_SHARD_SPECS:
        worker_id = int(spec["worker_id"])
        assignment = gpu_assignments.get(worker_id)
        if not isinstance(assignment, Mapping):
            raise ValueError(f"missing accepted GPU assignment for worker {worker_id}")
        expected_uuid = assignment.get("gpu_uuid")
        if not isinstance(expected_uuid, str) or not expected_uuid.startswith("GPU-"):
            raise ValueError(f"invalid accepted GPU UUID for worker {worker_id}")
        datasets = tuple(str(value) for value in spec["datasets"])
        selected = [dict(task) for task in tasks if str(task["dataset"]) in datasets]
        shard: dict[str, Any] = {
            "format_version": 1,
            "scope": "formal_search_dataset_shard",
            "shard_id": spec["shard_id"],
            "worker_id": worker_id,
            "expected_gpu_index": int(spec["expected_gpu_index"]),
            "expected_gpu_uuid": expected_uuid,
            "datasets": list(datasets),
            "task_count": len(selected),
            "parent_matrix_sha256": parent_matrix_sha256,
            "source_id": parent_manifest["source_id"],
            "configuration_fingerprint": parent_manifest["configuration_fingerprint"],
            "tasks": selected,
        }
        shard["shard_payload_sha256"] = shard_payload_sha256(shard)
        shards[str(spec["shard_id"])] = shard
    audit = audit_formal_shards(
        parent_manifest,
        shards,
        expected_parent_matrix_sha256=parent_matrix_sha256,
    )
    return shards, audit


def audit_formal_shard(
    parent_manifest: Mapping[str, Any],
    shard: Mapping[str, Any],
    *,
    expected_parent_matrix_sha256: str,
) -> dict[str, Any]:
    if shard.get("format_version") != 1 or shard.get("scope") != "formal_search_dataset_shard":
        raise ValueError("unsupported formal shard format or scope")
    if shard.get("parent_matrix_sha256") != expected_parent_matrix_sha256:
        raise ValueError("formal shard parent matrix SHA-256 mismatch")
    if shard.get("source_id") != parent_manifest.get("source_id"):
        raise ValueError("formal shard source ID mismatch")
    if shard.get("configuration_fingerprint") != parent_manifest.get("configuration_fingerprint"):
        raise ValueError("formal shard method config fingerprint mismatch")
    if shard.get("shard_payload_sha256") != shard_payload_sha256(shard):
        raise ValueError("formal shard payload SHA-256 mismatch")
    worker_id = int(shard.get("worker_id", -1))
    spec = next(
        (row for row in FORMAL_SHARD_SPECS if int(row["worker_id"]) == worker_id),
        None,
    )
    if spec is None or shard.get("shard_id") != spec["shard_id"]:
        raise ValueError("formal shard worker/shard identity mismatch")
    if int(shard.get("expected_gpu_index", -1)) != int(spec["expected_gpu_index"]):
        raise ValueError("formal shard expected GPU index mismatch")
    if tuple(shard.get("datasets", ())) != tuple(spec["datasets"]):
        raise ValueError("formal shard dataset assignment mismatch")
    expected_tasks = [
        task
        for task in parent_manifest["tasks"]
        if str(task["dataset"]) in set(spec["datasets"])
    ]
    actual_tasks = shard.get("tasks")
    if not isinstance(actual_tasks, list) or actual_tasks != expected_tasks:
        raise ValueError("formal shard tasks differ from the ordered parent task records")
    if int(shard.get("task_count", -1)) != 30 or len(actual_tasks) != 30:
        raise ValueError("formal shard must contain exactly 30 tasks")
    combinations = {
        (str(task["dataset"]), str(task["method"]), int(task["search_seed"]))
        for task in actual_tasks
    }
    expected_combinations = {
        (dataset, method, seed)
        for dataset in spec["datasets"]
        for method in cross_dataset_runner.METHOD_KEYS
        for seed in cross_dataset_runner.FORMAL_SEARCH_SEEDS
    }
    if combinations != expected_combinations or len(combinations) != 30:
        raise ValueError("formal shard dataset/method/seed Cartesian product is invalid")
    requested = sum(int(task.get("candidate_budget", 300)) for task in actual_tasks)
    if requested != 9_000 or any(int(task.get("candidate_budget", 300)) != 300 for task in actual_tasks):
        raise ValueError("formal shard must request 9,000 total and 300 per task")
    expected_uuid = shard.get("expected_gpu_uuid")
    if not isinstance(expected_uuid, str) or not expected_uuid.startswith("GPU-"):
        raise ValueError("formal shard expected GPU UUID is invalid")
    return {
        "status": "completed",
        "shard_id": shard["shard_id"],
        "worker_id": worker_id,
        "datasets": list(spec["datasets"]),
        "task_count": 30,
        "requested_full_evaluations": requested,
        "shard_payload_sha256": shard["shard_payload_sha256"],
    }


def audit_formal_shards(
    parent_manifest: Mapping[str, Any],
    shards: Mapping[str, Mapping[str, Any]],
    *,
    expected_parent_matrix_sha256: str,
) -> dict[str, Any]:
    if set(shards) != {str(spec["shard_id"]) for spec in FORMAL_SHARD_SPECS}:
        raise ValueError("formal shards must contain exactly the two fixed shard IDs")
    reports = [
        audit_formal_shard(
            parent_manifest,
            shards[str(spec["shard_id"])],
            expected_parent_matrix_sha256=expected_parent_matrix_sha256,
        )
        for spec in FORMAL_SHARD_SPECS
    ]
    task_id_sets = [
        {str(task["task_id"]) for task in shards[str(spec["shard_id"])]["tasks"]}
        for spec in FORMAL_SHARD_SPECS
    ]
    parent_ids = {str(task["task_id"]) for task in parent_manifest["tasks"]}
    intersection = task_id_sets[0] & task_id_sets[1]
    union = task_id_sets[0] | task_id_sets[1]
    if intersection:
        raise ValueError("formal shards overlap: " + ", ".join(sorted(intersection)))
    if union != parent_ids or len(union) != 60:
        raise ValueError("formal shard union does not equal the parent 60-task matrix")
    return {
        "status": "completed",
        "parent_matrix_sha256": expected_parent_matrix_sha256,
        "global_task_count": len(parent_ids),
        "shard_task_counts": {row["shard_id"]: row["task_count"] for row in reports},
        "intersection_task_count": 0,
        "union_task_count": len(union),
        "total_requested_full_evaluations": sum(
            row["requested_full_evaluations"] for row in reports
        ),
        "shards": reports,
    }


def write_formal_shards(
    shards: Mapping[str, Mapping[str, Any]],
    *,
    output_root: str | os.PathLike[str],
) -> dict[str, str]:
    root = Path(output_root)
    output: dict[str, str] = {}
    for spec in FORMAL_SHARD_SPECS:
        shard_id = str(spec["shard_id"])
        path = root / str(spec["filename"])
        _atomic_json(path, dict(shards[shard_id]))
        digest = sha256_file(path)
        _atomic_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
        output[shard_id] = digest
    return output


def write_formal_manifest(
    manifest: dict[str, Any],
    audit: dict[str, Any],
    *,
    output_json: str | os.PathLike[str],
    output_csv: str | os.PathLike[str],
    output_audit: str | os.PathLike[str],
) -> str:
    json_path = Path(output_json)
    csv_path = Path(output_csv)
    _atomic_json(json_path, manifest)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest["tasks"]:
        raise ValueError("cannot write an empty formal task CSV")
    fieldnames = list(manifest["tasks"][0])
    rows = []
    for task in manifest["tasks"]:
        rows.append(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in task.items()
            }
        )
    fd, temporary = tempfile.mkstemp(
        prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    _atomic_json(Path(output_audit), audit)
    digest = sha256_file(json_path)
    _atomic_text(
        json_path.with_suffix(".sha256"),
        f"{digest}  {json_path.name}\n",
    )
    return digest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--training-mode-decisions", required=True)
    parser.add_argument("--method-config", default=str(cross_dataset_runner.DEFAULT_METHOD_CONFIG))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-audit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, audit = build_formal_manifest(
        repo_root=cross_dataset_runner.ROOT,
        source_manifest_path=args.source_manifest,
        frozen_source_id=args.source_id,
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        manifest_root=args.manifest_root,
        training_mode_decisions=args.training_mode_decisions,
        method_config=args.method_config,
        python_executable=args.python,
        protocol_id=args.protocol_id,
        artifact_root=args.artifact_root,
    )
    digest = write_formal_manifest(
        manifest,
        audit,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_audit=args.output_audit,
    )
    print(json.dumps({**audit, "manifest_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
