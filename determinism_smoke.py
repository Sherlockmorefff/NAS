"""Fresh-process GPU determinism and heterogeneous-GPU calibration smoke tools."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from deterministic_runtime import prepare_deterministic_environment

prepare_deterministic_environment()

import deterministic_three_strategy_pipeline as pipeline


SMOKE_FIELDS = (
    "dataset",
    "candidate_fingerprint",
    "architecture",
    "hyperparameters",
    "search_seed",
    "evaluation_replicate",
    "training_seed",
    "decoder_seed",
    "validation_accuracy",
    "test_accuracy",
    "epochs_ran",
    "best_epoch",
    "trajectory_hash",
    "model_final_state_hash",
    "deterministic_provenance",
    "process_id",
    "gpu_name",
    "gpu_uuid",
    "driver_version",
    "source_method",
    "source_search_seed",
)


def _old_tasks(matrix_path: Path) -> dict[str, dict[str, Any]]:
    matrix = pipeline.read_json(matrix_path)
    return {str(row["task_id"]): dict(row) for row in matrix["tasks"]}


def _history_record(task: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
    path = Path(str(task["output_directory"])) / "history_final.json"
    for row in pipeline.read_json(path):
        if row.get("candidate_fingerprint") == fingerprint:
            return dict(row)
    raise ValueError(f"candidate {fingerprint} is absent from {path}")


def _read_reproducibility_rows(
    reproducibility_root: Path, name: str
) -> list[dict[str, str]]:
    with (reproducibility_root / name).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_smoke_plan(
    root: Path = pipeline.PIPELINE_ROOT,
    *,
    legacy_matrix: Path,
    legacy_repro_root: Path,
) -> dict[str, Any]:
    rows = _read_reproducibility_rows(
        legacy_repro_root, "reproducibility_pairs.csv"
    )
    outliers = _read_reproducibility_rows(
        legacy_repro_root, "reproducibility_outliers.csv"
    )
    ordinary = next(
        row
        for row in rows
        if row["dataset"] == "citeseer"
        and row["classification"] == "exact_validation_match"
    )
    pubmed_candidates = [
        row
        for row in outliers
        if row["dataset"] == "pubmed" and int(row["search_seed"]) == 7
    ]
    if not pubmed_candidates:
        pubmed_candidates = [
            row
            for row in rows
            if row["dataset"] == "pubmed" and int(row["search_seed"]) == 7
        ]
    pubmed = max(pubmed_candidates, key=lambda row: float(row["absolute_difference"]))
    excluded = {ordinary["candidate_fingerprint"], pubmed["candidate_fingerprint"]}
    repeated = next(
        row
        for row in rows
        if {row["method_a"], row["method_b"]} == {"G100", "G150"}
        and row["candidate_fingerprint"] not in excluded
    )
    selections = (
        ("ordinary_citeseer", ordinary, False),
        ("pubmed_seed7_max_old_outlier", pubmed, False),
        ("cross_method_repeated_candidate", repeated, True),
    )
    tasks = _old_tasks(legacy_matrix)
    candidates = []
    for role, row, repeat_both_sources in selections:
        source_rows = []
        for suffix in ("a", "b"):
            task_id = row[f"task_id_{suffix}"]
            task = tasks[task_id]
            history = _history_record(task, row["candidate_fingerprint"])
            source_rows.append(
                {
                    "method": task["method"],
                    "method_label": task["method_label"],
                    "search_seed": int(task["search_seed"]),
                    "task_id": task_id,
                    "history_path": str((Path(task["output_directory"]) / "history_final.json").resolve()),
                    "z_search": history["z_search"],
                    "candidate_fingerprint": history["candidate_fingerprint"],
                    "decoder_seed": int(history["decoder_seed"]),
                    "training_seed": int(history["candidate_eval_seed"]),
                }
            )
        if source_rows[0]["search_seed"] != source_rows[1]["search_seed"]:
            raise ValueError("cross-source smoke pair must preserve search seed")
        candidates.append(
            {
                "role": role,
                "dataset": row["dataset"],
                "candidate_fingerprint": row["candidate_fingerprint"],
                "old_absolute_difference": float(row["absolute_difference"]),
                "sources": source_rows,
                "repetitions_per_source": 3 if repeat_both_sources else None,
                "total_repetitions": 6 if repeat_both_sources else 3,
            }
        )
    identity = pipeline.current_source_identity(root)
    return {
        "format_version": 1,
        "source_id": identity["source_id"],
        "git_head": identity["git_head"],
        "dirty_diff_sha256": identity["dirty_diff_sha256"],
        "checkpoint": str(pipeline.CHECKPOINT.resolve()),
        "candidates": candidates,
    }


def _phase4_args(dataset: str, method: str, search_seed: int):
    import cross_dataset_runner

    config = cross_dataset_runner.load_and_validate_method_config(pipeline.METHOD_CONFIG)
    command, _, _ = cross_dataset_runner.build_search_command(
        python_executable=str(pipeline.PYTHON_BIN),
        checkpoint=str(pipeline.CHECKPOINT),
        data_root=str(pipeline.DATA_ROOT),
        run_tag="determinism_smoke",
        dataset=dataset,
        method_key=method,
        search_seed=search_seed,
        config=config,
        expected_dataset_manifest=pipeline.MANIFEST_ROOT / dataset / "dataset_manifest.json",
        repo_root=pipeline.REPO_ROOT,
        search_seed_role="formal",
    )
    return cross_dataset_runner.validate_phase4_command(command)


def run_search_smoke_once(spec_path: Path, output: Path) -> dict[str, Any]:
    spec = pipeline.read_json(spec_path)
    import torch
    import bo_phase4
    from dataset_utils import load_dataset_from_request, resolve_dataset_request
    from deterministic_runtime import assert_strict_torch_determinism, deterministic_provenance

    assert_strict_torch_determinism(torch)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; determinism smoke refuses CPU fallback")
    device = torch.device("cuda:0")
    runtime = deterministic_provenance(torch, device=device)
    args = _phase4_args(spec["dataset"], spec["source_method"], int(spec["source_search_seed"]))
    args.output = str(output.parent)
    request = resolve_dataset_request(
        spec["dataset"],
        str(pipeline.DATA_ROOT),
        split_seed=0 if spec["dataset"] == "dblp" else None,
    )
    bundle = load_dataset_from_request(request)
    args.dataset = request.canonical_name
    args.data_root = request.data_root
    args.split_seed = request.split_seed
    args.dataset_context = bundle.context
    args.deterministic_provenance = runtime
    bundle.to(device)
    vae = bo_phase4.load_vae(args, device, __import__("logging").getLogger("determinism_smoke"))
    z_tensor, result = bo_phase4.eval_candidate(
        vae,
        spec["z_search"],
        bundle.data,
        bundle.num_features,
        bundle.num_classes,
        args,
        device,
        step=0,
        evaluation_seed=int(spec["expected_training_seed"]),
        evaluation_stage="determinism_smoke",
        evaluation_fidelity="full",
        candidate_fingerprint=spec["candidate_fingerprint"],
    )
    if result.get("valid") is not True:
        raise RuntimeError("determinism smoke candidate training was invalid")
    for field in ("trajectory_hash", "model_final_state_hash"):
        if not pipeline._valid_sha256(result.get(field)):
            raise RuntimeError(f"determinism smoke result is missing {field}")
    row = {
        "dataset": spec["dataset"],
        "candidate_fingerprint": result["candidate_fingerprint"],
        "architecture": pipeline._architecture(result),
        "hyperparameters": pipeline._hyperparameters(result),
        "search_seed": int(spec["source_search_seed"]),
        "evaluation_replicate": int(spec["evaluation_replicate"]),
        "training_seed": int(result["candidate_eval_seed"]),
        "decoder_seed": int(result["decoder_seed"]),
        "validation_accuracy": float(result["val_acc"]),
        "test_accuracy": None,
        "epochs_ran": int(result["epochs_ran"]),
        "best_epoch": result["best_epoch"],
        "trajectory_hash": result["trajectory_hash"],
        "model_final_state_hash": result["model_final_state_hash"],
        "deterministic_provenance": runtime,
        "process_id": os.getpid(),
        "gpu_name": runtime["gpu_name"],
        "gpu_uuid": runtime["gpu_uuid"],
        "driver_version": runtime["driver_version"],
        "source_method": spec["source_method"],
        "source_search_seed": int(spec["source_search_seed"]),
        "canonical_z_search": z_tensor.tolist(),
    }
    pipeline.atomic_json(output, row)
    return row


def _gpu_inventory() -> list[dict[str, str]]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi unavailable: {completed.stderr.strip()}")
    rows = []
    for raw in completed.stdout.splitlines():
        fields = [value.strip() for value in raw.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"invalid nvidia-smi row: {raw!r}")
        rows.append(dict(zip(("index", "gpu_uuid", "gpu_name", "driver_version"), fields)))
    return rows


def _assert_gpu_idle(gpu_id: str) -> dict[str, str]:
    inventory = _gpu_inventory()
    selected = next((row for row in inventory if row["index"] == str(gpu_id)), None)
    if selected is None:
        raise RuntimeError(f"GPU index does not exist: {gpu_id}")
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot audit GPU processes: {completed.stderr.strip()}")
    busy = [line for line in completed.stdout.splitlines() if line.strip().startswith(selected["gpu_uuid"])]
    if busy:
        raise RuntimeError("GPU is occupied; smoke was not started: " + "; ".join(busy))
    return selected


def _smoke_specs(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for candidate in plan["candidates"]:
        sources = candidate["sources"]
        chosen_sources = (
            [sources[0], sources[1]]
            if candidate["role"] == "cross_method_repeated_candidate"
            else [sources[0]]
        )
        replicate = 0
        for source in chosen_sources:
            for _ in range(3):
                specs.append(
                    {
                        "role": candidate["role"],
                        "dataset": candidate["dataset"],
                        "candidate_fingerprint": candidate["candidate_fingerprint"],
                        "z_search": source["z_search"],
                        "source_method": source["method"],
                        "source_search_seed": source["search_seed"],
                        "expected_training_seed": source["training_seed"],
                        "expected_decoder_seed": source["decoder_seed"],
                        "evaluation_replicate": replicate,
                    }
                )
                replicate += 1
    return specs


def _audit_smoke_runs(plan: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in runs:
        groups.setdefault((str(row["dataset"]), str(row["candidate_fingerprint"])), []).append(row)
    candidate_reports = []
    overall = True
    exact_fields = (
        "candidate_fingerprint",
        "architecture",
        "hyperparameters",
        "training_seed",
        "decoder_seed",
        "validation_accuracy",
        "epochs_ran",
        "best_epoch",
        "trajectory_hash",
        "model_final_state_hash",
        "gpu_uuid",
    )
    for candidate in plan["candidates"]:
        key = (candidate["dataset"], candidate["candidate_fingerprint"])
        candidate_runs = groups.get(key, [])
        mismatches = []
        if len(candidate_runs) != int(candidate["total_repetitions"]):
            mismatches.append("repetition_count")
        for field in exact_fields:
            values = {pipeline.canonical_json_sha256(row.get(field)) for row in candidate_runs}
            if len(values) != 1:
                mismatches.append(field)
        if candidate["role"] == "cross_method_repeated_candidate":
            if set(row["source_method"] for row in candidate_runs) != {"G100", "G150"}:
                mismatches.append("cross_method_sources")
        passed = not mismatches
        overall = overall and passed
        candidate_reports.append(
            {
                "role": candidate["role"],
                "dataset": candidate["dataset"],
                "candidate_fingerprint": candidate["candidate_fingerprint"],
                "repetitions": len(candidate_runs),
                "source_methods": sorted({row["source_method"] for row in candidate_runs}),
                "bitwise_exact": passed,
                "mismatched_fields": mismatches,
            }
        )
    return {"status": "passed" if overall else "failed", "candidates": candidate_reports}


def run_strict_smoke(
    gpu_id: str,
    root: Path = pipeline.PIPELINE_ROOT,
    *,
    artifact_label: str | None = None,
    acceptance_name: str = "DETERMINISM_ACCEPTED",
    expected_gpu_uuid: str | None = None,
    worker_id: int | None = None,
    legacy_matrix: Path,
    legacy_repro_root: Path,
) -> dict[str, Any]:
    identity = pipeline.current_source_identity(root)
    suffix = "" if artifact_label is None else f"_{artifact_label}"
    smoke_directory = root / ("determinism_smoke" if artifact_label is None else f"determinism_smoke_{artifact_label}")
    report_path = root / f"determinism_smoke_report{suffix}.json"
    report_markdown_path = root / f"determinism_smoke_report{suffix}.md"
    runs_path = root / f"determinism_smoke_runs{suffix}.csv"
    try:
        gpu = _assert_gpu_idle(gpu_id)
        if expected_gpu_uuid is not None and gpu["gpu_uuid"] != expected_gpu_uuid:
            raise RuntimeError(
                f"GPU UUID mismatch: expected={expected_gpu_uuid} actual={gpu['gpu_uuid']}"
            )
    except Exception as exc:
        blocked = {
            "status": "blocked",
            "source_id": identity["source_id"],
            "git_head": identity["git_head"],
            "dirty_diff_sha256": identity["dirty_diff_sha256"],
            "reason": f"{type(exc).__name__}: {exc}",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        attempts = root / f"determinism_smoke_attempts{suffix}"
        name = datetime.now(timezone.utc).strftime("blocked_%Y%m%dT%H%M%S%fZ.json")
        pipeline.atomic_json(attempts / name, blocked)
        raise RuntimeError(blocked["reason"]) from exc
    plan = build_smoke_plan(
        root,
        legacy_matrix=legacy_matrix,
        legacy_repro_root=legacy_repro_root,
    )
    raw_root = smoke_directory / "raw"
    if raw_root.exists() and any(raw_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing smoke raw files: {raw_root}")
    pipeline.atomic_json(smoke_directory / "candidate_plan.json", plan)
    runs = []
    for index, spec in enumerate(_smoke_specs(plan)):
        spec_path = smoke_directory / "specs" / f"run_{index:03d}.json"
        output = raw_root / f"run_{index:03d}.json"
        pipeline.atomic_json(spec_path, spec)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["CUBLAS_WORKSPACE_CONFIG"] = prepare_deterministic_environment()
        env["PYTHONHASHSEED"] = "0"
        completed = subprocess.run(
            [str(pipeline.PYTHON_BIN), str(Path(__file__).resolve()), "search-once", "--spec", str(spec_path), "--output", str(output)],
            cwd=pipeline.REPO_ROOT,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            report = {
                "status": "failed",
                "source_id": identity["source_id"],
                "failed_run_index": index,
                "failed_spec": spec,
                "exit_code": completed.returncode,
            }
            pipeline.atomic_json(report_path, report)
            pipeline.atomic_text(report_markdown_path, f"# Determinism smoke failed\n\nRun {index} exited with {completed.returncode}. No acceptance marker was created.\n")
            raise RuntimeError(f"determinism smoke subprocess {index} failed")
        runs.append(pipeline.read_json(output))
    audit = _audit_smoke_runs(plan, runs)
    report = {
        **audit,
        "source_id": identity["source_id"],
        "git_head": identity["git_head"],
        "dirty_diff_sha256": identity["dirty_diff_sha256"],
        "gpu": gpu,
        "worker_id": worker_id,
        "run_count": len(runs),
        "smoke_runs_sha256": pipeline.canonical_json_sha256(runs),
        "accepted_at": datetime.now(timezone.utc).isoformat() if audit["status"] == "passed" else None,
    }
    pipeline.write_csv(runs_path, runs, SMOKE_FIELDS)
    pipeline.atomic_json(report_path, report)
    lines = ["# Strict GPU determinism smoke", "", f"Status: **{audit['status']}**", ""]
    for row in audit["candidates"]:
        lines.append(f"- {row['role']}: `{row['candidate_fingerprint']}` — repetitions={row['repetitions']}, bitwise_exact={str(row['bitwise_exact']).lower()}, sources={','.join(row['source_methods'])}")
    pipeline.atomic_text(report_markdown_path, "\n".join(lines) + "\n")
    if audit["status"] != "passed":
        raise RuntimeError("strict determinism smoke did not pass bitwise audit")
    output_hash = pipeline.canonical_json_sha256(
        {
            "runs": pipeline.sha256_file(runs_path),
            "report": pipeline.sha256_file(report_path),
            "markdown": pipeline.sha256_file(report_markdown_path),
        }
    )
    runtime = runs[0]["deterministic_provenance"]
    pipeline.write_gate(
        root / acceptance_name,
        source_id=identity["source_id"],
        input_sha256=pipeline.canonical_json_sha256(plan),
        output_sha256=output_hash,
        details={
            "git_head": identity["git_head"],
            "dirty_diff_sha256": identity["dirty_diff_sha256"],
            "software_hardware": runtime,
            "gpu_uuid": gpu["gpu_uuid"],
            "gpu_index": int(gpu["index"]),
            "worker_id": worker_id,
            "artifact_label": artifact_label,
            "runs_path": str(runs_path.resolve()),
            "report_path": str(report_path.resolve()),
            "candidates": audit["candidates"],
            "smoke_run_count": len(runs),
            "smoke_runs_sha256": pipeline.sha256_file(runs_path),
            "accepted_at": report["accepted_at"],
        },
    )
    return report


def build_dual_gpu_acceptance(root: Path) -> dict[str, Any]:
    """Combine two independently generated strict gates after cross-GPU audit."""

    identity = pipeline.current_source_identity(root)
    gates = {
        worker_id: pipeline.load_gate(
            root / f"DETERMINISM_GPU{worker_id}_ACCEPTED",
            expected_source_id=identity["source_id"],
        )
        for worker_id in (0, 1)
    }
    expected_datasets = {0: ["citeseer", "pubmed"], 1: ["dblp", "flickr"]}
    assignments: dict[str, Any] = {}
    runs_by_worker: dict[int, list[dict[str, str]]] = {}
    for worker_id, gate in gates.items():
        details = gate.get("details", {})
        if int(details.get("worker_id", -1)) != worker_id:
            raise RuntimeError(f"per-GPU determinism gate worker mismatch: {worker_id}")
        if int(details.get("gpu_index", -1)) != worker_id:
            raise RuntimeError(f"per-GPU determinism gate physical index mismatch: {worker_id}")
        runs_path = Path(str(details.get("runs_path", "")))
        if not runs_path.is_file() or pipeline.sha256_file(runs_path) != details.get("smoke_runs_sha256"):
            raise RuntimeError(f"per-GPU determinism run artifact mismatch: worker {worker_id}")
        with runs_path.open("r", encoding="utf-8", newline="") as handle:
            runs_by_worker[worker_id] = [dict(row) for row in csv.DictReader(handle)]
        assignments[str(worker_id)] = {
            "worker_id": worker_id,
            "gpu_index": int(details["gpu_index"]),
            "gpu_uuid": details["gpu_uuid"],
            "datasets": expected_datasets[worker_id],
            "software_hardware": details["software_hardware"],
            "per_gpu_acceptance_path": str((root / f"DETERMINISM_GPU{worker_id}_ACCEPTED").resolve()),
            "per_gpu_acceptance_sha256": pipeline.sha256_file(root / f"DETERMINISM_GPU{worker_id}_ACCEPTED"),
            "smoke_runs_sha256": details["smoke_runs_sha256"],
        }
    if assignments["0"]["gpu_uuid"] == assignments["1"]["gpu_uuid"]:
        raise RuntimeError("dual-GPU acceptance requires distinct physical GPUs")
    same_model = (
        assignments["0"]["software_hardware"].get("gpu_name")
        == assignments["1"]["software_hardware"].get("gpu_name")
    )
    exact_fields = (
        "dataset",
        "candidate_fingerprint",
        "architecture",
        "hyperparameters",
        "search_seed",
        "evaluation_replicate",
        "training_seed",
        "decoder_seed",
        "validation_accuracy",
        "epochs_ran",
        "best_epoch",
        "trajectory_hash",
        "model_final_state_hash",
        "source_method",
        "source_search_seed",
    )
    cross_gpu_mismatches: list[dict[str, Any]] = []
    if len(runs_by_worker[0]) != 12 or len(runs_by_worker[1]) != 12:
        raise RuntimeError("each GPU must provide exactly 12 strict smoke runs")
    for index, (left, right) in enumerate(zip(runs_by_worker[0], runs_by_worker[1])):
        mismatched = [field for field in exact_fields if left.get(field) != right.get(field)]
        if mismatched:
            cross_gpu_mismatches.append({"run_index": index, "fields": mismatched})
    cross_gpu_bitwise_exact = not cross_gpu_mismatches
    if same_model and not cross_gpu_bitwise_exact:
        raise RuntimeError("same-model GPUs produced non-bitwise-exact strict smoke outputs")
    summary = {
        "mode": "fixed_dataset_dual_gpu",
        "source_id": identity["source_id"],
        "gpu_assignments": assignments,
        "same_gpu_model": same_model,
        "cross_gpu_bitwise_exact": cross_gpu_bitwise_exact,
        "cross_gpu_mismatches": cross_gpu_mismatches,
        "smoke_run_count_per_gpu": 12,
    }
    input_hash = pipeline.canonical_json_sha256(
        {str(worker_id): pipeline.sha256_file(root / f"DETERMINISM_GPU{worker_id}_ACCEPTED") for worker_id in (0, 1)}
    )
    output_hash = pipeline.canonical_json_sha256(summary)
    return pipeline.write_gate(
        root / "DETERMINISM_ACCEPTED",
        source_id=identity["source_id"],
        input_sha256=input_hash,
        output_sha256=output_hash,
        details={
            "git_head": identity["git_head"],
            "dirty_diff_sha256": identity["dirty_diff_sha256"],
            **summary,
        },
    )


def run_calibration_once(spec: Mapping[str, Any], output: Path) -> dict[str, Any]:
    import torch
    import final_eval
    from dataset_utils import load_dataset_from_request, resolve_dataset_request
    from deterministic_runtime import assert_strict_torch_determinism, deterministic_provenance

    assert_strict_torch_determinism(torch)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    runtime = deterministic_provenance(torch, device=device)
    request = resolve_dataset_request(spec["dataset"], str(pipeline.DATA_ROOT), split_seed=0 if spec["dataset"] == "dblp" else None)
    bundle = load_dataset_from_request(request)
    bundle.to(device)
    architecture = spec["architecture"]
    hp = spec["hyperparameters"]
    training_seed = pipeline.expected_test_training_seed(spec["candidate_fingerprint"], spec["test_seed"])
    result = final_eval.train_and_eval_arch(
        config={
            "operations": architecture["operations"],
            "edges": [tuple(edge) for edge in architecture["edges"]],
            "effective_layers": sum(op != "Identity" for op in architecture["operations"]),
        },
        data=bundle.data,
        in_ch=bundle.num_features,
        out_ch=bundle.num_classes,
        lr=float(hp["lr"]), dropout=float(hp["dropout"]), hidden_dim=int(hp["hidden_dim"]),
        weight_decay=float(hp["l2"]), gcnii_alpha=0.1, gcnii_theta=0.5,
        gat_heads=int(hp.get("gat_heads", 1)), sage_aggr=str(hp.get("sage_aggr", "mean")),
        gin_eps=float(hp.get("gin_eps", 0.0)), gat_heads_by_layer=hp.get("gat_heads_by_layer"),
        sage_aggr_by_layer=hp.get("sage_aggr_by_layer"), gin_eps_by_layer=hp.get("gin_eps_by_layer"),
        device=device, max_epochs=150, patience=40, seed=training_seed,
        track_test=True, return_metadata=True,
        return_reproducibility_metadata=True,
    )
    val, valid, test, metadata = final_eval._unpack_training_result(result)
    if not valid:
        raise RuntimeError("calibration candidate training was invalid")
    row = {
        "dataset": spec["dataset"], "candidate_fingerprint": spec["candidate_fingerprint"],
        "test_seed": spec["test_seed"], "training_seed": training_seed,
        "validation_accuracy": val, "test_accuracy": test,
        "epochs_ran": metadata.get("epochs_ran"), "best_epoch": metadata.get("best_epoch"),
        "trajectory_hash": metadata.get("trajectory_hash"), "model_final_state_hash": metadata.get("model_final_state_hash"),
        "gpu_name": runtime["gpu_name"], "gpu_uuid": runtime["gpu_uuid"], "driver_version": runtime["driver_version"],
    }
    pipeline.atomic_json(output, row)
    return row


def run_cross_gpu_calibration(gpu0: str, gpu1: str, root: Path = pipeline.PIPELINE_ROOT) -> dict[str, Any]:
    identity = pipeline.current_source_identity()
    pipeline.load_gate(root / "TOP10_EXTRACTION_AUDIT_PASSED", expected_source_id=identity["source_id"])
    candidates = pipeline.load_unique_candidates(root)
    chosen = {dataset: next(row for row in candidates if row["dataset"] == dataset) for dataset in pipeline.DATASETS}
    runs = []
    for dataset, candidate in chosen.items():
        spec = {"dataset": dataset, "candidate_fingerprint": candidate["candidate_fingerprint"], "architecture": candidate["architecture"], "hyperparameters": candidate["hyperparameters"], "test_seed": 0}
        for gpu_id in (gpu0, gpu1):
            fd, raw = tempfile.mkstemp(prefix="dvae_calibration_", suffix=".json", dir=root)
            os.close(fd)
            output = Path(raw)
            output.unlink()
            spec_path = output.with_suffix(".spec.json")
            pipeline.atomic_json(spec_path, spec)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["CUBLAS_WORKSPACE_CONFIG"] = prepare_deterministic_environment()
            completed = subprocess.run([str(pipeline.PYTHON_BIN), str(Path(__file__).resolve()), "calibration-once", "--spec", str(spec_path), "--output", str(output)], cwd=pipeline.REPO_ROOT, env=env, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"calibration failed for {dataset} GPU {gpu_id}")
            runs.append(pipeline.read_json(output))
    report = {
        "status": "passed",
        "source_id": identity["source_id"],
        "datasets": list(pipeline.DATASETS),
        "gpu_ids": [str(gpu0), str(gpu1)],
        "runs": runs,
        "bitwise_cross_model_claim": False,
        "note": "Calibration records hardware differences; heterogeneous GPU models are not claimed bitwise deterministic.",
    }
    pipeline.atomic_json(root / "cross_gpu_calibration_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    strict = sub.add_parser("strict")
    strict.add_argument("--gpu-id", required=True)
    strict.add_argument("--root", default=str(pipeline.PIPELINE_ROOT))
    strict.add_argument("--artifact-label")
    strict.add_argument("--acceptance-name", default="DETERMINISM_ACCEPTED")
    strict.add_argument("--expected-gpu-uuid")
    strict.add_argument("--worker-id", type=int)
    strict.add_argument(
        "--legacy-matrix",
        required=True,
        help="formal matrix used to select historical determinism candidates",
    )
    strict.add_argument(
        "--legacy-repro-root",
        required=True,
        help="directory containing reproducibility_pairs/outliers CSV inputs",
    )
    dual = sub.add_parser("accept-dual")
    dual.add_argument("--root", required=True)
    once = sub.add_parser("search-once")
    once.add_argument("--spec", required=True)
    once.add_argument("--output", required=True)
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--gpu0-id", required=True)
    calibration.add_argument("--gpu1-id", required=True)
    calibration_once = sub.add_parser("calibration-once")
    calibration_once.add_argument("--spec", required=True)
    calibration_once.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "strict":
        result = run_strict_smoke(
            args.gpu_id,
            Path(args.root).resolve(),
            artifact_label=args.artifact_label,
            acceptance_name=args.acceptance_name,
            expected_gpu_uuid=args.expected_gpu_uuid,
            worker_id=args.worker_id,
            legacy_matrix=pipeline.resolve_repository_path(args.legacy_matrix),
            legacy_repro_root=pipeline.resolve_repository_path(
                args.legacy_repro_root
            ),
        )
    elif args.command == "accept-dual":
        result = build_dual_gpu_acceptance(Path(args.root).resolve())
    elif args.command == "search-once":
        result = run_search_smoke_once(Path(args.spec), Path(args.output))
    elif args.command == "calibrate":
        result = run_cross_gpu_calibration(args.gpu0_id, args.gpu1_id)
    else:
        result = run_calibration_once(pipeline.read_json(Path(args.spec)), Path(args.output))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
