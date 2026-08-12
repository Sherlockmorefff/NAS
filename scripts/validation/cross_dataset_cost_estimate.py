"""Build a transparent GPU-hour estimate from persistent preflight artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from surrogate.checkpoint_io import atomic_json_dump


METHODS = ("S0", "G100", "G150")
TARGET_DATASETS = ("citeseer", "pubmed", "dblp", "flickr")
FORMAL_LOW_FIDELITY_CANDIDATES = 207
FORMAL_LOW_FIDELITY_EPOCHS = 20
FORMAL_FULL_EVALUATIONS = 300
FORMAL_SEARCH_SEEDS = 5


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _reference_method_overhead(legacy_results: Path) -> dict[str, dict[str, float]]:
    paths = {
        "S0": legacy_results / "formal_seedfair_full300_schur_global4_seed0_pool768",
        "G100": legacy_results / "formal_seedfair_full300_gmm_ted_lowfid_gmm_fit_pool_exp100_global4_seed0_pool768",
        "G150": legacy_results / "formal_seedfair_full300_gmm_ted_lowfid_gmm_fit_pool_exp150_global4_seed0_pool768",
    }
    result: dict[str, dict[str, float]] = {}
    for method, root in paths.items():
        history = _read(root / "history_final.json")
        summary = _read(root / "run_summary.json")
        evaluation_seconds = sum(float(row.get("eval_seconds") or 0.0) for row in history)
        low_fidelity_seconds = 0.0
        budget_path = root / "budget_summary.json"
        if budget_path.is_file():
            low_fidelity_seconds = float(
                _read(budget_path).get("low_fidelity_wall_seconds") or 0.0
            )
        elapsed_seconds = float(summary["elapsed_seconds"])
        result[method] = {
            "source_elapsed_seconds": elapsed_seconds,
            "source_full_evaluation_seconds": evaluation_seconds,
            "source_low_fidelity_seconds": low_fidelity_seconds,
            "non_gnn_search_overhead_seconds": max(
                0.0,
                elapsed_seconds - evaluation_seconds - low_fidelity_seconds,
            ),
        }
    return result


def _dataset_measurements(preflight: Path, dataset: str) -> dict[str, Any]:
    timing = _read(preflight / "timing150" / dataset / "smoke_result.json")
    full_rates: list[float] = []
    low_rates: list[float] = []
    for method_dir in ("S0", "G100_initial_pass", "G150_initial_pass"):
        history = _read(
            preflight / "method_path" / dataset / method_dir / "history_final.json"
        )
        full_rates.extend(
            float(row["eval_seconds"]) / int(row["epochs_ran"])
            for row in history
            if bool(row.get("valid")) and int(row.get("epochs_ran") or 0) > 0
        )
    for method_dir in ("G100_initial_pass", "G150_initial_pass"):
        history = _read(
            preflight
            / "method_path"
            / dataset
            / method_dir
            / "low_fidelity_history.json"
        )
        low_rates.extend(
            float(row["gpu_seconds"]) / int(row["actual_epochs"])
            for row in history
            if int(row.get("actual_epochs") or 0) > 0
        )
    observed_early_stop_epochs = int(timing["epochs_ran"])
    lower_full = float(timing["runtime_seconds"])
    baseline_full = max(
        lower_full,
        statistics.median(full_rates) * observed_early_stop_epochs,
    )
    upper_full = max(full_rates) * 150.0
    return {
        "timing_candidate_fingerprint": timing["candidate_fingerprint"],
        "timing_requested_epochs": 150,
        "timing_patience": 40,
        "timing_actual_epochs": observed_early_stop_epochs,
        "timing_best_epoch": int(timing["best_epoch"]),
        "timing_runtime_seconds": lower_full,
        "timing_peak_allocated_bytes": int(timing["peak_allocated_bytes"]),
        "timing_peak_reserved_bytes": int(timing["peak_reserved_bytes"]),
        "full_seconds_per_epoch_observed": {
            "minimum": min(full_rates),
            "median": statistics.median(full_rates),
            "maximum": max(full_rates),
            "sample_count": len(full_rates),
        },
        "single_full_evaluation_seconds": {
            "lower_observed_candidate": lower_full,
            "baseline_architecture_median_with_observed_early_stop": baseline_full,
            "upper_architecture_max_without_early_stop": upper_full,
        },
        "low_fidelity_seconds_per_epoch_observed": {
            "minimum": min(low_rates),
            "median": statistics.median(low_rates),
            "maximum": max(low_rates),
            "sample_count": len(low_rates),
        },
    }


def _cora_reference_measurements(legacy_results: Path, preflight: Path) -> dict[str, Any]:
    timing = _read(preflight / "timing150" / "cora" / "smoke_result.json")
    roots = (
        legacy_results / "formal_seedfair_full300_schur_global4_seed0_pool768",
        legacy_results / "formal_seedfair_full300_gmm_ted_lowfid_gmm_fit_pool_exp100_global4_seed0_pool768",
        legacy_results / "formal_seedfair_full300_gmm_ted_lowfid_gmm_fit_pool_exp150_global4_seed0_pool768",
    )
    full_rates: list[float] = []
    low_rates: list[float] = []
    for root in roots:
        full_rates.extend(
            float(row["eval_seconds"]) / int(row["epochs_ran"])
            for row in _read(root / "history_final.json")
            if int(row.get("epochs_ran") or 0) > 0
        )
        low_path = root / "low_fidelity_history.json"
        if low_path.is_file():
            low_rates.extend(
                float(row["gpu_seconds"]) / int(row["actual_epochs"])
                for row in _read(low_path)
                if int(row.get("actual_epochs") or 0) > 0
            )
    observed_epochs = int(timing["epochs_ran"])
    lower_full = float(timing["runtime_seconds"])
    return {
        "reference_only_not_in_75_run_new_dataset_matrix": True,
        "timing_candidate_fingerprint": timing["candidate_fingerprint"],
        "timing_requested_epochs": 150,
        "timing_patience": 40,
        "timing_actual_epochs": observed_epochs,
        "timing_best_epoch": int(timing["best_epoch"]),
        "timing_runtime_seconds": lower_full,
        "timing_peak_allocated_bytes": int(timing["peak_allocated_bytes"]),
        "timing_peak_reserved_bytes": int(timing["peak_reserved_bytes"]),
        "full_seconds_per_epoch_observed": {
            "minimum": min(full_rates),
            "median": statistics.median(full_rates),
            "maximum": max(full_rates),
            "sample_count": len(full_rates),
        },
        "single_full_evaluation_seconds": {
            "lower_observed_candidate": lower_full,
            "baseline_architecture_median_with_observed_early_stop": max(
                lower_full, statistics.median(full_rates) * observed_epochs
            ),
            "upper_architecture_max_without_early_stop": max(full_rates) * 150.0,
        },
        "low_fidelity_seconds_per_epoch_observed": {
            "minimum": min(low_rates),
            "median": statistics.median(low_rates),
            "maximum": max(low_rates),
            "sample_count": len(low_rates),
        },
    }


def _estimate_dataset(
    measurements: dict[str, Any], overhead: dict[str, dict[str, float]]
) -> dict[str, Any]:
    full = measurements["single_full_evaluation_seconds"]
    low_rates = measurements["low_fidelity_seconds_per_epoch_observed"]
    low_seconds = {
        "lower": FORMAL_LOW_FIDELITY_CANDIDATES
        * FORMAL_LOW_FIDELITY_EPOCHS
        * float(low_rates["minimum"]),
        "baseline": FORMAL_LOW_FIDELITY_CANDIDATES
        * FORMAL_LOW_FIDELITY_EPOCHS
        * float(low_rates["median"]),
        "upper": FORMAL_LOW_FIDELITY_CANDIDATES
        * FORMAL_LOW_FIDELITY_EPOCHS
        * float(low_rates["maximum"]),
    }
    searches: dict[str, dict[str, float]] = {}
    for method in METHODS:
        base_overhead = overhead[method]["non_gnn_search_overhead_seconds"]
        method_low = low_seconds if method != "S0" else {
            "lower": 0.0,
            "baseline": 0.0,
            "upper": 0.0,
        }
        searches[method] = {
            "lower_seconds": base_overhead
            + FORMAL_FULL_EVALUATIONS * float(full["lower_observed_candidate"])
            + method_low["lower"],
            "baseline_seconds": base_overhead
            + FORMAL_FULL_EVALUATIONS
            * float(full["baseline_architecture_median_with_observed_early_stop"])
            + method_low["baseline"],
            "upper_seconds": 1.25 * base_overhead
            + FORMAL_FULL_EVALUATIONS
            * float(full["upper_architecture_max_without_early_stop"])
            + method_low["upper"],
        }
    fifteen = {
        bound: FORMAL_SEARCH_SEEDS
        * sum(row[f"{bound}_seconds"] for row in searches.values())
        for bound in ("lower", "baseline", "upper")
    }
    return {
        **measurements,
        "formal_low_fidelity": {
            "candidate_count_per_gmm_search": FORMAL_LOW_FIDELITY_CANDIDATES,
            "epochs_per_candidate": FORMAL_LOW_FIDELITY_EPOCHS,
            "early_stopping": False,
            "seconds_per_gmm_search": low_seconds,
        },
        "one_300_full_search_seconds": searches,
        "fifteen_formal_search_runs_seconds": fifteen,
        "fifteen_formal_search_runs_gpu_hours": {
            key: value / 3600.0 for key, value in fifteen.items()
        },
    }


def _two_gpu_makespan_seconds(durations: Sequence[float]) -> float:
    """Deterministic longest-processing-time schedule for two identical GPUs."""

    loads = [0.0, 0.0]
    for duration in sorted((float(value) for value in durations), reverse=True):
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += duration
    return max(loads)


def build(
    repo: Path,
    *,
    legacy_root: Path | None = None,
    flickr_continuous_preflight: Path | None = None,
) -> dict[str, Any]:
    archive = legacy_root or repo / "legacy_artifacts" / "pre_20260813"
    legacy_results = archive / "results"
    preflight = legacy_results / "cross_dataset_v1_preflight"
    overhead = _reference_method_overhead(legacy_results)
    measurements = {
        dataset: _dataset_measurements(preflight, dataset)
        for dataset in TARGET_DATASETS
    }
    continuous_summary = None
    if flickr_continuous_preflight is not None:
        continuous = _read(flickr_continuous_preflight)
        if (
            continuous.get("status") != "completed"
            or continuous.get("oom") is not False
        ):
            raise ValueError(
                "Flickr continuous extreme preflight is not completed without OOM"
            )
        full_candidate = continuous["full_fidelity_candidate"]
        wall_seconds = float(full_candidate["candidate_evaluation_wall_seconds"])
        actual_epochs = int(full_candidate["actual_epochs"])
        if actual_epochs <= 0:
            raise ValueError("Flickr extreme preflight recorded no completed epochs")
        extrapolated_150 = wall_seconds / actual_epochs * 150.0
        flickr_rates = measurements["flickr"]["full_seconds_per_epoch_observed"]
        flickr_rates["maximum"] = max(
            float(flickr_rates["maximum"]), wall_seconds / actual_epochs
        )
        flickr_rates["minimum"] = min(
            float(flickr_rates["minimum"]), wall_seconds / actual_epochs
        )
        flickr_rates["sample_count"] = int(flickr_rates["sample_count"]) + 1
        flickr_full = measurements["flickr"]["single_full_evaluation_seconds"]
        flickr_full["upper_architecture_max_without_early_stop"] = max(
            float(flickr_full["upper_architecture_max_without_early_stop"]),
            extrapolated_150,
        )
        continuous_summary = {
            "path": str(flickr_continuous_preflight.resolve()),
            "candidate_fingerprint": full_candidate["candidate_fingerprint"],
            "pure_training_and_evaluator_wall_seconds": wall_seconds,
            "actual_epochs": actual_epochs,
            "early_stopping": bool(full_candidate["early_stopping"]),
            "seconds_per_epoch": wall_seconds / actual_epochs,
            "150_epoch_upper_seconds": extrapolated_150,
            "data_loading_seconds": float(continuous["data_loading_seconds"]),
            "dataset_transfer_seconds": float(
                continuous["dataset_transfer_seconds"]
            ),
            "vae_loading_seconds": float(continuous["vae_loading_seconds"]),
            "decode_only_seconds": float(
                continuous["decode_only"]["decode_search"]["decode_seconds"]
            ),
            "complete_preflight_process_wall_seconds": float(
                continuous["complete_process_wall_seconds"]
            ),
            "peak_allocated_bytes": int(full_candidate["peak_allocated_bytes"]),
            "peak_reserved_bytes": int(full_candidate["peak_reserved_bytes"]),
            "nvidia_smi_peak_used_mib": int(
                continuous["nvml_or_nvidia_smi_peak_used_mib"]
            ),
        }
    datasets = {
        dataset: _estimate_dataset(measurements[dataset], overhead)
        for dataset in TARGET_DATASETS
    }
    datasets["ogbn-arxiv"] = {
        "status": "blocked_resource_preflight_oom",
        "included_in_formal_matrix": False,
        "reason": "recorded full-batch OOM on a 24 GB RTX 4500 Ada",
        "resource_preflight": str(
            preflight
            / "resource_gpu_host"
            / "ogbn-arxiv"
            / "resource_preflight.json"
        ),
    }
    feasible = list(TARGET_DATASETS)
    bounds = ("lower", "baseline", "upper")
    sixty_run_gpu_hours = {
        bound: sum(
            datasets[name]["fifteen_formal_search_runs_seconds"][bound]
            for name in feasible
        )
        / 3600.0
        for bound in bounds
    }
    two_gpu_wall_hours = {}
    for bound in bounds:
        jobs = [
            datasets[name]["one_300_full_search_seconds"][method][
                f"{bound}_seconds"
            ]
            for name in feasible
            for method in METHODS
            for _seed in range(FORMAL_SEARCH_SEEDS)
        ]
        two_gpu_wall_hours[bound] = _two_gpu_makespan_seconds(jobs) / 3600.0
    final_evaluation = {}
    final_total_upper_seconds = 0.0
    for dataset in TARGET_DATASETS:
        seconds_per_epoch = float(
            datasets[dataset]["full_seconds_per_epoch_observed"]["maximum"]
        )
        final_single_upper = seconds_per_epoch * 300.0
        pre_dedup_evaluations = 15 * 10 * 10
        dataset_upper = final_single_upper * pre_dedup_evaluations
        final_total_upper_seconds += dataset_upper
        final_evaluation[dataset] = {
            "histories": 15,
            "top_k_per_history": 10,
            "replicates_per_candidate": 10,
            "pre_dedup_evaluation_count_upper": pre_dedup_evaluations,
            "single_300_epoch_evaluation_upper_seconds": final_single_upper,
            "pre_dedup_gpu_hours_upper": dataset_upper / 3600.0,
            "post_dedup_formula": (
                "unique_top10_candidate_fingerprints_for_dataset * 10 replicates "
                "* single_300_epoch_evaluation_seconds / 3600"
            ),
        }
    return {
        "format_version": 1,
        "scope": "preflight estimate only; no formal search or final evaluation launched",
        "formal_matrix": {
            "datasets": list(TARGET_DATASETS),
            "target_datasets": 4,
            "methods_per_dataset": 3,
            "search_seeds_per_method": 5,
            "planned_search_runs": 60,
            "ogbn_arxiv_status": "blocked_resource_preflight_oom",
            "ogbn_arxiv_included": False,
        },
        "early_stopping_assumptions": {
            "lower_and_baseline": "use actual 150-max/patience-40 timing epochs from one decoded candidate; baseline combines those epochs with the median seconds/epoch across reduced path candidates",
            "upper": "assumes all 150 epochs and the maximum observed seconds/epoch; no early stopping",
            "low_fidelity": "formal fixed 20 epochs with patience 0",
        },
        "exact_gp_qlogei_overhead_reference": overhead,
        "overhead_semantics": {
            "candidate_evaluation": "real decoder plus full-batch GNN train/eval wall time",
            "data_and_process_startup": "reported separately and not multiplied as pure training",
            "gp_qlogei": "non-GNN reference overhead retained from accepted formal histories",
        },
        "flickr_continuous_extreme_measurement": continuous_summary,
        "datasets": datasets,
        "formal_60_run_search_gpu_hours": sixty_run_gpu_hours,
        "formal_60_run_two_gpu_wall_hours": two_gpu_wall_hours,
        "two_gpu_schedule": {
            "policy": "deterministic longest-processing-time over 60 independent runs",
            "gpu_count": 2,
            "max_concurrent_training_processes_per_gpu": 1,
        },
        "top_k_final_evaluation": {
            "status": "not_started",
            "included_in_formal_search_estimate": False,
            "protocol": {
                "top_k": 10,
                "replicates": 10,
                "eval_epochs": 300,
                "patience": 80,
            },
            "by_dataset": final_evaluation,
            "pre_dedup_gpu_hours_upper_total": final_total_upper_seconds / 3600.0,
            "post_dedup_total_formula": (
                "sum over datasets of (unique top-10 candidate fingerprints across "
                "15 formal histories * 10 replicates * dataset-specific final "
                "evaluation seconds) / 3600"
            ),
            "reason": "estimated separately; launch is outside the current authorization",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument(
        "--legacy-root",
        default=None,
        help="archive containing the historical results/ input tree",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--flickr-continuous-preflight", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build(
        Path(args.repo).resolve(),
        legacy_root=(
            None if args.legacy_root is None else Path(args.legacy_root).resolve()
        ),
        flickr_continuous_preflight=(
            None
            if args.flickr_continuous_preflight is None
            else Path(args.flickr_continuous_preflight).resolve()
        ),
    )
    atomic_json_dump(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
