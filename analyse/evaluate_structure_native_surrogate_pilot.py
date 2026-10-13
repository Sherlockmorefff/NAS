#!/usr/bin/env python3
"""Merge and evaluate the frozen structure-native prospective pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse.run_structure_native_surrogate_labels import contains_test_named_field, load_jsonl
from analyse.structure_native_surrogate_pilot import (
    FINAL_BOOTSTRAPS,
    MINIMUM_RELIABLE_GAP,
    MODEL_NAMES,
    REPO_ROOT,
    attach_intervals,
    dump_csv,
    dump_json,
    load_json,
    metric_bundle,
    protected_hashes,
    sha256_file,
    stratified_bootstrap_samples,
    summarize_with_bootstrap,
    verify_frozen_plan,
    write_sha256_sidecar,
)


def t_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "seed_count": 0,
            "validation_mean": None,
            "validation_sample_std": None,
            "validation_ci95_half_width": None,
            "validation_ci95_low": None,
            "validation_ci95_high": None,
        }
    mean = statistics.fmean(values)
    if len(values) < 2:
        std = 0.0
        half = None
    else:
        std = statistics.stdev(values)
        half = float(t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    return {
        "seed_count": len(values),
        "validation_mean": mean,
        "validation_sample_std": std,
        "validation_ci95_half_width": half,
        "validation_ci95_low": None if half is None else mean - half,
        "validation_ci95_high": None if half is None else mean + half,
    }


def configuration_statistics(
    seed_rows: Sequence[Mapping[str, Any]], selection: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["config_id"])].append(row)
    output = []
    for config in selection:
        values = [
            float(row["val_acc"])
            for row in grouped.get(str(config["config_id"]), [])
            if row["terminal_status"] == "success" and bool(row["valid"])
        ]
        terminal_counts = Counter(
            str(row["terminal_status"])
            for row in grouped.get(str(config["config_id"]), [])
        )
        output.append(
            {
                "config_id": config["config_id"],
                "architecture_fingerprint": config["architecture_fingerprint"],
                "selection_stratum": config["selection_stratum"],
                "assigned_gpu": config["assigned_gpu"],
                **t_interval(values),
                "terminal_counts": dict(terminal_counts),
                "retry_count": sum(int(row["retry_count"]) for row in grouped.get(str(config["config_id"]), [])),
            }
        )
    return output


def load_all_seed_rows(output: Path) -> list[dict[str, Any]]:
    rows = []
    keys = set()
    for shard in (0, 1):
        path = output / f"phase_5/shards/gpu{shard}/seed_results.jsonl"
        for row in load_jsonl(path):
            key = (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
            if key in keys:
                raise ValueError(f"duplicate cross-shard merge key: {key}")
            keys.add(key)
            rows.append(row)
    return rows


def expected_keys(
    plan: Mapping[str, Any],
    selection: Sequence[Mapping[str, Any]],
    adaptive_ids: set[str] | None,
) -> set[tuple[str, str, int]]:
    output = {
        (str(row["config_id"]), str(row["architecture_fingerprint"]), int(seed))
        for row in selection
        for seed in plan["initial_seeds"]
    }
    if adaptive_ids is not None:
        output.update(
            (str(row["config_id"]), str(row["architecture_fingerprint"]), int(seed))
            for row in selection
            if str(row["config_id"]) in adaptive_ids
            for seed in plan["adaptive_seeds"]
        )
    return output


def run_initial_merge(plan_path: Path, output: Path) -> dict[str, Any]:
    plan = verify_frozen_plan(plan_path)
    selection = load_json(Path(plan["selection_path"]))
    rows = load_all_seed_rows(output)
    initial_seed_set = set(map(int, plan["initial_seeds"]))
    initial_rows = [row for row in rows if int(row["seed"]) in initial_seed_set]
    observed = {
        (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
        for row in initial_rows
    }
    expected = expected_keys(plan, selection, None)
    if observed != expected:
        raise RuntimeError(f"initial task mismatch: missing={len(expected-observed)}, unexpected={len(observed-expected)}")
    statistics_rows = configuration_statistics(initial_rows, selection)
    adaptive_ids = sorted(
        str(row["config_id"])
        for row in statistics_rows
        if int(row["seed_count"]) < 15
        or row["validation_ci95_half_width"] is None
        or float(row["validation_ci95_half_width"]) > 0.006
    )
    phase = output / "phase_5"
    dump_json(phase / "initial_config_statistics.json", statistics_rows)
    dump_csv(phase / "initial_config_statistics.csv", statistics_rows)
    adaptive = {
        "rule": "validation_t_ci95_half_width_gt_0.006_or_fewer_than_15_successful_seeds",
        "prediction_correctness_read": False,
        "config_ids": adaptive_ids,
        "configuration_count": len(adaptive_ids),
        "seeds": list(plan["adaptive_seeds"]),
        "initial_task_count": len(initial_rows),
        "initial_seed_result_sha256_by_shard": {
            str(shard): sha256_file(output / f"phase_5/shards/gpu{shard}/seed_results.jsonl")
            for shard in (0, 1)
        },
    }
    adaptive_path = phase / "adaptive_selection.json"
    dump_json(adaptive_path, adaptive)
    write_sha256_sidecar(adaptive_path)
    result = {
        "classification": "INITIAL_LABELS_COMPLETE",
        "initial_task_count": len(initial_rows),
        "adaptive_configuration_count": len(adaptive_ids),
        "adaptive_task_count": len(adaptive_ids) * len(plan["adaptive_seeds"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def pair_details(rows: Sequence[Mapping[str, Any]], *, scope: str, model: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["architecture_fingerprint"]))
    output = []
    for left_index in range(len(ordered)):
        for right_index in range(left_index + 1, len(ordered)):
            left, right = ordered[left_index], ordered[right_index]
            delta = float(left["validation_mean"]) - float(right["validation_mean"])
            if delta == 0:
                continue
            higher, lower = (left, right) if delta > 0 else (right, left)
            disjoint = (
                float(higher["validation_ci95_low"]) > float(lower["validation_ci95_high"])
                or float(lower["validation_ci95_low"]) > float(higher["validation_ci95_high"])
            )
            reliable = abs(delta) >= MINIMUM_RELIABLE_GAP and disjoint
            output.append(
                {
                    "scope": scope,
                    "model": model,
                    "pair_key": "::".join(sorted((str(left["architecture_fingerprint"]), str(right["architecture_fingerprint"])))),
                    "higher_config_id": higher["config_id"],
                    "higher_architecture_fingerprint": higher["architecture_fingerprint"],
                    "higher_validation_mean": higher["validation_mean"],
                    "lower_config_id": lower["config_id"],
                    "lower_architecture_fingerprint": lower["architecture_fingerprint"],
                    "lower_validation_mean": lower["validation_mean"],
                    "validation_gap": abs(delta),
                    "confidence_intervals_disjoint": disjoint,
                    "reliable": reliable,
                    "prediction_correct": float(higher["prediction"]) > float(lower["prediction"]),
                    "higher_prediction": higher["prediction"],
                    "lower_prediction": lower["prediction"],
                }
            )
    return output


def paired_metric_difference(
    candidate_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    candidate_samples: Mapping[str, Sequence[float | None]],
    baseline_samples: Mapping[str, Sequence[float | None]],
    metric: str,
) -> dict[str, Any]:
    paired = [
        float(candidate) - float(baseline)
        for candidate, baseline in zip(candidate_samples[metric], baseline_samples[metric])
        if candidate is not None and baseline is not None
    ]
    estimate = float(candidate_summary[metric]) - float(baseline_summary[metric])
    return {
        "candidate": candidate_summary["model"],
        "baseline": baseline_summary["model"],
        "metric": metric,
        "estimate": estimate,
        "ci95_low": float(np.quantile(paired, 0.025)) if paired else None,
        "ci95_high": float(np.quantile(paired, 0.975)) if paired else None,
        "bootstrap_valid_replicates": len(paired),
        "shared_architecture_indices": True,
    }


def gate_model(
    model: Mapping[str, Any],
    refit_head: Mapping[str, Any],
    difference: Mapping[str, Any],
    strata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checks = {
        "reliable_pair_count_at_least_20": int(model["reliable_pair_count"]) >= 20,
        "reliable_architecture_count_at_least_12": int(model["reliable_architecture_count"]) >= 12,
        "kendall_ci_low_gt_zero": model["kendall_ci95_low"] is not None and float(model["kendall_ci95_low"]) > 0.0,
        "reliable_pairwise_ci_low_gt_0_5": model["reliable_pairwise_accuracy_ci95_low"] is not None and float(model["reliable_pairwise_accuracy_ci95_low"]) > 0.5,
        "kendall_increment_vs_refit_head_at_least_0_05": float(difference["estimate"]) >= 0.05,
        "kendall_increment_ci_low_gt_zero": difference["ci95_low"] is not None and float(difference["ci95_low"]) > 0.0,
        "ndcg_at_10_not_below_refit_head": float(model["ndcg_at_10"]) >= float(refit_head["ndcg_at_10"]) - 1e-12,
        "regret_at_1_not_worse_by_more_than_0_006": float(model["regret_at_1"]) <= float(refit_head["regret_at_1"]) + 0.006 + 1e-12,
        "regret_at_3_not_worse_by_more_than_0_006": float(model["regret_at_3"]) <= float(refit_head["regret_at_3"]) + 0.006 + 1e-12,
        "regret_at_5_not_above_refit_head": float(model["regret_at_5"]) <= float(refit_head["regret_at_5"]) + 1e-12,
        "prediction_interval_coverage_at_least_0_85": float(model["prediction_interval_coverage"]) >= 0.85,
        "prediction_interval_mean_width_below_refit_head": float(model["prediction_interval_mean_width"]) < float(refit_head["prediction_interval_mean_width"]) - 1e-12,
        "both_selection_strata_kendall_nonnegative": all(
            strata[name].get("kendall") is not None and float(strata[name]["kendall"]) >= 0.0
            for name in ("hash_uniform", "structure_diverse")
        ),
    }
    return {"model": model["model"], "passed": all(checks.values()), "checks": checks}


def final_classification(
    summaries: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
    differences: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    d_pass = bool(gates["d_structure"]["passed"])
    h_pass = bool(gates["h_structure"]["passed"])
    h_vs_d = differences[("h_structure", "d_structure")]
    ensembles = [model for model in ("head_h_mean", "head_h_conservative", "head_h_weighted") if gates[model]["passed"]]
    single_models = ["d_structure", "h_structure"]
    best_single = max(single_models, key=lambda model: float(summaries[model]["kendall"]))
    validated_ensemble = None
    for ensemble in sorted(ensembles, key=lambda model: float(summaries[model]["kendall"]), reverse=True):
        difference = differences[(ensemble, best_single)]
        if float(difference["estimate"]) >= 0.05 and difference["ci95_low"] is not None and float(difference["ci95_low"]) > 0.0:
            validated_ensemble = ensemble
            break
    if d_pass and not (float(h_vs_d["estimate"]) >= 0.05 and h_vs_d["ci95_low"] is not None and float(h_vs_d["ci95_low"]) > 0.0):
        classification = "D_VALIDATED_FOR_OFFLINE_REPLAY"
    elif h_pass and float(h_vs_d["estimate"]) >= 0.05 and h_vs_d["ci95_low"] is not None and float(h_vs_d["ci95_low"]) > 0.0:
        classification = "H_VALIDATED_FOR_OFFLINE_REPLAY"
    elif validated_ensemble is not None:
        classification = "HEAD_H_ENSEMBLE_VALIDATED_FOR_OFFLINE_REPLAY"
    else:
        head_fp = int(summaries["refit_head"]["head_top_quartile_false_positive_count"])
        conservative_fp = int(summaries["head_h_conservative"]["head_top_quartile_false_positive_count"])
        filter_signal = (
            not any(gates[model]["passed"] for model in ("d_structure", "h_structure", *["head_h_mean", "head_h_conservative", "head_h_weighted"]))
            and head_fp - conservative_fp >= 2
            and float(summaries["head_h_conservative"]["regret_at_1"]) <= float(summaries["refit_head"]["regret_at_1"]) + 1e-12
            and float(summaries["head_h_conservative"]["regret_at_3"]) <= float(summaries["refit_head"]["regret_at_3"]) + 1e-12
        )
        structural_models = ("d_structure", "h_structure", "head_h_mean", "head_h_conservative", "head_h_weighted")
        insufficient_pairs = int(summaries["refit_head"]["reliable_pair_count"]) < 20
        intervals_too_wide = all(
            float(summaries[model]["kendall_ci95_high"]) - float(summaries[model]["kendall_ci95_low"]) > 0.8
            for model in structural_models
        )
        if filter_signal:
            classification = "H_FILTER_SIGNAL_ONLY"
        elif insufficient_pairs or intervals_too_wide:
            classification = "SURROGATE_PILOT_UNDERPOWERED"
        else:
            classification = "NO_STRUCTURE_SURROGATE_VALIDATED"
    return classification, {
        "d_passed": d_pass,
        "h_passed": h_pass,
        "validated_ensemble": validated_ensemble,
        "best_single_model": best_single,
        "h_minus_d_kendall": h_vs_d,
    }


def resolve_classification(
    protocol_errors: Sequence[str],
    summaries: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
    differences: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Pure seven-branch resolver used by evaluation and lightweight tests."""

    if protocol_errors:
        return "PROTOCOL_OR_DATA_ERROR", {"protocol_errors": list(protocol_errors)}
    return final_classification(summaries, gates, differences)


def run_final(plan_path: Path, output: Path) -> dict[str, Any]:
    plan = verify_frozen_plan(plan_path)
    selection = load_json(Path(plan["selection_path"]))
    adaptive_path = output / "phase_5/adaptive_selection.json"
    adaptive_sidecar = adaptive_path.with_suffix(".sha256")
    if not adaptive_path.is_file() or not adaptive_sidecar.is_file() or adaptive_sidecar.read_text().split()[0] != sha256_file(adaptive_path):
        raise RuntimeError("adaptive selection is missing or changed")
    adaptive_ids = set(map(str, load_json(adaptive_path)["config_ids"]))
    rows = load_all_seed_rows(output)
    observed = {
        (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
        for row in rows
    }
    expected = expected_keys(plan, selection, adaptive_ids)
    protocol_errors = []
    if observed != expected:
        protocol_errors.append(f"task mismatch missing={len(expected-observed)} unexpected={len(observed-expected)}")
    if any(row["terminal_status"] != "success" or not bool(row["valid"]) for row in rows):
        protocol_errors.append("one or more prospective tasks are not valid successes")
    if any(contains_test_named_field(row) for row in rows):
        protocol_errors.append("test-named field in prospective results")
    if any(str(row["dataset_content_fingerprint"]) != plan["labeling_protocol"]["dataset_content_fingerprint"] or str(row["split_fingerprint"]) != plan["labeling_protocol"]["split_fingerprint"] for row in rows):
        protocol_errors.append("dataset or split fingerprint drift")
    if any(int(row["eval_epochs"]) != 300 or int(row["patience"]) != 80 for row in rows):
        protocol_errors.append("training protocol drift")
    if any(int(row["actual_gpu"]["physical_index"]) != int(row["assigned_gpu"]) or str(row["actual_gpu"]["uuid"]) != str(plan["shard_to_gpu_uuid"][str(row["assigned_gpu"])]) for row in rows):
        protocol_errors.append("GPU assignment drift")
    statistics_rows = configuration_statistics(rows, selection)
    dump_json(output / "phase_5/prospective_config_statistics.json", statistics_rows)
    dump_csv(output / "phase_5/prospective_config_statistics.csv", statistics_rows)
    merged_path = output / "phase_5/prospective_seed_results.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (str(item["config_id"]), int(item["seed"]))):
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    predictions = load_json(Path(plan["frozen_predictions_path"]))
    labels = {str(row["config_id"]): row for row in statistics_rows}
    model_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        label = labels[str(prediction["config_id"])]
        model_rows[str(prediction["model"])].append(
            {
                **dict(prediction),
                "validation_mean": label["validation_mean"],
                "validation_ci95_low": label["validation_ci95_low"],
                "validation_ci95_high": label["validation_ci95_high"],
            }
        )
    if set(model_rows) != set(MODEL_NAMES) or any(len(values) != 24 for values in model_rows.values()):
        protocol_errors.append("frozen prediction coverage mismatch")
    development_fingerprints = {
        row["architecture_fingerprint"]
        for row in load_json(output / "development/development_80_manifest.json")
    }
    if development_fingerprints & {row["architecture_fingerprint"] for row in selection}:
        protocol_errors.append("development/prospective architecture overlap")
    protected_after = protected_hashes()
    if protected_after != plan["protected_hashes"]:
        protocol_errors.append("protected file hash drift")
    phase = output / "phase_6"
    phase.mkdir(parents=True, exist_ok=True)
    if protocol_errors:
        classification = {
            "classification": "PROTOCOL_OR_DATA_ERROR",
            "protocol_errors": protocol_errors,
            "offline_bo_executed": False,
            "formal_bo_executed": False,
            "vae_training_executed": False,
            "latent_distance_gp_used": False,
        }
        dump_json(output / "final_classification.json", classification)
        return classification
    base_rows = [
        {
            "architecture_fingerprint": row["architecture_fingerprint"],
            "selection_stratum": row["selection_stratum"],
        }
        for row in statistics_rows
    ]
    shared_samples = stratified_bootstrap_samples(base_rows, replicates=FINAL_BOOTSTRAPS, seed=20260824)
    summaries: dict[str, dict[str, Any]] = {}
    bootstrap_values: dict[str, dict[str, list[float | None]]] = {}
    stratified_rows = []
    stratified_lookup: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    pair_rows = []
    for model in MODEL_NAMES:
        summary, samples = summarize_with_bootstrap(model_rows[model], shared_samples, interval_alpha=0.10)
        summary.update({"model": model, "scope": "all_24"})
        summaries[model] = summary
        bootstrap_values[model] = samples
        pair_rows.extend(pair_details(model_rows[model], scope="all_24", model=model))
        for stratum in ("hash_uniform", "structure_diverse"):
            subset = [row for row in model_rows[model] if row["selection_stratum"] == stratum]
            subset_samples = stratified_bootstrap_samples(subset, replicates=FINAL_BOOTSTRAPS, seed=20260824)
            stratum_summary, _ = summarize_with_bootstrap(subset, subset_samples, interval_alpha=0.10)
            stratum_summary.update({"model": model, "scope": stratum})
            stratified_rows.append(stratum_summary)
            stratified_lookup[model][stratum] = stratum_summary
            pair_rows.extend(pair_details(subset, scope=stratum, model=model))
        equal = {"model": model, "scope": "two_strata_equal_weight"}
        for metric in ("kendall", "pairwise_accuracy", "reliable_pairwise_accuracy", "ndcg_at_5", "ndcg_at_10", "regret_at_1", "regret_at_3", "regret_at_5", "regret_at_10", "rmse", "mae", "prediction_interval_coverage", "prediction_interval_mean_width", "prediction_interval_score"):
            values = [stratified_lookup[model][stratum].get(metric) for stratum in ("hash_uniform", "structure_diverse")]
            valid = [float(value) for value in values if value is not None]
            equal[metric] = statistics.fmean(valid) if valid else None
        stratified_rows.append(equal)
    differences: dict[tuple[str, str], dict[str, Any]] = {}
    difference_rows = []
    for candidate in MODEL_NAMES:
        for baseline in ("refit_head", "d_structure", "h_structure"):
            if candidate == baseline:
                difference = {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": "kendall",
                    "estimate": 0.0,
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                    "bootstrap_valid_replicates": FINAL_BOOTSTRAPS,
                    "shared_architecture_indices": True,
                }
            else:
                difference = paired_metric_difference(
                    summaries[candidate],
                    summaries[baseline],
                    bootstrap_values[candidate],
                    bootstrap_values[baseline],
                    "kendall",
                )
            differences[(candidate, baseline)] = difference
            difference_rows.append(difference)
    gates = {
        model: gate_model(
            summaries[model],
            summaries["refit_head"],
            differences[(model, "refit_head")],
            stratified_lookup[model],
        )
        for model in MODEL_NAMES
    }
    classification_name, classification_evidence = final_classification(summaries, gates, differences)
    false_positive_rows = [
        {
            "model": model,
            "predicted_top_quartile_count": math.ceil(24 / 4),
            "true_below_median_false_positive_count": summaries[model]["head_top_quartile_false_positive_count"],
            "false_positive_architecture_fingerprints": summaries[model]["head_top_quartile_false_positive_fingerprints"],
        }
        for model in MODEL_NAMES
    ]
    dump_json(phase / "prospective_model_metrics.json", list(summaries.values()))
    dump_csv(phase / "prospective_model_metrics.csv", list(summaries.values()))
    dump_json(phase / "stratified_model_metrics.json", stratified_rows)
    dump_csv(phase / "stratified_model_metrics.csv", stratified_rows)
    dump_json(phase / "paired_model_differences.json", difference_rows)
    dump_csv(phase / "paired_model_differences.csv", difference_rows)
    dump_json(phase / "prospective_pair_details.json", pair_rows)
    dump_csv(phase / "prospective_pair_details.csv", pair_rows)
    dump_json(phase / "false_positive_filter_analysis.json", false_positive_rows)
    dump_csv(phase / "false_positive_filter_analysis.csv", false_positive_rows)
    dump_json(phase / "model_gates.json", list(gates.values()))
    classification = {
        "classification": classification_name,
        "model_passes": {model: bool(gate["passed"]) for model, gate in gates.items()},
        "classification_evidence": classification_evidence,
        "prospective_configuration_count": 24,
        "prospective_architecture_count": 24,
        "total_seed_task_count": len(rows),
        "adaptive_configuration_count": len(adaptive_ids),
        "reliable_pair_count": summaries["refit_head"]["reliable_pair_count"],
        "reliable_architecture_count": summaries["refit_head"]["reliable_architecture_count"],
        "bootstrap_replicates": FINAL_BOOTSTRAPS,
        "bootstrap_unit": "selection-stratified architecture fingerprint preserving 12/12",
        "test_label_used": False,
        "vae_training_executed": False,
        "latent_distance_gp_used": False,
        "offline_bo_executed": False,
        "formal_bo_executed": False,
        "protected_files_unchanged": True,
        "recommended_next_step": "fixed_log_offline_replay_validation_only" if classification_name.endswith("VALIDATED_FOR_OFFLINE_REPLAY") else "none_automatic",
    }
    dump_json(output / "final_classification.json", classification)
    integrity = {
        "passed": True,
        "expected_task_count": len(expected),
        "observed_task_count": len(observed),
        "duplicate_task_count": len(rows) - len(observed),
        "missing_task_count": len(expected - observed),
        "unexpected_task_count": len(observed - expected),
        "terminal_status_counts": dict(Counter(row["terminal_status"] for row in rows)),
        "retry_count": sum(int(row["retry_count"]) for row in rows),
        "seed_count_distribution": dict(Counter(int(row["seed_count"]) for row in statistics_rows)),
        "gpu_task_counts": dict(Counter(int(row["assigned_gpu"]) for row in rows)),
        "dataset_content_fingerprints": sorted({row["dataset_content_fingerprint"] for row in rows}),
        "split_fingerprints": sorted({row["split_fingerprint"] for row in rows}),
        "selection_development_overlap_count": 0,
        "frozen_predictions_verified": True,
        "protected_hashes_before": plan["protected_hashes"],
        "protected_hashes_after": protected_after,
        "protected_files_unchanged": True,
        "test_label_used": False,
    }
    dump_json(output / "result_integrity_audit.json", integrity)
    report = f"""# Structure-native surrogate pilot

Final classification: **`{classification_name}`**

- Prospective architectures: 24 (12 hash-uniform and 12 structure-diverse).
- Validation-only seed tasks: {len(rows)}; adaptive configurations: {len(adaptive_ids)}.
- Reliable pairs: {summaries['refit_head']['reliable_pair_count']} involving {summaries['refit_head']['reliable_architecture_count']} architectures.
- D Kendall: {summaries['d_structure']['kendall']:.6f}; H Kendall: {summaries['h_structure']['kendall']:.6f}; Refit Head Kendall: {summaries['refit_head']['kendall']:.6f}.
- No VAE training, latent-distance GP, test label, offline BO, or formal BO was used.

Even an `OFFLINE_REPLAY` classification authorizes only a later fixed-log replay study. It does not modify or run BO.
"""
    (output / "final_goal_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(classification, indent=2, sort_keys=True))
    return classification


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initial-merge", "final"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "initial-merge":
        run_initial_merge(args.plan.resolve(), args.output.resolve())
    else:
        run_final(args.plan.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
