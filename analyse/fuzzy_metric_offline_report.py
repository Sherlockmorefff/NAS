#!/usr/bin/env python3
"""Summarize saved Stage-2 tasks after completion or a hard-budget stop.

This module never loads A_validation.json, a global oracle, B labels, a VAE,
or a dataset, and never fits a model. It reads frozen identity metadata and
already-persisted task predictions/evaluations. Its output is deliberately
outer_summary.json, separate from the original all-576-task summary.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyse.fuzzy_metric_offline_core import aggregate_evaluations

METHODS = ("regression", "N0", "N1", "N2")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean_tree(values):
    if not values:
        return None
    if all(numeric(value) for value in values):
        return float(statistics.mean(values))
    if all(isinstance(value, dict) for value in values):
        shared = set(values[0]).intersection(*(set(value) for value in values[1:]))
        return {key: mean_tree([value[key] for value in values]) for key in sorted(shared)}
    return None


def range_tree(values):
    if not values:
        return None
    if all(numeric(value) for value in values):
        return {"values": list(values), "mean": float(statistics.mean(values)),
                "min": min(values), "max": max(values),
                "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else None}
    if all(isinstance(value, dict) for value in values):
        shared = set(values[0]).intersection(*(set(value) for value in values[1:]))
        return {key: range_tree([value[key] for value in values]) for key in sorted(shared)}
    return None


def check_frozen_metadata(output):
    protocol = read_json(output / "protocol.json")
    partition = read_json(output / "partition.json")
    matrix = read_json(output / "task_matrix.json")
    boundary = read_json(output / "label_boundary.json")
    protocol_body = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    if digest(protocol_body) != protocol["protocol_hash"]:
        raise RuntimeError("protocol identity changed")
    if digest(partition) != protocol["partition_hash"] or digest(matrix) != protocol["task_matrix_hash"]:
        raise RuntimeError("frozen partition or task matrix changed")
    if sorted(boundary["exported_ids"]) != sorted(partition["A_ids"]):
        raise RuntimeError("saved export boundary differs from A membership")
    if boundary["B_parsed_records"] != 0 or boundary["test_records"] != 0:
        raise RuntimeError("saved label boundary does not certify exclusion of B/test records")
    if set(partition["A_ids"]) & set(partition["B_ids"]) or set(partition["A_architectures"]) & set(partition["B_architectures"]):
        raise RuntimeError("A/B candidate or architecture membership overlaps")
    if len({task["task_id"] for task in matrix}) != len(matrix):
        raise RuntimeError("duplicate planned task IDs")
    source_checks = []
    for relative, expected in protocol["source_identity"].items():
        archived = output / "source_snapshot" / relative
        snapshot_hash = sha256(archived)
        if snapshot_hash != expected:
            raise RuntimeError(f"archived source identity changed: {relative}")
        current = ROOT / relative
        current_hash = sha256(current) if current.is_file() else None
        source_checks.append({"path": relative, "archived_sha256": snapshot_hash,
                              "current_sha256": current_hash, "current_matches_archive": current_hash == expected})
    core_relative = "analyse/fuzzy_metric_offline_core.py"
    if sha256(ROOT / core_relative) != protocol["source_identity"][core_relative]:
        raise RuntimeError("aggregation helper differs from the frozen numerical core")
    return protocol, partition, matrix, boundary, source_checks


def scan_saved_results(output, protocol, matrix, boundary):
    planned = {task["task_id"]: task for task in matrix}
    completed, predictions, evaluations = {}, {}, {}
    failed, attempts = [], {}
    selection_cache = {}
    for start_path in sorted((output / "tasks").glob("*/attempt*/started.json")):
        start = read_json(start_path)
        task_id = start["task"]["task_id"]
        attempt = int(start_path.parent.name.removeprefix("attempt"))
        if task_id not in planned or task_id != start_path.parent.parent.name:
            raise RuntimeError("unexpected saved task attempt")
        expected_task = copy.deepcopy(planned[task_id])
        if expected_task["stage"] == "refit":
            family = expected_task["family"]
            if family not in selection_cache:
                selection_cache[family] = read_json(output / expected_task["setting_source"])
            expected_task["setting"] = selection_cache[family]["settings"][expected_task["method"]]
        if start["task"] != expected_task:
            raise RuntimeError(f"saved task differs from frozen plan: {task_id}")
        identity = digest({"task": expected_task, "protocol": protocol["protocol_hash"], "A_labels": boundary["exported_hash"]})
        if start["identity"] != identity:
            raise RuntimeError(f"saved attempt identity mismatch: {task_id}")
        key = (task_id, attempt)
        attempts[key] = {"task_id": task_id, "attempt": attempt,
                         "directory": str(start_path.parent.relative_to(output)), "status": "started_unresolved"}
        done_path, failure_path = start_path.parent / "completed.json", start_path.parent / "failed.json"
        if done_path.exists() and failure_path.exists():
            raise RuntimeError(f"attempt has both completion and failure: {task_id}")
        if failure_path.exists():
            failure = read_json(failure_path)
            if failure["identity"] != identity or failure["task_id"] != task_id or int(failure["attempt"]) != attempt:
                raise RuntimeError("failure record identity mismatch")
            traceback_text = failure.get("traceback", "")
            budget_block = "hard fit budget reached" in traceback_text
            attempts[key]["status"] = "budget_blocked" if budget_block else "failed"
            failed.append({**attempts[key], "elapsed_seconds": failure.get("elapsed_seconds"),
                "failure_record": str(failure_path.relative_to(output)),
                "error_last_line": traceback_text.strip().splitlines()[-1] if traceback_text.strip() else None,
                "budget_blocked": budget_block})
        if not done_path.exists():
            continue
        result = read_json(done_path)
        if task_id in completed:
            raise RuntimeError(f"multiple completed attempts for one task: {task_id}")
        if (result["identity"] != identity or result["task_id"] != task_id
                or int(result["attempt"]) != attempt or result["setting"] != expected_task["setting"]
                or result["family"] != expected_task["family"] or result["stage"] != expected_task["stage"]
                or result["seed"] != expected_task["seed"]):
            raise RuntimeError(f"completion metadata differs: {task_id}")
        if Path(result["directory"]).resolve() != start_path.parent.resolve():
            raise RuntimeError("completion directory differs from its saved attempt")
        required = {"model.pt", "predictions_before_reveal.json", "evaluation.json"}
        if set(result["artifact_hashes"]) != required:
            raise RuntimeError("completion artifact registry differs")
        for name in sorted(required):
            if sha256(start_path.parent / name) != result["artifact_hashes"][name]:
                raise RuntimeError(f"completed artifact changed: {task_id}/{name}")
        bundle = read_json(start_path.parent / "predictions_before_reveal.json")
        evaluation = read_json(start_path.parent / "evaluation.json")
        if (bundle["train_ids"] != sorted(expected_task["train_ids"])
                or bundle["query_ids"] != sorted(expected_task["query_ids"])
                or bundle["setting"] != expected_task["setting"] or bundle["model_seed"] != expected_task["seed"]
                or bundle["query_labels_used"] is not False or bundle["test_labels_used"] is not False):
            raise RuntimeError(f"frozen prediction boundary differs: {task_id}")
        if bundle["standardization"]["fit_ids"] != sorted(expected_task["train_ids"]):
            raise RuntimeError("standardization used a different training partition")
        if evaluation["metrics"] != result["metrics"]:
            raise RuntimeError("completed metrics differ from saved evaluation")
        evaluated_ids = [row["candidate_id"] for row in evaluation["per_query"]]
        if sorted(evaluated_ids) != sorted(expected_task["query_ids"]) or len(set(evaluated_ids)) != len(evaluated_ids):
            raise RuntimeError("evaluation candidate identity differs")
        access = read_json(start_path.parent / "evaluation_access.json")
        if (access["prediction_hash"] != result["artifact_hashes"]["predictions_before_reveal.json"]
                or access["model_hash"] != result["artifact_hashes"]["model.pt"]
                or sorted(access["ids"]) != sorted(expected_task["query_ids"])):
            raise RuntimeError("evaluation access was not bound to the saved predictions/model")
        attempts[key]["status"] = "completed"
        completed[task_id], predictions[task_id], evaluations[task_id] = result, bundle, evaluation
    marker_count = len(list((output / "tasks").glob("*/attempt*/completed.json")))
    if marker_count != len(completed):
        raise RuntimeError("completion marker has no recognized started attempt")
    return completed, predictions, evaluations, attempts, failed, selection_cache


def budget_summary(output, config, completed, attempts, failures):
    path = output / "fit_attempts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    event_keys = [(row["task_id"], int(row["attempt"]), row["kind"]) for row in rows]
    if len(event_keys) != len(set(event_keys)):
        raise RuntimeError("duplicate charged fit-start records")
    if any((task_id, attempt) not in attempts or kind not in {"mapping", "gp"} for task_id, attempt, kind in event_keys):
        raise RuntimeError("unrecognized charged fit event")
    charged_by_attempt = defaultdict(set)
    for task_id, attempt, kind in event_keys:
        charged_by_attempt[(task_id, attempt)].add(kind)
    counts = {}
    for kind in ("mapping", "gp"):
        selected = [(task_id, attempt) for task_id, attempt, event_kind in event_keys if event_kind == kind]
        statuses = Counter(attempts[key]["status"] for key in selected)
        limit = config[f"{kind}_fit_limit"]
        counts[kind] = {"started": len(selected), "limit": limit, "remaining": max(0, limit - len(selected)),
            "completed_task_fits": statuses["completed"], "charged_failed_attempt_fits": statuses["failed"],
            "charged_budget_blocked_attempt_fits": statuses["budget_blocked"],
            "started_unresolved_fits": statuses["started_unresolved"], "hard_cap_reached": len(selected) >= limit}
        if len(selected) > limit:
            raise RuntimeError("saved fit ledger exceeds the frozen hard budget")
    failed = []
    budget_blocks = []
    for row in failures:
        charged = sorted(charged_by_attempt[(row["task_id"], row["attempt"])])
        item = {**row, "charged_fit_kinds": charged, "no_fit_started": not charged}
        (budget_blocks if row["budget_blocked"] else failed).append(item)
    unresolved = [row for row in attempts.values() if row["status"] == "started_unresolved"]
    return {"mapping": counts["mapping"], "gp": counts["gp"], "ledger_records": len(rows),
        "completed_tasks": len(completed), "failed_attempts_count": len(failed), "failed_attempts": failed,
        "budget_blocked_attempts_count": len(budget_blocks), "budget_blocked_attempts": budget_blocks,
        "started_unresolved_attempts": unresolved,
        "failed_attempt_elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0) for row in failed),
        "budget_blocked_attempt_elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0) for row in budget_blocks),
        "count_semantics": "fit starts are charged; failed tasks and uncharged budget refusals are distinct; no synthetic checks included",
        "ledger_sha256": sha256(path) if path.exists() else None}


def diagnostics_summary(completed, bundles):
    rows, grouped = [], defaultdict(list)
    for task_id, result in sorted(completed.items()):
        value = bundles[task_id]["diagnostics"]
        item = {"task_id": task_id, "family": result["family"], "stage": result["stage"], "seed": result["seed"],
            "setting": result["setting"], "initialization_hash": bundles[task_id]["initialization_hash"],
            "initial_losses": value["initial_losses"], "final_losses": value["final_losses"],
            "initial_embedding_norm": value["initial_embedding_norm"], "train_embedding_norm": value["train_embedding_norm"],
            "query_embedding_norm": value["query_embedding_norm"], "positive_pair_ratio": value["positive_pair_ratio"],
            "positive_pair_count": value["positive_pair_count"], "unordered_pair_count": value["unordered_pair_count"],
            "mapping_gradients_finite": value["mapping_gradients_finite"], "gp_gradients_finite": value["gp_gradients_finite"],
            "mapping_seconds": value["mapping_seconds"], "gp_seconds": value["gp_seconds"],
            "prediction_seconds": value["prediction_seconds"], "task_elapsed_seconds": result["elapsed_seconds"],
            "mapping_steps": value["mapping_steps"], "gp_steps": value["gp_steps"],
            "full_loss_trajectory_file": str(Path(result["directory"]) / "predictions_before_reveal.json")}
        rows.append(item)
        grouped[f"{result['stage']}/{result['setting']['method']}"].append(item)
    keys = ("initial_losses", "final_losses", "initial_embedding_norm", "train_embedding_norm", "query_embedding_norm",
            "positive_pair_ratio", "mapping_seconds", "gp_seconds", "prediction_seconds", "task_elapsed_seconds")
    return {"task_rows": rows, "by_stage_method": {key: {"completed_tasks": len(group),
        "mean": mean_tree([{field: row[field] for field in keys} for row in group]),
        "range": range_tree([{field: row[field] for field in keys} for row in group])}
        for key, group in sorted(grouped.items())},
        "total_mapping_seconds_completed_tasks": sum(row["mapping_seconds"] for row in rows),
        "total_gp_seconds_completed_tasks": sum(row["gp_seconds"] for row in rows),
        "total_prediction_seconds_completed_tasks": sum(row["prediction_seconds"] for row in rows),
        "total_task_elapsed_seconds_completed_tasks": sum(row["task_elapsed_seconds"] for row in rows),
        "time_semantics": "sum of task/fit times, not parallel wall clock; failed-task elapsed time is separately recorded"}


def outer_comparisons(aggregate, config):
    baseline = aggregate["regression"]["mean_seed_metrics"]
    comparisons = {}
    for method in ("N0", "N1", "N2"):
        current = aggregate[method]["mean_seed_metrics"]
        base_metric = baseline["main_neighborhood_metric"]
        relative = 1 - current["main_neighborhood_metric"] / base_metric if base_metric > 0 else None
        mae_delta = current["gp_mae"] - baseline["gp_mae"]
        intervals, notes = {}, []
        for level in ("90", "95"):
            coverage_delta = current[f"coverage{level}"] - baseline[f"coverage{level}"]
            width_delta = current[f"width{level}"] - baseline[f"width{level}"]
            intervals[level] = {"observation_coverage": current[f"coverage{level}"],
                "regression_observation_coverage": baseline[f"coverage{level}"], "coverage_delta": coverage_delta,
                "observation_width": current[f"width{level}"], "regression_observation_width": baseline[f"width{level}"],
                "width_delta": width_delta, "coverage_lower_than_regression": coverage_delta < 0,
                "interval_wider_than_regression": width_delta > 0,
                "below_nominal_coverage": current[f"coverage{level}"] < int(level) / 100,
                "coverage_loss_with_narrower_intervals": coverage_delta < 0 and width_delta < 0}
            if coverage_delta < 0:
                notes.append(f"{level}%区间覆盖率比仅回归低{abs(coverage_delta):.6f}")
            if width_delta > 0:
                notes.append(f"{level}%区间平均宽度比仅回归增加{width_delta:.6f}")
            if current[f"coverage{level}"] < int(level) / 100:
                notes.append(f"{level}%区间覆盖率低于其名义覆盖水平")
        paired = []
        for index, seed in enumerate(config["model_seeds"]):
            first = aggregate[method]["seed_metrics"][index]
            second = aggregate["regression"]["seed_metrics"][index]
            paired.append({"model_seed": seed,
                "neighbor_metric_difference": first["main_neighborhood_metric"] - second["main_neighborhood_metric"],
                "gp_mae_difference": first["gp_mae"] - second["gp_mae"]})
        comparisons[method] = {"relative_neighbor_improvement": relative,
            "absolute_neighbor_difference": current["main_neighborhood_metric"] - base_metric,
            "gp_mae_increase": mae_delta, "prediction_gate": mae_delta <= config["gp_mae_tolerance"],
            "neighborhood_gate": relative is not None and relative >= config["neighborhood_improvement_required"],
            "development_gate": relative is not None and relative >= config["neighborhood_improvement_required"] and mae_delta <= config["gp_mae_tolerance"],
            "paired_seed_differences": paired, "interval_comparison": intervals,
            "plain_interval_flags": notes,
            "interval_interpretation": "描述性差值；未冻结显著退化阈值，不据此宣称GP探索可靠"}
    n1 = aggregate["N1"]["mean_seed_metrics"]["main_neighborhood_metric"]
    n2 = aggregate["N2"]["mean_seed_metrics"]["main_neighborhood_metric"]
    difference = 1 - n2 / n1 if n1 > 0 else None
    both_predict = all(comparisons[method]["prediction_gate"] for method in ("N1", "N2"))
    simpler = both_predict and ((difference is not None and difference < config["n1_simplicity_tolerance"]) or (n1 == n2 == 0))
    eligible = [method for method in ("N0", "N1", "N2") if comparisons[method]["development_gate"]]
    recommended = None
    if eligible:
        recommended = "N1" if simpler and "N1" in eligible else min(eligible,
            key=lambda method: (aggregate[method]["mean_seed_metrics"]["main_neighborhood_metric"], METHODS.index(method)))
    return {"comparisons": comparisons, "N2_relative_improvement_over_N1": difference,
        "N2_absolute_neighbor_difference_from_N1": n2 - n1, "N1_N2_both_satisfy_prediction_requirement": both_predict,
        "N1_simplicity_preference": simpler, "simplicity_tolerance": config["n1_simplicity_tolerance"],
        "recommended_method_by_outer_development_rule": recommended,
        "recommendation_is_fixed_final_configuration": False,
        "rule_interpretation": "10%邻域改善和0.001误差门槛是开发推进规则；2%为择简容差，不是显著性或统计等效"}


def build_report(output):
    output = Path(output).resolve()
    protocol, partition, matrix, boundary, source_checks = check_frozen_metadata(output)
    ledger_path = output / "fit_attempts.jsonl"
    ledger_before = sha256(ledger_path) if ledger_path.exists() else None
    completed, bundles, evaluations, attempts, failures, selections = scan_saved_results(output, protocol, matrix, boundary)
    config = protocol["config"]
    budget = budget_summary(output, config, completed, attempts, failures)
    families = [fold["family"] for fold in partition["folds"]]
    seeds = list(config["model_seeds"])
    aggregate, missing_outer = {}, []
    for method in METHODS:
        seed_results, seed_metrics = [], []
        for seed in seeds:
            rows = [row for row in completed.values() if row["stage"] == "refit" and row["family"] in families
                    and row["seed"] == seed and row["setting"]["method"] == method]
            by_family = {row["family"]: row for row in rows}
            if len(rows) != len(families) or set(by_family) != set(families):
                missing_outer.append({"method": method, "model_seed": seed,
                                      "missing_families": sorted(set(families) - set(by_family))})
                continue
            merged = aggregate_evaluations([evaluations[by_family[family]["task_id"]] for family in families])
            observed_ids = [row["candidate_id"] for row in merged["per_query"]]
            if sorted(observed_ids) != sorted(partition["A_ids"]) or len(set(observed_ids)) != len(observed_ids):
                raise RuntimeError("outer merge must contain every A candidate exactly once per method/model seed")
            seed_metrics.append(merged["metrics"])
            seed_results.append({"model_seed": seed, "families": families, "aggregate": merged,
                "fold_settings": {family: by_family[family]["setting"] for family in families}})
        if len(seed_results) == len(seeds):
            subgroup_names = set(seed_results[0]["aggregate"]["subgroups"])
            subgroup_summary = {}
            for subgroup in sorted(subgroup_names):
                subgroup_metrics = [row["aggregate"]["subgroups"][subgroup]["metrics"] for row in seed_results]
                subgroup_summary[subgroup] = {"seed_metrics": subgroup_metrics,
                    "mean_seed_metrics": mean_tree(subgroup_metrics), "seed_metric_ranges": range_tree(subgroup_metrics)}
            tolerance_counts = []
            for row in seed_results:
                query_rows = row["aggregate"]["per_query"]
                counts = {}
                for tolerance in config["evaluation_tolerances"]:
                    key = str(tolerance)
                    values = [query["tolerances"][key] for query in query_rows]
                    existing = sum(value["reference_has_neighbor"] for value in values)
                    counts[key] = {"queries_with_eligible_reference": existing,
                        "queries_without_eligible_reference": len(values) - existing,
                        "retrieved_pairs": 5 * len(values),
                        "wrong_retrieved_pairs": sum(5 - value["retrieved_eligible_count"] for value in values),
                        "eligible_reference_relationships": sum(value["reference_eligible_count"] for value in values)}
                tolerance_counts.append({"model_seed": row["model_seed"], "counts": counts})
            aggregate[method] = {"model_seeds": seeds, "seed_metrics": seed_metrics, "per_seed": seed_results,
                "subgroup_seed_summaries": subgroup_summary, "tolerance_counts_by_seed": tolerance_counts,
                "mean_seed_metrics": mean_tree(seed_metrics), "seed_metric_ranges": range_tree(seed_metrics),
                "aggregation_rule": "within each model seed merge five folds once per A candidate; then average three seed metrics"}
    outer_complete = not missing_outer and set(aggregate) == set(METHODS)
    outer_task_ids = {task["task_id"] for task in matrix if task["family"] in families}
    outer_matrix_complete = outer_task_ids <= set(completed)
    final_inner = {task["task_id"] for task in matrix if task["family"] == "final_A" and task["stage"] == "inner"}
    final_refits = {task["task_id"] for task in matrix if task["family"] == "final_A" and task["stage"] == "refit"}
    all_final_inner = final_inner <= set(completed)
    fixed_config = None
    final_selection_path = output / "selection/final_A.json"
    if all_final_inner and final_selection_path.exists():
        fixed_config = read_json(final_selection_path)["settings"]
    pending = sorted({task["task_id"] for task in matrix} - set(completed))
    selection_by_fold = {family: selections.get(family) or (read_json(output / "selection" / f"{family}.json")
        if (output / "selection" / f"{family}.json").exists() else None) for family in families}
    comparison = outer_comparisons(aggregate, config) if outer_complete else None
    report = {"format_version": "fuzzy_metric_stage2_saved_outer_report_v1",
        "status": "completed_stage2" if not pending else "partial_stage2_outer_complete" if outer_complete and outer_matrix_complete else "partial_stage2_outer_incomplete",
        "stage2_complete": not pending, "outer_evaluation_complete": outer_complete, "outer_480_task_matrix_complete": outer_matrix_complete,
        "expected_total_tasks": len(matrix), "completed_tasks": len(completed), "pending_task_ids": pending,
        "pending_final_A_inner_ids": sorted(final_inner - set(completed)), "pending_final_A_refit_ids": sorted(final_refits - set(completed)),
        "completed_final_A_inner_tasks": len(final_inner & set(completed)),
        "completed_final_A_refit_tasks": len(final_refits & set(completed)),
        "missing_outer_groups": missing_outer, "aggregate": aggregate, "outer_comparison": comparison,
        "outer_fold_selections": selection_by_fold, "fixed_config": fixed_config,
        "fixed_config_status": "saved_selection_from_complete_final_A_grid" if fixed_config is not None else "not_available_final_A_grid_or_saved_selection_incomplete",
        "budget": budget, "diagnostics": diagnostics_summary(completed, bundles),
        "existing_audit_records": {name: {"sha256": sha256(output / name), "record": read_json(output / name)}
            for name in ("real_output_checks.json", "budget_stop.json") if (output / name).is_file()},
        "data": {"A_candidates": len(partition["A_ids"]), "B_candidates": len(partition["B_ids"]),
            "A_architectures": len(partition["A_architectures"]), "B_architectures": len(partition["B_architectures"]),
            "A_hash": partition["A_hash"], "B_hash": partition["B_hash"],
            "stage3_original_capacity_feasible": partition["stage3_original_capacity_feasible"],
            "p2_capacity_checks": partition["p2_checks"], "use": partition["data_use"]},
        "frozen_inputs": {"protocol_hash": protocol["protocol_hash"], "partition_hash": protocol["partition_hash"],
            "task_matrix_hash": protocol["task_matrix_hash"], "label_export_hash_from_metadata_only": boundary["exported_hash"],
            "prepared_manifest": protocol["prepared_manifest"], "config": config,
            "source_archive_checks": source_checks, "recorded_host": protocol["host"], "recorded_branch": protocol["branch"], "recorded_head": protocol["head"]},
        "reporting_boundary": {"raw_A_labels_read": False, "raw_B_labels_read": False, "test_results_read": False,
            "mapping_fits_started": 0, "gp_fits_started": 0, "gnn_trainings_started": 0,
            "label_source": "only previously saved evaluation artifacts", "B_parsed_records_as_recorded": boundary["B_parsed_records"],
            "test_records_as_recorded": boundary["test_records"], "reporter_sha256": sha256(Path(__file__))},
        "interpretation": "已完成外层结果属于开发评价；预算阻塞与科学结果分别报告，缺失最终A选参和重训不得标完成；未启动阶段3",
        "stage3_transfer": "参数和规则须待最终A完整选参；禁止把全A权重暗中带入B初始50训练"}
    if ledger_before != (sha256(ledger_path) if ledger_path.exists() else None):
        raise RuntimeError("fit ledger changed while reporting; wait for workers to stop before saving")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing Stage-2 result root")
    parser.add_argument("--write", action="store_true", help="write a new outer_summary.json; never overwrite")
    args = parser.parse_args(argv)
    report = build_report(args.output)
    destination = args.output / "outer_summary.json"
    if args.write:
        if report["budget"]["started_unresolved_attempts"]:
            raise RuntimeError("unresolved attempts remain; refuse to publish a final snapshot")
        with destination.open("x") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "completed_tasks": report["completed_tasks"],
        "pending_task_count": len(report["pending_task_ids"]), "outer_evaluation_complete": report["outer_evaluation_complete"],
        "mapping_fits": report["budget"]["mapping"]["started"], "gp_fits": report["budget"]["gp"]["started"],
        "fixed_config": report["fixed_config"], "outer_comparison": report["outer_comparison"],
        "saved": str(destination) if args.write else None}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
