#!/usr/bin/env python3
"""Merge and analyse the frozen deterministic batch pilot at pool level."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse.original_head_deterministic_batch_pilot import (
    CLASSIFICATIONS, POOL_COUNT, TRAINING_SEEDS, contains_test_field, dump_csv, dump_json,
    file_identity, load_json, mixed_distance, protected_paths, sha256_file, stable_hash,
    verify_freeze,
)
from analyse.run_original_head_batch_pilot_labels import load_jsonl, result_key


STRATEGIES = (
    "HEAD_PARTITION_DIVERSITY", "HEAD_ONLY", "RANDOM_PARTITION_DIVERSITY",
)
ALIASES = {
    "A": "HEAD_PARTITION_DIVERSITY",
    "B": "HEAD_ONLY",
    "C": "RANDOM_PARTITION_DIVERSITY",
}
BOOTSTRAP_REPLICATES = 20000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_complete_seed_results(
    tasks: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if contains_test_field(results):
        raise ValueError("test-named field in GNN results")
    task_keys = [result_key(row) for row in tasks]
    result_keys = [result_key(row) for row in results]
    duplicate_task_keys = len(task_keys) - len(set(task_keys))
    duplicate_result_keys = len(result_keys) - len(set(result_keys))
    expected = set(task_keys)
    actual = set(result_keys)
    by_config: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    invariant_fields: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    for row in results:
        config_key = (
            int(row["pool_id"]), str(row["config_id"]), str(row["architecture_fingerprint"]),
            str(row["hp_fingerprint"]),
        )
        by_config[config_key].add(int(row["seed"]))
        invariant_fields[config_key].add((
            row["dataset_content_fingerprint"], row["split_fingerprint"], row["hp_mode"],
            int(row["eval_epochs"]), int(row["patience"]), int(row["assigned_gpu"]),
        ))
    missing_seed_configs = [
        {"config_key": list(key), "seeds": sorted(seeds)}
        for key, seeds in by_config.items() if seeds != set(TRAINING_SEEDS)
    ]
    protocol_conflicts = [list(key) for key, values in invariant_fields.items() if len(values) != 1]
    statuses = Counter(str(row.get("terminal_status")) for row in results)
    audit = {
        "passed": (
            not duplicate_task_keys and not duplicate_result_keys and expected == actual
            and not missing_seed_configs and not protocol_conflicts
            and statuses == Counter({"success": len(results)})
            and all(bool(row.get("valid")) for row in results)
        ),
        "expected_task_count": len(tasks), "actual_result_count": len(results),
        "duplicate_task_key_count": duplicate_task_keys,
        "duplicate_result_key_count": duplicate_result_keys,
        "missing_keys": [list(key) for key in sorted(expected - actual)],
        "foreign_keys": [list(key) for key in sorted(actual - expected)],
        "missing_seed_configs": missing_seed_configs,
        "protocol_conflicts": protocol_conflicts,
        "terminal_status_counts": dict(statuses),
        "invalid_count": sum(not bool(row.get("valid")) for row in results),
        "test_field_hits": contains_test_field(results),
    }
    return audit


def aggregate_configurations(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(
            int(row["pool_id"]), str(row["config_id"]), str(row["architecture_fingerprint"]),
            str(row["hp_fingerprint"]),
        )].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        values = [float(row["val_acc"]) for row in sorted(rows, key=lambda row: int(row["seed"]))]
        output.append({
            "pool_id": key[0], "config_id": key[1], "architecture_fingerprint": key[2],
            "hp_fingerprint": key[3], "seed_count": len(values),
            "seeds": sorted(int(row["seed"]) for row in rows),
            "seed_validation_values": values,
            "mean_validation": float(np.mean(values)),
            "std_validation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        })
    return output


def _batch_structure_hp_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    distances = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            distance, _ = mixed_distance(rows[left], rows[right])
            distances.append(distance)
    fp_counts = Counter(str(row["architecture_fingerprint"]) for row in rows)
    return {
        "unique_architecture_count": len(fp_counts),
        "same_architecture_multi_hp_count": sum(count > 1 for count in fp_counts.values()),
        "average_pairwise_structure_hp_distance": float(np.mean(distances)),
        "minimum_pairwise_structure_hp_distance": float(np.min(distances)),
    }


def build_pool_metrics(
    selections: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]],
    config_values: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_by_id = {str(row["config_id"]): dict(row) for row in candidates}
    ordered: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selections:
        ordered[(int(row["pool_id"]), str(row["strategy"]))].append(row)
    budget_rows = []
    batch_rows = []
    for pool_id in range(POOL_COUNT):
        for strategy in STRATEGIES:
            selected = sorted(ordered[(pool_id, strategy)], key=lambda row: int(row["budget_order"]))
            if len(selected) != 4:
                raise ValueError(f"missing frozen selections for pool={pool_id}, strategy={strategy}")
            configs = [candidate_by_id[str(row["config_id"])] for row in selected]
            batch_rows.append({"pool_id": pool_id, "strategy": strategy, **_batch_structure_hp_summary(configs)})
            for budget in (1, 2, 4):
                prefix = selected[:budget]
                values = [config_values[str(row["config_id"])] for row in prefix]
                best_index = int(np.argmax(values))
                budget_rows.append({
                    "pool_id": pool_id, "strategy": strategy, "budget": budget,
                    "best_so_far": float(values[best_index]),
                    "best_config_id": str(prefix[best_index]["config_id"]),
                })
    best4 = {(row["pool_id"], row["strategy"]): row for row in budget_rows if row["budget"] == 4}
    union_ids_by_pool: dict[int, set[str]] = defaultdict(set)
    for row in selections:
        union_ids_by_pool[int(row["pool_id"])].add(str(row["config_id"]))
    regret_rows = []
    for pool_id in range(POOL_COUNT):
        union_best = max(config_values[config_id] for config_id in union_ids_by_pool[pool_id])
        union_best_ids = sorted(
            config_id for config_id in union_ids_by_pool[pool_id]
            if abs(config_values[config_id] - union_best) <= 1e-12
        )
        for row in [item for item in budget_rows if item["pool_id"] == pool_id]:
            regret_rows.append({
                **row, "observed_union_best": float(union_best),
                "observed_union_best_config_ids": union_best_ids,
                "observed_union_regret": float(union_best - row["best_so_far"]),
                "provides_observed_union_best": abs(row["best_so_far"] - union_best) <= 1e-12,
                "not_full_candidate_pool_regret": True,
            })
    return budget_rows, batch_rows, regret_rows


def paired_differences(budget_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best = {
        (int(row["pool_id"]), str(row["strategy"])): float(row["best_so_far"])
        for row in budget_rows if int(row["budget"]) == 4
    }
    output = []
    for pool_id in range(POOL_COUNT):
        a, b, c = (best[(pool_id, ALIASES[alias])] for alias in ("A", "B", "C"))
        output.append({
            "pool_id": pool_id, "A_best_of_4": a, "B_best_of_4": b, "C_best_of_4": c,
            "A_minus_C": a - c, "A_minus_B": a - b, "B_minus_C": b - c,
        })
    return output


def paired_pool_bootstrap(
    differences: Sequence[Mapping[str, Any]], replicates: int, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(differences) != POOL_COUNT:
        raise ValueError("bootstrap requires all eight pools")
    names = ("A_minus_C", "A_minus_B", "B_minus_C")
    matrix = np.asarray([[float(row[name]) for name in names] for row in differences], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, POOL_COUNT, size=(replicates, POOL_COUNT))
    samples = matrix[indices].mean(axis=1)
    rows = [
        {"replicate": index, **{name: float(samples[index, column]) for column, name in enumerate(names)}}
        for index in range(replicates)
    ]
    summary = {
        "replicates": replicates, "seed": seed, "outer_unit": "pool",
        "sample_size_per_replicate": POOL_COUNT,
        "shared_indices_for_all_comparisons": True,
        "indices_sha256": hashlib.sha256(indices.astype(np.int16).tobytes()).hexdigest(),
        "comparisons": {
            name: {
                "observed_mean": float(np.mean(matrix[:, column])),
                "bootstrap_mean": float(np.mean(samples[:, column])),
                "ci95_lower": float(np.quantile(samples[:, column], 0.025)),
                "ci95_upper": float(np.quantile(samples[:, column], 0.975)),
            }
            for column, name in enumerate(names)
        },
    }
    return rows, summary


def classify(
    differences: Sequence[Mapping[str, Any]], bootstrap: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    arrays = {
        name: np.asarray([float(row[name]) for row in differences], dtype=np.float64)
        for name in ("A_minus_C", "A_minus_B", "B_minus_C")
    }
    comparisons = bootstrap["comparisons"]
    wins = {name: int(np.sum(values > 0)) for name, values in arrays.items()}
    losses = {name: int(np.sum(values < 0)) for name, values in arrays.items()}
    ties = {name: int(np.sum(values == 0)) for name, values in arrays.items()}
    means = {name: float(np.mean(values)) for name, values in arrays.items()}
    a_gate = (
        means["A_minus_C"] >= 0.006
        and float(comparisons["A_minus_C"]["ci95_lower"]) > 0.0
        and wins["A_minus_C"] >= 6
        and float(comparisons["A_minus_B"]["ci95_lower"]) > -0.006
    )
    b_gate = (
        means["B_minus_C"] >= 0.006
        and float(comparisons["B_minus_C"]["ci95_lower"]) > 0.0
        and wins["B_minus_C"] >= 6
    )
    random_direction = (
        means["A_minus_C"] <= 0.0 and wins["A_minus_C"] <= 4
        and means["B_minus_C"] <= 0.0 and wins["B_minus_C"] <= 4
    )
    if a_gate:
        classification = "HEAD_DIVERSITY_PROMOTED"
    elif b_gate:
        classification = "HEAD_ONLY_PROMOTED"
    elif random_direction:
        classification = "RANDOM_DIVERSITY_RETAINED"
    else:
        classification = "PILOT_INCONCLUSIVE"
    evidence = {
        "means": means, "wins": wins, "losses": losses, "ties": ties,
        "bootstrap": comparisons, "A_gate_passed": a_gate,
        "B_gate_passed_after_A_failed": (not a_gate and b_gate),
        "random_direction_definition_passed": random_direction,
    }
    return classification, evidence


def head_diagnostics(
    config_rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = {str(row["config_id"]): float(row["mean_validation"]) for row in config_rows}
    means = {str(row["config_id"]): float(row["head_prediction_mean"]) for row in predictions}
    output = []
    for pool_id in range(POOL_COUNT):
        ids = sorted(row["config_id"] for row in config_rows if int(row["pool_id"]) == pool_id)
        truth = np.asarray([values[config_id] for config_id in ids], dtype=np.float64)
        pred = np.asarray([means[config_id] for config_id in ids], dtype=np.float64)
        tau = kendalltau(pred, truth, variant="b").statistic
        pred_order = sorted(ids, key=lambda config_id: (-means[config_id], config_id))
        errors = [
            {
                "pool_id": pool_id, "config_id": config_id,
                "predicted_rank": rank,
                "head_prediction_mean": means[config_id], "mean_validation": values[config_id],
                "absolute_error": abs(means[config_id] - values[config_id]),
            }
            for rank, config_id in enumerate(pred_order[:4], 1)
        ]
        output.extend(errors)
        output.append({
            "pool_id": pool_id, "record_type": "pool_summary",
            "observed_union_config_count": len(ids),
            "kendall_tau_b": None if math.isnan(float(tau)) else float(tau),
            "mae": float(np.mean(np.abs(pred - truth))),
        })
    summaries = [row for row in output if row.get("record_type") == "pool_summary"]
    return output, {
        "descriptive_only": True,
        "observed_selected_union_only": True,
        "not_a_promotion_gate": True,
        "macro_kendall_tau_b": float(np.mean([row["kendall_tau_b"] for row in summaries if row["kendall_tau_b"] is not None])),
        "macro_mae": float(np.mean([row["mae"] for row in summaries])),
    }


def protected_after_audit(before_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    before = load_json(before_path)
    after = {str(path.resolve()): file_identity(path) for path in protected_paths()}
    mismatches = []
    for path, expected in before.items():
        actual = after.get(path)
        if actual is None or actual["sha256"] != expected["sha256"]:
            mismatches.append({"path": path, "before": expected, "after": actual})
    return after, {"passed": not mismatches, "mismatches": mismatches}


def write_protocol_failure(root: Path, reason: str, evidence: Mapping[str, Any]) -> None:
    dump_json(root / "final_classification.json", {
        "classification": "EXECUTION_OR_PROTOCOL_FAILURE", "created_at": utc_now(),
        "reason": reason, "evidence": dict(evidence),
        "allows_predictor_guided_mutation_search": False,
    })


def analyse(root: Path) -> None:
    freeze = verify_freeze(root)
    if not freeze["passed"]:
        write_protocol_failure(root, "pre-label immutable manifest mismatch", freeze)
        return
    prereg = load_json(root / "preregistration.json")
    tasks = load_json(root / "selected_union_tasks.json")
    selections = load_json(root / "frozen_strategy_selections.json")
    candidates = load_json(root / "frozen_candidate_pools.json")
    predictions = load_json(root / "frozen_head_predictions.json")
    shard_paths = [root / f"phase_b/shards/gpu{gpu}/seed_results.jsonl" for gpu in (0, 1)]
    if not all(path.is_file() for path in shard_paths):
        write_protocol_failure(root, "missing GPU shard result", {"paths": [str(path) for path in shard_paths]})
        return
    shard_rows = [load_jsonl(path) for path in shard_paths]
    results = shard_rows[0] + shard_rows[1]
    audit = validate_complete_seed_results(tasks, results)
    dump_json(root / "phase_b/gnn_merge_audit.json", audit)
    dump_json(root / "phase_b/seed_results_merged.json", sorted(results, key=result_key))
    dump_csv(root / "phase_b/seed_results_merged.csv", sorted(results, key=result_key))
    if not audit["passed"]:
        write_protocol_failure(root, "GNN task completeness or protocol audit failed", audit)
        return

    config_rows = aggregate_configurations(results)
    analysis_root = root / "analysis"
    dump_json(analysis_root / "config_level_validation.json", config_rows)
    dump_csv(analysis_root / "config_level_validation.csv", config_rows)
    dump_json(analysis_root / "config_seed_completeness.json", audit)
    config_values = {str(row["config_id"]): float(row["mean_validation"]) for row in config_rows}
    budgets, batch_rows, regrets = build_pool_metrics(selections, candidates, config_values)
    differences = paired_differences(budgets)
    dump_csv(analysis_root / "pool_strategy_budget_metrics.csv", budgets)
    dump_json(analysis_root / "batch_diversity_metrics.json", batch_rows)
    dump_csv(analysis_root / "batch_diversity_metrics.csv", batch_rows)
    dump_csv(analysis_root / "observed_union_regret.csv", regrets)
    dump_csv(analysis_root / "paired_pool_differences.csv", differences)
    bootstrap_rows, bootstrap = paired_pool_bootstrap(
        differences, BOOTSTRAP_REPLICATES, int(prereg["statistics"]["bootstrap_seed"]),
    )
    dump_csv(analysis_root / "paired_pool_bootstrap_20000.csv", bootstrap_rows)
    dump_json(analysis_root / "bootstrap_20000_summary.json", bootstrap)
    classification, evidence = classify(differences, bootstrap)
    dump_json(analysis_root / "final_gate_evidence.json", evidence)
    diagnostics, diagnostic_summary = head_diagnostics(config_rows, predictions)
    dump_json(analysis_root / "head_observed_union_diagnostics.json", {
        "summary": diagnostic_summary, "rows": diagnostics,
    })

    after, protected_audit = protected_after_audit(root / "protected_hashes_before.json")
    dump_json(root / "protected_hashes_after.json", after)
    dump_json(root / "protected_objects_audit.json", protected_audit)
    if not protected_audit["passed"]:
        write_protocol_failure(root, "protected object changed", protected_audit)
        return

    union_count = len({row["config_id"] for row in tasks})
    provider_counts = Counter(
        row["strategy"] for row in regrets
        if int(row["budget"]) == 4 and bool(row["provides_observed_union_best"])
    )
    final = {
        "format_version": prereg["format_version"], "created_at": utc_now(),
        "classification": classification,
        "gate_evidence": evidence,
        "unique_evaluated_configuration_count": union_count,
        "gnn_task_count": len(results),
        "gpu_task_counts": dict(Counter(int(row["assigned_gpu"]) for row in results)),
        "success_count": sum(row["terminal_status"] == "success" for row in results),
        "failure_count": sum(row["terminal_status"] == "failed" for row in results),
        "invalid_count": sum(row["terminal_status"] == "invalid" for row in results),
        "retry_count": sum(int(row["retry_count"]) for row in results),
        "observed_union_best_provider_pool_counts": dict(provider_counts),
        "head_diagnostics": diagnostic_summary,
        "protected_objects_unchanged": True,
        "allows_predictor_guided_mutation_search": classification in {
            "HEAD_DIVERSITY_PROMOTED", "HEAD_ONLY_PROMOTED",
        },
        "formal_bo_allowed": False,
        "test_metric_used": False,
    }
    if classification not in CLASSIFICATIONS:
        raise RuntimeError(classification)
    dump_json(root / "final_classification.json", final)
    report = [
        "# Original Head deterministic batch pilot — final goal report",
        "",
        f"- Final classification: `{classification}`",
        f"- Evaluated union: {union_count} unique architecture+HP configurations / {len(results)} GNN tasks.",
        f"- GPU tasks: {dict(Counter(int(row['assigned_gpu']) for row in results))}; failures={final['failure_count']}, retries={final['retry_count']}.",
        f"- A-C mean: {evidence['means']['A_minus_C']:.6f}; CI95 [{evidence['bootstrap']['A_minus_C']['ci95_lower']:.6f}, {evidence['bootstrap']['A_minus_C']['ci95_upper']:.6f}]; wins={evidence['wins']['A_minus_C']}/8.",
        f"- A-B mean: {evidence['means']['A_minus_B']:.6f}; CI95 [{evidence['bootstrap']['A_minus_B']['ci95_lower']:.6f}, {evidence['bootstrap']['A_minus_B']['ci95_upper']:.6f}].",
        f"- B-C mean: {evidence['means']['B_minus_C']:.6f}; CI95 [{evidence['bootstrap']['B_minus_C']['ci95_lower']:.6f}, {evidence['bootstrap']['B_minus_C']['ci95_upper']:.6f}]; wins={evidence['wins']['B_minus_C']}/8.",
        "- Regret is observed-union regret only; it is not full-candidate-pool true regret or a true top-k hit rate.",
        f"- Predictor-guided multi-round mutation search allowed: {final['allows_predictor_guided_mutation_search']}.",
        "- Formal BO, test evaluation, and all model training remained out of scope.",
    ]
    (root / "final_goal_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (root / "validation_report.md").write_text(
        "# Validation report\n\n"
        f"- Immutable freeze: {freeze['passed']}\n"
        f"- GNN completeness/protocol: {audit['passed']}\n"
        f"- Protected objects unchanged: {protected_audit['passed']}\n"
        f"- Test-named result fields: {bool(audit['test_field_hits'])}\n"
        f"- Classification: `{classification}`\n",
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    analyse(args.root.resolve())


if __name__ == "__main__":
    main()
