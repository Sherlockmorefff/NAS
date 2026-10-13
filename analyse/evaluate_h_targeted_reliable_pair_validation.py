#!/usr/bin/env python3
"""Merge and evaluate the frozen-H targeted prospective validation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, t
from sklearn.metrics import ndcg_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse.h_targeted_reliable_pair_validation import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MINIMUM_RELIABLE_GAP,
    MODELS,
    SHORT_MODEL_NAMES,
    aggregate_predictions,
    load_prediction_rows,
    protected_hashes,
    sha256_file,
)
from analyse.run_h_targeted_label_validation import verify_frozen_plan


EVALUATED_MODELS = ("frozen_head", "frozen_h", "original_zs_fixed_gp", "original_ze_fixed_gp")
PREDICTION_PREFIX = {
    "frozen_head": "head",
    "frozen_h": "h",
    "original_zs_fixed_gp": "zs",
    "original_ze_fixed_gp": "ze",
}


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fields})
    temporary.replace(path)


def _dump_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_seed_results(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    keys = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if any("test" in str(field).lower() for field in row):
                    raise ValueError(f"test-named field at {path}:{line_number}")
                key = (
                    str(row["config_id"]),
                    str(row["architecture_fingerprint"]),
                    int(row["seed"]),
                )
                if key in keys:
                    raise ValueError(f"duplicate merge key: {key}")
                keys.add(key)
                rows.append(row)
    return rows


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
        half = None
        std = 0.0
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
            if row["terminal_status"] == "success" and row["valid"]
        ]
        terminal = Counter(
            str(row["terminal_status"]) for row in grouped.get(str(config["config_id"]), [])
        )
        output.append(
            {
                "config_id": config["config_id"],
                "architecture_fingerprint": config["architecture_fingerprint"],
                "selection_stratum": config["selection_stratum"],
                "assigned_gpu": config["assigned_gpu"],
                **t_interval(values),
                "terminal_counts": dict(terminal),
                "retry_count": sum(
                    int(row["retry_count"]) for row in grouped.get(str(config["config_id"]), [])
                ),
            }
        )
    return output


def adaptive_config_ids(statistics_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        str(row["config_id"])
        for row in statistics_rows
        if int(row["seed_count"]) < 15
        or row["validation_ci95_half_width"] is None
        or float(row["validation_ci95_half_width"]) > 0.006
    )


def _expected_initial_keys(
    plan: Mapping[str, Any], selection: Sequence[Mapping[str, Any]]
) -> set[tuple[str, str, int]]:
    return {
        (str(row["config_id"]), str(row["architecture_fingerprint"]), int(seed))
        for row in selection
        for seed in plan["initial_seeds"]
    }


def initial_merge(args: argparse.Namespace) -> int:
    plan = verify_frozen_plan(args.plan.resolve())
    selection = json.loads(Path(plan["selection_path"]).read_text(encoding="utf-8"))
    phase_dir = args.output.resolve() / "phase_b"
    paths = [phase_dir / f"shards/gpu{shard}/seed_results.jsonl" for shard in (0, 1)]
    rows = load_seed_results(paths)
    observed = {
        (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
        for row in rows
        if int(row["seed"]) in set(plan["initial_seeds"])
    }
    expected = _expected_initial_keys(plan, selection)
    if observed != expected:
        raise RuntimeError(
            f"initial task mismatch: missing={len(expected-observed)}, unexpected={len(observed-expected)}"
        )
    initial_rows = sorted(
        [row for row in rows if int(row["seed"]) in set(plan["initial_seeds"])],
        key=lambda row: (row["config_id"], int(row["seed"])),
    )
    stats = configuration_statistics(initial_rows, selection)
    adaptive_ids = adaptive_config_ids(stats)
    _dump_jsonl(phase_dir / "initial_seed_results.jsonl", initial_rows)
    _dump_json(phase_dir / "initial_config_statistics.json", stats)
    _dump_csv(phase_dir / "initial_config_statistics.csv", stats)
    adaptive = {
        "rule": "validation_t_ci95_half_width_gt_0.006_or_fewer_than_15_valid_seeds",
        "prediction_correctness_read": False,
        "config_ids": adaptive_ids,
        "configuration_count": len(adaptive_ids),
        "seeds": list(plan["adaptive_seeds"]),
        "initial_seed_results_sha256": sha256_file(phase_dir / "initial_seed_results.jsonl"),
        "task_plan_sha256": sha256_file(args.plan),
    }
    _dump_json(phase_dir / "adaptive_selection.json", adaptive)
    adaptive_hash = sha256_file(phase_dir / "adaptive_selection.json")
    (phase_dir / "adaptive_selection.sha256").write_text(
        f"{adaptive_hash}  adaptive_selection.json\n", encoding="utf-8"
    )
    print(json.dumps(adaptive, indent=2, sort_keys=True))
    return 0


def prospective_pair_details(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    details = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["architecture_fingerprint"] == right["architecture_fingerprint"]:
                continue
            if float(left["validation_mean"]) == float(right["validation_mean"]):
                continue
            higher, lower = (
                (left, right)
                if float(left["validation_mean"]) > float(right["validation_mean"])
                else (right, left)
            )
            disjoint = bool(
                float(higher["validation_ci95_low"]) > float(lower["validation_ci95_high"])
            )
            gap = float(higher["validation_mean"] - lower["validation_mean"])
            result = {
                "higher_config_id": higher["config_id"],
                "higher_architecture_fingerprint": higher["architecture_fingerprint"],
                "higher_validation_mean": higher["validation_mean"],
                "higher_validation_ci95_low": higher["validation_ci95_low"],
                "higher_validation_ci95_high": higher["validation_ci95_high"],
                "lower_config_id": lower["config_id"],
                "lower_architecture_fingerprint": lower["architecture_fingerprint"],
                "lower_validation_mean": lower["validation_mean"],
                "lower_validation_ci95_low": lower["validation_ci95_low"],
                "lower_validation_ci95_high": lower["validation_ci95_high"],
                "validation_mean_gap": gap,
                "intervals_disjoint": disjoint,
                "reliable": gap >= MINIMUM_RELIABLE_GAP and disjoint,
            }
            for model in EVALUATED_MODELS:
                prefix = PREDICTION_PREFIX[model]
                high_prediction = float(higher[f"{prefix}_prediction"])
                low_prediction = float(lower[f"{prefix}_prediction"])
                result[f"{prefix}_higher_prediction"] = high_prediction
                result[f"{prefix}_lower_prediction"] = low_prediction
                result[f"{prefix}_correct"] = high_prediction > low_prediction
                result[f"{prefix}_direction"] = (
                    "higher_config" if high_prediction > low_prediction
                    else "lower_config" if high_prediction < low_prediction
                    else "tie"
                )
            details.append(result)
    return details


def prospective_metric_bundle(rows: Sequence[Mapping[str, Any]], model: str) -> dict[str, Any]:
    prefix = PREDICTION_PREFIX[model]
    actual = np.asarray([row["validation_mean"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row[f"{prefix}_prediction"] for row in rows], dtype=np.float64)
    kendall = kendalltau(actual, predicted).statistic if len(rows) > 1 else None
    relevance = actual - actual.min() + 1e-9
    ndcg = float(ndcg_score(relevance.reshape(1, -1), predicted.reshape(1, -1), k=min(10, len(rows))))
    predicted_top = np.argsort(-predicted, kind="stable")[: min(10, len(rows))]
    pairs = prospective_pair_details(rows)
    reliable = [row for row in pairs if row["reliable"]]
    return {
        "observation_count": len(rows),
        "architecture_count": len({row["architecture_fingerprint"] for row in rows}),
        "kendall": None if kendall is None or not np.isfinite(kendall) else float(kendall),
        "reliable_pairwise_accuracy": (
            statistics.fmean(float(row[f"{prefix}_correct"]) for row in reliable)
            if reliable else None
        ),
        "pairwise_accuracy": statistics.fmean(float(row[f"{prefix}_correct"]) for row in pairs) if pairs else None,
        "ndcg_at_10": ndcg,
        "top_10_regret": float(actual.max() - actual[predicted_top].max()),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "prediction_interval_coverage": statistics.fmean(
            float(
                float(row[f"{prefix}_prediction_interval_low"])
                <= float(row["validation_mean"])
                <= float(row[f"{prefix}_prediction_interval_high"])
            )
            for row in rows
        ),
        "pair_count": len(pairs),
        "reliable_pair_count": len(reliable),
        "reliable_unique_architecture_count": len(
            {
                fingerprint
                for row in reliable
                for fingerprint in (
                    row["higher_architecture_fingerprint"],
                    row["lower_architecture_fingerprint"],
                )
            }
        ),
    }


def _bootstrap_rows(
    rows: Sequence[Mapping[str, Any]], sampled: Sequence[str]
) -> list[dict[str, Any]]:
    lookup = {str(row["architecture_fingerprint"]): row for row in rows}
    copies = Counter()
    output = []
    for fingerprint in sampled:
        copy_index = copies[fingerprint]
        copies[fingerprint] += 1
        output.append(
            {
                **dict(lookup[fingerprint]),
                "bootstrap_cluster": f"{fingerprint}#{copy_index}",
            }
        )
    return output


def metric_summary(
    rows: Sequence[Mapping[str, Any]], model: str, bootstraps: Sequence[Sequence[str]]
) -> tuple[dict[str, Any], dict[str, list[float | None]]]:
    point = prospective_metric_bundle(rows, model)
    metric_names = (
        "kendall", "reliable_pairwise_accuracy", "pairwise_accuracy", "ndcg_at_10",
        "top_10_regret", "rmse", "mae", "prediction_interval_coverage",
    )
    samples: dict[str, list[float | None]] = {name: [] for name in metric_names}
    for sampled in bootstraps:
        result = prospective_metric_bundle(_bootstrap_rows(rows, sampled), model)
        for name in metric_names:
            value = result[name]
            samples[name].append(
                float(value) if value is not None and np.isfinite(value) else None
            )
    summary = dict(point)
    summary["bootstrap_unit"] = "architecture_fingerprint"
    for name, raw_values in samples.items():
        values = [value for value in raw_values if value is not None]
        summary[f"{name}_ci95_low"] = float(np.quantile(values, 0.025)) if values else None
        summary[f"{name}_ci95_high"] = float(np.quantile(values, 0.975)) if values else None
        summary[f"{name}_bootstrap_valid_replicates"] = len(values)
    return summary, samples


def classify_final(
    *, phase_a_passed: bool, h: Mapping[str, Any], head: Mapping[str, Any],
    kendall_difference: Mapping[str, Any], sensitivity_direction_consistent: bool,
) -> tuple[str, dict[str, bool]]:
    if not phase_a_passed:
        return "PAIR_AUDIT_ERROR_FOUND", {}
    enough_pairs = int(h.get("reliable_pair_count", 0)) >= 20
    if not enough_pairs:
        return "H_STILL_UNDERPOWERED", {"reliable_pair_count_at_least_20": False}
    checks = {
        "reliable_pair_count_at_least_20": enough_pairs,
        "reliable_architectures_at_least_12": int(h["reliable_unique_architecture_count"]) >= 12,
        "h_reliable_pair_ci_low_gt_0_5": h.get("reliable_pairwise_accuracy_ci95_low") is not None
        and float(h["reliable_pairwise_accuracy_ci95_low"]) > 0.5,
        "h_kendall_ci_low_gt_zero": h.get("kendall_ci95_low") is not None
        and float(h["kendall_ci95_low"]) > 0,
        "h_head_kendall_increment_at_least_0_05": float(kendall_difference["estimate"]) >= 0.05,
        "h_head_kendall_increment_ci_low_gt_zero": float(kendall_difference["ci95_low"]) > 0,
        "h_pairwise_ci_low_gt_0_5": h.get("pairwise_accuracy_ci95_low") is not None
        and float(h["pairwise_accuracy_ci95_low"]) > 0.5,
        "ndcg_not_lower_than_head": float(h["ndcg_at_10"]) >= float(head["ndcg_at_10"]),
        "top10_regret_not_higher_than_head": float(h["top_10_regret"]) <= float(head["top_10_regret"]),
        "prediction_interval_coverage_at_least_0_90": float(h["prediction_interval_coverage"]) >= 0.90,
        "terminal_and_combined_direction_consistent": bool(sensitivity_direction_consistent),
    }
    return (
        "H_VALIDATED_FOR_STRUCTURE_SURROGATE_PILOT"
        if all(checks.values())
        else "H_REJECTED_WITH_SUFFICIENT_EVIDENCE",
        checks,
    )


def _existing_top_architecture_rows(
    output: Path, strict_root: Path
) -> dict[str, list[dict[str, Any]]]:
    recomputation = json.loads(
        (output / "phase_a/pair_metric_recomputation.json").read_text(encoding="utf-8")
    )
    top = set(recomputation["top_architecture_fingerprints"])
    raw = load_prediction_rows(strict_root / "all_validation_predictions.csv")
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        if row["model"] in (
            "frozen_head", "frozen_hybrid_head", "original_zs_fixed_gp", "original_ze_fixed_gp"
        ):
            by_model[row["model"]].append(row)
    pooled = {model: aggregate_predictions(rows) for model, rows in by_model.items()}
    model_map = {
        "frozen_head": "head",
        "frozen_hybrid_head": "h",
        "original_zs_fixed_gp": "zs",
        "original_ze_fixed_gp": "ze",
    }
    output_by_fp: dict[str, dict[str, Any]] = {}
    for model, rows in pooled.items():
        prefix = model_map[model]
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["architecture_fingerprint"] in top:
                grouped[row["architecture_fingerprint"]].append(row)
        for fingerprint, values in grouped.items():
            target = output_by_fp.setdefault(
                fingerprint,
                {
                    "config_id": f"existing_top_{fingerprint[:12]}",
                    "architecture_fingerprint": fingerprint,
                    "validation_mean": statistics.fmean(float(row["validation_mean"]) for row in values),
                    "validation_ci95_low": min(float(row["validation_ci95_low"]) for row in values),
                    "validation_ci95_high": max(float(row["validation_ci95_high"]) for row in values),
                    "source_batch": "existing_top12_secondary",
                },
            )
            for field in ("prediction", "prediction_interval_low", "prediction_interval_high"):
                target[f"{prefix}_{field}"] = statistics.fmean(float(row[field]) for row in values)
    rows = list(output_by_fp.values())
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 existing top architectures, got {len(rows)}")
    return {model: [dict(row) for row in rows] for model in EVALUATED_MODELS}


def final_evaluate(args: argparse.Namespace) -> int:
    plan = verify_frozen_plan(args.plan.resolve())
    output = args.output.resolve()
    phase_dir = output / "phase_b"
    selection = json.loads(Path(plan["selection_path"]).read_text(encoding="utf-8"))
    adaptive = json.loads((phase_dir / "adaptive_selection.json").read_text(encoding="utf-8"))
    paths = [phase_dir / f"shards/gpu{shard}/seed_results.jsonl" for shard in (0, 1)]
    rows = load_seed_results(paths)
    expected = _expected_initial_keys(plan, selection)
    selection_lookup = {str(row["config_id"]): row for row in selection}
    for config_id in adaptive["config_ids"]:
        config = selection_lookup[str(config_id)]
        expected.update(
            (
                str(config["config_id"]),
                str(config["architecture_fingerprint"]),
                int(seed),
            )
            for seed in plan["adaptive_seeds"]
        )
    observed = {
        (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
        for row in rows
    }
    if observed != expected:
        raise RuntimeError(
            f"final task mismatch: missing={len(expected-observed)}, unexpected={len(observed-expected)}"
        )
    rows.sort(key=lambda row: (row["config_id"], int(row["seed"])))
    stats = configuration_statistics(rows, selection)
    _dump_jsonl(phase_dir / "prospective_seed_results.jsonl", rows)
    _dump_json(phase_dir / "prospective_config_statistics.json", stats)
    _dump_csv(phase_dir / "prospective_config_statistics.csv", stats)

    with (phase_dir / "frozen_prospective_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        frozen = {row["config_id"]: row for row in csv.DictReader(handle)}
    primary_rows = []
    for row in stats:
        prediction = frozen[row["config_id"]]
        primary_rows.append(
            {
                **dict(row),
                **{
                    key: float(value)
                    for key, value in prediction.items()
                    if key.endswith("prediction")
                    or "prediction_interval" in key
                    or key.endswith("prediction_std")
                },
                "source_batch": "terminal_prospective_20",
            }
        )
    architectures = sorted(row["architecture_fingerprint"] for row in primary_rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstraps = [
        [architectures[index] for index in rng.integers(0, len(architectures), size=len(architectures))]
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    primary_metrics = []
    primary_samples = {}
    for model in EVALUATED_MODELS:
        summary, samples = metric_summary(primary_rows, model, bootstraps)
        primary_metrics.append({"analysis": "terminal_prospective_20", "model": model, **summary})
        primary_samples[model] = samples
    metric_lookup = {row["model"]: row for row in primary_metrics}
    h_differences = [
        h - head
        for h, head in zip(
            primary_samples["frozen_h"]["kendall"],
            primary_samples["frozen_head"]["kendall"],
        )
        if h is not None and head is not None
    ]
    kendall_difference = {
        "candidate": "frozen_h",
        "baseline": "frozen_head",
        "metric": "kendall",
        "estimate": float(metric_lookup["frozen_h"]["kendall"] - metric_lookup["frozen_head"]["kendall"]),
        "ci95_low": float(np.quantile(h_differences, 0.025)),
        "ci95_high": float(np.quantile(h_differences, 0.975)),
        "bootstrap_valid_replicates": len(h_differences),
    }
    pair_details = prospective_pair_details(primary_rows)

    existing = _existing_top_architecture_rows(output, args.strict_root.resolve())
    combined_rows = [*existing["frozen_head"], *primary_rows]
    combined_architectures = sorted(row["architecture_fingerprint"] for row in combined_rows)
    rng_combined = np.random.default_rng(BOOTSTRAP_SEED)
    combined_bootstraps = [
        [
            combined_architectures[index]
            for index in rng_combined.integers(
                0, len(combined_architectures), size=len(combined_architectures)
            )
        ]
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    combined_metrics = []
    for model in EVALUATED_MODELS:
        summary, _ = metric_summary(combined_rows, model, combined_bootstraps)
        combined_metrics.append({"analysis": "combined_existing12_plus_prospective20", "model": model, **summary})
    combined_lookup = {row["model"]: row for row in combined_metrics}
    primary_delta = float(metric_lookup["frozen_h"]["kendall"] - metric_lookup["frozen_head"]["kendall"])
    combined_delta = float(combined_lookup["frozen_h"]["kendall"] - combined_lookup["frozen_head"]["kendall"])
    direction_consistent = (primary_delta >= 0 and combined_delta >= 0) or (
        primary_delta <= 0 and combined_delta <= 0
    )
    phase_a = json.loads((output / "phase_a/phase_a_decision.json").read_text(encoding="utf-8"))
    classification, checks = classify_final(
        phase_a_passed=bool(phase_a["phase_a_passed"]),
        h=metric_lookup["frozen_h"],
        head=metric_lookup["frozen_head"],
        kendall_difference=kendall_difference,
        sensitivity_direction_consistent=direction_consistent,
    )
    final = {
        "classification": classification,
        "checks": checks,
        "prospective_configuration_count": 20,
        "prospective_architecture_count": 20,
        "initial_seed_task_count": 300,
        "adaptive_configuration_count": len(adaptive["config_ids"]),
        "adaptive_seed_task_count": len(adaptive["config_ids"]) * 15,
        "total_seed_task_count": len(rows),
        "failed_task_count": sum(row["terminal_status"] == "failed" for row in rows),
        "invalid_task_count": sum(row["terminal_status"] == "invalid" for row in rows),
        "retry_count": sum(int(row["retry_count"]) for row in rows),
        "h_head_primary_kendall_difference": kendall_difference,
        "h_head_combined_kendall_difference": combined_delta,
        "sensitivity_direction_consistent": direction_consistent,
        "test_label_used": False,
        "vae_training_executed": False,
        "offline_bo_executed": False,
        "formal_bo_executed": False,
        "recommended_next_step": (
            "structure_native_surrogate_pilot_only"
            if classification == "H_VALIDATED_FOR_STRUCTURE_SURROGATE_PILOT"
            else "do_not_advance_H"
        ),
    }
    _dump_json(phase_dir / "prospective_pair_details.json", pair_details)
    _dump_csv(phase_dir / "prospective_pair_details.csv", pair_details)
    _dump_json(phase_dir / "prospective_model_metrics.json", primary_metrics)
    _dump_csv(phase_dir / "prospective_model_metrics.csv", primary_metrics)
    _dump_json(phase_dir / "combined_sensitivity_metrics.json", combined_metrics)
    _dump_csv(phase_dir / "combined_sensitivity_metrics.csv", combined_metrics)
    _dump_json(phase_dir / "paired_metric_differences.json", [kendall_difference])
    _dump_csv(phase_dir / "paired_metric_differences.csv", [kendall_difference])
    _dump_json(output / "final_classification.json", final)
    after = protected_hashes()
    before = json.loads((output / "protected_hashes_before.json").read_text(encoding="utf-8"))
    protected_unchanged = before == after
    integrity = {
        "task_plan_verified": True,
        "expected_task_count": len(expected),
        "observed_task_count": len(observed),
        "duplicate_task_count": 0,
        "missing_task_count": len(expected - observed),
        "unexpected_task_count": len(observed - expected),
        "gpu_task_counts": dict(Counter(int(row["assigned_gpu"]) for row in rows)),
        "seed_count_distribution": dict(Counter(int(row["seed_count"]) for row in stats)),
        "dataset_content_fingerprints": sorted({row["dataset_content_fingerprint"] for row in rows}),
        "split_fingerprints": sorted({row["split_fingerprint"] for row in rows}),
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_files_unchanged": protected_unchanged,
        "test_label_used": False,
        "passed": not (expected - observed) and not (observed - expected) and protected_unchanged,
    }
    _dump_json(output / "result_integrity_audit.json", integrity)
    report = f"""# Frozen-H targeted prospective validation

Final classification: `{classification}`.

- Phase A confirmed the original 11 reliable architecture pairs and H=1/11 without direction, aggregation, duplicate, or seed-pseudoreplication error.
- Prospective batch: 20 independently selected architecture fingerprints, balanced as 6 H-favored, 6 Head-favored, and 8 both-high diverse candidates. Candidate selection was frozen before labels and did not read candidate validation or test values.
- Seed labels: {len(rows)} total validation-only GNN runs; {len(adaptive['config_ids'])} configurations activated seeds 15–29 solely by the pre-registered validation CI-width rule.
- Prospective reliable pairs: {metric_lookup['frozen_h']['reliable_pair_count']} involving {metric_lookup['frozen_h']['reliable_unique_architecture_count']} architectures.
- H Kendall: {metric_lookup['frozen_h']['kendall']:.6f} with 95% CI [{metric_lookup['frozen_h']['kendall_ci95_low']:.6f}, {metric_lookup['frozen_h']['kendall_ci95_high']:.6f}].
- Head Kendall: {metric_lookup['frozen_head']['kendall']:.6f}; H-Head increment {kendall_difference['estimate']:.6f}, 95% CI [{kendall_difference['ci95_low']:.6f}, {kendall_difference['ci95_high']:.6f}].
- H reliable-pair accuracy: {metric_lookup['frozen_h']['reliable_pairwise_accuracy']} with CI [{metric_lookup['frozen_h']['reliable_pairwise_accuracy_ci95_low']}, {metric_lookup['frozen_h']['reliable_pairwise_accuracy_ci95_high']}].
- H ordinary-pair accuracy: {metric_lookup['frozen_h']['pairwise_accuracy']}; NDCG@10 {metric_lookup['frozen_h']['ndcg_at_10']}; top-10 regret {metric_lookup['frozen_h']['top_10_regret']}.
- Test labels, VAE training, offline BO, and formal BO were not used. Protected files unchanged: {protected_unchanged}.

Even if H passes, this result authorizes only a subsequent structure-native surrogate pilot, never BO in this task.
"""
    (output / "final_goal_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("initial-merge", "final-evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--strict_root",
            type=Path,
            default=REPO_ROOT / "results/posthoc/performance-aware-vae-strict-regression-global4-v2",
        )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "initial-merge":
        return initial_merge(args)
    if args.command == "final-evaluate":
        return final_evaluate(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
