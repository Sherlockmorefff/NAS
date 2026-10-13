#!/usr/bin/env python3
"""Leakage-safe representation comparison for matched relabel outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import structure_feature_vector


REPRESENTATIONS = ("Zs", "Ze", "D", "H")
METRICS = (
    "late_50_kendall",
    "top_region_kendall",
    "top_pairwise_accuracy",
    "ndcg_at_10",
    "top_10_regret",
    "prediction_interval_coverage",
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("matched relabel input must be a JSON/JSONL list of objects")
    loaded = [dict(row) for row in rows]
    selection_path = path.parent / "matched_relabel_selection.json"
    if selection_path.is_file():
        with selection_path.open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
        regions = {
            str(row["point_id"]): str(row["selection_region"])
            for row in selection
        }
        for row in loaded:
            row["selection_region"] = regions.get(str(row["point_id"]))
    return loaded


def _consistent(group: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = [row.get(field) for row in group]
    canonical = [json.dumps(value, sort_keys=True) for value in values]
    if len(set(canonical)) != 1:
        raise ValueError(f"matched seeds disagree on field {field!r}")
    return values[0]


def aggregate_multiseed(
    rows: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int] = (0, 1, 2),
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["point_id"]), str(row["decoder_policy"]))].append(row)
    aggregated = []
    for (point_id, policy), group in sorted(grouped.items()):
        successful = [
            row
            for row in group
            if bool(row.get("valid", False))
            and row.get("terminal_status", "success") == "success"
        ]
        seeds = [int(row["seed"]) for row in successful]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate successful seeds for {(point_id, policy)}")
        if set(seeds) != set(map(int, expected_seeds)):
            continue
        accuracies = np.asarray([float(row["val_acc"]) for row in successful])
        base = {
            "point_id": point_id,
            "decoder_policy": policy,
            "architecture_version": _consistent(successful, "architecture_version"),
            "architecture_fingerprint": _consistent(successful, "architecture_fingerprint"),
            "operations": _consistent(successful, "operations"),
            "edges": _consistent(successful, "edges"),
            "z_arch": _consistent(successful, "z_arch"),
            "mu_arch": _consistent(successful, "mu_arch"),
            "raw_hp_vector": _consistent(successful, "raw_hp_vector"),
            "condition_mask_vector": _consistent(successful, "condition_mask_vector"),
            "time_index": min(int(row.get("time_index", 0)) for row in successful),
            "selection_region": _consistent(successful, "selection_region"),
            "label": float(accuracies.mean()),
            "seed_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
            "seed_count": len(accuracies),
            "label_mode": "multiseed_mean",
        }
        aggregated.append(base)
    return aggregated


def single_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if not bool(row.get("valid", False)):
            continue
        output.append(
            {
                **row,
                "label": float(row["val_acc"]),
                "label_mode": "single_seed",
                "time_index": int(row.get("time_index", 0)),
            }
        )
    return output


def representation_vector(
    record: Mapping[str, Any], representation: str
) -> tuple[np.ndarray, list[str]]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")
    hp = np.asarray(record["raw_hp_vector"], dtype=np.float64)
    hp_names = [f"raw_hp_{index}" for index in range(hp.size)]
    z = np.asarray(record["z_arch"], dtype=np.float64)
    mu = np.asarray(record["mu_arch"], dtype=np.float64)
    structure, structure_names = structure_feature_vector(
        {"operations": record["operations"], "edges": record["edges"]},
        record["condition_mask_vector"],
    )
    if representation == "Zs":
        return np.concatenate((z, hp)), [*[f"z_arch_{i}" for i in range(z.size)], *hp_names]
    if representation == "Ze":
        return np.concatenate((mu, hp)), [*[f"mu_arch_{i}" for i in range(mu.size)], *hp_names]
    if representation == "D":
        return np.concatenate((structure, hp)), [*structure_names, *hp_names]
    return (
        np.concatenate((mu, structure, hp)),
        [*[f"mu_arch_{i}" for i in range(mu.size)], *structure_names, *hp_names],
    )


def representation_matrix(
    records: Sequence[Mapping[str, Any]], representation: str
) -> tuple[np.ndarray, list[str]]:
    rows = []
    names: list[str] | None = None
    for record in records:
        vector, current_names = representation_vector(record, representation)
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError("representation feature schema changed between records")
        rows.append(vector)
    return np.stack(rows), names or []


def assert_no_architecture_leakage(
    records: Sequence[Mapping[str, Any]], train: Sequence[int], validation: Sequence[int]
) -> None:
    train_groups = {str(records[index]["architecture_fingerprint"]) for index in train}
    validation_groups = {
        str(records[index]["architecture_fingerprint"]) for index in validation
    }
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(f"architecture leakage across train/validation: {sorted(overlap)}")


def grouped_repeated_splits(
    records: Sequence[Mapping[str, Any]],
    repeats: int,
    seed: int,
    validation_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record["architecture_fingerprint"])].append(index)
    group_names = sorted(groups)
    if len(group_names) < 4:
        return []
    validation_count = max(1, min(len(group_names) - 2, round(len(group_names) * validation_fraction)))
    splits = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        shuffled = list(np.asarray(group_names)[rng.permutation(len(group_names))])
        validation_groups = set(shuffled[:validation_count])
        train = [index for group in shuffled[validation_count:] for index in groups[group]]
        validation = [index for group in shuffled[:validation_count] for index in groups[group]]
        assert_no_architecture_leakage(records, train, validation)
        splits.append(
            {"split_id": f"group_{repeat:03d}", "train": train, "validation": validation}
        )
    return splits


def time_prefix_splits(
    records: Sequence[Mapping[str, Any]],
    min_train: int = 8,
    step: int = 5,
    validation_window: int = 10,
) -> list[dict[str, Any]]:
    if min_train < 2 or step <= 0 or validation_window <= 0:
        raise ValueError("invalid time-prefix split settings")
    ordered_indices = sorted(
        range(len(records)), key=lambda index: (int(records[index]["time_index"]), index)
    )
    position = {index: offset for offset, index in enumerate(ordered_indices)}
    group_positions: dict[str, list[int]] = defaultdict(list)
    for index in ordered_indices:
        group_positions[str(records[index]["architecture_fingerprint"])].append(position[index])
    splits = []
    for cutoff in range(min_train, len(records) - 1, step):
        validation_end = min(len(records), cutoff + validation_window)
        train_groups = {
            group for group, positions in group_positions.items() if max(positions) < cutoff
        }
        validation_groups = {
            group
            for group, positions in group_positions.items()
            if min(positions) >= cutoff and min(positions) < validation_end
        }
        train = [index for index in ordered_indices if str(records[index]["architecture_fingerprint"]) in train_groups]
        validation = [
            index
            for index in ordered_indices
            if str(records[index]["architecture_fingerprint"]) in validation_groups
            and position[index] < validation_end
        ]
        if len(train) < 4 or len(validation) < 2:
            continue
        assert_no_architecture_leakage(records, train, validation)
        splits.append(
            {"split_id": f"prefix_{cutoff:04d}", "train": train, "validation": validation}
        )
    return splits


def standardize_train_validation(
    x_train: np.ndarray, x_validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit all standardization statistics on training observations only."""

    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_validation), scaler


def _safe_kendall(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if actual.size < 2 or np.unique(actual).size < 2 or np.unique(predicted).size < 2:
        return None
    value = kendalltau(actual, predicted).statistic
    return None if not np.isfinite(value) else float(value)


def _pairwise_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    correct = total = 0
    for left in range(actual.size):
        for right in range(left + 1, actual.size):
            actual_sign = np.sign(actual[left] - actual[right])
            predicted_sign = np.sign(predicted[left] - predicted[right])
            if actual_sign == 0:
                continue
            correct += int(actual_sign == predicted_sign)
            total += 1
    return correct / total if total else None


def _ndcg(actual: np.ndarray, predicted: np.ndarray, k: int = 10) -> float | None:
    if actual.size == 0:
        return None
    relevance = actual - actual.min()
    k = min(k, actual.size)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    predicted_order = np.argsort(-predicted, kind="stable")[:k]
    ideal_order = np.argsort(-actual, kind="stable")[:k]
    dcg = float(np.sum(relevance[predicted_order] * discounts))
    ideal = float(np.sum(relevance[ideal_order] * discounts))
    return 1.0 if ideal == 0.0 else dcg / ideal


def decision_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    predicted_std: np.ndarray,
    times: np.ndarray,
) -> dict[str, float | None]:
    late_order = np.argsort(times, kind="stable")[-min(50, actual.size) :]
    top_threshold = float(np.quantile(actual, 0.8))
    top = np.flatnonzero(actual >= top_threshold)
    predicted_top = np.argsort(-predicted, kind="stable")[: min(10, actual.size)]
    coverage = np.mean(
        (actual >= predicted - 1.96 * predicted_std)
        & (actual <= predicted + 1.96 * predicted_std)
    )
    return {
        "late_50_kendall": _safe_kendall(actual[late_order], predicted[late_order]),
        "top_region_kendall": _safe_kendall(actual[top], predicted[top]),
        "top_pairwise_accuracy": _pairwise_accuracy(actual[top], predicted[top]),
        "ndcg_at_10": _ndcg(actual, predicted, 10),
        "top_10_regret": float(actual.max() - actual[predicted_top].max()),
        "prediction_interval_coverage": float(coverage),
    }


def _bootstrap_ci(values: Sequence[float], seed: int, replicates: int) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [None, None]
    if array.size == 1:
        return [float(array[0]), float(array[0])]
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, array.size, size=(replicates, array.size))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate_protocol(
    records: Sequence[Mapping[str, Any]],
    splits: Sequence[Mapping[str, Any]],
    protocol: str,
    policy: str,
    label_mode: str,
    seed: int,
    bootstrap_replicates: int,
    *,
    return_split_metrics: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    split_metrics: dict[tuple[str, str], list[float]] = defaultdict(list)
    predictions = []
    split_metric_rows = []
    for representation in REPRESENTATIONS:
        x, _names = representation_matrix(records, representation)
        for split_number, split in enumerate(splits):
            train = np.asarray(split["train"], dtype=int)
            validation = np.asarray(split["validation"], dtype=int)
            assert_no_architecture_leakage(records, train, validation)
            x_train, x_validation, _scaler = standardize_train_validation(
                x[train], x[validation]
            )
            y_train = np.asarray([records[index]["label"] for index in train], dtype=float)
            y_validation = np.asarray(
                [records[index]["label"] for index in validation], dtype=float
            )
            model = BayesianRidge(compute_score=False)
            model.fit(x_train, y_train)
            predicted, predicted_std = model.predict(x_validation, return_std=True)
            times = np.asarray(
                [records[index]["time_index"] for index in validation], dtype=int
            )
            metrics = decision_metrics(y_validation, predicted, predicted_std, times)
            for metric, value in metrics.items():
                if value is not None:
                    split_metrics[(representation, metric)].append(float(value))
                    split_metric_rows.append(
                        {
                            "policy": policy,
                            "label_mode": label_mode,
                            "protocol": protocol,
                            "representation": representation,
                            "split_id": split["split_id"],
                            "metric": metric,
                            "value": float(value),
                        }
                    )
            for local_index, record_index in enumerate(validation):
                predictions.append(
                    {
                        "policy": policy,
                        "label_mode": label_mode,
                        "protocol": protocol,
                        "representation": representation,
                        "split_id": split["split_id"],
                        "point_id": records[record_index]["point_id"],
                        "architecture_fingerprint": records[record_index]["architecture_fingerprint"],
                        "actual": float(y_validation[local_index]),
                        "predicted": float(predicted[local_index]),
                        "predicted_std": float(predicted_std[local_index]),
                    }
                )
    summaries = []
    for representation in REPRESENTATIONS:
        for metric in METRICS:
            values = split_metrics[(representation, metric)]
            ci = _bootstrap_ci(
                values,
                seed + 1009 * REPRESENTATIONS.index(representation) + 97 * METRICS.index(metric),
                bootstrap_replicates,
            )
            summaries.append(
                {
                    "policy": policy,
                    "label_mode": label_mode,
                    "protocol": protocol,
                    "representation": representation,
                    "metric": metric,
                    "estimate": float(np.mean(values)) if values else None,
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "split_count": len(values),
                }
            )
    if return_split_metrics:
        return summaries, predictions, split_metric_rows
    return summaries, predictions


def diagnostic_conclusion(
    metrics: Sequence[Mapping[str, Any]], aggregated: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    seed_stds = [float(row["seed_std"]) for row in aggregated if int(row["seed_count"]) > 1]
    seed_ci_half_widths = [
        1.96 * float(row["seed_std"]) / math.sqrt(int(row["seed_count"]))
        for row in aggregated
        if int(row["seed_count"]) > 1
    ]
    median_seed_std = float(np.median(seed_stds)) if seed_stds else None
    median_seed_ci = float(np.median(seed_ci_half_widths)) if seed_ci_half_widths else None
    top_rows = [row for row in aggregated if row.get("selection_region") == "top20"]
    top_seed_stds = [
        float(row["seed_std"]) for row in top_rows if int(row["seed_count"]) > 1
    ]
    top_seed_ci_half_widths = [
        1.96 * float(row["seed_std"]) / math.sqrt(int(row["seed_count"]))
        for row in top_rows
        if int(row["seed_count"]) > 1
    ]
    top_median_seed_std = (
        float(np.median(top_seed_stds)) if top_seed_stds else None
    )
    top_median_seed_ci = (
        float(np.median(top_seed_ci_half_widths))
        if top_seed_ci_half_widths
        else None
    )
    reference_std = (
        top_median_seed_std if top_median_seed_std is not None else median_seed_std
    )
    reference_ci = (
        top_median_seed_ci if top_median_seed_ci is not None else median_seed_ci
    )
    noise_comparable = bool(
        (reference_std is not None and reference_std >= 0.006)
        or (reference_ci is not None and reference_ci >= 0.006)
    )
    primary = {
        (str(row["policy"]), str(row["representation"]), str(row["metric"])): row
        for row in metrics
        if row["label_mode"] == "multiseed_mean"
        and row["protocol"] == "group_repeated"
    }
    candidates = []
    for policy in sorted({key[0] for key in primary}):
        baseline = {
            metric: primary.get((policy, "Zs", metric)) for metric in METRICS
        }
        for representation in ("Ze", "D", "H"):
            current = {
                metric: primary.get((policy, representation, metric)) for metric in METRICS
            }
            required = [
                baseline["late_50_kendall"],
                baseline["ndcg_at_10"],
                baseline["top_10_regret"],
                current["top_region_kendall"],
                current["late_50_kendall"],
                current["ndcg_at_10"],
                current["top_10_regret"],
            ]
            if any(row is None or row["estimate"] is None for row in required):
                continue
            passed = (
                current["top_region_kendall"]["ci95_low"] is not None
                and current["top_region_kendall"]["ci95_low"] > 0.0
                and current["late_50_kendall"]["estimate"]
                - baseline["late_50_kendall"]["estimate"]
                >= 0.05
                and current["ndcg_at_10"]["estimate"]
                >= baseline["ndcg_at_10"]["estimate"]
                and current["top_10_regret"]["estimate"]
                <= baseline["top_10_regret"]["estimate"]
            )
            if passed:
                candidates.append({"policy": policy, "representation": representation})
    if noise_comparable:
        classification = "NOISE_DOMINATED"
        reason = "multi-seed variability or its 95% interval is comparable to 0.006"
    elif candidates:
        classification = "REPRESENTATION_CANDIDATE"
        reason = "at least one representation passes every pre-specified ranking gate"
    else:
        classification = "INSUFFICIENT_EVIDENCE"
        reason = "sample size, unavailable metrics, or conflicting representation gates"
    return {
        "classification": classification,
        "reason": reason,
        "median_multiseed_std": median_seed_std,
        "median_multiseed_ci95_half_width": median_seed_ci,
        "top20_observations": len(top_rows),
        "top20_median_multiseed_std": top_median_seed_std,
        "top20_median_multiseed_ci95_half_width": top_median_seed_ci,
        "noise_reference_gap": 0.006,
        "passing_candidates": candidates,
        "formal_bo_modified": False,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    output: Path,
    conclusion: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    lines = [
        "# Matched representation diagnostic",
        "",
        f"Conclusion: `{conclusion['classification']}`",
        "",
        str(conclusion["reason"]),
        "",
        "This report is diagnostic only. It does not modify or select the formal BO surrogate.",
        "",
        "## Multi-seed noise",
        "",
        f"All-observation median std: {conclusion['median_multiseed_std']}",
        "",
        f"Top-20% median std: {conclusion['top20_median_multiseed_std']}",
        "",
        f"Top-20% median 95% CI half-width: {conclusion['top20_median_multiseed_ci95_half_width']}",
        "",
        f"Reference gap: {conclusion['noise_reference_gap']}",
        "",
        "## Primary grouped, multi-seed metrics",
        "",
        "| Policy | Representation | Metric | Estimate | 95% CI | Splits |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in metrics:
        if row["label_mode"] != "multiseed_mean" or row["protocol"] != "group_repeated":
            continue
        estimate = "NA" if row["estimate"] is None else f"{row['estimate']:.6f}"
        interval = (
            "NA"
            if row["ci95_low"] is None
            else f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]"
        )
        lines.append(
            f"| {row['policy']} | {row['representation']} | {row['metric']} | "
            f"{estimate} | {interval} | {row['split_count']} |"
        )
    lines.extend(["", "## Dataset", "", f"Aggregated observations: {metadata['aggregated_observations']}"])
    (output / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare matched NAS representations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected_seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="exact successful seed set required for each point-policy aggregate",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--repeated_splits", type=int, default=20)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--prefix_min_train", type=int, default=8)
    parser.add_argument("--prefix_step", type=int, default=5)
    parser.add_argument("--prefix_validation_window", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeated_splits <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("split and bootstrap counts must be positive")
    raw = _load_rows(args.input)
    expected_seeds = sorted(set(map(int, args.expected_seeds)))
    if not expected_seeds or any(seed < 0 for seed in expected_seeds):
        raise ValueError("--expected_seeds must contain non-negative integers")
    aggregated = aggregate_multiseed(raw, expected_seeds)
    single = single_seed_rows(raw)
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for label_mode, dataset in (("multiseed_mean", aggregated),):
        for policy in sorted({str(row["decoder_policy"]) for row in dataset}):
            records = sorted(
                [row for row in dataset if row["decoder_policy"] == policy],
                key=lambda row: (int(row["time_index"]), str(row["point_id"]), int(row.get("seed", -1))),
            )
            protocols = {
                "time_prefix": time_prefix_splits(
                    records,
                    min_train=args.prefix_min_train,
                    step=args.prefix_step,
                    validation_window=args.prefix_validation_window,
                ),
                "group_repeated": grouped_repeated_splits(
                    records, args.repeated_splits, args.seed
                ),
            }
            for protocol, splits in protocols.items():
                metrics, predictions = evaluate_protocol(
                    records,
                    splits,
                    protocol,
                    policy,
                    label_mode,
                    args.seed,
                    args.bootstrap_replicates,
                )
                all_metrics.extend(metrics)
                all_predictions.extend(predictions)
    conclusion = diagnostic_conclusion(all_metrics, aggregated)
    metadata = {
        "input": str(args.input.resolve()),
        "raw_single_seed_observations": len(single),
        "single_seed_usage": "training_noise_only",
        "selection_region_counts": {
            region: sum(row.get("selection_region") == region for row in aggregated)
            for region in sorted(
                {str(row.get("selection_region")) for row in aggregated}
            )
        },
        "aggregated_observations": len(aggregated),
        "policies_analyzed_separately": sorted({row["decoder_policy"] for row in aggregated}),
        "representations": list(REPRESENTATIONS),
        "standardization": "training_split_only",
        "grouping_key": "architecture_fingerprint",
        "seed": args.seed,
        "expected_seeds": expected_seeds,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "representation_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "metrics": all_metrics, "conclusion": conclusion},
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    _write_csv(output / "representation_metrics.csv", all_metrics)
    _write_csv(output / "representation_predictions.csv", all_predictions)
    with (output / "diagnostic_conclusion.json").open("w", encoding="utf-8") as handle:
        json.dump(conclusion, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    _write_report(output, conclusion, all_metrics, metadata)
    print(json.dumps(conclusion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
