#!/usr/bin/env python3
"""Build and freeze the validation-only structure-native surrogate pilot.

The command has two explicit stages. ``phase0`` independently replays the
previous targeted-H result without training. ``prepare`` builds the 80-
architecture development set, trains frozen-encoder predictor candidates,
selects 24 history candidates without labels or predictions, and freezes every
prediction and label task before the prospective Cora labels are read.

This module never imports a BO entry point, never trains a VAE, never uses a
latent-distance GP, and rejects test-named label fields recursively.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import (
    architecture_fingerprint,
    config_to_igraph,
    load_joint_vae,
    normalize_architecture_config,
    structure_feature_vector,
)
from analyse.h_targeted_reliable_pair_validation import (
    aggregate_predictions,
    frozen_head_h_predictions,
    load_prediction_rows,
    sha256_file,
)
from analyse.performance_aware_vae_pilot import contains_test_field, set_all_seeds, stable_hash
from analyse.performance_aware_vae_strict_regression import StrictPerformanceHead
from analyse.vae_label_expansion import hp_fingerprint


FORMAT_VERSION = "structure_native_surrogate_pilot_global4_v1"
OUTPUT_RELATIVE = "results/posthoc/structure-native-surrogate-pilot-global4-v1"
H_TARGETED_RELATIVE = "results/posthoc/h-targeted-reliable-pair-validation-global4-v1"
STRICT_RELATIVE = "results/posthoc/performance-aware-vae-strict-regression-global4-v2"
UNIFIED_RELATIVE = "results/posthoc/vae-label-expansion-global4-v1"
QUARANTINED_RELATIVE = "results/posthoc/structure-native-surrogate-pilot-global4-v1-invalid-freeze-audit-20260823"
HISTORY_GLOB = "results/cora-original-ceiling-hunt-v1/original/search_seed*/history_final.json"
MODEL_NAMES = (
    "original_frozen_head",
    "original_frozen_h",
    "refit_head",
    "d_structure",
    "h_structure",
    "head_h_mean",
    "head_h_conservative",
    "head_h_weighted",
)
TRAINED_MODELS = ("refit_head", "d_structure", "h_structure")
ENSEMBLE_MODELS = ("head_h_mean", "head_h_conservative", "head_h_weighted")
TRAINING_SEEDS = (0, 1, 2)
WEIGHT_MODES = ("architecture_balanced", "architecture_balanced_uncertainty")
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
LEARNING_RATES = (0.003, 0.01)
OUTER_REPEATS = 20
OUTER_VALIDATION_FRACTION = 0.25
INNER_VALIDATION_FRACTION = 0.20
MAXIMUM_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 30
MINIMUM_RELIABLE_GAP = 0.006
PHASE0_BOOTSTRAPS = 2000
FINAL_BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 20260823
INITIAL_SEEDS = tuple(range(15))
ADAPTIVE_SEEDS = tuple(range(15, 30))
DATASET_FINGERPRINT = "373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7"
SPLIT_FINGERPRINT = "bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09"

PROTECTED_RELATIVE_PATHS = (
    "bo_phase4.py",
    "eval_utils.py",
    "checkpoints/joint_model_pipeline_global4_best.pth",
    f"{UNIFIED_RELATIVE}/task_plan.json",
    f"{UNIFIED_RELATIVE}/unified_training_data_no_test.json",
    f"{UNIFIED_RELATIVE}/unified_seed_level_audit.jsonl",
    f"{STRICT_RELATIVE}/task_plan.json",
    f"{STRICT_RELATIVE}/fixed_split_manifest.json",
    f"{STRICT_RELATIVE}/all_validation_predictions.csv",
    f"{STRICT_RELATIVE}/final_classification.json",
    f"{H_TARGETED_RELATIVE}/phase_b/targeted_task_plan.json",
    f"{H_TARGETED_RELATIVE}/phase_b/targeted_selection.json",
    f"{H_TARGETED_RELATIVE}/phase_b/frozen_prospective_predictions.csv",
    f"{H_TARGETED_RELATIVE}/final_classification.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fields})
    temporary.replace(path)


def write_sha256_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def protected_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path.resolve())] = sha256_file(path)
    return result


def assert_no_test_fields(value: Any, *, context: str) -> None:
    if contains_test_field(value):
        raise ValueError(f"test-named field is forbidden in {context}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _architecture_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("bootstrap_id", row["architecture_fingerprint"]))].append(row)
    output = []
    for cluster, values in sorted(groups.items()):
        output.append(
            {
                "cluster": cluster,
                "architecture_fingerprint": str(values[0]["architecture_fingerprint"]),
                "validation_mean": statistics.fmean(float(row["validation_mean"]) for row in values),
                "validation_ci95_low": statistics.fmean(float(row["validation_ci95_low"]) for row in values),
                "validation_ci95_high": statistics.fmean(float(row["validation_ci95_high"]) for row in values),
                "prediction": statistics.fmean(float(row["prediction"]) for row in values),
                "prediction_interval_low": statistics.fmean(float(row["prediction_interval_low"]) for row in values),
                "prediction_interval_high": statistics.fmean(float(row["prediction_interval_high"]) for row in values),
                "selection_stratum": str(values[0].get("selection_stratum", "all")),
            }
        )
    return output


def interval_score(actual: float, low: float, high: float, alpha: float) -> float:
    score = high - low
    if actual < low:
        score += (2.0 / alpha) * (low - actual)
    elif actual > high:
        score += (2.0 / alpha) * (actual - high)
    return float(score)


def metric_bundle(rows: Sequence[Mapping[str, Any]], *, interval_alpha: float) -> dict[str, Any]:
    arch = _architecture_rows(rows)
    if not arch:
        raise ValueError("metric bundle requires at least one architecture")
    actual = np.asarray([row["validation_mean"] for row in arch], dtype=np.float64)
    predicted = np.asarray([row["prediction"] for row in arch], dtype=np.float64)
    tau = kendalltau(actual, predicted).statistic if len(arch) > 1 else None
    pair_correct: list[float] = []
    reliable_correct: list[float] = []
    reliable_architectures: set[str] = set()
    within_pair_correct: list[float] = []
    cross_pair_correct: list[float] = []
    for left in range(len(arch)):
        for right in range(left + 1, len(arch)):
            delta = float(actual[left] - actual[right])
            if delta == 0:
                continue
            correct = float((predicted[left] - predicted[right]) * delta > 0)
            pair_correct.append(correct)
            if arch[left]["selection_stratum"] == arch[right]["selection_stratum"]:
                within_pair_correct.append(correct)
            else:
                cross_pair_correct.append(correct)
            disjoint = (
                float(arch[left]["validation_ci95_low"]) > float(arch[right]["validation_ci95_high"])
                or float(arch[right]["validation_ci95_low"]) > float(arch[left]["validation_ci95_high"])
            )
            if abs(delta) >= MINIMUM_RELIABLE_GAP and disjoint:
                reliable_correct.append(correct)
                reliable_architectures.update((arch[left]["architecture_fingerprint"], arch[right]["architecture_fingerprint"]))
    relevance = actual - actual.min() + 1e-9
    result: dict[str, Any] = {
        "architecture_count": len(arch),
        "kendall": None if tau is None or not np.isfinite(tau) else float(tau),
        "pairwise_accuracy": statistics.fmean(pair_correct) if pair_correct else None,
        "pair_count": len(pair_correct),
        "reliable_pairwise_accuracy": statistics.fmean(reliable_correct) if reliable_correct else None,
        "reliable_pair_count": len(reliable_correct),
        "reliable_architecture_count": len(reliable_architectures),
        "within_stratum_pairwise_accuracy": statistics.fmean(within_pair_correct) if within_pair_correct else None,
        "within_stratum_pair_count": len(within_pair_correct),
        "cross_stratum_pairwise_accuracy": statistics.fmean(cross_pair_correct) if cross_pair_correct else None,
        "cross_stratum_pair_count": len(cross_pair_correct),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mae": float(np.mean(np.abs(predicted - actual))),
    }
    for k in (5, 10):
        result[f"ndcg_at_{k}"] = float(ndcg_score(relevance.reshape(1, -1), predicted.reshape(1, -1), k=min(k, len(arch))))
    for k in (1, 3, 5, 10):
        selected = np.argsort(-predicted, kind="stable")[: min(k, len(arch))]
        result[f"regret_at_{k}"] = float(actual.max() - actual[selected].max())
    lows = np.asarray([row["prediction_interval_low"] for row in arch], dtype=np.float64)
    highs = np.asarray([row["prediction_interval_high"] for row in arch], dtype=np.float64)
    widths = highs - lows
    result.update(
        {
            "prediction_interval_coverage": float(np.mean((lows <= actual) & (actual <= highs))),
            "prediction_interval_mean_width": float(widths.mean()),
            "prediction_interval_median_width": float(np.median(widths)),
            "prediction_interval_score": float(np.mean([interval_score(float(y), float(lo), float(hi), interval_alpha) for y, lo, hi in zip(actual, lows, highs)])),
            "prediction_interval_out_of_bounds_fraction": float(np.mean((lows < 0.0) | (highs > 1.0))),
        }
    )
    top_count = max(1, math.ceil(len(arch) / 4))
    top = np.argsort(-predicted, kind="stable")[:top_count]
    median = float(np.median(actual))
    false_positive_indices = [int(index) for index in top if actual[index] < median]
    result["head_top_quartile_false_positive_count"] = len(false_positive_indices)
    result["head_top_quartile_false_positive_fingerprints"] = [arch[index]["architecture_fingerprint"] for index in false_positive_indices]
    return result


def _bootstrap_rows(rows: Sequence[Mapping[str, Any]], sample: Sequence[tuple[str, int]]) -> list[dict[str, Any]]:
    lookup: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        lookup[str(row["architecture_fingerprint"])].append(row)
    output = []
    for copy_index, (fingerprint, draw_index) in enumerate(sample):
        bootstrap_id = f"{fingerprint}#{draw_index}#{copy_index}"
        output.extend({**dict(row), "bootstrap_id": bootstrap_id} for row in lookup[fingerprint])
    return output


def stratified_bootstrap_samples(
    rows: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> list[list[tuple[str, int]]]:
    fingerprint_strata: dict[str, str] = {}
    for row in rows:
        fingerprint = str(row["architecture_fingerprint"])
        stratum = str(row.get("selection_stratum", "all"))
        previous = fingerprint_strata.setdefault(fingerprint, stratum)
        if previous != stratum:
            raise ValueError(f"architecture {fingerprint} appears in multiple bootstrap strata")
    strata: dict[str, list[str]] = defaultdict(list)
    for fingerprint, stratum in fingerprint_strata.items():
        strata[stratum].append(fingerprint)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        current: list[tuple[str, int]] = []
        for stratum in sorted(strata):
            ordered = sorted(strata[stratum])
            draws = rng.integers(0, len(ordered), size=len(ordered))
            current.extend((ordered[int(index)], position) for position, index in enumerate(draws))
        samples.append(current)
    return samples


def summarize_with_bootstrap(
    rows: Sequence[Mapping[str, Any]], samples: Sequence[Sequence[tuple[str, int]]], *, interval_alpha: float
) -> tuple[dict[str, Any], dict[str, list[float | None]]]:
    point = metric_bundle(rows, interval_alpha=interval_alpha)
    metric_names = [key for key, value in point.items() if isinstance(value, (int, float)) and not key.endswith("count")]
    values: dict[str, list[float | None]] = {key: [] for key in metric_names}
    for sample in samples:
        boot = metric_bundle(_bootstrap_rows(rows, sample), interval_alpha=interval_alpha)
        for key in metric_names:
            value = boot.get(key)
            values[key].append(float(value) if value is not None and np.isfinite(value) else None)
    result = dict(point)
    result["bootstrap_replicates"] = len(samples)
    result["bootstrap_unit"] = "stratified_architecture_fingerprint"
    for key, raw in values.items():
        valid = [value for value in raw if value is not None]
        result[f"{key}_ci95_low"] = float(np.quantile(valid, 0.025)) if valid else None
        result[f"{key}_ci95_high"] = float(np.quantile(valid, 0.975)) if valid else None
        result[f"{key}_bootstrap_valid_replicates"] = len(valid)
    return result, values


def _phase0_rows() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    h_root = REPO_ROOT / H_TARGETED_RELATIVE
    predictions = load_csv(h_root / "phase_b/frozen_prospective_predictions.csv")
    statistics_rows = {str(row["config_id"]): row for row in load_json(h_root / "phase_b/prospective_config_statistics.json")}
    selection = {str(row["config_id"]): row for row in load_json(h_root / "phase_b/targeted_selection.json")}
    prefixes = {
        "original_frozen_head": "head",
        "original_frozen_h": "h",
        "original_zs": "zs",
        "original_ze": "ze",
    }
    base = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        config_id = str(prediction["config_id"])
        stats = statistics_rows[config_id]
        selected = selection[config_id]
        identity = {
            "config_id": config_id,
            "architecture_fingerprint": str(prediction["architecture_fingerprint"]),
            "selection_stratum": str(prediction["selection_stratum"]),
            "validation_mean": float(stats["validation_mean"]),
            "validation_ci95_low": float(stats["validation_ci95_low"]),
            "validation_ci95_high": float(stats["validation_ci95_high"]),
            "source_search_seed": int(selected["source_search_seed"]),
            "source_step": int(selected["source_step"]),
        }
        base.append(identity)
        for model, prefix in prefixes.items():
            by_model[model].append(
                {
                    **identity,
                    "model": model,
                    "prediction": float(prediction[f"{prefix}_prediction"]),
                    "prediction_interval_low": float(prediction[f"{prefix}_prediction_interval_low"]),
                    "prediction_interval_high": float(prediction[f"{prefix}_prediction_interval_high"]),
                }
            )
    return base, dict(by_model)


def _combined_old_new_kendall(new_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    strict_root = REPO_ROOT / STRICT_RELATIVE
    unified = load_json(REPO_ROOT / UNIFIED_RELATIVE / "unified_training_data_no_test.json")
    prediction_rows = load_prediction_rows(strict_root / "all_validation_predictions.csv")
    mapping = {"original_frozen_head": "frozen_head", "original_frozen_h": "frozen_hybrid_head"}
    output: dict[str, float] = {}
    architecture_means: dict[str, list[float]] = defaultdict(list)
    for row in unified:
        architecture_means[str(row["architecture_fingerprint"])].append(float(row["validation_mean"]))
    top = set(
        fingerprint
        for fingerprint, _values in sorted(
            architecture_means.items(),
            key=lambda item: (-statistics.fmean(item[1]), item[0]),
        )[:12]
    )
    for output_name, source_model in mapping.items():
        filtered = [
            row for row in prediction_rows
            if row["model"] == source_model
            and row["dataset_variant"] == "full"
            and row["protocol"] == "group_repeated"
        ]
        aggregated = aggregate_predictions(filtered)
        old_rows = [row for row in aggregated if row["architecture_fingerprint"] in top]
        combined = [*old_rows, *[dict(row) for row in new_rows[output_name]]]
        output[output_name] = float(metric_bundle(combined, interval_alpha=0.05)["kendall"])
    return output


def run_phase0(output: Path) -> dict[str, Any]:
    phase = output / "phase_0"
    phase.mkdir(parents=True, exist_ok=False)
    base, by_model = _phase0_rows()
    counts = Counter(row["selection_stratum"] for row in base)
    expected_counts = Counter({"both_high_diverse": 8, "h_high_h_over_head": 6, "head_high_head_over_h": 6})
    if counts != expected_counts:
        raise RuntimeError(f"Phase 0 stratum mismatch: {counts}")
    samples = stratified_bootstrap_samples(base, replicates=PHASE0_BOOTSTRAPS, seed=BOOTSTRAP_SEED)
    combined_rows = []
    sample_values: dict[str, dict[str, list[float | None]]] = {}
    for model, rows in by_model.items():
        summary, values = summarize_with_bootstrap(rows, samples, interval_alpha=0.05)
        summary["model"] = model
        summary["scope"] = "three_strata_combined"
        combined_rows.append(summary)
        sample_values[model] = values
    stratum_rows = []
    equal_weight_rows = []
    for model, rows in by_model.items():
        per_stratum = []
        for stratum in sorted(expected_counts):
            subset = [row for row in rows if row["selection_stratum"] == stratum]
            subset_samples = stratified_bootstrap_samples(subset, replicates=PHASE0_BOOTSTRAPS, seed=BOOTSTRAP_SEED)
            summary, _values = summarize_with_bootstrap(subset, subset_samples, interval_alpha=0.05)
            summary.update({"model": model, "scope": "within_stratum", "selection_stratum": stratum})
            stratum_rows.append(summary)
            per_stratum.append(summary)
        equal = {"model": model, "scope": "three_strata_equal_weight"}
        for metric in ("kendall", "pairwise_accuracy", "reliable_pairwise_accuracy", "ndcg_at_10", "regret_at_1", "regret_at_3", "regret_at_5", "regret_at_10", "rmse", "mae"):
            values = [float(row[metric]) for row in per_stratum if row.get(metric) is not None]
            equal[metric] = statistics.fmean(values) if values else None
        equal_weight_rows.append(equal)
    combined_kendall = _combined_old_new_kendall(by_model)
    indexed = {row["model"]: row for row in combined_rows}
    reference = {
        "new20_head_kendall": indexed["original_frozen_head"]["kendall"],
        "new20_h_kendall": indexed["original_frozen_h"]["kendall"],
        "new20_head_reliable_pairwise_accuracy": indexed["original_frozen_head"]["reliable_pairwise_accuracy"],
        "new20_h_reliable_pairwise_accuracy": indexed["original_frozen_h"]["reliable_pairwise_accuracy"],
        "new20_h_minus_head_kendall": indexed["original_frozen_h"]["kendall"] - indexed["original_frozen_head"]["kendall"],
        "combined_old12_new20_head_kendall": combined_kendall["original_frozen_head"],
        "combined_old12_new20_h_kendall": combined_kendall["original_frozen_h"],
        "new20_label_range": max(row["validation_mean"] for row in base) - min(row["validation_mean"] for row in base),
    }
    expected = {
        "new20_head_kendall": 0.11578947368421053,
        "new20_h_kendall": 0.631578947368421,
        "new20_head_reliable_pairwise_accuracy": 0.5668789808917197,
        "new20_h_reliable_pairwise_accuracy": 0.8662420382165605,
        "new20_h_minus_head_kendall": 0.5157894736842105,
        "combined_old12_new20_head_kendall": -0.125,
        "combined_old12_new20_h_kendall": 0.036290322580645164,
    }
    mismatches = {key: {"expected": value, "observed": reference[key]} for key, value in expected.items() if not math.isclose(float(reference[key]), value, rel_tol=0.0, abs_tol=1e-10)}
    decision = "INPUT_RESULT_MISMATCH" if mismatches else "PHASE_0_PASSED"
    dump_json(phase / "stratified_metrics.json", stratum_rows)
    dump_csv(phase / "stratified_metrics.csv", stratum_rows)
    dump_json(phase / "combined_metrics.json", combined_rows)
    dump_csv(phase / "combined_metrics.csv", combined_rows)
    dump_json(phase / "equal_weight_strata_metrics.json", equal_weight_rows)
    dump_csv(phase / "equal_weight_strata_metrics.csv", equal_weight_rows)
    dump_json(phase / "reference_recomputation.json", {"decision": decision, "reference": reference, "expected": expected, "mismatches": mismatches})
    dump_json(phase / "phase_0_decision.json", {"classification": decision, "stratum_counts": dict(counts), "bootstrap_replicates": PHASE0_BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED, "shared_stratified_architecture_indices": True})
    report = f"""# Phase 0 supplemental analysis

Decision: **{decision}**

- Frozen strata: 8 `both_high_diverse`, 6 `h_high_h_over_head`, 6 `head_high_head_over_h`.
- Bootstrap: {PHASE0_BOOTSTRAPS} shared, architecture-level draws preserving the 8/6/6 ratio.
- New-20 Head/H Kendall: {reference['new20_head_kendall']:.6f} / {reference['new20_h_kendall']:.6f}.
- New-20 Head/H reliable-pair accuracy: {reference['new20_head_reliable_pairwise_accuracy']:.6f} / {reference['new20_h_reliable_pairwise_accuracy']:.6f}.
- H-minus-Head Kendall: {reference['new20_h_minus_head_kendall']:.6f}.
- Old-12 plus new-20 Head/H Kendall: {reference['combined_old12_new20_head_kendall']:.6f} / {reference['combined_old12_new20_h_kendall']:.6f}.
- New-label range: {reference['new20_label_range']:.6f}.
- Head/H mean interval widths: {indexed['original_frozen_head']['prediction_interval_mean_width']:.6f} / {indexed['original_frozen_h']['prediction_interval_mean_width']:.6f}.
- Head/H out-of-[0,1] interval fractions: {indexed['original_frozen_head']['prediction_interval_out_of_bounds_fraction']:.3f} / {indexed['original_frozen_h']['prediction_interval_out_of_bounds_fraction']:.3f}.

Coverage is reported together with width, interval score, and boundary violations. One-hundred-percent coverage is not interpreted as effective uncertainty calibration.
"""
    (phase / "phase_0_report.md").write_text(report, encoding="utf-8")
    return {"classification": decision, "reference": reference}


def architecture_balanced_weights(
    rows: Sequence[Mapping[str, Any]], *, uncertainty_weighted: bool
) -> np.ndarray:
    """Return weights whose total is identical for every architecture."""

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["architecture_fingerprint"])].append(index)
    weights = np.zeros(len(rows), dtype=np.float64)
    for indices in groups.values():
        if uncertainty_weighted:
            inverse = np.asarray(
                [
                    1.0
                    / max(float(rows[index].get("validation_sample_std", 0.0)) ** 2, 1e-6)
                    for index in indices
                ],
                dtype=np.float64,
            )
            inverse = np.minimum(inverse, np.quantile(inverse, 0.90) if len(inverse) > 1 else inverse[0])
            local = inverse / inverse.sum()
        else:
            local = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
        for index, value in zip(indices, local):
            weights[index] = float(value)
    weights /= weights.sum()
    architecture_totals = [float(weights[indices].sum()) for indices in groups.values()]
    if max(architecture_totals) - min(architecture_totals) > 1e-12:
        raise AssertionError("architecture weights are not balanced")
    return weights


def build_development_dataset(output: Path) -> list[dict[str, Any]]:
    unified_root = REPO_ROOT / UNIFIED_RELATIVE
    h_root = REPO_ROOT / H_TARGETED_RELATIVE
    old_rows = load_json(unified_root / "unified_training_data_no_test.json")
    new_statistics = {
        str(row["config_id"]): row
        for row in load_json(h_root / "phase_b/prospective_config_statistics.json")
    }
    new_selection = load_json(h_root / "phase_b/targeted_selection.json")
    assert_no_test_fields(old_rows, context="unified development labels")
    assert_no_test_fields(new_statistics, context="targeted prospective statistics")
    if len({row["architecture_fingerprint"] for row in old_rows}) != 60:
        raise ValueError("unified label pool is not 60 architectures")
    rows: list[dict[str, Any]] = []
    for source in old_rows:
        row = dict(source)
        row.update(
            {
                "development_source": "unified_60",
                "label_protocol": "cora_global4_300epochs_patience80_validation_only",
                "structure_vector": structure_feature_vector(row, [1.0] * 4)[0].tolist(),
            }
        )
        rows.append(row)
    for selected in new_selection:
        stats = new_statistics[str(selected["config_id"])]
        row = {
            key: selected[key]
            for key in (
                "config_id",
                "architecture_fingerprint",
                "hp_fingerprint",
                "operations",
                "edges",
                "raw_hp_vector",
                "mu_arch",
                "z_arch",
                "lr",
                "dropout",
                "hidden_dim",
                "l2",
                "source_search_seed",
                "source_step",
                "selection_stratum",
            )
        }
        row.update(
            {
                "validation_mean": float(stats["validation_mean"]),
                "validation_ci95_low": float(stats["validation_ci95_low"]),
                "validation_ci95_high": float(stats["validation_ci95_high"]),
                "validation_ci95_half_width": float(stats["validation_ci95_half_width"]),
                "validation_sample_std": float(stats["validation_sample_std"]),
                "seed_count": int(stats["seed_count"]),
                "development_source": "h_targeted_20",
                "label_protocol": "cora_global4_300epochs_patience80_validation_only",
                "structure_vector": structure_feature_vector(selected, [1.0] * 4)[0].tolist(),
            }
        )
        rows.append(row)
    if len(rows) != 96:
        raise ValueError(f"expected 96 development configurations, found {len(rows)}")
    fingerprints = {str(row["architecture_fingerprint"]) for row in rows}
    if len(fingerprints) != 80:
        raise ValueError(f"expected 80 development architectures, found {len(fingerprints)}")
    if len({str(row["config_id"]) for row in rows}) != len(rows):
        raise ValueError("development config IDs are not unique")
    assert_no_test_fields(rows, context="80-architecture development manifest")
    main_weights = architecture_balanced_weights(rows, uncertainty_weighted=False)
    sensitivity_weights = architecture_balanced_weights(rows, uncertainty_weighted=True)
    for index, row in enumerate(rows):
        row["architecture_balanced_weight"] = float(main_weights[index])
        row["architecture_balanced_uncertainty_weight"] = float(sensitivity_weights[index])
    development_dir = output / "development"
    dump_json(development_dir / "development_80_manifest.json", rows)
    dump_csv(development_dir / "development_80_manifest.csv", rows)
    source_counts = Counter(row["development_source"] for row in rows)
    config_counts = Counter(row["architecture_fingerprint"] for row in rows)
    audit = {
        "passed": True,
        "configuration_count": len(rows),
        "architecture_count": len(fingerprints),
        "source_configuration_counts": dict(source_counts),
        "source_architecture_counts": {
            source: len({row["architecture_fingerprint"] for row in rows if row["development_source"] == source})
            for source in source_counts
        },
        "source_fingerprint_overlap_count": len(
            {row["architecture_fingerprint"] for row in rows if row["development_source"] == "unified_60"}
            & {row["architecture_fingerprint"] for row in rows if row["development_source"] == "h_targeted_20"}
        ),
        "configurations_per_architecture_distribution": dict(Counter(config_counts.values())),
        "maximum_configurations_per_architecture": max(config_counts.values()),
        "architecture_balanced_total_weight": 1.0 / 80.0,
        "architecture_balanced_weight_spread": max(
            sum(row["architecture_balanced_weight"] for row in rows if row["architecture_fingerprint"] == fingerprint)
            for fingerprint in fingerprints
        ) - min(
            sum(row["architecture_balanced_weight"] for row in rows if row["architecture_fingerprint"] == fingerprint)
            for fingerprint in fingerprints
        ),
        "contains_test_named_field": False,
        "label_protocol": "Cora/global4/300 epochs/patience 80/validation accuracy",
        "dataset_content_fingerprint": DATASET_FINGERPRINT,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "time_prefix_analysis_available": False,
        "time_prefix_limitation": "The unified-60 time index and five per-search-seed source steps for the targeted-20 do not define a justified common chronology; no synthetic cross-source time ordering was created.",
    }
    dump_json(development_dir / "development_data_audit.json", audit)
    (development_dir / "development_data_audit.md").write_text(
        "# 80-architecture development-data audit\n\n"
        "Status: **PASS**\n\n"
        "The manifest contains 96 configurations grouped into 80 disjoint architecture fingerprints. "
        "Every architecture has equal total main-analysis weight. Only Cora/global4/300-epoch/patience-80 "
        "validation labels are retained, and no test-named field is present. A unified time-prefix analysis "
        "is not claimed because the two source pools do not share a defensible total chronology.\n",
        encoding="utf-8",
    )
    return rows


def build_fixed_splits(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[str(row["architecture_fingerprint"])].append(str(row["config_id"]))
    fingerprints = sorted(groups)
    validation_count = round(len(fingerprints) * OUTER_VALIDATION_FRACTION)
    splits = []
    for repeat in range(OUTER_REPEATS):
        shuffled = sorted(fingerprints, key=lambda fingerprint: stable_hash("structure-native-outer", repeat, fingerprint))
        validation_fingerprints = set(shuffled[:validation_count])
        train_fingerprints = set(shuffled[validation_count:])
        if train_fingerprints & validation_fingerprints:
            raise AssertionError("outer architecture leakage")
        splits.append(
            {
                "split_id": f"group_{repeat:03d}",
                "protocol": "architecture_group_repeated",
                "train_architecture_fingerprints": sorted(train_fingerprints),
                "validation_architecture_fingerprints": sorted(validation_fingerprints),
                "train_config_ids": sorted(config_id for fp in train_fingerprints for config_id in groups[fp]),
                "validation_config_ids": sorted(config_id for fp in validation_fingerprints for config_id in groups[fp]),
                "architecture_leakage_count": 0,
            }
        )
    return splits


def inner_split(
    train_rows: Sequence[Mapping[str, Any]], *, split_id: str, training_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fingerprints = sorted(
        {str(row["architecture_fingerprint"]) for row in train_rows},
        key=lambda fingerprint: stable_hash("structure-native-inner", split_id, training_seed, fingerprint),
    )
    count = max(4, round(len(fingerprints) * INNER_VALIDATION_FRACTION))
    validation = set(fingerprints[:count])
    fit = [dict(row) for row in train_rows if str(row["architecture_fingerprint"]) not in validation]
    stop = [dict(row) for row in train_rows if str(row["architecture_fingerprint"]) in validation]
    if {row["architecture_fingerprint"] for row in fit} & {row["architecture_fingerprint"] for row in stop}:
        raise AssertionError("inner architecture leakage")
    return fit, stop, {
        "split_id": split_id,
        "training_seed": training_seed,
        "fit_architecture_fingerprints": sorted({row["architecture_fingerprint"] for row in fit}),
        "validation_architecture_fingerprints": sorted(validation),
        "architecture_leakage_count": 0,
    }


def _raw_feature_matrix(rows: Sequence[Mapping[str, Any]], model_name: str) -> np.ndarray:
    hp = np.asarray([row["raw_hp_vector"] for row in rows], dtype=np.float64)
    if model_name == "refit_head":
        return np.concatenate((np.asarray([row["mu_arch"] for row in rows], dtype=np.float64), hp), axis=1)
    structure = np.asarray([row["structure_vector"] for row in rows], dtype=np.float64)
    if model_name == "d_structure":
        return np.concatenate((structure, hp), axis=1)
    if model_name == "h_structure":
        return np.concatenate((np.asarray([row["mu_arch"] for row in rows], dtype=np.float64), structure, hp), axis=1)
    raise ValueError(model_name)


def _weighted_mean_std(matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum(((matrix - mean) ** 2) * weights[:, None], axis=0)
    std = np.sqrt(np.maximum(variance, 0.0))
    std[std < 1e-8] = 1.0
    return mean, std


def fit_scaling(
    rows: Sequence[Mapping[str, Any]], model_name: str, *, uncertainty_weighted: bool
) -> dict[str, Any]:
    weights = architecture_balanced_weights(rows, uncertainty_weighted=uncertainty_weighted)
    features = _raw_feature_matrix(rows, model_name)
    y = np.asarray([row["validation_mean"] for row in rows], dtype=np.float64)
    x_mean, x_std = _weighted_mean_std(features, weights)
    y_mean = float(np.sum(y * weights))
    y_std = float(np.sqrt(np.sum(((y - y_mean) ** 2) * weights)))
    return {
        "feature_mean": x_mean.tolist(),
        "feature_std": x_std.tolist(),
        "y_mean": y_mean,
        "y_std": max(y_std, 1e-8),
        "fitted_on_training_only": True,
        "architecture_balanced": True,
        "uncertainty_weighted": uncertainty_weighted,
    }


def scaled_features(rows: Sequence[Mapping[str, Any]], model_name: str, scaling: Mapping[str, Any]) -> np.ndarray:
    raw = _raw_feature_matrix(rows, model_name)
    return (raw - np.asarray(scaling["feature_mean"], dtype=np.float64)) / np.asarray(scaling["feature_std"], dtype=np.float64)


def _train_once(
    fit_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]] | None,
    *,
    model_name: str,
    learning_rate: float,
    seed: int,
    maximum_epochs: int,
    uncertainty_weighted: bool,
) -> tuple[StrictPerformanceHead, dict[str, Any], int, float | None, list[dict[str, Any]]]:
    set_all_seeds(seed)
    scaling = fit_scaling(fit_rows, model_name, uncertainty_weighted=uncertainty_weighted)
    x = torch.tensor(scaled_features(fit_rows, model_name, scaling), dtype=torch.float32)
    y = torch.tensor(
        (np.asarray([row["validation_mean"] for row in fit_rows], dtype=np.float64) - float(scaling["y_mean"])) / float(scaling["y_std"]),
        dtype=torch.float32,
    )
    weights = torch.tensor(
        architecture_balanced_weights(fit_rows, uncertainty_weighted=uncertainty_weighted),
        dtype=torch.float32,
    )
    head = StrictPerformanceHead(x.shape[1])
    optimizer = torch.optim.Adam(head.parameters(), lr=float(learning_rate))
    best_state = None
    best_score = float("inf")
    best_epoch = 1
    no_improvement = 0
    trace = []
    for epoch in range(1, maximum_epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = head(x)
        loss = torch.sum(weights * (predictions - y) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite predictor loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        trace_row = {"epoch": epoch, "weighted_mse": float(loss.detach())}
        trace.append(trace_row)
        if validation_rows is None:
            continue
        observed = np.asarray([row["validation_mean"] for row in validation_rows], dtype=np.float64)
        predicted = predict_head(head, validation_rows, model_name, scaling)
        validation_weights = architecture_balanced_weights(validation_rows, uncertainty_weighted=False)
        score = float(np.sqrt(np.sum(validation_weights * (predicted - observed) ** 2)))
        trace_row["inner_architecture_balanced_rmse"] = score
        if score < best_score - 1e-10:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= EARLY_STOPPING_PATIENCE:
                break
    if validation_rows is not None:
        if best_state is None:
            raise RuntimeError("inner model selection produced no finite state")
        head.load_state_dict(best_state)
    else:
        best_epoch = maximum_epochs
        best_score = None
    return head, scaling, best_epoch, best_score, trace


def predict_head(
    head: StrictPerformanceHead,
    rows: Sequence[Mapping[str, Any]],
    model_name: str,
    scaling: Mapping[str, Any],
) -> np.ndarray:
    head.eval()
    features = torch.tensor(scaled_features(rows, model_name, scaling), dtype=torch.float32)
    with torch.no_grad():
        normalized = head(features).numpy()
    return normalized * float(scaling["y_std"]) + float(scaling["y_mean"])


def train_development_models(
    output: Path, rows: Sequence[Mapping[str, Any]], splits: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {str(row["config_id"]): row for row in rows}
    results: list[dict[str, Any]] = []
    inner_manifests: list[dict[str, Any]] = []
    task_count = len(splits) * len(TRAINING_SEEDS) * len(WEIGHT_MODES) * len(TRAINED_MODELS)
    completed = 0
    for split in splits:
        outer_train = [lookup[config_id] for config_id in split["train_config_ids"]]
        outer_validation = [lookup[config_id] for config_id in split["validation_config_ids"]]
        for training_seed in TRAINING_SEEDS:
            inner_fit, inner_stop, inner_manifest = inner_split(
                outer_train, split_id=str(split["split_id"]), training_seed=training_seed
            )
            inner_manifests.append(inner_manifest)
            for weight_mode in WEIGHT_MODES:
                uncertainty_weighted = weight_mode == "architecture_balanced_uncertainty"
                for model_name in TRAINED_MODELS:
                    candidates = []
                    for learning_rate in LEARNING_RATES:
                        selection_seed = int(stable_hash("structure-native-select", split["split_id"], training_seed, model_name, learning_rate)[:8], 16)
                        head, scaling, best_epoch, score, trace = _train_once(
                            inner_fit,
                            inner_stop,
                            model_name=model_name,
                            learning_rate=learning_rate,
                            seed=selection_seed,
                            maximum_epochs=MAXIMUM_EPOCHS,
                            uncertainty_weighted=uncertainty_weighted,
                        )
                        candidates.append(
                            {
                                "learning_rate": learning_rate,
                                "selected_epoch": best_epoch,
                                "inner_rmse": score,
                                "head_state": {name: value.detach().clone() for name, value in head.state_dict().items()},
                                "scaling": scaling,
                                "inner_predictions": predict_head(head, inner_stop, model_name, scaling),
                                "trace": trace,
                            }
                        )
                    selected = min(candidates, key=lambda item: (float(item["inner_rmse"]), float(item["learning_rate"])))
                    final_seed = int(stable_hash("structure-native-final", split["split_id"], training_seed, model_name)[:8], 16)
                    final_head, final_scaling, _epoch, _score, final_trace = _train_once(
                        outer_train,
                        None,
                        model_name=model_name,
                        learning_rate=float(selected["learning_rate"]),
                        seed=final_seed,
                        maximum_epochs=int(selected["selected_epoch"]),
                        uncertainty_weighted=uncertainty_weighted,
                    )
                    validation_predictions = predict_head(final_head, outer_validation, model_name, final_scaling)
                    checkpoint = output / "models/checkpoints" / weight_mode / str(split["split_id"]) / f"seed_{training_seed}" / model_name / "checkpoint.pth"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "format_version": FORMAT_VERSION,
                        "model_name": model_name,
                        "weight_mode": weight_mode,
                        "split_id": split["split_id"],
                        "training_seed": training_seed,
                        "feature_schema": {
                            "refit_head": ["mu_arch", "raw_hp_vector"],
                            "d_structure": ["structure_vector", "raw_hp_vector"],
                            "h_structure": ["mu_arch", "structure_vector", "raw_hp_vector"],
                        }[model_name],
                        "vae_trained": False,
                        "latent_distance_gp_used": False,
                        "head_input_dim": int(final_head.network[0].in_features),
                        "head_state": {name: value.detach().cpu() for name, value in final_head.state_dict().items()},
                        "scaling": final_scaling,
                        "learning_rate": selected["learning_rate"],
                        "selected_epoch": selected["selected_epoch"],
                        "inner_selection_metric": "architecture_balanced_rmse",
                    }
                    temporary = checkpoint.with_suffix(".tmp")
                    torch.save(payload, temporary)
                    temporary.replace(checkpoint)
                    result = {
                        "task_id": f"{weight_mode}__{split['split_id']}__seed_{training_seed}__{model_name}",
                        "model": model_name,
                        "weight_mode": weight_mode,
                        "split_id": split["split_id"],
                        "training_seed": training_seed,
                        "checkpoint_path": str(checkpoint.resolve()),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "selected_learning_rate": selected["learning_rate"],
                        "selected_epoch": selected["selected_epoch"],
                        "inner_architecture_balanced_rmse": selected["inner_rmse"],
                        "inner_validation_config_ids": [row["config_id"] for row in inner_stop],
                        "inner_validation_predictions": [float(value) for value in selected["inner_predictions"]],
                        "validation_predictions": [
                            {
                                "config_id": row["config_id"],
                                "architecture_fingerprint": row["architecture_fingerprint"],
                                "prediction": float(value),
                            }
                            for row, value in zip(outer_validation, validation_predictions)
                        ],
                        "final_training_trace": final_trace,
                    }
                    results.append(result)
                    completed += 1
                    if completed % 30 == 0 or completed == task_count:
                        print(f"MODEL_PROGRESS completed={completed}/{task_count}", flush=True)
    if len({row["task_id"] for row in results}) != task_count:
        raise AssertionError("development task IDs are not unique and complete")
    dump_json(output / "development/fixed_inner_split_manifest.json", inner_manifests)
    compact = [{key: value for key, value in row.items() if key not in {"final_training_trace", "inner_validation_predictions", "validation_predictions"}} for row in results]
    dump_json(output / "models/model_training_results.json", compact)
    dump_csv(output / "models/model_training_results.csv", compact)
    prediction_rows = []
    for result in results:
        for prediction in result["validation_predictions"]:
            source = lookup[str(prediction["config_id"])]
            prediction_rows.append(
                {
                    **prediction,
                    "model": result["model"],
                    "weight_mode": result["weight_mode"],
                    "split_id": result["split_id"],
                    "training_seed": result["training_seed"],
                    "validation_mean": source["validation_mean"],
                    "validation_ci95_low": source["validation_ci95_low"],
                    "validation_ci95_high": source["validation_ci95_high"],
                    "selection_stratum": source["development_source"],
                }
            )
    return results, prediction_rows, inner_manifests


def aggregate_oof_predictions(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    label_lookup = {str(row["config_id"]): row for row in labels}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)
    output = []
    for config_id, values in sorted(grouped.items()):
        label = label_lookup[config_id]
        output.append(
            {
                "config_id": config_id,
                "architecture_fingerprint": label["architecture_fingerprint"],
                "validation_mean": label["validation_mean"],
                "validation_ci95_low": label["validation_ci95_low"],
                "validation_ci95_high": label["validation_ci95_high"],
                "selection_stratum": label["development_source"],
                "prediction": statistics.fmean(float(row["prediction"]) for row in values),
                "prediction_source_count": len(values),
            }
        )
    if {row["config_id"] for row in output} != set(label_lookup):
        missing = set(label_lookup) - {row["config_id"] for row in output}
        raise RuntimeError(f"OOF predictions do not cover development configurations: {sorted(missing)}")
    return output


def original_frozen_development_predictions(
    development: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    strict_root = REPO_ROOT / STRICT_RELATIVE
    h_root = REPO_ROOT / H_TARGETED_RELATIVE
    labels = {str(row["config_id"]): row for row in development}
    raw = load_prediction_rows(strict_root / "all_validation_predictions.csv")
    result: dict[str, list[dict[str, Any]]] = {}
    for output_model, source_model in (
        ("original_frozen_head", "frozen_head"),
        ("original_frozen_h", "frozen_hybrid_head"),
    ):
        existing = aggregate_predictions(
            [
                row
                for row in raw
                if row["model"] == source_model
                and row["dataset_variant"] == "full"
                and row["protocol"] == "group_repeated"
            ]
        )
        prefix = "head" if output_model == "original_frozen_head" else "h"
        targeted = load_csv(h_root / "phase_b/frozen_prospective_predictions.csv")
        predictions: dict[str, tuple[float, int]] = {
            str(row["config_id"]): (float(row["prediction"]), int(row["prediction_source_count"]))
            for row in existing
        }
        predictions.update(
            {
                str(row["config_id"]): (float(row[f"{prefix}_prediction"]), 150)
                for row in targeted
            }
        )
        if set(predictions) != set(labels):
            raise RuntimeError(f"{output_model} does not cover the 80-architecture development set")
        result[output_model] = [
            {
                "config_id": config_id,
                "architecture_fingerprint": labels[config_id]["architecture_fingerprint"],
                "validation_mean": labels[config_id]["validation_mean"],
                "validation_ci95_low": labels[config_id]["validation_ci95_low"],
                "validation_ci95_high": labels[config_id]["validation_ci95_high"],
                "selection_stratum": labels[config_id]["development_source"],
                "prediction": predictions[config_id][0],
                "prediction_source_count": predictions[config_id][1],
            }
            for config_id in sorted(labels)
        ]
    return result


def select_inner_ensemble_weights(
    task_results: Sequence[Mapping[str, Any]], development: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    labels = {str(row["config_id"]): row for row in development}
    grouped: dict[tuple[str, int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for result in task_results:
        grouped[(str(result["split_id"]), int(result["training_seed"]), str(result["weight_mode"]))][str(result["model"])] = result
    output = []
    for (split_id, training_seed, weight_mode), models in sorted(grouped.items()):
        if "refit_head" not in models or "h_structure" not in models:
            raise RuntimeError("inner ensemble selection is missing Head or H")
        head = models["refit_head"]
        hybrid = models["h_structure"]
        if head["inner_validation_config_ids"] != hybrid["inner_validation_config_ids"]:
            raise RuntimeError("Head/H inner validation membership differs")
        config_ids = list(head["inner_validation_config_ids"])
        grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for config_id, head_prediction, h_prediction in zip(
            config_ids,
            head["inner_validation_predictions"],
            hybrid["inner_validation_predictions"],
        ):
            label = labels[str(config_id)]
            grouped[str(label["architecture_fingerprint"])].append(
                (float(label["validation_mean"]), float(head_prediction), float(h_prediction))
            )
        actual = np.asarray(
            [statistics.fmean(value[0] for value in grouped[fingerprint]) for fingerprint in sorted(grouped)],
            dtype=np.float64,
        )
        head_values = np.asarray(
            [statistics.fmean(value[1] for value in grouped[fingerprint]) for fingerprint in sorted(grouped)],
            dtype=np.float64,
        )
        h_values = np.asarray(
            [statistics.fmean(value[2] for value in grouped[fingerprint]) for fingerprint in sorted(grouped)],
            dtype=np.float64,
        )
        candidates = []
        for h_weight in WEIGHT_GRID:
            predicted = (1.0 - h_weight) * head_values + h_weight * h_values
            tau = kendalltau(actual, predicted).statistic
            candidates.append(
                {
                    "h_weight": h_weight,
                    "head_weight": 1.0 - h_weight,
                    "inner_kendall": None if tau is None or not np.isfinite(tau) else float(tau),
                    "inner_rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
                }
            )
        selected = max(candidates, key=lambda row: (-1.0 if row["inner_kendall"] is None else row["inner_kendall"], -row["inner_rmse"], -abs(row["h_weight"] - 0.5)))
        output.append(
            {
                "split_id": split_id,
                "training_seed": training_seed,
                "weight_mode": weight_mode,
                "selection_source": "inner_architecture_group_validation_only",
                "outer_validation_labels_used": False,
                "inner_metric_unit": "architecture_fingerprint_equal_weight",
                "inner_architecture_count": len(grouped),
                "inner_configuration_count": len(config_ids),
                "grid": list(WEIGHT_GRID),
                "candidates": candidates,
                **selected,
            }
        )
    return output


def ensemble_oof_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    weight_selections: Sequence[Mapping[str, Any]],
    development: Sequence[Mapping[str, Any]],
    *,
    weight_mode: str,
) -> dict[str, list[dict[str, Any]]]:
    labels = {str(row["config_id"]): row for row in development}
    weights = {
        (str(row["split_id"]), int(row["training_seed"])): float(row["h_weight"])
        for row in weight_selections
        if row["weight_mode"] == weight_mode
    }
    grouped: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    for row in prediction_rows:
        if row["weight_mode"] != weight_mode or row["model"] not in {"refit_head", "h_structure"}:
            continue
        grouped[(str(row["split_id"]), int(row["training_seed"]), str(row["config_id"]))][str(row["model"])] = float(row["prediction"])
    raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (split_id, training_seed, config_id), values in sorted(grouped.items()):
        if set(values) != {"refit_head", "h_structure"}:
            raise RuntimeError("ensemble OOF row lacks Head or H")
        head = values["refit_head"]
        hybrid = values["h_structure"]
        h_weight = weights[(split_id, training_seed)]
        predictions = {
            "head_h_mean": 0.5 * (head + hybrid),
            "head_h_conservative": min(head, hybrid),
            "head_h_weighted": (1.0 - h_weight) * head + h_weight * hybrid,
        }
        label = labels[config_id]
        for model, prediction in predictions.items():
            raw[model].append(
                {
                    "config_id": config_id,
                    "architecture_fingerprint": label["architecture_fingerprint"],
                    "prediction": float(prediction),
                    "model": model,
                    "weight_mode": weight_mode,
                    "split_id": split_id,
                    "training_seed": training_seed,
                }
            )
    return {model: aggregate_oof_predictions(rows, development) for model, rows in raw.items()}


def conformal_calibration(rows: Sequence[Mapping[str, Any]], *, coverage: float = 0.90) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(abs(float(row["prediction"]) - float(row["validation_mean"])))
    scores = sorted(max(values) for values in grouped.values())
    if not scores:
        raise ValueError("conformal calibration needs architecture residuals")
    rank = min(len(scores), math.ceil((len(scores) + 1) * coverage))
    quantile = float(scores[rank - 1])
    return {
        "method": "architecture_group_max_absolute_oof_residual",
        "nominal_coverage": coverage,
        "architecture_score_count": len(scores),
        "finite_sample_rank": rank,
        "half_width": quantile,
        "mean_width": 2.0 * quantile,
        "calibration_uses_prospective24_labels": False,
    }


def attach_intervals(rows: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]) -> list[dict[str, Any]]:
    half_width = float(calibration["half_width"])
    return [
        {
            **dict(row),
            "prediction_interval_low": float(row["prediction"]) - half_width,
            "prediction_interval_high": float(row["prediction"]) + half_width,
        }
        for row in rows
    ]


def build_development_predictions(
    output: Path,
    development: Sequence[Mapping[str, Any]],
    task_results: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    weights = select_inner_ensemble_weights(task_results, development)
    dump_json(output / "models/ensemble_weight_selections.json", weights)
    dump_csv(output / "models/ensemble_weight_selections.csv", weights)
    model_rows = original_frozen_development_predictions(development)
    for model in TRAINED_MODELS:
        subset = [row for row in prediction_rows if row["model"] == model and row["weight_mode"] == "architecture_balanced"]
        model_rows[model] = aggregate_oof_predictions(subset, development)
    model_rows.update(ensemble_oof_rows(prediction_rows, weights, development, weight_mode="architecture_balanced"))
    if set(model_rows) != set(MODEL_NAMES):
        raise RuntimeError(f"development prediction model set mismatch: {sorted(model_rows)}")
    calibrations: dict[str, dict[str, Any]] = {}
    metrics = []
    all_rows = []
    source_by_fingerprint = {
        str(row["architecture_fingerprint"]): str(row["development_source"])
        for row in development
    }
    samples = stratified_bootstrap_samples(
        [
            {"architecture_fingerprint": fingerprint, "selection_stratum": source}
            for fingerprint, source in sorted(source_by_fingerprint.items())
        ],
        replicates=PHASE0_BOOTSTRAPS,
        seed=BOOTSTRAP_SEED + 1,
    )
    for model, rows in model_rows.items():
        calibration = conformal_calibration(rows)
        calibrations[model] = calibration
        interval_rows = attach_intervals(rows, calibration)
        summary, _ = summarize_with_bootstrap(interval_rows, samples, interval_alpha=0.10)
        summary.update({"model": model, "weight_mode": "architecture_balanced", "scope": "development_oof"})
        metrics.append(summary)
        all_rows.extend({**row, "model": model} for row in interval_rows)
        model_rows[model] = interval_rows
    sensitivity_metrics = []
    for model in TRAINED_MODELS:
        subset = [row for row in prediction_rows if row["model"] == model and row["weight_mode"] == "architecture_balanced_uncertainty"]
        aggregated = aggregate_oof_predictions(subset, development)
        calibration = conformal_calibration(aggregated)
        interval_rows = attach_intervals(aggregated, calibration)
        summary = metric_bundle(interval_rows, interval_alpha=0.10)
        summary.update({"model": model, "weight_mode": "architecture_balanced_uncertainty", "scope": "development_oof_sensitivity"})
        sensitivity_metrics.append(summary)
    dump_json(output / "models/development_oof_predictions.json", all_rows)
    dump_csv(output / "models/development_oof_predictions.csv", all_rows)
    dump_json(output / "models/conformal_calibrations.json", calibrations)
    dump_json(output / "models/development_metrics.json", metrics)
    dump_csv(output / "models/development_metrics.csv", metrics)
    dump_json(output / "models/uncertainty_weight_sensitivity.json", sensitivity_metrics)
    dump_csv(output / "models/uncertainty_weight_sensitivity.csv", sensitivity_metrics)
    return model_rows, calibrations, weights


def project_unbiased_history_candidate(
    source: Mapping[str, Any], *, source_path: Path, source_index: int
) -> tuple[dict[str, Any], list[str]]:
    """Strict allowlist projection; performance and prediction values are never copied."""

    config = normalize_architecture_config({"operations": source["operations"], "edges": source["edges"]})
    z_search = source["z_search"]
    raw_hp = [float(value) for value in z_search[12:16]]
    if len(raw_hp) != 4:
        raise ValueError("history candidate is not global4")
    parent = source_path.parent.name
    if not parent.startswith("search_seed"):
        raise ValueError(f"cannot derive search seed from {source_path}")
    search_seed = int(parent.removeprefix("search_seed"))
    step = int(source.get("step", source_index))
    fingerprint = architecture_fingerprint(config)
    projected = {
        "architecture_fingerprint": fingerprint,
        "operations": list(config["operations"]),
        "edges": [list(edge) for edge in config["edges"]],
        "raw_hp_vector": raw_hp,
        "hp_fingerprint": stable_json_hash(raw_hp),
        "source_search_seed": search_seed,
        "source_step": step,
        "source_index": source_index,
        "source_path": str(source_path.resolve()),
        "structure_vector": structure_feature_vector(config, [1.0] * 4)[0].tolist(),
    }
    allowed_source_reads = {"operations", "edges", "z_search[12:16]", "step"}
    masked_keys = sorted(str(key) for key in source if str(key) not in {"operations", "edges", "z_search", "step"})
    projected["selection_projection_sha256"] = stable_json_hash(projected)
    projected["selection_allowlist"] = sorted(allowed_source_reads)
    return projected, masked_keys


def build_unbiased_candidate_pool(development: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing = {str(row["architecture_fingerprint"]) for row in development}
    quarantined_selection_path = REPO_ROOT / QUARANTINED_RELATIVE / "phase_4/frozen_selection_24.json"
    quarantined = set()
    if quarantined_selection_path.is_file():
        quarantined = {
            str(row["architecture_fingerprint"])
            for row in load_json(quarantined_selection_path)
        }
    excluded = existing | quarantined
    history_paths = sorted(REPO_ROOT.glob(HISTORY_GLOB))
    if not history_paths:
        raise FileNotFoundError(HISTORY_GLOB)
    unique: dict[str, dict[str, Any]] = {}
    masked = set()
    source_records = 0
    projection_failures = []
    for path in history_paths:
        records = load_json(path)
        for index, source in enumerate(records):
            source_records += 1
            try:
                projected, masked_keys = project_unbiased_history_candidate(source, source_path=path, source_index=index)
            except Exception as exc:
                projection_failures.append({"path": str(path), "index": index, "error": str(exc)})
                continue
            masked.update(masked_keys)
            fingerprint = str(projected["architecture_fingerprint"])
            if fingerprint in excluded:
                continue
            current = unique.get(fingerprint)
            ordering = stable_hash("unbiased-one-hp", projected["architecture_fingerprint"], projected["hp_fingerprint"], projected["source_search_seed"], projected["source_step"], projected["source_index"])
            if current is None or ordering < current["representative_ordering_hash"]:
                unique[fingerprint] = {**projected, "representative_ordering_hash": ordering}
    pool = sorted(unique.values(), key=lambda row: row["architecture_fingerprint"])
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_seed[int(row["source_search_seed"])].append(row)
    for seed, rows in by_seed.items():
        ordered = sorted(rows, key=lambda row: (int(row["source_step"]), str(row["architecture_fingerprint"])))
        for rank, row in enumerate(ordered):
            row["time_quartile"] = min(3, (4 * rank) // len(ordered))
            row["time_quartile_definition"] = "within_search_seed_rank_quartile"
    audit = {
        "passed": not projection_failures and len(pool) >= 24,
        "source_paths": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in history_paths],
        "source_record_count": source_records,
        "projection_failure_count": len(projection_failures),
        "projection_failures": projection_failures,
        "projected_unique_nonoverlap_architecture_count": len(pool),
        "existing_development_architecture_count": len(existing),
        "quarantined_prefreeze_architecture_count": len(quarantined),
        "total_excluded_architecture_count": len(excluded),
        "candidate_existing_overlap_count": len(existing & set(unique)),
        "candidate_quarantined_overlap_count": len(quarantined & set(unique)),
        "quarantined_selection_path": str(quarantined_selection_path.resolve()) if quarantined_selection_path.is_file() else None,
        "quarantined_selection_sha256": sha256_file(quarantined_selection_path) if quarantined_selection_path.is_file() else None,
        "one_hp_configuration_per_architecture": len(pool) == len({row["architecture_fingerprint"] for row in pool}),
        "read_allowlist": ["operations", "edges", "z_search[12:16] as raw_hp_vector", "step", "source path search seed", "source record index"],
        "masked_source_keys": sorted(masked),
        "used_validation_or_test_values": False,
        "used_value_reward_or_performance_rank": False,
        "used_model_prediction_or_disagreement": False,
        "candidate_pool_contains_prediction_or_label_fields": False,
        "selection_inputs": ["operations", "edges", "structure_vector", "raw_hp_vector", "source_search_seed", "source_step", "stable_hash"],
        "search_seed_time_quartile_cell_counts": {
            f"seed_{seed}_q{quartile}": sum(int(row["source_search_seed"]) == seed and int(row["time_quartile"]) == quartile for row in pool)
            for seed in sorted(by_seed)
            for quartile in range(4)
        },
    }
    return pool, audit


def select_unbiased_candidates(pool: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells = {(seed, quartile): [] for seed in range(5) for quartile in range(4)}
    for row in pool:
        cells.setdefault((int(row["source_search_seed"]), int(row["time_quartile"])), []).append(row)
    selected: list[dict[str, Any]] = []
    target_cells = [(seed, quartile) for quartile in range(4) for seed in [((quartile * 3 + offset) % 5) for offset in range(3)]]
    for seed, quartile in target_cells:
        eligible = cells.get((seed, quartile), [])
        if not eligible:
            raise RuntimeError(f"INSUFFICIENT_UNBIASED_CANDIDATES: empty seed/time cell {(seed, quartile)}")
        chosen = min(eligible, key=lambda row: stable_hash("hash-uniform-clean-restart-v1", row["architecture_fingerprint"], row["hp_fingerprint"]))
        selected.append({**dict(chosen), "selection_stratum": "hash_uniform"})
    selected_fingerprints = {row["architecture_fingerprint"] for row in selected}
    eligible = [dict(row) for row in pool if row["architecture_fingerprint"] not in selected_fingerprints]
    chosen_diverse: list[dict[str, Any]] = []
    seed_counts: Counter[int] = Counter()
    quartile_counts: Counter[int] = Counter()
    while len(chosen_diverse) < 12:
        best = None
        for row in eligible:
            if row["architecture_fingerprint"] in {item["architecture_fingerprint"] for item in chosen_diverse}:
                continue
            vector = np.asarray(row["structure_vector"], dtype=np.float64)
            if chosen_diverse:
                minimum_distance = min(float(np.mean(np.abs(vector - np.asarray(item["structure_vector"], dtype=np.float64)))) for item in chosen_diverse)
            else:
                minimum_distance = 1.0
            seed = int(row["source_search_seed"])
            quartile = int(row["time_quartile"])
            key = (
                minimum_distance,
                -seed_counts[seed],
                -quartile_counts[quartile],
                stable_hash("structure-diverse-clean-restart-v1", row["architecture_fingerprint"]),
            )
            if best is None or key > best[0]:
                best = (key, row)
        if best is None:
            raise RuntimeError("INSUFFICIENT_UNBIASED_CANDIDATES: diversity selector exhausted")
        chosen = {**dict(best[1]), "selection_stratum": "structure_diverse"}
        chosen_diverse.append(chosen)
        seed_counts[int(chosen["source_search_seed"])] += 1
        quartile_counts[int(chosen["time_quartile"])] += 1
    selected.extend(chosen_diverse)
    selected.sort(key=lambda row: (row["selection_stratum"], row["architecture_fingerprint"]))
    for index, row in enumerate(selected):
        row["selection_index"] = index
        row["assigned_gpu"] = index % 2
        row["config_id"] = f"structure_native_{row['source_search_seed']}_{row['source_step']}_{row['architecture_fingerprint'][:12]}"
        row.update(realize_global4_hp(row))
        row["hp_fingerprint"] = hp_fingerprint(row)
    counts = Counter(row["selection_stratum"] for row in selected)
    gpu_counts = Counter(int(row["assigned_gpu"]) for row in selected)
    if len(selected) != 24 or len({row["architecture_fingerprint"] for row in selected}) != 24:
        raise AssertionError("unbiased selection is not 24 unique architectures")
    if counts != Counter({"hash_uniform": 12, "structure_diverse": 12}):
        raise AssertionError("unbiased selection is not 12/12")
    if gpu_counts != Counter({0: 12, 1: 12}):
        raise AssertionError("unbiased selection is not 12/12 across GPUs")
    audit = {
        "passed": True,
        "selection_count": len(selected),
        "unique_architecture_count": len({row["architecture_fingerprint"] for row in selected}),
        "stratum_counts": dict(counts),
        "gpu_counts": dict(gpu_counts),
        "hash_uniform_target_cells": [{"search_seed": seed, "time_quartile": quartile} for seed, quartile in target_cells],
        "hash_uniform_uses_stable_hash_only_within_cells": True,
        "structure_diverse_uses_structure_features_only": True,
        "selection_used_label_values": False,
        "selection_used_model_predictions": False,
        "selection_used_model_disagreement": False,
    }
    return selected, audit


def realize_global4_hp(candidate: Mapping[str, Any]) -> dict[str, Any]:
    from eval_utils import decode_hp_by_mode

    z_search = torch.tensor(
        [*([0.0] * 12), *map(float, candidate["raw_hp_vector"])],
        dtype=torch.float32,
    )
    hp = decode_hp_by_mode(
        z_search=z_search,
        arch_nz=12,
        config=dict(candidate),
        hp_mode="global4",
        log_lr_min=-4.0,
        log_lr_max=-1.5,
        dropout_min=0.1,
        dropout_max=0.6,
    )
    return {
        "lr": float(hp["lr"]),
        "dropout": float(hp["dropout"]),
        "hidden_dim": int(hp["hidden_dim"]),
        "l2": float(hp["weight_decay"]),
        "gat_heads": int(hp["gat_heads"]),
        "sage_aggr": str(hp["sage_aggr"]),
        "gin_eps": float(hp["gin_eps"]),
        "gat_heads_by_layer": hp["gat_heads_by_layer"],
        "sage_aggr_by_layer": hp["sage_aggr_by_layer"],
        "gin_eps_by_layer": hp["gin_eps_by_layer"],
    }


def encode_selected_mu(selection: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checkpoint = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
    before = sha256_file(checkpoint)
    model = load_joint_vae(checkpoint, "global4", device="cpu")
    for parameter in model.arch_vae.parameters():
        parameter.requires_grad_(False)
    graphs = [config_to_igraph(row) for row in selection]
    model.arch_vae.eval()
    with torch.no_grad():
        mu, _ = model.arch_vae.encode(graphs)
    after = sha256_file(checkpoint)
    if before != after:
        raise RuntimeError("base VAE checkpoint changed during frozen encoding")
    return [
        {**dict(row), "mu_arch": [float(value) for value in mu[index].cpu().tolist()]}
        for index, row in enumerate(selection)
    ]


def frozen_candidate_predictions(
    selection: Sequence[Mapping[str, Any]],
    task_results: Sequence[Mapping[str, Any]],
    calibrations: Mapping[str, Mapping[str, Any]],
    weight_selections: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strict_root = REPO_ROOT / STRICT_RELATIVE
    original = frozen_head_h_predictions(selection, strict_root)
    original_lookup = {str(row["config_id"]): row for row in original}
    raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_instance: dict[tuple[str, int, str], dict[str, np.ndarray]] = defaultdict(dict)
    main_tasks = [row for row in task_results if row["weight_mode"] == "architecture_balanced"]
    for task in main_tasks:
        checkpoint = Path(task["checkpoint_path"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("vae_trained") is not False or payload.get("latent_distance_gp_used") is not False:
            raise RuntimeError(f"forbidden model state in checkpoint {checkpoint}")
        model_name = str(task["model"])
        head = StrictPerformanceHead(int(payload["head_input_dim"]))
        head.load_state_dict(payload["head_state"])
        predicted = predict_head(head, selection, model_name, payload["scaling"])
        key = (str(task["split_id"]), int(task["training_seed"]), str(task["weight_mode"]))
        per_instance[key][model_name] = predicted
        for row, value in zip(selection, predicted):
            raw[model_name].append(
                {
                    "config_id": row["config_id"],
                    "architecture_fingerprint": row["architecture_fingerprint"],
                    "prediction": float(value),
                    "split_id": task["split_id"],
                    "training_seed": task["training_seed"],
                }
            )
    weight_lookup = {
        (str(row["split_id"]), int(row["training_seed"])): float(row["h_weight"])
        for row in weight_selections
        if row["weight_mode"] == "architecture_balanced"
    }
    for (split_id, training_seed, _weight_mode), values in sorted(per_instance.items()):
        if "refit_head" not in values or "h_structure" not in values:
            raise RuntimeError("candidate ensemble instance lacks Refit Head or H")
        head = values["refit_head"]
        hybrid = values["h_structure"]
        h_weight = weight_lookup[(split_id, training_seed)]
        ensembles = {
            "head_h_mean": 0.5 * (head + hybrid),
            "head_h_conservative": np.minimum(head, hybrid),
            "head_h_weighted": (1.0 - h_weight) * head + h_weight * hybrid,
        }
        for model_name, predicted in ensembles.items():
            for row, value in zip(selection, predicted):
                raw[model_name].append(
                    {
                        "config_id": row["config_id"],
                        "architecture_fingerprint": row["architecture_fingerprint"],
                        "prediction": float(value),
                        "split_id": split_id,
                        "training_seed": training_seed,
                    }
                )
    output: list[dict[str, Any]] = []
    for row in selection:
        config_id = str(row["config_id"])
        baseline = original_lookup[config_id]
        values = {
            "original_frozen_head": float(baseline["head_prediction"]),
            "original_frozen_h": float(baseline["h_prediction"]),
        }
        source_counts = {"original_frozen_head": 150, "original_frozen_h": 150}
        for model_name in (*TRAINED_MODELS, *ENSEMBLE_MODELS):
            predictions = [float(item["prediction"]) for item in raw[model_name] if str(item["config_id"]) == config_id]
            if len(predictions) != OUTER_REPEATS * len(TRAINING_SEEDS):
                raise RuntimeError(f"expected 60 frozen {model_name} predictions for {config_id}, found {len(predictions)}")
            values[model_name] = statistics.fmean(predictions)
            source_counts[model_name] = len(predictions)
        for model_name in MODEL_NAMES:
            half_width = float(calibrations[model_name]["half_width"])
            output.append(
                {
                    "config_id": config_id,
                    "architecture_fingerprint": row["architecture_fingerprint"],
                    "selection_stratum": row["selection_stratum"],
                    "model": model_name,
                    "prediction": values[model_name],
                    "prediction_interval_low": values[model_name] - half_width,
                    "prediction_interval_high": values[model_name] + half_width,
                    "prediction_interval_nominal_coverage": 0.90,
                    "prediction_source_count": source_counts[model_name],
                    "predictions_frozen_before_prospective_labels": True,
                }
            )
    audit = {
        "model_count": len(MODEL_NAMES),
        "configuration_count": len(selection),
        "prediction_row_count": len(output),
        "expected_prediction_row_count": len(MODEL_NAMES) * len(selection),
        "predictions_frozen_before_prospective_labels": True,
        "calibration_source": "80-architecture development OOF residuals only",
        "latent_distance_gp_used": False,
        "vae_training_executed": False,
        "original_checkpoint_hashes": getattr(frozen_head_h_predictions, "checkpoint_hashes", {}),
    }
    return output, audit


def _actual_gpus() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        parts = [piece.strip() for piece in line.split(",", 3)]
        if len(parts) == 4:
            rows.append({"index": int(parts[0]), "name": parts[1], "uuid": parts[2], "memory_mib": int(parts[3])})
    if len(rows) != 2 or {row["index"] for row in rows} != {0, 1}:
        raise RuntimeError(f"expected exactly GPUs 0 and 1, found {rows}")
    return sorted(rows, key=lambda row: row["index"])


def verify_frozen_plan(plan_path: Path) -> dict[str, Any]:
    sidecar = plan_path.with_suffix(".sha256")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != sha256_file(plan_path) or fields[1] != plan_path.name:
        raise RuntimeError("structure-native task-plan sidecar mismatch")
    plan = load_json(plan_path)
    for section in ("code_hashes", "input_hashes", "protected_hashes"):
        for raw_path, expected in plan[section].items():
            path = Path(raw_path)
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"frozen {section} mismatch: {path}")
    for key in ("selection", "frozen_predictions", "model_checkpoint_manifest"):
        path = Path(plan[f"{key}_path"])
        if not path.is_file() or sha256_file(path) != plan[f"{key}_sha256"]:
            raise RuntimeError(f"frozen {key} mismatch")
    checkpoints = load_json(Path(plan["model_checkpoint_manifest_path"]))
    for row in checkpoints:
        path = Path(row["checkpoint_path"])
        if not path.is_file() or sha256_file(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"frozen model checkpoint mismatch: {path}")
    return plan


def prepare_experiment(output: Path) -> dict[str, Any]:
    decision_path = output / "phase_0/phase_0_decision.json"
    if not decision_path.is_file() or load_json(decision_path)["classification"] != "PHASE_0_PASSED":
        raise RuntimeError("Phase 0 did not pass; preparation is forbidden")
    if (output / "phase_4/label_task_plan.json").exists():
        raise FileExistsError("frozen label task plan already exists; refusing to overwrite")
    development = build_development_dataset(output)
    splits = build_fixed_splits(development)
    dump_json(output / "development/fixed_outer_split_manifest.json", splits)
    dump_csv(output / "development/fixed_outer_split_manifest.csv", splits)
    write_sha256_sidecar(output / "development/fixed_outer_split_manifest.json")
    task_results, prediction_rows, _inner = train_development_models(output, development, splits)
    model_rows, calibrations, weight_selections = build_development_predictions(
        output, development, task_results, prediction_rows
    )
    pool, pool_audit = build_unbiased_candidate_pool(development)
    if len(pool) < 24:
        dump_json(output / "final_classification.json", {"classification": "INSUFFICIENT_UNBIASED_CANDIDATES", "candidate_count": len(pool)})
        return {"classification": "INSUFFICIENT_UNBIASED_CANDIDATES"}
    dump_json(output / "phase_4/unbiased_candidate_pool.json", pool)
    dump_csv(output / "phase_4/unbiased_candidate_pool.csv", pool)
    dump_json(output / "phase_4/candidate_pool_audit.json", pool_audit)
    selection, selection_audit = select_unbiased_candidates(pool)
    development_fingerprints = {row["architecture_fingerprint"] for row in development}
    selection_fingerprints = {row["architecture_fingerprint"] for row in selection}
    overlap = development_fingerprints & selection_fingerprints
    if overlap:
        raise RuntimeError(f"candidate/development architecture overlap: {sorted(overlap)}")
    quarantined_selection_path = REPO_ROOT / QUARANTINED_RELATIVE / "phase_4/frozen_selection_24.json"
    quarantined_fingerprints = (
        {row["architecture_fingerprint"] for row in load_json(quarantined_selection_path)}
        if quarantined_selection_path.is_file()
        else set()
    )
    quarantined_overlap = quarantined_fingerprints & selection_fingerprints
    if quarantined_overlap:
        raise RuntimeError(f"candidate/quarantined-prefreeze architecture overlap: {sorted(quarantined_overlap)}")
    selection = encode_selected_mu(selection)
    selection_path = output / "phase_4/frozen_selection_24.json"
    dump_json(selection_path, selection)
    dump_csv(output / "phase_4/frozen_selection_24.csv", selection)
    write_sha256_sidecar(selection_path)
    selection_audit.update(
        {
            "development_overlap_count": len(overlap),
            "quarantined_prefreeze_overlap_count": len(quarantined_overlap),
            "development_fingerprint_count": len(development_fingerprints),
            "selection_fingerprint_count": len(selection_fingerprints),
            "selection_sha256": sha256_file(selection_path),
        }
    )
    dump_json(output / "phase_4/selection_audit.json", selection_audit)
    frozen_predictions, prediction_audit = frozen_candidate_predictions(
        selection, task_results, calibrations, weight_selections
    )
    prediction_path = output / "phase_4/frozen_candidate_predictions.json"
    dump_json(prediction_path, frozen_predictions)
    dump_csv(output / "phase_4/frozen_candidate_predictions.csv", frozen_predictions)
    write_sha256_sidecar(prediction_path)
    dump_json(output / "phase_4/frozen_prediction_audit.json", prediction_audit)
    checkpoint_manifest = [
        {
            "task_id": row["task_id"],
            "model": row["model"],
            "weight_mode": row["weight_mode"],
            "split_id": row["split_id"],
            "training_seed": row["training_seed"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
        for row in task_results
    ]
    checkpoint_manifest_path = output / "models/model_checkpoint_manifest.json"
    dump_json(checkpoint_manifest_path, checkpoint_manifest)
    write_sha256_sidecar(checkpoint_manifest_path)
    code_paths = [
        REPO_ROOT / "analyse/structure_native_surrogate_pilot.py",
        REPO_ROOT / "analyse/run_structure_native_surrogate_labels.py",
        REPO_ROOT / "analyse/evaluate_structure_native_surrogate_pilot.py",
        REPO_ROOT / "architecture_diagnostics.py",
        REPO_ROOT / "eval_utils.py",
    ]
    for path in code_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_paths = [
        REPO_ROOT / UNIFIED_RELATIVE / "unified_training_data_no_test.json",
        REPO_ROOT / UNIFIED_RELATIVE / "unified_seed_level_audit.jsonl",
        REPO_ROOT / UNIFIED_RELATIVE / "protocol_compatibility.json",
        REPO_ROOT / STRICT_RELATIVE / "task_plan.json",
        REPO_ROOT / STRICT_RELATIVE / "fixed_split_manifest.json",
        REPO_ROOT / STRICT_RELATIVE / "all_validation_predictions.csv",
        REPO_ROOT / STRICT_RELATIVE / "final_classification.json",
        REPO_ROOT / H_TARGETED_RELATIVE / "phase_b/targeted_task_plan.json",
        REPO_ROOT / H_TARGETED_RELATIVE / "phase_b/targeted_selection.json",
        REPO_ROOT / H_TARGETED_RELATIVE / "phase_b/frozen_prospective_predictions.csv",
        REPO_ROOT / H_TARGETED_RELATIVE / "phase_b/prospective_config_statistics.json",
        REPO_ROOT / H_TARGETED_RELATIVE / "phase_b/prospective_seed_results.jsonl",
        REPO_ROOT / H_TARGETED_RELATIVE / "final_classification.json",
        REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth",
        output / "phase_0/phase_0_decision.json",
        output / "development/development_80_manifest.json",
        output / "development/fixed_outer_split_manifest.json",
        output / "development/fixed_inner_split_manifest.json",
        output / "models/conformal_calibrations.json",
        selection_path,
        prediction_path,
        checkpoint_manifest_path,
    ]
    input_paths.extend(sorted(REPO_ROOT.glob(HISTORY_GLOB)))
    for quarantined_input in (
        REPO_ROOT / QUARANTINED_RELATIVE / "quarantine_reason.json",
        quarantined_selection_path,
    ):
        if quarantined_input.is_file():
            input_paths.append(quarantined_input)
    gpus = _actual_gpus()
    plan = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "diagnostic_only": True,
        "bo_allowed": False,
        "offline_bo_allowed": False,
        "vae_training_allowed": False,
        "latent_distance_gp_allowed": False,
        "test_labels_allowed": False,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "frozen_predictions_path": str(prediction_path.resolve()),
        "frozen_predictions_sha256": sha256_file(prediction_path),
        "model_checkpoint_manifest_path": str(checkpoint_manifest_path.resolve()),
        "model_checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "code_hashes": {str(path.resolve()): sha256_file(path) for path in code_paths},
        "input_hashes": {str(path.resolve()): sha256_file(path) for path in input_paths},
        "protected_hashes": load_json(output / "protected_hashes_before.json"),
        "development": {
            "architecture_count": 80,
            "configuration_count": 96,
            "outer_protocol": "architecture_group_repeated",
            "outer_repeats": OUTER_REPEATS,
            "inner_architecture_grouped": True,
            "training_seeds": list(TRAINING_SEEDS),
            "predictor_family": "Linear(input,32)-ReLU-Linear(32,1)",
            "same_capacity_all_trained_representations": True,
            "learning_rate_grid": list(LEARNING_RATES),
            "maximum_epochs": MAXIMUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "main_loss_weighting": "architecture_balanced",
            "sensitivity_loss_weighting": "architecture_balanced_within_architecture_uncertainty",
            "standardization": "training_fold_only_architecture_weighted",
            "conformal": "architecture-group max absolute OOF residual, 90%",
            "time_prefix_analysis": False,
        },
        "candidate_selection": {
            "candidate_count": 24,
            "strata": {"hash_uniform": 12, "structure_diverse": 12},
            "uses_label_or_prediction_values": False,
            "exact_historical_operations_edges_hp": True,
            "redecoded": False,
        },
        "labeling_protocol": {
            "dataset": "cora",
            "hp_mode": "global4",
            "max_epochs": 300,
            "patience": 80,
            "metric": "validation_accuracy",
            "track_test": False,
            "dataset_content_fingerprint": DATASET_FINGERPRINT,
            "split_fingerprint": SPLIT_FINGERPRINT,
            "initial_seeds": list(INITIAL_SEEDS),
            "adaptive_seeds": list(ADAPTIVE_SEEDS),
            "adaptive_rule": "validation_t_ci95_half_width_gt_0.006_or_fewer_than_15_successful_seeds",
            "adaptive_rule_uses_prediction_correctness": False,
            "maximum_retries": 2,
        },
        "initial_seeds": list(INITIAL_SEEDS),
        "adaptive_seeds": list(ADAPTIVE_SEEDS),
        "num_shards": 2,
        "shard_to_gpu_uuid": {str(row["index"]): row["uuid"] for row in gpus},
        "gpu_inventory": gpus,
        "expected_initial_task_count": 24 * len(INITIAL_SEEDS),
        "maximum_total_task_count": 24 * (len(INITIAL_SEEDS) + len(ADAPTIVE_SEEDS)),
        "merge_key": ["config_id", "architecture_fingerprint", "seed"],
    }
    plan_path = output / "phase_4/label_task_plan.json"
    dump_json(plan_path, plan)
    write_sha256_sidecar(plan_path)
    dump_json(
        output / "phase_4/freeze_audit.json",
        {
            "passed": True,
            "frozen_before_prospective_labels": True,
            "prospective_label_file_existed_at_freeze": False,
            "selection_sha256": plan["selection_sha256"],
            "frozen_predictions_sha256": plan["frozen_predictions_sha256"],
            "model_checkpoint_manifest_sha256": plan["model_checkpoint_manifest_sha256"],
            "task_plan_sha256": sha256_file(plan_path),
            "protected_hashes": plan["protected_hashes"],
        },
    )
    return {"classification": "PREPARED_AND_FROZEN", "plan_path": str(plan_path), "plan_sha256": sha256_file(plan_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("phase0", "prepare", "verify-plan"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE)
    parser.add_argument("--plan", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if args.command == "phase0":
        if output.exists():
            raise FileExistsError(f"output directory already exists: {output}")
        output.mkdir(parents=True)
        dump_json(output / "protected_hashes_before.json", protected_hashes())
        status = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True).stdout.splitlines()
        dump_json(output / "workspace_before.json", {"created_at": utc_now(), "branch": subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True).stdout.strip(), "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True).stdout.strip(), "git_status_short": status})
        result = run_phase0(output)
    elif args.command == "prepare":
        if not output.is_dir():
            raise FileNotFoundError(output)
        result = prepare_experiment(output)
    else:
        plan_path = (args.plan or output / "phase_4/label_task_plan.json").resolve()
        result = {"classification": "FROZEN_PLAN_VERIFIED", "plan_sha256": sha256_file(plan_path), "plan": verify_frozen_plan(plan_path)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
