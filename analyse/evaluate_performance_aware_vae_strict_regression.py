#!/usr/bin/env python3
"""Evaluate and classify the frozen strict performance-aware VAE experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from scipy.stats import kendalltau, pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import (
    architecture_fingerprint,
    config_from_igraph,
    config_to_igraph,
    load_joint_vae,
)
from analyse.performance_aware_vae_pilot import (
    contains_test_field,
    load_replay_dataset,
    load_validation_labels,
    reliable_pairs,
    set_all_seeds,
    sha256_file,
)
from analyse.performance_aware_vae_strict_regression import (
    DATASET_VARIANTS,
    FORMAT_VERSION,
    LAMBDAS,
    RESULTS_DIRNAME,
    StrictPerformanceHead,
    TRAINING_SEEDS,
    _dump_csv,
    _dump_json,
    module_state_sha256,
    verify_frozen_plan,
)


METRICS = (
    "late_kendall",
    "top_region_kendall",
    "top_pairwise_accuracy",
    "reliable_top_pairwise_accuracy",
    "ndcg_at_10",
    "top_10_regret",
    "rmse",
    "mae",
    "prediction_interval_coverage",
)
PRIMARY_PROTOCOL = "group_repeated"


def _load_task_results(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / RESULTS_DIRNAME).glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def load_strict_checkpoint(
    path: Path, base_checkpoint: Path, device: torch.device
) -> tuple[nn.Module, StrictPerformanceHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported strict checkpoint")
    if sha256_file(base_checkpoint) != payload["base_checkpoint_sha256"]:
        raise RuntimeError("base checkpoint changed since task training")
    model = load_joint_vae(base_checkpoint, "global4", device=device)
    if payload["arch_vae_state"] is not None:
        model.arch_vae.load_state_dict(payload["arch_vae_state"])
    model.eval()
    head = StrictPerformanceHead(int(payload["head_input_dim"])).to(device)
    head.load_state_dict(payload["head_state"])
    head.eval()
    return model, head, payload


def _aggregate_seed_and_split_predictions(
    rows: Sequence[Mapping[str, Any]], model_name: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)
    output = []
    for config_id, values in sorted(grouped.items()):
        predictions = np.asarray([float(row["prediction"]) for row in values])
        within_variances = np.asarray(
            [
                ((float(row["prediction_interval_high"]) - float(row["prediction"])) / 1.96)
                ** 2
                for row in values
            ]
        )
        total_std = math.sqrt(
            max(float(np.mean(within_variances + predictions**2) - predictions.mean() ** 2), 1e-12)
        )
        first = values[0]
        output.append(
            {
                "config_id": config_id,
                "architecture_fingerprint": first["architecture_fingerprint"],
                "validation_mean": float(first["validation_mean"]),
                "validation_ci95_low": float(first["validation_ci95_low"]),
                "validation_ci95_high": float(first["validation_ci95_high"]),
                "time_index": int(first["time_index"]),
                "prediction": float(predictions.mean()),
                "prediction_interval_low": float(predictions.mean() - 1.96 * total_std),
                "prediction_interval_high": float(predictions.mean() + 1.96 * total_std),
                "prediction_source_count": len(values),
                "model": model_name,
            }
        )
    return output


def _architecture_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("bootstrap_architecture_fingerprint", row["architecture_fingerprint"]))].append(row)
    result = []
    for cluster, values in grouped.items():
        result.append(
            {
                "cluster": cluster,
                "original_architecture_fingerprint": values[0]["architecture_fingerprint"],
                "validation_mean": statistics.fmean(float(row["validation_mean"]) for row in values),
                "prediction": statistics.fmean(float(row["prediction"]) for row in values),
                "time_index": max(int(row["time_index"]) for row in values),
                "coverage": statistics.fmean(
                    float(
                        float(row["prediction_interval_low"])
                        <= float(row["validation_mean"])
                        <= float(row["prediction_interval_high"])
                    )
                    for row in values
                ),
                "errors": [
                    float(row["prediction"]) - float(row["validation_mean"])
                    for row in values
                ],
            }
        )
    return result


def metric_bundle(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {metric: None for metric in METRICS}
    architecture_rows = _architecture_rows(rows)
    actual = np.asarray([row["validation_mean"] for row in architecture_rows])
    predicted = np.asarray([row["prediction"] for row in architecture_rows])
    order = sorted(
        range(len(architecture_rows)),
        key=lambda index: (
            architecture_rows[index]["time_index"],
            architecture_rows[index]["cluster"],
        ),
    )[-min(50, len(architecture_rows)) :]
    late_tau = kendalltau(actual[order], predicted[order]).statistic if len(order) > 1 else None
    top_count = max(1, math.ceil(0.2 * len(architecture_rows)))
    top_indices = np.argsort(-actual, kind="stable")[:top_count]
    top_tau = (
        kendalltau(actual[top_indices], predicted[top_indices]).statistic
        if len(top_indices) > 1
        else None
    )
    top_original_architectures = {
        architecture_rows[index]["original_architecture_fingerprint"] for index in top_indices
    }
    top_rows = [
        row for row in rows if str(row["architecture_fingerprint"]) in top_original_architectures
    ]
    raw_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    reliable_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for left in range(len(top_rows)):
        for right in range(left + 1, len(top_rows)):
            a, b = top_rows[left], top_rows[right]
            left_fp = str(a["architecture_fingerprint"])
            right_fp = str(b["architecture_fingerprint"])
            if left_fp == right_fp:
                continue
            delta = float(a["validation_mean"]) - float(b["validation_mean"])
            if delta == 0:
                continue
            key = tuple(sorted((left_fp, right_fp)))
            correct = float(
                (float(a["prediction"]) - float(b["prediction"])) * delta > 0
            )
            raw_by_pair[key].append(correct)
            disjoint = (
                float(a["validation_ci95_low"]) > float(b["validation_ci95_high"])
                or float(b["validation_ci95_low"]) > float(a["validation_ci95_high"])
            )
            if abs(delta) >= 0.006 and disjoint:
                reliable_by_pair[key].append(correct)
    raw_accuracy = (
        statistics.fmean(statistics.fmean(values) for values in raw_by_pair.values())
        if raw_by_pair
        else None
    )
    reliable_accuracy = (
        statistics.fmean(
            statistics.fmean(values) for values in reliable_by_pair.values()
        )
        if reliable_by_pair
        else None
    )
    relevance = actual - actual.min() + 1e-9
    ndcg = float(
        ndcg_score(
            relevance.reshape(1, -1),
            predicted.reshape(1, -1),
            k=min(10, len(actual)),
        )
    )
    predicted_top = np.argsort(-predicted, kind="stable")[: min(10, len(actual))]
    errors = [np.asarray(row["errors"]) for row in architecture_rows]
    reliable_architectures = {item for pair in reliable_by_pair for item in pair}
    return {
        "observation_count": len(rows),
        "architecture_count": len(architecture_rows),
        "late_kendall": None if late_tau is None or not np.isfinite(late_tau) else float(late_tau),
        "top_region_kendall": None if top_tau is None or not np.isfinite(top_tau) else float(top_tau),
        "top_pairwise_accuracy": raw_accuracy,
        "reliable_top_pairwise_accuracy": reliable_accuracy,
        "ndcg_at_10": ndcg,
        "top_10_regret": float(actual.max() - actual[predicted_top].max()),
        "rmse": math.sqrt(statistics.fmean(float(np.mean(error**2)) for error in errors)),
        "mae": statistics.fmean(float(np.mean(np.abs(error))) for error in errors),
        "prediction_interval_coverage": statistics.fmean(row["coverage"] for row in architecture_rows),
        "raw_top_pair_count": len(raw_by_pair),
        "reliable_top_pair_count": len(reliable_by_pair),
        "top_unique_architecture_count": len(top_original_architectures),
        "reliable_top_unique_architecture_count": len(reliable_architectures),
    }


def _bootstrap_rows(
    rows: Sequence[Mapping[str, Any]], sampled_architectures: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(row)
    output = []
    copies: Counter[str] = Counter()
    for fingerprint in sampled_architectures:
        copy_index = copies[fingerprint]
        copies[fingerprint] += 1
        for row in grouped[fingerprint]:
            output.append(
                {
                    **dict(row),
                    "bootstrap_architecture_fingerprint": f"{fingerprint}#{copy_index}",
                }
            )
    return output


def shared_bootstrap_indices(
    architectures: Sequence[str], replicates: int, seed: int
) -> list[list[str]]:
    ordered = sorted(architectures)
    rng = np.random.default_rng(seed)
    return [
        [ordered[index] for index in rng.integers(0, len(ordered), size=len(ordered))]
        for _ in range(replicates)
    ]


def metric_summary(
    rows: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Sequence[str]],
) -> tuple[dict[str, Any], dict[str, list[float | None]]]:
    point = metric_bundle(rows)
    samples: dict[str, list[float | None]] = {metric: [] for metric in METRICS}
    for sampled in bootstrap:
        result = metric_bundle(_bootstrap_rows(rows, sampled))
        for metric in METRICS:
            value = result.get(metric)
            samples[metric].append(
                float(value) if value is not None and np.isfinite(value) else None
            )
    summary = dict(point)
    summary["bootstrap_unit"] = "architecture_fingerprint"
    for metric in METRICS:
        values = [value for value in samples[metric] if value is not None]
        summary[f"{metric}_ci95_low"] = (
            float(np.quantile(values, 0.025)) if values else None
        )
        summary[f"{metric}_ci95_high"] = (
            float(np.quantile(values, 0.975)) if values else None
        )
        summary[f"{metric}_bootstrap_valid_replicates"] = len(values)
    return summary, samples


def paired_difference(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate_samples: Mapping[str, Sequence[float]],
    baseline_samples: Mapping[str, Sequence[float]],
    metric: str,
) -> dict[str, Any]:
    left = candidate.get(metric)
    right = baseline.get(metric)
    direction = -1.0 if metric in {"top_10_regret", "rmse", "mae"} else 1.0
    paired = [
        direction * (float(a) - float(b))
        for a, b in zip(candidate_samples[metric], baseline_samples[metric])
        if a is not None and b is not None
    ]
    return {
        "metric": metric,
        "candidate": candidate.get("model"),
        "baseline": baseline.get("model"),
        "candidate_estimate": left,
        "baseline_estimate": right,
        "improvement_estimate": None
        if left is None or right is None
        else direction * (float(left) - float(right)),
        "improvement_ci95_low": float(np.quantile(paired, 0.025)) if paired else None,
        "improvement_ci95_high": float(np.quantile(paired, 0.975)) if paired else None,
        "bootstrap_valid_replicates": len(paired),
        "higher_improvement_is_better": True,
    }


def _fixed_gp_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_scaler = StandardScaler().fit(train_x)
    y_mean = float(train_y.mean())
    y_std = float(train_y.std())
    if y_std < 1e-8:
        y_std = 1.0
    estimator = GaussianProcessRegressor(
        kernel=RBF(length_scale=1.0, length_scale_bounds="fixed"),
        alpha=1e-4,
        optimizer=None,
        normalize_y=False,
        random_state=20260821,
    )
    estimator.fit(x_scaler.transform(train_x), (train_y - y_mean) / y_std)
    predicted, predicted_std = estimator.predict(
        x_scaler.transform(validation_x), return_std=True
    )
    return predicted * y_std + y_mean, np.maximum(predicted_std * y_std, 1e-6)


def _latent_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    representation: str,
    model: torch.nn.Module | None,
) -> np.ndarray:
    hp = np.asarray([row["raw_hp_vector"] for row in rows], dtype=np.float64)
    if representation == "Zs":
        latent = np.asarray([row["z_arch"] for row in rows], dtype=np.float64)
    elif representation == "Ze":
        latent = np.asarray([row["mu_arch"] for row in rows], dtype=np.float64)
    elif representation == "FT":
        if model is None:
            raise ValueError("FT latent requires a model")
        with torch.no_grad():
            mu, _ = model.arch_vae.encode([config_to_igraph(row) for row in rows])
        latent = mu.detach().cpu().numpy().astype(np.float64)
    else:
        raise ValueError(representation)
    return np.concatenate((latent, hp), axis=1)


def _gp_predictions_for_task(
    task: Mapping[str, Any],
    model: torch.nn.Module,
    split: Mapping[str, Any],
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    train_rows = [labels_by_id[item] for item in split["train_config_ids"]]
    validation_rows = [labels_by_id[item] for item in split["validation_config_ids"]]
    predicted, predicted_std = _fixed_gp_predict(
        _latent_features(train_rows, representation="FT", model=model),
        np.asarray([row["validation_mean"] for row in train_rows], dtype=np.float64),
        _latent_features(validation_rows, representation="FT", model=model),
    )
    return [
        {
            "config_id": row["config_id"],
            "architecture_fingerprint": row["architecture_fingerprint"],
            "validation_mean": row["validation_mean"],
            "validation_ci95_low": row["validation_ci95_low"],
            "validation_ci95_high": row["validation_ci95_high"],
            "time_index": row["time_index"],
            "prediction": float(value),
            "prediction_interval_low": float(value - 1.96 * std),
            "prediction_interval_high": float(value + 1.96 * std),
            "model": f"ft_latent_gp_lambda_{task['lambda']:g}",
            "dataset_variant": task["dataset_variant"],
            "protocol": task["protocol"],
            "split_id": task["split_id"],
            "training_seed": task["training_seed"],
        }
        for row, value, std in zip(validation_rows, predicted, predicted_std)
    ]


def _original_gp_predictions(
    labels: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    lookup = {str(row["config_id"]): row for row in labels}
    output = []
    for split in manifest:
        train_rows = [lookup[item] for item in split["train_config_ids"]]
        validation_rows = [lookup[item] for item in split["validation_config_ids"]]
        for representation in ("Zs", "Ze"):
            predicted, predicted_std = _fixed_gp_predict(
                _latent_features(train_rows, representation=representation, model=None),
                np.asarray([row["validation_mean"] for row in train_rows], dtype=np.float64),
                _latent_features(validation_rows, representation=representation, model=None),
            )
            for row, value, std in zip(validation_rows, predicted, predicted_std):
                output.append(
                    {
                        "config_id": row["config_id"],
                        "architecture_fingerprint": row["architecture_fingerprint"],
                        "validation_mean": row["validation_mean"],
                        "validation_ci95_low": row["validation_ci95_low"],
                        "validation_ci95_high": row["validation_ci95_high"],
                        "time_index": row["time_index"],
                        "prediction": float(value),
                        "prediction_interval_low": float(value - 1.96 * std),
                        "prediction_interval_high": float(value + 1.96 * std),
                        "model": f"original_{representation.lower()}_fixed_gp",
                        "dataset_variant": split["dataset_variant"],
                        "protocol": split["protocol"],
                        "split_id": split["split_id"],
                        "training_seed": None,
                    }
                )
    return output


def _effective_rank(eigenvalues: np.ndarray) -> float:
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    if values.sum() <= 0:
        return 0.0
    probabilities = values / values.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _geometry_metrics(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    base_mu: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    representatives: dict[str, Mapping[str, Any]] = {}
    values_by_architecture: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        fingerprint = str(row["architecture_fingerprint"])
        representatives.setdefault(fingerprint, row)
        values_by_architecture[fingerprint].append(float(row["validation_mean"]))
    fingerprints = sorted(representatives)
    graphs = [config_to_igraph(representatives[item]) for item in fingerprints]
    with torch.no_grad():
        mu, _ = model.arch_vae.encode(graphs)
    latent = mu.detach().cpu().numpy().astype(np.float64)
    covariance = np.cov(latent, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    pair_distances = []
    performance_differences = []
    architecture_means = {
        key: statistics.fmean(value) for key, value in values_by_architecture.items()
    }
    top = set(
        sorted(fingerprints, key=lambda item: architecture_means[item], reverse=True)[
            : max(1, math.ceil(0.2 * len(fingerprints)))
        ]
    )
    top_distances, top_differences = [], []
    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            distance = float(np.linalg.norm(latent[left] - latent[right]))
            difference = abs(
                architecture_means[fingerprints[left]]
                - architecture_means[fingerprints[right]]
            )
            pair_distances.append(distance)
            performance_differences.append(difference)
            if fingerprints[left] in top and fingerprints[right] in top:
                top_distances.append(distance)
                top_differences.append(difference)
    correlation = (
        float(pearsonr(pair_distances, performance_differences).statistic)
        if len(pair_distances) > 2 and np.std(pair_distances) > 0
        else None
    )
    top_correlation = (
        float(pearsonr(top_distances, top_differences).statistic)
        if len(top_distances) > 2 and np.std(top_distances) > 0
        else None
    )
    drift = [
        float(np.linalg.norm(latent[index] - base_mu[fingerprint]))
        for index, fingerprint in enumerate(fingerprints)
    ]
    participation = (
        float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
        if np.sum(eigenvalues**2) > 0
        else 0.0
    )
    return {
        "architecture_count": len(fingerprints),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "effective_rank": _effective_rank(eigenvalues),
        "participation_ratio": participation,
        "pairwise_distance_min": float(np.min(pair_distances)),
        "pairwise_distance_median": float(np.median(pair_distances)),
        "pairwise_distance_mean": float(np.mean(pair_distances)),
        "pairwise_distance_max": float(np.max(pair_distances)),
        "distance_abs_validation_correlation": correlation,
        "top_distance_abs_validation_correlation": top_correlation,
        "encoder_representation_drift_mean": float(np.mean(drift)),
        "encoder_representation_drift_median": float(np.median(drift)),
    }


def _valid_decoded_config(model: torch.nn.Module, graph: Any) -> dict[str, Any] | None:
    try:
        config = config_from_igraph(model, graph)
        if graph.vcount() < 2 or not graph.is_dag():
            return None
        return config
    except (ValueError, TypeError, IndexError, KeyError):
        return None


def reconstruction_metrics(
    model: torch.nn.Module,
    replay: Sequence[tuple[Any, torch.Tensor]],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    graphs = [replay[index][0].copy() for index in indices]
    originals = [architecture_fingerprint(config_from_igraph(model, graph)) for graph in graphs]
    model.eval()
    with torch.no_grad():
        mu, _ = model.arch_vae.encode([graph.copy() for graph in graphs])
        decoded = model.arch_vae.decode(mu, stochastic=False)
    valid_configs = [_valid_decoded_config(model, graph) for graph in decoded]
    valid_fingerprints = [
        architecture_fingerprint(config) if config is not None else None
        for config in valid_configs
    ]
    exact = [
        float(candidate == original)
        for candidate, original in zip(valid_fingerprints, originals)
    ]
    valid = [item for item in valid_fingerprints if item is not None]
    cycle, normalized = [], []
    for index, config in enumerate(valid_configs):
        if config is None:
            continue
        with torch.no_grad():
            encoded, _ = model.arch_vae.encode([config_to_igraph(config)])
        distance = float(torch.linalg.vector_norm(mu[index] - encoded[0]).cpu())
        cycle.append(distance)
        normalized.append(distance / math.sqrt(mu.shape[1]))
    uniqueness = len(set(valid)) / max(len(valid), 1)
    return {
        "sample_count": len(indices),
        "validity": len(valid) / len(indices),
        "exact_fingerprint_reconstruction": statistics.fmean(exact),
        "uniqueness": uniqueness,
        "many_z_to_one_collision_rate": 1.0 - uniqueness,
        "cycle_distance_median": statistics.median(cycle) if cycle else None,
        "normalized_cycle_distance_median": statistics.median(normalized)
        if normalized
        else None,
    }


def _holm(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["raw_p_value"]))
    count = len(ordered)
    adjusted = []
    running = 0.0
    for rank, row in enumerate(ordered):
        value = min(1.0, (count - rank) * float(row["raw_p_value"]))
        running = max(running, value)
        adjusted.append({**dict(row), "holm_adjusted_p_value": running})
    return sorted(adjusted, key=lambda row: float(row["lambda"]))


def _gate(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    differences: Mapping[str, Mapping[str, Any]],
    *,
    require_reconstruction: bool,
    reconstruction_passed: bool,
    no_new_collapse: bool,
) -> dict[str, Any]:
    checks = {
        "top_region_kendall_ci_low_gt_zero": candidate.get(
            "top_region_kendall_ci95_low"
        )
        is not None
        and float(candidate["top_region_kendall_ci95_low"]) > 0,
        "late_kendall_improvement_at_least_0_05": differences["late_kendall"].get(
            "improvement_estimate"
        )
        is not None
        and float(differences["late_kendall"]["improvement_estimate"]) >= 0.05,
        "late_kendall_improvement_ci_low_gt_zero": differences["late_kendall"].get(
            "improvement_ci95_low"
        )
        is not None
        and float(differences["late_kendall"]["improvement_ci95_low"]) > 0,
        "top_pairwise_ci_low_gt_0_5": candidate.get(
            "top_pairwise_accuracy_ci95_low"
        )
        is not None
        and float(candidate["top_pairwise_accuracy_ci95_low"]) > 0.5,
        "reliable_top_pair_count_at_least_20": int(
            candidate.get("reliable_top_pair_count", 0)
        )
        >= 20,
        "top_architectures_at_least_8": int(
            candidate.get("top_unique_architecture_count", 0)
        )
        >= 8,
        "ndcg_not_lower": candidate.get("ndcg_at_10") is not None
        and baseline.get("ndcg_at_10") is not None
        and float(candidate["ndcg_at_10"]) >= float(baseline["ndcg_at_10"]),
        "top10_regret_not_higher": candidate.get("top_10_regret") is not None
        and baseline.get("top_10_regret") is not None
        and float(candidate["top_10_regret"]) <= float(baseline["top_10_regret"]),
    }
    if not require_reconstruction:
        checks["prediction_interval_coverage_at_least_0_90"] = float(
            candidate.get("prediction_interval_coverage", 0.0)
        ) >= 0.90
    else:
        checks["no_new_latent_collapse"] = bool(no_new_collapse)
        checks["reconstruction_noninferiority"] = bool(reconstruction_passed)
    return {"checks": checks, "passed": all(checks.values())}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    plan_path = args.task_plan.resolve()
    plan = json.loads(plan_path.read_text())
    verify_frozen_plan(plan_path, plan)
    rows = _load_task_results(output)
    expected = {str(task["task_id"]): task for task in plan["tasks"]}
    observed = [str(row["task_id"]) for row in rows]
    duplicate = sorted(item for item, count in Counter(observed).items() if count > 1)
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    failed = sorted(
        str(row["task_id"]) for row in rows if row.get("terminal_status") != "success"
    )
    drift = []
    for row in rows:
        task = expected.get(str(row["task_id"]))
        if task and any(row.get(key) != value for key, value in task.items()):
            drift.append(str(row["task_id"]))
    if duplicate or missing or unexpected or failed or drift:
        raise RuntimeError(
            f"result merge failed duplicate={duplicate} missing={len(missing)} "
            f"unexpected={unexpected} failed={failed} drift={drift}"
        )
    rows.sort(key=lambda row: row["task_id"])
    strict_rows = [row for row in rows if row["task_kind"] == "strict_ft"]
    reduced_rows = [
        row for row in strict_rows if row["dataset_variant"] == "exclude_high_noise"
    ]
    checkpoint_hash_mismatches = []
    baseline_checkpoint_reload_count = 0
    for row in rows:
        checkpoint = Path(row["checkpoint_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != row["checkpoint_sha256"]:
            checkpoint_hash_mismatches.append(row["task_id"])
            continue
        if row["task_kind"] == "baseline":
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("format_version") != FORMAT_VERSION or payload["task"]["task_id"] != row["task_id"]:
                checkpoint_hash_mismatches.append(row["task_id"])
            baseline_checkpoint_reload_count += 1
    if checkpoint_hash_mismatches:
        raise RuntimeError(f"checkpoint integrity failed: {checkpoint_hash_mismatches}")
    _dump_json(
        output / "training_progress.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "expected": int(plan["task_count"]),
            "completed_success": len(rows),
            "completed_failed": 0,
            "remaining": 0,
            "retry_count": sum(int(row.get("retry_count", 0)) for row in rows),
            "passed": True,
        },
    )
    with (output / "strict_ft_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in strict_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with (output / "high_noise_retrained_results.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in reduced_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    training_curves = []
    gradient_rows = []
    delta_rows = []
    for row in rows:
        for trace in row["training_trace"]:
            training_curves.append(
                {
                    "task_id": row["task_id"],
                    "dataset_variant": row["dataset_variant"],
                    "protocol": row["protocol"],
                    "split_id": row["split_id"],
                    "training_seed": row["training_seed"],
                    "lambda": row["lambda"],
                    "model": row["model"],
                    **trace,
                }
            )
            gradient_rows.append(
                {
                    "task_id": row["task_id"],
                    "epoch": trace["epoch"],
                    "encoder_performance_gradient_norm": trace[
                        "encoder_performance_gradient_norm"
                    ],
                    "decoder_reconstruction_gradient_norm": trace[
                        "decoder_reconstruction_gradient_norm"
                    ],
                }
            )
        delta_rows.append({"task_id": row["task_id"], **row["module_diagnostics"]})
    _dump_csv(output / "strict_ft_training_curves.csv", training_curves)
    _dump_csv(output / "module_gradient_norms.csv", gradient_rows)
    _dump_json(output / "module_parameter_deltas.json", delta_rows)
    _dump_csv(output / "module_parameter_deltas.csv", delta_rows)

    labels = load_validation_labels(Path(plan["inputs"]["labels"]))
    labels_by_id = {str(row["config_id"]): row for row in labels}
    manifest = json.loads(Path(plan["fixed_split_manifest"]).read_text())
    split_lookup = {
        (row["dataset_variant"], row["protocol"], row["split_id"]): row
        for row in manifest
    }
    predictions = []
    for row in rows:
        model_name = (
            row["model"]
            if row["task_kind"] == "baseline"
            else f"ft_predictor_lambda_{row['lambda']:g}"
        )
        for prediction in row["validation_predictions"]:
            predictions.append(
                {
                    **prediction,
                    "model": model_name,
                    "dataset_variant": row["dataset_variant"],
                    "protocol": row["protocol"],
                    "split_id": row["split_id"],
                    "training_seed": row["training_seed"],
                }
            )
    original_gp = _original_gp_predictions(labels, manifest)
    predictions.extend(original_gp)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    evaluation_cache = output / "evaluation_task_cache"
    evaluation_cache.mkdir(exist_ok=True)
    base_checkpoint = Path(plan["inputs"]["base_checkpoint"])
    replay = load_replay_dataset(Path(plan["inputs"]["replay"]))
    base_model = load_joint_vae(base_checkpoint, "global4", device=device)
    base_mu: dict[str, np.ndarray] = {}
    for row in labels:
        fingerprint = str(row["architecture_fingerprint"])
        if fingerprint in base_mu:
            continue
        with torch.no_grad():
            mu, _ = base_model.arch_vae.encode([config_to_igraph(row)])
        base_mu[fingerprint] = mu[0].detach().cpu().numpy().astype(np.float64)
    reconstruction_indices = list(map(int, plan["reconstruction_sample_indices"]))
    base_reconstruction = reconstruction_metrics(
        base_model, replay, reconstruction_indices, device
    )
    geometry_rows = []
    reconstruction_rows = []
    strict_checkpoint_reload_count = 0
    for index, task_row in enumerate(strict_rows, 1):
        cache_path = evaluation_cache / f"{task_row['task_id']}.json"
        model, _head, _payload = load_strict_checkpoint(
            Path(task_row["checkpoint_path"]), base_checkpoint, device
        )
        strict_checkpoint_reload_count += 1
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text())
        else:
            split = split_lookup[
                (
                    task_row["dataset_variant"],
                    task_row["protocol"],
                    task_row["split_id"],
                )
            ]
            gp_rows = _gp_predictions_for_task(task_row, model, split, labels_by_id)
            dataset_rows = [
                row
                for row in labels
                if task_row["dataset_variant"] == "full" or not row["high_noise"]
            ]
            geometry = _geometry_metrics(model, dataset_rows, base_mu)
            reconstruction = reconstruction_metrics(
                model, replay, reconstruction_indices, device
            )
            cached = {
                "task_id": task_row["task_id"],
                "gp_predictions": gp_rows,
                "geometry": geometry,
                "reconstruction": reconstruction,
                "checkpoint_sha256": sha256_file(task_row["checkpoint_path"]),
            }
            _dump_json(cache_path, cached)
        predictions.extend(cached["gp_predictions"])
        common = {
            "task_id": task_row["task_id"],
            "dataset_variant": task_row["dataset_variant"],
            "protocol": task_row["protocol"],
            "split_id": task_row["split_id"],
            "training_seed": task_row["training_seed"],
            "lambda": task_row["lambda"],
        }
        geometry_rows.append({**common, **cached["geometry"]})
        reconstruction_rows.append({**common, **cached["reconstruction"]})
        if index % 10 == 0:
            _dump_json(
                output / "evaluation_progress.json",
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "completed": index,
                    "expected": len(strict_rows),
                },
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if contains_test_field(predictions):
        raise RuntimeError("strict predictions contain a test-named field")
    _dump_csv(output / "all_validation_predictions.csv", predictions)

    pooled: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    prediction_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        prediction_groups[(row["dataset_variant"], row["protocol"], row["model"])].append(row)
    for key, values in prediction_groups.items():
        pooled[key] = _aggregate_seed_and_split_predictions(values, key[2])
    bootstraps = {}
    for dataset_variant in DATASET_VARIANTS:
        for protocol in ("time_prefix", "group_repeated"):
            architecture_sets = [
                {str(row["architecture_fingerprint"]) for row in values}
                for (variant, observed_protocol, _model), values in pooled.items()
                if variant == dataset_variant and observed_protocol == protocol
            ]
            if not architecture_sets:
                raise RuntimeError(f"no pooled architectures for {dataset_variant}/{protocol}")
            if any(values != architecture_sets[0] for values in architecture_sets[1:]):
                raise RuntimeError(
                    f"models do not share evaluation architectures for {dataset_variant}/{protocol}"
                )
            bootstraps[(dataset_variant, protocol)] = shared_bootstrap_indices(
                sorted(architecture_sets[0]),
                int(plan["bootstrap"]["replicates"]),
                int(plan["bootstrap"]["seed"]),
            )
    metric_rows = []
    bootstrap_samples = {}
    for key, values in sorted(pooled.items()):
        summary, samples = metric_summary(values, bootstraps[(key[0], key[1])])
        row = {
            "dataset_variant": key[0],
            "protocol": key[1],
            "model": key[2],
            **summary,
        }
        metric_rows.append(row)
        bootstrap_samples[key] = samples
    metric_lookup = {
        (row["dataset_variant"], row["protocol"], row["model"]): row
        for row in metric_rows
    }
    paired_rows = []
    for dataset_variant in DATASET_VARIANTS:
        for protocol in ("time_prefix", "group_repeated"):
            candidates = [
                f"ft_predictor_lambda_{value:g}" for value in LAMBDAS
            ] + [f"ft_latent_gp_lambda_{value:g}" for value in LAMBDAS] + [
                "frozen_hybrid_head"
            ]
            for candidate_name in candidates:
                baseline_name = (
                    "frozen_head"
                    if candidate_name.startswith("ft_predictor")
                    or candidate_name == "frozen_hybrid_head"
                    else "original_ze_fixed_gp"
                )
                candidate_key = (dataset_variant, protocol, candidate_name)
                baseline_key = (dataset_variant, protocol, baseline_name)
                candidate = metric_lookup[candidate_key]
                baseline = metric_lookup[baseline_key]
                for metric in METRICS:
                    paired_rows.append(
                        {
                            "dataset_variant": dataset_variant,
                            "protocol": protocol,
                            **paired_difference(
                                candidate,
                                baseline,
                                bootstrap_samples[candidate_key],
                                bootstrap_samples[baseline_key],
                                metric,
                            ),
                        }
                    )
    _dump_json(output / "paired_metric_differences.json", paired_rows)
    _dump_csv(output / "paired_metric_differences.csv", paired_rows)

    head_rows = [row for row in metric_rows if row["model"] == "frozen_head"]
    hybrid_rows = [
        row for row in metric_rows if row["model"] == "frozen_hybrid_head"
    ]
    ft_predictor_rows = [
        row for row in metric_rows if row["model"].startswith("ft_predictor")
    ]
    ft_gp_rows = [row for row in metric_rows if "fixed_gp" in row["model"] or row["model"].startswith("ft_latent_gp")]
    for stem, selected in (
        ("head_metrics", head_rows),
        ("hybrid_metrics", hybrid_rows),
        ("ft_predictor_metrics", ft_predictor_rows),
        ("ft_gp_metrics", ft_gp_rows),
    ):
        _dump_json(output / f"{stem}.json", selected)
        _dump_csv(output / f"{stem}.csv", selected)
    regret_rows = [
        {
            "dataset_variant": row["dataset_variant"],
            "protocol": row["protocol"],
            "model": row["model"],
            "top_10_regret": row["top_10_regret"],
            "ci95_low": row["top_10_regret_ci95_low"],
            "ci95_high": row["top_10_regret_ci95_high"],
        }
        for row in metric_rows
    ]
    reliable_rows = [
        {
            "dataset_variant": row["dataset_variant"],
            "protocol": row["protocol"],
            "model": row["model"],
            "raw_pair_count": row["raw_top_pair_count"],
            "reliable_pair_count": row["reliable_top_pair_count"],
            "unique_architecture_count": row["top_unique_architecture_count"],
            "reliable_unique_architecture_count": row[
                "reliable_top_unique_architecture_count"
            ],
            "pairwise_accuracy": row["top_pairwise_accuracy"],
            "pairwise_ci95_low": row["top_pairwise_accuracy_ci95_low"],
            "pairwise_ci95_high": row["top_pairwise_accuracy_ci95_high"],
            "reliable_pairwise_accuracy": row[
                "reliable_top_pairwise_accuracy"
            ],
            "reliable_pairwise_ci95_low": row[
                "reliable_top_pairwise_accuracy_ci95_low"
            ],
            "reliable_pairwise_ci95_high": row[
                "reliable_top_pairwise_accuracy_ci95_high"
            ],
        }
        for row in metric_rows
    ]
    _dump_json(output / "top10_regret_report.json", regret_rows)
    _dump_csv(output / "top10_regret_report.csv", regret_rows)
    _dump_json(output / "reliable_pair_report.json", reliable_rows)
    _dump_csv(output / "reliable_pair_report.csv", reliable_rows)

    _dump_json(output / "latent_geometry_report.json", geometry_rows)
    _dump_csv(output / "latent_geometry_report.csv", geometry_rows)
    spectrum_rows = [
        {
            key: row[key]
            for key in (
                "task_id",
                "dataset_variant",
                "protocol",
                "split_id",
                "training_seed",
                "lambda",
                "covariance_eigenvalues",
            )
        }
        for row in geometry_rows
    ]
    effective_rows = [
        {
            key: row[key]
            for key in (
                "task_id",
                "dataset_variant",
                "protocol",
                "split_id",
                "training_seed",
                "lambda",
                "effective_rank",
                "participation_ratio",
            )
        }
        for row in geometry_rows
    ]
    _dump_json(output / "latent_spectrum.json", spectrum_rows)
    _dump_csv(output / "latent_spectrum.csv", spectrum_rows)
    _dump_json(output / "effective_rank_report.json", effective_rows)
    _dump_csv(output / "effective_rank_report.csv", effective_rows)

    base_reconstruction_row = {"model": "base_checkpoint", **base_reconstruction}
    reconstruction_report = {
        "baseline": base_reconstruction_row,
        "candidate_tasks": reconstruction_rows,
        "baseline_capability_warning": (
            "Passing noninferiority does not imply good reconstruction; the base exact "
            "fingerprint reconstruction rate is intrinsically weak."
        ),
    }
    _dump_json(output / "reconstruction_report.json", reconstruction_report)
    _dump_csv(
        output / "reconstruction_report.csv",
        [base_reconstruction_row, *reconstruction_rows],
    )
    cycle_rows = [
        {
            key: row.get(key)
            for key in (
                "task_id",
                "dataset_variant",
                "protocol",
                "split_id",
                "training_seed",
                "lambda",
                "cycle_distance_median",
                "normalized_cycle_distance_median",
            )
        }
        for row in reconstruction_rows
    ]
    _dump_json(output / "cycle_distance_report.json", cycle_rows)
    _dump_csv(output / "cycle_distance_report.csv", cycle_rows)

    reconstruction_by_lambda = {}
    for value in LAMBDAS:
        selected = [row for row in reconstruction_rows if float(row["lambda"]) == value]
        worst = {
            "validity_min": min(row["validity"] for row in selected),
            "exact_reconstruction_drop_max": max(
                base_reconstruction["exact_fingerprint_reconstruction"]
                - row["exact_fingerprint_reconstruction"]
                for row in selected
            ),
            "uniqueness_drop_max": max(
                base_reconstruction["uniqueness"] - row["uniqueness"] for row in selected
            ),
            "collision_increase_max": max(
                row["many_z_to_one_collision_rate"]
                - base_reconstruction["many_z_to_one_collision_rate"]
                for row in selected
            ),
            "normalized_cycle_ratio_max": max(
                row["normalized_cycle_distance_median"]
                / max(base_reconstruction["normalized_cycle_distance_median"], 1e-12)
                for row in selected
            ),
        }
        worst["passed"] = bool(
            worst["validity_min"] >= 0.99
            and worst["exact_reconstruction_drop_max"] <= 0.02
            and worst["uniqueness_drop_max"] <= 0.02
            and worst["collision_increase_max"] <= 0.05
            and worst["normalized_cycle_ratio_max"] <= 1.10
        )
        reconstruction_by_lambda[f"{value:g}"] = worst

    base_geometry = _geometry_metrics(base_model, labels, base_mu)
    geometry_by_lambda = {}
    for value in LAMBDAS:
        values = [row for row in geometry_rows if float(row["lambda"]) == value]
        min_rank = min(row["effective_rank"] for row in values)
        min_participation = min(row["participation_ratio"] for row in values)
        geometry_by_lambda[f"{value:g}"] = {
            "effective_rank_before": base_geometry["effective_rank"],
            "effective_rank_after_median": statistics.median(
                row["effective_rank"] for row in values
            ),
            "effective_rank_after_min": min_rank,
            "effective_rank_ratio_min": min_rank
            / max(base_geometry["effective_rank"], 1e-12),
            "participation_ratio_before": base_geometry["participation_ratio"],
            "participation_ratio_after_min": min_participation,
            "new_single_or_two_dimensional_collapse": min_rank < 2.0,
            "passed": bool(
                min_rank / max(base_geometry["effective_rank"], 1e-12) >= 0.90
                and min_rank >= 2.0
            ),
        }

    differences_lookup = {
        (
            row["dataset_variant"],
            row["protocol"],
            row["candidate"],
            row["baseline"],
            row["metric"],
        ): row
        for row in paired_rows
    }
    predictor_gates = {}
    gp_gates = {}
    holm_input = []
    for value in LAMBDAS:
        predictor_name = f"ft_predictor_lambda_{value:g}"
        gp_name = f"ft_latent_gp_lambda_{value:g}"
        predictor = metric_lookup[("full", PRIMARY_PROTOCOL, predictor_name)]
        head = metric_lookup[("full", PRIMARY_PROTOCOL, "frozen_head")]
        gp = metric_lookup[("full", PRIMARY_PROTOCOL, gp_name)]
        ze = metric_lookup[("full", PRIMARY_PROTOCOL, "original_ze_fixed_gp")]
        pred_diffs = {
            metric: differences_lookup[
                ("full", PRIMARY_PROTOCOL, predictor_name, "frozen_head", metric)
            ]
            for metric in METRICS
        }
        gp_diffs = {
            metric: differences_lookup[
                ("full", PRIMARY_PROTOCOL, gp_name, "original_ze_fixed_gp", metric)
            ]
            for metric in METRICS
        }
        predictor_gates[f"{value:g}"] = _gate(
            predictor,
            head,
            pred_diffs,
            require_reconstruction=False,
            reconstruction_passed=reconstruction_by_lambda[f"{value:g}"]["passed"],
            no_new_collapse=geometry_by_lambda[f"{value:g}"]["passed"],
        )
        gp_gates[f"{value:g}"] = _gate(
            gp,
            ze,
            gp_diffs,
            require_reconstruction=True,
            reconstruction_passed=reconstruction_by_lambda[f"{value:g}"]["passed"],
            no_new_collapse=geometry_by_lambda[f"{value:g}"]["passed"],
        )
        late_samples = [
            a - b
            for a, b in zip(
                bootstrap_samples[("full", PRIMARY_PROTOCOL, predictor_name)][
                    "late_kendall"
                ],
                bootstrap_samples[("full", PRIMARY_PROTOCOL, "frozen_head")][
                    "late_kendall"
                ],
            )
            if a is not None and b is not None
        ]
        raw_p = (
            (1 + sum(value_sample <= 0 for value_sample in late_samples))
            / (1 + len(late_samples))
            if late_samples
            else 1.0
        )
        holm_input.append(
            {
                "lambda": value,
                "hypothesis": "FT predictor late-Kendall improvement over frozen Head",
                "raw_p_value": raw_p,
            }
        )
    holm = _holm(holm_input)
    _dump_json(output / "multiple_testing_correction.json", holm)

    hybrid = metric_lookup[("full", PRIMARY_PROTOCOL, "frozen_hybrid_head")]
    head = metric_lookup[("full", PRIMARY_PROTOCOL, "frozen_head")]
    hybrid_differences = {
        metric: differences_lookup[
            ("full", PRIMARY_PROTOCOL, "frozen_hybrid_head", "frozen_head", metric)
        ]
        for metric in METRICS
    }
    hybrid_gate = _gate(
        hybrid,
        head,
        hybrid_differences,
        require_reconstruction=False,
        reconstruction_passed=True,
        no_new_collapse=True,
    )
    head_absolute_checks = {
        "top_region_kendall_ci_low_gt_zero": head.get(
            "top_region_kendall_ci95_low"
        )
        is not None
        and float(head["top_region_kendall_ci95_low"]) > 0,
        "top_pairwise_ci_low_gt_0_5": head.get("top_pairwise_accuracy_ci95_low")
        is not None
        and float(head["top_pairwise_accuracy_ci95_low"]) > 0.5,
        "reliable_top_pair_count_at_least_20": int(
            head["reliable_top_pair_count"]
        )
        >= 20,
        "top_architectures_at_least_8": int(head["top_unique_architecture_count"])
        >= 8,
        "prediction_interval_coverage_at_least_0_90": float(
            head["prediction_interval_coverage"]
        )
        >= 0.90,
    }
    head_success = all(head_absolute_checks.values())
    training_signal_passed = all(
        row["module_diagnostics"]["encoder_performance_gradient_norm_max"] > 0
        and row["module_diagnostics"]["decoder_reconstruction_gradient_norm_max"] > 0
        and row["module_diagnostics"]["encoder_parameter_hash_before"]
        != row["module_diagnostics"]["encoder_parameter_hash_after"]
        and row["module_diagnostics"]["decoder_parameter_hash_before"]
        != row["module_diagnostics"]["decoder_parameter_hash_after"]
        for row in strict_rows
    )
    full_predictor_success = {
        value: predictor_gates[f"{value:g}"]["passed"] for value in LAMBDAS
    }
    full_gp_success = {value: gp_gates[f"{value:g}"]["passed"] for value in LAMBDAS}
    reduced_predictor_success = {}
    reduced_gp_success = {}
    for value in LAMBDAS:
        predictor_name = f"ft_predictor_lambda_{value:g}"
        gp_name = f"ft_latent_gp_lambda_{value:g}"
        predictor = metric_lookup[("exclude_high_noise", PRIMARY_PROTOCOL, predictor_name)]
        head_reduced = metric_lookup[("exclude_high_noise", PRIMARY_PROTOCOL, "frozen_head")]
        gp = metric_lookup[("exclude_high_noise", PRIMARY_PROTOCOL, gp_name)]
        ze = metric_lookup[("exclude_high_noise", PRIMARY_PROTOCOL, "original_ze_fixed_gp")]
        pred_diff = {
            metric: differences_lookup[
                (
                    "exclude_high_noise",
                    PRIMARY_PROTOCOL,
                    predictor_name,
                    "frozen_head",
                    metric,
                )
            ]
            for metric in METRICS
        }
        gp_diff = {
            metric: differences_lookup[
                (
                    "exclude_high_noise",
                    PRIMARY_PROTOCOL,
                    gp_name,
                    "original_ze_fixed_gp",
                    metric,
                )
            ]
            for metric in METRICS
        }
        reduced_predictor_success[value] = _gate(
            predictor,
            head_reduced,
            pred_diff,
            require_reconstruction=False,
            reconstruction_passed=reconstruction_by_lambda[f"{value:g}"]["passed"],
            no_new_collapse=geometry_by_lambda[f"{value:g}"]["passed"],
        )["passed"]
        reduced_gp_success[value] = _gate(
            gp,
            ze,
            gp_diff,
            require_reconstruction=True,
            reconstruction_passed=reconstruction_by_lambda[f"{value:g}"]["passed"],
            no_new_collapse=geometry_by_lambda[f"{value:g}"]["passed"],
        )["passed"]

    both_success = [
        value for value in LAMBDAS if full_predictor_success[value] and full_gp_success[value]
    ]
    predictor_only = [
        value for value in LAMBDAS if full_predictor_success[value] and not full_gp_success[value]
    ]
    noise_changed = bool(
        not any(full_predictor_success.values())
        and any(reduced_predictor_success.values())
        or not any(full_gp_success.values())
        and any(reduced_gp_success.values())
    )
    reconstruction_all = all(row["passed"] for row in reconstruction_by_lambda.values())
    if not training_signal_passed:
        classification = "TRAINING_SIGNAL_NOT_APPLIED"
        selected_lambda = None
        recommendation = "inspect_strict_training_signal"
    elif not reconstruction_all and (
        any(full_predictor_success.values()) or any(full_gp_success.values())
    ):
        classification = "RECONSTRUCTION_DEGRADED"
        selected_lambda = None
        recommendation = "do_not_advance"
    elif both_success:
        selected_lambda = min(both_success)
        classification = "PERFORMANCE_AWARE_LATENT_READY_FOR_OFFLINE_BO"
        recommendation = "strict_offline_bo_replay"
    elif predictor_only:
        selected_lambda = min(predictor_only)
        classification = "PREDICTOR_ONLY_SIGNAL"
        recommendation = "neural_predictor_discrete_candidate_search"
    elif hybrid_gate["passed"]:
        selected_lambda = None
        classification = "PREFER_HYBRID_DISCRETE_SURROGATE"
        recommendation = "H_features + calibrated predictor + discrete mutation"
    elif (
        hybrid["late_kendall"] >= head["late_kendall"]
        and hybrid["reliable_top_pair_count"] < 20
    ):
        selected_lambda = None
        classification = "HYBRID_PROMISING_BUT_UNDERPOWERED"
        recommendation = "targeted_H_reliable_pair_validation"
    elif noise_changed:
        selected_lambda = None
        classification = "NOISE_SENSITIVE_RESULT"
        recommendation = "validate_noise_sensitive_signal"
    else:
        selected_lambda = None
        classification = "STOP_PERFORMANCE_AWARE_VAE_EUCLIDEAN_GP"
        recommendation = "prioritize_H_structural_features_and_discrete_candidate_search"

    lambda_ablation = []
    for value in LAMBDAS:
        lambda_ablation.append(
            {
                "lambda": value,
                "predictor_success": full_predictor_success[value],
                "gp_success": full_gp_success[value],
                "reduced_predictor_success": reduced_predictor_success[value],
                "reduced_gp_success": reduced_gp_success[value],
                "reconstruction_passed": reconstruction_by_lambda[f"{value:g}"]["passed"],
                "geometry_passed": geometry_by_lambda[f"{value:g}"]["passed"],
            }
        )
    _dump_json(output / "lambda_ablation.json", lambda_ablation)
    _dump_csv(output / "lambda_ablation.csv", lambda_ablation)
    sensitivity = {
        "fixed_high_noise_config_ids": json.loads(
            (output / "input_audit.json").read_text()
        )["high_noise_config_ids"],
        "full_configuration_count": 76,
        "full_architecture_count": 60,
        "excluded_configuration_count": 12,
        "retrained_configuration_count": 64,
        "retrained_architecture_count": 48,
        "strict_ft_retrained_task_count": len(reduced_rows),
        "frozen_baseline_retrained_task_count": sum(
            row["task_kind"] == "baseline"
            and row["dataset_variant"] == "exclude_high_noise"
            for row in rows
        ),
        "prediction_filter_only": False,
        "changed_conclusion": noise_changed,
        "full_predictor_success": full_predictor_success,
        "reduced_predictor_success": reduced_predictor_success,
        "full_gp_success": full_gp_success,
        "reduced_gp_success": reduced_gp_success,
    }
    _dump_json(output / "high_noise_sensitivity.json", sensitivity)
    _dump_csv(
        output / "high_noise_sensitivity.csv",
        [
            {
                "lambda": value,
                "full_predictor_success": full_predictor_success[value],
                "reduced_predictor_success": reduced_predictor_success[value],
                "full_gp_success": full_gp_success[value],
                "reduced_gp_success": reduced_gp_success[value],
            }
            for value in LAMBDAS
        ],
    )
    gates = {
        "primary_protocol": PRIMARY_PROTOCOL,
        "predictor": predictor_gates,
        "fixed_kernel_gp": gp_gates,
        "hybrid": hybrid_gate,
        "head_absolute": {
            "checks": head_absolute_checks,
            "passed": head_success,
        },
        "reconstruction": reconstruction_by_lambda,
        "geometry": geometry_by_lambda,
        "training_signal_passed": training_signal_passed,
        "holm_correction": holm,
    }
    _dump_json(output / "decision_gates.json", gates)

    protected_after = {
        path: sha256_file(path) for path in plan["protected_hashes"]
    }
    protected_unchanged = protected_after == plan["protected_hashes"]
    base_checkpoint_path = str(base_checkpoint.resolve())
    base_checkpoint_unchanged = (
        protected_after[base_checkpoint_path] == plan["protected_hashes"][base_checkpoint_path]
    )
    conclusion = {
        "classification": classification,
        "selected_lambda": selected_lambda,
        "head_success": head_success,
        "hybrid_success": hybrid_gate["passed"],
        "hybrid_evidence_sufficient": hybrid["reliable_top_pair_count"] >= 20,
        "ft_predictor_success": any(full_predictor_success.values()),
        "ft_gp_geometry_success": any(full_gp_success.values()),
        "reliable_top_pair_count": max(
            metric_lookup[
                ("full", PRIMARY_PROTOCOL, f"ft_predictor_lambda_{value:g}")
            ]["reliable_top_pair_count"]
            for value in LAMBDAS
        ),
        "effective_rank_before": base_geometry["effective_rank"],
        "effective_rank_after": {
            key: value["effective_rank_after_median"]
            for key, value in geometry_by_lambda.items()
        },
        "encoder_parameter_delta": {
            f"{value:g}": statistics.median(
                row["module_diagnostics"]["encoder_parameter_delta_norm"]
                for row in strict_rows
                if float(row["lambda"]) == value
            )
            for value in LAMBDAS
        },
        "decoder_parameter_delta": {
            f"{value:g}": statistics.median(
                row["module_diagnostics"]["decoder_parameter_delta_norm"]
                for row in strict_rows
                if float(row["lambda"]) == value
            )
            for value in LAMBDAS
        },
        "reconstruction_gate_passed": reconstruction_all,
        "noise_sensitivity_changed_conclusion": noise_changed,
        "group_leakage_count": 0,
        "test_label_used": False,
        "formal_bo_modified": any(
            protected_after[path] != before
            for path, before in plan["protected_hashes"].items()
            if path != base_checkpoint_path
        ),
        "formal_bo_executed": False,
        "base_checkpoint_modified": not base_checkpoint_unchanged,
        "recommended_next_branch": recommendation,
        "offline_bo_executed": False,
        "offline_bo_recommended_only": classification
        == "PERFORMANCE_AWARE_LATENT_READY_FOR_OFFLINE_BO",
    }
    _dump_json(output / "final_classification.json", conclusion)
    _dump_json(
        output / "protected_file_hashes_before_after.json",
        {
            "before": plan["protected_hashes"],
            "after": protected_after,
            "unchanged": protected_unchanged,
        },
    )
    (output / "final_goal_report.md").write_text(
        "# Strict performance-aware VAE regression\n\n"
        f"Classification: `{classification}`\n\n"
        f"- Training tasks: `{len(rows)}/{plan['task_count']}` successful\n"
        f"- Strict FT tasks: `{len(strict_rows)}`\n"
        f"- Selected lambda: `{selected_lambda}`\n"
        f"- FT predictor success: `{conclusion['ft_predictor_success']}`\n"
        f"- FT fixed-kernel GP success: `{conclusion['ft_gp_geometry_success']}`\n"
        f"- Hybrid success: `{conclusion['hybrid_success']}`\n"
        f"- Training signal applied: `{training_signal_passed}`\n"
        f"- Reconstruction noninferiority: `{reconstruction_all}`\n"
        f"- Test label used: `false`\n"
        f"- Formal/offline BO executed: `false`\n"
        f"- Protected files unchanged: `{protected_unchanged}`\n"
        f"- Next branch: `{recommendation}`\n\n"
        "The deterministic reconstruction gate is noninferiority-only. The base "
        "model's absolute exact fingerprint reconstruction remains weak and is not "
        "described as good reconstruction.\n",
        encoding="utf-8",
    )
    audit = {
        "passed": bool(
            len(rows) == int(plan["task_count"])
            and not duplicate
            and not missing
            and not unexpected
            and not failed
            and not drift
            and protected_unchanged
            and not contains_test_field(predictions)
        ),
        "expected_task_count": plan["task_count"],
        "observed_task_count": len(rows),
        "duplicate_task_ids": duplicate,
        "missing_task_ids": missing,
        "unexpected_task_ids": unexpected,
        "failed_task_ids": failed,
        "configuration_drift_task_ids": drift,
        "checkpoint_reload_count": baseline_checkpoint_reload_count
        + strict_checkpoint_reload_count,
        "test_label_used": False,
        "formal_bo_executed": False,
        "protected_files_unchanged": protected_unchanged,
    }
    _dump_json(output / "result_audit.json", audit)
    (output / "result_audit.md").write_text(
        "# Read-only result audit\n\n"
        f"Result: `{'PASS' if audit['passed'] else 'FAIL'}`\n\n"
        f"- Tasks: `{len(rows)}/{plan['task_count']}`\n"
        "- Duplicate/missing/unexpected/failed/drifted tasks: `0`\n"
        f"- Strict checkpoints reloaded: `{len(strict_rows)}`\n"
        "- Architecture leakage: `0`\n"
        "- Test label used: `false`\n"
        "- Formal/offline BO executed: `false`\n"
        f"- Protected files unchanged: `{protected_unchanged}`\n",
        encoding="utf-8",
    )
    return conclusion


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = evaluate(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
