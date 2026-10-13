#!/usr/bin/env python3
"""Merge and independently evaluate the frozen performance-aware VAE pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler

from architecture_diagnostics import (
    architecture_fingerprint,
    config_from_igraph,
    config_to_igraph,
    load_joint_vae,
    structure_feature_vector,
)
from eval_utils import _decode_arch
from analyse.performance_aware_vae_pilot import (
    FORMAT_VERSION,
    NOISE_MODES,
    RESULTS_JSONL,
    TRAINING_SEEDS,
    VARIANTS,
    _load_jsonl,
    _predict,
    contains_test_field,
    load_replay_dataset,
    load_task_checkpoint,
    load_validation_labels,
    reliable_pairs,
    set_all_seeds,
    sha256_file,
    stable_hash,
    verify_frozen_plan,
)


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def task_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["outer_fold"]), int(row["training_seed"]), str(row["variant"]),
        float(row["performance_strength"]), str(row["noise_mode"]),
    )


def select_inner_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["outer_fold"]), str(row["variant"]),
                str(row["noise_mode"]), float(row["performance_strength"]),
            )
        ].append(row)
    choices: dict[tuple[int, str, str], float] = {}
    choice_scores: dict[tuple[int, str, str, float], tuple[float, ...]] = {}
    for key, candidates in grouped.items():
        pairwise_values = [
            float(row["inner_metrics"]["reliable_pairwise_accuracy"])
            for row in candidates
            if row["inner_metrics"].get("reliable_pairwise_accuracy") is not None
        ]
        kendall_values = [
            float(row["inner_metrics"]["kendall"])
            for row in candidates if row["inner_metrics"].get("kendall") is not None
        ]
        score = (
            statistics.fmean(pairwise_values) if pairwise_values else -1.0,
            statistics.fmean(kendall_values) if kendall_values else -1.0,
            -statistics.fmean(float(row["inner_metrics"]["rmse"]) for row in candidates),
            -key[3],
        )
        choice_scores[key] = score
    families = {(key[0], key[1], key[2]) for key in grouped}
    for family in families:
        candidates = [key for key in grouped if key[:3] == family]
        choices[family] = max(candidates, key=lambda key: choice_scores[key])[3]
    selected = [
        dict(row) for row in rows
        if float(row["performance_strength"])
        == choices[(int(row["outer_fold"]), str(row["variant"]), str(row["noise_mode"]))]
    ]
    expected = 5 * len(TRAINING_SEEDS) * len(VARIANTS) * len(NOISE_MODES)
    if len(selected) != expected:
        raise RuntimeError(f"inner selection produced {len(selected)} rows, expected {expected}")
    return selected


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]], representation: str, model=None
) -> np.ndarray:
    hp = np.asarray([row["raw_hp_vector"] for row in rows], dtype=np.float64)
    if representation == "Zs":
        return np.asarray([row["z_search"] for row in rows], dtype=np.float64)
    if representation in {"Ze", "H", "new_latent"}:
        if representation == "new_latent":
            if model is None:
                raise ValueError("new_latent requires a model")
            graphs = [config_to_igraph(row) for row in rows]
            with torch.no_grad():
                mu, _ = model.arch_vae.encode(graphs)
            mu_values = mu.detach().cpu().numpy().astype(np.float64)
        else:
            mu_values = np.asarray([row["mu_arch"] for row in rows], dtype=np.float64)
    structures = None
    if representation in {"D", "H"}:
        structures = np.stack(
            [structure_feature_vector(row, [1.0] * 4)[0] for row in rows]
        )
    if representation in {"Ze", "new_latent"}:
        return np.concatenate([mu_values, hp], axis=1)
    if representation == "D":
        return np.concatenate([structures, hp], axis=1)
    if representation == "H":
        return np.concatenate([mu_values, structures, hp], axis=1)
    raise ValueError(f"unsupported representation {representation}")


def _ridge_predictions(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    representation: str,
    *,
    model=None,
) -> tuple[np.ndarray, float]:
    train_x = _feature_matrix(train_rows, representation, model=model)
    test_x = _feature_matrix(test_rows, representation, model=model)
    scaler = StandardScaler().fit(train_x)
    estimator = Ridge(alpha=1.0).fit(
        scaler.transform(train_x),
        np.asarray([row["validation_mean"] for row in train_rows], dtype=np.float64),
    )
    train_predictions = estimator.predict(scaler.transform(train_x))
    residual_std = float(
        np.sqrt(np.mean(
            (train_predictions - np.asarray([row["validation_mean"] for row in train_rows])) ** 2
        ))
    )
    return estimator.predict(scaler.transform(test_x)), max(residual_std, 1e-6)


def build_heldout_predictions(
    plan: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    output: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    selected_lookup = {
        (
            int(row["outer_fold"]), int(row["training_seed"]),
            str(row["variant"]), str(row["noise_mode"]),
        ): row
        for row in selected
    }
    predictions: list[dict[str, Any]] = []
    original_checkpoint = Path(plan["inputs"]["original_checkpoint"])
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        fold_index = int(fold["outer_fold"])
        train_arch = set(fold["outer_train_architectures"])
        test_arch = set(fold["outer_test_architectures"])
        if train_arch & test_arch:
            raise RuntimeError("architecture leakage during heldout prediction")
        train_rows = [row for row in labels if row["architecture_fingerprint"] in train_arch]
        test_rows = [row for row in labels if row["architecture_fingerprint"] in test_arch]
        for representation in ("Zs", "Ze", "D", "H"):
            values, residual_std = _ridge_predictions(train_rows, test_rows, representation)
            for row, value in zip(test_rows, values):
                predictions.append(
                    {
                        "outer_fold": fold_index, "training_seed": None,
                        "model": representation, "variant": "fixed_representation",
                        "noise_mode": None, "config_id": row["config_id"],
                        "architecture_fingerprint": row["architecture_fingerprint"],
                        "validation_mean": row["validation_mean"],
                        "validation_ci95_low": row["validation_ci95_low"],
                        "validation_ci95_high": row["validation_ci95_high"],
                        "prediction": float(value),
                        "prediction_interval_low": float(value - 1.96 * residual_std),
                        "prediction_interval_high": float(value + 1.96 * residual_std),
                    }
                )
        for seed in TRAINING_SEEDS:
            for variant in VARIANTS:
                for noise_mode in NOISE_MODES:
                    selected_row = selected_lookup[(fold_index, seed, variant, noise_mode)]
                    source_checkpoint = Path(selected_row["checkpoint_path"])
                    selected_name = f"fold{fold_index}_seed{seed}_{variant}_{noise_mode}.pth"
                    destination = checkpoint_dir / selected_name
                    shutil.copy2(source_checkpoint, destination)
                    if sha256_file(destination) != selected_row["checkpoint_sha256"]:
                        raise RuntimeError("selected checkpoint copy hash mismatch")
                    model, head, payload = load_task_checkpoint(
                        destination, original_checkpoint, device
                    )
                    inner_errors = np.asarray(
                        [
                            item["prediction"] - item["validation_mean"]
                            for item in selected_row["inner_predictions"]
                        ], dtype=np.float64,
                    )
                    residual_std = max(float(np.sqrt(np.mean(inner_errors ** 2))), 1e-6)
                    own_values = np.asarray(
                        [item["prediction"] for item in selected_row["outer_predictions"]],
                        dtype=np.float64,
                    )
                    if len(own_values) != len(test_rows):
                        raise RuntimeError("selected task outer predictions do not cover fold")
                    own_lookup = {
                        item["config_id"]: float(item["prediction"])
                        for item in selected_row["outer_predictions"]
                    }
                    model_name = f"{variant}_head"
                    for row in test_rows:
                        value = own_lookup[row["config_id"]]
                        predictions.append(
                            {
                                "outer_fold": fold_index, "training_seed": seed,
                                "model": model_name, "variant": variant,
                                "noise_mode": noise_mode, "config_id": row["config_id"],
                                "architecture_fingerprint": row["architecture_fingerprint"],
                                "validation_mean": row["validation_mean"],
                                "validation_ci95_low": row["validation_ci95_low"],
                                "validation_ci95_high": row["validation_ci95_high"],
                                "prediction": value,
                                "prediction_interval_low": value - 1.96 * residual_std,
                                "prediction_interval_high": value + 1.96 * residual_std,
                                "selected_strength": selected_row["performance_strength"],
                                "checkpoint": str(destination.resolve()),
                                "checkpoint_sha256": sha256_file(destination),
                            }
                        )
                    if variant != "original_checkpoint_probe":
                        latent_values, latent_residual = _ridge_predictions(
                            train_rows, test_rows, "new_latent", model=model
                        )
                        latent_name = f"{variant}_latent"
                        for row, value in zip(test_rows, latent_values):
                            predictions.append(
                                {
                                    "outer_fold": fold_index, "training_seed": seed,
                                    "model": latent_name, "variant": variant,
                                    "noise_mode": noise_mode, "config_id": row["config_id"],
                                    "architecture_fingerprint": row["architecture_fingerprint"],
                                    "validation_mean": row["validation_mean"],
                                    "validation_ci95_low": row["validation_ci95_low"],
                                    "validation_ci95_high": row["validation_ci95_high"],
                                    "prediction": float(value),
                                    "prediction_interval_low": float(value - 1.96 * latent_residual),
                                    "prediction_interval_high": float(value + 1.96 * latent_residual),
                                    "selected_strength": selected_row["performance_strength"],
                                    "checkpoint": str(destination.resolve()),
                                    "checkpoint_sha256": sha256_file(destination),
                                }
                            )
    return predictions


def _architecture_aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(row)
    return [
        {
            "architecture_fingerprint": fingerprint,
            "validation_mean": statistics.fmean(float(row["validation_mean"]) for row in values),
            "prediction": statistics.fmean(float(row["prediction"]) for row in values),
        }
        for fingerprint, values in grouped.items()
    ]


def metric_bundle(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    y = np.asarray([row["validation_mean"] for row in rows], dtype=np.float64)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=np.float64)
    grouped_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["architecture_fingerprint"])].append(row)
    architecture_errors = [
        np.asarray(
            [float(row["prediction"]) - float(row["validation_mean"]) for row in values],
            dtype=np.float64,
        )
        for values in grouped_rows.values()
    ]
    architecture_rows = _architecture_aggregate(rows)
    latest_time = {
        fingerprint: max(int(row.get("time_index", 0)) for row in values)
        for fingerprint, values in grouped_rows.items()
    }
    ordered_late = sorted(
        architecture_rows,
        key=lambda row: (latest_time[row["architecture_fingerprint"]], row["architecture_fingerprint"]),
    )[-50:]
    late_tau = kendalltau(
        [row["validation_mean"] for row in ordered_late],
        [row["prediction"] for row in ordered_late],
    ).statistic
    top_count = max(1, math.ceil(0.2 * len(architecture_rows)))
    top_arch = {
        row["architecture_fingerprint"]
        for row in sorted(architecture_rows, key=lambda row: row["validation_mean"], reverse=True)[:top_count]
    }
    top_arch_rows = [row for row in architecture_rows if row["architecture_fingerprint"] in top_arch]
    top_tau = kendalltau(
        [row["validation_mean"] for row in top_arch_rows],
        [row["prediction"] for row in top_arch_rows],
    ).statistic
    top_config_rows = [row for row in rows if row["architecture_fingerprint"] in top_arch]
    all_top_pairs = [
        (left, right, 1 if float(a["validation_mean"]) > float(b["validation_mean"]) else -1)
        for left, a in enumerate(top_config_rows)
        for right, b in enumerate(top_config_rows)
        if left < right
        and a["architecture_fingerprint"] != b["architecture_fingerprint"]
        and float(a["validation_mean"]) != float(b["validation_mean"])
    ]
    reliable_top_pairs = reliable_pairs(top_config_rows)
    def pair_summary(pairs: Sequence[tuple[int, int, int]]) -> tuple[float | None, int, int]:
        if not pairs:
            return None, 0, 0
        by_architecture_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
        for left, right, direction in pairs:
            left_fp = str(top_config_rows[left]["architecture_fingerprint"])
            right_fp = str(top_config_rows[right]["architecture_fingerprint"])
            key = tuple(sorted((left_fp, right_fp)))
            by_architecture_pair[key].append(
                float(
                (float(top_config_rows[left]["prediction"]) - float(top_config_rows[right]["prediction"]))
                * direction > 0
                )
            )
        participating = {fingerprint for pair in by_architecture_pair for fingerprint in pair}
        return (
            statistics.fmean(statistics.fmean(values) for values in by_architecture_pair.values()),
            len(by_architecture_pair),
            len(participating),
        )
    all_pair_accuracy, all_pair_count, _ = pair_summary(all_top_pairs)
    reliable_pair_accuracy, reliable_pair_count, reliable_architecture_count = pair_summary(
        reliable_top_pairs
    )
    architecture_y = np.asarray(
        [row["validation_mean"] for row in architecture_rows], dtype=np.float64
    )
    architecture_prediction = np.asarray(
        [row["prediction"] for row in architecture_rows], dtype=np.float64
    )
    relevance = architecture_y - architecture_y.min() + 1e-9
    ndcg = float(
        ndcg_score(
            relevance.reshape(1, -1), architecture_prediction.reshape(1, -1),
            k=min(10, len(architecture_rows)),
        )
    )
    architecture_coverages = [
        statistics.fmean(
            float(row["prediction_interval_low"] <= row["validation_mean"] <= row["prediction_interval_high"])
            for row in values
            if row.get("prediction_interval_low") is not None
        )
        for values in grouped_rows.values()
        if any(row.get("prediction_interval_low") is not None for row in values)
    ]
    return {
        "observation_count": len(rows),
        "architecture_count": len(architecture_rows),
        "mae": statistics.fmean(float(np.mean(np.abs(errors))) for errors in architecture_errors),
        "rmse": math.sqrt(statistics.fmean(float(np.mean(errors ** 2)) for errors in architecture_errors)),
        "late_kendall": None if not np.isfinite(late_tau) else float(late_tau),
        "top20_kendall": None if not np.isfinite(top_tau) else float(top_tau),
        "top_pairwise_accuracy": all_pair_accuracy,
        "top_architecture_pair_count": all_pair_count,
        "reliable_top_pairwise_accuracy": reliable_pair_accuracy,
        "reliable_top_pair_count": reliable_pair_count,
        "reliable_top_participating_architecture_count": reliable_architecture_count,
        "ndcg_at_10": ndcg,
        "prediction_interval_coverage": statistics.fmean(architecture_coverages) if architecture_coverages else None,
    }


def _aggregate_seed_predictions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)
    output = []
    for values in grouped.values():
        first = dict(values[0])
        for field in ("prediction", "prediction_interval_low", "prediction_interval_high"):
            first[field] = statistics.fmean(float(row[field]) for row in values)
        first["training_seed"] = "mean"
        output.append(first)
    return output


def architecture_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    baseline_rows: Sequence[Mapping[str, Any]] | None = None,
    replicates: int = 2000,
    seed: int = 20260839,
) -> dict[str, Any]:
    lookup = defaultdict(list)
    for row in rows:
        lookup[str(row["architecture_fingerprint"])].append(row)
    baseline_lookup = defaultdict(list)
    if baseline_rows is not None:
        for row in baseline_rows:
            baseline_lookup[str(row["architecture_fingerprint"])].append(row)
        if set(lookup) != set(baseline_lookup):
            raise ValueError("paired architecture bootstrap requires matching groups")
    fingerprints = sorted(lookup)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(fingerprints, size=len(fingerprints), replace=True)
        current_rows = []
        comparison_rows = []
        for occurrence, fingerprint in enumerate(sampled):
            synthetic = f"{fingerprint}#{occurrence}"
            for row in lookup[str(fingerprint)]:
                current_rows.append({**row, "architecture_fingerprint": synthetic})
            if baseline_rows is not None:
                for row in baseline_lookup[str(fingerprint)]:
                    comparison_rows.append({**row, "architecture_fingerprint": synthetic})
        current_value = metric_bundle(current_rows).get(metric)
        if current_value is None:
            continue
        if baseline_rows is not None:
            baseline_value = metric_bundle(comparison_rows).get(metric)
            if baseline_value is None:
                continue
            current_value = float(current_value) - float(baseline_value)
        values.append(float(current_value))
    estimate = metric_bundle(rows).get(metric)
    if baseline_rows is not None and estimate is not None:
        estimate = float(estimate) - float(metric_bundle(baseline_rows)[metric])
    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(values, 0.025)) if values else None,
        "ci95_high": float(np.quantile(values, 0.975)) if values else None,
        "bootstrap_valid_replicates": len(values),
        "bootstrap_unit": "architecture_fingerprint",
    }


def summarize_performance(
    predictions: Sequence[Mapping[str, Any]], *, replicates: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row["model"]), row.get("noise_mode"))].append(row)
    metric_rows = []
    aggregate_rows: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for key, values in sorted(grouped.items(), key=str):
        aggregate = _aggregate_seed_predictions(values) if values[0].get("training_seed") is not None else [dict(row) for row in values]
        aggregate_rows[key] = aggregate
        metrics = metric_bundle(aggregate)
        for metric, estimate in metrics.items():
            if metric.endswith("count"):
                metric_rows.append({"model": key[0], "noise_mode": key[1], "scope": "seed_mean", "metric": metric, "estimate": estimate})
                continue
            ci = architecture_bootstrap_ci(aggregate, metric, replicates=replicates)
            metric_rows.append({"model": key[0], "noise_mode": key[1], "scope": "seed_mean", "metric": metric, **ci})
        seeds = sorted({row.get("training_seed") for row in values if row.get("training_seed") is not None})
        for seed in seeds:
            seed_metrics = metric_bundle([row for row in values if row.get("training_seed") == seed])
            for metric, estimate in seed_metrics.items():
                metric_rows.append({"model": key[0], "noise_mode": key[1], "scope": f"seed_{seed}", "metric": metric, "estimate": estimate})
        for fold_index in range(5):
            fold_values = [row for row in values if int(row["outer_fold"]) == fold_index]
            if not fold_values:
                continue
            if fold_values[0].get("training_seed") is not None:
                fold_values = _aggregate_seed_predictions(fold_values)
            fold_metrics = metric_bundle(fold_values)
            for metric, estimate in fold_metrics.items():
                metric_rows.append(
                    {"model": key[0], "noise_mode": key[1],
                     "scope": f"fold_{fold_index}", "metric": metric, "estimate": estimate}
                )

    baseline = aggregate_rows[("original_checkpoint_probe_head", "bounded_noise_weight")]
    increment_rows = []
    for key, values in aggregate_rows.items():
        if key[1] != "bounded_noise_weight" or key[0] == "original_checkpoint_probe_head":
            continue
        for metric in ("late_kendall", "reliable_top_pairwise_accuracy"):
            increment_rows.append(
                {"model": key[0], "noise_mode": key[1], "metric": metric,
                 **architecture_bootstrap_ci(values, metric, baseline_rows=baseline, replicates=replicates)}
            )
    primary = {"baseline_key": ["original_checkpoint_probe_head", "bounded_noise_weight"], "increments": increment_rows}
    return metric_rows, increment_rows, primary


def _effective_rank(covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    if eigenvalues.sum() <= 0:
        return 0.0
    probabilities = eigenvalues / eigenvalues.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def checkpoint_safety_metrics(
    model,
    replay_rows,
    sample_indices: Sequence[int],
    *,
    decoder_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    graphs = [replay_rows[index][0].copy() for index in sample_indices]
    model.eval()
    with torch.no_grad():
        mu, logvar = model.arch_vae.encode(graphs)
    mu_np = mu.detach().cpu().numpy()
    logvar_np = logvar.detach().cpu().numpy()
    kl_per_dim = np.mean(-0.5 * (1 + logvar_np - mu_np ** 2 - np.exp(logvar_np)), axis=0)
    covariance = np.cov(mu_np, rowvar=False)
    result = {
        "nan_or_inf": bool(not np.isfinite(mu_np).all() or not np.isfinite(logvar_np).all()),
        "kl_active_dimensions": int(np.sum(kl_per_dim > 0.01)),
        "kl_per_dimension": kl_per_dim.tolist(),
        "covariance_effective_rank": _effective_rank(covariance),
        "latent_variance_median": float(np.median(np.var(mu_np, axis=0))),
        "policies": {},
    }
    for policy in ("historical_modal3", "deterministic_greedy"):
        decoded_fingerprints = []
        exact = []
        cycle = []
        valid = 0
        for position, (graph, latent) in enumerate(zip(graphs, mu)):
            true_config = config_from_igraph(model, graph)
            decoded = _decode_arch(
                model, latent, device, n_trials=3,
                decoder_seed=decoder_seed + position,
                decoder_policy=policy,
            )
            if decoded is None:
                continue
            valid += 1
            decoded_fp = architecture_fingerprint(decoded)
            decoded_fingerprints.append(decoded_fp)
            exact.append(float(decoded_fp == architecture_fingerprint(true_config)))
            with torch.no_grad():
                decoded_mu, _ = model.arch_vae.encode([config_to_igraph(decoded)])
            cycle.append(float(torch.linalg.vector_norm(latent - decoded_mu[0]).detach().cpu()))
        result["policies"][policy] = {
            "sample_count": len(graphs), "valid_count": valid,
            "decode_validity": valid / len(graphs),
            "exact_reconstruction": statistics.fmean(exact) if exact else 0.0,
            "unique_architecture_rate": len(set(decoded_fingerprints)) / max(len(decoded_fingerprints), 1),
            "cycle_distance_median": statistics.median(cycle) if cycle else None,
        }
    return result


def safety_audit(
    plan: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    best_variant: str,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = load_replay_dataset(Path(plan["inputs"]["replay"]))
    sample_indices = plan["fixed_reconstruction_sample_indices"]
    original_checkpoint = Path(plan["inputs"]["original_checkpoint"])
    original_model = load_joint_vae(original_checkpoint, "global4", device=device)
    baseline = checkpoint_safety_metrics(
        original_model, replay, sample_indices, decoder_seed=20260841, device=device
    )
    candidate_rows = [
        row for row in selected
        if row["variant"] == best_variant and row["noise_mode"] == "bounded_noise_weight"
    ]
    candidate_metrics = []
    checkpoint_dir = output / "checkpoints"
    for row in candidate_rows:
        path = checkpoint_dir / f"fold{row['outer_fold']}_seed{row['training_seed']}_{row['variant']}_{row['noise_mode']}.pth"
        model, _head, _payload = load_task_checkpoint(path, original_checkpoint, device)
        candidate_metrics.append(
            {
                "outer_fold": row["outer_fold"], "training_seed": row["training_seed"],
                "checkpoint": str(path.resolve()), "checkpoint_sha256": sha256_file(path),
                "metrics": checkpoint_safety_metrics(
                    model, replay, sample_indices,
                    decoder_seed=20260841, device=device,
                ),
            }
        )
    def median_path(path: Sequence[str]) -> float:
        values = []
        for row in candidate_metrics:
            value: Any = row["metrics"]
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        return statistics.median(values)
    thresholds = plan["thresholds"]
    policy_comparisons = {}
    safety_passed = not baseline["nan_or_inf"] and all(not row["metrics"]["nan_or_inf"] for row in candidate_metrics)
    for policy in plan["decoder_policies"]:
        base = baseline["policies"][policy]
        current = {
            key: median_path(["policies", policy, key])
            for key in ("decode_validity", "exact_reconstruction", "unique_architecture_rate", "cycle_distance_median")
        }
        current["decode_validity_drop"] = base["decode_validity"] - current["decode_validity"]
        current["exact_reconstruction_drop"] = base["exact_reconstruction"] - current["exact_reconstruction"]
        current["cycle_distance_ratio"] = (
            current["cycle_distance_median"] / base["cycle_distance_median"]
            if base["cycle_distance_median"] and base["cycle_distance_median"] > 0 else 1.0
        )
        current["passed"] = bool(
            current["decode_validity_drop"] <= thresholds["decode_validity_drop_max"]
            and current["exact_reconstruction_drop"] <= thresholds["exact_reconstruction_drop_max"]
            and current["cycle_distance_ratio"] <= thresholds["cycle_distance_ratio_max"]
        )
        safety_passed = safety_passed and current["passed"]
        policy_comparisons[policy] = {"baseline": base, "candidate_median": current}
    active_candidate = median_path(["kl_active_dimensions"])
    rank_candidate = median_path(["covariance_effective_rank"])
    baseline_active = float(baseline["kl_active_dimensions"])
    active_ratio = 1.0 if baseline_active == 0 else active_candidate / baseline_active
    rank_ratio = rank_candidate / max(float(baseline["covariance_effective_rank"]), 1e-12)
    collapse = {
        "baseline": {
            "kl_active_dimensions": baseline["kl_active_dimensions"],
            "covariance_effective_rank": baseline["covariance_effective_rank"],
            "latent_variance_median": baseline["latent_variance_median"],
        },
        "candidate_median": {
            "kl_active_dimensions": active_candidate,
            "covariance_effective_rank": rank_candidate,
            "latent_variance_median": median_path(["latent_variance_median"]),
        },
        "kl_active_dimension_ratio": active_ratio,
        "effective_rank_ratio": rank_ratio,
        "passed": bool(
            active_ratio >= 1.0 - thresholds["kl_active_dimension_drop_max_fraction"]
            and rank_ratio >= 1.0 - thresholds["effective_rank_drop_max_fraction"]
        ),
    }
    safety_passed = safety_passed and collapse["passed"]
    reconstruction = {
        "fixed_sample_indices": sample_indices,
        "sample_count": len(sample_indices), "best_variant": best_variant,
        "baseline": baseline, "candidate_checkpoints": candidate_metrics,
        "policy_comparisons": policy_comparisons, "passed": safety_passed,
    }
    return reconstruction, collapse


def classify_pilot(
    *,
    preflight_blocked: bool,
    integrity_ok: bool,
    safety_ok: bool,
    improvement_ci_low: float | None,
    reliable_pair_estimate: float | None,
    reliable_pair_ci_low: float | None,
    top20_ci_high: float | None,
    reproducing_seed_count: int,
    positive_fold_count: int,
    reliable_pair_count: int,
    reliable_pair_architecture_count: int,
    minimum_reliable_pair_count: int,
    minimum_reliable_pair_architecture_count: int,
) -> str:
    if preflight_blocked:
        return "BLOCKED_PREFLIGHT"
    if not integrity_ok or not safety_ok:
        return "PILOT_FAILED_SAFETY"
    performance_passed = bool(
        improvement_ci_low is not None and improvement_ci_low > 0
        and reliable_pair_count >= minimum_reliable_pair_count
        and reliable_pair_architecture_count >= minimum_reliable_pair_architecture_count
        and reliable_pair_estimate is not None and reliable_pair_estimate >= 0.60
        and reliable_pair_ci_low is not None and reliable_pair_ci_low > 0.50
        and top20_ci_high is not None and top20_ci_high >= 0
        and reproducing_seed_count >= 2
        and positive_fold_count >= 2
    )
    return "PILOT_SUCCESS" if performance_passed else "PILOT_INCONCLUSIVE"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    plan = json.loads(args.task_plan.read_text(encoding="utf-8"))
    if plan["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported plan format")
    verify_frozen_plan(args.task_plan.resolve(), plan)
    expected_tasks = {task["task_id"]: task for task in plan["tasks"]}
    rows = [row for path in args.shard_results for row in _load_jsonl(path)]
    observed_ids = [row["task_id"] for row in rows]
    duplicate_ids = sorted(task_id for task_id, count in Counter(observed_ids).items() if count > 1)
    missing_ids = sorted(set(expected_tasks) - set(observed_ids))
    unexpected_ids = sorted(set(observed_ids) - set(expected_tasks))
    drift = [
        row["task_id"] for row in rows
        if row["task_id"] in expected_tasks
        and any(row[key] != expected_tasks[row["task_id"]][key] for key in expected_tasks[row["task_id"]])
    ]
    failed = [row["task_id"] for row in rows if row.get("terminal_status") != "success"]
    if duplicate_ids or missing_ids or unexpected_ids or drift or failed:
        raise RuntimeError(
            f"pilot merge failed duplicates={duplicate_ids} missing={missing_ids} "
            f"unexpected={unexpected_ids} drift={drift} failed={failed}"
        )
    rows.sort(key=lambda row: row["task_id"])
    merged_path = output / "merged_training_results.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    merge_audit = {
        "expected_task_count": len(expected_tasks), "observed_task_count": len(rows),
        "duplicate_task_ids": duplicate_ids, "missing_task_ids": missing_ids,
        "unexpected_task_ids": unexpected_ids, "configuration_drift_task_ids": drift,
        "failed_task_ids": failed, "passed": True,
        "shard_result_sha256": {str(path.resolve()): sha256_file(path) for path in args.shard_results},
    }
    _dump_json(output / "merge_audit.json", merge_audit)
    (output / "merge_audit.md").write_text(
        f"# Pilot merge audit\n\nResult: `PASS`\n\n- Tasks: `{len(rows)}`\n- Duplicates: `0`\n- Missing: `0`\n- Drift: `0`\n",
        encoding="utf-8",
    )
    labels = load_validation_labels(Path(plan["inputs"]["labels"]))
    folds = json.loads(Path(plan["folds_file"]).read_text(encoding="utf-8"))
    selected = select_inner_candidates(rows)
    _dump_json(output / "selected_inner_candidates.json", selected)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    predictions = build_heldout_predictions(plan, folds, labels, selected, output, device)
    for row in predictions:
        source = next(item for item in labels if item["config_id"] == row["config_id"])
        row["time_index"] = source["time_index"]
    if contains_test_field(predictions):
        raise RuntimeError("heldout validation predictions contain test fields")
    _dump_csv(output / "heldout_predictions.csv", predictions)
    metric_rows, increment_rows, primary = summarize_performance(
        predictions, replicates=int(plan["bootstrap_replicates"])
    )
    _dump_csv(output / "performance_metrics.csv", metric_rows)
    _dump_json(output / "performance_metrics.json", metric_rows)
    _dump_csv(output / "representation_comparison.csv", increment_rows)
    _dump_json(output / "representation_comparison.json", {"increments": increment_rows, "primary": primary})

    def metric_row(model: str, metric: str, noise="bounded_noise_weight") -> Mapping[str, Any]:
        return next(
            row for row in metric_rows
            if row["model"] == model and row["noise_mode"] == noise
            and row["scope"] == "seed_mean" and row["metric"] == metric
        )
    candidate_models = [
        "encoder_regression_head", "encoder_regression_latent",
        "encoder_regression_ranking_head", "encoder_regression_ranking_latent",
    ]
    increment_lookup = {(row["model"], row["metric"]): row for row in increment_rows}
    observed_best_model = max(
        candidate_models,
        key=lambda model: (
            increment_lookup[(model, "reliable_top_pairwise_accuracy")]["estimate"]
            if increment_lookup[(model, "reliable_top_pairwise_accuracy")]["estimate"] is not None else -999,
            increment_lookup[(model, "late_kendall")]["estimate"]
            if increment_lookup[(model, "late_kendall")]["estimate"] is not None else -999,
        ),
    )
    primary_model = str(plan["primary_candidate_model"])
    if primary_model not in candidate_models:
        raise RuntimeError("frozen primary candidate model is invalid")
    primary_variant = (
        "encoder_regression_ranking" if primary_model.startswith("encoder_regression_ranking")
        else "encoder_regression"
    )
    reconstruction, collapse = safety_audit(plan, selected, primary_variant, output, device)
    _dump_json(output / "reconstruction_audit.json", reconstruction)
    _dump_json(output / "latent_collapse_audit.json", collapse)
    (output / "reconstruction_audit.md").write_text(
        f"# Reconstruction audit\n\nPre-registered primary variant: `{primary_variant}`\n\nResult: `{'PASS' if reconstruction['passed'] else 'FAIL'}`\n",
        encoding="utf-8",
    )
    (output / "latent_collapse_audit.md").write_text(
        f"# Latent collapse audit\n\nResult: `{'PASS' if collapse['passed'] else 'FAIL'}`\n\n"
        f"- KL active dimension ratio: `{collapse['kl_active_dimension_ratio']}`\n"
        f"- Effective rank ratio: `{collapse['effective_rank_ratio']}`\n",
        encoding="utf-8",
    )

    sensitivity = []
    for model in candidate_models:
        for metric in ("late_kendall", "reliable_top_pairwise_accuracy", "rmse"):
            weighted = metric_row(model, metric, "bounded_noise_weight")["estimate"]
            unweighted = metric_row(model, metric, "no_noise_weight")["estimate"]
            sensitivity.append(
                {"model": model, "metric": metric, "bounded_noise_weight": weighted,
                 "no_noise_weight": unweighted,
                 "difference_weighted_minus_unweighted": None if weighted is None or unweighted is None else weighted - unweighted}
            )
    _dump_json(output / "sensitivity_analysis.json", sensitivity)
    (output / "sensitivity_analysis.md").write_text(
        "# Noise-weight sensitivity\n\nAll high-noise configurations were retained. "
        "The JSON report compares bounded noise weights with architecture-balanced, unweighted-noise training.\n",
        encoding="utf-8",
    )

    primary_increment_options = [
        increment_lookup[(primary_model, "late_kendall")],
        increment_lookup[(primary_model, "reliable_top_pairwise_accuracy")],
    ]
    primary_improvement = max(
        primary_increment_options,
        key=lambda row: -999 if row["ci95_low"] is None else row["ci95_low"],
    )
    reliable_metric = metric_row(primary_model, "reliable_top_pairwise_accuracy")
    top_metric = metric_row(primary_model, "top20_kendall")
    reliable_count = int(metric_row(primary_model, "reliable_top_pair_count")["estimate"])
    reliable_architecture_count = int(
        metric_row(primary_model, "reliable_top_participating_architecture_count")["estimate"]
    )
    baseline_model = "original_checkpoint_probe_head"
    seed_reproduction = 0
    for seed in TRAINING_SEEDS:
        current = next(row for row in metric_rows if row["model"] == primary_model and row["noise_mode"] == "bounded_noise_weight" and row["scope"] == f"seed_{seed}" and row["metric"] == primary_improvement["metric"])
        baseline = next(row for row in metric_rows if row["model"] == baseline_model and row["noise_mode"] == "bounded_noise_weight" and row["scope"] == f"seed_{seed}" and row["metric"] == primary_improvement["metric"])
        if current["estimate"] is not None and baseline["estimate"] is not None and current["estimate"] > baseline["estimate"]:
            seed_reproduction += 1
    primary_prediction_rows = [
        row for row in predictions
        if row["model"] == primary_model and row["noise_mode"] == "bounded_noise_weight"
    ]
    baseline_prediction_rows = [
        row for row in predictions
        if row["model"] == baseline_model and row["noise_mode"] == "bounded_noise_weight"
    ]
    positive_folds = 0
    for fold_index in range(5):
        current_rows = _aggregate_seed_predictions(
            [row for row in primary_prediction_rows if row["outer_fold"] == fold_index]
        )
        baseline_rows = _aggregate_seed_predictions(
            [row for row in baseline_prediction_rows if row["outer_fold"] == fold_index]
        )
        current = metric_bundle(current_rows)
        baseline = metric_bundle(baseline_rows)
        metric_name = primary_improvement["metric"]
        if current.get(metric_name) is not None and baseline.get(metric_name) is not None and current[metric_name] > baseline[metric_name]:
            positive_folds += 1
    protected_after = {
        path: sha256_file(path) for path in plan["protected_file_hashes_before"]
    }
    protected_report = {
        "before": plan["protected_file_hashes_before"], "after": protected_after,
        "unchanged": protected_after == plan["protected_file_hashes_before"],
    }
    _dump_json(output / "protected_file_hashes_before_after.json", protected_report)
    classification = classify_pilot(
        preflight_blocked=False,
        integrity_ok=protected_report["unchanged"] and not contains_test_field(predictions),
        safety_ok=bool(reconstruction["passed"] and collapse["passed"]),
        improvement_ci_low=primary_improvement["ci95_low"],
        reliable_pair_estimate=reliable_metric["estimate"],
        reliable_pair_ci_low=reliable_metric.get("ci95_low"),
        top20_ci_high=top_metric.get("ci95_high"),
        reproducing_seed_count=seed_reproduction,
        positive_fold_count=positive_folds,
        reliable_pair_count=reliable_count,
        reliable_pair_architecture_count=reliable_architecture_count,
        minimum_reliable_pair_count=int(plan["minimum_reliable_top_architecture_pairs"]),
        minimum_reliable_pair_architecture_count=int(plan["minimum_reliable_top_participating_architectures"]),
    )
    conclusion = {
        "classification": classification,
        "primary_candidate_model": primary_model,
        "primary_candidate_variant": primary_variant,
        "observed_best_model_descriptive_only": observed_best_model,
        "performance_improvement": primary_improvement,
        "reliable_top_pairwise": reliable_metric, "top20_kendall": top_metric,
        "reproducing_seed_count": seed_reproduction,
        "positive_fold_count": positive_folds,
        "reliable_top_pair_count": reliable_count,
        "reliable_top_participating_architecture_count": reliable_architecture_count,
        "reconstruction_safety_passed": reconstruction["passed"],
        "latent_collapse_safety_passed": collapse["passed"],
        "protected_files_unchanged": protected_report["unchanged"],
        "test_label_used": False, "architecture_leakage_count": 0,
        "formal_bo_modified_or_run": False,
        "offline_bo_validation_allowed": classification == "PILOT_SUCCESS",
        "recommended_next_step": (
            "offline_or_shadow_bo_validation" if classification == "PILOT_SUCCESS"
            else "do_not_advance_to_offline_bo"
        ),
    }
    _dump_json(output / "pilot_conclusion.json", conclusion)
    (output / "final_goal_report.md").write_text(
        "# Minimal performance-aware VAE pilot\n\n"
        f"Conclusion: `{classification}`\n\n"
        f"- Training tasks: `{len(rows)}`\n"
        f"- Pre-registered primary model: `{primary_model}`\n"
        f"- Observed best model (descriptive only): `{observed_best_model}`\n"
        f"- Improvement metric: `{primary_improvement['metric']}`; estimate `{primary_improvement['estimate']}`; "
        f"95% CI `[{primary_improvement['ci95_low']}, {primary_improvement['ci95_high']}]`\n"
        f"- Reliable top pairwise: `{reliable_metric['estimate']}`; 95% CI "
        f"`[{reliable_metric.get('ci95_low')}, {reliable_metric.get('ci95_high')}]`\n"
        f"- Reproducing seeds: `{seed_reproduction}/3`; positive folds: `{positive_folds}/5`\n"
        f"- Reconstruction safety: `{reconstruction['passed']}`\n"
        f"- Latent collapse safety: `{collapse['passed']}`\n"
        f"- Protected files unchanged: `{protected_report['unchanged']}`\n"
        "- Test label used: `false`\n"
        "- Formal BO executed or modified: `false`\n\n"
        f"Next step: `{conclusion['recommended_next_step']}`\n",
        encoding="utf-8",
    )
    environment = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_plan_sha256": sha256_file(args.task_plan),
        "merged_results_sha256": sha256_file(merged_path),
        "device": str(device), "torch": torch.__version__,
    }
    _dump_json(output / "run_environment.json", environment)
    return conclusion


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_plan", type=Path, required=True)
    parser.add_argument("--shard_results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    conclusion = evaluate(parse_args(argv))
    print(json.dumps(conclusion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
