"""Numerical core for Stage 2, with an explicit prediction/evaluation boundary.

``fit_predict`` has no query-label argument and performs no file I/O.  The caller
must persist its label-free prediction bundle before calling
``evaluate_predictions``. Neither function trains GNNs or loads a VAE/dataset.
The final all-A archive fit can pass an empty query list.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Mapping, Sequence

import gpytorch
import numpy as np
import torch

from analyse.fuzzy_metric_replay import ReplayGP, digest, tensor_hash, validate_candidates
from analyse.joint_configuration_metric_gate import hp_feature_vector
from surrogate.fuzzy_metric import JointMetricModel, metric_loss, neighborhood_affinity


FORMAT_VERSION = "fuzzy_metric_offline_core_v1"
LABEL_KEYS = {"validation_mean", "validation_se", "validation_seeds", "training_seeds"}
EVALUATION_TOLERANCES = (0.006, 0.012)
NORMAL_QUANTILES = {"90": 1.6448536269514722, "95": 1.959963984540054}


def _label_arrays(ids: Sequence[str], labels: Mapping[str, Any]):
    if set(labels) != set(ids):
        raise ValueError("label IDs must match exactly the requested partition")
    means, errors, seeds = [], [], []
    for cid in ids:
        row = labels[cid]
        if set(row) != LABEL_KEYS or row["training_seeds"] != [0, 1, 2]:
            raise ValueError("validation label schema or matched training seeds changed")
        values = np.asarray(row["validation_seeds"], dtype=np.float64)
        mean, error = float(row["validation_mean"]), float(row["validation_se"])
        if (values.shape != (3,) or not np.isfinite(values).all()
                or not np.isfinite([mean, error]).all() or error < 0
                or not ((values >= 0) & (values <= 1)).all() or not 0 <= mean <= 1):
            raise ValueError("validation observations must be finite accuracy proportions")
        if not np.isclose(mean, values.mean(), rtol=0, atol=1e-12):
            raise ValueError("validation mean disagrees with its three seed observations")
        if not np.isclose(error, values.std(ddof=1) / math.sqrt(3), rtol=0, atol=1e-12):
            raise ValueError("validation SE disagrees with its three seed observations")
        means.append(mean)
        errors.append(error)
        seeds.append(values)
    return (torch.tensor(means, dtype=torch.double), torch.tensor(errors, dtype=torch.double),
            torch.tensor(np.asarray(seeds), dtype=torch.double))


def _state_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _state_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_cpu(item) for item in value)
    return copy.deepcopy(value)


def _gradient_norm(module: torch.nn.Module, label: str) -> float:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError(f"missing or nonfinite {label} gradients")
    norm = torch.sqrt(sum(gradient.detach().double().square().sum() for gradient in gradients))
    if not torch.isfinite(norm):
        raise RuntimeError(f"nonfinite {label} gradient norm")
    return float(norm)


def _loss_numbers(losses):
    if not all(torch.isfinite(value).all() for value in losses.values()):
        raise RuntimeError("nonfinite representation loss")
    return {key: float(value.detach()) for key, value in losses.items()}


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None,
                "p10": None, "p25": None, "p75": None, "p90": None}
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max()),
            **{f"p{p}": float(np.percentile(values, p)) for p in (10, 25, 75, 90)}}


def fit_predict(
    train_candidates: Sequence[Mapping[str, Any]],
    query_candidates: Sequence[Mapping[str, Any]],
    base_encoding: Mapping[str, Sequence[float]],
    train_labels: Mapping[str, Any],
    setting: Mapping[str, Any],
    model_seed: int,
    common_config: Mapping[str, Any],
    *,
    on_fit_start: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit from scratch using training labels; predict an entirely unlabeled holdout.

    Common options are representation_steps=100, gp_steps=20, learning_rate=.001,
    gp_learning_rate=.05, lambda_e=.001, gp_seed=0. A callback is invoked just
    before mapping/GP optimization starts so the caller can charge actual fits.
    Stable candidate-ID ordering resolves all input and nearest-neighbor ties.
    """
    started = time.monotonic()
    train = sorted(copy.deepcopy(list(train_candidates)), key=lambda c: c["candidate_id"])
    query = sorted(copy.deepcopy(list(query_candidates)), key=lambda c: c["candidate_id"])
    validate_candidates(train + query)
    train_ids, query_ids = [c["candidate_id"] for c in train], [c["candidate_id"] for c in query]
    if len(train_ids) < 5:
        raise ValueError("at least five training references are required for five-neighbor evaluation")
    if len(set(train_ids + query_ids)) != len(train_ids) + len(query_ids):
        raise ValueError("candidate IDs must be unique and train/query disjoint")
    train_architectures = {c["architecture_fingerprint"] for c in train}
    if train_architectures & {c["architecture_fingerprint"] for c in query}:
        raise ValueError("train and query architecture groups overlap")
    method = str(setting["method"])
    if method not in {"N0", "N1", "N2", "regression"}:
        raise ValueError("method must be N0, N1, N2, or regression")
    enabled = method != "regression"
    epsilon = float(setting["epsilon"]) if enabled else None
    lambda_f = float(setting["lambda_f"]) if enabled else 0.0
    lambda_e = float(common_config.get("lambda_e", 0.001))
    representation_steps, gp_steps = int(common_config.get("representation_steps", 100)), int(common_config.get("gp_steps", 20))
    learning_rate, gp_learning_rate = float(common_config.get("learning_rate", 0.001)), float(common_config.get("gp_learning_rate", 0.05))
    if representation_steps <= 0 or gp_steps <= 0:
        raise ValueError("mapping and GP steps must be positive")
    if min(learning_rate, gp_learning_rate) <= 0 or not np.isfinite([learning_rate, gp_learning_rate]).all():
        raise ValueError("learning rates must be finite and positive")
    if enabled and (not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be finite and positive")
    identity = digest({"train": train, "query": query})
    y, se, seed_y = _label_arrays(train_ids, train_labels)
    z = torch.tensor([base_encoding[c["architecture_fingerprint"]] for c in train], dtype=torch.float32)
    hp = torch.tensor([hp_feature_vector(c) for c in train], dtype=torch.float32)
    if z.ndim != 2 or z.shape[1] == 0 or not torch.isfinite(z).all() or not torch.isfinite(hp).all():
        raise ValueError("training base encodings and HP features must be finite matrices")
    z_mean, z_scale = z.mean(0), z.std(0, correction=0).clamp_min(1e-6)
    hp_mean, hp_scale = hp.mean(0), hp.std(0, correction=0).clamp_min(1e-6)
    y_mean, y_scale = y.mean(), y.std(correction=0).clamp_min(1e-8)
    z_input, hp_input = (z - z_mean) / z_scale, (hp - hp_mean) / hp_scale
    standardized_y = ((y - y_mean) / y_scale).float()
    if enabled:
        affinity = neighborhood_affinity(y, mode=method, epsilon=epsilon, k=10)
        paired_difference = seed_y[:, None, :] - seed_y[None, :, :]
        paired_se = paired_difference.std(-1, correction=1) / math.sqrt(3)
        weights = 1.0 / (1.0 + (paired_se / epsilon).square())
    else:
        affinity = torch.zeros(len(train), len(train), dtype=torch.double)
        weights = torch.zeros_like(affinity)
    model = JointMetricModel(z_dim=z.shape[1], seed=int(model_seed))
    initial_state = _state_cpu(model.state_dict())
    initialization_hash = tensor_hash(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    with torch.no_grad():
        initial_e, initial_p = model(z_input, hp_input)
        initial_losses = _loss_numbers(metric_loss(initial_e, initial_p, standardized_y, affinity,
            lambda_f=lambda_f, lambda_e=lambda_e, pair_weights=weights))
    loss_history = []
    if on_fit_start is not None:
        on_fit_start("mapping")
    mapping_started = time.monotonic()
    for step in range(representation_steps):
        optimizer.zero_grad(set_to_none=True)
        embedding, prediction = model(z_input, hp_input)
        losses = metric_loss(embedding, prediction, standardized_y, affinity,
                             lambda_f=lambda_f, lambda_e=lambda_e, pair_weights=weights)
        numbers = _loss_numbers(losses)
        losses["total"].backward()
        numbers.update(step=step + 1, gradient_norm=_gradient_norm(model, "mapping"))
        optimizer.step()
        loss_history.append(numbers)
    mapping_seconds = time.monotonic() - mapping_started
    model.eval()
    with torch.no_grad():
        train_e, train_p = model(z_input, hp_input)
        final_losses = _loss_numbers(metric_loss(train_e, train_p, standardized_y, affinity,
            lambda_f=lambda_f, lambda_e=lambda_e, pair_weights=weights))
    if not torch.isfinite(train_e).all():
        raise RuntimeError("nonfinite fitted training embeddings")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(common_config.get("gp_seed", 0)))
        gp = ReplayGP(train_e.detach().double(), (y - y_mean) / y_scale, se, float(y_scale)).double()
    gp.train()
    gp.likelihood.train()
    gp_optimizer = torch.optim.Adam(gp.parameters(), lr=gp_learning_rate)
    marginal_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    gp_loss_history = []
    if on_fit_start is not None:
        on_fit_start("gp")
    gp_started = time.monotonic()
    with gpytorch.settings.cholesky_jitter(double_value=1e-6):
        for step in range(gp_steps):
            gp_optimizer.zero_grad(set_to_none=True)
            loss = -marginal_likelihood(gp(gp.train_inputs[0]), gp.train_targets)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite GP loss")
            loss.backward()
            gradient_norm = _gradient_norm(gp, "GP")
            gp_optimizer.step()
            gp_loss_history.append({"step": step + 1, "negative_marginal_log_likelihood": float(loss.detach()),
                                    "gradient_norm": gradient_norm})
    gp_seconds = time.monotonic() - gp_started
    gp.eval()
    gp.likelihood.eval()
    pooled_noise = float(se.square().mean())
    learned_noise = float(gp.likelihood.second_noise.detach()) * float(y_scale) ** 2
    if not np.isfinite([pooled_noise, learned_noise]).all() or min(pooled_noise, learned_noise) < 0:
        raise RuntimeError("nonfinite or negative observation noise")
    prediction_started = time.monotonic()
    predictions = []
    if query:
        query_z = torch.tensor([base_encoding[c["architecture_fingerprint"]] for c in query], dtype=torch.float32)
        query_hp = torch.tensor([hp_feature_vector(c) for c in query], dtype=torch.float32)
        if query_z.shape[1:] != z.shape[1:] or not torch.isfinite(query_z).all() or not torch.isfinite(query_hp).all():
            raise ValueError("query base encodings and HP features must match finite training inputs")
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(double_value=1e-6):
            query_e, _ = model((query_z - z_mean) / z_scale, (query_hp - hp_mean) / hp_scale)
            latent = gp(query_e.double())  # Latent posterior: do not apply likelihood twice.
            means = (latent.mean * y_scale + y_mean).numpy()
            variances = (latent.variance.clamp_min(0) * y_scale.square()).numpy()
            distances = torch.cdist(query_e.double(), train_e.double()).numpy()
            query_norms = query_e.norm(dim=1).numpy()
        if not all(np.isfinite(value).all() for value in (means, variances, distances, query_norms)):
            raise RuntimeError("nonfinite query predictions or distances")
        for index, candidate in enumerate(query):
            neighbors = sorted(range(len(train_ids)), key=lambda j: (float(distances[index, j]), train_ids[j]))[:5]
            latent_variance = float(variances[index])
            observation_variance = latent_variance + pooled_noise + learned_noise
            row = {"candidate_id": candidate["candidate_id"],
                   "architecture_fingerprint": candidate["architecture_fingerprint"],
                   "mean": float(means[index]), "latent_variance": latent_variance,
                   "pooled_observation_noise_variance": pooled_noise, "learned_noise_variance": learned_noise,
                   "observation_variance": observation_variance,
                   "neighbor_ids": [train_ids[j] for j in neighbors],
                   "neighbor_distances": [float(distances[index, j]) for j in neighbors],
                   "embedding_norm": float(query_norms[index])}
            for level, quantile in NORMAL_QUANTILES.items():
                for target, variance in (("observation", observation_variance), ("latent", latent_variance)):
                    half_width = quantile * math.sqrt(variance)
                    row[f"{target}_interval{level}"] = [row["mean"] - half_width, row["mean"] + half_width]
            predictions.append(row)
    ranking = [row["candidate_id"] for row in sorted(predictions, key=lambda row: (-row["mean"], row["candidate_id"]))]
    left, right = torch.triu_indices(len(train), len(train), 1)
    positive_pairs = int((affinity[left, right] > 0).sum())
    total_pairs = int(left.numel())
    standardization = {"fit_ids": train_ids, "hp_feature_order": ["log10_lr", "dropout", "log2_hidden_dim", "log10_l2"],
        "z_mean": z_mean.tolist(), "z_scale": z_scale.tolist(), "hp_mean": hp_mean.tolist(), "hp_scale": hp_scale.tolist(),
        "y_mean": float(y_mean), "y_scale": float(y_scale), "input_scale_floor": 1e-6, "target_scale_floor": 1e-8,
        "target_standard_deviation_correction": 0, "input_standard_deviation_correction": 0}
    diagnostics = {"initial_losses": initial_losses, "final_losses": final_losses, "loss_history": loss_history,
        "gp_loss_history": gp_loss_history, "mapping_gradients_finite": True, "gp_gradients_finite": True,
        "initial_embedding_norm": _distribution(initial_e.norm(dim=1).numpy()),
        "train_embedding_norm": _distribution(train_e.norm(dim=1).numpy()),
        "query_embedding_norm": _distribution([row["embedding_norm"] for row in predictions]),
        "positive_pair_count": positive_pairs, "unordered_pair_count": total_pairs,
        "positive_pair_ratio": positive_pairs / total_pairs, "neighborhood_enabled": enabled,
        "noise_weight_rule": "matched_three_seed_difference_SE_training_only" if enabled else "disabled",
        "pair_weight_distribution": _distribution(weights[left, right].numpy()) if enabled else None,
        "pooled_observation_noise_variance": pooled_noise, "learned_noise_variance": learned_noise,
        "mapping_seconds": mapping_seconds, "gp_seconds": gp_seconds,
        "prediction_seconds": time.monotonic() - prediction_started,
        "total_seconds": time.monotonic() - started,
        "mapping_steps": representation_steps, "gp_steps": gp_steps,
        "mapping_fit_count": 1, "gp_fit_count": 1, "new_gnn_trainings": 0}
    bundle = {"format_version": FORMAT_VERSION, "setting": copy.deepcopy(dict(setting)), "model_seed": int(model_seed),
        "train_ids": train_ids, "query_ids": query_ids, "candidate_identity_hash": identity,
        "train_architecture_by_id": {c["candidate_id"]: c["architecture_fingerprint"] for c in train},
        "query_architecture_by_id": {c["candidate_id"]: c["architecture_fingerprint"] for c in query},
        "initialization_hash": initialization_hash, "standardization": standardization,
        "diagnostics": diagnostics, "predictions": predictions, "query_ranking": ranking, "top5_ids": ranking[:5],
        "interval_target": "finite_three_seed_validation_mean_observation",
        "variance_units": "accuracy_proportion_squared", "intervals_clipped_to_unit_range": False,
        "noise_components": {"pooled_training_SE_squared": pooled_noise,
                             "learned_additional_GP_noise_accuracy_squared": learned_noise},
        "query_labels_used": False, "test_labels_used": False,
        "tie_rule": "distance_then_candidate_id; prediction_mean_desc_then_candidate_id",
        "evaluation_tolerances": list(EVALUATION_TOLERANCES), "evaluation_neighbor_rule": "absolute_difference<=tolerance"}
    state = {"format_version": FORMAT_VERSION, "setting": copy.deepcopy(dict(setting)), "model_seed": int(model_seed),
        "common_config": copy.deepcopy(dict(common_config)), "train_ids": train_ids, "query_ids": query_ids,
        "candidate_identity_hash": identity, "initialization_hash": initialization_hash,
        "model_initial_state": initial_state, "model": _state_cpu(model.state_dict()),
        "optimizer": _state_cpu(optimizer.state_dict()), "gp": _state_cpu(gp.state_dict()),
        "gp_optimizer": _state_cpu(gp_optimizer.state_dict()), "standardization": copy.deepcopy(standardization),
        "train_embeddings": train_e.detach().cpu().clone(), "gp_train_inputs": _state_cpu(gp.train_inputs[0]),
        "gp_train_targets": _state_cpu(gp.train_targets), "gp_fixed_noise_normalized": _state_cpu((se / y_scale).square()),
        "torch_rng": torch.get_rng_state().clone()}
    return bundle, state


def summarize_query_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize already-evaluated rows; also usable after joining five outer folds.

    No independence is asserted for queries, folds, or model seeds. Callers must
    join each candidate exactly once per model seed before using this helper.
    """
    rows = list(records)
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("each candidate must have exactly one held-out evaluation per model seed")
    if not rows:
        return {"query_count": 0, "primary_neighbor_median": None, "neighbor_mean_abs_difference_median": None,
                "main_neighborhood_metric": None, "gp_mae": None, "tolerances": {}}
    primary = float(np.median([row["neighbor_mean_abs_difference"] for row in rows]))
    summary = {"query_count": len(rows), "primary_neighbor_median": primary,
        "neighbor_mean_abs_difference_median": primary, "main_neighborhood_metric": primary,
        "gp_mae": float(np.mean([row["gp_absolute_error"] for row in rows])),
        "neighbor_mean_abs_difference_distribution": _distribution([row["neighbor_mean_abs_difference"] for row in rows]),
        "neighbor_abs_difference_distribution": _distribution([value for row in rows for value in row["neighbor_abs_differences"]]),
        "tolerances": {}}
    for level in NORMAL_QUANTILES:
        for target in ("observation", "latent"):
            summary[f"{target}_coverage{level}"] = float(np.mean([row[f"{target}_covered{level}"] for row in rows]))
            summary[f"{target}_width{level}"] = float(np.mean([row[f"{target}_width{level}"] for row in rows]))
        summary[f"coverage{level}"] = summary[f"observation_coverage{level}"]
        summary[f"width{level}"] = summary[f"observation_width{level}"]
    for tolerance in EVALUATION_TOLERANCES:
        key = str(tolerance)
        values = [row["tolerances"][key] for row in rows]
        recalls = [value["retrieval_recall"] for value in values if value["retrieval_recall"] is not None]
        cross_recalls = [value["cross_architecture_retrieval_recall"] for value in values
                         if value["cross_architecture_retrieval_recall"] is not None]
        summary["tolerances"][key] = {
            "wrong_neighbor_ratio": float(np.mean([value["wrong_neighbor_ratio"] for value in values])),
            "reference_has_neighbor_fraction": float(np.mean([value["reference_has_neighbor"] for value in values])),
            "reference_eligible_count_distribution": _distribution([value["reference_eligible_count"] for value in values]),
            "retrieval_precision": float(np.mean([value["retrieval_precision"] for value in values])),
            "retrieval_recall_when_reference_exists": float(np.mean(recalls)) if recalls else None,
            "cross_architecture_eligible_count_distribution": _distribution([value["cross_architecture_eligible_count"] for value in values]),
            "cross_architecture_has_neighbor_fraction": float(np.mean([value["cross_architecture_has_neighbor"] for value in values])),
            "cross_architecture_retrieval_precision": float(np.mean([value["cross_architecture_retrieval_precision"] for value in values])),
            "cross_architecture_retrieval_recall_when_reference_exists": float(np.mean(cross_recalls)) if cross_recalls else None,
        }
    return summary


def evaluate_predictions(bundle: Mapping[str, Any], train_labels: Mapping[str, Any], query_labels: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a previously saved prediction bundle; query labels first enter here.

    Top 10 percent is exactly ceil(0.1*n), accuracy-descending and ID ties. Its
    metrics are descriptive post-hoc summaries, never an input to fit or selection.
    Error means difference > tolerance; equality is therefore an eligible neighbor.
    """
    train_ids, query_ids = list(bundle["train_ids"]), list(bundle["query_ids"])
    train_y, _, _ = _label_arrays(train_ids, train_labels)
    query_y, _, _ = _label_arrays(query_ids, query_labels)
    predictions = list(bundle["predictions"])
    if [row["candidate_id"] for row in predictions] != query_ids:
        raise ValueError("prediction rows must match the frozen query ID order")
    if not query_ids:
        return {"metrics": summarize_query_records([]), "per_query": [], "subgroups": {}, "top5": None,
                "evaluation_only": True, "test_labels_used": False}
    train_mean = {cid: float(train_y[i]) for i, cid in enumerate(train_ids)}
    query_mean = {cid: float(query_y[i]) for i, cid in enumerate(query_ids)}
    per_query = []
    for prediction in predictions:
        cid = prediction["candidate_id"]
        neighbors = list(prediction["neighbor_ids"])
        if len(neighbors) != 5 or len(set(neighbors)) != 5 or not set(neighbors) <= set(train_ids):
            raise ValueError("stored neighbors must be five distinct training reference IDs")
        truth = query_mean[cid]
        differences = [abs(truth - train_mean[neighbor]) for neighbor in neighbors]
        row = {"candidate_id": cid, "architecture_fingerprint": prediction["architecture_fingerprint"],
            "true_validation_mean": truth, "prediction_mean": prediction["mean"],
            "neighbor_ids": neighbors, "neighbor_distances": list(prediction["neighbor_distances"]),
            "neighbor_abs_differences": differences, "neighbor_mean_abs_difference": float(np.mean(differences)),
            "gp_absolute_error": abs(truth - prediction["mean"]), "tolerances": {}}
        for level in NORMAL_QUANTILES:
            for target in ("observation", "latent"):
                lower, upper = prediction[f"{target}_interval{level}"]
                row[f"{target}_covered{level}"] = bool(lower <= truth <= upper)
                row[f"{target}_width{level}"] = float(upper - lower)
        for tolerance in EVALUATION_TOLERANCES:
            eligible = {reference for reference in train_ids if abs(truth - train_mean[reference]) <= tolerance}
            cross = {reference for reference in eligible
                     if bundle["train_architecture_by_id"][reference] != prediction["architecture_fingerprint"]}
            found = len(set(neighbors) & eligible)
            cross_found = len(set(neighbors) & cross)
            row["tolerances"][str(tolerance)] = {
                "reference_eligible_count": len(eligible), "reference_has_neighbor": bool(eligible),
                "retrieved_eligible_count": found, "retrieval_precision": found / 5,
                "retrieval_recall": found / len(eligible) if eligible else None,
                "wrong_neighbor_ratio": 1.0 - found / 5,
                "cross_architecture_eligible_count": len(cross), "cross_architecture_has_neighbor": bool(cross),
                "cross_architecture_retrieved_eligible_count": cross_found,
                "cross_architecture_retrieval_precision": cross_found / 5,
                "cross_architecture_retrieval_recall": cross_found / len(cross) if cross else None}
        per_query.append(row)
    true_ranking = sorted(query_ids, key=lambda cid: (-query_mean[cid], cid))
    top10_ids = true_ranking[:max(1, math.ceil(0.1 * len(query_ids)))]
    for row in per_query:
        row["posthoc_top10_percent"] = row["candidate_id"] in set(top10_ids)
    groups = {"posthoc_top10_percent": {"candidate_ids": top10_ids,
        "metrics": summarize_query_records([row for row in per_query if row["candidate_id"] in set(top10_ids)])}}
    for tolerance in EVALUATION_TOLERANCES:
        key = str(tolerance)
        for label, exists in (("with_eligible_reference", True), ("without_eligible_reference", False)):
            selected = [row for row in per_query if row["tolerances"][key]["reference_has_neighbor"] is exists]
            groups[f"{key}_{label}"] = {"candidate_ids": [row["candidate_id"] for row in selected],
                                         "metrics": summarize_query_records(selected)}
        cross_rows = [row for row in per_query if row["tolerances"][key]["cross_architecture_has_neighbor"]]
        groups[f"{key}_cross_architecture_close_performance"] = {
            "candidate_ids": [row["candidate_id"] for row in cross_rows], "metrics": summarize_query_records(cross_rows)}
    expected_ranking = [row["candidate_id"] for row in sorted(predictions, key=lambda row: (-row["mean"], row["candidate_id"]))]
    if list(bundle["query_ranking"]) != expected_ranking or list(bundle["top5_ids"]) != expected_ranking[:5]:
        raise ValueError("query ranking differs from saved prediction means and candidate-ID ties")
    selected_ids = expected_ranking[:5]
    best_pool = max(query_mean.values())
    best_selected = max(query_mean[cid] for cid in selected_ids)
    top5 = {"selected_ids": selected_ids, "selected_count": len(selected_ids),
        "best_selected_validation_mean": best_selected, "whole_holdout_best_validation_mean": best_pool,
        "whole_holdout_best_id": true_ranking[0], "regret": best_pool - best_selected,
        "reference_pool": "entire_current_holdout", "selection_used_query_labels": False}
    metrics = summarize_query_records(per_query)
    metrics["top5_regret"] = top5["regret"]
    return {"metrics": metrics, "per_query": per_query, "subgroups": groups, "top5": top5,
        "evaluation_only": True, "evaluation_tolerances": list(EVALUATION_TOLERANCES),
        "eligible_boundary": "absolute_difference<=tolerance", "test_labels_used": False,
        "top10_rule": "ceil(0.1*holdout_count), true_validation_mean_desc_then_candidate_id",
        "inference_claim": "development_evaluation_not_independent_search_or_confirmation"}


def aggregate_evaluations(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Join disjoint held-out folds for one method/model seed, then recompute.

    Fold search pools differ, so their top-five regrets remain separate. The
    combined top-10-percent subgroup is assigned only after all held-out labels
    have been evaluated. It is not a filter for training or inner selection.
    """
    evaluations = list(evaluations)
    rows = [copy.deepcopy(row) for evaluation in evaluations for row in evaluation["per_query"]]
    rows.sort(key=lambda row: row["candidate_id"])
    metrics = summarize_query_records(rows)  # Also checks candidate uniqueness.
    fold_regrets = [{"fold_index": index, "query_count": len(evaluation["per_query"]),
                    "top5": copy.deepcopy(evaluation.get("top5"))}
                   for index, evaluation in enumerate(evaluations)]
    if not rows:
        return {"metrics": metrics, "per_query": [], "subgroups": {}, "fold_regrets": fold_regrets,
                "fold_count": len(evaluations), "evaluation_only": True}
    top_count = max(1, math.ceil(0.1 * len(rows)))
    top_ids = [row["candidate_id"] for row in sorted(rows,
        key=lambda row: (-row["true_validation_mean"], row["candidate_id"]))[:top_count]]
    top_set = set(top_ids)
    for row in rows:
        row["posthoc_top10_percent"] = row["candidate_id"] in top_set
    groups = {"posthoc_top10_percent": {"candidate_ids": top_ids,
        "metrics": summarize_query_records([row for row in rows if row["candidate_id"] in top_set])}}
    for tolerance in EVALUATION_TOLERANCES:
        key = str(tolerance)
        for label, exists in (("with_eligible_reference", True), ("without_eligible_reference", False)):
            selected = [row for row in rows if row["tolerances"][key]["reference_has_neighbor"] is exists]
            groups[f"{key}_{label}"] = {"candidate_ids": [row["candidate_id"] for row in selected],
                                        "metrics": summarize_query_records(selected)}
        cross_rows = [row for row in rows if row["tolerances"][key]["cross_architecture_has_neighbor"]]
        groups[f"{key}_cross_architecture_close_performance"] = {
            "candidate_ids": [row["candidate_id"] for row in cross_rows], "metrics": summarize_query_records(cross_rows)}
    return {"metrics": metrics, "per_query": rows, "subgroups": groups, "fold_regrets": fold_regrets,
        "fold_count": len(evaluations), "evaluation_only": True, "test_labels_used": False,
        "top5_note": "fold-specific pools; no fictitious pooled top-five selection",
        "aggregation_rule": "join each candidate once within a model seed, then recompute query metrics",
        "top10_rule": "ceil(0.1*all_heldout_count), true_validation_mean_desc_then_candidate_id"}
