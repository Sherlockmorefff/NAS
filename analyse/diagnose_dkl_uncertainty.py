"""Read-only prequential uncertainty diagnostics for Exact GP and small DKL GP.

The replay protocol is identical to compare_exact_vs_dkl_prequential.py:
predict first, reveal the current observation second, then append and optionally
refit.  Diagnostic posteriors never replace the production predict() or qLogEI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyse.compare_exact_vs_dkl_prequential import (
    _fit_predictors,
    _initial_training_rows,
    load_ordered_history,
)
from eval_utils import SEED_DERIVATION, stable_seed
from hp_modes import hp_dim_from_mode, validate_hp_mode
from surrogate.accuracy_gp import GP_NOISE_FLOOR, AccuracyGPPredictor
from surrogate.checkpoint_io import atomic_json_dump
from surrogate.dkl_accuracy_gp import DKLAccuracyGPPredictor, validate_dkl_config


SURROGATE_TYPES = ("exact_gp", "dkl_gp")
DISTANCE_QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)
NEAREST_QUANTILES = (0.01, 0.05, 0.50, 0.95)
PAIR_THRESHOLDS = (1e-8, 1e-6, 1e-4)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_paths", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_init", type=int, default=50)
    parser.add_argument("--max_records", type=int, default=170)
    parser.add_argument("--arch_nz", type=int, default=12)
    parser.add_argument(
        "--hp_mode",
        default="global4",
        choices=("global4", "hybrid_cond7", "layer_cond19"),
    )
    parser.add_argument("--z_bound", type=float, default=2.5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--search_seeds", type=int, nargs="*", default=None)
    parser.add_argument("--use_conditional_kernel", action="store_true")
    parser.add_argument("--gp_refit_every", type=int, default=5)
    parser.add_argument("--gp_init_steps", type=int, default=20)
    parser.add_argument("--gp_refit_steps", type=int, default=20)
    parser.add_argument("--dkl_hidden_dim", type=int, default=32)
    parser.add_argument("--dkl_feature_dim", type=int, choices=(4, 8), default=8)
    parser.add_argument("--dkl_activation", choices=("silu", "relu"), default="silu")
    parser.add_argument("--dkl_lr", type=float, default=0.01)
    parser.add_argument("--dkl_weight_decay", type=float, default=1e-4)
    parser.add_argument("--dkl_grad_clip", type=float, default=5.0)
    parser.add_argument("--dkl_init_steps", type=int, default=200)
    parser.add_argument("--dkl_refit_steps", type=int, default=50)
    parser.add_argument("--dkl_early_stopping_patience", type=int, default=25)
    parser.add_argument("--dkl_min_delta", type=float, default=1e-5)
    parser.add_argument("--window_size", type=int, default=30)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    args.hp_mode = validate_hp_mode(args.hp_mode)
    for name in (
        "n_init",
        "max_records",
        "arch_nz",
        "gp_refit_every",
        "gp_init_steps",
        "gp_refit_steps",
        "window_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if int(args.n_init) > int(args.max_records):
        raise ValueError("--n_init cannot exceed --max_records")
    if not math.isfinite(float(args.z_bound)) or float(args.z_bound) <= 0.0:
        raise ValueError("--z_bound must be positive and finite")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    if args.search_seeds is not None and len(args.search_seeds) not in (
        0,
        len(args.history_paths),
    ):
        raise ValueError("--search_seeds must contain one seed per history path")
    validate_dkl_config(
        hidden_dim=args.dkl_hidden_dim,
        feature_dim=args.dkl_feature_dim,
        activation=args.dkl_activation,
        lr=args.dkl_lr,
        weight_decay=args.dkl_weight_decay,
        grad_clip=args.dkl_grad_clip,
        init_steps=args.dkl_init_steps,
        refit_steps=args.dkl_refit_steps,
        early_stopping_patience=args.dkl_early_stopping_patience,
        min_delta=args.dkl_min_delta,
    )


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tensor_stats(values: torch.Tensor) -> dict[str, float | None]:
    flat = values.detach().double().reshape(-1)
    if flat.numel() == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "min": float(flat.min().cpu()),
        "max": float(flat.max().cpu()),
    }


def _quantiles(values: torch.Tensor, probabilities: Iterable[float]) -> dict[str, float | None]:
    flat = values.detach().double().reshape(-1)
    result: dict[str, float | None] = {}
    for probability in probabilities:
        key = f"q{int(round(float(probability) * 100)):02d}"
        result[key] = (
            None
            if flat.numel() == 0
            else float(torch.quantile(flat, float(probability)).cpu())
        )
    return result


def pairwise_distance_statistics(features: torch.Tensor) -> dict[str, Any]:
    """Return deterministic Euclidean pair/nearest-neighbour diagnostics."""

    matrix = torch.as_tensor(features, dtype=torch.double)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    finite = bool(torch.isfinite(matrix).all())
    count = int(matrix.shape[0])
    result: dict[str, Any] = {
        "feature_has_nonfinite": not finite,
        "unique_feature_ratio": None,
        **{f"pair_distance_{key}": None for key in ("q01", "q05", "q50", "q95", "q99")},
        **{f"nearest_distance_{key}": None for key in ("q01", "q05", "q50", "q95")},
        "pair_count_lt_1e_8": None,
        "pair_count_lt_1e_6": None,
        "pair_count_lt_1e_4": None,
    }
    if not finite:
        return result
    result["unique_feature_ratio"] = (
        None if count == 0 else float(torch.unique(matrix, dim=0).shape[0] / count)
    )
    if count < 2:
        for key in ("pair_count_lt_1e_8", "pair_count_lt_1e_6", "pair_count_lt_1e_4"):
            result[key] = 0
        return result
    pairwise = torch.pdist(matrix, p=2)
    for key, value in _quantiles(pairwise, DISTANCE_QUANTILES).items():
        result[f"pair_distance_{key}"] = value
    distances = torch.cdist(matrix, matrix, p=2)
    distances.fill_diagonal_(math.inf)
    nearest = distances.min(dim=1).values
    for key, value in _quantiles(nearest, NEAREST_QUANTILES).items():
        result[f"nearest_distance_{key}"] = value
    threshold_keys = (
        "pair_count_lt_1e_8",
        "pair_count_lt_1e_6",
        "pair_count_lt_1e_4",
    )
    for threshold, key in zip(PAIR_THRESHOLDS, threshold_keys):
        result[key] = int((pairwise < threshold).sum().cpu())
    return result


def _kernel_state(predictor: AccuracyGPPredictor) -> dict[str, Any]:
    if predictor.model is None:
        raise RuntimeError("GP predictor is not fitted")
    covar_module = predictor.model.covar_module
    kernel = getattr(covar_module, "latent_kernel", covar_module)
    base_kernel = getattr(kernel, "base_kernel", kernel)
    lengthscale = getattr(base_kernel, "lengthscale", None)
    lengthscales = (
        [] if lengthscale is None else lengthscale.detach().double().reshape(-1).cpu().tolist()
    )
    outputscale_value = getattr(kernel, "outputscale", None)
    outputscale = (
        None
        if outputscale_value is None
        else float(outputscale_value.detach().reshape(-1)[0].cpu())
    )
    lengthscale_tensor = torch.as_tensor(lengthscales, dtype=torch.double)
    return {
        "kernel_outputscale": outputscale,
        "ard_lengthscales_json": _json_value(lengthscales),
        "lengthscale_min": (
            None if not lengthscales else float(lengthscale_tensor.min().cpu())
        ),
        "lengthscale_median": (
            None if not lengthscales else float(lengthscale_tensor.median().cpu())
        ),
        "lengthscale_max": (
            None if not lengthscales else float(lengthscale_tensor.max().cpu())
        ),
    }


def _noise_state(predictor: AccuracyGPPredictor) -> dict[str, Any]:
    likelihood = predictor.likelihood
    noise = float(likelihood.noise.detach().reshape(-1)[0].cpu())
    constraint = getattr(likelihood.noise_covar, "raw_noise_constraint", None)
    lower_bound_value = None if constraint is None else getattr(constraint, "lower_bound", None)
    lower_bound = (
        None
        if lower_bound_value is None
        else float(torch.as_tensor(lower_bound_value).detach().max().cpu())
    )
    comparison_floor = GP_NOISE_FLOOR if lower_bound is None else max(lower_bound, GP_NOISE_FLOOR)
    return {
        "likelihood_noise_variance_standardized": noise,
        "likelihood_noise_variance_raw": noise * predictor.y_std**2,
        "likelihood_noise_std_raw": math.sqrt(max(noise, 0.0)) * predictor.y_std,
        "likelihood_noise_constraint_lower_bound": lower_bound,
        "noise_near_constraint_lower_bound": noise <= comparison_floor * 1.05,
    }


def _feature_state(predictor: AccuracyGPPredictor) -> dict[str, Any]:
    common: dict[str, Any] = {
        "has_feature_extractor": isinstance(predictor, DKLAccuracyGPPredictor),
        "feature_dim_mean_json": _json_value([]),
        "feature_dim_std_json": _json_value([]),
        "feature_dim_min_json": _json_value([]),
        "feature_dim_max_json": _json_value([]),
        "feature_std_min": None,
        "feature_std_median": None,
        "feature_std_max": None,
        "feature_norm_min": None,
        "feature_norm_median": None,
        "feature_norm_max": None,
        "mlp_parameter_norms_json": _json_value({}),
        **pairwise_distance_statistics(torch.empty(0, 0, dtype=torch.double)),
    }
    if not isinstance(predictor, DKLAccuracyGPPredictor):
        return common
    if predictor.model is None:
        raise RuntimeError("DKL predictor is not fitted")
    model_input = predictor._model_inputs(
        predictor.train_X_norm,
        predictor.train_condition_masks,
    )
    extractor = predictor.model.covar_module.feature_extractor
    with torch.no_grad():
        features = extractor(model_input).detach().double()
    finite = bool(torch.isfinite(features).all())
    common.update(pairwise_distance_statistics(features))
    common["feature_has_nonfinite"] = not finite
    common["mlp_parameter_norms_json"] = _json_value(
        {
            name: float(parameter.detach().double().norm().cpu())
            for name, parameter in extractor.named_parameters()
        }
    )
    if not finite or features.numel() == 0:
        return common
    dimension_std = features.std(dim=0, unbiased=False)
    feature_norm = features.norm(dim=1)
    common.update(
        {
            "feature_dim_mean_json": _json_value(features.mean(dim=0).cpu().tolist()),
            "feature_dim_std_json": _json_value(dimension_std.cpu().tolist()),
            "feature_dim_min_json": _json_value(features.min(dim=0).values.cpu().tolist()),
            "feature_dim_max_json": _json_value(features.max(dim=0).values.cpu().tolist()),
            "feature_std_min": float(dimension_std.min().cpu()),
            "feature_std_median": float(dimension_std.median().cpu()),
            "feature_std_max": float(dimension_std.max().cpu()),
            "feature_norm_min": float(feature_norm.min().cpu()),
            "feature_norm_median": float(feature_norm.median().cpu()),
            "feature_norm_max": float(feature_norm.max().cpu()),
        }
    )
    return common


def diagnostic_state(
    predictor: AccuracyGPPredictor,
    *,
    search_seed: int,
    history_step: int,
    node: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture model/data diagnostics without modifying state or consuming RNG."""

    if predictor.model is None:
        raise RuntimeError("GP predictor is not fitted")
    surrogate_type = (
        "dkl_gp" if isinstance(predictor, DKLAccuracyGPPredictor) else "exact_gp"
    )
    normalized_y = predictor.train_Y_norm.detach().double().reshape(-1)
    raw_y = normalized_y * predictor.y_std + predictor.y_mean
    raw_stats = _tensor_stats(raw_y)
    normalized_stats = _tensor_stats(normalized_y)
    mean_constant_value = getattr(predictor.model.mean_module, "constant", None)
    training_summary = (
        dict(predictor.training_summary)
        if isinstance(predictor, DKLAccuracyGPPredictor)
        else {}
    )
    hyperparameter_row: dict[str, Any] = {
        "search_seed": int(search_seed),
        "surrogate_type": surrogate_type,
        "node": node,
        "history_step": int(history_step),
        "train_size": int(predictor.train_size),
        "raw_y_mean": raw_stats["mean"],
        "raw_y_std": raw_stats["std"],
        "raw_y_min": raw_stats["min"],
        "raw_y_max": raw_stats["max"],
        "standardized_y_mean": normalized_stats["mean"],
        "standardized_y_std": normalized_stats["std"],
        "standardized_y_min": normalized_stats["min"],
        "standardized_y_max": normalized_stats["max"],
        "fixed_y_mean": float(predictor.y_mean),
        "fixed_y_std": float(predictor.y_std),
        "mean_module_constant_standardized": (
            None
            if mean_constant_value is None
            else float(mean_constant_value.detach().reshape(-1)[0].cpu())
        ),
        **_noise_state(predictor),
        **_kernel_state(predictor),
        "training_requested_steps": training_summary.get("requested_steps"),
        "training_executed_steps": training_summary.get("steps_ran"),
        "training_best_step": training_summary.get("best_step"),
        "training_initial_loss": training_summary.get("initial_loss"),
        "training_best_loss": training_summary.get("best_loss"),
        "training_final_loss": training_summary.get("final_loss"),
        "training_early_stopped": training_summary.get("early_stopped"),
        "training_gradient_norm": training_summary.get("gradient_norm"),
        "training_feature_gradient_norm": training_summary.get("feature_gradient_norm"),
        "training_likelihood_noise_initial": training_summary.get(
            "likelihood_noise_initial"
        ),
        "training_likelihood_noise_final": training_summary.get(
            "likelihood_noise_final"
        ),
        "training_lengthscale_initial_json": _json_value(
            training_summary.get("lengthscale_initial", [])
        ),
        "training_lengthscale_final_json": _json_value(
            training_summary.get("lengthscale_final", [])
        ),
        "training_outputscale_initial": training_summary.get("outputscale_initial"),
        "training_outputscale_final": training_summary.get("outputscale_final"),
    }
    feature_row = {
        "search_seed": int(search_seed),
        "surrogate_type": surrogate_type,
        "node": node,
        "history_step": int(history_step),
        "train_size": int(predictor.train_size),
        **_feature_state(predictor),
    }
    return hyperparameter_row, feature_row


def posterior_diagnostic(
    predictor: AccuracyGPPredictor,
    X_raw: Any,
    condition_mask: Any = None,
) -> dict[str, float]:
    """Compute latent and correctly noised observed posterior moments."""

    if predictor.model is None:
        raise RuntimeError("GP predictor is not fitted")
    X_norm = predictor._normalized_raw(X_raw)
    model_input = predictor._model_inputs(X_norm, condition_mask)
    model = predictor.model
    model_was_training = model.training
    likelihood_was_training = model.likelihood.training
    try:
        model.eval()
        model.likelihood.eval()
        with torch.no_grad():
            latent = model.posterior(model_input, observation_noise=False)
            observed = model.posterior(model_input, observation_noise=True)
            latent_mean = float(latent.mean.reshape(-1)[0].cpu())
            observed_mean = float(observed.mean.reshape(-1)[0].cpu())
            latent_variance = float(latent.variance.reshape(-1)[0].cpu())
            observed_variance = float(observed.variance.reshape(-1)[0].cpu())
    finally:
        model.train(model_was_training)
        model.likelihood.train(likelihood_was_training)
    values = (latent_mean, observed_mean, latent_variance, observed_variance)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite latent or observed posterior diagnostic")
    latent_variance = max(latent_variance, 0.0)
    observed_variance = max(observed_variance, 0.0)
    latent_std = math.sqrt(latent_variance)
    observed_std = math.sqrt(observed_variance)
    noise_contribution = max(observed_variance - latent_variance, 0.0)
    return {
        "latent_mean_standardized": latent_mean,
        "observed_mean_standardized": observed_mean,
        "latent_variance_standardized": latent_variance,
        "latent_std_standardized": latent_std,
        "observed_variance_standardized": observed_variance,
        "observed_std_standardized": observed_std,
        "latent_mean_raw": latent_mean * predictor.y_std + predictor.y_mean,
        "observed_mean_raw": observed_mean * predictor.y_std + predictor.y_mean,
        "latent_variance_raw": latent_variance * predictor.y_std**2,
        "latent_std_raw": latent_std * predictor.y_std,
        "observed_variance_raw": observed_variance * predictor.y_std**2,
        "observed_std_raw": observed_std * predictor.y_std,
        "likelihood_noise_variance_standardized": float(
            predictor.likelihood.noise.detach().reshape(-1)[0].cpu()
        ),
        "likelihood_noise_contribution_variance_standardized": noise_contribution,
        "likelihood_noise_contribution_variance_raw": noise_contribution
        * predictor.y_std**2,
        "likelihood_noise_contribution_std_raw": math.sqrt(noise_contribution)
        * predictor.y_std,
    }


def run_history_diagnostics(
    path_value: str | Path,
    records: list[dict[str, Any]],
    search_seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay one history with the exact same no-lookahead update sequence."""

    initial = _initial_training_rows(records, args.n_init)
    predictors, _ = _fit_predictors(initial, args, search_seed)
    predictions: list[dict[str, Any]] = []
    hyperparameters: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []

    def snapshot(node: str, step: int) -> None:
        for predictor in predictors.values():
            hyperparameter_row, feature_row = diagnostic_state(
                predictor,
                search_seed=search_seed,
                history_step=step,
                node=node,
            )
            hyperparameter_row["history_path"] = str(path_value)
            feature_row["history_path"] = str(path_value)
            hyperparameters.append(hyperparameter_row)
            features.append(feature_row)

    initial_step = int(records[int(args.n_init) - 1]["step"])
    snapshot("initial_fit", initial_step)
    valid_update_count = 0
    for record in records[int(args.n_init) :]:
        z = record["z_search"]
        mask = record["condition_mask_full"] if args.use_conditional_kernel else None
        # Posterior moments are computed before this record's target is read.
        query_diagnostics = {
            name: posterior_diagnostic(predictor, z, mask)
            for name, predictor in predictors.items()
        }
        actual = float(record["val_acc"])
        valid = bool(record["valid"])
        optimize_performed = valid and (
            (valid_update_count + 1) % int(args.gp_refit_every) == 0
        )
        for name, predictor in predictors.items():
            diagnostic = query_diagnostics[name]
            mean_raw = float(diagnostic["latent_mean_raw"])
            observed_mean_raw = float(diagnostic["observed_mean_raw"])
            latent_std_raw = float(diagnostic["latent_std_raw"])
            observed_std_raw = float(diagnostic["observed_std_raw"])
            predictions.append(
                {
                    "history_path": str(path_value),
                    "search_seed": int(search_seed),
                    "surrogate_type": name,
                    "step": int(record["step"]),
                    "train_size_before": int(predictor.train_size),
                    "valid": valid,
                    "actual_raw": actual,
                    "residual_actual_minus_prediction_raw": actual - mean_raw,
                    "latent_covered_95": bool(
                        mean_raw - 1.96 * latent_std_raw
                        <= actual
                        <= mean_raw + 1.96 * latent_std_raw
                    ),
                    "observed_covered_95": bool(
                        observed_mean_raw - 1.96 * observed_std_raw
                        <= actual
                        <= observed_mean_raw + 1.96 * observed_std_raw
                    ),
                    "optimize_performed_after_reveal": optimize_performed,
                    **diagnostic,
                }
            )
        if valid:
            for name, predictor in predictors.items():
                predictor.append_observation(z, actual, condition_mask=mask)
                if name == "dkl_gp":
                    refit_seed = stable_seed(
                        search_seed,
                        "dkl_refit",
                        int(predictor.train_size),
                        int(valid_update_count) + 1,
                    )
                    predictor.refit(
                        optimize=optimize_performed,
                        steps=(
                            int(args.dkl_refit_steps) if optimize_performed else None
                        ),
                        training_seed=refit_seed,
                    )
                else:
                    predictor.refit(
                        optimize=optimize_performed,
                        steps=(int(args.gp_refit_steps) if optimize_performed else None),
                    )
            if optimize_performed:
                snapshot("optimized_refit", int(record["step"]))
            valid_update_count += 1
    snapshot("final", int(records[-1]["step"]))
    return predictions, hyperparameters, features


def uncertainty_summary(
    prediction_rows: list[dict[str, Any]],
    *,
    n_init: int,
    max_records: int,
    window_size: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        key = (str(row["history_path"]), int(row["search_seed"]), str(row["surrogate_type"]))
        groups.setdefault(key, []).append(row)
    for (history_path, seed, surrogate_type), rows in sorted(groups.items()):
        valid_rows = [row for row in rows if row["valid"]]
        windows: list[tuple[str, int | None, int | None, list[dict[str, Any]]]] = [
            ("overall", None, None, valid_rows)
        ]
        window_starts = sorted(
            {
                int(n_init)
                + ((int(row["step"]) - int(n_init)) // int(window_size))
                * int(window_size)
                for row in valid_rows
            }
        )
        for start in window_starts:
            end = min(start + int(window_size) - 1, int(max_records) - 1)
            windows.append(
                (
                    f"steps_{start}_{end}",
                    start,
                    end,
                    [row for row in valid_rows if start <= int(row["step"]) <= end],
                )
            )
        for label, start, end, selected in windows:
            result.append(
                {
                    "history_path": history_path,
                    "search_seed": seed,
                    "surrogate_type": surrogate_type,
                    "window": label,
                    "window_start": start,
                    "window_end": end,
                    "n_predictions": len(selected),
                    "latent_coverage_95": (
                        None
                        if not selected
                        else sum(bool(row["latent_covered_95"]) for row in selected)
                        / len(selected)
                    ),
                    "observed_coverage_95": (
                        None
                        if not selected
                        else sum(bool(row["observed_covered_95"]) for row in selected)
                        / len(selected)
                    ),
                    "mean_latent_std_raw": (
                        None
                        if not selected
                        else sum(float(row["latent_std_raw"]) for row in selected)
                        / len(selected)
                    ),
                    "mean_observed_std_raw": (
                        None
                        if not selected
                        else sum(float(row["observed_std_raw"]) for row in selected)
                        / len(selected)
                    ),
                    "mean_likelihood_noise_std_raw": (
                        None
                        if not selected
                        else sum(
                            float(row["likelihood_noise_contribution_std_raw"])
                            for row in selected
                        )
                        / len(selected)
                    ),
                    "mae_raw": (
                        None
                        if not selected
                        else sum(
                            abs(float(row["residual_actual_minus_prediction_raw"]))
                            for row in selected
                        )
                        / len(selected)
                    ),
                }
            )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    field_units = {
        "standardized variance": "variance after fixed initial-y standardization",
        "raw variance": "standardized_variance * fixed_y_std^2",
        "raw std": "standardized_std * fixed_y_std",
        "likelihood noise": "GaussianLikelihood observation-noise variance in standardized y units",
        "latent posterior": "SingleTaskGP.posterior(..., observation_noise=False)",
        "observed posterior": "SingleTaskGP.posterior(..., observation_noise=True); noise added exactly once",
        "residual": "actual_raw - latent_mean_raw",
        "feature distances": "Euclidean distance in learned feature-extractor output space",
        "training losses": "negative exact marginal log likelihood in standardized y units",
        "gradient norms": "pre-clipping L2 norm from the final executed DKL optimization step",
    }
    config = vars(args).copy()
    config.update(
        {
            "search_dim": int(args.arch_nz) + hp_dim_from_mode(args.hp_mode),
            "surrogate_types": list(SURROGATE_TYPES),
            "seed_derivation": SEED_DERIVATION,
            "protocol": "predict latent+observed -> reveal actual -> append -> optional refit",
            "diagnostics_are_read_only": True,
            "field_units": field_units,
        }
    )
    atomic_json_dump(config, output / "diagnostic_config.json")
    all_predictions: list[dict[str, Any]] = []
    all_hyperparameters: list[dict[str, Any]] = []
    all_features: list[dict[str, Any]] = []
    explicit_seeds = list(args.search_seeds or [])
    for index, history_path in enumerate(args.history_paths):
        records, search_seed = load_ordered_history(
            history_path,
            n_init=args.n_init,
            max_records=args.max_records,
            arch_nz=args.arch_nz,
            hp_mode=args.hp_mode,
            explicit_seed=(explicit_seeds[index] if explicit_seeds else None),
        )
        predictions, hyperparameters, features = run_history_diagnostics(
            history_path,
            records,
            search_seed,
            args,
        )
        all_predictions.extend(predictions)
        all_hyperparameters.extend(hyperparameters)
        all_features.extend(features)
    summary = uncertainty_summary(
        all_predictions,
        n_init=args.n_init,
        max_records=args.max_records,
        window_size=args.window_size,
    )
    _write_csv(output / "hyperparameter_trace.csv", all_hyperparameters)
    _write_csv(output / "feature_trace.csv", all_features)
    _write_csv(output / "uncertainty_predictions.csv", all_predictions)
    _write_csv(output / "uncertainty_summary.csv", summary)


if __name__ == "__main__":
    main()
