"""Leakage-free prequential screening of exact GP versus the small DKL GP.

This is off-policy surrogate screening on recorded search trajectories.  It is
not a substitute for running and evaluating DKL-BO.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_utils import SEED_DERIVATION, stable_seed
from hp_modes import condition_mask_vector_from_ops, hp_dim_from_mode, validate_hp_mode
from surrogate.accuracy_gp import AccuracyGPPredictor
from surrogate.checkpoint_io import atomic_json_dump
from surrogate.dkl_accuracy_gp import DKLAccuracyGPPredictor, validate_dkl_config
from surrogate.metrics import prediction_metrics, prequential_ranking_metrics


PREDICTION_FIELDS = (
    "history_path", "search_seed", "surrogate_type", "step",
    "train_size_before", "train_size_after", "valid", "actual", "pred_mean",
    "pred_std", "lower_95", "upper_95", "residual", "abs_error", "covered_95",
    "update_performed", "optimize_performed", "refit_seed",
    "initial_fit_seconds", "prediction_seconds", "update_seconds",
)

METRIC_DIRECTIONS = {
    "mae": "lower",
    "rmse": "lower",
    "bias": "absolute_lower",
    "spearman": "higher",
    "predictive_95_coverage": "higher",
    "pairwise_ranking_accuracy": "higher",
    "ndcg_at_10": "higher",
    "top10_recall": "higher",
    "top_region_mae": "lower",
    "best_candidate_regret": "lower",
    "predicted_top10_regret": "lower",
    "tail_60_spearman": "higher",
    "tail_60_top_region_mae": "lower",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_paths", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_init", type=int, default=50)
    parser.add_argument("--max_records", type=int, default=170)
    parser.add_argument("--arch_nz", type=int, default=12)
    parser.add_argument("--hp_mode", default="global4", choices=("global4", "hybrid_cond7", "layer_cond19"))
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
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    args.hp_mode = validate_hp_mode(args.hp_mode)
    for name in ("n_init", "max_records", "arch_nz", "gp_refit_every", "gp_init_steps", "gp_refit_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if int(args.n_init) > int(args.max_records):
        raise ValueError("--n_init cannot exceed --max_records")
    if not math.isfinite(float(args.z_bound)) or float(args.z_bound) <= 0.0:
        raise ValueError("--z_bound must be positive and finite")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    if args.search_seeds is not None and len(args.search_seeds) not in (0, len(args.history_paths)):
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


def _finite_number(value: Any, field: str, path: Path, step: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: step {step} field {field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}: step {step} field {field} must be finite")
    return result


def _condition_mask(record: dict[str, Any], hp_mode: str, arch_nz: int, path: Path) -> list[float]:
    hp_dim = hp_dim_from_mode(hp_mode)
    search_dim = int(arch_nz) + hp_dim
    value = record.get("condition_mask_vector")
    if value is None and isinstance(record.get("condition_mask"), dict):
        value = record["condition_mask"].get("vector", record["condition_mask"].get("mask"))
    if value is None:
        operations = record.get("operations")
        if operations is None and isinstance(record.get("config"), dict):
            operations = record["config"].get("operations")
        if not isinstance(operations, list):
            raise ValueError(
                f"{path}: step {record['step']} lacks both condition mask and operations"
            )
        value = [1.0] * int(arch_nz) + condition_mask_vector_from_ops(
            [str(operation) for operation in operations], hp_mode
        )
    try:
        mask = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: step {record['step']} has a non-numeric condition mask") from exc
    if mask.size == hp_dim:
        mask = np.concatenate((np.ones(int(arch_nz), dtype=np.float64), mask))
    if mask.size != search_dim:
        raise ValueError(
            f"{path}: step {record['step']} condition mask has length {mask.size}, "
            f"expected {hp_dim} or {search_dim}"
        )
    if not np.isfinite(mask).all() or bool(((mask < 0.0) | (mask > 1.0)).any()):
        raise ValueError(f"{path}: step {record['step']} condition mask must be finite in [0, 1]")
    return mask.tolist()


def load_ordered_history(
    path_value: str | Path,
    *,
    n_init: int,
    max_records: int,
    arch_nz: int,
    hp_mode: str,
    explicit_seed: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Strictly load chronological records without aggregating or peeking at labels."""

    path = Path(path_value)
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to read history {path}: {exc}") from exc
    if not isinstance(root, list):
        raise ValueError(f"{path}: history JSON root must be a list")
    search_dim = int(arch_nz) + hp_dim_from_mode(hp_mode)
    records: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    discovered_seeds: set[int] = set()
    previous_step: int | None = None
    for index, raw in enumerate(root):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: record {index} must be an object")
        required = {"step", "valid", "z_search", "val_acc", "hp_mode", "search_dim"}
        missing = sorted(required.difference(raw))
        if missing:
            raise ValueError(f"{path}: record {index} is missing fields {missing}")
        try:
            step = int(raw["step"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: record {index} step must be an integer") from exc
        if step in seen_steps:
            raise ValueError(f"{path}: duplicate step {step}")
        if previous_step is not None and step <= previous_step:
            raise ValueError(f"{path}: steps must be strictly increasing in file order")
        seen_steps.add(step)
        previous_step = step
        if step >= int(max_records):
            continue
        if not isinstance(raw["valid"], bool):
            raise ValueError(f"{path}: step {step} valid must be boolean")
        if raw["hp_mode"] != hp_mode:
            raise ValueError(
                f"{path}: step {step} hp_mode mismatch: expected {hp_mode!r}, got {raw['hp_mode']!r}"
            )
        try:
            record_search_dim = int(raw["search_dim"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: step {step} search_dim must be an integer") from exc
        if record_search_dim != search_dim:
            raise ValueError(
                f"{path}: step {step} search_dim mismatch: expected {search_dim}, got {raw['search_dim']!r}"
            )
        try:
            z = np.asarray(raw["z_search"], dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: step {step} z_search must be numeric") from exc
        if z.size != search_dim or not np.isfinite(z).all():
            raise ValueError(
                f"{path}: step {step} z_search must contain {search_dim} finite values"
            )
        actual = _finite_number(raw["val_acc"], "val_acc", path, step)
        if raw.get("search_seed") is not None:
            discovered_seeds.add(int(raw["search_seed"]))
        record = dict(raw)
        record["step"] = step
        record["z_search"] = z.tolist()
        record["val_acc"] = actual
        record["condition_mask_full"] = _condition_mask(record, hp_mode, arch_nz, path)
        records.append(record)
    if len(records) < int(n_init):
        raise ValueError(f"{path}: requires at least {n_init} records below max_records, got {len(records)}")
    if len(discovered_seeds) > 1:
        raise ValueError(f"{path}: inconsistent search_seed values: {sorted(discovered_seeds)}")
    if explicit_seed is None:
        if not discovered_seeds:
            raise ValueError(f"{path}: search_seed is missing; pass --search_seeds")
        search_seed = next(iter(discovered_seeds))
    else:
        search_seed = int(explicit_seed)
        if discovered_seeds and discovered_seeds != {search_seed}:
            raise ValueError(
                f"{path}: explicit search seed {search_seed} disagrees with history {sorted(discovered_seeds)}"
            )
    return records, search_seed


def _initial_training_rows(records: list[dict[str, Any]], n_init: int) -> list[dict[str, Any]]:
    initial = [record for record in records[: int(n_init)] if record["valid"]]
    if len(initial) < 2:
        raise ValueError("the initial prefix must contain at least two valid GP-training records")
    return initial


def _fit_predictors(
    initial: list[dict[str, Any]],
    args: argparse.Namespace,
    search_seed: int,
) -> tuple[dict[str, AccuracyGPPredictor], dict[str, float]]:
    X = [row["z_search"] for row in initial]
    y = [row["val_acc"] for row in initial]
    masks = [row["condition_mask_full"] for row in initial]
    common = {
        "arch_nz": int(args.arch_nz),
        "hp_mode": args.hp_mode,
        "z_bound": float(args.z_bound),
        "use_conditional_kernel": bool(args.use_conditional_kernel),
        "condition_masks": masks if args.use_conditional_kernel else None,
        "device": args.device,
    }
    started = time.monotonic()
    exact = AccuracyGPPredictor.fit_offline(
        X, y, fit_steps=int(args.gp_init_steps),
        metadata={"surrogate_type": "exact_gp", "seed": int(search_seed)}, **common,
    )
    exact_seconds = time.monotonic() - started
    dkl_seed = stable_seed(search_seed, "dkl_initial_fit")
    started = time.monotonic()
    dkl = DKLAccuracyGPPredictor.fit_offline(
        X,
        y,
        fit_steps=int(args.dkl_init_steps),
        training_seed=dkl_seed,
        hidden_dim=int(args.dkl_hidden_dim),
        feature_dim=int(args.dkl_feature_dim),
        activation=args.dkl_activation,
        lr=float(args.dkl_lr),
        weight_decay=float(args.dkl_weight_decay),
        grad_clip=float(args.dkl_grad_clip),
        init_steps=int(args.dkl_init_steps),
        refit_steps=int(args.dkl_refit_steps),
        early_stopping_patience=int(args.dkl_early_stopping_patience),
        min_delta=float(args.dkl_min_delta),
        metadata={
            "surrogate_type": "dkl_gp",
            "seed": int(search_seed),
            "seed_derivation": SEED_DERIVATION,
            "dkl_training_seed": int(dkl_seed),
        },
        **common,
    )
    dkl_seconds = time.monotonic() - started
    return {"exact_gp": exact, "dkl_gp": dkl}, {
        "exact_gp": exact_seconds, "dkl_gp": dkl_seconds,
    }


def run_history(
    path_value: str | Path,
    records: list[dict[str, Any]],
    search_seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Predict first, reveal the current target second, then update from it."""

    initial = _initial_training_rows(records, args.n_init)
    predictors, fit_seconds = _fit_predictors(initial, args, search_seed)
    valid_update_count = 0
    rows: list[dict[str, Any]] = []
    for record in records[int(args.n_init) :]:
        z = record["z_search"]
        mask = record["condition_mask_full"] if args.use_conditional_kernel else None
        # These predictions happen before this record's val_acc is assigned below.
        predictions: dict[str, dict[str, float]] = {}
        prediction_seconds: dict[str, float] = {}
        for name, predictor in predictors.items():
            started = time.monotonic()
            predictions[name] = predictor.predict(z, condition_mask=mask)
            prediction_seconds[name] = time.monotonic() - started
        actual = float(record["val_acc"])
        valid = bool(record["valid"])
        for name, predictor in predictors.items():
            prediction = predictions[name]
            before = predictor.train_size
            update_performed = False
            optimize_performed = False
            refit_seed: int | None = None
            update_seconds = 0.0
            if valid:
                started = time.monotonic()
                predictor.append_observation(z, actual, condition_mask=mask)
                optimize_performed = (valid_update_count + 1) % int(args.gp_refit_every) == 0
                if name == "dkl_gp":
                    refit_seed = stable_seed(
                        search_seed,
                        "dkl_refit",
                        int(predictor.train_size),
                        int(valid_update_count) + 1,
                    )
                    predictor.refit(
                        optimize=optimize_performed,
                        steps=int(args.dkl_refit_steps) if optimize_performed else None,
                        training_seed=refit_seed,
                    )
                else:
                    predictor.refit(
                        optimize=optimize_performed,
                        steps=int(args.gp_refit_steps) if optimize_performed else None,
                    )
                update_seconds = time.monotonic() - started
                update_performed = True
            residual = actual - float(prediction["mean"])
            rows.append(
                {
                    "history_path": str(path_value),
                    "search_seed": int(search_seed),
                    "surrogate_type": name,
                    "step": int(record["step"]),
                    "train_size_before": int(before),
                    "train_size_after": int(predictor.train_size),
                    "valid": valid,
                    "actual": actual,
                    "pred_mean": float(prediction["mean"]),
                    "pred_std": float(prediction["std"]),
                    "lower_95": float(prediction["lower_95"]),
                    "upper_95": float(prediction["upper_95"]),
                    "residual": residual,
                    "abs_error": abs(residual),
                    "covered_95": bool(prediction["lower_95"] <= actual <= prediction["upper_95"]),
                    "update_performed": update_performed,
                    "optimize_performed": optimize_performed,
                    "refit_seed": refit_seed,
                    "initial_fit_seconds": float(fit_seconds[name]),
                    "prediction_seconds": float(prediction_seconds[name]),
                    "update_seconds": float(update_seconds),
                }
            )
        if valid:
            valid_update_count += 1
    metrics = [
        {"history_path": str(path_value), "search_seed": int(search_seed), "surrogate_type": name,
         **compute_metrics([row for row in rows if row["surrogate_type"] == name])}
        for name in ("exact_gp", "dkl_gp")
    ]
    return rows, metrics


def compute_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    actual = [float(row["actual"]) for row in valid]
    predicted = [float(row["pred_mean"]) for row in valid]
    stds = [float(row["pred_std"]) for row in valid]
    base = prediction_metrics(actual, predicted, stds)
    ranking_records = [
        {"valid": True, "val_acc": y, "gp_pred_mean": p, "gp_pred_std": s}
        for y, p, s in zip(actual, predicted, stds)
    ]
    ranking = prequential_ranking_metrics(ranking_records, max(1, len(ranking_records)))
    result: dict[str, Any] = {
        "n_predictions": len(valid),
        "mae": base["mae"],
        "rmse": base["rmse"],
        "bias": statistics.fmean(p - y for p, y in zip(predicted, actual)) if actual else None,
        "spearman": base["spearman"],
        "predictive_95_coverage": base["predictive_95_coverage"],
        "pairwise_ranking_accuracy": ranking["prequential_pairwise_accuracy"],
        "pairwise_pair_count": ranking["prequential_pairwise_pairs"],
        "ndcg_at_10": ranking["prequential_ndcg_at_10"],
        "top10_recall": ranking["prequential_top10_recall"],
        "top_region_mae": ranking["prequential_top_region_mae"],
        "best_candidate_regret": None,
        "predicted_top10_regret": ranking["prequential_top10_regret"],
        "tail_60_count": min(60, len(valid)),
        "tail_60_spearman": None,
        "tail_60_top_region_mae": None,
    }
    if actual:
        predicted_best_index = max(range(len(predicted)), key=lambda index: (predicted[index], -index))
        result["best_candidate_regret"] = max(actual) - actual[predicted_best_index]
    tail_records = ranking_records[-60:]
    if tail_records:
        tail = prequential_ranking_metrics(tail_records, len(tail_records))
        result["tail_60_spearman"] = tail["prequential_spearman"]
        result["tail_60_top_region_mae"] = tail["prequential_top_region_mae"]
    return result


def _paired_summary(per_seed: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in per_seed:
        seed = int(row["search_seed"])
        surrogate_type = str(row["surrogate_type"])
        seed_rows = by_seed.setdefault(seed, {})
        if surrogate_type in seed_rows:
            raise ValueError(
                f"duplicate per-seed metrics for search seed {seed} and {surrogate_type}"
            )
        seed_rows[surrogate_type] = row
    summary: dict[str, Any] = {
        "analysis_type": "off-policy surrogate screening on recorded search trajectories",
        "warning": "This paired prequential analysis cannot replace a true DKL-BO run.",
        "difference_definition": "dkl_gp - exact_gp, computed within each search seed",
        "seed_count": len(by_seed),
        "metrics": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        differences: list[dict[str, Any]] = []
        wins = ties = losses = 0
        for seed in sorted(by_seed):
            pair = by_seed[seed]
            if set(pair) != {"exact_gp", "dkl_gp"}:
                raise ValueError(f"search seed {seed} does not contain both surrogate results")
            exact = pair["exact_gp"].get(metric)
            dkl = pair["dkl_gp"].get(metric)
            if exact is None or dkl is None:
                continue
            difference = float(dkl) - float(exact)
            differences.append({"search_seed": seed, "exact_gp": exact, "dkl_gp": dkl, "difference": difference})
            tolerance = 1e-12
            score = difference
            if direction == "lower":
                score = -difference
            elif direction == "absolute_lower":
                score = abs(float(exact)) - abs(float(dkl))
            if score > tolerance:
                wins += 1
            elif score < -tolerance:
                losses += 1
            else:
                ties += 1
        values = [row["difference"] for row in differences]
        sign_flip_p = None
        nonzero = [value for value in values if abs(value) > 1e-15]
        if nonzero and len(nonzero) <= 20:
            observed = abs(statistics.fmean(nonzero))
            extreme = 0
            total = 1 << len(nonzero)
            for bits in range(total):
                permuted = [value if bits & (1 << index) else -value for index, value in enumerate(nonzero)]
                extreme += int(abs(statistics.fmean(permuted)) >= observed - 1e-15)
            sign_flip_p = extreme / total
        metric_summary = {
            "direction": direction,
            "valid_seed_count": len(values),
            "mean_difference": statistics.fmean(values) if values else None,
            "sample_sd_difference": statistics.stdev(values) if len(values) >= 2 else None,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "exact_sign_flip_p_two_sided": sign_flip_p,
            "per_seed_differences": differences,
        }
        summary["metrics"][metric] = metric_summary
        csv_rows.append({"metric": metric, **{key: value for key, value in metric_summary.items() if key != "per_seed_differences"}})
    return summary, csv_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields) if fields is not None else list(rows[0]) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "search_dim": int(args.arch_nz) + hp_dim_from_mode(args.hp_mode),
            "surrogate_types": ["exact_gp", "dkl_gp"],
            "seed_derivation": SEED_DERIVATION,
            "analysis_type": "off-policy surrogate screening on recorded search trajectories",
            "no_future_leakage_protocol": "predict -> reveal current val_acc -> append -> optional refit",
        }
    )
    atomic_json_dump(config, output / "config.json")
    all_predictions: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    explicit = list(args.search_seeds or [])
    for index, history_path in enumerate(args.history_paths):
        records, search_seed = load_ordered_history(
            history_path,
            n_init=args.n_init,
            max_records=args.max_records,
            arch_nz=args.arch_nz,
            hp_mode=args.hp_mode,
            explicit_seed=explicit[index] if explicit else None,
        )
        predictions, metrics = run_history(history_path, records, search_seed, args)
        all_predictions.extend(predictions)
        all_metrics.extend(metrics)
    _write_csv(output / "prequential_predictions.csv", all_predictions, PREDICTION_FIELDS)
    _write_csv(output / "per_seed_metrics.csv", all_metrics)
    paired, paired_rows = _paired_summary(all_metrics)
    atomic_json_dump(paired, output / "paired_summary.json")
    _write_csv(output / "paired_summary.csv", paired_rows)


if __name__ == "__main__":
    main()
