"""CLI for fitting and evaluating an offline Phase4 accuracy GP."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surrogate.accuracy_gp import AccuracyGPPredictor
from surrogate.checkpoint_io import atomic_json_dump
from surrogate.history_dataset import grouped_train_holdout_split, load_history_dataset
from surrogate.metrics import prediction_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline GP val_acc predictor")
    parser.add_argument("--history_paths", nargs="+", required=True)
    parser.add_argument("--hp_mode", required=True, choices=("global4", "hybrid_cond7", "layer_cond19"))
    parser.add_argument("--arch_nz", type=int, default=12)
    parser.add_argument("--z_bound", type=float, default=2.5)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--vae_version", default="")
    parser.add_argument("--eval_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--use_conditional_kernel", action="store_true")
    parser.add_argument("--fit_steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/gp_predictor/accuracy_gp_offline.pt")
    parser.add_argument("--prediction_output", default="results/gp_predictor/offline_holdout_predictions.csv")
    parser.add_argument("--metrics_output", default="results/gp_predictor/offline_metrics.json")
    return parser.parse_args()


def _write_predictions(path: str, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["sample_index", "val_acc", "gp_pred_mean", "gp_pred_std"]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    logger = logging.getLogger("train_accuracy_gp")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    metadata = {
        "dataset": args.dataset,
        "metric": "val_acc",
        "eval_epochs": int(args.eval_epochs),
        "patience": int(args.patience),
        "vae_checkpoint": args.checkpoint,
        "vae_version": args.vae_version or os.path.basename(args.checkpoint),
        "seed": int(args.seed),
    }
    dataset = load_history_dataset(
        args.history_paths,
        hp_mode=args.hp_mode,
        arch_nz=args.arch_nz,
        expected_metadata={
            "dataset": args.dataset,
            "eval_epochs": args.eval_epochs,
            "patience": args.patience,
            "vae_version": args.vae_version or None,
        },
        logger=logger,
    )
    train_idx, holdout_idx = grouped_train_holdout_split(
        dataset, holdout_frac=args.holdout_frac, seed=args.seed
    )
    train_architecture_keys = [dataset.architecture_keys[int(index)] for index in train_idx]
    holdout_architecture_keys = [dataset.architecture_keys[int(index)] for index in holdout_idx]
    metadata.update({
        "train_architecture_keys": sorted(set(train_architecture_keys)),
        "holdout_architecture_keys": sorted(set(holdout_architecture_keys)),
        "holdout_size": int(holdout_idx.size),
        "grouped_split": "architecture_key",
    })
    logger.info("Grouped split: train=%d holdout=%d", train_idx.size, holdout_idx.size)
    predictor = AccuracyGPPredictor.fit_offline(
        dataset.X[train_idx], dataset.y[train_idx],
        arch_nz=args.arch_nz, hp_mode=args.hp_mode, z_bound=args.z_bound,
        use_conditional_kernel=args.use_conditional_kernel,
        condition_masks=dataset.condition_masks[train_idx] if args.use_conditional_kernel else None,
        metadata=metadata,
        holdout_X_raw=dataset.X[holdout_idx], holdout_Y=dataset.y[holdout_idx],
        holdout_condition_masks=dataset.condition_masks[holdout_idx] if args.use_conditional_kernel else None,
        device=args.device, fit_steps=args.fit_steps,
    )
    predictions = predictor.predict_batch(
        dataset.X[holdout_idx],
        condition_masks=dataset.condition_masks[holdout_idx] if args.use_conditional_kernel else None,
    )
    rows: list[dict[str, object]] = []
    for index, true_value, pred in zip(holdout_idx, dataset.y[holdout_idx], predictions):
        rows.append({
            "sample_index": int(index), "val_acc": float(true_value),
            "gp_pred_mean": pred["mean"], "gp_pred_std": pred["std"],
            "gp_pred_95_low": pred["lower_95"], "gp_pred_95_high": pred["upper_95"],
            "gp_covered_by_95": bool(pred["lower_95"] <= true_value <= pred["upper_95"]),
            "architecture_key": dataset.architecture_keys[int(index)],
        })
    metrics = prediction_metrics(
        dataset.y[holdout_idx], [row["gp_pred_mean"] for row in rows],
        [row["gp_pred_std"] for row in rows],
    )
    metrics.update({
        "train_size": int(train_idx.size), "holdout_size": int(holdout_idx.size),
        "hp_mode": args.hp_mode, "filter_counts": dataset.filter_counts,
    })
    predictor.save(args.output)
    _write_predictions(args.prediction_output, rows)
    atomic_json_dump(metrics, args.metrics_output)
    logger.info("Saved GP checkpoint: %s", args.output)
    logger.info("Holdout metrics: %s", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
