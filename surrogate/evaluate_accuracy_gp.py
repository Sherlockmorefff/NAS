"""Evaluate a saved accuracy GP on its immutable checkpoint holdout set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surrogate.accuracy_gp import AccuracyGPPredictor
from surrogate.checkpoint_io import atomic_json_dump
from surrogate.metrics import prediction_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an accuracy GP checkpoint holdout")
    parser.add_argument("--gp_checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prediction_output", default="results/gp_predictor/evaluation_predictions.csv")
    parser.add_argument("--metrics_output", default="results/gp_predictor/evaluation_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = AccuracyGPPredictor.load(args.gp_checkpoint, device=args.device)
    if predictor.holdout_X_raw is None or predictor.holdout_Y is None:
        raise ValueError("checkpoint has no fixed holdout set")
    masks = predictor.holdout_condition_masks if predictor.use_conditional_kernel else None
    predictions = predictor.predict_batch(predictor.holdout_X_raw, condition_masks=masks)
    actual = predictor.holdout_Y.tolist()
    metrics = prediction_metrics(actual, [p["mean"] for p in predictions], [p["std"] for p in predictions])
    metrics.update({"train_size": predictor.train_size, "holdout_size": len(actual)})
    target = Path(args.prediction_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        fields = ["sample_index", "val_acc", "gp_pred_mean", "gp_pred_std", "gp_pred_95_low", "gp_pred_95_high"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (true_value, pred) in enumerate(zip(actual, predictions)):
            writer.writerow({
                "sample_index": index, "val_acc": true_value, "gp_pred_mean": pred["mean"],
                "gp_pred_std": pred["std"], "gp_pred_95_low": pred["lower_95"],
                "gp_pred_95_high": pred["upper_95"],
            })
    atomic_json_dump(metrics, args.metrics_output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
