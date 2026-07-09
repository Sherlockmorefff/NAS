"""Generate non-interactive Phase4 GP prediction and convergence plots."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        warnings.warn(f"missing plot input: {path}", RuntimeWarning, stacklevel=2)
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _numbers(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            warnings.warn(f"missing/non-numeric field {field}; affected rows skipped", RuntimeWarning, stacklevel=2)
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _series(rows: list[dict[str, str]], *fields: str) -> list[tuple[float, ...]]:
    output: list[tuple[float, ...]] = []
    for row in rows:
        try:
            values = tuple(float(row[field]) for field in fields)
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values):
            output.append(values)
    if not output:
        warnings.warn(f"no finite rows for fields {fields}", RuntimeWarning, stacklevel=2)
    return output


def plot_gp_records(output: str | Path) -> list[Path]:
    """Create all requested plots from a Phase4 output directory."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output)
    predictions = _read_csv(output_path / "gp_predictions.csv")
    convergence = _read_csv(output_path / "gp_convergence_history.csv")
    if not predictions:
        warnings.warn("gp_predictions.csv is empty; prediction plots skipped", RuntimeWarning, stacklevel=2)
        return []
    output_path.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def save(fig: Any, name: str) -> None:
        path = output_path / name
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    fig, ax = plt.subplots(figsize=(6, 6))
    for stage, color in (("offline_init", "tab:blue"), ("online_bo", "tab:orange")):
        points = _series([row for row in predictions if row.get("gp_stage") == stage], "val_acc", "gp_pred_mean")
        if points:
            actual, predicted = zip(*points)
            ax.scatter(actual, predicted, alpha=0.7, label=stage, color=color)
    all_values = _numbers(predictions, "val_acc") + _numbers(predictions, "gp_pred_mean")
    if all_values:
        low, high = min(all_values), max(all_values)
        ax.plot([low, high], [low, high], "k--", label="y=x")
    ax.set(xlabel="Actual val_acc", ylabel="GP prediction", title="GP prediction vs actual")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "gp_pred_vs_actual.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for stage, color in (("offline_init", "tab:blue"), ("online_bo", "tab:orange")):
        points = _series([row for row in predictions if row.get("gp_stage") == stage], "step", "gp_abs_error")
        if points:
            x, y = zip(*points)
            ax.plot(x, y, "o-", label=stage, color=color, alpha=0.8)
    ax.set(xlabel="Step", ylabel="Absolute error", title="GP absolute error by step")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "gp_abs_error_by_step.png")

    interval = _series(predictions, "step", "gp_pred_mean", "gp_pred_95_low", "gp_pred_95_high", "val_acc")
    fig, ax = plt.subplots(figsize=(9, 5))
    if interval:
        step, mean, low, high, actual = map(list, zip(*interval))
        ax.fill_between(step, low, high, alpha=0.2, label="95% predictive interval")
        ax.plot(step, mean, "-", label="GP mean")
        ax.scatter(step, actual, s=18, color="black", label="Actual val_acc")
    ax.set(xlabel="Step", ylabel="val_acc", title="GP predictive interval")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "gp_prediction_interval_by_step.png")

    error = _series(predictions, "step", "gp_abs_error", "gp_squared_error")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if error:
        steps, absolute, squared = map(list, zip(*error))
        cumulative_mae = [sum(absolute[: index + 1]) / (index + 1) for index in range(len(absolute))]
        cumulative_rmse = [math.sqrt(sum(squared[: index + 1]) / (index + 1)) for index in range(len(squared))]
        ax.plot(steps, cumulative_mae, label="Cumulative MAE")
        ax.plot(steps, cumulative_rmse, label="Cumulative RMSE")
    ax.set(xlabel="Step", ylabel="Error", title="Cumulative GP error")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "gp_cumulative_error.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if convergence:
        for field, label in (
            ("holdout_mae", "Holdout MAE"),
            ("prequential_window_mae", "Prequential MAE"),
            ("probe_mean_std", "Probe mean std"),
        ):
            points = _series(convergence, "online_samples", field)
            if points:
                x, y = zip(*points)
                ax.plot(x, y, "o-", label=label)
    ax.set(xlabel="Online samples", ylabel="Metric", title="GP convergence metrics")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "gp_convergence_metrics.png")

    timing = _series(predictions, "step", "eval_seconds", "gp_update_seconds")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if timing:
        step, evaluation, update = map(list, zip(*timing))
        ax.plot(step, evaluation, label="GNN evaluation")
        ax.plot(step, update, label="GP update")
    ax.set(xlabel="Step", ylabel="Seconds", title="Evaluation and GP update cost")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "gp_time_cost.png")
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase4 GP prediction records")
    parser.add_argument("--output", required=True, help="Phase4 output directory")
    args = parser.parse_args()
    for path in plot_gp_records(args.output):
        print(path)


if __name__ == "__main__":
    main()
