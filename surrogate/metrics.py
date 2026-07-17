"""Shared predictive metrics and adaptive Phase4 convergence monitoring."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable


PREQUENTIAL_METRIC_FIELDS = (
    "prequential_window_size",
    "prequential_window_mae",
    "prequential_window_rmse",
    "prequential_window_pearson",
    "prequential_window_coverage_95",
    "prequential_window_mean_std",
    "prequential_spearman",
    "prequential_pairwise_accuracy",
    "prequential_pairwise_pairs",
    "prequential_ndcg_at_10",
    "prequential_top10_recall",
    "prequential_top10_regret",
    "prequential_prediction_bias",
    "prequential_top_region_mae",
)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_pairs(actual: Iterable[float], predicted: Iterable[float]) -> tuple[list[float], list[float], list[int]]:
    actual_values = [float(value) for value in actual]
    predicted_values = [float(value) for value in predicted]
    if len(actual_values) != len(predicted_values):
        raise ValueError(
            f"actual and predicted lengths differ: {len(actual_values)} vs {len(predicted_values)}"
        )
    keep = [index for index, (y, p) in enumerate(zip(actual_values, predicted_values)) if math.isfinite(y) and math.isfinite(p)]
    return [actual_values[index] for index in keep], [predicted_values[index] for index in keep], keep


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    x_centered = [value - x_mean for value in x]
    y_centered = [value - y_mean for value in y]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator <= 0.0:
        return None
    value = sum(a * b for a, b in zip(x_centered, y_centered)) / denominator
    return value if math.isfinite(value) else None


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def prediction_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
    predicted_std: Iterable[float] | None = None,
) -> dict[str, float | int | None]:
    """Compute the canonical offline/online GP metric set."""

    actual_values = list(actual)
    predicted_values = list(predicted)
    y, p, keep = _finite_pairs(actual_values, predicted_values)
    if not y:
        return {
            "mae": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
            "predictive_95_coverage": None,
            "mean_predictive_std": None,
            "median_predictive_std": None,
            "size": 0,
        }
    error = [actual_value - predicted_value for actual_value, predicted_value in zip(y, p)]
    result: dict[str, float | int | None] = {
        "mae": statistics.fmean(abs(value) for value in error),
        "rmse": math.sqrt(statistics.fmean(value * value for value in error)),
        "pearson": _correlation(y, p),
        "spearman": _correlation(_rankdata(y), _rankdata(p)),
        "predictive_95_coverage": None,
        "mean_predictive_std": None,
        "median_predictive_std": None,
        "size": len(y),
    }
    if predicted_std is not None:
        std_all = [float(value) for value in predicted_std]
        if len(std_all) != len(actual_values):
            raise ValueError("predicted_std length must match actual length")
        std = [std_all[index] for index in keep]
        if not all(math.isfinite(value) and value >= 0.0 for value in std):
            raise ValueError("predicted_std must contain finite non-negative values")
        result["predictive_95_coverage"] = statistics.fmean(
            float(predicted_value - 1.96 * std_value <= actual_value <= predicted_value + 1.96 * std_value)
            for actual_value, predicted_value, std_value in zip(y, p, std)
        )
        result["mean_predictive_std"] = statistics.fmean(std)
        result["median_predictive_std"] = statistics.median(std)
    return result


def prediction_record_fields(
    prediction: dict[str, float],
    actual: float,
    train_size_before: int,
    train_size_after: int,
) -> dict[str, Any]:
    """Return deterministic error fields for one pre-update prediction."""

    actual = float(actual)
    mean = float(prediction["mean"])
    std = float(prediction["std"])
    low = float(prediction["lower_95"])
    high = float(prediction["upper_95"])
    values = [actual, mean, std, low, high]
    if not all(math.isfinite(value) for value in values) or std < 0.0:
        raise ValueError("prediction record values must be finite and std must be non-negative")
    residual = actual - mean
    return {
        "gp_pred_mean": mean,
        "gp_pred_std": std,
        "gp_pred_95_low": low,
        "gp_pred_95_high": high,
        "gp_residual": residual,
        "gp_abs_error": abs(residual),
        "gp_squared_error": residual * residual,
        "gp_standardized_residual": residual / max(std, 1e-8),
        "gp_covered_by_95": bool(low <= actual <= high),
        "gp_train_size_before": int(train_size_before),
        "gp_train_size_after": int(train_size_after),
    }


def prequential_ranking_metrics(
    records: Iterable[dict[str, Any]],
    prequential_window: int,
) -> dict[str, float | int | None]:
    """Compute deterministic rolling metrics from genuine pre-update predictions.

    Records without a finite prediction and actual value, or records explicitly
    marked invalid, are excluded before taking the most recent window.  Standard
    deviation is only required for uncertainty-dependent metrics.
    """

    window_size = int(prequential_window)
    if window_size <= 0:
        raise ValueError("prequential_window must be positive")

    valid: list[dict[str, float | None]] = []
    for record in records:
        if not isinstance(record, dict) or record.get("valid") is False:
            continue
        actual = _finite_float(record.get("val_acc"))
        predicted = _finite_float(record.get("gp_pred_mean"))
        if actual is None or predicted is None:
            continue
        predicted_std = _finite_float(record.get("gp_pred_std"))
        if predicted_std is not None and predicted_std < 0.0:
            predicted_std = None
        valid.append(
            {
                "actual": actual,
                "predicted": predicted,
                "predicted_std": predicted_std,
            }
        )

    window = valid[-window_size:]
    y = [float(row["actual"]) for row in window]
    predicted = [float(row["predicted"]) for row in window]
    base = prediction_metrics(y, predicted)
    result: dict[str, float | int | None] = {
        "prequential_window_size": len(window),
        "prequential_window_mae": base["mae"],
        "prequential_window_rmse": base["rmse"],
        "prequential_window_pearson": base["pearson"],
        "prequential_window_coverage_95": None,
        "prequential_window_mean_std": None,
        "prequential_spearman": base["spearman"],
        "prequential_pairwise_accuracy": None,
        "prequential_pairwise_pairs": 0,
        "prequential_ndcg_at_10": None,
        "prequential_top10_recall": None,
        "prequential_top10_regret": None,
        "prequential_prediction_bias": None,
        "prequential_top_region_mae": None,
    }
    if not window:
        return result

    uncertainty_rows = [row for row in window if row["predicted_std"] is not None]
    if uncertainty_rows:
        result["prequential_window_coverage_95"] = statistics.fmean(
            float(
                float(row["predicted"]) - 1.96 * float(row["predicted_std"])
                <= float(row["actual"])
                <= float(row["predicted"]) + 1.96 * float(row["predicted_std"])
            )
            for row in uncertainty_rows
        )
        result["prequential_window_mean_std"] = statistics.fmean(
            float(row["predicted_std"]) for row in uncertainty_rows
        )

    correct_pairs = 0
    pair_count = 0
    for left in range(len(window)):
        for right in range(left + 1, len(window)):
            actual_delta = y[left] - y[right]
            if abs(actual_delta) < 0.002:
                continue
            pair_count += 1
            predicted_delta = predicted[left] - predicted[right]
            correct_pairs += int(actual_delta * predicted_delta > 0.0)
    result["prequential_pairwise_pairs"] = pair_count
    if pair_count:
        result["prequential_pairwise_accuracy"] = correct_pairs / pair_count

    count = len(window)
    k = min(10, count)
    predicted_order = sorted(range(count), key=lambda index: (-predicted[index], index))
    actual_order = sorted(range(count), key=lambda index: (-y[index], index))
    relevance = [value - min(y) for value in y]
    ideal_dcg = sum(
        relevance[index] / math.log2(rank + 2.0)
        for rank, index in enumerate(actual_order[:k])
    )
    if ideal_dcg > 0.0:
        predicted_dcg = sum(
            relevance[index] / math.log2(rank + 2.0)
            for rank, index in enumerate(predicted_order[:k])
        )
        result["prequential_ndcg_at_10"] = predicted_dcg / ideal_dcg

    predicted_top = set(predicted_order[:k])
    actual_top = set(actual_order[:k])
    result["prequential_top10_recall"] = len(predicted_top & actual_top) / k
    predicted_top_best = max(y[index] for index in predicted_top)
    result["prequential_top10_regret"] = max(0.0, max(y) - predicted_top_best)
    result["prequential_prediction_bias"] = statistics.fmean(
        predicted_value - actual_value
        for predicted_value, actual_value in zip(predicted, y)
    )
    top_region_count = max(1, math.ceil(0.2 * count))
    result["prequential_top_region_mae"] = statistics.fmean(
        abs(predicted[index] - y[index]) for index in actual_order[:top_region_count]
    )
    return result


@dataclass
class ConvergenceDecision:
    converged: bool
    stop_reason: str | None
    deferred_reason: str | None = None


class GPConvergenceMonitor:
    """Track GP stability and hard stopping conditions for adaptive BO."""

    def __init__(
        self,
        *,
        min_bo_samples: int = 20,
        max_bo_samples: int = 60,
        convergence_check_every: int = 5,
        convergence_patience: int = 3,
        prequential_window: int = 10,
        mae_relative_tol: float = 0.01,
        mae_absolute_tol: float = 0.002,
        std_relative_tol: float = 0.02,
        spearman_tol: float = 0.01,
        degradation_tolerance: float = 0.01,
        best_acc_patience: int = 20,
        best_acc_min_delta: float = 0.001,
        max_wall_time_hours: float = 4.0,
    ) -> None:
        positive = {
            "min_bo_samples": min_bo_samples,
            "max_bo_samples": max_bo_samples,
            "convergence_check_every": convergence_check_every,
            "convergence_patience": convergence_patience,
            "prequential_window": prequential_window,
            "best_acc_patience": best_acc_patience,
        }
        if any(int(value) <= 0 for value in positive.values()):
            raise ValueError(f"monitor integer settings must be positive: {positive}")
        if int(min_bo_samples) > int(max_bo_samples):
            raise ValueError("min_bo_samples cannot exceed max_bo_samples")
        self.min_bo_samples = int(min_bo_samples)
        self.max_bo_samples = int(max_bo_samples)
        self.check_every = int(convergence_check_every)
        self.patience = int(convergence_patience)
        self.prequential_window = int(prequential_window)
        self.mae_relative_tol = float(mae_relative_tol)
        self.mae_absolute_tol = float(mae_absolute_tol)
        self.std_relative_tol = float(std_relative_tol)
        self.spearman_tol = float(spearman_tol)
        self.degradation_tolerance = float(degradation_tolerance)
        self.best_acc_patience = int(best_acc_patience)
        self.best_acc_min_delta = float(best_acc_min_delta)
        self.max_wall_seconds = float(max_wall_time_hours) * 3600.0
        self.history: list[dict[str, Any]] = []
        self.prequential: list[dict[str, Any]] = []
        self.stable_checks = 0
        self.valid_convergence_checks = 0
        self.best_actual = -math.inf
        self.no_best_improvement_steps = 0
        self.stagnation_deferred_events: list[dict[str, Any]] = []

    def observe_bo_result(
        self, actual: float, prediction: dict[str, Any]
    ) -> dict[str, float | int | None]:
        actual = float(actual)
        if prediction.get("valid") is not False:
            mean = _finite_float(prediction.get("gp_pred_mean"))
            if math.isfinite(actual) and mean is not None:
                self.prequential.append(
                    {
                        "valid": True,
                        "val_acc": actual,
                        "gp_pred_mean": mean,
                        "gp_pred_std": prediction.get("gp_pred_std"),
                    }
                )
        if actual >= self.best_actual + self.best_acc_min_delta:
            self.best_actual = actual
            self.no_best_improvement_steps = 0
        else:
            self.best_actual = max(self.best_actual, actual)
            self.no_best_improvement_steps += 1
        return self.current_prequential_metrics()

    def current_prequential_metrics(self) -> dict[str, float | int | None]:
        return prequential_ranking_metrics(self.prequential, self.prequential_window)

    def should_check(self, online_samples: int) -> bool:
        return int(online_samples) >= self.min_bo_samples and int(online_samples) % self.check_every == 0

    @staticmethod
    def _relative_decrease(previous: float | None, current: float | None) -> float | None:
        if previous is None or current is None or not math.isfinite(previous) or not math.isfinite(current):
            return None
        return (previous - current) / max(abs(previous), 1e-12)

    def add_check(
        self,
        *,
        online_samples: int,
        gp_train_size: int,
        holdout_metrics: dict[str, Any] | None,
        probe_stds: Iterable[float],
        elapsed_seconds: float,
        eval_seconds: Iterable[float],
        gp_update_seconds: Iterable[float],
    ) -> dict[str, Any]:
        probe = [float(value) for value in probe_stds]
        if not probe or not all(math.isfinite(value) for value in probe):
            raise ValueError("probe_stds must be a non-empty finite sequence")
        preq = self.current_prequential_metrics()
        holdout = holdout_metrics or {}
        current_mae = _finite_float(holdout.get("mae"))
        holdout_spearman = _finite_float(holdout.get("spearman"))
        current_preq = _finite_float(preq.get("prequential_window_mae"))
        current_preq_spearman = _finite_float(preq.get("prequential_spearman"))
        has_holdout_error = current_mae is not None
        evidence_source = "holdout" if holdout_metrics is not None else "prequential"
        prequential_window_ready = (
            int(preq["prequential_window_size"]) >= self.prequential_window
        )
        evidence_ready = has_holdout_error if evidence_source == "holdout" else (
            prequential_window_ready
            and current_preq is not None
            and current_preq_spearman is not None
        )
        current_error = current_mae if evidence_source == "holdout" else current_preq
        current_spearman = (
            holdout_spearman if evidence_source == "holdout" else current_preq_spearman
        )
        previous = self.history[-1] if self.history else None
        previous_error = None
        previous_spearman = None
        if previous is not None and previous.get("convergence_evidence_source") == evidence_source:
            previous_error = previous.get(
                "holdout_mae" if evidence_source == "holdout" else "prequential_window_mae"
            )
            previous_spearman = previous.get(
                "holdout_spearman" if evidence_source == "holdout" else "prequential_spearman"
            )
        relative_mae = self._relative_decrease(previous_error, current_error)
        probe_mean = statistics.fmean(probe)
        relative_std = self._relative_decrease(None if previous is None else previous.get("probe_mean_std"), probe_mean)
        spearman_improvement = None
        if current_spearman is not None and previous_spearman is not None:
            spearman_improvement = float(current_spearman) - float(previous_spearman)
        preq_change = None
        if previous is not None and current_preq is not None and previous.get("prequential_window_mae") is not None:
            preq_change = abs(float(current_preq) - float(previous["prequential_window_mae"]))

        comparisons = [
            relative_mae is None or (-self.degradation_tolerance <= relative_mae < self.mae_relative_tol),
            relative_std is None or (-self.degradation_tolerance <= relative_std < self.std_relative_tol),
            spearman_improvement is None or (-self.degradation_tolerance <= spearman_improvement < self.spearman_tol),
            preq_change is None or preq_change < self.mae_absolute_tol,
        ]
        past_holdout = [row["holdout_mae"] for row in self.history if row.get("holdout_mae") is not None]
        past_preq = [row["prequential_window_mae"] for row in self.history if row.get("prequential_window_mae") is not None]
        not_degraded = True
        if current_mae is not None and past_holdout:
            not_degraded &= float(current_mae) <= min(past_holdout) + self.degradation_tolerance
        if current_preq is not None and past_preq:
            not_degraded &= float(current_preq) <= min(past_preq) + self.degradation_tolerance
        if evidence_source == "holdout":
            stable = previous is not None and has_holdout_error and all(comparisons) and not_degraded
        else:
            scratch_comparisons_available = all(
                value is not None
                for value in (relative_mae, relative_std, spearman_improvement, preq_change)
            )
            stable = (
                previous is not None
                and previous.get("convergence_evidence_ready") is True
                and evidence_ready
                and scratch_comparisons_available
                and all(comparisons)
                and not_degraded
            )
        if evidence_ready:
            self.valid_convergence_checks += 1
        self.stable_checks = self.stable_checks + 1 if stable else 0
        converged = int(online_samples) >= self.min_bo_samples and self.stable_checks >= self.patience

        not_ready_reason = None
        if evidence_source == "holdout" and not has_holdout_error:
            not_ready_reason = "holdout_error_unavailable"
        elif evidence_source == "prequential" and not prequential_window_ready:
            not_ready_reason = (
                "prequential_window_incomplete:"
                f"{preq['prequential_window_size']}/{self.prequential_window}"
            )
        elif evidence_source == "prequential" and current_preq_spearman is None:
            not_ready_reason = "prequential_spearman_unavailable"

        eval_recent = list(eval_seconds)[-5:]
        update_recent = list(gp_update_seconds)[-5:]
        median_eval = float(statistics.median(eval_recent)) if eval_recent else 0.0
        median_update = float(statistics.median(update_recent)) if update_recent else 0.0
        row = {
            "check_index": len(self.history),
            "online_samples": int(online_samples),
            "gp_train_size": int(gp_train_size),
            "holdout_mae": current_mae,
            "holdout_rmse": holdout.get("rmse"),
            "holdout_pearson": holdout.get("pearson"),
            "holdout_spearman": holdout_spearman,
            "holdout_coverage_95": holdout.get("predictive_95_coverage"),
            **preq,
            "probe_mean_std": probe_mean,
            "probe_median_std": statistics.median(probe),
            "probe_p90_std": _percentile(probe, 90.0),
            "relative_mae_improvement": relative_mae,
            "relative_std_reduction": relative_std,
            "spearman_improvement": spearman_improvement,
            "prequential_mae_change": preq_change,
            "stable_this_check": bool(stable),
            "stable_checks": int(self.stable_checks),
            "valid_convergence_check": bool(evidence_ready),
            "valid_convergence_checks": int(self.valid_convergence_checks),
            "convergence_evidence_source": evidence_source,
            "convergence_evidence_ready": bool(evidence_ready),
            "convergence_not_ready_reason": not_ready_reason,
            "converged": bool(converged),
            "best_actual_val_acc": None if self.best_actual == -math.inf else float(self.best_actual),
            "no_best_improvement_steps": int(self.no_best_improvement_steps),
            "median_eval_seconds": median_eval,
            "median_gp_update_seconds": median_update,
            "estimated_next_step_seconds": median_eval + median_update,
            "elapsed_seconds": float(elapsed_seconds),
        }
        self.history.append(row)
        return row

    def stop_decision(self, online_samples: int, elapsed_seconds: float, estimated_next_seconds: float) -> ConvergenceDecision:
        if self.history and self.history[-1].get("converged"):
            return ConvergenceDecision(True, "gp_prediction_converged")
        stagnated = (
            int(online_samples) >= self.min_bo_samples
            and self.no_best_improvement_steps >= self.best_acc_patience
        )
        stagnation_gate_ready = self.valid_convergence_checks >= self.patience
        if stagnated and stagnation_gate_ready:
            return ConvergenceDecision(False, "bo_best_acc_stagnated")
        deferred_reason = None
        if stagnated:
            deferred_reason = (
                "bo_best_acc_stagnation_deferred:"
                f"valid_convergence_checks={self.valid_convergence_checks},"
                f"required={self.patience}"
            )
            event = {
                "online_samples": int(online_samples),
                "valid_convergence_checks": int(self.valid_convergence_checks),
                "required_convergence_checks": int(self.patience),
                "reason": deferred_reason,
            }
            if not self.stagnation_deferred_events or self.stagnation_deferred_events[-1] != event:
                self.stagnation_deferred_events.append(event)
        if int(online_samples) >= self.max_bo_samples:
            return ConvergenceDecision(False, "sample_budget_reached", deferred_reason)
        if int(online_samples) >= self.min_bo_samples and float(elapsed_seconds) + float(estimated_next_seconds) > self.max_wall_seconds:
            return ConvergenceDecision(False, "time_budget_reached", deferred_reason)
        return ConvergenceDecision(False, None, deferred_reason)
