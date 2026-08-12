#!/usr/bin/env python3
"""Compare initialization/search runs without counterfactual label leakage.

Each labelled input is analyzed only with labels contained in its own full-
fidelity history. Geometry is label-free. Cross-run labels are never joined by
candidate identity, because that would only be valid for a fully evaluated,
explicitly shared candidate pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate.metrics import prediction_metrics, prequential_ranking_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-aware comparison of Schur and WGMM-TED run artifacts",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable run label and run directory (or full-history JSON path)",
    )
    parser.add_argument("--output", required=True, help="new or empty output directory")
    parser.add_argument(
        "--accuracy_threshold", action="append", type=float, default=[],
        help="repeatable validation-accuracy threshold for time-to-threshold",
    )
    parser.add_argument(
        "--initial_limit", type=int, default=None,
        help="optional full-evaluation prefix for initialization-only ablations",
    )
    parser.add_argument(
        "--final_results",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "optional explicit final_results JSON for a run label; search validation "
            "records are never treated as test accuracy"
        ),
    )
    return parser.parse_args(argv)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_runs(values: Sequence[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must use LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if not label or label in labels:
            raise ValueError(f"run labels must be non-empty and unique: {label!r}")
        if not path.exists():
            raise FileNotFoundError(path)
        labels.add(label)
        parsed.append((label, path))
    return parsed


def _parse_final_results(values: Sequence[str], run_labels: set[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--final_results must use LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if label not in run_labels:
            raise ValueError(f"--final_results label has no matching --run: {label!r}")
        if label in parsed:
            raise ValueError(f"duplicate --final_results label: {label!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed[label] = path
    return parsed


def _history_path(path: Path) -> tuple[Path, Path]:
    if path.is_file():
        return path, path.parent
    for name in ("history_final.json", "initialization_full_history.json"):
        candidate = path / name
        if candidate.exists():
            return candidate, path
    raise FileNotFoundError(f"no history_final.json or initialization_full_history.json in {path}")


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fingerprint(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(-1))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _geometry(records: list[dict[str, Any]]) -> dict[str, Any]:
    vectors = [row.get("z_search") for row in records if isinstance(row.get("z_search"), list)]
    if not vectors:
        return {}
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("history z_search values must form a finite 2D matrix")
    covariance = np.cov(X, rowvar=False) if X.shape[0] > 1 else np.zeros((X.shape[1], X.shape[1]))
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    logdet = float(np.sum(np.log(np.maximum(eigenvalues, 1e-12))))
    if X.shape[0] > 1:
        distances = np.sqrt(np.maximum(
            np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1), 0.0,
        ))
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
    else:
        nearest = None
    vector_keys = [_fingerprint(row) for row in X]
    architecture_keys = [
        json.dumps(
            {"operations": row.get("operations", []), "edges": row.get("edges", [])},
            sort_keys=True,
        )
        for row in records
    ]
    hp = X[:, 12:] if X.shape[1] > 12 else np.empty((X.shape[0], 0))
    result = {
        "latent_coverage_covariance_trace": float(np.trace(covariance)),
        "latent_coverage_logdet_regularized": logdet,
        "nearest_neighbor_mean": None if nearest is None else float(np.mean(nearest)),
        "nearest_neighbor_p10": None if nearest is None else float(np.percentile(nearest, 10)),
        "nearest_neighbor_median": None if nearest is None else float(np.median(nearest)),
        "nearest_neighbor_p90": None if nearest is None else float(np.percentile(nearest, 90)),
        "z_search_duplicate_rate": 1.0 - len(set(vector_keys)) / len(vector_keys),
        "architecture_duplicate_rate": 1.0 - len(set(architecture_keys)) / len(architecture_keys),
        "hp_dimension_count": int(hp.shape[1]),
        "hp_mean_range": None if not hp.size else float(np.mean(np.ptp(hp, axis=0))),
        "hp_mean_std": None if not hp.size else float(np.mean(np.std(hp, axis=0))),
    }
    return result


def _prequential(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = []
    for row in records:
        actual = _finite(row.get("val_acc"))
        mean = _finite(row.get("gp_pred_mean"))
        std = _finite(row.get("gp_pred_std"))
        if bool(row.get("valid")) and actual is not None and mean is not None:
            usable.append((row, actual, mean, std))
    if not usable:
        return {"prequential_count": 0}
    base = prediction_metrics(
        [item[1] for item in usable],
        [item[2] for item in usable],
        [item[3] for item in usable] if all(item[3] is not None for item in usable) else None,
    )
    ranking = prequential_ranking_metrics(
        [item[0] for item in usable], prequential_window=len(usable),
    )
    latent_flags = [item[0].get("latent_covered_95") for item in usable]
    observed_flags = [item[0].get("observed_covered_95") for item in usable]
    return {
        "prequential_count": len(usable),
        "prequential_mae": base.get("mae"),
        "prequential_rmse": base.get("rmse"),
        "prequential_pearson": base.get("pearson"),
        "prequential_spearman": base.get("spearman"),
        "prequential_predictive_coverage_95": base.get("predictive_95_coverage"),
        "prequential_latent_coverage_95": (
            None if any(value is None for value in latent_flags)
            else float(np.mean([bool(value) for value in latent_flags]))
        ),
        "prequential_observed_coverage_95": (
            None if any(value is None for value in observed_flags)
            else float(np.mean([bool(value) for value in observed_flags]))
        ),
        "top_region_ndcg_at_10": ranking.get("prequential_ndcg_at_10"),
        "top_region_top10_recall": ranking.get("prequential_top10_recall"),
        "top_region_top10_regret": ranking.get("prequential_top10_regret"),
        "top_region_mae": ranking.get("prequential_top_region_mae"),
    }


def _final_test_metrics(path: Path | None) -> dict[str, Any]:
    unavailable = {
        "final_results_path": None if path is None else str(path),
        "final_results_sha256": None if path is None else _sha256_file(path),
        "test_metric_status": "unavailable:no_explicit_final_results",
        "final_test_mean_best": None,
        "final_test_std_at_best": None,
        "final_test_candidate_rank": None,
        "final_test_search_step": None,
    }
    if path is None:
        return unavailable
    payload = _read_json(path)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"final_results must be a JSON list of objects: {path}")
    usable = []
    for row in payload:
        test_mean = _finite(row.get("test_mean"))
        n_valid = row.get("n_valid")
        if str(row.get("source")) != "history" or test_mean is None:
            continue
        if isinstance(n_valid, bool) or not isinstance(n_valid, int) or n_valid <= 0:
            continue
        usable.append((row, test_mean))
    if not usable:
        return {
            **unavailable,
            "test_metric_status": "unavailable:no_valid_history_final_eval",
        }
    row, test_mean = min(
        usable,
        key=lambda item: (
            -item[1],
            int(item[0].get("candidate_rank", 10**9)),
        ),
    )
    return {
        "final_results_path": str(path),
        "final_results_sha256": _sha256_file(path),
        "test_metric_status": "available:explicit_final_results",
        "final_test_mean_best": float(test_mean),
        "final_test_std_at_best": _finite(row.get("test_std")),
        "final_test_candidate_rank": row.get("candidate_rank"),
        "final_test_search_step": row.get("search_step"),
    }


def analyze_run(
    label: str,
    input_path: Path,
    *,
    initial_limit: int | None,
    thresholds: Sequence[float],
    final_results_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    history_path, run_dir = _history_path(input_path)
    payload = _read_json(history_path)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"history must be a list of objects: {history_path}")
    full = [
        row for row in payload
        if str(row.get("evaluation_fidelity", "full")) == "full"
    ]
    if initial_limit is not None:
        full = full[: int(initial_limit)]
    valid_values = [
        float(row["val_acc"]) for row in full
        if bool(row.get("valid")) and _finite(row.get("val_acc")) is not None
    ]
    best_curve: list[dict[str, Any]] = []
    current = -math.inf
    for index, row in enumerate(full, start=1):
        value = _finite(row.get("val_acc"))
        if bool(row.get("valid")) and value is not None:
            current = max(current, value)
        best_curve.append(
            {
                "label": label,
                "full_evaluation": index,
                "best_val_acc": None if current == -math.inf else current,
                "evaluation_stage": row.get("evaluation_stage"),
            }
        )
    finite_curve = [row["best_val_acc"] for row in best_curve if row["best_val_acc"] is not None]
    auc = None if not finite_curve else float(np.mean(finite_curve))
    cluster_counts = Counter(
        int(row["cluster_id"]) for row in full if row.get("cluster_id") is not None
    )
    operation_counts = Counter(op for row in full for op in row.get("operations", []))
    layer_counts = Counter(len(row.get("operations", [])) for row in full)
    edge_counts = Counter(len(row.get("edges", [])) for row in full)
    metrics: dict[str, Any] = {
        "label": label,
        "history_path": str(history_path),
        "full_evaluation_count": len(full),
        "valid_full_evaluation_count": len(valid_values),
        "valid_decode_rate": None if not full else len(valid_values) / len(full),
        "best_val_acc": None if not valid_values else max(valid_values),
        "best_so_far_auc_mean": auc,
        "cluster_selected_counts_json": json.dumps(cluster_counts, sort_keys=True),
        "operation_counts_json": json.dumps(operation_counts, sort_keys=True),
        "layer_count_distribution_json": json.dumps(layer_counts, sort_keys=True),
        "edge_count_distribution_json": json.dumps(edge_counts, sort_keys=True),
        "step_300_best_val": None,
        "label_evidence_scope": "this run's own full-fidelity history only",
        "counterfactual_claim_allowed": False,
    }
    metrics.update(_geometry(full))
    metrics.update(_prequential(full))
    metrics.update(_final_test_metrics(final_results_path))
    if len(full) >= 300:
        step_300 = [
            _finite(row.get("val_acc")) for row in full[:300] if bool(row.get("valid"))
        ]
        metrics["step_300_best_val"] = max(
            (value for value in step_300 if value is not None), default=None,
        )
    for threshold in thresholds:
        reached = next(
            (row["full_evaluation"] for row in best_curve
             if row["best_val_acc"] is not None and row["best_val_acc"] >= threshold),
            None,
        )
        metrics[f"full_evals_to_val_{threshold:g}"] = reached

    assignment_path = run_dir / "wgmm_assignments.csv"
    if not assignment_path.exists():
        assignment_path = run_dir / "gmm_cluster_assignments.csv"
    if assignment_path.exists():
        with open(assignment_path, "r", encoding="utf-8", newline="") as handle:
            assignments = list(csv.DictReader(handle))
        entropies = [_finite(row.get("responsibility_entropy")) for row in assignments]
        finite_entropy = [value for value in entropies if value is not None]
        candidate_counts = Counter(int(row["cluster_id"]) for row in assignments)
        metrics["cluster_candidate_counts_json"] = json.dumps(candidate_counts, sort_keys=True)
        metrics["responsibility_entropy_mean"] = (
            None if not finite_entropy else float(np.mean(finite_entropy))
        )
        quota_deviation = []
        for cluster_id, selected in cluster_counts.items():
            expected = len(full) * candidate_counts[cluster_id] / max(1, len(assignments))
            quota_deviation.append(abs(selected - expected))
        metrics["cluster_quota_mean_absolute_deviation"] = (
            None if not quota_deviation else float(np.mean(quota_deviation))
        )

    budget_path = run_dir / "budget_summary.json"
    if budget_path.exists():
        budget = _read_json(budget_path)
        for key in (
            "requested_shortlist_count",
            "completed_low_fidelity_count",
            "low_fidelity_candidate_count",
            "low_fidelity_invalid_count",
            "invalid_replenished_count",
            "promoted_count",
            "promoted_low_fidelity_invalid_count",
            "promoted_full_invalid_count",
            "low_fidelity_actual_epochs",
            "low_fidelity_wall_seconds",
            "low_fidelity_gpu_seconds",
            "low_fidelity_equivalent_full_evaluations",
            "selected_gmm_n_components",
        ):
            metrics[key] = budget.get(key)
    return metrics, best_curve


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"--output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    runs = _parse_runs(args.run)
    final_results = _parse_final_results(
        args.final_results,
        {label for label, _path in runs},
    )
    metrics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for label, path in runs:
        run_metrics, run_curve = analyze_run(
            label,
            path,
            initial_limit=args.initial_limit,
            thresholds=args.accuracy_threshold,
            final_results_path=final_results.get(label),
        )
        metrics.append(run_metrics)
        curves.extend(run_curve)
    _atomic_csv(metrics, output / "per_run_metrics.csv")
    _atomic_csv(curves, output / "best_so_far.csv")
    _atomic_json(
        {
            "config": vars(args),
            "runs": metrics,
            "interpretation": {
                "geometry": "label-free selected-point coverage; not performance evidence",
                "observed_metrics": "computed only from each run's own full-fidelity labels",
                "test_metrics": (
                    "available only from an explicitly supplied final_results artifact; "
                    "search validation accuracy is never relabeled as test accuracy"
                ),
                "offline_limit": (
                    "no counterfactual superiority claim unless an explicitly shared candidate "
                    "pool has complete full-fidelity labels"
                ),
            },
        },
        output / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
