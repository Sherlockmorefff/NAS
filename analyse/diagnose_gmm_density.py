"""Diagnostic report for weighted diagonal GMM density over NAS history."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


HP_MODE_CHOICES = ("global4", "hybrid_cond7", "layer_cond19")
PROBE_STATUS = "skipped"
PROBE_REASON = "reliable manual 2xGCN z_arch construction is unavailable"
PROBE_LIMITATION = (
    "this diagnostic cannot yet answer whether 2xGCN+HP is high-density "
    "or high-predicted-val under the surrogate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose weighted diagonal GMM density")
    parser.add_argument("--history_paths", nargs="+", default=[])
    parser.add_argument("--hp_mode", choices=HP_MODE_CHOICES, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--arch_nz", type=int, default=12)
    parser.add_argument("--top_frac", type=float, default=0.3)
    parser.add_argument("--n_components", type=int, default=4)
    parser.add_argument("--weight_temp", type=float, default=8.0)
    parser.add_argument("--z_bound", type=float, default=2.5)
    parser.add_argument("--use_canonical_z", action="store_true")
    parser.add_argument("--inactive_value", type=float, default=0.5)
    parser.add_argument("--include_2gcn_probe", action="store_true")
    parser.add_argument("--probe_hp", choices=("official", "default", "from_best_z"), default="official")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--best_z", type=str, default="")
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def history_records_from_json(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in ("history", "records", "trials", "results"):
            value = obj.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if any(key in obj for key in ("z_search", "val_acc", "value", "best_value")):
            return [obj]
    return []


def value_from_record(record: dict[str, Any]) -> float | None:
    for key in ("val_acc", "value", "best_value"):
        if key in record:
            return safe_float(record.get(key))
    return None


def history_records(paths: list[str], warnings: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            warnings.append(f"history path not found: {path}")
            continue
        try:
            obj = read_json(path)
        except Exception as exc:
            warnings.append(f"failed to read history JSON {path}: {exc}")
            continue
        out.extend(history_records_from_json(obj))
    return out


def load_runtime(warnings: list[str]):
    try:
        import numpy as np  # type: ignore

        from hp_modes import hp_dim_from_mode  # type: ignore
        from weighted_diag_gmm_init import (  # type: ignore
            WeightedDiagonalGMM,
            load_history_vectors,
            softmax_weights,
            standardize_apply,
            standardize_fit,
        )
    except Exception as exc:
        warnings.append(f"required GMM diagnostic dependency unavailable: {exc}")
        return None
    return {
        "np": np,
        "hp_dim_from_mode": hp_dim_from_mode,
        "WeightedDiagonalGMM": WeightedDiagonalGMM,
        "load_history_vectors": load_history_vectors,
        "softmax_weights": softmax_weights,
        "standardize_apply": standardize_apply,
        "standardize_fit": standardize_fit,
    }


def canonicalized_history_path(
    args: argparse.Namespace,
    output_dir: Path,
    search_dim: int,
    warnings: list[str],
) -> list[str]:
    try:
        from canonicalize_search_space import canonicalize_history_record  # type: ignore
    except Exception as exc:
        warnings.append(f"canonicalization unavailable: {exc}")
        return list(args.history_paths)

    records = history_records(list(args.history_paths), warnings)
    canonical_records = []
    for record in records:
        row = canonicalize_history_record(
            record,
            arch_nz=args.arch_nz,
            inactive_value=args.inactive_value,
        )
        if row.get("canonicalization_applied"):
            row["z_search"] = row.get("z_search_canonical")
        if row.get("z_search") is not None and len(row.get("z_search", [])) == search_dim:
            canonical_records.append(row)
        else:
            warnings.append(f"canonicalized record skipped: {row.get('canonicalization_warning', 'unknown warning')}")
    path = output_dir / "canonicalized_history_for_gmm_density.json"
    path.write_text(json.dumps(canonical_records, indent=2, ensure_ascii=False), encoding="utf-8")
    return [path.as_posix()]


def fit_gmm(runtime: dict[str, Any], X, y, args: argparse.Namespace):
    np = runtime["np"]
    top_frac = min(max(float(args.top_frac), 1e-6), 1.0)
    n_top = max(1, int(math.ceil(X.shape[0] * top_frac)))
    order = np.argsort(y)[::-1]
    top_idx = order[:n_top]
    X_top = X[top_idx]
    y_top = y[top_idx]
    X_top_std, mean, std = runtime["standardize_fit"](X_top)
    weights = runtime["softmax_weights"](y_top, temperature=args.weight_temp)
    gmm = runtime["WeightedDiagonalGMM"](
        n_components=args.n_components,
        max_iter=100,
        tol=1e-4,
        var_floor=1e-4,
        reg_covar=1e-6,
        random_state=42,
    ).fit(X_top_std, sample_weight=weights)
    X_std = runtime["standardize_apply"](X, mean, std)
    return gmm, X_std, top_idx, weights


def component_log_prob(np, gmm, X_std):
    X = np.asarray(X_std, dtype=np.float64)
    diff = X[:, None, :] - gmm.means_[None, :, :]
    log_det = np.sum(np.log(gmm.vars_), axis=1)
    quad = np.sum((diff * diff) / gmm.vars_[None, :, :], axis=2)
    d = X.shape[1]
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det[None, :] + quad)


def logsumexp(np, values, axis: int = -1):
    vmax = np.max(values, axis=axis, keepdims=True)
    vmax = np.where(np.isfinite(vmax), vmax, 0.0)
    out = vmax + np.log(np.sum(np.exp(values - vmax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def score_samples(runtime: dict[str, Any], gmm, X_std):
    np = runtime["np"]
    if hasattr(gmm, "score_samples") and callable(getattr(gmm, "score_samples")):
        log_density = gmm.score_samples(X_std)
        log_prob = component_log_prob(np, gmm, X_std)
    else:
        log_prob = component_log_prob(np, gmm, X_std)
        log_density = logsumexp(np, log_prob + np.log(np.maximum(gmm.weights_, 1e-300))[None, :], axis=1)
    log_resp = log_prob + np.log(np.maximum(gmm.weights_, 1e-300))[None, :] - log_density[:, None]
    resp = np.exp(log_resp)
    diff = X_std[:, None, :] - gmm.means_[None, :, :]
    mahal = np.sum((diff * diff) / gmm.vars_[None, :, :], axis=2)
    nearest_component = np.argmin(mahal, axis=1)
    return log_density, resp, mahal, nearest_component


def ranks_to_percentiles(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0 for _ in values]
    n = len(values)
    if n <= 1:
        return [100.0 for _ in values]
    for rank, (idx, _value) in enumerate(indexed):
        out[idx] = 100.0 * rank / (n - 1)
    return out


def pearson(values_a: list[float], values_b: list[float]) -> float | None:
    n = len(values_a)
    if n < 2:
        return None
    mean_a = sum(values_a) / n
    mean_b = sum(values_b) / n
    da = [v - mean_a for v in values_a]
    db = [v - mean_b for v in values_b]
    denom = math.sqrt(sum(v * v for v in da) * sum(v * v for v in db))
    if denom <= 1e-12:
        return None
    return sum(a * b for a, b in zip(da, db)) / denom


def spearman(values_a: list[float], values_b: list[float]) -> float | None:
    return pearson(ranks_to_percentiles(values_a), ranks_to_percentiles(values_b))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def top_recall_by_density(y: list[float], density: list[float], frac: float) -> float | None:
    n = len(y)
    if n == 0:
        return None
    k = max(1, int(math.ceil(n * frac)))
    top_y = {idx for idx, _ in sorted(enumerate(y), key=lambda item: item[1], reverse=True)[:k]}
    top_density = {idx for idx, _ in sorted(enumerate(density), key=lambda item: item[1], reverse=True)[:k]}
    return len(top_y & top_density) / k


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_idx",
        "val_acc",
        "log_density",
        "density_percentile",
        "nearest_component",
        "max_responsibility",
        "min_diagonal_mahalanobis_distance",
        "is_top_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    lines = [
        "# GMM Density Diagnostic",
        "",
        f"- status: {payload.get('status')}",
        f"- hp_mode: {payload.get('hp_mode')}",
        "- GMM density indicates how much a point resembles the historical high-performing region.",
        "- GMM density is not predicted accuracy.",
        "- GMM is fitted only on the selected top-fraction history samples, then all valid history samples are scored under that fitted density.",
        "- Log density is computed as log p(z) = logsumexp_k [log pi_k + log N(z | mu_k, diag(var_k))].",
        "- pi_k uses fitted `gmm.weights_`; history softmax sample weights only influence EM fitting.",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "n_samples",
        "n_top_samples",
        "spearman_log_density_val_acc",
        "pearson_log_density_val_acc",
        "top20_recall_by_density",
        "top30_recall_by_density",
        "mean_val_acc_high_density",
        "mean_val_acc_low_density",
        "probe_status",
        "probe_reason",
        "probe_limitation",
    ]:
        lines.append(f"- {key}: {summary.get(key, 'not available')}")
    if payload.get("probe"):
        lines.extend(["", "## 2xGCN Probe", ""])
        for key, value in payload["probe"].items():
            lines.append(f"- {key}: {value}")
    warnings = payload.get("warnings", [])
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def skipped_payload(args: argparse.Namespace, warnings: list[str]) -> dict[str, Any]:
    return {
        "status": "skipped",
        "hp_mode": args.hp_mode,
        "history_paths": list(args.history_paths),
        "summary": {
            "probe_status": PROBE_STATUS,
            "probe_reason": PROBE_REASON,
            "probe_limitation": PROBE_LIMITATION,
        },
        "probe": None,
        "warnings": warnings,
    }


def maybe_probe(runtime: dict[str, Any], gmm, args: argparse.Namespace, warnings: list[str]):
    if not args.include_2gcn_probe:
        return None
    if args.probe_hp != "from_best_z":
        warnings.append(
            "2xGCN probe skipped: reliable manual 2xGCN z_arch encoding is unavailable; "
            "zero or random z_arch probes are forbidden."
        )
        return {
            "probe_status": PROBE_STATUS,
            "probe_reason": PROBE_REASON,
            "probe_limitation": PROBE_LIMITATION,
        }
    if not args.best_z or not Path(args.best_z).exists():
        warnings.append("2xGCN probe skipped: --probe_hp from_best_z requires an existing --best_z path")
        return {
            "probe_status": PROBE_STATUS,
            "probe_reason": PROBE_REASON,
            "probe_limitation": PROBE_LIMITATION,
        }
    warnings.append(
        "2xGCN probe skipped: from_best_z supplies an existing z_arch, but this diagnostic does not "
        "rewrite HP tails in v1 without a reliable HP encoder."
    )
    return {
        "probe_status": PROBE_STATUS,
        "probe_reason": PROBE_REASON,
        "probe_limitation": PROBE_LIMITATION,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or f"results/diagnostics/gmm_density_{args.hp_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    runtime = load_runtime(warnings)
    if runtime is None:
        payload = skipped_payload(args, warnings)
        (output_dir / "gmm_density_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_report(output_dir / "gmm_density_report.md", payload)
        print(f"GMM density diagnostic skipped; wrote {output_dir}")
        return

    search_dim = int(args.arch_nz) + int(runtime["hp_dim_from_mode"](args.hp_mode))
    history_paths = list(args.history_paths)
    if args.use_canonical_z:
        history_paths = canonicalized_history_path(args, output_dir, search_dim, warnings)

    X, y, meta = runtime["load_history_vectors"](history_paths, args.hp_mode, search_dim)
    if X.shape[0] == 0:
        warnings.append("no valid history samples available for GMM density diagnostic")
        payload = skipped_payload(args, warnings)
        payload["meta"] = meta
        (output_dir / "gmm_density_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_report(output_dir / "gmm_density_report.md", payload)
        print(f"GMM density diagnostic skipped; wrote {output_dir}")
        return

    gmm, X_std, top_idx, sample_weights = fit_gmm(runtime, X, y, args)
    log_density, resp, mahal, nearest_component = score_samples(runtime, gmm, X_std)
    density_percentiles = ranks_to_percentiles([float(v) for v in log_density])
    top_set = set(int(idx) for idx in top_idx.tolist())

    rows: list[dict[str, Any]] = []
    for idx in range(X.shape[0]):
        rows.append(
            {
                "sample_idx": idx,
                "val_acc": float(y[idx]),
                "log_density": float(log_density[idx]),
                "density_percentile": float(density_percentiles[idx]),
                "nearest_component": int(nearest_component[idx]),
                "max_responsibility": float(resp[idx].max()),
                "min_diagonal_mahalanobis_distance": float(mahal[idx].min()),
                "is_top_fraction": idx in top_set,
            }
        )

    n = len(rows)
    k30 = max(1, int(math.ceil(n * 0.3)))
    by_density = sorted(rows, key=lambda row: row["log_density"], reverse=True)
    high = by_density[:k30]
    low = by_density[-k30:]
    y_list = [float(v) for v in y.tolist()]
    d_list = [float(v) for v in log_density.tolist()]
    summary = {
        "status": "completed",
        "n_samples": int(n),
        "n_top_samples": int(len(top_idx)),
        "spearman_log_density_val_acc": spearman(d_list, y_list),
        "pearson_log_density_val_acc": pearson(d_list, y_list),
        "top20_recall_by_density": top_recall_by_density(y_list, d_list, 0.2),
        "top30_recall_by_density": top_recall_by_density(y_list, d_list, 0.3),
        "mean_val_acc_high_density": mean([row["val_acc"] for row in high]),
        "mean_val_acc_low_density": mean([row["val_acc"] for row in low]),
        "gmm_n_components": int(gmm.n_components),
        "gmm_weights": [float(v) for v in gmm.weights_.tolist()],
        "history_softmax_sample_weight_note": "sample weights were used for EM fitting and are not pi_k",
        "top_history_softmax_sample_weights": [float(v) for v in sample_weights.tolist()],
        "probe_status": PROBE_STATUS,
        "probe_reason": PROBE_REASON,
        "probe_limitation": PROBE_LIMITATION,
    }
    payload = {
        "status": "completed",
        "hp_mode": args.hp_mode,
        "history_paths": history_paths,
        "use_canonical_z": bool(args.use_canonical_z),
        "summary": summary,
        "meta": meta,
        "probe": maybe_probe(runtime, gmm, args, warnings),
        "warnings": warnings,
    }
    write_csv(output_dir / "gmm_density_per_sample.csv", rows)
    (output_dir / "gmm_density_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(output_dir / "gmm_density_report.md", payload)
    print(f"GMM density diagnostic completed; wrote {output_dir}")


if __name__ == "__main__":
    main()
