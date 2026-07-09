"""Diagnostic holdout report for the Phase4 standard GP surrogate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surrogate.metrics import prediction_metrics


HP_MODE_CHOICES = ("global4", "hybrid_cond7", "layer_cond19")
PROBE_STATUS = "skipped"
PROBE_REASON = "reliable manual 2xGCN z_arch construction is unavailable"
PROBE_LIMITATION = (
    "this diagnostic cannot yet answer whether 2xGCN+HP is high-density "
    "or high-predicted-val under the surrogate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose standard BO GP surrogate on history holdout")
    parser.add_argument("--history_path", type=str, required=True)
    parser.add_argument("--hp_mode", choices=HP_MODE_CHOICES, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--arch_nz", type=int, default=12)
    parser.add_argument("--z_bound", type=float, default=2.5)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_canonical_z", action="store_true")
    parser.add_argument("--inactive_value", type=float, default=0.5)
    parser.add_argument("--probe_2gcn", action="store_true")
    parser.add_argument("--probe_hp", choices=("official", "default", "from_best_z"), default="official")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--best_z", type=str, default="")
    parser.add_argument("--actually_eval_probe", action="store_true")
    parser.add_argument("--cora_root", type=str, default="/tmp/Cora")
    parser.add_argument("--eval_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
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


def load_runtime(warnings: list[str]):
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from botorch.models import SingleTaskGP  # type: ignore
        from gpytorch.mlls import ExactMarginalLogLikelihood  # type: ignore
        from hp_modes import hp_dim_from_mode  # type: ignore
    except Exception as exc:
        warnings.append(f"required GP diagnostic dependency unavailable: {exc}")
        return None

    fit_function_name = ""
    fit_function = None
    try:
        from botorch.fit import fit_gpytorch_mll  # type: ignore

        fit_function = fit_gpytorch_mll
        fit_function_name = "fit_gpytorch_mll"
    except Exception:
        try:
            from botorch.fit import fit_gpytorch_model  # type: ignore

            fit_function = fit_gpytorch_model
            fit_function_name = "fit_gpytorch_model"
        except Exception as exc:
            warnings.append(f"no compatible BoTorch fit function available: {exc}")
            return None

    return {
        "np": np,
        "torch": torch,
        "SingleTaskGP": SingleTaskGP,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "hp_dim_from_mode": hp_dim_from_mode,
        "fit_function": fit_function,
        "fit_function_name": fit_function_name,
    }


def load_history_records(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        warnings.append(f"history path not found: {path}")
        return []
    try:
        return history_records_from_json(read_json(path))
    except Exception as exc:
        warnings.append(f"failed to read history JSON {path}: {exc}")
        return []


def maybe_canonicalize_records(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    search_dim: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not args.use_canonical_z:
        return records
    try:
        from canonicalize_search_space import canonicalize_history_record  # type: ignore
    except Exception as exc:
        warnings.append(f"canonicalization unavailable: {exc}")
        return records
    out = []
    for record in records:
        row = canonicalize_history_record(
            record,
            arch_nz=args.arch_nz,
            inactive_value=args.inactive_value,
        )
        if row.get("canonicalization_applied"):
            row["z_search"] = row.get("z_search_canonical")
        if row.get("z_search") is not None and len(row.get("z_search", [])) == search_dim:
            out.append(row)
        else:
            warnings.append(f"canonicalized record skipped: {row.get('canonicalization_warning', 'unknown warning')}")
    return out


def matrix_from_history(runtime: dict[str, Any], records: list[dict[str, Any]], args: argparse.Namespace, warnings: list[str]):
    np = runtime["np"]
    search_dim = int(args.arch_nz) + int(runtime["hp_dim_from_mode"](args.hp_mode))
    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    for record in maybe_canonicalize_records(records, args, search_dim, warnings):
        if record.get("valid") is not True:
            continue
        row_mode = record.get("hp_mode")
        z_search = record.get("z_search")
        y = value_from_record(record)
        if row_mode != args.hp_mode or y is None or z_search is None:
            continue
        try:
            z = np.asarray(z_search, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if z.shape[0] != search_dim or not np.isfinite(z).all():
            continue
        X_rows.append(z.tolist())
        y_rows.append(float(y))
    if not X_rows:
        return np.empty((0, search_dim), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.asarray(X_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


def y_standardize(runtime: dict[str, Any], y):
    torch = runtime["torch"]
    y_t = torch.as_tensor(y, dtype=torch.double).view(-1, 1)
    y_mean = y_t.mean()
    y_std = y_t.std().clamp(min=1e-6)
    return (y_t - y_mean) / y_std, y_mean, y_std


def split_indices(runtime: dict[str, Any], n: int, holdout_frac: float, seed: int):
    np = runtime["np"]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_holdout = max(1, int(math.ceil(n * min(max(float(holdout_frac), 1e-6), 0.9))))
    if n - n_holdout < 2:
        n_holdout = max(1, n - 2)
    holdout = perm[:n_holdout]
    train = perm[n_holdout:]
    return train, holdout


def fit_and_predict(runtime: dict[str, Any], X, y, args: argparse.Namespace):
    np = runtime["np"]
    torch = runtime["torch"]
    from surrogate.accuracy_gp import normalize_search_vector

    Xn = normalize_search_vector(
        torch.as_tensor(X, dtype=torch.double),
        arch_nz=args.arch_nz, hp_mode=args.hp_mode, z_bound=args.z_bound
    )
    train_idx, holdout_idx = split_indices(runtime, X.shape[0], args.holdout_frac, args.seed)
    X_train = Xn[train_idx]
    X_holdout = Xn[holdout_idx]
    y_train = y[train_idx]
    y_holdout = y[holdout_idx]
    Yn_train, y_mean, y_std = y_standardize(runtime, y_train)
    gp = runtime["SingleTaskGP"](X_train.double(), Yn_train.double())
    mll = runtime["ExactMarginalLogLikelihood"](gp.likelihood, gp)
    runtime["fit_function"](mll)
    gp.eval()
    gp.likelihood.eval()
    with torch.no_grad():
        posterior = gp.posterior(X_holdout.double())
        pred_mean_n = posterior.mean.view(-1)
        pred_std_n = posterior.variance.clamp_min(1e-12).sqrt().view(-1)
    pred_mean = pred_mean_n * y_std + y_mean
    pred_std = pred_std_n * y_std
    pred_low = pred_mean - 1.96 * pred_std
    pred_high = pred_mean + 1.96 * pred_std
    nearest_dist = torch.cdist(X_holdout.double(), X_train.double()).min(dim=1).values

    rows = []
    for pos, idx in enumerate(holdout_idx.tolist()):
        true_y = float(y_holdout[pos])
        low = float(pred_low[pos])
        high = float(pred_high[pos])
        rows.append(
            {
                "sample_idx": int(idx),
                "true_val_acc": true_y,
                "pred_mean": float(pred_mean[pos]),
                "pred_std": float(pred_std[pos]),
                "pred_95_low": low,
                "pred_95_high": high,
                "covered_by_95": bool(low <= true_y <= high),
                "nearest_train_distance": float(nearest_dist[pos]),
            }
        )

    true_vals = [row["true_val_acc"] for row in rows]
    pred_vals = [row["pred_mean"] for row in rows]
    summary = prediction_metrics(true_vals, pred_vals, [row["pred_std"] for row in rows])
    summary.update({
        "average_predictive_std": summary["mean_predictive_std"],
        "n_train": int(len(train_idx)), "n_holdout": int(len(holdout_idx)),
    })
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_idx",
        "true_val_acc",
        "pred_mean",
        "pred_std",
        "pred_95_low",
        "pred_95_high",
        "covered_by_95",
        "nearest_train_distance",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def skipped_payload(args: argparse.Namespace, warnings: list[str], fit_function_name: str = "") -> dict[str, Any]:
    return {
        "status": "skipped",
        "hp_mode": args.hp_mode,
        "history_path": args.history_path,
        "fit_function": fit_function_name or "not available",
        "gp_type": "standard SingleTaskGP",
        "use_canonical_z": bool(args.use_canonical_z),
        "probe": None,
        "summary": {
            "probe_status": PROBE_STATUS,
            "probe_reason": PROBE_REASON,
            "probe_limitation": PROBE_LIMITATION,
        },
        "warnings": warnings,
    }


def maybe_probe(args: argparse.Namespace, warnings: list[str]) -> dict[str, Any] | None:
    if not args.probe_2gcn:
        return None
    if args.actually_eval_probe:
        warnings.append(
            "actual 2xGCN probe evaluation skipped: reliable 2xGCN z_arch construction is unavailable, "
            "so no probe vector is available to train."
        )
    warnings.append(
        "2xGCN probe skipped: this diagnostic cannot reliably encode manual 2xGCN into z_arch; "
        "zero or random z_arch probes are forbidden."
    )
    return {
        "probe_status": PROBE_STATUS,
        "probe_reason": PROBE_REASON,
        "probe_limitation": PROBE_LIMITATION,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    lines = [
        "# BO GP Surrogate Diagnostic",
        "",
        f"- status: {payload.get('status')}",
        f"- hp_mode: {payload.get('hp_mode')}",
        f"- GP type: {payload.get('gp_type')}",
        f"- BoTorch fit function: {payload.get('fit_function')}",
        "- This diagnostic GP uses standard SingleTaskGP only; it does not use ConditionalMaskedKernel.",
        "- This script diagnoses standard GP holdout prediction only.",
        "- It does not reproduce the full offline-checkpoint, conditional-mask, pure-qLogEI, and online-update Phase4 process.",
        "- If the original search used --use_conditional_kernel, this diagnostic may differ from the search GP.",
        "- GP surrogate predictions are search-time validation accuracy predictions, not final test accuracy.",
        "- GP posterior std is not multi-seed training std.",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "rmse",
        "mae",
        "spearman",
        "pearson",
        "predictive_95_coverage",
        "average_predictive_std",
        "n_train",
        "n_holdout",
        "probe_status",
        "probe_reason",
        "probe_limitation",
    ]:
        lines.append(f"- {key}: {summary.get(key, 'not available')}")
    if payload.get("probe"):
        lines.extend(["", "## 2xGCN Probe", ""])
        for key, value in payload["probe"].items():
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_id = Path(args.history_path).parent.name or "history"
    output_dir = Path(args.output_dir or f"results/diagnostics/gp_surrogate_{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    runtime = load_runtime(warnings)
    if runtime is None:
        payload = skipped_payload(args, warnings)
        (output_dir / "gp_surrogate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_csv(output_dir / "gp_surrogate_holdout_predictions.csv", [])
        write_report(output_dir / "gp_surrogate_report.md", payload)
        print(f"GP surrogate diagnostic skipped; wrote {output_dir}")
        return

    records = load_history_records(Path(args.history_path), warnings)
    X, y = matrix_from_history(runtime, records, args, warnings)
    if X.shape[0] < 4:
        warnings.append(f"not enough valid history samples for holdout GP diagnostic: n={X.shape[0]}")
        payload = skipped_payload(args, warnings, runtime["fit_function_name"])
        (output_dir / "gp_surrogate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_csv(output_dir / "gp_surrogate_holdout_predictions.csv", [])
        write_report(output_dir / "gp_surrogate_report.md", payload)
        print(f"GP surrogate diagnostic skipped; wrote {output_dir}")
        return

    try:
        rows, summary = fit_and_predict(runtime, X, y, args)
        status = "completed"
    except Exception as exc:
        warnings.append(f"GP fit/predict failed: {exc}")
        payload = skipped_payload(args, warnings, runtime["fit_function_name"])
        (output_dir / "gp_surrogate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_csv(output_dir / "gp_surrogate_holdout_predictions.csv", [])
        write_report(output_dir / "gp_surrogate_report.md", payload)
        print(f"GP surrogate diagnostic skipped; wrote {output_dir}")
        return

    payload = {
        "status": status,
        "hp_mode": args.hp_mode,
        "history_path": args.history_path,
        "fit_function": runtime["fit_function_name"],
        "gp_type": "standard SingleTaskGP",
        "use_canonical_z": bool(args.use_canonical_z),
        "summary": {
            **summary,
            "probe_status": PROBE_STATUS,
            "probe_reason": PROBE_REASON,
            "probe_limitation": PROBE_LIMITATION,
        },
        "probe": maybe_probe(args, warnings),
        "warnings": warnings,
    }
    write_csv(output_dir / "gp_surrogate_holdout_predictions.csv", rows)
    (output_dir / "gp_surrogate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(output_dir / "gp_surrogate_report.md", payload)
    print(f"GP surrogate diagnostic completed; wrote {output_dir}")


if __name__ == "__main__":
    main()
