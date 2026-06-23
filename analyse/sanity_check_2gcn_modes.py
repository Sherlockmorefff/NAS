"""Quick diagnostic for manual 2xGCN invariance across hp_mode settings."""

from __future__ import annotations

import argparse
import ast
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HP_MODES = ("global4", "hybrid_cond7", "layer_cond19")
FALLBACK_2GCN = {
    "operations": ["GCNConv", "GCNConv", "Identity"],
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "effective_layers": 2,
}
HP_PROFILES = [
    {
        "name": "current_default",
        "lr": 1e-3,
        "dropout": 0.5,
        "hidden_dim": 64,
        "l2": 5e-4,
    },
    {
        "name": "official_like_gcn",
        "lr": 0.01,
        "dropout": 0.5,
        "hidden_dim": 16,
        "l2": 5e-4,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check manual 2xGCN across hp_modes")
    parser.add_argument("--cora_root", type=str, default="/tmp/Cora")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--full_epochs", type=int, default=200)
    parser.add_argument("--full_patience", type=int, default=30)
    parser.add_argument("--run_full_check", action="store_true")
    parser.add_argument("--output", type=str, default="results/diagnostics/2gcn_mode_sanity.json")
    parser.add_argument("--output_md", type=str, default="results/diagnostics/2gcn_mode_sanity.md")
    parser.add_argument("--log_dir", type=str, default="logs/diagnostics")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def config_from_ops(ops: list[str], edges: list[tuple[int, int]] | list[list[int]]) -> dict[str, Any]:
    return {
        "operations": list(ops),
        "edges": [tuple(edge) for edge in edges],
        "effective_layers": sum(1 for op in ops if op != "Identity"),
    }


def fallback_config(warnings: list[str]) -> dict[str, Any]:
    warnings.append("warning: using fallback manual 2xGCN definition")
    return config_from_ops(FALLBACK_2GCN["operations"], FALLBACK_2GCN["edges"])


def load_2gcn_from_final_eval(path: Path, warnings: list[str]) -> dict[str, Any]:
    """Statically parse final_eval.py for build_candidates/base_2gcn."""

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        warnings.append(f"warning: failed to read final_eval.py for baseline definition: {exc}")
        return fallback_config(warnings)

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        warnings.append(f"warning: failed to parse final_eval.py AST: {exc}")
        return fallback_config(warnings)

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "build_candidates":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "base_2gcn" for target in child.targets):
                continue
            call = child.value
            if not isinstance(call, ast.Call):
                continue
            func_name = getattr(call.func, "id", getattr(call.func, "attr", ""))
            if func_name != "config_from_ops" or len(call.args) < 2:
                continue
            try:
                ops = ast.literal_eval(call.args[0])
                edges = ast.literal_eval(call.args[1])
            except Exception as exc:
                warnings.append(f"warning: failed to evaluate base_2gcn AST literals: {exc}")
                return fallback_config(warnings)
            if isinstance(ops, list) and isinstance(edges, list):
                return config_from_ops([str(op) for op in ops], edges)

    warnings.append("warning: could not identify build_candidates/base_2gcn in final_eval.py")
    return fallback_config(warnings)


def cora_data_dir(cora_root: str) -> tuple[Path, Path]:
    root = Path(cora_root)
    pyg_root = root.parent if root.name == "Cora" else root
    return pyg_root, pyg_root / "Cora"


def cora_artifacts_available(cora_root: str) -> tuple[bool, str]:
    _pyg_root, data_dir = cora_data_dir(cora_root)
    processed = data_dir / "processed"
    raw = data_dir / "raw"
    if processed.exists() and any(processed.iterdir()):
        return True, ""
    if raw.exists() and any(raw.iterdir()):
        return True, ""
    return False, (
        f"Cora data not found under {data_dir}. Skipping to avoid an implicit download "
        "during this diagnostic."
    )


def load_runtime(warnings: list[str]):
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import torch_geometric.transforms as T  # type: ignore
        from torch_geometric.datasets import Planetoid  # type: ignore

        import eval_utils  # type: ignore
        import hp_modes  # type: ignore
    except Exception as exc:
        warnings.append(f"warning: required dependency unavailable: {exc}")
        return None
    return {
        "np": np,
        "torch": torch,
        "T": T,
        "Planetoid": Planetoid,
        "eval_utils": eval_utils,
        "hp_modes": hp_modes,
    }


def load_cora(runtime: dict[str, Any], cora_root: str, device):
    pyg_root, _data_dir = cora_data_dir(cora_root)
    dataset = runtime["Planetoid"](
        root=pyg_root.as_posix(),
        name="Cora",
        transform=runtime["T"].NormalizeFeatures(),
    )
    return dataset[0].to(device), dataset.num_features, dataset.num_classes


def condition_mask_for(runtime: dict[str, Any], ops: list[str], hp_mode: str) -> dict[str, Any]:
    return runtime["hp_modes"].condition_mask_dict_from_ops(ops, hp_mode)


def mask_expectation_warning(mask: dict[str, Any], hp_mode: str) -> str:
    vector = list(mask.get("vector", []))
    if hp_mode == "global4":
        if vector != [1.0, 1.0, 1.0, 1.0]:
            return "global4 mask is not all-active for global HP"
        return ""
    if hp_mode == "hybrid_cond7":
        if vector[:4] != [1.0, 1.0, 1.0, 1.0] or any(float(v) != 0.0 for v in vector[4:]):
            return "hybrid_cond7 conditional GAT/SAGE/GIN mask is not inactive for pure GCN"
        return ""
    if hp_mode == "layer_cond19":
        if vector[:4] != [1.0, 1.0, 1.0, 1.0] or any(float(v) != 0.0 for v in vector[4:]):
            return "layer_cond19 per-layer GAT/SAGE/GIN mask is not inactive for pure GCN"
        return ""
    return ""


def initial_logits_and_params(
    runtime: dict[str, Any],
    config: dict[str, Any],
    data,
    in_ch: int,
    out_ch: int,
    hp: dict[str, Any],
    seed: int,
    device,
) -> tuple[int | None, Any, str]:
    torch = runtime["torch"]
    np = runtime["np"]
    eval_utils = runtime["eval_utils"]
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = eval_utils.DynamicGNN(
            config,
            in_ch,
            out_ch,
            dropout=float(hp["dropout"]),
            hidden_dim=int(hp["hidden_dim"]),
        ).to(device)
        n_params = int(sum(param.numel() for param in model.parameters()))
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index).detach().cpu()
        return n_params, logits, ""
    except Exception as exc:
        return None, None, f"initial logits unavailable: {exc}"


def run_single(
    runtime: dict[str, Any],
    config: dict[str, Any],
    data,
    in_ch: int,
    out_ch: int,
    hp_mode: str,
    hp: dict[str, Any],
    seed: int,
    epochs: int,
    patience: int,
    device,
) -> tuple[dict[str, Any], Any]:
    mask = condition_mask_for(runtime, config["operations"], hp_mode)
    mask_warning = mask_expectation_warning(mask, hp_mode)
    n_params, logits, logits_warning = initial_logits_and_params(
        runtime,
        config,
        data,
        in_ch,
        out_ch,
        hp,
        seed,
        device,
    )
    row: dict[str, Any] = {
        "hp_mode": hp_mode,
        "seed": int(seed),
        "hp_profile": hp["name"],
        "operations": list(config["operations"]),
        "edges": [list(edge) for edge in config["edges"]],
        "effective_layers": int(config["effective_layers"]),
        "lr": float(hp["lr"]),
        "dropout": float(hp["dropout"]),
        "hidden_dim": int(hp["hidden_dim"]),
        "l2": float(hp["l2"]),
        "val_acc": None,
        "test_acc": None,
        "is_valid": False,
        "n_params": n_params,
        "condition_mask": mask,
        "condition_mask_vector": mask.get("vector"),
        "first_forward_logits_available": logits is not None,
        "warning": "; ".join(msg for msg in (mask_warning, logits_warning) if msg),
    }

    try:
        val_acc, is_valid, test_acc = runtime["eval_utils"].train_and_eval_arch(
            config=config,
            data=data,
            in_ch=in_ch,
            out_ch=out_ch,
            lr=float(hp["lr"]),
            dropout=float(hp["dropout"]),
            hidden_dim=int(hp["hidden_dim"]),
            weight_decay=float(hp["l2"]),
            device=device,
            max_epochs=int(epochs),
            patience=int(patience),
            seed=int(seed),
            track_test=True,
        )
        row["val_acc"] = float(val_acc)
        row["test_acc"] = float(test_acc)
        row["is_valid"] = bool(is_valid)
        row["abnormally_low_accuracy"] = (
            safe_float(val_acc) is not None
            and safe_float(test_acc) is not None
            and (float(val_acc) < 0.2 or float(test_acc) < 0.2)
        )
    except Exception as exc:
        row["warning"] = "; ".join(msg for msg in (row["warning"], f"train/eval failed: {exc}") if msg)
        row["abnormally_low_accuracy"] = None
    return row, logits


def max_abs_tensor_diff(runtime: dict[str, Any], tensors: list[Any]) -> float | None:
    if len(tensors) < 2 or any(tensor is None for tensor in tensors):
        return None
    torch = runtime["torch"]
    max_diff = 0.0
    for idx, left in enumerate(tensors):
        for right in tensors[idx + 1:]:
            diff = torch.max(torch.abs(left - right)).item()
            max_diff = max(max_diff, float(diff))
    return max_diff


def summarize(rows: list[dict[str, Any]], logits_by_key: dict[tuple[str, int], list[Any]], runtime: dict[str, Any] | None):
    max_val_diff = 0.0
    max_test_diff = 0.0
    max_logits_diff = 0.0
    logits_available = False
    warnings_out: list[str] = []
    pass_flag = True
    n_total = len(rows)
    n_valid = sum(1 for row in rows if row.get("is_valid") is True)
    n_invalid = n_total - n_valid

    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((row["hp_profile"], row["seed"]), []).append(row)

    for key, group_rows in by_key.items():
        if len(group_rows) != len(HP_MODES):
            pass_flag = False
            warnings_out.append(f"missing hp_mode result for {key}")
            continue
        val_values = [safe_float(row.get("val_acc")) for row in group_rows]
        test_values = [safe_float(row.get("test_acc")) for row in group_rows]
        if any(value is None for value in val_values + test_values):
            pass_flag = False
            warnings_out.append(f"missing val/test accuracy for {key}")
            continue
        val_diff = max(val_values) - min(val_values)  # type: ignore[arg-type]
        test_diff = max(test_values) - min(test_values)  # type: ignore[arg-type]
        max_val_diff = max(max_val_diff, float(val_diff))
        max_test_diff = max(max_test_diff, float(test_diff))
        if val_diff > 1e-6 or test_diff > 1e-6:
            pass_flag = False
            warnings_out.append(
                f"mode variance detected for {key}: val_diff={val_diff:.6g}, test_diff={test_diff:.6g}"
            )

        if runtime is not None:
            logits_diff = max_abs_tensor_diff(runtime, logits_by_key.get(key, []))
            if logits_diff is not None:
                logits_available = True
                max_logits_diff = max(max_logits_diff, logits_diff)
                if logits_diff > 1e-6:
                    pass_flag = False
                    warnings_out.append(f"initial logits differ for {key}: max_abs_diff={logits_diff:.6g}")

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "all_runs_valid": n_total > 0 and n_invalid == 0,
        "max_abs_val_diff_across_modes": max_val_diff if rows else None,
        "max_abs_test_diff_across_modes": max_test_diff if rows else None,
        "max_abs_initial_logits_diff_across_modes": max_logits_diff if logits_available else None,
        "initial_logits_check": "available" if logits_available else "not available",
        "mode_invariance_pass": bool(pass_flag) if rows else None,
        "warnings": warnings_out,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fmt(value: Any) -> str:
    num = safe_float(value)
    if num is None:
        return "not available"
    return f"{num:.4f}"


def table_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 2xGCN hp_mode Sanity Check",
        "",
        f"- status: {payload.get('status')}",
        f"- run_full_check: {payload.get('run_full_check')}",
        f"- epochs: {payload.get('epochs')}",
        f"- patience: {payload.get('patience')}",
        "- quick run only checks code path and mode consistency; low/invalid accuracy in a 1-epoch dry run is not a final performance conclusion.",
        "",
        "## Manual 2xGCN Definition",
        "",
        f"- operations: `{payload.get('config', {}).get('operations')}`",
        f"- edges: `{payload.get('config', {}).get('edges')}`",
        "",
    ]

    if payload.get("status") == "skipped":
        summary = payload.get("summary", {})
        lines.extend(
            [
                "## Summary",
                "",
                f"- n_total: {summary.get('n_total')}",
                f"- n_valid: {summary.get('n_valid')}",
                f"- n_invalid: {summary.get('n_invalid')}",
                f"- all_runs_valid: {summary.get('all_runs_valid')}",
                f"- mode_invariance_pass: {summary.get('mode_invariance_pass')}",
                "- mode_invariance_pass only checks whether hp_modes behave consistently under the same fixed architecture and HP.",
                "- It does not by itself prove that the 2xGCN baseline reaches a good Cora accuracy.",
                "",
                "## Skipped",
                "",
            ]
        )
        for warning in payload.get("warnings", []):
            lines.append(f"- {warning}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    summary = payload.get("summary", {})
    lines.extend(
        [
            "## Summary",
            "",
            f"- n_total: {summary.get('n_total')}",
            f"- n_valid: {summary.get('n_valid')}",
            f"- n_invalid: {summary.get('n_invalid')}",
            f"- all_runs_valid: {summary.get('all_runs_valid')}",
            f"- max_abs_val_diff_across_modes: {fmt(summary.get('max_abs_val_diff_across_modes'))}",
            f"- max_abs_test_diff_across_modes: {fmt(summary.get('max_abs_test_diff_across_modes'))}",
            f"- max_abs_initial_logits_diff_across_modes: {fmt(summary.get('max_abs_initial_logits_diff_across_modes'))}",
            f"- initial_logits_check: {summary.get('initial_logits_check')}",
            f"- mode_invariance_pass: {summary.get('mode_invariance_pass')}",
            "- mode_invariance_pass only checks whether hp_modes behave consistently under the same fixed architecture and HP.",
            "- It does not by itself prove that the 2xGCN baseline reaches a good Cora accuracy.",
            "",
            "## Results",
            "",
            table_row(["hp_profile", "seed", "hp_mode", "val_acc", "test_acc", "is_valid", "n_params", "low_acc", "warning"]),
            table_row(["---"] * 9),
        ]
    )
    for row in payload.get("results", []):
        lines.append(
            table_row(
                [
                    row.get("hp_profile"),
                    row.get("seed"),
                    row.get("hp_mode"),
                    fmt(row.get("val_acc")),
                    fmt(row.get("test_acc")),
                    row.get("is_valid"),
                    row.get("n_params", "not available"),
                    row.get("abnormally_low_accuracy", "not available"),
                    row.get("warning", ""),
                ]
            )
        )
    lines.extend(["", "## Warnings", ""])
    all_warnings = list(payload.get("warnings", [])) + list(summary.get("warnings", []))
    if all_warnings:
        for warning in all_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def skipped_payload(args: argparse.Namespace, config: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return {
        "status": "skipped",
        "generated_at": utc_now(),
        "run_full_check": bool(args.run_full_check),
        "epochs": int(args.full_epochs if args.run_full_check else args.epochs),
        "patience": int(args.full_patience if args.run_full_check else args.patience),
        "seeds": list(args.seeds),
        "hp_modes": list(HP_MODES),
        "hp_profiles": HP_PROFILES,
        "config": {
            "operations": list(config["operations"]),
            "edges": [list(edge) for edge in config["edges"]],
            "effective_layers": int(config["effective_layers"]),
        },
        "results": [],
        "summary": {
            "n_total": 0,
            "n_valid": 0,
            "n_invalid": 0,
            "all_runs_valid": False,
            "max_abs_val_diff_across_modes": None,
            "max_abs_test_diff_across_modes": None,
            "max_abs_initial_logits_diff_across_modes": None,
            "initial_logits_check": "not available",
            "mode_invariance_pass": None,
            "warnings": [],
        },
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    project_root = Path(__file__).resolve().parent
    final_eval_path = project_root / "final_eval.py"
    if not final_eval_path.exists():
        warnings.append(f"warning: final_eval.py not found at script directory path: {final_eval_path}")
    config = load_2gcn_from_final_eval(final_eval_path, warnings)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    data_available, data_warning = cora_artifacts_available(args.cora_root)
    if not data_available:
        warnings.append(f"warning: {data_warning}")

    runtime = load_runtime(warnings)
    if not data_available or runtime is None:
        payload = skipped_payload(args, config, warnings)
        write_json(Path(args.output), payload)
        write_markdown(Path(args.output_md), payload)
        print(f"Sanity check skipped; wrote {args.output} and {args.output_md}")
        return

    torch = runtime["torch"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = int(args.full_epochs if args.run_full_check else args.epochs)
    patience = int(args.full_patience if args.run_full_check else args.patience)

    try:
        data, in_ch, out_ch = load_cora(runtime, args.cora_root, device)
    except Exception as exc:
        warnings.append(f"warning: failed to load Cora without crashing: {exc}")
        payload = skipped_payload(args, config, warnings)
        write_json(Path(args.output), payload)
        write_markdown(Path(args.output_md), payload)
        print(f"Sanity check skipped; wrote {args.output} and {args.output_md}")
        return

    results: list[dict[str, Any]] = []
    logits_by_key: dict[tuple[str, int], list[Any]] = {}
    for hp in HP_PROFILES:
        for seed in args.seeds:
            key = (hp["name"], int(seed))
            logits_by_key[key] = []
            for hp_mode in HP_MODES:
                row, logits = run_single(
                    runtime,
                    config,
                    data,
                    in_ch,
                    out_ch,
                    hp_mode,
                    hp,
                    int(seed),
                    epochs,
                    patience,
                    device,
                )
                results.append(row)
                logits_by_key[key].append(logits)

    summary = summarize(results, logits_by_key, runtime)
    payload = {
        "status": "completed",
        "generated_at": utc_now(),
        "run_full_check": bool(args.run_full_check),
        "epochs": epochs,
        "patience": patience,
        "seeds": list(args.seeds),
        "device": str(device),
        "hp_modes": list(HP_MODES),
        "hp_profiles": HP_PROFILES,
        "config": {
            "operations": list(config["operations"]),
            "edges": [list(edge) for edge in config["edges"]],
            "effective_layers": int(config["effective_layers"]),
        },
        "results": results,
        "summary": summary,
        "warnings": warnings,
    }
    write_json(Path(args.output), payload)
    write_markdown(Path(args.output_md), payload)
    print(f"Sanity check completed; wrote {args.output} and {args.output_md}")


if __name__ == "__main__":
    main()
