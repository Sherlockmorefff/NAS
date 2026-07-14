"""Multi-seed final evaluation of architectures recorded during NAS search."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_utils import (  # noqa: E402
    DEFAULT_GAT_HEADS,
    DEFAULT_GIN_EPS,
    DEFAULT_SAGE_AGGR,
    train_and_eval_arch,
)
from hp_modes import validate_hp_mode  # noqa: E402


HP_MODE_CHOICES = ("global4", "hybrid_cond7", "layer_cond19")
DEFAULT_LR = 1e-3
DEFAULT_DROPOUT = 0.5
DEFAULT_HIDDEN_DIM = 64
DEFAULT_L2 = 5e-4

REQUIRED_HISTORY_FIELDS = (
    "operations",
    "edges",
    "val_acc",
    "valid",
    "lr",
    "dropout",
    "hidden_dim",
)
SIGNATURE_FIELDS = (
    "operations",
    "edges",
    "lr",
    "dropout",
    "hidden_dim",
    "l2",
    "gat_heads",
    "sage_aggr",
    "gin_eps",
    "gat_heads_by_layer",
    "sage_aggr_by_layer",
    "gin_eps_by_layer",
)
PER_SEED_FIELDS = (
    "candidate_rank",
    "source",
    "name",
    "search_step",
    "search_val_acc",
    "seed",
    "val_acc",
    "test_acc",
    "valid",
    "train_time_seconds",
    "operations",
    "edges",
    "lr",
    "dropout",
    "hidden_dim",
    "l2",
    "gat_heads",
    "sage_aggr",
    "gin_eps",
    "gat_heads_by_layer",
    "sage_aggr_by_layer",
    "gin_eps_by_layer",
)
AGGREGATE_FIELDS = (
    "candidate_rank",
    "source",
    "name",
    "search_step",
    "search_val_acc",
    "val_mean",
    "val_std",
    "val_min",
    "val_max",
    "test_mean",
    "test_std",
    "test_min",
    "test_max",
    "n_valid",
    "n_invalid",
    "n_seeds",
    "operations",
    "edges",
    "lr",
    "dropout",
    "hidden_dim",
    "l2",
    "gat_heads",
    "sage_aggr",
    "gin_eps",
    "gat_heads_by_layer",
    "sage_aggr_by_layer",
    "gin_eps_by_layer",
)


def setup_logger(log_dir: str, script_name: str, version: str):
    ts = datetime.now().strftime("%m%d_%H%M")
    log_subdir = os.path.join(log_dir, script_name)
    os.makedirs(log_subdir, exist_ok=True)
    log_path = os.path.join(log_subdir, f"train_{version}_{ts}.log")

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info("log file: %s", log_path)
    return logger, log_path


def save_args_json(args: argparse.Namespace, log_path: str) -> str:
    json_path = log_path.replace(".log", ".json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)
    return json_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final multi-seed evaluation of real history_final.json records"
    )
    parser.add_argument("--history_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--history_steps", type=int, nargs="+")
    parser.add_argument(
        "--deduplicate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate identical architecture and hyperparameter records (default: on).",
    )
    parser.add_argument("--sort_by", type=str, default="val_acc")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--eval_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--hp_mode", type=str, default="global4", choices=HP_MODE_CHOICES)
    parser.add_argument("--cora_root", type=str, default="/tmp/Cora")
    parser.add_argument("--output", type=str, default="results/final_eval_history_topk")
    parser.add_argument("--version", type=str, default="final_eval_history_topk")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--include_baselines", action="store_true")
    parser.add_argument("--gcnii_alpha", type=float, default=0.1)
    parser.add_argument("--gcnii_theta", type=float, default=0.5)
    return parser.parse_args(argv)


def load_cora(root: str, device: torch.device):
    pyg_root = os.path.dirname(root) if os.path.basename(root) == "Cora" else root
    dataset = Planetoid(root=pyg_root, name="Cora", transform=T.NormalizeFeatures())
    return dataset[0].to(device), dataset.num_features, dataset.num_classes


def _finite_float(value: Any, field: str, record_index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"history record {record_index} has non-numeric {field}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"history record {record_index} has non-finite {field}: {value!r}")
    return number


def _normalize_valid_record(record: dict[str, Any], record_index: int) -> dict[str, Any]:
    missing = [field for field in REQUIRED_HISTORY_FIELDS if field not in record]
    if "l2" not in record and "weight_decay" not in record:
        missing.append("l2 or weight_decay")
    if missing:
        raise ValueError(
            f"valid history record {record_index} is missing required fields: "
            + ", ".join(missing)
        )
    if not isinstance(record["operations"], list):
        raise ValueError(f"history record {record_index} operations must be a list")
    if not isinstance(record["edges"], list):
        raise ValueError(f"history record {record_index} edges must be a list")

    edges: list[list[int]] = []
    for edge in record["edges"]:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(
                f"history record {record_index} contains an invalid edge: {edge!r}"
            )
        try:
            edges.append([int(edge[0]), int(edge[1])])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"history record {record_index} contains a non-integer edge: {edge!r}"
            ) from exc

    normalized = dict(record)
    normalized["operations"] = [str(operation) for operation in record["operations"]]
    normalized["edges"] = edges
    normalized["val_acc"] = _finite_float(record["val_acc"], "val_acc", record_index)
    normalized["lr"] = _finite_float(record["lr"], "lr", record_index)
    normalized["dropout"] = _finite_float(record["dropout"], "dropout", record_index)
    try:
        normalized["hidden_dim"] = int(record["hidden_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"history record {record_index} has invalid hidden_dim: {record['hidden_dim']!r}"
        ) from exc
    if normalized["hidden_dim"] <= 0:
        raise ValueError(f"history record {record_index} hidden_dim must be positive")
    l2_value = record["l2"] if "l2" in record else record["weight_decay"]
    normalized["l2"] = _finite_float(l2_value, "l2", record_index)
    normalized["gat_heads"] = int(record.get("gat_heads", DEFAULT_GAT_HEADS))
    normalized["sage_aggr"] = str(record.get("sage_aggr", DEFAULT_SAGE_AGGR))
    normalized["gin_eps"] = _finite_float(
        record.get("gin_eps", DEFAULT_GIN_EPS), "gin_eps", record_index
    )
    normalized.setdefault("gat_heads_by_layer", None)
    normalized.setdefault("sage_aggr_by_layer", None)
    normalized.setdefault("gin_eps_by_layer", None)
    normalized.setdefault("condition_mask", None)
    normalized.setdefault("condition_mask_vector", None)
    normalized.setdefault("z_search", None)
    normalized.setdefault("gp_pred_mean", None)
    normalized.setdefault("gp_pred_std", None)
    normalized.setdefault("gp_abs_error", None)
    normalized.setdefault("gp_stage", None)
    return normalized


def load_history_records(history_path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(history_path):
        raise FileNotFoundError(f"history file not found: {history_path}")
    with open(history_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"history file must contain a JSON list: {history_path}")

    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(payload):
        if not isinstance(raw_record, dict):
            raise ValueError(f"history record {index} must be a JSON object")
        if "valid" not in raw_record or not isinstance(raw_record["valid"], bool):
            raise ValueError(f"history record {index} must contain a boolean valid field")
        record = dict(raw_record)
        record["_history_index"] = index
        if raw_record["valid"]:
            record = _normalize_valid_record(record, index)
            record["_history_index"] = index
        records.append(record)
    return records


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def record_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_freeze(record.get(field)) for field in SIGNATURE_FIELDS)


def _sort_value(record: dict[str, Any], sort_by: str) -> float:
    if sort_by not in record:
        index = record.get("_history_index", "?")
        raise ValueError(f"valid history record {index} has no sort field {sort_by!r}")
    return _finite_float(record[sort_by], sort_by, int(record.get("_history_index", -1)))


def select_history_records(
    records: Sequence[dict[str, Any]],
    top_k: int,
    history_steps: Sequence[int] | None = None,
    sort_by: str = "val_acc",
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    valid_records = [record for record in records if record.get("valid") is True]

    if history_steps is not None:
        by_step: dict[int, list[dict[str, Any]]] = {}
        for record in valid_records:
            if record.get("step") is not None:
                by_step.setdefault(int(record["step"]), []).append(record)
        available_steps = {
            int(record["step"])
            for record in records
            if isinstance(record.get("step"), (int, float))
        }
        missing_steps = [int(step) for step in history_steps if int(step) not in available_steps]
        if missing_steps:
            raise ValueError(f"requested history steps not found: {missing_steps}")
        ordered = [
            record
            for step in history_steps
            for record in by_step.get(int(step), [])
        ]
    else:
        ordered = sorted(valid_records, key=lambda row: _sort_value(row, sort_by), reverse=True)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in ordered:
        signature = record_signature(record)
        if deduplicate and signature in seen:
            continue
        seen.add(signature)
        selected_record = {key: value for key, value in record.items() if key != "_history_index"}
        selected_record.update(
            {
                "search_rank": len(selected) + 1,
                "search_step": record.get("step"),
                "search_val_acc": float(record["val_acc"]),
                "search_gp_pred_mean": record.get("gp_pred_mean"),
                "search_gp_pred_std": record.get("gp_pred_std"),
                "search_gp_abs_error": record.get("gp_abs_error"),
                "search_gp_stage": record.get("gp_stage"),
                "source": "history",
            }
        )
        selected.append(selected_record)
        if len(selected) >= top_k:
            break
    return selected


def _candidate_from_history(record: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(record)
    candidate["candidate_rank"] = int(record["search_rank"])
    candidate["name"] = (
        f"history_rank_{candidate['candidate_rank']:03d}_step_{record.get('search_step')}"
    )
    candidate["source"] = "history"
    return candidate


def _baseline_candidate(
    name: str,
    operations: list[str],
    lr: float,
    dropout: float,
    hidden_dim: int,
    l2: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "source": "baseline",
        "search_rank": None,
        "search_step": None,
        "search_val_acc": None,
        "operations": operations,
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        "lr": float(lr),
        "dropout": float(dropout),
        "hidden_dim": int(hidden_dim),
        "l2": float(l2),
        "gat_heads": DEFAULT_GAT_HEADS,
        "sage_aggr": DEFAULT_SAGE_AGGR,
        "gin_eps": DEFAULT_GIN_EPS,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
    }


def build_baseline_candidates(start_rank: int = 1) -> list[dict[str, Any]]:
    baselines = [
        _baseline_candidate(
            "BASE_2xGCN_defaultHP",
            ["GCNConv", "GCNConv", "Identity"],
            DEFAULT_LR,
            DEFAULT_DROPOUT,
            DEFAULT_HIDDEN_DIM,
            DEFAULT_L2,
        ),
        _baseline_candidate(
            "BASE_1xGCN_defaultHP",
            ["GCNConv", "Identity", "Identity"],
            DEFAULT_LR,
            DEFAULT_DROPOUT,
            DEFAULT_HIDDEN_DIM,
            DEFAULT_L2,
        ),
        _baseline_candidate(
            "BASE_2xGCN_officialHP",
            ["GCNConv", "GCNConv", "Identity"],
            0.01,
            0.5,
            16,
            5e-4,
        ),
    ]
    for offset, baseline in enumerate(baselines):
        baseline["candidate_rank"] = start_rank + offset
    return baselines


def _stats(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def evaluate_candidate(
    candidate: dict[str, Any],
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = {
        "operations": list(candidate["operations"]),
        "edges": [tuple(edge) for edge in candidate["edges"]],
        "effective_layers": sum(
            1 for operation in candidate["operations"] if operation != "Identity"
        ),
    }
    per_seed: list[dict[str, Any]] = []
    for index in range(int(args.n_seeds)):
        seed = int(args.seed_start) + index
        started = time.monotonic()
        try:
            val_acc, is_valid, test_acc = train_and_eval_arch(
                config=config,
                data=data,
                in_ch=in_ch,
                out_ch=out_ch,
                lr=float(candidate["lr"]),
                dropout=float(candidate["dropout"]),
                hidden_dim=int(candidate["hidden_dim"]),
                weight_decay=float(candidate["l2"]),
                gcnii_alpha=float(args.gcnii_alpha),
                gcnii_theta=float(args.gcnii_theta),
                gat_heads=int(candidate.get("gat_heads", DEFAULT_GAT_HEADS)),
                sage_aggr=str(candidate.get("sage_aggr", DEFAULT_SAGE_AGGR)),
                gin_eps=float(candidate.get("gin_eps", DEFAULT_GIN_EPS)),
                gat_heads_by_layer=candidate.get("gat_heads_by_layer"),
                sage_aggr_by_layer=candidate.get("sage_aggr_by_layer"),
                gin_eps_by_layer=candidate.get("gin_eps_by_layer"),
                device=device,
                max_epochs=int(args.eval_epochs),
                patience=int(args.patience),
                seed=seed,
                track_test=True,
            )
        except Exception:
            logger.exception(
                "candidate rank=%s seed=%s raised during final evaluation; recording invalid",
                candidate["candidate_rank"],
                seed,
            )
            val_acc, is_valid, test_acc = 0.0, False, 0.0
        elapsed = time.monotonic() - started
        row = {
            "candidate_rank": int(candidate["candidate_rank"]),
            "source": candidate["source"],
            "name": candidate["name"],
            "search_step": candidate.get("search_step"),
            "search_val_acc": candidate.get("search_val_acc"),
            "seed": seed,
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "valid": bool(is_valid),
            "train_time_seconds": float(elapsed),
            "operations": list(candidate["operations"]),
            "edges": [list(edge) for edge in candidate["edges"]],
            "lr": float(candidate["lr"]),
            "dropout": float(candidate["dropout"]),
            "hidden_dim": int(candidate["hidden_dim"]),
            "l2": float(candidate["l2"]),
            "gat_heads": int(candidate.get("gat_heads", DEFAULT_GAT_HEADS)),
            "sage_aggr": str(candidate.get("sage_aggr", DEFAULT_SAGE_AGGR)),
            "gin_eps": float(candidate.get("gin_eps", DEFAULT_GIN_EPS)),
            "gat_heads_by_layer": candidate.get("gat_heads_by_layer"),
            "sage_aggr_by_layer": candidate.get("sage_aggr_by_layer"),
            "gin_eps_by_layer": candidate.get("gin_eps_by_layer"),
        }
        per_seed.append(row)
        logger.info(
            "  seed=%d val=%.4f test=%.4f %s time=%.2fs",
            seed,
            row["val_acc"],
            row["test_acc"],
            "[valid]" if row["valid"] else "[invalid]",
            elapsed,
        )

    valid_rows = [row for row in per_seed if row["valid"]]
    val_stats = _stats(row["val_acc"] for row in valid_rows)
    test_stats = _stats(row["test_acc"] for row in valid_rows)
    aggregate = {
        "candidate_rank": int(candidate["candidate_rank"]),
        "source": candidate["source"],
        "name": candidate["name"],
        "search_step": candidate.get("search_step"),
        "search_val_acc": candidate.get("search_val_acc"),
        "val_mean": val_stats["mean"],
        "val_std": val_stats["std"],
        "val_min": val_stats["min"],
        "val_max": val_stats["max"],
        "test_mean": test_stats["mean"],
        "test_std": test_stats["std"],
        "test_min": test_stats["min"],
        "test_max": test_stats["max"],
        "n_valid": len(valid_rows),
        "n_invalid": len(per_seed) - len(valid_rows),
        "n_seeds": len(per_seed),
        "operations": list(candidate["operations"]),
        "edges": [list(edge) for edge in candidate["edges"]],
        "lr": float(candidate["lr"]),
        "dropout": float(candidate["dropout"]),
        "hidden_dim": int(candidate["hidden_dim"]),
        "l2": float(candidate["l2"]),
        "gat_heads": int(candidate.get("gat_heads", DEFAULT_GAT_HEADS)),
        "sage_aggr": str(candidate.get("sage_aggr", DEFAULT_SAGE_AGGR)),
        "gin_eps": float(candidate.get("gin_eps", DEFAULT_GIN_EPS)),
        "gat_heads_by_layer": candidate.get("gat_heads_by_layer"),
        "sage_aggr_by_layer": candidate.get("sage_aggr_by_layer"),
        "gin_eps_by_layer": candidate.get("gin_eps_by_layer"),
        "per_seed": per_seed,
    }
    return aggregate, per_seed


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: str, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _best_result(
    results: Sequence[dict[str, Any]], field: str, source: str | None = None
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in results
        if row.get(field) is not None and (source is None or row.get("source") == source)
    ]
    return max(eligible, key=lambda row: float(row[field])) if eligible else None


def _short_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {field: result.get(field) for field in AGGREGATE_FIELDS}


def _format_metric(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "n/a"
    return f"{mean:.4f}+/-{std:.4f}"


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.n_seeds <= 0:
        raise ValueError("--n_seeds must be positive")
    if args.eval_epochs <= 0:
        raise ValueError("--eval_epochs must be positive")
    if args.patience <= 0:
        raise ValueError("--patience must be positive")
    args.hp_mode = validate_hp_mode(args.hp_mode)

    os.makedirs(args.output, exist_ok=True)
    logger, log_path = setup_logger(args.log_dir, "final_eval", args.version)
    save_args_json(args, log_path)

    logger.info("history path: %s", args.history_path)
    records = load_history_records(args.history_path)
    selected = select_history_records(
        records,
        top_k=args.top_k,
        history_steps=args.history_steps,
        sort_by=args.sort_by,
        deduplicate=args.deduplicate,
    )
    if not selected:
        raise ValueError("no valid history records were selected for final evaluation")

    logger.info("Selected top-k history records:")
    for record in selected:
        logger.info(
            "rank=%d step=%s search_val=%.4f ops=%s edges=%s "
            "hp={lr=%g, dropout=%g, hidden_dim=%d, l2=%g, gat_heads=%d, "
            "sage_aggr=%s, gin_eps=%g}",
            record["search_rank"],
            record.get("search_step"),
            record["search_val_acc"],
            record["operations"],
            record["edges"],
            record["lr"],
            record["dropout"],
            record["hidden_dim"],
            record["l2"],
            record["gat_heads"],
            record["sage_aggr"],
            record["gin_eps"],
        )
    logger.info("selected steps: %s", [record.get("search_step") for record in selected])

    selected_path = os.path.join(
        args.output, f"selected_history_records_{args.hp_mode}.json"
    )
    with open(selected_path, "w", encoding="utf-8") as file:
        json.dump(selected, file, indent=2, ensure_ascii=False)

    candidates = [_candidate_from_history(record) for record in selected]
    if args.include_baselines:
        candidates.extend(build_baseline_candidates(start_rank=len(candidates) + 1))

    torch.manual_seed(args.seed_start)
    np.random.seed(args.seed_start)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(
        "device=%s hp_mode=%s n_seeds=%d seeds=%d..%d",
        device,
        args.hp_mode,
        args.n_seeds,
        args.seed_start,
        args.seed_start + args.n_seeds - 1,
    )
    data, in_ch, out_ch = load_cora(args.cora_root, device)
    logger.info("Cora: %d features, %d classes", in_ch, out_ch)

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    per_seed_results: list[dict[str, Any]] = []
    for candidate in candidates:
        logger.info("=" * 66)
        logger.info(
            "candidate rank=%d source=%s name=%s search_step=%s search_val=%s",
            candidate["candidate_rank"],
            candidate["source"],
            candidate["name"],
            candidate.get("search_step"),
            candidate.get("search_val_acc"),
        )
        logger.info(
            "  operations=%s edges=%s hp={lr=%g, dropout=%g, hidden_dim=%d, l2=%g, "
            "gat_heads=%d, sage_aggr=%s, gin_eps=%g}",
            candidate["operations"],
            candidate["edges"],
            candidate["lr"],
            candidate["dropout"],
            candidate["hidden_dim"],
            candidate["l2"],
            candidate["gat_heads"],
            candidate["sage_aggr"],
            candidate["gin_eps"],
        )
        aggregate, seed_rows = evaluate_candidate(
            candidate, data, in_ch, out_ch, args, device, logger
        )
        results.append(aggregate)
        per_seed_results.extend(seed_rows)
        logger.info(
            "  final val=%s test=%s valid=%d invalid=%d",
            _format_metric(aggregate["val_mean"], aggregate["val_std"]),
            _format_metric(aggregate["test_mean"], aggregate["test_std"]),
            aggregate["n_valid"],
            aggregate["n_invalid"],
        )

    results_json_path = os.path.join(args.output, f"final_results_{args.hp_mode}.json")
    results_csv_path = os.path.join(args.output, f"final_results_{args.hp_mode}.csv")
    per_seed_csv_path = os.path.join(
        args.output, f"final_results_per_seed_{args.hp_mode}.csv"
    )
    with open(results_json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    write_csv(results_csv_path, results, AGGREGATE_FIELDS)
    write_csv(per_seed_csv_path, per_seed_results, PER_SEED_FIELDS)

    best_by_test = _best_result(results, "test_mean")
    best_by_val = _best_result(results, "val_mean")
    best_by_search = _best_result(results, "search_val_acc", source="history")
    final_history_best = _best_result(results, "test_mean", source="history")
    search_and_final_match = bool(
        best_by_search is not None
        and final_history_best is not None
        and best_by_search["candidate_rank"] == final_history_best["candidate_rank"]
    )
    summary = {
        "history_path": args.history_path,
        "hp_mode": args.hp_mode,
        "top_k": args.top_k,
        "selected_count": len(selected),
        "selected_steps": [record.get("search_step") for record in selected],
        "deduplicate": args.deduplicate,
        "sort_by": args.sort_by,
        "n_seeds": args.n_seeds,
        "seed_start": args.seed_start,
        "eval_epochs": args.eval_epochs,
        "patience": args.patience,
        "include_baselines": args.include_baselines,
        "best_by_test_mean": _short_result(best_by_test),
        "best_by_val_mean": _short_result(best_by_val),
        "best_by_search_val_acc": _short_result(best_by_search),
        "search_best_matches_final_history_test_best": search_and_final_match,
    }
    summary_path = os.path.join(
        args.output, f"final_eval_summary_{args.hp_mode}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    if best_by_test is not None:
        logger.info(
            "Final best by test_mean: rank=%d step=%s search_val=%s test=%s",
            best_by_test["candidate_rank"],
            best_by_test.get("search_step"),
            best_by_test.get("search_val_acc"),
            _format_metric(best_by_test["test_mean"], best_by_test["test_std"]),
        )
    if best_by_val is not None:
        logger.info(
            "Final best by val_mean: rank=%d step=%s search_val=%s val=%s",
            best_by_val["candidate_rank"],
            best_by_val.get("search_step"),
            best_by_val.get("search_val_acc"),
            _format_metric(best_by_val["val_mean"], best_by_val["val_std"]),
        )
    logger.info(
        "Search-stage best and final history test best match: %s",
        search_and_final_match,
    )
    logger.info(
        "saved: %s, %s, %s, %s, %s",
        results_json_path,
        results_csv_path,
        per_seed_csv_path,
        selected_path,
        summary_path,
    )
    logger.info("run complete in %.1f minutes", (time.monotonic() - started) / 60.0)


if __name__ == "__main__":
    main()
