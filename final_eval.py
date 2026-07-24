"""Seed-fair, independently retrained final evaluation of NAS history candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
import warnings
from datetime import datetime
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_utils import (  # noqa: E402
    DEFAULT_GAT_HEADS,
    DEFAULT_GIN_EPS,
    DEFAULT_SAGE_AGGR,
    stable_seed,
    train_and_eval_arch,
)
from hp_modes import validate_hp_mode  # noqa: E402
from surrogate.checkpoint_io import atomic_json_dump  # noqa: E402


HP_MODE_CHOICES = ("global4", "hybrid_cond7", "layer_cond19")
DEFAULT_LR = 1e-3
DEFAULT_DROPOUT = 0.5
DEFAULT_HIDDEN_DIM = 64
DEFAULT_L2 = 5e-4

FINAL_EVALUATION_SEED_SCHEME = (
    "final_base_candidate_fingerprint_replicate_v1"
)
LEGACY_SEQUENTIAL_SEED_SCHEME = "legacy_seed_start_plus_index"
PROGRESS_FORMAT_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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
IDENTITY_FIELDS = (
    "method_label",
    "search_seed",
    "candidate_rank",
    "search_step",
    "search_val_acc",
    "candidate_fingerprint",
    "replicate_id",
    "final_base_seed",
    "final_eval_seed",
    "final_evaluation_seed_scheme",
)
ARCHITECTURE_HP_FIELDS = (
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
    "gcnii_alpha",
    "gcnii_theta",
    "hp_mode",
)
PER_REPLICATE_FIELDS = (
    *IDENTITY_FIELDS,
    "source",
    "name",
    "seed",
    "val_acc",
    "test_acc",
    "valid",
    "best_epoch",
    "epochs_ran",
    "train_time_seconds",
    *ARCHITECTURE_HP_FIELDS,
)
# Backward-compatible public name used by existing callers/tests.
PER_SEED_FIELDS = PER_REPLICATE_FIELDS
AGGREGATE_FIELDS = (
    "method_label",
    "search_seed",
    "candidate_rank",
    "source",
    "name",
    "search_step",
    "search_val_acc",
    "candidate_fingerprint",
    "final_base_seed",
    "final_evaluation_seed_scheme",
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
    "n_replicates",
    "n_seeds",
    *ARCHITECTURE_HP_FIELDS,
)


def setup_logger(log_dir: str, script_name: str, version: str):
    ts = datetime.now().strftime("%m%d_%H%M%S")
    log_subdir = os.path.join(log_dir, script_name)
    os.makedirs(log_subdir, exist_ok=True)
    log_path = os.path.join(log_subdir, f"train_{version}_{ts}.log")

    logger = logging.getLogger(f"{script_name}.{version}.{ts}")
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
        json.dump(
            {key: value for key, value in vars(args).items() if not key.startswith("_")},
            file,
            indent=2,
            ensure_ascii=False,
        )
    return json_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed-fair final validation/test of top-k history candidates"
    )
    parser.add_argument("--history_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--history_steps", type=int, nargs="+")
    parser.add_argument(
        "--deduplicate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate identical architecture and hyperparameter records.",
    )
    parser.add_argument("--sort_by", type=str, default="val_acc")

    parser.add_argument("--final_base_seed", type=int)
    parser.add_argument("--n_replicates", type=int, default=10)
    parser.add_argument("--method_label", type=str)
    parser.add_argument("--search_seed", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--legacy_sequential_seeds",
        action="store_true",
        help=(
            "Explicitly use deprecated seed_start + replicate_index seeding. "
            "Never use this for formal evaluation."
        ),
    )
    parser.add_argument("--n_seeds", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--seed_start", type=int, default=0, help=argparse.SUPPRESS)

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

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    legacy_options_explicit = any(
        option in raw_argv for option in ("--seed_start", "--n_seeds")
    )
    if legacy_options_explicit and not args.legacy_sequential_seeds:
        parser.error(
            "--seed_start/--n_seeds require --legacy_sequential_seeds; "
            "they are never interpreted as --final_base_seed/--n_replicates"
        )
    args._legacy_seed_options_explicit = legacy_options_explicit
    return args


def load_cora(root: str, device: torch.device):
    pyg_root = os.path.dirname(root) if os.path.basename(root) == "Cora" else root
    dataset = Planetoid(root=pyg_root, name="Cora", transform=T.NormalizeFeatures())
    return dataset[0].to(device), dataset.num_features, dataset.num_classes


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_evaluation_seed(
    final_base_seed: int,
    candidate_fingerprint: str,
    replicate_id: int,
) -> int:
    """Derive the formal final-evaluation seed from exactly three variable inputs."""

    if not isinstance(candidate_fingerprint, str) or not SHA256_RE.fullmatch(
        candidate_fingerprint
    ):
        raise ValueError(
            "candidate_fingerprint must be a lowercase SHA-256 hexadecimal string"
        )
    replicate_id = int(replicate_id)
    if replicate_id < 0:
        raise ValueError("replicate_id must be non-negative")
    return stable_seed(
        int(final_base_seed),
        "final_evaluation",
        candidate_fingerprint,
        replicate_id,
    )


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
    normalized["operations"] = [str(item) for item in record["operations"]]
    normalized["edges"] = edges
    normalized["val_acc"] = _finite_float(record["val_acc"], "val_acc", record_index)
    normalized["lr"] = _finite_float(record["lr"], "lr", record_index)
    normalized["dropout"] = _finite_float(record["dropout"], "dropout", record_index)
    try:
        normalized["hidden_dim"] = int(record["hidden_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"history record {record_index} has invalid hidden_dim: "
            f"{record['hidden_dim']!r}"
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
    for field in (
        "gat_heads_by_layer",
        "sage_aggr_by_layer",
        "gin_eps_by_layer",
        "condition_mask",
        "condition_mask_vector",
        "z_search",
        "gp_pred_mean",
        "gp_pred_std",
        "gp_abs_error",
        "gp_stage",
    ):
        normalized.setdefault(field, None)
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


def _existing_candidate_fingerprint(z_search: Any) -> str:
    # Import lazily so dry, dependency-stubbed history tests stay lightweight.
    from initialization_wgmm_ted import candidate_fingerprint

    return candidate_fingerprint(z_search)


def validate_formal_history(
    records: Sequence[dict[str, Any]], expected_search_seed: int
) -> None:
    valid_records = [record for record in records if record.get("valid") is True]
    if not valid_records:
        raise ValueError("history contains no valid candidates")
    for record in valid_records:
        index = int(record.get("_history_index", -1))
        fingerprint = record.get("candidate_fingerprint")
        if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
            raise ValueError(
                f"valid history record {index} has a missing or invalid "
                "candidate_fingerprint; a lowercase SHA-256 hex string is required"
            )
        if record.get("z_search") is not None:
            recomputed = _existing_candidate_fingerprint(record["z_search"])
            if fingerprint != recomputed:
                raise ValueError(
                    f"history record {index} candidate_fingerprint does not match "
                    f"z_search: expected {recomputed}, got {fingerprint}"
                )
        if "search_seed" not in record:
            raise ValueError(f"valid history record {index} is missing search_seed")
        try:
            record_search_seed = int(record["search_seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"valid history record {index} has invalid search_seed: "
                f"{record['search_seed']!r}"
            ) from exc
        if record_search_seed != int(expected_search_seed):
            raise ValueError(
                f"history record {index} search_seed mismatch: expected "
                f"{expected_search_seed}, got {record_search_seed}"
            )


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


def _tie_step(record: dict[str, Any]) -> tuple[int, int]:
    step = record.get("step")
    try:
        return (0, int(step))
    except (TypeError, ValueError):
        return (1, int(record.get("_history_index", 0)))


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
            record for step in history_steps for record in by_step.get(int(step), [])
        ]
    else:
        ordered = sorted(
            valid_records,
            key=lambda row: (
                -_sort_value(row, sort_by),
                _tie_step(row),
                str(row.get("candidate_fingerprint", "")),
                int(row.get("_history_index", 0)),
            ),
        )

    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in ordered:
        signature = record_signature(record)
        if deduplicate and signature in seen:
            continue
        seen.add(signature)
        selected_record = {
            key: value for key, value in record.items() if key != "_history_index"
        }
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
        f"history_rank_{candidate['candidate_rank']:03d}_step_"
        f"{record.get('search_step')}"
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
        "candidate_fingerprint": None,
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


def _is_formal(args: argparse.Namespace) -> bool:
    # Namespace objects from old API users do not have n_replicates and remain legacy.
    return hasattr(args, "n_replicates") and not bool(
        getattr(args, "legacy_sequential_seeds", False)
    )


def _replicate_count(args: argparse.Namespace) -> int:
    return int(args.n_replicates if _is_formal(args) else args.n_seeds)


def _seed_for_replicate(
    args: argparse.Namespace, candidate: dict[str, Any], replicate_id: int
) -> int:
    if _is_formal(args):
        return final_evaluation_seed(
            args.final_base_seed,
            candidate["candidate_fingerprint"],
            replicate_id,
        )
    return int(args.seed_start) + replicate_id


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


def _common_result_fields(
    candidate: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    formal = _is_formal(args)
    return {
        "method_label": getattr(args, "method_label", None),
        "search_seed": getattr(args, "search_seed", None),
        "candidate_rank": int(candidate["candidate_rank"]),
        "source": candidate["source"],
        "name": candidate["name"],
        "search_step": candidate.get("search_step"),
        "search_val_acc": candidate.get("search_val_acc"),
        "candidate_fingerprint": candidate.get("candidate_fingerprint"),
        "final_base_seed": getattr(args, "final_base_seed", None) if formal else None,
        "final_evaluation_seed_scheme": (
            FINAL_EVALUATION_SEED_SCHEME
            if formal
            else LEGACY_SEQUENTIAL_SEED_SCHEME
        ),
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
        "gcnii_alpha": float(args.gcnii_alpha),
        "gcnii_theta": float(args.gcnii_theta),
        "hp_mode": getattr(args, "hp_mode", None),
    }


def _manifest_row(
    candidate: dict[str, Any], args: argparse.Namespace, replicate_id: int
) -> dict[str, Any]:
    seed = _seed_for_replicate(args, candidate, replicate_id)
    return {
        **_common_result_fields(candidate, args),
        "replicate_id": replicate_id,
        "final_eval_seed": seed,
        "seed": seed,
        "val_acc": None,
        "test_acc": None,
        "valid": None,
        "best_epoch": None,
        "epochs_ran": None,
        "train_time_seconds": None,
    }


def _unpack_training_result(result: tuple) -> tuple[float, bool, float, dict[str, Any]]:
    if len(result) == 4:
        val_acc, is_valid, test_acc, metadata = result
        metadata = metadata or {}
    elif len(result) == 3:
        # Compatibility with test doubles and older external implementations.
        val_acc, is_valid, test_acc = result
        metadata = {}
    else:
        raise ValueError(
            "train_and_eval_arch returned an unexpected result; expected "
            "(val_acc, valid, test_acc, metadata)"
        )
    return float(val_acc), bool(is_valid), float(test_acc), dict(metadata)


def evaluate_candidate(
    candidate: dict[str, Any],
    data,
    in_ch: int,
    out_ch: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
    existing_rows: Sequence[dict[str, Any]] | None = None,
    on_replicate_complete: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = {
        "operations": list(candidate["operations"]),
        "edges": [tuple(edge) for edge in candidate["edges"]],
        "effective_layers": sum(
            operation != "Identity" for operation in candidate["operations"]
        ),
    }
    existing_by_id = {
        int(row["replicate_id"]): dict(row) for row in (existing_rows or [])
    }
    per_replicate: list[dict[str, Any]] = []
    for replicate_id in range(_replicate_count(args)):
        seed = _seed_for_replicate(args, candidate, replicate_id)
        if replicate_id in existing_by_id:
            row = existing_by_id[replicate_id]
            if int(row.get("final_eval_seed", -1)) != seed:
                raise ValueError(
                    "resume record final_eval_seed mismatch for candidate "
                    f"{candidate['candidate_rank']} replicate {replicate_id}"
                )
            per_replicate.append(row)
            logger.info(
                "  replicate_id=%d final_eval_seed=%d [resumed]", replicate_id, seed
            )
            continue

        started = time.monotonic()
        metadata: dict[str, Any] = {}
        try:
            result = train_and_eval_arch(
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
                return_metadata=True,
            )
            val_acc, is_valid, test_acc, metadata = _unpack_training_result(result)
        except Exception:
            logger.exception(
                "candidate rank=%s replicate_id=%s final_eval_seed=%s raised; "
                "recording invalid without changing its seed",
                candidate["candidate_rank"],
                replicate_id,
                seed,
            )
            val_acc, is_valid, test_acc = 0.0, False, 0.0
        elapsed = time.monotonic() - started
        row = {
            **_manifest_row(candidate, args, replicate_id),
            "val_acc": val_acc,
            "test_acc": test_acc,
            "valid": is_valid,
            "best_epoch": metadata.get("best_epoch"),
            "epochs_ran": metadata.get("epochs_ran", 0),
            "train_time_seconds": float(elapsed),
            "completed": True,
        }
        per_replicate.append(row)
        if on_replicate_complete is not None:
            on_replicate_complete(row)
        logger.info(
            "  replicate_id=%d final_eval_seed=%d val=%.4f test=%.4f %s "
            "best_epoch=%s epochs_ran=%s time=%.2fs",
            replicate_id,
            seed,
            row["val_acc"],
            row["test_acc"],
            "[valid]" if row["valid"] else "[invalid]",
            row["best_epoch"],
            row["epochs_ran"],
            elapsed,
        )

    per_replicate.sort(key=lambda row: int(row["replicate_id"]))
    valid_rows = [row for row in per_replicate if row["valid"]]
    val_stats = _stats(row["val_acc"] for row in valid_rows)
    test_stats = _stats(row["test_acc"] for row in valid_rows)
    aggregate = {
        **_common_result_fields(candidate, args),
        "val_mean": val_stats["mean"],
        "val_std": val_stats["std"],
        "val_min": val_stats["min"],
        "val_max": val_stats["max"],
        "test_mean": test_stats["mean"],
        "test_std": test_stats["std"],
        "test_min": test_stats["min"],
        "test_max": test_stats["max"],
        "n_valid": len(valid_rows),
        "n_invalid": len(per_replicate) - len(valid_rows),
        "n_replicates": len(per_replicate),
        "n_seeds": len(per_replicate),
        "per_replicate": per_replicate,
        "per_seed": per_replicate,
    }
    return aggregate, per_replicate


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: str, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(fields), extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: _csv_value(row.get(field)) for field in fields}
                )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _best_result(
    results: Sequence[dict[str, Any]], field: str, source: str | None = None
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in results
        if row.get(field) is not None
        and (source is None or row.get("source") == source)
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            -float(row[field]),
            int(row.get("candidate_rank", 0)),
            str(row.get("candidate_fingerprint", "")),
        ),
    )[0]


def select_by_final_validation(
    results: Sequence[dict[str, Any]],
    n_replicates: int,
    require_all_valid: bool = True,
) -> dict[str, Any] | None:
    """Select using final validation only; test metrics never enter this ordering."""

    eligible = [
        row
        for row in results
        if row.get("val_mean") is not None
        and (
            not require_all_valid
            or int(row.get("n_valid", -1)) == int(n_replicates)
        )
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            -float(row["val_mean"]),
            int(row.get("candidate_rank", 0)),
            str(row.get("candidate_fingerprint", "")),
        ),
    )[0]


def _short_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {field: result.get(field) for field in AGGREGATE_FIELDS}


def _format_metric(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "n/a"
    return f"{mean:.4f}+/-{std:.4f}"


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("top_k", "eval_epochs", "patience"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    args.hp_mode = validate_hp_mode(args.hp_mode)
    if _is_formal(args):
        if args.final_base_seed is None:
            raise ValueError(
                "--final_base_seed must be explicitly provided in formal mode"
            )
        if not isinstance(args.method_label, str) or not args.method_label.strip():
            raise ValueError("--method_label must be explicitly provided in formal mode")
        if args.search_seed is None:
            raise ValueError("--search_seed must be explicitly provided in formal mode")
        if int(args.n_replicates) <= 0:
            raise ValueError("--n_replicates must be positive")
        if args.sort_by != "val_acc":
            raise ValueError("formal mode requires --sort_by val_acc")
        if args.history_steps is not None:
            raise ValueError("formal mode selects top-k independently; --history_steps is legacy-only")
        if args.include_baselines:
            raise ValueError(
                "--include_baselines is not supported in formal fingerprint-seed mode"
            )
    elif int(args.n_seeds) <= 0:
        raise ValueError("--n_seeds must be positive")


def _formal_config(
    args: argparse.Namespace, history_digest: str
) -> dict[str, Any]:
    return {
        "format_version": PROGRESS_FORMAT_VERSION,
        "history_path": os.path.abspath(args.history_path),
        "history_sha256": history_digest,
        "method_label": args.method_label,
        "search_seed": int(args.search_seed),
        "final_base_seed": int(args.final_base_seed),
        "top_k": int(args.top_k),
        "n_replicates": int(args.n_replicates),
        "eval_epochs": int(args.eval_epochs),
        "patience": int(args.patience),
        "hp_mode": args.hp_mode,
        "cora_root": os.path.abspath(args.cora_root),
        "deduplicate": bool(args.deduplicate),
        "sort_by": args.sort_by,
        "seed_scheme": FINAL_EVALUATION_SEED_SCHEME,
        "gcnii_alpha": float(args.gcnii_alpha),
        "gcnii_theta": float(args.gcnii_theta),
    }


def _load_or_create_formal_state(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    os.makedirs(args.output, exist_ok=True)
    config_path = os.path.join(args.output, "final_eval_config.json")
    progress_path = os.path.join(args.output, "final_eval_progress.json")
    formal_artifacts = (
        config_path,
        progress_path,
        os.path.join(args.output, f"final_results_{args.hp_mode}.json"),
        os.path.join(args.output, f"final_results_{args.hp_mode}.csv"),
        os.path.join(args.output, f"final_results_per_seed_{args.hp_mode}.csv"),
        os.path.join(args.output, f"final_eval_summary_{args.hp_mode}.json"),
        os.path.join(args.output, f"selected_history_records_{args.hp_mode}.json"),
        os.path.join(args.output, f"seed_manifest_{args.hp_mode}.json"),
        os.path.join(args.output, f"seed_manifest_{args.hp_mode}.csv"),
        os.path.join(args.output, f"dry_run_summary_{args.hp_mode}.json"),
    )
    existing_artifacts = [path for path in formal_artifacts if os.path.exists(path)]

    if not args.resume:
        if existing_artifacts:
            raise FileExistsError(
                "formal output already exists; refuse to overwrite without --resume: "
                + ", ".join(existing_artifacts)
            )
        atomic_json_dump(config, config_path)
        return progress_path, []

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"--resume requires existing formal config: {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        saved_config = json.load(handle)
    if saved_config != config:
        differing = sorted(
            key
            for key in set(saved_config) | set(config)
            if saved_config.get(key) != config.get(key)
        )
        raise ValueError(
            "resume configuration mismatch for fields: " + ", ".join(differing)
        )
    if not os.path.exists(progress_path):
        return progress_path, []
    with open(progress_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != PROGRESS_FORMAT_VERSION
        or payload.get("config") != config
        or not isinstance(payload.get("records"), list)
    ):
        raise ValueError("resume progress file is malformed or has mismatched config")
    return progress_path, list(payload["records"])


def _validate_resume_records(
    records: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[tuple[str, int], dict[str, Any]]:
    candidate_by_fingerprint = {
        str(candidate["candidate_fingerprint"]): candidate for candidate in candidates
    }
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict) or row.get("completed") is not True:
            raise ValueError("resume progress contains an incomplete record")
        fingerprint = row.get("candidate_fingerprint")
        if fingerprint not in candidate_by_fingerprint:
            raise ValueError("resume progress contains an unselected candidate")
        replicate_id = int(row.get("replicate_id", -1))
        if not 0 <= replicate_id < int(args.n_replicates):
            raise ValueError("resume progress contains an invalid replicate_id")
        key = (fingerprint, replicate_id)
        if key in indexed:
            raise ValueError("resume progress contains a duplicate candidate/replicate")
        candidate = candidate_by_fingerprint[fingerprint]
        expected_seed = _seed_for_replicate(args, candidate, replicate_id)
        expected_common = _common_result_fields(candidate, args)
        mismatched_fields = [
            field
            for field, expected_value in expected_common.items()
            if row.get(field) != expected_value
        ]
        if (
            int(row.get("final_eval_seed", -1)) != expected_seed
            or row.get("final_evaluation_seed_scheme")
            != FINAL_EVALUATION_SEED_SCHEME
            or mismatched_fields
        ):
            details = (
                f" fields={mismatched_fields}" if mismatched_fields else ""
            )
            raise ValueError(
                "resume record metadata/configuration mismatch" + details
            )
        if not isinstance(row.get("valid"), bool):
            raise ValueError("resume record valid must be boolean")
        for metric in ("val_acc", "test_acc", "train_time_seconds"):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"resume record {metric} must be finite")
        if float(row["train_time_seconds"]) < 0.0:
            raise ValueError("resume record train_time_seconds must be non-negative")
        indexed[key] = dict(row)
    return indexed


def _selected_manifest(
    candidates: Sequence[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for candidate in candidates:
        item = {
            **candidate,
            **_common_result_fields(candidate, args),
            "replicates": [
                {
                    "replicate_id": replicate_id,
                    "final_eval_seed": _seed_for_replicate(
                        args, candidate, replicate_id
                    ),
                }
                for replicate_id in range(_replicate_count(args))
            ],
        }
        manifest.append(item)
    return manifest


def run_final_evaluation(
    args: argparse.Namespace, logger: logging.Logger
) -> dict[str, Any]:
    _validate_args(args)
    formal = _is_formal(args)
    if not formal:
        warning = (
            "LEGACY SEQUENTIAL SEED MODE: using seed_start + replicate_index. "
            "This mode is deprecated and forbidden for formal batch evaluation."
        )
        warnings.warn(warning, FutureWarning, stacklevel=2)
        logger.warning(warning)

    logger.info("history path: %s", args.history_path)
    history_digest = sha256_file(args.history_path)
    records = load_history_records(args.history_path)
    if formal:
        validate_formal_history(records, args.search_seed)
    selected = select_history_records(
        records,
        top_k=args.top_k,
        history_steps=args.history_steps,
        sort_by=args.sort_by,
        deduplicate=args.deduplicate,
    )
    if not selected:
        raise ValueError("no valid history records were selected for final evaluation")
    if formal and len(selected) != int(args.top_k):
        raise ValueError(
            f"formal top-k requested {args.top_k} candidates but only "
            f"{len(selected)} unique valid candidates were selected"
        )
    candidates = [_candidate_from_history(record) for record in selected]
    if args.include_baselines:
        candidates.extend(build_baseline_candidates(start_rank=len(candidates) + 1))

    logger.info("selected %d independent history candidates", len(selected))
    for candidate in candidates:
        logger.info(
            "candidate_rank=%d method_label=%s search_seed=%s search_step=%s "
            "search_val_acc=%s candidate_fingerprint=%s",
            candidate["candidate_rank"],
            getattr(args, "method_label", None),
            getattr(args, "search_seed", None),
            candidate.get("search_step"),
            candidate.get("search_val_acc"),
            candidate.get("candidate_fingerprint"),
        )
        for replicate_id in range(_replicate_count(args)):
            logger.info(
                "  replicate_id=%d final_eval_seed=%d seed_scheme=%s",
                replicate_id,
                _seed_for_replicate(args, candidate, replicate_id),
                (
                    FINAL_EVALUATION_SEED_SCHEME
                    if formal
                    else LEGACY_SEQUENTIAL_SEED_SCHEME
                ),
            )

    os.makedirs(args.output, exist_ok=True)
    selected_path = os.path.join(
        args.output, f"selected_history_records_{args.hp_mode}.json"
    )
    seed_manifest_path = os.path.join(
        args.output, f"seed_manifest_{args.hp_mode}.json"
    )
    manifest_rows = [
        _manifest_row(candidate, args, replicate_id)
        for candidate in candidates
        for replicate_id in range(_replicate_count(args))
    ]

    progress_path: str | None = None
    progress_records: list[dict[str, Any]] = []
    resume_index: dict[tuple[str, int], dict[str, Any]] = {}
    config: dict[str, Any] | None = None
    if formal:
        config = _formal_config(args, history_digest)
        progress_path, progress_records = _load_or_create_formal_state(args, config)
        resume_index = _validate_resume_records(progress_records, candidates, args)
    elif not args.resume and (
        os.path.exists(selected_path) or os.path.exists(seed_manifest_path)
    ):
        raise FileExistsError(
            "legacy output exists; refuse to overwrite without --resume"
        )

    atomic_json_dump(_selected_manifest(candidates, args), selected_path)
    atomic_json_dump(manifest_rows, seed_manifest_path)
    write_csv(
        os.path.join(args.output, f"seed_manifest_{args.hp_mode}.csv"),
        manifest_rows,
        PER_REPLICATE_FIELDS,
    )

    if args.dry_run:
        dry_summary = {
            "status": "dry_run_preflight_passed",
            "history_path": os.path.abspath(args.history_path),
            "history_sha256": history_digest,
            "method_label": getattr(args, "method_label", None),
            "search_seed": getattr(args, "search_seed", None),
            "top_k": args.top_k,
            "n_replicates": _replicate_count(args),
            "final_base_seed": getattr(args, "final_base_seed", None),
            "final_evaluation_seed_scheme": (
                FINAL_EVALUATION_SEED_SCHEME
                if formal
                else LEGACY_SEQUENTIAL_SEED_SCHEME
            ),
            "selected_manifest": selected_path,
            "seed_manifest": seed_manifest_path,
        }
        atomic_json_dump(
            dry_summary,
            os.path.join(args.output, f"dry_run_summary_{args.hp_mode}.json"),
        )
        logger.info(
            "dry-run preflight passed; dataset and train_and_eval_arch were not loaded/called"
        )
        return dry_summary

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(
        "device=%s hp_mode=%s n_replicates=%d seed_scheme=%s",
        device,
        args.hp_mode,
        _replicate_count(args),
        FINAL_EVALUATION_SEED_SCHEME if formal else LEGACY_SEQUENTIAL_SEED_SCHEME,
    )
    data, in_ch, out_ch = load_cora(args.cora_root, device)
    logger.info("Cora: %d features, %d classes", in_ch, out_ch)

    def save_progress(row: dict[str, Any]) -> None:
        if not formal:
            return
        assert progress_path is not None and config is not None
        progress_records.append(dict(row))
        atomic_json_dump(
            {
                "format_version": PROGRESS_FORMAT_VERSION,
                "config": config,
                "records": progress_records,
            },
            progress_path,
        )

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    per_replicate_results: list[dict[str, Any]] = []
    for candidate in candidates:
        fingerprint = str(candidate.get("candidate_fingerprint"))
        existing = [
            row
            for (saved_fingerprint, _), row in resume_index.items()
            if saved_fingerprint == fingerprint
        ]
        aggregate, replicate_rows = evaluate_candidate(
            candidate,
            data,
            in_ch,
            out_ch,
            args,
            device,
            logger,
            existing_rows=existing,
            on_replicate_complete=save_progress,
        )
        results.append(aggregate)
        per_replicate_results.extend(replicate_rows)
        logger.info(
            "candidate_rank=%d final val=%s test=%s valid=%d invalid=%d",
            aggregate["candidate_rank"],
            _format_metric(aggregate["val_mean"], aggregate["val_std"]),
            _format_metric(aggregate["test_mean"], aggregate["test_std"]),
            aggregate["n_valid"],
            aggregate["n_invalid"],
        )

    results_json_path = os.path.join(
        args.output, f"final_results_{args.hp_mode}.json"
    )
    results_csv_path = os.path.join(
        args.output, f"final_results_{args.hp_mode}.csv"
    )
    per_replicate_csv_path = os.path.join(
        args.output, f"final_results_per_seed_{args.hp_mode}.csv"
    )
    atomic_json_dump(results, results_json_path)
    write_csv(results_csv_path, results, AGGREGATE_FIELDS)
    write_csv(per_replicate_csv_path, per_replicate_results, PER_REPLICATE_FIELDS)

    selected_by_validation = select_by_final_validation(
        results,
        _replicate_count(args),
        require_all_valid=formal,
    )
    oracle_best_by_test = _best_result(results, "test_mean")
    best_by_search = _best_result(results, "search_val_acc", source="history")
    summary = {
        "status": (
            "complete"
            if selected_by_validation is not None
            else "failed_no_fully_valid_candidate"
        ),
        "history_path": os.path.abspath(args.history_path),
        "history_sha256": history_digest,
        "method_label": getattr(args, "method_label", None),
        "search_seed": getattr(args, "search_seed", None),
        "hp_mode": args.hp_mode,
        "top_k": args.top_k,
        "selected_count": len(selected),
        "selected_steps": [record.get("search_step") for record in selected],
        "deduplicate": args.deduplicate,
        "sort_by": args.sort_by,
        "n_replicates": _replicate_count(args),
        "final_base_seed": getattr(args, "final_base_seed", None),
        "final_evaluation_seed_scheme": (
            FINAL_EVALUATION_SEED_SCHEME
            if formal
            else LEGACY_SEQUENTIAL_SEED_SCHEME
        ),
        "eval_epochs": args.eval_epochs,
        "patience": args.patience,
        "selected_by_final_validation": _short_result(selected_by_validation),
        "selected_candidate_test_mean": (
            None
            if selected_by_validation is None
            else selected_by_validation["test_mean"]
        ),
        "selected_candidate_test_std": (
            None
            if selected_by_validation is None
            else selected_by_validation["test_std"]
        ),
        "best_by_search_val_acc": _short_result(best_by_search),
        "oracle_diagnostic_not_for_selection": {
            "best_by_test_mean": _short_result(oracle_best_by_test),
            "warning": (
                "Oracle diagnostic only. test_acc was not used for candidate "
                "selection, tie-breaking, retries, or seed derivation."
            ),
        },
    }
    summary_path = os.path.join(
        args.output, f"final_eval_summary_{args.hp_mode}.json"
    )
    atomic_json_dump(summary, summary_path)

    if selected_by_validation is None:
        raise RuntimeError(
            "no candidate has n_valid == n_replicates; formal selection failed "
            "without replacing any invalid replicate seed"
        )
    logger.info(
        "selected by final validation: candidate_rank=%d val=%s; "
        "selected candidate test=%s",
        selected_by_validation["candidate_rank"],
        _format_metric(
            selected_by_validation["val_mean"], selected_by_validation["val_std"]
        ),
        _format_metric(
            selected_by_validation["test_mean"],
            selected_by_validation["test_std"],
        ),
    )
    logger.info(
        "run complete in %.1f minutes; saved %s, %s, %s, %s",
        (time.monotonic() - started) / 60.0,
        results_json_path,
        results_csv_path,
        per_replicate_csv_path,
        summary_path,
    )
    return summary


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger(args.log_dir, "final_eval", args.version)
    save_args_json(args, log_path)
    run_final_evaluation(args, logger)


if __name__ == "__main__":
    main()
