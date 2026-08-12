"""Collect NAS + HPO experiment outputs into an analysis bundle.

The collector only parses existing artifacts under results/. It does not run
training, search, final evaluation, or mutate original result files.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment_paths import posthoc_dir, validate_protocol_id


HP_MODE_SEARCH_DIM = {
    "global4": 16,
    "hybrid_cond7": 19,
    "layer_cond19": 31,
}

ACC_FIELDS = {
    "best_val_acc",
    "val_acc",
    "value",
    "val_mean",
    "test_mean",
    "search_best_val_acc",
    "best_final_test_mean",
    "final_manual_2gcn_searched_hp_test_mean",
    "final_nas_arch_default_hp_test_mean",
    "final_nas_arch_global_hp_test_mean",
    "final_nas_arch_conditional_hp_test_mean",
    "final_nas_full_hp_test_mean",
    "test_min",
    "test_max",
    "test_median",
}
STD_FIELDS = {
    "val_std",
    "test_std",
    "val_se",
    "test_se",
    "val_ci95_low",
    "val_ci95_high",
    "test_ci95_low",
    "test_ci95_high",
    "best_final_test_std",
}
DELTA_FIELDS = {
    "search_to_best_final_gap",
    "nas_full_minus_arch_default",
    "global_hp_minus_default",
    "cond_hp_minus_default",
    "full_hp_minus_global_hp",
}
SCI_FIELDS = {"lr", "best_lr", "l2", "best_l2", "weight_decay"}
FOUR_DEC_FIELDS = {"dropout", "best_dropout", "y_min", "y_max", "y_mean", "top_frac"}

SEARCH_CSV_FIELDS = [
    "run_id",
    "result_dir",
    "phase",
    "method",
    "hp_mode",
    "search_dim",
    "history_path",
    "has_history",
    "has_best_z_final",
    "has_best_z_search_final",
    "has_gmm_summary",
    "gmm_enabled",
    "gmm_n_loaded",
    "gmm_n_valid",
    "gmm_n_used_top",
    "gmm_sampled_count",
    "gmm_enqueued_count",
    "gmm_evaluated_count",
    "n_total",
    "n_valid",
    "n_invalid",
    "n_lhs_init",
    "n_gmm_init",
    "n_warm_start",
    "n_bo",
    "n_tpe",
    "best_step",
    "best_type",
    "best_val_acc",
    "best_valid",
    "best_lr",
    "best_dropout",
    "best_hidden_dim",
    "best_l2",
    "best_gat_heads",
    "best_sage_aggr",
    "best_gin_eps",
    "best_operations",
    "best_num_ops",
    "best_num_edges",
    "z_search_len",
    "notes",
]

FINAL_CSV_FIELDS = [
    "eval_id",
    "result_dir",
    "json_path",
    "hp_mode",
    "candidate_name",
    "group",
    "description",
    "val_mean",
    "val_std",
    "val_se",
    "val_ci95_low",
    "val_ci95_high",
    "test_mean",
    "test_std",
    "test_se",
    "test_ci95_low",
    "test_ci95_high",
    "test_min",
    "test_max",
    "test_median",
    "n_seeds",
    "val_list",
    "test_list",
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
    "operations",
    "num_ops",
    "num_edges",
    "source_best_z",
    "source_checkpoint",
    "source_run_id",
    "source_phase",
    "source_method",
    "source_hp_mode",
    "source_search_best_val_acc",
    "source_search_best_type",
    "candidate_family",
    "architecture_origin",
    "hp_origin",
    "is_baseline",
    "baseline_key",
    "dedup_key",
    "is_duplicate_baseline",
    "notes",
]

PER_SEED_FIELDS = [
    "eval_id",
    "result_dir",
    "hp_mode",
    "candidate_name",
    "group",
    "candidate_family",
    "source_run_id",
    "seed_idx",
    "val_acc",
    "test_acc",
    "operations",
    "lr",
    "dropout",
    "hidden_dim",
    "l2",
]

SEARCH_TO_FINAL_FIELDS = [
    "source_run_id",
    "source_phase",
    "source_method",
    "source_hp_mode",
    "search_best_val_acc",
    "search_best_type",
    "search_best_operations",
    "final_manual_2gcn_searched_hp_test_mean",
    "final_manual_2gcn_searched_hp_test_std",
    "final_nas_arch_default_hp_test_mean",
    "final_nas_arch_global_hp_test_mean",
    "final_nas_arch_conditional_hp_test_mean",
    "final_nas_full_hp_test_mean",
    "best_final_candidate_name",
    "best_final_candidate_family",
    "best_final_test_mean",
    "best_final_test_std",
    "search_to_best_final_gap",
    "nas_full_minus_arch_default",
    "global_hp_minus_default",
    "cond_hp_minus_default",
    "full_hp_minus_global_hp",
    "notes",
]

GMM_FIELDS = [
    "run_id",
    "result_dir",
    "gmm_summary_path",
    "hp_mode",
    "search_dim",
    "n_loaded",
    "n_valid",
    "n_used_top",
    "n_components",
    "top_frac",
    "y_min",
    "y_max",
    "y_mean",
    "sampled_count",
    "enqueued_count",
    "evaluated_count",
    "loader_n_loaded",
    "loader_n_valid",
    "loader_n_skipped",
    "loader_n_bad_json",
    "loader_n_missing_path",
    "history_paths",
    "notes",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect NAS + HPO result summaries")
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument(
        "--legacy-root",
        type=str,
        default=None,
        help="explicit legacy root; reads its results/ child without modifying it",
    )
    parser.add_argument("--protocol-id", default=None)
    parser.add_argument("--analysis-name", default="collect-experiment-results")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--include_history_glob", type=str, default="history_final.json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any):
    if value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def safe_int(value: Any):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def round_float(value: Any, digits: int = 6):
    num = safe_float(value)
    if num is None:
        return ""
    return round(num, digits)


def format_float(value: Any, digits: int = 4) -> str:
    num = safe_float(value)
    if num is None:
        return ""
    return f"{num:.{digits}f}"


def format_acc(value: Any) -> str:
    return format_float(value, 4)


def format_std(value: Any) -> str:
    return format_float(value, 4)


def format_ci(value: Any) -> str:
    return format_float(value, 4)


def format_lr(value: Any) -> str:
    num = safe_float(value)
    if num is None:
        return ""
    return f"{num:.2e}"


def format_l2(value: Any) -> str:
    return format_lr(value)


def format_delta(value: Any) -> str:
    num = safe_float(value)
    if num is None:
        return ""
    return f"{num:+.4f}"


def relpath(path: str | Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return Path(path).as_posix()


def read_json(path: Path, warnings: list[str]):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        warnings.append(f"Failed to parse JSON {path}: {exc}")
        return None


def parse_listish(value: Any):
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                continue
    return None


def short_json(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    parsed = parse_listish(value)
    if parsed is not None:
        out = []
        for item in parsed:
            num = safe_float(item)
            out.append(round(num, digits) if num is not None else item)
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def join_ops(value: Any, sep: str = "|") -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return sep.join(str(v) for v in value)
    return str(value)


def compact_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def infer_hp_mode_from_name(name: str) -> str:
    for mode in HP_MODE_SEARCH_DIM:
        if mode in name:
            return mode
    return "unknown"


def infer_phase_method(dirname: str) -> tuple[str, str]:
    name = Path(dirname).name
    if name.startswith("bo_phase4_tpe"):
        return "phase4_tpe", "tpe"
    if name.startswith("bo_phase4"):
        return "phase4", "bo"
    if name.startswith("bo_phase3"):
        return "phase3", "bo"
    return "unknown", "unknown"


def history_records_from_json(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in ("history", "records", "trials", "results"):
            value = obj.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if any(key in obj for key in ("val_acc", "z_search", "z_arch", "valid")):
            return [obj]
    return []


def final_eval_records_from_json(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in ("results", "candidates", "evaluations"):
            value = obj.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if any(key in obj for key in ("test_mean", "val_mean", "candidate_name", "name")):
            return [obj]
    return []


def scan_result_dirs(results_root: Path) -> tuple[list[Path], list[Path]]:
    patterns = ["bo_phase3*", "bo_phase4*", "bo_phase4_tpe*", "final_eval*"]
    dirs: list[Path] = []
    if results_root.exists():
        for pattern in patterns:
            dirs.extend(path for path in results_root.glob(pattern) if path.is_dir())
        dirs.extend(path.parent for path in results_root.rglob("history_final.json"))
        dirs.extend(path.parent for path in results_root.rglob("final_results*.json"))
    unique_dirs = sorted({path.resolve(): path for path in dirs}.values(), key=lambda p: p.as_posix())
    final_dirs = [
        path
        for path in unique_dirs
        if path.name.startswith("final_eval")
        or any(path.glob("final_results*.json"))
    ]
    return unique_dirs, final_dirs


def choose_best_record(records: list[dict[str, Any]], notes: list[str]):
    valid_records = [
        row for row in records
        if row.get("valid") is True and safe_float(row.get("val_acc")) is not None
    ]
    pool = valid_records
    if not pool:
        pool = [row for row in records if safe_float(row.get("val_acc")) is not None]
        if pool:
            notes.append("no_valid_record_best_from_all")
    if not pool:
        return None
    return max(pool, key=lambda row: safe_float(row.get("val_acc")) or float("-inf"))


def source_run_id_from_best_z(source_best_z: Any) -> str:
    if not source_best_z:
        return ""
    path = Path(str(source_best_z))
    if path.name.endswith(".pt") and path.parent.name:
        return path.parent.name
    parts = path.parts
    if len(parts) >= 2:
        return parts[-2]
    return ""


def classify_candidate(row: dict[str, Any]) -> dict[str, Any]:
    group = str(row.get("group") or "")
    name = str(row.get("candidate_name") or "")
    if group == "baseline":
        family = "baseline"
        arch_origin = "manual"
        hp_origin = "default"
        is_baseline = True
    elif name == "Manual_2xGCN_searchedHP":
        family = "manual_arch_searched_hp"
        arch_origin = "manual"
        hp_origin = "searched"
        is_baseline = False
    elif group == "arch_only":
        family = "nas_arch_default_hp"
        arch_origin = "nas"
        hp_origin = "default"
        is_baseline = False
    elif group == "arch_global_hp":
        family = "nas_arch_global_hp"
        arch_origin = "nas"
        hp_origin = "searched_global"
        is_baseline = False
    elif group == "cond_only":
        family = "nas_arch_conditional_hp"
        arch_origin = "nas"
        hp_origin = "searched_conditional"
        is_baseline = False
    elif group == "full":
        family = "nas_arch_full_hp"
        arch_origin = "nas"
        hp_origin = "searched_full"
        is_baseline = False
    else:
        family = group or "unknown"
        arch_origin = ""
        hp_origin = ""
        is_baseline = False
    return {
        "candidate_family": family,
        "architecture_origin": arch_origin,
        "hp_origin": hp_origin,
        "is_baseline": is_baseline,
    }


def baseline_key(row: dict[str, Any]) -> str:
    if not row.get("is_baseline"):
        return ""
    return "|".join([
        str(row.get("candidate_name", "")),
        str(row.get("operations", "")),
        str(row.get("lr", "")),
        str(row.get("dropout", "")),
        str(row.get("hidden_dim", "")),
        str(row.get("l2", "")),
    ])


def dedup_key(row: dict[str, Any]) -> str:
    return "|".join([
        str(row.get("candidate_name", "")),
        str(row.get("hp_mode", "")),
        str(row.get("operations", "")),
        str(row.get("source_run_id", "")),
        str(row.get("lr", "")),
        str(row.get("dropout", "")),
        str(row.get("hidden_dim", "")),
        str(row.get("l2", "")),
    ])


def mean_std_ci(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    mean = safe_float(row.get(f"{prefix}_mean"))
    std = safe_float(row.get(f"{prefix}_std"))
    n = safe_int(row.get("n_seeds"))
    if mean is None or std is None or n is None or n <= 1:
        return {
            f"{prefix}_se": "",
            f"{prefix}_ci95_low": "",
            f"{prefix}_ci95_high": "",
        }
    se = std / math.sqrt(n)
    return {
        f"{prefix}_se": se,
        f"{prefix}_ci95_low": mean - 1.96 * se,
        f"{prefix}_ci95_high": mean + 1.96 * se,
    }


def list_stats(values: Any) -> dict[str, Any]:
    parsed = parse_listish(values)
    nums = [safe_float(v) for v in parsed] if parsed is not None else []
    nums = [v for v in nums if v is not None]
    if not nums:
        return {"test_min": "", "test_max": "", "test_median": ""}
    sorted_nums = sorted(nums)
    mid = len(sorted_nums) // 2
    if len(sorted_nums) % 2 == 1:
        median = sorted_nums[mid]
    else:
        median = 0.5 * (sorted_nums[mid - 1] + sorted_nums[mid])
    return {
        "test_min": min(sorted_nums),
        "test_max": max(sorted_nums),
        "test_median": median,
    }


def parse_gmm_summary(path: Path, result_dir: Path, root: Path, warnings: list[str]) -> dict[str, Any] | None:
    obj = read_json(path, warnings)
    if not isinstance(obj, dict):
        return None
    notes = []
    row = {
        "run_id": result_dir.name,
        "result_dir": relpath(result_dir, root),
        "gmm_summary_path": relpath(path, root),
        "hp_mode": obj.get("hp_mode", infer_hp_mode_from_name(result_dir.name)),
        "search_dim": obj.get("search_dim", ""),
        "n_loaded": obj.get("n_loaded", ""),
        "n_valid": obj.get("n_valid", ""),
        "n_used_top": obj.get("n_used_top", ""),
        "n_components": obj.get("n_components", ""),
        "top_frac": obj.get("top_frac", ""),
        "y_min": obj.get("y_min", ""),
        "y_max": obj.get("y_max", ""),
        "y_mean": obj.get("y_mean", ""),
        "sampled_count": obj.get("sampled_count", ""),
        "enqueued_count": obj.get("enqueued_count", ""),
        "evaluated_count": obj.get("evaluated_count", ""),
        "loader_n_loaded": obj.get("loader_n_loaded", ""),
        "loader_n_valid": obj.get("loader_n_valid", ""),
        "loader_n_skipped": obj.get("loader_n_skipped", ""),
        "loader_n_bad_json": obj.get("loader_n_bad_json", ""),
        "loader_n_missing_path": obj.get("loader_n_missing_path", ""),
        "history_paths": compact_json(obj.get("history_paths", obj.get("loader_history_paths", []))),
        "notes": ";".join(notes),
    }
    return row


def summarize_search_dir(result_dir: Path, args: argparse.Namespace, root: Path, warnings: list[str]):
    history_matches = sorted(result_dir.glob(args.include_history_glob))
    history_path = history_matches[0] if history_matches else result_dir / "history_final.json"
    has_history = history_path.exists()
    gmm_summaries = sorted(result_dir.glob("gmm_init_summary*.json"))
    best_z_final = result_dir / "best_z_final.pt"
    best_z_search_final = result_dir / "best_z_search_final.pt"
    notes: list[str] = []
    records: list[dict[str, Any]] = []

    if has_history:
        obj = read_json(history_path, warnings)
        records = history_records_from_json(obj)
        if obj is not None and not records:
            notes.append("history_json_unrecognized")
    else:
        notes.append("missing_history")

    phase, method = infer_phase_method(result_dir.name)
    hp_mode = "unknown"
    search_dim = None
    if records:
        for rec in records:
            if rec.get("hp_mode"):
                hp_mode = str(rec.get("hp_mode"))
                break
        if hp_mode == "unknown":
            hp_mode = infer_hp_mode_from_name(result_dir.name)
        for rec in records:
            search_dim = safe_int(rec.get("search_dim"))
            if search_dim is not None:
                break
    else:
        hp_mode = infer_hp_mode_from_name(result_dir.name)

    if search_dim is None:
        search_dim = HP_MODE_SEARCH_DIM.get(hp_mode)

    type_counts = Counter(str(rec.get("type", "")) for rec in records)
    best = choose_best_record(records, notes)
    ops = best.get("operations") if best else None
    edges = best.get("edges") if best else None
    z_search = best.get("z_search") if best else None
    gmm_enabled = bool(gmm_summaries or type_counts.get("gmm_init", 0) > 0)

    if not best_z_final.exists():
        notes.append("missing_best_z_final")
    if not best_z_search_final.exists():
        notes.append("missing_best_z_search_final")
    if gmm_enabled and not gmm_summaries:
        notes.append("missing_gmm_summary")
        warnings.append(f"GMM run missing gmm summary: {result_dir}")
    if not gmm_enabled:
        notes.append("no_gmm_init_detected")

    row = {
        "run_id": result_dir.name,
        "result_dir": relpath(result_dir, root),
        "phase": phase,
        "method": method,
        "hp_mode": hp_mode,
        "search_dim": "" if search_dim is None else search_dim,
        "history_path": relpath(history_path, root) if has_history else "",
        "has_history": has_history,
        "has_best_z_final": best_z_final.exists(),
        "has_best_z_search_final": best_z_search_final.exists(),
        "has_gmm_summary": bool(gmm_summaries),
        "gmm_enabled": gmm_enabled,
        "gmm_n_loaded": "",
        "gmm_n_valid": "",
        "gmm_n_used_top": "",
        "gmm_sampled_count": "",
        "gmm_enqueued_count": "",
        "gmm_evaluated_count": "",
        "n_total": len(records),
        "n_valid": sum(1 for rec in records if rec.get("valid") is True),
        "n_invalid": sum(1 for rec in records if rec.get("valid") is False),
        "n_lhs_init": type_counts.get("lhs_init", 0) + type_counts.get("init", 0),
        "n_gmm_init": type_counts.get("gmm_init", 0),
        "n_warm_start": type_counts.get("warm_start", 0),
        "n_bo": type_counts.get("bo", 0),
        "n_tpe": type_counts.get("tpe", 0),
        "best_step": "" if best is None else best.get("step", ""),
        "best_type": "" if best is None else best.get("type", ""),
        "best_val_acc": "" if best is None else best.get("val_acc", ""),
        "best_valid": "" if best is None else best.get("valid", ""),
        "best_lr": "" if best is None else best.get("lr", ""),
        "best_dropout": "" if best is None else best.get("dropout", ""),
        "best_hidden_dim": "" if best is None else best.get("hidden_dim", ""),
        "best_l2": "" if best is None else best.get("l2", ""),
        "best_gat_heads": "" if best is None else best.get("gat_heads", ""),
        "best_sage_aggr": "" if best is None else best.get("sage_aggr", ""),
        "best_gin_eps": "" if best is None else best.get("gin_eps", ""),
        "best_operations": join_ops(ops),
        "best_num_ops": "" if not isinstance(ops, list) else len([op for op in ops if op != "Identity"]),
        "best_num_edges": "" if not isinstance(edges, list) else len(edges),
        "z_search_len": "" if not isinstance(z_search, list) else len(z_search),
        "notes": ";".join(notes),
    }

    best_record = {
        "run_id": result_dir.name,
        "result_dir": relpath(result_dir, root),
        "phase": phase,
        "method": method,
        "hp_mode": hp_mode,
        "search_dim": search_dim,
        "best_record": best,
        "best_z_final_path": relpath(best_z_final, root) if best_z_final.exists() else "",
        "best_z_search_final_path": relpath(best_z_search_final, root) if best_z_search_final.exists() else "",
        "gmm_summary_path": relpath(gmm_summaries[0], root) if gmm_summaries else "",
    }
    return row, best_record


def summarize_final_json(json_path: Path, result_dir: Path, root: Path, warnings: list[str]) -> list[dict[str, Any]]:
    obj = read_json(json_path, warnings)
    if obj is None:
        return []
    records = final_eval_records_from_json(obj)
    if not records:
        warnings.append(f"Unrecognized final_eval JSON structure: {json_path}")
        return []

    rows = []
    inferred_mode = infer_hp_mode_from_name(result_dir.name)
    for idx, record in enumerate(records):
        ops = record.get("operations")
        edges = record.get("edges")
        val_list = record.get("val_list")
        test_list = record.get("test_list")
        parsed_val = parse_listish(val_list)
        parsed_test = parse_listish(test_list)
        n_seeds = ""
        if isinstance(parsed_test, list):
            n_seeds = len(parsed_test)
        elif isinstance(parsed_val, list):
            n_seeds = len(parsed_val)

        notes = []
        if "test_mean" not in record:
            notes.append("missing_test_mean")
        if "val_mean" not in record:
            notes.append("missing_val_mean")

        hp_mode = str(record.get("hp_mode") or inferred_mode)
        row = {
            "eval_id": f"{result_dir.name}:{json_path.stem}:{idx}",
            "result_dir": relpath(result_dir, root),
            "json_path": relpath(json_path, root),
            "hp_mode": hp_mode,
            "candidate_name": record.get("candidate_name") or record.get("name", ""),
            "group": record.get("group", ""),
            "description": record.get("description", ""),
            "val_mean": record.get("val_mean", ""),
            "val_std": record.get("val_std", ""),
            "test_mean": record.get("test_mean", ""),
            "test_std": record.get("test_std", ""),
            "n_seeds": n_seeds,
            "val_list": short_json(parsed_val),
            "test_list": short_json(parsed_test),
            "lr": record.get("lr", ""),
            "dropout": record.get("dropout", ""),
            "hidden_dim": record.get("hidden_dim", ""),
            "l2": record.get("l2", ""),
            "gat_heads": record.get("gat_heads", ""),
            "sage_aggr": record.get("sage_aggr", ""),
            "gin_eps": record.get("gin_eps", ""),
            "gat_heads_by_layer": short_json(record.get("gat_heads_by_layer")),
            "sage_aggr_by_layer": short_json(record.get("sage_aggr_by_layer")),
            "gin_eps_by_layer": short_json(record.get("gin_eps_by_layer")),
            "operations": join_ops(ops),
            "num_ops": "" if not isinstance(ops, list) else len([op for op in ops if op != "Identity"]),
            "num_edges": "" if not isinstance(edges, list) else len(edges),
            "source_best_z": record.get("source_best_z", ""),
            "source_checkpoint": record.get("source_checkpoint", ""),
            "notes": ";".join(notes),
        }
        row.update(mean_std_ci(row, "val"))
        row.update(mean_std_ci(row, "test"))
        row.update(list_stats(parsed_test))
        row.update(classify_candidate(row))
        rows.append(row)
    return rows


def link_final_rows(final_rows: list[dict[str, Any]], search_by_run_id: dict[str, dict[str, Any]], warnings: list[str]) -> None:
    baseline_seen: set[str] = set()
    duplicate_baselines = 0
    for row in final_rows:
        source_run = source_run_id_from_best_z(row.get("source_best_z"))
        row["source_run_id"] = source_run
        src = search_by_run_id.get(source_run)
        if source_run and src is None:
            warnings.append(f"Could not link source_run_id={source_run} for eval_id={row.get('eval_id')}")
        row["source_phase"] = "" if src is None else src.get("phase", "")
        row["source_method"] = "" if src is None else src.get("method", "")
        row["source_hp_mode"] = "" if src is None else src.get("hp_mode", "")
        row["source_search_best_val_acc"] = "" if src is None else src.get("best_val_acc", "")
        row["source_search_best_type"] = "" if src is None else src.get("best_type", "")
        row["baseline_key"] = baseline_key(row)
        row["dedup_key"] = dedup_key(row)
        if row.get("is_baseline"):
            key = row["baseline_key"]
            row["is_duplicate_baseline"] = key in baseline_seen
            if key in baseline_seen:
                duplicate_baselines += 1
            baseline_seen.add(key)
        else:
            row["is_duplicate_baseline"] = False
    if duplicate_baselines:
        warnings.append(f"Duplicate baseline rows detected: {duplicate_baselines}")


def dedup_final_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_baseline: set[str] = set()
    out = []
    for row in final_rows:
        if row.get("is_baseline"):
            key = row.get("baseline_key", "")
            if key in seen_baseline:
                continue
            seen_baseline.add(key)
        out.append(row)
    return out


def make_per_seed_rows(final_rows: list[dict[str, Any]], warnings: list[str]) -> list[dict[str, Any]]:
    out = []
    for row in final_rows:
        val_list = parse_listish(row.get("val_list"))
        test_list = parse_listish(row.get("test_list"))
        if val_list is None or test_list is None:
            warnings.append(f"Could not parse val_list/test_list for eval_id={row.get('eval_id')}")
            continue
        n = min(len(val_list), len(test_list))
        if len(val_list) != len(test_list):
            warnings.append(
                f"val_list/test_list length mismatch for eval_id={row.get('eval_id')}: "
                f"{len(val_list)} vs {len(test_list)}"
            )
        for idx in range(n):
            out.append({
                "eval_id": row.get("eval_id", ""),
                "result_dir": row.get("result_dir", ""),
                "hp_mode": row.get("hp_mode", ""),
                "candidate_name": row.get("candidate_name", ""),
                "group": row.get("group", ""),
                "candidate_family": row.get("candidate_family", ""),
                "source_run_id": row.get("source_run_id", ""),
                "seed_idx": idx,
                "val_acc": val_list[idx],
                "test_acc": test_list[idx],
                "operations": row.get("operations", ""),
                "lr": row.get("lr", ""),
                "dropout": row.get("dropout", ""),
                "hidden_dim": row.get("hidden_dim", ""),
                "l2": row.get("l2", ""),
            })
    return out


def best_metric(rows: list[dict[str, Any]], family: str, metric: str = "test_mean"):
    candidates = [row for row in rows if row.get("candidate_family") == family and safe_float(row.get(metric)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: safe_float(row.get(metric)) or float("-inf"))


def subtract(a: Any, b: Any):
    af = safe_float(a)
    bf = safe_float(b)
    if af is None or bf is None:
        return ""
    return af - bf


def build_search_to_final(search_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        if row.get("source_run_id"):
            by_source[str(row["source_run_id"])].append(row)

    out = []
    for srow in search_rows:
        run_id = str(srow.get("run_id", ""))
        rows = by_source.get(run_id, [])
        if not rows:
            continue
        manual = best_metric(rows, "manual_arch_searched_hp")
        arch_default = best_metric(rows, "nas_arch_default_hp")
        global_hp = best_metric(rows, "nas_arch_global_hp")
        cond_hp = best_metric(rows, "nas_arch_conditional_hp")
        full = best_metric(rows, "nas_arch_full_hp")
        best_final = max(rows, key=lambda row: safe_float(row.get("test_mean")) or float("-inf"))
        notes = ["metric_mismatch_val_vs_test"]
        out.append({
            "source_run_id": run_id,
            "source_phase": srow.get("phase", ""),
            "source_method": srow.get("method", ""),
            "source_hp_mode": srow.get("hp_mode", ""),
            "search_best_val_acc": srow.get("best_val_acc", ""),
            "search_best_type": srow.get("best_type", ""),
            "search_best_operations": srow.get("best_operations", ""),
            "final_manual_2gcn_searched_hp_test_mean": "" if manual is None else manual.get("test_mean", ""),
            "final_manual_2gcn_searched_hp_test_std": "" if manual is None else manual.get("test_std", ""),
            "final_nas_arch_default_hp_test_mean": "" if arch_default is None else arch_default.get("test_mean", ""),
            "final_nas_arch_global_hp_test_mean": "" if global_hp is None else global_hp.get("test_mean", ""),
            "final_nas_arch_conditional_hp_test_mean": "" if cond_hp is None else cond_hp.get("test_mean", ""),
            "final_nas_full_hp_test_mean": "" if full is None else full.get("test_mean", ""),
            "best_final_candidate_name": best_final.get("candidate_name", ""),
            "best_final_candidate_family": best_final.get("candidate_family", ""),
            "best_final_test_mean": best_final.get("test_mean", ""),
            "best_final_test_std": best_final.get("test_std", ""),
            "search_to_best_final_gap": subtract(best_final.get("test_mean"), srow.get("best_val_acc")),
            "nas_full_minus_arch_default": subtract(
                "" if full is None else full.get("test_mean"),
                "" if arch_default is None else arch_default.get("test_mean"),
            ),
            "global_hp_minus_default": subtract(
                "" if global_hp is None else global_hp.get("test_mean"),
                "" if arch_default is None else arch_default.get("test_mean"),
            ),
            "cond_hp_minus_default": subtract(
                "" if cond_hp is None else cond_hp.get("test_mean"),
                "" if arch_default is None else arch_default.get("test_mean"),
            ),
            "full_hp_minus_global_hp": subtract(
                "" if full is None else full.get("test_mean"),
                "" if global_hp is None else global_hp.get("test_mean"),
            ),
            "notes": ";".join(notes),
        })
    return out


def csv_value(field: str, value: Any):
    if value is None:
        return ""
    if field in ACC_FIELDS or field in STD_FIELDS or field in DELTA_FIELDS:
        return round_float(value, 6)
    if field in SCI_FIELDS:
        return round_float(value, 10)
    if field in FOUR_DEC_FIELDS:
        return round_float(value, 6)
    if field in {"val_list", "test_list"}:
        return short_json(value)
    return value


def pretty_value(field: str, value: Any):
    if value is None or value == "":
        return ""
    if field in ACC_FIELDS:
        return format_acc(value)
    if field in STD_FIELDS:
        return format_std(value)
    if field in DELTA_FIELDS:
        return format_delta(value)
    if field in SCI_FIELDS:
        return format_lr(value)
    if field in FOUR_DEC_FIELDS:
        return format_float(value, 4)
    if field in {"val_list", "test_list", "history_paths", "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer"}:
        return short_json(value)
    return value


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]], pretty: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalizer = pretty_value if pretty else csv_value
            writer.writerow({field: normalizer(field, row.get(field, "")) for field in fields})


def table_row(values: list[Any]) -> str:
    return "| " + " | ".join("" if v is None else str(v) for v in values) + " |"


def numeric_key(row: dict[str, Any], key: str) -> float:
    value = safe_float(row.get(key))
    return value if value is not None else float("-inf")


def best_by_hp_mode(rows: list[dict[str, Any]], metric: str) -> dict[str, dict[str, Any]]:
    out = {}
    for mode in ("global4", "hybrid_cond7", "layer_cond19"):
        mode_rows = [row for row in rows if row.get("hp_mode") == mode and safe_float(row.get(metric)) is not None]
        if mode_rows:
            out[mode] = max(mode_rows, key=lambda row: numeric_key(row, metric))
    return out


def write_report(
    path: Path,
    search_rows: list[dict[str, Any]],
    final_dedup_rows: list[dict[str, Any]],
    search_to_final_rows: list[dict[str, Any]],
    gmm_rows: list[dict[str, Any]],
) -> None:
    lines = ["# NAS + HPO Experiment Summary", ""]
    sorted_search = sorted(search_rows, key=lambda row: numeric_key(row, "best_val_acc"), reverse=True)
    sorted_final = sorted(final_dedup_rows, key=lambda row: numeric_key(row, "test_mean"), reverse=True)

    lines.extend([
        "## Search Runs Overview",
        "",
        table_row(["rank", "run_id", "phase", "method", "hp_mode", "search_dim", "gmm_enabled", "n_total", "n_valid", "best_val_acc", "best_type", "best_operations"]),
        table_row(["---"] * 12),
    ])
    for rank, row in enumerate(sorted_search, 1):
        lines.append(table_row([
            rank,
            row.get("run_id", ""),
            row.get("phase", ""),
            row.get("method", ""),
            row.get("hp_mode", ""),
            row.get("search_dim", ""),
            row.get("gmm_enabled", ""),
            row.get("n_total", ""),
            row.get("n_valid", ""),
            format_acc(row.get("best_val_acc")),
            row.get("best_type", ""),
            row.get("best_operations", ""),
        ]))

    lines.extend([
        "",
        "## Final Eval Overview",
        "",
        table_row(["rank", "candidate_name", "group", "hp_mode", "test_mean", "test_std", "val_mean", "val_std", "operations"]),
        table_row(["---"] * 9),
    ])
    for rank, row in enumerate(sorted_final, 1):
        lines.append(table_row([
            rank,
            row.get("candidate_name", ""),
            row.get("group", ""),
            row.get("hp_mode", ""),
            format_acc(row.get("test_mean")),
            format_std(row.get("test_std")),
            format_acc(row.get("val_mean")),
            format_std(row.get("val_std")),
            row.get("operations", ""),
        ]))

    lines.extend(["", "## Search-to-final Summary", ""])
    lines.append(table_row([
        "source_run_id",
        "search_best_val_acc",
        "best_final_candidate_name",
        "best_final_test_mean",
        "final_nas_full_hp_test_mean",
        "final_manual_2gcn_searched_hp_test_mean",
    ]))
    lines.append(table_row(["---"] * 6))
    for row in search_to_final_rows:
        lines.append(table_row([
            row.get("source_run_id", ""),
            format_acc(row.get("search_best_val_acc")),
            row.get("best_final_candidate_name", ""),
            format_acc(row.get("best_final_test_mean")),
            format_acc(row.get("final_nas_full_hp_test_mean")),
            format_acc(row.get("final_manual_2gcn_searched_hp_test_mean")),
        ]))

    lines.extend(["", "## GMM Summary", ""])
    lines.append(table_row(["run_id", "hp_mode", "n_valid", "n_used_top", "sampled_count", "enqueued_count", "evaluated_count"]))
    lines.append(table_row(["---"] * 7))
    for row in gmm_rows:
        lines.append(table_row([
            row.get("run_id", ""),
            row.get("hp_mode", ""),
            row.get("n_valid", ""),
            row.get("n_used_top", ""),
            row.get("sampled_count", ""),
            row.get("enqueued_count", ""),
            row.get("evaluated_count", ""),
        ]))

    lines.extend(["", "## Best By hp_mode", ""])
    search_best = best_by_hp_mode(search_rows, "best_val_acc")
    final_best = best_by_hp_mode(final_dedup_rows, "test_mean")
    for mode in ("global4", "hybrid_cond7", "layer_cond19"):
        s = search_best.get(mode)
        f = final_best.get(mode)
        lines.append(f"### {mode}")
        lines.append(
            f"- best search run: {s.get('run_id')} best_val_acc={format_acc(s.get('best_val_acc'))}"
            if s else "- best search run: N/A"
        )
        lines.append(
            f"- best final_eval candidate: {f.get('candidate_name')} test_mean={format_acc(f.get('test_mean'))}"
            if f else "- best final_eval candidate: N/A"
        )
        lines.append("")

    best_search = sorted_search[0] if sorted_search and safe_float(sorted_search[0].get("best_val_acc")) is not None else None
    best_final = sorted_final[0] if sorted_final and safe_float(sorted_final[0].get("test_mean")) is not None else None
    gmm_has_records = any(int(row.get("n_gmm_init") or 0) > 0 for row in search_rows)
    missing_history = [row["run_id"] for row in search_rows if not row.get("has_history")]
    missing_best_z = [row["run_id"] for row in search_rows if not row.get("has_best_z_final") and not row.get("has_best_z_search_final")]
    modes_with_final = {row.get("hp_mode") for row in final_dedup_rows}
    missing_final_modes = [mode for mode in ("global4", "hybrid_cond7", "layer_cond19") if mode not in modes_with_final]

    lines.extend(["## Automatic Observations", ""])
    lines.append(
        f"- Highest search best_val_acc: {best_search.get('hp_mode')} / {best_search.get('run_id')} = {format_acc(best_search.get('best_val_acc'))}"
        if best_search else "- Highest search best_val_acc: N/A"
    )
    lines.append(
        f"- Highest final_eval test_mean: {best_final.get('hp_mode')} / {best_final.get('candidate_name')} = {format_acc(best_final.get('test_mean'))}"
        if best_final else "- Highest final_eval test_mean: N/A"
    )
    lines.append(f"- GMM init produced records: {gmm_has_records}")
    lines.append(f"- Search runs missing history: {', '.join(missing_history) if missing_history else 'none'}")
    lines.append(f"- Search runs missing best_z files: {', '.join(missing_best_z) if missing_best_z else 'none'}")
    lines.append(f"- hp_modes missing final_eval rows: {', '.join(missing_final_modes) if missing_final_modes else 'none'}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_analysis_notes(
    path: Path,
    search_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    final_dedup_rows: list[dict[str, Any]],
    search_to_final_rows: list[dict[str, Any]],
    gmm_rows: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
) -> None:
    lines = ["# Analysis Notes", ""]
    lines.extend([
        "## Dataset scan summary",
        f"- n_search_runs: {len(search_rows)}",
        f"- n_final_eval_rows: {len(final_rows)}",
        f"- n_unique_final_eval_rows_after_dedup: {len(final_dedup_rows)}",
        f"- n_gmm_summaries: {len(gmm_rows)}",
        f"- warnings: {len(warnings)}",
        f"- errors: {len(errors)}",
        "",
    ])

    lines.extend(["## Search ranking", ""])
    for rank, row in enumerate(sorted(search_rows, key=lambda r: numeric_key(r, "best_val_acc"), reverse=True)[:5], 1):
        lines.append(f"{rank}. {row.get('run_id')} ({row.get('hp_mode')}) best_val_acc={format_acc(row.get('best_val_acc'))}")
    lines.append("")

    lines.extend(["## Final-eval ranking", ""])
    for rank, row in enumerate(sorted(final_dedup_rows, key=lambda r: numeric_key(r, "test_mean"), reverse=True)[:10], 1):
        lines.append(f"{rank}. {row.get('candidate_name')} ({row.get('hp_mode')}) test_mean={format_acc(row.get('test_mean'))} test_std={format_std(row.get('test_std'))}")
    lines.append("")

    lines.extend(["## Best by hp_mode", ""])
    search_best = best_by_hp_mode(search_rows, "best_val_acc")
    final_best = best_by_hp_mode(final_dedup_rows, "test_mean")
    for mode in ("global4", "hybrid_cond7", "layer_cond19"):
        s = search_best.get(mode)
        f = final_best.get(mode)
        lines.append(f"- {mode} search: {s.get('run_id')} best_val_acc={format_acc(s.get('best_val_acc'))}" if s else f"- {mode} search: N/A")
        lines.append(f"- {mode} final: {f.get('candidate_name')} test_mean={format_acc(f.get('test_mean'))}" if f else f"- {mode} final: N/A")
    lines.append("")

    lines.extend(["## Search-to-final observations", ""])
    manual_rows = [row for row in search_to_final_rows if safe_float(row.get("final_manual_2gcn_searched_hp_test_mean")) is not None]
    if manual_rows:
        best_manual = max(manual_rows, key=lambda r: numeric_key(r, "final_manual_2gcn_searched_hp_test_mean"))
        lines.append(f"- Best searched HP transferred to Manual_2xGCN: {best_manual.get('source_run_id')} test_mean={format_acc(best_manual.get('final_manual_2gcn_searched_hp_test_mean'))}")
    full_positive = [row for row in search_to_final_rows if (safe_float(row.get("nas_full_minus_arch_default")) or 0) > 0]
    global_positive = [row for row in search_to_final_rows if (safe_float(row.get("global_hp_minus_default")) or 0) > 0]
    cond_positive = [row for row in search_to_final_rows if (safe_float(row.get("cond_hp_minus_default")) or 0) > 0]
    lines.append(f"- NAS_full exceeds NAS_arch_defaultHP in {len(full_positive)} linked runs.")
    lines.append(f"- Global HP improves NAS architecture in {len(global_positive)} linked runs.")
    lines.append(f"- Conditional HP improves NAS architecture in {len(cond_positive)} linked runs.")
    lines.append("")

    lines.extend([
        "## Caveats",
        "- search best_val_acc and final test_mean are different metrics and should not be treated as equivalent.",
        "- Baseline rows may repeat across hp_mode final_eval directories; prefer dedup outputs for leaderboards.",
        "- When final_eval n_seeds is small, interpret mean together with std and CI.",
        "- With only one search run, GMM, TPE/BO, and random seed effects cannot be cleanly separated.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.results_root is not None and args.legacy_root is not None:
        raise ValueError("--results_root and --legacy-root are mutually exclusive")
    if args.protocol_id is not None:
        validate_protocol_id(args.protocol_id)
    if args.results_root is not None:
        results_root = Path(args.results_root)
    elif args.legacy_root is not None:
        results_root = Path(args.legacy_root) / "results"
    elif args.protocol_id is not None:
        results_root = project_root / "results" / "search" / args.protocol_id
    else:
        results_root = project_root / "results"
    if args.output is not None:
        output_dir = Path(args.output)
    elif args.protocol_id is not None:
        output_dir = posthoc_dir(
            project_root,
            args.protocol_id,
            args.analysis_name,
        )
    else:
        output_dir = project_root / "results_analysis_bundle"
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    errors: list[str] = []
    scanned_json_files: list[str] = []

    if not results_root.exists():
        warnings.append(f"results_root does not exist: {results_root}")

    result_dirs, final_dirs = scan_result_dirs(results_root)
    if args.verbose:
        print(f"Scanned result dirs: {len(result_dirs)}")

    search_rows: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []
    gmm_rows: list[dict[str, Any]] = []

    for result_dir in result_dirs:
        for gmm_path in sorted(result_dir.glob("gmm_init_summary*.json")):
            scanned_json_files.append(relpath(gmm_path, project_root))
            row = parse_gmm_summary(gmm_path, result_dir, project_root, warnings)
            if row is not None:
                gmm_rows.append(row)

    gmm_by_run = {row["run_id"]: row for row in gmm_rows}

    for result_dir in result_dirs:
        if result_dir.name.startswith("final_eval"):
            continue
        phase, _method = infer_phase_method(result_dir.name)
        if phase == "unknown":
            continue
        row, best_record = summarize_search_dir(result_dir, args, project_root, warnings)
        gmm = gmm_by_run.get(row["run_id"])
        if gmm:
            row.update({
                "gmm_n_loaded": gmm.get("n_loaded", ""),
                "gmm_n_valid": gmm.get("n_valid", ""),
                "gmm_n_used_top": gmm.get("n_used_top", ""),
                "gmm_sampled_count": gmm.get("sampled_count", ""),
                "gmm_enqueued_count": gmm.get("enqueued_count", ""),
                "gmm_evaluated_count": gmm.get("evaluated_count", ""),
            })
        search_rows.append(row)
        best_records.append(best_record)
        if row.get("history_path"):
            scanned_json_files.append(row["history_path"])

    search_by_run_id = {row["run_id"]: row for row in search_rows}

    final_rows: list[dict[str, Any]] = []
    for final_dir in final_dirs:
        for json_path in sorted(final_dir.glob("*.json")):
            scanned_json_files.append(relpath(json_path, project_root))
            final_rows.extend(summarize_final_json(json_path, final_dir, project_root, warnings))

    link_final_rows(final_rows, search_by_run_id, warnings)
    final_dedup_rows = dedup_final_rows(final_rows)
    per_seed_rows = make_per_seed_rows(final_rows, warnings)
    search_to_final_rows = build_search_to_final(search_rows, final_rows)

    write_csv(output_dir / "search_runs_summary.csv", SEARCH_CSV_FIELDS, search_rows)
    write_csv(output_dir / "final_eval_summary.csv", FINAL_CSV_FIELDS, final_rows)
    write_csv(output_dir / "final_eval_summary_dedup.csv", FINAL_CSV_FIELDS, final_dedup_rows)
    write_csv(output_dir / "final_eval_per_seed.csv", PER_SEED_FIELDS, per_seed_rows)
    write_csv(output_dir / "search_to_final_summary.csv", SEARCH_TO_FINAL_FIELDS, search_to_final_rows)
    write_csv(output_dir / "gmm_summary_flat.csv", GMM_FIELDS, gmm_rows)

    write_csv(output_dir / "search_runs_summary_pretty.csv", SEARCH_CSV_FIELDS, search_rows, pretty=True)
    write_csv(output_dir / "final_eval_summary_pretty.csv", FINAL_CSV_FIELDS, final_rows, pretty=True)
    write_csv(output_dir / "final_eval_summary_dedup_pretty.csv", FINAL_CSV_FIELDS, final_dedup_rows, pretty=True)
    write_csv(output_dir / "search_to_final_summary_pretty.csv", SEARCH_TO_FINAL_FIELDS, search_to_final_rows, pretty=True)
    write_csv(output_dir / "gmm_summary_flat_pretty.csv", GMM_FIELDS, gmm_rows, pretty=True)

    best_records_payload = {
        "generated_at": utc_now(),
        "records": best_records,
    }
    (output_dir / "best_records.json").write_text(
        json.dumps(best_records_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(output_dir / "comparison_report.md", search_rows, final_dedup_rows, search_to_final_rows, gmm_rows)
    write_analysis_notes(
        output_dir / "analysis_notes.md",
        search_rows,
        final_rows,
        final_dedup_rows,
        search_to_final_rows,
        gmm_rows,
        warnings,
        errors,
    )

    generated_files = [
        "manifest.json",
        "search_runs_summary.csv",
        "final_eval_summary.csv",
        "best_records.json",
        "comparison_report.md",
        "final_eval_summary_dedup.csv",
        "final_eval_per_seed.csv",
        "search_to_final_summary.csv",
        "gmm_summary_flat.csv",
        "analysis_notes.md",
        "search_runs_summary_pretty.csv",
        "final_eval_summary_pretty.csv",
        "final_eval_summary_dedup_pretty.csv",
        "search_to_final_summary_pretty.csv",
        "gmm_summary_flat_pretty.csv",
    ]

    manifest = {
        "generated_at": utc_now(),
        "project_root": project_root.as_posix(),
        "output_dir": relpath(output_dir, project_root),
        "scanned_result_dirs": [relpath(path, project_root) for path in result_dirs],
        "scanned_json_files": sorted(set(scanned_json_files)),
        "n_search_runs": len(search_rows),
        "n_final_eval_rows": len(final_rows),
        "n_final_eval_rows_dedup": len(final_dedup_rows),
        "n_per_seed_rows": len(per_seed_rows),
        "n_gmm_summaries": len(gmm_rows),
        "generated_files": [relpath(output_dir / name, project_root) for name in generated_files],
        "formatting": {
            "acc_digits": 4,
            "csv_float_round_digits": 6,
            "lr_format": "scientific_2",
            "pretty_outputs": True,
        },
        "warnings": warnings,
        "errors": errors,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Results analysis bundle: {output_dir.as_posix()}")
    print("Generated files:")
    for name in generated_files:
        print(f"  {output_dir / name}")


if __name__ == "__main__":
    main()
