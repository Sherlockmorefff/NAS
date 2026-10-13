#!/usr/bin/env python3
"""Prepare, adapt, and analyze the unified 300/80 Cora label expansion.

The module is diagnostic-only.  It never launches BO or trains a VAE.  Its
``prepare`` command freezes the 26 configurations and the complete task rule
before labels are observed; ``adaptive-plan`` applies the pre-registered CI
rule; ``analyze`` creates the test-free label set and representation report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, t
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import (
    architecture_fingerprint,
    encode_architecture_mu,
    load_joint_vae,
    normalize_architecture_config,
)
from analyse.compare_matched_representations import (
    METRICS,
    REPRESENTATIONS,
    evaluate_protocol,
    grouped_repeated_splits,
    time_prefix_splits,
)
from analyse.stage3_noise_labels import compute_vae_finetune_ready
from eval_utils import decode_hp_by_mode


EXPECTED_CONTEXT = {
    "dataset": "cora",
    "dataset_content_fingerprint": "373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7",
    "split_fingerprint": "bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09",
    "graph_transform": "NormalizeFeatures",
    "training_mode": "full_batch",
    "metric": "accuracy",
    "hp_mode": "global4",
    "normalization": "none",
}
EXPECTED_DATASET_CONTEXT = {
    "canonical_name": "cora",
    "source_class": "torch_geometric.datasets.Planetoid",
    "source_version": "2.7.0",
    "torch_version": "2.5.1+cu121",
    "pyg_version": "2.7.0",
    "ogb_version": "1.3.6",
    "dataset_content_fingerprint": EXPECTED_CONTEXT["dataset_content_fingerprint"],
    "split_protocol": "planetoid_public",
    "split_seed": None,
    "split_fingerprint": EXPECTED_CONTEXT["split_fingerprint"],
    "graph_transform": EXPECTED_CONTEXT["graph_transform"],
    "training_mode": EXPECTED_CONTEXT["training_mode"],
    "metric": EXPECTED_CONTEXT["metric"],
}
INITIAL_SEEDS = tuple(range(15))
ADAPTIVE_SEEDS = tuple(range(15, 30))
ALL_SEEDS = tuple(range(30))
CI_HALF_WIDTH_THRESHOLD = 0.006
PROTOCOL = {"eval_epochs": 300, "patience": 80}
REPRESENTATION_GATE = {
    "top_pairwise_estimate_min": 0.60,
    "top_pairwise_ci95_low_min": 0.50,
    "prediction_interval_coverage_min": 0.80,
    "prediction_interval_coverage_max": 0.98,
    "relative_late_or_top_pairwise_ci95_low_min": 0.0,
    "top_kendall_must_not_be_significantly_negative": True,
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(dict(value))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hp_fingerprint(record: Mapping[str, Any]) -> str:
    fields = (
        "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr",
        "gin_eps", "gat_heads_by_layer", "sage_aggr_by_layer",
        "gin_eps_by_layer", "normalization",
    )
    payload = {field: record.get(field) for field in fields}
    payload["normalization"] = payload.get("normalization") or "none"
    return _canonical_hash(payload)


def _t_interval(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"ci95_low": None, "ci95_high": None, "ci95_half_width": None}
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"ci95_low": mean, "ci95_high": mean, "ci95_half_width": None}
    half = float(
        t.ppf(0.975, len(values) - 1)
        * statistics.stdev(values)
        / math.sqrt(len(values))
    )
    return {"ci95_low": mean - half, "ci95_high": mean + half, "ci95_half_width": half}


def _read_histories(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for search_seed, path in enumerate(paths):
        payload = _load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a list")
        for index, raw in enumerate(payload):
            row = dict(raw)
            row["_search_seed"] = int(row.get("search_seed", search_seed))
            row["_step"] = int(row.get("step", index))
            row["_history_path"] = str(path.resolve())
            row["_history_index"] = index
            config = normalize_architecture_config(
                {"operations": row["operations"], "edges": row["edges"]}
            )
            row["operations"] = config["operations"]
            row["edges"] = config["edges"]
            row["architecture_fingerprint"] = architecture_fingerprint(config)
            row["hp_fingerprint"] = str(row.get("hp_fingerprint") or hp_fingerprint(row))
            rows.append(row)
    return rows


def _parse_json_cell(value: str) -> Any:
    return None if value == "" else json.loads(value)


def _read_confirmation(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = dict(raw)
                for field in ("operations", "edges", "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer", "deterministic_provenance"):
                    row[field] = _parse_json_cell(str(row.get(field, "")))
                for field in ("search_seed", "candidate_rank", "search_step", "replicate_id", "final_eval_seed", "seed", "hidden_dim", "gat_heads", "best_epoch", "epochs_ran"):
                    row[field] = int(row[field])
                for field in ("search_val_acc", "val_acc", "test_acc", "lr", "dropout", "l2", "gin_eps", "gcnii_alpha", "gcnii_theta", "train_time_seconds"):
                    row[field] = float(row[field])
                row["valid"] = str(row["valid"]).lower() == "true"
                row["architecture_fingerprint"] = architecture_fingerprint(row)
                rows.append(row)
    return rows


def _confirmation_configs(
    confirmation_rows: Sequence[Mapping[str, Any]],
    history_lookup: Mapping[tuple[int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in confirmation_rows:
        groups[(int(row["search_seed"]), int(row["candidate_rank"]), str(row["candidate_fingerprint"]))].append(row)
    output = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["seed"]))
        history = history_lookup[(key[0], key[2])]
        first = group[0]
        seeds = [int(row["seed"]) for row in group]
        if seeds != list(ALL_SEEDS) or any(not row["valid"] for row in group):
            raise ValueError(f"confirmation config {key} is not 30-seed complete")
        config = normalize_architecture_config(first)
        output.append(
            {
                "config_id": f"existing_s{key[0]}_rank{key[1]:02d}",
                "source": "existing_confirmation",
                "source_search_seed": key[0],
                "source_step": int(first["search_step"]),
                "candidate_rank": key[1],
                "candidate_fingerprint": key[2],
                "architecture_fingerprint": architecture_fingerprint(config),
                "hp_fingerprint": str(history["hp_fingerprint"]),
                "operations": config["operations"],
                "edges": config["edges"],
                "z_search": list(history["z_search"][:16]),
                "z_arch": list(history["z_arch"][:12]),
                "raw_hp_vector": list(history["z_search"][12:16]),
                "lr": float(first["lr"]),
                "dropout": float(first["dropout"]),
                "hidden_dim": int(first["hidden_dim"]),
                "l2": float(first["l2"]),
                "gat_heads": int(first["gat_heads"]),
                "sage_aggr": str(first["sage_aggr"]),
                "gin_eps": float(first["gin_eps"]),
                "gat_heads_by_layer": first["gat_heads_by_layer"],
                "sage_aggr_by_layer": first["sage_aggr_by_layer"],
                "gin_eps_by_layer": first["gin_eps_by_layer"],
                "normalization": "none",
                "original_single_validation": float(first["search_val_acc"]),
                "selection_region": "confirmed_top10",
                "selection_reason": "all 50 existing 30-seed confirmation configurations retained",
                "time_index": key[0] * 500 + int(first["search_step"]),
            }
        )
    return output


def _bridge_candidates(
    stage3_selection: Sequence[Mapping[str, Any]],
    excluded: set[str],
) -> list[dict[str, Any]]:
    deduplicated: dict[str, Mapping[str, Any]] = {}
    for row in sorted(
        stage3_selection,
        key=lambda value: (int(value["time_index"]), str(value["point_id"])),
    ):
        fp = str(row["historical_fingerprint"])
        if fp not in excluded:
            deduplicated.setdefault(fp, row)
    selected = []
    for region in ("top20", "middle"):
        pool = [
            row for row in deduplicated.values()
            if row["selection_region"] == region
        ]
        pool.sort(
            key=lambda row: _canonical_hash(
                {"rule": "bridge_selection_v1", "fingerprint": row["historical_fingerprint"]}
            )
        )
        if len(pool) < 3:
            raise ValueError(f"bridge region {region} has fewer than three eligible architectures")
        selected.extend(pool[:3])

    output = []
    for row in selected:
        config = normalize_architecture_config(row["historical_config"])
        hp = decode_hp_by_mode(
            z_search=torch.tensor(row["z_search"], dtype=torch.float32),
            arch_nz=12,
            config=config,
            hp_mode="global4",
            log_lr_min=-4.0,
            log_lr_max=-1.5,
            dropout_min=0.1,
            dropout_max=0.6,
        )
        candidate = {
            "config_id": f"bridge_{row['selection_region']}_{str(row['point_id'])[:12]}",
            "source": "stage3_protocol_bridge",
            "source_search_seed": None,
            "source_step": int(row["time_index"]),
            "candidate_rank": None,
            "candidate_fingerprint": str(row["point_id"]),
            "architecture_fingerprint": str(row["historical_fingerprint"]),
            "operations": config["operations"],
            "edges": config["edges"],
            "z_search": list(row["z_search"]),
            "z_arch": list(row["z_arch"]),
            "raw_hp_vector": list(row["raw_hp_vector"]),
            "lr": float(hp["lr"]),
            "dropout": float(hp["dropout"]),
            "hidden_dim": int(hp["hidden_dim"]),
            "l2": float(hp["l2"]),
            "gat_heads": int(hp["gat_heads"]),
            "sage_aggr": str(hp["sage_aggr"]),
            "gin_eps": float(hp["gin_eps"]),
            "gat_heads_by_layer": hp["gat_heads_by_layer"],
            "sage_aggr_by_layer": hp["sage_aggr_by_layer"],
            "gin_eps_by_layer": hp["gin_eps_by_layer"],
            "normalization": "none",
            "realized_hp": hp,
            "original_single_validation": float(row["historical_val_acc"]),
            "selection_region": f"bridge_{row['selection_region']}",
            "selection_reason": "fixed-hash choice after stable (time_index, point_id) architecture deduplication",
            "previous_protocol": {"eval_epochs": 150, "patience": 20},
            "previous_seed_values": None,
            "time_index": 2500 + int(row["time_index"]),
        }
        candidate["hp_fingerprint"] = hp_fingerprint(candidate)
        output.append(candidate)
    if len(output) != 6 or len({row["architecture_fingerprint"] for row in output}) != 6:
        raise AssertionError("bridge selection must contain six unique architectures")
    return output


def _assign_complete_regions(
    representatives: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in representatives),
        key=lambda row: (
            float(row["val_acc"]), int(row["_search_seed"]), int(row["_step"]),
            str(row["candidate_fingerprint"]),
        ),
    )
    n = len(ordered)
    for index, row in enumerate(ordered):
        percentile = (index + 0.5) / n
        row["exclusive_region"] = (
            "lower" if percentile < 0.35 else "middle" if percentile < 0.65 else "upper"
        )
        row["representative_rank_ascending"] = index
        row["representative_percentile"] = percentile
    counts = {region: sum(row["exclusive_region"] == region for row in ordered) for region in ("lower", "middle", "upper")}
    if sum(counts.values()) != n or any(
        sum(row["exclusive_region"] == region for region in counts) != 1
        for row in ordered
    ):
        raise AssertionError("history regions must be exclusive and complete")
    return ordered


def _segment_select(
    rows: Sequence[Mapping[str, Any]], region: str
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            float(row["val_acc"]), int(row["_search_seed"]), int(row["_step"]),
            str(row["candidate_fingerprint"]),
        ),
    )
    segments = np.array_split(np.arange(len(ordered)), 10)
    selected = []
    seed_counts = {seed: 0 for seed in range(5)}
    for segment_index, indices in enumerate(segments):
        segment = [ordered[int(index)] for index in indices]
        preferred_seed = segment_index % 5
        preferred = [row for row in segment if int(row["_search_seed"]) == preferred_seed]
        if not preferred:
            minimum = min(seed_counts[int(row["_search_seed"])] for row in segment)
            preferred = [row for row in segment if seed_counts[int(row["_search_seed"])] == minimum]
        chosen = min(
            preferred,
            key=lambda row: _canonical_hash(
                {
                    "rule": "search_history_segment_v1",
                    "region": region,
                    "segment": segment_index,
                    "fingerprint": row["architecture_fingerprint"],
                }
            ),
        )
        chosen["selection_segment"] = segment_index
        chosen["segment_size"] = len(segment)
        selected.append(chosen)
        seed_counts[int(chosen["_search_seed"])] += 1
    return selected


def _search_history_candidates(
    histories: Sequence[Mapping[str, Any]], excluded: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    representatives: dict[str, Mapping[str, Any]] = {}
    for row in sorted(
        histories,
        key=lambda value: (
            int(value["_search_seed"]), int(value["_step"]),
            str(value["candidate_fingerprint"]),
        ),
    ):
        fp = str(row["architecture_fingerprint"])
        representatives.setdefault(fp, row)
    eligible = [row for fp, row in representatives.items() if fp not in excluded]
    region_rows = _assign_complete_regions(eligible)
    selected = []
    for region in ("middle", "lower"):
        pool = [row for row in region_rows if row["exclusive_region"] == region]
        selected.extend(_segment_select(pool, region))
    output = []
    for row in selected:
        config = normalize_architecture_config(row)
        candidate = {
            "config_id": f"history_{row['exclusive_region']}_s{row['_search_seed']}_step{row['_step']:03d}",
            "source": "search_history_expansion",
            "source_search_seed": int(row["_search_seed"]),
            "source_step": int(row["_step"]),
            "candidate_rank": None,
            "candidate_fingerprint": str(row["candidate_fingerprint"]),
            "architecture_fingerprint": str(row["architecture_fingerprint"]),
            "hp_fingerprint": str(row["hp_fingerprint"]),
            "operations": config["operations"],
            "edges": config["edges"],
            "z_search": list(row["z_search"][:16]),
            "z_arch": list(row["z_arch"][:12]),
            "raw_hp_vector": list(row["z_search"][12:16]),
            "lr": float(row["lr"]),
            "dropout": float(row["dropout"]),
            "hidden_dim": int(row["hidden_dim"]),
            "l2": float(row["l2"]),
            "gat_heads": int(row.get("gat_heads", 1)),
            "sage_aggr": str(row.get("sage_aggr", "mean")),
            "gin_eps": float(row.get("gin_eps", 0.0)),
            "gat_heads_by_layer": row.get("gat_heads_by_layer"),
            "sage_aggr_by_layer": row.get("sage_aggr_by_layer"),
            "gin_eps_by_layer": row.get("gin_eps_by_layer"),
            "normalization": "none",
            "realized_hp": {
                field: row.get(field)
                for field in ("lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps", "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer")
            },
            "original_single_validation": float(row["val_acc"]),
            "selection_region": str(row["exclusive_region"]),
            "selection_segment": int(row["selection_segment"]),
            "segment_size": int(row["segment_size"]),
            "selection_reason": "one fixed-hash representative from rank segment with pre-registered search-seed balance",
            "representative_rule": "earliest (search_seed, step, candidate_fingerprint) per architecture",
            "representative_percentile": float(row["representative_percentile"]),
            "time_index": int(row["_search_seed"]) * 500 + int(row["_step"]),
        }
        output.append(candidate)
    audit = {
        "representative_architecture_count_before_exclusion": len(representatives),
        "eligible_architecture_count": len(eligible),
        "exclusive_region_counts": {
            region: sum(row["exclusive_region"] == region for row in region_rows)
            for region in ("lower", "middle", "upper")
        },
        "exclusive_region_count_sum": len(region_rows),
        "selected_region_counts": {
            region: sum(row["selection_region"] == region for row in output)
            for region in ("middle", "lower")
        },
        "selected_search_seed_counts": {
            str(seed): sum(row["source_search_seed"] == seed for row in output)
            for seed in range(5)
        },
    }
    if len(output) != 20 or len({row["architecture_fingerprint"] for row in output}) != 20:
        raise AssertionError("history expansion must contain twenty unique architectures")
    if audit["exclusive_region_count_sum"] != audit["eligible_architecture_count"]:
        raise AssertionError("exclusive history regions are incomplete")
    return output, audit


def _baseline_statistics(
    confirmation_rows: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 2000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config_lookup = {
        (int(row["source_search_seed"]), int(row["candidate_rank"]), str(row["candidate_fingerprint"])): row
        for row in configs
    }
    grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in confirmation_rows:
        grouped[(int(row["search_seed"]), int(row["candidate_rank"]), str(row["candidate_fingerprint"]))].append(row)
    stats = []
    for key, group in sorted(grouped.items()):
        values = [float(row["val_acc"]) for row in sorted(group, key=lambda row: int(row["seed"]))]
        config = config_lookup[key]
        stats.append(
            {
                "config_id": config["config_id"],
                "architecture_fingerprint": config["architecture_fingerprint"],
                "hp_fingerprint": config["hp_fingerprint"],
                "seed_count": len(values),
                "validation_mean": statistics.fmean(values),
                "validation_sample_std": statistics.stdev(values),
                **_t_interval(values),
            }
        )
    # One stable configuration per architecture, matching the pre-registered fact.
    first_by_arch: dict[str, Mapping[str, Any]] = {}
    for config in sorted(
        configs,
        key=lambda row: (
            int(row["source_search_seed"]), int(row["candidate_rank"]),
            int(row["source_step"]), str(row["candidate_fingerprint"]),
        ),
    ):
        first_by_arch.setdefault(str(config["architecture_fingerprint"]), config)
    rows_by_config = {
        config["config_id"]: sorted(
            grouped[(int(config["source_search_seed"]), int(config["candidate_rank"]), str(config["candidate_fingerprint"]))],
            key=lambda row: int(row["seed"]),
        )
        for config in configs
    }
    architecture_rows = []
    odd, even = [], []
    for fingerprint, config in sorted(first_by_arch.items()):
        rows = rows_by_config[config["config_id"]]
        odd_mean = statistics.fmean(float(row["val_acc"]) for row in rows if int(row["seed"]) % 2)
        even_mean = statistics.fmean(float(row["val_acc"]) for row in rows if int(row["seed"]) % 2 == 0)
        odd.append(odd_mean)
        even.append(even_mean)
        architecture_rows.append(
            {
                "architecture_fingerprint": fingerprint,
                "stable_config_id": config["config_id"],
                "odd_seed_mean": odd_mean,
                "even_seed_mean": even_mean,
            }
        )
    tau = float(kendalltau(odd, even).statistic)
    rng = np.random.default_rng(20260821)
    bootstrap = []
    odd_array, even_array = np.asarray(odd), np.asarray(even)
    for _ in range(bootstrap_replicates):
        indices = rng.integers(0, len(odd), len(odd))
        value = kendalltau(odd_array[indices], even_array[indices]).statistic
        if np.isfinite(value):
            bootstrap.append(float(value))
    summary = {
        "configuration_count": len(stats),
        "independent_architecture_count": len(architecture_rows),
        "ci95_half_width_median": float(np.median([row["ci95_half_width"] for row in stats])),
        "configurations_half_width_le_0_006": sum(row["ci95_half_width"] <= 0.006 for row in stats),
        "odd_even_kendall": tau,
        "odd_even_bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "odd_even_bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "stable_duplicate_resolution": "first config by (search_seed, candidate_rank, search_step, candidate_fingerprint)",
    }
    return stats, architecture_rows, summary


def _encode_catalog(model: Any, rows: Sequence[dict[str, Any]]) -> None:
    cache: dict[str, list[float]] = {}
    for row in rows:
        fingerprint = str(row["architecture_fingerprint"])
        if fingerprint not in cache:
            cache[fingerprint] = encode_architecture_mu(
                model, {"operations": row["operations"], "edges": row["edges"]}
            )
        row["mu_arch"] = cache[fingerprint]


def prepare(
    history_paths: Sequence[Path],
    confirmation_paths: Sequence[Path],
    stage3_results: Path,
    stage3_selection_path: Path,
    checkpoint: Path,
    seed_manifest: Path,
    output: Path,
    code_paths: Sequence[Path],
) -> dict[str, Any]:
    if (output / "task_plan.json").exists():
        raise FileExistsError("task_plan.json already exists; use a new versioned output directory")
    output.mkdir(parents=True, exist_ok=True)
    histories = _read_histories(history_paths)
    confirmation_rows = _read_confirmation(confirmation_paths)
    confirmation_protocol_configs = []
    for path in confirmation_paths:
        config_path = path.parent / "final_eval_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing confirmation protocol config: {config_path}")
        config = _load_json(config_path)
        confirmation_protocol_configs.append({"path": str(config_path.resolve()), "config": config})
    history_lookup = {
        (int(row["_search_seed"]), str(row["candidate_fingerprint"])): row
        for row in histories
    }
    configs = _confirmation_configs(confirmation_rows, history_lookup)
    baseline_stats, baseline_arch_rows, baseline_summary = _baseline_statistics(
        confirmation_rows, configs
    )
    stage3_selection = _load_json(stage3_selection_path)
    if not isinstance(stage3_selection, list):
        raise ValueError("Stage 3 selection is not a list")
    confirmed_fingerprints = {row["architecture_fingerprint"] for row in configs}
    stage3_fingerprints = {str(row["historical_fingerprint"]) for row in stage3_selection}
    facts = {
        "history_row_count": len(histories),
        "history_architecture_count": len({row["architecture_fingerprint"] for row in histories}),
        "history_architecture_hp_count": len({(row["architecture_fingerprint"], row["hp_fingerprint"]) for row in histories}),
        "confirmation_seed_row_count": len(confirmation_rows),
        "confirmation_configuration_count": len(configs),
        "confirmation_architecture_count": len(confirmed_fingerprints),
        "stage3_historical_configuration_count": len(stage3_selection),
        "stage3_historical_architecture_count": len(stage3_fingerprints),
        "confirmation_stage3_architecture_overlap": len(confirmed_fingerprints & stage3_fingerprints),
        "confirmation_stage3_architecture_union": len(confirmed_fingerprints | stage3_fingerprints),
        "baseline_statistics": baseline_summary,
    }
    expected = {
        "history_row_count": 2500,
        "history_architecture_count": 1199,
        "history_architecture_hp_count": 2478,
        "confirmation_seed_row_count": 1500,
        "confirmation_configuration_count": 50,
        "confirmation_architecture_count": 34,
        "stage3_historical_configuration_count": 18,
        "stage3_historical_architecture_count": 16,
        "confirmation_stage3_architecture_overlap": 1,
        "confirmation_stage3_architecture_union": 49,
    }
    mismatches = {
        field: {"expected": value, "actual": facts[field]}
        for field, value in expected.items() if facts[field] != value
    }
    context_values = {
        tuple(str(row[field]) for field in EXPECTED_CONTEXT)
        for row in confirmation_rows
    }
    expected_context_tuple = tuple(str(EXPECTED_CONTEXT[field]) for field in EXPECTED_CONTEXT)
    context_matches = context_values == {expected_context_tuple}
    confirmation_protocol_matches = all(
        int(item["config"].get("eval_epochs", -1)) == 300
        and int(item["config"].get("patience", -1)) == 80
        and item["config"].get("hp_mode") == "global4"
        and item["config"].get("confirmation_seeds") == list(ALL_SEEDS)
        and item["config"].get("seed_scheme") == "common_confirmation_seed_manifest_v1"
        and item["config"].get("dataset_context", {}).get("canonical_name") == "cora"
        and item["config"].get("dataset_context", {}).get("dataset_content_fingerprint") == EXPECTED_CONTEXT["dataset_content_fingerprint"]
        and item["config"].get("dataset_context", {}).get("split_fingerprint") == EXPECTED_CONTEXT["split_fingerprint"]
        for item in confirmation_protocol_configs
    )
    protocol = {
        **EXPECTED_CONTEXT,
        **PROTOCOL,
        "common_seeds": list(ALL_SEEDS),
        "seed_manifest": str(seed_manifest.resolve()),
        "seed_manifest_sha256": sha256_file(seed_manifest),
        "same_core_training_entry": "eval_utils.train_and_eval_arch via run_unified_label_expansion.py",
        "test_selection_policy": "test accuracy is recorded only in sealed audit rows and never used for selection or labels",
        "context_matches_all_confirmation_rows": context_matches,
        "confirmation_protocol_configs_match": confirmation_protocol_matches,
        "confirmation_protocol_configs": confirmation_protocol_configs,
    }
    _dump_json(output / "source_overlap_audit.json", {"known_fact_audit": facts, "expected": expected, "mismatches": mismatches})
    _dump_json(output / "protocol_compatibility.json", protocol)
    (output / "source_overlap_audit.md").write_text(
        "# Source overlap audit\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in facts.items())
        + f"\n\nMismatches: `{mismatches}`\n",
        encoding="utf-8",
    )
    if mismatches or not context_matches or not confirmation_protocol_matches:
        _dump_json(
            output / "protocol_or_data_inconsistency.json",
            {
                "classification": "PROTOCOL_OR_DATA_INCONSISTENCY",
                "mismatches": mismatches,
                "context_matches": context_matches,
                "confirmation_protocol_matches": confirmation_protocol_matches,
            },
        )
        raise ValueError("known input counts or confirmation protocol do not match")

    bridge = _bridge_candidates(stage3_selection, confirmed_fingerprints)
    excluded = confirmed_fingerprints | {row["architecture_fingerprint"] for row in bridge}
    history_selected, region_audit = _search_history_candidates(histories, excluded)
    selected = bridge + history_selected
    selected_fingerprints = [row["architecture_fingerprint"] for row in selected]
    if len(selected) != 26 or len(set(selected_fingerprints)) != 26:
        raise AssertionError("selected expansion must contain 26 unique architectures")
    if set(selected_fingerprints) & confirmed_fingerprints:
        raise AssertionError("new selection overlaps confirmed architecture fingerprints")
    sorted_fingerprints = sorted(selected_fingerprints)
    gpu_by_fp = {fp: index % 2 for index, fp in enumerate(sorted_fingerprints)}
    for row in selected:
        row["assigned_gpu"] = gpu_by_fp[row["architecture_fingerprint"]]
        row["initial_seeds"] = list(INITIAL_SEEDS)
        row["adaptive_seed_rule"] = "run seeds 15-29 iff 0-14 validation mean t-CI half-width > 0.006"
    if {gpu: sum(row["assigned_gpu"] == gpu for row in selected) for gpu in (0, 1)} != {0: 13, 1: 13}:
        raise AssertionError("alternating fingerprint shards must contain 13 configurations each")

    model = load_joint_vae(checkpoint, "global4", device=torch.device("cpu"))
    _encode_catalog(model, configs)
    _encode_catalog(model, selected)
    _dump_json(output / "baseline_50_configurations.json", configs)
    _dump_csv(output / "baseline_50_configurations.csv", configs)
    _dump_json(output / "baseline_50_configuration_statistics.json", baseline_stats)
    _dump_csv(output / "baseline_50_configuration_statistics.csv", baseline_stats)
    _dump_json(output / "baseline_34_architecture_ordering.json", baseline_arch_rows)
    _dump_json(output / "selected_26_architectures.json", selected)
    _dump_csv(output / "selected_26_architectures.csv", selected)
    _dump_json(output / "history_region_audit.json", region_audit)
    for gpu in (0, 1):
        _dump_json(
            output / f"gpu{gpu}_initial_shard_plan.json",
            {
                "gpu": gpu,
                "config_ids": [row["config_id"] for row in selected if row["assigned_gpu"] == gpu],
                "architecture_fingerprints": [row["architecture_fingerprint"] for row in selected if row["assigned_gpu"] == gpu],
                "seeds": list(INITIAL_SEEDS),
                "expected_training_count": 13 * len(INITIAL_SEEDS),
            },
        )
    protocol_config_paths = [path.parent / "final_eval_config.json" for path in confirmation_paths]
    input_paths = [*history_paths, *confirmation_paths, *protocol_config_paths, stage3_results, stage3_selection_path, checkpoint, seed_manifest, *code_paths]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"task plan inputs are missing: {missing}")
    initial_tasks = [
        {
            "source": row["source"],
            "config_id": row["config_id"],
            "architecture_fingerprint": row["architecture_fingerprint"],
            "hp_fingerprint": row["hp_fingerprint"],
            "seed": seed,
            "assigned_gpu": row["assigned_gpu"],
        }
        for row in sorted(selected, key=lambda value: value["architecture_fingerprint"])
        for seed in INITIAL_SEEDS
    ]
    task_keys = {
        (row["source"], row["architecture_fingerprint"], row["hp_fingerprint"], row["seed"])
        for row in initial_tasks
    }
    if len(initial_tasks) != 390 or len(task_keys) != 390:
        raise AssertionError("initial task plan must contain 390 unique seed-level keys")
    task_plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format_version": "vae_label_expansion_global4_v1",
        "diagnostic_only": True,
        "analysis_code_frozen_before_plan": True,
        "protocol": protocol,
        "selection_file": str((output / "selected_26_architectures.json").resolve()),
        "selection_sha256": sha256_file(output / "selected_26_architectures.json"),
        "configuration_count": 26,
        "independent_architecture_count": 26,
        "initial_seeds": list(INITIAL_SEEDS),
        "initial_training_count": 390,
        "initial_tasks": initial_tasks,
        "sharding": "sort architecture_fingerprint, alternate GPU 0/1; all seeds of a config stay together",
        "gpu_configuration_counts": {str(gpu): sum(row["assigned_gpu"] == gpu for row in selected) for gpu in (0, 1)},
        "adaptive_rule": {
            "statistic": "95% t interval half-width of validation mean after seeds 0-14",
            "threshold": CI_HALF_WIDTH_THRESHOLD,
            "comparison": ">",
            "additional_seeds": list(ADAPTIVE_SEEDS),
            "retain_high_noise_after_30": True,
        },
        "retry_rule": {"maximum_retries_after_initial_attempt": 2, "record_each_failure": True},
        "merge_key": ["source", "architecture_fingerprint", "hp_fingerprint", "seed"],
        "representation_gate": REPRESENTATION_GATE,
        "input_sha256": {str(path.resolve()): sha256_file(path) for path in input_paths},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True).stdout.strip(),
    }
    _dump_json(output / "task_plan.json", task_plan)
    (output / "task_plan.sha256").write_text(
        f"{sha256_file(output / 'task_plan.json')}  task_plan.json\n", encoding="utf-8"
    )
    return task_plan


def _new_result_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["source"]), str(row["architecture_fingerprint"]),
        str(row["hp_fingerprint"]), int(row["seed"]),
    )


def adaptive_plan(
    selected_path: Path,
    initial_results: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    selected = _load_json(selected_path)
    rows = [row for path in initial_results for row in _load_jsonl(path)]
    expected = {
        (row["source"], row["architecture_fingerprint"], row["hp_fingerprint"], seed)
        for row in selected for seed in INITIAL_SEEDS
    }
    by_key = {_new_result_key(row): row for row in rows}
    if len(rows) != 390 or len(by_key) != 390 or set(by_key) != expected:
        raise ValueError("initial results are not exactly 390 unique pre-registered keys")
    statistics_rows = []
    adaptive_ids = []
    for config in selected:
        values = [
            float(by_key[(config["source"], config["architecture_fingerprint"], config["hp_fingerprint"], seed)]["val_acc"])
            for seed in INITIAL_SEEDS
            if by_key[(config["source"], config["architecture_fingerprint"], config["hp_fingerprint"], seed)].get("terminal_status") == "success"
        ]
        interval = _t_interval(values)
        complete_initial_labels = len(values) == len(INITIAL_SEEDS)
        needs_more = bool(
            complete_initial_labels
            and interval["ci95_half_width"] is not None
            and interval["ci95_half_width"] > CI_HALF_WIDTH_THRESHOLD
        )
        if not complete_initial_labels:
            decision = "NOT_TRIGGERED_INCOMPLETE_INITIAL_LABELS"
        elif needs_more:
            decision = "TRIGGERED_HALF_WIDTH_GT_0_006"
        else:
            decision = "NOT_TRIGGERED_HALF_WIDTH_LE_0_006"
        if needs_more:
            adaptive_ids.append(config["config_id"])
        statistics_rows.append(
            {
                "config_id": config["config_id"],
                "assigned_gpu": config["assigned_gpu"],
                "successful_seed_count": len(values),
                "validation_mean": statistics.fmean(values) if values else None,
                "validation_sample_std": statistics.stdev(values) if len(values) > 1 else None,
                **interval,
                "adaptive_seeds_required": needs_more,
                "adaptive_decision": decision,
            }
        )
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": "seeds 15-29 iff all seeds 0-14 succeed and validation t-CI half-width > 0.006",
        "threshold": CI_HALF_WIDTH_THRESHOLD,
        "adaptive_config_ids": adaptive_ids,
        "adaptive_configuration_count": len(adaptive_ids),
        "additional_seeds": list(ADAPTIVE_SEEDS),
        "additional_training_count": len(adaptive_ids) * len(ADAPTIVE_SEEDS),
        "by_gpu": {
            str(gpu): [row["config_id"] for row in selected if row["config_id"] in adaptive_ids and row["assigned_gpu"] == gpu]
            for gpu in (0, 1)
        },
        "initial_statistics": statistics_rows,
        "initial_result_sha256": {str(path.resolve()): sha256_file(path) for path in initial_results},
    }
    _dump_json(output / "adaptive_seed_plan.json", plan)
    _dump_csv(output / "initial_15seed_statistics.csv", statistics_rows)
    (output / "adaptive_seed_plan.sha256").write_text(
        f"{sha256_file(output / 'adaptive_seed_plan.json')}  adaptive_seed_plan.json\n",
        encoding="utf-8",
    )
    for gpu in (0, 1):
        _dump_json(
            output / f"gpu{gpu}_adaptive_config_ids.json",
            plan["by_gpu"][str(gpu)],
        )
    return plan


def _existing_seed_rows(
    confirmation_paths: Sequence[Path],
    baseline_configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw = _read_confirmation(confirmation_paths)
    lookup = {
        (int(config["source_search_seed"]), int(config["candidate_rank"]), str(config["candidate_fingerprint"])): config
        for config in baseline_configs
    }
    output = []
    for row in raw:
        config = lookup[(int(row["search_seed"]), int(row["candidate_rank"]), str(row["candidate_fingerprint"]))]
        output.append(
            {
                "source": config["source"],
                "config_id": config["config_id"],
                "architecture_fingerprint": config["architecture_fingerprint"],
                "hp_fingerprint": config["hp_fingerprint"],
                "seed": int(row["seed"]),
                "val_acc": float(row["val_acc"]),
                "sealed_test_acc": float(row["test_acc"]),
                "test_field_status": "SEALED_AUDIT_ONLY_NOT_USED",
                "valid": bool(row["valid"]),
                "terminal_status": "success" if row["valid"] else "invalid",
                "dataset": row["dataset"],
                "dataset_content_fingerprint": row["dataset_content_fingerprint"],
                "split_fingerprint": row["split_fingerprint"],
                "graph_transform": row["graph_transform"],
                "training_mode": row["training_mode"],
                "hp_mode": row["hp_mode"],
                "normalization": row["normalization"],
                "eval_epochs": 300,
                "patience": 80,
            }
        )
    return output


def _configuration_statistics(
    rows: Sequence[Mapping[str, Any]],
    catalogs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)
    output = []
    for config_id, group in sorted(grouped.items()):
        values = [
            float(row["val_acc"]) for row in group
            if row.get("terminal_status") == "success" and row.get("val_acc") is not None
        ]
        interval = _t_interval(values)
        config = catalogs[config_id]
        output.append(
            {
                "config_id": config_id,
                "source": config["source"],
                "architecture_group_id": config["architecture_fingerprint"],
                "architecture_fingerprint": config["architecture_fingerprint"],
                "hp_fingerprint": config["hp_fingerprint"],
                "expected_seed_count": len(group),
                "successful_seed_count": len(values),
                "validation_mean": statistics.fmean(values) if values else None,
                "validation_sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "validation_median": float(np.median(values)) if values else None,
                "validation_mad": float(np.median(np.abs(np.asarray(values) - np.median(values)))) if values else None,
                **interval,
                "high_noise": bool(interval["ci95_half_width"] is None or interval["ci95_half_width"] > CI_HALF_WIDTH_THRESHOLD),
                "invalid_count": sum(row.get("terminal_status") == "invalid" for row in group),
                "failed_count": sum(row.get("terminal_status") == "failed" for row in group),
                "retry_count": sum(int(row.get("retry_count", 0)) for row in group),
            }
        )
    return output


def _ordering_gate(
    seed_rows: Sequence[Mapping[str, Any]],
    baseline_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first_by_arch: dict[str, Mapping[str, Any]] = {}
    for config in sorted(
        baseline_configs,
        key=lambda row: (
            int(row["source_search_seed"]), int(row["candidate_rank"]),
            int(row["source_step"]), str(row["candidate_fingerprint"]),
        ),
    ):
        first_by_arch.setdefault(str(config["architecture_fingerprint"]), config)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["config_id"])].append(row)
    odd, even = [], []
    for config in first_by_arch.values():
        rows = grouped[config["config_id"]]
        odd.append(statistics.fmean(float(row["val_acc"]) for row in rows if int(row["seed"]) % 2))
        even.append(statistics.fmean(float(row["val_acc"]) for row in rows if int(row["seed"]) % 2 == 0))
    estimate = float(kendalltau(odd, even).statistic)
    rng = np.random.default_rng(20260821)
    samples = []
    odd_array, even_array = np.asarray(odd), np.asarray(even)
    for _ in range(2000):
        indices = rng.integers(0, len(odd), len(odd))
        value = kendalltau(odd_array[indices], even_array[indices]).statistic
        if np.isfinite(value):
            samples.append(float(value))
    low, high = float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
    return {
        "architecture_count": len(odd),
        "estimate": estimate,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "passed": bool(estimate >= 0.30 and low > 0.0),
        "stable_config_rule": "first by (search_seed, candidate_rank, search_step, candidate_fingerprint)",
    }


def _paired_representation_increments(
    split_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (row["protocol"], row["split_id"], row["representation"], row["metric"]): float(row["value"])
        for row in split_rows
    }
    output = []
    for protocol in sorted({str(row["protocol"]) for row in split_rows}):
        split_ids = sorted({str(row["split_id"]) for row in split_rows if row["protocol"] == protocol})
        for representation in ("Ze", "D", "H"):
            for metric in METRICS:
                values = []
                for split_id in split_ids:
                    baseline = lookup.get((protocol, split_id, "Zs", metric))
                    current = lookup.get((protocol, split_id, representation, metric))
                    if baseline is None or current is None:
                        continue
                    values.append(
                        baseline - current if metric == "top_10_regret" else current - baseline
                    )
                rng = np.random.default_rng(
                    20260821 + 101 * REPRESENTATIONS.index(representation) + 17 * METRICS.index(metric)
                )
                bootstrap = []
                if values:
                    array = np.asarray(values)
                    bootstrap = rng.choice(array, size=(2000, len(array)), replace=True).mean(axis=1)
                output.append(
                    {
                        "protocol": protocol,
                        "representation": representation,
                        "metric": metric,
                        "improvement_direction": "positive_is_better",
                        "paired_split_count": len(values),
                        "estimate": float(np.mean(values)) if values else None,
                        "ci95_low": float(np.quantile(bootstrap, 0.025)) if len(bootstrap) else None,
                        "ci95_high": float(np.quantile(bootstrap, 0.975)) if len(bootstrap) else None,
                    }
                )
    return output


def _representation_candidates(
    metrics: Sequence[Mapping[str, Any]],
    increments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary = {
        (row["representation"], row["metric"]): row
        for row in metrics if row["protocol"] == "group_repeated"
    }
    delta = {
        (row["representation"], row["metric"]): row
        for row in increments if row["protocol"] == "group_repeated"
    }
    candidates = []
    for representation in ("Ze", "D", "H"):
        pairwise = primary.get((representation, "top_pairwise_accuracy"))
        top_kendall = primary.get((representation, "top_region_kendall"))
        coverage = primary.get((representation, "prediction_interval_coverage"))
        late_delta = delta.get((representation, "late_50_kendall"))
        pair_delta = delta.get((representation, "top_pairwise_accuracy"))
        if any(row is None or row.get("estimate") is None for row in (pairwise, top_kendall, coverage, late_delta, pair_delta)):
            continue
        relative_pass = bool(
            (late_delta["ci95_low"] is not None and late_delta["ci95_low"] > 0.0)
            or (pair_delta["ci95_low"] is not None and pair_delta["ci95_low"] > 0.0)
        )
        passed = bool(
            pairwise["estimate"] >= 0.60
            and pairwise["ci95_low"] is not None
            and pairwise["ci95_low"] > 0.50
            and top_kendall["ci95_high"] is not None
            and top_kendall["ci95_high"] >= 0.0
            and 0.80 <= coverage["estimate"] <= 0.98
            and relative_pass
        )
        candidates.append(
            {
                "representation": representation,
                "passed": passed,
                "top_pairwise_estimate": pairwise["estimate"],
                "top_pairwise_ci95_low": pairwise["ci95_low"],
                "top_kendall_ci95_high": top_kendall["ci95_high"],
                "prediction_interval_coverage": coverage["estimate"],
                "relative_late_ci95_low": late_delta["ci95_low"],
                "relative_top_pairwise_ci95_low": pair_delta["ci95_low"],
            }
        )
    return candidates


def _contains_test_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any("test" in str(key).lower() or _contains_test_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_test_key(item) for item in value)
    return False


def analyze(
    output: Path,
    confirmation_paths: Sequence[Path],
    initial_results: Sequence[Path],
    adaptive_results: Sequence[Path],
) -> dict[str, Any]:
    baseline_configs = _load_json(output / "baseline_50_configurations.json")
    selected = _load_json(output / "selected_26_architectures.json")
    adaptive = _load_json(output / "adaptive_seed_plan.json")
    catalogs = {row["config_id"]: row for row in [*baseline_configs, *selected]}
    existing_rows = _existing_seed_rows(confirmation_paths, baseline_configs)
    new_rows = [row for path in [*initial_results, *adaptive_results] if path.is_file() for row in _load_jsonl(path)]
    selected_by_id = {row["config_id"]: row for row in selected}
    for row in new_rows:
        if row["config_id"] not in selected_by_id:
            raise ValueError(f"new result config {row['config_id']} is outside frozen selection")
    expected_new = set()
    adaptive_ids = set(adaptive["adaptive_config_ids"])
    for config in selected:
        seeds = ALL_SEEDS if config["config_id"] in adaptive_ids else INITIAL_SEEDS
        expected_new.update(
            (config["source"], config["architecture_fingerprint"], config["hp_fingerprint"], seed)
            for seed in seeds
        )
    new_by_key = {_new_result_key(row): row for row in new_rows}
    if len(new_rows) != len(new_by_key) or set(new_by_key) != expected_new:
        raise ValueError(
            f"new result merge mismatch duplicates={len(new_rows)-len(new_by_key)} "
            f"missing={len(expected_new-set(new_by_key))} unexpected={len(set(new_by_key)-expected_new)}"
        )
    for config in selected:
        group = [row for row in new_rows if row["config_id"] == config["config_id"]]
        for field in ("architecture_fingerprint", "hp_fingerprint", "operations", "edges", "lr", "dropout", "hidden_dim", "l2", "dataset_content_fingerprint", "split_fingerprint", "eval_epochs", "patience", "normalization"):
            values = {json.dumps(row.get(field), sort_keys=True) for row in group}
            if len(values) != 1:
                raise ValueError(f"seed invariant {field} changed for {config['config_id']}")
        first = group[0]
        if any(str(first[field]) != str(EXPECTED_CONTEXT[field]) for field in EXPECTED_CONTEXT):
            raise ValueError(f"protocol context mismatch for {config['config_id']}")
        if int(first["eval_epochs"]) != 300 or int(first["patience"]) != 80:
            raise ValueError(f"training budget mismatch for {config['config_id']}")
    audit_rows = [*existing_rows]
    for row in new_rows:
        audit = dict(row)
        if "test_acc" in audit:
            audit["sealed_test_acc"] = audit.pop("test_acc")
        audit["test_field_status"] = "SEALED_AUDIT_ONLY_NOT_USED"
        audit_rows.append(audit)
    _dump_jsonl(output / "unified_seed_level_audit.jsonl", audit_rows)

    config_stats = _configuration_statistics(audit_rows, catalogs)
    _dump_json(output / "unified_76_configuration_statistics.json", config_stats)
    _dump_csv(output / "unified_76_configuration_statistics.csv", config_stats)
    architecture_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in config_stats:
        architecture_groups[row["architecture_fingerprint"]].append(row)
    architecture_stats = [
        {
            "architecture_group_id": fingerprint,
            "configuration_count": len(group),
            "config_ids": [row["config_id"] for row in group],
            "mean_of_configuration_validation_means": statistics.fmean(float(row["validation_mean"]) for row in group),
        }
        for fingerprint, group in sorted(architecture_groups.items())
    ]
    _dump_json(output / "unified_60_architecture_group_statistics.json", architecture_stats)
    _dump_csv(output / "unified_60_architecture_group_statistics.csv", architecture_stats)

    training_rows = []
    stats_by_id = {row["config_id"]: row for row in config_stats}
    for config_id, config in sorted(catalogs.items()):
        stats = stats_by_id[config_id]
        training_rows.append(
            {
                "config_id": config_id,
                "source": config["source"],
                "architecture_group_id": config["architecture_fingerprint"],
                "architecture_fingerprint": config["architecture_fingerprint"],
                "hp_fingerprint": config["hp_fingerprint"],
                "operations": config["operations"],
                "edges": config["edges"],
                "z_search": config["z_search"],
                "z_arch": config["z_arch"],
                "mu_arch": config["mu_arch"],
                "raw_hp_vector": config["raw_hp_vector"],
                "lr": config["lr"],
                "dropout": config["dropout"],
                "hidden_dim": config["hidden_dim"],
                "l2": config["l2"],
                "normalization": "none",
                "validation_mean": stats["validation_mean"],
                "validation_sample_std": stats["validation_sample_std"],
                "validation_ci95_low": stats["ci95_low"],
                "validation_ci95_high": stats["ci95_high"],
                "validation_ci95_half_width": stats["ci95_half_width"],
                "seed_count": stats["successful_seed_count"],
                "high_noise": stats["high_noise"],
                "time_index": config["time_index"],
            }
        )
    if _contains_test_key(training_rows):
        raise AssertionError("unified training data contains a test-named field")
    _dump_json(output / "unified_training_data_no_test.json", training_rows)
    _dump_csv(output / "unified_training_data_no_test.csv", training_rows)

    high_noise = [row for row in config_stats if row["high_noise"]]
    _dump_json(output / "high_noise_configurations.json", high_noise)
    _dump_csv(output / "high_noise_configurations.csv", high_noise)
    # Bridge report compares protocol means only; no test metric is loaded.
    stage3_rows = _load_jsonl(REPO_ROOT / "results/posthoc/stage3-noise-labels-global4-v1/stage3_merged_results.jsonl")
    bridge_report = []
    audit_by_config_seed = {
        (str(row["config_id"]), int(row["seed"])): row for row in audit_rows
    }
    for config in selected:
        if config["source"] != "stage3_protocol_bridge":
            continue
        old_rows = sorted(
            [row for row in stage3_rows if row["point_id"] == config["candidate_fingerprint"] and row["decoder_policy"] == "historical_modal3"],
            key=lambda row: int(row["seed"]),
        )
        old = [float(row["val_acc"]) for row in old_rows]
        paired_differences = [
            float(audit_by_config_seed[(config["config_id"], int(row["seed"]))]["val_acc"])
            - float(row["val_acc"])
            for row in old_rows
            if (config["config_id"], int(row["seed"])) in audit_by_config_seed
            and audit_by_config_seed[(config["config_id"], int(row["seed"]))]["terminal_status"] == "success"
        ]
        current = stats_by_id[config["config_id"]]
        bridge_report.append(
            {
                "config_id": config["config_id"],
                "architecture_fingerprint": config["architecture_fingerprint"],
                "old_protocol": {"eval_epochs": 150, "patience": 20, "seed_count": len(old), "validation_mean": statistics.fmean(old)},
                "new_protocol": {"eval_epochs": 300, "patience": 80, "seed_count": current["successful_seed_count"], "validation_mean": current["validation_mean"]},
                "new_minus_old_validation_mean": float(current["validation_mean"]) - statistics.fmean(old),
                "paired_common_seed_count": len(paired_differences),
                "paired_new_minus_old_mean": statistics.fmean(paired_differences),
                "paired_new_minus_old_sample_std": statistics.stdev(paired_differences),
                **{
                    f"paired_new_minus_old_{key}": value
                    for key, value in _t_interval(paired_differences).items()
                },
            }
        )
    _dump_json(output / "protocol_bridge_differences.json", bridge_report)
    _dump_csv(output / "protocol_bridge_differences.csv", bridge_report)

    representation_records = []
    for row in training_rows:
        representation_records.append(
            {
                "point_id": row["config_id"],
                "decoder_policy": "unified_300_80",
                "architecture_fingerprint": row["architecture_fingerprint"],
                "operations": row["operations"],
                "edges": row["edges"],
                "z_arch": row["z_arch"],
                "mu_arch": row["mu_arch"],
                "raw_hp_vector": row["raw_hp_vector"],
                "condition_mask_vector": [1.0, 1.0, 1.0, 1.0],
                "time_index": row["time_index"],
                "label": row["validation_mean"],
            }
        )
    representation_records.sort(key=lambda row: (int(row["time_index"]), str(row["point_id"])))
    protocols = {
        "time_prefix": time_prefix_splits(representation_records, min_train=20, step=10, validation_window=15),
        "group_repeated": grouped_repeated_splits(representation_records, repeats=50, seed=20260821, validation_fraction=0.30),
    }
    split_audit = []
    leakage_count = 0
    for protocol_name, splits in protocols.items():
        for split in splits:
            train_groups = {
                representation_records[index]["architecture_fingerprint"]
                for index in split["train"]
            }
            validation_groups = {
                representation_records[index]["architecture_fingerprint"]
                for index in split["validation"]
            }
            overlap = sorted(train_groups & validation_groups)
            leakage_count += len(overlap)
            split_audit.append(
                {
                    "protocol": protocol_name,
                    "split_id": split["split_id"],
                    "train_observation_count": len(split["train"]),
                    "validation_observation_count": len(split["validation"]),
                    "train_architecture_group_count": len(train_groups),
                    "validation_architecture_group_count": len(validation_groups),
                    "leakage_count": len(overlap),
                    "leaked_architecture_fingerprints": overlap,
                }
            )
    if leakage_count:
        raise ValueError(f"representation split architecture leakage count={leakage_count}")
    _dump_json(output / "representation_split_audit.json", split_audit)
    _dump_csv(output / "representation_split_audit.csv", split_audit)
    all_metrics, all_predictions, all_split_rows = [], [], []
    for protocol_name, splits in protocols.items():
        metrics, predictions, split_rows = evaluate_protocol(
            representation_records, splits, protocol_name, "unified_300_80",
            "validation_seed_mean", 20260821, 2000,
            return_split_metrics=True,
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        all_split_rows.extend(split_rows)
    increments = _paired_representation_increments(all_split_rows)
    candidates = _representation_candidates(all_metrics, increments)
    _dump_json(output / "representation_comparison.json", {"metrics": all_metrics, "paired_increments": increments, "candidate_gates": candidates})
    _dump_csv(output / "representation_metrics.csv", all_metrics)
    _dump_csv(output / "representation_predictions.csv", all_predictions)
    _dump_csv(output / "representation_paired_increments.csv", increments)

    ordering = _ordering_gate(existing_rows, baseline_configs)
    half_widths = [float(row["ci95_half_width"]) for row in config_stats if row["ci95_half_width"] is not None]
    independent_count = len(architecture_groups)
    configuration_count = len(config_stats)
    label_noise_gate = bool(
        float(np.median(half_widths)) <= 0.006
        and sum(value <= 0.010 for value in half_widths) / configuration_count >= 0.75
        and all(row["successful_seed_count"] == row["expected_seed_count"] for row in config_stats)
    )
    ordering_passed = bool(ordering["passed"])
    vae_ready = compute_vae_finetune_ready(
        noise_gate_passed=label_noise_gate,
        independent_architecture_count=independent_count,
        minimum_independent_architectures=60,
        group_leakage_count=leakage_count,
        test_label_used=False,
        ordering_gate_passed=ordering_passed,
    )
    passed = {row["representation"] for row in candidates if row["passed"]}
    if independent_count < 60:
        classification = "INSUFFICIENT_INDEPENDENT_ARCHITECTURES"
        recommendation = "expand_unique_300_80_architecture_labels"
    elif not label_noise_gate:
        classification = "LABEL_NOISE_STILL_TOO_HIGH"
        recommendation = "improve_or_model_validation_label_uncertainty"
    elif "Ze" in passed:
        classification = "PREFER_ENCODED_LATENT"
        recommendation = "controlled_encoded_latent_replay"
    elif passed & {"D", "H"}:
        classification = "PREFER_ARCHITECTURE_NATIVE_SURROGATE"
        recommendation = "architecture_native_surrogate_pilot"
    else:
        classification = "LABEL_POOL_READY_FOR_VAE_PILOT"
        recommendation = "minimal_performance_aware_vae_pilot"
    readiness = {
        "classification": classification,
        "vae_finetune_ready": vae_ready,
        "independent_architecture_count": independent_count,
        "configuration_label_count": configuration_count,
        "label_noise_gate_passed": label_noise_gate,
        "ordering_gate_passed": ordering_passed,
        "group_leakage_count": leakage_count,
        "test_label_used": False,
        "recommended_next_branch": recommendation,
        "numeric_evidence": {
            "validation_ci95_half_width_median": float(np.median(half_widths)),
            "fraction_ci95_half_width_le_0_010": sum(value <= 0.010 for value in half_widths) / configuration_count,
            "high_noise_configuration_count": len(high_noise),
            "ordering": ordering,
            "representation_candidate_gates": candidates,
            "seed_level_audit_count": len(audit_rows),
        },
        "formal_bo_modified": False,
        "vae_checkpoint_modified": False,
    }
    _dump_json(output / "vae_pilot_readiness.json", readiness)
    merge_audit = {
        "baseline_seed_rows": len(existing_rows),
        "new_seed_rows": len(new_rows),
        "total_seed_rows": len(audit_rows),
        "expected_new_keys": len(expected_new),
        "duplicate_new_keys": len(new_rows) - len(new_by_key),
        "missing_new_keys": 0,
        "unexpected_new_keys": 0,
        "configuration_count": configuration_count,
        "independent_architecture_count": independent_count,
        "protocol": {**PROTOCOL, **EXPECTED_CONTEXT},
        "test_training_data_key_scan_passed": not _contains_test_key(training_rows),
    }
    _dump_json(output / "unified_merge_audit.json", merge_audit)
    report = [
        "# Unified 300/80 label expansion", "",
        f"Conclusion: `{classification}`", "",
        f"- Configuration labels: `{configuration_count}`",
        f"- Independent architectures: `{independent_count}`",
        f"- Median validation t-CI half-width: `{np.median(half_widths)}`",
        f"- Fraction half-width <= 0.010: `{readiness['numeric_evidence']['fraction_ci95_half_width_le_0_010']}`",
        f"- High-noise configurations after adaptive rule: `{len(high_noise)}`",
        f"- High-performance ordering tau: `{ordering['estimate']}`; CI `[{ordering['bootstrap_ci95_low']}, {ordering['bootstrap_ci95_high']}]`",
        f"- Group leakage count: `{leakage_count}`",
        "- Test label used: `false`",
        f"- VAE fine-tune ready: `{vae_ready}`", "",
        f"Recommended next branch: `{recommendation}`", "",
        "This run did not execute formal BO or train/fine-tune a VAE.",
    ]
    (output / "final_goal_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    environment = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-geometric", "numpy", "scipy", "scikit-learn", "python-igraph")
        },
        "code_sha256": {
            str(path.resolve()): sha256_file(path)
            for path in (
                REPO_ROOT / "analyse/vae_label_expansion.py",
                REPO_ROOT / "analyse/run_unified_label_expansion.py",
                REPO_ROOT / "analyse/compare_matched_representations.py",
                REPO_ROOT / "analyse/stage3_noise_labels.py",
                REPO_ROOT / "eval_utils.py",
                REPO_ROOT / "architecture_diagnostics.py",
            )
        },
        "result_sources": {str(path.resolve()): sha256_file(path) for path in [*initial_results, *adaptive_results] if path.is_file()},
    }
    _dump_json(output / "run_environment.json", environment)
    return readiness


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--histories", type=Path, nargs=5, required=True)
    prepare_parser.add_argument("--confirmations", type=Path, nargs=5, required=True)
    prepare_parser.add_argument("--stage3_results", type=Path, required=True)
    prepare_parser.add_argument("--stage3_selection", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--seed_manifest", type=Path, required=True)
    prepare_parser.add_argument("--code", type=Path, nargs="+", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    adaptive_parser = commands.add_parser("adaptive-plan")
    adaptive_parser.add_argument("--selected", type=Path, required=True)
    adaptive_parser.add_argument("--initial_results", type=Path, nargs=2, required=True)
    adaptive_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--confirmations", type=Path, nargs=5, required=True)
    analyze_parser.add_argument("--initial_results", type=Path, nargs=2, required=True)
    analyze_parser.add_argument("--adaptive_results", type=Path, nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.histories, args.confirmations, args.stage3_results,
            args.stage3_selection, args.checkpoint, args.seed_manifest,
            args.output, args.code,
        )
        summary = {
            "configuration_count": result["configuration_count"],
            "initial_training_count": result["initial_training_count"],
            "gpu_configuration_counts": result["gpu_configuration_counts"],
            "selection_sha256": result["selection_sha256"],
        }
    elif args.command == "adaptive-plan":
        result = adaptive_plan(args.selected, args.initial_results, args.output)
        summary = {
            "adaptive_configuration_count": result["adaptive_configuration_count"],
            "additional_training_count": result["additional_training_count"],
            "by_gpu": result["by_gpu"],
        }
    else:
        result = analyze(args.output, args.confirmations, args.initial_results, args.adaptive_results)
        summary = result
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
