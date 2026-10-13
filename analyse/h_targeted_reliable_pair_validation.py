#!/usr/bin/env python3
"""Audit reliable high-performance pairs before targeted frozen-H validation.

Phase A is deliberately read-only with respect to the source experiment.  It
reconstructs the strict experiment's full/group-repeated aggregation directly
from validation-only predictions and writes all diagnostics to a new directory.
No model training, dataset loading, or BO entry point is imported by this
command.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import architecture_fingerprint, structure_feature_vector


FORMAT_VERSION = "h_targeted_reliable_pair_validation_global4_v1"
DATASET_VARIANT = "full"
PROTOCOL = "group_repeated"
MINIMUM_RELIABLE_GAP = 0.006
TOP_FRACTION = 0.20
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260821
MODELS = (
    "frozen_head",
    "frozen_hybrid_head",
    "original_zs_fixed_gp",
    "original_ze_fixed_gp",
    "ft_latent_gp_lambda_0.1",
    "ft_latent_gp_lambda_0.5",
    "ft_latent_gp_lambda_0.9",
)
SHORT_MODEL_NAMES = {
    "frozen_head": "head",
    "frozen_hybrid_head": "h",
    "original_zs_fixed_gp": "zs",
    "original_ze_fixed_gp": "ze",
    "ft_latent_gp_lambda_0.1": "ft_gp_lambda_0_1",
    "ft_latent_gp_lambda_0.5": "ft_gp_lambda_0_5",
    "ft_latent_gp_lambda_0.9": "ft_gp_lambda_0_9",
}
PROTECTED_RELATIVE_PATHS = (
    "bo_phase4.py",
    "eval_utils.py",
    "checkpoints/joint_model_pipeline_global4_best.pth",
    "results/posthoc/performance-aware-vae-strict-regression-global4-v2/task_plan.json",
    "results/posthoc/performance-aware-vae-strict-regression-global4-v2/fixed_split_manifest.json",
    "results/posthoc/performance-aware-vae-strict-regression-global4-v2/all_validation_predictions.csv",
    "results/posthoc/performance-aware-vae-strict-regression-global4-v2/final_classification.json",
    "results/posthoc/vae-label-expansion-global4-v1/unified_training_data_no_test.json",
)
HISTORY_GLOB = "results/cora-original-ceiling-hunt-v1/original/search_seed*/history_final.json"
FORBIDDEN_CANDIDATE_FIELD_TOKENS = ("val", "validation", "test", "acc", "label", "value")
INITIAL_SEEDS = tuple(range(15))
ADAPTIVE_SEEDS = tuple(range(15, 30))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fields})
    temporary.replace(path)


def protected_hashes(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    result = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path.resolve())] = sha256_file(path)
    return result


def contains_forbidden_candidate_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_CANDIDATE_FIELD_TOKENS):
                return True
            if contains_forbidden_candidate_field(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(contains_forbidden_candidate_field(child) for child in value)
    return False


def existing_architecture_fingerprints_only(path: Path) -> set[str]:
    """Project the old label file to fingerprints without reading label values."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("existing-label input is not a list")
    fingerprints = {str(row["architecture_fingerprint"]) for row in raw}
    if len(fingerprints) != 60:
        raise ValueError(f"expected 60 existing architecture fingerprints, got {len(fingerprints)}")
    return fingerprints


def project_history_candidate(
    source: Mapping[str, Any], *, source_path: Path, source_index: int
) -> tuple[dict[str, Any], list[str]]:
    """Apply a strict allowlist before a history record can enter selection."""

    from architecture_diagnostics import normalize_architecture_config

    masked = sorted(
        key
        for key in source
        if any(token in str(key).lower() for token in FORBIDDEN_CANDIDATE_FIELD_TOKENS)
    )
    config = normalize_architecture_config(
        {"operations": source["operations"], "edges": source["edges"]}
    )
    z_arch = [float(value) for value in source["z_arch"]]
    z_search = [float(value) for value in source["z_search"]]
    raw_hp = z_search[12:16]
    if len(z_arch) != 12 or len(z_search) < 16 or len(raw_hp) != 4:
        raise ValueError("history candidate is not arch12+global4")
    if config["effective_layers"] <= 0:
        raise ValueError("history candidate has no effective operation")
    projected = {
        "source": "historical_phase4_candidate",
        "source_path": str(source_path.resolve()),
        "source_index": int(source_index),
        "source_search_seed": int(source["search_seed"]),
        "source_step": int(source["step"]),
        "operations": list(config["operations"]),
        "edges": [list(edge) for edge in config["edges"]],
        "z_arch": z_arch,
        "raw_hp_vector": raw_hp,
    }
    projected["architecture_fingerprint"] = architecture_fingerprint(projected)
    projected["source_projected_sha256"] = stable_json_hash(projected)
    if contains_forbidden_candidate_field(projected):
        raise AssertionError("allowlisted history projection contains a forbidden label field")
    return projected, masked


def deduplicate_candidates(
    candidates: Sequence[Mapping[str, Any]], excluded_fingerprints: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        dict(row)
        for row in candidates
        if str(row["architecture_fingerprint"]) not in excluded_fingerprints
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[str(row["architecture_fingerprint"])].append(row)
    output = []
    for fingerprint, values in sorted(grouped.items()):
        chosen = min(
            values,
            key=lambda row: stable_json_hash(
                {
                    "rule": "one_hp_per_architecture_v1",
                    "source_path": row["source_path"],
                    "source_index": row["source_index"],
                    "z_arch": row["z_arch"],
                    "raw_hp_vector": row["raw_hp_vector"],
                }
            ),
        )
        output.append(dict(chosen))
    return output, {
        "projected_record_count": len(candidates),
        "projected_unique_architecture_count": len(
            {str(row["architecture_fingerprint"]) for row in candidates}
        ),
        "excluded_existing_record_count": len(candidates) - len(eligible),
        "excluded_existing_unique_architecture_count": len(
            {str(row["architecture_fingerprint"]) for row in candidates}
            & excluded_fingerprints
        ),
        "eligible_record_count_before_one_hp_rule": len(eligible),
        "eligible_unique_architecture_count": len(output),
        "duplicate_hp_records_removed": len(eligible) - len(output),
    }


def _actual_gpus() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        rows.append(
            {
                "physical_index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": int(parts[3]),
            }
        )
    if len(rows) != 2:
        raise RuntimeError(f"expected two GPUs, found {len(rows)}")
    return rows


def _realize_candidate_hp(candidate: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from eval_utils import decode_hp_by_mode

    z_search = torch.tensor(
        [*map(float, candidate["z_arch"]), *map(float, candidate["raw_hp_vector"])],
        dtype=torch.float32,
    )
    hp = decode_hp_by_mode(
        z_search=z_search,
        arch_nz=12,
        config=dict(candidate),
        hp_mode="global4",
        log_lr_min=-4.0,
        log_lr_max=-1.5,
        dropout_min=0.1,
        dropout_max=0.6,
    )
    return {
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
    }


def _ensemble_statistics(values: np.ndarray, within_std: np.ndarray) -> dict[str, float]:
    mean = float(values.mean())
    variance = max(float(np.mean(within_std**2 + values**2) - mean**2), 1e-12)
    std = math.sqrt(variance)
    return {
        "prediction": mean,
        "prediction_std": std,
        "prediction_interval_low": mean - 1.96 * std,
        "prediction_interval_high": mean + 1.96 * std,
        "prediction_source_count": int(values.size),
    }


def frozen_head_h_predictions(
    candidates: Sequence[Mapping[str, Any]], strict_root: Path
) -> list[dict[str, Any]]:
    import torch
    from analyse.performance_aware_vae_strict_regression import StrictPerformanceHead
    from architecture_diagnostics import config_to_igraph, load_joint_vae

    base_checkpoint = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
    model = load_joint_vae(base_checkpoint, "global4", device="cpu")
    graphs = [config_to_igraph(row) for row in candidates]
    with torch.no_grad():
        mu, _ = model.arch_vae.encode(graphs)
    mu_array = mu.detach().cpu().numpy().astype(np.float32)
    structure = np.stack(
        [structure_feature_vector(row, [1.0] * 4)[0] for row in candidates]
    ).astype(np.float32)
    hp = np.asarray([row["raw_hp_vector"] for row in candidates], dtype=np.float32)
    collected: dict[str, list[np.ndarray]] = {"frozen_head": [], "frozen_hybrid_head": []}
    calibration: dict[str, list[float]] = {"frozen_head": [], "frozen_hybrid_head": []}
    checkpoint_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for model_name in collected:
        paths = sorted(
            (strict_root / "checkpoints" / "full" / "group_repeated").glob(
                f"group_*/seed_*/{model_name}/checkpoint.pth"
            )
        )
        if len(paths) != 150:
            raise RuntimeError(f"expected 150 {model_name} checkpoints, found {len(paths)}")
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["arch_vae_state"] is not None:
                raise RuntimeError(f"frozen model checkpoint unexpectedly contains VAE state: {path}")
            scaling = payload["scaling"]
            hp_scaled = (hp - np.asarray(scaling["hp_mean"], dtype=np.float32)) / np.asarray(
                scaling["hp_std"], dtype=np.float32
            )
            parts = [mu_array, hp_scaled]
            if model_name == "frozen_hybrid_head":
                parts.insert(1, structure)
            features = torch.tensor(np.concatenate(parts, axis=1), dtype=torch.float32)
            head = StrictPerformanceHead(int(payload["head_input_dim"]))
            head.load_state_dict(payload["head_state"])
            head.eval()
            with torch.no_grad():
                normalized = head(features).numpy()
            predictions = (
                normalized * float(scaling["y_std"]) + float(scaling["y_mean"])
            ).astype(np.float64)
            collected[model_name].append(predictions)
            calibration[model_name].append(float(payload["calibration_std"]))
            checkpoint_hashes[model_name].append(
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
            )
    output = []
    for index, candidate in enumerate(candidates):
        row = {
            "config_id": candidate["config_id"],
            "architecture_fingerprint": candidate["architecture_fingerprint"],
        }
        for model_name in collected:
            matrix = np.stack(collected[model_name], axis=0)
            stats = _ensemble_statistics(
                matrix[:, index], np.asarray(calibration[model_name], dtype=np.float64)
            )
            prefix = "head" if model_name == "frozen_head" else "h"
            row.update({f"{prefix}_{key}": value for key, value in stats.items()})
        row["h_minus_head"] = float(row["h_prediction"] - row["head_prediction"])
        output.append(row)
    # The hashes are intentionally attached to the function for the caller's audit;
    # they are not selection features.
    frozen_head_h_predictions.checkpoint_hashes = dict(checkpoint_hashes)
    return output


def _rank01(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.argsort(np.asarray(values, dtype=np.float64), kind="stable"), kind="stable")
    return order.astype(np.float64) / max(1, len(values) - 1)


def _standardized(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return (values - mean) / std


def _greedy_diverse_select(
    eligible: Sequence[Mapping[str, Any]], count: int, selected_global: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if len(eligible) < count:
        raise RuntimeError(f"eligible stratum has {len(eligible)} candidates; need {count}")
    candidate_rows = [dict(row) for row in eligible]
    latent = _standardized(np.asarray([row["mu_arch"] for row in candidate_rows], dtype=np.float64))
    structure = np.asarray([row["structure_vector"] for row in candidate_rows], dtype=np.float64)
    base = np.asarray([float(row["selection_priority"]) for row in candidate_rows])
    chosen_indices: list[int] = []
    while len(chosen_indices) < count:
        best = None
        for index, row in enumerate(candidate_rows):
            if index in chosen_indices:
                continue
            if not chosen_indices:
                latent_diversity = 1.0
                structure_diversity = 1.0
            else:
                latent_distances = [float(np.linalg.norm(latent[index] - latent[item])) for item in chosen_indices]
                structure_distances = [float(np.mean(np.abs(structure[index] - structure[item]))) for item in chosen_indices]
                latent_diversity = min(latent_distances)
                structure_diversity = min(structure_distances)
            score = float(base[index] + 0.05 * latent_diversity + 0.20 * structure_diversity)
            key = (score, stable_json_hash({"fp": row["architecture_fingerprint"], "rule": "diverse_v1"}))
            if best is None or key > best[0]:
                best = (key, index)
        if best is None:
            raise RuntimeError("diversity selection exhausted")
        chosen_indices.append(best[1])
    chosen = [candidate_rows[index] for index in chosen_indices]
    prior_fps = {str(row["architecture_fingerprint"]) for row in selected_global}
    if prior_fps & {str(row["architecture_fingerprint"]) for row in chosen}:
        raise AssertionError("candidate selected in multiple strata")
    return chosen


def select_targeted_candidates(
    pool: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction_lookup = {str(row["config_id"]): row for row in predictions}
    rows = [{**dict(row), **dict(prediction_lookup[str(row["config_id"])])} for row in pool]
    h_values = np.asarray([row["h_prediction"] for row in rows], dtype=np.float64)
    head_values = np.asarray([row["head_prediction"] for row in rows], dtype=np.float64)
    deltas = h_values - head_values
    thresholds = {
        "high_prediction_quantile": 0.70,
        "clear_difference_tail_quantile": 0.80,
        "h_high": float(np.quantile(h_values, 0.70)),
        "head_high": float(np.quantile(head_values, 0.70)),
        "delta_h_favored": float(np.quantile(deltas, 0.80)),
        "delta_head_favored": float(np.quantile(deltas, 0.20)),
    }
    h_rank = _rank01(h_values)
    head_rank = _rank01(head_values)
    delta_rank = _rank01(deltas)
    for index, row in enumerate(rows):
        row["h_rank01"] = float(h_rank[index])
        row["head_rank01"] = float(head_rank[index])
        row["delta_rank01"] = float(delta_rank[index])

    selected: list[dict[str, Any]] = []
    h_eligible = [
        row for row in rows
        if row["h_prediction"] >= thresholds["h_high"]
        and row["h_minus_head"] >= thresholds["delta_h_favored"]
    ]
    for row in h_eligible:
        row["selection_priority"] = 0.65 * row["h_rank01"] + 0.35 * row["delta_rank01"]
    h_selected = _greedy_diverse_select(h_eligible, 6, selected)
    for row in h_selected:
        row["selection_stratum"] = "h_high_h_over_head"
    selected.extend(h_selected)

    selected_fps = {row["architecture_fingerprint"] for row in selected}
    head_eligible = [
        row for row in rows
        if row["architecture_fingerprint"] not in selected_fps
        and row["head_prediction"] >= thresholds["head_high"]
        and row["h_minus_head"] <= thresholds["delta_head_favored"]
    ]
    for row in head_eligible:
        row["selection_priority"] = 0.65 * row["head_rank01"] + 0.35 * (1.0 - row["delta_rank01"])
    head_selected = _greedy_diverse_select(head_eligible, 6, selected)
    for row in head_selected:
        row["selection_stratum"] = "head_high_head_over_h"
    selected.extend(head_selected)

    selected_fps = {row["architecture_fingerprint"] for row in selected}
    both_eligible = [
        row for row in rows
        if row["architecture_fingerprint"] not in selected_fps
        and row["h_prediction"] >= thresholds["h_high"]
        and row["head_prediction"] >= thresholds["head_high"]
    ]
    for row in both_eligible:
        row["selection_priority"] = 0.5 * row["h_rank01"] + 0.5 * row["head_rank01"]
    both_selected = _greedy_diverse_select(both_eligible, 8, selected)
    for row in both_selected:
        row["selection_stratum"] = "both_high_diverse"
    selected.extend(both_selected)

    selected.sort(key=lambda row: (row["selection_stratum"], row["architecture_fingerprint"]))
    for index, row in enumerate(selected):
        row["selection_index"] = index
        row["assigned_gpu"] = index % 2
    if len(selected) != 20 or len({row["architecture_fingerprint"] for row in selected}) != 20:
        raise AssertionError("targeted selection is not 20 unique architectures")
    if Counter(row["assigned_gpu"] for row in selected) != Counter({0: 10, 1: 10}):
        raise AssertionError("targeted selection is not split 10/10")
    return selected, {
        "thresholds": thresholds,
        "eligible_counts": {
            "h_high_h_over_head": len(h_eligible),
            "head_high_head_over_h": len(head_eligible),
            "both_high_diverse": len(both_eligible),
        },
        "selected_counts": dict(Counter(row["selection_stratum"] for row in selected)),
        "gpu_counts": dict(Counter(row["assigned_gpu"] for row in selected)),
        "selection_uses_label_fields": False,
    }


def frozen_gp_predictions_for_selection(
    selection: Sequence[Mapping[str, Any]], strict_root: Path, unified_root: Path
) -> list[dict[str, Any]]:
    from analyse.evaluate_performance_aware_vae_strict_regression import _fixed_gp_predict

    labels = load_validation_labels(unified_root / "unified_training_data_no_test.json")
    labels_by_id = {str(row["config_id"]): row for row in labels}
    manifest = json.loads((strict_root / "fixed_split_manifest.json").read_text(encoding="utf-8"))
    splits = [
        row for row in manifest
        if row["dataset_variant"] == "full" and row["protocol"] == "group_repeated"
    ]
    if len(splits) != 50:
        raise RuntimeError(f"expected 50 full/group_repeated splits, found {len(splits)}")
    candidate_hp = np.asarray([row["raw_hp_vector"] for row in selection], dtype=np.float64)
    candidate_features = {
        "zs": np.concatenate(
            (np.asarray([row["z_arch"] for row in selection], dtype=np.float64), candidate_hp), axis=1
        ),
        "ze": np.concatenate(
            (np.asarray([row["mu_arch"] for row in selection], dtype=np.float64), candidate_hp), axis=1
        ),
    }
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    collected_std: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in splits:
        train = [labels_by_id[str(item)] for item in split["train_config_ids"]]
        y = np.asarray([row["validation_mean"] for row in train], dtype=np.float64)
        hp = np.asarray([row["raw_hp_vector"] for row in train], dtype=np.float64)
        for representation, source_name in (("z_arch", "zs"), ("mu_arch", "ze")):
            x = np.concatenate(
                (np.asarray([row[representation] for row in train], dtype=np.float64), hp), axis=1
            )
            predicted, std = _fixed_gp_predict(x, y, candidate_features[source_name])
            collected[source_name].append(predicted)
            collected_std[source_name].append(std)
    output = []
    for index, row in enumerate(selection):
        result = {
            "config_id": row["config_id"],
            "architecture_fingerprint": row["architecture_fingerprint"],
        }
        for name in ("zs", "ze"):
            matrix = np.stack(collected[name], axis=0)
            std_matrix = np.stack(collected_std[name], axis=0)
            stats = _ensemble_statistics(matrix[:, index], std_matrix[:, index])
            result.update({f"{name}_{key}": value for key, value in stats.items()})
        output.append(result)
    return output


def prepare_phase_b(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    phase_dir = output / "phase_b"
    phase_a_decision = json.loads((output / "phase_a/phase_a_decision.json").read_text(encoding="utf-8"))
    if not phase_a_decision["phase_a_passed"]:
        raise RuntimeError("Phase A did not authorize candidate freezing")
    if phase_dir.exists() and any(phase_dir.iterdir()):
        raise FileExistsError(f"Phase B output is not empty: {phase_dir}")
    phase_dir.mkdir(parents=True)
    strict_root = args.strict_root.resolve()
    unified_root = args.unified_root.resolve()
    history_paths = sorted(REPO_ROOT.glob(HISTORY_GLOB))
    if len(history_paths) != 5:
        raise RuntimeError(f"expected five Phase 4 history files, found {len(history_paths)}")
    existing = existing_architecture_fingerprints_only(
        unified_root / "unified_training_data_no_test.json"
    )
    projected = []
    masked_fields = set()
    projection_failures = []
    for path in history_paths:
        source_rows = json.loads(path.read_text(encoding="utf-8"))
        for index, source in enumerate(source_rows):
            try:
                candidate, masked = project_history_candidate(
                    source, source_path=path, source_index=index
                )
                projected.append(candidate)
                masked_fields.update(masked)
            except Exception as exc:
                projection_failures.append(
                    {"source_path": str(path.resolve()), "source_index": index, "error": str(exc)}
                )
    pool, dedup_audit = deduplicate_candidates(projected, existing)
    if len(pool) < 20:
        decision = {
            "classification": "INSUFFICIENT_TARGET_CANDIDATES",
            "candidate_count": len(pool),
            "training_started": False,
        }
        _dump_json(phase_dir / "phase_b_stop_decision.json", decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 3

    from architecture_diagnostics import config_to_igraph, load_joint_vae
    import torch

    model = load_joint_vae(
        REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth",
        "global4",
        device="cpu",
    )
    with torch.no_grad():
        mu, _ = model.arch_vae.encode([config_to_igraph(row) for row in pool])
    for index, row in enumerate(pool):
        row["mu_arch"] = mu[index].detach().cpu().to(torch.float32).tolist()
        row.update(_realize_candidate_hp(row))
        row["config_id"] = f"targeted_{row['source_search_seed']}_{row['source_step']:03d}_{row['architecture_fingerprint'][:12]}"
        row["hp_fingerprint"] = stable_json_hash(
            {key: row[key] for key in ("lr", "dropout", "hidden_dim", "l2")}
        )
        row["structure_vector"] = structure_feature_vector(row, [1.0] * 4)[0].tolist()
        if contains_forbidden_candidate_field(row):
            raise AssertionError("candidate pool contains a forbidden label field")

    predictions = frozen_head_h_predictions(pool, strict_root)
    selection, selection_audit = select_targeted_candidates(pool, predictions)
    if {row["architecture_fingerprint"] for row in selection} & existing:
        raise AssertionError("targeted selection overlaps existing 60 architectures")
    prospective_gp = frozen_gp_predictions_for_selection(selection, strict_root, unified_root)
    gp_lookup = {row["config_id"]: row for row in prospective_gp}
    frozen_prospective = []
    for row in selection:
        frozen_prospective.append(
            {
                "config_id": row["config_id"],
                "architecture_fingerprint": row["architecture_fingerprint"],
                "selection_stratum": row["selection_stratum"],
                "head_prediction": row["head_prediction"],
                "head_prediction_interval_low": row["head_prediction_interval_low"],
                "head_prediction_interval_high": row["head_prediction_interval_high"],
                "h_prediction": row["h_prediction"],
                "h_prediction_interval_low": row["h_prediction_interval_low"],
                "h_prediction_interval_high": row["h_prediction_interval_high"],
                **{
                    key: value
                    for key, value in gp_lookup[row["config_id"]].items()
                    if key.startswith("zs_") or key.startswith("ze_")
                },
            }
        )
    gpus = _actual_gpus()
    code_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "analyse/run_h_targeted_label_validation.py",
        REPO_ROOT / "analyse/evaluate_h_targeted_reliable_pair_validation.py",
        REPO_ROOT / "tests/test_h_targeted_reliable_pair_validation.py",
        REPO_ROOT / "architecture_diagnostics.py",
        REPO_ROOT / "eval_utils.py",
    ]
    for path in code_paths:
        if not path.is_file():
            raise FileNotFoundError(f"required frozen code is missing: {path}")
    candidate_audit = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "source_kind": "five_phase4_history_final_files",
        "source_record_count": sum(len(json.loads(path.read_text(encoding="utf-8"))) for path in history_paths),
        "source_files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in history_paths],
        "source_fields_explicitly_masked": sorted(masked_fields),
        "candidate_selection_read_fields": [
            "operations", "edges", "z_arch", "z_search[12:16]", "search_seed", "step"
        ],
        "candidate_selection_used_validation_or_test_values": False,
        "projection_failure_count": len(projection_failures),
        "projection_failures": projection_failures,
        **dedup_audit,
        "existing_architecture_count": len(existing),
        "candidate_existing_architecture_overlap": len(
            {row["architecture_fingerprint"] for row in pool} & existing
        ),
        "one_hp_per_architecture": len(pool)
        == len({row["architecture_fingerprint"] for row in pool}),
        "candidate_count": len(pool),
        "selection": selection_audit,
        "head_h_checkpoint_hashes": frozen_head_h_predictions.checkpoint_hashes,
        "test_label_used": False,
    }
    candidate_report = f"""# Phase B frozen candidate pool

- Source: {candidate_audit['source_record_count']} records from five Phase 4 history files.
- Label-bearing source fields were explicitly masked before projection: {', '.join(candidate_audit['source_fields_explicitly_masked'])}.
- Eligible pool: {len(pool)} unique architecture fingerprints after excluding all existing 60 labeled fingerprints and applying one HP per architecture.
- Existing-fingerprint overlap: 0.
- Frozen Head/H inference: 150 full/group-repeated checkpoints per model with each checkpoint's training-fold HP scaling.
- Selection: 6 H-favored, 6 Head-favored, and 8 both-high diverse architectures; GPU assignment is 10/10.
- Candidate selection used no validation or test values.
"""
    _dump_json(phase_dir / "candidate_pool_audit.json", {"audit": candidate_audit, "candidates": pool})
    _dump_csv(phase_dir / "candidate_pool_audit.csv", pool)
    (phase_dir / "candidate_pool_audit.md").write_text(candidate_report, encoding="utf-8")
    _dump_csv(phase_dir / "frozen_candidate_predictions.csv", predictions)
    _dump_json(phase_dir / "targeted_selection.json", selection)
    _dump_csv(phase_dir / "targeted_selection.csv", selection)
    _dump_csv(phase_dir / "frozen_prospective_predictions.csv", frozen_prospective)

    input_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in [
            *history_paths,
            strict_root / "task_plan.json",
            strict_root / "fixed_split_manifest.json",
            strict_root / "all_validation_predictions.csv",
            strict_root / "final_classification.json",
            unified_root / "unified_training_data_no_test.json",
            unified_root / "unified_seed_level_audit.jsonl",
            unified_root / "high_noise_configurations.json",
            REPO_ROOT / "data/mini_gnn_dataset_global4.pkl",
            REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth",
            phase_dir / "candidate_pool_audit.json",
            phase_dir / "frozen_candidate_predictions.csv",
            phase_dir / "targeted_selection.json",
            phase_dir / "targeted_selection.csv",
            phase_dir / "frozen_prospective_predictions.csv",
        ]
    }
    task_plan = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "diagnostic_only": True,
        "vae_training_allowed": False,
        "bo_allowed": False,
        "test_labels_allowed": False,
        "selection_path": str((phase_dir / "targeted_selection.json").resolve()),
        "selection_sha256": sha256_file(phase_dir / "targeted_selection.json"),
        "configuration_count": 20,
        "architecture_count": 20,
        "initial_seeds": list(INITIAL_SEEDS),
        "adaptive_seeds": list(ADAPTIVE_SEEDS),
        "adaptive_rule": {
            "activate_when_validation_t_ci95_half_width_gt": 0.006,
            "activate_when_fewer_than_initial_valid_seeds": 15,
            "decision_independent_of_prediction_correctness": True,
        },
        "maximum_retries": 2,
        "training_protocol": {
            "dataset": "cora",
            "hp_mode": "global4",
            "max_epochs": 300,
            "patience": 80,
            "track_test": False,
            "dataset_content_fingerprint": "373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7",
            "split_fingerprint": "bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09",
        },
        "gpus": gpus,
        "shard_to_gpu_uuid": {str(row["physical_index"]): row["uuid"] for row in gpus},
        "initial_task_count": 20 * len(INITIAL_SEEDS),
        "maximum_adaptive_task_count": 20 * len(ADAPTIVE_SEEDS),
        "merge_key": ["config_id", "architecture_fingerprint", "seed"],
        "code_hashes": {str(path): sha256_file(path) for path in code_paths},
        "input_hashes": input_hashes,
        "protected_hashes": protected_hashes(),
    }
    _dump_json(phase_dir / "targeted_task_plan.json", task_plan)
    plan_hash = sha256_file(phase_dir / "targeted_task_plan.json")
    (phase_dir / "targeted_task_plan.sha256").write_text(
        f"{plan_hash}  targeted_task_plan.json\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "candidate_count": len(pool),
                "selected_count": len(selection),
                "selection_strata": selection_audit["selected_counts"],
                "gpu_counts": selection_audit["gpu_counts"],
                "task_plan_sha256": plan_hash,
                "training_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def load_validation_labels(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("validation labels must be a non-empty JSON list")
    required = {
        "config_id",
        "architecture_fingerprint",
        "operations",
        "edges",
        "raw_hp_vector",
        "validation_mean",
        "validation_ci95_low",
        "validation_ci95_high",
        "time_index",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"validation label row {index} missing {sorted(missing)}")
        if any("test" in str(key).lower() for key in row):
            raise ValueError(f"validation label row {index} contains a test-named field")
        if architecture_fingerprint(row) != str(row["architecture_fingerprint"]):
            raise ValueError(f"architecture fingerprint drift in label row {index}")
        if len(row["raw_hp_vector"]) != 4:
            raise ValueError(f"label row {index} is not global4")
    return [dict(row) for row in rows]


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    numeric = (
        "prediction",
        "prediction_interval_low",
        "prediction_interval_high",
        "validation_mean",
        "validation_ci95_low",
        "validation_ci95_high",
    )
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any("test" in name.lower() for name in reader.fieldnames):
            raise ValueError("prediction file contains a test-named field")
        for row in reader:
            if row["dataset_variant"] != DATASET_VARIANT or row["protocol"] != PROTOCOL:
                continue
            if row["model"] not in MODELS:
                continue
            parsed = dict(row)
            for field in numeric:
                parsed[field] = float(parsed[field])
            parsed["time_index"] = int(parsed["time_index"])
            parsed["training_seed"] = (
                None if parsed["training_seed"] == "" else int(parsed["training_seed"])
            )
            rows.append(parsed)
    if not rows:
        raise ValueError("no full/group_repeated prediction rows found")
    return rows


def aggregate_predictions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated split and initialization predictions by configuration."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)
    output = []
    for config_id, values in sorted(grouped.items()):
        identities = {
            (
                str(row["architecture_fingerprint"]),
                float(row["validation_mean"]),
                float(row["validation_ci95_low"]),
                float(row["validation_ci95_high"]),
                int(row["time_index"]),
            )
            for row in values
        }
        if len(identities) != 1:
            raise ValueError(f"label identity drift across predictions for {config_id}")
        predictions = np.asarray([float(row["prediction"]) for row in values], dtype=np.float64)
        variances = np.asarray(
            [
                ((float(row["prediction_interval_high"]) - float(row["prediction"])) / 1.96)
                ** 2
                for row in values
            ],
            dtype=np.float64,
        )
        total_variance = max(
            float(np.mean(variances + predictions**2) - predictions.mean() ** 2),
            1e-12,
        )
        first = values[0]
        output.append(
            {
                "config_id": config_id,
                "architecture_fingerprint": str(first["architecture_fingerprint"]),
                "validation_mean": float(first["validation_mean"]),
                "validation_ci95_low": float(first["validation_ci95_low"]),
                "validation_ci95_high": float(first["validation_ci95_high"]),
                "time_index": int(first["time_index"]),
                "prediction": float(predictions.mean()),
                "prediction_interval_low": float(predictions.mean() - 1.96 * math.sqrt(total_variance)),
                "prediction_interval_high": float(predictions.mean() + 1.96 * math.sqrt(total_variance)),
                "prediction_source_count": len(values),
                "split_id_count": len({str(row["split_id"]) for row in values}),
                "training_seed_count": len(
                    {int(row["training_seed"]) for row in values if row["training_seed"] is not None}
                ),
                "model": str(first["model"]),
            }
        )
    return output


def independently_aggregate_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], float]:
    """Second aggregation route using explicit sums rather than NumPy means."""

    sums: dict[tuple[str, str], float] = defaultdict(float)
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (str(row["model"]), str(row["config_id"]))
        sums[key] += float(row["prediction"])
        counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums}


def architecture_level_labels(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(row)
    return [
        {
            "architecture_fingerprint": fingerprint,
            "validation_mean": statistics.fmean(float(row["validation_mean"]) for row in values),
            "prediction": statistics.fmean(float(row["prediction"]) for row in values),
            "config_ids": sorted(str(row["config_id"]) for row in values),
            "time_index": max(int(row["time_index"]) for row in values),
        }
        for fingerprint, values in sorted(grouped.items())
    ]


def select_top_architectures(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    architectures = architecture_level_labels(rows)
    count = max(1, math.ceil(TOP_FRACTION * len(architectures)))
    ranked = sorted(
        architectures,
        key=lambda row: (-float(row["validation_mean"]), str(row["architecture_fingerprint"])),
    )
    return [str(row["architecture_fingerprint"]) for row in ranked[:count]]


def intervals_disjoint(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        float(left["validation_ci95_low"]) > float(right["validation_ci95_high"])
        or float(right["validation_ci95_low"]) > float(left["validation_ci95_high"])
    )


def reliable_config_pairs(
    rows: Sequence[Mapping[str, Any]], top_architectures: Iterable[str]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int]:
    top = set(map(str, top_architectures))
    selected = [dict(row) for row in rows if str(row["architecture_fingerprint"]) in top]
    raw_architecture_pairs = set()
    reliable = []
    seen_config_pairs = set()
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            left_fp = str(left["architecture_fingerprint"])
            right_fp = str(right["architecture_fingerprint"])
            if left_fp == right_fp:
                continue
            raw_architecture_pairs.add(tuple(sorted((left_fp, right_fp))))
            config_key = tuple(sorted((str(left["config_id"]), str(right["config_id"]))))
            if config_key in seen_config_pairs:
                raise AssertionError(f"symmetric or duplicate config pair: {config_key}")
            seen_config_pairs.add(config_key)
            delta = float(left["validation_mean"]) - float(right["validation_mean"])
            if abs(delta) >= MINIMUM_RELIABLE_GAP and intervals_disjoint(left, right):
                higher, lower = (left, right) if delta > 0 else (right, left)
                reliable.append((higher, lower))
    return reliable, len(raw_architecture_pairs)


def build_pair_details(
    pooled_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    head_rows = list(pooled_by_model["frozen_head"])
    top_architectures = select_top_architectures(head_rows)
    contributing, raw_pair_count = reliable_config_pairs(head_rows, top_architectures)
    prediction_lookup = {
        model: {str(row["config_id"]): row for row in rows}
        for model, rows in pooled_by_model.items()
    }
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for higher, lower in contributing:
        key = tuple(
            sorted(
                (
                    str(higher["architecture_fingerprint"]),
                    str(lower["architecture_fingerprint"]),
                )
            )
        )
        grouped[key].append((higher, lower))

    details = []
    model_correct_counts = Counter()
    model_fraction_sums = Counter()
    for pair_index, (key, values) in enumerate(sorted(grouped.items()), 1):
        higher_fp = str(values[0][0]["architecture_fingerprint"])
        lower_fp = str(values[0][1]["architecture_fingerprint"])
        if any(
            str(higher["architecture_fingerprint"]) != higher_fp
            or str(lower["architecture_fingerprint"]) != lower_fp
            for higher, lower in values
        ):
            raise AssertionError("reliable architecture pair has inconsistent label direction")
        contributions = []
        model_results: dict[str, Any] = {}
        for higher, lower in values:
            contribution = {
                "higher_config_id": higher["config_id"],
                "higher_architecture_fingerprint": higher_fp,
                "higher_validation_mean": higher["validation_mean"],
                "higher_validation_ci95_low": higher["validation_ci95_low"],
                "higher_validation_ci95_high": higher["validation_ci95_high"],
                "lower_config_id": lower["config_id"],
                "lower_architecture_fingerprint": lower_fp,
                "lower_validation_mean": lower["validation_mean"],
                "lower_validation_ci95_low": lower["validation_ci95_low"],
                "lower_validation_ci95_high": lower["validation_ci95_high"],
                "validation_mean_gap": float(higher["validation_mean"] - lower["validation_mean"]),
                "intervals_disjoint": intervals_disjoint(higher, lower),
            }
            for model in MODELS:
                high_prediction = float(prediction_lookup[model][str(higher["config_id"])]["prediction"])
                low_prediction = float(prediction_lookup[model][str(lower["config_id"])]["prediction"])
                short = SHORT_MODEL_NAMES[model]
                contribution[f"{short}_higher_prediction"] = high_prediction
                contribution[f"{short}_lower_prediction"] = low_prediction
                contribution[f"{short}_direction"] = (
                    "higher_config" if high_prediction > low_prediction
                    else "lower_config" if high_prediction < low_prediction
                    else "tie"
                )
                contribution[f"{short}_correct"] = high_prediction > low_prediction
            contributions.append(contribution)

        for model in MODELS:
            short = SHORT_MODEL_NAMES[model]
            correctness = [float(row[f"{short}_correct"]) for row in contributions]
            fraction = statistics.fmean(correctness)
            model_results[f"{short}_correct_fraction"] = fraction
            model_results[f"{short}_correct"] = fraction > 0.5
            model_results[f"{short}_direction"] = (
                "higher_config" if fraction > 0.5
                else "lower_config" if fraction < 0.5
                else "mixed_or_tie"
            )
            model_results[f"{short}_higher_prediction"] = statistics.fmean(
                float(row[f"{short}_higher_prediction"]) for row in contributions
            )
            model_results[f"{short}_lower_prediction"] = statistics.fmean(
                float(row[f"{short}_lower_prediction"]) for row in contributions
            )
            model_fraction_sums[model] += fraction
            model_correct_counts[model] += int(fraction > 0.5)

        higher_ids = sorted({str(row[0]["config_id"]) for row in values})
        lower_ids = sorted({str(row[1]["config_id"]) for row in values})
        higher_label_rows = [labels_by_id[item] for item in higher_ids]
        lower_label_rows = [labels_by_id[item] for item in lower_ids]
        details.append(
            {
                "pair_id": f"reliable_pair_{pair_index:02d}",
                "architecture_pair_key": list(key),
                "higher_config_id": ";".join(higher_ids),
                "higher_config_ids": higher_ids,
                "higher_architecture_fingerprint": higher_fp,
                "higher_validation_mean": statistics.fmean(float(row["validation_mean"]) for row in higher_label_rows),
                "higher_validation_ci95_low": min(float(row["validation_ci95_low"]) for row in higher_label_rows),
                "higher_validation_ci95_high": max(float(row["validation_ci95_high"]) for row in higher_label_rows),
                "lower_config_id": ";".join(lower_ids),
                "lower_config_ids": lower_ids,
                "lower_architecture_fingerprint": lower_fp,
                "lower_validation_mean": statistics.fmean(float(row["validation_mean"]) for row in lower_label_rows),
                "lower_validation_ci95_low": min(float(row["validation_ci95_low"]) for row in lower_label_rows),
                "lower_validation_ci95_high": max(float(row["validation_ci95_high"]) for row in lower_label_rows),
                "validation_mean_gap": statistics.fmean(float(row["validation_mean_gap"]) for row in contributions),
                "contributing_config_pair_count": len(contributions),
                "contributing_config_pairs": contributions,
                "shared_architecture_fingerprint": higher_fp == lower_fp,
                "same_architecture_different_hp_across_pair": False,
                "within_side_multiple_hp": len(higher_ids) > 1 or len(lower_ids) > 1,
                **model_results,
            }
        )

    audit = {
        "top_architecture_count": len(top_architectures),
        "top_architecture_fingerprints": sorted(top_architectures),
        "top_configuration_count": sum(
            str(row["architecture_fingerprint"]) in set(top_architectures) for row in head_rows
        ),
        "raw_architecture_pair_count": raw_pair_count,
        "reliable_configuration_pair_count": len(contributing),
        "reliable_architecture_pair_count": len(details),
        "model_correct_architecture_pair_counts": dict(model_correct_counts),
        "model_architecture_pair_accuracy": {
            model: model_fraction_sums[model] / len(details) for model in MODELS
        },
    }
    return details, audit


def validation_seed_interval_audit(
    seed_audit_path: Path,
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute intervals while reading only validation fields from the audit."""

    values: dict[str, list[float]] = defaultdict(list)
    seed_keys = set()
    ignored_test_named_fields = set()
    with seed_audit_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            ignored_test_named_fields.update(
                key for key in source if "test" in str(key).lower()
            )
            selected = {
                "config_id": str(source["config_id"]),
                "seed": int(source["seed"]),
                "val_acc": source.get("val_acc"),
                "terminal_status": str(source["terminal_status"]),
                "valid": bool(source["valid"]),
            }
            key = (selected["config_id"], selected["seed"])
            if key in seed_keys:
                raise ValueError(f"seed-level pseudo-replication at line {line_number}: {key}")
            seed_keys.add(key)
            if selected["terminal_status"] == "success" and selected["valid"]:
                values[selected["config_id"]].append(float(selected["val_acc"]))
    mismatches = []
    counts = Counter()
    for label in labels:
        config_id = str(label["config_id"])
        observed = values[config_id]
        counts[len(observed)] += 1
        if len(observed) < 2:
            mismatches.append({"config_id": config_id, "reason": "fewer than two validation seeds"})
            continue
        mean = statistics.fmean(observed)
        half = float(t.ppf(0.975, len(observed) - 1) * statistics.stdev(observed) / math.sqrt(len(observed)))
        expected = (
            float(label["validation_mean"]),
            float(label["validation_ci95_low"]),
            float(label["validation_ci95_high"]),
        )
        recomputed = (mean, mean - half, mean + half)
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(expected, recomputed)):
            mismatches.append(
                {
                    "config_id": config_id,
                    "expected": expected,
                    "recomputed": recomputed,
                }
            )
    return {
        "source": str(seed_audit_path.resolve()),
        "read_fields": ["config_id", "seed", "val_acc", "terminal_status", "valid"],
        "ignored_test_named_fields": sorted(ignored_test_named_fields),
        "test_values_read": False,
        "unique_config_seed_key_count": len(seed_keys),
        "successful_validation_seed_count": sum(len(item) for item in values.values()),
        "seed_counts_by_configuration": {str(key): value for key, value in sorted(counts.items())},
        "interval_mismatches": mismatches,
        "passed": not mismatches,
    }


def direction_audit(
    details: Sequence[Mapping[str, Any]], aggregation_checks: Mapping[str, Any]
) -> dict[str, Any]:
    keys = [tuple(row["architecture_pair_key"]) for row in details]
    reversed_duplicates = [key for key in keys if (key[1], key[0]) in set(keys)]
    expected_counts = {
        "frozen_head": 2,
        "frozen_hybrid_head": 1,
        "original_zs_fixed_gp": 6,
        "original_ze_fixed_gp": 5,
        "ft_latent_gp_lambda_0.1": 5,
        "ft_latent_gp_lambda_0.5": 5,
        "ft_latent_gp_lambda_0.9": 5,
    }
    observed_counts = aggregation_checks["model_correct_architecture_pair_counts"]
    checks = {
        "higher_validation_mean_is_correct_answer": all(
            all(
                float(pair["higher_validation_mean"]) > float(pair["lower_validation_mean"])
                for pair in row["contributing_config_pairs"]
            )
            for row in details
        ),
        "larger_prediction_is_better": all(
            all(
                pair[f"{short}_correct"]
                == (pair[f"{short}_higher_prediction"] > pair[f"{short}_lower_prediction"])
                for short in SHORT_MODEL_NAMES.values()
                for pair in row["contributing_config_pairs"]
            )
            for row in details
        ),
        "no_symmetric_or_duplicate_architecture_pairs": len(keys) == len(set(keys)) and not reversed_duplicates,
        "no_within_architecture_hp_pair": all(
            not bool(row["shared_architecture_fingerprint"])
            and not bool(row["same_architecture_different_hp_across_pair"])
            for row in details
        ),
        "reliable_pair_count_is_11": len(details) == 11,
        "reference_correct_counts_match": all(observed_counts.get(model) == count for model, count in expected_counts.items()),
        "aggregation_route_predictions_match": bool(aggregation_checks["independent_aggregation_match"]),
        "validation_intervals_from_seed_labels": bool(aggregation_checks["validation_seed_interval_audit_passed"]),
        "architecture_bootstrap_shared_indices": bool(aggregation_checks["architecture_bootstrap_shared_indices"]),
        "h_feature_protocol_matches_strict_experiment": bool(aggregation_checks["h_feature_protocol_matches"]),
    }
    return {
        "format_version": FORMAT_VERSION,
        "checks": checks,
        "expected_model_correct_counts": expected_counts,
        "observed_model_correct_counts": observed_counts,
        "reversed_duplicate_keys": reversed_duplicates,
        "passed": all(checks.values()),
    }


def phase_a(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    phase_dir = output / "phase_a"
    if phase_dir.exists() and any(phase_dir.iterdir()) and not args.overwrite_phase_a:
        raise FileExistsError(f"Phase A output is not empty: {phase_dir}")
    phase_dir.mkdir(parents=True, exist_ok=True)
    strict_root = args.strict_root.resolve()
    unified_root = args.unified_root.resolve()
    predictions_path = strict_root / "all_validation_predictions.csv"
    labels_path = unified_root / "unified_training_data_no_test.json"
    seed_audit_path = unified_root / "unified_seed_level_audit.jsonl"
    labels = load_validation_labels(labels_path)
    labels_by_id = {str(row["config_id"]): row for row in labels}
    if len(labels_by_id) != len(labels):
        raise ValueError("duplicate validation config IDs")
    prediction_rows = load_prediction_rows(predictions_path)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_model[str(row["model"])].append(row)
    if set(by_model) != set(MODELS):
        raise ValueError(f"prediction models mismatch: {sorted(by_model)}")
    pooled = {model: aggregate_predictions(by_model[model]) for model in MODELS}
    if any(len(rows) != len(labels) for rows in pooled.values()):
        raise ValueError("not every model aggregated to every validation configuration")
    if any({row["config_id"] for row in rows} != set(labels_by_id) for rows in pooled.values()):
        raise ValueError("prediction and label configuration IDs differ")

    independent = independently_aggregate_predictions(prediction_rows)
    aggregation_differences = []
    for model, rows in pooled.items():
        for row in rows:
            expected = independent[(model, str(row["config_id"]))]
            if not math.isclose(float(row["prediction"]), expected, rel_tol=0.0, abs_tol=1e-12):
                aggregation_differences.append(
                    {"model": model, "config_id": row["config_id"], "left": row["prediction"], "right": expected}
                )
    seed_interval = validation_seed_interval_audit(seed_audit_path, labels)
    details, recomputation = build_pair_details(pooled, labels_by_id)

    strict_plan = json.loads((strict_root / "task_plan.json").read_text(encoding="utf-8"))
    bootstrap = strict_plan["bootstrap"]
    bootstrap_architectures = sorted(
        {str(row["architecture_fingerprint"]) for row in pooled["frozen_head"]}
    )
    rng = np.random.default_rng(int(bootstrap["seed"]))
    shared_indices = [
        [bootstrap_architectures[index] for index in rng.integers(0, len(bootstrap_architectures), size=len(bootstrap_architectures))]
        for _ in range(int(bootstrap["replicates"]))
    ]
    structure_dimension = len(structure_feature_vector(labels[0], [1.0] * 4)[0])
    h_checkpoints = list((strict_root / "checkpoints" / "full" / "group_repeated").glob("group_*/seed_*/frozen_hybrid_head/checkpoint.pth"))
    head_dimensions = set()
    if h_checkpoints:
        import torch

        for checkpoint in h_checkpoints:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            head_dimensions.add(int(payload["head_input_dim"]))
    h_expected_dimension = 12 + structure_dimension + 4
    h_feature_matches = head_dimensions == {h_expected_dimension}

    aggregation_checks = {
        **recomputation,
        "raw_prediction_row_count_by_model": {model: len(by_model[model]) for model in MODELS},
        "pooled_configuration_count_by_model": {model: len(pooled[model]) for model in MODELS},
        "prediction_source_count_distribution_by_model": {
            model: dict(sorted(Counter(row["prediction_source_count"] for row in pooled[model]).items()))
            for model in MODELS
        },
        "independent_aggregation_difference_count": len(aggregation_differences),
        "independent_aggregation_differences": aggregation_differences,
        "independent_aggregation_match": not aggregation_differences,
        "validation_seed_interval_audit_passed": seed_interval["passed"],
        "validation_seed_interval_audit": seed_interval,
        "bootstrap_unit": bootstrap["unit"],
        "bootstrap_replicates": bootstrap["replicates"],
        "bootstrap_seed": bootstrap["seed"],
        "architecture_bootstrap_shared_indices": bool(bootstrap["shared_indices"])
        and bootstrap["unit"] == "architecture_fingerprint",
        "shared_bootstrap_indices_sha256": stable_json_hash(shared_indices),
        "h_structure_feature_dimension": structure_dimension,
        "h_expected_head_input_dimension": h_expected_dimension,
        "h_checkpoint_head_input_dimensions": sorted(head_dimensions),
        "h_feature_protocol_matches": h_feature_matches,
        "h_features": ["base_mu_arch", "strict_structure_feature_vector", "train_fold_standardized_global4_hp"],
        "test_values_used": False,
    }
    pair_direction = direction_audit(details, aggregation_checks)
    mismatches = []
    if not pair_direction["passed"]:
        mismatches.append("pair direction/construction audit failed")
    if not seed_interval["passed"]:
        mismatches.append("validation seed interval recomputation failed")
    decision = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "classification": "PAIR_AUDIT_ERROR_FOUND" if mismatches else "PHASE_A_PASSED_PROCEED_TO_PHASE_B",
        "phase_a_passed": not mismatches,
        "new_label_training_allowed": not mismatches,
        "mismatches": mismatches,
        "reliable_architecture_pair_count": len(details),
        "h_correct_architecture_pair_count": recomputation["model_correct_architecture_pair_counts"]["frozen_hybrid_head"],
        "independent_recomputation_routes": 2,
        "test_label_used": False,
        "vae_training_executed": False,
        "gnn_training_executed": False,
        "bo_executed": False,
    }
    report = f"""# Phase A reliable-pair audit

Decision: `{decision['classification']}`.

- Full/group-repeated configurations: {len(labels)} across {len(set(row['architecture_fingerprint'] for row in labels))} architecture fingerprints.
- Top region: {recomputation['top_architecture_count']} independent architectures and {recomputation['top_configuration_count']} architecture+HP configurations.
- Raw top architecture pairs: {recomputation['raw_architecture_pair_count']}.
- Reliable pairs: {recomputation['reliable_architecture_pair_count']} architecture pairs from {recomputation['reliable_configuration_pair_count']} contributing configuration pairs.
- Correct architecture pairs: Head {recomputation['model_correct_architecture_pair_counts']['frozen_head']}/11; H {recomputation['model_correct_architecture_pair_counts']['frozen_hybrid_head']}/11; Zs {recomputation['model_correct_architecture_pair_counts']['original_zs_fixed_gp']}/11; Ze {recomputation['model_correct_architecture_pair_counts']['original_ze_fixed_gp']}/11; each fine-tuned latent GP {recomputation['model_correct_architecture_pair_counts']['ft_latent_gp_lambda_0.5']}/11.
- Direction is always higher validation mean and larger prediction. Symmetric pairs are deduplicated, same-architecture HP pairs are excluded, repeated splits/seeds are pooled per configuration, and confidence intervals are recomputed only from validation seeds.
- Bootstrap unit is `architecture_fingerprint` with {bootstrap['replicates']} shared-index replicates.
- Test values, VAE/GNN training, and BO were not used.

The two architectures with multiple top-region HP configurations are retained explicitly in each pair's `contributing_config_pairs`; they are not treated as independent architecture pairs.
"""
    _dump_json(phase_dir / "reliable_pair_details.json", details)
    _dump_csv(phase_dir / "reliable_pair_details.csv", details)
    _dump_json(phase_dir / "pair_direction_audit.json", pair_direction)
    _dump_json(phase_dir / "pair_metric_recomputation.json", aggregation_checks)
    (phase_dir / "pair_audit_report.md").write_text(report, encoding="utf-8")
    _dump_json(phase_dir / "phase_a_decision.json", decision)
    _dump_json(output / "protected_hashes_before.json", protected_hashes())
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if not mismatches else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    phase = subparsers.add_parser("phase-a", help="run the read-only reliable-pair audit")
    phase.add_argument(
        "--strict_root",
        type=Path,
        default=REPO_ROOT / "results/posthoc/performance-aware-vae-strict-regression-global4-v2",
    )
    phase.add_argument(
        "--unified_root",
        type=Path,
        default=REPO_ROOT / "results/posthoc/vae-label-expansion-global4-v1",
    )
    phase.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/posthoc/h-targeted-reliable-pair-validation-global4-v1",
    )
    phase.add_argument("--overwrite_phase_a", action="store_true")
    prepare = subparsers.add_parser("prepare-b", help="freeze the label-blind targeted plan")
    prepare.add_argument(
        "--strict_root",
        type=Path,
        default=REPO_ROOT / "results/posthoc/performance-aware-vae-strict-regression-global4-v2",
    )
    prepare.add_argument(
        "--unified_root",
        type=Path,
        default=REPO_ROOT / "results/posthoc/vae-label-expansion-global4-v1",
    )
    prepare.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/posthoc/h-targeted-reliable-pair-validation-global4-v1",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "phase-a":
        return phase_a(args)
    if args.command == "prepare-b":
        return prepare_phase_b(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
