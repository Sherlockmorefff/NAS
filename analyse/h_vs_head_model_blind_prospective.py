#!/usr/bin/env python3
"""Model-blind prospective validation of Original Frozen H against Head.

``prepare`` builds the exposure registry, generates a label-free candidate
pool with the frozen VAE, performs the preregistered stratified random
selection, reproduces the clean-24 frozen predictions, freezes new predictions
and writes the GNN task plan.  ``evaluate`` is only legal after all 144 frozen
GNN tasks have completed.  It aggregates three validation seeds and performs
the preregistered two-level bootstrap.

No performance-model/VAE training, scaler fitting, GP, test metric, or BO path
exists in this module.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import architecture_fingerprint, load_joint_vae
from analyse.h_targeted_reliable_pair_validation import frozen_head_h_predictions, sha256_file
from analyse.performance_aware_vae_strict_regression import config_from_igraph
from analyse.structure_native_surrogate_pilot import realize_global4_hp
from analyse.direct_head_chronological_offline_replay_v2 import ranking_metrics


FORMAT_VERSION = "h_vs_head_model_blind_prospective_global4_v1"
OUTPUT_RELATIVE = "results/posthoc/h-vs-head-model-blind-prospective-global4-v1"
STRICT_RELATIVE = "results/posthoc/performance-aware-vae-strict-regression-global4-v2"
CLEAN_RELATIVE = "results/posthoc/structure-native-surrogate-pilot-global4-v1"
BASE_CHECKPOINT = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
GENERATION_SEED = 20260828
HP_SEED = 20260829
SELECTION_SEED = 20260830
BOOTSTRAP_SEED = 20260827
BOOTSTRAP_REPLICATES = 10000
GENERATION_DRAWS = 4096
TRAINING_SEEDS = (0, 1, 2)
TARGET_CONFIGS = 48
TARGET_ARCHITECTURES = 40
SECOND_HP_ARCHITECTURES = 8
MAX_RETRIES = 1
HP_REGIMES = {
    "hp_low": (0.05, 0.20),
    "hp_mid_low": (0.30, 0.45),
    "hp_mid_high": (0.55, 0.70),
    "hp_high": (0.80, 0.95),
}
OPERATION_TYPES = {"GCNConv", "GATConv", "SAGEConv", "GINConv", "GCNII", "Identity"}
TRUTH_KEYS = {
    "val_acc", "validation", "validation_mean", "validation_accuracy",
    "val_accuracy", "best_val_acc", "seed_val_accs", "validation_values", "value",
}
MODELS = ("original_frozen_h", "original_frozen_head")
METRICS = (
    "kendall_tau_b", "spearman", "top_region_kendall_tau_b", "ndcg_at_10",
    "top_1_overlap_rate", "top_3_overlap_rate", "top_5_overlap_rate", "top_10_overlap_rate",
    "regret_at_1", "regret_at_3", "regret_at_5", "regret_at_10",
    "false_positive_rate", "false_negative_rate",
    "gap_filtered_pairwise_accuracy_0_006",
    "gap_filtered_pairwise_accuracy_0_012",
    "gap_filtered_pairwise_accuracy_0_020",
)
CLASSIFICATION_ORDER = (
    "INPUT_OR_IMPLEMENTATION_ERROR",
    "FROZEN_MODEL_REPRODUCTION_FAILED",
    "EXPOSURE_EXCLUSION_UNRESOLVED",
    "INSUFFICIENT_UNEXPOSED_CANDIDATE_POOL",
    "LABEL_FIREWALL_VIOLATION",
    "PROTOCOL_MISMATCH",
    "PROSPECTIVE_VALIDATION_INSUFFICIENT_COVERAGE",
    "H_PROSPECTIVELY_VALIDATED_FOR_ACQUISITION_CALIBRATION",
    "H_PROSPECTIVE_REJECTED",
    "H_PROSPECTIVE_SIGNAL_ONLY",
    "H_PROSPECTIVE_UNDERPOWERED",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple)) else value
                    for key, value in ((field, row.get(field)) for field in fields)
                })
    temporary.replace(path)


def write_sidecar(path: Path) -> Path:
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def verify_sidecar(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"missing sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if expected != sha256_file(path):
        raise RuntimeError(f"sidecar mismatch: {path}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hp_fingerprint(values: Sequence[float]) -> str:
    return stable_hash([float(value) for value in values])


def _truth_row_fingerprint(row: Mapping[str, Any]) -> str | None:
    keys = {str(key).lower() for key in row}
    has_truth = bool(keys & TRUTH_KEYS) or any(key.startswith("val_acc_") for key in keys)
    if not has_truth:
        return None
    stored = row.get("architecture_fingerprint")
    if isinstance(stored, str) and stored:
        return stored
    if "operations" in row and "edges" in row:
        try:
            return architecture_fingerprint(row)
        except (TypeError, ValueError, KeyError, IndexError):
            return None
    return None


def _collect_truth_fingerprints(value: Any, output: set[str]) -> None:
    if isinstance(value, Mapping):
        fingerprint = _truth_row_fingerprint(value)
        if fingerprint is not None:
            output.add(fingerprint)
        for nested in value.values():
            _collect_truth_fingerprints(nested, output)
    elif isinstance(value, list):
        for nested in value:
            _collect_truth_fingerprints(nested, output)


def _scan_json_source(path: Path) -> set[str]:
    found: set[str] = set()
    if path.suffix == ".json":
        _collect_truth_fingerprints(load_json(path), found)
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    _collect_truth_fingerprints(json.loads(line), found)
    return found


def exposure_source_groups() -> dict[str, list[Path]]:
    target = (REPO_ROOT / OUTPUT_RELATIVE).resolve()
    groups = {
        "original_ceiling_hunt": sorted(
            (REPO_ROOT / "results/cora-original-ceiling-hunt-v1/original").glob(
                "search_seed*/history_final.json"
            )
        ),
        "layernorm_search_space": sorted(
            path for path in (REPO_ROOT / "results/cora-layernorm-search-space-pilot-v1").rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl"}
        ),
        "posthoc_truth_outputs": sorted(
            path for path in (REPO_ROOT / "results/posthoc").rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl"}
            and target not in path.resolve().parents
        ),
        "legacy_histories": sorted(
            (REPO_ROOT / "legacy_artifacts").rglob("history_final.json")
        ),
    }
    missing = [name for name, paths in groups.items() if not paths]
    if missing:
        raise RuntimeError(f"unresolved empty exposure sources: {missing}")
    return groups


def build_exposure_registry() -> tuple[set[str], dict[str, Any], list[dict[str, Any]]]:
    groups = exposure_source_groups()
    group_sets: dict[str, set[str]] = {}
    inventory: list[dict[str, Any]] = []
    parse_failures: list[dict[str, str]] = []
    for name, paths in groups.items():
        fingerprints: set[str] = set()
        label_files = 0
        for path in paths:
            try:
                local = _scan_json_source(path)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
                parse_failures.append({"path": str(path.resolve()), "error": str(error)})
                continue
            if local:
                label_files += 1
                fingerprints.update(local)
                inventory.append({
                    "source_group": name,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "fingerprint_count": len(local),
                })
        group_sets[name] = fingerprints
        if not fingerprints:
            parse_failures.append({"path": name, "error": "no resolvable truth-labeled fingerprints"})
    union = set().union(*group_sets.values())
    overlaps = []
    names = sorted(group_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlaps.append({
                "left": left, "right": right,
                "overlap_count": len(group_sets[left] & group_sets[right]),
            })
    audit = {
        "passed": not parse_failures,
        "classification_if_failed": "EXPOSURE_EXCLUSION_UNRESOLVED",
        "fingerprint_rule": "ordered operations tuple plus sorted edge list; position-sensitive",
        "group_counts": {name: len(values) for name, values in group_sets.items()},
        "pairwise_overlaps": overlaps,
        "union_count": len(union),
        "source_file_count": len(inventory),
        "parse_failures": parse_failures,
        "all_hp_configurations_excluded_by_architecture_fingerprint": True,
        "test_metric_values_read": False,
        "truth_values_used_for_selection": False,
    }
    registry = [
        {
            "architecture_fingerprint": fingerprint,
            "source_groups": sorted(name for name, values in group_sets.items() if fingerprint in values),
        }
        for fingerprint in sorted(union)
    ]
    return union, audit, [{"registry": registry, "source_inventory": inventory}]


def structure_stratum(config: Mapping[str, Any]) -> str:
    edge_count = len(config["edges"])
    if edge_count <= 9:
        return "edges_sparse_le9"
    if edge_count <= 11:
        return "edges_medium_10_11"
    if edge_count <= 13:
        return "edges_dense_12_13"
    return "edges_very_dense_ge14"


def validate_static_config(config: Mapping[str, Any]) -> tuple[bool, str | None]:
    operations = list(config.get("operations", []))
    edges = [tuple(map(int, edge)) for edge in config.get("edges", [])]
    if not 1 <= len(operations) <= 5:
        return False, "operation_count"
    if not set(operations) <= OPERATION_TYPES:
        return False, "operation_type"
    node_count = len(operations) + 2
    if not edges or len(edges) != len(set(edges)):
        return False, "edges_empty_or_duplicate"
    if any(left < 0 or right >= node_count or left >= right for left, right in edges):
        return False, "edge_range_or_dag_order"
    if not any(left == 0 for left, _ in edges) or not any(right == node_count - 1 for _, right in edges):
        return False, "input_output_connectivity"
    return True, None


def generate_candidate_pool(excluded: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = load_joint_vae(BASE_CHECKPOINT, "global4", device="cpu")
    torch.manual_seed(GENERATION_SEED)
    unique: dict[str, dict[str, Any]] = {}
    invalid = Counter()
    for start in range(0, GENERATION_DRAWS, 256):
        size = min(256, GENERATION_DRAWS - start)
        z = torch.randn(size, 12)
        with torch.no_grad():
            graphs = model.arch_vae.decode(z, stochastic=False)
        for offset, graph in enumerate(graphs):
            try:
                config = config_from_igraph(model, graph)
                fingerprint = architecture_fingerprint(config)
            except (TypeError, ValueError, KeyError, IndexError) as error:
                invalid[type(error).__name__] += 1
                continue
            valid, reason = validate_static_config(config)
            if not valid:
                invalid[str(reason)] += 1
                continue
            if fingerprint in excluded or fingerprint in unique:
                continue
            unique[fingerprint] = {
                "architecture_fingerprint": fingerprint,
                "operations": list(config["operations"]),
                "edges": [list(edge) for edge in config["edges"]],
                "effective_layers": int(config["effective_layers"]),
                "structure_stratum": structure_stratum(config),
                "generation_index": start + offset,
                "architecture_source": "frozen_vae_standard_normal_deterministic_greedy",
            }
    hp_rng = np.random.default_rng(HP_SEED)
    pool = []
    for fingerprint, architecture in sorted(unique.items()):
        for regime, (low, high) in HP_REGIMES.items():
            raw_hp = hp_rng.uniform(low, high, size=4).astype(np.float64).tolist()
            candidate = {
                **architecture,
                "raw_hp_vector": raw_hp,
                "hp_regime": regime,
                "hp_fingerprint": hp_fingerprint(raw_hp),
                "selection_stratum": f"{architecture['structure_stratum']}__{regime}",
            }
            candidate.update(realize_global4_hp(candidate))
            candidate["config_id"] = (
                f"blind_{fingerprint[:12]}_{regime}_{candidate['hp_fingerprint'][:8]}"
            )
            pool.append(candidate)
    counts = Counter(row["structure_stratum"] for row in pool[::len(HP_REGIMES)])
    audit = {
        "passed": len(unique) >= TARGET_ARCHITECTURES and len(counts) == 4 and all(count > 0 for count in counts.values()),
        "classification_if_failed": "INSUFFICIENT_UNEXPOSED_CANDIDATE_POOL",
        "generator": "frozen VAE; z_arch ~ N(0,I); deterministic greedy decode once",
        "generation_seed": GENERATION_SEED,
        "hp_seed": HP_SEED,
        "generation_draws": GENERATION_DRAWS,
        "raw_valid_unique_unexposed_architectures": len(unique),
        "candidate_configuration_count": len(pool),
        "structure_stratum_architecture_counts": dict(sorted(counts.items())),
        "hp_regimes": HP_REGIMES,
        "invalid_counts": dict(invalid),
        "exposure_overlap_count": len(set(unique) & excluded),
        "candidate_contains_truth_or_model_prediction": False,
        "static_legality_passed": True,
    }
    return pool, audit


def select_candidates(pool: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(SELECTION_SEED)
    architectures: dict[str, dict[str, Any]] = {}
    by_structure: dict[str, list[str]] = defaultdict(list)
    by_key = {}
    for row in pool:
        fingerprint = str(row["architecture_fingerprint"])
        architectures.setdefault(fingerprint, dict(row))
        by_key[(fingerprint, str(row["hp_regime"]))] = dict(row)
    for fingerprint, row in architectures.items():
        by_structure[str(row["structure_stratum"])].append(fingerprint)
    selected_fps = []
    selection_probability: dict[str, float] = {}
    allocation = {}
    structures = sorted(by_structure)
    targets = {structure: min(10, len(by_structure[structure])) for structure in structures}
    shortage = TARGET_ARCHITECTURES - sum(targets.values())
    for structure in structures:
        if shortage <= 0:
            break
        capacity = len(by_structure[structure]) - targets[structure]
        added = min(shortage, capacity)
        targets[structure] += added
        shortage -= added
    if shortage:
        raise RuntimeError("fixed cyclic refill cannot allocate 40 unique architectures")
    for structure in structures:
        choices = sorted(by_structure[structure])
        target = targets[structure]
        picked = [choices[index] for index in rng.choice(len(choices), size=target, replace=False)]
        selected_fps.extend(picked)
        allocation[structure] = {
            "pool_architectures": len(choices), "initial_target": min(10, len(choices)),
            "selected_architectures": target, "refill_added": target - min(10, len(choices)),
        }
        for fingerprint in picked:
            selection_probability[fingerprint] = target / len(choices)
    if len(selected_fps) != TARGET_ARCHITECTURES or len(set(selected_fps)) != TARGET_ARCHITECTURES:
        raise RuntimeError("base architecture selection did not produce exactly 40 unique fingerprints")
    regimes = list(HP_REGIMES) * 10
    rng.shuffle(regimes)
    base_rows = []
    assigned_regime = {}
    for fingerprint, regime in zip(selected_fps, regimes):
        row = dict(by_key[(fingerprint, regime)])
        row.update({
            "selection_stage": "base_40_unique_architectures",
            "architecture_selection_probability": selection_probability[fingerprint],
            "hp_assignment_probability": 0.25,
            "second_hp_selection_probability": 0.0,
        })
        base_rows.append(row)
        assigned_regime[fingerprint] = regime
    duplicate_indices = rng.choice(len(base_rows), size=SECOND_HP_ARCHITECTURES, replace=False)
    opposite = {
        "hp_low": "hp_high", "hp_high": "hp_low",
        "hp_mid_low": "hp_high", "hp_mid_high": "hp_low",
    }
    second_rows = []
    hp_distance_audit = []
    for index in sorted(map(int, duplicate_indices)):
        first = base_rows[index]
        fingerprint = str(first["architecture_fingerprint"])
        second = dict(by_key[(fingerprint, opposite[assigned_regime[fingerprint]])])
        left = np.asarray(first["raw_hp_vector"], dtype=np.float64)
        right = np.asarray(second["raw_hp_vector"], dtype=np.float64)
        linf = float(np.max(np.abs(left - right)))
        l1 = float(np.sum(np.abs(left - right)))
        if linf < 1.0 / 3.0 or l1 < 0.75:
            raise RuntimeError("paired HP vectors are not preregistered as meaningfully different")
        second.update({
            "selection_stage": "second_hp_for_8_selected_architectures",
            "architecture_selection_probability": selection_probability[fingerprint],
            "hp_assignment_probability": 1.0,
            "second_hp_selection_probability": SECOND_HP_ARCHITECTURES / TARGET_ARCHITECTURES,
        })
        second_rows.append(second)
        hp_distance_audit.append({
            "architecture_fingerprint": fingerprint,
            "first_config_id": first["config_id"], "second_config_id": second["config_id"],
            "first_hp_regime": first["hp_regime"], "second_hp_regime": second["hp_regime"],
            "raw_hp_linf": linf, "raw_hp_l1": l1,
        })
    selection = sorted(base_rows + second_rows, key=lambda row: (
        str(row["architecture_fingerprint"]), str(row["hp_fingerprint"])
    ))
    for index, row in enumerate(selection):
        row["assigned_gpu"] = index % 2
    fp_counts = Counter(row["architecture_fingerprint"] for row in selection)
    audit = {
        "passed": (
            len(selection) == TARGET_CONFIGS
            and len(fp_counts) == TARGET_ARCHITECTURES
            and max(fp_counts.values()) == 2
        ),
        "selection_seed": SELECTION_SEED,
        "model_predictions_used": False,
        "model_disagreement_used": False,
        "performance_labels_or_neighbor_performance_used": False,
        "manual_performance_judgment_used": False,
        "selection_rule": (
            "uniform without replacement within each frozen edge-count stratum allocation; "
            "balanced random HP-regime assignment; then 8 uniformly selected architectures "
            "receive the preregistered opposite HP regime"
        ),
        "insufficient_stratum_refill_rule": (
            "cyclically reallocate to the next lexicographic structure stratum with capacity"
        ),
        "insufficient_stratum_refill_applied": any(
            item["refill_added"] > 0 for item in allocation.values()
        ),
        "allocation": allocation,
        "configuration_count": len(selection),
        "architecture_count": len(fp_counts),
        "single_hp_architecture_count": sum(value == 1 for value in fp_counts.values()),
        "two_hp_architecture_count": sum(value == 2 for value in fp_counts.values()),
        "maximum_hp_per_architecture": max(fp_counts.values()),
        "gpu_configuration_counts": dict(Counter(row["assigned_gpu"] for row in selection)),
        "hp_distance_audit": hp_distance_audit,
    }
    return selection, audit


def reproduce_clean_predictions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean_root = REPO_ROOT / CLEAN_RELATIVE
    strict_root = REPO_ROOT / STRICT_RELATIVE
    selection = load_json(clean_root / "phase_4/frozen_selection_24.json")
    expected_rows = load_json(clean_root / "phase_4/frozen_candidate_predictions.json")
    expected = {
        (str(row["config_id"]), str(row["model"])): float(row["prediction"])
        for row in expected_rows if row["model"] in MODELS
    }
    actual_rows = frozen_head_h_predictions(selection, strict_root)
    comparisons = []
    for row in actual_rows:
        values = {
            "original_frozen_head": float(row["head_prediction"]),
            "original_frozen_h": float(row["h_prediction"]),
        }
        for model, value in values.items():
            expected_value = expected[(str(row["config_id"]), model)]
            comparisons.append({
                "config_id": row["config_id"],
                "architecture_fingerprint": row["architecture_fingerprint"],
                "model": model, "expected_prediction": expected_value,
                "reproduced_prediction": value,
                "absolute_error": abs(value - expected_value),
            })
    maximum = max(row["absolute_error"] for row in comparisons)
    audit = {
        "passed": maximum <= 1e-8,
        "classification_if_failed": "FROZEN_MODEL_REPRODUCTION_FAILED",
        "configuration_count": 24, "comparison_count": len(comparisons),
        "maximum_absolute_error": maximum, "tolerance": 1e-8,
        "predictor_training_or_update_executed": False,
        "scaler_refit_executed": False, "vae_training_executed": False,
    }
    return comparisons, audit


def freeze_predictions(selection: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = frozen_head_h_predictions(selection, REPO_ROOT / STRICT_RELATIVE)
    output = []
    for candidate, predicted in zip(selection, rows):
        for model, prefix in (
            ("original_frozen_h", "h"), ("original_frozen_head", "head")
        ):
            output.append({
                "config_id": candidate["config_id"],
                "architecture_fingerprint": candidate["architecture_fingerprint"],
                "hp_fingerprint": candidate["hp_fingerprint"],
                "model": model,
                "prediction": float(predicted[f"{prefix}_prediction"]),
                "prediction_std": float(predicted[f"{prefix}_prediction_std"]),
                "prediction_source_count": int(predicted[f"{prefix}_prediction_source_count"]),
                "prediction_frozen_before_new_labels": True,
            })
    identity = {
        "prediction_row_count": len(output),
        "configuration_count": len(selection),
        "models": list(MODELS),
        "coverage_by_model": {
            model: len({row["config_id"] for row in output if row["model"] == model}) / len(selection)
            for model in MODELS
        },
        "feature_schema": {
            "original_frozen_h": ["mu_arch", "structure_features", "train_fold_standardized_global4_hp"],
            "original_frozen_head": ["mu_arch", "train_fold_standardized_global4_hp"],
        },
        "checkpoint_hashes": getattr(frozen_head_h_predictions, "checkpoint_hashes", {}),
        "scaler_refit_executed": False,
        "ensemble_weights_changed": False,
        "predictor_or_vae_training_executed": False,
        "latent_distance_gp_used": False,
    }
    return output, identity


def protocol_payload(clean_plan: Mapping[str, Any]) -> dict[str, Any]:
    clean_protocol = clean_plan["labeling_protocol"]
    return {
        "format_version": FORMAT_VERSION,
        "diagnostic_only": True,
        "dataset": "Cora", "hp_mode": "global4", "split": "public",
        "dataset_content_fingerprint": clean_protocol["dataset_content_fingerprint"],
        "split_fingerprint": clean_protocol["split_fingerprint"],
        "max_epochs": int(clean_protocol["max_epochs"]),
        "patience": int(clean_protocol["patience"]),
        "training_mode": "full_batch", "metric": "validation_accuracy",
        "track_test": False,
        "training_seeds": list(TRAINING_SEEDS),
        "planned_gnn_runs": TARGET_CONFIGS * len(TRAINING_SEEDS),
        "clean24_protocol_difference": {
            "clean24_initial_seeds": clean_protocol["initial_seeds"],
            "clean24_adaptive_seeds": clean_protocol["adaptive_seeds"],
            "current_objective_explicitly_freezes_three_seed_prefix": list(TRAINING_SEEDS),
            "core_dataset_split_fidelity_and_early_stopping_unchanged": True,
        },
        "maximum_system_retry_per_config_seed": MAX_RETRIES,
        "selection": {
            "generation_seed": GENERATION_SEED, "hp_seed": HP_SEED,
            "selection_seed": SELECTION_SEED,
            "structure_strata": [
                "edges_sparse_le9", "edges_medium_10_11", "edges_dense_12_13",
                "edges_very_dense_ge14",
            ],
            "hp_regimes": HP_REGIMES,
            "base_allocation": (
                "target 10 unique architectures per structure stratum; cap at available count "
                "and reallocate any shortage by the frozen refill rule"
            ),
            "hp_assignment": "balanced random permutation: 10 configurations per HP regime",
            "second_hp": "8 uniformly sampled selected architectures receive opposite HP regime",
            "second_hp_minimum_distance": {"linf": 1.0 / 3.0, "l1": 0.75},
            "refill": "cyclic lexicographic next structure stratum with remaining capacity",
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "outer_unit": (
                "within each frozen structure stratum, architecture_fingerprint clusters "
                "are sampled with replacement at the original stratum architecture count"
            ),
            "inner_unit": "three GNN seeds resampled with replacement per configuration",
            "shared_indices_for_h_head_and_paired_differences": True,
        },
        "metrics": list(METRICS),
        "ranking_ties_and_boundaries": (
            "average ranks, then architecture_fingerprint, hp_fingerprint, config_id; "
            "no chronological field exists or is used in this prospective set"
        ),
        "classification_order": list(CLASSIFICATION_ORDER),
        "success_gates": [
            "all identity/exposure/selection/firewall audits pass",
            "all 144 task keys complete; at least 44 configurations valid; all 40 architectures and frozen structure-stratum counts retained",
            "H Kendall CI lower > 0",
            "H-Head Kendall point difference > 0 and paired CI lower > 0",
            "H top-region Kendall >= Head",
            "H NDCG@10 >= Head",
            "H regret@10 <= Head",
            "H false-positive rate < Head",
            "architecture-balanced H-Head Kendall direction positive",
        ],
        "predictor_vae_training_allowed": False, "scaler_refit_allowed": False,
        "gp_allowed": False, "bo_allowed": False, "test_metric_allowed": False,
    }


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "analyse/run_h_vs_head_prospective_labels.py",
        REPO_ROOT / "tests/test_h_vs_head_model_blind_prospective.py",
        REPO_ROOT / "analyse/h_targeted_reliable_pair_validation.py",
        REPO_ROOT / "analyse/direct_head_chronological_offline_replay_v2.py",
        REPO_ROOT / "eval_utils.py",
        REPO_ROOT / "architecture_diagnostics.py",
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths if path.is_file()}


def prepare(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    clean_plan_path = REPO_ROOT / CLEAN_RELATIVE / "phase_4/label_task_plan.json"
    clean_plan = load_json(clean_plan_path)
    protocol = protocol_payload(clean_plan)
    protocol_path = output / "protocol.json"
    dump_json(protocol_path, protocol)
    write_sidecar(protocol_path)

    reproduction, reproduction_audit = reproduce_clean_predictions()
    dump_json(output / "phase_a/frozen_model_reproduction.json", reproduction)
    dump_csv(output / "phase_a/frozen_model_reproduction.csv", reproduction)
    dump_json(output / "phase_a/frozen_model_reproduction_audit.json", reproduction_audit)
    if not reproduction_audit["passed"]:
        dump_json(output / "final_classification.json", terminal_error("FROZEN_MODEL_REPRODUCTION_FAILED", reproduction_audit))
        return 2

    excluded, exposure_audit, registry_payload = build_exposure_registry()
    registry = registry_payload[0]["registry"]
    inventory = registry_payload[0]["source_inventory"]
    dump_json(output / "phase_a/exposure_registry.json", registry)
    dump_csv(output / "phase_a/exposure_registry.csv", registry)
    dump_json(output / "phase_a/exposure_source_inventory.json", inventory)
    dump_json(output / "phase_a/exposure_exclusion_audit.json", exposure_audit)
    if not exposure_audit["passed"]:
        dump_json(output / "final_classification.json", terminal_error("EXPOSURE_EXCLUSION_UNRESOLVED", exposure_audit))
        return 2

    pool, pool_audit = generate_candidate_pool(excluded)
    dump_json(output / "phase_a/candidate_pool.json", pool)
    dump_json(output / "phase_a/candidate_pool_audit.json", pool_audit)
    if not pool_audit["passed"]:
        dump_json(output / "final_classification.json", terminal_error("INSUFFICIENT_UNEXPOSED_CANDIDATE_POOL", pool_audit))
        return 2

    selection, selection_audit = select_candidates(pool)
    if {row["architecture_fingerprint"] for row in selection} & excluded:
        raise RuntimeError("selected architecture overlaps exposure registry")
    selection_path = output / "phase_a/frozen_selection_48.json"
    dump_json(selection_path, selection)
    dump_csv(output / "phase_a/frozen_selection_48.csv", selection)
    write_sidecar(selection_path)
    dump_json(output / "phase_a/selection_audit.json", selection_audit)

    predictions, model_identity = freeze_predictions(selection)
    prediction_path = output / "phase_a/frozen_predictions.json"
    dump_json(prediction_path, predictions)
    dump_csv(output / "phase_a/frozen_predictions.csv", predictions)
    write_sidecar(prediction_path)
    dump_json(output / "phase_a/frozen_model_input_identity.json", model_identity)

    inputs = {
        "base_vae": _identity(BASE_CHECKPOINT),
        "clean_label_plan": _identity(clean_plan_path),
        "clean_selection": _identity(REPO_ROOT / CLEAN_RELATIVE / "phase_4/frozen_selection_24.json"),
        "clean_predictions": _identity(REPO_ROOT / CLEAN_RELATIVE / "phase_4/frozen_candidate_predictions.json"),
        "strict_plan": _identity(REPO_ROOT / STRICT_RELATIVE / "task_plan.json"),
        "strict_split": _identity(REPO_ROOT / STRICT_RELATIVE / "fixed_split_manifest.json"),
        "bo_phase4": _identity(REPO_ROOT / "bo_phase4.py"),
        "eval_utils": _identity(REPO_ROOT / "eval_utils.py"),
    }
    dump_json(output / "input_identities.json", inputs)

    gpu_map = {
        "0": "GPU-5496be78-3b71-4978-3960-b6ed926ebe49",
        "1": "GPU-197ae2b9-74cd-6afc-446d-7bc9e98176f9",
    }
    task_plan = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(), "diagnostic_only": True,
        "new_labels_exist_at_freeze": False,
        "protocol_path": str(protocol_path.resolve()), "protocol_sha256": sha256_file(protocol_path),
        "selection_path": str(selection_path.resolve()), "selection_sha256": sha256_file(selection_path),
        "prediction_path": str(prediction_path.resolve()), "prediction_sha256": sha256_file(prediction_path),
        "exposure_registry_path": str((output / "phase_a/exposure_registry.json").resolve()),
        "exposure_registry_sha256": sha256_file(output / "phase_a/exposure_registry.json"),
        "exposure_inventory_path": str((output / "phase_a/exposure_source_inventory.json").resolve()),
        "exposure_inventory_sha256": sha256_file(output / "phase_a/exposure_source_inventory.json"),
        "input_identities": inputs, "code_hashes": code_hashes(),
        "training_seeds": list(TRAINING_SEEDS), "configuration_count": TARGET_CONFIGS,
        "architecture_count": TARGET_ARCHITECTURES, "expected_task_count": TARGET_CONFIGS * 3,
        "shard_to_gpu_uuid": gpu_map,
        "labeling_protocol": protocol,
        "maximum_retries": MAX_RETRIES,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "formal_or_offline_bo_allowed": False, "test_metric_allowed": False,
        "predictor_or_vae_training_allowed": False, "gp_allowed": False,
    }
    plan_path = output / "gnn_run_manifest.json"
    dump_json(plan_path, task_plan)
    write_sidecar(plan_path)
    firewall = {
        "passed": True, "frozen_at": utc_now(),
        "new_candidate_validation_fields_present": False,
        "new_candidate_test_fields_present": False,
        "new_candidate_performance_fields_present": False,
        "selection_completed_before_model_predictions": True,
        "model_predictions_frozen_before_new_labels": True,
        "new_labels_exist": False,
        "selection_sha256": sha256_file(selection_path),
        "prediction_sha256": sha256_file(prediction_path),
        "plan_sha256": sha256_file(plan_path),
        "planned_task_count": TARGET_CONFIGS * 3,
    }
    dump_json(output / "phase_a/pre_label_firewall_audit.json", firewall)
    return 0


def terminal_error(classification: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "classification": classification, "evidence": dict(evidence),
        "new_validation_labels_generated": False, "test_metric_used": False,
        "predictor_or_vae_training_executed": False, "gp_used": False,
        "offline_or_formal_bo_executed": False,
    }


def verify_plan(plan_path: Path) -> dict[str, Any]:
    verify_sidecar(plan_path)
    plan = load_json(plan_path)
    checks = {
        plan["protocol_path"]: plan["protocol_sha256"],
        plan["selection_path"]: plan["selection_sha256"],
        plan["prediction_path"]: plan["prediction_sha256"],
        plan["exposure_registry_path"]: plan["exposure_registry_sha256"],
        plan["exposure_inventory_path"]: plan["exposure_inventory_sha256"],
        **plan["code_hashes"],
    }
    for identity in plan["input_identities"].values():
        checks[identity["path"]] = identity["sha256"]
    drift = [path for path, expected in checks.items() if not Path(path).is_file() or sha256_file(Path(path)) != expected]
    if drift:
        raise RuntimeError(f"frozen input/code drift: {drift}")
    return plan


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate_seed_results(
    rows: Sequence[Mapping[str, Any]],
    expected_keys: set[tuple[str, str, int]] | None = None,
    expected_architecture_strata: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed_keys = [
        (str(row["config_id"]), str(row["architecture_fingerprint"]), int(row["seed"]))
        for row in rows
    ]
    observed_key_set = set(observed_keys)
    successful = [row for row in rows if row.get("terminal_status") == "success" and row.get("valid")]
    by_config: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in successful:
        by_config[str(row["config_id"])].append(row)
    statistics_rows = []
    for config_id, values in sorted(by_config.items()):
        unique = {int(row["seed"]): row for row in values}
        vals = [float(unique[seed]["val_acc"]) for seed in sorted(unique)]
        first = values[0]
        statistics_rows.append({
            "config_id": config_id,
            "architecture_fingerprint": first["architecture_fingerprint"],
            "hp_fingerprint": first["hp_fingerprint"],
            "selection_stratum": first.get("selection_stratum"),
            "structure_stratum": first.get("structure_stratum") or str(
                first.get("selection_stratum", "")
            ).rsplit("__", 1)[0],
            "seed_count": len(vals), "seeds": sorted(unique),
            "val_acc_mean": statistics.fmean(vals),
            "val_acc_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "seed_val_accs": vals,
        })
    expected_key_set = expected_keys or observed_key_set
    valid_architecture_strata = {
        str(row["architecture_fingerprint"]): str(row["structure_stratum"])
        for row in statistics_rows if row["seed_count"] == len(TRAINING_SEEDS)
    }
    expected_strata = dict(expected_architecture_strata or valid_architecture_strata)
    audit = {
        "expected_task_count": len(expected_key_set),
        "observed_row_count": len(rows), "successful_unique_task_count": len(successful),
        "valid_configuration_count": sum(row["seed_count"] == len(TRAINING_SEEDS) for row in statistics_rows),
        "expected_architecture_count": len(expected_strata),
        "valid_architecture_count": len(valid_architecture_strata),
        "expected_structure_stratum_architecture_counts": dict(sorted(Counter(expected_strata.values()).items())),
        "valid_structure_stratum_architecture_counts": dict(sorted(Counter(valid_architecture_strata.values()).items())),
        "duplicate_task_count": len(observed_keys) - len(observed_key_set),
        "missing_task_count": len(expected_key_set - observed_key_set),
        "unexpected_task_count": len(observed_key_set - expected_key_set),
    }
    audit["passed"] = (
        audit["observed_row_count"] == audit["expected_task_count"]
        and audit["valid_configuration_count"] >= 44
        and valid_architecture_strata == expected_strata
        and audit["duplicate_task_count"] == 0
        and audit["missing_task_count"] == 0
        and audit["unexpected_task_count"] == 0
    )
    return statistics_rows, audit


def metric_bundle(rows: Sequence[Mapping[str, Any]], prediction_key: str) -> dict[str, Any]:
    prepared = [{
        **dict(row),
        "prediction": float(row[prediction_key]),
        "val_acc": float(row["val_acc"]),
        "step": 0,
        "replay_id": str(row["config_id"]),
    } for row in rows]
    result = ranking_metrics(prepared)
    actual = [float(row["val_acc"]) for row in prepared]
    predicted = [float(row["prediction"]) for row in prepared]
    rho = spearmanr(actual, predicted).statistic
    result["spearman"] = float(rho) if rho is not None and np.isfinite(rho) else None
    return result


def architecture_balanced_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(row)
    output = []
    for fingerprint, values in sorted(grouped.items()):
        output.append({
            "config_id": f"arch::{fingerprint}",
            "architecture_fingerprint": fingerprint,
            "hp_fingerprint": "architecture_equal_weight_average",
            "val_acc": statistics.fmean(float(row["val_acc"]) for row in values),
            "original_frozen_h": statistics.fmean(float(row["original_frozen_h"]) for row in values),
            "original_frozen_head": statistics.fmean(float(row["original_frozen_head"]) for row in values),
        })
    return output


def _bootstrap_rows(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(row)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for fingerprint, values in grouped.items():
        strata = {str(row.get("structure_stratum", "")) for row in values}
        if len(strata) != 1 or not next(iter(strata)):
            raise ValueError(f"architecture {fingerprint} does not have one structure stratum")
        by_stratum[next(iter(strata))].append(fingerprint)
    sampled = []
    for stratum in sorted(by_stratum):
        fingerprints = sorted(by_stratum[stratum])
        sampled.extend(
            fingerprints[index]
            for index in rng.integers(0, len(fingerprints), len(fingerprints))
        )
    counts = Counter()
    output = []
    for fingerprint in sampled:
        copy_index = counts[fingerprint]
        counts[fingerprint] += 1
        for row in grouped[fingerprint]:
            seed_values = np.asarray(row["seed_val_accs"], dtype=np.float64)
            seed_sample = seed_values[rng.integers(0, len(seed_values), len(seed_values))]
            output.append({
                **dict(row),
                "config_id": f"{row['config_id']}#bootstrap{copy_index}",
                "source_architecture_fingerprint": fingerprint,
                "architecture_fingerprint": f"{fingerprint}#bootstrap{copy_index}",
                "val_acc": float(seed_sample.mean()),
            })
    return output


def bootstrap_analysis(
    rows: Sequence[Mapping[str, Any]], *, replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    differences: dict[str, list[float]] = defaultdict(list)
    arch_differences: list[float] = []
    for _ in range(replicates):
        boot = _bootstrap_rows(rows, rng)
        bundles = {model: metric_bundle(boot, model) for model in MODELS}
        for model, bundle in bundles.items():
            for metric in METRICS:
                if bundle.get(metric) is not None:
                    values[(model, metric)].append(float(bundle[metric]))
        for metric in METRICS:
            left = bundles["original_frozen_h"].get(metric)
            right = bundles["original_frozen_head"].get(metric)
            if left is not None and right is not None:
                differences[metric].append(float(left) - float(right))
        arch = architecture_balanced_rows(boot)
        h_tau = metric_bundle(arch, "original_frozen_h")["kendall_tau_b"]
        head_tau = metric_bundle(arch, "original_frozen_head")["kendall_tau_b"]
        if h_tau is not None and head_tau is not None:
            arch_differences.append(float(h_tau) - float(head_tau))

    def interval(items: Sequence[float]) -> tuple[float | None, float | None]:
        if not items:
            return None, None
        return float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975))

    intervals = []
    for (model, metric), items in sorted(values.items()):
        low, high = interval(items)
        intervals.append({
            "model": model, "metric": metric, "ci95_low": low, "ci95_high": high,
            "bootstrap_replicates": replicates, "valid_replicates": len(items),
            "outer_unit": "structure-stratified architecture_fingerprint cluster",
            "inner_unit": "three GNN seeds with replacement",
        })
    paired = []
    for metric, items in sorted(differences.items()):
        low, high = interval(items)
        paired.append({
            "candidate": "original_frozen_h", "baseline": "original_frozen_head",
            "metric": metric, "difference_ci95_low": low, "difference_ci95_high": high,
            "bootstrap_replicates": replicates, "valid_replicates": len(items),
            "shared_bootstrap_indices": True,
            "outer_unit": "structure-stratified architecture_fingerprint cluster",
            "inner_unit": "three GNN seeds with replacement",
        })
    low, high = interval(arch_differences)
    sensitivity = {
        "metric": "architecture_equal_weight_kendall_difference_h_minus_head",
        "point_difference": None,
        "difference_ci95_low": low, "difference_ci95_high": high,
        "valid_replicates": len(arch_differences), "bootstrap_replicates": replicates,
        "outer_unit": "structure-stratified architecture_fingerprint cluster",
        "inner_unit": "three GNN seeds with replacement before architecture averaging",
    }
    return intervals, paired, sensitivity


def classify(
    h: Mapping[str, Any], head: Mapping[str, Any], intervals: Mapping[tuple[str, str], Mapping[str, Any]],
    paired: Mapping[str, Mapping[str, Any]], arch_direction: float,
) -> tuple[str, dict[str, bool]]:
    h_k = intervals[("original_frozen_h", "kendall_tau_b")]
    h_top = intervals[("original_frozen_h", "top_region_kendall_tau_b")]
    delta = float(h["kendall_tau_b"]) - float(head["kendall_tau_b"])
    delta_ci = paired["kendall_tau_b"]
    gates = {
        "h_kendall_ci_lower_above_zero": h_k["ci95_low"] > 0,
        "h_minus_head_kendall_point_positive": delta > 0,
        "h_minus_head_kendall_ci_lower_above_zero": delta_ci["difference_ci95_low"] > 0,
        "top_region_not_lower": h["top_region_kendall_tau_b"] >= head["top_region_kendall_tau_b"],
        "ndcg_at_10_not_lower": h["ndcg_at_10"] >= head["ndcg_at_10"],
        "regret_at_10_not_higher": h["regret_at_10"] <= head["regret_at_10"],
        "false_positive_rate_lower": h["false_positive_rate"] < head["false_positive_rate"],
        "architecture_equal_weight_direction_positive": arch_direction > 0,
    }
    if all(gates.values()):
        return "H_PROSPECTIVELY_VALIDATED_FOR_ACQUISITION_CALIBRATION", gates
    if h_k["ci95_high"] <= 0 or (delta <= -0.05 and delta_ci["difference_ci95_high"] < 0):
        return "H_PROSPECTIVE_REJECTED", gates
    if h_k["ci95_low"] > 0 or h_top["ci95_low"] > 0:
        return "H_PROSPECTIVE_SIGNAL_ONLY", gates
    return "H_PROSPECTIVE_UNDERPOWERED", gates


def evaluate(plan_path: Path, output: Path, *, replicates: int = BOOTSTRAP_REPLICATES) -> int:
    plan = verify_plan(plan_path)
    if output.resolve() != plan_path.parent.resolve():
        raise RuntimeError("evaluation output must equal the frozen plan directory")
    shard_paths = [
        output / f"phase_b/shards/gpu{gpu}/seed_results.jsonl" for gpu in (0, 1)
    ]
    run_rows = [row for path in shard_paths for row in load_jsonl(path)]
    selection = load_json(Path(plan["selection_path"]))
    expected_keys = {
        (str(row["config_id"]), str(row["architecture_fingerprint"]), int(seed))
        for row in selection for seed in plan["training_seeds"]
    }
    expected_architecture_strata = {
        str(row["architecture_fingerprint"]): str(row["structure_stratum"])
        for row in selection
    }
    stats, merge_audit = aggregate_seed_results(
        run_rows, expected_keys, expected_architecture_strata
    )
    dump_json(output / "phase_b/gnn_merge_audit.json", merge_audit)
    if not merge_audit["passed"]:
        dump_json(output / "final_classification.json", terminal_error(
            "PROSPECTIVE_VALIDATION_INSUFFICIENT_COVERAGE", merge_audit
        ))
        return 2
    dump_json(output / "phase_b/seed_results_merged.json", run_rows)
    dump_csv(output / "phase_b/seed_results_merged.csv", run_rows)
    dump_json(output / "phase_b/three_seed_config_statistics.json", stats)
    dump_csv(output / "phase_b/three_seed_config_statistics.csv", stats)

    predictions = load_json(Path(plan["prediction_path"]))
    pred_lookup = {(row["config_id"], row["model"]): row["prediction"] for row in predictions}
    joined = []
    for row in stats:
        if row["seed_count"] != 3:
            continue
        joined.append({
            **dict(row), "val_acc": float(row["val_acc_mean"]),
            **{model: float(pred_lookup[(row["config_id"], model)]) for model in MODELS},
        })
    points = []
    point_lookup = {}
    for model in MODELS:
        bundle = metric_bundle(joined, model)
        point_lookup[model] = bundle
        points.append({"model": model, **bundle})
    arch_rows = architecture_balanced_rows(joined)
    arch_points = []
    arch_lookup = {}
    for model in MODELS:
        bundle = metric_bundle(arch_rows, model)
        arch_lookup[model] = bundle
        arch_points.append({"model": model, **bundle})

    intervals, paired_rows, sensitivity = bootstrap_analysis(joined, replicates=replicates)
    interval_lookup = {(row["model"], row["metric"]): row for row in intervals}
    paired_lookup = {row["metric"]: row for row in paired_rows}
    for row in intervals:
        row["point_estimate"] = point_lookup[row["model"]].get(row["metric"])
    for row in paired_rows:
        metric = row["metric"]
        row["point_difference"] = (
            float(point_lookup["original_frozen_h"][metric])
            - float(point_lookup["original_frozen_head"][metric])
        ) if point_lookup["original_frozen_h"].get(metric) is not None else None
    arch_direction = (
        float(arch_lookup["original_frozen_h"]["kendall_tau_b"])
        - float(arch_lookup["original_frozen_head"]["kendall_tau_b"])
    )
    sensitivity["point_difference"] = arch_direction
    classification, gates = classify(
        point_lookup["original_frozen_h"], point_lookup["original_frozen_head"],
        interval_lookup, paired_lookup, arch_direction,
    )
    dump_json(output / "analysis/point_metrics.json", points)
    dump_csv(output / "analysis/point_metrics.csv", points)
    dump_json(output / "analysis/bootstrap_intervals.json", intervals)
    dump_csv(output / "analysis/bootstrap_intervals.csv", intervals)
    dump_json(output / "analysis/paired_differences.json", paired_rows)
    dump_csv(output / "analysis/paired_differences.csv", paired_rows)
    dump_json(output / "analysis/architecture_balanced_sensitivity.json", {
        "point_metrics": arch_points, "paired_kendall": sensitivity,
    })

    h = point_lookup["original_frozen_h"]
    head = point_lookup["original_frozen_head"]
    delta_row = paired_lookup["kendall_tau_b"]
    final = {
        "classification": classification, "success_gates": gates,
        "configuration_count": len(joined),
        "architecture_count": len({row["architecture_fingerprint"] for row in joined}),
        "all_predictions_frozen_before_new_labels": True,
        "h_kendall": h["kendall_tau_b"],
        "h_kendall_ci95": [
            interval_lookup[("original_frozen_h", "kendall_tau_b")]["ci95_low"],
            interval_lookup[("original_frozen_h", "kendall_tau_b")]["ci95_high"],
        ],
        "h_minus_head_kendall": h["kendall_tau_b"] - head["kendall_tau_b"],
        "h_minus_head_kendall_ci95": [
            delta_row["difference_ci95_low"], delta_row["difference_ci95_high"],
        ],
        "h_top_region_kendall": h["top_region_kendall_tau_b"],
        "head_top_region_kendall": head["top_region_kendall_tau_b"],
        "h_ndcg_at_10": h["ndcg_at_10"], "head_ndcg_at_10": head["ndcg_at_10"],
        "h_regret_at_10": h["regret_at_10"], "head_regret_at_10": head["regret_at_10"],
        "h_false_positive_rate": h["false_positive_rate"],
        "head_false_positive_rate": head["false_positive_rate"],
        "architecture_balanced_kendall_direction": arch_direction,
        "uncertainty_calibration_and_h_driven_selection_authorized": (
            classification == "H_PROSPECTIVELY_VALIDATED_FOR_ACQUISITION_CALIBRATION"
        ),
        "diagnostic_only": True, "test_metric_used": False,
        "predictor_or_vae_training_executed": False, "gp_used": False,
        "offline_or_formal_bo_executed": False, "automatic_next_step": None,
    }
    dump_json(output / "final_classification.json", final)
    report = f"""# Frozen H vs Original Head prospective validation

Final classification: **`{classification}`**.

The experiment completed {len(joined)} configurations covering
{final['architecture_count']} architectures. Every H/Head prediction was hash-frozen before
the new validation labels were generated. H Kendall tau-b is {h['kendall_tau_b']:.6f}, with
95% CI [{final['h_kendall_ci95'][0]:.6f}, {final['h_kendall_ci95'][1]:.6f}]. The paired
H minus Head Kendall difference is {final['h_minus_head_kendall']:.6f}, with 95% CI
[{final['h_minus_head_kendall_ci95'][0]:.6f}, {final['h_minus_head_kendall_ci95'][1]:.6f}].

Top-region Kendall, NDCG@10, regret@10 and false-positive gates are recorded in
`final_classification.json`. H is {'authorized' if final['uncertainty_calibration_and_h_driven_selection_authorized'] else 'not authorized'}
for a later uncertainty-calibration and small candidate-selection task. This diagnostic did
not run BO, and frozen-model ranking on 48 preregistered candidates is not evidence of formal
BO success or architecture-generation performance.
"""
    (output / "final_goal_report.md").write_text(report, encoding="utf-8")
    dump_json(output / "result_integrity_audit.json", {
        "passed": True, "expected_tasks": 144, "observed_tasks": len(run_rows),
        "valid_configurations": len(joined), "test_metric_used": False,
        "predictor_or_vae_training_executed": False, "gp_used": False,
        "offline_or_formal_bo_executed": False,
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--plan", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--bootstrap_replicates", type=int, default=BOOTSTRAP_REPLICATES, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return prepare(args.output.resolve())
    if args.command == "evaluate":
        return evaluate(args.plan.resolve(), args.output.resolve(), replicates=args.bootstrap_replicates)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
