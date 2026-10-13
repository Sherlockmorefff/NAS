#!/usr/bin/env python3
"""Frozen-head chronological ranking replay with a strict label firewall.

``prepare`` performs every operation that must precede access to historical
validation labels: frozen-model reproduction, architecture exclusion,
protocol/schema audit, label-free history projection, architecture encoding,
and frozen prediction generation.  ``evaluate`` is a separate command that
will only attach ``val_acc`` after verifying the frozen plan and every input
and code hash.

The module never trains a predictor or VAE, never calls a BO entry point, and
never reconstructs the unavailable Zs RBF-GP posterior.  When no serialized
Zs model can legally cover the replay candidates, preparation fails closed as
``INSUFFICIENT_REPLAY_COVERAGE`` before any historical label is read.
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
import random
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import architecture_fingerprint
from analyse.h_targeted_reliable_pair_validation import (
    frozen_head_h_predictions,
    sha256_file,
)
from analyse.performance_aware_vae_strict_regression import StrictPerformanceHead
from analyse.structure_native_surrogate_pilot import (
    encode_selected_mu,
    predict_head,
)


FORMAT_VERSION = "direct_head_chronological_offline_replay_global4_v1"
OUTPUT_RELATIVE = "results/posthoc/direct-head-chronological-offline-replay-global4-v1"
STRUCTURE_RELATIVE = "results/posthoc/structure-native-surrogate-pilot-global4-v1"
QUARANTINE_RELATIVE = (
    "results/posthoc/structure-native-surrogate-pilot-global4-v1-"
    "invalid-freeze-audit-20260823"
)
STRICT_RELATIVE = "results/posthoc/performance-aware-vae-strict-regression-global4-v2"
H_TARGETED_RELATIVE = "results/posthoc/h-targeted-reliable-pair-validation-global4-v1"
UNIFIED_RELATIVE = "results/posthoc/vae-label-expansion-global4-v1"
HISTORY_RELATIVE = "results/cora-original-ceiling-hunt-v1/original"
MODEL_NAMES = (
    "refit_head",
    "original_frozen_head",
    "original_frozen_h",
    "original_zs_fixed_surrogate",
    "fixed_random_ranking",
)
PREDICTOR_MODELS = MODEL_NAMES[:4]
PREFIX_SPECS = {
    "prefix_20": ("prefix", 100),
    "prefix_40": ("prefix", 200),
    "prefix_60": ("prefix", 300),
    "prefix_80": ("prefix", 400),
    "prefix_100": ("prefix", 500),
    "late_50": ("suffix", 450),
    "late_100": ("suffix", 400),
}
GAP_THRESHOLDS = (0.006, 0.012, 0.020)
TOP_K_VALUES = (1, 3, 5, 10)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260825
RANDOM_BASELINE_DRAWS = 1000
RANDOM_BASELINE_SEED = 20260826
REPRODUCTION_TOLERANCE = 1e-8
FORBIDDEN_PROJECTION_KEY_PARTS = (
    "val",
    "validation",
    "test",
    "value",
    "reward",
    "rank",
    "performance",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(stable_json_hash(list(parts))[:8], 16)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
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


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    temporary.replace(path)


def write_sha256_sidecar(path: Path) -> Path:
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def recursively_named(value: Any, parts: Sequence[str]) -> list[str]:
    hits: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                key_text = str(key).lower()
                nested_path = f"{path}.{key}" if path else str(key)
                if any(part in key_text for part in parts):
                    hits.append(nested_path)
                visit(nested, nested_path)
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")

    visit(value, "")
    return hits


def assert_projection_is_label_free(rows: Sequence[Mapping[str, Any]]) -> None:
    hits = recursively_named(rows, FORBIDDEN_PROJECTION_KEY_PARTS)
    if hits:
        raise ValueError(f"label/performance field leaked into projection: {hits[:5]}")


def workspace_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            list(args), cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True,
        )
        return result.stdout.rstrip("\n")

    return {
        "captured_at": utc_now(),
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short").splitlines(),
    }


def history_paths() -> list[Path]:
    paths = [
        REPO_ROOT / HISTORY_RELATIVE / f"search_seed{seed}/history_final.json"
        for seed in range(5)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing history files: {missing}")
    return paths


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def collect_input_identities() -> dict[str, Any]:
    structure = REPO_ROOT / STRUCTURE_RELATIVE
    strict = REPO_ROOT / STRICT_RELATIVE
    h_targeted = REPO_ROOT / H_TARGETED_RELATIVE
    unified = REPO_ROOT / UNIFIED_RELATIVE
    quarantine = REPO_ROOT / QUARANTINE_RELATIVE
    named = {
        "bo_phase4": REPO_ROOT / "bo_phase4.py",
        "eval_utils": REPO_ROOT / "eval_utils.py",
        "base_vae": REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth",
        "refit_checkpoint_manifest": structure / "models/model_checkpoint_manifest.json",
        "refit_training_results": structure / "models/model_training_results.json",
        "refit_development_manifest": structure / "development/development_80_manifest.json",
        "clean24_selection": structure / "phase_4/frozen_selection_24.json",
        "clean24_frozen_predictions": structure / "phase_4/frozen_candidate_predictions.json",
        "clean24_final_classification": structure / "final_classification.json",
        "quarantine24_selection": quarantine / "phase_4/frozen_selection_24.json",
        "quarantine_reason": quarantine / "quarantine_reason.json",
        "strict_task_plan": strict / "task_plan.json",
        "strict_split_manifest": strict / "fixed_split_manifest.json",
        "strict_validation_predictions": strict / "all_validation_predictions.csv",
        "h_targeted_selection": h_targeted / "phase_b/targeted_selection.json",
        "h_targeted_frozen_predictions": h_targeted / "phase_b/frozen_prospective_predictions.csv",
        "unified_training_data": unified / "unified_training_data_no_test.json",
    }
    for seed, path in enumerate(history_paths()):
        named[f"history_seed_{seed}"] = path
        for filename in ("dataset_manifest.json", "history_metadata.json", "run_provenance.json"):
            companion = path.parent / filename
            if companion.is_file():
                named[f"history_seed_{seed}_{filename}"] = companion
    missing = [str(path) for path in named.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")

    checkpoint_paths = sorted(
        (strict / "checkpoints/full/group_repeated").glob(
            "group_*/seed_*/frozen_head/checkpoint.pth"
        )
    ) + sorted(
        (strict / "checkpoints/full/group_repeated").glob(
            "group_*/seed_*/frozen_hybrid_head/checkpoint.pth"
        )
    )
    if len(checkpoint_paths) != 300:
        raise RuntimeError(f"expected 300 original Head/H checkpoints, found {len(checkpoint_paths)}")
    refit_manifest = load_json(named["refit_checkpoint_manifest"])
    refit_paths = [
        Path(row["checkpoint_path"])
        for row in refit_manifest
        if row["model"] == "refit_head" and row["weight_mode"] == "architecture_balanced"
    ]
    if len(refit_paths) != 60:
        raise RuntimeError(f"expected 60 main Refit Head checkpoints, found {len(refit_paths)}")
    identities = {name: _identity(path) for name, path in sorted(named.items())}
    identities["original_head_h_checkpoints"] = [_identity(path) for path in checkpoint_paths]
    identities["refit_head_checkpoints"] = [_identity(path) for path in sorted(refit_paths)]
    return identities


def _load_clean24() -> list[dict[str, Any]]:
    path = REPO_ROOT / STRUCTURE_RELATIVE / "phase_4/frozen_selection_24.json"
    rows = [dict(row) for row in load_json(path)]
    if len(rows) != 24 or len({row["architecture_fingerprint"] for row in rows}) != 24:
        raise RuntimeError("clean prospective selection is not 24 unique architectures")
    return rows


def _refit_tasks() -> list[dict[str, Any]]:
    path = REPO_ROOT / STRUCTURE_RELATIVE / "models/model_training_results.json"
    tasks = [
        dict(row) for row in load_json(path)
        if row["model"] == "refit_head" and row["weight_mode"] == "architecture_balanced"
    ]
    if len(tasks) != 60:
        raise RuntimeError(f"expected 60 Refit Head tasks, found {len(tasks)}")
    return tasks


def frozen_refit_predictions(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]] | None = None
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    tasks = list(tasks or _refit_tasks())
    collected: list[np.ndarray] = []
    identities: list[dict[str, Any]] = []
    for task in tasks:
        checkpoint = Path(task["checkpoint_path"])
        before = sha256_file(checkpoint)
        if before != str(task["checkpoint_sha256"]):
            raise RuntimeError(f"Refit Head checkpoint drift: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("model_name") != "refit_head":
            raise RuntimeError(f"wrong checkpoint model: {checkpoint}")
        if payload.get("feature_schema") != ["mu_arch", "raw_hp_vector"]:
            raise RuntimeError(f"Refit Head feature schema drift: {checkpoint}")
        if payload.get("vae_trained") is not False or payload.get("latent_distance_gp_used") is not False:
            raise RuntimeError(f"forbidden state in Refit Head checkpoint: {checkpoint}")
        head = StrictPerformanceHead(int(payload["head_input_dim"]))
        head.load_state_dict(payload["head_state"])
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        predicted = predict_head(head, rows, "refit_head", payload["scaling"])
        collected.append(np.asarray(predicted, dtype=np.float64))
        after = sha256_file(checkpoint)
        if after != before:
            raise RuntimeError(f"checkpoint changed during inference: {checkpoint}")
        identities.append({
            "path": str(checkpoint.resolve()),
            "sha256": before,
            "split_id": task["split_id"],
            "training_seed": task["training_seed"],
            "feature_schema": payload["feature_schema"],
            "head_input_dim": int(payload["head_input_dim"]),
        })
    return np.stack(collected, axis=0).mean(axis=0), identities


def reproduce_clean24_predictions(output: Path) -> dict[str, Any]:
    clean = encode_selected_mu(_load_clean24())
    refit, refit_identities = frozen_refit_predictions(clean)
    original = frozen_head_h_predictions(clean, REPO_ROOT / STRICT_RELATIVE)
    original_lookup = {str(row["config_id"]): row for row in original}
    expected_rows = load_json(
        REPO_ROOT / STRUCTURE_RELATIVE / "phase_4/frozen_candidate_predictions.json"
    )
    expected = {
        (str(row["config_id"]), str(row["model"])): float(row["prediction"])
        for row in expected_rows
        if row["model"] in {"refit_head", "original_frozen_head", "original_frozen_h"}
    }
    comparisons: list[dict[str, Any]] = []
    for index, row in enumerate(clean):
        config_id = str(row["config_id"])
        actual = {
            "refit_head": float(refit[index]),
            "original_frozen_head": float(original_lookup[config_id]["head_prediction"]),
            "original_frozen_h": float(original_lookup[config_id]["h_prediction"]),
        }
        for model, value in actual.items():
            expected_value = expected[(config_id, model)]
            comparisons.append({
                "config_id": config_id,
                "architecture_fingerprint": row["architecture_fingerprint"],
                "model": model,
                "expected_prediction": expected_value,
                "reproduced_prediction": value,
                "absolute_error": abs(value - expected_value),
            })
    max_error = max(row["absolute_error"] for row in comparisons)
    passed = max_error <= REPRODUCTION_TOLERANCE
    phase = output / "phase_1"
    dump_json(phase / "frozen_model_reproduction.json", comparisons)
    dump_csv(phase / "frozen_model_reproduction.csv", comparisons)
    model_identity = {
        "passed": passed,
        "maximum_absolute_error": max_error,
        "tolerance": REPRODUCTION_TOLERANCE,
        "configuration_count": len(clean),
        "prediction_count": len(comparisons),
        "refit_head": {
            "checkpoint_count": len(refit_identities),
            "checkpoints": refit_identities,
            "input_fields": ["mu_arch", "raw_hp_vector"],
            "structure_features_used": False,
            "z_arch_used": False,
            "z_search_distance_used": False,
        },
        "original_head_h": {
            "checkpoint_count_per_model": 150,
            "checkpoint_hashes": getattr(frozen_head_h_predictions, "checkpoint_hashes", {}),
            "mu_encoding_source": "frozen base VAE encode(config_to_igraph(operations, edges))",
        },
        "parameter_training_or_update_executed": False,
        "vae_training_executed": False,
        "latent_distance_gp_used": False,
    }
    dump_json(phase / "model_identity_and_input_audit.json", model_identity)
    decision = {
        "classification": "PHASE_1_PASSED" if passed else "FROZEN_MODEL_REPRODUCTION_FAILED",
        "continue": passed,
        "maximum_absolute_error": max_error,
    }
    dump_json(phase / "phase_1_decision.json", decision)
    return decision


def exclusion_sources() -> dict[str, set[str]]:
    structure = REPO_ROOT / STRUCTURE_RELATIVE
    quarantine = REPO_ROOT / QUARANTINE_RELATIVE
    development = load_json(structure / "development/development_80_manifest.json")
    clean24 = load_json(structure / "phase_4/frozen_selection_24.json")
    quarantine24 = load_json(quarantine / "phase_4/frozen_selection_24.json")
    development_fps = {str(row["architecture_fingerprint"]) for row in development}
    sources = {
        "development_80": development_fps,
        "refit_training_tuning_calibration": set(development_fps),
        "clean_prospective_24": {str(row["architecture_fingerprint"]) for row in clean24},
        "quarantined_24": {str(row["architecture_fingerprint"]) for row in quarantine24},
    }
    if len(sources["development_80"]) != 80:
        raise RuntimeError("development source is not 80 unique architectures")
    if len(sources["clean_prospective_24"]) != 24 or len(sources["quarantined_24"]) != 24:
        raise RuntimeError("prospective/quarantine exclusion source is not 24 architectures")
    return sources


def _project_history_record(
    source: Mapping[str, Any], *, source_path: Path, source_index: int
) -> dict[str, Any]:
    required = (
        "search_seed", "step", "operations", "edges", "z_search", "z_arch",
        "architecture_fingerprint", "hp_fingerprint", "condition_mask_vector",
    )
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"missing history projection fields {missing}")
    fingerprint = architecture_fingerprint(source)
    if fingerprint != str(source["architecture_fingerprint"]):
        raise ValueError("stored history architecture fingerprint drift")
    z_search = [float(value) for value in source["z_search"]]
    z_arch = [float(value) for value in source["z_arch"]]
    if len(z_search) != 16 or len(z_arch) != 12:
        raise ValueError("history latent vector has unexpected length")
    realized_fields = (
        "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
        "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer", "normalization",
    )
    realized = {key: source.get(key) for key in realized_fields}
    row = {
        "replay_id": (
            f"seed{int(source['search_seed'])}:step{int(source['step'])}:"
            f"{fingerprint}:{source['hp_fingerprint']}"
        ),
        "config_id": (
            f"history_seed{int(source['search_seed'])}_step{int(source['step'])}_"
            f"{fingerprint[:12]}_{str(source['hp_fingerprint'])[:12]}"
        ),
        "search_seed": int(source["search_seed"]),
        "step": int(source["step"]),
        "source_index": int(source_index),
        "source_history_path": str(source_path.resolve()),
        "source_history_sha256": sha256_file(source_path),
        "operations": list(source["operations"]),
        "edges": [list(edge) for edge in source["edges"]],
        "z_search": z_search,
        "z_arch": z_arch,
        "raw_hp_vector": z_search[12:16],
        "realized_hp": realized,
        "architecture_fingerprint": fingerprint,
        "hp_fingerprint": str(source["hp_fingerprint"]),
        "condition_mask": source.get("condition_mask"),
        "condition_mask_vector": [float(value) for value in source["condition_mask_vector"]],
        "architecture_source": "historical_operations_and_edges_without_decode",
    }
    assert_projection_is_label_free([row])
    return row


def project_histories() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], set[str]]:
    sources = exclusion_sources()
    excluded = set().union(*sources.values())
    output: list[dict[str, Any]] = []
    audits: dict[int, dict[str, Any]] = {}
    for expected_seed, path in enumerate(history_paths()):
        source_rows = load_json(path)
        projected = [
            _project_history_record(row, source_path=path, source_index=index)
            for index, row in enumerate(source_rows)
        ]
        if {row["search_seed"] for row in projected} != {expected_seed}:
            raise RuntimeError(f"history seed identity mismatch: {path}")
        eligible = [row for row in projected if row["architecture_fingerprint"] not in excluded]
        first: dict[tuple[str, str], dict[str, Any]] = {}
        for row in sorted(eligible, key=lambda item: (int(item["step"]), int(item["source_index"]))):
            key = (str(row["architecture_fingerprint"]), str(row["hp_fingerprint"]))
            first.setdefault(key, row)
        retained = sorted(first.values(), key=lambda item: (int(item["step"]), int(item["source_index"])))
        output.extend(retained)
        audits[expected_seed] = {
            "search_seed": expected_seed,
            "raw_record_count": len(projected),
            "raw_architecture_count": len({row["architecture_fingerprint"] for row in projected}),
            "excluded_record_count": len(projected) - len(eligible),
            "after_exclusion_record_count": len(eligible),
            "duplicate_architecture_hp_records_removed": len(eligible) - len(retained),
            "retained_configuration_count": len(retained),
            "retained_architecture_count": len({row["architecture_fingerprint"] for row in retained}),
            "minimum_step": min(row["step"] for row in retained),
            "maximum_step": max(row["step"] for row in retained),
        }
    assert_projection_is_label_free(output)
    return output, audits, excluded


def build_exclusion_outputs(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sources = exclusion_sources()
    projection, trajectory_audits, excluded = project_histories()
    rows = [
        {
            "architecture_fingerprint": fingerprint,
            "sources": sorted(name for name, values in sources.items() if fingerprint in values),
        }
        for fingerprint in sorted(excluded)
    ]
    overlaps = []
    names = sorted(sources)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlaps.append({"left": left, "right": right, "overlap_count": len(sources[left] & sources[right])})
    audit = {
        "passed": all(item["retained_architecture_count"] >= 100 for item in trajectory_audits.values()),
        "exclusion_unit": "architecture_fingerprint",
        "source_counts": {name: len(values) for name, values in sources.items()},
        "union_count": len(excluded),
        "source_overlaps": overlaps,
        "trajectory_counts": list(trajectory_audits.values()),
        "all_hp_configurations_of_excluded_architectures_removed": True,
        "same_architecture_hp_configuration_keeps_first_step_only": True,
    }
    phase = output / "phase_2"
    dump_json(phase / "excluded_architecture_fingerprints.json", rows)
    dump_csv(phase / "excluded_architecture_fingerprints.csv", rows)
    dump_json(phase / "exclusion_audit.json", audit)
    dump_csv(phase / "history_exclusion_counts.csv", list(trajectory_audits.values()))
    return audit, projection


def audit_history_protocol(output: Path) -> dict[str, Any]:
    trajectory_rows = []
    for seed, path in enumerate(history_paths()):
        rows = load_json(path)
        key_union = sorted({str(key) for row in rows for key in row})
        test_keys = [key for key in key_union if "test" in key.lower()]
        dataset_contexts = {
            stable_json_hash(row.get("dataset_context")): row.get("dataset_context") for row in rows
        }
        trajectory_rows.append({
            "search_seed": seed,
            "record_count": len(rows),
            "step_min": min(int(row["step"]) for row in rows),
            "step_max": max(int(row["step"]) for row in rows),
            "unique_step_count": len({int(row["step"]) for row in rows}),
            "validation_field": "val_acc" if "val_acc" in key_union else None,
            "value_alias_present": "value" in key_union,
            "test_named_top_level_fields": test_keys,
            "hp_modes": sorted({str(row["hp_mode"]) for row in rows}),
            "normalizations": sorted({str(row["normalization"]) for row in rows}),
            "evaluation_fidelities": sorted({str(row["evaluation_fidelity"]) for row in rows}),
            "dataset_contexts": list(dataset_contexts.values()),
            "single_validation_seed_per_configuration": True,
            "evaluation_seed_scheme": sorted({str(row["candidate_evaluation_seed_scheme"]) for row in rows}),
            "eval_epochs": 150,
            "patience": 40,
        })
    protocol_identities = {
        stable_json_hash({
            "hp_modes": row["hp_modes"],
            "normalizations": row["normalizations"],
            "evaluation_fidelities": row["evaluation_fidelities"],
            "dataset_contexts": row["dataset_contexts"],
            "eval_epochs": row["eval_epochs"],
            "patience": row["patience"],
        })
        for row in trajectory_rows
    }
    compatible = (
        len(protocol_identities) == 1
        and all(not row["test_named_top_level_fields"] for row in trajectory_rows)
        and all(row["validation_field"] == "val_acc" for row in trajectory_rows)
        and all(row["unique_step_count"] == 500 for row in trajectory_rows)
    )
    audit = {
        "passed": compatible,
        "classification_if_failed": "REPLAY_LABEL_PROTOCOL_INCOMPATIBLE",
        "validation_label_field": "val_acc",
        "validation_label_source": "single seeded Cora validation evaluation recorded by historical Phase 4",
        "value_field_treated_as_performance_alias_and_never_used_for_projection": True,
        "within_trajectory_protocol_compatible": compatible,
        "cross_trajectory_absolute_values_pooled": False,
        "aggregation_rule": "within-search-seed ranking metrics and five-seed equal-weight macro average",
        "historical_protocol": {
            "dataset": "Cora",
            "hp_mode": "global4",
            "eval_epochs": 150,
            "patience": 40,
            "normalization": "none",
            "graph_transform": "NormalizeFeatures",
            "training_mode": "full_batch",
            "metric": "accuracy",
        },
        "historical_labels_are_single_seed_and_noisier_than_300_80_multiseed_labels": True,
        "pair_gap_thresholds": list(GAP_THRESHOLDS),
        "single_run_pairs_called_reliable_pairs": False,
        "trajectory_audits": trajectory_rows,
    }
    phase = output / "phase_3"
    dump_json(phase / "history_protocol_audit.json", audit)
    dump_csv(phase / "history_protocol_audit.csv", trajectory_rows)
    return audit


def freeze_available_predictions(
    output: Path, projection: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = encode_selected_mu(projection)
    refit, refit_identities = frozen_refit_predictions(encoded)
    originals = frozen_head_h_predictions(encoded, REPO_ROOT / STRICT_RELATIVE)
    original_lookup = {str(row["config_id"]): row for row in originals}
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(encoded):
        source = original_lookup[str(row["config_id"])]
        values = {
            "refit_head": float(refit[index]),
            "original_frozen_head": float(source["head_prediction"]),
            "original_frozen_h": float(source["h_prediction"]),
        }
        for model, value in values.items():
            predictions.append({
                "replay_id": row["replay_id"],
                "search_seed": row["search_seed"],
                "step": row["step"],
                "architecture_fingerprint": row["architecture_fingerprint"],
                "hp_fingerprint": row["hp_fingerprint"],
                "model": model,
                "prediction": value,
                "prediction_frozen_before_history_validation_connection": True,
            })
    candidate_count = len(encoded)
    coverage = {
        "refit_head": len({row["replay_id"] for row in predictions if row["model"] == "refit_head"}) / candidate_count,
        "original_frozen_head": len({row["replay_id"] for row in predictions if row["model"] == "original_frozen_head"}) / candidate_count,
        "original_frozen_h": len({row["replay_id"] for row in predictions if row["model"] == "original_frozen_h"}) / candidate_count,
        "original_zs_fixed_surrogate": 0.0,
    }
    zs_audit = {
        "required_model": "original_zs_fixed_surrogate",
        "serialized_generalizable_model_found": False,
        "coverage": 0.0,
        "minimum_required_coverage": 0.95,
        "available_strict_artifact": str((REPO_ROOT / STRICT_RELATIVE / "all_validation_predictions.csv").resolve()),
        "available_artifact_scope": "OOF predictions for prior labeled configurations only",
        "available_h_targeted_scope": "20 already evaluated targeted candidates only",
        "only_generalizing_implementation": "analyse.evaluate_performance_aware_vae_strict_regression._fixed_gp_predict",
        "only_implementation_calls_standard_scaler_fit": True,
        "only_implementation_calls_gaussian_process_regressor_fit": True,
        "only_implementation_kernel": "fixed RBF over z_arch + HP",
        "blocked_by_goal_rules": [
            "no predictor training, fine-tuning, or update",
            "no latent-space Euclidean-distance GP",
        ],
        "forbidden_posterior_reconstruction_executed": False,
    }
    phase = output / "phase_4"
    projection_path = phase / "unlabeled_candidate_projection.json"
    prediction_path = phase / "frozen_predictions.json"
    dump_json(projection_path, list(projection))
    dump_csv(phase / "unlabeled_candidate_projection.csv", list(projection))
    dump_json(prediction_path, predictions)
    dump_csv(phase / "frozen_predictions.csv", predictions)
    dump_json(phase / "original_zs_coverage_audit.json", zs_audit)
    dump_json(phase / "common_candidate_coverage.json", {
        "candidate_count": candidate_count,
        "minimum_required_coverage": 0.95,
        "coverage_by_model": coverage,
        "all_necessary_models_cover_at_least_95_percent": all(value >= 0.95 for value in coverage.values()),
    })
    write_sha256_sidecar(projection_path)
    write_sha256_sidecar(prediction_path)
    model_identity = {
        "refit_head_checkpoints": refit_identities,
        "original_head_h_checkpoints": getattr(frozen_head_h_predictions, "checkpoint_hashes", {}),
        "base_vae_sha256": sha256_file(REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"),
        "historical_mu_source": "frozen VAE encode of exact stored operations and edges",
        "historical_decode_executed": False,
        "parameter_training_or_update_executed": False,
        "vae_training_executed": False,
        "latent_distance_gp_used": False,
    }
    dump_json(phase / "frozen_model_identity.json", model_identity)
    return predictions, {"coverage": coverage, "zs_audit": zs_audit}


def rows_for_window(rows: Sequence[Mapping[str, Any]], window: str) -> list[dict[str, Any]]:
    if window not in PREFIX_SPECS:
        raise KeyError(window)
    mode, threshold = PREFIX_SPECS[window]
    if mode == "prefix":
        selected = [dict(row) for row in rows if int(row["step"]) < threshold]
    else:
        selected = [dict(row) for row in rows if int(row["step"]) >= threshold]
    return sorted(selected, key=lambda row: (int(row["step"]), str(row["architecture_fingerprint"]), str(row["hp_fingerprint"])))


def _descending_ids(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    return [
        str(row["replay_id"])
        for row in sorted(
            rows,
            key=lambda row: (
                -float(row[key]), int(row["step"]),
                str(row["architecture_fingerprint"]), str(row["hp_fingerprint"]),
            ),
        )
    ]


def pairwise_accuracy(
    rows: Sequence[Mapping[str, Any]], prediction_key: str, *, gap: float = 0.0
) -> tuple[float | None, int]:
    correct = 0
    count = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            label_delta = float(left["validation"]) - float(right["validation"])
            if abs(label_delta) < gap or label_delta == 0.0:
                continue
            prediction_delta = float(left[prediction_key]) - float(right[prediction_key])
            if prediction_delta == 0.0:
                correct += 0.5
            elif prediction_delta * label_delta > 0:
                correct += 1
            count += 1
    return (float(correct / count) if count else None), count


def ranking_metric_bundle(
    rows: Sequence[Mapping[str, Any]], prediction_key: str = "prediction"
) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("ranking metrics require at least two candidates")
    actual = np.asarray([float(row["validation"]) for row in rows], dtype=np.float64)
    predicted = np.asarray([float(row[prediction_key]) for row in rows], dtype=np.float64)
    tau = kendalltau(actual, predicted).statistic
    rho = spearmanr(actual, predicted).statistic
    ordinary, ordinary_count = pairwise_accuracy(rows, prediction_key)
    result: dict[str, Any] = {
        "candidate_count": len(rows),
        "architecture_count": len({row["architecture_fingerprint"] for row in rows}),
        "kendall": None if tau is None or not np.isfinite(tau) else float(tau),
        "spearman": None if rho is None or not np.isfinite(rho) else float(rho),
        "pairwise_accuracy": ordinary,
        "pair_count": ordinary_count,
    }
    for gap in GAP_THRESHOLDS:
        value, count = pairwise_accuracy(rows, prediction_key, gap=gap)
        suffix = f"gap_{gap:.3f}".replace(".", "_")
        result[f"pairwise_accuracy_{suffix}"] = value
        result[f"pair_count_{suffix}"] = count

    true_ids = _descending_ids(rows, "validation")
    predicted_ids = _descending_ids(rows, prediction_key)
    true_position = {item: index for index, item in enumerate(true_ids)}
    top_count = max(2, math.ceil(0.20 * len(rows)))
    top_ids = set(true_ids[:top_count])
    top_rows = [row for row in rows if str(row["replay_id"]) in top_ids]
    top_actual = np.asarray([float(row["validation"]) for row in top_rows], dtype=np.float64)
    top_predicted = np.asarray([float(row[prediction_key]) for row in top_rows], dtype=np.float64)
    top_tau = kendalltau(top_actual, top_predicted).statistic
    top_pair, top_pair_count = pairwise_accuracy(top_rows, prediction_key)
    result.update({
        "top_region_count": top_count,
        "top_region_kendall": None if top_tau is None or not np.isfinite(top_tau) else float(top_tau),
        "top_region_pairwise_accuracy": top_pair,
        "top_region_pair_count": top_pair_count,
    })
    relevance = actual.reshape(1, -1)
    scores = predicted.reshape(1, -1)
    for k in (5, 10):
        result[f"ndcg_at_{k}"] = float(ndcg_score(relevance, scores, k=min(k, len(rows))))
    best = float(actual.max())
    for k in TOP_K_VALUES:
        use_k = min(k, len(rows))
        predicted_top = set(predicted_ids[:use_k])
        true_top = set(true_ids[:use_k])
        overlap = len(predicted_top & true_top)
        result[f"recall_at_{k}"] = float(overlap / len(true_top))
        result[f"precision_at_{k}"] = float(overlap / len(predicted_top))
        selected_actual = [float(rows[index]["validation"]) for index in range(len(rows)) if str(rows[index]["replay_id"]) in predicted_top]
        result[f"regret_at_{k}"] = float(best - max(selected_actual))

    predicted_top10_count = max(1, math.ceil(0.10 * len(rows)))
    predicted_top20_count = max(1, math.ceil(0.20 * len(rows)))
    true_top10_count = max(1, math.ceil(0.10 * len(rows)))
    predicted_top10 = set(predicted_ids[:predicted_top10_count])
    predicted_top20 = set(predicted_ids[:predicted_top20_count])
    true_top10 = set(true_ids[:true_top10_count])
    median = float(np.median(actual))
    by_id = {str(row["replay_id"]): row for row in rows}
    false_positives = sum(float(by_id[item]["validation"]) < median for item in predicted_top10)
    false_negatives = len(true_top10 - predicted_top20)
    result.update({
        "predicted_top_10_percent_count": predicted_top10_count,
        "true_top_10_percent_count": true_top10_count,
        "false_positive_rate": float(false_positives / predicted_top10_count),
        "false_negative_rate": float(false_negatives / true_top10_count),
    })
    return result


def fixed_random_scores(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, float]:
    ids = [str(row["replay_id"]) for row in rows]
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return {item: float(len(shuffled) - index) for index, item in enumerate(shuffled)}


def _bootstrap_rows(
    rows: Sequence[Mapping[str, Any]], sampled_fingerprints: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(row)
    output: list[dict[str, Any]] = []
    draw_counts: Counter[str] = Counter()
    for fingerprint in sampled_fingerprints:
        copy_index = draw_counts[fingerprint]
        draw_counts[fingerprint] += 1
        for row in grouped[fingerprint]:
            output.append({
                **dict(row),
                "replay_id": f"{row['replay_id']}#bootstrap{copy_index}",
            })
    return output


def shared_cluster_bootstrap(
    rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *, candidate_key: str, baseline_key: str, metric: str,
    replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    fingerprints = {
        search_seed: sorted({str(row["architecture_fingerprint"]) for row in rows})
        for search_seed, rows in rows_by_seed.items()
    }
    candidate_values: list[float] = []
    baseline_values: list[float] = []
    differences: list[float] = []
    for _ in range(replicates):
        candidate_per_seed: list[float] = []
        baseline_per_seed: list[float] = []
        for search_seed in sorted(rows_by_seed):
            fps = fingerprints[search_seed]
            sampled = [fps[index] for index in rng.integers(0, len(fps), size=len(fps))]
            boot = _bootstrap_rows(rows_by_seed[search_seed], sampled)
            candidate = ranking_metric_bundle(boot, candidate_key).get(metric)
            baseline = ranking_metric_bundle(boot, baseline_key).get(metric)
            if candidate is None or baseline is None:
                break
            candidate_per_seed.append(float(candidate))
            baseline_per_seed.append(float(baseline))
        if len(candidate_per_seed) != len(rows_by_seed):
            continue
        candidate_macro = statistics.fmean(candidate_per_seed)
        baseline_macro = statistics.fmean(baseline_per_seed)
        candidate_values.append(candidate_macro)
        baseline_values.append(baseline_macro)
        differences.append(candidate_macro - baseline_macro)
    if not differences:
        return {
            "bootstrap_valid_replicates": 0,
            "candidate_ci95_low": None,
            "candidate_ci95_high": None,
            "baseline_ci95_low": None,
            "baseline_ci95_high": None,
            "difference_ci95_low": None,
            "difference_ci95_high": None,
        }
    return {
        "bootstrap_replicates": replicates,
        "bootstrap_valid_replicates": len(differences),
        "bootstrap_unit": "search-seed-stratified architecture_fingerprint cluster",
        "shared_candidate_indices": True,
        "candidate_ci95_low": float(np.quantile(candidate_values, 0.025)),
        "candidate_ci95_high": float(np.quantile(candidate_values, 0.975)),
        "baseline_ci95_low": float(np.quantile(baseline_values, 0.025)),
        "baseline_ci95_high": float(np.quantile(baseline_values, 0.975)),
        "difference_ci95_low": float(np.quantile(differences, 0.025)),
        "difference_ci95_high": float(np.quantile(differences, 0.975)),
        "difference_ci_width": float(np.quantile(differences, 0.975) - np.quantile(differences, 0.025)),
    }


def classify_replay(
    *, reproduction_failed: bool = False, insufficient_unseen: bool = False,
    protocol_incompatible: bool = False, insufficient_coverage: bool = False,
    input_or_implementation_error: bool = False, all_success_gates: bool = False,
    signal_only: bool = False, underpowered: bool = False,
) -> str:
    if input_or_implementation_error:
        return "INPUT_OR_IMPLEMENTATION_ERROR"
    if reproduction_failed:
        return "FROZEN_MODEL_REPRODUCTION_FAILED"
    if insufficient_unseen:
        return "INSUFFICIENT_UNSEEN_HISTORY"
    if protocol_incompatible:
        return "REPLAY_LABEL_PROTOCOL_INCOMPATIBLE"
    if insufficient_coverage:
        return "INSUFFICIENT_REPLAY_COVERAGE"
    if all_success_gates:
        return "DIRECT_HEAD_VALIDATED_FOR_RANKING_LOSS_PILOT"
    if underpowered:
        return "DIRECT_HEAD_REPLAY_UNDERPOWERED"
    if signal_only:
        return "DIRECT_HEAD_SIGNAL_ONLY"
    return "DIRECT_HEAD_REPLAY_REJECTED"


def write_terminal_outputs(
    output: Path, classification: str, *, evidence: Mapping[str, Any]
) -> None:
    payload = {
        "classification": classification,
        "evidence": dict(evidence),
        "diagnostic_only": True,
        "historical_validation_connected": False,
        "predictor_training_or_update_executed": False,
        "vae_training_executed": False,
        "latent_distance_gp_used": False,
        "gnn_training_executed": False,
        "offline_bo_executed": False,
        "formal_bo_executed": False,
        "test_label_used": False,
        "recommended_next_step": "provide_serialized_legal_zs_model_or_explicit_protocol_revision",
    }
    dump_json(output / "final_classification.json", payload)
    report = f"""# Direct-head chronological offline replay

Final classification: **`{classification}`**

Preparation reproduced Refit Head and Original Frozen Head/H on the clean 24-architecture set, built the architecture-level exclusion list, audited the five historical protocols, and froze all legally available history predictions before reading historical validation.

The replay stopped before historical label connection because the required Original Zs model has zero legal coverage. No serialized, generalizable frozen Zs model exists. The only available generalizing implementation refits a StandardScaler and an RBF Gaussian process over `z_arch + HP`, which violates the frozen-model and no-latent-distance-GP rules.

This is a fixed-history diagnostic, not BO. It did not train a predictor or VAE, run a GNN, use test accuracy, or run offline/formal BO. No historical validation metric is reported because the label firewall remained closed.
"""
    (output / "final_goal_report.md").write_text(report, encoding="utf-8")


def build_replay_plan(
    output: Path, *, input_identities: Mapping[str, Any], coverage: Mapping[str, float],
) -> dict[str, Any]:
    projection = output / "phase_4/unlabeled_candidate_projection.json"
    predictions = output / "phase_4/frozen_predictions.json"
    source_path = Path(__file__).resolve()
    test_path = REPO_ROOT / "tests/test_direct_head_chronological_offline_replay.py"
    plan = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "diagnostic_only": True,
        "historical_validation_connected_at_freeze": False,
        "predictor_training_allowed": False,
        "vae_training_allowed": False,
        "latent_distance_gp_allowed": False,
        "gnn_training_allowed": False,
        "offline_bo_allowed": False,
        "formal_bo_allowed": False,
        "test_labels_allowed": False,
        "input_identities": input_identities,
        "code_hashes": {
            str(source_path): sha256_file(source_path),
            **({str(test_path.resolve()): sha256_file(test_path)} if test_path.is_file() else {}),
        },
        "projection_path": str(projection.resolve()),
        "projection_sha256": sha256_file(projection),
        "frozen_predictions_path": str(predictions.resolve()),
        "frozen_predictions_sha256": sha256_file(predictions),
        "coverage_by_model": dict(coverage),
        "minimum_model_coverage": 0.95,
        "time_windows": {
            name: {"mode": mode, "raw_step_threshold": threshold}
            for name, (mode, threshold) in PREFIX_SPECS.items()
        },
        "duplicate_rule": "within search seed keep first step for architecture_fingerprint + hp_fingerprint",
        "point_metric_unit": "first-observed architecture+HP configuration",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "first_stratum": "five search_seed trajectories retained and equally weighted",
            "second_unit": "architecture_fingerprint cluster within time window",
            "shared_indices_for_models_and_differences": True,
        },
        "random_baseline": {
            "draws": RANDOM_BASELINE_DRAWS,
            "seed": RANDOM_BASELINE_SEED,
            "fixed_per_search_seed_and_time_window": True,
        },
        "metric_definitions": {
            "top_region": "ceil(20% of current raw-clock window) by historical validation",
            "top_k": list(TOP_K_VALUES),
            "gap_pairwise_thresholds": list(GAP_THRESHOLDS),
            "false_positive": "predicted top 10% with historical validation strictly below window median",
            "false_negative": "true top 10% absent from predicted top 20%",
            "regret": "best validation in window minus best validation among predicted top-k",
            "macro_aggregation": "equal mean over exactly five search_seed trajectories",
        },
        "classification_options": [
            "DIRECT_HEAD_VALIDATED_FOR_RANKING_LOSS_PILOT",
            "DIRECT_HEAD_SIGNAL_ONLY",
            "DIRECT_HEAD_REPLAY_REJECTED",
            "DIRECT_HEAD_REPLAY_UNDERPOWERED",
            "REPLAY_LABEL_PROTOCOL_INCOMPATIBLE",
            "INSUFFICIENT_UNSEEN_HISTORY",
            "INSUFFICIENT_REPLAY_COVERAGE",
            "FROZEN_MODEL_REPRODUCTION_FAILED",
            "INPUT_OR_IMPLEMENTATION_ERROR",
        ],
        "terminal_pre_label_classification": "INSUFFICIENT_REPLAY_COVERAGE",
    }
    path = output / "phase_4/replay_plan.json"
    dump_json(path, plan)
    write_sha256_sidecar(path)
    return plan


def prepare(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = workspace_snapshot()
    identities = collect_input_identities()
    dump_json(output / "workspace_before.json", snapshot)
    dump_json(output / "input_identities_before.json", identities)

    phase1 = reproduce_clean24_predictions(output)
    if not phase1["continue"]:
        classification = classify_replay(reproduction_failed=True)
        write_terminal_outputs(output, classification, evidence=phase1)
        return 2

    exclusion_audit, projection = build_exclusion_outputs(output)
    if not exclusion_audit["passed"]:
        classification = classify_replay(insufficient_unseen=True)
        write_terminal_outputs(output, classification, evidence=exclusion_audit)
        return 2

    protocol = audit_history_protocol(output)
    if not protocol["passed"]:
        classification = classify_replay(protocol_incompatible=True)
        write_terminal_outputs(output, classification, evidence=protocol)
        return 2

    predictions, prediction_audit = freeze_available_predictions(output, projection)
    coverage = prediction_audit["coverage"]
    plan = build_replay_plan(output, input_identities=identities, coverage=coverage)
    insufficient = any(float(coverage[model]) < 0.95 for model in PREDICTOR_MODELS)
    if insufficient:
        classification = classify_replay(insufficient_coverage=True)
        evidence = {
            "coverage_by_model": coverage,
            "minimum_required_coverage": 0.95,
            "candidate_count": len(projection),
            "available_prediction_row_count": len(predictions),
            "original_zs_audit": prediction_audit["zs_audit"],
            "replay_plan_sha256": sha256_file(output / "phase_4/replay_plan.json"),
        }
        write_terminal_outputs(output, classification, evidence=evidence)
        return 0
    raise RuntimeError("evaluate stage is not expected until all required frozen models are legally available")


def verify_frozen_plan(plan_path: Path) -> dict[str, Any]:
    plan = load_json(plan_path)
    sidecar = Path(str(plan_path) + ".sha256")
    expected_line = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if expected_line != sha256_file(plan_path):
        raise RuntimeError("replay plan sidecar mismatch")
    for key in ("projection", "frozen_predictions"):
        path = Path(plan[f"{key}_path"])
        if sha256_file(path) != plan[f"{key}_sha256"]:
            raise RuntimeError(f"frozen {key} drift")
    for path_text, expected in plan["code_hashes"].items():
        if sha256_file(Path(path_text)) != expected:
            raise RuntimeError(f"frozen code drift: {path_text}")
    return plan


def evaluate(plan_path: Path, output: Path) -> int:
    plan = verify_frozen_plan(plan_path)
    if plan.get("terminal_pre_label_classification"):
        raise RuntimeError(
            "historical validation firewall remains closed because preparation ended as "
            f"{plan['terminal_pre_label_classification']}"
        )
    raise RuntimeError("no label evaluation is authorized by the current frozen plan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE,
    )
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--plan", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return prepare(args.output.resolve())
    if args.command == "evaluate":
        return evaluate(args.plan.resolve(), args.output.resolve())
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
