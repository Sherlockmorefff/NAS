#!/usr/bin/env python3
"""Chronological offline replay for the frozen direct performance head (v2).

Preparation reuses the immutable v1 label-free projection and predictions,
verifies their provenance, and freezes the amended protocol plus all random
rankings.  Evaluation is a separate command and reads only ``val_acc`` after
the complete pre-label freeze has been verified.

This module performs inference/statistics only.  It never trains a predictor,
VAE or GNN, never reconstructs the unavailable Zs GP, and never runs BO.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, rankdata
from sklearn.metrics import ndcg_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import analyse.direct_head_chronological_offline_replay as v1
from analyse.h_targeted_reliable_pair_validation import sha256_file


FORMAT_VERSION = "direct_head_chronological_offline_replay_global4_v2"
V1_RELATIVE = "results/posthoc/direct-head-chronological-offline-replay-global4-v1"
OUTPUT_RELATIVE = "results/posthoc/direct-head-chronological-offline-replay-global4-v2"
WINDOWS = dict(v1.PREFIX_SPECS)
MANDATORY_PREDICTORS = ("refit_head", "original_frozen_head", "original_frozen_h")
MODELS = MANDATORY_PREDICTORS + ("fixed_random_ranking",)
OPTIONAL_MODELS = ("original_zs_fixed_surrogate",)
TOP_K_VALUES = (1, 3, 5, 10)
GAP_THRESHOLDS = (0.006, 0.012, 0.020)
RANDOM_DRAWS = 1000
RANDOM_SEED = 20260826
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260825
MINIMUM_COVERAGE = 0.95
FORBIDDEN_PROJECTION_PARTS = (
    "val", "validation", "test", "value", "reward", "rank", "performance",
)
CLASSIFICATIONS = (
    "INPUT_OR_IMPLEMENTATION_ERROR",
    "FROZEN_MODEL_REPRODUCTION_FAILED",
    "INSUFFICIENT_UNSEEN_HISTORY",
    "REPLAY_LABEL_PROTOCOL_INCOMPATIBLE",
    "INSUFFICIENT_REPLAY_COVERAGE",
    "DIRECT_HEAD_VALIDATED_FOR_RANKING_LOSS_PILOT",
    "DIRECT_HEAD_REPLAY_REJECTED",
    "DIRECT_HEAD_SIGNAL_ONLY",
    "DIRECT_HEAD_REPLAY_UNDERPOWERED",
)
BOOTSTRAP_METRICS = (
    "kendall_tau_b",
    "top_region_kendall_tau_b",
    "top_1_overlap_rate", "top_1_any_hit_rate", "regret_at_1",
    "top_3_overlap_rate", "top_3_any_hit_rate", "regret_at_3",
    "top_5_overlap_rate", "top_5_any_hit_rate", "regret_at_5",
    "top_10_overlap_rate", "top_10_any_hit_rate", "regret_at_10",
    "ndcg_at_10", "false_positive_rate", "false_negative_rate",
    "gap_filtered_pairwise_accuracy_0_006",
    "gap_filtered_pairwise_accuracy_0_012",
    "gap_filtered_pairwise_accuracy_0_020",
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
        raise RuntimeError(f"missing SHA-256 sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError(f"SHA-256 sidecar mismatch: {path}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(stable_hash(list(parts))[:16], 16) % (2**32)


def _iter_identities(value: Any):
    if isinstance(value, Mapping):
        if {"path", "sha256"} <= set(value):
            yield value
        else:
            for nested in value.values():
                yield from _iter_identities(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_identities(nested)


def verify_v1(v1_root: Path) -> dict[str, Any]:
    """Verify the entire immutable v1 evidence chain without rebuilding it."""
    differences: list[str] = []
    manifest = v1_root / "artifact_manifest.sha256"
    if not manifest.is_file():
        differences.append("missing artifact_manifest.sha256")
    else:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(None, 1)
            target = v1_root / relative.strip().removeprefix("./")
            if not target.is_file():
                differences.append(f"missing v1 artifact: {relative}")
            elif sha256_file(target) != expected:
                differences.append(f"v1 artifact hash drift: {relative}")

    phase4 = v1_root / "phase_4"
    for name in (
        "unlabeled_candidate_projection.json", "frozen_predictions.json", "replay_plan.json",
    ):
        try:
            verify_sidecar(phase4 / name)
        except Exception as error:  # reported as an input error, never rebuilt
            differences.append(str(error))

    projection = load_json(phase4 / "unlabeled_candidate_projection.json")
    predictions = load_json(phase4 / "frozen_predictions.json")
    plan = load_json(phase4 / "replay_plan.json")
    final = load_json(v1_root / "final_classification.json")
    integrity = load_json(v1_root / "result_integrity_audit.json")
    exclusion = load_json(v1_root / "phase_2/exclusion_audit.json")
    phase1 = load_json(v1_root / "phase_1/phase_1_decision.json")

    label_hits = v1.recursively_named(projection, FORBIDDEN_PROJECTION_PARTS)
    if label_hits:
        differences.append(f"forbidden projection fields: {label_hits[:5]}")
    expected_counts = {
        "trajectory_count": 5,
        "excluded_architecture_count": 128,
        "candidate_count": 1630,
        "architecture_count": 1075,
    }
    observed_counts = {
        "trajectory_count": len({int(row["search_seed"]) for row in projection}),
        "excluded_architecture_count": int(exclusion["union_count"]),
        "candidate_count": len(projection),
        "architecture_count": len({row["architecture_fingerprint"] for row in projection}),
    }
    for key, expected in expected_counts.items():
        if observed_counts[key] != expected:
            differences.append(f"{key}: expected {expected}, got {observed_counts[key]}")
    coverage = {}
    replay_ids = {str(row["replay_id"]) for row in projection}
    for model in MANDATORY_PREDICTORS:
        model_ids = {str(row["replay_id"]) for row in predictions if row["model"] == model}
        coverage[model] = len(model_ids & replay_ids) / len(replay_ids)
        if model_ids != replay_ids:
            differences.append(f"prediction coverage/key mismatch: {model}")
    prediction_keys = [(row["replay_id"], row["model"]) for row in predictions]
    if len(prediction_keys) != 4890 or len(prediction_keys) != len(set(prediction_keys)):
        differences.append("v1 frozen prediction row/key count mismatch")
    if final.get("historical_validation_connected") is not False:
        differences.append("v1 final classification says historical labels were connected")
    if integrity.get("history_validation_connected") is not False:
        differences.append("v1 integrity audit says historical labels were connected")
    if phase1.get("classification") != "PHASE_1_PASSED":
        differences.append("v1 frozen-model reproduction did not pass")
    try:
        v1.verify_frozen_plan(phase4 / "replay_plan.json")
    except Exception as error:
        differences.append(f"v1 frozen plan verification failed: {error}")
    for identity in _iter_identities(plan.get("input_identities", {})):
        path = Path(identity["path"])
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            differences.append(f"v1 input identity drift: {path}")
    for path_text, expected in plan.get("code_hashes", {}).items():
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != expected:
            differences.append(f"v1 code identity drift: {path}")
    return {
        "passed": not differences,
        "classification_if_failed": "INPUT_OR_IMPLEMENTATION_ERROR",
        "differences": differences,
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "coverage_by_mandatory_predictor": coverage,
        "projection_label_free": not label_hits,
        "historical_validation_connected": False,
        "v1_classification_preserved": final.get("classification"),
        "v1_root": str(v1_root.resolve()),
        "v1_artifact_manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        "projection_sha256": sha256_file(phase4 / "unlabeled_candidate_projection.json"),
        "prediction_sha256": sha256_file(phase4 / "frozen_predictions.json"),
        "replay_plan_sha256": sha256_file(phase4 / "replay_plan.json"),
    }


def rows_for_window(rows: Sequence[Mapping[str, Any]], window: str) -> list[dict[str, Any]]:
    return v1.rows_for_window(rows, window)


def freeze_random_rankings(
    projection: Sequence[Mapping[str, Any]], output: Path,
) -> dict[str, Any]:
    """Freeze complete label-free random permutations for every seed/window."""
    arrays: dict[str, np.ndarray] = {}
    index_payload: dict[str, Any] = {}
    manifest_cells: list[dict[str, Any]] = []
    for search_seed in range(5):
        seed_rows = [row for row in projection if int(row["search_seed"]) == search_seed]
        for window in WINDOWS:
            selected = rows_for_window(seed_rows, window)
            candidate_ids = [str(row["replay_id"]) for row in selected]
            cell = f"seed_{search_seed}__{window}"
            cell_seed = stable_seed(RANDOM_SEED, search_seed, window, candidate_ids)
            rng = np.random.default_rng(cell_seed)
            matrix = np.stack([rng.permutation(len(candidate_ids)) for _ in range(RANDOM_DRAWS)])
            matrix = matrix.astype(np.uint16, copy=False)
            arrays[cell] = matrix
            index_payload[cell] = {
                "search_seed": search_seed,
                "window": window,
                "candidate_ids": candidate_ids,
            }
            manifest_cells.append({
                "array_key": cell,
                "search_seed": search_seed,
                "window": window,
                "draw_count": RANDOM_DRAWS,
                "candidate_count": len(candidate_ids),
                "cell_seed": cell_seed,
                "candidate_ids_sha256": stable_hash(candidate_ids),
                "array_bytes_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
            })
    random_dir = output / "pre_label_freeze/random_rankings"
    random_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = random_dir / "frozen_random_rankings.npz"
    temporary = random_dir / "frozen_random_rankings.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(arrays_path)
    index_path = random_dir / "candidate_index.json"
    dump_json(index_path, index_payload)
    manifest_path = random_dir / "random_ranking_manifest.json"
    manifest = {
        "format_version": FORMAT_VERSION,
        "label_values_used": False,
        "random_seed": RANDOM_SEED,
        "draws_per_search_seed_window": RANDOM_DRAWS,
        "cell_count": len(manifest_cells),
        "cells": manifest_cells,
        "arrays_sha256": sha256_file(arrays_path),
        "candidate_index_sha256": sha256_file(index_path),
    }
    dump_json(manifest_path, manifest)
    for path in (arrays_path, index_path, manifest_path):
        write_sidecar(path)
    return {
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": sha256_file(arrays_path),
        "index_path": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "cell_count": len(manifest_cells),
        "draw_count": RANDOM_DRAWS,
        "label_values_used": False,
    }


def protocol_amendment() -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "amends_without_modifying": "direct_head_chronological_offline_replay_global4_v1",
        "v1_classification_remains": "INSUFFICIENT_REPLAY_COVERAGE",
        "mandatory_models": list(MODELS),
        "optional_models": {
            "original_zs_fixed_surrogate": {
                "status": "NOT_AVAILABLE_LEGALLY",
                "included_in_common_coverage": False,
                "triggers_insufficient_coverage": False,
                "reason": "no serialized generalizable frozen model exists",
            }
        },
        "forbidden_zs_actions": [
            "StandardScaler refit", "RBF-GP refit or reconstruction",
            "latent-space Euclidean-distance GP", "imputation from labels or other models",
        ],
        "minimum_mandatory_coverage": MINIMUM_COVERAGE,
        "historical_validation_connected_at_amendment_freeze": False,
        "diagnostic_only": True,
        "predictor_training_allowed": False,
        "vae_or_gnn_training_allowed": False,
        "offline_or_formal_bo_allowed": False,
        "test_labels_allowed": False,
    }


def _amendment_markdown(amendment: Mapping[str, Any]) -> str:
    return f"""# Direct Head replay v2 protocol amendment

v1 remains immutable with classification **`{amendment['v1_classification_remains']}`**.

v2 requires Refit Head, Original Frozen Head, Original Frozen H, and complete
frozen random rankings. Original Zs is **`NOT_AVAILABLE_LEGALLY`**, is optional,
is excluded from common coverage, and cannot trigger a coverage failure.

No scaler or GP is reconstructed or refitted. No predictor, VAE, or GNN is
trained. No test label or BO path is allowed. Historical `val_acc` may be joined
only after this amendment, the replay plan, model identities, and all random
rankings have been written and hash-frozen.
"""


def _code_identities() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "tests/test_direct_head_chronological_offline_replay_v2.py",
        REPO_ROOT / "analyse/direct_head_chronological_offline_replay.py",
        REPO_ROOT / "architecture_diagnostics.py",
        REPO_ROOT / "analyse/h_targeted_reliable_pair_validation.py",
        REPO_ROOT / "analyse/evaluate_performance_aware_vae_strict_regression.py",
        REPO_ROOT / "analyse/structure_native_surrogate_pilot.py",
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths if path.is_file()}


def prepare(output: Path, v1_root: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audit = verify_v1(v1_root)
    dump_json(output / "v1_reuse_integrity_audit.json", audit)
    if not audit["passed"]:
        final = terminal_error("INPUT_OR_IMPLEMENTATION_ERROR", audit)
        dump_json(output / "final_classification.json", final)
        return 2

    projection_path = v1_root / "phase_4/unlabeled_candidate_projection.json"
    prediction_path = v1_root / "phase_4/frozen_predictions.json"
    projection = load_json(projection_path)
    predictions = load_json(prediction_path)
    amendment = protocol_amendment()
    amendment_path = output / "protocol_amendment_v2.json"
    dump_json(amendment_path, amendment)
    write_sidecar(amendment_path)
    amendment_md = output / "protocol_amendment_v2.md"
    amendment_md.write_text(_amendment_markdown(amendment), encoding="utf-8")
    write_sidecar(amendment_md)

    random_identity = freeze_random_rankings(projection, output)
    prediction_identity = {
        "candidate_count": len(projection),
        "architecture_count": len({row["architecture_fingerprint"] for row in projection}),
        "v1_projection_path": str(projection_path.resolve()),
        "v1_projection_sha256": sha256_file(projection_path),
        "v1_predictions_path": str(prediction_path.resolve()),
        "v1_predictions_sha256": sha256_file(prediction_path),
        "coverage_by_mandatory_predictor": audit["coverage_by_mandatory_predictor"],
        "fixed_random_ranking_coverage": 1.0,
        "all_mandatory_models_cover_at_least_95_percent": all(
            value >= MINIMUM_COVERAGE
            for value in audit["coverage_by_mandatory_predictor"].values()
        ),
        "original_zs_fixed_surrogate": "NOT_AVAILABLE_LEGALLY",
        "original_zs_in_common_coverage": False,
        "prediction_frozen_before_historical_label_connection": True,
        "predictor_or_vae_parameters_updated": False,
        "historical_decode_executed": False,
        "latent_distance_gp_used": False,
        "prediction_models": sorted({row["model"] for row in predictions}),
    }
    identity_path = output / "pre_label_freeze/frozen_model_prediction_identity.json"
    dump_json(identity_path, prediction_identity)
    write_sidecar(identity_path)

    plan = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "diagnostic_only": True,
        "historical_validation_connected_at_freeze": False,
        "v1_integrity_audit_sha256": sha256_file(output / "v1_reuse_integrity_audit.json"),
        "v1_projection_path": str(projection_path.resolve()),
        "v1_projection_sha256": sha256_file(projection_path),
        "v1_predictions_path": str(prediction_path.resolve()),
        "v1_predictions_sha256": sha256_file(prediction_path),
        "v1_replay_plan_path": str((v1_root / "phase_4/replay_plan.json").resolve()),
        "v1_replay_plan_sha256": sha256_file(v1_root / "phase_4/replay_plan.json"),
        "protocol_amendment_path": str(amendment_path.resolve()),
        "protocol_amendment_sha256": sha256_file(amendment_path),
        "model_prediction_identity_path": str(identity_path.resolve()),
        "model_prediction_identity_sha256": sha256_file(identity_path),
        "random_rankings": random_identity,
        "code_hashes": _code_identities(),
        "mandatory_models": list(MODELS),
        "optional_model_status": {"original_zs_fixed_surrogate": "NOT_AVAILABLE_LEGALLY"},
        "minimum_coverage": MINIMUM_COVERAGE,
        "time_windows": {
            name: {"mode": mode, "raw_step_threshold": threshold}
            for name, (mode, threshold) in WINDOWS.items()
        },
        "primary_window": "prefix_100",
        "direction_stability_windows": ["prefix_60", "prefix_80", "prefix_100"],
        "late_50_hard_gate": False,
        "ranking_ties": {
            "rank_method": "average",
            "kendall_variant": "tau-b",
            "exact_boundary_tie_break": [
                "average_rank", "first_step", "architecture_fingerprint",
                "hp_fingerprint", "replay_id",
            ],
        },
        "metrics": [
            "kendall_tau_b", "top_region_kendall_tau_b", "top_k_overlap",
            "top_k_any_hit", "ndcg_at_10", "regret_at_1_3_5_10",
            "false_positive_rate", "false_negative_rate",
            "gap_filtered_pairwise_accuracy_0.006_0.012_0.020",
        ],
        "pair_name": "gap-filtered pairwise accuracy",
        "aggregation": "compute within each search seed, then equal-weight mean over five seeds",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "cell_seed_derivation": "stable SHA-256 of bootstrap seed, search_seed, and window",
            "strata": "five search seeds retained and equally weighted",
            "cluster_unit": "architecture_fingerprint within search seed and window",
            "same_architecture_hp_configurations_sampled_together": True,
            "shared_indices_for_all_models_and_paired_differences": True,
            "metrics_for_every_time_window": list(BOOTSTRAP_METRICS),
        },
        "random_baseline": {
            "draws": RANDOM_DRAWS, "seed": RANDOM_SEED,
            "depends_only_on": ["candidate_id", "search_seed", "time_window"],
            "full_rankings_frozen": True,
        },
        "success_gates": [
            "all audits pass",
            "prefix_100 refit kendall CI lower > 0",
            "prefix_100 refit top-region kendall CI lower > 0",
            "prefix_100 refit-head kendall delta >= 0.05 and CI lower > 0",
            "at least four of five seed deltas positive",
            "prefix_60, prefix_80, prefix_100 deltas all positive",
            "prefix_100 refit NDCG@10 >= original head",
            "prefix_100 refit regret@10 <= original head",
        ],
        "classification_order": list(CLASSIFICATIONS),
        "classification_precedence_note": (
            "input/reproduction/history/protocol/coverage errors override all statistical "
            "classifications; among statistical classes use validated, rejected, signal-only, "
            "then underpowered"
        ),
        "training_allowed": False,
        "gp_or_scaler_reconstruction_allowed": False,
        "gnn_or_vae_training_allowed": False,
        "bo_allowed": False,
        "test_labels_allowed": False,
        "historical_label_field_allowed_after_freeze": "val_acc",
        "history_alias_value_allowed": False,
    }
    plan_path = output / "replay_plan_v2.json"
    dump_json(plan_path, plan)
    write_sidecar(plan_path)
    freeze = {
        "passed": True,
        "frozen_at": utc_now(),
        "historical_validation_connected": False,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "random_rankings_sha256": random_identity["arrays_sha256"],
        "candidate_count": len(projection),
        "prediction_row_count": len(predictions),
    }
    dump_json(output / "pre_label_freeze/pre_label_freeze_audit.json", freeze)
    return 0


def verify_v2_plan(plan_path: Path) -> dict[str, Any]:
    verify_sidecar(plan_path)
    plan = load_json(plan_path)
    checks = {
        plan["v1_projection_path"]: plan["v1_projection_sha256"],
        plan["v1_predictions_path"]: plan["v1_predictions_sha256"],
        plan["v1_replay_plan_path"]: plan["v1_replay_plan_sha256"],
        plan["protocol_amendment_path"]: plan["protocol_amendment_sha256"],
        plan["model_prediction_identity_path"]: plan["model_prediction_identity_sha256"],
        plan["random_rankings"]["arrays_path"]: plan["random_rankings"]["arrays_sha256"],
        plan["random_rankings"]["index_path"]: plan["random_rankings"]["index_sha256"],
        plan["random_rankings"]["manifest_path"]: plan["random_rankings"]["manifest_sha256"],
        **plan["code_hashes"],
    }
    mismatches = []
    for path_text, expected in checks.items():
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(path_text)
    if mismatches:
        raise RuntimeError(f"frozen v2 plan input/code drift: {mismatches}")
    return plan


def join_historical_validation(
    projection: Sequence[Mapping[str, Any]], *, history_loader=load_json,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join exactly ``val_acc`` by v1 source identity; never read ``value``."""
    by_path: dict[str, list[Mapping[str, Any]]] = {}
    joined: list[dict[str, Any]] = []
    failures: list[str] = []
    history_hashes: dict[str, str] = {}
    for projected in projection:
        path_text = str(projected["source_history_path"])
        if path_text not in by_path:
            path = Path(path_text)
            by_path[path_text] = history_loader(path)
            history_hashes[path_text] = sha256_file(path)
        source = by_path[path_text][int(projected["source_index"])]
        if any("test" in str(key).lower() for key in source):
            failures.append(f"test-named field at {path_text}:{projected['source_index']}")
            continue
        if "val_acc" not in source:
            failures.append(f"missing val_acc at {path_text}:{projected['source_index']}")
            continue
        key_matches = (
            int(source["search_seed"]) == int(projected["search_seed"])
            and int(source["step"]) == int(projected["step"])
            and str(source["architecture_fingerprint"]) == str(projected["architecture_fingerprint"])
            and str(source["hp_fingerprint"]) == str(projected["hp_fingerprint"])
        )
        if not key_matches:
            failures.append(f"source identity mismatch: {projected['replay_id']}")
            continue
        value = float(source["val_acc"])
        if not math.isfinite(value):
            failures.append(f"non-finite val_acc: {projected['replay_id']}")
            continue
        joined.append({
            "replay_id": projected["replay_id"],
            "search_seed": int(projected["search_seed"]),
            "step": int(projected["step"]),
            "architecture_fingerprint": projected["architecture_fingerprint"],
            "hp_fingerprint": projected["hp_fingerprint"],
            "val_acc": value,
        })
    audit = {
        "passed": not failures and len(joined) == len(projection),
        "joined_row_count": len(joined),
        "expected_row_count": len(projection),
        "label_field_read": "val_acc",
        "value_alias_read": False,
        "test_label_used": False,
        "source_history_hashes": history_hashes,
        "failures": failures,
        "historical_protocol": {
            "dataset": "Cora", "hp_mode": "global4", "epochs": 150,
            "patience": 40, "split": "public", "validation_runs_per_configuration": 1,
        },
        "noise_note": "single-run 150/40 labels are noisier than 300/80 multi-seed validation",
    }
    return joined, audit


def _stable_order(rows: Sequence[Mapping[str, Any]], key: str) -> list[int]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    average = rankdata(-values, method="average")
    return sorted(range(len(rows)), key=lambda index: (
        float(average[index]), int(rows[index]["step"]),
        str(rows[index]["architecture_fingerprint"]), str(rows[index]["hp_fingerprint"]),
        str(rows[index]["replay_id"]),
    ))


def _finite_kendall(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    value = kendalltau(actual, predicted, variant="b").statistic
    return float(value) if value is not None and np.isfinite(value) else None


def _gap_pair_accuracy(actual: np.ndarray, predicted: np.ndarray, gap: float) -> tuple[float | None, int]:
    left, right = np.triu_indices(len(actual), 1)
    label_delta = actual[left] - actual[right]
    use = np.abs(label_delta) >= gap
    use &= label_delta != 0.0
    if not np.any(use):
        return None, 0
    prediction_delta = predicted[left] - predicted[right]
    products = prediction_delta[use] * label_delta[use]
    score = np.where(products > 0, 1.0, np.where(prediction_delta[use] == 0, 0.5, 0.0))
    return float(score.mean()), int(len(score))


def ranking_metrics(rows: Sequence[Mapping[str, Any]], prediction_key: str = "prediction") -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("ranking metrics require at least two rows")
    actual = np.asarray([float(row["val_acc"]) for row in rows], dtype=np.float64)
    predicted = np.asarray([float(row[prediction_key]) for row in rows], dtype=np.float64)
    true_order = _stable_order(rows, "val_acc")
    pred_order = _stable_order(rows, prediction_key)
    top_region_count = max(2, math.ceil(0.20 * len(rows)))
    top_indices = true_order[:top_region_count]
    result: dict[str, Any] = {
        "candidate_count": len(rows),
        "architecture_count": len({row["architecture_fingerprint"] for row in rows}),
        "kendall_tau_b": _finite_kendall(actual, predicted),
        "top_region_count": top_region_count,
        "top_region_kendall_tau_b": _finite_kendall(actual[top_indices], predicted[top_indices]),
        "ndcg_at_10": float(ndcg_score(actual.reshape(1, -1), predicted.reshape(1, -1), k=min(10, len(rows)), ignore_ties=False)),
        "ranking_tie_method": "average",
    }
    best = float(actual.max())
    for k in TOP_K_VALUES:
        use_k = min(k, len(rows))
        true_top = set(true_order[:use_k])
        predicted_top = set(pred_order[:use_k])
        overlap = len(true_top & predicted_top)
        result[f"top_{k}_overlap_rate"] = float(overlap / use_k)
        result[f"top_{k}_any_hit"] = bool(overlap > 0)
        result[f"top_{k}_any_hit_rate"] = float(overlap > 0)
        result[f"regret_at_{k}"] = float(best - max(actual[list(predicted_top)]))
    top10_count = max(1, math.ceil(0.10 * len(rows)))
    top20_count = max(1, math.ceil(0.20 * len(rows)))
    predicted_top10 = set(pred_order[:top10_count])
    predicted_top20 = set(pred_order[:top20_count])
    true_top10 = set(true_order[:top10_count])
    median = float(np.median(actual))
    result["false_positive_rate"] = float(sum(actual[index] < median for index in predicted_top10) / top10_count)
    result["false_negative_rate"] = float(len(true_top10 - predicted_top20) / top10_count)
    for gap in GAP_THRESHOLDS:
        suffix = f"{gap:.3f}".replace(".", "_")
        value, count = _gap_pair_accuracy(actual, predicted, gap)
        result[f"gap_filtered_pairwise_accuracy_{suffix}"] = value
        result[f"gap_filtered_pair_count_{suffix}"] = count
    return result


def _macro(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and not isinstance(row.get(key), bool)]
        result[key] = statistics.fmean(values) if len(values) == 5 else None
    return result


def _cluster_sample(rows: Sequence[Mapping[str, Any]], sampled_fps: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(row)
    counts: Counter[str] = Counter()
    sampled_rows: list[dict[str, Any]] = []
    for fingerprint in sampled_fps:
        copy_index = counts[fingerprint]
        counts[fingerprint] += 1
        for row in grouped[fingerprint]:
            sampled_rows.append({**row, "replay_id": f"{row['replay_id']}#b{copy_index}"})
    return sampled_rows


def _quantiles(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _bootstrap_cell(
    task: tuple[int, str, list[dict[str, Any]], list[dict[str, float]], int, int],
) -> tuple[int, str, dict[str, dict[str, list[float]]]]:
    search_seed, window, base, cell_random_scores, replicates, cell_seed = task
    rng = np.random.default_rng(cell_seed)
    fingerprints = sorted({str(row["architecture_fingerprint"]) for row in base})
    output: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in BOOTSTRAP_METRICS} for model in MODELS
    }
    for replicate in range(replicates):
        sampled = [fingerprints[index] for index in rng.integers(0, len(fingerprints), len(fingerprints))]
        boot = _cluster_sample(base, sampled)
        for model in MODELS:
            if model == "fixed_random_ranking":
                score_lookup = cell_random_scores[replicate % RANDOM_DRAWS]
                model_rows = [{
                    **row,
                    "prediction": score_lookup[row["replay_id"].split("#b", 1)[0]],
                } for row in boot]
            else:
                model_rows = [{**row, "prediction": row[model]} for row in boot]
            bundle = ranking_metrics(model_rows)
            for metric in BOOTSTRAP_METRICS:
                value = bundle.get(metric)
                output[model][metric].append(float(value) if value is not None else float("nan"))
    return search_seed, window, output


def shared_bootstrap(
    rows: Sequence[Mapping[str, Any]], random_scores: Mapping[tuple[int, str, int], Mapping[str, float]],
    *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shared five-stratum architecture-cluster bootstrap.

    Every requested ranking metric is bootstrapped for every window.  Metrics
    with no valid value in a resample are omitted from that replicate and the
    resulting effective count is reported explicitly.
    """
    model_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    difference_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    by_cell = {
        (search_seed, window): [
            row for row in rows if int(row["search_seed"]) == search_seed and row["window"] == window
        ]
        for search_seed in range(5) for window in WINDOWS
    }
    tasks = []
    for search_seed in range(5):
        for window in WINDOWS:
            tasks.append((
                search_seed,
                window,
                by_cell[(search_seed, window)],
                [random_scores[(search_seed, window, draw)] for draw in range(RANDOM_DRAWS)],
                replicates,
                stable_seed(seed, search_seed, window),
            ))
    workers = min(len(tasks), 35)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        cell_results = list(executor.map(_bootstrap_cell, tasks))
    by_result = {(search_seed, window): values for search_seed, window, values in cell_results}
    macro_arrays: dict[tuple[str, str, str], np.ndarray] = {}
    for window in WINDOWS:
        for model in MODELS:
            for metric in BOOTSTRAP_METRICS:
                per_seed_arrays = [
                    np.asarray(by_result[(search_seed, window)][model][metric], dtype=np.float64)
                    for search_seed in range(5)
                ]
                stacked = np.stack(per_seed_arrays, axis=0)
                valid = np.all(np.isfinite(stacked), axis=0)
                macro = np.full(replicates, np.nan, dtype=np.float64)
                macro[valid] = stacked[:, valid].mean(axis=0)
                macro_arrays[(model, window, metric)] = macro
                model_values[(model, window, metric)] = list(macro[np.isfinite(macro)])
        for metric in BOOTSTRAP_METRICS:
            candidate = macro_arrays[("refit_head", window, metric)]
            for baseline in ("original_frozen_head", "original_frozen_h", "fixed_random_ranking"):
                baseline_values = macro_arrays[(baseline, window, metric)]
                common = np.isfinite(candidate) & np.isfinite(baseline_values)
                difference_values[("refit_head", baseline, window, metric)] = list(
                    candidate[common] - baseline_values[common]
                )
    intervals = []
    for (model, window, metric), values in sorted(model_values.items()):
        low, high = _quantiles(values)
        intervals.append({
            "model": model, "window": window, "metric": metric,
            "bootstrap_replicates": replicates, "valid_replicates": len(values),
            "ci95_low": low, "ci95_high": high,
            "bootstrap_seed": seed,
            "bootstrap_unit": "search-seed-stratified architecture_fingerprint cluster",
        })
    differences = []
    for (candidate, baseline, window, metric), values in sorted(difference_values.items()):
        low, high = _quantiles(values)
        differences.append({
            "candidate": candidate, "baseline": baseline, "window": window, "metric": metric,
            "bootstrap_replicates": replicates, "valid_replicates": len(values),
            "difference_ci95_low": low, "difference_ci95_high": high,
            "shared_bootstrap_indices": True,
        })
    return intervals, differences


def classify(
    *, audits_pass: bool = True, protocol_compatible: bool = True,
    enough_unseen: bool = True, enough_coverage: bool = True,
    reproduction_pass: bool = True, input_error: bool = False,
    refit_kendall: float | None = None, refit_kendall_low: float | None = None,
    refit_kendall_high: float | None = None, refit_top_low: float | None = None,
    delta: float | None = None, delta_low: float | None = None,
    delta_high: float | None = None, positive_seed_count: int = 0,
    stable_prefixes: bool = False, ndcg_not_lower: bool = False,
    regret_not_higher: bool = False,
) -> tuple[str, dict[str, bool]]:
    if input_error or not audits_pass:
        return "INPUT_OR_IMPLEMENTATION_ERROR", {}
    if not reproduction_pass:
        return "FROZEN_MODEL_REPRODUCTION_FAILED", {}
    if not enough_unseen:
        return "INSUFFICIENT_UNSEEN_HISTORY", {}
    if not protocol_compatible:
        return "REPLAY_LABEL_PROTOCOL_INCOMPATIBLE", {}
    if not enough_coverage:
        return "INSUFFICIENT_REPLAY_COVERAGE", {}
    gates = {
        "refit_kendall_ci_lower_above_zero": refit_kendall_low is not None and refit_kendall_low > 0,
        "refit_top_region_kendall_ci_lower_above_zero": refit_top_low is not None and refit_top_low > 0,
        "delta_at_least_0_05": delta is not None and delta >= 0.05,
        "delta_ci_lower_above_zero": delta_low is not None and delta_low > 0,
        "at_least_four_of_five_seed_deltas_positive": positive_seed_count >= 4,
        "prefix_60_80_100_deltas_positive": stable_prefixes,
        "ndcg_at_10_not_lower": ndcg_not_lower,
        "regret_at_10_not_higher": regret_not_higher,
    }
    if all(gates.values()):
        return "DIRECT_HEAD_VALIDATED_FOR_RANKING_LOSS_PILOT", gates
    rejected = (
        refit_kendall_high is not None and refit_kendall_high <= 0
    ) or (
        delta is not None and delta <= -0.05 and delta_high is not None and delta_high < 0
    )
    if rejected:
        return "DIRECT_HEAD_REPLAY_REJECTED", gates
    if (refit_kendall_low is not None and refit_kendall_low > 0) or (
        refit_top_low is not None and refit_top_low > 0
    ):
        return "DIRECT_HEAD_SIGNAL_ONLY", gates
    return "DIRECT_HEAD_REPLAY_UNDERPOWERED", gates


def terminal_error(classification: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "classification": classification, "evidence": dict(evidence),
        "historical_validation_connected": False, "diagnostic_only": True,
        "predictor_training_or_update_executed": False, "vae_training_executed": False,
        "gnn_training_executed": False, "latent_distance_gp_used": False,
        "offline_bo_executed": False, "formal_bo_executed": False,
        "test_label_used": False,
    }


def _load_random(plan: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    index = load_json(Path(plan["random_rankings"]["index_path"]))
    arrays = np.load(Path(plan["random_rankings"]["arrays_path"]), allow_pickle=False)
    return index, arrays


def _random_score_maps(index: Mapping[str, Any], arrays: Any) -> dict[tuple[int, str, int], dict[str, float]]:
    output = {}
    for cell, info in index.items():
        ids = info["candidate_ids"]
        for draw, permutation in enumerate(arrays[cell]):
            output[(int(info["search_seed"]), str(info["window"]), draw)] = {
                ids[int(candidate_index)]: float(len(ids) - rank)
                for rank, candidate_index in enumerate(permutation)
            }
    return output


def evaluate(plan_path: Path, output: Path, *, bootstrap_replicates: int = BOOTSTRAP_REPLICATES) -> int:
    plan = verify_v2_plan(plan_path)
    if output.resolve() != plan_path.parent.resolve():
        raise RuntimeError("evaluation output must be the frozen v2 plan directory")
    if (output / "historical_label_join_audit.json").exists():
        raise FileExistsError("historical labels were already joined for this frozen plan")
    projection = load_json(Path(plan["v1_projection_path"]))
    frozen_predictions = load_json(Path(plan["v1_predictions_path"]))
    labels, join_audit = join_historical_validation(projection)
    dump_json(output / "historical_label_join_audit.json", join_audit)
    dump_json(output / "historical_validation_labels.json", labels)
    dump_csv(output / "historical_validation_labels.csv", labels)
    if not join_audit["passed"]:
        final = terminal_error("REPLAY_LABEL_PROTOCOL_INCOMPATIBLE", join_audit)
        dump_json(output / "final_classification.json", final)
        return 2

    label_by_id = {row["replay_id"]: float(row["val_acc"]) for row in labels}
    pred_by_id: dict[str, dict[str, float]] = defaultdict(dict)
    for row in frozen_predictions:
        pred_by_id[str(row["replay_id"])][str(row["model"])] = float(row["prediction"])
    base_rows = [{
        **dict(row), "val_acc": label_by_id[str(row["replay_id"])],
        **pred_by_id[str(row["replay_id"])],
    } for row in projection]
    random_index, random_arrays = _load_random(plan)
    random_scores = _random_score_maps(random_index, random_arrays)

    metric_rows: list[dict[str, Any]] = []
    random_draw_rows: list[dict[str, Any]] = []
    metric_keys: set[str] = set()
    window_rows: list[dict[str, Any]] = []
    for search_seed in range(5):
        trajectory = [row for row in base_rows if int(row["search_seed"]) == search_seed]
        for window in WINDOWS:
            selected = rows_for_window(trajectory, window)
            window_rows.extend({**row, "window": window} for row in selected)
            for model in MANDATORY_PREDICTORS:
                bundle = ranking_metrics([{**row, "prediction": row[model]} for row in selected])
                metric_keys.update(key for key, value in bundle.items() if isinstance(value, (int, float)) and not isinstance(value, bool))
                metric_rows.append({"search_seed": search_seed, "window": window, "model": model, **bundle})
            random_bundles = []
            for draw in range(RANDOM_DRAWS):
                scores = random_scores[(search_seed, window, draw)]
                bundle = ranking_metrics([{**row, "prediction": scores[row["replay_id"]]} for row in selected])
                random_bundles.append(bundle)
                random_draw_rows.append({"search_seed": search_seed, "window": window, "draw": draw, **bundle})
            random_mean = {}
            for key in metric_keys:
                values = [float(bundle[key]) for bundle in random_bundles if bundle.get(key) is not None and not isinstance(bundle.get(key), bool)]
                if values:
                    random_mean[key] = statistics.fmean(values)
            random_mean.update({
                "candidate_count": len(selected),
                "architecture_count": len({row["architecture_fingerprint"] for row in selected}),
                "ranking_tie_method": "average",
                "random_draws": RANDOM_DRAWS,
            })
            metric_rows.append({"search_seed": search_seed, "window": window, "model": "fixed_random_ranking", **random_mean})

    numeric_keys = sorted({
        key for row in metric_rows for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and key not in {"search_seed", "random_draws"}
    })
    macro_rows = []
    for window in WINDOWS:
        for model in MODELS:
            selected = [row for row in metric_rows if row["window"] == window and row["model"] == model]
            macro_rows.append({"window": window, "model": model, **_macro(selected, numeric_keys)})
    dump_json(output / "metrics/per_seed_window_model_metrics.json", metric_rows)
    dump_csv(output / "metrics/per_seed_window_model_metrics.csv", metric_rows)
    dump_csv(output / "metrics/random_1000_draw_metrics.csv", random_draw_rows)
    dump_json(output / "metrics/five_seed_macro_metrics.json", macro_rows)
    dump_csv(output / "metrics/five_seed_macro_metrics.csv", macro_rows)

    random_summary: list[dict[str, Any]] = []
    for search_seed in range(5):
        for window in WINDOWS:
            selected = [
                row for row in random_draw_rows
                if row["search_seed"] == search_seed and row["window"] == window
            ]
            for metric in BOOTSTRAP_METRICS:
                values = [float(row[metric]) for row in selected if row.get(metric) is not None]
                low, high = _quantiles(values)
                random_summary.append({
                    "scope": "search_seed", "search_seed": search_seed, "window": window,
                    "metric": metric, "draw_count": len(values),
                    "mean": statistics.fmean(values) if values else None,
                    "random_range_95_low": low, "random_range_95_high": high,
                })
    for window in WINDOWS:
        window_rows_by_draw = {
            draw: [
                row for row in random_draw_rows
                if row["window"] == window and row["draw"] == draw
            ]
            for draw in range(RANDOM_DRAWS)
        }
        for metric in BOOTSTRAP_METRICS:
            macro_values = []
            for draw_rows in window_rows_by_draw.values():
                values = [float(row[metric]) for row in draw_rows if row.get(metric) is not None]
                if len(values) == 5:
                    macro_values.append(statistics.fmean(values))
            low, high = _quantiles(macro_values)
            random_summary.append({
                "scope": "five_seed_equal_weight_macro", "search_seed": None,
                "window": window, "metric": metric, "draw_count": len(macro_values),
                "mean": statistics.fmean(macro_values) if macro_values else None,
                "random_range_95_low": low, "random_range_95_high": high,
            })
    dump_json(output / "metrics/random_baseline_summary.json", random_summary)
    dump_csv(output / "metrics/random_baseline_summary.csv", random_summary)

    intervals, differences = shared_bootstrap(
        window_rows, random_scores, replicates=bootstrap_replicates, seed=BOOTSTRAP_SEED,
    )
    interval_lookup = {(row["model"], row["window"], row["metric"]): row for row in intervals}
    difference_lookup = {(row["baseline"], row["window"], row["metric"]): row for row in differences}
    macro_lookup = {(row["model"], row["window"]): row for row in macro_rows}
    for row in intervals:
        point = macro_lookup[(row["model"], row["window"])].get(row["metric"])
        row["point_estimate"] = point
    for row in differences:
        candidate = macro_lookup[("refit_head", row["window"])].get(row["metric"])
        baseline = macro_lookup[(row["baseline"], row["window"])].get(row["metric"])
        row["point_difference"] = None if candidate is None or baseline is None else candidate - baseline
    dump_json(output / "bootstrap/bootstrap_intervals.json", intervals)
    dump_csv(output / "bootstrap/bootstrap_intervals.csv", intervals)
    dump_json(output / "bootstrap/paired_model_differences.json", differences)
    dump_csv(output / "bootstrap/paired_model_differences.csv", differences)

    stability = []
    for window in ("prefix_20", "prefix_40", "prefix_60", "prefix_80", "prefix_100", "late_100", "late_50"):
        refit = macro_lookup[("refit_head", window)]["kendall_tau_b"]
        head = macro_lookup[("original_frozen_head", window)]["kendall_tau_b"]
        per_seed = []
        for search_seed in range(5):
            lookup = {(row["model"], row["search_seed"]): row for row in metric_rows if row["window"] == window}
            per_seed.append(lookup[("refit_head", search_seed)]["kendall_tau_b"] - lookup[("original_frozen_head", search_seed)]["kendall_tau_b"])
        stability.append({
            "window": window, "refit_kendall": refit, "original_head_kendall": head,
            "refit_minus_original_head": refit - head,
            "positive_seed_count": sum(value > 0 for value in per_seed),
            "seed_differences": per_seed,
        })
    dump_json(output / "metrics/time_prefix_stability.json", stability)
    dump_csv(output / "metrics/time_prefix_stability.csv", stability)

    topk_keys = [key for key in numeric_keys if key.startswith("top_") or key.startswith("regret_") or key in {"ndcg_at_10", "false_positive_rate", "false_negative_rate"}]
    gap_keys = [key for key in numeric_keys if key.startswith("gap_filtered_")]
    topk_rows = [{"search_seed": row["search_seed"], "window": row["window"], "model": row["model"], **{key: row.get(key) for key in topk_keys}} for row in metric_rows]
    gap_rows = [{"search_seed": row["search_seed"], "window": row["window"], "model": row["model"], **{key: row.get(key) for key in gap_keys}} for row in metric_rows]
    dump_csv(output / "metrics/topk_regret_false_positive_negative.csv", topk_rows)
    dump_csv(output / "metrics/gap_filtered_pairwise.csv", gap_rows)

    prefix = macro_lookup[("refit_head", "prefix_100")]
    head = macro_lookup[("original_frozen_head", "prefix_100")]
    refit_k = interval_lookup[("refit_head", "prefix_100", "kendall_tau_b")]
    refit_top = interval_lookup[("refit_head", "prefix_100", "top_region_kendall_tau_b")]
    delta_k = difference_lookup[("original_frozen_head", "prefix_100", "kendall_tau_b")]
    stability_lookup = {row["window"]: row for row in stability}
    classification, gates = classify(
        refit_kendall=prefix["kendall_tau_b"],
        refit_kendall_low=refit_k["ci95_low"], refit_kendall_high=refit_k["ci95_high"],
        refit_top_low=refit_top["ci95_low"],
        delta=delta_k["point_difference"], delta_low=delta_k["difference_ci95_low"],
        delta_high=delta_k["difference_ci95_high"],
        positive_seed_count=stability_lookup["prefix_100"]["positive_seed_count"],
        stable_prefixes=all(stability_lookup[name]["refit_minus_original_head"] > 0 for name in ("prefix_60", "prefix_80", "prefix_100")),
        ndcg_not_lower=prefix["ndcg_at_10"] >= head["ndcg_at_10"],
        regret_not_higher=prefix["regret_at_10"] <= head["regret_at_10"],
    )
    final = {
        "classification": classification,
        "success_gates": gates,
        "historical_validation_connected": True,
        "historical_label_field": "val_acc",
        "test_label_used": False,
        "refit_head_passed_preregistered_standard": classification == "DIRECT_HEAD_VALIDATED_FOR_RANKING_LOSS_PILOT",
        "ranking_loss_pilot_authorized": classification == "DIRECT_HEAD_VALIDATED_FOR_RANKING_LOSS_PILOT",
        "prefix_100": {
            "refit_kendall": prefix["kendall_tau_b"],
            "refit_kendall_ci95": [refit_k["ci95_low"], refit_k["ci95_high"]],
            "top_region_kendall": prefix["top_region_kendall_tau_b"],
            "top_region_kendall_ci95": [refit_top["ci95_low"], refit_top["ci95_high"]],
            "refit_minus_original_head_kendall": delta_k["point_difference"],
            "refit_minus_original_head_kendall_ci95": [delta_k["difference_ci95_low"], delta_k["difference_ci95_high"]],
            "positive_seed_count": stability_lookup["prefix_100"]["positive_seed_count"],
            "refit_ndcg_at_10": prefix["ndcg_at_10"],
            "original_head_ndcg_at_10": head["ndcg_at_10"],
            "refit_regret_at_10": prefix["regret_at_10"],
            "original_head_regret_at_10": head["regret_at_10"],
        },
        "original_zs_fixed_surrogate": "NOT_AVAILABLE_LEGALLY",
        "diagnostic_only": True,
        "predictor_training_or_update_executed": False,
        "vae_training_executed": False,
        "gnn_training_executed": False,
        "latent_distance_gp_used": False,
        "offline_bo_executed": False,
        "formal_bo_executed": False,
        "automatic_next_step": None,
    }
    dump_json(output / "final_classification.json", final)
    dump_json(output / "result_integrity_audit.json", {
        "passed": True, "frozen_plan_verified": True,
        "historical_validation_connected": True, "joined_label_count": len(labels),
        "mandatory_predictor_coverage": 1.0, "random_coverage": 1.0,
        "original_zs_status": "NOT_AVAILABLE_LEGALLY", "test_label_used": False,
        "predictor_training_or_update_executed": False, "vae_or_gnn_training_executed": False,
        "latent_distance_gp_used": False, "offline_or_formal_bo_executed": False,
    })
    _write_report(output, final)
    return 0


def _write_report(output: Path, final: Mapping[str, Any]) -> None:
    evidence = final["prefix_100"]
    authorized = "允许" if final["ranking_loss_pilot_authorized"] else "不允许"
    text = f"""# Direct Head chronological offline replay global4 v2

最终分类：**`{final['classification']}`**。

v2 已在协议、三套冻结预测和 1,000 组随机排序全部冻结后，连接五条历史轨迹的
`val_acc`；未读取 `value` 或测试指标。Refit Head 的 prefix-100 Kendall 为
{evidence['refit_kendall']:.6f}，相对 Original Frozen Head 的提升为
{evidence['refit_minus_original_head_kendall']:.6f}，95% 区间为
[{evidence['refit_minus_original_head_kendall_ci95'][0]:.6f},
{evidence['refit_minus_original_head_kendall_ci95'][1]:.6f}]。五条轨迹中有
{evidence['positive_seed_count']} 条方向为正。

top-region Kendall、NDCG@10 与 regret@10 的证据及各门槛见
`final_classification.json`。按预注册规则，当前**{authorized}**进入排序损失预测头
小试验；本任务本身没有训练任何模型。

历史标签是 150/40 协议下的单次验证，噪声高于 300/80 多种子标签。该诊断只评估
冻结预测器在固定历史候选上的排序，不能恢复缺失的 Zs 潜空间 GP，也不能证明模型
会生成新架构，更不能据此直接启动离线或正式 BO。
"""
    (output / "final_goal_report.md").write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE)
    prepare_parser.add_argument("--v1", type=Path, default=REPO_ROOT / V1_RELATIVE)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--plan", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--bootstrap_replicates", type=int, default=BOOTSTRAP_REPLICATES, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return prepare(args.output.resolve(), args.v1.resolve())
    if args.command == "evaluate":
        return evaluate(args.plan.resolve(), args.output.resolve(), bootstrap_replicates=args.bootstrap_replicates)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
