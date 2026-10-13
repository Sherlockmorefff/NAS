#!/usr/bin/env python3
"""Audit, analyse, classify, and package Original Head attribution v2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, norm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse.original_head_confirmatory_attribution_v2 import (
    BOOTSTRAP_REPLICATES,
    FINAL_CLASSIFICATIONS,
    FORMAT_VERSION,
    POOL_COUNT,
    TRAINING_SEEDS,
    V1_RELATIVE,
    dump_csv,
    dump_json,
    dump_text,
    file_identity,
    load_json,
    protected_paths,
    sha256_file,
    stable_hash,
    strict_test_key_hits,
    verify_freeze,
)
from analyse.run_original_head_confirmatory_attribution_v2_labels import (
    load_jsonl,
    result_key,
)


STRATEGY_INDEX = {
    "B": 0,
    "M0": 1,
    "M1": 2,
    "C0": 3,
    "C1": 4,
}
COMPARISONS = ("B_minus_C", "B_minus_M", "M_minus_C")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def config_key(row: Mapping[str, Any]) -> tuple[int, str, str, str]:
    return (
        int(row["pool_id"]),
        str(row["joint_config_fingerprint"]),
        str(row["architecture_fingerprint"]),
        str(row["hp_fingerprint"]),
    )


def validate_complete_seed_results(
    tasks: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    test_hits = strict_test_key_hits(results)
    task_keys = [result_key(row) for row in tasks]
    result_keys = [result_key(row) for row in results]
    expected, actual = set(task_keys), set(result_keys)
    duplicate_tasks = len(task_keys) - len(expected)
    duplicate_results = len(result_keys) - len(actual)
    by_config: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        by_config[config_key(row)].append(row)
    missing_seeds = []
    protocol_conflicts = []
    trajectory_errors = []
    gpu_split_errors = []
    for key, rows in sorted(by_config.items()):
        seeds = [int(row["seed"]) for row in rows]
        if set(seeds) != set(TRAINING_SEEDS) or len(seeds) != len(TRAINING_SEEDS):
            missing_seeds.append({"config_key": list(key), "seeds": sorted(seeds)})
        invariants = {
            (
                row.get("dataset_content_fingerprint"), row.get("split_fingerprint"),
                row.get("hp_mode"), row.get("normalization"), int(row.get("eval_epochs", -1)),
                int(row.get("patience", -1)), int(row.get("assigned_gpu", -1)),
                row.get("actual_gpu", {}).get("uuid"),
            )
            for row in rows
        }
        if len(invariants) != 1:
            protocol_conflicts.append(list(key))
        assigned = {int(row["assigned_gpu"]) for row in rows}
        physical = {int(row["actual_gpu"]["physical_index"]) for row in rows}
        if len(assigned) != 1 or assigned != physical:
            gpu_split_errors.append({"config_key": list(key), "assigned": sorted(assigned), "physical": sorted(physical)})
        for row in rows:
            trajectory = row.get("trajectory")
            if not isinstance(trajectory, list) or not trajectory:
                trajectory_errors.append({"task_key": row.get("task_key"), "reason": "missing trajectory"})
                continue
            if any(set(event) != {"epoch_1based", "loss", "validation_accuracy"} for event in trajectory):
                trajectory_errors.append({"task_key": row.get("task_key"), "reason": "trajectory schema"})
            if stable_hash(trajectory) != row.get("sanitized_trajectory_hash"):
                trajectory_errors.append({"task_key": row.get("task_key"), "reason": "trajectory hash"})
            epochs = [int(event["epoch_1based"]) for event in trajectory]
            if epochs != list(range(1, len(trajectory) + 1)):
                trajectory_errors.append({"task_key": row.get("task_key"), "reason": "epoch sequence"})
            if "epochs_ran" in row and int(row["epochs_ran"]) != len(trajectory):
                trajectory_errors.append({"task_key": row.get("task_key"), "reason": "epochs_ran mismatch"})
    statuses = Counter(str(row.get("terminal_status")) for row in results)
    contexts = {
        (
            row.get("dataset_content_fingerprint"), row.get("split_fingerprint"),
            row.get("hp_mode"), row.get("normalization"),
            int(row.get("eval_epochs", -1)), int(row.get("patience", -1)),
        )
        for row in results
    }
    audit = {
        "passed": (
            not test_hits and not duplicate_tasks and not duplicate_results
            and expected == actual and not missing_seeds and not protocol_conflicts
            and not trajectory_errors and not gpu_split_errors and len(contexts) == 1
            and statuses == Counter({"success": len(results)})
            and all(bool(row.get("valid")) for row in results)
        ),
        "expected_task_count": len(tasks),
        "actual_result_count": len(results),
        "configuration_count": len(by_config),
        "duplicate_task_key_count": duplicate_tasks,
        "duplicate_result_key_count": duplicate_results,
        "missing_keys": [list(key) for key in sorted(expected - actual)],
        "foreign_keys": [list(key) for key in sorted(actual - expected)],
        "missing_seed_configs": missing_seeds,
        "protocol_conflicts": protocol_conflicts,
        "gpu_split_errors": gpu_split_errors,
        "trajectory_errors": trajectory_errors,
        "terminal_status_counts": dict(statuses),
        "invalid_count": sum(not bool(row.get("valid")) for row in results),
        "retry_count": sum(int(row.get("retry_count", 0)) for row in results),
        "test_field_hits": test_hits,
        "protocol_contexts": [list(row) for row in sorted(contexts, key=str)],
    }
    return audit


def aggregate_configurations(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[config_key(row)].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["seed"]))
        values = [float(row["val_acc"]) for row in ordered]
        output.append({
            "pool_id": key[0],
            "joint_config_fingerprint": key[1],
            "architecture_fingerprint": key[2],
            "hp_fingerprint": key[3],
            "config_id": str(ordered[0]["config_id"]),
            "seeds": [int(row["seed"]) for row in ordered],
            "seed_validation_values": values,
            "mean_validation": float(np.mean(values)),
            "std_validation": float(np.std(values, ddof=1)),
            "best_epoch_by_seed": [int(row.get("best_epoch", -1)) for row in ordered],
            "stop_epoch_by_seed": [int(row.get("stopped_epoch", row.get("epochs_ran", -1))) for row in ordered],
            "trajectory_hashes": [str(row["sanitized_trajectory_hash"]) for row in ordered],
            "assigned_gpu": int(ordered[0]["assigned_gpu"]),
        })
    return output


def _selection_key(row: Mapping[str, Any]) -> str:
    strategy = str(row["strategy"])
    if strategy == "HEAD_GLOBAL_TOP4":
        return "B"
    batch = int(row["random_batch"])
    if strategy == "MATCHED_HP_RANDOM_ARCH":
        return f"M{batch}"
    if strategy == "BLIND_JOINT_RANDOM":
        return f"C{batch}"
    raise ValueError(f"unknown strategy: {strategy}")


def build_slot_seed_values(
    selections: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = {
        (int(row["pool_id"]), str(row["joint_config_fingerprint"]), int(row["seed"])): float(row["val_acc"])
        for row in results
    }
    values = np.full((POOL_COUNT, 5, 4, len(TRAINING_SEEDS)), np.nan, dtype=np.float64)
    slot_rows = []
    seen = set()
    for selection in selections:
        pool = int(selection["pool_id"])
        key = _selection_key(selection)
        strategy_index = STRATEGY_INDEX[key]
        slot = int(selection["budget_order"]) - 1
        cell = (pool, key, slot)
        if cell in seen:
            raise ValueError(f"duplicate frozen selection slot: {cell}")
        seen.add(cell)
        seed_values = [
            labels[(pool, str(selection["joint_config_fingerprint"]), seed)]
            for seed in TRAINING_SEEDS
        ]
        values[pool, strategy_index, slot, :] = seed_values
        slot_rows.append({
            "pool_id": pool,
            "batch_key": key,
            "budget_order": slot + 1,
            "joint_config_fingerprint": selection["joint_config_fingerprint"],
            "architecture_fingerprint": selection["architecture_fingerprint"],
            "hp_fingerprint": selection["hp_fingerprint"],
            "seed_validation_values": seed_values,
            "mean_validation": float(np.mean(seed_values)),
        })
    expected = POOL_COUNT * 5 * 4
    if len(seen) != expected or np.isnan(values).any():
        raise ValueError(f"selection tensor incomplete: cells={len(seen)}, expected={expected}")
    return values, slot_rows


def observed_pool_metrics(slot_seed_values: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config_means = slot_seed_values.mean(axis=3)
    best = config_means.max(axis=2)
    pool_rows = []
    budget_rows = []
    for pool in range(POOL_COUNT):
        b = float(best[pool, 0])
        m_batches = best[pool, 1:3]
        c_batches = best[pool, 3:5]
        m = float(np.mean(m_batches))
        c = float(np.mean(c_batches))
        pool_rows.append({
            "pool_id": pool,
            "B_best_of_4": b,
            "M0_best_of_4": float(m_batches[0]),
            "M1_best_of_4": float(m_batches[1]),
            "M_mean_batch_best_of_4": m,
            "C0_best_of_4": float(c_batches[0]),
            "C1_best_of_4": float(c_batches[1]),
            "C_mean_batch_best_of_4": c,
            "B_minus_C": b - c,
            "B_minus_M": b - m,
            "M_minus_C": m - c,
        })
        for budget in (1, 2, 4):
            prefix = config_means[pool, :, :budget]
            prefix_best = prefix.max(axis=1)
            budget_rows.extend([
                {"pool_id": pool, "strategy": "B", "budget": budget, "best_so_far": float(prefix_best[0])},
                {"pool_id": pool, "strategy": "M", "budget": budget, "best_so_far": float(np.mean(prefix_best[1:3]))},
                {"pool_id": pool, "strategy": "C", "budget": budget, "best_so_far": float(np.mean(prefix_best[3:5]))},
            ])
    return pool_rows, budget_rows


def _bootstrap_indices(replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pools = rng.integers(0, POOL_COUNT, size=(replicates, POOL_COUNT), dtype=np.int16)
    batches = rng.integers(0, 2, size=(replicates, POOL_COUNT, 2), dtype=np.uint8)
    seeds = rng.integers(0, len(TRAINING_SEEDS), size=(replicates, POOL_COUNT, 3), dtype=np.uint8)
    return pools.astype("<i2", copy=False), batches.astype("u1", copy=False), seeds.astype("u1", copy=False)


def hierarchical_bootstrap(
    slot_seed_values: np.ndarray, replicates: int, seed: int, chunk_size: int = 1000,
) -> tuple[np.ndarray, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if slot_seed_values.shape != (POOL_COUNT, 5, 4, len(TRAINING_SEEDS)):
        raise ValueError(f"unexpected slot tensor shape: {slot_seed_values.shape}")
    pool_indices, batch_indices, seed_indices = _bootstrap_indices(replicates, seed)
    samples = np.empty((replicates, 3), dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        width = stop - start
        accumulated = np.zeros((width, 3), dtype=np.float64)
        for draw_position in range(POOL_COUNT):
            sampled_pools = pool_indices[start:stop, draw_position].astype(np.intp)
            selected = slot_seed_values[sampled_pools]
            shared_seed_indices = seed_indices[start:stop, draw_position, :]
            gather = np.broadcast_to(
                shared_seed_indices[:, None, None, :], selected.shape,
            )
            resampled_config_means = np.take_along_axis(selected, gather, axis=3).mean(axis=3)
            best = resampled_config_means.max(axis=2)
            matched_batch_indices = batch_indices[start:stop, draw_position, :].astype(np.intp)
            m = np.take_along_axis(best[:, 1:3], matched_batch_indices, axis=1).mean(axis=1)
            c = np.take_along_axis(best[:, 3:5], matched_batch_indices, axis=1).mean(axis=1)
            b = best[:, 0]
            accumulated[:, 0] += b - c
            accumulated[:, 1] += b - m
            accumulated[:, 2] += m - c
        samples[start:stop] = accumulated / POOL_COUNT
    tag = b"original-head-attribution-v2-hierarchical-bootstrap-v1\x00"
    hashes = {
        "pool_indices_sha256": hashlib.sha256(pool_indices.tobytes(order="C")).hexdigest(),
        "matched_batch_indices_sha256": hashlib.sha256(batch_indices.tobytes(order="C")).hexdigest(),
        "shared_seed_indices_sha256": hashlib.sha256(seed_indices.tobytes(order="C")).hexdigest(),
        "combined_indices_sha256": hashlib.sha256(
            tag + pool_indices.tobytes(order="C") + batch_indices.tobytes(order="C")
            + seed_indices.tobytes(order="C")
        ).hexdigest(),
    }
    summary = {
        "replicates": replicates,
        "seed": seed,
        "prng": "numpy.random.default_rng/PCG64",
        "numpy_version": np.__version__,
        "pool_sample_shape": list(pool_indices.shape),
        "batch_sample_shape": list(batch_indices.shape),
        "seed_sample_shape": list(seed_indices.shape),
        "pool_dtype": str(pool_indices.dtype),
        "batch_dtype": str(batch_indices.dtype),
        "seed_dtype": str(seed_indices.dtype),
        "quantile_method": "numpy.quantile default linear",
        "same_outer_pool_indices_all_comparisons": True,
        "same_matched_batch_indices_for_M_and_C": True,
        "same_seed_indices_all_configs_and_strategies_within_pool_draw": True,
        "best_of_4_recomputed_after_seed_resampling": True,
        **hashes,
        "comparisons": {
            name: {
                "bootstrap_mean": float(np.mean(samples[:, index])),
                "ci95_lower": float(np.quantile(samples[:, index], 0.025)),
                "ci95_upper": float(np.quantile(samples[:, index], 0.975)),
                "ci97_5_lower": float(np.quantile(samples[:, index], 0.0125)),
                "ci97_5_upper": float(np.quantile(samples[:, index], 0.9875)),
            }
            for index, name in enumerate(COMPARISONS)
        },
    }
    return samples, summary, (pool_indices, batch_indices, seed_indices)


def pool_only_bootstrap(
    pool_rows: Sequence[Mapping[str, Any]], replicates: int, seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(
        [[float(row[name]) for name in COMPARISONS] for row in pool_rows], dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, POOL_COUNT, size=(replicates, POOL_COUNT), dtype=np.int16).astype("<i2", copy=False)
    samples = matrix[indices].mean(axis=1)
    summary = {
        "replicates": replicates,
        "seed": seed,
        "outer_unit": "pool only",
        "fixed_observed_three_seed_configuration_means": True,
        "fixed_arithmetic_mean_of_two_random_batches": True,
        "indices_sha256": hashlib.sha256(indices.tobytes(order="C")).hexdigest(),
        "comparisons": {
            name: {
                "bootstrap_mean": float(np.mean(samples[:, index])),
                "ci95_lower": float(np.quantile(samples[:, index], 0.025)),
                "ci95_upper": float(np.quantile(samples[:, index], 0.975)),
                "ci97_5_lower": float(np.quantile(samples[:, index], 0.0125)),
                "ci97_5_upper": float(np.quantile(samples[:, index], 0.9875)),
            }
            for index, name in enumerate(COMPARISONS)
        },
    }
    return samples, summary


def classify(
    pool_rows: Sequence[Mapping[str, Any]], hierarchical: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    arrays = {
        name: np.asarray([float(row[name]) for row in pool_rows], dtype=np.float64)
        for name in COMPARISONS
    }
    means = {name: float(np.mean(values)) for name, values in arrays.items()}
    wins = {name: int(np.sum(values > 0)) for name, values in arrays.items()}
    losses = {name: int(np.sum(values < 0)) for name, values in arrays.items()}
    ties = {name: int(np.sum(values == 0)) for name, values in arrays.items()}
    ci = hierarchical["comparisons"]
    joint = (
        means["B_minus_C"] >= 0.006
        and float(ci["B_minus_C"]["ci95_lower"]) > 0.0
        and wins["B_minus_C"] >= 15
    )
    architecture = (
        joint and means["B_minus_M"] >= 0.006
        and float(ci["B_minus_M"]["ci97_5_lower"]) > 0.0
        and wins["B_minus_M"] >= 15
    )
    hp = (
        joint and means["M_minus_C"] >= 0.006
        and float(ci["M_minus_C"]["ci97_5_lower"]) > 0.0
        and wins["M_minus_C"] >= 15
    )
    if joint and architecture:
        classification = "JOINT_AND_ARCH_CONFIRMED"
    elif joint and hp:
        classification = "HP_FILTERING_ONLY"
    elif joint:
        classification = "JOINT_CONFIRMED_ATTRIBUTION_INCONCLUSIVE"
    elif means["B_minus_C"] <= 0.0 and wins["B_minus_C"] <= 12:
        classification = "NO_HEAD_ADVANTAGE"
    else:
        classification = "CONFIRMATORY_INCONCLUSIVE"
    evidence = {
        "means": means,
        "strict_wins": wins,
        "strict_losses": losses,
        "ties": ties,
        "joint_gate_passed": joint,
        "architecture_component_gate_passed": architecture,
        "hp_component_gate_passed": hp,
        "architecture_ci95_noninferiority_lower_above_minus_0_006": (
            float(ci["B_minus_M"]["ci95_lower"]) > -0.006
        ),
        "architecture_may_be_harmful_if_hp_only": (
            hp and not architecture and float(ci["B_minus_M"]["ci95_lower"]) <= -0.006
        ),
        "joint_gate_definition": "mean>=0.006, hierarchical CI95 lower>0, strict wins>=15/24",
        "component_gate_definition": (
            "joint gate passed, mean>=0.006, hierarchical Bonferroni CI97.5 lower>0, "
            "strict wins>=15/24"
        ),
    }
    return classification, evidence


def leave_one_pool_out(pool_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for removed in range(POOL_COUNT):
        retained = [row for row in pool_rows if int(row["pool_id"]) != removed]
        rows.append({
            "removed_pool": removed,
            **{name: float(np.mean([float(row[name]) for row in retained])) for name in COMPARISONS},
        })
    return rows


def observed_union_regret(
    selections: Sequence[Mapping[str, Any]], config_rows: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    means = {
        (int(row["pool_id"]), str(row["joint_config_fingerprint"])): float(row["mean_validation"])
        for row in config_rows
    }
    union: dict[int, set[str]] = defaultdict(set)
    for row in selections:
        union[int(row["pool_id"])].add(str(row["joint_config_fingerprint"]))
    rows = []
    by_pool = {int(row["pool_id"]): row for row in pool_rows}
    for pool in range(POOL_COUNT):
        union_best = max(means[(pool, fingerprint)] for fingerprint in union[pool])
        values = {
            "B": float(by_pool[pool]["B_best_of_4"]),
            "M": float(by_pool[pool]["M_mean_batch_best_of_4"]),
            "C": float(by_pool[pool]["C_mean_batch_best_of_4"]),
        }
        for strategy, value in values.items():
            rows.append({
                "pool_id": pool,
                "strategy": strategy,
                "observed_union_best": union_best,
                "strategy_value": value,
                "observed_union_regret": union_best - value,
                "not_full_256_candidate_pool_regret": True,
            })
    return rows


def head_descriptive_metrics(
    prediction_rows: Sequence[Mapping[str, Any]], config_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prediction = {
        (int(row["pool_id"]), str(row["joint_config_fingerprint"])): float(row["head_prediction_mean"])
        for row in prediction_rows
    }
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in config_rows:
        grouped[int(row["pool_id"])].append(row)
    output = []
    for pool, rows in sorted(grouped.items()):
        observed = np.asarray([float(row["mean_validation"]) for row in rows])
        predicted = np.asarray([prediction[(pool, str(row["joint_config_fingerprint"]))] for row in rows])
        tau = kendalltau(predicted, observed, variant="b")
        predicted_order = np.argsort(-predicted, kind="stable")
        output.append({
            "pool_id": pool,
            "observed_union_configuration_count": len(rows),
            "kendall_tau_b": None if not math.isfinite(float(tau.statistic)) else float(tau.statistic),
            "kendall_pvalue_descriptive": None if not math.isfinite(float(tau.pvalue)) else float(tau.pvalue),
            "mae": float(np.mean(np.abs(predicted - observed))),
            "predicted_top4_observed_mean": float(np.mean(observed[predicted_order[:4]])),
            "predicted_top1_observed": float(observed[predicted_order[0]]),
            "observed_union_best": float(np.max(observed)),
            "classification_gate_input": False,
        })
    return output


def power_and_strata(
    pool_rows: Sequence[Mapping[str, Any]], selections: Sequence[Mapping[str, Any]],
    generation_audits: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generation = {int(row["pool_id"]): row for row in generation_audits}
    b_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selections:
        if row["strategy"] == "HEAD_GLOBAL_TOP4":
            b_rows[int(row["pool_id"])].append(row)
    strata = []
    for row in pool_rows:
        pool = int(row["pool_id"])
        repeat_pattern = sorted(
            Counter(str(item["architecture_fingerprint"]) for item in b_rows[pool]).values(),
            reverse=True,
        )
        strata.append({
            **dict(row),
            "pool_order_group": "early_12" if pool < 12 else "late_12",
            "fresh_unique_rate": float(generation[pool]["fresh_unique_rate"]),
            "draws_used": int(generation[pool]["draws_used"]),
            "B_hp_regime_counts": dict(Counter(str(item["hp_regime"]) for item in b_rows[pool])),
            "B_repeat_architecture_pattern": repeat_pattern,
            "descriptive_only": True,
        })
    power = {"pool_count": POOL_COUNT, "components": {}}
    for name in COMPARISONS:
        values = np.asarray([float(row[name]) for row in pool_rows], dtype=np.float64)
        sd = float(np.std(values, ddof=1))
        alpha_quantile = 0.9875 if name != "B_minus_C" else 0.975
        mde = (float(norm.ppf(alpha_quantile)) + float(norm.ppf(0.8))) * sd / math.sqrt(POOL_COUNT)
        power["components"][name] = {
            "paired_pool_sd": sd,
            "normal_approximate_80pct_mde": max(0.006, mde),
            "approximation_only": True,
        }
    power["statement"] = (
        "The effective confirmatory sample size is 24 pools. Two random batches and three GNN "
        "seeds reduce within-pool uncertainty but do not increase n. The normal MDE is descriptive; "
        "the hierarchical max-based bootstrap can require a larger effect. No claim of 80% power "
        "for a true effect of exactly 0.006 is made."
    )
    return power, strata


def protected_after_audit(root: Path) -> dict[str, Any]:
    before = load_json(root / "protected_hashes_before.json")
    rows = []
    mismatches = []
    for path_string, identity in sorted(before.items()):
        path = Path(path_string)
        actual = file_identity(path) if path.is_file() else None
        row = {"path": path_string, "before": identity, "after": actual}
        rows.append(row)
        if actual is None or actual["sha256"] != identity["sha256"]:
            mismatches.append(row)
    return {"passed": not mismatches, "files": rows, "mismatches": mismatches}


def _secondary_v1_v2_summary(root: Path, pool_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [
        REPO_ROOT / V1_RELATIVE / "analysis/paired_pool_differences.csv",
        REPO_ROOT / V1_RELATIVE / "paired_pool_differences.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    v2 = [float(row["B_minus_C"]) for row in pool_rows]
    output: dict[str, Any] = {
        "classification_input": False,
        "v2_pool_count": len(v2),
        "v2_mean_B_minus_C": float(np.mean(v2)),
        "statement": "Any v1+v2 aggregation is secondary and cannot replace the independent v2 classification.",
    }
    if path is not None:
        import csv
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        field = "B_minus_C"
        if rows and field in rows[0]:
            v1 = [float(row[field]) for row in rows]
            output.update({
                "v1_source": str(path),
                "v1_pool_count": len(v1),
                "v1_mean_B_minus_C": float(np.mean(v1)),
                "unweighted_all_pool_mean": float(np.mean(v1 + v2)),
            })
    return output


def analyse(root: Path, shard_paths: Sequence[Path]) -> str:
    freeze = verify_freeze(root)
    if not freeze["passed"]:
        raise RuntimeError(f"pre-label immutable manifest failed: {freeze}")
    tasks = load_json(root / "selected_union_tasks.json")
    selections = load_json(root / "frozen_strategy_selections.json")
    results: list[dict[str, Any]] = []
    shard_counts = []
    for shard, path in enumerate(shard_paths):
        rows = load_jsonl(path)
        results.extend(rows)
        shard_counts.append({
            "shard": shard,
            "path": str(path),
            "result_count": len(rows),
            "configuration_count": len({config_key(row) for row in rows}),
            "gpu_uuids": sorted({row.get("actual_gpu", {}).get("uuid") for row in rows}),
        })
    results = sorted(results, key=result_key)
    audit = validate_complete_seed_results(tasks, results)
    phase_b = root / "phase_b"
    analysis = root / "analysis"
    phase_b.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    dump_json(analysis / "task_result_completeness_audit.json", {**audit, "shards": shard_counts})
    if not audit["passed"]:
        dump_json(root / "final_classification.json", {
            "format_version": FORMAT_VERSION,
            "classification": "EXECUTION_OR_PROTOCOL_FAILURE",
            "created_at": utc_now(),
            "evidence": audit,
            "next_stage_authorization": "none",
        })
        return "EXECUTION_OR_PROTOCOL_FAILURE"
    dump_jsonl(phase_b / "seed_results_merged.jsonl", results)
    compact_results = [{key: value for key, value in row.items() if key != "trajectory"} for row in results]
    dump_csv(phase_b / "seed_results_merged_summary.csv", compact_results)
    trajectory_rows = [{
        "task_key": row["task_key"],
        "pool_id": row["pool_id"],
        "config_id": row["config_id"],
        "seed": row["seed"],
        "sanitized_trajectory_hash": row["sanitized_trajectory_hash"],
        "epoch_count": len(row["trajectory"]),
        "best_epoch": row.get("best_epoch"),
        "stopped_epoch": row.get("stopped_epoch"),
    } for row in results]
    dump_json(analysis / "trajectory_integrity.json", {
        "passed": True,
        "trajectory_count": len(trajectory_rows),
        "all_saved_events_have_only_epoch_loss_validation": True,
        "used_for_selection_or_classification": False,
        "rows": trajectory_rows,
    })

    config_rows = aggregate_configurations(results)
    dump_json(analysis / "configuration_level_validation.json", config_rows)
    dump_csv(analysis / "configuration_level_validation.csv", config_rows)
    slot_values, slot_rows = build_slot_seed_values(selections, results)
    dump_json(analysis / "frozen_selection_slot_labels.json", slot_rows)
    dump_csv(analysis / "frozen_selection_slot_labels.csv", slot_rows)
    pool_rows, budget_rows = observed_pool_metrics(slot_values)
    dump_json(analysis / "pool_level_primary_metrics.json", pool_rows)
    dump_csv(analysis / "pool_level_primary_metrics.csv", pool_rows)
    dump_csv(analysis / "budget_1_2_4_descriptive.csv", budget_rows)

    prereg = load_json(root / "preregistration.json")
    samples, hierarchical, indices = hierarchical_bootstrap(
        slot_values,
        int(prereg["statistics"]["hierarchical_bootstrap_replicates"]),
        int(prereg["statistics"]["hierarchical_bootstrap_seed"]),
    )
    for index, name in enumerate(COMPARISONS):
        hierarchical["comparisons"][name]["observed_mean"] = float(
            np.mean([float(row[name]) for row in pool_rows])
        )
    dump_json(analysis / "hierarchical_bootstrap_summary.json", hierarchical)
    bootstrap_rows = [
        {"replicate": index, **{name: float(samples[index, column]) for column, name in enumerate(COMPARISONS)}}
        for index in range(samples.shape[0])
    ]
    dump_csv(analysis / "hierarchical_bootstrap_50000.csv", bootstrap_rows)
    pool_samples, pool_only = pool_only_bootstrap(
        pool_rows,
        int(prereg["statistics"]["hierarchical_bootstrap_replicates"]),
        int(prereg["statistics"]["pool_only_bootstrap_seed"]),
    )
    dump_json(analysis / "pool_only_bootstrap_summary.json", pool_only)
    dump_csv(analysis / "pool_only_bootstrap_50000.csv", [
        {"replicate": index, **{name: float(pool_samples[index, column]) for column, name in enumerate(COMPARISONS)}}
        for index in range(pool_samples.shape[0])
    ])

    classification, gate = classify(pool_rows, hierarchical)
    if classification not in FINAL_CLASSIFICATIONS:
        raise RuntimeError(f"invalid classification: {classification}")
    dump_json(analysis / "final_gate_evidence.json", gate)
    lopo = leave_one_pool_out(pool_rows)
    dump_csv(analysis / "leave_one_pool_out.csv", lopo)
    dump_json(analysis / "leave_one_pool_out_summary.json", {
        name: {
            "minimum": min(float(row[name]) for row in lopo),
            "maximum": max(float(row[name]) for row in lopo),
            "all_positive": all(float(row[name]) > 0 for row in lopo),
        }
        for name in COMPARISONS
    })
    regret = observed_union_regret(selections, config_rows, pool_rows)
    dump_csv(analysis / "observed_union_regret_descriptive.csv", regret)
    head = head_descriptive_metrics(
        load_json(root / "frozen_selected_union_head_predictions.json"), config_rows,
    )
    dump_csv(analysis / "head_observed_union_descriptive.csv", head)
    power, strata = power_and_strata(
        pool_rows, selections, load_json(root / "candidate_generation_audit.json"),
    )
    dump_json(analysis / "power_and_mde.json", power)
    dump_json(analysis / "pool_strata_descriptive.json", strata)
    dump_json(analysis / "v1_v2_secondary_summary.json", _secondary_v1_v2_summary(root, pool_rows))
    after = protected_after_audit(root)
    dump_json(root / "protected_hashes_after.json", after)
    if not after["passed"]:
        classification = "EXECUTION_OR_PROTOCOL_FAILURE"
        gate["protected_object_failure"] = after["mismatches"]

    authorization = {
        "JOINT_AND_ARCH_CONFIRMED": "single-generation local architecture mutation gate only",
        "HP_FILTERING_ONLY": "Original Head may select HP only; architecture must remain non-Head",
        "JOINT_CONFIRMED_ATTRIBUTION_INCONCLUSIVE": "none; architecture mutation not authorized",
        "NO_HEAD_ADVANTAGE": "none; stop the Head-dominated route",
        "CONFIRMATORY_INCONCLUSIVE": "none; do not append pools or alter strategies",
        "INSUFFICIENT_FRESH_CANDIDATES": "none",
        "EXECUTION_OR_PROTOCOL_FAILURE": "none",
    }[classification]
    final = {
        "format_version": FORMAT_VERSION,
        "classification": classification,
        "created_at": utc_now(),
        "evidence": gate,
        "hierarchical_bootstrap": hierarchical,
        "next_stage_authorization": authorization,
        "multi_generation_mutation_or_evolution_authorized": False,
        "formal_bo_authorized": False,
        "test_evaluation_authorized": False,
        "v1_v2_combination_used_for_classification": False,
    }
    dump_json(root / "final_classification.json", final)
    validation_lines = [
        "# Validation report",
        "",
        f"- Final classification: `{classification}`",
        f"- Seed tasks/configurations: {len(results)}/{len(config_rows)}",
        f"- Complete protocol audit: {audit['passed']}",
        f"- Protected objects unchanged: {after['passed']}",
        f"- B-C mean: {gate['means']['B_minus_C']:.6f}",
        f"- B-M mean: {gate['means']['B_minus_M']:.6f}",
        f"- M-C mean: {gate['means']['M_minus_C']:.6f}",
        f"- Joint gate: {gate['joint_gate_passed']}",
        f"- Architecture gate: {gate['architecture_component_gate_passed']}",
        f"- HP gate: {gate['hp_component_gate_passed']}",
        "- All primary inference used 24 equally weighted pools.",
        "- No test-named field occurs in any saved seed result or trajectory.",
        "- Trajectories were not used for selection or classification.",
        "- Observed-union regret is not full-pool true regret.",
    ]
    dump_text(root / "validation_report.md", "\n".join(validation_lines) + "\n")
    report_lines = [
        "# Final Goal report",
        "",
        f"Classification: **{classification}**",
        "",
        f"The frozen union contained {len(config_rows)} configurations and {len(results)} three-seed tasks.",
        (
            f"Across 24 pools, mean B-C={gate['means']['B_minus_C']:.6f}, "
            f"B-M={gate['means']['B_minus_M']:.6f}, and M-C={gate['means']['M_minus_C']:.6f}."
        ),
        "",
        f"Next-stage authorization: {authorization}.",
        "",
        "No VAE/Head/H/Refit model was trained, no BO or mutation search was run, and no test metric was saved.",
    ]
    dump_text(root / "final_goal_report.md", "\n".join(report_lines) + "\n")
    print(json.dumps({
        "classification": classification,
        "configuration_count": len(config_rows),
        "task_count": len(results),
        "means": gate["means"],
        "wins": gate["strict_wins"],
        "next_stage_authorization": authorization,
    }, indent=2, sort_keys=True))
    return classification


def package(root: Path, archive_name: str) -> Path:
    required = [
        root / "final_classification.json",
        root / "independent_data_execution_audit.md",
        root / "independent_statistical_audit.md",
        root / "validation_report.md",
        root / "final_goal_report.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot package before final audits: {missing}")
    freeze = verify_freeze(root)
    if not freeze["passed"]:
        raise RuntimeError(f"pre-label files drifted before packaging: {freeze}")
    dependency = load_json(root / "source_dependency_manifest.json")
    snapshot_rows = []
    snapshot_root = root / "source_snapshot"
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    for row in dependency["files"]:
        if not row["requires_delivery_snapshot"]:
            continue
        source = Path(row["path"])
        current_hash = sha256_file(source)
        if current_hash != row["sha256"]:
            raise RuntimeError(f"source dependency drifted: {source}")
        target = snapshot_root / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshot_hash = sha256_file(target)
        snapshot_rows.append({
            "source_path": str(source),
            "relative_path": row["relative_path"],
            "snapshot_path": str(target.relative_to(root)),
            "frozen_sha256": row["sha256"],
            "current_sha256": current_hash,
            "snapshot_sha256": snapshot_hash,
            "three_way_match": row["sha256"] == current_hash == snapshot_hash,
        })
    snapshot_audit = {
        "passed": len(snapshot_rows) == dependency["snapshot_required_count"]
        and all(row["three_way_match"] for row in snapshot_rows),
        "required_count": dependency["snapshot_required_count"],
        "snapshot_count": len(snapshot_rows),
        "files": snapshot_rows,
    }
    dump_json(root / "source_snapshot_audit.json", snapshot_audit)
    if not snapshot_audit["passed"]:
        raise RuntimeError("source snapshot closure audit failed")
    archive = root / archive_name
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    dump_json(root / "delivery_package.json", {
        "created_at": utc_now(),
        "archive_name": archive.name,
        "includes_source_dependency_snapshot_closure": True,
        "source_snapshot_file_count": len(snapshot_rows),
        "does_not_include_model_checkpoints": True,
        "does_not_include_dataset_copy": True,
        "external_hashed_dependencies_required_for_reexecution": True,
        "pre_label_manifest_root_hash": freeze["recomputed_root_hash"],
    })
    exclusions = {archive.resolve(), sidecar.resolve(), (root / "artifact_manifest.json").resolve()}
    payloads = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() not in exclusions
        and not path.name.endswith(".tar.gz") and not path.name.endswith(".tar.gz.sha256")
    )
    manifest_rows = [
        {
            "relative_path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in payloads
    ]
    dump_json(root / "artifact_manifest.json", {
        "created_at": utc_now(),
        "file_count": len(manifest_rows),
        "files": manifest_rows,
    })
    payloads.append(root / "artifact_manifest.json")
    if archive.exists():
        archive.unlink()
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in sorted(payloads):
                    info = tar.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    archive_hash = sha256_file(archive)
    sidecar.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        if any(not member.isfile() for member in members):
            raise RuntimeError("archive contains a non-regular member")
        member_names = {member.name for member in members}
        expected_names = {str(path.relative_to(root)) for path in payloads}
        if member_names != expected_names:
            raise RuntimeError("archive member set differs from packaged payload")
        for member in members:
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            archived_hash = hashlib.sha256(extracted.read()).hexdigest()
            if archived_hash != sha256_file(root / member.name):
                raise RuntimeError(f"archive member hash mismatch: {member.name}")
    print(json.dumps({
        "archive": str(archive),
        "sha256": archive_hash,
        "member_count": len(payloads),
        "source_snapshot_count": len(snapshot_rows),
    }, indent=2, sort_keys=True))
    return archive


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyse_parser = subparsers.add_parser("analyse")
    analyse_parser.add_argument("--root", type=Path, required=True)
    analyse_parser.add_argument("--shard0", type=Path, required=True)
    analyse_parser.add_argument("--shard1", type=Path, required=True)
    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--root", type=Path, required=True)
    package_parser.add_argument(
        "--archive-name",
        default="original-head-confirmatory-attribution-global4-v2-20260825.tar.gz",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "analyse":
        analyse(args.root.resolve(), [args.shard0.resolve(), args.shard1.resolve()])
    else:
        package(args.root.resolve(), args.archive_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
