#!/usr/bin/env python3
"""Analyse, stabilize, and package the high-regime validation experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import analyse_original_head_confirmatory_attribution_v2 as v2_analysis
from analyse import high_regime_prior_model_blind_validation as plan
from analyse import run_high_regime_prior_model_blind_validation_labels as runner


POOL_COUNT = plan.POOL_COUNT
INITIAL_SEEDS = plan.INITIAL_SEEDS
STABILIZATION_SEEDS = plan.STABILIZATION_SEEDS
STRATEGY_INDEX = {"HIGH_REGIME_PRIOR": 0, "BALANCED_RANDOM_HP": 1}
INDEX_STRATEGY = {value: key for key, value in STRATEGY_INDEX.items()}

utc_now = plan.utc_now
load_json = plan.load_json
dump_json = plan.dump_json
dump_csv = plan.dump_csv
dump_text = plan.dump_text
file_identity = plan.file_identity
sha256_file = plan.sha256_file
stable_hash = plan.stable_hash
strict_test_key_hits = plan.strict_test_key_hits
dump_jsonl = v2_analysis.dump_jsonl


def result_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return runner.result_key(row)


def config_key(row: Mapping[str, Any]) -> tuple[int, str, str, str]:
    return (
        int(row["pool_id"]), str(row["joint_config_fingerprint"]),
        str(row["architecture_fingerprint"]), str(row["hp_fingerprint"]),
    )


def validate_stage(
    tasks: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int], stage: str,
) -> dict[str, Any]:
    task_keys = [result_key(row) for row in tasks]
    result_keys = [result_key(row) for row in results]
    expected, actual = set(task_keys), set(result_keys)
    by_config: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        by_config[config_key(row)].append(row)
    seed_errors = []
    protocol_errors = []
    trajectory_errors = []
    gpu_errors = []
    for key, rows in sorted(by_config.items()):
        seeds = [int(row["seed"]) for row in rows]
        if sorted(seeds) != sorted(expected_seeds) or len(seeds) != len(expected_seeds):
            seed_errors.append({"config_key": list(key), "seeds": sorted(seeds)})
        invariants = {(
            row.get("dataset_content_fingerprint"), row.get("split_fingerprint"),
            row.get("hp_mode"), row.get("normalization"), row.get("eval_epochs"),
            row.get("patience"), row.get("task_stage"), row.get("assigned_gpu"),
        ) for row in rows}
        if len(invariants) != 1 or next(iter(invariants))[6] != stage:
            protocol_errors.append(list(key))
        assigned = {int(row["assigned_gpu"]) for row in rows}
        physical = {int(row["actual_gpu"]["physical_index"]) for row in rows}
        if assigned != physical or len(assigned) != 1:
            gpu_errors.append({"config_key": list(key), "assigned": sorted(assigned), "physical": sorted(physical)})
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
    statuses = Counter(str(row.get("terminal_status")) for row in results)
    test_hits = strict_test_key_hits(results)
    passed = (
        len(task_keys) == len(expected) and len(result_keys) == len(actual)
        and expected == actual and not seed_errors and not protocol_errors
        and not trajectory_errors and not gpu_errors and not test_hits
        and statuses == Counter({"success": len(results)})
        and all(bool(row.get("valid")) for row in results)
    )
    return {
        "passed": passed, "stage": stage, "expected_task_count": len(tasks),
        "actual_result_count": len(results), "configuration_count": len(by_config),
        "duplicate_task_count": len(task_keys) - len(expected),
        "duplicate_result_count": len(result_keys) - len(actual),
        "missing_keys": [list(key) for key in sorted(expected - actual)],
        "foreign_keys": [list(key) for key in sorted(actual - expected)],
        "seed_errors": seed_errors, "protocol_errors": protocol_errors,
        "trajectory_errors": trajectory_errors, "gpu_errors": gpu_errors,
        "terminal_status_counts": dict(statuses),
        "retry_count": sum(int(row.get("retry_count", 0)) for row in results),
        "test_field_hits": test_hits,
    }


def aggregate_configurations(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[config_key(row)].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["seed"]))
        values = np.asarray([float(row["val_acc"]) for row in ordered], dtype=np.float64)
        output.append({
            "pool_id": key[0], "joint_config_fingerprint": key[1],
            "architecture_fingerprint": key[2], "hp_fingerprint": key[3],
            "config_id": str(ordered[0]["config_id"]), "batch_id": int(ordered[0]["batch_id"]),
            "architecture_slot": int(ordered[0]["architecture_slot"]),
            "hp_regime": str(ordered[0]["hp_regime"]),
            "seeds": [int(row["seed"]) for row in ordered],
            "seed_validation_values": values.tolist(),
            "mean_validation": float(values.mean()),
            "std_validation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "assigned_gpu": int(ordered[0]["assigned_gpu"]),
        })
    return output


def build_slot_seed_values(
    selections: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = {
        (int(row["pool_id"]), str(row["joint_config_fingerprint"]), int(row["seed"])): float(row["val_acc"])
        for row in results
    }
    values = np.full((POOL_COUNT, 2, 2, 4, len(INITIAL_SEEDS)), np.nan, dtype=np.float64)
    rows = []
    seen = set()
    for selection in selections:
        pool = int(selection["pool_id"])
        batch = int(selection["batch_id"])
        strategy = str(selection["strategy"])
        strategy_index = STRATEGY_INDEX[strategy]
        slot = int(selection["budget_order"]) - 1
        cell = (pool, batch, strategy_index, slot)
        if cell in seen:
            raise ValueError(f"duplicate frozen selection slot: {cell}")
        seen.add(cell)
        seed_values = [
            labels[(pool, str(selection["joint_config_fingerprint"]), seed)]
            for seed in INITIAL_SEEDS
        ]
        values[pool, batch, strategy_index, slot, :] = seed_values
        rows.append({
            "pool_id": pool, "batch_id": batch, "strategy": strategy,
            "budget_order": slot + 1,
            "joint_config_fingerprint": selection["joint_config_fingerprint"],
            "architecture_fingerprint": selection["architecture_fingerprint"],
            "hp_fingerprint": selection["hp_fingerprint"],
            "hp_regime": selection["hp_regime"],
            "seed_validation_values": seed_values,
            "mean_validation": float(np.mean(seed_values)),
        })
    if len(seen) != POOL_COUNT * 2 * 2 * 4 or np.isnan(values).any():
        raise ValueError("F/R selection tensor is incomplete")
    # Shared high is an identity pointer: its seed arrays must be exactly equal.
    for pool in range(POOL_COUNT):
        for batch in range(2):
            f_rows = [r for r in rows if r["pool_id"] == pool and r["batch_id"] == batch and r["strategy"] == "HIGH_REGIME_PRIOR"]
            r_rows = [r for r in rows if r["pool_id"] == pool and r["batch_id"] == batch and r["strategy"] == "BALANCED_RANDOM_HP"]
            shared = [(f, r) for f in f_rows for r in r_rows if f["joint_config_fingerprint"] == r["joint_config_fingerprint"]]
            if len(shared) != 1 or shared[0][0]["seed_validation_values"] != shared[0][1]["seed_validation_values"]:
                raise ValueError("shared high configuration lost label identity")
    return values, rows


def observed_pool_metrics(values: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config_means = values.mean(axis=4)
    best = config_means.max(axis=3)
    pool_rows = []
    budget_rows = []
    for pool in range(POOL_COUNT):
        f_batches = best[pool, :, 0]
        r_batches = best[pool, :, 1]
        f = float(f_batches.mean())
        r = float(r_batches.mean())
        pool_rows.append({
            "pool_id": pool, "F_batch0_best4": float(f_batches[0]),
            "F_batch1_best4": float(f_batches[1]), "F_mean_batch_best4": f,
            "R_batch0_best4": float(r_batches[0]),
            "R_batch1_best4": float(r_batches[1]), "R_mean_batch_best4": r,
            "F_minus_R": f - r,
        })
        for budget in (1, 2, 4):
            prefix = config_means[pool, :, :, :budget].max(axis=2)
            budget_rows.extend([
                {"pool_id": pool, "strategy": "F", "budget": budget, "best_so_far": float(prefix[:, 0].mean())},
                {"pool_id": pool, "strategy": "R", "budget": budget, "best_so_far": float(prefix[:, 1].mean())},
            ])
    return pool_rows, budget_rows


def _bootstrap_indices(replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pools = rng.integers(0, POOL_COUNT, size=(replicates, POOL_COUNT), dtype=np.int16).astype("<i2", copy=False)
    batches = rng.integers(0, 2, size=(replicates, POOL_COUNT, 2), dtype=np.uint8).astype("u1", copy=False)
    seeds = rng.integers(0, len(INITIAL_SEEDS), size=(replicates, POOL_COUNT, 3), dtype=np.uint8).astype("u1", copy=False)
    return pools, batches, seeds


def hierarchical_bootstrap(
    values: np.ndarray, replicates: int, seed: int, chunk_size: int = 1000,
) -> tuple[np.ndarray, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if values.shape != (POOL_COUNT, 2, 2, 4, len(INITIAL_SEEDS)):
        raise ValueError(f"unexpected F/R tensor shape: {values.shape}")
    pool_indices, batch_indices, seed_indices = _bootstrap_indices(replicates, seed)
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        width = stop - start
        accumulated = np.zeros(width, dtype=np.float64)
        for position in range(POOL_COUNT):
            sampled_pools = pool_indices[start:stop, position].astype(np.intp)
            selected = values[sampled_pools]
            shared_seeds = seed_indices[start:stop, position, :]
            gather_seeds = np.broadcast_to(shared_seeds[:, None, None, None, :], selected.shape)
            config_means = np.take_along_axis(selected, gather_seeds, axis=4).mean(axis=4)
            best = config_means.max(axis=3)  # width,batch,strategy
            matched_batches = batch_indices[start:stop, position, :].astype(np.intp)
            gather_batches = np.broadcast_to(matched_batches[:, :, None], (width, 2, 2))
            sampled = np.take_along_axis(best, gather_batches, axis=1).mean(axis=1)
            accumulated += sampled[:, 0] - sampled[:, 1]
        samples[start:stop] = accumulated / POOL_COUNT
    tag = b"high-regime-model-blind-hierarchical-bootstrap-v1\x00"
    summary = {
        "replicates": replicates, "seed": seed, "prng": "numpy.random.default_rng/PCG64",
        "numpy_version": np.__version__, "quantile_method": "numpy.quantile default linear",
        "pool_shape": list(pool_indices.shape), "batch_shape": list(batch_indices.shape),
        "seed_shape": list(seed_indices.shape), "pool_dtype": str(pool_indices.dtype),
        "batch_dtype": str(batch_indices.dtype), "seed_dtype": str(seed_indices.dtype),
        "pool_indices_sha256": hashlib.sha256(pool_indices.tobytes(order="C")).hexdigest(),
        "matched_batch_indices_sha256": hashlib.sha256(batch_indices.tobytes(order="C")).hexdigest(),
        "shared_seed_indices_sha256": hashlib.sha256(seed_indices.tobytes(order="C")).hexdigest(),
        "combined_indices_sha256": hashlib.sha256(
            tag + pool_indices.tobytes(order="C") + batch_indices.tobytes(order="C") + seed_indices.tobytes(order="C")
        ).hexdigest(),
        "same_outer_pool_indices_F_R": True, "same_matched_batch_indices_F_R": True,
        "same_seed_indices_all_configs_and_strategies": True,
        "best_of_4_recomputed_after_seed_resampling": True,
        "F_minus_R": {
            "bootstrap_mean": float(samples.mean()),
            "ci95_lower": float(np.quantile(samples, 0.025)),
            "ci95_upper": float(np.quantile(samples, 0.975)),
        },
    }
    return samples, summary, (pool_indices, batch_indices, seed_indices)


def pool_only_bootstrap(pool_rows: Sequence[Mapping[str, Any]], replicates: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray([float(row["F_minus_R"]) for row in pool_rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, POOL_COUNT, size=(replicates, POOL_COUNT), dtype=np.int16).astype("<i2", copy=False)
    samples = values[indices].mean(axis=1)
    return samples, {
        "replicates": replicates, "seed": seed, "outer_unit": "pool only",
        "indices_sha256": hashlib.sha256(indices.tobytes(order="C")).hexdigest(),
        "fixed_observed_three_seed_means": True, "fixed_two_batch_arithmetic_mean": True,
        "F_minus_R": {
            "bootstrap_mean": float(samples.mean()),
            "ci95_lower": float(np.quantile(samples, 0.025)),
            "ci95_upper": float(np.quantile(samples, 0.975)),
        },
    }


def classify(pool_rows: Sequence[Mapping[str, Any]], hierarchical: Mapping[str, Any], agreement: int) -> tuple[str, dict[str, Any]]:
    values = np.asarray([float(row["F_minus_R"]) for row in pool_rows], dtype=np.float64)
    mean = float(values.mean())
    wins, losses, ties = int((values > 0).sum()), int((values < 0).sum()), int((values == 0).sum())
    ci = hierarchical["F_minus_R"]
    gate = agreement >= 191 and mean >= 0.006 and float(ci["ci95_lower"]) > 0 and wins >= 15
    if gate:
        classification = "HIGH_REGIME_PRIOR_CONFIRMED_HEAD_REDUNDANT"
    elif agreement >= 191 and mean <= 0 and float(ci["ci95_upper"]) < 0.006:
        classification = "HIGH_REGIME_PRIOR_NOT_CONFIRMED"
    else:
        classification = "HIGH_REGIME_PRIOR_INCONCLUSIVE"
    non_ties = wins + losses
    sign = stats.binomtest(wins, n=non_ties, p=0.5, alternative="two-sided") if non_ties else None
    evidence = {
        "head_high_agreement_count": agreement, "head_high_agreement_gate": agreement >= 191,
        "mean_F": float(np.mean([row["F_mean_batch_best4"] for row in pool_rows])),
        "mean_R": float(np.mean([row["R_mean_batch_best4"] for row in pool_rows])),
        "mean_F_minus_R": mean, "strict_wins": wins, "strict_losses": losses, "ties": ties,
        "hierarchical_ci95_lower": float(ci["ci95_lower"]),
        "hierarchical_ci95_upper": float(ci["ci95_upper"]),
        "mean_gate": mean >= 0.006, "ci_lower_gate": float(ci["ci95_lower"]) > 0,
        "win_gate": wins >= 15, "all_primary_gates_passed": gate,
        "exact_two_sided_sign_test_pvalue": float(sign.pvalue) if sign else None,
        "classification_rule": "pre-registered v1 high-regime decision tree",
    }
    return classification, evidence


def _read_shards(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    summary = []
    for shard, path in enumerate(paths):
        rows = runner.load_jsonl(path)
        results.extend(rows)
        summary.append({
            "shard": shard, "path": str(path), "result_count": len(rows),
            "configuration_count": len({config_key(row) for row in rows}),
            "gpu_uuids": sorted({row.get("actual_gpu", {}).get("uuid") for row in rows}),
        })
    return sorted(results, key=result_key), summary


def _freeze_top8(root: Path, config_rows: Sequence[Mapping[str, Any]], selections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    config_lookup = {str(row["joint_config_fingerprint"]): row for row in config_rows}
    frozen = []
    for strategy in ("HIGH_REGIME_PRIOR", "BALANCED_RANDOM_HP"):
        strategy_fps = sorted({
            str(row["joint_config_fingerprint"]) for row in selections if row["strategy"] == strategy
        })
        if len(strategy_fps) != 192:
            raise RuntimeError(f"{strategy} does not have 192 unique initial configurations")
        ranked = sorted(strategy_fps, key=lambda fp: (-float(config_lookup[fp]["mean_validation"]), fp))
        for rank, fp in enumerate(ranked[:8], 1):
            row = config_lookup[fp]
            frozen.append({
                "strategy": strategy, "initial_rank": rank,
                "joint_config_fingerprint": fp, "pool_id": int(row["pool_id"]),
                "batch_id": int(row["batch_id"]), "architecture_slot": int(row["architecture_slot"]),
                "architecture_fingerprint": row["architecture_fingerprint"],
                "hp_fingerprint": row["hp_fingerprint"], "hp_regime": row["hp_regime"],
                "initial_three_seed_mean_validation": float(row["mean_validation"]),
                "selection_seed_ids": list(INITIAL_SEEDS),
                "tie_break": "joint_config_fingerprint",
            })
    union_by_fp: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frozen:
        union_by_fp[str(row["joint_config_fingerprint"])].append(row)
    initial_union = {str(row["joint_config_fingerprint"]): row for row in load_json(root / "selected_union_configs.json")}
    union = []
    tasks = []
    for fp, rows in sorted(union_by_fp.items()):
        source = dict(initial_union[fp])
        selected_by = sorted([{
            "strategy": row["strategy"], "initial_rank": row["initial_rank"],
        } for row in rows], key=lambda row: (row["strategy"], row["initial_rank"]))
        config = {**source, "selected_by": selected_by, "task_stage": "stabilization"}
        union.append(config)
        for seed in STABILIZATION_SEEDS:
            tasks.append({
                **config, "seed": seed,
                "task_key": f"stabilization::{config['pool_id']}::{fp}::{seed}",
            })
    if len(union) > 16 or len(tasks) != len(union) * 27 or len(tasks) > 432:
        raise RuntimeError("top8 union exceeds pre-registered stabilization budget")
    stabilization = root / "stabilization"
    dump_json(stabilization / "frozen_top8_selections.json", frozen)
    dump_csv(stabilization / "frozen_top8_selections.csv", frozen)
    dump_json(stabilization / "stabilization_union_configs.json", union)
    dump_csv(stabilization / "stabilization_union_configs.csv", union)
    dump_json(stabilization / "stabilization_tasks.json", tasks)
    dump_csv(stabilization / "stabilization_tasks.csv", tasks)
    audit = {
        "passed": len(frozen) == 16 and len(union) <= 16 and len(tasks) <= 432
        and all(row["selection_seed_ids"] == [0, 1, 2] for row in frozen)
        and set(int(row["seed"]) for row in tasks) == set(STABILIZATION_SEEDS)
        and len({row["task_key"] for row in tasks}) == len(tasks),
        "top8_slot_count": len(frozen), "union_configuration_count": len(union),
        "additional_task_count": len(tasks), "maximum_additional_tasks": 432,
        "strategy_overlap_configuration_count": 16 - len(union),
        "top8_selection_read_seed_ids": [0, 1, 2],
        "additional_seed_ids": list(STABILIZATION_SEEDS),
    }
    dump_json(stabilization / "stabilization_firewall_audit.json", audit)
    if not audit["passed"] or strict_test_key_hits(tasks):
        raise RuntimeError(f"stabilization freeze audit failed: {audit}")
    frozen_names = [
        "phase_b/initial/gpu0/seed_results.jsonl", "phase_b/initial/gpu1/seed_results.jsonl",
        "phase_b/initial/seed_results_merged.jsonl", "analysis/initial_configuration_validation.json",
        "analysis/pool_level_primary_metrics.json", "analysis/hierarchical_bootstrap_summary.json",
        "analysis/final_gate_evidence.json", "first_stage_classification.json",
        "stabilization/frozen_top8_selections.json", "stabilization/frozen_top8_selections.csv",
        "stabilization/stabilization_union_configs.json", "stabilization/stabilization_union_configs.csv",
        "stabilization/stabilization_tasks.json", "stabilization/stabilization_tasks.csv",
        "stabilization/stabilization_firewall_audit.json",
    ]
    entries = [{**file_identity(root / name), "relative_path": name} for name in frozen_names]
    root_hash = stable_hash([(row["relative_path"], row["sha256"]) for row in entries])
    dump_json(stabilization / "stabilization_immutable_manifest.json", {
        "format_version": plan.FORMAT_VERSION, "created_at": utc_now(), "files": entries,
        "manifest_root_hash": root_hash, "immutable_after_stabilization_label_connection": True,
    })
    return {**audit, "manifest_root_hash": root_hash}


def analyse_initial(root: Path, shard_paths: Sequence[Path]) -> str:
    freeze = plan.verify_freeze(root)
    if not freeze["passed"]:
        raise RuntimeError(f"pre-label freeze failed: {freeze}")
    tasks = load_json(root / "selected_union_tasks.json")
    selections = load_json(root / "frozen_strategy_selections.json")
    results, shards = _read_shards(shard_paths)
    audit = validate_stage(tasks, results, INITIAL_SEEDS, "initial")
    phase = root / "phase_b/initial"
    analysis = root / "analysis"
    dump_json(analysis / "initial_task_result_completeness_audit.json", {**audit, "shards": shards})
    if not audit["passed"]:
        dump_json(root / "final_classification.json", {
            "format_version": plan.FORMAT_VERSION, "classification": "EXECUTION_OR_PROTOCOL_FAILURE",
            "created_at": utc_now(), "evidence": audit, "test_evaluation_authorized": False,
        })
        return "EXECUTION_OR_PROTOCOL_FAILURE"
    dump_jsonl(phase / "seed_results_merged.jsonl", results)
    dump_csv(phase / "seed_results_merged_summary.csv", [{k: v for k, v in row.items() if k != "trajectory"} for row in results])
    config_rows = aggregate_configurations(results)
    dump_json(analysis / "initial_configuration_validation.json", config_rows)
    dump_csv(analysis / "initial_configuration_validation.csv", config_rows)
    values, slot_rows = build_slot_seed_values(selections, results)
    dump_json(analysis / "initial_frozen_slot_labels.json", slot_rows)
    dump_csv(analysis / "initial_frozen_slot_labels.csv", slot_rows)
    pool_rows, budget_rows = observed_pool_metrics(values)
    dump_json(analysis / "pool_level_primary_metrics.json", pool_rows)
    dump_csv(analysis / "pool_level_primary_metrics.csv", pool_rows)
    dump_csv(analysis / "budget_1_2_4_descriptive.csv", budget_rows)
    prereg = load_json(root / "preregistration.json")
    samples, hierarchical, _ = hierarchical_bootstrap(
        values, int(prereg["statistics"]["hierarchical_bootstrap_replicates"]),
        int(prereg["statistics"]["hierarchical_bootstrap_seed"]),
    )
    observed_mean = float(np.mean([row["F_minus_R"] for row in pool_rows]))
    hierarchical["F_minus_R"]["observed_mean"] = observed_mean
    dump_json(analysis / "hierarchical_bootstrap_summary.json", hierarchical)
    dump_csv(analysis / "hierarchical_bootstrap_50000.csv", [
        {"replicate": index, "F_minus_R": float(value)} for index, value in enumerate(samples)
    ])
    pool_samples, pool_only = pool_only_bootstrap(
        pool_rows, int(prereg["statistics"]["hierarchical_bootstrap_replicates"]),
        int(prereg["statistics"]["pool_only_bootstrap_seed"]),
    )
    dump_json(analysis / "pool_only_bootstrap_summary.json", pool_only)
    dump_csv(analysis / "pool_only_bootstrap_50000.csv", [
        {"replicate": index, "F_minus_R": float(value)} for index, value in enumerate(pool_samples)
    ])
    lopo = []
    for removed in range(POOL_COUNT):
        retained = [float(row["F_minus_R"]) for row in pool_rows if int(row["pool_id"]) != removed]
        lopo.append({"removed_pool": removed, "mean_F_minus_R": float(np.mean(retained))})
    dump_csv(analysis / "leave_one_pool_out.csv", lopo)
    agreement = int(load_json(root / "new_pool_head_high_regime_collapse_audit.json")["agreement_count"])
    classification, gate = classify(pool_rows, hierarchical, agreement)
    dump_json(analysis / "final_gate_evidence.json", gate)
    first = {
        "format_version": plan.FORMAT_VERSION, "classification": classification,
        "created_at": utc_now(), "evidence": gate, "hierarchical_bootstrap": hierarchical,
        "classification_immutable_after_initial_analysis": True,
        "method_name_if_confirmed": "model-blind architecture search with a high-regime HP prior",
        "mutation_evolution_bo_authorized": False, "test_evaluation_performed": False,
        "stabilization_authorized": classification in plan.FINAL_CLASSIFICATIONS[:3],
    }
    dump_json(root / "first_stage_classification.json", first)
    if classification not in plan.FINAL_CLASSIFICATIONS[:3]:
        dump_json(root / "final_classification.json", first)
        return classification
    stabilization = _freeze_top8(root, config_rows, selections)
    print(json.dumps({
        "classification": classification, "mean_F_minus_R": gate["mean_F_minus_R"],
        "ci95": [gate["hierarchical_ci95_lower"], gate["hierarchical_ci95_upper"]],
        "wins": gate["strict_wins"], "top8_union_configs": stabilization["union_configuration_count"],
        "additional_tasks": stabilization["additional_task_count"],
    }, indent=2, sort_keys=True))
    return classification


def _strategy_top8_statistics(
    top8: Sequence[Mapping[str, Any]], initial_results: Sequence[Mapping[str, Any]],
    additional_results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined = list(initial_results) + list(additional_results)
    values: dict[str, dict[int, float]] = defaultdict(dict)
    for row in combined:
        values[str(row["joint_config_fingerprint"])][int(row["seed"])] = float(row["val_acc"])
    rows = []
    summary = {}
    for strategy in ("HIGH_REGIME_PRIOR", "BALANCED_RANDOM_HP"):
        strategy_rows = []
        for frozen in [row for row in top8 if row["strategy"] == strategy]:
            fp = str(frozen["joint_config_fingerprint"])
            seed_map = values[fp]
            if set(seed_map) != set(range(30)):
                raise RuntimeError(f"top8 configuration lacks fixed seeds 0..29: {fp}")
            all_values = np.asarray([seed_map[seed] for seed in range(30)], dtype=np.float64)
            holdout = np.asarray([seed_map[seed] for seed in STABILIZATION_SEEDS], dtype=np.float64)
            sem = stats.sem(all_values)
            half = float(stats.t.ppf(0.975, 29) * sem)
            row = {
                **dict(frozen), "all30_mean_validation": float(all_values.mean()),
                "all30_std_validation": float(all_values.std(ddof=1)),
                "all30_ci95_lower": float(all_values.mean() - half),
                "all30_ci95_upper": float(all_values.mean() + half),
                "holdout27_mean_validation_sensitivity": float(holdout.mean()),
                "winner_curse_initial_minus_all30": float(frozen["initial_three_seed_mean_validation"] - all_values.mean()),
                "all30_seed_values": all_values.tolist(),
                "holdout27_not_used_to_rewrite_first_stage_classification": True,
            }
            strategy_rows.append(row)
        ranked = sorted(strategy_rows, key=lambda row: (-row["all30_mean_validation"], row["joint_config_fingerprint"]))
        for rank, row in enumerate(ranked, 1):
            row["final_all30_rank"] = rank
        initial_order = sorted(strategy_rows, key=lambda row: int(row["initial_rank"]))
        final_rank = {row["joint_config_fingerprint"]: row["final_all30_rank"] for row in ranked}
        initial_ranks = np.arange(1, 9, dtype=float)
        final_ranks = np.asarray([final_rank[row["joint_config_fingerprint"]] for row in initial_order], dtype=float)
        tau = stats.kendalltau(initial_ranks, final_ranks)
        summary[strategy] = {
            "final_candidate_joint_config_fingerprint": ranked[0]["joint_config_fingerprint"],
            "final_candidate_all30_mean_validation": ranked[0]["all30_mean_validation"],
            "final_candidate_all30_std_validation": ranked[0]["all30_std_validation"],
            "final_candidate_ci95": [ranked[0]["all30_ci95_lower"], ranked[0]["all30_ci95_upper"]],
            "top1_minus_top2_all30": ranked[0]["all30_mean_validation"] - ranked[1]["all30_mean_validation"],
            "initial_vs_all30_top8_kendall_tau": float(tau.statistic),
            "initial_vs_all30_top8_kendall_pvalue": float(tau.pvalue),
            "selection_rule": "highest all-30-seed mean within frozen initial top8; tie by joint fingerprint",
            "holdout27_sensitivity_reported": True,
        }
        rows.extend(ranked)
    return rows, summary


def analyse_final(root: Path, shard_paths: Sequence[Path]) -> str:
    initial_freeze = plan.verify_freeze(root)
    stabilization_freeze = runner.verify_stabilization_freeze(root)
    if not initial_freeze["passed"] or not stabilization_freeze["passed"]:
        raise RuntimeError("initial or stabilization immutable freeze failed")
    first = load_json(root / "first_stage_classification.json")
    if first["classification"] not in plan.FINAL_CLASSIFICATIONS[:3]:
        raise RuntimeError("first-stage classification did not authorize stabilization")
    initial_results = runner.load_jsonl(root / "phase_b/initial/seed_results_merged.jsonl")
    tasks = load_json(root / "stabilization/stabilization_tasks.json")
    additional, shards = _read_shards(shard_paths)
    audit = validate_stage(tasks, additional, STABILIZATION_SEEDS, "stabilization")
    dump_json(root / "analysis/stabilization_task_result_completeness_audit.json", {**audit, "shards": shards})
    if not audit["passed"]:
        dump_json(root / "final_classification.json", {
            "format_version": plan.FORMAT_VERSION, "classification": "EXECUTION_OR_PROTOCOL_FAILURE",
            "created_at": utc_now(), "first_stage_classification": first["classification"],
            "evidence": audit, "test_evaluation_authorized": False,
        })
        return "EXECUTION_OR_PROTOCOL_FAILURE"
    phase = root / "phase_c/stabilization"
    dump_jsonl(phase / "seed_results_merged.jsonl", additional)
    dump_csv(phase / "seed_results_merged_summary.csv", [{k: v for k, v in row.items() if k != "trajectory"} for row in additional])
    top8 = load_json(root / "stabilization/frozen_top8_selections.json")
    rows, summary = _strategy_top8_statistics(top8, initial_results, additional)
    dump_json(root / "stabilization/strategy_top8_30seed_statistics.json", rows)
    dump_csv(root / "stabilization/strategy_top8_30seed_statistics.csv", rows)
    dump_json(root / "stabilization/final_candidate_summary.json", summary)
    for strategy, filename in (
        ("HIGH_REGIME_PRIOR", "F_final_candidate.json"),
        ("BALANCED_RANDOM_HP", "R_final_candidate.json"),
    ):
        candidate = next(row for row in rows if row["strategy"] == strategy and row["final_all30_rank"] == 1)
        dump_json(root / f"stabilization/{filename}", candidate)
    after = v2_analysis.protected_after_audit(root)
    dump_json(root / "protected_hashes_after.json", after)
    classification = first["classification"] if after["passed"] else "EXECUTION_OR_PROTOCOL_FAILURE"
    all_results = initial_results + additional
    starts = [datetime.fromisoformat(row["started_at"]) for row in all_results]
    finishes = [datetime.fromisoformat(row["finished_at"]) for row in all_results]
    execution = {
        "architecture_count": 192, "initial_union_configuration_count": 336,
        "initial_task_count": len(initial_results),
        "stabilization_union_configuration_count": len({row["joint_config_fingerprint"] for row in additional}),
        "stabilization_task_count": len(additional), "total_task_count": len(all_results),
        "success_count": sum(row["terminal_status"] == "success" for row in all_results),
        "failed_count": sum(row["terminal_status"] == "failed" for row in all_results),
        "invalid_count": sum(not bool(row["valid"]) for row in all_results),
        "retry_count": sum(int(row["retry_count"]) for row in all_results),
        "first_started_at": min(starts).isoformat(), "last_finished_at": max(finishes).isoformat(),
        "combined_wall_seconds": (max(finishes) - min(starts)).total_seconds(),
        "summed_task_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in all_results),
        "gpu_uuids": sorted({row["actual_gpu"]["uuid"] for row in all_results}),
    }
    dump_json(root / "execution_summary.json", execution)
    next_authorization = (
        "separately preregistered one-time test reveal of the two frozen final candidates only"
        if classification in plan.FINAL_CLASSIFICATIONS[:3] else "none"
    )
    final = {
        **first, "classification": classification, "created_at": utc_now(),
        "first_stage_classification_unchanged": classification == first["classification"],
        "stabilization": summary, "execution": execution,
        "next_stage_authorization": next_authorization,
        "one_time_test_reveal_performed": False,
        "one_time_test_reveal_authorized_only_as_separate_frozen_stage": classification in plan.FINAL_CLASSIFICATIONS[:3],
        "mutation_evolution_bo_authorized": False,
        "model_or_proxy_training_performed": False,
        "formal_method_name": (
            "model-blind architecture search with a high-regime HP prior"
            if classification == "HIGH_REGIME_PRIOR_CONFIRMED_HEAD_REDUNDANT"
            else "balanced-random joint NAS+HPO"
        ),
    }
    dump_json(root / "final_classification.json", final)
    gate = first["evidence"]
    dump_text(root / "validation_report.md", "\n".join([
        "# Validation report", "", f"- Classification: `{classification}`",
        f"- H/F agreement: {gate['head_high_agreement_count']}/192",
        f"- Initial tasks: {len(initial_results)}; stabilization tasks: {len(additional)}; total: {len(all_results)}",
        f"- F-R mean: {gate['mean_F_minus_R']:.6f}",
        f"- Hierarchical 95% CI: [{gate['hierarchical_ci95_lower']:.6f}, {gate['hierarchical_ci95_upper']:.6f}]",
        f"- F wins: {gate['strict_wins']}/24", f"- Protected objects unchanged: {after['passed']}",
        "- No test-named metric was saved; trajectories contain epoch/loss/validation only.",
        "- The first-stage classification was not rewritten by the 30-seed stabilization.",
    ]) + "\n")
    dump_text(root / "final_goal_report.md", "\n".join([
        "# Final Goal report", "", f"Classification: **{classification}**", "",
        f"The pre-label Head/high agreement was {gate['head_high_agreement_count']}/192.",
        f"Across 24 pools, F-R={gate['mean_F_minus_R']:.6f} with 95% hierarchical interval "
        f"[{gate['hierarchical_ci95_lower']:.6f}, {gate['hierarchical_ci95_upper']:.6f}] and {gate['strict_wins']} wins.",
        "", f"F final candidate mean={summary['HIGH_REGIME_PRIOR']['final_candidate_all30_mean_validation']:.6f}; "
        f"R final candidate mean={summary['BALANCED_RANDOM_HP']['final_candidate_all30_mean_validation']:.6f}.",
        "", f"Next authorization: {next_authorization}.", "",
        "No proxy was trained, no mutation/evolution/BO was run, and no test metric was accessed.",
    ]) + "\n")
    print(json.dumps({
        "classification": classification, "execution": execution, "gate": gate,
        "final_candidates": summary, "next_stage_authorization": next_authorization,
    }, indent=2, sort_keys=True))
    return classification


def package(root: Path, archive_name: str) -> Path:
    freeze = runner.verify_stabilization_freeze(root)
    if not freeze["passed"]:
        raise RuntimeError(f"stabilization freeze drifted before package: {freeze}")
    return v2_analysis.package(root, archive_name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    initial = sub.add_parser("analyse-initial")
    initial.add_argument("--root", type=Path, required=True)
    initial.add_argument("--shard0", type=Path, required=True)
    initial.add_argument("--shard1", type=Path, required=True)
    final = sub.add_parser("analyse-final")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--shard0", type=Path, required=True)
    final.add_argument("--shard1", type=Path, required=True)
    package_parser = sub.add_parser("package")
    package_parser.add_argument("--root", type=Path, required=True)
    package_parser.add_argument("--archive-name", default="high-regime-prior-model-blind-validation-global4-v1-20260826.tar.gz")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if args.command == "analyse-initial":
        analyse_initial(root, [args.shard0.resolve(), args.shard1.resolve()])
    elif args.command == "analyse-final":
        analyse_final(root, [args.shard0.resolve(), args.shard1.resolve()])
    else:
        package(root, args.archive_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
