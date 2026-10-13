#!/usr/bin/env python3
"""Freeze the high-regime HP prior/model-blind architecture experiment.

This module has no validation-label input.  It verifies the v2 evidence,
creates one globally generated and randomly permuted reservoir of 192 fresh
architectures, freezes the F/R assignments, and writes the immutable initial
three-seed task plan.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import original_head_confirmatory_attribution_v2 as v2
from analyse import original_head_deterministic_batch_pilot as v1


FORMAT_VERSION = "high_regime_prior_model_blind_validation_global4_v1"
OUTPUT_RELATIVE = "results/search/high-regime-prior-model-blind-validation-global4-v1"
V2_RELATIVE = "results/posthoc/original-head-confirmatory-attribution-global4-v2"
STRICT_RELATIVE = v2.STRICT_RELATIVE
BASE_CHECKPOINT = v2.BASE_CHECKPOINT
MASTER_SEED = 20260826
POOL_COUNT = 24
BATCHES_PER_POOL = 2
ARCHITECTURES_PER_BATCH = 4
ARCHITECTURE_COUNT = POOL_COUNT * BATCHES_PER_POOL * ARCHITECTURES_PER_BATCH
HP_REGIMES = dict(v2.HP_REGIMES)
REGIME_ORDER = ("hp_low", "hp_mid_low", "hp_mid_high", "hp_high")
INITIAL_DRAWS = 4096
MAX_DRAWS = 65536
DECODE_BATCH = 256
INITIAL_SEEDS = (0, 1, 2)
STABILIZATION_SEEDS = tuple(range(3, 30))
MAX_SYSTEM_RETRIES = 2
BOOTSTRAP_REPLICATES = 50000
FINAL_CLASSIFICATIONS = (
    "HIGH_REGIME_PRIOR_CONFIRMED_HEAD_REDUNDANT",
    "HIGH_REGIME_PRIOR_NOT_CONFIRMED",
    "HIGH_REGIME_PRIOR_INCONCLUSIVE",
    "PREDICTOR_NOT_COLLAPSED_TO_HIGH_REGIME",
    "INSUFFICIENT_FRESH_CANDIDATES",
    "INPUT_EVIDENCE_MISMATCH",
    "EXECUTION_OR_PROTOCOL_FAILURE",
)
EXPECTED_NEW_FILES = (
    "analyse/high_regime_prior_model_blind_validation.py",
    "analyse/run_high_regime_prior_model_blind_validation_labels.py",
    "analyse/analyse_high_regime_prior_model_blind_validation.py",
    "tests/test_high_regime_prior_model_blind_validation.py",
)

utc_now = v2.utc_now
sha256_file = v2.sha256_file
stable_hash = v2.stable_hash
load_json = v2.load_json
dump_json = v2.dump_json
dump_csv = v2.dump_csv
dump_text = v2.dump_text
file_identity = v2.file_identity
run_text = v2.run_text
strict_test_key_hits = v2.strict_test_key_hits
joint_config_fingerprint = v2.joint_config_fingerprint


def derived_seed(*namespace: Any) -> int:
    return int(stable_hash([MASTER_SEED, *namespace])[:8], 16) & 0x7FFFFFFF


def _head_blob_identity(relative: str) -> tuple[bool, str | None]:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative], cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not tracked:
        return False, None
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return (result.returncode == 0,
            hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None)


def source_dependency_manifest() -> dict[str, Any]:
    entries = [REPO_ROOT / relative for relative in EXPECTED_NEW_FILES]
    missing = [str(path) for path in entries if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing experiment source files: {missing}")
    closure = v2.discover_local_source_closure(entries)
    rows = []
    for path in closure:
        relative = str(path.relative_to(REPO_ROOT))
        tracked, head_hash = _head_blob_identity(relative)
        current_hash = sha256_file(path)
        recoverable = bool(tracked and head_hash == current_hash)
        rows.append({
            **file_identity(path), "relative_path": relative,
            "tracked_by_git": tracked, "head_blob_sha256": head_hash,
            "recoverable_from_clean_head": recoverable,
            "requires_delivery_snapshot": not recoverable,
        })
    return {
        "created_at": utc_now(), "entry_files": list(EXPECTED_NEW_FILES),
        "dependency_count": len(rows),
        "snapshot_required_count": sum(row["requires_delivery_snapshot"] for row in rows),
        "files": rows,
    }


def verify_v2_evidence() -> dict[str, Any]:
    root = REPO_ROOT / V2_RELATIVE
    required = [
        "final_classification.json", "pre_label_immutable_manifest.json",
        "pre_label_firewall_audit.json", "artifact_manifest.json",
        "frozen_candidate_pools.json", "frozen_head_predictions.json",
        "frozen_strategy_selections.json", "exposure_registry.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return {"passed": False, "reason": "missing v2 artifacts", "missing": missing}
    final = load_json(root / "final_classification.json")
    frozen = load_json(root / "pre_label_immutable_manifest.json")
    frozen_bad = []
    for row in frozen["files"]:
        path = root / row["relative_path"]
        actual = sha256_file(path) if path.is_file() else None
        size = path.stat().st_size if path.is_file() else None
        if actual != row["sha256"] or size != row["size_bytes"]:
            frozen_bad.append(row["relative_path"])
    recomputed_root = stable_hash([
        (row["relative_path"], row["sha256"]) for row in frozen["files"]
    ])
    artifact = load_json(root / "artifact_manifest.json")
    artifact_by_path = {row["relative_path"]: row for row in artifact["files"]}
    artifact_targets = ["final_classification.json", "pre_label_immutable_manifest.json"]
    artifact_bad = []
    for relative in artifact_targets:
        entry = artifact_by_path.get(relative)
        path = root / relative
        if entry is None or sha256_file(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            artifact_bad.append(relative)
    passed = (
        final.get("classification") == "HP_FILTERING_ONLY"
        and not frozen_bad and not artifact_bad
        and recomputed_root == frozen.get("manifest_root_hash")
        and load_json(root / "pre_label_firewall_audit.json").get("passed") is True
    )
    return {
        "passed": passed,
        "classification": final.get("classification"),
        "final_classification": file_identity(root / "final_classification.json"),
        "pre_label_manifest": file_identity(root / "pre_label_immutable_manifest.json"),
        "stored_pre_label_root_hash": frozen.get("manifest_root_hash"),
        "recomputed_pre_label_root_hash": recomputed_root,
        "pre_label_file_count": len(frozen["files"]),
        "pre_label_mismatches": frozen_bad,
        "artifact_manifest_mismatches": artifact_bad,
    }


def recompute_v2_hp_collapse() -> dict[str, Any]:
    root = REPO_ROOT / V2_RELATIVE
    candidates = load_json(root / "frozen_candidate_pools.json")
    predictions = load_json(root / "frozen_head_predictions.json")
    selections = load_json(root / "frozen_strategy_selections.json")
    by_config = {str(row["config_id"]): row for row in candidates}
    means = {str(row["config_id"]): float(row["head_prediction_mean"]) for row in predictions}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row["architecture_fingerprint"])].append(row)
    argmax_rows = []
    tie_count = 0
    for architecture_fp, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda row: (-means[str(row["config_id"])], str(row["config_id"])))
        best = ranked[0]
        if len(ranked) > 1 and means[str(ranked[0]["config_id"])] == means[str(ranked[1]["config_id"])]:
            tie_count += 1
        argmax_rows.append({
            "architecture_fingerprint": architecture_fp,
            "config_id": best["config_id"], "hp_regime": best["hp_regime"],
            "head_prediction_mean": means[str(best["config_id"])],
        })
    argmax_counts = Counter(row["hp_regime"] for row in argmax_rows)
    b_rows = [row for row in selections if row["strategy"] == "HEAD_GLOBAL_TOP4"]
    b_regimes = Counter(str(row["hp_regime"]) for row in b_rows)
    expected = (
        len(candidates) == 1536 * 4 and len(grouped) == 1536
        and argmax_counts == Counter({"hp_high": 1535, "hp_mid_high": 1})
        and len(b_rows) == 96
        and b_regimes == Counter({"hp_high": 87, "hp_mid_high": 9})
        and tie_count == 0
        and all(str(row["config_id"]) in by_config for row in b_rows)
    )
    return {
        "passed": expected,
        "candidate_configuration_count": len(candidates),
        "architecture_count": len(grouped),
        "per_architecture_argmax_regime_counts": dict(sorted(argmax_counts.items())),
        "b_slot_regime_counts": dict(sorted(b_regimes.items())),
        "b_slot_count": len(b_rows), "head_argmax_tie_count": tie_count,
        "expected_per_architecture": {"hp_high": 1535, "hp_mid_high": 1},
        "expected_b_slots": {"hp_high": 87, "hp_mid_high": 9},
        "argmax_rows": argmax_rows,
        "source_files": {
            name: file_identity(root / name) for name in (
                "frozen_candidate_pools.json", "frozen_head_predictions.json",
                "frozen_strategy_selections.json",
            )
        },
    }


def protected_paths() -> list[Path]:
    paths = set(v2.protected_paths())
    v2_root = REPO_ROOT / V2_RELATIVE
    paths.update(path.resolve() for path in v2_root.rglob("*") if path.is_file())
    for relative in (
        "analyse/original_head_confirmatory_attribution_v2.py",
        "analyse/run_original_head_confirmatory_attribution_v2_labels.py",
        "analyse/analyse_original_head_confirmatory_attribution_v2.py",
        "tests/test_original_head_confirmatory_attribution_v2.py",
    ):
        path = REPO_ROOT / relative
        if path.is_file():
            paths.add(path.resolve())
    return sorted(paths)


def write_failure(output: Path, classification: str, evidence: Mapping[str, Any]) -> None:
    if classification not in FINAL_CLASSIFICATIONS:
        raise ValueError(classification)
    dump_json(output / "final_classification.json", {
        "format_version": FORMAT_VERSION, "classification": classification,
        "created_at": utc_now(), "evidence": dict(evidence),
        "label_training_started": False, "next_stage_authorized": False,
        "test_evaluation_authorized": False,
    })


def build_exposure_registry(output: Path) -> tuple[set[str], list[dict[str, Any]], dict[str, Any]]:
    registry, _, metadata = v2.build_exposure_registry(output)
    v2_candidates_path = REPO_ROOT / V2_RELATIVE / "frozen_candidate_pools.json"
    v2_rows = load_json(v2_candidates_path)
    v2_fps = {str(row["architecture_fingerprint"]) for row in v2_rows}
    before = len(registry)
    registry.update(v2_fps)
    metadata = dict(metadata)
    metadata["explicit_v2_full_candidate_pool"] = {
        **file_identity(v2_candidates_path),
        "architecture_count": len(v2_fps),
        "already_covered_count": len(v2_fps) - (len(registry) - before),
        "coverage_after_union": len(v2_fps & registry),
        "resolved_by": [str(v2_candidates_path.resolve())],
        "required_full_pool_coverage": True,
    }
    metadata["audit"] = {
        **metadata["audit"],
        "v2_candidate_architecture_count": len(v2_fps),
        "v2_candidate_fingerprints_covered": len(v2_fps & registry),
        "union_count_with_v2": len(registry),
        "passed": metadata["audit"]["passed"] and len(v2_fps & registry) == 1536,
    }
    return registry, [{"architecture_fingerprint": fp} for fp in sorted(registry)], metadata


def _decode_global_reservoir(model: Any, excluded: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed("global_fresh_reservoir_draws"))
    unique: dict[str, dict[str, Any]] = {}
    invalid = Counter()
    excluded_count = duplicate_count = processed = 0
    draw_target = INITIAL_DRAWS
    schedule = []
    while True:
        while processed < draw_target:
            size = min(DECODE_BATCH, draw_target - processed)
            z = torch.randn(size, 12, generator=generator)
            with torch.no_grad():
                graphs = model.arch_vae.decode(z, stochastic=False)
            for offset, graph in enumerate(graphs):
                try:
                    config = v1.config_from_igraph(model, graph)
                    fingerprint = v1.architecture_fingerprint(config)
                except (TypeError, ValueError, KeyError, IndexError) as error:
                    invalid[type(error).__name__] += 1
                    continue
                valid, reason = v1.validate_static_config(config)
                if not valid:
                    invalid[str(reason)] += 1
                    continue
                if fingerprint in excluded:
                    excluded_count += 1
                    continue
                if fingerprint in unique:
                    duplicate_count += 1
                    continue
                unique[fingerprint] = {
                    "architecture_fingerprint": fingerprint,
                    "operations": list(config["operations"]),
                    "edges": [list(map(int, edge)) for edge in config["edges"]],
                    "effective_layers": int(config["effective_layers"]),
                    "generation_index": processed + offset,
                    "architecture_source": "frozen_vae_standard_normal_deterministic_greedy",
                }
            processed += size
        schedule.append({"cumulative_draws": processed, "fresh_unique_count": len(unique)})
        if len(unique) >= ARCHITECTURE_COUNT or draw_target >= MAX_DRAWS:
            break
        draw_target = min(MAX_DRAWS, draw_target * 2)
    ordered = sorted(unique.values(), key=lambda row: (row["generation_index"], row["architecture_fingerprint"]))
    rng = np.random.default_rng(derived_seed("global_reservoir_frozen_permutation"))
    permutation = rng.permutation(len(ordered)) if ordered else np.asarray([], dtype=int)
    selected = [dict(ordered[int(index)]) for index in permutation[:ARCHITECTURE_COUNT]]
    for reservoir_order, row in enumerate(selected):
        row.update({
            "reservoir_order": reservoir_order,
            "pool_id": reservoir_order // (BATCHES_PER_POOL * ARCHITECTURES_PER_BATCH),
            "batch_id": (reservoir_order // ARCHITECTURES_PER_BATCH) % BATCHES_PER_POOL,
            "architecture_slot": reservoir_order % ARCHITECTURES_PER_BATCH + 1,
        })
    legal_count = processed - sum(invalid.values())
    audit = {
        "passed": len(selected) == ARCHITECTURE_COUNT,
        "draw_seed": derived_seed("global_fresh_reservoir_draws"),
        "permutation_seed": derived_seed("global_reservoir_frozen_permutation"),
        "global_reservoir_constructed_before_pool_partition": True,
        "pool_sequential_exhaustion_used": False,
        "draw_schedule": schedule, "draws_used": processed,
        "fresh_unique_before_permutation": len(unique),
        "selected_architecture_count": len(selected),
        "legal_rate": legal_count / processed if processed else 0.0,
        "fresh_unique_rate": len(unique) / processed if processed else 0.0,
        "excluded_reject_count": excluded_count,
        "duplicate_reject_count": duplicate_count,
        "invalid_counts": dict(sorted(invalid.items())),
        "permutation_sha256": stable_hash([int(value) for value in permutation.tolist()]),
    }
    return selected, audit


def generate_candidates(exposure: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = v1.load_joint_vae(BASE_CHECKPOINT, "global4", device="cpu")
    model.arch_vae.eval()
    architectures, audit = _decode_global_reservoir(model, exposure)
    candidates = []
    for architecture in architectures:
        rng = np.random.default_rng(derived_seed("hp_regimes", architecture["architecture_fingerprint"]))
        for regime in REGIME_ORDER:
            low, high = HP_REGIMES[regime]
            raw_hp = rng.uniform(low, high, size=4).astype(np.float64).tolist()
            hp_fp = v1.hp_fingerprint(raw_hp)
            row = {**architecture, "hp_regime": regime, "raw_hp_vector": raw_hp, "hp_fingerprint": hp_fp}
            row.update(v1.realize_global4_hp(row))
            row["joint_config_fingerprint"] = joint_config_fingerprint(row["architecture_fingerprint"], hp_fp)
            row["config_id"] = (
                f"hr_p{row['pool_id']}_b{row['batch_id']}_s{row['architecture_slot']}_"
                f"{row['architecture_fingerprint'][:12]}_{regime}_{hp_fp[:8]}"
            )
            candidates.append(row)
    return architectures, candidates, audit


R_SELECTOR_SCHEMA = {
    "pool_id", "batch_id", "architecture_slot", "architecture_fingerprint", "regime_config_ids",
}


def balanced_random_hp_assignment(rows: Sequence[Mapping[str, Any]], pool_id: int, batch_id: int) -> dict[int, str]:
    v2._validate_exact_schema(rows, R_SELECTOR_SCHEMA, "balanced-random HP selector")
    if len(rows) != ARCHITECTURES_PER_BATCH:
        raise ValueError("balanced-random selector requires four architecture slots")
    ordered = sorted(rows, key=lambda row: int(row["architecture_slot"]))
    if [int(row["architecture_slot"]) for row in ordered] != [1, 2, 3, 4]:
        raise ValueError("architecture slots must be exactly 1..4")
    for row in ordered:
        if set(row["regime_config_ids"]) != set(REGIME_ORDER):
            raise ValueError("each architecture must expose exactly four frozen regimes")
    rng = np.random.default_rng(derived_seed("balanced_random_hp_permutation", pool_id, batch_id))
    regimes = [REGIME_ORDER[int(index)] for index in rng.permutation(4)]
    return {int(row["architecture_slot"]): regime for row, regime in zip(ordered, regimes)}


def freeze_formal_strategies(candidates: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_batch_arch: dict[tuple[int, int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in candidates:
        key = (int(row["pool_id"]), int(row["batch_id"]), str(row["architecture_fingerprint"]))
        by_batch_arch[key][str(row["hp_regime"])] = row
    selections = []
    batch_audits = []
    for pool_id in range(POOL_COUNT):
        for batch_id in range(BATCHES_PER_POOL):
            groups = [
                regimes for (pool, batch, _), regimes in by_batch_arch.items()
                if pool == pool_id and batch == batch_id
            ]
            groups = sorted(groups, key=lambda regimes: int(next(iter(regimes.values()))["architecture_slot"]))
            narrow = [{
                "pool_id": pool_id, "batch_id": batch_id,
                "architecture_slot": int(next(iter(regimes.values()))["architecture_slot"]),
                "architecture_fingerprint": str(next(iter(regimes.values()))["architecture_fingerprint"]),
                "regime_config_ids": {name: str(config["config_id"]) for name, config in regimes.items()},
            } for regimes in groups]
            assignment = balanced_random_hp_assignment(narrow, pool_id, batch_id)
            union_fps = set()
            shared = 0
            for regimes in groups:
                slot = int(next(iter(regimes.values()))["architecture_slot"])
                f = dict(regimes["hp_high"])
                r_regime = assignment[slot]
                r = dict(regimes[r_regime])
                for strategy, selected in (("HIGH_REGIME_PRIOR", f), ("BALANCED_RANDOM_HP", r)):
                    selections.append({
                        **selected, "strategy": strategy, "budget_order": slot,
                        "r_assigned_regime": r_regime if strategy == "BALANCED_RANDOM_HP" else None,
                        "selector_used_head": False,
                    })
                union_fps.update((f["joint_config_fingerprint"], r["joint_config_fingerprint"]))
                if r_regime == "hp_high":
                    shared += int(f["joint_config_fingerprint"] == r["joint_config_fingerprint"])
            r_rows = [row for row in selections if row["pool_id"] == pool_id and row["batch_id"] == batch_id and row["strategy"] == "BALANCED_RANDOM_HP"]
            batch_audits.append({
                "pool_id": pool_id, "batch_id": batch_id,
                "F_slot_count": 4, "R_slot_count": len(r_rows),
                "R_regime_counts": dict(Counter(row["hp_regime"] for row in r_rows)),
                "shared_high_configuration_count": shared,
                "unique_union_configuration_count": len(union_fps),
                "passed": len(r_rows) == 4 and Counter(row["hp_regime"] for row in r_rows) == Counter(REGIME_ORDER)
                and shared == 1 and len(union_fps) == 7,
            })
    audit = {
        "passed": len(selections) == ARCHITECTURE_COUNT * 2 and all(row["passed"] for row in batch_audits),
        "selection_slot_count": len(selections), "batch_count": len(batch_audits),
        "batch_audits": batch_audits,
        "F_selector_received_head": False, "R_selector_received_head": False,
        "R_selector_received_validation_or_trajectory": False,
    }
    return selections, audit


def make_union_tasks(selections: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selections:
        grouped[(int(row["pool_id"]), str(row["joint_config_fingerprint"]))].append(row)
    union = []
    tasks = []
    for (pool_id, joint_fp), rows in sorted(grouped.items()):
        representative = dict(sorted(rows, key=lambda row: (row["config_id"], row["strategy"]))[0])
        selected_by = sorted([{
            "strategy": row["strategy"], "batch_id": int(row["batch_id"]),
            "budget_order": int(row["budget_order"]),
        } for row in rows], key=lambda row: (row["strategy"], row["batch_id"], row["budget_order"]))
        config = {
            **{key: representative[key] for key in v2.CANDIDATE_HP_SCHEMA},
            "pool_id": pool_id, "batch_id": int(representative["batch_id"]),
            "architecture_slot": int(representative["architecture_slot"]),
            "config_id": str(representative["config_id"]),
            "joint_config_fingerprint": joint_fp,
            "assigned_gpu": 0 if pool_id < POOL_COUNT // 2 else 1,
            "selected_by": selected_by, "task_stage": "initial",
        }
        union.append(config)
        for seed in INITIAL_SEEDS:
            tasks.append({
                **config, "seed": seed,
                "task_key": f"initial::{pool_id}::{joint_fp}::{seed}",
            })
    return union, tasks


def verify_freeze(output: Path) -> dict[str, Any]:
    manifest = load_json(output / "pre_label_immutable_manifest.json")
    mismatches = []
    for row in manifest["files"]:
        path = output / row["relative_path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != row["sha256"]:
            mismatches.append({"relative_path": row["relative_path"], "expected": row["sha256"], "actual": actual})
    root_hash = stable_hash([(row["relative_path"], row["sha256"]) for row in manifest["files"]])
    dependency = load_json(output / "source_dependency_manifest.json")
    source_mismatches = []
    for row in dependency["files"]:
        path = Path(row["path"])
        actual = sha256_file(path) if path.is_file() else None
        if actual != row["sha256"]:
            source_mismatches.append({"path": row["path"], "expected": row["sha256"], "actual": actual})
    return {
        "passed": not mismatches and not source_mismatches and root_hash == manifest["manifest_root_hash"],
        "mismatches": mismatches, "dependency_mismatches": source_mismatches,
        "stored_root_hash": manifest["manifest_root_hash"], "recomputed_root_hash": root_hash,
    }


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    v2_evidence = verify_v2_evidence()
    dump_json(output / "v2_input_identity_audit.json", v2_evidence)
    collapse = recompute_v2_hp_collapse() if v2_evidence["passed"] else {"passed": False}
    dump_json(output / "v2_head_hp_collapse_audit.json", collapse)
    if not v2_evidence["passed"] or not collapse["passed"]:
        write_failure(output, "INPUT_EVIDENCE_MISMATCH", {"v2_evidence": v2_evidence, "collapse": collapse})
        return
    compute_apps = run_text([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits",
    ])
    if compute_apps.strip():
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "GPU occupied before preparation", "processes": compute_apps.splitlines()})
        return

    dependency = source_dependency_manifest()
    dump_json(output / "source_dependency_manifest.json", dependency)
    dump_json(output / "source_identity.json", v2.source_identity(output, dependency))
    dump_json(output / "protected_hashes_before.json", {
        str(path): file_identity(path) for path in protected_paths()
    })
    v2_root = REPO_ROOT / V2_RELATIVE
    v2_prereg = load_json(v2_root / "preregistration.json")
    strict_root = REPO_ROOT / STRICT_RELATIVE
    head_paths = sorted((strict_root / "checkpoints/full/group_repeated").glob("group_*/seed_*/frozen_head/checkpoint.pth"))
    if len(head_paths) != 150:
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "Original Head member count", "found": len(head_paths)})
        return
    dump_json(output / "input_manifest.json", {
        "format_version": FORMAT_VERSION, "base_vae": file_identity(BASE_CHECKPOINT),
        "head_members": [file_identity(path) for path in head_paths],
        "head_usage": "pre-label HP-collapse diagnosis only; never architecture or F/R selection",
        "dataset_content_fingerprint": v2_prereg["training"]["dataset_content_fingerprint"],
        "split_fingerprint": v2_prereg["training"]["split_fingerprint"],
        "v2_files": {name: file_identity(v2_root / name) for name in (
            "final_classification.json", "pre_label_immutable_manifest.json",
            "frozen_candidate_pools.json", "frozen_head_predictions.json",
            "frozen_strategy_selections.json", "exposure_registry.json",
        )},
        "source_dependency_manifest": file_identity(output / "source_dependency_manifest.json"),
    })

    exposure, exposure_rows, exposure_metadata = build_exposure_registry(output)
    dump_json(output / "exposure_registry.json", exposure_rows)
    dump_json(output / "exposure_source_inventory.json", exposure_metadata)
    architectures, candidates, generation_audit = generate_candidates(exposure)
    dump_json(output / "global_reservoir.json", architectures)
    dump_json(output / "candidate_generation_audit.json", generation_audit)
    if len(architectures) != ARCHITECTURE_COUNT:
        write_failure(output, "INSUFFICIENT_FRESH_CANDIDATES", {"generation_audit": generation_audit})
        return
    architecture_fps = {row["architecture_fingerprint"] for row in architectures}
    batch_counts = Counter((row["pool_id"], row["batch_id"]) for row in architectures)
    candidate_audit = {
        "passed": len(architecture_fps) == ARCHITECTURE_COUNT and not (architecture_fps & exposure)
        and all(count == 4 for count in batch_counts.values()) and len(batch_counts) == 48
        and len(candidates) == ARCHITECTURE_COUNT * 4 and not strict_test_key_hits(candidates),
        "architecture_count": len(architecture_fps), "configuration_count": len(candidates),
        "exposure_overlap_count": len(architecture_fps & exposure),
        "pool_count": len({row["pool_id"] for row in architectures}),
        "batch_counts": {f"{p}:{b}": count for (p, b), count in sorted(batch_counts.items())},
        "cross_pool_batch_architecture_overlap_count": len(architectures) - len(architecture_fps),
        "replicate_claim": "24 non-overlapping replicate blocks; not asserted IID",
    }
    dump_json(output / "candidate_pool_audit.json", candidate_audit)
    if not candidate_audit["passed"]:
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "candidate audit failed", "audit": candidate_audit})
        return
    dump_json(output / "frozen_candidate_configs.json", candidates)
    dump_csv(output / "frozen_candidate_configs.csv", candidates)

    matrix, members, checkpoints = v1.head_member_predictions(candidates, strict_root)
    if matrix.shape != (150, len(candidates)):
        raise RuntimeError(f"unexpected Head prediction matrix: {matrix.shape}")
    prediction_rows = []
    for index, candidate in enumerate(candidates):
        values = matrix[:, index].astype(float).tolist()
        prediction_rows.append({
            "pool_id": candidate["pool_id"], "batch_id": candidate["batch_id"],
            "architecture_slot": candidate["architecture_slot"],
            "architecture_fingerprint": candidate["architecture_fingerprint"],
            "config_id": candidate["config_id"], "hp_regime": candidate["hp_regime"],
            "hp_fingerprint": candidate["hp_fingerprint"],
            "joint_config_fingerprint": candidate["joint_config_fingerprint"],
            "head_prediction_mean": float(np.mean(values)),
            "head_prediction_std_descriptive_only": float(np.std(values, ddof=1)),
            "head_member_predictions": values, "member_count": 150,
            "used_for_architecture_generation_or_order": False,
            "used_by_F_or_R_selector": False,
        })
    dump_json(output / "frozen_head_hp_diagnostic_predictions.json", prediction_rows)
    dump_csv(output / "frozen_head_hp_diagnostic_predictions.csv", prediction_rows)
    dump_json(output / "frozen_head_member_identity.json", {"members": members, "checkpoints": checkpoints})
    means = {row["config_id"]: row["head_prediction_mean"] for row in prediction_rows}
    by_arch: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_arch[str(row["architecture_fingerprint"])].append(row)
    choice_rows = []
    for architecture_fp, rows in sorted(by_arch.items()):
        ranked = sorted(rows, key=lambda row: (-float(means[row["config_id"]]), str(row["config_id"])))
        h = ranked[0]
        f = next(row for row in rows if row["hp_regime"] == "hp_high")
        choice_rows.append({
            "architecture_fingerprint": architecture_fp, "pool_id": h["pool_id"],
            "batch_id": h["batch_id"], "architecture_slot": h["architecture_slot"],
            "H_choice_config_id": h["config_id"], "H_choice_regime": h["hp_regime"],
            "F_choice_config_id": f["config_id"], "F_choice_regime": "hp_high",
            "H_equals_F": h["joint_config_fingerprint"] == f["joint_config_fingerprint"],
            "tie_break": "descending mean then config_id",
        })
    agreement = sum(row["H_equals_F"] for row in choice_rows)
    collapse_audit = {
        "passed": agreement >= 191, "classification_if_passed": "HEAD_HP_COLLAPSES_TO_HIGH_REGIME",
        "agreement_count": agreement, "denominator": ARCHITECTURE_COUNT,
        "agreement_rate": agreement / ARCHITECTURE_COUNT,
        "minimum_required": 191,
        "H_choice_regime_counts": dict(Counter(row["H_choice_regime"] for row in choice_rows)),
        "head_used_for_formal_selection": False, "rows": choice_rows,
    }
    dump_json(output / "new_pool_head_high_regime_collapse_audit.json", collapse_audit)
    if not collapse_audit["passed"]:
        write_failure(output, "PREDICTOR_NOT_COLLAPSED_TO_HIGH_REGIME", collapse_audit)
        return

    selections, selection_audit = freeze_formal_strategies(candidates)
    dump_json(output / "frozen_strategy_selections.json", selections)
    dump_csv(output / "frozen_strategy_selections.csv", selections)
    dump_json(output / "strategy_selection_audit.json", selection_audit)
    union, tasks = make_union_tasks(selections)
    dump_json(output / "selected_union_configs.json", union)
    dump_csv(output / "selected_union_configs.csv", union)
    dump_json(output / "selected_union_tasks.json", tasks)
    dump_csv(output / "selected_union_tasks.csv", tasks)
    exact_counts = len(union) == 336 and len(tasks) == 1008
    gpu_task_counts = Counter(int(row["assigned_gpu"]) for row in tasks)

    preregistration = {
        "format_version": FORMAT_VERSION, "created_at": utc_now(),
        "scientific_name": "High-regime HP prior with model-blind architecture validation search, global4 v1",
        "method_name": "model-blind architecture search with a high-regime HP prior",
        "method_is_bo": False, "predictor_guided_architecture_search": False,
        "master_seed": MASTER_SEED, "pool_count": POOL_COUNT,
        "batches_per_pool": BATCHES_PER_POOL, "architectures_per_batch": ARCHITECTURES_PER_BATCH,
        "architecture_count": ARCHITECTURE_COUNT, "hp_regimes": HP_REGIMES,
        "initial_seeds": list(INITIAL_SEEDS), "stabilization_seeds": list(STABILIZATION_SEEDS),
        "strategies": {
            "F": "each architecture's own pre-generated hp_high configuration",
            "R": "same four architecture slots; blind permutation using each regime exactly once per batch",
            "shared_high": "one exact architecture+hp configuration per batch; one training result shared",
        },
        "training": {
            "dataset": "Cora", "split": "public", "hp_mode": "global4",
            "dataset_content_fingerprint": v2_prereg["training"]["dataset_content_fingerprint"],
            "split_fingerprint": v2_prereg["training"]["split_fingerprint"],
            "max_epochs": 300, "patience": 80, "track_test": False,
            "return_trajectory": True, "maximum_system_retries": MAX_SYSTEM_RETRIES,
            "shard_to_gpu_uuid": v2_prereg["training"]["shard_to_gpu_uuid"],
        },
        "statistics": {
            "primary_unit": "24 equally weighted pools",
            "batch_value": "best-of-4 after three-seed configuration means",
            "pool_value": "arithmetic mean of two frozen batch values; never take the better batch",
            "primary_comparison": "F_minus_R",
            "hierarchical_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "hierarchical_bootstrap_seed": derived_seed("hierarchical_bootstrap"),
            "pool_only_bootstrap_seed": derived_seed("pool_only_bootstrap"),
            "hierarchy": "shared pool, matched batch, and shared seed-ID indices; recompute best-of-4 after seed resampling",
            "gate": ["mean F-R >= 0.006", "hierarchical CI95 lower > 0", "F>R in at least 15/24 pools"],
            "sign_test": "exact two-sided binomial after excluding ties; sensitivity only",
        },
        "first_stage_classifications": list(FINAL_CLASSIFICATIONS),
        "stabilization": {
            "eligible_first_stage": list(FINAL_CLASSIFICATIONS[:3]),
            "top_k_per_strategy": 8,
            "top8_source": "initial seeds 0,1,2 only; descending mean then joint fingerprint",
            "union_deduplicated": True, "additional_seeds": list(STABILIZATION_SEEDS),
            "final_candidate_rule": "within frozen strategy top8, highest all-30-seed validation mean; tie by joint fingerprint",
            "maximum_union_configs": 16, "maximum_additional_tasks": 432,
            "total_task_hard_cap": 1440,
        },
        "forbidden": [
            "model/scaler training", "Head architecture selection", "mutation/evolution/population/tournament",
            "GP/BO/acquisition/DPP", "Head uncertainty", "latent Euclidean distance", "ASHA",
            "test metric", "post-label strategy or gate changes",
        ],
        "source_code_hashes": {str((REPO_ROOT / path).resolve()): sha256_file(REPO_ROOT / path) for path in EXPECTED_NEW_FILES},
        "source_dependency_manifest_sha256": sha256_file(output / "source_dependency_manifest.json"),
    }
    dump_json(output / "preregistration.json", preregistration)
    firewall = {
        "passed": candidate_audit["passed"] and collapse_audit["passed"] and selection_audit["passed"]
        and exact_counts and gpu_task_counts == Counter({0: 504, 1: 504})
        and not strict_test_key_hits(selections) and not strict_test_key_hits(tasks)
        and len({row["task_key"] for row in tasks}) == 1008,
        "historical_exposure_overlap_count": candidate_audit["exposure_overlap_count"],
        "architecture_overlap_across_pool_or_batch": 0,
        "pool_count": 24, "batches_per_pool": 2, "architectures_per_batch": 4,
        "F_slots": 192, "R_slots": 192, "unique_union_configurations": len(union),
        "initial_task_count": len(tasks), "expected_initial_task_count": 1008,
        "gpu_task_counts": dict(gpu_task_counts),
        "H_used_only_for_prelabel_collapse_audit": True,
        "H_used_by_architecture_or_formal_selector": False,
        "R_selector_schema": sorted(R_SELECTOR_SCHEMA),
        "R_selector_received_head_or_label_or_trajectory": False,
        "head_std_used": False, "H_or_refit_or_GP_used": False,
        "mu_euclidean_distance_used": False, "track_test": False,
        "selection_test_field_hits": strict_test_key_hits(selections),
        "task_test_field_hits": strict_test_key_hits(tasks), "labels_connected": False,
    }
    dump_json(output / "pre_label_firewall_audit.json", firewall)
    if not firewall["passed"]:
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "pre-label firewall failed", "firewall": firewall})
        return

    frozen_names = [
        "v2_input_identity_audit.json", "v2_head_hp_collapse_audit.json",
        "source_dependency_manifest.json", "source_identity.json", "environment_lock.txt",
        "tracked_diff.binary.patch", "protected_hashes_before.json", "input_manifest.json",
        "exposure_registry.json", "exposure_source_inventory.json", "global_reservoir.json",
        "candidate_generation_audit.json", "candidate_pool_audit.json",
        "frozen_candidate_configs.json", "frozen_candidate_configs.csv",
        "frozen_head_hp_diagnostic_predictions.json", "frozen_head_hp_diagnostic_predictions.csv",
        "frozen_head_member_identity.json", "new_pool_head_high_regime_collapse_audit.json",
        "frozen_strategy_selections.json", "frozen_strategy_selections.csv",
        "strategy_selection_audit.json", "selected_union_configs.json", "selected_union_configs.csv",
        "selected_union_tasks.json", "selected_union_tasks.csv", "preregistration.json",
        "pre_label_firewall_audit.json",
    ]
    entries = [{**file_identity(output / name), "relative_path": name} for name in frozen_names]
    root_hash = stable_hash([(row["relative_path"], row["sha256"]) for row in entries])
    dump_json(output / "pre_label_immutable_manifest.json", {
        "format_version": FORMAT_VERSION, "created_at": utc_now(), "files": entries,
        "manifest_root_hash": root_hash, "immutable_after_new_label_connection": True,
    })
    print(json.dumps({
        "status": "PRE_LABEL_FREEZE_COMPLETE", "v2_collapse": collapse["passed"],
        "new_head_high_agreement": agreement, "architectures": len(architectures),
        "candidate_configurations": len(candidates), "union_configurations": len(union),
        "tasks": len(tasks), "gpu_task_counts": dict(gpu_task_counts),
        "manifest_root_hash": root_hash,
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-freeze"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output.resolve())
    else:
        audit = verify_freeze(args.output.resolve())
        print(json.dumps(audit, indent=2, sort_keys=True))
        if not audit["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
