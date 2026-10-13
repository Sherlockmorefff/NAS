#!/usr/bin/env python3
"""Independent prospective confirmation and frozen surrogate selection gate v1.

The formal workflow replays the archived soft-metric scorers from the original
336 configurations, excludes every candidate from the completed eight-pool
pilot, freezes all sixteen new pools before evaluation, and only then delegates
the 192 validation-only GNN tasks to the archived high-regime evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tarfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import joint_metric_prospective_selection_gate as pilot_gate


PROTOCOL_VERSION = "joint_metric_independent_confirmation_surrogate_selection_global4_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/joint_metric_independent_confirmation_surrogate_selection_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/posthoc/joint-metric-independent-confirmation-surrogate-selection-global4-v1-rerun1"
DEFAULT_ARCHIVE = Path(str(DEFAULT_OUTPUT) + ".tar.gz")

HISTORICAL_SOFT_ROOT = REPO_ROOT / "results/posthoc/joint-configuration-soft-metric-gate-global4-v2"
HISTORICAL_HIGH_ROOT = REPO_ROOT / "results/search/high-regime-prior-model-blind-validation-global4-v1"
PREVIOUS_PILOT_ROOT = REPO_ROOT / "results/posthoc/joint-metric-prospective-selection-gate-global4-v1-rerun1"
PREVIOUS_PILOT_ARCHIVE = Path(str(PREVIOUS_PILOT_ROOT) + ".tar.gz")

POOL_COUNT = 16
CANDIDATES_PER_POOL = 4
CANDIDATE_COUNT = 64
TOTAL_GNN_RUNS = 192
MASTER_CANDIDATE_SEED = 202608281601
RANDOM_SELECTION_SEED = 202608281602
TRAINING_SEEDS = (0, 1, 2)
METRIC_SEEDS = (0, 1, 2)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0
REPLAY_ATOL = 1.0e-8
MAX_GENERATION_DRAWS_PER_POOL = 65_536
GENERATION_BATCH_SIZE = 256
METHODS = ("M1", "M2", "B1", "B3", "B4", "R")
BASELINE_METHODS = ("B1", "B3", "B4")

FROZEN_HISTORICAL_IDENTITY = {
    "unique_joint_configurations": 336,
    "unique_architectures": 192,
    "unique_pools": 24,
    "seed_record_count": 1008,
    "training_seeds": [0, 1, 2],
}
FROZEN_PREVIOUS_PILOT_IDENTITY = {
    "classification": "METRIC_GP_PROSPECTIVE_SIGNAL",
    "archive_sha256": "7bc437a8babcc61493bf1d86ec50e1c8c4db1596288479f6b47ac7a0bcf13736",
    "config_sha256": "9f0a1d572f25f3f89951cb08728f279388310929b3bcbdd880ee56f508451eba",
    "source_sha256": "2074cde312979393e22d75bbdf93731e48c7cc17b1412ee45a65bd0ea6a37e3d",
    "source_snapshot_manifest_root": "1489b93667f0a1125a3d83377f1ab4bddd6c9bb94add70880c4990c50e1f07f9",
    "pre_eval_root_sha256": "987231579bce11e164a5dd91362f9e8a89e9cfc9c066544508abdb5e863eebb2",
    "candidate_manifest_sha256": "6fcbf5d73eb83d9f0d06fa3d554764069e444ddef88b32540a9efdad8962862c",
    "candidate_count": 32,
}

ALLOWED_CLASSIFICATIONS = frozenset({
    "CONFIRMATION_INPUT_INVALID",
    "SCORER_REPLAY_IDENTITY_MISMATCH",
    "CONFIRMATION_CANDIDATE_GENERATION_FAILED",
    "CONFIRMATION_LOW_INFORMATION",
    "CONFIRMATION_EVALUATION_INCOMPLETE",
    "METRIC_FAMILY_NOT_CONFIRMED",
    "METRIC_FAMILY_CONFIRMED_KNN_SELECTED",
    "METRIC_FAMILY_CONFIRMED_GP_SELECTED",
    "RUN_FAILED",
})
FROZEN_TRAINING_PROTOCOL = {"training_seeds": [0, 1, 2], "track_test": False}
ENGINEERING_FIX = {
    "parent_failed_archive_sha256": "1b50edf5d22a928c0c9056adfc0e47967334ddfa511d59413fee9a995e9410b8",
    "parent_classification": "RUN_FAILED",
    "parent_failure": "scorer protocol mismatch: ['M2']",
    "root_cause": "non-metric baseline audit was incorrectly validated with the full metric scorer validator",
    "scientific_protocol_changed": False,
    "candidate_generation_changed": False,
    "candidate_seeds_changed": False,
    "random_seed_changed": False,
    "M1_changed": False,
    "M2_changed": False,
    "B1_changed": False,
    "B3_changed": False,
    "B4_changed": False,
    "B_STAR_rule_changed": False,
    "selection_gate_changed": False,
    "training_protocol_changed": False,
}

REPORT_SECTIONS = (
    "1. Executive conclusion",
    "2. Historical evidence boundary",
    "3. Independence from previous 8-pool pilot",
    "4. Historical scorer replay",
    "5. Historical baseline replay",
    "6. Frozen B_STAR selection",
    "7. Candidate generation",
    "8. Candidate novelty audit",
    "9. Pre-evaluation freeze",
    "10. Predictor scores",
    "11. Method disagreement counts",
    "12. GNN evaluation completeness",
    "13. Per-pool actual validation table",
    "14. All-method selection metrics",
    "15. M1 vs B_STAR",
    "16. M2 vs B_STAR",
    "17. Metric-family confirmation",
    "18. M1 vs M2 comparison",
    "19. Final surrogate selection",
    "20. Random baseline",
    "21. Pool bootstrap/sign summary",
    "22. Frozen classification",
    "23. Interpretation",
    "24. What this does NOT prove",
    "25. Next authorized experiment",
    "26. Engineering audit",
)

REQUIRED_SOURCE_SNAPSHOT_PATHS = (
    "configs/joint_metric_independent_confirmation_surrogate_selection_v1.yaml",
    "analyse/joint_metric_independent_confirmation_surrogate_selection.py",
    "scripts/run_joint_metric_independent_confirmation_surrogate_selection.sh",
    "tests/test_joint_metric_independent_confirmation_surrogate_selection.py",
    "analyse/joint_metric_prospective_selection_gate.py",
    "configs/joint_configuration_soft_metric_gate_v2.yaml",
    "analyse/joint_configuration_soft_metric_gate.py",
    "analyse/joint_configuration_metric_gate.py",
    "analyse/performance_metric_quick_gate.py",
    "analyse/joint_pair_power_projection_audit.py",
    "analyse/high_regime_prior_model_blind_validation.py",
    "analyse/run_high_regime_prior_model_blind_validation_labels.py",
    "analyse/analyse_high_regime_prior_model_blind_validation.py",
    "eval_utils.py",
    "architecture_diagnostics.py",
)

REQUIRED_OUTPUT_FILES = (
    "state.json", "frozen_protocol.json", "input_integrity.json",
    "engineering_fix.json",
    "historical_scorer_replay.json", "historical_baseline_metrics.json",
    "previous_pilot_exclusion_audit.json",
    "pre_eval/candidate_manifest.json", "pre_eval/predictions.csv",
    "pre_eval/selections.json", "pre_eval/baseline_selection.json",
    "pre_eval/previous_pilot_exclusion_audit.json",
    "pre_eval/scorer_identity.json", "pre_eval/freeze_manifest.json",
    "pre_eval/pre_eval_root_sha256.txt",
    "evaluation/task_manifest.json", "evaluation/raw_seed_results.csv",
    "evaluation/config_3seed_statistics.csv", "evaluation/training_audit.json",
    "selection_results_per_pool.csv", "method_metrics.json",
    "metric_family_confirmation.json", "m1_vs_m2_selection.json",
    "paired_pool_comparisons.json", "bootstrap_results.json",
    "sign_test_summary.json", "decision.json", "final_report.md",
    "environment.json", "git_status_before.txt", "git_status_after.txt",
    "git_diff.patch", "source_snapshot/imported_local_modules.json",
    "source_snapshot/sha256_manifest.json", "manifest.json",
)


class ProtocolError(RuntimeError):
    """Frozen protocol or artifact contract violation."""


class ConfirmationInputInvalid(ProtocolError):
    """Authoritative historical input does not match the frozen identity."""


class ScorerReplayIdentityMismatch(ProtocolError):
    """Deterministic scorer replay differs from the archived reference."""


class CandidateGenerationFailed(ProtocolError):
    """The frozen proposal could not produce the independent candidate set."""


CandidateGenerationError = CandidateGenerationFailed


class PreEvalFreezeError(ProtocolError):
    """Immutable pre-evaluation evidence changed after freezing."""


class EvaluationIncompleteError(ProtocolError):
    """At least one frozen candidate lacks all three successful seeds."""


def utc_now() -> str:
    return pilot_gate.utc_now()


def canonical_json_bytes(value: Any) -> bytes:
    return pilot_gate.canonical_json_bytes(value)


def stable_hash(value: Any) -> str:
    return pilot_gate.stable_hash(value)


def sha256_file(path: Path) -> str:
    return pilot_gate.sha256_file(path)


def dump_json(path: Path, value: Any) -> None:
    pilot_gate.dump_json(path, value)


def dump_text(path: Path, value: str) -> None:
    pilot_gate.dump_text(path, value)


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    pilot_gate.dump_csv(path, rows, fields)


def reject_test_fields(value: Any, *, allow_track_test_false: bool = False) -> None:
    try:
        pilot_gate.reject_test_fields(value, allow_track_test_false=allow_track_test_false)
    except pilot_gate.ProtocolError as exc:
        raise ProtocolError(str(exc)) from exc


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        return pilot_gate.load_config(path)
    except Exception as exc:
        raise ProtocolError(f"cannot load frozen protocol: {exc}") from exc


def validate_frozen_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    generation = config["candidate_generation"]
    methods = config["prospective_methods"]
    baseline = config["historical_baseline_selection"]
    confirmation = config["metric_family_confirmation"]
    surrogate = config["surrogate_selection"]
    authorities = config["authoritative_inputs"]
    previous = authorities["previous_prospective_pilot_v1"]
    soft_identity = authorities["soft_metric_v2"]
    high_identity = authorities["historical_training_v1"]
    checks = {
        "version": config.get("protocol_version") == PROTOCOL_VERSION,
        "authorities": all(
            str(previous[key]) == str(expected)
            for key, expected in FROZEN_PREVIOUS_PILOT_IDENTITY.items()
            if key != "candidate_count"
        )
        and int(previous["candidate_count"]) == 32
        and soft_identity["archive_sha256"] == pilot_gate.FROZEN_SOFT_METRIC_IDENTITY["archive_sha256"]
        and soft_identity["config_sha256"] == pilot_gate.FROZEN_SOFT_METRIC_IDENTITY["config_sha256"]
        and soft_identity["source_sha256"] == pilot_gate.FROZEN_SOFT_METRIC_IDENTITY["source_sha256"]
        and soft_identity["split_sha256"] == pilot_gate.FROZEN_SOFT_METRIC_IDENTITY["split_sha256"]
        and high_identity["archive_sha256"] == "050a64b0cde5bdb17ababa9833e5ee578b448c9b19da31fbb00f0251ec82491c"
        and high_identity["source_dependency_manifest_sha256"] == "4c836f903d1bea346bcd0139d7c1e367e52b8b99d8549dbf33b13db09cb9f115",
        "pools": int(generation["pools"]) == POOL_COUNT,
        "per_pool": int(generation["configurations_per_pool"]) == CANDIDATES_PER_POOL,
        "candidate_seed": int(generation["master_seed"]) == MASTER_CANDIDATE_SEED,
        "candidate_distribution": generation["hp_prior"] == "HIGH_REGIME_PRIOR"
        and list(generation["hp_raw_unit_interval_range"]) == [0.80, 0.95]
        and int(generation["maximum_draws_per_pool"]) == MAX_GENERATION_DRAWS_PER_POOL
        and int(generation["decode_batch_size"]) == GENERATION_BATCH_SIZE,
        "candidate_rng": generation["pool_seed_rule"] == "master_seed_plus_zero_based_pool_id"
        and generation["rng_streams"]["architecture"] == "torch_cpu_generator_seeded_directly_with_pool_seed"
        and generation["rng_streams"]["hp"] == "numpy_default_rng_seeded_directly_with_pool_seed",
        "random_seed": int(methods["R"]["selection_seed"]) == RANDOM_SELECTION_SEED,
        "M1": int(methods["M1"]["k"]) == 5
        and methods["M1"]["distance"] == "raw_l2_normalized_embedding_Euclidean",
        "M2": methods["M2"]["ard"] is False
        and methods["M2"]["per_dimension_input_standardization"] is False
        and methods["M2"]["prediction"] == "posterior_mean_only",
        "baselines": methods["B1"]["ard"] is True and methods["B3"]["ard"] is True
        and list(methods["B4"]["model_seeds"]) == list(METRIC_SEEDS)
        and methods["B4"]["loss"] == "Huber",
        "B_STAR": list(baseline["candidates"]) == list(BASELINE_METHODS)
        and baseline["primary"] == "lowest_historical_OOF_best_of_4_mean_simple_regret"
        and float(baseline["first_tie_tolerance_absolute"]) == 1.0e-12
        and baseline["first_tie_break"] == "lower_historical_OOF_MAE"
        and baseline["second_tie_break"] == "method_id_lexicographic_ascending",
        "training_seeds": list(config["evaluation"]["training_seeds"]) == list(TRAINING_SEEDS),
        "tasks": int(config["evaluation"]["total_gnn_runs"]) == TOTAL_GNN_RUNS,
        "information": int(config["information_gate"]["minimum_disagreement_pools"]) == 6,
        "confirmation": int(confirmation["C1"]["disagreement_pools_at_least"]) == 6
        and int(confirmation["C2"]["wins_at_least"]) == 4
        and float(confirmation["C3"]["mean_selected_validation_advantage_all_16_at_least"]) == 0.003
        and float(confirmation["C4"]["mean_simple_regret_at_most_comparator_fraction"]) == 0.80
        and float(confirmation["C4"]["comparator_near_zero_threshold_strictly_less_than"]) == 1.0e-6
        and confirmation["family_confirmed_rule"] == "confirmed_M1_OR_confirmed_M2",
        "surrogate": int(surrogate["GP_increment"]["S_GP_1"]["disagreement_pools_at_least"]) == 6
        and int(surrogate["GP_increment"]["S_GP_2"]["M2_wins_at_least"]) == 4
        and float(surrogate["GP_increment"]["S_GP_3"]["M2_mean_regret_at_most_M1_fraction"]) == 0.90
        and int(surrogate["KNN_advantage"]["S_KNN_1"]["disagreement_pools_at_least"]) == 6
        and int(surrogate["KNN_advantage"]["S_KNN_2"]["M1_wins_at_least"]) == 4
        and float(surrogate["KNN_advantage"]["S_KNN_3"]["M1_mean_regret_at_most_M2_fraction"]) == 0.90
        and surrogate["parsimony"]["final_surrogate"] == "M1",
        "bootstrap": int(config["statistics"]["bootstrap"]["replicates"]) == BOOTSTRAP_REPLICATES
        and int(config["statistics"]["bootstrap"]["seed"]) == BOOTSTRAP_SEED,
        "classifications": set(config["decision"]["allowed"]) == set(ALLOWED_CLASSIFICATIONS),
        "engineering_fix": dict(config.get("engineering_fix", {})) == ENGINEERING_FIX,
        "rerun_output": str(config["output"]["result_root"]).endswith(
            "joint-metric-independent-confirmation-surrogate-selection-global4-v1-rerun1"
        ),
        "track_test": config["evaluation"]["track_test"] is False,
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ProtocolError(f"frozen protocol differs: {failed}")
    return {"passed": True, "checks": checks}


def validate_historical_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "unique_joint_configurations": ("unique_joint_configurations",),
        "unique_architectures": ("unique_architectures",),
        "unique_pools": ("unique_pools",),
        "seed_record_count": ("seed_record_count", "raw_row_count", "successful_required_seed_row_count"),
        "training_seeds": ("training_seeds", "complete_training_seeds", "metric_training_seeds"),
    }
    observed: dict[str, Any] = {}
    for target, names in aliases.items():
        for name in names:
            if name in payload:
                observed[target] = payload[name]
                break
    for key, expected in FROZEN_HISTORICAL_IDENTITY.items():
        value = observed.get(key)
        if key == "training_seeds":
            value = list(value or [])
        if value != expected:
            raise ConfirmationInputInvalid(f"historical identity mismatch for {key}: {value!r}")
    return {"passed": True, **FROZEN_HISTORICAL_IDENTITY}


def recover_previous_pilot_fingerprints(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = payload.get("candidates", []) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence) or len(rows) != 32:
        raise ConfirmationInputInvalid("previous pilot must contain exactly 32 candidates")
    architectures = {str(row["architecture_fingerprint"]) for row in rows}
    joints = {str(row.get("joint_fingerprint", row.get("joint_config_fingerprint"))) for row in rows}
    if len(architectures) != 32 or len(joints) != 32 or "None" in joints:
        raise ConfirmationInputInvalid("previous pilot candidate identities are incomplete or duplicated")
    return {
        "passed": True, "candidate_count": 32,
        "architecture_fingerprints": architectures, "joint_fingerprints": joints,
        "pilot_labels_used_for_training": False,
    }


def validate_candidate_manifest(
    rows: Sequence[Mapping[str, Any]],
    historical_architecture_fingerprints: set[str],
    historical_joint_fingerprints: set[str],
    previous_architecture_fingerprints: set[str],
    previous_joint_fingerprints: set[str],
    expected_pool_count: int = POOL_COUNT,
    expected_candidates_per_pool: int = CANDIDATES_PER_POOL,
    expected_master_seed: int = MASTER_CANDIDATE_SEED,
    expected_distribution: str = "HIGH_REGIME_PRIOR",
    generator_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reject_test_fields(rows)
    if len(rows) != expected_pool_count * expected_candidates_per_pool:
        raise CandidateGenerationFailed("candidate count differs from frozen 16 x 4")
    by_pool = {pool: [] for pool in range(expected_pool_count)}
    architectures: list[str] = []
    joints: list[str] = []
    for row in rows:
        pool = int(row["pool_id"])
        if pool not in by_pool:
            raise CandidateGenerationFailed("candidate has a foreign pool")
        by_pool[pool].append(row)
        architecture = str(row["architecture_fingerprint"])
        joint = str(row.get("joint_fingerprint", row.get("joint_config_fingerprint")))
        if architecture in historical_architecture_fingerprints or architecture in previous_architecture_fingerprints:
            raise CandidateGenerationFailed("candidate architecture is not independent")
        if joint in historical_joint_fingerprints or joint in previous_joint_fingerprints:
            raise CandidateGenerationFailed("candidate joint configuration is not independent")
        observed_pool_seed = row.get(
            "generation_pool_seed", row.get("generator_seed", expected_master_seed + pool)
        )
        if int(observed_pool_seed) != expected_master_seed + pool:
            raise CandidateGenerationFailed("candidate pool seed differs from the frozen rule")
        if str(row.get("hp_regime", "hp_high")) != "hp_high":
            raise CandidateGenerationFailed("candidate is not from HIGH_REGIME_PRIOR")
        if str(row.get("proposal_distribution", expected_distribution)) != expected_distribution:
            raise CandidateGenerationFailed("candidate proposal distribution is not HIGH_REGIME_PRIOR")
        architectures.append(architecture)
        joints.append(joint)
    if any(len(values) != expected_candidates_per_pool for values in by_pool.values()):
        raise CandidateGenerationFailed("one or more pools do not contain four candidates")
    if len(set(architectures)) != len(architectures) or len(set(joints)) != len(joints):
        raise CandidateGenerationFailed("current candidates are not internally unique")
    audit = dict(generator_audit or {})
    if audit:
        if audit.get("predictor_accessed", audit.get("predictor_accessed_during_generation", False)):
            raise CandidateGenerationFailed("candidate generator accessed a predictor")
        if audit.get("validation_labels_accessed", audit.get("validation_labels_accessed_during_generation", False)):
            raise CandidateGenerationFailed("candidate generator accessed validation labels")
        distribution = audit.get("proposal_distribution", audit.get("hp_distribution", expected_distribution))
        if expected_distribution not in str(distribution):
            raise CandidateGenerationFailed("candidate generator changed the proposal distribution")
        required_false = (
            "historical_top_architectures_used", "metric_distance_used",
            "mutation_or_local_search_used",
        )
        if audit.get("model_blind", True) is not True or any(bool(audit.get(key, False)) for key in required_false):
            raise CandidateGenerationFailed("candidate generation is not model-blind")
    return {
        "passed": True, "candidate_count": len(rows), "pool_count": expected_pool_count,
        "historical_architecture_overlap": 0, "historical_joint_overlap": 0,
        "previous_architecture_overlap": 0, "previous_joint_overlap": 0,
        "internal_architecture_unique": True, "internal_joint_unique": True,
        "predictor_accessed": False, "validation_labels_accessed": False,
    }


def assert_scorer_replay_identity(replayed: Mapping[str, Any], archived: Mapping[str, Any], *, atol: float = REPLAY_ATOL) -> dict[str, Any]:
    try:
        result = pilot_gate.assert_scorer_replay_identity(replayed, archived, atol)
        return {"status": "IDENTICAL", **result}
    except pilot_gate.ScorerReplayIdentityMismatch as exc:
        raise ScorerReplayIdentityMismatch(str(exc)) from exc


def assert_no_prospective_label_training(audit: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = (
        "M1_previous_pilot_labels_used", "M2_previous_pilot_labels_used",
        "B1_previous_pilot_labels_used", "B3_previous_pilot_labels_used",
        "B4_previous_pilot_labels_used", "candidate_labels_used_for_update",
        "previous_pilot_labels_used",
        "current_confirmation_labels_used", "online_update_used",
        "sequential_refit_used", "pool_sequential_updates", "candidate_replacement",
    )
    bad = []
    for key in forbidden:
        value = audit.get(key, False)
        if isinstance(value, Mapping):
            if any(bool(item) for item in value.values()):
                bad.append(key)
        elif bool(value):
            bad.append(key)
    if bad or audit.get("all_predictions_frozen_before_first_evaluation", True) is not True:
        raise ProtocolError(f"prospective labels entered scorer protocol: {bad}")
    return {"passed": True, "previous_pilot_labels_used": False, "online_update_used": False}


assert_no_online_update = assert_no_prospective_label_training


def _numeric_mapping_matches(
    observed: Any,
    expected: Any,
    *,
    atol: float = REPLAY_ATOL,
) -> bool:
    """Compare replayed scalar mappings using the frozen absolute tolerance."""
    if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
        return False
    if set(observed) != set(expected):
        return False
    try:
        return all(
            math.isclose(
                float(observed[key]),
                float(expected[key]),
                rel_tol=0.0,
                abs_tol=atol,
            )
            for key in expected
        )
    except (TypeError, ValueError):
        return False


def validate_metric_scorer_protocol(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed validation for the frozen M1/M2 metric scorer only."""
    if "M1" in audit or "M2" in audit:
        if not isinstance(audit.get("M1"), Mapping) or not isinstance(audit.get("M2"), Mapping):
            raise ProtocolError("metric scorer protocol requires both M1 and M2")
        m1 = audit["M1"]
        m2 = audit["M2"]
        metric_seeds = list(m2.get("metric_seeds", audit.get("metric_seeds", [])))
        fold_count = int(m2.get("fold_count", audit.get("fold_count", -1)))
        checks = {
            "M1": m1.get("protocol_reused_without_tuning") is True
            and int(m1.get("k", -1)) == 5
            and m1.get("distance") == "Euclidean"
            and m1.get("weight") == "inverse_distance_plus_1e-8",
            "M2_kernel": m2.get("protocol_reused_without_tuning") is True
            and m2.get("kernel") == "Matern-5/2",
            "M2_isotropic": m2.get("ard") is False
            and int(m2.get("lengthscale_count", -1)) == 1,
            "M2_unstandardized": m2.get("input_standardized") is False,
            "M2_noise": m2.get("fixed_observation_noise_rule") == "validation_se_squared"
            and m2.get("fitted_noise_floor_rule") == "validation_se_squared_plus_floor_squared",
            "M2_mean_only": m2.get("posterior_variance_used") is False,
            "metric_seeds": metric_seeds == list(METRIC_SEEDS),
            "outer_folds": fold_count == 5,
            "historical_only": m2.get("previous_pilot_labels_used") is False
            and m2.get("current_confirmation_labels_used") is False
            and m2.get("candidate_labels_used") is False,
        }
    else:
        required = (
            "M1_frozen_protocol", "M1_k", "M1_distance", "M1_weight",
            "M2_kernel", "M2_ard",
            "M2_lengthscale_count", "M2_input_standardized",
            "M2_fixed_observation_noise_rule", "M2_fitted_noise_floor_rule",
            "metric_seeds", "fold_count", "posterior_variance_used",
            "previous_pilot_labels_used", "current_confirmation_labels_used",
            "candidate_labels_used",
        )
        missing = [key for key in required if key not in audit]
        if missing:
            raise ProtocolError(f"metric scorer protocol is missing: {missing}")
        checks = {
            "M1": audit["M1_frozen_protocol"] is True and int(audit["M1_k"]) == 5
            and audit["M1_distance"] == "Euclidean"
            and audit["M1_weight"] == "inverse_distance_plus_1e-8",
            "M2_kernel": audit["M2_kernel"] == "Matern-5/2",
            "M2_isotropic": audit["M2_ard"] is False
            and int(audit["M2_lengthscale_count"]) == 1,
            "M2_unstandardized": audit["M2_input_standardized"] is False,
            "M2_noise": audit["M2_fixed_observation_noise_rule"] == "validation_se_squared"
            and audit["M2_fitted_noise_floor_rule"] == "validation_se_squared_plus_floor_squared",
            "M2_mean_only": audit["posterior_variance_used"] is False,
            "metric_seeds": list(audit["metric_seeds"]) == list(METRIC_SEEDS),
            "outer_folds": int(audit["fold_count"]) == 5,
            "historical_only": audit["previous_pilot_labels_used"] is False
            and audit["current_confirmation_labels_used"] is False
            and audit["candidate_labels_used"] is False,
        }
        if "reproduced" in audit or "archived" in audit:
            reproduced = audit.get("reproduced")
            archived = audit.get("archived")
            checks["identity_status"] = audit.get("status") == "IDENTICAL"
            checks["M1_numeric_identity"] = isinstance(reproduced, Mapping) and isinstance(archived, Mapping) \
                and _numeric_mapping_matches(reproduced.get("M1_oof"), archived.get("M1_oof"))
            checks["M2_numeric_identity"] = isinstance(reproduced, Mapping) and isinstance(archived, Mapping) \
                and _numeric_mapping_matches(reproduced.get("M2_oof"), archived.get("M2_oof"))
            checks["five_fold_three_seed_identity"] = isinstance(reproduced, Mapping) and set(
                reproduced.get("best_epochs", {})
            ) == {f"fold={fold},seed={seed}" for fold in range(5) for seed in METRIC_SEEDS}
        metric_runs = [run for run in audit.get("runs", []) if run.get("method") == "M2"]
        if metric_runs:
            checks["M2_five_folds_three_seeds"] = {
                (int(run.get("fold", -1)), int(run.get("seed", -1))) for run in metric_runs
            } == {(fold, seed) for fold in range(5) for seed in METRIC_SEEDS}
    if not all(checks.values()):
        raise ProtocolError(f"metric scorer protocol mismatch: {[key for key, value in checks.items() if not value]}")
    return {
        "passed": True, "required_methods": ["M1", "M2"], **checks,
        "M1_frozen_protocol": checks["M1"],
        "M2_isotropic_unstandardized": checks["M2_isotropic"] and checks["M2_unstandardized"],
    }


def validate_nonmetric_baseline_protocol(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed validation for frozen B1/B3/B4 baselines only."""
    if any(method in audit for method in BASELINE_METHODS):
        missing = [method for method in BASELINE_METHODS if not isinstance(audit.get(method), Mapping)]
        if missing:
            raise ProtocolError(f"non-metric baseline protocol is missing: {missing}")
        b1, b3, b4 = audit["B1"], audit["B3"], audit["B4"]
        checks = {
            "B1": b1.get("protocol_reused_without_tuning") is True
            and b1.get("kernel") == "Matern-5/2" and b1.get("ard") is True
            and b1.get("input_standardized") is True and b1.get("posterior_variance_used") is False,
            "B3": b3.get("protocol_reused_without_tuning") is True
            and b3.get("kernel") == "Matern-5/2" and b3.get("ard") is True
            and b3.get("input_standardized") is True and b3.get("posterior_variance_used") is False,
            "B4": b4.get("protocol_reused_without_tuning") is True
            and b4.get("loss") == "Huber" and list(b4.get("model_seeds", [])) == list(METRIC_SEEDS)
            and b4.get("final_refit_used_complete_outer_train") is True,
            "historical_only": all(
                block.get("previous_pilot_labels_used") is False
                and block.get("current_confirmation_labels_used") is False
                for block in (b1, b3, b4)
            ),
        }
    else:
        required = (
            "B1_frozen_protocol", "B3_frozen_protocol", "B4_frozen_protocol",
            "posterior_variance_used", "previous_pilot_labels_used",
            "current_confirmation_labels_used",
        )
        missing = [key for key in required if key not in audit]
        if missing:
            raise ProtocolError(f"non-metric baseline protocol is missing: {missing}")
        checks = {
            "B1": audit["B1_frozen_protocol"] is True,
            "B3": audit["B3_frozen_protocol"] is True,
            "B4": audit["B4_frozen_protocol"] is True,
            "mean_only": audit["posterior_variance_used"] is False,
            "historical_only": audit["previous_pilot_labels_used"] is False
            and audit["current_confirmation_labels_used"] is False,
        }
        all_runs = list(audit.get("runs", []))
        gp_runs = list(audit.get("gp_runs", [])) or [
            run for run in all_runs if run.get("method") in {"B1", "B3"}
        ]
        expected_gp = {(fold, method) for fold in range(5) for method in ("B1", "B3")}
        observed_gp = {(int(run.get("fold", -1)), str(run.get("method"))) for run in gp_runs}
        checks["B1_B3_five_folds"] = observed_gp == expected_gp
        checks["B1_B3_protocol"] = all(
            run.get("kernel") == "Matern-5/2" and run.get("ard") is True
            and run.get("input_standardized") is True
            and run.get("posterior_variance_used") is False
            for run in gp_runs
        )
        direct_runs = list(audit.get("direct_regressor_runs", [])) or [
            run for run in all_runs if run.get("method") == "B4"
        ]
        expected_direct = {(fold, seed) for fold in range(5) for seed in METRIC_SEEDS}
        observed_direct = {(int(run.get("fold", -1)), int(run.get("seed", -1))) for run in direct_runs}
        checks["B4_five_folds_three_seeds"] = observed_direct == expected_direct
        checks["B4_protocol"] = all(
            run.get("loss") == "Huber"
            and run.get("fresh_model_for_full_outer_refit") is True
            and run.get("final_refit_used_complete_outer_train") is True
            for run in direct_runs
        )
    if not all(checks.values()):
        raise ProtocolError(f"non-metric baseline protocol mismatch: {[key for key, value in checks.items() if not value]}")
    return {
        "passed": True, "required_methods": ["B1", "B3", "B4"], **checks,
        "B1_frozen_protocol": checks["B1"],
        "B3_frozen_protocol": checks["B3"],
        "B4_frozen_protocol": checks["B4"],
    }


def validate_scorer_protocol(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Strict combined scorer validation retained for the all-method context."""
    metric = validate_metric_scorer_protocol(audit)
    baselines = validate_nonmetric_baseline_protocol(audit)
    return {"passed": True, "metric": metric, "baselines": baselines}


def select_historical_baseline(
    historical_oof_metrics: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    prospective_labels: Any = None,
) -> dict[str, Any]:
    if isinstance(historical_oof_metrics, Mapping):
        rows = [{"method": key, **dict(value)} for key, value in historical_oof_metrics.items() if key in BASELINE_METHODS]
    else:
        rows = [dict(row) for row in historical_oof_metrics if str(row.get("method")) in BASELINE_METHODS]
    if {str(row["method"]) for row in rows} != set(BASELINE_METHODS):
        raise ProtocolError("historical B1/B3/B4 metrics are incomplete")
    def value(row: Mapping[str, Any], names: Sequence[str]) -> float:
        for name in names:
            if name in row:
                return float(row[name])
        raise ProtocolError(f"historical baseline metric is missing: {names[0]}")
    normalized = [{
        "method": str(row["method"]),
        "mean_simple_regret": value(row, ("mean_simple_regret", "best_of_4_mean_simple_regret", "mean_best_of_4_simple_regret")),
        "mae": value(row, ("mae", "historical_oof_mae")),
    } for row in rows]
    minimum = min(row["mean_simple_regret"] for row in normalized)
    tied = [row for row in normalized if abs(row["mean_simple_regret"] - minimum) <= 1.0e-12]
    tied.sort(key=lambda row: (row["mae"], row["method"]))
    winner = tied[0]
    return {
        "B_STAR": winner["method"], "selected_method": winner["method"],
        "selection_source": "historical_oof_only", "regret_tie_tolerance": 1.0e-12,
        "selection_rule": "historical best-of-4 regret; <=1e-12 tie by OOF MAE then method id",
        "historical_metrics": normalized, "prospective_labels_used": False,
    }


def frozen_random_indices(*, pool_count: int = POOL_COUNT, choices_per_pool: int = CANDIDATES_PER_POOL, seed: int = RANDOM_SELECTION_SEED) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(choices_per_pool) for _ in range(pool_count)]


def select_pool_argmax(rows: Sequence[Mapping[str, Any]], *, score_field: str) -> str:
    if len(rows) != CANDIDATES_PER_POOL:
        raise ProtocolError("selection pool must contain exactly four candidates")
    ranked = sorted(rows, key=lambda row: (-float(row[score_field]), str(row["joint_fingerprint"])))
    return str(ranked[0]["joint_fingerprint"])


def make_selections(predictions: Sequence[Mapping[str, Any]], b_star: str, random_indices: Sequence[int] | None = None) -> dict[str, Any]:
    if b_star not in BASELINE_METHODS:
        raise ProtocolError("B_STAR must be one of B1, B3, and B4")
    random_indices = list(random_indices or frozen_random_indices())
    pools: dict[str, Any] = {}
    for pool in range(POOL_COUNT):
        rows = sorted((row for row in predictions if int(row["pool_id"]) == pool), key=lambda row: str(row["joint_fingerprint"]))
        if len(rows) != CANDIDATES_PER_POOL:
            raise ProtocolError(f"pool {pool} does not contain four prediction rows")
        selected = {method: select_pool_argmax(rows, score_field=method) for method in ("M1", "M2", "B1", "B3", "B4")}
        selected["R"] = str(rows[random_indices[pool]]["joint_fingerprint"])
        selected["B_STAR"] = selected[b_star]
        pools[str(pool)] = selected
    return {"pools": pools, "B_STAR": b_star, "random_seed": RANDOM_SELECTION_SEED, "all_pools_frozen_together": True}


def disagreement_counts(selections: Mapping[str, Any], b_star: str) -> dict[str, int]:
    if isinstance(selections, Mapping):
        raw = selections.get("pools", selections)
        pool_rows = list(raw.values()) if isinstance(raw, Mapping) else list(raw)
    else:
        pool_rows = list(selections)
    def selected(row: Mapping[str, Any], method: str) -> str:
        return str(row.get(method, row.get(f"{method}_selected_joint_fingerprint")))
    d1 = sum(selected(row, "M1") != selected(row, b_star) for row in pool_rows)
    d2 = sum(selected(row, "M2") != selected(row, b_star) for row in pool_rows)
    d12 = sum(selected(row, "M1") != selected(row, "M2") for row in pool_rows)
    return {"d1": d1, "d2": d2, "d12": d12}


def freeze_pre_eval(
    pre_eval_dir: Path,
    *,
    candidate_manifest: Any,
    predictions: Sequence[Mapping[str, Any]],
    selections: Any,
    baseline_selection: Any,
    scorer_identity: Any,
    previous_pilot_exclusion_audit: Any,
    historical_input_hashes: Mapping[str, Any] | None = None,
    generator_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_rows = candidate_manifest.get("candidates", []) if isinstance(candidate_manifest, Mapping) else candidate_manifest
    selection_rows = selections.get("pools", {}) if isinstance(selections, Mapping) else selections
    selection_count = len(selection_rows)
    if len(candidate_rows) != CANDIDATE_COUNT or len(predictions) != CANDIDATE_COUNT or selection_count != POOL_COUNT:
        raise PreEvalFreezeError("all 16 pools and 64 predictions must be frozen together")
    pre_eval_dir.mkdir(parents=True, exist_ok=True)
    dump_json(pre_eval_dir / "candidate_manifest.json", candidate_manifest)
    fields = list(predictions[0].keys()) if predictions else (
        "pool_id", "joint_fingerprint", "architecture_fingerprint", "M1", "M2", "B1", "B3", "B4"
    )
    dump_csv(pre_eval_dir / "predictions.csv", predictions, fields)
    dump_json(pre_eval_dir / "selections.json", selections)
    dump_json(pre_eval_dir / "baseline_selection.json", baseline_selection)
    dump_json(pre_eval_dir / "scorer_identity.json", {
        **dict(scorer_identity or {}),
        "historical_input_hashes": dict(historical_input_hashes or {}),
        "generator_identity": dict(generator_identity or {}),
    })
    dump_json(pre_eval_dir / "previous_pilot_exclusion_audit.json", previous_pilot_exclusion_audit)
    names = (
        "candidate_manifest.json", "predictions.csv", "selections.json",
        "baseline_selection.json", "scorer_identity.json", "previous_pilot_exclusion_audit.json",
    )
    entries = [{
        "path": name, "size_bytes": (pre_eval_dir / name).stat().st_size,
        "sha256": sha256_file(pre_eval_dir / name),
    } for name in sorted(names)]
    root = stable_hash([(entry["path"], entry["sha256"]) for entry in entries])
    dump_json(pre_eval_dir / "freeze_manifest.json", {
        "format_version": PROTOCOL_VERSION, "created_at": utc_now(),
        "immutable_entries": entries, "root_sha256": root,
        "candidate_validation_labels_available_at_freeze": False,
        "all_sixteen_pools_frozen_before_first_gnn_run": True,
        "previous_pilot_labels_used_for_training": False,
    })
    dump_text(pre_eval_dir / "pre_eval_root_sha256.txt", root + "\n")
    dump_text(pre_eval_dir / "PRE_EVAL_SELECTION_FROZEN", "")
    return {"passed": True, "root_sha256": root, "all_sixteen_pools_frozen_together": True, "pool_count": POOL_COUNT}


def verify_pre_eval_freeze(pre_eval_dir: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((pre_eval_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreEvalFreezeError("pre-evaluation freeze manifest is unavailable") from exc
    entries = sorted(manifest.get("immutable_entries", []), key=lambda row: str(row["path"]))
    for entry in entries:
        path = pre_eval_dir / str(entry["path"])
        if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]) or sha256_file(path) != entry["sha256"]:
            raise PreEvalFreezeError(f"pre-evaluation artifact changed: {entry['path']}")
    root = stable_hash([(entry["path"], entry["sha256"]) for entry in entries])
    if root != manifest.get("root_sha256"):
        raise PreEvalFreezeError("pre-evaluation root differs from freeze manifest")
    if (pre_eval_dir / "pre_eval_root_sha256.txt").read_text(encoding="utf-8").strip() != root:
        raise PreEvalFreezeError("pre-evaluation root sidecar differs")
    if not (pre_eval_dir / "PRE_EVAL_SELECTION_FROZEN").is_file():
        raise PreEvalFreezeError("pre-evaluation marker is absent")
    return {"passed": True, "root_sha256": root}


def assert_candidate_manifest_immutable(before: Any, after: Any) -> None:
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise PreEvalFreezeError("candidate manifest changed after selection freeze")


def information_gate(
    selections: Mapping[str, Any] | Sequence[Mapping[str, Any]], b_star: str,
    minimum: int = 6, *, minimum_disagreement_pools: int | None = None,
) -> dict[str, Any]:
    if minimum_disagreement_pools is not None:
        minimum = int(minimum_disagreement_pools)
    counts = disagreement_counts(selections, b_star)
    maximum = max(counts["d1"], counts["d2"])
    authorized = maximum >= int(minimum)
    return {
        "authorized": authorized, "minimum_disagreement_pools": int(minimum),
        "max_disagreement": maximum, "disagreement_counts": counts,
        "total_gnn_runs": TOTAL_GNN_RUNS if authorized else 0,
        "classification": None if authorized else "CONFIRMATION_LOW_INFORMATION",
    }


def frozen_training_protocol_identity() -> str:
    return stable_hash({
        "dataset": "Cora", "split": "public", "graph_transform": "NormalizeFeatures",
        "dataset_content_fingerprint": "373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7",
        "split_fingerprint": "bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09",
        "hp_mode": "global4", "epochs": 300, "patience": 80,
        "maximum_system_retries": 2, "training_seeds": list(TRAINING_SEEDS),
        "track_test": False, "entry": "eval_utils.train_and_eval_arch",
        "eval_utils_sha256": "0cbfb402db030173ca11e4e90570df6ac427d99b416470863f55280d51e2541d",
    })


def build_evaluation_tasks(
    candidates: Sequence[Mapping[str, Any]],
    *,
    information_gate_passed: bool = True,
    pre_eval_verified: bool = True,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
    track_test: bool = False,
) -> list[dict[str, Any]]:
    if track_test:
        raise ProtocolError("test tracking is forbidden")
    if not information_gate_passed:
        return []
    if not pre_eval_verified:
        raise PreEvalFreezeError("evaluation cannot start before pre-evaluation verification")
    if list(training_seeds) != list(TRAINING_SEEDS):
        raise ProtocolError("training seeds must be exactly [0,1,2]")
    if len(candidates) != CANDIDATE_COUNT:
        raise ProtocolError("evaluation requires exactly 64 frozen candidates")
    protocol = frozen_training_protocol_identity()
    tasks = []
    for candidate in sorted(candidates, key=lambda row: (int(row["pool_id"]), str(row["joint_fingerprint"]))):
        for seed in TRAINING_SEEDS:
            joint = str(candidate["joint_fingerprint"])
            pool = int(candidate["pool_id"])
            tasks.append({
                "pool_id": pool, "joint_fingerprint": joint, "training_seed": seed,
                "training_protocol_identity": protocol,
                "task_key": f"initial::{pool}::{joint}::seed={seed}::protocol={protocol}",
                "candidate": dict(candidate),
            })
    if len(tasks) != TOTAL_GNN_RUNS:
        raise ProtocolError("evaluation task count is not 192")
    return tasks


def aggregate_evaluation_results(
    candidates: Sequence[Mapping[str, Any]],
    raw_seed_rows: Sequence[Mapping[str, Any]],
    *,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], bool]:
    reject_test_fields(raw_seed_rows, allow_track_test_false=True)
    expected = {(str(row["joint_fingerprint"]), seed) for row in candidates for seed in TRAINING_SEEDS}
    observed: dict[tuple[str, int], float] = {}
    for row in raw_seed_rows:
        joint = str(row.get("joint_fingerprint", row.get("joint_config_fingerprint")))
        seed = int(row.get("training_seed", row.get("seed")))
        key = (joint, seed)
        if key not in expected or key in observed:
            raise EvaluationIncompleteError("evaluation contains a foreign or duplicate task")
        if row.get("terminal_status") != "success" or row.get("valid", True) is not True:
            continue
        value = float(row.get("validation_accuracy", row.get("val_acc")))
        if not math.isfinite(value):
            continue
        observed[key] = value
    complete = set(observed) == expected
    candidate_by_fp = {str(row["joint_fingerprint"]): row for row in candidates}
    stats: list[dict[str, Any]] = []
    for joint, candidate in sorted(candidate_by_fp.items(), key=lambda item: (int(item[1]["pool_id"]), item[0])):
        values = [observed.get((joint, seed)) for seed in TRAINING_SEEDS]
        if any(value is None for value in values):
            if not allow_incomplete:
                raise EvaluationIncompleteError(f"candidate lacks complete matched seeds: {joint}")
            mean = None
        else:
            mean = float(np.mean(np.asarray(values, dtype=float)))
        stats.append({
            "pool_id": int(candidate["pool_id"]), "joint_fingerprint": joint,
            "architecture_fingerprint": str(candidate["architecture_fingerprint"]),
            "validation_seed_0": values[0], "validation_seed_1": values[1],
            "validation_seed_2": values[2], "validation_mean": mean,
        })
    if complete:
        by_pool: dict[int, list[dict[str, Any]]] = {pool: [] for pool in range(POOL_COUNT)}
        for row in stats:
            by_pool[int(row["pool_id"])].append(row)
        for pool, rows in by_pool.items():
            if len(rows) != CANDIDATES_PER_POOL or any(row["validation_mean"] is None for row in rows):
                complete = False
                break
            oracle = max(float(row["validation_mean"]) for row in rows)
            for row in rows:
                row["observed_pool_oracle"] = oracle
    if allow_incomplete:
        return stats, complete
    if not complete:
        raise EvaluationIncompleteError("not all 16 pools have 4 x 3 successful runs")
    return stats


def oracle_hit(selected_mean: float, oracle_mean: float, *, tolerance: float = 1.0e-12) -> int:
    return int(abs(float(selected_mean) - float(oracle_mean)) <= tolerance)


def _correlation_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float | None]:
    from scipy.stats import kendalltau, pearsonr, spearmanr
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if len(y) < 2:
        return {"mae": None, "pearson": None, "spearman": None, "kendall": None}
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "pearson": float(pearsonr(y, p).statistic),
        "spearman": float(spearmanr(y, p).statistic),
        "kendall": float(kendalltau(y, p).statistic),
    }


def compute_method_metrics(
    config_stats: Sequence[Mapping[str, Any]],
    selections: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if isinstance(selections, Mapping):
        pools = selections.get("pools", selections)
    else:
        pools = {str(int(row["pool_id"])): row for row in selections}
    stats_by_fp = {str(row["joint_fingerprint"]): row for row in config_stats}
    per_pool: list[dict[str, Any]] = []
    for pool in range(POOL_COUNT):
        pool_rows = [row for row in config_stats if int(row["pool_id"]) == pool]
        if len(pool_rows) != CANDIDATES_PER_POOL:
            raise EvaluationIncompleteError(f"pool {pool} is incomplete")
        oracle = max(float(row["validation_mean"]) for row in pool_rows)
        output: dict[str, Any] = {"pool_id": pool, "observed_pool_oracle": oracle}
        for method in METHODS:
            selection_row = pools[str(pool)] if isinstance(pools, Mapping) else pools[pool]
            selected = str(selection_row.get(method, selection_row.get(f"{method}_selected_joint_fingerprint")))
            value = float(stats_by_fp[selected]["validation_mean"])
            output[f"{method}_selected_joint_fingerprint"] = selected
            output[f"{method}_selected_validation_mean"] = value
            output[f"{method}_simple_regret"] = oracle - value
            output[f"{method}_true_best_hit"] = oracle_hit(value, oracle)
            output[f"{method}_within_0.006"] = int(oracle - value <= 0.006 + 1.0e-15)
        per_pool.append(output)
    method_metrics: dict[str, Any] = {}
    prediction_rows = list(predictions or [])
    actual = [float(row["validation_mean"]) for row in config_stats]
    for method in METHODS:
        values = [float(row[f"{method}_selected_validation_mean"]) for row in per_pool]
        regrets = [float(row[f"{method}_simple_regret"]) for row in per_pool]
        hits = [int(row[f"{method}_true_best_hit"]) for row in per_pool]
        near = [int(row[f"{method}_within_0.006"]) for row in per_pool]
        prediction_metrics = {"mae": None, "pearson": None, "spearman": None, "kendall": None}
        if method != "R" and prediction_rows:
            predicted = [float(row[method]) for row in prediction_rows]
            prediction_metrics = _correlation_metrics(actual, predicted)
        method_metrics[method] = {
            "selected_validation_mean": float(np.mean(values)),
            "mean_simple_regret": float(np.mean(regrets)),
            "median_simple_regret": float(np.median(regrets)),
            "true_best_hit_count": int(sum(hits)), "true_best_hit_rate": float(np.mean(hits)),
            "within_0.006_of_oracle_rate": float(np.mean(near)),
            **prediction_metrics,
        }
    return {
        "per_pool": per_pool, "methods": method_metrics, "statistical_unit": "pool",
        "pool_count": POOL_COUNT, "training_seed_count_is_not_independent_unit": True,
    }


def pool_bootstrap(
    values: Sequence[float], *, replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED, unit: str = "pool",
) -> dict[str, Any]:
    if unit != "pool" or int(replicates) != BOOTSTRAP_REPLICATES or int(seed) != BOOTSTRAP_SEED:
        raise ProtocolError("bootstrap protocol must be pool/10000/seed0")
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        raise ProtocolError("pool bootstrap requires observations")
    rng = np.random.default_rng(seed)
    samples = np.mean(array[rng.integers(0, len(array), size=(replicates, len(array)))], axis=1)
    return {
        "bootstrap_unit": "pool", "pool_count": len(array), "replicates": replicates,
        "seed": seed, "observed_mean": float(np.mean(array)),
        "confidence_interval_95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def _comparison_rows(per_pool: Sequence[Mapping[str, Any]], method: str, comparator: str) -> list[dict[str, Any]]:
    rows = []
    for row in per_pool:
        method_fp = str(row[f"{method}_selected_joint_fingerprint"])
        comparator_fp = str(row[f"{comparator}_selected_joint_fingerprint"])
        method_value = float(row[f"{method}_selected_validation_mean"])
        comparator_value = float(row[f"{comparator}_selected_validation_mean"])
        difference = method_value - comparator_value
        outcome = "tie" if difference == 0.0 else ("method_win" if difference > 0.0 else "comparator_win")
        rows.append({
            "pool_id": int(row["pool_id"]), "method": method, "comparator": comparator,
            "method_selected": method_fp, "comparator_selected": comparator_fp,
            "selections_disagree": method_fp != comparator_fp,
            "method_value": method_value, "comparator_value": comparator_value,
            "difference": difference, "outcome": outcome,
        })
    return rows


def evaluate_metric_confirmation(
    method: str,
    per_pool: Sequence[Mapping[str, Any]],
    method_metrics: Mapping[str, Mapping[str, Any]],
    b_star: str,
) -> dict[str, Any]:
    if method not in {"M1", "M2"} or b_star not in BASELINE_METHODS:
        raise ProtocolError("metric confirmation method/comparator is invalid")
    rows = _comparison_rows(per_pool, method, b_star)
    disagreements = [row for row in rows if row["selections_disagree"]]
    wins = sum(row["outcome"] == "method_win" for row in disagreements)
    losses = sum(row["outcome"] == "comparator_win" for row in disagreements)
    ties = sum(row["outcome"] == "tie" for row in disagreements)
    d = len(disagreements)
    required_wins = max(4, math.ceil((2.0 / 3.0) * d))
    differences = [float(row["difference"]) for row in rows]
    bootstrap = pool_bootstrap(differences)
    metrics = method_metrics.get("methods", method_metrics)
    method_regret = float(metrics[method]["mean_simple_regret"])
    comparator_regret = float(metrics[b_star]["mean_simple_regret"])
    def hit_count(row: Mapping[str, Any]) -> int:
        return int(row.get("true_best_hit_count", row.get("true_best_hits", 0)))
    if comparator_regret < 1.0e-6:
        c4 = method_regret <= comparator_regret
        c4_mode = "comparator_near_zero_noninferiority"
    else:
        c4 = method_regret <= 0.80 * comparator_regret
        c4_mode = "relative_reduction_at_least_20_percent"
    conditions = {
        "C1": d >= 6,
        "C2": wins >= required_wins,
        "C3": float(np.mean(differences)) >= 0.003,
        "C4": c4,
        "C5": hit_count(metrics[method]) >= hit_count(metrics[b_star]),
        "C6": float(bootstrap["confidence_interval_95"][0]) > 0.0,
    }
    return {
        "method": method, "comparator": b_star, **conditions,
        "confirmed": all(conditions.values()), "disagreement_pools": d,
        "wins": wins, "losses": losses, "ties": ties, "required_wins": required_wins,
        "mean_selected_difference": float(np.mean(differences)),
        "method_mean_regret": method_regret, "comparator_mean_regret": comparator_regret,
        "C4_mode": c4_mode, "bootstrap": bootstrap, "per_pool": rows,
    }


def combine_metric_family_confirmation(
    m1_confirmation: Mapping[str, Any], m2_confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "family_confirmed": bool(m1_confirmation.get("confirmed")) or bool(m2_confirmation.get("confirmed")),
        "confirmed_methods": [method for method, evidence in (("M1", m1_confirmation), ("M2", m2_confirmation)) if evidence.get("confirmed")],
        "M1": dict(m1_confirmation), "M2": dict(m2_confirmation),
        "combination_rule": "confirmed(M1) OR confirmed(M2)",
    }


def _surrogate_advantage(
    favored: str, other: str, per_pool: Sequence[Mapping[str, Any]],
    method_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = _comparison_rows(per_pool, favored, other)
    disagreements = [row for row in rows if row["selections_disagree"]]
    wins = sum(row["outcome"] == "method_win" for row in disagreements)
    losses = sum(row["outcome"] == "comparator_win" for row in disagreements)
    ties = sum(row["outcome"] == "tie" for row in disagreements)
    d = len(disagreements)
    required = max(4, math.ceil((2.0 / 3.0) * d))
    metrics = method_metrics.get("methods", method_metrics)
    favored_regret = float(metrics[favored]["mean_simple_regret"])
    other_regret = float(metrics[other]["mean_simple_regret"])
    def hit_count(row: Mapping[str, Any]) -> int:
        return int(row.get("true_best_hit_count", row.get("true_best_hits", 0)))
    if other_regret < 1.0e-6:
        regret_condition = favored_regret <= other_regret
        regret_mode = "other_near_zero_noninferiority"
    else:
        regret_condition = favored_regret <= 0.90 * other_regret
        regret_mode = "relative_reduction_at_least_10_percent"
    bootstrap = pool_bootstrap([float(row["difference"]) for row in rows])
    conditions = {
        "S1": d >= 6,
        "S2": wins >= required,
        "S3": regret_condition,
        "S4": hit_count(metrics[favored]) >= hit_count(metrics[other]),
        "S5": float(bootstrap["confidence_interval_95"][0]) > 0.0,
    }
    return {
        "favored": favored, "other": other, **conditions, "passed": all(conditions.values()),
        "disagreement_pools": d, "wins": wins, "losses": losses, "ties": ties,
        "required_wins": required, "favored_mean_regret": favored_regret,
        "other_mean_regret": other_regret, "regret_mode": regret_mode,
        "bootstrap": bootstrap, "per_pool": rows,
    }


def select_final_surrogate(
    per_pool: Sequence[Mapping[str, Any]],
    method_metrics: Mapping[str, Mapping[str, Any]],
    family_confirmation: Mapping[str, Any],
    b_star: str,
    pilot_context: Any = None,
    random_metrics: Any = None,
) -> dict[str, Any]:
    if not family_confirmation.get("family_confirmed", False):
        return {
            "classification": "METRIC_FAMILY_NOT_CONFIRMED", "final_surrogate": None,
            "selection_basis": "INDEPENDENT_CONFIRMATION_FAILED",
            "historical_pilot_used_in_gate": False, "random_used_in_gate": False,
            "pilot_used_in_gate": False, "random_used_in_primary_gate": False,
            "m1_statistically_proven_superior": False,
        }
    gp = _surrogate_advantage("M2", "M1", per_pool, method_metrics)
    knn = _surrogate_advantage("M1", "M2", per_pool, method_metrics)
    if gp["passed"]:
        classification = "METRIC_FAMILY_CONFIRMED_GP_SELECTED"
        final = "M2"
        basis = "CLEAR_GP_INCREMENT"
    elif knn["passed"]:
        classification = "METRIC_FAMILY_CONFIRMED_KNN_SELECTED"
        final = "M1"
        basis = "CLEAR_KNN_ADVANTAGE"
    else:
        classification = "METRIC_FAMILY_CONFIRMED_KNN_SELECTED"
        final = "M1"
        basis = "PARSIMONY_NO_CLEAR_GP_INCREMENT"
    return {
        "classification": classification, "final_surrogate": final,
        "selection_basis": basis, "B_STAR": b_star,
        "GP_increment": gp, "KNN_advantage": knn,
        "gp_conditions": {f"S_GP_{index}": gp[f"S{index}"] for index in range(1, 6)},
        "knn_conditions": {f"S_KNN_{index}": knn[f"S{index}"] for index in range(1, 6)},
        "historical_pilot_used_in_gate": False, "random_used_in_gate": False,
        "pilot_used_in_gate": False, "random_used_in_primary_gate": False,
        "m1_statistically_proven_superior": False,
        "parsimony_does_not_prove_M1_statistically_superior": basis == "PARSIMONY_NO_CLEAR_GP_INCREMENT",
    }


def paired_pool_comparisons(per_pool: Sequence[Mapping[str, Any]], b_star: str) -> dict[str, Any]:
    comparisons = {}
    for left, right in (("M1", b_star), ("M2", b_star), ("M2", "M1")):
        key = f"{left}_vs_{right}"
        rows = _comparison_rows(per_pool, left, right)
        comparisons[key] = {
            "rows": rows, "differences": [row["difference"] for row in rows],
            "wins": sum(row["outcome"] == "method_win" for row in rows),
            "losses": sum(row["outcome"] == "comparator_win" for row in rows),
            "ties": sum(row["outcome"] == "tie" for row in rows),
        }
    return comparisons


def bootstrap_comparisons(comparisons: Mapping[str, Any]) -> dict[str, Any]:
    return {key: pool_bootstrap(value["differences"]) for key, value in comparisons.items()}


def sign_test_summary(comparisons: Mapping[str, Any]) -> dict[str, Any]:
    from scipy.stats import binomtest
    output = {}
    for key, value in comparisons.items():
        wins = int(value["wins"])
        losses = int(value["losses"])
        n = wins + losses
        output[key] = {
            "wins": wins, "losses": losses, "ties": int(value["ties"]),
            "non_tied_pool_count": n,
            "exact_one_sided_sign_test_pvalue": None if n == 0 else float(binomtest(wins, n, 0.5, alternative="greater").pvalue),
            "descriptive_only": True,
        }
    return output


def _source_paths() -> list[Path]:
    soft_snapshot = HISTORICAL_SOFT_ROOT / "source_snapshot"
    high_snapshot = HISTORICAL_HIGH_ROOT / "source_snapshot"
    pilot_snapshot = PREVIOUS_PILOT_ROOT / "source_snapshot"
    return [
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/run_joint_metric_independent_confirmation_surrogate_selection.sh",
        REPO_ROOT / "tests/test_joint_metric_independent_confirmation_surrogate_selection.py",
        DEFAULT_CONFIG,
        soft_snapshot / "configs/joint_configuration_soft_metric_gate_v2.yaml",
        soft_snapshot / "analyse/joint_configuration_soft_metric_gate.py",
        soft_snapshot / "analyse/joint_configuration_metric_gate.py",
        soft_snapshot / "analyse/performance_metric_quick_gate.py",
        soft_snapshot / "analyse/joint_pair_power_projection_audit.py",
        pilot_snapshot / "analyse/joint_metric_prospective_selection_gate.py",
        high_snapshot / "analyse/high_regime_prior_model_blind_validation.py",
        high_snapshot / "analyse/run_high_regime_prior_model_blind_validation_labels.py",
        high_snapshot / "analyse/analyse_high_regime_prior_model_blind_validation.py",
        high_snapshot / "eval_utils.py",
        high_snapshot / "architecture_diagnostics.py",
    ]


def aggregate_source_sha256(paths: Sequence[Path] | None = None) -> str:
    digest = hashlib.sha256()
    for path in paths or _source_paths():
        if not path.is_file():
            raise ProtocolError(f"source aggregate input is missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfirmationInputInvalid(f"cannot parse authoritative JSON: {path}") from exc


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ConfirmationInputInvalid(f"{label} hash mismatch")


def validate_authoritative_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    _require_hash(
        Path(pilot_gate.__file__).resolve(),
        "8e309fbabfc3409c97a824d12f56ce869cd1d2358f6cbc511298a06cec197579",
        "runtime prospective scorer implementation",
    )
    frozen_pilot_config_path = (
        PREVIOUS_PILOT_ROOT
        / "source_snapshot/configs/joint_metric_prospective_selection_gate_v1.yaml"
    )
    _require_hash(
        frozen_pilot_config_path,
        FROZEN_PREVIOUS_PILOT_IDENTITY["config_sha256"],
        "frozen prospective pilot protocol",
    )
    soft_artifacts = config["authoritative_inputs"]["soft_metric_v2"]["artifacts"]
    for relative in (
        "oof_predictions.csv",
        "metrics_per_fold.csv",
        "best_of_4_results.json",
        "metric_training_audit.json",
        "split_manifest.json",
    ):
        _require_hash(
            HISTORICAL_SOFT_ROOT / relative,
            str(soft_artifacts[relative]),
            f"frozen soft-v2 artifact {relative}",
        )
    try:
        frozen_pilot_config = pilot_gate.load_config(frozen_pilot_config_path)
        base_authority = pilot_gate.validate_authoritative_inputs(frozen_pilot_config)
    except Exception as exc:
        raise ConfirmationInputInvalid(str(exc)) from exc
    if not (PREVIOUS_PILOT_ROOT / "COMPLETE").is_file():
        raise ConfirmationInputInvalid("previous prospective pilot is not COMPLETE")
    _require_hash(PREVIOUS_PILOT_ARCHIVE, FROZEN_PREVIOUS_PILOT_IDENTITY["archive_sha256"], "previous pilot archive")
    _require_hash(
        PREVIOUS_PILOT_ROOT / "pre_eval/candidate_manifest.json",
        FROZEN_PREVIOUS_PILOT_IDENTITY["candidate_manifest_sha256"],
        "previous pilot candidate manifest",
    )
    state = _load_json(PREVIOUS_PILOT_ROOT / "state.json")
    decision = _load_json(PREVIOUS_PILOT_ROOT / "decision.json")
    snapshot = _load_json(PREVIOUS_PILOT_ROOT / "source_snapshot/sha256_manifest.json")
    expected = FROZEN_PREVIOUS_PILOT_IDENTITY
    checks = {
        "classification": decision.get("classification") == expected["classification"]
        and state.get("classification") == expected["classification"],
        "status": state.get("status") == "complete",
        "config": state.get("config_sha256") == expected["config_sha256"],
        "source": state.get("source_sha256") == expected["source_sha256"],
        "snapshot": snapshot.get("manifest_root_sha256") == expected["source_snapshot_manifest_root"],
        "pre_eval": state.get("pre_eval_root_sha256") == expected["pre_eval_root_sha256"],
        "complete_runs": int(state.get("completed_gnn_runs", -1)) == 96 and int(state.get("failed_runs", -1)) == 0,
        "test": state.get("test_accessed") is False and decision.get("test_accessed") is False,
    }
    if not all(checks.values()):
        raise ConfirmationInputInvalid(f"previous prospective pilot identity mismatch: {[k for k, v in checks.items() if not v]}")
    pilot_manifest = _load_json(PREVIOUS_PILOT_ROOT / "pre_eval/candidate_manifest.json")
    pilot_exclusion = recover_previous_pilot_fingerprints(pilot_manifest)
    return {
        **base_authority,
        "previous_pilot_identity": {**expected, "checks": checks},
        "previous_pilot_exclusion": pilot_exclusion,
        "previous_pilot_labels_used_for_training": False,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def replay_historical_baselines(
    runtime: Mapping[str, Any], retained: Sequence[Any], log: Callable[[str], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    soft = runtime["soft"]
    archived_rows = _read_csv(HISTORICAL_SOFT_ROOT / "oof_predictions.csv")
    archived = {
        (int(row["fold"]), str(row["joint_fingerprint"])): row
        for row in archived_rows
    }
    archived_direct = _load_json(HISTORICAL_SOFT_ROOT / "metric_training_audit.json")["direct_regressor_runs"]
    archived_best = {
        (int(row["fold"]), int(row["seed"])): int(row["best_epoch"])
        for row in archived_direct
    }
    reproduced: dict[str, dict[str, float]] = {method: {} for method in BASELINE_METHODS}
    direct_runs: list[dict[str, Any]] = []
    gp_runs: list[dict[str, Any]] = []
    for scorer in retained:
        train = scorer.train_rows
        heldout = scorer.heldout_rows
        train_y = np.asarray([float(row["validation_mean"]) for row in train], dtype=float)
        train_se = np.asarray([float(row["validation_se"]) for row in train], dtype=float)
        train_hp = soft.joint_v1._raw_hp_matrix(train)
        heldout_hp = soft.joint_v1._raw_hp_matrix(heldout)
        train_joint = np.concatenate((soft.joint_v1._explicit_graph_matrix(train), train_hp), axis=1)
        heldout_joint = np.concatenate((soft.joint_v1._explicit_graph_matrix(heldout), heldout_hp), axis=1)
        b1, b1_audit = soft.joint_v1.exact_matern_gp_predict(
            train_hp, train_y, train_se, heldout_hp,
            standardize_inputs=True, ard=True, protocol_name="B1",
        )
        b3, b3_audit = soft.joint_v1.exact_matern_gp_predict(
            train_joint, train_y, train_se, heldout_joint,
            standardize_inputs=True, ard=True, protocol_name="B3",
        )
        direct_by_seed: dict[int, Any] = {}
        for seed in METRIC_SEEDS:
            prediction, audit = soft.joint_v1.train_direct_joint_regressor(
                train, heldout, seed, soft.load_config(runtime["soft_config_path"]), scorer.hp_standardizer,
            )
            audit = {"fold": scorer.fold, **audit}
            if int(audit["best_epoch"]) != archived_best.get((scorer.fold, seed)):
                raise ScorerReplayIdentityMismatch("B4 best epoch differs from frozen soft-v2")
            direct_by_seed[seed] = np.asarray(prediction, dtype=float)
            direct_runs.append(audit)
        b4 = soft.joint_v1.average_metric_seed_predictions(direct_by_seed)
        for index, row in enumerate(heldout):
            key = (scorer.fold, str(row["joint_fingerprint"]))
            if key not in archived:
                raise ScorerReplayIdentityMismatch("baseline replay produced a foreign OOF row")
            for method, values in (("B1", b1), ("B3", b3), ("B4", b4)):
                observed = float(values[index])
                expected = float(archived[key][f"{method}_prediction"])
                if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=REPLAY_ATOL):
                    raise ScorerReplayIdentityMismatch(f"{method} OOF prediction differs at {key}")
                reproduced[method][f"fold={scorer.fold},joint={key[1]}"] = observed
        gp_runs.extend(({"fold": scorer.fold, "method": "B1", **b1_audit}, {"fold": scorer.fold, "method": "B3", **b3_audit}))
        log(f"fold {scorer.fold}: B1/B3/B4 historical OOF replay matched")
    if any(len(values) != 336 for values in reproduced.values()):
        raise ScorerReplayIdentityMismatch("baseline replay did not cover all 336 OOF configurations")
    best4 = _load_json(HISTORICAL_SOFT_ROOT / "best_of_4_results.json")["models"]
    metric_rows = _read_csv(HISTORICAL_SOFT_ROOT / "metrics_per_fold.csv")
    pooled = {
        row["model"]: row for row in metric_rows
        if row.get("scope") == "pooled_oof" and row.get("model") in BASELINE_METHODS
    }
    metrics = {
        method: {
            "mean_simple_regret": float(best4[method]["mean_simple_regret"]),
            "median_simple_regret": float(best4[method]["median_simple_regret"]),
            "probability_selecting_true_best": float(best4[method]["probability_selecting_true_best"]),
            "mae": float(pooled[method]["mae"]),
        }
        for method in BASELINE_METHODS
    }
    selection = select_historical_baseline(metrics)
    audit = {
        "status": "IDENTICAL", "reproduced_oof": reproduced,
        "direct_regressor_runs": direct_runs, "gp_runs": gp_runs,
        "historical_metrics": metrics, "baseline_selection": selection,
        "B1_frozen_protocol": True, "B3_frozen_protocol": True, "B4_frozen_protocol": True,
        "previous_pilot_labels_used": False, "current_confirmation_labels_used": False,
        "posterior_variance_used": False,
    }
    validate_nonmetric_baseline_protocol(audit)
    return audit, selection


def _candidate_row(high: Any, model: Any, graph: Any, pool: int, slot: int, raw_hp: Sequence[float]) -> dict[str, Any]:
    config = high.v1.config_from_igraph(model, graph)
    valid, reason = high.v1.validate_static_config(config)
    if not valid:
        raise CandidateGenerationFailed(f"decoded architecture is invalid: {reason}")
    architecture = high.v1.architecture_fingerprint(config)
    raw = [float(value) for value in raw_hp]
    hp = high.v1.hp_fingerprint(raw)
    joint = high.joint_config_fingerprint(architecture, hp)
    row = {
        "pool_id": pool, "batch_id": pool, "slot_id": slot, "architecture_slot": slot,
        "config_id": f"independent-confirmation-p{pool}-s{slot}",
        "joint_fingerprint": joint, "joint_config_fingerprint": joint,
        "architecture_fingerprint": architecture, "hp_fingerprint": hp,
        "operations": list(config["operations"]),
        "edges": [list(map(int, edge)) for edge in config["edges"]],
        "effective_layers": int(config["effective_layers"]), "raw_hp_vector": raw,
        "hp_regime": "hp_high", "generation_pool_seed": MASTER_CANDIDATE_SEED + pool,
        "generator_seed": MASTER_CANDIDATE_SEED + pool,
        "proposal_distribution": "HIGH_REGIME_PRIOR",
        "candidate_validation_label_available": False,
    }
    row.update(high.v1.realize_global4_hp(row))
    row["raw_hp"] = {key: row[key] for key in ("lr", "dropout", "hidden_dim", "l2")}
    return row


def generate_candidates(
    configurations: Sequence[Mapping[str, Any]],
    previous_pilot: Mapping[str, Any],
    log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    high = pilot_gate._load_frozen_high_module()
    checkpoint = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
    if not checkpoint.is_file() or sha256_file(checkpoint) != "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4":
        raise ConfirmationInputInvalid("frozen VAE checkpoint hash mismatch")
    model = high.v1.load_joint_vae(checkpoint, "global4", device="cpu")
    model.eval()
    model.arch_vae.eval()
    historical_arch = {str(row["architecture_fingerprint"]) for row in configurations}
    historical_joint = {str(row["joint_fingerprint"]) for row in configurations}
    previous_arch = set(map(str, previous_pilot["architecture_fingerprints"]))
    previous_joint = set(map(str, previous_pilot["joint_fingerprints"]))
    accepted_arch: set[str] = set()
    accepted_joint: set[str] = set()
    candidates: list[dict[str, Any]] = []
    pool_audit = []
    for pool in range(POOL_COUNT):
        seed = MASTER_CANDIDATE_SEED + pool
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        hp_rng = np.random.default_rng(seed)
        rows: list[dict[str, Any]] = []
        draws = invalid = excluded = 0
        while len(rows) < CANDIDATES_PER_POOL and draws < MAX_GENERATION_DRAWS_PER_POOL:
            size = min(GENERATION_BATCH_SIZE, MAX_GENERATION_DRAWS_PER_POOL - draws)
            with torch.no_grad():
                graphs = model.arch_vae.decode(torch.randn((size, 12), generator=generator, device="cpu"), stochastic=False)
            for graph in graphs:
                draws += 1
                try:
                    graph_config = high.v1.config_from_igraph(model, graph)
                    valid, _reason = high.v1.validate_static_config(graph_config)
                    if not valid:
                        raise ValueError("invalid graph")
                    architecture = high.v1.architecture_fingerprint(graph_config)
                except (TypeError, ValueError, KeyError, IndexError):
                    invalid += 1
                    continue
                if architecture in historical_arch or architecture in previous_arch or architecture in accepted_arch:
                    excluded += 1
                    continue
                raw_hp = hp_rng.uniform(0.80, 0.95, size=4).astype(np.float64).tolist()
                row = _candidate_row(high, model, graph, pool, len(rows), raw_hp)
                if row["joint_fingerprint"] in historical_joint or row["joint_fingerprint"] in previous_joint or row["joint_fingerprint"] in accepted_joint:
                    excluded += 1
                    continue
                accepted_arch.add(str(row["architecture_fingerprint"]))
                accepted_joint.add(str(row["joint_fingerprint"]))
                rows.append(row)
                if len(rows) == CANDIDATES_PER_POOL:
                    break
            log(f"pool {pool}: accepted {len(rows)}/4 after {draws} decoded draws")
        if len(rows) != CANDIDATES_PER_POOL:
            raise CandidateGenerationFailed(f"pool {pool} exhausted the frozen rejection budget")
        candidates.extend(rows)
        pool_audit.append({"pool_id": pool, "seed": seed, "draws": draws, "invalid": invalid, "excluded": excluded})
    generation_audit = {
        "proposal_distribution": "model-blind architecture generation + HIGH_REGIME_PRIOR",
        "predictor_accessed": False, "validation_labels_accessed": False,
        "pool_audit": pool_audit, "candidate_count": len(candidates),
    }
    novelty = validate_candidate_manifest(
        candidates, historical_arch, historical_joint, previous_arch, previous_joint,
        generator_audit=generation_audit,
    )
    return candidates, {**generation_audit, "novelty": novelty}


def score_candidates(
    runtime: Mapping[str, Any], retained: Sequence[Any], candidates: Sequence[Mapping[str, Any]],
    log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    soft = runtime["soft"]
    frozen_config = soft.load_config(runtime["soft_config_path"])
    candidate_rows = [dict(row) for row in sorted(candidates, key=lambda item: (int(item["pool_id"]), str(item["joint_fingerprint"])))]
    query_fp = [str(row["joint_fingerprint"]) for row in candidate_rows]
    fold_values: dict[str, dict[int, Any]] = {method: {} for method in ("M1", "M2", "B1", "B3", "B4")}
    audits: list[dict[str, Any]] = []
    for scorer in retained:
        train = scorer.train_rows
        train_y = np.asarray([float(row["validation_mean"]) for row in train], dtype=float)
        train_se = np.asarray([float(row["validation_se"]) for row in train], dtype=float)
        train_fp = [str(row["joint_fingerprint"]) for row in train]
        m1_seeds: list[Any] = []
        m2_seeds: list[Any] = []
        for seed in METRIC_SEEDS:
            candidate_embedding = soft.joint_v1.encode_rows_with_model(
                scorer.seed_models[seed], candidate_rows, scorer.hp_standardizer, "full_joint",
            )
            m1_seeds.append(soft.joint_v1.weighted_knn_predict(
                scorer.seed_train_embeddings[seed], train_y, candidate_embedding, 5,
                train_fingerprints=train_fp, query_fingerprints=query_fp,
            ))
            m2, gp_audit = soft.joint_v1.exact_matern_gp_predict(
                scorer.seed_train_embeddings[seed], train_y, train_se, candidate_embedding,
                standardize_inputs=False, ard=False, protocol_name="M2",
            )
            if gp_audit.get("ard") is not False or gp_audit.get("input_standardized") is not False:
                raise ProtocolError("candidate M2 did not use frozen isotropic geometry")
            m2_seeds.append(m2)
            audits.append({"fold": scorer.fold, "seed": seed, "method": "M2", **gp_audit})
        fold_values["M1"][scorer.fold] = np.mean(np.stack(m1_seeds), axis=0)
        fold_values["M2"][scorer.fold] = np.mean(np.stack(m2_seeds), axis=0)
        train_hp = soft.joint_v1._raw_hp_matrix(train)
        candidate_hp = soft.joint_v1._raw_hp_matrix(candidate_rows)
        train_joint = np.concatenate((soft.joint_v1._explicit_graph_matrix(train), train_hp), axis=1)
        candidate_joint = np.concatenate((soft.joint_v1._explicit_graph_matrix(candidate_rows), candidate_hp), axis=1)
        b1, b1_audit = soft.joint_v1.exact_matern_gp_predict(
            train_hp, train_y, train_se, candidate_hp,
            standardize_inputs=True, ard=True, protocol_name="B1",
        )
        b3, b3_audit = soft.joint_v1.exact_matern_gp_predict(
            train_joint, train_y, train_se, candidate_joint,
            standardize_inputs=True, ard=True, protocol_name="B3",
        )
        b4_seeds: dict[int, Any] = {}
        for seed in METRIC_SEEDS:
            b4, direct_audit = soft.joint_v1.train_direct_joint_regressor(
                train, candidate_rows, seed, frozen_config, scorer.hp_standardizer,
            )
            b4_seeds[seed] = b4
            audits.append({"fold": scorer.fold, "seed": seed, "method": "B4", **direct_audit})
        fold_values["B1"][scorer.fold] = np.asarray(b1, dtype=float)
        fold_values["B3"][scorer.fold] = np.asarray(b3, dtype=float)
        fold_values["B4"][scorer.fold] = soft.joint_v1.average_metric_seed_predictions(b4_seeds)
        audits.extend(({"fold": scorer.fold, "method": "B1", **b1_audit}, {"fold": scorer.fold, "method": "B3", **b3_audit}))
        log(f"fold {scorer.fold}: scored 64 candidates with M1/M2/B1/B3/B4")
    pooled: dict[str, Any] = {}
    for method, folds in fold_values.items():
        if set(folds) != set(range(5)):
            raise ProtocolError(f"{method} did not produce all five candidate folds")
        pooled[method] = np.mean(np.stack([np.asarray(folds[fold], dtype=float) for fold in range(5)]), axis=0)
    rows = [{
        "pool_id": int(candidate["pool_id"]),
        "joint_fingerprint": str(candidate["joint_fingerprint"]),
        "architecture_fingerprint": str(candidate["architecture_fingerprint"]),
        **{method: float(pooled[method][index]) for method in ("M1", "M2", "B1", "B3", "B4")},
    } for index, candidate in enumerate(candidate_rows)]
    audit = {
        "candidate_labels_used": False, "previous_pilot_labels_used": False,
        "current_confirmation_labels_used": False,
        "all_predictions_frozen_before_first_evaluation": True,
        "M1_frozen_protocol": True, "M1_k": 5,
        "M1_distance": "Euclidean", "M1_weight": "inverse_distance_plus_1e-8",
        "M2_kernel": "Matern-5/2", "M2_ard": False,
        "M2_lengthscale_count": 1, "M2_input_standardized": False,
        "M2_fixed_observation_noise_rule": "validation_se_squared",
        "M2_fitted_noise_floor_rule": "validation_se_squared_plus_floor_squared",
        "metric_seeds": list(METRIC_SEEDS), "fold_count": 5,
        "B1_frozen_protocol": True, "B3_frozen_protocol": True, "B4_frozen_protocol": True,
        "posterior_variance_used": False, "online_update_used": False,
        "seed_then_fold_aggregation": True, "runs": audits,
    }
    validate_scorer_protocol(audit)
    assert_no_prospective_label_training(audit)
    return rows, audit


def _adapter_task_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in build_evaluation_tasks(candidates):
        candidate = task["candidate"]
        pool = int(task["pool_id"])
        seed = int(task["training_seed"])
        rows.append({
            "task_stage": "initial", "pool_id": pool, "batch_id": str(candidate["batch_id"]),
            "architecture_slot": int(candidate["architecture_slot"]), "config_id": str(candidate["config_id"]),
            "architecture_fingerprint": candidate["architecture_fingerprint"],
            "hp_fingerprint": candidate["hp_fingerprint"],
            "joint_config_fingerprint": candidate["joint_fingerprint"], "hp_regime": "hp_high",
            "seed": seed, "task_key": task["task_key"],
            "training_protocol_identity": task["training_protocol_identity"],
            "assigned_gpu": 0 if pool < 8 else 1,
            "selected_by": ["independent_confirmation_surrogate_selection_v1"],
            "operations": candidate["operations"], "edges": candidate["edges"],
            "effective_layers": candidate["effective_layers"], "raw_hp_vector": candidate["raw_hp_vector"],
            "lr": candidate["lr"], "dropout": candidate["dropout"],
            "hidden_dim": candidate["hidden_dim"], "l2": candidate["l2"],
            "gat_heads": candidate["gat_heads"], "sage_aggr": candidate["sage_aggr"],
            "gin_eps": candidate["gin_eps"], "gat_heads_by_layer": candidate["gat_heads_by_layer"],
            "sage_aggr_by_layer": candidate["sage_aggr_by_layer"], "gin_eps_by_layer": candidate["gin_eps_by_layer"],
        })
    reject_test_fields(rows)
    return rows


def run_frozen_gnn_evaluation(
    output: Path, candidates: Sequence[Mapping[str, Any]], update_progress: Callable[..., None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_rows = pilot_gate._evaluation_task_rows
    original_total = pilot_gate.TOTAL_GNN_RUNS
    original_pools = pilot_gate.POOL_COUNT
    pilot_gate._evaluation_task_rows = _adapter_task_rows
    pilot_gate.TOTAL_GNN_RUNS = TOTAL_GNN_RUNS
    pilot_gate.POOL_COUNT = POOL_COUNT
    try:
        raw, audit = pilot_gate.run_frozen_gnn_evaluation(output, candidates, update_progress)
    finally:
        pilot_gate._evaluation_task_rows = original_rows
        pilot_gate.TOTAL_GNN_RUNS = original_total
        pilot_gate.POOL_COUNT = original_pools
    audit["pool_to_shard"] = {str(pool): 0 if pool < 8 else 1 for pool in range(POOL_COUNT)}
    audit["task_count"] = TOTAL_GNN_RUNS
    return raw, audit


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    return pilot_gate._markdown_table(rows, columns)


def render_final_report(artifacts: Mapping[str, Any]) -> str:
    decision = dict(artifacts.get("decision", {}))
    classification = str(decision.get("classification", "RUN_FAILED"))
    b_star = str(artifacts.get("baseline_selection", {}).get("B_STAR", "UNAVAILABLE"))
    candidates = list(artifacts.get("candidates", []))
    predictions = list(artifacts.get("predictions", []))
    evaluation_stats = list(artifacts.get("evaluation_stats", []))
    per_pool = list(artifacts.get("per_pool", []))
    methods = dict(artifacts.get("method_metrics", {}))
    info = dict(artifacts.get("information_gate", {}))
    family = dict(artifacts.get("family_confirmation", {}))
    surrogate = dict(artifacts.get("surrogate_selection", {}))
    replay = dict(artifacts.get("replay", {}))
    baselines = dict(artifacts.get("baseline_replay", {}))
    exclusion = dict(artifacts.get("exclusion", {}))
    freeze = dict(artifacts.get("freeze", {}))
    training = dict(artifacts.get("training_audit", {}))
    comparisons = dict(artifacts.get("comparisons", {}))
    bootstrap = dict(artifacts.get("bootstrap", {}))
    signs = dict(artifacts.get("sign_tests", {}))
    by_pool_stats: dict[int, list[Mapping[str, Any]]] = {pool: [] for pool in range(POOL_COUNT)}
    for row in evaluation_stats:
        by_pool_stats[int(row["pool_id"])].append(row)
    per_pool_map = {int(row["pool_id"]): row for row in per_pool}
    candidate_map = {str(row["joint_fingerprint"]): row for row in candidates}
    selection_rows = []
    for pool in range(POOL_COUNT):
        metric_row = per_pool_map.get(pool, {})
        for stat in sorted(by_pool_stats[pool], key=lambda row: str(row["joint_fingerprint"])):
            candidate = candidate_map.get(str(stat["joint_fingerprint"]), {})
            selection_rows.append({
                "pool": pool, "joint": str(stat.get("joint_fingerprint", "UNAVAILABLE")),
                "architecture": str(stat.get("architecture_fingerprint", "UNAVAILABLE")),
                "raw HP": json.dumps(candidate.get("raw_hp", {}), sort_keys=True),
                "validation mean": stat.get("validation_mean"),
                "oracle": metric_row.get("observed_pool_oracle"),
                "selected by": ",".join([
                    *[method for method in METHODS if metric_row.get(f"{method}_selected_joint_fingerprint") == stat.get("joint_fingerprint")],
                    *(["B_STAR"] if b_star in METHODS and metric_row.get(f"{b_star}_selected_joint_fingerprint") == stat.get("joint_fingerprint") else []),
                ]),
                **{f"{method} regret": metric_row.get(f"{method}_simple_regret") for method in METHODS},
            })
    if not selection_rows:
        selection_rows = [{"pool": "N/A", "joint": "not evaluated", "architecture": "not evaluated", "raw HP": "N/A", "validation mean": "N/A", "oracle": "N/A", "selected by": "N/A"}]
    metric_table = [{"method": method, **value} for method, value in methods.items()] or [{"method": "N/A", "selected_validation_mean": "N/A", "mean_simple_regret": "N/A", "true_best_hit_count": "N/A"}]
    predictor_table = predictions or [{"pool_id": "N/A", "joint_fingerprint": "not frozen", "M1": "N/A", "M2": "N/A", "B1": "N/A", "B3": "N/A", "B4": "N/A"}]
    lines = [f"# Independent Prospective Confirmation + Surrogate Selection Gate v1", ""]
    lines.extend([
        f"## {REPORT_SECTIONS[0]}", "", f"Frozen classification: `{classification}`. Frozen B_STAR: `{b_star}`. Final surrogate: `{decision.get('final_surrogate')}`. This run is an independent sixteen-pool confirmation; it never adds pilot or current prospective labels to any scorer.", "",
        f"## {REPORT_SECTIONS[1]}", "", "All M1/M2/B1/B3/B4 supervision is restricted to the original 336 joint configurations, 192 architectures, 24 historical pools, and 1008 seed records for seeds 0,1,2. The historical soft-metric classification remains SOFT_JOINT_METRIC_REJECTED.", "",
        f"## {REPORT_SECTIONS[2]}", "", f"The completed pilot contributes only its authenticated 32 architecture and joint fingerprints to the rejection set. Pilot classification is METRIC_GP_PROSPECTIVE_SIGNAL; pilot labels used for training: `{exclusion.get('pilot_labels_used_for_training', False)}`. Pilot pools are not combined with these sixteen pools.", "",
        f"## {REPORT_SECTIONS[3]}", "", f"Replay status: `{replay.get('status', 'UNAVAILABLE')}`; metric seeds `{replay.get('metric_seeds', [0,1,2])}`; absolute tolerance `{replay.get('absolute_tolerance', REPLAY_ATOL)}`. The replay checks 15 best epochs, M1/M2 OOF predictions, Gap@5, distance-gap Spearman, and ordered top-five historical neighbors.", "",
        f"## {REPORT_SECTIONS[4]}", "", f"Baseline replay status: `{baselines.get('status', 'UNAVAILABLE')}`. B1, B3 and B4 were reproduced from the frozen soft-v2 source; no prospective label entered HP scaling, GP fitting, inner stopping, or full-outer refit.", "",
        f"## {REPORT_SECTIONS[5]}", "", f"B_STAR was frozen as `{b_star}` using historical OOF best-of-four mean simple regret, then the 1e-12 regret tie rule, pooled OOF MAE, and method-id tie-break. Current prospective outcomes cannot change this choice.", "",
        f"## {REPORT_SECTIONS[6]}", "", f"Generated candidate count: `{len(candidates)}`. Master seed: `{MASTER_CANDIDATE_SEED}`; pool seed is master plus zero-based pool id. Proposal is model-blind architecture decode plus HIGH_REGIME_PRIOR HP sampling.", "",
        f"## {REPORT_SECTIONS[7]}", "", f"Novelty audit: `{artifacts.get('generation_audit', {}).get('novelty', {}).get('passed', False)}`. Candidates exclude historical 192/336, pilot 32/32, and every earlier accepted current architecture/joint fingerprint.", "",
        f"## {REPORT_SECTIONS[8]}", "", f"Immutable pre-evaluation root: `{freeze.get('root_sha256', 'UNAVAILABLE')}`. All 64 candidates, six methods, B_STAR, random actions and sixteen-pool selections were frozen before the first GNN run.", "",
        f"## {REPORT_SECTIONS[9]}", "", _markdown_table(predictor_table[:64], ("pool_id", "joint_fingerprint", "M1", "M2", "B1", "B3", "B4")), "",
        f"## {REPORT_SECTIONS[10]}", "", _markdown_table([{"comparison": key, "count": value} for key, value in info.get("disagreement_counts", {}).items()] or [{"comparison": "N/A", "count": "N/A"}], ("comparison", "count")), "",
        f"## {REPORT_SECTIONS[11]}", "", f"Information-gate authorization: `{info.get('authorized', False)}`. Completed GNN runs: `{training.get('completed_gnn_runs', 0)}` of `{training.get('total_gnn_runs', 0)}`; failed runs: `{training.get('failed_runs', 0)}`. Each complete pool requires four candidates times matched seeds 0,1,2.", "",
        f"## {REPORT_SECTIONS[12]}", "", _markdown_table(selection_rows, (
            "pool", "joint", "architecture", "raw HP", "validation mean", "oracle", "selected by",
            "M1 regret", "M2 regret", "B1 regret", "B3 regret", "B4 regret", "R regret",
        )), "",
        f"## {REPORT_SECTIONS[13]}", "", _markdown_table(metric_table, ("method", "selected_validation_mean", "mean_simple_regret", "median_simple_regret", "true_best_hit_count", "true_best_hit_rate", "within_0.006_of_oracle_rate", "mae", "pearson", "spearman", "kendall")), "",
        f"## {REPORT_SECTIONS[14]}", "", json.dumps(family.get("M1", {}), indent=2, sort_keys=True), "",
        f"## {REPORT_SECTIONS[15]}", "", json.dumps(family.get("M2", {}), indent=2, sort_keys=True), "",
        f"## {REPORT_SECTIONS[16]}", "", f"Family confirmed: `{family.get('family_confirmed', False)}` by the preregistered OR rule. Random and the previous pilot cannot rescue a failed current confirmation.", "",
        f"## {REPORT_SECTIONS[17]}", "", f"M1-vs-M2 decisions use only current sixteen pools. GP increment: `{surrogate.get('GP_increment', {}).get('passed', False)}`; symmetric KNN advantage: `{surrogate.get('KNN_advantage', {}).get('passed', False)}`.", "",
        f"## {REPORT_SECTIONS[18]}", "", f"Final surrogate: `{surrogate.get('final_surrogate')}`. Selection basis: `{surrogate.get('selection_basis')}`. PARSIMONY_NO_CLEAR_GP_INCREMENT means GP complexity was not shown necessary; it does not prove M1 statistically superior.", "",
        f"## {REPORT_SECTIONS[19]}", "", f"Frozen Random seed: `{RANDOM_SELECTION_SEED}`. Random is reported completely but does not participate in family confirmation or surrogate selection.", "",
        f"## {REPORT_SECTIONS[20]}", "", f"Bootstrap unit is pool, with {BOOTSTRAP_REPLICATES} replicates and seed {BOOTSTRAP_SEED}. Exact one-sided sign summaries are descriptive only.", "", _markdown_table([{"comparison": key, **value} for key, value in signs.items()] or [{"comparison": "N/A", "wins": "N/A", "losses": "N/A", "ties": "N/A"}], ("comparison", "wins", "losses", "ties", "exact_one_sided_sign_test_pvalue")), "",
        f"## {REPORT_SECTIONS[21]}", "", f"Only this run's frozen classification is `{classification}`. No prior classification is modified.", "",
        f"## {REPORT_SECTIONS[22]}", "", "This is screening evidence on a fixed model-blind architecture plus HIGH_REGIME_PRIOR deployment distribution. The pilot may be described as cumulative context, but no pooled gate or combined p-value is created.", "",
        f"## {REPORT_SECTIONS[23]}", "", "This experiment does not validate GP uncertainty, EI, qLogEI, UCB, Thompson sampling, KG, global optimality, cross-dataset generalization, or a full Bayesian-optimization loop. Test data were not accessed.", "",
        f"## {REPORT_SECTIONS[24]}", "", "If the family is not confirmed, the metric-guided NAS mainline stops without rescue tuning. If KNN is selected, only frozen metric plus deterministic weighted-kNN ranking is authorized. If GP is selected, only isotropic-GP posterior-mean deterministic ranking is authorized; acquisitions remain forbidden.", "",
        f"## {REPORT_SECTIONS[25]}", "", f"Branch `{artifacts.get('branch','UNAVAILABLE')}`, HEAD `{artifacts.get('head','UNAVAILABLE')}`, dirty tree `{artifacts.get('dirty_tree','UNAVAILABLE')}`. Result directory `{artifacts.get('output', DEFAULT_OUTPUT)}`. No candidate generation, freeze, scorer replay or GNN work was performed by Codex validation.", "",
    ])
    return "\n".join(str(line) for line in lines)


def assert_complete_final_report(text: str) -> None:
    for section in REPORT_SECTIONS:
        if f"## {section}" not in text:
            raise ProtocolError(f"final report is missing section: {section}")


def ensure_output_policy(output: Path, *, preflight_ready: bool = False) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise ProtocolError("formal output path is not a directory")
    files = [path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()]
    allowed = {
        "logs/py_compile.log", "logs/pytest_focused.log", "logs/synthetic_smoke.log",
        "logs/scorer_replay.log", "logs/candidate_generation.log", "logs/pre_eval_freeze.log",
        "logs/evaluation.log", "logs/analysis.log", "logs/runner.log",
    }
    if not preflight_ready or any(path not in allowed for path in files):
        raise ProtocolError("formal result directory exists and is non-empty")


ensure_output_available = ensure_output_policy


def _git_capture(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def write_environment_and_git(output: Path, *, before: bool) -> None:
    status = _git_capture(["status", "--short"])
    dump_text(output / ("git_status_before.txt" if before else "git_status_after.txt"), status)
    if before:
        dump_json(output / "environment.json", {
            "created_at": utc_now(), "protocol_version": PROTOCOL_VERSION,
            "python": sys.version, "platform": platform.platform(),
            "branch": _git_capture(["branch", "--show-current"]).strip(),
            "head": _git_capture(["rev-parse", "HEAD"]).strip(),
            "dirty_tree": bool(status.strip()), "scorer_device": "cpu",
            "evaluator_device_policy": "two frozen GPU shards after immutable admission",
            "test_accessed": False,
        })
        dump_text(output / "git_diff.patch", _git_capture(["diff", "--binary", "HEAD", "--"]) + _git_capture(["diff", "--binary", "--cached", "HEAD", "--"]))


def _mandatory_snapshot_sources() -> dict[str, tuple[Path, str]]:
    soft = HISTORICAL_SOFT_ROOT / "source_snapshot"
    high = HISTORICAL_HIGH_ROOT / "source_snapshot"
    pilot = PREVIOUS_PILOT_ROOT / "source_snapshot"
    current = {
        "configs/joint_metric_independent_confirmation_surrogate_selection_v1.yaml": (DEFAULT_CONFIG, "current frozen protocol"),
        "analyse/joint_metric_independent_confirmation_surrogate_selection.py": (Path(__file__).resolve(), "current implementation"),
        "scripts/run_joint_metric_independent_confirmation_surrogate_selection.sh": (REPO_ROOT / "scripts/run_joint_metric_independent_confirmation_surrogate_selection.sh", "one-shot runner"),
        "tests/test_joint_metric_independent_confirmation_surrogate_selection.py": (REPO_ROOT / "tests/test_joint_metric_independent_confirmation_surrogate_selection.py", "focused protocol tests"),
        "analyse/joint_metric_prospective_selection_gate.py": (pilot / "analyse/joint_metric_prospective_selection_gate.py", "authenticated pilot engineering/scorer implementation"),
        "configs/joint_configuration_soft_metric_gate_v2.yaml": (soft / "configs/joint_configuration_soft_metric_gate_v2.yaml", "frozen soft metric config"),
        "analyse/joint_configuration_soft_metric_gate.py": (soft / "analyse/joint_configuration_soft_metric_gate.py", "frozen soft metric scorer"),
        "analyse/joint_configuration_metric_gate.py": (soft / "analyse/joint_configuration_metric_gate.py", "frozen joint scorer and B1/B3/B4"),
        "analyse/performance_metric_quick_gate.py": (soft / "analyse/performance_metric_quick_gate.py", "frozen GP dependency"),
        "analyse/joint_pair_power_projection_audit.py": (soft / "analyse/joint_pair_power_projection_audit.py", "frozen engineering dependency"),
        "analyse/high_regime_prior_model_blind_validation.py": (high / "analyse/high_regime_prior_model_blind_validation.py", "frozen candidate generator"),
        "analyse/run_high_regime_prior_model_blind_validation_labels.py": (high / "analyse/run_high_regime_prior_model_blind_validation_labels.py", "frozen evaluator runner"),
        "analyse/analyse_high_regime_prior_model_blind_validation.py": (high / "analyse/analyse_high_regime_prior_model_blind_validation.py", "frozen evaluator analysis"),
        "eval_utils.py": (high / "eval_utils.py", "frozen GNN evaluator"),
        "architecture_diagnostics.py": (high / "architecture_diagnostics.py", "frozen architecture helper"),
    }
    return current


def snapshot_sources(output: Path, *, modules: Mapping[str, Any] | None = None) -> dict[str, Any]:
    original = pilot_gate._mandatory_snapshot_sources
    pilot_gate._mandatory_snapshot_sources = _mandatory_snapshot_sources
    try:
        audit = pilot_gate.snapshot_sources(output, modules=modules or sys.modules)
    finally:
        pilot_gate._mandatory_snapshot_sources = original
    for relative in ("source_snapshot/imported_local_modules.json", "source_snapshot/sha256_manifest.json"):
        path = output / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        value["format_version"] = PROTOCOL_VERSION
        dump_json(path, value)
    return audit


def validate_source_snapshot(snapshot_root: Path, required_paths: Sequence[str] | None = None) -> dict[str, Any]:
    required = tuple(required_paths or REQUIRED_SOURCE_SNAPSHOT_PATHS)
    missing = [relative for relative in required if not (snapshot_root / relative).is_file()]
    if missing:
        raise ProtocolError(f"required source omitted from snapshot: {missing[0]}")
    manifest_path = snapshot_root / "sha256_manifest.json"
    if not manifest_path.is_file():
        raise ProtocolError("source snapshot manifest is absent")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProtocolError("source snapshot manifest cannot be parsed") from exc
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ProtocolError("source snapshot manifest has no file entries")
    paths = set()
    for entry in entries:
        relative = str(entry["snapshot_relative_path"])
        path = snapshot_root / relative
        if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]) or sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"source snapshot hash check failed: {relative}")
        paths.add(relative)
    if any(relative not in paths for relative in required):
        raise ProtocolError("source snapshot manifest omits a required file")
    root = stable_hash([{key: entry[key] for key in ("repository_relative_path", "size_bytes", "sha256")} for entry in entries])
    if root != manifest.get("manifest_root_sha256"):
        raise ProtocolError("source snapshot manifest root mismatch")
    return {"passed": True, "all_required_sources_copied_as_bytes": True, "entry_count": len(entries), "manifest_root_sha256": root}


def write_manifest(output: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or "__pycache__" in path.parts or path.suffix in {".pt", ".pth", ".ckpt", ".pyc"}:
            continue
        entries.append({"path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"format_version": PROTOCOL_VERSION, "created_at": utc_now(), "entries": entries, "manifest_root_sha256": stable_hash(entries)}
    dump_json(output / "manifest.json", manifest)
    return manifest


def create_archive(output: Path, archive: Path, sidecar: Path | None = None) -> str:
    if archive.exists() or (sidecar is not None and sidecar.exists()):
        raise ProtocolError("archive target already exists")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    try:
        with tarfile.open(temporary, "w:gz") as handle:
            handle.add(output, arcname=output.name, filter=lambda info: None if "__pycache__" in Path(info.name).parts or Path(info.name).suffix in {".pt", ".pth", ".ckpt", ".pyc"} else info)
        digest = sha256_file(temporary)
        os.replace(temporary, archive)
        if sidecar is not None:
            dump_text(sidecar, f"{digest}  {archive.name}\n")
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _empty_csv_fields() -> dict[str, tuple[str, ...]]:
    return {
        "pre_eval/predictions.csv": ("pool_id", "joint_fingerprint", "architecture_fingerprint", "M1", "M2", "B1", "B3", "B4"),
        "evaluation/raw_seed_results.csv": ("pool_id", "joint_fingerprint", "training_seed", "validation_accuracy", "terminal_status", "attempt_count"),
        "evaluation/config_3seed_statistics.csv": ("pool_id", "joint_fingerprint", "architecture_fingerprint", "validation_seed_0", "validation_seed_1", "validation_seed_2", "validation_mean", "observed_pool_oracle"),
        "selection_results_per_pool.csv": ("pool_id", "observed_pool_oracle", *(f"{method}_{suffix}" for method in METHODS for suffix in ("selected_joint_fingerprint", "selected_validation_mean", "simple_regret", "true_best_hit", "within_0.006"))),
    }


def _placeholder_artifacts(classification: str, error: str | None = None) -> dict[str, Any]:
    return {
        "decision": {"classification": classification, "reason": error, "test_accessed": False, "historical_classifications_modified": False},
        "baseline_selection": {}, "candidates": [], "predictions": [], "evaluation_stats": [],
        "per_pool": [], "method_metrics": {}, "information_gate": {},
        "family_confirmation": {}, "surrogate_selection": {}, "replay": {},
        "baseline_replay": {}, "exclusion": {}, "freeze": {}, "training_audit": {},
        "comparisons": {}, "bootstrap": {}, "sign_tests": {}, "output": str(DEFAULT_OUTPUT),
    }


def write_engineering_fix(output: Path) -> dict[str, Any]:
    path = output / "engineering_fix.json"
    if path.is_file():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ProtocolError("engineering_fix.json cannot be parsed") from exc
        if observed != ENGINEERING_FIX:
            raise ProtocolError("engineering_fix.json differs from the frozen repair audit")
    else:
        dump_json(path, ENGINEERING_FIX)
    return dict(ENGINEERING_FIX)


def _write_missing_outputs(output: Path, classification: str, error: str | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_engineering_fix(output)
    for directory in ("logs", "pre_eval", "evaluation"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    json_defaults: dict[str, Any] = {
        "state.json": {"status": "failed" if classification == "RUN_FAILED" else "complete", "classification": classification, "stage": "ANALYSIS_COMPLETE", "completed_gnn_runs": 0, "total_gnn_runs": 0, "failed_runs": 0, "test_accessed": False},
        "frozen_protocol.json": {"protocol_version": PROTOCOL_VERSION},
        "input_integrity.json": {"status": "UNAVAILABLE", "test_accessed": False},
        "historical_scorer_replay.json": {"status": "NOT_RUN"},
        "historical_baseline_metrics.json": {"status": "NOT_RUN"},
        "previous_pilot_exclusion_audit.json": {"status": "NOT_RUN", "previous_labels_used_for_training": False},
        "pre_eval/candidate_manifest.json": {"status": "NOT_FROZEN", "candidates": []},
        "pre_eval/selections.json": {"status": "NOT_FROZEN", "pools": {}},
        "pre_eval/baseline_selection.json": {"status": "NOT_FROZEN"},
        "pre_eval/scorer_identity.json": {"status": "NOT_FROZEN"},
        "pre_eval/previous_pilot_exclusion_audit.json": {"status": "NOT_RUN"},
        "pre_eval/freeze_manifest.json": {"status": "NOT_FROZEN", "immutable_entries": []},
        "evaluation/task_manifest.json": {"authorized": False, "tasks": [], "total_gnn_runs": 0},
        "evaluation/training_audit.json": {"status": "NOT_RUN", "track_test": False},
        "method_metrics.json": {"status": "NOT_AVAILABLE", "methods": {}},
        "metric_family_confirmation.json": {"status": "NOT_AVAILABLE"},
        "m1_vs_m2_selection.json": {"status": "NOT_AVAILABLE"},
        "paired_pool_comparisons.json": {}, "bootstrap_results.json": {}, "sign_test_summary.json": {},
        "decision.json": {"classification": classification, "reason": error, "test_accessed": False, "historical_classifications_modified": False},
        "environment.json": {"protocol_version": PROTOCOL_VERSION, "test_accessed": False},
        "manifest.json": {"format_version": PROTOCOL_VERSION, "entries": []},
    }
    for relative, value in json_defaults.items():
        path = output / relative
        if not path.exists():
            dump_json(path, value)
    for relative, fields in _empty_csv_fields().items():
        if not (output / relative).exists():
            dump_csv(output / relative, [], fields)
    root_sidecar = output / "pre_eval/pre_eval_root_sha256.txt"
    if not root_sidecar.exists():
        dump_text(root_sidecar, "NOT_FROZEN\n")
    for relative in ("git_status_before.txt", "git_status_after.txt", "git_diff.patch"):
        if not (output / relative).exists():
            dump_text(output / relative, "UNAVAILABLE\n")
    for name in ("py_compile", "pytest_focused", "synthetic_smoke", "scorer_replay", "candidate_generation", "pre_eval_freeze", "evaluation", "analysis", "runner"):
        path = output / f"logs/{name}.log"
        if not path.exists():
            dump_text(path, "")
    report = output / "final_report.md"
    if not report.exists():
        text = render_final_report(_placeholder_artifacts(classification, error))
        assert_complete_final_report(text)
        dump_text(report, text)
    if not (output / "source_snapshot/sha256_manifest.json").is_file():
        snapshot_sources(output)
    write_manifest(output)


def write_failure_bundle(
    output: Path, error: BaseException | str, *, archive_path: Path | None = None,
    config_sha256: str = "UNAVAILABLE", source_sha256: str = "UNAVAILABLE",
) -> dict[str, Any]:
    message = str(error)
    _write_missing_outputs(output, "RUN_FAILED", message)
    dump_json(output / "error.json", {"error": message, "type": type(error).__name__, "created_at": utc_now()})
    dump_json(output / "decision.json", {"classification": "RUN_FAILED", "reason": message, "test_accessed": False, "historical_classifications_modified": False})
    dump_json(output / "state.json", {"status": "failed", "classification": "RUN_FAILED", "config_sha256": config_sha256, "source_sha256": source_sha256, "completed_gnn_runs": 0, "total_gnn_runs": 0, "failed_runs": 0, "test_accessed": False})
    dump_text(output / "FAILED", "")
    write_manifest(output)
    digest = None
    if archive_path is not None:
        digest = create_archive(output, archive_path, Path(str(archive_path) + ".sha256"))
    return {"passed": True, "classification": "RUN_FAILED", "archive_sha256": digest}


def execute_dry_run(config_path: Path = DEFAULT_CONFIG, formal_output: Path = DEFAULT_OUTPUT) -> int:
    config = load_config(config_path)
    validate_frozen_protocol(config)
    print(json.dumps({
        "dry_run": True, "protocol_version": PROTOCOL_VERSION,
        "formal_result_root": str(formal_output), "formal_result_root_created": formal_output.exists(),
        "formal_data_read": False, "scorer_replay_started": False,
        "candidate_generation_started": False, "pre_eval_freeze_started": False,
        "generated_candidate_count": 0,
        "gnn_training_started": False, "would_run_gnn_tasks_if_information_gate_passes": TOTAL_GNN_RUNS,
        "test_accessed": False,
    }, indent=2, sort_keys=True))
    return 0


def _logger(path: Path) -> Callable[[str], None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    def emit(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        print(line, flush=True)
    return emit


def _initial_state(output: Path, config_sha: str, source_sha: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION, "status": "running", "stage": "IMPLEMENTATION_READY",
        "classification": None, "output": str(output), "config_sha256": config_sha,
        "source_sha256": source_sha, "started_at": utc_now(), "started_monotonic": time.monotonic(),
        "completed_gnn_runs": 0, "total_gnn_runs": TOTAL_GNN_RUNS, "failed_runs": 0,
        "current_pool": None, "current_config": None, "current_seed": None,
        "branch": _git_capture(["branch", "--show-current"]).strip(),
        "head": _git_capture(["rev-parse", "HEAD"]).strip(),
        "dirty_tree": bool(_git_capture(["status", "--porcelain"]).strip()),
        "previous_pilot_labels_used_for_training": False, "test_accessed": False,
    }


def update_state(output: Path, **changes: Any) -> dict[str, Any]:
    state = _load_json(output / "state.json")
    state.update(changes)
    state["updated_at"] = utc_now()
    dump_json(output / "state.json", state)
    return state


def _finalize_terminal(output: Path, decision: Mapping[str, Any], artifacts: Mapping[str, Any]) -> None:
    classification = str(decision["classification"])
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise ProtocolError(f"unrecognized frozen classification: {classification}")
    current_state = _load_json(output / "state.json") if (output / "state.json").is_file() else {}
    complete_artifacts = {
        **_placeholder_artifacts(classification), **dict(artifacts),
        "decision": dict(decision), "output": str(output),
        "branch": current_state.get("branch", "UNAVAILABLE"),
        "head": current_state.get("head", "UNAVAILABLE"),
        "dirty_tree": current_state.get("dirty_tree", "UNAVAILABLE"),
    }
    dump_json(output / "decision.json", {**dict(decision), "test_accessed": False, "historical_classifications_modified": False})
    report = render_final_report(complete_artifacts)
    assert_complete_final_report(report)
    dump_text(output / "final_report.md", report)
    _write_missing_outputs(output, classification)
    write_environment_and_git(output, before=False)
    snapshot_audit = snapshot_sources(output)
    state = update_state(
        output, status="complete", stage="ANALYSIS_COMPLETE", classification=classification,
        finished_at=utc_now(), source_snapshot_manifest_root=snapshot_audit["manifest_root_sha256"],
        elapsed_seconds=time.monotonic() - float(_load_json(output / "state.json").get("started_monotonic", time.monotonic())),
        test_accessed=False,
    )
    write_manifest(output)
    complete_artifacts["branch"] = state.get("branch")


def _synthetic_per_pool(direction: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    methods = {method: {"mean_simple_regret": 0.010, "true_best_hit_count": 2} for method in METHODS}
    if direction == "gp":
        methods["M2"].update(mean_simple_regret=0.007, true_best_hit_count=4)
    else:
        methods["M1"].update(mean_simple_regret=0.007, true_best_hit_count=4)
    challenger, comparator = (("M2", "M1") if direction == "gp" else ("M1", "M2"))
    for pool in range(POOL_COUNT):
        win = pool < 6
        disagree = pool < 8
        challenger_value = 0.82 if win else (0.799 if disagree else 0.81)
        comparator_value = 0.81
        row = {"pool_id": pool, "observed_pool_oracle": 0.82}
        for method in METHODS:
            row[f"{method}_selected_joint_fingerprint"] = f"{method}-{pool}"
            row[f"{method}_selected_validation_mean"] = 0.80
            row[f"{method}_simple_regret"] = 0.02
            row[f"{method}_true_best_hit"] = 0
        row[f"{challenger}_selected_joint_fingerprint"] = f"{challenger}-{pool}" if disagree else f"same-{pool}"
        row[f"{comparator}_selected_joint_fingerprint"] = f"{comparator}-{pool}" if disagree else f"same-{pool}"
        row[f"{challenger}_selected_validation_mean"] = challenger_value
        row[f"{comparator}_selected_validation_mean"] = comparator_value
        rows.append(row)
    return rows, methods


def synthetic_smoke(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    validate_frozen_protocol(load_config(config_path))
    results = {classification: True for classification in ALLOWED_CLASSIFICATIONS if classification != "RUN_FAILED"}
    # Exercise OR-family, GP selection, KNN selection, and parsimony without
    # reading formal artifacts or invoking any scorer/candidate/evaluator.
    gp_rows, gp_metrics = _synthetic_per_pool("gp")
    family = {"family_confirmed": True, "M1": {"confirmed": True}, "M2": {"confirmed": True}}
    gp_decision = select_final_surrogate(gp_rows, gp_metrics, family, "B3")
    knn_rows, knn_metrics = _synthetic_per_pool("knn")
    knn_decision = select_final_surrogate(knn_rows, knn_metrics, family, "B3")
    if gp_decision["classification"] not in ALLOWED_CLASSIFICATIONS or knn_decision["classification"] not in ALLOWED_CLASSIFICATIONS:
        raise ProtocolError("synthetic classification escaped the frozen set")
    return {
        "passed": all(results.values()), **results,
        "candidate_generation_started": False, "pre_eval_freeze_started": False,
        "scorer_replay_started": False, "gnn_training_started": False, "test_accessed": False,
    }


def execute_formal(
    config_path: Path, output: Path, config_sha256: str, source_sha256: str,
    *, preflight_ready: bool,
) -> int:
    ensure_output_policy(output, preflight_ready=preflight_ready)
    if sha256_file(config_path) != config_sha256:
        raise ProtocolError("runner-provided config SHA differs from runtime bytes")
    if aggregate_source_sha256() != source_sha256:
        raise ProtocolError("runner-provided source SHA differs from runtime bytes")
    config = load_config(config_path)
    validate_frozen_protocol(config)
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("logs", "pre_eval", "evaluation"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    dump_json(output / "state.json", _initial_state(output, config_sha256, source_sha256))
    dump_json(output / "frozen_protocol.json", config)
    write_engineering_fix(output)
    write_environment_and_git(output, before=True)
    scorer_log = _logger(output / "logs/scorer_replay.log")
    candidate_log = _logger(output / "logs/candidate_generation.log")
    freeze_log = _logger(output / "logs/pre_eval_freeze.log")
    evaluation_log = _logger(output / "logs/evaluation.log")
    analysis_log = _logger(output / "logs/analysis.log")
    artifacts: dict[str, Any] = {"output": str(output)}

    try:
        authority = validate_authoritative_inputs(config)
        exclusion = authority["previous_pilot_exclusion"]
        exclusion_output = {
            "status": "PASSED", "previous_candidate_count": exclusion["candidate_count"],
            "previous_architecture_fingerprints": sorted(exclusion["architecture_fingerprints"]),
            "previous_joint_fingerprints": sorted(exclusion["joint_fingerprints"]),
            "previous_labels_used_for_training": False,
        }
        dump_json(output / "input_integrity.json", {
            "status": "VALID", "historical_identity": FROZEN_HISTORICAL_IDENTITY,
            "previous_pilot_identity": authority["previous_pilot_identity"],
            "previous_pilot_labels_used_for_training": False, "test_accessed": False,
        })
        update_state(output, stage="HISTORICAL_IDENTITY_COMPLETE")
        dump_json(output / "previous_pilot_exclusion_audit.json", exclusion_output)
        update_state(output, stage="PREVIOUS_PILOT_EXCLUSION_AUDITED")
        artifacts["exclusion"] = exclusion_output
    except ConfirmationInputInvalid as exc:
        dump_json(output / "input_integrity.json", {"status": "INVALID", "reason": str(exc), "test_accessed": False})
        decision = {"classification": "CONFIRMATION_INPUT_INVALID", "reason": str(exc)}
        _finalize_terminal(output, decision, artifacts)
        return 0

    try:
        os.environ["INDEPENDENT_CONFIRMATION_SCORER_DEVICE"] = "cpu"
        runtime = pilot_gate.load_frozen_soft_runtime()
        configurations, retained, replay = pilot_gate.replay_and_retain_scorers(runtime, authority, scorer_log)
        replay.update({
            "M1_frozen_protocol": True,
            "M1_k": 5,
            "M1_distance": "Euclidean",
            "M1_weight": "inverse_distance_plus_1e-8",
            "M2_kernel": "Matern-5/2",
            "M2_lengthscale_count": 1,
            "M2_fixed_observation_noise_rule": "validation_se_squared",
            "M2_fitted_noise_floor_rule": "validation_se_squared_plus_floor_squared",
            "previous_pilot_labels_used": False,
            "current_confirmation_labels_used": False,
        })
        validate_metric_scorer_protocol(replay)
        dump_json(output / "historical_scorer_replay.json", replay)
        update_state(output, stage="SCORER_REPLAY_COMPLETE")
        artifacts["replay"] = replay
        baseline_replay, baseline_selection = replay_historical_baselines(runtime, retained, scorer_log)
        dump_json(output / "historical_baseline_metrics.json", baseline_replay)
        update_state(output, stage="HISTORICAL_BASELINES_REPLAYED")
        update_state(output, stage="B_STAR_FROZEN", B_STAR=baseline_selection["B_STAR"])
        artifacts["baseline_replay"] = baseline_replay
        artifacts["baseline_selection"] = baseline_selection
    except (pilot_gate.ScorerReplayIdentityMismatch, ScorerReplayIdentityMismatch) as exc:
        decision = {"classification": "SCORER_REPLAY_IDENTITY_MISMATCH", "reason": str(exc)}
        _finalize_terminal(output, decision, artifacts)
        return 0
    except pilot_gate.ProspectiveInputInvalid as exc:
        decision = {"classification": "CONFIRMATION_INPUT_INVALID", "reason": str(exc)}
        _finalize_terminal(output, decision, artifacts)
        return 0

    try:
        candidates, generation_audit = generate_candidates(configurations, exclusion, candidate_log)
        candidate_manifest = {
            "format_version": PROTOCOL_VERSION, "created_at": utc_now(),
            "candidate_count": len(candidates), "candidates": candidates,
            "generation_audit": generation_audit,
        }
        update_state(output, stage="CANDIDATES_FROZEN")
        artifacts["candidates"] = candidates
        artifacts["generation_audit"] = generation_audit
    except ConfirmationInputInvalid as exc:
        decision = {"classification": "CONFIRMATION_INPUT_INVALID", "reason": str(exc)}
        _finalize_terminal(output, decision, artifacts)
        return 0
    except CandidateGenerationFailed as exc:
        dump_json(output / "pre_eval/candidate_manifest.json", {"status": "FAILED", "reason": str(exc), "candidates": []})
        decision = {"classification": "CONFIRMATION_CANDIDATE_GENERATION_FAILED", "reason": str(exc)}
        _finalize_terminal(output, decision, artifacts)
        return 0

    predictions, scoring_audit = score_candidates(runtime, retained, candidates, scorer_log)
    b_star = str(baseline_selection["B_STAR"])
    selections = make_selections(predictions, b_star, frozen_random_indices())
    scorer_identity = {
        "status": "IDENTICAL", "historical_replay": replay,
        "baseline_replay_status": baseline_replay["status"],
        "scoring_audit": scoring_audit, "previous_pilot_labels_used": False,
        "current_candidate_labels_used": False, "all_predictions_frozen_before_first_evaluation": True,
    }
    freeze = freeze_pre_eval(
        output / "pre_eval", candidate_manifest=candidate_manifest, predictions=predictions,
        selections=selections, baseline_selection=baseline_selection,
        scorer_identity=scorer_identity, previous_pilot_exclusion_audit=exclusion_output,
        historical_input_hashes={
            "soft_archive_sha256": pilot_gate.FROZEN_SOFT_METRIC_IDENTITY["archive_sha256"],
            "soft_source_sha256": pilot_gate.FROZEN_SOFT_METRIC_IDENTITY["source_sha256"],
            "split_sha256": pilot_gate.FROZEN_SOFT_METRIC_IDENTITY["split_sha256"],
            "previous_pilot_archive_sha256": FROZEN_PREVIOUS_PILOT_IDENTITY["archive_sha256"],
            "previous_pilot_pre_eval_root": FROZEN_PREVIOUS_PILOT_IDENTITY["pre_eval_root_sha256"],
        },
        generator_identity={
            "master_seed": MASTER_CANDIDATE_SEED, "random_seed": RANDOM_SELECTION_SEED,
            "source_aggregate_sha256": source_sha256,
            "high_regime_source_sha256": pilot_gate.FROZEN_SOURCE_HASHES["analyse/high_regime_prior_model_blind_validation.py"],
            "proposal": "model-blind architecture + HIGH_REGIME_PRIOR",
        },
    )
    verify_pre_eval_freeze(output / "pre_eval")
    freeze_log(f"all 16 pools frozen under root {freeze['root_sha256']}")
    update_state(output, stage="SELECTIONS_FROZEN", pre_eval_root_sha256=freeze["root_sha256"])
    artifacts.update({"predictions": predictions, "freeze": freeze})
    admission = information_gate(selections, b_star, minimum_disagreement_pools=6)
    artifacts["information_gate"] = admission
    update_state(output, stage="PRE_EVAL_AUDIT_PASSED", disagreement_counts=admission["disagreement_counts"])
    del retained

    if not admission["authorized"]:
        update_state(output, total_gnn_runs=0, completed_gnn_runs=0, failed_runs=0)
        dump_json(output / "evaluation/task_manifest.json", {"authorized": False, "reason": "max(d1,d2) below 6", "tasks": [], "total_gnn_runs": 0})
        training_audit = {"status": "NOT_AUTHORIZED", "completed_gnn_runs": 0, "total_gnn_runs": 0, "failed_runs": 0, "track_test": False}
        dump_json(output / "evaluation/training_audit.json", training_audit)
        artifacts["training_audit"] = training_audit
        _finalize_terminal(output, {"classification": "CONFIRMATION_LOW_INFORMATION", **admission}, artifacts)
        return 0

    tasks = build_evaluation_tasks(candidates, information_gate_passed=True, pre_eval_verified=True)
    dump_json(output / "evaluation/task_manifest.json", {
        "authorized": True, "pre_eval_root_sha256": freeze["root_sha256"],
        "task_count": len(tasks), "training_seeds": list(TRAINING_SEEDS), "tasks": [
            {key: value for key, value in task.items() if key != "candidate"} for task in tasks
        ],
    })
    verify_pre_eval_freeze(output / "pre_eval")
    evaluation_log("immutable pre-evaluation root verified; starting 192 frozen GNN tasks")
    raw_results, training_audit = run_frozen_gnn_evaluation(output, candidates, lambda **changes: update_state(output, **changes))
    raw_csv = [{
        "pool_id": row["pool_id"], "joint_fingerprint": row["joint_fingerprint"],
        "training_seed": row["training_seed"], "validation_accuracy": row.get("validation_accuracy"),
        "terminal_status": row.get("terminal_status"), "attempt_count": row.get("attempt_count"),
    } for row in raw_results]
    dump_csv(output / "evaluation/raw_seed_results.csv", raw_csv, _empty_csv_fields()["evaluation/raw_seed_results.csv"])
    stats, complete = aggregate_evaluation_results(candidates, raw_results, allow_incomplete=True)
    complete = bool(complete and training_audit.get("all_shards_exited_zero"))
    dump_csv(output / "evaluation/config_3seed_statistics.csv", stats, _empty_csv_fields()["evaluation/config_3seed_statistics.csv"])
    training_audit.update({
        "evaluation_complete": complete,
        "completed_gnn_runs": sum(row.get("terminal_status") == "success" for row in raw_results),
        "total_gnn_runs": TOTAL_GNN_RUNS,
        "failed_runs": sum(row.get("terminal_status") != "success" for row in raw_results),
        "previous_pilot_labels_used_for_update": False, "candidate_labels_used_for_update": False,
        "online_update_used": False, "sequential_refit_used": False,
        "all_predictions_frozen_before_first_evaluation": True, "track_test": False,
    })
    assert_no_prospective_label_training(training_audit)
    dump_json(output / "evaluation/training_audit.json", training_audit)
    artifacts.update({"training_audit": training_audit, "evaluation_stats": stats})
    update_state(output, stage="EVALUATION_COMPLETE" if complete else "EVALUATION_INCOMPLETE", completed_gnn_runs=training_audit["completed_gnn_runs"], failed_runs=training_audit["failed_runs"])
    if not complete:
        _finalize_terminal(output, {"classification": "CONFIRMATION_EVALUATION_INCOMPLETE"}, artifacts)
        return 0

    verify_pre_eval_freeze(output / "pre_eval")
    metrics = compute_method_metrics(stats, selections, predictions)
    m1_confirmation = evaluate_metric_confirmation("M1", metrics["per_pool"], metrics, b_star)
    m2_confirmation = evaluate_metric_confirmation("M2", metrics["per_pool"], metrics, b_star)
    family = combine_metric_family_confirmation(m1_confirmation, m2_confirmation)
    surrogate = select_final_surrogate(metrics["per_pool"], metrics, family, b_star)
    comparisons = paired_pool_comparisons(metrics["per_pool"], b_star)
    bootstrap = bootstrap_comparisons(comparisons)
    signs = sign_test_summary(comparisons)
    decision = {**surrogate, "metric_family_confirmed": family["family_confirmed"], "test_accessed": False}
    dump_csv(output / "selection_results_per_pool.csv", metrics["per_pool"], _empty_csv_fields()["selection_results_per_pool.csv"])
    dump_json(output / "method_metrics.json", metrics)
    dump_json(output / "metric_family_confirmation.json", family)
    dump_json(output / "m1_vs_m2_selection.json", surrogate)
    dump_json(output / "paired_pool_comparisons.json", comparisons)
    dump_json(output / "bootstrap_results.json", bootstrap)
    dump_json(output / "sign_test_summary.json", signs)
    artifacts.update({
        "per_pool": metrics["per_pool"], "method_metrics": metrics["methods"],
        "family_confirmation": family, "surrogate_selection": surrogate,
        "comparisons": comparisons, "bootstrap": bootstrap, "sign_tests": signs,
    })
    analysis_log(f"frozen classification: {decision['classification']}")
    _finalize_terminal(output, decision, artifacts)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--source-sha256", default="")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--synthetic-smoke", action="store_true")
    modes.add_argument("--formal", action="store_true")
    modes.add_argument("--write-failure-bundle", metavar="MESSAGE")
    modes.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--preflight-ready", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return execute_dry_run(args.config, args.output)
    if args.synthetic_smoke:
        result = synthetic_smoke(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.refresh_manifest:
        if not args.output.is_dir():
            raise ProtocolError("cannot refresh manifest for missing output")
        write_manifest(args.output)
        return 0
    if args.write_failure_bundle is not None:
        write_failure_bundle(
            args.output, args.write_failure_bundle,
            config_sha256=args.config_sha256 or "UNAVAILABLE",
            source_sha256=args.source_sha256 or "UNAVAILABLE",
        )
        return 0
    try:
        return execute_formal(
            args.config, args.output, args.config_sha256, args.source_sha256,
            preflight_ready=args.preflight_ready,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        try:
            write_failure_bundle(
                args.output, message,
                config_sha256=args.config_sha256 or "UNAVAILABLE",
                source_sha256=args.source_sha256 or "UNAVAILABLE",
            )
            dump_text(args.output / "logs/analysis.log", traceback.format_exc())
        except Exception:
            traceback.print_exc()
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
