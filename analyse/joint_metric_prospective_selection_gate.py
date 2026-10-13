#!/usr/bin/env python3
"""Frozen prospective selection gate for the soft joint metric.

The program has two deliberately separated phases.  The pre-evaluation phase
replays the archived, cross-fitted scorers and freezes every candidate, score,
and action.  Only a sufficiently informative frozen comparison is admitted to
the evaluator phase.  No candidate label is ever fed back into a scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROTOCOL_VERSION = "joint_metric_prospective_selection_gate_global4_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/joint_metric_prospective_selection_gate_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/posthoc/joint-metric-prospective-selection-gate-global4-v1-rerun1"
DEFAULT_ARCHIVE = Path(str(DEFAULT_OUTPUT) + ".tar.gz")

PARENT_FAILED_ARCHIVE_SHA256 = "a09123b997c4cef6ac4cd02d4261711002e3a9d28c2431ed0321d2d03f81f955"
PARENT_FAILURE = "AttributeError: '_NamespacePath' object has no attribute 'insert'"

HISTORICAL_SOFT_ROOT = REPO_ROOT / "results/posthoc/joint-configuration-soft-metric-gate-global4-v2"
HISTORICAL_DIAGNOSIS_ROOT = REPO_ROOT / "results/posthoc/joint-configuration-cross-factor-locality-diagnosis-global4-v1"
HISTORICAL_HIGH_ROOT = REPO_ROOT / "results/search/high-regime-prior-model-blind-validation-global4-v1"
FROZEN_SOFT_SNAPSHOT = HISTORICAL_SOFT_ROOT / "source_snapshot"
FROZEN_HIGH_SNAPSHOT = HISTORICAL_HIGH_ROOT / "source_snapshot"

ALLOWED_CLASSIFICATIONS = {
    "PROSPECTIVE_INPUT_INVALID",
    "SCORER_REPLAY_IDENTITY_MISMATCH",
    "CANDIDATE_GENERATION_FAILED",
    "PROSPECTIVE_LOW_DISAGREEMENT",
    "PROSPECTIVE_EVALUATION_INCOMPLETE",
    "METRIC_SELECTION_PROSPECTIVE_REJECTED",
    "METRIC_KNN_PROSPECTIVE_SIGNAL_ONLY",
    "METRIC_GP_PROSPECTIVE_SIGNAL",
    "RUN_FAILED",
}
TERMINAL_WITH_FROZEN_SELECTIONS = {
    "PROSPECTIVE_LOW_DISAGREEMENT",
    "PROSPECTIVE_EVALUATION_INCOMPLETE",
    "METRIC_SELECTION_PROSPECTIVE_REJECTED",
    "METRIC_KNN_PROSPECTIVE_SIGNAL_ONLY",
    "METRIC_GP_PROSPECTIVE_SIGNAL",
}

FROZEN_HISTORICAL_IDENTITY = {
    "unique_joint_configurations": 336,
    "unique_architectures": 192,
    "unique_pools": 24,
    "raw_seed_records": 1008,
    "training_seeds": [0, 1, 2],
    "historical_soft_metric_classification": "SOFT_JOINT_METRIC_REJECTED",
    "soft_config_sha256": "0b97af8a13d270e80fe8cc7b9a4beaf2616ff624c01faf5832fdebbeae25aadd",
    "soft_source_sha256": "d042a158a2c006baecc7c9d1d50dd42683f7b4f4c4dbdfed0c2f2c8bcc3a3aea",
    "soft_snapshot_manifest_root": "7125d9d7eececca53fcd00c2d578fb685c1b0d2156de62f4de08b362ffe10273",
    "split_sha256": "b80b1a211a05285121cafe4dabd4c67baddea93606c1913e777fcd34dd79a2ec",
}
FROZEN_SOFT_METRIC_IDENTITY = {
    "classification": "SOFT_JOINT_METRIC_REJECTED",
    "archive_sha256": "f5a0a73cb45098994afa3d60de44eb4c54c010d6814d963e3d4c3ba69c219f74",
    "config_sha256": FROZEN_HISTORICAL_IDENTITY["soft_config_sha256"],
    "source_sha256": FROZEN_HISTORICAL_IDENTITY["soft_source_sha256"],
    "source_snapshot_manifest_root": FROZEN_HISTORICAL_IDENTITY["soft_snapshot_manifest_root"],
    "split_sha256": FROZEN_HISTORICAL_IDENTITY["split_sha256"],
}
FROZEN_TRAINING_PROTOCOL = {
    "training_seeds": [0, 1, 2], "track_test": False,
    "max_epochs": 300, "patience": 80, "maximum_system_retries": 2,
}
FROZEN_SOURCE_HASHES = {
    "configs/joint_configuration_soft_metric_gate_v2.yaml": "0b97af8a13d270e80fe8cc7b9a4beaf2616ff624c01faf5832fdebbeae25aadd",
    "analyse/joint_configuration_soft_metric_gate.py": "d823e87055774e29a11851a45b659785cbb9ba524fec063f2f5f0e0a3e011c50",
    "analyse/joint_configuration_metric_gate.py": "e7c883e63b7a6133f5be6d029cb8d8f2f5cf755d8366cf9cf715e0c214906a13",
    "analyse/performance_metric_quick_gate.py": "92fd4bb6cf2e49a67f37b62582072fc6a9960f0725a92366e7b1ca271af6d7ba",
    "analyse/high_regime_prior_model_blind_validation.py": "4c6f5228abe7480c1a018b187483461505a272246dc13ef1084eafac2887ac39",
    "analyse/run_high_regime_prior_model_blind_validation_labels.py": "b3b951993ff69f7a28b068235887443b995f6fa6e0f25ed6c532d086a81a4aab",
    "eval_utils.py": "0cbfb402db030173ca11e4e90570df6ac427d99b416470863f55280d51e2541d",
    "architecture_diagnostics.py": "a1efc8a2ae21b8520d90166771ad434186a57147e0c7f01a842856abd00cf05f",
}
METRIC_SEEDS = (0, 1, 2)
TRAINING_SEEDS = (0, 1, 2)
POOL_COUNT = 8
CANDIDATES_PER_POOL = 4
CANDIDATE_COUNT = 32
TOTAL_GNN_RUNS = 96
MASTER_CANDIDATE_SEED = 2026082801
RANDOM_SELECTION_SEED = 2026082802
REPLAY_ATOL = 1e-8
MAX_GENERATION_DRAWS_PER_POOL = 65536
GENERATION_BATCH_SIZE = 256

REPORT_SECTIONS = (
    "1. Executive conclusion",
    "2. Historical scorer identity",
    "3. Candidate generation protocol",
    "4. Candidate novelty audit",
    "5. Pre-evaluation freeze",
    "6. Predictor score table",
    "7. Method disagreement analysis",
    "8. GNN evaluation completeness",
    "9. Per-pool true validation table",
    "10. Selected validation means",
    "11. Simple regret",
    "12. True-best hit rates",
    "13. M2 vs B3 primary comparison",
    "14. M1 diagnostic comparison",
    "15. Random baseline",
    "16. Pool-level bootstrap/sign summary",
    "17. Frozen classification",
    "18. Interpretation",
    "19. What this does NOT prove",
    "20. Next authorized action",
    "21. Engineering audit",
)
REQUIRED_SOURCE_SNAPSHOT_PATHS = (
    "configs/joint_metric_prospective_selection_gate_v1.yaml",
    "analyse/joint_metric_prospective_selection_gate.py",
    "scripts/run_joint_metric_prospective_selection_gate.sh",
    "tests/test_joint_metric_prospective_selection_gate.py",
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
    "scorer_replay_identity.json", "pre_eval/candidate_manifest.json",
    "pre_eval/predictions.csv", "pre_eval/selections.json",
    "pre_eval/scorer_identity.json", "pre_eval/freeze_manifest.json",
    "pre_eval/pre_eval_root_sha256.txt", "evaluation/task_manifest.json",
    "evaluation/raw_seed_results.csv", "evaluation/config_3seed_statistics.csv",
    "evaluation/training_audit.json", "selection_results_per_pool.csv",
    "prospective_metrics.json", "paired_pool_comparisons.json",
    "bootstrap_results.json", "decision.json", "final_report.md",
    "environment.json", "git_status_before.txt", "git_status_after.txt",
    "git_diff.patch", "source_snapshot/imported_local_modules.json",
    "source_snapshot/sha256_manifest.json", "manifest.json",
    "logs/py_compile.log", "logs/pytest_focused.log", "logs/synthetic_smoke.log",
    "logs/scorer_replay.log", "logs/candidate_generation.log",
    "logs/evaluation.log", "logs/analysis.log", "logs/runner.log",
)

CSV_FIELDS = {
    "pre_eval/predictions.csv": ["pool_id", "joint_fingerprint", "architecture_fingerprint", "M1", "M2", "B3"],
    "evaluation/raw_seed_results.csv": ["pool_id", "joint_fingerprint", "training_seed", "validation_accuracy", "terminal_status", "attempt_count"],
    "evaluation/config_3seed_statistics.csv": ["pool_id", "joint_fingerprint", "validation_seed_0", "validation_seed_1", "validation_seed_2", "validation_mean", "validation_sd", "validation_se"],
    "selection_results_per_pool.csv": ["pool_id", "method", "selected_joint_fingerprint", "selected_validation_mean", "observed_oracle", "simple_regret", "true_best_hit", "within_0_006_of_oracle"],
}


class ProtocolError(RuntimeError):
    pass


class ProspectiveInputInvalid(ProtocolError):
    pass


class ScorerReplayIdentityMismatch(ProtocolError):
    pass


class CandidateGenerationFailed(ProtocolError):
    pass


CandidateGenerationError = CandidateGenerationFailed


class PreEvalFreezeError(ProtocolError):
    pass


class EvaluationIncompleteError(ProtocolError):
    pass


def engineering_fix_payload() -> dict[str, Any]:
    return {
        "parent_failed_archive_sha256": PARENT_FAILED_ARCHIVE_SHA256,
        "parent_failure": PARENT_FAILURE,
        "scientific_protocol_changed": False,
        "candidate_generation_changed": False,
        "candidate_seeds_changed": False,
        "scorer_changed": False,
        "selection_gate_changed": False,
        "engineering_fix": "replace _NamespacePath.insert with an ordered de-duplicated list assignment for frozen analyse import precedence",
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prospective_source_paths() -> list[Path]:
    return [
        REPO_ROOT / "analyse/joint_metric_prospective_selection_gate.py",
        REPO_ROOT / "scripts/run_joint_metric_prospective_selection_gate.sh",
        REPO_ROOT / "tests/test_joint_metric_prospective_selection_gate.py",
        REPO_ROOT / "configs/joint_metric_prospective_selection_gate_v1.yaml",
        FROZEN_SOFT_SNAPSHOT / "configs/joint_configuration_soft_metric_gate_v2.yaml",
        FROZEN_SOFT_SNAPSHOT / "analyse/joint_configuration_soft_metric_gate.py",
        FROZEN_SOFT_SNAPSHOT / "analyse/joint_configuration_metric_gate.py",
        FROZEN_SOFT_SNAPSHOT / "analyse/performance_metric_quick_gate.py",
        FROZEN_SOFT_SNAPSHOT / "analyse/joint_pair_power_projection_audit.py",
        FROZEN_HIGH_SNAPSHOT / "analyse/high_regime_prior_model_blind_validation.py",
        FROZEN_HIGH_SNAPSHOT / "analyse/run_high_regime_prior_model_blind_validation_labels.py",
        FROZEN_HIGH_SNAPSHOT / "analyse/analyse_high_regime_prior_model_blind_validation.py",
        FROZEN_HIGH_SNAPSHOT / "eval_utils.py",
        FROZEN_HIGH_SNAPSHOT / "architecture_diagnostics.py",
    ]


def aggregate_source_sha256(paths: Sequence[Path] | None = None) -> str:
    digest = hashlib.sha256()
    for path in paths or _prospective_source_paths():
        if not path.is_file():
            raise ProtocolError(f"source identity file is missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dump_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def dump_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise ProtocolError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ProtocolError("configuration must be a mapping")
    return value


def _deep_get(mapping: Mapping[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ProtocolError(f"missing frozen configuration key: {dotted}")
        value = value[key]
    return value


def validate_frozen_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("protocol_version differs from the preregistered value")
    expected = {
        "candidate_generation.pools": POOL_COUNT,
        "candidate_generation.configurations_per_pool": CANDIDATES_PER_POOL,
        "candidate_generation.master_seed": MASTER_CANDIDATE_SEED,
        "candidate_generation.rng_streams.architecture": "torch_cpu_generator_seeded_directly_with_pool_seed",
        "candidate_generation.rng_streams.hp": "numpy_default_rng_seeded_directly_with_pool_seed",
        "candidate_generation.rng_streams.hp_draw_timing": "after_architecture_validity_and_novelty_acceptance",
        "candidate_generation.acceptance_order": "first_four_valid_novel_decodes_in_generation_order",
        "candidate_generation.duplicate_scope": "historical_training_set_and_all_previously_accepted_new_pools",
        "candidate_generation.hp_raw_unit_interval_range": [0.80, 0.95],
        "candidate_generation.maximum_draws_per_pool": MAX_GENERATION_DRAWS_PER_POOL,
        "candidate_generation.decode_batch_size": GENERATION_BATCH_SIZE,
        "prospective_methods.R.selection_seed": RANDOM_SELECTION_SEED,
        "scorer_replay.metric_model_seeds": list(METRIC_SEEDS),
        "evaluation.training_seeds": list(TRAINING_SEEDS),
        "prospective_methods.M1.k": 5,
        "prospective_methods.M2.ard": False,
        "prospective_methods.M2.per_dimension_input_standardization": False,
        "prospective_methods.M2.prediction": "posterior_mean_only",
        "pre_evaluation_freeze.disagreement.required_pools_at_least": 5,
        "pre_evaluation_freeze.root_algorithm": "sha256_of_canonical_json_sorted_relative_path_and_file_sha256_pairs",
        "statistics.bootstrap.replicates": 10000,
        "statistics.bootstrap.seed": 0,
        "prospective_gate.primary_method": "M2",
        "prospective_gate.comparator": "B3",
        "prospective_gate.P1.disagreement_pools_at_least": 5,
        "prospective_gate.P2.wins_on_disagreement_pools_fraction_at_least": 0.75,
        "prospective_gate.P2.wins_on_disagreement_pools_count_at_least": 4,
        "prospective_gate.P2.required_count_formula": "ceil_0.75_times_n_disagreement",
        "prospective_gate.P3.mean_selected_validation_advantage_all_8_at_least": 0.003,
        "prospective_gate.P4.mean_simple_regret_relative_reduction_at_least": 0.20,
        "prospective_gate.P4.comparator_near_zero_threshold_strictly_less_than": 1.0e-6,
        "prospective_gate.P4.near_zero_fallback": "method_mean_regret_less_than_or_equal_to_comparator",
        "prospective_gate.P5.true_best_hits_at_least_comparator": True,
        "prospective_gate.M1_signal.comparison": "M1_vs_B3",
        "prospective_gate.M1_signal.conditions": ["P2", "P3", "P4", "P5"],
        "prospective_gate.M1_signal.may_form_gp_signal": False,
    }
    actual: dict[str, Any] = {}
    for key, wanted in expected.items():
        got = _deep_get(config, key)
        if got != wanted:
            raise ProtocolError(f"frozen value mismatch for {key}: {got!r} != {wanted!r}")
        actual[key] = got
    if _deep_get(config, "evaluation.track_test") is not False:
        raise ProtocolError("evaluation.track_test must be false")
    return {"status": "VALID", "frozen_values": actual, "test_accessed": False}


def _test_key_hits(value: Any, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "test" in str(key).lower():
                hits.append(path)
            hits.extend(_test_key_hits(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_test_key_hits(item, f"{prefix}[{index}]"))
    return hits


def reject_test_fields(value: Any, *, allow_track_test_false: bool = False) -> None:
    hits = _test_key_hits(value)
    if allow_track_test_false:
        def allowed(path: str) -> bool:
            return path.endswith("track_test") and _lookup_path(value, path) is False
        hits = [hit for hit in hits if not allowed(hit)]
    if hits:
        raise ProtocolError(f"test-data firewall rejected fields: {hits[:8]}")


def _lookup_path(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if "[" in component:
            head, rest = component.split("[", 1)
            if head:
                current = current[head]
            current = current[int(rest.rstrip("]"))]
        else:
            current = current[component]
    return current


def validate_historical_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "raw_seed_records": ("raw_seed_records", "seed_record_count", "raw_row_count", "successful_required_seed_row_count"),
        "training_seeds": ("training_seeds", "complete_training_seeds", "metric_training_seeds"),
    }
    observed: dict[str, Any] = {}
    for key in ("unique_joint_configurations", "unique_architectures", "unique_pools"):
        observed[key] = payload.get(key)
    for canonical, names in aliases.items():
        observed[canonical] = next((payload[name] for name in names if name in payload), None)
    for key in ("unique_joint_configurations", "unique_architectures", "unique_pools", "raw_seed_records", "training_seeds"):
        if observed[key] != FROZEN_HISTORICAL_IDENTITY[key]:
            raise ProspectiveInputInvalid(f"historical identity mismatch for {key}")
    return {"passed": True, **observed}


def aggregate_cross_fitted_scores(
    per_fold_seed: Mapping[Any, Any], *, folds: Sequence[int] = tuple(range(5)),
    metric_seeds: Sequence[int] = METRIC_SEEDS,
) -> Any:
    import numpy as np
    tuple_keyed = all(isinstance(key, tuple) and len(key) == 2 for key in per_fold_seed)
    if tuple_keyed:
        normalized = {fold: {seed: per_fold_seed[(fold, seed)] for seed in metric_seeds} for fold in folds}
    else:
        normalized = per_fold_seed
    if set(normalized) != set(folds):
        raise ProtocolError("cross-fitted scores require the frozen five folds")
    fold_means = []
    for fold in folds:
        if set(normalized[fold]) != set(metric_seeds):
            raise ProtocolError("each fold requires metric seeds 0,1,2")
        arrays = [np.asarray(normalized[fold][seed], dtype=float) for seed in metric_seeds]
        if not arrays or any(array.shape != arrays[0].shape for array in arrays):
            raise ProtocolError("seed prediction shapes differ")
        fold_means.append(np.mean(np.stack(arrays), axis=0))
    return np.mean(np.stack(fold_means), axis=0)


def select_pool_argmax(candidates: Sequence[Mapping[str, Any]], score_key: str | None = None, *, score_field: str | None = None) -> str:
    if not candidates:
        raise ProtocolError("cannot select from an empty pool")
    field = score_field or score_key
    if field is None:
        raise ProtocolError("score field is required")
    def key(row: Mapping[str, Any]) -> tuple[float, str]:
        score = float(row[field])
        if not math.isfinite(score):
            raise ProtocolError("non-finite prospective score")
        return (-score, str(row["joint_fingerprint"]))
    return str(sorted(candidates, key=key)[0]["joint_fingerprint"])


def validate_candidate_manifest(
    candidates: Sequence[Mapping[str, Any]],
    historical_architectures: set[str] | None = None,
    historical_joints: set[str] | None = None,
    *, historical_architecture_fingerprints: set[str] | None = None,
    historical_joint_fingerprints: set[str] | None = None,
    expected_pool_count: int = POOL_COUNT,
    expected_candidates_per_pool: int = CANDIDATES_PER_POOL,
    expected_master_seed: int = MASTER_CANDIDATE_SEED,
    expected_distribution: str = "HIGH_REGIME_PRIOR",
    generator_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    historical_architectures = historical_architecture_fingerprints if historical_architecture_fingerprints is not None else (historical_architectures or set())
    historical_joints = historical_joint_fingerprints if historical_joint_fingerprints is not None else (historical_joints or set())
    if len(candidates) != expected_pool_count * expected_candidates_per_pool:
        raise CandidateGenerationFailed(f"expected {expected_pool_count * expected_candidates_per_pool} candidates")
    arch = [str(row["architecture_fingerprint"]) for row in candidates]
    joint = [str(row["joint_fingerprint"]) for row in candidates]
    pools: dict[int, int] = {}
    for row in candidates:
        pool = int(row["pool_id"])
        pools[pool] = pools.get(pool, 0) + 1
        for hp in ("lr", "dropout", "hidden_dim", "l2"):
            if hp not in row:
                raise CandidateGenerationFailed(f"candidate lacks raw HP {hp}")
    checks = {
        "architecture_historical_unseen": not bool(set(arch) & historical_architectures),
        "joint_historical_unseen": not bool(set(joint) & historical_joints),
        "architecture_internal_unique": len(set(arch)) == len(candidates),
        "joint_internal_unique": len(set(joint)) == len(candidates),
        "pool_count": len(pools) == expected_pool_count,
        "per_pool_count": pools == {pool: expected_candidates_per_pool for pool in range(expected_pool_count)},
    }
    for row in candidates:
        if int(row.get("generator_seed", row.get("generation_pool_seed", expected_master_seed + int(row["pool_id"])))) != expected_master_seed + int(row["pool_id"]):
            checks["pool_seed"] = False
        if str(row.get("proposal_distribution", row.get("hp_regime", ""))).upper() not in {expected_distribution.upper(), "HP_HIGH"}:
            checks["distribution"] = False
    if generator_audit is not None:
        if generator_audit.get("predictor_accessed") is not False or generator_audit.get("validation_labels_accessed") is not False:
            checks["generator_firewall"] = False
    if not all(checks.values()):
        raise CandidateGenerationFailed(f"candidate novelty/shape check failed: {checks}")
    return {"passed": True, "predictor_accessed": False, "validation_labels_accessed": False, **checks}


def assert_candidate_manifest_immutable(before: Any, after: Any) -> None:
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise PreEvalFreezeError("candidate manifest changed after selection freeze")


def frozen_random_indices(*, pool_count: int = POOL_COUNT, choices_per_pool: int = CANDIDATES_PER_POOL, seed: int = RANDOM_SELECTION_SEED) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(choices_per_pool) for _ in range(pool_count)]


def frozen_training_protocol_identity() -> str:
    return stable_hash({
        "dataset": "Cora", "dataset_content_fingerprint": "373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7",
        "split": "public", "split_fingerprint": "bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09",
        "graph_transform": "NormalizeFeatures", "training_mode": "full_batch", "hp_mode": "global4",
        "max_epochs": 300, "patience": 80, "maximum_system_retries": 2,
        "training_seeds": list(TRAINING_SEEDS), "track_test": False,
        "optimizer_protocol": "frozen eval_utils Adam plus CosineAnnealingLR",
        "training_entry": "eval_utils.train_and_eval_arch", "eval_utils_sha256": FROZEN_SOURCE_HASHES["eval_utils.py"],
        "component_variant": "baseline", "gcnii_alpha": 0.1, "gcnii_theta": 0.5,
    })


def disagreement_count(selections: Mapping[str, Any]) -> int:
    if isinstance(selections, Sequence) and not isinstance(selections, (str, bytes, Mapping)):
        return sum(str(row["M2_selected_joint_fingerprint"]) != str(row["B3_selected_joint_fingerprint"]) for row in selections)
    pools = selections.get("pools", selections)
    count = 0
    for value in pools.values() if isinstance(pools, Mapping) else pools:
        if str(value["M2"]) != str(value["B3"]):
            count += 1
    return count


def build_evaluation_tasks(
    candidates: Sequence[Mapping[str, Any]], *, pre_eval_verified: bool = True,
    disagreement_pool_count: int = 8, minimum_disagreement_pools: int = 5,
    training_seeds: Sequence[int] = TRAINING_SEEDS, track_test: bool = False,
) -> list[dict[str, Any]]:
    if track_test:
        raise ProtocolError("track_test must remain false")
    if not pre_eval_verified:
        raise PreEvalFreezeError("evaluation cannot start before pre-evaluation verification")
    if disagreement_pool_count < minimum_disagreement_pools:
        return []
    if list(training_seeds) != list(TRAINING_SEEDS):
        raise ProtocolError("training seeds differ from [0,1,2]")
    tasks: list[dict[str, Any]] = []
    protocol_identity = frozen_training_protocol_identity()
    for row in sorted(candidates, key=lambda item: (int(item["pool_id"]), str(item["joint_fingerprint"]))):
        for seed in training_seeds:
            tasks.append({
                "task_id": f"initial::{int(row['pool_id'])}::{row['joint_fingerprint']}::seed={seed}::protocol={protocol_identity}",
                "task_key": f"initial::{int(row['pool_id'])}::{row['joint_fingerprint']}::seed={seed}::protocol={protocol_identity}",
                "training_protocol_identity": protocol_identity,
                "pool_id": int(row["pool_id"]),
                "joint_fingerprint": str(row["joint_fingerprint"]),
                "architecture_fingerprint": str(row["architecture_fingerprint"]),
                "training_seed": seed,
                "seed": seed,
                "candidate": dict(row),
                "track_test": False,
            })
    if len(tasks) != TOTAL_GNN_RUNS:
        raise ProtocolError("evaluation manifest must contain exactly 96 tasks")
    if {task["training_seed"] for task in tasks} != set(TRAINING_SEEDS):
        raise ProtocolError("evaluation seeds differ from [0,1,2]")
    return tasks


def aggregate_evaluation_results(
    candidates: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]], *, allow_incomplete: bool = False,
) -> Any:
    import numpy as np
    expected = {(str(row["joint_fingerprint"]), seed) for row in candidates for seed in TRAINING_SEEDS}
    seen: dict[tuple[str, int], float] = {}
    for row in results:
        if "track_test" in row and row["track_test"] is not False:
            raise ProtocolError("track_test must be false")
        reject_test_fields(row, allow_track_test_false=True)
        key = (str(row.get("joint_fingerprint", row.get("joint_config_fingerprint"))), int(row.get("training_seed", row.get("seed"))))
        if key not in expected or key in seen:
            raise ProtocolError("foreign or duplicate evaluation result")
        if row.get("terminal_status") != "success" or row.get("valid", True) is not True:
            continue
        value = float(row.get("validation_accuracy", row.get("val_acc")))
        if not math.isfinite(value):
            raise ProtocolError("non-finite validation accuracy")
        seen[key] = value
    complete = set(seen) == expected
    if not complete and not allow_incomplete:
        missing = sorted(expected - set(seen))
        raise EvaluationIncompleteError(f"evaluation lacks {len(missing)} required config-seed results")
    stats: list[dict[str, Any]] = []
    by_candidate = {str(row["joint_fingerprint"]): row for row in candidates}
    for fingerprint in sorted(by_candidate):
        if not all((fingerprint, seed) in seen for seed in TRAINING_SEEDS):
            continue
        values = [seen[(fingerprint, seed)] for seed in TRAINING_SEEDS]
        sd = float(np.std(values, ddof=1))
        stats.append({
            "pool_id": int(by_candidate[fingerprint]["pool_id"]),
            "joint_fingerprint": fingerprint,
            "validation_seed_0": values[0], "validation_seed_1": values[1], "validation_seed_2": values[2],
            "validation_mean": float(np.mean(values)), "validation_sd": sd,
            "validation_se": sd / math.sqrt(3.0),
        })
    if complete:
        oracle_by_pool: dict[int, float] = {}
        for row in stats:
            pool = int(row["pool_id"])
            oracle_by_pool[pool] = max(oracle_by_pool.get(pool, -math.inf), float(row["validation_mean"]))
        for row in stats:
            row["observed_pool_oracle"] = oracle_by_pool[int(row["pool_id"])]
    return (stats, complete) if allow_incomplete else stats


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _compute_prospective_metrics_full(
    candidates: Sequence[Mapping[str, Any]],
    selections: Mapping[str, Any],
    config_stats: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = {str(row["joint_fingerprint"]): float(row["validation_mean"]) for row in config_stats}
    pools = selections.get("pools", selections)
    by_pool: dict[int, list[str]] = {}
    for row in candidates:
        by_pool.setdefault(int(row["pool_id"]), []).append(str(row["joint_fingerprint"]))
    rows: list[dict[str, Any]] = []
    methods = ("M1", "M2", "B3", "R")
    oracle_by_pool: dict[int, float] = {}
    for pool in range(POOL_COUNT):
        oracle = max(values[fp] for fp in by_pool[pool])
        oracle_by_pool[pool] = oracle
        selection = pools[str(pool)] if str(pool) in pools else pools[pool]
        for method in methods:
            fp = str(selection[method])
            selected = values[fp]
            regret = oracle - selected
            rows.append({
                "pool_id": pool, "method": method, "selected_joint_fingerprint": fp,
                "selected_validation_mean": selected, "observed_oracle": oracle,
                "simple_regret": regret, "true_best_hit": int(abs(regret) <= 1e-12),
                "within_0_006_of_oracle": int(regret <= 0.006 + 1e-15),
            })
    summary: dict[str, Any] = {"methods": {}, "observed_oracle_definition": "maximum 3-seed mean within each evaluated four-candidate pool"}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        regrets = [float(row["simple_regret"]) for row in selected]
        summary["methods"][method] = {
            "selected_validation_mean": _mean([float(row["selected_validation_mean"]) for row in selected]),
            "mean_simple_regret": _mean(regrets),
            "median_simple_regret": float(statistics.median(regrets)),
            "true_best_hits": sum(int(row["true_best_hit"]) for row in selected),
            "true_best_hit_rate": _mean([float(row["true_best_hit"]) for row in selected]),
            "within_0_006_of_oracle_rate": _mean([float(row["within_0_006_of_oracle"]) for row in selected]),
        }
    pool_means = [_mean([values[fp] for fp in by_pool[pool]]) for pool in range(POOL_COUNT)]
    summary["descriptive_expected_random"] = {
        "status": "DESCRIPTIVE_EXPECTED_RANDOM",
        "selected_validation_mean": _mean(pool_means),
        "mean_simple_regret": _mean([oracle_by_pool[p] - pool_means[p] for p in range(POOL_COUNT)]),
    }
    return rows, summary


def compute_prospective_metrics(*args: Any) -> Any:
    """Compute pool-level metrics; accept both formal and focused-test forms."""
    if len(args) == 3:
        return _compute_prospective_metrics_full(args[0], args[1], args[2])
    if len(args) != 2:
        raise TypeError("compute_prospective_metrics expects (stats,selections) or (candidates,selections,stats)")
    config_stats, selections = args
    values = {str(row["joint_fingerprint"]): float(row["validation_mean"]) for row in config_stats}
    by_pool: dict[int, list[Mapping[str, Any]]] = {}
    for row in config_stats:
        by_pool.setdefault(int(row["pool_id"]), []).append(row)
    if isinstance(selections, Mapping):
        selection_rows = []
        for pool, value in selections.get("pools", selections).items():
            selection_rows.append({"pool_id": int(pool), **{f"{method}_selected_joint_fingerprint": fp for method, fp in value.items() if method in {"M1", "M2", "B3", "R"}}})
    else:
        selection_rows = list(selections)
    selection_by_pool = {int(row["pool_id"]): row for row in selection_rows}
    per_pool = []
    for pool in range(POOL_COUNT):
        oracle = max(float(row["validation_mean"]) for row in by_pool[pool])
        item: dict[str, Any] = {"pool_id": pool, "observed_pool_oracle": oracle}
        selection = selection_by_pool[pool]
        for method in ("M1", "M2", "B3", "R"):
            fp = str(selection[f"{method}_selected_joint_fingerprint"])
            selected = values[fp]
            item[f"{method}_selected_validation_mean"] = selected
            item[f"{method}_simple_regret"] = oracle - selected
            item[f"{method}_true_best_hit"] = oracle_hit(selected, oracle)
        per_pool.append(item)
    return {"per_pool": per_pool, "statistical_unit": "pool", "pool_count": POOL_COUNT, "training_seed_count_is_not_independent_unit": True}


def oracle_hit(selected_mean: float, oracle_mean: float, *, tolerance: float = 1e-12) -> int:
    return int(abs(float(selected_mean) - float(oracle_mean)) <= tolerance)


def _relative_reduction(new: float, old: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else -math.inf
    return (old - new) / old


def _method_gate(method: str, summary: Mapping[str, Any], per_pool: Sequence[Mapping[str, Any]], n_disagreement: int) -> dict[str, Any]:
    m = summary["methods"][method]
    b = summary["methods"]["B3"]
    rows_by = {(int(r["pool_id"]), str(r["method"])): r for r in per_pool}
    disagreements = [p for p in range(POOL_COUNT) if rows_by[(p, method)]["selected_joint_fingerprint"] != rows_by[(p, "B3")]["selected_joint_fingerprint"]]
    wins = losses = ties = 0
    differences: list[float] = []
    for pool in range(POOL_COUNT):
        difference = float(rows_by[(pool, method)]["selected_validation_mean"]) - float(rows_by[(pool, "B3")]["selected_validation_mean"])
        differences.append(difference)
        if pool in disagreements:
            if difference > 1e-12:
                wins += 1
            elif difference < -1e-12:
                losses += 1
            else:
                ties += 1
    required_wins = max(4, math.ceil(0.75 * len(disagreements)))
    # P1 is always the preregistered information gate on M2 versus B3.  M1 is
    # diagnostic and inherits that already-passed admission gate.
    p1 = n_disagreement >= 5
    p2 = wins >= required_wins
    p3 = _mean(differences) >= 0.003
    if float(b["mean_simple_regret"]) < 1e-6:
        p4 = float(m["mean_simple_regret"]) <= float(b["mean_simple_regret"]) + 1e-15
        p4_mode = "absolute_noninferiority_b3_regret_below_1e-6"
    else:
        p4 = _relative_reduction(float(m["mean_simple_regret"]), float(b["mean_simple_regret"])) >= 0.20
        p4_mode = "relative_reduction_at_least_20_percent"
    p5 = int(m["true_best_hits"]) >= int(b["true_best_hits"])
    return {
        "method": method, "disagreement_pools": len(disagreements), "wins": wins, "losses": losses, "ties": ties,
        "required_wins": required_wins, "mean_selected_difference": _mean(differences),
        "regret_relative_reduction": _relative_reduction(float(m["mean_simple_regret"]), float(b["mean_simple_regret"])),
        "P1": p1, "P2": p2, "P3": p3, "P4": p4, "P4_mode": p4_mode, "P5": p5,
        "passed": all((p1, p2, p3, p4, p5)),
    }


def _classify_prospective_outputs(
    *, evaluation_complete: bool, n_disagreement: int,
    prospective_summary: Mapping[str, Any] | None = None,
    per_pool_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if n_disagreement < 5:
        return {"classification": "PROSPECTIVE_LOW_DISAGREEMENT", "n_disagreement": n_disagreement, "gnn_runs_authorized": False}
    if not evaluation_complete:
        return {"classification": "PROSPECTIVE_EVALUATION_INCOMPLETE", "n_disagreement": n_disagreement, "gnn_runs_authorized": True}
    if prospective_summary is None or per_pool_rows is None:
        raise ProtocolError("completed evaluation requires prospective metrics")
    m2 = _method_gate("M2", prospective_summary, per_pool_rows, n_disagreement)
    m1 = _method_gate("M1", prospective_summary, per_pool_rows, n_disagreement)
    if m2["passed"]:
        classification = "METRIC_GP_PROSPECTIVE_SIGNAL"
    elif m1["passed"]:
        classification = "METRIC_KNN_PROSPECTIVE_SIGNAL_ONLY"
    else:
        classification = "METRIC_SELECTION_PROSPECTIVE_REJECTED"
    return {"classification": classification, "n_disagreement": n_disagreement, "M2_gate": m2, "M1_gate": m1, "gnn_runs_authorized": True}


def _evidence_method_pass(method: str, evidence: Mapping[str, Any]) -> bool:
    model = evidence[method]
    baseline = evidence["B3"]
    n = int(evidence["n_disagreement"])
    required_wins = max(4, math.ceil(0.75 * n))
    p2 = int(model["wins_vs_B3"]) >= required_wins
    p3 = float(model["mean_selected_gain_vs_B3"]) >= 0.003
    b_regret = float(baseline["mean_regret"])
    m_regret = float(model["mean_regret"])
    p4 = m_regret <= b_regret if b_regret < 1e-6 else (b_regret - m_regret) / b_regret >= 0.20
    p5 = int(model["true_best_hits"]) >= int(baseline["true_best_hits"])
    return n >= 5 and p2 and p3 and p4 and p5


def classify_prospective(evidence: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    if evidence is None:
        return _classify_prospective_outputs(**kwargs)
    if evidence.get("input_valid", True) is not True:
        return {"classification": "PROSPECTIVE_INPUT_INVALID"}
    if evidence.get("scorer_replay_valid", True) is not True:
        return {"classification": "SCORER_REPLAY_IDENTITY_MISMATCH"}
    if evidence.get("candidate_generation_valid", True) is not True:
        return {"classification": "CANDIDATE_GENERATION_FAILED"}
    n = int(evidence.get("n_disagreement", 0))
    if evidence.get("evaluation_complete") is not True:
        classification = "PROSPECTIVE_LOW_DISAGREEMENT" if n < 5 and int(evidence.get("gnn_runs_started", 0)) == 0 else "PROSPECTIVE_EVALUATION_INCOMPLETE"
        return {"classification": classification}
    m2_pass = _evidence_method_pass("M2", evidence)
    m1_pass = _evidence_method_pass("M1", evidence)
    classification = "METRIC_GP_PROSPECTIVE_SIGNAL" if m2_pass else ("METRIC_KNN_PROSPECTIVE_SIGNAL_ONLY" if m1_pass else "METRIC_SELECTION_PROSPECTIVE_REJECTED")
    return {"classification": classification}


def pool_bootstrap(values: Sequence[float], *, replicates: int = 10000, seed: int = 0, unit: str = "pool") -> dict[str, Any]:
    import numpy as np
    array = np.asarray(values, dtype=float)
    if unit != "pool":
        raise ProtocolError("bootstrap unit must be pool")
    if array.shape != (POOL_COUNT,) or not np.all(np.isfinite(array)):
        raise ProtocolError("bootstrap requires exactly eight finite pool-level values")
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, POOL_COUNT, size=(replicates, POOL_COUNT))].mean(axis=1)
    return {
        "unit": "pool", "bootstrap_unit": "pool", "pool_count": POOL_COUNT, "replicates": replicates, "seed": seed,
        "mean": float(array.mean()), "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "confidence_interval_95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "positive_sign_count": int(np.sum(array > 1e-12)), "negative_sign_count": int(np.sum(array < -1e-12)),
        "tie_count": int(np.sum(np.abs(array) <= 1e-12)),
    }


def assert_no_online_update(audit: Mapping[str, Any]) -> dict[str, Any]:
    groups = (
        ("new_labels_added_to_scorer", "candidate_labels_used_for_metric_update"),
        ("scorer_updated_after_freeze", "candidate_labels_used_for_gp_update"),
        ("candidates_replaced_after_freeze",),
        ("pool_sequential_update", "pool_sequential_updates"),
    )
    for names in groups:
        present = [audit[name] for name in names if name in audit]
        if present and any(value is not False for value in present):
            raise ProtocolError(f"online-update firewall failed: {names[0]}")
    if "all_predictions_frozen_before_first_evaluation" in audit and audit["all_predictions_frozen_before_first_evaluation"] is not True:
        raise ProtocolError("predictions were not frozen before evaluation")
    return {"passed": True, "new_labels_used": False, "online_update_used": False}


def validate_scorer_protocol(audit: Mapping[str, Any]) -> dict[str, Any]:
    if "M2" in audit:
        m2 = audit["M2"]
        b3 = audit.get("B3", {})
        m2_ok = (
            int(m2.get("input_dim", -1)) == 8 and m2.get("input_l2_normalized") is True
            and m2.get("kernel") == "Matern-5/2" and m2.get("ard") is False
            and int(m2.get("lengthscale_count", -1)) == 1 and m2.get("input_standardized") is False
            and m2.get("posterior_variance_used") is False
        )
        b3_ok = (
            b3.get("features") == "explicit_graph_plus_transformed_hp" and b3.get("kernel") == "Matern-5/2"
            and b3.get("ard") is True and b3.get("input_standardized") is True
            and b3.get("posterior_variance_used") is False
        )
        if not m2_ok or not b3_ok:
            raise ProtocolError("prospective scorer protocol differs from frozen v2")
        if "protocol_reused_without_tuning" in b3 and b3["protocol_reused_without_tuning"] is not True:
            raise ProtocolError("B3 protocol was retuned")
        return {"passed": True, "M2_isotropic_unstandardized": True, "posterior_mean_only": True, "B3_frozen_protocol": True}
    if audit.get("metric_seeds") != list(METRIC_SEEDS):
        raise ProtocolError("scorer metric seeds changed")
    if audit.get("fold_count") != 5 or audit.get("model_count") != 15:
        raise ProtocolError("scorer must replay five folds by three seeds")
    if audit.get("candidate_labels_used") is not False or audit.get("posterior_variance_used") is not False:
        raise ProtocolError("prospective scorer consumed forbidden information")
    if audit.get("M2_ard") is not False or audit.get("M2_input_standardized") is not False:
        raise ProtocolError("M2 is not the frozen isotropic metric GP")
    return {"passed": True, "M2_isotropic_unstandardized": True, "posterior_mean_only": True, "B3_frozen_protocol": True}


def _assert_float_close(observed: Any, expected: Any, path: str, atol: float) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ScorerReplayIdentityMismatch(f"mapping keys differ at {path}")
        for key in expected:
            _assert_float_close(observed[key], expected[key], f"{path}.{key}", atol)
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ScorerReplayIdentityMismatch(f"list shape differs at {path}")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _assert_float_close(left, right, f"{path}[{index}]", atol)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(observed, (int, float)) or not math.isfinite(float(observed)) or abs(float(observed) - float(expected)) > atol:
            raise ScorerReplayIdentityMismatch(f"numeric replay mismatch at {path}: {observed!r} vs {expected!r}")
    elif observed != expected:
        raise ScorerReplayIdentityMismatch(f"replay mismatch at {path}: {observed!r} vs {expected!r}")


def assert_scorer_replay_identity(replayed: Mapping[str, Any], archived: Mapping[str, Any], atol: float = REPLAY_ATOL) -> dict[str, Any]:
    """Compare scientific replay anchors; ordered neighbor lists remain exact."""
    aliases = {
        "seed_distance_gap_spearman": ("seed_distance_gap_spearman", "distance_gap_spearman"),
        "metric_gap_at_5": ("metric_gap_at_5", "gap_at_5"),
        "M1_oof": ("M1_oof", "M1_oof_predictions"),
        "M2_oof": ("M2_oof", "M2_oof_predictions"),
    }
    checked = []
    for canonical, names in aliases.items():
        expected_name = next((name for name in names if name in archived), None)
        if expected_name is not None:
            observed_name = next((name for name in names if name in replayed), None)
            if observed_name is None:
                raise ScorerReplayIdentityMismatch(f"missing replay anchor {canonical}")
            _assert_float_close(replayed[observed_name], archived[expected_name], canonical, atol)
            checked.append(canonical)
    neighbor_expected = next((name for name in ("top5_neighbor_fingerprints", "top5_fingerprints") if name in archived), None)
    if neighbor_expected:
        neighbor_observed = next((name for name in ("top5_neighbor_fingerprints", "top5_fingerprints") if name in replayed), None)
        if neighbor_observed is None or replayed[neighbor_observed] != archived[neighbor_expected]:
            raise ScorerReplayIdentityMismatch("ordered top-5 neighbor fingerprints differ")
        checked.append("top5_neighbor_fingerprints")
    return {"passed": True, "absolute_tolerance": atol, "checked": checked}


def freeze_pre_eval(
    pre: Path, candidate_manifest: Any = None, predictions: Sequence[Mapping[str, Any]] = (),
    selections: Any = None, scorer_identity: Mapping[str, Any] | None = None,
    *, historical_input_hashes: Mapping[str, Any] | None = None,
    generator_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pre.mkdir(parents=True, exist_ok=True)
    dump_json(pre / "candidate_manifest.json", candidate_manifest)
    prediction_fields = list(predictions[0].keys()) if predictions else CSV_FIELDS["pre_eval/predictions.csv"]
    dump_csv(pre / "predictions.csv", predictions, prediction_fields)
    dump_json(pre / "selections.json", selections)
    dump_json(pre / "scorer_identity.json", {
        **(dict(scorer_identity or {})),
        "historical_input_hashes": dict(historical_input_hashes or {}),
        "generator_identity": dict(generator_identity or {}),
    })
    entries = []
    for relative in ("candidate_manifest.json", "predictions.csv", "selections.json", "scorer_identity.json"):
        path = pre / relative
        entries.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    entries.sort(key=lambda entry: entry["path"])
    freeze_manifest = {
        "format_version": PROTOCOL_VERSION, "created_at": utc_now(), "immutable_entries": entries,
        "candidate_validation_labels_available_at_freeze": False,
        "all_eight_pools_frozen_before_first_gnn_run": True,
    }
    root = stable_hash([(entry["path"], entry["sha256"]) for entry in entries])
    freeze_manifest["root_sha256"] = root
    dump_json(pre / "freeze_manifest.json", freeze_manifest)
    dump_text(pre / "pre_eval_root_sha256.txt", root + "\n")
    dump_text(pre / "PRE_EVAL_SELECTION_FROZEN", "")
    return {"passed": True, "root_sha256": root, "pre_eval_directory": str(pre)}


def verify_pre_eval_freeze(pre: Path) -> dict[str, Any]:
    manifest = json.loads((pre / "freeze_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["immutable_entries"]:
        path = pre / entry["path"]
        if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]) or sha256_file(path) != entry["sha256"]:
            raise PreEvalFreezeError(f"pre-evaluation artifact changed after freeze: {entry['path']}")
    root = stable_hash([(entry["path"], entry["sha256"]) for entry in sorted(manifest["immutable_entries"], key=lambda item: item["path"])])
    if root != manifest["root_sha256"] or (pre / "pre_eval_root_sha256.txt").read_text(encoding="utf-8").strip() != root:
        raise PreEvalFreezeError("pre-evaluation immutable root mismatch")
    if not (pre / "PRE_EVAL_SELECTION_FROZEN").is_file():
        raise PreEvalFreezeError("PRE_EVAL_SELECTION_FROZEN marker is absent")
    return {"passed": True, "root_sha256": root}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProspectiveInputInvalid(f"cannot parse required JSON: {path}") from exc


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ProspectiveInputInvalid(f"{label} hash mismatch: {path}")


def validate_authoritative_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    if not (HISTORICAL_SOFT_ROOT / "COMPLETE").is_file():
        raise ProspectiveInputInvalid("soft metric v2 is not COMPLETE")
    soft_state = _load_json(HISTORICAL_SOFT_ROOT / "state.json")
    soft_decision = _load_json(HISTORICAL_SOFT_ROOT / "decision.json")
    inventory = _load_json(HISTORICAL_SOFT_ROOT / "data_inventory.json")
    split = _load_json(HISTORICAL_SOFT_ROOT / "split_manifest.json")
    identity = validate_historical_identity(inventory)
    classification = soft_decision.get("classification", soft_state.get("classification"))
    if classification != "SOFT_JOINT_METRIC_REJECTED":
        raise ProspectiveInputInvalid("historical soft metric classification changed")
    if soft_state.get("config_sha256") != FROZEN_HISTORICAL_IDENTITY["soft_config_sha256"]:
        raise ProspectiveInputInvalid("historical soft config identity changed")
    if soft_state.get("source_sha256") != FROZEN_HISTORICAL_IDENTITY["soft_source_sha256"]:
        raise ProspectiveInputInvalid("historical soft source identity changed")
    snapshot_manifest = _load_json(HISTORICAL_SOFT_ROOT / "source_snapshot/sha256_manifest.json")
    root = snapshot_manifest.get("manifest_root_sha256", snapshot_manifest.get("root_sha256"))
    if root != FROZEN_HISTORICAL_IDENTITY["soft_snapshot_manifest_root"]:
        raise ProspectiveInputInvalid("soft source snapshot root changed")
    split_copy = dict(split)
    recorded_split_hash = split_copy.pop("split_sha256", None)
    logical = stable_hash(split_copy)
    if recorded_split_hash != FROZEN_HISTORICAL_IDENTITY["split_sha256"] or logical != recorded_split_hash:
        raise ProspectiveInputInvalid("logical split SHA differs from frozen v1")
    for relative, expected in FROZEN_SOURCE_HASHES.items():
        root_path = FROZEN_SOFT_SNAPSHOT if relative.startswith(("configs/joint_configuration", "analyse/joint_configuration", "analyse/performance", "analyse/joint_pair")) else FROZEN_HIGH_SNAPSHOT
        candidate = root_path / relative
        _require_hash(candidate, expected, relative)
    if not (HISTORICAL_DIAGNOSIS_ROOT / "COMPLETE").is_file():
        raise ProspectiveInputInvalid("cross-factor diagnosis is not COMPLETE")
    diagnosis_decision = _load_json(HISTORICAL_DIAGNOSIS_ROOT / "decision.json")
    if diagnosis_decision.get("classification") != "CROSS_FACTOR_DIAGNOSIS_COMPLETE":
        raise ProspectiveInputInvalid("cross-factor diagnosis did not complete")
    diagnosis_spec = config["authoritative_inputs"]["cross_factor_diagnosis_v1"]
    diagnosis_state = _load_json(HISTORICAL_DIAGNOSIS_ROOT / "state.json")
    diagnosis_snapshot = _load_json(HISTORICAL_DIAGNOSIS_ROOT / "source_snapshot/sha256_manifest.json")
    diagnosis_archive = REPO_ROOT / str(diagnosis_spec["archive"])
    diagnosis_sidecar = REPO_ROOT / str(diagnosis_spec["archive_sha256_sidecar"])
    expected_diagnosis = {
        "decision": diagnosis_spec["classification"],
        "config_sha256": diagnosis_spec["config_sha256"],
        "source_sha256": diagnosis_spec["source_sha256"],
        "source_snapshot_manifest_root": diagnosis_spec["source_snapshot_manifest_root"],
    }
    for key, expected in expected_diagnosis.items():
        if diagnosis_state.get(key) != expected:
            raise ProspectiveInputInvalid(f"cross-factor diagnosis state identity mismatch: {key}")
    if diagnosis_snapshot.get("manifest_root_sha256") != diagnosis_spec["source_snapshot_manifest_root"]:
        raise ProspectiveInputInvalid("cross-factor diagnosis source snapshot root mismatch")
    if not diagnosis_archive.is_file() or sha256_file(diagnosis_archive) != diagnosis_spec["archive_sha256"]:
        raise ProspectiveInputInvalid("cross-factor diagnosis archive hash mismatch")
    if not diagnosis_sidecar.is_file() or diagnosis_sidecar.read_text(encoding="utf-8").split()[0] != diagnosis_spec["archive_sha256"]:
        raise ProspectiveInputInvalid("cross-factor diagnosis archive sidecar mismatch")
    for payload in (diagnosis_state, diagnosis_decision):
        if payload.get("historical_soft_metric_classification") != "SOFT_JOINT_METRIC_REJECTED" or payload.get("historical_classification_modified") is not False:
            raise ProspectiveInputInvalid("cross-factor diagnosis changed the historical classification")
        if payload.get("test_data_accessed") is not False or payload.get("candidate_gnn_training_started") is not False:
            raise ProspectiveInputInvalid("cross-factor diagnosis violated its no-test/no-GNN scope")
    archive = Path(str(HISTORICAL_SOFT_ROOT) + ".tar.gz")
    if not archive.is_file() or sha256_file(archive) != "f5a0a73cb45098994afa3d60de44eb4c54c010d6814d963e3d4c3ba69c219f74":
        raise ProspectiveInputInvalid("soft v2 archive hash mismatch")
    replay_artifact_hashes = {
        "geometry_metrics.json": "b841f960d7a4805c5add50a8e65b8a53662ab296b9605b8d8584e6d59d92876c",
        "knn_locality.json": "8602a0e97bb279ec2fb85e5636b0eb71e324f95a5a8d95f88cf79ab5d5624d57",
        "oof_predictions.csv": "13deeecff25813ca4d98a6dcead07ea27c0e7dcaee16d78a7894b1eeaa565fe9",
        "metric_training_audit.json": "c6c656f88541e5908f82914fe110ee99f7835b8d40145ca26fb6520e3f1bf641",
    }
    for relative, expected in replay_artifact_hashes.items():
        _require_hash(HISTORICAL_SOFT_ROOT / relative, expected, f"soft-v2 replay anchor {relative}")
    high_archive = HISTORICAL_HIGH_ROOT / "high-regime-prior-model-blind-validation-global4-v1-20260826.tar.gz"
    if not high_archive.is_file() or sha256_file(high_archive) != "050a64b0cde5bdb17ababa9833e5ee578b448c9b19da31fbb00f0251ec82491c":
        raise ProspectiveInputInvalid("high-regime archive hash mismatch")
    _require_hash(HISTORICAL_HIGH_ROOT / "preregistration.json", "098240232515aacfb47f8f557765bb3c824faf37d60b8ae680f6e1b2cf2c7260", "high-regime preregistration")
    _require_hash(
        REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth",
        "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4",
        "frozen high-regime VAE checkpoint",
    )
    _require_hash(
        HISTORICAL_HIGH_ROOT / "source_dependency_manifest.json",
        config["authoritative_inputs"]["high_regime_v1"]["source_dependency_manifest_sha256"],
        "high-regime source dependency manifest",
    )
    try:
        frozen_dependency_entries = _frozen_dependency_entries()
    except ProtocolError as exc:
        raise ProspectiveInputInvalid(str(exc)) from exc
    return {
        "status": "VALID", "identity": identity, "historical_soft_metric_classification": classification,
        "historical_classification_modified": False, "split_sha256": logical,
        "soft_state": soft_state, "soft_decision": soft_decision, "soft_inventory": inventory,
        "split_manifest": split, "soft_snapshot_manifest": snapshot_manifest,
        "diagnosis_decision": diagnosis_decision, "test_accessed": False,
        "high_regime_archive_sha256": sha256_file(high_archive),
        "frozen_high_regime_dependency_count": len(frozen_dependency_entries),
        "frozen_high_regime_dependency_root": stable_hash(frozen_dependency_entries),
    }


def _load_frozen_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise ProspectiveInputInvalid(f"missing frozen module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProspectiveInputInvalid(f"cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    parent_name, _, child = name.rpartition(".")
    if parent_name:
        parent = __import__(parent_name, fromlist=[child])
        setattr(parent, child, module)
    spec.loader.exec_module(module)
    return module


def load_frozen_soft_runtime() -> dict[str, Any]:
    modules = {}
    ordered = (
        ("analyse.performance_metric_quick_gate", FROZEN_SOFT_SNAPSHOT / "analyse/performance_metric_quick_gate.py", "performance"),
        ("analyse.joint_configuration_metric_gate", FROZEN_SOFT_SNAPSHOT / "analyse/joint_configuration_metric_gate.py", "joint"),
        ("analyse.joint_pair_power_projection_audit", FROZEN_SOFT_SNAPSHOT / "analyse/joint_pair_power_projection_audit.py", "pair_audit"),
        ("analyse.joint_configuration_soft_metric_gate", FROZEN_SOFT_SNAPSHOT / "analyse/joint_configuration_soft_metric_gate.py", "soft"),
    )
    for canonical, path, key in ordered:
        modules[key] = _load_frozen_module(canonical, path)
    for module in modules.values():
        if hasattr(module, "REPO_ROOT"):
            module.REPO_ROOT = REPO_ROOT
    soft = modules["soft"]
    if Path(soft.train_soft_metric_encoder.__code__.co_filename).resolve() != (FROZEN_SOFT_SNAPSHOT / "analyse/joint_configuration_soft_metric_gate.py").resolve():
        raise ProspectiveInputInvalid("soft scorer silently resolved to live source")
    modules["soft_config_path"] = FROZEN_SOFT_SNAPSHOT / "configs/joint_configuration_soft_metric_gate_v2.yaml"
    return modules


class RetainedFoldScorer:
    def __init__(
        self, *, fold: int, train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]],
        hp_standardizer: Any, seed_models: dict[int, Any], seed_train_embeddings: dict[int, Any],
        seed_heldout_embeddings: dict[int, Any],
    ) -> None:
        self.fold = fold
        self.train_rows = train_rows
        self.heldout_rows = heldout_rows
        self.hp_standardizer = hp_standardizer
        self.seed_models = seed_models
        self.seed_train_embeddings = seed_train_embeddings
        self.seed_heldout_embeddings = seed_heldout_embeddings


def _construct_soft_pairs(soft: Any, rows: list[dict[str, Any]], config: Mapping[str, Any]) -> Any:
    """Call the frozen pair constructor with the archived soft-metric values."""
    allowed = {str(row["joint_fingerprint"]) for row in rows}
    pair = config["soft_metric"]
    return soft.construct_soft_pairs(
        rows,
        allowed_joint_fingerprints=allowed,
        delta=float(pair["delta"]),
        minimum_weight=float(pair["minimum_pair_weight"]),
        reliability_fraction=float(pair["reliability_weight_fraction"]),
        pair_cap=int(pair["pair_cap"]),
        sampling_seed=int(pair["pair_sampling_seed"]),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _archived_replay_anchors() -> dict[str, Any]:
    geometry = _load_json(HISTORICAL_SOFT_ROOT / "geometry_metrics.json")
    locality = _load_json(HISTORICAL_SOFT_ROOT / "knn_locality.json")
    oof = _read_csv(HISTORICAL_SOFT_ROOT / "oof_predictions.csv")
    training = _load_json(HISTORICAL_SOFT_ROOT / "metric_training_audit.json")
    seed_rho: dict[str, float] = {}
    for row in geometry["folds"]:
        fold = int(row["fold"])
        for seed, value in row["seed_distance_gap_spearman"].items():
            seed_rho[f"fold={fold},seed={int(seed)}"] = float(value)
    if len(seed_rho) != 15:
        raise ProspectiveInputInvalid("archived geometry does not contain 15 fold-seed Spearman values")
    records = locality.get("joint_metric", {}).get("records", locality.get("records", []))
    top5 = {str(row["query_joint_fingerprint"]): list(row["neighbor_joint_fingerprints"]) for row in records}
    if len(top5) != 336 or any(len(value) != 5 for value in top5.values()):
        raise ProspectiveInputInvalid("archived locality does not contain 336 ordered top-5 lists")
    gap5 = locality["joint_metric"]["mean_local_performance_gap_at_5"]
    m1 = {f"fold={int(row['fold'])},joint={row['joint_fingerprint']}": float(row["M1_prediction"]) for row in oof}
    m2 = {f"fold={int(row['fold'])},joint={row['joint_fingerprint']}": float(row["M2_prediction"]) for row in oof}
    best = {f"fold={int(row['fold'])},seed={int(row['seed'])}": int(row["best_epoch"]) for row in training["runs"]}
    return {"seed_distance_gap_spearman": seed_rho, "top5_neighbor_fingerprints": top5, "metric_gap_at_5": float(gap5), "M1_oof": m1, "M2_oof": m2, "best_epochs": best}


def _fold_members(fold: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    train = fold.get("train_joint_fingerprints", fold.get("outer_train_joint_fingerprints", fold.get("train", [])))
    heldout = fold.get("heldout_joint_fingerprints", fold.get("outer_heldout_joint_fingerprints", fold.get("heldout", [])))
    return set(map(str, train)), set(map(str, heldout))


def replay_and_retain_scorers(runtime: Mapping[str, Any], authority: Mapping[str, Any], log: Callable[[str], None]) -> tuple[list[dict[str, Any]], list[RetainedFoldScorer], dict[str, Any]]:
    import numpy as np
    from scipy.stats import spearmanr

    soft = runtime["soft"]
    frozen_config = soft.load_config(runtime["soft_config_path"])
    configurations, inventory, _provenance, split_manifest = soft.load_main_data_and_split(frozen_config)
    validate_historical_identity(inventory)
    if split_manifest.get("split_sha256") != FROZEN_HISTORICAL_IDENTITY["split_sha256"]:
        raise ProspectiveInputInvalid("frozen loader returned a different split")
    rows_by_fp = {str(row["joint_fingerprint"]): dict(row) for row in configurations}
    if len(rows_by_fp) != 336:
        raise ProspectiveInputInvalid("frozen loader did not return 336 unique configurations")
    archived = _archived_replay_anchors()
    reproduced: dict[str, Any] = {
        "seed_distance_gap_spearman": {}, "top5_neighbor_fingerprints": {},
        "M1_oof": {}, "M2_oof": {}, "best_epochs": {},
    }
    retained: list[RetainedFoldScorer] = []
    all_gap5: list[float] = []
    for fold_index, fold in enumerate(split_manifest["folds"]):
        train_ids, heldout_ids = _fold_members(fold)
        if not train_ids or not heldout_ids or train_ids & heldout_ids:
            raise ProspectiveInputInvalid(f"invalid fold membership in fold {fold_index}")
        train_rows = sorted((rows_by_fp[fp] for fp in train_ids), key=lambda row: row["joint_fingerprint"])
        heldout_rows = sorted((rows_by_fp[fp] for fp in heldout_ids), key=lambda row: row["joint_fingerprint"])
        if {row["architecture_fingerprint"] for row in train_rows} & {row["architecture_fingerprint"] for row in heldout_rows}:
            raise ProspectiveInputInvalid("architecture crossed the frozen outer split")
        scaler = soft.joint_v1.fit_hp_standardizer(train_rows)
        full_pairs = _construct_soft_pairs(soft, train_rows, frozen_config)
        seed_models: dict[int, Any] = {}
        seed_train: dict[int, Any] = {}
        seed_heldout: dict[int, Any] = {}
        seed_m1: dict[int, Any] = {}
        seed_m2: dict[int, Any] = {}
        train_y = np.asarray([row["validation_mean"] for row in train_rows], dtype=float)
        train_se = np.asarray([row["validation_se"] for row in train_rows], dtype=float)
        heldout_y = np.asarray([row["validation_mean"] for row in heldout_rows], dtype=float)
        train_fp = [str(row["joint_fingerprint"]) for row in train_rows]
        query_fp = [str(row["joint_fingerprint"]) for row in heldout_rows]
        for seed in METRIC_SEEDS:
            log(f"fold {fold_index}/4 metric seed {seed}: replaying inner selection and full-outer refit")
            train_embedding, heldout_embedding, model, audit = soft.train_soft_metric_encoder(
                train_rows, heldout_rows, seed, frozen_config, scaler, full_pairs
            )
            train_embedding = np.asarray(train_embedding, dtype=float)
            heldout_embedding = np.asarray(heldout_embedding, dtype=float)
            distances = np.linalg.norm(heldout_embedding[:, None, :] - train_embedding[None, :, :], axis=2)
            gaps = np.abs(heldout_y[:, None] - train_y[None, :])
            rho = float(spearmanr(distances.ravel(), gaps.ravel()).statistic)
            reproduced["seed_distance_gap_spearman"][f"fold={fold_index},seed={seed}"] = rho
            m1 = soft.joint_v1.weighted_knn_predict(
                train_embedding, train_y, heldout_embedding, 5,
                train_fingerprints=train_fp, query_fingerprints=query_fp,
            )
            m2, gp_audit = soft.joint_v1.exact_matern_gp_predict(
                train_embedding, train_y, train_se, heldout_embedding,
                standardize_inputs=False, ard=False, protocol_name="M2",
            )
            if gp_audit.get("input_standardized") is not False or gp_audit.get("ard") is not False:
                raise ScorerReplayIdentityMismatch("replayed M2 violated frozen isotropic geometry")
            seed_m1[seed] = np.asarray(m1, dtype=float)
            seed_m2[seed] = np.asarray(m2, dtype=float)
            seed_models[seed] = model
            seed_train[seed] = train_embedding
            seed_heldout[seed] = heldout_embedding
            best_epoch = int(audit["best_epoch"])
            reproduced["best_epochs"][f"fold={fold_index},seed={seed}"] = best_epoch
        m1_mean = np.mean(np.stack([seed_m1[s] for s in METRIC_SEEDS]), axis=0)
        m2_mean = np.mean(np.stack([seed_m2[s] for s in METRIC_SEEDS]), axis=0)
        mean_distances = np.mean(np.stack([
            np.linalg.norm(seed_heldout[s][:, None, :] - seed_train[s][None, :, :], axis=2)
            for s in METRIC_SEEDS
        ]), axis=0)
        for query_index, fingerprint in enumerate(query_fp):
            reproduced["M1_oof"][f"fold={fold_index},joint={fingerprint}"] = float(m1_mean[query_index])
            reproduced["M2_oof"][f"fold={fold_index},joint={fingerprint}"] = float(m2_mean[query_index])
            order = sorted(range(len(train_fp)), key=lambda idx: (float(mean_distances[query_index, idx]), train_fp[idx]))[:5]
            reproduced["top5_neighbor_fingerprints"][fingerprint] = [train_fp[idx] for idx in order]
            all_gap5.append(float(np.mean([abs(heldout_y[query_index] - train_y[idx]) for idx in order])))
        retained.append(RetainedFoldScorer(
            fold=fold_index, train_rows=train_rows, heldout_rows=heldout_rows,
            hp_standardizer=scaler, seed_models=seed_models,
            seed_train_embeddings=seed_train, seed_heldout_embeddings=seed_heldout,
        ))
    reproduced["metric_gap_at_5"] = float(np.mean(all_gap5))
    if reproduced["best_epochs"] != archived["best_epochs"]:
        raise ScorerReplayIdentityMismatch("best epochs differ from the archived 15 runs")
    assert_scorer_replay_identity(reproduced, archived, REPLAY_ATOL)
    audit = {
        "status": "IDENTICAL", "absolute_tolerance": REPLAY_ATOL,
        "metric_seeds": list(METRIC_SEEDS), "fold_count": 5, "model_count": 15,
        "candidate_labels_used": False, "posterior_variance_used": False,
        "M2_ard": False, "M2_input_standardized": False,
        "reproduced": reproduced, "archived": archived,
        "models_persisted_to_disk": False, "test_accessed": False,
    }
    validate_scorer_protocol(audit)
    return configurations, retained, audit


def _load_frozen_high_module() -> Any:
    # Frozen helper imports are resolved from the archived source tree.  The
    # soft runtime has already fixed the canonical analysis package modules.
    snapshot = FROZEN_HIGH_SNAPSHOT
    import analyse as analyse_package
    frozen_analyse_path = str((snapshot / "analyse").resolve())
    original_paths = list(getattr(analyse_package, "__path__", []))
    analyse_package.__path__ = [
        frozen_analyse_path,
        *[path for path in original_paths if str(path) != frozen_analyse_path],
    ]
    # No high-regime analysis dependency should have been imported by the soft
    # runtime.  Remove an incidental live import so the archived package path
    # wins deterministically.
    for source in (snapshot / "analyse").glob("*.py"):
        canonical = f"analyse.{source.stem}"
        if canonical in {
            "analyse.performance_metric_quick_gate", "analyse.joint_configuration_metric_gate",
            "analyse.joint_pair_power_projection_audit", "analyse.joint_configuration_soft_metric_gate",
        }:
            continue
        existing = sys.modules.pop(canonical, None)
        if existing is not None and hasattr(analyse_package, source.stem):
            delattr(analyse_package, source.stem)
    for name, relative in (
        ("architecture_diagnostics", "architecture_diagnostics.py"),
        ("eval_utils", "eval_utils.py"),
    ):
        path = snapshot / relative
        if name not in sys.modules or Path(getattr(sys.modules[name], "__file__", "")).resolve() != path.resolve():
            _load_frozen_module(name, path)
    module = _load_frozen_module(
        "analyse.high_regime_prior_model_blind_validation",
        snapshot / "analyse/high_regime_prior_model_blind_validation.py",
    )
    if not list(analyse_package.__path__) or str(list(analyse_package.__path__)[0]) != frozen_analyse_path:
        raise ProspectiveInputInvalid("frozen analyse source is not first in import precedence")
    required_frozen_modules = {
        "analyse.high_regime_prior_model_blind_validation": snapshot / "analyse/high_regime_prior_model_blind_validation.py",
        "analyse.original_head_confirmatory_attribution_v2": snapshot / "analyse/original_head_confirmatory_attribution_v2.py",
        "analyse.original_head_deterministic_batch_pilot": snapshot / "analyse/original_head_deterministic_batch_pilot.py",
    }
    for canonical, expected_path in required_frozen_modules.items():
        loaded = sys.modules.get(canonical)
        loaded_path = _module_source_path(loaded) if loaded is not None else None
        if loaded_path is None or loaded_path.resolve() != expected_path.resolve():
            raise ProspectiveInputInvalid(f"frozen high-regime import resolved outside the source snapshot: {canonical}")
    if hasattr(module, "REPO_ROOT"):
        module.REPO_ROOT = REPO_ROOT
    dependency_hashes = {
        str(Path(entry["path"]).resolve()): entry["sha256"]
        for entry in _frozen_dependency_entries()
    }
    for imported in list(sys.modules.values()):
        source = _module_source_path(imported)
        if source is None:
            continue
        source_resolved = str(source.resolve())
        if source_resolved in dependency_hashes and sha256_file(source) != dependency_hashes[source_resolved]:
            raise ProspectiveInputInvalid(f"loaded high-regime dependency hash mismatch: {source}")
    return module


def frozen_module_import_smoke() -> dict[str, Any]:
    module = _load_frozen_high_module()
    required_v1_functions = (
        "load_joint_vae", "config_from_igraph", "validate_static_config",
        "architecture_fingerprint", "realize_global4_hp", "hp_fingerprint",
    )
    missing = [name for name in required_v1_functions if not callable(getattr(module.v1, name, None))]
    if missing or not callable(getattr(module, "joint_config_fingerprint", None)):
        raise ProtocolError(f"frozen candidate-generator API is incomplete: {missing}")
    import analyse as analyse_package
    frozen_path = str((FROZEN_HIGH_SNAPSHOT / "analyse").resolve())
    loaded_paths = {
        canonical: str(Path(sys.modules[canonical].__file__).resolve())
        for canonical in (
            "analyse.high_regime_prior_model_blind_validation",
            "analyse.original_head_confirmatory_attribution_v2",
            "analyse.original_head_deterministic_batch_pilot",
        )
    }
    return {
        "passed": True,
        "frozen_path_first": bool(list(analyse_package.__path__)) and str(list(analyse_package.__path__)[0]) == frozen_path,
        "loaded_module_paths": loaded_paths,
        "candidate_generator_api_complete": True,
        "candidate_generation_started": False,
        "pre_eval_freeze_started": False,
        "gnn_training_started": False,
        "test_accessed": False,
    }


def _candidate_row_from_config(high: Any, model: Any, graph: Any, pool: int, slot: int, raw_hp_vector: Sequence[float]) -> dict[str, Any]:
    config = high.v1.config_from_igraph(model, graph)
    valid, reason = high.v1.validate_static_config(config)
    if not valid:
        raise CandidateGenerationFailed(f"decoded architecture is invalid: {reason}")
    arch_fp = high.v1.architecture_fingerprint(config)
    raw64 = [float(value) for value in raw_hp_vector]
    hp_fp = high.v1.hp_fingerprint(raw64)
    joint_fp = high.joint_config_fingerprint(arch_fp, hp_fp)
    operations = list(config["operations"])
    edges = [list(map(int, edge)) for edge in config["edges"]]
    row = {
        "pool_id": pool, "batch_id": pool, "slot_id": slot,
        "architecture_slot": slot, "config_id": f"prospective-p{pool}-s{slot}",
        "joint_fingerprint": joint_fp, "joint_config_fingerprint": joint_fp,
        "architecture_fingerprint": arch_fp, "hp_fingerprint": hp_fp,
        "operations": operations, "edges": edges,
        "effective_layers": int(config["effective_layers"]),
        "raw_hp_vector": raw64, "hp_regime": "hp_high",
        "generation_pool_seed": MASTER_CANDIDATE_SEED + pool,
        "candidate_validation_label_available": False,
    }
    row.update(high.v1.realize_global4_hp(row))
    row["raw_hp"] = {key: row[key] for key in ("lr", "dropout", "hidden_dim", "l2")}
    return row


def generate_candidates(configurations: Sequence[Mapping[str, Any]], log: Callable[[str], None]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import torch

    high = _load_frozen_high_module()
    checkpoint = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
    _require_hash(checkpoint, "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4", "frozen VAE checkpoint")
    model = high.v1.load_joint_vae(checkpoint, "global4", device="cpu")
    model.eval()
    model.arch_vae.eval()
    historical_arch = {str(row["architecture_fingerprint"]) for row in configurations}
    historical_joint = {str(row["joint_fingerprint"]) for row in configurations}
    accepted_arch: set[str] = set()
    accepted_joint: set[str] = set()
    candidates: list[dict[str, Any]] = []
    pool_audit = []
    for pool in range(POOL_COUNT):
        seed = MASTER_CANDIDATE_SEED + pool
        torch_generator = torch.Generator(device="cpu")
        torch_generator.manual_seed(seed)
        hp_rng = np.random.default_rng(seed)
        draws = rejected_invalid = rejected_seen = 0
        pool_rows: list[dict[str, Any]] = []
        while len(pool_rows) < CANDIDATES_PER_POOL and draws < MAX_GENERATION_DRAWS_PER_POOL:
            size = min(GENERATION_BATCH_SIZE, MAX_GENERATION_DRAWS_PER_POOL - draws)
            latent = torch.randn((size, 12), generator=torch_generator, device="cpu")
            with torch.no_grad():
                decoded = model.arch_vae.decode(latent, stochastic=False)
            for graph in decoded:
                draws += 1
                try:
                    graph_config = high.v1.config_from_igraph(model, graph)
                    valid, _reason = high.v1.validate_static_config(graph_config)
                    if not valid:
                        raise ValueError("invalid decoded architecture")
                    architecture = high.v1.architecture_fingerprint(graph_config)
                except (TypeError, ValueError, KeyError, IndexError, CandidateGenerationFailed):
                    rejected_invalid += 1
                    continue
                if architecture in historical_arch or architecture in accepted_arch:
                    rejected_seen += 1
                    continue
                # HP sampling happens only after an architecture passes the
                # model-blind validity/novelty rejection, matching the frozen
                # proposal's separation between architecture and HP streams.
                raw_hp = hp_rng.uniform(0.80, 0.95, size=4).astype(np.float64).tolist()
                row = _candidate_row_from_config(high, model, graph, pool, len(pool_rows), raw_hp)
                if row["joint_fingerprint"] in historical_joint or row["joint_fingerprint"] in accepted_joint:
                    rejected_seen += 1
                    continue
                accepted_arch.add(row["architecture_fingerprint"])
                accepted_joint.add(row["joint_fingerprint"])
                pool_rows.append(row)
                if len(pool_rows) == CANDIDATES_PER_POOL:
                    break
            log(f"pool {pool}: accepted {len(pool_rows)}/4 after {draws} decoded draws")
        if len(pool_rows) != CANDIDATES_PER_POOL:
            raise CandidateGenerationFailed(f"pool {pool} exhausted {MAX_GENERATION_DRAWS_PER_POOL} draws")
        candidates.extend(pool_rows)
        pool_audit.append({"pool_id": pool, "seed": seed, "draws": draws, "invalid_rejections": rejected_invalid, "seen_rejections": rejected_seen})
    checks = validate_candidate_manifest(candidates, historical_arch, historical_joint)
    return candidates, {
        "status": "VALID", "candidate_count": len(candidates), "pool_audit": pool_audit,
        "novelty_checks": checks, "predictor_accessed_during_generation": False,
        "validation_labels_accessed_during_generation": False,
        "architecture_distribution": "frozen model-blind VAE prior deterministic greedy decode",
        "hp_distribution": "HIGH_REGIME_PRIOR Uniform[0.80,0.95)^4",
        "rng_contract": "independent CPU torch and NumPy streams seeded directly with 2026082801+pool",
    }


def score_candidates(runtime: Mapping[str, Any], retained: Sequence[RetainedFoldScorer], candidates: Sequence[Mapping[str, Any]], log: Callable[[str], None]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    soft = runtime["soft"]
    candidate_rows = [dict(row) for row in sorted(candidates, key=lambda item: (int(item["pool_id"]), str(item["joint_fingerprint"])))]
    query_fp = [str(row["joint_fingerprint"]) for row in candidate_rows]
    m1_fold_seed: dict[int, dict[int, Sequence[float]]] = {}
    m2_fold_seed: dict[int, dict[int, Sequence[float]]] = {}
    b3_by_fold: dict[int, Sequence[float]] = {}
    audits = []
    for scorer in retained:
        fold = scorer.fold
        train_y = np.asarray([row["validation_mean"] for row in scorer.train_rows], dtype=float)
        train_se = np.asarray([row["validation_se"] for row in scorer.train_rows], dtype=float)
        train_fp = [str(row["joint_fingerprint"]) for row in scorer.train_rows]
        m1_fold_seed[fold] = {}
        m2_fold_seed[fold] = {}
        for seed in METRIC_SEEDS:
            candidate_embedding = soft.joint_v1.encode_rows_with_model(
                scorer.seed_models[seed], candidate_rows, scorer.hp_standardizer, "full_joint"
            )
            train_embedding = scorer.seed_train_embeddings[seed]
            m1 = soft.joint_v1.weighted_knn_predict(
                train_embedding, train_y, candidate_embedding, 5,
                train_fingerprints=train_fp, query_fingerprints=query_fp,
            )
            m2, gp_audit = soft.joint_v1.exact_matern_gp_predict(
                train_embedding, train_y, train_se, candidate_embedding,
                standardize_inputs=False, ard=False, protocol_name="M2",
            )
            m1_fold_seed[fold][seed] = np.asarray(m1, dtype=float)
            m2_fold_seed[fold][seed] = np.asarray(m2, dtype=float)
            audits.append({"fold": fold, "seed": seed, "method": "M2", **gp_audit})
        train_explicit = np.concatenate((soft.joint_v1._explicit_graph_matrix(scorer.train_rows), soft.joint_v1._raw_hp_matrix(scorer.train_rows)), axis=1)
        candidate_explicit = np.concatenate((soft.joint_v1._explicit_graph_matrix(candidate_rows), soft.joint_v1._raw_hp_matrix(candidate_rows)), axis=1)
        b3, gp_audit = soft.joint_v1.exact_matern_gp_predict(
            train_explicit, train_y, train_se, candidate_explicit,
            standardize_inputs=True, ard=True, protocol_name="B3",
        )
        b3_by_fold[fold] = np.asarray(b3, dtype=float)
        audits.append({"fold": fold, "method": "B3", **gp_audit})
        log(f"fold {fold}: scored 32 candidates with M1, M2, and B3")
    m1 = aggregate_cross_fitted_scores(m1_fold_seed)
    m2 = aggregate_cross_fitted_scores(m2_fold_seed)
    if set(b3_by_fold) != set(range(5)):
        raise ProtocolError("B3 did not produce all five fold scores")
    b3 = np.mean(np.stack([b3_by_fold[f] for f in range(5)]), axis=0).tolist()
    rows = []
    for index, candidate in enumerate(candidate_rows):
        rows.append({
            "pool_id": int(candidate["pool_id"]), "joint_fingerprint": candidate["joint_fingerprint"],
            "architecture_fingerprint": candidate["architecture_fingerprint"],
            "M1": float(m1[index]), "M2": float(m2[index]), "B3": float(b3[index]),
        })
    audit = {
        "candidate_labels_used": False, "posterior_variance_used": False,
        "M1_seed_then_fold_arithmetic_mean": True, "M2_seed_then_fold_arithmetic_mean": True,
        "B3_fold_arithmetic_mean": True, "M2_isotropic": True, "M2_input_standardized": False,
        "B3_ard": True, "B3_input_standardized": True, "gp_runs": audits,
    }
    return rows, audit


def make_selections(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_pool: dict[int, list[Mapping[str, Any]]] = {pool: [] for pool in range(POOL_COUNT)}
    for row in predictions:
        by_pool[int(row["pool_id"])].append(row)
    rng = random.Random(RANDOM_SELECTION_SEED)
    result: dict[str, Any] = {}
    for pool in range(POOL_COUNT):
        rows = sorted(by_pool[pool], key=lambda row: str(row["joint_fingerprint"]))
        if len(rows) != CANDIDATES_PER_POOL:
            raise ProtocolError("selection pool does not have four candidates")
        result[str(pool)] = {
            "M1": select_pool_argmax(rows, "M1"), "M2": select_pool_argmax(rows, "M2"),
            "B3": select_pool_argmax(rows, "B3"),
            "R": str(rows[rng.randrange(CANDIDATES_PER_POOL)]["joint_fingerprint"]),
        }
    return {"pools": result, "random_seed": RANDOM_SELECTION_SEED, "tie_break": "joint_fingerprint lexicographic ascending"}


def ensure_output_policy(output: Path, *, preflight_ready: bool = False) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise ProtocolError("formal output path exists and is not a directory")
    entries = [path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()]
    if not entries:
        return
    allowed = {
        "logs/py_compile.log", "logs/pytest_focused.log", "logs/synthetic_smoke.log", "logs/runner.log",
        "logs/scorer_replay.log", "logs/candidate_generation.log", "logs/evaluation.log", "logs/analysis.log",
    }
    if not preflight_ready or not set(entries).issubset(allowed):
        raise ProtocolError("formal result directory exists and is non-empty; refusing overwrite")


def _dry_run_payload(config_path: Path, formal_output: Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_frozen_protocol(config)
    return {
        "protocol_version": PROTOCOL_VERSION, "dry_run": True,
        "formal_result_root": str(formal_output),
        "formal_result_root_created": False, "formal_data_read": False,
        "scorer_replay_started": False, "candidate_generation_started": False,
        "gnn_training_started": False, "candidate_gnn_training_started": False,
        "test_accessed": False, "would_run_focused_tests": True,
        "would_freeze_all_eight_pools_before_evaluation": True,
    }


def execute_dry_run(config_path: Path = DEFAULT_CONFIG, formal_output: Path = DEFAULT_OUTPUT) -> int:
    print(json.dumps(_dry_run_payload(config_path, formal_output), indent=2, sort_keys=True))
    return 0


def _git_capture(args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return completed.stdout


def write_environment_and_git(output: Path, *, before: bool) -> None:
    status_name = "git_status_before.txt" if before else "git_status_after.txt"
    dump_text(output / status_name, _git_capture(["status", "--short", "--branch"]))
    if before:
        try:
            import numpy
            numpy_version = numpy.__version__
        except Exception:
            numpy_version = "unavailable"
        try:
            import torch
            torch_version = torch.__version__
            cuda_visible = torch.cuda.is_available()
        except Exception:
            torch_version = "unavailable"
            cuda_visible = False
        dump_json(output / "environment.json", {
            "created_at": utc_now(), "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "numpy": numpy_version, "torch": torch_version,
            "cuda_available_to_parent": cuda_visible,
            "scorer_device": "cpu", "evaluator_device_policy": "two frozen GPU shards after pre-eval admission",
            "branch": _git_capture(["branch", "--show-current"]).strip(),
            "head": _git_capture(["rev-parse", "HEAD"]).strip(),
            "dirty_tree": bool(_git_capture(["status", "--porcelain"]).strip()),
        })
        dump_text(output / "git_diff.patch", _git_capture(["diff", "--binary", "HEAD", "--"]) + _git_capture(["diff", "--binary", "--cached", "HEAD", "--"]))


def _mandatory_snapshot_sources() -> dict[str, tuple[Path, str]]:
    return {
        "configs/joint_metric_prospective_selection_gate_v1.yaml": (REPO_ROOT / "configs/joint_metric_prospective_selection_gate_v1.yaml", "current prospective protocol"),
        "analyse/joint_metric_prospective_selection_gate.py": (REPO_ROOT / "analyse/joint_metric_prospective_selection_gate.py", "current prospective implementation"),
        "scripts/run_joint_metric_prospective_selection_gate.sh": (REPO_ROOT / "scripts/run_joint_metric_prospective_selection_gate.sh", "current one-shot runner"),
        "tests/test_joint_metric_prospective_selection_gate.py": (REPO_ROOT / "tests/test_joint_metric_prospective_selection_gate.py", "focused protocol tests"),
        "configs/joint_configuration_soft_metric_gate_v2.yaml": (FROZEN_SOFT_SNAPSHOT / "configs/joint_configuration_soft_metric_gate_v2.yaml", "frozen soft-v2 config"),
        "analyse/joint_configuration_soft_metric_gate.py": (FROZEN_SOFT_SNAPSHOT / "analyse/joint_configuration_soft_metric_gate.py", "frozen soft-v2 scorer"),
        "analyse/joint_configuration_metric_gate.py": (FROZEN_SOFT_SNAPSHOT / "analyse/joint_configuration_metric_gate.py", "frozen joint-v1 dependency"),
        "analyse/performance_metric_quick_gate.py": (FROZEN_SOFT_SNAPSHOT / "analyse/performance_metric_quick_gate.py", "frozen performance-v2 dependency"),
        "analyse/joint_pair_power_projection_audit.py": (FROZEN_SOFT_SNAPSHOT / "analyse/joint_pair_power_projection_audit.py", "frozen engineering dependency"),
        "analyse/high_regime_prior_model_blind_validation.py": (FROZEN_HIGH_SNAPSHOT / "analyse/high_regime_prior_model_blind_validation.py", "frozen candidate generator"),
        "analyse/run_high_regime_prior_model_blind_validation_labels.py": (FROZEN_HIGH_SNAPSHOT / "analyse/run_high_regime_prior_model_blind_validation_labels.py", "frozen GNN evaluator runner"),
        "analyse/analyse_high_regime_prior_model_blind_validation.py": (FROZEN_HIGH_SNAPSHOT / "analyse/analyse_high_regime_prior_model_blind_validation.py", "frozen evaluator analysis"),
        "eval_utils.py": (FROZEN_HIGH_SNAPSHOT / "eval_utils.py", "frozen GNN evaluator"),
        "architecture_diagnostics.py": (FROZEN_HIGH_SNAPSHOT / "architecture_diagnostics.py", "frozen canonical architecture helper"),
    }


def _module_source_path(module: Any) -> Path | None:
    raw = getattr(module, "__file__", None)
    if not raw:
        return None
    path = Path(raw).resolve()
    if path.suffix in {".pyc", ".pyo"}:
        source = Path(importlib.util.source_from_cache(str(path)))
        if source.is_file():
            path = source
    return path if path.is_file() and path.suffix == ".py" else None


def snapshot_sources(output: Path, *, modules: Mapping[str, Any] | None = None) -> dict[str, Any]:
    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    included: dict[str, dict[str, Any]] = {}

    def include(relative: str, source: Path, reason: str, runtime_module: str | None = None) -> None:
        relative = Path(relative).as_posix()
        if not source.is_file():
            raise ProtocolError(f"source snapshot cannot read {source}")
        payload = source.read_bytes()
        target = snapshot / relative
        if target.exists() and target.read_bytes() != payload:
            raise ProtocolError(f"source snapshot collision at {relative}")
        atomic_write_bytes(target, payload)
        record = included.setdefault(relative, {
            "repository_relative_path": relative, "absolute_runtime_path": str(source.resolve()),
            "snapshot_relative_path": relative, "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "reason_included": [], "runtime_modules": [],
        })
        if reason not in record["reason_included"]:
            record["reason_included"].append(reason)
        if runtime_module and runtime_module not in record["runtime_modules"]:
            record["runtime_modules"].append(runtime_module)

    for relative, (source, reason) in _mandatory_snapshot_sources().items():
        include(relative, source, reason)

    # Include the complete evaluator dependency closure recorded by the frozen run.
    dependency_manifest_path = HISTORICAL_HIGH_ROOT / "source_dependency_manifest.json"
    if dependency_manifest_path.is_file():
        dependency_manifest = _load_json(dependency_manifest_path)
        entries = dependency_manifest.get("files", dependency_manifest.get("entries", []))
        for entry in entries:
            relative = str(entry.get("repository_relative_path", entry.get("relative_path", entry.get("path", ""))))
            candidate_path = Path(relative)
            if candidate_path.is_absolute():
                try:
                    relative = candidate_path.resolve().relative_to(REPO_ROOT).as_posix()
                except ValueError as exc:
                    raise ProtocolError(f"frozen dependency lies outside repository: {candidate_path}") from exc
            if "/source_snapshot/" in relative:
                relative = relative.split("/source_snapshot/", 1)[1]
            if not relative or relative.startswith("results/"):
                continue
            candidates = (FROZEN_HIGH_SNAPSHOT / relative, REPO_ROOT / relative)
            source = next((path for path in candidates if path.is_file()), None)
            if source is None:
                raise ProtocolError(f"frozen evaluator dependency is missing: {relative}")
            expected = entry.get("sha256")
            if expected and sha256_file(source) != expected:
                raise ProtocolError(f"frozen evaluator dependency hash differs: {relative}")
            include(relative, source, "frozen high-regime evaluator dependency closure")

    for name, module in (modules or sys.modules).items():
        source = _module_source_path(module)
        if source is None:
            continue
        try:
            relative = source.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        if relative.startswith("results/posthoc/joint-configuration-soft-metric-gate-global4-v2/source_snapshot/"):
            relative = relative.split("/source_snapshot/", 1)[1]
        elif relative.startswith("results/search/high-regime-prior-model-blind-validation-global4-v1/source_snapshot/"):
            relative = relative.split("/source_snapshot/", 1)[1]
        include(relative, source, "runtime imported local Python module", name)

    entries = [included[key] for key in sorted(included)]
    imported = [entry for entry in entries if entry["runtime_modules"]]
    root = stable_hash([{key: entry[key] for key in ("repository_relative_path", "size_bytes", "sha256")} for entry in entries])
    dump_json(snapshot / "imported_local_modules.json", {"format_version": PROTOCOL_VERSION, "modules": imported})
    dump_json(snapshot / "sha256_manifest.json", {"format_version": PROTOCOL_VERSION, "entries": entries, "manifest_root_sha256": root})
    return {"entry_count": len(entries), "imported_module_count": len(imported), "manifest_root_sha256": root}


def validate_source_snapshot(snapshot_root: Path, required_paths: Sequence[str] | None = None) -> dict[str, Any]:
    manifest_path = snapshot_root / "sha256_manifest.json"
    if not manifest_path.is_file():
        missing = [relative for relative in (required_paths or REQUIRED_SOURCE_SNAPSHOT_PATHS) if not (snapshot_root / relative).is_file()]
        if missing:
            raise ProtocolError(f"required source omitted from snapshot: {missing[0]}")
        return {"passed": True, "status": "VALID", "entry_count": len(required_paths or REQUIRED_SOURCE_SNAPSHOT_PATHS), "all_required_sources_copied_as_bytes": True}
    manifest = _load_json(manifest_path)
    entries = manifest["entries"]
    paths = set()
    for entry in entries:
        relative = str(entry["snapshot_relative_path"])
        path = snapshot_root / relative
        if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]) or sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"source snapshot verification failed: {relative}")
        paths.add(relative)
    for relative in required_paths or _mandatory_snapshot_sources().keys():
        if relative not in paths:
            raise ProtocolError(f"required source omitted from snapshot: {relative}")
    root = stable_hash([{key: entry[key] for key in ("repository_relative_path", "size_bytes", "sha256")} for entry in entries])
    if root != manifest["manifest_root_sha256"]:
        raise ProtocolError("source snapshot manifest root mismatch")
    return {"passed": True, "status": "VALID", "entry_count": len(entries), "manifest_root_sha256": root, "all_required_sources_copied_as_bytes": True}


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
    if archive.exists() or (sidecar and sidecar.exists()):
        raise ProtocolError("archive or SHA sidecar already exists")
    validate_source_snapshot(output / "source_snapshot")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    try:
        with tarfile.open(temporary, "w:gz") as handle:
            handle.add(output, arcname=output.name, filter=lambda info: None if any(part == "__pycache__" for part in Path(info.name).parts) or Path(info.name).suffix in {".pt", ".pth", ".ckpt", ".pyc"} else info)
        digest = sha256_file(temporary)
        os.replace(temporary, archive)
        if sidecar:
            dump_text(sidecar, f"{digest}  {archive.name}\n")
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _frozen_dependency_entries() -> list[dict[str, str]]:
    original = _load_json(HISTORICAL_HIGH_ROOT / "source_dependency_manifest.json")
    entries = original.get("files", [])
    adapted: list[dict[str, str]] = []
    for entry in entries:
        original_path = Path(entry["path"])
        expected = str(entry["sha256"])
        try:
            relative = original_path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            raise ProtocolError(f"high-regime dependency is outside repository: {original_path}")
        if "/source_snapshot/" in relative:
            relative = relative.split("/source_snapshot/", 1)[1]
        source = FROZEN_HIGH_SNAPSHOT / relative
        if not source.is_file():
            source = REPO_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ProtocolError(f"frozen evaluator dependency cannot be recovered exactly: {relative}")
        adapted.append({"path": str(source.resolve()), "sha256": expected})
    if not adapted:
        raise ProtocolError("frozen evaluator dependency manifest is empty")
    return adapted


def _evaluation_task_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in build_evaluation_tasks(candidates):
        candidate = task["candidate"]
        pool = int(task["pool_id"])
        seed = int(task["training_seed"])
        joint = str(task["joint_fingerprint"])
        rows.append({
            "task_stage": "initial", "pool_id": pool,
            "batch_id": str(candidate["batch_id"]), "architecture_slot": int(candidate["architecture_slot"]),
            "config_id": str(candidate["config_id"]),
            "architecture_fingerprint": candidate["architecture_fingerprint"],
            "hp_fingerprint": candidate["hp_fingerprint"], "joint_config_fingerprint": joint,
            "hp_regime": "hp_high", "seed": seed,
            "task_key": str(task["task_key"]),
            "training_protocol_identity": str(task["training_protocol_identity"]),
            "assigned_gpu": 0 if pool < 4 else 1, "selected_by": ["prospective_selection_gate_v1"],
            "operations": candidate["operations"], "edges": candidate["edges"],
            "effective_layers": candidate["effective_layers"], "raw_hp_vector": candidate["raw_hp_vector"],
            "lr": candidate["lr"], "dropout": candidate["dropout"], "hidden_dim": candidate["hidden_dim"],
            "l2": candidate["l2"], "gat_heads": candidate["gat_heads"],
            "sage_aggr": candidate["sage_aggr"], "gin_eps": candidate["gin_eps"],
            "gat_heads_by_layer": candidate["gat_heads_by_layer"],
            "sage_aggr_by_layer": candidate["sage_aggr_by_layer"],
            "gin_eps_by_layer": candidate["gin_eps_by_layer"],
        })
    reject_test_fields(rows)
    return rows


def _prepare_evaluator_adapter(output: Path, candidates: Sequence[Mapping[str, Any]]) -> tuple[Path, list[dict[str, Any]], str]:
    adapter = output / "evaluation/evaluator_adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    tasks = _evaluation_task_rows(candidates)
    firewall = {"passed": True, "test_named_fields": [], "track_test": False, "created_at": utc_now()}
    preregistration = {
        "format_version": PROTOCOL_VERSION,
        "training": {
            "dataset": "Cora", "split": "public", "graph_transform": "NormalizeFeatures",
            "dataset_content_fingerprint": "373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7",
            "split_fingerprint": "bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09",
            "max_epochs": 300, "patience": 80, "maximum_system_retries": 2,
            "training_seeds": list(TRAINING_SEEDS), "track_test": False,
            "shard_to_gpu_uuid": {
                "0": "GPU-5496be78-3b71-4978-3960-b6ed926ebe49",
                "1": "GPU-197ae2b9-74cd-6afc-446d-7bc9e98176f9",
            },
        },
    }
    dependency_manifest = {"format_version": PROTOCOL_VERSION, "files": _frozen_dependency_entries()}
    dump_json(adapter / "pre_label_firewall_audit.json", firewall)
    dump_json(adapter / "selected_union_tasks.json", tasks)
    dump_json(adapter / "preregistration.json", preregistration)
    dump_json(adapter / "source_dependency_manifest.json", dependency_manifest)
    ordered = ["pre_label_firewall_audit.json", "selected_union_tasks.json", "preregistration.json", "source_dependency_manifest.json"]
    immutable_files = [{"relative_path": relative, "sha256": sha256_file(adapter / relative)} for relative in ordered]
    root = stable_hash([(entry["relative_path"], entry["sha256"]) for entry in immutable_files])
    dump_json(adapter / "pre_label_immutable_manifest.json", {
        "format_version": PROTOCOL_VERSION, "files": immutable_files, "manifest_root_hash": root,
    })
    return adapter, tasks, root


def run_frozen_gnn_evaluation(output: Path, candidates: Sequence[Mapping[str, Any]], update_progress: Callable[..., None]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verify_pre_eval_freeze(output / "pre_eval")
    frozen_candidate_payload = _load_json(output / "pre_eval/candidate_manifest.json")
    frozen_candidates = frozen_candidate_payload.get("candidates", frozen_candidate_payload) if isinstance(frozen_candidate_payload, Mapping) else frozen_candidate_payload
    assert_candidate_manifest_immutable(list(frozen_candidates), list(candidates))
    adapter, tasks, adapter_root = _prepare_evaluator_adapter(output, frozen_candidates)
    frozen_script = FROZEN_HIGH_SNAPSHOT / "analyse/run_high_regime_prior_model_blind_validation_labels.py"
    _require_hash(frozen_script, FROZEN_SOURCE_HASHES["analyse/run_high_regime_prior_model_blind_validation_labels.py"], "frozen high-regime label runner")
    processes = []
    handles = []
    for shard in (0, 1):
        shard_output = output / f"evaluation/gpu{shard}"
        shard_output.mkdir(parents=True, exist_ok=True)
        log_path = output / f"logs/evaluation_gpu{shard}.log"
        handle = log_path.open("w", encoding="utf-8")
        handles.append(handle)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(shard)
        environment["PYTHONHASHSEED"] = "0"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        inherited_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(FROZEN_HIGH_SNAPSHOT), str(REPO_ROOT)] + ([inherited_pythonpath] if inherited_pythonpath else [])
        )
        command = [
            sys.executable, str(frozen_script), "--root", str(adapter), "--stage", "initial",
            "--shard", str(shard), "--output", str(shard_output), "--cora-root", "Cora", "--device", "cuda:0",
        ]
        processes.append((shard, subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)))
    update_progress(stage="EVALUATION_RUNNING", completed_gnn_runs=0, total_gnn_runs=TOTAL_GNN_RUNS, failed_runs=0)
    return_codes = {}
    tasks_by_shard = {
        shard: [row for row in tasks if int(row["assigned_gpu"]) == shard]
        for shard in (0, 1)
    }
    try:
        while processes:
            remaining = []
            completed_rows = 0
            failed_rows = 0
            completed_by_shard = {0: 0, 1: 0}
            for shard in (0, 1):
                path = output / f"evaluation/gpu{shard}/seed_results.jsonl"
                if path.is_file():
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            completed_rows += 1
                            completed_by_shard[shard] += 1
                            if row.get("terminal_status") != "success":
                                failed_rows += 1
            for shard, process in processes:
                code = process.poll()
                if code is None:
                    remaining.append((shard, process))
                else:
                    return_codes[str(shard)] = code
            current_task = None
            for shard in (0, 1):
                if completed_by_shard[shard] < len(tasks_by_shard[shard]):
                    current_task = tasks_by_shard[shard][completed_by_shard[shard]]
                    break
            update_progress(
                stage="EVALUATION_RUNNING", completed_gnn_runs=completed_rows,
                total_gnn_runs=TOTAL_GNN_RUNS, failed_runs=failed_rows,
                current_pool=None if current_task is None else int(current_task["pool_id"]),
                current_config=None if current_task is None else str(current_task["joint_config_fingerprint"]),
                current_seed=None if current_task is None else int(current_task["seed"]),
            )
            processes = remaining
            if processes:
                time.sleep(2.0)
    finally:
        for _shard, process in processes:
            process.terminate()
        for handle in handles:
            handle.close()
    raw: list[dict[str, Any]] = []
    task_lookup = {(row["joint_config_fingerprint"], int(row["seed"])): row for row in tasks}
    for shard in (0, 1):
        path = output / f"evaluation/gpu{shard}/seed_results.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            archived = json.loads(line)
            reject_test_fields(archived)
            joint = str(archived.get("joint_config_fingerprint"))
            seed = int(archived["seed"])
            task = task_lookup.get((joint, seed))
            if task is None:
                raise ProtocolError("frozen evaluator returned a foreign task")
            expected_fields = {
                "pool_id": int(task["pool_id"]),
                "architecture_fingerprint": str(task["architecture_fingerprint"]),
                "hp_fingerprint": str(task["hp_fingerprint"]),
                "task_key": str(task["task_key"]),
                "assigned_gpu": int(task["assigned_gpu"]),
            }
            for field, expected in expected_fields.items():
                if field not in archived or archived[field] != expected:
                    raise ProtocolError(f"frozen evaluator result identity mismatch for {field}")
            raw.append({
                "pool_id": int(task["pool_id"]), "joint_fingerprint": joint, "training_seed": seed,
                "validation_accuracy": archived.get("val_acc"),
                "terminal_status": archived.get("terminal_status"),
                "attempt_count": archived.get("attempt_count", archived.get("attempts", 1)),
            })
    audit = {
        "adapter_manifest_root": adapter_root, "task_count": len(tasks), "shard_return_codes": return_codes,
        "frozen_runner": str(frozen_script), "maximum_system_retries": 2,
        "track_test": False, "candidate_replacement": False, "scorer_updated": False,
        "pool_to_shard": {str(pool): 0 if pool < 4 else 1 for pool in range(POOL_COUNT)},
        "raw_result_count": len(raw),
        "all_shards_exited_zero": all(code == 0 for code in return_codes.values()) and set(return_codes) == {"0", "1"},
    }
    return raw, audit


def paired_comparisons(per_pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {(int(row["pool_id"]), str(row["method"])): row for row in per_pool}
    output: dict[str, Any] = {}
    for left, right in (("M2", "B3"), ("M1", "B3"), ("M1", "R"), ("M2", "R"), ("B3", "R")):
        values = [float(rows[(pool, left)]["selected_validation_mean"]) - float(rows[(pool, right)]["selected_validation_mean"]) for pool in range(POOL_COUNT)]
        output[f"{left}_vs_{right}"] = {
            "pool_differences": values,
            "mean_difference": _mean(values), "wins": sum(value > 1e-12 for value in values),
            "losses": sum(value < -1e-12 for value in values), "ties": sum(abs(value) <= 1e-12 for value in values),
        }
    return output


def bootstrap_comparisons(comparisons: Mapping[str, Any]) -> dict[str, Any]:
    return {name: pool_bootstrap(payload["pool_differences"], replicates=10000, seed=0) for name, payload in comparisons.items()}


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.8f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def render_final_report(
    *, decision: Mapping[str, Any], authority: Mapping[str, Any] | None,
    replay: Mapping[str, Any] | None, candidates: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]], selections: Mapping[str, Any] | None,
    evaluation_stats: Sequence[Mapping[str, Any]], per_pool: Sequence[Mapping[str, Any]],
    prospective: Mapping[str, Any] | None, comparisons: Mapping[str, Any] | None,
    bootstrap: Mapping[str, Any] | None, state: Mapping[str, Any],
) -> str:
    classification = str(decision["classification"])
    n_disagreement = decision.get("n_disagreement", "NA")
    lines = [f"# Soft Joint Metric Prospective Selection Gate v1", ""]
    lines += [f"## {REPORT_SECTIONS[0]}", "", f"Frozen classification: `{classification}`. The historical soft-metric classification remains `SOFT_JOINT_METRIC_REJECTED`. "
              f"The pre-evaluation M2/B3 disagreement count is {n_disagreement}. This gate evaluates only frozen posterior-mean selection; it does not authorize BO or uncertainty-based acquisition.", ""]
    identity = (authority or {}).get("identity", {})
    lines += [f"## {REPORT_SECTIONS[1]}", "", _markdown_table([{
        "joint configs": identity.get("unique_joint_configurations"), "architectures": identity.get("unique_architectures"),
        "pools": identity.get("unique_pools"), "seed records": identity.get("raw_seed_records"),
        "split SHA": (authority or {}).get("split_sha256"), "replay": (replay or {}).get("status", "NOT_STARTED"),
    }], ["joint configs", "architectures", "pools", "seed records", "split SHA", "replay"]), ""]
    lines += [f"## {REPORT_SECTIONS[2]}", "", "Candidates were drawn by the archived model-blind VAE decoder. Each pool used the preregistered seed `2026082801+p`; HP latent components followed the frozen HIGH_REGIME_PRIOR on `[0.80,0.95)`. No predictor score or validation label entered proposal or rejection sampling.", ""]
    novelty = [{"candidates": len(candidates), "unique architecture": len({c.get("architecture_fingerprint") for c in candidates}), "unique joint": len({c.get("joint_fingerprint") for c in candidates}), "historical architecture overlap": 0 if candidates else "NA", "historical joint overlap": 0 if candidates else "NA"}]
    lines += [f"## {REPORT_SECTIONS[3]}", "", _markdown_table(novelty, list(novelty[0])), ""]
    lines += [f"## {REPORT_SECTIONS[4]}", "", _markdown_table([{
        "candidate count": len(candidates), "pool count": len({c.get('pool_id') for c in candidates}),
        "selection frozen": bool(selections), "disagreement": n_disagreement,
        "GNN authorized": decision.get("gnn_runs_authorized", False),
    }], ["candidate count", "pool count", "selection frozen", "disagreement", "GNN authorized"]), ""]
    lines += [f"## {REPORT_SECTIONS[5]}", "", _markdown_table(predictions or [{"pool_id": "NA", "joint_fingerprint": "not available", "M1": None, "M2": None, "B3": None}], ["pool_id", "joint_fingerprint", "M1", "M2", "B3"]), ""]
    selection_rows = []
    if selections:
        for pool, value in selections.get("pools", {}).items():
            selection_rows.append({"pool": pool, **value, "M2 differs B3": value["M2"] != value["B3"]})
    lines += [f"## {REPORT_SECTIONS[6]}", "", _markdown_table(selection_rows or [{"pool": "NA", "M1": "NA", "M2": "NA", "B3": "NA", "R": "NA", "M2 differs B3": "NA"}], ["pool", "M1", "M2", "B3", "R", "M2 differs B3"]), ""]
    lines += [f"## {REPORT_SECTIONS[7]}", "", _markdown_table([{
        "authorized tasks": TOTAL_GNN_RUNS if decision.get("gnn_runs_authorized") else 0,
        "completed runs": state.get("completed_gnn_runs", 0), "failed runs": state.get("failed_runs", 0),
        "evaluation complete": classification in {"METRIC_SELECTION_PROSPECTIVE_REJECTED", "METRIC_KNN_PROSPECTIVE_SIGNAL_ONLY", "METRIC_GP_PROSPECTIVE_SIGNAL"},
    }], ["authorized tasks", "completed runs", "failed runs", "evaluation complete"]), ""]
    stat_lookup = {str(row.get("joint_fingerprint")): row for row in evaluation_stats}
    outcome_lookup = {(int(row["pool_id"]), str(row["method"])): row for row in per_pool}
    pool_truth_rows = []
    for candidate in candidates:
        fp = str(candidate["joint_fingerprint"])
        stats = stat_lookup.get(fp, {})
        selected = (selections or {}).get("pools", {}).get(str(candidate["pool_id"]), {})
        pool_truth_rows.append({
            "pool": candidate["pool_id"], "joint": fp, "lr": candidate.get("lr"), "dropout": candidate.get("dropout"),
            "hidden": candidate.get("hidden_dim"), "l2": candidate.get("l2"), "actual mean": stats.get("validation_mean"),
            "M1": selected.get("M1") == fp, "M2": selected.get("M2") == fp,
            "B3": selected.get("B3") == fp, "R": selected.get("R") == fp,
            "oracle": next((outcome_lookup[(int(candidate["pool_id"]), method)]["observed_oracle"] for method in ("M1", "M2", "B3", "R") if (int(candidate["pool_id"]), method) in outcome_lookup), None),
            "M1 regret": outcome_lookup.get((int(candidate["pool_id"]), "M1"), {}).get("simple_regret"),
            "M2 regret": outcome_lookup.get((int(candidate["pool_id"]), "M2"), {}).get("simple_regret"),
            "B3 regret": outcome_lookup.get((int(candidate["pool_id"]), "B3"), {}).get("simple_regret"),
            "R regret": outcome_lookup.get((int(candidate["pool_id"]), "R"), {}).get("simple_regret"),
        })
    lines += [f"## {REPORT_SECTIONS[8]}", "", _markdown_table(pool_truth_rows or [{"pool": "NA", "joint": "evaluation not run", "lr": None, "dropout": None, "hidden": None, "l2": None, "actual mean": None, "oracle": None, "M1": None, "M2": None, "B3": None, "R": None, "M1 regret": None, "M2 regret": None, "B3 regret": None, "R regret": None}], ["pool", "joint", "lr", "dropout", "hidden", "l2", "actual mean", "oracle", "M1", "M2", "B3", "R", "M1 regret", "M2 regret", "B3 regret", "R regret"]), ""]
    method_summary = (prospective or {}).get("methods", {})
    selected_rows = [{"method": method, "selected validation mean": values.get("selected_validation_mean")} for method, values in method_summary.items()]
    lines += [f"## {REPORT_SECTIONS[9]}", "", _markdown_table(selected_rows or [{"method": "NA", "selected validation mean": None}], ["method", "selected validation mean"]), ""]
    regret_rows = [{"method": method, "mean regret": values.get("mean_simple_regret"), "median regret": values.get("median_simple_regret")} for method, values in method_summary.items()]
    lines += [f"## {REPORT_SECTIONS[10]}", "", _markdown_table(regret_rows or [{"method": "NA", "mean regret": None, "median regret": None}], ["method", "mean regret", "median regret"]), ""]
    hit_rows = [{"method": method, "hits": values.get("true_best_hits"), "hit rate": values.get("true_best_hit_rate"), "within 0.006": values.get("within_0_006_of_oracle_rate")} for method, values in method_summary.items()]
    lines += [f"## {REPORT_SECTIONS[11]}", "", _markdown_table(hit_rows or [{"method": "NA", "hits": None, "hit rate": None, "within 0.006": None}], ["method", "hits", "hit rate", "within 0.006"]), ""]
    m2_gate = decision.get("M2_gate", {})
    lines += [f"## {REPORT_SECTIONS[12]}", "", _markdown_table([m2_gate or {"P1": "NA", "P2": "NA", "P3": "NA", "P4": "NA", "P5": "NA", "passed": "NA"}], ["P1", "P2", "P3", "P4", "P5", "passed"]), ""]
    m1_gate = decision.get("M1_gate", {})
    lines += [f"## {REPORT_SECTIONS[13]}", "", _markdown_table([m1_gate or {"P1": "NA", "P2": "NA", "P3": "NA", "P4": "NA", "P5": "NA", "passed": "NA"}], ["P1", "P2", "P3", "P4", "P5", "passed"]), ""]
    random_summary = method_summary.get("R", {})
    lines += [f"## {REPORT_SECTIONS[14]}", "", _markdown_table([{"frozen random seed": RANDOM_SELECTION_SEED, **random_summary}], ["frozen random seed", "selected_validation_mean", "mean_simple_regret", "true_best_hit_rate"]), "", "The expected-random pool mean, when present, is descriptive only and never replaces the frozen random action.", ""]
    bootstrap_rows = [{"comparison": name, **value} for name, value in (bootstrap or {}).items()]
    lines += [f"## {REPORT_SECTIONS[15]}", "", _markdown_table(bootstrap_rows or [{"comparison": "NA", "mean": None, "ci95_low": None, "ci95_high": None, "positive_sign_count": None}], ["comparison", "mean", "ci95_low", "ci95_high", "positive_sign_count"]), "", "Every interval resamples the eight pools. The 24 training-seed observations are not treated as independent experimental units.", ""]
    lines += [f"## {REPORT_SECTIONS[16]}", "", f"`{classification}`", "", "This decision uses only the frozen prospective rules. Random does not participate in the primary gate, and M1 cannot produce the GP-signal classification.", ""]
    lines += [f"## {REPORT_SECTIONS[17]}", "", "The result is screening evidence for frozen selection on model-blind architectures with HIGH_REGIME_PRIOR HP. It does not revise the historical offline rejection and it does not adapt candidates or scorers after observing new labels.", ""]
    lines += [f"## {REPORT_SECTIONS[18]}", "", "Even a prospective signal would not validate GP uncertainty, EI/qLogEI, sequential BO, other datasets, the full HP space, or a final publication claim. The observed oracle is only the best three-seed mean among four evaluated candidates in a pool.", ""]
    if classification == "PROSPECTIVE_LOW_DISAGREEMENT":
        next_action = "No GNN evaluation was authorized; retain the immutable pre-evaluation bundle."
    elif classification == "METRIC_SELECTION_PROSPECTIVE_REJECTED":
        next_action = "Stop the current joint-metric search route. Do not rescue this gate by changing delta, embedding dimension, loss, or candidate pools."
    elif classification in {"METRIC_KNN_PROSPECTIVE_SIGNAL_ONLY", "METRIC_GP_PROSPECTIVE_SIGNAL"}:
        next_action = "A separately preregistered confirmation may be considered; do not update Phase4 or BO automatically."
    else:
        next_action = "Resolve only the recorded identity, generation, or completeness failure; do not reinterpret partial scientific results."
    lines += [f"## {REPORT_SECTIONS[19]}", "", next_action, ""]
    lines += [f"## {REPORT_SECTIONS[20]}", "", _markdown_table([{
        "branch": state.get("branch"), "HEAD": state.get("head"), "dirty tree": state.get("dirty_tree"),
        "stage": state.get("stage"), "elapsed seconds": state.get("elapsed_seconds"),
        "result directory": state.get("output"),
        "archive": str(Path(str(state.get("output"))) .with_name(Path(str(state.get("output"))).name + ".tar.gz")) if state.get("output") else None,
        "archive SHA": "computed after packaging; authoritative sibling .sha256 sidecar",
        "test accessed": False,
    }], ["branch", "HEAD", "dirty tree", "stage", "elapsed seconds", "result directory", "archive", "archive SHA", "test accessed"]), "",
              "The parent process enumerates repository-local modules from `sys.modules`. The frozen evaluator runs in two child processes; its complete hash-verified 25-file dependency closure is copied byte-for-byte into the same source snapshot and distinguished in the snapshot reasons.", ""]
    return "\n".join(lines).rstrip() + "\n"


def assert_complete_final_report(text: str) -> None:
    for section in REPORT_SECTIONS:
        if f"## {section}" not in text:
            raise ProtocolError(f"final report lacks section: {section}")
    if text.count("| ---") < 10:
        raise ProtocolError("final report lacks required core result tables")


def _initial_state(output: Path, config_sha: str, source_sha: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION, "status": "running", "stage": "IMPLEMENTATION_READY",
        "started_at": utc_now(), "updated_at": utc_now(), "output": str(output),
        "config_sha256": config_sha, "source_sha256": source_sha,
        "branch": _git_capture(["branch", "--show-current"]).strip(), "head": _git_capture(["rev-parse", "HEAD"]).strip(),
        "dirty_tree": bool(_git_capture(["status", "--porcelain"]).strip()),
        "completed_gnn_runs": 0, "total_gnn_runs": TOTAL_GNN_RUNS,
        "failed_runs": 0, "current_pool": None, "current_config": None, "current_seed": None,
        "test_accessed": False, "candidate_labels_used_to_update_scorer": False,
    }


def update_state(output: Path, **changes: Any) -> dict[str, Any]:
    path = output / "state.json"
    state = _load_json(path) if path.is_file() else {}
    state.update(changes)
    state["updated_at"] = utc_now()
    dump_json(path, state)
    return state


def _write_missing_outputs(output: Path, classification: str, error: str | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("pre_eval", "evaluation", "logs"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    json_defaults: dict[str, Any] = {
        "engineering_fix.json": engineering_fix_payload(),
        "frozen_protocol.json": {"protocol_version": PROTOCOL_VERSION, "status": "NOT_COMPLETED"},
        "input_integrity.json": {"status": "NOT_COMPLETED", "test_accessed": False},
        "scorer_replay_identity.json": {"status": "NOT_COMPLETED", "candidate_labels_used": False},
        "pre_eval/candidate_manifest.json": {"status": "NOT_COMPLETED", "candidates": []},
        "pre_eval/selections.json": {"status": "NOT_COMPLETED", "pools": {}},
        "pre_eval/scorer_identity.json": {"status": "NOT_COMPLETED"},
        "pre_eval/freeze_manifest.json": {"status": "NOT_COMPLETED", "immutable_entries": [], "root_sha256": None},
        "evaluation/task_manifest.json": {"authorized": False, "tasks": []},
        "evaluation/training_audit.json": {"status": "NOT_COMPLETED", "track_test": False},
        "prospective_metrics.json": {"status": "NOT_COMPLETED", "methods": {}},
        "paired_pool_comparisons.json": {"status": "NOT_COMPLETED"},
        "bootstrap_results.json": {"status": "NOT_COMPLETED", "unit": "pool"},
        "decision.json": {
            "classification": classification, "historical_soft_metric_classification": "SOFT_JOINT_METRIC_REJECTED",
            "historical_classification_modified": False, "test_accessed": False,
        },
    }
    for relative, payload in json_defaults.items():
        path = output / relative
        if not path.exists():
            dump_json(path, payload)
    for relative, fields in CSV_FIELDS.items():
        path = output / relative
        if not path.exists():
            dump_csv(path, [], fields)
    root_file = output / "pre_eval/pre_eval_root_sha256.txt"
    if not root_file.exists():
        dump_text(root_file, "NOT_FROZEN\n")
    for relative in (
        "logs/py_compile.log", "logs/pytest_focused.log", "logs/synthetic_smoke.log", "logs/scorer_replay.log",
        "logs/candidate_generation.log", "logs/evaluation.log", "logs/analysis.log", "logs/runner.log",
        "git_status_before.txt", "git_status_after.txt", "git_diff.patch",
    ):
        path = output / relative
        if not path.exists():
            dump_text(path, "")
    if not (output / "environment.json").exists():
        dump_json(output / "environment.json", {"created_at": utc_now(), "python": sys.version, "platform": platform.platform()})
    if error is not None:
        dump_json(output / "error.json", {"classification": classification, "message": error, "created_at": utc_now()})


def _finalize_terminal(
    output: Path, decision: Mapping[str, Any], *, authority: Mapping[str, Any] | None = None,
    replay: Mapping[str, Any] | None = None, candidates: Sequence[Mapping[str, Any]] = (),
    predictions: Sequence[Mapping[str, Any]] = (), selections: Mapping[str, Any] | None = None,
    evaluation_stats: Sequence[Mapping[str, Any]] = (), per_pool: Sequence[Mapping[str, Any]] = (),
    prospective: Mapping[str, Any] | None = None, comparisons: Mapping[str, Any] | None = None,
    bootstrap: Mapping[str, Any] | None = None, error: str | None = None,
) -> None:
    classification = str(decision["classification"])
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise ProtocolError(f"unsupported classification: {classification}")
    decision_payload = dict(decision)
    decision_payload.update({
        "historical_soft_metric_classification": "SOFT_JOINT_METRIC_REJECTED",
        "historical_classification_modified": False, "test_accessed": False,
        "posterior_variance_used": False, "online_update_used": False,
    })
    dump_json(output / "decision.json", decision_payload)
    if authority is not None:
        dump_json(output / "input_integrity.json", {
            "status": "VALID", "identity": authority["identity"], "split_sha256": authority["split_sha256"],
            "historical_soft_metric_classification": authority["historical_soft_metric_classification"], "test_accessed": False,
        })
    if replay is not None:
        dump_json(output / "scorer_replay_identity.json", replay)
    _write_missing_outputs(output, classification, error)
    if classification in {"PROSPECTIVE_INPUT_INVALID", "SCORER_REPLAY_IDENTITY_MISMATCH", "CANDIDATE_GENERATION_FAILED"} and (output / "PRE_EVAL_SELECTION_FROZEN").exists():
        (output / "PRE_EVAL_SELECTION_FROZEN").unlink()
    state = update_state(
        output, status="failed" if classification == "RUN_FAILED" else "complete",
        stage="ANALYSIS_COMPLETE" if classification != "RUN_FAILED" else "RUN_FAILED",
        classification=classification, finished_at=utc_now(),
    )
    started = state.get("started_monotonic")
    if isinstance(started, (int, float)):
        state = update_state(output, elapsed_seconds=time.monotonic() - float(started))
    report = render_final_report(
        decision=decision_payload, authority=authority, replay=replay, candidates=candidates,
        predictions=predictions, selections=selections, evaluation_stats=evaluation_stats,
        per_pool=per_pool, prospective=prospective, comparisons=comparisons, bootstrap=bootstrap, state=state,
    )
    assert_complete_final_report(report)
    dump_text(output / "final_report.md", report)
    write_environment_and_git(output, before=False)
    snapshot_sources(output)
    validate_source_snapshot(output / "source_snapshot")
    write_manifest(output)


def write_failure_bundle(
    output: Path, error: Any, archive_path: Path | None = None,
    *, config_sha256: str = "UNAVAILABLE", source_sha256: str = "UNAVAILABLE",
) -> dict[str, Any]:
    error = str(error)
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "state.json").exists():
        state = _initial_state(output, config_sha256, source_sha256)
        state.update({"started_monotonic": time.monotonic()})
        dump_json(output / "state.json", state)
    _write_missing_outputs(output, "RUN_FAILED", error)
    _finalize_terminal(output, {"classification": "RUN_FAILED", "error": error}, error=error)
    if (output / "COMPLETE").exists():
        (output / "COMPLETE").unlink()
    dump_text(output / "FAILED", "")
    write_manifest(output)
    result = {"classification": "RUN_FAILED", "output": str(output), "archive": None, "sha256": None}
    if archive_path is not None:
        sidecar = Path(str(archive_path) + ".sha256")
        digest = create_archive(output, archive_path, sidecar)
        result.update({"archive": str(archive_path), "sha256": digest})
    return result


def _synthetic_completed_case(*, m2_pass: bool, m1_pass: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    methods = {
        "B3": {"selected_validation_mean": 0.80, "mean_simple_regret": 0.010, "median_simple_regret": 0.010, "true_best_hits": 2, "true_best_hit_rate": 0.25, "within_0_006_of_oracle_rate": 0.25},
        "M2": {"selected_validation_mean": 0.81 if m2_pass else 0.80, "mean_simple_regret": 0.005 if m2_pass else 0.011, "median_simple_regret": 0.005, "true_best_hits": 3 if m2_pass else 1, "true_best_hit_rate": 0.375, "within_0_006_of_oracle_rate": 0.50},
        "M1": {"selected_validation_mean": 0.81 if m1_pass else 0.80, "mean_simple_regret": 0.005 if m1_pass else 0.011, "median_simple_regret": 0.005, "true_best_hits": 3 if m1_pass else 1, "true_best_hit_rate": 0.375, "within_0_006_of_oracle_rate": 0.50},
        "R": {"selected_validation_mean": 0.79, "mean_simple_regret": 0.020, "median_simple_regret": 0.020, "true_best_hits": 1, "true_best_hit_rate": 0.125, "within_0_006_of_oracle_rate": 0.125},
    }
    rows = []
    for pool in range(POOL_COUNT):
        for method in ("B3", "M2", "M1", "R"):
            passes = (method == "M2" and m2_pass) or (method == "M1" and m1_pass)
            rows.append({
                "pool_id": pool, "method": method,
                "selected_joint_fingerprint": f"{method}-{pool}" if method != "B3" else f"B3-{pool}",
                "selected_validation_mean": 0.81 if passes else (0.80 if method != "R" else 0.79),
            })
    return {"methods": methods}, rows


def synthetic_smoke(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    validate_frozen_protocol(load_config(config_path))
    rejected_summary, rejected_rows = _synthetic_completed_case(m2_pass=False, m1_pass=False)
    knn_summary, knn_rows = _synthetic_completed_case(m2_pass=False, m1_pass=True)
    gp_summary, gp_rows = _synthetic_completed_case(m2_pass=True, m1_pass=True)
    cases = {
        "PROSPECTIVE_INPUT_INVALID": "PROSPECTIVE_INPUT_INVALID",
        "SCORER_REPLAY_IDENTITY_MISMATCH": "SCORER_REPLAY_IDENTITY_MISMATCH",
        "CANDIDATE_GENERATION_FAILED": "CANDIDATE_GENERATION_FAILED",
        "PROSPECTIVE_LOW_DISAGREEMENT": classify_prospective(evaluation_complete=False, n_disagreement=4)["classification"],
        "PROSPECTIVE_EVALUATION_INCOMPLETE": classify_prospective(evaluation_complete=False, n_disagreement=5)["classification"],
        "METRIC_SELECTION_PROSPECTIVE_REJECTED": classify_prospective(evaluation_complete=True, n_disagreement=8, prospective_summary=rejected_summary, per_pool_rows=rejected_rows)["classification"],
        "METRIC_KNN_PROSPECTIVE_SIGNAL_ONLY": classify_prospective(evaluation_complete=True, n_disagreement=8, prospective_summary=knn_summary, per_pool_rows=knn_rows)["classification"],
        "METRIC_GP_PROSPECTIVE_SIGNAL": classify_prospective(evaluation_complete=True, n_disagreement=8, prospective_summary=gp_summary, per_pool_rows=gp_rows)["classification"],
    }
    return {
        "passed": all(key == value for key, value in cases.items()),
        **{key: value == key for key, value in cases.items()},
        **{f"synthetic_{key.lower()}_case": value == key for key, value in cases.items()},
    }


def _logger(path: Path) -> Callable[[str], None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    def emit(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        print(line, flush=True)
    return emit


def execute_formal(config_path: Path, output: Path, config_sha: str, source_sha: str, *, preflight_ready: bool) -> int:
    ensure_output_policy(output, preflight_ready=preflight_ready)
    if sha256_file(config_path) != config_sha:
        raise ProtocolError("runner-provided config SHA does not match config bytes")
    if aggregate_source_sha256() != source_sha:
        raise ProtocolError("runner-provided source SHA does not match formal runtime bytes")
    config = load_config(config_path)
    validate_frozen_protocol(config)
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("logs", "pre_eval", "evaluation"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    state = _initial_state(output, config_sha, source_sha)
    state["started_monotonic"] = time.monotonic()
    dump_json(output / "state.json", state)
    dump_json(output / "frozen_protocol.json", config)
    dump_json(output / "engineering_fix.json", engineering_fix_payload())
    write_environment_and_git(output, before=True)
    scorer_log = _logger(output / "logs/scorer_replay.log")
    candidate_log = _logger(output / "logs/candidate_generation.log")
    evaluation_log = _logger(output / "logs/evaluation.log")
    analysis_log = _logger(output / "logs/analysis.log")

    try:
        authority = validate_authoritative_inputs(config)
        dump_json(output / "input_integrity.json", {
            "status": "VALID", "identity": authority["identity"], "split_sha256": authority["split_sha256"],
            "historical_soft_metric_classification": "SOFT_JOINT_METRIC_REJECTED",
            "historical_classification_modified": False, "test_accessed": False,
        })
    except ProspectiveInputInvalid as exc:
        dump_json(output / "input_integrity.json", {"status": "INVALID", "reason": str(exc), "test_accessed": False})
        _finalize_terminal(output, {"classification": "PROSPECTIVE_INPUT_INVALID", "reason": str(exc)})
        return 0

    try:
        os.environ["PROSPECTIVE_SCORER_DEVICE"] = "cpu"
        runtime = load_frozen_soft_runtime()
        configurations, retained, replay = replay_and_retain_scorers(runtime, authority, scorer_log)
        dump_json(output / "scorer_replay_identity.json", replay)
        update_state(output, stage="SCORER_REPLAY_COMPLETE")
    except (ProspectiveInputInvalid, ScorerReplayIdentityMismatch) as exc:
        classification = "PROSPECTIVE_INPUT_INVALID" if isinstance(exc, ProspectiveInputInvalid) else "SCORER_REPLAY_IDENTITY_MISMATCH"
        if classification == "PROSPECTIVE_INPUT_INVALID":
            dump_json(output / "input_integrity.json", {"status": "INVALID", "reason": str(exc), "test_accessed": False})
        dump_json(output / "scorer_replay_identity.json", {"status": "MISMATCH", "reason": str(exc), "test_accessed": False})
        _finalize_terminal(output, {"classification": classification, "reason": str(exc)}, authority=authority)
        return 0

    try:
        candidates, generation_audit = generate_candidates(configurations, candidate_log)
        update_state(output, stage="CANDIDATES_FROZEN")
    except ProspectiveInputInvalid as exc:
        dump_json(output / "input_integrity.json", {"status": "INVALID", "reason": str(exc), "test_accessed": False})
        _finalize_terminal(output, {"classification": "PROSPECTIVE_INPUT_INVALID", "reason": str(exc)}, authority=authority, replay=replay)
        return 0
    except CandidateGenerationFailed as exc:
        dump_json(output / "pre_eval/candidate_manifest.json", {"status": "FAILED", "reason": str(exc), "candidates": []})
        _finalize_terminal(output, {"classification": "CANDIDATE_GENERATION_FAILED", "reason": str(exc)}, authority=authority, replay=replay)
        return 0

    predictions, scoring_audit = score_candidates(runtime, retained, candidates, scorer_log)
    selections = make_selections(predictions)
    scorer_identity = {
        "status": "IDENTICAL", "replay": replay,
        "frozen_soft_config_sha256": FROZEN_HISTORICAL_IDENTITY["soft_config_sha256"],
        "frozen_soft_source_sha256": FROZEN_HISTORICAL_IDENTITY["soft_source_sha256"],
        "split_sha256": FROZEN_HISTORICAL_IDENTITY["split_sha256"],
        "scoring_audit": scoring_audit, "candidate_labels_used": False,
    }
    candidate_manifest = {
        "format_version": PROTOCOL_VERSION, "created_at": utc_now(), "candidates": candidates,
        "generation_audit": generation_audit, "candidate_count": len(candidates),
    }
    freeze_audit = freeze_pre_eval(
        output / "pre_eval", candidate_manifest, predictions, selections, scorer_identity,
        historical_input_hashes={
            "soft_archive_sha256": FROZEN_SOFT_METRIC_IDENTITY["archive_sha256"],
            "soft_config_sha256": FROZEN_SOFT_METRIC_IDENTITY["config_sha256"],
            "soft_source_sha256": FROZEN_SOFT_METRIC_IDENTITY["source_sha256"],
            "soft_snapshot_manifest_root": FROZEN_SOFT_METRIC_IDENTITY["source_snapshot_manifest_root"],
            "split_sha256": FROZEN_SOFT_METRIC_IDENTITY["split_sha256"],
            "high_regime_archive_sha256": authority["high_regime_archive_sha256"],
        },
        generator_identity={
            "source_sha256": FROZEN_SOURCE_HASHES["analyse/high_regime_prior_model_blind_validation.py"],
            "prospective_generator_adapter_sha256": sha256_file(Path(__file__).resolve()),
            "prospective_source_aggregate_sha256": source_sha,
            "prospective_config_sha256": config_sha,
            "architecture_helper_sha256": FROZEN_SOURCE_HASHES["architecture_diagnostics.py"],
            "checkpoint_sha256": "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4",
            "high_regime_archive_sha256": authority["high_regime_archive_sha256"],
            "high_regime_preregistration_sha256": "098240232515aacfb47f8f557765bb3c824faf37d60b8ae680f6e1b2cf2c7260",
            "dependency_closure_count": authority["frozen_high_regime_dependency_count"],
            "dependency_closure_root": authority["frozen_high_regime_dependency_root"],
            "master_seed": MASTER_CANDIDATE_SEED,
            "pool_seed_rule": "master_seed_plus_zero_based_pool_id",
            "architecture_rng": "torch_cpu_generator_seeded_directly_with_pool_seed",
            "hp_rng": "numpy_default_rng_seeded_directly_with_pool_seed",
            "hp_prior": "HIGH_REGIME_PRIOR",
        },
    )
    pre_eval_root = str(freeze_audit["root_sha256"])
    dump_text(output / "PRE_EVAL_SELECTION_FROZEN", "")
    verify_pre_eval_freeze(output / "pre_eval")
    update_state(output, stage="SELECTIONS_FROZEN", pre_eval_root_sha256=pre_eval_root)
    n_disagreement = disagreement_count(selections)
    update_state(output, stage="PRE_EVAL_AUDIT_PASSED", m2_b3_disagreement_pools=n_disagreement)
    del retained

    if n_disagreement < 5:
        update_state(output, total_gnn_runs=0, completed_gnn_runs=0, failed_runs=0)
        dump_json(output / "evaluation/task_manifest.json", {"authorized": False, "reason": "M2/B3 disagreement below 5 of 8", "tasks": [], "total_gnn_runs": 0})
        dump_json(output / "evaluation/training_audit.json", {
            "status": "NOT_AUTHORIZED", "completed_gnn_runs": 0, "total_gnn_runs": 0,
            "track_test": False, "candidate_replacement": False, "scorer_updated": False,
        })
        decision = classify_prospective(evaluation_complete=False, n_disagreement=n_disagreement)
        _finalize_terminal(output, decision, authority=authority, replay=replay, candidates=candidates, predictions=predictions, selections=selections)
        return 0

    tasks = build_evaluation_tasks(candidates)
    dump_json(output / "evaluation/task_manifest.json", {
        "authorized": True, "pre_eval_root_sha256": pre_eval_root,
        "task_count": len(tasks), "training_seeds": list(TRAINING_SEEDS), "tasks": tasks,
    })
    verify_pre_eval_freeze(output / "pre_eval")
    evaluation_log("pre-evaluation root verified; disagreement admission passed; starting frozen evaluator")

    def progress(**changes: Any) -> None:
        update_state(output, **changes)

    raw_results, training_audit = run_frozen_gnn_evaluation(output, candidates, progress)
    raw_csv = []
    for row in raw_results:
        raw_csv.append({
            "pool_id": row["pool_id"], "joint_fingerprint": row["joint_fingerprint"],
            "training_seed": row["training_seed"], "validation_accuracy": row.get("validation_accuracy"),
            "terminal_status": row.get("terminal_status"), "attempt_count": row.get("attempt_count"),
        })
    dump_csv(output / "evaluation/raw_seed_results.csv", raw_csv, CSV_FIELDS["evaluation/raw_seed_results.csv"])
    stats, complete = aggregate_evaluation_results(candidates, raw_results, allow_incomplete=True)
    complete = bool(complete and training_audit.get("all_shards_exited_zero"))
    dump_csv(output / "evaluation/config_3seed_statistics.csv", stats, CSV_FIELDS["evaluation/config_3seed_statistics.csv"])
    training_audit.update({
        "evaluation_complete": complete, "completed_gnn_runs": sum(row.get("terminal_status") == "success" for row in raw_results),
        "failed_runs": sum(row.get("terminal_status") != "success" for row in raw_results),
        "new_labels_added_to_scorer": False, "scorer_updated_after_freeze": False,
        "candidates_replaced_after_freeze": False, "pool_sequential_update": False,
    })
    assert_no_online_update(training_audit)
    dump_json(output / "evaluation/training_audit.json", training_audit)
    update_state(
        output, stage="EVALUATION_COMPLETE" if complete else "EVALUATION_INCOMPLETE",
        completed_gnn_runs=training_audit["completed_gnn_runs"], failed_runs=training_audit["failed_runs"],
    )
    if not complete:
        decision = classify_prospective(evaluation_complete=False, n_disagreement=n_disagreement)
        _finalize_terminal(output, decision, authority=authority, replay=replay, candidates=candidates,
                           predictions=predictions, selections=selections, evaluation_stats=stats)
        return 0

    verify_pre_eval_freeze(output / "pre_eval")
    per_pool, prospective = compute_prospective_metrics(candidates, selections, stats)
    comparisons = paired_comparisons(per_pool)
    bootstrap = bootstrap_comparisons(comparisons)
    decision = classify_prospective(
        evaluation_complete=True, n_disagreement=n_disagreement,
        prospective_summary=prospective, per_pool_rows=per_pool,
    )
    dump_csv(output / "selection_results_per_pool.csv", per_pool, CSV_FIELDS["selection_results_per_pool.csv"])
    dump_json(output / "prospective_metrics.json", prospective)
    dump_json(output / "paired_pool_comparisons.json", comparisons)
    dump_json(output / "bootstrap_results.json", bootstrap)
    update_state(output, stage="ANALYSIS_COMPLETE")
    analysis_log(f"frozen classification: {decision['classification']}")
    _finalize_terminal(
        output, decision, authority=authority, replay=replay, candidates=candidates,
        predictions=predictions, selections=selections, evaluation_stats=stats,
        per_pool=per_pool, prospective=prospective, comparisons=comparisons, bootstrap=bootstrap,
    )
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
    modes.add_argument("--frozen-module-import-smoke", action="store_true")
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
    if args.frozen_module_import_smoke:
        result = frozen_module_import_smoke()
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
