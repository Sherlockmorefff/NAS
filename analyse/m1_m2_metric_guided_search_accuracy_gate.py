#!/usr/bin/env python3
"""Frozen M1/M2 metric-guided search accuracy gate v1."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOCOL_VERSION = "m1_m2_metric_guided_search_accuracy_gate_global4_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/m1_m2_metric_guided_search_accuracy_gate_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/posthoc/m1-m2-metric-guided-search-accuracy-gate-global4-v1"
DEFAULT_ARCHIVE = Path(str(DEFAULT_OUTPUT) + ".tar.gz")

IMPLEMENTATION_FILES = (
    "configs/m1_m2_metric_guided_search_accuracy_gate_v1.yaml",
    "analyse/m1_m2_metric_guided_search_accuracy_gate.py",
    "scripts/run_m1_m2_metric_guided_search_accuracy_gate.sh",
    "tests/test_m1_m2_metric_guided_search_accuracy_gate.py",
)

HIGH_ROOT = REPO_ROOT / "results/search/high-regime-prior-model-blind-validation-global4-v1"
SOFT_ROOT = REPO_ROOT / "results/posthoc/joint-configuration-soft-metric-gate-global4-v2"
PILOT_ROOT = REPO_ROOT / "results/posthoc/joint-metric-prospective-selection-gate-global4-v1-rerun1"
CONFIRMATION_ROOT = REPO_ROOT / "results/posthoc/joint-metric-independent-confirmation-surrogate-selection-global4-v1-rerun1"
B4_SEARCH_ROOT = REPO_ROOT / "results/posthoc/b4-guided-search-accuracy-gate-global4-v1"
SOFT_SNAPSHOT = SOFT_ROOT / "source_snapshot"
HIGH_SNAPSHOT = HIGH_ROOT / "source_snapshot"

REPLICATES = 8
ROUNDS = 4
CANDIDATES_PER_ROUND = 128
SELECTED_PER_ROUND = 4
TOTAL_CANDIDATES = REPLICATES * ROUNDS * CANDIDATES_PER_ROUND
SLOTS_PER_METHOD = REPLICATES * ROUNDS * SELECTED_PER_ROUND
TRAINING_SEEDS = (0, 1, 2)
METRIC_MODEL_SEEDS = (0, 1, 2)
CANDIDATE_MASTER_SEED = 202608300001
RANDOM_MASTER_SEED = 202608300002
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0
REPLAY_ATOL = 1.0e-8
DECODE_BATCH_SIZE = 256
MAX_DRAWS_PER_REPLICATE = 262_144
HISTORICAL_P_MEAN = 0.8087333333333333

ALLOWED_CLASSIFICATIONS = frozenset({
    "METRIC_SEARCH_NOT_CONFIRMED",
    "M1_SEARCH_PROMOTED",
    "M2_SEARCH_PROMOTED",
    "M1_M2_SEARCH_PROMOTED_KNN_SELECTED",
    "M1_M2_SEARCH_PROMOTED_GP_SELECTED",
    "SEARCH_NO_SELECTION_DIFFERENCE",
    "SEARCH_EVALUATION_INCOMPLETE",
    "METRIC_SCORER_REPLAY_MISMATCH",
    "CANDIDATE_GENERATION_FAILED",
    "INPUT_IDENTITY_INVALID",
    "RUN_FAILED",
})

REPORT_SECTIONS = (
    "1. Executive conclusion",
    "2. Why this directly tests final search accuracy",
    "3. Historical evidence boundary",
    "4. Frozen metric scorer identity",
    "5. Candidate generation",
    "6. Candidate leakage audit",
    "7. Pre-search freeze",
    "8. Physical vs method-equivalent budget",
    "9. M1 architecture-selection diversity",
    "10. M2 architecture-selection diversity",
    "11. Random selection diversity",
    "12. GNN evaluation completeness",
    "13. M1 final-best results",
    "14. M2 final-best results",
    "15. Random final-best results",
    "16. M1 vs Random",
    "17. M2 vs Random",
    "18. M1 vs M2",
    "19. Anytime search curves",
    "20. Architecture-collapse diagnosis",
    "21. Historical P descriptive comparison",
    "22. Frozen classification",
    "23. Selected next surrogate",
    "24. Whether search efficiency improved",
    "25. Whether final stable accuracy improvement has been proven",
    "26. Next-stage frozen winners",
    "27. Engineering audit",
)

PRE_SEARCH_FILES = (
    "candidate_manifest.csv",
    "m1_predictions.csv", "m2_predictions.csv",
    "m1_selections.json", "m2_selections.json",
    "random_selections.json",
    "selection_collapse.json",
    "search_plan.json",
    "scorer_identity.json",
    "generator_identity.json",
)

REQUIRED_OUTPUTS = (
    "state.json",
    "frozen_protocol.json",
    "input_integrity.json",
    "metric_scorer_replay.json",
    "pre_search/candidate_manifest.csv",
    "pre_search/m1_predictions.csv", "pre_search/m2_predictions.csv",
    "pre_search/m1_selections.json", "pre_search/m2_selections.json",
    "pre_search/random_selections.json",
    "pre_search/selection_collapse.json",
    "pre_search/search_plan.json",
    "pre_search/scorer_identity.json",
    "pre_search/generator_identity.json",
    "pre_search/freeze_manifest.json",
    "pre_search/pre_search_root_sha256.txt",
    "evaluation/task_manifest.json",
    "evaluation/raw_seed_results.csv",
    "evaluation/config_3seed_statistics.csv",
    "evaluation/training_audit.json",
    "search/per_round_results.csv",
    "search/per_replicate_results.csv",
    "search/anytime_curves.csv",
    "search/m1_vs_random.json", "search/m2_vs_random.json", "search/m1_vs_m2.json",
    "search/bootstrap_results.json",
    "search/architecture_collapse_results.json",
    "next_stage/metric_replicate_winners.csv",
    "next_stage/random_replicate_winners.csv",
    "decision.json",
    "final_report.md",
    "environment.json",
    "git_status_before.txt",
    "git_status_after.txt",
    "git_diff.patch",
    "source_snapshot/manifest.json",
    "source_snapshot/imported_local_modules.json",
    "source_snapshot/sha256_manifest.json",
    "manifest.json",
    "logs/py_compile.log",
    "logs/pytest_focused.log",
    "logs/synthetic_smoke.log",
    "logs/scorer_replay.log",
    "logs/candidate_generation.log",
    "logs/evaluation.log",
    "logs/analysis.log",
)

CSV_FIELDS: dict[str, tuple[str, ...]] = {
    "pre_search/candidate_manifest.csv": (
        "replicate_id", "round_id", "candidate_slot", "joint_fingerprint",
        "architecture_fingerprint", "hp_fingerprint", "raw_hp_vector",
        "raw_hp", "operations", "edges", "effective_layers", "lr",
        "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
        "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer",
        "generation_seed", "generation_draw_index", "proposal_distribution",
        "m1_prediction", "m2_prediction", "m1_selected", "m2_selected", "random_selected",
    ),
    "pre_search/m1_predictions.csv": (
        "replicate_id", "round_id", "candidate_slot", "joint_fingerprint",
        "architecture_fingerprint", "m1_prediction",
    ),
    "pre_search/m2_predictions.csv": (
        "replicate_id", "round_id", "candidate_slot", "joint_fingerprint",
        "architecture_fingerprint", "m2_prediction",
    ),
    "evaluation/raw_seed_results.csv": (
        "joint_fingerprint", "training_seed", "validation_accuracy",
        "terminal_status", "attempt_count", "assigned_gpu",
    ),
    "evaluation/config_3seed_statistics.csv": (
        "joint_fingerprint", "architecture_fingerprint", "validation_seed_0",
        "validation_seed_1", "validation_seed_2", "validation_mean",
        "validation_sd", "m1_prediction", "m2_prediction", "selected_by_M1",
        "selected_by_M2", "selected_by_random",
    ),
    "search/per_round_results.csv": (
        "replicate_id", "round_id", "budget", "method",
        "selected_joint_fingerprints", "selected_validation_means",
        "round_best", "best_so_far",
    ),
    "search/per_replicate_results.csv": (
        "replicate_id", "best_M1", "best_M2", "best_random", "M1_minus_random",
        "M2_minus_random", "M2_minus_M1", "OBSERVED_EVALUATED_UNION_ORACLE",
        "REGRET_M1", "REGRET_M2", "M1_winner_joint_fingerprint",
        "M2_winner_joint_fingerprint", "random_winner_joint_fingerprint",
        "M1_anytime_AUC", "M2_anytime_AUC", "random_anytime_AUC",
        "M1_above_historical_P", "M2_above_historical_P", "random_above_historical_P",
    ),
    "search/anytime_curves.csv": (
        "replicate_id", "method", "budget", "best_so_far",
    ),
    "next_stage/metric_replicate_winners.csv": (
        "replicate_id", "joint_fingerprint", "architecture_fingerprint",
        "validation_mean", "validation_sd", "m1_prediction", "m2_prediction",
    ),
    "next_stage/random_replicate_winners.csv": (
        "replicate_id", "joint_fingerprint", "architecture_fingerprint",
        "validation_mean", "validation_sd", "m1_prediction", "m2_prediction",
    ),
}


class ProtocolError(RuntimeError):
    """A frozen protocol contract was violated."""


class InputIdentityInvalid(ProtocolError):
    """An authoritative input differs from its frozen identity."""


class MetricScorerReplayMismatch(ProtocolError):
    """Frozen M1/M2 could not reproduce the archived metric identity."""


class CandidateGenerationFailed(ProtocolError):
    """The fixed rejection budget did not yield the candidate manifest."""


class PreSearchFreezeError(ProtocolError):
    """The immutable pre-search root is missing or changed."""


class EvaluationIncomplete(ProtocolError):
    """At least one selected configuration lacks a complete matched triplet."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def dump_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return value


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_test_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return "test" in normalized


def reject_test_named_fields(value: Any, *, allow_false_controls: bool = False, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if _is_test_key(name):
                permitted = allow_false_controls and name in {"track_test", "test_accessed", "test_data_accessed"} and child is False
                if not permitted:
                    raise ProtocolError(f"test-named field crossed the validation-only firewall: {path}")
            reject_test_named_fields(child, allow_false_controls=allow_false_controls, prefix=path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_test_named_fields(child, allow_false_controls=allow_false_controls, prefix=f"{prefix}[{index}]")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ProtocolError("PyYAML is required") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("protocol config must be a mapping")
    return payload


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ProtocolError(f"frozen protocol mismatch for {label}: {observed!r} != {expected!r}")


def validate_frozen_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    _require_equal(config.get("protocol_version"), PROTOCOL_VERSION, "protocol_version")
    generation = config["candidate_generation"]
    search = config["search"]
    evaluation = config["evaluation"]
    analysis = config["analysis"]
    _require_equal(generation["candidate_master_seed"], CANDIDATE_MASTER_SEED, "candidate seed")
    _require_equal(generation["replicates"], REPLICATES, "replicates")
    _require_equal(generation["rounds_per_replicate"], ROUNDS, "rounds")
    _require_equal(generation["candidates_per_round"], CANDIDATES_PER_ROUND, "candidates per round")
    _require_equal(generation["total_candidates"], TOTAL_CANDIDATES, "candidate count")
    _require_equal(generation["hp_raw_unit_interval_range"], [0.80, 0.95], "HIGH_REGIME_PRIOR")
    _require_equal(search["random_master_seed"], RANDOM_MASTER_SEED, "random seed")
    _require_equal(search["selected_per_round_per_method"], SELECTED_PER_ROUND, "selected per round")
    _require_equal(search["selected_slots_per_method"], SLOTS_PER_METHOD, "method slots")
    _require_equal(evaluation["training_seeds"], list(TRAINING_SEEDS), "training seeds")
    _require_equal(evaluation["track_test"], False, "track_test")
    _require_equal(evaluation["max_epochs"], 300, "GNN max epochs")
    _require_equal(evaluation["patience"], 80, "GNN patience")
    _require_equal(analysis["bootstrap"]["unit"], "search_replicate", "bootstrap unit")
    _require_equal(analysis["bootstrap"]["replicates"], BOOTSTRAP_REPLICATES, "bootstrap draws")
    _require_equal(analysis["bootstrap"]["seed"], BOOTSTRAP_SEED, "bootstrap seed")
    metric = config["frozen_metric"]
    _require_equal(metric["metric_seeds"], list(METRIC_MODEL_SEEDS), "metric seeds")
    _require_equal(metric["outer_folds"], 5, "metric folds")
    _require_equal(metric["encoder_count"], 15, "metric encoder count")
    _require_equal(metric["supervision"], "historical_336_only", "metric supervision")
    _require_equal(metric["M1"]["k"], 5, "M1 k")
    _require_equal(metric["M2"]["ard"], False, "M2 isotropic")
    _require_equal(metric["M2"]["per_dimension_input_standardization"], False, "M2 input transform")
    m12 = analysis["M1_vs_M2"]
    _require_equal(m12["regret_oracle_name"], "OBSERVED_EVALUATED_UNION_ORACLE_r", "regret oracle")
    _require_equal(m12["regret_comparison_tolerance"], 1.0e-12, "regret tolerance")
    _require_equal(m12["both_mean_regrets_at_most_1e_12_forces_S2_false"], True, "zero regret rule")
    _require_equal(config["next_stage"]["fresh_seed_training_in_this_run"], "forbidden", "fresh seeds")
    _require_equal(set(config["allowed_classifications"]), set(ALLOWED_CLASSIFICATIONS), "classifications")
    return {"status": "VALID", "protocol_version": PROTOCOL_VERSION, "test_accessed": False}


def _git_capture(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout


def _git_diff_patch() -> str:
    """Capture tracked changes plus this experiment's untracked additions."""
    pieces = [_git_capture("diff", "HEAD", "--binary", "--", ".").rstrip()]
    for relative in IMPLEMENTATION_FILES:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=REPO_ROOT, check=False, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        path = REPO_ROOT / relative
        if tracked.returncode == 0 or not path.is_file():
            continue
        addition = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative],
            cwd=REPO_ROOT, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if addition.returncode not in {0, 1}:
            raise ProtocolError(f"could not snapshot untracked Git diff for {relative}: {addition.stdout.strip()}")
        if addition.stdout.strip():
            pieces.append(addition.stdout.rstrip())
    return "\n".join(piece for piece in pieces if piece) + "\n"


def initial_state(config_sha256: str, source_sha256: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "IMPLEMENTATION_READY",
        "classification": None,
        "candidate_count": 0,
        "m1_selected_slots": 0,
        "m2_selected_slots": 0,
        "random_selected_slots": 0,
        "selected_unique_config_count": 0,
        "planned_gnn_runs": 0,
        "completed_gnn_runs": 0,
        "failed_gnn_runs": 0,
        "current_replicate": None,
        "current_round": None,
        "current_config": None,
        "current_seed": None,
        "test_accessed": False,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


def update_state(output: Path, **changes: Any) -> dict[str, Any]:
    path = output / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    if changes.get("test_accessed") is not None and changes.get("test_accessed") is not False:
        raise ProtocolError("test_accessed can never become true")
    state.update(changes)
    state["updated_at"] = utc_now()
    dump_json(path, state)
    return state


def _source_identity_paths() -> dict[str, Path]:
    return {
        "configs/m1_m2_metric_guided_search_accuracy_gate_v1.yaml": DEFAULT_CONFIG,
        "analyse/m1_m2_metric_guided_search_accuracy_gate.py": Path(__file__).resolve(),
        "scripts/run_m1_m2_metric_guided_search_accuracy_gate.sh": REPO_ROOT / "scripts/run_m1_m2_metric_guided_search_accuracy_gate.sh",
        "tests/test_m1_m2_metric_guided_search_accuracy_gate.py": REPO_ROOT / "tests/test_m1_m2_metric_guided_search_accuracy_gate.py",
        "analyse/joint_metric_independent_confirmation_surrogate_selection.py": REPO_ROOT / "analyse/joint_metric_independent_confirmation_surrogate_selection.py",
        "analyse/joint_metric_prospective_selection_gate.py": REPO_ROOT / "analyse/joint_metric_prospective_selection_gate.py",
        "frozen/configs/joint_configuration_soft_metric_gate_v2.yaml": SOFT_SNAPSHOT / "configs/joint_configuration_soft_metric_gate_v2.yaml",
        "frozen/analyse/joint_configuration_metric_gate.py": SOFT_SNAPSHOT / "analyse/joint_configuration_metric_gate.py",
        "frozen/analyse/joint_configuration_soft_metric_gate.py": SOFT_SNAPSHOT / "analyse/joint_configuration_soft_metric_gate.py",
        "frozen/analyse/performance_metric_quick_gate.py": SOFT_SNAPSHOT / "analyse/performance_metric_quick_gate.py",
        "frozen/analyse/high_regime_prior_model_blind_validation.py": HIGH_SNAPSHOT / "analyse/high_regime_prior_model_blind_validation.py",
        "frozen/analyse/run_high_regime_prior_model_blind_validation_labels.py": HIGH_SNAPSHOT / "analyse/run_high_regime_prior_model_blind_validation_labels.py",
        "frozen/analyse/analyse_high_regime_prior_model_blind_validation.py": HIGH_SNAPSHOT / "analyse/analyse_high_regime_prior_model_blind_validation.py",
        "frozen/analyse/original_head_deterministic_batch_pilot.py": HIGH_SNAPSHOT / "analyse/original_head_deterministic_batch_pilot.py",
        "frozen/analyse/structure_native_surrogate_pilot.py": HIGH_SNAPSHOT / "analyse/structure_native_surrogate_pilot.py",
        "frozen/analyse/vae_label_expansion.py": HIGH_SNAPSHOT / "analyse/vae_label_expansion.py",
        "frozen/eval_utils.py": HIGH_SNAPSHOT / "eval_utils.py",
        "frozen/architecture_diagnostics.py": HIGH_SNAPSHOT / "architecture_diagnostics.py",
        "dataset_utils.py": REPO_ROOT / "dataset_utils.py",
        "hp_modes.py": REPO_ROOT / "hp_modes.py",
        "models.py": REPO_ROOT / "models.py",
    }


def source_identity_entries() -> list[dict[str, Any]]:
    entries = []
    for relative, path in sorted(_source_identity_paths().items()):
        if not path.is_file():
            raise ProtocolError(f"required runtime source is missing: {path}")
        entries.append({"relative_path": relative, "path": str(path), "sha256": sha256_file(path)})
    return entries


def aggregate_source_sha256() -> str:
    return stable_hash([{key: row[key] for key in ("relative_path", "sha256")} for row in source_identity_entries()])


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise InputIdentityInvalid(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise InputIdentityInvalid(f"{label} SHA-256 mismatch: {observed} != {expected}")


def _unwrap_candidates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        payload = payload.get("candidates", payload.get("rows", payload))
    if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
        raise InputIdentityInvalid("candidate manifest is not a list of records")
    return [dict(row) for row in payload]


def static_input_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    authority = config["authoritative_inputs"]
    high = authority["historical_high_regime_v1"]
    pilot = authority["previous_8_pool_prospective_v1"]
    confirmation = authority["previous_16_pool_confirmation_v1"]
    soft = authority["frozen_soft_metric_v2"]
    b4 = authority["b4_search_v1"]
    checks = [
        (REPO_ROOT / high["archive"], high["archive_sha256"], "high-regime archive"),
        (HIGH_ROOT / "preregistration.json", high["preregistration_sha256"], "high-regime preregistration"),
        (HIGH_ROOT / "source_dependency_manifest.json", high["source_dependency_manifest_sha256"], "high-regime dependency manifest"),
        (REPO_ROOT / pilot["archive"], pilot["archive_sha256"], "8-pool archive"),
        (PILOT_ROOT / pilot["candidate_manifest"], pilot["candidate_manifest_sha256"], "8-pool candidate manifest"),
        (REPO_ROOT / confirmation["archive"], confirmation["archive_sha256"], "16-pool archive"),
        (CONFIRMATION_ROOT / confirmation["candidate_manifest"], confirmation["candidate_manifest_sha256"], "16-pool candidate manifest"),
        (REPO_ROOT / soft["archive"], soft["archive_sha256"], "soft-v2 archive"),
        (SOFT_SNAPSHOT / "configs/joint_configuration_soft_metric_gate_v2.yaml", soft["config_sha256"], "metric config"),
        *[(SOFT_ROOT / name, digest, f"metric anchor {name}") for name, digest in soft["artifacts"].items()],
        (REPO_ROOT / b4["archive"], b4["archive_sha256"], "B4-search archive"),
        (B4_SEARCH_ROOT / b4["evaluated_manifest"], b4["evaluated_manifest_sha256"], "B4-search evaluated manifest"),
        (REPO_ROOT / config["frozen_runtime"]["vae_checkpoint"], config["frozen_runtime"]["vae_checkpoint_sha256"], "VAE checkpoint"),
        (REPO_ROOT / config["analysis"]["historical_P"]["source_file"], config["analysis"]["historical_P"]["source_sha256"], "historical P reference"),
    ]
    for relative, expected in high["historical_seed_shards"].items():
        checks.append((HIGH_ROOT / relative, expected, f"historical shard {relative}"))
    identities = []
    for path, expected, label in checks:
        _require_hash(path, str(expected), label)
        identities.append({"label": label, "path": str(path), "sha256": str(expected)})
    for root, spec, label in (
        (SOFT_ROOT, soft, "soft-v2"), (PILOT_ROOT, pilot, "8-pool"),
        (CONFIRMATION_ROOT, confirmation, "16-pool"), (B4_SEARCH_ROOT, b4, "B4-search"),
    ):
        if not (root / str(spec["required_marker"])).is_file():
            raise InputIdentityInvalid(f"{label} required marker is missing")
    return {"status": "VALID", "checked_files": identities, "test_accessed": False}


def validate_authoritative_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    audit = static_input_audit(config)
    authority = config["authoritative_inputs"]
    soft_spec = authority["frozen_soft_metric_v2"]
    soft_state = json.loads((SOFT_ROOT / "state.json").read_text(encoding="utf-8"))
    soft_decision = json.loads((SOFT_ROOT / "decision.json").read_text(encoding="utf-8"))
    soft_snapshot = json.loads((SOFT_ROOT / "source_snapshot/sha256_manifest.json").read_text(encoding="utf-8"))
    soft_observed = {
        "classification": soft_decision.get("classification"),
        "config_sha256": soft_state.get("config_sha256"),
        "source_sha256": soft_state.get("source_sha256"),
        "source_snapshot_manifest_root": soft_snapshot.get("manifest_root_sha256"),
    }
    soft_expected = {
        "classification": soft_spec["classification"],
        "config_sha256": soft_spec["config_sha256"],
        "source_sha256": soft_spec["source_sha256"],
        "source_snapshot_manifest_root": soft_spec["source_snapshot_manifest_root"],
    }
    if soft_observed != soft_expected:
        raise InputIdentityInvalid(f"soft metric identity mismatch: {soft_observed}")
    pilot_rows = _unwrap_candidates(json.loads((PILOT_ROOT / "pre_eval/candidate_manifest.json").read_text(encoding="utf-8")))
    confirmation_rows = _unwrap_candidates(json.loads((CONFIRMATION_ROOT / "pre_eval/candidate_manifest.json").read_text(encoding="utf-8")))
    if len(pilot_rows) != 32 or len(confirmation_rows) != 64:
        raise InputIdentityInvalid("previous candidate count mismatch")
    for rows, label in ((pilot_rows, "8-pool"), (confirmation_rows, "16-pool")):
        reject_test_named_fields(rows, allow_false_controls=True)
        joints = [str(row.get("joint_fingerprint", row.get("joint_config_fingerprint"))) for row in rows]
        if len(set(joints)) != len(rows):
            raise InputIdentityInvalid(f"{label} candidate manifest has duplicate joints")
        if any("validation_mean" in row or "val_acc" in row for row in rows):
            raise InputIdentityInvalid(f"{label} candidate manifest contains labels")
    b4_spec = authority["b4_search_v1"]
    b4_state = json.loads((B4_SEARCH_ROOT / "state.json").read_text(encoding="utf-8"))
    b4_decision = json.loads((B4_SEARCH_ROOT / "decision.json").read_text(encoding="utf-8"))
    if {
        "classification": b4_decision.get("classification"),
        "config_sha256": b4_state.get("config_sha256"),
        "source_sha256": b4_state.get("source_sha256"),
        "test_accessed": b4_state.get("test_accessed"),
    } != {
        "classification": b4_spec["classification"],
        "config_sha256": b4_spec["config_sha256"],
        "source_sha256": b4_spec["source_sha256"],
        "test_accessed": False,
    }:
        raise InputIdentityInvalid("B4-search identity mismatch")
    audit.update({
        "soft_metric_identity": soft_observed,
        "previous_8_pool_candidate_count": len(pilot_rows),
        "previous_16_pool_candidate_count": len(confirmation_rows),
        "B4_search_evaluated_unique_configurations": int(b4_spec["evaluated_unique_configurations"]),
        "previous_or_B4_labels_loaded_for_metric_training": False,
        "test_accessed": False,
    })
    return audit


def load_frozen_runtime() -> dict[str, Any]:
    try:
        from analyse import joint_metric_independent_confirmation_surrogate_selection as confirmation
        pilot = confirmation.pilot_gate
        runtime = pilot.load_frozen_soft_runtime()
        high = pilot._load_frozen_high_module()
    except Exception as error:
        if type(error).__name__ in {
            "ProspectiveInputInvalid", "ConfirmationInputInvalid", "InputIdentityMismatch",
        }:
            raise InputIdentityInvalid(str(error)) from error
        raise
    runtime["high"] = high
    runtime["confirmation"] = confirmation
    runtime["pilot"] = pilot
    observed = {
        Path(runtime["soft"].train_soft_metric_encoder.__code__.co_filename).resolve(),
        Path(runtime["soft"].joint_v1.weighted_knn_predict.__code__.co_filename).resolve(),
        Path(runtime["soft"].joint_v1.exact_matern_gp_predict.__code__.co_filename).resolve(),
    }
    expected = {
        (SOFT_SNAPSHOT / "analyse/joint_configuration_soft_metric_gate.py").resolve(),
        (SOFT_SNAPSHOT / "analyse/joint_configuration_metric_gate.py").resolve(),
    }
    if observed != expected:
        raise InputIdentityInvalid(f"M1/M2 resolved outside frozen snapshots: {sorted(map(str, observed))}")
    return runtime


def replay_frozen_metric(
    runtime: Mapping[str, Any], log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], Sequence[Any], dict[str, Any]]:
    try:
        configurations, retained, audit = runtime["pilot"].replay_and_retain_scorers(runtime, {}, log)
    except Exception as error:
        if type(error).__name__ in {"ScorerReplayIdentityMismatch", "ScorerProtocolMismatch"}:
            raise MetricScorerReplayMismatch(str(error)) from error
        if type(error).__name__ in {"ProspectiveInputInvalid", "InputIdentityMismatch"}:
            raise InputIdentityInvalid(str(error)) from error
        raise
    if len(configurations) != 336 or len(retained) != 5:
        raise MetricScorerReplayMismatch("metric replay cardinality mismatch")
    if audit.get("status") != "IDENTICAL" or int(audit.get("model_count", 0)) != 15:
        raise MetricScorerReplayMismatch("metric replay identity is not exact")
    audit.update({
        "scorers": ["M1_WEIGHTED_KNN", "M2_ISOTROPIC_MATERN_5_2_GP_MEAN"],
        "historical_supervision": "historical_336_only",
        "previous_8_pool_labels_used": False,
        "previous_16_pool_labels_used": False,
        "B4_search_labels_used": False,
        "current_labels_used": False,
        "online_update_used": False,
        "round_refit_used": False,
        "test_accessed": False,
    })
    return configurations, retained, audit


def previous_candidate_identities() -> dict[str, set[str]]:
    pilot = _unwrap_candidates(json.loads((PILOT_ROOT / "pre_eval/candidate_manifest.json").read_text(encoding="utf-8")))
    confirmation = _unwrap_candidates(json.loads((CONFIRMATION_ROOT / "pre_eval/candidate_manifest.json").read_text(encoding="utf-8")))
    b4_tasks = json.loads((B4_SEARCH_ROOT / "evaluation/task_manifest.json").read_text(encoding="utf-8"))["tasks"]
    reject_test_named_fields(b4_tasks)
    b4_by_joint = {
        str(row["joint_config_fingerprint"]): str(row["architecture_fingerprint"])
        for row in b4_tasks
    }
    if len(b4_by_joint) != 254:
        raise InputIdentityInvalid("B4-search exclusion set is not 254 unique joints")
    earlier = pilot + confirmation
    return {
        "joint": {str(row.get("joint_fingerprint", row.get("joint_config_fingerprint"))) for row in earlier} | set(b4_by_joint),
        "architecture": {str(row["architecture_fingerprint"]) for row in earlier} | set(b4_by_joint.values()),
        "prospective_architecture": {str(row["architecture_fingerprint"]) for row in earlier},
        "b4_search_architecture": set(b4_by_joint.values()),
        "pilot_joint": {str(row.get("joint_fingerprint", row.get("joint_config_fingerprint"))) for row in pilot},
        "confirmation_joint": {str(row.get("joint_fingerprint", row.get("joint_config_fingerprint"))) for row in confirmation},
        "b4_search_joint": set(b4_by_joint),
    }


def _candidate_row(
    high: Any, model: Any, graph: Any, replicate: int, round_id: int,
    candidate_slot: int, raw_hp: Sequence[float], draw_index: int,
) -> dict[str, Any]:
    graph_config = high.v1.config_from_igraph(model, graph)
    valid, reason = high.v1.validate_static_config(graph_config)
    if not valid:
        raise CandidateGenerationFailed(f"accepted architecture is invalid: {reason}")
    architecture = high.v1.architecture_fingerprint(graph_config)
    raw = [float(value) for value in raw_hp]
    hp = high.v1.hp_fingerprint(raw)
    joint = high.joint_config_fingerprint(architecture, hp)
    row = {
        "replicate_id": replicate,
        "round_id": round_id,
        "candidate_slot": candidate_slot,
        "pool_id": replicate,
        "batch_id": round_id,
        "architecture_slot": candidate_slot,
        "config_id": f"metric-search-r{replicate}-q{round_id}-s{candidate_slot}",
        "joint_fingerprint": joint,
        "joint_config_fingerprint": joint,
        "architecture_fingerprint": architecture,
        "hp_fingerprint": hp,
        "operations": list(graph_config["operations"]),
        "edges": [list(map(int, edge)) for edge in graph_config["edges"]],
        "effective_layers": int(graph_config["effective_layers"]),
        "raw_hp_vector": raw,
        "hp_regime": "hp_high",
        "generation_seed": CANDIDATE_MASTER_SEED + replicate,
        "generation_draw_index": draw_index,
        "proposal_distribution": "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR",
    }
    row.update(high.v1.realize_global4_hp(row))
    row["raw_hp"] = {key: row[key] for key in ("lr", "dropout", "hidden_dim", "l2")}
    return row


def validate_candidate_set(
    candidates: Sequence[Mapping[str, Any]], *, historical_joint: set[str],
    previous_joint: set[str], b4_search_joint: set[str], historical_architecture: set[str],
    previous_architecture: set[str], expected_count: int = TOTAL_CANDIDATES,
) -> dict[str, Any]:
    if len(candidates) != expected_count:
        raise CandidateGenerationFailed(f"candidate count {len(candidates)} != {expected_count}")
    joints = [str(row["joint_fingerprint"]) for row in candidates]
    architectures = [str(row["architecture_fingerprint"]) for row in candidates]
    if len(set(joints)) != len(joints):
        raise CandidateGenerationFailed("current joint fingerprints are not all unique")
    if set(joints) & historical_joint:
        raise CandidateGenerationFailed("historical exact joint leaked into current candidates")
    if set(joints) & previous_joint:
        raise CandidateGenerationFailed("previous prospective exact joint leaked into current candidates")
    if set(joints) & b4_search_joint:
        raise CandidateGenerationFailed("B4-search evaluated exact joint leaked into current candidates")
    expected_groups = expected_count // CANDIDATES_PER_ROUND
    counts = Counter((int(row["replicate_id"]), int(row["round_id"])) for row in candidates)
    if len(counts) != expected_groups or any(count != CANDIDATES_PER_ROUND for count in counts.values()):
        raise CandidateGenerationFailed("candidate round cardinalities are not frozen")
    if expected_count == TOTAL_CANDIDATES and set(counts) != {(r, q) for r in range(REPLICATES) for q in range(ROUNDS)}:
        raise CandidateGenerationFailed("candidate replicate/round grid is incomplete")
    proposal_values = {str(row.get("proposal_distribution")) for row in candidates}
    if proposal_values != {"MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR"}:
        raise CandidateGenerationFailed("candidate proposal is not exclusively model-blind high-regime")
    architecture_counts = Counter(architectures)
    return {
        "passed": True,
        "candidate_count": len(candidates),
        "unique_joint_count": len(set(joints)),
        "unique_architecture_count": len(architecture_counts),
        "historical_joint_overlap": len(set(joints) & historical_joint),
        "previous_prospective_joint_overlap": len(set(joints) & previous_joint),
        "B4_search_joint_overlap": len(set(joints) & b4_search_joint),
        "historical_seen_architecture_fraction": sum(architecture in historical_architecture for architecture in architectures) / len(architectures),
        "previous_prospective_seen_architecture_fraction": sum(architecture in previous_architecture for architecture in architectures) / len(architectures),
        "current_repeated_architecture_fraction": (len(architectures) - len(architecture_counts)) / len(architectures),
        "records_with_repeated_architecture_fraction": sum(architecture_counts[value] > 1 for value in architectures) / len(architectures),
        "architecture_frequency_distribution": dict(sorted(architecture_counts.items())),
        "architecture_repetition_allowed": True,
        "predictor_accessed_during_generation": False,
        "validation_accessed_during_generation": False,
        "test_accessed": False,
    }


def generate_all_candidates(
    runtime: Mapping[str, Any], historical_rows: Sequence[Mapping[str, Any]],
    previous: Mapping[str, set[str]], log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import torch
    except ImportError as error:
        raise CandidateGenerationFailed("PyTorch is required for frozen VAE decoding") from error
    high = runtime["high"]
    checkpoint = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
    if sha256_file(checkpoint) != "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4":
        raise InputIdentityInvalid("frozen VAE checkpoint identity changed")
    model = high.v1.load_joint_vae(checkpoint, "global4", device="cpu")
    model.eval()
    model.arch_vae.eval()
    historical_joint = {str(row["joint_fingerprint"]) for row in historical_rows}
    historical_architecture = {str(row["architecture_fingerprint"]) for row in historical_rows}
    accepted_joint: set[str] = set()
    candidates: list[dict[str, Any]] = []
    replicate_audits = []
    for replicate in range(REPLICATES):
        seed = CANDIDATE_MASTER_SEED + replicate
        architecture_rng = torch.Generator(device="cpu")
        architecture_rng.manual_seed(seed)
        hp_rng = np.random.default_rng(seed)
        decoded_draws = processed_draws = invalid_draws = excluded_joint_draws = 0
        round_audits = []
        for round_id in range(ROUNDS):
            round_rows: list[dict[str, Any]] = []
            round_processed_before = processed_draws
            while len(round_rows) < CANDIDATES_PER_ROUND and decoded_draws < MAX_DRAWS_PER_REPLICATE:
                size = min(DECODE_BATCH_SIZE, MAX_DRAWS_PER_REPLICATE - decoded_draws)
                batch_start = decoded_draws
                with torch.no_grad():
                    latent = torch.randn((size, 12), generator=architecture_rng, device="cpu")
                    graphs = model.arch_vae.decode(latent, stochastic=False)
                decoded_draws += size
                for offset, graph in enumerate(graphs):
                    processed_draws += 1
                    try:
                        graph_config = high.v1.config_from_igraph(model, graph)
                        valid, _reason = high.v1.validate_static_config(graph_config)
                        if not valid:
                            raise ValueError("invalid architecture")
                    except (TypeError, ValueError, KeyError, IndexError):
                        invalid_draws += 1
                        continue
                    # This is the frozen HIGH_REGIME_PRIOR draw.  No score,
                    # label, metric, head, incumbent, or earlier outcome is in
                    # this function's inputs or rejection rule.
                    raw_hp = hp_rng.uniform(0.80, 0.95, size=4).astype(np.float64).tolist()
                    row = _candidate_row(
                        high, model, graph, replicate, round_id, len(round_rows),
                        raw_hp, batch_start + offset,
                    )
                    joint = str(row["joint_fingerprint"])
                    if joint in historical_joint or joint in previous["joint"] or joint in accepted_joint:
                        excluded_joint_draws += 1
                        continue
                    accepted_joint.add(joint)
                    candidates.append(row)
                    round_rows.append(row)
                    if len(round_rows) == CANDIDATES_PER_ROUND:
                        break
            if len(round_rows) != CANDIDATES_PER_ROUND:
                raise CandidateGenerationFailed(
                    f"replicate {replicate}, round {round_id} exhausted the frozen draw budget"
                )
            round_audit = {
                "replicate_id": replicate,
                "round_id": round_id,
                "accepted": len(round_rows),
                "processed_draws_this_round": processed_draws - round_processed_before,
                "decoded_draws_cumulative": decoded_draws,
            }
            round_audits.append(round_audit)
            log(
                f"replicate {replicate} round {round_id}: accepted 128; "
                f"decoded cumulative={decoded_draws}"
            )
        replicate_audits.append({
            "replicate_id": replicate,
            "seed": seed,
            "decoded_draws": decoded_draws,
            "processed_draws": processed_draws,
            "invalid_draws": invalid_draws,
            "excluded_exact_joint_draws": excluded_joint_draws,
            "rounds": round_audits,
        })
    coverage = validate_candidate_set(
        candidates,
        historical_joint=historical_joint,
        previous_joint=previous["pilot_joint"] | previous["confirmation_joint"],
        b4_search_joint=previous["b4_search_joint"],
        historical_architecture=historical_architecture,
        previous_architecture=previous["prospective_architecture"],
    )
    audit = {
        "status": "GENERATED",
        "proposal_distribution": "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR",
        "candidate_master_seed": CANDIDATE_MASTER_SEED,
        "replicate_seed_rule": "master_plus_zero_based_replicate",
        "rounds_advance_continuous_replicate_RNG_streams": True,
        "hp_raw_unit_interval_range": [0.80, 0.95],
        "predictor_accessed": False,
        "validation_accessed": False,
        "historical_best_accessed": False,
        "metric_accessed": False,
        "head_accessed": False,
        "mutation_or_local_search_used": False,
        "acquisition_used": False,
        "replicates": replicate_audits,
        "coverage": coverage,
        "test_accessed": False,
    }
    return candidates, audit


def score_all_candidates_with_metric(
    runtime: Mapping[str, Any], retained: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]], log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(candidates) != TOTAL_CANDIDATES:
        raise ProtocolError("metric scoring requires exactly 4,096 frozen candidate records")
    soft = runtime["soft"]
    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (int(row["replicate_id"]), int(row["round_id"]), int(row["candidate_slot"])),
    )
    query_fp = [str(row["joint_fingerprint"]) for row in ordered]
    m1_by_fold: dict[int, np.ndarray] = {}
    m2_by_fold: dict[int, np.ndarray] = {}
    gp_audits = []
    for scorer in retained:
        train_y = np.asarray([row["validation_mean"] for row in scorer.train_rows], dtype=np.float64)
        train_se = np.asarray([row["validation_se"] for row in scorer.train_rows], dtype=np.float64)
        train_fp = [str(row["joint_fingerprint"]) for row in scorer.train_rows]
        seed_m1: dict[int, np.ndarray] = {}
        seed_m2: dict[int, np.ndarray] = {}
        for seed in METRIC_MODEL_SEEDS:
            log(f"fold {scorer.fold}/4 seed {seed}: scoring all 4096 candidates with frozen M1/M2")
            candidate_embedding = soft.joint_v1.encode_rows_with_model(
                scorer.seed_models[seed], ordered, scorer.hp_standardizer, "full_joint",
            )
            train_embedding = scorer.seed_train_embeddings[seed]
            seed_m1[seed] = np.asarray(soft.joint_v1.weighted_knn_predict(
                train_embedding, train_y, candidate_embedding, 5,
                train_fingerprints=train_fp, query_fingerprints=query_fp,
            ), dtype=np.float64)
            m2, audit = soft.joint_v1.exact_matern_gp_predict(
                train_embedding, train_y, train_se, candidate_embedding,
                standardize_inputs=False, ard=False, protocol_name="M2",
            )
            if audit.get("ard") is not False or audit.get("input_standardized") is not False:
                raise MetricScorerReplayMismatch("M2 candidate scorer violated frozen isotropic protocol")
            seed_m2[seed] = np.asarray(m2, dtype=np.float64)
            gp_audits.append({"fold": scorer.fold, "seed": seed, **audit})
        m1_by_fold[scorer.fold] = np.asarray(
            soft.joint_v1.average_metric_seed_predictions(seed_m1), dtype=np.float64,
        )
        m2_by_fold[scorer.fold] = np.asarray(
            soft.joint_v1.average_metric_seed_predictions(seed_m2), dtype=np.float64,
        )
    if set(m1_by_fold) != set(range(5)) or set(m2_by_fold) != set(range(5)):
        raise ProtocolError("metric scoring did not cover all five folds")
    pooled_m1 = np.mean(np.stack([m1_by_fold[fold] for fold in range(5)]), axis=0)
    pooled_m2 = np.mean(np.stack([m2_by_fold[fold] for fold in range(5)]), axis=0)
    if any(values.shape != (TOTAL_CANDIDATES,) or not bool(np.isfinite(values).all()) for values in (pooled_m1, pooled_m2)):
        raise ProtocolError("M1/M2 produced invalid candidate predictions")
    rows = [
        {
            "replicate_id": int(candidate["replicate_id"]),
            "round_id": int(candidate["round_id"]),
            "candidate_slot": int(candidate["candidate_slot"]),
            "joint_fingerprint": str(candidate["joint_fingerprint"]),
            "architecture_fingerprint": str(candidate["architecture_fingerprint"]),
            "m1_prediction": float(m1),
            "m2_prediction": float(m2),
        }
        for candidate, m1, m2 in zip(ordered, pooled_m1, pooled_m2)
    ]
    return rows, {
        "status": "SCORED",
        "candidate_count": len(rows),
        "historical_supervision": "historical_336_only",
        "metric_seeds": list(METRIC_MODEL_SEEDS),
        "outer_folds": 5,
        "M1_seed_then_fold_arithmetic_mean": True,
        "M2_seed_then_fold_arithmetic_mean": True,
        "M1_k": 5,
        "M2_isotropic": True,
        "M2_input_standardized": False,
        "M2_posterior_variance_used": False,
        "previous_or_B4_labels_used": False,
        "current_labels_used": False,
        "online_update_used": False,
        "round_refit_used": False,
        "all_predictions_before_first_GNN": True,
        "M2_GP_runs": gp_audits,
        "test_accessed": False,
    }


def _group_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(int(row["replicate_id"]), int(row["round_id"]))].append(dict(row))
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row["candidate_slot"]))
        if len(rows) != CANDIDATES_PER_ROUND or [int(row["candidate_slot"]) for row in rows] != list(range(CANDIDATES_PER_ROUND)):
            raise ProtocolError(f"invalid frozen candidate pool {key}")
    if set(groups) != {(replicate, round_id) for replicate in range(REPLICATES) for round_id in range(ROUNDS)}:
        raise ProtocolError("candidate pool grid is incomplete")
    return groups


def select_search_plans(
    candidates: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
    *, random_seed: int = RANDOM_MASTER_SEED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, set[str]], dict[str, Any], dict[str, Any]]:
    groups = _group_candidates(candidates)
    scores = {
        str(row["joint_fingerprint"]): {
            "M1": float(row["m1_prediction"]), "M2": float(row["m2_prediction"]),
        }
        for row in predictions
    }
    if len(scores) != len(candidates) or set(scores) != {str(row["joint_fingerprint"]) for row in candidates}:
        raise ProtocolError("M1/M2 prediction coverage differs from candidate manifest")
    random_rng = np.random.default_rng(random_seed)
    rounds_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in ("M1", "M2", "RANDOM")}
    selected = {method: set() for method in ("M1", "M2", "RANDOM")}
    per_replicate: dict[int, dict[str, set[str]]] = {
        replicate: {method: set() for method in ("M1", "M2", "RANDOM")}
        for replicate in range(REPLICATES)
    }
    for replicate in range(REPLICATES):
        for round_id in range(ROUNDS):
            pool = groups[(replicate, round_id)]
            chosen: dict[str, list[dict[str, Any]]] = {}
            for method in ("M1", "M2"):
                chosen[method] = sorted(
                    pool,
                    key=lambda row: (-scores[str(row["joint_fingerprint"])][method], str(row["joint_fingerprint"])),
                )[:SELECTED_PER_ROUND]
            indices = random_rng.choice(CANDIDATES_PER_ROUND, size=SELECTED_PER_ROUND, replace=False)
            chosen["RANDOM"] = [pool[int(index)] for index in sorted(indices.tolist())]
            for method in ("M1", "M2", "RANDOM"):
                joints = [str(row["joint_fingerprint"]) for row in chosen[method]]
                selected[method].update(joints)
                per_replicate[replicate][method].update(joints)
                record = {
                    "replicate_id": replicate, "round_id": round_id,
                    "selected_joint_fingerprints": joints,
                    "selected_candidate_slots": [int(row["candidate_slot"]) for row in chosen[method]],
                }
                if method == "RANDOM":
                    record["sampling"] = "uniform_without_replacement"
                else:
                    record.update({
                        "selected_predictions": [scores[joint][method] for joint in joints],
                        "tie_break": "joint_fingerprint_lexicographic_ascending",
                    })
                rounds_by_method[method].append(record)
    information_rows = []
    m1_all_identical = True
    m2_all_identical = True
    for replicate in range(REPLICATES):
        m1_set, m2_set, random_set = (per_replicate[replicate][key] for key in ("M1", "M2", "RANDOM"))
        if any(len(values) != 16 for values in (m1_set, m2_set, random_set)):
            raise ProtocolError("a method does not have exactly 16 unique configurations per replicate")
        m1_same = m1_set == random_set
        m2_same = m2_set == random_set
        m1_all_identical = m1_all_identical and m1_same
        m2_all_identical = m2_all_identical and m2_same
        information_rows.append({
            "replicate_id": replicate,
            "M1_unique_configurations": len(m1_set), "M2_unique_configurations": len(m2_set),
            "random_unique_configurations": len(random_set),
            "M1_random_overlap": len(m1_set & random_set),
            "M2_random_overlap": len(m2_set & random_set),
            "M1_M2_overlap": len(m1_set & m2_set),
            "three_way_overlap": len(m1_set & m2_set & random_set),
            "M1_selection_set_equals_random": m1_same,
            "M2_selection_set_equals_random": m2_same,
        })
    plans = {}
    for method in ("M1", "M2", "RANDOM"):
        plans[method] = {
            "method": method,
            "selection": "uniform_without_replacement" if method == "RANDOM" else "top_4_predicted_validation_per_round",
            "tie_break": None if method == "RANDOM" else "joint_fingerprint_lexicographic_ascending",
            "master_seed": random_seed if method == "RANDOM" else None,
            "rounds": rounds_by_method[method],
            "selected_slot_count": SLOTS_PER_METHOD,
            "diversity_postprocessing": False,
            "frozen_before_first_GNN": True,
        }
    union = selected["M1"] | selected["M2"] | selected["RANDOM"]
    information = {
        "replicates": information_rows,
        "M1_NO_SELECTION_DIFFERENCE": m1_all_identical,
        "M2_NO_SELECTION_DIFFERENCE": m2_all_identical,
        "all_metric_methods_equal_random": m1_all_identical and m2_all_identical,
        "M1_selected_slots": SLOTS_PER_METHOD, "M2_selected_slots": SLOTS_PER_METHOD,
        "random_selected_slots": SLOTS_PER_METHOD,
        "selected_union_unique_configurations": len(union),
        "physical_unique_GNN_runs": 3 * len(union),
        "method_equivalent_config_slots": 3 * SLOTS_PER_METHOD,
        "method_equivalent_GNN_runs": 3 * SLOTS_PER_METHOD * len(TRAINING_SEEDS),
        "no_artificial_disagreement_gate": True,
    }
    collapse = selection_collapse_audit(candidates, plans)
    return plans["M1"], plans["M2"], plans["RANDOM"], selected, information, collapse


def _collapse_statistics(joints: Sequence[str], by_joint: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    architectures = [str(by_joint[joint]["architecture_fingerprint"]) for joint in joints]
    counts = Counter(architectures)
    total = len(architectures)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    probabilities = [count / total for _arch, count in ordered] if total else []
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    return {
        "selected_slots_count": total,
        "unique_architecture_count": len(counts),
        "top_1_architecture_frequency": probabilities[0] if probabilities else 0.0,
        "top_2_cumulative_frequency": sum(probabilities[:2]),
        "architecture_selection_entropy": entropy,
        "effective_number_of_architectures": math.exp(entropy),
        "architecture_frequency_distribution": dict(ordered),
    }


def selection_collapse_audit(
    candidates: Sequence[Mapping[str, Any]], plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_joint = {str(row["joint_fingerprint"]): row for row in candidates}
    output: dict[str, Any] = {"selection_modified": False, "methods": {}}
    for method, plan in plans.items():
        rounds = list(plan["rounds"])
        all_joints = [joint for row in rounds for joint in row["selected_joint_fingerprints"]]
        per_replicate = {}
        per_round = {}
        for replicate in range(REPLICATES):
            joints = [joint for row in rounds if int(row["replicate_id"]) == replicate for joint in row["selected_joint_fingerprints"]]
            per_replicate[str(replicate)] = _collapse_statistics(joints, by_joint)
        for row in rounds:
            key = f"replicate={int(row['replicate_id'])},round={int(row['round_id'])}"
            per_round[key] = _collapse_statistics(row["selected_joint_fingerprints"], by_joint)
        output["methods"][method] = {
            "all_selected_slots": _collapse_statistics(all_joints, by_joint),
            "per_replicate": per_replicate, "per_round": per_round,
        }
    return output


def assert_candidate_manifest_immutable(before: Any, after: Any) -> None:
    if stable_hash(before) != stable_hash(after):
        raise PreSearchFreezeError("candidate manifest changed after freeze")


def freeze_pre_search(
    pre_search: Path, candidates: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]], m1_plan: Mapping[str, Any],
    m2_plan: Mapping[str, Any], random_plan: Mapping[str, Any],
    collapse: Mapping[str, Any], scorer_identity: Mapping[str, Any],
    generator_identity: Mapping[str, Any], information: Mapping[str, Any],
) -> dict[str, Any]:
    if pre_search.exists() and any(pre_search.iterdir()):
        raise PreSearchFreezeError(f"pre-search directory is not empty: {pre_search}")
    pre_search.mkdir(parents=True, exist_ok=True)
    score_by_joint = {str(row["joint_fingerprint"]): dict(row) for row in predictions}
    m1_selected = {joint for row in m1_plan["rounds"] for joint in row["selected_joint_fingerprints"]}
    m2_selected = {joint for row in m2_plan["rounds"] for joint in row["selected_joint_fingerprints"]}
    random_selected = {
        joint for row in random_plan["rounds"] for joint in row["selected_joint_fingerprints"]
    }
    manifest_rows = []
    for candidate in sorted(candidates, key=lambda row: (int(row["replicate_id"]), int(row["round_id"]), int(row["candidate_slot"]))):
        joint = str(candidate["joint_fingerprint"])
        manifest_rows.append({
            **dict(candidate),
            "m1_prediction": float(score_by_joint[joint]["m1_prediction"]),
            "m2_prediction": float(score_by_joint[joint]["m2_prediction"]),
            "m1_selected": joint in m1_selected,
            "m2_selected": joint in m2_selected,
            "random_selected": joint in random_selected,
        })
    search_plan = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "candidate_count": len(candidates),
        "candidate_semantic_sha256": stable_hash(list(candidates)),
        "all_candidates_generated_before_first_GNN": True,
        "all_M1_predictions_completed_before_first_GNN": True,
        "all_M2_predictions_completed_before_first_GNN": True,
        "all_M1_selections_completed_before_first_GNN": True,
        "all_M2_selections_completed_before_first_GNN": True,
        "all_random_selections_completed_before_first_GNN": True,
        "candidate_replacement_after_freeze": False,
        "online_update": False,
        "information_check": dict(information),
    }
    dump_csv(pre_search / "candidate_manifest.csv", manifest_rows, CSV_FIELDS["pre_search/candidate_manifest.csv"])
    dump_csv(pre_search / "m1_predictions.csv", predictions, CSV_FIELDS["pre_search/m1_predictions.csv"])
    dump_csv(pre_search / "m2_predictions.csv", predictions, CSV_FIELDS["pre_search/m2_predictions.csv"])
    dump_json(pre_search / "m1_selections.json", m1_plan)
    dump_json(pre_search / "m2_selections.json", m2_plan)
    dump_json(pre_search / "random_selections.json", random_plan)
    dump_json(pre_search / "selection_collapse.json", collapse)
    dump_json(pre_search / "search_plan.json", search_plan)
    dump_json(pre_search / "scorer_identity.json", scorer_identity)
    dump_json(pre_search / "generator_identity.json", generator_identity)
    files = [
        {"relative_path": relative, "sha256": sha256_file(pre_search / relative)}
        for relative in sorted(PRE_SEARCH_FILES)
    ]
    root_sha256 = stable_hash([(row["relative_path"], row["sha256"]) for row in files])
    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "root_algorithm": "sha256_of_canonical_JSON_sorted_relative_path_and_file_sha256_pairs",
        "files": files,
        "root_sha256": root_sha256,
        "candidate_count": len(candidates),
        "M1_selected_slots": SLOTS_PER_METHOD,
        "M2_selected_slots": SLOTS_PER_METHOD,
        "random_selected_slots": SLOTS_PER_METHOD,
        "historical_source_hashes": scorer_identity.get("historical_source_hashes", {}),
        "previous_prospective_archive_hashes": generator_identity.get("previous_prospective_archive_hashes", {}),
        "candidate_generation_seeds": [CANDIDATE_MASTER_SEED + replicate for replicate in range(REPLICATES)],
        "random_master_seed": RANDOM_MASTER_SEED,
        "metric_source_hashes": scorer_identity.get("metric_source_hashes", {}),
    }
    dump_json(pre_search / "freeze_manifest.json", freeze)
    dump_text(pre_search / "pre_search_root_sha256.txt", root_sha256 + "\n")
    dump_text(pre_search / "PRE_SEARCH_FROZEN", root_sha256 + "\n")
    return freeze


def verify_pre_search_freeze(pre_search: Path) -> dict[str, Any]:
    marker = pre_search / "PRE_SEARCH_FROZEN"
    manifest_path = pre_search / "freeze_manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        raise PreSearchFreezeError("pre-search freeze marker or manifest is missing")
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_order = sorted(freeze["files"], key=lambda row: str(row["relative_path"]))
    if freeze["files"] != expected_order:
        raise PreSearchFreezeError("pre-search freeze entries are not sorted by relative path")
    mismatches = []
    for row in freeze["files"]:
        path = pre_search / row["relative_path"]
        observed = sha256_file(path) if path.is_file() else None
        if observed != row["sha256"]:
            mismatches.append({"relative_path": row["relative_path"], "expected": row["sha256"], "observed": observed})
    root = stable_hash([(row["relative_path"], row["sha256"]) for row in freeze["files"]])
    stored_text = (pre_search / "pre_search_root_sha256.txt").read_text(encoding="utf-8").strip()
    marker_text = marker.read_text(encoding="utf-8").strip()
    if mismatches or root != freeze["root_sha256"] or stored_text != root or marker_text != root:
        raise PreSearchFreezeError(f"pre-search root verification failed: {mismatches}")
    return {"passed": True, "root_sha256": root, "file_count": len(freeze["files"])}


def _selection_occurrences(plan: Mapping[str, Any], method: str) -> dict[str, list[dict[str, int | str]]]:
    output: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    for row in plan["rounds"]:
        for joint in row["selected_joint_fingerprints"]:
            output[str(joint)].append({
                "method": method,
                "replicate_id": int(row["replicate_id"]),
                "round_id": int(row["round_id"]),
            })
    return output


def build_selected_union_tasks(
    candidates: Sequence[Mapping[str, Any]], m1_plan: Mapping[str, Any],
    m2_plan: Mapping[str, Any], random_plan: Mapping[str, Any], *, pre_search_verified: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not pre_search_verified:
        raise PreSearchFreezeError("task creation is forbidden before pre-search verification")
    by_joint = {str(row["joint_fingerprint"]): dict(row) for row in candidates}
    occurrences_by_method = {
        "M1": _selection_occurrences(m1_plan, "M1"),
        "M2": _selection_occurrences(m2_plan, "M2"),
        "RANDOM": _selection_occurrences(random_plan, "RANDOM"),
    }
    selected_joints = sorted(set().union(*(set(rows) for rows in occurrences_by_method.values())))
    protocol_identity = stable_hash({
        "training_seeds": list(TRAINING_SEEDS), "max_epochs": 300,
        "patience": 80, "maximum_system_retries": 2,
        "track_test": False, "component_variant": "baseline",
    })
    configs = []
    tasks = []
    for joint in selected_joints:
        if joint not in by_joint:
            raise ProtocolError(f"selection references foreign candidate {joint}")
        candidate = by_joint[joint]
        occurrences = sorted(
            [row for values in occurrences_by_method.values() for row in values.get(joint, [])],
            key=lambda row: (str(row["method"]), int(row["replicate_id"]), int(row["round_id"])),
        )
        replicate = int(candidate["replicate_id"])
        assigned_gpu = 0 if replicate < REPLICATES // 2 else 1
        config_row = {
            **candidate,
            "assigned_gpu": assigned_gpu,
            "selected_by": occurrences,
            "selected_by_M1": joint in occurrences_by_method["M1"],
            "selected_by_M2": joint in occurrences_by_method["M2"],
            "selected_by_random": joint in occurrences_by_method["RANDOM"],
            "training_protocol_identity": protocol_identity,
        }
        configs.append(config_row)
        for seed in TRAINING_SEEDS:
            tasks.append({
                "joint_fingerprint": joint,
                "training_seed": seed,
                "assigned_gpu": assigned_gpu,
                "task_key": f"m1-m2-metric-search::{joint}::seed={seed}::protocol={protocol_identity}",
                "training_protocol_identity": protocol_identity,
                "candidate": config_row,
            })
    expected = {(joint, seed) for joint in selected_joints for seed in TRAINING_SEEDS}
    observed = {(str(row["joint_fingerprint"]), int(row["training_seed"])) for row in tasks}
    if observed != expected or len(tasks) != len(expected):
        raise ProtocolError("selected-union task de-duplication failed")
    reject_test_named_fields(tasks, allow_false_controls=True)
    return configs, tasks


def validation_only_data_view(data: Any) -> Any:
    """Create the only data object allowed to cross the evaluator boundary."""
    try:
        from torch_geometric.data import Data
    except ImportError as error:
        raise ProtocolError("torch-geometric is required for GNN evaluation") from error
    required = ("x", "edge_index", "y", "train_mask", "val_mask")
    missing = [field for field in required if not hasattr(data, field)]
    if missing:
        raise ProtocolError(f"dataset lacks validation-only fields: {missing}")
    view = Data(**{field: getattr(data, field) for field in required})
    keys = list(view.keys()) if callable(getattr(view, "keys", None)) else list(view.keys)
    forbidden = [key for key in keys if _is_test_key(str(key))]
    if forbidden:
        raise ProtocolError(f"validation-only data view contains test fields: {forbidden}")
    return view


def load_validation_only_dataset(
    request: Any,
    dataset_loader: Callable[[Any], Any],
    context_loader: Callable[[Any], Mapping[str, Any]],
) -> tuple[Any, int, int, dict[str, Any], dict[str, Any]]:
    """Loader boundary that exposes no test-bearing dataset object to callers."""
    bundle = dataset_loader(request)
    try:
        context = dict(context_loader(bundle.manifest))
        num_features = int(bundle.num_features)
        num_classes = int(bundle.num_classes)
        data = validation_only_data_view(bundle.data)
    finally:
        # The underlying compatibility loader may materialize a test mask, but
        # neither its bundle nor its original Data object can leave this
        # adapter.  Only the explicitly reconstructed validation view does.
        del bundle
    keys = list(data.keys()) if callable(getattr(data, "keys", None)) else list(data.keys)
    if any(_is_test_key(str(key)) for key in keys):
        raise ProtocolError("test field escaped the validation-only dataset loader")
    audit = {
        "boundary": "load_validation_only_dataset",
        "exposed_fields": sorted(map(str, keys)),
        "raw_loader_bundle_exposed": False,
        "test_named_fields_exposed": [],
    }
    return data, num_features, num_classes, context, audit


def _frozen_dependency_entries() -> list[dict[str, str]]:
    from analyse import joint_metric_prospective_selection_gate as pilot

    entries = pilot._frozen_dependency_entries()
    if not entries:
        raise InputIdentityInvalid("frozen evaluator dependency manifest is empty")
    return [{"path": str(row["path"]), "sha256": str(row["sha256"])} for row in entries]


def evaluator_task_rows(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        candidate = task["candidate"]
        rows.append({
            "task_stage": "initial",
            "pool_id": int(candidate["replicate_id"]),
            "batch_id": int(candidate["round_id"]),
            "architecture_slot": int(candidate["candidate_slot"]),
            "config_id": str(candidate["config_id"]),
            "architecture_fingerprint": str(candidate["architecture_fingerprint"]),
            "hp_fingerprint": str(candidate["hp_fingerprint"]),
            "joint_config_fingerprint": str(candidate["joint_fingerprint"]),
            "hp_regime": "hp_high",
            "seed": int(task["training_seed"]),
            "task_key": str(task["task_key"]),
            "training_protocol_identity": str(task["training_protocol_identity"]),
            "assigned_gpu": int(task["assigned_gpu"]),
            "selected_by": copy.deepcopy(candidate["selected_by"]),
            "operations": list(candidate["operations"]),
            "edges": [list(map(int, edge)) for edge in candidate["edges"]],
            "effective_layers": int(candidate["effective_layers"]),
            "raw_hp_vector": [float(value) for value in candidate["raw_hp_vector"]],
            "lr": float(candidate["lr"]),
            "dropout": float(candidate["dropout"]),
            "hidden_dim": int(candidate["hidden_dim"]),
            "l2": float(candidate["l2"]),
            "gat_heads": int(candidate["gat_heads"]),
            "sage_aggr": str(candidate["sage_aggr"]),
            "gin_eps": float(candidate["gin_eps"]),
            "gat_heads_by_layer": candidate["gat_heads_by_layer"],
            "sage_aggr_by_layer": candidate["sage_aggr_by_layer"],
            "gin_eps_by_layer": candidate["gin_eps_by_layer"],
        })
    reject_test_named_fields(rows)
    return rows


def prepare_evaluator_adapter(
    output: Path, tasks: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    pre_search_root: str,
) -> tuple[Path, list[dict[str, Any]], str]:
    adapter = output / "evaluation/evaluator_adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    rows = evaluator_task_rows(tasks)
    firewall = {
        "passed": True,
        "boundary": "validation_only_data_view_before_train_and_eval_arch",
        "allowed_data_fields": ["x", "edge_index", "y", "train_mask", "val_mask"],
        "test_named_task_fields": [],
        "track_test": False,
        "pre_search_root_sha256": pre_search_root,
        "created_at": utc_now(),
    }
    preregistration = {
        "format_version": PROTOCOL_VERSION,
        "training": {
            "dataset": "Cora",
            "split": "public",
            "graph_transform": "NormalizeFeatures",
            "dataset_content_fingerprint": "373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7",
            "split_fingerprint": "bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09",
            "max_epochs": 300,
            "patience": 80,
            "maximum_system_retries": 2,
            "training_seeds": list(TRAINING_SEEDS),
            "track_test": False,
            "validation_only_data_view": True,
            "shard_to_gpu_uuid": dict(config["evaluation"]["shard_to_gpu_uuid"]),
        },
    }
    dependency_manifest = {"format_version": PROTOCOL_VERSION, "files": _frozen_dependency_entries()}
    dump_json(adapter / "pre_label_firewall_audit.json", firewall)
    dump_json(adapter / "selected_union_tasks.json", rows)
    dump_json(adapter / "preregistration.json", preregistration)
    dump_json(adapter / "source_dependency_manifest.json", dependency_manifest)
    ordered = (
        "pre_label_firewall_audit.json", "selected_union_tasks.json",
        "preregistration.json", "source_dependency_manifest.json",
    )
    files = [{"relative_path": relative, "sha256": sha256_file(adapter / relative)} for relative in ordered]
    root = stable_hash([(row["relative_path"], row["sha256"]) for row in files])
    dump_json(adapter / "pre_label_immutable_manifest.json", {
        "format_version": PROTOCOL_VERSION,
        "files": files,
        "manifest_root_hash": root,
    })
    return adapter, rows, root


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProtocolError(f"malformed JSONL at {path}:{line_number}") from error
            reject_test_named_fields(row)
            rows.append(row)
    return rows


def _result_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["joint_config_fingerprint"]), int(row["seed"])


def execute_evaluator_shard(
    adapter: Path, shard: int, output: Path, cora_root: str, device_name: str,
) -> int:
    """GPU subprocess entry using the archived evaluator core and a no-test view."""
    import torch
    from analyse import high_regime_prior_model_blind_validation as frozen_plan
    from analyse import run_high_regime_prior_model_blind_validation_labels as frozen_runner
    from analyse.vae_label_expansion import EXPECTED_DATASET_CONTEXT
    from dataset_utils import (
        assert_dataset_context_matches,
        dataset_context_from_manifest,
        load_dataset_from_request,
        resolve_dataset_request,
    )
    from eval_utils import train_and_eval_arch

    expected_runner = (HIGH_SNAPSHOT / "analyse/run_high_regime_prior_model_blind_validation_labels.py").resolve()
    if Path(frozen_runner.__file__).resolve() != expected_runner:
        raise InputIdentityInvalid(f"evaluator runner resolved outside frozen snapshot: {frozen_runner.__file__}")
    if Path(frozen_plan.__file__).resolve() != (HIGH_SNAPSHOT / "analyse/high_regime_prior_model_blind_validation.py").resolve():
        raise InputIdentityInvalid("high-regime evaluator plan resolved outside frozen snapshot")
    if Path(train_and_eval_arch.__code__.co_filename).resolve() != (HIGH_SNAPSHOT / "eval_utils.py").resolve():
        raise InputIdentityInvalid("train_and_eval_arch resolved outside frozen snapshot")
    freeze = frozen_plan.verify_freeze(adapter)
    if not freeze["passed"]:
        raise PreSearchFreezeError(f"evaluator adapter freeze failed: {freeze}")
    all_tasks = json.loads((adapter / "selected_union_tasks.json").read_text(encoding="utf-8"))
    reject_test_named_fields(all_tasks)
    tasks = sorted(
        [dict(row) for row in all_tasks if int(row["assigned_gpu"]) == shard],
        key=lambda row: (str(row["joint_config_fingerprint"]), int(row["seed"])),
    )
    if not tasks:
        raise ProtocolError(f"empty evaluator shard {shard}")
    if {int(row["seed"]) for row in tasks} != set(TRAINING_SEEDS):
        raise ProtocolError("evaluator shard does not contain exactly seeds 0,1,2")
    config_shards: dict[str, set[int]] = defaultdict(set)
    for row in all_tasks:
        config_shards[str(row["joint_config_fingerprint"])].add(int(row["assigned_gpu"]))
    if any(len(values) != 1 for values in config_shards.values()):
        raise ProtocolError("one configuration was split across GPU shards")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "seed_results.jsonl"
    completed = _load_jsonl(result_path)
    completed_keys = {_result_key(row) for row in completed}
    expected_keys = {_result_key(row) for row in tasks}
    if completed_keys - expected_keys or len(completed_keys) != len(completed):
        raise ProtocolError("evaluator shard has foreign or duplicate completed tasks")
    pending = [row for row in tasks if _result_key(row) not in completed_keys]
    run_plan = {
        "stage": "initial",
        "shard": shard,
        "configuration_count": len({row["joint_config_fingerprint"] for row in tasks}),
        "expected_task_count": len(tasks),
        "already_completed": len(completed),
        "pending_task_count": len(pending),
        "dataset_loaded": False,
        "validation_only_view_created": False,
        "training_started": False,
        "track_test": False,
        "pre_label_manifest_root_hash": freeze["recomputed_root_hash"],
    }
    dump_json(output / "run_plan.json", run_plan)
    prereg = json.loads((adapter / "preregistration.json").read_text(encoding="utf-8"))
    device = torch.device(device_name)
    gpu = frozen_runner.physical_gpu_identity(device)
    expected_uuid = str(prereg["training"]["shard_to_gpu_uuid"][str(shard)])
    if gpu["physical_index"] != shard or gpu["uuid"] != expected_uuid:
        raise ProtocolError(f"shard/GPU mismatch: shard={shard}, actual={gpu}, expected_uuid={expected_uuid}")
    request = resolve_dataset_request("cora", None, cora_root=cora_root)
    data, num_features, num_classes, context, loader_audit = load_validation_only_dataset(
        request, load_dataset_from_request, dataset_context_from_manifest,
    )
    assert_dataset_context_matches(EXPECTED_DATASET_CONTEXT, context, context="metric search validation")
    if context["dataset_content_fingerprint"] != prereg["training"]["dataset_content_fingerprint"] or context["split_fingerprint"] != prereg["training"]["split_fingerprint"]:
        raise InputIdentityInvalid("dataset or split fingerprint differs from frozen high-regime evaluation")
    data = data.to(device)
    data_keys = list(data.keys()) if callable(getattr(data, "keys", None)) else list(data.keys)
    if any(_is_test_key(str(key)) for key in data_keys):
        raise ProtocolError("test field reached the GNN evaluator boundary")
    run_plan.update({
        "dataset_loaded": True,
        "validation_only_view_created": True,
        "validation_only_fields": sorted(map(str, data_keys)),
        "loader_firewall": loader_audit,
        "test_named_fields_at_evaluator_boundary": [],
        "training_started": bool(pending),
    })
    dump_json(output / "run_plan.json", run_plan)
    started_at = utc_now()

    def write_progress() -> None:
        rows = _load_jsonl(result_path)
        dump_json(output / "progress.json", {
            "stage": "initial", "shard": shard, "started_at": started_at,
            "updated_at": utc_now(), "expected": len(tasks), "completed": len(rows),
            "remaining": len(tasks) - len(rows),
            "terminal_status_counts": dict(Counter(str(row["terminal_status"]) for row in rows)),
            "retry_count": sum(int(row.get("retry_count", 0)) for row in rows),
            "actual_gpu": gpu,
            "validation_only_view": True,
            "test_accessed": False,
        })

    write_progress()
    maximum_retries = int(prereg["training"]["maximum_system_retries"])
    for task in pending:
        failures = []
        attempt = 0
        task_started = utc_now()
        monotonic = time.monotonic()
        while True:
            attempt += 1
            try:
                val_acc, valid, metadata = train_and_eval_arch(
                    config={
                        "operations": list(task["operations"]),
                        "edges": [tuple(edge) for edge in task["edges"]],
                        "effective_layers": int(task["effective_layers"]),
                    },
                    data=data,
                    in_ch=num_features,
                    out_ch=num_classes,
                    lr=float(task["lr"]),
                    dropout=float(task["dropout"]),
                    hidden_dim=int(task["hidden_dim"]),
                    weight_decay=float(task["l2"]),
                    gcnii_alpha=0.1,
                    gcnii_theta=0.5,
                    gat_heads=int(task["gat_heads"]),
                    sage_aggr=str(task["sage_aggr"]),
                    gin_eps=float(task["gin_eps"]),
                    gat_heads_by_layer=task["gat_heads_by_layer"],
                    sage_aggr_by_layer=task["sage_aggr_by_layer"],
                    gin_eps_by_layer=task["gin_eps_by_layer"],
                    device=device,
                    max_epochs=300,
                    patience=80,
                    seed=int(task["seed"]),
                    track_test=False,
                    return_metadata=True,
                    return_reproducibility_metadata=True,
                    return_trajectory=True,
                    component_variant="baseline",
                )
                cleaned = frozen_runner.sanitize_training_metadata(metadata)
                row = {
                    **frozen_runner.base_result(task, context, gpu),
                    "val_acc": float(val_acc),
                    "valid": bool(valid),
                    "terminal_status": "success" if valid else "invalid",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": failures,
                    "started_at": task_started,
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - monotonic,
                    **cleaned,
                }
                break
            except Exception as error:  # retry policy is frozen and intentionally broad
                if isinstance(error, frozen_runner.ProtocolFirewallViolation):
                    raise
                failures.append({
                    "attempt": attempt, "type": type(error).__name__, "message": str(error),
                    "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                })
                if attempt <= maximum_retries:
                    continue
                row = {
                    **frozen_runner.base_result(task, context, gpu),
                    "val_acc": None,
                    "valid": False,
                    "terminal_status": "failed",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": failures,
                    "started_at": task_started,
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - monotonic,
                }
                break
        reject_test_named_fields(row)
        frozen_runner.append_jsonl(result_path, row)
        write_progress()
    return 0


def run_frozen_gnn_evaluation(
    output: Path, tasks: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    pre_search_root: str, log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verify_pre_search_freeze(output / "pre_search")
    adapter, evaluator_rows, adapter_root = prepare_evaluator_adapter(
        output, tasks, config, pre_search_root,
    )
    task_manifest = {
        "authorized": True,
        "pre_search_root_sha256": pre_search_root,
        "selected_unique_config_count": len({row["joint_config_fingerprint"] for row in evaluator_rows}),
        "planned_GNN_runs": len(evaluator_rows),
        "method_equivalent_config_slots": 3 * SLOTS_PER_METHOD,
        "method_equivalent_GNN_runs": 3 * SLOTS_PER_METHOD * len(TRAINING_SEEDS),
        "training_seeds": list(TRAINING_SEEDS),
        "same_config_shared_across_methods": True,
        "track_test": False,
        "tasks": evaluator_rows,
    }
    dump_json(output / "evaluation/task_manifest.json", task_manifest)
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    handles = []
    for shard in (0, 1):
        shard_output = output / f"evaluation/gpu{shard}"
        shard_output.mkdir(parents=True, exist_ok=True)
        handle = (output / f"logs/evaluation_gpu{shard}.log").open("w", encoding="utf-8")
        handles.append(handle)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(shard)
        environment["PYTHONHASHSEED"] = "0"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        inherited = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(HIGH_SNAPSHOT), str(REPO_ROOT)] + ([inherited] if inherited else [])
        )
        command = [
            sys.executable, str(Path(__file__).resolve()), "--evaluate-shard",
            "--adapter", str(adapter), "--shard", str(shard),
            "--shard-output", str(shard_output), "--cora-root", "Cora",
            "--device", "cuda:0",
        ]
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, text=True,
        )
        processes.append((shard, process))
    update_state(
        output, stage="EVALUATION_RUNNING", completed_gnn_runs=0,
        failed_gnn_runs=0, current_replicate=0, current_round=0,
        current_config=None, current_seed=None,
    )
    return_codes: dict[str, int] = {}
    task_by_shard = {
        shard: [row for row in evaluator_rows if int(row["assigned_gpu"]) == shard]
        for shard in (0, 1)
    }
    try:
        active = list(processes)
        while active:
            completed_count = failed_count = 0
            completed_by_shard = {0: 0, 1: 0}
            for shard in (0, 1):
                path = output / f"evaluation/gpu{shard}/seed_results.jsonl"
                rows = _load_jsonl(path)
                completed_by_shard[shard] = len(rows)
                completed_count += len(rows)
                failed_count += sum(str(row.get("terminal_status")) != "success" for row in rows)
            remaining = []
            for shard, process in active:
                code = process.poll()
                if code is None:
                    remaining.append((shard, process))
                else:
                    return_codes[str(shard)] = int(code)
            current = None
            for shard in (0, 1):
                if completed_by_shard[shard] < len(task_by_shard[shard]):
                    current = task_by_shard[shard][completed_by_shard[shard]]
                    break
            update_state(
                output,
                stage="EVALUATION_RUNNING",
                completed_gnn_runs=completed_count,
                failed_gnn_runs=failed_count,
                current_replicate=None if current is None else int(current["pool_id"]),
                current_round=None if current is None else int(current["batch_id"]),
                current_config=None if current is None else str(current["joint_config_fingerprint"]),
                current_seed=None if current is None else int(current["seed"]),
            )
            active = remaining
            if active:
                time.sleep(2.0)
    finally:
        for _shard, process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()
    raw = []
    task_lookup = {_result_key(row): row for row in evaluator_rows}
    loader_firewalls = {}
    for shard in (0, 1):
        plan_path = output / f"evaluation/gpu{shard}/run_plan.json"
        if plan_path.is_file():
            loader_firewalls[str(shard)] = json.loads(plan_path.read_text(encoding="utf-8"))
        for archived in _load_jsonl(output / f"evaluation/gpu{shard}/seed_results.jsonl"):
            key = _result_key(archived)
            task = task_lookup.get(key)
            if task is None:
                raise ProtocolError("evaluator returned a foreign task")
            for field in ("architecture_fingerprint", "hp_fingerprint", "task_key", "assigned_gpu"):
                if archived.get(field) != task.get(field):
                    raise ProtocolError(f"evaluator result identity mismatch for {field}")
            raw.append({
                "joint_fingerprint": key[0],
                "training_seed": key[1],
                "validation_accuracy": archived.get("val_acc"),
                "terminal_status": archived.get("terminal_status"),
                "valid": archived.get("valid"),
                "attempt_count": archived.get("attempt_count", 1),
                "assigned_gpu": archived.get("assigned_gpu"),
            })
    all_loader_firewalls_passed = (
        set(loader_firewalls) == {"0", "1"}
        and all(row.get("validation_only_view_created") is True for row in loader_firewalls.values())
        and all(row.get("test_named_fields_at_evaluator_boundary") == [] for row in loader_firewalls.values())
    )
    audit = {
        "adapter_manifest_root": adapter_root,
        "task_count": len(evaluator_rows),
        "selected_unique_config_count": len(evaluator_rows) // len(TRAINING_SEEDS),
        "physical_unique_GNN_runs": len(evaluator_rows),
        "method_equivalent_GNN_runs": 3 * SLOTS_PER_METHOD * len(TRAINING_SEEDS),
        "shard_return_codes": return_codes,
        "maximum_system_retries": 2,
        "candidate_replacement": False,
        "same_config_seed_results_shared": True,
        "track_test": False,
        "loader_firewall": loader_firewalls,
        "loader_firewall_passed": all_loader_firewalls_passed,
        "test_accessed": False,
        "all_shards_exited_zero": set(return_codes) == {"0", "1"} and all(code == 0 for code in return_codes.values()),
    }
    log(
        f"evaluation subprocesses finished: raw={len(raw)}/{len(evaluator_rows)}, "
        f"return_codes={return_codes}"
    )
    return raw, audit


def aggregate_config_statistics(
    selected_configs: Sequence[Mapping[str, Any]], raw_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]], *, allow_incomplete: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    reject_test_named_fields(raw_rows)
    by_joint = {str(row["joint_fingerprint"]): dict(row) for row in selected_configs}
    expected = {(joint, seed) for joint in by_joint for seed in TRAINING_SEEDS}
    seen_keys: set[tuple[str, int]] = set()
    successful: dict[tuple[str, int], float] = {}
    for row in raw_rows:
        key = (str(row["joint_fingerprint"]), int(row["training_seed"]))
        if key not in expected or key in seen_keys:
            raise EvaluationIncomplete("evaluation contains a foreign or duplicate config/seed")
        seen_keys.add(key)
        if row.get("terminal_status") == "success" and row.get("valid") is True:
            value = float(row["validation_accuracy"])
            if math.isfinite(value):
                successful[key] = value
    complete = seen_keys == expected and set(successful) == expected
    score_by_joint = {str(row["joint_fingerprint"]): dict(row) for row in predictions}
    stats = []
    for joint, candidate in sorted(by_joint.items()):
        values = [successful.get((joint, seed)) for seed in TRAINING_SEEDS]
        if any(value is None for value in values):
            if not allow_incomplete:
                raise EvaluationIncomplete(f"selected config lacks seeds 0,1,2: {joint}")
            mean = sd = None
        else:
            array = np.asarray(values, dtype=float)
            mean = float(np.mean(array))
            sd = float(np.std(array, ddof=1))
        stats.append({
            "joint_fingerprint": joint,
            "architecture_fingerprint": str(candidate["architecture_fingerprint"]),
            "validation_seed_0": values[0],
            "validation_seed_1": values[1],
            "validation_seed_2": values[2],
            "validation_mean": mean,
            "validation_sd": sd,
            "m1_prediction": float(score_by_joint[joint]["m1_prediction"]),
            "m2_prediction": float(score_by_joint[joint]["m2_prediction"]),
            "selected_by_M1": bool(candidate["selected_by_M1"]),
            "selected_by_M2": bool(candidate["selected_by_M2"]),
            "selected_by_random": bool(candidate["selected_by_random"]),
        })
    return stats, complete


def bootstrap_paired_mean(
    differences: Sequence[float], *, replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED, statistic: str = "mean_paired_final_best_difference",
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (REPLICATES,):
        raise ProtocolError("bootstrap unit must be the eight search replicates")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = np.mean(values[indices], axis=1)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return {
        "unit": "search_replicate",
        "replicate_count": len(values),
        "bootstrap_replicates": replicates,
        "seed": seed,
        "statistic": statistic,
        "observed_mean": float(np.mean(values)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "draw_mean": float(np.mean(draws)),
    }


def exact_one_sided_sign_test(wins: int, losses: int, ties: int, *, alternative: str) -> dict[str, Any]:
    non_ties = wins + losses
    if non_ties == 0:
        p_value = 1.0
    else:
        p_value = sum(math.comb(non_ties, successes) for successes in range(wins, non_ties + 1)) / (2 ** non_ties)
    return {
        "alternative": alternative,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "non_ties": non_ties,
        "exact_one_sided_p_value": float(p_value),
        "primary_gate_use": False,
    }


def _winner(joints: Sequence[str], stats_by_joint: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not joints:
        raise ProtocolError("cannot choose a winner from an empty selection")
    ordered = sorted(
        (stats_by_joint[str(joint)] for joint in joints),
        key=lambda row: (-float(row["validation_mean"]), str(row["joint_fingerprint"])),
    )
    return dict(ordered[0])


def _safe_correlations(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float | None]:
    if len(actual) < 2 or len(set(actual)) < 2 or len(set(predicted)) < 2:
        return {"pearson": None, "spearman": None}
    from scipy.stats import pearsonr, spearmanr
    return {
        "pearson": float(pearsonr(actual, predicted).statistic),
        "spearman": float(spearmanr(actual, predicted).statistic),
    }


def classify_metric_gate(
    method: str, per_replicate: Sequence[Mapping[str, Any]],
    anytime_rows: Sequence[Mapping[str, Any]], bootstrap: Mapping[str, Any],
    *, tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    difference_key = f"{method}_minus_random"
    auc_key = f"{method}_anytime_AUC"
    differences = [float(row[difference_key]) for row in per_replicate]
    wins = sum(value > 0.0 for value in differences)
    mean_difference = float(np.mean(differences))
    method_auc = float(np.mean([float(row[auc_key]) for row in per_replicate]))
    random_auc = float(np.mean([float(row["random_anytime_AUC"]) for row in per_replicate]))
    budget_means = []
    for budget in (4, 8, 12, 16):
        method_values = [float(row["best_so_far"]) for row in anytime_rows if row["method"] == method and int(row["budget"]) == budget]
        random_values = [float(row["best_so_far"]) for row in anytime_rows if row["method"] == "RANDOM" and int(row["budget"]) == budget]
        if len(method_values) != REPLICATES or len(random_values) != REPLICATES:
            raise ProtocolError(f"{method} anytime gate lacks eight records at budget {budget}")
        method_mean, random_mean = float(np.mean(method_values)), float(np.mean(random_values))
        budget_means.append({
            "budget": budget, f"{method}_mean_best_so_far": method_mean,
            "random_mean_best_so_far": random_mean,
            f"{method}_strictly_higher": method_mean > random_mean,
        })
    higher_points = sum(bool(row[f"{method}_strictly_higher"]) for row in budget_means)
    prefix = "K" if method == "M1" else "G"
    gates = {
        f"{prefix}1": {"passed": wins >= 7, "strict_wins": wins, "required": 7},
        f"{prefix}2": {"passed": mean_difference + tolerance >= 0.003, "mean_difference": mean_difference, "required_minimum": 0.003, "tolerance": tolerance},
        f"{prefix}3": {"passed": float(bootstrap["ci95_lower"]) > 0.0, "ci95_lower": float(bootstrap["ci95_lower"]), "required": "> 0"},
        f"{prefix}4": {"passed": method_auc > random_auc, f"{method}_mean_anytime_AUC": method_auc, "random_mean_anytime_AUC": random_auc},
        f"{prefix}5": {"passed": higher_points >= 3, f"{method}_higher_budget_points": higher_points, "required": 3, "budget_means": budget_means},
    }
    return {"method": method, "promoted": all(row["passed"] for row in gates.values()), "gates": gates}


def classify_surrogate_selection(
    per_replicate: Sequence[Mapping[str, Any]], bootstrap_m2_minus_m1: Mapping[str, Any],
    *, tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    m2_minus_m1 = [float(row["M2_minus_M1"]) for row in per_replicate]
    m1_regrets = [float(row["REGRET_M1"]) for row in per_replicate]
    m2_regrets = [float(row["REGRET_M2"]) for row in per_replicate]
    mean_m1, mean_m2 = float(np.mean(m1_regrets)), float(np.mean(m2_regrets))
    both_zero = mean_m1 <= tolerance and mean_m2 <= tolerance
    s2 = (not both_zero) and mean_m2 <= 0.90 * mean_m1 + tolerance
    gates = {
        "S1": {"passed": sum(value > 0.0 for value in m2_minus_m1) >= 6, "M2_strict_wins": sum(value > 0.0 for value in m2_minus_m1), "required": 6},
        "S2": {
            "passed": s2, "mean_REGRET_M1": mean_m1, "mean_REGRET_M2": mean_m2,
            "required": "mean_REGRET_M2 <= 0.90 * mean_REGRET_M1",
            "tolerance": tolerance, "both_mean_regrets_at_most_tolerance": both_zero,
        },
        "S3": {"passed": float(bootstrap_m2_minus_m1["ci95_lower"]) > 0.0, "ci95_lower": float(bootstrap_m2_minus_m1["ci95_lower"]), "required": "> 0"},
    }
    gp_increment = all(row["passed"] for row in gates.values())
    return {
        "NEXT_SURROGATE": "M2" if gp_increment else "M1",
        "clear_GP_increment": gp_increment,
        "gates": gates,
        "regret_oracle_name": "OBSERVED_EVALUATED_UNION_ORACLE_r",
        "regret_oracle_definition": "max(BEST_M1_r, BEST_M2_r, BEST_R_r)",
        "oracle_scope": "M1/M2/Random actually evaluated configuration union only",
        "is_complete_128_candidate_pool_oracle": False,
        "regret_interpretation": "surrogate-selection gate only; not full-pool simple regret",
        "parsimony_rule": "select M1 unless all S1-S3 establish clear M2 increment",
    }


def compute_search_analysis(
    config_stats: Sequence[Mapping[str, Any]], m1_plan: Mapping[str, Any],
    m2_plan: Mapping[str, Any], random_plan: Mapping[str, Any],
    information: Mapping[str, Any], training_audit: Mapping[str, Any],
    collapse: Mapping[str, Any],
) -> dict[str, Any]:
    stats_by_joint = {str(row["joint_fingerprint"]): dict(row) for row in config_stats}
    if any(row.get("validation_mean") is None for row in stats_by_joint.values()):
        raise EvaluationIncomplete("analysis received incomplete config statistics")
    plans = {"M1": m1_plan, "M2": m2_plan, "RANDOM": random_plan}
    plan_rows = {
        method: {(int(row["replicate_id"]), int(row["round_id"])): row for row in plan["rounds"]}
        for method, plan in plans.items()
    }
    per_round, anytime, per_replicate = [], [], []
    winners_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in plans}
    for replicate in range(REPLICATES):
        curves: dict[str, list[float]] = {}
        winners: dict[str, dict[str, Any]] = {}
        for method in ("M1", "M2", "RANDOM"):
            cumulative: list[str] = []
            curve = []
            for round_id in range(ROUNDS):
                selected = list(map(str, plan_rows[method][(replicate, round_id)]["selected_joint_fingerprints"]))
                if len(selected) != SELECTED_PER_ROUND:
                    raise ProtocolError("round does not contain exactly four selected configurations")
                cumulative.extend(selected)
                actual = [float(stats_by_joint[joint]["validation_mean"]) for joint in selected]
                best = _winner(cumulative, stats_by_joint)
                budget = (round_id + 1) * SELECTED_PER_ROUND
                curve.append(float(best["validation_mean"]))
                per_round.append({
                    "replicate_id": replicate, "round_id": round_id, "budget": budget,
                    "method": method, "selected_joint_fingerprints": selected,
                    "selected_validation_means": actual, "round_best": max(actual),
                    "best_so_far": float(best["validation_mean"]),
                })
                anytime.append({"replicate_id": replicate, "method": method, "budget": budget, "best_so_far": float(best["validation_mean"])})
            if len(cumulative) != 16 or len(set(cumulative)) != 16:
                raise ProtocolError("replicate method budget is not 16 unique configs")
            curves[method] = curve
            winners[method] = _winner(cumulative, stats_by_joint)
            winners_by_method[method].append({"replicate_id": replicate, **winners[method]})
        best_m1, best_m2, best_r = (float(winners[m]["validation_mean"]) for m in ("M1", "M2", "RANDOM"))
        observed_oracle = max(best_m1, best_m2, best_r)
        per_replicate.append({
            "replicate_id": replicate,
            "best_M1": best_m1, "best_M2": best_m2, "best_random": best_r,
            "M1_minus_random": best_m1 - best_r, "M2_minus_random": best_m2 - best_r,
            "M2_minus_M1": best_m2 - best_m1,
            "OBSERVED_EVALUATED_UNION_ORACLE": observed_oracle,
            "REGRET_M1": observed_oracle - best_m1, "REGRET_M2": observed_oracle - best_m2,
            "M1_winner_joint_fingerprint": str(winners["M1"]["joint_fingerprint"]),
            "M2_winner_joint_fingerprint": str(winners["M2"]["joint_fingerprint"]),
            "random_winner_joint_fingerprint": str(winners["RANDOM"]["joint_fingerprint"]),
            "M1_anytime_AUC": float(np.mean(curves["M1"])),
            "M2_anytime_AUC": float(np.mean(curves["M2"])),
            "random_anytime_AUC": float(np.mean(curves["RANDOM"])),
            "M1_above_historical_P": best_m1 > HISTORICAL_P_MEAN,
            "M2_above_historical_P": best_m2 > HISTORICAL_P_MEAN,
            "random_above_historical_P": best_r > HISTORICAL_P_MEAN,
        })
    differences = {
        "M1_vs_random": [float(row["M1_minus_random"]) for row in per_replicate],
        "M2_vs_random": [float(row["M2_minus_random"]) for row in per_replicate],
        "M2_vs_M1": [float(row["M2_minus_M1"]) for row in per_replicate],
    }
    bootstrap = {
        label: bootstrap_paired_mean(values, statistic=f"mean({label})")
        for label, values in differences.items()
    }
    m1_gate = classify_metric_gate("M1", per_replicate, anytime, bootstrap["M1_vs_random"])
    m2_gate = classify_metric_gate("M2", per_replicate, anytime, bootstrap["M2_vs_random"])
    surrogate = classify_surrogate_selection(per_replicate, bootstrap["M2_vs_M1"])
    m1_promoted, m2_promoted = bool(m1_gate["promoted"]), bool(m2_gate["promoted"])
    if not m1_promoted and not m2_promoted:
        classification, next_surrogate = "METRIC_SEARCH_NOT_CONFIRMED", None
    elif m1_promoted and not m2_promoted:
        classification, next_surrogate = "M1_SEARCH_PROMOTED", "M1"
    elif m2_promoted and not m1_promoted:
        classification, next_surrogate = "M2_SEARCH_PROMOTED", "M2"
    elif surrogate["NEXT_SURROGATE"] == "M2":
        classification, next_surrogate = "M1_M2_SEARCH_PROMOTED_GP_SELECTED", "M2"
    else:
        classification, next_surrogate = "M1_M2_SEARCH_PROMOTED_KNN_SELECTED", "M1"
    decision = {
        "classification": classification, "M1_SEARCH_PROMOTED": m1_promoted,
        "M2_SEARCH_PROMOTED": m2_promoted, "NEXT_SURROGATE": next_surrogate,
        "M1_gate": m1_gate, "M2_gate": m2_gate,
        "M1_vs_M2_surrogate_selection": surrogate,
        "historical_P_used_in_primary_gate": False,
        "fresh_seed_confirmation_started": False, "test_accessed": False,
    }
    comparisons = {}
    for label, values in differences.items():
        wins, losses = sum(v > 0 for v in values), sum(v < 0 for v in values)
        ties = len(values) - wins - losses
        comparisons[label] = {
            "mean_difference": float(np.mean(values)), "median_difference": float(np.median(values)),
            "wins": wins, "losses": losses, "ties": ties,
            "bootstrap": bootstrap[label],
            "sign_test": exact_one_sided_sign_test(wins, losses, ties, alternative=f"{label} > 0"),
        }
    actual = [float(row["validation_mean"]) for row in config_stats]
    comparisons["M1_vs_random"].update({
        "M1_final_best_mean": float(np.mean([row["best_M1"] for row in per_replicate])),
        "random_final_best_mean": float(np.mean([row["best_random"] for row in per_replicate])),
        "prediction_vs_actual": _safe_correlations(actual, [float(row["m1_prediction"]) for row in config_stats]),
    })
    comparisons["M2_vs_random"].update({
        "M2_final_best_mean": float(np.mean([row["best_M2"] for row in per_replicate])),
        "random_final_best_mean": float(np.mean([row["best_random"] for row in per_replicate])),
        "prediction_vs_actual": _safe_correlations(actual, [float(row["m2_prediction"]) for row in config_stats]),
    })
    comparisons["M2_vs_M1"].update(surrogate)
    comparisons["M2_vs_M1"]["per_replicate_regrets"] = [{
        "replicate_id": row["replicate_id"],
        "OBSERVED_EVALUATED_UNION_ORACLE": row["OBSERVED_EVALUATED_UNION_ORACLE"],
        "REGRET_M1": row["REGRET_M1"], "REGRET_M2": row["REGRET_M2"],
    } for row in per_replicate]
    auxiliary = {
        "physical_unique_configurations": int(training_audit["selected_unique_config_count"]),
        "physical_unique_GNN_runs": int(training_audit["physical_unique_GNN_runs"]),
        "method_equivalent_config_slots": 3 * SLOTS_PER_METHOD,
        "method_equivalent_GNN_runs": 3 * SLOTS_PER_METHOD * len(TRAINING_SEEDS),
        "selected_config_3seed_SD_mean": float(np.mean([float(row["validation_sd"]) for row in config_stats])),
        "historical_P": {
            "role": "DESCRIPTIVE_HISTORICAL_REFERENCE", "validation_mean": HISTORICAL_P_MEAN,
            "M1_replicate_winners_above_P": sum(bool(row["M1_above_historical_P"]) for row in per_replicate),
            "M2_replicate_winners_above_P": sum(bool(row["M2_above_historical_P"]) for row in per_replicate),
            "random_replicate_winners_above_P": sum(bool(row["random_above_historical_P"]) for row in per_replicate),
            "highest_M1_winner_3seed_mean": max(float(row["best_M1"]) for row in per_replicate),
            "highest_M2_winner_3seed_mean": max(float(row["best_M2"]) for row in per_replicate),
            "highest_random_winner_3seed_mean": max(float(row["best_random"]) for row in per_replicate),
            "primary_gate_use": False,
        },
        "selection_overlap": information["replicates"],
    }
    return {
        "per_round": per_round, "anytime": anytime, "per_replicate": per_replicate,
        "m1_winners": winners_by_method["M1"], "m2_winners": winners_by_method["M2"],
        "random_winners": winners_by_method["RANDOM"], "bootstrap": bootstrap,
        "m1_vs_random": comparisons["M1_vs_random"],
        "m2_vs_random": comparisons["M2_vs_random"], "m1_vs_m2": comparisons["M2_vs_M1"],
        "architecture_collapse": collapse, "auxiliary": auxiliary, "decision": decision,
    }


def write_search_outputs(output: Path, analysis: Mapping[str, Any]) -> None:
    dump_csv(output / "search/per_round_results.csv", analysis["per_round"], CSV_FIELDS["search/per_round_results.csv"])
    dump_csv(output / "search/per_replicate_results.csv", analysis["per_replicate"], CSV_FIELDS["search/per_replicate_results.csv"])
    dump_csv(output / "search/anytime_curves.csv", analysis["anytime"], CSV_FIELDS["search/anytime_curves.csv"])
    dump_json(output / "search/m1_vs_random.json", analysis["m1_vs_random"])
    dump_json(output / "search/m2_vs_random.json", analysis["m2_vs_random"])
    dump_json(output / "search/m1_vs_m2.json", analysis["m1_vs_m2"])
    dump_json(output / "search/bootstrap_results.json", analysis["bootstrap"])
    dump_json(output / "search/architecture_collapse_results.json", analysis["architecture_collapse"])


def _winner_csv_rows(winners: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "replicate_id": int(row["replicate_id"]),
        "joint_fingerprint": str(row["joint_fingerprint"]),
        "architecture_fingerprint": str(row["architecture_fingerprint"]),
        "validation_mean": float(row["validation_mean"]),
        "validation_sd": float(row["validation_sd"]),
        "m1_prediction": float(row["m1_prediction"]),
        "m2_prediction": float(row["m2_prediction"]),
    } for row in winners]


def select_top3_winners(winners: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        _winner_csv_rows(winners),
        key=lambda row: (-float(row["validation_mean"]), str(row["joint_fingerprint"])),
    )
    selected = []
    seen = set()
    for row in ordered:
        joint = str(row["joint_fingerprint"])
        if joint in seen:
            continue
        seen.add(joint)
        selected.append(row)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ProtocolError("fewer than three unique replicate winners are available")
    return selected


def write_next_stage(
    output: Path, classification: str, next_surrogate: str | None,
    m1_winners: Sequence[Mapping[str, Any]], m2_winners: Sequence[Mapping[str, Any]],
    random_winners: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    next_stage = output / "next_stage"
    next_stage.mkdir(parents=True, exist_ok=True)
    metric_method = next_surrogate or "M1"
    metric_winners = m2_winners if metric_method == "M2" else m1_winners
    metric_rows = _winner_csv_rows(metric_winners)
    random_rows = _winner_csv_rows(random_winners)
    dump_csv(next_stage / "metric_replicate_winners.csv", metric_rows, CSV_FIELDS["next_stage/metric_replicate_winners.csv"])
    dump_csv(next_stage / "random_replicate_winners.csv", random_rows, CSV_FIELDS["next_stage/random_replicate_winners.csv"])
    promoted_classes = {
        "M1_SEARCH_PROMOTED", "M2_SEARCH_PROMOTED",
        "M1_M2_SEARCH_PROMOTED_KNN_SELECTED", "M1_M2_SEARCH_PROMOTED_GP_SELECTED",
    }
    if classification not in promoted_classes:
        return {"promoted": False, "reported_metric_method": metric_method, "replicate_winner_tables_written": True, "fresh_seed_training_started": False}
    metric_top3 = select_top3_winners(metric_winners)
    random_top3 = select_top3_winners(random_winners)
    dump_json(next_stage / "metric_top3.json", {
        "method": metric_method, "selection_rule": "validation_mean_desc_then_joint_lex",
        "winners": metric_top3, "authorized_use": "fresh_seed_validation_confirmation_only",
    })
    dump_json(next_stage / "random_top3.json", {
        "method": "RANDOM", "selection_rule": "validation_mean_desc_then_joint_lex",
        "winners": random_top3, "authorized_use": "fresh_seed_validation_confirmation_only",
    })
    dump_json(next_stage / "historical_P_reference.json", {
        "role": "DESCRIPTIVE_HISTORICAL_REFERENCE",
        "joint_fingerprint": "30e0bb108c1fdbe8db186152a09c3f2fc6e812be189db920fe4d0d2b0d3cb1a2",
        "validation_seed_count": 30,
        "validation_mean": HISTORICAL_P_MEAN,
        "formal_inference_in_current_stage": False,
    })
    frozen_files = (
        "metric_replicate_winners.csv", "random_replicate_winners.csv",
        "metric_top3.json", "random_top3.json", "historical_P_reference.json",
    )
    entries = [{"relative_path": relative, "sha256": sha256_file(next_stage / relative)} for relative in frozen_files]
    root = stable_hash([(row["relative_path"], row["sha256"]) for row in entries])
    dump_text(next_stage / "winner_freeze_root_sha256.txt", root + "\n")
    dump_text(next_stage / "NEXT_STAGE_WINNERS_FROZEN", root + "\n")
    return {
        "promoted": True,
        "metric_method": metric_method,
        "root_sha256": root,
        "files": entries,
        "fresh_seed_training_started": False,
    }


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.9f}"
            elif isinstance(value, (list, Mapping)):
                value = json.dumps(value, sort_keys=True, separators=(",", ":"))
            values.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join((header, separator, *body))


def _render_final_report_legacy(artifacts: Mapping[str, Any]) -> str:
    decision = dict(artifacts.get("decision", {}))
    classification = str(decision.get("classification", "RUN_FAILED"))
    comparison = dict(artifacts.get("final_comparison", {}))
    per_replicate = list(artifacts.get("per_replicate", []))
    per_round = list(artifacts.get("per_round", []))
    anytime = list(artifacts.get("anytime", []))
    bootstrap = dict(artifacts.get("bootstrap", {}))
    sign = dict(artifacts.get("sign_test", {}))
    input_integrity = dict(artifacts.get("input_integrity", {}))
    replay = dict(artifacts.get("replay", {}))
    generation = dict(artifacts.get("generation", {}))
    information = dict(artifacts.get("information", {}))
    training = dict(artifacts.get("training_audit", {}))
    next_stage = dict(artifacts.get("next_stage", {}))
    detailed_rows = []
    for replicate in range(REPLICATES):
        endpoint = next((row for row in per_replicate if int(row["replicate_id"]) == replicate), {})
        for round_id in range(ROUNDS):
            for method in ("B4_GUIDED", "MODEL_BLIND_RANDOM_HIGH_REGIME"):
                row = next((item for item in per_round if int(item["replicate_id"]) == replicate and int(item["round_id"]) == round_id and item["method"] == method), None)
                if row:
                    detailed_rows.append({
                        "replicate": replicate,
                        "round": round_id + 1,
                        "method": method,
                        "selected actual 3-seed means": row["selected_validation_means"],
                        "best after budget": row["best_so_far"],
                        "final B4": endpoint.get("best_B4"),
                        "final Random": endpoint.get("best_random"),
                        "difference": endpoint.get("difference"),
                    })
    coverage = generation.get("coverage", {})
    gates = decision.get("gates", {})
    promoted = classification == "B4_SEARCH_PROMOTED"
    lines = [
        f"# B4-Guided Search Accuracy Gate v1 — {classification}", "",
        f"## {REPORT_SECTIONS[0]}", "",
        f"Frozen classification: `{classification}`. The primary endpoint is final-best 3-seed validation accuracy at an equal 16-configuration budget per method and search replicate.", "",
        f"## {REPORT_SECTIONS[1]}", "",
        "Both methods see the same 128 model-blind high-regime candidates per round and use the same four real evaluation slots. The only changed operation is pure frozen B4 ranking versus uniform random sampling, so the paired final-best endpoint directly tests search accuracy rather than predictor MAE.", "",
        f"## {REPORT_SECTIONS[2]}", "",
        "B4 supervision is restricted to the historical 336-config archive. Labels from the prior 8-pool, prior 16-pool, and current search are never added to B4. Historical P is descriptive only and is absent from S1–S5.", "",
        f"Input identity status: `{input_integrity.get('status', 'UNAVAILABLE')}`; test accessed: `{input_integrity.get('test_accessed', False)}`.", "",
        f"## {REPORT_SECTIONS[3]}", "",
        f"Replay status: `{replay.get('status', 'UNAVAILABLE')}`. Historical OOF rows: `{replay.get('historical_OOF_count', 0)}`; previous 16-pool predictions: `{replay.get('previous_16_pool_prediction_count', 0)}`; maximum absolute replay error: `{replay.get('maximum_absolute_error', 'N/A')}`.", "",
        f"## {REPORT_SECTIONS[4]}", "",
        "Eight independent replicate RNG streams use seeds 202608290001+r. Each stream advances in fixed round order through standard-normal deterministic-greedy VAE decodes and independent four-dimensional Uniform[0.80,0.95) HIGH_REGIME_PRIOR draws. Candidate generation receives no scorer, validation, metric, head, acquisition, mutation, incumbent, or historical-best signal.", "",
        f"## {REPORT_SECTIONS[5]}", "",
        f"Candidate count: `{coverage.get('candidate_count', 0)}`; unique joints: `{coverage.get('unique_joint_count', 0)}`; unique architectures: `{coverage.get('unique_architecture_count', 0)}`. Historical-seen architecture fraction: `{coverage.get('historical_seen_architecture_fraction', 'N/A')}`; previous-prospective-seen architecture fraction: `{coverage.get('previous_prospective_seen_architecture_fraction', 'N/A')}`; current repeated-architecture fraction: `{coverage.get('current_repeated_architecture_fraction', 'N/A')}`. Exact historical/previous/current joint uniqueness is fail-closed; architecture reuse is allowed and reported.", "",
        f"## {REPORT_SECTIONS[6]}", "",
        f"Pre-search root: `{artifacts.get('pre_search_root', 'NOT_FROZEN')}`. All candidates, B4 predictions, B4 selections, and Random selections were frozen before task creation or GNN execution: `{bool(artifacts.get('pre_search_root'))}`.", "",
        f"## {REPORT_SECTIONS[7]}", "",
        f"Method-equivalent budget is 256 config slots / 768 seed runs. Physical selected union is `{information.get('selected_union_unique_configurations', training.get('selected_unique_config_count', 0))}` configs / `{training.get('physical_unique_GNN_runs', information.get('physical_unique_GNN_runs', 0))}` seed runs; overlap is shared without changing either method's 16-config-per-replicate budget.", "",
        f"## {REPORT_SECTIONS[8]}", "",
        f"Evaluation complete: `{training.get('evaluation_complete', False)}`; planned: `{training.get('task_count', 0)}`; completed successful: `{training.get('completed_gnn_runs', 0)}`; failed/incomplete: `{training.get('failed_gnn_runs', 0)}`; loader firewall passed: `{training.get('loader_firewall_passed', False)}`; track_test: `{training.get('track_test', False)}`.", "",
        f"## {REPORT_SECTIONS[9]}", "",
        _markdown_table(per_replicate, ("replicate_id", "best_B4", "best_random", "difference", "B4_winner_joint_fingerprint", "random_winner_joint_fingerprint")), "",
        "Each replicate's four rounds and all four selected actual means per method are shown below.", "",
        _markdown_table(detailed_rows, ("replicate", "round", "method", "selected actual 3-seed means", "best after budget", "final B4", "final Random", "difference")), "",
        f"## {REPORT_SECTIONS[10]}", "",
        f"Mean paired difference: `{comparison.get('mean_difference', 'N/A')}`; median difference: `{comparison.get('median_difference', 'N/A')}`. These eight paired replicate differences are the primary data points.", "",
        f"## {REPORT_SECTIONS[11]}", "",
        f"Bootstrap unit: `{bootstrap.get('unit', 'search_replicate')}`; draws: `{bootstrap.get('bootstrap_replicates', 0)}`; seed: `{bootstrap.get('seed', BOOTSTRAP_SEED)}`; 95% percentile CI: `[{bootstrap.get('ci95_lower', 'N/A')}, {bootstrap.get('ci95_upper', 'N/A')}]`.", "",
        f"## {REPORT_SECTIONS[12]}", "",
        _markdown_table(anytime, ("replicate_id", "method", "budget", "best_so_far")), "",
        f"## {REPORT_SECTIONS[13]}", "",
        f"B4 mean AUC: `{comparison.get('B4_mean_anytime_AUC', 'N/A')}`; Random mean AUC: `{comparison.get('random_mean_anytime_AUC', 'N/A')}`; mean gain: `{comparison.get('mean_anytime_gain', 'N/A')}`; median gain: `{comparison.get('median_anytime_gain', 'N/A')}`. AUC is the arithmetic mean of best-after-4/8/12/16 on the accuracy scale.", "",
        f"## {REPORT_SECTIONS[14]}", "",
        f"Wins/losses/ties: `{comparison.get('wins', 0)}/{comparison.get('losses', 0)}/{comparison.get('ties', 0)}`. Exact one-sided sign-test p-value: `{sign.get('exact_one_sided_p_value', 'N/A')}`; this is auxiliary and does not replace S1–S5.", "",
        f"## {REPORT_SECTIONS[15]}", "",
        f"Historical P 30-seed validation mean is `{HISTORICAL_P_MEAN}`. B4 winners above P: `{comparison.get('historical_P', {}).get('B4_replicate_winners_above_P', 0)}/8`; Random winners above P: `{comparison.get('historical_P', {}).get('random_replicate_winners_above_P', 0)}/8`; highest B4/Random winner means: `{comparison.get('historical_P', {}).get('highest_B4_winner_3seed_mean', 'N/A')}` / `{comparison.get('historical_P', {}).get('highest_random_winner_3seed_mean', 'N/A')}`. These point estimates only nominate a later fresh-seed confirmation and are not formal comparisons with P.", "",
        f"## {REPORT_SECTIONS[16]}", "",
        f"Frozen result: `{classification}`. S1–S5: `{json.dumps(gates, sort_keys=True)}`. No gate was modified after labels appeared.", "",
        f"## {REPORT_SECTIONS[17]}", "",
        ("Search efficiency improved under the frozen gate because all S1–S5 passed." if promoted else "The preregistered evidence did not confirm improved B4 search efficiency."), "",
        f"## {REPORT_SECTIONS[18]}", "",
        ("This proves a paired improvement over the specified random high-regime search baseline at the current 3-seed search budget; it does not prove superiority to historical P or a population-level expected-accuracy claim." if promoted else "Absolute accuracy improvement has not been proven. Three-seed winners are noisy search estimates and historical P cannot rescue the classification."), "",
        f"## {REPORT_SECTIONS[19]}", "",
        f"Replicate winner tables were written: `{bool(artifacts.get('b4_winners') or artifacts.get('random_winners'))}`. Promoted top-3 freeze: `{next_stage.get('promoted', False)}`; freeze root: `{next_stage.get('root_sha256', 'NOT_CREATED')}`; fresh seeds started: `{next_stage.get('fresh_seed_training_started', False)}`.", "",
        f"## {REPORT_SECTIONS[20]}", "",
        "The endpoint uses only eight search replicates and 3-seed configuration estimates. Candidate deployment is restricted to the frozen VAE model-blind architecture distribution plus HIGH_REGIME_PRIOR, and the comparison does not estimate test accuracy or directly establish expected accuracy against historical P.", "",
        f"## {REPORT_SECTIONS[21]}", "",
        f"Protocol version: `{PROTOCOL_VERSION}`. Pre-search immutability verified before task creation/evaluation/analysis. Source bytes, imported repo-local modules, Git before/after status, diff, environment, manifest, archive, and SHA-256 sidecar are bundled. Test accessed: `false`; candidate replacement: `false`; online B4 update: `false`.", "",
    ]
    return "\n".join(lines)


def render_final_report(artifacts: Mapping[str, Any]) -> str:
    decision = dict(artifacts.get("decision", {}))
    classification = str(decision.get("classification", "RUN_FAILED"))
    per_replicate = list(artifacts.get("per_replicate", []))
    per_round = list(artifacts.get("per_round", []))
    anytime = list(artifacts.get("anytime", []))
    replay = dict(artifacts.get("replay", {}))
    generation = dict(artifacts.get("generation", {}))
    coverage = dict(generation.get("coverage", {}))
    information = dict(artifacts.get("information", {}))
    training = dict(artifacts.get("training_audit", {}))
    collapse = dict(artifacts.get("architecture_collapse", {})).get("methods", {})
    m1_vs_r = dict(artifacts.get("m1_vs_random", {}))
    m2_vs_r = dict(artifacts.get("m2_vs_random", {}))
    m1_vs_m2 = dict(artifacts.get("m1_vs_m2", {}))
    auxiliary = dict(artifacts.get("auxiliary", {}))
    next_stage = dict(artifacts.get("next_stage", {}))
    detailed_rows = [{
        "replicate": int(row["replicate_id"]), "round": int(row["round_id"]) + 1,
        "method": row["method"], "selected configs": row["selected_joint_fingerprints"],
        "actual 3-seed means": row["selected_validation_means"],
        "best after budget": row["best_so_far"],
    } for row in per_round]
    promoted = classification in {
        "M1_SEARCH_PROMOTED", "M2_SEARCH_PROMOTED",
        "M1_M2_SEARCH_PROMOTED_KNN_SELECTED", "M1_M2_SEARCH_PROMOTED_GP_SELECTED",
    }
    sections = {
        0: f"Frozen classification: `{classification}`. Primary endpoints are paired final-best 3-seed validation estimates at 16 configurations per method and replicate.",
        1: "M1, M2, and Random receive identical model-blind high-regime pools and equal real budgets. Only screening differs, so the experiment tests final search accuracy rather than predictor MAE.",
        2: "Metric supervision is historical-336 only. Prior prospective, B4-search, and current labels never train or update M1/M2. Historical P is descriptive only.",
        3: f"Replay status `{replay.get('status', 'NOT_RUN')}`; encoder count `{replay.get('model_count', 0)}`; folds×seeds `5×3`; required anchors include best epochs, M1/M2 OOF, Gap@5, Spearman, and all ordered top-5 neighbors.",
        4: f"Proposal `{generation.get('proposal_distribution', 'NOT_RUN')}`, master seed `{generation.get('candidate_master_seed', CANDIDATE_MASTER_SEED)}`; fixed non-adaptive replicate RNG streams; HIGH_REGIME_PRIOR Uniform[0.80,0.95].",
        5: f"Candidates/unique joints/unique architectures: `{coverage.get('candidate_count', 0)}` / `{coverage.get('unique_joint_count', 0)}` / `{coverage.get('unique_architecture_count', 0)}`. Historical, prior-8, prior-16, B4-evaluated, and current duplicate joints are excluded; architecture repetition is allowed and audited.",
        6: f"Pre-search root `{artifacts.get('pre_search_root', 'NOT_FROZEN')}`. All 4096 candidates, both score tables, three selections, and collapse audit were frozen before any GNN.",
        7: f"Method-equivalent budget: 384 configs/1152 runs. Physical union: `{training.get('selected_unique_config_count', information.get('selected_union_unique_configurations', 0))}` configs / `{training.get('physical_unique_GNN_runs', 0)}` runs; overlap is trained once and shared.",
        8: f"`{json.dumps(collapse.get('M1', {}).get('all_selected_slots', {}), sort_keys=True)}`",
        9: f"`{json.dumps(collapse.get('M2', {}).get('all_selected_slots', {}), sort_keys=True)}`",
        10: f"`{json.dumps(collapse.get('RANDOM', {}).get('all_selected_slots', {}), sort_keys=True)}`",
        11: f"Complete `{training.get('evaluation_complete', False)}`; completed/failed `{training.get('completed_gnn_runs', 0)}` / `{training.get('failed_gnn_runs', 0)}`; validation-only loader firewall `{training.get('loader_firewall_passed', False)}`; track_test `false`.",
        12: _markdown_table(per_replicate, ("replicate_id", "best_M1", "M1_winner_joint_fingerprint")),
        13: _markdown_table(per_replicate, ("replicate_id", "best_M2", "M2_winner_joint_fingerprint")),
        14: _markdown_table(per_replicate, ("replicate_id", "best_random", "random_winner_joint_fingerprint")),
        15: f"`{json.dumps(m1_vs_r, sort_keys=True)}`",
        16: f"`{json.dumps(m2_vs_r, sort_keys=True)}`",
        17: f"`{json.dumps(m1_vs_m2, sort_keys=True)}`\n\n`OBSERVED_EVALUATED_UNION_ORACLE_r = max(BEST_M1_r, BEST_M2_r, BEST_R_r)` covers only actually evaluated M1/M2/Random union configs, not the complete 128-candidate pool. Its regrets are only for surrogate selection and are not full-pool simple regret.",
        18: _markdown_table(detailed_rows, ("replicate", "round", "method", "selected configs", "actual 3-seed means", "best after budget")) + "\n\n" + _markdown_table(anytime, ("replicate_id", "method", "budget", "best_so_far")) + "\n\n" + _markdown_table(per_replicate, ("replicate_id", "best_M1", "best_M2", "best_random", "M1_minus_random", "M2_minus_random", "M2_minus_M1", "M1_anytime_AUC", "M2_anytime_AUC", "random_anytime_AUC")),
        19: f"Collapse was measured at all-slot, replicate, and round scopes before evaluation and never altered selection: `{json.dumps(collapse, sort_keys=True)}`.",
        20: f"`{json.dumps(auxiliary.get('historical_P', {}), sort_keys=True)}`. These point estimates cannot establish superiority to P.",
        21: f"`{classification}`. M1 gate `{json.dumps(decision.get('M1_gate', {}), sort_keys=True)}`; M2 gate `{json.dumps(decision.get('M2_gate', {}), sort_keys=True)}`.",
        22: f"NEXT_SURROGATE `{decision.get('NEXT_SURROGATE')}`. M1 is the parsimony default unless all preregistered GP-increment gates pass.",
        23: "At least one frozen metric method passed its complete final-best/anytime gate." if promoted else "Neither metric method passed the complete frozen search-accuracy gate.",
        24: "Stable absolute accuracy improvement is not proven by 3-seed search winners; fresh matched-seed confirmation is separate and was not started.",
        25: f"Replicate winner tables written; promoted top-3 freeze `{next_stage.get('promoted', False)}`; metric `{next_stage.get('metric_method', next_stage.get('reported_metric_method'))}`; root `{next_stage.get('root_sha256', 'NOT_CREATED')}`; fresh seeds `false`.",
        26: f"Protocol `{PROTOCOL_VERSION}`; immutable pre-search verified before tasks; source bytes/imported local modules, Git audit, manifest, archive, and SHA sidecar bundled; test accessed `false`; no replacement or online update.",
    }
    lines = [f"# M1/M2 Metric-Guided Search Accuracy Gate v1 — {classification}", ""]
    for index, title in enumerate(REPORT_SECTIONS):
        lines.extend((f"## {title}", "", sections[index], ""))
    return "\n".join(lines)


def assert_complete_final_report(text: str) -> None:
    missing = [section for section in REPORT_SECTIONS if f"## {section}" not in text]
    if missing:
        raise ProtocolError(f"final report is missing required sections: {missing}")


def _module_source_path(module: Any) -> Path | None:
    raw = getattr(module, "__file__", None)
    if not raw:
        return None
    path = Path(raw)
    if path.suffix in {".pyc", ".pyo"}:
        candidate = Path(importlib.util.source_from_cache(str(path)))
        if candidate.is_file():
            path = candidate
    return path.resolve() if path.is_file() else None


def snapshot_sources(output: Path, *, best_effort: bool = False) -> dict[str, Any]:
    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, Any]] = {}
    snapshot_errors: list[dict[str, str]] = []

    def copy_source(source: Path, relative: str, role: str) -> bool:
        if not source.is_file():
            message = f"source snapshot input is missing: {source}"
            if not best_effort:
                raise ProtocolError(message)
            snapshot_errors.append({"role": role, "source_path": str(source), "error": message})
            return False
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256_file(target) != sha256_file(source):
            message = f"source snapshot collision at {relative}"
            if not best_effort:
                raise ProtocolError(message)
            snapshot_errors.append({"role": role, "source_path": str(source), "error": message})
            return False
        if not target.exists():
            shutil.copy2(source, target)
        copied[relative] = {
            "relative_path": relative,
            "source_path": str(source.resolve()),
            "sha256": sha256_file(target),
            "role": role,
        }
        return True

    for relative, source in _source_identity_paths().items():
        copy_source(source, relative, "formal runtime source")
    # The evaluator dependency closure includes the architecture parser, HP
    # transform, model implementation, validation evaluator, and dataset loader.
    dependency_records = []
    try:
        frozen_dependencies = _frozen_dependency_entries()
    except Exception as error:
        if not best_effort:
            raise
        frozen_dependencies = []
        snapshot_errors.append({
            "role": "frozen high-regime evaluator dependency closure",
            "source_path": "UNAVAILABLE",
            "error": f"{type(error).__name__}: {error}",
        })
    for index, entry in enumerate(frozen_dependencies):
        source = Path(entry["path"]).resolve()
        try:
            repo_relative = source.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            repo_relative = f"external/{index:03d}-{source.name}"
        if "/source_snapshot/" in repo_relative:
            repo_relative = repo_relative.split("/source_snapshot/", 1)[1]
        relative = f"evaluator_dependencies/{repo_relative}"
        if copy_source(source, relative, "frozen high-regime evaluator dependency"):
            dependency_records.append({"source_path": str(source), "snapshot_relative_path": relative, "sha256": entry["sha256"]})
    imported = []
    used_import_targets: set[str] = set()
    for module_name, module in sorted(sys.modules.items()):
        source = _module_source_path(module)
        if source is None:
            continue
        try:
            source.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if source.suffix != ".py":
            continue
        base = module_name.replace(".", "/") or "__main__"
        relative = f"imported_local_modules/{base}.py"
        if relative in used_import_targets:
            relative = f"imported_local_modules/{base}-{sha256_file(source)[:12]}.py"
        used_import_targets.add(relative)
        if copy_source(source, relative, f"sys.modules:{module_name}"):
            imported.append({
                "module": module_name,
                "source_path": str(source),
                "snapshot_relative_path": relative,
                "sha256": sha256_file(source),
            })
    dump_json(snapshot / "imported_local_modules.json", imported)
    dump_json(snapshot / "manifest.json", {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "copied_source_count": len(copied),
        "mandatory_sources": list(copied.values()),
        "evaluator_dependency_count": len(dependency_records),
        "evaluator_dependencies": dependency_records,
        "imported_local_module_count": len(imported),
        "source_bytes_copied": not snapshot_errors,
        "best_effort": best_effort,
        "snapshot_error_count": len(snapshot_errors),
        "snapshot_errors": snapshot_errors,
    })
    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json" and path.suffix not in {".pyc", ".pyo"}:
            files.append({"relative_path": path.relative_to(snapshot).as_posix(), "sha256": sha256_file(path)})
    root = stable_hash([(row["relative_path"], row["sha256"]) for row in files])
    payload = {
        "format_version": PROTOCOL_VERSION,
        "files": files,
        "file_count": len(files),
        "manifest_root_sha256": root,
    }
    dump_json(snapshot / "sha256_manifest.json", payload)
    return payload


def write_environment_and_git(output: Path, *, before: bool) -> None:
    if before:
        environment = {
            "created_at": utc_now(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
            "repository": str(REPO_ROOT),
            "branch": _git_capture("branch", "--show-current").strip(),
            "HEAD": _git_capture("rev-parse", "HEAD").strip(),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "metric_scorer_device": "cpu",
            "test_accessed": False,
        }
        dump_json(output / "environment.json", environment)
        dump_text(output / "git_status_before.txt", _git_capture("status", "--short", "--branch"))
    else:
        dump_text(output / "git_status_after.txt", _git_capture("status", "--short", "--branch"))
        dump_text(output / "git_diff.patch", _git_diff_patch())


def _copy_preflight_logs(output: Path, preflight_log_dir: Path | None) -> None:
    expected = {
        "py_compile.log": "py_compile.log",
        "pytest_focused.log": "pytest_focused.log",
        "synthetic_smoke.log": "synthetic_smoke.log",
        "dry_run.log": "dry_run.log",
    }
    for source_name, target_name in expected.items():
        target = output / "logs" / target_name
        if preflight_log_dir is not None and (preflight_log_dir / source_name).is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preflight_log_dir / source_name, target)
        elif not target.exists():
            dump_text(target, "preflight log unavailable\n")


def _logger(path: Path) -> Callable[[str], None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now()}] {message}\n")

    return log


def ensure_output_policy(output: Path, archive: Path) -> None:
    if output.exists():
        raise ProtocolError(f"formal result path already exists; refusing overwrite: {output}")
    if archive.exists() or Path(str(archive) + ".sha256").exists():
        raise ProtocolError(f"formal archive target already exists; refusing overwrite: {archive}")
    try:
        output.resolve().relative_to((REPO_ROOT / "results/posthoc").resolve())
    except ValueError as error:
        raise ProtocolError("formal output must stay under results/posthoc") from error


def _placeholder_json(classification: str, reason: str | None = None) -> dict[str, Any]:
    payload = {"status": "UNAVAILABLE", "classification": classification, "test_accessed": False}
    if reason:
        payload["reason"] = reason
    return payload


def ensure_required_outputs(output: Path, classification: str, reason: str | None = None) -> None:
    for relative, fields in CSV_FIELDS.items():
        path = output / relative
        if not path.exists():
            dump_csv(path, [], fields)
    json_paths = [relative for relative in REQUIRED_OUTPUTS if relative.endswith(".json")]
    for relative in json_paths:
        path = output / relative
        if not path.exists():
            dump_json(path, _placeholder_json(classification, reason))
    text_paths = [relative for relative in REQUIRED_OUTPUTS if relative.endswith((".txt", ".patch", ".md", ".log"))]
    for relative in text_paths:
        path = output / relative
        if not path.exists():
            if relative == "pre_search/pre_search_root_sha256.txt":
                dump_text(path, "NOT_FROZEN\n")
            else:
                dump_text(path, f"Unavailable for {classification}: {reason or 'not reached'}\n")


def write_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and path.suffix not in {".pyc", ".pyo"}:
            files.append({
                "relative_path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "file_count": len(files),
        "files": files,
        "manifest_root_sha256": stable_hash([(row["relative_path"], row["sha256"]) for row in files]),
    }
    dump_json(output / "manifest.json", payload)
    return payload


def create_archive(output: Path, archive: Path) -> str:
    if archive.exists() or Path(str(archive) + ".sha256").exists():
        raise ProtocolError(f"refusing to overwrite archive target: {archive}")
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")

    def safe_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
            return None
        return info

    with tarfile.open(temporary, "w:gz") as handle:
        handle.add(output, arcname=output.name, recursive=True, filter=safe_filter)
    digest = sha256_file(temporary)
    os.replace(temporary, archive)
    dump_text(Path(str(archive) + ".sha256"), f"{digest}  {archive.name}\n")
    return digest


def finalize_bundle(
    output: Path, archive: Path, artifacts: dict[str, Any], classification: str,
    *, reason: str | None = None, run_failed: bool = False,
) -> None:
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise ProtocolError(f"illegal terminal classification: {classification}")
    decision = dict(artifacts.get("decision", {}))
    decision.update({
        "classification": classification,
        "reason": reason,
        "test_accessed": False,
        "gate_modified_after_results": False,
        "fresh_seed_confirmation_started": False,
    })
    artifacts["decision"] = decision
    dump_json(output / "decision.json", decision)
    analysed = {
        "METRIC_SEARCH_NOT_CONFIRMED", "M1_SEARCH_PROMOTED", "M2_SEARCH_PROMOTED",
        "M1_M2_SEARCH_PROMOTED_KNN_SELECTED", "M1_M2_SEARCH_PROMOTED_GP_SELECTED",
    }
    current_stage = "ANALYSIS_COMPLETE" if classification in analysed else json.loads((output / "state.json").read_text(encoding="utf-8")).get("stage")
    update_state(output, stage=current_stage, classification=classification, test_accessed=False)
    ensure_required_outputs(output, classification, reason)
    report = render_final_report(artifacts)
    assert_complete_final_report(report)
    dump_text(output / "final_report.md", report)
    # Snapshot after all scientific imports and before packaging.
    snapshot = snapshot_sources(
        output,
        best_effort=run_failed or classification in {"INPUT_IDENTITY_INVALID", "RUN_FAILED"},
    )
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    state["source_snapshot_manifest_root"] = snapshot["manifest_root_sha256"]
    state["updated_at"] = utc_now()
    dump_json(output / "state.json", state)
    write_environment_and_git(output, before=False)
    if run_failed:
        dump_text(output / "FAILED", classification + "\n")
        if not (output / "error.json").exists():
            dump_json(output / "error.json", {"classification": classification, "reason": reason, "test_accessed": False})
    else:
        dump_text(output / "COMPLETE", classification + "\n")
    write_manifest(output)
    digest = create_archive(output, archive)
    # Archive is external to the immutable result bundle, so the sidecar does
    # not mutate the packaged directory or its manifest.
    artifacts["archive_sha256"] = digest


def _synthetic_candidate(replicate: int, round_id: int, slot: int) -> dict[str, Any]:
    joint = hashlib.sha256(f"synthetic::{replicate}::{round_id}::{slot}".encode()).hexdigest()
    return {
        "replicate_id": replicate,
        "round_id": round_id,
        "candidate_slot": slot,
        "joint_fingerprint": joint,
        "architecture_fingerprint": f"arch-{slot % 17}",
        "proposal_distribution": "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR",
    }


def _synthetic_analysis_case_legacy(*, promoted: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    b4_rounds = []
    random_rounds = []
    stats = []
    for replicate in range(REPLICATES):
        for round_id in range(ROUNDS):
            b4_joints = [f"b4-{replicate}-{round_id}-{slot}" for slot in range(SELECTED_PER_ROUND)]
            random_joints = [f"random-{replicate}-{round_id}-{slot}" for slot in range(SELECTED_PER_ROUND)]
            b4_rounds.append({"replicate_id": replicate, "round_id": round_id, "selected_joint_fingerprints": b4_joints})
            random_rounds.append({"replicate_id": replicate, "round_id": round_id, "selected_joint_fingerprints": random_joints})
            for slot, joint in enumerate(b4_joints):
                value = 0.79 + 0.004 * round_id + 0.0001 * slot
                if not promoted:
                    value -= 0.02
                stats.append({
                    "joint_fingerprint": joint, "architecture_fingerprint": f"ab-{replicate}-{round_id}-{slot}",
                    "validation_mean": value, "validation_sd": 0.004, "b4_prediction": value + 0.001,
                    "selected_by_B4": True, "selected_by_random": False,
                })
            for slot, joint in enumerate(random_joints):
                value = 0.78 + 0.003 * round_id + 0.0001 * slot
                stats.append({
                    "joint_fingerprint": joint, "architecture_fingerprint": f"ar-{replicate}-{round_id}-{slot}",
                    "validation_mean": value, "validation_sd": 0.005, "b4_prediction": value - 0.001,
                    "selected_by_B4": False, "selected_by_random": True,
                })
    b4_plan = {"rounds": b4_rounds}
    random_plan = {"rounds": random_rounds}
    information = {
        "replicates": [{"replicate_id": replicate, "overlap_configurations": 0} for replicate in range(REPLICATES)],
        "selected_union_unique_configurations": 256,
        "physical_unique_GNN_runs": 768,
    }
    return b4_plan, random_plan, information, stats


def _synthetic_smoke_legacy(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    validate_frozen_protocol(load_config(config_path))
    formal_before = DEFAULT_OUTPUT.exists()
    b4_plan, random_plan, information, stats = _synthetic_analysis_case(promoted=True)
    training = {
        "selected_unique_config_count": 256,
        "physical_unique_GNN_runs": 768,
    }
    promoted = compute_search_analysis(stats, b4_plan, random_plan, information, training)
    b4_plan_bad, random_plan_bad, information_bad, stats_bad = _synthetic_analysis_case(promoted=False)
    not_confirmed = compute_search_analysis(stats_bad, b4_plan_bad, random_plan_bad, information_bad, training)
    candidates = [_synthetic_candidate(replicate, round_id, slot) for replicate in range(REPLICATES) for round_id in range(ROUNDS) for slot in range(CANDIDATES_PER_ROUND)]
    # Precompute the frozen random samples, then make those exact slots the B4
    # top four.  This exercises the no-selection-difference condition without
    # touching the formal generator.
    rng = np.random.default_rng(RANDOM_MASTER_SEED)
    top_slots: dict[tuple[int, int], set[int]] = {}
    for replicate in range(REPLICATES):
        for round_id in range(ROUNDS):
            top_slots[(replicate, round_id)] = set(map(int, rng.choice(CANDIDATES_PER_ROUND, 4, replace=False)))
    predictions = [{
        "joint_fingerprint": row["joint_fingerprint"],
        "b4_prediction": 1.0 if int(row["candidate_slot"]) in top_slots[(int(row["replicate_id"]), int(row["round_id"]))] else 0.0,
    } for row in candidates]
    no_diff_b4, no_diff_random, _selected, no_diff = select_search_plans(candidates, predictions)
    missing_raw = []
    selected_configs = []
    for index, row in enumerate(stats[:2]):
        selected_configs.append({**row, "selected_by_B4": row["selected_by_B4"], "selected_by_random": row["selected_by_random"]})
        for seed in TRAINING_SEEDS:
            if index == 1 and seed == 2:
                continue
            missing_raw.append({
                "joint_fingerprint": row["joint_fingerprint"], "training_seed": seed,
                "validation_accuracy": row["validation_mean"], "terminal_status": "success", "valid": True,
            })
    _incomplete_stats, incomplete = aggregate_config_statistics(selected_configs, missing_raw, stats, allow_incomplete=True)
    with tempfile.TemporaryDirectory(prefix="b4-search-smoke-") as temporary:
        root = Path(temporary)
        # A compact freeze exercise uses the full synthetic pool but no formal output.
        score_rows = [{
            "replicate_id": row["replicate_id"], "round_id": row["round_id"],
            "candidate_slot": row["candidate_slot"], "joint_fingerprint": row["joint_fingerprint"],
            "architecture_fingerprint": row["architecture_fingerprint"],
            "b4_prediction": next(item["b4_prediction"] for item in predictions if item["joint_fingerprint"] == row["joint_fingerprint"]),
        } for row in candidates]
        freeze = freeze_pre_search(
            root / "pre_search", candidates, score_rows, no_diff_b4, no_diff_random,
            {"status": "IDENTICAL"}, {"status": "SYNTHETIC"}, no_diff,
        )
        verified = verify_pre_search_freeze(root / "pre_search")
        next_stage = write_next_stage(
            root, "B4_SEARCH_PROMOTED", promoted["b4_winners"], promoted["random_winners"],
        )
        marker = (root / "next_stage/NEXT_STAGE_WINNERS_FROZEN").is_file()
    passed = all((
        promoted["decision"]["classification"] == "B4_SEARCH_PROMOTED",
        not_confirmed["decision"]["classification"] == "B4_SEARCH_NOT_CONFIRMED",
        no_diff["all_8_replicate_selection_sets_identical"] is True,
        incomplete is False,
        verified["root_sha256"] == freeze["root_sha256"],
        next_stage["promoted"] is True,
        marker,
        DEFAULT_OUTPUT.exists() == formal_before,
    ))
    return {
        "passed": passed,
        "promoted_case": promoted["decision"]["classification"],
        "not_confirmed_case": not_confirmed["decision"]["classification"],
        "no_selection_difference_detected": no_diff["all_8_replicate_selection_sets_identical"],
        "incomplete_case_failed_closed": not incomplete,
        "pre_search_root_recomputed": True,
        "promoted_winner_freeze_created": marker,
        "formal_candidate_generation_started": False,
        "formal_search_plan_frozen": False,
        "GNN_training_started": False,
        "formal_result_directory_created": DEFAULT_OUTPUT.exists() != formal_before,
        "test_accessed": False,
    }


def _synthetic_metric_case(case: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if case not in {"both_gp", "m1_only", "none", "both_zero"}:
        raise ValueError(case)
    offsets = {
        "both_gp": {"M1": 0.010, "M2": 0.014, "RANDOM": 0.0},
        "m1_only": {"M1": 0.010, "M2": 0.001, "RANDOM": 0.0},
        "none": {"M1": 0.001, "M2": -0.001, "RANDOM": 0.0},
        "both_zero": {"M1": 0.010, "M2": 0.010, "RANDOM": 0.0},
    }[case]
    rounds = {method: [] for method in ("M1", "M2", "RANDOM")}
    stats = []
    for replicate in range(REPLICATES):
        for round_id in range(ROUNDS):
            for method in rounds:
                joints = [f"{method.lower()}-{replicate}-{round_id}-{slot}" for slot in range(SELECTED_PER_ROUND)]
                rounds[method].append({"replicate_id": replicate, "round_id": round_id, "selected_joint_fingerprints": joints})
                for slot, joint in enumerate(joints):
                    value = 0.77 + 0.001 * replicate + 0.0002 * round_id + 0.00001 * slot + offsets[method]
                    stats.append({
                        "joint_fingerprint": joint, "architecture_fingerprint": f"arch-{method}-{slot}",
                        "validation_mean": value, "validation_sd": 0.004,
                        "m1_prediction": value + (0.001 if method == "M1" else 0.0),
                        "m2_prediction": value + (0.001 if method == "M2" else 0.0),
                        "selected_by_M1": method == "M1", "selected_by_M2": method == "M2",
                        "selected_by_random": method == "RANDOM",
                    })
    plans = {method: {"method": method, "rounds": rows} for method, rows in rounds.items()}
    information = {"replicates": [], "selected_union_unique_configurations": len(stats)}
    return plans["M1"], plans["M2"], plans["RANDOM"], information, stats


def synthetic_smoke(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    validate_frozen_protocol(load_config(config_path))
    formal_before = DEFAULT_OUTPUT.exists()
    training = {"selected_unique_config_count": 384, "physical_unique_GNN_runs": 1152}
    analyses = {}
    for case in ("both_gp", "m1_only", "none", "both_zero"):
        m1, m2, random_plan, information, stats = _synthetic_metric_case(case)
        analyses[case] = compute_search_analysis(
            stats, m1, m2, random_plan, information, training,
            {"selection_modified": False, "methods": {}},
        )
    candidates = [
        _synthetic_candidate(replicate, round_id, slot)
        for replicate in range(REPLICATES)
        for round_id in range(ROUNDS)
        for slot in range(CANDIDATES_PER_ROUND)
    ]
    predictions = [{
        "replicate_id": row["replicate_id"], "round_id": row["round_id"],
        "candidate_slot": row["candidate_slot"], "joint_fingerprint": row["joint_fingerprint"],
        "architecture_fingerprint": row["architecture_fingerprint"],
        "m1_prediction": float(row["candidate_slot"]),
        "m2_prediction": float(-row["candidate_slot"]),
    } for row in candidates]
    m1_plan, m2_plan, random_plan, _selected, information, collapse = select_search_plans(candidates, predictions)
    with tempfile.TemporaryDirectory(prefix="m1-m2-metric-search-smoke-") as temporary:
        root = Path(temporary)
        freeze = freeze_pre_search(
            root / "pre_search", candidates, predictions, m1_plan, m2_plan,
            random_plan, collapse, {"status": "IDENTICAL"}, {"status": "SYNTHETIC"}, information,
        )
        verified = verify_pre_search_freeze(root / "pre_search")
        configs, tasks = build_selected_union_tasks(
            candidates, m1_plan, m2_plan, random_plan, pre_search_verified=True,
        )
        raw = [{
            "joint_fingerprint": row["joint_fingerprint"], "training_seed": row["training_seed"],
            "validation_accuracy": 0.8, "terminal_status": "success", "valid": True,
        } for row in tasks]
        aggregate_config_statistics(configs, raw, predictions)
        incomplete = aggregate_config_statistics(configs, raw[:-1], predictions, allow_incomplete=True)[1]
        promoted = analyses["both_gp"]
        next_stage = write_next_stage(
            root, promoted["decision"]["classification"], promoted["decision"]["NEXT_SURROGATE"],
            promoted["m1_winners"], promoted["m2_winners"], promoted["random_winners"],
        )
        marker = (root / "next_stage/NEXT_STAGE_WINNERS_FROZEN").is_file()
    passed = all((
        analyses["both_gp"]["decision"]["classification"] == "M1_M2_SEARCH_PROMOTED_GP_SELECTED",
        analyses["m1_only"]["decision"]["classification"] == "M1_SEARCH_PROMOTED",
        analyses["none"]["decision"]["classification"] == "METRIC_SEARCH_NOT_CONFIRMED",
        analyses["both_zero"]["decision"]["classification"] == "M1_M2_SEARCH_PROMOTED_KNN_SELECTED",
        analyses["both_zero"]["decision"]["M1_vs_M2_surrogate_selection"]["gates"]["S2"]["passed"] is False,
        verified["root_sha256"] == freeze["root_sha256"], not incomplete,
        len(tasks) == 3 * len(configs), next_stage["promoted"] is True, marker,
        DEFAULT_OUTPUT.exists() == formal_before,
    ))
    return {
        "passed": passed,
        "both_promoted_GP_case": analyses["both_gp"]["decision"]["classification"],
        "M1_only_case": analyses["m1_only"]["decision"]["classification"],
        "not_confirmed_case": analyses["none"]["decision"]["classification"],
        "zero_regret_parsimony_case": analyses["both_zero"]["decision"]["classification"],
        "observed_union_oracle_rule_checked": True,
        "incomplete_case_failed_closed": not incomplete,
        "pre_search_root_recomputed": True, "promoted_winner_freeze_created": marker,
        "formal_candidate_generation_started": False, "formal_search_plan_frozen": False,
        "GNN_training_started": False, "formal_result_directory_created": DEFAULT_OUTPUT.exists() != formal_before,
        "test_accessed": False,
    }


def execute_dry_run(config_path: Path, formal_output: Path) -> int:
    existed_before = formal_output.exists()
    config = load_config(config_path)
    protocol = validate_frozen_protocol(config)
    static = static_input_audit(config)
    payload = {
        "status": "DRY_RUN_READY",
        "protocol": protocol,
        "input_file_identity": static["status"],
        "planned_candidate_count": TOTAL_CANDIDATES,
        "planned_M1_slots": SLOTS_PER_METHOD,
        "planned_M2_slots": SLOTS_PER_METHOD,
        "planned_random_slots": SLOTS_PER_METHOD,
        "maximum_physical_GNN_runs": 1152,
        "method_equivalent_GNN_runs": 1152,
        "metric_replay_started": False,
        "formal_candidate_generation_started": False,
        "formal_search_plan_frozen": False,
        "GNN_training_started": False,
        "formal_output_existed_before": existed_before,
        "formal_output_created": formal_output.exists() and not existed_before,
        "test_accessed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _terminal_without_winners(output: Path) -> None:
    dump_csv(output / "next_stage/metric_replicate_winners.csv", [], CSV_FIELDS["next_stage/metric_replicate_winners.csv"])
    dump_csv(output / "next_stage/random_replicate_winners.csv", [], CSV_FIELDS["next_stage/random_replicate_winners.csv"])


def execute_formal(
    config_path: Path, output: Path, config_sha256: str, source_sha256: str,
    preflight_log_dir: Path | None,
) -> int:
    archive = Path(str(output) + ".tar.gz")
    ensure_output_policy(output, archive)
    if not config_sha256 or sha256_file(config_path) != config_sha256:
        raise ProtocolError("runner-provided config SHA-256 does not match runtime bytes")
    if not source_sha256 or aggregate_source_sha256() != source_sha256:
        raise ProtocolError("runner-provided aggregate source SHA-256 does not match runtime bytes")
    config = load_config(config_path)
    validate_frozen_protocol(config)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("logs", "pre_search", "evaluation", "search", "next_stage"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    dump_json(output / "state.json", initial_state(config_sha256, source_sha256))
    dump_json(output / "frozen_protocol.json", config)
    _copy_preflight_logs(output, preflight_log_dir)
    write_environment_and_git(output, before=True)
    scorer_log = _logger(output / "logs/scorer_replay.log")
    candidate_log = _logger(output / "logs/candidate_generation.log")
    evaluation_log = _logger(output / "logs/evaluation.log")
    analysis_log = _logger(output / "logs/analysis.log")
    artifacts: dict[str, Any] = {
        "output": str(output),
        "decision": {},
        "test_accessed": False,
    }

    try:
        input_integrity = validate_authoritative_inputs(config)
        # Reuse the completed confirmation's authenticated upstream audit so
        # the soft-v2, pilot, split, checkpoint, and dependency closure are
        # checked with the same frozen identities.
        from analyse import joint_metric_independent_confirmation_surrogate_selection as confirmation
        upstream = confirmation.validate_authoritative_inputs(confirmation.load_config())
        input_integrity["upstream_confirmation_authority"] = {
            "status": upstream["status"],
            "historical_identity": upstream.get("historical_identity"),
            "previous_pilot_identity": upstream.get("previous_pilot_identity"),
            "previous_pilot_labels_used_for_training": False,
            "test_accessed": False,
        }
        dump_json(output / "input_integrity.json", input_integrity)
        artifacts["input_integrity"] = input_integrity
    except InputIdentityInvalid as error:
        input_integrity = {"status": "INVALID", "reason": str(error), "test_accessed": False}
        dump_json(output / "input_integrity.json", input_integrity)
        artifacts["input_integrity"] = input_integrity
        _terminal_without_winners(output)
        finalize_bundle(output, archive, artifacts, "INPUT_IDENTITY_INVALID", reason=str(error))
        return 0
    except Exception as error:
        # Upstream audit exception types live in archived modules and are
        # intentionally normalized into the input-identity classification.
        if type(error).__name__ in {"ProspectiveInputInvalid", "ConfirmationInputInvalid", "InputIdentityMismatch"}:
            input_integrity = {"status": "INVALID", "reason": str(error), "test_accessed": False}
            dump_json(output / "input_integrity.json", input_integrity)
            artifacts["input_integrity"] = input_integrity
            _terminal_without_winners(output)
            finalize_bundle(output, archive, artifacts, "INPUT_IDENTITY_INVALID", reason=str(error))
            return 0
        raise

    try:
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        runtime = load_frozen_runtime()
        historical_rows, retained_scorers, replay = replay_frozen_metric(runtime, scorer_log)
        dump_json(output / "metric_scorer_replay.json", replay)
        artifacts["replay"] = replay
        update_state(output, stage="SCORER_REPLAY_COMPLETE")
    except MetricScorerReplayMismatch as error:
        replay = {
            "status": "MISMATCH", "reason": str(error),
            "GNN_training_started": False, "test_accessed": False,
        }
        dump_json(output / "metric_scorer_replay.json", replay)
        artifacts["replay"] = replay
        _terminal_without_winners(output)
        finalize_bundle(output, archive, artifacts, "METRIC_SCORER_REPLAY_MISMATCH", reason=str(error))
        return 0
    except InputIdentityInvalid as error:
        _terminal_without_winners(output)
        finalize_bundle(output, archive, artifacts, "INPUT_IDENTITY_INVALID", reason=str(error))
        return 0

    try:
        previous = previous_candidate_identities()
        if (
            len(previous["pilot_joint"]) != 32
            or len(previous["confirmation_joint"]) != 64
            or len(previous["b4_search_joint"]) != 254
        ):
            raise InputIdentityInvalid("previous exact-joint exclusion cardinality changed")
        candidates, generation = generate_all_candidates(runtime, historical_rows, previous, candidate_log)
        artifacts["generation"] = generation
        update_state(output, stage="CANDIDATES_GENERATED", candidate_count=len(candidates))
    except CandidateGenerationFailed as error:
        dump_csv(output / "pre_search/candidate_manifest.csv", [], CSV_FIELDS["pre_search/candidate_manifest.csv"])
        artifacts["generation"] = {"status": "FAILED", "reason": str(error), "test_accessed": False}
        _terminal_without_winners(output)
        finalize_bundle(output, archive, artifacts, "CANDIDATE_GENERATION_FAILED", reason=str(error))
        return 0
    except InputIdentityInvalid as error:
        _terminal_without_winners(output)
        finalize_bundle(output, archive, artifacts, "INPUT_IDENTITY_INVALID", reason=str(error))
        return 0

    try:
        predictions, scoring_audit = score_all_candidates_with_metric(
            runtime, retained_scorers, candidates, scorer_log,
        )
    except MetricScorerReplayMismatch as error:
        replay = {**replay, "status": "MISMATCH", "candidate_scoring_reason": str(error)}
        dump_json(output / "metric_scorer_replay.json", replay)
        artifacts["replay"] = replay
        _terminal_without_winners(output)
        finalize_bundle(output, archive, artifacts, "METRIC_SCORER_REPLAY_MISMATCH", reason=str(error))
        return 0
    m1_plan, m2_plan, random_plan, selected, information, collapse = select_search_plans(
        candidates, predictions,
    )
    frozen_files = config["frozen_runtime"]["files"]
    scorer_identity = {
        "status": "IDENTICAL",
        "scorers": ["M1_WEIGHTED_KNN", "M2_ISOTROPIC_MATERN_5_2_GP_MEAN"],
        "replay": replay,
        "candidate_scoring": scoring_audit,
        "historical_source_hashes": {
            "high_regime_archive": config["authoritative_inputs"]["historical_high_regime_v1"]["archive_sha256"],
            "soft_v2_archive": config["authoritative_inputs"]["frozen_soft_metric_v2"]["archive_sha256"],
            "split_logical_sha256": config["authoritative_inputs"]["frozen_soft_metric_v2"]["split_logical_sha256"],
        },
        "metric_source_hashes": {
            key: frozen_files[key]
            for key in (
                "configs/joint_configuration_soft_metric_gate_v2.yaml",
                "analyse/joint_configuration_soft_metric_gate.py",
                "analyse/joint_configuration_metric_gate.py",
                "analyse/performance_metric_quick_gate.py",
            )
        },
        "previous_prospective_or_B4_labels_used": False,
        "current_labels_used": False,
        "all_predictions_before_first_GNN": True,
        "test_accessed": False,
    }
    generator_identity = {
        "status": "IDENTICAL",
        "candidate_master_seed": CANDIDATE_MASTER_SEED,
        "replicate_seeds": [CANDIDATE_MASTER_SEED + replicate for replicate in range(REPLICATES)],
        "random_master_seed": RANDOM_MASTER_SEED,
        "VAE_checkpoint_sha256": config["frozen_runtime"]["vae_checkpoint_sha256"],
        "high_regime_generator_sha256": frozen_files["analyse/high_regime_prior_model_blind_validation.py"],
        "previous_prospective_archive_hashes": {
            "8_pool": config["authoritative_inputs"]["previous_8_pool_prospective_v1"]["archive_sha256"],
            "16_pool": config["authoritative_inputs"]["previous_16_pool_confirmation_v1"]["archive_sha256"],
            "B4_search": config["authoritative_inputs"]["b4_search_v1"]["archive_sha256"],
        },
        "generation_audit": generation,
        "test_accessed": False,
    }
    candidate_hash_before = stable_hash(candidates)
    freeze = freeze_pre_search(
        output / "pre_search", candidates, predictions, m1_plan, m2_plan,
        random_plan, collapse, scorer_identity, generator_identity, information,
    )
    verified = verify_pre_search_freeze(output / "pre_search")
    assert_candidate_manifest_immutable(candidates, candidates)
    if stable_hash(candidates) != candidate_hash_before:
        raise PreSearchFreezeError("candidate records changed during pre-search freeze")
    artifacts.update({
        "information": information,
        "architecture_collapse": collapse,
        "pre_search_root": verified["root_sha256"],
        "freeze": freeze,
    })
    union_count = len(selected["M1"] | selected["M2"] | selected["RANDOM"])
    update_state(
        output,
        stage="SEARCH_PLAN_FROZEN",
        m1_selected_slots=SLOTS_PER_METHOD,
        m2_selected_slots=SLOTS_PER_METHOD,
        random_selected_slots=SLOTS_PER_METHOD,
        selected_unique_config_count=union_count,
        planned_gnn_runs=union_count * len(TRAINING_SEEDS),
    )
    del retained_scorers

    if information["all_metric_methods_equal_random"]:
        training_audit = {
            "status": "NOT_AUTHORIZED",
            "reason": "both metric methods equal Random in all eight replicate 16-config selection sets",
            "evaluation_complete": False,
            "selected_unique_config_count": 0,
            "physical_unique_GNN_runs": 0,
            "method_equivalent_GNN_runs": 1152,
            "task_count": 0,
            "completed_gnn_runs": 0,
            "failed_gnn_runs": 0,
            "track_test": False,
            "test_accessed": False,
        }
        dump_json(output / "evaluation/task_manifest.json", {
            "authorized": False, "reason": training_audit["reason"], "tasks": [],
            "planned_GNN_runs": 0, "track_test": False, "test_accessed": False,
        })
        dump_json(output / "evaluation/training_audit.json", training_audit)
        artifacts["training_audit"] = training_audit
        _terminal_without_winners(output)
        finalize_bundle(output, archive, artifacts, "SEARCH_NO_SELECTION_DIFFERENCE", reason=training_audit["reason"])
        return 0

    verify_pre_search_freeze(output / "pre_search")
    if stable_hash(candidates) != candidate_hash_before:
        raise PreSearchFreezeError("candidate manifest changed between freeze and task creation")
    selected_configs, tasks = build_selected_union_tasks(
        candidates, m1_plan, m2_plan, random_plan, pre_search_verified=True,
    )
    update_state(
        output, stage="EVALUATION_RUNNING",
        selected_unique_config_count=len(selected_configs), planned_gnn_runs=len(tasks),
        completed_gnn_runs=0, failed_gnn_runs=0,
    )
    raw, training_audit = run_frozen_gnn_evaluation(
        output, tasks, config, verified["root_sha256"], evaluation_log,
    )
    raw_csv = [{key: row.get(key) for key in CSV_FIELDS["evaluation/raw_seed_results.csv"]} for row in raw]
    dump_csv(output / "evaluation/raw_seed_results.csv", raw_csv, CSV_FIELDS["evaluation/raw_seed_results.csv"])
    config_stats, complete = aggregate_config_statistics(
        selected_configs, raw, predictions, allow_incomplete=True,
    )
    complete = bool(
        complete
        and training_audit["all_shards_exited_zero"]
        and training_audit["loader_firewall_passed"]
    )
    dump_csv(output / "evaluation/config_3seed_statistics.csv", config_stats, CSV_FIELDS["evaluation/config_3seed_statistics.csv"])
    successful_count = sum(row.get("terminal_status") == "success" and row.get("valid") is True for row in raw)
    training_audit.update({
        "evaluation_complete": complete,
        "completed_gnn_runs": successful_count,
        "failed_gnn_runs": len(tasks) - successful_count,
        "task_count": len(tasks),
        "track_test": False,
        "test_accessed": False,
        "candidate_replacement": False,
        "retry_changed_config_or_seed": False,
        "fresh_seed_confirmation_started": False,
    })
    dump_json(output / "evaluation/training_audit.json", training_audit)
    artifacts["training_audit"] = training_audit
    update_state(
        output,
        stage="EVALUATION_COMPLETE" if complete else "EVALUATION_RUNNING",
        completed_gnn_runs=successful_count,
        failed_gnn_runs=len(tasks) - successful_count,
        current_replicate=None, current_round=None, current_config=None, current_seed=None,
    )
    if not complete:
        _terminal_without_winners(output)
        finalize_bundle(
            output, archive, artifacts, "SEARCH_EVALUATION_INCOMPLETE",
            reason="at least one selected config lacks a successful result for seed 0, 1, or 2",
        )
        return 0

    verify_pre_search_freeze(output / "pre_search")
    analysis = compute_search_analysis(
        config_stats, m1_plan, m2_plan, random_plan, information, training_audit, collapse,
    )
    write_search_outputs(output, analysis)
    classification = str(analysis["decision"]["classification"])
    next_stage = write_next_stage(
        output, classification, analysis["decision"]["NEXT_SURROGATE"],
        analysis["m1_winners"], analysis["m2_winners"], analysis["random_winners"],
    )
    dump_json(output / "decision.json", analysis["decision"])
    artifacts.update(analysis)
    artifacts["next_stage"] = next_stage
    analysis_log(f"frozen classification: {classification}")
    update_state(output, stage="ANALYSIS_COMPLETE", classification=classification)
    finalize_bundle(output, archive, artifacts, classification)
    return 0


def write_failure_bundle(
    output: Path, message: str, *, config_path: Path = DEFAULT_CONFIG,
    config_sha256: str = "UNAVAILABLE", source_sha256: str = "UNAVAILABLE",
) -> None:
    archive = Path(str(output) + ".tar.gz")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    for directory in ("logs", "pre_search", "evaluation", "search", "next_stage"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    if not (output / "state.json").is_file():
        dump_json(output / "state.json", initial_state(config_sha256, source_sha256))
    if not (output / "frozen_protocol.json").is_file():
        try:
            dump_json(output / "frozen_protocol.json", load_config(config_path))
        except Exception:
            dump_json(output / "frozen_protocol.json", _placeholder_json("RUN_FAILED", message))
    if not (output / "environment.json").is_file():
        write_environment_and_git(output, before=True)
    dump_json(output / "error.json", {
        "classification": "RUN_FAILED",
        "message": message,
        "traceback": traceback.format_exc(),
        "test_accessed": False,
    })
    _terminal_without_winners(output)
    artifacts = {
        "decision": {"classification": "RUN_FAILED"},
        "input_integrity": _placeholder_json("RUN_FAILED", message),
        "replay": _placeholder_json("RUN_FAILED", message),
        "generation": _placeholder_json("RUN_FAILED", message),
        "training_audit": _placeholder_json("RUN_FAILED", message),
    }
    finalize_bundle(output, archive, artifacts, "RUN_FAILED", reason=message, run_failed=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--source-sha256", default="")
    parser.add_argument("--preflight-log-dir", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--shard", type=int, choices=(0, 1))
    parser.add_argument("--shard-output", type=Path)
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument("--device", default="cuda:0")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--synthetic-smoke", action="store_true")
    modes.add_argument("--formal", action="store_true")
    modes.add_argument("--source-identity", action="store_true")
    modes.add_argument("--evaluate-shard", action="store_true")
    modes.add_argument("--write-failure-bundle", metavar="MESSAGE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source_identity:
        print(aggregate_source_sha256())
        return 0
    if args.dry_run:
        return execute_dry_run(args.config, args.output)
    if args.synthetic_smoke:
        result = synthetic_smoke(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.evaluate_shard:
        if args.adapter is None or args.shard is None or args.shard_output is None:
            raise ProtocolError("--evaluate-shard requires --adapter, --shard, and --shard-output")
        return execute_evaluator_shard(args.adapter, args.shard, args.shard_output, args.cora_root, args.device)
    if args.write_failure_bundle is not None:
        write_failure_bundle(
            args.output, args.write_failure_bundle, config_path=args.config,
            config_sha256=args.config_sha256 or "UNAVAILABLE",
            source_sha256=args.source_sha256 or "UNAVAILABLE",
        )
        return 0
    try:
        return execute_formal(
            args.config, args.output, args.config_sha256, args.source_sha256,
            args.preflight_log_dir,
        )
    except Exception as error:
        # If formal initialization did not create the output, preserve the
        # overwrite error rather than mutating a pre-existing target.
        if not args.output.exists():
            raise
        try:
            write_failure_bundle(
                args.output, f"{type(error).__name__}: {error}",
                config_path=args.config,
                config_sha256=args.config_sha256 or "UNAVAILABLE",
                source_sha256=args.source_sha256 or "UNAVAILABLE",
            )
        except Exception as bundle_error:
            print(f"failure bundle creation also failed: {bundle_error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
