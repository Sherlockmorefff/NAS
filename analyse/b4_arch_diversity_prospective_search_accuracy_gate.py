#!/usr/bin/env python3
"""Prospective B4 architecture-diversity search accuracy gate v1.

Formal mode authenticates the post-hoc authorization and the previous B4
experiment, deterministically reconstructs the frozen historical-only B4
ensemble, freezes new model-blind candidate pools and three selection plans,
and only then evaluates their de-duplicated union.  Dry-run and synthetic
smoke never generate formal candidates, freeze a plan, initialize CUDA, or
train a GNN.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from analyse import b4_guided_search_accuracy_gate as base


PROTOCOL_VERSION = "b4_arch_diversity_prospective_search_accuracy_gate_global4_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/b4_arch_diversity_prospective_search_accuracy_gate_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/posthoc/b4-arch-diversity-prospective-search-accuracy-gate-global4-v1"
DEFAULT_ARCHIVE = Path(str(DEFAULT_OUTPUT) + ".tar.gz")

DIAGNOSIS_ROOT = REPO_ROOT / "results/posthoc/cross-method-search-collapse-diagnosis-global4-v1"
OLD_B4_ROOT = REPO_ROOT / "results/posthoc/b4-guided-search-accuracy-gate-global4-v1"
OLD_METRIC_ROOT = REPO_ROOT / "results/posthoc/m1-m2-metric-guided-search-accuracy-gate-global4-v1"

IMPLEMENTATION_FILES = (
    "configs/b4_arch_diversity_prospective_search_accuracy_gate_v1.yaml",
    "analyse/b4_arch_diversity_prospective_search_accuracy_gate.py",
    "scripts/run_b4_arch_diversity_prospective_search_accuracy_gate.sh",
    "tests/test_b4_arch_diversity_prospective_search_accuracy_gate.py",
)

REPLICATES = 12
ROUNDS = 4
CANDIDATES_PER_ROUND = 128
SELECTED_PER_ROUND = 4
TOTAL_CANDIDATES = REPLICATES * ROUNDS * CANDIDATES_PER_ROUND
SLOTS_PER_METHOD = REPLICATES * ROUNDS * SELECTED_PER_ROUND
TRAINING_SEEDS = (0, 1, 2)
B4_MODEL_SEEDS = (0, 1, 2)
CANDIDATE_MASTER_SEED = 202608310001
RANDOM_MASTER_SEED = 202608310002
BOOTSTRAP_REPLICATES = 10_000
REPLAY_ATOL = 1.0e-8
TOLERANCE = 1.0e-12
MINIMUM_GAIN = 0.003
DECODE_BATCH_SIZE = 256
MAX_DRAWS_PER_REPLICATE = 262_144
HISTORICAL_P_MEAN = 0.8087333333333333
HISTORICAL_P_JOINT = "30e0bb108c1fdbe8db186152a09c3f2fc6e812be189db920fe4d0d2b0d3cb1a2"
HISTORICAL_P_SOURCE = REPO_ROOT / "results/search/high-regime-prior-model-blind-validation-global4-v1/stabilization/F_final_candidate.json"
HISTORICAL_P_SOURCE_SHA256 = "d00dd5e43065dae6cac95b5b44d25e8d81da80a3f2b4875d876c6a98d6ec0f5d"

PURE = "PURE_B4_TOP4"
P2 = "B4_P2_DIVERSITY"
RANDOM = "MODEL_BLIND_RANDOM_HIGH_REGIME"
METHODS = (PURE, P2, RANDOM)

ALLOWED_CLASSIFICATIONS = frozenset({
    "INPUT_IDENTITY_INVALID",
    "B4_SCORER_REPLAY_MISMATCH",
    "CANDIDATE_GENERATION_FAILED",
    "P2_POLICY_INFEASIBLE",
    "DIVERSITY_INTERVENTION_LOW_INFORMATION",
    "DIVERSITY_EVALUATION_INCOMPLETE",
    "DIVERSITY_SEARCH_NOT_CONFIRMED",
    "DIVERSITY_MECHANISM_CONFIRMED_NO_SEARCH_PROMOTION",
    "DIVERSITY_SEARCH_PROMOTED",
    "RUN_FAILED",
})

REPORT_SECTIONS = (
    "1. Executive conclusion",
    "2. Why this is prospective and confirmatory",
    "3. Historical diagnosis boundary",
    "4. Frozen B4 identity",
    "5. Frozen P2 intervention",
    "6. Candidate-generation protocol",
    "7. Candidate leakage exclusion",
    "8. Candidate-pool architecture richness",
    "9. Pre-search immutable freeze",
    "10. P2 feasibility",
    "11. Intervention information gate",
    "12. Physical vs method-equivalent budget",
    "13. GNN evaluation completeness",
    "14. PURE B4 architecture allocation",
    "15. P2 architecture allocation",
    "16. Random architecture allocation",
    "17. Manipulation check",
    "18. Per-replicate final best",
    "19. P2 vs PURE B4",
    "20. P2 vs Random",
    "21. PURE B4 vs Random",
    "22. Bootstrap results",
    "23. Anytime curves",
    "24. Anytime AUC",
    "25. Repeat-refinement behavior",
    "26. Historical P descriptive comparison",
    "27. Frozen A gate",
    "28. Frozen B gate",
    "29. Final classification",
    "30. Whether diversity mechanism was confirmed",
    "31. Whether search accuracy promotion was confirmed",
    "32. Whether stable absolute accuracy improvement has been proven",
    "33. Next-stage frozen winners",
    "34. Limitations",
    "35. Engineering audit",
)

PRE_SEARCH_FILES = (
    "candidate_manifest.csv",
    "b4_predictions.csv",
    "pure_b4_selections.json",
    "p2_selections.json",
    "random_selections.json",
    "selection_overlap.json",
    "diversity_manipulation.json",
    "search_plan.json",
    "scorer_identity.json",
    "generator_identity.json",
    "exclusion_manifest.json",
)

REQUIRED_OUTPUTS = (
    "state.json", "frozen_protocol.json", "input_integrity.json",
    "b4_scorer_replay.json", "authorization_replay.json",
    "pre_search/candidate_manifest.csv", "pre_search/b4_predictions.csv",
    "pre_search/pure_b4_selections.json", "pre_search/p2_selections.json",
    "pre_search/random_selections.json", "pre_search/selection_overlap.json",
    "pre_search/diversity_manipulation.json", "pre_search/search_plan.json",
    "pre_search/scorer_identity.json", "pre_search/generator_identity.json",
    "pre_search/exclusion_manifest.json", "pre_search/freeze_manifest.json",
    "pre_search/pre_search_root_sha256.txt", "pre_search/PRE_SEARCH_FROZEN",
    "evaluation/task_manifest.json", "evaluation/raw_seed_results.csv",
    "evaluation/config_3seed_statistics.csv", "evaluation/training_audit.json",
    "search/per_round_results.csv", "search/per_replicate_results.csv",
    "search/anytime_curves.csv", "search/p2_vs_pure.json",
    "search/p2_vs_random.json", "search/pure_vs_random.json",
    "search/bootstrap_results.json", "search/observed_union_regret.json",
    "search/architecture_diversity_results.json", "search/repeat_refinement.json",
    "decision.json", "final_report.md", "environment.json",
    "git_status_before.txt", "git_status_after.txt", "git_diff.patch",
    "source_snapshot/imported_local_modules.json",
    "source_snapshot/sha256_manifest.json", "manifest.json",
    "logs/py_compile.log", "logs/pytest_focused.log",
    "logs/synthetic_smoke.log", "logs/scorer_replay.log",
    "logs/candidate_generation.log", "logs/evaluation.log", "logs/analysis.log",
)

CANDIDATE_FIELDS = (
    "replicate_id", "round_id", "candidate_slot", "joint_fingerprint",
    "architecture_fingerprint", "hp_fingerprint", "raw_hp_vector", "raw_hp",
    "operations", "edges", "effective_layers", "lr", "dropout", "hidden_dim",
    "l2", "gat_heads", "sage_aggr", "gin_eps", "gat_heads_by_layer",
    "sage_aggr_by_layer", "gin_eps_by_layer", "generation_seed",
    "generation_draw_index", "proposal_distribution", "b4_prediction",
    "pure_b4_selected", "p2_selected", "random_selected",
)


class ProtocolError(RuntimeError):
    """A frozen protocol invariant was violated."""


class InputIdentityInvalid(ProtocolError):
    """An authoritative input differs from its preregistered identity."""


class B4ScorerReplayMismatch(ProtocolError):
    """The frozen B4 identity could not be replayed exactly."""


class CandidateGenerationFailed(ProtocolError):
    """The fixed candidate generator could not fill the frozen grid."""


class P2PolicyInfeasible(ProtocolError):
    """P2 could not select four legal candidates in every round."""


class LowInformation(ProtocolError):
    """The frozen P2 allocation does not diverge enough from pure B4."""


class EvaluationIncomplete(ProtocolError):
    """A selected configuration lacks a complete seed triplet."""


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
    temporary.write_bytes(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    )
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


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = tuple(dict.fromkeys(key for row in rows for key in row)) or ("status",)
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


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ProtocolError("PyYAML is required") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("protocol config must be a mapping")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ProtocolError(f"frozen protocol mismatch for {label}: {actual!r} != {expected!r}")


def validate_frozen_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    generation = config["candidate_generation"]
    search = config["search"]
    evaluation = config["evaluation"]
    analysis = config["analysis"]
    _require_equal(config["protocol_version"], PROTOCOL_VERSION, "protocol version")
    _require_equal(generation["candidate_master_seed"], CANDIDATE_MASTER_SEED, "candidate seed")
    _require_equal(generation["replicates"], REPLICATES, "replicates")
    _require_equal(generation["rounds_per_replicate"], ROUNDS, "rounds")
    _require_equal(generation["candidates_per_round"], CANDIDATES_PER_ROUND, "pool size")
    _require_equal(generation["total_candidates"], TOTAL_CANDIDATES, "candidate total")
    _require_equal(generation["hp_raw_unit_interval_range"], [0.80, 0.95], "HIGH_REGIME_PRIOR")
    _require_equal(search["methods"], list(METHODS), "methods")
    _require_equal(search["random_master_seed"], RANDOM_MASTER_SEED, "random seed")
    _require_equal(search["selected_slots_per_method"], SLOTS_PER_METHOD, "method slots")
    _require_equal(evaluation["training_seeds"], list(TRAINING_SEEDS), "training seeds")
    _require_equal(evaluation["method_equivalent_config_slots"], 576, "method-equivalent configs")
    _require_equal(evaluation["method_equivalent_GNN_runs"], 1728, "method-equivalent runs")
    _require_equal(evaluation["track_test"], False, "track_test")
    _require_equal(config["p2_policy"]["name"], "ROUND_UNIQUE_CAP2", "P2")
    _require_equal(config["p2_policy"]["maximum_selected_slots_per_architecture_per_replicate"], 2, "P2 cap")
    _require_equal(analysis["bootstrap_replicates"], BOOTSTRAP_REPLICATES, "bootstrap draws")
    _require_equal(analysis["A_p2_vs_pure"]["bootstrap_seed"], 0, "A bootstrap seed")
    _require_equal(analysis["B_p2_vs_random"]["bootstrap_seed"], 1, "B bootstrap seed")
    _require_equal(config["next_stage"]["fresh_training_seeds"], {"start": 1000, "stop_inclusive": 1029}, "fresh seeds")
    _require_equal(set(config["allowed_classifications"]), set(ALLOWED_CLASSIFICATIONS), "classifications")
    return {"status": "VALID", "protocol_version": PROTOCOL_VERSION, "test_accessed": False}


def _git_capture(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout


def _git_diff_patch() -> str:
    pieces = [_git_capture("diff", "HEAD", "--binary", "--", ".").rstrip()]
    for relative in IMPLEMENTATION_FILES:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative], cwd=REPO_ROOT,
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        path = REPO_ROOT / relative
        if tracked.returncode == 0 or not path.is_file():
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative],
            cwd=REPO_ROOT, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode not in {0, 1}:
            raise ProtocolError(f"could not snapshot untracked source: {relative}")
        pieces.append(result.stdout.rstrip())
    return "\n".join(piece for piece in pieces if piece) + "\n"


def initial_state(config_sha256: str, source_sha256: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION, "stage": "IMPLEMENTATION_READY",
        "classification": None, "candidate_count": 0, "pure_b4_slots": 0,
        "p2_slots": 0, "random_slots": 0, "selected_unique_config_count": 0,
        "planned_gnn_runs": 0, "completed_gnn_runs": 0, "failed_gnn_runs": 0,
        "current_replicate": None, "current_round": None, "current_config": None,
        "current_seed": None, "test_accessed": False,
        "config_sha256": config_sha256, "source_sha256": source_sha256,
        "created_at": utc_now(), "updated_at": utc_now(),
    }


def update_state(output: Path, **changes: Any) -> dict[str, Any]:
    state_path = output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if changes.get("test_accessed") not in {None, False}:
        raise ProtocolError("test_accessed can never become true")
    state.update(changes)
    state["updated_at"] = utc_now()
    dump_json(state_path, state)
    return state


def _require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise InputIdentityInvalid(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise InputIdentityInvalid(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return {"label": label, "path": str(path), "sha256": observed}


def _verify_archive(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"label": label, "present": False, "fallback": "directory_manifest_and_internal_hashes"}
    identity = _require_hash(path, expected, label)
    sidecar = Path(str(path) + ".sha256")
    if sidecar.is_file():
        fields = sidecar.read_text(encoding="utf-8").split()
        if not fields or fields[0] != expected:
            raise InputIdentityInvalid(f"{label} sidecar mismatch: {sidecar}")
    return {**identity, "present": True, "sidecar_valid_if_present": True}


def verify_directory_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise InputIdentityInvalid(f"missing directory manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for entry in manifest.get("files", []):
        relative = str(entry["relative_path"])
        path = root / relative
        if not path.is_file():
            raise InputIdentityInvalid(f"manifest entry is missing: {path}")
        digest = sha256_file(path)
        if digest != entry.get("sha256") or path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise InputIdentityInvalid(f"manifest entry mismatch: {path}")
        pairs.append((relative, digest))
    observed_root = stable_hash(pairs)
    if observed_root != manifest.get("manifest_root_sha256"):
        raise InputIdentityInvalid(f"directory manifest root mismatch: {root}")
    return {
        "path": str(manifest_path), "file_count": len(pairs),
        "manifest_root_sha256": observed_root, "all_file_hashes_valid": True,
    }


def verify_historical_pre_search(root: Path) -> dict[str, Any]:
    pre = root / "pre_search"
    try:
        freeze = json.loads((pre / "freeze_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise InputIdentityInvalid(f"cannot read historical pre-search freeze: {root}") from error
    pairs = []
    for entry in freeze.get("files", []):
        relative = str(entry["relative_path"])
        path = pre / relative
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise InputIdentityInvalid(f"historical frozen file mismatch: {path}")
        pairs.append((relative, str(entry["sha256"])))
    observed = stable_hash(pairs)
    markers = (
        freeze.get("root_sha256"),
        (pre / "pre_search_root_sha256.txt").read_text(encoding="utf-8").strip(),
        (pre / "PRE_SEARCH_FROZEN").read_text(encoding="utf-8").strip(),
    )
    if not pairs or any(value != observed for value in markers):
        raise InputIdentityInvalid(f"historical pre-search root mismatch: {root}")
    return {"root_sha256": observed, "file_count": len(pairs)}


def static_input_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    authority = config["authoritative_inputs"]
    diagnosis = authority["collapse_diagnosis"]
    old_b4 = authority["previous_b4_search"]
    old_metric = authority["previous_metric_search"]
    pilot = authority["previous_8_pool"]
    confirmation = authority["previous_16_pool"]
    frozen_b4 = authority["frozen_b4_protocol"]
    checks = [
        _require_hash(DIAGNOSIS_ROOT / "manifest.json", diagnosis["manifest_sha256"], "diagnosis manifest"),
        _require_hash(DIAGNOSIS_ROOT / "decision.json", diagnosis["decision_sha256"], "diagnosis decision"),
        _require_hash(
            DIAGNOSIS_ROOT / "next_experiment/prospective_diversity_test_spec.json",
            diagnosis["prospective_spec_sha256"], "diagnosis prospective specification",
        ),
        _require_hash(OLD_B4_ROOT / "manifest.json", old_b4["manifest_sha256"], "old B4 manifest"),
        _require_hash(
            OLD_B4_ROOT / "pre_search/candidate_manifest.csv",
            old_b4["candidate_manifest_sha256"], "old B4 candidate manifest",
        ),
        _require_hash(
            OLD_B4_ROOT / "pre_search/b4_predictions.csv",
            old_b4["predictions_sha256"], "old B4 frozen predictions",
        ),
        _require_hash(
            OLD_METRIC_ROOT / "pre_search/candidate_manifest.csv",
            old_metric["candidate_manifest_sha256"], "old M1/M2 candidate manifest",
        ),
        _require_hash(
            REPO_ROOT / "results/posthoc/joint-metric-prospective-selection-gate-global4-v1-rerun1/pre_eval/candidate_manifest.json",
            pilot["candidate_manifest_sha256"], "8-pool candidate manifest",
        ),
        _require_hash(
            REPO_ROOT / "results/posthoc/joint-metric-independent-confirmation-surrogate-selection-global4-v1-rerun1/pre_eval/candidate_manifest.json",
            confirmation["candidate_manifest_sha256"], "16-pool candidate manifest",
        ),
        _require_hash(REPO_ROOT / frozen_b4["config"], frozen_b4["config_sha256"], "frozen B4 search config"),
        _require_hash(REPO_ROOT / frozen_b4["source"], frozen_b4["source_sha256"], "frozen B4 search source"),
        _require_hash(HISTORICAL_P_SOURCE, HISTORICAL_P_SOURCE_SHA256, "historical P reference"),
    ]
    checks.append(_require_hash(Path(base.__file__).resolve(), frozen_b4["source_sha256"], "loaded B4 search source"))
    archive_checks = [
        _verify_archive(REPO_ROOT / diagnosis["archive"], diagnosis["archive_sha256"], "diagnosis archive"),
        _verify_archive(REPO_ROOT / old_b4["archive"], old_b4["archive_sha256"], "old B4 archive"),
        _verify_archive(REPO_ROOT / old_metric["archive"], old_metric["archive_sha256"], "old M1/M2 archive"),
    ]
    for root, label in ((DIAGNOSIS_ROOT, "diagnosis"), (OLD_B4_ROOT, "old B4")):
        if not (root / "COMPLETE").is_file():
            raise InputIdentityInvalid(f"{label} COMPLETE marker is missing")
    old_b4_pre = verify_historical_pre_search(OLD_B4_ROOT)
    old_metric_pre = verify_historical_pre_search(OLD_METRIC_ROOT)
    if old_b4_pre["root_sha256"] != old_b4["pre_search_root_sha256"]:
        raise InputIdentityInvalid("old B4 pre-search root differs from frozen identity")
    if old_metric_pre["root_sha256"] != old_metric["pre_search_root_sha256"]:
        raise InputIdentityInvalid("old M1/M2 pre-search root differs from frozen identity")
    return {
        "status": "VALID", "checked_files": checks, "archives": archive_checks,
        "diagnosis_manifest": verify_directory_manifest(DIAGNOSIS_ROOT),
        "old_b4_manifest": verify_directory_manifest(OLD_B4_ROOT),
        "old_b4_pre_search": old_b4_pre,
        "old_metric_pre_search": old_metric_pre,
        "test_accessed": False,
    }


def validate_authoritative_inputs(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = static_input_audit(config)
    authority = config["authoritative_inputs"]
    diagnosis = json.loads((DIAGNOSIS_ROOT / "decision.json").read_text(encoding="utf-8"))
    diagnosis_state = json.loads((DIAGNOSIS_ROOT / "state.json").read_text(encoding="utf-8"))
    prospective = json.loads(
        (DIAGNOSIS_ROOT / "next_experiment/prospective_diversity_test_spec.json").read_text(encoding="utf-8")
    )
    expected_diagnosis = {
        "classification": authority["collapse_diagnosis"]["required_classification"],
        "SYSTEMATIC_COLLAPSE": authority["collapse_diagnosis"]["required_systematic_collapse"],
        "SATURATION_PATTERN": authority["collapse_diagnosis"]["required_saturation_pattern"],
        "SCORER_FOR_NEXT_DIVERSITY_TEST": authority["collapse_diagnosis"]["required_scorer"],
        "SELECTED_DIVERSITY_POLICY": authority["collapse_diagnosis"]["required_policy"],
        "test_accessed": False,
    }
    observed_diagnosis = {key: diagnosis.get(key) for key in expected_diagnosis}
    if observed_diagnosis != expected_diagnosis:
        raise InputIdentityInvalid(f"diagnosis authorization mismatch: {observed_diagnosis}")
    if diagnosis_state.get("classification") != expected_diagnosis["classification"] or diagnosis_state.get("test_accessed") is not False:
        raise InputIdentityInvalid("diagnosis state identity mismatch")
    if prospective.get("selected_scorer") != "B4" or prospective.get("selected_diversity_policy") != "P2":
        raise InputIdentityInvalid("prospective specification does not freeze B4 + P2")
    old_decision = json.loads((OLD_B4_ROOT / "decision.json").read_text(encoding="utf-8"))
    old_state = json.loads((OLD_B4_ROOT / "state.json").read_text(encoding="utf-8"))
    required_old = authority["previous_b4_search"]["required_classification"]
    if old_decision.get("classification") != required_old or old_state.get("classification") != required_old:
        raise InputIdentityInvalid("old B4 classification mismatch")
    if old_decision.get("test_accessed") is not False or old_state.get("test_accessed") is not False:
        raise InputIdentityInvalid("old B4 test firewall identity mismatch")
    base_config = base.load_config(base.DEFAULT_CONFIG)
    base.validate_frozen_protocol(base_config)
    upstream = base.validate_authoritative_inputs(base_config)
    authorization = {
        "status": "IDENTICAL", "classification": diagnosis["classification"],
        "SYSTEMATIC_COLLAPSE": diagnosis["SYSTEMATIC_COLLAPSE"],
        "SATURATION_PATTERN": diagnosis["SATURATION_PATTERN"],
        "SCORER_FOR_NEXT_DIVERSITY_TEST": diagnosis["SCORER_FOR_NEXT_DIVERSITY_TEST"],
        "SELECTED_DIVERSITY_POLICY": diagnosis["SELECTED_DIVERSITY_POLICY"],
        "prospective_specification_only": prospective.get("status") == "SPECIFICATION_ONLY",
        "candidate_generation_started_in_diagnosis": prospective.get("candidate_generation_started"),
        "GNN_training_started_in_diagnosis": prospective.get("GNN_training_started"),
        "test_accessed": False,
    }
    audit.update({
        "authorization": observed_diagnosis,
        "old_b4_classification": required_old,
        "upstream_frozen_B4_inputs": upstream,
        "prospective_or_search_labels_authorized_for_B4_training": False,
        "test_accessed": False,
    })
    return audit, authorization


def _parse_candidate_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    json_fields = {
        "raw_hp_vector", "raw_hp", "operations", "edges", "gat_heads_by_layer",
        "sage_aggr_by_layer", "gin_eps_by_layer",
    }
    int_fields = {
        "replicate_id", "round_id", "candidate_slot", "pool_id", "batch_id",
        "architecture_slot", "effective_layers", "hidden_dim", "gat_heads",
        "generation_seed", "generation_draw_index",
    }
    float_fields = {"lr", "dropout", "l2", "gin_eps", "b4_prediction", "m1_prediction", "m2_prediction"}
    bool_fields = {"b4_selected", "m1_selected", "m2_selected", "random_selected"}
    for raw in read_csv(path):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if value == "":
                row[key] = None
            elif key in json_fields:
                row[key] = json.loads(value)
            elif key in int_fields:
                row[key] = int(value)
            elif key in float_fields:
                row[key] = float(value)
            elif key in bool_fields:
                row[key] = value.lower() == "true"
            else:
                row[key] = value
        row.setdefault("joint_config_fingerprint", row.get("joint_fingerprint"))
        rows.append(row)
    return rows


def score_candidates_with_frozen_b4(
    runtime: Mapping[str, Any], folds: Sequence[base.B4Fold],
    candidates: Sequence[Mapping[str, Any]], log: Callable[[str], None],
    *, label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        raise ProtocolError(f"B4 scoring received an empty {label} candidate set")
    soft = runtime["soft"]
    frozen_config = soft.load_config(runtime["soft_config_path"])
    base_config = base.load_config(base.DEFAULT_CONFIG)
    expected_epochs = base._expected_best_epochs(base_config)
    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (
            int(row.get("replicate_id", row.get("pool_id", 0))),
            int(row.get("round_id", row.get("batch_id", 0))),
            int(row.get("candidate_slot", row.get("architecture_slot", 0))),
            str(row["joint_fingerprint"]),
        ),
    )
    fold_predictions: dict[int, np.ndarray] = {}
    runs = []
    for fold in folds:
        seed_predictions: dict[int, np.ndarray] = {}
        for seed in B4_MODEL_SEEDS:
            log(f"{label}: fold {fold.fold}/4 seed {seed}, {len(ordered)} frozen candidates")
            prediction, audit = soft.joint_v1.train_direct_joint_regressor(
                fold.train_rows, ordered, seed, frozen_config, fold.hp_standardizer,
            )
            if int(audit["best_epoch"]) != expected_epochs[(fold.fold, seed)]:
                raise B4ScorerReplayMismatch(
                    f"{label} best epoch mismatch fold={fold.fold}, seed={seed}: {audit['best_epoch']}"
                )
            seed_predictions[seed] = np.asarray(prediction, dtype=np.float64)
            runs.append({"fold": fold.fold, "seed": seed, "candidate_set": label, **audit})
        fold_predictions[fold.fold] = np.asarray(
            soft.joint_v1.average_metric_seed_predictions(seed_predictions), dtype=np.float64,
        )
    pooled = np.mean(np.stack([fold_predictions[index] for index in range(5)], axis=0), axis=0)
    if pooled.shape != (len(ordered),) or not bool(np.isfinite(pooled).all()):
        raise B4ScorerReplayMismatch(f"B4 produced invalid {label} predictions")
    rows = [
        {
            "replicate_id": int(candidate.get("replicate_id", candidate.get("pool_id", 0))),
            "round_id": int(candidate.get("round_id", candidate.get("batch_id", 0))),
            "candidate_slot": int(candidate.get("candidate_slot", candidate.get("architecture_slot", 0))),
            "joint_fingerprint": str(candidate["joint_fingerprint"]),
            "architecture_fingerprint": str(candidate["architecture_fingerprint"]),
            "b4_prediction": float(value),
        }
        for candidate, value in zip(ordered, pooled)
    ]
    return rows, {
        "status": "SCORED", "candidate_set": label, "candidate_count": len(rows),
        "historical_supervision": "historical_336_only",
        "model_seeds": list(B4_MODEL_SEEDS), "outer_folds": 5,
        "seed_then_fold_arithmetic_mean": True,
        "deterministic_reconstruction_is_identity_replay_not_scientific_refit": True,
        "previous_prospective_labels_used": False, "old_search_labels_used": False,
        "current_labels_used": False, "online_update_used": False,
        "round_refit_used": False, "all_predictions_before_first_GNN": True,
        "runs": runs, "test_accessed": False,
    }


def replay_frozen_b4(
    runtime: Mapping[str, Any], folds: Sequence[base.B4Fold],
    log: Callable[[str], None],
) -> dict[str, Any]:
    base_config = base.load_config(base.DEFAULT_CONFIG)
    upstream = base.replay_frozen_b4(runtime, folds, base_config, log)
    old_manifest_path = OLD_B4_ROOT / "pre_search/candidate_manifest.csv"
    old_predictions_path = OLD_B4_ROOT / "pre_search/b4_predictions.csv"
    old_candidates = _parse_candidate_csv(old_manifest_path)
    old_reference_rows = read_csv(old_predictions_path)
    if len(old_candidates) != 4096 or len(old_reference_rows) != 4096:
        raise B4ScorerReplayMismatch("old B4 search replay anchors must contain 4096 rows")
    if len({str(row["joint_fingerprint"]) for row in old_candidates}) != 4096:
        raise B4ScorerReplayMismatch("old B4 search candidate anchors are not unique")
    reference = {str(row["joint_fingerprint"]): float(row["b4_prediction"]) for row in old_reference_rows}
    if len(reference) != 4096:
        raise B4ScorerReplayMismatch("old B4 search prediction anchors are not unique")
    replayed_rows, old_audit = score_candidates_with_frozen_b4(
        runtime, folds, old_candidates, log, label="previous_B4_search_4096_replay",
    )
    maximum_error = float(upstream["maximum_absolute_error"])
    for row in replayed_rows:
        joint = str(row["joint_fingerprint"])
        if joint not in reference:
            raise B4ScorerReplayMismatch(f"old B4 replay contains a foreign joint: {joint}")
        error = abs(float(row["b4_prediction"]) - reference[joint])
        maximum_error = max(maximum_error, error)
        if error > REPLAY_ATOL:
            raise B4ScorerReplayMismatch(f"old B4 search prediction mismatch at {joint}: {error}")
    if {str(row["joint_fingerprint"]) for row in replayed_rows} != set(reference):
        raise B4ScorerReplayMismatch("old B4 search replay coverage is incomplete")
    return {
        **upstream,
        "status": "IDENTICAL",
        "previous_B4_search_prediction_count": 4096,
        "maximum_absolute_error": maximum_error,
        "absolute_tolerance": REPLAY_ATOL,
        "previous_B4_search_replay_audit": old_audit,
        "deterministic_reconstruction_role": "frozen_identity_replay",
        "B4_retrained_or_updated_scientifically": False,
        "old_search_labels_loaded_for_training": False,
        "current_labels_loaded_for_training": False,
        "anchors": {
            **upstream["anchors"],
            "previous_B4_search_candidates_sha256": sha256_file(old_manifest_path),
            "previous_B4_search_predictions_sha256": sha256_file(old_predictions_path),
            "previous_B4_search_pre_search_root": verify_historical_pre_search(OLD_B4_ROOT)["root_sha256"],
        },
        "test_accessed": False,
    }


def _unwrap_candidates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        payload = payload.get("candidates", payload.get("rows", payload))
    if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
        raise InputIdentityInvalid("candidate manifest is not a record list")
    return [dict(row) for row in payload]


def load_exclusion_identities(
    historical_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    pilot_path = REPO_ROOT / "results/posthoc/joint-metric-prospective-selection-gate-global4-v1-rerun1/pre_eval/candidate_manifest.json"
    confirmation_path = REPO_ROOT / "results/posthoc/joint-metric-independent-confirmation-surrogate-selection-global4-v1-rerun1/pre_eval/candidate_manifest.json"
    old_b4_path = OLD_B4_ROOT / "pre_search/candidate_manifest.csv"
    old_metric_path = OLD_METRIC_ROOT / "pre_search/candidate_manifest.csv"
    pilot = _unwrap_candidates(json.loads(pilot_path.read_text(encoding="utf-8")))
    confirmation = _unwrap_candidates(json.loads(confirmation_path.read_text(encoding="utf-8")))
    old_b4 = _parse_candidate_csv(old_b4_path)
    old_metric = _parse_candidate_csv(old_metric_path)
    source_rows: dict[str, Sequence[Mapping[str, Any]]] = {
        "historical_336": historical_rows,
        "previous_8_pool_32": pilot,
        "previous_16_pool_64": confirmation,
        "old_b4_search_all_4096": old_b4,
        "old_m1_m2_search_all_4096": old_metric,
    }
    expected_counts = {
        "historical_336": 336, "previous_8_pool_32": 32,
        "previous_16_pool_64": 64, "old_b4_search_all_4096": 4096,
        "old_m1_m2_search_all_4096": 4096,
    }
    joint_sets: dict[str, set[str]] = {}
    architecture_sets: dict[str, set[str]] = {}
    for name, rows in source_rows.items():
        joints = {
            str(row.get("joint_fingerprint", row.get("joint_config_fingerprint")))
            for row in rows
        }
        architectures = {str(row["architecture_fingerprint"]) for row in rows}
        if len(rows) != expected_counts[name] or len(joints) != expected_counts[name] or "None" in joints:
            raise InputIdentityInvalid(f"exclusion identity cardinality mismatch: {name}")
        joint_sets[name] = joints
        architecture_sets[name] = architectures
    overlaps = []
    names = list(joint_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlaps.append({"left": left, "right": right, "joint_overlap": len(joint_sets[left] & joint_sets[right])})
    if any(row["joint_overlap"] for row in overlaps):
        raise InputIdentityInvalid(f"authoritative exclusion sets unexpectedly overlap: {overlaps}")
    all_joint = set().union(*joint_sets.values())
    all_architecture = set().union(*architecture_sets.values())
    if len(all_joint) != 8624:
        raise InputIdentityInvalid(f"exclusion union count {len(all_joint)} != 8624")
    authority = config["authoritative_inputs"]
    sources = [
        {"id": "historical_336", "count": 336, "origin": "authenticated historical B4 loader"},
        {"id": "previous_8_pool_32", "count": 32, "path": str(pilot_path), "sha256": sha256_file(pilot_path)},
        {"id": "previous_16_pool_64", "count": 64, "path": str(confirmation_path), "sha256": sha256_file(confirmation_path)},
        {"id": "old_b4_search_all_4096", "count": 4096, "path": str(old_b4_path), "sha256": sha256_file(old_b4_path), "pre_search_root": verify_historical_pre_search(OLD_B4_ROOT)["root_sha256"]},
        {"id": "old_m1_m2_search_all_4096", "count": 4096, "path": str(old_metric_path), "sha256": sha256_file(old_metric_path), "pre_search_root": verify_historical_pre_search(OLD_METRIC_ROOT)["root_sha256"]},
    ]
    if sources[1]["sha256"] != authority["previous_8_pool"]["candidate_manifest_sha256"]:
        raise InputIdentityInvalid("8-pool exclusion identity changed")
    if sources[2]["sha256"] != authority["previous_16_pool"]["candidate_manifest_sha256"]:
        raise InputIdentityInvalid("16-pool exclusion identity changed")
    return {
        "joint": all_joint,
        "architecture": all_architecture,
        "historical_architecture": architecture_sets["historical_336"],
        "prior_search_architecture": architecture_sets["old_b4_search_all_4096"] | architecture_sets["old_m1_m2_search_all_4096"],
        **{f"joint_{name}": joints for name, joints in joint_sets.items()},
    }, {
        "status": "LOADED", "sources": sources, "pairwise_overlaps": overlaps,
        "joint_union_count": len(all_joint), "architecture_union_count": len(all_architecture),
        "all_current_exact_joints_must_be_outside_union": True,
        "current_earlier_candidates_excluded_online": True, "test_accessed": False,
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
        "replicate_id": replicate, "round_id": round_id,
        "candidate_slot": candidate_slot, "pool_id": replicate,
        "batch_id": round_id, "architecture_slot": candidate_slot,
        "config_id": f"b4-diversity-r{replicate}-q{round_id}-s{candidate_slot}",
        "joint_fingerprint": joint, "joint_config_fingerprint": joint,
        "architecture_fingerprint": architecture, "hp_fingerprint": hp,
        "operations": list(graph_config["operations"]),
        "edges": [list(map(int, edge)) for edge in graph_config["edges"]],
        "effective_layers": int(graph_config["effective_layers"]),
        "raw_hp_vector": raw, "hp_regime": "hp_high",
        "generation_seed": CANDIDATE_MASTER_SEED + replicate,
        "generation_draw_index": draw_index,
        "proposal_distribution": "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR",
    }
    row.update(high.v1.realize_global4_hp(row))
    row["raw_hp"] = {key: row[key] for key in ("lr", "dropout", "hidden_dim", "l2")}
    return row


def validate_candidate_set(
    candidates: Sequence[Mapping[str, Any]], exclusions: Mapping[str, set[str]],
    *, expected_count: int = TOTAL_CANDIDATES,
) -> dict[str, Any]:
    if len(candidates) != expected_count:
        raise CandidateGenerationFailed(f"candidate count {len(candidates)} != {expected_count}")
    joints = [str(row["joint_fingerprint"]) for row in candidates]
    architectures = [str(row["architecture_fingerprint"]) for row in candidates]
    if len(set(joints)) != len(joints):
        raise CandidateGenerationFailed("current exact joint fingerprints are not globally unique")
    if set(joints) & exclusions["joint"]:
        raise CandidateGenerationFailed("an excluded exact joint leaked into current candidates")
    counts = Counter((int(row["replicate_id"]), int(row["round_id"])) for row in candidates)
    if expected_count == TOTAL_CANDIDATES:
        expected_grid = {(replicate, round_id) for replicate in range(REPLICATES) for round_id in range(ROUNDS)}
        if set(counts) != expected_grid or any(value != CANDIDATES_PER_ROUND for value in counts.values()):
            raise CandidateGenerationFailed("candidate replicate/round grid is incomplete")
    if {str(row["proposal_distribution"]) for row in candidates} != {"MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR"}:
        raise CandidateGenerationFailed("candidate generation was not exclusively model-blind high-regime")
    architecture_counts = Counter(architectures)
    pool_unique = [
        len({str(row["architecture_fingerprint"]) for row in candidates if int(row["replicate_id"]) == r and int(row["round_id"]) == q})
        for r in range(REPLICATES) for q in range(ROUNDS)
    ] if expected_count == TOTAL_CANDIDATES else []
    return {
        "passed": True, "candidate_count": len(candidates),
        "unique_joint_count": len(set(joints)),
        "unique_architecture_count": len(architecture_counts),
        "architecture_repetition_rate": (len(architectures) - len(architecture_counts)) / len(architectures),
        "records_with_repeated_architecture_fraction": sum(architecture_counts[value] > 1 for value in architectures) / len(architectures),
        "historical_seen_architecture_fraction": sum(value in exclusions["historical_architecture"] for value in architectures) / len(architectures),
        "prior_search_seen_architecture_fraction": sum(value in exclusions["prior_search_architecture"] for value in architectures) / len(architectures),
        "per_pool_unique_architectures": pool_unique,
        "per_pool_unique_architectures_min": min(pool_unique) if pool_unique else None,
        "per_pool_unique_architectures_median": statistics.median(pool_unique) if pool_unique else None,
        "per_pool_unique_architectures_mean": statistics.fmean(pool_unique) if pool_unique else None,
        "per_pool_unique_architectures_max": max(pool_unique) if pool_unique else None,
        "architecture_reuse_allowed": True,
        "predictor_accessed_during_generation": False,
        "validation_accessed_during_generation": False,
        "test_accessed": False,
    }


def generate_all_candidates(
    runtime: Mapping[str, Any], exclusions: Mapping[str, set[str]],
    log: Callable[[str], None],
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
    accepted_joint: set[str] = set()
    candidates: list[dict[str, Any]] = []
    replicate_audits = []
    for replicate in range(REPLICATES):
        seed = CANDIDATE_MASTER_SEED + replicate
        architecture_rng = torch.Generator(device="cpu")
        architecture_rng.manual_seed(seed)
        hp_rng = np.random.default_rng(seed)
        decoded_draws = processed_draws = invalid_draws = excluded_draws = 0
        round_audits = []
        for round_id in range(ROUNDS):
            round_rows: list[dict[str, Any]] = []
            processed_before = processed_draws
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
                    raw_hp = hp_rng.uniform(0.80, 0.95, size=4).astype(np.float64).tolist()
                    row = _candidate_row(
                        high, model, graph, replicate, round_id, len(round_rows),
                        raw_hp, batch_start + offset,
                    )
                    joint = str(row["joint_fingerprint"])
                    if joint in exclusions["joint"] or joint in accepted_joint:
                        excluded_draws += 1
                        continue
                    accepted_joint.add(joint)
                    candidates.append(row)
                    round_rows.append(row)
                    if len(round_rows) == CANDIDATES_PER_ROUND:
                        break
            if len(round_rows) != CANDIDATES_PER_ROUND:
                raise CandidateGenerationFailed(f"replicate {replicate} round {round_id} exhausted fixed draw budget")
            round_audits.append({
                "replicate_id": replicate, "round_id": round_id, "accepted": len(round_rows),
                "processed_draws_this_round": processed_draws - processed_before,
                "decoded_draws_cumulative": decoded_draws,
            })
            log(f"replicate {replicate} round {round_id}: accepted 128; decoded={decoded_draws}")
        replicate_audits.append({
            "replicate_id": replicate, "seed": seed, "decoded_draws": decoded_draws,
            "processed_draws": processed_draws, "invalid_draws": invalid_draws,
            "excluded_exact_joint_draws": excluded_draws, "rounds": round_audits,
        })
    coverage = validate_candidate_set(candidates, exclusions)
    return candidates, {
        "status": "GENERATED", "candidate_master_seed": CANDIDATE_MASTER_SEED,
        "replicate_seed_rule": "master_plus_zero_based_replicate",
        "rounds_advance_continuous_replicate_RNG_streams": True,
        "proposal_distribution": "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR",
        "hp_raw_unit_interval_range": [0.80, 0.95],
        "predictor_accessed": False, "validation_accessed": False,
        "historical_best_accessed": False, "metric_accessed": False,
        "head_accessed": False, "mutation_or_local_search_used": False,
        "acquisition_used": False, "replicates": replicate_audits,
        "coverage": coverage, "test_accessed": False,
    }


def _group_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(int(row["replicate_id"]), int(row["round_id"]))].append(dict(row))
    expected = {(replicate, round_id) for replicate in range(REPLICATES) for round_id in range(ROUNDS)}
    if set(groups) != expected:
        raise ProtocolError("candidate pool grid is incomplete")
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row["candidate_slot"]))
        if len(rows) != CANDIDATES_PER_ROUND or [int(row["candidate_slot"]) for row in rows] != list(range(CANDIDATES_PER_ROUND)):
            raise ProtocolError(f"invalid candidate pool: {key}")
    return groups


def _plan(method: str, rounds: list[dict[str, Any]], **metadata: Any) -> dict[str, Any]:
    return {
        "method": method, "rounds": rounds,
        "selected_slot_count": len(rounds) * SELECTED_PER_ROUND,
        "frozen_before_first_GNN": True,
        "validation_used": False, "test_accessed": False, **metadata,
    }


def select_search_plans(
    candidates: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
    *, random_seed: int = RANDOM_MASTER_SEED,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, Any], dict[str, Any]]:
    groups = _group_candidates(candidates)
    score_by_joint = {str(row["joint_fingerprint"]): float(row["b4_prediction"]) for row in predictions}
    candidate_joints = {str(row["joint_fingerprint"]) for row in candidates}
    if len(score_by_joint) != len(candidates) or set(score_by_joint) != candidate_joints:
        raise ProtocolError("frozen B4 prediction coverage differs from candidate manifest")
    forbidden_label_keys = {"validation_mean", "validation_accuracy", "val_acc"}
    if any(forbidden_label_keys & set(row) for row in candidates):
        raise ProtocolError("validation labels reached selection planning")
    random_rng = np.random.default_rng(random_seed)
    pure_rounds: list[dict[str, Any]] = []
    p2_rounds: list[dict[str, Any]] = []
    random_rounds: list[dict[str, Any]] = []
    selected = {method: set() for method in METHODS}
    per_replicate = {replicate: {method: set() for method in METHODS} for replicate in range(REPLICATES)}
    p2_architecture_counts: Counter[tuple[int, str]] = Counter()
    for replicate in range(REPLICATES):
        for round_id in range(ROUNDS):
            pool = groups[(replicate, round_id)]
            ranked = sorted(
                pool,
                key=lambda row: (-score_by_joint[str(row["joint_fingerprint"])], str(row["joint_fingerprint"])),
            )
            pure_selected = ranked[:SELECTED_PER_ROUND]
            p2_selected = []
            round_architectures: set[str] = set()
            for rank, candidate in enumerate(ranked, 1):
                architecture = str(candidate["architecture_fingerprint"])
                if architecture in round_architectures:
                    continue
                if p2_architecture_counts[(replicate, architecture)] >= 2:
                    continue
                selected_row = dict(candidate)
                selected_row["original_rank"] = rank
                p2_selected.append(selected_row)
                round_architectures.add(architecture)
                p2_architecture_counts[(replicate, architecture)] += 1
                if len(p2_selected) == SELECTED_PER_ROUND:
                    break
            if len(p2_selected) != SELECTED_PER_ROUND:
                raise P2PolicyInfeasible(f"P2 cannot fill replicate={replicate}, round={round_id}")
            indices = random_rng.choice(CANDIDATES_PER_ROUND, size=SELECTED_PER_ROUND, replace=False)
            random_selected = [pool[int(index)] for index in sorted(indices.tolist())]
            method_rows = {PURE: pure_selected, P2: p2_selected, RANDOM: random_selected}
            for method, chosen in method_rows.items():
                joints = [str(row["joint_fingerprint"]) for row in chosen]
                selected[method].update(joints)
                per_replicate[replicate][method].update(joints)
            pure_rounds.append({
                "replicate_id": replicate, "round_id": round_id,
                "selected_joint_fingerprints": [str(row["joint_fingerprint"]) for row in pure_selected],
                "selected_candidate_slots": [int(row["candidate_slot"]) for row in pure_selected],
                "selected_predictions": [score_by_joint[str(row["joint_fingerprint"])] for row in pure_selected],
                "selected_original_ranks": [1, 2, 3, 4],
                "tie_break": "joint_fingerprint_lexicographic_ascending",
            })
            p2_rounds.append({
                "replicate_id": replicate, "round_id": round_id,
                "selected_joint_fingerprints": [str(row["joint_fingerprint"]) for row in p2_selected],
                "selected_candidate_slots": [int(row["candidate_slot"]) for row in p2_selected],
                "selected_predictions": [score_by_joint[str(row["joint_fingerprint"])] for row in p2_selected],
                "selected_original_ranks": [int(row["original_rank"]) for row in p2_selected],
                "selected_architecture_fingerprints": [str(row["architecture_fingerprint"]) for row in p2_selected],
                "tie_break": "joint_fingerprint_lexicographic_ascending",
            })
            random_rounds.append({
                "replicate_id": replicate, "round_id": round_id,
                "selected_joint_fingerprints": [str(row["joint_fingerprint"]) for row in random_selected],
                "selected_candidate_slots": [int(row["candidate_slot"]) for row in random_selected],
                "sampling": "uniform_without_replacement",
            })
    plans = {
        PURE: _plan(PURE, pure_rounds, selection="frozen_B4_top_4", tie_break="joint_fingerprint_lexicographic_ascending"),
        P2: _plan(
            P2, p2_rounds, policy="ROUND_UNIQUE_CAP2",
            within_round_unique_architectures=4,
            maximum_architecture_slots_per_replicate=2,
            tie_break="joint_fingerprint_lexicographic_ascending",
        ),
        RANDOM: _plan(
            RANDOM, random_rounds, master_seed=random_seed,
            sampling="numpy_default_rng_uniform_without_replacement",
            rng_order="replicate_then_round", overlap_redraw=False,
        ),
    }
    feasibility = audit_p2_feasibility(plans[P2], candidates)
    overlap_rows = []
    divergence = []
    for replicate in range(REPLICATES):
        pure_set = per_replicate[replicate][PURE]
        p2_set = per_replicate[replicate][P2]
        random_set = per_replicate[replicate][RANDOM]
        if any(len(values) != 16 for values in (pure_set, p2_set, random_set)):
            raise ProtocolError(f"selection method lacks 16 unique configs in replicate {replicate}")
        value = 16 - len(pure_set & p2_set)
        divergence.append(value)
        overlap_rows.append({
            "replicate_id": replicate,
            "pure_p2_overlap": len(pure_set & p2_set),
            "pure_random_overlap": len(pure_set & random_set),
            "p2_random_overlap": len(p2_set & random_set),
            "triple_overlap": len(pure_set & p2_set & random_set),
            "DIVERGENCE_r": value,
        })
    information = {
        "definition": "16 - |PURE_B4_selection_set intersect P2_selection_set|",
        "per_replicate": overlap_rows,
        "replicates_with_divergence_at_least_4": sum(value >= 4 for value in divergence),
        "median_divergence": float(statistics.median(divergence)),
        "minimum_required_replicates": 8, "minimum_required_median": 4,
        "passed": sum(value >= 4 for value in divergence) >= 8 and statistics.median(divergence) >= 4,
        "candidate_regeneration_for_disagreement": False,
        "test_accessed": False,
    }
    union = set().union(*selected.values())
    overlap = {
        "per_replicate": overlap_rows,
        "overall": {
            "pure_p2_overlap": len(selected[PURE] & selected[P2]),
            "pure_random_overlap": len(selected[PURE] & selected[RANDOM]),
            "p2_random_overlap": len(selected[P2] & selected[RANDOM]),
            "triple_overlap": len(selected[PURE] & selected[P2] & selected[RANDOM]),
            "selected_union_unique_configurations": len(union),
            "physical_unique_GNN_runs": len(union) * 3,
            "method_equivalent_config_slots": 576,
            "method_equivalent_GNN_runs": 1728,
        },
        "overlap_redraw_performed": False, "test_accessed": False,
    }
    return plans, selected, feasibility, {"selection_overlap": overlap, "information_gate": information}


def audit_p2_feasibility(plan: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_joint = {str(row["joint_fingerprint"]): row for row in candidates}
    plan_rounds = list(plan.get("rounds", []))
    if len(plan_rounds) != REPLICATES * ROUNDS:
        raise P2PolicyInfeasible("P2 plan does not contain 48 rounds")
    observed_grid = [
        (int(row["replicate_id"]), int(row["round_id"])) for row in plan_rounds
    ]
    expected_grid = [(replicate, round_id) for replicate in range(REPLICATES) for round_id in range(ROUNDS)]
    if observed_grid != expected_grid or len(set(observed_grid)) != REPLICATES * ROUNDS:
        raise P2PolicyInfeasible("P2 plan is not the exact ordered 12x4 replicate/round grid")
    round_rows = []
    per_replicate_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in plan_rounds:
        replicate = int(row["replicate_id"])
        round_id = int(row["round_id"])
        joints = list(map(str, row["selected_joint_fingerprints"]))
        if len(joints) != 4 or len(set(joints)) != 4:
            raise P2PolicyInfeasible(f"P2 selection cardinality failed at {replicate}/{round_id}")
        architectures = [str(by_joint[joint]["architecture_fingerprint"]) for joint in joints]
        if len(set(architectures)) != 4:
            raise P2PolicyInfeasible(f"P2 round architecture uniqueness failed at {replicate}/{round_id}")
        per_replicate_counts[replicate].update(architectures)
        round_rows.append({
            "replicate_id": replicate, "round_id": round_id,
            "selected_config_count": len(joints), "unique_architecture_count": len(set(architectures)),
        })
    maximum = max((max(counts.values(), default=0) for counts in per_replicate_counts.values()), default=0)
    if set(per_replicate_counts) != set(range(REPLICATES)) or maximum > 2:
        raise P2PolicyInfeasible(f"P2 replicate architecture cap failed: maximum={maximum}")
    if any(sum(counts.values()) != ROUNDS * SELECTED_PER_ROUND for counts in per_replicate_counts.values()):
        raise P2PolicyInfeasible("P2 does not allocate exactly 16 slots per replicate")
    return {
        "passed": True, "legal_rounds": len(round_rows), "required_legal_rounds": 48,
        "every_round_unique_architectures": 4,
        "maximum_observed_architecture_slots_per_replicate": maximum,
        "maximum_allowed_architecture_slots_per_replicate": 2,
        "rounds": round_rows, "validation_used": False, "test_accessed": False,
    }


def concentration(architectures: Sequence[str]) -> dict[str, Any]:
    total = len(architectures)
    counts = Counter(map(str, architectures))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    shares = [count / total for _architecture, count in ordered] if total else []
    entropy = -sum(value * math.log(value) for value in shares if value > 0)
    return {
        "selected_slot_count": total, "unique_architecture_count": len(counts),
        "architecture_frequency_distribution": dict(ordered),
        "top1_share": sum(shares[:1]), "top2_share": sum(shares[:2]),
        "top4_share": sum(shares[:4]), "shannon_entropy": entropy,
        "effective_architecture_count": math.exp(entropy) if total else 0.0,
        "HHI": sum(value * value for value in shares),
    }


def _selection_records(
    plans: Mapping[str, Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_joint = {str(row["joint_fingerprint"]): row for row in candidates}
    records: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for method in METHODS:
        for round_row in plans[method]["rounds"]:
            replicate = int(round_row["replicate_id"])
            round_id = int(round_row["round_id"])
            for position, joint in enumerate(map(str, round_row["selected_joint_fingerprints"])):
                candidate = by_joint[joint]
                records[method].append({
                    "method": method, "replicate_id": replicate, "round_id": round_id,
                    "selected_position": position, "joint_fingerprint": joint,
                    "architecture_fingerprint": str(candidate["architecture_fingerprint"]),
                })
    return records


def _repeat_fractions(records: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    within_duplicate_slots = repeat_slots = 0
    for replicate in range(REPLICATES):
        previously_seen: set[str] = set()
        for round_id in range(ROUNDS):
            architectures = [
                str(row["architecture_fingerprint"]) for row in records
                if int(row["replicate_id"]) == replicate and int(row["round_id"]) == round_id
            ]
            within_duplicate_slots += len(architectures) - len(set(architectures))
            repeat_slots += sum(architecture in previously_seen for architecture in architectures)
            previously_seen.update(architectures)
    total = len(records)
    return within_duplicate_slots / total, repeat_slots / total


def diversity_manipulation_audit(
    plans: Mapping[str, Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = _selection_records(plans, candidates)
    methods: dict[str, Any] = {}
    for method, rows in records.items():
        overall = concentration([str(row["architecture_fingerprint"]) for row in rows])
        within, repeat = _repeat_fractions(rows)
        per_replicate = []
        for replicate in range(REPLICATES):
            subset = [row for row in rows if int(row["replicate_id"]) == replicate]
            metric = concentration([str(row["architecture_fingerprint"]) for row in subset])
            per_replicate.append({"replicate_id": replicate, **metric})
        methods[method] = {
            "overall": {**overall, "within_round_duplicate_fraction": within, "cross_round_repeat_fraction": repeat},
            "per_replicate": per_replicate,
        }
    pure = methods[PURE]["overall"]
    p2 = methods[P2]["overall"]
    return {
        "status": "PASSED",
        "methods": methods,
        "PURE_vs_P2": {
            "unique_architecture_gain": p2["unique_architecture_count"] - pure["unique_architecture_count"],
            "effective_architecture_gain": p2["effective_architecture_count"] - pure["effective_architecture_count"],
            "repeat_fraction_reduction": pure["cross_round_repeat_fraction"] - p2["cross_round_repeat_fraction"],
            "top2_share_reduction": pure["top2_share"] - p2["top2_share"],
        },
        "selection_modified_after_audit": False, "test_accessed": False,
    }


def freeze_pre_search(
    pre_search: Path, candidates: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]], plans: Mapping[str, Mapping[str, Any]],
    overlap: Mapping[str, Any], manipulation: Mapping[str, Any],
    information: Mapping[str, Any], feasibility: Mapping[str, Any],
    scorer_identity: Mapping[str, Any], generator_identity: Mapping[str, Any],
    exclusion_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if pre_search.exists() and any(pre_search.iterdir()):
        raise ProtocolError(f"pre-search directory is not empty: {pre_search}")
    pre_search.mkdir(parents=True, exist_ok=True)
    score_by_joint = {str(row["joint_fingerprint"]): float(row["b4_prediction"]) for row in predictions}
    selected = {
        method: {
            str(joint) for row in plans[method]["rounds"]
            for joint in row["selected_joint_fingerprints"]
        }
        for method in METHODS
    }
    manifest_rows = []
    for candidate in sorted(
        candidates,
        key=lambda row: (int(row["replicate_id"]), int(row["round_id"]), int(row["candidate_slot"])),
    ):
        joint = str(candidate["joint_fingerprint"])
        manifest_rows.append({
            **dict(candidate), "b4_prediction": score_by_joint[joint],
            "pure_b4_selected": joint in selected[PURE],
            "p2_selected": joint in selected[P2],
            "random_selected": joint in selected[RANDOM],
        })
    search_plan = {
        "protocol_version": PROTOCOL_VERSION, "created_at": utc_now(),
        "candidate_count": len(candidates), "candidate_semantic_sha256": stable_hash(list(candidates)),
        "all_candidates_generated_before_first_GNN": True,
        "all_B4_predictions_completed_before_first_GNN": True,
        "all_PURE_selections_completed_before_first_GNN": True,
        "all_P2_selections_completed_before_first_GNN": True,
        "all_random_selections_completed_before_first_GNN": True,
        "P2_feasibility": dict(feasibility), "information_gate": dict(information),
        "candidate_replacement_after_freeze": False, "online_update": False,
        "GNN_training_started": False, "test_accessed": False,
    }
    dump_csv(pre_search / "candidate_manifest.csv", manifest_rows, CANDIDATE_FIELDS)
    dump_csv(
        pre_search / "b4_predictions.csv", predictions,
        ("replicate_id", "round_id", "candidate_slot", "joint_fingerprint", "architecture_fingerprint", "b4_prediction"),
    )
    dump_json(pre_search / "pure_b4_selections.json", plans[PURE])
    dump_json(pre_search / "p2_selections.json", plans[P2])
    dump_json(pre_search / "random_selections.json", plans[RANDOM])
    dump_json(pre_search / "selection_overlap.json", overlap)
    dump_json(pre_search / "diversity_manipulation.json", manipulation)
    dump_json(pre_search / "search_plan.json", search_plan)
    dump_json(pre_search / "scorer_identity.json", scorer_identity)
    dump_json(pre_search / "generator_identity.json", generator_identity)
    dump_json(pre_search / "exclusion_manifest.json", exclusion_manifest)
    files = [
        {"relative_path": relative, "sha256": sha256_file(pre_search / relative)}
        for relative in sorted(PRE_SEARCH_FILES)
    ]
    root = stable_hash([(row["relative_path"], row["sha256"]) for row in files])
    freeze = {
        "protocol_version": PROTOCOL_VERSION, "created_at": utc_now(),
        "root_algorithm": "sha256_of_canonical_JSON_sorted_relative_path_and_file_sha256_pairs",
        "files": files, "root_sha256": root, "candidate_count": len(candidates),
        "pure_b4_slots": SLOTS_PER_METHOD, "p2_slots": SLOTS_PER_METHOD,
        "random_slots": SLOTS_PER_METHOD,
        "candidate_generation_seeds": [CANDIDATE_MASTER_SEED + replicate for replicate in range(REPLICATES)],
        "random_master_seed": RANDOM_MASTER_SEED,
        "input_hashes": scorer_identity.get("input_hashes", {}),
        "source_hashes": scorer_identity.get("source_hashes", {}),
        "test_accessed": False,
    }
    dump_json(pre_search / "freeze_manifest.json", freeze)
    dump_text(pre_search / "pre_search_root_sha256.txt", root + "\n")
    dump_text(pre_search / "PRE_SEARCH_FROZEN", root + "\n")
    return freeze


def verify_pre_search_freeze(pre_search: Path) -> dict[str, Any]:
    marker = pre_search / "PRE_SEARCH_FROZEN"
    freeze_path = pre_search / "freeze_manifest.json"
    if not marker.is_file() or not freeze_path.is_file():
        raise ProtocolError("pre-search marker or freeze manifest is missing")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = sorted(freeze["files"], key=lambda row: str(row["relative_path"]))
    if freeze["files"] != expected:
        raise ProtocolError("pre-search freeze entries are not lexicographically sorted")
    for row in freeze["files"]:
        path = pre_search / str(row["relative_path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ProtocolError(f"pre-search frozen file changed: {path}")
    root = stable_hash([(row["relative_path"], row["sha256"]) for row in freeze["files"]])
    roots = (
        freeze.get("root_sha256"),
        (pre_search / "pre_search_root_sha256.txt").read_text(encoding="utf-8").strip(),
        marker.read_text(encoding="utf-8").strip(),
    )
    if any(value != root for value in roots):
        raise ProtocolError("pre-search root cannot be reproduced")
    return {"passed": True, "root_sha256": root, "file_count": len(freeze["files"])}


def _selection_occurrences(plan: Mapping[str, Any], method: str) -> dict[str, list[dict[str, Any]]]:
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plan["rounds"]:
        for joint in row["selected_joint_fingerprints"]:
            occurrences[str(joint)].append({
                "method": method, "replicate_id": int(row["replicate_id"]),
                "round_id": int(row["round_id"]),
            })
    return occurrences


def build_selected_union_tasks(
    candidates: Sequence[Mapping[str, Any]], plans: Mapping[str, Mapping[str, Any]],
    *, pre_search_verified: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not pre_search_verified:
        raise ProtocolError("task creation is forbidden before pre-search verification")
    by_joint = {str(row["joint_fingerprint"]): dict(row) for row in candidates}
    occurrences = {method: _selection_occurrences(plans[method], method) for method in METHODS}
    selected_joints = sorted(set().union(*(set(values) for values in occurrences.values())))
    protocol_identity = stable_hash({
        "training_seeds": list(TRAINING_SEEDS), "max_epochs": 300,
        "patience": 80, "maximum_system_retries": 2, "track_test": False,
        "component_variant": "baseline", "evaluator": "frozen_high_regime_v1",
    })
    configs = []
    tasks = []
    for joint in selected_joints:
        if joint not in by_joint:
            raise ProtocolError(f"selection references foreign candidate: {joint}")
        candidate = by_joint[joint]
        selected_by = sorted(
            [entry for method in METHODS for entry in occurrences[method].get(joint, [])],
            key=lambda row: (str(row["method"]), int(row["replicate_id"]), int(row["round_id"])),
        )
        replicate = int(candidate["replicate_id"])
        assigned_gpu = 0 if replicate < REPLICATES // 2 else 1
        config_row = {
            **candidate, "assigned_gpu": assigned_gpu, "selected_by": selected_by,
            "selected_by_pure_b4": joint in occurrences[PURE],
            "selected_by_p2": joint in occurrences[P2],
            "selected_by_random": joint in occurrences[RANDOM],
            "training_protocol_identity": protocol_identity,
        }
        configs.append(config_row)
        for seed in TRAINING_SEEDS:
            tasks.append({
                "joint_fingerprint": joint, "training_seed": seed,
                "assigned_gpu": assigned_gpu,
                "task_key": f"b4-diversity::{joint}::seed={seed}::protocol={protocol_identity}",
                "training_protocol_identity": protocol_identity, "candidate": config_row,
            })
    expected = {(joint, seed) for joint in selected_joints for seed in TRAINING_SEEDS}
    observed = {(str(row["joint_fingerprint"]), int(row["training_seed"])) for row in tasks}
    if observed != expected or len(observed) != len(tasks):
        raise ProtocolError("selected-union config/seed de-duplication failed")
    base.reject_test_named_fields(tasks, allow_false_controls=True)
    return configs, tasks


def run_frozen_gnn_evaluation(
    output: Path, tasks: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    pre_search_root: str, log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The frozen base evaluator writes its task manifest before launching GPU
    # subprocesses.  Its formula is two-method specific (2 * SLOTS_PER_METHOD),
    # so adapt that bookkeeping constant to 576/1728 before the write.  No
    # evaluator, task, seed, retry, or scientific behavior is changed.
    old_slots = base.SLOTS_PER_METHOD
    base.SLOTS_PER_METHOD = 288
    try:
        raw, audit = base.run_frozen_gnn_evaluation(output, tasks, config, pre_search_root, log)
    finally:
        base.SLOTS_PER_METHOD = old_slots
    task_path = output / "evaluation/task_manifest.json"
    task_manifest = json.loads(task_path.read_text(encoding="utf-8"))
    task_manifest.update({
        "method_equivalent_config_slots": 576,
        "method_equivalent_GNN_runs": 1728,
        "three_method_equal_budget": True,
    })
    dump_json(task_path, task_manifest)
    audit.update({
        "method_equivalent_config_slots": 576,
        "method_equivalent_GNN_runs": 1728,
        "three_method_equal_budget": True,
    })
    return raw, audit


def aggregate_config_statistics(
    selected_configs: Sequence[Mapping[str, Any]], raw_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]], *, allow_incomplete: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    base.reject_test_named_fields(raw_rows)
    by_joint = {str(row["joint_fingerprint"]): dict(row) for row in selected_configs}
    expected = {(joint, seed) for joint in by_joint for seed in TRAINING_SEEDS}
    seen: set[tuple[str, int]] = set()
    successful: dict[tuple[str, int], float] = {}
    for row in raw_rows:
        key = (str(row["joint_fingerprint"]), int(row["training_seed"]))
        if key not in expected or key in seen:
            raise EvaluationIncomplete("evaluation contains a foreign or duplicate config/seed")
        seen.add(key)
        if row.get("terminal_status") == "success" and row.get("valid") is True:
            value = float(row["validation_accuracy"])
            if math.isfinite(value):
                successful[key] = value
    complete = seen == expected and set(successful) == expected
    score = {str(row["joint_fingerprint"]): float(row["b4_prediction"]) for row in predictions}
    stats = []
    for joint, candidate in sorted(by_joint.items()):
        values = [successful.get((joint, seed)) for seed in TRAINING_SEEDS]
        if any(value is None for value in values):
            if not allow_incomplete:
                raise EvaluationIncomplete(f"selected config lacks seeds 0,1,2: {joint}")
            mean = sd = None
        else:
            array = np.asarray(values, dtype=float)
            mean, sd = float(np.mean(array)), float(np.std(array, ddof=1))
        stats.append({
            "joint_fingerprint": joint,
            "architecture_fingerprint": str(candidate["architecture_fingerprint"]),
            "validation_seed_0": values[0], "validation_seed_1": values[1],
            "validation_seed_2": values[2], "validation_mean": mean,
            "validation_sd": sd, "b4_prediction": score[joint],
            "selected_by_pure_b4": bool(candidate["selected_by_pure_b4"]),
            "selected_by_p2": bool(candidate["selected_by_p2"]),
            "selected_by_random": bool(candidate["selected_by_random"]),
        })
    return stats, complete


def bootstrap_paired_mean(differences: Sequence[float], *, seed: int) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (REPLICATES,) or not bool(np.isfinite(values).all()):
        raise ProtocolError("bootstrap requires exactly 12 finite replicate differences")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    estimates = np.mean(values[indices], axis=1)
    return {
        "unit": "search_replicate", "input_replicates": len(values),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "seed": seed,
        "observed_mean": float(np.mean(values)),
        "CI95_lower": float(np.quantile(estimates, 0.025)),
        "CI95_upper": float(np.quantile(estimates, 0.975)),
        "percentile_interval": True,
    }


def _winner(joints: Sequence[str], stats_by_joint: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not joints:
        raise ProtocolError("winner requested from an empty selection")
    return dict(sorted(
        (stats_by_joint[str(joint)] for joint in joints),
        key=lambda row: (-float(row["validation_mean"]), str(row["joint_fingerprint"])),
    )[0])


def classify_comparison(
    per_replicate: Sequence[Mapping[str, Any]], curves: Sequence[Mapping[str, Any]],
    *, left: str, right: str, bootstrap_seed: int, primary: bool,
) -> dict[str, Any]:
    differences = [float(row[f"best_{left}"]) - float(row[f"best_{right}"]) for row in per_replicate]
    wins = sum(value > TOLERANCE for value in differences)
    losses = sum(value < -TOLERANCE for value in differences)
    ties = REPLICATES - wins - losses
    mean_difference = float(np.mean(differences))
    bootstrap = bootstrap_paired_mean(differences, seed=bootstrap_seed)
    auc_left = float(np.mean([float(row[f"{left}_anytime_AUC"]) for row in per_replicate]))
    auc_right = float(np.mean([float(row[f"{right}_anytime_AUC"]) for row in per_replicate]))
    budget_rows = []
    for budget in (4, 8, 12, 16):
        left_values = [float(row["best_so_far"]) for row in curves if row["method"] == left and int(row["budget"]) == budget]
        right_values = [float(row["best_so_far"]) for row in curves if row["method"] == right and int(row["budget"]) == budget]
        if len(left_values) != REPLICATES or len(right_values) != REPLICATES:
            raise ProtocolError(f"anytime curve coverage missing at budget {budget}")
        left_mean, right_mean = float(np.mean(left_values)), float(np.mean(right_values))
        budget_rows.append({
            "budget": budget, f"{left}_mean_best_so_far": left_mean,
            f"{right}_mean_best_so_far": right_mean,
            "left_strictly_higher": left_mean > right_mean + TOLERANCE,
        })
    higher_points = sum(bool(row["left_strictly_higher"]) for row in budget_rows)
    gates = {
        "1": {"passed": wins >= 8, "strict_wins": wins, "required": 8},
        "2": {"passed": mean_difference + TOLERANCE >= MINIMUM_GAIN, "mean_difference": mean_difference, "required_minimum": MINIMUM_GAIN, "tolerance": TOLERANCE},
        "3": {"passed": bootstrap["CI95_lower"] > TOLERANCE, "CI95_lower": bootstrap["CI95_lower"], "required": "> 0", "tolerance": TOLERANCE},
        "4": {"passed": auc_left > auc_right + TOLERANCE, f"{left}_mean_anytime_AUC": auc_left, f"{right}_mean_anytime_AUC": auc_right},
        "5": {"passed": higher_points >= 3, "higher_budget_points": higher_points, "required": 3, "budget_means": budget_rows},
    }
    return {
        "comparison": f"{left}_vs_{right}", "scientific_role": "primary" if primary else "secondary_replication",
        "wins": wins, "losses": losses, "ties": ties,
        "mean_final_best_gain": mean_difference, "median_final_best_gain": float(np.median(differences)),
        "differences": differences, "bootstrap": bootstrap,
        "mean_anytime_AUC_gain": auc_left - auc_right,
        "gates": gates, "all_five_gates_passed": all(row["passed"] for row in gates.values()),
        "historical_P_used": False, "observed_union_regret_used": False,
        "gate_modified_after_results": False, "test_accessed": False,
    }


def repeat_refinement_analysis(
    plans: Mapping[str, Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]],
    stats_by_joint: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_joint = {str(row["joint_fingerprint"]): row for row in candidates}
    output: dict[str, Any] = {}
    for method in (PURE, P2, RANDOM):
        refinement_rows = []
        winner_rows = []
        for replicate in range(REPLICATES):
            occurrences: dict[str, list[tuple[int, str]]] = defaultdict(list)
            selected_joints = []
            for round_row in plans[method]["rounds"]:
                if int(round_row["replicate_id"]) != replicate:
                    continue
                round_id = int(round_row["round_id"])
                for joint in map(str, round_row["selected_joint_fingerprints"]):
                    architecture = str(by_joint[joint]["architecture_fingerprint"])
                    occurrences[architecture].append((round_id, joint))
                    selected_joints.append(joint)
            first_appearance_joints = []
            later_repeat_joints = []
            for architecture, entries in sorted(occurrences.items()):
                first_round = min(round_id for round_id, _joint in entries)
                first = [joint for round_id, joint in entries if round_id == first_round]
                later = [joint for round_id, joint in entries if round_id > first_round]
                first_appearance_joints.extend(first)
                later_repeat_joints.extend(later)
                if later:
                    first_best = max(float(stats_by_joint[joint]["validation_mean"]) for joint in first)
                    later_best = max(float(stats_by_joint[joint]["validation_mean"]) for joint in later)
                    refinement_rows.append({
                        "method": method, "replicate_id": replicate,
                        "architecture_fingerprint": architecture, "first_round": first_round,
                        "FIRST_BEST": first_best, "LATER_REPEAT_BEST": later_best,
                        "REFINEMENT_GAIN": max(0.0, later_best - first_best),
                    })
            final = _winner(selected_joints, stats_by_joint)
            first_best_row = _winner(first_appearance_joints, stats_by_joint)
            later_best_row = _winner(later_repeat_joints, stats_by_joint) if later_repeat_joints else None
            winner_is_repeat = str(final["joint_fingerprint"]) in set(later_repeat_joints)
            winner_rows.append({
                "method": method, "replicate_id": replicate,
                "FINAL_BEST": float(final["validation_mean"]),
                "FINAL_WINNER_ORIGIN": "LATER_REPEAT" if winner_is_repeat else "FIRST_APPEARANCE",
                "BEST_FIRST_APPEARANCE": float(first_best_row["validation_mean"]),
                "BEST_LATER_REPEAT": None if later_best_row is None else float(later_best_row["validation_mean"]),
                "REPEAT_WINNER_GAIN": max(0.0, float(final["validation_mean"]) - float(first_best_row["validation_mean"])) if winner_is_repeat else 0.0,
            })
        gains = [float(row["REFINEMENT_GAIN"]) for row in refinement_rows]
        winner_gains = [float(row["REPEAT_WINNER_GAIN"]) for row in winner_rows]
        output[method] = {
            "definitions": {
                "FIRST_BEST": "best config in architecture first selected round",
                "LATER_REPEAT_BEST": "best config in a strictly later selected round",
                "REFINEMENT_GAIN": "max(0, LATER_REPEAT_BEST - FIRST_BEST)",
            },
            "refinement_rows": refinement_rows, "winner_rows": winner_rows,
            "summary": {
                "architectures_with_later_repeat": len(gains),
                "mean_refinement_gain": float(np.mean(gains)) if gains else None,
                "median_refinement_gain": float(np.median(gains)) if gains else None,
                "fraction_refinement_gain_positive": sum(value > TOLERANCE for value in gains) / len(gains) if gains else None,
                "fraction_refinement_gain_at_least_0_003": sum(value + TOLERANCE >= 0.003 for value in gains) / len(gains) if gains else None,
                "fraction_refinement_gain_at_least_0_006": sum(value + TOLERANCE >= 0.006 for value in gains) / len(gains) if gains else None,
                "repeat_winner_replicates": sum(row["FINAL_WINNER_ORIGIN"] == "LATER_REPEAT" for row in winner_rows),
                "mean_REPEAT_WINNER_GAIN": float(np.mean(winner_gains)),
                "median_REPEAT_WINNER_GAIN": float(np.median(winner_gains)),
            },
        }
    output["scientific_role"] = "interpretive_non_gating"
    output["test_accessed"] = False
    return output


def compute_search_analysis(
    config_stats: Sequence[Mapping[str, Any]], plans: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]], manipulation: Mapping[str, Any],
    information: Mapping[str, Any], training_audit: Mapping[str, Any],
) -> dict[str, Any]:
    stats_by_joint = {str(row["joint_fingerprint"]): dict(row) for row in config_stats}
    per_round = []
    curves = []
    per_replicate = []
    method_winners = {method: [] for method in METHODS}
    for replicate in range(REPLICATES):
        best_by_method: dict[str, float] = {}
        auc_by_method: dict[str, float] = {}
        winner_by_method: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            cumulative: list[str] = []
            anytime = []
            for round_id in range(ROUNDS):
                round_plan = next(
                    row for row in plans[method]["rounds"]
                    if int(row["replicate_id"]) == replicate and int(row["round_id"]) == round_id
                )
                joints = list(map(str, round_plan["selected_joint_fingerprints"]))
                cumulative.extend(joints)
                validation = [float(stats_by_joint[joint]["validation_mean"]) for joint in joints]
                best_so_far = max(float(stats_by_joint[joint]["validation_mean"]) for joint in cumulative)
                budget = (round_id + 1) * SELECTED_PER_ROUND
                per_round.append({
                    "replicate_id": replicate, "round_id": round_id, "budget": budget,
                    "method": method, "selected_joint_fingerprints": joints,
                    "selected_validation_means": validation, "round_best": max(validation),
                    "best_so_far": best_so_far,
                })
                curves.append({
                    "replicate_id": replicate, "method": method,
                    "budget": budget, "best_so_far": best_so_far,
                })
                anytime.append(best_so_far)
            winner = _winner(cumulative, stats_by_joint)
            winner.update({"replicate_id": replicate, "method": method})
            method_winners[method].append(winner)
            winner_by_method[method] = winner
            best_by_method[method] = float(winner["validation_mean"])
            auc_by_method[method] = float(np.mean(anytime))
        per_replicate.append({
            "replicate_id": replicate,
            **{f"best_{method}": best_by_method[method] for method in METHODS},
            **{f"{method}_winner_joint_fingerprint": winner_by_method[method]["joint_fingerprint"] for method in METHODS},
            **{f"{method}_anytime_AUC": auc_by_method[method] for method in METHODS},
            "D_P2_PURE": best_by_method[P2] - best_by_method[PURE],
            "D_P2_RANDOM": best_by_method[P2] - best_by_method[RANDOM],
            "D_PURE_RANDOM": best_by_method[PURE] - best_by_method[RANDOM],
            **{f"{method}_above_historical_P": best_by_method[method] > HISTORICAL_P_MEAN for method in METHODS},
        })
    p2_pure = classify_comparison(per_replicate, curves, left=P2, right=PURE, bootstrap_seed=0, primary=True)
    p2_random = classify_comparison(per_replicate, curves, left=P2, right=RANDOM, bootstrap_seed=1, primary=True)
    pure_random = classify_comparison(per_replicate, curves, left=PURE, right=RANDOM, bootstrap_seed=2, primary=False)
    intervention = bool(p2_pure["all_five_gates_passed"])
    efficacy = bool(p2_random["all_five_gates_passed"])
    if not intervention:
        classification = "DIVERSITY_SEARCH_NOT_CONFIRMED"
    elif not efficacy:
        classification = "DIVERSITY_MECHANISM_CONFIRMED_NO_SEARCH_PROMOTION"
    else:
        classification = "DIVERSITY_SEARCH_PROMOTED"
    oracle_rows = []
    for row in per_replicate:
        oracle = max(float(row[f"best_{method}"]) for method in METHODS)
        oracle_rows.append({
            "replicate_id": row["replicate_id"],
            "OBSERVED_EVALUATED_UNION_ORACLE_r": oracle,
            **{f"observed_union_regret_{method}": oracle - float(row[f"best_{method}"]) for method in METHODS},
        })
    by_joint_candidate = {str(row["joint_fingerprint"]): row for row in candidates}
    p2_only = []
    p2_new_architecture = []
    for replicate in range(REPLICATES):
        pure_joints = {
            str(joint) for round_row in plans[PURE]["rounds"] if int(round_row["replicate_id"]) == replicate
            for joint in round_row["selected_joint_fingerprints"]
        }
        p2_joints = {
            str(joint) for round_row in plans[P2]["rounds"] if int(round_row["replicate_id"]) == replicate
            for joint in round_row["selected_joint_fingerprints"]
        }
        pure_architectures = {str(by_joint_candidate[joint]["architecture_fingerprint"]) for joint in pure_joints}
        p2_only.extend(float(stats_by_joint[joint]["validation_mean"]) for joint in p2_joints - pure_joints)
        p2_new_architecture.extend(
            float(stats_by_joint[joint]["validation_mean"])
            for joint in p2_joints if str(by_joint_candidate[joint]["architecture_fingerprint"]) not in pure_architectures
        )
    architecture_results = {
        "pre_registered_manipulation": manipulation,
        "per_replicate_architecture_coverage": {
            method: [
                {"replicate_id": row["replicate_id"], "unique_architecture_count": row["unique_architecture_count"], "effective_architecture_count": row["effective_architecture_count"]}
                for row in manipulation["methods"][method]["per_replicate"]
            ] for method in METHODS
        },
        "P2_only_joint_validation_distribution": _numeric_summary(p2_only),
        "P2_architecture_unseen_by_PURE_in_replicate_validation_distribution": _numeric_summary(p2_new_architecture),
        "posthoc_metrics_used_in_classification": False,
        "test_accessed": False,
    }
    repeat = repeat_refinement_analysis(plans, candidates, stats_by_joint)
    decision = {
        "classification": classification,
        "DIVERSITY_INTERVENTION_CONFIRMED": intervention,
        "DIVERSITY_SEARCH_EFFICACY_CONFIRMED": efficacy,
        "A_gate": p2_pure["gates"], "B_gate": p2_random["gates"],
        "information_gate_passed": information["passed"],
        "historical_P_used_in_primary_gates": False,
        "observed_union_regret_used_in_primary_gates": False,
        "gate_modified_after_results": False,
        "fresh_seed_confirmation_started": False,
        "test_accessed": False,
    }
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise ProtocolError(f"unregistered classification: {classification}")
    bootstrap = {
        "P2_vs_PURE": p2_pure["bootstrap"],
        "P2_vs_RANDOM": p2_random["bootstrap"],
        "PURE_vs_RANDOM_secondary": pure_random["bootstrap"],
    }
    return {
        "per_round": per_round, "per_replicate": per_replicate,
        "anytime_curves": curves, "p2_vs_pure": p2_pure,
        "p2_vs_random": p2_random, "pure_vs_random": pure_random,
        "bootstrap": bootstrap,
        "observed_union_regret": {
            "oracle_name": "OBSERVED_EVALUATED_UNION_ORACLE_r",
            "oracle_scope": "PURE/P2/Random actually evaluated configuration union only",
            "is_complete_128_candidate_pool_oracle": False,
            "scientific_role": "descriptive_only", "primary_gate_use": False,
            "rows": oracle_rows,
        },
        "architecture_diversity_results": architecture_results,
        "repeat_refinement": repeat, "decision": decision,
        "method_winners": method_winners,
        "historical_P": {
            "role": "DESCRIPTIVE_HISTORICAL_REFERENCE",
            "joint_fingerprint": HISTORICAL_P_JOINT,
            "validation_seed_count": 30, "validation_mean": HISTORICAL_P_MEAN,
            "source_path": str(HISTORICAL_P_SOURCE),
            "source_sha256": HISTORICAL_P_SOURCE_SHA256,
            "replicate_point_estimates_above": {
                method: sum(bool(row[f"{method}_above_historical_P"]) for row in per_replicate)
                for method in METHODS
            },
            "highest_3seed_winner": {
                method: max(float(row["validation_mean"]) for row in method_winners[method])
                for method in METHODS
            },
            "formal_inference_made": False,
        },
        "training_audit": dict(training_audit),
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "Q25": None, "Q75": None, "best": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values), "mean": float(np.mean(array)), "median": float(np.median(array)),
        "Q25": float(np.quantile(array, 0.25)), "Q75": float(np.quantile(array, 0.75)),
        "best": float(np.max(array)),
    }


def write_search_outputs(output: Path, analysis: Mapping[str, Any]) -> None:
    dump_csv(output / "search/per_round_results.csv", analysis["per_round"])
    dump_csv(output / "search/per_replicate_results.csv", analysis["per_replicate"])
    dump_csv(output / "search/anytime_curves.csv", analysis["anytime_curves"])
    dump_json(output / "search/p2_vs_pure.json", analysis["p2_vs_pure"])
    dump_json(output / "search/p2_vs_random.json", analysis["p2_vs_random"])
    dump_json(output / "search/pure_vs_random.json", analysis["pure_vs_random"])
    dump_json(output / "search/bootstrap_results.json", analysis["bootstrap"])
    dump_json(output / "search/observed_union_regret.json", analysis["observed_union_regret"])
    dump_json(output / "search/architecture_diversity_results.json", analysis["architecture_diversity_results"])
    dump_json(output / "search/repeat_refinement.json", analysis["repeat_refinement"])


def select_top3_winners(winners: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for row in sorted(
        (dict(value) for value in winners),
        key=lambda value: (-float(value["validation_mean"]), str(value["joint_fingerprint"])),
    ):
        joint = str(row["joint_fingerprint"])
        if joint in seen:
            continue
        seen.add(joint)
        selected.append(row)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ProtocolError("could not freeze three unique replicate winners")
    return selected


def write_next_stage(
    output: Path, classification: str,
    method_winners: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    next_stage = output / "next_stage"
    next_stage.mkdir(parents=True, exist_ok=True)
    if classification != "DIVERSITY_SEARCH_PROMOTED":
        return {
            "frozen": False, "reason": "classification is not DIVERSITY_SEARCH_PROMOTED",
            "fresh_seed_confirmation_started": False,
        }
    names = {P2: "p2", PURE: "pure_b4", RANDOM: "random"}
    freeze_files = []
    top3_payload: dict[str, Any] = {}
    for method in METHODS:
        prefix = names[method]
        winners = [dict(row) for row in method_winners[method]]
        if len(winners) != REPLICATES:
            raise ProtocolError(f"{method} does not have 12 replicate winners")
        csv_path = next_stage / f"{prefix}_replicate_winners.csv"
        dump_csv(csv_path, winners)
        top3 = select_top3_winners(winners)
        top3_payload[method] = top3
        top_path = next_stage / f"{prefix}_top3.json"
        dump_json(top_path, {
            "method": method, "selection_rule": "validation_mean_descending_then_joint_fingerprint_ascending",
            "unique_joint_configurations": True, "winners": top3,
            "scientific_role": "fresh_seed_confirmation_candidates_only",
        })
        freeze_files.extend([csv_path, top_path])
    historical_path = next_stage / "historical_P_reference.json"
    dump_json(historical_path, {
        "role": "DESCRIPTIVE_HISTORICAL_REFERENCE",
        "joint_fingerprint": HISTORICAL_P_JOINT,
        "validation_seed_count": 30, "validation_mean": HISTORICAL_P_MEAN,
        "source_path": str(HISTORICAL_P_SOURCE),
        "source_sha256": HISTORICAL_P_SOURCE_SHA256,
        "formal_comparison_in_search_stage": False,
    })
    seed_path = next_stage / "fresh_seed_spec.json"
    fresh_seeds = list(range(1000, 1030))
    if set(fresh_seeds) & set(TRAINING_SEEDS):
        raise ProtocolError("fresh seed specification overlaps search seeds")
    dump_json(seed_path, {
        "training_seeds": fresh_seeds, "count": 30,
        "matched_across": ["P2_top3", "PURE_B4_top3", "Random_top3", "historical_P"],
        "executed_in_this_run": False, "GNN_runs_started": 0, "test_accessed": False,
    })
    freeze_files.extend([historical_path, seed_path])
    pairs = [
        (path.relative_to(next_stage).as_posix(), sha256_file(path))
        for path in sorted(freeze_files, key=lambda value: value.relative_to(next_stage).as_posix())
    ]
    root = stable_hash(pairs)
    dump_text(next_stage / "winner_freeze_root_sha256.txt", root + "\n")
    dump_text(next_stage / "NEXT_STAGE_WINNERS_FROZEN", root + "\n")
    return {
        "frozen": True, "root_sha256": root, "files": pairs,
        "top3": top3_payload, "fresh_training_seeds": fresh_seeds,
        "fresh_seed_confirmation_started": False,
    }


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def render_final_report(artifacts: Mapping[str, Any]) -> str:
    analysis = artifacts["analysis"]
    decision = analysis["decision"]
    manipulation = artifacts["manipulation"]
    overlap = artifacts["overlap"]
    generation = artifacts["generation"]
    training = artifacts["training_audit"]
    replicate_fields = (
        "replicate_id", f"best_{PURE}", f"best_{P2}", f"best_{RANDOM}",
        "D_P2_PURE", "D_P2_RANDOM", "D_PURE_RANDOM",
    )
    detail_lines = []
    for replicate in range(REPLICATES):
        detail_lines.append(f"### Replicate {replicate}")
        for method in METHODS:
            detail_lines.append(f"\n**{method}**")
            for row in analysis["per_round"]:
                if int(row["replicate_id"]) == replicate and row["method"] == method:
                    values = ", ".join(f"{value:.9f}" for value in row["selected_validation_means"])
                    detail_lines.append(
                        f"- round {row['round_id']}: four 3-seed validation estimates [{values}]; "
                        f"best after {row['budget']} = {row['best_so_far']:.9f}"
                    )
        rep = analysis["per_replicate"][replicate]
        detail_lines.append(
            f"\nFinal: P2={rep[f'best_{P2}']:.9f}, PURE={rep[f'best_{PURE}']:.9f}, "
            f"Random={rep[f'best_{RANDOM}']:.9f}; P2-PURE={rep['D_P2_PURE']:.9f}; "
            f"P2-Random={rep['D_P2_RANDOM']:.9f}."
        )
    allocation = {
        method: manipulation["methods"][method]["overall"] for method in METHODS
    }
    section_text: dict[str, str] = {
        REPORT_SECTIONS[0]: f"Frozen classification: **{decision['classification']}**.",
        REPORT_SECTIONS[1]: "All 6,144 candidates, B4 scores, and three allocation plans were frozen on new pools before labels. P2 and PURE differ only by the preregistered architecture constraint.",
        REPORT_SECTIONS[2]: "The collapse diagnosis authorized B4 + P2 planning but supplied no labels, choices, or thresholds to this prospective experiment. Its post-hoc mechanism claim is not reused as confirmation.",
        REPORT_SECTIONS[3]: f"B4 replay status: {artifacts['replay']['status']}; historical OOF=336, earlier confirmation=64, previous B4-search predictions=4,096; maximum absolute replay error={artifacts['replay']['maximum_absolute_error']}. Deterministic historical-only reconstruction is identity replay, not a label update.",
        REPORT_SECTIONS[4]: "P2 is ROUND_UNIQUE_CAP2: four architectures per round and at most two selected slots per architecture within a replicate, scanning the unchanged B4 ranking with joint-fingerprint tie-break.",
        REPORT_SECTIONS[5]: "Candidates use the frozen model-blind CPU VAE decode and HIGH_REGIME_PRIOR raw HP uniform range [0.80, 0.95], candidate seed 202608310001+r. No scorer or label enters proposal generation.",
        REPORT_SECTIONS[6]: f"All {generation['coverage']['unique_joint_count']} joint fingerprints are unique and outside the authenticated 8,624-joint exclusion union.",
        REPORT_SECTIONS[7]: f"Unique architectures overall: {generation['coverage']['unique_architecture_count']}; per-pool unique architecture median: {generation['coverage']['per_pool_unique_architectures_median']}.",
        REPORT_SECTIONS[8]: f"Pre-search root: `{artifacts['pre_search_root']}`. It was reverified before task creation, evaluation, and analysis.",
        REPORT_SECTIONS[9]: f"P2 feasibility passed={artifacts['feasibility']['passed']}; legal rounds={artifacts['feasibility']['legal_rounds']}/48; observed replicate cap={artifacts['feasibility']['maximum_observed_architecture_slots_per_replicate']}.",
        REPORT_SECTIONS[10]: f"Information gate passed={artifacts['information']['passed']}; replicates with divergence >=4={artifacts['information']['replicates_with_divergence_at_least_4']}/12; median divergence={artifacts['information']['median_divergence']}.",
        REPORT_SECTIONS[11]: f"Method-equivalent budget is 576 config slots / 1,728 seed runs. Physical union is {overlap['overall']['selected_union_unique_configurations']} unique configs / {overlap['overall']['physical_unique_GNN_runs']} seed runs.",
        REPORT_SECTIONS[12]: f"Evaluation complete={training['evaluation_complete']}; completed physical runs={training['completed_gnn_runs']}/{training['task_count']}; failed={training['failed_gnn_runs']}; test_accessed=false.",
        REPORT_SECTIONS[13]: json.dumps(allocation[PURE], sort_keys=True),
        REPORT_SECTIONS[14]: json.dumps(allocation[P2], sort_keys=True),
        REPORT_SECTIONS[15]: json.dumps(allocation[RANDOM], sort_keys=True),
        REPORT_SECTIONS[16]: json.dumps(manipulation["PURE_vs_P2"], sort_keys=True),
        REPORT_SECTIONS[17]: _markdown_table(analysis["per_replicate"], replicate_fields) + "\n\n" + "\n".join(detail_lines),
        REPORT_SECTIONS[18]: json.dumps(analysis["p2_vs_pure"], sort_keys=True),
        REPORT_SECTIONS[19]: json.dumps(analysis["p2_vs_random"], sort_keys=True),
        REPORT_SECTIONS[20]: json.dumps(analysis["pure_vs_random"], sort_keys=True),
        REPORT_SECTIONS[21]: json.dumps(analysis["bootstrap"], sort_keys=True),
        REPORT_SECTIONS[22]: _markdown_table(analysis["anytime_curves"], ("replicate_id", "method", "budget", "best_so_far")),
        REPORT_SECTIONS[23]: "AUC is the arithmetic mean of best-so-far after budgets 4, 8, 12, and 16. " + json.dumps({name: analysis[name]["mean_anytime_AUC_gain"] for name in ("p2_vs_pure", "p2_vs_random", "pure_vs_random")}, sort_keys=True),
        REPORT_SECTIONS[24]: json.dumps(analysis["repeat_refinement"], sort_keys=True),
        REPORT_SECTIONS[25]: json.dumps(analysis["historical_P"], sort_keys=True) + " These 3-seed point estimates are descriptive and do not establish superiority to the 30-seed P reference.",
        REPORT_SECTIONS[26]: json.dumps(decision["A_gate"], sort_keys=True),
        REPORT_SECTIONS[27]: json.dumps(decision["B_gate"], sort_keys=True),
        REPORT_SECTIONS[28]: decision["classification"],
        REPORT_SECTIONS[29]: f"Diversity intervention confirmed versus PURE B4: {decision['DIVERSITY_INTERVENTION_CONFIRMED']}.",
        REPORT_SECTIONS[30]: f"Search efficacy confirmed versus Random: {decision['DIVERSITY_SEARCH_EFFICACY_CONFIRMED']}; promotion requires both primary gates.",
        REPORT_SECTIONS[31]: "Stable absolute accuracy improvement has not been proven. Search winners are only 3-seed validation estimates; 30 fresh matched seeds are a separate authorized stage only after promotion.",
        REPORT_SECTIONS[32]: json.dumps(artifacts["next_stage"], sort_keys=True),
        REPORT_SECTIONS[33]: "Only the evaluated union is labelled; there is no full 128-candidate oracle. Winner selection and mechanism summaries remain subject to 3-seed noise. Historical P is not a primary comparator.",
        REPORT_SECTIONS[34]: f"track_test=false; test_accessed=false; candidate replacement=false; matched seeds={list(TRAINING_SEEDS)}; source snapshot and manifests preserve actual bytes; classification gates were frozen before evaluation.",
    }
    chunks = ["# B4 Architecture-Diversity Prospective Search Accuracy Gate v1"]
    for heading in REPORT_SECTIONS:
        chunks.extend(["", f"## {heading}", "", section_text[heading]])
    return "\n".join(chunks).rstrip() + "\n"


def assert_complete_final_report(text: str) -> None:
    missing = [heading for heading in REPORT_SECTIONS if f"## {heading}" not in text]
    if missing:
        raise ProtocolError(f"final report is missing sections: {missing}")


def _source_identity_paths() -> dict[str, Path]:
    paths = {
        relative: REPO_ROOT / relative for relative in IMPLEMENTATION_FILES
    }
    paths.update({
        "frozen/base_b4_search_config.yaml": base.DEFAULT_CONFIG,
        "frozen/base_b4_search_source.py": Path(base.__file__).resolve(),
        "frozen/archived_b4_search_source.py": OLD_B4_ROOT / "source_snapshot/analyse/b4_guided_search_accuracy_gate.py",
        "frozen/p2_diagnosis_source.py": DIAGNOSIS_ROOT / "source_snapshot/analyse/cross_method_search_collapse_diagnosis.py",
        "frozen/direct_joint_regressor_source.py": base.SOFT_SNAPSHOT / "analyse/joint_configuration_metric_gate.py",
        "frozen/soft_metric_source.py": base.SOFT_SNAPSHOT / "analyse/joint_configuration_soft_metric_gate.py",
        "frozen/high_regime_generator.py": base.HIGH_SNAPSHOT / "analyse/high_regime_prior_model_blind_validation.py",
        "frozen/high_regime_evaluator.py": base.HIGH_SNAPSHOT / "analyse/run_high_regime_prior_model_blind_validation_labels.py",
        "frozen/eval_utils.py": base.HIGH_SNAPSHOT / "eval_utils.py",
        "frozen/architecture_diagnostics.py": base.HIGH_SNAPSHOT / "architecture_diagnostics.py",
        "hp_modes.py": REPO_ROOT / "hp_modes.py",
        "dataset_utils.py": REPO_ROOT / "dataset_utils.py",
        "models.py": REPO_ROOT / "models.py",
        "authoritative/historical_P_reference.json": HISTORICAL_P_SOURCE,
    })
    return paths


def source_identity_entries() -> list[dict[str, Any]]:
    entries = []
    for relative, path in sorted(_source_identity_paths().items()):
        if not path.is_file():
            raise ProtocolError(f"required source file is missing: {path}")
        entries.append({"relative_path": relative, "path": str(path), "sha256": sha256_file(path)})
    for index, row in enumerate(base._frozen_dependency_entries()):
        path = Path(str(row["path"])).resolve()
        if not path.is_file() or sha256_file(path) != str(row["sha256"]):
            raise ProtocolError(f"frozen evaluator dependency identity mismatch: {path}")
        entries.append({
            "relative_path": f"frozen/evaluator_dependency_{index:02d}/{path.name}",
            "path": str(path), "sha256": str(row["sha256"]),
        })
    return sorted(entries, key=lambda row: str(row["relative_path"]))


def aggregate_source_sha256() -> str:
    return stable_hash([
        {key: row[key] for key in ("relative_path", "sha256")}
        for row in source_identity_entries()
    ])


def _module_source_path(module: Any) -> Path | None:
    raw = getattr(module, "__file__", None)
    if not raw:
        return None
    path = Path(raw).resolve()
    if path.suffix in {".pyc", ".pyo"}:
        candidate = Path(str(path)[:-1])
        if candidate.is_file():
            path = candidate
    try:
        path.relative_to(REPO_ROOT)
    except ValueError:
        return None
    return path if path.is_file() and path.suffix == ".py" else None


def snapshot_sources(output: Path, *, best_effort: bool = False) -> dict[str, Any]:
    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    declared_rows = []
    try:
        entries = source_identity_entries()
    except Exception as error:
        if not best_effort:
            raise
        entries = []
        declared_rows.append({"status": "INCOMPLETE", "reason": f"{type(error).__name__}: {error}"})
        for relative, path in sorted(_source_identity_paths().items()):
            if path.is_file():
                entries.append({"relative_path": relative, "path": str(path), "sha256": sha256_file(path)})
    for row in entries:
        source = Path(str(row["path"]))
        destination = snapshot / "declared" / str(row["relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        observed = sha256_file(destination)
        if observed != row["sha256"]:
            raise ProtocolError(f"source snapshot byte mismatch: {source}")
        declared_rows.append({
            "relative_path": destination.relative_to(snapshot).as_posix(),
            "source_path": str(source), "sha256": observed, "actual_bytes_saved": True,
        })
    imported = []
    seen_paths: set[Path] = set()
    for module_name, module in sorted(sys.modules.items()):
        source = _module_source_path(module)
        if source is None:
            continue
        imported.append({"module": module_name, "source_path": str(source), "sha256": sha256_file(source)})
        if source in seen_paths:
            continue
        seen_paths.add(source)
        relative = source.relative_to(REPO_ROOT)
        destination = snapshot / "imported_modules" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise ProtocolError(f"imported module snapshot byte mismatch: {source}")
    dump_json(snapshot / "imported_local_modules.json", {
        "repo_root": str(REPO_ROOT), "module_count": len(imported), "modules": imported,
    })
    all_rows = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            all_rows.append({
                "relative_path": path.relative_to(snapshot).as_posix(),
                "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
    payload = {
        "protocol_version": PROTOCOL_VERSION, "created_at": utc_now(),
        "declared_sources": declared_rows, "imported_local_module_count": len(imported),
        "files": all_rows,
        "manifest_root_sha256": stable_hash([(row["relative_path"], row["sha256"]) for row in all_rows]),
        "actual_bytes_saved": True,
    }
    dump_json(snapshot / "sha256_manifest.json", payload)
    return payload


def write_environment_and_git(output: Path, *, before: bool) -> None:
    if before:
        environment = {
            "created_at": utc_now(), "platform": platform.platform(),
            "python": sys.version, "python_executable": sys.executable,
            "cwd": str(Path.cwd()), "repo_root": str(REPO_ROOT),
            "branch": _git_capture("branch", "--show-current").strip(),
            "HEAD": _git_capture("rev-parse", "HEAD").strip(),
            "dirty_tree": bool(_git_capture("status", "--porcelain").strip()),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "B4_SEARCH_SCORER_DEVICE": os.environ.get("B4_SEARCH_SCORER_DEVICE", "cpu"),
            "track_test": False, "test_accessed": False,
        }
        dump_json(output / "environment.json", environment)
        dump_text(output / "git_status_before.txt", _git_capture("status", "--porcelain=v1", "--branch"))
    else:
        dump_text(output / "git_status_after.txt", _git_capture("status", "--porcelain=v1", "--branch"))
        dump_text(output / "git_diff.patch", _git_diff_patch())


def _copy_preflight_logs(output: Path, preflight: Path | None) -> None:
    mapping = {
        "py_compile.log": "py_compile.log", "pytest_focused.log": "pytest_focused.log",
        "synthetic_smoke.log": "synthetic_smoke.log", "dry_run.log": "dry_run.log",
    }
    for source_name, target_name in mapping.items():
        source = None if preflight is None else preflight / source_name
        destination = output / "logs" / target_name
        if source is not None and source.is_file():
            shutil.copy2(source, destination)
        elif not destination.exists():
            dump_text(destination, "preflight log unavailable\n")


def _logger(path: Path) -> Callable[[str], None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now()}] {message}\n")
            handle.flush()

    return log


def ensure_output_policy(output: Path, archive: Path) -> None:
    sidecar = Path(str(archive) + ".sha256")
    for target in (output, archive, sidecar):
        if target.exists() or target.is_symlink():
            raise ProtocolError(f"formal output target already exists: {target}")
    try:
        output.resolve().relative_to((REPO_ROOT / "results/posthoc").resolve())
    except ValueError as error:
        raise ProtocolError("formal output must remain under results/posthoc") from error


def write_manifest(output: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rows.append({
                "relative_path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
    payload = {
        "protocol_version": PROTOCOL_VERSION, "created_at": utc_now(),
        "file_count": len(rows), "files": rows,
        "manifest_root_sha256": stable_hash([(row["relative_path"], row["sha256"]) for row in rows]),
    }
    dump_json(output / "manifest.json", payload)
    return payload


def create_archive(output: Path, archive: Path) -> str:
    sidecar = Path(str(archive) + ".sha256")
    if archive.exists() or sidecar.exists():
        raise ProtocolError(f"archive target already exists: {archive}")
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    with tarfile.open(temporary, "w:gz") as handle:
        handle.add(output, arcname=output.name)
    digest = sha256_file(temporary)
    os.replace(temporary, archive)
    dump_text(sidecar, f"{digest}  {archive.name}\n")
    return digest


def _placeholder(classification: str, reason: str | None) -> dict[str, Any]:
    return {"status": "NOT_AVAILABLE", "classification": classification, "reason": reason, "test_accessed": False}


def ensure_required_outputs(output: Path, classification: str, reason: str | None = None) -> None:
    for relative in REQUIRED_OUTPUTS:
        path = output / relative
        if path.exists() or relative in {
            "pre_search/PRE_SEARCH_FROZEN", "pre_search/freeze_manifest.json",
            "pre_search/pre_search_root_sha256.txt", "source_snapshot/sha256_manifest.json",
            "source_snapshot/imported_local_modules.json", "manifest.json",
            "final_report.md", "state.json", "frozen_protocol.json", "decision.json",
            "environment.json", "git_status_before.txt", "git_status_after.txt", "git_diff.patch",
        }:
            continue
        if path.suffix == ".json":
            dump_json(path, _placeholder(classification, reason))
        elif path.suffix == ".csv":
            dump_csv(path, [], ("status",))
        elif path.suffix in {".log", ".txt"}:
            dump_text(path, f"{classification}: {reason or ''}\n")


def finalize_bundle(
    output: Path, archive: Path, classification: str, artifacts: Mapping[str, Any],
    *, reason: str | None = None, run_failed: bool = False,
) -> str:
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise ProtocolError(f"cannot finalize unregistered classification: {classification}")
    decision_path = output / "decision.json"
    if not decision_path.is_file():
        dump_json(decision_path, {
            "classification": classification, "reason": reason,
            "fresh_seed_confirmation_started": False, "test_accessed": False,
        })
    state = update_state(
        output, stage="ANALYSIS_COMPLETE" if not run_failed else "RUN_FAILED",
        classification=classification, current_replicate=None, current_round=None,
        current_config=None, current_seed=None, test_accessed=False,
    )
    ensure_required_outputs(output, classification, reason)
    if not (output / "final_report.md").is_file():
        dump_text(output / "final_report.md", (
            f"# B4 Architecture-Diversity Prospective Search Accuracy Gate v1 — {classification}\n\n"
            f"The frozen experiment terminated before full analysis. Reason: {reason}. "
            "No test data were accessed.\n"
        ))
    write_environment_and_git(output, before=False)
    snapshot_sources(output, best_effort=classification in {"INPUT_IDENTITY_INVALID", "RUN_FAILED"})
    marker = "FAILED" if run_failed else "COMPLETE"
    dump_text(output / marker, classification + "\n")
    write_manifest(output)
    digest = create_archive(output, archive)
    state["archive_sha256"] = digest
    return digest


def _synthetic_candidates() -> list[dict[str, Any]]:
    rows = []
    for replicate in range(REPLICATES):
        for round_id in range(ROUNDS):
            for slot in range(CANDIDATES_PER_ROUND):
                architecture_index = 0 if slot < 8 else slot - 7
                rows.append({
                    "replicate_id": replicate, "round_id": round_id,
                    "candidate_slot": slot,
                    "joint_fingerprint": f"synthetic-j-{replicate:02d}-{round_id}-{slot:03d}",
                    "joint_config_fingerprint": f"synthetic-j-{replicate:02d}-{round_id}-{slot:03d}",
                    "architecture_fingerprint": f"synthetic-a-{replicate:02d}-{architecture_index:03d}",
                    "hp_fingerprint": f"synthetic-h-{replicate:02d}-{round_id}-{slot:03d}",
                    "raw_hp_vector": [0.9, 0.9, 0.9, 0.9],
                    "raw_hp": {"lr": 0.01, "dropout": 0.5, "hidden_dim": 128, "l2": 0.0005},
                    "operations": ["GCNConv"], "edges": [[0, 1]], "effective_layers": 1,
                    "lr": 0.01, "dropout": 0.5, "hidden_dim": 128, "l2": 0.0005,
                    "gat_heads": 1, "sage_aggr": "mean", "gin_eps": 0.0,
                    "gat_heads_by_layer": [1], "sage_aggr_by_layer": ["mean"],
                    "gin_eps_by_layer": [0.0], "generation_seed": CANDIDATE_MASTER_SEED + replicate,
                    "generation_draw_index": round_id * CANDIDATES_PER_ROUND + slot,
                    "proposal_distribution": "MODEL_BLIND_ARCHITECTURE_PLUS_HIGH_REGIME_PRIOR",
                    "config_id": f"synthetic-{replicate}-{round_id}-{slot}",
                })
    return rows


def _synthetic_comparison_case() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_replicate = []
    curves = []
    for replicate in range(REPLICATES):
        pure_best = 0.780 + replicate * 0.0001
        p2_best = pure_best + 0.005
        random_best = pure_best - 0.004
        row = {"replicate_id": replicate}
        for method, best, auc in (
            (PURE, pure_best, pure_best - 0.002),
            (P2, p2_best, p2_best - 0.001),
            (RANDOM, random_best, random_best - 0.002),
        ):
            row[f"best_{method}"] = best
            row[f"{method}_anytime_AUC"] = auc
            for budget_index, budget in enumerate((4, 8, 12, 16)):
                curves.append({
                    "replicate_id": replicate, "method": method, "budget": budget,
                    "best_so_far": best - 0.003 + 0.001 * budget_index,
                })
        per_replicate.append(row)
    return per_replicate, curves


def synthetic_smoke(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    validate_frozen_protocol(config)
    formal_before = DEFAULT_OUTPUT.exists()
    candidates = _synthetic_candidates()
    predictions = [
        {
            "replicate_id": row["replicate_id"], "round_id": row["round_id"],
            "candidate_slot": row["candidate_slot"], "joint_fingerprint": row["joint_fingerprint"],
            "architecture_fingerprint": row["architecture_fingerprint"],
            "b4_prediction": 1.0 - 0.001 * int(row["candidate_slot"]),
        }
        for row in candidates
    ]
    plans, selected, feasibility, selection_audits = select_search_plans(candidates, predictions)
    manipulation = diversity_manipulation_audit(plans, candidates)
    per_replicate, curves = _synthetic_comparison_case()
    comparison_a = classify_comparison(per_replicate, curves, left=P2, right=PURE, bootstrap_seed=0, primary=True)
    comparison_b = classify_comparison(per_replicate, curves, left=P2, right=RANDOM, bootstrap_seed=1, primary=True)
    with tempfile.TemporaryDirectory(prefix="b4-diversity-smoke-") as temporary:
        pre = Path(temporary) / "pre_search"
        freeze_pre_search(
            pre, candidates, predictions, plans,
            selection_audits["selection_overlap"], manipulation,
            selection_audits["information_gate"], feasibility,
            {"status": "SYNTHETIC", "input_hashes": {}, "source_hashes": {}},
            {"status": "SYNTHETIC"}, {"status": "SYNTHETIC"},
        )
        freeze_verified = verify_pre_search_freeze(pre)
    p2_counts = Counter(
        (int(row["replicate_id"]), architecture)
        for row in plans[P2]["rounds"]
        for architecture in row["selected_architecture_fingerprints"]
    )
    passed = all((
        len(candidates) == TOTAL_CANDIDATES,
        all(len(values) == SLOTS_PER_METHOD for values in selected.values()),
        feasibility["legal_rounds"] == 48,
        feasibility["maximum_observed_architecture_slots_per_replicate"] <= 2,
        max(p2_counts.values()) <= 2,
        selection_audits["information_gate"]["passed"],
        manipulation["methods"][P2]["overall"]["within_round_duplicate_fraction"] == 0.0,
        comparison_a["all_five_gates_passed"], comparison_b["all_five_gates_passed"],
        freeze_verified["passed"], not formal_before or DEFAULT_OUTPUT.exists(),
        DEFAULT_OUTPUT.exists() == formal_before,
    ))
    return {
        "passed": passed, "candidate_count": len(candidates),
        "P2_feasible_rounds": feasibility["legal_rounds"],
        "information_gate_passed": selection_audits["information_gate"]["passed"],
        "promoted_A_case": comparison_a["all_five_gates_passed"],
        "promoted_B_case": comparison_b["all_five_gates_passed"],
        "pre_search_root_recomputed": freeze_verified["passed"],
        "formal_candidate_generation_started": False,
        "formal_search_plan_frozen": False, "GNN_training_started": False,
        "CUDA_initialized": False, "formal_output_created": DEFAULT_OUTPUT.exists() != formal_before,
        "test_accessed": False,
    }


def execute_dry_run(config_path: Path, formal_output: Path) -> int:
    existed_before = formal_output.exists()
    config = load_config(config_path)
    protocol = validate_frozen_protocol(config)
    static = static_input_audit(config)
    payload = {
        "status": "DRY_RUN_READY", "protocol": protocol,
        "input_file_identity": static["status"],
        "planned_candidate_count": TOTAL_CANDIDATES,
        "planned_PURE_B4_slots": SLOTS_PER_METHOD,
        "planned_P2_slots": SLOTS_PER_METHOD,
        "planned_random_slots": SLOTS_PER_METHOD,
        "maximum_physical_GNN_runs": 1728,
        "method_equivalent_GNN_runs": 1728,
        "authorization_replay_started": False, "B4_replay_started": False,
        "formal_candidate_generation_started": False,
        "formal_search_plan_frozen": False, "GNN_training_started": False,
        "formal_output_existed_before": existed_before,
        "formal_output_created": formal_output.exists() and not existed_before,
        "test_accessed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _terminal_without_evaluation(output: Path, reason: str) -> dict[str, Any]:
    task_manifest = {
        "authorized": False, "reason": reason, "tasks": [],
        "selected_unique_config_count": 0, "planned_GNN_runs": 0,
        "method_equivalent_config_slots": 576,
        "method_equivalent_GNN_runs": 1728, "training_seeds": list(TRAINING_SEEDS),
        "track_test": False, "test_accessed": False,
    }
    audit = {
        "status": "NOT_AUTHORIZED", "reason": reason,
        "evaluation_complete": False, "selected_unique_config_count": 0,
        "physical_unique_GNN_runs": 0, "method_equivalent_GNN_runs": 1728,
        "task_count": 0, "completed_gnn_runs": 0, "failed_gnn_runs": 0,
        "candidate_replacement": False, "track_test": False, "test_accessed": False,
    }
    dump_json(output / "evaluation/task_manifest.json", task_manifest)
    dump_json(output / "evaluation/training_audit.json", audit)
    return audit


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
    artifacts: dict[str, Any] = {"test_accessed": False}

    try:
        input_integrity, authorization = validate_authoritative_inputs(config)
        dump_json(output / "input_integrity.json", input_integrity)
        dump_json(output / "authorization_replay.json", authorization)
        artifacts.update({"input_integrity": input_integrity, "authorization": authorization})
    except Exception as error:
        if isinstance(error, (InputIdentityInvalid, base.InputIdentityInvalid)) or type(error).__name__ in {
            "ProspectiveInputInvalid", "ConfirmationInputInvalid", "InputIdentityMismatch",
        }:
            reason = f"{type(error).__name__}: {error}"
            dump_json(output / "input_integrity.json", {"status": "INVALID", "reason": reason, "test_accessed": False})
            dump_json(output / "authorization_replay.json", {"status": "NOT_REPLAYED", "reason": reason, "test_accessed": False})
            _terminal_without_evaluation(output, reason)
            finalize_bundle(output, archive, "INPUT_IDENTITY_INVALID", artifacts, reason=reason)
            return 0
        raise

    try:
        os.environ.update({
            "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "B4_SEARCH_SCORER_DEVICE": "cpu",
        })
        runtime = base.load_frozen_runtime()
        historical_rows, folds, historical_inventory = base.load_historical_folds(runtime)
        replay = replay_frozen_b4(runtime, folds, scorer_log)
        dump_json(output / "b4_scorer_replay.json", replay)
        artifacts.update({"replay": replay, "historical_inventory": historical_inventory})
        update_state(output, stage="SCORER_REPLAY_COMPLETE")
    except (B4ScorerReplayMismatch, base.B4ScorerReplayMismatch) as error:
        reason = f"{type(error).__name__}: {error}"
        dump_json(output / "b4_scorer_replay.json", {"status": "MISMATCH", "reason": reason, "test_accessed": False})
        _terminal_without_evaluation(output, reason)
        finalize_bundle(output, archive, "B4_SCORER_REPLAY_MISMATCH", artifacts, reason=reason)
        return 0
    except (InputIdentityInvalid, base.InputIdentityInvalid) as error:
        reason = f"{type(error).__name__}: {error}"
        _terminal_without_evaluation(output, reason)
        finalize_bundle(output, archive, "INPUT_IDENTITY_INVALID", artifacts, reason=reason)
        return 0

    try:
        exclusions, exclusion_manifest = load_exclusion_identities(historical_rows, config)
        candidates, generation = generate_all_candidates(runtime, exclusions, candidate_log)
        artifacts.update({"generation": generation, "exclusion_manifest": exclusion_manifest})
        update_state(output, stage="CANDIDATES_GENERATED", candidate_count=len(candidates))
    except CandidateGenerationFailed as error:
        reason = str(error)
        dump_json(output / "pre_search/exclusion_manifest.json", artifacts.get("exclusion_manifest", {"status": "FAILED", "reason": reason}))
        _terminal_without_evaluation(output, reason)
        finalize_bundle(output, archive, "CANDIDATE_GENERATION_FAILED", artifacts, reason=reason)
        return 0
    except (InputIdentityInvalid, base.InputIdentityInvalid) as error:
        reason = str(error)
        _terminal_without_evaluation(output, reason)
        finalize_bundle(output, archive, "INPUT_IDENTITY_INVALID", artifacts, reason=reason)
        return 0

    try:
        predictions, scoring = score_candidates_with_frozen_b4(
            runtime, folds, candidates, scorer_log, label="current_prospective_6144",
        )
        plans, selected, feasibility, selection_audits = select_search_plans(candidates, predictions)
    except (B4ScorerReplayMismatch, base.B4ScorerReplayMismatch) as error:
        reason = f"{type(error).__name__}: {error}"
        dump_json(output / "b4_scorer_replay.json", {
            **replay, "status": "MISMATCH_DURING_CURRENT_SCORING", "reason": reason,
            "GNN_training_started": False, "test_accessed": False,
        })
        _terminal_without_evaluation(output, reason)
        finalize_bundle(output, archive, "B4_SCORER_REPLAY_MISMATCH", artifacts, reason=reason)
        return 0
    except P2PolicyInfeasible as error:
        reason = str(error)
        score_by_joint = {str(row["joint_fingerprint"]): float(row["b4_prediction"]) for row in predictions}
        dump_csv(output / "pre_search/candidate_manifest.csv", [
            {
                **dict(row), "b4_prediction": score_by_joint[str(row["joint_fingerprint"])],
                "pure_b4_selected": None, "p2_selected": None, "random_selected": None,
            }
            for row in candidates
        ], CANDIDATE_FIELDS)
        dump_csv(
            output / "pre_search/b4_predictions.csv", predictions,
            ("replicate_id", "round_id", "candidate_slot", "joint_fingerprint", "architecture_fingerprint", "b4_prediction"),
        )
        dump_json(output / "pre_search/p2_feasibility_failure.json", {
            "status": "INFEASIBLE", "reason": reason,
            "candidate_count": len(candidates), "prediction_count": len(predictions),
            "all_candidates_generated_before_failure": True,
            "all_B4_predictions_completed_before_failure": True,
            "GNN_training_started": False, "test_accessed": False,
        })
        _terminal_without_evaluation(output, reason)
        finalize_bundle(output, archive, "P2_POLICY_INFEASIBLE", artifacts, reason=reason)
        return 0
    manipulation = diversity_manipulation_audit(plans, candidates)
    overlap = selection_audits["selection_overlap"]
    information = selection_audits["information_gate"]
    scorer_identity = {
        "status": "IDENTICAL", "scorer": "B4_DIRECT_JOINT_REGRESSOR",
        "replay": replay, "candidate_scoring": scoring,
        "deterministic_reconstruction_role": "identity_replay_only",
        "historical_supervision": "historical_336_only",
        "prospective_labels_used": False, "current_labels_used": False,
        "input_hashes": {
            "diagnosis_archive": config["authoritative_inputs"]["collapse_diagnosis"]["archive_sha256"],
            "diagnosis_manifest": config["authoritative_inputs"]["collapse_diagnosis"]["manifest_sha256"],
            "diagnosis_decision": config["authoritative_inputs"]["collapse_diagnosis"]["decision_sha256"],
            "old_B4_archive": config["authoritative_inputs"]["previous_b4_search"]["archive_sha256"],
            "old_B4_candidate_manifest": config["authoritative_inputs"]["previous_b4_search"]["candidate_manifest_sha256"],
            "old_B4_predictions": replay["anchors"]["previous_B4_search_predictions_sha256"],
            "old_M1_M2_archive": config["authoritative_inputs"]["previous_metric_search"]["archive_sha256"],
            "old_M1_M2_candidate_manifest": config["authoritative_inputs"]["previous_metric_search"]["candidate_manifest_sha256"],
            "previous_8_pool_candidate_manifest": config["authoritative_inputs"]["previous_8_pool"]["candidate_manifest_sha256"],
            "previous_16_pool_candidate_manifest": config["authoritative_inputs"]["previous_16_pool"]["candidate_manifest_sha256"],
            "historical_P_reference": HISTORICAL_P_SOURCE_SHA256,
        },
        "source_hashes": {
            "current_config": sha256_file(config_path),
            "current_analysis": sha256_file(Path(__file__).resolve()),
            "current_aggregate_source_identity": source_sha256,
            "B4_search_source": config["authoritative_inputs"]["frozen_b4_protocol"]["source_sha256"],
            "direct_regressor": sha256_file(base.SOFT_SNAPSHOT / "analyse/joint_configuration_metric_gate.py"),
            "high_regime_generator": sha256_file(base.HIGH_SNAPSHOT / "analyse/high_regime_prior_model_blind_validation.py"),
            "high_regime_evaluator": sha256_file(base.HIGH_SNAPSHOT / "analyse/run_high_regime_prior_model_blind_validation_labels.py"),
            "frozen_eval_utils": sha256_file(base.HIGH_SNAPSHOT / "eval_utils.py"),
            "frozen_architecture_features": sha256_file(base.HIGH_SNAPSHOT / "architecture_diagnostics.py"),
            "HP_transform": sha256_file(REPO_ROOT / "hp_modes.py"),
            "dataset_loader": sha256_file(REPO_ROOT / "dataset_utils.py"),
        },
        "all_predictions_before_first_GNN": True, "test_accessed": False,
    }
    generator_identity = {
        "status": "IDENTICAL", "candidate_master_seed": CANDIDATE_MASTER_SEED,
        "replicate_seeds": [CANDIDATE_MASTER_SEED + replicate for replicate in range(REPLICATES)],
        "random_master_seed": RANDOM_MASTER_SEED,
        "random_rng_order": "single_numpy_default_rng_replicate_then_round",
        "VAE_checkpoint_sha256": "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4",
        "high_regime_generator_sha256": sha256_file(base.HIGH_SNAPSHOT / "analyse/high_regime_prior_model_blind_validation.py"),
        "generation_audit": generation, "test_accessed": False,
    }
    candidate_hash = stable_hash(candidates)
    freeze = freeze_pre_search(
        output / "pre_search", candidates, predictions, plans, overlap,
        manipulation, information, feasibility, scorer_identity,
        generator_identity, exclusion_manifest,
    )
    verified = verify_pre_search_freeze(output / "pre_search")
    if stable_hash(candidates) != candidate_hash:
        raise ProtocolError("candidate records changed during immutable freeze")
    artifacts.update({
        "scoring": scoring, "plans": plans, "feasibility": feasibility,
        "manipulation": manipulation, "overlap": overlap,
        "information": information, "freeze": freeze,
        "pre_search_root": verified["root_sha256"],
    })
    union_count = len(set().union(*selected.values()))
    update_state(
        output, stage="SELECTIONS_FROZEN", pure_b4_slots=SLOTS_PER_METHOD,
        p2_slots=SLOTS_PER_METHOD, random_slots=SLOTS_PER_METHOD,
        selected_unique_config_count=union_count, planned_gnn_runs=union_count * 3,
    )
    if not information["passed"]:
        reason = "frozen PURE/P2 divergence failed the preregistered information gate"
        training_audit = _terminal_without_evaluation(output, reason)
        artifacts["training_audit"] = training_audit
        finalize_bundle(output, archive, "DIVERSITY_INTERVENTION_LOW_INFORMATION", artifacts, reason=reason)
        return 0
    update_state(output, stage="INFORMATION_GATE_PASSED")
    verify_pre_search_freeze(output / "pre_search")
    if stable_hash(candidates) != candidate_hash:
        raise ProtocolError("candidate records changed between freeze and task creation")
    selected_configs, tasks = build_selected_union_tasks(candidates, plans, pre_search_verified=True)
    update_state(
        output, selected_unique_config_count=len(selected_configs), planned_gnn_runs=len(tasks),
        completed_gnn_runs=0, failed_gnn_runs=0,
    )
    del folds
    raw, training_audit = run_frozen_gnn_evaluation(
        output, tasks, config, verified["root_sha256"], evaluation_log,
    )
    dump_csv(
        output / "evaluation/raw_seed_results.csv", raw,
        ("joint_fingerprint", "training_seed", "validation_accuracy", "terminal_status", "valid", "attempt_count", "assigned_gpu"),
    )
    config_stats, complete = aggregate_config_statistics(
        selected_configs, raw, predictions, allow_incomplete=True,
    )
    complete = bool(
        complete and training_audit["all_shards_exited_zero"]
        and training_audit["loader_firewall_passed"]
    )
    dump_csv(
        output / "evaluation/config_3seed_statistics.csv", config_stats,
        ("joint_fingerprint", "architecture_fingerprint", "validation_seed_0", "validation_seed_1", "validation_seed_2", "validation_mean", "validation_sd", "b4_prediction", "selected_by_pure_b4", "selected_by_p2", "selected_by_random"),
    )
    successful_count = sum(row.get("terminal_status") == "success" and row.get("valid") is True for row in raw)
    training_audit.update({
        "evaluation_complete": complete, "completed_gnn_runs": successful_count,
        "failed_gnn_runs": len(tasks) - successful_count, "task_count": len(tasks),
        "physical_unique_configurations": len(selected_configs),
        "physical_unique_GNN_runs": len(tasks), "method_equivalent_config_slots": 576,
        "method_equivalent_GNN_runs": 1728, "candidate_replacement": False,
        "retry_changed_config_or_seed": False, "same_config_results_shared": True,
        "track_test": False, "test_accessed": False,
        "fresh_seed_confirmation_started": False,
    })
    dump_json(output / "evaluation/training_audit.json", training_audit)
    artifacts["training_audit"] = training_audit
    update_state(
        output, stage="EVALUATION_COMPLETE" if complete else "EVALUATION_RUNNING",
        completed_gnn_runs=successful_count, failed_gnn_runs=len(tasks) - successful_count,
        current_replicate=None, current_round=None, current_config=None, current_seed=None,
    )
    if not complete:
        reason = "at least one selected configuration lacks a successful result for seed 0, 1, or 2"
        finalize_bundle(output, archive, "DIVERSITY_EVALUATION_INCOMPLETE", artifacts, reason=reason)
        return 0
    verify_pre_search_freeze(output / "pre_search")
    analysis = compute_search_analysis(
        config_stats, plans, candidates, manipulation, information, training_audit,
    )
    write_search_outputs(output, analysis)
    classification = str(analysis["decision"]["classification"])
    next_stage = write_next_stage(output, classification, analysis["method_winners"])
    dump_json(output / "decision.json", analysis["decision"])
    artifacts.update({"analysis": analysis, "next_stage": next_stage})
    report = render_final_report(artifacts)
    assert_complete_final_report(report)
    dump_text(output / "final_report.md", report)
    analysis_log(f"frozen classification: {classification}")
    update_state(output, stage="ANALYSIS_COMPLETE", classification=classification)
    finalize_bundle(output, archive, classification, artifacts)
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
            dump_json(output / "frozen_protocol.json", _placeholder("RUN_FAILED", message))
    if not (output / "environment.json").is_file():
        write_environment_and_git(output, before=True)
    dump_json(output / "error.json", {
        "classification": "RUN_FAILED", "message": message,
        "traceback": traceback.format_exc(), "test_accessed": False,
    })
    if not (output / "input_integrity.json").is_file():
        dump_json(output / "input_integrity.json", _placeholder("RUN_FAILED", message))
    if not (output / "authorization_replay.json").is_file():
        dump_json(output / "authorization_replay.json", _placeholder("RUN_FAILED", message))
    if not (output / "b4_scorer_replay.json").is_file():
        dump_json(output / "b4_scorer_replay.json", _placeholder("RUN_FAILED", message))
    if not (output / "evaluation/training_audit.json").is_file():
        _terminal_without_evaluation(output, message)
    dump_json(output / "decision.json", {
        "classification": "RUN_FAILED", "reason": message,
        "fresh_seed_confirmation_started": False, "test_accessed": False,
    })
    finalize_bundle(
        output, archive, "RUN_FAILED", {"test_accessed": False},
        reason=message, run_failed=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--source-sha256", default="")
    parser.add_argument("--preflight-log-dir", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--synthetic-smoke", action="store_true")
    modes.add_argument("--formal", action="store_true")
    modes.add_argument("--source-identity", action="store_true")
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
