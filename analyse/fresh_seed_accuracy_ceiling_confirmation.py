#!/usr/bin/env python3
"""Two-stage fresh-seed accuracy-ceiling confirmation.

The formal workflow ranks only configurations with completed historical
three-seed validation evaluations, freezes six challengers before any fresh
label exists, screens them on matched seeds 1000..1014, then confirms the
frozen top two against the historical reference P on disjoint matched seeds
2000..2029.  Test data are forbidden throughout; only the two GPU evaluator
child processes may initialize CUDA.
"""

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

from analyse import b4_guided_search_accuracy_gate as base  # noqa: E402
from analyse import b4_arch_diversity_prospective_search_accuracy_gate as diversity  # noqa: E402


PROTOCOL_VERSION = "fresh_seed_accuracy_ceiling_confirmation_global4_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/fresh_seed_accuracy_ceiling_confirmation_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/posthoc/fresh-seed-accuracy-ceiling-confirmation-global4-v1"
DEFAULT_ARCHIVE = Path(str(DEFAULT_OUTPUT) + ".tar.gz")

IMPLEMENTATION_FILES = (
    "configs/fresh_seed_accuracy_ceiling_confirmation_v1.yaml",
    "analyse/fresh_seed_accuracy_ceiling_confirmation.py",
    "scripts/run_fresh_seed_accuracy_ceiling_confirmation.sh",
    "tests/test_fresh_seed_accuracy_ceiling_confirmation.py",
)

SOURCE_IDS = ("A", "B", "C")
SOURCE_ROOTS = {
    "A": REPO_ROOT / "results/posthoc/b4-guided-search-accuracy-gate-global4-v1",
    "B": REPO_ROOT / "results/posthoc/m1-m2-metric-guided-search-accuracy-gate-global4-v1",
    "C": REPO_ROOT / "results/posthoc/b4-arch-diversity-prospective-search-accuracy-gate-global4-v1",
}
HIGH_ROOT = REPO_ROOT / "results/search/high-regime-prior-model-blind-validation-global4-v1"

P_JOINT = "30e0bb108c1fdbe8db186152a09c3f2fc6e812be189db920fe4d0d2b0d3cb1a2"
P_ARCH = "960efb3b97f00491c3956657cfd57958effb6096f65b1db2fa8aaae586b82207"
P_HP = "911259943106a1edad300bc5300a3883ece8cbffba48b95e6c61b6e5eb4b679d"
P_OLD_MEAN = 0.8087333333333333
P_OLD_SD = 0.007583709097675123
P_OLD_CI = (0.8059015298168731, 0.8115651368497935)

EXPECTED_UNIVERSE = 1121
EXPECTED_TOP6 = (
    "1cd14d582a9a13e905bead7e23b8a0846dac3391a0e4a3b480075b97021f7ef1",
    "07811e7fb8f3c58b70ea651bfa3688d0d4666bdcf8fb862695ea41445ef518fd",
    "9aeeb39570122b480970c38bc002ab5cbc7badad5b561642de2f8e3bb064d60c",
    "020b3ba68414773cc9c3b743c9bbf9294eb36308af23604d63d3b26de8245599",
    "515307ff81a6d08d8a3980c297196396130b3f9a867385fb77085830c1284d68",
    "97743f62bfcd5816b368d533b0f2f2f443ae76b239b2d36647f4540c19a57d5f",
)
HISTORICAL_SEEDS = (0, 1, 2)
STAGE1_SEEDS = tuple(range(1000, 1015))
STAGE2_SEEDS = tuple(range(2000, 2030))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 0
NUMERICAL_TOLERANCE = 1.0e-12
PRACTICAL_MARGIN = 0.003

ALLOWED_CLASSIFICATIONS = frozenset({
    "INPUT_IDENTITY_INVALID",
    "INPUT_PROTOCOL_MISMATCH",
    "CANDIDATE_RECOVERY_FAILED",
    "FRESH_SEED_COLLISION",
    "STAGE1_EVALUATION_INCOMPLETE",
    "STAGE2_EVALUATION_INCOMPLETE",
    "PRACTICAL_ACCURACY_IMPROVEMENT_CONFIRMED",
    "SMALL_STABLE_ACCURACY_GAIN",
    "NO_STABLE_ACCURACY_GAIN",
    "RUN_FAILED",
})

STATE_STAGES = (
    "IMPLEMENTATION_READY", "INPUTS_VALIDATED", "STAGE1_FROZEN",
    "STAGE1_RUNNING", "STAGE1_COMPLETE", "STAGE2_SELECTION_FROZEN",
    "STAGE2_RUNNING", "STAGE2_COMPLETE", "ANALYSIS_COMPLETE",
)

REPORT_SECTIONS = (
    "1. Executive conclusion", "2. Scientific question",
    "3. Why direct 3-seed winner comparison is invalid",
    "4. Two-stage fresh-seed design", "5. Input archive identity",
    "6. Protocol compatibility", "7. 1121-config candidate universe",
    "8. Historical P identity", "9. Frozen top-6 candidates",
    "10. Candidate provenance", "11. Stage1 fresh seeds",
    "12. Stage1 evaluation completeness", "13. Stage1 paired results",
    "14. Frozen Stage2 top-2", "15. Stage2 fresh seeds",
    "16. Stage2 evaluation completeness", "17. Stage2 absolute validation means",
    "18. Stage2 paired differences vs P", "19. Bonferroni multiplicity control",
    "20. Stable-positive-gain gates", "21. Practical-gain gates",
    "22. Winner/loss/tie by matched seed", "23. Winner's curse from 3 →15 →30 seeds",
    "24. Historical P old30 vs fresh P", "25. Final classification",
    "26. New stable best if any", "27. Candidate-level accuracy interpretation",
    "28. Why this does not establish method superiority",
    "29. What this says about current accuracy ceiling", "30. Next research direction",
    "31. Limitations", "32. Engineering audit",
)

REQUIRED_OUTPUTS = (
    "state.json", "frozen_protocol.json", "input_integrity.json",
    "protocol_compatibility.json", "candidate_universe.csv",
    "historical_P_identity.json", "pre_stage1/candidate_universe.csv",
    "pre_stage1/top6_candidates.json", "pre_stage1/candidate_provenance.csv",
    "pre_stage1/historical_P_identity.json", "pre_stage1/stage1_seed_spec.json",
    "pre_stage1/stage2_seed_spec.json", "pre_stage1/stage2_selection_rule.json",
    "pre_stage1/evaluation_protocol_identity.json", "pre_stage1/freeze_manifest.json",
    "pre_stage1/pre_stage1_root_sha256.txt", "stage1/task_manifest.json",
    "stage1/raw_seed_results.csv", "stage1/paired_differences.csv",
    "stage1/candidate_statistics.csv", "stage1/stage1_ranking.csv",
    "stage1/stage2_selection.json", "stage1/stage2_selection_manifest.json",
    "stage1/stage2_selection_root_sha256.txt", "stage2/task_manifest.json",
    "stage2/raw_seed_results.csv", "stage2/paired_differences.csv",
    "stage2/candidate_statistics.csv", "stage2/bootstrap_results.json",
    "stage2/multiplicity_control.json", "stage2/fresh_P_statistics.json",
    "analysis/winner_curse_analysis.json", "analysis/provenance_summary.json",
    "analysis/historical_vs_fresh_P.json", "analysis/method_attribution_boundary.json",
    "decision.json", "final_report.md", "environment.json", "git_status_before.txt",
    "git_status_after.txt", "git_diff.patch", "source_snapshot/imported_local_modules.json",
    "source_snapshot/sha256_manifest.json", "manifest.json",
)


class ProtocolError(RuntimeError):
    """A frozen protocol contract was violated."""


class InputIdentityInvalid(ProtocolError):
    """An authoritative archive or recorded result changed."""


class InputProtocolMismatch(ProtocolError):
    """The three source evaluators are scientifically incompatible."""


class CandidateRecoveryFailed(ProtocolError):
    """A selected joint configuration could not be recovered exactly."""


class FreshSeedCollision(ProtocolError):
    """A supposedly fresh configuration/seed pair already has a real result."""


class StageEvaluationIncomplete(ProtocolError):
    """A stage lacks at least one required successful validation result."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


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
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8") + b"\n")
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


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ProtocolError("PyYAML is required") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError("protocol config must be a mapping")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ProtocolError(f"frozen protocol mismatch for {label}: {actual!r} != {expected!r}")


def validate_frozen_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    _require_equal(config["protocol_version"], PROTOCOL_VERSION, "protocol version")
    _require_equal(config["track_test"], False, "test firewall")
    _require_equal(config["candidate_universe"]["expected_unique_joint_config_count"], EXPECTED_UNIVERSE, "universe count")
    _require_equal(tuple(config["candidate_universe"]["expected_top6_joint_fingerprints"]), EXPECTED_TOP6, "top6 identity")
    _require_equal(tuple(config["stage1"]["training_seeds"]), STAGE1_SEEDS, "Stage1 seeds")
    _require_equal(tuple(config["stage2"]["training_seeds"]), STAGE2_SEEDS, "Stage2 seeds")
    _require_equal(config["budget"]["stage1_GNN_runs"], 105, "Stage1 budget")
    _require_equal(config["budget"]["stage2_GNN_runs"], 90, "Stage2 budget")
    _require_equal(config["budget"]["total_GNN_runs"], 195, "total budget")
    _require_equal(config["inference"]["paired_bootstrap"]["replicates"], BOOTSTRAP_REPLICATES, "bootstrap replicates")
    _require_equal(config["inference"]["paired_bootstrap"]["master_seed"], BOOTSTRAP_SEED, "bootstrap seed")
    _require_equal(config["inference"]["multiplicity_control"]["one_sided_alpha_per_challenger"], 0.025, "Bonferroni alpha")
    _require_equal(config["inference"]["multiplicity_control"]["lower_bound_percentile"], 2.5, "Bonferroni percentile")
    _require_equal(config["inference"]["practical_gain"]["minimum_mean_delta"], PRACTICAL_MARGIN, "practical margin")
    _require_equal(set(config["allowed_classifications"]), set(ALLOWED_CLASSIFICATIONS), "classifications")
    _require_equal(tuple(config["state_stages"]), STATE_STAGES, "state stages")
    if set(STAGE1_SEEDS) & set(STAGE2_SEEDS):
        raise ProtocolError("Stage1 and Stage2 seeds overlap")
    return {"status": "VALID", "protocol_version": PROTOCOL_VERSION, "test_accessed": False}


def initial_state(config_sha256: str, source_sha256: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION, "stage": "IMPLEMENTATION_READY",
        "classification": None, "source_config_count": 0, "unique_candidate_count": 0,
        "screen_candidate_count": 0, "stage1_planned_runs": 105,
        "stage1_completed_runs": 0, "stage1_failed_runs": 0,
        "stage2_candidate_count": 0, "stage2_planned_runs": 90,
        "stage2_completed_runs": 0, "stage2_failed_runs": 0,
        "current_stage": None, "current_config": None, "current_seed": None,
        "test_accessed": False, "config_sha256": config_sha256,
        "source_sha256": source_sha256, "created_at": utc_now(), "updated_at": utc_now(),
    }


def update_state(output: Path, **changes: Any) -> dict[str, Any]:
    if changes.get("test_accessed") not in {None, False}:
        raise ProtocolError("test_accessed can never become true")
    path = output / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state.update(changes)
    state["updated_at"] = utc_now()
    dump_json(path, state)
    return state


def _require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise InputIdentityInvalid(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise InputIdentityInvalid(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return {"label": label, "path": str(path), "sha256": observed}


def verify_directory_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise InputIdentityInvalid(f"missing directory manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for entry in manifest.get("files", []):
        relative = str(entry["relative_path"])
        target = root / relative
        if not target.is_file():
            raise InputIdentityInvalid(f"manifest entry is missing: {target}")
        digest = sha256_file(target)
        if digest != entry.get("sha256") or target.stat().st_size != int(entry.get("size_bytes", -1)):
            raise InputIdentityInvalid(f"manifest entry mismatch: {target}")
        pairs.append((relative, digest))
    observed = stable_hash(pairs)
    if observed != manifest.get("manifest_root_sha256"):
        raise InputIdentityInvalid(f"directory manifest root mismatch: {root}")
    return {"path": str(path), "file_count": len(pairs), "manifest_root_sha256": observed,
            "all_file_hashes_valid": True}


def _verify_archive(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"label": label, "present": False, "fallback": "directory manifest"}
    identity = _require_hash(path, expected, label)
    sidecar = Path(str(path) + ".sha256")
    if sidecar.is_file() and sidecar.read_text(encoding="utf-8").split()[0] != expected:
        raise InputIdentityInvalid(f"{label} sidecar mismatch")
    return {**identity, "present": True, "sidecar_valid_if_present": True}


def _scientific_training_identity(root: Path) -> dict[str, Any]:
    prereg = json.loads((root / "evaluation/evaluator_adapter/preregistration.json").read_text(encoding="utf-8"))
    training = prereg["training"]
    keys = (
        "dataset", "dataset_content_fingerprint", "split", "split_fingerprint",
        "graph_transform", "max_epochs", "patience", "maximum_system_retries",
        "training_seeds", "track_test", "validation_only_data_view", "shard_to_gpu_uuid",
    )
    normalized = {key: training[key] for key in keys}
    dependency = json.loads((root / "evaluation/evaluator_adapter/source_dependency_manifest.json").read_text(encoding="utf-8"))
    normalized["source_dependencies"] = dependency["files"]
    return normalized


def validate_authoritative_inputs(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    checks = []
    archives = []
    protocols: dict[str, dict[str, Any]] = {}
    identities = {}
    files = config["source_files"]
    for source_id in SOURCE_IDS:
        spec = config["authoritative_sources"][source_id]
        root = SOURCE_ROOTS[source_id]
        if not (root / spec["required_complete_marker"]).is_file():
            raise InputIdentityInvalid(f"Source {source_id} COMPLETE marker is missing")
        archive = REPO_ROOT / spec["archive"]
        archives.append(_verify_archive(archive, spec["archive_sha256"], f"Source {source_id} archive"))
        for relative_key, hash_key in (
            ("manifest", "manifest_sha256"), ("decision", "decision_sha256"),
            ("frozen_protocol", "frozen_protocol_sha256"),
            ("candidate_manifest", "candidate_manifest_sha256"),
            ("task_manifest", "task_manifest_sha256"),
            ("config_statistics", "config_statistics_sha256"),
            ("training_audit", "training_audit_sha256"),
        ):
            checks.append(_require_hash(root / files[relative_key], spec[hash_key], f"Source {source_id} {relative_key}"))
        decision = json.loads((root / files["decision"]).read_text(encoding="utf-8"))
        state = json.loads((root / files["state"]).read_text(encoding="utf-8"))
        audit = json.loads((root / files["training_audit"]).read_text(encoding="utf-8"))
        if decision.get("classification") != spec["required_classification"] or state.get("classification") != spec["required_classification"]:
            raise InputIdentityInvalid(f"Source {source_id} classification mismatch")
        if any(value is not False for value in (decision.get("test_accessed"), state.get("test_accessed"), audit.get("test_accessed"))):
            raise InputIdentityInvalid(f"Source {source_id} test firewall identity mismatch")
        if audit.get("track_test") is not False or audit.get("evaluation_complete") is not True:
            raise InputIdentityInvalid(f"Source {source_id} evaluation is not complete validation-only data")
        directory = verify_directory_manifest(root)
        protocols[source_id] = _scientific_training_identity(root)
        identities[source_id] = {
            "archive_sha256": spec["archive_sha256"], "classification": decision["classification"],
            "manifest": directory, "evaluated_config_count": spec["evaluated_config_count"],
            "test_accessed": False,
        }
    reference = protocols["A"]
    mismatches = {source_id: payload for source_id, payload in protocols.items() if payload != reference}
    compatibility = {
        "status": "COMPATIBLE" if not mismatches else "MISMATCH",
        "normalized_scientific_identity": reference,
        "per_source_identity_sha256": {key: stable_hash(value) for key, value in protocols.items()},
        "training_protocol_label_hashes_are_non_authoritative": True,
        "mismatches": mismatches, "test_accessed": False,
    }
    if mismatches:
        raise InputProtocolMismatch(f"source evaluator protocols differ: {sorted(mismatches)}")
    return {
        "status": "VALID", "source_identities": identities, "checked_files": checks,
        "archives": archives, "all_complete": True, "all_test_accessed_false": True,
        "test_accessed": False,
    }, compatibility


def _parse_candidate_manifest(path: Path) -> list[dict[str, Any]]:
    return diversity._parse_candidate_csv(path)


def _provenance_from_stats(source_id: str, row: Mapping[str, str]) -> list[str]:
    mapping = {
        "A": (("selected_by_B4", "B4_GUIDED"), ("selected_by_random", "MODEL_BLIND_RANDOM_HIGH_REGIME")),
        "B": (("selected_by_M1", "M1_METRIC_KNN_GUIDED"), ("selected_by_M2", "M2_METRIC_GP_GUIDED"),
              ("selected_by_random", "M12_MODEL_BLIND_RANDOM_HIGH_REGIME")),
        "C": (("selected_by_pure_b4", "PURE_B4_TOP4"), ("selected_by_p2", "B4_P2_DIVERSITY"),
              ("selected_by_random", "MODEL_BLIND_RANDOM_HIGH_REGIME")),
    }
    return [label for field, label in mapping[source_id] if str(row.get(field, "")).lower() == "true"]


def recompute_fingerprints(config_row: Mapping[str, Any]) -> tuple[str, str, str]:
    architecture = stable_hash({
        "operations": list(config_row["operations"]),
        "edges": sorted([list(map(int, edge)) for edge in config_row["edges"]]),
    })
    hp = stable_hash([float(value) for value in config_row["raw_hp_vector"]])
    joint = stable_hash({"architecture_fingerprint": architecture, "hp_fingerprint": hp})
    return architecture, hp, joint


def build_candidate_universe(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = config["source_files"]
    by_joint: dict[str, dict[str, Any]] = {}
    source_counts = {}
    duplicate_sources: dict[str, list[str]] = defaultdict(list)
    for source_id in SOURCE_IDS:
        root = SOURCE_ROOTS[source_id]
        stats = read_csv(root / files["config_statistics"])
        manifest_rows = _parse_candidate_manifest(root / files["candidate_manifest"])
        task_payload = json.loads((root / files["task_manifest"]).read_text(encoding="utf-8"))
        task_rows = task_payload.get("tasks", [])
        tasks_by_joint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for task in task_rows:
            tasks_by_joint[str(task.get("joint_config_fingerprint", task.get("joint_fingerprint")))].append(task)
        manifest_by_joint = defaultdict(list)
        for row in manifest_rows:
            manifest_by_joint[str(row["joint_fingerprint"])].append(row)
        expected_count = int(config["authoritative_sources"][source_id]["evaluated_config_count"])
        if len(stats) != expected_count:
            raise InputIdentityInvalid(f"Source {source_id} config count is {len(stats)}, expected {expected_count}")
        if len({row["joint_fingerprint"] for row in stats}) != len(stats):
            raise InputIdentityInvalid(f"Source {source_id} statistics contain duplicate joints")
        for stat in stats:
            joint = str(stat["joint_fingerprint"])
            values = [float(stat[f"validation_seed_{seed}"]) for seed in HISTORICAL_SEEDS]
            if not all(math.isfinite(value) for value in values):
                raise InputIdentityInvalid(f"Source {source_id} contains incomplete seed0/1/2 values")
            observed_mean = float(stat["validation_mean"])
            if abs(observed_mean - statistics.fmean(values)) > NUMERICAL_TOLERANCE:
                raise InputIdentityInvalid(f"Source {source_id} validation mean mismatch for {joint}")
            candidates = manifest_by_joint.get(joint, [])
            if len(candidates) != 1:
                raise CandidateRecoveryFailed(f"Source {source_id} has {len(candidates)} candidate definitions for {joint}")
            historical_tasks = tasks_by_joint.get(joint, [])
            if len(historical_tasks) != 3 or {int(task.get("seed", task.get("training_seed"))) for task in historical_tasks} != set(HISTORICAL_SEEDS):
                raise CandidateRecoveryFailed(f"Source {source_id} lacks exact seed0/1/2 task identity for {joint}")
            candidate = copy.deepcopy(candidates[0])
            architecture, hp, recomputed_joint = recompute_fingerprints(candidate)
            if architecture != str(candidate["architecture_fingerprint"]) or hp != str(candidate["hp_fingerprint"]) or recomputed_joint != joint:
                raise CandidateRecoveryFailed(f"fingerprint reconstruction failed for {joint}")
            entry = {
                **candidate, "joint_fingerprint": joint,
                "historical_seed_0": values[0], "historical_seed_1": values[1],
                "historical_seed_2": values[2], "historical_validation_mean": observed_mean,
                "historical_validation_sd": float(stat["validation_sd"]),
                "historical_score": round(observed_mean, 12),
                "source_experiments": [source_id],
                "source_ids": [config["authoritative_sources"][source_id]["source_id"]],
                "originally_selected_by": _provenance_from_stats(source_id, stat),
                "historical_task_keys": sorted(str(task.get("task_key")) for task in historical_tasks),
            }
            if joint in by_joint:
                existing = by_joint[joint]
                identity_fields = ("architecture_fingerprint", "hp_fingerprint", "operations", "edges", "raw_hp_vector")
                if any(existing[field] != entry[field] for field in identity_fields) or abs(float(existing["historical_validation_mean"]) - observed_mean) > NUMERICAL_TOLERANCE:
                    raise InputIdentityInvalid(f"cross-source duplicate differs for {joint}")
                existing["source_experiments"].append(source_id)
                existing["source_ids"].extend(entry["source_ids"])
                existing["originally_selected_by"] = sorted(set(existing["originally_selected_by"]) | set(entry["originally_selected_by"]))
                duplicate_sources[joint].append(source_id)
            else:
                by_joint[joint] = entry
        source_counts[source_id] = len(stats)
    if len(by_joint) != EXPECTED_UNIVERSE:
        raise InputIdentityInvalid(f"candidate union is {len(by_joint)}, expected {EXPECTED_UNIVERSE}")
    p_removed = by_joint.pop(P_JOINT, None)
    ordered = sorted(by_joint.values(), key=lambda row: (-float(row["historical_score"]), str(row["joint_fingerprint"])))
    audit = {
        "source_config_counts": source_counts, "source_config_count": sum(source_counts.values()),
        "unique_joint_config_count_before_P_exclusion": EXPECTED_UNIVERSE,
        "P_present_in_source_union": p_removed is not None,
        "challenger_universe_count": len(ordered),
        "cross_source_duplicate_count": EXPECTED_UNIVERSE - sum(source_counts.values()),
        "cross_source_duplicate_joints": sorted(duplicate_sources),
        "only_complete_real_seed_0_1_2_evaluations": True,
        "predictor_only_candidates_included": False, "test_accessed": False,
    }
    return ordered, audit


def select_top6(universe: Sequence[Mapping[str, Any]], expected: Sequence[str] = EXPECTED_TOP6) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in universe),
                     key=lambda row: (-round(float(row["historical_validation_mean"]), 12), str(row["joint_fingerprint"])))
    top = ordered[:6]
    observed = tuple(str(row["joint_fingerprint"]) for row in top)
    if set(observed) != set(expected) or observed != tuple(expected):
        raise InputIdentityInvalid(f"frozen top6 identity mismatch: {observed}")
    for rank, row in enumerate(top, 1):
        row["historical_rank"] = rank
        row["historical_score"] = round(float(row["historical_validation_mean"]), 12)
    return top


def recover_historical_p(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["historical_P"]
    _require_hash(REPO_ROOT / spec["identity_source"], spec["identity_source_sha256"], "historical P identity")
    config_path = REPO_ROOT / spec["configuration_source"]
    _require_hash(config_path, spec["configuration_source_sha256"], "historical P configuration source")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("configs", payload.get("candidates", []))
    matches = [row for row in rows if str(row.get("joint_config_fingerprint", row.get("joint_fingerprint"))) == P_JOINT]
    if len(matches) != 1:
        raise CandidateRecoveryFailed(f"historical P configuration matches={len(matches)}")
    row = copy.deepcopy(matches[0])
    row["joint_fingerprint"] = P_JOINT
    architecture, hp, joint = recompute_fingerprints(row)
    if (architecture, hp, joint) != (P_ARCH, P_HP, P_JOINT):
        raise CandidateRecoveryFailed("historical P fingerprint reconstruction failed")
    return {
        **row, "role": "REFERENCE_CONFIGURATION", "architecture_fingerprint": P_ARCH,
        "hp_fingerprint": P_HP, "joint_fingerprint": P_JOINT,
        "historical_seed_count": 30, "historical_validation_mean": P_OLD_MEAN,
        "historical_validation_sd": P_OLD_SD, "historical_validation_ci95": list(P_OLD_CI),
        "historical_statistics_role": "DESCRIPTIVE_ONLY",
        "fresh_matched_seed_retraining_required": True, "test_accessed": False,
    }


def candidate_universe_rows(universe: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "joint_fingerprint": row["joint_fingerprint"], "architecture_fingerprint": row["architecture_fingerprint"],
        "hp_fingerprint": row["hp_fingerprint"], "historical_seed_0": row["historical_seed_0"],
        "historical_seed_1": row["historical_seed_1"], "historical_seed_2": row["historical_seed_2"],
        "historical_validation_mean": row["historical_validation_mean"],
        "historical_validation_sd": row["historical_validation_sd"], "historical_score": row["historical_score"],
        "source_experiments": row["source_experiments"], "originally_selected_by": row["originally_selected_by"],
    } for row in universe]


UNIVERSE_FIELDS = (
    "joint_fingerprint", "architecture_fingerprint", "hp_fingerprint",
    "historical_seed_0", "historical_seed_1", "historical_seed_2",
    "historical_validation_mean", "historical_validation_sd", "historical_score",
    "source_experiments", "originally_selected_by",
)


def rank_historical_candidates(
    rows: Sequence[Mapping[str, Any]], p_joint: str = P_JOINT, top_n: int = 6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pure ranking helper: provenance and predictions are intentionally ignored."""
    deduplicated: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        joint = str(row["joint_fingerprint"])
        if joint == p_joint:
            continue
        score = round(float(row.get("historical_validation_mean", row.get("validation_mean"))), 12)
        row["historical_score"] = score
        if joint in deduplicated:
            previous = deduplicated[joint]
            if abs(float(previous["historical_score"]) - score) > NUMERICAL_TOLERANCE:
                raise InputIdentityInvalid(f"duplicate historical score mismatch for {joint}")
            continue
        deduplicated[joint] = row
    ranked = sorted(deduplicated.values(), key=lambda row: (-float(row["historical_score"]), str(row["joint_fingerprint"])))
    return ranked, [dict(row) for row in ranked[:top_n]]


def collision_pairs_from_rows(
    rows: Sequence[Mapping[str, Any]], target_joints: set[str], target_seeds: set[int],
) -> set[tuple[str, int]]:
    collisions = set()
    for row in rows:
        joint = row.get("joint_fingerprint", row.get("joint_config_fingerprint"))
        seed = row.get("training_seed", row.get("seed"))
        if joint is None or seed is None:
            continue
        try:
            pair = (str(joint), int(seed))
        except (TypeError, ValueError):
            continue
        if pair[0] in target_joints and pair[1] in target_seeds:
            collisions.add(pair)
    return collisions


def audit_fresh_seed_collisions(
    configs: Sequence[Mapping[str, Any]], scan_roots: Sequence[Path], *, exclude_root: Path | None = None,
) -> dict[str, Any]:
    target_joints = {str(row["joint_fingerprint"]) for row in configs}
    target_seeds = set(STAGE1_SEEDS) | set(STAGE2_SEEDS)
    collisions: set[tuple[str, int]] = set()
    scanned = []
    inventory = []
    for root in scan_roots:
        if not root.exists():
            continue
        # Inventory every line-oriented result format under the two declared
        # historical result roots.  CSV files are admitted by schema rather
        # than filename so an older evaluator naming convention cannot evade
        # the fresh-pair audit.  JSON specifications are deliberately excluded:
        # they may preregister seed IDs without ever evaluating them.
        paths = sorted(set(root.rglob("*.jsonl")) | set(root.rglob("*.csv")))
        for path in paths:
            if exclude_root is not None:
                try:
                    path.resolve().relative_to(exclude_root.resolve())
                    continue
                except ValueError:
                    pass
            rows: list[dict[str, Any]] = []
            try:
                if path.suffix == ".jsonl":
                    with path.open("r", encoding="utf-8") as handle:
                        rows = [json.loads(line) for line in handle if line.strip()]
                else:
                    with path.open("r", encoding="utf-8", newline="") as handle:
                        reader = csv.DictReader(handle)
                        fields = set(reader.fieldnames or [])
                        has_joint = bool(fields & {"joint_fingerprint", "joint_config_fingerprint"})
                        has_seed = bool(fields & {"training_seed", "seed"})
                        has_result = bool(fields & {"validation_accuracy", "val_acc", "terminal_status", "valid"})
                        rows = [dict(row) for row in reader] if has_joint and has_seed and has_result else []
            except (OSError, csv.Error, json.JSONDecodeError) as error:
                raise InputIdentityInvalid(f"cannot parse historical result file {path}") from error
            actual_rows = [row for row in rows if (
                (row.get("joint_fingerprint") is not None or row.get("joint_config_fingerprint") is not None)
                and (row.get("training_seed") is not None or row.get("seed") is not None)
                and any(key in row for key in ("validation_accuracy", "val_acc", "terminal_status", "valid"))
            )]
            inventory.append({"path": str(path), "sha256": sha256_file(path),
                              "parsed_as_real_evaluation_results": bool(actual_rows)})
            if not actual_rows:
                continue
            found = collision_pairs_from_rows(actual_rows, target_joints, target_seeds)
            collisions.update(found)
            scanned.append({
                "path": str(path), "sha256": sha256_file(path), "row_count": len(actual_rows),
                "target_collision_count": len(found),
            })
    payload = {
        "status": "COLLISION" if collisions else "FRESH",
        "target_configuration_count": len(target_joints),
        "target_seed_count": len(target_seeds),
        "target_pair_count": len(target_joints) * len(target_seeds),
        "inventoried_line_oriented_file_count": len(inventory), "file_inventory": inventory,
        "scanned_result_file_count": len(scanned), "scanned_files": scanned,
        "collisions": [{"joint_fingerprint": joint, "training_seed": seed}
                       for joint, seed in sorted(collisions)],
        "seed_ranges_changed": False, "test_accessed": False,
    }
    if collisions:
        raise FreshSeedCollision(f"found {len(collisions)} previously evaluated fresh config/seed pairs")
    return payload


def _freeze_files(root: Path, relatives: Sequence[str], *, manifest_name: str,
                  root_name: str, marker_name: str) -> dict[str, Any]:
    ordered = sorted(relatives)
    files = [{"relative_path": relative, "sha256": sha256_file(root / relative)} for relative in ordered]
    root_sha = stable_hash([(row["relative_path"], row["sha256"]) for row in files])
    manifest = {"protocol_version": PROTOCOL_VERSION, "files": files, "root_sha256": root_sha,
                "created_at": utc_now(), "test_accessed": False}
    dump_json(root / manifest_name, manifest)
    dump_text(root / root_name, root_sha + "\n")
    dump_text(root / marker_name, root_sha + "\n")
    return manifest


def _verify_frozen_files(root: Path, *, manifest_name: str, root_name: str,
                         marker_name: str) -> dict[str, Any]:
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if [row["relative_path"] for row in files] != sorted(row["relative_path"] for row in files):
        raise ProtocolError(f"immutable manifest is not lexicographically ordered: {root}")
    pairs = []
    for entry in files:
        path = root / str(entry["relative_path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"immutable file changed: {path}")
        pairs.append((str(entry["relative_path"]), str(entry["sha256"])))
    observed = stable_hash(pairs)
    recorded = (
        manifest.get("root_sha256"), (root / root_name).read_text(encoding="utf-8").strip(),
        (root / marker_name).read_text(encoding="utf-8").strip(),
    )
    if not pairs or any(value != observed for value in recorded):
        raise ProtocolError(f"immutable root mismatch: {root}")
    return {"passed": True, "root_sha256": observed, "file_count": len(pairs)}


def freeze_pre_stage1(
    root: Path, universe: Sequence[Mapping[str, Any]], top6: Sequence[Mapping[str, Any]],
    p: Mapping[str, Any], protocol_identity: Mapping[str, Any], config: Mapping[str, Any],
    collision_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    universe_rows = candidate_universe_rows(universe)
    dump_csv(root / "candidate_universe.csv", universe_rows, UNIVERSE_FIELDS)
    dump_json(root / "top6_candidates.json", {
        "status": "FROZEN", "selection_uses_only_historical_seed_0_1_2_means": True,
        "ranking_round_decimals": 12, "tie_break": "joint_fingerprint_ascending",
        "candidates": [copy.deepcopy(dict(row)) for row in top6], "test_accessed": False,
    })
    provenance = [{
        "historical_rank": row["historical_rank"], "joint_fingerprint": row["joint_fingerprint"],
        "source_experiments": row["source_experiments"], "originally_selected_by": row["originally_selected_by"],
        "selection_role": "DESCRIPTIVE_ONLY",
    } for row in top6]
    dump_csv(root / "candidate_provenance.csv", provenance,
             ("historical_rank", "joint_fingerprint", "source_experiments", "originally_selected_by", "selection_role"))
    dump_json(root / "historical_P_identity.json", dict(p))
    dump_json(root / "stage1_seed_spec.json", {
        "training_seeds": list(STAGE1_SEEDS), "configurations": 7, "planned_GNN_runs": 105,
        "matched_reference_retraining": True, "track_test": False,
    })
    dump_json(root / "stage2_seed_spec.json", {
        "training_seeds": list(STAGE2_SEEDS), "configurations": 3, "planned_GNN_runs": 90,
        "disjoint_from_stage1": True, "matched_reference_retraining": True, "track_test": False,
    })
    dump_json(root / "stage2_selection_rule.json", {
        "input": "Stage1 matched paired differences only", "challenger_count": 2,
        "ranking": ["mean_paired_difference_descending", "joint_fingerprint_ascending"],
        "stage1_may_not_classify_final_accuracy": True,
        "final_gate_uses_stage2_only": True,
        "statistical_gates": copy.deepcopy(config["inference"]), "test_accessed": False,
    })
    dump_json(root / "evaluation_protocol_identity.json", dict(protocol_identity))
    dump_json(root / "fresh_seed_collision_audit.json", dict(collision_audit or {
        "status": "SYNTHETIC_NOT_SCANNED", "test_accessed": False,
    }))
    relatives = (
        "candidate_universe.csv", "top6_candidates.json", "candidate_provenance.csv",
        "historical_P_identity.json", "stage1_seed_spec.json", "stage2_seed_spec.json",
        "stage2_selection_rule.json", "evaluation_protocol_identity.json", "fresh_seed_collision_audit.json",
    )
    return _freeze_files(root, relatives, manifest_name="freeze_manifest.json",
                         root_name="pre_stage1_root_sha256.txt", marker_name="STAGE1_FROZEN")


def verify_pre_stage1(root: Path) -> dict[str, Any]:
    return _verify_frozen_files(root, manifest_name="freeze_manifest.json",
                                root_name="pre_stage1_root_sha256.txt", marker_name="STAGE1_FROZEN")


def _normalize_evaluator_candidate(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    joint = str(row["joint_fingerprint"])
    raw_hp = row.get("raw_hp") or {
        "lr": float(row["lr"]), "dropout": float(row["dropout"]),
        "hidden_dim": int(row["hidden_dim"]), "l2": float(row["l2"]),
    }
    return {
        **copy.deepcopy(dict(row)), "joint_fingerprint": joint,
        "joint_config_fingerprint": joint, "replicate_id": index,
        "round_id": 0, "candidate_slot": index, "pool_id": index, "batch_id": 0,
        "architecture_slot": index, "config_id": str(row.get("config_id", f"fresh-{joint[:12]}")),
        "raw_hp": raw_hp, "selected_by": [{"method": "FRESH_SEED_CONFIRMATION"}],
    }


def build_stage_tasks(
    configs: Sequence[Mapping[str, Any]], seeds: Sequence[int], stage: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = [_normalize_evaluator_candidate(row, index) for index, row in enumerate(sorted(configs, key=lambda r: str(r["joint_fingerprint"]))) ]
    protocol_identity = stable_hash({
        "dataset": "Cora", "split": "public", "max_epochs": 300, "patience": 80,
        "maximum_system_retries": 2, "stage": stage, "training_seeds": list(seeds),
        "track_test": False, "component_variant": "baseline",
    })
    tasks = []
    for index, candidate in enumerate(normalized):
        assigned_gpu = index % 2
        candidate["assigned_gpu"] = assigned_gpu
        candidate["training_protocol_identity"] = protocol_identity
        for seed in seeds:
            tasks.append({
                "joint_fingerprint": candidate["joint_fingerprint"], "training_seed": int(seed),
                "assigned_gpu": assigned_gpu, "task_key": f"fresh::{stage}::{candidate['joint_fingerprint']}::seed={seed}",
                "training_protocol_identity": protocol_identity, "candidate": candidate,
            })
    expected = {(str(row["joint_fingerprint"]), int(seed)) for row in normalized for seed in seeds}
    observed = {(str(row["joint_fingerprint"]), int(row["training_seed"])) for row in tasks}
    if expected != observed or len(tasks) != len(expected):
        raise ProtocolError(f"{stage} task construction did not produce one exact matched task per config/seed")
    return normalized, tasks


def evaluator_task_rows(tasks: Sequence[Mapping[str, Any]], stage: str) -> list[dict[str, Any]]:
    rows = base.evaluator_task_rows(tasks)
    for row in rows:
        row["task_stage"] = stage
    base.reject_test_named_fields(rows)
    return rows


def prepare_stage_evaluator_adapter(
    stage_root: Path, tasks: Sequence[Mapping[str, Any]], stage: str,
    seeds: Sequence[int], config: Mapping[str, Any], prerequisite_root: str,
) -> tuple[Path, list[dict[str, Any]], str]:
    adapter = stage_root / "evaluator_adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    rows = evaluator_task_rows(tasks, stage)
    dump_json(adapter / "pre_label_firewall_audit.json", {
        "passed": True, "stage": stage, "allowed_data_fields": ["x", "edge_index", "y", "train_mask", "val_mask"],
        "test_named_task_fields": [], "track_test": False,
        "prerequisite_immutable_root_sha256": prerequisite_root, "created_at": utc_now(),
    })
    dump_json(adapter / "selected_union_tasks.json", rows)
    dump_json(adapter / "preregistration.json", {
        "format_version": PROTOCOL_VERSION, "training": {
            "dataset": "Cora", "split": "public", "graph_transform": "NormalizeFeatures",
            "dataset_content_fingerprint": config["evaluation_protocol"]["dataset_content_fingerprint"],
            "split_fingerprint": config["evaluation_protocol"]["split_fingerprint"],
            "max_epochs": 300, "patience": 80, "maximum_system_retries": 2,
            "training_seeds": list(seeds), "track_test": False,
            "validation_only_data_view": True,
            "shard_to_gpu_uuid": copy.deepcopy(config["evaluation_protocol"]["shard_to_gpu_uuid"]),
        },
    })
    dependencies = list(base._frozen_dependency_entries())
    runtime_sources = (Path(__file__).resolve(), Path(base.__file__).resolve(), Path(diversity.__file__).resolve())
    by_path = {str(row["path"]): dict(row) for row in dependencies}
    for source in runtime_sources:
        by_path[str(source)] = {"path": str(source), "sha256": sha256_file(source)}
    dump_json(adapter / "source_dependency_manifest.json", {
        "format_version": PROTOCOL_VERSION,
        "files": [by_path[path] for path in sorted(by_path)],
        "includes_fresh_seed_wrapper_and_candidate_recovery_modules": True,
    })
    relatives = ("pre_label_firewall_audit.json", "preregistration.json",
                 "selected_union_tasks.json", "source_dependency_manifest.json")
    files = [{"relative_path": relative, "sha256": sha256_file(adapter / relative)} for relative in relatives]
    root_sha = stable_hash([(row["relative_path"], row["sha256"]) for row in files])
    dump_json(adapter / "pre_label_immutable_manifest.json", {
        "format_version": PROTOCOL_VERSION, "files": files, "manifest_root_hash": root_sha,
    })
    return adapter, rows, root_sha


def execute_evaluator_shard(adapter: Path, shard: int, output: Path,
                            cora_root: str, device: str) -> int:
    prereg = json.loads((adapter / "preregistration.json").read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in prereg["training"]["training_seeds"])
    if seeds not in {STAGE1_SEEDS, STAGE2_SEEDS}:
        raise ProtocolError(f"evaluator adapter contains unauthorized fresh seeds: {seeds}")
    previous = base.TRAINING_SEEDS
    try:
        base.TRAINING_SEEDS = seeds
        code = base.execute_evaluator_shard(adapter, shard, output, cora_root, device)
        modules = []
        for name, module in sorted(sys.modules.items()):
            source = getattr(module, "__file__", None)
            if not source:
                continue
            path = Path(source).resolve()
            if path.suffix == ".pyc" and path.with_suffix("").is_file():
                path = path.with_suffix("")
            try:
                path.relative_to(REPO_ROOT)
            except ValueError:
                continue
            if path.is_file() and path.suffix == ".py":
                modules.append({"module": name, "path": str(path), "sha256": sha256_file(path)})
        dump_json(output / "imported_local_modules.json", {"modules": modules, "test_accessed": False})
        return code
    finally:
        base.TRAINING_SEEDS = previous


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_stage_evaluation(
    output: Path, stage: str, configs: Sequence[Mapping[str, Any]], seeds: Sequence[int],
    config: Mapping[str, Any], prerequisite_root: str, log: Callable[[str], None],
    expected_source_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage_root = output / stage
    normalized, tasks = build_stage_tasks(configs, seeds, stage)
    adapter, evaluator_rows, adapter_root = prepare_stage_evaluator_adapter(
        stage_root, tasks, stage, seeds, config, prerequisite_root,
    )
    if expected_source_sha256 is not None and aggregate_source_sha256() != expected_source_sha256:
        raise ProtocolError(f"runtime source bytes changed before {stage} evaluator launch")
    dump_json(stage_root / "task_manifest.json", {
        "stage": stage, "authorized": True, "prerequisite_root_sha256": prerequisite_root,
        "evaluated_config_count": len(normalized), "planned_GNN_runs": len(evaluator_rows),
        "training_seeds": list(seeds), "matched_reference_retraining": True,
        "track_test": False, "tasks": evaluator_rows,
    })
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    handles = []
    for shard in (0, 1):
        shard_output = stage_root / f"gpu{shard}"
        shard_output.mkdir(parents=True, exist_ok=True)
        handle = (output / f"logs/{stage}_gpu{shard}.log").open("w", encoding="utf-8")
        handles.append(handle)
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(shard), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
        })
        inherited = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join([str(base.HIGH_SNAPSHOT), str(REPO_ROOT)] + ([inherited] if inherited else []))
        command = [
            sys.executable, str(Path(__file__).resolve()), "--evaluate-shard", "--adapter", str(adapter),
            "--shard", str(shard), "--shard-output", str(shard_output), "--cora-root", "Cora", "--device", "cuda:0",
        ]
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=handle,
                                   stderr=subprocess.STDOUT, text=True)
        processes.append((shard, process))
    update_state(output, stage="STAGE1_RUNNING" if stage == "stage1" else "STAGE2_RUNNING",
                 current_stage=stage, current_config=None, current_seed=None)
    return_codes: dict[str, int] = {}
    try:
        active = list(processes)
        while active:
            raw_count = sum(len(_load_jsonl(stage_root / f"gpu{shard}/seed_results.jsonl")) for shard in (0, 1))
            updates = {f"{stage}_completed_runs": raw_count}
            remaining = []
            for shard, process in active:
                code = process.poll()
                if code is None:
                    remaining.append((shard, process))
                else:
                    return_codes[str(shard)] = int(code)
            update_state(output, **updates)
            active = remaining
            if active:
                time.sleep(2.0)
    finally:
        for _shard, process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()
    task_lookup = {(str(row["joint_config_fingerprint"]), int(row["seed"])): row for row in evaluator_rows}
    raw = []
    loader_firewalls = {}
    for shard in (0, 1):
        plan_path = stage_root / f"gpu{shard}/run_plan.json"
        if plan_path.is_file():
            loader_firewalls[str(shard)] = json.loads(plan_path.read_text(encoding="utf-8"))
        for archived in _load_jsonl(stage_root / f"gpu{shard}/seed_results.jsonl"):
            key = (str(archived["joint_config_fingerprint"]), int(archived["seed"]))
            task = task_lookup.get(key)
            if task is None:
                raise StageEvaluationIncomplete(f"{stage} evaluator returned foreign task {key}")
            for field in ("architecture_fingerprint", "hp_fingerprint", "task_key", "assigned_gpu"):
                if archived.get(field) != task.get(field):
                    raise StageEvaluationIncomplete(f"{stage} evaluator identity mismatch for {field}")
            raw.append({
                "joint_fingerprint": key[0], "training_seed": key[1],
                "validation_accuracy": archived.get("val_acc"), "terminal_status": archived.get("terminal_status"),
                "valid": archived.get("valid"), "attempt_count": archived.get("attempt_count", 1),
                "retry_count": archived.get("retry_count", 0), "assigned_gpu": archived.get("assigned_gpu"),
            })
    audit = {
        "stage": stage, "adapter_manifest_root": adapter_root, "task_count": len(evaluator_rows),
        "completed_result_count": len(raw), "shard_return_codes": return_codes,
        "all_shards_exited_zero": set(return_codes) == {"0", "1"} and all(code == 0 for code in return_codes.values()),
        "loader_firewall": loader_firewalls,
        "loader_firewall_passed": set(loader_firewalls) == {"0", "1"} and all(
            row.get("validation_only_view_created") is True and row.get("test_named_fields_at_evaluator_boundary") == []
            for row in loader_firewalls.values()),
        "retry_preserved_exact_config_and_seed": True, "candidate_replacement": False,
        "track_test": False, "test_accessed": False,
    }
    log(f"{stage} evaluator finished: {len(raw)}/{len(evaluator_rows)} results, return_codes={return_codes}")
    return raw, audit


def validate_stage_results(
    raw: Sequence[Mapping[str, Any]], config_joints: Sequence[str], seeds: Sequence[int], stage: str,
) -> dict[str, Any]:
    expected = {(str(joint), int(seed)) for joint in config_joints for seed in seeds}
    seen: set[tuple[str, int]] = set()
    successful = set()
    failed = []
    for row in raw:
        key = (str(row["joint_fingerprint"]), int(row["training_seed"]))
        if key not in expected or key in seen:
            raise StageEvaluationIncomplete(f"{stage} has a foreign or duplicate result {key}")
        seen.add(key)
        value = row.get("validation_accuracy")
        if row.get("terminal_status") == "success" and row.get("valid") is True and value is not None and math.isfinite(float(value)):
            successful.add(key)
        else:
            failed.append(key)
    complete = seen == expected and successful == expected
    audit = {
        "stage": stage, "expected_runs": len(expected), "observed_runs": len(seen),
        "successful_runs": len(successful), "failed_or_missing_runs": len(expected - successful),
        "complete": complete, "all_configs_use_identical_matched_seed_set": complete,
        "track_test": False, "test_accessed": False,
    }
    if not complete:
        missing = sorted(expected - successful)
        raise StageEvaluationIncomplete(f"{stage} is incomplete: {len(missing)} failed or missing runs")
    return audit


def count_unique_successful_pairs(
    raw: Sequence[Mapping[str, Any]], config_joints: Sequence[str], seeds: Sequence[int],
) -> int:
    expected = {(str(joint), int(seed)) for joint in config_joints for seed in seeds}
    successful = set()
    for row in raw:
        try:
            key = (str(row["joint_fingerprint"]), int(row["training_seed"]))
            value = float(row["validation_accuracy"])
        except (KeyError, TypeError, ValueError):
            continue
        if (key in expected and row.get("terminal_status") == "success"
                and row.get("valid") is True and math.isfinite(value)):
            successful.add(key)
    return len(successful)


def paired_candidate_statistics(
    raw: Sequence[Mapping[str, Any]], candidate_joints: Sequence[str], p_joint: str,
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = {(str(row["joint_fingerprint"]), int(row["training_seed"])): float(row["validation_accuracy"]) for row in raw}
    paired = []
    stats = []
    for joint in sorted(map(str, candidate_joints)):
        deltas = []
        candidate_values = []
        p_values = []
        for seed in seeds:
            candidate_value = values[(joint, int(seed))]
            p_value = values[(p_joint, int(seed))]
            delta = candidate_value - p_value
            candidate_values.append(candidate_value)
            p_values.append(p_value)
            deltas.append(delta)
            paired.append({
                "joint_fingerprint": joint, "training_seed": int(seed),
                "candidate_validation": candidate_value, "P_validation": p_value,
                "paired_difference": delta,
            })
        tolerance = NUMERICAL_TOLERANCE
        stats.append({
            "joint_fingerprint": joint, "seed_count": len(deltas),
            "candidate_validation_mean": statistics.fmean(candidate_values),
            "candidate_validation_sd": statistics.stdev(candidate_values),
            "P_validation_mean": statistics.fmean(p_values), "P_validation_sd": statistics.stdev(p_values),
            "mean_paired_difference": statistics.fmean(deltas),
            "median_paired_difference": statistics.median(deltas),
            "paired_difference_sd": statistics.stdev(deltas),
            "wins": sum(delta > tolerance for delta in deltas),
            "losses": sum(delta < -tolerance for delta in deltas),
            "ties": sum(abs(delta) <= tolerance for delta in deltas),
            "paired_differences": deltas,
        })
    return paired, stats


def rank_stage1_candidates(stats: Sequence[Mapping[str, Any]], top_n: int = 2) -> list[dict[str, Any]]:
    ranked = sorted((dict(row) for row in stats),
                    key=lambda row: (-float(row["mean_paired_difference"]), str(row["joint_fingerprint"])))
    for rank, row in enumerate(ranked, 1):
        row["stage1_rank"] = rank
        row["advanced_to_stage2"] = rank <= top_n
    return ranked


def freeze_stage2_selection(stage1_root: Path, ranking: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [dict(row) for row in ranking[:2]]
    if len(selected) != 2 or len({row["joint_fingerprint"] for row in selected}) != 2:
        raise ProtocolError("Stage2 selection must contain two unique challengers")
    dump_json(stage1_root / "stage2_selection.json", {
        "status": "FROZEN", "selection_uses_stage1_only": True,
        "final_classification_made": False, "challengers": selected,
        "historical_P_joint_fingerprint": P_JOINT, "test_accessed": False,
    })
    manifest = _freeze_files(
        stage1_root,
        ("raw_seed_results.csv", "paired_differences.csv", "candidate_statistics.csv",
         "stage1_ranking.csv", "stage2_selection.json"),
        manifest_name="stage2_selection_manifest.json",
        root_name="stage2_selection_root_sha256.txt", marker_name="STAGE2_SELECTION_FROZEN",
    )
    return manifest


def verify_stage2_selection(stage1_root: Path) -> dict[str, Any]:
    return _verify_frozen_files(stage1_root, manifest_name="stage2_selection_manifest.json",
                                root_name="stage2_selection_root_sha256.txt", marker_name="STAGE2_SELECTION_FROZEN")


def paired_bootstrap(
    deltas: Sequence[float], rng: np.random.Generator | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    values = np.asarray(list(deltas), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not bool(np.isfinite(values).all()):
        raise ProtocolError("paired bootstrap requires a non-empty finite one-dimensional delta vector")
    generator = np.random.default_rng(BOOTSTRAP_SEED) if rng is None else rng
    indices = generator.integers(0, len(values), size=(int(replicates), len(values)))
    means = values[indices].mean(axis=1)
    return {
        "sampling_unit": "matched_training_seed", "seed_count": len(values),
        "replicates": int(replicates), "resampling": "with_replacement",
        "observed_mean": float(values.mean()),
        "bonferroni_one_sided_alpha": 0.025,
        "bonferroni_lower_bound_percentile": 2.5,
        "bonferroni_lower_bound": float(np.percentile(means, 2.5)),
        "descriptive_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "bootstrap_mean": float(means.mean()),
    }


def classify_stage2(stats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in stats]
    practical = [row for row in rows if row.get("PRACTICAL_GAIN") is True]
    stable = [row for row in rows if row.get("STABLE_POSITIVE_GAIN") is True]
    if practical:
        classification = "PRACTICAL_ACCURACY_IMPROVEMENT_CONFIRMED"
    elif stable:
        classification = "SMALL_STABLE_ACCURACY_GAIN"
    else:
        classification = "NO_STABLE_ACCURACY_GAIN"
    return {
        "classification": classification, "stage2_only_confirmatory": True,
        "stage1_labels_used_in_final_gate": False, "historical_seed_0_1_2_used_in_final_gate": False,
        "old_P_mean_used_in_final_gate": False, "predictor_score_used_in_final_gate": False,
        "method_provenance_used_in_final_gate": False,
        "practical_candidate_count": len(practical), "stable_candidate_count": len(stable),
        "test_accessed": False,
    }


def select_new_stable_best(stats: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    qualified = [dict(row) for row in stats if row.get("STABLE_POSITIVE_GAIN") is True]
    if not qualified:
        return None
    best_level = 2 if any(row.get("PRACTICAL_GAIN") is True for row in qualified) else 1
    qualified = [row for row in qualified if (2 if row.get("PRACTICAL_GAIN") is True else 1) == best_level]
    maximum = max(float(row["mean_paired_difference"]) for row in qualified)
    tied = [row for row in qualified if abs(float(row["mean_paired_difference"]) - maximum) <= NUMERICAL_TOLERANCE]
    winner = min(tied, key=lambda row: str(row["joint_fingerprint"]))
    winner["qualification_level"] = "PRACTICAL_GAIN" if best_level == 2 else "SMALL_STABLE_GAIN"
    return winner


def analyze_stage2(stats: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    bootstrap_results = {}
    for raw in sorted(stats, key=lambda row: str(row["joint_fingerprint"])):
        row = dict(raw)
        bootstrap = paired_bootstrap(row["paired_differences"], rng=generator)
        mean_delta = float(row["mean_paired_difference"])
        lower = float(bootstrap["bonferroni_lower_bound"])
        sp1 = mean_delta > NUMERICAL_TOLERANCE
        sp2 = lower > NUMERICAL_TOLERANCE
        stable = sp1 and sp2
        pg1 = stable
        pg2 = mean_delta >= PRACTICAL_MARGIN - NUMERICAL_TOLERANCE
        row.update({
            "standard_error_paired_difference": float(row["paired_difference_sd"]) / math.sqrt(int(row["seed_count"])),
            "bonferroni_lower_bound": lower, "descriptive_ci95": bootstrap["descriptive_ci95"],
            "SP1_mean_delta_positive": sp1, "SP2_bonferroni_lower_positive": sp2,
            "STABLE_POSITIVE_GAIN": stable, "PG1_stable_positive_gain": pg1,
            "PG2_mean_delta_at_least_0_003": pg2, "PRACTICAL_GAIN": pg1 and pg2,
        })
        results.append(row)
        bootstrap_results[str(row["joint_fingerprint"])] = bootstrap
    decision = classify_stage2(results)
    winner = select_new_stable_best(results)
    decision["new_stable_best_joint_fingerprint"] = None if winner is None else winner["joint_fingerprint"]
    decision["candidate_results"] = [{
        key: row[key] for key in (
            "joint_fingerprint", "mean_paired_difference", "bonferroni_lower_bound",
            "SP1_mean_delta_positive", "SP2_bonferroni_lower_positive",
            "STABLE_POSITIVE_GAIN", "PG2_mean_delta_at_least_0_003", "PRACTICAL_GAIN",
        )
    } for row in results]
    return results, bootstrap_results, decision


def winner_curse_analysis(
    stage2_joints: Sequence[str], universe_by_joint: Mapping[str, Mapping[str, Any]],
    stage1_stats: Sequence[Mapping[str, Any]], stage2_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage1 = {str(row["joint_fingerprint"]): row for row in stage1_stats}
    stage2 = {str(row["joint_fingerprint"]): row for row in stage2_stats}
    rows = []
    for joint in stage2_joints:
        original = float(universe_by_joint[joint]["historical_validation_mean"])
        stage1_mean = float(stage1[joint]["candidate_validation_mean"])
        stage2_mean = float(stage2[joint]["candidate_validation_mean"])
        rows.append({
            "joint_fingerprint": joint, "original_3seed_validation_mean": original,
            "stage1_15seed_validation_mean": stage1_mean, "stage2_30seed_validation_mean": stage2_mean,
            "original_to_stage1_shift": stage1_mean - original,
            "original_to_stage2_shift": stage2_mean - original,
            "role": "DESCRIPTIVE_ONLY", "final_gate_use": False,
        })
    return {"candidates": rows, "winner_curse_analysis_is_descriptive": True, "test_accessed": False}


def _raw_csv_fields() -> tuple[str, ...]:
    return ("joint_fingerprint", "training_seed", "validation_accuracy", "terminal_status",
            "valid", "attempt_count", "retry_count", "assigned_gpu")


def _paired_csv_fields() -> tuple[str, ...]:
    return ("joint_fingerprint", "training_seed", "candidate_validation", "P_validation", "paired_difference")


def _stats_csv_fields(stage2: bool = False) -> tuple[str, ...]:
    core = (
        "joint_fingerprint", "seed_count", "candidate_validation_mean", "candidate_validation_sd",
        "P_validation_mean", "P_validation_sd", "mean_paired_difference", "median_paired_difference",
        "paired_difference_sd", "wins", "losses", "ties",
    )
    if not stage2:
        return core
    return core + (
        "standard_error_paired_difference", "bonferroni_lower_bound", "descriptive_ci95",
        "SP1_mean_delta_positive", "SP2_bonferroni_lower_positive", "STABLE_POSITIVE_GAIN",
        "PG1_stable_positive_gain", "PG2_mean_delta_at_least_0_003", "PRACTICAL_GAIN",
    )


def write_stage1_analysis(output: Path, raw: Sequence[Mapping[str, Any]], top6: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage = output / "stage1"
    dump_csv(stage / "raw_seed_results.csv", raw, _raw_csv_fields())
    candidate_joints = [str(row["joint_fingerprint"]) for row in top6]
    paired, stats = paired_candidate_statistics(raw, candidate_joints, P_JOINT, STAGE1_SEEDS)
    ranking = rank_stage1_candidates(stats)
    dump_csv(stage / "paired_differences.csv", paired, _paired_csv_fields())
    dump_csv(stage / "candidate_statistics.csv", stats, _stats_csv_fields())
    dump_csv(stage / "stage1_ranking.csv", ranking, _stats_csv_fields() + ("stage1_rank", "advanced_to_stage2"))
    return stats, ranking


def write_stage2_analysis(
    output: Path, raw: Sequence[Mapping[str, Any]], stage2_joints: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    stage = output / "stage2"
    dump_csv(stage / "raw_seed_results.csv", raw, _raw_csv_fields())
    paired, basic_stats = paired_candidate_statistics(raw, stage2_joints, P_JOINT, STAGE2_SEEDS)
    stats, bootstraps, decision = analyze_stage2(basic_stats)
    dump_csv(stage / "paired_differences.csv", paired, _paired_csv_fields())
    dump_csv(stage / "candidate_statistics.csv", stats, _stats_csv_fields(stage2=True))
    dump_json(stage / "bootstrap_results.json", {
        "master_seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
        "candidate_order": sorted(stage2_joints), "results": bootstraps,
        "sampling_unit": "matched_training_seed", "test_accessed": False,
    })
    dump_json(stage / "multiplicity_control.json", {
        "method": "Bonferroni", "family_wise_alpha": 0.05, "comparison_count": 2,
        "one_sided_alpha_per_challenger": 0.025, "lower_bound_percentile": 2.5,
        "ordinary_descriptive_ci95_final_gate_use": False,
        "candidate_gates": decision["candidate_results"], "test_accessed": False,
    })
    p_values = [float(row["validation_accuracy"]) for row in raw if str(row["joint_fingerprint"]) == P_JOINT]
    fresh_p = {
        "joint_fingerprint": P_JOINT, "seed_count": len(p_values),
        "fresh_stage2_validation_mean": statistics.fmean(p_values),
        "fresh_stage2_validation_sd": statistics.stdev(p_values),
        "historical_old30_validation_mean": P_OLD_MEAN,
        "fresh_minus_historical_mean": statistics.fmean(p_values) - P_OLD_MEAN,
        "cross_seed_set_comparison_role": "DESCRIPTIVE_ONLY", "final_gate_use": False,
        "test_accessed": False,
    }
    dump_json(stage / "fresh_P_statistics.json", fresh_p)
    return stats, bootstraps, decision, paired


def write_final_candidate(
    output: Path, classification: str, winner: Mapping[str, Any] | None,
    top_by_joint: Mapping[str, Mapping[str, Any]], stage1_stats: Sequence[Mapping[str, Any]],
    stage2_raw: Sequence[Mapping[str, Any]], stage2_paired: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = output / "final_candidate"
    root.mkdir(parents=True, exist_ok=True)
    if classification not in {"PRACTICAL_ACCURACY_IMPROVEMENT_CONFIRMED", "SMALL_STABLE_ACCURACY_GAIN"}:
        return {"status": "NOT_FROZEN", "reason": "no stable positive challenger", "test_accessed": False}
    if winner is None:
        raise ProtocolError("positive classification lacks a new stable best")
    joint = str(winner["joint_fingerprint"])
    identity = copy.deepcopy(dict(top_by_joint[joint]))
    stage1 = next(row for row in stage1_stats if str(row["joint_fingerprint"]) == joint)
    candidate_rows = [dict(row) for row in stage2_raw if str(row["joint_fingerprint"]) == joint]
    paired_rows = [dict(row) for row in stage2_paired if str(row["joint_fingerprint"]) == joint]
    stable = {
        "classification": classification, "qualification_level": winner["qualification_level"],
        "joint_fingerprint": joint, "architecture_fingerprint": identity["architecture_fingerprint"],
        "hp_fingerprint": identity["hp_fingerprint"], "source_experiments": identity["source_experiments"],
        "originally_selected_by": identity["originally_selected_by"],
        "original_3seed_validation_mean": identity["historical_validation_mean"],
        "stage1_15seed_validation_mean": stage1["candidate_validation_mean"],
        "stage2_30seed_validation_mean": winner["candidate_validation_mean"],
        "fresh_P_stage2_mean": winner["P_validation_mean"],
        "mean_paired_difference": winner["mean_paired_difference"],
        "bonferroni_lower_bound": winner["bonferroni_lower_bound"],
        "candidate_level_claim_only": True, "method_superiority_claim": False,
        "test_accessed": False,
    }
    dump_json(root / "new_stable_best.json", stable)
    dump_csv(root / "new_stable_best_seed_results.csv", candidate_rows, _raw_csv_fields())
    dump_csv(root / "paired_comparison_vs_P.csv", paired_rows, _paired_csv_fields())
    dump_json(root / "candidate_identity.json", identity)
    dump_json(root / "candidate_provenance.json", {
        "joint_fingerprint": joint, "source_experiments": identity["source_experiments"],
        "originally_selected_by": identity["originally_selected_by"],
        "role": "DESCRIPTIVE_ONLY", "selection_or_gate_use": False,
    })
    relatives = (
        "candidate_identity.json", "candidate_provenance.json", "new_stable_best.json",
        "new_stable_best_seed_results.csv", "paired_comparison_vs_P.csv",
    )
    files = [{"relative_path": relative, "sha256": sha256_file(root / relative)} for relative in sorted(relatives)]
    root_sha = stable_hash([(row["relative_path"], row["sha256"]) for row in files])
    dump_json(root / "freeze_manifest.json", {"files": files, "root_sha256": root_sha})
    dump_text(root / "NEW_STABLE_BEST_FROZEN", root_sha + "\n")
    return {"status": "FROZEN", "root_sha256": root_sha, "joint_fingerprint": joint}


def _git_capture(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=False, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def _git_diff_patch() -> str:
    pieces = [_git_capture("diff", "HEAD", "--binary", "--", ".").rstrip()]
    for relative in IMPLEMENTATION_FILES:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative], cwd=REPO_ROOT,
                                 check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        path = REPO_ROOT / relative
        if tracked.returncode == 0 or not path.is_file():
            continue
        result = subprocess.run(["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative],
                                cwd=REPO_ROOT, check=False, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode not in {0, 1}:
            raise ProtocolError(f"cannot snapshot untracked source {relative}")
        pieces.append(result.stdout.rstrip())
    return "\n".join(piece for piece in pieces if piece) + "\n"


def write_environment_and_git(output: Path, *, before: bool) -> None:
    if before:
        environment = {
            "protocol_version": PROTOCOL_VERSION, "created_at": utc_now(),
            "python": sys.version, "platform": platform.platform(), "executable": sys.executable,
            "cwd": str(Path.cwd()), "branch": _git_capture("branch", "--show-current").strip(),
            "HEAD": _git_capture("rev-parse", "HEAD").strip(),
            "CUDA_initialized_in_parent": False, "GNN_training_in_parent": False,
            "test_accessed": False,
        }
        dump_json(output / "environment.json", environment)
        dump_text(output / "git_status_before.txt", _git_capture("status", "--porcelain=v1", "--branch"))
    else:
        dump_text(output / "git_status_after.txt", _git_capture("status", "--porcelain=v1", "--branch"))
        dump_text(output / "git_diff.patch", _git_diff_patch())


def _copy_preflight_logs(output: Path, preflight: Path | None) -> None:
    names = ("environment_audit.log", "py_compile.log", "pytest_focused.log", "synthetic_smoke.log", "dry_run.log")
    for name in names:
        destination = output / "logs" / name
        source = None if preflight is None else preflight / name
        if source is not None and source.is_file():
            shutil.copy2(source, destination)
        else:
            dump_text(destination, "preflight log unavailable\n")


def _logger(path: Path) -> Callable[[str], None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now()}] {message}\n")
            handle.flush()
    return log


def aggregate_source_sha256() -> str:
    rows = []
    for relative in IMPLEMENTATION_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ProtocolError(f"implementation source is missing: {relative}")
        rows.append((relative, sha256_file(path)))
    for path in sorted((Path(base.__file__).resolve(), Path(diversity.__file__).resolve()), key=str):
        rows.append((path.relative_to(REPO_ROOT).as_posix(), sha256_file(path)))
    return stable_hash(rows)


def _local_python_modules() -> list[dict[str, str]]:
    rows = []
    seen = set()
    for name, module in sorted(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        path = Path(source).resolve()
        if path.suffix == ".pyc":
            try:
                source_path = Path(importlib.util.source_from_cache(str(path)))
            except (ValueError, NotImplementedError):
                source_path = path
            if source_path.is_file():
                path = source_path
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if not path.is_file() or path.suffix != ".py" or path in seen:
            continue
        seen.add(path)
        rows.append({"module": name, "path": str(path), "relative_path": relative.as_posix(),
                     "sha256": sha256_file(path)})
    return rows


def snapshot_sources(output: Path, config: Mapping[str, Any] | None = None, *, best_effort: bool = False) -> dict[str, Any]:
    root = output / "source_snapshot"
    root.mkdir(parents=True, exist_ok=True)
    declared: list[tuple[str, Path]] = [(f"implementation/{relative}", REPO_ROOT / relative) for relative in IMPLEMENTATION_FILES]
    if config is not None:
        eval_spec = config.get("evaluation_protocol", {})
        for key in (
            "frozen_evaluator_source", "frozen_training_source", "dataset_loader_source",
            "model_builder_source", "hp_builder_source", "architecture_parser_source",
        ):
            if key in eval_spec:
                declared.append((f"declared/{key}/{Path(str(eval_spec[key])).name}", REPO_ROOT / str(eval_spec[key])))
        try:
            for index, entry in enumerate(base._frozen_dependency_entries()):
                declared.append((f"frozen_dependency/{index:02d}-{Path(entry['path']).name}", Path(entry["path"])))
        except Exception:
            if not best_effort:
                raise
    inventory = []
    missing = []
    used_destinations = set()
    for relative, source in declared:
        source = source.resolve()
        if not source.is_file():
            missing.append({"source": str(source), "requested_relative_path": relative})
            if not best_effort:
                raise InputIdentityInvalid(f"source snapshot input is missing: {source}")
            continue
        destination_relative = relative
        suffix = 1
        while destination_relative in used_destinations:
            destination_relative = f"{relative}.{suffix}"
            suffix += 1
        used_destinations.add(destination_relative)
        destination = root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        inventory.append({"role": "declared", "source_path": str(source),
                          "snapshot_path": destination_relative, "sha256": sha256_file(destination)})
    modules = _local_python_modules()
    child_modules = []
    for path in sorted(output.glob("stage?/gpu?/imported_local_modules.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        child_modules.extend(payload.get("modules", []))
    module_by_path = {str(row["path"]): row for row in modules}
    for row in child_modules:
        module_by_path.setdefault(str(row["path"]), dict(row))
    imported = []
    for index, row in enumerate(sorted(module_by_path.values(), key=lambda item: str(item["path"]))):
        source = Path(str(row["path"])).resolve()
        if not source.is_file():
            continue
        try:
            relative = source.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        destination_relative = f"imported_local/{relative}"
        destination = root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        imported.append({"source_path": str(source), "snapshot_path": destination_relative,
                         "sha256": sha256_file(destination), "index": index})
    dump_json(root / "imported_local_modules.json", {
        "parent_and_evaluator_child_modules": imported, "test_accessed": False,
    })
    all_files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            all_files.append({"relative_path": path.relative_to(root).as_posix(),
                              "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "files": all_files, "declared_inventory": inventory,
        "missing_best_effort": missing, "snapshot_root_sha256": stable_hash([(row["relative_path"], row["sha256"]) for row in all_files]),
        "actual_bytes_saved": True, "test_accessed": False,
    }
    dump_json(root / "sha256_manifest.json", manifest)
    return manifest


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
            rows.append({"relative_path": path.relative_to(output).as_posix(),
                         "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {
        "protocol_version": PROTOCOL_VERSION, "created_at": utc_now(), "file_count": len(rows),
        "files": rows, "manifest_root_sha256": stable_hash([(row["relative_path"], row["sha256"]) for row in rows]),
    }
    dump_json(output / "manifest.json", payload)
    return payload


def verify_manifest(output: Path) -> dict[str, Any]:
    return verify_directory_manifest(output)


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
    return {"status": "NOT_AVAILABLE", "classification": classification, "reason": reason,
            "test_accessed": False}


def ensure_required_outputs(output: Path, classification: str, reason: str | None) -> None:
    immutable_nonplaceholders = {
        "pre_stage1/pre_stage1_root_sha256.txt", "stage1/stage2_selection_root_sha256.txt",
        "source_snapshot/imported_local_modules.json", "source_snapshot/sha256_manifest.json",
        "manifest.json", "final_report.md", "state.json", "frozen_protocol.json", "decision.json",
        "environment.json", "git_status_before.txt", "git_status_after.txt", "git_diff.patch",
    }
    for relative in REQUIRED_OUTPUTS:
        path = output / relative
        if path.exists() or relative in immutable_nonplaceholders:
            continue
        if path.suffix == ".json":
            dump_json(path, _placeholder(classification, reason))
        elif path.suffix == ".csv":
            dump_csv(path, [], ("status",))


def render_final_report(artifacts: Mapping[str, Any]) -> str:
    decision = artifacts.get("decision", {})
    classification = decision.get("classification", artifacts.get("classification", "RUN_FAILED"))
    reason = artifacts.get("reason")
    universe_audit = artifacts.get("universe_audit", {})
    top6 = artifacts.get("top6", [])
    stage1_stats = artifacts.get("stage1_stats", [])
    stage2_stats = artifacts.get("stage2_stats", [])
    compatibility = artifacts.get("protocol_compatibility", {})
    winner_curse = artifacts.get("winner_curse", {})
    fresh_p = artifacts.get("fresh_p", {})
    lines = [f"# Fresh-Seed Accuracy Ceiling Confirmation v1 — {classification}", ""]
    content = {
        REPORT_SECTIONS[0]: f"Frozen classification: **{classification}**." + (f" Termination reason: {reason}." if reason else ""),
        REPORT_SECTIONS[1]: "This experiment asks whether the highest already-evaluated configurations beat historical reference P on completely fresh matched training seeds.",
        REPORT_SECTIONS[2]: "The original seed 0/1/2 values selected candidates and are subject to winner's curse; they never enter the final gate.",
        REPORT_SECTIONS[3]: "Stage1 screens six challengers on 15 fresh seeds; Stage2 alone confirms two frozen challengers on 30 disjoint fresh seeds.",
        REPORT_SECTIONS[4]: f"Input identity status: {artifacts.get('input_integrity', {}).get('status', 'NOT_AVAILABLE')}. Three immutable source archives and directory manifests were audited.",
        REPORT_SECTIONS[5]: f"Evaluator compatibility: {compatibility.get('status', 'NOT_AVAILABLE')}. Cora/public split and frozen validation-only training protocol are shared.",
        REPORT_SECTIONS[6]: f"Unique complete historical configurations: {universe_audit.get('unique_joint_config_count_before_P_exclusion', 'NOT_AVAILABLE')}; challenger universe after P exclusion: {universe_audit.get('challenger_universe_count', 'NOT_AVAILABLE')}.",
        REPORT_SECTIONS[7]: f"P joint `{P_JOINT}`; old30 mean {P_OLD_MEAN:.12f} and SD {P_OLD_SD:.12f} are descriptive only. P was freshly retrained on every matched seed.",
        REPORT_SECTIONS[8]: "\n".join(f"- {row.get('historical_rank')}. `{row.get('joint_fingerprint')}`: historical 3-seed mean {float(row.get('historical_validation_mean', 0)):.12f}" for row in top6) or "Not available.",
        REPORT_SECTIONS[9]: "Source experiment and original selection-method flags were reported descriptively and never affected ranking, screening, or classification.",
        REPORT_SECTIONS[10]: f"Frozen Stage1 seeds: {STAGE1_SEEDS[0]}..{STAGE1_SEEDS[-1]}; planned runs: 105.",
        REPORT_SECTIONS[11]: json.dumps(artifacts.get("stage1_completeness", _placeholder(classification, reason)), sort_keys=True),
        REPORT_SECTIONS[12]: "\n".join(f"- `{row.get('joint_fingerprint')}`: matched mean delta {float(row.get('mean_paired_difference', 0)):+.6f}" for row in stage1_stats) or "Not available.",
        REPORT_SECTIONS[13]: ", ".join(f"`{joint}`" for joint in artifacts.get("stage2_joints", [])) or "Not available.",
        REPORT_SECTIONS[14]: f"Frozen Stage2 seeds: {STAGE2_SEEDS[0]}..{STAGE2_SEEDS[-1]}; planned runs: 90; no Stage1 seed is reused.",
        REPORT_SECTIONS[15]: json.dumps(artifacts.get("stage2_completeness", _placeholder(classification, reason)), sort_keys=True),
        REPORT_SECTIONS[16]: "\n".join(f"- `{row.get('joint_fingerprint')}`: candidate {float(row.get('candidate_validation_mean', 0)):.6f}; matched P {float(row.get('P_validation_mean', 0)):.6f}" for row in stage2_stats) or "Not available.",
        REPORT_SECTIONS[17]: "\n".join(f"- `{row.get('joint_fingerprint')}`: mean {float(row.get('mean_paired_difference', 0)):+.6f}, median {float(row.get('median_paired_difference', 0)):+.6f}, SD {float(row.get('paired_difference_sd', 0)):.6f}" for row in stage2_stats) or "Not available.",
        REPORT_SECTIONS[18]: "Two simultaneous challenger comparisons use 20,000 paired-seed bootstrap replicates, master seed 0, and a one-sided Bonferroni alpha of 0.025; the gate uses the 2.5th percentile lower bound.",
        REPORT_SECTIONS[19]: "\n".join(f"- `{row.get('joint_fingerprint')}`: SP1={row.get('SP1_mean_delta_positive')}, SP2={row.get('SP2_bonferroni_lower_positive')}, stable={row.get('STABLE_POSITIVE_GAIN')}" for row in stage2_stats) or "Not available.",
        REPORT_SECTIONS[20]: "\n".join(f"- `{row.get('joint_fingerprint')}`: mean≥+0.003={row.get('PG2_mean_delta_at_least_0_003')}, practical={row.get('PRACTICAL_GAIN')}" for row in stage2_stats) or "Not available.",
        REPORT_SECTIONS[21]: "\n".join(f"- `{row.get('joint_fingerprint')}`: {row.get('wins')}/{row.get('losses')}/{row.get('ties')} wins/losses/ties" for row in stage2_stats) or "Not available.",
        REPORT_SECTIONS[22]: json.dumps(winner_curse, sort_keys=True),
        REPORT_SECTIONS[23]: json.dumps(fresh_p, sort_keys=True),
        REPORT_SECTIONS[24]: f"The preregistered Stage2-only classification is **{classification}**.",
        REPORT_SECTIONS[25]: f"Frozen new stable best: `{decision.get('new_stable_best_joint_fingerprint')}`." if decision.get("new_stable_best_joint_fingerprint") else "No new stable best was frozen.",
        REPORT_SECTIONS[26]: "Any positive result establishes only that a specific configuration is stable and better than P under this evaluator.",
        REPORT_SECTIONS[27]: "The candidates were not assigned by a randomized method comparison. Provenance cannot establish B4, P2, M1, M2, or Random superiority.",
        REPORT_SECTIONS[28]: "The result directly updates evidence about the best configuration currently found, not the causal quality of its discovery method.",
        REPORT_SECTIONS[29]: "If no stable gain is found, further surrogate ranking work should pause in favor of search-space, model-capacity, or training-ceiling investigation.",
        REPORT_SECTIONS[30]: "Only two Stage2 challengers and one dataset are confirmed; Stage1 screening can miss a true challenger, and seed-level inference does not generalize across datasets.",
        REPORT_SECTIONS[31]: "All identities, immutable freezes, matched tasks, validation-only firewall, source bytes, Git state, manifest, archive, and SHA-256 sidecar are recorded. Test access is false.",
    }
    for section in REPORT_SECTIONS:
        lines.extend([f"## {section}", "", content[section], ""])
    return "\n".join(lines).rstrip() + "\n"


def assert_complete_final_report(report: str) -> None:
    missing = [section for section in REPORT_SECTIONS if f"## {section}" not in report]
    if missing:
        raise ProtocolError(f"final report is missing sections: {missing}")


def finalize_bundle(
    output: Path, archive: Path, classification: str, artifacts: dict[str, Any],
    config: Mapping[str, Any] | None, *, reason: str | None = None, run_failed: bool = False,
) -> str:
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise ProtocolError(f"unregistered classification: {classification}")
    artifacts["classification"] = classification
    artifacts["reason"] = reason
    if not (output / "decision.json").is_file():
        decision = {
            "classification": classification, "reason": reason,
            "stage2_only_confirmatory": classification in {
                "PRACTICAL_ACCURACY_IMPROVEMENT_CONFIRMED", "SMALL_STABLE_ACCURACY_GAIN", "NO_STABLE_ACCURACY_GAIN"
            },
            "test_accessed": False,
        }
        dump_json(output / "decision.json", decision)
        artifacts["decision"] = decision
    update_state(output, stage="ANALYSIS_COMPLETE" if not run_failed else "RUN_FAILED",
                 classification=classification, current_stage=None, current_config=None,
                 current_seed=None, test_accessed=False)
    ensure_required_outputs(output, classification, reason)
    if not (output / "final_report.md").is_file():
        report = render_final_report(artifacts)
        assert_complete_final_report(report)
        dump_text(output / "final_report.md", report)
    write_environment_and_git(output, before=False)
    snapshot_sources(output, config, best_effort=run_failed or classification == "INPUT_IDENTITY_INVALID")
    dump_text(output / ("FAILED" if run_failed else "COMPLETE"), classification + "\n")
    write_manifest(output)
    return create_archive(output, archive)


def _synthetic_config_row(index: int, mean: float) -> dict[str, Any]:
    operations = ["GCNConv"]
    edges = [[0, 1]]
    architecture = stable_hash({"operations": operations, "edges": edges})
    raw_hp = [0.85 + index * 0.001, 0.86, 0.87, 0.88]
    hp = stable_hash(raw_hp)
    joint = stable_hash({"architecture_fingerprint": architecture, "hp_fingerprint": hp})
    return {
        "joint_fingerprint": joint, "architecture_fingerprint": architecture, "hp_fingerprint": hp,
        "operations": operations, "edges": edges, "effective_layers": 1,
        "raw_hp_vector": raw_hp, "raw_hp": {"lr": 0.01, "dropout": 0.5, "hidden_dim": 64, "l2": 0.0005},
        "lr": 0.01, "dropout": 0.5, "hidden_dim": 64, "l2": 0.0005,
        "gat_heads": 1, "sage_aggr": "mean", "gin_eps": 0.0,
        "gat_heads_by_layer": None, "sage_aggr_by_layer": None, "gin_eps_by_layer": None,
        "historical_seed_0": mean - 0.001, "historical_seed_1": mean,
        "historical_seed_2": mean + 0.001, "historical_validation_mean": mean,
        "historical_validation_sd": 0.001, "historical_score": round(mean, 12),
        "source_experiments": ["SYNTHETIC"], "originally_selected_by": ["SYNTHETIC"],
    }


def _synthetic_raw(configs: Sequence[Mapping[str, Any]], p: Mapping[str, Any],
                   seeds: Sequence[int], gains: Mapping[str, float]) -> list[dict[str, Any]]:
    rows = []
    for seed_index, seed in enumerate(seeds):
        p_value = 0.800 + (seed_index % 5) * 0.001
        rows.append({"joint_fingerprint": p["joint_fingerprint"], "training_seed": seed,
                     "validation_accuracy": p_value, "terminal_status": "success", "valid": True,
                     "attempt_count": 1, "retry_count": 0, "assigned_gpu": 0})
        for index, candidate in enumerate(configs):
            gain = float(gains[str(candidate["joint_fingerprint"])])
            rows.append({"joint_fingerprint": candidate["joint_fingerprint"], "training_seed": seed,
                         "validation_accuracy": p_value + gain + ((seed_index + index) % 3 - 1) * 0.0001,
                         "terminal_status": "success", "valid": True, "attempt_count": 1,
                         "retry_count": 0, "assigned_gpu": index % 2})
    return rows


def synthetic_smoke(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    validate_frozen_protocol(config)
    formal_before = DEFAULT_OUTPUT.exists()
    synthetic = [_synthetic_config_row(index, 0.810 - index * 0.001) for index in range(8)]
    ranked, top = rank_historical_candidates(synthetic, top_n=6)
    for rank, row in enumerate(top, 1):
        row["historical_rank"] = rank
    p = _synthetic_config_row(99, 0.800)
    p["role"] = "REFERENCE_CONFIGURATION"
    stage1_gains = {str(row["joint_fingerprint"]): 0.006 - index * 0.0005 for index, row in enumerate(top)}
    stage1_raw = _synthetic_raw(top, p, STAGE1_SEEDS, stage1_gains)
    stage1_audit = validate_stage_results(stage1_raw, [row["joint_fingerprint"] for row in top] + [p["joint_fingerprint"]], STAGE1_SEEDS, "stage1")
    _paired1, stats1 = paired_candidate_statistics(stage1_raw, [row["joint_fingerprint"] for row in top], p["joint_fingerprint"], STAGE1_SEEDS)
    stage1_ranking = rank_stage1_candidates(stats1)
    stage2_candidates = [next(row for row in top if row["joint_fingerprint"] == ranked_row["joint_fingerprint"]) for ranked_row in stage1_ranking[:2]]
    stage2_gains = {str(row["joint_fingerprint"]): 0.005 - index * 0.0005 for index, row in enumerate(stage2_candidates)}
    stage2_raw = _synthetic_raw(stage2_candidates, p, STAGE2_SEEDS, stage2_gains)
    stage2_audit = validate_stage_results(stage2_raw, [row["joint_fingerprint"] for row in stage2_candidates] + [p["joint_fingerprint"]], STAGE2_SEEDS, "stage2")
    _paired2, stats2 = paired_candidate_statistics(stage2_raw, [row["joint_fingerprint"] for row in stage2_candidates], p["joint_fingerprint"], STAGE2_SEEDS)
    analyzed, bootstrap, decision = analyze_stage2(stats2)
    with tempfile.TemporaryDirectory(prefix="fresh-ceiling-smoke-") as temporary:
        root = Path(temporary) / "pre_stage1"
        freeze_pre_stage1(root, ranked, top, p, {"status": "SYNTHETIC"}, config)
        frozen = verify_pre_stage1(root)
        stage1_root = Path(temporary) / "stage1"
        stage1_root.mkdir()
        dump_csv(stage1_root / "raw_seed_results.csv", stage1_raw, _raw_csv_fields())
        paired1, _ = paired_candidate_statistics(stage1_raw, [row["joint_fingerprint"] for row in top], p["joint_fingerprint"], STAGE1_SEEDS)
        dump_csv(stage1_root / "paired_differences.csv", paired1, _paired_csv_fields())
        dump_csv(stage1_root / "candidate_statistics.csv", stats1, _stats_csv_fields())
        dump_csv(stage1_root / "stage1_ranking.csv", stage1_ranking, _stats_csv_fields() + ("stage1_rank", "advanced_to_stage2"))
        selection = freeze_stage2_selection(stage1_root, stage1_ranking)
        selection_verified = verify_stage2_selection(stage1_root)
    passed = all((
        len(ranked) == 8, len(top) == 6, stage1_audit["complete"], stage2_audit["complete"],
        len(bootstrap) == 2, decision["classification"] == "PRACTICAL_ACCURACY_IMPROVEMENT_CONFIRMED",
        all(row["PRACTICAL_GAIN"] for row in analyzed), frozen["passed"],
        selection["root_sha256"] == selection_verified["root_sha256"],
        DEFAULT_OUTPUT.exists() == formal_before,
    ))
    return {
        "passed": passed, "synthetic_universe_count": len(ranked), "synthetic_top6_count": len(top),
        "stage1_complete": stage1_audit["complete"], "stage2_complete": stage2_audit["complete"],
        "synthetic_classification": decision["classification"],
        "formal_output_created": DEFAULT_OUTPUT.exists() != formal_before,
        "GNN_training_started": False, "evaluator_subprocess_started": False,
        "CUDA_initialized": False, "test_accessed": False,
    }


def execute_dry_run(config_path: Path, formal_output: Path = DEFAULT_OUTPUT) -> int:
    existed_before = formal_output.exists()
    config = load_config(config_path)
    protocol = validate_frozen_protocol(config)
    input_integrity, compatibility = validate_authoritative_inputs(config)
    universe, universe_audit = build_candidate_universe(config)
    top6 = select_top6(universe)
    p = recover_historical_p(config)
    payload = {
        "status": "DRY_RUN_READY", "protocol": protocol,
        "input_file_identity": input_integrity["status"],
        "protocol_compatibility": compatibility["status"],
        "expected_candidate_union_count": EXPECTED_UNIVERSE,
        "observed_candidate_union_count": universe_audit["unique_joint_config_count_before_P_exclusion"],
        "top6_identity_check": [row["joint_fingerprint"] for row in top6] == list(EXPECTED_TOP6),
        "historical_P_identity": p["joint_fingerprint"],
        "stage1_planned_GNN_runs": 105, "stage2_planned_GNN_runs": 90,
        "total_planned_GNN_runs": 195, "fresh_seed_collision_scan_started": False,
        "formal_output_existed_before": existed_before,
        "formal_output_created": formal_output.exists() and not existed_before,
        "GNN_training_started": False, "evaluator_subprocess_started": False,
        "CUDA_initialized": False, "test_accessed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _prepare_output(output: Path, config: Mapping[str, Any], config_sha256: str,
                    source_sha256: str, preflight: Path | None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("logs", "pre_stage1", "stage1", "stage2", "analysis", "final_candidate"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    dump_json(output / "state.json", initial_state(config_sha256, source_sha256))
    dump_json(output / "frozen_protocol.json", dict(config))
    _copy_preflight_logs(output, preflight)
    write_environment_and_git(output, before=True)


def _science_terminal(
    output: Path, archive: Path, classification: str, reason: str,
    artifacts: dict[str, Any], config: Mapping[str, Any],
) -> int:
    dump_json(output / "decision.json", {
        "classification": classification, "reason": reason,
        "final_gate_executed": False, "test_accessed": False,
    })
    artifacts["decision"] = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    finalize_bundle(output, archive, classification, artifacts, config, reason=reason)
    return 0


def execute_formal(
    config_path: Path, output: Path, config_sha256: str, source_sha256: str,
    preflight_log_dir: Path | None,
) -> int:
    archive = Path(str(output) + ".tar.gz")
    ensure_output_policy(output, archive)
    if not config_sha256 or sha256_file(config_path) != config_sha256:
        raise ProtocolError("runner-provided config SHA-256 differs from runtime bytes")
    if not source_sha256 or aggregate_source_sha256() != source_sha256:
        raise ProtocolError("runner-provided source identity differs from runtime bytes")
    config = load_config(config_path)
    validate_frozen_protocol(config)
    _prepare_output(output, config, config_sha256, source_sha256, preflight_log_dir)
    input_log = _logger(output / "logs/input_audit.log")
    stage1_log = _logger(output / "logs/stage1_evaluation.log")
    stage2_log = _logger(output / "logs/stage2_evaluation.log")
    analysis_log = _logger(output / "logs/analysis.log")
    artifacts: dict[str, Any] = {"test_accessed": False}

    try:
        integrity, compatibility = validate_authoritative_inputs(config)
        dump_json(output / "input_integrity.json", integrity)
        dump_json(output / "protocol_compatibility.json", compatibility)
        artifacts.update({"input_integrity": integrity, "protocol_compatibility": compatibility})
        input_log("three authoritative archives, classifications, manifests, and evaluator protocols validated")
    except InputProtocolMismatch as error:
        dump_json(output / "input_integrity.json", _placeholder("INPUT_PROTOCOL_MISMATCH", str(error)))
        dump_json(output / "protocol_compatibility.json", {"status": "MISMATCH", "reason": str(error), "test_accessed": False})
        return _science_terminal(output, archive, "INPUT_PROTOCOL_MISMATCH", str(error), artifacts, config)
    except (InputIdentityInvalid, base.InputIdentityInvalid) as error:
        dump_json(output / "input_integrity.json", {"status": "INVALID", "reason": str(error), "test_accessed": False})
        dump_json(output / "protocol_compatibility.json", _placeholder("INPUT_IDENTITY_INVALID", str(error)))
        return _science_terminal(output, archive, "INPUT_IDENTITY_INVALID", str(error), artifacts, config)
    update_state(output, stage="INPUTS_VALIDATED")

    try:
        universe, universe_audit = build_candidate_universe(config)
        top6 = select_top6(universe)
        p = recover_historical_p(config)
    except (InputIdentityInvalid, base.InputIdentityInvalid) as error:
        return _science_terminal(output, archive, "INPUT_IDENTITY_INVALID", str(error), artifacts, config)
    except CandidateRecoveryFailed as error:
        return _science_terminal(output, archive, "CANDIDATE_RECOVERY_FAILED", str(error), artifacts, config)
    artifacts.update({"universe_audit": universe_audit, "top6": top6, "historical_P": p})
    rows = candidate_universe_rows(universe)
    dump_csv(output / "candidate_universe.csv", rows, UNIVERSE_FIELDS)
    dump_json(output / "historical_P_identity.json", p)
    update_state(output, source_config_count=1121, unique_candidate_count=1121, screen_candidate_count=6)

    try:
        collision = audit_fresh_seed_collisions(
            [*top6, p], [REPO_ROOT / "results/search", REPO_ROOT / "results/posthoc"], exclude_root=output,
        )
    except FreshSeedCollision as error:
        dump_json(output / "pre_stage1/fresh_seed_collision_audit.json", {
            "status": "COLLISION", "reason": str(error), "test_accessed": False,
        })
        return _science_terminal(output, archive, "FRESH_SEED_COLLISION", str(error), artifacts, config)
    freeze = freeze_pre_stage1(output / "pre_stage1", universe, top6, p, compatibility, config, collision)
    verified = verify_pre_stage1(output / "pre_stage1")
    artifacts["pre_stage1_freeze"] = verified
    update_state(output, stage="STAGE1_FROZEN")

    stage1_configs = [*top6, p]
    try:
        verify_pre_stage1(output / "pre_stage1")
        stage1_raw, stage1_training = run_stage_evaluation(
            output, "stage1", stage1_configs, STAGE1_SEEDS, config, verified["root_sha256"], stage1_log,
            expected_source_sha256=source_sha256,
        )
        stage1_completeness = validate_stage_results(
            stage1_raw, [row["joint_fingerprint"] for row in stage1_configs], STAGE1_SEEDS, "stage1",
        )
        if not stage1_training["all_shards_exited_zero"] or not stage1_training["loader_firewall_passed"]:
            raise StageEvaluationIncomplete("Stage1 evaluator subprocess or validation-only firewall failed")
    except StageEvaluationIncomplete as error:
        if "stage1_raw" in locals():
            dump_csv(output / "stage1/raw_seed_results.csv", stage1_raw, _raw_csv_fields())
        incomplete_rows = locals().get("stage1_raw", [])
        successful_count = count_unique_successful_pairs(
            incomplete_rows, [row["joint_fingerprint"] for row in stage1_configs], STAGE1_SEEDS,
        )
        update_state(output, stage1_completed_runs=successful_count,
                     stage1_failed_runs=105 - successful_count)
        return _science_terminal(output, archive, "STAGE1_EVALUATION_INCOMPLETE", str(error), artifacts, config)
    artifacts["stage1_completeness"] = stage1_completeness
    stage1_stats, stage1_ranking = write_stage1_analysis(output, stage1_raw, top6)
    artifacts["stage1_stats"] = stage1_stats
    update_state(output, stage="STAGE1_COMPLETE", stage1_completed_runs=105, stage1_failed_runs=0,
                 current_stage=None, current_config=None, current_seed=None)
    selection_manifest = freeze_stage2_selection(output / "stage1", stage1_ranking)
    selection_verified = verify_stage2_selection(output / "stage1")
    stage2_joints = [str(row["joint_fingerprint"]) for row in stage1_ranking[:2]]
    artifacts.update({"stage2_joints": stage2_joints, "stage2_selection": selection_verified})
    update_state(output, stage="STAGE2_SELECTION_FROZEN", stage2_candidate_count=2)

    top_by_joint = {str(row["joint_fingerprint"]): row for row in top6}
    stage2_configs = [top_by_joint[joint] for joint in stage2_joints] + [p]
    try:
        verify_pre_stage1(output / "pre_stage1")
        verify_stage2_selection(output / "stage1")
        stage2_raw, stage2_training = run_stage_evaluation(
            output, "stage2", stage2_configs, STAGE2_SEEDS, config,
            selection_manifest["root_sha256"], stage2_log,
            expected_source_sha256=source_sha256,
        )
        stage2_completeness = validate_stage_results(
            stage2_raw, [row["joint_fingerprint"] for row in stage2_configs], STAGE2_SEEDS, "stage2",
        )
        if not stage2_training["all_shards_exited_zero"] or not stage2_training["loader_firewall_passed"]:
            raise StageEvaluationIncomplete("Stage2 evaluator subprocess or validation-only firewall failed")
    except StageEvaluationIncomplete as error:
        if "stage2_raw" in locals():
            dump_csv(output / "stage2/raw_seed_results.csv", stage2_raw, _raw_csv_fields())
        incomplete_rows = locals().get("stage2_raw", [])
        successful_count = count_unique_successful_pairs(
            incomplete_rows, [row["joint_fingerprint"] for row in stage2_configs], STAGE2_SEEDS,
        )
        update_state(output, stage2_completed_runs=successful_count,
                     stage2_failed_runs=90 - successful_count)
        return _science_terminal(output, archive, "STAGE2_EVALUATION_INCOMPLETE", str(error), artifacts, config)
    artifacts["stage2_completeness"] = stage2_completeness
    update_state(output, stage="STAGE2_COMPLETE", stage2_completed_runs=90, stage2_failed_runs=0,
                 current_stage=None, current_config=None, current_seed=None)

    stage2_stats, bootstraps, decision, stage2_paired = write_stage2_analysis(output, stage2_raw, stage2_joints)
    winner = select_new_stable_best(stage2_stats)
    universe_by_joint = {str(row["joint_fingerprint"]): row for row in universe}
    winner_curse = winner_curse_analysis(stage2_joints, universe_by_joint, stage1_stats, stage2_stats)
    dump_json(output / "analysis/winner_curse_analysis.json", winner_curse)
    dump_json(output / "analysis/provenance_summary.json", {
        "top6": [{"joint_fingerprint": row["joint_fingerprint"], "source_experiments": row["source_experiments"],
                  "originally_selected_by": row["originally_selected_by"]} for row in top6],
        "selection_and_gate_use": False, "test_accessed": False,
    })
    p_stage1_values = [float(row["validation_accuracy"]) for row in stage1_raw if row["joint_fingerprint"] == P_JOINT]
    fresh_p = json.loads((output / "stage2/fresh_P_statistics.json").read_text(encoding="utf-8"))
    historical_vs_fresh = {
        "old30_mean": P_OLD_MEAN, "old30_sd": P_OLD_SD, "old30_ci95": list(P_OLD_CI),
        "stage1_15seed_mean": statistics.fmean(p_stage1_values),
        "stage2_30seed_mean": fresh_p["fresh_stage2_validation_mean"],
        "stage1_minus_old30": statistics.fmean(p_stage1_values) - P_OLD_MEAN,
        "stage2_minus_old30": fresh_p["fresh_stage2_validation_mean"] - P_OLD_MEAN,
        "role": "DESCRIPTIVE_ONLY", "failure_condition": False, "test_accessed": False,
    }
    dump_json(output / "analysis/historical_vs_fresh_P.json", historical_vs_fresh)
    boundary = {
        "randomized_method_comparison": False, "candidate_level_accuracy_claim_only": True,
        "method_level_superiority_claim": False, "provenance_used_for_selection_or_gate": False,
        "B4_P2_M1_M2_Random_superiority_established": False, "test_accessed": False,
    }
    dump_json(output / "analysis/method_attribution_boundary.json", boundary)
    classification = str(decision["classification"])
    final_candidate = write_final_candidate(
        output, classification, winner, top_by_joint, stage1_stats, stage2_raw, stage2_paired,
    )
    decision.update({
        "final_candidate_freeze": final_candidate,
        "stage1_screening_only": True, "Stage2_is_only_confirmatory_dataset": True,
        "candidate_level_accuracy_claim_only": True, "method_superiority_claim": False,
        "planned_GNN_runs": 195, "completed_GNN_runs": 195,
        "track_test": False, "test_accessed": False,
    })
    dump_json(output / "decision.json", decision)
    artifacts.update({
        "stage2_stats": stage2_stats, "bootstrap_results": bootstraps, "decision": decision,
        "winner_curse": winner_curse, "fresh_p": fresh_p, "historical_vs_fresh_P": historical_vs_fresh,
        "method_attribution_boundary": boundary, "final_candidate": final_candidate,
    })
    report = render_final_report(artifacts)
    assert_complete_final_report(report)
    dump_text(output / "final_report.md", report)
    analysis_log(f"frozen Stage2-only classification: {classification}")
    update_state(output, stage="ANALYSIS_COMPLETE", classification=classification)
    finalize_bundle(output, archive, classification, artifacts, config)
    return 0


def write_failure_bundle(
    output: Path, message: str, *, config_path: Path = DEFAULT_CONFIG,
    config_sha256: str = "UNAVAILABLE", source_sha256: str = "UNAVAILABLE",
) -> None:
    archive = Path(str(output) + ".tar.gz")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    for directory in ("logs", "pre_stage1", "stage1", "stage2", "analysis", "final_candidate"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    try:
        config = load_config(config_path)
    except Exception:
        config = None
    if not (output / "state.json").is_file():
        dump_json(output / "state.json", initial_state(config_sha256, source_sha256))
    if not (output / "frozen_protocol.json").is_file():
        dump_json(output / "frozen_protocol.json", config if config is not None else _placeholder("RUN_FAILED", message))
    if not (output / "environment.json").is_file():
        write_environment_and_git(output, before=True)
    for relative in ("input_integrity.json", "protocol_compatibility.json"):
        if not (output / relative).is_file():
            dump_json(output / relative, _placeholder("RUN_FAILED", message))
    dump_json(output / "error.json", {
        "classification": "RUN_FAILED", "message": message, "traceback": traceback.format_exc(),
        "test_accessed": False,
    })
    decision = {"classification": "RUN_FAILED", "reason": message, "test_accessed": False}
    dump_json(output / "decision.json", decision)
    finalize_bundle(output, archive, "RUN_FAILED", {"decision": decision, "test_accessed": False},
                    config, reason=message, run_failed=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--source-sha256", default="")
    parser.add_argument("--preflight-log-dir", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--shard", type=int)
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
            raise ProtocolError("evaluator child requires adapter, shard, and shard-output")
        return execute_evaluator_shard(args.adapter, args.shard, args.shard_output,
                                       args.cora_root, args.device)
    if args.write_failure_bundle is not None:
        write_failure_bundle(args.output, args.write_failure_bundle, config_path=args.config,
                             config_sha256=args.config_sha256 or "UNAVAILABLE",
                             source_sha256=args.source_sha256 or "UNAVAILABLE")
        return 0
    try:
        return execute_formal(args.config, args.output, args.config_sha256, args.source_sha256,
                              args.preflight_log_dir)
    except Exception as error:
        if not args.output.exists():
            raise
        try:
            write_failure_bundle(args.output, f"{type(error).__name__}: {error}",
                                 config_path=args.config,
                                 config_sha256=args.config_sha256 or "UNAVAILABLE",
                                 source_sha256=args.source_sha256 or "UNAVAILABLE")
        except Exception as bundle_error:
            print(f"failure bundle creation also failed: {bundle_error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
