#!/usr/bin/env python3
"""Zero-GNN post-hoc cross-method search-collapse diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
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
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "cross_method_search_collapse_diagnosis_global4_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/cross_method_search_collapse_diagnosis_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/posthoc/cross-method-search-collapse-diagnosis-global4-v1"
B4_ROOT = REPO_ROOT / "results/posthoc/b4-guided-search-accuracy-gate-global4-v1"
METRIC_ROOT = REPO_ROOT / "results/posthoc/m1-m2-metric-guided-search-accuracy-gate-global4-v1"
REPLICATES = 8
ROUNDS = 4
CANDIDATES_PER_ROUND = 128
SELECTED_PER_ROUND = 4
RANKING_K = (4, 8, 16, 32)
TOLERANCE = 1.0e-12
METHODS = ("B4", "M1", "M2")
ALL_METHODS = ("B4", "B4_RANDOM", "M1", "M2", "M12_RANDOM")
PAIR_RANDOM = {"B4": "B4_RANDOM", "M1": "M12_RANDOM", "M2": "M12_RANDOM"}
ALLOWED_CLASSIFICATIONS = {
    "DIAGNOSIS_INPUT_INVALID", "NO_SYSTEMATIC_SEARCH_COLLAPSE",
    "COLLAPSE_PRESENT_BUT_DIVERSITY_TEST_NOT_JUSTIFIED",
    "DIVERSITY_TEST_AUTHORIZED", "RUN_FAILED",
}
IMPLEMENTATION_FILES = (
    "configs/cross_method_search_collapse_diagnosis_v1.yaml",
    "analyse/cross_method_search_collapse_diagnosis.py",
    "scripts/run_cross_method_search_collapse_diagnosis.sh",
    "tests/test_cross_method_search_collapse_diagnosis.py",
)
REPORT_SECTIONS = (
    "1. Executive conclusion",
    "2. Why this diagnosis is post-hoc rather than confirmatory",
    "3. Input integrity",
    "4. Historical search outcomes",
    "5. Candidate-pool architecture availability",
    "6. B4 architecture concentration",
    "7. M1 architecture concentration",
    "8. M2 architecture concentration",
    "9. Paired Random architecture diversity",
    "10. Predictor ranking concentration",
    "11. Within-round duplicates",
    "12. Cross-round repeat consumption",
    "13. Architecture refinement from repeated HP evaluations",
    "14. Whether final winners came from repeated architectures",
    "15. Diversity vs final-best association",
    "16. Anytime saturation",
    "17. Cross-method consistency",
    "18. P1 structural effect",
    "19. P2 structural effect",
    "20. Rank sacrifice",
    "21. Prediction-score sacrifice",
    "22. Existing-label coverage",
    "23. Why observed subset replay is not causal evidence",
    "24. Frozen diagnostic gates",
    "25. Final classification",
    "26. Whether a prospective diversity test is authorized",
    "27. Selected scorer for next experiment",
    "28. Selected diversity policy",
    "29. What the next prospective experiment must test",
    "30. What this diagnosis does NOT prove",
)


class DiagnosisError(RuntimeError):
    """Base protocol error."""


class InputInvalid(DiagnosisError):
    """Frozen input identity or schema is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def dump_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return value


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    import yaml
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosisError("protocol config is not a mapping")
    return value


def validate_protocol(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise DiagnosisError("protocol version mismatch")
    if config.get("new_gnn_runs") != 0 or config.get("gpu_use") != "forbidden":
        raise DiagnosisError("zero-compute contract changed")
    if config.get("test_access") != "forbidden":
        raise DiagnosisError("test firewall changed")
    grid = config["search_grid"]
    expected = {
        "replicates": REPLICATES, "rounds": ROUNDS,
        "candidates_per_round": CANDIDATES_PER_ROUND,
        "selected_per_round": SELECTED_PER_ROUND,
        "selected_slots_per_method": 128,
        "ranking_top_k": list(RANKING_K),
    }
    if any(grid.get(key) != value for key, value in expected.items()):
        raise DiagnosisError("search grid changed")
    if float(config["policy_feasibility"]["tolerance"]) != TOLERANCE:
        raise DiagnosisError("policy tolerance changed")
    allowed = set(config["authorization"]["allowed_classifications"])
    if allowed != ALLOWED_CLASSIFICATIONS:
        raise DiagnosisError("classification set changed")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout


def _git_diff_with_untracked() -> str:
    pieces = [_git("diff", "HEAD", "--binary", "--", ".").rstrip()]
    for relative in IMPLEMENTATION_FILES:
        path = REPO_ROOT / relative
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative], cwd=REPO_ROOT,
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if tracked or not path.is_file():
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative],
            cwd=REPO_ROOT, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode not in (0, 1):
            raise DiagnosisError(f"cannot capture source addition: {relative}")
        pieces.append(result.stdout.rstrip())
    return "\n".join(piece for piece in pieces if piece) + "\n"


def verify_directory_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise InputInvalid(f"missing input manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recompute = []
    for entry in manifest.get("files", []):
        file_path = root / str(entry["relative_path"])
        if not file_path.is_file():
            raise InputInvalid(f"manifest file missing: {file_path}")
        observed = sha256_file(file_path)
        if observed != entry.get("sha256") or file_path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise InputInvalid(f"manifest mismatch: {file_path}")
        recompute.append((str(entry["relative_path"]), str(entry["sha256"])))
    root_hash = stable_hash(recompute)
    if root_hash != manifest.get("manifest_root_sha256"):
        raise InputInvalid(f"manifest root mismatch: {root}")
    return {
        "root": str(root), "file_count": len(recompute),
        "manifest_root_sha256": root_hash, "all_file_hashes_valid": True,
    }


def verify_pre_search_freeze(root: Path) -> dict[str, Any]:
    pre = root / "pre_search"
    freeze = json.loads((pre / "freeze_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in freeze["files"]:
        path = pre / entry["relative_path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise InputInvalid(f"pre-search frozen file mismatch: {path}")
        rows.append((entry["relative_path"], entry["sha256"]))
    observed = stable_hash(rows)
    markers = [
        freeze.get("root_sha256"),
        (pre / "pre_search_root_sha256.txt").read_text(encoding="utf-8").strip(),
        (pre / "PRE_SEARCH_FROZEN").read_text(encoding="utf-8").strip(),
    ]
    if any(value != observed for value in markers):
        raise InputInvalid(f"pre-search root mismatch: {root}")
    return {"root_sha256": observed, "file_count": len(rows)}


def validate_input_root(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    if not (root / "COMPLETE").is_file():
        raise InputInvalid(f"COMPLETE missing: {root}")
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    decision = json.loads((root / "decision.json").read_text(encoding="utf-8"))
    if decision.get("classification") != spec["required_classification"]:
        raise InputInvalid(f"classification mismatch: {root}")
    if state.get("classification") != spec["required_classification"]:
        raise InputInvalid(f"state classification mismatch: {root}")
    if state.get("test_accessed") is not False or decision.get("test_accessed") is not False:
        raise InputInvalid(f"input test firewall is not clean: {root}")
    archive = REPO_ROOT / spec["archive"]
    archive_check: dict[str, Any]
    if archive.is_file():
        observed = sha256_file(archive)
        if observed != spec["archive_sha256"]:
            raise InputInvalid(f"archive SHA mismatch: {archive}")
        sidecar = Path(str(archive) + ".sha256")
        if sidecar.is_file() and sidecar.read_text(encoding="utf-8").split()[0] != observed:
            raise InputInvalid(f"archive sidecar mismatch: {sidecar}")
        archive_check = {"present": True, "sha256": observed, "sidecar_valid_if_present": True}
    else:
        archive_check = {"present": False, "fallback": "directory_manifest_and_internal_hashes"}
    return {
        "classification": decision["classification"], "test_accessed": False,
        "archive": archive_check, "manifest": verify_directory_manifest(root),
        "pre_search": verify_pre_search_freeze(root),
    }


def _candidate_rows(path: Path, score_fields: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        row: dict[str, Any] = {
            "replicate_id": int(raw["replicate_id"]), "round_id": int(raw["round_id"]),
            "candidate_slot": int(raw["candidate_slot"]),
            "joint_fingerprint": raw["joint_fingerprint"],
            "architecture_fingerprint": raw["architecture_fingerprint"],
        }
        for field in score_fields:
            row[field] = float(raw[field])
        rows.append(row)
    if len(rows) != REPLICATES * ROUNDS * CANDIDATES_PER_ROUND:
        raise InputInvalid(f"candidate count mismatch: {path}")
    counts = Counter((row["replicate_id"], row["round_id"]) for row in rows)
    if set(counts.values()) != {CANDIDATES_PER_ROUND} or len(counts) != REPLICATES * ROUNDS:
        raise InputInvalid(f"candidate pool grid mismatch: {path}")
    if len({row["joint_fingerprint"] for row in rows}) != len(rows):
        raise InputInvalid(f"candidate joints are not unique: {path}")
    return rows


def _labels(path: Path) -> dict[str, float]:
    values = {}
    for row in read_csv(path):
        joint = row["joint_fingerprint"]
        if joint in values:
            raise InputInvalid(f"duplicate evaluated joint: {path}")
        values[joint] = float(row["validation_mean"])
    return values


def _selection_records(
    plan_path: Path, candidates: Sequence[Mapping[str, Any]], labels: Mapping[str, float],
    method: str, universe: str,
) -> list[dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    by_joint = {row["joint_fingerprint"]: row for row in candidates}
    records = []
    if len(plan.get("rounds", [])) != REPLICATES * ROUNDS:
        raise InputInvalid(f"selection round count mismatch: {plan_path}")
    for round_row in plan["rounds"]:
        replicate = int(round_row["replicate_id"])
        round_id = int(round_row["round_id"])
        joints = list(map(str, round_row["selected_joint_fingerprints"]))
        if len(joints) != SELECTED_PER_ROUND or len(set(joints)) != SELECTED_PER_ROUND:
            raise InputInvalid(f"selection cardinality mismatch: {plan_path}")
        for selected_position, joint in enumerate(joints):
            candidate = by_joint.get(joint)
            if candidate is None or (candidate["replicate_id"], candidate["round_id"]) != (replicate, round_id):
                raise InputInvalid(f"selection references foreign candidate: {joint}")
            if joint not in labels:
                raise InputInvalid(f"selected candidate lacks historical validation: {joint}")
            records.append({
                "universe": universe, "method": method, "replicate_id": replicate,
                "round_id": round_id, "selected_position": selected_position,
                "joint_fingerprint": joint,
                "architecture_fingerprint": candidate["architecture_fingerprint"],
                "validation_mean": float(labels[joint]),
            })
    if len(records) != 128:
        raise InputInvalid(f"selected slot count mismatch: {plan_path}")
    return records


def _verify_prediction_file(
    path: Path, candidates: Sequence[Mapping[str, Any]], score_field: str,
) -> None:
    observed = {row["joint_fingerprint"]: float(row[score_field]) for row in read_csv(path)}
    expected = {row["joint_fingerprint"]: float(row[score_field]) for row in candidates}
    if observed != expected:
        raise InputInvalid(f"prediction file differs from frozen candidate manifest: {path}")


def load_frozen_inputs(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    b4_integrity = validate_input_root(B4_ROOT, config["inputs"]["b4"])
    metric_integrity = validate_input_root(METRIC_ROOT, config["inputs"]["metric"])
    b4_candidates = _candidate_rows(B4_ROOT / "pre_search/candidate_manifest.csv", ("b4_prediction",))
    metric_candidates = _candidate_rows(
        METRIC_ROOT / "pre_search/candidate_manifest.csv", ("m1_prediction", "m2_prediction"),
    )
    _verify_prediction_file(B4_ROOT / "pre_search/b4_predictions.csv", b4_candidates, "b4_prediction")
    _verify_prediction_file(METRIC_ROOT / "pre_search/m1_predictions.csv", metric_candidates, "m1_prediction")
    _verify_prediction_file(METRIC_ROOT / "pre_search/m2_predictions.csv", metric_candidates, "m2_prediction")
    b4_labels = _labels(B4_ROOT / "evaluation/config_3seed_statistics.csv")
    metric_labels = _labels(METRIC_ROOT / "evaluation/config_3seed_statistics.csv")
    methods = {
        "B4": {
            "universe": "B4", "score_field": "b4_prediction", "candidates": b4_candidates,
            "records": _selection_records(B4_ROOT / "pre_search/b4_selections.json", b4_candidates, b4_labels, "B4", "B4"),
            "labels": b4_labels, "root": B4_ROOT,
        },
        "B4_RANDOM": {
            "universe": "B4", "score_field": None, "candidates": b4_candidates,
            "records": _selection_records(B4_ROOT / "pre_search/random_selections.json", b4_candidates, b4_labels, "B4_RANDOM", "B4"),
            "labels": b4_labels, "root": B4_ROOT,
        },
        "M1": {
            "universe": "M12", "score_field": "m1_prediction", "candidates": metric_candidates,
            "records": _selection_records(METRIC_ROOT / "pre_search/m1_selections.json", metric_candidates, metric_labels, "M1", "M12"),
            "labels": metric_labels, "root": METRIC_ROOT,
        },
        "M2": {
            "universe": "M12", "score_field": "m2_prediction", "candidates": metric_candidates,
            "records": _selection_records(METRIC_ROOT / "pre_search/m2_selections.json", metric_candidates, metric_labels, "M2", "M12"),
            "labels": metric_labels, "root": METRIC_ROOT,
        },
        "M12_RANDOM": {
            "universe": "M12", "score_field": None, "candidates": metric_candidates,
            "records": _selection_records(METRIC_ROOT / "pre_search/random_selections.json", metric_candidates, metric_labels, "M12_RANDOM", "M12"),
            "labels": metric_labels, "root": METRIC_ROOT,
        },
    }
    integrity = {
        "status": "VALID", "b4": b4_integrity, "metric": metric_integrity,
        "candidate_universes_kept_separate": True,
        "B4_candidate_count": len(b4_candidates), "M12_candidate_count": len(metric_candidates),
        "historical_scientific_results_recomputed": False,
        "new_GNN_runs": 0, "GPU_used": False, "test_accessed": False,
    }
    return methods, integrity


def concentration(architectures: Sequence[str]) -> dict[str, Any]:
    total = len(architectures)
    counts = Counter(map(str, architectures))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    shares = [count / total for _architecture, count in ordered] if total else []
    entropy = -sum(share * math.log(share) for share in shares if share > 0)
    effective = math.exp(entropy) if total else 0.0
    return {
        "selected_slot_count": total,
        "unique_architecture_count": len(counts),
        "architecture_frequency_distribution": dict(ordered),
        "top1_share": sum(shares[:1]), "top2_share": sum(shares[:2]),
        "top4_share": sum(shares[:4]), "shannon_entropy": entropy,
        "effective_architectures": effective,
        "HHI": sum(share * share for share in shares),
        "normalized_effective_architecture_ratio": effective / total if total else 0.0,
    }


def selection_diversity(methods: Mapping[str, Any]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]
]:
    overall_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    overall_by_method: dict[str, dict[str, Any]] = {}
    per_rep_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for method in ALL_METHODS:
        records = methods[method]["records"]
        overall = concentration([row["architecture_fingerprint"] for row in records])
        overall_by_method[method] = overall
        overall_rows.append({"method": method, "universe": methods[method]["universe"], **overall})
        for replicate in range(REPLICATES):
            subset = [row for row in records if row["replicate_id"] == replicate]
            stats = concentration([row["architecture_fingerprint"] for row in subset])
            row = {"method": method, "universe": methods[method]["universe"], "replicate_id": replicate, **stats}
            replicate_rows.append(row)
            per_rep_lookup[(method, replicate)] = row
        for replicate in range(REPLICATES):
            for round_id in range(ROUNDS):
                subset = [
                    row for row in records
                    if row["replicate_id"] == replicate and row["round_id"] == round_id
                ]
                stats = concentration([row["architecture_fingerprint"] for row in subset])
                round_rows.append({
                    "method": method, "universe": methods[method]["universe"],
                    "replicate_id": replicate, "round_id": round_id,
                    "within_round_duplicate_slots": SELECTED_PER_ROUND - stats["unique_architecture_count"],
                    **stats,
                })
    for method in METHODS:
        random_method = PAIR_RANDOM[method]
        learned = overall_by_method[method]
        random = overall_by_method[random_method]
        target = next(row for row in overall_rows if row["method"] == method)
        target.update({
            "paired_random": random_method,
            "effective_ratio_to_random": learned["effective_architectures"] / random["effective_architectures"],
            "unique_ratio_to_random": learned["unique_architecture_count"] / random["unique_architecture_count"],
            "top2_excess_over_random": learned["top2_share"] - random["top2_share"],
            "HHI_ratio_to_random": learned["HHI"] / random["HHI"],
        })
        for replicate in range(REPLICATES):
            learned_rep = per_rep_lookup[(method, replicate)]
            random_rep = per_rep_lookup[(random_method, replicate)]
            learned_rep.update({
                "paired_random": random_method,
                "effective_ratio_to_random": learned_rep["effective_architectures"] / random_rep["effective_architectures"],
                "unique_ratio_to_random": learned_rep["unique_architecture_count"] / random_rep["unique_architecture_count"],
                "top2_excess_over_random": learned_rep["top2_share"] - random_rep["top2_share"],
                "HHI_ratio_to_random": learned_rep["HHI"] / random_rep["HHI"],
            })
    return overall_rows, replicate_rows, round_rows, overall_by_method


def candidate_pool_diversity(methods: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    universe_candidates = {"B4": methods["B4"]["candidates"], "M12": methods["M1"]["candidates"]}
    summary = {}
    for universe, candidates in universe_candidates.items():
        universe_rows = []
        for replicate in range(REPLICATES):
            for round_id in range(ROUNDS):
                pool = [
                    row for row in candidates
                    if row["replicate_id"] == replicate and row["round_id"] == round_id
                ]
                stats = concentration([row["architecture_fingerprint"] for row in pool])
                record = {
                    "universe": universe, "replicate_id": replicate, "round_id": round_id,
                    "candidate_count": len(pool),
                    "candidate_unique_architectures": stats["unique_architecture_count"],
                    "candidate_architecture_entropy": stats["shannon_entropy"],
                    "candidate_effective_architectures": stats["effective_architectures"],
                    "candidate_top1_share": stats["top1_share"],
                    "candidate_top2_share": stats["top2_share"],
                }
                rows.append(record)
                universe_rows.append(record)
        values = [row["candidate_unique_architectures"] for row in universe_rows]
        summary[universe] = {
            "pool_count": len(values), "minimum": min(values),
            "median": float(np.median(values)), "mean": float(np.mean(values)),
            "maximum": max(values), "CANDIDATE_POOL_RICH": float(np.median(values)) >= 64,
        }
    summary["POOL_RICH"] = all(summary[name]["CANDIDATE_POOL_RICH"] for name in ("B4", "M12"))
    return rows, summary


def ranked_candidates(method: str, methods: Mapping[str, Any]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    info = methods[method]
    score_field = info["score_field"]
    if score_field is None:
        raise DiagnosisError(f"random method has no scorer ranking: {method}")
    pools: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for replicate in range(REPLICATES):
        for round_id in range(ROUNDS):
            rows = [
                dict(row) for row in info["candidates"]
                if row["replicate_id"] == replicate and row["round_id"] == round_id
            ]
            rows.sort(key=lambda row: (-float(row[score_field]), str(row["joint_fingerprint"])))
            for index, row in enumerate(rows, 1):
                row["original_rank"] = index
                row["score"] = float(row[score_field])
            pools[(replicate, round_id)] = rows
    return pools


def ranking_concentration(methods: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for (replicate, round_id), ranking in ranked_candidates(method, methods).items():
            for top_k in RANKING_K:
                stats = concentration([row["architecture_fingerprint"] for row in ranking[:top_k]])
                rows.append({
                    "method": method, "universe": methods[method]["universe"],
                    "replicate_id": replicate, "round_id": round_id, "top_k": top_k,
                    "unique_architecture_count": stats["unique_architecture_count"],
                    "effective_architectures": stats["effective_architectures"],
                    "top1_share": stats["top1_share"], "top2_share": stats["top2_share"],
                })
    return rows


def repeat_diagnosis(method: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slot_rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    replicate_repeat = []
    for replicate in range(REPLICATES):
        selected = [dict(row) for row in records if row["replicate_id"] == replicate]
        prior_architectures: set[str] = set()
        for round_id in range(ROUNDS):
            current = [row for row in selected if row["round_id"] == round_id]
            round_counts = Counter(row["architecture_fingerprint"] for row in current)
            for row in current:
                architecture = row["architecture_fingerprint"]
                later_repeat = architecture in prior_architectures
                slot_rows.append({
                    **row,
                    "temporal_slot_type": "LATER_REPEAT" if later_repeat else "FIRST_APPEARANCE",
                    "repeat_architecture_slot": later_repeat,
                    "new_architecture_slot": not later_repeat,
                    "within_round_duplicate": round_counts[architecture] > 1,
                })
            prior_architectures.update(round_counts)
        rep_slots = [row for row in slot_rows if row["method"] == method and row["replicate_id"] == replicate]
        repeat_count = sum(bool(row["repeat_architecture_slot"]) for row in rep_slots)
        replicate_repeat.append({
            "method": method, "replicate_id": replicate,
            "first_time_architecture_slots": 16 - repeat_count,
            "repeat_architecture_slots": repeat_count,
            "repeat_fraction": repeat_count / 16,
        })
        by_architecture: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_architecture[row["architecture_fingerprint"]].append(row)
        for architecture, values in by_architecture.items():
            first_round = min(row["round_id"] for row in values)
            first_values = [row for row in values if row["round_id"] == first_round]
            later_values = [row for row in values if row["round_id"] > first_round]
            if later_values:
                first_best = max(float(row["validation_mean"]) for row in first_values)
                later_best = max(float(row["validation_mean"]) for row in later_values)
                refinement_rows.append({
                    "method": method, "replicate_id": replicate,
                    "architecture_fingerprint": architecture, "first_round": first_round,
                    "first_appearance_config_count": len(first_values),
                    "later_repeat_config_count": len(later_values),
                    "FIRST_BEST": first_best, "LATER_REPEAT_BEST": later_best,
                    "REFINEMENT_GAIN": max(0.0, later_best - first_best),
                })
        first_slots = [row for row in rep_slots if row["temporal_slot_type"] == "FIRST_APPEARANCE"]
        repeat_slots = [row for row in rep_slots if row["temporal_slot_type"] == "LATER_REPEAT"]
        winner = sorted(
            rep_slots, key=lambda row: (-float(row["validation_mean"]), str(row["joint_fingerprint"])),
        )[0]
        best_first = max(float(row["validation_mean"]) for row in first_slots)
        best_repeat = max((float(row["validation_mean"]) for row in repeat_slots), default=None)
        repeat_winner = winner["temporal_slot_type"] == "LATER_REPEAT"
        winner_rows.append({
            "method": method, "replicate_id": replicate,
            "BEST_FIRST_APPEARANCE": best_first, "BEST_LATER_REPEAT": best_repeat,
            "FINAL_BEST": float(winner["validation_mean"]),
            "final_winner_origin": winner["temporal_slot_type"],
            "REPEAT_WINNER_GAIN": float(winner["validation_mean"]) - best_first if repeat_winner else 0.0,
            "winner_joint_fingerprint": winner["joint_fingerprint"],
        })
    repeat_fractions = [row["repeat_fraction"] for row in replicate_repeat]
    refinement_gains = [row["REFINEMENT_GAIN"] for row in refinement_rows]
    repeat_winner_gains = [row["REPEAT_WINNER_GAIN"] for row in winner_rows]
    new_values = [float(row["validation_mean"]) for row in slot_rows if row["temporal_slot_type"] == "FIRST_APPEARANCE"]
    repeat_values = [float(row["validation_mean"]) for row in slot_rows if row["temporal_slot_type"] == "LATER_REPEAT"]
    def distribution(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {key: None for key in ("count", "mean", "median", "Q25", "Q75", "best")}
        return {
            "count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "Q25": float(np.quantile(values, 0.25)), "Q75": float(np.quantile(values, 0.75)),
            "best": max(values),
        }
    summary = {
        "method": method, "median_repeat_fraction": float(np.median(repeat_fractions)),
        "mean_repeat_fraction": float(np.mean(repeat_fractions)),
        "architectures_with_later_repeat": len(refinement_rows),
        "mean_refinement_gain": float(np.mean(refinement_gains)) if refinement_gains else None,
        "median_refinement_gain": float(np.median(refinement_gains)) if refinement_gains else None,
        "fraction_refinement_gain_positive": float(np.mean(np.asarray(refinement_gains) > 0)) if refinement_gains else None,
        "fraction_refinement_gain_at_least_0.003": float(np.mean(np.asarray(refinement_gains) >= 0.003)) if refinement_gains else None,
        "fraction_refinement_gain_at_least_0.006": float(np.mean(np.asarray(refinement_gains) >= 0.006)) if refinement_gains else None,
        "repeat_winner_replicates": sum(row["final_winner_origin"] == "LATER_REPEAT" for row in winner_rows),
        "mean_REPEAT_WINNER_GAIN": float(np.mean(repeat_winner_gains)),
        "median_REPEAT_WINNER_GAIN": float(np.median(repeat_winner_gains)),
        "repeat_winner_gain_at_least_0.003_replicates": sum(value >= 0.003 for value in repeat_winner_gains),
        "repeat_winner_gain_at_least_0.006_replicates": sum(value >= 0.006 for value in repeat_winner_gains),
        "new_architecture_slot_validation": distribution(new_values),
        "repeat_architecture_slot_validation": distribution(repeat_values),
        "selection_bias_warning": "descriptive only; scorer selection bias prevents causal interpretation",
    }
    return {
        "slot_rows": slot_rows, "replicate_repeat": replicate_repeat,
        "refinement_rows": refinement_rows, "winner_rows": winner_rows, "summary": summary,
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    from scipy.stats import rankdata
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    xr = np.asarray(rankdata(x), dtype=float)
    yr = np.asarray(rankdata(y), dtype=float)
    xr -= xr.mean()
    yr -= yr.mean()
    denominator = float(np.sqrt(np.sum(xr * xr) * np.sum(yr * yr)))
    return float(np.sum(xr * yr) / denominator) if denominator else None


def exact_permutation_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    from scipy.stats import rankdata
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"spearman_rho": None, "exact_two_sided_permutation_p": None, "permutations": 0}
    xr = np.asarray(rankdata(x), dtype=float)
    yr = np.asarray(rankdata(y), dtype=float)
    xr -= xr.mean(); yr -= yr.mean()
    denominator = float(np.sqrt(np.sum(xr * xr) * np.sum(yr * yr)))
    observed = float(np.sum(xr * yr) / denominator)
    extreme = total = 0
    for permutation in itertools.permutations(range(len(y))):
        rho = float(np.sum(xr * yr[list(permutation)]) / denominator)
        total += 1
        extreme += abs(rho) + 1e-15 >= abs(observed)
    return {
        "spearman_rho": observed,
        "exact_two_sided_permutation_p": extreme / total,
        "permutations": total,
    }


def anytime_saturation(methods: Mapping[str, Any]) -> dict[str, Any]:
    source_method = {
        "B4": (B4_ROOT, "B4_GUIDED", "MODEL_BLIND_RANDOM_HIGH_REGIME", "B4_SEARCH_NOT_CONFIRMED"),
        "M1": (METRIC_ROOT, "M1", "RANDOM", "METRIC_SEARCH_NOT_CONFIRMED"),
        "M2": (METRIC_ROOT, "M2", "RANDOM", "METRIC_SEARCH_NOT_CONFIRMED"),
    }
    output: dict[str, Any] = {"methods": {}}
    early_count = 0
    for method, (root, source_learned, source_random, classification) in source_method.items():
        rows = read_csv(root / "search/anytime_curves.csv")
        budget_gains = {}
        per_rep_auc = {name: defaultdict(list) for name in (source_learned, source_random)}
        for row in rows:
            if row["method"] in per_rep_auc:
                per_rep_auc[row["method"]][int(row["replicate_id"])].append(float(row["best_so_far"]))
        for budget in (4, 8, 12, 16):
            learned_values = [
                float(row["best_so_far"]) for row in rows
                if row["method"] == source_learned and int(row["budget"]) == budget
            ]
            random_values = [
                float(row["best_so_far"]) for row in rows
                if row["method"] == source_random and int(row["budget"]) == budget
            ]
            if len(learned_values) != REPLICATES or len(random_values) != REPLICATES:
                raise InputInvalid(f"anytime grid mismatch for {method} at {budget}")
            budget_gains[str(budget)] = float(np.mean(learned_values) - np.mean(random_values))
        learned_auc = float(np.mean([np.mean(per_rep_auc[source_learned][rep]) for rep in range(REPLICATES)]))
        random_auc = float(np.mean([np.mean(per_rep_auc[source_random][rep]) for rep in range(REPLICATES)]))
        gain_values = list(budget_gains.values())
        early_signal = learned_auc > random_auc and classification in {
            "B4_SEARCH_NOT_CONFIRMED", "METRIC_SEARCH_NOT_CONFIRMED",
        }
        early_count += early_signal
        output["methods"][method] = {
            "paired_random": PAIR_RANDOM[method], "budget_mean_best_so_far_gain": budget_gains,
            "GAIN_4": budget_gains["4"], "GAIN_8": budget_gains["8"],
            "GAIN_12": budget_gains["12"], "GAIN_16": budget_gains["16"],
            "early_to_final_decay": budget_gains["4"] - budget_gains["16"],
            "max_to_final_decay": max(gain_values) - budget_gains["16"],
            "method_mean_anytime_AUC": learned_auc, "random_mean_anytime_AUC": random_auc,
            "ANYTIME_AUC_GAIN": learned_auc - random_auc,
            "FINAL_BEST_GAIN": budget_gains["16"],
            "previous_frozen_classification": classification,
            "original_final_search_gate_promoted": False,
            "EARLY_SEARCH_SIGNAL": early_signal,
        }
    output["early_search_signal_method_count"] = early_count
    output["SATURATION_PATTERN"] = early_count >= 2
    return output


def collapse_gates(
    overall: Mapping[str, Mapping[str, Any]], per_replicate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lookup = {(row["method"], int(row["replicate_id"])): row for row in per_replicate}
    methods = {}
    for method in METHODS:
        random = PAIR_RANDOM[method]
        effective_ratio = overall[method]["effective_architectures"] / overall[random]["effective_architectures"]
        learned_median = float(np.median([lookup[(method, rep)]["unique_architecture_count"] for rep in range(REPLICATES)]))
        random_median = float(np.median([lookup[(random, rep)]["unique_architecture_count"] for rep in range(REPLICATES)]))
        indicators = {
            "C1": {"passed": effective_ratio <= 0.35, "effective_ratio_to_random": effective_ratio, "maximum": 0.35},
            "C2": {"passed": overall[method]["top2_share"] >= 0.50, "top2_share": overall[method]["top2_share"], "minimum": 0.50},
            "C3": {
                "passed": learned_median <= 0.60 * random_median,
                "method_median_unique_architectures": learned_median,
                "random_median_unique_architectures": random_median, "maximum_ratio": 0.60,
            },
        }
        methods[method] = {
            "paired_random": random, "indicators": indicators,
            "passed_indicator_count": sum(value["passed"] for value in indicators.values()),
            "METHOD_COLLAPSE_STRONG": sum(value["passed"] for value in indicators.values()) >= 2,
        }
    return {
        "methods": methods,
        "SYSTEMATIC_COLLAPSE": all(methods[method]["METHOD_COLLAPSE_STRONG"] for method in METHODS),
        "thresholds_frozen": True,
    }


def diversity_performance_association(
    per_replicate: Sequence[Mapping[str, Any]], repeat_results: Mapping[str, Any],
) -> dict[str, Any]:
    lookup = {(row["method"], int(row["replicate_id"])): row for row in per_replicate}
    output = {"unit": "search_replicate", "n": REPLICATES, "small_n_diagnostic_only": True, "methods": {}}
    for method in METHODS:
        random = PAIR_RANDOM[method]
        final_lookup = {
            int(row["replicate_id"]): float(row["FINAL_BEST"])
            for row in repeat_results[method]["winner_rows"]
        }
        random_final_lookup = {
            int(row["replicate_id"]): float(row["FINAL_BEST"])
            for row in repeat_results[random]["winner_rows"]
        }
        unique = [float(lookup[(method, rep)]["unique_architecture_count"]) for rep in range(REPLICATES)]
        effective = [float(lookup[(method, rep)]["effective_architectures"]) for rep in range(REPLICATES)]
        final = [final_lookup[rep] for rep in range(REPLICATES)]
        ratio = [
            float(lookup[(method, rep)]["effective_architectures"]) /
            float(lookup[(random, rep)]["effective_architectures"])
            for rep in range(REPLICATES)
        ]
        gain = [final_lookup[rep] - random_final_lookup[rep] for rep in range(REPLICATES)]
        output["methods"][method] = {
            "unique_architecture_count_vs_final_best": exact_permutation_spearman(unique, final),
            "effective_architectures_vs_final_best": exact_permutation_spearman(effective, final),
            "effective_ratio_to_random_vs_final_best_gain": exact_permutation_spearman(ratio, gain),
        }
    return output


def build_counterfactual_policy(
    method: str, methods: Mapping[str, Any], policy: str,
) -> list[dict[str, Any]]:
    if policy not in {"P1", "P2"}:
        raise DiagnosisError(f"unknown policy: {policy}")
    rankings = ranked_candidates(method, methods)
    selected: list[dict[str, Any]] = []
    for replicate in range(REPLICATES):
        architecture_slots: Counter[str] = Counter()
        for round_id in range(ROUNDS):
            round_architectures: set[str] = set()
            chosen = []
            ranking = rankings[(replicate, round_id)]
            scores = np.asarray([row["score"] for row in ranking], dtype=float)
            mean, std = float(np.mean(scores)), float(np.std(scores))
            degenerate = std < TOLERANCE
            for candidate in ranking:
                architecture = str(candidate["architecture_fingerprint"])
                if architecture in round_architectures:
                    continue
                if policy == "P2" and architecture_slots[architecture] >= 2:
                    continue
                chosen.append(candidate)
                round_architectures.add(architecture)
                if len(chosen) == SELECTED_PER_ROUND:
                    break
            for position, candidate in enumerate(chosen):
                architecture = str(candidate["architecture_fingerprint"])
                architecture_slots[architecture] += 1
                selected.append({
                    "universe": methods[method]["universe"], "method": method, "policy": policy,
                    "replicate_id": replicate, "round_id": round_id,
                    "selected_position": position,
                    "joint_fingerprint": candidate["joint_fingerprint"],
                    "architecture_fingerprint": architecture,
                    "original_rank": int(candidate["original_rank"]),
                    "frozen_score": float(candidate["score"]),
                    "pool_score_z": 0.0 if degenerate else (float(candidate["score"]) - mean) / std,
                    "degenerate_score_pool": degenerate,
                    "validation_used_for_selection": False,
                })
    return selected


def structural_policy_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = concentration([row["architecture_fingerprint"] for row in records])
    unique_per_replicate = []
    repeat_count = duplicate_count = 0
    complete_rounds = 0
    for replicate in range(REPLICATES):
        rep = [row for row in records if row["replicate_id"] == replicate]
        unique_per_replicate.append(len({row["architecture_fingerprint"] for row in rep}))
        prior: set[str] = set()
        for round_id in range(ROUNDS):
            current = [row for row in rep if row["round_id"] == round_id]
            if len(current) == SELECTED_PER_ROUND:
                complete_rounds += 1
            counts = Counter(row["architecture_fingerprint"] for row in current)
            duplicate_count += sum(count - 1 for count in counts.values())
            repeat_count += sum(row["architecture_fingerprint"] in prior for row in current)
            prior.update(counts)
    return {
        "selected_slots": len(records),
        "unique_architecture_count": overall["unique_architecture_count"],
        "median_unique_architectures_per_replicate": float(np.median(unique_per_replicate)),
        "effective_architectures": overall["effective_architectures"],
        "top1_share": overall["top1_share"], "top2_share": overall["top2_share"],
        "repeat_fraction": repeat_count / len(records) if records else None,
        "within_round_duplicate_fraction": duplicate_count / len(records) if records else None,
        "complete_rounds": complete_rounds,
    }


def _rank_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranks = [int(row["original_rank"]) for row in records]
    return {
        "selected_slots": len(ranks), "mean_rank": float(np.mean(ranks)) if ranks else None,
        "median_rank": float(np.median(ranks)) if ranks else None,
        "P90_rank": float(np.quantile(ranks, 0.90)) if ranks else None,
        "max_rank": max(ranks) if ranks else None,
        "fraction_rank_at_most_8": float(np.mean(np.asarray(ranks) <= 8)) if ranks else None,
        "fraction_rank_at_most_16": float(np.mean(np.asarray(ranks) <= 16)) if ranks else None,
        "fraction_rank_at_most_32": float(np.mean(np.asarray(ranks) <= 32)) if ranks else None,
    }


def _original_rank_records(method: str, methods: Mapping[str, Any]) -> list[dict[str, Any]]:
    rank_lookup = {
        row["joint_fingerprint"]: row
        for ranking in ranked_candidates(method, methods).values() for row in ranking
    }
    output = []
    for selected in methods[method]["records"]:
        ranked = rank_lookup[selected["joint_fingerprint"]]
        output.append({
            **selected, "policy": "ORIGINAL", "original_rank": ranked["original_rank"],
            "frozen_score": ranked["score"],
        })
    return output


def _score_sacrifice(
    method: str, methods: Mapping[str, Any], policy_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rankings = ranked_candidates(method, methods)
    pool_rows = []
    for replicate in range(REPLICATES):
        for round_id in range(ROUNDS):
            ranking = rankings[(replicate, round_id)]
            scores = np.asarray([row["score"] for row in ranking], dtype=float)
            mean, std = float(np.mean(scores)), float(np.std(scores))
            degenerate = std < TOLERANCE
            def mean_z(records: Sequence[Mapping[str, Any]]) -> float:
                selected = [
                    row for row in records
                    if row["replicate_id"] == replicate and row["round_id"] == round_id
                ]
                if len(selected) != SELECTED_PER_ROUND:
                    return float("nan")
                if degenerate:
                    return 0.0
                return float(np.mean([(float(row["frozen_score"]) - mean) / std for row in selected]))
            original = mean_z(policy_records["ORIGINAL"])
            p1, p2 = mean_z(policy_records["P1"]), mean_z(policy_records["P2"])
            pool_rows.append({
                "method": method, "replicate_id": replicate, "round_id": round_id,
                "degenerate_score_pool": degenerate, "original_top4_mean_z": original,
                "P1_mean_z": p1, "P2_mean_z": p2,
                "z_sacrifice_P1": 0.0 if degenerate else original - p1,
                "z_sacrifice_P2": 0.0 if degenerate else original - p2,
            })
    summary = {}
    for policy in ("P1", "P2"):
        values = [float(row[f"z_sacrifice_{policy}"]) for row in pool_rows]
        summary[policy] = {
            "mean_z_score_sacrifice": float(np.mean(values)),
            "median_z_score_sacrifice": float(np.median(values)),
            "P90_z_score_sacrifice": float(np.quantile(values, 0.90)),
            "degenerate_score_pool_count": sum(bool(row["degenerate_score_pool"]) for row in pool_rows),
        }
    return {"pool_rows": pool_rows, "summary": summary}


def counterfactual_diagnosis(methods: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "selections": {"P1": [], "P2": []}, "structural": {},
        "rank_sacrifice": {}, "score_sacrifice": {}, "coverage": {},
        "observed_subset": [], "preferred_policy": {},
    }
    for method in METHODS:
        original = _original_rank_records(method, methods)
        p1 = build_counterfactual_policy(method, methods, "P1")
        p2 = build_counterfactual_policy(method, methods, "P2")
        output["selections"]["P1"].extend(p1)
        output["selections"]["P2"].extend(p2)
        records = {"ORIGINAL": original, "P1": p1, "P2": p2}
        original_sets = {
            (replicate, round_id): {
                row["joint_fingerprint"] for row in original
                if row["replicate_id"] == replicate and row["round_id"] == round_id
            }
            for replicate in range(REPLICATES) for round_id in range(ROUNDS)
        }
        output["structural"][method] = {}
        for policy, policy_rows in records.items():
            metrics = structural_policy_metrics(policy_rows)
            metrics["CHURN"] = sum(
                SELECTED_PER_ROUND - len(original_sets[(replicate, round_id)] & {
                    row["joint_fingerprint"] for row in policy_rows
                    if row["replicate_id"] == replicate and row["round_id"] == round_id
                })
                for replicate in range(REPLICATES) for round_id in range(ROUNDS)
            )
            output["structural"][method][policy] = metrics
        output["rank_sacrifice"][method] = {
            "ORIGINAL": {**_rank_summary(original), "mean_rank_is_fixed_top4": 2.5},
            "P1": _rank_summary(p1), "P2": _rank_summary(p2),
        }
        score = _score_sacrifice(method, methods, records)
        output["score_sacrifice"][method] = score
        output["coverage"][method] = {}
        evaluated = methods[method]["labels"]
        for policy, policy_rows in (("P1", p1), ("P2", p2)):
            covered = [row for row in policy_rows if row["joint_fingerprint"] in evaluated]
            fully_rounds = sum(
                sum(
                    row["joint_fingerprint"] in evaluated for row in policy_rows
                    if row["replicate_id"] == replicate and row["round_id"] == round_id
                ) == SELECTED_PER_ROUND
                for replicate in range(REPLICATES) for round_id in range(ROUNDS)
            )
            fully_replicates = sum(
                sum(
                    row["joint_fingerprint"] in evaluated for row in policy_rows
                    if row["replicate_id"] == replicate
                ) == ROUNDS * SELECTED_PER_ROUND
                for replicate in range(REPLICATES)
            )
            output["coverage"][method][policy] = {
                "selected_slots_covered": len(covered), "selected_slots_total": 128,
                "coverage_fraction": len(covered) / 128,
                "fully_covered_rounds": fully_rounds, "fully_covered_replicates": fully_replicates,
                "unobserved_labels_imputed": 0, "policy_choice_influenced_by_coverage": False,
            }
            output["observed_subset"].extend({
                "method": method, "policy": policy,
                "replicate_id": row["replicate_id"], "round_id": row["round_id"],
                "joint_fingerprint": row["joint_fingerprint"],
                "architecture_fingerprint": row["architecture_fingerprint"],
                "validation_mean": float(evaluated[row["joint_fingerprint"]]),
                "evidence_role": "OBSERVED_SUBSET_ONLY",
            } for row in covered)
        output["preferred_policy"][method] = {}
        for policy in ("P1", "P2"):
            structural = output["structural"][method][policy]
            original_structural = output["structural"][method]["ORIGINAL"]
            ranks = output["rank_sacrifice"][method][policy]
            sacrifice = output["score_sacrifice"][method]["summary"][policy]
            gates = {
                "F1": structural["complete_rounds"] == 32,
                "F2": structural["median_unique_architectures_per_replicate"] + TOLERANCE >= 1.50 * original_structural["median_unique_architectures_per_replicate"],
                "F3": ranks["median_rank"] <= 16 + TOLERANCE,
                "F4": ranks["fraction_rank_at_most_32"] + TOLERANCE >= 0.90,
                "F5": sacrifice["mean_z_score_sacrifice"] <= 0.50 + TOLERANCE,
            }
            output["preferred_policy"][method][policy] = {
                "gates": gates, "feasible": all(gates.values()),
            }
        output["preferred_policy"][method]["PREFERRED_POLICY"] = (
            "P2" if output["preferred_policy"][method]["P2"]["feasible"]
            else "P1" if output["preferred_policy"][method]["P1"]["feasible"]
            else "NONE"
        )
    return output


def select_next_scorer(anytime: Mapping[str, Any], counterfactual: Mapping[str, Any]) -> str | None:
    eligible = [
        method for method in METHODS
        if anytime["methods"][method]["ANYTIME_AUC_GAIN"] > 0
        and counterfactual["preferred_policy"][method]["PREFERRED_POLICY"] != "NONE"
    ]
    if not eligible:
        return None
    max_auc = max(anytime["methods"][method]["ANYTIME_AUC_GAIN"] for method in eligible)
    eligible = [
        method for method in eligible
        if max_auc - anytime["methods"][method]["ANYTIME_AUC_GAIN"] <= TOLERANCE
    ]
    max_final = max(anytime["methods"][method]["FINAL_BEST_GAIN"] for method in eligible)
    eligible = [
        method for method in eligible
        if max_final - anytime["methods"][method]["FINAL_BEST_GAIN"] <= TOLERANCE
    ]
    return sorted(eligible)[0]


def authorization_decision(
    pool_summary: Mapping[str, Any], collapse: Mapping[str, Any],
    anytime: Mapping[str, Any], counterfactual: Mapping[str, Any],
) -> dict[str, Any]:
    scorer = select_next_scorer(anytime, counterfactual)
    gates = {
        "A1_POOL_RICH": bool(pool_summary["POOL_RICH"]),
        "A2_SYSTEMATIC_COLLAPSE": bool(collapse["SYSTEMATIC_COLLAPSE"]),
        "A3_SATURATION_PATTERN": bool(anytime["SATURATION_PATTERN"]),
        "A4_POLICY_AVAILABLE": any(
            counterfactual["preferred_policy"][method]["PREFERRED_POLICY"] != "NONE"
            for method in METHODS
        ),
    }
    if not gates["A2_SYSTEMATIC_COLLAPSE"]:
        classification = "NO_SYSTEMATIC_SEARCH_COLLAPSE"
    elif all(gates.values()):
        classification = "DIVERSITY_TEST_AUTHORIZED"
    else:
        classification = "COLLAPSE_PRESENT_BUT_DIVERSITY_TEST_NOT_JUSTIFIED"
    return {
        "classification": classification, "authorization_gates": gates,
        "POOL_RICH": gates["A1_POOL_RICH"],
        "SYSTEMATIC_COLLAPSE": gates["A2_SYSTEMATIC_COLLAPSE"],
        "SATURATION_PATTERN": gates["A3_SATURATION_PATTERN"],
        "POLICY_AVAILABLE": gates["A4_POLICY_AVAILABLE"],
        "SCORER_FOR_NEXT_DIVERSITY_TEST": scorer,
        "SELECTED_DIVERSITY_POLICY": None if scorer is None else counterfactual["preferred_policy"][scorer]["PREFERRED_POLICY"],
        "historical_classifications_modified": False,
        "causal_claim_made": False,
        "new_GNN_runs": 0, "GPU_used": False, "test_accessed": False,
    }


def build_method_summary(
    methods: Mapping[str, Any], overall_rows: Sequence[Mapping[str, Any]],
    repeat_results: Mapping[str, Any], collapse: Mapping[str, Any],
    anytime: Mapping[str, Any], counterfactual: Mapping[str, Any],
) -> list[dict[str, Any]]:
    overall = {row["method"]: row for row in overall_rows}
    rows = []
    for method in ALL_METHODS:
        row = {
            "method": method, "universe": methods[method]["universe"],
            "paired_random": PAIR_RANDOM.get(method),
            "unique_architectures": overall[method]["unique_architecture_count"],
            "effective_architectures": overall[method]["effective_architectures"],
            "top1_share": overall[method]["top1_share"],
            "top2_share": overall[method]["top2_share"],
            "top4_share": overall[method]["top4_share"],
            "HHI": overall[method]["HHI"],
            "normalized_effective_ratio": overall[method]["normalized_effective_architecture_ratio"],
            "repeat_fraction": repeat_results[method]["summary"]["mean_repeat_fraction"],
            "median_repeat_fraction": repeat_results[method]["summary"]["median_repeat_fraction"],
            "repeat_winner_replicates": repeat_results[method]["summary"]["repeat_winner_replicates"],
        }
        if method in METHODS:
            row.update({
                "effective_ratio_to_random": overall[method]["effective_ratio_to_random"],
                "unique_ratio_to_random": overall[method]["unique_ratio_to_random"],
                "top2_excess_over_random": overall[method]["top2_excess_over_random"],
                "HHI_ratio_to_random": overall[method]["HHI_ratio_to_random"],
                "METHOD_COLLAPSE_STRONG": collapse["methods"][method]["METHOD_COLLAPSE_STRONG"],
                "ANYTIME_AUC_GAIN": anytime["methods"][method]["ANYTIME_AUC_GAIN"],
                "FINAL_BEST_GAIN": anytime["methods"][method]["FINAL_BEST_GAIN"],
                "EARLY_SEARCH_SIGNAL": anytime["methods"][method]["EARLY_SEARCH_SIGNAL"],
                "P1_feasible": counterfactual["preferred_policy"][method]["P1"]["feasible"],
                "P2_feasible": counterfactual["preferred_policy"][method]["P2"]["feasible"],
                "PREFERRED_POLICY": counterfactual["preferred_policy"][method]["PREFERRED_POLICY"],
            })
        rows.append(row)
    return rows


def _summary_stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)), "median": float(np.median(values)),
        "mean": float(np.mean(values)), "maximum": float(np.max(values)),
    }


def create_figures(
    output: Path, method_summary: Sequence[Mapping[str, Any]],
    per_replicate: Sequence[Mapping[str, Any]], ranking_rows: Sequence[Mapping[str, Any]],
    repeat_results: Mapping[str, Any], anytime: Mapping[str, Any], counterfactual: Mapping[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = {row["method"]: row for row in method_summary}
    learned = list(METHODS)
    randoms = [PAIR_RANDOM[method] for method in learned]
    x = np.arange(3)

    def paired_bars(filename: str, field: str, ylabel: str) -> None:
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.bar(x - 0.18, [summary[m][field] for m in learned], 0.36, label="learned scorer")
        axis.bar(x + 0.18, [summary[m][field] for m in randoms], 0.36, label="paired Random")
        axis.set_xticks(x, learned); axis.set_ylabel(ylabel); axis.legend(); axis.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(figures / filename, dpi=160); plt.close(fig)

    paired_bars("architecture_effective_number.png", "effective_architectures", "Effective architectures")
    paired_bars("top2_architecture_share.png", "top2_share", "Top-2 architecture share")

    fig, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(6)
    labels = ["B4", "B4 R", "M1", "M12 R", "M2", "M12 R"]
    method_order = ["B4", "B4_RANDOM", "M1", "M12_RANDOM", "M2", "M12_RANDOM"]
    values = [[row["unique_architecture_count"] for row in per_replicate if row["method"] == method] for method in method_order]
    axis.boxplot(values, positions=positions); axis.set_xticks(positions, labels); axis.set_ylabel("Unique architectures / replicate")
    axis.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(figures / "unique_architectures_per_replicate.png", dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        values = anytime["methods"][method]["budget_mean_best_so_far_gain"]
        axis.plot([4, 8, 12, 16], [values[str(k)] for k in (4, 8, 12, 16)], marker="o", label=method)
    axis.axhline(0, color="black", linewidth=0.8); axis.set_xlabel("Evaluated configs"); axis.set_ylabel("Mean best-so-far gain vs paired Random")
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(figures / "anytime_gain_curves.png", dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        means = [float(np.mean([row["unique_architecture_count"] for row in ranking_rows if row["method"] == method and row["top_k"] == k])) for k in RANKING_K]
        axis.plot(RANKING_K, means, marker="o", label=method)
    axis.set_xlabel("Top K frozen ranking"); axis.set_ylabel("Mean unique architectures"); axis.legend(); axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figures / "ranking_unique_architectures_topk.png", dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    colors = {"B4": "C0", "M1": "C1", "M2": "C2"}
    for method in METHODS:
        repeat = {row["replicate_id"]: row["repeat_fraction"] for row in repeat_results[method]["replicate_repeat"]}
        gain = anytime["methods"][method]
        per_rep_final = {
            int(row["replicate_id"]): float(row["FINAL_BEST"])
            for row in repeat_results[method]["winner_rows"]
        }
        random_final = {
            int(row["replicate_id"]): float(row["FINAL_BEST"])
            for row in repeat_results[PAIR_RANDOM[method]]["winner_rows"]
        }
        axis.scatter([repeat[r] for r in range(8)], [per_rep_final[r] - random_final[r] for r in range(8)], label=method, color=colors[method])
    axis.axhline(0, color="black", linewidth=0.8); axis.set_xlabel("Repeat fraction"); axis.set_ylabel("Final-best gain vs paired Random")
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(figures / "repeat_fraction_vs_final_gain.png", dpi=160); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    markers = {"ORIGINAL": "o", "P1": "s", "P2": "^"}
    for method in METHODS:
        for policy in ("ORIGINAL", "P1", "P2"):
            structural = counterfactual["structural"][method][policy]
            rank = counterfactual["rank_sacrifice"][method][policy]["mean_rank"]
            axis.scatter(rank, structural["effective_architectures"], marker=markers[policy], label=f"{method}-{policy}")
    axis.set_xlabel("Mean original scorer rank"); axis.set_ylabel("Effective architectures"); axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(figures / "counterfactual_diversity_tradeoff.png", dpi=160); plt.close(fig)


def write_next_experiment(output: Path, decision: Mapping[str, Any]) -> None:
    if decision["classification"] != "DIVERSITY_TEST_AUTHORIZED":
        return
    scorer = decision["SCORER_FOR_NEXT_DIVERSITY_TEST"]
    policy = decision["SELECTED_DIVERSITY_POLICY"]
    spec = {
        "status": "SPECIFICATION_ONLY", "scientific_role": "prospective_confirmatory_intervention",
        "selected_scorer": scorer, "selected_diversity_policy": policy,
        "methods": [
            "PURE_FROZEN_SCORER_TOP4", "FROZEN_SCORER_PLUS_ARCH_DIVERSITY",
            "MODEL_BLIND_RANDOM_HIGH_REGIME",
        ],
        "required_controls": {
            "new_candidate_pools": True, "same_candidate_pools_across_methods": True,
            "same_real_evaluation_budget": True, "same_training_seeds": True,
            "candidate_generation_model_blind": True, "test_access": False,
        },
        "causal_question": "Does the frozen architecture-diversity intervention improve final search accuracy relative to pure scorer top-4?",
        "candidate_generation_started": False, "selections_frozen": False,
        "GNN_training_started": False, "new_GNN_runs": 0,
    }
    dump_json(output / "next_experiment/prospective_diversity_test_spec.json", spec)
    dump_text(output / "next_experiment/prospective_diversity_test_summary.md", "\n".join((
        "# Prospective diversity intervention specification", "",
        f"Selected frozen scorer: `{scorer}`.", "",
        f"Selected structural policy: `{policy}`.", "",
        "A new preregistered experiment must compare pure scorer top-4, the same scorer plus the frozen architecture-diversity policy, and model-blind Random on identical new pools, budgets, and seeds.", "",
        "This file authorizes specification work only. No candidate pool, selection, GPU task, GNN training, or test access was started here.", "",
    )))


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(artifacts: Mapping[str, Any]) -> str:
    decision = artifacts["decision"]
    summary = artifacts["method_summary"]
    learned_summary = [row for row in summary if row["method"] in METHODS]
    random_summary = [row for row in summary if "RANDOM" in row["method"]]
    counterfactual = artifacts["counterfactual"]
    repeat_results = artifacts["repeat_results"]
    sections = {
        0: f"Frozen exploratory classification: `{decision['classification']}`. This classification only controls whether a new prospective diversity intervention is worth testing.",
        1: "The architecture-collapse hypothesis was formulated after observing completed search outcomes. Every association and replay here is exploratory/post-hoc and cannot establish that collapse caused final-accuracy failure.",
        2: f"Both input archives/directories, classifications, manifests, pre-search roots, and no-test states validated: `{artifacts['input_integrity']['status']}`. Candidate universes were never pooled.",
        3: "Historical classifications remain B4_SEARCH_NOT_CONFIRMED and METRIC_SEARCH_NOT_CONFIRMED. This diagnosis does not rewrite either decision.",
        4: f"`{json.dumps(artifacts['pool_summary'], sort_keys=True)}`",
        5: _markdown_table([row for row in summary if row["method"] == "B4"], ("method", "unique_architectures", "effective_architectures", "top2_share", "repeat_fraction", "effective_ratio_to_random")),
        6: _markdown_table([row for row in summary if row["method"] == "M1"], ("method", "unique_architectures", "effective_architectures", "top2_share", "repeat_fraction", "effective_ratio_to_random")),
        7: _markdown_table([row for row in summary if row["method"] == "M2"], ("method", "unique_architectures", "effective_architectures", "top2_share", "repeat_fraction", "effective_ratio_to_random")),
        8: _markdown_table(random_summary, ("method", "unique_architectures", "effective_architectures", "top2_share", "repeat_fraction")),
        9: "Frozen score rankings were examined at top K = 4, 8, 16, 32 without recomputation or validation labels; see `ranking_concentration.csv` and `figures/ranking_unique_architectures_topk.png`.",
        10: "Within-round duplicates are reported for every method and all 32 rounds in `per_round_diversity.csv`; simultaneous duplicates are not counted as later repeats.",
        11: _markdown_table(learned_summary, ("method", "repeat_fraction", "median_repeat_fraction")),
        12: _markdown_table([repeat_results[m]["summary"] for m in METHODS], ("method", "architectures_with_later_repeat", "mean_refinement_gain", "median_refinement_gain", "fraction_refinement_gain_at_least_0.003", "fraction_refinement_gain_at_least_0.006")),
        13: _markdown_table([repeat_results[m]["summary"] for m in METHODS], ("method", "repeat_winner_replicates", "mean_REPEAT_WINNER_GAIN", "median_REPEAT_WINNER_GAIN", "repeat_winner_gain_at_least_0.003_replicates")),
        14: f"`{json.dumps(artifacts['associations'], sort_keys=True)}`. With n=8, these are small-n descriptive permutation diagnostics only.",
        15: f"`{json.dumps(artifacts['anytime'], sort_keys=True)}`",
        16: f"`{json.dumps(artifacts['collapse_gate'], sort_keys=True)}`",
        17: _markdown_table([{"method": m, **counterfactual["structural"][m]["P1"]} for m in METHODS], ("method", "unique_architecture_count", "median_unique_architectures_per_replicate", "effective_architectures", "repeat_fraction", "CHURN")),
        18: _markdown_table([{"method": m, **counterfactual["structural"][m]["P2"]} for m in METHODS], ("method", "unique_architecture_count", "median_unique_architectures_per_replicate", "effective_architectures", "repeat_fraction", "CHURN")),
        19: f"`{json.dumps(counterfactual['rank_sacrifice'], sort_keys=True)}`",
        20: f"`{json.dumps({m: counterfactual['score_sacrifice'][m]['summary'] for m in METHODS}, sort_keys=True)}`",
        21: f"`{json.dumps(counterfactual['coverage'], sort_keys=True)}`",
        22: "Covered counterfactual configurations are labeled OBSERVED_SUBSET_ONLY. Missing labels were never imputed, predictions were never treated as outcomes, and coverage did not affect policy choice or classification.",
        23: f"POOL_RICH `{decision['POOL_RICH']}`; SYSTEMATIC_COLLAPSE `{decision['SYSTEMATIC_COLLAPSE']}`; SATURATION_PATTERN `{decision['SATURATION_PATTERN']}`; POLICY_AVAILABLE `{decision['POLICY_AVAILABLE']}`.",
        24: f"Final classification: `{decision['classification']}`.",
        25: f"Prospective diversity test authorized: `{decision['classification'] == 'DIVERSITY_TEST_AUTHORIZED'}`. No next experiment was executed.",
        26: f"Planning heuristic selected scorer: `{decision['SCORER_FOR_NEXT_DIVERSITY_TEST']}`.",
        27: f"Frozen preferred policy for that scorer: `{decision['SELECTED_DIVERSITY_POLICY']}`.",
        28: "The next experiment must use entirely new shared candidate pools and compare pure frozen scorer top-4, the same scorer plus the frozen diversity intervention, and model-blind Random under identical real budgets and training seeds.",
        29: "This diagnosis does NOT prove architecture collapse caused failed final-accuracy confirmation, does not establish scorer superiority, does not infer unobserved validation, and does not access test data. Causal confirmation requires the new preregistered intervention.",
    }
    lines = [f"# Cross-Method Search-Collapse Diagnosis v1 — {decision['classification']}", ""]
    for index, title in enumerate(REPORT_SECTIONS):
        lines.extend((f"## {title}", "", sections[index], ""))
    return "\n".join(lines)


def assert_report_complete(text: str) -> None:
    missing = [section for section in REPORT_SECTIONS if f"## {section}" not in text]
    if missing:
        raise DiagnosisError(f"report sections missing: {missing}")


def write_environment(output: Path, *, before: bool) -> None:
    if before:
        dump_json(output / "environment.json", {
            "created_at": utc_now(), "platform": platform.platform(), "python": sys.version,
            "executable": sys.executable, "cwd": str(Path.cwd()),
            "branch": _git("branch", "--show-current").strip(),
            "HEAD": _git("rev-parse", "HEAD").strip(),
            "GPU_used": False, "CUDA_initialized": False,
            "new_GNN_runs": 0, "test_accessed": False,
        })
        dump_text(output / "git_status_before.txt", _git("status", "--short", "--branch"))
    else:
        dump_text(output / "git_status_after.txt", _git("status", "--short", "--branch"))
        dump_text(output / "git_diff.patch", _git_diff_with_untracked())


def snapshot_sources(output: Path) -> dict[str, Any]:
    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    files = []
    for relative in IMPLEMENTATION_FILES:
        source = REPO_ROOT / relative
        if not source.is_file():
            raise DiagnosisError(f"source snapshot input missing: {source}")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append({"relative_path": relative, "sha256": sha256_file(target)})
    payload = {
        "protocol_version": PROTOCOL_VERSION, "files": files,
        "manifest_root_sha256": stable_hash([(row["relative_path"], row["sha256"]) for row in files]),
        "source_bytes_copied": True,
    }
    dump_json(snapshot / "sha256_manifest.json", payload)
    return payload


def write_manifest(output: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts:
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


def create_archive(output: Path) -> str:
    archive = Path(str(output) + ".tar.gz")
    sidecar = Path(str(archive) + ".sha256")
    if archive.exists() or sidecar.exists():
        raise DiagnosisError(f"archive target exists: {archive}")
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    with tarfile.open(temporary, "w:gz") as handle:
        handle.add(output, arcname=output.name)
    digest = sha256_file(temporary)
    os.replace(temporary, archive)
    dump_text(sidecar, f"{digest}  {archive.name}\n")
    return digest


def ensure_output_absent(output: Path) -> None:
    targets = (output, Path(str(output) + ".tar.gz"), Path(str(output) + ".tar.gz.sha256"))
    if any(path.exists() or path.is_symlink() for path in targets):
        raise DiagnosisError(f"formal output target already exists: {next(path for path in targets if path.exists() or path.is_symlink())}")
    try:
        output.resolve().relative_to((REPO_ROOT / "results/posthoc").resolve())
    except ValueError as error:
        raise DiagnosisError("formal output must remain under results/posthoc") from error


def execute_formal(config_path: Path, output: Path) -> int:
    ensure_output_absent(output)
    config = load_config(config_path)
    validate_protocol(config)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("counterfactual", "figures", "source_snapshot"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    state = {
        "protocol_version": PROTOCOL_VERSION, "stage": "INPUT_VALIDATION",
        "classification": None, "new_GNN_runs": 0, "GPU_used": False,
        "CUDA_initialized": False, "evaluator_subprocess_started": False,
        "next_experiment_started": False, "test_accessed": False,
        "created_at": utc_now(), "updated_at": utc_now(),
    }
    dump_json(output / "state.json", state)
    dump_json(output / "frozen_protocol.json", config)
    write_environment(output, before=True)
    try:
        methods, input_integrity = load_frozen_inputs(config)
    except Exception as error:
        classification = "DIAGNOSIS_INPUT_INVALID" if isinstance(error, InputInvalid) else "RUN_FAILED"
        dump_json(output / "input_integrity.json", {
            "status": "INVALID", "reason": f"{type(error).__name__}: {error}",
            "new_GNN_runs": 0, "GPU_used": False, "test_accessed": False,
        })
        decision = {
            "classification": classification, "reason": str(error),
            "POOL_RICH": False, "SYSTEMATIC_COLLAPSE": False,
            "SATURATION_PATTERN": False, "POLICY_AVAILABLE": False,
            "SCORER_FOR_NEXT_DIVERSITY_TEST": None, "SELECTED_DIVERSITY_POLICY": None,
            "new_GNN_runs": 0, "GPU_used": False, "test_accessed": False,
        }
        dump_json(output / "decision.json", decision)
        state.update({"stage": "FAILED", "classification": classification, "updated_at": utc_now()})
        dump_json(output / "state.json", state)
        dump_text(output / "FAILED", classification + "\n")
        dump_text(output / "final_report.md", f"# Cross-Method Search-Collapse Diagnosis v1 — {classification}\n\nInput validation failed: {error}\n")
        write_environment(output, before=False)
        snapshot_sources(output); write_manifest(output); create_archive(output)
        return 0
    dump_json(output / "input_integrity.json", input_integrity)
    state.update({"stage": "ANALYSIS_RUNNING", "updated_at": utc_now()})
    dump_json(output / "state.json", state)

    overall_rows, per_replicate, per_round, overall = selection_diversity(methods)
    pool_rows, pool_summary = candidate_pool_diversity(methods)
    ranking_rows = ranking_concentration(methods)
    repeat_results = {method: repeat_diagnosis(method, methods[method]["records"]) for method in ALL_METHODS}
    anytime = anytime_saturation(methods)
    collapse = collapse_gates(overall, per_replicate)
    associations = diversity_performance_association(per_replicate, repeat_results)
    counterfactual = counterfactual_diagnosis(methods)
    decision = authorization_decision(pool_summary, collapse, anytime, counterfactual)
    method_summary = build_method_summary(
        methods, overall_rows, repeat_results, collapse, anytime, counterfactual,
    )

    dump_csv(output / "method_summary.csv", method_summary)
    dump_csv(output / "candidate_pool_diversity.csv", pool_rows)
    dump_csv(output / "ranking_concentration.csv", ranking_rows)
    dump_csv(output / "selection_concentration.csv", overall_rows)
    dump_csv(output / "per_replicate_diversity.csv", per_replicate)
    dump_csv(output / "per_round_diversity.csv", per_round)
    dump_csv(output / "repeat_slot_analysis.csv", [row for method in ALL_METHODS for row in repeat_results[method]["slot_rows"]])
    dump_csv(output / "architecture_refinement_analysis.csv", [row for method in ALL_METHODS for row in repeat_results[method]["refinement_rows"]])
    dump_csv(output / "repeat_winner_analysis.csv", [row for method in ALL_METHODS for row in repeat_results[method]["winner_rows"]])
    dump_json(output / "diversity_performance_association.json", associations)
    dump_json(output / "anytime_saturation.json", anytime)
    dump_json(output / "collapse_gate.json", collapse)
    dump_csv(output / "counterfactual/p1_round_unique_selections.csv", counterfactual["selections"]["P1"])
    dump_csv(output / "counterfactual/p2_round_unique_cap2_selections.csv", counterfactual["selections"]["P2"])
    dump_json(output / "counterfactual/policy_structural_metrics.json", {
        "structural": counterfactual["structural"], "feasibility": counterfactual["preferred_policy"],
        "validation_used_for_policy_choice": False,
    })
    dump_json(output / "counterfactual/rank_sacrifice.json", counterfactual["rank_sacrifice"])
    dump_json(output / "counterfactual/score_sacrifice.json", {
        method: counterfactual["score_sacrifice"][method] for method in METHODS
    })
    dump_json(output / "counterfactual/existing_label_coverage.json", counterfactual["coverage"])
    dump_json(output / "counterfactual/observed_subset_results.json", {
        "evidence_role": "OBSERVED_SUBSET_ONLY",
        "unobserved_labels_imputed": 0, "classification_influenced": False,
        "rows": counterfactual["observed_subset"],
    })
    dump_json(output / "decision.json", decision)
    write_next_experiment(output, decision)
    create_figures(output, method_summary, per_replicate, ranking_rows, repeat_results, anytime, counterfactual)
    artifacts = {
        "decision": decision, "input_integrity": input_integrity,
        "method_summary": method_summary, "pool_summary": pool_summary,
        "associations": associations, "anytime": anytime, "collapse_gate": collapse,
        "counterfactual": counterfactual, "repeat_results": repeat_results,
    }
    report = render_report(artifacts)
    assert_report_complete(report)
    dump_text(output / "final_report.md", report)
    state.update({
        "stage": "ANALYSIS_COMPLETE", "classification": decision["classification"],
        "new_GNN_runs": 0, "GPU_used": False, "CUDA_initialized": False,
        "evaluator_subprocess_started": False, "next_experiment_started": False,
        "test_accessed": False, "updated_at": utc_now(),
    })
    dump_json(output / "state.json", state)
    write_environment(output, before=False)
    snapshot_sources(output)
    dump_text(output / "COMPLETE", decision["classification"] + "\n")
    write_manifest(output)
    create_archive(output)
    return 0


def synthetic_smoke() -> dict[str, Any]:
    stats = concentration(["a", "a", "b", "c"])
    records = []
    for replicate in range(REPLICATES):
        for round_id in range(ROUNDS):
            for slot in range(SELECTED_PER_ROUND):
                architecture = f"arch-{slot}" if round_id == 0 else ("arch-0" if slot == 0 else f"arch-{round_id}-{slot}")
                records.append({
                    "universe": "SYNTHETIC", "method": "SYNTHETIC",
                    "replicate_id": replicate, "round_id": round_id,
                    "selected_position": slot, "joint_fingerprint": f"j-{replicate}-{round_id}-{slot}",
                    "architecture_fingerprint": architecture,
                    "validation_mean": 0.70 + 0.01 * round_id + 0.001 * slot,
                })
    repeats = repeat_diagnosis("SYNTHETIC", records)
    candidates = []
    for replicate in range(REPLICATES):
        for round_id in range(ROUNDS):
            for slot in range(CANDIDATES_PER_ROUND):
                candidates.append({
                    "replicate_id": replicate, "round_id": round_id, "candidate_slot": slot,
                    "joint_fingerprint": f"c-{replicate}-{round_id}-{slot}",
                    "architecture_fingerprint": f"arch-{slot % 80}",
                    "score": float(CANDIDATES_PER_ROUND - slot),
                })
    synthetic_methods = {"X": {"universe": "S", "score_field": "score", "candidates": candidates}}
    p1 = build_counterfactual_policy("X", synthetic_methods, "P1")
    p2 = build_counterfactual_policy("X", synthetic_methods, "P2")
    passed = all((
        stats["unique_architecture_count"] == 3,
        stats["effective_architectures"] == math.exp(stats["shannon_entropy"]),
        stats["HHI"] == 0.375, stats["top2_share"] == 0.75,
        all(len({row["architecture_fingerprint"] for row in p1 if row["replicate_id"] == r and row["round_id"] == q}) == 4 for r in range(8) for q in range(4)),
        all(max(Counter(row["architecture_fingerprint"] for row in p2 if row["replicate_id"] == r).values()) <= 2 for r in range(8)),
        repeats["summary"]["mean_repeat_fraction"] > 0,
    ))
    return {
        "passed": passed, "new_GNN_runs": 0, "GPU_used": False,
        "CUDA_initialized": False, "evaluator_subprocess_started": False,
        "formal_output_created": False, "test_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--formal", action="store_true")
    modes.add_argument("--synthetic-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic_smoke:
        result = synthetic_smoke()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    try:
        return execute_formal(args.config, args.output)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
