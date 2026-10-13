#!/usr/bin/env python3
"""Validation-only pair-power projection audit for frozen joint configurations.

The formal path reconstructs validation results that already exist on disk and
performs only paired-t interval arithmetic.  It never invokes a GNN evaluator,
metric encoder, regressor, GP, VAE/decoder, or acquisition function.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# These imports intentionally reuse only frozen validation loaders, pair
# arithmetic, split checks, and hashing.  No training/evaluation entry point is
# imported or called by this audit.
from analyse import joint_configuration_metric_gate as joint_v1  # noqa: E402
from analyse import performance_metric_quick_gate as performance_v2  # noqa: E402,F401


FORMAT_VERSION = "joint_configuration_pair_power_projection_global4_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/joint_pair_power_projection_audit_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/posthoc/joint-configuration-pair-power-projection-global4-v1"
LABELS = ("positive", "negative", "ambiguous")
_T_CRITICAL_CACHE: dict[int, float] = {}
ALLOWED_CLASSIFICATIONS = {
    "PAIR_POWER_INPUT_INVALID",
    "PAIR_POWER_SUPPORT_LIMITED",
    "PAIR_POWER_PRECISION_LIMITED",
    "PAIR_POWER_MIXED_LIMITED",
    "RUN_FAILED",
}
CSV_FIELDS: dict[str, tuple[str, ...]] = {
    "seed_coverage_by_config.csv": (
        "joint_fingerprint", "architecture_fingerprint", "pool_id",
        "observed_validation_seeds", "existing_common_prefix_seed_count",
        "maximum_observed_seed", "coverage_status",
    ),
    "projection_by_n_and_fold.csv": (
        "n", "fold", "train_configuration_count", "total_pair_count",
        "positive_pairs", "negative_pairs", "ambiguous_pairs",
        "anchors_with_positive_and_negative",
        "cross_architecture_positive_pairs", "cross_factor_positive_pairs",
        "reliable_negative_pairs", "deficit_anchors_with_positive_and_negative",
        "deficit_cross_architecture_positive_pairs",
        "deficit_cross_factor_positive_pairs", "deficit_reliable_negative_pairs",
        "fold_gate_pass",
    ),
    "pair_minimum_n.csv": (
        "fold", "left_joint_fingerprint", "right_joint_fingerprint",
        "mean_difference_seed3", "paired_sd_seed3", "cross_architecture",
        "cross_factor", "minimum_n_for_positive", "minimum_n_for_negative",
        "never_positive_under_fixed_mean", "never_negative_under_fixed_mean",
        "positive_not_reached_by_n500", "negative_not_reached_by_n500",
    ),
    "calibration_prefix_results.csv": (
        "status", "n", "pair_count", "actual_positive", "actual_negative",
        "actual_ambiguous", "projected_positive", "projected_negative",
        "projected_ambiguous", "overall_label_accuracy", "macro_f1",
        "positive_precision", "positive_recall", "negative_precision",
        "negative_recall", "mean_absolute_ci_radius_error",
        "median_absolute_ci_radius_error", "mean_absolute_mean_drift",
        "median_absolute_mean_drift", "undefined_metrics",
    ),
    "full_topup_cost.csv": (
        "status", "target_n", "config_count", "configs_already_complete",
        "configs_needing_topup", "additional_gnn_runs",
        "total_resulting_seed_records", "gpu_hour_estimate",
    ),
    "topup_candidate_priority.csv": (
        "status", "target_n", "priority_rank_within_target",
        "joint_fingerprint", "architecture_fingerprint", "pool_id",
        "existing_common_prefix_seed_count", "additional_runs_to_target_n",
        "projected_positive_degree",
        "projected_cross_architecture_positive_degree",
        "projected_cross_factor_positive_degree",
        "number_of_outer_folds_in_which_it_appears_in_train",
        "number_of_outer_folds_in_which_it_can_create_a_missing_cross_factor_edge",
        "number_of_outer_folds_in_which_it_can_create_a_new_eligible_anchor",
    ),
}
REQUIRED_JSON_OUTPUTS = (
    "state.json", "frozen_protocol.json", "input_integrity.json",
    "data_inventory.json", "original_pair_gate_reproduction.json",
    "point_limit_pair_inventory.json", "projection_summary.json",
    "minimum_n_summary.json", "calibration_inventory.json",
    "calibration_transition_matrices.json", "calibration_metrics.json",
    "decision.json", "environment.json", "manifest.json",
)
REQUIRED_TEXT_OUTPUTS = (
    "final_report.md", "git_status.txt", "git_diff.patch",
    "logs/py_compile.log", "logs/pytest_focused.log",
    "logs/synthetic_smoke.log", "logs/run.log",
)
MANDATORY_SNAPSHOT_FILES = (
    "configs/joint_pair_power_projection_audit_v1.yaml",
    "analyse/joint_pair_power_projection_audit.py",
    "scripts/run_joint_pair_power_projection_audit.sh",
    "tests/test_joint_pair_power_projection_audit.py",
    "analyse/joint_configuration_metric_gate.py",
    "analyse/performance_metric_quick_gate.py",
)


class ProtocolError(RuntimeError):
    """A fail-closed audit, identity, leakage, or artifact violation."""


class InputInvalidError(ProtocolError):
    """The frozen v1 or validation input identity is invalid."""


class CalibrationUnavailable(ProtocolError):
    """The optional 30-seed reference cannot be safely recovered."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def dump_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n",
    )


def dump_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def dump_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row.get(field), sort_keys=True, separators=(",", ":"))
                if isinstance(row.get(field), (list, tuple, dict)) else row.get(field)
                for field in fields
            })
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ProtocolError("PyYAML is required to load the frozen audit config") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol_version") != FORMAT_VERSION:
        raise ProtocolError("audit config protocol_version mismatch")
    pair = value["pair_protocol"]
    if (
        float(pair["confidence"]) != 0.90
        or float(pair["equivalence_margin"]) != 0.006
        or float(pair["far_margin"]) != 0.012
    ):
        raise ProtocolError("frozen pair margins or confidence changed")
    if list(value["projection"]["seed_grid"]) != [3, 5, 7, 10, 15, 20, 30, 50, 100, 200]:
        raise ProtocolError("projection seed grid changed")
    if list(value["calibration"]["prefix_grid"]) != [3, 5, 7, 10, 15, 20, 30]:
        raise ProtocolError("calibration prefix grid changed")
    if list(value["cost"]["target_seed_counts"]) != [5, 7, 10, 15, 20, 30]:
        raise ProtocolError("cost seed targets changed")
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str, *, input_identity: bool = False) -> None:
    if condition:
        return
    if input_identity:
        raise InputInvalidError(message)
    raise ProtocolError(message)


def _relative_path(configured: str) -> Path:
    path = (REPO_ROOT / configured).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ProtocolError(f"configured path escapes repository: {configured}") from error
    return path


def ensure_output_available(output: Path, *, preflight_ready: bool = False) -> None:
    if output.exists() and not output.is_dir():
        raise ProtocolError(f"formal result path is not a directory: {output}")
    if not output.exists():
        return
    entries = {path.relative_to(output).as_posix() for path in output.rglob("*")}
    allowed = {
        "logs", "logs/py_compile.log", "logs/pytest_focused.log",
        "logs/synthetic_smoke.log", "logs/run.log",
    }
    if not entries:
        return
    if preflight_ready and entries <= allowed:
        return
    raise ProtocolError(f"formal result directory exists and is non-empty: {output}")


def validate_authoritative_v1(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["authoritative_v1"]
    root = _relative_path(str(frozen["result_root"]))
    archive = _relative_path(str(frozen["archive"]))
    sidecar = _relative_path(str(frozen["archive_sha256_sidecar"]))
    required = {
        "state": root / "state.json",
        "decision": root / "decision.json",
        "inventory": root / "data_inventory.json",
        "split": root / "split_manifest.json",
        "pairs": root / "pair_inventory.json",
        "marker": root / str(frozen["required_marker"]),
        "archive": archive,
        "archive_sidecar": sidecar,
    }
    for name, path in required.items():
        _require(path.is_file(), f"authoritative v1 {name} is missing: {path}", input_identity=True)
    archive_sha = sha256_file(archive)
    expected_archive_sha = str(frozen["archive_sha256"])
    _require(archive_sha == expected_archive_sha, "authoritative v1 archive SHA-256 mismatch", input_identity=True)
    sidecar_token = sidecar.read_text(encoding="utf-8").strip().split()
    _require(bool(sidecar_token) and sidecar_token[0] == expected_archive_sha,
             "authoritative v1 archive sidecar mismatch", input_identity=True)

    state = _read_json(required["state"])
    decision = _read_json(required["decision"])
    inventory = _read_json(required["inventory"])
    split = _read_json(required["split"])
    pair_inventory = _read_json(required["pairs"])
    expected_classification = str(frozen["classification"])
    checks = {
        "state_status": state.get("status") == "complete",
        "state_classification": state.get("decision") == expected_classification,
        "decision_classification": decision.get("classification") == expected_classification,
        "config_sha256": state.get("config_sha256") == str(frozen["config_sha256"]),
        "source_sha256": state.get("source_sha256") == str(frozen["source_sha256"]),
        "state_split_sha256": state.get("split_sha256") == str(frozen["split_sha256"]),
        "target": inventory.get("target") == str(frozen["target"]),
        "joint_count": int(inventory.get("unique_joint_configurations", -1)) == int(frozen["unique_joint_configurations"]),
        "architecture_count": int(inventory.get("unique_architectures", -1)) == int(frozen["unique_architectures"]),
        "pool_count": int(inventory.get("unique_pools", -1)) == int(frozen["unique_pools"]),
        "training_seeds": list(inventory.get("complete_training_seeds", [])) == list(frozen["training_seeds"]),
        "raw_validation_target": inventory.get("hp_residualization_used") is False,
        "test_data_not_accessed": state.get("test_data_accessed") is False,
        "candidate_training_not_started": state.get("candidate_gnn_training_started") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    _require(not failed, f"authoritative v1 identity checks failed: {failed}", input_identity=True)
    recorded_split_sha = str(split.get("split_sha256", ""))
    split_payload = dict(split)
    split_payload.pop("split_sha256", None)
    rebuilt_split_sha = joint_v1.stable_hash(split_payload)
    _require(
        recorded_split_sha == str(frozen["split_sha256"]) == rebuilt_split_sha,
        "authoritative logical split SHA-256 mismatch",
        input_identity=True,
    )
    _require(len(split.get("folds", [])) == 5, "authoritative split is not five-fold", input_identity=True)
    return {
        "status": "VALID",
        "result_root": str(root),
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "archive_sidecar": str(sidecar),
        "config_sha256": state["config_sha256"],
        "source_sha256": state["source_sha256"],
        "split_sha256": rebuilt_split_sha,
        "classification": expected_classification,
        "unique_joint_configurations": int(inventory["unique_joint_configurations"]),
        "unique_architectures": int(inventory["unique_architectures"]),
        "unique_pools": int(inventory["unique_pools"]),
        "training_seeds": list(inventory["complete_training_seeds"]),
        "target": inventory["target"],
        "test_data_accessed": False,
        "candidate_gnn_training_started": False,
        "split_manifest": split,
        "original_pair_inventory": pair_inventory,
        "checks": checks,
    }


def load_main_configurations(
    config: Mapping[str, Any], integrity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    try:
        configurations, inventory, provenance = joint_v1.load_and_aggregate_source(config)
        joint_v1.assert_split_integrity(configurations, integrity["split_manifest"])
    except joint_v1.ProtocolError as error:
        raise InputInvalidError(str(error)) from error
    expected = config["authoritative_v1"]
    checks = {
        "joint_count": len(configurations) == int(expected["unique_joint_configurations"]),
        "architecture_count": len({row["architecture_fingerprint"] for row in configurations}) == int(expected["unique_architectures"]),
        "pool_count": len({int(row["pool_id"]) for row in configurations}) == int(expected["unique_pools"]),
        "training_seeds": inventory.get("complete_training_seeds") == [0, 1, 2],
        "target": inventory.get("target") == "raw_validation_accuracy_seed_mean",
        "no_hp_residualization": inventory.get("hp_residualization_used") is False,
        "no_test_data": provenance.get("test_data_accessed") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    _require(not failed, f"reconstructed main data checks failed: {failed}", input_identity=True)
    inventory = {
        **inventory,
        "unique_pools": len({int(row["pool_id"]) for row in configurations}),
        "primary_projection_parameter_seeds": [0, 1, 2],
        "phase_c_values_used_in_primary_projection": False,
        "metric_model_training_started": False,
        "gnn_training_started": False,
        "test_data_accessed": False,
    }
    return configurations, inventory, provenance


def t_critical_90(n: int) -> float:
    if int(n) < 3:
        raise ProtocolError("paired projection requires n >= 3")
    if int(n) in _T_CRITICAL_CACHE:
        return _T_CRITICAL_CACHE[int(n)]
    try:
        from scipy.stats import t
    except ImportError as error:
        raise ProtocolError("SciPy is required for exact Student-t critical values") from error
    value = float(t.ppf(0.95, df=int(n) - 1))
    if not math.isfinite(value):
        raise ProtocolError("non-finite Student-t critical value")
    _T_CRITICAL_CACHE[int(n)] = value
    return value


def projected_interval(
    mean3: float, sd3: float, n: int, confidence: float = 0.90,
) -> dict[str, float]:
    if float(confidence) != 0.90:
        raise ProtocolError("only the frozen 90% paired interval is supported")
    mean_value = float(mean3)
    sd_value = float(sd3)
    if not math.isfinite(mean_value) or not math.isfinite(sd_value) or sd_value < 0.0:
        raise ProtocolError("projection mean and paired SD must be finite, with SD nonnegative")
    critical = t_critical_90(int(n))
    radius = critical * sd_value / math.sqrt(float(n))
    return {
        "mean_difference": mean_value,
        "paired_sd": sd_value,
        "t_critical": critical,
        "degrees_of_freedom": int(n) - 1,
        "radius": radius,
        "ci_lower": mean_value - radius,
        "ci_upper": mean_value + radius,
    }


def label_interval(
    lower: float, upper: float, *, equivalence_margin: float = 0.006,
    far_margin: float = 0.012,
) -> str:
    if float(lower) >= -float(equivalence_margin) and float(upper) <= float(equivalence_margin):
        return "positive"
    if float(upper) < -float(far_margin) or float(lower) > float(far_margin):
        return "negative"
    return "ambiguous"


def classify_projected_pair(
    mean3: float, sd3: float, n: int, *, equivalence_margin: float = 0.006,
    far_margin: float = 0.012,
) -> dict[str, Any]:
    interval = projected_interval(mean3, sd3, n)
    return {
        **interval,
        "label": label_interval(
            interval["ci_lower"], interval["ci_upper"],
            equivalence_margin=equivalence_margin, far_margin=far_margin,
        ),
    }


def classify_point_limit(
    mean3: float, *, equivalence_margin: float = 0.006,
    far_margin: float = 0.012,
) -> str:
    return label_interval(
        float(mean3), float(mean3), equivalence_margin=equivalence_margin,
        far_margin=far_margin,
    )


def _pair_thresholds(config: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in config["pair_protocol"]["power_gate_per_fold"].items()
    }


def build_fold_pair_base(
    train_rows: Sequence[Mapping[str, Any]], *, allowed_joint_fingerprints: set[str],
    equivalence_margin: float = 0.006, far_margin: float = 0.012,
) -> dict[str, Any]:
    materialized = [dict(row) for row in train_rows]
    observed = {str(row["joint_fingerprint"]) for row in materialized}
    if observed != set(map(str, allowed_joint_fingerprints)):
        forbidden = sorted(observed - set(map(str, allowed_joint_fingerprints)))
        missing = sorted(set(map(str, allowed_joint_fingerprints)) - observed)
        raise ProtocolError(
            f"fold pair base differs from allowed outer-train set; forbidden={forbidden[:3]} missing={missing[:3]}"
        )
    try:
        standardizer = joint_v1.fit_hp_standardizer(materialized)
        inventory = joint_v1.construct_statistical_pairs(
            materialized,
            allowed_joint_fingerprints=set(observed),
            hp_standardizer=standardizer,
            hp_median_fit_rows=materialized,
            equivalence_margin=float(equivalence_margin),
            far_margin=float(far_margin),
        )
    except joint_v1.ProtocolError as error:
        raise ProtocolError(str(error)) from error
    expected_pairs = len(materialized) * (len(materialized) - 1) // 2
    if len(inventory["pairs"]) != expected_pairs:
        raise ProtocolError("fold pair enumeration is not complete and unordered")
    pairs = []
    for pair in inventory["pairs"]:
        pairs.append({
            "left": str(pair["left"]),
            "right": str(pair["right"]),
            "mean3": float(pair["mean_difference"]),
            "sd3": float(pair["paired_standard_error"]) * math.sqrt(3.0),
            "hp_distance": float(pair["hp_distance"]),
            "cross_architecture": bool(pair["cross_architecture"]),
            "cross_factor": bool(pair["cross_factor"]),
            "label_n3_v1": str(pair["label"]),
        })
    return {
        "train_joint_fingerprints": sorted(observed),
        "train_configuration_count": len(materialized),
        "hp_standardizer": standardizer,
        "hp_distance_median": float(inventory["hp_distance_median"]),
        "pairs": pairs,
        "v1_inventory": inventory,
        "outer_heldout_labels_used": False,
        "primary_parameter_seeds": [0, 1, 2],
    }


def _gate_inventory(
    base: Mapping[str, Any], pair_labels: Sequence[str], thresholds: Mapping[str, int],
) -> dict[str, Any]:
    pairs = list(base["pairs"])
    if len(pairs) != len(pair_labels):
        raise ProtocolError("pair label vector length mismatch")
    positive_neighbors: dict[str, set[str]] = defaultdict(set)
    negative_neighbors: dict[str, set[str]] = defaultdict(set)
    counts_by_label = {label: 0 for label in LABELS}
    cross_architecture_positive = 0
    cross_factor_positive = 0
    for pair, label in zip(pairs, pair_labels):
        if label not in counts_by_label:
            raise ProtocolError(f"unknown pair label: {label}")
        counts_by_label[label] += 1
        left, right = str(pair["left"]), str(pair["right"])
        if label == "positive":
            positive_neighbors[left].add(right)
            positive_neighbors[right].add(left)
            cross_architecture_positive += int(bool(pair["cross_architecture"]))
            cross_factor_positive += int(bool(pair["cross_factor"]))
        elif label == "negative":
            negative_neighbors[left].add(right)
            negative_neighbors[right].add(left)
    anchors = list(base["train_joint_fingerprints"])
    eligible = sorted(
        anchor for anchor in anchors
        if positive_neighbors.get(anchor) and negative_neighbors.get(anchor)
    )
    counts = {
        "positive_pairs": counts_by_label["positive"],
        "negative_pairs": counts_by_label["negative"],
        "ambiguous_pairs": counts_by_label["ambiguous"],
        "anchors_with_positive_and_negative": len(eligible),
        "cross_architecture_positive_pairs": cross_architecture_positive,
        "cross_factor_positive_pairs": cross_factor_positive,
        "reliable_negative_pairs": counts_by_label["negative"],
    }
    deficits = {key: max(0, int(thresholds[key]) - int(counts[key])) for key in thresholds}
    try:
        joint_v1.assert_pair_power({"counts": counts}, thresholds)
        gate_pass = True
    except joint_v1.UnderpoweredError:
        gate_pass = False
    return {
        "train_configuration_count": int(base["train_configuration_count"]),
        "total_pair_count": len(pairs),
        **counts,
        "deficits": deficits,
        "failed_gate_items": [key for key, value in deficits.items() if value > 0],
        "fold_gate_pass": gate_pass,
        "eligible_anchors": eligible,
        "positive_neighbors": {key: sorted(value) for key, value in positive_neighbors.items()},
        "negative_neighbors": {key: sorted(value) for key, value in negative_neighbors.items()},
        "pair_labels": list(pair_labels),
    }


def inventory_at_n(
    base: Mapping[str, Any], n: int, thresholds: Mapping[str, int],
    *, equivalence_margin: float = 0.006, far_margin: float = 0.012,
) -> dict[str, Any]:
    labels = [
        classify_projected_pair(
            pair["mean3"], pair["sd3"], int(n),
            equivalence_margin=equivalence_margin, far_margin=far_margin,
        )["label"]
        for pair in base["pairs"]
    ]
    return _gate_inventory(base, labels, thresholds)


def point_limit_inventory(
    base: Mapping[str, Any], thresholds: Mapping[str, int],
    *, equivalence_margin: float = 0.006, far_margin: float = 0.012,
) -> dict[str, Any]:
    labels = [
        classify_point_limit(
            pair["mean3"], equivalence_margin=equivalence_margin,
            far_margin=far_margin,
        )
        for pair in base["pairs"]
    ]
    return _gate_inventory(base, labels, thresholds)


def classify_point_gate(fold_passes: Sequence[bool]) -> str:
    values = [bool(value) for value in fold_passes]
    if len(values) != 5:
        raise ProtocolError("point-limit classification requires exactly five fold statuses")
    if all(values):
        return "PAIR_POWER_PRECISION_LIMITED"
    if not any(values):
        return "PAIR_POWER_SUPPORT_LIMITED"
    return "PAIR_POWER_MIXED_LIMITED"


def minimum_seed_requirements(
    pair: Mapping[str, Any], *, n_min: int = 3, n_max: int = 500,
    equivalence_margin: float = 0.006, far_margin: float = 0.012,
) -> dict[str, Any]:
    mean3 = float(pair["mean3"])
    sd3 = float(pair["sd3"])
    never_positive = abs(mean3) > float(equivalence_margin)
    never_negative = abs(mean3) <= float(far_margin)
    def first_n(label: str, impossible: bool) -> int | None:
        if impossible:
            return None
        def reaches(n: int) -> bool:
            return classify_projected_pair(
                mean3, sd3, n, equivalence_margin=equivalence_margin,
                far_margin=far_margin,
            )["label"] == label
        if not reaches(int(n_max)):
            return None
        lower, upper = int(n_min), int(n_max)
        while lower < upper:
            middle = (lower + upper) // 2
            if reaches(middle):
                upper = middle
            else:
                lower = middle + 1
        return lower

    minimum_positive = first_n("positive", never_positive)
    minimum_negative = first_n("negative", never_negative)
    return {
        "minimum_n_for_positive": minimum_positive,
        "minimum_n_for_negative": minimum_negative,
        "never_positive_under_fixed_mean": never_positive,
        "never_negative_under_fixed_mean": never_negative,
        "positive_not_reached_by_n500": minimum_positive is None and not never_positive,
        "negative_not_reached_by_n500": minimum_negative is None and not never_negative,
    }


def _linear_quantile(values: Sequence[int], probability: float) -> float | None:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_minimum_n_rows(
    rows: Sequence[Mapping[str, Any]], category: str,
) -> dict[str, Any]:
    if category == "all_positive_candidates":
        scope = list(rows)
    elif category == "cross_architecture_positive_candidates":
        scope = [row for row in rows if bool(row["cross_architecture"])]
    elif category == "cross_factor_positive_candidates":
        scope = [row for row in rows if bool(row["cross_factor"])]
    else:
        raise ProtocolError(f"unknown minimum-n category: {category}")
    selected = [row for row in scope if not bool(row["never_positive_under_fixed_mean"])]
    finite = [int(row["minimum_n_for_positive"]) for row in selected if row["minimum_n_for_positive"] is not None]
    return {
        "category": category,
        "scope_pair_count": len(scope),
        "candidate_pair_count": len(selected),
        "never_positive_pair_count": len(scope) - len(selected),
        "finite_by_n500_count": len(finite),
        "not_reached_by_n500_count": len(selected) - len(finite),
        "median": _linear_quantile(finite, 0.50),
        "q75": _linear_quantile(finite, 0.75),
        "q90": _linear_quantile(finite, 0.90),
        "q95": _linear_quantile(finite, 0.95),
        "maximum_finite_n": max(finite) if finite else None,
        "quantile_method": "linear_interpolation_of_ordered_finite_integer_n",
    }


def _reject_test_fields(value: Any, context: str) -> None:
    hits = joint_v1.nested_test_key_hits(value)
    if hits:
        raise InputInvalidError(f"test field rejected at loader layer in {context}: {hits[:5]}")


def _same_value(left: Any, right: Any) -> bool:
    return joint_v1.stable_hash(left) == joint_v1.stable_hash(right)


def load_calibration_subset(
    config: Mapping[str, Any], main_configurations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    calibration = config["calibration"]
    root = _relative_path(str(config["data"]["source_root"]))
    union_path = root / str(calibration["union_configs"])
    statistics_path = root / str(calibration["strategy_statistics"])
    shard_paths = [root / str(relative) for relative in calibration["result_shards"]]
    merged_path = root / str(calibration["merged_result_crosscheck"])
    paths = [union_path, statistics_path, *shard_paths, merged_path]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise CalibrationUnavailable(f"calibration files are missing: {missing}")
    expected_hashes = {
        union_path: str(calibration["union_config_sha256"]),
        statistics_path: str(calibration["strategy_statistics_sha256"]),
        merged_path: str(calibration["merged_result_sha256"]),
    }
    mismatches = [str(path) for path, expected in expected_hashes.items() if sha256_file(path) != expected]
    if mismatches:
        raise InputInvalidError(f"calibration frozen file SHA-256 mismatch: {mismatches}")
    concatenated = b"".join(path.read_bytes() for path in shard_paths)
    if concatenated != merged_path.read_bytes():
        raise InputInvalidError("calibration raw shards do not byte-concatenate to the frozen merged JSONL")

    union_rows = _read_json(union_path)
    statistics_rows = _read_json(statistics_path)
    if not isinstance(union_rows, list) or not isinstance(statistics_rows, list):
        raise InputInvalidError("calibration identity artifacts must be JSON arrays")
    _reject_test_fields(union_rows, str(union_path))
    _reject_test_fields(statistics_rows, str(statistics_path))
    union_by_joint: dict[str, dict[str, Any]] = {}
    for raw in union_rows:
        if not isinstance(raw, Mapping) or "joint_config_fingerprint" not in raw:
            raise InputInvalidError("calibration union row lacks a joint fingerprint")
        joint = str(raw["joint_config_fingerprint"])
        if joint in union_by_joint:
            raise InputInvalidError(f"duplicate calibration union identity: {joint}")
        union_by_joint[joint] = dict(raw)
    expected_count = int(calibration["expected_configurations"])
    if len(union_by_joint) != expected_count:
        raise InputInvalidError("calibration union does not contain exactly 14 unique configurations")
    main_by_joint = {str(row["joint_fingerprint"]): dict(row) for row in main_configurations}
    if not set(union_by_joint) <= set(main_by_joint):
        raise InputInvalidError("calibration union contains a configuration outside the primary 336")
    for joint, union in union_by_joint.items():
        main = main_by_joint[joint]
        identity_checks = {
            "architecture_fingerprint": str(union.get("architecture_fingerprint")) == str(main["architecture_fingerprint"]),
            "hp_fingerprint": str(union.get("hp_fingerprint")) == str(main["hp_fingerprint"]),
            "pool_id": int(union.get("pool_id", -1)) == int(main["pool_id"]),
            "batch_id": int(union.get("batch_id", -1)) == int(main["batch_id"]),
            "architecture_slot": int(union.get("architecture_slot", -1)) == int(main["slot_id"]),
            "operations": _same_value(union.get("operations"), main["operations"]),
            "edges": _same_value(sorted(union.get("edges", [])), sorted(main["edges"])),
            "lr": float(union.get("lr")) == float(main["lr"]),
            "dropout": float(union.get("dropout")) == float(main["dropout"]),
            "hidden_dim": int(union.get("hidden_dim")) == int(main["hidden_dim"]),
            "l2": float(union.get("l2")) == float(main["l2"]),
        }
        failed = sorted(key for key, passed in identity_checks.items() if not passed)
        if failed:
            raise InputInvalidError(f"calibration union/main identity conflict for {joint}: {failed}")

    raw_rows: list[dict[str, Any]] = []
    for path in shard_paths:
        try:
            raw_rows.extend(joint_v1._read_result_jsonl(path))
        except joint_v1.ProtocolError as error:
            raise InputInvalidError(str(error)) from error
    expected_raw_rows = expected_count * 27
    if len(raw_rows) != expected_raw_rows:
        raise InputInvalidError(
            f"calibration seed-row count mismatch: observed={len(raw_rows)} expected={expected_raw_rows}"
        )
    values_by_joint: dict[str, dict[int, float]] = {
        joint: {
            0: float(main_by_joint[joint]["validation_seed_0"]),
            1: float(main_by_joint[joint]["validation_seed_1"]),
            2: float(main_by_joint[joint]["validation_seed_2"]),
        }
        for joint in union_by_joint
    }
    source_rows_by_joint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_rows:
        joint = str(raw["joint_config_fingerprint"])
        if joint not in union_by_joint:
            raise InputInvalidError(f"foreign configuration in calibration raw results: {joint}")
        if raw.get("terminal_status") != "success" or raw.get("valid") is not True:
            raise InputInvalidError(f"unsuccessful calibration validation row: {joint}/{raw.get('seed')}")
        if str(raw.get("dataset", "")).lower() != "cora" or str(raw.get("metric")) != "accuracy":
            raise InputInvalidError("calibration row is not Cora validation accuracy")
        seed = int(raw["seed"])
        if seed < 3 or seed > 29:
            raise InputInvalidError(f"foreign calibration seed: {seed}")
        value = float(raw["val_acc"])
        if not math.isfinite(value):
            raise InputInvalidError("non-finite calibration validation accuracy")
        if seed in values_by_joint[joint]:
            raise InputInvalidError(f"duplicate/conflicting calibration joint/seed: {joint}/{seed}")
        union = union_by_joint[joint]
        try:
            canonical = joint_v1.normalize_architecture(raw)
        except joint_v1.ProtocolError as error:
            raise InputInvalidError(str(error)) from error
        identity_checks = {
            "architecture_fingerprint": str(raw["architecture_fingerprint"]) == str(union["architecture_fingerprint"]),
            "hp_fingerprint": str(raw["hp_fingerprint"]) == str(union["hp_fingerprint"]),
            "pool_id": int(raw["pool_id"]) == int(union["pool_id"]),
            "batch_id": int(raw["batch_id"]) == int(union["batch_id"]),
            "architecture_slot": int(raw["architecture_slot"]) == int(union["architecture_slot"]),
            "operations": _same_value(canonical["operations"], union["operations"]),
            "edges": _same_value(canonical["edges"], sorted(union["edges"])),
            "architecture_hash": joint_v1.architecture_fingerprint(canonical) == str(raw["architecture_fingerprint"]),
            "hp_hash": joint_v1.hp_fingerprint([float(value) for value in raw["raw_hp_vector"]]) == str(raw["hp_fingerprint"]),
        }
        failed = sorted(key for key, passed in identity_checks.items() if not passed)
        if failed:
            raise InputInvalidError(f"calibration raw identity conflict for {joint}/{seed}: {failed}")
        expected_joint = joint_v1.stable_hash({
            "architecture_fingerprint": str(raw["architecture_fingerprint"]),
            "hp_fingerprint": str(raw["hp_fingerprint"]),
        })
        if expected_joint != joint:
            raise InputInvalidError(f"calibration joint fingerprint mismatch: {joint}/{seed}")
        main_protocol = str(main_by_joint[joint]["training_protocol_identity"])
        if joint_v1.stable_hash(joint_v1._protocol_identity(raw)) != main_protocol:
            raise InputInvalidError(f"calibration training protocol identity mismatch: {joint}/{seed}")
        values_by_joint[joint][seed] = value
        source_rows_by_joint[joint].append(raw)

    required_seeds = set(range(30))
    for joint, seed_values in values_by_joint.items():
        if set(seed_values) != required_seeds:
            raise InputInvalidError(
                f"calibration configuration lacks exact seeds 0..29: {joint}/{sorted(seed_values)}"
            )

    stats_by_joint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in statistics_rows:
        if not isinstance(raw, Mapping):
            raise InputInvalidError("calibration statistics row is not a mapping")
        joint = str(raw.get("joint_config_fingerprint", ""))
        if joint not in union_by_joint:
            raise InputInvalidError(f"foreign configuration in calibration statistics: {joint}")
        if list(raw.get("selection_seed_ids", [])) != [0, 1, 2]:
            raise InputInvalidError("calibration statistics selection seeds are not [0,1,2]")
        reported = [float(value) for value in raw.get("all30_seed_values", [])]
        actual = [values_by_joint[joint][seed] for seed in range(30)]
        if reported != actual:
            raise InputInvalidError(f"calibration statistics/raw validation conflict: {joint}")
        stats_by_joint[joint].append(dict(raw))
    if set(stats_by_joint) != set(union_by_joint):
        raise InputInvalidError("calibration statistics do not cover all 14 union configurations")
    for joint, rows in stats_by_joint.items():
        arrays = {stable_hash(row["all30_seed_values"]) for row in rows}
        if len(arrays) != 1:
            raise InputInvalidError(f"cross-strategy calibration statistics conflict: {joint}")

    subset = {
        joint: {
            "joint_fingerprint": joint,
            "architecture_fingerprint": str(union_by_joint[joint]["architecture_fingerprint"]),
            "hp_fingerprint": str(union_by_joint[joint]["hp_fingerprint"]),
            "pool_id": int(union_by_joint[joint]["pool_id"]),
            "values": [values_by_joint[joint][seed] for seed in range(30)],
            "seeds": list(range(30)),
        }
        for joint in sorted(union_by_joint)
    }
    inventory = {
        "calibration_status": "AVAILABLE",
        "configuration_count": len(subset),
        "pair_count": len(subset) * (len(subset) - 1) // 2,
        "required_seed_range": [0, 29],
        "complete_seed_coverage": True,
        "unique_architectures": len({row["architecture_fingerprint"] for row in subset.values()}),
        "unique_hp_fingerprints": len({row["hp_fingerprint"] for row in subset.values()}),
        "unique_pools": len({row["pool_id"] for row in subset.values()}),
        "union_configs": {"path": str(union_path), "sha256": sha256_file(union_path)},
        "strategy_statistics": {"path": str(statistics_path), "sha256": sha256_file(statistics_path)},
        "raw_result_shards": [
            {"path": str(path), "sha256": sha256_file(path), "rows": sum(1 for row in raw_rows if row["source_file"] == str(path.resolve()))}
            for path in shard_paths
        ],
        "merged_crosscheck": {"path": str(merged_path), "sha256": sha256_file(merged_path), "byte_equal_to_shard_concatenation": True},
        "reference_name": "30-seed reference",
        "subset_limitation": "Selected high-performing stabilization candidates; not a random representative sample of the 336 configurations.",
        "used_for_model_training": False,
        "used_for_primary_projection_parameters": False,
        "may_change_primary_classification": False,
        "test_data_accessed": False,
    }
    return subset, inventory


def paired_prefix_statistics(
    left_values: Sequence[float], right_values: Sequence[float], n: int,
) -> dict[str, float]:
    if int(n) < 3:
        raise ProtocolError("paired prefix requires n >= 3")
    # Access exactly the requested prefix.  Tests use guarded sequences to
    # enforce the separation between projection parameters and actual prefixes.
    differences = [float(left_values[index]) - float(right_values[index]) for index in range(int(n))]
    mean = sum(differences) / float(n)
    variance = sum((value - mean) ** 2 for value in differences) / float(int(n) - 1)
    sd = math.sqrt(max(variance, 0.0))
    interval = projected_interval(mean, sd, int(n))
    return {**interval, "label": label_interval(interval["ci_lower"], interval["ci_upper"])}


def project_calibration_pair_from_first_three(
    left_values: Sequence[float], right_values: Sequence[float], n: int,
) -> dict[str, Any]:
    left_first_three = [float(left_values[index]) for index in range(3)]
    right_first_three = [float(right_values[index]) for index in range(3)]
    seed3 = paired_prefix_statistics(left_first_three, right_first_three, 3)
    return classify_projected_pair(seed3["mean_difference"], seed3["paired_sd"], int(n))


def _median(values: Sequence[float]) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _classification_metrics(
    actual: Sequence[str], projected: Sequence[str],
) -> dict[str, Any]:
    if len(actual) != len(projected) or not actual:
        raise ProtocolError("calibration classification vectors must be nonempty and aligned")
    undefined: list[dict[str, str]] = []
    per_label: dict[str, dict[str, float | None]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        true_positive = sum(a == label and p == label for a, p in zip(actual, projected))
        predicted_count = sum(p == label for p in projected)
        actual_count = sum(a == label for a in actual)
        if predicted_count:
            precision: float | None = true_positive / predicted_count
        else:
            precision = None
            undefined.append({"metric": f"{label}_precision", "reason": "no projected labels in class"})
        if actual_count:
            recall: float | None = true_positive / actual_count
        else:
            recall = None
            undefined.append({"metric": f"{label}_recall", "reason": "no actual-prefix labels in class"})
        precision_for_f1 = 0.0 if precision is None else precision
        recall_for_f1 = 0.0 if recall is None else recall
        denominator = precision_for_f1 + recall_for_f1
        f1 = 0.0 if denominator == 0.0 else 2.0 * precision_for_f1 * recall_for_f1 / denominator
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision, "recall": recall, "f1_zero_division_zero": f1,
            "actual_count": actual_count, "projected_count": predicted_count,
        }
    return {
        "overall_label_accuracy": sum(a == p for a, p in zip(actual, projected)) / len(actual),
        "macro_f1": sum(f1_values) / len(f1_values),
        "positive_precision": per_label["positive"]["precision"],
        "positive_recall": per_label["positive"]["recall"],
        "negative_precision": per_label["negative"]["precision"],
        "negative_recall": per_label["negative"]["recall"],
        "per_label": per_label,
        "undefined_metrics": undefined,
        "macro_f1_zero_division_policy": "undefined class precision/recall contributes class F1=0",
    }


def run_calibration_replay(
    subset: Mapping[str, Mapping[str, Any]], prefix_grid: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = [subset[key] for key in sorted(subset)]
    if len(rows) != 14:
        raise ProtocolError("calibration replay requires exactly 14 configurations")
    result_rows: list[dict[str, Any]] = []
    transition_output: dict[str, Any] = {"status": "AVAILABLE", "label_order": list(LABELS), "by_n": {}}
    detailed_metrics: dict[str, Any] = {
        "calibration_status": "AVAILABLE",
        "reference_name": "30-seed reference",
        "projected_parameters_use_only_seeds": [0, 1, 2],
        "actual_prefix_uses_seeds": "0..n-1",
        "subset_limitation": "High-performing stabilization candidates; not a random representative sample of all 336 configurations.",
        "by_n": {},
    }
    for n in map(int, prefix_grid):
        actual_labels: list[str] = []
        projected_labels: list[str] = []
        radius_errors: list[float] = []
        mean_drifts: list[float] = []
        transitions = {projected: {actual: 0 for actual in LABELS} for projected in LABELS}
        for left, right in itertools.combinations(rows, 2):
            actual = paired_prefix_statistics(left["values"], right["values"], n)
            projected = project_calibration_pair_from_first_three(left["values"], right["values"], n)
            actual_label = str(actual["label"])
            projected_label = str(projected["label"])
            actual_labels.append(actual_label)
            projected_labels.append(projected_label)
            transitions[projected_label][actual_label] += 1
            radius_errors.append(abs(float(actual["radius"]) - float(projected["radius"])))
            mean_drifts.append(abs(float(actual["mean_difference"]) - float(projected["mean_difference"])))
        metrics = _classification_metrics(actual_labels, projected_labels)
        actual_counts = {label: actual_labels.count(label) for label in LABELS}
        projected_counts = {label: projected_labels.count(label) for label in LABELS}
        row = {
            "status": "AVAILABLE", "n": n, "pair_count": len(actual_labels),
            **{f"actual_{label}": actual_counts[label] for label in LABELS},
            **{f"projected_{label}": projected_counts[label] for label in LABELS},
            "overall_label_accuracy": metrics["overall_label_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "positive_precision": metrics["positive_precision"],
            "positive_recall": metrics["positive_recall"],
            "negative_precision": metrics["negative_precision"],
            "negative_recall": metrics["negative_recall"],
            "mean_absolute_ci_radius_error": sum(radius_errors) / len(radius_errors),
            "median_absolute_ci_radius_error": _median(radius_errors),
            "mean_absolute_mean_drift": sum(mean_drifts) / len(mean_drifts),
            "median_absolute_mean_drift": _median(mean_drifts),
            "undefined_metrics": metrics["undefined_metrics"],
        }
        result_rows.append(row)
        transition_output["by_n"][str(n)] = transitions
        detailed_metrics["by_n"][str(n)] = {
            **row, "per_label": metrics["per_label"],
            "macro_f1_zero_division_policy": metrics["macro_f1_zero_division_policy"],
            "transition_matrix_orientation": "rows=projected, columns=actual-prefix",
        }
    return result_rows, transition_output, detailed_metrics


def existing_common_prefix_seed_count(seeds: Iterable[int]) -> int:
    materialized = {int(seed) for seed in seeds}
    count = 0
    while count in materialized:
        count += 1
    return count


def full_topup_cost(
    coverage: Mapping[str, set[int]], target_seed_counts: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in map(int, target_seed_counts):
        prefix_counts = [existing_common_prefix_seed_count(seeds) for seeds in coverage.values()]
        additional = [max(0, target - count) for count in prefix_counts]
        rows.append({
            "status": "AVAILABLE", "target_n": target, "config_count": len(coverage),
            "configs_already_complete": sum(count >= target for count in prefix_counts),
            "configs_needing_topup": sum(value > 0 for value in additional),
            "additional_gnn_runs": sum(additional),
            "total_resulting_seed_records": len(coverage) * target,
            "gpu_hour_estimate": "unavailable",
        })
    return rows


def priority_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["number_of_outer_folds_in_which_it_can_create_a_missing_cross_factor_edge"]),
        -int(row["number_of_outer_folds_in_which_it_can_create_a_new_eligible_anchor"]),
        -int(row["projected_cross_factor_positive_degree"]),
        -int(row["projected_cross_architecture_positive_degree"]),
        int(row["additional_runs_to_target_n"]),
        str(row["joint_fingerprint"]),
    )


def build_priority_rows(
    configurations: Sequence[Mapping[str, Any]], fold_bases: Mapping[int, Mapping[str, Any]],
    coverage: Mapping[str, set[int]], target_seed_counts: Sequence[int],
    thresholds: Mapping[str, int],
) -> list[dict[str, Any]]:
    by_joint = {str(row["joint_fingerprint"]): dict(row) for row in configurations}
    output: list[dict[str, Any]] = []
    for target in map(int, target_seed_counts):
        positive_neighbors: dict[str, set[str]] = defaultdict(set)
        cross_arch_neighbors: dict[str, set[str]] = defaultdict(set)
        cross_factor_neighbors: dict[str, set[str]] = defaultdict(set)
        appearances: dict[str, set[int]] = defaultdict(set)
        missing_cross_factor_folds: dict[str, set[int]] = defaultdict(set)
        new_anchor_folds: dict[str, set[int]] = defaultdict(set)
        for fold, base in sorted(fold_bases.items()):
            inventory3 = inventory_at_n(base, 3, thresholds)
            inventory_target = inventory_at_n(base, target, thresholds)
            eligible3 = set(inventory3["eligible_anchors"])
            eligible_target = set(inventory_target["eligible_anchors"])
            for joint in base["train_joint_fingerprints"]:
                appearances[str(joint)].add(int(fold))
                if joint in eligible_target and joint not in eligible3:
                    new_anchor_folds[str(joint)].add(int(fold))
            for pair, label3, label_target in zip(
                base["pairs"], inventory3["pair_labels"], inventory_target["pair_labels"],
            ):
                left, right = str(pair["left"]), str(pair["right"])
                if label_target == "positive":
                    positive_neighbors[left].add(right)
                    positive_neighbors[right].add(left)
                    if bool(pair["cross_architecture"]):
                        cross_arch_neighbors[left].add(right)
                        cross_arch_neighbors[right].add(left)
                    if bool(pair["cross_factor"]):
                        cross_factor_neighbors[left].add(right)
                        cross_factor_neighbors[right].add(left)
                        if label3 != "positive":
                            missing_cross_factor_folds[left].add(int(fold))
                            missing_cross_factor_folds[right].add(int(fold))
        ranked: list[dict[str, Any]] = []
        for joint in sorted(by_joint):
            config_row = by_joint[joint]
            prefix = existing_common_prefix_seed_count(coverage[joint])
            ranked.append({
                "status": "PLANNING_ONLY", "target_n": target,
                "priority_rank_within_target": 0,
                "joint_fingerprint": joint,
                "architecture_fingerprint": str(config_row["architecture_fingerprint"]),
                "pool_id": int(config_row["pool_id"]),
                "existing_common_prefix_seed_count": prefix,
                "additional_runs_to_target_n": max(0, target - prefix),
                "projected_positive_degree": len(positive_neighbors[joint]),
                "projected_cross_architecture_positive_degree": len(cross_arch_neighbors[joint]),
                "projected_cross_factor_positive_degree": len(cross_factor_neighbors[joint]),
                "number_of_outer_folds_in_which_it_appears_in_train": len(appearances[joint]),
                "number_of_outer_folds_in_which_it_can_create_a_missing_cross_factor_edge": len(missing_cross_factor_folds[joint]),
                "number_of_outer_folds_in_which_it_can_create_a_new_eligible_anchor": len(new_anchor_folds[joint]),
            })
        ranked.sort(key=priority_sort_key)
        for rank, row in enumerate(ranked, 1):
            row["priority_rank_within_target"] = rank
        output.extend(ranked)
    return output


def _compact_gate_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in inventory.items()
        if key not in {"positive_neighbors", "negative_neighbors", "pair_labels"}
    }


def run_primary_projection(
    config: Mapping[str, Any], configurations: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
) -> tuple[
    dict[int, dict[str, Any]], dict[str, Any], dict[str, Any],
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any],
]:
    thresholds = _pair_thresholds(config)
    pair_protocol = config["pair_protocol"]
    eq = float(pair_protocol["equivalence_margin"])
    far = float(pair_protocol["far_margin"])
    by_joint = {str(row["joint_fingerprint"]): dict(row) for row in configurations}
    original_by_fold = {
        int(row["fold"]): row for row in integrity["original_pair_inventory"].get("folds", [])
    }
    if set(original_by_fold) != set(range(5)):
        raise InputInvalidError("authoritative v1 pair inventory does not contain folds 0..4")
    fold_bases: dict[int, dict[str, Any]] = {}
    reproduction_folds: list[dict[str, Any]] = []
    for fold_row in integrity["split_manifest"]["folds"]:
        fold = int(fold_row["fold"])
        train_ids = set(map(str, fold_row["train_joint_fingerprints"]))
        heldout_ids = set(map(str, fold_row["heldout_joint_fingerprints"]))
        if train_ids & heldout_ids:
            raise InputInvalidError(f"train/heldout joint overlap in frozen fold {fold}")
        try:
            train_rows = [by_joint[joint] for joint in sorted(train_ids)]
        except KeyError as error:
            raise InputInvalidError(f"split references unknown configuration: {error}") from error
        base = build_fold_pair_base(
            train_rows, allowed_joint_fingerprints=train_ids,
            equivalence_margin=eq, far_margin=far,
        )
        fold_bases[fold] = base
        current_counts = dict(base["v1_inventory"]["counts"])
        projected3 = inventory_at_n(base, 3, thresholds, equivalence_margin=eq, far_margin=far)
        expected = dict(original_by_fold[fold]["counts"])
        count_keys = (
            "positive_pairs", "negative_pairs", "ambiguous_pairs",
            "anchors_with_positive_and_negative", "cross_architecture_positive_pairs",
            "cross_factor_positive_pairs", "reliable_negative_pairs",
        )
        matches = {
            key: int(current_counts[key]) == int(expected[key]) == int(projected3[key])
            for key in count_keys
        }
        hp_median_match = math.isclose(
            float(base["hp_distance_median"]),
            float(original_by_fold[fold]["hp_distance_median"]),
            rel_tol=0.0, abs_tol=1e-14,
        )
        compact_pairs = [
            {
                "left": pair["left"], "right": pair["right"],
                "label": pair["label_n3_v1"],
                "cross_architecture": pair["cross_architecture"],
                "cross_factor": pair["cross_factor"],
            }
            for pair in base["pairs"]
        ]
        label_hash = joint_v1.stable_hash(compact_pairs)
        label_hash_match = label_hash == str(original_by_fold[fold]["pair_labels_sha256"])
        if not all(matches.values()) or not hp_median_match or not label_hash_match:
            raise InputInvalidError(
                f"n=3 pair-gate reproduction differs from authoritative v1 in fold {fold}"
            )
        deficits = {
            key: max(0, int(thresholds[key]) - int(current_counts[key]))
            for key in thresholds
        }
        reproduction_folds.append({
            "fold": fold,
            "train_configuration_count": len(train_rows),
            "heldout_configuration_count": len(heldout_ids),
            "counts": current_counts,
            "deficits": deficits,
            "failed_gate_items": [key for key, value in deficits.items() if value > 0],
            "fold_gate_pass": not any(deficits.values()),
            "hp_distance_median": base["hp_distance_median"],
            "hp_distance_median_matches_v1": hp_median_match,
            "pair_labels_sha256": label_hash,
            "pair_labels_sha256_matches_v1": label_hash_match,
            "all_counts_match_v1": all(matches.values()),
            "outer_heldout_labels_used": False,
        })
    reproduction = {
        "status": "REPRODUCED",
        "confidence": 0.90, "equivalence_margin": eq, "far_margin": far,
        "split_sha256": integrity["split_sha256"],
        "thresholds": thresholds,
        "folds": reproduction_folds,
        "all_folds_exactly_reproduced": all(row["all_counts_match_v1"] for row in reproduction_folds),
        "heldout_validation_used": False,
    }

    projection_rows: list[dict[str, Any]] = []
    projection_summary_rows: list[dict[str, Any]] = []
    first_all_fold_pass_n: int | None = None
    for n in map(int, config["projection"]["seed_grid"]):
        inventories: list[dict[str, Any]] = []
        for fold, base in sorted(fold_bases.items()):
            inventory = inventory_at_n(base, n, thresholds, equivalence_margin=eq, far_margin=far)
            inventories.append(inventory)
            projection_rows.append({
                "n": n, "fold": fold,
                "train_configuration_count": inventory["train_configuration_count"],
                "total_pair_count": inventory["total_pair_count"],
                "positive_pairs": inventory["positive_pairs"],
                "negative_pairs": inventory["negative_pairs"],
                "ambiguous_pairs": inventory["ambiguous_pairs"],
                "anchors_with_positive_and_negative": inventory["anchors_with_positive_and_negative"],
                "cross_architecture_positive_pairs": inventory["cross_architecture_positive_pairs"],
                "cross_factor_positive_pairs": inventory["cross_factor_positive_pairs"],
                "reliable_negative_pairs": inventory["reliable_negative_pairs"],
                "deficit_anchors_with_positive_and_negative": inventory["deficits"]["anchors_with_positive_and_negative"],
                "deficit_cross_architecture_positive_pairs": inventory["deficits"]["cross_architecture_positive_pairs"],
                "deficit_cross_factor_positive_pairs": inventory["deficits"]["cross_factor_positive_pairs"],
                "deficit_reliable_negative_pairs": inventory["deficits"]["reliable_negative_pairs"],
                "fold_gate_pass": inventory["fold_gate_pass"],
            })
        all_pass = all(bool(item["fold_gate_pass"]) for item in inventories)
        if all_pass and first_all_fold_pass_n is None:
            first_all_fold_pass_n = n
        projection_summary_rows.append({
            "n": n,
            "minimum_positive_across_folds": min(item["positive_pairs"] for item in inventories),
            "minimum_cross_architecture_positive_across_folds": min(item["cross_architecture_positive_pairs"] for item in inventories),
            "minimum_cross_factor_positive_across_folds": min(item["cross_factor_positive_pairs"] for item in inventories),
            "minimum_eligible_anchors_across_folds": min(item["anchors_with_positive_and_negative"] for item in inventories),
            "minimum_reliable_negative_across_folds": min(item["reliable_negative_pairs"] for item in inventories),
            "folds_passing": sum(bool(item["fold_gate_pass"]) for item in inventories),
            "all_five_folds_pass": all_pass,
        })
    projection_summary = {
        "projection_kind": "conditional plug-in planning projection",
        "future_mean_difference": "fixed to matched-seed mean from seeds 0,1,2",
        "future_paired_sd": "fixed to matched-seed sample SD from seeds 0,1,2",
        "seed_grid": list(config["projection"]["seed_grid"]),
        "fold_summaries": projection_summary_rows,
        "first_n_where_all_five_folds_pass": first_all_fold_pass_n,
        "all_five_folds_fail_through_n200": first_all_fold_pass_n is None,
        "probability_guarantee_claimed": False,
    }

    point_folds: list[dict[str, Any]] = []
    for fold, base in sorted(fold_bases.items()):
        inventory = point_limit_inventory(base, thresholds, equivalence_margin=eq, far_margin=far)
        point_folds.append({"fold": fold, **_compact_gate_inventory(inventory)})
    classification = classify_point_gate([bool(row["fold_gate_pass"]) for row in point_folds])
    point_output = {
        "limit": "infinite_precision_radius_zero",
        "point_equivalent_rule": "abs(mean_difference_seed3) <= 0.006",
        "point_far_rule": "mean_difference_seed3 < -0.012 or > 0.012",
        "thresholds": thresholds,
        "folds": point_folds,
        "folds_passing": sum(bool(row["fold_gate_pass"]) for row in point_folds),
        "all_five_folds_pass": all(bool(row["fold_gate_pass"]) for row in point_folds),
        "classification": classification,
    }

    pair_minimum_rows: list[dict[str, Any]] = []
    minimum_summaries: list[dict[str, Any]] = []
    for fold, base in sorted(fold_bases.items()):
        fold_rows: list[dict[str, Any]] = []
        for pair in base["pairs"]:
            minimums = minimum_seed_requirements(
                pair, n_min=3, n_max=500,
                equivalence_margin=eq, far_margin=far,
            )
            row = {
                "fold": fold,
                "left_joint_fingerprint": pair["left"],
                "right_joint_fingerprint": pair["right"],
                "mean_difference_seed3": pair["mean3"],
                "paired_sd_seed3": pair["sd3"],
                "cross_architecture": pair["cross_architecture"],
                "cross_factor": pair["cross_factor"],
                **minimums,
            }
            fold_rows.append(row)
            pair_minimum_rows.append(row)
        for category in (
            "all_positive_candidates", "cross_architecture_positive_candidates",
            "cross_factor_positive_candidates",
        ):
            minimum_summaries.append({"fold": fold, **summarize_minimum_n_rows(fold_rows, category)})
    minimum_output = {
        "search_range": [3, 500],
        "fixed_mean_and_sd_source": "matched validation seeds 0,1,2",
        "fold_category_summaries": minimum_summaries,
        "never_positive_rule": "abs(mean_difference_seed3) > 0.006",
        "finite_values_only_in_quantiles": True,
    }
    return (
        fold_bases, reproduction, point_output, projection_rows,
        projection_summary, pair_minimum_rows, minimum_output,
    )


def _git_capture(args: Sequence[str]) -> str:
    process = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if process.returncode != 0:
        raise ProtocolError(f"git {' '.join(args)} failed: {process.stdout.strip()}")
    return process.stdout


def write_git_and_environment(output: Path) -> dict[str, Any]:
    branch = _git_capture(["branch", "--show-current"]).strip()
    head = _git_capture(["rev-parse", "HEAD"]).strip()
    status = _git_capture(["status", "--short", "--branch"])
    diff = _git_capture(["diff", "--binary", "--no-ext-diff", "HEAD", "--"])
    dump_text(output / "git_status.txt", status)
    dump_text(output / "git_diff.patch", diff)
    packages: dict[str, str | None] = {}
    for distribution in ("PyYAML", "numpy", "scipy"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    environment = {
        "timestamp_utc": utc_now(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "machine": platform.machine(), "packages": packages,
        "branch": branch, "head": head,
        "dirty_tree": bool(status.splitlines()[1:]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_only": True, "gpu_used": False, "network_access_used": False,
        "gnn_training_started": False, "metric_model_training_started": False,
        "regressor_training_started": False, "gp_fitting_started": False,
        "acquisition_used": False, "test_data_accessed": False,
    }
    dump_json(output / "environment.json", environment)
    return environment


def _source_path_for_module(module: Any) -> Path | None:
    raw = getattr(module, "__file__", None)
    if not raw:
        return None
    path = Path(str(raw))
    if path.suffix in {".pyc", ".pyo"}:
        try:
            inspected = inspect.getsourcefile(module)
        except (TypeError, OSError):
            inspected = None
        if inspected:
            path = Path(inspected)
        else:
            candidate = path.with_suffix(".py")
            if candidate.is_file():
                path = candidate
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(REPO_ROOT.resolve())
    except (FileNotFoundError, ValueError):
        return None
    return resolved


def snapshot_runtime_sources(
    output: Path, mandatory_files: Sequence[str], *, modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot_root = output / "source_snapshot"
    sources: dict[str, dict[str, Any]] = {}
    for relative in mandatory_files:
        source = _relative_path(str(relative))
        if not source.is_file() or not os.access(source, os.R_OK):
            raise ProtocolError(f"mandatory source snapshot file is unreadable: {source}")
        sources[str(relative)] = {
            "source": source,
            "reasons": ["mandatory_frozen_runtime_source"],
            "module_names": [],
        }
    imported_entries: list[dict[str, Any]] = []
    for module_name, module in sorted((modules or sys.modules).items()):
        source = _source_path_for_module(module)
        if source is None:
            continue
        relative = source.relative_to(REPO_ROOT.resolve()).as_posix()
        entry = sources.setdefault(relative, {
            "source": source,
            "reasons": [],
            "module_names": [],
        })
        if "runtime_imported_local_python_module" not in entry["reasons"]:
            entry["reasons"].append("runtime_imported_local_python_module")
        entry["module_names"].append(str(module_name))
    manifest_entries: list[dict[str, Any]] = []
    for relative, metadata in sorted(sources.items()):
        source = Path(metadata["source"])
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(destination, source.read_bytes())
        source_sha = sha256_file(source)
        copied_sha = sha256_file(destination)
        if source_sha != copied_sha or source.stat().st_size != destination.stat().st_size:
            raise ProtocolError(f"source snapshot byte verification failed: {relative}")
        record = {
            "repository_relative_path": relative,
            "absolute_runtime_path": str(source),
            "snapshot_relative_path": f"source_snapshot/{relative}",
            "size_bytes": source.stat().st_size,
            "sha256": source_sha,
            "reason_included": sorted(metadata["reasons"]),
            "module_names": sorted(set(metadata["module_names"])),
        }
        manifest_entries.append(record)
        if "runtime_imported_local_python_module" in metadata["reasons"]:
            imported_entries.append(record)
    imported_payload = {
        "format_version": FORMAT_VERSION,
        "repository_root": str(REPO_ROOT.resolve()),
        "module_count": len(imported_entries),
        "modules": imported_entries,
    }
    manifest_root = stable_hash(manifest_entries)
    manifest_payload = {
        "format_version": FORMAT_VERSION,
        "repository_root": str(REPO_ROOT.resolve()),
        "entry_count": len(manifest_entries),
        "manifest_root_sha256": manifest_root,
        "entries": manifest_entries,
    }
    dump_json(snapshot_root / "imported_local_modules.json", imported_payload)
    dump_json(snapshot_root / "sha256_manifest.json", manifest_payload)
    return manifest_payload


def verify_source_snapshot(output: Path) -> dict[str, Any]:
    path = output / "source_snapshot/sha256_manifest.json"
    if not path.is_file():
        raise ProtocolError("source snapshot SHA-256 manifest is missing")
    payload = _read_json(path)
    entries = payload.get("entries", [])
    if stable_hash(entries) != payload.get("manifest_root_sha256"):
        raise ProtocolError("source snapshot manifest-root hash mismatch")
    for entry in entries:
        copied = output / str(entry["snapshot_relative_path"])
        if not copied.is_file():
            raise ProtocolError(f"source snapshot file missing: {copied}")
        if copied.stat().st_size != int(entry["size_bytes"]) or sha256_file(copied) != str(entry["sha256"]):
            raise ProtocolError(f"source snapshot hash cannot be reproduced: {copied}")
    return payload


def _format_cell(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).replace("|", "\\|")


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    materialized = list(rows) or [{key: "UNAVAILABLE" for key, _ in columns}]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_cell(row.get(key)) for key, _ in columns) + " |"
        for row in materialized
    ]
    return "\n".join([header, separator, *body])


def render_final_report(context: Mapping[str, Any]) -> str:
    classification = str(context.get("classification", "RUN_FAILED"))
    integrity = dict(context.get("integrity") or {})
    inventory = dict(context.get("inventory") or {})
    reproduction = dict(context.get("reproduction") or {})
    point = dict(context.get("point") or {})
    projection_rows = list(context.get("projection_rows") or [])
    projection_summary = dict(context.get("projection_summary") or {})
    minimum_summary = dict(context.get("minimum_summary") or {})
    calibration_inventory = dict(context.get("calibration_inventory") or {})
    calibration_rows = list(context.get("calibration_rows") or [])
    transitions = dict(context.get("calibration_transitions") or {})
    cost_rows = list(context.get("cost_rows") or [])
    priority_rows = list(context.get("priority_rows") or [])
    environment = dict(context.get("environment") or {})
    first_n = projection_summary.get("first_n_where_all_five_folds_pass")
    point_passes = int(point.get("folds_passing", 0))
    calibration_status = calibration_inventory.get("calibration_status", "UNAVAILABLE")
    archive_path = str(context.get("archive_path", "UNAVAILABLE"))
    result_path = str(context.get("result_path", "UNAVAILABLE"))
    archive_sha_description = (
        "由一键脚本在归档完成后计算；权威值位于同名 `.tar.gz.sha256` sidecar。"
        "由于归档不能稳定地在自身内容中嵌入自身哈希，本报告不伪造自引用值。"
    )
    cost_inline = ", ".join(
        f"n={row.get('target_n')}: {row.get('additional_gnn_runs')}"
        for row in cost_rows if row.get("status") == "AVAILABLE"
    ) or "不可安全计算"
    if classification == "PAIR_POWER_PRECISION_LIMITED":
        bottleneck = "seed precision（五折在无限精度点极限下全部通过）"
    elif classification == "PAIR_POWER_SUPPORT_LIMITED":
        bottleneck = "pair support（五折在无限精度点极限下全部失败）"
    elif classification == "PAIR_POWER_MIXED_LIMITED":
        bottleneck = "seed precision 与 pair support 混合限制"
    elif classification == "PAIR_POWER_INPUT_INVALID":
        bottleneck = "输入身份无效，科学投影已停止"
    else:
        bottleneck = "运行失败，未形成科学结论"

    lines: list[str] = [
        "# Joint configuration hard-triplet Pair-Power Projection Audit v1",
        "",
        "## 1. Executive conclusion",
        "",
        f"本轮冻结分类为 `{classification}`。当前瓶颈判定：{bottleneck}。"
        "该判定只取决于五个 outer-train fold 的 infinite-precision point-limit gate，"
        "不会使用 30-seed calibration 改票，也不会改写既有 `JOINT_METRIC_UNDERPOWERED`。",
        "",
        f"无限精度下通过的 folds 为 {point_passes}/5。plug-in projection 首次五折同时通过的 n 为 "
        f"`{first_n if first_n is not None else 'NONE_THROUGH_200'}`。"
        + ("固定网格到 n=200 仍未全部通过。" if first_n is None else "该 n 只是条件规划投影，不是真实未来保证。"),
        "",
        f"全部 336 配置补齐到共同 prefix 的新增 GNN 任务数摘要：{cost_inline}。"
        f"30-seed calibration 状态为 `{calibration_status}`。本审计新增 GNN 训练为 0，"
        "metric encoder、regressor、GP、acquisition 运行均为 0，test 数据与 test 指标访问为 0。",
        "",
        "本审计只回答 hard statistical-equivalence pair gate 的规划问题。它不是 metric model 训练，"
        "不是性能预测实验，也不授权补 seed。所有 future mean 与 future paired SD 都固定为前三个"
        "validation training seeds 的 plug-in 值，因此以下 n 只表示在该条件假设下区间会怎样收缩。",
        "",
        "## 2. Input integrity",
        "",
        f"权威 v1 结果目录：`{integrity.get('result_root', 'UNAVAILABLE')}`；输入归档："
        f"`{integrity.get('archive', 'UNAVAILABLE')}`。归档 SHA-256 为 "
        f"`{integrity.get('archive_sha256', 'UNAVAILABLE')}`；v1 config SHA-256 为 "
        f"`{integrity.get('config_sha256', 'UNAVAILABLE')}`；source SHA-256 为 "
        f"`{integrity.get('source_sha256', 'UNAVAILABLE')}`；logical split SHA-256 为 "
        f"`{integrity.get('split_sha256', 'UNAVAILABLE')}`。",
        "",
        f"重建 inventory：joint configurations={inventory.get('unique_joint_configurations', 0)}，"
        f"architectures={inventory.get('unique_architectures', 0)}，pools={inventory.get('unique_pools', 0)}。"
        f"训练 seed 完整性为 `{inventory.get('complete_training_seeds', [])}`，target 为 "
        f"`{inventory.get('target', 'UNAVAILABLE')}`。30-seed calibration 配置数为 "
        f"{calibration_inventory.get('configuration_count', 0)}。",
        "",
        markdown_table([
            {
                "status": integrity.get("status", "INVALID"),
                "classification": integrity.get("classification", "UNAVAILABLE"),
                "joint": integrity.get("unique_joint_configurations", 0),
                "architecture": integrity.get("unique_architectures", 0),
                "pools": integrity.get("unique_pools", 0),
                "seeds": integrity.get("training_seeds", []),
                "test": integrity.get("test_data_accessed", False),
            }
        ], (
            ("status", "integrity"), ("classification", "v1 classification"),
            ("joint", "joint configs"), ("architecture", "architectures"),
            ("pools", "pools"), ("seeds", "training seeds"), ("test", "test accessed"),
        )),
        "",
        "输入检查同时验证 COMPLETE marker、archive sidecar、state/decision 分类、三项权威哈希、"
        "canonical logical split hash、raw validation target、336/192/24 inventory，以及没有启动候选训练"
        "或读取 test。主数据仅从配置列出的 phase-B 两个初始 shard 读取，未递归扫描 results。",
        "",
        "## 3. Existing v1 failure recap",
        "",
        "下表是 n=3 的冻结 pair gate 复现。每一 fold 仅使用 outer-train configurations；"
        "同 fold 的 HP standardizer 与 HP-distance median 也只由 outer-train 拟合。"
        "positive/negative/ambiguous 完全复用 v1 matched-seed paired 90% t interval 规则。",
        "",
        markdown_table([
            {
                "fold": row.get("fold"),
                "positive": (row.get("counts") or {}).get("positive_pairs"),
                "cross_arch": (row.get("counts") or {}).get("cross_architecture_positive_pairs"),
                "cross_factor": (row.get("counts") or {}).get("cross_factor_positive_pairs"),
                "anchors": (row.get("counts") or {}).get("anchors_with_positive_and_negative"),
                "negative": (row.get("counts") or {}).get("reliable_negative_pairs"),
                "failed": row.get("failed_gate_items"),
                "pass": row.get("fold_gate_pass"),
            }
            for row in reproduction.get("folds", [])
        ], (
            ("fold", "fold"), ("positive", "positive"),
            ("cross_arch", "cross-arch positive"),
            ("cross_factor", "cross-factor positive"),
            ("anchors", "eligible anchors"), ("negative", "reliable negative"),
            ("failed", "failed items"), ("pass", "gate pass"),
        )),
        "",
        "冻结门槛分别为 eligible anchors≥30、cross-architecture positives≥50、"
        "cross-factor positives≥25、reliable negatives≥200。n=3 复现还核对每 fold 的完整 pair-label hash"
        "和 HP median；因此本审计不是从另一个 split 或另一个 cross-factor 定义重新估算。held-out"
        " validation 从未进入 pair arithmetic。",
        "",
        "## 4. Point-limit support analysis",
        "",
        "point limit 令区间半径为 0：abs(mean3)≤0.006 为 point-equivalent，mean3<-0.012 或"
        " mean3>0.012 为 point-far，其余仍 ambiguous。它回答当前 336 配置的前三 seed 均值本身是否"
        "提供足够近等 pair；它不假设未来新增架构或 HP。",
        "",
        markdown_table([
            {
                "fold": row.get("fold"), "positive": row.get("positive_pairs"),
                "cross_arch": row.get("cross_architecture_positive_pairs"),
                "cross_factor": row.get("cross_factor_positive_pairs"),
                "anchors": row.get("anchors_with_positive_and_negative"),
                "negative": row.get("reliable_negative_pairs"),
                "failed": row.get("failed_gate_items"), "pass": row.get("fold_gate_pass"),
            }
            for row in point.get("folds", [])
        ], (
            ("fold", "fold"), ("positive", "point-equivalent"),
            ("cross_arch", "cross-arch equivalent"),
            ("cross_factor", "cross-factor equivalent"),
            ("anchors", "eligible anchors"), ("negative", "point-far"),
            ("failed", "failed items"), ("pass", "gate pass"),
        )),
        "",
        "如果某 fold 在无限精度下仍失败，则仅增加同一批配置的 seeds 只能让区间逼近这些固定均值，"
        "不能创造 abs(mean3)≤0.006 的新支持；这就是 support limitation 的含义。相反，若所有 folds"
        "在 point limit 通过而有限 n 不通过，现有近等 pair 支持理论上足够，主要问题才是区间精度。"
        "部分通过对应 mixed limitation。",
        "",
        "## 5. Plug-in projection results",
        "",
        "对每个 pair 固定 mean3 和 sample paired SD3，并使用 t(0.95,n-1)×SD3/sqrt(n)。"
        "下表完整列出 n×fold 计数及门槛缺口；positive 与 negative 都只能由区间规则产生，ambiguous"
        "不会被挪入任一可靠类别。",
        "",
        markdown_table(projection_rows, (
            ("n", "n"), ("fold", "fold"),
            ("train_configuration_count", "train configs"), ("total_pair_count", "pairs"),
            ("positive_pairs", "positive"), ("negative_pairs", "negative"),
            ("ambiguous_pairs", "ambiguous"),
            ("anchors_with_positive_and_negative", "eligible anchors"),
            ("cross_architecture_positive_pairs", "cross-arch positive"),
            ("cross_factor_positive_pairs", "cross-factor positive"),
            ("reliable_negative_pairs", "reliable negative"),
            ("deficit_anchors_with_positive_and_negative", "anchor deficit"),
            ("deficit_cross_architecture_positive_pairs", "cross-arch deficit"),
            ("deficit_cross_factor_positive_pairs", "cross-factor deficit"),
            ("deficit_reliable_negative_pairs", "negative deficit"),
            ("fold_gate_pass", "pass"),
        )),
        "",
        "五折摘要：",
        "",
        markdown_table(projection_summary.get("fold_summaries", []), (
            ("n", "n"), ("minimum_positive_across_folds", "min positive"),
            ("minimum_cross_architecture_positive_across_folds", "min cross-arch"),
            ("minimum_cross_factor_positive_across_folds", "min cross-factor"),
            ("minimum_eligible_anchors_across_folds", "min anchors"),
            ("minimum_reliable_negative_across_folds", "min negative"),
            ("folds_passing", "folds passing"), ("all_five_folds_pass", "all-five pass"),
        )),
        "",
        f"首次全部通过：`{first_n if first_n is not None else 'NONE_THROUGH_200'}`。"
        "该结果是 conditional plug-in planning projection，不是未来 seed 的观测标签、成功概率或置信保证。",
        "",
        "## 6. Minimum required n distribution",
        "",
        "每个 outer-train unordered pair 在同一 fixed-mean/fixed-SD 假设下搜索整数 n=3..500。"
        "abs(mean3)>0.006 的 pair 明确标记 never-positive-under-fixed-mean；等于边界但到 n=500"
        "仍未收缩到半径 0 的 pair 不会被错误标记为数学上永不可能。quantile 只对 n≤500 的有限"
        "minimum_n_for_positive 计算。",
        "",
        markdown_table(minimum_summary.get("fold_category_summaries", []), (
            ("fold", "fold"), ("category", "candidate category"),
            ("scope_pair_count", "scope pairs"),
            ("candidate_pair_count", "candidates"),
            ("never_positive_pair_count", "never-positive"),
            ("finite_by_n500_count", "finite by 500"),
            ("not_reached_by_n500_count", "not reached"),
            ("median", "median n"), ("q75", "q75"), ("q90", "q90"),
            ("q95", "q95"), ("maximum_finite_n", "max finite n"),
        )),
        "",
        "三类分别是所有 point-positive candidates、cross-architecture candidates、cross-factor"
        " candidates。完整逐 pair 记录保存在 pair_minimum_n.csv；报告中的聚合不会隐藏 never-positive"
        "或 n=500 尚不可达的数量。",
        "",
        "## 7. 30-seed calibration",
        "",
        f"calibration 状态：`{calibration_status}`。可用时，14 个配置产生 91 个 unordered pairs；"
        "actual-prefix 在 n 时只读取 seeds 0..n-1，而 projected-prefix 永远只从 seeds 0,1,2"
        "估计 mean3 与 SD3。seeds 3..29 从未反馈到主 336 projection 参数或分类。",
        "",
        markdown_table(calibration_rows, (
            ("status", "status"), ("n", "n"), ("pair_count", "pairs"),
            ("actual_positive", "actual positive"), ("actual_negative", "actual negative"),
            ("actual_ambiguous", "actual ambiguous"),
            ("projected_positive", "projected positive"),
            ("projected_negative", "projected negative"),
            ("projected_ambiguous", "projected ambiguous"),
            ("overall_label_accuracy", "accuracy"), ("macro_f1", "macro-F1"),
            ("positive_precision", "pos precision"), ("positive_recall", "pos recall"),
            ("negative_precision", "neg precision"), ("negative_recall", "neg recall"),
            ("mean_absolute_ci_radius_error", "mean radius error"),
            ("median_absolute_ci_radius_error", "median radius error"),
            ("mean_absolute_mean_drift", "mean drift"),
            ("median_absolute_mean_drift", "median drift"),
        )),
        "",
        "Transition matrices（rows=projected，columns=actual-prefix）：",
        "",
    ]
    transition_rows: list[dict[str, Any]] = []
    for n, matrix in sorted((transitions.get("by_n") or {}).items(), key=lambda item: int(item[0])):
        for projected_label in LABELS:
            transition_rows.append({
                "n": n, "projected": projected_label,
                **{actual: (matrix.get(projected_label) or {}).get(actual, 0) for actual in LABELS},
            })
    lines.extend([
        markdown_table(transition_rows, (
            ("n", "n"), ("projected", "projected label"),
            ("positive", "actual positive"), ("negative", "actual negative"),
            ("ambiguous", "actual ambiguous"),
        )),
        "",
        "Undefined precision/recall denominators及原因逐 n 写入 calibration_metrics.json 和 CSV 的"
        " undefined_metrics 字段；macro-F1 对未定义类别采用明确的 zero-division=0 约定。"
        "本 subset 来自稳定化阶段高性能候选，不是 336 配置的随机代表样本，因此只能称为"
        "30-seed reference，不能称为真值，也不能据此修改主分类。",
        "",
        "## 8. Cost analysis",
        "",
        "成本按每个 joint configuration 已有的连续 seed prefix 计算；完整 0..29 的 14 个"
        "stabilization 配置不会重复计费。additional_gnn_runs 是新增 GNN 训练任务数，不是独立科学"
        "重复数。历史字段不足以冻结未来每任务 wall-clock 协议，因此 GPU-hour estimate unavailable，"
        "没有猜测训练时长。",
        "",
        markdown_table(cost_rows, (
            ("status", "status"), ("target_n", "target n"),
            ("config_count", "configs"), ("configs_already_complete", "already complete"),
            ("configs_needing_topup", "need top-up"),
            ("additional_gnn_runs", "additional GNN runs"),
            ("total_resulting_seed_records", "resulting records"),
            ("gpu_hour_estimate", "GPU-hour estimate"),
        )),
        "",
        "## 9. Priority candidates",
        "",
        "排名仅用于 PLANNING_ONLY。missing-cross-factor contribution 定义为：pair 在 n=3 不是"
        "positive，但在目标 n 成为 cross-factor positive；new eligible anchor 定义为配置在 n=3"
        "没有同时拥有 positive/negative，而在目标 n 获得两者。degree 对跨 folds 重复邻居去重，"
        "fold contribution 每配置每 fold 至多计一次。以下为 target n=30 的前 20 个配置。",
        "",
        markdown_table([
            row for row in priority_rows if int(row.get("target_n", -1)) == 30
        ][:20], (
            ("priority_rank_within_target", "rank"), ("joint_fingerprint", "joint fingerprint"),
            ("architecture_fingerprint", "architecture fingerprint"), ("pool_id", "pool"),
            ("existing_common_prefix_seed_count", "existing prefix"),
            ("additional_runs_to_target_n", "additional runs"),
            ("projected_positive_degree", "positive degree"),
            ("projected_cross_architecture_positive_degree", "cross-arch degree"),
            ("projected_cross_factor_positive_degree", "cross-factor degree"),
            ("number_of_outer_folds_in_which_it_appears_in_train", "train folds"),
            ("number_of_outer_folds_in_which_it_can_create_a_missing_cross_factor_edge", "missing-edge folds"),
            ("number_of_outer_folds_in_which_it_can_create_a_new_eligible_anchor", "new-anchor folds"),
        )),
        "",
        "该表未授权补 seed，未生成训练任务或命令，也不会自动选择正式 top-up 集合。"
        "排序只把当前 projection 中可能影响 pair gate 的配置置前；它不是性能排名、架构推荐或 NAS 决策。",
        "",
        "## 10. Interpretation and limitations",
        "",
        "Primary plug-in projection 固定了三 seed 的 matched mean 与 sample paired SD。三点 SD"
        "本身很不稳定，且未来 seed 加入后 paired mean 可能漂移；所以区间随 n 收缩的计算虽使用"
        "准确 t 临界值，输入参数仍只是条件性估计。projection 是资源规划工具，不是 metric 方法"
        "有效性证据、概率保证或前瞻验证。",
        "",
        "30-seed reference 只覆盖稳定化阶段选择的高性能候选，选择机制会改变 pair-gap 与 variance"
        "分布，不能把 calibration accuracy 外推为全部 336 配置的误差率。actual prefix 也只是随 n"
        "变化的参考标签，不是绝对真值。calibration 无论好坏都不会改变 point-limit 的 frozen"
        "classification。",
        "",
        "本轮没有训练 joint metric、kNN、GP、direct regressor 或任何 GNN，没有执行 VAE/decoder、"
        "NAO、qLogEI/EI/UCB/TS/KG 或其他 acquisition。没有访问 test mask、test labels、test"
        " predictions 或 test accuracy。既有 `JOINT_METRIC_UNDERPOWERED` 尚未被改写，本审计也不"
        "自动决定下一轮应增加 seed、改变配置支持，或放宽 hard-triplet 协议。",
        "",
        "## 11. Frozen classification",
        "",
        f"本轮唯一 pair-power 分类：`{classification}`。允许分类集合为 INPUT_INVALID、"
        "SUPPORT_LIMITED、PRECISION_LIMITED、MIXED_LIMITED 与 RUN_FAILED；科学分类严格由五折"
        "infinite-precision point-limit pass pattern 产生。v1 的分类保持 `JOINT_METRIC_UNDERPOWERED`。",
        "",
        "## 12. Engineering audit",
        "",
        markdown_table([{
            "branch": environment.get("branch", "UNAVAILABLE"),
            "head": environment.get("head", "UNAVAILABLE"),
            "dirty": environment.get("dirty_tree", "UNAVAILABLE"),
            "py_compile": context.get("py_compile_status", "recorded in logs/py_compile.log"),
            "pytest": context.get("pytest_status", "recorded in logs/pytest_focused.log"),
            "smoke": context.get("smoke_status", "recorded in logs/synthetic_smoke.log"),
            "elapsed": context.get("elapsed_seconds", "UNAVAILABLE"),
            "result": result_path,
            "archive": archive_path,
            "snapshot": context.get("snapshot_manifest_root", "UNAVAILABLE"),
        }], (
            ("branch", "branch"), ("head", "HEAD"), ("dirty", "dirty tree"),
            ("py_compile", "py_compile"), ("pytest", "focused pytest"),
            ("smoke", "synthetic smoke"), ("elapsed", "formal seconds"),
            ("result", "result directory"), ("archive", "archive path"),
            ("snapshot", "source snapshot manifest root"),
        )),
        "",
        f"Archive SHA-256：{archive_sha_description} 正式 archive path 为 `{archive_path}`。"
        "source snapshot 保存六个强制文件的运行时精确字节，并枚举 sys.modules 中 repository root"
        "下所有实际加载的本地 Python 源码；每项记录 runtime absolute path、repository-relative"
        "path、size、SHA-256 与 inclusion reason，打包前重新复算。",
        "",
        f"运行失败或输入无效原因（若有）：`{context.get('reason', 'NONE')}`。"
        "所有 shell 长输出分别保存在四个 logs 文件中；COMPLETE/FAILED marker 在最终 manifest"
        "刷新前写入。正式程序只读允许的冻结目录和源码，没有联网，也没有修改历史结果。",
        "",
    ])
    return "\n".join(lines)


def validate_final_report(report: str, required_sections: Sequence[str]) -> None:
    missing = [
        section for index, section in enumerate(required_sections, 1)
        if f"## {index}. {section}" not in report
    ]
    if missing:
        raise ProtocolError(f"final report lacks required sections: {missing}")
    nonwhitespace = len("".join(report.split()))
    if nonwhitespace < 3000:
        raise ProtocolError(f"final report is too short: {nonwhitespace} non-whitespace characters")
    for section_number in (3, 4, 5, 6, 7, 8, 9, 12):
        start_token = f"## {section_number}. "
        start = report.find(start_token)
        end = report.find(f"## {section_number + 1}. ", start + 1) if section_number < 12 else len(report)
        section_text = report[start:end]
        table_lines = [line for line in section_text.splitlines() if line.startswith("|")]
        if len(table_lines) < 3:
            raise ProtocolError(f"final report section {section_number} lacks a nonempty result table")


def write_manifest(output: Path) -> None:
    files: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(output).as_posix()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo", ".pt", ".pth", ".ckpt"}:
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump_json(output / "manifest.json", {
        "format_version": FORMAT_VERSION, "generated_at_utc": utc_now(),
        "file_count": len(files), "files": files,
        "manifest_root_sha256": stable_hash(files),
    })


def create_archive(output: Path, archive: Path, sidecar: Path | None = None) -> str:
    if not output.is_dir():
        raise ProtocolError(f"cannot archive missing result directory: {output}")
    if archive.exists() or (sidecar is not None and sidecar.exists()):
        raise ProtocolError("formal archive or SHA-256 sidecar already exists")
    verify_source_snapshot(output)
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    archive.parent.mkdir(parents=True, exist_ok=True)

    def filter_entry(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        name = Path(info.name).name
        if "__pycache__" in parts or name.endswith((".pyc", ".pyo", ".pt", ".pth", ".ckpt")):
            return None
        return info

    try:
        with tarfile.open(temporary, "w:gz") as handle:
            handle.add(output, arcname=output.name, recursive=True, filter=filter_entry)
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256_file(archive)
    if sidecar is not None:
        atomic_write_bytes(sidecar, f"{digest}  {archive.name}\n".encode("utf-8"))
    return digest


def initial_state(config_sha256: str, source_sha256: str) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "status": "running", "started_at_utc": utc_now(), "updated_at_utc": utc_now(),
        "config_sha256": config_sha256, "source_sha256": source_sha256,
        "authoritative_v1_split_sha256": None, "decision": None,
        "formal_data_read": False, "calibration_status": None,
        "candidate_gnn_training_started": False, "metric_model_training_started": False,
        "regressor_training_started": False, "gp_fitting_started": False,
        "acquisition_used": False, "test_data_accessed": False,
    }


def update_state(output: Path, state: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    updated = {**state, **changes, "updated_at_utc": utc_now()}
    dump_json(output / "state.json", updated)
    return updated


def _write_if_missing_json(path: Path, value: Any) -> None:
    if not path.exists():
        dump_json(path, value)


def _write_required_placeholders(output: Path, reason: str) -> None:
    status_payload = {
        "format_version": FORMAT_VERSION, "status": "UNAVAILABLE",
        "reason": reason, "rows": [{"status": "UNAVAILABLE", "reason": reason}],
    }
    for name in REQUIRED_JSON_OUTPUTS:
        _write_if_missing_json(output / name, status_payload)
    for name, fields in CSV_FIELDS.items():
        path = output / name
        if not path.exists():
            first = fields[0]
            dump_csv(path, [{first: "UNAVAILABLE"}], fields)
    for name in REQUIRED_TEXT_OUTPUTS:
        path = output / name
        if not path.exists():
            dump_text(path, "" if name.startswith("logs/") else f"UNAVAILABLE: {reason}\n")


def _fallback_environment(output: Path, reason: str) -> dict[str, Any]:
    environment = {
        "status": "UNAVAILABLE", "reason": reason,
        "branch": "UNAVAILABLE", "head": "UNAVAILABLE", "dirty_tree": "UNAVAILABLE",
        "cpu_only": True, "gpu_used": False, "network_access_used": False,
        "gnn_training_started": False, "metric_model_training_started": False,
        "regressor_training_started": False, "gp_fitting_started": False,
        "acquisition_used": False, "test_data_accessed": False,
    }
    dump_json(output / "environment.json", environment)
    if not (output / "git_status.txt").exists():
        dump_text(output / "git_status.txt", f"unavailable: {reason}\n")
    if not (output / "git_diff.patch").exists():
        dump_text(output / "git_diff.patch", "")
    return environment


def _finalize_report_and_snapshot(
    output: Path, config: Mapping[str, Any], context: dict[str, Any],
) -> dict[str, Any]:
    snapshot = snapshot_runtime_sources(
        output,
        list(config.get("source_snapshot", {}).get("mandatory_files", MANDATORY_SNAPSHOT_FILES)),
    )
    verify_source_snapshot(output)
    context["snapshot_manifest_root"] = snapshot["manifest_root_sha256"]
    context["result_path"] = str(output)
    context["archive_path"] = str(_relative_path(str(config["output"]["archive"])))
    report = render_final_report(context)
    validate_final_report(report, list(config["reporting"]["required_sections"]))
    dump_text(output / "final_report.md", report)
    return snapshot


def finalize_input_invalid(
    output: Path, config: Mapping[str, Any], state: Mapping[str, Any],
    reason: str, *, environment: Mapping[str, Any] | None = None,
) -> int:
    dump_json(output / "input_integrity.json", {
        "status": "INVALID", "classification": "PAIR_POWER_INPUT_INVALID",
        "reason": reason, "test_data_accessed": False,
    })
    _write_required_placeholders(output, reason)
    decision = {
        "classification": "PAIR_POWER_INPUT_INVALID", "reason": reason,
        "classification_source": "authoritative input identity validation",
        "calibration_changed_classification": False,
        "v1_classification_modified": False,
        "gnn_training_started": False, "test_data_accessed": False,
    }
    dump_json(output / "decision.json", decision)
    env = dict(environment or _fallback_environment(output, reason))
    context = {
        "classification": "PAIR_POWER_INPUT_INVALID", "reason": reason,
        "environment": env, "integrity": {"status": "INVALID"},
        "inventory": {}, "reproduction": {}, "point": {},
        "projection_rows": [], "projection_summary": {}, "minimum_summary": {},
        "calibration_inventory": {"calibration_status": "UNAVAILABLE"},
        "calibration_rows": [], "calibration_transitions": {},
        "cost_rows": [], "priority_rows": [],
    }
    _finalize_report_and_snapshot(output, config, context)
    update_state(
        output, state, status="complete", decision="PAIR_POWER_INPUT_INVALID",
        error=reason, formal_data_read=True,
    )
    write_manifest(output)
    return 0


def write_failure_bundle(
    output: Path, message: str, config_sha256: str, source_sha256: str,
    *, config_path: Path = DEFAULT_CONFIG,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    _write_required_placeholders(output, message)
    dump_json(output / "error.json", {
        "classification": "RUN_FAILED", "message": message,
        "timestamp_utc": utc_now(), "traceback": traceback.format_exc()
        if sys.exc_info()[0] is not None else None,
    })
    try:
        config = load_config(config_path)
    except Exception:
        config = {
            "source_snapshot": {"mandatory_files": list(MANDATORY_SNAPSHOT_FILES)},
            "reporting": {
                "required_sections": [
                    "Executive conclusion", "Input integrity", "Existing v1 failure recap",
                    "Point-limit support analysis", "Plug-in projection results",
                    "Minimum required n distribution", "30-seed calibration",
                    "Cost analysis", "Priority candidates", "Interpretation and limitations",
                    "Frozen classification", "Engineering audit",
                ]
            },
            "output": {
                "archive": "results/posthoc/joint-configuration-pair-power-projection-global4-v1.tar.gz"
            },
        }
    state_path = output / "state.json"
    try:
        state = _read_json(state_path) if state_path.is_file() else initial_state(config_sha256, source_sha256)
    except Exception:
        state = initial_state(config_sha256, source_sha256)
    try:
        environment = write_git_and_environment(output)
    except Exception as git_error:
        environment = _fallback_environment(output, f"{message}; git capture failed: {git_error}")
    decision = {
        "classification": "RUN_FAILED", "reason": message,
        "calibration_changed_classification": False,
        "v1_classification_modified": False,
        "gnn_training_started": False, "test_data_accessed": False,
    }
    dump_json(output / "decision.json", decision)
    if not (output / "frozen_protocol.json").exists():
        dump_json(output / "frozen_protocol.json", {
            "format_version": FORMAT_VERSION, "status": "failure_before_protocol_freeze",
            "config_sha256": config_sha256, "source_sha256": source_sha256,
        })
    context = {
        "classification": "RUN_FAILED", "reason": message, "environment": environment,
        "integrity": {}, "inventory": {}, "reproduction": {}, "point": {},
        "projection_rows": [], "projection_summary": {}, "minimum_summary": {},
        "calibration_inventory": {"calibration_status": "UNAVAILABLE"},
        "calibration_rows": [], "calibration_transitions": {},
        "cost_rows": [], "priority_rows": [], "result_path": str(output),
    }
    try:
        _finalize_report_and_snapshot(output, config, context)
    except Exception as snapshot_error:
        # Preserve the original failure and make the snapshot failure explicit.
        dump_json(output / "source_snapshot_error.json", {
            "original_error": message, "snapshot_error": str(snapshot_error),
        })
        raise
    update_state(
        output, state, status="failed", decision="RUN_FAILED", error=message,
        candidate_gnn_training_started=False, metric_model_training_started=False,
        test_data_accessed=False,
    )
    complete = output / "COMPLETE"
    if complete.exists():
        complete.unlink()
    (output / "FAILED").touch()
    write_manifest(output)


def _aggregate_source_sha(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def execute_formal(
    config_path: Path, output: Path, config_sha256: str, source_sha256: str,
    *, preflight_ready: bool,
) -> int:
    started = time.monotonic()
    config = load_config(config_path)
    ensure_output_available(output, preflight_ready=preflight_ready)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    state = initial_state(config_sha256, source_sha256)
    dump_json(output / "state.json", state)
    dump_json(output / "frozen_protocol.json", {
        "format_version": FORMAT_VERSION, "config": config,
        "config_path": str(config_path.resolve()), "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "reused_functions": [
            "paired_difference_interval", "classify_pair", "fit_hp_standardizer",
            "_pairwise_hp_median", "construct_statistical_pairs", "assert_pair_power",
        ],
        "primary_projection": "validation-only paired interval arithmetic",
        "candidate_gnn_training": False, "metric_model_training": False,
        "regressor_training": False, "gp_fitting": False,
        "acquisition": False, "test_data_access": False,
    })
    try:
        environment = write_git_and_environment(output)
    except Exception as error:
        environment = _fallback_environment(output, str(error))
    try:
        integrity = validate_authoritative_v1(config)
        dump_json(output / "input_integrity.json", {
            key: value for key, value in integrity.items()
            if key not in {"split_manifest", "original_pair_inventory"}
        })
        state = update_state(
            output, state, formal_data_read=True,
            authoritative_v1_split_sha256=integrity["split_sha256"],
        )
        configurations, inventory, provenance = load_main_configurations(config, integrity)
        inventory["data_provenance"] = provenance
        dump_json(output / "data_inventory.json", inventory)

        (
            fold_bases, reproduction, point_output, projection_rows,
            projection_summary, pair_minimum_rows, minimum_output,
        ) = run_primary_projection(config, configurations, integrity)
        dump_json(output / "original_pair_gate_reproduction.json", reproduction)
        dump_json(output / "point_limit_pair_inventory.json", point_output)
        dump_csv(
            output / "projection_by_n_and_fold.csv", projection_rows,
            CSV_FIELDS["projection_by_n_and_fold.csv"],
        )
        dump_json(output / "projection_summary.json", projection_summary)
        dump_csv(output / "pair_minimum_n.csv", pair_minimum_rows, CSV_FIELDS["pair_minimum_n.csv"])
        dump_json(output / "minimum_n_summary.json", minimum_output)

        calibration_subset: dict[str, dict[str, Any]] = {}
        try:
            calibration_subset, calibration_inventory = load_calibration_subset(config, configurations)
            calibration_rows, calibration_transitions, calibration_metrics = run_calibration_replay(
                calibration_subset, config["calibration"]["prefix_grid"],
            )
        except CalibrationUnavailable as error:
            calibration_inventory = {
                "calibration_status": "UNAVAILABLE", "reason": str(error),
                "configuration_count": 0, "used_for_primary_projection_parameters": False,
                "may_change_primary_classification": False, "test_data_accessed": False,
            }
            calibration_rows = [{
                "status": "UNAVAILABLE", "n": "UNAVAILABLE", "pair_count": 0,
                "undefined_metrics": [{"reason": str(error)}],
            }]
            calibration_transitions = {"status": "UNAVAILABLE", "reason": str(error), "by_n": {}}
            calibration_metrics = {
                "calibration_status": "UNAVAILABLE", "reason": str(error),
                "main_projection_continued": True,
            }
        dump_json(output / "calibration_inventory.json", calibration_inventory)
        dump_csv(
            output / "calibration_prefix_results.csv", calibration_rows,
            CSV_FIELDS["calibration_prefix_results.csv"],
        )
        dump_json(output / "calibration_transition_matrices.json", calibration_transitions)
        dump_json(output / "calibration_metrics.json", calibration_metrics)
        state = update_state(output, state, calibration_status=calibration_inventory["calibration_status"])

        coverage: dict[str, set[int]] = {
            str(row["joint_fingerprint"]): {0, 1, 2} for row in configurations
        }
        if calibration_inventory["calibration_status"] == "AVAILABLE":
            for joint, row in calibration_subset.items():
                coverage[joint] = set(map(int, row["seeds"]))
            cost_rows = full_topup_cost(coverage, config["cost"]["target_seed_counts"])
            priority_rows = build_priority_rows(
                configurations, fold_bases, coverage,
                config["priority"]["target_seed_counts"], _pair_thresholds(config),
            )
        else:
            unavailable_reason = "SPEC_UNDERDEFINED: trusted stabilization seed coverage unavailable"
            cost_rows = [{
                "status": "SPEC_UNDERDEFINED", "target_n": target,
                "config_count": len(configurations), "gpu_hour_estimate": "unavailable",
            } for target in config["cost"]["target_seed_counts"]]
            priority_rows = [{
                "status": "SPEC_UNDERDEFINED", "target_n": target,
                "priority_rank_within_target": 0, "joint_fingerprint": "UNAVAILABLE",
                "architecture_fingerprint": "UNAVAILABLE", "pool_id": "UNAVAILABLE",
                "existing_common_prefix_seed_count": 3,
                "additional_runs_to_target_n": "UNAVAILABLE",
                "projected_positive_degree": "UNAVAILABLE",
                "projected_cross_architecture_positive_degree": "UNAVAILABLE",
                "projected_cross_factor_positive_degree": "UNAVAILABLE",
                "number_of_outer_folds_in_which_it_appears_in_train": "UNAVAILABLE",
                "number_of_outer_folds_in_which_it_can_create_a_missing_cross_factor_edge": "UNAVAILABLE",
                "number_of_outer_folds_in_which_it_can_create_a_new_eligible_anchor": "UNAVAILABLE",
                "reason": unavailable_reason,
            } for target in config["priority"]["target_seed_counts"]]
        coverage_rows = []
        by_joint = {str(row["joint_fingerprint"]): row for row in configurations}
        for joint in sorted(coverage):
            seeds = sorted(coverage[joint])
            coverage_rows.append({
                "joint_fingerprint": joint,
                "architecture_fingerprint": by_joint[joint]["architecture_fingerprint"],
                "pool_id": by_joint[joint]["pool_id"],
                "observed_validation_seeds": seeds,
                "existing_common_prefix_seed_count": existing_common_prefix_seed_count(seeds),
                "maximum_observed_seed": max(seeds),
                "coverage_status": "TRUSTED",
            })
        dump_csv(
            output / "seed_coverage_by_config.csv", coverage_rows,
            CSV_FIELDS["seed_coverage_by_config.csv"],
        )
        dump_csv(output / "full_topup_cost.csv", cost_rows, CSV_FIELDS["full_topup_cost.csv"])
        dump_csv(
            output / "topup_candidate_priority.csv", priority_rows,
            CSV_FIELDS["topup_candidate_priority.csv"],
        )

        classification = str(point_output["classification"])
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise ProtocolError(f"unfrozen classification produced: {classification}")
        decision = {
            "classification": classification,
            "classification_source": "infinite_precision_point_limit_five_fold_gate_only",
            "point_limit_fold_passes": [
                bool(row["fold_gate_pass"]) for row in point_output["folds"]
            ],
            "first_n_where_all_five_folds_pass": projection_summary["first_n_where_all_five_folds_pass"],
            "calibration_status": calibration_inventory["calibration_status"],
            "calibration_changed_classification": False,
            "v1_classification": "JOINT_METRIC_UNDERPOWERED",
            "v1_classification_modified": False,
            "topup_authorized": False, "training_commands_generated": False,
            "gnn_training_started": False, "metric_model_training_started": False,
            "test_data_accessed": False,
        }
        dump_json(output / "decision.json", decision)
        elapsed = time.monotonic() - started
        context = {
            "classification": classification, "reason": "NONE",
            "integrity": integrity, "inventory": inventory,
            "reproduction": reproduction, "point": point_output,
            "projection_rows": projection_rows, "projection_summary": projection_summary,
            "minimum_summary": minimum_output,
            "calibration_inventory": calibration_inventory,
            "calibration_rows": calibration_rows,
            "calibration_transitions": calibration_transitions,
            "cost_rows": cost_rows, "priority_rows": priority_rows,
            "environment": environment, "elapsed_seconds": elapsed,
            "py_compile_status": "PASS (see logs/py_compile.log)",
            "pytest_status": "PASS (see logs/pytest_focused.log)",
            "smoke_status": "PASS (see logs/synthetic_smoke.log)",
        }
        snapshot = _finalize_report_and_snapshot(output, config, context)
        state = update_state(
            output, state, status="complete", decision=classification,
            calibration_status=calibration_inventory["calibration_status"],
            formal_elapsed_seconds=elapsed,
            source_snapshot_manifest_root=snapshot["manifest_root_sha256"],
            candidate_gnn_training_started=False, metric_model_training_started=False,
            regressor_training_started=False, gp_fitting_started=False,
            acquisition_used=False, test_data_accessed=False,
        )
        write_manifest(output)
        return 0
    except (InputInvalidError, joint_v1.ProtocolError) as error:
        return finalize_input_invalid(
            output, config, state, str(error), environment=environment,
        )


def dependency_inventory() -> dict[str, bool]:
    output: dict[str, bool] = {}
    for name in ("yaml", "numpy", "scipy"):
        try:
            __import__(name)
            output[name] = True
        except ImportError:
            output[name] = False
    return output


def dry_run(config_path: Path, output: Path) -> int:
    config = load_config(config_path)
    payload = {
        "protocol_version": config["protocol_version"],
        "mode": "dry-run", "cpu_only": True,
        "formal_data_read": False, "formal_result_root_created": False,
        "candidate_gnn_training_started": False, "metric_model_training_started": False,
        "regressor_training_started": False, "gp_fitting_started": False,
        "acquisition_used": False, "test_data_accessed": False,
        "configured_output": str(output), "configured_output_exists": output.exists(),
        "dependencies": dependency_inventory(),
        "formal_actions": [
            "validate authoritative v1 identities", "reconstruct validation-only configurations",
            "reproduce n=3 pair gate", "project paired intervals", "evaluate point limit",
            "replay optional 30-seed reference", "audit top-up cost and planning-only priority",
            "snapshot runtime sources", "write report and bundle",
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def synthetic_smoke() -> int:
    positive = classify_projected_pair(0.001, 0.0, 3)["label"] == "positive"
    negative = classify_projected_pair(0.020, 0.0, 3)["label"] == "negative"
    ambiguous = classify_projected_pair(0.008, 0.0, 3)["label"] == "ambiguous"
    point_boundaries = (
        classify_point_limit(0.006) == "positive"
        and classify_point_limit(-0.006) == "positive"
        and classify_point_limit(0.012) == "ambiguous"
        and classify_point_limit(0.0120001) == "negative"
    )
    classifications = {
        "synthetic_support_limited": classify_point_gate([False] * 5) == "PAIR_POWER_SUPPORT_LIMITED",
        "synthetic_precision_limited": classify_point_gate([True] * 5) == "PAIR_POWER_PRECISION_LIMITED",
        "synthetic_mixed_limited": classify_point_gate([True, False, True, False, False]) == "PAIR_POWER_MIXED_LIMITED",
    }
    payload = {
        "passed": positive and negative and ambiguous and point_boundaries and all(classifications.values()),
        "matched_seed_positive": positive, "matched_seed_negative": negative,
        "matched_seed_ambiguous": ambiguous, "point_boundaries": point_boundaries,
        **classifications,
        "model_training_started": False, "formal_data_read": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-sha256")
    parser.add_argument("--source-sha256")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--formal", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--synthetic-smoke", action="store_true")
    modes.add_argument("--refresh-manifest", action="store_true")
    modes.add_argument("--write-failure-bundle", metavar="MESSAGE")
    parser.add_argument("--preflight-ready", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    output = args.output.resolve()
    if args.dry_run:
        return dry_run(config_path, output)
    if args.synthetic_smoke:
        return synthetic_smoke()
    config_sha256 = args.config_sha256 or sha256_file(config_path)
    source_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/run_joint_pair_power_projection_audit.sh",
        REPO_ROOT / "tests/test_joint_pair_power_projection_audit.py",
        REPO_ROOT / "analyse/joint_configuration_metric_gate.py",
        REPO_ROOT / "analyse/performance_metric_quick_gate.py",
    ]
    source_sha256 = args.source_sha256 or _aggregate_source_sha(source_paths)
    if args.refresh_manifest:
        write_manifest(output)
        return 0
    if args.write_failure_bundle is not None:
        write_failure_bundle(
            output, str(args.write_failure_bundle), config_sha256, source_sha256,
            config_path=config_path,
        )
        return 0
    if args.formal:
        try:
            return execute_formal(
                config_path, output, config_sha256, source_sha256,
                preflight_ready=bool(args.preflight_ready),
            )
        except Exception as error:
            try:
                write_failure_bundle(
                    output, str(error), config_sha256, source_sha256,
                    config_path=config_path,
                )
            except Exception as bundle_error:
                print(f"formal audit failed: {error}", file=sys.stderr)
                print(f"failure bundle also failed: {bundle_error}", file=sys.stderr)
            raise
    raise ProtocolError("no execution mode selected")


if __name__ == "__main__":
    raise SystemExit(main())
