#!/usr/bin/env python3
"""Read-only cross-factor locality diagnosis for the frozen soft-metric v2 gate.

The formal path deterministically replays only the 15 archived metric encoders.
It never trains a candidate GNN, regressor, GP, decoder, or acquisition model and
never reads a test-labelled field.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FORMAT_VERSION = "joint_configuration_cross_factor_locality_diagnosis_global4_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/joint_configuration_cross_factor_locality_diagnosis_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/posthoc/joint-configuration-cross-factor-locality-diagnosis-global4-v1"
METRIC_SEEDS = (0, 1, 2)
DELTA_CLOSES = (0.006, 0.012)
TOP_K_VALUES = (5, 10)
ALLOWED_DECISIONS = {
    "CROSS_FACTOR_DIAGNOSIS_COMPLETE",
    "REPLAY_INPUT_IDENTITY_MISMATCH",
    "REPLAY_IDENTITY_MISMATCH",
    "RUN_FAILED",
}
FROZEN_INPUT_IDENTITY = {
    "classification": "SOFT_JOINT_METRIC_REJECTED",
    "archive_sha256": "f5a0a73cb45098994afa3d60de44eb4c54c010d6814d963e3d4c3ba69c219f74",
    "config_sha256": "0b97af8a13d270e80fe8cc7b9a4beaf2616ff624c01faf5832fdebbeae25aadd",
    "source_sha256": "d042a158a2c006baecc7c9d1d50dd42683f7b4f4c4dbdfed0c2f2c8bcc3a3aea",
    "source_snapshot_manifest_root": "7125d9d7eececca53fcd00c2d578fb685c1b0d2156de62f4de08b362ffe10273",
    "split_sha256": "b80b1a211a05285121cafe4dabd4c67baddea93606c1913e777fcd34dd79a2ec",
    "unique_joint_configurations": 336, "unique_architectures": 192,
    "pool_count": 24, "seed_record_count": 1008, "training_seeds": [0, 1, 2],
}
FROZEN_SOFT_METRIC_PARAMETERS = {
    "delta": 0.012, "embedding_dim": 8, "metric_seeds": [0, 1, 2],
    "outer_folds": 5,
    "hp_features": ["log10(lr)", "dropout", "log2(hidden_dim)", "log10(l2)"],
}
REPORT_SECTIONS = (
    "Executive conclusion", "Input identity", "Replay identity", "Original G3 recap",
    "Bottom-20% performance-gap audit", "Absolute gap-bin distance analysis",
    "Top-5/top-10 neighbor composition", "Cross-factor precision/recall/hit",
    "Relevant-rank analysis", "Distance AUC and Cliff's delta",
    "HP-near vs HP-far locality", "Metric-seed stability", "Posthoc interpretation",
    "What this does NOT change", "Recommended next scientific question", "Engineering audit",
)
REQUIRED_SOURCE_SNAPSHOT_PATHS = (
    "configs/joint_configuration_cross_factor_locality_diagnosis_v1.yaml",
    "analyse/joint_configuration_cross_factor_locality_diagnosis.py",
    "scripts/run_joint_configuration_cross_factor_locality_diagnosis.sh",
    "tests/test_joint_configuration_cross_factor_locality_diagnosis.py",
    "configs/joint_configuration_soft_metric_gate_v2.yaml",
    "analyse/joint_configuration_soft_metric_gate.py",
    "analyse/joint_configuration_metric_gate.py",
    "analyse/performance_metric_quick_gate.py",
    "analyse/joint_pair_power_projection_audit.py",
    "scripts/run_joint_configuration_soft_metric_gate.sh",
    "tests/test_joint_configuration_soft_metric_gate.py",
)
REQUIRED_OUTPUTS = (
    "state.json", "frozen_protocol.json", "input_integrity.json", "replay_identity.json",
    "g3_quantile_alignment.csv", "cross_factor_gap_bins.csv",
    "cross_factor_gap_bin_summary.json", "neighbor_composition_per_query.csv",
    "neighbor_composition_summary.json", "cross_factor_retrieval_per_query.csv",
    "cross_factor_retrieval_metrics.json", "cross_factor_rank_retrieval.json",
    "distance_discrimination_per_query.csv", "distance_discrimination.json",
    "hp_near_far_locality.json", "metric_seed_stability.json", "decision.json",
    "final_report.md", "environment.json", "git_status.txt", "git_diff.patch",
    "source_snapshot/imported_local_modules.json", "source_snapshot/sha256_manifest.json",
    "manifest.json", "logs/py_compile.log", "logs/pytest_focused.log",
    "logs/synthetic_smoke.log", "logs/run.log",
)


class ProtocolError(RuntimeError):
    """The frozen diagnosis protocol, safety boundary, or numerical contract failed."""


class ReplayInputIdentityMismatch(ProtocolError):
    """The completed soft-v2 input is not the frozen authoritative experiment."""


class ReplayIdentityMismatch(ProtocolError):
    """The deterministic encoder replay does not reproduce archived geometry."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
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
    atomic_write_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")


def dump_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def dump_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None,
) -> None:
    materialized = list(rows)
    if fields is None:
        fields = sorted({key for row in materialized for key in row}) or ("status",)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _require(condition: bool, message: str, *, input_identity: bool = False) -> None:
    if condition:
        return
    if input_identity:
        raise ReplayInputIdentityMismatch(message)
    raise ProtocolError(message)


def _repo_path(raw: str) -> Path:
    path = (REPO_ROOT / str(raw)).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ProtocolError(f"configured path escapes repository: {raw}") from error
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ProtocolError("PyYAML is required for the frozen diagnosis config") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "diagnosis config must be a mapping")
    validate_frozen_protocol(value)
    return value


def validate_frozen_protocol(config: Mapping[str, Any]) -> None:
    replay = config["replay"]
    identity_gate = replay["identity_gate"]
    soft = config["frozen_soft_metric_contract"]
    checks = {
        "version": config.get("protocol_version") == FORMAT_VERSION,
        "seeds": list(replay["metric_model_seeds"]) == [0, 1, 2],
        "cpu": config["cpu_only"] is True and replay["device"] == "cpu",
        "deterministic": replay["deterministic"] is True,
        "tolerance": float(identity_gate["absolute_tolerance"]) == 1.0e-8,
        "mean_distance": replay["distance_seed_aggregation"]
        == "arithmetic_mean_of_seed_distance_matrices",
        "delta": float(soft["delta"]) == 0.012,
        "embedding": int(soft["embedding_dim"]) == 8,
        "no_models": replay["allowed_training"] == "frozen_soft_metric_encoder_only"
        and set(replay["forbidden_replays"]) == {
            "B0", "B1", "B2", "B3", "B4", "M2_GP", "M3", "M4",
            "best_of_4", "direct_joint_regressor",
        }
        and config["data"]["candidate_gnn_training"] == "forbidden"
        and config["data"]["stabilization_data_access"] == "forbidden",
        "bins": [row["upper"] for row in config["absolute_gap_bins"]] == [
            0.006, 0.012, 0.025, 0.050, None,
        ],
        "topk": list(config["neighbor_composition"]["k"]) == [5, 10],
        "bootstrap": int(config["bootstrap"]["replicates"]) == 2000
        and int(config["bootstrap"]["seed"]) == 0
        and config["bootstrap"]["unit"] == "heldout_query",
        "decisions": set(config["decision"]["allowed"]) == ALLOWED_DECISIONS,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    _require(not failed, f"frozen diagnosis protocol changed: {failed}")


def nested_test_key_hits(value: Any, prefix: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if "test" in str(key).lower():
                hits.append(child_path)
            hits.extend(nested_test_key_hits(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(nested_test_key_hits(child, f"{prefix}[{index}]"))
    return hits


def reject_test_fields(value: Any, context: str = "record") -> None:
    hits = nested_test_key_hits(value)
    if hits:
        raise ReplayInputIdentityMismatch(
            f"test field rejected at loader boundary in {context}: {hits[:5]}"
        )


assert_no_test_fields = reject_test_fields


def validate_input_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    failed = [
        key for key, expected in FROZEN_INPUT_IDENTITY.items()
        if payload.get(key) != expected
    ]
    if failed:
        raise ReplayInputIdentityMismatch(f"frozen soft-v2 identity mismatch: {failed}")
    return dict(payload)


def extract_replay_seed_values(values: Mapping[int, Any]) -> list[float]:
    try:
        return [float(values[seed]) for seed in METRIC_SEEDS]
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("replay requires validation seeds exactly 0,1,2") from error


def average_metric_seed_distance_matrices(matrices: Sequence[Any]) -> Any:
    import numpy as np
    _require(len(matrices) == 3, "distance aggregation requires exactly three metric seeds")
    arrays = [np.asarray(matrix, dtype=np.float64) for matrix in matrices]
    _require(len({array.shape for array in arrays}) == 1, "metric seed matrix shapes differ")
    return np.mean(np.stack(arrays, axis=0), axis=0)


def replay_runtime_contract() -> dict[str, Any]:
    return {
        "device": "cpu", "cuda_visible_devices": "", "deterministic": True,
        "torch_num_threads": 1, "metric_seeds": [0, 1, 2],
        "candidate_gnn_training": False, "regressor_training": False,
        "gp_fitting": False, "acquisition": False,
    }


def validate_frozen_soft_metric_parameters(parameters: Mapping[str, Any]) -> None:
    _require(dict(parameters) == FROZEN_SOFT_METRIC_PARAMETERS,
             "frozen soft-metric parameters changed")


def assert_replay_scope(components: Sequence[str]) -> None:
    _require(list(components) == ["metric_encoder"],
             "replay scope permits only the frozen metric encoder")


def assert_replay_identity(
    replayed: Mapping[str, Any], archived: Mapping[str, Any], *, atol: float = 1.0e-8,
) -> dict[str, Any]:
    def compare(left: Any, right: Any, path: str) -> list[str]:
        if isinstance(right, Mapping):
            if not isinstance(left, Mapping) or set(left) != set(right):
                return [path]
            return [
                mismatch
                for key in sorted(right)
                for mismatch in compare(left[key], right[key], f"{path}.{key}")
            ]
        if isinstance(right, list):
            if not isinstance(left, list) or len(left) != len(right):
                return [path]
            return [
                mismatch
                for index, expected in enumerate(right)
                for mismatch in compare(left[index], expected, f"{path}[{index}]")
            ]
        if isinstance(right, (int, float)) and not isinstance(right, bool):
            try:
                return [] if abs(float(left) - float(right)) <= float(atol) else [path]
            except (TypeError, ValueError):
                return [path]
        return [] if left == right else [path]

    mismatches = compare(replayed, archived, "$")
    if mismatches:
        raise ReplayIdentityMismatch(f"deterministic replay mismatch: {mismatches[:10]}")
    return {"passed": True, "absolute_tolerance": float(atol), "mismatches": []}


def gap_bin_label(gap: float) -> str:
    value = float(gap)
    _require(math.isfinite(value) and value >= 0.0, "performance gap must be finite and nonnegative")
    if value <= 0.006:
        return "bin1_[0,0.006]"
    if value <= 0.012:
        return "bin2_(0.006,0.012]"
    if value <= 0.025:
        return "bin3_(0.012,0.025]"
    if value <= 0.050:
        return "bin4_(0.025,0.050]"
    return "bin5_(0.050,inf)"


GAP_BIN_ORDER = (
    "bin1_[0,0.006]", "bin2_(0.006,0.012]", "bin3_(0.012,0.025]",
    "bin4_(0.025,0.050]", "bin5_(0.050,inf)",
)


def stable_top_k(distances: Sequence[float], fingerprints: Sequence[str], k: int) -> list[int]:
    _require(len(distances) == len(fingerprints), "distance/fingerprint length mismatch")
    _require(0 < int(k) <= len(distances), "invalid top-k request")
    return sorted(
        range(len(distances)), key=lambda index: (float(distances[index]), str(fingerprints[index])),
    )[: int(k)]


def stable_distance_order(distances: Sequence[float], fingerprints: Sequence[str]) -> Any:
    import numpy as np
    return np.asarray(stable_top_k(distances, fingerprints, len(distances)), dtype=np.int64)


def stable_topk_fingerprints(
    distances: Sequence[float], fingerprints: Sequence[str], *, k: int,
) -> list[str]:
    return [str(fingerprints[index]) for index in stable_top_k(distances, fingerprints, k)]


def assign_absolute_gap_bin(gap: float) -> str:
    return {
        "bin1_[0,0.006]": "gap_le_0.006",
        "bin2_(0.006,0.012]": "gap_0.006_0.012",
        "bin3_(0.012,0.025]": "gap_0.012_0.025",
        "bin4_(0.025,0.050]": "gap_0.025_0.050",
        "bin5_(0.050,inf)": "gap_gt_0.050",
    }[gap_bin_label(gap)]


def architecture_difference_rate(
    heldout_architectures: Sequence[str], train_architectures: Sequence[str],
) -> float:
    total = len(heldout_architectures) * len(train_architectures)
    _require(total > 0, "architecture rate requires nonempty heldout/train identities")
    return sum(
        str(left) != str(right)
        for left in heldout_architectures for right in train_architectures
    ) / total


def assert_architecture_disjoint(
    heldout_architectures: Sequence[str], train_architectures: Sequence[str],
) -> None:
    _require(not set(map(str, heldout_architectures)) & set(map(str, train_architectures)),
             "architecture fingerprint crosses outer fold")


def fit_hp_far_context(train_hp: Any) -> dict[str, Any]:
    import numpy as np
    values = np.asarray(train_hp, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 4 and len(values) >= 2,
             "outer-train HP matrix must be n x 4")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    standardized = (values - mean) / scale
    pair_distances = [
        float(np.linalg.norm(standardized[left] - standardized[right]))
        for left, right in itertools.combinations(range(len(values)), 2)
    ]
    return {
        "mean": mean.tolist(), "scale": scale.tolist(),
        "train_pair_median_distance": float(np.median(np.asarray(pair_distances))),
        "fit_scope": "outer_train_only", "fit_row_count": len(values),
    }


def hp_far_matrix(query_hp: Any, train_hp: Any, context: Mapping[str, Any]) -> Any:
    import numpy as np
    query = np.asarray(query_hp, dtype=np.float64)
    train = np.asarray(train_hp, dtype=np.float64)
    mean = np.asarray(context["mean"], dtype=np.float64)
    scale = np.asarray(context["scale"], dtype=np.float64)
    distances = np.linalg.norm(
        ((query - mean) / scale)[:, None, :] - ((train - mean) / scale)[None, :, :], axis=2,
    )
    return distances > float(context["train_pair_median_distance"])


def cross_factor_relevant_mask(
    *, performance_gaps: Any, hp_far: Any, architecture_different: Any,
    delta_close: float,
) -> Any:
    import numpy as np
    gaps = np.asarray(performance_gaps, dtype=np.float64)
    hp = np.asarray(hp_far, dtype=bool)
    architecture = np.asarray(architecture_different, dtype=bool)
    _require(gaps.shape == hp.shape == architecture.shape, "relevance arrays differ in shape")
    return architecture & hp & (gaps <= float(delta_close))


def retrieval_metrics_for_query(
    *, distances: Any, reference_fingerprints: Sequence[str], performance_gaps: Any,
    hp_far: Any, architecture_different: Any, delta_close: float, k: int,
) -> dict[str, Any]:
    import numpy as np
    distance = np.asarray(distances, dtype=np.float64)
    relevant = cross_factor_relevant_mask(
        performance_gaps=performance_gaps, hp_far=hp_far,
        architecture_different=architecture_different, delta_close=delta_close,
    )
    order = stable_top_k(distance, reference_fingerprints, int(k))
    relevant_count = int(np.sum(relevant))
    retrieved = int(np.sum(relevant[np.asarray(order, dtype=int)]))
    prevalence = relevant_count / len(distance)
    precision = retrieved / int(k)
    eligible = relevant_count > 0
    return {
        "k": int(k), "delta_close": float(delta_close), "eligible": eligible,
        "relevant_count": relevant_count, "relevant_retrieved": retrieved,
        "precision": precision, "recall": retrieved / relevant_count if eligible else None,
        "hit": int(retrieved > 0) if eligible else None,
        "conditional_hit": int(retrieved > 0) if eligible else None,
        "base_prevalence": prevalence,
        "precision_lift": precision / prevalence if prevalence > 0 else None,
        "top_k_fingerprints": [str(reference_fingerprints[index]) for index in order],
    }


def summarize_retrieval_queries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np
    materialized = list(rows)
    eligible = [row for row in materialized if bool(row["eligible"])]
    _require(bool(materialized), "retrieval summary requires query rows")
    return {
        "query_count": len(materialized), "eligible_query_count": len(eligible),
        "no_opportunity_query_count": len(materialized) - len(eligible),
        "precision_mean": float(np.mean([float(row["precision"]) for row in materialized])),
        "conditional_recall_mean": float(np.mean([float(row["recall"]) for row in eligible])) if eligible else None,
        "conditional_hit_rate": float(np.mean([float(row["hit"]) for row in eligible])) if eligible else None,
        "base_prevalence": float(np.mean([float(row["base_prevalence"]) for row in materialized])),
    }


def relevant_rank_metrics_for_query(
    *, distances: Any, reference_fingerprints: Sequence[str], relevant: Any,
) -> dict[str, Any]:
    import numpy as np
    mask = np.asarray(relevant, dtype=bool)
    order = stable_top_k(distances, reference_fingerprints, len(reference_fingerprints))
    ranks = [rank for rank, index in enumerate(order, start=1) if bool(mask[index])]
    return {
        "eligible": bool(ranks), "relevant_ranks": ranks,
        "best_rank": min(ranks) if ranks else None,
        "median_rank": float(np.median(np.asarray(ranks))) if ranks else None,
    }


def summarize_relevant_rank_queries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np
    eligible = [row for row in rows if bool(row["eligible"])]
    _require(bool(eligible), "rank summary requires an eligible query")
    best = np.asarray([int(row["best_rank"]) for row in eligible], dtype=np.float64)
    medians = np.asarray([float(row["median_rank"]) for row in eligible], dtype=np.float64)
    return {
        "eligible_query_count": len(eligible), "mean_best_rank": float(np.mean(best)),
        "median_best_rank": float(np.median(best)), "median_relevant_rank": float(np.median(medians)),
        "mrr": float(np.mean(1.0 / best)),
        "hit_top1_rate": float(np.mean(best <= 1)), "hit_top5_rate": float(np.mean(best <= 5)),
        "hit_top10_rate": float(np.mean(best <= 10)), "hit_top20_rate": float(np.mean(best <= 20)),
    }


def assert_shared_query_reference_cells(
    metric_cells: Sequence[tuple[str, str]], baseline_cells: Sequence[tuple[str, str]],
) -> None:
    _require(list(metric_cells) == list(baseline_cells),
             "metric and explicit baseline query/reference cells differ")


def adjacent_bin_median_increases(medians: Sequence[float]) -> dict[str, int]:
    _require(len(medians) == 5, "gap-bin transition diagnostic requires five medians")
    return {
        "increasing_transitions": sum(float(right) > float(left) for left, right in zip(medians, medians[1:])),
        "total_transitions": 4,
    }


def build_decision(classification: str) -> dict[str, Any]:
    _require(classification in ALLOWED_DECISIONS, "unfrozen diagnosis decision")
    return {
        "classification": classification,
        "historical_soft_metric_classification": "SOFT_JOINT_METRIC_REJECTED",
        "historical_classification_modified": False,
        "new_method_readiness_authorized": False, "bo_modified": False,
        "candidate_gnn_training_started": False, "test_data_accessed": False,
    }


def _load_module_from_snapshot(name: str, path: Path, package: Any) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    setattr(package, name.rsplit(".", 1)[-1], module)
    spec.loader.exec_module(module)
    module.REPO_ROOT = REPO_ROOT
    return module


def load_frozen_v2_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the archived scientific runtime into its original canonical names."""
    snapshot_root = _repo_path(str(config["frozen_v2_runtime"]["snapshot_root"]))
    analyse_package = importlib.import_module("analyse")
    paths = {
        "performance": snapshot_root / "analyse/performance_metric_quick_gate.py",
        "joint": snapshot_root / "analyse/joint_configuration_metric_gate.py",
        "pair_audit": snapshot_root / "analyse/joint_pair_power_projection_audit.py",
        "soft": snapshot_root / "analyse/joint_configuration_soft_metric_gate.py",
    }
    modules: dict[str, Any] = {}
    load_order = (
        ("performance", "analyse.performance_metric_quick_gate"),
        ("joint", "analyse.joint_configuration_metric_gate"),
        ("pair_audit", "analyse.joint_pair_power_projection_audit"),
        ("soft", "analyse.joint_configuration_soft_metric_gate"),
    )
    for key, canonical_name in load_order:
        modules[key] = _load_module_from_snapshot(
            canonical_name, paths[key], analyse_package,
        )
    for key, module in modules.items():
        resolved = Path(module.__file__).resolve()
        _require(snapshot_root in resolved.parents, f"{key} did not load from frozen snapshot")
        representative = (
            module.train_soft_metric_encoder if key == "soft"
            else module.explicit_graph_features if key == "joint"
            else module.stable_hash
        )
        _require(snapshot_root in Path(representative.__code__.co_filename).resolve().parents,
                 f"{key} function bytecode is not frozen snapshot code")
    return {**modules, "snapshot_root": snapshot_root}


def validate_authoritative_input(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["authoritative_soft_metric_v2"]
    root = _repo_path(str(frozen["result_root"]))
    archive = _repo_path(str(frozen["archive"]))
    sidecar = _repo_path(str(frozen["archive_sha256_sidecar"]))
    paths = {
        "marker": root / str(frozen["required_marker"]), "state": root / "state.json",
        "decision": root / "decision.json", "inventory": root / "data_inventory.json",
        "geometry": root / "geometry_metrics.json", "locality": root / "knn_locality.json",
        "split": root / "split_manifest.json", "training_audit": root / "metric_training_audit.json",
        "snapshot_manifest": _repo_path(str(config["frozen_v2_runtime"]["manifest"])),
        "archive": archive, "sidecar": sidecar,
    }
    missing = [f"{name}:{path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ReplayInputIdentityMismatch(f"authoritative soft-v2 files missing: {missing}")
    try:
        state = _read_json(paths["state"])
        decision = _read_json(paths["decision"])
        inventory = _read_json(paths["inventory"])
        geometry = _read_json(paths["geometry"])
        locality = _read_json(paths["locality"])
        split = _read_json(paths["split"])
        training_audit = _read_json(paths["training_audit"])
        snapshot_manifest = _read_json(paths["snapshot_manifest"])
    except Exception as error:
        raise ReplayInputIdentityMismatch(f"authoritative JSON cannot be read: {error}") from error
    try:
        training_runs = list(training_audit["runs"])
        training_keys = [(int(run["fold"]), int(run["seed"])) for run in training_runs]
        locality_records = list(locality["joint_metric"]["records"])
        locality_queries = [str(row["query_joint_fingerprint"]) for row in locality_records]
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayInputIdentityMismatch(
            f"authoritative replay-audit schema is invalid: {error}"
        ) from error
    split_payload = dict(split)
    recorded_split_sha = str(split_payload.pop("split_sha256", ""))
    archive_sha = sha256_file(archive)
    sidecar_token = sidecar.read_text(encoding="utf-8").strip().split()[0]
    identity = {
        "classification": decision.get("classification"), "archive_sha256": archive_sha,
        "config_sha256": state.get("config_sha256"), "source_sha256": state.get("source_sha256"),
        "source_snapshot_manifest_root": state.get("source_snapshot_manifest_root"),
        "split_sha256": recorded_split_sha,
        "unique_joint_configurations": inventory.get("unique_joint_configurations"),
        "unique_architectures": inventory.get("unique_architectures"),
        "pool_count": inventory.get("unique_pools"),
        "seed_record_count": inventory.get("raw_row_count"),
        "training_seeds": inventory.get("complete_training_seeds"),
    }
    validate_input_identity_payload(identity)
    checks = {
        "state_complete": state.get("status") == "complete",
        "state_decision": state.get("decision") == FROZEN_INPUT_IDENTITY["classification"],
        "archive_sidecar": sidecar_token == archive_sha,
        "snapshot_manifest_root": snapshot_manifest.get("manifest_root_sha256")
        == FROZEN_INPUT_IDENTITY["source_snapshot_manifest_root"],
        "snapshot_manifest_recomputes": stable_hash(snapshot_manifest.get("entries", []))
        == snapshot_manifest.get("manifest_root_sha256"),
        "logical_split_recomputes": stable_hash(split_payload) == recorded_split_sha,
        "target": inventory.get("target") == "raw_validation_accuracy_seed_mean",
        "test_not_accessed": state.get("test_data_accessed") is False
        and inventory.get("test_data_accessed") is False,
        "G1": decision["geometry_gate"]["checks"]["G1_median_spearman"] is True,
        "G2": decision["geometry_gate"]["checks"]["G2_positive_seed_fold_pattern"] is True,
        "G3": decision["geometry_gate"]["checks"]["G3_cross_factor_locality"] is False,
        "G4": decision["geometry_gate"]["checks"]["G4_local_gap_at_5"] is True,
        "M1": decision["M1_gate"]["passed"] is True,
        "M2": decision["M2_gate"]["passed"] is True,
        "training_audit_exact_coverage": len(training_keys) == 15
        and len(set(training_keys)) == 15
        and set(training_keys) == {(fold, seed) for fold in range(5) for seed in METRIC_SEEDS},
        "top5_query_exact_coverage": len(locality_queries) == 336
        and len(set(locality_queries)) == 336,
    }
    expected_values = frozen["expected_key_values"]
    observed_values = {
        "median_distance_gap_spearman": decision["geometry_gate"]["median_seed_fold_distance_gap_spearman"],
        "cross_factor_locality_ratio": geometry["cross_factor_locality_ratio"],
        "metric_gap_at_5": geometry["joint_metric_local_performance_gap_at_5"],
        "explicit_joint_feature_gap_at_5": geometry["explicit_joint_feature_local_performance_gap_at_5"],
    }
    checks.update({
        f"key_value_{key}": math.isclose(float(value), float(expected_values[key]), abs_tol=1.0e-12)
        for key, value in observed_values.items()
    })
    snapshot_root = _repo_path(str(config["frozen_v2_runtime"]["snapshot_root"]))
    manifest_by_relative = {
        str(entry["repository_relative_path"]): entry for entry in snapshot_manifest["entries"]
    }
    for relative, expected in config["frozen_v2_runtime"]["files"].items():
        source = snapshot_root / str(relative)
        checks[f"frozen_source_{relative}"] = (
            source.is_file() and sha256_file(source) == str(expected["sha256"])
            and manifest_by_relative.get(str(relative), {}).get("sha256") == str(expected["sha256"])
        )
    for entry in snapshot_manifest["entries"]:
        source = snapshot_root / str(entry["repository_relative_path"])
        checks[f"manifest_entry_{entry['repository_relative_path']}"] = (
            source.is_file() and source.stat().st_size == int(entry["size_bytes"])
            and sha256_file(source) == str(entry["sha256"])
        )
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ReplayInputIdentityMismatch(f"authoritative soft-v2 checks failed: {failed}")
    return {
        "status": "VALID", **identity, "identity": identity, "checks": checks,
        "paths": {key: str(value) for key, value in paths.items()},
        "state": state, "decision": decision, "inventory": inventory,
        "geometry": geometry, "locality": locality, "split": split,
        "training_audit": training_audit, "snapshot_manifest": snapshot_manifest,
        "historical_classification_modified": False, "test_data_accessed": False,
    }


def archived_replay_identity(authority: Mapping[str, Any]) -> dict[str, Any]:
    geometry = authority["geometry"]
    locality = authority["locality"]
    return {
        "distance_gap_spearman": {
            f"fold_{int(fold['fold'])}": {
                f"seed_{seed}": float(fold["seed_distance_gap_spearman"][str(seed)])
                for seed in METRIC_SEEDS
            }
            for fold in geometry["folds"]
        },
        "cross_factor_locality_ratio": {
            f"fold_{int(fold['fold'])}": float(fold["cross_factor_locality_ratio"])
            for fold in geometry["folds"]
        },
        "metric_top5_fingerprints": {
            str(row["query_joint_fingerprint"]): list(row["neighbor_joint_fingerprints"])
            for row in locality["joint_metric"]["records"]
        },
        "pooled_metric_gap_at_5": float(
            locality["joint_metric"]["mean_local_performance_gap_at_5"]
        ),
    }


def replay_frozen_metric_encoders(
    config: Mapping[str, Any], authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replay exactly 5 folds x 3 seeds, returning matrices but no model weights."""
    import numpy as np
    import torch
    assert_replay_scope(["metric_encoder"])
    runtime = load_frozen_v2_runtime(config)
    soft = runtime["soft"]
    frozen_config_path = runtime["snapshot_root"] / "configs/joint_configuration_soft_metric_gate_v2.yaml"
    frozen_config = soft.load_config(frozen_config_path)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        configurations, inventory, provenance, manifest = soft.load_main_data_and_split(
            frozen_config
        )
    except Exception as error:
        if type(error).__name__ in {"InputIdentityMismatch", "ProtocolError"}:
            raise ReplayInputIdentityMismatch(
                f"frozen soft-v2 data identity/loader check failed: {error}"
            ) from error
        raise
    replay_identity_payload = {
        "classification": FROZEN_INPUT_IDENTITY["classification"],
        "archive_sha256": FROZEN_INPUT_IDENTITY["archive_sha256"],
        "config_sha256": FROZEN_INPUT_IDENTITY["config_sha256"],
        "source_sha256": FROZEN_INPUT_IDENTITY["source_sha256"],
        "source_snapshot_manifest_root": FROZEN_INPUT_IDENTITY["source_snapshot_manifest_root"],
        "split_sha256": str(manifest["split_sha256"]),
        "unique_joint_configurations": int(inventory["unique_joint_configurations"]),
        "unique_architectures": int(inventory["unique_architectures"]),
        "pool_count": int(inventory["unique_pools"]),
        "seed_record_count": int(inventory["raw_row_count"]),
        "training_seeds": list(inventory["complete_training_seeds"]),
    }
    validate_input_identity_payload(replay_identity_payload)
    archived_runs = {
        (int(run["fold"]), int(run["seed"])): run
        for run in authority["training_audit"]["runs"]
    }
    expected_run_keys = {(fold, seed) for fold in range(5) for seed in METRIC_SEEDS}
    if set(archived_runs) != expected_run_keys:
        raise ReplayInputIdentityMismatch(
            "archive metric training audit lacks exact 5-fold x 3-seed coverage"
        )
    fold_replays: list[dict[str, Any]] = []
    geometry_results: list[dict[str, Any]] = []
    replay_training_audits: list[dict[str, Any]] = []
    for fold in manifest["folds"]:
        fold_id = int(fold["fold"])
        train_ids = set(map(str, fold["train_joint_fingerprints"]))
        heldout_ids = set(map(str, fold["heldout_joint_fingerprints"]))
        train_rows = sorted(
            (dict(row) for row in configurations if str(row["joint_fingerprint"]) in train_ids),
            key=lambda row: str(row["joint_fingerprint"]),
        )
        heldout_rows = sorted(
            (dict(row) for row in configurations if str(row["joint_fingerprint"]) in heldout_ids),
            key=lambda row: str(row["joint_fingerprint"]),
        )
        assert_architecture_disjoint(
            [str(row["architecture_fingerprint"]) for row in heldout_rows],
            [str(row["architecture_fingerprint"]) for row in train_rows],
        )
        hp_standardizer = soft.joint_v1.fit_hp_standardizer(train_rows)
        pair_config = frozen_config["soft_metric"]
        full_pairs = soft.construct_soft_pairs(
            train_rows, allowed_joint_fingerprints=train_ids,
            delta=float(pair_config["delta"]),
            minimum_weight=float(pair_config["minimum_pair_weight"]),
            reliability_fraction=float(pair_config["reliability_weight_fraction"]),
            pair_cap=int(pair_config["pair_cap"]),
            sampling_seed=int(pair_config["pair_sampling_seed"]),
        )
        seed_matrices: dict[int, Any] = {}
        for seed in METRIC_SEEDS:
            print(f"replay fold={fold_id} metric_seed={seed}", flush=True)
            train_embedding, heldout_embedding, _model, audit = soft.train_soft_metric_encoder(
                train_rows, heldout_rows, seed, frozen_config, hp_standardizer, full_pairs,
            )
            archived = archived_runs[(fold_id, seed)]
            if int(audit["best_epoch"]) != int(archived["best_epoch"]):
                raise ReplayIdentityMismatch(
                    f"best epoch mismatch fold={fold_id} seed={seed}: "
                    f"{audit['best_epoch']} != {archived['best_epoch']}"
                )
            seed_matrices[seed] = np.linalg.norm(
                heldout_embedding[:, None, :] - train_embedding[None, :, :], axis=2,
            )
            replay_training_audits.append({
                "fold": fold_id, "seed": seed, "best_epoch": int(audit["best_epoch"]),
                "archived_best_epoch": int(archived["best_epoch"]),
                "full_outer_refit": bool(audit["final_refit_used_complete_outer_train"]),
            })
        train_hp = soft.joint_v1._raw_hp_matrix(train_rows)
        heldout_hp = soft.joint_v1._raw_hp_matrix(heldout_rows)
        train_graph = soft.joint_v1._explicit_graph_matrix(train_rows)
        heldout_graph = soft.joint_v1._explicit_graph_matrix(heldout_rows)
        train_joint = np.concatenate((train_graph, train_hp), axis=1)
        heldout_joint = np.concatenate((heldout_graph, heldout_hp), axis=1)
        explicit_train, explicit_heldout, explicit_transform = soft.joint_v1.standardize_train_query(
            train_joint, heldout_joint,
        )
        explicit_distances = np.linalg.norm(
            explicit_heldout[:, None, :] - explicit_train[None, :, :], axis=2,
        )
        geometry = soft.geometry_for_fold_soft(
            train_rows, heldout_rows, seed_matrices, explicit_distances,
            hp_standardizer, fold_id,
        )
        geometry_results.append({"geometry": geometry})
        fold_replays.append({
            "fold": fold_id, "train_rows": train_rows, "heldout_rows": heldout_rows,
            "seed_distance_matrices": seed_matrices,
            "mean_distance_matrix": average_metric_seed_distance_matrices(
                [seed_matrices[seed] for seed in METRIC_SEEDS]
            ),
            "explicit_distance_matrix": explicit_distances,
            "raw_train_hp": train_hp, "raw_heldout_hp": heldout_hp,
            "hp_standardizer": hp_standardizer,
            "explicit_transform": explicit_transform, "geometry": geometry,
        })
    aggregate_geometry, _locality = soft.aggregate_geometry_soft(geometry_results)
    replayed_identity = {
        "distance_gap_spearman": {
            f"fold_{fold['fold']}": {
                f"seed_{seed}": float(fold["geometry"]["seed_distance_gap_spearman"][str(seed)])
                for seed in METRIC_SEEDS
            }
            for fold in fold_replays
        },
        "cross_factor_locality_ratio": {
            f"fold_{fold['fold']}": float(fold["geometry"]["cross_factor_locality_ratio"])
            for fold in fold_replays
        },
        "metric_top5_fingerprints": {
            str(row["query_joint_fingerprint"]): list(row["neighbor_joint_fingerprints"])
            for fold in fold_replays for row in fold["geometry"]["joint_metric_knn"]["records"]
        },
        "pooled_metric_gap_at_5": float(aggregate_geometry["joint_metric_local_performance_gap_at_5"]),
    }
    archived_identity = archived_replay_identity(authority)
    gate = assert_replay_identity(
        replayed_identity, archived_identity,
        atol=float(config["replay"]["identity_gate"]["absolute_tolerance"]),
    )
    replay_audit = {
        "status": "REPRODUCED", "identity_gate": gate,
        "replayed": replayed_identity, "archived": archived_identity,
        "training_runs": replay_training_audits, "runtime_contract": replay_runtime_contract(),
        "frozen_module_paths": {
            key: str(module.__file__) for key, module in runtime.items()
            if key in {"soft", "joint", "performance", "pair_audit"}
        },
        "live_repo_root_binding": str(REPO_ROOT), "model_weights_saved": False,
        "GP_or_regressor_replayed": False, "stabilization_seeds_used": False,
        "test_data_accessed": False,
    }
    return configurations, fold_replays, replay_audit


def build_cross_factor_cells(fold_replays: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    all_folds: list[dict[str, Any]] = []
    for replay in fold_replays:
        train_rows = list(replay["train_rows"])
        heldout_rows = list(replay["heldout_rows"])
        raw_train_hp = np.asarray(replay["raw_train_hp"], dtype=np.float64)
        raw_heldout_hp = np.asarray(replay["raw_heldout_hp"], dtype=np.float64)
        frozen_standardizer = replay["hp_standardizer"]
        context = {
            "mean": list(frozen_standardizer["mean"]),
            "scale": list(frozen_standardizer["scale"]),
            "fit_scope": "outer_train_only", "fit_row_count": len(train_rows),
        }
        mean = np.asarray(context["mean"], dtype=np.float64)
        scale = np.asarray(context["scale"], dtype=np.float64)
        standardized_train = (raw_train_hp - mean[None, :]) / scale[None, :]
        standardized_heldout = (raw_heldout_hp - mean[None, :]) / scale[None, :]
        train_pair_distances = [
            float(np.linalg.norm(standardized_train[left] - standardized_train[right]))
            for left, right in itertools.combinations(range(len(train_rows)), 2)
        ]
        context["train_pair_median_distance"] = float(
            np.median(np.asarray(train_pair_distances, dtype=np.float64))
        )
        archived_hp_median = float(replay["geometry"]["cross_factor_hp_distance_train_median"])
        _require(abs(context["train_pair_median_distance"] - archived_hp_median) <= 1.0e-12,
                 "diagnostic HP-far threshold does not reproduce frozen G3 threshold")
        hp_far = np.zeros((len(heldout_rows), len(train_rows)), dtype=bool)
        for query_index in range(len(heldout_rows)):
            for reference_index in range(len(train_rows)):
                hp_far[query_index, reference_index] = float(np.linalg.norm(
                    standardized_heldout[query_index] - standardized_train[reference_index]
                )) > archived_hp_median
        mean_distance = np.asarray(replay["mean_distance_matrix"], dtype=np.float64)
        explicit_distance = np.asarray(replay["explicit_distance_matrix"], dtype=np.float64)
        seed_distances = replay["seed_distance_matrices"]
        fold_cells: list[dict[str, Any]] = []
        for query_index, query in enumerate(heldout_rows):
            query_cells: list[dict[str, Any]] = []
            for reference_index, reference in enumerate(train_rows):
                architecture_different = (
                    str(query["architecture_fingerprint"])
                    != str(reference["architecture_fingerprint"])
                )
                cell = {
                    "fold": int(replay["fold"]),
                    "query_joint_fingerprint": str(query["joint_fingerprint"]),
                    "reference_joint_fingerprint": str(reference["joint_fingerprint"]),
                    "query_architecture_fingerprint": str(query["architecture_fingerprint"]),
                    "reference_architecture_fingerprint": str(reference["architecture_fingerprint"]),
                    "architecture_different": architecture_different,
                    "hp_far": bool(hp_far[query_index, reference_index]),
                    "hp_near": not bool(hp_far[query_index, reference_index]),
                    "performance_gap": abs(
                        float(query["validation_mean"]) - float(reference["validation_mean"])
                    ),
                    "metric_mean_distance": float(mean_distance[query_index, reference_index]),
                    "explicit_joint_distance": float(explicit_distance[query_index, reference_index]),
                    **{
                        f"metric_seed_{seed}_distance": float(
                            seed_distances[seed][query_index, reference_index]
                        ) for seed in METRIC_SEEDS
                    },
                }
                query_cells.append(cell)
                fold_cells.append(cell)
            _require(len(query_cells) == len(train_rows), "query/reference cell coverage mismatch")
        rate = sum(bool(cell["architecture_different"]) for cell in fold_cells) / len(fold_cells)
        _require(rate == 1.0, "heldout/train architecture-different rate is not exactly one")
        _require(sum(bool(cell["hp_far"]) for cell in fold_cells)
                 == int(replay["geometry"]["cross_factor_cell_count"]),
                 "diagnostic cross-factor cell count does not reproduce frozen G3")
        all_folds.append({
            "fold": int(replay["fold"]), "cells": fold_cells,
            "hp_context": context, "architecture_different_rate": rate,
            "query_count": len(heldout_rows), "reference_count": len(train_rows),
        })
    return all_folds


def _quantile_summary(values: Sequence[float]) -> dict[str, Any]:
    import numpy as np
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {
            "count": 0, "minimum": None, "q10": None, "q25": None, "median": None,
            "q75": None, "q90": None, "maximum": None, "mean": None,
        }
    return {
        "count": int(len(array)), "minimum": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)), "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)), "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)), "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def bootstrap_query_summary(
    values: Sequence[float], *, replicates: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    import numpy as np
    array = np.asarray(list(values), dtype=np.float64)
    _require(len(array) > 0 and np.isfinite(array).all(), "bootstrap requires finite query values")
    rng = np.random.default_rng(int(seed))
    mean_samples = np.empty(int(replicates), dtype=np.float64)
    median_samples = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sample = array[rng.integers(0, len(array), size=len(array))]
        mean_samples[index] = float(np.mean(sample))
        median_samples[index] = float(np.median(sample))
    return {
        "query_count": int(len(array)), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "bootstrap_unit": "heldout_query",
        "bootstrap_replicates": int(replicates), "bootstrap_seed": int(seed),
        "mean_ci95": [float(np.quantile(mean_samples, 0.025)), float(np.quantile(mean_samples, 0.975))],
        "median_ci95": [
            float(np.quantile(median_samples, 0.025)), float(np.quantile(median_samples, 0.975)),
        ],
    }


def auc_and_cliffs_delta(close_distances: Sequence[float], far_distances: Sequence[float]) -> dict[str, float]:
    import numpy as np
    close = np.asarray(list(close_distances), dtype=np.float64)
    far = np.asarray(list(far_distances), dtype=np.float64)
    _require(len(close) > 0 and len(far) > 0, "AUC requires both close and far references")
    comparisons = far[:, None] - close[None, :]
    greater = float(np.mean(comparisons > 0.0))
    less = float(np.mean(comparisons < 0.0))
    ties = float(np.mean(comparisons == 0.0))
    return {"auc": greater + 0.5 * ties, "cliffs_delta": greater - less}


def relevant_reference(cell: Mapping[str, Any], delta_close: float) -> bool:
    return bool(
        cell["architecture_different"] and cell["hp_far"]
        and float(cell["performance_gap"]) <= float(delta_close)
    )


def compute_retrieval_for_query(
    cells: Sequence[Mapping[str, Any]], *, distance_field: str, k: int, delta_close: float,
) -> dict[str, Any]:
    _require(len(cells) >= int(k), "retrieval query has fewer references than k")
    fingerprints = [str(cell["reference_joint_fingerprint"]) for cell in cells]
    order = stable_top_k([float(cell[distance_field]) for cell in cells], fingerprints, int(k))
    relevant = [relevant_reference(cell, delta_close) for cell in cells]
    relevant_count = sum(relevant)
    retrieved = sum(relevant[index] for index in order)
    prevalence = relevant_count / len(cells)
    precision = retrieved / int(k)
    return {
        "relevant_reference_count": relevant_count,
        "eligible": relevant_count > 0,
        "retrieved_relevant_count": retrieved,
        "precision": precision,
        "recall": retrieved / relevant_count if relevant_count else None,
        "hit": bool(retrieved > 0) if relevant_count else None,
        "base_prevalence": prevalence,
        "precision_lift": precision / prevalence if prevalence > 0.0 else None,
        "precision_denominator": int(k),
        "recall_denominator": relevant_count if relevant_count else None,
        "top_k_reference_fingerprints": [fingerprints[index] for index in order],
    }


def rank_relevant_for_query(
    cells: Sequence[Mapping[str, Any]], *, distance_field: str, delta_close: float,
) -> dict[str, Any] | None:
    fingerprints = [str(cell["reference_joint_fingerprint"]) for cell in cells]
    order = stable_top_k([float(cell[distance_field]) for cell in cells], fingerprints, len(cells))
    ranks = [rank for rank, index in enumerate(order, start=1) if relevant_reference(cells[index], delta_close)]
    if not ranks:
        return None
    import numpy as np
    return {
        "relevant_count": len(ranks), "best_rank": int(min(ranks)),
        "median_rank": float(np.median(np.asarray(ranks, dtype=np.float64))),
        "reciprocal_best_rank": 1.0 / min(ranks),
        "hit_top1": min(ranks) <= 1, "hit_top5": min(ranks) <= 5,
        "hit_top10": min(ranks) <= 10, "hit_top20": min(ranks) <= 20,
    }


# Public aliases used by focused protocol tests.
query_distance_discrimination = auc_and_cliffs_delta
query_bootstrap_summary = bootstrap_query_summary


def _group_cells_by_query(fold_cells: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in fold_cells["cells"]:
        grouped[str(cell["query_joint_fingerprint"])].append(dict(cell))
    return [
        sorted(grouped[query], key=lambda row: str(row["reference_joint_fingerprint"]))
        for query in sorted(grouped)
    ]


def diagnose_g3_quantile_alignment(
    fold_cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in fold_cells:
        cross = [
            dict(cell) for cell in fold["cells"]
            if bool(cell["architecture_different"]) and bool(cell["hp_far"])
        ]
        _require(bool(cross), f"fold {fold['fold']} has no cross-factor cells")
        ordered = sorted(
            cross,
            key=lambda row: (
                float(row["performance_gap"]), str(row["query_joint_fingerprint"]),
                str(row["reference_joint_fingerprint"]),
            ),
        )
        group_size = int(math.ceil(0.20 * len(ordered)))
        for name, selected in (("bottom_20_low_gap", ordered[:group_size]),
                               ("top_20_high_gap", ordered[-group_size:])):
            gaps = [float(row["performance_gap"]) for row in selected]
            summary = _quantile_summary(gaps)
            rows.append({
                "fold": int(fold["fold"]), "group": name,
                "cross_factor_cell_count": len(ordered), "group_count": len(selected),
                **summary,
                "fraction_gap_le_0.006": sum(value <= 0.006 for value in gaps) / len(gaps),
                "fraction_gap_le_0.012": sum(value <= 0.012 for value in gaps) / len(gaps),
                "fraction_gap_le_0.025": sum(value <= 0.025 for value in gaps) / len(gaps),
                "group_definition_unchanged": True,
            })
    return rows


def diagnose_gap_bins(
    fold_cells: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    distance_fields = {
        "joint_metric_mean": "metric_mean_distance",
        "explicit_joint_features": "explicit_joint_distance",
        **{f"joint_metric_seed_{seed}": f"metric_seed_{seed}_distance" for seed in METRIC_SEEDS},
    }
    rows: list[dict[str, Any]] = []
    for fold in fold_cells:
        cross = [
            cell for cell in fold["cells"]
            if bool(cell["architecture_different"]) and bool(cell["hp_far"])
        ]
        for bin_name in GAP_BIN_ORDER:
            selected = [cell for cell in cross if gap_bin_label(cell["performance_gap"]) == bin_name]
            for method, field in distance_fields.items():
                summary = _quantile_summary([float(cell[field]) for cell in selected])
                rows.append({
                    "fold": int(fold["fold"]), "gap_bin": bin_name, "method": method,
                    "pair_count": len(selected),
                    "query_count": len({str(cell["query_joint_fingerprint"]) for cell in selected}),
                    "mean_distance": summary["mean"], "median_distance": summary["median"],
                    "q25_distance": summary["q25"], "q75_distance": summary["q75"],
                })
    summary: dict[str, Any] = {"by_method": {}, "interpretation_is_gate": False}
    for method in distance_fields:
        method_rows = [row for row in rows if row["method"] == method]
        pooled_medians = []
        pooled_rows = []
        for bin_name in GAP_BIN_ORDER:
            cells = [
                cell for fold in fold_cells for cell in fold["cells"]
                if bool(cell["architecture_different"]) and bool(cell["hp_far"])
                and gap_bin_label(cell["performance_gap"]) == bin_name
            ]
            field = distance_fields[method]
            values = [float(cell[field]) for cell in cells]
            q = _quantile_summary(values)
            pooled_medians.append(q["median"])
            pooled_rows.append({"gap_bin": bin_name, "pair_count": len(cells), **q})
        if all(value is not None for value in pooled_medians):
            transition = adjacent_bin_median_increases(pooled_medians)
        else:
            transition = {
                "increasing_transitions": None, "total_transitions": 4,
                "undefined_reason": "one or more absolute gap bins contain no cross-factor cell",
            }
        summary["by_method"][method] = {
            "fold_rows": method_rows, "pooled": pooled_rows,
            "adjacent_bin_transitions": transition,
        }
    return rows, summary


def diagnose_neighbor_composition(
    fold_cells: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int = 2000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    methods = {
        "joint_metric": "metric_mean_distance",
        "explicit_joint_features": "explicit_joint_distance",
    }
    rows: list[dict[str, Any]] = []
    measures = (
        "gap_le_0.006_rate", "gap_le_0.012_rate", "hp_far_rate", "hp_near_rate",
        "hp_far_and_gap_le_0.006_rate", "hp_far_and_gap_le_0.012_rate",
        "architecture_different_rate",
    )
    for fold in fold_cells:
        for cells in _group_cells_by_query(fold):
            query = str(cells[0]["query_joint_fingerprint"])
            references = [str(cell["reference_joint_fingerprint"]) for cell in cells]
            for method, field in methods.items():
                for k in TOP_K_VALUES:
                    order = stable_top_k([float(cell[field]) for cell in cells], references, k)
                    selected = [cells[index] for index in order]
                    row = {
                        "fold": int(fold["fold"]), "query_joint_fingerprint": query,
                        "method": method, "k": k,
                        "gap_le_0.006_rate": sum(float(x["performance_gap"]) <= 0.006 for x in selected) / k,
                        "gap_le_0.012_rate": sum(float(x["performance_gap"]) <= 0.012 for x in selected) / k,
                        "hp_far_rate": sum(bool(x["hp_far"]) for x in selected) / k,
                        "hp_near_rate": sum(bool(x["hp_near"]) for x in selected) / k,
                        "hp_far_and_gap_le_0.006_rate": sum(
                            bool(x["hp_far"]) and float(x["performance_gap"]) <= 0.006
                            for x in selected
                        ) / k,
                        "hp_far_and_gap_le_0.012_rate": sum(
                            bool(x["hp_far"]) and float(x["performance_gap"]) <= 0.012
                            for x in selected
                        ) / k,
                        "architecture_different_rate": sum(
                            bool(x["architecture_different"]) for x in selected
                        ) / k,
                        "neighbor_joint_fingerprints": [
                            str(x["reference_joint_fingerprint"]) for x in selected
                        ],
                    }
                    _require(row["architecture_different_rate"] == 1.0,
                             "neighbor architecture-different rate is not one")
                    rows.append(row)
    summary: dict[str, Any] = {
        "bootstrap_unit": "heldout_query", "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": 0, "architecture_difference_due_to_split": True, "groups": {},
    }
    for method in methods:
        for k in TOP_K_VALUES:
            subset = [row for row in rows if row["method"] == method and int(row["k"]) == k]
            summary["groups"][f"{method}_k{k}"] = {
                measure: bootstrap_query_summary(
                    [float(row[measure]) for row in subset], replicates=bootstrap_replicates, seed=0,
                ) for measure in measures
            }
    return rows, summary


def diagnose_retrieval(
    fold_cells: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    methods = {
        "joint_metric": "metric_mean_distance",
        "explicit_joint_features": "explicit_joint_distance",
    }
    rows: list[dict[str, Any]] = []
    for fold in fold_cells:
        for cells in _group_cells_by_query(fold):
            query = str(cells[0]["query_joint_fingerprint"])
            for method, field in methods.items():
                for delta in DELTA_CLOSES:
                    for k in TOP_K_VALUES:
                        metrics = compute_retrieval_for_query(
                            cells, distance_field=field, k=k, delta_close=delta,
                        )
                        rows.append({
                            "fold": int(fold["fold"]), "query_joint_fingerprint": query,
                            "method": method, "delta_close": delta, "k": k, **metrics,
                        })
    summary: dict[str, Any] = {"groups": {}}
    for method in methods:
        for delta in DELTA_CLOSES:
            for k in TOP_K_VALUES:
                subset = [
                    row for row in rows if row["method"] == method
                    and float(row["delta_close"]) == delta and int(row["k"]) == k
                ]
                eligible = [row for row in subset if bool(row["eligible"])]
                total_relevant = sum(int(row["relevant_reference_count"]) for row in subset)
                total_references = sum(
                    next(len(cells) for fold in fold_cells for cells in _group_cells_by_query(fold)
                         if str(cells[0]["query_joint_fingerprint"]) == row["query_joint_fingerprint"])
                    for row in subset
                )
                prevalence = total_relevant / total_references
                precision = sum(float(row["precision"]) for row in subset) / len(subset)
                summary["groups"][f"{method}_delta{delta:.3f}_k{k}"] = {
                    "query_count": len(subset), "eligible_query_count": len(eligible),
                    "no_opportunity_query_count": len(subset) - len(eligible),
                    "precision_at_k": precision,
                    "conditional_recall_at_k": (
                        sum(float(row["recall"]) for row in eligible) / len(eligible)
                        if eligible else None
                    ),
                    "conditional_hit_at_k": (
                        sum(bool(row["hit"]) for row in eligible) / len(eligible)
                        if eligible else None
                    ),
                    "relevant_base_prevalence": prevalence,
                    "precision_lift": precision / prevalence if prevalence > 0 else None,
                    "precision_denominator_per_query": k,
                    "conditional_denominator": "eligible_heldout_query",
                }
    return rows, summary


def diagnose_ranks(fold_cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = {
        "joint_metric": "metric_mean_distance",
        "explicit_joint_features": "explicit_joint_distance",
    }
    output: dict[str, Any] = {"groups": {}}
    for method, field in methods.items():
        for delta in DELTA_CLOSES:
            per_query = []
            for fold in fold_cells:
                for cells in _group_cells_by_query(fold):
                    result = rank_relevant_for_query(cells, distance_field=field, delta_close=delta)
                    if result is not None:
                        per_query.append({
                            "fold": int(fold["fold"]),
                            "query_joint_fingerprint": str(cells[0]["query_joint_fingerprint"]),
                            **result,
                        })
            _require(bool(per_query), f"no rank-eligible queries for delta={delta}")
            import numpy as np
            best = np.asarray([row["best_rank"] for row in per_query], dtype=np.float64)
            output["groups"][f"{method}_delta{delta:.3f}"] = {
                "eligible_query_count": len(per_query), "per_query": per_query,
                "median_best_rank": float(np.median(best)), "mean_best_rank": float(np.mean(best)),
                "MRR": float(np.mean(1.0 / best)),
                **{
                    f"relevant_enters_top_{cutoff}_rate": float(np.mean(best <= cutoff))
                    for cutoff in (1, 5, 10, 20)
                },
            }
    return output


def diagnose_distance_discrimination(
    fold_cells: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int = 2000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    methods = {
        "joint_metric": "metric_mean_distance",
        "explicit_joint_features": "explicit_joint_distance",
    }
    rows: list[dict[str, Any]] = []
    for fold in fold_cells:
        for cells in _group_cells_by_query(fold):
            cross = [cell for cell in cells if cell["architecture_different"] and cell["hp_far"]]
            for method, field in methods.items():
                for delta in DELTA_CLOSES:
                    close = [float(cell[field]) for cell in cross if cell["performance_gap"] <= delta]
                    far = [float(cell[field]) for cell in cross if cell["performance_gap"] > 0.050]
                    if close and far:
                        rows.append({
                            "fold": int(fold["fold"]),
                            "query_joint_fingerprint": str(cells[0]["query_joint_fingerprint"]),
                            "method": method, "delta_close": delta,
                            "close_reference_count": len(close), "far_reference_count": len(far),
                            **auc_and_cliffs_delta(close, far),
                        })
    output: dict[str, Any] = {
        "statistical_unit": "heldout_query", "groups": {},
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": 0,
    }
    for method in methods:
        for delta in DELTA_CLOSES:
            subset = [
                row for row in rows if row["method"] == method
                and float(row["delta_close"]) == delta
            ]
            _require(bool(subset), f"no discrimination-eligible queries for {method}/{delta}")
            output["groups"][f"{method}_close{delta:.3f}_vs_far0.050"] = {
                "eligible_query_count": len(subset),
                "auc": bootstrap_query_summary(
                    [float(row["auc"]) for row in subset], replicates=bootstrap_replicates, seed=0,
                ),
                "cliffs_delta": bootstrap_query_summary(
                    [float(row["cliffs_delta"]) for row in subset],
                    replicates=bootstrap_replicates, seed=0,
                ),
            }
    return rows, output


def diagnose_hp_near_far(fold_cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np
    methods = {
        "joint_metric": "metric_mean_distance",
        "explicit_joint_features": "explicit_joint_distance",
    }
    output: dict[str, Any] = {"top_k": 10, "methods": {}}
    for method, field in methods.items():
        selected_by_stratum: dict[str, list[float]] = {"hp_near": [], "hp_far": []}
        far_counts: list[int] = []
        for fold in fold_cells:
            for cells in _group_cells_by_query(fold):
                fingerprints = [str(cell["reference_joint_fingerprint"]) for cell in cells]
                selected = [
                    cells[index] for index in stable_top_k(
                        [float(cell[field]) for cell in cells], fingerprints, 10,
                    )
                ]
                far_count = sum(bool(cell["hp_far"]) for cell in selected)
                far_counts.append(far_count)
                for cell in selected:
                    selected_by_stratum["hp_far" if cell["hp_far"] else "hp_near"].append(
                        float(cell["performance_gap"])
                    )
        strata = {}
        for name, gaps in selected_by_stratum.items():
            _require(bool(gaps), f"{method} top-10 has no {name} neighbors")
            strata[name] = {
                "neighbor_count": len(gaps), "mean_raw_performance_gap": float(np.mean(gaps)),
                "median_raw_performance_gap": float(np.median(gaps)),
                "gap_le_0.006_rate": sum(gap <= 0.006 for gap in gaps) / len(gaps),
                "gap_le_0.012_rate": sum(gap <= 0.012 for gap in gaps) / len(gaps),
            }
        output["methods"][method] = {
            "strata": strata, "query_count": len(far_counts),
            "queries_with_at_least_1_hp_far_neighbor": sum(value >= 1 for value in far_counts),
            "queries_with_at_least_3_hp_far_neighbors": sum(value >= 3 for value in far_counts),
            "queries_with_at_least_5_hp_far_neighbors": sum(value >= 5 for value in far_counts),
        }
    return output


def diagnose_metric_seed_stability(fold_cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"metric_seeds": [0, 1, 2], "best_seed_selected": False, "seeds": {}}
    for seed in METRIC_SEEDS:
        distance_field = f"metric_seed_{seed}_distance"
        bin_medians = {}
        fold_bin_medians: dict[str, dict[str, float | None]] = {}
        retrieval = {}
        discrimination_rows = []
        for bin_name in GAP_BIN_ORDER:
            values = [
                float(cell[distance_field]) for fold in fold_cells for cell in fold["cells"]
                if cell["architecture_different"] and cell["hp_far"]
                and gap_bin_label(cell["performance_gap"]) == bin_name
            ]
            bin_medians[bin_name] = _quantile_summary(values)["median"]
        for fold in fold_cells:
            fold_bin_medians[str(fold["fold"])] = {}
            for bin_name in GAP_BIN_ORDER:
                values = [
                    float(cell[distance_field]) for cell in fold["cells"]
                    if cell["architecture_different"] and cell["hp_far"]
                    and gap_bin_label(cell["performance_gap"]) == bin_name
                ]
                fold_bin_medians[str(fold["fold"])][bin_name] = _quantile_summary(values)["median"]
        for delta in DELTA_CLOSES:
            for k in TOP_K_VALUES:
                query_rows = []
                for fold in fold_cells:
                    for cells in _group_cells_by_query(fold):
                        query_rows.append(compute_retrieval_for_query(
                            cells, distance_field=distance_field, k=k, delta_close=delta,
                        ))
                eligible = [row for row in query_rows if row["eligible"]]
                retrieval[f"delta{delta:.3f}_k{k}"] = {
                    "precision": sum(row["precision"] for row in query_rows) / len(query_rows),
                    "conditional_hit": (
                        sum(bool(row["hit"]) for row in eligible) / len(eligible) if eligible else None
                    ),
                }
        for fold in fold_cells:
            for cells in _group_cells_by_query(fold):
                cross = [cell for cell in cells if cell["architecture_different"] and cell["hp_far"]]
                close = [float(cell[distance_field]) for cell in cross if cell["performance_gap"] <= 0.012]
                far = [float(cell[distance_field]) for cell in cross if cell["performance_gap"] > 0.050]
                if close and far:
                    discrimination_rows.append(auc_and_cliffs_delta(close, far)["auc"])
        output["seeds"][str(seed)] = {
            "gap_bin_median_distances": bin_medians,
            "gap_bin_median_distances_by_fold": fold_bin_medians,
            "retrieval": retrieval,
            "close12_vs_far_auc": bootstrap_query_summary(discrimination_rows),
        }
    return output


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


def markdown_table(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]],
) -> str:
    materialized = list(rows) or [{key: "UNAVAILABLE" for key, _ in columns}]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format_cell(row.get(key)) for key, _ in columns) + " |"
        for row in materialized
    )
    return "\n".join(lines)


def _report_group(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return dict((payload.get("groups") or {}).get(key) or {})


def render_final_report(context: Mapping[str, Any]) -> str:
    classification = str(context.get("classification", "RUN_FAILED"))
    authority = dict(context.get("authority") or {})
    replay = dict(context.get("replay") or {})
    g3_rows = list(context.get("g3_rows") or [])
    gap_summary = dict(context.get("gap_summary") or {})
    neighbor_summary = dict(context.get("neighbor_summary") or {})
    retrieval = dict(context.get("retrieval") or {})
    ranks = dict(context.get("ranks") or {})
    discrimination = dict(context.get("discrimination") or {})
    hp_locality = dict(context.get("hp_locality") or {})
    seed_stability = dict(context.get("seed_stability") or {})
    environment = dict(context.get("environment") or {})
    reason = str(context.get("reason") or "NONE")

    low_rows = [row for row in g3_rows if row.get("group") == "bottom_20_low_gap"]
    high_rows = [row for row in g3_rows if row.get("group") == "top_20_high_gap"]
    input_table = [{
        "classification": (authority.get("identity") or {}).get("classification", "UNAVAILABLE"),
        "joint": (authority.get("identity") or {}).get("unique_joint_configurations", "UNAVAILABLE"),
        "architectures": (authority.get("identity") or {}).get("unique_architectures", "UNAVAILABLE"),
        "pools": (authority.get("identity") or {}).get("pool_count", "UNAVAILABLE"),
        "records": (authority.get("identity") or {}).get("seed_record_count", "UNAVAILABLE"),
        "split": (authority.get("identity") or {}).get("split_sha256", "UNAVAILABLE"),
    }]
    replay_table = [{
        "status": replay.get("status", "UNAVAILABLE"),
        "runs": len(replay.get("training_runs") or []),
        "atol": (replay.get("identity_gate") or {}).get("absolute_tolerance", "UNAVAILABLE"),
        "top5": "exact order" if replay.get("status") == "REPRODUCED" else "UNAVAILABLE",
        "GP": replay.get("GP_or_regressor_replayed", False),
        "test": replay.get("test_data_accessed", False),
    }]
    archived_geometry = authority.get("geometry") or {}
    original_g3_table = [{
        "median_rho": (authority.get("decision") or {}).get("geometry_gate", {}).get(
            "median_seed_fold_distance_gap_spearman", "UNAVAILABLE"
        ),
        "R_cross": archived_geometry.get("cross_factor_locality_ratio", "UNAVAILABLE"),
        "G3": (authority.get("decision") or {}).get("geometry_gate", {}).get(
            "checks", {}
        ).get("G3_cross_factor_locality", "UNAVAILABLE"),
        "metric_gap5": archived_geometry.get(
            "joint_metric_local_performance_gap_at_5", "UNAVAILABLE"
        ),
        "explicit_gap5": archived_geometry.get(
            "explicit_joint_feature_local_performance_gap_at_5", "UNAVAILABLE"
        ),
    }]
    bin_rows = []
    for method, payload in (gap_summary.get("by_method") or {}).items():
        for row in payload.get("pooled", []):
            bin_rows.append({"method": method, **row})
    neighbor_rows = []
    for key, payload in (neighbor_summary.get("groups") or {}).items():
        neighbor_rows.append({
            "group": key,
            "close06": (payload.get("gap_le_0.006_rate") or {}).get("mean"),
            "close12": (payload.get("gap_le_0.012_rate") or {}).get("mean"),
            "hp_far": (payload.get("hp_far_rate") or {}).get("mean"),
            "cross_close12": (payload.get("hp_far_and_gap_le_0.012_rate") or {}).get("mean"),
        })
    retrieval_rows = [{"group": key, **value} for key, value in (retrieval.get("groups") or {}).items()]
    rank_rows = [{"group": key, **value} for key, value in (ranks.get("groups") or {}).items()]
    discrimination_rows = []
    for key, payload in (discrimination.get("groups") or {}).items():
        discrimination_rows.append({
            "group": key, "eligible": payload.get("eligible_query_count"),
            "auc_mean": (payload.get("auc") or {}).get("mean"),
            "auc_ci": (payload.get("auc") or {}).get("mean_ci95"),
            "cliff_mean": (payload.get("cliffs_delta") or {}).get("mean"),
        })
    hp_rows = []
    for method, payload in (hp_locality.get("methods") or {}).items():
        for stratum, values in (payload.get("strata") or {}).items():
            hp_rows.append({"method": method, "stratum": stratum, **values})
    stability_rows = []
    for seed, payload in (seed_stability.get("seeds") or {}).items():
        stability_rows.append({
            "seed": seed,
            "precision5_close12": (payload.get("retrieval") or {}).get(
                "delta0.012_k5", {}
            ).get("precision"),
            "hit5_close12": (payload.get("retrieval") or {}).get(
                "delta0.012_k5", {}
            ).get("conditional_hit"),
            "auc_close12": (payload.get("close12_vs_far_auc") or {}).get("mean"),
            "bin_medians": payload.get("gap_bin_median_distances"),
        })
    interpretation = context.get("posthoc_interpretation") or {
        "status": "POSTHOC_INTERPRETATION", "choice": "D",
        "reason": "SPEC_UNDERDEFINED or diagnosis unavailable",
    }
    if low_rows:
        q1_fractions = [float(row["fraction_gap_le_0.012"]) for row in low_rows]
        q1_close06 = [float(row["fraction_gap_le_0.006"]) for row in low_rows]
        q1 = (
            "Q1: The unchanged G3 bottom quintile contains fold fractions "
            f"{min(q1_fractions):.3f}–{max(q1_fractions):.3f} (mean "
            f"{sum(q1_fractions) / len(q1_fractions):.3f}) at gap<=0.012 and mean "
            f"{sum(q1_close06) / len(q1_close06):.3f} at gap<=0.006. Because the prompt does "
            "not numerically define ‘mainly’, the report gives the exact fractions rather than "
            "inventing a yes/no cutoff."
        )
    else:
        q1 = "Q1: unavailable because the diagnosis did not pass both identity gates."
    q2_comparisons = []
    for delta in DELTA_CLOSES:
        for k in TOP_K_VALUES:
            joint = _report_group(retrieval, f"joint_metric_delta{delta:.3f}_k{k}")
            explicit = _report_group(retrieval, f"explicit_joint_features_delta{delta:.3f}_k{k}")
            if joint and explicit:
                q2_comparisons.append(
                    f"delta={delta:.3f},k={k}: precision {joint['precision_at_k']:.4f} vs "
                    f"{explicit['precision_at_k']:.4f}, conditional recall "
                    f"{joint['conditional_recall_at_k']:.4f} vs "
                    f"{explicit['conditional_recall_at_k']:.4f}, hit "
                    f"{joint['conditional_hit_at_k']:.4f} vs "
                    f"{explicit['conditional_hit_at_k']:.4f}"
                )
    q2 = (
        "Q2: Joint metric versus explicit features on identical cells — "
        + ("; ".join(q2_comparisons) if q2_comparisons else "unavailable before replay identity passes")
        + ". Higher values indicate easier retrieval; no new pass gate is inferred."
    )
    metric_hp = (hp_locality.get("methods") or {}).get("joint_metric", {})
    metric_strata = metric_hp.get("strata") or {}
    if metric_strata.get("hp_near") and metric_strata.get("hp_far"):
        near = metric_strata["hp_near"]
        far = metric_strata["hp_far"]
        q3 = (
            "Q3: In metric top-10, HP-near versus HP-far mean gaps are "
            f"{near['mean_raw_performance_gap']:.6f} and {far['mean_raw_performance_gap']:.6f}; "
            f"gap<=0.012 rates are {near['gap_le_0.012_rate']:.4f} and "
            f"{far['gap_le_0.012_rate']:.4f}. Queries with >=1/3/5 HP-far neighbors are "
            f"{metric_hp.get('queries_with_at_least_1_hp_far_neighbor')}/"
            f"{metric_hp.get('queries_with_at_least_3_hp_far_neighbors')}/"
            f"{metric_hp.get('queries_with_at_least_5_hp_far_neighbors')}."
        )
    else:
        q3 = "Q3: unavailable before replay identity passes."
    q3 += (
        " Architecture difference is 100% because the frozen outer split holds architecture "
        "groups out; it is not learned evidence."
    )
    q4 = (
        f"POSTHOC_INTERPRETATION={interpretation.get('choice', 'D')}: "
        f"{interpretation.get('reason', 'SPEC_UNDERDEFINED')}. This label is descriptive only."
    )
    sections = [
        ("Executive conclusion",
         f"The diagnosis decision is `{classification}`. {q1} {q2} {q3} {q4} "
         "No new GNN was trained and no test data or test metric was accessed.",
         markdown_table([{
             "diagnosis": classification, "historical": "SOFT_JOINT_METRIC_REJECTED",
             "modified": False, "posthoc": interpretation.get("choice", "D"),
             "reason": reason,
         }], (("diagnosis", "diagnosis"), ("historical", "historical class"),
              ("modified", "historical modified"), ("posthoc", "A/B/C/D"),
              ("reason", "run note")))),
        ("Input identity",
         "The completed soft-metric-v2 directory, archive sidecar, logical split hash, source "
         "snapshot root, cardinalities, validation-only target, and historical frozen gates are "
         "verified before encoder replay. Any mismatch stops with REPLAY_INPUT_IDENTITY_MISMATCH.",
         markdown_table(input_table, (("classification", "historical class"), ("joint", "joint configs"),
                                      ("architectures", "architectures"), ("pools", "pools"),
                                      ("records", "seed records"), ("split", "logical split SHA")))),
        ("Replay identity",
         "Only the 15 frozen CPU metric encoders are replayed. The archived inner split, pair "
         "construction, model seeds, best epoch, full outer-train refit, HP standardizer, and L2 "
         "normalization are used. Fifteen Spearman values, five G3 ratios, 336 ordered top-5 lists, "
         "and pooled Gap@5 must reproduce within 1e-8 before any diagnosis is calculated.",
         markdown_table(replay_table, (("status", "status"), ("runs", "encoder runs"),
                                      ("atol", "absolute tolerance"), ("top5", "top-5 check"),
                                      ("GP", "GP/regressor replayed"), ("test", "test accessed")))),
        ("Original G3 recap",
         "G1, G2, G4, M1, and M2 passed historically, while G3 required R_cross < 0.85 and failed. "
         "This posthoc analysis neither changes that gate nor recomputes GP or best-of-4 results.",
         markdown_table(original_g3_table, (("median_rho", "median distance-gap rho"),
                                            ("R_cross", "R_cross"), ("G3", "G3 passed"),
                                            ("metric_gap5", "metric Gap@5"),
                                            ("explicit_gap5", "explicit Gap@5")))),
        ("Bottom-20% performance-gap audit",
         q1 + " The high-gap quintile is included as the unchanged comparator. Fractions use raw "
         "validation seed-mean gaps and inclusive thresholds.",
         markdown_table(low_rows + high_rows, (("fold", "fold"), ("group", "group"),
                                               ("group_count", "cells"), ("minimum", "min gap"),
                                               ("median", "median gap"), ("maximum", "max gap"),
                                               ("fraction_gap_le_0.006", "fraction <=.006"),
                                               ("fraction_gap_le_0.012", "fraction <=.012")))),
        ("Absolute gap-bin distance analysis",
         "Fixed, preregistered absolute bins replace no historical gate. The table pools all five "
         "heldout-to-train cross-factor cells; the JSON also reports every fold and each metric seed. "
         "The number of strictly increasing adjacent median transitions is descriptive only.",
         markdown_table(bin_rows, (("method", "distance"), ("gap_bin", "gap bin"),
                                   ("count", "pairs"), ("median", "median distance"),
                                   ("q25", "q25"), ("q75", "q75")))),
        ("Top-5/top-10 neighbor composition",
         q3 + " Bootstrap intervals resample the 336 heldout queries 2,000 times with seed 0; "
         "neighbor pairs are never treated as independent replicates.",
         markdown_table(neighbor_rows, (("group", "method/k"), ("close06", "gap<=.006"),
                                        ("close12", "gap<=.012"), ("hp_far", "HP-far"),
                                        ("cross_close12", "HP-far & gap<=.012")))),
        ("Cross-factor precision/recall/hit",
         q2 + " Precision always divides by k. Recall and hit are conditional on a query having at "
         "least one relevant reference; no-opportunity queries are counted but excluded from those "
         "conditional denominators. Base prevalence uses all outer-train references.",
         markdown_table(retrieval_rows, (("group", "method/delta/k"),
                                         ("eligible_query_count", "eligible queries"),
                                         ("no_opportunity_query_count", "no opportunity"),
                                         ("precision_at_k", "precision"),
                                         ("conditional_recall_at_k", "conditional recall"),
                                         ("conditional_hit_at_k", "conditional hit"),
                                         ("precision_lift", "precision lift")))),
        ("Relevant-rank analysis",
         "All ranks are one-based and use distance followed by reference joint fingerprint as the "
         "deterministic tie-break. MRR is averaged across eligible queries from each query’s best "
         "relevant rank; no-opportunity queries are not assigned artificial ranks.",
         markdown_table(rank_rows, (("group", "method/delta"),
                                    ("eligible_query_count", "eligible queries"),
                                    ("median_best_rank", "median best rank"),
                                    ("mean_best_rank", "mean best rank"), ("MRR", "MRR"),
                                    ("relevant_enters_top_5_rate", "hit top-5"),
                                    ("relevant_enters_top_10_rate", "hit top-10")))),
        ("Distance AUC and Cliff's delta",
         "Within each eligible heldout query, AUC estimates P(distance_far > distance_close), with "
         "half credit for ties. Cliff’s delta subtracts the reverse probability. Close uses 0.006 or "
         "0.012; far is strictly above 0.050. Cross-query summaries use query-cluster bootstrap.",
         markdown_table(discrimination_rows, (("group", "method/comparison"),
                                              ("eligible", "eligible queries"),
                                              ("auc_mean", "mean AUC"),
                                              ("auc_ci", "AUC 95% CI"),
                                              ("cliff_mean", "mean Cliff delta")))),
        ("HP-near vs HP-far locality",
         q3 + " The table compares raw performance gaps in each method’s top-10 neighbors. Counts "
         "of queries having at least 1, 3, or 5 HP-far neighbors are retained in the JSON output.",
         markdown_table(hp_rows, (("method", "method"), ("stratum", "HP stratum"),
                                  ("neighbor_count", "neighbors"),
                                  ("mean_raw_performance_gap", "mean gap"),
                                  ("median_raw_performance_gap", "median gap"),
                                  ("gap_le_0.006_rate", "gap<=.006"),
                                  ("gap_le_0.012_rate", "gap<=.012")))),
        ("Metric-seed stability",
         "Seeds 0, 1, and 2 are reported separately for the key cross-factor diagnostics. The "
         "three matrices are still averaged for the primary diagnosis; no best initialization is "
         "selected and no seed-specific result can replace D_mean.",
         markdown_table(stability_rows, (("seed", "metric seed"),
                                         ("precision5_close12", "Precision@5 close12"),
                                         ("hit5_close12", "Hit@5 close12"),
                                         ("auc_close12", "AUC close12 vs far"),
                                         ("bin_medians", "gap-bin medians")))),
        ("Posthoc interpretation",
         q4 + " The prompt supplies no quantitative decision rule mapping the diagnostic statistics "
         "to A/B/C. Following SPEC_UNDERDEFINED, the program does not invent one and records D when "
         "a unique choice cannot be frozen from the protocol.",
         markdown_table([interpretation], (("status", "status"), ("choice", "choice"),
                                           ("reason", "reason")))),
        ("What this does NOT change",
         "The historical SOFT_JOINT_METRIC_REJECTED classification remains authoritative. This "
         "diagnosis cannot emit GP_READY, LOCALITY_READY, BO_READY, or any acquisition authorization. "
         "It does not change delta, loss, encoder, split, seeds, or any project mainline component.",
         markdown_table([{
             "historical": "SOFT_JOINT_METRIC_REJECTED", "modified": False,
             "new_GNN": False, "test": False, "GP_replay": False, "BO_authorized": False,
         }], (("historical", "historical class"), ("modified", "modified"),
              ("new_GNN", "new GNN"), ("test", "test accessed"),
              ("GP_replay", "GP replay"), ("BO_authorized", "BO authorized")))),
        ("Recommended next scientific question",
         "The next experiment design should preregister a quantitative criterion that distinguishes "
         "quantile-target misalignment from true cross-factor retrieval failure. This report supplies "
         "the fixed-threshold retrieval and rank evidence needed to formulate that question; it does "
         "not schedule seeds, train a new model, or authorize a method change.",
         markdown_table([{
             "question": "preregister cross-factor retrieval criterion", "authorized": False,
             "training_command": False, "method_change": False,
         }], (("question", "next question"), ("authorized", "authorized"),
              ("training_command", "training command"), ("method_change", "method change")))),
        ("Engineering audit",
         "The bundle records branch, HEAD, dirty-tree state, source/config hashes, deterministic CPU "
         "contract, elapsed time, logs, and exact source bytes. The archive SHA is computed after "
         "packaging and is authoritative in the sibling .sha256 sidecar.",
         markdown_table([{
             "branch": environment.get("branch", "UNAVAILABLE"),
             "HEAD": environment.get("head", "UNAVAILABLE"),
             "dirty": environment.get("dirty_tree", "UNAVAILABLE"),
             "elapsed": context.get("elapsed_seconds", "UNAVAILABLE"),
             "result": context.get("result_path", "UNAVAILABLE"),
             "snapshot": context.get("snapshot_manifest_root", "UNAVAILABLE"),
         }], (("branch", "branch"), ("HEAD", "HEAD"), ("dirty", "dirty tree"),
              ("elapsed", "formal seconds"), ("result", "result directory"),
              ("snapshot", "snapshot root")))),
    ]
    return "\n\n".join(
        ["# Cross-factor locality diagnosis v1"] + [
            f"## {index}. {title}\n\n{text}\n\n{table}"
            for index, (title, text, table) in enumerate(sections, start=1)
        ]
    ) + "\n"


def assert_complete_final_report(report: str) -> None:
    for index, title in enumerate(REPORT_SECTIONS, start=1):
        heading = f"## {index}. {title}"
        _require(heading in report, f"final report missing section: {heading}")
        start = report.index(heading)
        end = (
            report.find(f"## {index + 1}. ", start + 1)
            if index < len(REPORT_SECTIONS) else len(report)
        )
        table_lines = [line for line in report[start:end].splitlines() if line.startswith("|")]
        _require(len(table_lines) >= 3, f"final report section {index} lacks a data table")


def _git_capture(args: Sequence[str]) -> str:
    process = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if process.returncode != 0:
        raise ProtocolError(f"git {' '.join(args)} failed: {process.stdout.strip()}")
    return process.stdout


def write_git_and_environment(
    output: Path, *, metric_encoder_replay_started: bool = False,
) -> dict[str, Any]:
    branch = _git_capture(["branch", "--show-current"]).strip()
    head = _git_capture(["rev-parse", "HEAD"]).strip()
    status = _git_capture(["status", "--short", "--branch"])
    diff = _git_capture(["diff", "--binary", "--no-ext-diff", "HEAD", "--"])
    dump_text(output / "git_status.txt", status)
    dump_text(output / "git_diff.patch", diff)
    packages: dict[str, Any] = {}
    for distribution in ("PyYAML", "numpy", "scipy", "torch"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    environment = {
        "timestamp_utc": utc_now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(), "packages": packages,
        "branch": branch, "head": head, "dirty_tree": bool(status.splitlines()[1:]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_only": True, "gpu_used": False, "network_access_used": False,
        "candidate_gnn_training_started": False,
        "metric_encoder_replay_started": bool(metric_encoder_replay_started),
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
        inspected = inspect.getsourcefile(module)
        if inspected:
            path = Path(inspected)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(REPO_ROOT.resolve())
    except (FileNotFoundError, ValueError, TypeError):
        return None
    return resolved if resolved.suffix == ".py" else None


def snapshot_runtime_sources(
    output: Path, config: Mapping[str, Any], *, modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot_root = output / "source_snapshot"
    sources: dict[str, dict[str, Any]] = {}

    def include(relative: str, source: Path, reason: str, module_name: str | None = None) -> None:
        _require(source.is_file() and os.access(source, os.R_OK),
                 f"source snapshot input unreadable: {source}")
        entry = sources.get(relative)
        if entry is not None and Path(entry["source"]).read_bytes() != source.read_bytes():
            raise ProtocolError(f"conflicting source bytes for snapshot destination: {relative}")
        entry = sources.setdefault(relative, {"source": source, "reasons": [], "modules": []})
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)
        if module_name and module_name not in entry["modules"]:
            entry["modules"].append(module_name)

    for relative in config["source_snapshot"]["mandatory_new_files"]:
        include(str(relative), _repo_path(str(relative)), "mandatory_diagnosis_source")
    frozen_root = _repo_path(str(config["frozen_v2_runtime"]["snapshot_root"]))
    frozen_manifest = _read_json(frozen_root / "sha256_manifest.json")
    for entry in frozen_manifest["entries"]:
        relative = str(entry["repository_relative_path"])
        source = frozen_root / relative
        _require(source.is_file() and sha256_file(source) == str(entry["sha256"]),
                 f"frozen v2 source snapshot entry changed: {relative}")
        include(relative, source, "frozen_soft_v2_runtime_source")
    imported_records: list[dict[str, Any]] = []
    for module_name, module in sorted((modules or sys.modules).items()):
        source = _source_path_for_module(module)
        if source is None:
            continue
        try:
            frozen_relative = source.relative_to(frozen_root.resolve()).as_posix()
            destination_relative = frozen_relative
        except ValueError:
            destination_relative = source.relative_to(REPO_ROOT.resolve()).as_posix()
        include(destination_relative, source, "runtime_imported_local_python_module", module_name)
    manifest_entries = []
    for relative, metadata in sorted(sources.items()):
        source = Path(metadata["source"])
        destination = snapshot_root / relative
        atomic_write_bytes(destination, source.read_bytes())
        _require(sha256_file(destination) == sha256_file(source),
                 f"source snapshot copy hash mismatch: {relative}")
        record = {
            "repository_relative_path": relative, "absolute_runtime_path": str(source),
            "snapshot_relative_path": f"source_snapshot/{relative}",
            "size_bytes": source.stat().st_size, "sha256": sha256_file(source),
            "reason_included": sorted(metadata["reasons"]),
            "module_names": sorted(metadata["modules"]),
        }
        manifest_entries.append(record)
        if metadata["modules"]:
            imported_records.append(record)
    payload = {
        "format_version": FORMAT_VERSION, "repository_root": str(REPO_ROOT.resolve()),
        "entry_count": len(manifest_entries), "entries": manifest_entries,
        "manifest_root_sha256": stable_hash(manifest_entries),
    }
    dump_json(snapshot_root / "imported_local_modules.json", {
        "format_version": FORMAT_VERSION, "repository_root": str(REPO_ROOT.resolve()),
        "module_count": len(imported_records), "modules": imported_records,
    })
    dump_json(snapshot_root / "sha256_manifest.json", payload)
    verify_source_snapshot(output)
    return payload


def verify_source_snapshot(output: Path) -> dict[str, Any]:
    payload = _read_json(output / "source_snapshot/sha256_manifest.json")
    entries = list(payload.get("entries") or [])
    _require(stable_hash(entries) == payload.get("manifest_root_sha256"),
             "source snapshot manifest root mismatch")
    for entry in entries:
        path = output / str(entry["snapshot_relative_path"])
        _require(path.is_file() and path.stat().st_size == int(entry["size_bytes"])
                 and sha256_file(path) == str(entry["sha256"]),
                 f"source snapshot hash mismatch: {path}")
    present = {str(entry["repository_relative_path"]) for entry in entries}
    missing = sorted(set(REQUIRED_SOURCE_SNAPSHOT_PATHS) - present)
    _require(not missing, f"source snapshot required files missing: {missing}")
    return payload


def write_manifest(output: Path) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if "__pycache__" in path.parts or path.suffix in {
            ".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".onnx", ".joblib",
        }:
            continue
        files.append({
            "path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    dump_json(output / "manifest.json", {
        "format_version": FORMAT_VERSION, "generated_at_utc": utc_now(),
        "file_count": len(files), "files": files, "manifest_root_sha256": stable_hash(files),
    })


def create_archive(output: Path, archive: Path, sidecar: Path | None = None) -> str:
    _require(output.is_dir(), f"cannot archive missing output: {output}")
    _require(not archive.exists() and (sidecar is None or not sidecar.exists()),
             "archive or sidecar already exists")
    verify_source_snapshot(output)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")

    def filter_entry(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "__pycache__" in Path(info.name).parts or Path(info.name).name.endswith(
            (".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".onnx", ".joblib")
        ):
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


def ensure_formal_output_available(output: Path, *, preflight_ready: bool = False) -> None:
    if not output.exists():
        return
    _require(output.is_dir(), f"formal output path is not a directory: {output}")
    entries = [path.relative_to(output).as_posix() for path in output.rglob("*")]
    if not entries:
        _require(not preflight_ready, "runner unexpectedly supplied an empty preflight directory")
        return
    if preflight_ready:
        allowed = {
            "logs", "logs/py_compile.log", "logs/pytest_focused.log",
            "logs/synthetic_smoke.log", "logs/run.log",
        }
        unexpected = sorted(set(entries) - allowed)
        _require(not unexpected, f"formal preflight directory contains unexpected entries: {unexpected}")
        _require(all((output / name).is_file() for name in allowed if name != "logs"),
                 "formal preflight logs are incomplete")
        return
    raise ProtocolError(f"formal result directory exists and is non-empty: {output}")


ensure_output_policy = ensure_formal_output_available


def _initial_state(config_sha256: str, source_sha256: str) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION, "status": "running",
        "started_at_utc": utc_now(), "updated_at_utc": utc_now(),
        "config_sha256": config_sha256, "source_sha256": source_sha256,
        "decision": None, "formal_data_read": False,
        "metric_encoder_replay_started": False, "metric_encoder_replay_completed": False,
        "candidate_gnn_training_started": False, "regressor_training_started": False,
        "gp_fitting_started": False, "acquisition_used": False,
        "stabilization_data_accessed": False, "test_data_accessed": False,
    }


def _update_state(output: Path, state: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    updated = {**state, **changes, "updated_at_utc": utc_now()}
    dump_json(output / "state.json", updated)
    return updated


def _placeholder_payload(reason: str) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION, "status": "UNAVAILABLE", "reason": reason,
        "rows": [{"status": "UNAVAILABLE", "reason": reason}],
    }


def _write_missing_outputs(output: Path, reason: str) -> None:
    json_names = [name for name in REQUIRED_OUTPUTS if name.endswith(".json")]
    csv_names = [name for name in REQUIRED_OUTPUTS if name.endswith(".csv")]
    text_names = [name for name in REQUIRED_OUTPUTS if name.endswith((".md", ".txt", ".patch", ".log"))]
    for name in json_names:
        path = output / name
        if not path.exists() and name not in {
            "manifest.json", "source_snapshot/imported_local_modules.json",
            "source_snapshot/sha256_manifest.json", "decision.json", "state.json",
            "frozen_protocol.json",
        }:
            dump_json(path, _placeholder_payload(reason))
    for name in csv_names:
        path = output / name
        if not path.exists():
            dump_csv(path, [{"status": "UNAVAILABLE", "reason": reason}], ("status", "reason"))
    for name in text_names:
        path = output / name
        if not path.exists():
            dump_text(path, "" if name.startswith("logs/") or name == "git_diff.patch"
                      else f"UNAVAILABLE: {reason}\n")


def _fallback_environment(output: Path, reason: str) -> dict[str, Any]:
    environment = {
        "status": "UNAVAILABLE", "reason": reason, "branch": "UNAVAILABLE",
        "head": "UNAVAILABLE", "dirty_tree": "UNAVAILABLE", "cpu_only": True,
        "gpu_used": False, "candidate_gnn_training_started": False,
        "metric_encoder_replay_started": False, "regressor_training_started": False,
        "gp_fitting_started": False, "acquisition_used": False,
        "stabilization_data_accessed": False, "test_data_accessed": False,
    }
    dump_json(output / "environment.json", environment)
    if not (output / "git_status.txt").exists():
        dump_text(output / "git_status.txt", f"unavailable: {reason}\n")
    if not (output / "git_diff.patch").exists():
        dump_text(output / "git_diff.patch", "")
    return environment


def _finalize_report_snapshot(
    output: Path, config: Mapping[str, Any], context: dict[str, Any],
) -> dict[str, Any]:
    snapshot = snapshot_runtime_sources(output, config)
    context["snapshot_manifest_root"] = snapshot["manifest_root_sha256"]
    context["result_path"] = str(output)
    report = render_final_report(context)
    assert_complete_final_report(report)
    dump_text(output / "final_report.md", report)
    return snapshot


def _finalize_stopped_run(
    *, output: Path, config: Mapping[str, Any], state: Mapping[str, Any],
    classification: str, reason: str, environment: Mapping[str, Any],
    authority: Mapping[str, Any] | None = None,
) -> int:
    _write_missing_outputs(output, reason)
    if classification == "REPLAY_INPUT_IDENTITY_MISMATCH":
        dump_json(output / "input_integrity.json", {
            "status": "INVALID", "authoritative_archive_identity_status": (
                "VALID" if authority else "INVALID"
            ),
            "reason": reason, "test_data_accessed": False,
        })
        dump_json(output / "replay_identity.json", {
            "status": "INPUT_INVALID", "reason": reason,
            "diagnosis_executed": False, "test_data_accessed": False,
        })
    elif not (output / "replay_identity.json").exists():
        dump_json(output / "replay_identity.json", {
            "status": "MISMATCH",
            "reason": reason, "test_data_accessed": False,
        })
    decision = {**build_decision(classification), "reason": reason}
    dump_json(output / "decision.json", decision)
    snapshot = _finalize_report_snapshot(output, config, {
        "classification": classification, "reason": reason,
        "authority": dict(authority or {}), "replay": _read_json(output / "replay_identity.json"),
        "environment": dict(environment), "posthoc_interpretation": {
            "status": "POSTHOC_INTERPRETATION", "choice": "D",
            "reason": "diagnosis stopped at a frozen identity gate",
        },
    })
    _update_state(
        output, state, status="complete", decision=classification, reason=reason,
        source_snapshot_manifest_root=snapshot["manifest_root_sha256"],
        candidate_gnn_training_started=False, regressor_training_started=False,
        gp_fitting_started=False, acquisition_used=False,
        stabilization_data_accessed=False, test_data_accessed=False,
    )
    write_manifest(output)
    return 0


def write_failure_bundle(
    output_root: Path, error: BaseException | str, archive_path: Path | None = None,
    *, config_path: Path = DEFAULT_CONFIG, config_sha256: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    exception = error if isinstance(error, BaseException) else RuntimeError(str(error))
    message = str(exception)
    config_sha = config_sha256 or (sha256_file(config_path) if config_path.is_file() else "UNAVAILABLE")
    source_sha = source_sha256 or "UNAVAILABLE"
    try:
        config = load_config(config_path)
    except Exception as config_error:
        raise ProtocolError(f"cannot build source-complete failure bundle: {config_error}") from config_error
    _write_missing_outputs(output, message)
    dump_json(output / "error.json", {
        "classification": "RUN_FAILED", "error_type": type(exception).__name__,
        "message": message, "timestamp_utc": utc_now(),
        "traceback": traceback.format_exc() if sys.exc_info()[0] is not None else None,
    })
    try:
        environment = write_git_and_environment(output)
    except Exception as git_error:
        environment = _fallback_environment(output, f"{message}; git capture failed: {git_error}")
    if not (output / "frozen_protocol.json").exists():
        dump_json(output / "frozen_protocol.json", {
            "format_version": FORMAT_VERSION, "status": "failure_bundle",
            "config": config, "config_sha256": config_sha, "source_sha256": source_sha,
        })
    decision = {**build_decision("RUN_FAILED"), "reason": message}
    dump_json(output / "decision.json", decision)
    snapshot = _finalize_report_snapshot(output, config, {
        "classification": "RUN_FAILED", "reason": message, "environment": environment,
        "posthoc_interpretation": {
            "status": "POSTHOC_INTERPRETATION", "choice": "D", "reason": "run failed",
        },
    })
    state_path = output / "state.json"
    try:
        state = _read_json(state_path) if state_path.is_file() else _initial_state(config_sha, source_sha)
    except Exception:
        state = _initial_state(config_sha, source_sha)
    _update_state(
        output, state, status="failed", decision="RUN_FAILED", error=message,
        source_snapshot_manifest_root=snapshot["manifest_root_sha256"],
        candidate_gnn_training_started=False, regressor_training_started=False,
        gp_fitting_started=False, acquisition_used=False,
        stabilization_data_accessed=False, test_data_accessed=False,
    )
    complete = output / "COMPLETE"
    if complete.exists():
        complete.unlink()
    (output / "FAILED").touch()
    write_manifest(output)
    result: dict[str, Any] = {"output_root": str(output), "classification": "RUN_FAILED"}
    if archive_path is not None:
        digest = create_archive(output, Path(archive_path).resolve())
        result.update({"archive_path": str(Path(archive_path).resolve()), "archive_sha256": digest})
    return result


def _aggregate_source_sha(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _formal_source_paths(config: Mapping[str, Any]) -> list[Path]:
    frozen = _repo_path(str(config["frozen_v2_runtime"]["snapshot_root"]))
    return [
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/run_joint_configuration_cross_factor_locality_diagnosis.sh",
        REPO_ROOT / "tests/test_joint_configuration_cross_factor_locality_diagnosis.py",
        frozen / "configs/joint_configuration_soft_metric_gate_v2.yaml",
        frozen / "analyse/joint_configuration_soft_metric_gate.py",
        frozen / "analyse/joint_configuration_metric_gate.py",
        frozen / "analyse/performance_metric_quick_gate.py",
        frozen / "analyse/joint_pair_power_projection_audit.py",
    ]


def _interpretation_payload() -> dict[str, Any]:
    return {
        "status": "POSTHOC_INTERPRETATION", "choice": "D",
        "reason": (
            "SPEC_UNDERDEFINED: the protocol freezes the diagnostics but supplies no quantitative "
            "rule that uniquely maps them to A, B, or C; D records that no unique choice is authorized"
        ),
        "historical_classification_modified": False,
    }


def execute_formal(
    *, config_path: Path, output: Path, config_sha256: str, source_sha256: str,
    preflight_ready: bool,
) -> int:
    started = time.monotonic()
    config = load_config(config_path)
    _require(sha256_file(config_path) == config_sha256, "runner config SHA-256 mismatch")
    expected_source_sha = _aggregate_source_sha(_formal_source_paths(config))
    _require(expected_source_sha == source_sha256, "runner source SHA-256 mismatch")
    ensure_formal_output_available(output, preflight_ready=preflight_ready)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    state = _initial_state(config_sha256, source_sha256)
    dump_json(output / "state.json", state)
    dump_json(output / "frozen_protocol.json", {
        "format_version": FORMAT_VERSION, "config": config,
        "config_path": str(config_path.resolve()), "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "frozen_scientific_runtime": "soft-v2 source_snapshot exact bytes",
        "replayed_components": ["metric_encoder"],
        "forbidden_components_replayed": [], "candidate_gnn_training": False,
        "regressor_training": False, "gp_fitting": False, "acquisition": False,
        "stabilization_data_access": False, "test_data_access": False,
    })
    try:
        environment = write_git_and_environment(output)
    except Exception as error:
        environment = _fallback_environment(output, str(error))
    try:
        authority = validate_authoritative_input(config)
    except ReplayInputIdentityMismatch as error:
        return _finalize_stopped_run(
            output=output, config=config, state=state,
            classification="REPLAY_INPUT_IDENTITY_MISMATCH", reason=str(error),
            environment=environment,
        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        return _finalize_stopped_run(
            output=output, config=config, state=state,
            classification="REPLAY_INPUT_IDENTITY_MISMATCH",
            reason=f"authoritative identity validation failed: {error}",
            environment=environment,
        )
    dump_json(output / "input_integrity.json", {
        "status": "VALID", "identity": authority["identity"], "checks": authority["checks"],
        "historical_classification_modified": False,
        "candidate_gnn_training_started": False, "test_data_accessed": False,
    })
    state = _update_state(output, state, formal_data_read=True)
    try:
        state = _update_state(output, state, metric_encoder_replay_started=True)
        environment["metric_encoder_replay_started"] = True
        dump_json(output / "environment.json", environment)
        _configurations, fold_replays, replay_audit = replay_frozen_metric_encoders(
            config, authority,
        )
        dump_json(output / "replay_identity.json", replay_audit)
        state = _update_state(
            output, state, metric_encoder_replay_completed=True,
            metric_encoder_replay_count=15,
        )
    except ReplayInputIdentityMismatch as error:
        return _finalize_stopped_run(
            output=output, config=config, state=state,
            classification="REPLAY_INPUT_IDENTITY_MISMATCH", reason=str(error),
            environment=environment, authority=authority,
        )
    except ReplayIdentityMismatch as error:
        dump_json(output / "replay_identity.json", {
            "status": "MISMATCH", "reason": str(error),
            "absolute_tolerance": config["replay"]["identity_gate"]["absolute_tolerance"],
            "diagnosis_executed": False, "test_data_accessed": False,
        })
        return _finalize_stopped_run(
            output=output, config=config, state=state,
            classification="REPLAY_IDENTITY_MISMATCH", reason=str(error),
            environment=environment, authority=authority,
        )

    fold_cells = build_cross_factor_cells(fold_replays)
    g3_rows = diagnose_g3_quantile_alignment(fold_cells)
    gap_rows, gap_summary = diagnose_gap_bins(fold_cells)
    neighbor_rows, neighbor_summary = diagnose_neighbor_composition(fold_cells)
    retrieval_rows, retrieval = diagnose_retrieval(fold_cells)
    ranks = diagnose_ranks(fold_cells)
    discrimination_rows, discrimination = diagnose_distance_discrimination(fold_cells)
    hp_locality = diagnose_hp_near_far(fold_cells)
    seed_stability = diagnose_metric_seed_stability(fold_cells)
    dump_csv(output / "g3_quantile_alignment.csv", g3_rows)
    dump_csv(output / "cross_factor_gap_bins.csv", gap_rows)
    dump_json(output / "cross_factor_gap_bin_summary.json", gap_summary)
    dump_csv(output / "neighbor_composition_per_query.csv", neighbor_rows)
    dump_json(output / "neighbor_composition_summary.json", neighbor_summary)
    dump_csv(output / "cross_factor_retrieval_per_query.csv", retrieval_rows)
    dump_json(output / "cross_factor_retrieval_metrics.json", retrieval)
    dump_json(output / "cross_factor_rank_retrieval.json", ranks)
    dump_csv(output / "distance_discrimination_per_query.csv", discrimination_rows)
    dump_json(output / "distance_discrimination.json", discrimination)
    dump_json(output / "hp_near_far_locality.json", hp_locality)
    dump_json(output / "metric_seed_stability.json", seed_stability)
    interpretation = _interpretation_payload()
    decision = {
        **build_decision("CROSS_FACTOR_DIAGNOSIS_COMPLETE"),
        "replay_identity_passed": True, "diagnostic_only": True,
        "posthoc_interpretation": interpretation,
    }
    dump_json(output / "decision.json", decision)
    elapsed = time.monotonic() - started
    snapshot = _finalize_report_snapshot(output, config, {
        "classification": "CROSS_FACTOR_DIAGNOSIS_COMPLETE", "authority": authority,
        "replay": replay_audit, "g3_rows": g3_rows, "gap_summary": gap_summary,
        "neighbor_summary": neighbor_summary, "retrieval": retrieval, "ranks": ranks,
        "discrimination": discrimination, "hp_locality": hp_locality,
        "seed_stability": seed_stability, "environment": environment,
        "posthoc_interpretation": interpretation, "elapsed_seconds": elapsed,
    })
    _update_state(
        output, state, status="complete", decision="CROSS_FACTOR_DIAGNOSIS_COMPLETE",
        formal_elapsed_seconds=elapsed,
        source_snapshot_manifest_root=snapshot["manifest_root_sha256"],
        historical_soft_metric_classification="SOFT_JOINT_METRIC_REJECTED",
        historical_classification_modified=False,
        candidate_gnn_training_started=False, regressor_training_started=False,
        gp_fitting_started=False, acquisition_used=False,
        stabilization_data_accessed=False, test_data_accessed=False,
    )
    write_manifest(output)
    return 0


def dependency_inventory() -> dict[str, bool]:
    output: dict[str, bool] = {}
    for name in ("yaml", "numpy", "scipy", "sklearn", "torch"):
        try:
            __import__(name)
            output[name] = True
        except ImportError:
            output[name] = False
    return output


def execute_dry_run(
    *, formal_output: Path = DEFAULT_OUTPUT, config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    payload = {
        "protocol_version": config["protocol_version"], "dry_run": True,
        "cpu_only": True, "formal_data_read": False,
        "formal_result_root_created": False, "formal_replay_executed": False,
        "metric_encoder_replay_started": False, "candidate_gnn_training_started": False,
        "regressor_training_started": False, "gp_fitting_started": False,
        "acquisition_used": False, "stabilization_data_accessed": False,
        "test_data_accessed": False, "configured_output": str(formal_output),
        "configured_output_exists": formal_output.exists(), "dependencies": dependency_inventory(),
        "formal_scope": [
            "validate frozen soft-v2 identity", "replay exactly 15 frozen metric encoders",
            "enforce replay identity gate", "calculate read-only cross-factor diagnostics",
            "write source-complete bundle",
        ],
    }
    _require(not formal_output.exists(), "dry-run target already exists; formal run would refuse overwrite")
    return payload


def synthetic_smoke() -> dict[str, Any]:
    bins = [assign_absolute_gap_bin(value) for value in (0.006, 0.012, 0.025, 0.050, 0.051)]
    retrieval = retrieval_metrics_for_query(
        distances=[0.1, 0.2, 0.3, 0.4, 0.5],
        reference_fingerprints=["a", "b", "c", "d", "e"],
        performance_gaps=[0.005, 0.020, 0.004, 0.080, 0.090],
        hp_far=[True, True, False, True, True],
        architecture_different=[True] * 5, delta_close=0.006, k=5,
    )
    discrimination = auc_and_cliffs_delta([0.1, 0.2], [0.8, 0.9])
    replay_fixture = {
        "distance_gap_spearman": {"fold_0": {"seed_0": 0.2}},
        "cross_factor_locality_ratio": {"fold_0": 0.96},
        "metric_top5_fingerprints": {"q": ["a", "b", "c", "d", "e"]},
        "pooled_metric_gap_at_5": 0.06,
    }
    checks = {
        "gap_bin_boundaries": bins == [
            "gap_le_0.006", "gap_0.006_0.012", "gap_0.012_0.025",
            "gap_0.025_0.050", "gap_gt_0.050",
        ],
        "retrieval_definition": retrieval["relevant_count"] == 1
        and retrieval["precision"] == 0.2,
        "distance_discrimination": discrimination == {"auc": 1.0, "cliffs_delta": 1.0},
        "replay_identity": assert_replay_identity(replay_fixture, replay_fixture)["passed"],
        "historical_classification_immutable": build_decision(
            "CROSS_FACTOR_DIAGNOSIS_COMPLETE"
        )["historical_classification_modified"] is False,
    }
    return {
        "passed": all(checks.values()), **checks, "formal_data_read": False,
        "formal_replay_executed": False, "metric_encoder_replay_started": False,
        "candidate_gnn_training_started": False, "test_data_accessed": False,
    }


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
        print(json.dumps(execute_dry_run(formal_output=output, config_path=config_path),
                         indent=2, sort_keys=True))
        return 0
    if args.synthetic_smoke:
        payload = synthetic_smoke()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 1
    config_sha = args.config_sha256 or sha256_file(config_path)
    if args.source_sha256:
        source_sha = args.source_sha256
    else:
        source_sha = _aggregate_source_sha(_formal_source_paths(load_config(config_path)))
    if args.refresh_manifest:
        write_manifest(output)
        return 0
    if args.write_failure_bundle is not None:
        write_failure_bundle(
            output, str(args.write_failure_bundle), config_path=config_path,
            config_sha256=config_sha, source_sha256=source_sha,
        )
        return 0
    if args.formal:
        try:
            return execute_formal(
                config_path=config_path, output=output, config_sha256=config_sha,
                source_sha256=source_sha, preflight_ready=bool(args.preflight_ready),
            )
        except Exception as error:
            try:
                write_failure_bundle(
                    output, error, config_path=config_path, config_sha256=config_sha,
                    source_sha256=source_sha,
                )
            except Exception as bundle_error:
                print(f"formal diagnosis failed: {error}", file=sys.stderr)
                print(f"failure bundle also failed: {bundle_error}", file=sys.stderr)
            raise
    raise ProtocolError("no execution mode selected")


if __name__ == "__main__":
    raise SystemExit(main())
