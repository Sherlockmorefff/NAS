#!/usr/bin/env python3
"""Audit Original Frozen Head ensemble uncertainty and selective risk.

The ``prepare`` command performs every operation that is allowed to see model
predictions but not external validation labels.  It reconstructs strict
architecture-level OOF membership for development-80, computes the single
preregistered conformal calibration, deterministically recomputes the pure
member-prediction sample standard deviation (ddof=1), freezes external
uncertainty tables and random-rejection permutations, and writes a hash-bound
analysis plan.

The ``evaluate`` command first verifies that frozen plan, then and only then
connects clean-24 and new-48 validation labels and performs the preregistered
metrics and bootstrap.  No model fitting, VAE update, GNN training, GP, test
metric, or BO path exists in this module.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score, roc_auc_score
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse.h_targeted_reliable_pair_validation import sha256_file
from analyse.performance_aware_vae_strict_regression import StrictPerformanceHead
from architecture_diagnostics import config_to_igraph, load_joint_vae


FORMAT_VERSION = "original_head_uncertainty_selective_risk_audit_global4_v1"
OUTPUT_RELATIVE = "results/posthoc/original-head-uncertainty-selective-risk-audit-global4-v1"
STRICT_RELATIVE = "results/posthoc/performance-aware-vae-strict-regression-global4-v2"
STRUCTURE_RELATIVE = "results/posthoc/structure-native-surrogate-pilot-global4-v1"
NEW48_RELATIVE = "results/posthoc/h-vs-head-model-blind-prospective-global4-v1"
BASE_CHECKPOINT = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
BOOTSTRAP_SEED = 20260831
RANDOM_REJECTION_SEED = 20260832
BOOTSTRAP_REPLICATES = 10_000
RANDOM_REJECTION_DRAWS = 10_000
EPSILON = 1e-8
EXCLUSION_RATES = (0.10, 0.25, 0.50)
INTERVAL_LEVELS = (0.90, 0.95)
TOP_K_VALUES = (1, 3, 5, 10)
GAP_THRESHOLDS = (0.006, 0.012, 0.020)
MEAN_REPRODUCTION_TOLERANCE = 1e-7
FORBIDDEN_EXTERNAL_KEYS = (
    "val", "validation", "test", "label", "accuracy", "acc", "reward", "error",
)
CLASSIFICATION_ORDER = (
    "INPUT_OR_IMPLEMENTATION_ERROR",
    "HEAD_OOF_UNCERTAINTY_UNAVAILABLE",
    "LABEL_FIREWALL_VIOLATION",
    "PROTOCOL_MISMATCH",
    "INPUT_IDENTITY_MISMATCH",
    "HEAD_UNCERTAINTY_VALIDATED_FOR_ACQUISITION_PILOT",
    "HEAD_UNCERTAINTY_REJECTED",
    "HEAD_UNCERTAINTY_SIGNAL_ONLY",
    "HEAD_UNCERTAINTY_UNDERPOWERED",
)


class InputIdentityMismatch(RuntimeError):
    """Raised when a frozen plan or one of its bound inputs has drifted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple)) else value
                    for key, value in ((field, row.get(field)) for field in fields)
                })
    temporary.replace(path)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_sidecar(path: Path) -> Path:
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def verify_sidecar(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"missing SHA-256 sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if expected != sha256_file(path):
        raise RuntimeError(f"SHA-256 sidecar mismatch: {path}")


def identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "tests/test_original_head_uncertainty_selective_risk_audit.py",
        REPO_ROOT / "architecture_diagnostics.py",
        REPO_ROOT / "analyse/performance_aware_vae_strict_regression.py",
        REPO_ROOT / "analyse/h_targeted_reliable_pair_validation.py",
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths if path.is_file()}


def contains_forbidden_external_key(value: Any, path: str = "root") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_EXTERNAL_KEYS):
                hits.append(f"{path}.{key}")
            hits.extend(contains_forbidden_external_key(nested, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            hits.extend(contains_forbidden_external_key(nested, f"{path}[{index}]"))
    return hits


def finite_conformal_quantile(values: Sequence[float], coverage: float) -> tuple[float, int]:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("conformal scores must be nonempty and finite")
    rank = min(int(array.size), math.ceil((array.size + 1) * coverage))
    return float(array[rank - 1]), rank


def member_prediction_sample_std(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("member predictions must contain at least two finite values")
    return float(array.std(ddof=1))


def frozen_random_permutations(count: int, draws: int, seed: int) -> np.ndarray:
    if count < 2 or draws < 1:
        raise ValueError("random rejection requires at least two candidates and one draw")
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(count) for _ in range(draws)]).astype(np.uint16)


def _strict_split_records(strict_root: Path) -> dict[str, dict[str, Any]]:
    records = [
        row for row in load_json(strict_root / "fixed_split_manifest.json")
        if row["dataset_variant"] == "full" and row["protocol"] == "group_repeated"
    ]
    if len(records) != 50 or len({row["split_id"] for row in records}) != 50:
        raise RuntimeError("strict full/group_repeated split manifest must contain 50 unique splits")
    for row in records:
        train = set(row["train_architecture_fingerprints"])
        validation = set(row["validation_architecture_fingerprints"])
        if train & validation or int(row.get("leakage_count", -1)) != 0:
            raise RuntimeError(f"architecture leakage in strict split {row['split_id']}")
    return {str(row["split_id"]): dict(row) for row in records}


def head_member_predictions(
    candidates: Sequence[Mapping[str, Any]], strict_root: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the 150 raw member predictions without any calibration variance."""
    if not candidates:
        raise ValueError("candidate list is empty")
    split_records = _strict_split_records(strict_root)
    model = load_joint_vae(BASE_CHECKPOINT, "global4", device="cpu")
    graphs = [config_to_igraph(row) for row in candidates]
    with torch.no_grad():
        mu, _ = model.arch_vae.encode(graphs)
    mu_array = mu.detach().cpu().numpy().astype(np.float32)
    hp = np.asarray([row["raw_hp_vector"] for row in candidates], dtype=np.float32)
    paths = sorted(
        (strict_root / "checkpoints/full/group_repeated").glob(
            "group_*/seed_*/frozen_head/checkpoint.pth"
        )
    )
    if len(paths) != 150:
        raise RuntimeError(f"expected 150 frozen Head checkpoints, found {len(paths)}")
    predictions: list[np.ndarray] = []
    members: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        task = payload["task"]
        if (
            task["dataset_variant"] != "full"
            or task["protocol"] != "group_repeated"
            or task["model"] != "frozen_head"
            or payload["arch_vae_state"] is not None
            or int(payload["head_input_dim"]) != 16
        ):
            raise RuntimeError(f"unexpected frozen Head checkpoint schema: {path}")
        split_id = str(task["split_id"])
        seed = int(task["training_seed"])
        if split_id not in split_records:
            raise RuntimeError(f"checkpoint references unknown split: {path}")
        member_id = f"{split_id}__seed{seed}"
        scaling = payload["scaling"]
        hp_scaled = (hp - np.asarray(scaling["hp_mean"], dtype=np.float32)) / np.asarray(
            scaling["hp_std"], dtype=np.float32
        )
        features = torch.tensor(np.concatenate([mu_array, hp_scaled], axis=1), dtype=torch.float32)
        head = StrictPerformanceHead(16)
        head.load_state_dict(payload["head_state"])
        head.eval()
        with torch.no_grad():
            normalized = head(features).numpy()
        member_prediction = (
            normalized * float(scaling["y_std"]) + float(scaling["y_mean"])
        ).astype(np.float64)
        predictions.append(member_prediction)
        members.append({
            "member_id": member_id,
            "split_id": split_id,
            "training_seed": seed,
            "trained_architecture_fingerprints": list(
                split_records[split_id]["train_architecture_fingerprints"]
            ),
        })
        checkpoints.append({
            "member_id": member_id,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "head_input_dim": 16,
            "feature_schema": ["mu_arch", "train_fold_standardized_global4_hp"],
            "arch_vae_state_is_none": True,
        })
    if len({row["member_id"] for row in members}) != 150:
        raise RuntimeError("frozen Head member IDs are not unique")
    return np.stack(predictions, axis=0), members, checkpoints


def _prior_development_head_predictions(structure_root: Path) -> dict[str, dict[str, Any]]:
    rows = [
        row for row in load_json(structure_root / "models/development_oof_predictions.json")
        if row["model"] == "original_frozen_head"
    ]
    if len(rows) != 96:
        raise RuntimeError("prior development Head OOF artifact must contain 96 rows")
    return {str(row["config_id"]): dict(row) for row in rows}


def build_development_oof(
    development: Sequence[Mapping[str, Any]], matrix: np.ndarray,
    members: Sequence[Mapping[str, Any]], strict_root: Path, structure_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if len(development) != 96 or len({row["architecture_fingerprint"] for row in development}) != 80:
        raise RuntimeError("development manifest must contain 96 configurations / 80 architectures")
    if matrix.shape != (150, len(development)):
        raise RuntimeError("development member matrix has unexpected shape")
    prior = _prior_development_head_predictions(structure_root)
    raw = [
        row for row in load_csv(strict_root / "all_validation_predictions.csv")
        if row["dataset_variant"] == "full"
        and row["protocol"] == "group_repeated"
        and row["model"] == "frozen_head"
    ]
    raw_lookup = {
        (str(row["config_id"]), f"{row['split_id']}__seed{int(row['training_seed'])}"):
        float(row["prediction"])
        for row in raw
    }
    member_index = {str(row["member_id"]): index for index, row in enumerate(members)}
    eligible_by_fp: dict[str, list[str]] = {}
    for fingerprint in sorted({str(row["architecture_fingerprint"]) for row in development}):
        eligible_by_fp[fingerprint] = [
            str(member["member_id"]) for member in members
            if fingerprint not in set(member["trained_architecture_fingerprints"])
        ]
        if len(eligible_by_fp[fingerprint]) < 2:
            raise RuntimeError(f"strict OOF uncertainty unavailable for architecture {fingerprint}")
    output: list[dict[str, Any]] = []
    raw_mismatches: list[dict[str, Any]] = []
    mean_errors: list[float] = []
    fp_member_sets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for config_index, source in enumerate(development):
        config_id = str(source["config_id"])
        fingerprint = str(source["architecture_fingerprint"])
        eligible = eligible_by_fp[fingerprint]
        fp_member_sets[fingerprint].add(tuple(eligible))
        values = np.asarray([matrix[member_index[item], config_index] for item in eligible], dtype=np.float64)
        prediction = float(values.mean())
        sigma = member_prediction_sample_std(values)
        prior_row = prior[config_id]
        mean_error = abs(prediction - float(prior_row["prediction"]))
        mean_errors.append(mean_error)
        if source["development_source"] == "unified_60":
            raw_ids = {
                member_id for (row_config_id, member_id) in raw_lookup if row_config_id == config_id
            }
            if raw_ids != set(eligible):
                raw_mismatches.append({
                    "config_id": config_id,
                    "reason": "raw OOF member identity mismatch",
                    "expected_count": len(eligible),
                    "observed_count": len(raw_ids),
                })
            else:
                maximum = max(
                    abs(raw_lookup[(config_id, member_id)] - matrix[member_index[member_id], config_index])
                    for member_id in eligible
                )
                if maximum > MEAN_REPRODUCTION_TOLERANCE:
                    raw_mismatches.append({
                        "config_id": config_id,
                        "reason": "raw OOF prediction mismatch",
                        "maximum_absolute_error": maximum,
                    })
        y = float(source["validation_mean"])
        abs_error = abs(y - prediction)
        output.append({
            "config_id": config_id,
            "architecture_fingerprint": fingerprint,
            "development_source": source["development_source"],
            "validation_mean": y,
            "prediction": prediction,
            "prediction_source_count": len(eligible),
            "member_prediction_sample_std_ddof1": sigma,
            "uncertainty_definition": "sample standard deviation of eligible member prediction means; ddof=1",
            "absolute_error": abs_error,
            "standardized_absolute_residual": abs_error / (sigma + EPSILON),
            "eligible_member_ids_sha256": stable_hash(eligible),
        })
    inconsistent_fp = [fingerprint for fingerprint, values in fp_member_sets.items() if len(values) != 1]
    maximum_mean_error = max(mean_errors)
    membership_rows = [
        {
            "architecture_fingerprint": fingerprint,
            "eligible_member_count": len(member_ids),
            "eligible_member_ids": member_ids,
            "no_eligible_member_trained_on_architecture": all(
                fingerprint not in set(members[member_index[item]]["trained_architecture_fingerprints"])
                for item in member_ids
            ),
        }
        for fingerprint, member_ids in sorted(eligible_by_fp.items())
    ]
    count_distribution = Counter(row["prediction_source_count"] for row in output)
    audit = {
        "passed": (
            not raw_mismatches
            and not inconsistent_fp
            and maximum_mean_error <= MEAN_REPRODUCTION_TOLERANCE
            and all(row["no_eligible_member_trained_on_architecture"] for row in membership_rows)
        ),
        "classification_if_failed": "HEAD_OOF_UNCERTAINTY_UNAVAILABLE",
        "configuration_count": len(output),
        "architecture_count": len(eligible_by_fp),
        "eligible_member_count_min": min(len(values) for values in eligible_by_fp.values()),
        "eligible_member_count_max": max(len(values) for values in eligible_by_fp.values()),
        "configuration_member_count_distribution": dict(sorted(count_distribution.items())),
        "same_architecture_all_hp_share_member_set": not inconsistent_fp,
        "architecture_member_set_inconsistencies": inconsistent_fp,
        "raw_oof_mismatches": raw_mismatches,
        "maximum_prior_oof_mean_reproduction_error": maximum_mean_error,
        "mean_reproduction_tolerance": MEAN_REPRODUCTION_TOLERANCE,
        "in_sample_prediction_used": False,
        "member_calibration_std_used": False,
    }
    return output, audit, membership_rows


def calibration_from_development(
    development_oof: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scores = [float(row["standardized_absolute_residual"]) for row in development_oof]
    residuals = [float(row["absolute_error"]) for row in development_oof]
    quantiles: dict[str, Any] = {
        "calibration_configuration_count": len(development_oof),
        "epsilon": EPSILON,
        "score_definition": "abs(y-mu)/(member_prediction_sample_std_ddof1+1e-8)",
        "finite_sample_rule": "rank=min(n,ceil((n+1)*coverage)), one-indexed order statistic",
        "external_labels_used": False,
    }
    baseline: dict[str, Any] = {
        "calibration_configuration_count": len(development_oof),
        "definition": "constant symmetric half width from development OOF absolute residuals",
        "external_labels_used": False,
    }
    for coverage in INTERVAL_LEVELS:
        suffix = str(int(coverage * 100))
        q, rank = finite_conformal_quantile(scores, coverage)
        half_width, residual_rank = finite_conformal_quantile(residuals, coverage)
        quantiles[f"q{suffix}"] = q
        quantiles[f"q{suffix}_rank"] = rank
        baseline[f"half_width_{suffix}"] = half_width
        baseline[f"half_width_{suffix}_rank"] = residual_rank
    quantiles["calibration_scores_sha256"] = stable_hash(scores)
    baseline["absolute_residuals_sha256"] = stable_hash(residuals)
    return quantiles, baseline


def _head_prediction_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    selected = [row for row in rows if row["model"] == "original_frozen_head"]
    return {str(row["config_id"]): dict(row) for row in selected}


def build_external_uncertainty(
    clean: Sequence[Mapping[str, Any]], new48: Sequence[Mapping[str, Any]],
    matrix: np.ndarray, member_count: int, structure_root: Path, new48_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if matrix.shape != (member_count, len(clean) + len(new48)) or member_count != 150:
        raise RuntimeError("external member prediction matrix has unexpected shape")
    clean_prior = _head_prediction_lookup(
        load_json(structure_root / "phase_4/frozen_candidate_predictions.json")
    )
    new_prior = _head_prediction_lookup(load_json(new48_root / "phase_a/frozen_predictions.json"))
    output: list[dict[str, Any]] = []
    mean_errors: dict[str, list[float]] = defaultdict(list)
    legacy_differences: list[float] = []
    legacy_ratios: list[float] = []
    frozen_identity_failures: list[str] = []
    for index, (role, source) in enumerate(
        [("clean24", row) for row in clean] + [("new48", row) for row in new48]
    ):
        config_id = str(source["config_id"])
        prior = clean_prior[config_id] if role == "clean24" else new_prior[config_id]
        expected_flag = (
            prior.get("predictions_frozen_before_prospective_labels")
            if role == "clean24" else prior.get("prediction_frozen_before_new_labels")
        )
        if int(prior.get("prediction_source_count", -1)) != 150 or expected_flag is not True:
            frozen_identity_failures.append(config_id)
        values = matrix[:, index].astype(np.float64)
        prediction = float(values.mean())
        sigma = member_prediction_sample_std(values)
        error = abs(prediction - float(prior["prediction"]))
        mean_errors[role].append(error)
        legacy = float(prior["prediction_std"]) if role == "new48" else None
        if legacy is not None:
            legacy_differences.append(legacy - sigma)
            legacy_ratios.append(legacy / sigma if sigma > 0 else math.inf)
        output.append({
            "dataset_role": role,
            "config_id": config_id,
            "architecture_fingerprint": str(source["architecture_fingerprint"]),
            "hp_fingerprint": str(source["hp_fingerprint"]),
            "structure_stratum": source.get("structure_stratum", source.get("selection_stratum")),
            "prediction": float(prior["prediction"]),
            "recomputed_member_prediction_mean": prediction,
            "prediction_mean_reproduction_delta_abs": error,
            "prediction_source_count": 150,
            "member_prediction_sample_std_ddof1": sigma,
            "legacy_total_predictive_std": legacy,
            "legacy_std_definition": (
                "sqrt(mean(member_calibration_std^2 + member_prediction^2) - mean_prediction^2)"
                if legacy is not None else None
            ),
            "primary_uncertainty_definition": "sample standard deviation of 150 member prediction means; ddof=1",
            "member_predictions_sha256": stable_hash(values.tolist()),
            "prediction_frozen_before_external_truth_join": True,
        })
    audit = {
        "passed": (
            len(output) == 72
            and len({row["config_id"] for row in output}) == 72
            and not frozen_identity_failures
            and max(error for values in mean_errors.values() for error in values)
            <= MEAN_REPRODUCTION_TOLERANCE
        ),
        "classification_if_failed": "PROTOCOL_MISMATCH",
        "clean24_configuration_count": len(clean),
        "new48_configuration_count": len(new48),
        "member_count": member_count,
        "clean24_maximum_mean_reproduction_error": max(mean_errors["clean24"]),
        "new48_maximum_mean_reproduction_error": max(mean_errors["new48"]),
        "mean_reproduction_tolerance": MEAN_REPRODUCTION_TOLERANCE,
        "frozen_prediction_identity_failures": frozen_identity_failures,
        "legacy_new48_std_is_member_sample_std": False,
        "legacy_new48_std_absolute_difference_min": min(legacy_differences),
        "legacy_new48_std_absolute_difference_max": max(legacy_differences),
        "legacy_to_member_sample_std_ratio_min": min(legacy_ratios),
        "legacy_to_member_sample_std_ratio_median": float(np.median(legacy_ratios)),
        "legacy_to_member_sample_std_ratio_max": max(legacy_ratios),
        "external_label_values_read": False,
        "model_training_or_update_executed": False,
        "scaler_refit_executed": False,
    }
    return output, audit


def protocol_payload() -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "diagnostic_only": True,
        "uncertainty_definition": "sample standard deviation of eligible member prediction means; ddof=1",
        "legacy_new48_prediction_std_policy": (
            "retain as legacy_total_predictive_std only; it mixes member calibration variance and is not used"
        ),
        "epsilon": EPSILON,
        "calibration": {
            "source": "development-80 strict architecture-level OOF only",
            "method": "single finite-sample symmetric conformal scale",
            "coverage_levels": list(INTERVAL_LEVELS),
            "constant_width_baseline": "same finite-sample ranks on absolute OOF residuals",
            "bias_correction": False,
            "external_label_tuning": False,
        },
        "external_primary": "new48 model-blind prospective set",
        "external_secondary": "clean24 directional reproduction only",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "development_unit": "architecture_fingerprint cluster",
            "new48_outer_unit": "structure-stratified architecture_fingerprint cluster",
            "new48_inner_unit": "three GNN seeds resampled within each configuration",
            "clean24_unit": "architecture_fingerprint cluster with available GNN seeds resampled",
            "shared_indices_for_all_metrics_and_differences": True,
        },
        "random_rejection": {
            "draws": RANDOM_REJECTION_DRAWS,
            "seed": RANDOM_REJECTION_SEED,
            "unit": "configuration",
            "exclusion_rates": list(EXCLUSION_RATES),
        },
        "ranking_tie_break": "average rank, then architecture_fingerprint, hp_fingerprint, config_id",
        "classification_order": list(CLASSIFICATION_ORDER),
        "predictor_training_allowed": False,
        "vae_training_allowed": False,
        "gnn_training_allowed": False,
        "gp_allowed": False,
        "bo_allowed": False,
        "test_metric_allowed": False,
    }


def input_identities() -> dict[str, dict[str, Any]]:
    strict = REPO_ROOT / STRICT_RELATIVE
    structure = REPO_ROOT / STRUCTURE_RELATIVE
    new48 = REPO_ROOT / NEW48_RELATIVE
    paths = {
        "base_vae": BASE_CHECKPOINT,
        "strict_plan": strict / "task_plan.json",
        "strict_split": strict / "fixed_split_manifest.json",
        "strict_validation_predictions": strict / "all_validation_predictions.csv",
        "development_manifest": structure / "development/development_80_manifest.json",
        "development_prior_oof": structure / "models/development_oof_predictions.json",
        "clean_selection": structure / "phase_4/frozen_selection_24.json",
        "clean_frozen_predictions": structure / "phase_4/frozen_candidate_predictions.json",
        "clean_seed_labels_future_only": structure / "phase_5/prospective_seed_results.jsonl",
        "new48_selection": new48 / "phase_a/frozen_selection_48.json",
        "new48_frozen_predictions": new48 / "phase_a/frozen_predictions.json",
        "new48_seed_labels_future_only": new48 / "phase_b/seed_results_merged.json",
        "new48_artifact_manifest": new48 / "artifact_manifest.json",
    }
    return {name: identity(path) for name, path in paths.items()}


def terminal_error(classification: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "classification": classification,
        "evidence": dict(evidence),
        "external_validation_labels_connected": False,
        "test_metric_used": False,
        "gnn_or_predictor_or_vae_training_executed": False,
        "gp_or_bo_executed": False,
        "automatic_next_step": None,
    }


def record_command_failure(
    output: Path, classification: str, error: BaseException,
) -> None:
    """Persist a preregistered terminal error instead of failing without attribution."""
    output.mkdir(parents=True, exist_ok=True)
    final = terminal_error(classification, {
        "error_type": type(error).__name__,
        "message": str(error),
    })
    final["external_validation_labels_connected"] = (
        output / "external_labels_joined.json"
    ).is_file()
    dump_json(output / "final_classification.json", final)


def prepare(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    strict = REPO_ROOT / STRICT_RELATIVE
    structure = REPO_ROOT / STRUCTURE_RELATIVE
    new48_root = REPO_ROOT / NEW48_RELATIVE
    protocol = protocol_payload()
    protocol_path = output / "protocol.json"
    dump_json(protocol_path, protocol)
    write_sidecar(protocol_path)
    inputs = input_identities()
    dump_json(output / "input_identities.json", inputs)

    development = load_json(structure / "development/development_80_manifest.json")
    clean = load_json(structure / "phase_4/frozen_selection_24.json")
    new48 = load_json(new48_root / "phase_a/frozen_selection_48.json")
    if contains_forbidden_external_key(clean) or contains_forbidden_external_key(new48):
        evidence = {"reason": "external selection contains forbidden label-like fields"}
        dump_json(output / "final_classification.json", terminal_error("LABEL_FIREWALL_VIOLATION", evidence))
        return 2
    all_candidates = [dict(row) for row in development + clean + new48]
    matrix, members, checkpoint_manifest = head_member_predictions(all_candidates, strict)
    checkpoint_path = output / "head_checkpoint_manifest.json"
    dump_json(checkpoint_path, checkpoint_manifest)
    write_sidecar(checkpoint_path)

    development_matrix = matrix[:, : len(development)]
    external_matrix = matrix[:, len(development):]
    development_oof, oof_audit, membership = build_development_oof(
        development, development_matrix, members, strict, structure
    )
    dump_json(output / "oof_membership_audit.json", {"audit": oof_audit, "architectures": membership})
    dump_json(output / "development_80_oof_uncertainty.json", development_oof)
    dump_csv(output / "development_80_oof_uncertainty.csv", development_oof)
    if not oof_audit["passed"]:
        dump_json(output / "final_classification.json", terminal_error("HEAD_OOF_UNCERTAINTY_UNAVAILABLE", oof_audit))
        return 2

    quantiles, baseline = calibration_from_development(development_oof)
    quantile_path = output / "calibration_quantiles.json"
    baseline_path = output / "constant_width_baseline.json"
    dump_json(quantile_path, quantiles)
    dump_json(baseline_path, baseline)
    write_sidecar(quantile_path)
    write_sidecar(baseline_path)

    external, external_audit = build_external_uncertainty(
        clean, new48, external_matrix, len(members), structure, new48_root
    )
    external_path = output / "frozen_external_uncertainty_predictions.json"
    dump_json(external_path, external)
    dump_csv(output / "frozen_external_uncertainty_predictions.csv", external)
    write_sidecar(external_path)
    dump_json(output / "external_uncertainty_definition_audit.json", external_audit)
    if not external_audit["passed"]:
        dump_json(output / "final_classification.json", terminal_error("PROTOCOL_MISMATCH", external_audit))
        return 2

    new_rows = sorted(
        [row for row in external if row["dataset_role"] == "new48"],
        key=lambda row: str(row["config_id"]),
    )
    candidate_index = [
        {"index": index, "config_id": row["config_id"], "architecture_fingerprint": row["architecture_fingerprint"]}
        for index, row in enumerate(new_rows)
    ]
    candidate_index_path = output / "random_rejection_candidate_index.json"
    dump_json(candidate_index_path, candidate_index)
    write_sidecar(candidate_index_path)
    permutations = frozen_random_permutations(
        len(new_rows), RANDOM_REJECTION_DRAWS, RANDOM_REJECTION_SEED
    )
    permutation_path = output / "random_rejection_permutations.npz"
    np.savez_compressed(permutation_path, permutations=permutations)
    write_sidecar(permutation_path)
    random_manifest = {
        "draws": RANDOM_REJECTION_DRAWS,
        "seed": RANDOM_REJECTION_SEED,
        "shape": list(permutations.shape),
        "dtype": str(permutations.dtype),
        "candidate_index_path": str(candidate_index_path.resolve()),
        "candidate_index_sha256": sha256_file(candidate_index_path),
        "permutations_path": str(permutation_path.resolve()),
        "permutations_sha256": sha256_file(permutation_path),
        "label_values_used": False,
    }
    random_manifest_path = output / "random_rejection_manifest.json"
    dump_json(random_manifest_path, random_manifest)
    write_sidecar(random_manifest_path)

    external_hits = contains_forbidden_external_key(external)
    firewall = {
        "passed": not external_hits,
        "frozen_at": utc_now(),
        "external_prediction_forbidden_key_hits": external_hits,
        "external_label_values_read": False,
        "external_label_files_only_hashed_for_future_identity": True,
        "development_calibration_labels_used": True,
        "uncertainty_formula_selected_from_objective_before_external_label_join": True,
        "legacy_new48_std_retained_but_not_used": True,
        "predictor_or_vae_training_executed": False,
        "scaler_refit_executed": False,
    }
    firewall_path = output / "pre_label_firewall_audit.json"
    dump_json(firewall_path, firewall)
    if not firewall["passed"]:
        dump_json(output / "final_classification.json", terminal_error("LABEL_FIREWALL_VIOLATION", firewall))
        return 2

    frozen_outputs = {
        "protocol": identity(protocol_path),
        "head_checkpoint_manifest": identity(checkpoint_path),
        "oof_membership_audit": identity(output / "oof_membership_audit.json"),
        "development_oof_uncertainty": identity(output / "development_80_oof_uncertainty.json"),
        "calibration_quantiles": identity(quantile_path),
        "constant_width_baseline": identity(baseline_path),
        "external_uncertainty": identity(external_path),
        "external_uncertainty_definition_audit": identity(output / "external_uncertainty_definition_audit.json"),
        "candidate_index": identity(candidate_index_path),
        "random_permutations": identity(permutation_path),
        "random_manifest": identity(random_manifest_path),
        "firewall": identity(firewall_path),
    }
    plan = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "diagnostic_only": True,
        "protocol": protocol,
        "input_identities": inputs,
        "code_hashes": code_hashes(),
        "frozen_prelabel_outputs": frozen_outputs,
        "external_label_values_connected_at_freeze": False,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "random_rejection_draws": RANDOM_REJECTION_DRAWS,
        "random_rejection_seed": RANDOM_REJECTION_SEED,
        "test_metric_allowed": False,
        "training_allowed": False,
        "gp_or_bo_allowed": False,
    }
    plan_path = output / "analysis_plan.json"
    dump_json(plan_path, plan)
    write_sidecar(plan_path)
    return 0


def verify_plan(plan_path: Path) -> dict[str, Any]:
    try:
        verify_sidecar(plan_path)
        plan = load_json(plan_path)
    except Exception as error:
        raise InputIdentityMismatch(
            f"cannot verify frozen plan identity: {error}"
        ) from error
    mismatches: list[str] = []
    for collection in (plan["input_identities"], plan["frozen_prelabel_outputs"]):
        for row in collection.values():
            path = Path(row["path"])
            if not path.is_file() or sha256_file(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
                mismatches.append(str(path))
    for path_text, expected in plan["code_hashes"].items():
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(str(path))
    if mismatches:
        raise InputIdentityMismatch(
            f"frozen plan/input/code drift: {sorted(set(mismatches))}"
        )
    return plan


def connect_external_labels(
    external: Sequence[Mapping[str, Any]], structure_root: Path, new48_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    clean_seeds = load_jsonl(structure_root / "phase_5/prospective_seed_results.jsonl")
    new_seeds = load_json(new48_root / "phase_b/seed_results_merged.json")
    source_by_role = {"clean24": clean_seeds, "new48": new_seeds}
    joined: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for role in ("clean24", "new48"):
        predictions = {str(row["config_id"]): dict(row) for row in external if row["dataset_role"] == role}
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        forbidden_rows = 0
        for row in source_by_role[role]:
            if any("test" in str(key).lower() for key in row):
                forbidden_rows += 1
            if row.get("terminal_status") == "success" and row.get("valid"):
                grouped[str(row["config_id"])].append(row)
        failures: list[str] = []
        duplicate_seed_rows = 0
        fingerprint_mismatches: list[str] = []
        unexpected_config_ids = sorted(set(grouped) - set(predictions))
        for config_id, prediction in sorted(predictions.items()):
            values = grouped.get(config_id, [])
            unique = {int(row["seed"]): row for row in values}
            duplicate_seed_rows += len(values) - len(unique)
            minimum = 3 if role == "new48" else 15
            if len(unique) < minimum:
                failures.append(config_id)
                continue
            if any(
                str(row["architecture_fingerprint"]) != str(prediction["architecture_fingerprint"])
                for row in unique.values()
            ):
                fingerprint_mismatches.append(config_id)
                continue
            seed_values = [float(unique[seed]["val_acc"]) for seed in sorted(unique)]
            joined.append({
                **prediction,
                "seed_count": len(seed_values),
                "seed_values": seed_values,
                "mean_val_acc": statistics.fmean(seed_values),
            })
        audits[role] = {
            "passed": (
                not failures
                and not fingerprint_mismatches
                and not unexpected_config_ids
                and duplicate_seed_rows == 0
                and forbidden_rows == 0
                and len(grouped) == len(predictions)
            ),
            "prediction_configuration_count": len(predictions),
            "joined_configuration_count": sum(row["dataset_role"] == role for row in joined),
            "failed_config_ids": failures,
            "fingerprint_mismatch_config_ids": fingerprint_mismatches,
            "unexpected_config_ids": unexpected_config_ids,
            "duplicate_config_seed_rows": duplicate_seed_rows,
            "source_rows_with_test_named_fields": forbidden_rows,
            "label_field_read": "val_acc",
            "test_metric_used": False,
        }
    return joined, audits["clean24"], audits["new48"]


def _stable_order(rows: Sequence[Mapping[str, Any]], key: str, descending: bool = True) -> list[int]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return sorted(range(len(rows)), key=lambda index: (
        -float(values[index]) if descending else float(values[index]),
        str(rows[index]["architecture_fingerprint"]),
        str(rows[index]["hp_fingerprint"]),
        str(rows[index]["config_id"]),
    ))


def _finite_correlation(left: Sequence[float], right: Sequence[float], kind: str) -> float | None:
    if kind == "spearman":
        value = spearmanr(left, right).statistic
    elif kind == "kendall":
        value = kendalltau(left, right, variant="b").statistic
    else:
        raise ValueError(kind)
    return float(value) if value is not None and np.isfinite(value) else None


def ranking_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("ranking metrics require at least two configurations")
    actual = np.asarray([float(row["mean_val_acc"]) for row in rows], dtype=np.float64)
    predicted = np.asarray([float(row["prediction"]) for row in rows], dtype=np.float64)
    true_order = _stable_order(rows, "mean_val_acc")
    pred_order = _stable_order(rows, "prediction")
    best = float(actual.max())
    result: dict[str, Any] = {
        "configuration_count": len(rows),
        "architecture_count": len({row["architecture_fingerprint"] for row in rows}),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "kendall_tau_b": _finite_correlation(actual, predicted, "kendall"),
        "ndcg_at_10": float(ndcg_score(
            actual.reshape(1, -1), predicted.reshape(1, -1),
            k=min(10, len(rows)), ignore_ties=False,
        )),
    }
    top10_count = max(1, math.ceil(0.10 * len(rows)))
    predicted_top10 = set(pred_order[:top10_count])
    true_top10 = set(true_order[:top10_count])
    predicted_top20 = set(pred_order[: max(1, math.ceil(0.20 * len(rows)))])
    median = float(np.median(actual))
    result["false_positive_rate"] = float(
        sum(actual[index] < median for index in predicted_top10) / top10_count
    )
    result["false_negative_rate"] = float(len(true_top10 - predicted_top20) / top10_count)
    for k in TOP_K_VALUES:
        use_k = min(k, len(rows))
        predicted_top = set(pred_order[:use_k])
        result[f"regret_at_{k}"] = float(best - max(actual[list(predicted_top)]))
    return result


def interval_score(y: np.ndarray, low: np.ndarray, high: np.ndarray, alpha: float) -> np.ndarray:
    width = high - low
    below = np.where(y < low, (2.0 / alpha) * (low - y), 0.0)
    above = np.where(y > high, (2.0 / alpha) * (y - high), 0.0)
    return width + below + above


def interval_metrics(
    rows: Sequence[Mapping[str, Any]], quantiles: Mapping[str, Any], baseline: Mapping[str, Any],
) -> dict[str, float]:
    y = np.asarray([float(row["mean_val_acc"]) for row in rows], dtype=np.float64)
    mu = np.asarray([float(row["prediction"]) for row in rows], dtype=np.float64)
    sigma = np.asarray([float(row["member_prediction_sample_std_ddof1"]) for row in rows], dtype=np.float64)
    output: dict[str, float] = {}
    for coverage in INTERVAL_LEVELS:
        suffix = str(int(coverage * 100))
        alpha = 1.0 - coverage
        q = float(quantiles[f"q{suffix}"])
        half = q * sigma
        low = mu - half
        high = mu + half
        fixed_half = float(baseline[f"half_width_{suffix}"])
        fixed_low = mu - fixed_half
        fixed_high = mu + fixed_half
        hetero_scores = interval_score(y, low, high, alpha)
        fixed_scores = interval_score(y, fixed_low, fixed_high, alpha)
        output.update({
            f"hetero_{suffix}_coverage": float(np.mean((y >= low) & (y <= high))),
            f"hetero_{suffix}_mean_width": float(np.mean(high - low)),
            f"hetero_{suffix}_median_width": float(np.median(high - low)),
            f"hetero_{suffix}_interval_score": float(hetero_scores.mean()),
            f"hetero_{suffix}_clipped_coverage": float(np.mean((y >= np.clip(low, 0, 1)) & (y <= np.clip(high, 0, 1)))),
            f"hetero_{suffix}_clipped_mean_width": float(np.mean(np.clip(high, 0, 1) - np.clip(low, 0, 1))),
            f"fixed_{suffix}_coverage": float(np.mean((y >= fixed_low) & (y <= fixed_high))),
            f"fixed_{suffix}_mean_width": float(np.mean(fixed_high - fixed_low)),
            f"fixed_{suffix}_median_width": float(np.median(fixed_high - fixed_low)),
            f"fixed_{suffix}_interval_score": float(fixed_scores.mean()),
            f"hetero_minus_fixed_{suffix}_mean_width": float(np.mean(high - low) - np.mean(fixed_high - fixed_low)),
            f"hetero_minus_fixed_{suffix}_interval_score": float(hetero_scores.mean() - fixed_scores.mean()),
        })
    return output


def error_uncertainty_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    sigma = np.asarray([float(row["member_prediction_sample_std_ddof1"]) for row in rows], dtype=np.float64)
    errors = np.asarray([
        abs(float(row["mean_val_acc"]) - float(row["prediction"])) for row in rows
    ], dtype=np.float64)
    uncertain_count = math.ceil(0.25 * len(rows))
    uncertain = set(_stable_order(rows, "member_prediction_sample_std_ddof1")[:uncertain_count])
    remaining = [index for index in range(len(rows)) if index not in uncertain]
    high_error_count = math.ceil(0.20 * len(rows))
    error_order = sorted(range(len(rows)), key=lambda index: (
        -errors[index], str(rows[index]["config_id"])
    ))
    binary = np.zeros(len(rows), dtype=np.int64)
    binary[error_order[:high_error_count]] = 1
    high_mae = float(np.mean(errors[list(uncertain)]))
    remaining_mae = float(np.mean(errors[remaining]))
    return {
        "std_abs_error_spearman": _finite_correlation(sigma, errors, "spearman"),
        "std_squared_error_spearman": _finite_correlation(sigma, errors ** 2, "spearman"),
        "uncertain_top25_mae": high_mae,
        "uncertain_remaining75_mae": remaining_mae,
        "uncertain_top25_minus_remaining75_mae": high_mae - remaining_mae,
        "top20_absolute_error_auroc": float(roc_auc_score(binary, sigma)),
    }


def selective_risk_metrics(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    full = ranking_metrics(rows)
    output = [{"exclusion_rate": 0.0, "excluded_configuration_count": 0, **full}]
    flat = {f"full_{key}": value for key, value in full.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    uncertainty_order = _stable_order(rows, "member_prediction_sample_std_ddof1")
    for rate in EXCLUSION_RATES:
        excluded_count = math.ceil(rate * len(rows))
        excluded = set(uncertainty_order[:excluded_count])
        retained = [dict(row) for index, row in enumerate(rows) if index not in excluded]
        metrics = ranking_metrics(retained)
        suffix = str(int(rate * 100))
        output.append({"exclusion_rate": rate, "excluded_configuration_count": excluded_count, **metrics})
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[f"exclude_{suffix}_{key}"] = value
                if key in full and isinstance(full[key], (int, float)):
                    flat[f"exclude_{suffix}_minus_full_{key}"] = float(value) - float(full[key])
    return output, flat


def false_high_rank_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actual = np.asarray([float(row["mean_val_acc"]) for row in rows], dtype=np.float64)
    sigma = np.asarray([float(row["member_prediction_sample_std_ddof1"]) for row in rows], dtype=np.float64)
    median = float(np.median(actual))
    top_count = max(1, math.ceil(0.10 * len(rows)))
    predicted_top = _stable_order(rows, "prediction")[:top_count]
    false_indices = [index for index in predicted_top if actual[index] < median]
    other = [index for index in range(len(rows)) if index not in false_indices]
    excluded = set(_stable_order(rows, "member_prediction_sample_std_ddof1")[: math.ceil(0.25 * len(rows))])
    retained = [dict(row) for index, row in enumerate(rows) if index not in excluded]
    retained_actual = np.asarray([float(row["mean_val_acc"]) for row in retained], dtype=np.float64)
    retained_top = _stable_order(retained, "prediction")[: max(1, math.ceil(0.10 * len(retained)))]
    retained_false = [index for index in retained_top if retained_actual[index] < median]
    return {
        "full_actual_median": median,
        "full_predicted_top10_count": top_count,
        "full_false_high_rank_count": len(false_indices),
        "false_high_rank_mean_prediction_std": float(np.mean(sigma[false_indices])) if false_indices else None,
        "false_high_rank_median_prediction_std": float(np.median(sigma[false_indices])) if false_indices else None,
        "other_mean_prediction_std": float(np.mean(sigma[other])) if other else None,
        "other_median_prediction_std": float(np.median(sigma[other])) if other else None,
        "after_exclude25_predicted_top10_count": len(retained_top),
        "after_exclude25_false_high_rank_count": len(retained_false),
        "false_high_rank_count_reduction": len(false_indices) - len(retained_false),
    }


def analysis_bundle(
    rows: Sequence[Mapping[str, Any]], quantiles: Mapping[str, Any], baseline: Mapping[str, Any],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for source in (
        error_uncertainty_metrics(rows), interval_metrics(rows, quantiles, baseline),
        selective_risk_metrics(rows)[1], false_high_rank_metrics(rows),
    ):
        for key, value in source.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                output[key] = float(value)
    return output


def random_rejection_baselines(
    rows: Sequence[Mapping[str, Any]], permutations: np.ndarray,
) -> list[dict[str, Any]]:
    if permutations.shape != (RANDOM_REJECTION_DRAWS, len(rows)):
        raise RuntimeError("frozen random rejection permutation shape mismatch")
    keys = [
        "mae", "rmse", "kendall_tau_b", "ndcg_at_10", "false_positive_rate",
        "regret_at_1", "regret_at_3", "regret_at_5", "regret_at_10",
    ]
    output: list[dict[str, Any]] = []
    for rate in EXCLUSION_RATES:
        excluded_count = math.ceil(rate * len(rows))
        values: dict[str, list[float]] = defaultdict(list)
        for permutation in permutations:
            excluded = set(map(int, permutation[:excluded_count]))
            retained = [dict(row) for index, row in enumerate(rows) if index not in excluded]
            metrics = ranking_metrics(retained)
            for key in keys:
                value = metrics.get(key)
                if value is not None:
                    values[key].append(float(value))
        for key in keys:
            array = np.asarray(values[key], dtype=np.float64)
            output.append({
                "exclusion_rate": rate,
                "excluded_configuration_count": excluded_count,
                "metric": key,
                "draw_count": len(array),
                "mean": float(array.mean()),
                "median": float(np.median(array)),
                "q05": float(np.quantile(array, 0.05)),
                "q95": float(np.quantile(array, 0.95)),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
            })
    return output


def _new48_bootstrap_rows(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(row)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for fingerprint, values in grouped.items():
        strata = {str(row["structure_stratum"]) for row in values}
        if len(strata) != 1:
            raise RuntimeError("new48 architecture spans structure strata")
        by_stratum[next(iter(strata))].append(fingerprint)
    sampled: list[str] = []
    for stratum in sorted(by_stratum):
        fingerprints = sorted(by_stratum[stratum])
        sampled.extend(
            fingerprints[index]
            for index in rng.integers(0, len(fingerprints), len(fingerprints))
        )
    counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for fingerprint in sampled:
        copy_index = counts[fingerprint]
        counts[fingerprint] += 1
        for row in grouped[fingerprint]:
            seeds = np.asarray(row["seed_values"], dtype=np.float64)
            sampled_seeds = seeds[rng.integers(0, len(seeds), len(seeds))]
            output.append({
                **dict(row),
                "config_id": f"{row['config_id']}#bootstrap{copy_index}",
                "architecture_fingerprint": f"{fingerprint}#bootstrap{copy_index}",
                "mean_val_acc": float(sampled_seeds.mean()),
            })
    return output


def _cluster_bootstrap_rows(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator, resample_seeds: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["architecture_fingerprint"])].append(row)
    fingerprints = sorted(grouped)
    sampled = [fingerprints[index] for index in rng.integers(0, len(fingerprints), len(fingerprints))]
    counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for fingerprint in sampled:
        copy_index = counts[fingerprint]
        counts[fingerprint] += 1
        for row in grouped[fingerprint]:
            copied = {
                **dict(row),
                "config_id": f"{row['config_id']}#bootstrap{copy_index}",
                "architecture_fingerprint": f"{fingerprint}#bootstrap{copy_index}",
            }
            if resample_seeds:
                seeds = np.asarray(row["seed_values"], dtype=np.float64)
                copied["mean_val_acc"] = float(
                    seeds[rng.integers(0, len(seeds), len(seeds))].mean()
                )
            output.append(copied)
    return output


def bootstrap_analysis(
    development: Sequence[Mapping[str, Any]], clean: Sequence[Mapping[str, Any]],
    new48: Sequence[Mapping[str, Any]], quantiles: Mapping[str, Any], baseline: Mapping[str, Any],
    replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    point_new = analysis_bundle(new48, quantiles, baseline)
    point_clean = error_uncertainty_metrics(clean)
    dev_scores = [float(row["standardized_absolute_residual"]) for row in development]
    point_dev = {
        "q90": finite_conformal_quantile(dev_scores, 0.90)[0],
        "q95": finite_conformal_quantile(dev_scores, 0.95)[0],
        "std_abs_error_spearman": _finite_correlation(
            [float(row["member_prediction_sample_std_ddof1"]) for row in development],
            [float(row["absolute_error"]) for row in development], "spearman",
        ),
    }
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(replicates):
        boot_new = _new48_bootstrap_rows(new48, rng)
        for key, value in analysis_bundle(boot_new, quantiles, baseline).items():
            values[("new48", key)].append(value)
        boot_clean = _cluster_bootstrap_rows(clean, rng, resample_seeds=True)
        for key, value in error_uncertainty_metrics(boot_clean).items():
            if value is not None and np.isfinite(value):
                values[("clean24", key)].append(float(value))
        boot_dev = _cluster_bootstrap_rows(development, rng, resample_seeds=False)
        scores = [float(row["standardized_absolute_residual"]) for row in boot_dev]
        dev_values = {
            "q90": finite_conformal_quantile(scores, 0.90)[0],
            "q95": finite_conformal_quantile(scores, 0.95)[0],
            "std_abs_error_spearman": _finite_correlation(
                [float(row["member_prediction_sample_std_ddof1"]) for row in boot_dev],
                [float(row["absolute_error"]) for row in boot_dev], "spearman",
            ),
        }
        for key, value in dev_values.items():
            if value is not None and np.isfinite(value):
                values[("development80", key)].append(float(value))
    point_lookup = {
        "new48": point_new,
        "clean24": point_clean,
        "development80": point_dev,
    }
    output: list[dict[str, Any]] = []
    for (scope, metric), items in sorted(values.items()):
        array = np.asarray(items, dtype=np.float64)
        output.append({
            "scope": scope,
            "metric": metric,
            "point_estimate": point_lookup[scope].get(metric),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
            "bootstrap_replicates": replicates,
            "valid_replicates": len(array),
            "shared_bootstrap_indices": True,
        })
    return output


def classify(
    new_metrics: Mapping[str, Any], clean_metrics: Mapping[str, Any],
    intervals: Mapping[tuple[str, str], Mapping[str, Any]],
    random_summary: Sequence[Mapping[str, Any]], audits_passed: bool,
) -> tuple[str, dict[str, bool]]:
    spearman_ci = intervals[("new48", "std_abs_error_spearman")]
    delta_ci = intervals[("new48", "uncertain_top25_minus_remaining75_mae")]
    random_mae = next(
        row for row in random_summary if row["exclusion_rate"] == 0.25 and row["metric"] == "mae"
    )
    gates = {
        "identity_oof_firewall_audits_pass": audits_passed,
        "std_abs_error_spearman_ci_lower_above_zero": spearman_ci["ci95_low"] > 0,
        "uncertain_top25_mae_higher_and_delta_ci_lower_above_zero": (
            new_metrics["uncertain_top25_minus_remaining75_mae"] > 0
            and delta_ci["ci95_low"] > 0
        ),
        "hetero_90_coverage_at_least_0_85": new_metrics["hetero_90_coverage"] >= 0.85,
        "hetero_95_coverage_at_least_0_90": new_metrics["hetero_95_coverage"] >= 0.90,
        "hetero_90_interval_score_not_worse_than_fixed": (
            new_metrics["hetero_90_interval_score"] <= new_metrics["fixed_90_interval_score"]
        ),
        "hetero_95_interval_score_not_worse_than_fixed": (
            new_metrics["hetero_95_interval_score"] <= new_metrics["fixed_95_interval_score"]
        ),
        "exclude25_mae_lower_than_full": new_metrics["exclude_25_mae"] < new_metrics["full_mae"],
        "exclude25_mae_below_random_q05": new_metrics["exclude_25_mae"] < random_mae["q05"],
        "exclude25_kendall_not_lower_than_full": (
            new_metrics["exclude_25_kendall_tau_b"] >= new_metrics["full_kendall_tau_b"]
        ),
        "exclude25_false_positive_rate_not_higher_than_full": (
            new_metrics["exclude_25_false_positive_rate"] <= new_metrics["full_false_positive_rate"]
        ),
        "clean24_std_abs_error_spearman_positive": (
            clean_metrics["std_abs_error_spearman"] is not None
            and clean_metrics["std_abs_error_spearman"] > 0
        ),
    }
    if all(gates.values()):
        return "HEAD_UNCERTAINTY_VALIDATED_FOR_ACQUISITION_PILOT", gates
    rejected = (
        spearman_ci["ci95_high"] <= 0
        or (
            new_metrics["uncertain_top25_minus_remaining75_mae"] <= 0
            and delta_ci["ci95_high"] <= 0
        )
        or new_metrics["hetero_90_coverage"] < 0.75
        or new_metrics["hetero_95_coverage"] < 0.85
    )
    if rejected:
        return "HEAD_UNCERTAINTY_REJECTED", gates
    if (
        new_metrics["std_abs_error_spearman"] > 0
        or new_metrics["uncertain_top25_minus_remaining75_mae"] > 0
    ):
        return "HEAD_UNCERTAINTY_SIGNAL_ONLY", gates
    return "HEAD_UNCERTAINTY_UNDERPOWERED", gates


def evaluate(
    plan_path: Path, output: Path, replicates: int = BOOTSTRAP_REPLICATES,
) -> int:
    plan = verify_plan(plan_path)
    if output.resolve() != plan_path.parent.resolve():
        raise RuntimeError("evaluation output must equal the frozen plan directory")
    structure = REPO_ROOT / STRUCTURE_RELATIVE
    new48_root = REPO_ROOT / NEW48_RELATIVE
    external = load_json(Path(plan["frozen_prelabel_outputs"]["external_uncertainty"]["path"]))
    forbidden = contains_forbidden_external_key(external)
    if forbidden:
        final = terminal_error("LABEL_FIREWALL_VIOLATION", {"forbidden_keys": forbidden})
        dump_json(output / "final_classification.json", final)
        return 2
    joined, clean_audit, new_audit = connect_external_labels(external, structure, new48_root)
    dump_json(output / "clean24_label_connection_audit.json", clean_audit)
    dump_json(output / "new48_label_connection_audit.json", new_audit)
    if not clean_audit["passed"] or not new_audit["passed"]:
        final = terminal_error("PROTOCOL_MISMATCH", {"clean24": clean_audit, "new48": new_audit})
        dump_json(output / "final_classification.json", final)
        return 2
    dump_json(output / "external_labels_joined.json", joined)
    dump_csv(output / "external_labels_joined.csv", joined)
    clean = [row for row in joined if row["dataset_role"] == "clean24"]
    new48 = [row for row in joined if row["dataset_role"] == "new48"]
    quantiles = load_json(Path(plan["frozen_prelabel_outputs"]["calibration_quantiles"]["path"]))
    baseline = load_json(Path(plan["frozen_prelabel_outputs"]["constant_width_baseline"]["path"]))
    development = load_json(Path(plan["frozen_prelabel_outputs"]["development_oof_uncertainty"]["path"]))

    new_error = error_uncertainty_metrics(new48)
    clean_error = error_uncertainty_metrics(clean)
    intervals = interval_metrics(new48, quantiles, baseline)
    selective_rows, selective_flat = selective_risk_metrics(new48)
    false_high = false_high_rank_metrics(new48)
    new_metrics = {
        **new_error, **intervals, **selective_flat,
        **{key: value for key, value in false_high.items() if isinstance(value, (int, float)) and value is not None},
    }
    dump_json(output / "external_error_uncertainty_metrics.json", {
        "new48": new_error, "clean24": clean_error,
    })
    dump_json(output / "interval_coverage_and_score.json", intervals)
    dump_json(output / "selective_risk_curves.json", selective_rows)
    dump_csv(output / "selective_risk_curves.csv", selective_rows)
    dump_json(output / "false_high_rank_analysis.json", false_high)

    permutation_path = Path(plan["frozen_prelabel_outputs"]["random_permutations"]["path"])
    permutations = np.load(permutation_path)["permutations"]
    candidate_index = load_json(Path(plan["frozen_prelabel_outputs"]["candidate_index"]["path"]))
    if [row["config_id"] for row in new48] != [row["config_id"] for row in candidate_index]:
        raise RuntimeError("new48 label join order differs from frozen random candidate index")
    random_summary = random_rejection_baselines(new48, permutations)
    dump_json(output / "random_rejection_baselines.json", random_summary)
    dump_csv(output / "random_rejection_baselines.csv", random_summary)

    bootstrap_rows = bootstrap_analysis(
        development, clean, new48, quantiles, baseline,
        replicates=replicates, seed=int(plan["bootstrap_seed"]),
    )
    dump_json(output / "bootstrap_intervals.json", bootstrap_rows)
    dump_csv(output / "bootstrap_intervals.csv", bootstrap_rows)
    interval_lookup = {(row["scope"], row["metric"]): row for row in bootstrap_rows}
    classification, gates = classify(
        new_metrics, clean_error, interval_lookup, random_summary,
        audits_passed=clean_audit["passed"] and new_audit["passed"],
    )
    final = {
        "classification": classification,
        "success_gates": gates,
        "development_strict_architecture_level_oof": True,
        "development_configuration_count": len(development),
        "development_architecture_count": len({row["architecture_fingerprint"] for row in development}),
        "new48_configuration_count": len(new48),
        "new48_architecture_count": len({row["architecture_fingerprint"] for row in new48}),
        "prediction_std_abs_error_spearman": new_error["std_abs_error_spearman"],
        "prediction_std_abs_error_spearman_ci95": [
            interval_lookup[("new48", "std_abs_error_spearman")]["ci95_low"],
            interval_lookup[("new48", "std_abs_error_spearman")]["ci95_high"],
        ],
        "coverage_90": intervals["hetero_90_coverage"],
        "coverage_95": intervals["hetero_95_coverage"],
        "hetero_interval_score_better_or_equal_to_fixed_90": (
            intervals["hetero_90_interval_score"] <= intervals["fixed_90_interval_score"]
        ),
        "hetero_interval_score_better_or_equal_to_fixed_95": (
            intervals["hetero_95_interval_score"] <= intervals["fixed_95_interval_score"]
        ),
        "exclude25_mae": selective_flat["exclude_25_mae"],
        "full_mae": selective_flat["full_mae"],
        "exclude25_kendall": selective_flat["exclude_25_kendall_tau_b"],
        "full_kendall": selective_flat["full_kendall_tau_b"],
        "exclude25_false_positive_rate": selective_flat["exclude_25_false_positive_rate"],
        "full_false_positive_rate": selective_flat["full_false_positive_rate"],
        "clean24_prediction_std_abs_error_spearman": clean_error["std_abs_error_spearman"],
        "uncertainty_acquisition_pilot_authorized": (
            classification == "HEAD_UNCERTAINTY_VALIDATED_FOR_ACQUISITION_PILOT"
        ),
        "deterministic_head_ranking_plus_structure_diversity_route": (
            classification != "HEAD_UNCERTAINTY_VALIDATED_FOR_ACQUISITION_PILOT"
        ),
        "test_metric_used": False,
        "gnn_or_predictor_or_vae_training_executed": False,
        "gp_or_bo_executed": False,
        "automatic_next_step": None,
    }
    dump_json(output / "final_classification.json", final)
    report = f"""# Original Head uncertainty and selective-risk audit

Final classification: **`{classification}`**.

1. Development-80 uses strict architecture-level OOF predictions for 96 configurations and 80 architectures.
2. New-48 member sample standard deviation versus absolute error Spearman is `{new_error['std_abs_error_spearman']:.6f}`, with 95% CI `[{final['prediction_std_abs_error_spearman_ci95'][0]:.6f}, {final['prediction_std_abs_error_spearman_ci95'][1]:.6f}]`.
3. Unclipped calibrated interval coverage is `{intervals['hetero_90_coverage']:.6f}` at 90% and `{intervals['hetero_95_coverage']:.6f}` at 95%.
4. Heteroscedastic interval scores are {'not worse than' if final['hetero_interval_score_better_or_equal_to_fixed_90'] and final['hetero_interval_score_better_or_equal_to_fixed_95'] else 'not consistently better than'} the fixed-width baselines.
5. Excluding the highest-uncertainty 25% changes MAE from `{selective_flat['full_mae']:.6f}` to `{selective_flat['exclude_25_mae']:.6f}`, Kendall from `{selective_flat['full_kendall_tau_b']:.6f}` to `{selective_flat['exclude_25_kendall_tau_b']:.6f}`, and false-positive rate from `{selective_flat['full_false_positive_rate']:.6f}` to `{selective_flat['exclude_25_false_positive_rate']:.6f}`.
6. Clean-24 uncertainty/error Spearman is `{clean_error['std_abs_error_spearman']:.6f}`.
7. A small uncertainty-acquisition pilot is {'authorized' if final['uncertainty_acquisition_pilot_authorized'] else 'not authorized'}.
8. The deterministic Original Head ranking plus structure-diversity route is {'the formal next research route' if final['deterministic_head_ranking_plus_structure_diversity_route'] else 'not activated by this result'}.

This diagnostic did not train a model, use a test metric, run GP/BO, or automatically start a follow-up experiment.
"""
    (output / "final_goal_report.md").write_text(report, encoding="utf-8")
    dump_json(output / "result_integrity_audit.json", {
        "passed": True,
        "plan_sha256": sha256_file(plan_path),
        "clean24_label_join_passed": clean_audit["passed"],
        "new48_label_join_passed": new_audit["passed"],
        "bootstrap_replicates": replicates,
        "test_metric_used": False,
        "gnn_or_predictor_or_vae_training_executed": False,
        "gp_or_bo_executed": False,
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--plan", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--bootstrap_replicates", type=int, default=BOOTSTRAP_REPLICATES, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    try:
        if args.command == "prepare":
            return prepare(output)
        if args.command == "evaluate":
            return evaluate(
                args.plan.resolve(), output, replicates=args.bootstrap_replicates,
            )
        raise AssertionError(args.command)
    except InputIdentityMismatch as error:
        record_command_failure(output, "INPUT_IDENTITY_MISMATCH", error)
        return 2
    except Exception as error:
        record_command_failure(output, "INPUT_OR_IMPLEMENTATION_ERROR", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
