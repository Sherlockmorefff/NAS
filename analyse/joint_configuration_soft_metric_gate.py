#!/usr/bin/env python3
"""One-shot offline gate for an uncertainty-weighted joint performance metric.

Only existing Cora validation records are used.  The formal path trains the
frozen metric encoder and offline predictors; it never trains a candidate GNN,
reads a test-labelled field, invokes a decoder, or runs an acquisition method.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import joint_configuration_metric_gate as joint_v1  # noqa: E402
from analyse import joint_pair_power_projection_audit as pair_audit  # noqa: E402
from analyse import performance_metric_quick_gate as performance_v2  # noqa: E402,F401


FORMAT_VERSION = "joint_configuration_soft_metric_gate_global4_v2"
DEFAULT_CONFIG = REPO_ROOT / "configs/joint_configuration_soft_metric_gate_v2.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/posthoc/joint-configuration-soft-metric-gate-global4-v2"
METRIC_SEEDS = (0, 1, 2)
MODEL_NAMES = (
    "B0", "B1", "B2", "B3", "B4", "EJ_KNN",
    "M1", "M2", "M3", "M4",
    "ABL_ARCH_M1", "ABL_ARCH_M2", "ABL_HP_M1", "ABL_HP_M2",
    "M2_architecture_shuffled", "M2_hp_shuffled",
)
ALLOWED_CLASSIFICATIONS = {
    "INPUT_IDENTITY_MISMATCH",
    "SOFT_JOINT_METRIC_REJECTED",
    "SOFT_JOINT_METRIC_GEOMETRY_ONLY",
    "SOFT_JOINT_METRIC_LOCALITY_ONLY",
    "SOFT_JOINT_METRIC_FEATURE_ONLY",
    "SOFT_JOINT_METRIC_NOT_JOINT",
    "SOFT_JOINT_METRIC_GP_READY",
    "RUN_FAILED",
}
REQUIRED_OUTPUTS = (
    "state.json", "frozen_protocol.json", "input_integrity.json",
    "data_inventory.json", "split_manifest.json", "pair_weight_summary.json",
    "metric_training_audit.json", "geometry_metrics.json", "knn_locality.json",
    "metrics_per_fold.csv", "oof_predictions.csv", "best_of_4_batches.json",
    "best_of_4_results.json", "branch_ablation.json", "shuffle_controls.json",
    "sensitivity_30seed.json", "decision.json", "final_report.md",
    "environment.json", "git_status.txt", "git_diff.patch", "manifest.json",
    "source_snapshot/imported_local_modules.json",
    "source_snapshot/sha256_manifest.json", "logs/py_compile.log",
    "logs/pytest_focused.log", "logs/synthetic_smoke.log", "logs/run.log",
)


class ProtocolError(RuntimeError):
    """A fail-closed protocol, leakage, or numerical error."""


class InputIdentityMismatch(ProtocolError):
    """The frozen data, result, or split identity is not authoritative."""


class SpecUnderdefinedError(ProtocolError):
    """A reachable scientific decision was not defined by the prompt."""


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
    return joint_v1.sha256_file(path)


def dump_json(path: Path, value: Any) -> None:
    joint_v1.dump_json(path, value)


def dump_text(path: Path, value: str) -> None:
    joint_v1.dump_text(path, value)


def dump_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None,
) -> None:
    joint_v1.dump_csv(path, rows, fields)


def _require(condition: bool, message: str, *, identity: bool = False) -> None:
    if condition:
        return
    if identity:
        raise InputIdentityMismatch(message)
    raise ProtocolError(message)


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ProtocolError("PyYAML is required for the frozen protocol") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "config must be a mapping")
    validate_frozen_protocol(value)
    return value


def validate_frozen_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == FORMAT_VERSION, "protocol_version mismatch")
    soft = config["soft_metric"]
    training = config["training"]
    methods = config["metric_methods"]
    frozen_checks = {
        "embedding_dim": int(config["encoder"]["output_dim"]) == 8,
        "metric_seeds": list(training["metric_model_seeds"]) == [0, 1, 2],
        "delta": float(soft["delta"]) == 0.012,
        "minimum_pair_weight": float(soft["minimum_pair_weight"]) == 0.20,
        "reliability_fraction": float(soft["reliability_weight_fraction"]) == 0.80,
        "pair_cap": int(soft["pair_cap"]) == 50000,
        "pair_seed": int(soft["pair_sampling_seed"]) == 0,
        "knn_k": int(methods["M1"]["k"]) == 5,
        "local_gp_k": int(methods["M3"]["neighbors"]) == 32,
        "epochs": int(training["max_epochs"]) == 200,
        "patience": int(training["patience"]) == 20,
        "learning_rate": float(training["learning_rate"]) == 1.0e-3,
        "weight_decay": float(training["weight_decay"]) == 1.0e-4,
        "M2_unstandardized": methods["M2"]["per_dimension_standardization"] is False,
        "M2_isotropic": methods["M2"]["ard"] is False and int(methods["M2"]["lengthscale_count"]) == 1,
        "M4_diagnostic": methods["M4"]["may_form_gp_ready"] is False,
        "hard_triplets_forbidden": soft["hard_triplets"] == "forbidden",
        "hp_residualization_forbidden": config["data"]["hp_residualization"] == "forbidden",
        "hp_encoder": list(config["encoder"]["hp_layers"]) == [4, 16, 16],
        "fusion": list(config["encoder"]["fusion_layers"]) == [48, 32, 8],
        "components": list(config["encoder"]["joint_components"]) == [
            "architecture_projection", "hp_representation", "elementwise_interaction",
        ],
        "smooth_l1": float(soft["smooth_l1_beta"]) == 1.0,
        "M1_raw_distance": methods["M1"]["distance"] == "raw_embedding_Euclidean"
        and methods["M1"]["per_dimension_standardization"] is False,
        "M3": int(methods["M3"]["neighbors"]) == 32 and methods["M3"]["gp_protocol"] == "M2",
        "controls": config["controls"]["retrain_for_controls"] is False
        and int(config["controls"]["shuffle_seed"]) == 0,
        "best4": int(config["best_of_4"]["batch_size"]) == 4
        and int(config["best_of_4"]["exhaustive_combination_limit"]) == 5000
        and int(config["best_of_4"]["sampled_batch_count"]) == 1000
        and int(config["best_of_4"]["sampled_batch_seed"]) == 0,
        "allowed_labels": set(config["decision"]["allowed"]) == ALLOWED_CLASSIFICATIONS,
        "sensitivity_only": config["sensitivity_30seed"]["may_train_or_select_parameters"] is False
        and config["sensitivity_30seed"]["may_change_classification"] is False,
    }
    geometry = config["decision"]["geometry_gate"]
    m1 = config["decision"]["M1_gate_over_explicit_joint_feature_knn"]
    m2 = config["decision"]["M2_gate_over_strongest_non_metric_gp"]
    contribution = config["decision"]["joint_contribution_gate"]
    frozen_checks.update({
        "geometry_thresholds": (
            float(geometry["median_fold_spearman_across_metric_seeds_strictly_greater_than"]) == 0.20
            and int(geometry["positive_seed_count_at_least"]) == 2
            and int(geometry["positive_folds_per_seed_at_least"]) == 4
            and float(geometry["cross_factor_locality_ratio_strictly_less_than"]) == 0.85
            and float(geometry["local_gap5_relative_reduction_over_explicit_knn_at_least"]) == 0.05
        ),
        "M1_thresholds": (
            float(m1["mae_relative_improvement_at_least"]) == 0.05
            and float(m1["top20_precision_absolute_improvement_at_least"]) == 0.05
            and float(m1["best_of_4_regret_relative_improvement_at_least"]) == 0.10
        ),
        "M2_thresholds": (
            float(m2["mae_relative_improvement_at_least"]) == 0.05
            and float(m2["top20_precision_absolute_improvement_at_least"]) == 0.05
            and float(m2["best_of_4_regret_relative_improvement_at_least"]) == 0.10
            and int(m2["lower_regret_outer_folds_at_least"]) == 4
        ),
        "joint_thresholds": (
            int(contribution["required_better_metric_count_vs_each_branch"]) == 2
            and all(
                float(contribution[name][key]) == expected
                for name in ("architecture_shuffle", "hp_shuffle")
                for key, expected in (
                    ("mae_relative_degradation_at_least", 0.03),
                    ("top20_precision_absolute_decrease_at_least", 0.03),
                    ("regret_relative_degradation_at_least", 0.05),
                )
            )
        ),
    })
    failed = sorted(key for key, passed in frozen_checks.items() if not passed)
    _require(not failed, f"frozen protocol constants changed: {failed}")


def aggregate_three_seed_validation(seed_values: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in seed_values]
    _require(len(values) == 3, "raw validation aggregation requires exactly three seeds")
    _require(all(math.isfinite(value) for value in values), "validation values must be finite")
    mean = sum(values) / 3.0
    variance = sum((value - mean) ** 2 for value in values) / 2.0
    sd = math.sqrt(max(variance, 0.0))
    return {"mean": mean, "sd": sd, "se": sd / math.sqrt(3.0)}


def paired_soft_statistics(
    left: Mapping[str, Any], right: Mapping[str, Any], *, delta: float = 0.012,
    minimum_weight: float = 0.20, reliability_fraction: float = 0.80,
) -> dict[str, Any]:
    differences = [
        float(left[f"validation_seed_{seed}"]) - float(right[f"validation_seed_{seed}"])
        for seed in METRIC_SEEDS
    ]
    summary = aggregate_three_seed_validation(differences)
    gap = abs(float(summary["mean"]))
    similarity = soft_similarity(gap, delta)
    weight = uncertainty_weight(
        float(summary["se"]), delta=delta, minimum_weight=minimum_weight,
        reliability_fraction=reliability_fraction,
    )
    reliability = float(delta) ** 2 / (float(delta) ** 2 + float(summary["se"]) ** 2)
    return {
        "matched_seed_differences": differences,
        "mean_difference": float(summary["mean"]),
        "paired_sd": float(summary["sd"]),
        "paired_se": float(summary["se"]),
        "performance_gap": gap, "soft_similarity": similarity,
        "reliability": reliability, "weight": weight,
    }


def soft_similarity(performance_gap: float, delta: float = 0.012) -> float:
    gap = float(performance_gap)
    scale = float(delta)
    _require(math.isfinite(gap) and gap >= 0.0, "performance gap must be finite and nonnegative")
    _require(math.isfinite(scale) and scale > 0.0, "delta must be finite and positive")
    return math.exp(-0.5 * (gap / scale) ** 2)


def uncertainty_weight(
    paired_se: float, *, delta: float = 0.012, minimum_weight: float = 0.20,
    reliability_fraction: float = 0.80,
) -> float:
    se = float(paired_se)
    scale = float(delta)
    floor = float(minimum_weight)
    fraction = float(reliability_fraction)
    _require(math.isfinite(se) and se >= 0.0, "paired SE must be finite and nonnegative")
    _require(scale > 0.0 and floor == 0.20 and fraction == 0.80, "uncertainty-weight constants changed")
    reliability = scale * scale / (scale * scale + se * se)
    weight = floor + fraction * reliability
    _require(0.20 <= weight <= 1.0, "pair weight escaped [0.20,1.00]")
    return weight


def construct_soft_pairs(
    rows: Sequence[Mapping[str, Any]], *, allowed_joint_fingerprints: set[str],
    delta: float = 0.012, minimum_weight: float = 0.20,
    reliability_fraction: float = 0.80, pair_cap: int = 50000,
    sampling_seed: int = 0,
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    observed = {str(row["joint_fingerprint"]) for row in materialized}
    allowed = set(map(str, allowed_joint_fingerprints))
    if observed != allowed:
        raise ProtocolError(
            "soft pair rows must equal the declared training scope; "
            f"foreign={sorted(observed - allowed)[:3]} missing={sorted(allowed - observed)[:3]}"
        )
    combinations = list(itertools.combinations(range(len(materialized)), 2))
    candidate_count = len(combinations)
    if candidate_count > int(pair_cap):
        selected_positions = sorted(
            random.Random(int(sampling_seed)).sample(range(candidate_count), int(pair_cap))
        )
        selected = [combinations[position] for position in selected_positions]
        mode = "uniform_sample_without_replacement_seed_0"
    else:
        selected_positions = list(range(candidate_count))
        selected = combinations
        mode = "all_unordered_pairs"
    pairs: list[dict[str, Any]] = []
    for left_index, right_index in selected:
        statistics = paired_soft_statistics(
            materialized[left_index], materialized[right_index], delta=delta,
            minimum_weight=minimum_weight, reliability_fraction=reliability_fraction,
        )
        pairs.append({
            "left_index": left_index, "right_index": right_index,
            "left": str(materialized[left_index]["joint_fingerprint"]),
            "right": str(materialized[right_index]["joint_fingerprint"]),
            **statistics,
        })
    _require(all(pair["left"] != pair["right"] for pair in pairs), "self pair entered soft loss")
    return {
        "pairs": pairs, "row_joint_fingerprints": [
            str(row["joint_fingerprint"]) for row in materialized
        ],
        "allowed_joint_fingerprints": sorted(allowed),
        "candidate_pair_count": candidate_count, "selected_pair_count": len(pairs),
        "selection_mode": mode, "sampling_seed": int(sampling_seed),
        "selected_candidate_positions_sha256": stable_hash(selected_positions),
        "performance_aware_sampling_used": False,
        "architecture_aware_sampling_used": False,
        "hp_aware_sampling_used": False,
        "outer_heldout_labels_used": False,
    }


def summarize_soft_pairs(inventory: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as error:
        raise ProtocolError("NumPy is required for pair summaries") from error
    pairs = list(inventory["pairs"])
    _require(bool(pairs), "soft pair inventory is empty")
    output = {
        key: np.asarray([float(pair[key]) for pair in pairs], dtype=np.float64)
        for key in ("performance_gap", "paired_se", "soft_similarity", "reliability", "weight")
    }
    summaries: dict[str, Any] = {}
    for key, values in output.items():
        summaries[key] = {
            "minimum": float(values.min()), "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, 0.75)),
            "maximum": float(values.max()), "mean": float(values.mean()),
        }
    return {
        "candidate_pair_count": int(inventory["candidate_pair_count"]),
        "selected_pair_count": int(inventory["selected_pair_count"]),
        "selection_mode": inventory["selection_mode"],
        "sampling_seed": int(inventory["sampling_seed"]),
        "selected_candidate_positions_sha256": inventory["selected_candidate_positions_sha256"],
        "summaries": summaries,
        "weight_bounds_verified": bool(output["weight"].min() >= 0.20 and output["weight"].max() <= 1.0),
        "all_pairs_have_nonzero_weight": bool(output["weight"].min() > 0.0),
        "hard_labels_or_triplets_used": False,
        "outer_heldout_labels_used": False,
    }


def weighted_soft_metric_loss(embeddings: Any, pair_inventory: Mapping[str, Any], torch: Any) -> Any:
    pairs = list(pair_inventory["pairs"])
    _require(bool(pairs), "soft metric loss requires at least one pair")
    left = torch.tensor([int(pair["left_index"]) for pair in pairs], dtype=torch.long)
    right = torch.tensor([int(pair["right_index"]) for pair in pairs], dtype=torch.long)
    targets = torch.tensor([float(pair["soft_similarity"]) for pair in pairs], dtype=embeddings.dtype)
    weights = torch.tensor([float(pair["weight"]) for pair in pairs], dtype=embeddings.dtype)
    predicted = (1.0 + torch.sum(embeddings[left] * embeddings[right], dim=1)) / 2.0
    losses = torch.nn.functional.smooth_l1_loss(
        predicted, targets, reduction="none", beta=1.0,
    )
    denominator = torch.sum(weights)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        raise ProtocolError("soft metric loss has invalid total weight")
    loss = torch.sum(weights * losses) / denominator
    if not bool(torch.isfinite(loss)):
        raise ProtocolError("soft metric loss is non-finite")
    return loss


def train_soft_metric_encoder(
    outer_train: Sequence[Mapping[str, Any]], outer_heldout: Sequence[Mapping[str, Any]],
    seed: int, config: Mapping[str, Any], hp_standardizer: Mapping[str, Any],
    full_pair_inventory: Mapping[str, Any],
    *, stage_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ProtocolError("PyTorch and NumPy are required for soft metric training") from error
    train_ids = {str(row["joint_fingerprint"]) for row in outer_train}
    heldout_ids = {str(row["joint_fingerprint"]) for row in outer_heldout}
    _require(not train_ids & heldout_ids, "outer-held-out configuration entered metric training")
    _require(
        set(full_pair_inventory["allowed_joint_fingerprints"]) == train_ids,
        "full-refit pair inventory does not equal outer-train scope",
    )
    training = config["training"]
    soft = config["soft_metric"]
    inner_train, inner_validation = joint_v1.inner_architecture_split(
        outer_train, int(seed), float(config["split"]["inner_train_fraction"]),
    )
    inner_train_ids = {str(row["joint_fingerprint"]) for row in inner_train}
    inner_validation_ids = {str(row["joint_fingerprint"]) for row in inner_validation}
    _require(not inner_train_ids & inner_validation_ids, "inner architecture split leakage")
    common_pair_kwargs = {
        "delta": float(soft["delta"]),
        "minimum_weight": float(soft["minimum_pair_weight"]),
        "reliability_fraction": float(soft["reliability_weight_fraction"]),
        "pair_cap": int(soft["pair_cap"]),
        "sampling_seed": int(soft["pair_sampling_seed"]),
    }
    fit_pairs = construct_soft_pairs(
        inner_train, allowed_joint_fingerprints=inner_train_ids, **common_pair_kwargs,
    )
    validation_pairs = construct_soft_pairs(
        inner_validation, allowed_joint_fingerprints=inner_validation_ids,
        **common_pair_kwargs,
    )
    joint_v1._seed_everything(int(seed), torch)
    model_class, _regressor_class = joint_v1._joint_model_classes(torch)
    early_model = model_class()
    early_initial_hash = joint_v1._state_tensor_hash(early_model)
    optimizer = torch.optim.Adam(
        early_model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    fit_graphs = joint_v1._graph_tensors(inner_train, torch)
    validation_graphs = joint_v1._graph_tensors(inner_validation, torch)
    fit_hp = torch.tensor(joint_v1.transform_hp(inner_train, hp_standardizer), dtype=torch.float32)
    validation_hp = torch.tensor(
        joint_v1.transform_hp(inner_validation, hp_standardizer), dtype=torch.float32,
    )
    best_epoch = 0
    best_validation_loss = math.inf
    stale = 0
    epochs_run = 0
    training_losses: list[float] = []
    validation_losses: list[float] = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        epochs_run = epoch
        early_model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_soft_metric_loss(
            early_model(fit_graphs, fit_hp, "full_joint"), fit_pairs, torch,
        )
        loss.backward()
        optimizer.step()
        early_model.eval()
        with torch.no_grad():
            validation_loss = float(weighted_soft_metric_loss(
                early_model(validation_graphs, validation_hp, "full_joint"),
                validation_pairs, torch,
            ))
        training_losses.append(float(loss.detach()))
        validation_losses.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    _require(best_epoch > 0, "inner validation selected no best epoch")
    if stage_callback is not None:
        stage_callback({
            "stage": "inner_early_stop", "model_identity": id(early_model),
            "fit_joint_fingerprints": sorted(inner_train_ids),
            "validation_joint_fingerprints": sorted(inner_validation_ids),
            "fit_pair_endpoint_fingerprints": sorted(inner_train_ids),
            "validation_pair_endpoint_fingerprints": sorted(inner_validation_ids),
            "best_epoch": best_epoch, "epochs": epochs_run,
        })

    joint_v1._seed_everything(int(seed), torch)
    full_graphs = joint_v1._graph_tensors(outer_train, torch)
    full_hp = torch.tensor(joint_v1.transform_hp(outer_train, hp_standardizer), dtype=torch.float32)
    holder: dict[str, Any] = {}

    def factory() -> Any:
        model = model_class()
        holder["initial_hash"] = joint_v1._state_tensor_hash(model)
        return model

    def refit_epoch(model: Any, rows: Sequence[Mapping[str, Any]], epoch: int) -> None:
        _require(
            [str(row["joint_fingerprint"]) for row in rows]
            == [str(row["joint_fingerprint"]) for row in outer_train],
            "final metric refit did not receive complete ordered outer-train",
        )
        if "optimizer" not in holder:
            holder["optimizer"] = torch.optim.Adam(
                model.parameters(), lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
        final_optimizer = holder["optimizer"]
        model.train()
        final_optimizer.zero_grad(set_to_none=True)
        loss = weighted_soft_metric_loss(
            model(full_graphs, full_hp, "full_joint"), full_pair_inventory, torch,
        )
        loss.backward()
        final_optimizer.step()

    final_model, refit = joint_v1.fresh_full_outer_refit(
        early_model, outer_train, best_epoch, factory, refit_epoch,
    )
    if stage_callback is not None:
        stage_callback({
            "stage": "full_outer_refit", "model_identity": id(final_model),
            "fit_joint_fingerprints": sorted(train_ids),
            "best_epoch": best_epoch, "epochs": best_epoch,
        })
    final_model.eval()
    heldout_graphs = joint_v1._graph_tensors(outer_heldout, torch)
    heldout_hp = torch.tensor(
        joint_v1.transform_hp(outer_heldout, hp_standardizer), dtype=torch.float32,
    )
    with torch.no_grad():
        train_embeddings = final_model(full_graphs, full_hp, "full_joint").cpu().numpy().astype("float64")
        heldout_embeddings = final_model(
            heldout_graphs, heldout_hp, "full_joint",
        ).cpu().numpy().astype("float64")
    _require(train_embeddings.shape == (len(outer_train), 8), "train embedding shape mismatch")
    _require(heldout_embeddings.shape == (len(outer_heldout), 8), "heldout embedding shape mismatch")
    _require(
        bool(np.allclose(np.linalg.norm(train_embeddings, axis=1), 1.0, atol=1e-5)),
        "train embeddings are not L2 normalized",
    )
    _require(
        bool(np.allclose(np.linalg.norm(heldout_embeddings, axis=1), 1.0, atol=1e-5)),
        "heldout embeddings are not L2 normalized",
    )
    audit = {
        "seed": int(seed), "loss": "uncertainty_weighted_SmoothL1_soft_similarity",
        "inner_train_joint_fingerprints": sorted(inner_train_ids),
        "inner_validation_joint_fingerprints": sorted(inner_validation_ids),
        "inner_train_pair_summary": summarize_soft_pairs(fit_pairs),
        "inner_validation_pair_summary": summarize_soft_pairs(validation_pairs),
        "inner_validation_entered_inner_train_pairs": False,
        "outer_heldout_labels_used_for_pairs_or_training": False,
        "best_epoch": best_epoch, "early_stop_epochs_run": epochs_run,
        "best_inner_validation_loss": best_validation_loss,
        "training_loss_first": training_losses[0], "training_loss_last": training_losses[-1],
        "validation_loss_first": validation_losses[0], "validation_loss_last": validation_losses[-1],
        "early_model_identity": id(early_model), "final_model_identity": id(final_model),
        "fresh_model_for_full_outer_refit": bool(refit["fresh_model_for_full_outer_refit"]),
        "initial_weights_reproduced_from_same_seed": early_initial_hash == str(holder["initial_hash"]),
        "final_refit_joint_fingerprints": refit["final_refit_joint_fingerprints"],
        "final_refit_used_complete_outer_train": sorted(train_ids) == sorted(refit["final_refit_joint_fingerprints"]),
        "final_refit_epochs": refit["final_refit_epochs"],
        "full_outer_pair_summary": summarize_soft_pairs(full_pair_inventory),
        "hard_triplets_or_pair_power_gate_used": False,
        "hp_residualization_used": False,
    }
    return train_embeddings, heldout_embeddings, final_model, audit


def fusion_contract_probe(architecture_projection: Any, hp_representation: Any) -> Any:
    return joint_v1.fusion_contract_probe(architecture_projection, hp_representation)


def _fuse_components(model: Any, architecture: Any, hp: Any, torch: Any) -> Any:
    interaction = architecture * hp
    joint = torch.cat((architecture, hp, interaction), dim=1)
    return torch.nn.functional.normalize(model.fusion(joint), p=2, dim=1)


def control_embeddings_from_model(
    model: Any, rows: Sequence[Mapping[str, Any]], hp_standardizer: Mapping[str, Any],
    *, mode: str, fold: int, seed: int = 0, scope: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ProtocolError("PyTorch and NumPy are required for representation controls") from error
    allowed = {
        "full_joint", "architecture_only", "hp_only",
        "architecture_shuffled", "hp_shuffled",
    }
    _require(mode in allowed, f"unknown representation control: {mode}")
    if scope is None:
        scope = (
            "within_outer_fold_heldout_representations"
            if mode in {"architecture_shuffled", "hp_shuffled"}
            else "outer_fold_rows"
        )
    if mode in {"architecture_shuffled", "hp_shuffled"}:
        _require(
            scope in {"outer_heldout", "within_outer_fold_heldout_representations"},
            "shuffle is allowed only on held-out representations",
        )
    model.eval()
    graphs = joint_v1._graph_tensors(rows, torch)
    hp_features = torch.tensor(joint_v1.transform_hp(rows, hp_standardizer), dtype=torch.float32)
    with torch.no_grad():
        _embedding, components = model(
            graphs, hp_features, "full_joint", return_components=True,
        )
        architecture, hp, _interaction = components
        permutation = list(range(len(rows)))
        if mode in {"architecture_shuffled", "hp_shuffled"}:
            random.Random(int(seed) + int(fold) * 1_000_003).shuffle(permutation)
            indices = torch.tensor(permutation, dtype=torch.long)
            if mode == "architecture_shuffled":
                architecture = architecture[indices]
            else:
                hp = hp[indices]
        elif mode == "architecture_only":
            hp = torch.zeros_like(hp)
        elif mode == "hp_only":
            architecture = torch.zeros_like(architecture)
        controlled = _fuse_components(model, architecture, hp, torch)
    values = controlled.cpu().numpy().astype("float64")
    _require(
        bool(np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5)),
        "control embedding is not L2 normalized",
    )
    return values, {
        "mode": mode, "fold": int(fold), "seed": int(seed),
        "permutation": permutation,
        "permutation_sha256": stable_hash(permutation),
        "scope": scope,
        "reference_embeddings_shuffled": False,
        "retrained": False,
        "architecture_multiset_preserved": mode != "architecture_shuffled" or sorted(permutation) == list(range(len(rows))),
        "hp_multiset_preserved": mode != "hp_shuffled" or sorted(permutation) == list(range(len(rows))),
    }


def average_metric_seed_predictions(predictions: Mapping[int, Any]) -> Any:
    try:
        return joint_v1.average_metric_seed_predictions(predictions)
    except joint_v1.ProtocolError as error:
        raise ProtocolError(str(error)) from error


def _repository_path(raw: str) -> Path:
    path = (REPO_ROOT / str(raw)).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ProtocolError(f"configured path escapes repository: {raw}") from error
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_input_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    identities = config["authoritative_inputs"]
    output: dict[str, Any] = {"status": "VALID", "checks": {}}
    for label, expected in identities.items():
        root = _repository_path(str(expected["result_root"]))
        archive = _repository_path(str(expected["archive"]))
        marker = root / str(expected["required_marker"])
        state_path = root / "state.json"
        decision_path = root / "decision.json"
        for path in (marker, state_path, decision_path, archive):
            _require(path.is_file(), f"authoritative {label} input is missing: {path}", identity=True)
        state = _read_json(state_path)
        decision = _read_json(decision_path)
        checks = {
            "complete": state.get("status") == "complete",
            "state_classification": state.get("decision") == str(expected["classification"]),
            "decision_classification": decision.get("classification") == str(expected["classification"]),
            "archive_sha256": sha256_file(archive) == str(expected["archive_sha256"]),
            "config_sha256": state.get("config_sha256") == str(expected["config_sha256"]),
            "source_sha256": state.get("source_sha256") == str(expected["source_sha256"]),
        }
        if label == "joint_metric_v1":
            checks["split_sha256"] = state.get("split_sha256") == str(expected["split_sha256"])
            checks["test_not_accessed"] = state.get("test_data_accessed") is False
            checks["candidate_training_not_started"] = state.get("candidate_gnn_training_started") is False
        elif label == "pair_power_audit_v1":
            checks["split_sha256"] = state.get("authoritative_v1_split_sha256") == str(
                config["split"]["expected_sha256"]
            )
            checks["first_projection_n"] = int(decision.get("first_n_where_all_five_folds_pass", -1)) == int(
                expected["first_all_five_fold_projection_n"]
            )
            checks["point_limit_all_pass"] = decision.get("point_limit_fold_passes") == [True] * 5
        failed = sorted(key for key, passed in checks.items() if not passed)
        _require(not failed, f"authoritative {label} identity mismatch: {failed}", identity=True)
        output[label] = {
            "result_root": str(root), "archive": str(archive),
            "classification": decision["classification"],
            "archive_sha256": sha256_file(archive),
            "config_sha256": state["config_sha256"],
            "source_sha256": state["source_sha256"], "checks": checks,
        }
        output["checks"].update({f"{label}.{key}": value for key, value in checks.items()})
    return output


def assert_input_identity(inventory: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    data = config["data"]
    checks = {
        "seed_records": int(inventory.get("raw_row_count", -1)) == int(data["expected_seed_records"]),
        "successful_seed_records": int(inventory.get("successful_required_seed_row_count", -1)) == int(data["expected_seed_records"]),
        "joint_configurations": int(inventory.get("unique_joint_configurations", -1)) == int(data["expected_unique_joint_configurations"]),
        "architectures": int(inventory.get("unique_architectures", -1)) == int(data["expected_unique_architectures"]),
        "pools": int(inventory.get("unique_pools", -1)) == int(data["expected_unique_pools"]),
        "seeds": list(inventory.get("complete_training_seeds", [])) == [0, 1, 2],
        "target": inventory.get("target") == "raw_validation_accuracy_seed_mean",
        "no_residualization": inventory.get("hp_residualization_used") is False,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    _require(not failed, f"main validation input identity mismatch: {failed}", identity=True)


def validate_frozen_split_manifest(
    configurations: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], expected_sha256: str,
) -> dict[str, Any]:
    payload = dict(manifest)
    recorded = str(payload.pop("split_sha256", ""))
    rebuilt = joint_v1.stable_hash(payload)
    _require(
        recorded == rebuilt == str(expected_sha256),
        "frozen outer split logical SHA-256 mismatch",
        identity=True,
    )
    try:
        joint_v1.assert_split_integrity(configurations, manifest)
    except joint_v1.ProtocolError as error:
        raise InputIdentityMismatch(str(error)) from error
    _require(len(manifest.get("folds", [])) == 5, "frozen split is not five-fold", identity=True)
    return {
        "split_sha256": rebuilt, "fold_count": 5,
        "split_kind": manifest.get("split_kind"),
        "source": "joint metric v1 frozen split manifest",
        "joint_architecture_pool_integrity_verified": True,
        "repartitioned": False,
    }


def load_main_data_and_split(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        configurations, inventory, provenance = joint_v1.load_and_aggregate_source(config)
    except joint_v1.ProtocolError as error:
        raise InputIdentityMismatch(str(error)) from error
    inventory = {
        **inventory,
        "unique_pools": len({int(row["pool_id"]) for row in configurations}),
        "metric_training_seeds": [0, 1, 2],
        "stabilization_seeds_used_for_training": False,
        "candidate_gnn_training_started": False,
        "test_data_accessed": False,
    }
    assert_input_identity(inventory, config)
    for row in configurations:
        summary = aggregate_three_seed_validation([
            row["validation_seed_0"], row["validation_seed_1"], row["validation_seed_2"],
        ])
        _require(math.isclose(summary["mean"], float(row["validation_mean"]), abs_tol=1e-15),
                 "raw validation mean aggregation mismatch", identity=True)
        _require(math.isclose(summary["sd"], float(row["validation_sd"]), abs_tol=1e-15),
                 "raw validation SD aggregation mismatch", identity=True)
        _require(math.isclose(summary["se"], float(row["validation_se"]), abs_tol=1e-15),
                 "raw validation SE aggregation mismatch", identity=True)
    split_path = _repository_path(str(config["split"]["source_manifest"]))
    _require(split_path.is_file(), "frozen v1 split manifest is missing", identity=True)
    manifest = _read_json(split_path)
    split_audit = validate_frozen_split_manifest(
        configurations, manifest, str(config["split"]["expected_sha256"]),
    )
    provenance = {
        **provenance, "split_manifest_path": str(split_path),
        "split_manifest_logical_sha256": split_audit["split_sha256"],
        "stabilization_read_for_training": False,
    }
    return configurations, inventory, provenance, manifest


def _spearman(left: Any, right: Any) -> float:
    import numpy as np
    from scipy.stats import spearmanr
    value = float(spearmanr(np.asarray(left), np.asarray(right)).statistic)
    return value if math.isfinite(value) else 0.0


def geometry_for_fold_soft(
    train_rows: Sequence[Mapping[str, Any]], heldout_rows: Sequence[Mapping[str, Any]],
    seed_distance_matrices: Mapping[int, Any], explicit_distance_matrix: Any,
    hp_standardizer: Mapping[str, Any], fold: int,
) -> dict[str, Any]:
    import numpy as np
    _require(set(seed_distance_matrices) == set(METRIC_SEEDS), "geometry requires metric seeds 0,1,2")
    train_y = np.asarray([float(row["validation_mean"]) for row in train_rows], dtype=np.float64)
    heldout_y = np.asarray([float(row["validation_mean"]) for row in heldout_rows], dtype=np.float64)
    gaps = np.abs(heldout_y[:, None] - train_y[None, :])
    seed_spearman = {
        str(seed): _spearman(np.asarray(seed_distance_matrices[seed]).reshape(-1), gaps.reshape(-1))
        for seed in METRIC_SEEDS
    }
    metric_distances = np.mean(
        np.stack([np.asarray(seed_distance_matrices[seed], dtype=np.float64) for seed in METRIC_SEEDS]),
        axis=0,
    )
    explicit_distances = np.asarray(explicit_distance_matrix, dtype=np.float64)
    _require(metric_distances.shape == gaps.shape == explicit_distances.shape,
             "geometry distance matrix shape mismatch")

    def local_gap(distance_matrix: Any) -> dict[str, Any]:
        distance = np.asarray(distance_matrix, dtype=np.float64)
        records: list[dict[str, Any]] = []
        for query_index, query in enumerate(heldout_rows):
            order = sorted(
                range(len(train_rows)),
                key=lambda index: (
                    float(distance[query_index, index]),
                    str(train_rows[index]["joint_fingerprint"]),
                ),
            )[:5]
            neighbor_gaps = [float(gaps[query_index, index]) for index in order]
            records.append({
                "fold": int(fold),
                "query_joint_fingerprint": str(query["joint_fingerprint"]),
                "neighbor_joint_fingerprints": [
                    str(train_rows[index]["joint_fingerprint"]) for index in order
                ],
                "neighbor_performance_gaps": neighbor_gaps,
                "local_performance_gap_at_5": sum(neighbor_gaps) / 5.0,
            })
        return {
            "mean_local_performance_gap_at_5": float(np.mean([
                row["local_performance_gap_at_5"] for row in records
            ])),
            "median_local_performance_gap_at_5": float(np.median([
                row["local_performance_gap_at_5"] for row in records
            ])),
            "query_count": len(records), "records": records,
        }

    hp_train = joint_v1.transform_hp(train_rows, hp_standardizer)
    hp_heldout = joint_v1.transform_hp(heldout_rows, hp_standardizer)
    train_hp_distances = [
        float(np.linalg.norm(hp_train[left] - hp_train[right]))
        for left, right in itertools.combinations(range(len(train_rows)), 2)
    ]
    hp_median = float(np.median(np.asarray(train_hp_distances, dtype=np.float64)))
    cross_cells: list[dict[str, Any]] = []
    for query_index, query in enumerate(heldout_rows):
        for reference_index, reference in enumerate(train_rows):
            architecture_different = (
                str(query["architecture_fingerprint"])
                != str(reference["architecture_fingerprint"])
            )
            hp_far = float(np.linalg.norm(
                hp_heldout[query_index] - hp_train[reference_index]
            )) > hp_median
            if architecture_different and hp_far:
                cross_cells.append({
                    "query": str(query["joint_fingerprint"]),
                    "reference": str(reference["joint_fingerprint"]),
                    "performance_gap": float(gaps[query_index, reference_index]),
                    "metric_distance": float(metric_distances[query_index, reference_index]),
                })
    _require(bool(cross_cells), "cross-factor heldout/train cell is empty")
    ordered = sorted(
        cross_cells,
        key=lambda row: (row["performance_gap"], row["query"], row["reference"]),
    )
    group_size = max(1, int(math.ceil(0.20 * len(ordered))))
    low = ordered[:group_size]
    high = ordered[-group_size:]
    low_distances = [float(row["metric_distance"]) for row in low]
    high_distances = [float(row["metric_distance"]) for row in high]
    ratio = float(np.median(low_distances) / max(float(np.median(high_distances)), 1e-12))
    return {
        "fold": int(fold), "seed_distance_gap_spearman": seed_spearman,
        "metric_seed_distance_aggregation": "arithmetic_mean_distance_matrix",
        "joint_metric_knn": local_gap(metric_distances),
        "explicit_joint_feature_knn": local_gap(explicit_distances),
        "cross_factor_hp_distance_train_median": hp_median,
        "cross_factor_cell_count": len(ordered), "cross_factor_group_size": group_size,
        "cross_factor_low_gap_metric_distances": low_distances,
        "cross_factor_high_gap_metric_distances": high_distances,
        "cross_factor_low_gap_performance_gaps": [float(row["performance_gap"]) for row in low],
        "cross_factor_high_gap_performance_gaps": [float(row["performance_gap"]) for row in high],
        "cross_factor_locality_ratio": ratio,
        "cross_factor_group_definition": "deterministic bottom/top ceil(20%) rank within fold",
        "pair_scope": "heldout_query_to_outer_train_reference_only",
        "training_set_internal_geometry_used_as_main_evidence": False,
    }


def aggregate_geometry_soft(fold_results: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    folds = [result["geometry"] for result in fold_results]
    low_distances = [
        float(value) for fold in folds for value in fold["cross_factor_low_gap_metric_distances"]
    ]
    high_distances = [
        float(value) for fold in folds for value in fold["cross_factor_high_gap_metric_distances"]
    ]
    metric_records = [record for fold in folds for record in fold["joint_metric_knn"]["records"]]
    explicit_records = [record for fold in folds for record in fold["explicit_joint_feature_knn"]["records"]]
    _require(len(metric_records) == len(explicit_records), "kNN geometry query coverage mismatch")
    metric_gap = float(np.mean([row["local_performance_gap_at_5"] for row in metric_records]))
    explicit_gap = float(np.mean([row["local_performance_gap_at_5"] for row in explicit_records]))
    gap_reduction = (explicit_gap - metric_gap) / max(abs(explicit_gap), 1e-12)
    geometry = {
        "format_version": FORMAT_VERSION,
        "folds": [{
            "fold": int(fold["fold"]),
            "seed_distance_gap_spearman": fold["seed_distance_gap_spearman"],
            "cross_factor_locality_ratio": fold["cross_factor_locality_ratio"],
            "cross_factor_cell_count": fold["cross_factor_cell_count"],
            "cross_factor_group_size": fold["cross_factor_group_size"],
            "joint_metric_mean_gap_at_5": fold["joint_metric_knn"]["mean_local_performance_gap_at_5"],
            "explicit_joint_mean_gap_at_5": fold["explicit_joint_feature_knn"]["mean_local_performance_gap_at_5"],
        } for fold in folds],
        "cross_factor_locality_ratio": float(
            np.median(np.asarray(low_distances, dtype=np.float64))
            / max(float(np.median(np.asarray(high_distances, dtype=np.float64))), 1e-12)
        ),
        "cross_factor_low_gap_pair_count": len(low_distances),
        "cross_factor_high_gap_pair_count": len(high_distances),
        "joint_metric_local_performance_gap_at_5": metric_gap,
        "explicit_joint_feature_local_performance_gap_at_5": explicit_gap,
        "local_gap5_relative_reduction": gap_reduction,
        "pair_scope": "pooled heldout-query to outer-train references",
        "training_set_internal_geometry_used_as_main_evidence": False,
    }
    locality = {
        "format_version": FORMAT_VERSION,
        "joint_metric": {
            "mean_local_performance_gap_at_5": metric_gap,
            "query_count": len(metric_records), "records": metric_records,
        },
        "explicit_joint_feature": {
            "mean_local_performance_gap_at_5": explicit_gap,
            "query_count": len(explicit_records), "records": explicit_records,
        },
        "relative_reduction": gap_reduction, "neighbors": 5,
        "metric_distance": "raw Euclidean distance in L2-normalized 8D embedding",
    }
    return geometry, locality


def evaluate_outer_fold_soft(
    fold: Mapping[str, Any], configurations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any], full_pair_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
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
    _require(len(train_rows) == len(train_ids) and len(heldout_rows) == len(heldout_ids),
             "outer fold lookup is incomplete")
    fold_id = int(fold["fold"])
    hp_standardizer = joint_v1.fit_hp_standardizer(train_rows)
    train_y = np.asarray([float(row["validation_mean"]) for row in train_rows], dtype=np.float64)
    heldout_y = np.asarray([float(row["validation_mean"]) for row in heldout_rows], dtype=np.float64)
    train_se = np.asarray([float(row["validation_se"]) for row in train_rows], dtype=np.float64)
    train_hp = joint_v1._raw_hp_matrix(train_rows)
    heldout_hp = joint_v1._raw_hp_matrix(heldout_rows)
    train_graph = joint_v1._explicit_graph_matrix(train_rows)
    heldout_graph = joint_v1._explicit_graph_matrix(heldout_rows)
    train_joint = np.concatenate((train_graph, train_hp), axis=1)
    heldout_joint = np.concatenate((heldout_graph, heldout_hp), axis=1)
    explicit_train, explicit_heldout, explicit_transform = joint_v1.standardize_train_query(
        train_joint, heldout_joint,
    )
    explicit_distances = np.linalg.norm(
        explicit_heldout[:, None, :] - explicit_train[None, :, :], axis=2,
    )
    train_fingerprints = [str(row["joint_fingerprint"]) for row in train_rows]
    heldout_fingerprints = [str(row["joint_fingerprint"]) for row in heldout_rows]
    predictions: dict[str, Any] = {
        "B0": np.full(len(heldout_rows), float(np.mean(train_y)), dtype=np.float64),
    }
    gp_audits: dict[str, Any] = {
        "B0": {"method": "outer_train_mean", "outer_train_only": True},
    }
    predictions["B1"], gp_audits["B1"] = joint_v1.exact_matern_gp_predict(
        train_hp, train_y, train_se, heldout_hp,
        standardize_inputs=True, ard=True, protocol_name="B1",
    )
    predictions["B2"], gp_audits["B2"] = joint_v1.exact_matern_gp_predict(
        train_graph, train_y, train_se, heldout_graph,
        standardize_inputs=True, ard=True, protocol_name="B2",
    )
    predictions["B3"], gp_audits["B3"] = joint_v1.exact_matern_gp_predict(
        train_joint, train_y, train_se, heldout_joint,
        standardize_inputs=True, ard=True, protocol_name="B3",
    )
    predictions["EJ_KNN"] = joint_v1.weighted_knn_predict(
        explicit_train, train_y, explicit_heldout, 5,
        train_fingerprints=train_fingerprints, query_fingerprints=heldout_fingerprints,
    )

    direct_seed_predictions: dict[int, Any] = {}
    direct_audits: list[dict[str, Any]] = []
    for seed in METRIC_SEEDS:
        prediction, audit = joint_v1.train_direct_joint_regressor(
            train_rows, heldout_rows, seed, config, hp_standardizer,
        )
        direct_seed_predictions[seed] = prediction
        direct_audits.append(audit)
    predictions["B4"] = average_metric_seed_predictions(direct_seed_predictions)

    seed_predictions: dict[str, dict[int, Any]] = {
        model: {} for model in (
            "M1", "M2", "M3", "M4", "ABL_ARCH_M1", "ABL_ARCH_M2",
            "ABL_HP_M1", "ABL_HP_M2", "M2_architecture_shuffled", "M2_hp_shuffled",
        )
    }
    metric_audits: list[dict[str, Any]] = []
    seed_distances: dict[int, Any] = {}
    local_gp_audits: dict[str, Any] = {}
    control_audits: list[dict[str, Any]] = []
    for seed in METRIC_SEEDS:
        train_embedding, heldout_embedding, model, training_audit = train_soft_metric_encoder(
            train_rows, heldout_rows, seed, config, hp_standardizer, full_pair_inventory,
        )
        training_audit["fold"] = fold_id
        metric_audits.append(training_audit)
        seed_predictions["M1"][seed] = joint_v1.weighted_knn_predict(
            train_embedding, train_y, heldout_embedding, 5,
            train_fingerprints=train_fingerprints, query_fingerprints=heldout_fingerprints,
        )
        seed_predictions["M2"][seed], training_audit["M2_gp"] = joint_v1.exact_matern_gp_predict(
            train_embedding, train_y, train_se, heldout_embedding,
            standardize_inputs=False, ard=False, protocol_name="M2",
        )
        seed_predictions["M3"][seed], local_audit = joint_v1.local_gp_predict(
            train_embedding, train_y, train_se, heldout_embedding,
            int(config["metric_methods"]["M3"]["neighbors"]),
        )
        local_gp_audits[str(seed)] = {
            "query_count": len(local_audit), "neighbors": 32,
            "all_isotropic_unstandardized": all(
                item["gp"]["ard"] is False and item["gp"]["input_standardized"] is False
                for item in local_audit
            ),
        }
        seed_predictions["M4"][seed], training_audit["M4_gp"] = joint_v1.exact_matern_gp_predict(
            train_embedding, train_y, train_se, heldout_embedding,
            standardize_inputs=True, ard=True, protocol_name="M4",
        )
        seed_distances[seed] = np.linalg.norm(
            heldout_embedding[:, None, :] - train_embedding[None, :, :], axis=2,
        )
        for short, mode in (("ARCH", "architecture_only"), ("HP", "hp_only")):
            branch_train, audit_train = control_embeddings_from_model(
                model, train_rows, hp_standardizer, mode=mode, fold=fold_id, seed=0,
                scope="outer_train",
            )
            branch_heldout, audit_heldout = control_embeddings_from_model(
                model, heldout_rows, hp_standardizer, mode=mode, fold=fold_id, seed=0,
                scope="outer_heldout",
            )
            seed_predictions[f"ABL_{short}_M1"][seed] = joint_v1.weighted_knn_predict(
                branch_train, train_y, branch_heldout, 5,
                train_fingerprints=train_fingerprints, query_fingerprints=heldout_fingerprints,
            )
            seed_predictions[f"ABL_{short}_M2"][seed], branch_gp = joint_v1.exact_matern_gp_predict(
                branch_train, train_y, train_se, branch_heldout,
                standardize_inputs=False, ard=False, protocol_name="M2",
            )
            control_audits.append({
                "seed": seed, "branch": mode, "train": audit_train,
                "heldout": audit_heldout, "gp": branch_gp,
            })
        for label, mode in (
            ("M2_architecture_shuffled", "architecture_shuffled"),
            ("M2_hp_shuffled", "hp_shuffled"),
        ):
            shuffled, shuffle_audit = control_embeddings_from_model(
                model, heldout_rows, hp_standardizer, mode=mode,
                fold=fold_id, seed=int(config["controls"]["shuffle_seed"]),
                scope="outer_heldout",
            )
            seed_predictions[label][seed], shuffle_gp = joint_v1.exact_matern_gp_predict(
                train_embedding, train_y, train_se, shuffled,
                standardize_inputs=False, ard=False, protocol_name="M2",
            )
            control_audits.append({
                "seed": seed, "branch": mode, "heldout": shuffle_audit,
                "gp": shuffle_gp,
            })
    for model, by_seed in seed_predictions.items():
        predictions[model] = average_metric_seed_predictions(by_seed)
    geometry = geometry_for_fold_soft(
        train_rows, heldout_rows, seed_distances, explicit_distances,
        hp_standardizer, fold_id,
    )
    oof_rows: list[dict[str, Any]] = []
    for index, row in enumerate(heldout_rows):
        output = {
            "fold": fold_id, "joint_fingerprint": str(row["joint_fingerprint"]),
            "architecture_fingerprint": str(row["architecture_fingerprint"]),
            "hp_fingerprint": str(row["hp_fingerprint"]),
            "pool_id": int(row["pool_id"]), "batch_id": int(row["batch_id"]),
            "slot_id": int(row["slot_id"]),
            "raw_validation_mean": float(heldout_y[index]),
            "validation_se": float(row["validation_se"]),
        }
        for model, values in predictions.items():
            output[f"{model}_prediction"] = float(values[index])
        oof_rows.append(output)
    return {
        "fold": fold_id, "oof_rows": oof_rows, "geometry": geometry,
        "metric_training_audits": metric_audits,
        "direct_regressor_audits": direct_audits,
        "gp_audits": gp_audits, "local_gp_audits": local_gp_audits,
        "control_audits": control_audits,
        "hp_standardizer": hp_standardizer,
        "explicit_joint_feature_transform": explicit_transform,
        "metric_seed_aggregation": "arithmetic_mean_all_seeds_0_1_2",
        "best_metric_seed_selected": False,
        "outer_heldout_labels_used_for_training": False,
    }


def _relative_improvement(baseline: float, challenger: float) -> float:
    return (float(baseline) - float(challenger)) / max(abs(float(baseline)), 1.0e-12)


def _composite_prediction_gate(
    challenger: str, baseline: str,
    predictions: Mapping[str, Mapping[str, Any]], best_of_four: Mapping[str, Any],
    thresholds: Mapping[str, Any], *, require_fold_consistency: bool,
) -> dict[str, Any]:
    """Apply one model's complete metric tuple against one fixed baseline."""
    result = joint_v1._model_composite_gate(
        challenger, baseline, predictions, best_of_four, thresholds,
        require_fold_consistency=require_fold_consistency,
    )
    result["cross_model_metric_splicing_used"] = False
    return result


def _branch_contribution(
    full: str, branch: str, predictions: Mapping[str, Mapping[str, Any]],
    best_of_four: Mapping[str, Any], required_count: int,
) -> dict[str, Any]:
    comparisons = {
        "mae": float(predictions[full]["mae"]) < float(predictions[branch]["mae"]),
        "top20_precision": (
            float(predictions[full]["top20_precision"])
            > float(predictions[branch]["top20_precision"])
        ),
        "best_of_4_regret": (
            float(best_of_four["models"][full]["mean_simple_regret"])
            < float(best_of_four["models"][branch]["mean_simple_regret"])
        ),
    }
    count = sum(bool(value) for value in comparisons.values())
    return {
        "full_model": full, "branch_model": branch,
        "strictly_better": comparisons, "better_metric_count": count,
        "required_better_metric_count": int(required_count),
        "passed": count >= int(required_count),
    }


def _shuffle_degradation(
    full: str, shuffled: str, predictions: Mapping[str, Mapping[str, Any]],
    best_of_four: Mapping[str, Any], thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    full_metrics = predictions[full]
    shuffled_metrics = predictions[shuffled]
    full_regret = float(best_of_four["models"][full]["mean_simple_regret"])
    shuffled_regret = float(best_of_four["models"][shuffled]["mean_simple_regret"])
    values = {
        "mae_relative_degradation": (
            float(shuffled_metrics["mae"]) - float(full_metrics["mae"])
        ) / max(abs(float(full_metrics["mae"])), 1.0e-12),
        "top20_precision_absolute_decrease": (
            float(full_metrics["top20_precision"])
            - float(shuffled_metrics["top20_precision"])
        ),
        "regret_relative_degradation": (
            shuffled_regret - full_regret
        ) / max(abs(full_regret), 1.0e-12),
    }
    checks = {
        "mae": values["mae_relative_degradation"]
        >= float(thresholds["mae_relative_degradation_at_least"]),
        "top20_precision": values["top20_precision_absolute_decrease"]
        >= float(thresholds["top20_precision_absolute_decrease_at_least"]),
        "best_of_4_regret": values["regret_relative_degradation"]
        >= float(thresholds["regret_relative_degradation_at_least"]),
    }
    return {
        "full_model": full, "shuffled_model": shuffled, **values,
        "threshold_checks": checks, "passed": any(checks.values()),
    }


def classify_gate_states(
    *, geometry_passed: bool, m1_passed: bool, m2_passed: bool,
    joint_contribution_passed: bool, feature_diagnostic_passed: bool,
) -> str:
    """Frozen classification state machine; no calibration input is accepted."""
    if not geometry_passed:
        return "SOFT_JOINT_METRIC_REJECTED"
    if m2_passed:
        if not m1_passed and joint_contribution_passed:
            raise SpecUnderdefinedError(
                "SPEC_UNDERDEFINED: geometry and M2 pass, M1 fails, and joint "
                "contribution passes; the frozen prompt assigns no classification"
            )
        if not joint_contribution_passed:
            return "SOFT_JOINT_METRIC_NOT_JOINT"
        return "SOFT_JOINT_METRIC_GP_READY"
    if m1_passed:
        return "SOFT_JOINT_METRIC_LOCALITY_ONLY"
    if feature_diagnostic_passed:
        return "SOFT_JOINT_METRIC_FEATURE_ONLY"
    return "SOFT_JOINT_METRIC_GEOMETRY_ONLY"


def classify_soft_joint_metric_gate(
    geometry: Mapping[str, Any], predictions: Mapping[str, Mapping[str, Any]],
    best_of_four: Mapping[str, Any], decision_config: Mapping[str, Any],
) -> dict[str, Any]:
    geometry_thresholds = decision_config["geometry_gate"]
    seed_fold_values: dict[str, list[float]] = {str(seed): [] for seed in METRIC_SEEDS}
    for fold in geometry["folds"]:
        for seed in METRIC_SEEDS:
            seed_fold_values[str(seed)].append(
                float(fold["seed_distance_gap_spearman"][str(seed)])
            )
    all_values = [value for values in seed_fold_values.values() for value in values]
    _require(len(all_values) == 15, "geometry gate requires exactly 3 seeds x 5 folds")
    ordered = sorted(all_values)
    median_spearman = ordered[len(ordered) // 2]
    positive_fold_counts = {
        seed: sum(value > 0.0 for value in values)
        for seed, values in seed_fold_values.items()
    }
    positive_seed_count = sum(
        count >= int(geometry_thresholds["positive_folds_per_seed_at_least"])
        for count in positive_fold_counts.values()
    )
    geometry_checks = {
        "G1_median_spearman": median_spearman
        > float(geometry_thresholds["median_fold_spearman_across_metric_seeds_strictly_greater_than"]),
        "G2_positive_seed_fold_pattern": positive_seed_count
        >= int(geometry_thresholds["positive_seed_count_at_least"]),
        "G3_cross_factor_locality": float(geometry["cross_factor_locality_ratio"])
        < float(geometry_thresholds["cross_factor_locality_ratio_strictly_less_than"]),
        "G4_local_gap_at_5": float(geometry["local_gap5_relative_reduction"])
        >= float(geometry_thresholds["local_gap5_relative_reduction_over_explicit_knn_at_least"]),
    }
    geometry_gate = {
        "median_seed_fold_distance_gap_spearman": median_spearman,
        "positive_folds_by_metric_seed": positive_fold_counts,
        "metric_seeds_meeting_positive_fold_requirement": positive_seed_count,
        "cross_factor_locality_ratio": float(geometry["cross_factor_locality_ratio"]),
        "local_gap5_relative_reduction": float(geometry["local_gap5_relative_reduction"]),
        "checks": geometry_checks, "passed": all(geometry_checks.values()),
    }
    strongest = min(
        decision_config["strongest_non_metric_gp"]["candidates"],
        key=lambda model: (float(predictions[str(model)]["mae"]), str(model)),
    )
    m1_gate = _composite_prediction_gate(
        "M1", "EJ_KNN", predictions, best_of_four,
        decision_config["M1_gate_over_explicit_joint_feature_knn"],
        require_fold_consistency=False,
    )
    m2_gate = _composite_prediction_gate(
        "M2", str(strongest), predictions, best_of_four,
        decision_config["M2_gate_over_strongest_non_metric_gp"],
        require_fold_consistency=True,
    )
    feature_gates = {
        model: _composite_prediction_gate(
            model, str(strongest), predictions, best_of_four,
            decision_config["feature_only_diagnostic_gate"],
            require_fold_consistency=True,
        )
        for model in ("M4", "B4")
    }
    contribution = decision_config["joint_contribution_gate"]
    required = int(contribution["required_better_metric_count_vs_each_branch"])
    branch = {
        "versus_hp_only": _branch_contribution(
            "M2", "ABL_HP_M2", predictions, best_of_four, required,
        ),
        "versus_architecture_only": _branch_contribution(
            "M2", "ABL_ARCH_M2", predictions, best_of_four, required,
        ),
    }
    shuffles = {
        "architecture": _shuffle_degradation(
            "M2", "M2_architecture_shuffled", predictions, best_of_four,
            contribution["architecture_shuffle"],
        ),
        "hp": _shuffle_degradation(
            "M2", "M2_hp_shuffled", predictions, best_of_four,
            contribution["hp_shuffle"],
        ),
    }
    joint_passed = (
        all(value["passed"] for value in branch.values())
        and all(value["passed"] for value in shuffles.values())
    )
    feature_passed = any(value["passed"] for value in feature_gates.values())
    classification = classify_gate_states(
        geometry_passed=bool(geometry_gate["passed"]),
        m1_passed=bool(m1_gate["passed"]), m2_passed=bool(m2_gate["passed"]),
        joint_contribution_passed=joint_passed,
        feature_diagnostic_passed=feature_passed,
    )
    _require(classification in ALLOWED_CLASSIFICATIONS, "classifier emitted an unfrozen label")
    return {
        "classification": classification, "geometry_gate": geometry_gate,
        "strongest_non_metric_gp": str(strongest), "M1_gate": m1_gate,
        "M2_gate": m2_gate, "feature_diagnostic_gates": feature_gates,
        "joint_contribution_gate": {
            "branch_comparisons": branch, "shuffle_degradation": shuffles,
            "passed": joint_passed,
        },
        "M4_may_replace_M2": False,
        "cross_model_metric_splicing_used": False,
        "posterior_mean_only": True, "uncertainty_or_acquisition_validated": False,
        "formal_bo_modified": False, "historical_classifications_modified": False,
        "candidate_gnn_training_started": False, "test_data_accessed": False,
    }


def summarize_controls(
    predictions: Mapping[str, Mapping[str, Any]], best_of_four: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    def compact(model: str) -> dict[str, Any]:
        return {
            "model": model, "mae": float(predictions[model]["mae"]),
            "top20_precision": float(predictions[model]["top20_precision"]),
            "mean_best_of_4_regret": float(
                best_of_four["models"][model]["mean_simple_regret"]
            ),
        }
    ablation = {
        "format_version": FORMAT_VERSION,
        "models": [compact(model) for model in (
            "M2", "ABL_ARCH_M2", "ABL_HP_M2", "M1", "ABL_ARCH_M1", "ABL_HP_M1",
        )],
        "joint_contribution": decision["joint_contribution_gate"]["branch_comparisons"],
        "controls_retrained": False,
        "source": "same frozen full-joint encoder; branch representation removed before fusion",
    }
    shuffle = {
        "format_version": FORMAT_VERSION,
        "models": [compact(model) for model in (
            "M2", "M2_architecture_shuffled", "M2_hp_shuffled",
        )],
        "degradation": decision["joint_contribution_gate"]["shuffle_degradation"],
        "seed": 0, "scope": "heldout representations within each outer fold",
        "controls_retrained": False,
    }
    return ablation, shuffle


def build_prediction_artifacts(
    fold_results: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    oof_rows = [dict(row) for result in fold_results for row in result["oof_rows"]]
    identities = [str(row["joint_fingerprint"]) for row in oof_rows]
    _require(len(oof_rows) == 336 and len(set(identities)) == 336,
             "OOF prediction coverage is not exactly the frozen 336 configurations")
    metric_rows: list[dict[str, Any]] = []
    pooled: dict[str, Any] = {}
    for model in MODEL_NAMES:
        pooled_metrics = joint_v1.prediction_metrics(oof_rows, model, 0.20)
        pooled[model] = pooled_metrics
        metric_rows.append({"scope": "pooled_oof", "fold": "ALL", "model": model, **pooled_metrics})
        for fold in range(5):
            fold_rows = [row for row in oof_rows if int(row["fold"]) == fold]
            metrics = joint_v1.prediction_metrics(fold_rows, model, 0.20)
            metric_rows.append({"scope": "outer_fold", "fold": fold, "model": model, **metrics})
    batch_manifest, best_of_four = joint_v1.best_of_four_evaluation(
        oof_rows, config, list(MODEL_NAMES),
    )
    _require(bool(batch_manifest["shared_by_all_models"]), "best-of-4 batches are not shared")
    return oof_rows, pooled, {"format_version": FORMAT_VERSION, "rows": metric_rows}, {
        "manifest": batch_manifest, "results": best_of_four,
    }


def sensitivity_30seed(
    config: Mapping[str, Any], configurations: Sequence[Mapping[str, Any]],
    frozen_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Run only after classification; return descriptive label-drift summaries."""
    before = stable_hash(frozen_decision)
    adapted = copy.deepcopy(dict(config))
    adapted["calibration"] = copy.deepcopy(dict(config["sensitivity_30seed"]))
    try:
        subset, inventory = pair_audit.load_calibration_subset(adapted, configurations)
    except pair_audit.CalibrationUnavailable as error:
        return {
            "status": "UNAVAILABLE", "reason": str(error),
            "classification_before": str(frozen_decision["classification"]),
            "classification_after": str(frozen_decision["classification"]),
            "classification_changed": False, "used_for_training": False,
            "used_for_parameter_or_threshold_selection": False,
            "nearest_neighbor_stability": {
                "status": "SPEC_UNDERDEFINED",
                "reason": "the prompt freezes no reference set, distance aggregation, or statistic",
            },
        }
    except pair_audit.InputInvalidError as error:
        raise InputIdentityMismatch(f"30-seed sensitivity identity conflict: {error}") from error
    try:
        import numpy as np
        from scipy.stats import kendalltau, spearmanr
    except ImportError as error:
        raise ProtocolError("NumPy and SciPy are required for 30-seed sensitivity") from error
    records: list[dict[str, Any]] = []
    for joint in sorted(subset):
        values = [float(value) for value in subset[joint]["values"]]
        _require(len(values) == 30, "30-seed reference is incomplete", identity=True)
        mean3 = float(np.mean(np.asarray(values[:3], dtype=np.float64)))
        mean30 = float(np.mean(np.asarray(values, dtype=np.float64)))
        records.append({
            "joint_fingerprint": joint,
            "architecture_fingerprint": subset[joint]["architecture_fingerprint"],
            "mean_3_seed": mean3, "mean_30_seed_reference": mean30,
            "signed_drift": mean30 - mean3, "absolute_drift": abs(mean30 - mean3),
        })
    values3 = np.asarray([row["mean_3_seed"] for row in records], dtype=np.float64)
    values30 = np.asarray([row["mean_30_seed_reference"] for row in records], dtype=np.float64)

    def finite(value: Any) -> float:
        result = float(value)
        return result if math.isfinite(result) else 0.0

    output = {
        "status": "DESCRIPTIVE_SENSITIVITY_ONLY", "inventory": inventory,
        "records": records, "configuration_count": len(records),
        "mean_absolute_mean_drift": float(np.mean(np.abs(values30 - values3))),
        "median_absolute_mean_drift": float(np.median(np.abs(values30 - values3))),
        "maximum_absolute_mean_drift": float(np.max(np.abs(values30 - values3))),
        "spearman_3_seed_vs_30_seed": finite(spearmanr(values3, values30).statistic),
        "kendall_3_seed_vs_30_seed": finite(kendalltau(values3, values30).statistic),
        "reference_name": "30-seed reference",
        "subset_limitation": (
            "Selected high-performing stabilization candidates; not a random representative "
            "sample of the 336 configurations."
        ),
        "used_for_training": False, "used_for_parameter_or_threshold_selection": False,
        "used_to_modify_delta_loss_or_gate": False,
        "nearest_neighbor_stability": {
            "status": "SPEC_UNDERDEFINED",
            "reason": "the prompt freezes no reference set, distance aggregation, or statistic",
        },
        "classification_before": str(frozen_decision["classification"]),
        "classification_after": str(frozen_decision["classification"]),
        "classification_changed": False,
    }
    _require(before == stable_hash(frozen_decision), "sensitivity mutated the frozen decision")
    return output


def _format_cell(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).replace("|", "\\|")


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    materialized = list(rows) or [{key: "UNAVAILABLE" for key, _label in columns}]
    header = "| " + " | ".join(label for _key, label in columns) + " |"
    separator = "| " + " | ".join("---" for _key, _label in columns) + " |"
    body = [
        "| " + " | ".join(_format_cell(row.get(key)) for key, _label in columns) + " |"
        for row in materialized
    ]
    return "\n".join([header, separator, *body])


def render_final_report(context: Mapping[str, Any]) -> str:
    decision = dict(context.get("decision") or {"classification": "RUN_FAILED"})
    classification = str(decision.get("classification", "RUN_FAILED"))
    identity = dict(context.get("input_integrity") or {})
    inventory = dict(context.get("data_inventory") or {})
    split = dict(context.get("split_manifest") or {})
    pairs = dict(context.get("pair_weight_summary") or {})
    training = dict(context.get("metric_training_audit") or {})
    geometry = dict(context.get("geometry") or {})
    locality = dict(context.get("knn_locality") or {})
    prediction_rows = list(context.get("prediction_metric_rows") or [])
    best = dict(context.get("best_of_four") or {})
    ablation = dict(context.get("branch_ablation") or {})
    shuffles = dict(context.get("shuffle_controls") or {})
    sensitivity = dict(context.get("sensitivity") or {})
    environment = dict(context.get("environment") or {})

    prediction_main = [
        row for row in prediction_rows
        if row.get("scope") == "pooled_oof" and row.get("model") in {"B0", "B1", "B2", "B3", "B4", "M1", "M2", "M3", "M4"}
    ]
    best_rows = [
        {"model": model, **summary}
        for model, summary in (best.get("models") or {}).items()
        if model in {"B0", "B1", "B2", "B3", "B4", "M1", "M2", "M3", "M4"}
    ]
    geometry_rows = list(geometry.get("folds") or [])
    pair_rows = [
        {"fold": fold.get("fold"), **(fold.get("summary") or {})}
        for fold in pairs.get("folds", [])
    ]
    training_rows = list(training.get("runs") or [])
    branch_rows = list(ablation.get("models") or [])
    shuffle_rows = list(shuffles.get("models") or [])
    sensitivity_rows = list(sensitivity.get("records") or [])[:14]
    lines = [
        "# Joint configuration uncertainty-aware Soft Performance Metric Gate v2", "",
        "## 1. Executive conclusion", "",
        f"The frozen classification is `{classification}`. The gate evaluates whether an 8D, "
        "L2-normalized joint architecture-and-hyperparameter embedding forms useful validation-performance "
        "distance geometry. It does not validate GP uncertainty or any acquisition function.", "",
        markdown_table([{
            "classification": classification,
            "geometry": (decision.get("geometry_gate") or {}).get("passed"),
            "M1": (decision.get("M1_gate") or {}).get("passed"),
            "M2": (decision.get("M2_gate") or {}).get("passed"),
            "joint": (decision.get("joint_contribution_gate") or {}).get("passed"),
            "gnn": False, "test": False,
        }], (("classification", "classification"), ("geometry", "geometry pass"),
             ("M1", "kNN pass"), ("M2", "isotropic GP pass"),
             ("joint", "joint contribution"), ("gnn", "new GNN"), ("test", "test accessed"))), "",
        "The nearest-five and isotropic-GP claims are kept separate: M1 measures direct local propagation, "
        "while M2 asks whether one isotropic Matérn lengthscale can exploit the learned Euclidean space. "
        "M4 and the direct regressor are diagnostics only and cannot substitute for either primary gate.", "",
        "## 2. Input identity", "",
        "Both historical experiment identities are checked before any metric encoder, regressor, or GP starts. "
        "The historical `JOINT_METRIC_UNDERPOWERED` and `PAIR_POWER_PRECISION_LIMITED` decisions remain unchanged.", "",
        markdown_table([identity], (("status", "status"), ("joint_metric_v1_classification", "hard v1"),
                                         ("pair_power_classification", "pair-power"),
                                         ("joint_metric_v1_archive_sha256", "v1 archive SHA"),
                                         ("pair_power_archive_sha256", "audit archive SHA"))), "",
        "## 3. Data and split", "",
        "The target is the arithmetic mean of raw validation accuracy at training seeds 0, 1, and 2. "
        "There is no HP residualization and no stabilization seed enters primary training. The exact frozen "
        "five-fold manifest is loaded rather than regenerated; architecture, pool, and joint identities do not cross folds.", "",
        markdown_table([{
            "records": inventory.get(
                "successful_required_seed_row_count", inventory.get("raw_row_count"),
            ),
            "joint": inventory.get("unique_joint_configurations"),
            "arch": inventory.get("unique_architectures"), "pools": inventory.get("unique_pools"),
            "seeds": inventory.get("metric_training_seeds"), "split": split.get("split_sha256"),
        }], (("records", "seed records"), ("joint", "joint configs"), ("arch", "architectures"),
             ("pools", "pools"), ("seeds", "training seeds"), ("split", "logical split SHA"))), "",
        "## 4. Soft metric formulation", "",
        "For every selected unordered outer-train pair, the matched-seed mean gap determines "
        "`exp(-0.5*(gap/0.012)^2)`. The paired standard error only attenuates supervision through "
        "`0.20 + 0.80*delta^2/(delta^2+SE^2)`. All weights therefore remain in [0.20,1.00]; "
        "no hard equivalence label, triplet, reconstruction, ranking, or auxiliary regression loss enters the encoder.", "",
        markdown_table(pair_rows, (("fold", "fold"), ("candidate_pair_count", "candidate pairs"),
                                   ("selected_pair_count", "used pairs"), ("selection_mode", "selection"),
                                   ("weight_minimum", "min weight"), ("weight_maximum", "max weight"))), "",
        "## 5. Training audit", "",
        "Each fold and metric seed uses architecture-grouped inner 80/20 early stopping. Inner-validation "
        "pairs are disjoint from inner-train pairs. After choosing best epoch, the encoder is freshly initialized "
        "with the same seed and refit on complete outer-train data for exactly that epoch count.", "",
        markdown_table(training_rows, (("fold", "fold"), ("seed", "metric seed"),
                                       ("best_epoch", "best epoch"),
                                       ("final_refit_used_complete_outer_train", "full refit"),
                                       ("outer_heldout_labels_used_for_pairs_or_training", "heldout used"))), "",
        "## 6. Geometry results", "",
        "All geometry cells are held-out queries against outer-train references. Distance-gap Spearman is "
        "reported per metric seed and fold. Cross-factor locality compares different-architecture, HP-far "
        "cells in deterministic bottom/top performance-gap quintiles; smaller ratios indicate better geometry.", "",
        markdown_table(geometry_rows, (("fold", "fold"), ("seed_distance_gap_spearman", "seed Spearman"),
                                       ("cross_factor_locality_ratio", "cross-factor ratio"),
                                       ("joint_metric_mean_gap_at_5", "metric Gap@5"),
                                       ("explicit_joint_mean_gap_at_5", "explicit Gap@5"))), "",
        "## 7. kNN locality results", "",
        "M1 uses raw Euclidean distance in the normalized embedding with k=5 and inverse-distance weights. "
        "The explicit-joint comparator uses the same k and weighting after an outer-train-fitted feature standardizer.", "",
        markdown_table([{
            "metric_gap": (locality.get("joint_metric") or {}).get("mean_local_performance_gap_at_5"),
            "explicit_gap": (locality.get("explicit_joint_feature") or {}).get("mean_local_performance_gap_at_5"),
            "reduction": locality.get("relative_reduction"),
            "mae_pass": (decision.get("M1_gate") or {}).get("passed"),
        }], (("metric_gap", "metric Gap@5"), ("explicit_gap", "explicit Gap@5"),
             ("reduction", "relative reduction"), ("mae_pass", "M1 composite pass"))), "",
        "## 8. GP results", "",
        "B1, B2, and B3 are ARD Matérn-5/2 baselines. M2 consumes the untouched normalized 8D embedding, "
        "uses one isotropic lengthscale, and fits its noise floor only on outer-train. M3 is local diagnostic; "
        "M4 standardizes dimensions and uses ARD, so an M4-only gain is feature evidence rather than metric evidence.", "",
        markdown_table([row for row in prediction_main if row.get("model") in {"B1", "B2", "B3", "M2", "M3", "M4"}],
                       (("model", "model"), ("mae", "MAE"), ("top20_precision", "Top20 precision"),
                        ("spearman", "Spearman"), ("ndcg_at_20_percent", "NDCG@20%"))), "",
        "## 9. Prediction metrics", "",
        "Every value below is strict pooled OOF prediction against the same raw validation target and frozen folds. "
        "No metric seed is selected: seed predictions are always arithmetic averaged.", "",
        markdown_table(prediction_main, (("model", "model"), ("mae", "MAE"), ("rmse", "RMSE"),
                                         ("pearson", "Pearson"), ("spearman", "Spearman"),
                                         ("kendall", "Kendall"), ("top20_precision", "Top20 precision"),
                                         ("top_region_kendall", "top-region Kendall"),
                                         ("ndcg_at_20_percent", "NDCG@20%"))), "",
        "## 10. best-of-4 replay", "",
        "One fold-specific batch manifest is constructed once and shared byte-for-byte across all models. "
        "When combinations exceed 5,000, exactly 1,000 seed-0 batches are used.", "",
        markdown_table(best_rows, (("model", "model"), ("mean_simple_regret", "mean regret"),
                                   ("median_simple_regret", "median regret"),
                                   ("probability_selecting_true_best", "P(select true best)"),
                                   ("selected_raw_validation_mean", "selected raw mean"))), "",
        "## 11. architecture/HP ablations", "",
        "Architecture-only and HP-only results use the same trained full encoder with the omitted representation "
        "zeroed before interaction/fusion; controls are not retrained and cannot alter the main protocol.", "",
        markdown_table(branch_rows + [
            {
                "model": name,
                "mae": f"better_count={value.get('better_metric_count')}",
                "top20_precision": f"required={value.get('required_better_metric_count')}",
                "mean_best_of_4_regret": f"pass={value.get('passed')}",
                "passed": value.get("passed"),
            }
            for name, value in (
                ((decision.get("joint_contribution_gate") or {}).get("branch_comparisons") or {}).items()
            )
        ], (("model", "model/control"), ("mae", "MAE / comparison"),
                                     ("top20_precision", "Top20 precision"),
                                     ("mean_best_of_4_regret", "mean regret"),
                                     ("passed", "contribution pass"))), "",
        "## 12. shuffle controls", "",
        "With fixed seed 0, only held-out architecture or HP representations are permuted within their own fold; "
        "the other branch and unshuffled outer-train references remain fixed. This diagnoses branch shortcuts without retraining.", "",
        markdown_table(shuffle_rows + [
            {
                "model": f"{name}_degradation",
                "mae": value.get("mae_relative_degradation"),
                "top20_precision": value.get("top20_precision_absolute_decrease"),
                "mean_best_of_4_regret": value.get("regret_relative_degradation"),
                "passed": value.get("passed"),
            }
            for name, value in (
                ((decision.get("joint_contribution_gate") or {}).get("shuffle_degradation") or {}).items()
            )
        ], (("model", "model/control"), ("mae", "MAE / relative degradation"),
                                      ("top20_precision", "Top20 precision"),
                                      ("mean_best_of_4_regret", "regret / relative degradation"),
                                      ("passed", "control pass"))), "",
        "## 13. 30-seed sensitivity", "",
        "This selected high-performance subset is not a random sample. It is read only after classification freezes, "
        "cannot retrain or tune anything, and is called a 30-seed reference rather than truth. Nearest-neighbor "
        "stability remains SPEC_UNDERDEFINED because no statistic/reference set was frozen.", "",
        markdown_table(sensitivity_rows, (("joint_fingerprint", "joint"), ("mean_3_seed", "mean3"),
                                          ("mean_30_seed_reference", "mean30 reference"),
                                          ("absolute_drift", "absolute drift"))), "",
        "## 14. Frozen classification", "",
        "The result below comes only from frozen OOF geometry, M1, M2, and joint contribution gates. "
        "Sensitivity cannot modify it, M4 cannot replace M2, and metrics from different baselines are never spliced.", "",
        markdown_table([decision], (("classification", "classification"),
                                    ("strongest_non_metric_gp", "strongest non-metric GP"),
                                    ("M4_may_replace_M2", "M4 replaces M2"),
                                    ("cross_model_metric_splicing_used", "metric splicing"))), "",
        "## 15. Interpretation", "",
        "A geometry pass means held-out configurations with smaller validation gaps tend to lie closer and that "
        "cross-factor low-gap pairs are more local. M1 establishes direct neighbor propagation; M2 establishes "
        "whether that geometry supports an isotropic GP posterior mean. GP_READY would only justify one prospective "
        "discrete-candidate-pool validation, never immediate Phase4 or BO changes.", "",
        markdown_table([{"geometry_claimed": (decision.get("geometry_gate") or {}).get("passed"),
                         "knn_claimed": (decision.get("M1_gate") or {}).get("passed"),
                         "gp_claimed": (decision.get("M2_gate") or {}).get("passed"),
                         "bo_modified": False}],
                       (("geometry_claimed", "geometry"), ("knn_claimed", "kNN"),
                        ("gp_claimed", "isotropic GP"), ("bo_modified", "formal BO modified"))), "",
        "## 16. Limitations", "",
        "Only three validation seeds supervise the main experiment, so pair means and paired SE estimates remain noisy. "
        "The soft weighting attenuates uncertainty but does not eliminate label drift. GP uncertainty and noise calibration "
        "are not gated, the stabilization subset is selected, and the direct regressor/M4 only diagnose supervised features. "
        "No candidate GNN, decoder, acquisition, test mask, test labels, prediction, or accuracy is used.", "",
        markdown_table([{"primary_seed_count": 3, "uncertainty_validated": False,
                         "sensitivity_representative": False, "new_gnn": False, "test": False}],
                       (("primary_seed_count", "primary seeds"),
                        ("uncertainty_validated", "GP uncertainty validated"),
                        ("sensitivity_representative", "30-seed subset random"),
                        ("new_gnn", "new GNN"), ("test", "test accessed"))), "",
        "## 17. Engineering audit", "",
        "The one-shot runner performs compilation, focused CPU tests, synthetic classification smoke, formal offline "
        "evaluation, source snapshot, manifest, archive, and SHA-256 publication. The archive cannot embed its own "
        "final digest without self-reference; the sibling sidecar is authoritative.", "",
        markdown_table([{
            "branch": environment.get("branch"), "head": environment.get("head"),
            "dirty": environment.get("dirty_tree"), "cpu": environment.get("cpu_only"),
            "result": context.get("result_path"), "archive": context.get("archive_path"),
            "snapshot": context.get("snapshot_manifest_root"),
        }], (("branch", "branch"), ("head", "HEAD"), ("dirty", "dirty tree"),
             ("cpu", "CPU only"), ("result", "result directory"),
             ("archive", "archive"), ("snapshot", "snapshot root"))), "",
    ]
    return "\n".join(lines)


def validate_final_report(report: str, required_sections: Sequence[str]) -> None:
    for index, title in enumerate(required_sections, start=1):
        heading = f"## {index}. {title}"
        _require(heading in report, f"final report is missing section: {heading}")
        start = report.index(heading)
        next_start = report.find(f"## {index + 1}. ", start + 1) if index < len(required_sections) else len(report)
        table_lines = [line for line in report[start:next_start].splitlines() if line.startswith("|")]
        _require(len(table_lines) >= 3, f"final report section {index} lacks a nonempty table")
    _require(len("".join(report.split())) >= 3000, "final report is unexpectedly empty")


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
    for name in ("PyYAML", "numpy", "scipy", "scikit-learn", "torch", "gpytorch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = {
        "format_version": FORMAT_VERSION, "timestamp_utc": utc_now(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": packages, "branch": branch, "head": head,
        "dirty_tree": bool(status.splitlines()[1:]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_only": True, "gpu_used": False, "network_access_used": False,
        "candidate_gnn_training_started": False,
        "offline_metric_encoder_training_permitted": True,
        "offline_direct_regressor_training_permitted": True,
        "offline_gp_fitting_permitted": True,
        "acquisition_used": False, "test_data_accessed": False,
    }
    dump_json(output / "environment.json", environment)
    return environment


def ensure_output_policy(output: Path, *, preflight_ready: bool = False) -> None:
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise ProtocolError(f"formal result path is not a safe directory: {output}")
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


def create_source_snapshot(
    output: Path, mandatory_files: Sequence[str], *, modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = pair_audit.snapshot_runtime_sources(
        output, mandatory_files, modules=modules or sys.modules,
    )
    payload = {**payload, "format_version": FORMAT_VERSION}
    dump_json(output / "source_snapshot/sha256_manifest.json", payload)
    imported_path = output / "source_snapshot/imported_local_modules.json"
    imported = _read_json(imported_path)
    dump_json(imported_path, {**imported, "format_version": FORMAT_VERSION})
    pair_audit.verify_source_snapshot(output)
    return payload


snapshot_runtime_sources = create_source_snapshot
_source_path_for_module = pair_audit._source_path_for_module
verify_source_snapshot = pair_audit.verify_source_snapshot


def write_manifest(output: Path) -> None:
    files: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if "__pycache__" in path.parts or path.suffix in {
            ".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".onnx", ".joblib",
        }:
            continue
        files.append({
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256_file(path),
        })
    dump_json(output / "manifest.json", {
        "format_version": FORMAT_VERSION, "generated_at_utc": utc_now(),
        "file_count": len(files), "files": files,
        "manifest_root_sha256": stable_hash(files),
    })


def create_archive(output: Path, archive: Path, sidecar: Path | None = None) -> str:
    return pair_audit.create_archive(output, archive, sidecar)


def _aggregate_source_sha(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/run_joint_configuration_soft_metric_gate.sh",
        REPO_ROOT / "tests/test_joint_configuration_soft_metric_gate.py",
        REPO_ROOT / "analyse/joint_configuration_metric_gate.py",
        REPO_ROOT / "analyse/performance_metric_quick_gate.py",
        REPO_ROOT / "analyse/joint_pair_power_projection_audit.py",
    ]


def initial_state(config_sha256: str, source_sha256: str) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION, "status": "running",
        "started_at_utc": utc_now(), "updated_at_utc": utc_now(),
        "config_sha256": config_sha256, "source_sha256": source_sha256,
        "decision": None, "formal_data_read": False,
        "candidate_gnn_training_started": False,
        "metric_model_training_started": False,
        "direct_regressor_training_started": False,
        "gp_fitting_started": False, "acquisition_used": False,
        "test_data_accessed": False,
    }


def update_state(output: Path, state: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    updated = {**state, **changes, "updated_at_utc": utc_now()}
    dump_json(output / "state.json", updated)
    return updated


def _placeholder_json(reason: str) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION, "status": "UNAVAILABLE",
        "reason": reason, "rows": [{"status": "UNAVAILABLE", "reason": reason}],
    }


def _write_if_missing(path: Path, value: Any) -> None:
    if not path.exists():
        dump_json(path, value)


def _write_required_placeholders(output: Path, reason: str) -> None:
    json_names = [name for name in REQUIRED_OUTPUTS if name.endswith(".json")]
    for name in json_names:
        _write_if_missing(output / name, _placeholder_json(reason))
    csv_fields = {
        "metrics_per_fold.csv": ("status", "reason"),
        "oof_predictions.csv": ("status", "reason"),
    }
    for name, fields in csv_fields.items():
        path = output / name
        if not path.exists():
            dump_csv(path, [{"status": "UNAVAILABLE", "reason": reason}], fields)
    for name in ("git_status.txt", "git_diff.patch", "final_report.md"):
        path = output / name
        if not path.exists():
            dump_text(path, "" if name == "git_diff.patch" else f"UNAVAILABLE: {reason}\n")
    for name in (
        "logs/py_compile.log", "logs/pytest_focused.log",
        "logs/synthetic_smoke.log", "logs/run.log",
    ):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            dump_text(path, "")


def _fallback_environment(output: Path, reason: str) -> dict[str, Any]:
    environment = {
        "format_version": FORMAT_VERSION, "status": "UNAVAILABLE", "reason": reason,
        "branch": "UNAVAILABLE", "head": "UNAVAILABLE", "dirty_tree": "UNAVAILABLE",
        "cpu_only": True, "gpu_used": False, "candidate_gnn_training_started": False,
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
    snapshot = create_source_snapshot(
        output, list(config["source_snapshot"]["mandatory_files"]),
    )
    context["snapshot_manifest_root"] = snapshot["manifest_root_sha256"]
    context["result_path"] = str(output)
    context["archive_path"] = str(_repository_path(str(config["output"]["archive"])))
    report = render_final_report(context)
    validate_final_report(report, list(config["reporting"]["required_sections"]))
    dump_text(output / "final_report.md", report)
    return snapshot


def write_failure_bundle(
    output: Path, message: str, config_sha256: str, source_sha256: str,
    *, config_path: Path = DEFAULT_CONFIG,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    _write_required_placeholders(output, message)
    dump_json(output / "error.json", {
        "classification": "RUN_FAILED", "message": message,
        "timestamp_utc": utc_now(),
        "traceback": traceback.format_exc() if sys.exc_info()[0] is not None else None,
    })
    try:
        config = load_config(config_path)
    except Exception:
        config = {
            "source_snapshot": {"mandatory_files": [
                "configs/joint_configuration_soft_metric_gate_v2.yaml",
                "analyse/joint_configuration_soft_metric_gate.py",
                "scripts/run_joint_configuration_soft_metric_gate.sh",
                "tests/test_joint_configuration_soft_metric_gate.py",
                "analyse/joint_configuration_metric_gate.py",
                "analyse/performance_metric_quick_gate.py",
            ]},
            "reporting": {"required_sections": [
                "Executive conclusion", "Input identity", "Data and split",
                "Soft metric formulation", "Training audit", "Geometry results",
                "kNN locality results", "GP results", "Prediction metrics",
                "best-of-4 replay", "architecture/HP ablations", "shuffle controls",
                "30-seed sensitivity", "Frozen classification", "Interpretation",
                "Limitations", "Engineering audit",
            ]},
            "output": {"archive": "results/posthoc/joint-configuration-soft-metric-gate-global4-v2.tar.gz"},
        }
    state_path = output / "state.json"
    try:
        state = _read_json(state_path) if state_path.is_file() else initial_state(config_sha256, source_sha256)
    except Exception:
        state = initial_state(config_sha256, source_sha256)
    try:
        environment = write_git_and_environment(output)
    except Exception as error:
        environment = _fallback_environment(output, f"{message}; environment capture failed: {error}")
    decision = {
        "classification": "RUN_FAILED", "reason": message,
        "candidate_gnn_training_started": False, "acquisition_used": False,
        "test_data_accessed": False, "historical_classifications_modified": False,
    }
    dump_json(output / "decision.json", decision)
    if not (output / "frozen_protocol.json").exists():
        dump_json(output / "frozen_protocol.json", {
            "format_version": FORMAT_VERSION, "status": "failure_before_protocol_freeze",
            "config_sha256": config_sha256, "source_sha256": source_sha256,
        })
    context = {
        "decision": decision, "environment": environment,
        "input_integrity": _read_json(output / "input_integrity.json"),
        "data_inventory": _read_json(output / "data_inventory.json"),
        "split_manifest": _read_json(output / "split_manifest.json"),
        "pair_weight_summary": _read_json(output / "pair_weight_summary.json"),
        "metric_training_audit": _read_json(output / "metric_training_audit.json"),
        "geometry": _read_json(output / "geometry_metrics.json"),
        "knn_locality": _read_json(output / "knn_locality.json"),
        "branch_ablation": _read_json(output / "branch_ablation.json"),
        "shuffle_controls": _read_json(output / "shuffle_controls.json"),
        "sensitivity": _read_json(output / "sensitivity_30seed.json"),
    }
    try:
        _finalize_report_and_snapshot(output, config, context)
    except Exception as error:
        raise ProtocolError(f"failure bundle source snapshot/report failed: {error}") from error
    update_state(
        output, state, status="failed", decision="RUN_FAILED", error=message,
        candidate_gnn_training_started=False, acquisition_used=False,
        test_data_accessed=False,
    )
    (output / "COMPLETE").unlink(missing_ok=True)
    (output / "FAILED").touch()
    write_manifest(output)


def _formal_context_from_output(
    output: Path, decision: Mapping[str, Any], environment: Mapping[str, Any],
    *, prediction_metric_rows: Sequence[Mapping[str, Any]] | None = None,
    best_of_four: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "decision": dict(decision), "environment": dict(environment),
        "input_integrity": _read_json(output / "input_integrity.json"),
        "data_inventory": _read_json(output / "data_inventory.json"),
        "split_manifest": _read_json(output / "split_manifest.json"),
        "pair_weight_summary": _read_json(output / "pair_weight_summary.json"),
        "metric_training_audit": _read_json(output / "metric_training_audit.json"),
        "geometry": _read_json(output / "geometry_metrics.json"),
        "knn_locality": _read_json(output / "knn_locality.json"),
        "prediction_metric_rows": list(prediction_metric_rows or []),
        "best_of_four": dict(best_of_four or {}),
        "branch_ablation": _read_json(output / "branch_ablation.json"),
        "shuffle_controls": _read_json(output / "shuffle_controls.json"),
        "sensitivity": _read_json(output / "sensitivity_30seed.json"),
    }


def finalize_input_identity_mismatch(
    output: Path, config: Mapping[str, Any], state: Mapping[str, Any],
    reason: str, environment: Mapping[str, Any],
) -> int:
    _write_required_placeholders(output, reason)
    integrity = {
        "format_version": FORMAT_VERSION, "status": "INVALID",
        "classification": "INPUT_IDENTITY_MISMATCH", "reason": reason,
        "model_training_started": False, "candidate_gnn_training_started": False,
        "test_data_accessed": False,
    }
    dump_json(output / "input_integrity.json", integrity)
    decision = {
        "classification": "INPUT_IDENTITY_MISMATCH", "reason": reason,
        "classification_source": "pre-training frozen input identity audit",
        "historical_classifications_modified": False,
        "model_training_started": False, "candidate_gnn_training_started": False,
        "test_data_accessed": False,
    }
    dump_json(output / "decision.json", decision)
    context = _formal_context_from_output(output, decision, environment)
    snapshot = _finalize_report_and_snapshot(output, config, context)
    update_state(
        output, state, status="complete", decision="INPUT_IDENTITY_MISMATCH",
        formal_data_read=True, error=reason,
        source_snapshot_manifest_root=snapshot["manifest_root_sha256"],
        candidate_gnn_training_started=False, metric_model_training_started=False,
        direct_regressor_training_started=False, gp_fitting_started=False,
        acquisition_used=False, test_data_accessed=False,
    )
    write_manifest(output)
    return 0


def execute_formal(
    config_path: Path, output: Path, config_sha256: str, source_sha256: str,
    *, preflight_ready: bool = False,
) -> int:
    started = time.monotonic()
    ensure_output_policy(output, preflight_ready=preflight_ready)
    config = load_config(config_path)
    actual_config_sha = sha256_file(config_path)
    actual_source_sha = _aggregate_source_sha(_source_paths())
    _require(actual_config_sha == str(config_sha256), "runner/config SHA-256 mismatch")
    _require(actual_source_sha == str(source_sha256), "runner/source SHA-256 mismatch")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    state = initial_state(config_sha256, source_sha256)
    dump_json(output / "state.json", state)
    environment = write_git_and_environment(output)
    dump_json(output / "frozen_protocol.json", {
        "format_version": FORMAT_VERSION, "config_sha256": config_sha256,
        "source_sha256": source_sha256, "config": config,
        "scientific_definitions_frozen_before_formal_data_read": True,
        "hard_triplet_or_pair_power_gate_used": False,
        "candidate_gnn_training_allowed": False, "test_access_allowed": False,
    })
    try:
        authority = validate_input_identity(config)
        configurations, inventory, provenance, split_manifest = load_main_data_and_split(config)
    except InputIdentityMismatch as error:
        return finalize_input_identity_mismatch(
            output, config, state, str(error), environment,
        )
    except Exception as error:
        return finalize_input_identity_mismatch(
            output, config, state,
            f"frozen input identity could not be read or validated: {error}", environment,
        )
    state = update_state(output, state, formal_data_read=True)
    joint_identity = authority["joint_metric_v1"]
    pair_identity = authority["pair_power_audit_v1"]
    integrity = {
        "format_version": FORMAT_VERSION, "status": "VALID",
        "joint_metric_v1_classification": joint_identity["classification"],
        "pair_power_classification": pair_identity["classification"],
        "joint_metric_v1_archive_sha256": joint_identity["archive_sha256"],
        "pair_power_archive_sha256": pair_identity["archive_sha256"],
        "joint_metric_v1_config_sha256": joint_identity["config_sha256"],
        "joint_metric_v1_source_sha256": joint_identity["source_sha256"],
        "split_sha256": str(config["split"]["expected_sha256"]),
        "unique_joint_configurations": int(inventory["unique_joint_configurations"]),
        "unique_architectures": int(inventory["unique_architectures"]),
        "unique_pools": int(inventory["unique_pools"]),
        "training_seeds": [0, 1, 2], "target": "raw_validation_accuracy_seed_mean",
        "historical_classifications_modified": False,
        "candidate_gnn_training_started": False, "test_data_accessed": False,
        "authority_checks": authority["checks"], "data_provenance": provenance,
    }
    dump_json(output / "input_integrity.json", integrity)
    dump_json(output / "data_inventory.json", inventory)
    dump_json(output / "split_manifest.json", split_manifest)

    soft = config["soft_metric"]
    pair_inventories: dict[int, dict[str, Any]] = {}
    pair_fold_rows: list[dict[str, Any]] = []
    for fold in split_manifest["folds"]:
        fold_id = int(fold["fold"])
        train_ids = set(map(str, fold["train_joint_fingerprints"]))
        train_rows = sorted(
            (dict(row) for row in configurations if str(row["joint_fingerprint"]) in train_ids),
            key=lambda row: str(row["joint_fingerprint"]),
        )
        inventory_for_fold = construct_soft_pairs(
            train_rows, allowed_joint_fingerprints=train_ids,
            delta=float(soft["delta"]),
            minimum_weight=float(soft["minimum_pair_weight"]),
            reliability_fraction=float(soft["reliability_weight_fraction"]),
            pair_cap=int(soft["pair_cap"]), sampling_seed=int(soft["pair_sampling_seed"]),
        )
        pair_inventories[fold_id] = inventory_for_fold
        summary = summarize_soft_pairs(inventory_for_fold)
        pair_fold_rows.append({
            "fold": fold_id,
            "summary": {
                "candidate_pair_count": summary["candidate_pair_count"],
                "selected_pair_count": summary["selected_pair_count"],
                "selection_mode": summary["selection_mode"],
                "weight_minimum": summary["summaries"]["weight"]["minimum"],
                "weight_maximum": summary["summaries"]["weight"]["maximum"],
            },
            "details": summary,
        })
    pair_summary = {
        "format_version": FORMAT_VERSION, "folds": pair_fold_rows,
        "delta": 0.012, "minimum_weight": 0.20, "maximum_weight": 1.0,
        "all_outer_train_pairs_eligible": True, "hard_labels_or_triplets_used": False,
        "outer_heldout_labels_used": False,
    }
    dump_json(output / "pair_weight_summary.json", pair_summary)
    state = update_state(
        output, state, metric_model_training_started=True,
        direct_regressor_training_started=True, gp_fitting_started=True,
        candidate_gnn_training_started=False, test_data_accessed=False,
    )
    fold_results: list[dict[str, Any]] = []
    for fold in split_manifest["folds"]:
        fold_id = int(fold["fold"])
        fold_results.append(evaluate_outer_fold_soft(
            fold, configurations, config, pair_inventories[fold_id],
        ))
    oof_rows, pooled_predictions, metric_output, replay = build_prediction_artifacts(
        fold_results, config,
    )
    geometry, locality = aggregate_geometry_soft(fold_results)
    decision = classify_soft_joint_metric_gate(
        geometry, pooled_predictions, replay["results"], config["decision"],
    )
    metric_runs = [
        dict(audit) for result in fold_results for audit in result["metric_training_audits"]
    ]
    training_audit = {
        "format_version": FORMAT_VERSION, "runs": metric_runs,
        "direct_regressor_runs": [
            {**audit, "fold": result["fold"]}
            for result in fold_results for audit in result["direct_regressor_audits"]
        ],
        "metric_seeds": [0, 1, 2],
        "metric_seed_aggregation": "arithmetic_mean_all_seeds_0_1_2",
        "best_metric_seed_selected": False,
        "final_refit_from_fresh_initialization": True,
        "outer_heldout_labels_used_for_training": False,
        "stabilization_seeds_used_for_training": False,
        "hp_residualization_used": False,
        "fold_method_audits": [{
            "fold": int(result["fold"]),
            "baseline_gp_audits": result["gp_audits"],
            "local_gp_audits": result["local_gp_audits"],
            "representation_control_audits": result["control_audits"],
            "hp_standardizer": result["hp_standardizer"],
            "explicit_joint_feature_transform": result["explicit_joint_feature_transform"],
            "metric_seed_aggregation": result["metric_seed_aggregation"],
            "best_metric_seed_selected": result["best_metric_seed_selected"],
            "outer_heldout_labels_used_for_training": result[
                "outer_heldout_labels_used_for_training"
            ],
        } for result in fold_results],
        "candidate_gnn_training_started": False, "test_data_accessed": False,
    }
    dump_json(output / "metric_training_audit.json", training_audit)
    dump_json(output / "geometry_metrics.json", geometry)
    dump_json(output / "knn_locality.json", locality)
    dump_csv(output / "metrics_per_fold.csv", metric_output["rows"])
    dump_csv(output / "oof_predictions.csv", oof_rows)
    dump_json(output / "best_of_4_batches.json", replay["manifest"])
    dump_json(output / "best_of_4_results.json", replay["results"])
    branch_ablation, shuffle_controls = summarize_controls(
        pooled_predictions, replay["results"], decision,
    )
    dump_json(output / "branch_ablation.json", branch_ablation)
    dump_json(output / "shuffle_controls.json", shuffle_controls)
    dump_json(output / "decision.json", decision)

    # The selected 30-seed subset is deliberately invisible until the main decision is frozen.
    decision_hash_before_sensitivity = stable_hash(decision)
    sensitivity = sensitivity_30seed(config, configurations, decision)
    _require(stable_hash(decision) == decision_hash_before_sensitivity,
             "30-seed sensitivity changed the frozen decision")
    dump_json(output / "sensitivity_30seed.json", sensitivity)
    context = _formal_context_from_output(
        output, decision, environment,
        prediction_metric_rows=metric_output["rows"],
        best_of_four=replay["results"],
    )
    snapshot = _finalize_report_and_snapshot(output, config, context)
    elapsed = time.monotonic() - started
    update_state(
        output, state, status="complete", decision=decision["classification"],
        formal_elapsed_seconds=elapsed,
        source_snapshot_manifest_root=snapshot["manifest_root_sha256"],
        sensitivity_status=sensitivity["status"],
        metric_model_training_started=True, direct_regressor_training_started=True,
        gp_fitting_started=True, candidate_gnn_training_started=False,
        acquisition_used=False, test_data_accessed=False,
    )
    write_manifest(output)
    return 0


def dependency_inventory() -> dict[str, bool]:
    output: dict[str, bool] = {}
    for name in ("yaml", "numpy", "scipy", "sklearn", "torch", "gpytorch"):
        try:
            __import__(name)
            output[name] = True
        except ImportError:
            output[name] = False
    return output


def dry_run(config_path: Path, output: Path) -> int:
    config = load_config(config_path)
    payload = {
        "protocol_version": config["protocol_version"], "mode": "dry-run",
        "cpu_only": True, "formal_data_read": False,
        "formal_result_root_created": False, "result_root_created": False,
        "model_training_started": False, "metric_model_training_started": False,
        "direct_regressor_training_started": False, "gp_fitting_started": False,
        "candidate_gnn_training_started": False, "acquisition_used": False,
        "test_data_accessed": False, "configured_output": str(output),
        "configured_output_exists": output.exists(), "dependencies": dependency_inventory(),
        "formal_actions": [
            "validate both frozen historical identities before model work",
            "load exactly 336 validation-only joint configurations and frozen five-fold split",
            "train three uncertainty-weighted soft metric seeds per outer fold",
            "fit frozen baselines, kNN, isotropic/local/ARD GPs and OOF controls",
            "freeze classification before optional descriptive 30-seed sensitivity",
            "snapshot runtime sources, report, manifest and archive bundle",
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def synthetic_smoke(config_path: Path = DEFAULT_CONFIG) -> int:
    config = load_config(config_path)
    similarity_values = [soft_similarity(value, 0.012) for value in (0.0, 0.006, 0.012, 0.024)]
    weight_values = [uncertainty_weight(value) for value in (0.0, 0.003, 0.012, 1.0)]
    classifications = {
        "synthetic_rejected": classify_gate_states(
            geometry_passed=False, m1_passed=True, m2_passed=True,
            joint_contribution_passed=True, feature_diagnostic_passed=True,
        ) == "SOFT_JOINT_METRIC_REJECTED",
        "synthetic_geometry_only": classify_gate_states(
            geometry_passed=True, m1_passed=False, m2_passed=False,
            joint_contribution_passed=False, feature_diagnostic_passed=False,
        ) == "SOFT_JOINT_METRIC_GEOMETRY_ONLY",
        "synthetic_locality_only": classify_gate_states(
            geometry_passed=True, m1_passed=True, m2_passed=False,
            joint_contribution_passed=False, feature_diagnostic_passed=False,
        ) == "SOFT_JOINT_METRIC_LOCALITY_ONLY",
        "synthetic_feature_only": classify_gate_states(
            geometry_passed=True, m1_passed=False, m2_passed=False,
            joint_contribution_passed=False, feature_diagnostic_passed=True,
        ) == "SOFT_JOINT_METRIC_FEATURE_ONLY",
        "synthetic_not_joint": classify_gate_states(
            geometry_passed=True, m1_passed=True, m2_passed=True,
            joint_contribution_passed=False, feature_diagnostic_passed=False,
        ) == "SOFT_JOINT_METRIC_NOT_JOINT",
        "synthetic_gp_ready": classify_gate_states(
            geometry_passed=True, m1_passed=True, m2_passed=True,
            joint_contribution_passed=True, feature_diagnostic_passed=False,
        ) == "SOFT_JOINT_METRIC_GP_READY",
    }
    payload = {
        "passed": (
            all(classifications.values())
            and all(left >= right for left, right in zip(similarity_values, similarity_values[1:]))
            and all(left >= right for left, right in zip(weight_values, weight_values[1:]))
            and min(weight_values) >= 0.20 and max(weight_values) <= 1.0
            and config["metric_methods"]["M2"]["ard"] is False
            and config["metric_methods"]["M2"]["per_dimension_standardization"] is False
            and config["metric_methods"]["M4"]["may_form_gp_ready"] is False
        ),
        **classifications,
        "soft_similarity_monotone": all(
            left >= right for left, right in zip(similarity_values, similarity_values[1:])
        ),
        "uncertainty_weight_monotone": all(
            left >= right for left, right in zip(weight_values, weight_values[1:])
        ),
        "weight_bounds": min(weight_values) >= 0.20 and max(weight_values) <= 1.0,
        "M2_isotropic_unstandardized_gp": True, "M4_cannot_replace_M2": True,
        "model_training_started": False, "formal_data_read": False,
        "candidate_gnn_training_started": False, "test_data_accessed": False,
    }
    payload.update({
        f"{key}_case": value for key, value in classifications.items()
    })
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
        return synthetic_smoke(config_path)
    config_sha256 = args.config_sha256 or sha256_file(config_path)
    source_sha256 = args.source_sha256 or _aggregate_source_sha(_source_paths())
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
                print(f"formal soft metric gate failed: {error}", file=sys.stderr)
                print(f"failure bundle also failed: {bundle_error}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return 1
    raise ProtocolError("no execution mode selected")


if __name__ == "__main__":
    raise SystemExit(main())
