"""Validation-first Cora LayerNorm architecture-subgroup confirmation."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import cora_accuracy_stage_a as stage_a
from analyse import cora_component_screen as b0
from analyse.cora_offline_predictor import hp_features
from evaluation_errors import InfrastructureEvaluationError
from eval_utils import train_and_eval_arch


CROSSOVER_ROOT = REPO_ROOT / "results/posthoc/cora-layernorm-crossover-v1"
CROSSOVER_FREEZE_COMMIT = "222da3f0d5bec656a00c2244d4987406f087c52f"
ARTIFACT_ROOT = REPO_ROOT / "legacy_artifacts/pre_20260813/results"
CHECKPOINT = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
SIMILAR_COUNT = 3
CONTROL_COUNT = 3
REPLICATES = 3
VARIANTS = ("baseline", "layer_norm")
EXPECTED_TRAININGS = 36
RESPONDER_THRESHOLD = 0.004
CONTROL_STRUCTURE_FLOOR = 0.5


class SubgroupError(RuntimeError):
    """Raised when selection, provenance, or experiment invariants fail."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _frozen_file(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{CROSSOVER_FREEZE_COMMIT}:{path}"], cwd=REPO_ROOT,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def _verify_crossover_freeze() -> dict[str, Any]:
    path = CROSSOVER_ROOT / "freeze_manifest.json"
    freeze = stage_a.load_json(path)
    if freeze.get("archive_status") != "FROZEN":
        raise SubgroupError("crossover archive is not frozen")
    for relative, expected in freeze["result_hashes"].items():
        result_path = CROSSOVER_ROOT / relative
        if stage_a.sha256_file(result_path) != expected:
            raise SubgroupError(f"crossover result hash mismatch: {relative}")
        committed = _frozen_file(str(result_path.relative_to(REPO_ROOT)))
        if hashlib.sha256(committed).hexdigest() != expected:
            raise SubgroupError(f"crossover commit mismatch: {relative}")
    for relative, expected in freeze["source_hashes"].items():
        if hashlib.sha256(_frozen_file(relative)).hexdigest() != expected:
            raise SubgroupError(f"crossover source mismatch: {relative}")
    return freeze


def _candidate_by_role(manifest: Mapping[str, Any], role: str) -> dict[str, Any]:
    matches = [row for row in manifest["candidate_panel"] if row["panel_role"] == role]
    if len(matches) != 1:
        raise SubgroupError(f"expected one {role}, got {len(matches)}")
    return matches[0]


def _predecessor_pattern(record: Mapping[str, Any]) -> list[int]:
    counts = {node: 0 for node in range(1, len(record["operations"]) + 1)}
    for _source, target in record["edges"]:
        if int(target) in counts:
            counts[int(target)] += 1
    return [counts[node] for node in sorted(counts)]


def _structure_features(record: Mapping[str, Any]) -> dict[str, Any]:
    operations = list(record["operations"])
    predecessor_pattern = _predecessor_pattern(record)
    stress = b0._static_truncation_stress(record)
    return {
        "effective_depth": sum(operation != "Identity" for operation in operations),
        "has_identity": "Identity" in operations,
        "has_gcnii": "GCNII" in operations,
        "has_gcn_gin_sage_mix": all(
            operation in operations for operation in ("GCNConv", "GINConv", "SAGEConv")
        ),
        "predecessor_pattern": predecessor_pattern,
        "multi_predecessor_nodes": sum(value > 1 for value in predecessor_pattern),
        "mixed_width_events": int(stress["mismatch_events"]),
        "mixed_width_discarded_columns": int(stress["static_columns_discarded"]),
    }


def _structure_distances(
    record: Mapping[str, Any], median: Mapping[str, Any],
) -> dict[str, float]:
    operation = stage_a.op_distance(record, median)
    edge = stage_a.edge_distance(record, median)
    return {
        "operation_distance": operation,
        "edge_distance": edge,
        "combined_distance": (operation + edge) / 2.0,
        "path_distance": stage_a.path_distance(record, median),
    }


def _similar_eligible(record: Mapping[str, Any]) -> bool:
    features = _structure_features(record)
    return (
        features["effective_depth"] == 5
        and not features["has_identity"]
        and not features["has_gcnii"]
        and features["has_gcn_gin_sage_mix"]
        and features["multi_predecessor_nodes"] >= 3
        and features["mixed_width_events"] > 0
    )


def _candidate_payload(
    record: Mapping[str, Any], *, group: str, candidate_id: str,
    median: Mapping[str, Any], hp_center: np.ndarray, hp_scale: np.ndarray,
    paired_candidate_id: str | None, paired_hp_distance: float | None,
) -> dict[str, Any]:
    fields = (
        "candidate_fingerprint", "architecture_fingerprint", "operations", "edges",
        "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
        "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer", "val_acc",
        "_method", "_run_seed", "_run_name", "full_evaluation_index",
    )
    payload = {field: copy.deepcopy(record.get(field)) for field in fields}
    hp_vector = hp_features(record)[0]
    median_hp = hp_features(median)[0]
    payload.update(
        {
            "group": group,
            "candidate_id": candidate_id,
            "paired_candidate_id": paired_candidate_id,
            "selection_validation": float(record["val_acc"]),
            "structure_features": _structure_features(record),
            "structure_distance_to_median": _structure_distances(record, median),
            "hp_distance_to_median": float(
                np.linalg.norm((hp_vector - median_hp) / hp_scale)
            ),
            "paired_hp_distance": paired_hp_distance,
        }
    )
    if any("test" in key.lower() for key in payload):
        raise SubgroupError("candidate payload contains a test field")
    return payload


def _best_validation_by_architecture(
    records: Sequence[Mapping[str, Any]], excluded: set[str], validation_floor: float,
) -> list[dict[str, Any]]:
    best: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if (
            not bool(record.get("valid", True))
            or record["architecture_fingerprint"] in excluded
            or float(record["val_acc"]) < validation_floor
        ):
            continue
        fingerprint = record["architecture_fingerprint"]
        previous = best.get(fingerprint)
        if previous is None or (
            -float(record["val_acc"]), record["candidate_fingerprint"]
        ) < (-float(previous["val_acc"]), previous["candidate_fingerprint"]):
            best[fingerprint] = record
    return [copy.deepcopy(best[key]) for key in sorted(best)]


def select_candidates(
    deduplicated: Sequence[Mapping[str, Any]], panel: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if any(any("test" in key.lower() for key in row) for row in deduplicated):
        raise SubgroupError("selection population contains test fields")
    median = next(row for row in panel if row["panel_role"] == "median_1")
    excluded = {row["architecture_fingerprint"] for row in panel}
    valid_values = [float(row["val_acc"]) for row in deduplicated if row.get("valid", True)]
    validation_floor = float(np.median(valid_values))
    representatives = _best_validation_by_architecture(
        deduplicated, excluded, validation_floor
    )
    hp_matrix = np.asarray([hp_features(row)[0] for row in representatives], dtype=float)
    hp_center = hp_matrix.mean(axis=0)
    hp_scale = hp_matrix.std(axis=0)
    if np.any(hp_scale <= 0):
        raise SubgroupError("HP standardization scale is not positive")
    similar_pool = [row for row in representatives if _similar_eligible(row)]
    similar_pool.sort(
        key=lambda row: (
            _structure_distances(row, median)["combined_distance"],
            -float(row["val_acc"]), row["candidate_fingerprint"],
        )
    )
    similar = similar_pool[:SIMILAR_COUNT]
    if len(similar) != SIMILAR_COUNT:
        raise SubgroupError("not enough similar architectures")
    similar_architectures = {row["architecture_fingerprint"] for row in similar}
    controls = [
        row for row in representatives
        if row["architecture_fingerprint"] not in similar_architectures
        and _structure_distances(row, median)["combined_distance"]
        >= CONTROL_STRUCTURE_FLOOR
    ]
    controls.sort(key=lambda row: row["candidate_fingerprint"])
    cost = np.asarray(
        [
            [
                float(
                    np.linalg.norm(
                        (hp_features(similar_row)[0] - hp_features(control_row)[0])
                        / hp_scale
                    )
                )
                for control_row in controls
            ]
            for similar_row in similar
        ],
        dtype=float,
    )
    tie_break = np.arange(cost.shape[1], dtype=float) * 1e-12
    row_indices, column_indices = linear_sum_assignment(cost + tie_break)
    if list(row_indices) != list(range(SIMILAR_COUNT)):
        raise SubgroupError("control assignment did not cover all similar candidates")
    selected = []
    pairs = []
    for index, similar_row in enumerate(similar, start=1):
        similar_id = f"S{index}"
        selected.append(
            _candidate_payload(
                similar_row, group="similar", candidate_id=similar_id,
                median=median, hp_center=hp_center, hp_scale=hp_scale,
                paired_candidate_id=f"C{index}", paired_hp_distance=0.0,
            )
        )
    for row_index, column_index in zip(row_indices, column_indices):
        index = int(row_index) + 1
        control_row = controls[int(column_index)]
        distance = float(cost[int(row_index), int(column_index)])
        control_id = f"C{index}"
        selected.append(
            _candidate_payload(
                control_row, group="control", candidate_id=control_id,
                median=median, hp_center=hp_center, hp_scale=hp_scale,
                paired_candidate_id=f"S{index}", paired_hp_distance=distance,
            )
        )
        pairs.append(
            {
                "similar_candidate_id": f"S{index}",
                "control_candidate_id": control_id,
                "standardized_hp_distance": distance,
            }
        )
    if len({row["architecture_fingerprint"] for row in selected}) != 6:
        raise SubgroupError("selected architecture fingerprints are not unique")
    if {row["architecture_fingerprint"] for row in selected} & excluded:
        raise SubgroupError("selected set contains a frozen panel architecture")
    audit = {
        "validation_floor": validation_floor,
        "validation_floor_definition": "median validation among valid deduplicated candidates",
        "representative_rule": (
            "highest-validation candidate per architecture; candidate fingerprint tie-break"
        ),
        "structure_distance_definition": "(operation_distance + edge_distance) / 2",
        "similar_filter": (
            "effective depth 5; no Identity/GCNII; contains GCN/GIN/SAGE; "
            ">=3 multi-predecessor nodes; mixed-width events >0"
        ),
        "similar_rank_rule": (
            "combined structure distance ascending, validation descending, fingerprint"
        ),
        "control_filter": f"structure distance to Median-1 >= {CONTROL_STRUCTURE_FLOOR}",
        "control_assignment": "global minimum standardized HP distance; unique controls",
        "hp_feature_names": hp_features(median)[1],
        "hp_center": hp_center.tolist(),
        "hp_scale": hp_scale.tolist(),
        "similar_eligible_count": len(similar_pool),
        "control_eligible_count": len(controls),
        "pairs": pairs,
    }
    return selected, audit


def _known_baseline_reuse_check(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths = [
        REPO_ROOT / "results/posthoc/cora-accuracy-decoder-b0-v1/track_b/screen_progress.json",
        REPO_ROOT / "results/posthoc/cora-activation-v1/screen_progress.json",
        REPO_ROOT / "results/posthoc/cora-concat-fusion-v1/screen_progress.json",
        REPO_ROOT / "results/posthoc/cora-layernorm-v1/screen_progress.json",
        REPO_ROOT / "results/posthoc/cora-layernorm-crossover-v1/crossover_progress.json",
    ]
    selected_fingerprints = {row["candidate_fingerprint"] for row in selected}
    matches = []
    checked = []
    for path in paths:
        if not path.exists():
            continue
        checked.append(
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": stage_a.sha256_file(path)}
        )
        for row in stage_a.load_json(path):
            if (
                row.get("variant") == "baseline"
                and row.get("candidate_fingerprint") in selected_fingerprints
            ):
                matches.append(
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "candidate_fingerprint": row["candidate_fingerprint"],
                        "replicate_id": row.get("replicate_id"),
                        "training_seed": row.get("training_seed"),
                    }
                )
    return {
        "checked_sources": checked,
        "exact_candidate_matches": matches,
        "reused_baseline_trainings": 0,
        "decision": (
            "No selected candidate has exact baseline provenance under the matched-seed "
            "protocol; all baselines are retrained."
        ),
    }


def prepare_manifest(cora_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze = _verify_crossover_freeze()
    layer_manifest = stage_a.load_json(
        REPO_ROOT / "results/posthoc/cora-layernorm-v1/layernorm_manifest.json"
    )
    gate, full, _low = stage_a.integrity_gate(
        REPO_ROOT, ARTIFACT_ROOT, CHECKPOINT, cora_root, None
    )
    if gate["status"] != "PASS" or not gate["semantics"]["search_records_contain_test_fields"] is False:
        raise SubgroupError("historical selection source is not validation-only")
    deduplicated, deduplication = stage_a.deduplicate_candidate_population(full)
    selected, selection_audit = select_candidates(
        deduplicated, layer_manifest["candidate_panel"]
    )
    manifest = {
        "format_version": 1,
        "experiment": "Cora LayerNorm architecture subgroup confirmation",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "crossover_freeze_commit": CROSSOVER_FREEZE_COMMIT,
        "crossover_freeze_sha256": stage_a.sha256_file(
            CROSSOVER_ROOT / "freeze_manifest.json"
        ),
        "crossover_result_sha256": freeze["result_hashes"]["crossover_results.json"],
        "dataset_identity": gate["cora_dataset"],
        "history_integrity": {
            "status": gate["status"],
            "population": gate["population"],
            "identity": gate["identity"],
            "semantics": gate["semantics"],
            "per_run": gate["per_run"],
            "deduplication": deduplication,
        },
        "candidate_selection_endpoint": "historical validation only",
        "selection_test_access": "PROHIBITED_UNTIL_MANIFEST_FROZEN",
        "selection_audit": selection_audit,
        "selected_candidates": selected,
        "baseline_reuse": _known_baseline_reuse_check(selected),
        "variants": list(VARIANTS),
        "replicates": REPLICATES,
        "new_training_count": EXPECTED_TRAININGS,
        "max_epochs": b0.MAX_EPOCHS,
        "patience": b0.PATIENCE,
        "optimizer": "Adam, unchanged train_and_eval_arch implementation",
        "scheduler": "CosineAnnealingLR, unchanged train_and_eval_arch implementation",
        "layernorm_placement": "conv -> LayerNorm -> ReLU -> dropout",
        "special_operation_rule": {
            "Identity": "bypasses LayerNorm",
            "GCNII": "same LayerNorm(hidden_dim) placement as other message-passing ops",
        },
        "matched_seed_rule": (
            "sha256_v1(BASE_SEED, b0_matched_training, candidate_fingerprint, "
            "replicate_id); variant excluded"
        ),
        "seed_mapping": {
            row["candidate_id"]: [
                b0.matched_training_seed(row["candidate_fingerprint"], replicate)
                for replicate in range(REPLICATES)
            ]
            for row in selected
        },
        "primary_endpoint": "validation matched delta",
        "architecture_responder_rule": (
            "mean validation delta >=0.004 and >=2/3 matched validation deltas >0"
        ),
        "group_decision_rule": {
            "ARCHITECTURE_SUBGROUP_CONFIRMED": (
                "similar responders >=2 and control responders <=1"
            ),
            "BROADER_LAYERNORM_EFFECT": (
                "similar responders >=2 and control responders >=2"
            ),
            "HP_OR_MIXED_EFFECT": (
                "subgroup/broader conditions fail; >=2 HP-matched pairs agree in responder "
                "status, including >=1 pair where both respond"
            ),
            "NO_GENERALIZATION": "similar responders <=1 and HP_OR_MIXED_EFFECT is false",
            "INCONCLUSIVE": "any invalid or failed paired run",
        },
        "test_access_rule": (
            "write validation_decision.json first; only then read test_accuracy from raw progress; "
            "test cannot change responder labels or group verdict"
        ),
        "source_identity": {
            "subgroup_script_sha256": stage_a.sha256_file(Path(__file__).resolve()),
            "evaluator_sha256": stage_a.sha256_file(REPO_ROOT / "eval_utils.py"),
            "checkpoint_sha256": stage_a.sha256_file(CHECKPOINT),
        },
    }
    return manifest, selected


def _train(
    candidate: Mapping[str, Any], variant: str, replicate: int, data,
    in_ch: int, out_ch: int, device: torch.device,
) -> dict[str, Any]:
    seed = b0.matched_training_seed(candidate["candidate_fingerprint"], replicate)
    config = {
        "operations": list(candidate["operations"]),
        "edges": [tuple(edge) for edge in candidate["edges"]],
        "effective_layers": sum(op != "Identity" for op in candidate["operations"]),
    }
    started = time.monotonic()
    try:
        validation, valid, test, metadata = train_and_eval_arch(
            config=config, data=data, in_ch=in_ch, out_ch=out_ch,
            lr=float(candidate["lr"]), dropout=float(candidate["dropout"]),
            hidden_dim=int(candidate["hidden_dim"]), weight_decay=float(candidate["l2"]),
            gat_heads=int(candidate.get("gat_heads") or 1),
            sage_aggr=str(candidate.get("sage_aggr") or "mean"),
            gin_eps=float(candidate.get("gin_eps") or 0.0),
            gat_heads_by_layer=candidate.get("gat_heads_by_layer"),
            sage_aggr_by_layer=candidate.get("sage_aggr_by_layer"),
            gin_eps_by_layer=candidate.get("gin_eps_by_layer"),
            device=device, max_epochs=b0.MAX_EPOCHS, patience=b0.PATIENCE,
            seed=seed, track_test=True, return_metadata=True,
            return_reproducibility_metadata=True, component_variant=variant,
            activation_variant="relu", return_trajectory=True,
        )
    except InfrastructureEvaluationError as exc:
        validation, valid, test = 0.0, False, 0.0
        metadata = {
            "best_epoch": None, "stopped_epoch": None, "epochs_ran": 0,
            "trajectory": [], "trajectory_hash": None,
            "model_final_state_hash": None, "parameter_count": None,
            "failure": {
                "type": type(exc).__name__, "message": str(exc), "context": exc.context,
            },
        }
    return {
        "group": candidate["group"], "candidate_id": candidate["candidate_id"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "architecture_fingerprint": candidate["architecture_fingerprint"],
        "variant": variant, "replicate_id": int(replicate),
        "training_seed": int(seed), "validation_accuracy": float(validation),
        "test_accuracy": float(test), "valid": bool(valid),
        "runtime_seconds": float(time.monotonic() - started), **metadata,
    }


def _validation_view(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if "test" not in key.lower()}


def analyze_validation(
    manifest: Mapping[str, Any], validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if any(any("test" in key.lower() for key in row) for row in validation_rows):
        raise SubgroupError("validation decision received a test field")
    expected = {
        (candidate["candidate_fingerprint"], variant, replicate)
        for candidate in manifest["selected_candidates"]
        for variant in VARIANTS
        for replicate in range(REPLICATES)
    }
    index = {
        (row["candidate_fingerprint"], row["variant"], int(row["replicate_id"])): row
        for row in validation_rows
    }
    if set(index) != expected or len(validation_rows) != EXPECTED_TRAININGS:
        raise SubgroupError("validation matrix mismatch")
    by_candidate = {}
    failures = []
    for candidate in manifest["selected_candidates"]:
        fingerprint = candidate["candidate_fingerprint"]
        paired = []
        for replicate in range(REPLICATES):
            baseline = index[(fingerprint, "baseline", replicate)]
            normalized = index[(fingerprint, "layer_norm", replicate)]
            expected_seed = b0.matched_training_seed(fingerprint, replicate)
            if (
                int(baseline["training_seed"]) != expected_seed
                or int(normalized["training_seed"]) != expected_seed
            ):
                raise SubgroupError("paired seed mismatch")
            pair = {
                "replicate_id": replicate, "training_seed": expected_seed,
                "baseline_validation": baseline["validation_accuracy"],
                "layernorm_validation": normalized["validation_accuracy"],
                "validation_delta": (
                    normalized["validation_accuracy"] - baseline["validation_accuracy"]
                ),
                "baseline_best_epoch": baseline["best_epoch"],
                "layernorm_best_epoch": normalized["best_epoch"],
                "baseline_runtime_seconds": baseline["runtime_seconds"],
                "layernorm_runtime_seconds": normalized["runtime_seconds"],
                "baseline_parameter_count": baseline["parameter_count"],
                "layernorm_parameter_count": normalized["parameter_count"],
                "valid": bool(baseline["valid"] and normalized["valid"]),
                "failures": [baseline.get("failure"), normalized.get("failure")],
            }
            if not pair["valid"] or any(pair["failures"]):
                failures.append(pair)
            paired.append(pair)
        delta = _mean([row["validation_delta"] for row in paired])
        positive = sum(row["validation_delta"] > 0 for row in paired)
        responder = delta >= RESPONDER_THRESHOLD and positive >= 2
        by_candidate[candidate["candidate_id"]] = {
            "group": candidate["group"],
            "candidate_fingerprint": fingerprint,
            "architecture_fingerprint": candidate["architecture_fingerprint"],
            "baseline_validation_mean": _mean(
                [row["baseline_validation"] for row in paired]
            ),
            "layernorm_validation_mean": _mean(
                [row["layernorm_validation"] for row in paired]
            ),
            "validation_delta_mean": delta,
            "positive_validation_seeds": positive,
            "responder": responder,
            "per_seed": paired,
        }
    similar = sum(
        row["responder"] for row in by_candidate.values() if row["group"] == "similar"
    )
    control = sum(
        row["responder"] for row in by_candidate.values() if row["group"] == "control"
    )
    pair_status = []
    for pair in manifest["selection_audit"]["pairs"]:
        similar_status = by_candidate[pair["similar_candidate_id"]]["responder"]
        control_status = by_candidate[pair["control_candidate_id"]]["responder"]
        pair_status.append(
            {
                **pair, "similar_responder": similar_status,
                "control_responder": control_status,
                "agreement": similar_status == control_status,
            }
        )
    agreeing_pairs = sum(row["agreement"] for row in pair_status)
    both_responder_pairs = sum(
        row["similar_responder"] and row["control_responder"] for row in pair_status
    )
    if failures:
        verdict = "INCONCLUSIVE"
    elif similar >= 2 and control <= 1:
        verdict = "ARCHITECTURE_SUBGROUP_CONFIRMED"
    elif similar >= 2 and control >= 2:
        verdict = "BROADER_LAYERNORM_EFFECT"
    elif agreeing_pairs >= 2 and both_responder_pairs >= 1:
        verdict = "HP_OR_MIXED_EFFECT"
    else:
        verdict = "NO_GENERALIZATION"
    next_step = {
        "ARCHITECTURE_SUBGROUP_CONFIRMED": "TEST_HIGH_PERFORMANCE_SUBGROUP",
        "BROADER_LAYERNORM_EFFECT": "ADD_LAYERNORM_AS_SEARCHABLE_CHOICE",
        "HP_OR_MIXED_EFFECT": "INVESTIGATE_HP_INTERACTION",
        "NO_GENERALIZATION": "MOVE_TO_PERFORMANCE_AWARE_REPRESENTATION",
        "INCONCLUSIVE": "MORE_EVIDENCE",
    }[verdict]
    return {
        "primary_endpoint": "validation matched delta",
        "verdict": verdict,
        "recommended_next_step": next_step,
        "similar_responders": similar,
        "control_responders": control,
        "failed_pairs": len(failures),
        "hp_matched_pair_status": pair_status,
        "by_candidate": by_candidate,
    }


def attach_test_results(
    decision: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(decision))
    index = {
        (row["candidate_fingerprint"], row["variant"], int(row["replicate_id"])): row
        for row in raw_rows
    }
    for candidate_id, summary in result["by_candidate"].items():
        fingerprint = summary["candidate_fingerprint"]
        test_pairs = []
        for replicate in range(REPLICATES):
            baseline = index[(fingerprint, "baseline", replicate)]
            normalized = index[(fingerprint, "layer_norm", replicate)]
            test_pairs.append(
                {
                    "replicate_id": replicate,
                    "training_seed": baseline["training_seed"],
                    "baseline_test": baseline["test_accuracy"],
                    "layernorm_test": normalized["test_accuracy"],
                    "test_delta": normalized["test_accuracy"] - baseline["test_accuracy"],
                }
            )
        summary["baseline_test_mean"] = _mean([row["baseline_test"] for row in test_pairs])
        summary["layernorm_test_mean"] = _mean(
            [row["layernorm_test"] for row in test_pairs]
        )
        summary["test_delta_mean"] = _mean([row["test_delta"] for row in test_pairs])
        summary["positive_test_seeds"] = sum(row["test_delta"] > 0 for row in test_pairs)
        summary["per_seed_test"] = test_pairs
    result["test_read_after_validation_decision"] = True
    result["test_did_not_change_verdict"] = result["verdict"] == decision["verdict"]
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument(
        "--output", default="results/posthoc/cora-layernorm-subgroup-v1"
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = (REPO_ROOT / args.output).resolve()
    result_path = output / "subgroup_results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite subgroup result: {result_path}")
    manifest, selected = prepare_manifest((REPO_ROOT / args.cora_root).resolve())
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "subgroup_manifest.json"
    if manifest_path.exists():
        if stage_a.load_json(manifest_path) != manifest:
            raise SubgroupError("existing subgroup manifest differs from current design")
    else:
        stage_a.dump_json(manifest_path, manifest)
    resource_gate = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "prepare_only": bool(args.prepare_only), "training_started": False,
    }
    if torch.cuda.is_available():
        resource_gate["device_name"] = torch.cuda.get_device_name(0)
    stage_a.dump_json(output / "resource_gate.json", resource_gate)
    if args.prepare_only:
        print(output)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("LayerNorm subgroup screen requires CUDA; CPU fallback is prohibited")
    resource_gate["training_started"] = True
    stage_a.dump_json(output / "resource_gate.json", resource_gate)
    device = torch.device("cuda:0")
    data, in_ch, out_ch = b0._load_cora((REPO_ROOT / args.cora_root).resolve(), device)
    progress_path = output / "raw_progress_test_sealed.json"
    rows = stage_a.load_json(progress_path) if progress_path.exists() else []
    completed = {
        (row["candidate_fingerprint"], row["variant"], int(row["replicate_id"]))
        for row in rows
    }
    for candidate in selected:
        for variant in VARIANTS:
            for replicate in range(REPLICATES):
                key = (candidate["candidate_fingerprint"], variant, replicate)
                if key in completed:
                    continue
                row = _train(candidate, variant, replicate, data, in_ch, out_ch, device)
                rows.append(row)
                stage_a.dump_json(progress_path, rows)
                print(
                    f"{candidate['candidate_id']} {variant} seed{replicate}: "
                    f"validation={row['validation_accuracy']:.3f}"
                )
    validation_rows = [_validation_view(row) for row in rows]
    decision = analyze_validation(manifest, validation_rows)
    decision_path = output / "validation_decision.json"
    stage_a.dump_json(decision_path, decision)
    decision_hash = stage_a.sha256_file(decision_path)
    result = attach_test_results(decision, rows)
    result["validation_decision_sha256_before_test_read"] = decision_hash
    stage_a.dump_json(result_path, result)
    stage_a.dump_json(
        output / "training_cost.json",
        {
            "reused_trainings": 0, "new_trainings": len(rows),
            "gpu_seconds": sum(float(row["runtime_seconds"]) for row in rows),
        },
    )
    print(f"verdict={result['verdict']}")
    print(f"recommended_next_step={result['recommended_next_step']}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
