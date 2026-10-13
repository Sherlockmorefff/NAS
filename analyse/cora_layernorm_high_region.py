"""Validation-first Cora LayerNorm high-performance-region confirmation."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import cora_accuracy_stage_a as stage_a
from analyse import cora_component_screen as b0
from analyse import cora_layernorm_subgroup as subgroup


SUBGROUP_ROOT = REPO_ROOT / "results/posthoc/cora-layernorm-subgroup-v1"
SUBGROUP_FREEZE_COMMIT = "a6cea690a5b4e55ccfc769cd22bb38665a2e2942"
ARTIFACT_ROOT = REPO_ROOT / "legacy_artifacts/pre_20260813/results"
CHECKPOINT = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
ARCHIVE_BEST_VALIDATION = 0.828
SELECTION_COUNT = 5
MIN_STRUCTURE_DISTANCE = 0.20
RESPONDER_THRESHOLD = 0.004
REPLICATES = 3
VARIANTS = ("baseline", "layer_norm")
EXPECTED_TRAININGS = 30


class HighRegionError(RuntimeError):
    """Raised when high-region selection or experiment invariants fail."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _frozen_file(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SUBGROUP_FREEZE_COMMIT}:{path}"], cwd=REPO_ROOT,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def _verify_subgroup_freeze() -> dict[str, Any]:
    path = SUBGROUP_ROOT / "freeze_manifest.json"
    freeze = stage_a.load_json(path)
    if freeze.get("archive_status") != "FROZEN":
        raise HighRegionError("subgroup archive is not frozen")
    manifest_expected = freeze["candidate_manifest"]["sha256"]
    manifest_path = SUBGROUP_ROOT / "subgroup_manifest.json"
    if stage_a.sha256_file(manifest_path) != manifest_expected:
        raise HighRegionError("subgroup manifest hash mismatch")
    if hashlib.sha256(
        _frozen_file(str(manifest_path.relative_to(REPO_ROOT)))
    ).hexdigest() != manifest_expected:
        raise HighRegionError("subgroup commit manifest mismatch")
    for relative, expected in freeze["result_hashes"].items():
        result_path = SUBGROUP_ROOT / relative
        if stage_a.sha256_file(result_path) != expected:
            raise HighRegionError(f"subgroup result hash mismatch: {relative}")
        if hashlib.sha256(
            _frozen_file(str(result_path.relative_to(REPO_ROOT)))
        ).hexdigest() != expected:
            raise HighRegionError(f"subgroup commit result mismatch: {relative}")
    for relative, expected in freeze["source_hashes"].items():
        if hashlib.sha256(_frozen_file(relative)).hexdigest() != expected:
            raise HighRegionError(f"subgroup source mismatch: {relative}")
    return freeze


def _structure_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return (
        stage_a.op_distance(left, right) + stage_a.edge_distance(left, right)
    ) / 2.0


def _best_validation_by_architecture(
    records: Sequence[Mapping[str, Any]], excluded: set[str],
) -> list[dict[str, Any]]:
    best: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not bool(record.get("valid", True)) or record["architecture_fingerprint"] in excluded:
            continue
        fingerprint = record["architecture_fingerprint"]
        previous = best.get(fingerprint)
        if previous is None or (
            -float(record["val_acc"]), record["candidate_fingerprint"]
        ) < (-float(previous["val_acc"]), previous["candidate_fingerprint"]):
            best[fingerprint] = record
    return sorted(
        (copy.deepcopy(row) for row in best.values()),
        key=lambda row: (-float(row["val_acc"]), row["candidate_fingerprint"]),
    )


def _candidate_payload(
    record: Mapping[str, Any], selection_rank: int, selected_index: int,
    minimum_distance: float | None,
) -> dict[str, Any]:
    fields = (
        "candidate_fingerprint", "architecture_fingerprint", "operations", "edges",
        "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
        "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer", "val_acc",
        "_method", "_run_seed", "_run_name", "full_evaluation_index",
    )
    payload = {field: copy.deepcopy(record.get(field)) for field in fields}
    payload.update(
        {
            "group": "high_region",
            "candidate_id": f"H{selected_index}",
            "historical_validation": float(record["val_acc"]),
            "selection_rank": int(selection_rank),
            "minimum_structure_distance_to_earlier_selection": minimum_distance,
            "architecture_summary": {
                "operation_count": len(record["operations"]),
                "operations": list(record["operations"]),
                "edge_count": len(record["edges"]),
                "effective_depth": sum(op != "Identity" for op in record["operations"]),
                "has_identity": "Identity" in record["operations"],
                "has_gcnii": "GCNII" in record["operations"],
            },
        }
    )
    if any("test" in key.lower() for key in payload):
        raise HighRegionError("selected candidate payload contains a test field")
    return payload


def select_high_region(
    deduplicated: Sequence[Mapping[str, Any]], excluded: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if any(any("test" in key.lower() for key in row) for row in deduplicated):
        raise HighRegionError("selection population contains test fields")
    ranked = _best_validation_by_architecture(deduplicated, excluded)
    selected: list[tuple[int, Mapping[str, Any], float | None]] = []
    for rank, record in enumerate(ranked, start=1):
        minimum = None
        if selected:
            minimum = min(_structure_distance(record, prior[1]) for prior in selected)
            if minimum < MIN_STRUCTURE_DISTANCE:
                continue
        selected.append((rank, record, minimum))
        if len(selected) == SELECTION_COUNT:
            break
    if len(selected) != SELECTION_COUNT:
        raise HighRegionError("could not select five structurally diverse candidates")
    payloads = [
        _candidate_payload(record, rank, index, minimum)
        for index, (rank, record, minimum) in enumerate(selected, start=1)
    ]
    if len({row["architecture_fingerprint"] for row in payloads}) != SELECTION_COUNT:
        raise HighRegionError("selected architecture fingerprints are not unique")
    pairwise = []
    for left_index, left in enumerate(payloads):
        for right in payloads[left_index + 1:]:
            pairwise.append(
                {
                    "left": left["candidate_id"], "right": right["candidate_id"],
                    "combined_structure_distance": _structure_distance(left, right),
                }
            )
    audit = {
        "selection_endpoint": "historical validation only",
        "representative_rule": (
            "highest-validation candidate per architecture; candidate fingerprint tie-break"
        ),
        "rank_rule": "validation descending, candidate fingerprint tie-break",
        "diversity_rule": (
            f"greedy rank order with minimum pairwise discrete structure distance "
            f">={MIN_STRUCTURE_DISTANCE}"
        ),
        "structure_distance_definition": "(operation_distance + edge_distance) / 2",
        "remaining_unique_architectures": len(ranked),
        "available_at_or_above_0_82": sum(float(row["val_acc"]) >= 0.82 for row in ranked),
        "selected_ranks": [row["selection_rank"] for row in payloads],
        "pairwise_structure_distances": pairwise,
    }
    return payloads, audit


def _known_baseline_reuse_check(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths = [
        REPO_ROOT / "results/posthoc/cora-accuracy-decoder-b0-v1/track_b/screen_progress.json",
        REPO_ROOT / "results/posthoc/cora-layernorm-v1/screen_progress.json",
        REPO_ROOT / "results/posthoc/cora-layernorm-crossover-v1/crossover_progress.json",
        REPO_ROOT / "results/posthoc/cora-layernorm-subgroup-v1/raw_progress_test_sealed.json",
    ]
    fingerprints = {row["candidate_fingerprint"] for row in selected}
    matches = []
    checked = []
    for path in paths:
        if not path.exists():
            continue
        checked.append(
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": stage_a.sha256_file(path)}
        )
        for row in stage_a.load_json(path):
            if row.get("variant") == "baseline" and row.get("candidate_fingerprint") in fingerprints:
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
            "No selected candidate has exact matched-seed baseline provenance; all baselines "
            "are retrained."
        ),
    }


def prepare_manifest(cora_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze = _verify_subgroup_freeze()
    layer_manifest = stage_a.load_json(
        REPO_ROOT / "results/posthoc/cora-layernorm-v1/layernorm_manifest.json"
    )
    subgroup_manifest = stage_a.load_json(SUBGROUP_ROOT / "subgroup_manifest.json")
    gate, full, _low = stage_a.integrity_gate(
        REPO_ROOT, ARTIFACT_ROOT, CHECKPOINT, cora_root, None
    )
    if gate["status"] != "PASS" or gate["semantics"]["search_records_contain_test_fields"]:
        raise HighRegionError("historical selection source is not validation-only")
    deduplicated, deduplication = stage_a.deduplicate_candidate_population(full)
    excluded = {
        row["architecture_fingerprint"] for row in layer_manifest["candidate_panel"]
    } | {
        row["architecture_fingerprint"] for row in subgroup_manifest["selected_candidates"]
    }
    selected, selection_audit = select_high_region(deduplicated, excluded)
    manifest = {
        "format_version": 1,
        "experiment": "Cora LayerNorm high-performance-region gate",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "subgroup_freeze_commit": SUBGROUP_FREEZE_COMMIT,
        "subgroup_freeze_sha256": stage_a.sha256_file(
            SUBGROUP_ROOT / "freeze_manifest.json"
        ),
        "subgroup_result_sha256": freeze["result_hashes"]["subgroup_results.json"],
        "dataset_identity": gate["cora_dataset"],
        "history_integrity": {
            "status": gate["status"], "population": gate["population"],
            "identity": gate["identity"], "semantics": gate["semantics"],
            "per_run": gate["per_run"], "deduplication": deduplication,
        },
        "archive_best_validation": ARCHIVE_BEST_VALIDATION,
        "excluded_architecture_fingerprints": sorted(excluded),
        "candidate_selection_endpoint": "historical validation only",
        "selection_test_access": "PROHIBITED_UNTIL_MANIFEST_FROZEN",
        "selection_audit": selection_audit,
        "selected_candidates": selected,
        "baseline_reuse": _known_baseline_reuse_check(selected),
        "variants": list(VARIANTS), "replicates": REPLICATES,
        "new_training_count": EXPECTED_TRAININGS,
        "max_epochs": b0.MAX_EPOCHS, "patience": b0.PATIENCE,
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
        "decision_rule": {
            "CEILING_BREAK": (
                "at least one responder has LayerNorm mean validation >0.828"
            ),
            "HIGH_REGION_GAIN": (
                "no ceiling break; >=3/5 responders and overall mean validation delta >=0.004"
            ),
            "LOW_REGION_ONLY": (
                "<=1/5 responders and overall mean validation delta <0.004"
            ),
            "INCONCLUSIVE": "any other pattern, or any invalid/failed paired run",
        },
        "test_access_rule": (
            "write validation_decision.json first; only then read test_accuracy from raw progress; "
            "test cannot change responder labels or verdict"
        ),
        "source_identity": {
            "high_region_script_sha256": stage_a.sha256_file(Path(__file__).resolve()),
            "evaluator_sha256": stage_a.sha256_file(REPO_ROOT / "eval_utils.py"),
            "checkpoint_sha256": stage_a.sha256_file(CHECKPOINT),
        },
    }
    return manifest, selected


def analyze_validation(
    manifest: Mapping[str, Any], validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if any(any("test" in key.lower() for key in row) for row in validation_rows):
        raise HighRegionError("validation decision received a test field")
    expected = {
        (candidate["candidate_fingerprint"], variant, replicate)
        for candidate in manifest["selected_candidates"]
        for variant in VARIANTS for replicate in range(REPLICATES)
    }
    index = {
        (row["candidate_fingerprint"], row["variant"], int(row["replicate_id"])): row
        for row in validation_rows
    }
    if set(index) != expected or len(validation_rows) != EXPECTED_TRAININGS:
        raise HighRegionError("validation matrix mismatch")
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
                raise HighRegionError("paired seed mismatch")
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
        layernorm_mean = _mean([row["layernorm_validation"] for row in paired])
        by_candidate[candidate["candidate_id"]] = {
            "candidate_fingerprint": fingerprint,
            "architecture_fingerprint": candidate["architecture_fingerprint"],
            "historical_validation": candidate["historical_validation"],
            "selection_rank": candidate["selection_rank"],
            "baseline_validation_mean": _mean(
                [row["baseline_validation"] for row in paired]
            ),
            "layernorm_validation_mean": layernorm_mean,
            "validation_delta_mean": delta,
            "positive_validation_seeds": positive,
            "responder": responder,
            "beats_archive_best": responder and layernorm_mean > ARCHIVE_BEST_VALIDATION,
            "per_seed": paired,
        }
    responders = sum(row["responder"] for row in by_candidate.values())
    overall_delta = _mean(
        [row["validation_delta"] for row in by_candidate.values() for row in row["per_seed"]]
    )
    ceiling_breakers = [
        candidate_id for candidate_id, row in by_candidate.items()
        if row["beats_archive_best"]
    ]
    if failures:
        verdict = "INCONCLUSIVE"
    elif ceiling_breakers:
        verdict = "CEILING_BREAK"
    elif responders >= 3 and overall_delta >= RESPONDER_THRESHOLD:
        verdict = "HIGH_REGION_GAIN"
    elif responders <= 1 and overall_delta < RESPONDER_THRESHOLD:
        verdict = "LOW_REGION_ONLY"
    else:
        verdict = "INCONCLUSIVE"
    next_step = {
        "CEILING_BREAK": "CONFIRM_CEILING_BREAK",
        "HIGH_REGION_GAIN": "ADD_LAYERNORM_TO_SEARCH_SPACE",
        "LOW_REGION_ONLY": "MOVE_TO_PERFORMANCE_AWARE_REPRESENTATION",
        "INCONCLUSIVE": "MORE_EVIDENCE",
    }[verdict]
    nas_decision = {
        "CEILING_BREAK": "YES", "HIGH_REGION_GAIN": "YES",
        "LOW_REGION_ONLY": "NO", "INCONCLUSIVE": "CONDITIONAL",
    }[verdict]
    return {
        "primary_endpoint": "validation matched delta",
        "archive_best_validation": ARCHIVE_BEST_VALIDATION,
        "verdict": verdict, "recommended_next_step": next_step,
        "should_layernorm_enter_nas": nas_decision,
        "responder_count": responders,
        "overall_validation_delta_mean": overall_delta,
        "ceiling_breakers": ceiling_breakers,
        "failed_pairs": len(failures), "by_candidate": by_candidate,
    }


def attach_test_results(
    decision: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(decision))
    index = {
        (row["candidate_fingerprint"], row["variant"], int(row["replicate_id"])): row
        for row in raw_rows
    }
    for summary in result["by_candidate"].values():
        fingerprint = summary["candidate_fingerprint"]
        pairs = []
        for replicate in range(REPLICATES):
            baseline = index[(fingerprint, "baseline", replicate)]
            normalized = index[(fingerprint, "layer_norm", replicate)]
            pairs.append(
                {
                    "replicate_id": replicate, "training_seed": baseline["training_seed"],
                    "baseline_test": baseline["test_accuracy"],
                    "layernorm_test": normalized["test_accuracy"],
                    "test_delta": normalized["test_accuracy"] - baseline["test_accuracy"],
                }
            )
        summary["baseline_test_mean"] = _mean([row["baseline_test"] for row in pairs])
        summary["layernorm_test_mean"] = _mean([row["layernorm_test"] for row in pairs])
        summary["test_delta_mean"] = _mean([row["test_delta"] for row in pairs])
        summary["positive_test_seeds"] = sum(row["test_delta"] > 0 for row in pairs)
        summary["per_seed_test"] = pairs
    result["test_read_after_validation_decision"] = True
    result["test_did_not_change_verdict"] = result["verdict"] == decision["verdict"]
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument(
        "--output", default="results/posthoc/cora-layernorm-high-region-v1"
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = (REPO_ROOT / args.output).resolve()
    result_path = output / "high_region_results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite high-region result: {result_path}")
    manifest, selected = prepare_manifest((REPO_ROOT / args.cora_root).resolve())
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "high_region_manifest.json"
    if manifest_path.exists():
        if stage_a.load_json(manifest_path) != manifest:
            raise HighRegionError("existing high-region manifest differs from current design")
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
        raise RuntimeError("LayerNorm high-region gate requires CUDA; CPU fallback is prohibited")
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
                row = subgroup._train(
                    candidate, variant, replicate, data, in_ch, out_ch, device
                )
                rows.append(row)
                stage_a.dump_json(progress_path, rows)
                print(
                    f"{candidate['candidate_id']} {variant} seed{replicate}: "
                    f"validation={row['validation_accuracy']:.3f}"
                )
    validation_rows = [subgroup._validation_view(row) for row in rows]
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
