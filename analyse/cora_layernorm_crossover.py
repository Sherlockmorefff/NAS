"""Matched-seed Cora architecture x HP crossover for the LayerNorm responder."""

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
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import cora_accuracy_stage_a as stage_a
from analyse import cora_component_screen as b0
from evaluation_errors import InfrastructureEvaluationError
from eval_utils import train_and_eval_arch


LAYERNORM_ROOT = REPO_ROOT / "results/posthoc/cora-layernorm-v1"
LAYERNORM_FREEZE_COMMIT = "a089b0fcafdb9ecdad02dd622799214e4725be33"
REPLICATES = 3
EXPECTED_NEW_TRAININGS = 12
GAIN_THRESHOLD = 0.004
HP_FIELDS = ("lr", "dropout", "hidden_dim", "l2")
VARIANTS = ("baseline", "layer_norm")


class CrossoverError(RuntimeError):
    """Raised when crossover design or provenance invariants fail."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _frozen_file(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{LAYERNORM_FREEZE_COMMIT}:{path}"],
        cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def _verify_layernorm_freeze() -> dict[str, Any]:
    freeze_path = LAYERNORM_ROOT / "freeze_manifest.json"
    freeze = stage_a.load_json(freeze_path)
    if freeze.get("archive_status") != "FROZEN":
        raise CrossoverError("LayerNorm archive is not frozen")
    for relative, expected in freeze["result_hashes"].items():
        path = LAYERNORM_ROOT / relative
        if stage_a.sha256_file(path) != expected:
            raise CrossoverError(f"LayerNorm result hash mismatch: {relative}")
        committed = _frozen_file(str(path.relative_to(REPO_ROOT)))
        if hashlib.sha256(committed).hexdigest() != expected:
            raise CrossoverError(f"freeze commit result mismatch: {relative}")
    for relative, expected in freeze["source_hashes"].items():
        committed = _frozen_file(relative)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise CrossoverError(f"freeze commit source mismatch: {relative}")
    return freeze


def _candidate_by_role(manifest: Mapping[str, Any], role: str) -> dict[str, Any]:
    matches = [row for row in manifest["candidate_panel"] if row["panel_role"] == role]
    if len(matches) != 1:
        raise CrossoverError(f"expected one candidate for {role}, got {len(matches)}")
    return copy.deepcopy(matches[0])


def crossover_candidate(
    architecture: Mapping[str, Any], hp: Mapping[str, Any], role: str,
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(architecture))
    for field in HP_FIELDS:
        candidate[field] = hp[field]
    hp_payload = {field: candidate[field] for field in HP_FIELDS}
    candidate["panel_role"] = role
    candidate["architecture_source_role"] = architecture["panel_role"]
    candidate["hp_source_role"] = hp["panel_role"]
    candidate["crossover_hp_fingerprint"] = stage_a.canonical_json_sha256(hp_payload)
    candidate["candidate_fingerprint"] = stage_a.canonical_json_sha256(
        {
            "experiment": "cora_layernorm_crossover_v1",
            "architecture_fingerprint": candidate["architecture_fingerprint"],
            "hp_fingerprint": candidate["crossover_hp_fingerprint"],
        }
    )
    return candidate


def _predecessor_counts(candidate: Mapping[str, Any]) -> list[int]:
    counts = {node: 0 for node in range(1, len(candidate["operations"]) + 2)}
    for source, target in candidate["edges"]:
        if int(target) in counts:
            counts[int(target)] += 1
    return [counts[node] for node in sorted(counts)]


def _structure_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    stress = candidate["truncation_stress"]
    return {
        "operations": candidate["operations"],
        "edges": candidate["edges"],
        "edge_count": len(candidate["edges"]),
        "operation_depth": len(candidate["operations"]),
        "effective_message_passing_depth": sum(
            operation != "Identity" for operation in candidate["operations"]
        ),
        "predecessor_counts_operations_and_output": _predecessor_counts(candidate),
        "mixed_width_predecessor": bool(stress["mismatch_events"]),
        "mixed_width_mismatch_events": int(stress["mismatch_events"]),
        "static_columns_discarded": int(stress["static_columns_discarded"]),
        "contains_gcnii": "GCNII" in candidate["operations"],
        "hidden_dim": int(candidate["hidden_dim"]),
        "lr": float(candidate["lr"]),
        "dropout": float(candidate["dropout"]),
        "weight_decay": float(candidate["l2"]),
    }


def _existing_combo(
    role: str, candidate: Mapping[str, Any], result: Mapping[str, Any],
) -> dict[str, Any]:
    summary = result["by_architecture"][candidate["panel_role"]]
    return {
        "combination": role,
        "architecture_source_role": candidate["panel_role"],
        "hp_source_role": candidate["panel_role"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "baseline_validation_mean": summary["baseline_validation_mean"],
        "layernorm_validation_mean": summary["layernorm_validation_mean"],
        "validation_delta_mean": summary["validation_delta_mean"],
        "validation_positive_seeds": summary["validation_positive_seeds"],
        "baseline_test_mean": summary["baseline_test_mean"],
        "layernorm_test_mean": summary["layernorm_test_mean"],
        "test_delta_mean": summary["test_delta_mean"],
        "test_positive_seeds": summary["test_positive_seeds"],
        "per_seed": summary["per_seed"],
        "source": "reused_frozen_layernorm_screen",
    }


def prepare_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze = _verify_layernorm_freeze()
    layer_manifest = stage_a.load_json(LAYERNORM_ROOT / "layernorm_manifest.json")
    layer_result = stage_a.load_json(LAYERNORM_ROOT / "layernorm_results.json")
    if layer_result["classification"] != "ARCHITECTURE_SPECIFIC":
        raise CrossoverError("LayerNorm freeze is not ARCHITECTURE_SPECIFIC")
    if layer_manifest["dataset_identity"]["split_protocol"] != "planetoid_public":
        raise CrossoverError("unexpected split protocol")
    top = _candidate_by_role(layer_manifest, "top_1")
    median = _candidate_by_role(layer_manifest, "median_1")
    cross = [
        crossover_candidate(median, top, "median_arch_top_hp"),
        crossover_candidate(top, median, "top_arch_median_hp"),
    ]
    if len({row["candidate_fingerprint"] for row in cross}) != 2:
        raise CrossoverError("crossover fingerprints are not distinct")
    existing = [
        _existing_combo("median_arch_median_hp", median, layer_result),
        _existing_combo("top_arch_top_hp", top, layer_result),
    ]
    manifest = {
        "format_version": 1,
        "experiment": "Cora LayerNorm architecture x HP crossover",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "research_question": (
            "Is the Median-1 LayerNorm gain caused by architecture, HP, or their joint setting?"
        ),
        "layernorm_freeze_commit": LAYERNORM_FREEZE_COMMIT,
        "layernorm_freeze_sha256": stage_a.sha256_file(
            LAYERNORM_ROOT / "freeze_manifest.json"
        ),
        "candidate_manifest": freeze["candidate_manifest"],
        "dataset_identity": layer_manifest["dataset_identity"],
        "structure_comparison": {
            row["panel_role"]: _structure_summary(row)
            for row in layer_manifest["candidate_panel"]
        },
        "hp_swap_fields": list(HP_FIELDS),
        "existing_combinations": existing,
        "new_combinations": cross,
        "variants": list(VARIANTS),
        "replicates": REPLICATES,
        "new_training_count": EXPECTED_NEW_TRAININGS,
        "max_epochs": b0.MAX_EPOCHS,
        "patience": b0.PATIENCE,
        "optimizer": "Adam, unchanged train_and_eval_arch implementation",
        "scheduler": "CosineAnnealingLR, unchanged train_and_eval_arch implementation",
        "layernorm_placement": "conv -> LayerNorm -> ReLU -> dropout",
        "matched_seed_scheme": (
            "sha256_v1(BASE_SEED, b0_matched_training, crossover_candidate_fingerprint, "
            "replicate_id); variant excluded"
        ),
        "seed_mapping": {
            row["panel_role"]: [
                b0.matched_training_seed(row["candidate_fingerprint"], replicate)
                for replicate in range(REPLICATES)
            ]
            for row in cross
        },
        "primary_endpoint": "validation matched delta",
        "test_access_rule": (
            "test is read only after matrix, combinations, seed mapping, and decision rule "
            "are written to the manifest"
        ),
        "decision_rule": {
            "gain": "mean validation delta >=0.004 and >=2/3 positive matched seeds",
            "no_gain": "mean validation delta <=0 and <=1/3 positive matched seeds",
            "borderline": "anything between gain and no_gain",
            "ARCHITECTURE_EFFECT": (
                "Median architecture + Top HP is gain; Top architecture + Median HP is no_gain"
            ),
            "HP_EFFECT": (
                "Top architecture + Median HP is gain; Median architecture + Top HP is no_gain"
            ),
            "JOINT_EFFECT": "both new crossover combinations are no_gain",
            "BROADER_GAIN": "both new crossover combinations are gain",
            "INCONCLUSIVE": "any remaining pattern, including a borderline combination",
        },
        "source_identity": {
            "crossover_script_sha256": stage_a.sha256_file(Path(__file__).resolve()),
            "evaluator_sha256": stage_a.sha256_file(REPO_ROOT / "eval_utils.py"),
        },
    }
    return manifest, cross


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
        "combination": candidate["panel_role"],
        "architecture_source_role": candidate["architecture_source_role"],
        "hp_source_role": candidate["hp_source_role"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "architecture_fingerprint": candidate["architecture_fingerprint"],
        "variant": variant, "replicate_id": int(replicate),
        "training_seed": int(seed), "validation_accuracy": float(validation),
        "test_accuracy": float(test), "valid": bool(valid),
        "runtime_seconds": float(time.monotonic() - started), **metadata,
    }


def _state(rows: Sequence[Mapping[str, Any]]) -> str:
    mean_delta = _mean([row["validation_delta"] for row in rows])
    positive = sum(row["validation_delta"] > 0 for row in rows)
    if mean_delta >= GAIN_THRESHOLD and positive >= 2:
        return "gain"
    if mean_delta <= 0 and positive <= 1:
        return "no_gain"
    return "borderline"


def analyze(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {
        (candidate["candidate_fingerprint"], variant, replicate)
        for candidate in manifest["new_combinations"]
        for variant in VARIANTS
        for replicate in range(REPLICATES)
    }
    index = {
        (row["candidate_fingerprint"], row["variant"], int(row["replicate_id"])): row
        for row in rows
    }
    if set(index) != expected or len(rows) != EXPECTED_NEW_TRAININGS:
        raise CrossoverError("new training matrix mismatch")
    new_summaries = {}
    failures = []
    for candidate in manifest["new_combinations"]:
        paired = []
        fingerprint = candidate["candidate_fingerprint"]
        for replicate in range(REPLICATES):
            baseline = index[(fingerprint, "baseline", replicate)]
            normalized = index[(fingerprint, "layer_norm", replicate)]
            expected_seed = b0.matched_training_seed(fingerprint, replicate)
            if (
                int(baseline["training_seed"]) != expected_seed
                or int(normalized["training_seed"]) != expected_seed
            ):
                raise CrossoverError("paired training seed mismatch")
            item = {
                "replicate_id": replicate,
                "training_seed": expected_seed,
                "baseline_validation": baseline["validation_accuracy"],
                "layernorm_validation": normalized["validation_accuracy"],
                "validation_delta": (
                    normalized["validation_accuracy"] - baseline["validation_accuracy"]
                ),
                "baseline_test": baseline["test_accuracy"],
                "layernorm_test": normalized["test_accuracy"],
                "test_delta": normalized["test_accuracy"] - baseline["test_accuracy"],
                "baseline_best_epoch": baseline["best_epoch"],
                "layernorm_best_epoch": normalized["best_epoch"],
                "baseline_runtime_seconds": baseline["runtime_seconds"],
                "layernorm_runtime_seconds": normalized["runtime_seconds"],
                "baseline_parameter_count": baseline["parameter_count"],
                "layernorm_parameter_count": normalized["parameter_count"],
                "valid": bool(baseline["valid"] and normalized["valid"]),
                "failures": [baseline.get("failure"), normalized.get("failure")],
            }
            if not item["valid"] or any(item["failures"]):
                failures.append(item)
            paired.append(item)
        state = _state(paired)
        new_summaries[candidate["panel_role"]] = {
            "architecture_source_role": candidate["architecture_source_role"],
            "hp_source_role": candidate["hp_source_role"],
            "candidate_fingerprint": fingerprint,
            "validation_state": state,
            "baseline_validation_mean": _mean([row["baseline_validation"] for row in paired]),
            "layernorm_validation_mean": _mean([row["layernorm_validation"] for row in paired]),
            "validation_delta_mean": _mean([row["validation_delta"] for row in paired]),
            "validation_positive_seeds": sum(row["validation_delta"] > 0 for row in paired),
            "baseline_test_mean": _mean([row["baseline_test"] for row in paired]),
            "layernorm_test_mean": _mean([row["layernorm_test"] for row in paired]),
            "test_delta_mean": _mean([row["test_delta"] for row in paired]),
            "test_positive_seeds": sum(row["test_delta"] > 0 for row in paired),
            "per_seed": paired,
        }
    median_top = new_summaries["median_arch_top_hp"]["validation_state"]
    top_median = new_summaries["top_arch_median_hp"]["validation_state"]
    if failures:
        verdict = "INCONCLUSIVE"
    elif median_top == "gain" and top_median == "gain":
        verdict = "BROADER_GAIN"
    elif median_top == "gain" and top_median == "no_gain":
        verdict = "ARCHITECTURE_EFFECT"
    elif median_top == "no_gain" and top_median == "gain":
        verdict = "HP_EFFECT"
    elif median_top == "no_gain" and top_median == "no_gain":
        verdict = "JOINT_EFFECT"
    else:
        verdict = "INCONCLUSIVE"
    next_step = {
        "ARCHITECTURE_EFFECT": "TEST_SIMILAR_ARCHITECTURES",
        "HP_EFFECT": "TEST_LAYERNORM_HP_INTERACTION",
        "JOINT_EFFECT": "MOVE_TO_PERFORMANCE_AWARE_REPRESENTATION",
        "BROADER_GAIN": "EXPAND_LAYERNORM",
        "INCONCLUSIVE": "MORE_EVIDENCE",
    }[verdict]
    return {
        "verdict": verdict,
        "recommended_next_step": next_step,
        "primary_endpoint": "validation matched delta",
        "failed_pairs": len(failures),
        "existing_combinations": manifest["existing_combinations"],
        "new_combinations": new_summaries,
        "new_training_rows": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument(
        "--output", default="results/posthoc/cora-layernorm-crossover-v1"
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = (REPO_ROOT / args.output).resolve()
    result_path = output / "crossover_results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite crossover result: {result_path}")
    manifest, candidates = prepare_manifest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "crossover_manifest.json"
    if manifest_path.exists():
        if stage_a.load_json(manifest_path) != manifest:
            raise CrossoverError("existing crossover manifest differs from current design")
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
        raise RuntimeError("LayerNorm crossover requires CUDA; CPU fallback is prohibited")
    resource_gate["training_started"] = True
    stage_a.dump_json(output / "resource_gate.json", resource_gate)
    device = torch.device("cuda:0")
    data, in_ch, out_ch = b0._load_cora((REPO_ROOT / args.cora_root).resolve(), device)
    progress_path = output / "crossover_progress.json"
    rows = stage_a.load_json(progress_path) if progress_path.exists() else []
    completed = {
        (row["candidate_fingerprint"], row["variant"], int(row["replicate_id"]))
        for row in rows
    }
    for candidate in candidates:
        for variant in VARIANTS:
            for replicate in range(REPLICATES):
                key = (candidate["candidate_fingerprint"], variant, replicate)
                if key in completed:
                    continue
                row = _train(candidate, variant, replicate, data, in_ch, out_ch, device)
                rows.append(row)
                stage_a.dump_json(progress_path, rows)
                print(
                    f"{candidate['panel_role']} {variant} seed{replicate}: "
                    f"val={row['validation_accuracy']:.3f} test={row['test_accuracy']:.3f}"
                )
    result = analyze(manifest, rows)
    stage_a.dump_json(result_path, result)
    stage_a.dump_json(
        output / "training_cost.json",
        {
            "reused_trainings": 12,
            "new_trainings": len(rows),
            "gpu_seconds": sum(float(row["runtime_seconds"]) for row in rows),
        },
    )
    print(f"verdict={result['verdict']}")
    print(f"recommended_next_step={result['recommended_next_step']}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
