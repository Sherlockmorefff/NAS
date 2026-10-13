"""Minimal, preregistered Cora B0 component screen.

Preparation and analysis are CPU-safe.  The baseline regression and 45-run
screen require CUDA and refuse to fall back to CPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import cora_accuracy_stage_a as stage_a
from evaluation_errors import InfrastructureEvaluationError
from eval_utils import COMPONENT_VARIANTS, stable_seed, train_and_eval_arch


BASE_SEED = 20260816
VARIANTS = tuple(COMPONENT_VARIANTS)
REPLICATES = 3
MAX_EPOCHS = 300
PATIENCE = 80
BASELINE_REFERENCE = (
    "legacy_artifacts/pre_20260813/results/"
    "final_eval_global4_prequential_v2_search_seed4/final_results_global4.json"
)


def _candidate_payload(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    fields = (
        "candidate_fingerprint", "architecture_fingerprint", "operations", "edges",
        "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
        "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer", "val_acc",
        "_method", "_run_seed", "_run_name", "full_evaluation_index",
    )
    return {"panel_role": role, **{field: row.get(field) for field in fields}}


def _static_truncation_stress(row: Mapping[str, Any], in_ch: int = 1433) -> dict[str, int]:
    operations = list(row["operations"])
    edges = [tuple(edge) for edge in row["edges"]]
    predecessors: dict[int, list[int]] = {}
    for source, destination in edges:
        predecessors.setdefault(int(destination), []).append(int(source))
    node_width = {0: int(in_ch)}
    mismatch_events = 0
    columns_discarded = 0
    for node in range(1, len(operations) + 2):
        preds = predecessors.get(node, [node - 1])
        widths = [node_width.get(pred, int(row["hidden_dim"])) for pred in preds]
        running = widths[0]
        for width in widths[1:]:
            if running != width:
                mismatch_events += 1
                columns_discarded += abs(running - width)
                running = min(running, width)
        if node <= len(operations):
            operation = operations[node - 1]
            node_width[node] = running if operation == "Identity" else int(row["hidden_dim"])
    return {
        "mismatch_events": mismatch_events,
        "static_columns_discarded": columns_discarded,
    }


def select_panel(deduplicated: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in deduplicated if bool(row.get("valid", True))]
    ranked = sorted(
        valid,
        key=lambda row: (-float(row["val_acc"]), row["candidate_fingerprint"]),
    )
    top = ranked[0]
    selected_architectures = {top["architecture_fingerprint"]}

    median_value = float(np.median([row["val_acc"] for row in valid]))
    median = min(
        (row for row in valid if row["architecture_fingerprint"] not in selected_architectures),
        key=lambda row: (
            abs(float(row["val_acc"]) - median_value),
            row["candidate_fingerprint"],
        ),
    )
    selected_architectures.add(median["architecture_fingerprint"])

    def stress_key(row: Mapping[str, Any]) -> tuple[float, ...]:
        stress = _static_truncation_stress(row)
        diversity = min(
            (
                stage_a.op_distance(row, reference)
                + stage_a.edge_distance(row, reference)
            ) / 2.0
            for reference in (top, median)
        )
        return (
            -float(stress["mismatch_events"]),
            -float(stress["static_columns_discarded"]),
            -diversity,
            -float(row["val_acc"]),
            row["candidate_fingerprint"],
        )

    stress_pool = [
        row for row in valid
        if row["architecture_fingerprint"] not in selected_architectures
        and float(row["val_acc"]) >= median_value
    ]
    stress = min(
        stress_pool,
        key=stress_key,
    )
    panel = [
        _candidate_payload(top, "top_1"),
        _candidate_payload(median, "median_1"),
        _candidate_payload(stress, "diverse_truncation_stress_1"),
    ]
    for candidate in panel:
        candidate["truncation_stress"] = _static_truncation_stress(candidate)
    if len({row["architecture_fingerprint"] for row in panel}) != 3:
        raise RuntimeError("candidate panel architecture fingerprints are not distinct")
    return panel


def _load_baseline_reference(path: Path) -> dict[str, Any]:
    rows = stage_a.load_json(path)
    candidate = rows[0]
    seed_row = next(row for row in candidate["per_seed"] if int(row["seed"]) == 0)
    return {
        "source": str(path.relative_to(REPO_ROOT)),
        "candidate_rank": int(candidate["candidate_rank"]),
        "seed": 0,
        "max_epochs": 300,
        "patience": 80,
        "config": {
            "operations": list(candidate["operations"]),
            "edges": [list(edge) for edge in candidate["edges"]],
            "effective_layers": sum(op != "Identity" for op in candidate["operations"]),
        },
        "hp": {
            key: candidate.get(key)
            for key in (
                "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
                "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer",
            )
        },
        "expected": {
            "validation_accuracy": float(seed_row["val_acc"]),
            "test_accuracy": float(seed_row["test_acc"]),
            "valid": bool(seed_row["valid"]),
            "trajectory_hash": seed_row.get("trajectory_hash"),
            "model_final_state_hash": seed_row.get("model_final_state_hash"),
        },
        "comparability_note": (
            "The legacy artifact records validation/test but predates trajectory and model-state "
            "hash fields; those hashes are captured for this run but cannot be compared retroactively."
        ),
    }


def prepare_manifest(artifact_root: Path, checkpoint: Path, cora_root: Path) -> dict[str, Any]:
    gate, full, _low = stage_a.integrity_gate(
        REPO_ROOT, artifact_root, checkpoint, cora_root, None
    )
    deduplicated, deduplication = stage_a.deduplicate_candidate_population(full)
    panel = select_panel(deduplicated)
    return {
        "format_version": 1,
        "B0_INDEPENDENT_OF_DECODER": True,
        "dependency_chain": (
            "frozen candidate manifest -> operations/edges + frozen HP -> "
            "train_and_eval_arch -> DynamicGNN"
        ),
        "excluded_dependencies": ["latent decode", "stochastic decoder", "GP", "BO", "LF promotion"],
        "integrity_status": gate["status"],
        "dataset_identity": gate["cora_dataset"],
        "deduplication": deduplication,
        "candidate_selection_endpoint": "historical validation only",
        "stress_candidate_floor": "historical validation >= population median",
        "candidate_panel": panel,
        "variants": list(VARIANTS),
        "variant_definitions": {
            "baseline": "unchanged DynamicGNN combine and classifier",
            "projected_combine": "learned bias-free projections align each predecessor to the first predecessor width before summation",
            "batch_norm": "BatchNorm1d after each non-identity graph operation and before ReLU/dropout",
            "projected_residual": "learned bias-free projection adds the layer input to each non-identity graph operation output",
            "post_mlp1": "one Linear+ReLU post-processing layer immediately before the classifier",
        },
        "prelu": "DEFERRED",
        "replicates": REPLICATES,
        "training_count": 45,
        "maximum_training_count_with_regression": 46,
        "seed_scheme": "sha256_v1(BASE_SEED, b0_matched_training, candidate_fingerprint, replicate_id)",
        "seed_excludes_variant": True,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "primary_endpoint": "test accuracy at best-validation epoch, paired against baseline",
        "selection_rule": "classification uses preregistered paired validation and test summaries; no variant is redefined after observing test",
        "baseline_reference": _load_baseline_reference(REPO_ROOT / BASELINE_REFERENCE),
        "source_identity": {
            "component_screen_sha256": stage_a.sha256_file(Path(__file__).resolve()),
            "evaluator_sha256": stage_a.sha256_file(REPO_ROOT / "eval_utils.py"),
            "baseline_reference_sha256": stage_a.sha256_file(REPO_ROOT / BASELINE_REFERENCE),
        },
    }


def _load_cora(root: Path, device: torch.device):
    pyg_root = root.parent if root.name == "Cora" else root
    dataset = Planetoid(
        root=str(pyg_root), name="Cora", split="public", transform=T.NormalizeFeatures()
    )
    return dataset[0].to(device), int(dataset.num_features), int(dataset.num_classes)


def _train(config: Mapping[str, Any], hp: Mapping[str, Any], data, in_ch, out_ch, device, seed, variant):
    started = time.monotonic()
    try:
        result = train_and_eval_arch(
            config=dict(config), data=data, in_ch=in_ch, out_ch=out_ch,
            lr=float(hp["lr"]), dropout=float(hp["dropout"]), hidden_dim=int(hp["hidden_dim"]),
            weight_decay=float(hp["l2"]), gat_heads=int(hp.get("gat_heads") or 1),
            sage_aggr=str(hp.get("sage_aggr") or "mean"), gin_eps=float(hp.get("gin_eps") or 0.0),
            gat_heads_by_layer=hp.get("gat_heads_by_layer"),
            sage_aggr_by_layer=hp.get("sage_aggr_by_layer"),
            gin_eps_by_layer=hp.get("gin_eps_by_layer"), device=device,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=int(seed), track_test=True,
            return_metadata=True, return_reproducibility_metadata=True,
            component_variant=variant, return_trajectory=True,
        )
    except InfrastructureEvaluationError as exc:
        return {
            "validation_accuracy": 0.0, "test_accuracy": 0.0, "valid": False,
            "runtime_seconds": time.monotonic() - started, "best_epoch": None,
            "epochs_ran": 0, "parameter_count": None, "trajectory": [],
            "trajectory_hash": None, "model_final_state_hash": None,
            "failure": {
                "type": type(exc).__name__, "message": str(exc),
                "context": exc.context,
            },
        }
    validation, valid, test, metadata = result
    return {
        "validation_accuracy": float(validation), "test_accuracy": float(test),
        "valid": bool(valid), "runtime_seconds": time.monotonic() - started,
        **metadata,
    }


def run_baseline_regression(manifest, data, in_ch, out_ch, device) -> dict[str, Any]:
    reference = manifest["baseline_reference"]
    actual = _train(
        reference["config"], reference["hp"], data, in_ch, out_ch, device,
        reference["seed"], "baseline",
    )
    expected = reference["expected"]
    checks = {
        "validation_accuracy": math.isclose(actual["validation_accuracy"], expected["validation_accuracy"], abs_tol=1e-12),
        "test_accuracy": math.isclose(actual["test_accuracy"], expected["test_accuracy"], abs_tol=1e-12),
        "valid": actual["valid"] == expected["valid"],
    }
    for field in ("trajectory_hash", "model_final_state_hash"):
        checks[field] = (
            actual.get(field) == expected[field] if expected.get(field) is not None else None
        )
    return {
        "status": "PASS" if all(value is not False for value in checks.values()) else "FAIL",
        "checks": checks, "expected": expected, "actual": actual,
        "comparability_note": reference["comparability_note"],
    }


def _screen_row(candidate, variant, replicate, data, in_ch, out_ch, device):
    seed = matched_training_seed(candidate["candidate_fingerprint"], replicate)
    config = {
        "operations": list(candidate["operations"]),
        "edges": [tuple(edge) for edge in candidate["edges"]],
        "effective_layers": sum(op != "Identity" for op in candidate["operations"]),
    }
    result = _train(config, candidate, data, in_ch, out_ch, device, seed, variant)
    return {
        "panel_role": candidate["panel_role"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "architecture_fingerprint": candidate["architecture_fingerprint"],
        "variant": variant, "replicate_id": replicate, "training_seed": seed,
        **result,
    }


def _mean(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def matched_training_seed(candidate_fingerprint: str, replicate_id: int) -> int:
    return stable_seed(
        BASE_SEED,
        "b0_matched_training",
        candidate_fingerprint,
        int(replicate_id),
    )


def analyze(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = {
        (row["candidate_fingerprint"], int(row["replicate_id"]), row["variant"]): row
        for row in rows
    }
    results = {}
    for variant in VARIANTS:
        paired = []
        for key, row in index.items():
            fingerprint, replicate, current_variant = key
            if current_variant != variant:
                continue
            baseline = index[(fingerprint, replicate, "baseline")]
            parameter_ratio = (
                row["parameter_count"] / baseline["parameter_count"]
                if row.get("parameter_count") is not None
                and baseline.get("parameter_count") not in (None, 0)
                else None
            )
            epoch_delta = (
                row["best_epoch"] - baseline["best_epoch"]
                if row.get("best_epoch") is not None
                and baseline.get("best_epoch") is not None
                else None
            )
            paired.append({
                "candidate_fingerprint": fingerprint, "replicate_id": replicate,
                "validation_delta": row["validation_accuracy"] - baseline["validation_accuracy"],
                "test_delta": row["test_accuracy"] - baseline["test_accuracy"],
                "runtime_ratio": row["runtime_seconds"] / baseline["runtime_seconds"],
                "parameter_ratio": parameter_ratio,
                "epoch_delta": epoch_delta,
                "valid": bool(row["valid"]),
                "failure": row.get("failure"),
            })
        by_candidate = {}
        for fingerprint in sorted({row["candidate_fingerprint"] for row in paired}):
            subset = [row for row in paired if row["candidate_fingerprint"] == fingerprint]
            by_candidate[fingerprint] = {
                "validation_delta": _mean([row["validation_delta"] for row in subset]),
                "test_delta": _mean([row["test_delta"] for row in subset]),
                "positive_test_seeds": sum(row["test_delta"] > 0 for row in subset),
                "valid_runs": sum(row["valid"] for row in subset),
            }
        test_deltas = [row["test_delta"] for row in paired]
        validation_deltas = [row["validation_delta"] for row in paired]
        candidate_positive = sum(value["test_delta"] > 0 for value in by_candidate.values())
        seed_positive = sum(value > 0 for value in test_deltas)
        validation_candidate_positive = sum(
            value["validation_delta"] > 0 for value in by_candidate.values()
        )
        validation_seed_positive = sum(value > 0 for value in validation_deltas)
        if variant == "baseline":
            classification = "REFERENCE"
        elif len(paired) != 9 or not all(row["valid"] for row in paired):
            classification = "DROP"
        elif (
            candidate_positive == 3 and seed_positive >= 6
            and validation_candidate_positive >= 2 and validation_seed_positive >= 6
            and _mean(test_deltas) >= 0.005 and _mean(validation_deltas) > 0
        ):
            classification = "STRONG"
        elif any(
            value["test_delta"] >= 0.01 and value["positive_test_seeds"] >= 2
            and value["validation_delta"] > 0
            for value in by_candidate.values()
        ):
            classification = "INTERACTION-SIGNAL"
        elif _mean(test_deltas) > 0 and _mean(validation_deltas) >= 0:
            classification = "BORDERLINE"
        else:
            classification = "DROP"
        results[variant] = {
            "validation_delta": _mean(validation_deltas),
            "test_delta": _mean(test_deltas),
            "candidate_consistency": {
                "validation": f"{validation_candidate_positive}/3 positive means",
                "test": f"{candidate_positive}/3 positive means",
            },
            "seed_consistency": {
                "validation": f"{validation_seed_positive}/9 positive pairs",
                "test": f"{seed_positive}/9 positive pairs",
            },
            "runtime_ratio": _mean([row["runtime_ratio"] for row in paired]),
            "parameter_ratio": _mean([row["parameter_ratio"] for row in paired]),
            "best_epoch_delta": _mean([row["epoch_delta"] for row in paired]),
            "classification": classification, "by_candidate": by_candidate,
            "paired_differences": paired,
        }
    priority = {"STRONG": 0, "INTERACTION-SIGNAL": 1, "BORDERLINE": 2}
    survivors = sorted(
        (
            name for name, result in results.items()
            if name != "baseline" and result["classification"] in priority
        ),
        key=lambda name: (
            priority[results[name]["classification"]],
            -results[name]["validation_delta"],
            -results[name]["test_delta"],
            name,
        ),
    )[:2]
    return {"variants": results, "survivors": survivors, "survivor_limit": 2}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cora minimal B0 component screen")
    parser.add_argument("--artifact-root", default="legacy_artifacts/pre_20260813/results")
    parser.add_argument("--checkpoint", default="checkpoints/joint_model_pipeline_global4_best.pth")
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument("--output", default="results/posthoc/cora-accuracy-decoder-b0-v1/track_b")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "b0_manifest.json"
    if manifest_path.exists():
        manifest = stage_a.load_json(manifest_path)
    else:
        manifest = prepare_manifest(
            (REPO_ROOT / args.artifact_root).resolve(),
            (REPO_ROOT / args.checkpoint).resolve(),
            (REPO_ROOT / args.cora_root).resolve(),
        )
        stage_a.dump_json(manifest_path, manifest)

    resource_gate = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "prepare_only": bool(args.prepare_only),
        "training_started": False,
    }
    if torch.cuda.is_available():
        resource_gate["device_name"] = torch.cuda.get_device_name(0)
    stage_a.dump_json(output / "resource_gate.json", resource_gate)
    if args.prepare_only:
        print(output)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("B0 requires CUDA; CPU fallback is prohibited")

    resource_gate["training_started"] = True
    stage_a.dump_json(output / "resource_gate.json", resource_gate)
    device = torch.device("cuda:0")
    data, in_ch, out_ch = _load_cora((REPO_ROOT / args.cora_root).resolve(), device)
    regression_path = output / "baseline_regression.json"
    regression = (
        stage_a.load_json(regression_path)
        if regression_path.exists()
        else run_baseline_regression(manifest, data, in_ch, out_ch, device)
    )
    if not regression_path.exists():
        stage_a.dump_json(regression_path, regression)
    if regression["status"] != "PASS":
        raise RuntimeError("baseline regression failed; Track B stopped")

    progress_path = output / "screen_progress.json"
    rows = stage_a.load_json(progress_path) if progress_path.exists() else []
    completed = {
        (row["candidate_fingerprint"], row["variant"], int(row["replicate_id"]))
        for row in rows
    }
    for candidate in manifest["candidate_panel"]:
        for variant in VARIANTS:
            for replicate in range(REPLICATES):
                key = (candidate["candidate_fingerprint"], variant, replicate)
                if key in completed:
                    continue
                rows.append(_screen_row(candidate, variant, replicate, data, in_ch, out_ch, device))
                stage_a.dump_json(progress_path, rows)
    if len(rows) != 45:
        raise RuntimeError(f"screen cardinality mismatch: expected 45, got {len(rows)}")
    result = analyze(rows)
    stage_a.dump_json(output / "b0_results.json", result)
    cost = {
        "baseline_regression_trainings": 1,
        "screen_trainings": len(rows),
        "total_trainings": 1 + len(rows),
        "gpu_seconds": regression["actual"]["runtime_seconds"] + sum(row["runtime_seconds"] for row in rows),
    }
    stage_a.dump_json(output / "training_cost.json", cost)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
