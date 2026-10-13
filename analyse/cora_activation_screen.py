"""Minimal matched-seed Cora activation screen.

The screen reuses the nine frozen ReLU baseline runs from B0 and trains only
LeakyReLU and ELU on the same three architectures and matched seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from eval_utils import ACTIVATION_VARIANTS, train_and_eval_arch


BASELINE_ROOT = REPO_ROOT / "results/posthoc/cora-accuracy-decoder-b0-v1/track_b"
FREEZE_COMMIT = "6bd9a5b4f126de1ed51f211ea23482aae2028cd2"
FROZEN_EVALUATOR_SHA256 = "a829a14f33bc726ec430178c6799dea1ee47d293412c4a4b07a12bd78a6e5736"
NEW_ACTIVATIONS = ("leaky_relu", "elu")
REPLICATES = 3
EXPECTED_NEW_TRAININGS = 18
MEANINGFUL_DELTA = 0.005
STABLE_ARCH_DELTA = 0.004


class ActivationScreenError(RuntimeError):
    """Raised when provenance or screen invariants fail."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _concise_run(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "panel_role", "candidate_fingerprint", "architecture_fingerprint",
        "activation", "replicate_id", "training_seed", "validation_accuracy",
        "test_accuracy", "valid", "best_epoch", "stopped_epoch", "epochs_ran",
        "runtime_seconds", "parameter_count", "trajectory_hash",
        "model_final_state_hash", "failure",
    )
    return {field: row.get(field) for field in fields}


def _load_and_validate_baseline() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = BASELINE_ROOT / "b0_manifest.json"
    progress_path = BASELINE_ROOT / "screen_progress.json"
    manifest = stage_a.load_json(manifest_path)
    progress = stage_a.load_json(progress_path)
    if manifest["source_identity"]["evaluator_sha256"] != FROZEN_EVALUATOR_SHA256:
        raise ActivationScreenError("B0 evaluator fingerprint changed")
    frozen_source = subprocess.run(
        ["git", "show", f"{FREEZE_COMMIT}:eval_utils.py"], cwd=REPO_ROOT,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if hashlib.sha256(frozen_source).hexdigest() != FROZEN_EVALUATOR_SHA256:
        raise ActivationScreenError("freeze commit does not reproduce the B0 evaluator")
    if manifest.get("max_epochs") != b0.MAX_EPOCHS or manifest.get("patience") != b0.PATIENCE:
        raise ActivationScreenError("B0 epoch or patience setting changed")
    if manifest.get("replicates") != REPLICATES or not manifest.get("seed_excludes_variant"):
        raise ActivationScreenError("B0 matched-seed contract changed")
    if manifest["dataset_identity"]["split_protocol"] != "planetoid_public":
        raise ActivationScreenError("unexpected B0 split protocol")

    baseline = [dict(row, activation="relu") for row in progress if row.get("variant") == "baseline"]
    if len(baseline) != 9:
        raise ActivationScreenError(f"expected 9 reusable ReLU rows, got {len(baseline)}")
    expected = {
        (candidate["candidate_fingerprint"], replicate)
        for candidate in manifest["candidate_panel"]
        for replicate in range(REPLICATES)
    }
    observed = {
        (row["candidate_fingerprint"], int(row["replicate_id"])) for row in baseline
    }
    if observed != expected:
        raise ActivationScreenError("ReLU baseline matrix mismatch")
    roles = {candidate["candidate_fingerprint"]: candidate["panel_role"] for candidate in manifest["candidate_panel"]}
    for row in baseline:
        fingerprint = row["candidate_fingerprint"]
        replicate = int(row["replicate_id"])
        if row["training_seed"] != b0.matched_training_seed(fingerprint, replicate):
            raise ActivationScreenError("ReLU baseline matched seed mismatch")
        if row["panel_role"] != roles[fingerprint] or not bool(row.get("valid")):
            raise ActivationScreenError("ReLU baseline identity or validity mismatch")
        if row.get("failure") is not None:
            raise ActivationScreenError("ReLU baseline contains a recorded failure")
    return manifest, baseline


def prepare_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_manifest, baseline = _load_and_validate_baseline()
    candidate_seeds = {
        candidate["panel_role"]: [
            b0.matched_training_seed(candidate["candidate_fingerprint"], replicate)
            for replicate in range(REPLICATES)
        ]
        for candidate in baseline_manifest["candidate_panel"]
    }
    manifest = {
        "format_version": 1,
        "experiment": "Cora activation screen",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "activation_choice": {
            "selected": "leaky_relu",
            "alternative_not_run": "prelu",
            "reason": (
                "LeakyReLU provides a non-zero negative branch without learnable parameters; "
                "it keeps parameter count equal to ReLU and ELU and isolates activation shape."
            ),
            "leaky_relu_negative_slope": 0.01,
            "elu_alpha": 1.0,
        },
        "activations": list(ACTIVATION_VARIANTS),
        "new_activations": list(NEW_ACTIVATIONS),
        "relu_baseline_reused": True,
        "relu_retraining_count": 0,
        "candidate_panel": baseline_manifest["candidate_panel"],
        "replicates": REPLICATES,
        "new_training_count": EXPECTED_NEW_TRAININGS,
        "max_epochs": b0.MAX_EPOCHS,
        "patience": b0.PATIENCE,
        "optimizer": "Adam, unchanged train_and_eval_arch implementation",
        "scheduler": "CosineAnnealingLR, unchanged train_and_eval_arch implementation",
        "matched_seed_scheme": baseline_manifest["seed_scheme"],
        "seed_excludes_activation": True,
        "candidate_training_seeds": candidate_seeds,
        "dataset_identity": baseline_manifest["dataset_identity"],
        "baseline_provenance": {
            "root": str(BASELINE_ROOT.relative_to(REPO_ROOT)),
            "b0_manifest_sha256": stage_a.sha256_file(BASELINE_ROOT / "b0_manifest.json"),
            "screen_progress_sha256": stage_a.sha256_file(BASELINE_ROOT / "screen_progress.json"),
            "frozen_evaluator_sha256": FROZEN_EVALUATOR_SHA256,
            "freeze_commit": FREEZE_COMMIT,
            "baseline_rows": [_concise_run(row) for row in baseline],
        },
        "source_identity": {
            "activation_screen_sha256": stage_a.sha256_file(Path(__file__).resolve()),
            "current_evaluator_sha256": stage_a.sha256_file(REPO_ROOT / "eval_utils.py"),
        },
        "decision_rule": {
            "meaningful_mean_delta": MEANINGFUL_DELTA,
            "stable_architecture_delta": STABLE_ARCH_DELTA,
            "UNIVERSAL_SIGNAL": (
                "all 3 architectures stable in validation and test, >=2/3 positive matched seeds "
                "per endpoint, and overall mean validation/test delta >=0.005"
            ),
            "ARCHITECTURE_SPECIFIC": "exactly 1-2 architectures meet the stable rule",
            "BORDERLINE": "no stable architecture, but overall validation/test means are positive with >=4/9 positive pairs",
            "DROP": "otherwise, or any invalid/failed run",
        },
    }
    return manifest, baseline


def _train_candidate(
    candidate: Mapping[str, Any], activation: str, replicate: int,
    data, in_ch: int, out_ch: int, device: torch.device,
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
            return_reproducibility_metadata=True, component_variant="baseline",
            activation_variant=activation, return_trajectory=True,
        )
    except InfrastructureEvaluationError as exc:
        validation, valid, test = 0.0, False, 0.0
        metadata = {
            "best_epoch": None, "stopped_epoch": None, "epochs_ran": 0,
            "trajectory": [], "trajectory_hash": None,
            "model_final_state_hash": None, "parameter_count": None,
            "failure": {"type": type(exc).__name__, "message": str(exc), "context": exc.context},
        }
    return {
        "panel_role": candidate["panel_role"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "architecture_fingerprint": candidate["architecture_fingerprint"],
        "activation": activation,
        "replicate_id": int(replicate),
        "training_seed": int(seed),
        "validation_accuracy": float(validation),
        "test_accuracy": float(test),
        "valid": bool(valid),
        "runtime_seconds": float(time.monotonic() - started),
        **metadata,
    }


def _stable_architecture(seed_rows: Sequence[Mapping[str, Any]]) -> bool:
    return (
        _mean([row["validation_delta"] for row in seed_rows]) >= STABLE_ARCH_DELTA
        and _mean([row["test_delta"] for row in seed_rows]) >= STABLE_ARCH_DELTA
        and sum(row["validation_delta"] > 0 for row in seed_rows) >= 2
        and sum(row["test_delta"] > 0 for row in seed_rows) >= 2
    )


def analyze(
    manifest: Mapping[str, Any], baseline: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_index = {
        (row["candidate_fingerprint"], int(row["replicate_id"])): row for row in baseline
    }
    new_index = {
        (row["candidate_fingerprint"], int(row["replicate_id"]), row["activation"]): row
        for row in new_rows
    }
    expected = {
        (candidate["candidate_fingerprint"], replicate, activation)
        for candidate in manifest["candidate_panel"]
        for activation in NEW_ACTIVATIONS
        for replicate in range(REPLICATES)
    }
    if set(new_index) != expected:
        raise ActivationScreenError("new activation result matrix mismatch")

    activation_results = {}
    all_runs = [_concise_run(row) for row in baseline]
    for activation in NEW_ACTIVATIONS:
        paired = []
        by_architecture = {}
        for candidate in manifest["candidate_panel"]:
            fingerprint = candidate["candidate_fingerprint"]
            seed_rows = []
            for replicate in range(REPLICATES):
                row = new_index[(fingerprint, replicate, activation)]
                base = baseline_index[(fingerprint, replicate)]
                if int(row["training_seed"]) != int(base["training_seed"]):
                    raise ActivationScreenError("activation seed differs from ReLU baseline")
                parameter_delta = (
                    int(row["parameter_count"]) - int(base["parameter_count"])
                    if row.get("parameter_count") is not None else None
                )
                item = {
                    "replicate_id": replicate,
                    "training_seed": row["training_seed"],
                    "relu_validation": base["validation_accuracy"],
                    "activation_validation": row["validation_accuracy"],
                    "validation_delta": row["validation_accuracy"] - base["validation_accuracy"],
                    "relu_test": base["test_accuracy"],
                    "activation_test": row["test_accuracy"],
                    "test_delta": row["test_accuracy"] - base["test_accuracy"],
                    "relu_best_epoch": base["best_epoch"],
                    "activation_best_epoch": row["best_epoch"],
                    "relu_runtime_seconds": base["runtime_seconds"],
                    "activation_runtime_seconds": row["runtime_seconds"],
                    "relu_parameter_count": base["parameter_count"],
                    "activation_parameter_count": row["parameter_count"],
                    "parameter_delta": parameter_delta,
                    "valid": bool(row["valid"]),
                    "failure": row.get("failure"),
                }
                seed_rows.append(item)
                paired.append(item)
                all_runs.append(_concise_run(row))
            by_architecture[candidate["panel_role"]] = {
                "candidate_fingerprint": fingerprint,
                "relu_validation_mean": _mean([row["relu_validation"] for row in seed_rows]),
                "activation_validation_mean": _mean([row["activation_validation"] for row in seed_rows]),
                "validation_delta_mean": _mean([row["validation_delta"] for row in seed_rows]),
                "relu_test_mean": _mean([row["relu_test"] for row in seed_rows]),
                "activation_test_mean": _mean([row["activation_test"] for row in seed_rows]),
                "test_delta_mean": _mean([row["test_delta"] for row in seed_rows]),
                "validation_positive_seeds": sum(row["validation_delta"] > 0 for row in seed_rows),
                "test_positive_seeds": sum(row["test_delta"] > 0 for row in seed_rows),
                "stable_positive": _stable_architecture(seed_rows),
                "per_seed": seed_rows,
            }

        failed = [row for row in paired if not row["valid"] or row.get("failure") is not None]
        parameter_mismatches = [row for row in paired if row["parameter_delta"] != 0]
        stable_count = sum(row["stable_positive"] for row in by_architecture.values())
        validation_mean = _mean([row["validation_delta"] for row in paired])
        test_mean = _mean([row["test_delta"] for row in paired])
        validation_positive = sum(row["validation_delta"] > 0 for row in paired)
        test_positive = sum(row["test_delta"] > 0 for row in paired)
        if failed or parameter_mismatches:
            classification = "DROP"
        elif (
            stable_count == 3 and validation_mean >= MEANINGFUL_DELTA
            and test_mean >= MEANINGFUL_DELTA
        ):
            classification = "UNIVERSAL_SIGNAL"
        elif stable_count in (1, 2):
            classification = "ARCHITECTURE_SPECIFIC"
        elif (
            validation_mean > 0 and test_mean > 0
            and validation_positive >= 4 and test_positive >= 4
        ):
            classification = "BORDERLINE"
        else:
            classification = "DROP"
        activation_results[activation] = {
            "validation_delta_mean": validation_mean,
            "test_delta_mean": test_mean,
            "seed_consistency": {
                "validation": f"{validation_positive}/9 positive matched pairs",
                "test": f"{test_positive}/9 positive matched pairs",
            },
            "architecture_consistency": f"{stable_count}/3 stable positive architectures",
            "classification": classification,
            "failed_runs": len(failed),
            "parameter_mismatches": len(parameter_mismatches),
            "by_architecture": by_architecture,
        }

    universal = [
        name for name, row in activation_results.items()
        if row["classification"] == "UNIVERSAL_SIGNAL"
    ]
    best_activation = (
        max(
            universal,
            key=lambda name: (
                activation_results[name]["validation_delta_mean"],
                activation_results[name]["test_delta_mean"],
            ),
        )
        if universal else None
    )
    if best_activation is not None:
        next_step = "EXPAND_ACTIVATION"
    elif any(row["classification"] == "BORDERLINE" for row in activation_results.values()):
        next_step = "MORE_ACTIVATION_EVIDENCE"
    else:
        next_step = "TEST_CONCAT_FUSION"
    return {
        "relu_baseline_reused": True,
        "new_training_count": len(new_rows),
        "activation_results": activation_results,
        "best_activation": best_activation,
        "recommended_next_step": next_step,
        "runs": all_runs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument("--output", default="results/posthoc/cora-activation-screen-v1")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = (REPO_ROOT / args.output).resolve()
    result_path = output / "activation_results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed activation screen: {result_path}")
    manifest, baseline = prepare_manifest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "activation_manifest.json"
    if manifest_path.exists():
        if stage_a.load_json(manifest_path) != manifest:
            raise ActivationScreenError("existing activation manifest differs from current design")
    else:
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
        raise RuntimeError("activation screen requires CUDA; CPU fallback is prohibited")

    resource_gate["training_started"] = True
    stage_a.dump_json(output / "resource_gate.json", resource_gate)
    device = torch.device("cuda:0")
    data, in_ch, out_ch = b0._load_cora((REPO_ROOT / args.cora_root).resolve(), device)
    progress_path = output / "screen_progress.json"
    rows = stage_a.load_json(progress_path) if progress_path.exists() else []
    completed = {
        (row["candidate_fingerprint"], row["activation"], int(row["replicate_id"]))
        for row in rows
    }
    for candidate in manifest["candidate_panel"]:
        for activation in NEW_ACTIVATIONS:
            for replicate in range(REPLICATES):
                key = (candidate["candidate_fingerprint"], activation, replicate)
                if key in completed:
                    continue
                row = _train_candidate(
                    candidate, activation, replicate, data, in_ch, out_ch, device
                )
                rows.append(row)
                stage_a.dump_json(progress_path, rows)
                print(
                    f"{candidate['panel_role']} {activation} seed{replicate}: "
                    f"val={row['validation_accuracy']:.3f} test={row['test_accuracy']:.3f}"
                )
    if len(rows) != EXPECTED_NEW_TRAININGS:
        raise ActivationScreenError(
            f"expected {EXPECTED_NEW_TRAININGS} new runs, got {len(rows)}"
        )
    result = analyze(manifest, baseline, rows)
    stage_a.dump_json(result_path, result)
    stage_a.dump_json(
        output / "training_cost.json",
        {
            "relu_reused_runs": len(baseline),
            "new_trainings": len(rows),
            "gpu_seconds": sum(float(row["runtime_seconds"]) for row in rows),
        },
    )
    print(f"best_activation={result['best_activation']}")
    print(f"recommended_next_step={result['recommended_next_step']}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
