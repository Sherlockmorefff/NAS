"""Final low-cost matched-seed Cora LayerNorm component gate."""

from __future__ import annotations

import argparse
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
from eval_utils import DynamicGNN, train_and_eval_arch


BASELINE_ROOT = REPO_ROOT / "results/posthoc/cora-accuracy-decoder-b0-v1/track_b"
FREEZE_COMMIT = "3e0abe90edba516c0152cbd9f0bf2ca41addfe7f"
FROZEN_EVALUATOR_SHA256 = "c8fae22ae14fe3ad5a856a70eac38f877422f18811ff02743e0501ea2ca3a5fa"
REPLICATES = 3
EXPECTED_TRAININGS = 9
STABLE_ARCH_DELTA = 0.004


class LayerNormScreenError(RuntimeError):
    """Raised when LayerNorm provenance or matrix invariants fail."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _concise(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "panel_role", "candidate_fingerprint", "architecture_fingerprint",
        "variant", "replicate_id", "training_seed", "validation_accuracy",
        "test_accuracy", "valid", "best_epoch", "stopped_epoch", "epochs_ran",
        "runtime_seconds", "parameter_count", "trajectory_hash",
        "model_final_state_hash", "failure",
    )
    return {field: row.get(field) for field in fields}


def _load_baseline() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = stage_a.load_json(BASELINE_ROOT / "b0_manifest.json")
    progress = stage_a.load_json(BASELINE_ROOT / "screen_progress.json")
    if manifest["dataset_identity"]["split_protocol"] != "planetoid_public":
        raise LayerNormScreenError("unexpected Cora split protocol")
    if manifest.get("max_epochs") != b0.MAX_EPOCHS or manifest.get("patience") != b0.PATIENCE:
        raise LayerNormScreenError("B0 epoch or patience setting changed")
    if manifest.get("replicates") != REPLICATES or not manifest.get("seed_excludes_variant"):
        raise LayerNormScreenError("B0 seed contract changed")
    frozen = subprocess.run(
        ["git", "show", f"{FREEZE_COMMIT}:eval_utils.py"], cwd=REPO_ROOT,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if hashlib.sha256(frozen).hexdigest() != FROZEN_EVALUATOR_SHA256:
        raise LayerNormScreenError("freeze commit does not reproduce concat-frozen evaluator")
    baseline = [row for row in progress if row.get("variant") == "baseline"]
    if len(baseline) != 9:
        raise LayerNormScreenError(f"expected 9 baseline rows, got {len(baseline)}")
    expected = {
        (candidate["candidate_fingerprint"], replicate)
        for candidate in manifest["candidate_panel"]
        for replicate in range(REPLICATES)
    }
    observed = {(row["candidate_fingerprint"], int(row["replicate_id"])) for row in baseline}
    if observed != expected:
        raise LayerNormScreenError("baseline matrix mismatch")
    for row in baseline:
        expected_seed = b0.matched_training_seed(
            row["candidate_fingerprint"], int(row["replicate_id"])
        )
        if int(row["training_seed"]) != expected_seed or not bool(row.get("valid")):
            raise LayerNormScreenError("baseline seed or validity mismatch")
    return manifest, baseline


def _parameter_count(candidate: Mapping[str, Any], variant: str) -> int:
    config = {
        "operations": list(candidate["operations"]),
        "edges": [tuple(edge) for edge in candidate["edges"]],
        "effective_layers": sum(op != "Identity" for op in candidate["operations"]),
    }
    model = DynamicGNN(
        config, in_ch=1433, out_ch=7, dropout=float(candidate["dropout"]),
        hidden_dim=int(candidate["hidden_dim"]),
        gat_heads=int(candidate.get("gat_heads") or 1),
        sage_aggr=str(candidate.get("sage_aggr") or "mean"),
        gin_eps=float(candidate.get("gin_eps") or 0.0),
        gat_heads_by_layer=candidate.get("gat_heads_by_layer"),
        sage_aggr_by_layer=candidate.get("sage_aggr_by_layer"),
        gin_eps_by_layer=candidate.get("gin_eps_by_layer"),
        component_variant=variant, activation_variant="relu",
    )
    return sum(parameter.numel() for parameter in model.parameters())


def prepare_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    b0_manifest, baseline = _load_baseline()
    costs = {}
    for candidate in b0_manifest["candidate_panel"]:
        baseline_params = _parameter_count(candidate, "baseline")
        layernorm_params = _parameter_count(candidate, "layer_norm")
        costs[candidate["panel_role"]] = {
            "baseline_parameter_count": baseline_params,
            "layernorm_parameter_count": layernorm_params,
            "added_parameters": layernorm_params - baseline_params,
            "parameter_ratio": layernorm_params / baseline_params,
        }
    manifest = {
        "format_version": 1,
        "experiment": "Cora LayerNorm component gate",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "research_question": "Does the absence of normalization limit Cora accuracy?",
        "baseline": "no normalization, reused from frozen B0",
        "new_variant": "layer_norm",
        "placement": "message-passing op output -> LayerNorm -> ReLU -> dropout",
        "special_operation_rule": {
            "Identity": "bypasses LayerNorm because it is not a message-passing op",
            "GCNII": "uses the same LayerNorm(hidden_dim) placement as other message-passing ops",
        },
        "excluded_changes": [
            "BatchNorm", "GraphNorm", "PairNorm", "residual", "fusion",
            "activation", "MLP",
        ],
        "candidate_panel": b0_manifest["candidate_panel"],
        "replicates": REPLICATES,
        "new_training_count": EXPECTED_TRAININGS,
        "baseline_retraining_count": 0,
        "max_epochs": b0.MAX_EPOCHS,
        "patience": b0.PATIENCE,
        "optimizer": "Adam, unchanged train_and_eval_arch implementation",
        "scheduler": "CosineAnnealingLR, unchanged train_and_eval_arch implementation",
        "matched_seed_scheme": b0_manifest["seed_scheme"],
        "seed_excludes_variant": True,
        "candidate_training_seeds": {
            candidate["panel_role"]: [
                b0.matched_training_seed(candidate["candidate_fingerprint"], replicate)
                for replicate in range(REPLICATES)
            ]
            for candidate in b0_manifest["candidate_panel"]
        },
        "dataset_identity": b0_manifest["dataset_identity"],
        "expected_parameter_cost": costs,
        "baseline_provenance": {
            "root": str(BASELINE_ROOT.relative_to(REPO_ROOT)),
            "manifest_sha256": stage_a.sha256_file(BASELINE_ROOT / "b0_manifest.json"),
            "progress_sha256": stage_a.sha256_file(BASELINE_ROOT / "screen_progress.json"),
            "baseline_rows": [_concise(row) for row in baseline],
        },
        "source_identity": {
            "layernorm_screen_sha256": stage_a.sha256_file(Path(__file__).resolve()),
            "evaluator_sha256": stage_a.sha256_file(REPO_ROOT / "eval_utils.py"),
            "concat_freeze_commit": FREEZE_COMMIT,
            "concat_frozen_evaluator_sha256": FROZEN_EVALUATOR_SHA256,
        },
        "decision_rule": {
            "stable_architecture": (
                "mean validation and test delta >=0.004, with >=2/3 positive matched seeds "
                "on both endpoints"
            ),
            "UNIVERSAL_GAIN": "3/3 stable architectures",
            "ARCHITECTURE_SPECIFIC": "1-2/3 stable architectures",
            "INCONCLUSIVE": (
                "0 stable architectures, but positive overall validation/test means with "
                ">=4/9 positive matched pairs on both endpoints"
            ),
            "NO_GAIN": "otherwise, or any invalid/failed run",
        },
    }
    return manifest, baseline


def _train(
    candidate: Mapping[str, Any], replicate: int, data,
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
            return_reproducibility_metadata=True, component_variant="layer_norm",
            activation_variant="relu", return_trajectory=True,
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
        "variant": "layer_norm", "replicate_id": int(replicate),
        "training_seed": int(seed), "validation_accuracy": float(validation),
        "test_accuracy": float(test), "valid": bool(valid),
        "runtime_seconds": float(time.monotonic() - started), **metadata,
    }


def _stable(rows: Sequence[Mapping[str, Any]]) -> bool:
    return (
        _mean([row["validation_delta"] for row in rows]) >= STABLE_ARCH_DELTA
        and _mean([row["test_delta"] for row in rows]) >= STABLE_ARCH_DELTA
        and sum(row["validation_delta"] > 0 for row in rows) >= 2
        and sum(row["test_delta"] > 0 for row in rows) >= 2
    )


def analyze(
    manifest: Mapping[str, Any], baseline: Sequence[Mapping[str, Any]],
    layernorm: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_index = {
        (row["candidate_fingerprint"], int(row["replicate_id"])): row
        for row in baseline
    }
    layernorm_index = {
        (row["candidate_fingerprint"], int(row["replicate_id"])): row
        for row in layernorm
    }
    if set(layernorm_index) != set(baseline_index):
        raise LayerNormScreenError("LayerNorm matrix differs from baseline")
    by_architecture = {}
    paired_all = []
    for candidate in manifest["candidate_panel"]:
        fingerprint = candidate["candidate_fingerprint"]
        paired = []
        for replicate in range(REPLICATES):
            key = (fingerprint, replicate)
            base, row = baseline_index[key], layernorm_index[key]
            if int(row["training_seed"]) != int(base["training_seed"]):
                raise LayerNormScreenError("LayerNorm seed differs from baseline")
            item = {
                "replicate_id": replicate, "training_seed": row["training_seed"],
                "baseline_validation": base["validation_accuracy"],
                "layernorm_validation": row["validation_accuracy"],
                "validation_delta": row["validation_accuracy"] - base["validation_accuracy"],
                "baseline_test": base["test_accuracy"],
                "layernorm_test": row["test_accuracy"],
                "test_delta": row["test_accuracy"] - base["test_accuracy"],
                "baseline_parameter_count": base["parameter_count"],
                "layernorm_parameter_count": row["parameter_count"],
                "parameter_ratio": row["parameter_count"] / base["parameter_count"],
                "baseline_runtime_seconds": base["runtime_seconds"],
                "layernorm_runtime_seconds": row["runtime_seconds"],
                "baseline_best_epoch": base["best_epoch"],
                "layernorm_best_epoch": row["best_epoch"],
                "valid": bool(row["valid"],), "failure": row.get("failure"),
            }
            paired.append(item)
            paired_all.append(item)
        by_architecture[candidate["panel_role"]] = {
            "candidate_fingerprint": fingerprint,
            "baseline_validation_mean": _mean([row["baseline_validation"] for row in paired]),
            "layernorm_validation_mean": _mean([row["layernorm_validation"] for row in paired]),
            "validation_delta_mean": _mean([row["validation_delta"] for row in paired]),
            "baseline_test_mean": _mean([row["baseline_test"] for row in paired]),
            "layernorm_test_mean": _mean([row["layernorm_test"] for row in paired]),
            "test_delta_mean": _mean([row["test_delta"] for row in paired]),
            "validation_positive_seeds": sum(row["validation_delta"] > 0 for row in paired),
            "test_positive_seeds": sum(row["test_delta"] > 0 for row in paired),
            "stable_positive": _stable(paired),
            "baseline_parameter_count": paired[0]["baseline_parameter_count"],
            "layernorm_parameter_count": paired[0]["layernorm_parameter_count"],
            "parameter_ratio": paired[0]["parameter_ratio"],
            "baseline_runtime_mean": _mean([row["baseline_runtime_seconds"] for row in paired]),
            "layernorm_runtime_mean": _mean([row["layernorm_runtime_seconds"] for row in paired]),
            "per_seed": paired,
        }

    failures = [row for row in paired_all if not row["valid"] or row.get("failure") is not None]
    stable_count = sum(row["stable_positive"] for row in by_architecture.values())
    validation_mean = _mean([row["validation_delta"] for row in paired_all])
    test_mean = _mean([row["test_delta"] for row in paired_all])
    validation_positive = sum(row["validation_delta"] > 0 for row in paired_all)
    test_positive = sum(row["test_delta"] > 0 for row in paired_all)
    if failures:
        classification = "NO_GAIN"
    elif stable_count == 3:
        classification = "UNIVERSAL_GAIN"
    elif stable_count in (1, 2):
        classification = "ARCHITECTURE_SPECIFIC"
    elif (
        validation_mean > 0 and test_mean > 0
        and validation_positive >= 4 and test_positive >= 4
    ):
        classification = "INCONCLUSIVE"
    else:
        classification = "NO_GAIN"
    next_step = {
        "UNIVERSAL_GAIN": "EXPAND_LAYERNORM",
        "ARCHITECTURE_SPECIFIC": "TEST_LAYERNORM_INTERACTION",
        "INCONCLUSIVE": "MORE_LAYERNORM_EVIDENCE",
        "NO_GAIN": "MOVE_TO_PERFORMANCE_AWARE_REPRESENTATION",
    }[classification]
    return {
        "classification": classification,
        "recommended_next_step": next_step,
        "overall_validation_delta_mean": validation_mean,
        "overall_test_delta_mean": test_mean,
        "seed_consistency": {
            "validation": f"{validation_positive}/9 positive matched pairs",
            "test": f"{test_positive}/9 positive matched pairs",
        },
        "architecture_consistency": f"{stable_count}/3 stable positive architectures",
        "failed_runs": len(failures),
        "by_architecture": by_architecture,
        "layernorm_runs": [_concise(row) for row in layernorm],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cora-root", default="Cora")
    parser.add_argument("--output", default="results/posthoc/cora-layernorm-v1")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = (REPO_ROOT / args.output).resolve()
    result_path = output / "layernorm_results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed LayerNorm screen: {result_path}")
    manifest, baseline = prepare_manifest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "layernorm_manifest.json"
    if manifest_path.exists():
        if stage_a.load_json(manifest_path) != manifest:
            raise LayerNormScreenError("existing LayerNorm manifest differs from current design")
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
        raise RuntimeError("LayerNorm screen requires CUDA; CPU fallback is prohibited")

    resource_gate["training_started"] = True
    stage_a.dump_json(output / "resource_gate.json", resource_gate)
    device = torch.device("cuda:0")
    data, in_ch, out_ch = b0._load_cora((REPO_ROOT / args.cora_root).resolve(), device)
    progress_path = output / "screen_progress.json"
    rows = stage_a.load_json(progress_path) if progress_path.exists() else []
    completed = {
        (row["candidate_fingerprint"], int(row["replicate_id"])) for row in rows
    }
    for candidate in manifest["candidate_panel"]:
        for replicate in range(REPLICATES):
            key = (candidate["candidate_fingerprint"], replicate)
            if key in completed:
                continue
            row = _train(candidate, replicate, data, in_ch, out_ch, device)
            rows.append(row)
            stage_a.dump_json(progress_path, rows)
            print(
                f"{candidate['panel_role']} layer_norm seed{replicate}: "
                f"val={row['validation_accuracy']:.3f} test={row['test_accuracy']:.3f}"
            )
    if len(rows) != EXPECTED_TRAININGS:
        raise LayerNormScreenError(f"expected {EXPECTED_TRAININGS} rows, got {len(rows)}")
    result = analyze(manifest, baseline, rows)
    stage_a.dump_json(result_path, result)
    stage_a.dump_json(
        output / "training_cost.json",
        {
            "baseline_reused_runs": len(baseline),
            "new_trainings": len(rows),
            "gpu_seconds": sum(float(row["runtime_seconds"]) for row in rows),
        },
    )
    print(f"classification={result['classification']}")
    print(f"recommended_next_step={result['recommended_next_step']}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
