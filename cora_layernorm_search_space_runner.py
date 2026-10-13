"""Dry-run or launch the frozen Cora None/LayerNorm search-space pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import torch

import bo_phase4
from dataset_utils import evaluation_cache_key, legacy_cora_context
from eval_utils import isolated_rng, stable_seed
from layernorm_search_space import (
    NORMALIZATION_LAYER_NORM,
    NORMALIZATION_NONE,
    balanced_initialization_assignment,
    base_candidate_fingerprint,
    compose_search_vector,
    full_candidate_fingerprint,
    matched_training_seed,
    normalization_value,
)
from surrogate.accuracy_gp import AccuracyGPPredictor
from surrogate.checkpoint_io import atomic_json_dump


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/cora_layernorm_search_space_pilot.json"


class _QuietLogger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_historical_best(config: dict[str, Any]) -> dict[str, Any]:
    confirmation = config["confirmation"]
    manifest_path = ROOT / confirmation["historical_best_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported historical-best manifest format")
    if manifest.get("selection_metric") != "validation_accuracy":
        raise ValueError("historical best must be selected by validation accuracy")
    if manifest.get("selection_test_access") != "PROHIBITED":
        raise ValueError("historical-best selection must prohibit test access")
    source_path = ROOT / manifest["source_history"]
    if _sha256(source_path) != manifest["source_history_sha256"]:
        raise ValueError("historical-best source history hash mismatch")
    records = json.loads(source_path.read_text(encoding="utf-8"))
    index = int(manifest["source_history_index"])
    if index < 0 or index >= len(records):
        raise ValueError("historical-best source index is out of range")
    record = records[index]
    if any("test" in str(key).lower() for key in record):
        raise ValueError("historical-best source record contains a test field")
    expected = {
        "candidate_fingerprint": manifest["candidate_fingerprint"],
        "architecture_fingerprint": manifest["architecture_fingerprint"],
        "val_acc": manifest["historical_validation_accuracy"],
        "search_seed": manifest["source_search_seed"],
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"historical-best source record mismatch: {field}")
    valid = [
        row
        for row in records
        if row.get("valid") is True
        and row.get("evaluation_fidelity", "full") == "full"
        and row.get("val_acc") is not None
    ]
    if not valid or float(record["val_acc"]) != max(float(row["val_acc"]) for row in valid):
        raise ValueError("historical-best record is not best in its source history")
    return manifest


def build_confirmation_plan(config: dict[str, Any]) -> dict[str, Any]:
    confirmation = config["confirmation"]
    seed_manifest = json.loads(
        (ROOT / config["confirmation_seed_manifest"]).read_text(encoding="utf-8")
    )
    seeds = [int(seed) for seed in seed_manifest["seeds"]]
    historical = load_historical_best(config)
    slots = [
        {
            "candidate_id": f"{method}_search_seed{seed}_winner",
            "source": "nas_winner",
            "method": method,
            "search_seed": int(seed),
            "selection_metric": "validation_accuracy",
            "confirmation_seeds": seeds,
        }
        for method in ("original", "expanded")
        for seed in config["search_seeds"]
    ]
    slots.append(
        {
            "candidate_id": "historical_archive_best",
            "source": "historical_archive",
            "method": "archive_best",
            "search_seed": int(historical["source_search_seed"]),
            "candidate_fingerprint": historical["candidate_fingerprint"],
            "historical_validation_accuracy": float(
                historical["historical_validation_accuracy"]
            ),
            "selection_metric": "validation_accuracy",
            "confirmation_seeds": seeds,
        }
    )
    expected_candidates = int(confirmation["candidate_count"])
    expected_trainings = int(confirmation["training_count"])
    if len(slots) != expected_candidates:
        raise ValueError("confirmation candidate count mismatch")
    if sum(len(row["confirmation_seeds"]) for row in slots) != expected_trainings:
        raise ValueError("confirmation training count mismatch")
    if any(row["confirmation_seeds"] != seeds for row in slots):
        raise ValueError("all confirmation candidates must use the same seeds")
    return {
        "candidate_count": len(slots),
        "replicates_per_candidate": len(seeds),
        "training_count": expected_trainings,
        "nas_training_count": 6 * 150,
        "total_protocol_training_count": 6 * 150 + expected_trainings,
        "common_confirmation_seeds": seeds,
        "slots": slots,
    }


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported pilot config format")
    if payload.get("search_seeds") != [10, 11, 12]:
        raise ValueError("pilot search seeds must be exactly 10, 11, 12")
    if set(payload.get("methods", {})) != {"original", "expanded"}:
        raise ValueError("pilot methods must be exactly original and expanded")
    if payload["methods"]["original"]["normalization_space"] != "none":
        raise ValueError("original method must use normalization_space=none")
    if payload["methods"]["expanded"]["normalization_space"] != "none_layernorm":
        raise ValueError("expanded method must use normalization_space=none_layernorm")
    if payload.get("vae_retrain_required") is not False:
        raise ValueError("VAE retraining must remain disabled")
    checkpoint = ROOT / payload["checkpoint"]
    if _sha256(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError("VAE checkpoint hash mismatch")
    dataset_manifest = json.loads(
        (ROOT / payload["dataset_manifest"]).read_text(encoding="utf-8")
    )
    if dataset_manifest.get("split_fingerprint") != payload["split_fingerprint"]:
        raise ValueError("Cora split fingerprint mismatch")
    confirmation_manifest = json.loads(
        (ROOT / payload["confirmation_seed_manifest"]).read_text(encoding="utf-8")
    )
    confirmation_seeds = confirmation_manifest.get("seeds")
    if (
        not isinstance(confirmation_seeds, list)
        or len(confirmation_seeds) != 5
        or len(set(confirmation_seeds)) != 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in confirmation_seeds)
    ):
        raise ValueError("confirmation seed manifest must contain five unique integers")
    search = payload["search"]
    if int(search["n_init"]) + int(search["n_iter"]) != 150:
        raise ValueError("pilot must use exactly 50 initialization + 100 online evaluations")
    confirmation = payload.get("confirmation", {})
    expected_confirmation = {
        "nas_winner_count": 6,
        "historical_best_count": 1,
        "candidate_count": 7,
        "replicates_per_candidate": 5,
        "eval_epochs": 300,
        "patience": 80,
        "training_count": 35,
        "total_protocol_training_count": 935,
    }
    for field, expected in expected_confirmation.items():
        if int(confirmation.get(field, -1)) != expected:
            raise ValueError(f"confirmation protocol mismatch: {field}")
    load_historical_best(payload)
    return payload


def _initial_args(config: dict[str, Any], seed: int) -> SimpleNamespace:
    search = config["search"]
    return SimpleNamespace(
        hp_mode=search["hp_mode"],
        n_lhs_candidates=int(search["n_lhs_candidates"]),
        n_init=int(search["n_init"]),
        seed=int(seed),
        sigma_arch=float(search["sigma_arch"]),
        z_bound=float(search["z_bound"]),
        warm_start=search["warm_start"],
        warm_start_repeats=5,
        warm_start_noise=0.15,
    )


def _decode_architecture_fingerprints(
    config: dict[str, Any],
    seed: int,
    base_points: Sequence[torch.Tensor],
) -> list[str | None]:
    args = SimpleNamespace(
        hp_mode=config["search"]["hp_mode"],
        checkpoint=str((ROOT / config["checkpoint"]).resolve()),
    )
    vae = bo_phase4.load_vae(args, torch.device("cpu"), _QuietLogger())
    fingerprints = []
    for point in base_points:
        decoder_seed = stable_seed(int(seed), "decoder", point[: bo_phase4.ARCH_NZ])
        decoded = bo_phase4.decode_arch(
            vae,
            point[: bo_phase4.ARCH_NZ],
            torch.device("cpu"),
            n_trials=3,
            decoder_seed=decoder_seed,
        )
        fingerprints.append(bo_phase4._decoded_architecture_fingerprint(decoded))
    return fingerprints


def build_initialization_manifests(
    config: dict[str, Any], seed: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_points = bo_phase4.initial_points(
        _initial_args(config, seed), _QuietLogger()
    )
    if len(base_points) != 50:
        raise RuntimeError("S0 did not produce exactly 50 base points")
    base_fingerprints = [
        base_candidate_fingerprint(point.numpy()) for point in base_points
    ]
    if len(set(base_fingerprints)) != 50:
        raise RuntimeError("S0 base points are not unique")
    architecture_fingerprints = _decode_architecture_fingerprints(
        config, seed, base_points
    )
    expanded_assignment = balanced_initialization_assignment(
        base_fingerprints, search_seed=int(seed)
    )
    dataset_context = legacy_cora_context()

    manifests: dict[str, dict[str, Any]] = {}
    for method in ("original", "expanded"):
        method_config = config["methods"][method]
        space = method_config["normalization_space"]
        rows = []
        for index, (point, base_fp, architecture_fp) in enumerate(
            zip(base_points, base_fingerprints, architecture_fingerprints)
        ):
            normalization = (
                NORMALIZATION_NONE
                if method == "original"
                else expanded_assignment[base_fp]
            )
            full_fp = full_candidate_fingerprint(base_fp, normalization)
            training_seed = matched_training_seed(base_fp, 0)
            full_vector = compose_search_vector(
                point.numpy(),
                normalization,
                arch_nz=bo_phase4.ARCH_NZ,
                hp_mode=config["search"]["hp_mode"],
                normalization_space=space,
            )
            rows.append(
                {
                    "initial_index": index,
                    "base_vector": point.tolist(),
                    "full_vector": full_vector.tolist(),
                    "base_candidate_fingerprint": base_fp,
                    "full_candidate_fingerprint": full_fp,
                    "architecture_fingerprint": architecture_fp,
                    "normalization": normalization,
                    "normalization_value": normalization_value(normalization),
                    "replicate_id": 0,
                    "training_seed": training_seed,
                    "cache_key": evaluation_cache_key(
                        full_fp,
                        dataset_context,
                        evaluation_fidelity="full",
                        evaluation_seed=training_seed,
                    ),
                }
            )
        manifests[method] = {
            "format_version": 1,
            "protocol_id": config["protocol_id"],
            "method": method,
            "method_label": method_config["method_label"],
            "search_seed": int(seed),
            "normalization_space": space,
            "base_search_dim": 16,
            "gp_search_dim": 17 if method == "expanded" else 16,
            "initial_evaluation_count": len(rows),
            "rows": rows,
        }

    original = manifests["original"]
    expanded = manifests["expanded"]
    shared = sum(
        left["base_candidate_fingerprint"] == right["base_candidate_fingerprint"]
        and left["base_vector"] == right["base_vector"]
        and left["architecture_fingerprint"] == right["architecture_fingerprint"]
        for left, right in zip(original["rows"], expanded["rows"])
    )
    original_counts = {
        name: sum(row["normalization"] == name for row in original["rows"])
        for name in (NORMALIZATION_NONE, NORMALIZATION_LAYER_NORM)
    }
    expanded_counts = {
        name: sum(row["normalization"] == name for row in expanded["rows"])
        for name in (NORMALIZATION_NONE, NORMALIZATION_LAYER_NORM)
    }
    paired_cache_isolated = all(
        evaluation_cache_key(
            full_candidate_fingerprint(row["base_candidate_fingerprint"], NORMALIZATION_NONE),
            dataset_context,
            evaluation_fidelity="full",
            evaluation_seed=row["training_seed"],
        )
        != evaluation_cache_key(
            full_candidate_fingerprint(
                row["base_candidate_fingerprint"], NORMALIZATION_LAYER_NORM
            ),
            dataset_context,
            evaluation_fidelity="full",
            evaluation_seed=row["training_seed"],
        )
        for row in original["rows"]
    )
    comparison = {
        "search_seed": int(seed),
        "shared_base_points": shared,
        "original_counts": original_counts,
        "expanded_counts": expanded_counts,
        "base_vectors_identical": shared == 50,
        "expanded_strictly_balanced": expanded_counts
        == {NORMALIZATION_NONE: 25, NORMALIZATION_LAYER_NORM: 25},
        "matched_training_seeds": all(
            left["training_seed"] == right["training_seed"]
            for left, right in zip(original["rows"], expanded["rows"])
        ),
        "paired_cache_isolated": paired_cache_isolated,
    }
    if comparison != {
        **comparison,
        "shared_base_points": 50,
        "base_vectors_identical": True,
        "expanded_strictly_balanced": True,
        "matched_training_seeds": True,
        "paired_cache_isolated": True,
    }:
        raise RuntimeError(f"initialization fairness check failed: {comparison}")
    return original, expanded, comparison


def synthetic_acquisition_check(config: dict[str, Any]) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260816)
    base = torch.rand(8, 16, generator=generator, dtype=torch.double)
    base[:, :12] = base[:, :12] * 5.0 - 2.5
    states = torch.tensor([[float(index % 2)] for index in range(8)], dtype=torch.double)
    train_x = torch.cat((base, states), dim=1)
    train_y = torch.tensor([0.60, 0.69, 0.63, 0.73, 0.61, 0.71, 0.65, 0.76])
    with isolated_rng(20260816, torch.device("cpu")):
        predictor = AccuracyGPPredictor.fit_offline(
            train_x,
            train_y,
            arch_nz=12,
            hp_mode="global4",
            z_bound=2.5,
            include_normalization=True,
            fit_steps=2,
            device="cpu",
        )
    candidate = train_x[0, :16].clone()
    pair = torch.stack(
        (
            torch.cat((candidate, torch.tensor([0.0], dtype=torch.double))),
            torch.cat((candidate, torch.tensor([1.0], dtype=torch.double))),
        )
    )
    with isolated_rng(20260817, torch.device("cpu")):
        logei = predictor.make_logei(float(train_y.max()))
        with torch.no_grad():
            values = logei(
                predictor.acquisition_inputs(pair).double().unsqueeze(1)
            ).reshape(-1)
    selected_index = int(torch.argmax(values).item())
    return {
        "same_base_candidate": bool(torch.equal(pair[0, :16], pair[1, :16])),
        "normalization_values": [float(pair[0, -1]), float(pair[1, -1])],
        "none_logei": float(values[0].item()),
        "layernorm_logei": float(values[1].item()),
        "selected_normalization": [NORMALIZATION_NONE, NORMALIZATION_LAYER_NORM][selected_index],
        "selected_count": 1,
        "gp_search_dim": predictor.search_dim,
        "ard_dimension": int(
            predictor.model.covar_module.lengthscale.reshape(-1).numel()
        ),
    }


def build_search_command(
    config: dict[str, Any], method: str, seed: int, python: str
) -> tuple[list[str], Path, Path]:
    method_config = config["methods"][method]
    search = config["search"]
    output = ROOT / config["output_root"] / method / f"search_seed{seed}"
    log_dir = ROOT / config["log_root"] / method / f"search_seed{seed}"
    command = [
        python,
        str(ROOT / "bo_phase4.py"),
        "--checkpoint", str((ROOT / config["checkpoint"]).resolve()),
        "--dataset", "cora",
        "--data_root", str((ROOT / config["data_root"]).resolve()),
        "--expected_dataset_manifest", str((ROOT / config["dataset_manifest"]).resolve()),
        "--output", str(output),
        "--log_dir", str(log_dir),
        "--version", f"{config['protocol_id']}_{method}_seed{seed}",
        "--method_label", method_config["method_label"],
        "--normalization_space", method_config["normalization_space"],
        "--seed", str(seed),
        "--require_cuda",
    ]
    boolean_flags = {"use_conditional_kernel", "adaptive_sampling"}
    for name, value in search.items():
        if name in boolean_flags:
            if value:
                command.append(f"--{name}")
            continue
        command.extend((f"--{name}", str(value)))
    return command, output, log_dir


def validate_search_command(command: Sequence[str]) -> dict[str, Any]:
    previous = list(sys.argv)
    try:
        sys.argv = [str(command[1]), *[str(value) for value in command[2:]]]
        parsed = bo_phase4.parse_args()
    finally:
        sys.argv = previous
    bo_phase4.validate_layernorm_search_space_configuration(parsed)
    bo_phase4.validate_two_stage_initialization_config(parsed)
    return {
        "method_label": parsed.method_label,
        "search_seed": int(parsed.seed),
        "normalization_space": parsed.normalization_space,
        "search_dim": bo_phase4._search_dim(parsed),
        "planned_full_evaluations": int(parsed.n_init) + int(parsed.n_iter),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--method", choices=("original", "expanded"))
    parser.add_argument("--search-seed", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="/tmp/cora_layernorm_search_space_dry_run")
    parser.add_argument("--python", default=sys.executable)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(Path(args.config).resolve())
    if args.execute:
        if args.all or args.method is None or args.search_seed is None:
            raise ValueError("execution requires one explicit --method and --search-seed")
        command, output, log_dir = build_search_command(
            config, args.method, int(args.search_seed), args.python
        )
        validate_search_command(command)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing non-empty output directory: {output}")
        if log_dir.exists() and any(log_dir.iterdir()):
            raise FileExistsError(f"refusing non-empty log directory: {log_dir}")
        subprocess.run(command, cwd=ROOT, check=True)
        return 0

    selected_seeds = config["search_seeds"] if args.all else [args.search_seed]
    if selected_seeds == [None]:
        raise ValueError("dry-run requires --all or --search-seed")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    command_checks = []
    for seed in selected_seeds:
        original, expanded, comparison = build_initialization_manifests(
            config, int(seed)
        )
        seed_dir = output / f"search_seed{int(seed)}"
        atomic_json_dump(original, seed_dir / "original_initialization_manifest.json")
        atomic_json_dump(expanded, seed_dir / "expanded_initialization_manifest.json")
        atomic_json_dump(comparison, seed_dir / "initialization_comparison.json")
        comparisons.append(comparison)
        for method in ("original", "expanded"):
            command, _search_output, _log_dir = build_search_command(
                config, method, int(seed), args.python
            )
            command_checks.append(validate_search_command(command))
    acquisition = synthetic_acquisition_check(config)
    confirmation = build_confirmation_plan(config)
    summary = {
        "format_version": 1,
        "protocol_id": config["protocol_id"],
        "vae_retrain_required": False,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "split_fingerprint": config["split_fingerprint"],
        "config_sha256": _sha256(Path(args.config).resolve()),
        "training_started": False,
        "gpu_required": False,
        "initialization": comparisons,
        "acquisition": acquisition,
        "confirmation": confirmation,
        "command_validation": command_checks,
    }
    atomic_json_dump(summary, output / "dry_run_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
