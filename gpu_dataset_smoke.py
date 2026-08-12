"""One-candidate real-data GPU smoke for the unified NAS evaluation path."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Sequence

import torch

import bo_phase4
from dataset_utils import (
    load_dataset_from_request,
    resolve_dataset_request,
    validate_expected_dataset_manifest,
    write_dataset_artifacts,
)
from evaluation_errors import InfrastructureEvaluationError
from eval_utils import stable_seed
from initialization_wgmm_ted import candidate_fingerprint
from surrogate.checkpoint_io import atomic_json_dump


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real decoded GNN candidate as a GPU-only smoke"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_dataset_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool_size", type=int, default=768)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument(
        "--ogbn_arxiv_edge_mode",
        choices=("directed", "undirected"),
        default=None,
    )
    return parser.parse_args(argv)


def _evaluation_args(
    args: argparse.Namespace, dataset_context: dict[str, Any]
) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        hp_mode="global4",
        seed=int(args.seed),
        z_bound=2.5,
        log_lr_min=-4.0,
        log_lr_max=-1.5,
        dropout_min=0.1,
        dropout_max=0.6,
        gcnii_alpha=0.1,
        gcnii_theta=0.5,
        eval_epochs=int(args.max_epochs),
        patience=int(args.patience),
        output=str(args.output),
        dataset_context=dataset_context,
        dataset=dataset_context["canonical_name"],
        split_seed=dataset_context["split_seed"],
        ogbn_arxiv_edge_mode=args.ogbn_arxiv_edge_mode,
        initial_selection_strategy="dataset_gpu_smoke",
        method_label="dataset_gpu_smoke",
    )


def _pool_args(seed: int, pool_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        hp_mode="global4",
        n_lhs_candidates=int(pool_size),
        n_init=1,
        seed=int(seed),
        sigma_arch=0.8,
        z_bound=2.5,
    )


def _select_first_decodable(vae, pool, *, seed: int, device: torch.device):
    for index, z_search in enumerate(pool):
        decoder_seed = stable_seed(
            int(seed), "decoder", z_search[: bo_phase4.ARCH_NZ]
        )
        config = bo_phase4.decode_arch(
            vae,
            z_search[: bo_phase4.ARCH_NZ],
            device,
            n_trials=5,
            decoder_seed=decoder_seed,
        )
        if config is not None:
            return int(index), z_search
    raise RuntimeError("768-candidate pool contains no decodable architecture")


def _gpu_identity(device: torch.device) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    raw_uuid = getattr(properties, "uuid", None)
    return {
        "gpu_index": int(index),
        "gpu_name": properties.name,
        "gpu_uuid": None if raw_uuid is None else str(raw_uuid),
        "total_memory_bytes": int(properties.total_memory),
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "smoke_result.json"
    if not torch.cuda.is_available():
        payload = {
            "status": "blocked_cuda_unavailable",
            "dataset": args.dataset,
            "error": "CUDA is unavailable; CPU fallback is forbidden",
        }
        atomic_json_dump(payload, result_path)
        raise InfrastructureEvaluationError(payload["error"], context=payload)

    request = resolve_dataset_request(
        args.dataset,
        args.data_root,
        split_seed=args.split_seed,
        ogbn_arxiv_edge_mode=args.ogbn_arxiv_edge_mode,
    )
    device = torch.device("cuda:0")
    bundle = load_dataset_from_request(request)
    validate_expected_dataset_manifest(bundle, args.expected_dataset_manifest)
    write_dataset_artifacts(bundle, output)
    bundle.to(device)

    logger = logging.getLogger(f"gpu_dataset_smoke.{bundle.canonical_name}")
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    eval_args = _evaluation_args(args, bundle.context)
    vae = bo_phase4.load_vae(eval_args, device, logger)
    pool = bo_phase4.make_lhs_pool(
        _pool_args(int(args.seed), int(args.pool_size))
    )
    pool_index, z_search = _select_first_decodable(
        vae, pool, seed=int(args.seed), device=device
    )
    fingerprint = candidate_fingerprint(z_search.numpy())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    started = time.monotonic()
    try:
        evaluated_z, result = bo_phase4.eval_candidate(
            vae,
            z_search,
            bundle.data,
            bundle.num_features,
            bundle.num_classes,
            eval_args,
            device,
            step=0,
            evaluation_stage="dataset_gpu_smoke",
            evaluation_fidelity="full",
            candidate_fingerprint=fingerprint,
        )
        torch.cuda.synchronize(device)
    except Exception as exc:
        payload = {
            "status": "failed",
            "dataset": bundle.canonical_name,
            "candidate_pool_index": pool_index,
            "candidate_fingerprint": fingerprint,
            "gpu": _gpu_identity(device),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
            "runtime_seconds": float(time.monotonic() - started),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_json_dump(payload, result_path)
        raise

    runtime_seconds = float(time.monotonic() - started)
    record = bo_phase4.history_record(
        0,
        "dataset_gpu_smoke",
        evaluated_z,
        result,
        eval_args,
    )
    atomic_json_dump([record], output / "history_smoke.json")
    payload = {
        "status": "passed" if bool(result["valid"]) else "invalid_candidate",
        "dataset": bundle.canonical_name,
        "dataset_context": bundle.context,
        "candidate_pool_size": int(args.pool_size),
        "candidate_pool_index": pool_index,
        "candidate_fingerprint": fingerprint,
        "architecture": {
            "operations": result.get("operations"),
            "edges": result.get("edges"),
        },
        "hp": result.get("hp"),
        "validation_accuracy": float(result["val_acc"]),
        "valid": bool(result["valid"]),
        "best_epoch": result.get("best_epoch"),
        "stopped_epoch": result.get("stopped_epoch"),
        "epochs_ran": int(result.get("epochs_ran") or 0),
        "requested_max_epochs": int(args.max_epochs),
        "requested_patience": int(args.patience),
        "test_metric_evaluated": False,
        "history_path": str((output / "history_smoke.json").resolve()),
        "gpu": _gpu_identity(device),
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "runtime_seconds": runtime_seconds,
    }
    atomic_json_dump(payload, result_path)
    if not bool(result["valid"]):
        raise RuntimeError("decoded smoke candidate was structurally invalid")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
