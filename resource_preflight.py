"""Full-batch GPU resource preflight over static worst-case pool candidates."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import bo_phase4
from dataset_utils import (
    load_dataset_from_request,
    resolve_dataset_request,
    validate_expected_dataset_manifest,
    write_dataset_artifacts,
)
from eval_utils import DynamicGNN, decode_hp_by_mode, stable_seed
from evaluation_errors import InfrastructureEvaluationError
from initialization_wgmm_ted import candidate_fingerprint
from surrogate.checkpoint_io import atomic_json_dump


OPERATION_FAMILIES = (
    "GCNConv",
    "GATConv",
    "SAGEConv",
    "GINConv",
    "GCNII",
)


def estimate_activation_resources(
    *,
    num_nodes: int,
    num_edges: int,
    architecture: dict[str, Any],
    hp: dict[str, Any],
    dtype_bytes: int = 4,
) -> dict[str, int]:
    """Estimate full-batch message/node activations for label-free ranking.

    This is a deterministic preflight score, not a CUDA allocator prediction.
    It deliberately includes hidden width and every effective message-passing
    layer.  GAT layers additionally include their active head multiplicity.
    """

    nodes = int(num_nodes)
    edges = int(num_edges)
    element_bytes = int(dtype_bytes)
    hidden = int(hp["hidden_dim"])
    if min(nodes, edges, element_bytes, hidden) <= 0:
        raise ValueError("activation resource inputs must be positive")
    operations = [str(value) for value in architecture.get("operations", [])]
    effective_operations = [op for op in operations if op != "Identity"]
    effective_layers = int(architecture.get("effective_layers", 0))
    if effective_layers != len(effective_operations):
        raise ValueError(
            "architecture effective_layers does not match non-Identity operations"
        )
    gat_heads_by_layer = hp.get("gat_heads_by_layer")
    default_gat_heads = max(1, int(hp.get("gat_heads", 1)))
    layer_multipliers: list[int] = []
    gat_layer_count = 0
    for index, operation in enumerate(operations):
        if operation == "Identity":
            continue
        multiplier = 1
        if operation == "GATConv":
            gat_layer_count += 1
            if isinstance(gat_heads_by_layer, list) and index < len(
                gat_heads_by_layer
            ):
                multiplier = max(1, int(gat_heads_by_layer[index]))
            else:
                multiplier = default_gat_heads
        layer_multipliers.append(multiplier)
    message_elements = edges * hidden * sum(layer_multipliers)
    node_elements = nodes * hidden * (effective_layers + 2)
    edge_bytes = message_elements * element_bytes
    node_bytes = node_elements * element_bytes
    return {
        "hidden_dimension": hidden,
        "effective_layers": effective_layers,
        "gat_layer_count": gat_layer_count,
        "maximum_active_gat_heads": (
            max(layer_multipliers) if gat_layer_count else 1
        ),
        "edge_activation_estimate_bytes": int(edge_bytes),
        "node_activation_estimate_bytes": int(node_bytes),
        "activation_resource_estimate_bytes": int(edge_bytes + node_bytes),
    }


def _pool_args(seed: int, pool_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        hp_mode="global4",
        n_lhs_candidates=int(pool_size),
        n_init=50,
        seed=int(seed),
        sigma_arch=0.8,
        z_bound=2.5,
    )


def _decode_static_candidates(
    vae,
    pool: Sequence[torch.Tensor],
    *,
    seed: int,
    device: torch.device,
    in_ch: int,
    out_ch: int,
    num_nodes: int,
    num_edges: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, z_search in enumerate(pool):
        fingerprint = candidate_fingerprint(z_search.numpy())
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
        if config is None:
            continue
        hp = decode_hp_by_mode(
            z_search,
            bo_phase4.ARCH_NZ,
            config,
            "global4",
            -4.0,
            -1.5,
            0.1,
            0.6,
        )
        model = DynamicGNN(
            config,
            in_ch,
            out_ch,
            dropout=hp["dropout"],
            hidden_dim=hp["hidden_dim"],
            gat_heads=hp["gat_heads"],
            sage_aggr=hp["sage_aggr"],
            gin_eps=hp["gin_eps"],
        )
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        operations = list(config.get("operations", []))
        effective_layers = int(config.get("effective_layers", 0))
        attention_heads = (
            int(hp["gat_heads"]) if "GATConv" in operations else 1
        )
        resource_estimate = estimate_activation_resources(
            num_nodes=int(num_nodes),
            num_edges=int(num_edges),
            architecture={
                "operations": operations,
                "effective_layers": effective_layers,
            },
            hp=hp,
        )
        candidates.append(
            {
                "candidate_pool_index": int(index),
                "candidate_fingerprint": fingerprint,
                "z_search": z_search.tolist(),
                "architecture": {
                    "operations": operations,
                    "edges": [
                        list(edge) for edge in config.get("edges", [])
                    ],
                    "effective_layers": effective_layers,
                },
                "hp": hp,
                "parameter_count": int(parameter_count),
                "attention_heads": int(attention_heads),
                **resource_estimate,
            }
        )
    if not candidates:
        raise ValueError("candidate pool did not decode any buildable architecture")
    return candidates


def select_static_worst_cases(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[str, tuple[str, dict[str, Any]]] = {}

    def add(reason: str, row: dict[str, Any]) -> None:
        fingerprint = str(row["candidate_fingerprint"])
        if fingerprint in selected:
            prior_reason, prior_row = selected[fingerprint]
            selected[fingerprint] = (prior_reason + "+" + reason, prior_row)
        else:
            selected[fingerprint] = (reason, row)

    add(
        "max_hidden_dim",
        max(candidates, key=lambda row: int(row["hp"]["hidden_dim"])),
    )
    add(
        "max_effective_layers",
        max(
            candidates,
            key=lambda row: int(row["architecture"]["effective_layers"]),
        ),
    )
    add(
        "max_attention_heads",
        max(candidates, key=lambda row: int(row["attention_heads"])),
    )
    add(
        "max_parameter_count",
        max(candidates, key=lambda row: int(row["parameter_count"])),
    )
    add(
        "max_edge_activation_estimate",
        max(
            candidates,
            key=lambda row: int(row["edge_activation_estimate_bytes"]),
        ),
    )
    add(
        "max_activation_resource_estimate",
        max(
            candidates,
            key=lambda row: int(
                row.get(
                    "activation_resource_estimate_bytes",
                    row["edge_activation_estimate_bytes"],
                )
            ),
        ),
    )
    gat_candidates = [
        row
        for row in candidates
        if "GATConv" in row["architecture"]["operations"]
    ]
    if not gat_candidates:
        raise ValueError("768-candidate pool contains no GAT candidate")
    add(
        "gat_attention_family",
        max(
            gat_candidates,
            key=lambda row: (
                int(row["attention_heads"]),
                int(row["edge_activation_estimate_bytes"]),
            ),
        ),
    )
    for operation in OPERATION_FAMILIES:
        family = [
            row
            for row in candidates
            if operation in row["architecture"]["operations"]
        ]
        if family:
            add(
                f"operation_family_{operation}",
                max(family, key=lambda row: int(row["parameter_count"])),
            )

    output: list[dict[str, Any]] = []
    for reason, row in selected.values():
        output.append({**row, "selection_reason": reason})
    output.sort(key=lambda row: int(row["candidate_pool_index"]))
    return output


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


def _probe_candidate(
    row: dict[str, Any],
    *,
    data,
    in_ch: int,
    out_ch: int,
    device: torch.device,
) -> dict[str, Any]:
    config = {
        **row["architecture"],
        "edges": [tuple(edge) for edge in row["architecture"]["edges"]],
    }
    hp = row["hp"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    model = DynamicGNN(
        config,
        in_ch,
        out_ch,
        dropout=float(hp["dropout"]),
        hidden_dim=int(hp["hidden_dim"]),
        gat_heads=int(hp["gat_heads"]),
        sage_aggr=str(hp["sage_aggr"]),
        gin_eps=float(hp["gin_eps"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
    )

    started = time.monotonic()
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    torch.cuda.synchronize(device)
    forward_seconds = time.monotonic() - started

    started = time.monotonic()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    backward_step_seconds = time.monotonic() - started

    started = time.monotonic()
    model.eval()
    with torch.no_grad():
        validation_output = model(data.x, data.edge_index)
        if validation_output[data.val_mask].numel() == 0:
            raise ValueError("validation inference produced no selected outputs")
    torch.cuda.synchronize(device)
    inference_seconds = time.monotonic() - started
    result = {
        **row,
        "status": "passed",
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "forward_seconds": float(forward_seconds),
        "backward_optimizer_seconds": float(backward_step_seconds),
        "validation_inference_seconds": float(inference_seconds),
    }
    del validation_output, output, loss, optimizer, model
    torch.cuda.empty_cache()
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static worst-case full-batch GPU resource preflight"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_dataset_manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool_size", type=int, default=768)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument(
        "--ogbn_arxiv_edge_mode",
        choices=("directed", "undirected"),
        default=None,
    )
    return parser.parse_args(argv)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "resource_preflight.json"
    request = resolve_dataset_request(
        args.dataset,
        args.data_root,
        split_seed=args.split_seed,
        ogbn_arxiv_edge_mode=args.ogbn_arxiv_edge_mode,
    )
    if not torch.cuda.is_available():
        payload = {
            "status": "blocked_cuda_unavailable",
            "dataset": request.canonical_name,
            "training_mode": "full_batch",
            "pool_size": int(args.pool_size),
            "all_worst_cases_passed": False,
            "error": "CUDA is unavailable; full-batch protocol is not frozen",
        }
        atomic_json_dump(payload, result_path)
        raise InfrastructureEvaluationError(
            payload["error"], context=payload
        )
    if not args.expected_dataset_manifest:
        raise ValueError(
            "--expected_dataset_manifest is required for a real GPU preflight"
        )
    device = torch.device("cuda:0")
    bundle = load_dataset_from_request(request)
    validate_expected_dataset_manifest(
        bundle, args.expected_dataset_manifest
    )
    write_dataset_artifacts(bundle, output)
    bundle.to(device)

    logger = logging.getLogger("resource_preflight")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    vae_args = SimpleNamespace(
        hp_mode="global4",
        checkpoint=args.checkpoint,
    )
    vae = bo_phase4.load_vae(vae_args, device, logger)
    pool = bo_phase4.make_lhs_pool(
        _pool_args(int(args.seed), int(args.pool_size))
    )
    static_candidates = _decode_static_candidates(
        vae,
        pool,
        seed=int(args.seed),
        device=device,
        in_ch=bundle.num_features,
        out_ch=bundle.num_classes,
        num_nodes=bundle.num_nodes,
        num_edges=bundle.num_edges,
    )
    selected = select_static_worst_cases(static_candidates)
    gpu = _gpu_identity(device)
    probes: list[dict[str, Any]] = []
    all_passed = True
    for row in selected:
        try:
            probes.append(
                _probe_candidate(
                    row,
                    data=bundle.data,
                    in_ch=bundle.num_features,
                    out_ch=bundle.num_classes,
                    device=device,
                )
            )
        except Exception as exc:
            all_passed = False
            probes.append(
                {
                    **row,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(device)
                    ),
                }
            )
            torch.cuda.empty_cache()
            break
    payload = {
        "status": "passed" if all_passed else "failed",
        "dataset": bundle.canonical_name,
        "dataset_context": bundle.context,
        "candidate_pool_size": int(args.pool_size),
        "candidate_pool_fingerprint": bo_phase4.fingerprint_array(
            torch.stack(pool).numpy()
        ),
        "selection_uses_labels_or_accuracy": False,
        "training_mode": "full_batch",
        "gpu": gpu,
        "selected_candidate_count": len(selected),
        "all_worst_cases_passed": bool(all_passed),
        "probes": probes,
    }
    atomic_json_dump(payload, result_path)
    if not all_passed:
        raise InfrastructureEvaluationError(
            "one or more static worst-case candidates failed; "
            "full_batch protocol is not frozen",
            context={"resource_preflight": str(result_path)},
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_preflight(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
