"""Isolated one-step resource smoke for a persisted Flickr OOM candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Sequence

import torch

from dataset_utils import (
    load_dataset_from_request,
    resolve_dataset_request,
    validate_expected_dataset_manifest,
)
from evaluation_errors import is_infrastructure_failure
from flickr_continuous_extreme_preflight import ProcessMemoryMonitor
from resource_preflight import (
    _gpu_identity,
    _probe_candidate,
    estimate_activation_resources,
)
from surrogate.checkpoint_io import atomic_json_dump


EXPECTED_FINGERPRINT = (
    "1b09dd32bcdd4427f5de3cc41b9720957945028c9d6f5621c7e987008e997072"
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="one-step deterministic full-batch Flickr candidate smoke"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expected-dataset-manifest", required=True)
    parser.add_argument("--infrastructure-error", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _candidate_from_error(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("candidate_fingerprint") != EXPECTED_FINGERPRINT:
        raise ValueError(
            "infrastructure artifact is not the expected failed candidate: "
            f"{artifact.get('candidate_fingerprint')!r}"
        )
    context = artifact.get("context")
    if not isinstance(context, dict):
        raise ValueError("infrastructure artifact context is missing")
    architecture = context.get("architecture")
    hp = context.get("hp")
    if not isinstance(architecture, dict) or not isinstance(hp, dict):
        raise ValueError("infrastructure artifact architecture/hp is missing")
    operations = [str(value) for value in architecture.get("operations", [])]
    if operations != [
        "GATConv",
        "GATConv",
        "GCNConv",
        "SAGEConv",
        "GINConv",
    ]:
        raise ValueError(f"unexpected failed architecture: {operations}")
    if int(hp.get("hidden_dim", -1)) != 512:
        raise ValueError(f"unexpected failed hidden_dim: {hp.get('hidden_dim')!r}")
    row = {
        "candidate_pool_index": None,
        "candidate_fingerprint": EXPECTED_FINGERPRINT,
        "architecture": {
            "operations": operations,
            "edges": [list(edge) for edge in architecture.get("edges", [])],
            "effective_layers": len([op for op in operations if op != "Identity"]),
        },
        "hp": {
            **hp,
            "l2": float(hp["weight_decay"]),
            "gat_heads_by_layer": None,
            "sage_aggr_by_layer": None,
            "gin_eps_by_layer": None,
        },
        "source": "persisted_formal_infrastructure_error",
        "source_artifact": str(Path(path).resolve()),
    }
    return row, artifact


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    row, failure_artifact = _candidate_from_error(args.infrastructure_error)
    request = resolve_dataset_request("flickr", args.data_root)
    started_at = _now()
    started = time.monotonic()
    bundle = load_dataset_from_request(request)
    validate_expected_dataset_manifest(bundle, args.expected_dataset_manifest)
    device = torch.device("cuda:0")
    bundle.to(device)
    row.update(
        estimate_activation_resources(
            num_nodes=bundle.num_nodes,
            num_edges=bundle.num_edges,
            architecture=row["architecture"],
            hp=row["hp"],
        )
    )
    monitor = ProcessMemoryMonitor(os.getpid())
    monitor.start()
    status = "completed"
    error_type = None
    error_message = None
    error_traceback = None
    probe = None
    try:
        probe = _probe_candidate(
            row,
            data=bundle.data,
            in_ch=bundle.num_features,
            out_ch=bundle.num_classes,
            device=device,
        )
    except BaseException as exc:
        status = "oom_confirmed" if is_infrastructure_failure(exc) else "failed"
        error_type = type(exc).__name__
        error_message = str(exc)
        error_traceback = traceback.format_exc()
    finally:
        monitor.stop()
    payload = {
        "status": status,
        "purpose": "isolated_resource_smoke_not_formal_evaluation",
        "training_mode": "full_batch",
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "candidate_fingerprint": EXPECTED_FINGERPRINT,
        "candidate": row,
        "original_failure_gpu": failure_artifact.get("gpu"),
        "dataset_context": bundle.context,
        "gpu": _gpu_identity(device),
        "started_at": started_at,
        "ended_at": _now(),
        "wall_seconds": float(time.monotonic() - started),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "torch_allocated_bytes_after_probe": int(torch.cuda.memory_allocated(device)),
        "torch_reserved_bytes_after_probe": int(torch.cuda.memory_reserved(device)),
        "nvidia_smi_process_peak_used_mib": int(monitor.peak_used_mib),
        "nvidia_smi_sample_count": int(monitor.sample_count),
        "nvidia_smi_monitor_error_count": int(monitor.error_count),
        "probe": probe,
        "error_type": error_type,
        "error": error_message,
        "traceback": error_traceback,
        "formal_history_or_gp_written": False,
        "candidate_converted_to_invalid": False,
    }
    atomic_json_dump(payload, output)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] in ("completed", "oom_confirmed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
