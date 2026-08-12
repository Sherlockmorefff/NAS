"""Isolated Flickr preflight for decoded out-of-pool continuous extremes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bo_phase4
from dataset_utils import (
    load_dataset_from_request,
    resolve_dataset_request,
    validate_expected_dataset_manifest,
)
from eval_utils import DynamicGNN, decode_hp_by_mode, stable_seed
from evaluation_errors import InfrastructureEvaluationError
from hp_modes import HIDDEN_DIM_OPTIONS, MAX_OP_NODES, condition_mask_vector_from_ops
from initialization_wgmm_ted import candidate_fingerprint, fingerprint_array
from scripts.validation.resource_preflight import (
    _gpu_identity,
    _pool_args,
    _probe_candidate,
    estimate_activation_resources,
)
from surrogate.checkpoint_io import atomic_json_dump


SOURCE = "continuous_search_space_extreme"
FORMAL_POOL_SEEDS = (5, 6, 7, 8, 9)
FORMAL_HP_MODE = "global4"
FORMAL_Z_BOUND = 2.5
FORMAL_EVAL_EPOCHS = 150
FORMAL_PATIENCE = 40


def _utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


class ProcessMemoryMonitor:
    """Sample read-only nvidia-smi process memory for the current PID."""

    def __init__(self, pid: int, interval_seconds: float = 0.1) -> None:
        self.pid = int(pid)
        self.interval_seconds = float(interval_seconds)
        self.peak_used_mib = 0
        self.sample_count = 0
        self.error_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if completed.returncode != 0:
                    self.error_count += 1
                else:
                    for line in completed.stdout.splitlines():
                        fields = [value.strip() for value in line.split(",")]
                        if len(fields) != 2:
                            continue
                        try:
                            pid, used_mib = int(fields[0]), int(fields[1])
                        except ValueError:
                            continue
                        if pid == self.pid:
                            self.sample_count += 1
                            self.peak_used_mib = max(self.peak_used_mib, used_mib)
            except (OSError, subprocess.SubprocessError):
                self.error_count += 1
            self._stop.wait(self.interval_seconds)


def deterministic_arch_search_vectors(
    *,
    arch_nz: int = bo_phase4.ARCH_NZ,
    z_bound: float = FORMAL_Z_BOUND,
    sobol_count: int = 4096,
) -> list[torch.Tensor]:
    """Return deterministic corners plus interior Sobol decode-only probes."""

    dimension = int(arch_nz)
    if dimension <= 0 or dimension > 20:
        raise ValueError("arch_nz must be in [1, 20]")
    bound = float(z_bound)
    if bound <= 0.0:
        raise ValueError("z_bound must be positive")
    corners = []
    for value in range(1 << dimension):
        row = [
            bound if value & (1 << position) else -bound
            for position in range(dimension)
        ]
        corners.append(torch.tensor(row, dtype=torch.float32))
    if int(sobol_count) > 0:
        sobol_seed = stable_seed(
            20260731, "flickr_continuous_extreme_sobol", dimension
        ) % (2**31)
        engine = torch.quasirandom.SobolEngine(
            dimension, scramble=True, seed=int(sobol_seed)
        )
        interior = engine.draw(int(sobol_count)).float()
        interior = (interior * 2.0 - 1.0) * bound
        corners.extend(row.clone() for row in interior)
    return corners


def _formal_pool_identity() -> tuple[set[str], dict[str, str]]:
    members: set[str] = set()
    aggregate: dict[str, str] = {}
    for seed in FORMAL_POOL_SEEDS:
        pool = bo_phase4.make_lhs_pool(_pool_args(seed, 768))
        aggregate[str(seed)] = fingerprint_array(torch.stack(pool).numpy())
        members.update(candidate_fingerprint(row.numpy()) for row in pool)
    if len(members) != len(FORMAL_POOL_SEEDS) * 768:
        raise RuntimeError("formal candidate pools contain duplicate z_search rows")
    return members, aggregate


def _parameter_count(
    architecture: dict[str, Any],
    hp: dict[str, Any],
    *,
    in_ch: int,
    out_ch: int,
) -> int:
    model = DynamicGNN(
        architecture,
        in_ch,
        out_ch,
        dropout=float(hp["dropout"]),
        hidden_dim=int(hp["hidden_dim"]),
        gat_heads=int(hp["gat_heads"]),
        sage_aggr=str(hp["sage_aggr"]),
        gin_eps=float(hp["gin_eps"]),
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    del model
    return int(count)


def decode_continuous_extremes(
    vae,
    *,
    device: torch.device,
    search_seed: int,
    in_ch: int,
    out_ch: int,
    num_nodes: int,
    num_edges: int,
    sobol_count: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Decode candidates without labels and return a minimal extreme cover."""

    pool_members, pool_fingerprints = _formal_pool_identity()
    hp_tail = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
    if int(HIDDEN_DIM_OPTIONS[-1]) != 512:
        raise RuntimeError("formal hidden dimension maximum unexpectedly changed")
    candidates: list[dict[str, Any]] = []
    parameter_cache: dict[str, int] = {}
    for search_index, z_arch in enumerate(
        deterministic_arch_search_vectors(sobol_count=sobol_count)
    ):
        raw = torch.cat([z_arch, hp_tail])
        canonical_z, fingerprint = bo_phase4._canonical_candidate_identity(
            raw,
            hp_mode=FORMAL_HP_MODE,
            z_bound=FORMAL_Z_BOUND,
        )
        if fingerprint in pool_members:
            continue
        z_search = torch.tensor(canonical_z, dtype=torch.float32)
        decoder_seed = stable_seed(
            int(search_seed), "decoder", z_search[: bo_phase4.ARCH_NZ]
        )
        config = bo_phase4.decode_arch(
            vae,
            z_search[: bo_phase4.ARCH_NZ],
            device,
            n_trials=5,
            decoder_seed=decoder_seed,
        )
        if config is None or int(config.get("effective_layers", 0)) <= 0:
            continue
        operations = [str(value) for value in config.get("operations", [])]
        if len(operations) > MAX_OP_NODES:
            raise RuntimeError("decoder returned more than five operation nodes")
        hp = decode_hp_by_mode(
            z_search,
            bo_phase4.ARCH_NZ,
            config,
            FORMAL_HP_MODE,
            -4.0,
            -1.5,
            0.1,
            0.6,
        )
        if int(hp["hidden_dim"]) != 512:
            raise RuntimeError("hidden-dimension boundary did not decode to 512")
        if int(hp["gat_heads"]) != 1:
            raise RuntimeError("global4 must keep the formal GAT head count at one")
        architecture = {
            "operations": operations,
            "edges": [list(edge) for edge in config.get("edges", [])],
            "effective_layers": int(config["effective_layers"]),
        }
        architecture_key = json.dumps(
            architecture, sort_keys=True, separators=(",", ":")
        )
        if architecture_key not in parameter_cache:
            parameter_cache[architecture_key] = _parameter_count(
                {
                    **architecture,
                    "edges": [tuple(edge) for edge in architecture["edges"]],
                },
                hp,
                in_ch=in_ch,
                out_ch=out_ch,
            )
        resources = estimate_activation_resources(
            num_nodes=num_nodes,
            num_edges=num_edges,
            architecture=architecture,
            hp=hp,
        )
        mask = condition_mask_vector_from_ops(operations, FORMAL_HP_MODE)
        if mask != hp["condition_mask_vector"] or mask != [1.0] * 4:
            raise RuntimeError("global4 conditional HP mask changed unexpectedly")
        candidates.append(
            {
                "candidate_pool_index": None,
                "source": SOURCE,
                "decode_search_index": int(search_index),
                "candidate_fingerprint": fingerprint,
                "canonical_z_search": z_search.tolist(),
                "decoder_seed": int(decoder_seed),
                "architecture": architecture,
                "operations": operations,
                "effective_layers": int(architecture["effective_layers"]),
                "hidden_dimension": int(hp["hidden_dim"]),
                "gat_layer_count": int(resources["gat_layer_count"]),
                "heads": int(hp["gat_heads"]),
                "parameter_count": parameter_cache[architecture_key],
                "edge_activation_estimate_bytes": resources[
                    "edge_activation_estimate_bytes"
                ],
                "node_activation_estimate_bytes": resources[
                    "node_activation_estimate_bytes"
                ],
                "activation_resource_estimate_bytes": resources[
                    "activation_resource_estimate_bytes"
                ],
                "condition_mask_vector": mask,
                "hp": hp,
                "formal_pool_member": False,
            }
        )
    if not candidates:
        raise RuntimeError("decode-only search produced no valid out-of-pool candidates")
    max_layers = max(int(row["effective_layers"]) for row in candidates)
    if max_layers != MAX_OP_NODES:
        raise RuntimeError(
            f"decode-only search did not reach the five-layer boundary: {max_layers}"
        )
    joint = [
        row
        for row in candidates
        if int(row["hidden_dimension"]) == int(HIDDEN_DIM_OPTIONS[-1])
        and int(row["effective_layers"]) == MAX_OP_NODES
        and int(row["heads"]) == 1
    ]
    if not joint:
        raise RuntimeError("no decoded candidate reaches the formal joint boundary")

    selected: dict[str, dict[str, Any]] = {}

    def select(reason: str, key) -> None:
        row = max(joint, key=key)
        fingerprint = str(row["candidate_fingerprint"])
        if fingerprint not in selected:
            selected[fingerprint] = {**row, "selection_reasons": [reason]}
        else:
            selected[fingerprint]["selection_reasons"].append(reason)

    select(
        "max_gat_heavy_joint_boundary",
        lambda row: (
            int(row["gat_layer_count"]),
            int(row["activation_resource_estimate_bytes"]),
            int(row["parameter_count"]),
            str(row["candidate_fingerprint"]),
        ),
    )
    select(
        "max_parameter_count_joint_boundary",
        lambda row: (
            int(row["parameter_count"]),
            int(row["activation_resource_estimate_bytes"]),
            str(row["candidate_fingerprint"]),
        ),
    )
    select(
        "max_activation_resource_joint_boundary",
        lambda row: (
            int(row["activation_resource_estimate_bytes"]),
            int(row["gat_layer_count"]),
            int(row["parameter_count"]),
            str(row["candidate_fingerprint"]),
        ),
    )
    output = sorted(selected.values(), key=lambda row: row["candidate_fingerprint"])
    return output, pool_fingerprints


def _evaluator_args(
    *, output: Path,
    checkpoint: str,
    search_seed: int,
    dataset_context: dict[str, Any],
) -> SimpleNamespace:
    return SimpleNamespace(
        hp_mode=FORMAL_HP_MODE,
        z_bound=FORMAL_Z_BOUND,
        seed=int(search_seed),
        log_lr_min=-4.0,
        log_lr_max=-1.5,
        dropout_min=0.1,
        dropout_max=0.6,
        gcnii_alpha=0.1,
        gcnii_theta=0.5,
        eval_epochs=FORMAL_EVAL_EPOCHS,
        patience=FORMAL_PATIENCE,
        output=str(output),
        checkpoint=str(checkpoint),
        dataset_context=dataset_context,
        initial_selection_strategy="resource_preflight_only",
        method_label="resource_preflight_only",
    )


def _e2e_candidate(
    row: dict[str, Any],
    *,
    vae,
    data,
    in_ch: int,
    out_ch: int,
    args: SimpleNamespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started_at = _utc_now()
    started = time.monotonic()
    _z, result = bo_phase4.eval_candidate(
        vae,
        torch.tensor(row["canonical_z_search"], dtype=torch.float32),
        data,
        in_ch,
        out_ch,
        args,
        device,
        step=0,
        evaluation_stage="resource_preflight",
        evaluation_fidelity="full",
        candidate_fingerprint=row["candidate_fingerprint"],
    )
    torch.cuda.synchronize(device)
    wall_seconds = time.monotonic() - started
    actual_epochs = int(result.get("epochs_ran") or 0)
    return {
        "candidate_fingerprint": row["candidate_fingerprint"],
        "status": "completed" if bool(result.get("valid", False)) else "failed",
        "started_at": started_at,
        "ended_at": _utc_now(),
        "requested_epochs": FORMAL_EVAL_EPOCHS,
        "requested_patience": FORMAL_PATIENCE,
        "actual_epochs": actual_epochs,
        "best_epoch": result.get("best_epoch"),
        "early_stopping": actual_epochs < FORMAL_EVAL_EPOCHS,
        "candidate_evaluation_wall_seconds": float(wall_seconds),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "evaluation_seed": int(result["evaluation_seed"]),
        "decoder_seed": int(result["decoder_seed"]),
        "accuracy_recorded_or_used_for_selection": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--search-seed", type=int, default=5)
    parser.add_argument("--sobol-count", type=int, default=4096)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    process_started_at = _utc_now()
    process_started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if int(args.search_seed) not in FORMAL_POOL_SEEDS:
        raise ValueError("resource preflight search seed must be one of 5--9")
    if not torch.cuda.is_available():
        raise InfrastructureEvaluationError("CUDA is required for Flickr preflight")
    device = torch.device("cuda:0")
    pid = os.getpid()
    monitor = ProcessMemoryMonitor(pid)
    monitor.start()
    payload: dict[str, Any] = {
        "status": "failed",
        "dataset": "flickr",
        "training_mode": "full_batch",
        "source": SOURCE,
        "candidate_pool_index": None,
        "pid": pid,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unspecified"),
        "process_started_at": process_started_at,
        "eval_epochs": FORMAL_EVAL_EPOCHS,
        "patience": FORMAL_PATIENCE,
        "selection_uses_labels_or_accuracy": False,
        "accuracy_reported_or_used_for_method_selection": False,
        "formal_results_history_or_gp_written": False,
    }
    try:
        request = resolve_dataset_request("flickr", args.data_root)
        data_started = time.monotonic()
        bundle = load_dataset_from_request(request)
        validate_expected_dataset_manifest(
            bundle, args.expected_dataset_manifest
        )
        payload["data_loading_seconds"] = float(time.monotonic() - data_started)
        payload["dataset_context"] = bundle.context
        transfer_started = time.monotonic()
        bundle.to(device)
        torch.cuda.synchronize(device)
        payload["dataset_transfer_seconds"] = float(
            time.monotonic() - transfer_started
        )
        logger = logging.getLogger("flickr_continuous_extreme_preflight")
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        vae_started = time.monotonic()
        vae = bo_phase4.load_vae(
            SimpleNamespace(hp_mode=FORMAL_HP_MODE, checkpoint=args.checkpoint),
            device,
            logger,
        )
        payload["vae_loading_seconds"] = float(time.monotonic() - vae_started)
        decode_started = time.monotonic()
        candidates, pool_fingerprints = decode_continuous_extremes(
            vae,
            device=device,
            search_seed=int(args.search_seed),
            in_ch=bundle.num_features,
            out_ch=bundle.num_classes,
            num_nodes=bundle.num_nodes,
            num_edges=bundle.num_edges,
            sobol_count=int(args.sobol_count),
        )
        decode_seconds = time.monotonic() - decode_started
        decode_payload = {
            "status": "completed",
            "selection_uses_labels_or_accuracy": False,
            "continuous_bounds": {
                "architecture_latent": [-FORMAL_Z_BOUND, FORMAL_Z_BOUND],
                "hyperparameters_normalized": [0.0, 1.0],
                "arch_nz": bo_phase4.ARCH_NZ,
                "hp_mode": FORMAL_HP_MODE,
                "hp_dim": 4,
            },
            "decode_search": {
                "corner_count": 1 << bo_phase4.ARCH_NZ,
                "sobol_count": int(args.sobol_count),
                "decode_trials": 5,
                "decode_seconds": float(decode_seconds),
            },
            "formal_pool_fingerprints": pool_fingerprints,
            "formal_pool_size_total": len(FORMAL_POOL_SEEDS) * 768,
            "selected_candidate_count": len(candidates),
            "candidates": candidates,
            "minimal_cover_explanation": (
                "one candidate is used only when the GAT-heavy, parameter, and "
                "activation maxima share a fingerprint; otherwise their union is "
                "the smallest deterministic cover of conflicting decoded extrema"
            ),
        }
        atomic_json_dump(decode_payload, output / "decode_only_candidates.json")
        payload["decode_only"] = decode_payload

        probes: list[dict[str, Any]] = []
        for row in candidates:
            probe_started_at = _utc_now()
            try:
                probe = _probe_candidate(
                    {
                        **row,
                        "z_search": row["canonical_z_search"],
                        "attention_heads": row["heads"],
                    },
                    data=bundle.data,
                    in_ch=bundle.num_features,
                    out_ch=bundle.num_classes,
                    device=device,
                )
                probe["status"] = "completed"
                probe["started_at"] = probe_started_at
                probe["ended_at"] = _utc_now()
                probes.append(probe)
            except Exception as exc:
                probes.append(
                    {
                        "candidate_fingerprint": row["candidate_fingerprint"],
                        "status": "blocked" if "out of memory" in str(exc).lower() else "failed",
                        "started_at": probe_started_at,
                        "ended_at": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "peak_allocated_bytes": int(
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "peak_reserved_bytes": int(
                            torch.cuda.max_memory_reserved(device)
                        ),
                    }
                )
                raise
        payload["forward_backward_probes"] = probes
        dangerous = max(
            candidates,
            key=lambda row: (
                int(row["activation_resource_estimate_bytes"]),
                int(row["gat_layer_count"]),
                int(row["parameter_count"]),
                str(row["candidate_fingerprint"]),
            ),
        )
        evaluator_args = _evaluator_args(
            output=output,
            checkpoint=args.checkpoint,
            search_seed=int(args.search_seed),
            dataset_context=bundle.context,
        )
        payload["full_fidelity_candidate"] = _e2e_candidate(
            dangerous,
            vae=vae,
            data=bundle.data,
            in_ch=bundle.num_features,
            out_ch=bundle.num_classes,
            args=evaluator_args,
            device=device,
        )
        if payload["full_fidelity_candidate"]["status"] != "completed":
            raise RuntimeError("full-fidelity extreme candidate was invalid")
        payload["status"] = "completed"
        payload["oom"] = False
        payload["exit_code"] = 0
        payload["gpu"] = _gpu_identity(device)
        return payload
    except Exception as exc:
        is_oom = "out of memory" in str(exc).lower()
        payload["status"] = "blocked" if is_oom else "failed"
        payload["oom"] = bool(is_oom)
        payload["exit_code"] = 2
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["peak_allocated_bytes_at_failure"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        payload["peak_reserved_bytes_at_failure"] = int(
            torch.cuda.max_memory_reserved(device)
        )
        raise
    finally:
        monitor.stop()
        payload["nvml_or_nvidia_smi_peak_used_mib"] = int(
            monitor.peak_used_mib
        )
        payload["nvml_sample_count"] = int(monitor.sample_count)
        payload["nvml_monitor_error_count"] = int(monitor.error_count)
        payload["process_ended_at"] = _utc_now()
        payload["complete_process_wall_seconds"] = float(
            time.monotonic() - process_started
        )
        atomic_json_dump(payload, output / "continuous_extreme_preflight.json")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
