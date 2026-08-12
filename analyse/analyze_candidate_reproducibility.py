#!/usr/bin/env python3
"""Paired full-fidelity reproducibility diagnosis for fixed Phase4 candidates.

The coordinator reads only search validation histories, verifies candidate
identity across T1/C1, selects a deterministic ten-candidate manifest, and
launches each training repeat in a fresh Python subprocess.  The worker reuses
the persisted discrete architecture, effective hyperparameters, and candidate
evaluation seed; it never decodes a candidate and never requests test metrics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable, Sequence
import warnings


FORMAT_VERSION = 1
MAX_DIFFERENCE_FINGERPRINT = (
    "13070f8c82f634f641ff000b14f81d4323656cb635ba53e8a8c23d694bb349ed"
)
SHARED_BEST_FINGERPRINT = (
    "93248d66fe4355bb6d5d0df99cd3f7f92356bde7a4a062a250f8ba052e048481"
)
ANCHOR_FINGERPRINTS = (
    MAX_DIFFERENCE_FINGERPRINT,
    SHARED_BEST_FINGERPRINT,
)
EXPECTED_CANDIDATE_COUNT = 10
DEFAULT_REPEAT_COUNT = 3
STRICT_REPEAT_COUNT = 2
CUBLAS_STRICT_CONFIG = ":4096:8"
FORBIDDEN_TEST_FIELD_TOKENS = (
    "test_acc",
    "test_accuracy",
    "best_test",
    "test_mean",
    "test_std",
)
HP_FIELDS = (
    "lr",
    "dropout",
    "hidden_dim",
    "weight_decay",
    "gat_heads",
    "sage_aggr",
    "gin_eps",
    "gat_heads_by_layer",
    "sage_aggr_by_layer",
    "gin_eps_by_layer",
    "condition_mask_vector",
)
CSV_FIELDS = (
    "mode",
    "candidate_order",
    "candidate_fingerprint",
    "repeat_id",
    "validation_accuracy",
    "valid",
    "best_epoch",
    "stopped_epoch",
    "epochs_ran",
    "wall_time_seconds",
    "evaluation_seed",
    "python_hash_seed",
    "cublas_workspace_config",
    "deterministic_algorithms_enabled",
    "deterministic_algorithms_warn_only",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
    "cuda_visible_devices",
    "gpu_name",
    "exception_type",
    "exception_message",
    "warnings",
    "return_code",
    "worker_stderr",
)
CORA_RAW_FILES = (
    "ind.cora.x",
    "ind.cora.tx",
    "ind.cora.allx",
    "ind.cora.y",
    "ind.cora.ty",
    "ind.cora.ally",
    "ind.cora.graph",
    "ind.cora.test.index",
)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv_dump(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(CSV_FIELDS),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _forbidden_test_fields(record: dict[str, Any]) -> list[str]:
    forbidden: list[str] = []
    for key in record:
        lowered = str(key).lower()
        if any(token in lowered for token in FORBIDDEN_TEST_FIELD_TOKENS):
            forbidden.append(str(key))
    return sorted(forbidden)


def load_search_validation_history(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    history_path = Path(path).resolve()
    if not history_path.is_file():
        raise FileNotFoundError(f"history does not exist: {history_path}")
    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"history must contain a JSON list: {history_path}")
    records: list[dict[str, Any]] = []
    for position, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(
                f"history row {position} is not a JSON object: {history_path}"
            )
        forbidden = _forbidden_test_fields(raw)
        if forbidden:
            raise ValueError(
                f"history row {position} contains forbidden test fields "
                f"{forbidden}; this diagnostic accepts search validation only"
            )
        records.append(raw)
    return records


def _candidate_fingerprint(z_search: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(
        np.asarray(z_search, dtype=np.float32).reshape(1, -1)
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _effective_hp(record: dict[str, Any]) -> dict[str, Any]:
    if "l2" in record:
        weight_decay = record["l2"]
    elif "weight_decay" in record:
        weight_decay = record["weight_decay"]
    else:
        raise ValueError("candidate record is missing l2/weight_decay")
    hp = {
        "lr": float(record["lr"]),
        "dropout": float(record["dropout"]),
        "hidden_dim": int(record["hidden_dim"]),
        "weight_decay": float(weight_decay),
        "gat_heads": int(record.get("gat_heads", 1)),
        "sage_aggr": str(record.get("sage_aggr", "mean")),
        "gin_eps": float(record.get("gin_eps", 0.0)),
        "gat_heads_by_layer": record.get("gat_heads_by_layer"),
        "sage_aggr_by_layer": record.get("sage_aggr_by_layer"),
        "gin_eps_by_layer": record.get("gin_eps_by_layer"),
        "condition_mask_vector": record.get("condition_mask_vector"),
    }
    return hp


def _architecture_fingerprint(
    operations: Sequence[Any],
    edges: Sequence[Sequence[Any]],
) -> str:
    payload = {
        "operations": [str(value) for value in operations],
        "edges": sorted(
            [list(map(int, edge)) for edge in edges]
        ),
    }
    return _canonical_json_sha256(payload)


def _hp_fingerprint(hp: dict[str, Any]) -> str:
    return _canonical_json_sha256({field: hp.get(field) for field in HP_FIELDS})


def _full_evaluation_index(record: dict[str, Any]) -> int:
    value = record.get("full_evaluation_index", record.get("step"))
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("candidate record has no integer full evaluation index")
    return int(value)


def normalize_candidate_record(
    record: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    if record.get("evaluation_fidelity") != "full":
        raise ValueError(f"{source} record is not full fidelity")
    fingerprint = record.get("candidate_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(f"{source} candidate fingerprint is missing or invalid")
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise ValueError(
            f"{source} candidate fingerprint is not hexadecimal"
        ) from exc
    z_search = record.get("z_search")
    if not isinstance(z_search, list) or not z_search:
        raise ValueError(f"{source} candidate has no z_search")
    computed_candidate = _candidate_fingerprint(z_search)
    if computed_candidate != fingerprint:
        raise ValueError(
            f"{source} candidate fingerprint does not match canonical z_search: "
            f"expected {computed_candidate}, got {fingerprint}"
        )
    operations = record.get("operations")
    edges = record.get("edges")
    if not isinstance(operations, list) or not operations:
        raise ValueError(f"{source} candidate has no decoded operations")
    if not isinstance(edges, list):
        raise ValueError(f"{source} candidate has invalid edges")
    hp = _effective_hp(record)
    architecture_fingerprint = _architecture_fingerprint(operations, edges)
    stored_architecture_fingerprint = record.get("architecture_fingerprint")
    if stored_architecture_fingerprint != architecture_fingerprint:
        raise ValueError(
            f"{source} architecture fingerprint mismatch for {fingerprint}"
        )
    hp_fingerprint = _hp_fingerprint(hp)
    stored_hp_fingerprint = record.get("hp_fingerprint")
    if stored_hp_fingerprint != hp_fingerprint:
        raise ValueError(f"{source} HP fingerprint mismatch for {fingerprint}")
    evaluation_seed = record.get("evaluation_seed")
    decoder_seed = record.get("decoder_seed")
    search_seed = record.get("search_seed")
    for field, value in (
        ("evaluation_seed", evaluation_seed),
        ("decoder_seed", decoder_seed),
        ("search_seed", search_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{source} {field} is not an integer for {fingerprint}"
            )
    return {
        "candidate_fingerprint": fingerprint,
        "z_search": [float(value) for value in z_search],
        "operations": [str(value) for value in operations],
        "edges": [list(map(int, edge)) for edge in edges],
        "effective_layers": sum(
            str(operation) != "Identity" for operation in operations
        ),
        "hp": hp,
        "architecture_fingerprint": architecture_fingerprint,
        "hp_fingerprint": hp_fingerprint,
        "decoder_seed": int(decoder_seed),
        "evaluation_seed": int(evaluation_seed),
        "search_seed": int(search_seed),
        "candidate_evaluation_seed_scheme": record.get(
            "candidate_evaluation_seed_scheme"
        ),
        "seed_derivation": record.get("seed_derivation"),
        "full_evaluation_index": _full_evaluation_index(record),
        "validation_accuracy": float(record["val_acc"]),
        "best_epoch": record.get("best_epoch"),
        "stopped_epoch": record.get("stopped_epoch"),
        "epochs_ran": record.get("epochs_ran"),
    }


def _candidate_identity(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "candidate_fingerprint",
            "z_search",
            "operations",
            "edges",
            "effective_layers",
            "hp",
            "architecture_fingerprint",
            "hp_fingerprint",
            "decoder_seed",
            "evaluation_seed",
            "search_seed",
            "candidate_evaluation_seed_scheme",
            "seed_derivation",
        )
    }


def index_full_candidates(
    records: Sequence[dict[str, Any]],
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        if record.get("evaluation_fidelity") != "full":
            continue
        candidate = normalize_candidate_record(
            record,
            source=f"{source} history row {position}",
        )
        fingerprint = candidate["candidate_fingerprint"]
        prior = indexed.get(fingerprint)
        if prior is None:
            indexed[fingerprint] = candidate
            continue
        if _candidate_identity(prior) != _candidate_identity(candidate):
            raise ValueError(
                f"{source} contains inconsistent duplicate identity for "
                f"{fingerprint}"
            )
        if candidate["full_evaluation_index"] < prior["full_evaluation_index"]:
            indexed[fingerprint] = candidate
    return indexed


def _equidistant_positions(item_count: int, selection_count: int) -> list[int]:
    if selection_count <= 0:
        return []
    if item_count < selection_count:
        raise ValueError(
            f"cannot choose {selection_count} positions from {item_count} items"
        )
    if selection_count == 1:
        return [0]
    denominator = selection_count - 1
    positions = [
        (index * (item_count - 1) + denominator // 2) // denominator
        for index in range(selection_count)
    ]
    if len(set(positions)) != selection_count:
        raise RuntimeError("equidistant selection did not produce unique positions")
    return positions


def build_candidate_manifest(
    t1_history_path: str | os.PathLike[str],
    c1_history_path: str | os.PathLike[str],
) -> dict[str, Any]:
    t1_path = Path(t1_history_path).resolve()
    c1_path = Path(c1_history_path).resolve()
    t1 = index_full_candidates(
        load_search_validation_history(t1_path),
        source="T1",
    )
    c1 = index_full_candidates(
        load_search_validation_history(c1_path),
        source="C1",
    )
    overlap = sorted(set(t1).intersection(c1))
    for fingerprint in overlap:
        if _candidate_identity(t1[fingerprint]) != _candidate_identity(
            c1[fingerprint]
        ):
            raise ValueError(
                "T1/C1 identity or seed mismatch for overlapping candidate "
                f"{fingerprint}"
            )
    missing_anchors = [
        fingerprint
        for fingerprint in ANCHOR_FINGERPRINTS
        if fingerprint not in overlap
    ]
    if missing_anchors:
        raise ValueError(
            f"required anchor candidates are missing from T1/C1 overlap: "
            f"{missing_anchors}"
        )
    remaining = [
        fingerprint
        for fingerprint in overlap
        if fingerprint not in ANCHOR_FINGERPRINTS
    ]
    extra_count = EXPECTED_CANDIDATE_COUNT - len(ANCHOR_FINGERPRINTS)
    positions = _equidistant_positions(len(remaining), extra_count)
    selected = list(ANCHOR_FINGERPRINTS) + [
        remaining[position] for position in positions
    ]
    if len(selected) != EXPECTED_CANDIDATE_COUNT or len(set(selected)) != len(
        selected
    ):
        raise RuntimeError("candidate manifest selection is not ten unique rows")
    candidates: list[dict[str, Any]] = []
    for order, fingerprint in enumerate(selected):
        t1_row = t1[fingerprint]
        c1_row = c1[fingerprint]
        candidate = {
            **_candidate_identity(t1_row),
            "candidate_order": order,
            "historical": {
                "t1_validation_accuracy": t1_row["validation_accuracy"],
                "c1_validation_accuracy": c1_row["validation_accuracy"],
                "signed_c1_minus_t1": (
                    c1_row["validation_accuracy"]
                    - t1_row["validation_accuracy"]
                ),
                "absolute_difference": abs(
                    c1_row["validation_accuracy"]
                    - t1_row["validation_accuracy"]
                ),
                "t1_full_evaluation_index": t1_row[
                    "full_evaluation_index"
                ],
                "c1_full_evaluation_index": c1_row[
                    "full_evaluation_index"
                ],
                "t1_best_epoch": t1_row["best_epoch"],
                "c1_best_epoch": c1_row["best_epoch"],
                "t1_stopped_epoch": t1_row["stopped_epoch"],
                "c1_stopped_epoch": c1_row["stopped_epoch"],
                "t1_epochs_ran": t1_row["epochs_ran"],
                "c1_epochs_ran": c1_row["epochs_ran"],
            },
        }
        candidates.append(candidate)
    historical_differences = [
        {
            "candidate_fingerprint": fingerprint,
            "signed_c1_minus_t1": (
                c1[fingerprint]["validation_accuracy"]
                - t1[fingerprint]["validation_accuracy"]
            ),
        }
        for fingerprint in overlap
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "selection_algorithm": (
            "two_required_anchors_then_8_half_up_equidistant_positions_over_"
            "sorted_remaining_overlap_fingerprints"
        ),
        "t1_history": {
            "path": str(t1_path),
            "sha256": _sha256_file(t1_path),
            "full_unique_candidate_count": len(t1),
        },
        "c1_history": {
            "path": str(c1_path),
            "sha256": _sha256_file(c1_path),
            "full_unique_candidate_count": len(c1),
        },
        "overlap_full_unique_candidate_count": len(overlap),
        "required_anchor_fingerprints": list(ANCHOR_FINGERPRINTS),
        "equidistant_source_count": len(remaining),
        "equidistant_positions": positions,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "historical_overlap_differences": historical_differences,
    }
    manifest["manifest_sha256"] = _canonical_json_sha256(manifest)
    return manifest


def _metadata_checkpoint(results_directory: Path) -> tuple[Path, str]:
    path = results_directory / "candidate_pool_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"candidate pool metadata is missing: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    checkpoint = payload.get("vae_checkpoint")
    checkpoint_sha256 = payload.get("vae_checkpoint_sha256")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"vae_checkpoint is missing from {path}")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
    ):
        raise ValueError(f"vae_checkpoint_sha256 is missing from {path}")
    return Path(checkpoint).expanduser().resolve(), checkpoint_sha256


def resolve_checkpoint(
    t1_results: Path,
    c1_results: Path,
    explicit_checkpoint: str | None,
) -> Path:
    t1_checkpoint, t1_sha = _metadata_checkpoint(t1_results)
    c1_checkpoint, c1_sha = _metadata_checkpoint(c1_results)
    if t1_checkpoint != c1_checkpoint or t1_sha != c1_sha:
        raise ValueError("T1/C1 checkpoint provenance differs")
    checkpoint = (
        t1_checkpoint
        if explicit_checkpoint is None
        else Path(explicit_checkpoint).expanduser().resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    actual_sha = _sha256_file(checkpoint)
    if actual_sha != t1_sha:
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected {t1_sha}, got {actual_sha}"
        )
    return checkpoint


def _parse_csv_row(raw_line: str) -> list[str]:
    return next(csv.reader([raw_line]))


def query_gpu_state(
    gpu_index: int,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(int(gpu_index)),
            "--query-gpu=index,uuid,name,driver_version,memory.used,"
            "utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if query.returncode != 0:
        raise RuntimeError(
            "nvidia-smi GPU query failed: "
            f"{query.stderr.strip() or query.stdout.strip()}"
        )
    rows = [line for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"nvidia-smi returned {len(rows)} rows for GPU {gpu_index}"
        )
    values = [value.strip() for value in _parse_csv_row(rows[0])]
    if len(values) != 6:
        raise RuntimeError(f"unparseable nvidia-smi GPU row: {rows[0]!r}")
    index, uuid, name, driver, memory_used, utilization = values
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if compute.returncode != 0:
        raise RuntimeError(
            "nvidia-smi compute-process query failed: "
            f"{compute.stderr.strip() or compute.stdout.strip()}"
        )
    processes: list[dict[str, Any]] = []
    for line in compute.stdout.splitlines():
        if not line.strip():
            continue
        parts = [value.strip() for value in _parse_csv_row(line)]
        if len(parts) != 4 or parts[0] != uuid:
            continue
        processes.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_gpu_memory_mib": int(parts[3]),
            }
        )
    return {
        "index": int(index),
        "uuid": uuid,
        "name": name,
        "driver_version": driver,
        "memory_used_mib": int(memory_used),
        "utilization_percent": int(utilization),
        "compute_processes": processes,
        "idle": not processes,
    }


def _strict_backend_settings(torch_module: Any) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_STRICT_CONFIG:
        raise RuntimeError(
            "strict worker requires CUBLAS_WORKSPACE_CONFIG="
            f"{CUBLAS_STRICT_CONFIG} before Torch import"
        )
    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False


def _backend_snapshot(torch_module: Any) -> dict[str, Any]:
    warn_only_getter = getattr(
        torch_module,
        "is_deterministic_algorithms_warn_only_enabled",
        None,
    )
    return {
        "deterministic_algorithms_enabled": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            None if warn_only_getter is None else bool(warn_only_getter())
        ),
        "cudnn_deterministic": bool(
            torch_module.backends.cudnn.deterministic
        ),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch_module.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(
            torch_module.backends.cudnn.allow_tf32
        ),
    }


def _load_cora_without_download(
    cora_root: str,
    *,
    device: Any,
) -> tuple[Any, int, int]:
    root = Path(cora_root).expanduser().resolve()
    raw_directory = root / "raw"
    missing = [
        name
        for name in CORA_RAW_FILES
        if not (raw_directory / name).is_file()
        or (raw_directory / name).stat().st_size < 10
    ]
    if missing:
        raise FileNotFoundError(
            f"Cora raw files are missing under {raw_directory}; downloads are "
            f"disabled for this diagnostic: {missing}"
        )
    import torch_geometric.transforms as transforms
    from torch_geometric.datasets import Planetoid

    pyg_root = root.parent if root.name == "Cora" else root
    dataset = Planetoid(
        root=str(pyg_root),
        name="Cora",
        transform=transforms.NormalizeFeatures(),
    )
    return (
        dataset[0].to(device),
        int(dataset.num_features),
        int(dataset.num_classes),
    )


def run_worker(spec_path: Path, output_path: Path) -> int:
    with spec_path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    result: dict[str, Any] = {
        "mode": spec["mode"],
        "candidate_order": int(spec["candidate"]["candidate_order"]),
        "candidate_fingerprint": spec["candidate"][
            "candidate_fingerprint"
        ],
        "repeat_id": int(spec["repeat_id"]),
        "validation_accuracy": None,
        "valid": False,
        "best_epoch": None,
        "stopped_epoch": None,
        "epochs_ran": 0,
        "wall_time_seconds": None,
        "evaluation_seed": int(spec["candidate"]["evaluation_seed"]),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": None,
        "exception_type": "",
        "exception_message": "",
        "warnings": [],
        "return_code": 0,
        "worker_stderr": "",
    }
    try:
        import torch

        if spec["mode"] == "strict":
            _strict_backend_settings(torch)
        result.update(_backend_snapshot(torch))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the worker process")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "worker expected exactly one visible GPU after "
                "CUDA_VISIBLE_DEVICES isolation"
            )
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        result["gpu_name"] = torch.cuda.get_device_name(device)
        data, in_ch, out_ch = _load_cora_without_download(
            spec["cora_root"],
            device=device,
        )
        from eval_utils import train_and_eval_arch

        candidate = spec["candidate"]
        hp = candidate["hp"]
        config = {
            "operations": list(candidate["operations"]),
            "edges": [tuple(edge) for edge in candidate["edges"]],
            "effective_layers": int(candidate["effective_layers"]),
        }
        torch.cuda.synchronize(device)
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            train_result = train_and_eval_arch(
                config=config,
                data=data,
                in_ch=in_ch,
                out_ch=out_ch,
                lr=float(hp["lr"]),
                dropout=float(hp["dropout"]),
                hidden_dim=int(hp["hidden_dim"]),
                weight_decay=float(hp["weight_decay"]),
                gcnii_alpha=float(spec["gcnii_alpha"]),
                gcnii_theta=float(spec["gcnii_theta"]),
                gat_heads=int(hp["gat_heads"]),
                sage_aggr=str(hp["sage_aggr"]),
                gin_eps=float(hp["gin_eps"]),
                gat_heads_by_layer=hp.get("gat_heads_by_layer"),
                sage_aggr_by_layer=hp.get("sage_aggr_by_layer"),
                gin_eps_by_layer=hp.get("gin_eps_by_layer"),
                device=device,
                max_epochs=int(spec["eval_epochs"]),
                patience=int(spec["patience"]),
                seed=int(candidate["evaluation_seed"]),
                track_test=False,
                return_metadata=True,
            )
        torch.cuda.synchronize(device)
        result["wall_time_seconds"] = float(time.monotonic() - started)
        validation_accuracy, valid, metadata = train_result
        result.update(
            {
                "validation_accuracy": float(validation_accuracy),
                "valid": bool(valid),
                "best_epoch": metadata.get("best_epoch"),
                "stopped_epoch": metadata.get("stopped_epoch"),
                "epochs_ran": int(metadata.get("epochs_ran") or 0),
                "warnings": [
                    f"{warning.category.__name__}: {warning.message}"
                    for warning in captured
                ],
            }
        )
        if not bool(valid):
            result["exception_type"] = "InvalidTrainingEvaluation"
            result["exception_message"] = (
                "; ".join(result["warnings"])
                or "train_and_eval_arch returned valid=False"
            )
    except BaseException as exc:
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["return_code"] = 1
    _atomic_json_dump(result, output_path)
    return int(result["return_code"])


def _worker_spec(
    candidate: dict[str, Any],
    *,
    mode: str,
    repeat_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "mode": mode,
        "repeat_id": int(repeat_id),
        "candidate": candidate,
        "cora_root": str(Path(args.cora_root).expanduser().resolve()),
        "eval_epochs": int(args.eval_epochs),
        "patience": int(args.patience),
        "gcnii_alpha": float(args.gcnii_alpha),
        "gcnii_theta": float(args.gcnii_theta),
    }


def _run_one_subprocess(
    candidate: dict[str, Any],
    *,
    mode: str,
    repeat_id: int,
    args: argparse.Namespace,
    output_directory: Path,
    repo_root: Path,
) -> dict[str, Any]:
    gpu_state = query_gpu_state(int(args.gpu_index), repo_root=repo_root)
    if not gpu_state["idle"]:
        raise RuntimeError(
            f"GPU {args.gpu_index} became occupied; stop before launching "
            f"candidate {candidate['candidate_fingerprint']}: "
            f"{gpu_state['compute_processes']}"
        )
    worker_directory = output_directory / "worker_records"
    worker_directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{mode}_{int(candidate['candidate_order']):02d}_"
        f"{candidate['candidate_fingerprint'][:12]}_r{repeat_id:02d}"
    )
    spec_path = worker_directory / f"{stem}_spec.json"
    result_path = worker_directory / f"{stem}_result.json"
    _atomic_json_dump(
        _worker_spec(
            candidate,
            mode=mode,
            repeat_id=repeat_id,
            args=args,
        ),
        spec_path,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_index))
    if mode == "strict":
        environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_STRICT_CONFIG
        environment["PYTHONHASHSEED"] = str(
            int(candidate["evaluation_seed"])
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-spec",
        str(spec_path),
        "--worker-output",
        str(result_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if not result_path.is_file():
        raise RuntimeError(
            f"worker did not create {result_path}; return_code="
            f"{completed.returncode}, stderr={completed.stderr.strip()}"
        )
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    result["return_code"] = int(completed.returncode)
    result["worker_stderr"] = completed.stderr.strip()
    _atomic_json_dump(result, result_path)
    return result


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = float(probability) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def repeat_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    failures = 0
    wall_times: list[float] = []
    for row in rows:
        fingerprint = str(row["candidate_fingerprint"])
        grouped.setdefault(fingerprint, []).append(row)
        if (
            int(row.get("return_code") or 0) != 0
            or not bool(row.get("valid"))
            or row.get("validation_accuracy") is None
        ):
            failures += 1
        elif row.get("wall_time_seconds") is not None:
            wall_times.append(float(row["wall_time_seconds"]))

    comparisons: list[float] = []
    signed: list[float] = []
    per_candidate: list[dict[str, Any]] = []
    best_epoch_matches = 0
    epochs_ran_matches = 0
    comparable_candidates = 0
    exact_comparisons = 0
    for fingerprint in sorted(grouped):
        candidate_rows = sorted(
            grouped[fingerprint],
            key=lambda row: int(row["repeat_id"]),
        )
        successful = [
            row
            for row in candidate_rows
            if int(row.get("return_code") or 0) == 0
            and bool(row.get("valid"))
            and row.get("validation_accuracy") is not None
        ]
        values = [
            float(row["validation_accuracy"]) for row in successful
        ]
        if len(successful) >= 2:
            comparable_candidates += 1
            baseline = values[0]
            for value in values[1:]:
                difference = value - baseline
                signed.append(difference)
                comparisons.append(abs(difference))
                if difference == 0.0:
                    exact_comparisons += 1
            if len({row.get("best_epoch") for row in successful}) == 1:
                best_epoch_matches += 1
            if len({row.get("epochs_ran") for row in successful}) == 1:
                epochs_ran_matches += 1
        per_candidate.append(
            {
                "candidate_fingerprint": fingerprint,
                "successful_repeats": len(successful),
                "requested_repeats": len(candidate_rows),
                "validation_range": (
                    None if not values else max(values) - min(values)
                ),
                "validation_standard_deviation": (
                    None
                    if not values
                    else statistics.pstdev(values)
                ),
                "best_epoch_values": [
                    row.get("best_epoch") for row in successful
                ],
                "epochs_ran_values": [
                    row.get("epochs_ran") for row in successful
                ],
            }
        )
    return {
        "row_count": len(rows),
        "candidate_count": len(grouped),
        "failure_count": failures,
        "comparison_count": len(comparisons),
        "exact_validation_comparison_ratio": (
            None
            if not comparisons
            else exact_comparisons / len(comparisons)
        ),
        "validation_mae": (
            None if not comparisons else statistics.fmean(comparisons)
        ),
        "validation_max_absolute_difference": (
            None if not comparisons else max(comparisons)
        ),
        "validation_p90_absolute_difference": _percentile(
            comparisons, 0.90
        ),
        "validation_signed_mean_difference": (
            None if not signed else statistics.fmean(signed)
        ),
        "best_epoch_match_rate": (
            None
            if comparable_candidates == 0
            else best_epoch_matches / comparable_candidates
        ),
        "epochs_ran_match_rate": (
            None
            if comparable_candidates == 0
            else epochs_ran_matches / comparable_candidates
        ),
        "mean_wall_time_seconds": (
            None if not wall_times else statistics.fmean(wall_times)
        ),
        "per_candidate": per_candidate,
    }


def historical_overlap_statistics(manifest: dict[str, Any]) -> dict[str, Any]:
    signed = [
        float(row["signed_c1_minus_t1"])
        for row in manifest["historical_overlap_differences"]
    ]
    absolute = [abs(value) for value in signed]
    return {
        "candidate_count": len(signed),
        "exact_validation_ratio": (
            None
            if not signed
            else sum(value == 0.0 for value in signed) / len(signed)
        ),
        "validation_mae": (
            None if not absolute else statistics.fmean(absolute)
        ),
        "validation_max_absolute_difference": (
            None if not absolute else max(absolute)
        ),
        "validation_p90_absolute_difference": _percentile(absolute, 0.90),
        "validation_signed_mean_c1_minus_t1": (
            None if not signed else statistics.fmean(signed)
        ),
    }


def _strict_pilot_succeeded(
    rows: Sequence[dict[str, Any]],
) -> tuple[bool, str]:
    expected = {
        (fingerprint, repeat_id)
        for fingerprint in ANCHOR_FINGERPRINTS
        for repeat_id in range(STRICT_REPEAT_COUNT)
    }
    actual = {
        (str(row["candidate_fingerprint"]), int(row["repeat_id"]))
        for row in rows
    }
    if actual != expected:
        return False, "strict pilot did not complete all four required repeats"
    for row in rows:
        if int(row.get("return_code") or 0) != 0 or not bool(row.get("valid")):
            return (
                False,
                "strict pilot contains a failed/invalid worker: "
                f"{row.get('exception_type')}: "
                f"{row.get('exception_message')}",
            )
    for fingerprint in ANCHOR_FINGERPRINTS:
        candidate_rows = sorted(
            (
                row
                for row in rows
                if row["candidate_fingerprint"] == fingerprint
            ),
            key=lambda row: int(row["repeat_id"]),
        )
        identity = {
            (
                row["validation_accuracy"],
                row.get("best_epoch"),
                row.get("stopped_epoch"),
                row.get("epochs_ran"),
            )
            for row in candidate_rows
        }
        if len(identity) != 1:
            return (
                False,
                f"strict pilot is not exactly repeatable for {fingerprint}",
            )
    return True, ""


def build_summary(
    manifest: dict[str, Any],
    default_rows: Sequence[dict[str, Any]],
    strict_rows: Sequence[dict[str, Any]],
    *,
    strict_pilot_succeeded: bool,
    strict_pilot_reason: str,
    strict_expanded: bool,
) -> dict[str, Any]:
    default_stats = repeat_statistics(default_rows)
    strict_stats = repeat_statistics(strict_rows)
    historical = historical_overlap_statistics(manifest)
    default_by_fingerprint = {
        row["candidate_fingerprint"]: row
        for row in default_stats["per_candidate"]
    }
    max_candidate = default_by_fingerprint.get(
        MAX_DIFFERENCE_FINGERPRINT
    )
    drift_with_epoch_divergence = []
    for row in default_stats["per_candidate"]:
        if (
            row["validation_range"] is not None
            and row["validation_range"] > 0.0
            and (
                len(set(row["best_epoch_values"])) > 1
                or len(set(row["epochs_ran_values"])) > 1
            )
        ):
            drift_with_epoch_divergence.append(
                row["candidate_fingerprint"]
            )
    strict_mean = strict_stats["mean_wall_time_seconds"]
    default_mean = default_stats["mean_wall_time_seconds"]
    return {
        "format_version": FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "historical_t1_c1_overlap": historical,
        "default": default_stats,
        "strict": {
            **strict_stats,
            "pilot_succeeded": bool(strict_pilot_succeeded),
            "pilot_failure_reason": strict_pilot_reason,
            "expanded_to_all_candidates": bool(strict_expanded),
            "supported": (
                bool(strict_pilot_succeeded)
                and strict_stats["failure_count"] == 0
            ),
            "all_completed_candidate_repeats_exact": (
                strict_stats["exact_validation_comparison_ratio"] == 1.0
                and strict_stats["best_epoch_match_rate"] == 1.0
                and strict_stats["epochs_ran_match_rate"] == 1.0
            ),
        },
        "comparison": {
            "default_mae_minus_historical_mae": (
                None
                if default_stats["validation_mae"] is None
                or historical["validation_mae"] is None
                else default_stats["validation_mae"]
                - historical["validation_mae"]
            ),
            "default_max_minus_historical_max": (
                None
                if default_stats[
                    "validation_max_absolute_difference"
                ]
                is None
                or historical["validation_max_absolute_difference"] is None
                else default_stats[
                    "validation_max_absolute_difference"
                ]
                - historical["validation_max_absolute_difference"]
            ),
            "maximum_historical_difference_candidate_default_range": (
                None
                if max_candidate is None
                else max_candidate["validation_range"]
            ),
            "default_drift_with_best_epoch_or_early_stop_divergence": (
                drift_with_epoch_divergence
            ),
            "strict_to_default_mean_wall_time_ratio": (
                None
                if strict_mean is None
                or default_mean is None
                or default_mean <= 0.0
                else strict_mean / default_mean
            ),
        },
        "seed_identity_audit": {
            "t1_c1_identity_or_seed_mismatch_count": 0,
            "seed_leakage_evidence": False,
            "scope": (
                "candidate identity and recorded decoder/evaluation seed "
                "agreement only; runtime drift diagnosis does not alter the "
                "existing SHA-256 seed scheme"
            ),
        },
    }


def _default_output_directory(repo_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        repo_root
        / "results"
        / "diagnostics"
        / f"phase4_repro_seed0_{timestamp}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-candidate, fresh-process Phase4 validation "
            "reproducibility diagnostics"
        )
    )
    parser.add_argument(
        "--t1-results",
        default="results/phase4_t1_dev_global4_seed0",
    )
    parser.add_argument(
        "--c1-results",
        default="results/phase4_c1_dev_global4_seed0",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cora-root", default="/tmp/Cora")
    parser.add_argument("--output", default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--eval-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--gcnii-alpha", type=float, default=0.1)
    parser.add_argument("--gcnii-theta", type=float, default=0.5)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate histories and print the ten fixed fingerprints without GPU work",
    )
    parser.add_argument("--worker-spec", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.worker_spec is None) != (args.worker_output is None):
        parser.error("--worker-spec and --worker-output must be provided together")
    if args.eval_epochs <= 0 or args.patience <= 0:
        parser.error("--eval-epochs and --patience must be positive")
    if args.gpu_index < 0:
        parser.error("--gpu-index must be non-negative")
    return args


def coordinator_main(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    t1_results = Path(args.t1_results).expanduser()
    c1_results = Path(args.c1_results).expanduser()
    if not t1_results.is_absolute():
        t1_results = repo_root / t1_results
    if not c1_results.is_absolute():
        c1_results = repo_root / c1_results
    t1_results = t1_results.resolve()
    c1_results = c1_results.resolve()
    manifest = build_candidate_manifest(
        t1_results / "history_final.json",
        c1_results / "history_final.json",
    )
    checkpoint = resolve_checkpoint(
        t1_results,
        c1_results,
        args.checkpoint,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "candidate_manifest_sha256": manifest[
                        "manifest_sha256"
                    ],
                    "checkpoint": str(checkpoint),
                    "candidate_fingerprints": [
                        row["candidate_fingerprint"]
                        for row in manifest["candidates"]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    initial_gpu_state = query_gpu_state(
        int(args.gpu_index),
        repo_root=repo_root,
    )
    if not initial_gpu_state["idle"]:
        raise RuntimeError(
            f"GPU {args.gpu_index} is occupied: "
            f"{initial_gpu_state['compute_processes']}. Re-run the same "
            "command after it becomes idle; no training was started."
        )
    output_directory = (
        _default_output_directory(repo_root)
        if args.output is None
        else Path(args.output).expanduser()
    )
    if not output_directory.is_absolute():
        output_directory = repo_root / output_directory
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(
            f"diagnostic output already exists; choose a new path: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    _atomic_json_dump(manifest, output_directory / "candidate_manifest.json")

    sys.path.insert(0, str(repo_root))
    from eval_utils import SEED_DERIVATION
    from run_provenance import collect_run_provenance

    provenance = collect_run_provenance(
        repo_root=repo_root,
        checkpoint_path=checkpoint,
        argv=sys.argv,
        seed_derivation_version=SEED_DERIVATION,
        source_paths=(
            "analyse/analyze_candidate_reproducibility.py",
            "eval_utils.py",
            "hp_modes.py",
            "models.py",
            "run_provenance.py",
        ),
    )
    provenance["diagnostic_gpu_initial_state"] = initial_gpu_state
    provenance["diagnostic_profile"] = {
        "default_repeats_per_candidate": DEFAULT_REPEAT_COUNT,
        "strict_repeats_per_candidate": STRICT_REPEAT_COUNT,
        "eval_epochs": int(args.eval_epochs),
        "patience": int(args.patience),
        "gpu_index": int(args.gpu_index),
        "strict_settings": {
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_STRICT_CONFIG,
            "PYTHONHASHSEED": "<candidate evaluation seed>",
            "torch.use_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    _atomic_json_dump(
        provenance,
        output_directory / "runtime_provenance.json",
    )

    default_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    for candidate in manifest["candidates"]:
        for repeat_id in range(DEFAULT_REPEAT_COUNT):
            default_rows.append(
                _run_one_subprocess(
                    candidate,
                    mode="default",
                    repeat_id=repeat_id,
                    args=args,
                    output_directory=output_directory,
                    repo_root=repo_root,
                )
            )
            _atomic_csv_dump(
                default_rows,
                output_directory / "default_repeats.csv",
            )

    candidates_by_fingerprint = {
        row["candidate_fingerprint"]: row
        for row in manifest["candidates"]
    }
    strict_stopped = False
    for fingerprint in ANCHOR_FINGERPRINTS:
        candidate = candidates_by_fingerprint[fingerprint]
        for repeat_id in range(STRICT_REPEAT_COUNT):
            row = _run_one_subprocess(
                candidate,
                mode="strict",
                repeat_id=repeat_id,
                args=args,
                output_directory=output_directory,
                repo_root=repo_root,
            )
            strict_rows.append(row)
            _atomic_csv_dump(
                strict_rows,
                output_directory / "strict_repeats.csv",
            )
            if int(row.get("return_code") or 0) != 0 or not bool(
                row.get("valid")
            ):
                strict_stopped = True
                break
        if strict_stopped:
            break
    pilot_succeeded, pilot_reason = _strict_pilot_succeeded(strict_rows)
    strict_expanded = False
    if pilot_succeeded:
        strict_expanded = True
        strict_stopped = False
        for candidate in manifest["candidates"]:
            if candidate["candidate_fingerprint"] in ANCHOR_FINGERPRINTS:
                continue
            for repeat_id in range(STRICT_REPEAT_COUNT):
                row = _run_one_subprocess(
                    candidate,
                    mode="strict",
                    repeat_id=repeat_id,
                    args=args,
                    output_directory=output_directory,
                    repo_root=repo_root,
                )
                strict_rows.append(row)
                _atomic_csv_dump(
                    strict_rows,
                    output_directory / "strict_repeats.csv",
                )
                if int(row.get("return_code") or 0) != 0 or not bool(
                    row.get("valid")
                ):
                    strict_stopped = True
                    break
            if strict_stopped:
                break
    summary = build_summary(
        manifest,
        default_rows,
        strict_rows,
        strict_pilot_succeeded=pilot_succeeded,
        strict_pilot_reason=pilot_reason,
        strict_expanded=strict_expanded,
    )
    _atomic_json_dump(
        summary,
        output_directory / "reproducibility_summary.json",
    )
    print(str(output_directory))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_spec is not None:
        return run_worker(
            Path(args.worker_spec).resolve(),
            Path(args.worker_output).resolve(),
        )
    return coordinator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
