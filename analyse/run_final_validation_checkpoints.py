#!/usr/bin/env python3
"""Train and atomically freeze validation-best GNN checkpoints from a sealed view."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import freeze_final_validation_checkpoints as plan
from deterministic_runtime import assert_strict_torch_determinism
from eval_utils import DynamicGNN, isolated_rng


MAX_EPOCHS = 300
PATIENCE = 80
MAX_SYSTEM_RETRIES = 2
CHECKPOINT_PAYLOAD_KEYS = {
    "format_version", "pretraining_immutable_root", "protocol_sha256", "role",
    "strategy", "method_name", "pair_id", "seed", "joint_config_fingerprint",
    "candidate", "dataset_content_identity", "public_split_identity",
    "sealed_view_combined_tensor_root", "sealed_view_file_sha256", "best_epoch",
    "best_validation", "epochs_ran", "stopped_epoch", "trajectory_sha256",
    "best_state_tensor_sha256", "last_state_tensor_sha256",
    "best_epoch_is_last_epoch", "state_dict_contains_parameters_and_buffers",
    "selection_rule", "checkpoint_payload_identity_sha256", "state_dict",
}
RESULT_KEYS = {
    "format_version", "started_at_utc", "completed_at_utc", "terminal_status",
    "valid", "task_key", "partner_task_key", "pair_id", "seed", "role",
    "strategy", "joint_config_fingerprint", "assigned_physical_gpu_index",
    "assigned_gpu_uuid", "ordinal_on_gpu", "best_epoch", "best_validation",
    "epochs_ran", "trajectory", "trajectory_sha256", "best_state_tensor_sha256",
    "last_state_tensor_sha256", "checkpoint_relative_path", "checkpoint_file_sha256",
    "checkpoint_size_bytes", "checkpoint_payload_identity_sha256",
    "disk_reload_validation", "elapsed_seconds", "attempt_count", "retry_count",
    "actual_gpu", "attempt_failure_count",
}


class ProtocolError(RuntimeError):
    """A non-retryable protocol violation."""


def tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        components = (
            name.encode("utf-8"),
            str(value.dtype).encode("ascii"),
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"),
            value.numpy().tobytes(order="C"),
        )
        for component in components:
            digest.update(len(component).to_bytes(8, "big"))
            digest.update(component)
    return digest.hexdigest()


def clone_state_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone(memory_format=torch.contiguous_format)
        for name, value in module.state_dict().items()
    }


def trajectory_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return plan.stable_hash(list(rows))


def load_sealed_worker_view(path: Path) -> dict[str, Any]:
    value = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(value, dict):
        raise ProtocolError("sealed worker input must be a plain dictionary")
    if set(value) != plan.SEALED_KEYS:
        raise ProtocolError("sealed worker input violates exact key allowlist")
    plan.validate_sealed_view(value)
    return value


def move_view(view: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in view.items()
    }


def build_model(candidate: Mapping[str, Any], view: Mapping[str, Any], device: torch.device) -> DynamicGNN:
    config = dict(candidate["candidate_config"])
    return DynamicGNN(
        config=config,
        in_ch=int(view["num_features"]),
        out_ch=int(view["num_classes"]),
        dropout=float(config["dropout"]),
        hidden_dim=int(config["hidden_dim"]),
        gcnii_alpha=0.1,
        gcnii_theta=0.5,
        gat_heads=int(config["gat_heads"]),
        sage_aggr=str(config["sage_aggr"]),
        gin_eps=float(config["gin_eps"]),
        gat_heads_by_layer=config.get("gat_heads_by_layer"),
        sage_aggr_by_layer=config.get("sage_aggr_by_layer"),
        gin_eps_by_layer=config.get("gin_eps_by_layer"),
        component_variant="baseline",
        activation_variant="relu",
    ).to(device)


def validation_accuracy(model: torch.nn.Module, view: Mapping[str, Any]) -> float:
    model.eval()
    with torch.no_grad():
        output = model(view["x"], view["edge_index"])
        if not bool(torch.isfinite(output).all().detach().cpu().item()):
            raise FloatingPointError("non-finite validation output")
        mask = view["val_mask"]
        count = int(mask.sum().item())
        if count <= 0:
            raise ProtocolError("validation mask is empty")
        prediction = output.argmax(dim=1)
        correct = prediction[mask].eq(view["y_sealed"][mask]).sum().item()
        return float(correct / count)


def _checkpoint_metadata_identity(payload: Mapping[str, Any]) -> str:
    metadata = {
        key: value for key, value in payload.items()
        if key not in {"state_dict", "checkpoint_payload_identity_sha256"}
    }
    return plan.stable_hash(metadata)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return plan.sha256_file(path)


def atomic_json(path: Path, value: Any) -> None:
    plan.dump_json(path, value)


def _is_retryable_infrastructure_error(error: BaseException) -> bool:
    text = str(error).lower()
    tokens = (
        "cuda out of memory", "cublas", "cudnn", "cuda error",
        "device-side assert", "driver", "resource temporarily unavailable",
        "input/output error", "no space left on device",
    )
    return isinstance(error, (OSError, RuntimeError)) and any(token in text for token in tokens)


def train_validation_best(
    *,
    candidate: Mapping[str, Any],
    task: Mapping[str, Any],
    sealed_cpu: Mapping[str, Any],
    sealed_identity: Mapping[str, Any],
    protocol_sha256: str,
    pretraining_immutable_root: str,
    device: torch.device,
    checkpoint_path: Path,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict[str, Any]:
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    assert_strict_torch_determinism(torch)
    seed = int(task["seed"])
    view = move_view(sealed_cpu, device)
    started = time.monotonic()
    started_at_utc = plan.utc_now()
    with isolated_rng(seed, device):
        model = build_model(candidate, view, device)
        config = candidate["candidate_config"]
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["lr"]),
            weight_decay=float(config["l2"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=300, eta_min=0.01 * float(config["lr"]),
        )
        best_validation = -math.inf
        best_epoch: int | None = None
        best_state: dict[str, torch.Tensor] | None = None
        no_improve = 0
        trajectory: list[dict[str, Any]] = []
        epochs_ran = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(view["x"], view["edge_index"])
            if not bool(torch.isfinite(output).all().detach().cpu().item()):
                raise FloatingPointError("non-finite training output")
            train_mask = view["train_mask"]
            loss = F.cross_entropy(output[train_mask], view["y_sealed"][train_mask])
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_validation = validation_accuracy(model, view)
            epochs_ran = epoch + 1
            trajectory.append({
                "epoch_1based": epochs_ran,
                "loss": float(loss.detach().cpu().item()),
                "validation_accuracy": current_validation,
            })
            if current_validation > best_validation:
                best_validation = current_validation
                best_epoch = epochs_ran
                best_state = clone_state_to_cpu(model)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is None or best_epoch is None:
            raise ProtocolError("no validation-best state was captured")
        last_state = clone_state_to_cpu(model)
        last_state_sha256 = tensor_state_hash(last_state)
        best_state_sha256 = tensor_state_hash(best_state)
        if best_epoch != epochs_ran and best_state_sha256 == last_state_sha256:
            raise ProtocolError("non-final validation-best state is indistinguishable from last state")

        model.load_state_dict(best_state, strict=True)
        restored_validation = validation_accuracy(model, view)
        if restored_validation != best_validation:
            raise ProtocolError("restored validation-best state does not reproduce its score")
        metadata = {
            "format_version": plan.FORMAT_VERSION,
            "pretraining_immutable_root": pretraining_immutable_root,
            "protocol_sha256": protocol_sha256,
            "role": str(task["role"]),
            "strategy": str(task["strategy"]),
            "method_name": candidate["method_identity"],
            "pair_id": int(task["pair_id"]),
            "seed": seed,
            "joint_config_fingerprint": str(task["joint_config_fingerprint"]),
            "candidate": dict(candidate),
            "dataset_content_identity": plan.DATASET_CONTENT_ID,
            "public_split_identity": plan.PUBLIC_SPLIT_ID,
            "sealed_view_combined_tensor_root": sealed_identity["combined_tensor_root"],
            "sealed_view_file_sha256": sealed_identity["sealed_file"]["sha256"],
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "epochs_ran": epochs_ran,
            "stopped_epoch": epochs_ran,
            "trajectory_sha256": trajectory_hash(trajectory),
            "best_state_tensor_sha256": best_state_sha256,
            "last_state_tensor_sha256": last_state_sha256,
            "best_epoch_is_last_epoch": best_epoch == epochs_ran,
            "state_dict_contains_parameters_and_buffers": True,
            "selection_rule": "strict validation improvement; earliest epoch retained on tie",
        }
        payload = {**metadata, "checkpoint_payload_identity_sha256": plan.stable_hash(metadata), "state_dict": best_state}
        if set(payload) != CHECKPOINT_PAYLOAD_KEYS:
            raise ProtocolError("checkpoint payload violates exact key allowlist")
        checkpoint_sha256 = atomic_torch_save(payload, checkpoint_path)

        reloaded = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        if reloaded["checkpoint_payload_identity_sha256"] != _checkpoint_metadata_identity(reloaded):
            raise ProtocolError("checkpoint payload identity mismatch after reload")
        if tensor_state_hash(reloaded["state_dict"]) != best_state_sha256:
            raise ProtocolError("checkpoint tensor-state hash mismatch after reload")
        model.load_state_dict(reloaded["state_dict"], strict=True)
        disk_validation = validation_accuracy(model, view)
        if disk_validation != best_validation:
            raise ProtocolError("checkpoint disk reload does not reproduce validation")

    result = {
        "format_version": plan.FORMAT_VERSION,
        "started_at_utc": started_at_utc,
        "completed_at_utc": plan.utc_now(),
        "terminal_status": "success",
        "valid": True,
        "task_key": str(task["task_key"]),
        "partner_task_key": str(task["partner_task_key"]),
        "pair_id": int(task["pair_id"]),
        "seed": seed,
        "role": str(task["role"]),
        "strategy": str(task["strategy"]),
        "joint_config_fingerprint": str(task["joint_config_fingerprint"]),
        "assigned_physical_gpu_index": int(task["assigned_physical_gpu_index"]),
        "assigned_gpu_uuid": str(task["assigned_gpu_uuid"]),
        "ordinal_on_gpu": int(task["ordinal_on_gpu"]),
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "epochs_ran": epochs_ran,
        "trajectory": trajectory,
        "trajectory_sha256": trajectory_hash(trajectory),
        "best_state_tensor_sha256": best_state_sha256,
        "last_state_tensor_sha256": last_state_sha256,
        "checkpoint_relative_path": str(checkpoint_path.relative_to(checkpoint_path.parents[2])),
        "checkpoint_file_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_payload_identity_sha256": payload["checkpoint_payload_identity_sha256"],
        "disk_reload_validation": disk_validation,
        "elapsed_seconds": time.monotonic() - started,
    }
    if plan._contains_forbidden_metric_key(result):
        raise ProtocolError("training result contains a forbidden metric field")
    return result


def _gpu_identity(shard: int, device: torch.device) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(shard):
        raise ProtocolError(f"CUDA_VISIBLE_DEVICES must equal physical shard {shard}")
    if device.type != "cuda" or device.index not in (None, 0):
        raise ProtocolError("worker device must be cuda:0 within its single visible physical GPU")
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.splitlines()
    parsed = {}
    for line in rows:
        index, uuid, name = [part.strip() for part in line.split(",", 2)]
        parsed[int(index)] = {"physical_index": int(index), "uuid": uuid, "name": name}
    identity = parsed.get(shard)
    if identity is None or identity["uuid"] != plan.GPU_UUIDS[shard]:
        raise ProtocolError("physical GPU UUID contract mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ProtocolError("worker must see exactly one CUDA device")
    return {**identity, "cuda_visible_devices": visible, "logical_device": "cuda:0"}


def checkpoint_path_for(root: Path, task: Mapping[str, Any]) -> Path:
    return (
        root / "checkpoints" / f"pair_{int(task['pair_id']):02d}_seed_{int(task['seed'])}"
        / f"{task['role']}_{task['joint_config_fingerprint']}.pt"
    )


def _record_path(root: Path, shard: int, task: Mapping[str, Any]) -> Path:
    return root / "training" / f"gpu{shard}" / "records" / f"{task['task_key']}.json"


def _failure_path(root: Path, shard: int, task: Mapping[str, Any], attempt: int) -> Path:
    return root / "training" / f"gpu{shard}" / "failures" / f"{task['task_key']}__attempt_{attempt}.json"


def _verify_completed_record(root: Path, task: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    if set(record) != RESULT_KEYS:
        raise ProtocolError("resume result violates exact key allowlist")
    if record.get("terminal_status") != "success" or record.get("task_key") != task["task_key"]:
        raise ProtocolError("resume result identity mismatch")
    relative = Path(str(record["checkpoint_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ProtocolError("resume checkpoint path escapes result root")
    checkpoint = (root / relative).resolve()
    if checkpoint != checkpoint_path_for(root, task).resolve():
        raise ProtocolError("resume checkpoint path is not the frozen task path")
    if not checkpoint.is_file() or plan.sha256_file(checkpoint) != record["checkpoint_file_sha256"]:
        raise ProtocolError("resume checkpoint identity mismatch")
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if not isinstance(payload, Mapping) or set(payload) != CHECKPOINT_PAYLOAD_KEYS:
        raise ProtocolError("resume checkpoint payload key mismatch")
    if tensor_state_hash(payload["state_dict"]) != record["best_state_tensor_sha256"]:
        raise ProtocolError("resume checkpoint tensor-state mismatch")


def _rebuild_jsonl(root: Path, shard: int, tasks: Sequence[Mapping[str, Any]]) -> None:
    output = root / "training" / f"gpu{shard}" / "seed_results.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for task in sorted(tasks, key=lambda row: int(row["ordinal_on_gpu"])):
            record_path = _record_path(root, shard, task)
            if record_path.is_file():
                record = plan.load_json(record_path)
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, output)


def run_shard(root: Path, shard: int, device_string: str) -> None:
    freeze = plan.verify_manifest(root, root / "pretraining_immutable_manifest.json")
    if not freeze["passed"]:
        raise ProtocolError("pretraining immutable manifest drift")
    device = torch.device(device_string)
    actual_gpu = _gpu_identity(shard, device)
    all_tasks = plan.load_json(root / "training_tasks.json")
    tasks = sorted(
        [row for row in all_tasks if int(row["assigned_physical_gpu_index"]) == shard],
        key=lambda row: int(row["ordinal_on_gpu"]),
    )
    if len(tasks) != 30 or [int(row["ordinal_on_gpu"]) for row in tasks] != list(range(30)):
        raise ProtocolError("shard task schedule mismatch")
    for index in range(0, 30, 2):
        pair = tasks[index:index + 2]
        if len({row["pair_id"] for row in pair}) != 1 or [row["within_pair_order"] for row in pair] != [0, 1]:
            raise ProtocolError("paired tasks are not adjacent and ordered")
    candidates = {
        "P": plan.load_json(root / "frozen_candidate_P.json"),
        "Q": plan.load_json(root / "frozen_candidate_Q.json"),
    }
    sealed_cpu = load_sealed_worker_view(root / "sealed/train_val_sealed_view.pt")
    sealed_identity = plan.load_json(root / "sealed/train_val_sealed_view_identity.json")
    plan.verify_sealed_identity(
        sealed_cpu, root / "sealed/train_val_sealed_view.pt", sealed_identity,
    )
    protocol_sha256 = plan.file_identity(root / "protocol.json")["sha256"]
    pretraining_root = plan.load_json(root / "pretraining_immutable_manifest.json")["manifest_root_hash"]
    shard_dir = root / "training" / f"gpu{shard}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    plan.dump_json(shard_dir / "run_plan.json", {
        "format_version": plan.FORMAT_VERSION,
        "started_at_utc": plan.utc_now(),
        "shard": shard, "actual_gpu": actual_gpu,
        "pretraining_immutable_root": pretraining_root,
        "task_count": len(tasks), "task_keys": [row["task_key"] for row in tasks],
    })
    completed = 0
    for task in tasks:
        record_path = _record_path(root, shard, task)
        if record_path.is_file():
            _verify_completed_record(root, task, plan.load_json(record_path))
            completed += 1
            continue
        checkpoint = checkpoint_path_for(root, task)
        failures = []
        for attempt in range(1, MAX_SYSTEM_RETRIES + 2):
            try:
                result = train_validation_best(
                    candidate=candidates[task["role"]], task=task,
                    sealed_cpu=sealed_cpu, sealed_identity=sealed_identity,
                    protocol_sha256=protocol_sha256,
                    pretraining_immutable_root=pretraining_root,
                    device=device, checkpoint_path=checkpoint,
                )
                result["attempt_count"] = attempt
                result["retry_count"] = attempt - 1
                result["actual_gpu"] = actual_gpu
                result["attempt_failure_count"] = len(failures)
                if set(result) != RESULT_KEYS or plan._contains_forbidden_metric_key(result):
                    raise ProtocolError("final training result violates exact allowlist/firewall")
                atomic_json(record_path, result)
                completed += 1
                break
            except ProtocolError:
                raise
            except BaseException as error:
                retryable = _is_retryable_infrastructure_error(error)
                failure = {
                    "format_version": plan.FORMAT_VERSION,
                    "task_key": task["task_key"], "attempt": attempt,
                    "retryable_infrastructure_failure": retryable,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                atomic_json(_failure_path(root, shard, task, attempt), failure)
                failures.append(failure)
                if not retryable or attempt > MAX_SYSTEM_RETRIES:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        _rebuild_jsonl(root, shard, tasks)
        plan.dump_json(shard_dir / "progress.json", {
            "format_version": plan.FORMAT_VERSION,
            "shard": shard, "expected": len(tasks), "completed": completed,
            "remaining": len(tasks) - completed,
        })
        print(json.dumps({"shard": shard, "completed": completed, "expected": len(tasks)}, sort_keys=True), flush=True)
    _rebuild_jsonl(root, shard, tasks)
    plan.dump_json(shard_dir / "progress.json", {
        "format_version": plan.FORMAT_VERSION,
        "shard": shard, "expected": len(tasks), "completed": completed,
        "remaining": len(tasks) - completed, "terminal_status": "complete",
        "completed_at_utc": plan.utc_now(),
    })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_shard(args.root.resolve(), args.shard, args.device)


if __name__ == "__main__":
    main()
