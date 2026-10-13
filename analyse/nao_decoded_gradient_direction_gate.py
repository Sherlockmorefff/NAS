#!/usr/bin/env python3
"""Run the frozen NAO-style decoded gradient direction gate.

The experiment is deliberately narrow: prepare freezes anchors and proposals,
run evaluates validation-only shards, and analyse applies the preregistered gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture_diagnostics import (  # noqa: E402
    architecture_fingerprint,
    config_from_igraph,
    config_to_igraph,
    load_joint_vae,
    normalize_architecture_config,
)
from analyse import original_head_deterministic_batch_pilot as v1  # noqa: E402
from analyse.performance_aware_vae_strict_regression import (  # noqa: E402
    StrictPerformanceHead,
)
from analyse.run_original_head_confirmatory_attribution_v2_labels import (  # noqa: E402
    physical_gpu_identity,
    sanitize_training_metadata,
)
from analyse.vae_label_expansion import EXPECTED_DATASET_CONTEXT  # noqa: E402
from dataset_utils import (  # noqa: E402
    assert_dataset_context_matches,
    dataset_context_from_manifest,
    load_dataset_from_request,
    resolve_dataset_request,
)
from eval_utils import train_and_eval_arch  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/nao_decoded_gradient_direction_gate_global4_v1.json"
SCRIPT_RELATIVE = "analyse/nao_decoded_gradient_direction_gate.py"
TEST_RELATIVE = "tests/test_nao_decoded_gradient_direction_gate.py"
CONFIG_RELATIVE = "configs/nao_decoded_gradient_direction_gate_global4_v1.json"
CRITICAL_SOURCE_RELATIVES = (
    "architecture_diagnostics.py",
    "models.py",
    "nas_space.py",
    "hp_modes.py",
    "analyse/original_head_deterministic_batch_pilot.py",
    "analyse/performance_aware_vae_strict_regression.py",
    "analyse/run_original_head_confirmatory_attribution_v2_labels.py",
    "dataset_utils.py",
    "eval_utils.py",
)
PROTECTED_PATH_RELATIVES = (
    "results/final/frozen-validation-checkpoints-global4-v1",
    "results/final/frozen-checkpoint-cora-test-reveal-global4-v1",
    "TEST_REVEAL_READY",
)
FORBIDDEN_RESULT_KEYS = {"test_acc", "test_accuracy", "test_loss", "test_metric", "test_mask"}
TRANSIENT_INFRASTRUCTURE_TOKENS = (
    "connection reset",
    "driver shutting down",
    "resource temporarily unavailable",
    "cuda error: unknown error",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def freeze_json(path: Path, value: Any) -> None:
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError(f"frozen file differs on resume: {path}")
        return
    atomic_json(path, value)


def record_command(output: Path) -> None:
    row = {"at": utc_now(), "argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd())}
    path = output / "commands.jsonl"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def strict_result_key_hits(value: Any, path: str = "root") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            child_path = f"{path}.{key}"
            if lowered in FORBIDDEN_RESULT_KEYS or lowered.startswith("test_"):
                hits.append(child_path)
            hits.extend(strict_result_key_hits(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(strict_result_key_hits(child, f"{path}[{index}]"))
    return hits


def run_text(args: Sequence[str]) -> str:
    return subprocess.run(
        list(args), cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)}


def select_anchors(
    union_rows: Sequence[Mapping[str, Any]],
    statistics: Sequence[Mapping[str, Any]],
    p_joint: str,
    q_joint: str,
    target: int = 8,
) -> list[dict[str, Any]]:
    means: dict[str, set[float]] = {}
    for row in statistics:
        joint = str(row["joint_config_fingerprint"])
        means.setdefault(joint, set()).add(float(row["all30_mean_validation"]))
    candidates: list[dict[str, Any]] = []
    for source in union_rows:
        joint = str(source["joint_config_fingerprint"])
        if source.get("hp_regime") != "hp_high" or joint in {p_joint, q_joint}:
            continue
        values = means.get(joint, set())
        if len(values) != 1:
            raise RuntimeError(f"ambiguous 30-seed mean for joint fingerprint {joint}")
        row = dict(source)
        row["all30_mean_validation"] = next(iter(values))
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            -float(row["all30_mean_validation"]),
            str(row["architecture_fingerprint"]),
            str(row["joint_config_fingerprint"]),
        )
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        fingerprint = str(row["architecture_fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        row["anchor_index"] = len(unique)
        unique.append(row)
    if len(unique) < target:
        raise RuntimeError(f"only {len(unique)} eligible unique anchors; expected at least {target}")
    return unique[:target]


@dataclass(frozen=True)
class HeadMember:
    member_id: str
    head: torch.nn.Module
    hp_mean: np.ndarray
    hp_std: np.ndarray
    y_mean: float
    y_std: float
    checkpoint_path: str = ""


def _ensemble_scores_tensor(
    z_arch: torch.Tensor, raw_hp: np.ndarray, members: Sequence[HeadMember]
) -> torch.Tensor:
    if z_arch.ndim != 2 or z_arch.shape[1] != 12:
        raise ValueError(f"z_arch must have shape [N,12], got {tuple(z_arch.shape)}")
    hp = np.asarray(raw_hp, dtype=np.float32).copy()
    if hp.shape != (z_arch.shape[0], 4):
        raise ValueError(f"raw_hp must have shape [N,4], got {hp.shape}")
    predictions = []
    for member in members:
        scaled = (hp - member.hp_mean.astype(np.float32)) / member.hp_std.astype(np.float32)
        hp_tensor = torch.tensor(scaled, dtype=z_arch.dtype, device=z_arch.device)
        features = torch.cat([z_arch, hp_tensor], dim=1)
        prediction = member.head(features) * member.y_std + member.y_mean
        predictions.append(prediction)
    if not predictions:
        raise ValueError("Original Head ensemble is empty")
    return torch.stack(predictions, dim=0).mean(dim=0)


def ensemble_scores_and_arch_gradient(
    z_arch: np.ndarray, raw_hp: np.ndarray, members: Sequence[HeadMember]
) -> tuple[np.ndarray, np.ndarray]:
    z = torch.tensor(np.asarray(z_arch, dtype=np.float32), dtype=torch.float32, requires_grad=True)
    scores = _ensemble_scores_tensor(z, raw_hp, members)
    gradient = torch.autograd.grad(scores.sum(), z, create_graph=False, retain_graph=False)[0]
    return scores.detach().cpu().numpy().astype(np.float64), gradient.detach().cpu().numpy().astype(np.float64)


def ensemble_scores(
    z_arch: np.ndarray, raw_hp: np.ndarray, members: Sequence[HeadMember]
) -> np.ndarray:
    z = torch.tensor(np.asarray(z_arch, dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        scores = _ensemble_scores_tensor(z, raw_hp, members)
    return scores.cpu().numpy().astype(np.float64)


def deterministic_decode(model: Any, z_arch: Sequence[float]) -> dict[str, Any]:
    tensor = torch.tensor([list(z_arch)], dtype=torch.float32)
    with torch.no_grad():
        graphs = model.arch_vae.decode(tensor, stochastic=False)
    if len(graphs) != 1:
        raise RuntimeError(f"deterministic decoder returned {len(graphs)} graphs")
    config = normalize_architecture_config(config_from_igraph(model, graphs[0]))
    valid, reason = v1.validate_static_config(config)
    if not valid:
        raise ValueError(f"illegal decoded architecture: {reason}")
    return config


def choose_first_legal_step(
    parent_z: np.ndarray,
    unit_gradient: np.ndarray,
    step_sizes: Sequence[float],
    parent_fingerprint: str,
    prior_proposal_fingerprints: set[str],
    decode_fn: Callable[[np.ndarray], Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for alpha in step_sizes:
        plus_z = np.asarray(parent_z, dtype=np.float64) + float(alpha) * unit_gradient
        minus_z = np.asarray(parent_z, dtype=np.float64) - float(alpha) * unit_gradient
        attempt: dict[str, Any] = {"alpha": float(alpha), "accepted": False}
        try:
            plus_config = dict(decode_fn(plus_z))
            minus_config = dict(decode_fn(minus_z))
            plus_fp = architecture_fingerprint(plus_config)
            minus_fp = architecture_fingerprint(minus_config)
            attempt.update({"plus_fingerprint": plus_fp, "minus_fingerprint": minus_fp})
            reasons = []
            if plus_fp == parent_fingerprint:
                reasons.append("PLUS_EQUALS_PARENT")
            if minus_fp == parent_fingerprint:
                reasons.append("MINUS_EQUALS_PARENT")
            if plus_fp == minus_fp:
                reasons.append("PLUS_EQUALS_MINUS")
            if plus_fp in prior_proposal_fingerprints:
                reasons.append("PLUS_DUPLICATES_EARLIER_PROPOSAL")
            if minus_fp in prior_proposal_fingerprints:
                reasons.append("MINUS_DUPLICATES_EARLIER_PROPOSAL")
            if reasons:
                attempt["failure_reasons"] = reasons
                attempts.append(attempt)
                continue
            attempt["accepted"] = True
            attempts.append(attempt)
            return {
                "alpha": float(alpha),
                "plus_z_arch": plus_z.tolist(),
                "minus_z_arch": minus_z.tolist(),
                "plus_fingerprint": plus_fp,
                "minus_fingerprint": minus_fp,
                "plus_config": plus_config,
                "minus_config": minus_config,
            }, attempts
        except (TypeError, ValueError, KeyError, IndexError, RuntimeError) as error:
            attempt["failure_reasons"] = [f"DECODE_ERROR:{type(error).__name__}:{error}"]
            attempts.append(attempt)
    return None, attempts


def classify_gate(valid_count: int, plus_minus_wins: int, mean_plus_minus: float | None,
                  plus_parent_wins: int, mean_plus_parent: float | None,
                  config: Mapping[str, Any]) -> str:
    if valid_count < int(config["minimum_valid_anchors"]):
        return "NAO_DIRECTION_GATE_DECODE_REJECTED"
    passed = (
        plus_minus_wins >= int(config["minimum_plus_minus_wins"])
        and mean_plus_minus is not None
        and mean_plus_minus >= float(config["minimum_mean_delta_plus_minus"])
        and plus_parent_wins >= int(config["minimum_plus_parent_wins"])
        and mean_plus_parent is not None
        and mean_plus_parent >= float(config["minimum_mean_delta_plus_parent"])
    )
    return "NAO_DIRECTION_GATE_PASSED" if passed else "NAO_DIRECTION_GATE_REJECTED"


def task_result_key(row: Mapping[str, Any]) -> str:
    return str(row["task_key"])


def pending_tasks(tasks: Sequence[Mapping[str, Any]], completed_keys: set[str]) -> list[dict[str, Any]]:
    return [dict(task) for task in tasks if task_result_key(task) not in completed_keys]


def is_transient_infrastructure_error(error: BaseException) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return any(token in message for token in TRANSIENT_INFRASTRUCTURE_TOKENS)


def load_config(path: Path) -> dict[str, Any]:
    config = dict(load_json(path))
    if config.get("format_version") != "nao_decoded_gradient_direction_gate_global4_v1":
        raise ValueError("unexpected frozen config format")
    return config


def resolve_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    output = REPO_ROOT / str(config["output_relative"])
    historical = REPO_ROOT / str(config["historical_root"])
    strict_root = REPO_ROOT / str(config["strict_head_root"])
    checkpoint = REPO_ROOT / str(config["base_checkpoint"])
    return output, historical, strict_root, checkpoint


def encode_architecture_means(model: Any, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    graphs = [config_to_igraph(row) for row in rows]
    with torch.no_grad():
        mu, _ = model.arch_vae.encode(graphs)
    values = mu.detach().cpu().numpy().astype(np.float64)
    if values.shape != (len(rows), 12) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid encoded architecture means: shape={values.shape}")
    return values


def encode_hp_means(model: Any, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    hp = torch.tensor([row["raw_hp_vector"] for row in rows], dtype=torch.float32)
    with torch.no_grad():
        mu, _ = model.hp_vae.encode(hp)
    return mu.detach().cpu().numpy().astype(np.float64)


def median_nearest_neighbor_distance(latents: np.ndarray) -> float:
    if latents.ndim != 2 or len(latents) < 2:
        raise ValueError("reference latent matrix is too small")
    differences = latents[:, None, :] - latents[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    np.fill_diagonal(distances, np.inf)
    value = float(np.median(np.min(distances, axis=1)))
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid reference nearest-neighbor scale: {value}")
    return value


def load_frozen_head_members(strict_root: Path, expected_base_checkpoint_sha256: str) -> list[HeadMember]:
    paths = sorted(
        (strict_root / "checkpoints/full/group_repeated").glob(
            "group_*/seed_*/frozen_head/checkpoint.pth"
        )
    )
    if len(paths) != 150:
        raise RuntimeError(f"expected 150 Original Head checkpoints, found {len(paths)}")
    members: list[HeadMember] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        task = payload["task"]
        if (
            task["dataset_variant"] != "full"
            or task["protocol"] != "group_repeated"
            or task["model"] != "frozen_head"
            or payload["arch_vae_state"] is not None
            or int(payload["head_input_dim"]) != 16
            or payload.get("base_checkpoint_sha256") != expected_base_checkpoint_sha256
        ):
            raise RuntimeError(f"unexpected Original Head schema: {path}")
        head = StrictPerformanceHead(16)
        head.load_state_dict(payload["head_state"])
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        scaling = payload["scaling"]
        members.append(
            HeadMember(
                member_id=f"{task['split_id']}__seed{int(task['training_seed'])}",
                head=head,
                hp_mean=np.asarray(scaling["hp_mean"], dtype=np.float32),
                hp_std=np.asarray(scaling["hp_std"], dtype=np.float32),
                y_mean=float(scaling["y_mean"]),
                y_std=float(scaling["y_std"]),
                checkpoint_path=str(path.resolve()),
            )
        )
    if len({member.member_id for member in members}) != 150:
        raise RuntimeError("Original Head member IDs are not unique")
    return members


def source_identity(config: Mapping[str, Any], historical: Path) -> dict[str, Any]:
    files = {
        "eval_utils.py": REPO_ROOT / "eval_utils.py",
        "dataset_utils.py": REPO_ROOT / "dataset_utils.py",
        "historical_runner": REPO_ROOT / "analyse/run_high_regime_prior_model_blind_validation_labels.py",
        "historical_eval_utils_snapshot": historical / "source_snapshot/eval_utils.py",
        "historical_runner_snapshot": historical / "source_snapshot/analyse/run_high_regime_prior_model_blind_validation_labels.py",
    }
    identities = {name: file_identity(path) for name, path in files.items()}
    checks = {
        "eval_utils_expected": identities["eval_utils.py"]["sha256"] == config["expected_eval_utils_sha256"],
        "dataset_utils_expected": identities["dataset_utils.py"]["sha256"] == config["expected_dataset_utils_sha256"],
        "runner_expected": identities["historical_runner"]["sha256"] == config["expected_historical_runner_sha256"],
        "eval_utils_matches_snapshot": identities["eval_utils.py"]["sha256"] == identities["historical_eval_utils_snapshot"]["sha256"],
        "runner_matches_snapshot": identities["historical_runner"]["sha256"] == identities["historical_runner_snapshot"]["sha256"],
    }
    return {"files": identities, "checks": checks, "passed": all(checks.values())}


def environment_payload() -> dict[str, Any]:
    try:
        gpu_text = run_text([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu_text = ""
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_query": gpu_text.splitlines(),
        "branch": run_text(["git", "branch", "--show-current"]),
        "head": run_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_text(["git", "status", "--short"]).splitlines(),
        "protected_paths_touched": False,
    }


def load_parent_seed_results(
    historical: Path, anchors: Sequence[Mapping[str, Any]], seeds: Sequence[int],
    source_ok: bool,
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    wanted = {(str(anchor["joint_config_fingerprint"]), int(seed)) for anchor in anchors for seed in seeds}
    found: dict[tuple[str, int], dict[str, Any]] = {}
    path = historical / "phase_b/initial/seed_results_merged.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row.get("joint_config_fingerprint")), int(row.get("seed", -1)))
            if key in wanted:
                if key in found:
                    raise RuntimeError(f"duplicate historical parent seed result: {key}")
                found[key] = row
    reasons: list[str] = []
    if not source_ok:
        reasons.append("EVALUATOR_SOURCE_IDENTITY_MISMATCH")
    anchor_by_joint = {str(anchor["joint_config_fingerprint"]): anchor for anchor in anchors}
    safe_rows: list[dict[str, Any]] = []
    for joint, seed in sorted(wanted):
        row = found.get((joint, seed))
        anchor = anchor_by_joint[joint]
        if row is None:
            reasons.append(f"MISSING:{joint}:{seed}")
            continue
        checks = (
            row.get("architecture_fingerprint") == anchor["architecture_fingerprint"],
            row.get("hp_fingerprint") == anchor["hp_fingerprint"],
            row.get("joint_config_fingerprint") == anchor["joint_config_fingerprint"],
            row.get("raw_hp_vector") == anchor["raw_hp_vector"],
            row.get("operations") == anchor["operations"],
            row.get("edges") == anchor["edges"],
            row.get("dataset") == "cora",
            row.get("training_mode") == "full_batch",
            row.get("metric") == "accuracy",
            row.get("training_entry") == "eval_utils.train_and_eval_arch",
            int(row.get("eval_epochs", -1)) == 300,
            int(row.get("patience", -1)) == 80,
            row.get("terminal_status") == "success",
            row.get("valid") is True,
            strict_result_key_hits(row) == [],
        )
        if not all(checks):
            reasons.append(f"PROTOCOL_OR_IDENTITY_MISMATCH:{joint}:{seed}")
            continue
        safe_rows.append({
            "anchor_index": int(anchor["anchor_index"]),
            "role": "parent",
            "seed": seed,
            "task_key": f"a{int(anchor['anchor_index']):02d}_s{seed}_parent_reused",
            "architecture_fingerprint": row["architecture_fingerprint"],
            "hp_fingerprint": row["hp_fingerprint"],
            "joint_config_fingerprint": row["joint_config_fingerprint"],
            "val_acc": float(row["val_acc"]),
            "valid": True,
            "terminal_status": "success",
            "result_origin": "historical_parent_reuse",
            "dataset": row["dataset"],
            "dataset_content_fingerprint": row["dataset_content_fingerprint"],
            "split_fingerprint": row["split_fingerprint"],
            "metric": row["metric"],
            "eval_epochs": row["eval_epochs"],
            "patience": row["patience"],
            "training_entry": row["training_entry"],
        })
    return not reasons and len(safe_rows) == len(wanted), safe_rows, reasons


def build_task_plan(
    anchors: Sequence[Mapping[str, Any]], preflight_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int], parent_reuse: bool,
) -> list[dict[str, Any]]:
    anchor_by_index = {int(row["anchor_index"]): row for row in anchors}
    tasks: list[dict[str, Any]] = []
    shard_order = {0: 0, 1: 0}
    for proposal in preflight_rows:
        if not proposal.get("valid_proposal"):
            continue
        index = int(proposal["anchor_index"])
        anchor = anchor_by_index[index]
        for seed in seeds:
            shard = (index + int(seed)) % 2
            roles = ["plus", "minus"] if shard == 0 else ["minus", "plus"]
            if not parent_reuse:
                roles.append("parent")
            for role in roles:
                if role == "parent":
                    architecture = anchor
                    fingerprint = anchor["architecture_fingerprint"]
                else:
                    architecture = proposal[f"{role}_config"]
                    fingerprint = proposal[f"{role}_fingerprint"]
                task_key = f"a{index:02d}_s{int(seed)}_{role}"
                tasks.append({
                    "task_key": task_key,
                    "anchor_index": index,
                    "seed": int(seed),
                    "role": role,
                    "assigned_gpu": shard,
                    "order_in_shard": shard_order[shard],
                    "architecture_fingerprint": fingerprint,
                    "parent_joint_config_fingerprint": anchor["joint_config_fingerprint"],
                    "parent_hp_fingerprint": anchor["hp_fingerprint"],
                    "operations": list(architecture["operations"]),
                    "edges": [list(map(int, edge)) for edge in architecture["edges"]],
                    "effective_layers": int(architecture["effective_layers"]),
                    "raw_hp_vector": list(anchor["raw_hp_vector"]),
                    "lr": float(anchor["lr"]),
                    "dropout": float(anchor["dropout"]),
                    "hidden_dim": int(anchor["hidden_dim"]),
                    "l2": float(anchor["l2"]),
                    "gat_heads": int(anchor["gat_heads"]),
                    "sage_aggr": str(anchor["sage_aggr"]),
                    "gin_eps": float(anchor["gin_eps"]),
                    "gat_heads_by_layer": anchor["gat_heads_by_layer"],
                    "sage_aggr_by_layer": anchor["sage_aggr_by_layer"],
                    "gin_eps_by_layer": anchor["gin_eps_by_layer"],
                })
                shard_order[shard] += 1
    return tasks


def prepare(config_path: Path) -> int:
    started = time.monotonic()
    config = load_config(config_path)
    output, historical, strict_root, checkpoint = resolve_paths(config)
    output.mkdir(parents=True, exist_ok=True)
    record_command(output)
    if not (output / "PRE_EXECUTION_ESTIMATE.md").is_file():
        raise RuntimeError("PRE_EXECUTION_ESTIMATE.md must exist before prepare")

    union_path = historical / "stabilization/stabilization_union_configs.json"
    statistics_path = historical / "stabilization/strategy_top8_30seed_statistics.json"
    reservoir_path = historical / "global_reservoir.json"
    anchors = select_anchors(
        load_json(union_path), load_json(statistics_path),
        str(config["p_joint_fingerprint"]), str(config["q_joint_fingerprint"]),
        int(config["target_anchor_count"]),
    )
    freeze_json(output / "anchors.json", anchors)

    environment = environment_payload()
    atomic_json(output / "environment.json", environment)
    sources = source_identity(config, historical)
    checkpoint_identity = file_identity(checkpoint)
    if checkpoint_identity["sha256"] != config["expected_base_checkpoint_sha256"]:
        raise RuntimeError("frozen VAE checkpoint hash mismatch")

    model = load_joint_vae(checkpoint, "global4", device="cpu")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    reference_rows = load_json(reservoir_path)
    unique_reference = {str(row["architecture_fingerprint"]): row for row in reference_rows}
    if len(unique_reference) < int(config["minimum_reference_latents"]):
        raise RuntimeError("INSUFFICIENT_SCALE_REFERENCE")
    ordered_reference = [unique_reference[key] for key in sorted(unique_reference)]
    reference_latents = encode_architecture_means(model, ordered_reference)
    d_nn = median_nearest_neighbor_distance(reference_latents)
    step_sizes = [float(value) * d_nn for value in config["alpha_multipliers"]]

    anchor_latents = encode_architecture_means(model, anchors)
    hp_latents = encode_hp_means(model, anchors)
    basepoint_rows: list[dict[str, Any]] = []
    matched_indices: list[int] = []
    for index, anchor in enumerate(anchors):
        row: dict[str, Any] = {
            "anchor_index": index,
            "anchor_fingerprint": anchor["architecture_fingerprint"],
            "latent_source": "frozen_arch_encoder_mu",
            "z_arch": anchor_latents[index].tolist(),
            "z_hp_source": "frozen_hp_encoder_mu_record_only",
            "z_hp": hp_latents[index].tolist(),
            "predictor_hp_input": "member_train_fold_standardized_raw_global4_hp",
            "raw_hp_vector": list(anchor["raw_hp_vector"]),
        }
        try:
            decoded = deterministic_decode(model, anchor_latents[index])
            decoded_fp = architecture_fingerprint(decoded)
            row["basepoint_decoded_fingerprint"] = decoded_fp
            row["basepoint_matches"] = decoded_fp == anchor["architecture_fingerprint"]
        except Exception as error:
            row["basepoint_matches"] = False
            row["basepoint_decode_error"] = f"{type(error).__name__}: {error}"
        if row["basepoint_matches"]:
            matched_indices.append(index)
        else:
            row["failure_reason"] = "BASEPOINT_DECODE_MISMATCH"
        basepoint_rows.append(row)

    try:
        members = load_frozen_head_members(strict_root, str(config["expected_base_checkpoint_sha256"]))
    except Exception as error:
        payload = {
            "classification": "NO_DIFFERENTIABLE_FROZEN_HEAD",
            "error": f"{type(error).__name__}: {error}",
            "valid_proposal_count": 0,
            "anchors": basepoint_rows,
        }
        atomic_json(output / "preflight.json", payload)
        atomic_json(output / "task_plan.json", {"tasks": [], "task_count": 0})
        return 2

    member_identities = [file_identity(Path(member.checkpoint_path)) for member in members]
    atomic_json(output / "source_and_checkpoint_hashes.json", {
        "source_identity": sources,
        "base_checkpoint": checkpoint_identity,
        "original_head_checkpoints": member_identities,
        "input_artifacts": [file_identity(path) for path in (union_path, statistics_path, reservoir_path)],
    })

    if matched_indices:
        raw_hp = np.asarray([anchors[index]["raw_hp_vector"] for index in matched_indices], dtype=np.float32)
        scores, gradients = ensemble_scores_and_arch_gradient(anchor_latents[matched_indices], raw_hp, members)
        gradient_lookup = {index: gradients[offset] for offset, index in enumerate(matched_indices)}
        score_lookup = {index: float(scores[offset]) for offset, index in enumerate(matched_indices)}
    else:
        gradient_lookup = {}
        score_lookup = {}

    frozen_proposals: set[str] = set()
    preflight_rows: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors):
        row = dict(basepoint_rows[index])
        row["step_sizes_attempted"] = step_sizes
        row["predicted_parent"] = score_lookup.get(index)
        if index not in gradient_lookup:
            row["gradient_norm"] = None
            row["chosen_alpha"] = None
            row["valid_proposal"] = False
            preflight_rows.append(row)
            continue
        gradient = gradient_lookup[index]
        norm = float(np.linalg.norm(gradient))
        row["gradient_norm"] = norm
        if not np.isfinite(gradient).all() or not math.isfinite(norm) or norm <= float(config["gradient_norm_minimum"]):
            row.update({"chosen_alpha": None, "valid_proposal": False, "failure_reason": "INVALID_GRADIENT"})
            preflight_rows.append(row)
            continue
        unit = gradient / norm
        chosen, attempts = choose_first_legal_step(
            anchor_latents[index], unit, step_sizes, str(anchor["architecture_fingerprint"]),
            frozen_proposals, lambda z: deterministic_decode(model, z),
        )
        row["attempts"] = attempts
        if chosen is None:
            row.update({"chosen_alpha": None, "valid_proposal": False, "failure_reason": "NO_LEGAL_PAIRED_STEP"})
            preflight_rows.append(row)
            continue
        frozen_proposals.update({chosen["plus_fingerprint"], chosen["minus_fingerprint"]})
        proposal_scores = ensemble_scores(
            np.asarray([chosen["plus_z_arch"], chosen["minus_z_arch"]], dtype=np.float32),
            np.asarray([anchor["raw_hp_vector"], anchor["raw_hp_vector"]], dtype=np.float32),
            members,
        )
        row.update(chosen)
        row.update({
            "chosen_alpha": chosen["alpha"], "valid_proposal": True,
            "predicted_plus": float(proposal_scores[0]), "predicted_minus": float(proposal_scores[1]),
            "failure_reason": None,
        })
        preflight_rows.append(row)

    valid_count = sum(bool(row.get("valid_proposal")) for row in preflight_rows)
    classification = (
        "PRE_FLIGHT_PASSED" if valid_count >= int(config["minimum_valid_anchors"])
        else "NAO_DIRECTION_GATE_DECODE_REJECTED"
    )
    preflight_payload = {
        "classification": classification,
        "created_at": utc_now(),
        "reference_latent_count": len(ordered_reference),
        "reference_latent_source": "frozen_arch_encoder_mu",
        "median_nearest_neighbor_distance": d_nn,
        "alpha_multipliers": config["alpha_multipliers"],
        "step_sizes": step_sizes,
        "valid_proposal_count": valid_count,
        "target_anchor_count": int(config["target_anchor_count"]),
        "anchors": preflight_rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output / "preflight.json", preflight_payload)
    table_rows = []
    for row in preflight_rows:
        table_rows.append({
            "anchor_index": row["anchor_index"],
            "anchor_fingerprint": row["anchor_fingerprint"],
            "latent_source": row["latent_source"],
            "gradient_norm": row.get("gradient_norm"),
            "chosen_alpha": row.get("chosen_alpha"),
            "parent_fingerprint": row["anchor_fingerprint"],
            "plus_fingerprint": row.get("plus_fingerprint"),
            "minus_fingerprint": row.get("minus_fingerprint"),
            "predicted_parent": row.get("predicted_parent"),
            "predicted_plus": row.get("predicted_plus"),
            "predicted_minus": row.get("predicted_minus"),
            "valid_proposal": row.get("valid_proposal"),
            "failure_reason": row.get("failure_reason"),
        })
    atomic_csv(output / "proposal_table.csv", table_rows)

    if valid_count < int(config["minimum_valid_anchors"]):
        atomic_json(output / "parent_results.json", [])
        atomic_json(output / "task_plan.json", {
            "classification": classification, "parent_reuse": None,
            "tasks": [], "task_count": 0, "created_at": utc_now(),
        })
        return 0

    valid_anchors = [anchors[int(row["anchor_index"])] for row in preflight_rows if row.get("valid_proposal")]
    reuse, parent_rows, reuse_reasons = load_parent_seed_results(
        historical, valid_anchors, [int(seed) for seed in config["training_seeds"]], sources["passed"]
    )
    if not reuse:
        parent_rows = []
    atomic_json(output / "parent_results.json", parent_rows)
    tasks = build_task_plan(anchors, preflight_rows, config["training_seeds"], reuse)
    task_plan = {
        "classification": "READY_FOR_VALIDATION_ONLY_TRAINING",
        "created_at": utc_now(),
        "parent_reuse": reuse,
        "parent_reuse_reasons": reuse_reasons,
        "training_seeds": config["training_seeds"],
        "task_count": len(tasks),
        "tasks_per_gpu": {str(gpu): sum(int(task["assigned_gpu"]) == gpu for task in tasks) for gpu in (0, 1)},
        "tasks": tasks,
    }
    freeze_json(output / "task_plan.json", task_plan)
    return 0


def execute_validation_task(
    task: Mapping[str, Any], bundle: Any, device: torch.device,
    train_fn: Callable[..., Any] = train_and_eval_arch,
) -> tuple[float, bool, dict[str, Any]]:
    val_acc, valid, metadata = train_fn(
        config={
            "operations": list(task["operations"]),
            "edges": [tuple(edge) for edge in task["edges"]],
            "effective_layers": int(task["effective_layers"]),
        },
        data=bundle.data,
        in_ch=bundle.num_features,
        out_ch=bundle.num_classes,
        lr=float(task["lr"]),
        dropout=float(task["dropout"]),
        hidden_dim=int(task["hidden_dim"]),
        weight_decay=float(task["l2"]),
        gcnii_alpha=0.1,
        gcnii_theta=0.5,
        gat_heads=int(task["gat_heads"]),
        sage_aggr=str(task["sage_aggr"]),
        gin_eps=float(task["gin_eps"]),
        gat_heads_by_layer=task["gat_heads_by_layer"],
        sage_aggr_by_layer=task["sage_aggr_by_layer"],
        gin_eps_by_layer=task["gin_eps_by_layer"],
        device=device,
        max_epochs=300,
        patience=80,
        seed=int(task["seed"]),
        track_test=False,
        return_metadata=True,
        return_reproducibility_metadata=True,
        return_trajectory=True,
        component_variant="baseline",
    )
    cleaned = sanitize_training_metadata(metadata)
    if strict_result_key_hits(cleaned):
        raise RuntimeError(f"forbidden metric field in sanitized metadata: {strict_result_key_hits(cleaned)}")
    return float(val_acc), bool(valid), cleaned


def run_shard(config_path: Path, shard: int, device_text: str, cora_root: str,
              dry_run: bool = False) -> int:
    config = load_config(config_path)
    output, _historical, _strict_root, _checkpoint = resolve_paths(config)
    record_command(output)
    plan = load_json(output / "task_plan.json")
    tasks = sorted(
        [dict(task) for task in plan.get("tasks", []) if int(task["assigned_gpu"]) == shard],
        key=lambda task: int(task["order_in_shard"]),
    )
    run_dir = output / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*.json")):
        row = load_json(path)
        if strict_result_key_hits(row):
            raise RuntimeError(f"forbidden metric field in {path}: {strict_result_key_hits(row)}")
        key = task_result_key(row)
        if key in completed:
            raise RuntimeError(f"duplicate completed task key: {key}")
        completed[key] = row
    expected_keys = {task_result_key(task) for task in tasks}
    foreign = set(completed) - {task_result_key(task) for task in plan.get("tasks", [])}
    if foreign:
        raise RuntimeError(f"foreign task results present: {sorted(foreign)}")
    pending = pending_tasks(tasks, set(completed))
    progress_path = output / f"progress_gpu{shard}.json"
    progress = {
        "shard": shard,
        "device": device_text,
        "expected": len(expected_keys),
        "already_completed": len(expected_keys & set(completed)),
        "pending": len(pending),
        "dataset_loaded": False,
        "training_started": False,
        "updated_at": utc_now(),
    }
    atomic_json(progress_path, progress)
    if dry_run or not tasks:
        return 0

    device = torch.device(device_text)
    gpu = physical_gpu_identity(device)
    if int(gpu["physical_index"]) != shard:
        raise RuntimeError(f"GPU identity mismatch: shard={shard}, actual={gpu}")
    request = resolve_dataset_request("cora", None, cora_root=cora_root)
    loaded_bundle = load_dataset_from_request(request)
    context = dataset_context_from_manifest(loaded_bundle.manifest)
    assert_dataset_context_matches(EXPECTED_DATASET_CONTEXT, context, context="NAO gradient gate")
    bundle = SimpleNamespace(
        data=loaded_bundle.data.to(device),
        num_features=loaded_bundle.num_features,
        num_classes=loaded_bundle.num_classes,
    )
    progress.update({"dataset_loaded": True, "training_started": bool(pending), "actual_gpu": gpu})
    atomic_json(progress_path, {**progress, "updated_at": utc_now()})

    log_path = output / f"run_gpu{shard}.log"
    for task in pending:
        task_started_at = utc_now()
        monotonic = time.monotonic()
        failures = []
        attempt = 0
        while True:
            attempt += 1
            try:
                val_acc, valid, metadata = execute_validation_task(task, bundle, device)
                row = {
                    "task_key": task["task_key"],
                    "anchor_index": int(task["anchor_index"]),
                    "role": task["role"],
                    "seed": int(task["seed"]),
                    "architecture_fingerprint": task["architecture_fingerprint"],
                    "parent_joint_config_fingerprint": task["parent_joint_config_fingerprint"],
                    "parent_hp_fingerprint": task["parent_hp_fingerprint"],
                    "val_acc": val_acc,
                    "valid": valid,
                    "terminal_status": "success" if valid else "invalid",
                    "result_origin": "new_validation_only_run",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": failures,
                    "started_at": task_started_at,
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - monotonic,
                    "dataset": context["canonical_name"],
                    "dataset_content_fingerprint": context["dataset_content_fingerprint"],
                    "split_fingerprint": context["split_fingerprint"],
                    "metric": context["metric"],
                    "training_mode": context["training_mode"],
                    "eval_epochs": 300,
                    "patience": 80,
                    "training_entry": "eval_utils.train_and_eval_arch",
                    "actual_gpu": gpu,
                    **metadata,
                }
                break
            except Exception as error:
                failures.append({
                    "attempt": attempt,
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                })
                if attempt == 1 and is_transient_infrastructure_error(error):
                    continue
                row = {
                    "task_key": task["task_key"],
                    "anchor_index": int(task["anchor_index"]),
                    "role": task["role"],
                    "seed": int(task["seed"]),
                    "architecture_fingerprint": task["architecture_fingerprint"],
                    "parent_joint_config_fingerprint": task["parent_joint_config_fingerprint"],
                    "parent_hp_fingerprint": task["parent_hp_fingerprint"],
                    "val_acc": None,
                    "valid": False,
                    "terminal_status": "failed",
                    "result_origin": "new_validation_only_run",
                    "attempt_count": attempt,
                    "retry_count": attempt - 1,
                    "attempt_failures": failures,
                    "started_at": task_started_at,
                    "finished_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - monotonic,
                }
                break
        hits = strict_result_key_hits(row)
        if hits:
            raise RuntimeError(f"forbidden metric field would be written: {hits}")
        atomic_json(run_dir / f"{task['task_key']}.json", row)
        atomic_text(
            log_path,
            (log_path.read_text(encoding="utf-8") if log_path.exists() else "")
            + json.dumps({"at": utc_now(), "task_key": task["task_key"], "status": row["terminal_status"]})
            + "\n",
        )
        completed[task_result_key(row)] = row
        progress.update({
            "completed": len(expected_keys & set(completed)),
            "pending": len(expected_keys - set(completed)),
            "updated_at": utc_now(),
        })
        atomic_json(progress_path, progress)
    return 0


def exact_one_sided_sign_p(wins: int, count: int) -> float | None:
    if count <= 0:
        return None
    return sum(math.comb(count, value) for value in range(wins, count + 1)) / (2 ** count)


def bootstrap_ci(values: Sequence[float], draws: int, seed: int) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, array.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def leave_one_out(values: Sequence[float]) -> list[dict[str, Any]]:
    array = np.asarray(values, dtype=np.float64)
    rows = []
    for index in range(array.size):
        retained = np.delete(array, index)
        rows.append({
            "omitted_valid_index": index,
            "remaining_count": int(retained.size),
            "mean": float(retained.mean()) if retained.size else None,
            "positive_count": int(np.sum(retained > 0)),
        })
    return rows


def iso_seconds(first: str, last: str) -> float:
    start = datetime.fromisoformat(first.replace("Z", "+00:00"))
    finish = datetime.fromisoformat(last.replace("Z", "+00:00"))
    return (finish - start).total_seconds()


def write_report(output: Path, analysis: Mapping[str, Any], statistics: Sequence[Mapping[str, Any]]) -> None:
    classification = str(analysis["classification"])
    meaning = {
        "NAO_DIRECTION_GATE_PASSED": "冻结 predictor 的局部架构梯度经确定性 decoder 后，在本次高性能区域门控中产生了可重复的 validation 改善。",
        "NAO_DIRECTION_GATE_REJECTED": "局部 predictor 梯度经 decoder 后不能稳定改善真实 validation；停止当前 latent search 路线。",
        "NAO_DIRECTION_GATE_DECODE_REJECTED": "当前 decoder 不能稳定把局部 latent 移动锚定为成对的新架构；未启动 GNN。",
        "NAO_DIRECTION_GATE_INCONCLUSIVE": "required validation 任务不完整或工程审计失败，不能作科学判断。",
    }[classification]
    lines = [
        "# 结论", "", f"{classification}。{meaning}", "",
        "# 实际耗时", "",
        "执行前估算：best 1 h 23 min；expected 2 h 36 min；worst 4 h 55 min。",
        f"实际总耗时：{analysis['timing']['actual_total_seconds']:.1f} 秒。",
        f"GPU 任务累计耗时：{analysis['timing']['summed_gpu_task_seconds']:.1f} 秒。",
        f"新训练任务数：{analysis['new_training_run_count']}。", "",
        "# 零训练检查", "",
        f"8 个 anchor 中 {analysis['valid_anchor_count']} 个生成完整有效 proposal。",
        f"主要失败原因：{analysis.get('preflight_failure_summary') or '无'}。",
        f"是否启动 GNN：{'是' if analysis['new_training_run_count'] else '否'}。", "",
        "# 结果", "",
    ]
    if statistics:
        for row in statistics:
            if row.get("valid_statistics"):
                lines.append(
                    f"- anchor {row['anchor_index']}: parent={row['mean_parent']:.6f}, "
                    f"plus={row['mean_plus']:.6f}, minus={row['mean_minus']:.6f}, "
                    f"plus-minus={row['delta_plus_minus']:+.6f}, "
                    f"plus-parent={row['delta_plus_parent']:+.6f}"
                )
            else:
                lines.append(f"- anchor {row['anchor_index']}: 无有效统计（{row['status']}）")
    else:
        lines.append("无 GNN validation 结果。")
    lines.extend([
        "",
        f"汇总：plus>minus {analysis['plus_minus_win_count']}/8；"
        f"plus>parent {analysis['plus_parent_win_count']}/8；"
        f"平均 plus-minus={analysis['mean_delta_plus_minus']};"
        f"平均 plus-parent={analysis['mean_delta_plus_parent']}。",
        f"中位数：plus-minus={analysis['median_delta_plus_minus']}；plus-parent={analysis['median_delta_plus_parent']}。",
        f"单侧 exact sign-test p={analysis['sign_test_plus_minus_p']}。",
        f"anchor bootstrap 95% CI：plus-minus={analysis['bootstrap_plus_minus_ci95']}；"
        f"plus-parent={analysis['bootstrap_plus_parent_ci95']}。", "",
        "# 判断", "", analysis["decision_explanation"], "",
        "# 解释边界", "",
        "本实验只判断当前冻结 predictor 的局部 architecture-gradient 经当前确定性 decoder 后是否改善 validation。",
        "本实验不能说明 latent 欧氏距离有效，也不能说明 GP/BO、不确定性校准或完整 NAO 有效。", "",
        "# 下一步", "",
        analysis["next_step"], "",
        "本实验读取 test：否。", "",
    ])
    atomic_text(output / "REPORT.md", "\n".join(lines))


def critical_source_identities() -> list[dict[str, Any]]:
    rows = []
    for relative in CRITICAL_SOURCE_RELATIVES:
        identity = file_identity(REPO_ROOT / relative)
        identity["relative_path"] = relative
        rows.append(identity)
    return rows


def embedded_head_provenance(strict_root: Path, expected_base_sha256: str) -> dict[str, Any]:
    paths = sorted(
        (strict_root / "checkpoints/full/group_repeated").glob(
            "group_*/seed_*/frozen_head/checkpoint.pth"
        )
    )
    mismatches = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("base_checkpoint_sha256") != expected_base_sha256
            or payload.get("arch_vae_state") is not None
            or int(payload.get("head_input_dim", -1)) != 16
        ):
            mismatches.append(str(path.resolve()))
    return {
        "checkpoint_count": len(paths),
        "expected_base_checkpoint_sha256": expected_base_sha256,
        "all_embed_expected_base_checkpoint": len(paths) == 150 and not mismatches,
        "arch_vae_state_none_for_all": len(paths) == 150 and not mismatches,
        "mismatches": mismatches,
    }


def write_dirty_tree_and_protected_audits(output: Path) -> None:
    status = run_text(["git", "status", "--short"]).splitlines()
    tracked_patch = run_text(["git", "diff", "--binary"])
    staged_patch = run_text(["git", "diff", "--cached", "--binary"])
    atomic_text(output / "dirty_tree_tracked.patch", tracked_patch + ("\n" if tracked_patch else ""))
    atomic_text(output / "dirty_tree_staged.patch", staged_patch + ("\n" if staged_patch else ""))
    relevant_sources = critical_source_identities()
    atomic_json(output / "dirty_tree_snapshot.json", {
        "created_at": utc_now(),
        "git_head": run_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": status,
        "tracked_diff_sha256": sha256_file(output / "dirty_tree_tracked.patch"),
        "staged_diff_sha256": sha256_file(output / "dirty_tree_staged.patch"),
        "relevant_source_identities": relevant_sources,
        "note": "Pre-existing tracked and untracked worktree content was preserved; no reset or clean was run.",
    })
    states = []
    for relative in PROTECTED_PATH_RELATIVES:
        path = REPO_ROOT / relative
        states.append({
            "relative_path": relative,
            "pre_execution_exists": False,
            "post_execution_exists": path.exists(),
            "modified_by_experiment": False,
        })
    atomic_json(output / "protected_state_audit.json", {
        "created_at": utc_now(),
        "passed": all(not row["post_execution_exists"] for row in states),
        "paths": states,
        "git_destructive_commands_run": False,
    })


def create_archive(output: Path) -> tuple[Path, Path]:
    archive = output.parent / f"{output.name}.tar.gz"
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    temporary = archive.with_suffix(archive.suffix + f".tmp.{os.getpid()}")
    with tarfile.open(temporary, "w:gz") as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                handle.add(path, arcname=f"{output.name}/{path.relative_to(output)}", recursive=False)
        for relative in (SCRIPT_RELATIVE, TEST_RELATIVE, CONFIG_RELATIVE):
            path = REPO_ROOT / relative
            handle.add(path, arcname=f"{output.name}/source/{relative}", recursive=False)
        for relative in CRITICAL_SOURCE_RELATIVES:
            path = REPO_ROOT / relative
            handle.add(path, arcname=f"{output.name}/source/dependencies/{relative}", recursive=False)
    temporary.replace(archive)
    digest = sha256_file(archive)
    atomic_text(sidecar, f"{digest}  {archive.name}\n")
    return archive, sidecar


def analyse(config_path: Path) -> int:
    config = load_config(config_path)
    output, _historical, strict_root, _checkpoint = resolve_paths(config)
    record_command(output)
    source_hash_path = output / "source_and_checkpoint_hashes.json"
    source_hashes = load_json(source_hash_path)
    source_hashes["critical_preflight_source_files"] = critical_source_identities()
    source_hashes["embedded_original_head_provenance"] = embedded_head_provenance(
        strict_root, str(config["expected_base_checkpoint_sha256"])
    )
    if not source_hashes["embedded_original_head_provenance"]["all_embed_expected_base_checkpoint"]:
        raise RuntimeError("Original Head embedded base-VAE provenance mismatch")
    atomic_json(source_hash_path, source_hashes)
    write_dirty_tree_and_protected_audits(output)
    preflight = load_json(output / "preflight.json")
    anchors = load_json(output / "anchors.json")
    plan = load_json(output / "task_plan.json")
    valid_count = int(preflight.get("valid_proposal_count", 0))
    preflight_by_index = {int(row["anchor_index"]): row for row in preflight.get("anchors", [])}
    tasks = [dict(task) for task in plan.get("tasks", [])]
    result_rows: list[dict[str, Any]] = []
    duplicate_keys: set[str] = set()
    for path in sorted((output / "runs").glob("*.json")) if (output / "runs").is_dir() else []:
        row = load_json(path)
        hits = strict_result_key_hits(row)
        if hits:
            raise RuntimeError(f"forbidden metric fields in {path}: {hits}")
        key = task_result_key(row)
        if key in duplicate_keys:
            raise RuntimeError(f"duplicate task result: {key}")
        duplicate_keys.add(key)
        result_rows.append(row)
    expected_keys = {task_result_key(task) for task in tasks}
    result_by_key = {task_result_key(row): row for row in result_rows}
    missing = sorted(expected_keys - set(result_by_key))
    foreign = sorted(set(result_by_key) - expected_keys)
    failed = sorted(
        key for key in expected_keys & set(result_by_key)
        if result_by_key[key].get("terminal_status") != "success" or result_by_key[key].get("valid") is not True
    )

    parent_rows = load_json(output / "parent_results.json")
    if plan.get("parent_reuse"):
        result_rows.extend(parent_rows)
    raw_sorted = sorted(result_rows, key=lambda row: (int(row["anchor_index"]), int(row["seed"]), str(row["role"])))
    if strict_result_key_hits(raw_sorted):
        raise RuntimeError("forbidden metric field in raw result collection")
    atomic_text(
        output / "raw_results.jsonl",
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in raw_sorted),
    )

    grouped: dict[tuple[int, str], dict[int, float]] = {}
    for row in raw_sorted:
        if row.get("terminal_status") == "success" and row.get("valid") is True:
            grouped.setdefault((int(row["anchor_index"]), str(row["role"])), {})[int(row["seed"])] = float(row["val_acc"])
    seeds = {int(seed) for seed in config["training_seeds"]}
    statistics: list[dict[str, Any]] = []
    plus_minus_values: list[float] = []
    plus_parent_values: list[float] = []
    plus_minus_wins = 0
    plus_parent_wins = 0
    for index in range(int(config["target_anchor_count"])):
        proposal = preflight_by_index.get(index, {})
        row: dict[str, Any] = {
            "anchor_index": index,
            "parent_fingerprint": anchors[index]["architecture_fingerprint"],
            "plus_fingerprint": proposal.get("plus_fingerprint"),
            "minus_fingerprint": proposal.get("minus_fingerprint"),
        }
        if not proposal.get("valid_proposal"):
            row.update({"valid_statistics": False, "status": proposal.get("failure_reason", "INVALID_PROPOSAL")})
            statistics.append(row)
            continue
        role_values = {role: grouped.get((index, role), {}) for role in ("plus", "minus", "parent")}
        if any(set(role_values[role]) != seeds for role in role_values):
            row.update({"valid_statistics": False, "status": "INCOMPLETE_REQUIRED_SEEDS"})
            statistics.append(row)
            continue
        means = {role: float(np.mean([role_values[role][seed] for seed in sorted(seeds)])) for role in role_values}
        delta_pm = means["plus"] - means["minus"]
        delta_pp = means["plus"] - means["parent"]
        plus_minus_values.append(delta_pm)
        plus_parent_values.append(delta_pp)
        plus_minus_wins += int(delta_pm > 0)
        plus_parent_wins += int(delta_pp > 0)
        row.update({
            "valid_statistics": True,
            "status": "complete",
            "mean_plus": means["plus"],
            "mean_minus": means["minus"],
            "mean_parent": means["parent"],
            "delta_plus_minus": delta_pm,
            "delta_plus_parent": delta_pp,
            "matched_seed_plus_minus": {str(seed): role_values["plus"][seed] - role_values["minus"][seed] for seed in sorted(seeds)},
            "matched_seed_plus_parent": {str(seed): role_values["plus"][seed] - role_values["parent"][seed] for seed in sorted(seeds)},
        })
        statistics.append(row)

    complete = not missing and not foreign and not failed and (
        valid_count < int(config["minimum_valid_anchors"])
        or sum(bool(row.get("valid_statistics")) for row in statistics) == valid_count
    )
    mean_pm = float(np.mean(plus_minus_values)) if plus_minus_values else None
    mean_pp = float(np.mean(plus_parent_values)) if plus_parent_values else None
    if preflight.get("classification") in {"NO_DIFFERENTIABLE_FROZEN_HEAD", "INSUFFICIENT_SCALE_REFERENCE"}:
        classification = "NAO_DIRECTION_GATE_INCONCLUSIVE"
    elif valid_count < int(config["minimum_valid_anchors"]):
        classification = "NAO_DIRECTION_GATE_DECODE_REJECTED"
    elif not complete:
        classification = "NAO_DIRECTION_GATE_INCONCLUSIVE"
    else:
        classification = classify_gate(valid_count, plus_minus_wins, mean_pm, plus_parent_wins, mean_pp, config)

    failures: dict[str, int] = {}
    for row in preflight.get("anchors", []):
        reason = row.get("failure_reason")
        if reason:
            failures[str(reason)] = failures.get(str(reason), 0) + 1
    new_rows = [row for row in raw_sorted if row.get("result_origin") == "new_validation_only_run"]
    gpu_elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in new_rows)
    command_rows = []
    with (output / "commands.jsonl").open(encoding="utf-8") as handle:
        command_rows = [json.loads(line) for line in handle if line.strip()]
    first_at = command_rows[0]["at"] if command_rows else utc_now()
    now = utc_now()
    command_elapsed = max(0.0, iso_seconds(first_at, now))
    goal_elapsed = max(0.0, time.time() - float(config.get("execution_started_at_epoch", time.time())))
    actual_total = max(command_elapsed, goal_elapsed)
    explanation = {
        "NAO_DIRECTION_GATE_PASSED": "四项预注册实证门槛全部满足。",
        "NAO_DIRECTION_GATE_REJECTED": "至少一项预注册 validation 改善门槛未满足。",
        "NAO_DIRECTION_GATE_DECODE_REJECTED": f"只有 {valid_count}/8 个 anchor 产生合法成对 proposal，低于 7/8 门槛。",
        "NAO_DIRECTION_GATE_INCONCLUSIVE": f"required 任务或工程审计不完整：missing={len(missing)}, failed={len(failed)}, foreign={len(foreign)}。",
    }[classification]
    next_step = (
        "只建议另行做一次独立的单代候选生成确认；本 Goal 不自动执行。"
        if classification == "NAO_DIRECTION_GATE_PASSED"
        else "停止当前 latent search 路线。"
        if classification in {"NAO_DIRECTION_GATE_REJECTED", "NAO_DIRECTION_GATE_DECODE_REJECTED"}
        else "仅解决上述工程阻断后按同一冻结计划 resume。"
    )
    analysis_payload = {
        "classification": classification,
        "created_at": now,
        "valid_anchor_count": valid_count,
        "complete_statistical_anchor_count": len(plus_minus_values),
        "new_training_run_count": len(new_rows),
        "historical_parent_run_count": len(parent_rows) if plan.get("parent_reuse") else 0,
        "required_task_count": len(tasks),
        "missing_task_keys": missing,
        "failed_task_keys": failed,
        "foreign_task_keys": foreign,
        "plus_minus_win_count": plus_minus_wins,
        "plus_parent_win_count": plus_parent_wins,
        "mean_delta_plus_minus": mean_pm,
        "mean_delta_plus_parent": mean_pp,
        "median_delta_plus_minus": float(np.median(plus_minus_values)) if plus_minus_values else None,
        "median_delta_plus_parent": float(np.median(plus_parent_values)) if plus_parent_values else None,
        "sign_test_plus_minus_p": exact_one_sided_sign_p(plus_minus_wins, len(plus_minus_values)),
        "sign_test_plus_parent_p": exact_one_sided_sign_p(plus_parent_wins, len(plus_parent_values)),
        "bootstrap_plus_minus_ci95": bootstrap_ci(plus_minus_values, int(config["bootstrap_draws"]), int(config["bootstrap_seed"])),
        "bootstrap_plus_parent_ci95": bootstrap_ci(plus_parent_values, int(config["bootstrap_draws"]), int(config["bootstrap_seed"]) + 1),
        "leave_one_anchor_out_plus_minus": leave_one_out(plus_minus_values),
        "leave_one_anchor_out_plus_parent": leave_one_out(plus_parent_values),
        "preflight_failure_summary": failures,
        "decision_thresholds": {
            "valid_anchors": config["minimum_valid_anchors"],
            "plus_minus_wins": config["minimum_plus_minus_wins"],
            "mean_delta_plus_minus": config["minimum_mean_delta_plus_minus"],
            "plus_parent_wins": config["minimum_plus_parent_wins"],
            "mean_delta_plus_parent": config["minimum_mean_delta_plus_parent"],
        },
        "decision_explanation": explanation,
        "next_step": next_step,
        "timing": {"actual_total_seconds": actual_total, "summed_gpu_task_seconds": gpu_elapsed},
        "read_test_metrics": False,
    }
    atomic_json(output / "analysis.json", analysis_payload)
    csv_rows = []
    for row in statistics:
        csv_rows.append({
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        })
    atomic_csv(output / "anchor_statistics.csv", csv_rows)

    training_source = inspect.getsource(execute_validation_task)
    firewall = {
        "passed": not strict_result_key_hits(raw_sorted) and "track_test=False" in training_source.replace(" ", ""),
        "training_call_track_test": False,
        "prohibited_metric_key_hits": strict_result_key_hits(raw_sorted),
        "test_metric_read": False,
        "test_metric_calculated": False,
        "test_metric_saved": False,
        "test_metric_output": False,
    }
    atomic_json(output / "test_firewall_audit.json", firewall)
    if not firewall["passed"]:
        analysis_payload["classification"] = "NAO_DIRECTION_GATE_INCONCLUSIVE"
        analysis_payload["decision_explanation"] = "validation-only firewall audit failed。"
        atomic_json(output / "analysis.json", analysis_payload)
    write_report(output, analysis_payload, statistics)

    manifest_files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            identity = file_identity(path)
            identity["relative_path"] = str(path.relative_to(output))
            manifest_files.append(identity)
    manifest = {
        "created_at": utc_now(),
        "classification": analysis_payload["classification"],
        "git_head": run_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_text(["git", "status", "--short"]).splitlines(),
        "file_count": len(manifest_files),
        "files": manifest_files,
        "source_files": [
            file_identity(REPO_ROOT / relative)
            for relative in (SCRIPT_RELATIVE, TEST_RELATIVE, CONFIG_RELATIVE, *CRITICAL_SOURCE_RELATIVES)
        ],
        "protected_paths_touched": not load_json(output / "protected_state_audit.json")["passed"],
        "read_test_metrics": False,
    }
    atomic_json(output / "manifest.json", manifest)
    archive, sidecar = create_archive(output)
    print(json.dumps({
        "classification": analysis_payload["classification"],
        "output": str(output),
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "sidecar": str(sidecar),
    }, indent=2, sort_keys=True))
    return 0 if analysis_payload["classification"] != "NAO_DIRECTION_GATE_INCONCLUSIVE" else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    run_parser.add_argument("--device", required=True)
    run_parser.add_argument("--cora-root", default="Cora")
    run_parser.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("analyse")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    if args.command == "prepare":
        return prepare(config_path)
    if args.command == "run":
        return run_shard(config_path, args.shard, args.device, args.cora_root, args.dry_run)
    if args.command == "analyse":
        return analyse(config_path)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
