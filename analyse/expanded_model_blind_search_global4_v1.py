#!/usr/bin/env python3
"""Freeze and execute the Phase D expanded model-blind LayerNorm experiment.

The synthetic surface is CPU-only and uses fake data.  The dry-run verifies
the pinned identity inputs, freezes candidates and tasks without training a
GNN, and atomically publishes the immutable prelaunch tree.  Formal execution
is admitted only through the bound launch records and evaluates validation
accuracy behind the accepted Phase C data firewall.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
import ctypes
from datetime import datetime, timezone
import errno
import gzip
import hashlib
import inspect
import io
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse.fixed_layernorm_enhancement_v2 import (
    DATA_ALLOWLIST,
    TestAccessViolation,
    ValidationOnlyDataProxy,
    _aggregate_access_ledgers,
    _new_access_ledger,
    _proxy_internal_ledger,
    probe_track_test,
)
from evaluation_errors import InfrastructureEvaluationError


EXPERIMENT = "expanded-model-blind-search-global4-v1"
FORMAT_VERSION = "expanded-model-blind-search-global4-v1"
IMPLEMENTATION_REL = "analyse/expanded_model_blind_search_global4_v1.py"
CONFIG_REL = "configs/expanded_model_blind_search_global4_v1.json"
RUNNER_REL = "scripts/run_expanded_model_blind_search_global4_v1.sh"
TEST_REL = "tests/test_expanded_model_blind_search_global4_v1.py"
TARGET_RELS = (IMPLEMENTATION_REL, CONFIG_REL, RUNNER_REL, TEST_REL)
RESULT_REL = "results/posthoc/expanded-model-blind-search-global4-v1"
ARCHIVE_BASENAME = f"{EXPERIMENT}.tar.gz"
ARCHIVE_SIDECAR_BASENAME = f"{ARCHIVE_BASENAME}.sha256"
REPLICATE_IDS = tuple(f"{value:02d}" for value in range(1, 13))
SLOT_IDS = tuple(f"{value:02d}" for value in range(1, 17))
TRAINING_SEEDS = (0, 1, 2)
DECODE_BATCH_SIZE = 256
ARCHITECTURE_LATENT_DIMENSION = 12
MAX_REJECTIONS_PER_SLOT = 1000
METHOD_EQUIVALENT_TASK_COUNT = 1155
MAX_RETRIED_TASK_COUNT = 57

LAUNCH_RUNTIME_PATHS = {
    "launch/launch_request.json",
    "launch/runner_entered.json",
    "launch/runner_exit.json",
}
FORMAL_CORE_PATHS = {
    "formal/start.json",
    "formal/attempt_ledger.json",
    "formal/task_ledger.json",
    "formal/test_access_ledger.json",
    "formal/results.json",
    "formal/statistics.json",
    "formal/challengers.json",
    "formal/decision.json",
    "formal/state.json",
    "formal/terminal.json",
    "formal/manifest.json",
}
FORMAL_SHARD_PATHS = {f"formal/shards/worker_{worker_id}.json" for worker_id in range(4)}
SEALED_RUNTIME_PATHS = {ARCHIVE_BASENAME, ARCHIVE_SIDECAR_BASENAME, "completion.json"}
KNOWN_POST_PRELAUNCH_PATHS = (
    LAUNCH_RUNTIME_PATHS | FORMAL_CORE_PATHS | FORMAL_SHARD_PATHS | SEALED_RUNTIME_PATHS
)

PINNED_DEPENDENCIES = {
    "analyse/high_regime_prior_model_blind_validation.py":
        "4c6f5228abe7480c1a018b187483461505a272246dc13ef1084eafac2887ac39",
    "analyse/original_head_deterministic_batch_pilot.py":
        "381ab598aca5d75b82d5c4c152942fe96bc292a640979a2fcc4b3348a4ae6723",
    "architecture_diagnostics.py":
        "a1efc8a2ae21b8520d90166771ad434186a57147e0c7f01a842856abd00cf05f",
    "analyse/performance_aware_vae_strict_regression.py":
        "a2e722153f68c25eb593357b9313f7ec14408d0fbe34d8cc1b66f4820b503121",
    "analyse/structure_native_surrogate_pilot.py":
        "192771e783386a280b2c2c4b6934afdcbf41e6549c49d7ff7aea656b62e0ad96",
    "eval_utils.py":
        "0cbfb402db030173ca11e4e90570df6ac427d99b416470863f55280d51e2541d",
    "analyse/fixed_layernorm_enhancement_v2.py":
        "c766eb8f1032b5fb9eae1be5782f0d1335e7213e70264500dc67c8e6f7a60a54",
    "checkpoints/joint_model_pipeline_global4_best.pth":
        "e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4",
}

PINNED_GOVERNING_DOCUMENTS = {
    "plans/phase_a_e_frozen_master_protocol_2026-08-30.md":
        "e591c1ae6f966374885b06e1a0b76d201406a5460b8e4eec1e45d8e747e296eb",
    "plans/phase_c_v2_amendment_2026-08-30.md":
        "1d487f8bd5ddebc4a6b964f77975e3fc1931496ce4c800658b50cfa8640f4279",
    "Handoff.md":
        "ab3d5e96969a100e4c1e34f4118543cc88d388eb81f56f78a897d7d6b9a77d84",
}

PINNED_EXPOSURE_SOURCES = (
    {
        "path": (
            "results/search/high-regime-prior-model-blind-validation-global4-v1/"
            "exposure_registry.json"
        ),
        "size_bytes": 1498071,
        "sha256": "98c5f54345a41b90b1b89f0bf1a6d883974c36eadcb4291d93b529a068bb3d66",
        "architecture_count": 13871,
    },
    {
        "path": (
            "results/search/high-regime-prior-model-blind-validation-global4-v1/"
            "global_reservoir.json"
        ),
        "size_bytes": 164195,
        "sha256": "ff4bb7f06dc2424b6b686b2102d43f590c9cdaaf363f3ab9f020c3f50a63f619",
        "architecture_count": 192,
    },
)

P_ANCHOR_SOURCE = {
    "path": (
        "results/search/high-regime-prior-model-blind-validation-global4-v1/"
        "selected_union_configs.json"
    ),
    "sha256": "7ae5cb88664ffb7d3d33374766ff8b986f25f6d465ba3a65d968f1f03c73c792",
}

EXPECTED_P_ANCHOR = {
    "architecture_fingerprint":
        "960efb3b97f00491c3956657cfd57958effb6096f65b1db2fa8aaae586b82207",
    "hp_fingerprint":
        "911259943106a1edad300bc5300a3883ece8cbffba48b95e6c61b6e5eb4b679d",
    "joint_config_fingerprint":
        "30e0bb108c1fdbe8db186152a09c3f2fc6e812be189db920fe4d0d2b0d3cb1a2",
    "config_id": "hr_p22_b0_s4_960efb3b97f0_hp_high_91125994",
    "operations": ["GCNConv", "GATConv", "GCNII", "GCNII", "SAGEConv"],
    "edges": [
        [0, 1], [1, 2], [0, 2], [2, 3], [3, 4], [4, 5], [3, 5], [0, 5], [5, 6],
    ],
    "effective_layers": 5,
    "raw_hp_vector": [
        0.936598281105957,
        0.9034724664872731,
        0.8355336476052301,
        0.8524634455587309,
    ],
    "lr": 0.0219530947756673,
    "dropout": 0.5517362415790558,
    "hidden_dim": 512,
    "l2": 0.001,
    "gat_heads": 1,
    "sage_aggr": "mean",
    "gin_eps": 0.0,
    "gat_heads_by_layer": None,
    "sage_aggr_by_layer": None,
    "gin_eps_by_layer": None,
    "gcnii_alpha": None,
    "gcnii_theta": None,
    "hp_regime": "hp_high",
}

EXPECTED_PHASE_A = {
    "state": "COMPLETE",
    "classification": "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL",
    "boundary": False,
    "training": False,
    "test": False,
    "test_accessed": False,
    "phase_b_started": False,
    "phase_b_status": "NOT_RUN",
}

EXPECTED_PHASE_C = {
    "result_root": "results/posthoc/fixed-layernorm-enhancement-global4-v2-recovery1",
    "run_id": "phase-c-v2-14f9af76-6456-4303-a1a8-cbe5a116db7e",
    "classification": "FIXED_LN_RECIPE_AUTHORIZED_FOR_PHASE_D_TESTING",
    "formal_manifest_root_sha256":
        "e1d659676e3004c56ca72758dfe1698a5fbc29b00b9064fd5af9934c27b0e2d1",
    "archive_sha256":
        "c37714049f1fbcb8e2562cb4047b485242543c843ce409dfdc1ab1d4fb6579f8",
    "state": "COMPLETE",
    "scientific_decision": "REACHED",
    "planned_task_count": 240,
    "completed_task_count": 240,
    "failed_task_count": 0,
    "retry_task_count": 0,
    "test_accessed": False,
    "phase_d_started": False,
    "phase_e_started": False,
}

EXPECTED_WORKER_MAPPING = (
    {
        "worker_id": 0,
        "gpu_index": 0,
        "gpu_uuid": "GPU-5496be78-3b71-4978-3960-b6ed926ebe49",
    },
    {
        "worker_id": 1,
        "gpu_index": 1,
        "gpu_uuid": "GPU-197ae2b9-74cd-6afc-446d-7bc9e98176f9",
    },
    {
        "worker_id": 2,
        "gpu_index": 0,
        "gpu_uuid": "GPU-5496be78-3b71-4978-3960-b6ed926ebe49",
    },
    {
        "worker_id": 3,
        "gpu_index": 1,
        "gpu_uuid": "GPU-197ae2b9-74cd-6afc-446d-7bc9e98176f9",
    },
)

ARCHITECTURE_TOKEN_RE = re.compile(
    rb'"architecture_fingerprint"\s*:\s*"([0-9a-f]{64})"'
)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ProtocolFailure(RuntimeError):
    """A fail-closed Phase D protocol violation."""

    def __init__(self, stage: str, error_code: str):
        self.stage = str(stage)
        self.error_code = str(error_code)
        super().__init__(f"{self.stage}:{self.error_code}")


def _fail(stage: str, error_code: str) -> None:
    raise ProtocolFailure(stage, error_code)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def json_asset(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except OSError as exc:
        raise ProtocolFailure("INPUT_IDENTITY", "FILE_HASH_FAILED") from exc
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        _fail("INPUT_IDENTITY", "FILE_CHANGED_DURING_HASH")
    return digest.hexdigest()


def assert_file_identity(
    path: Path, expected_sha256: str, expected_size: int | None = None
) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ProtocolFailure("INPUT_IDENTITY", "REQUIRED_FILE_MISSING") from exc
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        _fail("INPUT_IDENTITY", "REQUIRED_REGULAR_FILE_MISSING")
    size = path.stat().st_size
    if expected_size is not None and size != int(expected_size):
        _fail("INPUT_IDENTITY", "REQUIRED_FILE_SIZE_MISMATCH")
    observed = sha256_path(path)
    if observed != expected_sha256:
        _fail("INPUT_IDENTITY", "REQUIRED_FILE_SHA256_MISMATCH")
    return {"path": str(path), "size_bytes": size, "sha256": observed}


def _read_pinned_bytes(
    path: Path, expected_sha256: str, expected_size: int | None = None
) -> bytes:
    assert_file_identity(path, expected_sha256, expected_size)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ProtocolFailure("INPUT_IDENTITY", "REQUIRED_FILE_READ_FAILED") from exc
    if (expected_size is not None and len(data) != int(expected_size)) or sha256_bytes(data) != expected_sha256:
        _fail("INPUT_IDENTITY", "REQUIRED_FILE_CHANGED_DURING_READ")
    return data


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate_json_key")
        output[key] = value
    return output


def _load_json_file(path: Path, stage: str, error_code: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolFailure(stage, error_code) from exc


def _governing_map(rows: Any) -> dict[str, str]:
    if not isinstance(rows, list):
        _fail("CONFIG", "GOVERNING_DOCUMENT_SCHEMA_MISMATCH")
    try:
        return {str(row["path"]): str(row["sha256"]) for row in rows}
    except (KeyError, TypeError) as exc:
        raise ProtocolFailure("CONFIG", "GOVERNING_DOCUMENT_SCHEMA_MISMATCH") from exc


def load_config(path: Path) -> dict[str, Any]:
    config = _load_json_file(path, "CONFIG", "CONFIG_LOAD_FAILED")
    if not isinstance(config, dict):
        _fail("CONFIG", "CONFIG_ROOT_NOT_OBJECT")
    checks = (
        config.get("schema_version") == FORMAT_VERSION,
        config.get("experiment") == EXPERIMENT,
        config.get("repository_root") == str(REPO_ROOT),
        config.get("required_host") == "ismc36",
        config.get("result_root") == RESULT_REL,
        config.get("tmux_name") == "expanded_model_blind_search_v1",
        config.get("result_root_must_be_absent_before_any_write") is True,
        config.get("phase_e_must_never_start") is True,
        _governing_map(config.get("governing_documents")) == PINNED_GOVERNING_DOCUMENTS,
        config.get("governing_documents_available_to_outer_controller_only") is True,
        config.get("dependency_hashes") == PINNED_DEPENDENCIES,
        config.get("accepted_upstream", {}).get("phase_a") == EXPECTED_PHASE_A,
        config.get("accepted_upstream", {}).get("phase_c") == EXPECTED_PHASE_C,
        config.get("accepted_upstream", {}).get(
            "use_as_declared_identity_only_without_rereading_outcome_values"
        ) is True,
        config.get("historical_exposure", {}).get("artifacts") == list(PINNED_EXPOSURE_SOURCES),
        config.get("historical_exposure", {}).get("input_overlap_count") == 0,
        config.get("historical_exposure", {}).get("union_architecture_count") == 14063,
        config.get("p_anchor", {}).get("identity_source", {}).get("path")
            == P_ANCHOR_SOURCE["path"],
        config.get("p_anchor", {}).get("identity_source", {}).get("sha256")
            == P_ANCHOR_SOURCE["sha256"],
        config.get("p_anchor", {}).get("row") == EXPECTED_P_ANCHOR,
        config.get("search_design", {}).get("replicate_ids") == list(REPLICATE_IDS),
        config.get("search_design", {}).get("slot_ids") == list(SLOT_IDS),
        config.get("search_design", {}).get("training_seeds") == list(TRAINING_SEEDS),
        config.get("search_design", {}).get("arms") == [
            {"arm": "Control", "component_variant": "baseline"},
            {"arm": "Intervention", "component_variant": "layer_norm"},
        ],
        config.get("search_design", {}).get("paired_unit_count") == 576,
        config.get("search_design", {}).get("search_task_count") == 1152,
        config.get("search_design", {}).get("task_counts") == {
            "Control": 576,
            "Intervention": 576,
            "P_anchor": 3,
            "method_equivalent_total": 1155,
        },
        config.get("generation", {}).get("architecture_latent_dimension") == 12,
        config.get("generation", {}).get("torch_generator_device") == "cpu",
        config.get("generation", {}).get("architecture_distribution") == "standard_normal",
        config.get("generation", {}).get("decode_batch_size") == 256,
        config.get("generation", {}).get("ordered_stream_consumption") is True,
        config.get("generation", {}).get("global_permutation_forbidden") is True,
        config.get("generation", {}).get("seed_derivation", {}).get("signed") is False,
        config.get("generation", {}).get("seed_derivation", {}).get(
            "architecture_material_template"
        ) == (
            "expanded-model-blind-search-global4-v1|architecture-slots|"
            "replicate={replicate_id}|2026-08-30"
        ),
        config.get("generation", {}).get("seed_derivation", {}).get(
            "raw_hp_material_template"
        ) == (
            "expanded-model-blind-search-global4-v1|raw-hp-uniforms|"
            "replicate={replicate_id}|2026-08-30"
        ),
        config.get("generation", {}).get("architecture_acceptance", {}).get(
            "maximum_allowed_rejections_per_slot"
        ) == 1000,
        config.get("generation", {}).get("raw_hp", {}).get("low") == 0.8,
        config.get("generation", {}).get("raw_hp", {}).get("high") == 0.95,
        config.get("generation", {}).get("raw_hp", {}).get("size") == 4,
        config.get("generation", {}).get("raw_hp", {}).get("dtype") == "float64",
        config.get("generation", {}).get("raw_hp", {}).get("coordinate_order") == [
            "u_lr", "u_dropout", "u_hidden", "u_l2"
        ],
        config.get("generation", {}).get("raw_hp", {}).get(
            "consume_only_after_architecture_acceptance"
        ) is True,
        config.get("training_protocol", {}).get("max_epochs") == 300,
        config.get("training_protocol", {}).get("patience") == 80,
        config.get("training_protocol", {}).get("hp_regime") == "hp_high",
        config.get("training_protocol", {}).get("seeds") == [0, 1, 2],
        config.get("training_protocol", {}).get("p_anchor_uses_same_protocol") is True,
        config.get("training_protocol", {}).get("layer_norm_contract") == {
            "order": ["operation", "LayerNorm", "activation", "dropout"],
            "identity_gets_layer_norm": False,
            "control_has_layer_norm": False,
            "intervention_has_only_layer_norm_change": True,
        },
        all(
            config.get("training_protocol", {}).get(key) is False
            for key in (
                "hidden_or_l2_expansion", "max_epochs_600_or_patience_160", "surrogate",
                "predictor", "bayesian_optimization", "diversity_selection",
                "adaptive_allocation", "candidate_replacement", "seed_replacement",
                "result_imputation",
            )
        ),
        config.get("workers", {}).get("count") == 4,
        config.get("workers", {}).get("gpus") == [
            {"index": 0, "uuid": EXPECTED_WORKER_MAPPING[0]["gpu_uuid"]},
            {"index": 1, "uuid": EXPECTED_WORKER_MAPPING[1]["gpu_uuid"]},
        ],
        config.get("workers", {}).get("mapping") == list(EXPECTED_WORKER_MAPPING),
        config.get("pair_equality", {}).get("only_fields_allowed_to_differ") == [
            "task_id", "arm", "component_variant"
        ],
        config.get("evaluation", {}).get("formal_track_test_literal") is False,
        config.get("evaluation", {}).get("data_allowlist")
            == ["x", "edge_index", "train_mask", "val_mask", "y"],
        {
            key: config.get("evaluation", {}).get(key)
            for key in ("dataset", "dataset_root", "split", "transform", "full_batch")
        } == {
            "dataset": "Cora",
            "dataset_root": "Cora",
            "split": "public",
            "transform": "NormalizeFeatures",
            "full_batch": True,
        },
        config.get("retry", {}).get("maximum_allowed_retried_task_count") == 57,
        config.get("retry", {}).get("maximum_retries_per_task") == 2,
        config.get("retry", {}).get("maximum_attempts_per_task") == 3,
        config.get("retry", {}).get("retryable_exception_types") == [
            "InfrastructureEvaluationError", "OSError", "TimeoutError"
        ],
        config.get("retry", {}).get("fail_immediately_on_retried_task_count") == 58,
        config.get("retry", {}).get("method_equivalent_task_count") == 1155,
        config.get("lifecycle", {}).get("modes") == [
            "synthetic-smoke", "dry-run", "record-launch-request", "runner-enter",
            "formal-run", "runner-exit",
        ],
        config.get("prelaunch", {}).get("source_snapshot_paths") == list(TARGET_RELS),
        config.get("prelaunch", {}).get("source_snapshot_count") == 4,
        set(config.get("prelaunch", {}).get("required_files", [])) == {
            "frozen_protocol.json", "inference_plan.json", "input_hashes.json",
            "frozen_exposure.json", "frozen_candidates.json", "frozen_tasks.json",
            "assignments.json", "test_firewall.json", "pre_execution_estimate.json",
            "environment.json", "state.json", "manifest.json",
            *(f"source_snapshot/{relative}" for relative in TARGET_RELS),
        },
        config.get("formal_artifacts", {}).get("archive", {}).get("basename")
            == ARCHIVE_BASENAME,
        config.get("formal_artifacts", {}).get("archive", {}).get("sidecar_basename")
            == ARCHIVE_SIDECAR_BASENAME,
        config.get("formal_artifacts", {}).get("archive", {}).get("exclude_from_archive") == [
            ARCHIVE_BASENAME, ARCHIVE_SIDECAR_BASENAME, "completion.json"
        ],
        config.get("formal_artifacts", {}).get("required") == [
            "formal/start.json", "formal/attempt_ledger.json", "formal/task_ledger.json",
            "formal/test_access_ledger.json", "formal/results.json",
            "formal/statistics.json", "formal/challengers.json", "formal/decision.json",
            "formal/state.json", "formal/terminal.json", "formal/manifest.json",
            ARCHIVE_BASENAME, ARCHIVE_SIDECAR_BASENAME,
        ],
        config.get("inference", {}).get("sign_flip", {}).get("sign_vector_count") == 4096,
        config.get("inference", {}).get("sign_flip", {}).get(
            "monte_carlo_plus_one_correction"
        ) is False,
        config.get("inference", {}).get("bootstrap", {}).get("replicates") == 100000,
        config.get("inference", {}).get("bootstrap", {}).get("lower_quantile") == 0.05,
        config.get("inference", {}).get("bootstrap", {}).get("quantile_method") == "linear",
        config.get("inference", {}).get("classification", {}).get("gates") == {
            "mean_delta_minimum_inclusive": 0.003,
            "one_sided_p_value_maximum_exclusive": 0.05,
            "bootstrap_lower_bound_minimum_exclusive": 0.0,
            "wins_minimum_inclusive": 8,
        },
        config.get("inference", {}).get("classification", {}).get("pass")
            == "EXPANDED_SEARCH_CONFIRMED_ON_CORA",
        config.get("inference", {}).get("classification", {}).get("complete_gate_miss")
            == "EXPANDED_SEARCH_NOT_CONFIRMED_ON_CORA",
        config.get("challengers", {}).get(
            "minimum_mean_advantage_over_fresh_p_anchor_inclusive"
        ) == 0.001,
        config.get("challengers", {}).get("maximum_selected") == 2,
        config.get("challengers", {}).get("start_phase_e") is False,
        config.get("runtime_estimate_minutes") == {
            "best": 165,
            "expected": 210,
            "worst_reference": 360,
        },
        config.get("terminal_contract", {}).get("complete_scientific_terminal") == {
            "planned_task_count": 1155,
            "completed_task_count": 1155,
            "failed_task_count": 0,
            "Control_completed_task_count": 576,
            "Intervention_completed_task_count": 576,
            "P_anchor_completed_task_count": 3,
            "complete_replicate_count": 12,
            "test_accessed": False,
            "phase_e_started": False,
        },
        config.get("terminal_contract", {}).get("run_failure_state") == "RUN_FAILED",
        config.get("terminal_contract", {}).get("run_failure_completion") == "INCOMPLETE",
    )
    if not all(checks):
        _fail("CONFIG", "FROZEN_CONTRACT_MISMATCH")
    return config


def _require_fixed_config_path(config_path: Path) -> Path:
    fixed = REPO_ROOT / CONFIG_REL
    try:
        observed = config_path.resolve(strict=True)
        expected = fixed.resolve(strict=True)
    except OSError as exc:
        raise ProtocolFailure("CONFIG", "FIXED_CONFIG_PATH_UNRESOLVED") from exc
    if observed != expected or config_path.is_symlink():
        _fail("CONFIG", "ALTERNATE_CONFIG_PATH_FORBIDDEN")
    return fixed


def derive_stream_seed(replicate_id: str, stream_name: str) -> int:
    if replicate_id not in REPLICATE_IDS:
        _fail("SEED_DERIVATION", "REPLICATE_ID_INVALID")
    if stream_name not in {"architecture-slots", "raw-hp-uniforms"}:
        _fail("SEED_DERIVATION", "STREAM_NAME_INVALID")
    material = (
        f"{EXPERIMENT}|{stream_name}|replicate={replicate_id}|2026-08-30"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def joint_config_fingerprint(architecture_fp: str, hp_fp: str) -> str:
    if not HEX64_RE.fullmatch(str(architecture_fp)) or not HEX64_RE.fullmatch(str(hp_fp)):
        _fail("FINGERPRINT", "JOINT_INPUT_FINGERPRINT_INVALID")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "architecture_fingerprint": str(architecture_fp),
                "hp_fingerprint": str(hp_fp),
            }
        )
    )


def load_exposure_basis(
    config: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    configured = config.get("historical_exposure", {}).get("artifacts")
    if configured != list(PINNED_EXPOSURE_SOURCES):
        _fail("EXPOSURE", "EXPOSURE_SOURCE_CONTRACT_MISMATCH")
    sets: list[set[str]] = []
    sources: list[dict[str, Any]] = []
    for expected in PINNED_EXPOSURE_SOURCES:
        path = repo_root / expected["path"]
        data = _read_pinned_bytes(path, expected["sha256"], expected["size_bytes"])
        matches = [value.decode("ascii") for value in ARCHITECTURE_TOKEN_RE.findall(data)]
        identities = set(matches)
        if len(matches) != expected["architecture_count"] or len(identities) != len(matches):
            _fail("EXPOSURE", "EXPOSURE_ARCHITECTURE_COUNT_MISMATCH")
        sets.append(identities)
        sources.append(
            {
                "path": expected["path"],
                "size_bytes": expected["size_bytes"],
                "sha256": expected["sha256"],
                "architecture_count": len(identities),
            }
        )
    overlap = sets[0] & sets[1]
    union = sets[0] | sets[1]
    if overlap or len(union) != 14063:
        _fail("EXPOSURE", "EXPOSURE_UNION_CONTRACT_MISMATCH")
    ordered = sorted(union)
    return {
        "schema_version": "phase-d-exposure-basis-v1",
        "sources": sources,
        "architecture_fingerprints": ordered,
        "source_counts": [len(values) for values in sets],
        "overlap_count": 0,
        "union_count": len(ordered),
        "commitment_sha256": sha256_bytes(canonical_json_bytes(ordered)),
        "identity_only_projection": True,
        "outcome_values_deserialized": False,
        "test_values_deserialized": False,
    }


def _smallest_json_object_containing(data: bytes, position: int) -> bytes:
    stack: list[int] = []
    candidates: list[tuple[int, int]] = []
    in_string = False
    escaped = False
    for index, byte in enumerate(data):
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte == 0x7B:
            stack.append(index)
        elif byte == 0x7D:
            if not stack:
                _fail("P_ANCHOR", "ANCHOR_SOURCE_JSON_STRUCTURE_INVALID")
            start = stack.pop()
            if start <= position <= index:
                candidates.append((start, index + 1))
    if in_string or stack or not candidates:
        _fail("P_ANCHOR", "ANCHOR_SOURCE_JSON_STRUCTURE_INVALID")
    start, end = min(candidates, key=lambda bounds: bounds[1] - bounds[0])
    return data[start:end]


def project_p_anchor(
    config: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    source = config.get("p_anchor", {}).get("identity_source", {})
    expected = config.get("p_anchor", {}).get("row")
    if (
        source.get("path") != P_ANCHOR_SOURCE["path"]
        or source.get("sha256") != P_ANCHOR_SOURCE["sha256"]
        or expected != EXPECTED_P_ANCHOR
    ):
        _fail("P_ANCHOR", "ANCHOR_CONTRACT_MISMATCH")
    data = _read_pinned_bytes(repo_root / source["path"], source["sha256"])
    token = f'"config_id": "{EXPECTED_P_ANCHOR["config_id"]}"'.encode("utf-8")
    positions = [match.start() for match in re.finditer(re.escape(token), data)]
    if len(positions) != 1:
        _fail("P_ANCHOR", "ANCHOR_CONFIG_ID_CARDINALITY_MISMATCH")
    object_bytes = _smallest_json_object_containing(data, positions[0])
    try:
        row = json.loads(object_bytes, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolFailure("P_ANCHOR", "ANCHOR_ROW_DECODE_FAILED") from exc
    stored_keys = {
        "architecture_fingerprint", "architecture_slot", "assigned_gpu", "batch_id",
        "config_id", "dropout", "edges", "effective_layers", "gat_heads",
        "gat_heads_by_layer", "gin_eps", "gin_eps_by_layer", "hidden_dim",
        "hp_fingerprint", "hp_regime", "joint_config_fingerprint", "l2", "lr",
        "operations", "pool_id", "raw_hp_vector", "sage_aggr",
        "sage_aggr_by_layer", "selected_by", "task_stage",
    }
    if not isinstance(row, dict) or set(row) != stored_keys:
        _fail("P_ANCHOR", "ANCHOR_ROW_SCHEMA_MISMATCH")
    projected = {
        key: row[key]
        for key in EXPECTED_P_ANCHOR
        if key not in {"gcnii_alpha", "gcnii_theta"}
    }
    projected["gcnii_alpha"] = None
    projected["gcnii_theta"] = None
    projected = {key: projected[key] for key in EXPECTED_P_ANCHOR}
    if projected != EXPECTED_P_ANCHOR:
        _fail("P_ANCHOR", "ANCHOR_EXACT_ROW_MISMATCH")
    from analyse import original_head_deterministic_batch_pilot as legacy

    architecture = {
        "operations": projected["operations"],
        "edges": projected["edges"],
        "effective_layers": projected["effective_layers"],
    }
    valid, reason = legacy.validate_static_config(architecture)
    if not valid or reason is not None:
        _fail("P_ANCHOR", "ANCHOR_STATIC_VALIDITY_MISMATCH")
    if legacy.architecture_fingerprint(architecture) != projected["architecture_fingerprint"]:
        _fail("P_ANCHOR", "ANCHOR_ARCHITECTURE_FINGERPRINT_MISMATCH")
    if legacy.hp_fingerprint(projected["raw_hp_vector"]) != projected["hp_fingerprint"]:
        _fail("P_ANCHOR", "ANCHOR_HP_FINGERPRINT_MISMATCH")
    if joint_config_fingerprint(
        projected["architecture_fingerprint"], projected["hp_fingerprint"]
    ) != projected["joint_config_fingerprint"]:
        _fail("P_ANCHOR", "ANCHOR_JOINT_FINGERPRINT_MISMATCH")
    realized = legacy.realize_global4_hp(projected)
    decoded_keys = (
        "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
        "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer",
    )
    if {key: realized[key] for key in decoded_keys} != {
        key: projected[key] for key in decoded_keys
    }:
        _fail("P_ANCHOR", "ANCHOR_HP_DECODE_MISMATCH")
    return projected


def consume_candidate_stream(
    decoder: Callable[[Any, int], Sequence[Any]],
    architecture_seed: int,
    hp_seed: int,
    historical_exposure: set[str],
    global_seen: set[str],
    replicate_id: str,
    *,
    slot_count: int = 16,
    decode_batch_size: int = 256,
    max_rejections_per_slot: int = 1000,
    converter: Callable[[Any], Mapping[str, Any]] | None = None,
    validator: Callable[[Mapping[str, Any]], tuple[bool, str | None]] | None = None,
    architecture_fingerprinter: Callable[[Mapping[str, Any]], str] | None = None,
    hp_fingerprinter: Callable[[Sequence[float]], str] | None = None,
    hp_realizer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if replicate_id not in REPLICATE_IDS:
        _fail("CANDIDATE_GENERATION", "REPLICATE_ID_INVALID")
    if slot_count < 1 or slot_count > 16 or decode_batch_size != 256:
        _fail("CANDIDATE_GENERATION", "STREAM_SHAPE_CONTRACT_MISMATCH")
    if max_rejections_per_slot != 1000:
        _fail("CANDIDATE_GENERATION", "REJECTION_LIMIT_CONTRACT_MISMATCH")
    if not (0 <= int(architecture_seed) < 2**64 and 0 <= int(hp_seed) < 2**64):
        _fail("CANDIDATE_GENERATION", "UNSIGNED_SEED_RANGE_INVALID")
    if any(not HEX64_RE.fullmatch(value) for value in historical_exposure | global_seen):
        _fail("CANDIDATE_GENERATION", "EXCLUSION_FINGERPRINT_INVALID")

    from analyse import original_head_deterministic_batch_pilot as legacy
    import numpy as np
    import torch

    convert = converter or (lambda graph: legacy.config_from_igraph(decoder.model, graph))
    validate = validator or legacy.validate_static_config
    architecture_fp = architecture_fingerprinter or legacy.architecture_fingerprint
    hp_fp = hp_fingerprinter or legacy.hp_fingerprint
    realize_hp = hp_realizer or legacy.realize_global4_hp

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(architecture_seed))
    hp_rng = np.random.default_rng(int(hp_seed))
    batch: list[Any] = []
    batch_offset = 0
    generated = 0
    consumed = 0
    batch_count = 0
    reason_counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    per_slot: list[dict[str, Any]] = []

    for slot_number in range(1, slot_count + 1):
        rejection_count = 0
        slot_reasons: Counter[str] = Counter()
        while True:
            if batch_offset == len(batch):
                decoded = list(decoder(generator, decode_batch_size))
                if len(decoded) != decode_batch_size:
                    _fail("CANDIDATE_GENERATION", "DECODE_BATCH_CARDINALITY_MISMATCH")
                batch = decoded
                batch_offset = 0
                generated += decode_batch_size
                batch_count += 1
            graph = batch[batch_offset]
            batch_offset += 1
            consumed += 1
            rejection_reason: str | None = None
            try:
                config_row = dict(convert(graph))
                fingerprint = str(architecture_fp(config_row))
            except (TypeError, ValueError, KeyError, IndexError) as exc:
                rejection_reason = f"conversion_failure:{type(exc).__name__}"
                config_row = {}
                fingerprint = ""
            if rejection_reason is None:
                valid, static_reason = validate(config_row)
                if not valid:
                    rejection_reason = f"legacy_static_invalid:{static_reason}"
                elif not HEX64_RE.fullmatch(fingerprint):
                    _fail("CANDIDATE_GENERATION", "ARCHITECTURE_FINGERPRINT_INVALID")
                elif fingerprint in historical_exposure:
                    rejection_reason = "historical_exposure"
                elif fingerprint in global_seen:
                    rejection_reason = "phase_d_duplicate"
            if rejection_reason is not None:
                rejection_count += 1
                reason_counts[rejection_reason] += 1
                slot_reasons[rejection_reason] += 1
                if rejection_count > max_rejections_per_slot:
                    _fail("CANDIDATE_GENERATION", "SLOT_REJECTION_1001_LIMIT_EXCEEDED")
                continue

            raw_array = hp_rng.uniform(0.80, 0.95, size=4)
            if raw_array.dtype != np.float64 or raw_array.shape != (4,):
                _fail("CANDIDATE_GENERATION", "RAW_HP_STREAM_SHAPE_OR_DTYPE_MISMATCH")
            raw_hp = raw_array.tolist()
            hp_fingerprint_value = str(hp_fp(raw_hp))
            if not HEX64_RE.fullmatch(hp_fingerprint_value):
                _fail("CANDIDATE_GENERATION", "HP_FINGERPRINT_INVALID")
            slot_id = f"{slot_number:02d}"
            candidate: dict[str, Any] = {
                "replicate_id": replicate_id,
                "slot_id": slot_id,
                "architecture_slot": slot_id,
                "architecture_seed": int(architecture_seed),
                "raw_hp_seed": int(hp_seed),
                "generation_index": consumed - 1,
                "architecture_source":
                    "global_model_blind_vae_standard_normal_deterministic_greedy",
                "architecture_fingerprint": fingerprint,
                "operations": [str(value) for value in config_row["operations"]],
                "edges": [list(map(int, edge)) for edge in config_row["edges"]],
                "effective_layers": int(config_row["effective_layers"]),
                "raw_hp_vector": raw_hp,
                "hp_fingerprint": hp_fingerprint_value,
                "hp_regime": "hp_high",
                "rejection_count_before_acceptance": rejection_count,
            }
            candidate["decoded_hp"] = dict(realize_hp(candidate))
            candidate["joint_config_fingerprint"] = joint_config_fingerprint(
                fingerprint, hp_fingerprint_value
            )
            candidate["config_id"] = (
                f"phase_d_r{replicate_id}_s{slot_id}_{fingerprint[:12]}_"
                f"hp_high_{hp_fingerprint_value[:8]}"
            )
            global_seen.add(fingerprint)
            candidates.append(candidate)
            per_slot.append(
                {
                    "slot_id": slot_id,
                    "accepted_generation_index": consumed - 1,
                    "architecture_fingerprint": fingerprint,
                    "rejection_count": rejection_count,
                    "rejection_reason_counts": dict(sorted(slot_reasons.items())),
                }
            )
            break

    unused = generated - consumed
    if unused != len(batch) - batch_offset or unused < 0:
        _fail("CANDIDATE_GENERATION", "UNUSED_BATCH_ACCOUNTING_MISMATCH")
    audit = {
        "replicate_id": replicate_id,
        "architecture_seed": int(architecture_seed),
        "raw_hp_seed": int(hp_seed),
        "decode_batch_size": decode_batch_size,
        "decode_batch_count": batch_count,
        "generated_draw_count": generated,
        "consumed_draw_count": consumed,
        "unused_generated_draw_count": unused,
        "unused_final_batch_values_reassigned": False,
        "accepted_slot_count": len(candidates),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "per_slot": per_slot,
    }
    return {"candidates": candidates, "audit": audit}


def generate_all_candidates(
    config: Mapping[str, Any], exposure: Mapping[str, Any]
) -> dict[str, Any]:
    if exposure.get("union_count") != 14063:
        _fail("CANDIDATE_GENERATION", "EXPOSURE_NOT_FROZEN")
    from analyse import original_head_deterministic_batch_pilot as legacy
    import torch

    checkpoint = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
    model = legacy.load_joint_vae(checkpoint, "global4", arch_nz=12, device="cpu")
    model.arch_vae.eval()
    historical = set(exposure["architecture_fingerprints"])
    global_seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    def decoder(generator: Any, batch_size: int) -> Sequence[Any]:
        latent = torch.randn(
            batch_size,
            ARCHITECTURE_LATENT_DIMENSION,
            generator=generator,
            device="cpu",
        )
        with torch.no_grad():
            return model.arch_vae.decode(latent, stochastic=False)

    decoder.model = model  # type: ignore[attr-defined]
    for replicate_id in REPLICATE_IDS:
        result = consume_candidate_stream(
            decoder,
            derive_stream_seed(replicate_id, "architecture-slots"),
            derive_stream_seed(replicate_id, "raw-hp-uniforms"),
            historical,
            global_seen,
            replicate_id,
        )
        candidates.extend(result["candidates"])
        audits.append(result["audit"])
    if (
        len(candidates) != 192
        or len(global_seen) != 192
        or global_seen & historical
        or [(row["replicate_id"], row["slot_id"]) for row in candidates]
        != [(replicate, slot) for replicate in REPLICATE_IDS for slot in SLOT_IDS]
    ):
        _fail("CANDIDATE_GENERATION", "GLOBAL_CANDIDATE_FREEZE_MISMATCH")
    return {
        "schema_version": "phase-d-candidate-freeze-v1",
        "candidate_count": len(candidates),
        "global_unique_architecture_count": len(global_seen),
        "historical_overlap_count": len(global_seen & historical),
        "candidate_commitment_sha256": sha256_bytes(canonical_json_bytes(candidates)),
        "candidates": candidates,
        "generation_audits": audits,
    }


def _decoded_hp_view(row: Mapping[str, Any]) -> dict[str, Any]:
    decoded = dict(row["decoded_hp"])
    expected_keys = {
        "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
        "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer",
    }
    if set(decoded) != expected_keys:
        _fail("TASK_FREEZE", "DECODED_HP_SCHEMA_MISMATCH")
    return decoded


def _training_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "Cora",
        "dataset_root": "Cora",
        "split": "public",
        "transform": "NormalizeFeatures",
        "full_batch": True,
        "hp_regime": "hp_high",
        "optimizer_checkpoint_evaluator": config["training_protocol"][
            "optimizer_checkpoint_and_evaluator"
        ],
        "checkpoint_rule": config["evaluation"]["checkpoint_rule"],
        "track_test": False,
        "surrogate": False,
        "predictor": False,
        "bayesian_optimization": False,
        "adaptive_allocation": False,
        "candidate_replacement": False,
        "seed_replacement": False,
        "result_imputation": False,
    }


def _task_base(
    row: Mapping[str, Any], seed: int, pair_index: int, mapping: Mapping[str, Any],
    training_semantics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "pair_id": f"pair-{pair_index:03d}",
        "pair_index": pair_index,
        "replicate_id": str(row["replicate_id"]),
        "slot_id": str(row["slot_id"]),
        "config_id": str(row["config_id"]),
        "architecture_fingerprint": str(row["architecture_fingerprint"]),
        "hp_fingerprint": str(row["hp_fingerprint"]),
        "joint_config_fingerprint": str(row["joint_config_fingerprint"]),
        "architecture": {
            "operations": list(row["operations"]),
            "edges": [list(edge) for edge in row["edges"]],
            "effective_layers": int(row["effective_layers"]),
        },
        "raw_hp_vector": list(row["raw_hp_vector"]),
        "decoded_hp": _decoded_hp_view(row),
        "hp_regime": "hp_high",
        "seed": int(seed),
        "max_epochs": 300,
        "patience": 80,
        "track_test": False,
        "training_semantics": dict(training_semantics),
        "worker_id": int(mapping["worker_id"]),
        "gpu_index": int(mapping["gpu_index"]),
        "gpu_uuid": str(mapping["gpu_uuid"]),
    }


def build_task_freeze(
    candidates: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(candidates) != 192 or anchor != EXPECTED_P_ANCHOR:
        _fail("TASK_FREEZE", "CANDIDATE_OR_ANCHOR_CARDINALITY_MISMATCH")
    ordered_candidates = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (str(row["replicate_id"]), str(row["slot_id"])),
    )
    expected_order = [(replicate, slot) for replicate in REPLICATE_IDS for slot in SLOT_IDS]
    if [(row["replicate_id"], row["slot_id"]) for row in ordered_candidates] != expected_order:
        _fail("TASK_FREEZE", "CANDIDATE_REPLICATE_SLOT_ORDER_MISMATCH")
    if len({row["architecture_fingerprint"] for row in ordered_candidates}) != 192:
        _fail("TASK_FREEZE", "CANDIDATE_ARCHITECTURE_NOT_GLOBALLY_UNIQUE")
    mapping_by_worker = {
        int(row["worker_id"]): row for row in config["workers"]["mapping"]
    }
    if tuple(config["workers"]["mapping"]) != EXPECTED_WORKER_MAPPING:
        _fail("TASK_FREEZE", "WORKER_MAPPING_MISMATCH")
    semantics = _training_semantics(config)
    tasks: list[dict[str, Any]] = []
    pair_index = 0
    for row in ordered_candidates:
        for seed in TRAINING_SEEDS:
            worker_id = pair_index % 4
            common = _task_base(
                row, seed, pair_index, mapping_by_worker[worker_id], semantics
            )
            control = {
                "task_id": (
                    f"search|r{row['replicate_id']}|s{row['slot_id']}|seed{seed}|Control"
                ),
                "arm": "Control",
                "component_variant": "baseline",
                **copy.deepcopy(common),
            }
            intervention = {
                "task_id": (
                    f"search|r{row['replicate_id']}|s{row['slot_id']}|seed{seed}|Intervention"
                ),
                "arm": "Intervention",
                "component_variant": "layer_norm",
                **copy.deepcopy(common),
            }
            tasks.extend((control, intervention))
            pair_index += 1
    anchor_decoded = {
        key: anchor[key]
        for key in (
            "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
            "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer",
        )
    }
    for seed in TRAINING_SEEDS:
        worker_id = seed
        mapping = mapping_by_worker[worker_id]
        tasks.append(
            {
                "task_id": f"p_anchor|seed{seed}",
                "arm": "P_anchor",
                "component_variant": "baseline",
                "pair_id": None,
                "pair_index": None,
                "replicate_id": None,
                "slot_id": None,
                "config_id": anchor["config_id"],
                "architecture_fingerprint": anchor["architecture_fingerprint"],
                "hp_fingerprint": anchor["hp_fingerprint"],
                "joint_config_fingerprint": anchor["joint_config_fingerprint"],
                "architecture": {
                    "operations": list(anchor["operations"]),
                    "edges": [list(edge) for edge in anchor["edges"]],
                    "effective_layers": int(anchor["effective_layers"]),
                },
                "raw_hp_vector": list(anchor["raw_hp_vector"]),
                "decoded_hp": dict(anchor_decoded),
                "hp_regime": "hp_high",
                "seed": seed,
                "max_epochs": 300,
                "patience": 80,
                "track_test": False,
                "training_semantics": dict(semantics),
                "worker_id": worker_id,
                "gpu_index": int(mapping["gpu_index"]),
                "gpu_uuid": str(mapping["gpu_uuid"]),
            }
        )
    assignments = [
        {
            "task_id": row["task_id"],
            "pair_id": row["pair_id"],
            "worker_id": row["worker_id"],
            "gpu_index": row["gpu_index"],
            "gpu_uuid": row["gpu_uuid"],
        }
        for row in tasks
    ]
    freeze = {
        "schema_version": "phase-d-task-freeze-v1",
        "planned_task_count": len(tasks),
        "paired_unit_count": pair_index,
        "tasks": tasks,
        "assignments": assignments,
    }
    validation = validate_task_freeze(freeze, config)
    freeze["validation"] = validation
    return freeze


def _without_pair_differences(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in task.items()
        if key not in {"task_id", "arm", "component_variant"}
    }


def validate_task_freeze(
    freeze: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    tasks = freeze.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != METHOD_EQUIVALENT_TASK_COUNT:
        _fail("TASK_FREEZE", "TASK_CARDINALITY_MISMATCH")
    if freeze.get("planned_task_count") != 1155 or freeze.get("paired_unit_count") != 576:
        _fail("TASK_FREEZE", "TASK_LEDGER_HEADER_MISMATCH")
    if len({str(row.get("task_id")) for row in tasks}) != 1155:
        _fail("TASK_FREEZE", "TASK_ID_NOT_UNIQUE")
    if Counter(str(row.get("arm")) for row in tasks) != Counter(
        {"Control": 576, "Intervention": 576, "P_anchor": 3}
    ):
        _fail("TASK_FREEZE", "ARM_TASK_COUNT_MISMATCH")
    search = [row for row in tasks if row.get("arm") in {"Control", "Intervention"}]
    anchors = [row for row in tasks if row.get("arm") == "P_anchor"]
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in search:
        key = (str(row.get("replicate_id")), str(row.get("slot_id")), int(row.get("seed")))
        groups.setdefault(key, []).append(row)
    expected_keys = [
        (replicate, slot, seed)
        for replicate in REPLICATE_IDS
        for slot in SLOT_IDS
        for seed in TRAINING_SEEDS
    ]
    if list(groups) != expected_keys:
        _fail("TASK_FREEZE", "PAIR_ORDER_MISMATCH")
    for pair_index, key in enumerate(expected_keys):
        members = groups[key]
        if len(members) != 2 or {row["arm"] for row in members} != {
            "Control", "Intervention"
        }:
            _fail("TASK_FREEZE", "PAIR_MEMBER_MISMATCH")
        control = next(row for row in members if row["arm"] == "Control")
        intervention = next(row for row in members if row["arm"] == "Intervention")
        if (
            control.get("component_variant") != "baseline"
            or intervention.get("component_variant") != "layer_norm"
            or control.get("pair_index") != pair_index
            or intervention.get("pair_index") != pair_index
            or control.get("worker_id") != pair_index % 4
            or intervention.get("worker_id") != pair_index % 4
            or _without_pair_differences(control) != _without_pair_differences(intervention)
        ):
            _fail("TASK_FREEZE", "PAIR_STRUCTURAL_EQUALITY_MISMATCH")
    mapping = {row["worker_id"]: row for row in EXPECTED_WORKER_MAPPING}
    anchor_decoded = {
        key: EXPECTED_P_ANCHOR[key]
        for key in (
            "lr", "dropout", "hidden_dim", "l2", "gat_heads", "sage_aggr", "gin_eps",
            "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer",
        )
    }
    expected_semantics = _training_semantics(config)
    expected_anchor_tasks: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        worker = mapping[seed]
        expected_anchor_tasks.append(
            {
                "task_id": f"p_anchor|seed{seed}",
                "arm": "P_anchor",
                "component_variant": "baseline",
                "pair_id": None,
                "pair_index": None,
                "replicate_id": None,
                "slot_id": None,
                "config_id": EXPECTED_P_ANCHOR["config_id"],
                "architecture_fingerprint": EXPECTED_P_ANCHOR["architecture_fingerprint"],
                "hp_fingerprint": EXPECTED_P_ANCHOR["hp_fingerprint"],
                "joint_config_fingerprint": EXPECTED_P_ANCHOR["joint_config_fingerprint"],
                "architecture": {
                    "operations": list(EXPECTED_P_ANCHOR["operations"]),
                    "edges": [list(edge) for edge in EXPECTED_P_ANCHOR["edges"]],
                    "effective_layers": EXPECTED_P_ANCHOR["effective_layers"],
                },
                "raw_hp_vector": list(EXPECTED_P_ANCHOR["raw_hp_vector"]),
                "decoded_hp": dict(anchor_decoded),
                "hp_regime": "hp_high",
                "seed": seed,
                "max_epochs": 300,
                "patience": 80,
                "track_test": False,
                "training_semantics": dict(expected_semantics),
                "worker_id": seed,
                "gpu_index": worker["gpu_index"],
                "gpu_uuid": worker["gpu_uuid"],
            }
        )
    if [dict(row) for row in anchors] != expected_anchor_tasks:
        _fail("TASK_FREEZE", "P_ANCHOR_EXACT_TASK_MISMATCH")
    for row in tasks:
        worker = mapping.get(row.get("worker_id"))
        if (
            worker is None
            or row.get("gpu_index") != worker["gpu_index"]
            or row.get("gpu_uuid") != worker["gpu_uuid"]
            or row.get("track_test") is not False
            or row.get("max_epochs") != 300
            or row.get("patience") != 80
            or row.get("hp_regime") != "hp_high"
            or row.get("training_semantics") != expected_semantics
        ):
            _fail("TASK_FREEZE", "TASK_TRAINING_OR_ASSIGNMENT_MISMATCH")
    if Counter(int(row["worker_id"]) for row in tasks) != Counter(
        {0: 289, 1: 289, 2: 289, 3: 288}
    ):
        _fail("TASK_FREEZE", "WORKER_TOTAL_COUNT_MISMATCH")
    if Counter(int(row["gpu_index"]) for row in tasks) != Counter({0: 578, 1: 577}):
        _fail("TASK_FREEZE", "GPU_TOTAL_COUNT_MISMATCH")
    if Counter((int(row["worker_id"]), str(row["arm"])) for row in search) != Counter(
        {
            **{(worker, "Control"): 144 for worker in range(4)},
            **{(worker, "Intervention"): 144 for worker in range(4)},
        }
    ):
        _fail("TASK_FREEZE", "WORKER_ARM_COUNT_MISMATCH")
    if Counter((int(row["gpu_index"]), str(row["arm"])) for row in search) != Counter(
        {
            (0, "Control"): 288,
            (1, "Control"): 288,
            (0, "Intervention"): 288,
            (1, "Intervention"): 288,
        }
    ):
        _fail("TASK_FREEZE", "GPU_ARM_COUNT_MISMATCH")
    assignments = freeze.get("assignments")
    expected_assignments = [
        {
            "task_id": row["task_id"],
            "pair_id": row["pair_id"],
            "worker_id": row["worker_id"],
            "gpu_index": row["gpu_index"],
            "gpu_uuid": row["gpu_uuid"],
        }
        for row in tasks
    ]
    if assignments != expected_assignments:
        _fail("TASK_FREEZE", "ASSIGNMENT_LEDGER_MISMATCH")
    return {
        "passed": True,
        "planned_task_count": 1155,
        "paired_unit_count": 576,
        "worker_task_counts": {str(key): value for key, value in sorted(
            Counter(int(row["worker_id"]) for row in tasks).items()
        )},
        "gpu_task_counts": {str(key): value for key, value in sorted(
            Counter(int(row["gpu_index"]) for row in tasks).items()
        )},
    }


def _is_retryable_exception(exc: BaseException) -> bool:
    return isinstance(exc, (InfrastructureEvaluationError, OSError, TimeoutError))


class RetryBudget:
    """Track the global distinct-task retry boundary for pure/local execution."""

    def __init__(self, limit: int = MAX_RETRIED_TASK_COUNT):
        if int(limit) != MAX_RETRIED_TASK_COUNT:
            _fail("RETRY", "RETRY_LIMIT_CONTRACT_MISMATCH")
        self.limit = int(limit)
        self._retried_task_ids: set[str] = set()

    @property
    def retry_task_count(self) -> int:
        return len(self._retried_task_ids)

    def authorize(self, task_id: str) -> None:
        identity = str(task_id)
        if not identity:
            _fail("RETRY", "TASK_ID_MISSING")
        if identity in self._retried_task_ids:
            return
        if len(self._retried_task_ids) >= self.limit:
            _fail("RETRY", "RETRIED_TASK_58_LIMIT_EXCEEDED")
        self._retried_task_ids.add(identity)


def execute_with_retries(
    task: Mapping[str, Any],
    evaluator: Callable[[dict[str, Any], int], Any],
    budget: RetryBudget,
) -> dict[str, Any]:
    frozen_task = dict(task)
    task_id = str(frozen_task.get("task_id", ""))
    if not task_id:
        _fail("RETRY", "TASK_ID_MISSING")
    for attempt in range(1, 4):
        try:
            value = evaluator(copy.deepcopy(frozen_task), attempt)
            return {"value": value, "attempts": attempt}
        except BaseException as exc:
            if not _is_retryable_exception(exc):
                raise
            if attempt == 3:
                raise ProtocolFailure("RETRY", "THIRD_ATTEMPT_FAILED") from exc
            if attempt == 1:
                budget.authorize(task_id)
    _fail("RETRY", "ATTEMPT_LOOP_CONTRACT_FAILURE")


def run_guarded_evaluation(
    task: Mapping[str, Any],
    data_proxy: ValidationOnlyDataProxy,
    device: Any,
    evaluator: Callable[..., Any] | None = None,
) -> Any:
    """The only formal evaluator boundary; test tracking is a literal false."""

    if task.get("track_test") is not False:
        _fail("TEST_FIREWALL", "FROZEN_TASK_TRACK_TEST_NOT_FALSE")
    if evaluator is None:
        from eval_utils import train_and_eval_arch

        evaluator = train_and_eval_arch
    hp = task["decoded_hp"]
    return evaluator(
        config=task["architecture"],
        data=data_proxy,
        in_ch=int(task["in_ch"]),
        out_ch=int(task["out_ch"]),
        lr=float(hp["lr"]),
        dropout=float(hp["dropout"]),
        hidden_dim=int(hp["hidden_dim"]),
        weight_decay=float(hp["l2"]),
        gat_heads=int(hp.get("gat_heads", 1)),
        sage_aggr=str(hp.get("sage_aggr", "mean")),
        gin_eps=float(hp.get("gin_eps", 0.0)),
        gat_heads_by_layer=hp.get("gat_heads_by_layer"),
        sage_aggr_by_layer=hp.get("sage_aggr_by_layer"),
        gin_eps_by_layer=hp.get("gin_eps_by_layer"),
        device=device,
        max_epochs=int(task["max_epochs"]),
        patience=int(task["patience"]),
        seed=int(task["seed"]),
        track_test=False,
        return_metadata=True,
        component_variant=str(task["component_variant"]),
    )


def assert_evaluator_ast_contract() -> dict[str, Any]:
    wrapper = ast.parse(inspect.getsource(run_guarded_evaluation))
    evaluator_returns = [
        node.value
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "evaluator"
    ]
    evaluator_calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluator"
    ]
    if len(evaluator_returns) != 1 or evaluator_calls != evaluator_returns:
        _fail("TEST_FIREWALL", "FORMAL_EVALUATOR_BOUNDARY_COUNT_MISMATCH")
    call = evaluator_returns[0]
    keywords = {item.arg: item.value for item in call.keywords if item.arg is not None}
    track_test_value = keywords.get("track_test")
    if not (
        isinstance(track_test_value, ast.Constant)
        and track_test_value.value is False
        and len([item for item in call.keywords if item.arg == "track_test"]) == 1
    ):
        _fail("TEST_FIREWALL", "FORMAL_WRAPPER_LITERAL_FALSE_MISMATCH")
    try:
        source = (REPO_ROOT / "eval_utils.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ProtocolFailure("TEST_FIREWALL", "EVALUATOR_SOURCE_PARSE_FAILED") from exc
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "train_and_eval_arch"
        ),
        None,
    )
    if target is None:
        _fail("TEST_FIREWALL", "ROOT_EVALUATOR_FUNCTION_MISSING")
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(target):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    accesses = [
        node
        for node in ast.walk(target)
        if isinstance(node, ast.Attribute) and node.attr == "test_mask"
    ]
    if not accesses:
        _fail("TEST_FIREWALL", "ROOT_TEST_MASK_NEGATIVE_CONTROL_MISSING")
    for access in accesses:
        ancestor = parents.get(access)
        protected = False
        while ancestor is not None and ancestor is not target:
            if (
                isinstance(ancestor, ast.If)
                and isinstance(ancestor.test, ast.Name)
                and ancestor.test.id == "track_test"
            ):
                protected = True
                break
            ancestor = parents.get(ancestor)
        if not protected:
            _fail("TEST_FIREWALL", "ROOT_TEST_MASK_NOT_TRACK_TEST_GUARDED")
    return {
        "wrapper_literal_track_test_false_calls": 1,
        "returned_evaluator_call_count": 1,
        "root_test_mask_access_count": len(accesses),
        "root_test_mask_all_guarded_by_track_test": True,
    }


def _contains_self_attribute(node: ast.AST, attribute: str) -> bool:
    return any(
        isinstance(nested, ast.Attribute)
        and isinstance(nested.value, ast.Name)
        and nested.value.id == "self"
        and nested.attr == attribute
        for nested in ast.walk(node)
    )


def _ancestor_test_strings(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST]
) -> set[str]:
    tests: set[str] = set()
    ancestor = parents.get(node)
    while ancestor is not None:
        if isinstance(ancestor, ast.If):
            tests.add(ast.unparse(ancestor.test))
        ancestor = parents.get(ancestor)
    return tests


def _assignment_to_name(statement: ast.stmt, name: str) -> ast.Assign | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
    ):
        return statement
    return None


def _layernorm_branch_statement(statement: ast.stmt) -> ast.Assign | None:
    current = statement if isinstance(statement, ast.If) else None
    while current is not None:
        if ast.unparse(current.test) == "self.component_variant == 'layer_norm'":
            if len(current.body) != 1:
                return None
            assignment = _assignment_to_name(current.body[0], "layer_out")
            if assignment is None or not isinstance(assignment.value, ast.Call):
                return None
            rendered = ast.unparse(assignment.value)
            if rendered != "self.layer_norms[str(i)](layer_out)":
                return None
            return assignment
        current = (
            current.orelse[0]
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If)
            else None
        )
    return None


def _validate_message_passing_branch(
    body: Sequence[ast.stmt], expected_layer_call: str
) -> bool:
    layer_indices: list[int] = []
    norm_indices: list[int] = []
    dropout_indices: list[int] = []
    for index, statement in enumerate(body):
        assignment = _assignment_to_name(statement, "layer_out")
        if assignment is not None and isinstance(assignment.value, ast.Call):
            if ast.unparse(assignment.value) == expected_layer_call:
                layer_indices.append(index)
        if _layernorm_branch_statement(statement) is not None:
            norm_indices.append(index)
        output = _assignment_to_name(statement, "out")
        if output is None or not isinstance(output.value, ast.Call):
            continue
        call = output.value
        if not (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "F"
            and call.func.attr == "dropout"
            and len(call.args) == 3
            and isinstance(call.args[0], ast.Call)
            and isinstance(call.args[0].func, ast.Name)
            and call.args[0].func.id == "_apply_activation"
            and len(call.args[0].args) == 2
            and isinstance(call.args[0].args[0], ast.Name)
            and call.args[0].args[0].id == "layer_out"
            and ast.unparse(call.args[0].args[1]) == "self.activation_variant"
            and ast.unparse(call.args[1]) == "self.dropout"
            and ast.unparse(call.args[2]) == "self.training"
        ):
            continue
        dropout_indices.append(index)
    return (
        len(layer_indices) == 1
        and len(norm_indices) == 1
        and len(dropout_indices) == 1
        and layer_indices[0] < norm_indices[0] < dropout_indices[0]
    )


def assert_layernorm_source_contract() -> dict[str, Any]:
    try:
        tree = ast.parse((REPO_ROOT / "eval_utils.py").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ProtocolFailure("LAYERNORM_CONTRACT", "SOURCE_PARSE_FAILED") from exc
    dynamic = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DynamicGNN"),
        None,
    )
    if dynamic is None:
        _fail("LAYERNORM_CONTRACT", "DYNAMIC_GNN_CLASS_MISSING")
    initializer = next(
        (node for node in dynamic.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"),
        None,
    )
    forward = next(
        (node for node in dynamic.body if isinstance(node, ast.FunctionDef) and node.name == "forward"),
        None,
    )
    if initializer is None or forward is None:
        _fail("LAYERNORM_CONTRACT", "DYNAMIC_GNN_METHOD_MISSING")
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(initializer):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    constructors = [
        node
        for node in ast.walk(initializer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LayerNorm"
    ]
    if len(constructors) != 1:
        _fail("LAYERNORM_CONTRACT", "LAYERNORM_CONSTRUCTOR_COUNT_MISMATCH")
    constructor_tests = _ancestor_test_strings(constructors[0], parents)
    if not {
        "self.component_variant == 'layer_norm'",
        "op != 'Identity'",
    } <= constructor_tests:
        _fail("LAYERNORM_CONTRACT", "IDENTITY_COULD_RECEIVE_LAYERNORM")
    loops = [node for node in ast.walk(forward) if isinstance(node, ast.For)]
    operation_loop: ast.For | None = None
    identity_branch: ast.If | None = None
    for loop in loops:
        candidate = next(
            (
                statement
                for statement in loop.body
                if isinstance(statement, ast.If)
                and ast.unparse(statement.test) == "op == 'Identity' or layer is None"
            ),
            None,
        )
        if candidate is not None:
            operation_loop = loop
            identity_branch = candidate
            break
    if operation_loop is None or identity_branch is None:
        _fail("LAYERNORM_CONTRACT", "IDENTITY_BRANCH_MISSING")
    if (
        [ast.unparse(statement) for statement in identity_branch.body] != ["out = agg"]
        or any(_contains_self_attribute(statement, "layer_norms") for statement in identity_branch.body)
        or len(identity_branch.orelse) != 1
        or not isinstance(identity_branch.orelse[0], ast.If)
    ):
        _fail("LAYERNORM_CONTRACT", "IDENTITY_BRANCH_NOT_BYPASS_ONLY")
    gcnii_branch = identity_branch.orelse[0]
    if ast.unparse(gcnii_branch.test) != "op == 'GCNII'":
        _fail("LAYERNORM_CONTRACT", "GCNII_BRANCH_MISSING")
    gcnii_ordered = _validate_message_passing_branch(
        gcnii_branch.body, "layer(agg, x_0, edge_index)"
    )
    generic_ordered = _validate_message_passing_branch(
        gcnii_branch.orelse, "layer(agg, edge_index)"
    )
    norm_calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call) and _contains_self_attribute(node.func, "layer_norms")
    ]
    if len(norm_calls) != 2 or not gcnii_ordered or not generic_ordered:
        _fail("LAYERNORM_CONTRACT", "FORWARD_PLACEMENT_MISMATCH")
    return {
        "passed": True,
        "identity_layernorm_count": 0,
        "message_passing_branch_count": len(norm_calls),
        "ordered_layernorm_branch_count": 2,
        "order": ["operation", "layer_norm", "activation", "dropout"],
    }


def synthetic_firewall_evidence() -> dict[str, Any]:
    class SyntheticData:
        x = object()
        edge_index = object()
        train_mask = object()
        val_mask = object()
        y = object()
        test_mask = object()

    proxy = ValidationOnlyDataProxy(SyntheticData())
    for name in sorted(DATA_ALLOWLIST):
        getattr(proxy, name)
    direct_denied = False
    try:
        _ = proxy.test_mask
    except TestAccessViolation:
        direct_denied = True
    probe_denied = False
    try:
        probe_track_test(proxy, track_test=True)
    except TestAccessViolation:
        probe_denied = True
    ledger = _proxy_internal_ledger(proxy)
    if not direct_denied or not probe_denied or ledger["test_accessed"] is not True:
        _fail("TEST_FIREWALL", "SYNTHETIC_NEGATIVE_CONTROL_FAILED")
    return {
        "schema_version": "phase-d-test-firewall-v1",
        "data_allowlist": sorted(DATA_ALLOWLIST),
        "direct_test_mask_negative_control": "PASS",
        "track_test_true_negative_control": "PASS",
        "evaluator_ast": assert_evaluator_ast_contract(),
        "layernorm_source_contract": assert_layernorm_source_contract(),
        "negative_control_ledger": ledger,
        "formal_track_test": False,
        "formal_test_accessed": False,
        "synthetic_only": True,
    }


def _as_twelve_deltas(deltas: Sequence[float]) -> Any:
    import numpy as np

    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (12,) or not bool(np.isfinite(values).all()):
        _fail("INFERENCE", "REPLICATE_DELTA_VECTOR_INVALID")
    return values


def exact_sign_flip(deltas: Sequence[float]) -> dict[str, Any]:
    values = _as_twelve_deltas(deltas)
    observed = float(values.mean())
    exceedance = 0
    permutation_count = 0
    for sign_tuple in itertools.product((-1.0, 1.0), repeat=12):
        permutation_count += 1
        statistic = float((values * sign_tuple).mean())
        if statistic >= observed:
            exceedance += 1
    if permutation_count != 4096:
        _fail("INFERENCE", "SIGN_FLIP_ENUMERATION_MISMATCH")
    return {
        "observed_mean": observed,
        "exceedance_count": exceedance,
        "permutation_count": permutation_count,
        "one_sided_p_value": exceedance / 4096.0,
        "monte_carlo_plus_one_correction": False,
    }


def bootstrap_lower_bound(
    deltas: Sequence[float], *, seed: int, resamples: int = 100000
) -> float:
    import numpy as np

    values = _as_twelve_deltas(deltas)
    if int(resamples) != 100000 or not (0 <= int(seed) < 2**64):
        _fail("INFERENCE", "BOOTSTRAP_CONTRACT_MISMATCH")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, 12, size=(100000, 12))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.05, method="linear"))


def _atomic_replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(json_asset(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise ProtocolFailure("WRITE", "ATOMIC_JSON_WRITE_FAILED") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.staging.", dir=path.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise ProtocolFailure("WRITE", "TARGET_ALREADY_EXISTS") from exc
    except OSError as exc:
        raise ProtocolFailure("WRITE", "EXCLUSIVE_WRITE_FAILED") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _safe_relative(relative: str) -> bool:
    pure = PurePosixPath(relative)
    return bool(relative) and not pure.is_absolute() and ".." not in pure.parts


def _rename_directory_noreplace(source: Path, target: Path, stage: str) -> None:
    if not sys.platform.startswith("linux"):
        _fail(stage, "ATOMIC_NOREPLACE_RENAME_UNSUPPORTED")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _fail(stage, "ATOMIC_NOREPLACE_RENAME_UNSUPPORTED")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ProtocolFailure(stage, "PUBLICATION_TARGET_ALREADY_EXISTS")
        raise ProtocolFailure(stage, "ATOMIC_NOREPLACE_RENAME_FAILED")
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _manifest_for_assets(assets: Mapping[str, bytes]) -> dict[str, Any]:
    records = [
        {"path": relative, "size_bytes": len(assets[relative]), "sha256": sha256_bytes(assets[relative])}
        for relative in sorted(assets)
    ]
    payload = {
        "schema_version": "phase-d-prelaunch-manifest-v1",
        "file_count": len(records),
        "files": records,
        "manifest_self_excluded": True,
        "root_encoding": "canonical-json-manifest-payload",
    }
    return {**payload, "manifest_root_sha256": sha256_bytes(canonical_json_bytes(payload))}


def _verify_manifest_payload(manifest: Mapping[str, Any], schema: str) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != schema:
        _fail("MANIFEST", "MANIFEST_SCHEMA_MISMATCH")
    root = manifest.get("manifest_root_sha256")
    payload = dict(manifest)
    payload.pop("manifest_root_sha256", None)
    if root != sha256_bytes(canonical_json_bytes(payload)):
        _fail("MANIFEST", "MANIFEST_ROOT_MISMATCH")
    rows = manifest.get("files")
    if not isinstance(rows, list) or manifest.get("file_count") != len(rows):
        _fail("MANIFEST", "MANIFEST_FILE_COUNT_MISMATCH")
    paths: list[str] = []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            _fail("MANIFEST", "MANIFEST_RECORD_INVALID")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or not _safe_relative(relative)
            or type(row.get("size_bytes")) is not int
            or not isinstance(row.get("sha256"), str)
            or not HEX64_RE.fullmatch(row["sha256"])
        ):
            _fail("MANIFEST", "MANIFEST_RECORD_INVALID")
        paths.append(relative)
        normalized.append(dict(row))
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        _fail("MANIFEST", "MANIFEST_PATH_ORDER_OR_UNIQUENESS_MISMATCH")
    return normalized


def _verify_live_phase_d_snapshot_assets(assets: Mapping[str, bytes]) -> None:
    snapshot_keys = {f"source_snapshot/{relative}" for relative in TARGET_RELS}
    present = snapshot_keys & set(assets)
    if present and present != snapshot_keys:
        _fail("SOURCE_BINDING", "PHASE_D_SOURCE_SNAPSHOT_SET_INCOMPLETE")
    if not present:
        return
    for relative in TARGET_RELS:
        path = REPO_ROOT / relative
        try:
            mode = path.lstat().st_mode
            live_bytes = path.read_bytes()
        except OSError as exc:
            raise ProtocolFailure("SOURCE_BINDING", "PHASE_D_SOURCE_RECHECK_FAILED") from exc
        if (
            not stat.S_ISREG(mode)
            or stat.S_ISLNK(mode)
            or live_bytes != assets[f"source_snapshot/{relative}"]
        ):
            _fail("SOURCE_BINDING", "PHASE_D_SOURCE_CHANGED_BEFORE_PUBLICATION")


def write_prelaunch(result_root: Path, assets: Mapping[str, bytes]) -> dict[str, Any]:
    if result_root.exists() or result_root.is_symlink():
        _fail("PRELAUNCH", "RESULT_ROOT_ALREADY_EXISTS")
    if not result_root.parent.is_dir():
        _fail("PRELAUNCH", "RESULT_ROOT_PARENT_MISSING")
    if "manifest.json" in assets:
        _fail("PRELAUNCH", "MANIFEST_MUST_BE_DERIVED")
    for relative, data in assets.items():
        if not isinstance(relative, str) or not _safe_relative(relative) or not isinstance(data, bytes):
            _fail("PRELAUNCH", "ASSET_PATH_OR_TYPE_INVALID")
        if relative.startswith(("launch/", "formal/")):
            _fail("PRELAUNCH", "PREMATURE_MUTABLE_NAMESPACE")
    manifest = _manifest_for_assets(assets)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{result_root.name}.prelaunch.", dir=result_root.parent)
    )
    published = False
    try:
        for relative in sorted(assets):
            _write_exclusive(staging / relative, assets[relative])
        _write_exclusive(staging / "manifest.json", json_asset(manifest))
        verify_prelaunch(staging)
        _verify_live_phase_d_snapshot_assets(assets)
        if result_root.exists() or result_root.is_symlink():
            _fail("PRELAUNCH", "RESULT_ROOT_APPEARED_DURING_STAGING")
        _rename_directory_noreplace(staging, result_root, "PRELAUNCH")
        published = True
        verify_prelaunch(result_root)
    except ProtocolFailure:
        raise
    except OSError as exc:
        raise ProtocolFailure("PRELAUNCH", "ATOMIC_DIRECTORY_PUBLICATION_FAILED") from exc
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return manifest


def verify_prelaunch(result_root: Path) -> dict[str, Any]:
    if not result_root.is_dir() or result_root.is_symlink():
        _fail("PRELAUNCH", "RESULT_ROOT_INVALID")
    manifest = _load_json_file(result_root / "manifest.json", "PRELAUNCH", "MANIFEST_LOAD_FAILED")
    if not isinstance(manifest, dict):
        _fail("PRELAUNCH", "MANIFEST_ROOT_NOT_OBJECT")
    rows = _verify_manifest_payload(manifest, "phase-d-prelaunch-manifest-v1")
    expected_prelaunch = {str(row["path"]) for row in rows} | {"manifest.json"}
    observed = _walk_regular_files(result_root)
    unexpected = observed - expected_prelaunch - KNOWN_POST_PRELAUNCH_PATHS
    if unexpected:
        _fail("PRELAUNCH", "UNAUTHORIZED_POST_PRELAUNCH_FILE")
    if "launch/runner_entered.json" in observed and "launch/launch_request.json" not in observed:
        _fail("PRELAUNCH", "RUNNER_ENTER_WITHOUT_LAUNCH_REQUEST")
    if any(relative.startswith("formal/") for relative in observed) and not {
        "launch/launch_request.json", "launch/runner_entered.json"
    } <= observed:
        _fail("PRELAUNCH", "FORMAL_NAMESPACE_WITHOUT_LAUNCH_BINDING")
    if observed & SEALED_RUNTIME_PATHS:
        if "formal/manifest.json" not in observed:
            _fail("PRELAUNCH", "SEALED_ARTIFACT_WITHOUT_FORMAL_MANIFEST")
        if "completion.json" in observed and not {
            ARCHIVE_BASENAME, ARCHIVE_SIDECAR_BASENAME
        } <= observed:
            _fail("PRELAUNCH", "COMPLETION_WITHOUT_ARCHIVE_AND_SIDECAR")
    for row in rows:
        path = result_root / row["path"]
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise ProtocolFailure("PRELAUNCH", "MANIFEST_FILE_MISSING") from exc
        if (
            not stat.S_ISREG(mode)
            or stat.S_ISLNK(mode)
            or path.stat().st_size != row["size_bytes"]
            or sha256_path(path) != row["sha256"]
        ):
            _fail("PRELAUNCH", "MANIFEST_FILE_IDENTITY_MISMATCH")
    return manifest


def _create_formal_with_atomic_start(
    formal_dir: Path, start_record: Mapping[str, Any]
) -> None:
    if formal_dir.exists() or formal_dir.is_symlink():
        _fail("FORMAL_START", "FORMAL_DIRECTORY_ALREADY_EXISTS")
    formal_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="phase_d_formal_start_", dir=formal_dir.parent))
    published = False
    try:
        _write_exclusive(staging / "start.json", json_asset(start_record))
        if formal_dir.exists() or formal_dir.is_symlink():
            _fail("FORMAL_START", "FORMAL_DIRECTORY_APPEARED_DURING_STAGING")
        _rename_directory_noreplace(staging, formal_dir, "FORMAL_START")
        published = True
    except ProtocolFailure:
        raise
    except OSError as exc:
        raise ProtocolFailure("FORMAL_START", "ATOMIC_FORMAL_START_FAILED") from exc
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _run_readonly(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def audit_gpu_admission(
    config: Mapping[str, Any], *, require_zero_compute_processes: bool
) -> dict[str, Any]:
    inventory = _run_readonly(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = _run_readonly(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if inventory.returncode != 0 or processes.returncode != 0:
        _fail("GPU_ADMISSION", "NVIDIA_INVENTORY_UNAVAILABLE")
    rows: list[dict[str, Any]] = []
    for line in inventory.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 7:
            _fail("GPU_ADMISSION", "GPU_INVENTORY_FORMAT_MISMATCH")
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "memory_total_mib": int(fields[3]),
                    "memory_used_mib": int(fields[4]),
                    "memory_free_mib": int(fields[5]),
                    "utilization_percent": int(fields[6]),
                }
            )
        except ValueError as exc:
            raise ProtocolFailure("GPU_ADMISSION", "GPU_INVENTORY_FORMAT_MISMATCH") from exc
    expected = [
        {"index": int(row["index"]), "uuid": str(row["uuid"])}
        for row in config["workers"]["gpus"]
    ]
    observed = [{"index": row["index"], "uuid": row["uuid"]} for row in rows]
    if observed != expected:
        _fail("GPU_ADMISSION", "GPU_INDEX_UUID_MAPPING_MISMATCH")
    process_lines = [line for line in processes.stdout.splitlines() if line.strip()]
    if require_zero_compute_processes and process_lines:
        _fail("GPU_ADMISSION", "GPU_COMPUTE_PROCESS_OCCUPIED")
    return {
        "gpus": rows,
        "compute_process_count": len(process_lines),
        "zero_compute_processes_required": require_zero_compute_processes,
    }


def audit_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    hostname = _run_readonly(["hostname"])
    branch = _run_readonly(["git", "branch", "--show-current"])
    head = _run_readonly(["git", "rev-parse", "HEAD"])
    status = _run_readonly(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    if any(result.returncode != 0 for result in (hostname, branch, head, status)):
        _fail("ENVIRONMENT", "ENVIRONMENT_COMMAND_FAILED")
    observed_host = hostname.stdout.strip().split(".", 1)[0]
    required_host = str(config["required_host"])
    if observed_host != required_host:
        _fail("ENVIRONMENT", "HOSTNAME_MISMATCH")
    gpu = audit_gpu_admission(config, require_zero_compute_processes=True)
    return {
        "schema_version": "phase-d-environment-v1",
        "hostname": hostname.stdout.strip(),
        "branch": branch.stdout.strip(),
        "head": head.stdout.strip(),
        "dirty_status_porcelain_z_sha256": sha256_bytes(status.stdout.encode("utf-8")),
        "dirty_status_entry_count": len([value for value in status.stdout.split("\0") if value]),
        "gpu_admission": gpu,
        "cuda_initialized_by_dry_run": False,
        "gnn_evaluations": 0,
        "test_accessed": False,
    }


def audit_dependencies(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("dependency_hashes") != PINNED_DEPENDENCIES:
        _fail("DEPENDENCIES", "DEPENDENCY_CONTRACT_MISMATCH")
    records = []
    for relative, expected in PINNED_DEPENDENCIES.items():
        identity = assert_file_identity(REPO_ROOT / relative, expected)
        records.append(
            {
                "path": relative,
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
    return {"records": records, "all_match": True}


def audit_phase_c_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    phase_c = config.get("accepted_upstream", {}).get("phase_c")
    if phase_c != EXPECTED_PHASE_C:
        _fail("PHASE_C_IDENTITY", "DECLARED_UPSTREAM_MISMATCH")
    root = REPO_ROOT / phase_c["result_root"]
    archive = root / "formal/fixed-layernorm-enhancement-global4-v2.tar.gz"
    archive_identity = assert_file_identity(archive, phase_c["archive_sha256"])
    manifest = root / "formal/manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        _fail("PHASE_C_IDENTITY", "FORMAL_MANIFEST_MISSING")
    data = manifest.read_bytes()
    matches = re.findall(
        rb'"manifest_root_sha256"\s*:\s*"([0-9a-f]{64})"', data
    )
    if len(matches) != 1 or matches[0].decode("ascii") != phase_c["formal_manifest_root_sha256"]:
        _fail("PHASE_C_IDENTITY", "FORMAL_MANIFEST_ROOT_MISMATCH")
    return {
        "declared_upstream": dict(phase_c),
        "formal_manifest": {
            "path": str(manifest.relative_to(REPO_ROOT)),
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
            "manifest_root_sha256": phase_c["formal_manifest_root_sha256"],
            "identity_fields_only_projected": True,
        },
        "archive": {
            "path": str(archive.relative_to(REPO_ROOT)),
            "size_bytes": archive_identity["size_bytes"],
            "sha256": archive_identity["sha256"],
        },
        "outcome_values_reread": False,
    }


def verify_live_source_snapshot_binding(
    result_root: Path, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    manifest_value = verify_prelaunch(result_root) if manifest is None else manifest
    records = {row["path"]: row for row in manifest_value["files"]}
    evidence: dict[str, Any] = {}
    for relative in TARGET_RELS:
        snapshot_relative = f"source_snapshot/{relative}"
        record = records.get(snapshot_relative)
        live = REPO_ROOT / relative
        snapshot = result_root / snapshot_relative
        if record is None or not live.is_file() or live.is_symlink() or not snapshot.is_file() or snapshot.is_symlink():
            _fail("SOURCE_BINDING", "SOURCE_SNAPSHOT_MISSING")
        live_bytes = live.read_bytes()
        snapshot_bytes = snapshot.read_bytes()
        if (
            live_bytes != snapshot_bytes
            or len(live_bytes) != record["size_bytes"]
            or sha256_bytes(live_bytes) != record["sha256"]
        ):
            _fail("SOURCE_BINDING", "LIVE_SOURCE_SNAPSHOT_MISMATCH")
        evidence[relative] = {
            "size_bytes": len(live_bytes),
            "sha256": record["sha256"],
            "live_snapshot_bytes_equal": True,
        }
    return evidence


def _build_frozen_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase-d-frozen-protocol-v1",
        "experiment": EXPERIMENT,
        "governing_documents": config["governing_documents"],
        "accepted_upstream": config["accepted_upstream"],
        "historical_exposure": config["historical_exposure"],
        "p_anchor": config["p_anchor"],
        "search_design": config["search_design"],
        "generation": config["generation"],
        "training_protocol": config["training_protocol"],
        "workers": config["workers"],
        "pair_equality": config["pair_equality"],
        "evaluation": config["evaluation"],
        "retry": config["retry"],
        "lifecycle": config["lifecycle"],
        "prelaunch": config["prelaunch"],
        "formal_artifacts": config["formal_artifacts"],
        "inference": config["inference"],
        "challengers": config["challengers"],
        "runtime_estimate_minutes": config["runtime_estimate_minutes"],
        "terminal_contract": config["terminal_contract"],
        "phase_e_started": False,
    }


def _build_prelaunch_assets(
    config: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    phase_c: Mapping[str, Any],
    exposure: Mapping[str, Any],
    anchor: Mapping[str, Any],
    candidates: Mapping[str, Any],
    task_freeze: Mapping[str, Any],
    environment: Mapping[str, Any],
    firewall: Mapping[str, Any],
) -> dict[str, bytes]:
    protocol_bytes = json_asset(_build_frozen_protocol(config))
    protocol_digest = hashlib.sha256(protocol_bytes).digest()
    protocol_sha = protocol_digest.hex()
    bootstrap_seed = int.from_bytes(protocol_digest[:8], "big", signed=False)
    inference_plan = {
        "schema_version": "phase-d-inference-plan-v1",
        "frozen_protocol_sha256": protocol_sha,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_seed_derivation": (
            "unsigned big-endian integer from first 8 bytes of SHA256 of exact "
            "frozen_protocol.json bytes"
        ),
        "bootstrap_replicates": 100000,
        "bootstrap_quantile": 0.05,
        "bootstrap_quantile_method": "linear",
        "sign_flip_permutations": 4096,
        "sign_flip_exhaustive": True,
        "sign_flip_plus_one_correction": False,
    }
    target_bytes: dict[str, bytes] = {}
    for relative in TARGET_RELS:
        path = REPO_ROOT / relative
        try:
            before = path.stat(follow_symlinks=False)
            data = path.read_bytes()
            after = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise ProtocolFailure("SOURCE_BINDING", "PHASE_D_SOURCE_READ_FAILED") from exc
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if (
            path.is_symlink()
            or not stat.S_ISREG(after.st_mode)
            or before_identity != after_identity
            or len(data) != after.st_size
        ):
            _fail("SOURCE_BINDING", "PHASE_D_SOURCE_CHANGED_DURING_READ")
        target_bytes[relative] = data
    target_identities = {
        relative: {
            "size_bytes": len(target_bytes[relative]),
            "sha256": sha256_bytes(target_bytes[relative]),
        }
        for relative in TARGET_RELS
    }
    input_hashes = {
        "schema_version": "phase-d-input-hashes-v1",
        "governing_document_declarations": PINNED_GOVERNING_DOCUMENTS,
        "governing_document_bytes_available_to_outer_controller_only": True,
        "dependencies": dependencies,
        "phase_c_identity": phase_c,
        "exposure_sources": exposure["sources"],
        "exposure_commitment_sha256": exposure["commitment_sha256"],
        "p_anchor_source": {
            **P_ANCHOR_SOURCE,
            "size_bytes": (REPO_ROOT / P_ANCHOR_SOURCE["path"]).stat().st_size,
        },
        "p_anchor_identity": {
            key: anchor[key]
            for key in (
                "architecture_fingerprint", "hp_fingerprint", "joint_config_fingerprint", "config_id"
            )
        },
        "phase_d_sources": target_identities,
    }
    frozen_exposure = dict(exposure)
    frozen_candidates = dict(candidates)
    assignments = {
        "schema_version": "phase-d-assignment-freeze-v1",
        "task_count": len(task_freeze["assignments"]),
        "assignments": task_freeze["assignments"],
        "worker_mapping": config["workers"]["mapping"],
        "worker_task_counts": task_freeze["validation"]["worker_task_counts"],
        "gpu_task_counts": task_freeze["validation"]["gpu_task_counts"],
    }
    frozen_tasks = {
        key: value for key, value in task_freeze.items() if key != "assignments"
    }
    assets: dict[str, bytes] = {
        "frozen_protocol.json": protocol_bytes,
        "inference_plan.json": json_asset(inference_plan),
        "input_hashes.json": json_asset(input_hashes),
        "frozen_exposure.json": json_asset(frozen_exposure),
        "frozen_candidates.json": json_asset(frozen_candidates),
        "frozen_tasks.json": json_asset(frozen_tasks),
        "assignments.json": json_asset(assignments),
        "test_firewall.json": json_asset(firewall),
        "pre_execution_estimate.json": json_asset(
            {
                "schema_version": "phase-d-pre-execution-estimate-v1",
                "best_minutes": 165,
                "expected_minutes": 210,
                "worst_reference_minutes": 360,
                "worker_count": 4,
                "planned_method_equivalent_tasks": 1155,
            }
        ),
        "environment.json": json_asset(environment),
        "state.json": json_asset(
            {
                "state": "DRY_RUN_COMPLETE_ZERO_GNN",
                "planned_task_count": 1155,
                "frozen_candidate_count": 192,
                "gnn_evaluations": 0,
                "physical_gnn_attempt_count": 0,
                "test_accessed": False,
                "phase_e_started": False,
                "formal_started": False,
            }
        ),
    }
    for relative in TARGET_RELS:
        assets[f"source_snapshot/{relative}"] = target_bytes[relative]
    required = set(config["prelaunch"]["required_files"])
    if set(assets) | {"manifest.json"} != required:
        _fail("PRELAUNCH", "REQUIRED_PRELAUNCH_FILE_SET_MISMATCH")
    return assets


def dry_run(config_path: Path) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    config = load_config(config_path)
    result_root = REPO_ROOT / RESULT_REL
    if result_root.exists() or result_root.is_symlink():
        _fail("DRY_RUN", "RESULT_ROOT_ALREADY_EXISTS")
    dependencies = audit_dependencies(config)
    phase_c = audit_phase_c_identity(config)
    exposure = load_exposure_basis(config)
    anchor = project_p_anchor(config)
    environment = audit_environment(config)
    firewall = synthetic_firewall_evidence()
    candidates = generate_all_candidates(config, exposure)
    task_freeze = build_task_freeze(candidates["candidates"], anchor, config)
    assets = _build_prelaunch_assets(
        config,
        dependencies,
        phase_c,
        exposure,
        anchor,
        candidates,
        task_freeze,
        environment,
        firewall,
    )
    manifest = write_prelaunch(result_root, assets)
    return {
        "status": "PASS",
        "stage": "DRY_RUN_COMPLETE_ZERO_GNN",
        "candidate_count": 192,
        "planned_task_count": 1155,
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "gnn_evaluations": 0,
        "physical_gnn_attempt_count": 0,
        "cuda_initialized": False,
        "test_accessed": False,
        "phase_e_started": False,
    }


def _load_bound_record(path: Path, stage: str, code: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        _fail(stage, code)
    value = _load_json_file(path, stage, code)
    if not isinstance(value, dict):
        _fail(stage, code)
    return value


def _snapshot_record_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        relative: next(
            row for row in manifest["files"] if row["path"] == f"source_snapshot/{relative}"
        )
        for relative in TARGET_RELS
    }


def record_launch_request(config_path: Path) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    result_root = REPO_ROOT / RESULT_REL
    manifest = verify_prelaunch(result_root)
    source_binding = verify_live_source_snapshot_binding(result_root, manifest)
    request = {
        "schema_version": "phase-d-launch-request-v1",
        "state": "FORMAL_FROZEN",
        "requested_at": utc_now(),
        "run_id": f"phase-d-{uuid.uuid4()}",
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "source_snapshots": _snapshot_record_map(manifest),
        "live_source_binding": source_binding,
        "frozen_protocol_sha256": sha256_path(result_root / "frozen_protocol.json"),
        "frozen_task_ledger_sha256": sha256_path(result_root / "frozen_tasks.json"),
        "frozen_assignment_ledger_sha256": sha256_path(result_root / "assignments.json"),
        "planned_task_count": 1155,
        "gnn_evaluations": 0,
        "test_accessed": False,
        "phase_e_started": False,
    }
    _write_exclusive(result_root / "launch/launch_request.json", json_asset(request))
    return {
        "status": "PASS",
        "stage": "LAUNCH_REQUEST_RECORDED_ZERO_GNN",
        "run_id": request["run_id"],
        "launch_request_sha256": sha256_path(result_root / "launch/launch_request.json"),
        "gnn_evaluations": 0,
        "test_accessed": False,
    }


def _verify_launch_request(
    result_root: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    path = result_root / "launch/launch_request.json"
    request = _load_bound_record(path, "LAUNCH", "LAUNCH_REQUEST_INVALID")
    expected_snapshots = _snapshot_record_map(manifest)
    if (
        request.get("schema_version") != "phase-d-launch-request-v1"
        or request.get("state") != "FORMAL_FROZEN"
        or request.get("prelaunch_manifest_root_sha256") != manifest["manifest_root_sha256"]
        or request.get("source_snapshots") != expected_snapshots
        or request.get("frozen_protocol_sha256") != sha256_path(result_root / "frozen_protocol.json")
        or request.get("frozen_task_ledger_sha256") != sha256_path(result_root / "frozen_tasks.json")
        or request.get("frozen_assignment_ledger_sha256") != sha256_path(result_root / "assignments.json")
        or request.get("planned_task_count") != 1155
        or request.get("gnn_evaluations") != 0
        or request.get("test_accessed") is not False
        or request.get("phase_e_started") is not False
        or not isinstance(request.get("run_id"), str)
        or not request["run_id"].startswith("phase-d-")
    ):
        _fail("LAUNCH", "LAUNCH_REQUEST_BINDING_MISMATCH")
    return request, sha256_path(path)


def record_runner_enter(config_path: Path) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    result_root = REPO_ROOT / RESULT_REL
    manifest = verify_prelaunch(result_root)
    verify_live_source_snapshot_binding(result_root, manifest)
    request, request_sha = _verify_launch_request(result_root, manifest)
    if (result_root / "formal").exists() or (result_root / "formal").is_symlink():
        _fail("RUNNER_ENTER", "FORMAL_NAMESPACE_ALREADY_EXISTS")
    record = {
        "schema_version": "phase-d-runner-enter-v1",
        "state": "RUNNER_ENTERED",
        "entered_at": utc_now(),
        "run_id": request["run_id"],
        "launch_request_sha256": request_sha,
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "gnn_evaluations": 0,
        "test_accessed": False,
    }
    _write_exclusive(result_root / "launch/runner_entered.json", json_asset(record))
    return {
        "status": "PASS",
        "stage": "RUNNER_ENTERED_ZERO_GNN",
        "run_id": request["run_id"],
        "gnn_evaluations": 0,
        "test_accessed": False,
    }


def _verify_runner_enter(
    result_root: Path, manifest: Mapping[str, Any], request: Mapping[str, Any], request_sha: str
) -> dict[str, Any]:
    record = _load_bound_record(
        result_root / "launch/runner_entered.json", "RUNNER_ENTER", "RUNNER_ENTER_RECORD_INVALID"
    )
    if (
        record.get("schema_version") != "phase-d-runner-enter-v1"
        or record.get("state") != "RUNNER_ENTERED"
        or record.get("run_id") != request["run_id"]
        or record.get("launch_request_sha256") != request_sha
        or record.get("prelaunch_manifest_root_sha256") != manifest["manifest_root_sha256"]
        or record.get("gnn_evaluations") != 0
        or record.get("test_accessed") is not False
    ):
        _fail("RUNNER_ENTER", "RUNNER_ENTER_BINDING_MISMATCH")
    return record


RESULT_TASK_BINDING_FIELDS = (
    "arm",
    "replicate_id",
    "slot_id",
    "architecture_fingerprint",
    "hp_fingerprint",
    "joint_config_fingerprint",
    "seed",
    "component_variant",
    "worker_id",
    "gpu_index",
    "gpu_uuid",
)

ATTEMPT_TASK_BINDING_FIELDS = (
    "arm",
    "replicate_id",
    "slot_id",
    "seed",
    "component_variant",
    "worker_id",
    "gpu_index",
    "gpu_uuid",
)


def _task_map(tasks: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Mapping[str, Any]]:
    try:
        task_by_id = {str(row["task_id"]): row for row in tasks}
    except (KeyError, TypeError) as exc:
        raise ProtocolFailure(stage, "FROZEN_TASK_ID_INVALID") from exc
    if len(task_by_id) != len(tasks) or any(not task_id for task_id in task_by_id):
        _fail(stage, "FROZEN_TASK_ID_NOT_UNIQUE")
    return task_by_id


def _validate_result_rows(
    tasks: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]], stage: str
) -> dict[str, Mapping[str, Any]]:
    task_by_id = _task_map(tasks, stage)
    result_by_id: dict[str, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping):
            _fail(stage, "RESULT_RECORD_INVALID")
        task_id = str(result.get("task_id", ""))
        task = task_by_id.get(task_id)
        if task is None or task_id in result_by_id:
            _fail(stage, "RESULT_TASK_ID_MISMATCH")
        if any(result.get(field) != task.get(field) for field in RESULT_TASK_BINDING_FIELDS):
            _fail(stage, "RESULT_FROZEN_TASK_BINDING_MISMATCH")
        try:
            accuracy = float(result["validation_accuracy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolFailure(stage, "RESULT_VALIDATION_ACCURACY_INVALID") from exc
        attempt_count = result.get("attempt_count")
        if (
            not math.isfinite(accuracy)
            or not 0.0 <= accuracy <= 1.0
            or type(attempt_count) is not int
            or not 1 <= attempt_count <= 3
            or result.get("test_accessed") is not False
        ):
            _fail(stage, "RESULT_VALUE_ATTEMPT_OR_FIREWALL_MISMATCH")
        result_by_id[task_id] = result
    return result_by_id


def _validate_attempt_and_result_records(
    tasks: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage = "FORMAL_LEDGER"
    task_by_id = _task_map(tasks, stage)
    result_by_id = _validate_result_rows(tasks, results, stage)
    attempts_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    allowed_statuses = {
        "IN_PROGRESS",
        "SUCCESS",
        "TRANSIENT_FAILURE",
        "NONRETRYABLE_FAILURE",
        "EVALUATOR_INVALID",
        "TEST_FIREWALL_VIOLATION",
        "RETRY_TASK_58_LIMIT_EXCEEDED",
    }
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            _fail(stage, "ATTEMPT_RECORD_INVALID")
        task_id = str(attempt.get("task_id", ""))
        task = task_by_id.get(task_id)
        attempt_number = attempt.get("attempt")
        if task is None or type(attempt_number) is not int or not 1 <= attempt_number <= 3:
            _fail(stage, "ATTEMPT_TASK_OR_NUMBER_INVALID")
        if any(attempt.get(field) != task.get(field) for field in ATTEMPT_TASK_BINDING_FIELDS):
            _fail(stage, "ATTEMPT_FROZEN_TASK_BINDING_MISMATCH")
        if attempt.get("status") not in allowed_statuses or not isinstance(
            attempt.get("test_access"), Mapping
        ):
            _fail(stage, "ATTEMPT_STATUS_OR_TEST_LEDGER_INVALID")
        attempts_by_task[task_id].append(attempt)
    for task_id, rows in attempts_by_task.items():
        rows.sort(key=lambda row: int(row["attempt"]))
        numbers = [int(row["attempt"]) for row in rows]
        if numbers != list(range(1, len(rows) + 1)):
            _fail(stage, "ATTEMPT_SEQUENCE_NOT_CONTIGUOUS")
        statuses = [str(row["status"]) for row in rows]
        if any(status != "TRANSIENT_FAILURE" for status in statuses[:-1]):
            _fail(stage, "ATTEMPT_RETRY_PREDECESSOR_NOT_TRANSIENT")
        result = result_by_id.get(task_id)
        if result is None:
            if "SUCCESS" in statuses:
                _fail(stage, "SUCCESS_ATTEMPT_WITHOUT_RESULT")
        elif statuses[-1] != "SUCCESS" or int(result["attempt_count"]) != numbers[-1]:
            _fail(stage, "RESULT_TERMINAL_ATTEMPT_MISMATCH")
    if set(result_by_id) - set(attempts_by_task):
        _fail(stage, "RESULT_WITHOUT_ATTEMPT")
    retried_ids = {
        task_id
        for task_id, rows in attempts_by_task.items()
        if max(int(row["attempt"]) for row in rows) > 1
    }
    return {
        "task_by_id": task_by_id,
        "result_by_id": result_by_id,
        "attempts_by_task": attempts_by_task,
        "retry_task_ids": retried_ids,
        "physical_gnn_attempt_count": len(attempts),
    }


def compute_statistics(
    results: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    inference_plan: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    task_by_id = _task_map(tasks, "STATISTICS")
    if len(task_by_id) != 1155 or len(results) != 1155:
        _fail("STATISTICS", "RESULT_CARDINALITY_MISMATCH")
    result_by_id = _validate_result_rows(tasks, results, "STATISTICS")
    if len(result_by_id) != 1155 or set(result_by_id) != set(task_by_id):
        _fail("STATISTICS", "RESULT_TASK_ID_MISMATCH")
    grouped: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    anchor_values: list[tuple[int, float]] = []
    for task_id, result in result_by_id.items():
        task = task_by_id[task_id]
        value = float(result["validation_accuracy"])
        if not math.isfinite(value):
            _fail("STATISTICS", "NONFINITE_VALIDATION_RESULT")
        if task["arm"] == "P_anchor":
            anchor_values.append((int(task["seed"]), value))
        else:
            grouped[
                (
                    str(task["replicate_id"]),
                    str(task["arm"]),
                    str(task["joint_config_fingerprint"]),
                )
            ].append((int(task["seed"]), value))
    if sorted(seed for seed, _ in anchor_values) != [0, 1, 2]:
        _fail("STATISTICS", "P_ANCHOR_SEED_COMPLETENESS_MISMATCH")
    anchor_mean = float(np.mean([value for _, value in sorted(anchor_values)]))
    config_summaries: list[dict[str, Any]] = []
    for key, observations in grouped.items():
        replicate_id, arm, joint = key
        ordered = sorted(observations)
        if [seed for seed, _ in ordered] != [0, 1, 2]:
            _fail("STATISTICS", "CONFIG_SEED_COMPLETENESS_MISMATCH")
        representative = next(
            task
            for task in tasks
            if task.get("replicate_id") == replicate_id
            and task.get("arm") == arm
            and task.get("joint_config_fingerprint") == joint
        )
        values = np.asarray([value for _, value in ordered], dtype=np.float64)
        config_summaries.append(
            {
                "replicate_id": replicate_id,
                "slot_id": representative["slot_id"],
                "arm": arm,
                "config_id": representative["config_id"],
                "architecture_fingerprint": representative["architecture_fingerprint"],
                "hp_fingerprint": representative["hp_fingerprint"],
                "joint_config_fingerprint": joint,
                "architecture": representative["architecture"],
                "raw_hp_vector": representative["raw_hp_vector"],
                "decoded_hp": representative["decoded_hp"],
                "seeds": [0, 1, 2],
                "validation_accuracies": values.tolist(),
                "config_mean_validation": float(values.mean()),
                "config_sample_sd": float(values.std(ddof=1)),
                "complete": True,
            }
        )
    if len(config_summaries) != 384:
        _fail("STATISTICS", "CONFIG_SUMMARY_CARDINALITY_MISMATCH")
    best_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for replicate_id in REPLICATE_IDS:
        selected: dict[str, dict[str, Any]] = {}
        for arm in ("Control", "Intervention"):
            eligible = [
                row
                for row in config_summaries
                if row["replicate_id"] == replicate_id and row["arm"] == arm
            ]
            if len(eligible) != 16:
                _fail("STATISTICS", "REPLICATE_ARM_CONFIG_COUNT_MISMATCH")
            eligible.sort(
                key=lambda row: (
                    -float(row["config_mean_validation"]),
                    str(row["joint_config_fingerprint"]),
                )
            )
            selected[arm] = eligible[0]
            best_rows.append(
                {
                    "replicate_id": replicate_id,
                    "arm": arm,
                    "joint_config_fingerprint": eligible[0]["joint_config_fingerprint"],
                    "slot_id": eligible[0]["slot_id"],
                    "final_best_mean_validation": eligible[0]["config_mean_validation"],
                }
            )
        deltas.append(
            float(selected["Intervention"]["config_mean_validation"])
            - float(selected["Control"]["config_mean_validation"])
        )
    sign_flip = exact_sign_flip(deltas)
    if (
        inference_plan.get("bootstrap_replicates") != 100000
        or inference_plan.get("sign_flip_permutations") != 4096
        or not isinstance(inference_plan.get("bootstrap_seed"), int)
    ):
        _fail("STATISTICS", "INFERENCE_PLAN_MISMATCH")
    lower = bootstrap_lower_bound(
        deltas, seed=int(inference_plan["bootstrap_seed"]), resamples=100000
    )
    mean_delta = float(np.mean(np.asarray(deltas, dtype=np.float64)))
    wins = sum(value > 0.0 for value in deltas)
    gates = {
        "mean_delta_at_least_0_003": mean_delta >= 0.003,
        "one_sided_p_value_below_0_05": sign_flip["one_sided_p_value"] < 0.05,
        "bootstrap_lower_bound_above_zero": lower > 0.0,
        "wins_at_least_8": wins >= 8,
    }
    classification = (
        "EXPANDED_SEARCH_CONFIRMED_ON_CORA"
        if all(gates.values())
        else "EXPANDED_SEARCH_NOT_CONFIRMED_ON_CORA"
    )
    return {
        "schema_version": "phase-d-statistics-v1",
        "complete_replicate_count": 12,
        "replicate_ids": list(REPLICATE_IDS),
        "complete_config_count": 384,
        "configuration_statistics": config_summaries,
        "replicate_arm_final_bests": best_rows,
        "replicate_deltas": [
            {"replicate_id": replicate, "delta": value}
            for replicate, value in zip(REPLICATE_IDS, deltas)
        ],
        "mean_delta": mean_delta,
        "wins": wins,
        "ties": sum(value == 0.0 for value in deltas),
        "exact_sign_flip": sign_flip,
        "bootstrap_replicates": 100000,
        "bootstrap_one_sided_95_lower_bound": lower,
        "fresh_p_anchor_mean_validation": anchor_mean,
        "classification_gates": gates,
        "classification": classification,
    }


def freeze_challengers(statistics: Mapping[str, Any]) -> dict[str, Any]:
    anchor_mean = float(statistics["fresh_p_anchor_mean_validation"])
    summaries = statistics.get("configuration_statistics", [])
    intervention = [
        dict(row)
        for row in summaries
        if row.get("arm") == "Intervention" and row.get("complete") is True
    ]
    if len(intervention) != 192:
        _fail("CHALLENGERS", "COMPLETE_INTERVENTION_CONFIG_COUNT_MISMATCH")
    eligible = [
        row
        for row in intervention
        if float(row["config_mean_validation"]) - anchor_mean >= 0.001
    ]
    eligible.sort(
        key=lambda row: (
            -float(row["config_mean_validation"]),
            float(row["config_sample_sd"]),
            str(row["joint_config_fingerprint"]),
        )
    )
    selected = eligible[:2]
    return {
        "schema_version": "phase-d-challenger-freeze-v1",
        "fresh_p_anchor_mean_validation": anchor_mean,
        "minimum_mean_advantage_inclusive": 0.001,
        "threshold_eligible_config_count_pre_cap": len(eligible),
        "eligible_challenger_count": len(selected),
        "challengers": selected,
        "phase_e_eligible": len(selected) > 0,
        "phase_e_started": False,
    }


class _AllowedTensorBundle:
    def __init__(self, values: Mapping[str, Any]):
        for key in DATA_ALLOWLIST:
            object.__setattr__(self, key, values[key])


def _worker_gpu_binding(config: Mapping[str, Any], gpu_index: int, gpu_uuid: str) -> dict[str, Any]:
    evidence = audit_gpu_admission(config, require_zero_compute_processes=False)
    match = next((row for row in evidence["gpus"] if row["index"] == gpu_index), None)
    if match is None or match["uuid"] != gpu_uuid:
        _fail("FORMAL_WORKER", "GPU_BINDING_MISMATCH")
    return {"gpu_index": gpu_index, "gpu_uuid": gpu_uuid, "binding_passed": True}


def _formal_worker(payload: Mapping[str, Any]) -> None:
    worker_id = int(payload["worker_id"])
    gpu_index = int(payload["gpu_index"])
    gpu_uuid = str(payload["gpu_uuid"])
    tasks = [dict(row) for row in payload["tasks"]]
    result_root = Path(payload["result_root"])
    shard_path = Path(payload["shard_path"])
    retry_counter = payload["retry_counter"]
    abort_event = payload["abort_event"]
    setup_barrier = payload["setup_barrier"]
    shard: dict[str, Any] = {
        "schema_version": "phase-d-worker-shard-v1",
        "worker_id": worker_id,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "status": "RUNNING",
        "setup_complete": False,
        "ledger_complete": False,
        "results": [],
        "attempts": [],
        "setup_access_ledger": None,
        "test_access": _new_access_ledger(),
    }
    try:
        manifest = verify_prelaunch(result_root)
        verify_live_source_snapshot_binding(result_root, manifest)
        config = load_config(result_root / f"source_snapshot/{CONFIG_REL}")
        shard["gpu_binding"] = _worker_gpu_binding(config, gpu_index, gpu_uuid)
        assert_evaluator_ast_contract()
        assert_layernorm_source_contract()
        setup_barrier.wait()
        if abort_event.is_set():
            _fail("FORMAL_WORKER", "PEER_WORKER_SETUP_FAILED")

        import torch
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures

        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
        dataset = Planetoid(
            root=str(REPO_ROOT / "Cora"),
            name="Cora",
            split="public",
            transform=NormalizeFeatures(),
        )
        loaded_data = dataset[0]
        setup_ledger = _new_access_ledger()
        setup_proxy = ValidationOnlyDataProxy(loaded_data, setup_ledger)
        tensors = {
            key: getattr(setup_proxy, key).to(device) for key in sorted(DATA_ALLOWLIST)
        }
        if setup_ledger["forbidden_request_count"] or setup_ledger["test_accessed"]:
            _fail("FORMAL_WORKER", "SETUP_FIREWALL_VIOLATION")
        in_ch = int(tensors["x"].shape[1])
        out_ch = int(tensors["y"].max().item()) + 1
        bundle = _AllowedTensorBundle(tensors)
        del setup_proxy
        del loaded_data
        del dataset
        shard["setup_access_ledger"] = setup_ledger
        shard["setup_complete"] = True
        shard["test_access"] = _aggregate_access_ledgers([setup_ledger])
        _atomic_replace_json(shard_path, shard)

        for frozen_task in tasks:
            if abort_event.is_set():
                break
            task = copy.deepcopy(frozen_task)
            if (
                task.get("worker_id") != worker_id
                or task.get("gpu_index") != gpu_index
                or task.get("gpu_uuid") != gpu_uuid
                or task.get("track_test") is not False
            ):
                _fail("FORMAL_WORKER", "TASK_WORKER_PROTOCOL_BINDING_MISMATCH")
            task["in_ch"] = in_ch
            task["out_ch"] = out_ch
            completed = False
            for attempt_number in range(1, 4):
                if attempt_number > 1 and abort_event.is_set():
                    break
                ledger = _new_access_ledger()
                proxy = ValidationOnlyDataProxy(bundle, ledger)
                attempt_record: dict[str, Any] = {
                    "task_id": task["task_id"],
                    "arm": task["arm"],
                    "replicate_id": task.get("replicate_id"),
                    "slot_id": task.get("slot_id"),
                    "seed": task["seed"],
                    "component_variant": task["component_variant"],
                    "worker_id": worker_id,
                    "gpu_index": gpu_index,
                    "gpu_uuid": gpu_uuid,
                    "attempt": attempt_number,
                    "status": "IN_PROGRESS",
                    "test_access": ledger,
                }
                shard["attempts"].append(attempt_record)
                shard["test_access"] = _aggregate_access_ledgers(
                    [setup_ledger] + [row["test_access"] for row in shard["attempts"]]
                )
                _atomic_replace_json(shard_path, shard)
                started = time.monotonic()
                try:
                    validation, valid, metadata = run_guarded_evaluation(
                        copy.deepcopy(task), proxy, device
                    )
                    elapsed = time.monotonic() - started
                    if not bool(valid):
                        attempt_record.update(
                            {"status": "EVALUATOR_INVALID", "elapsed_seconds": elapsed}
                        )
                        abort_event.set()
                        break
                    if ledger["forbidden_request_count"] or ledger["test_accessed"]:
                        attempt_record.update(
                            {"status": "TEST_FIREWALL_VIOLATION", "elapsed_seconds": elapsed}
                        )
                        abort_event.set()
                        break
                    attempt_record.update({"status": "SUCCESS", "elapsed_seconds": elapsed})
                    shard["results"].append(
                        {
                            "task_id": task["task_id"],
                            "arm": task["arm"],
                            "replicate_id": task.get("replicate_id"),
                            "slot_id": task.get("slot_id"),
                            "architecture_fingerprint": task["architecture_fingerprint"],
                            "hp_fingerprint": task["hp_fingerprint"],
                            "joint_config_fingerprint": task["joint_config_fingerprint"],
                            "seed": task["seed"],
                            "component_variant": task["component_variant"],
                            "worker_id": worker_id,
                            "gpu_index": gpu_index,
                            "gpu_uuid": gpu_uuid,
                            "validation_accuracy": float(validation),
                            "best_epoch": metadata.get("best_epoch"),
                            "epochs_ran": metadata.get("epochs_ran"),
                            "attempt_count": attempt_number,
                            "test_accessed": False,
                        }
                    )
                    completed = True
                    break
                except BaseException as exc:
                    elapsed = time.monotonic() - started
                    retryable = _is_retryable_exception(exc)
                    attempt_record.update(
                        {
                            "status": "TRANSIENT_FAILURE" if retryable else "NONRETRYABLE_FAILURE",
                            "exception_class": type(exc).__name__,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    if not retryable or attempt_number == 3:
                        abort_event.set()
                        break
                    if attempt_number == 1:
                        with retry_counter.get_lock():
                            retry_counter.value += 1
                            retry_count = int(retry_counter.value)
                        if retry_count > MAX_RETRIED_TASK_COUNT:
                            attempt_record["status"] = "RETRY_TASK_58_LIMIT_EXCEEDED"
                            abort_event.set()
                            break
                finally:
                    shard["test_access"] = _aggregate_access_ledgers(
                        [setup_ledger] + [row["test_access"] for row in shard["attempts"]]
                    )
                    _atomic_replace_json(shard_path, shard)
            if not completed:
                shard["status"] = "RUN_FAILED"
                abort_event.set()
                break
        if shard["status"] == "RUNNING":
            shard["status"] = "COMPLETE" if len(shard["results"]) == len(tasks) else "RUN_FAILED"
        shard["ledger_complete"] = all(
            row.get("status") != "IN_PROGRESS" for row in shard["attempts"]
        )
    except BaseException as exc:
        try:
            setup_barrier.abort()
        except BaseException as barrier_exc:
            shard["barrier_abort_exception_class"] = type(barrier_exc).__name__
        shard["status"] = "RUN_FAILED"
        shard["error_code"] = "WORKER_TOP_LEVEL_FAILURE"
        shard["exception_class"] = type(exc).__name__
        abort_event.set()
    try:
        _atomic_replace_json(shard_path, shard)
    except BaseException:
        os._exit(3)


def _run_formal_workers(
    tasks: Sequence[Mapping[str, Any]], result_root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    context = multiprocessing.get_context("spawn")
    retry_counter = context.Value("i", 0)
    abort_event = context.Event()
    setup_barrier = context.Barrier(4)
    processes: list[Any] = []
    shard_paths: list[Path] = []
    try:
        for mapping in config["workers"]["mapping"]:
            worker_id = int(mapping["worker_id"])
            shard_path = result_root / "formal/shards" / f"worker_{worker_id}.json"
            shard_paths.append(shard_path)
            worker_tasks = [row for row in tasks if int(row["worker_id"]) == worker_id]
            process = context.Process(
                target=_formal_worker,
                args=(
                    {
                        "worker_id": worker_id,
                        "gpu_index": int(mapping["gpu_index"]),
                        "gpu_uuid": str(mapping["gpu_uuid"]),
                        "tasks": worker_tasks,
                        "result_root": str(result_root),
                        "shard_path": str(shard_path),
                        "retry_counter": retry_counter,
                        "abort_event": abort_event,
                        "setup_barrier": setup_barrier,
                    },
                ),
                name=f"phase_d_worker_{worker_id}",
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    except BaseException as exc:
        abort_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise ProtocolFailure("FORMAL", "WORKER_PROCESS_START_OR_JOIN_FAILED") from exc
    shards = []
    for path in shard_paths:
        if path.is_file() and not path.is_symlink():
            shards.append(
                _load_bound_record(path, "FORMAL", "WORKER_SHARD_MISSING_OR_INVALID")
            )
    if any(process.exitcode != 0 for process in processes):
        for shard in shards:
            shard["worker_process_exit_failure"] = True
    return shards, int(retry_counter.value)


def _sort_attempts_and_results(
    shards: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    order = {str(row["task_id"]): index for index, row in enumerate(tasks)}
    observed_workers: set[int] = set()
    expected_mapping = {int(row["worker_id"]): row for row in EXPECTED_WORKER_MAPPING}
    for shard in shards:
        worker_id = shard.get("worker_id")
        if type(worker_id) is not int or worker_id in observed_workers:
            _fail("FORMAL_LEDGER", "WORKER_SHARD_ID_INVALID_OR_DUPLICATE")
        mapping = expected_mapping.get(worker_id)
        if (
            mapping is None
            or shard.get("gpu_index") != mapping["gpu_index"]
            or shard.get("gpu_uuid") != mapping["gpu_uuid"]
            or not isinstance(shard.get("attempts"), list)
            or not isinstance(shard.get("results"), list)
        ):
            _fail("FORMAL_LEDGER", "WORKER_SHARD_BINDING_INVALID")
        observed_workers.add(worker_id)
        for row in [*shard["attempts"], *shard["results"]]:
            if (
                row.get("worker_id") != worker_id
                or row.get("gpu_index") != mapping["gpu_index"]
                or row.get("gpu_uuid") != mapping["gpu_uuid"]
            ):
                _fail("FORMAL_LEDGER", "SHARD_ROW_WORKER_BINDING_MISMATCH")
    attempts = [dict(row) for shard in shards for row in shard.get("attempts", [])]
    results = [dict(row) for shard in shards for row in shard.get("results", [])]
    attempts.sort(key=lambda row: (order.get(str(row.get("task_id")), 10**9), int(row.get("attempt", 0))))
    results.sort(key=lambda row: order.get(str(row.get("task_id")), 10**9))
    return attempts, results


def _formal_test_ledger(
    shards: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ledgers: list[Mapping[str, Any]] = []
    setup_complete = len(shards) == 4
    worker_ledgers_complete = len(shards) == 4
    for shard in shards:
        setup = shard.get("setup_access_ledger")
        if isinstance(setup, dict):
            ledgers.append(setup)
        else:
            setup_complete = False
        worker_ledgers_complete = bool(
            worker_ledgers_complete and shard.get("ledger_complete") is True
        )
    for row in attempts:
        ledger = row.get("test_access")
        if isinstance(ledger, dict):
            ledgers.append(ledger)
        else:
            worker_ledgers_complete = False
    aggregate = _aggregate_access_ledgers(ledgers)
    if aggregate["test_accessed"]:
        access_status = "DETECTED"
    elif setup_complete and worker_ledgers_complete:
        access_status = "VERIFIED_NOT_ACCESSED"
    else:
        access_status = "UNVERIFIED_FAIL_CLOSED"
    return {
        "schema_version": "phase-d-formal-test-access-ledger-v1",
        **aggregate,
        "worker_setup_complete": setup_complete,
        "worker_ledgers_complete": worker_ledgers_complete,
        "test_access_status": access_status,
        "formal_track_test_literal": False,
        "negative_controls": "test_firewall.json",
    }


def _write_formal_ledgers(
    result_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]],
    retry_counter_value: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    attempts, results = _sort_attempts_and_results(shards, tasks)
    validation = _validate_attempt_and_result_records(tasks, attempts, results)
    result_ids = {str(row["task_id"]) for row in results}
    attempt_ids = {str(row["task_id"]) for row in attempts}
    retried_ids = set(validation["retry_task_ids"])
    if not 0 <= int(retry_counter_value) <= MAX_RETRIED_TASK_COUNT + 1:
        _fail("FORMAL_LEDGER", "RETRY_REQUEST_COUNTER_INVALID")
    terminal_attempt_by_task: dict[str, Mapping[str, Any]] = {}
    for row in attempts:
        terminal_attempt_by_task[str(row["task_id"])] = row
    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in result_ids:
            status_value = "SUCCESS"
        elif task_id in attempt_ids:
            status_value = "FAILED"
        else:
            status_value = "NOT_RUN"
        task_rows.append(
            {
                "task_id": task_id,
                "arm": task["arm"],
                "replicate_id": task.get("replicate_id"),
                "slot_id": task.get("slot_id"),
                "seed": task["seed"],
                "worker_id": task["worker_id"],
                "gpu_index": task["gpu_index"],
                "gpu_uuid": task["gpu_uuid"],
                "status": status_value,
                "attempt_count": int(terminal_attempt_by_task.get(task_id, {}).get("attempt", 0)),
            }
        )
    test_ledger = _formal_test_ledger(shards, attempts)
    _atomic_replace_json(
        result_root / "formal/attempt_ledger.json",
        {
            "schema_version": "phase-d-attempt-ledger-v1",
            "physical_gnn_attempt_count": len(attempts),
            "retry_task_count": len(retried_ids),
            "retry_request_count": int(retry_counter_value),
            "retry_limit_exceeded": int(retry_counter_value) > MAX_RETRIED_TASK_COUNT,
            "maximum_allowed_retry_task_count": 57,
            "attempts": attempts,
        },
    )
    _atomic_replace_json(
        result_root / "formal/task_ledger.json",
        {
            "schema_version": "phase-d-formal-task-ledger-v1",
            "planned_task_count": 1155,
            "completed_task_count": len(results),
            "failed_task_count": sum(row["status"] == "FAILED" for row in task_rows),
            "not_run_task_count": sum(row["status"] == "NOT_RUN" for row in task_rows),
            "tasks": task_rows,
        },
    )
    _atomic_replace_json(result_root / "formal/test_access_ledger.json", test_ledger)
    _atomic_replace_json(
        result_root / "formal/results.json",
        {
            "schema_version": "phase-d-results-v1",
            "completed_task_count": len(results),
            "failed_task_count": sum(row["status"] == "FAILED" for row in task_rows),
            "results": results,
        },
    )
    return attempts, results, test_ledger


def _complete_terminal_state(
    classification: str,
    test_ledger: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": "COMPLETE",
        "completion": "REACHED",
        "scientific_decision": "REACHED",
        "classification": classification,
        "planned_task_count": 1155,
        "completed_task_count": 1155,
        "failed_task_count": 0,
        "Control_completed_task_count": 576,
        "Intervention_completed_task_count": 576,
        "P_anchor_completed_task_count": 3,
        "complete_replicate_count": 12,
        "retry_task_count": int(attempt_ledger["retry_task_count"]),
        "physical_gnn_attempt_count": int(attempt_ledger["physical_gnn_attempt_count"]),
        "test_accessed": False,
        "test_access_status": test_ledger["test_access_status"],
        "phase_e_started": False,
    }


def _write_success_terminal(
    result_root: Path,
    statistics: Mapping[str, Any],
    challengers: Mapping[str, Any],
    test_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    classification = str(statistics["classification"])
    attempt_ledger = _load_bound_record(
        result_root / "formal/attempt_ledger.json",
        "TERMINAL",
        "ATTEMPT_LEDGER_MISSING_OR_INVALID",
    )
    state = _complete_terminal_state(classification, test_ledger, attempt_ledger)
    decision = {
        "schema_version": "phase-d-decision-v1",
        **state,
        "classification_gates": statistics["classification_gates"],
        "mean_delta": statistics["mean_delta"],
        "one_sided_p_value": statistics["exact_sign_flip"]["one_sided_p_value"],
        "bootstrap_lower_bound": statistics["bootstrap_one_sided_95_lower_bound"],
        "wins": statistics["wins"],
        "threshold_eligible_config_count_pre_cap": challengers[
            "threshold_eligible_config_count_pre_cap"
        ],
        "eligible_challenger_count": challengers["eligible_challenger_count"],
        "phase_e_eligible": challengers["phase_e_eligible"],
    }
    _atomic_replace_json(result_root / "formal/decision.json", decision)
    _atomic_replace_json(result_root / "formal/state.json", state)
    terminal = {
        "schema_version": "phase-d-terminal-v1",
        **state,
        "decision_sha256": sha256_path(result_root / "formal/decision.json"),
        "state_sha256": sha256_path(result_root / "formal/state.json"),
        "attempt_ledger_sha256": sha256_path(result_root / "formal/attempt_ledger.json"),
        "task_ledger_sha256": sha256_path(result_root / "formal/task_ledger.json"),
        "test_access_ledger_sha256": sha256_path(
            result_root / "formal/test_access_ledger.json"
        ),
        "results_sha256": sha256_path(result_root / "formal/results.json"),
        "statistics_sha256": sha256_path(result_root / "formal/statistics.json"),
        "challengers_sha256": sha256_path(result_root / "formal/challengers.json"),
        "phase_e_started": False,
    }
    _atomic_replace_json(result_root / "formal/terminal.json", terminal)
    return terminal


def _write_unverified_failure_ledgers(
    result_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    retry_counter_value: int,
    ledger_failure: BaseException,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    error_code = (
        ledger_failure.error_code
        if isinstance(ledger_failure, ProtocolFailure)
        else "UNCLASSIFIED_LEDGER_EVIDENCE_FAILURE"
    )
    retry_requests = min(max(int(retry_counter_value), 0), 58)
    task_rows = [
        {
            "task_id": task["task_id"],
            "arm": task["arm"],
            "replicate_id": task.get("replicate_id"),
            "slot_id": task.get("slot_id"),
            "seed": task["seed"],
            "worker_id": task["worker_id"],
            "gpu_index": task["gpu_index"],
            "gpu_uuid": task["gpu_uuid"],
            "status": "UNVERIFIED",
            "attempt_count": None,
        }
        for task in tasks
    ]
    test_ledger = {
        "schema_version": "phase-d-formal-test-access-ledger-v1",
        **_new_access_ledger(),
        "worker_setup_complete": False,
        "worker_ledgers_complete": False,
        "test_access_status": "UNVERIFIED_FAIL_CLOSED",
        "formal_track_test_literal": False,
        "negative_controls": "test_firewall.json",
        "evidence_discarded_fail_closed": True,
        "evidence_error_code": error_code,
    }
    _atomic_replace_json(
        result_root / "formal/attempt_ledger.json",
        {
            "schema_version": "phase-d-attempt-ledger-v1",
            "physical_gnn_attempt_count": None,
            "retry_task_count": None,
            "retry_request_count": retry_requests,
            "retry_limit_exceeded": retry_requests > 57,
            "maximum_allowed_retry_task_count": 57,
            "attempts": [],
            "evidence_discarded_fail_closed": True,
            "evidence_error_code": error_code,
        },
    )
    _atomic_replace_json(
        result_root / "formal/task_ledger.json",
        {
            "schema_version": "phase-d-formal-task-ledger-v1",
            "planned_task_count": 1155,
            "completed_task_count": None,
            "failed_task_count": None,
            "not_run_task_count": None,
            "unverified_task_count": 1155,
            "tasks": task_rows,
            "evidence_discarded_fail_closed": True,
            "evidence_error_code": error_code,
        },
    )
    _atomic_replace_json(result_root / "formal/test_access_ledger.json", test_ledger)
    _atomic_replace_json(
        result_root / "formal/results.json",
        {
            "schema_version": "phase-d-results-v1",
            "completed_task_count": None,
            "failed_task_count": None,
            "results": [],
            "evidence_discarded_fail_closed": True,
            "evidence_error_code": error_code,
        },
    )
    return [], [], test_ledger


def _write_failure_terminal(
    result_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]],
    retry_counter_value: int,
    failure: BaseException,
) -> dict[str, Any]:
    try:
        if len(shards) != 4:
            _fail("FORMAL_LEDGER", "WORKER_SHARD_SET_INCOMPLETE_OR_INVALID")
        attempts, results, test_ledger = _write_formal_ledgers(
            result_root, tasks, shards, retry_counter_value
        )
    except BaseException as ledger_failure:
        attempts, results, test_ledger = _write_unverified_failure_ledgers(
            result_root, tasks, retry_counter_value, ledger_failure
        )
    test_value: bool | None
    if test_ledger["test_accessed"]:
        test_value = True
    elif test_ledger["test_access_status"] == "VERIFIED_NOT_ACCESSED":
        test_value = False
    else:
        test_value = None
    failure_stage = failure.stage if isinstance(failure, ProtocolFailure) else "FORMAL"
    failure_code = (
        failure.error_code if isinstance(failure, ProtocolFailure) else "UNCLASSIFIED_FORMAL_FAILURE"
    )
    evidence_untrusted = test_ledger.get("evidence_discarded_fail_closed") is True
    trusted_completed_count: int | None = None if evidence_untrusted else len(results)
    statistics = {
        "schema_version": "phase-d-statistics-v1",
        "scientific_decision": "NOT_REACHED",
        "classification": None,
        "complete_replicate_count": 0,
        "completed_task_count": trusted_completed_count,
    }
    challengers = {
        "schema_version": "phase-d-challenger-freeze-v1",
        "threshold_eligible_config_count_pre_cap": 0,
        "eligible_challenger_count": 0,
        "challengers": [],
        "phase_e_eligible": False,
        "phase_e_started": False,
        "scientific_decision": "NOT_REACHED",
    }
    _atomic_replace_json(result_root / "formal/statistics.json", statistics)
    _atomic_replace_json(result_root / "formal/challengers.json", challengers)
    successful_ids = {str(row["task_id"]) for row in results}
    failed_count: int | None = (
        None
        if evidence_untrusted
        else len({str(row["task_id"]) for row in attempts} - successful_ids)
    )
    retry_task_count: int | None = (
        None
        if evidence_untrusted
        else len({str(row["task_id"]) for row in attempts if int(row.get("attempt", 0)) > 1})
    )
    physical_attempt_count: int | None = None if evidence_untrusted else len(attempts)
    state = {
        "status": "RUN_FAILED",
        "completion": "INCOMPLETE",
        "scientific_decision": "NOT_REACHED",
        "classification": None,
        "planned_task_count": 1155,
        "completed_task_count": trusted_completed_count,
        "failed_task_count": failed_count,
        "retry_task_count": retry_task_count,
        "physical_gnn_attempt_count": physical_attempt_count,
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        "failure_class": type(failure).__name__,
        "test_accessed": test_value,
        "test_access_status": test_ledger["test_access_status"],
        "phase_e_started": False,
    }
    decision = {"schema_version": "phase-d-decision-v1", **state}
    _atomic_replace_json(result_root / "formal/decision.json", decision)
    _atomic_replace_json(result_root / "formal/state.json", state)
    terminal = {
        "schema_version": "phase-d-terminal-v1",
        **state,
        "decision_sha256": sha256_path(result_root / "formal/decision.json"),
        "state_sha256": sha256_path(result_root / "formal/state.json"),
        "attempt_ledger_sha256": sha256_path(result_root / "formal/attempt_ledger.json"),
        "task_ledger_sha256": sha256_path(result_root / "formal/task_ledger.json"),
        "test_access_ledger_sha256": sha256_path(
            result_root / "formal/test_access_ledger.json"
        ),
        "results_sha256": sha256_path(result_root / "formal/results.json"),
        "statistics_sha256": sha256_path(result_root / "formal/statistics.json"),
        "challengers_sha256": sha256_path(result_root / "formal/challengers.json"),
        "phase_e_started": False,
    }
    _atomic_replace_json(result_root / "formal/terminal.json", terminal)
    return terminal


def _validate_terminal_consistency(result_root: Path) -> dict[str, Any]:
    terminal = _load_bound_record(
        result_root / "formal/terminal.json", "TERMINAL", "TERMINAL_MISSING_OR_INVALID"
    )
    formal_paths = {
        "decision": result_root / "formal/decision.json",
        "state": result_root / "formal/state.json",
        "attempt_ledger": result_root / "formal/attempt_ledger.json",
        "task_ledger": result_root / "formal/task_ledger.json",
        "test_access_ledger": result_root / "formal/test_access_ledger.json",
        "results": result_root / "formal/results.json",
        "statistics": result_root / "formal/statistics.json",
        "challengers": result_root / "formal/challengers.json",
    }
    records = {
        name: _load_bound_record(path, "TERMINAL", f"{name.upper()}_MISSING_OR_INVALID")
        for name, path in formal_paths.items()
    }
    hash_fields = {
        "decision_sha256": "decision",
        "state_sha256": "state",
        "attempt_ledger_sha256": "attempt_ledger",
        "task_ledger_sha256": "task_ledger",
        "test_access_ledger_sha256": "test_access_ledger",
        "results_sha256": "results",
        "statistics_sha256": "statistics",
        "challengers_sha256": "challengers",
    }
    if any(
        terminal.get(field) != sha256_path(formal_paths[name])
        for field, name in hash_fields.items()
    ):
        _fail("TERMINAL", "TERMINAL_ARTIFACT_HASH_BINDING_MISMATCH")
    state = records["state"]
    decision = records["decision"]
    attempt_ledger = records["attempt_ledger"]
    task_ledger = records["task_ledger"]
    test_ledger = records["test_access_ledger"]
    results_record = records["results"]
    statistics = records["statistics"]
    challengers = records["challengers"]
    frozen_tasks_record = _load_bound_record(
        result_root / "frozen_tasks.json", "TERMINAL", "FROZEN_TASKS_MISSING_OR_INVALID"
    )
    tasks = frozen_tasks_record.get("tasks")
    attempts = attempt_ledger.get("attempts")
    results = results_record.get("results")
    task_rows = task_ledger.get("tasks")
    if (
        frozen_tasks_record.get("schema_version") != "phase-d-task-freeze-v1"
        or not isinstance(tasks, list)
        or len(tasks) != 1155
        or attempt_ledger.get("schema_version") != "phase-d-attempt-ledger-v1"
        or not isinstance(attempts, list)
        or task_ledger.get("schema_version") != "phase-d-formal-task-ledger-v1"
        or not isinstance(task_rows, list)
        or results_record.get("schema_version") != "phase-d-results-v1"
        or not isinstance(results, list)
        or test_ledger.get("schema_version") != "phase-d-formal-test-access-ledger-v1"
    ):
        _fail("TERMINAL", "FORMAL_LEDGER_SCHEMA_MISMATCH")
    retry_request_count = attempt_ledger.get("retry_request_count")
    untrusted_flags = [
        record.get("evidence_discarded_fail_closed") is True
        for record in (attempt_ledger, task_ledger, test_ledger, results_record)
    ]
    if any(untrusted_flags) and not all(untrusted_flags):
        _fail("TERMINAL", "UNTRUSTED_EVIDENCE_FLAG_MISMATCH")
    evidence_untrusted = all(untrusted_flags)
    if type(retry_request_count) is not int or not 0 <= retry_request_count <= 58:
        _fail("TERMINAL", "ATTEMPT_LEDGER_HEADER_MISMATCH")
    if evidence_untrusted:
        if attempts or results:
            _fail("TERMINAL", "UNTRUSTED_LEDGER_MUST_NOT_ASSERT_OBSERVATIONS")
        retry_count: int | None = None
        result_ids: set[str] = set()
        failed_count: int | None = None
        not_run_count: int | None = None
        expected_task_rows = [
            {
                "task_id": task["task_id"],
                "arm": task["arm"],
                "replicate_id": task.get("replicate_id"),
                "slot_id": task.get("slot_id"),
                "seed": task["seed"],
                "worker_id": task["worker_id"],
                "gpu_index": task["gpu_index"],
                "gpu_uuid": task["gpu_uuid"],
                "status": "UNVERIFIED",
                "attempt_count": None,
            }
            for task in tasks
        ]
        if (
            task_rows != expected_task_rows
            or attempt_ledger.get("physical_gnn_attempt_count") is not None
            or attempt_ledger.get("retry_task_count") is not None
            or attempt_ledger.get("maximum_allowed_retry_task_count") != 57
            or attempt_ledger.get("retry_limit_exceeded") is not (retry_request_count > 57)
            or task_ledger.get("planned_task_count") != 1155
            or task_ledger.get("completed_task_count") is not None
            or task_ledger.get("failed_task_count") is not None
            or task_ledger.get("not_run_task_count") is not None
            or task_ledger.get("unverified_task_count") != 1155
            or results_record.get("completed_task_count") is not None
            or results_record.get("failed_task_count") is not None
        ):
            _fail("TERMINAL", "UNTRUSTED_LEDGER_HEADER_MISMATCH")
    else:
        shards = _available_shards(result_root)
        shard_attempts, shard_results = _sort_attempts_and_results(shards, tasks)
        if shard_attempts != attempts or shard_results != results:
            _fail("TERMINAL", "FORMAL_LEDGER_SHARD_BINDING_MISMATCH")
        reconstructed_test_ledger = _formal_test_ledger(shards, attempts)
        if test_ledger != reconstructed_test_ledger:
            _fail("TERMINAL", "TEST_ACCESS_LEDGER_RECOMPUTATION_MISMATCH")
        execution = _validate_attempt_and_result_records(tasks, attempts, results)
        retry_count = len(execution["retry_task_ids"])
        if (
            attempt_ledger.get("physical_gnn_attempt_count") != len(attempts)
            or attempt_ledger.get("retry_task_count") != retry_count
            or attempt_ledger.get("maximum_allowed_retry_task_count") != 57
            or not retry_count <= 57
            or not retry_count <= retry_request_count
            or attempt_ledger.get("retry_limit_exceeded") is not (retry_request_count > 57)
        ):
            _fail("TERMINAL", "ATTEMPT_LEDGER_HEADER_MISMATCH")
        result_ids = set(execution["result_by_id"])
        attempt_ids = set(execution["attempts_by_task"])
        expected_task_rows = []
        for task in tasks:
            task_id = str(task["task_id"])
            expected_task_rows.append(
                {
                    "task_id": task_id,
                    "arm": task["arm"],
                    "replicate_id": task.get("replicate_id"),
                    "slot_id": task.get("slot_id"),
                    "seed": task["seed"],
                    "worker_id": task["worker_id"],
                    "gpu_index": task["gpu_index"],
                    "gpu_uuid": task["gpu_uuid"],
                    "status": (
                        "SUCCESS"
                        if task_id in result_ids
                        else "FAILED" if task_id in attempt_ids else "NOT_RUN"
                    ),
                    "attempt_count": len(execution["attempts_by_task"].get(task_id, [])),
                }
            )
        if task_rows != expected_task_rows:
            _fail("TERMINAL", "TASK_LEDGER_FROZEN_TASK_BINDING_MISMATCH")
        failed_count = sum(row["status"] == "FAILED" for row in expected_task_rows)
        not_run_count = sum(row["status"] == "NOT_RUN" for row in expected_task_rows)
        if (
            task_ledger.get("planned_task_count") != 1155
            or task_ledger.get("completed_task_count") != len(results)
            or task_ledger.get("failed_task_count") != failed_count
            or task_ledger.get("not_run_task_count") != not_run_count
            or results_record.get("completed_task_count") != len(results)
            or results_record.get("failed_task_count") != failed_count
        ):
            _fail("TERMINAL", "TASK_OR_RESULT_LEDGER_HEADER_MISMATCH")
    test_accessed = test_ledger.get("test_accessed")
    test_status = test_ledger.get("test_access_status")
    forbidden_count = test_ledger.get("forbidden_request_count")
    if (
        type(test_accessed) is not bool
        or type(forbidden_count) is not int
        or forbidden_count < 0
        or test_status not in {
            "DETECTED", "VERIFIED_NOT_ACCESSED", "UNVERIFIED_FAIL_CLOSED"
        }
        or (test_status == "DETECTED") is not test_accessed
        or test_ledger.get("formal_track_test_literal") is not False
        or test_ledger.get("negative_controls") != "test_firewall.json"
    ):
        _fail("TERMINAL", "TEST_ACCESS_LEDGER_CONSISTENCY_MISMATCH")
    for key, value in state.items():
        if terminal.get(key) != value:
            _fail("TERMINAL", "TERMINAL_STATE_VALUE_MISMATCH")
    if set(terminal) != {"schema_version", *state.keys(), *hash_fields.keys()}:
        _fail("TERMINAL", "TERMINAL_FIELD_SET_MISMATCH")
    if terminal.get("schema_version") != "phase-d-terminal-v1":
        _fail("TERMINAL", "TERMINAL_SCHEMA_MISMATCH")
    status_value = terminal.get("status")
    if status_value == "COMPLETE":
        expected_statistics = compute_statistics(
            results,
            tasks,
            _load_bound_record(
                result_root / "inference_plan.json",
                "TERMINAL",
                "INFERENCE_PLAN_MISSING_OR_INVALID",
            ),
        )
        if statistics != expected_statistics:
            _fail("TERMINAL", "STATISTICS_RECOMPUTATION_MISMATCH")
        expected_challengers = freeze_challengers(statistics)
        if challengers != expected_challengers:
            _fail("TERMINAL", "CHALLENGER_FREEZE_MISMATCH")
        required = {
            "status": "COMPLETE",
            "completion": "REACHED",
            "scientific_decision": "REACHED",
            "classification": statistics["classification"],
            "planned_task_count": 1155,
            "completed_task_count": 1155,
            "failed_task_count": 0,
            "Control_completed_task_count": 576,
            "Intervention_completed_task_count": 576,
            "P_anchor_completed_task_count": 3,
            "complete_replicate_count": 12,
            "retry_task_count": retry_count,
            "physical_gnn_attempt_count": len(attempts),
            "test_accessed": False,
            "test_access_status": "VERIFIED_NOT_ACCESSED",
            "phase_e_started": False,
        }
        if state != required or terminal.get("completion") != "REACHED":
            _fail("TERMINAL", "COMPLETE_TERMINAL_CONTRACT_MISMATCH")
        if (
            len(result_ids) != 1155
            or failed_count != 0
            or not_run_count != 0
            or test_accessed is not False
            or forbidden_count != 0
            or test_ledger.get("worker_setup_complete") is not True
            or test_ledger.get("worker_ledgers_complete") is not True
            or len(_available_shards(result_root)) != 4
            or retry_request_count > 57
        ):
            _fail("TERMINAL", "COMPLETE_LEDGER_CONTRACT_MISMATCH")
        expected_decision = {
            "schema_version": "phase-d-decision-v1",
            **required,
            "classification_gates": statistics["classification_gates"],
            "mean_delta": statistics["mean_delta"],
            "one_sided_p_value": statistics["exact_sign_flip"]["one_sided_p_value"],
            "bootstrap_lower_bound": statistics["bootstrap_one_sided_95_lower_bound"],
            "wins": statistics["wins"],
            "threshold_eligible_config_count_pre_cap": challengers[
                "threshold_eligible_config_count_pre_cap"
            ],
            "eligible_challenger_count": challengers["eligible_challenger_count"],
            "phase_e_eligible": challengers["phase_e_eligible"],
        }
        if decision != expected_decision:
            _fail("TERMINAL", "COMPLETE_DECISION_BINDING_MISMATCH")
    elif status_value == "RUN_FAILED":
        trusted_completed_count = None if evidence_untrusted else len(results)
        trusted_physical_attempt_count = None if evidence_untrusted else len(attempts)
        expected_test_value = (
            True
            if test_accessed
            else False if test_status == "VERIFIED_NOT_ACCESSED" else None
        )
        if (
            terminal.get("completion") != "INCOMPLETE"
            or terminal.get("scientific_decision") != "NOT_REACHED"
            or terminal.get("classification") is not None
            or terminal.get("planned_task_count") != 1155
            or terminal.get("completed_task_count") != trusted_completed_count
            or terminal.get("failed_task_count") != failed_count
            or terminal.get("retry_task_count") != retry_count
            or terminal.get("physical_gnn_attempt_count") != trusted_physical_attempt_count
            or terminal.get("test_accessed") is not expected_test_value
            or terminal.get("test_access_status") != test_status
            or terminal.get("phase_e_started") is not False
            or not isinstance(terminal.get("failure_stage"), str)
            or not isinstance(terminal.get("failure_code"), str)
            or not isinstance(terminal.get("failure_class"), str)
        ):
            _fail("TERMINAL", "FAILED_TERMINAL_CONTRACT_MISMATCH")
        expected_failure_statistics = {
            "schema_version": "phase-d-statistics-v1",
            "scientific_decision": "NOT_REACHED",
            "classification": None,
            "complete_replicate_count": 0,
            "completed_task_count": trusted_completed_count,
        }
        expected_failure_challengers = {
            "schema_version": "phase-d-challenger-freeze-v1",
            "threshold_eligible_config_count_pre_cap": 0,
            "eligible_challenger_count": 0,
            "challengers": [],
            "phase_e_eligible": False,
            "phase_e_started": False,
            "scientific_decision": "NOT_REACHED",
        }
        if statistics != expected_failure_statistics or challengers != expected_failure_challengers:
            _fail("TERMINAL", "FAILED_SCIENTIFIC_ARTIFACT_MISMATCH")
        if decision != {"schema_version": "phase-d-decision-v1", **state}:
            _fail("TERMINAL", "FAILED_DECISION_BINDING_MISMATCH")
    else:
        _fail("TERMINAL", "TERMINAL_STATUS_INVALID")
    return terminal


def _walk_regular_files(root: Path) -> set[str]:
    observed: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in directory_names:
            mode = (current / name).lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                _fail("ARCHIVE", "NON_DIRECTORY_OR_SYMLINK_TREE_ENTRY")
        for name in file_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                _fail("ARCHIVE", "NONREGULAR_OR_SYMLINK_FILE")
            observed.add(path.relative_to(root).as_posix())
    return observed


def _file_record(result_root: Path, relative: str) -> dict[str, Any]:
    path = result_root / relative
    if not path.is_file() or path.is_symlink():
        _fail("ARCHIVE", "FORMAL_FILE_MISSING_OR_INVALID")
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def _formal_manifest(result_root: Path, prelaunch: Mapping[str, Any]) -> dict[str, Any]:
    prelaunch_paths = {str(row["path"]) for row in prelaunch["files"]} | {"manifest.json"}
    excluded = {
        "formal/manifest.json",
        ARCHIVE_BASENAME,
        ARCHIVE_SIDECAR_BASENAME,
        "completion.json",
    }
    observed = _walk_regular_files(result_root)
    formal_paths = sorted(observed - prelaunch_paths - excluded)
    if observed & {
        ARCHIVE_BASENAME,
        ARCHIVE_SIDECAR_BASENAME,
        "completion.json",
        "formal/manifest.json",
    }:
        _fail("ARCHIVE", "ARCHIVE_TARGET_ALREADY_EXISTS")
    required_formal = (
        (FORMAL_CORE_PATHS - {"formal/manifest.json"})
        | {"launch/launch_request.json", "launch/runner_entered.json"}
    )
    allowed_formal = required_formal | FORMAL_SHARD_PATHS
    if not required_formal <= set(formal_paths):
        _fail("ARCHIVE", "REQUIRED_FORMAL_ARTIFACT_MISSING")
    if not set(formal_paths) <= allowed_formal:
        _fail("ARCHIVE", "UNAUTHORIZED_FORMAL_ARTIFACT")
    if set(formal_paths) & prelaunch_paths:
        _fail("ARCHIVE", "PRELAUNCH_FORMAL_PATH_OVERLAP")
    records = [_file_record(result_root, relative) for relative in formal_paths]
    payload = {
        "schema_version": "phase-d-formal-manifest-v1",
        "file_count": len(records),
        "files": records,
        "prelaunch_manifest_root_sha256": prelaunch["manifest_root_sha256"],
        "manifest_self_excluded": True,
        "archive_self_excluded": True,
        "archive_sidecar_excluded": True,
        "completion_envelope_excluded": True,
    }
    return {**payload, "manifest_root_sha256": sha256_bytes(canonical_json_bytes(payload))}


def finalize_formal_archive(result_root: Path) -> dict[str, Any]:
    prelaunch = verify_prelaunch(result_root)
    _validate_terminal_consistency(result_root)
    archive_path = result_root / ARCHIVE_BASENAME
    sidecar_path = result_root / ARCHIVE_SIDECAR_BASENAME
    formal_manifest_path = result_root / "formal/manifest.json"
    if (
        archive_path.exists()
        or archive_path.is_symlink()
        or sidecar_path.exists()
        or sidecar_path.is_symlink()
        or formal_manifest_path.exists()
        or formal_manifest_path.is_symlink()
        or (result_root / "completion.json").exists()
        or (result_root / "completion.json").is_symlink()
    ):
        _fail("ARCHIVE", "FINALIZATION_TARGET_ALREADY_EXISTS")
    formal_manifest = _formal_manifest(result_root, prelaunch)
    _write_exclusive(formal_manifest_path, json_asset(formal_manifest))
    prelaunch_records = {str(row["path"]): dict(row) for row in prelaunch["files"]}
    formal_records = {str(row["path"]): dict(row) for row in formal_manifest["files"]}
    if set(prelaunch_records) & set(formal_records):
        _fail("ARCHIVE", "MANIFEST_PATH_OVERLAP")
    member_paths = sorted(
        set(prelaunch_records)
        | {"manifest.json"}
        | set(formal_records)
        | {"formal/manifest.json"}
    )
    if len(member_paths) != len(set(member_paths)):
        _fail("ARCHIVE", "ARCHIVE_MEMBER_PATH_NOT_UNIQUE")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{ARCHIVE_BASENAME}.", dir=result_root.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    linked = False
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
                with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                    for relative in member_paths:
                        data = (result_root / relative).read_bytes()
                        info = tarfile.TarInfo(name=f"{EXPERIMENT}/{relative}")
                        info.size = len(data)
                        info.mode = 0o644
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        archive.addfile(info, io.BytesIO(data))
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        try:
            os.link(temporary, archive_path)
        except FileExistsError as exc:
            raise ProtocolFailure("ARCHIVE", "ARCHIVE_TARGET_ALREADY_EXISTS") from exc
        linked = True
    except ProtocolFailure:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ProtocolFailure("ARCHIVE", "ARCHIVE_CREATION_FAILED") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    if not linked:
        _fail("ARCHIVE", "ARCHIVE_PUBLICATION_FAILED")
    archive_sha = sha256_path(archive_path)
    _write_exclusive(
        sidecar_path,
        f"{archive_sha}  {ARCHIVE_BASENAME}\n".encode("ascii"),
    )
    validation = validate_formal_archive(result_root)
    return {
        "archive_path": ARCHIVE_BASENAME,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha,
        "sidecar_path": ARCHIVE_SIDECAR_BASENAME,
        "formal_manifest_root_sha256": formal_manifest["manifest_root_sha256"],
        "validation": validation,
    }


def validate_formal_archive(result_root: Path) -> dict[str, Any]:
    prelaunch = verify_prelaunch(result_root)
    formal_manifest = _load_bound_record(
        result_root / "formal/manifest.json", "ARCHIVE", "FORMAL_MANIFEST_INVALID"
    )
    formal_rows = _verify_manifest_payload(
        formal_manifest, "phase-d-formal-manifest-v1"
    )
    if formal_manifest.get("prelaunch_manifest_root_sha256") != prelaunch[
        "manifest_root_sha256"
    ]:
        _fail("ARCHIVE", "FORMAL_PRELAUNCH_ROOT_MISMATCH")
    prelaunch_rows = _verify_manifest_payload(
        prelaunch, "phase-d-prelaunch-manifest-v1"
    )
    records = {
        str(row["path"]): dict(row) for row in [*prelaunch_rows, *formal_rows]
    }
    if len(records) != len(prelaunch_rows) + len(formal_rows):
        _fail("ARCHIVE", "MANIFEST_PATH_OVERLAP")
    for relative, record in records.items():
        path = result_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record["size_bytes"]
            or sha256_path(path) != record["sha256"]
        ):
            _fail("ARCHIVE", "MANIFESTED_LIVE_FILE_MISMATCH")
    archive_path = result_root / ARCHIVE_BASENAME
    sidecar_path = result_root / ARCHIVE_SIDECAR_BASENAME
    if not archive_path.is_file() or archive_path.is_symlink() or not sidecar_path.is_file() or sidecar_path.is_symlink():
        _fail("ARCHIVE", "ARCHIVE_OR_SIDECAR_MISSING")
    archive_sha = sha256_path(archive_path)
    try:
        sidecar_text = sidecar_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ProtocolFailure("ARCHIVE", "SIDECAR_READ_FAILED") from exc
    if sidecar_text != f"{archive_sha}  {ARCHIVE_BASENAME}\n":
        _fail("ARCHIVE", "SIDECAR_HASH_MISMATCH")
    expected = set(records) | {"manifest.json", "formal/manifest.json"}
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != len({member.name for member in members}):
                _fail("ARCHIVE", "DUPLICATE_ARCHIVE_MEMBER")
            if any(not member.isfile() for member in members):
                _fail("ARCHIVE", "NONREGULAR_ARCHIVE_MEMBER")
            observed: set[str] = set()
            for member in members:
                prefix = f"{EXPERIMENT}/"
                if not member.name.startswith(prefix):
                    _fail("ARCHIVE", "ARCHIVE_PREFIX_MISMATCH")
                relative = member.name[len(prefix):]
                if not _safe_relative(relative):
                    _fail("ARCHIVE", "ARCHIVE_MEMBER_PATH_INVALID")
                observed.add(relative)
                extracted = archive.extractfile(member)
                if extracted is None:
                    _fail("ARCHIVE", "ARCHIVE_MEMBER_UNREADABLE")
                data = extracted.read()
                if relative in records:
                    record = records[relative]
                    if len(data) != record["size_bytes"] or sha256_bytes(data) != record["sha256"]:
                        _fail("ARCHIVE", "ARCHIVE_MEMBER_IDENTITY_MISMATCH")
                elif relative in {"manifest.json", "formal/manifest.json"}:
                    if data != (result_root / relative).read_bytes():
                        _fail("ARCHIVE", "ARCHIVE_MANIFEST_BYTES_MISMATCH")
            if observed != expected:
                _fail("ARCHIVE", "ARCHIVE_MEMBER_SET_MISMATCH")
    except ProtocolFailure:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ProtocolFailure("ARCHIVE", "ARCHIVE_VALIDATION_FAILED") from exc
    if any(
        relative in expected
        for relative in (ARCHIVE_BASENAME, ARCHIVE_SIDECAR_BASENAME, "completion.json")
    ):
        _fail("ARCHIVE", "ARCHIVE_SELF_INCLUSION_MISMATCH")
    _validate_terminal_consistency(result_root)
    return {
        "passed": True,
        "archive_sha256": archive_sha,
        "archive_member_count": len(expected),
        "unique_regular_members": True,
        "member_set_matches_manifests": True,
        "archive_sidecar_completion_excluded": True,
    }


def _write_completion(
    result_root: Path, terminal: Mapping[str, Any], archive_record: Mapping[str, Any]
) -> dict[str, Any]:
    record = {
        "schema_version": "phase-d-completion-envelope-v1",
        "non_authoritative_outer_envelope": True,
        "status": terminal["status"],
        "completion": terminal["completion"],
        "classification": terminal.get("classification"),
        "terminal_sha256": sha256_path(result_root / "formal/terminal.json"),
        "formal_manifest_sha256": sha256_path(result_root / "formal/manifest.json"),
        "formal_manifest_root_sha256": archive_record["formal_manifest_root_sha256"],
        "archive_sha256": archive_record["archive_sha256"],
        "archive_sidecar_sha256": sha256_path(result_root / ARCHIVE_SIDECAR_BASENAME),
        "archive_validated": True,
        "test_accessed": terminal.get("test_accessed"),
        "phase_e_started": False,
        "created_at": utc_now(),
    }
    _write_exclusive(result_root / "completion.json", json_asset(record))
    return record


def _available_shards(result_root: Path) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for worker_id in range(4):
        path = result_root / "formal/shards" / f"worker_{worker_id}.json"
        if not path.is_file() or path.is_symlink():
            continue
        try:
            value = _load_json_file(path, "FORMAL", "WORKER_SHARD_INVALID")
        except ProtocolFailure:
            continue
        if isinstance(value, dict):
            shards.append(value)
    return shards


def _validate_inference_plan(result_root: Path) -> dict[str, Any]:
    plan = _load_bound_record(
        result_root / "inference_plan.json", "INFERENCE", "INFERENCE_PLAN_INVALID"
    )
    protocol_bytes = (result_root / "frozen_protocol.json").read_bytes()
    digest = hashlib.sha256(protocol_bytes).digest()
    expected_seed = int.from_bytes(digest[:8], "big", signed=False)
    if (
        plan.get("schema_version") != "phase-d-inference-plan-v1"
        or plan.get("frozen_protocol_sha256") != digest.hex()
        or plan.get("bootstrap_seed") != expected_seed
        or plan.get("bootstrap_replicates") != 100000
        or plan.get("sign_flip_permutations") != 4096
        or plan.get("sign_flip_exhaustive") is not True
        or plan.get("sign_flip_plus_one_correction") is not False
    ):
        _fail("INFERENCE", "INFERENCE_PLAN_BINDING_MISMATCH")
    return plan


def formal_run(config_path: Path) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    result_root = REPO_ROOT / RESULT_REL
    manifest = verify_prelaunch(result_root)
    source_binding = verify_live_source_snapshot_binding(result_root, manifest)
    config = load_config(result_root / f"source_snapshot/{CONFIG_REL}")
    audit_dependencies(config)
    audit_phase_c_identity(config)
    assert_evaluator_ast_contract()
    assert_layernorm_source_contract()
    request, request_sha = _verify_launch_request(result_root, manifest)
    runner_enter = _verify_runner_enter(result_root, manifest, request, request_sha)
    task_freeze = _load_bound_record(
        result_root / "frozen_tasks.json", "FORMAL", "FROZEN_TASKS_INVALID"
    )
    assignments = _load_bound_record(
        result_root / "assignments.json", "FORMAL", "ASSIGNMENTS_INVALID"
    )
    task_freeze["assignments"] = assignments.get("assignments")
    validate_task_freeze(task_freeze, config)
    tasks = [dict(row) for row in task_freeze["tasks"]]
    inference_plan = _validate_inference_plan(result_root)
    gpu_admission = audit_gpu_admission(config, require_zero_compute_processes=True)
    formal_dir = result_root / "formal"
    start_record = {
        "schema_version": "phase-d-formal-start-v1",
        "state": "FORMAL_STARTED",
        "started_at": utc_now(),
        "run_id": request["run_id"],
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "launch_request_sha256": request_sha,
        "runner_entered_sha256": sha256_path(result_root / "launch/runner_entered.json"),
        "source_live_snapshot_binding": source_binding,
        "frozen_task_ledger_sha256": sha256_path(result_root / "frozen_tasks.json"),
        "frozen_assignment_ledger_sha256": sha256_path(result_root / "assignments.json"),
        "worker_mapping": config["workers"]["mapping"],
        "gpu_admission": gpu_admission,
        "planned_task_count": 1155,
        "test_accessed": False,
        "phase_e_started": False,
    }
    _create_formal_with_atomic_start(formal_dir, start_record)
    retry_counter_value = 0
    shards: list[dict[str, Any]] = []
    try:
        shards, retry_counter_value = _run_formal_workers(tasks, result_root, config)
        if len(shards) != 4 or any(
            shard.get("status") != "COMPLETE"
            or shard.get("worker_process_exit_failure") is True
            for shard in shards
        ):
            _fail("FORMAL", "WORKER_REPORTED_FAILURE")
        if retry_counter_value > 57:
            _fail("FORMAL", "RETRIED_TASK_58_LIMIT_EXCEEDED")
        attempts, results, test_ledger = _write_formal_ledgers(
            result_root, tasks, shards, retry_counter_value
        )
        if (
            len(results) != 1155
            or {str(row["task_id"]) for row in results}
            != {str(row["task_id"]) for row in tasks}
            or test_ledger["test_accessed"] is not False
            or test_ledger["forbidden_request_count"] != 0
            or test_ledger["test_access_status"] != "VERIFIED_NOT_ACCESSED"
        ):
            _fail("FORMAL", "FORMAL_RESULT_OR_FIREWALL_COMPLETENESS_MISMATCH")
        result_arm_counts = Counter(str(row["arm"]) for row in results)
        if result_arm_counts != Counter(
            {"Control": 576, "Intervention": 576, "P_anchor": 3}
        ):
            _fail("FORMAL", "FORMAL_ARM_COMPLETION_MISMATCH")
        statistics = compute_statistics(results, tasks, inference_plan)
        challengers = freeze_challengers(statistics)
        _atomic_replace_json(formal_dir / "statistics.json", statistics)
        _atomic_replace_json(formal_dir / "challengers.json", challengers)
        terminal = _write_success_terminal(
            result_root, statistics, challengers, test_ledger
        )
        response = {
            "status": "PASS",
            "stage": "FORMAL_COMPLETE",
            "run_id": request["run_id"],
            "classification": statistics["classification"],
            "planned_task_count": 1155,
            "completed_task_count": 1155,
            "failed_task_count": 0,
            "retry_task_count": len(
                {str(row["task_id"]) for row in attempts if int(row["attempt"]) > 1}
            ),
            "physical_gnn_attempt_count": len(attempts),
            "test_accessed": False,
            "phase_e_started": False,
            "runner_enter_state": runner_enter["state"],
        }
    except BaseException as exc:
        available = _available_shards(result_root)
        if available:
            shards = available
        terminal = _write_failure_terminal(
            result_root, tasks, shards, retry_counter_value, exc
        )
        response = {
            "status": "FAIL_CLOSED",
            "stage": terminal["failure_stage"],
            "error_code": terminal["failure_code"],
            "run_id": request["run_id"],
            "completion": "INCOMPLETE",
            "classification": None,
            "completed_task_count": terminal["completed_task_count"],
            "failed_task_count": terminal["failed_task_count"],
            "retry_task_count": terminal["retry_task_count"],
            "physical_gnn_attempt_count": terminal["physical_gnn_attempt_count"],
            "test_accessed": terminal["test_accessed"],
            "phase_e_started": False,
        }
    archive_record = finalize_formal_archive(result_root)
    _write_completion(result_root, terminal, archive_record)
    response["archive_sha256"] = archive_record["archive_sha256"]
    return response


def record_runner_exit(config_path: Path) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    result_root = REPO_ROOT / RESULT_REL
    manifest = verify_prelaunch(result_root)
    request, request_sha = _verify_launch_request(result_root, manifest)
    _verify_runner_enter(result_root, manifest, request, request_sha)
    terminal_path = result_root / "formal/terminal.json"
    if terminal_path.is_file() and not terminal_path.is_symlink():
        terminal = _validate_terminal_consistency(result_root)
        validation = validate_formal_archive(result_root)
        completion = _load_bound_record(
            result_root / "completion.json", "RUNNER_EXIT", "COMPLETION_ENVELOPE_INVALID"
        )
        if (
            completion.get("terminal_sha256") != sha256_path(terminal_path)
            or completion.get("archive_sha256") != validation["archive_sha256"]
            or completion.get("phase_e_started") is not False
        ):
            _fail("RUNNER_EXIT", "COMPLETION_ENVELOPE_BINDING_MISMATCH")
        return {
            "status": "PASS",
            "stage": "RUNNER_EXIT_TERMINAL_DURABLE",
            "run_id": request["run_id"],
            "terminal_status": terminal["status"],
            "archive_sha256": validation["archive_sha256"],
            "phase_e_started": False,
        }
    record = {
        "schema_version": "phase-d-runner-exit-v1",
        "status": "RUN_FAILED",
        "completion": "INCOMPLETE",
        "error_code": "FORMAL_TERMINAL_MISSING",
        "run_id": request["run_id"],
        "launch_request_sha256": request_sha,
        "recorded_at": utc_now(),
        "phase_e_started": False,
    }
    _write_exclusive(result_root / "launch/runner_exit.json", json_asset(record))
    return {"status": "FAIL_CLOSED", "stage": "RUNNER_EXIT", **record}


def synthetic_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("experiment") != EXPERIMENT:
        _fail("SYNTHETIC_SMOKE", "CONFIG_EXPERIMENT_MISMATCH")

    class FakeDecoder:
        def __call__(self, generator: Any, batch_size: int) -> list[str]:
            if int(generator.initial_seed()) != derive_stream_seed("04", "architecture-slots"):
                _fail("SYNTHETIC_SMOKE", "FAKE_GENERATOR_SEED_MISMATCH")
            return ["accepted"] + [f"unused-{index}" for index in range(batch_size - 1)]

    def converter(value: str) -> dict[str, Any]:
        return {
            "tag": value,
            "operations": ["GCNConv"],
            "edges": [[0, 1], [1, 2]],
            "effective_layers": 1,
        }

    def architecture_fp(value: Mapping[str, Any]) -> str:
        return sha256_bytes(str(value["tag"]).encode("utf-8"))

    def hp_fp(values: Sequence[float]) -> str:
        return sha256_bytes(canonical_json_bytes([float(value) for value in values]))

    def realize(value: Mapping[str, Any]) -> dict[str, Any]:
        raw = value["raw_hp_vector"]
        return {
            "lr": float(raw[0]),
            "dropout": float(raw[1]),
            "hidden_dim": 64,
            "l2": float(raw[3]),
            "gat_heads": 1,
            "sage_aggr": "mean",
            "gin_eps": 0.0,
            "gat_heads_by_layer": None,
            "sage_aggr_by_layer": None,
            "gin_eps_by_layer": None,
        }

    stream = consume_candidate_stream(
        FakeDecoder(),
        derive_stream_seed("04", "architecture-slots"),
        derive_stream_seed("04", "raw-hp-uniforms"),
        set(),
        set(),
        "04",
        slot_count=1,
        converter=converter,
        validator=lambda _value: (True, None),
        architecture_fingerprinter=architecture_fp,
        hp_fingerprinter=hp_fp,
        hp_realizer=realize,
    )
    firewall = synthetic_firewall_evidence()
    sign_flip = exact_sign_flip([0.01] * 12)
    bootstrap = bootstrap_lower_bound(
        [
            -0.004, -0.001, 0.0, 0.001, 0.002, 0.003,
            0.004, 0.006, 0.007, 0.009, 0.011, 0.014,
        ],
        seed=17143013755787270563,
    )
    if (
        len(stream["candidates"]) != 1
        or stream["audit"]["generated_draw_count"] != 256
        or stream["audit"]["consumed_draw_count"] != 1
        or sign_flip["permutation_count"] != 4096
        or not math.isfinite(bootstrap)
    ):
        _fail("SYNTHETIC_SMOKE", "SYNTHETIC_CONTRACT_MISMATCH")
    return {
        "passed": True,
        "status": "PASS",
        "stage": "SYNTHETIC_SMOKE_ZERO_GNN",
        "unsigned_seed_checks": {
            "replicate_04_architecture": derive_stream_seed("04", "architecture-slots"),
            "replicate_04_raw_hp": derive_stream_seed("04", "raw-hp-uniforms"),
        },
        "candidate_stream": stream["audit"],
        "firewall": firewall,
        "sign_flip_permutation_count": sign_flip["permutation_count"],
        "bootstrap_resample_count": 100000,
        "fake_inputs_only": True,
        "real_dataset_accessed": False,
        "real_checkpoint_accessed": False,
        "gnn_evaluations": 0,
        "cuda_initialized": False,
        "test_accessed": False,
        "result_root_written": False,
        "phase_e_started": False,
    }


def _safe_failure(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, ProtocolFailure):
        return {
            "status": "FAIL_CLOSED",
            "stage": exc.stage,
            "error_code": exc.error_code,
            "exception_class": type(exc).__name__,
            "gnn_start_authorized": False,
            "test_accessed": None,
            "phase_e_started": False,
        }
    return {
        "status": "FAIL_CLOSED",
        "stage": "TOP_LEVEL",
        "error_code": "UNCLASSIFIED_TOP_LEVEL_FAILURE",
        "exception_class": type(exc).__name__,
        "gnn_start_authorized": False,
        "test_accessed": None,
        "phase_e_started": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / CONFIG_REL)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--synthetic-smoke", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--record-launch-request", action="store_true")
    modes.add_argument("--runner-enter", action="store_true")
    modes.add_argument("--formal-run", action="store_true")
    modes.add_argument("--runner-exit", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.synthetic_smoke:
            output = synthetic_smoke(load_config(args.config))
        elif args.dry_run:
            output = dry_run(args.config)
        elif args.record_launch_request:
            output = record_launch_request(args.config)
        elif args.runner_enter:
            output = record_runner_enter(args.config)
        elif args.formal_run:
            output = formal_run(args.config)
        else:
            output = record_runner_exit(args.config)
    except BaseException as exc:
        print(json.dumps(_safe_failure(exc), sort_keys=True))
        return 2
    print(json.dumps(output, sort_keys=True))
    return 0 if output.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
