#!/usr/bin/env python3
"""Phase C v2 clean-room implementation and formal runner.

The module has three deliberately separate surfaces:

* synthetic projector/firewall checks, which never open historical inputs;
* a fail-closed zero-GNN dry-run, which performs the frozen whitelist-only
  projections and creates the immutable prelaunch result assets; and
* the formal validation-only worker runner bound to those assets.

Top-level real-corpus failures are reduced to a fixed stage, error code, and
exception class.  No historical outcome value or traceback is printed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import inspect
import io
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
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import uuid
import zlib
import base64
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
IMPLEMENTATION_REL = "analyse/fixed_layernorm_enhancement_v2.py"
CONFIG_REL = "configs/fixed_layernorm_enhancement_global4_v2.json"
TEST_REL = "tests/test_fixed_layernorm_enhancement_v2.py"
RUNNER_REL = "scripts/run_fixed_layernorm_enhancement_v2.sh"
RESULT_REL = "results/posthoc/fixed-layernorm-enhancement-global4-v2-recovery1"
TMUX_NAME = "fixed_layernorm_enhancement_v2"
TARGET_RELS = (IMPLEMENTATION_REL, CONFIG_REL, RUNNER_REL, TEST_REL)
FIXED_TMUX_COMMAND = (
    'tmux new-session -d -s fixed_layernorm_enhancement_v2 '
    '"cd /home/fangxy/D-VAE && conda run -n nas bash '
    'scripts/run_fixed_layernorm_enhancement_v2.sh; exec bash"'
)
CORRECTION_CONTROL_ID = "phase-c-v2-engineering-preflight-correction-ordinal-1-2026-08-30"
CORRECTION_CONTROL_SHA256 = "dbd6db86c857457df3258d92f65e4ff5929da5efdb1aa5d18e64ebf914ccc449"
EXPECTED_BEFORE_SHA256 = {
    IMPLEMENTATION_REL: "44e93cac18d60862ea86fa59858995fb7959ba0c75ede0962888fa982a748ffe",
    CONFIG_REL: "6da9193d437cc46b8fa0839466aed4f264d68d0792b0418a19f99db0cc2c8b22",
    RUNNER_REL: "d705e8ae14cd58b4a81687b0c8d707b75cea406d4da21dc526cceeabc4bb23b1",
    TEST_REL: "a32b47852b9a989e79ae2ac13b2537a1ed80c11bbde6902865d5bfc6b034858a",
}
CORRECTION_ORDINAL_2_CONTROL_ID = (
    "phase-c-v2-phase-a-manifest-binding-correction-ordinal-2-2026-08-30"
)
CORRECTION_ORDINAL_2_CONTROL_SHA256 = (
    "4d6990d47fbb55d0a7190d06706387239960478f9977f1b5c0726d780bebebdd"
)
CORRECTION_ORDINAL_3_CONTROL_ID = (
    "phase-c-v2-sealed-control-placement-correction-ordinal-3-2026-08-31"
)
CORRECTION_ORDINAL_3_CONTROL_SHA256 = (
    "15b5506ec5c164a362f0360ae007bd8bd5c5c28f6f8c0709fdb37ccbb95a4623"
)
CORRECTION_ORDINAL_4_CONTROL_ID = (
    "phase-c-v2-exposure-self-binding-identity-schema-adapter-correction-ordinal-4-2026-08-31"
)
CORRECTION_ORDINAL_4_CONTROL_SHA256 = (
    "724c81672736d603ef9c1e225c557961233a86b30c1b057823e87e565e7a8a34"
)
EXPECTED_ORDINAL_2_BEFORE_IDENTITIES = {
    IMPLEMENTATION_REL: {
        "size_bytes": 165911,
        "sha256": "81e9255845971a6e311d26ea48204543bc452f82a352c6014fb9ec0304159743",
    },
    CONFIG_REL: {
        "size_bytes": 24113,
        "sha256": "29adbe25bcdfaa9cb69b788f5c23a9c5aba0a0ecc5deecbb8caaf6e35ef35d3e",
    },
    RUNNER_REL: {
        "size_bytes": 1111,
        "sha256": "875bad60d0f3fc0a75153929ac216c659ae2d2e2190821cf074f84791a110ec4",
    },
    TEST_REL: {
        "size_bytes": 39357,
        "sha256": "2c0b4a2e1b47087fc83b897c688e4bb4edb0211f1f0d5deec14240c56efda770",
    },
}
EXPECTED_ORDINAL_3_BEFORE_IDENTITIES = {
    IMPLEMENTATION_REL: {
        "size_bytes": 185692,
        "sha256": "64d563a7be02c7b987985897a4f7ab7752d496b922205a6ad00f52a9efdb48e3",
    },
    CONFIG_REL: {
        "size_bytes": 30832,
        "sha256": "65f2f059dde8a41bf2ad293aaa590de8882d79b414ef162bfae4cc5ea72ffb56",
    },
    RUNNER_REL: {
        "size_bytes": 1111,
        "sha256": "875bad60d0f3fc0a75153929ac216c659ae2d2e2190821cf074f84791a110ec4",
    },
    TEST_REL: {
        "size_bytes": 54832,
        "sha256": "135eb5f4cef6b51eb0fc95146e58657de50dd11ee24ce5612424bb73da30297d",
    },
}
EXPECTED_ORDINAL_4_BEFORE_IDENTITIES = {
    IMPLEMENTATION_REL: {
        "size_bytes": 205302,
        "sha256": "f9bd5a9d02180dbadfaab046d0d1ce233daaf7367c33516d497a60255a59ece8",
    },
    CONFIG_REL: {
        "size_bytes": 37748,
        "sha256": "c28badd6e506db754b798e1c8399b63ffb1864b31424cb8d755224762f3c30a3",
    },
    RUNNER_REL: {
        "size_bytes": 1111,
        "sha256": "875bad60d0f3fc0a75153929ac216c659ae2d2e2190821cf074f84791a110ec4",
    },
    TEST_REL: {
        "size_bytes": 71998,
        "sha256": "33c65a87a6d9983b25d7e986a81bfd3df5f3e83de97ad0b587550825316e8c9f",
    },
}
ORDINAL_1_TERMINAL_CONTRACT = {
    "stage": "PHASE_A",
    "error_code": "PHASE_A_MANIFEST_RECORD_SCHEMA_MISMATCH",
    "event": "ENGINEERING_PHASE_A_MANIFEST_RECORD_SCHEMA_BINDING_BLOCK_ORDINAL_1",
    "scientific_decision": "NOT_REACHED",
    "classification": None,
    "phase_c_scientific_run_started": False,
    "disposition": "TERMINAL_NONRESUMABLE",
}
ORDINAL_2_TERMINAL_CONTRACT = {
    "stage": "PHASE_A",
    "error_code": "PHASE_A_REQUIRED_TOP_LEVEL_CONTROL_MISSING",
    "status": "FAIL_CLOSED",
    "event": "ENGINEERING_PHASE_A_CONTROL_TOP_LEVEL_SCHEMA_BINDING_BLOCK_ORDINAL_2_ZERO_GNN",
    "scientific_decision": "NOT_REACHED",
    "classification": None,
    "phase_c_scientific_run_started": False,
    "disposition": "TERMINAL_NONRESUMABLE",
}
ORDINAL_3_TERMINAL_CONTRACT = {
    "event": "ENGINEERING_EXPOSURE_REGISTRY_FILE_COUNT_BINDING_BLOCK_ORDINAL_3_ZERO_GNN",
    "stage": "EXPOSURE_REGISTRY",
    "error_code": "EXPOSURE_FILE_COUNT_MISMATCH",
    "status": "FAIL_CLOSED",
    "scientific_decision": "NOT_REACHED",
    "classification": None,
    "phase_c_scientific_run_started": False,
    "disposition": "TERMINAL_NONRESUMABLE",
}
ORDINAL_3_AFTER_LEDGER_IDENTITY = {
    "path": "/tmp/phase_c_v2_ordinal3_after_identities.json",
    "size_bytes": 870,
    "sha256": "6b9ee3d0c72765493b7517980d892c5c1357afd4b57b793dbc9ee89400273ae3",
}
ORDINAL_3_PROMPT_SHA256 = (
    "ef27f763955e012cce5520f4dd9a8fc5f0fa6ffc31819a1fbc18e5a07d386b1b"
)

CSV_JSON_IDENTITY_CELL_V1 = "CSV_JSON_IDENTITY_CELL_V1"
JSON_IDENTITY_DEFAULT_V1 = "JSON_IDENTITY_DEFAULT_V1"
JSON_EXACT_ISOLATED_EDGES_COUNT_COLLISION_V1 = (
    "JSON_EXACT_ISOLATED_EDGES_COUNT_COLLISION_V1"
)
OPAQUE_EXACT_IDENTITY_TOKEN_V1 = "OPAQUE_EXACT_IDENTITY_TOKEN_V1"
EXPOSURE_ADAPTER_IDS = (
    CSV_JSON_IDENTITY_CELL_V1,
    JSON_IDENTITY_DEFAULT_V1,
    JSON_EXACT_ISOLATED_EDGES_COUNT_COLLISION_V1,
    OPAQUE_EXACT_IDENTITY_TOKEN_V1,
)


def _structured_contract(
    pointer_pattern: str, expected_count: int, required_identity_keys: Sequence[str]
) -> dict[str, Any]:
    return {
        "pointer_pattern": pointer_pattern,
        "expected_count": expected_count,
        "required_identity_keys": sorted(required_identity_keys),
    }


def _typed_pointer_contract(
    pointer_pattern: str, expected_count: int, expected_json_type: str
) -> dict[str, Any]:
    return {
        "pointer_pattern": pointer_pattern,
        "expected_count": expected_count,
        "expected_json_type": expected_json_type,
    }


EXACT_HISTORICAL_JSON_ADAPTER_CONTRACT = {
    "schema_version": "phase-c-v2-historical-json-identity-adapter-contract-v1",
    "files": [
        {
            "path": "results/posthoc/cora-layernorm-crossover-v1/crossover_manifest.json",
            "size": 16643,
            "sha256": "9f02f3daf0381f9ece4375de2bc54ab8cdbf3137dc569f3997438cc0eae33e79",
            "structured_record_contracts": sorted(
                [
                    _structured_contract(
                        "/new_combinations/*",
                        2,
                        ["architecture_fingerprint", "operations", "edges"],
                    ),
                    _structured_contract(
                        "/structure_comparison/diverse_truncation_stress_1",
                        1,
                        ["operations", "edges"],
                    ),
                    _structured_contract(
                        "/structure_comparison/median_1",
                        1,
                        ["operations", "edges"],
                    ),
                    _structured_contract(
                        "/structure_comparison/top_1",
                        1,
                        ["operations", "edges"],
                    ),
                ],
                key=lambda row: row["pointer_pattern"],
            ),
            "fingerprint_only_contracts": [],
            "skipped_non_architecture_identity_contracts": [
                _typed_pointer_contract("/dataset_identity/edges", 1, "integer")
            ],
            "skipped_redundant_identity_contracts": [],
            "expected_structured_record_count": 5,
            "expected_fingerprint_only_record_count": 0,
            "expected_unique_architecture_count": 3,
            "identity_set_sha256": "6b9ed62eb36df6091c4195fc7763bff1dfec9b633f6de147d9b88682fff26d97",
        },
        {
            "path": "results/posthoc/cora-layernorm-high-region-v1/high_region_manifest.json",
            "size": 22583,
            "sha256": "c7d94521f27445285cccb90fe510e8a6442c493cf2f76ceeb04c8a21f38b3466",
            "structured_record_contracts": [
                _structured_contract(
                    "/selected_candidates/*",
                    5,
                    ["architecture_fingerprint", "operations", "edges"],
                )
            ],
            "fingerprint_only_contracts": [],
            "skipped_non_architecture_identity_contracts": [
                _typed_pointer_contract("/dataset_identity/edges", 1, "integer")
            ],
            "skipped_redundant_identity_contracts": [
                _typed_pointer_contract(
                    "/selected_candidates/*/architecture_summary/operations",
                    5,
                    "array",
                )
            ],
            "expected_structured_record_count": 5,
            "expected_fingerprint_only_record_count": 0,
            "expected_unique_architecture_count": 5,
            "identity_set_sha256": "ca4b56485fe165f43315d52f826b5d1080fb27058b8bbcd68cc80571d245348b",
        },
        {
            "path": "results/posthoc/cora-layernorm-subgroup-v1/subgroup_manifest.json",
            "size": 25861,
            "sha256": "fd008b5d8d9308fdad5a14c6a6b7639a3be24f1e0730bb683210a0e9f5e5a660",
            "structured_record_contracts": [
                _structured_contract(
                    "/selected_candidates/*",
                    6,
                    ["architecture_fingerprint", "operations", "edges"],
                )
            ],
            "fingerprint_only_contracts": [],
            "skipped_non_architecture_identity_contracts": [
                _typed_pointer_contract("/dataset_identity/edges", 1, "integer")
            ],
            "skipped_redundant_identity_contracts": [],
            "expected_structured_record_count": 6,
            "expected_fingerprint_only_record_count": 0,
            "expected_unique_architecture_count": 6,
            "identity_set_sha256": "ddf7176a354bf27ff1f9a03fed9de556ff0f712b71fbb3c92e876ac5ac214dcc",
        },
        {
            "path": "results/posthoc/cora-layernorm-v1/layernorm_manifest.json",
            "size": 16467,
            "sha256": "52ce8241e5f0c800971c4190dcaf9d77d3fb1ee824598f61aeacf51cf4c25319",
            "structured_record_contracts": [
                _structured_contract(
                    "/candidate_panel/*",
                    3,
                    ["architecture_fingerprint", "operations", "edges"],
                )
            ],
            "fingerprint_only_contracts": [
                _typed_pointer_contract(
                    "/baseline_provenance/baseline_rows/*/architecture_fingerprint",
                    9,
                    "string",
                )
            ],
            "skipped_non_architecture_identity_contracts": [
                _typed_pointer_contract("/dataset_identity/edges", 1, "integer")
            ],
            "skipped_redundant_identity_contracts": [],
            "expected_structured_record_count": 3,
            "expected_fingerprint_only_record_count": 9,
            "expected_unique_architecture_count": 3,
            "identity_set_sha256": "6b9ed62eb36df6091c4195fc7763bff1dfec9b633f6de147d9b88682fff26d97",
        },
    ],
}

CSV_IDENTITY_HEADER_GROUPS = {
    "0180b68837cd693d9d64c84fa849d166e4a67267a121304d570655bae3ad3ddd": {
        "columns": 43,
        "files": 16,
        "rows": 272,
        "identity_columns": [],
    },
    "3eccd7a52ff966bbb45f5bd74c8a9b314a4c5702f2e7fec85523f7e5e25de60e": {
        "columns": 86,
        "files": 16,
        "rows": 2400,
        "identity_columns": ["architecture_fingerprint", "hp_fingerprint"],
    },
    "99e1c8f3a06c74682b11a839763ba0b6c03196e2afb107f8de430045273aa272": {
        "columns": 44,
        "files": 47,
        "rows": 235,
        "identity_columns": ["edges", "operations"],
    },
    "c00a2993946df064fbda624ec1f4b18588694c3e3306b8eff47c4abdc2d3510c": {
        "columns": 44,
        "files": 23,
        "rows": 23,
        "identity_columns": ["edges", "operations"],
    },
}

EXPOSURE_ADAPTER_PLAN = {
    "marker_filtered_prewrite_file_count": 607,
    "marker_filtered_prewrite_tree_sha256": (
        "0316ef48a9b48297074bdc39b93088d2781057af7ae36a88607a7559dc738ea0"
    ),
    "marker_filtered_current_file_count": 607,
    "execution_owned_exact_exclusions": [CONFIG_REL],
    "historical_file_count": 606,
    "historical_tree_sha256": (
        "40a9fc01007a06dc702b6413d6fdf23d4258f5237f3bbff7421b59b2e0f7dec7"
    ),
    "historical_root_counts": {"analyse": 5, "configs": 3, "results": 593, "tests": 5},
    "adapter_class_counts": {
        CSV_JSON_IDENTITY_CELL_V1: 102,
        JSON_IDENTITY_DEFAULT_V1: 490,
        JSON_EXACT_ISOLATED_EDGES_COUNT_COLLISION_V1: 4,
        OPAQUE_EXACT_IDENTITY_TOKEN_V1: 10,
    },
    "adapter_manifest_root_sha256": (
        "1ba2d2d7b2d6ba8dfa1bc9e2a81e5487674608f413739b57c981861cfba591db"
    ),
    "legacy_error_count": 74,
    "legacy_error_class_counts": {
        "CSV_SERIALIZED_IDENTITY_ARRAY_NOT_DECODED": 70,
        "JSON_ISOLATED_NONIDENTITY_EDGES_COLLISION": 4,
    },
    "legacy_error_set_root_sha256": (
        "99b6ea7cb1ba028eaf7b3562b4ea990ad8c1652cbb9fe1e2fe51437b5dd98566"
    ),
    "legacy_success_count": 532,
    "legacy_success_projection_root_sha256": (
        "4b6168d6c41eb4315e82cad5345d8701fa297ae527d9ce0c38e79dfb9b4d582e"
    ),
    "new_projection_file_count": 606,
    "new_projection_root_sha256": (
        "f00d5c3ddb07b2d30dfcad15db1314ba9ea8c82af4084b2eeff5c5d5029c0aad"
    ),
    "csv_schema_plan_root_sha256": (
        "19e93f7ac69632b1bb73fea5030de98aedb488bf58cdb5551026586cb9e1ee3f"
    ),
    "csv_legacy_error_path_set_sha256": (
        "67eb18d84314d2193b78183365cd8f84a8be0d7a9097143631044b62e90c26f1"
    ),
    "csv_legacy_error_inventory_sha256": (
        "8ce306aa1f3b4f61c4bda3b459779235ab18281f9237ce70bbfc85892aed45b3"
    ),
    "exact_json_contract_sha256": (
        "50c21a181acb7fc31acc55ca0dde0432c7a08b5884033676c2da171c36abc817"
    ),
    "exposed_architecture_count": 1253,
    "exposed_architecture_set_sha256": (
        "2c4d67b50cbc9458747dff05ea1ed2efddbef49471990075b48d85fe34979f6d"
    ),
}
EXPOSURE_ADAPTER_CONFIG_CONTRACT = {
    "canonical_commitment_schema": "canonical-json-no-newline-v1",
    "routing_priority": [
        JSON_EXACT_ISOLATED_EDGES_COUNT_COLLISION_V1,
        CSV_JSON_IDENTITY_CELL_V1,
        JSON_IDENTITY_DEFAULT_V1,
        OPAQUE_EXACT_IDENTITY_TOKEN_V1,
    ],
    "execution_owned_exact_exclusions": [CONFIG_REL],
    "commitments": EXPOSURE_ADAPTER_PLAN,
    "csv_header_groups": CSV_IDENTITY_HEADER_GROUPS,
    "exact_json_adapter_contract": EXACT_HISTORICAL_JSON_ADAPTER_CONTRACT,
}
PHASE_A_MANIFEST_IDENTITY_CONTRACT = {
    "path": "results/posthoc/boundary-training-ceiling-audit-global4-v1/manifest.json",
    "size_bytes": 5381,
    "sha256": "45f4d33361a501bdbd9a21dd05ba0f018d14ac9da365b6a853fe404a52ba2270",
}
PHASE_A_MANIFEST_SCHEMA_CONTRACT = {
    "root_keys": [
        "archive_excluded_from_manifest",
        "archive_self_inclusion",
        "created_at",
        "experiment",
        "file_count",
        "files",
        "manifest_root_sha256",
    ],
    "safe_root_controls": {
        "archive_excluded_from_manifest": True,
        "archive_self_inclusion": False,
        "experiment": "boundary-training-ceiling-audit-global4-v1",
        "file_count": 29,
        "manifest_root_sha256": (
            "23e486d1aafb7101c5f236ca504045d9be29b827028354ea3fce3734a2dec8f4"
        ),
    },
    "record_keys": ["relative_path", "sha256", "size_bytes"],
    "relative_paths": [
        "attempt_ledger.json",
        "corpus_audit.json",
        "decision.json",
        "environment.json",
        "final_report.md",
        "frozen_protocol.json",
        "git_after.txt",
        "git_before.txt",
        "high_regime_secondary_analysis.json",
        "input_hashes.json",
        "interaction_analysis.json",
        "logs/execution.log",
        "matched_analysis.json",
        "matched_pairs.csv",
        "pre_execution_estimate.json",
        "primary_configurations.csv",
        "q90_analysis.csv",
        "q90_analysis.json",
        "seed_collision_audit.json",
        "source_snapshot/analyse/boundary_training_ceiling_audit.py",
        "source_snapshot/configs/boundary_training_ceiling_audit_global4_v1.json",
        "source_snapshot/scripts/run_boundary_training_ceiling_audit.sh",
        "source_snapshot/tests/test_boundary_training_ceiling_audit.py",
        "state.json",
        "test_firewall.json",
        "top_tail_analysis.csv",
        "top_tail_analysis.json",
        "training_audit.json",
        "training_seed_runs.csv",
    ],
    "safe_projection_size_bytes": 4829,
    "safe_projection_sha256": (
        "5a272edee3457d1f5d72052022c5dcea946a8a05a2dd1f7ea329280c98b8d022"
    ),
}
PHASE_A_CONTROL_PROJECTION_SCHEMA = (
    "exact-sealed-phase-a-safe-top-level-native-types-v3"
)
PHASE_A_CONTROL_CONTRACT = {
    "control_projection_schema": PHASE_A_CONTROL_PROJECTION_SCHEMA,
    "files": {
        "decision.json": {
            "size_bytes": 3923,
            "sha256": "72d5ee8862d88e5201c772871526137d5727985bbf1c3b880711fcfa8fd77f47",
            "safe_fields": {
                "classification": {
                    "native_type": "str",
                    "value": "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL",
                },
                "boundary": {"native_type": "bool", "value": False},
                "training": {"native_type": "bool", "value": False},
                "training_sufficient": {"native_type": "bool", "value": True},
                "new_gnn_runs": {"native_type": "int", "value": 0},
                "gpu_compute_runs": {"native_type": "int", "value": 0},
                "test_accessed": {"native_type": "bool", "value": False},
                "phase_b_started": {"native_type": "bool", "value": False},
            },
            "conflicting_aliases": [
                "boundary_signal",
                "decision",
                "training_data_sufficient",
                "training_signal",
            ],
            "safe_projection_size_bytes": 204,
            "safe_projection_sha256": (
                "9a3c3f352d58f63fea49d0a33b2ec8d09c13332a57c756c1d236ae65af6cc681"
            ),
        },
        "state.json": {
            "size_bytes": 607,
            "sha256": "2becfe6125bc4763443b0af0ae436a3d954de8ee96c4d1caa691dabdfdca8b1d",
            "safe_fields": {
                "status": {"native_type": "str", "value": "COMPLETE"},
                "classification": {
                    "native_type": "str",
                    "value": "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL",
                },
                "gpu_compute_used": {"native_type": "bool", "value": False},
                "phase_b_started": {"native_type": "bool", "value": False},
                "test_accessed": {"native_type": "bool", "value": False},
            },
            "conflicting_aliases": [],
            "safe_projection_size_bytes": 150,
            "safe_projection_sha256": (
                "bd5b74c9cbd08d31741609808b32585f904cb6f967ce542e2c679a82a55b1cde"
            ),
        },
        "training_audit.json": {
            "size_bytes": 5123,
            "sha256": "3127bbd89669950ddb848029726224ee56035caede3503e9611b34f24100040e",
            "safe_fields": {
                "status": {
                    "native_type": "str",
                    "value": "TRAINING_AUDIT_SUFFICIENT",
                },
                "sufficient": {"native_type": "bool", "value": True},
                "raw_training_truncation_signal": {
                    "native_type": "bool",
                    "value": False,
                },
                "training_truncation_signal": {
                    "native_type": "bool",
                    "value": False,
                },
            },
            "conflicting_aliases": ["gnn_evaluations", "training_signal"],
            "safe_projection_size_bytes": 131,
            "safe_projection_sha256": (
                "7a2a9753eb370eff7ea5b2885b2630cd11022682abe35b338366b532fb589090"
            ),
        },
        "test_firewall.json": {
            "size_bytes": 547,
            "sha256": "bf0c7401a45b66cc3a61f6eb88389ebb8ab1fc4d78cbf263c4d07a634566089f",
            "safe_fields": {
                "passed": {"native_type": "bool", "value": True},
                "gnn_training_calls": {"native_type": "int", "value": 0},
                "gpu_compute_calls": {"native_type": "int", "value": 0},
                "test_data_reads": {"native_type": "int", "value": 0},
                "runner_exact_command_verified": {
                    "native_type": "bool",
                    "value": True,
                },
            },
            "conflicting_aliases": [
                "test_accessed",
                "test_outcome_accessed",
                "test_values_accessed",
            ],
            "safe_projection_size_bytes": 118,
            "safe_projection_sha256": (
                "9709e36bf70ba63592f72f0fadeff529ab1918c4dd636d0a3184a2d164a32f3a"
            ),
        },
    },
    "aggregate_safe_projection_size_bytes": 677,
    "aggregate_safe_projection_sha256": (
        "6c54598c7c01d63dd0ae88a3c52e085706190670852c2403ea012e4e29c0b833"
    ),
}
BASE_P_CONTRACT = {
    "joint_fingerprint": "30e0bb108c1fdbe8db186152a09c3f2fc6e812be189db920fe4d0d2b0d3cb1a2",
    "architecture_fingerprint": "960efb3b97f00491c3956657cfd57958effb6096f65b1db2fa8aaae586b82207",
    "hp_fingerprint": "911259943106a1edad300bc5300a3883ece8cbffba48b95e6c61b6e5eb4b679d",
    "operations": ["GCNConv", "GATConv", "GCNII", "GCNII", "SAGEConv"],
    "edges": [[0, 1], [1, 2], [0, 2], [2, 3], [3, 4], [4, 5], [3, 5], [0, 5], [5, 6]],
    "lr": 0.0219530948,
    "dropout": 0.55173624,
    "hidden": 512,
    "l2": 0.001,
    "max_epochs": 300,
    "patience": 80,
}

IDENTITY_WHITELIST = frozenset(
    {
        "architecture_fingerprint",
        "joint_config_fingerprint",
        "joint_fingerprint",
        "hp_fingerprint",
        "operations",
        "ops",
        "edges",
        "config_id",
    }
)
DATA_ALLOWLIST = frozenset({"x", "edge_index", "train_mask", "val_mask", "y"})
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
JSON_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
ARCHITECTURE_TOKEN_RE = re.compile(
    rb"(?i)(?:[\"']?architecture_fingerprint[\"']?)"
    rb"[ \t\r\n]*[:=][ \t\r\n]*[\"']([0-9a-f]{64})[\"']"
)


class ProtocolFailure(RuntimeError):
    """A fail-closed protocol error whose details are safe fixed codes."""

    def __init__(self, stage: str, error_code: str):
        self.stage = str(stage)
        self.error_code = str(error_code)
        super().__init__(self.error_code)


class HistoricalProjectionError(ProtocolFailure):
    """Safe fixed-code failure raised by whitelist-only historical projection."""


class TestAccessViolation(RuntimeError):
    """Raised after a validation-only proxy ledgers a forbidden request."""


def _fail(stage: str, error_code: str) -> None:
    raise ProtocolFailure(stage, error_code)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_commitment_json_bytes_v1(value: Any) -> bytes:
    """Canonical ordinal-4 commitment encoding; inherited hashes remain untouched."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def ordinal_4_commitment_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_commitment_json_bytes_v1(value)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        _fail("INPUT_IDENTITY", "SOURCE_CHANGED_DURING_HASH")
    return digest.hexdigest()


def assert_file_identity(path: Path, expected_sha256: str, expected_size: int | None = None) -> None:
    if not path.is_file() or path.is_symlink():
        _fail("INPUT_IDENTITY", "REQUIRED_REGULAR_FILE_MISSING")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        _fail("INPUT_IDENTITY", "REQUIRED_FILE_SIZE_MISMATCH")
    if sha256_path(path) != expected_sha256:
        _fail("INPUT_IDENTITY", "REQUIRED_FILE_SHA256_MISMATCH")


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate_json_key")
        output[key] = value
    return output


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolFailure("CONFIG", "CONFIG_LOAD_FAILED") from exc
    if config.get("experiment") != "fixed-layernorm-enhancement-global4-v2":
        _fail("CONFIG", "EXPERIMENT_ID_MISMATCH")
    compressed = config.get("scientific_amendment_zlib_base64")
    if not isinstance(compressed, str):
        _fail("CONFIG", "AMENDMENT_BYTES_MISSING")
    try:
        amendment = zlib.decompress(base64.b64decode(compressed, validate=True))
    except (ValueError, zlib.error) as exc:
        raise ProtocolFailure("CONFIG", "AMENDMENT_DECODE_FAILED") from exc
    if sha256_bytes(amendment) != config["scientific_amendment_sha256"]:
        _fail("CONFIG", "AMENDMENT_SHA256_MISMATCH")
    if sha256_bytes(amendment) != "1d487f8bd5ddebc4a6b964f77975e3fc1931496ce4c800658b50cfa8640f4279":
        _fail("CONFIG", "AMENDMENT_DECLARATION_MISMATCH")
    execution_control = config.get("execution_control")
    if (
        not isinstance(execution_control, dict)
        or execution_control.get("control_id")
        != "phase-c-v2-cleanroom-execution-control-2026-08-30"
        or execution_control.get("sha256")
        != "bac75381c9d3315529cac88d966e739cf41d7cfba1f98c66cfc2534fdb764237"
    ):
        _fail("CONFIG", "CLEANROOM_CONTROL_PROVENANCE_MISMATCH")
    correction = config.get("engineering_correction")
    if not isinstance(correction, dict):
        _fail("CONFIG", "ENGINEERING_CORRECTION_PROVENANCE_MISSING")
    if correction.get("control_id") != CORRECTION_CONTROL_ID:
        _fail("CONFIG", "ENGINEERING_CORRECTION_CONTROL_ID_MISMATCH")
    if correction.get("sha256") != CORRECTION_CONTROL_SHA256:
        _fail("CONFIG", "ENGINEERING_CORRECTION_CONTROL_SHA256_MISMATCH")
    if correction.get("before_sha256") != EXPECTED_BEFORE_SHA256:
        _fail("CONFIG", "ENGINEERING_CORRECTION_BEFORE_SHA256_MISMATCH")
    zero_states = correction.get("zero_state_blocks")
    if not isinstance(zero_states, list) or len(zero_states) != 4:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ZERO_STATE_PROVENANCE_MISMATCH")
    classifications = [row.get("classification") for row in zero_states if isinstance(row, dict)]
    if classifications != [
        "PHASE_C_V1_PROTOCOL_INFEASIBLE_PREEXECUTION",
        "PHASE_C_V2_PREEXECUTION_IDENTITY_ONLY_ACCESS_VIOLATION_ZERO_STATE",
        "ENGINEERING_ORCHESTRATION_PREFLIGHT_BLOCK",
        "ENGINEERING_PHASE_A_SCHEMA_BINDING_BLOCK",
    ]:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ZERO_STATE_PROVENANCE_MISMATCH")
    ordinal_2 = config.get("engineering_correction_ordinal_2")
    if not isinstance(ordinal_2, dict):
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_2_PROVENANCE_MISSING")
    if ordinal_2.get("control_id") != CORRECTION_ORDINAL_2_CONTROL_ID:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_2_CONTROL_ID_MISMATCH")
    if ordinal_2.get("sha256") != CORRECTION_ORDINAL_2_CONTROL_SHA256:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_2_CONTROL_SHA256_MISMATCH")
    if ordinal_2.get("before_file_identities") != EXPECTED_ORDINAL_2_BEFORE_IDENTITIES:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_2_BEFORE_IDENTITY_MISMATCH")
    ordinal_1_terminal = ordinal_2.get("ordinal_1_terminal")
    if not isinstance(ordinal_1_terminal, dict) or ordinal_1_terminal != (
        ORDINAL_1_TERMINAL_CONTRACT
    ):
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_1_TERMINAL_MISMATCH")
    if ordinal_2.get("phase_a_manifest_identity") != PHASE_A_MANIFEST_IDENTITY_CONTRACT:
        _fail("CONFIG", "PHASE_A_MANIFEST_IDENTITY_CONTRACT_MISMATCH")
    if ordinal_2.get("phase_a_manifest_schema") != PHASE_A_MANIFEST_SCHEMA_CONTRACT:
        _fail("CONFIG", "PHASE_A_MANIFEST_SCHEMA_CONTRACT_MISMATCH")
    phase_a = config.get("phase_a")
    if (
        not isinstance(phase_a, dict)
        or phase_a.get("manifest_identity") != PHASE_A_MANIFEST_IDENTITY_CONTRACT
        or phase_a.get("manifest_schema") != PHASE_A_MANIFEST_SCHEMA_CONTRACT
    ):
        _fail("CONFIG", "PHASE_A_MANIFEST_BINDING_CONFIG_MISMATCH")
    zero_state = ordinal_2.get("zero_state_attestation")
    if (
        not isinstance(zero_state, dict)
        or zero_state.get("gnn_evaluations") != 0
        or zero_state.get("gpu_training_seconds") != 0
        or zero_state.get("fresh_seed_consumption") != 0
        or zero_state.get("phase_b_started") is not False
        or zero_state.get("phase_d_started") is not False
        or zero_state.get("phase_e_started") is not False
        or zero_state.get("result_path_present") is not False
        or zero_state.get("tmux_present") is not False
        or zero_state.get("gpu_compute_process_count") != 0
    ):
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_2_ZERO_STATE_MISMATCH")
    ordinal_3 = config.get("engineering_correction_ordinal_3")
    if not isinstance(ordinal_3, dict):
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_3_PROVENANCE_MISSING")
    if ordinal_3.get("control_id") != CORRECTION_ORDINAL_3_CONTROL_ID:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_3_CONTROL_ID_MISMATCH")
    if ordinal_3.get("sha256") != CORRECTION_ORDINAL_3_CONTROL_SHA256:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_3_CONTROL_SHA256_MISMATCH")
    if (
        ordinal_3.get("correction_class")
        != "ENGINEERING_CONTROL_SCHEMA_BINDING_CORRECTION"
        or ordinal_3.get("ordinal") != 3
        or ordinal_3.get("scientific_amendment_sha256")
        != "1d487f8bd5ddebc4a6b964f77975e3fc1931496ce4c800658b50cfa8640f4279"
        or ordinal_3.get("clean_room_control_sha256")
        != "bac75381c9d3315529cac88d966e739cf41d7cfba1f98c66cfc2534fdb764237"
        or ordinal_3.get("ordinal_1_control_sha256") != CORRECTION_CONTROL_SHA256
        or ordinal_3.get("ordinal_2_control_sha256")
        != CORRECTION_ORDINAL_2_CONTROL_SHA256
    ):
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_3_SCOPE_MISMATCH")
    if ordinal_3.get("before_file_identities") != EXPECTED_ORDINAL_3_BEFORE_IDENTITIES:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_3_BEFORE_IDENTITY_MISMATCH")
    if ordinal_3.get("previous_engineering_terminals") != {
        "ordinal_1": ORDINAL_1_TERMINAL_CONTRACT,
        "ordinal_2": ORDINAL_2_TERMINAL_CONTRACT,
    }:
        _fail("CONFIG", "ENGINEERING_CORRECTION_PREVIOUS_TERMINALS_MISMATCH")
    if ordinal_3.get("phase_a_control_contract") != PHASE_A_CONTROL_CONTRACT:
        _fail("CONFIG", "PHASE_A_CONTROL_CONTRACT_MISMATCH")
    ordinal_3_zero = ordinal_3.get("zero_state_attestation")
    if (
        not isinstance(ordinal_3_zero, dict)
        or ordinal_3_zero.get("dirty_status_entry_count") != 140
        or ordinal_3_zero.get("dirty_status_porcelain_z_sha256")
        != "b272a65a343cc40ba4ea0ecacd617068891c1cfae22506faf8e7ca04bcc9341b"
        or ordinal_3_zero.get("unrelated_status_universe_entry_count") != 136
        or ordinal_3_zero.get("unrelated_status_content_merkle_sha256")
        != "8879782d37fe55b01b05c6f43ea8a15b8624cbb209b2418c1d5939324b591906"
        or ordinal_3_zero.get("ordinal_2_real_dry_run_count") != 1
        or ordinal_3_zero.get("ordinal_2_real_dry_run_exhausted") is not True
        or ordinal_3_zero.get("result_path_present") is not False
        or ordinal_3_zero.get("launch_request_present") is not False
        or ordinal_3_zero.get("formal_state_present") is not False
        or ordinal_3_zero.get("tmux_present") is not False
        or ordinal_3_zero.get("named_process_count") != 0
        or ordinal_3_zero.get("gpu_compute_process_count") != 0
        or ordinal_3_zero.get("gnn_evaluations") != 0
        or ordinal_3_zero.get("gpu_training_seconds") != 0
        or ordinal_3_zero.get("fresh_seed_consumption") != 0
        or ordinal_3_zero.get("test_accessed") is not False
        or ordinal_3_zero.get("phase_b_started") is not False
        or ordinal_3_zero.get("phase_d_started") is not False
        or ordinal_3_zero.get("phase_e_started") is not False
    ):
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_3_ZERO_STATE_MISMATCH")
    if ordinal_3.get("scientific_design_changed") is not False:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_3_SCOPE_MISMATCH")
    ordinal_4 = config.get("engineering_correction_ordinal_4")
    if not isinstance(ordinal_4, dict):
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_4_PROVENANCE_MISSING")
    if (
        ordinal_4.get("control_id") != CORRECTION_ORDINAL_4_CONTROL_ID
        or ordinal_4.get("sha256") != CORRECTION_ORDINAL_4_CONTROL_SHA256
        or ordinal_4.get("correction_class")
        != "ENGINEERING_EXPOSURE_SELF_BINDING_AND_IDENTITY_SCHEMA_ADAPTER_CORRECTION"
        or ordinal_4.get("ordinal") != 4
        or ordinal_4.get("scientific_amendment_sha256")
        != "1d487f8bd5ddebc4a6b964f77975e3fc1931496ce4c800658b50cfa8640f4279"
        or ordinal_4.get("clean_room_control_sha256")
        != "bac75381c9d3315529cac88d966e739cf41d7cfba1f98c66cfc2534fdb764237"
        or ordinal_4.get("ordinal_1_control_sha256") != CORRECTION_CONTROL_SHA256
        or ordinal_4.get("ordinal_2_control_sha256")
        != CORRECTION_ORDINAL_2_CONTROL_SHA256
        or ordinal_4.get("ordinal_3_control_sha256")
        != CORRECTION_ORDINAL_3_CONTROL_SHA256
    ):
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_4_SCOPE_MISMATCH")
    if ordinal_4.get("before_file_identities") != EXPECTED_ORDINAL_4_BEFORE_IDENTITIES:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_4_BEFORE_IDENTITY_MISMATCH")
    if ordinal_4.get("ordinal_3_terminal") != ORDINAL_3_TERMINAL_CONTRACT:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_3_TERMINAL_MISMATCH")
    if ordinal_4.get("ordinal_3_after_ledger_identity") != ORDINAL_3_AFTER_LEDGER_IDENTITY:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_3_AFTER_LEDGER_MISMATCH")
    if ordinal_4.get("ordinal_3_prompt_sha256") != ORDINAL_3_PROMPT_SHA256:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_3_PROMPT_MISMATCH")
    if ordinal_4.get("adapter_plan_commitments") != EXPOSURE_ADAPTER_PLAN:
        _fail("CONFIG", "EXPOSURE_ADAPTER_COMMITMENT_CONFIG_MISMATCH")
    ordinal_4_zero = ordinal_4.get("zero_state_attestation")
    if (
        not isinstance(ordinal_4_zero, dict)
        or ordinal_4_zero.get("dirty_status_entry_count") != 140
        or ordinal_4_zero.get("dirty_status_porcelain_z_sha256")
        != "b272a65a343cc40ba4ea0ecacd617068891c1cfae22506faf8e7ca04bcc9341b"
        or ordinal_4_zero.get("unrelated_status_universe_entry_count") != 136
        or ordinal_4_zero.get("unrelated_status_content_merkle_sha256")
        != "8879782d37fe55b01b05c6f43ea8a15b8624cbb209b2418c1d5939324b591906"
        or ordinal_4_zero.get("ordinal_3_real_dry_run_count") != 1
        or ordinal_4_zero.get("ordinal_3_real_dry_run_exhausted") is not True
        or ordinal_4_zero.get("ordinal_3_focused_pytest_cases_passed") != 193
        or ordinal_4_zero.get("result_path_present") is not False
        or ordinal_4_zero.get("launch_request_present") is not False
        or ordinal_4_zero.get("formal_state_present") is not False
        or ordinal_4_zero.get("tmux_present") is not False
        or ordinal_4_zero.get("named_process_count") != 0
        or ordinal_4_zero.get("gpu_compute_process_count") != 0
        or ordinal_4_zero.get("gnn_evaluations") != 0
        or ordinal_4_zero.get("gpu_training_seconds") != 0
        or ordinal_4_zero.get("fresh_seed_consumption") != 0
        or ordinal_4_zero.get("test_accessed") is not False
        or ordinal_4_zero.get("phase_b_started") is not False
        or ordinal_4_zero.get("phase_d_started") is not False
        or ordinal_4_zero.get("phase_e_started") is not False
    ):
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_4_ZERO_STATE_MISMATCH")
    if ordinal_4.get("scientific_design_changed") is not False:
        _fail("CONFIG", "ENGINEERING_CORRECTION_ORDINAL_4_SCOPE_MISMATCH")
    if config.get("exposure_adapter_plan") != EXPOSURE_ADAPTER_CONFIG_CONTRACT:
        _fail("CONFIG", "EXPOSURE_ADAPTER_PLAN_CONFIG_MISMATCH")
    if ordinal_4_commitment_sha256(EXACT_HISTORICAL_JSON_ADAPTER_CONTRACT) != (
        EXPOSURE_ADAPTER_PLAN["exact_json_contract_sha256"]
    ):
        _fail("CONFIG", "EXPOSURE_EXACT_JSON_CONTRACT_COMMITMENT_MISMATCH")
    if config["exposure_adapter_plan"]["execution_owned_exact_exclusions"] != [
        CONFIG_REL
    ]:
        _fail("CONFIG", "EXPOSURE_SELF_EXCLUSION_DECLARATION_MISMATCH")
    if config.get("launch_control", {}).get("fixed_tmux_command") != FIXED_TMUX_COMMAND:
        _fail("CONFIG", "FIXED_TMUX_COMMAND_MISMATCH")
    config["_amendment_bytes"] = amendment
    return config


class _ProjectedNode:
    __slots__ = ("fields", "children")

    def __init__(self, fields: Mapping[str, Any] | None = None, children: Sequence[Any] = ()):
        self.fields = dict(fields or {})
        self.children = list(children)


class IdentityJsonProjector:
    """JSON parser that decodes only identity-whitelist values.

    Non-whitelist scalars are skipped lexically.  Non-whitelist containers are
    traversed solely to discover nested whitelist keys.
    """

    def __init__(self, text: str, source: str):
        self.text = text
        self.source = source
        self.position = 0
        self.length = len(text)
        self.decoder = json.JSONDecoder()

    def parse(self) -> _ProjectedNode:
        node = self._parse_value(capture=False)
        self._skip_ws()
        if self.position != self.length:
            raise ValueError("trailing_json_data")
        return node if isinstance(node, _ProjectedNode) else _ProjectedNode()

    def _skip_ws(self) -> None:
        while self.position < self.length and self.text[self.position] in " \t\r\n":
            self.position += 1

    def _expect(self, character: str) -> None:
        self._skip_ws()
        if self.position >= self.length or self.text[self.position] != character:
            raise ValueError("invalid_json_delimiter")
        self.position += 1

    def _parse_string(self) -> str:
        self._skip_ws()
        try:
            value, end = self.decoder.raw_decode(self.text, self.position)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_json_string") from exc
        if not isinstance(value, str):
            raise ValueError("json_key_not_string")
        self.position = end
        return value

    def _parse_value(self, *, capture: bool) -> Any:
        self._skip_ws()
        if self.position >= self.length:
            raise ValueError("unexpected_json_eof")
        char = self.text[self.position]
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array(capture=capture)
        if capture:
            try:
                value, end = self.decoder.raw_decode(self.text, self.position)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid_captured_json_value") from exc
            self.position = end
            return value
        self._skip_scalar_lexically()
        return None

    def _parse_object(self) -> _ProjectedNode:
        fields: dict[str, Any] = {}
        children: list[_ProjectedNode] = []
        seen: set[str] = set()
        self._expect("{")
        self._skip_ws()
        if self.position < self.length and self.text[self.position] == "}":
            self.position += 1
            return _ProjectedNode()
        while True:
            key = self._parse_string()
            if key in seen:
                raise ValueError("duplicate_json_key")
            seen.add(key)
            self._expect(":")
            if key in IDENTITY_WHITELIST:
                fields[key] = self._parse_value(capture=True)
            else:
                child = self._parse_value(capture=False)
                if isinstance(child, _ProjectedNode):
                    children.append(child)
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == ",":
                self.position += 1
                continue
            if self.position < self.length and self.text[self.position] == "}":
                self.position += 1
                return _ProjectedNode(fields, children)
            raise ValueError("invalid_json_object")

    def _parse_array(self, *, capture: bool) -> Any:
        if capture:
            try:
                value, end = self.decoder.raw_decode(self.text, self.position)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid_captured_json_array") from exc
            self.position = end
            return value
        children: list[_ProjectedNode] = []
        self._expect("[")
        self._skip_ws()
        if self.position < self.length and self.text[self.position] == "]":
            self.position += 1
            return _ProjectedNode(children=children)
        while True:
            child = self._parse_value(capture=False)
            if isinstance(child, _ProjectedNode):
                children.append(child)
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == ",":
                self.position += 1
                continue
            if self.position < self.length and self.text[self.position] == "]":
                self.position += 1
                return _ProjectedNode(children=children)
            raise ValueError("invalid_json_array")

    def _skip_scalar_lexically(self) -> None:
        self._skip_ws()
        if self.position >= self.length:
            raise ValueError("missing_json_scalar")
        if self.text[self.position] == '"':
            self.position = self._scan_string_end(self.position)
            return
        start = self.position
        while self.position < self.length and self.text[self.position] not in ",}]\r\n\t ":
            self.position += 1
        if self.position == start:
            raise ValueError("empty_json_scalar")

    def _scan_string_end(self, start: int) -> int:
        position = start + 1
        while position < self.length:
            if self.text[position] == "\\":
                position += 2
                continue
            if self.text[position] == '"':
                return position + 1
            position += 1
        raise ValueError("unterminated_json_string")


def _validate_identity_value(key: str, value: Any) -> Any:
    if key in {
        "architecture_fingerprint",
        "joint_config_fingerprint",
        "joint_fingerprint",
        "hp_fingerprint",
    }:
        if not isinstance(value, str) or not HEX64_RE.fullmatch(value.casefold()):
            raise ValueError("invalid_identity_fingerprint")
        return value.casefold()
    if key == "config_id":
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise ValueError("invalid_config_id")
        return value
    if key in {"operations", "ops"}:
        if not isinstance(value, list) or len(value) > 32 or not all(isinstance(x, str) for x in value):
            raise ValueError("invalid_operations")
        return [str(x) for x in value]
    if key == "edges":
        if not isinstance(value, list) or len(value) > 256:
            raise ValueError("invalid_edges")
        edges: list[list[int]] = []
        for edge in value:
            if (
                not isinstance(edge, (list, tuple))
                or len(edge) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in edge)
            ):
                raise ValueError("invalid_edge")
            edges.append([int(edge[0]), int(edge[1])])
        return edges
    raise ValueError("non_whitelist_identity_key")


def _validate_ordinal_4_identity_value(key: str, value: Any) -> Any:
    if key in {
        "architecture_fingerprint",
        "joint_config_fingerprint",
        "joint_fingerprint",
        "hp_fingerprint",
    } and (not isinstance(value, str) or not HEX64_RE.fullmatch(value)):
        raise ValueError("ordinal_4_identity_fingerprint_not_lowercase_hex64")
    return _validate_identity_value(key, value)


def iter_projected_identity_records(node: _ProjectedNode) -> Iterator[dict[str, Any]]:
    """Yield deduplicated inherited identity records from a projected tree."""

    seen: set[bytes] = set()

    def walk(current: _ProjectedNode, inherited: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        merged = dict(inherited)
        for key, raw in current.fields.items():
            merged[key] = _validate_identity_value(key, raw)
        if current.fields:
            encoded = canonical_json_bytes(merged)
            if encoded not in seen:
                seen.add(encoded)
                yield dict(merged)
        for child in current.children:
            yield from walk(child, merged)

    yield from walk(node, {})


def project_identity_json_bytes(data: bytes, source: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8-sig")
        node = IdentityJsonProjector(text, source).parse()
        return list(iter_projected_identity_records(node))
    except (UnicodeError, ValueError) as exc:
        raise HistoricalProjectionError("IDENTITY_PROJECTION", "JSON_IDENTITY_PROJECTION_FAILED") from exc


def project_identity_jsonl_bytes(data: bytes, source: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError as exc:
        raise HistoricalProjectionError("IDENTITY_PROJECTION", "JSONL_UTF8_FAILED") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        rows.extend(project_identity_json_bytes(line.encode("utf-8"), f"{source}#line"))
    return rows


def _raw_csv_records(text: str) -> Iterator[str]:
    start = 0
    position = 0
    quoted = False
    while position < len(text):
        char = text[position]
        if char == '"':
            if quoted and position + 1 < len(text) and text[position + 1] == '"':
                position += 2
                continue
            quoted = not quoted
        if char in "\r\n" and not quoted:
            yield text[start:position]
            if char == "\r" and position + 1 < len(text) and text[position + 1] == "\n":
                position += 1
            start = position + 1
        position += 1
    if quoted:
        raise ValueError("unterminated_csv_quote")
    if start < len(text):
        yield text[start:]


def _raw_csv_fields(record: str) -> list[str]:
    fields: list[str] = []
    start = 0
    position = 0
    quoted = False
    while position < len(record):
        char = record[position]
        if char == '"':
            if quoted and position + 1 < len(record) and record[position + 1] == '"':
                position += 2
                continue
            quoted = not quoted
        elif char == "," and not quoted:
            fields.append(record[start:position])
            start = position + 1
        position += 1
    if quoted:
        raise ValueError("unterminated_csv_quote")
    fields.append(record[start:])
    return fields


def _decode_csv_cell(raw: str) -> str:
    if raw == "":
        return ""
    row = next(csv.reader([raw], strict=True))
    if len(row) != 1:
        raise ValueError("invalid_csv_cell")
    return row[0]


def project_csv_columns(data: bytes, allowed_columns: set[str]) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = data.decode("utf-8-sig")
        records = list(_raw_csv_records(text))
        while records and not records[-1].strip():
            records.pop()
        if not records:
            return [], []
        raw_header = _raw_csv_fields(records[0])
        header = [_decode_csv_cell(cell) for cell in raw_header]
        if len(header) != len(set(header)):
            raise ValueError("duplicate_csv_header")
        allowed_indices = {index for index, key in enumerate(header) if key in allowed_columns}
        rows: list[dict[str, str]] = []
        for record in records[1:]:
            if not record.strip():
                continue
            raw_fields = _raw_csv_fields(record)
            if len(raw_fields) != len(header):
                raise ValueError("csv_width_mismatch")
            projected = {
                header[index]: _decode_csv_cell(raw_fields[index]) for index in sorted(allowed_indices)
            }
            rows.append(projected)
        return header, rows
    except (UnicodeError, csv.Error, StopIteration, ValueError) as exc:
        raise HistoricalProjectionError("IDENTITY_PROJECTION", "CSV_IDENTITY_PROJECTION_FAILED") from exc


def _strict_json_array_identity_cell(cell: str) -> list[Any]:
    if not cell or cell[0] != "[":
        raise ValueError("identity_cell_not_json_array")
    try:
        value, end = json.JSONDecoder().raw_decode(cell, 0)
    except json.JSONDecodeError as exc:
        raise ValueError("identity_cell_invalid_json") from exc
    if cell[end:].strip() or not isinstance(value, list):
        raise ValueError("identity_cell_json_not_fully_consumed_array")
    return value


def _project_csv_identity_adapter_with_header_groups(
    data: bytes,
    source: str,
    header_groups: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode only whitelisted CSV identity cells under the frozen full headers."""

    try:
        header, raw_rows = project_csv_columns(data, set(IDENTITY_WHITELIST))
        header_sha256 = ordinal_4_commitment_sha256(header)
        header_contract = header_groups.get(header_sha256)
        identity_columns = sorted(set(header) & set(IDENTITY_WHITELIST))
        if (
            header_contract is None
            or len(header) != header_contract["columns"]
            or identity_columns != header_contract["identity_columns"]
        ):
            raise ValueError("csv_header_contract_mismatch")
        nonempty_identity_counts = {
            key: sum(row.get(key, "") != "" for row in raw_rows)
            for key in identity_columns
        }
        schema = {
            "header_sha256": header_sha256,
            "identity_columns": identity_columns,
            "row_count": len(raw_rows),
            "nonempty_identity_counts": dict(sorted(nonempty_identity_counts.items())),
        }
        projected_rows: list[dict[str, Any]] = []
        for raw_row in raw_rows:
            row: dict[str, Any] = {}
            for key in identity_columns:
                cell = raw_row.get(key, "")
                if cell == "":
                    continue
                value: Any = cell
                if key in {"operations", "ops", "edges"}:
                    value = _strict_json_array_identity_cell(cell)
                row[key] = _validate_ordinal_4_identity_value(key, value)
            if "operations" in row and "ops" in row:
                raise ValueError("ambiguous_operations_aliases")
            operations_present = "operations" in row or "ops" in row
            edges_present = "edges" in row
            if operations_present != edges_present:
                raise ValueError("partial_architecture_identity")
            _record_architecture(row)
            projected_rows.append(row)
        return projected_rows, schema
    except (HistoricalProjectionError, UnicodeError, ValueError) as exc:
        raise HistoricalProjectionError(
            "IDENTITY_PROJECTION", "CSV_JSON_IDENTITY_CELL_ADAPTER_FAILED"
        ) from exc


def project_csv_identity_adapter(
    data: bytes, source: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _project_csv_identity_adapter_with_header_groups(
        data, source, CSV_IDENTITY_HEADER_GROUPS
    )


def canonical_architecture_fingerprint(operations: Sequence[str], edges: Sequence[Sequence[int]]) -> str:
    """Call the repository's canonical helper via the interface exposed by eval_utils."""

    from architecture_diagnostics import architecture_fingerprint

    config = {
        "operations": [str(op) for op in operations],
        "edges": [(int(edge[0]), int(edge[1])) for edge in edges],
        "effective_layers": sum(str(op) != "Identity" for op in operations),
    }
    value = str(architecture_fingerprint(config)).casefold()
    if not HEX64_RE.fullmatch(value):
        raise ValueError("canonical_architecture_fingerprint_invalid")
    return value


def _record_architecture(record: Mapping[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    fingerprint = record.get("architecture_fingerprint")
    operations = record.get("operations", record.get("ops"))
    edges = record.get("edges")
    config = None
    if operations is not None and edges is not None:
        config = {
            "operations": [str(op) for op in operations],
            "edges": [[int(a), int(b)] for a, b in edges],
            "effective_layers": sum(str(op) != "Identity" for op in operations),
        }
        recomputed = canonical_architecture_fingerprint(config["operations"], config["edges"])
        if fingerprint is not None and recomputed != fingerprint:
            raise ValueError("architecture_fingerprint_mismatch")
        fingerprint = recomputed
    return (str(fingerprint).casefold() if fingerprint is not None else None), config


class _ExactHistoricalJsonIdentityProjector:
    """Duplicate-rejecting lexical projector for the four bound JSON schemas."""

    def __init__(self, text: str, contract: Mapping[str, Any]):
        self.text = text
        self.length = len(text)
        self.position = 0
        self.decoder = json.JSONDecoder()
        self.records: list[dict[str, Any]] = []
        self.counts: dict[tuple[str, str], int] = {}
        self.actions: list[tuple[str, tuple[str, ...], Mapping[str, Any]]] = []
        for category, kind in (
            ("structured_record_contracts", "record"),
            ("fingerprint_only_contracts", "fingerprint"),
            ("skipped_non_architecture_identity_contracts", "skip_non_architecture"),
            ("skipped_redundant_identity_contracts", "skip_redundant"),
        ):
            for row in contract[category]:
                pointer = str(row["pointer_pattern"])
                if not pointer.startswith("/"):
                    raise ValueError("invalid_pointer_pattern")
                parts = tuple(pointer[1:].split("/"))
                self.actions.append((kind, parts, row))
                self.counts[(kind, pointer)] = 0

    @staticmethod
    def _path_matches(
        pattern: Sequence[str], path: Sequence[tuple[str, bool]]
    ) -> bool:
        if len(pattern) != len(path):
            return False
        for expected, (observed, from_array) in zip(pattern, path):
            if expected == "*":
                if not from_array:
                    return False
            elif from_array or expected != observed:
                return False
        return True

    def _action(
        self, path: Sequence[tuple[str, bool]]
    ) -> tuple[str, Mapping[str, Any]] | None:
        matches = [
            (kind, row)
            for kind, pattern, row in self.actions
            if self._path_matches(pattern, path)
        ]
        if len(matches) > 1:
            raise ValueError("multiple_pointer_contract_match")
        return matches[0] if matches else None

    def parse(self) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
        self._parse_value(())
        self._skip_ws()
        if self.position != self.length:
            raise ValueError("trailing_json_data")
        for kind, _, row in self.actions:
            pointer = str(row["pointer_pattern"])
            if self.counts[(kind, pointer)] != row["expected_count"]:
                raise ValueError("pointer_contract_count_mismatch")
        return list(self.records), dict(self.counts)

    def _skip_ws(self) -> None:
        while self.position < self.length and self.text[self.position] in " \t\r\n":
            self.position += 1

    def _expect(self, character: str) -> None:
        self._skip_ws()
        if self.position >= self.length or self.text[self.position] != character:
            raise ValueError("invalid_json_delimiter")
        self.position += 1

    def _parse_string(self) -> str:
        self._skip_ws()
        try:
            value, end = self.decoder.raw_decode(self.text, self.position)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_json_string") from exc
        if not isinstance(value, str):
            raise ValueError("json_key_not_string")
        self.position = end
        return value

    def _capture_identity_value(self, key: str) -> Any:
        self._skip_ws()
        try:
            value, end = self.decoder.raw_decode(self.text, self.position)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_captured_identity_value") from exc
        self.position = end
        return _validate_ordinal_4_identity_value(key, value)

    def _parse_value(self, path: Sequence[tuple[str, bool]]) -> None:
        self._skip_ws()
        if self.position >= self.length:
            raise ValueError("unexpected_json_eof")
        action = self._action(path)
        if action is not None:
            kind, row = action
            pointer = str(row["pointer_pattern"])
            self.counts[(kind, pointer)] += 1
            if kind == "record":
                if self.text[self.position] != "{":
                    raise ValueError("structured_record_not_object")
                self._parse_object(path, row)
                return
            if kind == "fingerprint":
                if self.text[self.position] != '"':
                    raise ValueError("fingerprint_only_not_string")
                value = self._capture_identity_value("architecture_fingerprint")
                self.records.append({"architecture_fingerprint": value})
                return
            if kind == "skip_non_architecture":
                try:
                    value, end = self.decoder.raw_decode(self.text, self.position)
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid_isolated_edges_count") from exc
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("isolated_edges_count_not_integer")
                self.position = end
                return
            if kind == "skip_redundant":
                if self.text[self.position] != "[":
                    raise ValueError("redundant_identity_summary_not_array")
                self._parse_array(path)
                return
            raise ValueError("unknown_pointer_contract_action")
        char = self.text[self.position]
        if char == "{":
            self._parse_object(path, None)
        elif char == "[":
            self._parse_array(path)
        else:
            self._skip_scalar_lexically()

    def _parse_object(
        self,
        path: Sequence[tuple[str, bool]],
        record_contract: Mapping[str, Any] | None,
    ) -> None:
        fields: dict[str, Any] = {}
        seen: set[str] = set()
        required = (
            set(record_contract["required_identity_keys"])
            if record_contract is not None
            else set()
        )
        self._expect("{")
        self._skip_ws()
        if self.position < self.length and self.text[self.position] == "}":
            self.position += 1
            if record_contract is not None:
                raise ValueError("structured_record_required_identity_missing")
            return
        while True:
            key = self._parse_string()
            if key in seen:
                raise ValueError("duplicate_json_key")
            seen.add(key)
            self._expect(":")
            child_path = (*path, (key, False))
            if record_contract is not None and key in required:
                fields[key] = self._capture_identity_value(key)
            else:
                action = self._action(child_path)
                if key in IDENTITY_WHITELIST and action is None:
                    if record_contract is None:
                        raise ValueError("uncontracted_identity_pointer")
                    if key == "ops" and "operations" in required:
                        raise ValueError("ambiguous_operations_aliases")
                self._parse_value(child_path)
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == ",":
                self.position += 1
                continue
            if self.position < self.length and self.text[self.position] == "}":
                self.position += 1
                break
            raise ValueError("invalid_json_object")
        if record_contract is not None:
            if set(fields) != required:
                raise ValueError("structured_record_required_identity_missing")
            fingerprint, _ = _record_architecture(fields)
            if fingerprint is None:
                raise ValueError("structured_record_architecture_missing")
            self.records.append(fields)

    def _parse_array(self, path: Sequence[tuple[str, bool]]) -> None:
        self._expect("[")
        self._skip_ws()
        if self.position < self.length and self.text[self.position] == "]":
            self.position += 1
            return
        index = 0
        while True:
            self._parse_value((*path, (str(index), True)))
            index += 1
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == ",":
                self.position += 1
                continue
            if self.position < self.length and self.text[self.position] == "]":
                self.position += 1
                return
            raise ValueError("invalid_json_array")

    def _skip_scalar_lexically(self) -> None:
        self._skip_ws()
        if self.position >= self.length:
            raise ValueError("missing_json_scalar")
        if self.text[self.position] == '"':
            self.position = self._scan_string_end(self.position)
            return
        start = self.position
        while self.position < self.length and self.text[self.position] not in ",}]\r\n\t ":
            self.position += 1
        token = self.text[start : self.position]
        if token not in {"true", "false", "null"} and not JSON_NUMBER_RE.fullmatch(token):
            raise ValueError("invalid_json_scalar")

    def _scan_string_end(self, start: int) -> int:
        position = start + 1
        while position < self.length:
            character = self.text[position]
            if ord(character) < 0x20:
                raise ValueError("invalid_json_string_control")
            if character == "\\":
                position += 1
                if position >= self.length:
                    raise ValueError("unterminated_json_escape")
                escaped = self.text[position]
                if escaped == "u":
                    digits = self.text[position + 1 : position + 5]
                    if len(digits) != 4 or any(ch not in "0123456789abcdefABCDEF" for ch in digits):
                        raise ValueError("invalid_json_unicode_escape")
                    position += 5
                    continue
                if escaped not in '"\\/bfnrt':
                    raise ValueError("invalid_json_escape")
                position += 1
                continue
            if character == '"':
                return position + 1
            position += 1
        raise ValueError("unterminated_json_string")


def _exact_json_contract_for_path(relative: str) -> Mapping[str, Any] | None:
    matches = [
        row
        for row in EXACT_HISTORICAL_JSON_ADAPTER_CONTRACT["files"]
        if row["path"] == relative
    ]
    if len(matches) > 1:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_EXACT_JSON_CONTRACT_DUPLICATE_PATH")
    return matches[0] if matches else None


def _semantic_architecture_identity_set(
    data: bytes, projected: Sequence[Mapping[str, Any]]
) -> list[str]:
    fingerprints = {
        match.group(1).decode("ascii").casefold()
        for match in ARCHITECTURE_TOKEN_RE.finditer(data)
    }
    for record in projected:
        fingerprint, _ = _record_architecture(record)
        if fingerprint is not None:
            if not HEX64_RE.fullmatch(fingerprint):
                raise ValueError("semantic_architecture_fingerprint_invalid")
            fingerprints.add(fingerprint)
    return sorted(fingerprints)


def project_exact_historical_json_adapter(
    data: bytes,
    source: str,
    *,
    contract: Mapping[str, Any] | None = None,
    enforce_production_binding: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = contract or _exact_json_contract_for_path(source)
    if selected is None:
        raise HistoricalProjectionError(
            "IDENTITY_PROJECTION", "EXACT_JSON_CONTRACT_PATH_UNMATCHED"
        )
    if enforce_production_binding and (
        selected["path"] != source
        or selected["size"] != len(data)
        or selected["sha256"] != sha256_bytes(data)
    ):
        raise HistoricalProjectionError(
            "IDENTITY_PROJECTION", "EXACT_JSON_CONTRACT_FILE_IDENTITY_MISMATCH"
        )
    try:
        text = data.decode("utf-8-sig")
        records, counts = _ExactHistoricalJsonIdentityProjector(text, selected).parse()
        structured_count = sum(
            counts[("record", row["pointer_pattern"])]
            for row in selected["structured_record_contracts"]
        )
        fingerprint_only_count = sum(
            counts[("fingerprint", row["pointer_pattern"])]
            for row in selected["fingerprint_only_contracts"]
        )
        if (
            structured_count != selected["expected_structured_record_count"]
            or fingerprint_only_count
            != selected["expected_fingerprint_only_record_count"]
        ):
            raise ValueError("exact_json_record_count_mismatch")
        identities = _semantic_architecture_identity_set(data, records)
        if (
            len(identities) != selected["expected_unique_architecture_count"]
            or ordinal_4_commitment_sha256(identities)
            != selected["identity_set_sha256"]
        ):
            raise ValueError("exact_json_identity_set_mismatch")
        return records, {
            "structured_record_count": structured_count,
            "fingerprint_only_record_count": fingerprint_only_count,
            "unique_architecture_count": len(identities),
            "identity_set_sha256": ordinal_4_commitment_sha256(identities),
        }
    except (UnicodeError, ValueError) as exc:
        raise HistoricalProjectionError(
            "IDENTITY_PROJECTION", "EXACT_JSON_IDENTITY_ADAPTER_FAILED"
        ) from exc


def _discover_marker_filtered_exposure_files(config: Mapping[str, Any]) -> list[Path]:
    rule = config["exposure_rule"]
    suffixes = set(rule["suffixes"])
    markers = tuple(rule["excluded_markers"])
    matches: list[Path] = []
    for root_name in rule["roots"]:
        root = REPO_ROOT / root_name
        if not root.is_dir() or root.is_symlink():
            _fail("EXPOSURE_REGISTRY", "EXPOSURE_ROOT_INVALID")
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            directory_names[:] = sorted(
                name for name in directory_names if name != "__pycache__"
            )
            current = Path(directory)
            for name in sorted(file_names):
                path = current / name
                relative = path.relative_to(REPO_ROOT).as_posix()
                lowered = relative.casefold()
                try:
                    mode = path.lstat().st_mode
                except OSError as exc:
                    raise ProtocolFailure("EXPOSURE_REGISTRY", "EXPOSURE_STAT_FAILED") from exc
                if not stat.S_ISREG(mode):
                    continue
                if not any(token in lowered for token in rule["path_markers"]):
                    continue
                if any(marker in lowered for marker in markers):
                    continue
                if path.suffix.casefold() not in suffixes:
                    continue
                matches.append(path)
    return sorted(matches, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def _classify_exposure_files(config: Mapping[str, Any]) -> dict[str, Any]:
    marker_filtered = _discover_marker_filtered_exposure_files(config)
    execution_owned = [
        path
        for path in marker_filtered
        if path.relative_to(REPO_ROOT).as_posix() in TARGET_RELS
    ]
    excluded_relatives = [
        path.relative_to(REPO_ROOT).as_posix() for path in execution_owned
    ]
    declared = config["exposure_adapter_plan"][
        "execution_owned_exact_exclusions"
    ]
    if excluded_relatives != [CONFIG_REL] or declared != [CONFIG_REL]:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_SELF_EXCLUSION_SET_MISMATCH")
    historical = [
        path
        for path in marker_filtered
        if path.relative_to(REPO_ROOT).as_posix() != CONFIG_REL
    ]
    return {
        "marker_filtered": marker_filtered,
        "execution_owned": execution_owned,
        "historical": historical,
    }


def _discover_exposure_files(config: Mapping[str, Any]) -> list[Path]:
    return list(_classify_exposure_files(config)["historical"])


def _read_historical_exposure_binding(path: Path) -> dict[str, Any]:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            _fail("EXPOSURE_REGISTRY", "EXPOSURE_HISTORICAL_FILE_NONREGULAR")
        data = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ProtocolFailure(
            "EXPOSURE_REGISTRY", "EXPOSURE_HISTORICAL_FILE_READ_FAILED"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(data) != before.st_size:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_HISTORICAL_FILE_CHANGED_DURING_READ")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "size": len(data),
        "sha256": sha256_bytes(data),
        "data": data,
    }


def route_exposure_adapter(relative: str, size: int, sha256: str) -> str:
    exact = _exact_json_contract_for_path(relative)
    if exact is not None:
        if exact["size"] != size or exact["sha256"] != sha256:
            _fail("EXPOSURE_REGISTRY", "EXPOSURE_EXACT_JSON_FILE_IDENTITY_MISMATCH")
        return JSON_EXACT_ISOLATED_EDGES_COUNT_COLLISION_V1
    suffix = PurePosixPath(relative).suffix.casefold()
    if suffix == ".csv":
        return CSV_JSON_IDENTITY_CELL_V1
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return JSON_IDENTITY_DEFAULT_V1
    if suffix in {
        ".py",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
    }:
        return OPAQUE_EXACT_IDENTITY_TOKEN_V1
    _fail("EXPOSURE_REGISTRY", "EXPOSURE_ADAPTER_ROUTE_UNMATCHED")
    raise AssertionError("unreachable")


def _project_with_pinned_exposure_adapter(
    data: bytes,
    source: str,
    adapter_id: str,
    *,
    exact_contract: Mapping[str, Any] | None = None,
    enforce_production_binding: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if adapter_id == CSV_JSON_IDENTITY_CELL_V1:
        records, schema = project_csv_identity_adapter(data, source)
        return records, {"csv_schema": schema}
    if adapter_id == JSON_EXACT_ISOLATED_EDGES_COUNT_COLLISION_V1:
        records, evidence = project_exact_historical_json_adapter(
            data,
            source,
            contract=exact_contract,
            enforce_production_binding=enforce_production_binding,
        )
        return records, {"exact_json": evidence}
    if adapter_id == JSON_IDENTITY_DEFAULT_V1:
        suffix = PurePosixPath(source).suffix.casefold()
        if suffix == ".json":
            return project_identity_json_bytes(data, source), {}
        if suffix in {".jsonl", ".ndjson"}:
            return project_identity_jsonl_bytes(data, source), {}
        raise HistoricalProjectionError(
            "IDENTITY_PROJECTION", "DEFAULT_JSON_ADAPTER_SUFFIX_MISMATCH"
        )
    if adapter_id == OPAQUE_EXACT_IDENTITY_TOKEN_V1:
        return [], {}
    raise HistoricalProjectionError(
        "IDENTITY_PROJECTION", "EXPOSURE_PINNED_ADAPTER_UNKNOWN"
    )


def _legacy_exposure_projection(data: bytes, source: str) -> list[dict[str, Any]]:
    suffix = PurePosixPath(source).suffix.casefold()
    if suffix == ".json":
        return project_identity_json_bytes(data, source)
    if suffix in {".jsonl", ".ndjson"}:
        return project_identity_jsonl_bytes(data, source)
    if suffix == ".csv":
        _, rows = project_csv_columns(data, set(IDENTITY_WHITELIST))
        return [dict(row) for row in rows]
    return []


def _historical_metadata_tree_sha256(bindings: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in bindings:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_commitment(actual: str, expected: str, error_code: str) -> None:
    if actual != expected:
        _fail("EXPOSURE_REGISTRY", error_code)


def build_exposure_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    classification = _classify_exposure_files(config)
    marker_filtered = classification["marker_filtered"]
    historical_files = classification["historical"]
    expected_plan = EXPOSURE_ADAPTER_PLAN
    if len(marker_filtered) != expected_plan["marker_filtered_current_file_count"]:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_MARKER_FILTERED_FILE_COUNT_MISMATCH")
    if len(historical_files) != expected_plan["historical_file_count"]:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_FILE_COUNT_MISMATCH")

    bindings = [_read_historical_exposure_binding(path) for path in historical_files]
    tree_sha256 = _historical_metadata_tree_sha256(bindings)
    _require_commitment(
        tree_sha256,
        expected_plan["historical_tree_sha256"],
        "EXPOSURE_TREE_SHA256_MISMATCH",
    )
    root_counts: dict[str, int] = {}
    for row in bindings:
        root = str(row["path"]).split("/", 1)[0]
        root_counts[root] = root_counts.get(root, 0) + 1
    if dict(sorted(root_counts.items())) != expected_plan["historical_root_counts"]:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_ROOT_FILE_COUNTS_MISMATCH")

    adapter_manifest_records: list[dict[str, Any]] = []
    adapter_ids: dict[str, str] = {}
    adapter_class_counts = {adapter_id: 0 for adapter_id in EXPOSURE_ADAPTER_IDS}
    for row in bindings:
        adapter_id = route_exposure_adapter(
            str(row["path"]), int(row["size"]), str(row["sha256"])
        )
        adapter_ids[str(row["path"])] = adapter_id
        adapter_class_counts[adapter_id] += 1
        adapter_manifest_records.append(
            {
                "path": row["path"],
                "size_bytes": row["size"],
                "sha256": row["sha256"],
                "adapter_id": adapter_id,
            }
        )
    if adapter_class_counts != expected_plan["adapter_class_counts"]:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_ADAPTER_CLASS_COUNTS_MISMATCH")
    adapter_manifest_root = ordinal_4_commitment_sha256(adapter_manifest_records)
    _require_commitment(
        adapter_manifest_root,
        expected_plan["adapter_manifest_root_sha256"],
        "EXPOSURE_ADAPTER_MANIFEST_ROOT_MISMATCH",
    )
    _require_commitment(
        ordinal_4_commitment_sha256(EXACT_HISTORICAL_JSON_ADAPTER_CONTRACT),
        expected_plan["exact_json_contract_sha256"],
        "EXPOSURE_EXACT_JSON_CONTRACT_COMMITMENT_MISMATCH",
    )

    exposed: set[str] = set()
    csv_schema_plan: list[dict[str, Any]] = []
    csv_group_observed: dict[str, dict[str, int]] = {}
    legacy_error_records: list[dict[str, Any]] = []
    legacy_success_records: list[dict[str, Any]] = []
    new_projection_records: list[dict[str, Any]] = []
    exact_json_evidence: dict[str, Any] = {}
    csv_error_paths: list[str] = []
    csv_error_inventory: list[dict[str, Any]] = []

    for binding in bindings:
        relative = str(binding["path"])
        data = binding["data"]
        adapter_id = adapter_ids[relative]
        try:
            projected, evidence = _project_with_pinned_exposure_adapter(
                data, relative, adapter_id
            )
            new_identities = _semantic_architecture_identity_set(data, projected)
        except (HistoricalProjectionError, UnicodeError, ValueError):
            _fail("EXPOSURE_REGISTRY", "EXPOSURE_IDENTITY_ADAPTER_FAILURE")
        exposed.update(new_identities)
        new_identity_root = ordinal_4_commitment_sha256(new_identities)
        new_projection_records.append(
            {
                "path": relative,
                "size_bytes": binding["size"],
                "sha256": binding["sha256"],
                "identity_count": len(new_identities),
                "identity_set_sha256": new_identity_root,
                "adapter_id": adapter_id,
            }
        )
        csv_schema = evidence.get("csv_schema")
        if csv_schema is not None:
            csv_schema_plan.append(csv_schema)
            header_sha = csv_schema["header_sha256"]
            aggregate = csv_group_observed.setdefault(
                header_sha, {"files": 0, "rows": 0}
            )
            aggregate["files"] += 1
            aggregate["rows"] += csv_schema["row_count"]
        if "exact_json" in evidence:
            exact_json_evidence[relative] = evidence["exact_json"]

        try:
            legacy_projected = _legacy_exposure_projection(data, relative)
            legacy_identities = _semantic_architecture_identity_set(
                data, legacy_projected
            )
        except (HistoricalProjectionError, UnicodeError, ValueError):
            if adapter_id == CSV_JSON_IDENTITY_CELL_V1 and csv_schema is not None:
                if csv_schema["identity_columns"] != ["edges", "operations"]:
                    _fail("EXPOSURE_REGISTRY", "EXPOSURE_LEGACY_ERROR_CLASS_UNEXPECTED")
                legacy_error_class = "CSV_SERIALIZED_IDENTITY_ARRAY_NOT_DECODED"
                schema_sha256 = ordinal_4_commitment_sha256(csv_schema)
                csv_error_paths.append(relative)
                csv_error_inventory.append(
                    {
                        "path": relative,
                        "size": binding["size"],
                        "sha256": binding["sha256"],
                    }
                )
            elif adapter_id == JSON_EXACT_ISOLATED_EDGES_COUNT_COLLISION_V1:
                legacy_error_class = "JSON_ISOLATED_NONIDENTITY_EDGES_COLLISION"
                schema_sha256 = ordinal_4_commitment_sha256(
                    {"isolated_edges_collision_count": 1}
                )
            else:
                _fail("EXPOSURE_REGISTRY", "EXPOSURE_LEGACY_ERROR_CLASS_UNEXPECTED")
            legacy_error_records.append(
                {
                    "path": relative,
                    "size_bytes": binding["size"],
                    "sha256": binding["sha256"],
                    "legacy_error_class": legacy_error_class,
                    "adapter_id": adapter_id,
                    "schema_sha256": schema_sha256,
                }
            )
        else:
            legacy_identity_root = ordinal_4_commitment_sha256(legacy_identities)
            if legacy_identities != new_identities:
                _fail(
                    "EXPOSURE_REGISTRY",
                    "EXPOSURE_LEGACY_SUCCESS_NEW_PROJECTION_MISMATCH",
                )
            legacy_success_records.append(
                {
                    "path": relative,
                    "size_bytes": binding["size"],
                    "sha256": binding["sha256"],
                    "identity_count": len(legacy_identities),
                    "identity_set_sha256": legacy_identity_root,
                }
            )

    expected_csv_groups = {
        header_sha: {
            "files": row["files"],
            "rows": row["rows"],
        }
        for header_sha, row in CSV_IDENTITY_HEADER_GROUPS.items()
    }
    if csv_group_observed != expected_csv_groups:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_CSV_HEADER_GROUP_AGGREGATE_MISMATCH")
    csv_schema_plan_root = ordinal_4_commitment_sha256(csv_schema_plan)
    _require_commitment(
        csv_schema_plan_root,
        expected_plan["csv_schema_plan_root_sha256"],
        "EXPOSURE_CSV_SCHEMA_PLAN_ROOT_MISMATCH",
    )

    legacy_error_class_counts: dict[str, int] = {}
    for row in legacy_error_records:
        key = str(row["legacy_error_class"])
        legacy_error_class_counts[key] = legacy_error_class_counts.get(key, 0) + 1
    if (
        len(legacy_error_records) != expected_plan["legacy_error_count"]
        or legacy_error_class_counts != expected_plan["legacy_error_class_counts"]
    ):
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_LEGACY_ERROR_COUNTS_MISMATCH")
    legacy_error_root = ordinal_4_commitment_sha256(legacy_error_records)
    _require_commitment(
        legacy_error_root,
        expected_plan["legacy_error_set_root_sha256"],
        "EXPOSURE_LEGACY_ERROR_ROOT_MISMATCH",
    )
    _require_commitment(
        ordinal_4_commitment_sha256(sorted(csv_error_paths)),
        expected_plan["csv_legacy_error_path_set_sha256"],
        "EXPOSURE_CSV_LEGACY_ERROR_PATH_SET_MISMATCH",
    )
    _require_commitment(
        ordinal_4_commitment_sha256(csv_error_inventory),
        expected_plan["csv_legacy_error_inventory_sha256"],
        "EXPOSURE_CSV_LEGACY_ERROR_INVENTORY_MISMATCH",
    )
    if len(legacy_success_records) != expected_plan["legacy_success_count"]:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_LEGACY_SUCCESS_COUNT_MISMATCH")
    legacy_success_root = ordinal_4_commitment_sha256(legacy_success_records)
    _require_commitment(
        legacy_success_root,
        expected_plan["legacy_success_projection_root_sha256"],
        "EXPOSURE_LEGACY_SUCCESS_ROOT_MISMATCH",
    )
    if len(new_projection_records) != expected_plan["new_projection_file_count"]:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_NEW_PROJECTION_COUNT_MISMATCH")
    new_projection_root = ordinal_4_commitment_sha256(new_projection_records)
    _require_commitment(
        new_projection_root,
        expected_plan["new_projection_root_sha256"],
        "EXPOSURE_NEW_PROJECTION_ROOT_MISMATCH",
    )

    exposed_sorted = sorted(exposed)
    exposed_set_root = ordinal_4_commitment_sha256(exposed_sorted)
    if len(exposed_sorted) != expected_plan["exposed_architecture_count"]:
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_ARCHITECTURE_COUNT_MISMATCH")
    _require_commitment(
        exposed_set_root,
        expected_plan["exposed_architecture_set_sha256"],
        "EXPOSURE_ARCHITECTURE_SET_ROOT_MISMATCH",
    )

    file_records = [
        {"path": row["path"], "size": row["size"], "sha256": row["sha256"]}
        for row in bindings
    ]
    registry = {
        "schema_version": "phase-c-v2-layernorm-exposure-registry-ordinal-4-v1",
        "marker_filtered_current_file_count": len(marker_filtered),
        "execution_owned_exact_exclusions": [CONFIG_REL],
        "historical_exposure_file_count": len(bindings),
        "file_count": len(bindings),
        "parse_error_count": 0,
        "suppressed_error_count": 0,
        "tree_sha256": tree_sha256,
        "historical_root_counts": dict(sorted(root_counts.items())),
        "files": file_records,
        "adapter_manifest_records": adapter_manifest_records,
        "adapter_class_counts": adapter_class_counts,
        "adapter_manifest_root_sha256": adapter_manifest_root,
        "adapter_route_coverage": {
            "matched_file_count": len(adapter_manifest_records),
            "unmatched_file_count": 0,
            "multi_match_file_count": 0,
            "fallback_file_count": 0,
        },
        "legacy_error_records": legacy_error_records,
        "legacy_error_count": len(legacy_error_records),
        "legacy_error_class_counts": legacy_error_class_counts,
        "legacy_error_set_root_sha256": legacy_error_root,
        "legacy_success_records": legacy_success_records,
        "legacy_success_count": len(legacy_success_records),
        "legacy_success_projection_root_sha256": legacy_success_root,
        "legacy_success_new_projection_equal": True,
        "new_projection_records": new_projection_records,
        "new_projection_file_count": len(new_projection_records),
        "new_projection_root_sha256": new_projection_root,
        "csv_schema_plan": csv_schema_plan,
        "csv_schema_plan_root_sha256": csv_schema_plan_root,
        "exact_json_adapter_contract_sha256": ordinal_4_commitment_sha256(
            EXACT_HISTORICAL_JSON_ADAPTER_CONTRACT
        ),
        "exact_json_adapter_evidence": exact_json_evidence,
        "exposed_architecture_fingerprint_count": len(exposed_sorted),
        "exposed_architecture_fingerprint_set_sha256": exposed_set_root,
        "exposed_architecture_fingerprints": exposed_sorted,
        "identity_whitelist": sorted(IDENTITY_WHITELIST),
        "forbidden_outcome_families_entered_projection": False,
    }
    expected = config["exposure_expected"]
    if (
        registry["file_count"] != expected["file_count"]
        or registry["parse_error_count"] != expected["parse_error_count"]
        or registry["tree_sha256"] != expected["tree_sha256"]
        or registry["exposed_architecture_fingerprint_count"]
        != expected["architecture_count"]
    ):
        _fail("EXPOSURE_REGISTRY", "EXPOSURE_EXPECTED_CONTRACT_MISMATCH")
    return registry


def assert_frozen_safety_candidates_unexposed(
    config: Mapping[str, Any], exposure_set: set[str]
) -> None:
    frozen_candidates = {
        str(row["architecture_fingerprint"])
        for row in config["safety_selection_expected"]["selected"]
    }
    if frozen_candidates & exposure_set:
        _fail(
            "EXPOSURE_REGISTRY",
            "FROZEN_SAFETY_CANDIDATE_HISTORICAL_EXPOSURE",
        )


def select_safety_architectures(
    projected_rows: Sequence[Mapping[str, str]],
    exposure_set: set[str],
    base_architecture: str,
    selection_material: str,
) -> dict[str, Any]:
    strata = ("OH_BLIND_JOINT_RANDOM", "HR_BALANCED_RANDOM_HP")
    source_architectures = {
        source: {row["architecture_fingerprint"] for row in projected_rows if row["source"] == source}
        for source in strata
    }
    overlap = source_architectures[strata[0]] & source_architectures[strata[1]]
    ranked: dict[str, list[tuple[str, str]]] = {}
    eligible_counts: dict[str, int] = {}
    exposure_intersections: dict[str, int] = {}
    for source in strata:
        exposure_intersections[source] = len(source_architectures[source] & exposure_set)
        eligible = (
            source_architectures[source]
            - overlap
            - exposure_set
            - {base_architecture}
        )
        eligible_counts[source] = len(eligible)
        values = []
        for fingerprint in eligible:
            rank_hash = hashlib.sha256(
                selection_material.encode("utf-8")
                + b"\0"
                + source.encode("utf-8")
                + b"\0"
                + fingerprint.encode("ascii")
            ).hexdigest()
            values.append((rank_hash, fingerprint))
        ranked[source] = sorted(values, key=lambda item: (item[0], item[1]))
    if any(len(ranked[source]) < 2 for source in strata):
        _fail("SAFETY_SELECTION", "INSUFFICIENT_SOURCE_ELIGIBILITY")
    selected = []
    for source in strata:
        for index, (rank_hash, fingerprint) in enumerate(ranked[source][:2], start=1):
            selected.append(
                {
                    "source": source,
                    "rank": index,
                    "architecture_fingerprint": fingerprint,
                    "rank_hash": rank_hash,
                }
            )
    return {
        "cross_source_duplicate_architecture_count": len(overlap),
        "eligible_counts": eligible_counts,
        "exposure_intersection_counts": exposure_intersections,
        "selected": selected,
    }


def _unique_tar_member_bytes(archive_path: Path, member_name: str) -> bytes:
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = [member for member in archive.getmembers() if member.name == member_name]
            if len(members) != 1 or not members[0].isfile():
                _fail("ARCHIVE_IDENTITY", "ARCHIVE_MEMBER_CARDINALITY_MISMATCH")
            handle = archive.extractfile(members[0])
            if handle is None:
                _fail("ARCHIVE_IDENTITY", "ARCHIVE_MEMBER_UNREADABLE")
            return handle.read()
    except tarfile.TarError as exc:
        raise ProtocolFailure("ARCHIVE_IDENTITY", "ARCHIVE_OPEN_FAILED") from exc


_WRONG_CONTROL_CONTAINER = object()


class _ExactTopLevelControlProjector(IdentityJsonProjector):
    """Decode only one file's exact root controls and lexically skip all else."""

    def __init__(
        self,
        text: str,
        source: str,
        allowed_keys: Iterable[str],
        conflicting_aliases: Iterable[str] = (),
    ):
        super().__init__(text, source)
        self.control_allowed_keys = frozenset(str(key) for key in allowed_keys)
        self.conflicting_aliases = frozenset(str(key) for key in conflicting_aliases)
        self.top_level_fields: dict[str, Any] = {}
        self.aliases_seen: set[str] = set()

    def _skip_string_lexically(self) -> None:
        self._skip_ws()
        if self.position >= self.length or self.text[self.position] != '"':
            raise ValueError("json_string_expected")
        self.position += 1
        while self.position < self.length:
            character = self.text[self.position]
            if character == '"':
                self.position += 1
                return
            if ord(character) < 0x20:
                raise ValueError("json_string_control_character")
            if character == "\\":
                self.position += 1
                if self.position >= self.length:
                    raise ValueError("json_string_escape_eof")
                escape = self.text[self.position]
                if escape == "u":
                    digits = self.text[self.position + 1 : self.position + 5]
                    if len(digits) != 4 or any(
                        char not in "0123456789abcdefABCDEF" for char in digits
                    ):
                        raise ValueError("json_unicode_escape_invalid")
                    self.position += 5
                    continue
                if escape not in '"\\/bfnrt':
                    raise ValueError("json_escape_invalid")
            self.position += 1
        raise ValueError("unterminated_json_string")

    def _skip_value_lexically(self) -> None:
        self._skip_ws()
        if self.position >= self.length:
            raise ValueError("unexpected_json_eof")
        character = self.text[self.position]
        if character == '"':
            self._skip_string_lexically()
            return
        if character == "{":
            self._expect("{")
            seen: set[str] = set()
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == "}":
                self.position += 1
                return
            while True:
                key = self._parse_string()
                if key in seen:
                    raise ValueError("duplicate_json_key")
                seen.add(key)
                self._expect(":")
                self._skip_value_lexically()
                self._skip_ws()
                if self.position < self.length and self.text[self.position] == ",":
                    self.position += 1
                    continue
                if self.position < self.length and self.text[self.position] == "}":
                    self.position += 1
                    return
                raise ValueError("invalid_json_object")
        if character == "[":
            self._expect("[")
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == "]":
                self.position += 1
                return
            while True:
                self._skip_value_lexically()
                self._skip_ws()
                if self.position < self.length and self.text[self.position] == ",":
                    self.position += 1
                    continue
                if self.position < self.length and self.text[self.position] == "]":
                    self.position += 1
                    return
                raise ValueError("invalid_json_array")
        for literal in ("true", "false", "null"):
            if self.text.startswith(literal, self.position):
                self.position += len(literal)
                return
        number = JSON_NUMBER_RE.match(self.text, self.position)
        if number is None:
            raise ValueError("invalid_json_scalar")
        self.position = number.end()

    def _capture_safe_scalar(self) -> Any:
        self._skip_ws()
        if self.position >= self.length:
            raise ValueError("safe_scalar_missing")
        if self.text[self.position] in "[{":
            self._skip_value_lexically()
            return _WRONG_CONTROL_CONTAINER
        start = self.position
        self._skip_value_lexically()
        try:
            return json.loads(self.text[start : self.position])
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_safe_scalar") from exc

    def parse_controls(self) -> dict[str, Any]:
        seen: set[str] = set()
        self._expect("{")
        self._skip_ws()
        if self.position < self.length and self.text[self.position] == "}":
            self.position += 1
        else:
            while True:
                key = self._parse_string()
                if key in seen:
                    raise ValueError("duplicate_json_key")
                seen.add(key)
                self._expect(":")
                if key in self.control_allowed_keys:
                    self.top_level_fields[key] = self._capture_safe_scalar()
                else:
                    if key in self.conflicting_aliases:
                        self.aliases_seen.add(key)
                    self._skip_value_lexically()
                self._skip_ws()
                if self.position < self.length and self.text[self.position] == ",":
                    self.position += 1
                    continue
                if self.position < self.length and self.text[self.position] == "}":
                    self.position += 1
                    break
                raise ValueError("invalid_json_object")
        self._skip_ws()
        if self.position != self.length:
            raise ValueError("trailing_json_data")
        return dict(self.top_level_fields)


def project_phase_a_control_file(
    filename: str,
    data: bytes,
    contract: Mapping[str, Any] = PHASE_A_CONTROL_CONTRACT,
) -> dict[str, Any]:
    """Project and validate one named Phase A control without cross-file compensation."""

    if contract.get("control_projection_schema") != PHASE_A_CONTROL_PROJECTION_SCHEMA:
        _fail("PHASE_A", "PHASE_A_CONTROL_PROJECTION_SCHEMA_MISMATCH")
    files = contract.get("files")
    spec = files.get(filename) if isinstance(files, Mapping) else None
    if spec is None:
        _fail("PHASE_A", "PHASE_A_CONTROL_FILE_NOT_ALLOWLISTED")
    required = spec["safe_fields"]
    try:
        parser = _ExactTopLevelControlProjector(
            data.decode("utf-8-sig"), filename, required, spec["conflicting_aliases"]
        )
        projection = parser.parse_controls()
    except (UnicodeError, ValueError) as exc:
        error_code = (
            "PHASE_A_CONTROL_DUPLICATE_KEY"
            if str(exc) == "duplicate_json_key"
            else "PHASE_A_CONTROL_PARSE_FAILED"
        )
        raise HistoricalProjectionError("PHASE_A", error_code) from exc
    if parser.aliases_seen:
        _fail("PHASE_A", "PHASE_A_CONFLICTING_CONTROL_ALIAS")
    if set(projection) != set(required):
        _fail("PHASE_A", "PHASE_A_REQUIRED_TOP_LEVEL_CONTROL_MISSING")
    native_types = {"bool": bool, "int": int, "str": str}
    for key, field_contract in required.items():
        observed = projection[key]
        expected_type = native_types.get(field_contract["native_type"])
        if expected_type is None:
            _fail("PHASE_A", "PHASE_A_CONTROL_CONTRACT_TYPE_UNSUPPORTED")
        type_matches = type(observed) is expected_type
        if not type_matches:
            _fail("PHASE_A", "PHASE_A_CONTROL_NATIVE_TYPE_MISMATCH")
        if observed != field_contract["value"]:
            _fail("PHASE_A", "PHASE_A_CONTROL_VALUE_MISMATCH")
    projection_bytes = canonical_json_bytes(projection) + b"\n"
    if (
        len(projection_bytes) != spec["safe_projection_size_bytes"]
        or sha256_bytes(projection_bytes) != spec["safe_projection_sha256"]
    ):
        _fail("PHASE_A", "PHASE_A_CONTROL_SAFE_PROJECTION_DRIFT")
    return projection


class _PhaseAManifestProjector(IdentityJsonProjector):
    """Placement-aware lexical projector for the one sealed Phase A schema."""

    def __init__(self, text: str, contract: Mapping[str, Any]):
        super().__init__(text, "phase_a_manifest")
        self.contract = contract

    def _skip_string_lexically(self) -> None:
        self._skip_ws()
        if self.position >= self.length or self.text[self.position] != '"':
            raise ValueError("json_string_expected")
        self.position += 1
        while self.position < self.length:
            character = self.text[self.position]
            if character == '"':
                self.position += 1
                return
            if ord(character) < 0x20:
                raise ValueError("json_string_control_character")
            if character == "\\":
                self.position += 1
                if self.position >= self.length:
                    raise ValueError("json_string_escape_eof")
                escape = self.text[self.position]
                if escape == "u":
                    digits = self.text[self.position + 1 : self.position + 5]
                    if len(digits) != 4 or any(char not in "0123456789abcdefABCDEF" for char in digits):
                        raise ValueError("json_unicode_escape_invalid")
                    self.position += 5
                    continue
                if escape not in '"\\/bfnrt':
                    raise ValueError("json_escape_invalid")
            self.position += 1
        raise ValueError("unterminated_json_string")

    def _skip_value_lexically(self) -> None:
        """Validate and skip one JSON value without materializing it."""

        self._skip_ws()
        if self.position >= self.length:
            raise ValueError("unexpected_json_eof")
        character = self.text[self.position]
        if character == '"':
            self._skip_string_lexically()
            return
        if character == "{":
            self._expect("{")
            seen: set[str] = set()
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == "}":
                self.position += 1
                return
            while True:
                key = self._parse_string()
                if key in seen:
                    raise ValueError("duplicate_json_key")
                seen.add(key)
                self._expect(":")
                self._skip_value_lexically()
                self._skip_ws()
                if self.position < self.length and self.text[self.position] == ",":
                    self.position += 1
                    continue
                if self.position < self.length and self.text[self.position] == "}":
                    self.position += 1
                    return
                raise ValueError("invalid_json_object")
        if character == "[":
            self._expect("[")
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == "]":
                self.position += 1
                return
            while True:
                self._skip_value_lexically()
                self._skip_ws()
                if self.position < self.length and self.text[self.position] == ",":
                    self.position += 1
                    continue
                if self.position < self.length and self.text[self.position] == "]":
                    self.position += 1
                    return
                raise ValueError("invalid_json_array")
        for literal in ("true", "false", "null"):
            if self.text.startswith(literal, self.position):
                self.position += len(literal)
                return
        number = JSON_NUMBER_RE.match(self.text, self.position)
        if number is None:
            raise ValueError("invalid_json_scalar")
        self.position = number.end()

    def _capture_safe_scalar(self) -> Any:
        self._skip_ws()
        if self.position >= self.length or self.text[self.position] in "[{":
            raise ValueError("safe_scalar_expected")
        start = self.position
        self._skip_value_lexically()
        raw = self.text[start : self.position]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_safe_scalar") from exc

    def _parse_record(self) -> tuple[dict[str, Any], list[str]]:
        expected_keys = set(self.contract["record_keys"])
        fields: dict[str, Any] = {}
        seen: set[str] = set()
        self._expect("{")
        self._skip_ws()
        if self.position < self.length and self.text[self.position] == "}":
            self.position += 1
            return fields, []
        while True:
            key = self._parse_string()
            if key in seen:
                raise ValueError("duplicate_json_key")
            seen.add(key)
            self._expect(":")
            if key in expected_keys:
                fields[key] = self._capture_safe_scalar()
            else:
                self._skip_value_lexically()
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == ",":
                self.position += 1
                continue
            if self.position < self.length and self.text[self.position] == "}":
                self.position += 1
                break
            raise ValueError("invalid_json_object")
        return fields, sorted(seen)

    def _parse_files(self) -> list[tuple[dict[str, Any], list[str]]]:
        self._skip_ws()
        if self.position >= self.length or self.text[self.position] != "[":
            _fail("PHASE_A", "PHASE_A_MANIFEST_FILES_CONTAINER_MISMATCH")
        records: list[tuple[dict[str, Any], list[str]]] = []
        self._expect("[")
        self._skip_ws()
        if self.position < self.length and self.text[self.position] == "]":
            self.position += 1
            return records
        while True:
            self._skip_ws()
            if self.position >= self.length or self.text[self.position] != "{":
                _fail("PHASE_A", "PHASE_A_MANIFEST_RECORD_CONTAINER_MISMATCH")
            records.append(self._parse_record())
            self._skip_ws()
            if self.position < self.length and self.text[self.position] == ",":
                self.position += 1
                continue
            if self.position < self.length and self.text[self.position] == "]":
                self.position += 1
                return records
            raise ValueError("invalid_json_array")

    def parse_manifest(self) -> tuple[dict[str, Any], list[tuple[dict[str, Any], list[str]]]]:
        expected_root_keys = set(self.contract["root_keys"])
        root_controls: dict[str, Any] = {}
        records: list[tuple[dict[str, Any], list[str]]] | None = None
        seen: set[str] = set()
        self._expect("{")
        self._skip_ws()
        if self.position < self.length and self.text[self.position] == "}":
            self.position += 1
        else:
            while True:
                key = self._parse_string()
                if key in seen:
                    raise ValueError("duplicate_json_key")
                seen.add(key)
                self._expect(":")
                if key == "files":
                    records = self._parse_files()
                elif key == "created_at":
                    self._skip_value_lexically()
                elif key in self.contract["safe_root_controls"]:
                    root_controls[key] = self._capture_safe_scalar()
                else:
                    self._skip_value_lexically()
                self._skip_ws()
                if self.position < self.length and self.text[self.position] == ",":
                    self.position += 1
                    continue
                if self.position < self.length and self.text[self.position] == "}":
                    self.position += 1
                    break
                raise ValueError("invalid_json_object")
        self._skip_ws()
        if self.position != self.length:
            raise ValueError("trailing_json_data")
        if seen != expected_root_keys:
            _fail("PHASE_A", "PHASE_A_MANIFEST_ROOT_SCHEMA_MISMATCH")
        if records is None:
            _fail("PHASE_A", "PHASE_A_MANIFEST_FILES_CONTAINER_MISMATCH")
        return root_controls, records


def project_archive_manifest_records(
    data: bytes, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the sole real manifest schema, then normalize record identities."""

    try:
        parser = _PhaseAManifestProjector(data.decode("utf-8-sig"), contract)
        root_controls, projected_records = parser.parse_manifest()
    except (UnicodeError, ValueError) as exc:
        error_code = (
            "PHASE_A_MANIFEST_DUPLICATE_KEY"
            if str(exc) == "duplicate_json_key"
            else "PHASE_A_MANIFEST_LEXICAL_SCHEMA_MISMATCH"
        )
        raise HistoricalProjectionError("PHASE_A", error_code) from exc
    expected_controls = contract["safe_root_controls"]
    if set(root_controls) != set(expected_controls):
        _fail("PHASE_A", "PHASE_A_MANIFEST_SAFE_CONTROL_SCHEMA_MISMATCH")
    for key, expected in expected_controls.items():
        observed = root_controls[key]
        if type(observed) is not type(expected) or observed != expected:
            _fail("PHASE_A", "PHASE_A_MANIFEST_SAFE_CONTROL_IDENTITY_MISMATCH")
    expected_record_keys = list(contract["record_keys"])
    raw_records: list[dict[str, Any]] = []
    for fields, keys in projected_records:
        if keys != expected_record_keys or set(fields) != set(expected_record_keys):
            _fail("PHASE_A", "PHASE_A_MANIFEST_RECORD_SCHEMA_MISMATCH")
        relative_path = fields["relative_path"]
        size_bytes = fields["size_bytes"]
        digest = fields["sha256"]
        if (
            type(relative_path) is not str
            or type(size_bytes) is not int
            or size_bytes < 0
            or type(digest) is not str
            or not HEX64_RE.fullmatch(digest)
        ):
            _fail("PHASE_A", "PHASE_A_MANIFEST_RECORD_TYPE_MISMATCH")
        parsed_path = PurePosixPath(relative_path)
        if (
            not relative_path
            or "\x00" in relative_path
            or parsed_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            _fail("PHASE_A", "PHASE_A_MANIFEST_RECORD_PATH_UNSAFE")
        raw_records.append(dict(fields))
    expected_paths = list(contract["relative_paths"])
    observed_paths = [row["relative_path"] for row in raw_records]
    expected_count = expected_controls["file_count"]
    if len(raw_records) != expected_count or len(observed_paths) != len(set(observed_paths)):
        _fail("PHASE_A", "PHASE_A_MANIFEST_RECORD_CARDINALITY_MISMATCH")
    if set(observed_paths) != set(expected_paths):
        _fail("PHASE_A", "PHASE_A_MANIFEST_PATH_SET_MISMATCH")
    safe_projection = {
        "root_type": "dict",
        "root_keys": sorted(contract["root_keys"]),
        "file_count": root_controls["file_count"],
        "files_count": len(raw_records),
        "records": [
            {
                "keys": sorted(record),
                "types": {key: type(record[key]).__name__ for key in sorted(record)},
                "relative_path": record["relative_path"],
            }
            for record in raw_records
        ],
    }
    projection_bytes = canonical_json_bytes(safe_projection) + b"\n"
    if (
        len(projection_bytes) != contract["safe_projection_size_bytes"]
        or sha256_bytes(projection_bytes) != contract["safe_projection_sha256"]
    ):
        _fail("PHASE_A", "PHASE_A_MANIFEST_SAFE_PROJECTION_DRIFT")
    normalized = [
        {
            "path": record["relative_path"],
            "size": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in raw_records
    ]
    return {
        "records": normalized,
        "record_count": len(normalized),
        "manifest_root_sha256": root_controls["manifest_root_sha256"],
        "safe_projection_size_bytes": len(projection_bytes),
        "safe_projection_sha256": sha256_bytes(projection_bytes),
    }


def validate_phase_a_control_consistency(
    controls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-check already validated per-file projections and derive the audit return."""

    expected_files = set(PHASE_A_CONTROL_CONTRACT["files"])
    if set(controls) != expected_files:
        _fail("PHASE_A", "PHASE_A_CONTROL_FILE_SET_MISMATCH")
    decision = controls["decision.json"]
    state = controls["state.json"]
    training_audit = controls["training_audit.json"]
    firewall = controls["test_firewall.json"]
    frozen_classification = "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL"
    if (
        decision["classification"] != frozen_classification
        or state["classification"] != frozen_classification
        or decision["classification"] != state["classification"]
    ):
        _fail("PHASE_A", "PHASE_A_CLASSIFICATION_CONSISTENCY_MISMATCH")
    if decision["training_sufficient"] != training_audit["sufficient"]:
        _fail("PHASE_A", "PHASE_A_TRAINING_SUFFICIENCY_CONSISTENCY_MISMATCH")
    if decision["training"] != training_audit["training_truncation_signal"]:
        _fail("PHASE_A", "PHASE_A_FINAL_TRAINING_SIGNAL_CONSISTENCY_MISMATCH")
    if decision["phase_b_started"] is not False or state["phase_b_started"] is not False:
        _fail("PHASE_A", "PHASE_A_PHASE_B_CONSISTENCY_MISMATCH")
    if (
        decision["test_accessed"] is not False
        or state["test_accessed"] is not False
        or firewall["test_data_reads"] != 0
        or firewall["passed"] is not True
        or firewall["runner_exact_command_verified"] is not True
    ):
        _fail("PHASE_A", "PHASE_A_TEST_ZERO_CONSISTENCY_MISMATCH")
    if decision["new_gnn_runs"] != 0 or firewall["gnn_training_calls"] != 0:
        _fail("PHASE_A", "PHASE_A_GNN_ZERO_CONSISTENCY_MISMATCH")
    if (
        decision["gpu_compute_runs"] != 0
        or state["gpu_compute_used"] is not False
        or firewall["gpu_compute_calls"] != 0
    ):
        _fail("PHASE_A", "PHASE_A_GPU_ZERO_CONSISTENCY_MISMATCH")
    return {
        "scientific_state": state["status"],
        "decision": decision["classification"],
        "boundary": decision["boundary"],
        "training": decision["training"],
        "training_sufficient": decision["training_sufficient"],
        "gnn_evaluations": decision["new_gnn_runs"],
        "test_accessed": decision["test_accessed"],
    }


def phase_a_control_projection_evidence(
    controls: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any] = PHASE_A_CONTROL_CONTRACT,
) -> dict[str, Any]:
    """Recompute the four canonical safe projections and their aggregate."""

    files = contract["files"]
    if set(controls) != set(files):
        _fail("PHASE_A", "PHASE_A_CONTROL_FILE_SET_MISMATCH")
    identities: dict[str, dict[str, Any]] = {}
    for filename in sorted(files):
        projection_bytes = canonical_json_bytes(dict(controls[filename])) + b"\n"
        identity = {
            "size_bytes": len(projection_bytes),
            "sha256": sha256_bytes(projection_bytes),
        }
        expected = files[filename]
        if (
            identity["size_bytes"] != expected["safe_projection_size_bytes"]
            or identity["sha256"] != expected["safe_projection_sha256"]
        ):
            _fail("PHASE_A", "PHASE_A_CONTROL_SAFE_PROJECTION_DRIFT")
        identities[filename] = identity
    aggregate_bytes = canonical_json_bytes(
        {filename: dict(controls[filename]) for filename in sorted(controls)}
    ) + b"\n"
    aggregate = {
        "size_bytes": len(aggregate_bytes),
        "sha256": sha256_bytes(aggregate_bytes),
    }
    if (
        aggregate["size_bytes"] != contract["aggregate_safe_projection_size_bytes"]
        or aggregate["sha256"] != contract["aggregate_safe_projection_sha256"]
    ):
        _fail("PHASE_A", "PHASE_A_CONTROL_AGGREGATE_PROJECTION_DRIFT")
    return {"files": identities, "aggregate": aggregate}


def audit_phase_a(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["phase_a"]
    ordinal_3 = config.get("engineering_correction_ordinal_3")
    control_contract = (
        ordinal_3.get("phase_a_control_contract")
        if isinstance(ordinal_3, Mapping)
        else binding.get("control_contract")
    )
    if (
        not isinstance(control_contract, Mapping)
        or control_contract.get("control_projection_schema")
        != PHASE_A_CONTROL_PROJECTION_SCHEMA
        or set(control_contract.get("files", {})) != set(PHASE_A_CONTROL_CONTRACT["files"])
    ):
        _fail("PHASE_A", "PHASE_A_CONTROL_CONTRACT_MISMATCH")
    live = REPO_ROOT / binding["primary_table"]
    archive_path = REPO_ROOT / binding["archive"]
    assert_file_identity(live, binding["primary_sha256"], binding["primary_size"])
    assert_file_identity(archive_path, binding["archive_sha256"])
    archived_primary = _unique_tar_member_bytes(archive_path, binding["primary_archive_member"])
    if sha256_bytes(archived_primary) != binding["primary_sha256"] or archived_primary != live.read_bytes():
        _fail("PHASE_A", "PHASE_A_PRIMARY_ARCHIVE_MISMATCH")
    companion = archive_path.with_name(archive_path.name + ".sha256")
    if not companion.is_file():
        _fail("PHASE_A", "PHASE_A_OUTER_SHA_COMPANION_MISSING")
    companion_text = companion.read_text(encoding="ascii").strip().split()
    if not companion_text or companion_text[0] != binding["archive_sha256"]:
        _fail("PHASE_A", "PHASE_A_OUTER_SHA_COMPANION_MISMATCH")
    result_dir = live.parent
    manifest_identity = binding["manifest_identity"]
    manifest_contract = binding["manifest_schema"]
    live_manifest = REPO_ROOT / manifest_identity["path"]
    if live_manifest != result_dir / "manifest.json":
        _fail("PHASE_A", "PHASE_A_MANIFEST_PLACEMENT_MISMATCH")
    live_decision = result_dir / "decision.json"
    live_state = result_dir / "state.json"
    live_training = result_dir / "training_audit.json"
    live_firewall = result_dir / "test_firewall.json"
    for required in (live_manifest, live_decision, live_state, live_training, live_firewall):
        if not required.is_file() or required.is_symlink():
            _fail("PHASE_A", "PHASE_A_REQUIRED_CONTROL_MISSING")
    archive_prefix = binding["primary_archive_member"].rsplit("/", 1)[0]
    live_manifest_bytes = live_manifest.read_bytes()
    if (
        len(live_manifest_bytes) != manifest_identity["size_bytes"]
        or sha256_bytes(live_manifest_bytes) != manifest_identity["sha256"]
    ):
        _fail("PHASE_A", "PHASE_A_MANIFEST_BYTE_IDENTITY_MISMATCH")
    archived_manifest = _unique_tar_member_bytes(archive_path, f"{archive_prefix}/manifest.json")
    if (
        len(archived_manifest) != manifest_identity["size_bytes"]
        or sha256_bytes(archived_manifest) != manifest_identity["sha256"]
        or archived_manifest != live_manifest_bytes
    ):
        _fail("PHASE_A", "PHASE_A_MANIFEST_ARCHIVE_MISMATCH")
    manifest_projection = project_archive_manifest_records(archived_manifest, manifest_contract)
    manifest_rows = manifest_projection["records"]
    controls: dict[str, dict[str, Any]] = {}
    live_paths = {
        "decision.json": live_decision,
        "state.json": live_state,
        "training_audit.json": live_training,
        "test_firewall.json": live_firewall,
    }
    for filename, path in live_paths.items():
        member_name = f"{archive_prefix}/{filename}"
        archived_bytes = _unique_tar_member_bytes(archive_path, member_name)
        matches = [row for row in manifest_rows if row["path"] == filename]
        if len(matches) != 1:
            _fail("PHASE_A", "PHASE_A_CONTROL_MANIFEST_BINDING_MISSING")
        row = matches[0]
        if row["size"] != len(archived_bytes) or row["sha256"] != sha256_bytes(archived_bytes):
            _fail("PHASE_A", "PHASE_A_CONTROL_MANIFEST_IDENTITY_MISMATCH")
        live_bytes = path.read_bytes()
        if archived_bytes != live_bytes:
            _fail("PHASE_A", "PHASE_A_CONTROL_ARCHIVE_LIVE_MISMATCH")
        file_contract = control_contract["files"][filename]
        if (
            len(live_bytes) != file_contract["size_bytes"]
            or sha256_bytes(live_bytes) != file_contract["sha256"]
        ):
            _fail("PHASE_A", "PHASE_A_CONTROL_FROZEN_IDENTITY_MISMATCH")
        controls[filename] = project_phase_a_control_file(
            filename, live_bytes, control_contract
        )
    derived = validate_phase_a_control_consistency(controls)
    projection_evidence = phase_a_control_projection_evidence(
        controls, control_contract
    )
    return {
        "status": "PASS",
        **derived,
        "primary_sha256": binding["primary_sha256"],
        "archive_sha256": binding["archive_sha256"],
        "manifest_sha256": manifest_identity["sha256"],
        "manifest_size_bytes": manifest_identity["size_bytes"],
        "manifest_root_sha256": manifest_projection["manifest_root_sha256"],
        "manifest_record_count": manifest_projection["record_count"],
        "manifest_safe_projection_size_bytes": manifest_projection[
            "safe_projection_size_bytes"
        ],
        "manifest_safe_projection_sha256": manifest_projection[
            "safe_projection_sha256"
        ],
        "control_member_count": len(controls),
        "control_projection_schema": control_contract["control_projection_schema"],
        "control_projection_identities": projection_evidence["files"],
        "control_projection_aggregate": projection_evidence["aggregate"],
        "outer_sha_companion_verified": True,
        "archive_member_verified": True,
    }


def project_primary_table(config: Mapping[str, Any]) -> list[dict[str, str]]:
    binding = config["phase_a"]
    data = (REPO_ROOT / binding["primary_table"]).read_bytes()
    header, rows = project_csv_columns(data, {"architecture_fingerprint", "source"})
    if ",".join(header) != binding["primary_header"]:
        _fail("PRIMARY_PROJECTOR", "PRIMARY_HEADER_MISMATCH")
    projected: list[dict[str, str]] = []
    allowed_sources = {"OH_BLIND_JOINT_RANDOM", "HR_BALANCED_RANDOM_HP"}
    for row in rows:
        fingerprint = str(row.get("architecture_fingerprint", "")).casefold()
        source = str(row.get("source", ""))
        if not HEX64_RE.fullmatch(fingerprint):
            _fail("PRIMARY_PROJECTOR", "PRIMARY_ARCHITECTURE_ID_INVALID")
        if source in allowed_sources:
            projected.append({"architecture_fingerprint": fingerprint, "source": source})
    return projected


def audit_source_manifests(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    joined: dict[str, dict[str, Any]] = {}
    for binding in config["source_manifests"]:
        live_path = REPO_ROOT / binding["live_path"]
        archive_path = REPO_ROOT / binding["archive_path"]
        assert_file_identity(live_path, binding["sha256"])
        if not archive_path.is_file() or archive_path.is_symlink():
            _fail("SOURCE_MANIFEST", "SOURCE_ARCHIVE_MISSING")
        member = _unique_tar_member_bytes(archive_path, binding["archive_member"])
        live_bytes = live_path.read_bytes()
        if member != live_bytes or sha256_bytes(member) != binding["sha256"]:
            _fail("SOURCE_MANIFEST", "SOURCE_MANIFEST_ARCHIVE_MISMATCH")
        records = project_identity_json_bytes(live_bytes, binding["live_path"])
        for record in records:
            try:
                fingerprint, architecture = _record_architecture(record)
            except ValueError as exc:
                raise ProtocolFailure("SOURCE_MANIFEST", "SOURCE_ARCHITECTURE_IDENTITY_MISMATCH") from exc
            if fingerprint is None or architecture is None:
                continue
            previous = joined.get(fingerprint)
            if previous is not None and canonical_json_bytes(previous) != canonical_json_bytes(architecture):
                _fail("SOURCE_MANIFEST", "SOURCE_ARCHITECTURE_AMBIGUOUS")
            joined[fingerprint] = architecture
    return joined


def verify_selection(
    config: Mapping[str, Any], selection: Mapping[str, Any], joined: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    expected = config["safety_selection_expected"]
    if selection["cross_source_duplicate_architecture_count"] != 0:
        _fail("SAFETY_SELECTION", "CROSS_SOURCE_DUPLICATE_MISMATCH")
    if selection["eligible_counts"] != expected["eligible_counts"]:
        _fail("SAFETY_SELECTION", "ELIGIBLE_COUNT_MISMATCH")
    if any(value != 0 for value in selection["exposure_intersection_counts"].values()):
        _fail("SAFETY_SELECTION", "EXPOSURE_INTERSECTION_MISMATCH")
    actual_pairs = [
        (row["source"], row["rank"], row["architecture_fingerprint"], row["rank_hash"])
        for row in selection["selected"]
    ]
    expected_pairs = [
        (row["source"], row["rank"], row["architecture_fingerprint"], row["rank_hash"])
        for row in expected["selected"]
    ]
    if actual_pairs != expected_pairs:
        _fail("SAFETY_SELECTION", "FROZEN_CANDIDATE_OR_RANK_MISMATCH")
    output = dict(selection)
    enriched = []
    for row in selection["selected"]:
        fingerprint = row["architecture_fingerprint"]
        if fingerprint not in joined:
            _fail("SAFETY_SELECTION", "FROZEN_CANDIDATE_JOIN_MISSING")
        enriched.append({**row, **joined[fingerprint]})
    output["selected"] = enriched
    output["selector_input_fields"] = ["architecture_fingerprint", "source"]
    output["validation_or_test_entered_selector"] = False
    output["mutation_invariance_passed"] = True
    return output


def decode_hp_screening(config: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from eval_utils import decode_hp_by_mode

    decoded_points: list[dict[str, Any]] = []
    expected_rows = config["hp_screening"]["expected_decoded"]
    for index, raw in enumerate(config["hp_screening"]["raw_points"], start=1):
        tensor = torch.tensor(raw, dtype=torch.float32)
        hp = decode_hp_by_mode(
            z_search=tensor,
            arch_nz=0,
            config={"operations": [], "edges": [], "effective_layers": 0},
            hp_mode="global4",
            log_lr_min=-4.0,
            log_lr_max=-1.5,
            dropout_min=0.1,
            dropout_max=0.6,
        )
        expected = expected_rows[index - 1]
        if (
            abs(hp["lr"] - expected["lr"]) > 1e-12
            or abs(hp["dropout"] - expected["dropout"]) > 1e-12
            or hp["hidden_dim"] != expected["hidden"]
            or abs(hp["weight_decay"] - expected["l2"]) > 1e-12
            or hp["gat_heads"] != 1
            or hp["sage_aggr"] != "mean"
            or hp["gin_eps"] != 0.0
        ):
            _fail("HP_SCREENING", "HP_DECODER_OUTPUT_MISMATCH")
        decoded_points.append(
            {
                "point": index,
                "raw_coordinate_order": ["u_lr", "u_dropout", "u_hidden", "u_l2"],
                "raw": [float(value) for value in raw],
                "decoded": {
                    "lr": hp["lr"],
                    "dropout": hp["dropout"],
                    "hidden_dim": hp["hidden_dim"],
                    "weight_decay": hp["weight_decay"],
                    "gat_heads": hp["gat_heads"],
                    "sage_aggr": hp["sage_aggr"],
                    "gin_eps": hp["gin_eps"],
                    "gat_heads_by_layer": hp["gat_heads_by_layer"],
                    "sage_aggr_by_layer": hp["sage_aggr_by_layer"],
                    "gin_eps_by_layer": hp["gin_eps_by_layer"],
                },
            }
        )
    configurations: list[dict[str, Any]] = []
    for candidate in selection["selected"]:
        joint_ids: set[str] = set()
        for point in decoded_points:
            hp_fingerprint = sha256_bytes(canonical_json_bytes(point))
            joint_fingerprint = sha256_bytes(
                canonical_json_bytes(
                    {
                        "architecture_fingerprint": candidate["architecture_fingerprint"],
                        "hp_fingerprint": hp_fingerprint,
                    }
                )
            )
            if joint_fingerprint in joint_ids:
                _fail("HP_SCREENING", "DUPLICATE_ARCHITECTURE_HP_CONFIGURATION")
            joint_ids.add(joint_fingerprint)
            configurations.append(
                {
                    "source": candidate["source"],
                    "architecture_fingerprint": candidate["architecture_fingerprint"],
                    "operations": candidate["operations"],
                    "edges": candidate["edges"],
                    "effective_layers": candidate["effective_layers"],
                    "point": point["point"],
                    "raw": point["raw"],
                    "decoded": point["decoded"],
                    "hp_fingerprint": hp_fingerprint,
                    "joint_fingerprint": joint_fingerprint,
                }
            )
    if len(configurations) != 20 or len({row["joint_fingerprint"] for row in configurations}) != 20:
        _fail("HP_SCREENING", "HP_PANEL_CARDINALITY_MISMATCH")
    return {
        "schema_version": "phase-c-v2-hp-screening-freeze-v1",
        "decoder": "eval_utils.py::decode_hp_by_mode",
        "hp_mode": "global4",
        "direct_coordinate_bounds": [0.8, 0.95],
        "raw_coordinate_order": ["u_lr", "u_dropout", "u_hidden", "u_l2"],
        "decoded_points": decoded_points,
        "configurations": configurations,
        "configuration_count": 20,
        "selection_rule": "highest five-seed arithmetic mean; joint_fingerprint ascending tie-break",
        "always_frozen_before_primary": True,
    }


def build_seed_freeze(
    config: Mapping[str, Any], occupied_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scanner_path = REPO_ROOT / "analyse/final_checkpoint_seed_inventory.py"
    assert_file_identity(scanner_path, config["dependency_hashes"]["analyse/final_checkpoint_seed_inventory.py"])
    spec = importlib.util.spec_from_file_location("phase_c_v2_seed_inventory", scanner_path)
    if spec is None or spec.loader is None:
        _fail("SEED_INVENTORY", "SEED_SCANNER_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        if occupied_inventory is None:
            inventory = module.build_historical_gnn_seed_inventory(REPO_ROOT)
            module.verify_fixed_fresh_seeds(inventory)
            executed = {int(value) for value in inventory["executed_gnn_training_seeds"]}
        else:
            inventory = dict(occupied_inventory)
            module.verify_conservative_seed_collision_superset(inventory)
            executed = {int(value) for value in inventory["occupied_seed_values"]}
    except Exception as exc:
        raise ProtocolFailure("SEED_INVENTORY", "HISTORICAL_SEED_INVENTORY_FAILED") from exc
    selected: list[int] = []
    candidate = int(config["seed_rule"]["start"])
    while len(selected) < int(config["seed_rule"]["count"]):
        if candidate not in executed:
            selected.append(candidate)
        candidate += 1
    if len(selected) != 45 or len(set(selected)) != 45 or set(selected) & executed:
        _fail("SEED_INVENTORY", "PHASE_C_SEED_COLLISION_OR_CARDINALITY_FAILURE")
    primary = selected[:30]
    screening = selected[30:35]
    safety = selected[35:45]
    if set(primary) & set(screening) or set(primary) & set(safety) or set(screening) & set(safety):
        _fail("SEED_INVENTORY", "PHASE_C_SEED_PARTITION_OVERLAP")
    executed_hash = sha256_bytes(canonical_json_bytes(sorted(executed)))
    selected_hash = sha256_bytes(canonical_json_bytes(selected))
    source_hashes = [
        {
            "source": row["source"],
            "size": row["size"],
            "sha256": row["sha256"],
            "kind": row["kind"],
        }
        for row in inventory["sources"]
    ]
    return {
        "schema_version": "phase-c-v2-global-seed-collision-freeze-v1",
        "scanner_sha256": config["dependency_hashes"]["analyse/final_checkpoint_seed_inventory.py"],
        "scan_roots": inventory["scan_roots"],
        "scanner_inventory_root_sha256": (
            inventory.get("inventory_root_sha256")
            or inventory["set_commitments"]["S"]["root_sha256"]
        ),
        "scanner_source_count": inventory["source_count"],
        "scanner_source_hashes": source_hashes,
        "executed_gnn_training_seed_count": len(executed),
        "executed_gnn_training_seed_set_sha256": executed_hash,
        "selected_seed_count": len(selected),
        "selected_seed_set_sha256": selected_hash,
        "selected_seeds": selected,
        "primary_seeds": primary,
        "screening_seeds": screening,
        "safety_seeds": safety,
        "selected_collision_count": 0,
        "selected_collision_free": True,
        "partition_disjoint": True,
        "unknown_numeric_seed_semantics": False,
        "skipped_test_field_count": (
            inventory.get("skipped_test_field_count")
            if "skipped_test_field_count" in inventory else inventory["test_skip"]["count"]
        ),
        "skipped_test_field_path_commitment_sha256": (
            inventory.get("skipped_test_field_path_commitment_sha256")
            if "skipped_test_field_path_commitment_sha256" in inventory
            else inventory["test_skip"]["path_commitment_sha256"]
        ),
    }


class ValidationOnlyDataProxy:
    """Ledgered attribute firewall used by every formal evaluator call."""

    def __init__(self, data: Any, ledger: dict[str, Any] | None = None):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_ledger", ledger if ledger is not None else _new_access_ledger())

    def __getattribute__(self, name: str) -> Any:
        ledger = object.__getattribute__(self, "_ledger")
        lowered = str(name).casefold()
        if "test" in lowered or name not in DATA_ALLOWLIST:
            ledger["forbidden_request_count"] += 1
            ledger["forbidden_attribute_counts"][str(name)] = (
                ledger["forbidden_attribute_counts"].get(str(name), 0) + 1
            )
            if "test" in lowered:
                ledger["test_accessed"] = True
            raise TestAccessViolation("validation-only data attribute denied")
        ledger["allowed_request_count"] += 1
        ledger["allowed_attribute_counts"][name] += 1
        return getattr(object.__getattribute__(self, "_data"), name)


def _new_access_ledger() -> dict[str, Any]:
    return {
        "allowed_request_count": 0,
        "forbidden_request_count": 0,
        "allowed_attribute_counts": {key: 0 for key in sorted(DATA_ALLOWLIST)},
        "forbidden_attribute_counts": {},
        "test_accessed": False,
    }


def _proxy_internal_ledger(proxy: ValidationOnlyDataProxy) -> dict[str, Any]:
    """The only internal proxy bookkeeping access; never exposed as an attribute."""

    return object.__getattribute__(proxy, "_ledger")


def _aggregate_access_ledgers(ledgers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = _new_access_ledger()
    for ledger in ledgers:
        output["allowed_request_count"] += int(ledger["allowed_request_count"])
        output["forbidden_request_count"] += int(ledger["forbidden_request_count"])
        output["test_accessed"] = bool(output["test_accessed"] or ledger["test_accessed"])
        for name, count in ledger["allowed_attribute_counts"].items():
            output["allowed_attribute_counts"][name] += int(count)
        for name, count in ledger.get("forbidden_attribute_counts", {}).items():
            output["forbidden_attribute_counts"][name] = (
                output["forbidden_attribute_counts"].get(name, 0) + int(count)
            )
    return output


def probe_track_test(data: Any, *, track_test: bool) -> bool:
    if track_test:
        _ = data.test_mask
    return False


def run_guarded_evaluation(
    task: Mapping[str, Any],
    data_proxy: ValidationOnlyDataProxy,
    device: Any,
    evaluator: Callable[..., Any] | None = None,
) -> Any:
    """The sole formal evaluator boundary; track_test is a literal False."""

    if evaluator is None:
        from eval_utils import train_and_eval_arch

        evaluator = train_and_eval_arch
    hp = task["hp"]
    return evaluator(
        config=task["architecture"],
        data=data_proxy,
        in_ch=int(task["in_ch"]),
        out_ch=int(task["out_ch"]),
        lr=float(hp["lr"]),
        dropout=float(hp["dropout"]),
        hidden_dim=int(hp["hidden_dim"]),
        weight_decay=float(hp["weight_decay"]),
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
    wrapper_tree = ast.parse(inspect.getsource(run_guarded_evaluation))
    wrapper_calls = [node for node in ast.walk(wrapper_tree) if isinstance(node, ast.Call)]
    compliant_calls = 0
    for call in wrapper_calls:
        keywords = {item.arg: item.value for item in call.keywords if item.arg is not None}
        value = keywords.get("track_test")
        if isinstance(value, ast.Constant) and value.value is False:
            compliant_calls += 1
    if compliant_calls != 1:
        _fail("TEST_FIREWALL", "FORMAL_WRAPPER_TRACK_TEST_AST_MISMATCH")
    eval_source = (REPO_ROOT / "eval_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(eval_source)
    target = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_and_eval_arch"),
        None,
    )
    if target is None:
        _fail("TEST_FIREWALL", "ROOT_EVALUATOR_AST_MISSING")
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(target):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    accesses = [
        node for node in ast.walk(target) if isinstance(node, ast.Attribute) and node.attr == "test_mask"
    ]
    if not accesses:
        _fail("TEST_FIREWALL", "ROOT_TEST_MASK_NEGATIVE_CONTROL_MISSING")
    for access in accesses:
        ancestor = parents.get(access)
        protected = False
        while ancestor is not None and ancestor is not target:
            if isinstance(ancestor, ast.If) and isinstance(ancestor.test, ast.Name) and ancestor.test.id == "track_test":
                protected = True
                break
            ancestor = parents.get(ancestor)
        if not protected:
            _fail("TEST_FIREWALL", "ROOT_TEST_MASK_NOT_TRACK_TEST_GUARDED")
    return {
        "wrapper_literal_track_test_false_calls": compliant_calls,
        "root_test_mask_access_count": len(accesses),
        "root_test_mask_all_guarded_by_track_test": True,
    }


def synthetic_firewall_evidence() -> dict[str, Any]:
    class SyntheticData:
        x = object()
        edge_index = object()
        train_mask = object()
        val_mask = object()
        y = object()
        test_mask = object()

    ledger = _new_access_ledger()
    proxy = ValidationOnlyDataProxy(SyntheticData(), ledger)
    for key in sorted(DATA_ALLOWLIST):
        getattr(proxy, key)
    bypass_denials = []
    for key in ("_data", "_ledger", "ledger", "__dict__", "num_nodes"):
        try:
            getattr(proxy, key)
        except TestAccessViolation:
            bypass_denials.append(key)
    direct_failed = False
    try:
        _ = proxy.test_mask
    except TestAccessViolation:
        direct_failed = True
    probe_failed = False
    try:
        probe_track_test(proxy, track_test=True)
    except TestAccessViolation:
        probe_failed = True
    if not direct_failed or not probe_failed or bypass_denials != [
        "_data",
        "_ledger",
        "ledger",
        "__dict__",
        "num_nodes",
    ]:
        _fail("TEST_FIREWALL", "SYNTHETIC_TEST_MASK_NEGATIVE_CONTROL_FAILED")
    ast_evidence = assert_evaluator_ast_contract()
    return {
        "schema_version": "phase-c-v2-test-firewall-prelaunch-v1",
        "proxy_allowlist": sorted(DATA_ALLOWLIST),
        "direct_test_mask_negative_control": "PASS",
        "track_test_true_negative_control": "PASS",
        "private_and_nonallowlisted_bypass_negative_controls": bypass_denials,
        "formal_wrapper_ast": ast_evidence,
        "negative_control_ledger": ledger,
        "formal_test_accessed": False,
        "synthetic_only": True,
    }


def synthetic_projection_invariance() -> dict[str, Any]:
    identities = [
        ("a" * 64, "OH_BLIND_JOINT_RANDOM"),
        ("b" * 64, "OH_BLIND_JOINT_RANDOM"),
        ("c" * 64, "OH_BLIND_JOINT_RANDOM"),
        ("d" * 64, "HR_BALANCED_RANDOM_HP"),
        ("e" * 64, "HR_BALANCED_RANDOM_HP"),
        ("f" * 64, "HR_BALANCED_RANDOM_HP"),
    ]
    outcome_keys = {
        "validation": "validation",
        "accuracy": "accuracy",
        "loss": "loss",
        "predictor/prediction": "predictor_prediction",
        "test": "test",
    }

    def pipeline(mutated_family: str | None) -> tuple[dict[str, Any], int]:
        touch_count = 0
        exposure_documents = []
        for index, (fingerprint, _) in enumerate((identities[0], identities[3])):
            if mutated_family is not None:
                touch_count += 1
            exposure_documents.append(
                {
                    "architecture_fingerprint": fingerprint,
                    "validation": 9001 if mutated_family == "validation" else index,
                    "accuracy": 9002 if mutated_family == "accuracy" else index,
                    "loss": -9003 if mutated_family == "loss" else index,
                    "predictor_prediction": 9004
                    if mutated_family == "predictor/prediction"
                    else index,
                    "test": 9005 if mutated_family == "test" else index,
                }
            )
        projected_exposure, _ = _project_with_pinned_exposure_adapter(
            canonical_json_bytes(exposure_documents),
            "synthetic_exposure.json",
            JSON_IDENTITY_DEFAULT_V1,
            enforce_production_binding=False,
        )
        exposure_set = {
            str(row["architecture_fingerprint"])
            for row in projected_exposure
            if "architecture_fingerprint" in row
        }
        header = (
            "architecture_fingerprint,source,validation,accuracy,loss,"
            "predictor_prediction,test\n"
        )
        body_rows = []
        for fingerprint, source in identities:
            if mutated_family is not None:
                touch_count += 1
            body_rows.append(
                f"{fingerprint},{source},"
                f"{9001 if mutated_family == 'validation' else 1},"
                f"{9002 if mutated_family == 'accuracy' else 2},"
                f"{-9003 if mutated_family == 'loss' else 3},"
                f"{9004 if mutated_family == 'predictor/prediction' else 4},"
                f"{9005 if mutated_family == 'test' else 5}\n"
            )
        body = "".join(body_rows)
        _, primary_rows = project_csv_columns(
            (header + body).encode("utf-8"), {"architecture_fingerprint", "source"}
        )
        selection = select_safety_architectures(
            primary_rows,
            exposure_set,
            "9" * 64,
            "synthetic-selection",
        )
        return {
            "identity_projection": projected_exposure,
            "exposure_set": sorted(exposure_set),
            "primary_identity_projection": primary_rows,
            "eligible_counts": selection["eligible_counts"],
            "candidates": [row["architecture_fingerprint"] for row in selection["selected"]],
            "rank_hashes": [row["rank_hash"] for row in selection["selected"]],
        }, touch_count

    base, base_touch_count = pipeline(None)
    if base_touch_count != 0:
        _fail("SYNTHETIC_PROJECTOR", "BASE_MUTATION_TOUCH_COUNT_NONZERO")
    families: dict[str, Any] = {}
    for family in outcome_keys:
        mutated, touch_count = pipeline(family)
        if touch_count <= 0:
            _fail("SYNTHETIC_PROJECTOR", "FORBIDDEN_OUTCOME_MUTATION_ZERO_TOUCH")
        if mutated != base:
            _fail("SYNTHETIC_PROJECTOR", "FORBIDDEN_OUTCOME_MUTATION_CHANGED_SELECTION")
        families[family] = {
            "mutation_touch_count": touch_count,
            "pinned_adapter_id": JSON_IDENTITY_DEFAULT_V1,
            "identity_projection_invariant": True,
            "exposure_set_invariant": True,
            "primary_identity_projection_invariant": True,
            "eligible_counts_invariant": True,
            "candidate_set_invariant": True,
            "rank_hashes_invariant": True,
        }
    return {
        "status": "PASS",
        "forbidden_families_mutated": list(outcome_keys),
        "family_evidence": families,
        "identity_projection_invariant": True,
        "exposure_set_invariant": True,
        "primary_identity_projection_invariant": True,
        "eligible_counts_invariant": True,
        "candidate_set_invariant": True,
        "rank_hashes_invariant": True,
    }


def classify_primary(mean_delta: float, p_value: float, lower_bound: float) -> str:
    inferential = p_value < 0.05 and lower_bound > 0.0
    if mean_delta >= 0.003 and inferential:
        return "FIXED_LN_STRONG_SIGNAL"
    if 0.0 < mean_delta < 0.003 and inferential:
        return "FIXED_LN_AUXILIARY_SIGNAL"
    return "FIXED_LN_NOT_CONFIRMED"


def safety_pass(architecture_deltas: Mapping[str, float], sources: Mapping[str, str]) -> bool:
    values = list(architecture_deltas.values())
    if len(values) != 4:
        return False
    source_values: dict[str, list[float]] = {}
    for architecture, delta in architecture_deltas.items():
        source_values.setdefault(sources[architecture], []).append(float(delta))
    return (
        sum(delta > 0.0 for delta in values) >= 3
        and sum(values) / 4.0 > 0.0
        and all(delta > -0.003 for delta in values)
        and set(source_values) == {"OH_BLIND_JOINT_RANDOM", "HR_BALANCED_RANDOM_HP"}
        and all(len(group) == 2 and sum(group) / 2.0 > 0.0 for group in source_values.values())
    )


def budget_for_primary_classification(classification: str) -> int:
    return 240 if classification == "FIXED_LN_STRONG_SIGNAL" else 60


def _dependency_audit(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in config["dependency_hashes"].items():
        path = REPO_ROOT / relative
        assert_file_identity(path, expected)
        observed[relative] = expected
    return observed


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
    config: Mapping[str, Any], *, require_zero_compute_processes: bool, enforce_memory: bool = True
) -> dict[str, Any]:
    gpu = _run_readonly(
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
    if gpu.returncode or processes.returncode:
        _fail("GPU_ADMISSION", "GPU_INVENTORY_UNAVAILABLE")
    gpu_rows: list[dict[str, Any]] = []
    for line in gpu.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            _fail("GPU_ADMISSION", "GPU_INVENTORY_FORMAT_MISMATCH")
        try:
            row = {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mib": int(parts[3]),
                "memory_used_mib": int(parts[4]),
                "memory_free_mib": int(parts[5]),
                "utilization_percent": int(parts[6]),
            }
        except ValueError as exc:
            raise ProtocolFailure("GPU_ADMISSION", "GPU_INVENTORY_FORMAT_MISMATCH") from exc
        gpu_rows.append(row)
    expected = [
        {"index": int(row["index"]), "uuid": str(row["uuid"])}
        for row in config["workers"]["gpus"]
    ]
    observed = [{"index": row["index"], "uuid": row["uuid"]} for row in gpu_rows]
    if observed != expected:
        _fail("GPU_ADMISSION", "GPU_PHYSICAL_INDEX_UUID_MAPPING_MISMATCH")
    threshold = int(config["workers"]["minimum_free_memory_mib"])
    if enforce_memory and any(row["memory_free_mib"] < threshold for row in gpu_rows):
        _fail("GPU_ADMISSION", "GPU_RESOURCE_ADMISSION_FAILED")
    process_count = len([line for line in processes.stdout.splitlines() if line.strip()])
    if require_zero_compute_processes and process_count != 0:
        _fail("GPU_ADMISSION", "GPU_COMPUTE_PROCESS_OCCUPIED")
    return {
        "gpus": gpu_rows,
        "gpu_process_count": process_count,
        "zero_compute_processes_required": require_zero_compute_processes,
        "memory_threshold_enforced": enforce_memory,
        "minimum_free_memory_mib": threshold,
    }


def audit_worker_gpu_binding(
    config: Mapping[str, Any], expected_index: int, expected_uuid: str
) -> dict[str, Any]:
    evidence = audit_gpu_admission(
        config, require_zero_compute_processes=False, enforce_memory=False
    )
    row = next((item for item in evidence["gpus"] if item["index"] == expected_index), None)
    if row is None or row["uuid"] != expected_uuid:
        _fail("FORMAL_WORKER_GPU_BINDING", "WORKER_GPU_UUID_MISMATCH")
    return {
        "expected_gpu_index": expected_index,
        "expected_gpu_uuid": expected_uuid,
        "observed_gpu_index": row["index"],
        "observed_gpu_uuid": row["uuid"],
    }


def audit_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    branch = _run_readonly(["git", "branch", "--show-current"])
    head = _run_readonly(["git", "rev-parse", "HEAD"])
    status = _run_readonly(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    hostname = _run_readonly(["hostname"])
    if branch.returncode or head.returncode or status.returncode or hostname.returncode:
        _fail("ENVIRONMENT", "REPOSITORY_METADATA_COMMAND_FAILED")
    if branch.stdout.strip() != config["repository"]["branch"]:
        _fail("ENVIRONMENT", "BRANCH_MISMATCH")
    if head.stdout.strip() != config["repository"]["head"]:
        _fail("ENVIRONMENT", "HEAD_MISMATCH")
    dirty_status_sha = sha256_bytes(status.stdout.encode("utf-8"))
    if dirty_status_sha != config["engineering_correction"]["dirty_status_porcelain_z_sha256"]:
        _fail("ENVIRONMENT", "UNRELATED_DIRTY_STATUS_CHANGED")
    if hostname.stdout.strip() not in {"ismc36", "ismc36-ts"}:
        _fail("ENVIRONMENT", "HOSTNAME_MISMATCH")
    gpu_evidence = audit_gpu_admission(config, require_zero_compute_processes=True)
    tmux = _run_readonly(["tmux", "has-session", "-t", TMUX_NAME])
    if tmux.returncode == 0:
        _fail("ENVIRONMENT", "FORMAL_TMUX_ALREADY_EXISTS")
    if tmux.returncode not in {1}:
        _fail("ENVIRONMENT", "TMUX_ABSENCE_UNVERIFIED")
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "status": "PASS",
        "hostname": hostname.stdout.strip(),
        "branch": branch.stdout.strip(),
        "head": head.stdout.strip(),
        "dirty_status": [row for row in status.stdout.split("\0") if row],
        "dirty_status_sha256": dirty_status_sha,
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
        "gpus": gpu_evidence["gpus"],
        "gpu_process_count": gpu_evidence["gpu_process_count"],
        "minimum_free_memory_mib": gpu_evidence["minimum_free_memory_mib"],
        "tmux_absent": True,
        "worker_count": config["workers"]["count"],
        "worker_mapping": config["workers"]["mapping"],
    }


def build_runtime_estimate(config: Mapping[str, Any], worker_count: int) -> dict[str, Any]:
    samples: list[float] = []
    verified = []
    for record in config["runtime_cost_records"]:
        path = REPO_ROOT / record["path"]
        assert_file_identity(path, record["sha256"])
        per_training = float(record["gpu_seconds"]) / int(record["new_trainings"])
        samples.extend([per_training] * int(record["new_trainings"]))
        verified.append(
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "new_trainings": record["new_trainings"],
                "gpu_seconds": record["gpu_seconds"],
            }
        )
    if len(samples) != 87:
        _fail("RUNTIME_ESTIMATE", "COST_SAMPLE_CARDINALITY_MISMATCH")
    ordered = sorted(samples)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    mean = sum(samples) / len(samples)
    safety_factor = 1.5
    overhead_seconds = 300.0

    def branch(tasks: int) -> dict[str, float]:
        best = tasks * quantile(0.10) / worker_count + overhead_seconds
        expected = tasks * mean / worker_count * 1.25 + overhead_seconds
        worst = tasks * quantile(0.90) / worker_count * safety_factor + 2 * overhead_seconds
        return {
            "task_count": tasks,
            "best_seconds": best,
            "expected_seconds": expected,
            "worst_seconds": worst,
        }

    return {
        "schema_version": "phase-c-v2-runtime-estimate-v1",
        "cost_metadata_only": True,
        "historical_accuracy_or_completion_inferred": False,
        "verified_cost_records": verified,
        "historical_new_training_count": len(samples),
        "worker_count": worker_count,
        "archive_statistics_overhead_seconds": overhead_seconds,
        "worst_case_safety_factor": safety_factor,
        "per_training_gpu_seconds_summary": {
            "minimum": min(samples),
            "mean": mean,
            "median": quantile(0.5),
            "maximum": max(samples),
        },
        "minimum_60_gnn_branch": branch(60),
        "maximum_240_gnn_branch": branch(240),
    }


def validate_base_p_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    base = config.get("base_p")
    if not isinstance(base, dict) or canonical_json_bytes(base) != canonical_json_bytes(BASE_P_CONTRACT):
        _fail("BASE_P_IDENTITY", "BASE_P_EXACT_CONTRACT_MISMATCH")
    if canonical_architecture_fingerprint(base["operations"], base["edges"]) != base[
        "architecture_fingerprint"
    ]:
        _fail("BASE_P_IDENTITY", "BASE_P_ARCHITECTURE_FINGERPRINT_MISMATCH")
    for key in ("joint_fingerprint", "architecture_fingerprint", "hp_fingerprint"):
        if not HEX64_RE.fullmatch(str(base[key])):
            _fail("BASE_P_IDENTITY", "BASE_P_FINGERPRINT_FORMAT_MISMATCH")
    return dict(base)


def _task_pair_key(task: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(task["stage"]),
        str(task["architecture_fingerprint"]),
        str(task.get("joint_fingerprint") or ""),
        int(task["seed"]),
    )


def assign_stage_tasks(
    stage: str,
    tasks: Sequence[Mapping[str, Any]],
    worker_mapping: Sequence[Mapping[str, Any]],
    *,
    matched_pairs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping_by_worker = {int(row["worker_id"]): dict(row) for row in worker_mapping}
    if sorted(mapping_by_worker) != list(range(len(worker_mapping))):
        _fail("TASK_ASSIGNMENT", "WORKER_MAPPING_ID_MISMATCH")
    assigned: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    if matched_pairs:
        grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
        for task in tasks:
            if task.get("stage") != stage:
                _fail("TASK_ASSIGNMENT", "TASK_STAGE_MISMATCH")
            grouped.setdefault(_task_pair_key(task), []).append(task)
        for pair_index, pair_key in enumerate(sorted(grouped)):
            members = grouped[pair_key]
            variants = [str(row.get("component_variant")) for row in members]
            if len(members) != 2 or sorted(variants) != ["baseline", "layer_norm"]:
                _fail("TASK_ASSIGNMENT", "MATCHED_PAIR_CONDITION_MISMATCH")
            worker_id = pair_index % len(worker_mapping)
            mapping = mapping_by_worker[worker_id]
            pair_id = sha256_bytes(canonical_json_bytes(list(pair_key)))
            pair_record = {
                "pair_id": pair_id,
                "pair_key": {
                    "stage": pair_key[0],
                    "architecture_fingerprint": pair_key[1],
                    "joint_fingerprint_or_empty": pair_key[2],
                    "seed": pair_key[3],
                },
                "worker_id": worker_id,
                "gpu_index": int(mapping["gpu_index"]),
                "gpu_uuid": str(mapping["gpu_uuid"]),
                "task_ids": sorted(str(row["task_id"]) for row in members),
            }
            pair_records.append(pair_record)
            for task in sorted(members, key=lambda row: str(row["component_variant"])):
                assigned.append(
                    {
                        **dict(task),
                        "pair_id": pair_id,
                        "pair_key": pair_record["pair_key"],
                        "worker_id": worker_id,
                        "gpu_index": int(mapping["gpu_index"]),
                        "gpu_uuid": str(mapping["gpu_uuid"]),
                    }
                )
    else:
        for task_index, task in enumerate(sorted(tasks, key=lambda row: str(row["task_id"]))):
            if task.get("stage") != stage:
                _fail("TASK_ASSIGNMENT", "TASK_STAGE_MISMATCH")
            worker_id = task_index % len(worker_mapping)
            mapping = mapping_by_worker[worker_id]
            assigned.append(
                {
                    **dict(task),
                    "pair_id": None,
                    "pair_key": None,
                    "worker_id": worker_id,
                    "gpu_index": int(mapping["gpu_index"]),
                    "gpu_uuid": str(mapping["gpu_uuid"]),
                }
            )
    worker_task_counts = {
        str(worker_id): sum(int(row["worker_id"]) == worker_id for row in assigned)
        for worker_id in sorted(mapping_by_worker)
    }
    worker_pair_counts = {
        str(worker_id): sum(int(row["worker_id"]) == worker_id for row in pair_records)
        for worker_id in sorted(mapping_by_worker)
    }
    gpu_pair_counts = {
        str(gpu_index): sum(int(row["gpu_index"]) == gpu_index for row in pair_records)
        for gpu_index in sorted({int(row["gpu_index"]) for row in worker_mapping})
    }
    if stage == "primary" and (
        sorted(worker_pair_counts.values()) != [7, 7, 8, 8]
        or sorted(gpu_pair_counts.values()) != [15, 15]
    ):
        _fail("TASK_ASSIGNMENT", "PRIMARY_PAIR_BALANCE_MISMATCH")
    if stage == "safety" and (
        sorted(worker_pair_counts.values()) != [10, 10, 10, 10]
        or sorted(gpu_pair_counts.values()) != [20, 20]
    ):
        _fail("TASK_ASSIGNMENT", "SAFETY_PAIR_BALANCE_MISMATCH")
    if stage == "screening" and sorted(worker_task_counts.values()) != [25, 25, 25, 25]:
        _fail("TASK_ASSIGNMENT", "SCREENING_WORKER_BALANCE_MISMATCH")
    ledger = {
        "schema_version": "phase-c-v2-stage-assignment-ledger-v2",
        "stage": stage,
        "matched_pairs": matched_pairs,
        "assignment_rule": "sorted pair keys round-robin" if matched_pairs else "sorted task IDs round-robin",
        "task_count": len(assigned),
        "pair_count": len(pair_records),
        "worker_task_counts": worker_task_counts,
        "worker_pair_counts": worker_pair_counts,
        "gpu_pair_counts": gpu_pair_counts,
        "pairs": pair_records,
        "tasks": [
            {
                "task_id": row["task_id"],
                "pair_id": row["pair_id"],
                "worker_id": row["worker_id"],
                "gpu_index": row["gpu_index"],
                "gpu_uuid": row["gpu_uuid"],
            }
            for row in sorted(assigned, key=lambda item: str(item["task_id"]))
        ],
    }
    return assigned, ledger


def _base_p_evaluator_hp(base: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lr": base["lr"],
        "dropout": base["dropout"],
        "hidden_dim": base["hidden"],
        "weight_decay": base["l2"],
        "gat_heads": 1,
        "sage_aggr": "mean",
        "gin_eps": 0.0,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
    }


def validate_primary_task_contract(
    config: Mapping[str, Any], primary_tasks: Sequence[Mapping[str, Any]]
) -> None:
    base = validate_base_p_contract(config)
    primary_hp = _base_p_evaluator_hp(base)
    pairs: dict[int, set[str]] = {}
    for row in primary_tasks:
        pairs.setdefault(int(row["seed"]), set()).add(str(row["component_variant"]))
    if (
        len(primary_tasks) != 60
        or len(pairs) != 30
        or any(variants != {"baseline", "layer_norm"} for variants in pairs.values())
        or any(
            row["joint_fingerprint"] != base["joint_fingerprint"]
            or row["architecture_fingerprint"] != base["architecture_fingerprint"]
            or row["hp_fingerprint"] != base["hp_fingerprint"]
            or row["architecture"]["operations"] != base["operations"]
            or row["architecture"]["edges"] != base["edges"]
            or row["architecture"]["effective_layers"]
            != sum(op != "Identity" for op in base["operations"])
            or row["hp"] != primary_hp
            or row["max_epochs"] != 300
            or row["patience"] != 80
            for row in primary_tasks
        )
    ):
        _fail("BASE_P_IDENTITY", "PRIMARY_TASK_BASE_P_CONTRACT_MISMATCH")


def validate_task_freeze_contract(
    config: Mapping[str, Any], task_freeze: Mapping[str, Any]
) -> None:
    validate_primary_task_contract(config, task_freeze["primary_tasks"])
    expected_budget = {
        "minimum_completed_branch": 60,
        "conditional_screening": 100,
        "conditional_safety": 80,
        "maximum": 240,
    }
    expected_retry = {
        "identical_task_only": True,
        "maximum_retries_per_task": 2,
        "stop_if_retry_task_fraction_exceeds": 0.05,
        "stop_if_third_attempt_fails": True,
    }
    expected_stopping = {
        "non_strong_primary_stops_after_gnn": 60,
        "strong_primary_runs_screening_then_safety": True,
        "candidate_seed_condition_replacement": False,
        "imputation": False,
    }
    if (
        task_freeze.get("budget") != expected_budget
        or task_freeze.get("retry_rule") != expected_retry
        or task_freeze.get("stopping_rule") != expected_stopping
        or task_freeze.get("worker_count") != config["workers"]["count"]
        or task_freeze.get("worker_mapping") != config["workers"]["mapping"]
        or len(task_freeze.get("screening_tasks", [])) != 100
        or len(task_freeze.get("safety_templates", [])) != 4
    ):
        _fail("TASK_FREEZE", "FORMAL_TASK_FREEZE_CONTRACT_MISMATCH")
    for stage, matched in (("primary", True), ("screening", False)):
        tasks = task_freeze[f"{stage}_tasks"]
        reassigned, _ = assign_stage_tasks(
            stage, tasks, config["workers"]["mapping"], matched_pairs=matched
        )
        expected = {
            row["task_id"]: (row["pair_id"], row["worker_id"], row["gpu_index"], row["gpu_uuid"])
            for row in reassigned
        }
        observed = {
            row["task_id"]: (
                row.get("pair_id"),
                row.get("worker_id"),
                row.get("gpu_index"),
                row.get("gpu_uuid"),
            )
            for row in tasks
        }
        if observed != expected:
            _fail("TASK_FREEZE", "FORMAL_FROZEN_ASSIGNMENT_MISMATCH")


def validate_seed_task_binding(
    task_freeze: Mapping[str, Any], seed_freeze: Mapping[str, Any]
) -> None:
    primary = [int(value) for value in seed_freeze["primary_seeds"]]
    screening = [int(value) for value in seed_freeze["screening_seeds"]]
    safety = [int(value) for value in seed_freeze["safety_seeds"]]
    selected = [int(value) for value in seed_freeze["selected_seeds"]]
    if (
        len(primary) != 30
        or len(screening) != 5
        or len(safety) != 10
        or selected != primary + screening + safety
        or len(set(selected)) != 45
        or seed_freeze.get("selected_collision_count") != 0
        or seed_freeze.get("selected_collision_free") is not True
        or seed_freeze.get("partition_disjoint") is not True
    ):
        _fail("SEED_INVENTORY", "FORMAL_SEED_PARTITION_CONTRACT_MISMATCH")
    primary_task_seeds = {int(row["seed"]) for row in task_freeze["primary_tasks"]}
    screening_task_seeds = {int(row["seed"]) for row in task_freeze["screening_tasks"]}
    screening_joints = {str(row["joint_fingerprint"]) for row in task_freeze["screening_tasks"]}
    if (
        primary_task_seeds != set(primary)
        or screening_task_seeds != set(screening)
        or len(screening_joints) != 20
        or any(
            {int(row["seed"]) for row in task_freeze["screening_tasks"] if row["joint_fingerprint"] == joint}
            != set(screening)
            for joint in screening_joints
        )
        or any([int(value) for value in row["seeds"]] != safety for row in task_freeze["safety_templates"])
    ):
        _fail("SEED_INVENTORY", "FORMAL_TASK_SEED_BINDING_MISMATCH")


def validate_hp_task_binding(
    task_freeze: Mapping[str, Any], hp_freeze: Mapping[str, Any]
) -> None:
    configurations = hp_freeze.get("configurations")
    if not isinstance(configurations, list) or len(configurations) != 20:
        _fail("HP_SCREENING", "FORMAL_HP_FREEZE_CARDINALITY_MISMATCH")
    by_joint = {str(row["joint_fingerprint"]): row for row in configurations}
    if len(by_joint) != 20:
        _fail("HP_SCREENING", "FORMAL_HP_FREEZE_CARDINALITY_MISMATCH")
    screening_tasks = task_freeze["screening_tasks"]
    for task in screening_tasks:
        frozen = by_joint.get(str(task["joint_fingerprint"]))
        if frozen is None or (
            task["architecture_fingerprint"] != frozen["architecture_fingerprint"]
            or task["hp_fingerprint"] != frozen["hp_fingerprint"]
            or task["architecture"]["operations"] != frozen["operations"]
            or task["architecture"]["edges"] != frozen["edges"]
            or task["hp"] != frozen["decoded"]
            or task["component_variant"] != "baseline"
            or task["max_epochs"] != 300
            or task["patience"] != 80
        ):
            _fail("HP_SCREENING", "FORMAL_SCREENING_TASK_HP_BINDING_MISMATCH")
    panel_by_architecture: dict[str, set[str]] = {}
    for row in configurations:
        panel_by_architecture.setdefault(str(row["architecture_fingerprint"]), set()).add(
            str(row["joint_fingerprint"])
        )
    if len(panel_by_architecture) != 4 or any(len(values) != 5 for values in panel_by_architecture.values()):
        _fail("HP_SCREENING", "FORMAL_HP_PANEL_STRUCTURE_MISMATCH")
    templates = {str(row["architecture_fingerprint"]): row for row in task_freeze["safety_templates"]}
    if set(templates) != set(panel_by_architecture) or any(
        set(templates[architecture]["panel_joint_fingerprints"]) != joints
        or templates[architecture]["conditions"] != ["baseline", "layer_norm"]
        for architecture, joints in panel_by_architecture.items()
    ):
        _fail("HP_SCREENING", "FORMAL_SAFETY_TEMPLATE_HP_BINDING_MISMATCH")


def build_task_freeze(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    hp_freeze: Mapping[str, Any],
    seeds: Mapping[str, Any],
) -> dict[str, Any]:
    base = validate_base_p_contract(config)
    primary_hp = _base_p_evaluator_hp(base)
    primary = []
    for seed in seeds["primary_seeds"]:
        for variant in ("baseline", "layer_norm"):
            primary.append(
                {
                    "task_id": f"primary|{seed}|{variant}",
                    "stage": "primary",
                    "joint_fingerprint": base["joint_fingerprint"],
                    "architecture_fingerprint": base["architecture_fingerprint"],
                    "hp_fingerprint": base["hp_fingerprint"],
                    "architecture": {
                        "operations": base["operations"],
                        "edges": base["edges"],
                        "effective_layers": sum(op != "Identity" for op in base["operations"]),
                    },
                    "hp": primary_hp,
                    "seed": seed,
                    "component_variant": variant,
                    "max_epochs": base["max_epochs"],
                    "patience": base["patience"],
                }
            )
    screening = []
    for row in hp_freeze["configurations"]:
        for seed in seeds["screening_seeds"]:
            screening.append(
                {
                    "task_id": f"screening|{row['joint_fingerprint']}|{seed}|baseline",
                    "stage": "screening",
                    "source": row["source"],
                    "architecture_fingerprint": row["architecture_fingerprint"],
                    "joint_fingerprint": row["joint_fingerprint"],
                    "hp_fingerprint": row["hp_fingerprint"],
                    "point": row["point"],
                    "architecture": {
                        "operations": row["operations"],
                        "edges": row["edges"],
                        "effective_layers": row["effective_layers"],
                    },
                    "hp": row["decoded"],
                    "seed": seed,
                    "component_variant": "baseline",
                    "max_epochs": 300,
                    "patience": 80,
                }
            )
    safety_templates = []
    for candidate in selection["selected"]:
        panels = [
            row for row in hp_freeze["configurations"]
            if row["architecture_fingerprint"] == candidate["architecture_fingerprint"]
        ]
        safety_templates.append(
            {
                "source": candidate["source"],
                "architecture_fingerprint": candidate["architecture_fingerprint"],
                "panel_joint_fingerprints": [row["joint_fingerprint"] for row in panels],
                "seeds": seeds["safety_seeds"],
                "conditions": ["baseline", "layer_norm"],
            }
        )
    if len(primary) != 60 or len(screening) != 100 or len(safety_templates) != 4:
        _fail("TASK_FREEZE", "TASK_FREEZE_CARDINALITY_MISMATCH")
    primary, primary_assignment = assign_stage_tasks(
        "primary", primary, config["workers"]["mapping"], matched_pairs=True
    )
    screening, screening_assignment = assign_stage_tasks(
        "screening", screening, config["workers"]["mapping"], matched_pairs=False
    )
    output = {
        "schema_version": "phase-c-v2-task-freeze-v1",
        "primary_tasks": primary,
        "screening_tasks": screening,
        "safety_templates": safety_templates,
        "assignment_ledgers": {
            "primary": primary_assignment,
            "screening": screening_assignment,
            "safety_rule": {
                "matched_pairs": True,
                "assignment_rule": "sorted pair keys round-robin",
                "expected_pair_count": 40,
                "expected_worker_pair_counts": [10, 10, 10, 10],
                "expected_gpu_pair_counts": [20, 20],
            },
        },
        "budget": {
            "minimum_completed_branch": 60,
            "conditional_screening": 100,
            "conditional_safety": 80,
            "maximum": 240,
        },
        "retry_rule": {
            "identical_task_only": True,
            "maximum_retries_per_task": 2,
            "stop_if_retry_task_fraction_exceeds": 0.05,
            "stop_if_third_attempt_fails": True,
        },
        "stopping_rule": {
            "non_strong_primary_stops_after_gnn": 60,
            "strong_primary_runs_screening_then_safety": True,
            "candidate_seed_condition_replacement": False,
            "imputation": False,
        },
        "worker_count": config["workers"]["count"],
        "worker_mapping": config["workers"]["mapping"],
    }
    validate_task_freeze_contract(config, output)
    return output


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise ProtocolFailure("WRITE_SCOPE", "PRELAUNCH_TARGET_ALREADY_EXISTS") from exc
    except OSError as exc:
        raise ProtocolFailure("WRITE_SCOPE", "ATOMIC_EXCLUSIVE_WRITE_FAILED") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            None


def _json_asset(value: Any) -> bytes:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"


def _prelaunch_assets(
    config: Mapping[str, Any],
    environment: Mapping[str, Any],
    phase_a: Mapping[str, Any],
    exposure: Mapping[str, Any],
    selection: Mapping[str, Any],
    hp_freeze: Mapping[str, Any],
    seed_freeze: Mapping[str, Any],
    firewall: Mapping[str, Any],
    runtime: Mapping[str, Any],
    tasks: Mapping[str, Any],
    dependencies: Mapping[str, str],
) -> dict[str, bytes]:
    incident = config["execution_control"]["incident"]
    ordinal_1_target_records = {
        relative: {
            "before_sha256": config["engineering_correction"]["before_sha256"][relative],
            "after_sha256": config["engineering_correction_ordinal_2"][
                "before_file_identities"
            ][relative]["sha256"],
            "after_size": config["engineering_correction_ordinal_2"][
                "before_file_identities"
            ][relative]["size_bytes"],
        }
        for relative in TARGET_RELS
    }
    ordinal_2_target_records = {
        relative: {
            "before_sha256": config["engineering_correction_ordinal_2"][
                "before_file_identities"
            ][relative]["sha256"],
            "before_size": config["engineering_correction_ordinal_2"][
                "before_file_identities"
            ][relative]["size_bytes"],
            "after_sha256": config["engineering_correction_ordinal_3"][
                "before_file_identities"
            ][relative]["sha256"],
            "after_size": config["engineering_correction_ordinal_3"][
                "before_file_identities"
            ][relative]["size_bytes"],
        }
        for relative in TARGET_RELS
    }
    ordinal_3_target_records = {
        relative: {
            "before_sha256": config["engineering_correction_ordinal_3"][
                "before_file_identities"
            ][relative]["sha256"],
            "before_size": config["engineering_correction_ordinal_3"][
                "before_file_identities"
            ][relative]["size_bytes"],
            "after_sha256": config["engineering_correction_ordinal_4"][
                "before_file_identities"
            ][relative]["sha256"],
            "after_size": config["engineering_correction_ordinal_4"][
                "before_file_identities"
            ][relative]["size_bytes"],
        }
        for relative in TARGET_RELS
    }
    ordinal_4_target_records = {
        relative: {
            "before_sha256": config["engineering_correction_ordinal_4"][
                "before_file_identities"
            ][relative]["sha256"],
            "before_size": config["engineering_correction_ordinal_4"][
                "before_file_identities"
            ][relative]["size_bytes"],
            "after_sha256": sha256_path(REPO_ROOT / relative),
            "after_size": (REPO_ROOT / relative).stat().st_size,
        }
        for relative in TARGET_RELS
    }
    amendment_record = {
        "amendment_id": config["scientific_amendment_id"],
        "scientific_amendment_sha256": config["scientific_amendment_sha256"],
        "phase_protocol": config["experiment"],
        "v1_classification": "PHASE_C_V1_PROTOCOL_INFEASIBLE_PREEXECUTION",
        "zero_phase_c_labels_before_v2": True,
        "execution_control": {
            "control_id": config["execution_control"]["control_id"],
            "verified_sha256": config["execution_control"]["sha256"],
            "incident": incident,
            "engineering_correction": {
                "correction_class": "ENGINEERING_PREFLIGHT_CORRECTION",
                "ordinal": 1,
                "control_id": config["engineering_correction"]["control_id"],
                "verified_sha256": config["engineering_correction"]["sha256"],
                "target_files": ordinal_1_target_records,
                "zero_state_blocks": config["engineering_correction"]["zero_state_blocks"],
                "scientific_design_changed": False,
            },
            "engineering_correction_ordinal_2": {
                "correction_class": "ENGINEERING_SCHEMA_BINDING_CORRECTION",
                "ordinal": 2,
                "control_id": config["engineering_correction_ordinal_2"]["control_id"],
                "verified_sha256": config["engineering_correction_ordinal_2"]["sha256"],
                "ordinal_1_terminal": config["engineering_correction_ordinal_2"][
                    "ordinal_1_terminal"
                ],
                "target_files": ordinal_2_target_records,
                "phase_a_manifest_identity": config["engineering_correction_ordinal_2"][
                    "phase_a_manifest_identity"
                ],
                "phase_a_manifest_schema": config["engineering_correction_ordinal_2"][
                    "phase_a_manifest_schema"
                ],
                "zero_state_attestation": config["engineering_correction_ordinal_2"][
                    "zero_state_attestation"
                ],
                "scientific_design_changed": False,
            },
            "engineering_correction_ordinal_3": {
                "correction_class": "ENGINEERING_CONTROL_SCHEMA_BINDING_CORRECTION",
                "ordinal": 3,
                "control_id": config["engineering_correction_ordinal_3"]["control_id"],
                "verified_sha256": config["engineering_correction_ordinal_3"]["sha256"],
                "previous_engineering_terminals": config[
                    "engineering_correction_ordinal_3"
                ]["previous_engineering_terminals"],
                "target_files": ordinal_3_target_records,
                "phase_a_control_contract": config[
                    "engineering_correction_ordinal_3"
                ]["phase_a_control_contract"],
                "zero_state_attestation": config["engineering_correction_ordinal_3"][
                    "zero_state_attestation"
                ],
                "scientific_design_changed": False,
            },
            "engineering_correction_ordinal_4": {
                "correction_class": (
                    "ENGINEERING_EXPOSURE_SELF_BINDING_AND_IDENTITY_SCHEMA_ADAPTER_CORRECTION"
                ),
                "ordinal": 4,
                "control_id": config["engineering_correction_ordinal_4"]["control_id"],
                "verified_sha256": config["engineering_correction_ordinal_4"]["sha256"],
                "ordinal_3_terminal": config["engineering_correction_ordinal_4"][
                    "ordinal_3_terminal"
                ],
                "ordinal_3_after_ledger_identity": config[
                    "engineering_correction_ordinal_4"
                ]["ordinal_3_after_ledger_identity"],
                "target_files": ordinal_4_target_records,
                "adapter_plan_commitments": config[
                    "engineering_correction_ordinal_4"
                ]["adapter_plan_commitments"],
                "zero_state_attestation": config["engineering_correction_ordinal_4"][
                    "zero_state_attestation"
                ],
                "scientific_design_changed": False,
            },
            "provenance_statement": (
                "Formal results derive from the frozen deterministic design and fresh matched "
                "seeds. This statement does not claim the broader historical project never "
                "observed validation or test outcomes."
            ),
        },
    }
    input_hashes = {
        "dependencies": dependencies,
        "phase_a_primary_sha256": phase_a["primary_sha256"],
        "phase_a_archive_sha256": phase_a["archive_sha256"],
        "phase_a_manifest_sha256": phase_a["manifest_sha256"],
        "phase_a_manifest_safe_projection_sha256": phase_a[
            "manifest_safe_projection_sha256"
        ],
        "phase_a_control_projection_schema": phase_a["control_projection_schema"],
        "phase_a_control_projection_identities": phase_a[
            "control_projection_identities"
        ],
        "phase_a_control_projection_aggregate": phase_a[
            "control_projection_aggregate"
        ],
        "exposure_tree_sha256": exposure["tree_sha256"],
        "seed_inventory_root_sha256": seed_freeze["scanner_inventory_root_sha256"],
        "scientific_amendment_sha256": config["scientific_amendment_sha256"],
        "execution_control_sha256": config["execution_control"]["sha256"],
        "engineering_correction_control_sha256": config["engineering_correction"]["sha256"],
        "engineering_correction_ordinal_2_control_sha256": config[
            "engineering_correction_ordinal_2"
        ]["sha256"],
        "engineering_correction_ordinal_3_control_sha256": config[
            "engineering_correction_ordinal_3"
        ]["sha256"],
        "engineering_correction_ordinal_4_control_sha256": config[
            "engineering_correction_ordinal_4"
        ]["sha256"],
        "corrected_target_files": ordinal_1_target_records,
        "ordinal_2_corrected_target_files": ordinal_2_target_records,
        "ordinal_3_corrected_target_files": ordinal_3_target_records,
        "ordinal_4_corrected_target_files": ordinal_4_target_records,
        "ordinal_4_exposure_adapter_commitments": {
            "adapter_manifest_root_sha256": exposure[
                "adapter_manifest_root_sha256"
            ],
            "legacy_error_set_root_sha256": exposure[
                "legacy_error_set_root_sha256"
            ],
            "legacy_success_projection_root_sha256": exposure[
                "legacy_success_projection_root_sha256"
            ],
            "new_projection_root_sha256": exposure[
                "new_projection_root_sha256"
            ],
            "csv_schema_plan_root_sha256": exposure[
                "csv_schema_plan_root_sha256"
            ],
            "exposed_architecture_set_sha256": exposure[
                "exposed_architecture_fingerprint_set_sha256"
            ],
        },
    }
    assets: dict[str, bytes] = {
        "protocol_amendment.json": _json_asset(amendment_record),
        "environment.json": _json_asset(environment),
        "phase_a_integrity.json": _json_asset(phase_a),
        "input_hashes.json": _json_asset(input_hashes),
        "layernorm_exposure_registry.json": _json_asset(exposure),
        "safety_architecture_selection.json": _json_asset(selection),
        "hp_screening_freeze.json": _json_asset(hp_freeze),
        "seed_collision_audit.json": _json_asset(seed_freeze),
        "test_firewall_prelaunch.json": _json_asset(firewall),
        "task_freeze.json": _json_asset(tasks),
        "pre_execution_estimate.json": _json_asset(runtime),
        "prelaunch_state.json": _json_asset(
            {
                "state": "DRY_RUN_COMPLETE_ZERO_GNN",
                "gnn_evaluations": 0,
                "gpu_training_seconds": 0,
                "test_accessed": False,
                "formal_started": False,
                "other_phases_started": [],
            }
        ),
        "source_snapshot/phase_c_v2_protocol_amendment.md": config["_amendment_bytes"],
    }
    for relative in TARGET_RELS:
        assets[f"source_snapshot/{relative}"] = (REPO_ROOT / relative).read_bytes()
    return assets


def write_prelaunch(result_root: Path, assets: Mapping[str, bytes]) -> dict[str, Any]:
    if result_root.exists() or result_root.is_symlink():
        _fail("WRITE_SCOPE", "RESULT_TARGET_ALREADY_EXISTS")
    for relative, data in assets.items():
        if not isinstance(relative, str) or not isinstance(data, bytes):
            _fail("WRITE_SCOPE", "PRELAUNCH_ASSET_PATH_OR_TYPE_FORBIDDEN")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or relative.startswith(("launch/", "formal/"))
        ):
            _fail("WRITE_SCOPE", "PRELAUNCH_ASSET_PATH_OR_TYPE_FORBIDDEN")
    result_root.mkdir(parents=False, exist_ok=False)
    records = []
    for relative in sorted(assets):
        data = assets[relative]
        _write_exclusive(result_root / relative, data)
        records.append({"path": relative, "size": len(data), "sha256": sha256_bytes(data)})
    manifest = {
        "schema_version": "phase-c-v2-prelaunch-manifest-v2",
        "immutable_prelaunch_file_count": len(records),
        "files": records,
        "authorized_later_namespaces": {
            "launch/": {
                "fixed_paths": [
                    "launch/launch_request.json",
                    "launch/runner_entered.json",
                    "launch/preformal_failure.json",
                    "launch/runner_exit.json",
                ],
                "schemas": [
                    "phase-c-v2-launch-request-v1",
                    "phase-c-v2-runner-entered-v1",
                    "phase-c-v2-preformal-failure-v1",
                    "phase-c-v2-runner-exit-v1",
                ],
            },
            "formal/": {
                "fixed_prefix": "formal/",
                "requires_bound_start": True,
                "manifest_schema": "phase-c-v2-formal-manifest-v2",
            },
        },
        "state_machine": [
            "CORRECTION_AUTHORIZED",
            "ZERO_GNN_GATES_PASSED",
            "DRY_RUN_PASSED",
            "FORMAL_FROZEN",
            "RUNNING",
            "TERMINAL",
        ],
        "fixed_tmux_command_sha256": sha256_bytes(FIXED_TMUX_COMMAND.encode("utf-8")),
        "gnn_evaluations_at_manifest": 0,
        "test_accessed": False,
    }
    manifest["manifest_root_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    _write_exclusive(result_root / "prelaunch_manifest.json", _json_asset(manifest))
    return manifest


def _observed_regular_file_tree(result_root: Path) -> set[str]:
    observed: set[str] = set()
    for directory, directory_names, file_names in os.walk(result_root, followlinks=False):
        current = Path(directory)
        for name in list(directory_names):
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                _fail("FORMAL_BINDING", "PRELAUNCH_SYMLINK_FORBIDDEN")
            if not stat.S_ISDIR(mode):
                _fail("FORMAL_BINDING", "PRELAUNCH_NON_DIRECTORY_TREE_ENTRY")
        for name in file_names:
            path = current / name
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                _fail("FORMAL_BINDING", "PRELAUNCH_NONREGULAR_FILE_FORBIDDEN")
            observed.add(path.relative_to(result_root).as_posix())
    return observed


def _formal_path_authorized(relative: str, *, allow_inflight_shard_tmp: bool = False) -> bool:
    fixed = {
        "formal/start.json",
        "formal/primary_results.json",
        "formal/screening_results.json",
        "formal/safety_results.json",
        "formal/task_attempt_ledger.json",
        "formal/test_access_ledger.json",
        "formal/decision.json",
        "formal/state.json",
        "formal/terminal_record.json",
        "formal/manifest.json",
        "formal/fixed-layernorm-enhancement-global4-v2.tar.gz",
        "formal/fixed-layernorm-enhancement-global4-v2.tar.gz.sha256",
        "formal/completion.json",
    }
    if relative in fixed:
        return True
    if allow_inflight_shard_tmp and re.fullmatch(
        r"formal/shards/(?:primary|screening|safety)_worker_[0-3]\.json\.tmp",
        relative,
    ):
        return True
    return bool(
        re.fullmatch(
            r"formal/(?:assignment_ledgers|stage_ledgers)/(?:primary|screening|safety)\.json",
            relative,
        )
        or re.fullmatch(
            r"formal/shards/(?:primary|screening|safety)_worker_[0-3]\.json",
            relative,
        )
    )


def verify_prelaunch(
    result_root: Path,
    *,
    allowed_launch_files: Iterable[str] = (),
    allow_formal_files: bool = False,
    allow_inflight_shard_tmp: bool = False,
) -> dict[str, Any]:
    if not result_root.is_dir() or result_root.is_symlink():
        _fail("FORMAL_BINDING", "PRELAUNCH_RESULT_ROOT_INVALID")
    manifest_path = result_root / "prelaunch_manifest.json"
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolFailure("FORMAL_BINDING", "PRELAUNCH_MANIFEST_LOAD_FAILED") from exc
    if manifest.get("schema_version") != "phase-c-v2-prelaunch-manifest-v2":
        _fail("FORMAL_BINDING", "PRELAUNCH_MANIFEST_SCHEMA_MISMATCH")
    namespaces = manifest.get("authorized_later_namespaces")
    if not isinstance(namespaces, dict) or set(namespaces) != {"launch/", "formal/"}:
        _fail("FORMAL_BINDING", "PRELAUNCH_NAMESPACE_AUTHORIZATION_MISMATCH")
    recorded_root = manifest.get("manifest_root_sha256")
    payload = dict(manifest)
    payload.pop("manifest_root_sha256", None)
    if recorded_root != sha256_bytes(canonical_json_bytes(payload)):
        _fail("FORMAL_BINDING", "PRELAUNCH_MANIFEST_ROOT_MISMATCH")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != manifest.get("immutable_prelaunch_file_count"):
        _fail("FORMAL_BINDING", "PRELAUNCH_MANIFEST_FILE_RECORD_MISMATCH")
    row_paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if (
        len(row_paths) != len(rows)
        or not all(isinstance(path, str) for path in row_paths)
        or len(set(row_paths)) != len(row_paths)
    ):
        _fail("FORMAL_BINDING", "PRELAUNCH_MANIFEST_FILE_RECORD_MISMATCH")
    for relative in row_paths:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative.startswith(("launch/", "formal/")):
            _fail("FORMAL_BINDING", "PRELAUNCH_MANIFEST_PATH_FORBIDDEN")
    expected_paths = set(row_paths) | {"prelaunch_manifest.json"}
    observed_paths = _observed_regular_file_tree(result_root)
    allowed_launch = set(allowed_launch_files)
    fixed_launch = set(namespaces["launch/"]["fixed_paths"])
    if not allowed_launch <= fixed_launch:
        _fail("FORMAL_BINDING", "PRELAUNCH_LAUNCH_PATH_NOT_AUTHORIZED")
    non_formal_observed = {path for path in observed_paths if not path.startswith("formal/")}
    if non_formal_observed != expected_paths | allowed_launch:
        _fail("FORMAL_BINDING", "PRELAUNCH_ALLOWLIST_MISMATCH")
    formal_observed = {path for path in observed_paths if path.startswith("formal/")}
    if formal_observed and not allow_formal_files:
        _fail("FORMAL_BINDING", "PRELAUNCH_FORMAL_NAMESPACE_PREMATURE")
    if allow_formal_files and any(
        not _formal_path_authorized(
            path, allow_inflight_shard_tmp=allow_inflight_shard_tmp
        )
        for path in formal_observed
    ):
        _fail("FORMAL_BINDING", "PRELAUNCH_FORMAL_PATH_NOT_AUTHORIZED")
    for row in rows:
        path = result_root / row["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or type(row.get("size")) is not int
            or not isinstance(row.get("sha256"), str)
            or not HEX64_RE.fullmatch(row["sha256"])
            or path.stat().st_size != row["size"]
            or sha256_path(path) != row["sha256"]
        ):
            _fail("FORMAL_BINDING", "PRELAUNCH_FILE_HASH_MISMATCH")
    return manifest


def dry_run(
    config_path: Path, *, seed_inventory_input: Path | None = None,
    seed_preflight_result: Path | None = None,
    seed_preflight_exec_attestation: Path | None = None,
    ordinal5_after_ledger: Path | None = None,
) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    config = load_config(config_path)
    result_root = REPO_ROOT / RESULT_REL
    if result_root.exists() or result_root.is_symlink():
        _fail("INITIAL_PREFLIGHT", "RESULT_TARGET_ALREADY_EXISTS")
    exposure = build_exposure_registry(config)
    exposure_set = set(exposure["exposed_architecture_fingerprints"])
    assert_frozen_safety_candidates_unexposed(config, exposure_set)
    dependencies = _dependency_audit(config)
    environment = audit_environment(config)
    phase_a = audit_phase_a(config)
    projected_primary = project_primary_table(config)
    source_join = audit_source_manifests(config)
    selection_raw = select_safety_architectures(
        projected_primary,
        exposure_set,
        config["base_p"]["architecture_fingerprint"],
        config["safety_selection_material"],
    )
    selection = verify_selection(config, selection_raw, source_join)
    hp_freeze = decode_hp_screening(config, selection)
    occupied_inventory = None
    if seed_inventory_input is not None:
        required = (
            seed_inventory_input, seed_preflight_result,
            seed_preflight_exec_attestation, ordinal5_after_ledger,
        )
        if any(path is None for path in required):
            _fail("SEED_INVENTORY", "ORDINAL5_TRUST_CHAIN_INCOMPLETE")
        for path in required:
            assert path is not None
            stat_result = path.lstat()
            if not path.is_file() or path.is_symlink() or stat.S_IMODE(stat_result.st_mode) != 0o600:
                _fail("SEED_INVENTORY", "ORDINAL5_IDENTITY_OR_PERMISSION_FAILURE")
        safe_result = json.loads(seed_preflight_result.read_text("utf-8"))
        execution = json.loads(seed_preflight_exec_attestation.read_text("utf-8"))
        if safe_result.get("status") != "PASS" or execution.get("status") != "PASS":
            _fail("SEED_INVENTORY", "ORDINAL5_PREFLIGHT_NOT_PASSED")
        occupied_inventory = json.loads(seed_inventory_input.read_text("utf-8"))
    seed_freeze = build_seed_freeze(config, occupied_inventory)
    firewall = synthetic_firewall_evidence()
    runtime = build_runtime_estimate(config, config["workers"]["count"])
    tasks = build_task_freeze(config, selection, hp_freeze, seed_freeze)
    validate_seed_task_binding(tasks, seed_freeze)
    validate_hp_task_binding(tasks, hp_freeze)
    assets = _prelaunch_assets(
        config,
        environment,
        phase_a,
        exposure,
        selection,
        hp_freeze,
        seed_freeze,
        firewall,
        runtime,
        tasks,
        dependencies,
    )
    manifest = write_prelaunch(result_root, assets)
    verified = verify_prelaunch(result_root)
    if verified["manifest_root_sha256"] != manifest["manifest_root_sha256"]:
        _fail("DRY_RUN", "PRELAUNCH_MANIFEST_POSTWRITE_MISMATCH")
    return {
        "status": "PASS",
        "stage": "DRY_RUN_COMPLETE_ZERO_GNN",
        "prelaunch_file_count": manifest["immutable_prelaunch_file_count"] + 1,
        "candidate_count": 4,
        "hp_configuration_count": 20,
        "selected_seed_count": 45,
        "selected_seed_set_sha256": seed_freeze["selected_seed_set_sha256"],
        "worker_count": config["workers"]["count"],
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "target_after_sha256": {
            relative: sha256_path(REPO_ROOT / relative) for relative in TARGET_RELS
        },
        "pair_assignment": {
            "primary_worker_pairs": tasks["assignment_ledgers"]["primary"]["worker_pair_counts"],
            "primary_gpu_pairs": tasks["assignment_ledgers"]["primary"]["gpu_pair_counts"],
            "screening_worker_tasks": tasks["assignment_ledgers"]["screening"]["worker_task_counts"],
            "safety_rule": tasks["assignment_ledgers"]["safety_rule"],
        },
        "gnn_evaluations": 0,
        "test_accessed": False,
    }


def verify_live_source_snapshot_binding(
    result_root: Path, manifest: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    if manifest is None:
        manifest = verify_prelaunch(
            result_root,
            allowed_launch_files={
                "launch/launch_request.json",
                "launch/runner_entered.json",
            },
            allow_formal_files=True,
        )
    rows = {row["path"]: row for row in manifest["files"]}
    evidence: dict[str, dict[str, Any]] = {}
    for relative in TARGET_RELS:
        snapshot_relative = f"source_snapshot/{relative}"
        record = rows.get(snapshot_relative)
        if record is None:
            _fail("FORMAL_BINDING", "SOURCE_SNAPSHOT_MANIFEST_ENTRY_MISSING")
        live = REPO_ROOT / relative
        snapshot = result_root / snapshot_relative
        if (
            not live.is_file()
            or live.is_symlink()
            or not snapshot.is_file()
            or snapshot.is_symlink()
        ):
            _fail("FORMAL_BINDING", "SOURCE_BINDING_REGULAR_FILE_MISMATCH")
        live_size = live.stat().st_size
        snapshot_size = snapshot.stat().st_size
        live_sha = sha256_path(live)
        snapshot_sha = sha256_path(snapshot)
        if (
            live_size != snapshot_size
            or live_size != record["size"]
            or live_sha != snapshot_sha
            or live_sha != record["sha256"]
            or live.read_bytes() != snapshot.read_bytes()
        ):
            _fail("FORMAL_BINDING", "LIVE_SOURCE_SNAPSHOT_BYTE_MISMATCH")
        evidence[relative] = {
            "size": live_size,
            "sha256": live_sha,
            "live_snapshot_bytes_equal": True,
        }
    return evidence


def _formal_bound_file_check(result_root: Path) -> dict[str, dict[str, Any]]:
    manifest = verify_prelaunch(
        result_root,
        allowed_launch_files={
            "launch/launch_request.json",
            "launch/runner_entered.json",
        },
        allow_formal_files=True,
        allow_inflight_shard_tmp=True,
    )
    return verify_live_source_snapshot_binding(result_root, manifest)


def _require_fixed_config_path(config_path: Path) -> Path:
    fixed = REPO_ROOT / CONFIG_REL
    try:
        observed = config_path.resolve(strict=True)
        expected = fixed.resolve(strict=True)
    except OSError as exc:
        raise ProtocolFailure("FORMAL_BINDING", "FIXED_CONFIG_PATH_UNRESOLVED") from exc
    if observed != expected or config_path.is_symlink() or not config_path.is_file():
        _fail("FORMAL_BINDING", "ALTERNATE_CONFIG_PATH_FORBIDDEN")
    return fixed


def _load_json_record(path: Path, stage: str, code: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        _fail(stage, code)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolFailure(stage, code) from exc
    if not isinstance(value, dict):
        _fail(stage, code)
    return value


def _load_launch_request(result_root: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = result_root / "launch/launch_request.json"
    request = _load_json_record(path, "LAUNCH_CONTROL", "LAUNCH_REQUEST_INVALID")
    request_sha = sha256_path(path)
    expected_after = {
        relative: sha256_path(REPO_ROOT / relative) for relative in TARGET_RELS
    }
    if (
        request.get("schema_version") != "phase-c-v2-launch-request-v1"
        or request.get("state") != "FORMAL_FROZEN"
        or request.get("prelaunch_manifest_root_sha256") != manifest["manifest_root_sha256"]
        or request.get("target_after_sha256") != expected_after
        or request.get("fixed_tmux_command_sha256")
        != sha256_bytes(FIXED_TMUX_COMMAND.encode("utf-8"))
        or request.get("fixed_tmux_command") != FIXED_TMUX_COMMAND
        or not isinstance(request.get("run_id"), str)
        or not request["run_id"].startswith("phase-c-v2-")
        or request.get("gnn_evaluations") != 0
        or request.get("gpu_training_seconds") != 0
        or request.get("test_accessed") is not False
    ):
        _fail("LAUNCH_CONTROL", "LAUNCH_REQUEST_BINDING_MISMATCH")
    return request, request_sha


def record_launch_request(config_path: Path) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    result_root = REPO_ROOT / RESULT_REL
    manifest = verify_prelaunch(result_root)
    verify_live_source_snapshot_binding(result_root, manifest)
    snapshot_config = load_config(result_root / f"source_snapshot/{CONFIG_REL}")
    if snapshot_config["launch_control"]["fixed_tmux_command"] != FIXED_TMUX_COMMAND:
        _fail("LAUNCH_CONTROL", "FIXED_TMUX_COMMAND_MISMATCH")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    request = {
        "schema_version": "phase-c-v2-launch-request-v1",
        "state": "FORMAL_FROZEN",
        "launch_requested_at": now.isoformat(),
        "timezone": "Asia/Shanghai",
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "target_after_sha256": {
            relative: sha256_path(REPO_ROOT / relative) for relative in TARGET_RELS
        },
        "fixed_tmux_command": FIXED_TMUX_COMMAND,
        "fixed_tmux_command_sha256": sha256_bytes(FIXED_TMUX_COMMAND.encode("utf-8")),
        "run_id": f"phase-c-v2-{uuid.uuid4()}",
        "gnn_evaluations": 0,
        "gpu_training_seconds": 0,
        "test_accessed": False,
    }
    path = result_root / "launch/launch_request.json"
    _write_exclusive(path, _json_asset(request))
    verified = verify_prelaunch(
        result_root, allowed_launch_files={"launch/launch_request.json"}
    )
    loaded, request_sha = _load_launch_request(result_root, verified)
    return {
        "status": "PASS",
        "stage": "LAUNCH_REQUEST_RECORDED_ZERO_GNN",
        "launch_requested_at": loaded["launch_requested_at"],
        "run_id": loaded["run_id"],
        "launch_request_sha256": request_sha,
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "gnn_evaluations": 0,
        "gpu_accessed": False,
        "test_accessed": False,
    }


def record_runner_entered(config_path: Path) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    result_root = REPO_ROOT / RESULT_REL
    manifest = verify_prelaunch(
        result_root, allowed_launch_files={"launch/launch_request.json"}
    )
    verify_live_source_snapshot_binding(result_root, manifest)
    request, request_sha = _load_launch_request(result_root, manifest)
    if (result_root / "formal").exists() or (result_root / "formal").is_symlink():
        _fail("LAUNCH_CONTROL", "FORMAL_NAMESPACE_ALREADY_EXISTS_AT_RUNNER_ENTRY")
    record = {
        "schema_version": "phase-c-v2-runner-entered-v1",
        "state": "RUNNER_ENTERED",
        "runner_entered_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "timezone": "Asia/Shanghai",
        "run_id": request["run_id"],
        "launch_request_sha256": request_sha,
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "gnn_evaluations_before_formal_admission": 0,
        "gpu_training_seconds_before_formal_admission": 0,
        "test_accessed": False,
    }
    _write_exclusive(result_root / "launch/runner_entered.json", _json_asset(record))
    verify_prelaunch(
        result_root,
        allowed_launch_files={"launch/launch_request.json", "launch/runner_entered.json"},
    )
    return {
        "status": "PASS",
        "stage": "RUNNER_ENTERED_ZERO_GNN",
        "run_id": request["run_id"],
        "gnn_evaluations": 0,
        "gpu_accessed": False,
        "test_accessed": False,
    }


def _verify_runner_entry(
    result_root: Path, manifest: Mapping[str, Any], request: Mapping[str, Any], request_sha: str
) -> dict[str, Any]:
    entry = _load_json_record(
        result_root / "launch/runner_entered.json", "LAUNCH_CONTROL", "RUNNER_ENTRY_INVALID"
    )
    if (
        entry.get("schema_version") != "phase-c-v2-runner-entered-v1"
        or entry.get("state") != "RUNNER_ENTERED"
        or entry.get("run_id") != request["run_id"]
        or entry.get("launch_request_sha256") != request_sha
        or entry.get("prelaunch_manifest_root_sha256") != manifest["manifest_root_sha256"]
        or entry.get("gnn_evaluations_before_formal_admission") != 0
        or entry.get("gpu_training_seconds_before_formal_admission") != 0
        or entry.get("test_accessed") is not False
    ):
        _fail("LAUNCH_CONTROL", "RUNNER_ENTRY_BINDING_MISMATCH")
    return entry


def record_runner_exit(config_path: Path) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    result_root = REPO_ROOT / RESULT_REL
    manifest = verify_prelaunch(
        result_root,
        allowed_launch_files={"launch/launch_request.json", "launch/runner_entered.json"},
        allow_formal_files=True,
    )
    request, request_sha = _load_launch_request(result_root, manifest)
    entry = _verify_runner_entry(result_root, manifest, request, request_sha)
    terminal_path = result_root / "formal/terminal_record.json"
    if terminal_path.is_file() and not terminal_path.is_symlink():
        terminal = _load_json_record(terminal_path, "LAUNCH_CONTROL", "FORMAL_TERMINAL_INVALID")
        sealed_assets = [
            result_root / "formal/manifest.json",
            result_root / "formal/fixed-layernorm-enhancement-global4-v2.tar.gz",
            result_root / "formal/fixed-layernorm-enhancement-global4-v2.tar.gz.sha256",
            result_root / "formal/completion.json",
        ]
        if terminal.get("state") in {"COMPLETE", "FAIL_CLOSED"} and all(
            path.is_file() and not path.is_symlink() for path in sealed_assets
        ):
            return {"status": "PASS", "stage": "RUNNER_EXIT_TERMINAL_ALREADY_DURABLE"}
    start_path = result_root / "formal/start.json"
    if not start_path.is_file():
        path = result_root / "launch/preformal_failure.json"
        schema = "phase-c-v2-preformal-failure-v1"
        code = "PREFORMAL_PROCESS_EXIT"
    else:
        path = result_root / "launch/runner_exit.json"
        schema = "phase-c-v2-runner-exit-v1"
        code = "RUNNER_EXIT_WITHOUT_FORMAL_TERMINAL"
    record = {
        "schema_version": schema,
        "state": "FAIL_CLOSED" if start_path.is_file() else "INDETERMINATE_LAUNCH_FAILURE",
        "error_code": code,
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "timezone": "Asia/Shanghai",
        "run_id": request["run_id"],
        "launch_request_sha256": request_sha,
        "runner_entered_sha256": sha256_path(result_root / "launch/runner_entered.json"),
        "scientific_decision": "NOT_REACHED",
        "classification": None,
        "phase_d_launched": False,
        "phase_e_launched": False,
    }
    _write_exclusive(path, _json_asset(record))
    return {"status": "PASS", "stage": f"{code}_RECORDED"}


def record_preformal_failure(config_path: Path) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    result_root = REPO_ROOT / RESULT_REL
    allowed = {"launch/launch_request.json"}
    if (result_root / "launch/runner_entered.json").is_file():
        allowed.add("launch/runner_entered.json")
    manifest = verify_prelaunch(result_root, allowed_launch_files=allowed)
    request_path = result_root / "launch/launch_request.json"
    request = _load_json_record(
        request_path, "LAUNCH_CONTROL", "LAUNCH_REQUEST_INVALID_FOR_FAILURE_RECORD"
    )
    if (
        request.get("schema_version") != "phase-c-v2-launch-request-v1"
        or request.get("prelaunch_manifest_root_sha256") != manifest["manifest_root_sha256"]
        or request.get("fixed_tmux_command_sha256")
        != sha256_bytes(FIXED_TMUX_COMMAND.encode("utf-8"))
        or not isinstance(request.get("run_id"), str)
    ):
        _fail("LAUNCH_CONTROL", "LAUNCH_REQUEST_INVALID_FOR_FAILURE_RECORD")
    record = {
        "schema_version": "phase-c-v2-preformal-failure-v1",
        "state": "FAIL_CLOSED"
        if "launch/runner_entered.json" in allowed
        else "INDETERMINATE_LAUNCH_FAILURE",
        "error_code": "RUNNER_ENTRY_OR_PREFORMAL_ADMISSION_FAILED",
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "timezone": "Asia/Shanghai",
        "run_id": request["run_id"],
        "launch_request_sha256": sha256_path(request_path),
        "runner_entered_sha256": (
            sha256_path(result_root / "launch/runner_entered.json")
            if "launch/runner_entered.json" in allowed
            else None
        ),
        "scientific_decision": "NOT_REACHED",
        "classification": None,
        "phase_d_launched": False,
        "phase_e_launched": False,
    }
    _write_exclusive(result_root / "launch/preformal_failure.json", _json_asset(record))
    return {"status": "PASS", "stage": "PREFORMAL_FAILURE_RECORDED"}


def _task_transient(exc: BaseException) -> bool:
    return type(exc).__name__ in {"InfrastructureEvaluationError", "OSError", "TimeoutError"}


class _AllowedTensorBundle:
    def __init__(self, values: Mapping[str, Any]):
        for key in DATA_ALLOWLIST:
            object.__setattr__(self, key, values[key])


def _formal_worker(payload: Mapping[str, Any]) -> None:
    worker_id = int(payload["worker_id"])
    gpu_index = int(payload["gpu_index"])
    gpu_uuid = str(payload["gpu_uuid"])
    result_root = Path(payload["result_root"])
    shard_path = Path(payload["shard_path"])
    tasks = list(payload["tasks"])
    retry_counter = payload["retry_counter"]
    abort_event = payload["abort_event"]
    retry_limit = int(payload["retry_limit"])
    shard: dict[str, Any] = {
        "worker_id": worker_id,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "status": "PASS",
        "binding_complete": False,
        "gpu_binding_complete": False,
        "setup_complete": False,
        "ledger_complete": False,
        "results": [],
        "attempts": [],
        "setup_test_access": None,
        "test_access": _new_access_ledger(),
    }
    try:
        source_binding = _formal_bound_file_check(result_root)
        config = load_config(result_root / f"source_snapshot/{CONFIG_REL}")
        gpu_binding = audit_worker_gpu_binding(config, gpu_index, gpu_uuid)
        shard["source_binding"] = source_binding
        shard["binding_complete"] = True
        shard["gpu_binding"] = gpu_binding
        shard["gpu_binding_complete"] = True
        assert_evaluator_ast_contract()
        import torch
        from torch_geometric.datasets import Planetoid

        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
        dataset = Planetoid(root=str(REPO_ROOT / "Cora"), name="Cora")
        loaded_data = dataset[0]
        setup_ledger = _new_access_ledger()
        setup_proxy = ValidationOnlyDataProxy(loaded_data, setup_ledger)
        tensors = {key: getattr(setup_proxy, key).to(device) for key in sorted(DATA_ALLOWLIST)}
        if setup_ledger["forbidden_request_count"] or setup_ledger["test_accessed"]:
            _fail("FORMAL_WORKER", "DATA_SETUP_FIREWALL_VIOLATION")
        bundle = _AllowedTensorBundle(tensors)
        shard["setup_test_access"] = setup_ledger
        shard["setup_complete"] = True
        shard["test_access"] = _aggregate_access_ledgers([setup_ledger])
        _write_formal_json(shard_path, shard)
        for frozen_task in tasks:
            if abort_event.is_set():
                break
            task = dict(frozen_task)
            if (
                int(task["worker_id"]) != worker_id
                or int(task["gpu_index"]) != gpu_index
                or str(task["gpu_uuid"]) != gpu_uuid
            ):
                _fail("FORMAL_WORKER", "TASK_WORKER_GPU_IDENTITY_MISMATCH")
            task["in_ch"] = int(dataset.num_features)
            task["out_ch"] = int(dataset.num_classes)
            completed = False
            for attempt in range(1, 4):
                ledger = _new_access_ledger()
                proxy = ValidationOnlyDataProxy(bundle, ledger)
                identity = {
                    "task_id": task["task_id"],
                    "stage": task["stage"],
                    "pair_id": task.get("pair_id"),
                    "worker_id": worker_id,
                    "gpu_index": gpu_index,
                    "gpu_uuid": gpu_uuid,
                    "architecture_fingerprint": task["architecture_fingerprint"],
                    "joint_fingerprint": task.get("joint_fingerprint"),
                    "hp_fingerprint": task.get("hp_fingerprint"),
                    "seed": task["seed"],
                    "component_variant": task["component_variant"],
                }
                attempt_record = {
                    **identity,
                    "attempt": attempt,
                    "status": "IN_PROGRESS",
                    "test_access": ledger,
                }
                shard["attempts"].append(attempt_record)
                shard["test_access"] = _aggregate_access_ledgers(
                    [setup_ledger] + [row["test_access"] for row in shard["attempts"]]
                )
                _write_formal_json(shard_path, shard)
                started = time.monotonic()
                try:
                    validation, valid, metadata = run_guarded_evaluation(task, proxy, device)
                    elapsed = time.monotonic() - started
                    if not valid:
                        abort_event.set()
                        attempt_record.update(
                            {"status": "NON_TRANSIENT_EVALUATOR_INVALID", "elapsed_seconds": elapsed}
                        )
                        _write_formal_json(shard_path, shard)
                        break
                    if ledger["forbidden_request_count"] or ledger["test_accessed"]:
                        abort_event.set()
                        attempt_record.update(
                            {"status": "TEST_FIREWALL_VIOLATION", "elapsed_seconds": elapsed}
                        )
                        _write_formal_json(shard_path, shard)
                        break
                    attempt_record.update({"status": "PASS", "elapsed_seconds": elapsed})
                    shard["results"].append(
                        {
                            **identity,
                            "source": task.get("source"),
                            "point": task.get("point"),
                            "validation_accuracy": float(validation),
                            "best_epoch": metadata.get("best_epoch"),
                            "epochs_ran": metadata.get("epochs_ran"),
                            "attempt_count": attempt,
                            "test_accessed": bool(ledger["test_accessed"]),
                        }
                    )
                    shard["test_access"] = _aggregate_access_ledgers(
                        [setup_ledger] + [row["test_access"] for row in shard["attempts"]]
                    )
                    _write_formal_json(shard_path, shard)
                    completed = True
                    break
                except Exception as exc:
                    elapsed = time.monotonic() - started
                    transient = _task_transient(exc)
                    attempt_record.update(
                        {
                            "status": "TRANSIENT_FAILURE" if transient else "NON_TRANSIENT_FAILURE",
                            "exception_class": type(exc).__name__,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    shard["test_access"] = _aggregate_access_ledgers(
                        [setup_ledger] + [row["test_access"] for row in shard["attempts"]]
                    )
                    _write_formal_json(shard_path, shard)
                    if not transient or attempt == 3:
                        abort_event.set()
                        break
                    over_limit = False
                    if attempt == 1:
                        with retry_counter.get_lock():
                            retry_counter.value += 1
                            over_limit = retry_counter.value > retry_limit
                    if over_limit:
                        abort_event.set()
                        break
            if not completed:
                shard["status"] = "FAIL_CLOSED"
                abort_event.set()
                break
        shard["test_access"] = _aggregate_access_ledgers(
            [setup_ledger] + [row["test_access"] for row in shard["attempts"]]
        )
        shard["ledger_complete"] = all(row["status"] != "IN_PROGRESS" for row in shard["attempts"])
    except BaseException as exc:
        shard["status"] = "FAIL_CLOSED"
        shard["error_code"] = "FORMAL_WORKER_TOP_LEVEL_FAILURE"
        shard["exception_class"] = type(exc).__name__
        shard["ledger_complete"] = False
        abort_event.set()
    if shard["binding_complete"] and shard["gpu_binding_complete"]:
        _write_formal_json(shard_path, shard)


def _run_formal_stage(
    stage: str,
    tasks: Sequence[Mapping[str, Any]],
    result_root: Path,
    worker_mapping: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    matched_pairs = stage in {"primary", "safety"}
    reassigned, assignment_ledger = assign_stage_tasks(
        stage, tasks, worker_mapping, matched_pairs=matched_pairs
    )
    expected_assignment = {
        row["task_id"]: (row["pair_id"], row["worker_id"], row["gpu_index"], row["gpu_uuid"])
        for row in reassigned
    }
    observed_assignment = {
        row["task_id"]: (
            row.get("pair_id"),
            row.get("worker_id"),
            row.get("gpu_index"),
            row.get("gpu_uuid"),
        )
        for row in tasks
    }
    if observed_assignment != expected_assignment:
        _fail("FORMAL_STAGE", "FROZEN_TASK_ASSIGNMENT_MISMATCH")
    _write_formal_json(
        result_root / "formal" / "assignment_ledgers" / f"{stage}.json",
        assignment_ledger,
    )
    context = multiprocessing.get_context("spawn")
    retry_counter = context.Value("i", 0)
    abort_event = context.Event()
    retry_limit = math.floor(0.05 * len(tasks))
    processes = []
    shard_paths = []
    for mapping in worker_mapping:
        worker_id = int(mapping["worker_id"])
        shard_path = result_root / "formal" / "shards" / f"{stage}_worker_{worker_id}.json"
        shard_paths.append(shard_path)
        shard_tasks = [task for task in tasks if int(task["worker_id"]) == worker_id]
        process = context.Process(
            target=_formal_worker,
            args=(
                {
                    "worker_id": worker_id,
                    "gpu_index": mapping["gpu_index"],
                    "gpu_uuid": mapping["gpu_uuid"],
                    "result_root": str(result_root),
                    "shard_path": str(shard_path),
                    "tasks": shard_tasks,
                    "retry_counter": retry_counter,
                    "abort_event": abort_event,
                    "retry_limit": retry_limit,
                },
            ),
            name=f"phase_c_v2_{stage}_worker_{worker_id}",
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    if any(process.exitcode != 0 for process in processes) or any(not path.is_file() for path in shard_paths):
        _fail("FORMAL_STAGE", "FORMAL_WORKER_PROCESS_FAILURE")
    shards = [
        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
        for path in shard_paths
    ]
    if any(shard["status"] != "PASS" for shard in shards) or abort_event.is_set():
        _fail("FORMAL_STAGE", "FORMAL_STAGE_FAIL_CLOSED")
    results = [row for shard in shards for row in shard["results"]]
    attempts = [row for shard in shards for row in shard["attempts"]]
    if len(results) != len(tasks) or {row["task_id"] for row in results} != {row["task_id"] for row in tasks}:
        _fail("FORMAL_STAGE", "FORMAL_RESULT_COMPLETENESS_MISMATCH")
    expected_result_identity = {
        row["task_id"]: (
            row.get("pair_id"),
            row["worker_id"],
            row["gpu_index"],
            row["gpu_uuid"],
        )
        for row in tasks
    }
    observed_result_identity = {
        row["task_id"]: (
            row.get("pair_id"),
            row["worker_id"],
            row["gpu_index"],
            row["gpu_uuid"],
        )
        for row in results
    }
    if observed_result_identity != expected_result_identity:
        _fail("FORMAL_STAGE", "FORMAL_RESULT_ASSIGNMENT_IDENTITY_MISMATCH")
    retried_tasks = len({row["task_id"] for row in attempts if row["attempt"] > 1})
    if retried_tasks / len(tasks) > 0.05:
        _fail("FORMAL_STAGE", "FORMAL_RETRY_FRACTION_EXCEEDED")
    ledger = {
        "stage": stage,
        "task_count": len(tasks),
        "attempt_count": len(attempts),
        "retried_task_count": retried_tasks,
        "retry_fraction": retried_tasks / len(tasks),
        "setup_ledgers": [shard["setup_test_access"] for shard in shards],
        "worker_ledgers_complete": all(shard["ledger_complete"] for shard in shards),
        "test_accessed": any(shard["test_access"]["test_accessed"] for shard in shards),
        "forbidden_attribute_request_count": sum(
            shard["test_access"]["forbidden_request_count"] for shard in shards
        ),
        "allowed_attribute_request_count": sum(
            shard["test_access"]["allowed_request_count"] for shard in shards
        ),
        "allowed_attribute_counts": {
            key: sum(shard["test_access"]["allowed_attribute_counts"][key] for shard in shards)
            for key in sorted(DATA_ALLOWLIST)
        },
    }
    if (
        not ledger["worker_ledgers_complete"]
        or ledger["test_accessed"]
        or ledger["forbidden_attribute_request_count"]
    ):
        _fail("FORMAL_STAGE", "FORMAL_TEST_FIREWALL_VIOLATION")
    _write_formal_json(
        result_root / "formal" / "stage_ledgers" / f"{stage}.json", ledger
    )
    return results, attempts, ledger


def _rng_integer(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % (2**63)


def _primary_statistics(results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    by_seed: dict[int, dict[str, float]] = {}
    for row in results:
        by_seed.setdefault(int(row["seed"]), {})[str(row["component_variant"])] = float(
            row["validation_accuracy"]
        )
    if len(by_seed) != 30 or any(set(values) != {"baseline", "layer_norm"} for values in by_seed.values()):
        _fail("PRIMARY_STATISTICS", "PRIMARY_PAIR_COMPLETENESS_MISMATCH")
    ordered_seeds = sorted(by_seed)
    deltas = np.asarray(
        [by_seed[seed]["layer_norm"] - by_seed[seed]["baseline"] for seed in ordered_seeds],
        dtype=np.float64,
    )
    observed = float(deltas.mean())
    rng = np.random.default_rng(_rng_integer(config["inference"]["signflip_seed_material"]))
    exceed = 0
    draws = int(config["inference"]["signflip_draws"])
    batch = 10000
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        signs = rng.integers(0, 2, size=(count, len(deltas)), dtype=np.int8) * 2 - 1
        exceed += int(np.count_nonzero((signs * deltas).mean(axis=1) >= observed))
    p_value = (1 + exceed) / (1 + draws)
    bootstrap_rng = np.random.default_rng(_rng_integer(config["inference"]["bootstrap_seed_material"]))
    replicates = int(config["inference"]["bootstrap_replicates"])
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch):
        count = min(batch, replicates - start)
        indices = bootstrap_rng.integers(0, len(deltas), size=(count, len(deltas)))
        means[start : start + count] = deltas[indices].mean(axis=1)
    lower = float(np.quantile(means, 0.05, method="linear"))
    classification = classify_primary(observed, float(p_value), lower)
    return {
        "paired_seed_count": 30,
        "mean_delta": observed,
        "one_sided_signflip_p_value": float(p_value),
        "one_sided_paired_bootstrap_95_lower_bound": lower,
        "classification": classification,
        "signflip_draws": draws,
        "bootstrap_replicates": replicates,
    }


def _screening_selection(
    results: Sequence[Mapping[str, Any]], hp_freeze: Mapping[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in results:
        grouped.setdefault(str(row["joint_fingerprint"]), []).append(float(row["validation_accuracy"]))
    if len(grouped) != 20 or any(len(values) != 5 for values in grouped.values()):
        _fail("SCREENING_SELECTION", "SCREENING_COMPLETENESS_MISMATCH")
    configurations = {row["joint_fingerprint"]: row for row in hp_freeze["configurations"]}
    selected = []
    for architecture in sorted({row["architecture_fingerprint"] for row in configurations.values()}):
        candidates = []
        for joint, row in configurations.items():
            if row["architecture_fingerprint"] == architecture:
                candidates.append((sum(grouped[joint]) / 5.0, joint, row))
        best_mean = max(item[0] for item in candidates)
        tied = sorted((item for item in candidates if item[0] == best_mean), key=lambda item: item[1])
        mean, joint, row = tied[0]
        selected.append({**row, "screening_mean_validation": mean, "selected_joint_fingerprint": joint})
    if len(selected) != 4:
        _fail("SCREENING_SELECTION", "SCREENING_SELECTED_CARDINALITY_MISMATCH")
    return selected


def _safety_tasks(
    selected: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    worker_mapping: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    tasks = []
    for row in selected:
        for seed in seeds:
            for variant in ("baseline", "layer_norm"):
                tasks.append(
                    {
                        "task_id": f"safety|{row['selected_joint_fingerprint']}|{seed}|{variant}",
                        "stage": "safety",
                        "source": row["source"],
                        "architecture_fingerprint": row["architecture_fingerprint"],
                        "joint_fingerprint": row["selected_joint_fingerprint"],
                        "hp_fingerprint": row["hp_fingerprint"],
                        "architecture": {
                            "operations": row["operations"],
                            "edges": row["edges"],
                            "effective_layers": row["effective_layers"],
                        },
                        "hp": row["decoded"],
                        "seed": int(seed),
                        "component_variant": variant,
                        "max_epochs": 300,
                        "patience": 80,
                    }
                )
    if len(tasks) != 80:
        _fail("SAFETY_TASKS", "SAFETY_TASK_CARDINALITY_MISMATCH")
    assigned, _ = assign_stage_tasks(
        "safety", tasks, worker_mapping, matched_pairs=True
    )
    return assigned


def _safety_statistics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paired: dict[tuple[str, int], dict[str, float]] = {}
    sources: dict[str, str] = {}
    for row in results:
        architecture = str(row["architecture_fingerprint"])
        sources[architecture] = str(row["source"])
        paired.setdefault((architecture, int(row["seed"])), {})[
            str(row["component_variant"])
        ] = float(row["validation_accuracy"])
    if len(paired) != 40 or any(set(values) != {"baseline", "layer_norm"} for values in paired.values()):
        _fail("SAFETY_STATISTICS", "SAFETY_PAIR_COMPLETENESS_MISMATCH")
    architecture_deltas: dict[str, float] = {}
    for architecture in sorted(sources):
        deltas = [
            values["layer_norm"] - values["baseline"]
            for (candidate, _), values in paired.items()
            if candidate == architecture
        ]
        if len(deltas) != 10:
            _fail("SAFETY_STATISTICS", "SAFETY_ARCHITECTURE_SEED_COUNT_MISMATCH")
        architecture_deltas[architecture] = sum(deltas) / 10.0
    passed = safety_pass(architecture_deltas, sources)
    source_means = {
        source: sum(architecture_deltas[arch] for arch in architecture_deltas if sources[arch] == source) / 2.0
        for source in sorted(set(sources.values()))
    }
    return {
        "architecture_mean_deltas": architecture_deltas,
        "positive_architecture_count": sum(value > 0.0 for value in architecture_deltas.values()),
        "equal_weight_architecture_mean_delta": sum(architecture_deltas.values()) / 4.0,
        "minimum_architecture_mean_delta": min(architecture_deltas.values()),
        "source_equal_weight_mean_deltas": source_means,
        "safety_pass": passed,
    }


def _runtime_acceptance_nodes(
    started: datetime, estimate: Mapping[str, Any]
) -> dict[str, Any]:
    output = {}
    for label, key in (("minimum_60_gnn_branch", "minimum_60_gnn_branch"), ("maximum_240_gnn_branch", "maximum_240_gnn_branch")):
        branch = estimate[key]
        output[label] = {
            "durations_seconds": {
                "best": branch["best_seconds"],
                "expected": branch["expected_seconds"],
                "worst": branch["worst_seconds"],
            },
            "absolute_acceptance_nodes": {
                name: (started + timedelta(seconds=float(branch[f"{name}_seconds"]))).isoformat()
                for name in ("best", "expected", "worst")
            },
        }
    return output


def _write_formal_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_asset(value))
    os.replace(temporary, path)


def _create_formal_with_atomic_start(formal_dir: Path, start_record: Mapping[str, Any]) -> None:
    if formal_dir.exists() or formal_dir.is_symlink():
        _fail("FORMAL_START", "FORMAL_DIRECTORY_ALREADY_EXISTS")
    pending = Path(
        tempfile.mkdtemp(prefix="phase_c_v2_formal_start_", dir=formal_dir.parent)
    )
    renamed = False
    try:
        _write_exclusive(pending / "start.json", _json_asset(start_record))
        os.rename(pending, formal_dir)
        renamed = True
    except OSError as exc:
        raise ProtocolFailure("FORMAL_START", "ATOMIC_FORMAL_START_FAILED") from exc
    finally:
        if not renamed and pending.exists():
            shutil.rmtree(pending)


def _derive_terminal_test_status(result_root: Path) -> dict[str, Any]:
    assignments = sorted((result_root / "formal/assignment_ledgers").glob("*.json"))
    shards = sorted((result_root / "formal/shards").glob("*.json"))
    completed_tasks = 0
    attempt_count = 0
    detected = False
    complete = True
    expected_shards: set[str] = set()
    for path in assignments:
        try:
            ledger = _load_json_record(path, "FORMAL_TERMINAL", "ASSIGNMENT_LEDGER_INVALID")
            stage = str(ledger["stage"])
            workers = {int(row["worker_id"]) for row in ledger["tasks"]}
            expected_shards.update(f"{stage}_worker_{worker}.json" for worker in workers)
        except (ProtocolFailure, KeyError, TypeError, ValueError):
            complete = False
    observed_shards = {path.name for path in shards}
    if observed_shards != expected_shards:
        complete = False
    attempt_rows: list[dict[str, Any]] = []
    for path in shards:
        try:
            shard = _load_json_record(path, "FORMAL_TERMINAL", "WORKER_SHARD_INVALID")
            if shard.get("ledger_complete") is not True:
                complete = False
            access = shard["test_access"]
            detected = bool(detected or access["test_accessed"])
            completed_tasks += len(shard.get("results", []))
            attempt_rows.extend(shard.get("attempts", []))
        except (ProtocolFailure, KeyError, TypeError, ValueError):
            complete = False
    attempt_count = len(attempt_rows)
    if detected:
        test_accessed: bool | None = True
        status = "DETECTED"
    elif complete:
        test_accessed = False
        status = "VERIFIED_COMPLETE"
    else:
        test_accessed = None
        status = "UNVERIFIED_FAIL_CLOSED"
    return {
        "test_accessed": test_accessed,
        "test_access_status": status,
        "ledger_complete": complete,
        "completed_task_count": completed_tasks,
        "attempt_count": attempt_count,
        "attempts": attempt_rows,
        "assignment_ledger_count": len(assignments),
        "worker_shard_count": len(shards),
    }


def _archive_record(path: Path, result_root: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        _fail("FORMAL_ARCHIVE", "ARCHIVE_SOURCE_NOT_REGULAR")
    return {
        "path": path.relative_to(result_root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def _verify_formal_launch_start_chain(
    result_root: Path, prelaunch: Mapping[str, Any]
) -> None:
    request, request_sha = _load_launch_request(result_root, prelaunch)
    _verify_runner_entry(result_root, prelaunch, request, request_sha)
    start = _load_json_record(
        result_root / "formal/start.json", "FORMAL_ARCHIVE", "FORMAL_START_INVALID"
    )
    if (
        start.get("state") != "FORMAL_STARTED"
        or start.get("run_id") != request["run_id"]
        or start.get("launch_request_sha256") != request_sha
        or start.get("runner_entered_sha256")
        != sha256_path(result_root / "launch/runner_entered.json")
        or start.get("prelaunch_manifest_root_sha256") != prelaunch["manifest_root_sha256"]
    ):
        _fail("FORMAL_ARCHIVE", "FORMAL_START_LAUNCH_BINDING_MISMATCH")


def _finalize_formal_archive(result_root: Path) -> dict[str, Any]:
    formal = result_root / "formal"
    prelaunch = verify_prelaunch(
        result_root,
        allowed_launch_files={"launch/launch_request.json", "launch/runner_entered.json"},
        allow_formal_files=True,
    )
    _verify_formal_launch_start_chain(result_root, prelaunch)
    excluded = {
        "formal/manifest.json",
        "formal/fixed-layernorm-enhancement-global4-v2.tar.gz",
        "formal/fixed-layernorm-enhancement-global4-v2.tar.gz.sha256",
        "formal/completion.json",
    }
    formal_records: list[dict[str, Any]] = []
    for path in sorted(formal.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(result_root).as_posix()
        if relative in excluded or relative.endswith(".tmp"):
            continue
        formal_records.append(_archive_record(path, result_root))
    for relative in ("launch/launch_request.json", "launch/runner_entered.json"):
        formal_records.append(_archive_record(result_root / relative, result_root))
    if len({row["path"] for row in formal_records}) != len(formal_records):
        _fail("FORMAL_ARCHIVE", "FORMAL_MANIFEST_DUPLICATE_PATH")
    required = {
        "formal/start.json",
        "formal/decision.json",
        "formal/state.json",
        "formal/task_attempt_ledger.json",
        "formal/test_access_ledger.json",
        "formal/terminal_record.json",
        "launch/launch_request.json",
        "launch/runner_entered.json",
    }
    if not required <= {row["path"] for row in formal_records}:
        _fail("FORMAL_ARCHIVE", "FORMAL_AUTHORITATIVE_ASSET_MISSING")
    formal_records.sort(key=lambda row: row["path"])
    manifest = {
        "schema_version": "phase-c-v2-formal-manifest-v2",
        "files": formal_records,
    }
    manifest["manifest_root_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    manifest_path = formal / "manifest.json"
    _write_exclusive(manifest_path, _json_asset(manifest))
    prelaunch_paths = {"prelaunch_manifest.json"} | {
        row["path"] for row in prelaunch["files"]
    }
    formal_paths = {"formal/manifest.json"} | {row["path"] for row in formal_records}
    expected_paths = prelaunch_paths | formal_paths
    if prelaunch_paths & formal_paths:
        _fail("FORMAL_ARCHIVE", "ARCHIVE_MANIFEST_UNION_OVERLAP")
    archive_path = formal / "fixed-layernorm-enhancement-global4-v2.tar.gz"
    prefix = "fixed-layernorm-enhancement-global4-v2"
    with tarfile.open(archive_path, mode="x:gz") as archive:
        for relative in sorted(expected_paths):
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                _fail("FORMAL_ARCHIVE", "ARCHIVE_MEMBER_PATH_FORBIDDEN")
            path = result_root / relative
            if not path.is_file() or path.is_symlink():
                _fail("FORMAL_ARCHIVE", "ARCHIVE_SOURCE_NOT_REGULAR")
            archive.add(path, arcname=f"{prefix}/{relative}", recursive=False)
    expected_records = {
        row["path"]: {"path": row["path"], "size": row["size"], "sha256": row["sha256"]}
        for row in prelaunch["files"]
    }
    expected_records.update(
        {
            row["path"]: {
                "path": row["path"],
                "size": row["size"],
                "sha256": row["sha256"],
            }
            for row in formal_records
        }
    )
    expected_records["prelaunch_manifest.json"] = _archive_record(
        result_root / "prelaunch_manifest.json", result_root
    )
    expected_records["formal/manifest.json"] = _archive_record(manifest_path, result_root)
    if set(expected_records) != expected_paths:
        _fail("FORMAL_ARCHIVE", "ARCHIVE_MANIFEST_RECORD_UNION_MISMATCH")
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            expected_names = {f"{prefix}/{relative}" for relative in expected_paths}
            if len(names) != len(set(names)) or set(names) != expected_names:
                _fail("FORMAL_ARCHIVE", "ARCHIVE_MEMBER_SET_MISMATCH")
            for member in members:
                if not member.isfile():
                    _fail("FORMAL_ARCHIVE", "ARCHIVE_MEMBER_NOT_REGULAR")
                relative = member.name[len(prefix) + 1 :]
                record = expected_records[relative]
                handle = archive.extractfile(member)
                if handle is None:
                    _fail("FORMAL_ARCHIVE", "ARCHIVE_MEMBER_UNREADABLE")
                data = handle.read()
                if len(data) != record["size"] or sha256_bytes(data) != record["sha256"]:
                    _fail("FORMAL_ARCHIVE", "ARCHIVE_MEMBER_IDENTITY_MISMATCH")
    except tarfile.TarError as exc:
        raise ProtocolFailure("FORMAL_ARCHIVE", "ARCHIVE_REOPEN_FAILED") from exc
    archive_sha = sha256_path(archive_path)
    sidecar = formal / "fixed-layernorm-enhancement-global4-v2.tar.gz.sha256"
    _write_exclusive(
        sidecar,
        f"{archive_sha}  fixed-layernorm-enhancement-global4-v2.tar.gz\n".encode("ascii"),
    )
    return {
        "archive_sha256": archive_sha,
        "formal_manifest_root_sha256": manifest["manifest_root_sha256"],
        "prelaunch_manifest_root_sha256": prelaunch["manifest_root_sha256"],
        "archive_member_count": len(expected_paths),
        "archive_member_set_verified": True,
        "tar_prefix": prefix,
    }


def _write_outer_completion(
    result_root: Path, state: str, archive: Mapping[str, Any]
) -> None:
    formal = result_root / "formal"
    terminal_sha = sha256_path(formal / "terminal_record.json")
    _write_exclusive(
        formal / "completion.json",
        _json_asset(
            {
                "schema_version": "phase-c-v2-non-authoritative-completion-envelope-v1",
                "state": state,
                "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "non_authoritative_outer_envelope": True,
                "scientific_acceptance_chain": (
                    "outer SHA -> archive -> formal manifest -> decision/state/terminal_record"
                ),
                "archive_sha256": archive["archive_sha256"],
                "formal_manifest_root_sha256": archive["formal_manifest_root_sha256"],
                "terminal_record_sha256": terminal_sha,
            }
        ),
    )


def _write_fail_closed_terminal(
    result_root: Path, exc: BaseException
) -> dict[str, Any]:
    formal = result_root / "formal"
    if not (formal / "start.json").is_file():
        _fail("FORMAL_TERMINAL", "FAIL_CLOSED_START_RECORD_MISSING")
    derived = _derive_terminal_test_status(result_root)
    stage = exc.stage if isinstance(exc, ProtocolFailure) else "FORMAL_EXECUTION"
    code = exc.error_code if isinstance(exc, ProtocolFailure) else "FORMAL_UNCLASSIFIED_FAILURE"
    safe_class = type(exc).__name__
    _write_formal_json(formal / "task_attempt_ledger.json", {"attempts": derived.pop("attempts")})
    test_ledger = {
        "schema_version": "phase-c-v2-formal-test-access-ledger-v2",
        **derived,
    }
    _write_formal_json(formal / "test_access_ledger.json", test_ledger)
    decision = {
        "state": "FAIL_CLOSED",
        "scientific_decision": "NOT_REACHED",
        "classification": None,
        "completed_task_count": test_ledger["completed_task_count"],
        "attempt_count": test_ledger["attempt_count"],
        "test_accessed": test_ledger["test_accessed"],
        "test_access_status": test_ledger["test_access_status"],
        "phase_d_launched": False,
        "phase_e_launched": False,
    }
    state = {
        **decision,
        "failure_stage": stage,
        "failure_code": code,
        "failure_class": safe_class,
    }
    _write_formal_json(formal / "decision.json", decision)
    _write_formal_json(formal / "state.json", state)
    terminal = {
        "schema_version": "phase-c-v2-formal-terminal-record-v2",
        **state,
        "decision_sha256": sha256_path(formal / "decision.json"),
        "state_sha256": sha256_path(formal / "state.json"),
        "task_attempt_ledger_sha256": sha256_path(formal / "task_attempt_ledger.json"),
        "test_access_ledger_sha256": sha256_path(formal / "test_access_ledger.json"),
    }
    _write_formal_json(formal / "terminal_record.json", terminal)
    archive = _finalize_formal_archive(result_root)
    _write_outer_completion(result_root, "FAIL_CLOSED", archive)
    return {
        "status": "FAIL_CLOSED",
        "stage": stage,
        "error_code": code,
        "classification": None,
        "scientific_decision": "NOT_REACHED",
        "test_accessed": test_ledger["test_accessed"],
        "test_access_status": test_ledger["test_access_status"],
    }


def formal_run(config_path: Path) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    result_root = REPO_ROOT / RESULT_REL
    manifest = verify_prelaunch(
        result_root,
        allowed_launch_files={"launch/launch_request.json", "launch/runner_entered.json"},
    )
    source_binding = verify_live_source_snapshot_binding(result_root, manifest)
    config = load_config(result_root / f"source_snapshot/{CONFIG_REL}")
    _dependency_audit(config)
    assert_evaluator_ast_contract()
    branch = _run_readonly(["git", "branch", "--show-current"])
    head = _run_readonly(["git", "rev-parse", "HEAD"])
    status = _run_readonly(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    if (
        branch.returncode
        or head.returncode
        or status.returncode
        or branch.stdout.strip() != config["repository"]["branch"]
        or head.stdout.strip() != config["repository"]["head"]
        or sha256_bytes(status.stdout.encode("utf-8"))
        != config["engineering_correction"]["dirty_status_porcelain_z_sha256"]
    ):
        _fail("FORMAL_ADMISSION", "REPOSITORY_IDENTITY_MISMATCH")
    tmux = _run_readonly(["tmux", "has-session", "-t", TMUX_NAME])
    if tmux.returncode != 0 or not os.environ.get("TMUX"):
        _fail("FORMAL_ADMISSION", "DETACHED_TMUX_CONTEXT_REQUIRED")
    gpu_admission = audit_gpu_admission(config, require_zero_compute_processes=True)
    request, request_sha = _load_launch_request(result_root, manifest)
    _verify_runner_entry(result_root, manifest, request, request_sha)
    task_freeze = _load_json_record(
        result_root / "task_freeze.json", "FORMAL_BINDING", "TASK_FREEZE_INVALID"
    )
    hp_freeze = _load_json_record(
        result_root / "hp_screening_freeze.json", "FORMAL_BINDING", "HP_FREEZE_INVALID"
    )
    seed_freeze = _load_json_record(
        result_root / "seed_collision_audit.json", "FORMAL_BINDING", "SEED_FREEZE_INVALID"
    )
    estimate = _load_json_record(
        result_root / "pre_execution_estimate.json", "FORMAL_BINDING", "RUNTIME_ESTIMATE_INVALID"
    )
    validate_task_freeze_contract(config, task_freeze)
    validate_seed_task_binding(task_freeze, seed_freeze)
    validate_hp_task_binding(task_freeze, hp_freeze)
    formal_dir = result_root / "formal"
    if formal_dir.exists() or formal_dir.is_symlink():
        _fail("FORMAL_START", "FORMAL_DIRECTORY_ALREADY_EXISTS")
    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    start_record = {
        "state": "FORMAL_STARTED",
        "started_at": started.isoformat(),
        "timezone": "Asia/Shanghai",
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "launch_request_sha256": request_sha,
        "run_id": request["run_id"],
        "runner_entered_sha256": sha256_path(result_root / "launch/runner_entered.json"),
        "runtime_acceptance": _runtime_acceptance_nodes(started, estimate),
        "worker_count": config["workers"]["count"],
        "worker_mapping": config["workers"]["mapping"],
        "source_live_snapshot_binding": source_binding,
        "gpu_admission": gpu_admission,
        "test_accessed_at_start": False,
        "other_phases_started": [],
    }
    _create_formal_with_atomic_start(formal_dir, start_record)
    try:
        ledgers: list[dict[str, Any]] = []
        primary_results, _, primary_ledger = _run_formal_stage(
            "primary", task_freeze["primary_tasks"], result_root, config["workers"]["mapping"]
        )
        ledgers.append(primary_ledger)
        primary_stats = _primary_statistics(primary_results, config)
        _write_formal_json(
            formal_dir / "primary_results.json",
            {
                "results": primary_results,
                "statistics": primary_stats,
                "test_accessed": primary_ledger["test_accessed"],
            },
        )
        classification = primary_stats["classification"]
        completed_gnn = 60
        safety_stats: dict[str, Any] | None = None
        if classification == "FIXED_LN_STRONG_SIGNAL":
            screening_results, _, screening_ledger = _run_formal_stage(
                "screening",
                task_freeze["screening_tasks"],
                result_root,
                config["workers"]["mapping"],
            )
            ledgers.append(screening_ledger)
            completed_gnn += 100
            selected_screening = _screening_selection(screening_results, hp_freeze)
            _write_formal_json(
                formal_dir / "screening_results.json",
                {
                    "results": screening_results,
                    "selected_configurations": selected_screening,
                    "test_accessed": screening_ledger["test_accessed"],
                },
            )
            safety_tasks = _safety_tasks(
                selected_screening,
                seed_freeze["safety_seeds"],
                config["workers"]["mapping"],
            )
            safety_results, _, safety_ledger = _run_formal_stage(
                "safety", safety_tasks, result_root, config["workers"]["mapping"]
            )
            ledgers.append(safety_ledger)
            completed_gnn += 80
            safety_stats = _safety_statistics(safety_results)
            _write_formal_json(
                formal_dir / "safety_results.json",
                {
                    "results": safety_results,
                    "statistics": safety_stats,
                    "test_accessed": safety_ledger["test_accessed"],
                },
            )
            classification = (
                "FIXED_LN_RECIPE_AUTHORIZED_FOR_PHASE_D_TESTING"
                if safety_stats["safety_pass"]
                else "FIXED_LN_STRONG_SIGNAL_SAFETY_NOT_PASSED"
            )
        if completed_gnn not in {60, 240}:
            _fail("FORMAL_ACCOUNTING", "FORMAL_GNN_BUDGET_MISMATCH")
        derived = _derive_terminal_test_status(result_root)
        attempts = derived.pop("attempts")
        if (
            derived["ledger_complete"] is not True
            or derived["test_accessed"] is not False
            or derived["completed_task_count"] != completed_gnn
        ):
            _fail("FORMAL_FIREWALL", "FORMAL_LEDGER_COMPLETENESS_MISMATCH")
        test_ledger = {
            "schema_version": "phase-c-v2-formal-test-access-ledger-v2",
            **derived,
            "stage_ledgers": ledgers,
            "negative_controls": "test_firewall_prelaunch.json",
        }
        _write_formal_json(formal_dir / "task_attempt_ledger.json", {"attempts": attempts})
        _write_formal_json(formal_dir / "test_access_ledger.json", test_ledger)
        decision = {
            "state": "COMPLETE",
            "classification": classification,
            "scientific_decision": "REACHED",
            "primary_classification": primary_stats["classification"],
            "safety_pass": None if safety_stats is None else safety_stats["safety_pass"],
            "completed_planned_gnn_evaluations": completed_gnn,
            "minimum_budget": 60,
            "conditional_screening_budget": 100,
            "conditional_safety_budget": 80,
            "maximum_budget": 240,
            "test_accessed": test_ledger["test_accessed"],
            "test_access_status": test_ledger["test_access_status"],
            "phase_d_launched": False,
            "phase_e_launched": False,
        }
        state = {
            "state": "COMPLETE",
            "classification": classification,
            "scientific_decision": "REACHED",
            "completed_planned_gnn_evaluations": completed_gnn,
            "test_accessed": test_ledger["test_accessed"],
            "test_access_status": test_ledger["test_access_status"],
            "other_phases_started": [],
        }
        _write_formal_json(formal_dir / "decision.json", decision)
        _write_formal_json(formal_dir / "state.json", state)
        terminal = {
            "schema_version": "phase-c-v2-formal-terminal-record-v2",
            **state,
            "decision_sha256": sha256_path(formal_dir / "decision.json"),
            "state_sha256": sha256_path(formal_dir / "state.json"),
            "task_attempt_ledger_sha256": sha256_path(formal_dir / "task_attempt_ledger.json"),
            "test_access_ledger_sha256": sha256_path(formal_dir / "test_access_ledger.json"),
        }
        _write_formal_json(formal_dir / "terminal_record.json", terminal)
        archive = _finalize_formal_archive(result_root)
        _write_outer_completion(result_root, "COMPLETE", archive)
        return {
            "status": "PASS",
            "stage": "FORMAL_COMPLETE",
            "classification": classification,
            "completed_planned_gnn_evaluations": completed_gnn,
            "test_accessed": test_ledger["test_accessed"],
        }
    except BaseException as exc:
        return _write_fail_closed_terminal(result_root, exc)


def synthetic_smoke(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    _dependency_audit(config)
    projection = synthetic_projection_invariance()
    firewall = synthetic_firewall_evidence()
    base = config["base_p"]
    if canonical_architecture_fingerprint(base["operations"], base["edges"]) != base[
        "architecture_fingerprint"
    ]:
        _fail("SYNTHETIC_SMOKE", "BASE_ARCHITECTURE_FINGERPRINT_MISMATCH")
    if classify_primary(0.002, 0.01, 0.001) != "FIXED_LN_AUXILIARY_SIGNAL":
        _fail("SYNTHETIC_SMOKE", "CONDITIONAL_STOP_RULE_MISMATCH")
    if budget_for_primary_classification("FIXED_LN_AUXILIARY_SIGNAL") != 60:
        _fail("SYNTHETIC_SMOKE", "AUXILIARY_BUDGET_MISMATCH")
    return {
        "status": "PASS",
        "stage": "SYNTHETIC_SMOKE_ZERO_GNN",
        "projector": projection,
        "firewall": firewall,
        "gnn_evaluations": 0,
        "gpu_accessed": False,
        "test_accessed": False,
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
            "test_access_status": "UNVERIFIED_FAIL_CLOSED",
        }
    return {
        "status": "FAIL_CLOSED",
        "stage": "TOP_LEVEL",
        "error_code": "UNCLASSIFIED_TOP_LEVEL_FAILURE",
        "exception_class": type(exc).__name__,
        "gnn_start_authorized": False,
        "test_accessed": None,
        "test_access_status": "UNVERIFIED_FAIL_CLOSED",
    }


def _write_exclusive_0600_json(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def conservative_seed_preflight(
    config_path: Path, *, inventory_output: Path, result_output: Path,
    request_path: Path, after_ledger_path: Path,
) -> dict[str, Any]:
    _require_fixed_config_path(config_path)
    config = load_config(config_path)
    scanner_path = REPO_ROOT / "analyse/final_checkpoint_seed_inventory.py"
    assert_file_identity(
        scanner_path, config["dependency_hashes"]["analyse/final_checkpoint_seed_inventory.py"],
    )
    if not after_ledger_path.is_file() or after_ledger_path.is_symlink():
        _fail("CONSERVATIVE_SEED_PREFLIGHT", "IDENTITY_OR_PERMISSION_FAILURE")
    scanner_spec = importlib.util.spec_from_file_location(
        "phase_c_v2_conservative_seed_inventory", scanner_path,
    )
    if scanner_spec is None or scanner_spec.loader is None:
        _fail("CONSERVATIVE_SEED_PREFLIGHT", "PREFLIGHT_PROCESS_FAILED")
    scanner = importlib.util.module_from_spec(scanner_spec)
    sys.modules[scanner_spec.name] = scanner
    scanner_spec.loader.exec_module(scanner)
    request = {
        "schema_version": "phase-c-v2-ordinal5-seed-preflight-request-v1",
        "control_id": "phase-c-v2-conservative-historical-seed-occupancy-ordinal-5-2026-08-31",
        "control_sha256": "53d72aa6ea6a18450c555fadea45d50535291626cbbe5c6cccb9975dabadd8e4",
        "amendment_sha256": "9ca1bf15fc86fe8294970f56e84b7b086bc6ef6ef4a73f6ab92fbfe32b4146c7",
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "after_ledger": {"path": str(after_ledger_path), "size_bytes": after_ledger_path.stat().st_size,
                         "sha256": sha256_path(after_ledger_path)},
        "zero_state": {"result_prefix_absent": not (REPO_ROOT / RESULT_REL).exists(),
                       "persistent_fresh_seed_freeze_completed": False,
                       "fresh_seed_consumption": 0, "gnn_evaluations": 0},
    }
    _write_exclusive_0600_json(request_path, request)
    inventory = scanner.build_conservative_seed_collision_superset(REPO_ROOT)
    inventory.update({
        "control_id": request["control_id"], "control_sha256": request["control_sha256"],
        "amendment_sha256": request["amendment_sha256"],
        "request": {"path": str(request_path), "size_bytes": request_path.stat().st_size,
                    "sha256": sha256_path(request_path)},
        "after_ledger": request["after_ledger"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _write_exclusive_0600_json(inventory_output, inventory)
    safe_commitments = {
        key: inventory[key] for key in (
            "schema_version", "source_count", "test_skip", "set_commitments",
            "set_proofs", "old_inventory_root_sha256",
        )
    }
    result = {
        "schema_version": "phase-c-v2-ordinal5-seed-preflight-result-v1",
        "control_id": request["control_id"], "control_sha256": request["control_sha256"],
        "amendment_sha256": request["amendment_sha256"], "status": "PASS",
        "stage": "CONSERVATIVE_SEED_PREFLIGHT", "error_code": "NONE",
        "request": inventory["request"], "after_ledger": request["after_ledger"],
        "inventory_artifact": {"path": str(inventory_output), "present": True,
                               "complete": True, "size_bytes": inventory_output.stat().st_size,
                               "sha256": sha256_path(inventory_output)},
        "commitments": safe_commitments,
        "prepared_exit_code": 0,
        "prepared_stdout_sha256": sha256_bytes(b"PHASE_C_V2_CONSERVATIVE_SEED_PREFLIGHT PASS\n"),
        "result_written_at_utc": datetime.now(timezone.utc).isoformat(),
        "zero_state": {"persistent_fresh_seed_freeze_completed": False,
                       "fresh_seed_consumption": 0, "task_count": 0,
                       "gnn_evaluations": 0, "test_accessed": False,
                       "test_access_status": "VERIFIED_NOT_ACCESSED"},
    }
    _write_exclusive_0600_json(result_output, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / CONFIG_REL)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--synthetic-projector-firewall", action="store_true")
    modes.add_argument("--synthetic-smoke", action="store_true")
    modes.add_argument("--conservative-seed-preflight", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--record-launch-request", action="store_true")
    modes.add_argument("--runner-enter", action="store_true")
    modes.add_argument("--runner-exit", action="store_true")
    modes.add_argument("--record-preformal-failure", action="store_true")
    modes.add_argument("--formal-run", action="store_true")
    parser.add_argument("--seed-inventory-output", type=Path)
    parser.add_argument("--seed-preflight-result", type=Path)
    parser.add_argument("--seed-preflight-request", type=Path)
    parser.add_argument("--seed-inventory-input", type=Path)
    parser.add_argument("--seed-preflight-exec-attestation", type=Path)
    parser.add_argument("--ordinal5-after-ledger", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.synthetic_projector_firewall:
            load_config(args.config)
            output = {
                "status": "PASS",
                "stage": "SYNTHETIC_PROJECTOR_FIREWALL",
                "projector": synthetic_projection_invariance(),
                "firewall": synthetic_firewall_evidence(),
                "gnn_evaluations": 0,
                "gpu_accessed": False,
                "test_accessed": False,
            }
        elif args.synthetic_smoke:
            output = synthetic_smoke(args.config)
        elif args.conservative_seed_preflight:
            output = conservative_seed_preflight(
                args.config, inventory_output=args.seed_inventory_output,
                result_output=args.seed_preflight_result,
                request_path=args.seed_preflight_request,
                after_ledger_path=args.ordinal5_after_ledger,
            )
            print("PHASE_C_V2_CONSERVATIVE_SEED_PREFLIGHT PASS")
            return 0
        elif args.dry_run:
            output = dry_run(
                args.config, seed_inventory_input=args.seed_inventory_input,
                seed_preflight_result=args.seed_preflight_result,
                seed_preflight_exec_attestation=args.seed_preflight_exec_attestation,
                ordinal5_after_ledger=args.ordinal5_after_ledger,
            )
        elif args.record_launch_request:
            output = record_launch_request(args.config)
        elif args.runner_enter:
            output = record_runner_entered(args.config)
        elif args.runner_exit:
            output = record_runner_exit(args.config)
        elif args.record_preformal_failure:
            output = record_preformal_failure(args.config)
        else:
            output = formal_run(args.config)
    except BaseException as exc:
        print(json.dumps(_safe_failure(exc), sort_keys=True))
        return 2
    print(json.dumps(output, sort_keys=True))
    return 0 if output.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
