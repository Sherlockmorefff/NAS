from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "analyse/fixed_layernorm_enhancement_v2.py"
CONFIG_PATH = REPO_ROOT / "configs/fixed_layernorm_enhancement_global4_v2.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("fixed_layernorm_enhancement_v2", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def phase_c():
    return _load_module()


@pytest.fixture(scope="module")
def config(phase_c):
    return phase_c.load_config(CONFIG_PATH)


def _control_payloads():
    return {
        "decision.json": {
            "classification": "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL",
            "boundary": False,
            "training": False,
            "training_sufficient": True,
            "new_gnn_runs": 0,
            "gpu_compute_runs": 0,
            "test_accessed": False,
            "phase_b_started": False,
            "mean_validation": 0.987,
        },
        "state.json": {
            "status": "COMPLETE",
            "classification": "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL",
            "gpu_compute_used": False,
            "phase_b_started": False,
            "test_accessed": False,
            "accuracy": 0.876,
        },
        "training_audit.json": {
            "status": "TRAINING_AUDIT_SUFFICIENT",
            "sufficient": True,
            "raw_training_truncation_signal": False,
            "training_truncation_signal": False,
            "loss": 123.0,
        },
        "test_firewall.json": {
            "passed": True,
            "gnn_training_calls": 0,
            "gpu_compute_calls": 0,
            "test_data_reads": 0,
            "runner_exact_command_verified": True,
            "test_accuracy": 0.999,
        },
    }


def _real_manifest_records(phase_c):
    return [
        {
            "relative_path": relative_path,
            "size_bytes": index,
            "sha256": hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
        }
        for index, relative_path in enumerate(
            phase_c.PHASE_A_MANIFEST_SCHEMA_CONTRACT["relative_paths"], start=1
        )
    ]


def _real_manifest_payload(phase_c, *, records=None, root_updates=None):
    controls = copy.deepcopy(
        phase_c.PHASE_A_MANIFEST_SCHEMA_CONTRACT["safe_root_controls"]
    )
    payload = {
        "archive_excluded_from_manifest": controls[
            "archive_excluded_from_manifest"
        ],
        "archive_self_inclusion": controls["archive_self_inclusion"],
        "created_at": "2026-08-30T00:00:00+08:00",
        "experiment": controls["experiment"],
        "file_count": controls["file_count"],
        "files": copy.deepcopy(
            _real_manifest_records(phase_c) if records is None else records
        ),
        "manifest_root_sha256": controls["manifest_root_sha256"],
    }
    if root_updates:
        payload.update(root_updates)
    return payload


def _project_real_manifest(phase_c, payload):
    return phase_c.project_archive_manifest_records(
        phase_c._json_asset(payload), phase_c.PHASE_A_MANIFEST_SCHEMA_CONTRACT
    )


def _synthetic_phase_a_repo(phase_c, root: Path):
    result = root / "phase_a"
    result.mkdir(parents=True)
    prefix = "phase-a"
    primary = b"architecture_fingerprint,source\n"
    (result / "primary.csv").write_bytes(primary)
    controls = {
        name: phase_c._json_asset(value) for name, value in _control_payloads().items()
    }
    control_contract = copy.deepcopy(phase_c.PHASE_A_CONTROL_CONTRACT)
    for name, data in controls.items():
        control_contract["files"][name]["size_bytes"] = len(data)
        control_contract["files"][name]["sha256"] = phase_c.sha256_bytes(data)
    for name, data in controls.items():
        (result / name).write_bytes(data)
    payloads = {
        relative_path: f"synthetic:{relative_path}\n".encode("utf-8")
        for relative_path in phase_c.PHASE_A_MANIFEST_SCHEMA_CONTRACT[
            "relative_paths"
        ]
    }
    payloads.update(controls)
    manifest_rows = [
        {
            "relative_path": relative_path,
            "size_bytes": len(payloads[relative_path]),
            "sha256": phase_c.sha256_bytes(payloads[relative_path]),
        }
        for relative_path in phase_c.PHASE_A_MANIFEST_SCHEMA_CONTRACT[
            "relative_paths"
        ]
    ]
    manifest = phase_c._json_asset(
        _real_manifest_payload(phase_c, records=manifest_rows)
    )
    (result / "manifest.json").write_bytes(manifest)
    archive = result / "phase-a.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        members = {
            f"{prefix}/primary.csv": primary,
            f"{prefix}/manifest.json": manifest,
            **{f"{prefix}/{name}": data for name, data in payloads.items()},
        }
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    archive_sha = phase_c.sha256_path(archive)
    (result / "phase-a.tar.gz.sha256").write_text(
        f"{archive_sha}  phase-a.tar.gz\n", encoding="ascii"
    )
    binding = {
        "primary_table": "phase_a/primary.csv",
        "primary_sha256": phase_c.sha256_bytes(primary),
        "primary_size": len(primary),
        "archive": "phase_a/phase-a.tar.gz",
        "archive_sha256": archive_sha,
        "primary_archive_member": f"{prefix}/primary.csv",
        "manifest_identity": {
            "path": "phase_a/manifest.json",
            "size_bytes": len(manifest),
            "sha256": phase_c.sha256_bytes(manifest),
        },
        "manifest_schema": copy.deepcopy(
            phase_c.PHASE_A_MANIFEST_SCHEMA_CONTRACT
        ),
        "control_contract": control_contract,
    }
    return result, binding


def _synthetic_panel(config):
    selected = []
    configurations = []
    for candidate_index, row in enumerate(config["safety_selection_expected"]["selected"]):
        candidate = {
            **row,
            "operations": ["GCNConv", "SAGEConv"],
            "edges": [[0, 1], [1, 2], [2, 3]],
            "effective_layers": 2,
        }
        selected.append(candidate)
        for point in range(1, 6):
            configurations.append(
                {
                    "source": row["source"],
                    "architecture_fingerprint": row["architecture_fingerprint"],
                    "operations": candidate["operations"],
                    "edges": candidate["edges"],
                    "effective_layers": 2,
                    "point": point,
                    "raw": [0.8 + point / 100] * 4,
                    "decoded": {
                        "lr": 0.01 + point / 10000,
                        "dropout": 0.5,
                        "hidden_dim": 64,
                        "weight_decay": 0.001,
                        "gat_heads": 1,
                        "sage_aggr": "mean",
                        "gin_eps": 0.0,
                        "gat_heads_by_layer": None,
                        "sage_aggr_by_layer": None,
                        "gin_eps_by_layer": None,
                    },
                    "hp_fingerprint": hashlib.sha256(f"hp-{candidate_index}-{point}".encode()).hexdigest(),
                    "joint_fingerprint": hashlib.sha256(
                        f"joint-{candidate_index}-{point}".encode()
                    ).hexdigest(),
                }
            )
    return {"selected": selected}, {"configurations": configurations}


def _seed_freeze():
    return {
        "primary_seeds": list(range(1000, 1030)),
        "screening_seeds": list(range(1030, 1035)),
        "safety_seeds": list(range(1035, 1045)),
    }


def _synthetic_csv_adapter_contract(phase_c, header):
    header_sha = phase_c.ordinal_4_commitment_sha256(header)
    return {
        header_sha: {
            "columns": len(header),
            "files": 1,
            "rows": 1,
            "identity_columns": sorted(
                set(header) & set(phase_c.IDENTITY_WHITELIST)
            ),
        }
    }


def _synthetic_csv_bytes(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _synthetic_architecture_record(phase_c, index, *, explicit=True):
    first_ops = ["GCNConv", "GATConv", "GCNII", "GIN", "GraphConv", "ChebConv"]
    operations = [first_ops[index % len(first_ops)], "SAGEConv"]
    edges = [[0, 1], [1, 2], [2, 3]]
    fingerprint = phase_c.canonical_architecture_fingerprint(operations, edges)
    record = {"operations": operations, "edges": edges}
    if explicit:
        record["architecture_fingerprint"] = fingerprint
    return record, fingerprint


def _synthetic_exact_json_fixture(phase_c, file_index):
    contract = copy.deepcopy(
        phase_c.EXACT_HISTORICAL_JSON_ADAPTER_CONTRACT["files"][file_index]
    )
    payload = {
        "dataset_identity": {"edges": 17},
        "outcomes": {
            "validation": 0.91,
            "accuracy": 0.92,
            "loss": 0.1,
            "predictor_prediction": 0.93,
            "test": 0.94,
        },
    }
    fingerprints = []
    if file_index == 0:
        new_combinations = []
        for index in range(2):
            record, fingerprint = _synthetic_architecture_record(
                phase_c, index, explicit=True
            )
            new_combinations.append(record)
            fingerprints.append(fingerprint)
        comparison = {}
        for key, index in (
            ("diverse_truncation_stress_1", 0),
            ("median_1", 1),
            ("top_1", 2),
        ):
            record, fingerprint = _synthetic_architecture_record(
                phase_c, index, explicit=False
            )
            comparison[key] = record
            fingerprints.append(fingerprint)
        payload["new_combinations"] = new_combinations
        payload["structure_comparison"] = comparison
    elif file_index in {1, 2}:
        count = 5 if file_index == 1 else 6
        selected = []
        for index in range(count):
            record, fingerprint = _synthetic_architecture_record(
                phase_c, index, explicit=True
            )
            if file_index == 1:
                record["architecture_summary"] = {
                    "operations": list(record["operations"]),
                    "validation": 100 + index,
                }
            selected.append(record)
            fingerprints.append(fingerprint)
        payload["selected_candidates"] = selected
    else:
        panel = []
        for index in range(3):
            record, fingerprint = _synthetic_architecture_record(
                phase_c, index, explicit=True
            )
            panel.append(record)
            fingerprints.append(fingerprint)
        payload["candidate_panel"] = panel
        payload["baseline_provenance"] = {
            "baseline_rows": [
                {"architecture_fingerprint": fingerprints[index % 3]}
                for index in range(9)
            ]
        }
    identities = sorted(set(fingerprints))
    data = phase_c.canonical_json_bytes(payload)
    contract["path"] = f"synthetic/exact_adapter_{file_index}.json"
    contract["size"] = len(data)
    contract["sha256"] = phase_c.sha256_bytes(data)
    contract["expected_unique_architecture_count"] = len(identities)
    contract["identity_set_sha256"] = phase_c.ordinal_4_commitment_sha256(
        identities
    )
    return payload, data, contract, identities


def _rebind_synthetic_exact_contract(phase_c, contract, data):
    rebound = copy.deepcopy(contract)
    rebound["size"] = len(data)
    rebound["sha256"] = phase_c.sha256_bytes(data)
    return rebound


def _make_result_skeleton(phase_c, root: Path) -> Path:
    result = root / "result"
    manifest = phase_c.write_prelaunch(result, {"identity.json": b"{}\n"})
    request = {
        "schema_version": "phase-c-v2-launch-request-v1",
        "state": "FORMAL_FROZEN",
        "launch_requested_at": "2026-08-30T00:00:00+08:00",
        "timezone": "Asia/Shanghai",
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "target_after_sha256": {
            relative: phase_c.sha256_path(phase_c.REPO_ROOT / relative)
            for relative in phase_c.TARGET_RELS
        },
        "fixed_tmux_command": phase_c.FIXED_TMUX_COMMAND,
        "fixed_tmux_command_sha256": phase_c.sha256_bytes(
            phase_c.FIXED_TMUX_COMMAND.encode()
        ),
        "run_id": "phase-c-v2-synthetic",
        "gnn_evaluations": 0,
        "gpu_training_seconds": 0,
        "test_accessed": False,
    }
    phase_c._write_exclusive(result / "launch/launch_request.json", phase_c._json_asset(request))
    request_sha = phase_c.sha256_path(result / "launch/launch_request.json")
    entry = {
        "schema_version": "phase-c-v2-runner-entered-v1",
        "state": "RUNNER_ENTERED",
        "runner_entered_at": "2026-08-30T00:00:01+08:00",
        "timezone": "Asia/Shanghai",
        "run_id": request["run_id"],
        "launch_request_sha256": request_sha,
        "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        "gnn_evaluations_before_formal_admission": 0,
        "gpu_training_seconds_before_formal_admission": 0,
        "test_accessed": False,
    }
    phase_c._write_exclusive(result / "launch/runner_entered.json", phase_c._json_asset(entry))
    (result / "formal").mkdir()
    phase_c._write_formal_json(
        result / "formal/start.json",
        {
            "state": "FORMAL_STARTED",
            "run_id": request["run_id"],
            "launch_request_sha256": request_sha,
            "runner_entered_sha256": phase_c.sha256_path(
                result / "launch/runner_entered.json"
            ),
            "prelaunch_manifest_root_sha256": manifest["manifest_root_sha256"],
        },
    )
    return result


def _assert_archive_union(phase_c, result: Path):
    prelaunch = json.loads((result / "prelaunch_manifest.json").read_text())
    formal = json.loads((result / "formal/manifest.json").read_text())
    for manifest in (prelaunch, formal):
        recorded_root = manifest.pop("manifest_root_sha256")
        assert recorded_root == phase_c.sha256_bytes(phase_c.canonical_json_bytes(manifest))
        manifest["manifest_root_sha256"] = recorded_root
    expected = {"prelaunch_manifest.json", "formal/manifest.json"}
    expected.update(row["path"] for row in prelaunch["files"])
    expected.update(row["path"] for row in formal["files"])
    archive_path = result / "formal/fixed-layernorm-enhancement-global4-v2.tar.gz"
    sidecar = (archive_path.with_name(archive_path.name + ".sha256")).read_text().split()[0]
    assert phase_c.sha256_path(archive_path) == sidecar
    with tarfile.open(archive_path, "r:gz") as handle:
        members = handle.getmembers()
        assert len(members) == len({row.name for row in members})
        assert all(row.isfile() for row in members)
        observed = {
            row.name.removeprefix("fixed-layernorm-enhancement-global4-v2/")
            for row in members
        }
        records = {row["path"]: row for row in prelaunch["files"] + formal["files"]}
        for member in members:
            relative = member.name.removeprefix("fixed-layernorm-enhancement-global4-v2/")
            if relative in records:
                data = handle.extractfile(member).read()
                assert len(data) == records[relative]["size"]
                assert phase_c.sha256_bytes(data) == records[relative]["sha256"]
    assert observed == expected
    assert "formal/completion.json" not in observed
    assert "formal/fixed-layernorm-enhancement-global4-v2.tar.gz" not in observed
    assert "launch/launch_request.json" in {row["path"] for row in formal["files"]}
    assert "launch/runner_entered.json" in {row["path"] for row in formal["files"]}


def test_scientific_amendment_and_controls_are_exact(phase_c, config):
    assert len(config["_amendment_bytes"]) == 21075
    assert phase_c.sha256_bytes(config["_amendment_bytes"]) == (
        "1d487f8bd5ddebc4a6b964f77975e3fc1931496ce4c800658b50cfa8640f4279"
    )
    assert config["execution_control"]["sha256"] == (
        "bac75381c9d3315529cac88d966e739cf41d7cfba1f98c66cfc2534fdb764237"
    )
    assert config["engineering_correction"]["sha256"] == phase_c.CORRECTION_CONTROL_SHA256
    assert config["engineering_correction"]["before_sha256"] == phase_c.EXPECTED_BEFORE_SHA256
    assert len(config["engineering_correction"]["zero_state_blocks"]) == 4
    ordinal_2 = config["engineering_correction_ordinal_2"]
    assert ordinal_2["control_id"] == phase_c.CORRECTION_ORDINAL_2_CONTROL_ID
    assert ordinal_2["sha256"] == phase_c.CORRECTION_ORDINAL_2_CONTROL_SHA256
    assert ordinal_2["before_file_identities"] == (
        phase_c.EXPECTED_ORDINAL_2_BEFORE_IDENTITIES
    )
    assert ordinal_2["phase_a_manifest_identity"] == (
        phase_c.PHASE_A_MANIFEST_IDENTITY_CONTRACT
    )
    assert ordinal_2["phase_a_manifest_schema"] == (
        phase_c.PHASE_A_MANIFEST_SCHEMA_CONTRACT
    )
    assert ordinal_2["ordinal_1_terminal"]["event"] == (
        "ENGINEERING_PHASE_A_MANIFEST_RECORD_SCHEMA_BINDING_BLOCK_ORDINAL_1"
    )
    assert ordinal_2["ordinal_1_terminal"]["scientific_decision"] == "NOT_REACHED"
    assert ordinal_2["ordinal_1_terminal"]["classification"] is None
    assert ordinal_2["zero_state_attestation"]["gnn_evaluations"] == 0
    assert ordinal_2["zero_state_attestation"]["fresh_seed_consumption"] == 0
    ordinal_3 = config["engineering_correction_ordinal_3"]
    assert ordinal_3["control_id"] == phase_c.CORRECTION_ORDINAL_3_CONTROL_ID
    assert ordinal_3["sha256"] == phase_c.CORRECTION_ORDINAL_3_CONTROL_SHA256
    assert ordinal_3["before_file_identities"] == (
        phase_c.EXPECTED_ORDINAL_3_BEFORE_IDENTITIES
    )
    assert ordinal_3["previous_engineering_terminals"] == {
        "ordinal_1": phase_c.ORDINAL_1_TERMINAL_CONTRACT,
        "ordinal_2": phase_c.ORDINAL_2_TERMINAL_CONTRACT,
    }
    assert ordinal_3["phase_a_control_contract"] == phase_c.PHASE_A_CONTROL_CONTRACT
    assert ordinal_3["zero_state_attestation"]["ordinal_2_real_dry_run_count"] == 1
    assert ordinal_3["zero_state_attestation"]["gnn_evaluations"] == 0
    assert ordinal_3["scientific_design_changed"] is False
    ordinal_4 = config["engineering_correction_ordinal_4"]
    assert ordinal_4["control_id"] == phase_c.CORRECTION_ORDINAL_4_CONTROL_ID
    assert ordinal_4["sha256"] == phase_c.CORRECTION_ORDINAL_4_CONTROL_SHA256
    assert ordinal_4["before_file_identities"] == (
        phase_c.EXPECTED_ORDINAL_4_BEFORE_IDENTITIES
    )
    assert ordinal_4["ordinal_3_terminal"] == phase_c.ORDINAL_3_TERMINAL_CONTRACT
    assert ordinal_4["ordinal_3_after_ledger_identity"] == (
        phase_c.ORDINAL_3_AFTER_LEDGER_IDENTITY
    )
    assert ordinal_4["adapter_plan_commitments"] == phase_c.EXPOSURE_ADAPTER_PLAN
    assert ordinal_4["zero_state_attestation"][
        "ordinal_3_focused_pytest_cases_passed"
    ] == 193
    assert ordinal_4["zero_state_attestation"]["gnn_evaluations"] == 0
    assert ordinal_4["scientific_design_changed"] is False
    assert config["exposure_adapter_plan"] == (
        phase_c.EXPOSURE_ADAPTER_CONFIG_CONTRACT
    )
    assert config["execution_control"]["incident"]["classification"] == (
        "PHASE_C_V2_PREEXECUTION_IDENTITY_ONLY_ACCESS_VIOLATION_ZERO_STATE"
    )


def test_ordinal_2_prelaunch_provenance_is_additive_and_non_self_referential(
    phase_c, config
):
    projections = {
        name: phase_c.project_phase_a_control_file(
            name, phase_c._json_asset(payload)
        )
        for name, payload in _control_payloads().items()
    }
    projection_evidence = phase_c.phase_a_control_projection_evidence(projections)
    assets = phase_c._prelaunch_assets(
        config,
        {"status": "PASS"},
        {
            "primary_sha256": config["phase_a"]["primary_sha256"],
            "archive_sha256": config["phase_a"]["archive_sha256"],
            "manifest_sha256": config["phase_a"]["manifest_identity"]["sha256"],
            "manifest_safe_projection_sha256": config["phase_a"]["manifest_schema"][
                "safe_projection_sha256"
            ],
            "control_projection_schema": phase_c.PHASE_A_CONTROL_PROJECTION_SCHEMA,
            "control_projection_identities": projection_evidence["files"],
            "control_projection_aggregate": projection_evidence["aggregate"],
        },
        {
            "tree_sha256": "0" * 64,
            "adapter_manifest_root_sha256": "1" * 64,
            "legacy_error_set_root_sha256": "2" * 64,
            "legacy_success_projection_root_sha256": "3" * 64,
            "new_projection_root_sha256": "4" * 64,
            "csv_schema_plan_root_sha256": "5" * 64,
            "exposed_architecture_fingerprint_set_sha256": "6" * 64,
        },
        {"status": "PASS"},
        {"status": "PASS"},
        {"scanner_inventory_root_sha256": "1" * 64},
        {"status": "PASS"},
        {"status": "PASS"},
        {"status": "PASS"},
        {},
    )
    protocol = json.loads(assets["protocol_amendment.json"])
    execution = protocol["execution_control"]
    assert execution["verified_sha256"] == config["execution_control"]["sha256"]
    assert execution["incident"] == config["execution_control"]["incident"]
    ordinal_1 = execution["engineering_correction"]
    ordinal_2 = execution["engineering_correction_ordinal_2"]
    ordinal_3 = execution["engineering_correction_ordinal_3"]
    ordinal_4 = execution["engineering_correction_ordinal_4"]
    assert ordinal_1["verified_sha256"] == phase_c.CORRECTION_CONTROL_SHA256
    assert ordinal_1["target_files"][phase_c.IMPLEMENTATION_REL]["after_sha256"] == (
        phase_c.EXPECTED_ORDINAL_2_BEFORE_IDENTITIES[phase_c.IMPLEMENTATION_REL][
            "sha256"
        ]
    )
    assert ordinal_2["verified_sha256"] == phase_c.CORRECTION_ORDINAL_2_CONTROL_SHA256
    for relative in phase_c.TARGET_RELS:
        assert ordinal_2["target_files"][relative]["after_sha256"] == (
            phase_c.EXPECTED_ORDINAL_3_BEFORE_IDENTITIES[relative]["sha256"]
        )
        assert ordinal_2["target_files"][relative]["after_size"] == (
            phase_c.EXPECTED_ORDINAL_3_BEFORE_IDENTITIES[relative]["size_bytes"]
        )
    assert ordinal_2["target_files"][phase_c.RUNNER_REL]["after_sha256"] == (
        "875bad60d0f3fc0a75153929ac216c659ae2d2e2190821cf074f84791a110ec4"
    )
    assert ordinal_3["verified_sha256"] == phase_c.CORRECTION_ORDINAL_3_CONTROL_SHA256
    assert ordinal_3["previous_engineering_terminals"]["ordinal_2"] == (
        phase_c.ORDINAL_2_TERMINAL_CONTRACT
    )
    assert ordinal_3["phase_a_control_contract"] == phase_c.PHASE_A_CONTROL_CONTRACT
    for relative in phase_c.TARGET_RELS:
        assert ordinal_3["target_files"][relative]["before_sha256"] == (
            phase_c.EXPECTED_ORDINAL_3_BEFORE_IDENTITIES[relative]["sha256"]
        )
        assert ordinal_3["target_files"][relative]["after_sha256"] == (
            phase_c.EXPECTED_ORDINAL_4_BEFORE_IDENTITIES[relative]["sha256"]
        )
        assert ordinal_3["target_files"][relative]["after_size"] == (
            phase_c.EXPECTED_ORDINAL_4_BEFORE_IDENTITIES[relative]["size_bytes"]
        )
    assert ordinal_3["target_files"][phase_c.RUNNER_REL]["after_sha256"] == (
        "875bad60d0f3fc0a75153929ac216c659ae2d2e2190821cf074f84791a110ec4"
    )
    assert ordinal_4["verified_sha256"] == phase_c.CORRECTION_ORDINAL_4_CONTROL_SHA256
    assert ordinal_4["ordinal_3_terminal"] == phase_c.ORDINAL_3_TERMINAL_CONTRACT
    assert ordinal_4["adapter_plan_commitments"] == phase_c.EXPOSURE_ADAPTER_PLAN
    for relative in phase_c.TARGET_RELS:
        assert ordinal_4["target_files"][relative]["before_sha256"] == (
            phase_c.EXPECTED_ORDINAL_4_BEFORE_IDENTITIES[relative]["sha256"]
        )
        assert ordinal_4["target_files"][relative]["after_sha256"] == (
            phase_c.sha256_path(phase_c.REPO_ROOT / relative)
        )
        assert ordinal_4["target_files"][relative]["after_size"] == (
            phase_c.REPO_ROOT / relative
        ).stat().st_size
    assert "after_sha256" not in config["engineering_correction_ordinal_2"]
    assert "after_sha256" not in config["engineering_correction_ordinal_3"]
    assert "after_sha256" not in config["engineering_correction_ordinal_4"]


def test_all_scientific_constants_remain_frozen(phase_c, config):
    assert config["base_p"] == phase_c.BASE_P_CONTRACT
    assert [row["architecture_fingerprint"] for row in config["safety_selection_expected"]["selected"]] == [
        "c8997b13a29eb83cc6e86479197e01fedce82f3a9b7833302a29831c5baf4257",
        "c132164074f86a46208ed6fc88c1a5896313e5788052018b177cc77904561f0e",
        "4de35d83f0269c97f99369d038212e9e53d3f191eef4019e02630308e0d19820",
        "4914015d3629fecd2384e45d954746522838e303b72e448309ff13551845eb31",
    ]
    assert [row["rank_hash"] for row in config["safety_selection_expected"]["selected"]] == [
        "00f2b6cbf4e7c6f0f8d4c66c7c456e3c8eaf13b68417283e604ab9b31d0959d8",
        "01eaff54014e660a3d20cc102db6dd1a14d7511a8d925e411b1c5ed544e6ee39",
        "0262a3d9a5848eaec921cf74f003aa9b3ff233cf8fb3ead3ed394ccc5f3d2194",
        "0572f84a50c38954f4f765193e92983d9636d41ad6addb27baa9eb0f60d58a29",
    ]
    assert config["safety_selection_expected"]["eligible_counts"] == {
        "OH_BLIND_JOINT_RANDOM": 174,
        "HR_BALANCED_RANDOM_HP": 191,
    }
    assert len(config["hp_screening"]["raw_points"]) == 5
    assert config["seed_rule"] == {
        "start": 1000,
        "count": 45,
        "primary_count": 30,
        "screening_count": 5,
        "safety_count": 10,
    }
    assert config["inference"]["signflip_draws"] == 200000
    assert config["inference"]["bootstrap_replicates"] == 100000
    assert config["inference"]["signflip_seed_material"] == (
        "fixed-layernorm-enhancement-global4-v2|signflip|2026-08-30"
    )
    assert config["inference"]["bootstrap_seed_material"] == (
        "fixed-layernorm-enhancement-global4-v2|bootstrap|2026-08-30"
    )
    assert config["workers"]["count"] == 4
    assert [row["gpu_index"] for row in config["workers"]["mapping"]] == [0, 1, 0, 1]


def test_ordinal_4_canonical_commitment_is_independent_and_exact(phase_c):
    assert phase_c.canonical_commitment_json_bytes_v1({}) == b"{}"
    encoded = phase_c.canonical_commitment_json_bytes_v1({"b": 1, "a": 2})
    assert encoded == b'{"a":2,"b":1}'
    assert not encoded.startswith(b"\xef\xbb\xbf")
    assert not encoded.endswith(b"\n")
    assert phase_c.canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert phase_c.ordinal_4_commitment_sha256(
        phase_c.EXACT_HISTORICAL_JSON_ADAPTER_CONTRACT
    ) == "50c21a181acb7fc31acc55ca0dde0432c7a08b5884033676c2da171c36abc817"


def test_ordinal_4_adapter_plan_and_self_binding_constants_are_exact(phase_c, config):
    plan = phase_c.EXPOSURE_ADAPTER_PLAN
    assert plan["marker_filtered_prewrite_file_count"] == 607
    assert plan["marker_filtered_prewrite_tree_sha256"] == (
        "0316ef48a9b48297074bdc39b93088d2781057af7ae36a88607a7559dc738ea0"
    )
    assert plan["marker_filtered_current_file_count"] == 607
    assert plan["execution_owned_exact_exclusions"] == [phase_c.CONFIG_REL]
    assert plan["historical_file_count"] == 606
    assert plan["historical_tree_sha256"] == (
        "40a9fc01007a06dc702b6413d6fdf23d4258f5237f3bbff7421b59b2e0f7dec7"
    )
    assert plan["adapter_class_counts"] == {
        phase_c.CSV_JSON_IDENTITY_CELL_V1: 102,
        phase_c.JSON_IDENTITY_DEFAULT_V1: 490,
        phase_c.JSON_EXACT_ISOLATED_EDGES_COUNT_COLLISION_V1: 4,
        phase_c.OPAQUE_EXACT_IDENTITY_TOKEN_V1: 10,
    }
    assert plan["legacy_error_count"] == 74
    assert plan["legacy_success_count"] == 532
    assert plan["new_projection_file_count"] == 606
    assert plan["exposed_architecture_count"] == 1253
    assert config["exposure_rule"] == {
        "roots": ["analyse", "configs", "scripts", "tests", "results"],
        "path_markers": ["layernorm", "layer_norm", "layer-norm"],
        "suffixes": [
            ".py",
            ".json",
            ".jsonl",
            ".ndjson",
            ".csv",
            ".md",
            ".txt",
            ".yaml",
            ".yml",
        ],
        "excluded_markers": [
            "fixed_layernorm_enhancement_v2",
            "fixed-layernorm-enhancement-global4-v2",
        ],
        "identity_whitelist": [
            "architecture_fingerprint",
            "joint_config_fingerprint",
            "joint_fingerprint",
            "hp_fingerprint",
            "operations",
            "ops",
            "edges",
            "config_id",
        ],
    }


def test_exact_adapter_routing_and_production_identity_binding(phase_c):
    for row in phase_c.EXACT_HISTORICAL_JSON_ADAPTER_CONTRACT["files"]:
        assert phase_c.route_exposure_adapter(
            row["path"], row["size"], row["sha256"]
        ) == phase_c.JSON_EXACT_ISOLATED_EDGES_COUNT_COLLISION_V1
        with pytest.raises(phase_c.ProtocolFailure) as changed_sha:
            phase_c.route_exposure_adapter(row["path"], row["size"], "0" * 64)
        assert changed_sha.value.error_code == (
            "EXPOSURE_EXACT_JSON_FILE_IDENTITY_MISMATCH"
        )
        near = row["path"].replace(".json", "-backup.json")
        assert phase_c.route_exposure_adapter(
            near, row["size"], row["sha256"]
        ) == phase_c.JSON_IDENTITY_DEFAULT_V1
    assert phase_c.route_exposure_adapter("results/x/layernorm.csv", 1, "0" * 64) == (
        phase_c.CSV_JSON_IDENTITY_CELL_V1
    )
    assert phase_c.route_exposure_adapter("results/x/layernorm.jsonl", 1, "0" * 64) == (
        phase_c.JSON_IDENTITY_DEFAULT_V1
    )
    assert phase_c.route_exposure_adapter("results/x/layernorm.md", 1, "0" * 64) == (
        phase_c.OPAQUE_EXACT_IDENTITY_TOKEN_V1
    )


def test_default_json_jsonl_and_opaque_adapters_preserve_legacy_semantics(phase_c):
    fingerprint = "a" * 64
    json_data = phase_c.canonical_json_bytes(
        {"architecture_fingerprint": fingerprint, "validation": 0.99}
    )
    new_json, _ = phase_c._project_with_pinned_exposure_adapter(
        json_data,
        "synthetic.json",
        phase_c.JSON_IDENTITY_DEFAULT_V1,
        enforce_production_binding=False,
    )
    assert new_json == phase_c._legacy_exposure_projection(
        json_data, "synthetic.json"
    )
    jsonl_data = json_data + b"\n\n" + json_data + b"\n"
    new_jsonl, _ = phase_c._project_with_pinned_exposure_adapter(
        jsonl_data,
        "synthetic.jsonl",
        phase_c.JSON_IDENTITY_DEFAULT_V1,
        enforce_production_binding=False,
    )
    assert new_jsonl == phase_c._legacy_exposure_projection(
        jsonl_data, "synthetic.jsonl"
    )
    opaque = b'architecture_fingerprint="' + fingerprint.encode() + b'"\n'
    projected, _ = phase_c._project_with_pinned_exposure_adapter(
        opaque,
        "synthetic.md",
        phase_c.OPAQUE_EXACT_IDENTITY_TOKEN_V1,
        enforce_production_binding=False,
    )
    assert projected == []
    assert phase_c._semantic_architecture_identity_set(opaque, projected) == [
        fingerprint
    ]


def test_self_asset_classifier_excludes_only_exact_config(
    phase_c, tmp_path, monkeypatch
):
    for root in ("configs", "shadow"):
        (tmp_path / root).mkdir()
    exact = tmp_path / phase_c.CONFIG_REL
    exact.write_text("{}", encoding="utf-8")
    near = tmp_path / "configs/fixed_layernorm_enhancement_global4_v2-backup.json"
    near.write_text("{}", encoding="utf-8")
    same_basename_elsewhere = (
        tmp_path / "shadow/fixed_layernorm_enhancement_global4_v2.json"
    )
    same_basename_elsewhere.write_text("{}", encoding="utf-8")
    excluded_marker = tmp_path / "configs/fixed_layernorm_enhancement_v2.json"
    excluded_marker.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "configs/layernorm-symlink.json"
    symlink.symlink_to(near)
    (tmp_path / "configs/layernorm-nonregular.json").mkdir()
    synthetic_config = {
        "exposure_rule": {
            "roots": ["configs", "shadow"],
            "path_markers": ["layernorm", "layer_norm", "layer-norm"],
            "suffixes": [".json"],
            "excluded_markers": [
                "fixed_layernorm_enhancement_v2",
                "fixed-layernorm-enhancement-global4-v2",
            ],
        },
        "exposure_adapter_plan": {
            "execution_owned_exact_exclusions": [phase_c.CONFIG_REL]
        },
    }
    monkeypatch.setattr(phase_c, "REPO_ROOT", tmp_path)
    classified = phase_c._classify_exposure_files(synthetic_config)
    marker_relatives = [
        path.relative_to(tmp_path).as_posix()
        for path in classified["marker_filtered"]
    ]
    historical_relatives = [
        path.relative_to(tmp_path).as_posix() for path in classified["historical"]
    ]
    assert phase_c.CONFIG_REL in marker_relatives
    assert phase_c.CONFIG_REL not in historical_relatives
    assert near.relative_to(tmp_path).as_posix() in historical_relatives
    assert same_basename_elsewhere.relative_to(tmp_path).as_posix() in (
        historical_relatives
    )
    assert excluded_marker.relative_to(tmp_path).as_posix() not in marker_relatives
    assert symlink.relative_to(tmp_path).as_posix() not in marker_relatives
    assert "configs/layernorm-nonregular.json" not in marker_relatives


def test_csv_json_identity_cell_adapter_positive_and_outcome_lexical_skip(phase_c):
    operations = ["GCNConv", "SAGEConv"]
    edges = [[0, 1], [1, 2], [2, 3]]
    fingerprint = phase_c.canonical_architecture_fingerprint(operations, edges)
    header = [
        "architecture_fingerprint",
        "operations",
        "edges",
        "validation",
        "test_accuracy",
    ]
    data = _synthetic_csv_bytes(
        header,
        [
            [
                fingerprint,
                json.dumps(operations) + " \t",
                json.dumps(edges),
                "__import__('os').system('false')",
                "[not parsed as JSON]",
            ]
        ],
    )
    records, schema = phase_c._project_csv_identity_adapter_with_header_groups(
        data,
        "synthetic.csv",
        _synthetic_csv_adapter_contract(phase_c, header),
    )
    assert records == [
        {
            "architecture_fingerprint": fingerprint,
            "operations": operations,
            "edges": edges,
        }
    ]
    assert schema["row_count"] == 1
    assert schema["identity_columns"] == [
        "architecture_fingerprint",
        "edges",
        "operations",
    ]
    assert schema["nonempty_identity_counts"] == {
        "architecture_fingerprint": 1,
        "edges": 1,
        "operations": 1,
    }


def test_csv_identity_cells_reject_all_forbidden_schema_forms(phase_c):
    operations = ["GCNConv", "SAGEConv"]
    edges = [[0, 1], [1, 2], [2, 3]]
    fingerprint = phase_c.canonical_architecture_fingerprint(operations, edges)
    cases = [
        (["operations", "edges"], [[json.dumps(operations) + " trailing", json.dumps(edges)]]),
        (["operations", "edges"], [[json.dumps({"ops": operations}), json.dumps(edges)]]),
        (["operations", "edges"], [[repr(operations), json.dumps(edges)]]),
        (["operations", "edges"], [[json.dumps(operations), json.dumps([[False, 1]])]]),
        (["operations", "edges"], [[json.dumps(operations), json.dumps([[0]])]]),
        (["operations", "edges"], [[json.dumps(operations), ""]]),
        (
            ["operations", "ops", "edges"],
            [[json.dumps(operations), json.dumps(operations), json.dumps(edges)]],
        ),
        (
            ["architecture_fingerprint", "operations", "edges"],
            [["a" * 64 if fingerprint != "a" * 64 else "b" * 64, json.dumps(operations), json.dumps(edges)]],
        ),
        (["architecture_fingerprint"], [["A" * 64]]),
        (["operations", "edges"], [["[__import__('os')]", json.dumps(edges)]]),
        (["operations", "edges"], [[json.dumps([1]), json.dumps(edges)]]),
    ]
    for header, rows in cases:
        data = _synthetic_csv_bytes(header, rows)
        with pytest.raises(phase_c.HistoricalProjectionError):
            phase_c._project_csv_identity_adapter_with_header_groups(
                data,
                "synthetic.csv",
                _synthetic_csv_adapter_contract(phase_c, header),
            )


@pytest.mark.parametrize("file_index", [0, 1, 2, 3])
def test_four_exact_json_pointer_adapter_schemas_pass_synthetically(
    phase_c, file_index
):
    _, data, contract, identities = _synthetic_exact_json_fixture(
        phase_c, file_index
    )
    records, evidence = phase_c.project_exact_historical_json_adapter(
        data, contract["path"], contract=contract
    )
    assert phase_c._semantic_architecture_identity_set(data, records) == identities
    assert evidence["structured_record_count"] == contract[
        "expected_structured_record_count"
    ]
    assert evidence["fingerprint_only_record_count"] == contract[
        "expected_fingerprint_only_record_count"
    ]
    assert evidence["identity_set_sha256"] == contract["identity_set_sha256"]
    with pytest.raises(phase_c.HistoricalProjectionError):
        phase_c._legacy_exposure_projection(data, contract["path"])


def test_exact_json_pointer_adapter_rejects_binding_path_and_schema_mutations(phase_c):
    payload, data, contract, _ = _synthetic_exact_json_fixture(phase_c, 1)
    with pytest.raises(phase_c.HistoricalProjectionError):
        phase_c.project_exact_historical_json_adapter(
            data + b" ", contract["path"], contract=contract
        )
    with pytest.raises(phase_c.HistoricalProjectionError):
        phase_c.project_exact_historical_json_adapter(
            data, "synthetic/near.json", contract=contract
        )

    mutations = []
    wrong_dataset_type = copy.deepcopy(payload)
    wrong_dataset_type["dataset_identity"]["edges"] = True
    mutations.append(wrong_dataset_type)
    moved_dataset_pointer = copy.deepcopy(payload)
    moved_dataset_pointer["dataset_identity_near"] = moved_dataset_pointer.pop(
        "dataset_identity"
    )
    mutations.append(moved_dataset_pointer)
    wrong_container = copy.deepcopy(payload)
    wrong_container["selected_candidates"] = {
        str(index): row
        for index, row in enumerate(wrong_container["selected_candidates"])
    }
    mutations.append(wrong_container)
    missing_same_record_identity = copy.deepcopy(payload)
    missing_same_record_identity["selected_candidates"][0].pop("edges")
    missing_same_record_identity["edges"] = [[0, 1]]
    mutations.append(missing_same_record_identity)
    missing_summary = copy.deepcopy(payload)
    missing_summary["selected_candidates"][0].pop("architecture_summary")
    mutations.append(missing_summary)
    fingerprint_mismatch = copy.deepcopy(payload)
    current = fingerprint_mismatch["selected_candidates"][0][
        "architecture_fingerprint"
    ]
    fingerprint_mismatch["selected_candidates"][0][
        "architecture_fingerprint"
    ] = "a" * 64 if current != "a" * 64 else "b" * 64
    mutations.append(fingerprint_mismatch)
    uppercase_fingerprint = copy.deepcopy(payload)
    uppercase_fingerprint["selected_candidates"][0][
        "architecture_fingerprint"
    ] = uppercase_fingerprint["selected_candidates"][0][
        "architecture_fingerprint"
    ].upper()
    mutations.append(uppercase_fingerprint)
    arbitrary_invalid_edges = copy.deepcopy(payload)
    arbitrary_invalid_edges["uncontracted"] = {"edges": 3}
    mutations.append(arbitrary_invalid_edges)
    for mutated in mutations:
        mutated_data = phase_c.canonical_json_bytes(mutated)
        mutated_contract = _rebind_synthetic_exact_contract(
            phase_c, contract, mutated_data
        )
        with pytest.raises(phase_c.HistoricalProjectionError):
            phase_c.project_exact_historical_json_adapter(
                mutated_data,
                mutated_contract["path"],
                contract=mutated_contract,
            )

    duplicate = b'{"dataset_identity":{"edges":17},' + data[1:]
    duplicate_contract = _rebind_synthetic_exact_contract(
        phase_c, contract, duplicate
    )
    with pytest.raises(phase_c.HistoricalProjectionError):
        phase_c.project_exact_historical_json_adapter(
            duplicate, duplicate_contract["path"], contract=duplicate_contract
        )


def test_exact_json_outcome_mutation_requires_binding_then_is_semantically_invariant(
    phase_c,
):
    payload, data, contract, identities = _synthetic_exact_json_fixture(phase_c, 3)
    base_records, _ = phase_c.project_exact_historical_json_adapter(
        data, contract["path"], contract=contract
    )
    for index, family in enumerate(
        ["validation", "accuracy", "loss", "predictor_prediction", "test"],
        start=1,
    ):
        mutated = copy.deepcopy(payload)
        mutated["outcomes"][family] = 10000 + index
        mutated_data = phase_c.canonical_json_bytes(mutated)
        with pytest.raises(phase_c.HistoricalProjectionError):
            phase_c.project_exact_historical_json_adapter(
                mutated_data, contract["path"], contract=contract
            )
        rebound = _rebind_synthetic_exact_contract(phase_c, contract, mutated_data)
        mutated_records, _ = phase_c.project_exact_historical_json_adapter(
            mutated_data, rebound["path"], contract=rebound
        )
        assert phase_c._semantic_architecture_identity_set(
            mutated_data, mutated_records
        ) == identities
        assert phase_c._semantic_architecture_identity_set(data, base_records) == (
            identities
        )


def test_candidate_veto_is_fixed_and_precedes_dependencies_seed_and_result(
    phase_c, config, monkeypatch
):
    candidate = config["safety_selection_expected"]["selected"][0][
        "architecture_fingerprint"
    ]
    monkeypatch.setattr(
        phase_c,
        "build_exposure_registry",
        lambda unused: {"exposed_architecture_fingerprints": [candidate]},
    )
    dependency_called = False

    def forbidden_dependency_call(unused):
        nonlocal dependency_called
        dependency_called = True
        raise AssertionError("dependency path must not be reached")

    monkeypatch.setattr(phase_c, "_dependency_audit", forbidden_dependency_call)
    with pytest.raises(phase_c.ProtocolFailure) as veto:
        phase_c.dry_run(CONFIG_PATH)
    assert veto.value.stage == "EXPOSURE_REGISTRY"
    assert veto.value.error_code == "FROZEN_SAFETY_CANDIDATE_HISTORICAL_EXPOSURE"
    assert dependency_called is False
    assert not (REPO_ROOT / phase_c.RESULT_REL).exists()


def test_preledger_safe_failure_reports_unverified_null_test_status(phase_c):
    payload = phase_c._safe_failure(
        phase_c.ProtocolFailure("EXPOSURE_REGISTRY", "SYNTHETIC_FAILURE")
    )
    assert payload["test_accessed"] is None
    assert payload["test_access_status"] == "UNVERIFIED_FAIL_CLOSED"


def test_production_metadata_binding_changes_on_outcome_only_bytes(phase_c):
    base = [{"path": "results/x/layernorm.json", "size": 10, "sha256": "1" * 64}]
    mutated = [
        {"path": "results/x/layernorm.json", "size": 10, "sha256": "2" * 64}
    ]
    assert phase_c._historical_metadata_tree_sha256(base) != (
        phase_c._historical_metadata_tree_sha256(mutated)
    )


def test_real_phase_a_schema_training_false_passes(phase_c):
    projected = {
        name: phase_c.project_phase_a_control_file(name, phase_c._json_asset(payload))
        for name, payload in _control_payloads().items()
    }
    assert projected == {
        name: {
            key: field["value"]
            for key, field in phase_c.PHASE_A_CONTROL_CONTRACT["files"][name][
                "safe_fields"
            ].items()
        }
        for name in phase_c.PHASE_A_CONTROL_CONTRACT["files"]
    }
    assert projected["decision.json"]["training"] is False
    assert projected["decision.json"]["boundary"] is False
    assert projected["decision.json"]["training_sufficient"] is True
    evidence = phase_c.phase_a_control_projection_evidence(projected)
    assert evidence["aggregate"] == {
        "size_bytes": 677,
        "sha256": "6c54598c7c01d63dd0ae88a3c52e085706190670852c2403ea012e4e29c0b833",
    }


CONTROL_FIELD_CASES = [
    (filename, key)
    for filename, payload in _control_payloads().items()
    for key in payload
    if key
    in {
        "classification",
        "boundary",
        "training",
        "training_sufficient",
        "new_gnn_runs",
        "gpu_compute_runs",
        "test_accessed",
        "phase_b_started",
        "status",
        "gpu_compute_used",
        "sufficient",
        "raw_training_truncation_signal",
        "training_truncation_signal",
        "passed",
        "gnn_training_calls",
        "gpu_compute_calls",
        "test_data_reads",
        "runner_exact_command_verified",
    }
]


@pytest.mark.parametrize("filename,key", CONTROL_FIELD_CASES)
def test_every_phase_a_safe_field_is_required_at_root(phase_c, filename, key):
    payload = copy.deepcopy(_control_payloads()[filename])
    value = payload.pop(key)
    with pytest.raises(phase_c.ProtocolFailure) as missing:
        phase_c.project_phase_a_control_file(filename, phase_c._json_asset(payload))
    assert missing.value.error_code == "PHASE_A_REQUIRED_TOP_LEVEL_CONTROL_MISSING"

    payload["nested_controls"] = {key: value}
    with pytest.raises(phase_c.ProtocolFailure) as nested:
        phase_c.project_phase_a_control_file(filename, phase_c._json_asset(payload))
    assert nested.value.error_code == "PHASE_A_REQUIRED_TOP_LEVEL_CONTROL_MISSING"


@pytest.mark.parametrize("filename,key", CONTROL_FIELD_CASES)
def test_every_phase_a_safe_field_wrong_file_cannot_compensate(
    phase_c, filename, key
):
    payloads = copy.deepcopy(_control_payloads())
    value = payloads[filename].pop(key)
    filenames = list(payloads)
    wrong_filename = filenames[(filenames.index(filename) + 1) % len(filenames)]
    payloads[wrong_filename][key] = value
    wrong_file_fields = phase_c.PHASE_A_CONTROL_CONTRACT["files"][wrong_filename][
        "safe_fields"
    ]
    if key not in wrong_file_fields:
        wrong_projection = phase_c.project_phase_a_control_file(
            wrong_filename, phase_c._json_asset(payloads[wrong_filename])
        )
        assert key not in wrong_projection
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.project_phase_a_control_file(
            filename, phase_c._json_asset(payloads[filename])
        )
    assert error.value.error_code == "PHASE_A_REQUIRED_TOP_LEVEL_CONTROL_MISSING"


@pytest.mark.parametrize("filename,key", CONTROL_FIELD_CASES)
def test_every_phase_a_safe_field_rejects_wrong_native_type_and_value(
    phase_c, filename, key
):
    payload = copy.deepcopy(_control_payloads()[filename])
    expected = payload[key]
    if type(expected) is bool:
        wrong_type, wrong_value = 0, not expected
    elif type(expected) is int:
        wrong_type, wrong_value = False, expected + 1
    else:
        wrong_type, wrong_value = False, expected + "_WRONG"
    payload[key] = wrong_type
    with pytest.raises(phase_c.ProtocolFailure) as type_error:
        phase_c.project_phase_a_control_file(filename, phase_c._json_asset(payload))
    assert type_error.value.error_code == "PHASE_A_CONTROL_NATIVE_TYPE_MISMATCH"
    payload[key] = wrong_value
    with pytest.raises(phase_c.ProtocolFailure) as value_error:
        phase_c.project_phase_a_control_file(filename, phase_c._json_asset(payload))
    assert value_error.value.error_code == "PHASE_A_CONTROL_VALUE_MISMATCH"


@pytest.mark.parametrize("filename,key", CONTROL_FIELD_CASES)
def test_every_phase_a_safe_field_rejects_duplicate_key(phase_c, filename, key):
    payload = _control_payloads()[filename]
    encoded = phase_c.canonical_json_bytes(payload)
    encoded_value = phase_c.canonical_json_bytes(payload[key])
    needle = phase_c.canonical_json_bytes(key) + b":" + encoded_value
    duplicate = encoded.replace(needle, needle + b"," + needle, 1)
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.project_phase_a_control_file(filename, duplicate)
    assert error.value.error_code == "PHASE_A_CONTROL_DUPLICATE_KEY"


def test_phase_a_control_projector_rejects_duplicate_key_in_skipped_value(phase_c):
    payload = phase_c._json_asset(_control_payloads()["decision.json"]).replace(
        b'"mean_validation": 0.987',
        b'"mean_validation": {"hidden": 1, "hidden": 2}',
        1,
    )
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.project_phase_a_control_file("decision.json", payload)
    assert error.value.error_code == "PHASE_A_CONTROL_DUPLICATE_KEY"


@pytest.mark.parametrize(
    "mutation,code",
    [
        ({"training": "false"}, "PHASE_A_CONTROL_NATIVE_TYPE_MISMATCH"),
        ({"training": True}, "PHASE_A_CONTROL_VALUE_MISMATCH"),
        ({"training": None}, "PHASE_A_CONTROL_NATIVE_TYPE_MISMATCH"),
    ],
)
def test_phase_a_training_wrong_type_or_value_fails(phase_c, mutation, code):
    payload = _control_payloads()["decision.json"] | mutation
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.project_phase_a_control_file("decision.json", phase_c._json_asset(payload))
    assert error.value.error_code == code


def test_phase_a_alias_only_and_conflicting_alias_fail(phase_c):
    payload = _control_payloads()["decision.json"]
    alias_only = dict(payload)
    alias_only.pop("training")
    alias_only["training_signal"] = False
    with pytest.raises(phase_c.ProtocolFailure):
        phase_c.project_phase_a_control_file("decision.json", phase_c._json_asset(alias_only))
    conflicting = dict(payload, training_signal=False)
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.project_phase_a_control_file("decision.json", phase_c._json_asset(conflicting))
    assert error.value.error_code == "PHASE_A_CONFLICTING_CONTROL_ALIAS"


@pytest.mark.parametrize(
    "filename,legacy_payload",
    [
        ("decision.json", {"decision": "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL"}),
        ("training_audit.json", {"gnn_evaluations": 0}),
        ("test_firewall.json", {"test_accessed": False}),
    ],
)
def test_disproved_legacy_only_control_shapes_fail(phase_c, filename, legacy_payload):
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.project_phase_a_control_file(filename, phase_c._json_asset(legacy_payload))
    assert error.value.error_code == "PHASE_A_CONFLICTING_CONTROL_ALIAS"


@pytest.mark.parametrize(
    "filename,alias,value",
    [
        ("decision.json", "boundary_signal", False),
        ("decision.json", "decision", "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL"),
        ("decision.json", "training_data_sufficient", True),
        ("decision.json", "training_signal", False),
        ("training_audit.json", "gnn_evaluations", 0),
        ("training_audit.json", "training_signal", False),
        ("test_firewall.json", "test_accessed", False),
        ("test_firewall.json", "test_outcome_accessed", False),
        ("test_firewall.json", "test_values_accessed", False),
    ],
)
def test_every_conflicting_alias_and_old_new_dual_shape_fails(
    phase_c, filename, alias, value
):
    payload = copy.deepcopy(_control_payloads()[filename])
    payload[alias] = value
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.project_phase_a_control_file(filename, phase_c._json_asset(payload))
    assert error.value.error_code == "PHASE_A_CONFLICTING_CONTROL_ALIAS"


def test_phase_a_wrong_file_cannot_compensate(phase_c):
    state = _control_payloads()["state.json"] | {"training": False}
    decision = _control_payloads()["decision.json"].copy()
    decision.pop("training")
    phase_c.project_phase_a_control_file("state.json", phase_c._json_asset(state))
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.project_phase_a_control_file("decision.json", phase_c._json_asset(decision))
    assert error.value.error_code == "PHASE_A_REQUIRED_TOP_LEVEL_CONTROL_MISSING"


def test_phase_a_forbidden_scores_are_lexically_skipped(phase_c):
    first = _control_payloads()["decision.json"] | {
        "validation": {"trajectory": [0.1, 0.2]},
        "accuracy": [0.3],
        "loss": {"nested": 0.4},
        "predictor": "outcome-a",
        "test": {"outcome": 0.5},
    }
    second = copy.deepcopy(first)
    second.update(
        {
            "validation": {"trajectory": [-100.0]},
            "accuracy": [-200.0],
            "loss": {"nested": -300.0},
            "predictor": "outcome-b",
            "test": {"outcome": -500.0},
        }
    )
    first_projection = phase_c.project_phase_a_control_file(
        "decision.json", phase_c._json_asset(first)
    )
    second_projection = phase_c.project_phase_a_control_file(
        "decision.json", phase_c._json_asset(second)
    )
    assert first_projection == second_projection


@pytest.mark.parametrize(
    "filename,key,value,error_code",
    [
        (
            "decision.json",
            "classification",
            "MISMATCH",
            "PHASE_A_CLASSIFICATION_CONSISTENCY_MISMATCH",
        ),
        (
            "state.json",
            "classification",
            "MISMATCH",
            "PHASE_A_CLASSIFICATION_CONSISTENCY_MISMATCH",
        ),
        (
            "decision.json",
            "training_sufficient",
            False,
            "PHASE_A_TRAINING_SUFFICIENCY_CONSISTENCY_MISMATCH",
        ),
        (
            "training_audit.json",
            "sufficient",
            False,
            "PHASE_A_TRAINING_SUFFICIENCY_CONSISTENCY_MISMATCH",
        ),
        (
            "decision.json",
            "training",
            True,
            "PHASE_A_FINAL_TRAINING_SIGNAL_CONSISTENCY_MISMATCH",
        ),
        (
            "training_audit.json",
            "training_truncation_signal",
            True,
            "PHASE_A_FINAL_TRAINING_SIGNAL_CONSISTENCY_MISMATCH",
        ),
        (
            "decision.json",
            "phase_b_started",
            True,
            "PHASE_A_PHASE_B_CONSISTENCY_MISMATCH",
        ),
        (
            "state.json",
            "phase_b_started",
            True,
            "PHASE_A_PHASE_B_CONSISTENCY_MISMATCH",
        ),
        (
            "decision.json",
            "test_accessed",
            True,
            "PHASE_A_TEST_ZERO_CONSISTENCY_MISMATCH",
        ),
        (
            "state.json",
            "test_accessed",
            True,
            "PHASE_A_TEST_ZERO_CONSISTENCY_MISMATCH",
        ),
        (
            "test_firewall.json",
            "test_data_reads",
            1,
            "PHASE_A_TEST_ZERO_CONSISTENCY_MISMATCH",
        ),
        (
            "test_firewall.json",
            "passed",
            False,
            "PHASE_A_TEST_ZERO_CONSISTENCY_MISMATCH",
        ),
        (
            "test_firewall.json",
            "runner_exact_command_verified",
            False,
            "PHASE_A_TEST_ZERO_CONSISTENCY_MISMATCH",
        ),
        (
            "decision.json",
            "new_gnn_runs",
            1,
            "PHASE_A_GNN_ZERO_CONSISTENCY_MISMATCH",
        ),
        (
            "test_firewall.json",
            "gnn_training_calls",
            1,
            "PHASE_A_GNN_ZERO_CONSISTENCY_MISMATCH",
        ),
        (
            "decision.json",
            "gpu_compute_runs",
            1,
            "PHASE_A_GPU_ZERO_CONSISTENCY_MISMATCH",
        ),
        (
            "state.json",
            "gpu_compute_used",
            True,
            "PHASE_A_GPU_ZERO_CONSISTENCY_MISMATCH",
        ),
        (
            "test_firewall.json",
            "gpu_compute_calls",
            1,
            "PHASE_A_GPU_ZERO_CONSISTENCY_MISMATCH",
        ),
    ],
)
def test_phase_a_cross_file_consistency_rejects_isolated_mutations(
    phase_c, filename, key, value, error_code
):
    controls = {
        name: phase_c.project_phase_a_control_file(name, phase_c._json_asset(payload))
        for name, payload in _control_payloads().items()
    }
    controls[filename][key] = value
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.validate_phase_a_control_consistency(controls)
    assert error.value.error_code == error_code


def test_phase_a_raw_training_signal_has_own_frozen_contract_not_general_equality(
    phase_c,
):
    payload = copy.deepcopy(_control_payloads()["training_audit.json"])
    payload["raw_training_truncation_signal"] = True
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.project_phase_a_control_file(
            "training_audit.json", phase_c._json_asset(payload)
        )
    assert error.value.error_code == "PHASE_A_CONTROL_VALUE_MISMATCH"

    controls = {
        name: phase_c.project_phase_a_control_file(name, phase_c._json_asset(value))
        for name, value in _control_payloads().items()
    }
    controls["training_audit.json"]["raw_training_truncation_signal"] = True
    derived = phase_c.validate_phase_a_control_consistency(controls)
    assert derived["training"] is False


def test_phase_a_archive_member_manifest_live_binding_passes(phase_c, tmp_path, monkeypatch):
    _, binding = _synthetic_phase_a_repo(phase_c, tmp_path)
    monkeypatch.setattr(phase_c, "REPO_ROOT", tmp_path)
    audited = phase_c.audit_phase_a({"phase_a": binding})
    assert audited["control_member_count"] == 4
    assert audited["control_projection_schema"] == (
        "exact-sealed-phase-a-safe-top-level-native-types-v3"
    )
    assert audited["scientific_state"] == "COMPLETE"
    assert audited["decision"] == "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL"
    assert audited["boundary"] is False
    assert audited["training"] is False
    assert audited["training_sufficient"] is True
    assert audited["gnn_evaluations"] == 0
    assert audited["test_accessed"] is False
    assert audited["control_projection_aggregate"] == {
        "size_bytes": 677,
        "sha256": "6c54598c7c01d63dd0ae88a3c52e085706190670852c2403ea012e4e29c0b833",
    }
    assert audited["manifest_record_count"] == 29
    assert audited["manifest_safe_projection_size_bytes"] == 4829
    assert audited["manifest_safe_projection_sha256"] == (
        "5a272edee3457d1f5d72052022c5dcea946a8a05a2dd1f7ea329280c98b8d022"
    )


def test_phase_a_archive_live_mismatch_fails(phase_c, tmp_path, monkeypatch):
    result, binding = _synthetic_phase_a_repo(phase_c, tmp_path)
    (result / "decision.json").write_bytes(
        phase_c._json_asset(_control_payloads()["decision.json"] | {"extra": 1})
    )
    monkeypatch.setattr(phase_c, "REPO_ROOT", tmp_path)
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.audit_phase_a({"phase_a": binding})
    assert error.value.error_code == "PHASE_A_CONTROL_ARCHIVE_LIVE_MISMATCH"


def test_phase_a_archive_manifest_live_agreement_cannot_replace_frozen_control_identity(
    phase_c, tmp_path, monkeypatch
):
    result, binding = _synthetic_phase_a_repo(phase_c, tmp_path)
    mutated = phase_c._json_asset(
        _control_payloads()["decision.json"] | {"mean_validation": -999.0}
    )
    (result / "decision.json").write_bytes(mutated)
    manifest_payload = json.loads((result / "manifest.json").read_text())
    decision_record = next(
        row for row in manifest_payload["files"] if row["relative_path"] == "decision.json"
    )
    decision_record["size_bytes"] = len(mutated)
    decision_record["sha256"] = phase_c.sha256_bytes(mutated)
    manifest = phase_c._json_asset(manifest_payload)
    (result / "manifest.json").write_bytes(manifest)
    binding["manifest_identity"] = {
        "path": "phase_a/manifest.json",
        "size_bytes": len(manifest),
        "sha256": phase_c.sha256_bytes(manifest),
    }
    archive = result / "phase-a.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        members = {
            member.name: handle.extractfile(member).read()
            for member in handle.getmembers()
            if member.isfile()
        }
    members["phase-a/decision.json"] = mutated
    members["phase-a/manifest.json"] = manifest
    with tarfile.open(archive, "w:gz") as handle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    binding["archive_sha256"] = phase_c.sha256_path(archive)
    (result / "phase-a.tar.gz.sha256").write_text(
        f"{binding['archive_sha256']}  phase-a.tar.gz\n", encoding="ascii"
    )
    monkeypatch.setattr(phase_c, "REPO_ROOT", tmp_path)
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.audit_phase_a({"phase_a": binding})
    assert error.value.error_code == "PHASE_A_CONTROL_FROZEN_IDENTITY_MISMATCH"


def test_phase_a_manifest_live_byte_identity_drift_fails(phase_c, tmp_path, monkeypatch):
    result, binding = _synthetic_phase_a_repo(phase_c, tmp_path)
    (result / "manifest.json").write_bytes((result / "manifest.json").read_bytes() + b" ")
    monkeypatch.setattr(phase_c, "REPO_ROOT", tmp_path)
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.audit_phase_a({"phase_a": binding})
    assert error.value.error_code == "PHASE_A_MANIFEST_BYTE_IDENTITY_MISMATCH"


def test_phase_a_manifest_archive_live_drift_fails(phase_c, tmp_path, monkeypatch):
    result, binding = _synthetic_phase_a_repo(phase_c, tmp_path)
    archive = result / "phase-a.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        members = {
            member.name: handle.extractfile(member).read()
            for member in handle.getmembers()
            if member.isfile()
        }
    members["phase-a/manifest.json"] += b" "
    with tarfile.open(archive, "w:gz") as handle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    binding["archive_sha256"] = phase_c.sha256_path(archive)
    (result / "phase-a.tar.gz.sha256").write_text(
        f"{binding['archive_sha256']}  phase-a.tar.gz\n", encoding="ascii"
    )
    monkeypatch.setattr(phase_c, "REPO_ROOT", tmp_path)
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.audit_phase_a({"phase_a": binding})
    assert error.value.error_code == "PHASE_A_MANIFEST_ARCHIVE_MISMATCH"


def test_phase_a_manifest_wrong_placement_fails(phase_c, tmp_path, monkeypatch):
    _, binding = _synthetic_phase_a_repo(phase_c, tmp_path)
    binding["manifest_identity"]["path"] = "phase_a/decision.json"
    monkeypatch.setattr(phase_c, "REPO_ROOT", tmp_path)
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.audit_phase_a({"phase_a": binding})
    assert error.value.error_code == "PHASE_A_MANIFEST_PLACEMENT_MISMATCH"


def test_manifest_projector_rejects_duplicate_and_wrong_type(phase_c):
    duplicate = phase_c._json_asset(_real_manifest_payload(phase_c)).replace(
        b'"size_bytes": 1\n',
        b'"size_bytes": 1,\n        "size_bytes": 1\n',
        1,
    )
    with pytest.raises(phase_c.ProtocolFailure) as duplicate_error:
        phase_c.project_archive_manifest_records(
            duplicate, phase_c.PHASE_A_MANIFEST_SCHEMA_CONTRACT
        )
    assert duplicate_error.value.error_code == "PHASE_A_MANIFEST_DUPLICATE_KEY"
    wrong_type = _real_manifest_payload(phase_c)
    wrong_type["files"][0]["size_bytes"] = True
    with pytest.raises(phase_c.ProtocolFailure) as type_error:
        _project_real_manifest(phase_c, wrong_type)
    assert type_error.value.error_code == "PHASE_A_MANIFEST_RECORD_TYPE_MISMATCH"

    root_duplicate = phase_c._json_asset(_real_manifest_payload(phase_c)).replace(
        b'"file_count": 29,', b'"file_count": 29, "file_count": 29,', 1
    )
    with pytest.raises(phase_c.ProtocolFailure) as root_duplicate_error:
        phase_c.project_archive_manifest_records(
            root_duplicate, phase_c.PHASE_A_MANIFEST_SCHEMA_CONTRACT
        )
    assert root_duplicate_error.value.error_code == "PHASE_A_MANIFEST_DUPLICATE_KEY"


@pytest.mark.parametrize("field,value", [("relative_path", 1), ("sha256", 1)])
def test_manifest_projector_rejects_non_native_record_types(phase_c, field, value):
    payload = _real_manifest_payload(phase_c)
    payload["files"][0][field] = value
    with pytest.raises(phase_c.ProtocolFailure) as error:
        _project_real_manifest(phase_c, payload)
    assert error.value.error_code == "PHASE_A_MANIFEST_RECORD_TYPE_MISMATCH"


def test_manifest_projector_real_shape_passes_and_normalizes(phase_c):
    payload = _real_manifest_payload(phase_c)
    projected = _project_real_manifest(phase_c, payload)
    assert projected["record_count"] == 29
    assert projected["safe_projection_size_bytes"] == 4829
    assert projected["safe_projection_sha256"] == (
        "5a272edee3457d1f5d72052022c5dcea946a8a05a2dd1f7ea329280c98b8d022"
    )
    assert projected["manifest_root_sha256"] == (
        "23e486d1aafb7101c5f236ca504045d9be29b827028354ea3fce3734a2dec8f4"
    )
    assert [row["path"] for row in projected["records"]] == (
        phase_c.PHASE_A_MANIFEST_SCHEMA_CONTRACT["relative_paths"]
    )
    assert set(projected["records"][0]) == {"path", "size", "sha256"}
    assert all("relative_path" not in row and "size_bytes" not in row for row in projected["records"])


@pytest.mark.parametrize(
    "root_mutation,error_code",
    [
        ({"experiment": "wrong"}, "PHASE_A_MANIFEST_SAFE_CONTROL_IDENTITY_MISMATCH"),
        ({"file_count": True}, "PHASE_A_MANIFEST_SAFE_CONTROL_IDENTITY_MISMATCH"),
        ({"file_count": 28}, "PHASE_A_MANIFEST_SAFE_CONTROL_IDENTITY_MISMATCH"),
        (
            {"manifest_root_sha256": "0" * 64},
            "PHASE_A_MANIFEST_SAFE_CONTROL_IDENTITY_MISMATCH",
        ),
        (
            {"archive_excluded_from_manifest": False},
            "PHASE_A_MANIFEST_SAFE_CONTROL_IDENTITY_MISMATCH",
        ),
        (
            {"archive_self_inclusion": True},
            "PHASE_A_MANIFEST_SAFE_CONTROL_IDENTITY_MISMATCH",
        ),
    ],
)
def test_manifest_projector_rejects_wrong_safe_root_controls(
    phase_c, root_mutation, error_code
):
    with pytest.raises(phase_c.ProtocolFailure) as error:
        _project_real_manifest(
            phase_c,
            _real_manifest_payload(phase_c, root_updates=root_mutation),
        )
    assert error.value.error_code == error_code


def test_manifest_projector_rejects_wrong_root_keys_and_placement(phase_c):
    extra_root = _real_manifest_payload(phase_c, root_updates={"outcome_summary": 1.0})
    with pytest.raises(phase_c.ProtocolFailure) as extra_error:
        _project_real_manifest(phase_c, extra_root)
    assert extra_error.value.error_code == "PHASE_A_MANIFEST_ROOT_SCHEMA_MISMATCH"

    missing_root = _real_manifest_payload(phase_c)
    missing_root.pop("created_at")
    with pytest.raises(phase_c.ProtocolFailure) as missing_error:
        _project_real_manifest(phase_c, missing_root)
    assert missing_error.value.error_code == "PHASE_A_MANIFEST_ROOT_SCHEMA_MISMATCH"

    misplaced = _real_manifest_payload(phase_c)
    records = misplaced.pop("files")
    misplaced["container"] = {"files": records}
    with pytest.raises(phase_c.ProtocolFailure) as placement_error:
        _project_real_manifest(phase_c, misplaced)
    assert placement_error.value.error_code == "PHASE_A_MANIFEST_ROOT_SCHEMA_MISMATCH"


def test_manifest_projector_rejects_legacy_and_mixed_record_schemas(phase_c):
    legacy = _real_manifest_payload(phase_c)
    first = legacy["files"][0]
    legacy["files"][0] = {
        "path": first["relative_path"],
        "size": first["size_bytes"],
        "sha256": first["sha256"],
    }
    with pytest.raises(phase_c.ProtocolFailure) as legacy_error:
        _project_real_manifest(phase_c, legacy)
    assert legacy_error.value.error_code == "PHASE_A_MANIFEST_RECORD_SCHEMA_MISMATCH"

    mixed = _real_manifest_payload(phase_c)
    mixed["files"][0]["path"] = mixed["files"][0]["relative_path"]
    with pytest.raises(phase_c.ProtocolFailure) as mixed_error:
        _project_real_manifest(phase_c, mixed)
    assert mixed_error.value.error_code == "PHASE_A_MANIFEST_RECORD_SCHEMA_MISMATCH"


@pytest.mark.parametrize(
    "mutation,error_code",
    [
        ("remove_record", "PHASE_A_MANIFEST_RECORD_CARDINALITY_MISMATCH"),
        ("wrong_path", "PHASE_A_MANIFEST_PATH_SET_MISMATCH"),
        ("duplicate_path", "PHASE_A_MANIFEST_RECORD_CARDINALITY_MISMATCH"),
        ("uppercase_digest", "PHASE_A_MANIFEST_RECORD_TYPE_MISMATCH"),
        ("malformed_digest", "PHASE_A_MANIFEST_RECORD_TYPE_MISMATCH"),
        ("absolute_path", "PHASE_A_MANIFEST_RECORD_PATH_UNSAFE"),
        ("parent_path", "PHASE_A_MANIFEST_RECORD_PATH_UNSAFE"),
        ("reordered_records", "PHASE_A_MANIFEST_SAFE_PROJECTION_DRIFT"),
    ],
)
def test_manifest_projector_rejects_record_identity_drift(phase_c, mutation, error_code):
    payload = _real_manifest_payload(phase_c)
    if mutation == "remove_record":
        payload["files"].pop()
    elif mutation == "wrong_path":
        payload["files"][0]["relative_path"] = "wrong-safe-name.json"
    elif mutation == "duplicate_path":
        payload["files"][-1]["relative_path"] = payload["files"][0]["relative_path"]
    elif mutation == "uppercase_digest":
        payload["files"][0]["sha256"] = payload["files"][0]["sha256"].upper()
    elif mutation == "malformed_digest":
        payload["files"][0]["sha256"] = "0" * 63
    elif mutation == "absolute_path":
        payload["files"][0]["relative_path"] = "/absolute.json"
    elif mutation == "parent_path":
        payload["files"][0]["relative_path"] = "safe/../escape.json"
    elif mutation == "reordered_records":
        payload["files"][0], payload["files"][1] = payload["files"][1], payload["files"][0]
    with pytest.raises(phase_c.ProtocolFailure) as error:
        _project_real_manifest(phase_c, payload)
    assert error.value.error_code == error_code


def test_manifest_projector_requires_files_array_and_record_objects(phase_c):
    wrong_files = _real_manifest_payload(phase_c, root_updates={"files": {}})
    with pytest.raises(phase_c.ProtocolFailure) as files_error:
        _project_real_manifest(phase_c, wrong_files)
    assert files_error.value.error_code == "PHASE_A_MANIFEST_FILES_CONTAINER_MISMATCH"

    wrong_record = _real_manifest_payload(phase_c, records=["not-an-object"])
    with pytest.raises(phase_c.ProtocolFailure) as record_error:
        _project_real_manifest(phase_c, wrong_record)
    assert record_error.value.error_code == "PHASE_A_MANIFEST_RECORD_CONTAINER_MISMATCH"


def test_manifest_created_at_is_lexically_skipped(phase_c):
    first = _real_manifest_payload(
        phase_c,
        root_updates={
            "created_at": {
                "validation": 0.1,
                "accuracy": 0.2,
                "loss": 0.3,
                "predictor": 0.4,
                "test": 0.5,
            }
        },
    )
    second = copy.deepcopy(first)
    second["created_at"] = {
        "validation": -1000,
        "accuracy": -2000,
        "loss": -3000,
        "prediction": -4000,
        "test": -5000,
    }
    assert _project_real_manifest(phase_c, first) == _project_real_manifest(phase_c, second)


def test_outcome_mutation_invariance_each_family(phase_c):
    evidence = phase_c.synthetic_projection_invariance()
    assert set(evidence["family_evidence"]) == {
        "validation",
        "accuracy",
        "loss",
        "predictor/prediction",
        "test",
    }
    assert all(all(row.values()) for row in evidence["family_evidence"].values())
    assert all(
        row["mutation_touch_count"] > 0
        and row["pinned_adapter_id"] == phase_c.JSON_IDENTITY_DEFAULT_V1
        for row in evidence["family_evidence"].values()
    )


def test_primary_csv_projector_decodes_only_selector_boundary(phase_c):
    data = (
        "architecture_fingerprint,source,mean_val,test_accuracy,loss,predictor\n"
        + "a" * 64
        + ",OH_BLIND_JOINT_RANDOM,0.91,0.92,0.1,7.0\n"
    ).encode()
    header, rows = phase_c.project_csv_columns(data, {"architecture_fingerprint", "source"})
    assert len(header) == 6
    assert rows == [{"architecture_fingerprint": "a" * 64, "source": "OH_BLIND_JOINT_RANDOM"}]


def _paired_tasks(stage: str, pair_count: int):
    tasks = []
    architectures = 1 if stage == "primary" else 4
    for pair_index in range(pair_count):
        architecture = hashlib.sha256(f"arch-{pair_index % architectures}".encode()).hexdigest()
        joint = "" if stage == "primary" else hashlib.sha256(f"joint-{pair_index}".encode()).hexdigest()
        for variant in ("baseline", "layer_norm"):
            tasks.append(
                {
                    "task_id": f"{stage}|{pair_index}|{variant}",
                    "stage": stage,
                    "architecture_fingerprint": architecture,
                    "joint_fingerprint": joint,
                    "seed": 1000 + pair_index,
                    "component_variant": variant,
                }
            )
    return tasks


def test_primary_and_safety_pairs_colocate_and_balance(phase_c, config):
    primary, p_ledger = phase_c.assign_stage_tasks(
        "primary", _paired_tasks("primary", 30), config["workers"]["mapping"], matched_pairs=True
    )
    safety, s_ledger = phase_c.assign_stage_tasks(
        "safety", _paired_tasks("safety", 40), config["workers"]["mapping"], matched_pairs=True
    )
    for tasks in (primary, safety):
        by_pair = {}
        for task in tasks:
            by_pair.setdefault(task["pair_id"], set()).add(
                (task["worker_id"], task["gpu_index"], task["gpu_uuid"])
            )
        assert all(len(values) == 1 for values in by_pair.values())
    assert sorted(p_ledger["worker_pair_counts"].values()) == [7, 7, 8, 8]
    assert sorted(p_ledger["gpu_pair_counts"].values()) == [15, 15]
    assert sorted(s_ledger["worker_pair_counts"].values()) == [10, 10, 10, 10]
    assert sorted(s_ledger["gpu_pair_counts"].values()) == [20, 20]


def test_screening_assigns_25_tasks_per_worker(phase_c, config):
    tasks = [
        {
            "task_id": f"screening|{index}",
            "stage": "screening",
            "architecture_fingerprint": "a" * 64,
            "seed": 1000 + index,
            "component_variant": "baseline",
        }
        for index in range(100)
    ]
    _, ledger = phase_c.assign_stage_tasks(
        "screening", tasks, config["workers"]["mapping"], matched_pairs=False
    )
    assert list(ledger["worker_task_counts"].values()) == [25, 25, 25, 25]


def test_split_pair_fails_before_evaluator(phase_c, config):
    tasks = _paired_tasks("primary", 30)
    tasks.pop()
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.assign_stage_tasks(
            "primary", tasks, config["workers"]["mapping"], matched_pairs=True
        )
    assert error.value.error_code == "MATCHED_PAIR_CONDITION_MISMATCH"


@pytest.mark.parametrize("relative", [
    "analyse/fixed_layernorm_enhancement_v2.py",
    "configs/fixed_layernorm_enhancement_global4_v2.json",
    "scripts/run_fixed_layernorm_enhancement_v2.sh",
    "tests/test_fixed_layernorm_enhancement_v2.py",
])
def test_each_live_snapshot_one_byte_mutation_fails(phase_c, tmp_path, monkeypatch, relative):
    repo = tmp_path / "repo"
    assets = {}
    for target in phase_c.TARGET_RELS:
        live = repo / target
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_bytes(f"bytes:{target}".encode())
        assets[f"source_snapshot/{target}"] = live.read_bytes()
    result = repo / "result"
    phase_c.write_prelaunch(result, assets)
    snapshot = result / f"source_snapshot/{relative}"
    data = snapshot.read_bytes()
    snapshot.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    monkeypatch.setattr(phase_c, "REPO_ROOT", repo)
    with pytest.raises(phase_c.ProtocolFailure):
        phase_c.verify_live_source_snapshot_binding(result)


def test_five_allowed_fields_counted_and_bypasses_denied(phase_c):
    class Data:
        x = 1
        edge_index = 2
        train_mask = 3
        val_mask = 4
        y = 5
        test_mask = 6
        num_nodes = 7

    proxy = phase_c.ValidationOnlyDataProxy(Data())
    for name in sorted(phase_c.DATA_ALLOWLIST):
        getattr(proxy, name)
    for name in ("_data", "_ledger", "ledger", "__dict__", "num_nodes"):
        with pytest.raises(phase_c.TestAccessViolation):
            getattr(proxy, name)
    with pytest.raises(phase_c.TestAccessViolation):
        proxy.test_mask
    ledger = phase_c._proxy_internal_ledger(proxy)
    assert ledger["allowed_request_count"] == 5
    assert ledger["forbidden_request_count"] == 6
    assert all(value == 1 for value in ledger["allowed_attribute_counts"].values())
    assert ledger["test_accessed"] is True


def test_setup_and_task_ledgers_aggregate_exactly(phase_c):
    setup = phase_c._new_access_ledger()
    task = phase_c._new_access_ledger()
    setup["allowed_request_count"] = 5
    setup["allowed_attribute_counts"] = {key: 1 for key in sorted(phase_c.DATA_ALLOWLIST)}
    task["allowed_request_count"] = 9
    task["allowed_attribute_counts"]["x"] = 4
    task["allowed_attribute_counts"]["edge_index"] = 5
    combined = phase_c._aggregate_access_ledgers([setup, task])
    assert combined["allowed_request_count"] == 14
    assert combined["allowed_attribute_counts"]["x"] == 5
    assert combined["allowed_attribute_counts"]["edge_index"] == 6


def test_track_test_true_fails_and_literal_false_wrapper(phase_c):
    class Data:
        test_mask = object()

    proxy = phase_c.ValidationOnlyDataProxy(Data())
    with pytest.raises(phase_c.TestAccessViolation):
        phase_c.probe_track_test(proxy, track_test=True)
    assert phase_c._proxy_internal_ledger(proxy)["test_accessed"] is True
    evidence = phase_c.assert_evaluator_ast_contract()
    assert evidence["wrapper_literal_track_test_false_calls"] == 1
    assert evidence["root_test_mask_all_guarded_by_track_test"] is True


@pytest.mark.parametrize("variant", ["baseline", "layer_norm"])
def test_both_conditions_use_guarded_protocol(phase_c, variant):
    calls = []

    def evaluator(**kwargs):
        calls.append(kwargs)
        return 0.5, True, {"best_epoch": 1, "epochs_ran": 1}

    task = {
        "architecture": {"operations": ["GCNConv"], "edges": [[0, 1]], "effective_layers": 1},
        "hp": {"lr": 0.01, "dropout": 0.5, "hidden_dim": 64, "weight_decay": 0.001},
        "in_ch": 4,
        "out_ch": 2,
        "seed": 1234,
        "component_variant": variant,
        "max_epochs": 300,
        "patience": 80,
    }
    phase_c.run_guarded_evaluation(
        task, phase_c.ValidationOnlyDataProxy(type("Data", (), {})()), "cpu", evaluator=evaluator
    )
    assert calls[0]["track_test"] is False
    assert calls[0]["component_variant"] == variant
    assert calls[0]["max_epochs"] == 300
    assert calls[0]["patience"] == 80


def _fake_nvidia(monkeypatch, phase_c, config, *, reordered=False, occupied=False):
    gpu_rows = [
        f"{row['index']}, GPU, {row['uuid']}, 24570, 10, 24000, 0"
        for row in config["workers"]["gpus"]
    ]
    if reordered:
        gpu_rows.reverse()

    def fake(command):
        if any("--query-gpu=" in item for item in command):
            return subprocess.CompletedProcess(command, 0, "\n".join(gpu_rows) + "\n", "")
        compute = "GPU-x, 99, python, 100\n" if occupied else ""
        return subprocess.CompletedProcess(command, 0, compute, "")

    monkeypatch.setattr(phase_c, "_run_readonly", fake)


def test_gpu_occupied_and_reordered_fail_before_evaluator(phase_c, config, monkeypatch):
    _fake_nvidia(monkeypatch, phase_c, config, occupied=True)
    with pytest.raises(phase_c.ProtocolFailure) as occupied:
        phase_c.audit_gpu_admission(config, require_zero_compute_processes=True)
    assert occupied.value.error_code == "GPU_COMPUTE_PROCESS_OCCUPIED"
    _fake_nvidia(monkeypatch, phase_c, config, reordered=True)
    with pytest.raises(phase_c.ProtocolFailure) as reordered:
        phase_c.audit_gpu_admission(config, require_zero_compute_processes=True)
    assert reordered.value.error_code == "GPU_PHYSICAL_INDEX_UUID_MAPPING_MISMATCH"


def test_worker_uuid_mismatch_fails_before_cuda(phase_c, config, monkeypatch):
    _fake_nvidia(monkeypatch, phase_c, config)
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.audit_worker_gpu_binding(config, 0, "GPU-wrong")
    assert error.value.error_code == "WORKER_GPU_UUID_MISMATCH"


def test_primary_tasks_carry_complete_base_p_identity(phase_c, config):
    selection, panel = _synthetic_panel(config)
    freeze = phase_c.build_task_freeze(config, selection, panel, _seed_freeze())
    assert len(freeze["primary_tasks"]) == 60
    hp = {
        "lr": config["base_p"]["lr"],
        "dropout": config["base_p"]["dropout"],
        "hidden_dim": config["base_p"]["hidden"],
        "weight_decay": config["base_p"]["l2"],
        "gat_heads": 1,
        "sage_aggr": "mean",
        "gin_eps": 0.0,
        "gat_heads_by_layer": None,
        "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
    }
    assert all(
        row["joint_fingerprint"] == config["base_p"]["joint_fingerprint"]
        and row["architecture_fingerprint"] == config["base_p"]["architecture_fingerprint"]
        and row["hp_fingerprint"] == config["base_p"]["hp_fingerprint"]
        and row["hp"] == hp
        and row["max_epochs"] == 300
        and row["patience"] == 80
        for row in freeze["primary_tasks"]
    )
    assert freeze["budget"] == {
        "minimum_completed_branch": 60,
        "conditional_screening": 100,
        "conditional_safety": 80,
        "maximum": 240,
    }
    assert freeze["retry_rule"] == {
        "identical_task_only": True,
        "maximum_retries_per_task": 2,
        "stop_if_retry_task_fraction_exceeds": 0.05,
        "stop_if_third_attempt_fails": True,
    }
    assert freeze["stopping_rule"]["non_strong_primary_stops_after_gnn"] == 60
    assert freeze["stopping_rule"]["strong_primary_runs_screening_then_safety"] is True
    assert freeze["stopping_rule"]["candidate_seed_condition_replacement"] is False
    assert freeze["stopping_rule"]["imputation"] is False


def test_seed_hp_task_bindings_are_exact_and_collision_free(phase_c, config):
    selection, panel = _synthetic_panel(config)
    seeds = _seed_freeze()
    freeze = phase_c.build_task_freeze(config, selection, panel, seeds)
    seed_audit = {
        **seeds,
        "selected_seeds": seeds["primary_seeds"] + seeds["screening_seeds"] + seeds["safety_seeds"],
        "selected_collision_count": 0,
        "selected_collision_free": True,
        "partition_disjoint": True,
    }
    phase_c.validate_seed_task_binding(freeze, seed_audit)
    phase_c.validate_hp_task_binding(freeze, panel)
    broken = copy.deepcopy(seed_audit)
    broken["safety_seeds"][0] = broken["primary_seeds"][0]
    with pytest.raises(phase_c.ProtocolFailure):
        phase_c.validate_seed_task_binding(freeze, broken)


@pytest.mark.parametrize("field", ["joint_fingerprint", "hp_fingerprint"])
def test_base_p_opaque_fingerprint_mismatch_fails_without_recompute(phase_c, config, field):
    mutated = copy.deepcopy(config)
    mutated["base_p"][field] = "0" * 64
    with pytest.raises(phase_c.ProtocolFailure) as error:
        phase_c.validate_base_p_contract(mutated)
    assert error.value.error_code == "BASE_P_EXACT_CONTRACT_MISMATCH"


def test_decoder_points_and_twenty_configurations_unique(phase_c, config):
    selected, _ = _synthetic_panel(config)
    freeze = phase_c.decode_hp_screening(config, selected)
    assert freeze["configuration_count"] == 20
    assert len({row["joint_fingerprint"] for row in freeze["configurations"]}) == 20


def test_conditional_statistics_and_safety_rules_unchanged(phase_c):
    assert phase_c.classify_primary(0.003, 0.049, 0.00001) == "FIXED_LN_STRONG_SIGNAL"
    assert phase_c.classify_primary(0.002, 0.049, 0.00001) == "FIXED_LN_AUXILIARY_SIGNAL"
    assert phase_c.classify_primary(0.003, 0.05, 0.00001) == "FIXED_LN_NOT_CONFIRMED"
    assert phase_c.budget_for_primary_classification("FIXED_LN_STRONG_SIGNAL") == 240
    assert phase_c.budget_for_primary_classification("FIXED_LN_AUXILIARY_SIGNAL") == 60
    sources = {"oh1": "OH_BLIND_JOINT_RANDOM", "oh2": "OH_BLIND_JOINT_RANDOM", "hr1": "HR_BALANCED_RANDOM_HP", "hr2": "HR_BALANCED_RANDOM_HP"}
    assert phase_c.safety_pass({"oh1": 0.002, "oh2": 0.001, "hr1": 0.001, "hr2": -0.0005}, sources)
    assert not phase_c.safety_pass({"oh1": 0.004, "oh2": -0.004, "hr1": 0.001, "hr2": 0.001}, sources)


def test_failure_before_task_one_has_classification_null_chain(phase_c, tmp_path):
    result = _make_result_skeleton(phase_c, tmp_path)
    output = phase_c._write_fail_closed_terminal(
        result, phase_c.ProtocolFailure("BEFORE_TASK_1", "SYNTHETIC_FAILURE")
    )
    decision = json.loads((result / "formal/decision.json").read_text())
    assert output["classification"] is None
    assert decision["scientific_decision"] == "NOT_REACHED"
    assert decision["test_accessed"] is False
    assert decision["phase_d_launched"] is False
    _assert_archive_union(phase_c, result)


@pytest.mark.parametrize("failure_stage", ["MID_PRIMARY", "BEFORE_CONDITIONAL_SCREENING"])
def test_formal_midstage_failures_are_not_scientific_negatives(phase_c, tmp_path, failure_stage):
    result = _make_result_skeleton(phase_c, tmp_path)
    assignment = {
        "stage": "primary",
        "tasks": [{"task_id": "t1", "worker_id": 0}],
    }
    phase_c._write_formal_json(result / "formal/assignment_ledgers/primary.json", assignment)
    access = phase_c._new_access_ledger()
    shard = {
        "ledger_complete": True,
        "test_access": access,
        "results": [{"task_id": "t1"}],
        "attempts": [{"task_id": "t1", "attempt": 1, "test_access": access}],
    }
    phase_c._write_formal_json(result / "formal/shards/primary_worker_0.json", shard)
    phase_c._write_fail_closed_terminal(
        result, phase_c.ProtocolFailure(failure_stage, "SYNTHETIC_FAILURE")
    )
    decision = json.loads((result / "formal/decision.json").read_text())
    assert decision["classification"] is None
    assert decision["scientific_decision"] == "NOT_REACHED"
    assert decision["test_accessed"] is False
    _assert_archive_union(phase_c, result)


def test_crash_after_setup_before_attempt_is_unverified_not_false(phase_c, tmp_path):
    result = _make_result_skeleton(phase_c, tmp_path)
    phase_c._write_formal_json(
        result / "formal/assignment_ledgers/primary.json",
        {"stage": "primary", "tasks": [{"task_id": "t1", "worker_id": 0}]},
    )
    phase_c._write_formal_json(
        result / "formal/shards/primary_worker_0.json",
        {
            "ledger_complete": False,
            "setup_complete": True,
            "setup_test_access": phase_c._new_access_ledger(),
            "test_access": phase_c._new_access_ledger(),
            "results": [],
            "attempts": [],
        },
    )
    phase_c._write_fail_closed_terminal(
        result, phase_c.ProtocolFailure("AFTER_SETUP", "SYNTHETIC_CRASH")
    )
    decision = json.loads((result / "formal/decision.json").read_text())
    assert decision["test_accessed"] is None
    assert decision["test_access_status"] == "UNVERIFIED_FAIL_CLOSED"
    assert decision["classification"] is None


def test_detected_test_access_is_true_even_if_ledger_incomplete(phase_c, tmp_path):
    result = _make_result_skeleton(phase_c, tmp_path)
    phase_c._write_formal_json(
        result / "formal/assignment_ledgers/primary.json",
        {"stage": "primary", "tasks": [{"task_id": "t1", "worker_id": 0}]},
    )
    access = phase_c._new_access_ledger()
    access["test_accessed"] = True
    phase_c._write_formal_json(
        result / "formal/shards/primary_worker_0.json",
        {"ledger_complete": False, "test_access": access, "results": [], "attempts": []},
    )
    phase_c._write_fail_closed_terminal(
        result, phase_c.ProtocolFailure("FIREWALL", "SYNTHETIC_TEST_ACCESS")
    )
    decision = json.loads((result / "formal/decision.json").read_text())
    assert decision["test_accessed"] is True
    assert decision["test_access_status"] == "DETECTED"


def test_success_archive_has_non_circular_outer_chain(phase_c, tmp_path):
    result = _make_result_skeleton(phase_c, tmp_path)
    for name, value in {
        "decision.json": {"state": "COMPLETE", "classification": "SYNTHETIC"},
        "state.json": {"state": "COMPLETE"},
        "task_attempt_ledger.json": {"attempts": []},
        "test_access_ledger.json": {"test_accessed": False},
    }.items():
        phase_c._write_formal_json(result / f"formal/{name}", value)
    phase_c._write_formal_json(
        result / "formal/terminal_record.json",
        {
            "state": "COMPLETE",
            "decision_sha256": phase_c.sha256_path(result / "formal/decision.json"),
            "state_sha256": phase_c.sha256_path(result / "formal/state.json"),
        },
    )
    archive = phase_c._finalize_formal_archive(result)
    phase_c._write_outer_completion(result, "COMPLETE", archive)
    completion = json.loads((result / "formal/completion.json").read_text())
    assert completion["non_authoritative_outer_envelope"] is True
    _assert_archive_union(phase_c, result)


def test_formal_start_directory_appears_with_bound_start_atomically(
    phase_c, tmp_path, monkeypatch
):
    formal = tmp_path / "formal"
    original_mkdtemp = phase_c.tempfile.mkdtemp
    observed = {}

    def same_parent_mkdtemp(*, prefix, dir):
        observed["prefix"] = prefix
        observed["dir"] = Path(dir)
        return original_mkdtemp(prefix=prefix, dir=dir)

    monkeypatch.setattr(phase_c.tempfile, "mkdtemp", same_parent_mkdtemp)
    phase_c._create_formal_with_atomic_start(formal, {"state": "FORMAL_STARTED", "run_id": "r"})
    assert observed == {
        "prefix": "phase_c_v2_formal_start_",
        "dir": formal.parent,
    }
    assert json.loads((formal / "start.json").read_text()) == {
        "state": "FORMAL_STARTED",
        "run_id": "r",
    }
    with pytest.raises(phase_c.ProtocolFailure):
        phase_c._create_formal_with_atomic_start(formal, {"state": "FORMAL_STARTED"})


def _launch_repo(phase_c, tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    for relative in phase_c.TARGET_RELS:
        source = REPO_ROOT / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    result = repo / phase_c.RESULT_REL
    result.parent.mkdir(parents=True, exist_ok=True)
    assets = {
        f"source_snapshot/{relative}": (repo / relative).read_bytes()
        for relative in phase_c.TARGET_RELS
    }
    phase_c.write_prelaunch(result, assets)
    monkeypatch.setattr(phase_c, "REPO_ROOT", repo)
    return repo, result


def test_launch_request_is_exclusive_and_binds_all_fields(phase_c, tmp_path, monkeypatch):
    repo, result = _launch_repo(phase_c, tmp_path, monkeypatch)
    output = phase_c.record_launch_request(repo / phase_c.CONFIG_REL)
    request = json.loads((result / "launch/launch_request.json").read_text())
    assert output["gnn_evaluations"] == 0
    assert request["prelaunch_manifest_root_sha256"] == json.loads(
        (result / "prelaunch_manifest.json").read_text()
    )["manifest_root_sha256"]
    assert request["target_after_sha256"] == {
        relative: phase_c.sha256_path(repo / relative) for relative in phase_c.TARGET_RELS
    }
    assert request["fixed_tmux_command_sha256"] == phase_c.sha256_bytes(
        phase_c.FIXED_TMUX_COMMAND.encode()
    )
    assert request["run_id"].startswith("phase-c-v2-")
    with pytest.raises(phase_c.ProtocolFailure):
        phase_c.record_launch_request(repo / phase_c.CONFIG_REL)


def test_runner_entry_and_preformal_failure_bind_same_request(phase_c, tmp_path, monkeypatch):
    repo, result = _launch_repo(phase_c, tmp_path, monkeypatch)
    phase_c.record_launch_request(repo / phase_c.CONFIG_REL)
    phase_c.record_runner_entered(repo / phase_c.CONFIG_REL)
    request = json.loads((result / "launch/launch_request.json").read_text())
    entry = json.loads((result / "launch/runner_entered.json").read_text())
    assert entry["run_id"] == request["run_id"]
    assert entry["launch_request_sha256"] == phase_c.sha256_path(
        result / "launch/launch_request.json"
    )
    phase_c.record_runner_exit(repo / phase_c.CONFIG_REL)
    failure = json.loads((result / "launch/preformal_failure.json").read_text())
    assert failure["run_id"] == request["run_id"]
    assert failure["classification"] is None


def test_runner_exit_after_start_is_bound_and_non_scientific(phase_c, tmp_path, monkeypatch):
    repo, result = _launch_repo(phase_c, tmp_path, monkeypatch)
    phase_c.record_launch_request(repo / phase_c.CONFIG_REL)
    phase_c.record_runner_entered(repo / phase_c.CONFIG_REL)
    (result / "formal").mkdir()
    phase_c._write_formal_json(result / "formal/start.json", {"state": "FORMAL_STARTED"})
    phase_c.record_runner_exit(repo / phase_c.CONFIG_REL)
    record = json.loads((result / "launch/runner_exit.json").read_text())
    request = json.loads((result / "launch/launch_request.json").read_text())
    assert record["run_id"] == request["run_id"]
    assert record["classification"] is None
    assert record["scientific_decision"] == "NOT_REACHED"


def test_formal_and_worker_reverse_binding_order(phase_c):
    parent = phase_c.inspect.getsource(phase_c.formal_run)
    assert parent.index("verify_live_source_snapshot_binding") < parent.index(
        "_create_formal_with_atomic_start"
    )
    assert parent.index("audit_gpu_admission") < parent.index("_create_formal_with_atomic_start")
    assert '"launch_request_sha256": request_sha' in parent
    assert '"run_id": request["run_id"]' in parent
    assert '"runner_entered_sha256"' in parent
    worker = phase_c.inspect.getsource(phase_c._formal_worker)
    assert worker.index("_formal_bound_file_check") < worker.index("import torch")
    assert worker.index("audit_worker_gpu_binding") < worker.index("import torch")
    assert worker.index("audit_worker_gpu_binding") < worker.index("_write_formal_json(shard_path")
    assert worker.index("_write_formal_json(shard_path") < worker.index("run_guarded_evaluation")


def test_runner_clears_inherited_cuda_and_never_launches_other_phases(phase_c):
    runner = (REPO_ROOT / phase_c.RUNNER_REL).read_text()
    assert "unset CUDA_VISIBLE_DEVICES" in runner
    assert "--runner-enter" in runner
    assert "--formal-run" in runner
    assert "--runner-exit" in runner
    assert "phase_b" not in runner.casefold()
    assert "phase_d" not in runner.casefold()
    assert "phase_e" not in runner.casefold()


def test_synthetic_smoke_contract_is_zero_gnn(phase_c):
    projector = phase_c.synthetic_projection_invariance()
    firewall = phase_c.synthetic_firewall_evidence()
    assert projector["status"] == "PASS"
    assert firewall["formal_test_accessed"] is False
    assert firewall["synthetic_only"] is True
