from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "analyse/boundary_training_ceiling_audit.py"
SPEC = importlib.util.spec_from_file_location("boundary_training_ceiling_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def configuration(
    source: str,
    index: int,
    *,
    mean: float,
    hidden: int = 256,
    l2: float = 5e-4,
    u_lr: float = 0.5,
    u_dropout: float = 0.5,
    architecture: str | None = None,
) -> dict:
    return {
        "source": source,
        "joint_fingerprint": f"{index:064x}",
        "architecture_fingerprint": architecture or f"{index + 1000:064x}",
        "hp_fingerprint": f"{index + 2000:064x}",
        "operations": ["GCNConv"] * 5,
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
        "raw_hp_vector": [u_lr, u_dropout, 0.5, 0.5],
        "effective_layers": 5,
        "lr": 1e-3,
        "dropout": 0.35,
        "hidden": hidden,
        "l2": l2,
        "u_lr": u_lr,
        "u_dropout": u_dropout,
        "u_l2": audit.u_log(l2, audit.FROZEN_L2[0], audit.FROZEN_L2[-1]),
        "seed_values": [mean, mean, mean],
        "mean_val": mean,
    }


def training_records(source: str, count: int = 100) -> list[dict]:
    rows = []
    for index in range(count):
        top_like = index >= count - math.ceil(0.10 * count)
        for seed in audit.FROZEN_SEEDS:
            rows.append(
                {
                    "source": source,
                    "joint_fingerprint": f"{index:064x}",
                    "seed": seed,
                    "val_acc": float(index),
                    "terminal_status": "success",
                    "valid": True,
                    "best_epoch": 285 if top_like else 100,
                    "stopped_epoch": 300 if top_like else 180,
                    "epochs_ran": 300 if top_like else 180,
                    "eval_epochs": 300,
                    "patience": 80,
                }
            )
    return rows


def hr_manifest_source(expected: int) -> dict:
    return {
        "name": "HR_BALANCED_RANDOM_HP",
        "strategy": "BALANCED_RANDOM_HP",
        "expected_configurations": expected,
        "task_stage": "initial",
        "frozen_selection_task_stage_policy": audit.FROZEN_STAGE_KEY_ABSENT_POLICY,
    }


def selected_manifest_row(index: int, *, stage: str = "initial", strategy: str = "BALANCED_RANDOM_HP") -> dict:
    return {
        "joint_config_fingerprint": f"{index:064x}",
        "selected_by": [{"strategy": strategy}],
        "task_stage": stage,
    }


def frozen_manifest_row(index: int, *, strategy: str = "BALANCED_RANDOM_HP") -> dict:
    return {
        "joint_config_fingerprint": f"{index:064x}",
        "strategy": strategy,
    }


def test_frozen_config_and_firewall() -> None:
    config = json.loads((ROOT / audit.CONFIG_RELATIVE).read_text(encoding="utf-8"))
    audit.validate_frozen_config(config)
    result = audit.static_zero_compute_firewall(config)
    assert result["passed"] is True
    assert result["gnn_training_calls"] == result["gpu_compute_calls"] == result["test_data_reads"] == 0
    assert result["runner_exact_command_verified"] is True
    bad = copy.deepcopy(config)
    bad["gates"]["q90_gap_min"] = 0.002
    with pytest.raises(audit.ProtocolError):
        audit.validate_frozen_config(bad)
    bad = copy.deepcopy(config)
    bad["required_dependency_hashes"] = {}
    with pytest.raises(audit.ProtocolError):
        audit.validate_frozen_config(bad)
    bad = copy.deepcopy(config)
    bad["input_sources"][0]["strategy"] = "HIGH_REGIME_PRIOR"
    with pytest.raises(audit.ProtocolError):
        audit.validate_frozen_config(bad)
    hr_index = next(index for index, row in enumerate(config["input_sources"]) if row["name"] == "HR_BALANCED_RANDOM_HP")
    bad = copy.deepcopy(config)
    del bad["input_sources"][hr_index]["frozen_selection_task_stage_policy"]
    with pytest.raises(audit.ProtocolError):
        audit.validate_frozen_config(bad)
    bad = copy.deepcopy(config)
    bad["input_sources"][0]["frozen_selection_task_stage_policy"] = audit.FROZEN_STAGE_KEY_ABSENT_POLICY
    with pytest.raises(audit.ProtocolError):
        audit.validate_frozen_config(bad)


def test_real_hr_manifest_schema_adapter_uses_strategy_level_counts() -> None:
    config = json.loads((ROOT / audit.CONFIG_RELATIVE).read_text(encoding="utf-8"))
    source = next(row for row in config["input_sources"] if row["name"] == "HR_BALANCED_RANDOM_HP")
    root = ROOT / source["root"]
    selected_raw = json.loads((root / source["selection"]).read_text(encoding="utf-8"))
    frozen_raw = json.loads((root / source["frozen_selection"]).read_text(encoding="utf-8"))
    selected, frozen, schema = audit.select_primary_manifest_rows(source, selected_raw, frozen_raw)
    assert len(selected_raw) == schema["selected_manifest_total_rows"] == 336
    assert len(frozen_raw) == schema["frozen_manifest_total_rows"] == 384
    assert schema["selected_strategy_rows"] == schema["selected_rows_after_stage_filter"] == 192
    assert schema["frozen_strategy_rows"] == 192
    assert len(selected) == len(frozen) == 192
    assert all(row.get("task_stage") == "initial" for row in selected)
    assert all("task_stage" not in row for row in frozen)
    assert {row["joint_config_fingerprint"] for row in selected} == {
        row["joint_config_fingerprint"] for row in frozen
    }


def test_hr_manifest_schema_adapter_accepts_only_explicit_allowlisted_shape() -> None:
    source = hr_manifest_source(3)
    selected = [selected_manifest_row(index) for index in range(3)]
    selected.extend(selected_manifest_row(100 + index, strategy="OTHER") for index in range(2))
    selected.append(selected_manifest_row(150, stage="followup"))
    frozen = [frozen_manifest_row(index) for index in range(3)]
    frozen.extend(frozen_manifest_row(200 + index, strategy="OTHER") for index in range(4))
    selected_rows, frozen_rows, schema = audit.select_primary_manifest_rows(source, selected, frozen)
    assert len(selected_rows) == len(frozen_rows) == 3
    assert schema["selected_manifest_total_rows"] == 6
    assert schema["selected_strategy_rows"] == 4
    assert schema["selected_rows_after_stage_filter"] == 3
    assert schema["frozen_manifest_total_rows"] == 7
    wrong_source = {**source, "name": "NOT_ALLOWLISTED"}
    with pytest.raises(audit.ProtocolError):
        audit.select_primary_manifest_rows(wrong_source, selected, frozen)


@pytest.mark.parametrize("explicit_stage", ["initial", None], ids=["explicit_initial", "explicit_null"])
def test_hr_frozen_manifest_rejects_any_explicit_or_mixed_stage_key(explicit_stage: str | None) -> None:
    source = hr_manifest_source(3)
    selected = [selected_manifest_row(index) for index in range(3)]
    frozen = [frozen_manifest_row(index) for index in range(3)]
    frozen[0]["task_stage"] = explicit_stage
    with pytest.raises(audit.ProtocolError):
        audit.select_primary_manifest_rows(source, selected, frozen)


@pytest.mark.parametrize("explicit_stage", ["initial", "followup", None])
def test_hr_frozen_manifest_rejects_uniform_explicit_stage_schema(explicit_stage: str | None) -> None:
    source = hr_manifest_source(3)
    selected = [selected_manifest_row(index) for index in range(3)]
    frozen = [frozen_manifest_row(index) for index in range(3)]
    for row in frozen:
        row["task_stage"] = explicit_stage
    with pytest.raises(audit.ProtocolError):
        audit.select_primary_manifest_rows(source, selected, frozen)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "set_mismatch"])
def test_hr_manifest_count_unique_and_set_mismatches_fail_closed(mutation: str) -> None:
    source = hr_manifest_source(3)
    selected = [selected_manifest_row(index) for index in range(3)]
    frozen = [frozen_manifest_row(index) for index in range(3)]
    if mutation == "missing":
        frozen.pop()
    elif mutation == "extra":
        frozen.append(frozen_manifest_row(99))
    elif mutation == "duplicate":
        frozen[-1] = frozen_manifest_row(0)
    else:
        frozen[-1] = frozen_manifest_row(99)
    with pytest.raises(audit.ProtocolError):
        audit.select_primary_manifest_rows(source, selected, frozen)


def test_fingerprint_semantics_are_position_sensitive() -> None:
    operations = ["GCNConv", "GATConv"]
    edges = [[1, 2], [0, 1]]
    first = audit.architecture_fingerprint(operations, edges)
    assert first == audit.architecture_fingerprint(operations, list(reversed(edges)))
    assert first != audit.architecture_fingerprint(list(reversed(operations)), edges)
    hp = audit.hp_fingerprint([0.1, 0.2, 0.3, 0.4])
    assert audit.joint_fingerprint(first, hp) == audit.stable_hash(
        {"architecture_fingerprint": first, "hp_fingerprint": hp}
    )


def test_decoder_boundaries_and_continuous_bins() -> None:
    low = audit.decoded_from_raw([0.0, 0.0, 0.0, 0.0])
    high = audit.decoded_from_raw([1.0, 1.0, 1.0, 1.0])
    assert low == {"lr": pytest.approx(1e-4), "dropout": pytest.approx(0.1), "hidden": 16, "l2": 1e-5}
    assert high == {"lr": pytest.approx(10 ** -1.5), "dropout": pytest.approx(0.6), "hidden": 512, "l2": 1e-3}
    assert audit._continuous_bin(0.0) == 0
    assert audit._continuous_bin(0.2) == 1
    assert audit._continuous_bin(0.8) == 4
    assert audit._continuous_bin(1.0) == 4
    assert audit.u_log(5e-4, 1e-5, 1e-3) == pytest.approx(0.8494850021680094)


def test_q90_uses_linear_quantile() -> None:
    values = [0.0, 1.0, 2.0, 3.0]
    assert audit.q90(values) == pytest.approx(np.quantile(values, 0.9, method="linear"))


def test_top_tail_uses_fixed_ceil_and_equal_source_weights() -> None:
    primary = {}
    for source_index, source in enumerate(audit.PRIMARY_NAMES):
        rows = []
        for index in range(11):
            upper = index < (1 if source_index == 0 else 5)
            rows.append(
                configuration(
                    source,
                    source_index * 100 + index,
                    mean=1.0 if index < 2 else 0.5,
                    hidden=512 if upper else 256,
                    l2=1e-3 if upper else 5e-4,
                    u_lr=0.95 if upper else 0.5,
                    u_dropout=0.95 if upper else 0.5,
                )
            )
        primary[source] = rows
    rows, result = audit.top_tail_analysis(primary, (0.25, 0.10, 0.05))
    assert sum(row["n"] == 2 for row in rows if row["source"] in audit.PRIMARY_NAMES and row["subset"] == "Top10") == 16
    hidden_all = result["indicators"]["hidden_512"]["All"]
    assert hidden_all["equal_weight_proportion"] == pytest.approx((1 / 11 + 5 / 11) / 2)


def test_matching_exact_first_and_nonreuse() -> None:
    source = audit.PRIMARY_NAMES[0]
    arch = "a" * 64
    rows = [
        configuration(source, 1, mean=0.70, hidden=16, architecture=arch, u_lr=0.5, u_dropout=0.5),
        configuration(source, 2, mean=0.71, hidden=32, architecture=arch, u_lr=0.5, u_dropout=0.5),
        configuration(source, 3, mean=0.72, hidden=64, architecture=arch, u_lr=0.51, u_dropout=0.51),
        configuration(source, 4, mean=0.73, hidden=128, architecture=arch, u_lr=0.52, u_dropout=0.52),
    ]
    pairs = audit.greedy_pairs(rows, "hidden")
    assert pairs[0]["match_type"] == "exact"
    fingerprints = [value for row in pairs for value in (row["smaller_fingerprint"], row["larger_fingerprint"])]
    assert len(fingerprints) == len(set(fingerprints))


def test_matched_gate_is_insufficient_below_twenty() -> None:
    primary = {}
    for source_index, source in enumerate(audit.PRIMARY_NAMES):
        rows = []
        for index in range(10):
            arch = f"{source_index + 1:02x}{index // 2:062x}"
            rows.append(
                configuration(
                    source,
                    source_index * 100 + index,
                    mean=0.7 + index / 1000,
                    hidden=16 if index % 2 == 0 else 32,
                    architecture=arch,
                )
            )
        primary[source] = rows
    _, summary = audit.matched_analysis(primary)
    assert summary["hidden"]["pair_count"] == 10
    assert summary["hidden"]["status"] == "INSUFFICIENT_MATCHED_PAIRS"
    assert summary["hidden"]["passed"] is False


def test_interaction_empty_adjacent_cell_is_invalid() -> None:
    primary = {}
    for source_index, source in enumerate(audit.PRIMARY_NAMES):
        rows = []
        for index in range(30):
            rows.append(configuration(source, source_index * 100 + index, mean=0.8, hidden=512, l2=1e-3))
        for index in range(30, 60):
            rows.append(configuration(source, source_index * 100 + index, mean=0.79, hidden=256, l2=1e-3))
        primary[source] = rows
    result = audit.interaction_analysis(primary, 0, np.random.default_rng(1))
    assert result["valid"] is False
    assert result["interaction"] is False
    assert result["status"] == "INSUFFICIENT_INTERACTION_CELLS"


def test_training_logic_and_equal_source_weighting() -> None:
    records = {name: training_records(name) for name in (*audit.PRIMARY_NAMES, *audit.SECONDARY_NAMES)}
    detailed, result = audit.training_audit(records)
    assert len(detailed) == 5 * 100 * 3
    assert result["sufficient"] is True
    assert result["T1"]["passed"] is True
    assert result["T3"]["passed"] is True
    assert result["source_direction_count"] == 5
    assert result["training_truncation_signal"] is True
    assert {row["early_stop_reason"] for row in detailed} == {"patience", "max_epoch"}


def test_training_missing_fields_is_insufficient_not_exception() -> None:
    records = {name: training_records(name) for name in (*audit.PRIMARY_NAMES, *audit.SECONDARY_NAMES)}
    for row in records[audit.PRIMARY_NAMES[0]]:
        row["best_epoch"] = None
    _, result = audit.training_audit(records)
    assert result["sufficient"] is False
    assert result["status"] == "TRAINING_AUDIT_INSUFFICIENT"
    assert result["training_truncation_signal"] is False


def test_secondary_generator_membership_fails_closed() -> None:
    audit.validate_evaluated_generator_membership({"a", "b"}, {"a", "b", "c"}, "source")
    with pytest.raises(audit.ProtocolError):
        audit.validate_evaluated_generator_membership({"a", "outside"}, {"a", "b"}, "source")


@pytest.mark.parametrize(
    ("boundary", "training_signal", "sufficient", "expected"),
    [
        (True, True, True, "BOUNDARY_AND_TRAINING_BOTH_PLAUSIBLE"),
        (True, False, True, "HP_BOUNDARY_EXPANSION_AUTHORIZED"),
        (True, False, False, "HP_BOUNDARY_EXPANSION_AUTHORIZED"),
        (False, True, True, "TRAINING_CEILING_PRIORITY"),
        (False, False, True, "NO_SIMPLE_BOUNDARY_OR_TRAINING_SIGNAL"),
        (False, False, False, "AUDIT_INPUT_INCOMPATIBLE"),
    ],
)
def test_classification_table(boundary: bool, training_signal: bool, sufficient: bool, expected: str) -> None:
    top = {"gates": {"H-H1": {"passed": boundary}, "L-H1": {"passed": boundary}}}
    q90_result = {"gates": {"H-H2": {"passed": boundary}, "L-H2": {"passed": boundary}}}
    matched = {"hidden": {"passed": False}, "l2": {"passed": False}}
    interaction = {"interaction": False}
    training = {
        "training_truncation_signal": training_signal,
        "sufficient": sufficient,
        "status": "TRAINING_AUDIT_SUFFICIENT" if sufficient else "TRAINING_AUDIT_INSUFFICIENT",
    }
    result = audit.classify_decision(top, q90_result, matched, interaction, training)
    assert result["classification"] == expected


def test_bootstrap_seed_uses_first_eight_bytes() -> None:
    digest = "0001020304050607" + "00" * 24
    assert audit.bootstrap_seed_from_protocol(digest) == 0x0001020304050607


def test_archive_does_not_include_itself(tmp_path: Path) -> None:
    output = tmp_path / audit.EXPERIMENT
    output.mkdir()
    (output / "payload.txt").write_text("payload\n", encoding="utf-8")
    archive, digest = audit._create_archive(output)
    assert digest == audit.sha256_file(archive)
    with audit.tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert not any(name.endswith(".tar.gz") or name.endswith(".tar.gz.sha256") for name in names)


def test_formal_start_rechecks_script_config_and_runner(tmp_path: Path) -> None:
    output = tmp_path / audit.EXPERIMENT
    output.mkdir()
    config_path = audit.REPO_ROOT / audit.CONFIG_RELATIVE
    estimate = {
        "dry_run_passed": True,
        "script_sha256": audit.sha256_file(MODULE_PATH),
        "config_sha256": audit.sha256_file(config_path),
        "runner_sha256": audit.sha256_file(audit.REPO_ROOT / audit.RUNNER_RELATIVE),
    }
    audit.dump_json(output / "pre_execution_estimate.json", estimate)
    audit._validate_formal_start(output, estimate, config_path)
    (output / "unexpected.txt").write_text("collision\n", encoding="utf-8")
    with pytest.raises(audit.EngineeringError):
        audit._validate_formal_start(output, estimate, config_path)


def test_synthetic_smoke_is_zero_work() -> None:
    result = audit.synthetic_smoke()
    assert result["passed"] is True
    assert result["gpu"] == result["new_gnn"] == result["test_access"] == 0
