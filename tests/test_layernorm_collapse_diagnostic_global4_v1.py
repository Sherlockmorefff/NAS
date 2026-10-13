from __future__ import annotations

from collections import Counter
import copy
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "analyse/layernorm_collapse_diagnostic_global4_v1.py"
SPEC = importlib.util.spec_from_file_location("layernorm_collapse_diagnostic_global4_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
CONFIG_PATH = REPO_ROOT / "configs/layernorm_collapse_diagnostic_global4_v1.json"


@pytest.fixture(scope="module")
def config():
    return diagnostic.load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def selection(config):
    diagnostic.verify_pinned_files(config)
    return diagnostic.freeze_selection(config)


@pytest.fixture(scope="module")
def seeds(config):
    explicit_used = set()
    for source in config["seed_freeze"]["ledger_sources"]:
        document = diagnostic.read_json(REPO_ROOT / source["path"], reject_test_metrics=True)
        explicit_used.update(int(row["seed"]) for row in document["attempts"])
    # Independent executed-task ledgers prove every value in this range was used.
    explicit_used.update(range(2000, 2030))
    window_audit = {
        "schema_version": "candidate-window-seed-semantics-audit-v1",
        "window_start_inclusive": 2000,
        "window_end_inclusive": 2064,
        "all_unresolved_value_count_in_proof_window": 1,
        "all_unresolved_values_in_proof_window": [2000],
        "independently_recognized_used_unresolved_value_count_in_proof_window": 1,
        "independently_recognized_used_unresolved_values_in_proof_window": [2000],
        "blocking_unresolved_candidate_value_count": 0,
        "blocking_unresolved_candidate_values": [],
        "all_unresolved_group_count": 1,
        "all_unresolved_occurrence_count": 1,
        "all_unresolved_value_count": 1,
        "all_unresolved_values_commitment_sha256": diagnostic.commitment([2000]),
        "provisional_first_35": list(range(2030, 2065)),
        "exact_first_35_selection_provable": True,
    }
    window_audit["commitment_sha256"] = diagnostic.commitment(window_audit)
    synthetic_inventory = {
        "schema_version": "focused-test-inventory-v1",
        "scan_roots": ["analyse", "configs", "results", "tests"],
        "excluded_target": config["result_root"],
        "source_count": 0,
        "seed_field_match_count": 0,
        "executed_gnn_training_seeds": sorted(explicit_used),
        "skipped_test_field_count": 0,
        "skipped_test_field_path_commitment_sha256": "0" * 64,
        "test_value_policy": "focused test fixture",
        "inventory_root_sha256": "1" * 64,
        "candidate_window_audit": window_audit,
    }
    return diagnostic.freeze_seeds(config, synthetic_inventory)


@pytest.fixture(scope="module")
def conditions(config):
    return diagnostic.freeze_conditions(config)


@pytest.fixture(scope="module")
def tasks(selection, seeds, conditions, config):
    return diagnostic.build_tasks(selection, seeds, conditions, config)


def test_real_phase_d_selection_is_exact_and_disjoint(selection):
    expected = {
        "collapse": [
            "c5228f953800368187c751782dd8aed6550ed27a10a47ab976526bc2b9997251",
            "71ae4e64946929a92ae1ef61e883959e8d1b375451d9032940a601ca72268809",
            "3cf27e75be02d68dc278a999f220e5d58945caac536ef83e157051e32023581c",
            "0e378861b724c6b3c2b6ad4055c15552db93e1c493779a9db34911b78115d10a",
        ],
        "improved": [
            "5cde9d9d5e5fe6a52ccc77783bce0d58d92fb6e37370aa23a2a12c02936f3eaf",
            "0b5e55b929e8c3479f4f4905736bea1c747c77d849fc011a09521ba06c58ebe0",
            "1543fad7e3556ee0926aca0300e4f2eaa7c7c3497d89b91e42e5cb881d196827",
            "3cef3ae38d0369e22bce48516c4dac0e08e6ef414d140a5e72ee6cc1fe6526ac",
        ],
        "normal_decline": [
            "5ffc17b5d01b6c942c626f91ffa1f1b421221c68c7d95b1ae1e5c3f5f4ae1cc6",
            "8b7cc9b6a22d91e6a36ef7aa445569147b94d9b25b4a1b435d771dc61c365ce5",
            "92983b9a894d91040a60ef3fb6fe1844e83c595e77dcfafe9a18b705f84d3081",
            "3997ffea77cec765927245b02a132e4a799fc34b3176ef03e50e62dde70ac790",
        ],
    }
    observed = {
        group["name"]: [row["joint_config_fingerprint"] for row in group["configs"]]
        for group in selection["groups"]
    }
    assert selection["complete_paired_config_count"] == 192
    assert selection["qualifying_counts"] == {
        "collapse": 12,
        "positive_improvement_remaining": 77,
        "normal_decline_remaining": 30,
    }
    assert observed == expected
    flattened = [fingerprint for values in observed.values() for fingerprint in values]
    assert len(flattened) == len(set(flattened)) == 12


def _selection_row(fingerprint, control, layernorm, delta=None):
    return {
        "joint_config_fingerprint": fingerprint,
        "control_mean": control,
        "layernorm_mean": layernorm,
        "delta": layernorm - control if delta is None else delta,
        "layernorm_by_seed": [
            {"seed": seed, "validation_accuracy": layernorm} for seed in (0, 1, 2)
        ],
    }


def test_selection_tie_breaks_and_insufficient_group_are_predeclared():
    complete = []
    for index, fingerprint in enumerate(("d", "b", "a", "c", "e")):
        complete.append(_selection_row(fingerprint, 0.80 - index * 0.01, 0.316))
    for fingerprint in ("j", "h", "f", "g", "i"):
        complete.append(_selection_row(fingerprint, 0.60, 0.70))
    for index, fingerprint in enumerate(("o", "n", "m", "l", "k")):
        complete.append(_selection_row(fingerprint, 0.80, 0.75 + index * 0.005))
    groups, counts = diagnostic.select_configuration_groups(complete)
    assert [row["joint_config_fingerprint"] for row in groups["collapse"]] == [
        "d",
        "b",
        "a",
        "c",
    ]
    assert [row["joint_config_fingerprint"] for row in groups["improved"]] == [
        "f",
        "g",
        "h",
        "i",
    ]
    assert len(groups["normal_decline"]) == 4
    assert counts == {
        "collapse": 5,
        "positive_improvement_remaining": 5,
        "normal_decline_remaining": 5,
    }
    insufficient, _ = diagnostic.select_configuration_groups(complete[:3])
    assert any(len(rows) < 4 for rows in insufficient.values())


def test_actual_seed_scan_and_commitment_are_frozen(seeds):
    assert seeds["diagnostic_seeds"] == [2030, 2031, 2032, 2033, 2034]
    assert seeds["reserved_confirmation_seeds"] == list(range(2035, 2065))
    assert seeds["commitment_sha256"] == (
        "c6a823ab525d1c9f15de5bc634df96cb429e2e1b53a36ecbb99a77009ee58d51"
    )
    assert diagnostic.commitment(seeds["commitment_payload"]) == seeds[
        "commitment_sha256"
    ]
    assert diagnostic.commitment(seeds["global_inventory"]) == seeds[
        "global_inventory_attestation_sha256"
    ]
    assert diagnostic.validate_frozen_seed_record(
        seeds, diagnostic.load_config(CONFIG_PATH)
    )["passed"] is True
    used = {
        seed
        for ledger in seeds["ledger_sources"]
        for seed in ledger["used_seeds"]
    }
    assert not used.intersection(seeds["all_selected_seeds"])
    assert not set(seeds["diagnostic_seeds"]).intersection(
        seeds["reserved_confirmation_seeds"]
    )


def test_frozen_seed_commitments_reject_tampering(seeds, config):
    broken_payload = copy.deepcopy(seeds)
    broken_payload["commitment_payload"]["diagnostic_seeds"][0] = 2999
    with pytest.raises(diagnostic.ProtocolError, match="COMMITMENT_PAYLOAD_MISMATCH"):
        diagnostic.validate_frozen_seed_record(broken_payload, config)
    broken_inventory = copy.deepcopy(seeds)
    broken_inventory["global_inventory"]["source_count"] = 999
    with pytest.raises(diagnostic.ProtocolError, match="GLOBAL_INVENTORY_ATTESTATION_MISMATCH"):
        diagnostic.validate_frozen_seed_record(broken_inventory, config)


def test_ascending_unused_seed_scanner_skips_actual_values():
    assert diagnostic.ascending_unused_seeds({2000, 2002, 2005}, 2000, 5) == [
        2001,
        2003,
        2004,
        2006,
        2007,
    ]


def test_candidate_window_ambiguity_is_fail_closed_only_when_relevant():
    retained = diagnostic.retain_unresolved_seed_finding(
        {"semantic": "ambiguous", "seeds": [42, 2050, 100000]}
    )
    assert retained["semantic"] == "unresolved_seed_semantics_for_window_audit"
    assert retained["seeds"] == [42, 2050, 100000]


def test_candidate_window_inventory_wrapper_restores_and_fails_closed(
    config, monkeypatch, tmp_path
):
    import analyse.final_checkpoint_seed_inventory as scanner

    def exercise(seed, *, recognized_used=(), builder_failure=False):
        def base_normalize_relpath(value):
            return Path(str(value)).as_posix().lstrip("/")

        def base_structured_kind(_name):
            return "json"

        def base_walker(_value, *, source, path="$"):
            yield {
                "source": source,
                "field_path": f"{path}/seed",
                "semantic": "ambiguous",
                "executed_gnn": False,
                "seeds": [seed],
                "occurrence_count": 1,
            }

        def fake_builder(_root, *, scan_roots, excluded_target):
            assert scanner._normalize_relpath(
                tmp_path / "raw/class_map.json"
            ) == "raw/class_map.json"
            assert scanner._normalize_relpath(
                tmp_path / "raw/role.json"
            ) == "raw/role.json"
            assert scanner._normalize_relpath("elsewhere/input.json") == (
                "elsewhere/input.json"
            )
            assert scanner._structured_kind("raw/class_map.json") is None
            assert scanner._structured_kind("raw/role.json") is None
            assert scanner._structured_kind("raw/other.json") == "json"
            if builder_failure:
                raise RuntimeError("fixture builder failure")
            findings = list(scanner._walk_seed_fields({}, source="fixture.json"))
            unresolved = [row for row in findings if row["semantic"] == "ambiguous"]
            if unresolved:
                raise scanner.AmbiguousSeedSemanticsError(unresolved)
            semantic_sets = {
                row["semantic"]: sorted(set(row["seeds"])) for row in findings
            }
            payload = {
                "schema_version": "focused-window-wrapper-fixture-v1",
                "scan_roots": list(scan_roots),
                "excluded_target": excluded_target,
                "source_count": 1,
                "sources": [],
                "seed_field_match_count": len(findings),
                "seed_field_group_count": len(findings),
                "seed_field_matches": findings,
                "semantic_seed_sets": semantic_sets,
                "executed_gnn_training_seeds": list(recognized_used),
                "skipped_test_field_count": 0,
                "skipped_test_field_path_commitment_sha256": "0" * 64,
                "test_value_policy": "fixture",
            }
            payload["inventory_root_sha256"] = scanner._sha256_bytes(
                scanner._canonical_json(payload)
            )
            return payload

        monkeypatch.setattr(scanner, "_normalize_relpath", base_normalize_relpath)
        monkeypatch.setattr(scanner, "_structured_kind", base_structured_kind)
        monkeypatch.setattr(scanner, "_walk_seed_fields", base_walker)
        monkeypatch.setattr(
            scanner, "build_historical_gnn_seed_inventory", fake_builder
        )
        try:
            return diagnostic.build_candidate_window_gnn_seed_inventory(
                tmp_path, config
            )
        finally:
            assert scanner._walk_seed_fields is base_walker
            assert scanner._structured_kind is base_structured_kind
            assert scanner._normalize_relpath is base_normalize_relpath

    inventory, audit = exercise(42)
    assert inventory["semantic_seed_sets"] == {
        "unresolved_seed_semantics_for_window_audit": [42]
    }
    assert audit["all_unresolved_group_count"] == 1
    _, recognized_audit = exercise(2000, recognized_used=[2000])
    assert recognized_audit["all_unresolved_values_in_proof_window"] == [2000]
    assert recognized_audit[
        "independently_recognized_used_unresolved_values_in_proof_window"
    ] == [2000]
    assert recognized_audit["blocking_unresolved_candidate_values"] == []
    assert recognized_audit["exact_first_35_selection_provable"] is True
    with pytest.raises(RuntimeError, match="fixture builder failure"):
        exercise(42, builder_failure=True)
    with pytest.raises(
        diagnostic.ProtocolError,
        match="UNRESOLVED_SEED_SEMANTICS_IN_SELECTION_WINDOW",
    ):
        exercise(2010)


def test_six_conditions_change_only_the_frozen_dimension(tasks, config):
    by_block = {}
    for row in tasks["tasks"]:
        by_block.setdefault(row["block_id"], []).append(row)
    condition_by_id = {row["id"]: row for row in config["conditions"]}
    assert len(by_block) == 60
    for rows in by_block.values():
        assert {row["condition_id"] for row in rows} == set(diagnostic.CONDITION_IDS)
        base = rows[0]["base_hp"]
        for row in rows:
            condition = condition_by_id[row["condition_id"]]
            assert row["effective_hp"] == diagnostic._effective_hp(base, condition)
            changed = {
                key
                for key in ("lr", "dropout", "l2")
                if row["effective_hp"][key] != base[key]
            }
            expected = {
                "ln_lr_x0_5": {"lr"},
                "ln_lr_x0_25": {"lr"},
                "ln_dropout_x0_5": {"dropout"},
                "ln_weight_decay_x0_1": {"l2"},
            }.get(row["condition_id"], set())
            assert changed == expected


def test_task_count_colocation_and_williams_balance(tasks):
    validation = diagnostic.validate_tasks(tasks, diagnostic.load_config(CONFIG_PATH))
    assert validation["task_count"] == 360
    assert validation["block_count"] == 60
    assert validation["worker_task_counts"] == {"0": 90, "1": 90, "2": 90, "3": 90}
    assert validation["gpu_task_counts"] == {"0": 180, "1": 180}
    assert validation["no_ln_first_block_count"] == 10
    transitions = Counter()
    blocks = {}
    for row in tasks["tasks"]:
        blocks.setdefault(row["block_id"], []).append(row)
    for rows in blocks.values():
        ordered = [
            row["condition_id"] for row in sorted(rows, key=lambda item: item["execution_position"])
        ]
        transitions.update(zip(ordered, ordered[1:]))
        assert len({row["worker_id"] for row in rows}) == 1
        assert len({(row["gpu_index"], row["gpu_uuid"]) for row in rows}) == 1
    assert len(transitions) == 30
    assert set(transitions.values()) == {10}


def test_test_firewall_is_literal_and_has_no_forbidden_mask_access(tasks):
    audit = diagnostic.audit_test_firewall(tasks)
    assert audit["passed"] is True
    assert audit["formal_track_test_literal"] is False
    assert audit["test_mask_attribute_count_in_analyser"] == 0
    assert audit["runtime_proxy_call_count"] == 2
    assert audit["runtime_proxy_enabled"] is True
    assert all(row["track_test"] is False for row in tasks["tasks"])


def test_runtime_data_proxy_records_and_blocks_test_access():
    fields = {field: object() for field in diagnostic.ALLOWED_DATA_FIELDS}
    fields["test_mask"] = object()
    data = type("Data", (), fields)()
    ledger = diagnostic.new_data_access_ledger()
    proxy = diagnostic.ValidationOnlyDataProxy(data, ledger)
    assert proxy.x is data.x
    with pytest.raises(diagnostic.TestAccessViolation):
        getattr(proxy, "test_mask")
    observed = diagnostic.data_access_ledger(proxy)
    assert observed["allowed_attribute_counts"]["x"] == 1
    assert observed["forbidden_request_count"] == 1
    assert observed["test_accessed"] is True


def _synthetic_results(delta_by_condition):
    rows = []
    for config_index in range(12):
        fingerprint = f"fp-{config_index:02d}"
        for seed in diagnostic.DIAGNOSTIC_SEEDS:
            for condition_id in diagnostic.CONDITION_IDS:
                value = 0.75 + float(delta_by_condition.get(condition_id, 0.0))
                rows.append(
                    {
                        "task_id": f"{fingerprint}-{seed}-{condition_id}",
                        "joint_config_fingerprint": fingerprint,
                        "seed": seed,
                        "condition_id": condition_id,
                        "status": "COMPLETE",
                        "best_validation_accuracy": value,
                        "best_validation_prediction_counts": [250, 250, 0],
                        "validation_sample_count": 500,
                    }
                )
    return rows


def test_stable_recipe_gate_no_pass_and_fixed_tie_break(config):
    losing = _synthetic_results(
        {condition: -0.01 for condition in diagnostic.LN_CONDITION_IDS}
    )
    losing_stats = diagnostic.compute_statistics(losing, config)
    assert losing_stats["selected_condition_id"] is None
    tied = _synthetic_results({condition: 0.0 for condition in diagnostic.LN_CONDITION_IDS})
    tied_stats = diagnostic.compute_statistics(tied, config)
    assert tied_stats["eligible_condition_ids"] == list(diagnostic.LN_CONDITION_IDS)
    assert tied_stats["selected_condition_id"] == "ln_original"
    higher = _synthetic_results(
        {
            "ln_original": 0.01,
            "ln_lr_x0_5": 0.02,
            "ln_lr_x0_25": 0.0,
            "ln_dropout_x0_5": 0.0,
            "ln_weight_decay_x0_1": 0.0,
        }
    )
    assert diagnostic.compute_statistics(higher, config)["selected_condition_id"] == "ln_lr_x0_5"


def test_stable_recipe_gate_enforces_config_count_and_collapse_excess(config):
    config_gate = _synthetic_results({})
    for row in config_gate:
        if row["condition_id"] == "ln_original":
            index = int(row["joint_config_fingerprint"].split("-")[-1])
            row["best_validation_accuracy"] += -0.01 if index < 5 else 0.02
    summary = next(
        row
        for row in diagnostic.compute_statistics(config_gate, config)[
            "condition_summaries"
        ]
        if row["condition_id"] == "ln_original"
    )
    assert summary["mean_paired_delta"] > 0
    assert summary["config_mean_noninferior_count"] == 7
    assert summary["eligible"] is False

    collapse_gate = _synthetic_results({"ln_original": 0.03})
    baseline = next(
        row
        for row in collapse_gate
        if row["condition_id"] == "no_ln"
        and row["joint_config_fingerprint"] == "fp-00"
        and row["seed"] == diagnostic.DIAGNOSTIC_SEEDS[0]
    )
    baseline.update(
        best_validation_accuracy=0.316,
        best_validation_prediction_counts=[500, 0, 0],
    )
    ln_rows = [
        row
        for row in collapse_gate
        if row["condition_id"] == "ln_original"
        and row["joint_config_fingerprint"] == "fp-00"
    ][:3]
    for row in ln_rows:
        row.update(
            best_validation_accuracy=0.316,
            best_validation_prediction_counts=[500, 0, 0],
        )
    summary = next(
        row
        for row in diagnostic.compute_statistics(collapse_gate, config)[
            "condition_summaries"
        ]
        if row["condition_id"] == "ln_original"
    )
    assert summary["collapse_count"] == 3
    assert summary["no_ln_collapse_count"] == 1
    assert summary["collapse_excess_over_no_ln"] == 2
    assert summary["mean_paired_delta"] > 0
    assert summary["eligible"] is False


def test_collapse_definition_requires_both_exact_accuracy_and_one_class():
    base = {
        "best_validation_accuracy": 0.316,
        "best_validation_prediction_counts": [500, 0, 0],
        "validation_sample_count": 500,
    }
    assert diagnostic.is_collapse_result(base)
    mixed = dict(base, best_validation_prediction_counts=[499, 1, 0])
    assert not diagnostic.is_collapse_result(mixed)
    inexact = dict(base, best_validation_accuracy=0.3160000001)
    assert not diagnostic.is_collapse_result(inexact)


def test_complete_result_requires_history_best_epoch_and_prediction_counts(tasks):
    task = tasks["tasks"][0]
    result = {
        **diagnostic._result_identity(task),
        "status": "COMPLETE",
        "attempt_count": 1,
        "test_accessed": False,
        "nan_or_inf_detected": False,
        "nan_or_inf_stage": None,
        "data_access_ledger": {
            "allowed_request_count": 10,
            "forbidden_request_count": 0,
            "allowed_attribute_counts": {
                field: 0 for field in diagnostic.ALLOWED_DATA_FIELDS
            },
            "forbidden_attribute_counts": {},
            "test_accessed": False,
        },
        "history": [
            {"epoch_1based": 1, "training_loss": 1.2, "validation_accuracy": 0.70},
            {"epoch_1based": 2, "training_loss": 1.0, "validation_accuracy": 0.72},
        ],
        "epochs_ran": 2,
        "best_epoch": 2,
        "best_validation_accuracy": 0.72,
        "best_validation_prediction_counts": [200, 150, 150],
        "validation_sample_count": 500,
    }
    diagnostic._validate_complete_result(result, task)
    broken = copy.deepcopy(result)
    broken["best_validation_prediction_counts"] = [200, 150, 149]
    with pytest.raises(diagnostic.ProtocolError, match="PREDICTION_COUNT_TOTAL_MISMATCH"):
        diagnostic._validate_complete_result(broken, task)


def test_existing_result_root_stops_before_any_freeze(monkeypatch, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    reached = False

    def forbidden(_config):
        nonlocal reached
        reached = True
        raise AssertionError("selection must not run")

    monkeypatch.setattr(diagnostic, "resolve_result_root", lambda *_args, **_kwargs: existing)
    monkeypatch.setattr(diagnostic, "assert_host", lambda _config: {})
    monkeypatch.setattr(diagnostic, "freeze_selection", forbidden)
    with pytest.raises(diagnostic.ProtocolError, match="RESULT_ROOT_ALREADY_EXISTS"):
        diagnostic.dry_run(CONFIG_PATH)
    assert reached is False
    assert list(existing.iterdir()) == []


def test_formal_contract_is_one_attempt_no_retry_and_no_follow_on(config):
    assert config["training"]["planned_task_count"] == 12 * 6 * 5 == 360
    assert config["training"]["attempts_per_task"] == 1
    assert config["training"]["retry"] is False
    assert config["training"]["replacement"] is False
    assert config["training"]["adaptive_settings"] is False
    assert config["lifecycle"]["formal_launch_count"] == 1
    assert config["lifecycle"]["start_follow_on_phase"] is False
