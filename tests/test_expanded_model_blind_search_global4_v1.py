from __future__ import annotations

from collections import Counter
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "analyse/expanded_model_blind_search_global4_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/expanded_model_blind_search_global4_v1.json"
TARGET_PATHS = (
    "analyse/expanded_model_blind_search_global4_v1.py",
    "configs/expanded_model_blind_search_global4_v1.json",
    "scripts/run_expanded_model_blind_search_global4_v1.sh",
    "tests/test_expanded_model_blind_search_global4_v1.py",
)

GPU_0_UUID = "GPU-5496be78-3b71-4978-3960-b6ed926ebe49"
GPU_1_UUID = "GPU-197ae2b9-74cd-6afc-446d-7bc9e98176f9"

EXPECTED_P_ANCHOR = {
    "architecture_fingerprint": (
        "960efb3b97f00491c3956657cfd57958effb6096f65b1db2fa8aaae586b82207"
    ),
    "hp_fingerprint": (
        "911259943106a1edad300bc5300a3883ece8cbffba48b95e6c61b6e5eb4b679d"
    ),
    "joint_config_fingerprint": (
        "30e0bb108c1fdbe8db186152a09c3f2fc6e812be189db920fe4d0d2b0d3cb1a2"
    ),
    "config_id": "hr_p22_b0_s4_960efb3b97f0_hp_high_91125994",
    "operations": ["GCNConv", "GATConv", "GCNII", "GCNII", "SAGEConv"],
    "edges": [
        [0, 1],
        [1, 2],
        [0, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [3, 5],
        [0, 5],
        [5, 6],
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


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "expanded_model_blind_search_global4_v1", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def phase_d():
    return _load_module()


@pytest.fixture(scope="module")
def config(phase_d):
    return phase_d.load_config(CONFIG_PATH)


def _sha256_label(*parts: object) -> str:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class OrderedFakeDecoder:
    """Return a fixed ordered stream while retaining the production batch API."""

    def __init__(self, values: list[str], *, filler: str = "tail") -> None:
        self.values = list(values)
        self.filler = filler
        self.offset = 0
        self.calls: list[dict[str, int]] = []

    def __call__(self, generator: Any, batch_size: int) -> list[str]:
        seed = int(generator.initial_seed())
        self.calls.append({"seed": seed, "batch_size": int(batch_size)})
        output = self.values[self.offset : self.offset + batch_size]
        self.offset += len(output)
        if len(output) < batch_size:
            output.extend(
                f"{self.filler}-{self.offset + index}"
                for index in range(batch_size - len(output))
            )
        return output


def _fake_converter(graph: str) -> dict[str, Any]:
    if graph == "conversion-failure":
        raise ValueError("synthetic conversion rejection")
    return {
        "fixture_tag": graph,
        "operations": ["GCNConv"],
        "edges": [[0, 1], [1, 2]],
        "effective_layers": 1,
    }


def _fake_validator(config: dict[str, Any]) -> tuple[bool, str | None]:
    if config["fixture_tag"] in {"static-invalid", "reject"}:
        return False, "fixture_static_invalid"
    return True, None


def _fake_architecture_fingerprint(config: dict[str, Any]) -> str:
    return _sha256_label("architecture", config["fixture_tag"])


def _fake_hp_realizer(candidate: dict[str, Any]) -> dict[str, Any]:
    raw = candidate["raw_hp_vector"]
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


def _stream_candidates(result: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert isinstance(result, dict)
    assert isinstance(result.get("candidates"), list)
    assert isinstance(result.get("audit"), dict)
    return result["candidates"], result["audit"]


def test_unsigned_stream_seed_derivation_is_exact_and_not_sign_converted(phase_d):
    assert phase_d.derive_stream_seed("01", "architecture-slots") == 4095203679628602058
    assert phase_d.derive_stream_seed("01", "raw-hp-uniforms") == 8700971845320871746
    assert phase_d.derive_stream_seed("03", "raw-hp-uniforms") == 14999548378627945595
    assert phase_d.derive_stream_seed("04", "architecture-slots") == 14326530954504764964
    assert phase_d.derive_stream_seed("04", "raw-hp-uniforms") == 15023105018410458585
    assert phase_d.derive_stream_seed("12", "architecture-slots") == 9684994294416177191
    assert phase_d.derive_stream_seed("04", "architecture-slots") > 2**63
    assert phase_d.derive_stream_seed("04", "raw-hp-uniforms") > 2**63
    with pytest.raises((ValueError, phase_d.ProtocolFailure)):
        phase_d.derive_stream_seed("4", "architecture-slots")
    with pytest.raises((ValueError, phase_d.ProtocolFailure)):
        phase_d.derive_stream_seed("04", "unknown-stream")


def test_batched_candidate_stream_preserves_order_and_consumes_hp_only_on_acceptance(
    phase_d,
):
    accepted_a = _fake_architecture_fingerprint(_fake_converter("accepted-a"))
    accepted_b = _fake_architecture_fingerprint(_fake_converter("accepted-b"))
    historical = _fake_architecture_fingerprint(_fake_converter("historical"))
    already_global = _fake_architecture_fingerprint(_fake_converter("already-global"))
    decoder = OrderedFakeDecoder(
        [
            "conversion-failure",
            "static-invalid",
            "historical",
            "already-global",
            "accepted-a",
            "accepted-a",
            "accepted-b",
        ]
    )
    hp_inputs: list[tuple[float, ...]] = []

    def hp_fingerprinter(values: list[float]) -> str:
        hp_inputs.append(tuple(float(value) for value in values))
        return _sha256_label("hp", len(hp_inputs), *hp_inputs[-1])

    global_seen = {already_global}
    result = phase_d.consume_candidate_stream(
        decoder,
        architecture_seed=14326530954504764964,
        hp_seed=15023105018410458585,
        historical_exposure={historical},
        global_seen=global_seen,
        replicate_id="04",
        slot_count=2,
        decode_batch_size=256,
        max_rejections_per_slot=1000,
        converter=_fake_converter,
        validator=_fake_validator,
        architecture_fingerprinter=_fake_architecture_fingerprint,
        hp_fingerprinter=hp_fingerprinter,
        hp_realizer=_fake_hp_realizer,
    )
    candidates, audit = _stream_candidates(result)

    assert [row["architecture_fingerprint"] for row in candidates] == [
        accepted_a,
        accepted_b,
    ]
    assert [row["replicate_id"] for row in candidates] == ["04", "04"]
    assert [row["slot_id"] for row in candidates] == ["01", "02"]
    assert [row["rejection_count_before_acceptance"] for row in candidates] == [4, 1]
    assert len(hp_inputs) == 2
    expected_rng = np.random.default_rng(15023105018410458585)
    expected_hp = [
        tuple(expected_rng.uniform(0.80, 0.95, size=4).astype(np.float64).tolist())
        for _ in range(2)
    ]
    assert hp_inputs == expected_hp
    assert [tuple(row["raw_hp_vector"]) for row in candidates] == expected_hp
    assert all(row["decoded_hp"] == _fake_hp_realizer(row) for row in candidates)
    assert global_seen == {already_global, accepted_a, accepted_b}
    assert decoder.calls == [
        {"seed": 14326530954504764964, "batch_size": 256}
    ]
    assert audit["generated_draw_count"] == 256
    assert audit["consumed_draw_count"] == 7
    assert audit["unused_generated_draw_count"] == 249
    assert sum(audit["rejection_reason_counts"].values()) == 5
    reason_text = " ".join(audit["rejection_reason_counts"]).casefold()
    for marker in ("conversion", "static", "historical", "duplicate"):
        assert marker in reason_text


def test_exactly_1000_rejections_are_allowed_but_the_1001st_is_blocked(phase_d):
    hp_calls: list[tuple[float, ...]] = []

    def hp_fingerprinter(values: list[float]) -> str:
        hp_calls.append(tuple(float(value) for value in values))
        return _sha256_label("boundary-hp", len(hp_calls))

    allowed = phase_d.consume_candidate_stream(
        OrderedFakeDecoder(["reject"] * 1000 + ["accepted"]),
        architecture_seed=7,
        hp_seed=11,
        historical_exposure=set(),
        global_seen=set(),
        replicate_id="01",
        slot_count=1,
        decode_batch_size=256,
        max_rejections_per_slot=1000,
        converter=_fake_converter,
        validator=_fake_validator,
        architecture_fingerprinter=_fake_architecture_fingerprint,
        hp_fingerprinter=hp_fingerprinter,
        hp_realizer=_fake_hp_realizer,
    )
    candidates, audit = _stream_candidates(allowed)
    assert candidates[0]["rejection_count_before_acceptance"] == 1000
    assert audit["generated_draw_count"] == 1024
    assert audit["consumed_draw_count"] == 1001
    assert audit["unused_generated_draw_count"] == 23
    assert len(hp_calls) == 1

    hp_calls.clear()
    blocked_seen: set[str] = set()
    with pytest.raises(phase_d.ProtocolFailure) as error:
        phase_d.consume_candidate_stream(
            OrderedFakeDecoder(["reject"] * 1001 + ["accepted"]),
            architecture_seed=7,
            hp_seed=11,
            historical_exposure=set(),
            global_seen=blocked_seen,
            replicate_id="01",
            slot_count=1,
            decode_batch_size=256,
            max_rejections_per_slot=1000,
            converter=_fake_converter,
            validator=_fake_validator,
            architecture_fingerprinter=_fake_architecture_fingerprint,
            hp_fingerprinter=hp_fingerprinter,
            hp_realizer=_fake_hp_realizer,
        )
    assert "REJECTION" in str(error.value).upper()
    assert hp_calls == []
    assert blocked_seen == set()


def test_exact_exposure_basis_uses_only_the_two_authorized_identity_files(
    phase_d, config
):
    basis = phase_d.load_exposure_basis(config)
    assert basis["union_count"] == 14063
    assert basis["overlap_count"] == 0
    fingerprints = set(basis["architecture_fingerprints"])
    assert len(fingerprints) == 14063
    sources = {Path(row["path"]).name: row for row in basis["sources"]}
    assert set(sources) == {"exposure_registry.json", "global_reservoir.json"}
    assert sources["exposure_registry.json"] == {
        **sources["exposure_registry.json"],
        "size_bytes": 1498071,
        "sha256": "98c5f54345a41b90b1b89f0bf1a6d883974c36eadcb4291d93b529a068bb3d66",
        "architecture_count": 13871,
    }
    assert sources["global_reservoir.json"] == {
        **sources["global_reservoir.json"],
        "size_bytes": 164195,
        "sha256": "ff4bb7f06dc2424b6b686b2102d43f590c9cdaaf363f3ab9f020c3f50a63f619",
        "architecture_count": 192,
    }
    assert isinstance(basis["commitment_sha256"], str)
    assert len(basis["commitment_sha256"]) == 64


def test_exact_p_anchor_projection_and_all_three_fingerprints(phase_d, config):
    from analyse.original_head_deterministic_batch_pilot import hp_fingerprint
    from architecture_diagnostics import architecture_fingerprint

    anchor = phase_d.project_p_anchor(config)
    assert anchor == EXPECTED_P_ANCHOR
    assert architecture_fingerprint(anchor) == anchor["architecture_fingerprint"]
    assert hp_fingerprint(anchor["raw_hp_vector"]) == anchor["hp_fingerprint"]
    assert (
        phase_d.joint_config_fingerprint(
            anchor["architecture_fingerprint"], anchor["hp_fingerprint"]
        )
        == anchor["joint_config_fingerprint"]
    )


def test_raw_float64_hp_stream_matches_legacy_fingerprint_and_decode(phase_d):
    from analyse.original_head_deterministic_batch_pilot import hp_fingerprint
    from analyse.structure_native_surrogate_pilot import realize_global4_hp

    hp_seed = 14999548378627945595
    result = phase_d.consume_candidate_stream(
        OrderedFakeDecoder(["accepted"]),
        architecture_seed=3,
        hp_seed=hp_seed,
        historical_exposure=set(),
        global_seen=set(),
        replicate_id="03",
        slot_count=1,
        converter=_fake_converter,
        validator=_fake_validator,
        architecture_fingerprinter=_fake_architecture_fingerprint,
    )
    candidate = _stream_candidates(result)[0][0]
    expected_raw = (
        np.random.default_rng(hp_seed)
        .uniform(0.80, 0.95, size=4)
        .astype(np.float64)
        .tolist()
    )
    expected_candidate = {
        **candidate,
        "raw_hp_vector": expected_raw,
    }
    assert candidate["raw_hp_vector"] == expected_raw
    assert candidate["hp_fingerprint"] == hp_fingerprint(expected_raw)
    assert candidate["decoded_hp"] == realize_global4_hp(expected_candidate)


def _make_candidates(phase_d) -> list[dict[str, Any]]:
    from analyse.original_head_deterministic_batch_pilot import hp_fingerprint
    from analyse.structure_native_surrogate_pilot import realize_global4_hp
    from architecture_diagnostics import architecture_fingerprint

    operation_types = (
        "GCNConv",
        "GATConv",
        "SAGEConv",
        "GINConv",
        "GCNII",
        "Identity",
    )
    operation_vectors = itertools.islice(itertools.product(operation_types, repeat=4), 192)
    rows: list[dict[str, Any]] = []
    for index, operations_tuple in enumerate(operation_vectors):
        replicate_index, slot_index = divmod(index, 16)
        operations = list(operations_tuple)
        architecture = {
            "operations": operations,
            "edges": [[node, node + 1] for node in range(5)],
            "effective_layers": sum(operation != "Identity" for operation in operations),
        }
        raw_hp = (
            np.random.default_rng(10000 + index)
            .uniform(0.80, 0.95, size=4)
            .astype(np.float64)
            .tolist()
        )
        architecture_fp = architecture_fingerprint(architecture)
        hp_fp = hp_fingerprint(raw_hp)
        row = {
            "replicate_id": f"{replicate_index + 1:02d}",
            "slot_id": f"{slot_index + 1:02d}",
            **architecture,
            "architecture_fingerprint": architecture_fp,
            "raw_hp_vector": raw_hp,
            "hp_fingerprint": hp_fp,
            "hp_regime": "hp_high",
            "rejection_count_before_acceptance": 0,
        }
        row["decoded_hp"] = realize_global4_hp(row)
        row["joint_config_fingerprint"] = phase_d.joint_config_fingerprint(
            architecture_fp, hp_fp
        )
        row["config_id"] = (
            f"phase_d_r{row['replicate_id']}_s{row['slot_id']}_"
            f"{architecture_fp[:12]}_hp_high_{hp_fp[:8]}"
        )
        rows.append(row)
    assert len(rows) == 192
    assert len({row["architecture_fingerprint"] for row in rows}) == 192
    return rows


@pytest.fixture(scope="module")
def task_freeze(phase_d, config):
    candidates = _make_candidates(phase_d)
    anchor = phase_d.project_p_anchor(config)
    return phase_d.build_task_freeze(candidates, anchor, config)


def _without_pair_differences(task: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(task)
    for key in ("task_id", "arm", "component_variant"):
        result.pop(key)
    return result


def test_task_freeze_cardinality_pair_equality_and_fixed_worker_gpu_counts(
    phase_d, config, task_freeze
):
    validation = phase_d.validate_task_freeze(task_freeze, config)
    assert validation is None or validation.get("passed") is True
    tasks = task_freeze["tasks"]
    assert len(tasks) == 1155
    assert len({row["task_id"] for row in tasks}) == 1155
    assert Counter(row["arm"] for row in tasks) == Counter(
        {"Control": 576, "Intervention": 576, "P_anchor": 3}
    )

    search = [row for row in tasks if row["arm"] in {"Control", "Intervention"}]
    anchors = [row for row in tasks if row["arm"] == "P_anchor"]
    assert len(search) == 1152
    assert len(anchors) == 3
    expected_keys = [
        (f"{replicate:02d}", f"{slot:02d}", seed)
        for replicate in range(1, 13)
        for slot in range(1, 17)
        for seed in (0, 1, 2)
    ]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for task in search:
        key = (task["replicate_id"], task["slot_id"], int(task["seed"]))
        grouped.setdefault(key, []).append(task)
    assert list(grouped) == expected_keys
    assert len(grouped) == 576
    for pair_index, key in enumerate(expected_keys):
        members = grouped[key]
        assert {row["arm"] for row in members} == {"Control", "Intervention"}
        control = next(row for row in members if row["arm"] == "Control")
        intervention = next(row for row in members if row["arm"] == "Intervention")
        assert control["component_variant"] == "baseline"
        assert intervention["component_variant"] == "layer_norm"
        assert control["worker_id"] == intervention["worker_id"] == pair_index % 4
        assert _without_pair_differences(control) == _without_pair_differences(
            intervention
        )
        assert control["track_test"] is False
        assert control["max_epochs"] == 300
        assert control["patience"] == 80

    assert Counter((row["worker_id"], row["arm"]) for row in search) == Counter(
        {
            **{(worker, "Control"): 144 for worker in range(4)},
            **{(worker, "Intervention"): 144 for worker in range(4)},
        }
    )
    assert Counter(row["worker_id"] for row in tasks) == Counter(
        {0: 289, 1: 289, 2: 289, 3: 288}
    )
    assert Counter(row["gpu_index"] for row in tasks) == Counter({0: 578, 1: 577})
    assert Counter(
        (row["gpu_index"], row["arm"]) for row in search
    ) == Counter(
        {
            (0, "Control"): 288,
            (1, "Control"): 288,
            (0, "Intervention"): 288,
            (1, "Intervention"): 288,
        }
    )
    assert {(row["worker_id"], row["gpu_index"], row["gpu_uuid"]) for row in tasks} == {
        (0, 0, GPU_0_UUID),
        (1, 1, GPU_1_UUID),
        (2, 0, GPU_0_UUID),
        (3, 1, GPU_1_UUID),
    }
    assert [(row["seed"], row["worker_id"]) for row in anchors] == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]
    assert all(row["component_variant"] == "baseline" for row in anchors)
    assert all(row["track_test"] is False for row in anchors)


def test_task_validation_rejects_a_single_pair_hp_difference(phase_d, config, task_freeze):
    mutated = copy.deepcopy(task_freeze)
    intervention = next(
        row for row in mutated["tasks"] if row["arm"] == "Intervention"
    )
    intervention["decoded_hp"]["dropout"] += 0.01
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.validate_task_freeze(mutated, config)


def test_task_validation_requires_exact_p_anchor_payload_and_training_semantics(
    phase_d, config, task_freeze
):
    mutated_anchor = copy.deepcopy(task_freeze)
    anchor = next(row for row in mutated_anchor["tasks"] if row["arm"] == "P_anchor")
    anchor["raw_hp_vector"][0] = np.nextafter(
        anchor["raw_hp_vector"][0], float("inf")
    )
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.validate_task_freeze(mutated_anchor, config)

    mutated_semantics = copy.deepcopy(task_freeze)
    mutated_semantics["tasks"][0]["training_semantics"]["split"] = "random"
    mutated_semantics["tasks"][1]["training_semantics"]["split"] = "random"
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.validate_task_freeze(mutated_semantics, config)


def test_retry_budget_allows_57_distinct_tasks_and_blocks_the_58th(phase_d):
    budget = phase_d.RetryBudget(limit=57)
    for index in range(57):
        task = {"task_id": f"retry-{index:02d}", "worker_id": index % 4}
        calls: list[tuple[dict[str, Any], int]] = []

        def evaluator(current: dict[str, Any], attempt: int) -> str:
            calls.append((copy.deepcopy(current), attempt))
            if attempt == 1:
                raise OSError("synthetic transient")
            return current["task_id"]

        result = phase_d.execute_with_retries(task, evaluator, budget)
        assert result == {"value": task["task_id"], "attempts": 2}
        assert [attempt for _, attempt in calls] == [1, 2]
        assert calls[0][0] == calls[1][0] == task
    assert budget.retry_task_count == 57

    calls_58: list[int] = []

    def retry_58(_task: dict[str, Any], attempt: int) -> None:
        calls_58.append(attempt)
        raise TimeoutError("synthetic transient")

    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.execute_with_retries({"task_id": "retry-58"}, retry_58, budget)
    assert calls_58 == [1]
    assert budget.retry_task_count == 57


def test_retry_max_attempts_and_nonretryable_failures(phase_d):
    from evaluation_errors import InfrastructureEvaluationError

    budget = phase_d.RetryBudget(limit=57)
    attempts: list[int] = []

    def always_transient(_task: dict[str, Any], attempt: int) -> None:
        attempts.append(attempt)
        raise InfrastructureEvaluationError("synthetic infrastructure failure")

    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.execute_with_retries(
            {"task_id": "three-attempts"}, always_transient, budget
        )
    assert attempts == [1, 2, 3]
    assert budget.retry_task_count == 1

    nonretry_calls: list[int] = []

    def scientific_failure(_task: dict[str, Any], attempt: int) -> None:
        nonretry_calls.append(attempt)
        raise ValueError("synthetic scientific failure")

    with pytest.raises(ValueError):
        phase_d.execute_with_retries(
            {"task_id": "nonretryable"}, scientific_failure, budget
        )
    assert nonretry_calls == [1]
    assert budget.retry_task_count == 1


def test_validation_only_proxy_negative_control_and_literal_false_ast(phase_d):
    from analyse.fixed_layernorm_enhancement_v2 import (
        DATA_ALLOWLIST,
        TestAccessViolation,
        ValidationOnlyDataProxy,
        _proxy_internal_ledger,
        probe_track_test,
    )

    class Data:
        x = 1
        edge_index = 2
        train_mask = 3
        val_mask = 4
        y = 5
        test_mask = 6

    proxy = ValidationOnlyDataProxy(Data())
    for name in sorted(DATA_ALLOWLIST):
        getattr(proxy, name)
    with pytest.raises(TestAccessViolation):
        proxy.test_mask
    with pytest.raises(TestAccessViolation):
        probe_track_test(proxy, track_test=True)
    ledger = _proxy_internal_ledger(proxy)
    assert ledger["allowed_request_count"] == 5
    assert ledger["forbidden_request_count"] == 2
    assert ledger["test_accessed"] is True

    evidence = phase_d.assert_evaluator_ast_contract()
    assert evidence["wrapper_literal_track_test_false_calls"] == 1
    assert evidence["root_test_mask_access_count"] > 0
    assert evidence["root_test_mask_all_guarded_by_track_test"] is True


def test_layernorm_source_contract_is_op_then_norm_then_activation_then_dropout(
    phase_d,
):
    evidence = phase_d.assert_layernorm_source_contract()
    assert evidence["passed"] is True
    assert evidence["identity_layernorm_count"] == 0
    assert evidence["message_passing_branch_count"] == 2
    assert evidence["ordered_layernorm_branch_count"] == 2
    assert evidence["order"] == ["operation", "layer_norm", "activation", "dropout"]


def test_layernorm_source_contract_rejects_a_misplaced_norm_input(
    phase_d, tmp_path, monkeypatch
):
    source = (REPO_ROOT / "eval_utils.py").read_text(encoding="utf-8")
    original = "layer_out = self.layer_norms[str(i)](layer_out)"
    assert source.count(original) == 2
    mutated = source.replace(
        original, "layer_out = self.layer_norms[str(i)](agg)", 1
    )
    (tmp_path / "eval_utils.py").write_text(mutated, encoding="utf-8")
    monkeypatch.setattr(phase_d, "REPO_ROOT", tmp_path)
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.assert_layernorm_source_contract()


def test_exact_sign_flip_enumerates_all_4096_without_plus_one(phase_d):
    deltas = [0.01] * 12
    result = phase_d.exact_sign_flip(deltas)
    assert result["observed_mean"] == pytest.approx(0.01)
    assert result["exceedance_count"] == 1
    assert result["permutation_count"] == 4096
    assert result["one_sided_p_value"] == pytest.approx(1 / 4096)


def test_bootstrap_is_exactly_100000_whole_replicate_resamples_and_deterministic(
    phase_d,
):
    deltas = np.asarray(
        [-0.004, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.007, 0.009, 0.011, 0.014],
        dtype=np.float64,
    )
    seed = 17143013755787270563
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 12, size=(100000, 12))
    expected = float(
        np.quantile(deltas[indices].mean(axis=1), 0.05, method="linear")
    )
    first = phase_d.bootstrap_lower_bound(
        deltas.tolist(), seed=seed, resamples=100000
    )
    second = phase_d.bootstrap_lower_bound(
        deltas.tolist(), seed=seed, resamples=100000
    )
    assert first == expected
    assert second == expected


def _synthetic_inference_plan() -> dict[str, Any]:
    return {
        "bootstrap_seed": 17143013755787270563,
        "bootstrap_replicates": 100000,
        "sign_flip_permutations": 4096,
    }


def _synthetic_formal_results(
    tasks: list[dict[str, Any]], *, intervention_delta: float
) -> list[dict[str, Any]]:
    seed_offsets = {0: -0.001, 1: 0.0, 2: 0.001}
    results: list[dict[str, Any]] = []
    for task in tasks:
        seed = int(task["seed"])
        if task["arm"] == "P_anchor":
            validation_accuracy = 0.65 + seed_offsets[seed]
        else:
            slot = int(task["slot_id"])
            control_mean = 0.70 if slot in {15, 16} else 0.60 + slot * 0.001
            validation_accuracy = control_mean + seed_offsets[seed]
            if task["arm"] == "Intervention":
                validation_accuracy += intervention_delta
        results.append(
            {
                "task_id": task["task_id"],
                "arm": task["arm"],
                "replicate_id": task.get("replicate_id"),
                "slot_id": task.get("slot_id"),
                "architecture_fingerprint": task["architecture_fingerprint"],
                "hp_fingerprint": task["hp_fingerprint"],
                "joint_config_fingerprint": task["joint_config_fingerprint"],
                "seed": seed,
                "component_variant": task["component_variant"],
                "worker_id": task["worker_id"],
                "gpu_index": task["gpu_index"],
                "gpu_uuid": task["gpu_uuid"],
                "validation_accuracy": validation_accuracy,
                "best_epoch": 17,
                "epochs_ran": 31,
                "attempt_count": 1,
                "test_accessed": False,
            }
        )
    assert len(results) == 1155
    return results


@pytest.fixture(scope="module")
def confirmed_statistics(phase_d, task_freeze):
    return phase_d.compute_statistics(
        _synthetic_formal_results(
            task_freeze["tasks"], intervention_delta=0.004
        ),
        task_freeze["tasks"],
        _synthetic_inference_plan(),
    )


def test_compute_statistics_uses_exact_seed_sets_and_fingerprint_tie_break(
    task_freeze, confirmed_statistics
):
    statistics = confirmed_statistics
    assert statistics["complete_replicate_count"] == 12
    assert statistics["complete_config_count"] == 384
    assert statistics["fresh_p_anchor_mean_validation"] == pytest.approx(0.65)
    assert statistics["mean_delta"] == pytest.approx(0.004)
    assert statistics["wins"] == 12
    assert statistics["ties"] == 0
    assert statistics["classification"] == "EXPANDED_SEARCH_CONFIRMED_ON_CORA"
    assert all(statistics["classification_gates"].values())
    assert all(
        row["seeds"] == [0, 1, 2]
        and len(row["validation_accuracies"]) == 3
        and row["complete"] is True
        for row in statistics["configuration_statistics"]
    )
    replicate_one_slot_one_control = next(
        row
        for row in statistics["configuration_statistics"]
        if row["replicate_id"] == "01"
        and row["slot_id"] == "01"
        and row["arm"] == "Control"
    )
    assert replicate_one_slot_one_control["config_mean_validation"] == pytest.approx(
        0.601
    )
    assert replicate_one_slot_one_control["config_sample_sd"] == pytest.approx(0.001)

    tied_fingerprints = sorted(
        {
            task["joint_config_fingerprint"]
            for task in task_freeze["tasks"]
            if task["arm"] == "Control"
            and task["replicate_id"] == "01"
            and task["slot_id"] in {"15", "16"}
        }
    )
    assert len(tied_fingerprints) == 2
    replicate_one_bests = {
        row["arm"]: row
        for row in statistics["replicate_arm_final_bests"]
        if row["replicate_id"] == "01"
    }
    assert set(replicate_one_bests) == {"Control", "Intervention"}
    assert {
        row["joint_config_fingerprint"] for row in replicate_one_bests.values()
    } == {tied_fingerprints[0]}


def test_compute_statistics_gate_miss_is_complete_not_confirmed(
    phase_d, task_freeze
):
    statistics = phase_d.compute_statistics(
        _synthetic_formal_results(
            task_freeze["tasks"], intervention_delta=0.002
        ),
        task_freeze["tasks"],
        _synthetic_inference_plan(),
    )
    assert statistics["complete_replicate_count"] == 12
    assert statistics["mean_delta"] == pytest.approx(0.002)
    assert statistics["classification"] == "EXPANDED_SEARCH_NOT_CONFIRMED_ON_CORA"
    assert statistics["classification_gates"] == {
        "mean_delta_at_least_0_003": False,
        "one_sided_p_value_below_0_05": True,
        "bootstrap_lower_bound_above_zero": True,
        "wins_at_least_8": True,
    }


def test_compute_statistics_rejects_missing_duplicate_and_nonexact_seed_sets(
    phase_d, task_freeze
):
    tasks = task_freeze["tasks"]
    results = _synthetic_formal_results(tasks, intervention_delta=0.004)
    plan = _synthetic_inference_plan()

    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.compute_statistics(results[:-1], tasks, plan)

    duplicate_results = copy.deepcopy(results)
    duplicate_results[-1]["task_id"] = duplicate_results[0]["task_id"]
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.compute_statistics(duplicate_results, tasks, plan)

    duplicate_seed_tasks = copy.deepcopy(tasks)
    target = next(
        row
        for row in duplicate_seed_tasks
        if row["arm"] == "Control"
        and row["replicate_id"] == "01"
        and row["slot_id"] == "01"
        and row["seed"] == 1
    )
    target["seed"] = 0
    duplicate_seed_results = copy.deepcopy(results)
    result_target = next(
        row for row in duplicate_seed_results if row["task_id"] == target["task_id"]
    )
    result_target["seed"] = 0
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.compute_statistics(duplicate_seed_results, duplicate_seed_tasks, plan)


def test_freeze_challengers_applies_threshold_sort_cap_and_phase_e_flag(
    phase_d, confirmed_statistics
):
    statistics = copy.deepcopy(confirmed_statistics)
    statistics["fresh_p_anchor_mean_validation"] = 0.0
    intervention = [
        row
        for row in statistics["configuration_statistics"]
        if row["arm"] == "Intervention"
    ]
    assert len(intervention) == 192
    for index, row in enumerate(intervention):
        row["config_mean_validation"] = 0.0009
        row["config_sample_sd"] = 0.1 + index * 1e-6
    ranked = sorted(
        intervention[:4], key=lambda row: row["joint_config_fingerprint"]
    )
    ranked[0]["config_mean_validation"] = 0.010
    ranked[0]["config_sample_sd"] = 0.010
    ranked[1]["config_mean_validation"] = 0.010
    ranked[1]["config_sample_sd"] = 0.010
    ranked[2]["config_mean_validation"] = 0.010
    ranked[2]["config_sample_sd"] = 0.030
    ranked[3]["config_mean_validation"] = 0.001
    ranked[3]["config_sample_sd"] = 0.000

    frozen = phase_d.freeze_challengers(statistics)
    assert frozen["threshold_eligible_config_count_pre_cap"] == 4
    assert frozen["eligible_challenger_count"] == 2
    assert [
        row["joint_config_fingerprint"] for row in frozen["challengers"]
    ] == [
        ranked[0]["joint_config_fingerprint"],
        ranked[1]["joint_config_fingerprint"],
    ]
    assert frozen["phase_e_eligible"] is True
    assert frozen["phase_e_started"] is False

    for row in intervention:
        row["config_mean_validation"] = 0.0009
    none_frozen = phase_d.freeze_challengers(statistics)
    assert none_frozen["threshold_eligible_config_count_pre_cap"] == 0
    assert none_frozen["eligible_challenger_count"] == 0
    assert none_frozen["challengers"] == []
    assert none_frozen["phase_e_eligible"] is False
    assert none_frozen["phase_e_started"] is False


def test_freeze_challengers_rejects_an_incomplete_intervention_config(
    phase_d, confirmed_statistics
):
    statistics = copy.deepcopy(confirmed_statistics)
    intervention = next(
        row
        for row in statistics["configuration_statistics"]
        if row["arm"] == "Intervention"
    )
    intervention["complete"] = False
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.freeze_challengers(statistics)


def test_guarded_evaluation_forwards_frozen_kwargs_and_fails_closed(
    phase_d, task_freeze
):
    from analyse.fixed_layernorm_enhancement_v2 import (
        TestAccessViolation,
        ValidationOnlyDataProxy,
    )

    class Data:
        x = object()
        edge_index = object()
        train_mask = object()
        val_mask = object()
        y = object()
        test_mask = object()

    proxy = ValidationOnlyDataProxy(Data())
    device = object()
    captured: list[dict[str, Any]] = []
    sentinel = object()

    def evaluator(**kwargs: Any) -> object:
        captured.append(kwargs)
        return sentinel

    pair = [
        row
        for row in task_freeze["tasks"]
        if row.get("pair_index") == 0
    ]
    assert [row["arm"] for row in pair] == ["Control", "Intervention"]
    for task in pair:
        formal_task = {**task, "in_ch": 1433, "out_ch": 7}
        assert (
            phase_d.run_guarded_evaluation(
                formal_task, proxy, device, evaluator=evaluator
            )
            is sentinel
        )
        forwarded = captured[-1]
        hp = task["decoded_hp"]
        assert forwarded == {
            "config": task["architecture"],
            "data": proxy,
            "in_ch": 1433,
            "out_ch": 7,
            "lr": float(hp["lr"]),
            "dropout": float(hp["dropout"]),
            "hidden_dim": int(hp["hidden_dim"]),
            "weight_decay": float(hp["l2"]),
            "gat_heads": int(hp.get("gat_heads", 1)),
            "sage_aggr": str(hp.get("sage_aggr", "mean")),
            "gin_eps": float(hp.get("gin_eps", 0.0)),
            "gat_heads_by_layer": hp.get("gat_heads_by_layer"),
            "sage_aggr_by_layer": hp.get("sage_aggr_by_layer"),
            "gin_eps_by_layer": hp.get("gin_eps_by_layer"),
            "device": device,
            "max_epochs": 300,
            "patience": 80,
            "seed": int(task["seed"]),
            "track_test": False,
            "return_metadata": True,
            "component_variant": task["component_variant"],
        }

    invalid_task = {**pair[0], "in_ch": 1433, "out_ch": 7, "track_test": True}
    call_count = len(captured)
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.run_guarded_evaluation(
            invalid_task, proxy, device, evaluator=evaluator
        )
    assert len(captured) == call_count

    def forbidden_evaluator(**kwargs: Any) -> object:
        return kwargs["data"].test_mask

    formal_task = {**pair[0], "in_ch": 1433, "out_ch": 7}
    with pytest.raises(TestAccessViolation):
        phase_d.run_guarded_evaluation(
            formal_task, proxy, device, evaluator=forbidden_evaluator
        )


def test_formal_start_is_published_from_same_parent_and_never_overwritten(
    phase_d, tmp_path, monkeypatch
):
    formal = tmp_path / "formal"
    original_mkdtemp = phase_d.tempfile.mkdtemp
    observed: dict[str, Any] = {}

    def same_parent_mkdtemp(*, prefix: str, dir: str | os.PathLike[str]) -> str:
        observed["prefix"] = prefix
        observed["dir"] = Path(dir)
        return original_mkdtemp(prefix=prefix, dir=dir)

    monkeypatch.setattr(phase_d.tempfile, "mkdtemp", same_parent_mkdtemp)
    phase_d._create_formal_with_atomic_start(
        formal, {"state": "FORMAL_STARTED", "run_id": "synthetic-run"}
    )
    assert observed["dir"] == formal.parent
    assert json.loads((formal / "start.json").read_text(encoding="utf-8")) == {
        "state": "FORMAL_STARTED",
        "run_id": "synthetic-run",
    }
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d._create_formal_with_atomic_start(
            formal, {"state": "FORMAL_STARTED", "run_id": "replacement"}
        )


def test_formal_sealing_is_one_way_and_runner_contract_is_exact(tmp_path):
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    formal_run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "formal_run"
    )
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(formal_run):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    seal_calls = [
        node
        for node in ast.walk(formal_run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"finalize_formal_archive", "_write_completion"}
    ]
    assert len(seal_calls) == 2
    for call in seal_calls:
        ancestor = parents.get(call)
        while ancestor is not None and ancestor is not formal_run:
            assert not isinstance(ancestor, ast.Try)
            ancestor = parents.get(ancestor)

    runner_path = REPO_ROOT / "scripts/run_expanded_model_blind_search_global4_v1.sh"
    runner = runner_path.read_text(encoding="utf-8")
    assert runner.count("conda run -n nas python -B") == 3
    assert "unset CUDA_VISIBLE_DEVICES" in runner
    assert "--runner-enter" in runner
    assert "--formal-run" in runner
    assert "--runner-exit" in runner
    assert "tmux" not in runner
    assert "monitor" not in runner.lower()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_conda = fake_bin / "conda"
    fake_conda.write_text(
        """#!/usr/bin/env bash
phase="${!#}"
printf '%s\\n' "${phase}" >> "${PHASE_D_RUNNER_CALL_LOG}"
case "${phase}" in
  --runner-enter) exit "${PHASE_D_RUNNER_ENTER_RC}" ;;
  --formal-run) exit "${PHASE_D_RUNNER_FORMAL_RC}" ;;
  --runner-exit) exit "${PHASE_D_RUNNER_EXIT_RC}" ;;
  *) exit 99 ;;
esac
""",
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)

    def execute_runner(
        name: str, enter_rc: int, formal_rc: int, exit_rc: int
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        call_log = tmp_path / f"{name}.log"
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
                "PHASE_D_RUNNER_CALL_LOG": str(call_log),
                "PHASE_D_RUNNER_ENTER_RC": str(enter_rc),
                "PHASE_D_RUNNER_FORMAL_RC": str(formal_rc),
                "PHASE_D_RUNNER_EXIT_RC": str(exit_rc),
            }
        )
        completed = subprocess.run(
            ["bash", str(runner_path)],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        return completed, call_log.read_text(encoding="utf-8").splitlines()

    enter_failure, calls = execute_runner("enter-failure", 17, 0, 0)
    assert enter_failure.returncode == 17, enter_failure.stderr
    assert calls == ["--runner-enter"]

    formal_failure, calls = execute_runner("formal-failure", 0, 19, 23)
    assert formal_failure.returncode == 19, formal_failure.stderr
    assert calls == ["--runner-enter", "--formal-run", "--runner-exit"]

    exit_failure, calls = execute_runner("exit-failure", 0, 0, 23)
    assert exit_failure.returncode == 23, exit_failure.stderr
    assert calls == ["--runner-enter", "--formal-run", "--runner-exit"]


def _prelaunch_assets() -> dict[str, bytes]:
    protocol = {
        "experiment": "expanded-model-blind-search-global4-v1",
        "bootstrap_seed_in_protocol": False,
    }
    protocol_bytes = _json_bytes(protocol)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(protocol_bytes).digest()[:8], "big", signed=False
    )
    assets = {
        "frozen_protocol.json": protocol_bytes,
        "inference_plan.json": _json_bytes(
            {
                "frozen_protocol_sha256": protocol_sha,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_replicates": 100000,
                "sign_flip_permutations": 4096,
            }
        ),
        "input_hashes.json": _json_bytes({"synthetic": True}),
        "frozen_exposure.json": _json_bytes({"union_count": 14063}),
        "frozen_candidates.json": _json_bytes({"candidate_count": 192}),
        "frozen_tasks.json": _json_bytes({"planned_task_count": 1155}),
        "assignments.json": _json_bytes({"worker_count": 4}),
        "test_firewall.json": _json_bytes(
            {"track_test": False, "test_accessed": False}
        ),
        "pre_execution_estimate.json": _json_bytes(
            {"best_minutes": 165, "expected_minutes": 210, "worst_reference_minutes": 360}
        ),
        "environment.json": _json_bytes({"synthetic": True}),
        "state.json": _json_bytes(
            {"state": "PRELAUNCH_COMPLETE", "gnn_evaluations": 0, "test_accessed": False}
        ),
    }
    for relative in TARGET_PATHS:
        assets[f"source_snapshot/{relative}"] = (REPO_ROOT / relative).read_bytes()
    return assets


def _coherent_prelaunch_assets(task_freeze: dict[str, Any]) -> dict[str, bytes]:
    assets = _prelaunch_assets()
    assets["frozen_tasks.json"] = _json_bytes(
        {key: value for key, value in task_freeze.items() if key != "assignments"}
    )
    assets["assignments.json"] = _json_bytes(
        {
            "schema_version": "phase-d-assignment-freeze-v1",
            "task_count": 1155,
            "assignments": task_freeze["assignments"],
        }
    )
    inference_plan = json.loads(assets["inference_plan.json"])
    inference_plan.update(
        {
            "schema_version": "phase-d-inference-plan-v1",
            "sign_flip_exhaustive": True,
            "sign_flip_plus_one_correction": False,
        }
    )
    assets["inference_plan.json"] = _json_bytes(inference_plan)
    return assets


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _populate_formal_tree(
    phase_d, result: Path, task_freeze: dict[str, Any]
) -> None:
    launch = result / "launch"
    launch.mkdir()
    _write_json(launch / "launch_request.json", {"run_id": "synthetic-run"})
    _write_json(launch / "runner_entered.json", {"run_id": "synthetic-run"})
    formal = result / "formal"
    phase_d._create_formal_with_atomic_start(
        formal, {"state": "FORMAL_STARTED", "run_id": "synthetic-run"}
    )
    tasks = task_freeze["tasks"]
    results = _synthetic_formal_results(tasks, intervention_delta=0.004)
    mappings = {row["worker_id"]: row for row in phase_d.EXPECTED_WORKER_MAPPING}
    shards: list[dict[str, Any]] = []
    for worker_id in range(4):
        mapping = mappings[worker_id]
        worker_results = [row for row in results if row["worker_id"] == worker_id]
        attempts = [
            {
                "task_id": row["task_id"],
                "arm": row["arm"],
                "replicate_id": row["replicate_id"],
                "slot_id": row["slot_id"],
                "seed": row["seed"],
                "component_variant": row["component_variant"],
                "worker_id": worker_id,
                "gpu_index": mapping["gpu_index"],
                "gpu_uuid": mapping["gpu_uuid"],
                "attempt": 1,
                "status": "SUCCESS",
                "test_access": phase_d._new_access_ledger(),
            }
            for row in worker_results
        ]
        shard = {
            "schema_version": "phase-d-worker-shard-v1",
            "worker_id": worker_id,
            "gpu_index": mapping["gpu_index"],
            "gpu_uuid": mapping["gpu_uuid"],
            "status": "COMPLETE",
            "setup_complete": True,
            "ledger_complete": True,
            "results": worker_results,
            "attempts": attempts,
            "setup_access_ledger": phase_d._new_access_ledger(),
            "test_access": phase_d._new_access_ledger(),
        }
        shards.append(shard)
        _write_json(formal / "shards" / f"worker_{worker_id}.json", shard)
    _, formal_results, test_ledger = phase_d._write_formal_ledgers(
        result, tasks, shards, 0
    )
    statistics = phase_d.compute_statistics(
        formal_results, tasks, _synthetic_inference_plan()
    )
    challengers = phase_d.freeze_challengers(statistics)
    _write_json(formal / "statistics.json", statistics)
    _write_json(formal / "challengers.json", challengers)
    phase_d._write_success_terminal(result, statistics, challengers, test_ledger)


def _manifest_paths(payload: dict[str, Any]) -> set[str]:
    output = set()
    for row in payload["files"]:
        output.add(str(row.get("path", row.get("relative_path"))))
    return output


def test_manifest_archive_sidecar_and_terminal_consistency(
    phase_d, task_freeze, tmp_path
):
    result = tmp_path / "parent" / "result"
    result.parent.mkdir()
    phase_d.write_prelaunch(result, _coherent_prelaunch_assets(task_freeze))
    assert (result / "manifest.json").is_file()
    _populate_formal_tree(phase_d, result, task_freeze)

    terminal_path = result / "formal/terminal.json"
    valid_terminal_bytes = terminal_path.read_bytes()
    valid_terminal = json.loads(valid_terminal_bytes)
    invalid_terminal = {**valid_terminal, "completed_task_count": 1154}
    _write_json(terminal_path, invalid_terminal)
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.finalize_formal_archive(result)
    terminal_path.write_bytes(valid_terminal_bytes)

    decision_path = result / "formal/decision.json"
    valid_decision_bytes = decision_path.read_bytes()
    valid_decision = json.loads(valid_decision_bytes)
    invalid_decision = {**valid_decision, "wins": 0}
    _write_json(decision_path, invalid_decision)
    rebound_terminal = {
        **valid_terminal,
        "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
    }
    _write_json(terminal_path, rebound_terminal)
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.finalize_formal_archive(result)
    decision_path.write_bytes(valid_decision_bytes)
    terminal_path.write_bytes(valid_terminal_bytes)

    results_path = result / "formal/results.json"
    valid_results_bytes = results_path.read_bytes()
    valid_results = json.loads(valid_results_bytes)
    invalid_results = copy.deepcopy(valid_results)
    invalid_results["results"][0]["seed"] = 99
    _write_json(results_path, invalid_results)
    rebound_terminal = {
        **valid_terminal,
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
    }
    _write_json(terminal_path, rebound_terminal)
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.finalize_formal_archive(result)
    results_path.write_bytes(valid_results_bytes)
    terminal_path.write_bytes(valid_terminal_bytes)

    attempt_path = result / "formal/attempt_ledger.json"
    valid_attempt_bytes = attempt_path.read_bytes()
    invalid_attempts = json.loads(valid_attempt_bytes)
    invalid_attempts["attempts"][0]["test_access"]["test_accessed"] = True
    invalid_attempts["attempts"][0]["test_access"]["forbidden_request_count"] = 1
    _write_json(attempt_path, invalid_attempts)
    rebound_terminal = {
        **valid_terminal,
        "attempt_ledger_sha256": hashlib.sha256(attempt_path.read_bytes()).hexdigest(),
    }
    _write_json(terminal_path, rebound_terminal)
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.finalize_formal_archive(result)
    attempt_path.write_bytes(valid_attempt_bytes)
    terminal_path.write_bytes(valid_terminal_bytes)

    archive_record = phase_d.finalize_formal_archive(result)
    validation = phase_d.validate_formal_archive(result)
    assert validation["passed"] is True
    archive_candidates = list(
        result.rglob("expanded-model-blind-search-global4-v1.tar.gz")
    )
    assert len(archive_candidates) == 1
    archive = archive_candidates[0]
    sidecar = archive.with_name(archive.name + ".sha256")
    assert sidecar.is_file()
    assert sidecar.read_text(encoding="ascii").split()[0] == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    assert archive_record["archive_sha256"] == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()

    prelaunch_manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    formal_manifest = json.loads(
        (result / "formal/manifest.json").read_text(encoding="utf-8")
    )
    expected = {"manifest.json", "formal/manifest.json"}
    expected.update(_manifest_paths(prelaunch_manifest))
    expected.update(_manifest_paths(formal_manifest))
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        assert len(members) == len({member.name for member in members})
        assert all(member.isfile() for member in members)
        prefix = members[0].name.split("/", 1)[0]
        observed = {
            member.name.removeprefix(prefix + "/") for member in members
        }
    assert observed == expected
    assert not any(name.endswith(".tar.gz") for name in observed)
    assert not any(name.endswith(".tar.gz.sha256") for name in observed)
    assert not any(name.endswith("completion.json") for name in observed)

    sidecar.write_text("0" * 64 + f"  {archive.name}\n", encoding="ascii")
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.validate_formal_archive(result)


def test_fail_closed_terminal_is_consistent_without_claiming_completed_work(
    phase_d, task_freeze, tmp_path
):
    result = tmp_path / "failure-parent" / "result"
    result.parent.mkdir()
    phase_d.write_prelaunch(result, _coherent_prelaunch_assets(task_freeze))
    launch = result / "launch"
    launch.mkdir()
    _write_json(launch / "launch_request.json", {"run_id": "synthetic-failure"})
    _write_json(launch / "runner_entered.json", {"run_id": "synthetic-failure"})
    phase_d._create_formal_with_atomic_start(
        result / "formal",
        {"state": "FORMAL_STARTED", "run_id": "synthetic-failure"},
    )
    terminal = phase_d._write_failure_terminal(
        result,
        task_freeze["tasks"],
        [],
        0,
        phase_d.ProtocolFailure("FORMAL", "SYNTHETIC_FAILURE"),
    )
    assert terminal["status"] == "RUN_FAILED"
    assert terminal["completion"] == "INCOMPLETE"
    assert terminal["completed_task_count"] is None
    assert terminal["physical_gnn_attempt_count"] is None
    assert terminal["retry_task_count"] is None
    assert terminal["classification"] is None
    assert terminal["test_accessed"] is None
    archive_record = phase_d.finalize_formal_archive(result)
    assert archive_record["validation"]["passed"] is True


def test_prelaunch_refuses_existing_result_before_staging_write(
    phase_d, tmp_path, monkeypatch
):
    result = tmp_path / "result"
    result.mkdir()

    def forbidden_mkdtemp(*_args: Any, **_kwargs: Any) -> str:
        pytest.fail("staging was created after the result target already existed")

    monkeypatch.setattr(phase_d.tempfile, "mkdtemp", forbidden_mkdtemp)
    with pytest.raises(phase_d.ProtocolFailure):
        phase_d.write_prelaunch(result, _prelaunch_assets())
    assert list(result.iterdir()) == []


def test_synthetic_smoke_is_zero_gnn_zero_test_and_cli_does_not_create_result(
    phase_d, config, tmp_path, monkeypatch
):
    real_result = phase_d.REPO_ROOT / phase_d.RESULT_REL
    assert not real_result.exists()

    def forbidden_real_input(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("synthetic smoke attempted to read a real identity input")

    monkeypatch.setattr(phase_d, "load_exposure_basis", forbidden_real_input)
    monkeypatch.setattr(phase_d, "project_p_anchor", forbidden_real_input)
    direct = phase_d.synthetic_smoke(config)
    assert direct["passed"] is True
    assert direct["gnn_evaluations"] == 0
    assert direct["test_accessed"] is False
    assert direct["result_root_written"] is False
    assert not real_result.exists()

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--config",
            str(CONFIG_PATH),
            "--synthetic-smoke",
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    cli = json.loads(completed.stdout)
    assert cli["gnn_evaluations"] == 0
    assert cli["test_accessed"] is False
    assert cli["result_root_written"] is False
    assert not real_result.exists()
