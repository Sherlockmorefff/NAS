from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "analyse"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import final_checkpoint_seed_inventory as seed_inventory
from analyse import freeze_final_validation_checkpoints as freeze_plan
from analyse import run_final_validation_checkpoints as checkpoint_runner


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reveal = _load("future_reveal_worker", "reveal_frozen_checkpoints_once.py")
analysis = _load("future_reveal_analysis", "analyse_final_checkpoint_reveal.py")


def _identity(path: Path) -> dict:
    value = path.read_bytes()
    return {"size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


class FakeTorch:
    grad_enabled = True

    @staticmethod
    def load(handle, *, weights_only, map_location):
        assert weights_only is True and map_location == "cpu"
        return json.loads(handle.read().decode("utf-8"))

    @classmethod
    @contextlib.contextmanager
    def no_grad(cls):
        old = cls.grad_enabled
        cls.grad_enabled = False
        try:
            yield
        finally:
            cls.grad_enabled = old


class FakeModel:
    def __init__(self, score, counters):
        self.score, self.counters, self.training = score, counters, True

    def eval(self):
        self.training = False

    def __call__(self, offset):
        assert not self.training and not FakeTorch.grad_enabled
        self.counters["forwards"] += 1
        return self.score + offset


def _write_ready(tmp_path: Path, *, identical: bool = False):
    p_fp = "p" * 64
    q_fp = p_fp if identical else "q" * 64
    for role, fp in (("P", p_fp), ("Q", q_fp)):
        (tmp_path / f"frozen_candidate_{role}.json").write_text(
            json.dumps({"joint_config_fingerprint": fp, "candidate_config": {}}),
            encoding="utf-8",
        )

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    inventory, checkpoint_records, checkpoint_paths = [], [], {}
    for seed in reveal.EXPECTED_SEEDS:
        for role, fp, offset in (("P", p_fp, 0.01), ("Q", q_fp, 0.0)):
            filename = f"shared_{seed}.json" if identical else f"{role}_{seed}.json"
            path = checkpoint_dir / filename
            if not path.exists():
                path.write_text(
                    json.dumps({"score": 0.7 + offset + (seed % 3) / 1000}),
                    encoding="utf-8",
                )
            relative = str(path.relative_to(tmp_path))
            identity = _identity(path)
            checkpoint_paths[relative] = path
            row = {
                "task_key": f"{role}_{seed}", "seed": seed, "role": role,
                "joint_config_fingerprint": fp,
                "checkpoint_relative_path": relative,
                "checkpoint_size_bytes": identity["size_bytes"],
                "checkpoint_file_sha256": identity["sha256"],
            }
            inventory.append(row)
            checkpoint_records.append({
                "method_role": role, "seed": seed, "joint_fingerprint": fp,
                "relative_path": relative, "checkpoint_sha256": identity["sha256"],
                "checkpoint_size": identity["size_bytes"],
            })
    (tmp_path / "checkpoint_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")

    source_specs = (
        (reveal.WORKER_RELATIVE, MODULE_ROOT / "reveal_frozen_checkpoints_once.py"),
        (reveal.ANALYSIS_RELATIVE, MODULE_ROOT / "analyse_final_checkpoint_reveal.py"),
        (reveal.ADAPTER_RELATIVE, MODULE_ROOT / "frozen_cora_benchmark_reveal_adapter.py"),
        ("tests/test_final_checkpoint_reveal.py", Path(__file__)),
    )
    dependency_rows, frozen_sources = [], []
    for relative, path in source_specs:
        identity = _identity(path)
        dependency_rows.append({
            "relative_path": relative, "path": str(path.resolve()), **identity,
        })
        frozen_sources.append({"relative_path": relative, **identity})
    dependency = {"files": dependency_rows}
    (tmp_path / "source_dependency_manifest.json").write_text(json.dumps(dependency), encoding="utf-8")

    file_paths = [
        tmp_path / "source_dependency_manifest.json",
        tmp_path / "checkpoint_inventory.json",
        tmp_path / "frozen_candidate_P.json",
        tmp_path / "frozen_candidate_Q.json",
        *checkpoint_paths.values(),
    ]
    files = []
    for path in sorted(set(file_paths)):
        files.append({"relative_path": str(path.relative_to(tmp_path)), **_identity(path)})
    compact = [{key: row[key] for key in ("relative_path", "size_bytes", "sha256")} for row in files]
    manifest_root = reveal.stable_hash({"files": compact, "frozen_future_sources": frozen_sources})
    manifest = {
        "format_version": reveal.FORMAT_VERSION, "files": files,
        "frozen_future_sources": frozen_sources, "checkpoint_count": 30 if identical else 60,
        "manifest_root_hash": manifest_root,
    }
    (tmp_path / "ready_immutable_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    adapter_identity = next(row for row in frozen_sources if row["relative_path"] == reveal.ADAPTER_RELATIVE)
    ready = reveal.seal_ready_document({
        "format_version": reveal.FORMAT_VERSION,
        "ready_schema": reveal.READY_SCHEMA, "ready_root_scheme": reveal.READY_ROOT_SCHEME,
        "status": "TEST_REVEAL_READY", "classification": "TEST_REVEAL_READY",
        "method_level_state": reveal.METHOD_LEVEL_STATE,
        "method_level_status": reveal.METHOD_LEVEL_STATE,
        "P_joint_config_fingerprint": p_fp, "Q_joint_config_fingerprint": q_fp,
        "candidates": {"P": {"joint_fingerprint": p_fp}, "Q": {"joint_fingerprint": q_fp}},
        "final_candidates_identical": identical,
        "fresh_seeds": list(reveal.EXPECTED_SEEDS),
        "fresh_paired_seeds": list(reveal.EXPECTED_SEEDS),
        "checkpoint_count": 30 if identical else 60, "checkpoint_records": checkpoint_records,
        "all_checkpoint_file_and_tensor_hashes_verified": True,
        "ready_manifest_root": manifest_root, "real_benchmark_access_count": 0,
        "reveal_claim_directory_empty": True,
        "fixed_reveal_paths": {
            "claim": reveal.CLAIM_RELATIVE, "result": reveal.RESULT_RELATIVE,
            "failure": reveal.FAILURE_RELATIVE,
        },
        "frozen_benchmark_adapter": adapter_identity,
    })
    ready_path = tmp_path / "TEST_REVEAL_READY.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    return ready_path, ready


def _adapter(counters, *, fail_after=None, score_override=None):
    def score(value, _view):
        if fail_after is not None and counters["forwards"] >= fail_after:
            raise RuntimeError("synthetic post-access failure")
        return value if score_override is None else score_override
    return {
        "load_benchmark_view": lambda: counters.__setitem__(
            "benchmark_loads", counters["benchmark_loads"] + 1
        ) or {"model_args": [0.0], "model_kwargs": {}},
        "build_model": lambda payload, _row: FakeModel(payload["score"], counters),
        "score_output": score,
    }


def _run(tmp_path: Path, ready_path: Path, ready: dict, counters: dict, **adapter_options):
    return reveal._execute_with_synthetic_adapter_for_tests(
        ready_path=ready_path, expected_ready_root=ready["immutable_ready_root"],
        adapter_loader=lambda: _adapter(counters, **adapter_options), torch_module=FakeTorch,
    )


def test_real_ready_shape_claims_fixed_path_and_forwards_once(tmp_path):
    ready_path, ready = _write_ready(tmp_path)
    counters = {"benchmark_loads": 0, "forwards": 0}
    result = _run(tmp_path, ready_path, ready, counters)
    assert counters == {"benchmark_loads": 1, "forwards": 60}
    assert (tmp_path / reveal.CLAIM_RELATIVE).is_file()
    assert (tmp_path / reveal.RESULT_RELATIVE).is_file()
    assert result["all_physical_forward_counts_one"] is True
    bundle = reveal.validate_ready_bundle(ready_path, expected_ready_root=ready["immutable_ready_root"])
    report = analysis.analyse_reveal_result(result, ready_bundle=bundle)
    assert report["classification"] == analysis.P_ADVANTAGE
    assert report["method_level_state_changed"] is False


def test_repeat_rejected_before_any_adapter_access(tmp_path):
    ready_path, ready = _write_ready(tmp_path)
    counters = {"benchmark_loads": 0, "forwards": 0}
    _run(tmp_path, ready_path, ready, counters)
    forbidden = {"loads": 0}
    with pytest.raises(reveal.RevealAlreadyClaimed):
        reveal._execute_with_synthetic_adapter_for_tests(
            ready_path=ready_path, expected_ready_root=ready["immutable_ready_root"],
            adapter_loader=lambda: forbidden.__setitem__("loads", 1), torch_module=FakeTorch,
        )
    assert forbidden["loads"] == 0
    with pytest.raises(TypeError):
        reveal._execute_with_synthetic_adapter_for_tests(
            ready_path=ready_path, expected_ready_root=ready["immutable_ready_root"],
            claim_path=tmp_path / "alternate-claim", adapter_loader=lambda: None,
            torch_module=FakeTorch,
        )


def test_checkpoint_tamper_stops_before_claim(tmp_path):
    ready_path, ready = _write_ready(tmp_path)
    next((tmp_path / "checkpoints").iterdir()).write_text("tampered", encoding="utf-8")
    with pytest.raises(reveal.RevealProtocolError, match="identity mismatch"):
        _run(tmp_path, ready_path, ready, {"benchmark_loads": 0, "forwards": 0})
    assert not (tmp_path / reveal.CLAIM_RELATIVE).exists()


def test_post_access_failure_is_terminal_and_nonretryable(tmp_path):
    ready_path, ready = _write_ready(tmp_path)
    counters = {"benchmark_loads": 0, "forwards": 0}
    with pytest.raises(RuntimeError, match="post-access"):
        _run(tmp_path, ready_path, ready, counters, fail_after=1)
    assert counters["benchmark_loads"] == 1 and counters["forwards"] == 1
    assert (tmp_path / reveal.CLAIM_RELATIVE).is_file()
    failure = json.loads((tmp_path / reveal.FAILURE_RELATIVE).read_text())
    assert failure["terminal"] is True and failure["retry_permitted"] is False
    with pytest.raises(reveal.RevealAlreadyClaimed):
        _run(tmp_path, ready_path, ready, {"benchmark_loads": 0, "forwards": 0})


def test_accuracy_outside_unit_interval_is_terminal(tmp_path):
    ready_path, ready = _write_ready(tmp_path)
    with pytest.raises(reveal.RevealProtocolError, match=r"\[0,1\]"):
        _run(tmp_path, ready_path, ready, {"benchmark_loads": 0, "forwards": 0}, score_override=75.0)
    assert (tmp_path / reveal.FAILURE_RELATIVE).is_file()


def test_identical_candidates_share_checkpoint_and_analysis_special_case(tmp_path):
    ready_path, ready = _write_ready(tmp_path, identical=True)
    counters = {"benchmark_loads": 0, "forwards": 0}
    result = _run(tmp_path, ready_path, ready, counters)
    # Shared logical P/Q identities must produce one physical forward per seed.
    assert result["physical_checkpoint_count"] == 30
    assert counters["forwards"] == 30
    assert all(pair["delta"] == 0 for pair in result["pairs"])
    bundle = reveal.validate_ready_bundle(ready_path, expected_ready_root=ready["immutable_ready_root"])
    report = analysis.analyse_reveal_result(result, ready_bundle=bundle)
    assert report["classification"] == analysis.IDENTICAL


@pytest.mark.parametrize(
    ("mean_delta", "interval", "expected"),
    [
        (0.007, (0.001, 0.013), analysis.P_ADVANTAGE),
        (-0.002, (-0.005, 0.001), analysis.P_NONINFERIOR),
        (-0.007, (-0.013, -0.001), analysis.Q_ADVANTAGE),
        (0.001, (-0.008, 0.010), analysis.INCONCLUSIVE),
    ],
)
def test_frozen_classification_order(mean_delta, interval, expected):
    assert analysis.classify_frozen_candidates(
        mean_delta=mean_delta, ci95=interval, candidates_identical=False
    ) == expected


def test_bootstrap_is_fixed_deterministic_sensitivity():
    deltas = [0.001 * (index - 10) for index in range(30)]
    first = analysis.paired_bootstrap(deltas)
    second = analysis.paired_bootstrap(deltas)
    assert first == second
    assert first["replicates"] == 50_000
    assert first["role"] == "sensitivity_only_not_classification"


def test_analysis_rejects_self_consistent_result_not_joined_to_ready(tmp_path):
    ready_path, ready = _write_ready(tmp_path)
    result = _run(tmp_path, ready_path, ready, {"benchmark_loads": 0, "forwards": 0})
    result["checkpoint_scores"][0]["checkpoint_sha256"] = "f" * 64
    result["result_sha256"] = reveal.stable_hash(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    bundle = reveal.validate_ready_bundle(ready_path, expected_ready_root=ready["immutable_ready_root"])
    report = analysis.analyse_reveal_result(result, ready_bundle=bundle)
    assert report["classification"] == analysis.PROTOCOL_FAILURE


def test_adapter_and_ready_self_hash_tamper_rejected_before_claim(tmp_path):
    ready_path, ready = _write_ready(tmp_path)
    tampered = json.loads(ready_path.read_text())
    tampered["frozen_benchmark_adapter"]["sha256"] = "0" * 64
    ready_path.write_text(json.dumps(tampered))
    with pytest.raises(reveal.RevealProtocolError):
        _run(tmp_path, ready_path, ready, {"benchmark_loads": 0, "forwards": 0})
    assert not (tmp_path / reveal.CLAIM_RELATIVE).exists()


def _executed_seed_row(seed: int) -> dict:
    return {
        "dataset": "Cora", "hp_mode": "global4", "config_id": "cfg",
        "architecture_fingerprint": "a" * 64, "hp_fingerprint": "h" * 64,
        "seed": seed, "terminal_status": "success", "valid": True,
        "validation_accuracy": 0.75,
    }


def test_seed_scanner_token_boundary_target_exclusion_and_wrapper_root(tmp_path):
    assert not seed_inventory._is_test_key("latest_seed")
    assert not seed_inventory.has_forbidden_json_key('{"latest_seed":100005}')
    assert seed_inventory._is_test_key("test_accuracy")
    historical = tmp_path / "results" / "history"
    historical.mkdir(parents=True)
    (historical / "seed_results.jsonl").write_text(
        json.dumps(_executed_seed_row(7)) + "\n", encoding="utf-8"
    )
    excluded = tmp_path / freeze_plan.OUTPUT_RELATIVE
    excluded.mkdir(parents=True)
    (excluded / "seed_results.jsonl").write_text(
        json.dumps(_executed_seed_row(100000)) + "\n", encoding="utf-8"
    )
    wrapper = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=excluded,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert wrapper["passed"] and wrapper["used_training_seeds"] == [7]
    seed_inventory.verify_inventory_wrapper(wrapper, excluded_target=excluded)
    tampered = dict(wrapper); tampered["used_training_seeds"] = [7, 100000]
    with pytest.raises(seed_inventory.InventoryError):
        seed_inventory.verify_inventory_wrapper(tampered, excluded_target=excluded)

    retargeted_inventory = dict(wrapper["inventory"])
    retargeted_inventory["excluded_target"] = "results/final/not-the-fixed-target"
    retargeted_inventory.pop("inventory_root_sha256")
    retargeted_inventory["inventory_root_sha256"] = seed_inventory._sha256_bytes(
        seed_inventory._canonical_json(retargeted_inventory)
    )
    retargeted_wrapper = dict(wrapper)
    retargeted_wrapper["inventory"] = retargeted_inventory
    retargeted_wrapper["inventory_root"] = retargeted_inventory["inventory_root_sha256"]
    retargeted_wrapper.pop("wrapper_root_sha256")
    retargeted_wrapper["wrapper_root_sha256"] = seed_inventory._sha256_bytes(
        seed_inventory._canonical_json(retargeted_wrapper)
    )
    with pytest.raises(seed_inventory.InventoryError, match="non-fixed target"):
        seed_inventory.verify_inventory_wrapper(retargeted_wrapper, excluded_target=excluded)


def test_seed_scanner_fails_closed_on_collision_nested_schema_and_unknown_format(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    (results / "seed_results.jsonl").write_text(
        json.dumps(_executed_seed_row(100000)) + "\n", encoding="utf-8"
    )
    target = tmp_path / freeze_plan.OUTPUT_RELATIVE
    failed = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=target,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert not failed["passed"] and failed["fresh_seed_collisions"] == [100000]
    (results / "seed_results.jsonl").write_text(
        json.dumps({**_executed_seed_row(7), "seed": {"primary": 7}}) + "\n",
        encoding="utf-8",
    )
    nested = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=target,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert not nested["passed"] and "AmbiguousSeedSemanticsError" in nested["failure_reason"]
    (results / "seed_results.jsonl").unlink()
    (results / "unknown.sqlite").write_bytes(b"SQLite format 3\x00")
    unknown = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=target,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert not unknown["passed"] and "unsupported structured" in unknown["failure_reason"]


def test_seed_scanner_accepts_only_exact_manually_audited_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "results" / "audited.db"
    db_path.parent.mkdir(parents=True)
    audited_bytes = b"immutable-manually-audited-optuna-sqlite"
    db_path.write_bytes(audited_bytes)
    relative = "results/audited.db"
    audit = {
        "sha256": hashlib.sha256(audited_bytes).hexdigest(),
        "size": len(audited_bytes),
        "format": "optuna_sqlite",
        "seed_identifier_hits": 0,
        "nested_seed_key_hits": 0,
    }
    monkeypatch.setattr(seed_inventory, "AUDITED_NON_SEED_SQLITE", {relative: audit})
    target = tmp_path / freeze_plan.OUTPUT_RELATIVE
    accepted = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=target,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert accepted["passed"] and accepted["used_training_seeds"] == []
    row = next(item for item in accepted["inventory"]["sources"] if item["source"] == relative)
    assert row["kind"] == "audited_non_seed_sqlite"
    assert row["manual_schema_audit"]["sha256"] == audit["sha256"]

    db_path.write_bytes(audited_bytes + b"-changed")
    rejected = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=target,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert not rejected["passed"]
    assert "requires re-audit" in rejected["failure_reason"]

    db_path.write_bytes(audited_bytes)
    (db_path.parent / "audited.db-wal").write_bytes(b"uncheckpointed-sidecar")
    sidecar_rejected = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=target,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert not sidecar_rejected["passed"]
    assert "SQLite sidecar" in sidecar_rejected["failure_reason"]


@pytest.mark.parametrize(
    "member_name",
    [
        "nested/history.db",
        "nested/history.sqlite",
        "nested/history.sqlite3",
        "nested/history.parquet",
        "nested/history.zip",
        "nested/history.tar.gz",
    ],
)
def test_seed_scanner_fails_closed_on_unsupported_archive_member(tmp_path, member_name):
    archive_path = tmp_path / "results" / "history.tar.gz"
    archive_path.parent.mkdir(parents=True)
    payload = b"opaque-structured-history"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    target = tmp_path / freeze_plan.OUTPUT_RELATIVE
    report = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=target,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert not report["passed"]
    assert "inside tar archive requires audit" in report["failure_reason"]


def test_seed_scanner_exact_run_semantics_count_field_and_implicit_contract(tmp_path, monkeypatch):
    args_path = tmp_path / "results" / "run_args.json"
    result_path = tmp_path / "results" / "final.json"
    args_path.parent.mkdir(parents=True)
    args_bytes = json.dumps({"seed": 42, "n_seeds": 3, "version": "audited"}).encode()
    result_bytes = json.dumps({"val_list": [0.7, 0.8, 0.9]}).encode()
    args_path.write_bytes(args_bytes)
    result_path.write_bytes(result_bytes)
    monkeypatch.setattr(seed_inventory, "AUDITED_STRUCTURED_SEED_SEMANTICS", {
        "results/run_args.json": {
            "sha256": hashlib.sha256(args_bytes).hexdigest(), "size": len(args_bytes),
            "field_path": "$/seed", "expected_seeds": [42],
            "semantic": "executed_gnn_run_rng_initialization_seed", "executed_gnn": True,
        }
    })
    monkeypatch.setattr(seed_inventory, "AUDITED_IMPLICIT_GNN_SEED_SOURCES", {
        "results/final.json": {
            "sha256": hashlib.sha256(result_bytes).hexdigest(), "size": len(result_bytes),
            "field_path": "$/manual_contract/seed", "seeds": [0, 1, 2],
            "semantic": "executed_gnn_training_seed_from_frozen_code_contract",
            "executed_gnn": True,
        }
    })
    target = tmp_path / freeze_plan.OUTPUT_RELATIVE
    report = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=target,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert report["passed"]
    assert report["used_training_seeds"] == [0, 1, 2, 42]
    groups = report["inventory"]["semantic_seed_sets"]
    assert groups["executed_gnn_run_rng_initialization_seed"] == [42]
    assert groups["executed_gnn_training_seed_from_frozen_code_contract"] == [0, 1, 2]
    assert all("n_seeds" not in row["field_path"] for row in report["inventory"]["seed_field_matches"])


def test_seed_scanner_legacy_context_rules_and_per_seed_container(tmp_path):
    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    write("logs/bo_phase4/train_run.json", {
        "seed": 42, "cora_root": "Cora", "output": "results/run",
        "eval_epochs": 300, "patience": 80,
    })
    write("logs/final_eval/train_run.json", {
        "seed": 42, "n_seeds": 10, "n_eval_seeds": 10,
        "output": "results/final", "eval_epochs": 300, "patience": 80,
    })
    write("results/run/final_results.json", [{
        "operations": ["GCNConv"], "edges": [[0, 1]],
        "per_seed": [
            {"operations": ["GCNConv"], "edges": [[0, 1]], "seed": 3,
             "val_acc": 0.7, "valid": True, "best_epoch": 11},
            {"operations": ["GCNConv"], "edges": [[0, 1]], "seed": 4,
             "val_acc": 0.8, "valid": True, "best_epoch": 12},
        ],
    }])
    write("results/run/cluster_quota.json", {
        "cluster_plan_fingerprint": "x", "promotion": {}, "shortlist": [],
        "seed": [5, 17, 99],
    })
    csv_path = tmp_path / "results" / "run" / "gp_predictions.csv"
    csv_path.write_text(
        "candidate_eval_seed,val_acc,best_epoch,epochs_ran,valid,record_type\n"
        "123,0.77,9,12,true,evaluation\n",
        encoding="utf-8",
    )
    (tmp_path / "results" / "run" / "prequential_predictions.csv").write_text(
        "surrogate_type,refit_seed,valid,actual,train_size_before,train_size_after\n"
        "dkl,999,true,0.77,10,11\n",
        encoding="utf-8",
    )
    write("results/confirmation/selected_history_records_global4.json", [{
        "operations": ["GCNConv"], "edges": [[0, 1]], "val_acc": 0.8,
        "valid": True, "best_epoch": 7,
        "replicates": [{"replicate_id": 0, "final_eval_seed": 555}],
    }])
    write("results/posthoc/layernorm_results.json", {
        "by_architecture": {"A": {
            "per_seed": [{"training_seed": 777, "valid": True,
                          "baseline_validation": 0.7, "baseline_best_epoch": 8}],
            "validation_positive_seeds": [0],
        }},
        "analysis_seed": 222,
    })
    write("results/configs/confirmation/cora_confirmation_seeds.json", {
        "confirmation_seeds": [888, 889], "seeds": [888, 889],
        "candidate_training_seeds": {"A": [888, 890], "B": [889]},
    })
    write("results/stage3_task_plan.json", {
        "tasks": [{"task_id": "t", "seed": 14}],
        "existing_seeds": [0, 1, 2], "new_seeds": [3, 4],
    })
    (tmp_path / "results" / "paired_policy_differences.csv").write_text(
        "point_id,seed,difference\n1,15,0.01\n", encoding="utf-8",
    )
    write("results/random_rankings/random_ranking_manifest.json", {
        "cells": [{"cell_seed": 2222}],
    })
    write("results/matched_relabel_run_plan.json", {"seeds": [16, 17]})
    write("results/representation_comparison.json", {
        "metadata": {"seed": 20260821, "expected_seeds": [0, 1]},
    })
    write("results/replay_plan_v2.json", {"random_baseline": {"seed": 20260826}})
    target = tmp_path / freeze_plan.OUTPUT_RELATIVE
    report = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=target,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert report["passed"]
    assert report["used_training_seeds"] == [3, 4, 42, 123, 555, 777]
    semantics = report["inventory"]["semantic_seed_sets"]
    assert semantics["executed_gnn_run_rng_initialization_seed"] == [42]
    assert semantics["executed_gnn_training_seed"] == [3, 4, 777]
    assert semantics["executed_gnn_candidate_evaluation_seed"] == [123]
    assert semantics["non_training_final_eval_orchestration_seed"] == [42]
    assert semantics["non_training_cluster_or_selection_bucket"] == []
    assert semantics["non_gnn_proxy_model_training_seed"] == [999]
    assert semantics["executed_gnn_final_evaluation_seed"] == [555]
    assert set(semantics["declared_gnn_seed_list_not_execution_evidence"]) == {
        0, 1, 2, 3, 4, 16, 17, 888, 889, 890,
    }
    assert semantics["non_training_validation_subset_index"] == [0]
    assert 222 in semantics["non_training_random_or_statistical_seed"]
    assert 14 in semantics["planned_gnn_training_seed_not_executed"]
    assert semantics["derived_gnn_seed_reference_not_execution_evidence"] == [15]
    assert {2222, 20260826}.issubset(set(semantics["non_training_random_or_statistical_seed"]))
    assert 20260821 in semantics["non_training_analysis_seed"]
    assert {16, 17}.issubset(set(semantics["declared_gnn_seed_list_not_execution_evidence"]))
    assert 11 not in report["used_training_seeds"] and 12 not in report["used_training_seeds"]


def test_seed_scanner_schema_guards_for_current_plans_generation_and_non_gnn_training(tmp_path):
    def write(relative, value, *, jsonl=False):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value)
        path.write_text(payload + ("\n" if jsonl else ""), encoding="utf-8")

    write("results/plans/run_plan.json", {
        "seeds": [0, 1, 2], "shard": 0, "expected_task_count": 12,
        "pending_task_count": 0, "already_completed": 12, "dry_run": False,
    })
    write("results/plans/gpu0.stdout.json", {
        "seeds": [0, 1, 2], "shard": 0, "expected_training_count": 12,
        "pending_training_count": 12, "already_completed": 0, "dry_run": True,
    })
    write("results/plans/gpu0_initial_shard_plan.json", {
        "seeds": list(range(15)), "gpu": 0, "expected_training_count": 30,
        "architecture_fingerprints": ["a" * 64], "config_ids": ["cfg"],
    })
    write("results/plans/preregistration.json", {
        "training": {
            "seeds": [0, 1, 2], "dataset": "Cora", "max_epochs": 300,
            "patience": 80, "split": "public",
        }
    })
    write("results/plans/adaptive_selection.json", {
        "seeds": list(range(15, 30)), "config_ids": ["cfg"],
        "configuration_count": 1, "prediction_correctness_read": False,
        "rule": "frozen",
    })
    write("results/plans/label_task_plan.json", {
        "adaptive_seeds": list(range(15, 30)), "initial_seeds": list(range(15)),
        "code_hashes": {}, "merge_key": ["config_id", "seed"],
        "selection_sha256": "a" * 64,
    })
    write("results/plans/protocol.json", {
        "clean24_adaptive_seeds": list(range(15, 30)),
        "clean24_initial_seeds": list(range(15)),
        "core_dataset_split_fidelity_and_early_stopping_unchanged": True,
        "dataset": "Cora", "max_epochs": 300, "patience": 80,
        "training_seeds": [0, 1, 2], "maximum_system_retry_per_config_seed": 1,
    })
    write("results/plans/stabilization_firewall_audit.json", {
        "additional_seed_ids": list(range(3, 30)), "additional_task_count": 27,
        "maximum_additional_tasks": 432, "top8_selection_read_seed_ids": [0, 1, 2],
        "union_configuration_count": 1,
    })

    write("results/generation/candidate_pool_audit.json", {
        "generation_seed": 20260828, "hp_seed": 20260829,
        "generator": "frozen_vae", "generation_draws": 100,
        "hp_regimes": ["low", "high"], "invalid_counts": {},
    })
    write("results/generation/candidate_generation_audit.json", [{
        "seed": 77858950, "pool_id": 0, "draw_schedule": [64, 128],
        "draws_used": 64, "fresh_unique_rate": 0.8,
        "selected_architecture_count": 64,
    }])
    write("results/generation/partition_audit.json", [{
        "seed": 57904987, "pool_id": 0, "cluster_sizes": [4, 4, 4, 4],
        "distance_definition": "fixed", "medoid_config_ids": ["a", "b"],
        "silhouette": 0.2,
    }])
    write("results/generation/random_rejection_manifest.json", {
        "seed": 20260832, "draws": 100, "permutations_path": "permutations.npy",
        "permutations_sha256": "b" * 64, "shape": [10, 10],
    })
    write("results/generation/split_reproduction_audit.json", {
        "fixed_seed": 20260821, "expected_split_count": 150,
        "observed_split_count": 150, "grouped_repeated_parameters": {},
        "validation_membership_mismatches": 0,
    })

    vae_record = {
        "training_seed": 20260831, "task_id": "vae-0", "variant": "lambda_01",
        "outer_fold": 0, "noise_mode": "include", "performance_strength": 0.1,
    }
    write("results/posthoc/performance-aware-vae-pilot/task_plan.json", {
        "tasks": [vae_record],
    })
    write("results/posthoc/performance-aware-vae-pilot/training_results.jsonl", {
        **vae_record, "inner_metrics": {}, "outer_predictions": [],
        "original_checkpoint_sha256": "c" * 64, "best_epoch": 4,
    }, jsonl=True)
    write("results/posthoc/performance-aware-vae-pilot/reconstruction_audit.json", {
        "candidate_checkpoints": [{
            "training_seed": 20260832, "checkpoint": "vae.pt",
            "checkpoint_sha256": "d" * 64, "metrics": {}, "outer_fold": 1,
        }],
    })

    target = tmp_path / freeze_plan.OUTPUT_RELATIVE
    report = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=target,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert report["passed"] and report["used_training_seeds"] == []
    semantics = report["inventory"]["semantic_seed_sets"]
    assert semantics["non_seed_retry_limit_count"] == [1]
    assert {20260828, 20260829, 77858950}.issubset(
        set(semantics["non_training_candidate_generation_seed"])
    )
    assert semantics["non_training_partition_seed"] == [57904987]
    assert semantics["non_gnn_proxy_split_seed"] == [20260821]
    assert set(semantics["non_gnn_representation_or_proxy_training_seed"]) == {
        20260831, 20260832,
    }
    declared = set(semantics["declared_gnn_seed_list_not_execution_evidence"])
    assert set(range(30)).issubset(declared)
    assert 20260832 in semantics["non_training_random_or_statistical_seed"]


def test_seed_scanner_rng_provenance_is_narrow_and_not_gnn_execution(tmp_path):
    required = {
        "seed_derivation": "sha256_v1",
        "search_seed": 3,
        "lhs_seed": 3,
        "wgmm_fit_seed": 2445206625,
        "full_fidelity_seed_context": ["candidate", "full"],
        "low_fidelity_seed_context": ["candidate", "low"],
        "python_hash_used": False,
        # Optional GMM provenance fields vary across the 75 historical files.
        "gmm_component_selection": "fixed",
        "gmm_fixed_restart_seeds": [11, 12],
    }
    good_root = tmp_path / "good"
    good_path = good_root / "results" / "run" / "rng_provenance.json"
    good_path.parent.mkdir(parents=True)
    good_path.write_text(json.dumps(required), encoding="utf-8")
    good = seed_inventory.scan_repository_training_seeds(
        repo_root=good_root,
        excluded_target=good_root / freeze_plan.OUTPUT_RELATIVE,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert good["passed"] and good["used_training_seeds"] == []
    semantics = good["inventory"]["semantic_seed_sets"]
    assert semantics["non_training_candidate_generation_seed"] == [3]
    assert semantics["non_gnn_proxy_or_clustering_seed"] == [2445206625]

    bad_root = tmp_path / "bad"
    bad_path = bad_root / "results" / "run" / "rng_provenance.json"
    bad_path.parent.mkdir(parents=True)
    near_miss = dict(required)
    near_miss.pop("low_fidelity_seed_context")
    bad_path.write_text(json.dumps(near_miss), encoding="utf-8")
    bad = seed_inventory.scan_repository_training_seeds(
        repo_root=bad_root,
        excluded_target=bad_root / freeze_plan.OUTPUT_RELATIVE,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert not bad["passed"]
    assert "AmbiguousSeedSemanticsError" in bad["failure_reason"]


def test_seed_scanner_new_schema_rules_remain_fail_closed_for_near_misses(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    # A suggestive filename or one matching sibling key is insufficient: all
    # schema guards must be present before an unknown seed can be excluded.
    (results / "run_plan.json").write_text(json.dumps({
        "seeds": [81], "shard": 0, "expected_task_count": 1,
    }), encoding="utf-8")
    (results / "candidate_generation_audit.json").write_text(json.dumps([{
        "seed": 82, "pool_id": 0, "draw_schedule": [1], "draws_used": 1,
    }]), encoding="utf-8")
    (results / "performance-aware-vae-pilot-training.json").write_text(json.dumps({
        "training_seed": 83, "task_id": "unknown", "variant": "unknown",
    }), encoding="utf-8")
    report = seed_inventory.scan_repository_training_seeds(
        repo_root=tmp_path, excluded_target=tmp_path / freeze_plan.OUTPUT_RELATIVE,
        proposed_seeds=freeze_plan.FRESH_SEEDS,
    )
    assert not report["passed"]
    assert "AmbiguousSeedSemanticsError" in report["failure_reason"]
    assert all(value not in report.get("used_training_seeds", []) for value in (81, 82, 83))


def test_pair_plan_exact_fresh_seeds_same_gpu_adjacency_and_balance():
    candidates = {
        "P": {"joint_config_fingerprint": freeze_plan.EXPECTED_P, "strategy": "P-strategy"},
        "Q": {"joint_config_fingerprint": freeze_plan.EXPECTED_Q, "strategy": "Q-strategy"},
    }
    pairs, tasks = freeze_plan._pair_plan(candidates)
    assert len(pairs) == 30 and len(tasks) == 60
    assert {row["seed"] for row in pairs} == set(range(100000, 100030))
    assert [sum(row["assigned_physical_gpu_index"] == gpu for row in pairs) for gpu in (0, 1)] == [15, 15]
    for pair in pairs:
        rows = [row for row in tasks if row["pair_id"] == pair["pair_id"]]
        assert len(rows) == 2
        assert len({row["seed"] for row in rows}) == 1
        assert len({row["assigned_gpu_uuid"] for row in rows}) == 1
        assert sorted(row["within_pair_order"] for row in rows) == [0, 1]
    per_gpu = {}
    for gpu in (0, 1):
        gpu_pairs = [row for row in pairs if row["assigned_physical_gpu_index"] == gpu]
        per_gpu[gpu] = (
            sum(row["pair_execution_order"] == ["P", "Q"] for row in gpu_pairs),
            sum(row["pair_execution_order"] == ["Q", "P"] for row in gpu_pairs),
        )
    assert per_gpu == {0: (8, 7), 1: (7, 8)}


def _synthetic_sealed_view() -> dict:
    train = torch.zeros(freeze_plan.EXPECTED_CORA_NUM_NODES, dtype=torch.bool)
    val = torch.zeros_like(train)
    train[:freeze_plan.EXPECTED_CORA_TRAIN_COUNT] = True
    val[freeze_plan.EXPECTED_CORA_TRAIN_COUNT:
        freeze_plan.EXPECTED_CORA_TRAIN_COUNT + freeze_plan.EXPECTED_CORA_VALIDATION_COUNT] = True
    labels = torch.full((freeze_plan.EXPECTED_CORA_NUM_NODES,), -1, dtype=torch.long)
    labels[train | val] = torch.arange(int((train | val).sum())) % 7
    return {
        "x": torch.zeros(
            (freeze_plan.EXPECTED_CORA_NUM_NODES, freeze_plan.EXPECTED_CORA_NUM_FEATURES),
            dtype=torch.float32,
        ),
        "edge_index": torch.zeros((2, freeze_plan.EXPECTED_CORA_EDGE_ENTRIES), dtype=torch.long),
        "train_mask": train, "val_mask": val, "y_sealed": labels,
        "num_features": freeze_plan.EXPECTED_CORA_NUM_FEATURES, "num_classes": 7,
    }


def test_sealed_view_exact_allowlist_poison_and_weights_only_load(tmp_path):
    view = _synthetic_sealed_view()
    observed = freeze_plan.validate_sealed_view(view)
    assert observed["outside_nonnegative_label_count"] == 0
    path = tmp_path / "sealed.pt"; torch.save(view, path)
    loaded = checkpoint_runner.load_sealed_worker_view(path)
    assert set(loaded) == freeze_plan.SEALED_KEYS
    with pytest.raises(RuntimeError, match="key mismatch"):
        freeze_plan.validate_sealed_view({**view, "extra": torch.tensor(1)})
    poisoned = dict(view); poisoned["y_sealed"] = view["y_sealed"].clone()
    poisoned["y_sealed"][-1] = 0
    with pytest.raises(ValueError, match="poisoned"):
        freeze_plan.validate_sealed_view(poisoned)


def test_training_worker_signature_and_source_cannot_receive_dataset_or_test():
    signature = inspect.signature(checkpoint_runner.train_validation_best)
    assert "track_test" not in signature.parameters
    assert "cora_root" not in signature.parameters
    source = (MODULE_ROOT / "run_final_validation_checkpoints.py").read_text(encoding="utf-8")
    assert "Planetoid" not in source
    assert "torch_geometric.datasets" not in source


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[0.2, -0.1], [-0.2, 0.1]]))

    def forward(self, _x, _edge_index):
        return self.logits


def test_validation_best_captures_earliest_tie_and_not_last(tmp_path, monkeypatch):
    sequence = iter([0.5, 0.6, 0.6, 0.4, 0.6, 0.6])
    monkeypatch.setattr(checkpoint_runner, "build_model", lambda *_args, **_kwargs: _ToyModel())
    monkeypatch.setattr(checkpoint_runner, "validation_accuracy", lambda *_args, **_kwargs: next(sequence))
    candidate = {
        "method_identity": "synthetic",
        "candidate_config": {"lr": 0.01, "l2": 0.0},
    }
    task = {
        "task_key": "synthetic", "partner_task_key": "partner", "pair_id": 0,
        "seed": 100000, "role": "P", "strategy": "synthetic",
        "joint_config_fingerprint": "j" * 64,
        "assigned_physical_gpu_index": 0, "assigned_gpu_uuid": "gpu",
        "ordinal_on_gpu": 0,
    }
    sealed = {
        "x": torch.zeros((2, 1)), "edge_index": torch.empty((2, 0), dtype=torch.long),
        "train_mask": torch.tensor([True, True]), "val_mask": torch.tensor([True, True]),
        "y_sealed": torch.tensor([0, 1]), "num_features": 1, "num_classes": 2,
    }
    checkpoint = tmp_path / "root" / "checkpoints" / "pair" / "P.pt"
    result = checkpoint_runner.train_validation_best(
        candidate=candidate, task=task, sealed_cpu=sealed,
        sealed_identity={"combined_tensor_root": "s", "sealed_file": {"sha256": "f"}},
        protocol_sha256="p", pretraining_immutable_root="r", device=torch.device("cpu"),
        checkpoint_path=checkpoint, max_epochs=4, patience=4,
    )
    assert result["best_epoch"] == 2
    assert result["epochs_ran"] == 4
    assert result["best_state_tensor_sha256"] != result["last_state_tensor_sha256"]
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert set(payload) == checkpoint_runner.CHECKPOINT_PAYLOAD_KEYS
    assert payload["best_epoch"] == 2 and payload["best_validation"] == 0.6
