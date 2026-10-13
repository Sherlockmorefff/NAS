from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import eval_utils
from architecture_diagnostics import (
    architecture_fingerprint,
    config_to_igraph,
)
from analyse import compare_matched_representations as compare
from analyse import run_matched_relabel as relabel


class _Vertex:
    def __init__(self, vertex_type: int):
        self.vertex_type = vertex_type

    def __getitem__(self, key: str):
        assert key == "type"
        return self.vertex_type


class _Vertices:
    def __init__(self, types):
        self.types = list(types)

    def __getitem__(self, index):
        return _Vertex(self.types[index])


class _Graph:
    def __init__(self, types, edges):
        self.vs = _Vertices(types)
        self.edges = list(edges)

    def vcount(self):
        return len(self.vs.types)

    def get_edgelist(self):
        return list(self.edges)


class _Decoder:
    def __init__(self, graphs):
        self.graphs = list(graphs)
        self.calls = []

    def decode(self, _z, stochastic=True):
        self.calls.append(stochastic)
        index = min(len(self.calls) - 1, len(self.graphs) - 1)
        return [self.graphs[index]]


def _vae(decoder):
    return SimpleNamespace(
        arch_vae=decoder,
        op_mapping={2: "GCNConv", 3: "GATConv", 6: "Identity"},
    )


def test_deterministic_greedy_repeats_the_same_fingerprint():
    graph = _Graph([0, 2, 3, 1], [(0, 1), (1, 2), (2, 3)])
    decoder = _Decoder([graph])
    model = _vae(decoder)
    fingerprints = {
        architecture_fingerprint(
            eval_utils._decode_arch(
                model,
                torch.zeros(12),
                torch.device("cpu"),
                decoder_policy="deterministic_greedy",
            )
        )
        for _ in range(5)
    }
    assert len(fingerprints) == 1
    assert decoder.calls == [False] * 5


def test_historical_modal3_is_the_backward_compatible_default():
    first = _Graph([0, 2, 1], [(0, 1), (1, 2)])
    second = _Graph([0, 3, 1], [(0, 1), (1, 2)])
    decoder = _Decoder([first, second, second])
    config = eval_utils._decode_arch(
        _vae(decoder), torch.zeros(12), torch.device("cpu")
    )
    assert config["operations"] == ["GATConv"]
    assert decoder.calls == [True, True, True]
    assert inspect.signature(eval_utils.eval_z_search).parameters[
        "decoder_policy"
    ].default == "historical_modal3"


def test_architecture_fingerprint_sorts_edges_but_preserves_positions():
    left = {"operations": ["GCNConv", "GATConv"], "edges": [(1, 2), (0, 1), (2, 3)]}
    reordered = {"operations": ["GCNConv", "GATConv"], "edges": [(2, 3), (1, 2), (0, 1)]}
    changed_operation = {**left, "operations": ["GATConv", "GCNConv"]}
    changed_edge = {**left, "edges": [(0, 1), (2, 3)]}
    assert architecture_fingerprint(left) == architecture_fingerprint(reordered)
    assert architecture_fingerprint(left) != architecture_fingerprint(changed_operation)
    assert architecture_fingerprint(left) != architecture_fingerprint(changed_edge)


def test_config_to_igraph_preserves_recorded_node_positions():
    graph = config_to_igraph(
        {"operations": ["GATConv", "GCNConv"], "edges": [(0, 2), (2, 3)]}
    )
    assert graph.vs["type"] == [0, 3, 2, 1]
    assert graph.get_edgelist() == [(0, 2), (2, 3)]


def test_history_recovery_never_decodes_historical_architecture(monkeypatch):
    calls = []

    def fake_decode(*args, **kwargs):
        calls.append(kwargs["decoder_policy"])
        return {
            "effective_layers": 1,
            "operations": ["GATConv"],
            "edges": [(0, 1), (1, 2)],
        }

    monkeypatch.setattr(relabel, "_decode_arch", fake_decode)
    monkeypatch.setattr(relabel, "encode_architecture_mu", lambda *_args: [0.0] * 12)
    record = {
        "candidate_fingerprint": "point",
        "z_search": [0.0] * 12 + [0.5] * 4,
        "operations": ["GCNConv"],
        "edges": [(0, 1), (1, 2)],
        "val_acc": 0.8,
        "_time_index": 0,
        "_history_index": 0,
    }
    selected = relabel.build_selection(
        [record], object(), "global4", 1, device=torch.device("cpu")
    )
    assert calls == ["deterministic_greedy"]
    assert selected[0]["historical_config"]["operations"] == ["GCNConv"]


def test_dry_run_does_not_load_dataset_or_start_training(tmp_path, monkeypatch):
    history = tmp_path / "history.json"
    history.write_text("[]\n", encoding="utf-8")
    selection = [
        {
            "point_id": "p0",
            "time_index": 0,
            "history_path": str(history),
            "history_index": 0,
            "historical_val_acc": 0.8,
            "z_arch": [0.0] * 12,
            "raw_hp_vector": [0.5] * 4,
            "z_search": [0.0] * 12 + [0.5] * 4,
            "historical_config": {"operations": ["GCNConv"], "edges": [(0, 1), (1, 2)]},
            "greedy_config": {"operations": ["GATConv"], "edges": [(0, 1), (1, 2)]},
        }
    ]
    monkeypatch.setattr(relabel, "load_joint_vae", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(relabel, "build_selection", lambda *_args, **_kwargs: selection)
    monkeypatch.setattr(
        relabel,
        "load_dataset_from_request",
        lambda *_args, **_kwargs: pytest.fail("dry-run loaded a dataset"),
    )
    monkeypatch.setattr(
        relabel,
        "eval_z_search",
        lambda *_args, **_kwargs: pytest.fail("dry-run started training"),
    )
    output = tmp_path / "out"
    assert relabel.main(
        [
            "--history", str(history),
            "--checkpoint", str(tmp_path / "checkpoint.pth"),
                "--output", str(output),
                "--sample_count", "1",
                "--dry_run",
        ]
    ) == 0
    plan = json.loads((output / relabel.RUN_PLAN_JSON).read_text(encoding="utf-8"))
    assert plan["expected_total_trainings"] == 6
    assert not (output / relabel.RESULTS_JSONL).exists()


def test_fixed_selection_dry_run_bypasses_history_sampling_and_decoding(
    tmp_path, monkeypatch
):
    history = tmp_path / "history.json"
    history.write_text("[]\n", encoding="utf-8")
    historical = {"operations": ["GCNConv"], "edges": [(0, 1), (1, 2)]}
    greedy = {"operations": ["GATConv"], "edges": [(0, 1), (1, 2)]}
    selection = [{
        "point_id": "p0", "time_index": 0, "history_path": str(history),
        "history_index": 0, "historical_val_acc": 0.8,
        "z_arch": [0.0] * 12, "raw_hp_vector": [0.5] * 4,
        "z_search": [0.0] * 12 + [0.5] * 4,
        "historical_config": historical, "greedy_config": greedy,
        "historical_fingerprint": architecture_fingerprint(historical),
        "greedy_fingerprint": architecture_fingerprint(greedy),
        "historical_mu_arch": [0.0] * 12, "greedy_mu_arch": [0.1] * 12,
        "cycle_distance_l2_historical": 0.0,
        "cycle_distance_normalized_historical": 0.0,
        "cycle_distance_l2_greedy": 0.1,
        "cycle_distance_normalized_greedy": 0.1,
    }]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setattr(relabel, "load_joint_vae", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        relabel, "build_selection",
        lambda *_args, **_kwargs: pytest.fail("fixed selection rebuilt or decoded"),
    )
    monkeypatch.setattr(
        relabel, "load_dataset_from_request",
        lambda *_args, **_kwargs: pytest.fail("dry-run loaded a dataset"),
    )
    output = tmp_path / "out"
    assert relabel.main([
        "--history", str(history), "--checkpoint", str(tmp_path / "checkpoint.pth"),
        "--selection_file", str(selection_path), "--sample_count", "1",
        "--output", str(output), "--dry_run",
    ]) == 0
    plan = json.loads((output / relabel.RUN_PLAN_JSON).read_text())
    assert plan["selection_source"] == str(selection_path.resolve())


def test_resume_does_not_repeat_completed_point_policy_seed():
    selection = [{"point_id": "p0"}]
    completed = {("p0", "historical", 0), ("p0", "greedy", 1)}
    pending = relabel.pending_evaluations(selection, [0, 1], completed)
    keys = {(row[0]["point_id"], row[1], row[2]) for row in pending}
    assert keys == {("p0", "historical", 1), ("p0", "greedy", 0)}


def test_multiseed_aggregation_requires_all_three_successful_seeds():
    base = {
        "point_id": "p0",
        "decoder_policy": "historical_modal3",
        "architecture_version": "historical",
        "architecture_fingerprint": "arch",
        "operations": ["GCNConv"],
        "edges": [(0, 1), (1, 2)],
        "z_arch": [0.0] * 12,
        "mu_arch": [0.0] * 12,
        "raw_hp_vector": [0.5] * 4,
        "condition_mask_vector": [1.0] * 4,
        "time_index": 0,
        "val_acc": 0.8,
        "valid": True,
        "terminal_status": "success",
    }
    assert compare.aggregate_multiseed([{**base, "seed": 0}, {**base, "seed": 1}]) == []
    aggregated = compare.aggregate_multiseed(
        [{**base, "seed": seed, "val_acc": 0.8 + seed * 0.01} for seed in (0, 1, 2)]
    )
    assert len(aggregated) == 1
    assert aggregated[0]["seed_count"] == 3
    assert aggregated[0]["label"] == pytest.approx(0.81)


def test_grouped_splits_have_no_architecture_leakage():
    records = [
        {"architecture_fingerprint": f"arch-{index // 2}", "time_index": index}
        for index in range(12)
    ]
    splits = compare.grouped_repeated_splits(records, repeats=5, seed=7)
    assert len(splits) == 5
    for split in splits:
        compare.assert_no_architecture_leakage(
            records, split["train"], split["validation"]
        )


def test_standardization_uses_training_statistics_only():
    train = np.asarray([[0.0], [2.0]])
    validation = np.asarray([[1000.0]])
    transformed_train, transformed_validation, scaler = (
        compare.standardize_train_validation(train, validation)
    )
    assert scaler.mean_.tolist() == [1.0]
    assert np.allclose(transformed_train[:, 0], [-1.0, 1.0])
    assert transformed_validation[0, 0] == pytest.approx(999.0)


def test_seed_is_forwarded_from_diagnostic_to_eval_z_search(tmp_path, monkeypatch):
    # The existing eval_utils test covers eval_z_search -> train_and_eval_arch.
    # This test covers the diagnostic script -> eval_z_search half of the chain.
    history = tmp_path / "history.json"
    history.write_text("[]\n", encoding="utf-8")
    config = {"operations": ["GCNConv"], "edges": [(0, 1), (1, 2)]}
    selection = [{
        "point_id": "p0", "time_index": 0, "history_path": str(history),
        "history_index": 0, "historical_val_acc": 0.8,
        "z_search": [0.0] * 12 + [0.5] * 4,
        "historical_config": config, "greedy_config": config,
    }]
    captured = []
    monkeypatch.setattr(relabel, "load_joint_vae", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(relabel, "build_selection", lambda *_args, **_kwargs: selection)
    data = SimpleNamespace(to=lambda _device: object())
    bundle = SimpleNamespace(data=data, num_features=8, num_classes=3)
    monkeypatch.setattr(relabel, "load_dataset_from_request", lambda *_args: bundle)
    monkeypatch.setattr(relabel, "resolve_dataset_request", lambda *_args, **_kwargs: object())

    def fake_eval(*_args, **kwargs):
        captured.append(kwargs["candidate_eval_seed"])
        return {
            "decoder_policy": kwargs["decoder_policy"], "architecture_fingerprint": "arch",
            "z_arch": [0.0] * 12, "mu_arch": [0.0] * 12,
            "cycle_distance_l2": 0.0, "cycle_distance_normalized": 0.0,
            "seed": kwargs["candidate_eval_seed"], "operations": ["GCNConv"],
            "edges": [(0, 1), (1, 2)], "raw_hp_vector": [0.5] * 4,
            "realized_hp": {}, "condition_mask_vector": [1.0] * 4,
            "val_acc": 0.8, "valid": True,
        }

    monkeypatch.setattr(relabel, "eval_z_search", fake_eval)
    assert relabel.main([
        "--history", str(history), "--checkpoint", str(tmp_path / "checkpoint.pth"),
        "--output", str(tmp_path / "out"), "--seeds", "13", "--sample_count", "1",
    ]) == 0
    assert captured == [13, 13]


def test_training_exception_records_failed_terminal_and_continues(tmp_path, monkeypatch):
    history = tmp_path / "history.json"
    history.write_text("[]\n", encoding="utf-8")
    historical = {"operations": ["GCNConv"], "edges": [(0, 1), (1, 2)]}
    greedy = {"operations": ["GATConv"], "edges": [(0, 1), (1, 2)]}
    selection = [{
        "point_id": "p0", "time_index": 0, "history_path": str(history),
        "history_index": 0, "historical_val_acc": 0.8,
        "z_arch": [0.0] * 12, "raw_hp_vector": [0.5] * 4,
        "z_search": [0.0] * 12 + [0.5] * 4,
        "historical_config": historical, "greedy_config": greedy,
        "historical_fingerprint": architecture_fingerprint(historical),
        "greedy_fingerprint": architecture_fingerprint(greedy),
        "historical_mu_arch": [0.0] * 12, "greedy_mu_arch": [0.1] * 12,
        "cycle_distance_l2_historical": 0.0,
        "cycle_distance_normalized_historical": 0.0,
        "cycle_distance_l2_greedy": 0.1,
        "cycle_distance_normalized_greedy": 0.1,
    }]
    monkeypatch.setattr(relabel, "load_joint_vae", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(relabel, "build_selection", lambda *_args, **_kwargs: selection)
    data = SimpleNamespace(to=lambda _device: object())
    monkeypatch.setattr(
        relabel,
        "load_dataset_from_request",
        lambda *_args: SimpleNamespace(data=data, num_features=8, num_classes=3),
    )
    monkeypatch.setattr(relabel, "resolve_dataset_request", lambda *_args, **_kwargs: object())
    calls = 0

    def fake_eval(*_args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("permanent synthetic failure")
        return {
            "decoder_policy": kwargs["decoder_policy"],
            "architecture_fingerprint": architecture_fingerprint(greedy),
            "z_arch": [0.0] * 12, "mu_arch": [0.1] * 12,
            "cycle_distance_l2": 0.1, "cycle_distance_normalized": 0.1,
            "operations": ["GATConv"], "edges": [(0, 1), (1, 2)],
            "raw_hp_vector": [0.5] * 4, "realized_hp": {},
            "condition_mask_vector": [1.0] * 4,
            "val_acc": 0.81, "valid": True,
        }

    monkeypatch.setattr(relabel, "eval_z_search", fake_eval)
    output = tmp_path / "out"
    assert relabel.main([
        "--history", str(history), "--checkpoint", str(tmp_path / "checkpoint.pth"),
        "--output", str(output), "--seeds", "0", "--sample_count", "1",
    ]) == 0
    rows = [json.loads(line) for line in (output / relabel.RESULTS_JSONL).read_text().splitlines()]
    assert [row["terminal_status"] for row in rows] == ["failed", "success"]
    assert rows[0]["val_acc"] is None
    assert "permanent synthetic failure" in rows[0]["failure"]["traceback"]
    assert json.loads((output / relabel.PROGRESS_JSON).read_text())["remaining"] == 0


def test_retry_all_failures_records_each_attempt_and_stops_after_two_retries(
    tmp_path, monkeypatch
):
    history = tmp_path / "history.json"
    history.write_text("[]\n", encoding="utf-8")
    historical = {"operations": ["GCNConv"], "edges": [(0, 1), (1, 2)]}
    greedy = {"operations": ["GATConv"], "edges": [(0, 1), (1, 2)]}
    selection = [{
        "point_id": "p0", "time_index": 0, "history_path": str(history),
        "history_index": 0, "historical_val_acc": 0.8,
        "z_arch": [0.0] * 12, "raw_hp_vector": [0.5] * 4,
        "z_search": [0.0] * 12 + [0.5] * 4,
        "historical_config": historical, "greedy_config": greedy,
        "historical_fingerprint": architecture_fingerprint(historical),
        "greedy_fingerprint": architecture_fingerprint(greedy),
        "historical_mu_arch": [0.0] * 12, "greedy_mu_arch": [0.1] * 12,
        "cycle_distance_l2_historical": 0.0,
        "cycle_distance_normalized_historical": 0.0,
        "cycle_distance_l2_greedy": 0.1,
        "cycle_distance_normalized_greedy": 0.1,
    }]
    monkeypatch.setattr(relabel, "load_joint_vae", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(relabel, "build_selection", lambda *_args, **_kwargs: selection)
    data = SimpleNamespace(to=lambda _device: object())
    monkeypatch.setattr(
        relabel, "load_dataset_from_request",
        lambda *_args: SimpleNamespace(data=data, num_features=8, num_classes=3),
    )
    monkeypatch.setattr(relabel, "resolve_dataset_request", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        relabel, "eval_z_search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic retry")),
    )
    output = tmp_path / "out"
    assert relabel.main([
        "--history", str(history), "--checkpoint", str(tmp_path / "checkpoint.pth"),
        "--output", str(output), "--seeds", "0", "--sample_count", "1",
        "--max_retries", "2", "--retry_all_failures",
    ]) == 0
    rows = [json.loads(line) for line in (output / relabel.RESULTS_JSONL).read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["attempt_count"] == 3 for row in rows)
    assert all(row["retry_count"] == 2 for row in rows)
    assert all(len(row["attempt_failures"]) == 3 for row in rows)


def test_representation_comparison_writes_json_csv_and_markdown(tmp_path):
    rows = []
    for policy, version in (
        ("historical_modal3", "historical"),
        ("deterministic_greedy", "greedy"),
    ):
        for point in range(12):
            operations = ["GCNConv", "GATConv"]
            optional_edges = [
                edge
                for bit, edge in enumerate(((0, 2), (0, 3), (1, 3), (1, 2)))
                if point & (1 << bit)
            ]
            edges = [(0, 1), (2, 3), *optional_edges]
            fingerprint = architecture_fingerprint(
                {"operations": operations, "edges": edges}
            )
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "point_id": f"p{point}",
                        "decoder_policy": policy,
                        "architecture_version": version,
                        "architecture_fingerprint": fingerprint,
                        "operations": operations,
                        "edges": edges,
                        "z_arch": [point / 12.0] * 12,
                        "mu_arch": [point / 13.0] * 12,
                        "raw_hp_vector": [0.2, 0.4, 0.6, 0.8],
                        "condition_mask_vector": [1.0] * 4,
                        "time_index": point,
                        "seed": seed,
                        "val_acc": 0.7 + 0.005 * point + 0.0005 * seed,
                        "valid": True,
                    }
                )
    source = tmp_path / "matched.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "comparison"
    assert compare.main(
        [
            "--input", str(source),
            "--output", str(output),
            "--repeated_splits", "2",
            "--bootstrap_replicates", "20",
            "--prefix_min_train", "4",
            "--prefix_step", "4",
            "--prefix_validation_window", "4",
        ]
    ) == 0
    assert (output / "representation_comparison.json").is_file()
    assert (output / "representation_metrics.csv").is_file()
    assert (output / "representation_predictions.csv").is_file()
    assert (output / "diagnostic_report.md").is_file()
    conclusion = json.loads(
        (output / "diagnostic_conclusion.json").read_text(encoding="utf-8")
    )
    assert conclusion["classification"] in {
        "NOISE_DOMINATED",
        "REPRESENTATION_CANDIDATE",
        "INSUFFICIENT_EVIDENCE",
    }
    metrics = json.loads(
        (output / "representation_comparison.json").read_text(encoding="utf-8")
    )["metrics"]
    assert {row["label_mode"] for row in metrics} == {"multiseed_mean"}
