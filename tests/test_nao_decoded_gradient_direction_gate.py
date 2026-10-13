from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from analyse import nao_decoded_gradient_direction_gate as gate


def _union_row(joint: str, arch: str, regime: str = "hp_high") -> dict:
    return {
        "joint_config_fingerprint": joint,
        "architecture_fingerprint": arch,
        "hp_fingerprint": f"hp-{joint}",
        "hp_regime": regime,
    }


def _stat(joint: str, mean: float) -> dict:
    return {"joint_config_fingerprint": joint, "all30_mean_validation": mean}


def _config(operation: str) -> dict:
    return {"operations": [operation], "edges": [[0, 1], [1, 2]], "effective_layers": 1}


def test_p_and_q_are_excluded() -> None:
    union = [_union_row("P", "p"), _union_row("Q", "q"), _union_row("A", "a")]
    stats = [_stat("P", .9), _stat("Q", .8), _stat("A", .7)]
    selected = gate.select_anchors(union, stats, "P", "Q", target=1)
    assert [row["joint_config_fingerprint"] for row in selected] == ["A"]


def test_anchor_sort_and_architecture_dedup_are_deterministic() -> None:
    union = [
        _union_row("B", "same"), _union_row("A", "same"),
        _union_row("C", "c"), _union_row("D", "d", "hp_mid_high"),
    ]
    stats = [_stat("A", .8), _stat("B", .7), _stat("C", .75), _stat("D", 1.0)]
    first = gate.select_anchors(union, stats, "P", "Q", target=2)
    second = gate.select_anchors(list(reversed(union)), list(reversed(stats)), "P", "Q", target=2)
    assert [(row["joint_config_fingerprint"], row["anchor_index"]) for row in first] == [("A", 0), ("C", 1)]
    assert first == second


class ToyHead(torch.nn.Module):
    def __init__(self, weights: list[float]) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weights


def _member(member_id: str, arch_weight: float, y_scale: float = 1.0) -> gate.HeadMember:
    weights = [arch_weight] + [0.0] * 11 + [1.0] * 4
    return gate.HeadMember(
        member_id=member_id,
        head=ToyHead(weights),
        hp_mean=np.zeros(4, dtype=np.float32),
        hp_std=np.ones(4, dtype=np.float32),
        y_mean=0.0,
        y_std=y_scale,
    )


def test_gradient_only_uses_z_arch_and_hp_is_unchanged() -> None:
    z = np.zeros((1, 12), dtype=np.float32)
    hp = np.asarray([[.1, .2, .3, .4]], dtype=np.float32)
    before = hp.copy()
    _scores, gradient = gate.ensemble_scores_and_arch_gradient(z, hp, [_member("one", 2.0)])
    assert np.array_equal(hp, before)
    assert gradient.shape == (1, 12)
    assert gradient[0, 0] == 2.0
    assert np.all(gradient[0, 1:] == 0.0)


def test_ensemble_mean_gradient_is_mean_of_denormalized_members() -> None:
    z = np.zeros((1, 12), dtype=np.float32)
    hp = np.zeros((1, 4), dtype=np.float32)
    _scores, gradient = gate.ensemble_scores_and_arch_gradient(
        z, hp, [_member("one", 1.0, 2.0), _member("two", 3.0, 1.0)]
    )
    assert gradient[0, 0] == 2.5


def test_deterministic_decode_passes_stochastic_false(monkeypatch) -> None:
    calls = []

    class Decoder:
        def decode(self, tensor, stochastic):
            calls.append(stochastic)
            return [object()]

    model = SimpleNamespace(arch_vae=Decoder())
    monkeypatch.setattr(gate, "config_from_igraph", lambda _model, _graph: _config("GCNConv"))
    monkeypatch.setattr(gate, "normalize_architecture_config", lambda value: value)
    monkeypatch.setattr(gate.v1, "validate_static_config", lambda _value: (True, None))
    gate.deterministic_decode(model, np.zeros(12))
    assert calls == [False]


def test_first_legal_step_is_selected_without_validation_input() -> None:
    parent = _config("Identity")
    parent_fp = gate.architecture_fingerprint(parent)

    def decode(z):
        if abs(float(z[0])) < .2:
            return parent
        return _config("GCNConv" if z[0] > 0 else "GATConv")

    chosen, attempts = gate.choose_first_legal_step(
        np.zeros(12), np.eye(12)[0], [.1, .25, .5], parent_fp, set(), decode
    )
    assert chosen is not None and chosen["alpha"] == .25
    assert [row["accepted"] for row in attempts] == [False, True]


def test_duplicate_fingerprint_is_rejected() -> None:
    parent = _config("Identity")
    plus = _config("GCNConv")
    minus = _config("GATConv")

    def decode(z):
        return plus if z[0] > 0 else minus

    chosen, attempts = gate.choose_first_legal_step(
        np.zeros(12), np.eye(12)[0], [.25], gate.architecture_fingerprint(parent),
        {gate.architecture_fingerprint(plus)}, decode,
    )
    assert chosen is None
    assert "PLUS_DUPLICATES_EARLIER_PROPOSAL" in attempts[0]["failure_reasons"]


def test_gate_classification_logic() -> None:
    config = {
        "minimum_valid_anchors": 7,
        "minimum_plus_minus_wins": 7,
        "minimum_mean_delta_plus_minus": .003,
        "minimum_plus_parent_wins": 6,
        "minimum_mean_delta_plus_parent": .003,
    }
    assert gate.classify_gate(6, 8, .1, 8, .1, config) == "NAO_DIRECTION_GATE_DECODE_REJECTED"
    assert gate.classify_gate(8, 7, .003, 6, .003, config) == "NAO_DIRECTION_GATE_PASSED"
    assert gate.classify_gate(8, 6, .1, 8, .1, config) == "NAO_DIRECTION_GATE_REJECTED"


def test_validation_adapter_forces_track_test_false() -> None:
    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return .8, True, {
            "best_epoch": 3,
            "trajectory": [{
                "epoch_1based": 1,
                "loss": 1.0,
                "validation_accuracy": .8,
                "test_accuracy": None,
            }],
        }

    task = {
        "operations": ["GCNConv"], "edges": [[0, 1], [1, 2]], "effective_layers": 1,
        "lr": .01, "dropout": .5, "hidden_dim": 32, "l2": .0005,
        "gat_heads": 2, "sage_aggr": "mean", "gin_eps": 0.0,
        "gat_heads_by_layer": None, "sage_aggr_by_layer": None, "gin_eps_by_layer": None,
        "seed": 0,
    }
    bundle = SimpleNamespace(data=object(), num_features=16, num_classes=7)
    value, valid, metadata = gate.execute_validation_task(task, bundle, torch.device("cpu"), fake_train)
    assert value == .8 and valid and metadata["best_epoch"] == 3
    assert captured["track_test"] is False
    assert gate.strict_result_key_hits(metadata) == []


def test_resume_does_not_resubmit_completed_tasks() -> None:
    tasks = [{"task_key": "one"}, {"task_key": "two"}]
    assert gate.pending_tasks(tasks, {"one"}) == [{"task_key": "two"}]
