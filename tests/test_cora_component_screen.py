from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import eval_utils
from analyse import cora_component_screen as screen


REPO_ROOT = Path(__file__).resolve().parents[1]


def _skip_config():
    return {
        "operations": ["GCNConv", "GCNConv"],
        "effective_layers": 2,
        "edges": [(0, 1), (0, 2), (1, 2), (2, 3)],
    }


@pytest.mark.parametrize("variant", eval_utils.COMPONENT_VARIANTS)
def test_component_variants_produce_finite_logits(variant):
    model = eval_utils.DynamicGNN(
        _skip_config(), in_ch=8, out_ch=3, dropout=0.0, hidden_dim=4,
        component_variant=variant,
    )
    x = torch.randn(6, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
    )

    logits = model(x, edge_index)

    assert logits.shape == (6, 3)
    assert torch.isfinite(logits).all()


def test_explicit_baseline_is_bitwise_identical_to_default():
    x = torch.randn(6, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
    )
    torch.manual_seed(9812)
    default = eval_utils.DynamicGNN(
        _skip_config(), in_ch=8, out_ch=3, dropout=0.0, hidden_dim=4
    )
    torch.manual_seed(9812)
    explicit = eval_utils.DynamicGNN(
        _skip_config(), in_ch=8, out_ch=3, dropout=0.0, hidden_dim=4,
        component_variant="baseline",
    )

    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name in default.state_dict():
        assert torch.equal(default.state_dict()[name], explicit.state_dict()[name])
    assert torch.equal(default(x, edge_index), explicit(x, edge_index))


def test_matched_training_seed_does_not_include_variant():
    fingerprint = "a" * 64
    seeds = {
        screen.matched_training_seed(fingerprint, replicate)
        for _variant in screen.VARIANTS
        for replicate in (1,)
    }
    assert len(seeds) == 1


def test_track_a_source_has_no_gnn_training_call():
    source = (REPO_ROOT / "analyse/cora_accuracy_stage_ab.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "train_and_eval_arch" not in calls | attributes


def test_truncation_stress_detects_mixed_width_skip_combine():
    row = {
        "operations": ["GCNConv", "GCNConv"],
        "edges": [(0, 1), (0, 2), (1, 2), (2, 3)],
        "hidden_dim": 4,
    }
    assert screen._static_truncation_stress(row, in_ch=8) == {
        "mismatch_events": 1,
        "static_columns_discarded": 4,
    }


def test_component_training_returns_trajectory_and_parameter_count():
    data = SimpleNamespace(
        x=torch.randn(6, 8),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
        ),
        y=torch.tensor([0, 1, 2, 0, 1, 2]),
        train_mask=torch.tensor([True, True, True, False, False, False]),
        val_mask=torch.tensor([False, False, False, True, True, False]),
        test_mask=torch.tensor([False, False, False, False, False, True]),
    )
    result = eval_utils.train_and_eval_arch(
        _skip_config(), data, in_ch=8, out_ch=3, device=torch.device("cpu"),
        max_epochs=1, patience=1, seed=123, track_test=True,
        return_metadata=True, return_reproducibility_metadata=True,
        component_variant="post_mlp1", return_trajectory=True,
    )

    metadata = result[-1]
    assert len(metadata["trajectory"]) == 1
    assert metadata["parameter_count"] > 0
    assert len(metadata["trajectory_hash"]) == 64
    assert len(metadata["model_final_state_hash"]) == 64
