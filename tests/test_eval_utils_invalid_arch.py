from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import eval_utils
from evaluation_errors import InfrastructureEvaluationError


def _toy_data(in_ch: int = 8, out_ch: int = 3):
    return SimpleNamespace(
        x=torch.randn(6, in_ch),
        edge_index=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
        y=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long) % out_ch,
        train_mask=torch.tensor([True, True, True, False, False, False]),
        val_mask=torch.tensor([False, False, False, True, True, False]),
        test_mask=torch.tensor([False, False, False, False, False, True]),
    )


def test_dynamic_gnn_matches_intermediate_feature_dim_for_skip_edges():
    config = {
        "operations": ["GCNConv", "GCNConv"],
        "effective_layers": 2,
        # Node 2 is built from predecessor 0, so its GCNConv expects in_ch.
        # At runtime it aggregates node 0 (in_ch) and node 1 (hidden_dim), which
        # previously truncated to hidden_dim and crashed in the second GCNConv.
        "edges": [(0, 1), (0, 2), (1, 2), (2, 3)],
    }
    model = eval_utils.DynamicGNN(
        config,
        in_ch=8,
        out_ch=3,
        dropout=0.0,
        hidden_dim=4,
    )
    data = _toy_data(in_ch=8, out_ch=3)

    out = model(data.x, data.edge_index)

    assert out.shape == (data.x.shape[0], 3)
    assert torch.isfinite(out).all()
    assert model.layer_input_dims == [8, 8]


def test_match_feature_dim_preserves_dtype_device_and_grad():
    x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)

    padded = eval_utils._match_feature_dim(x, 5)
    truncated = eval_utils._match_feature_dim(padded, 3)

    assert padded.dtype == x.dtype
    assert padded.device == x.device
    assert truncated.shape == (4, 3)
    truncated.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


class _FailingDynamicGNN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (2708x512 and 1433x512)")


def test_train_and_eval_arch_runtime_error_returns_invalid(monkeypatch):
    monkeypatch.setattr(eval_utils, "DynamicGNN", _FailingDynamicGNN)

    with pytest.warns(RuntimeWarning, match=r"\[DynamicGNN eval failed\]"):
        result = eval_utils.train_and_eval_arch(
            {"operations": ["GCNConv"], "effective_layers": 1, "edges": [(0, 1), (1, 2)]},
            _toy_data(in_ch=8, out_ch=3),
            in_ch=8,
            out_ch=3,
            device=torch.device("cpu"),
            max_epochs=1,
            patience=1,
            track_test=False,
        )

    assert result == (0.0, False)


def test_train_and_eval_arch_runtime_error_returns_invalid_with_test(monkeypatch):
    monkeypatch.setattr(eval_utils, "DynamicGNN", _FailingDynamicGNN)

    with pytest.warns(RuntimeWarning, match=r"\[DynamicGNN eval failed\]"):
        result = eval_utils.train_and_eval_arch(
            {"operations": ["GCNConv"], "effective_layers": 1, "edges": [(0, 1), (1, 2)]},
            _toy_data(in_ch=8, out_ch=3),
            in_ch=8,
            out_ch=3,
            device=torch.device("cpu"),
            max_epochs=1,
            patience=1,
            track_test=True,
        )

    assert result == (0.0, False, 0.0)


class _OOMDynamicGNN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")


def test_cuda_oom_is_infrastructure_error_and_not_invalid(monkeypatch):
    monkeypatch.setattr(eval_utils, "DynamicGNN", _OOMDynamicGNN)

    with pytest.raises(
        InfrastructureEvaluationError, match="infrastructure failure"
    ) as caught:
        eval_utils.train_and_eval_arch(
            {
                "operations": ["GATConv"],
                "effective_layers": 1,
                "edges": [(0, 1), (1, 2)],
            },
            _toy_data(in_ch=8, out_ch=3),
            in_ch=8,
            out_ch=3,
            device=torch.device("cpu"),
            max_epochs=1,
            patience=1,
        )

    assert caught.value.context["architecture"]["operations"] == ["GATConv"]
    assert caught.value.context["stage"] == "eval"


@pytest.mark.parametrize(
    "operation",
    ["GCNConv", "GATConv", "SAGEConv", "GINConv", "GCNII", "Identity"],
)
def test_operation_families_support_dynamic_input_and_output_dims(operation):
    config = {
        "operations": [operation],
        "effective_layers": 0 if operation == "Identity" else 1,
        "edges": [(0, 1), (1, 2)],
    }
    model = eval_utils.DynamicGNN(
        config,
        in_ch=11,
        out_ch=5,
        dropout=0.0,
        hidden_dim=7,
        gat_heads=2,
    )
    data = _toy_data(in_ch=11, out_ch=5)

    output = model(data.x, data.edge_index)

    assert output.shape == (data.x.shape[0], 5)


def test_test_labels_do_not_affect_training_or_early_stopping():
    config = {
        "operations": ["GCNConv"],
        "effective_layers": 1,
        "edges": [(0, 1), (1, 2)],
    }
    first = _toy_data(in_ch=8, out_ch=3)
    second = _toy_data(in_ch=8, out_ch=3)
    second.x = first.x.clone()
    second.edge_index = first.edge_index.clone()
    second.y = first.y.clone()
    second.y[second.test_mask] = (second.y[second.test_mask] + 1) % 3

    first_result = eval_utils.train_and_eval_arch(
        config,
        first,
        in_ch=8,
        out_ch=3,
        device=torch.device("cpu"),
        max_epochs=3,
        patience=2,
        seed=123,
        track_test=False,
        return_metadata=True,
    )
    second_result = eval_utils.train_and_eval_arch(
        config,
        second,
        in_ch=8,
        out_ch=3,
        device=torch.device("cpu"),
        max_epochs=3,
        patience=2,
        seed=123,
        track_test=False,
        return_metadata=True,
    )

    assert first_result == second_result
