from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import eval_utils


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
