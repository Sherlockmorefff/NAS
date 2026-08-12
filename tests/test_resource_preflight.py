from __future__ import annotations

import json

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("botorch")

from evaluation_errors import InfrastructureEvaluationError
import resource_preflight


def _candidate(
    index,
    operation,
    *,
    hidden=64,
    layers=1,
    heads=1,
    parameters=100,
    edge_bytes=1000,
):
    return {
        "candidate_pool_index": index,
        "candidate_fingerprint": f"{index:064x}",
        "z_search": [0.0] * 16,
        "architecture": {
            "operations": [operation] * layers,
            "edges": [[0, 1], [1, 2]],
            "effective_layers": layers,
        },
        "hp": {
            "hidden_dim": hidden,
            "gat_heads": heads,
        },
        "parameter_count": parameters,
        "attention_heads": heads,
        "edge_activation_estimate_bytes": edge_bytes,
    }


def test_static_preflight_selection_covers_worst_metrics_and_families():
    candidates = [
        _candidate(0, "GCNConv", hidden=256, parameters=200),
        _candidate(1, "GATConv", heads=8, edge_bytes=9000),
        _candidate(2, "SAGEConv", layers=5, parameters=300),
        _candidate(3, "GINConv", parameters=500),
        _candidate(4, "GCNII", edge_bytes=12000),
    ]

    selected = resource_preflight.select_static_worst_cases(candidates)
    reasons = "+".join(row["selection_reason"] for row in selected)

    assert "max_hidden_dim" in reasons
    assert "max_effective_layers" in reasons
    assert "max_attention_heads" in reasons
    assert "max_parameter_count" in reasons
    assert "max_edge_activation_estimate" in reasons
    assert "max_activation_resource_estimate" in reasons
    assert "gat_attention_family" in reasons
    for operation in resource_preflight.OPERATION_FAMILIES:
        assert f"operation_family_{operation}" in reasons


def test_preflight_without_cuda_writes_blocked_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = resource_preflight.parse_args(
        [
            "--dataset",
            "flickr",
            "--data_root",
            str(tmp_path / "data"),
            "--checkpoint",
            str(tmp_path / "joint.pt"),
            "--output",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(InfrastructureEvaluationError, match="CUDA is unavailable"):
        resource_preflight.run_preflight(args)

    payload = json.loads(
        (tmp_path / "out" / "resource_preflight.json").read_text()
    )
    assert payload["status"] == "blocked_cuda_unavailable"
    assert payload["all_worst_cases_passed"] is False
    assert payload["training_mode"] == "full_batch"


def test_activation_resource_score_includes_hidden_layers_and_gat_heads():
    base = resource_preflight.estimate_activation_resources(
        num_nodes=10,
        num_edges=100,
        architecture={
            "operations": ["GCNConv", "Identity"],
            "effective_layers": 1,
        },
        hp={"hidden_dim": 16, "gat_heads": 1},
    )
    wider = resource_preflight.estimate_activation_resources(
        num_nodes=10,
        num_edges=100,
        architecture={
            "operations": ["GCNConv", "GCNConv"],
            "effective_layers": 2,
        },
        hp={"hidden_dim": 32, "gat_heads": 1},
    )
    gat = resource_preflight.estimate_activation_resources(
        num_nodes=10,
        num_edges=100,
        architecture={
            "operations": ["GATConv", "GATConv"],
            "effective_layers": 2,
        },
        hp={"hidden_dim": 32, "gat_heads": 8},
    )

    assert wider["edge_activation_estimate_bytes"] == 4 * base[
        "edge_activation_estimate_bytes"
    ]
    assert gat["gat_layer_count"] == 2
    assert gat["edge_activation_estimate_bytes"] == 8 * wider[
        "edge_activation_estimate_bytes"
    ]
    assert gat["activation_resource_estimate_bytes"] > gat[
        "edge_activation_estimate_bytes"
    ]
