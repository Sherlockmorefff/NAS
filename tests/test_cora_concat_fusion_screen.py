from __future__ import annotations

import copy

import pytest
import torch

import eval_utils
from analyse import cora_concat_fusion_screen as fusion


def _config(mixed: bool = True):
    edges = [(0, 1), (1, 2), (2, 3)]
    if mixed:
        edges.append((0, 2))
    return {
        "operations": ["GCNConv", "SAGEConv"],
        "effective_layers": 2,
        "edges": edges,
    }


def test_concat_linear_only_changes_mixed_width_fusion_nodes():
    mixed = eval_utils.DynamicGNN(
        _config(True), in_ch=8, out_ch=3, hidden_dim=4,
        component_variant="concat_linear",
    )
    equal = eval_utils.DynamicGNN(
        _config(False), in_ch=8, out_ch=3, hidden_dim=4,
        component_variant="concat_linear",
    )
    assert list(mixed.concat_projections) == ["n2"]
    assert list(equal.concat_projections) == []


def test_concat_linear_is_finite_and_adds_parameters():
    x = torch.randn(6, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
    )
    torch.manual_seed(17)
    baseline = eval_utils.DynamicGNN(_config(), 8, 3, hidden_dim=4, dropout=0.0)
    torch.manual_seed(17)
    concat = eval_utils.DynamicGNN(
        _config(), 8, 3, hidden_dim=4, dropout=0.0,
        component_variant="concat_linear",
    )
    assert torch.isfinite(concat(x, edge_index)).all()
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    concat_count = sum(parameter.numel() for parameter in concat.parameters())
    assert concat_count > baseline_count


def _screen_rows(deltas):
    manifest = {"candidate_panel": []}
    baseline, projected, concat = [], [], []
    for arch_index, (role, delta) in enumerate(deltas.items()):
        fingerprint = f"candidate-{arch_index}"
        manifest["candidate_panel"].append(
            {"panel_role": role, "candidate_fingerprint": fingerprint}
        )
        for replicate in range(3):
            common = {
                "panel_role": role, "candidate_fingerprint": fingerprint,
                "architecture_fingerprint": f"arch-{arch_index}",
                "replicate_id": replicate, "training_seed": arch_index * 10 + replicate,
                "valid": True, "best_epoch": 20, "stopped_epoch": 100,
                "epochs_ran": 100, "runtime_seconds": 1.0,
                "parameter_count": 100, "trajectory_hash": "a",
                "model_final_state_hash": "b", "failure": None,
            }
            base = {**common, "variant": "baseline", "validation_accuracy": 0.75, "test_accuracy": 0.74}
            proj = {**common, "variant": "projected_combine", "validation_accuracy": 0.752, "test_accuracy": 0.742, "parameter_count": 120}
            row = {**common, "variant": "concat_linear", "validation_accuracy": 0.75 + delta, "test_accuracy": 0.74 + delta, "parameter_count": 130}
            baseline.append(base); projected.append(proj); concat.append(row)
    return manifest, baseline, projected, concat


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ({"top_1": 0.006, "median_1": 0.006, "diverse": 0.006}, "UNIVERSAL_GAIN"),
        ({"top_1": -0.002, "median_1": 0.006, "diverse": -0.002}, "RESPONDER_GAIN"),
        ({"top_1": 0.002, "median_1": 0.002, "diverse": 0.002}, "INCONCLUSIVE"),
        ({"top_1": -0.002, "median_1": -0.002, "diverse": -0.002}, "NO_GAIN"),
    ],
)
def test_fusion_decision_classes(deltas, expected):
    manifest, baseline, projected, concat = _screen_rows(deltas)
    result = fusion.analyze(manifest, baseline, projected, concat)
    assert result["classification"] == expected


def test_concat_seed_mismatch_is_rejected():
    manifest, baseline, projected, concat = _screen_rows(
        {"top_1": 0.006, "median_1": 0.006, "diverse": 0.006}
    )
    concat[0] = copy.deepcopy(concat[0])
    concat[0]["training_seed"] += 1
    with pytest.raises(fusion.FusionScreenError, match="seed differs"):
        fusion.analyze(manifest, baseline, projected, concat)
