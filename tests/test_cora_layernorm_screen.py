from __future__ import annotations

import copy

import pytest
import torch

import eval_utils
from analyse import cora_layernorm_screen as layernorm


def _config():
    return {
        "operations": ["Identity", "GCNII", "SAGEConv"],
        "effective_layers": 2,
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
    }


def test_layernorm_is_only_added_after_message_passing_ops():
    model = eval_utils.DynamicGNN(
        _config(), in_ch=8, out_ch=3, hidden_dim=4,
        component_variant="layer_norm",
    )
    assert list(model.layer_norms) == ["1", "2"]
    assert all(tuple(norm.normalized_shape) == (4,) for norm in model.layer_norms.values())


def test_layernorm_is_finite_and_has_expected_parameter_cost():
    x = torch.randn(6, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
    )
    torch.manual_seed(19)
    baseline = eval_utils.DynamicGNN(_config(), 8, 3, hidden_dim=4, dropout=0.0)
    torch.manual_seed(19)
    normalized = eval_utils.DynamicGNN(
        _config(), 8, 3, hidden_dim=4, dropout=0.0,
        component_variant="layer_norm",
    )
    assert torch.isfinite(normalized(x, edge_index)).all()
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    normalized_count = sum(parameter.numel() for parameter in normalized.parameters())
    assert normalized_count - baseline_count == 2 * 2 * 4


def _screen_rows(deltas):
    manifest = {"candidate_panel": []}
    baseline, normalized = [], []
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
            row = {**common, "variant": "layer_norm", "validation_accuracy": 0.75 + delta, "test_accuracy": 0.74 + delta, "parameter_count": 110}
            baseline.append(base); normalized.append(row)
    return manifest, baseline, normalized


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ({"top_1": 0.006, "median_1": 0.006, "diverse": 0.006}, "UNIVERSAL_GAIN"),
        ({"top_1": -0.002, "median_1": 0.006, "diverse": -0.002}, "ARCHITECTURE_SPECIFIC"),
        ({"top_1": 0.002, "median_1": 0.002, "diverse": 0.002}, "INCONCLUSIVE"),
        ({"top_1": -0.002, "median_1": -0.002, "diverse": -0.002}, "NO_GAIN"),
    ],
)
def test_layernorm_decision_classes(deltas, expected):
    manifest, baseline, normalized = _screen_rows(deltas)
    result = layernorm.analyze(manifest, baseline, normalized)
    assert result["classification"] == expected


def test_layernorm_seed_mismatch_is_rejected():
    manifest, baseline, normalized = _screen_rows(
        {"top_1": 0.006, "median_1": 0.006, "diverse": 0.006}
    )
    normalized[0] = copy.deepcopy(normalized[0])
    normalized[0]["training_seed"] += 1
    with pytest.raises(layernorm.LayerNormScreenError, match="seed differs"):
        layernorm.analyze(manifest, baseline, normalized)
