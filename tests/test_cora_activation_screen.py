from __future__ import annotations

import copy

import pytest
import torch

import eval_utils
from analyse import cora_activation_screen as activation


def _config():
    return {
        "operations": ["GINConv", "GCNII"],
        "effective_layers": 2,
        "edges": [(0, 1), (1, 2), (2, 3)],
    }


def test_activation_variants_keep_parameter_count_and_finite_logits():
    x = torch.randn(6, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
    )
    counts = set()
    for variant in eval_utils.ACTIVATION_VARIANTS:
        torch.manual_seed(81)
        model = eval_utils.DynamicGNN(
            _config(), in_ch=8, out_ch=3, hidden_dim=4, dropout=0.0,
            activation_variant=variant,
        )
        logits = model(x, edge_index)
        assert torch.isfinite(logits).all()
        counts.add(sum(parameter.numel() for parameter in model.parameters()))
    assert len(counts) == 1


def test_explicit_relu_is_bitwise_identical_to_default():
    x = torch.randn(6, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
    )
    torch.manual_seed(9812)
    default = eval_utils.DynamicGNN(_config(), 8, 3, hidden_dim=4, dropout=0.0)
    torch.manual_seed(9812)
    explicit = eval_utils.DynamicGNN(
        _config(), 8, 3, hidden_dim=4, dropout=0.0,
        activation_variant="relu",
    )
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name in default.state_dict():
        assert torch.equal(default.state_dict()[name], explicit.state_dict()[name])
    assert torch.equal(default(x, edge_index), explicit(x, edge_index))


def _rows(delta_by_arch, activation_name="leaky_relu"):
    baseline = []
    new = []
    panel = []
    for arch_index, (role, delta) in enumerate(delta_by_arch.items()):
        fingerprint = f"candidate-{arch_index}"
        panel.append({"panel_role": role, "candidate_fingerprint": fingerprint})
        for replicate in range(3):
            seed = 100 * arch_index + replicate
            base = {
                "panel_role": role, "candidate_fingerprint": fingerprint,
                "architecture_fingerprint": f"arch-{arch_index}",
                "activation": "relu", "replicate_id": replicate,
                "training_seed": seed, "validation_accuracy": 0.75,
                "test_accuracy": 0.74, "valid": True, "best_epoch": 20,
                "stopped_epoch": 100, "epochs_ran": 100,
                "runtime_seconds": 1.0, "parameter_count": 100,
                "trajectory_hash": "a", "model_final_state_hash": "b",
                "failure": None,
            }
            row = copy.deepcopy(base)
            row.update({
                "activation": activation_name,
                "validation_accuracy": 0.75 + delta,
                "test_accuracy": 0.74 + delta,
            })
            baseline.append(base)
            new.append(row)
    return {"candidate_panel": panel}, baseline, new


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ({"top_1": 0.006, "median_1": 0.006, "diverse": 0.006}, "UNIVERSAL_SIGNAL"),
        ({"top_1": 0.006, "median_1": -0.002, "diverse": -0.002}, "ARCHITECTURE_SPECIFIC"),
        ({"top_1": 0.002, "median_1": 0.002, "diverse": 0.002}, "BORDERLINE"),
        ({"top_1": -0.002, "median_1": -0.002, "diverse": -0.002}, "DROP"),
    ],
)
def test_decision_classes(deltas, expected):
    manifest, baseline, leaky = _rows(deltas)
    _manifest, _baseline, elu = _rows(
        {role: -0.002 for role in deltas}, activation_name="elu"
    )
    result = activation.analyze(manifest, baseline, [*leaky, *elu])
    assert result["activation_results"]["leaky_relu"]["classification"] == expected


def test_invalid_activation_is_rejected():
    with pytest.raises(ValueError, match="unsupported activation_variant"):
        eval_utils.DynamicGNN(
            _config(), 8, 3, hidden_dim=4, activation_variant="prelu"
        )
