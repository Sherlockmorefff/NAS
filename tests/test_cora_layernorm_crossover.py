from __future__ import annotations

import copy

import pytest

from analyse import cora_component_screen as b0
from analyse import cora_layernorm_crossover as crossover


def _candidate(role, arch, hidden, lr=0.01, dropout=0.2, l2=0.001):
    return {
        "panel_role": role,
        "architecture_fingerprint": arch,
        "candidate_fingerprint": f"original-{role}",
        "operations": ["GCNConv"],
        "edges": [[0, 1], [1, 2]],
        "hidden_dim": hidden,
        "lr": lr,
        "dropout": dropout,
        "l2": l2,
    }


def test_crossover_candidate_keeps_architecture_and_swaps_only_requested_hp():
    median = _candidate("median_1", "arch-m", 256, 0.01, 0.3, 0.0001)
    top = _candidate("top_1", "arch-t", 512, 0.001, 0.1, 0.00001)
    crossed = crossover.crossover_candidate(median, top, "median_arch_top_hp")
    assert crossed["architecture_fingerprint"] == "arch-m"
    assert crossed["operations"] == median["operations"]
    assert crossed["edges"] == median["edges"]
    for field in crossover.HP_FIELDS:
        assert crossed[field] == top[field]
    assert crossed["architecture_source_role"] == "median_1"
    assert crossed["hp_source_role"] == "top_1"


def test_crossover_seed_excludes_variant():
    candidate = crossover.crossover_candidate(
        _candidate("median_1", "arch-m", 256),
        _candidate("top_1", "arch-t", 512),
        "median_arch_top_hp",
    )
    baseline_seed = b0.matched_training_seed(candidate["candidate_fingerprint"], 1)
    layernorm_seed = b0.matched_training_seed(candidate["candidate_fingerprint"], 1)
    assert baseline_seed == layernorm_seed


def _matrix(median_top_deltas, top_median_deltas, test_sign=1.0):
    candidates = [
        crossover.crossover_candidate(
            _candidate("median_1", "arch-m", 256),
            _candidate("top_1", "arch-t", 512),
            "median_arch_top_hp",
        ),
        crossover.crossover_candidate(
            _candidate("top_1", "arch-t", 512),
            _candidate("median_1", "arch-m", 256),
            "top_arch_median_hp",
        ),
    ]
    manifest = {
        "new_combinations": candidates,
        "existing_combinations": [{"combination": "mm"}, {"combination": "tt"}],
    }
    rows = []
    delta_map = {
        "median_arch_top_hp": median_top_deltas,
        "top_arch_median_hp": top_median_deltas,
    }
    for candidate in candidates:
        for variant in crossover.VARIANTS:
            for replicate in range(3):
                delta = delta_map[candidate["panel_role"]][replicate]
                seed = b0.matched_training_seed(candidate["candidate_fingerprint"], replicate)
                rows.append(
                    {
                        "candidate_fingerprint": candidate["candidate_fingerprint"],
                        "variant": variant,
                        "replicate_id": replicate,
                        "training_seed": seed,
                        "validation_accuracy": 0.75 + (delta if variant == "layer_norm" else 0),
                        "test_accuracy": 0.74 + (
                            test_sign * delta if variant == "layer_norm" else 0
                        ),
                        "best_epoch": 10,
                        "runtime_seconds": 1.0,
                        "parameter_count": 100 + (10 if variant == "layer_norm" else 0),
                        "valid": True,
                        "failure": None,
                    }
                )
    return manifest, rows


@pytest.mark.parametrize(
    ("median_top", "top_median", "expected"),
    [
        ([0.006] * 3, [-0.002] * 3, "ARCHITECTURE_EFFECT"),
        ([-0.002] * 3, [0.006] * 3, "HP_EFFECT"),
        ([-0.002] * 3, [-0.002] * 3, "JOINT_EFFECT"),
        ([0.006] * 3, [0.006] * 3, "BROADER_GAIN"),
        ([0.002] * 3, [-0.002] * 3, "INCONCLUSIVE"),
    ],
)
def test_validation_only_decision_classes(median_top, top_median, expected):
    manifest, rows = _matrix(median_top, top_median)
    assert crossover.analyze(manifest, rows)["verdict"] == expected
    _, opposite_test_rows = _matrix(median_top, top_median, test_sign=-100.0)
    assert crossover.analyze(manifest, opposite_test_rows)["verdict"] == expected


def test_seed_mismatch_is_rejected():
    manifest, rows = _matrix([0.006] * 3, [-0.002] * 3)
    rows = copy.deepcopy(rows)
    rows[0]["training_seed"] += 1
    with pytest.raises(crossover.CrossoverError, match="seed mismatch"):
        crossover.analyze(manifest, rows)


def test_incomplete_matrix_is_rejected():
    manifest, rows = _matrix([0.006] * 3, [-0.002] * 3)
    with pytest.raises(crossover.CrossoverError, match="matrix mismatch"):
        crossover.analyze(manifest, rows[:-1])
