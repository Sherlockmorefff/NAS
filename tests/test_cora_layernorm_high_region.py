from __future__ import annotations

import copy

import pytest

from analyse import cora_component_screen as b0
from analyse import cora_layernorm_high_region as high_region
from analyse import cora_layernorm_subgroup as subgroup


def _candidate(index):
    return {
        "candidate_id": f"H{index}",
        "group": "high_region",
        "candidate_fingerprint": f"candidate-{index}",
        "architecture_fingerprint": f"architecture-{index}",
        "historical_validation": 0.82,
        "selection_rank": index,
    }


def _matrix(deltas, normalized_means=None, test_delta=0.0):
    candidates = [_candidate(index) for index in range(1, 6)]
    manifest = {"selected_candidates": candidates}
    normalized_means = normalized_means or [0.82] * 5
    rows = []
    for candidate, delta, normalized_mean in zip(candidates, deltas, normalized_means):
        baseline_mean = normalized_mean - delta
        for variant in high_region.VARIANTS:
            for replicate in range(3):
                seed = b0.matched_training_seed(candidate["candidate_fingerprint"], replicate)
                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "group": candidate["group"],
                        "candidate_fingerprint": candidate["candidate_fingerprint"],
                        "architecture_fingerprint": candidate["architecture_fingerprint"],
                        "variant": variant, "replicate_id": replicate,
                        "training_seed": seed,
                        "validation_accuracy": (
                            normalized_mean if variant == "layer_norm" else baseline_mean
                        ),
                        "test_accuracy": 0.80 + (
                            test_delta if variant == "layer_norm" else 0
                        ),
                        "best_epoch": 10, "runtime_seconds": 1.0,
                        "parameter_count": 110 if variant == "layer_norm" else 100,
                        "valid": True, "failure": None,
                    }
                )
    return manifest, rows


@pytest.mark.parametrize(
    ("deltas", "means", "expected"),
    [
        ([0.006] * 5, [0.829, 0.82, 0.82, 0.82, 0.82], "CEILING_BREAK"),
        ([0.008, 0.008, 0.008, -0.002, -0.002], [0.82] * 5, "HIGH_REGION_GAIN"),
        ([-0.002] * 5, [0.82] * 5, "LOW_REGION_ONLY"),
        ([0.006, 0.006, -0.002, -0.002, -0.002], [0.82] * 5, "INCONCLUSIVE"),
    ],
)
def test_validation_only_decision_classes(deltas, means, expected):
    manifest, raw = _matrix(deltas, means, test_delta=100.0)
    decision = high_region.analyze_validation(
        manifest, [subgroup._validation_view(row) for row in raw]
    )
    assert decision["verdict"] == expected


def test_ceiling_break_requires_strictly_more_than_0828():
    manifest, raw = _matrix([0.006] * 5, [0.828] * 5)
    decision = high_region.analyze_validation(
        manifest, [subgroup._validation_view(row) for row in raw]
    )
    assert decision["verdict"] == "HIGH_REGION_GAIN"
    assert decision["ceiling_breakers"] == []


def test_validation_decision_rejects_test_fields():
    manifest, raw = _matrix([0.006] * 5)
    clean = [subgroup._validation_view(row) for row in raw]
    dirty = copy.deepcopy(clean)
    dirty[0]["test_accuracy"] = 1.0
    with pytest.raises(high_region.HighRegionError, match="test field"):
        high_region.analyze_validation(manifest, dirty)


def test_seed_mismatch_is_rejected():
    manifest, raw = _matrix([0.006] * 5)
    clean = [subgroup._validation_view(row) for row in raw]
    clean[0]["training_seed"] += 1
    with pytest.raises(high_region.HighRegionError, match="seed mismatch"):
        high_region.analyze_validation(manifest, clean)


def test_test_attachment_cannot_change_verdict():
    manifest, raw = _matrix([0.006] * 5, test_delta=-1.0)
    decision = high_region.analyze_validation(
        manifest, [subgroup._validation_view(row) for row in raw]
    )
    result = high_region.attach_test_results(decision, raw)
    assert result["verdict"] == "HIGH_REGION_GAIN"
    assert result["test_did_not_change_verdict"] is True
