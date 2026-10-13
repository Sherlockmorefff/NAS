from __future__ import annotations

import copy

import pytest

from analyse import cora_component_screen as b0
from analyse import cora_layernorm_subgroup as subgroup


def _candidate(candidate_id, group, fingerprint):
    return {
        "candidate_id": candidate_id,
        "group": group,
        "candidate_fingerprint": fingerprint,
        "architecture_fingerprint": f"arch-{fingerprint}",
    }


def _matrix(similar_deltas, control_deltas, test_delta=0.0):
    candidates = [
        *[_candidate(f"S{i+1}", "similar", f"s{i+1}") for i in range(3)],
        *[_candidate(f"C{i+1}", "control", f"c{i+1}") for i in range(3)],
    ]
    manifest = {
        "selected_candidates": candidates,
        "selection_audit": {
            "pairs": [
                {
                    "similar_candidate_id": f"S{i+1}",
                    "control_candidate_id": f"C{i+1}",
                    "standardized_hp_distance": 0.1,
                }
                for i in range(3)
            ]
        },
    }
    delta_map = {
        **{f"S{i+1}": delta for i, delta in enumerate(similar_deltas)},
        **{f"C{i+1}": delta for i, delta in enumerate(control_deltas)},
    }
    rows = []
    for candidate in candidates:
        for variant in subgroup.VARIANTS:
            for replicate in range(3):
                seed = b0.matched_training_seed(candidate["candidate_fingerprint"], replicate)
                delta = delta_map[candidate["candidate_id"]]
                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "group": candidate["group"],
                        "candidate_fingerprint": candidate["candidate_fingerprint"],
                        "architecture_fingerprint": candidate["architecture_fingerprint"],
                        "variant": variant,
                        "replicate_id": replicate,
                        "training_seed": seed,
                        "validation_accuracy": 0.75 + (
                            delta if variant == "layer_norm" else 0
                        ),
                        "test_accuracy": 0.74 + (
                            test_delta if variant == "layer_norm" else 0
                        ),
                        "best_epoch": 10,
                        "runtime_seconds": 1.0,
                        "parameter_count": 110 if variant == "layer_norm" else 100,
                        "valid": True,
                        "failure": None,
                    }
                )
    return manifest, rows


@pytest.mark.parametrize(
    ("similar", "control", "expected"),
    [
        ([0.006, 0.006, -0.002], [0.006, -0.002, -0.002], "ARCHITECTURE_SUBGROUP_CONFIRMED"),
        ([0.006, 0.006, -0.002], [0.006, 0.006, -0.002], "BROADER_LAYERNORM_EFFECT"),
        ([0.006, -0.002, -0.002], [0.006, -0.002, -0.002], "HP_OR_MIXED_EFFECT"),
        ([-0.002, -0.002, -0.002], [0.006, -0.002, -0.002], "NO_GENERALIZATION"),
    ],
)
def test_group_decisions_use_validation_only(similar, control, expected):
    manifest, raw = _matrix(similar, control, test_delta=100.0)
    decision = subgroup.analyze_validation(
        manifest, [subgroup._validation_view(row) for row in raw]
    )
    assert decision["verdict"] == expected


def test_architecture_responder_requires_mean_and_seed_consistency():
    manifest, raw = _matrix(
        [0.006, 0.003, -0.002], [-0.002, -0.002, -0.002]
    )
    decision = subgroup.analyze_validation(
        manifest, [subgroup._validation_view(row) for row in raw]
    )
    assert decision["by_candidate"]["S1"]["responder"] is True
    assert decision["by_candidate"]["S2"]["responder"] is False


def test_validation_view_and_decision_reject_test_fields():
    manifest, raw = _matrix([0.006] * 3, [-0.002] * 3)
    clean = [subgroup._validation_view(row) for row in raw]
    assert all(all("test" not in key for key in row) for row in clean)
    dirty = copy.deepcopy(clean)
    dirty[0]["test_accuracy"] = 1.0
    with pytest.raises(subgroup.SubgroupError, match="test field"):
        subgroup.analyze_validation(manifest, dirty)


def test_seed_mismatch_is_rejected():
    manifest, raw = _matrix([0.006] * 3, [-0.002] * 3)
    clean = [subgroup._validation_view(row) for row in raw]
    clean[0]["training_seed"] += 1
    with pytest.raises(subgroup.SubgroupError, match="seed mismatch"):
        subgroup.analyze_validation(manifest, clean)


def test_test_attachment_cannot_change_verdict():
    manifest, raw = _matrix([0.006] * 3, [-0.002] * 3, test_delta=-1.0)
    decision = subgroup.analyze_validation(
        manifest, [subgroup._validation_view(row) for row in raw]
    )
    result = subgroup.attach_test_results(decision, raw)
    assert result["verdict"] == "ARCHITECTURE_SUBGROUP_CONFIRMED"
    assert result["test_did_not_change_verdict"] is True
