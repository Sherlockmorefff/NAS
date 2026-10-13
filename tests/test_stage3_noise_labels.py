from __future__ import annotations

import json

import pytest

from architecture_diagnostics import architecture_fingerprint
from analyse import stage3_noise_labels as stage3


def _selection():
    rows = []
    optional = ((0, 2), (0, 3), (1, 3), (1, 2), (2, 4))
    for index in range(18):
        structure_index = 7 if index == 8 else index
        historical = {
            "operations": ["GCNConv", "GATConv", "SAGEConv"],
            "edges": [(0, 1), (3, 4), *[edge for bit, edge in enumerate(optional) if structure_index & (1 << bit)]],
        }
        greedy = {
            "operations": ["GATConv", "GCNConv", "SAGEConv"],
            "edges": historical["edges"],
        }
        rows.append(
            {
                "point_id": f"p{index:02d}",
                "selection_region": "top20" if index < 9 else "middle",
                "historical_config": historical,
                "greedy_config": greedy,
                "historical_fingerprint": architecture_fingerprint(historical),
                "greedy_fingerprint": architecture_fingerprint(greedy),
            }
        )
    return rows


def _result_rows(selection):
    output = []
    for point_index, point in enumerate(selection):
        for policy in stage3.POLICIES:
            version = stage3.VERSIONS[policy]
            config = point[f"{version}_config"]
            for seed in stage3.ALL_SEEDS:
                output.append(
                    {
                        "point_id": point["point_id"],
                        "decoder_policy": policy,
                        "architecture_version": version,
                        "architecture_fingerprint": point[f"{version}_fingerprint"],
                        "seed": seed,
                        "operations": config["operations"],
                        "edges": config["edges"],
                        "raw_hp_vector": [0.2, 0.4, 0.6, 0.8],
                        "realized_hp": {"lr": 0.01, "dropout": 0.3},
                        "condition_mask_vector": [1.0, 1.0, 1.0, 1.0],
                        "historical_val_acc": 0.75 + point_index * 0.002,
                        "val_acc": 0.75 + point_index * 0.002 + seed * 0.0001,
                        "valid": True,
                        "terminal_status": "success",
                        "retry_count": 0,
                    }
                )
    return output


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_exact_merge_has_540_unique_keys_and_checks_seed_invariants(tmp_path):
    selection = _selection()
    rows = _result_rows(selection)
    sources = []
    for name, seeds in (("stage2", {0, 1, 2}), ("odd", set(stage3.GPU0_SEEDS)), ("even", set(stage3.GPU1_SEEDS))):
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(path, [row for row in rows if row["seed"] in seeds])
        sources.append(path)
    merged, audit = stage3.exact_merge(selection, sources)
    assert len(merged) == 540
    assert audit["passed"] is True
    assert audit["seed_invariant_group_field_checks"] == 18 * 2 * len(stage3.INVARIANT_FIELDS)


def test_exact_merge_rejects_duplicate_or_seed_variant_rows(tmp_path):
    selection = _selection()
    rows = _result_rows(selection)
    complete = tmp_path / "complete.jsonl"
    _write_jsonl(complete, rows)
    duplicate = tmp_path / "duplicate.jsonl"
    _write_jsonl(duplicate, [rows[0]])
    with pytest.raises(ValueError, match="duplicates=1"):
        stage3.exact_merge(selection, [complete, duplicate])

    changed = [dict(row) for row in rows]
    changed[1] = {**changed[1], "raw_hp_vector": [0.9] * 4}
    variant = tmp_path / "variant.jsonl"
    _write_jsonl(variant, changed)
    with pytest.raises(ValueError, match="seed invariant raw_hp_vector"):
        stage3.exact_merge(selection, [variant])


def test_noise_gate_uses_nine_historical_top20_point_hp_intervals():
    stats = []
    for index in range(18):
        for policy in stage3.POLICIES:
            stats.append(
                {
                    "point_id": f"p{index:02d}",
                    "decoder_policy": policy,
                    "selection_region": "top20" if index < 9 else "middle",
                    "successful_seed_count": 15,
                    "terminal_invalid": 0,
                    "terminal_failed": 0,
                    "t_ci95_half_width": 0.005 if index < 9 else 0.02,
                }
            )
    gate = stage3.noise_gate(stats)
    assert gate["passed"] is True
    assert gate["point_count"] == 9
    stats[0]["t_ci95_half_width"] = 0.02
    assert stage3.noise_gate(stats)["fraction_half_width_le_0_010"] == pytest.approx(8 / 9)


def test_anomaly_classifier_distinguishes_collapse_bimodality_and_stability():
    collapsed = stage3.classify_anomaly([0.81] * 14 + [0.3])
    bimodal = stage3.classify_anomaly([0.7] * 7 + [0.82] * 8)
    stable = stage3.classify_anomaly([0.80 + index * 0.0001 for index in range(15)])
    assert collapsed["collapse"] is True
    assert bimodal["bimodality"] is True
    assert stable["classification"] == "stable"


def test_grouped_odd_even_ordering_deduplicates_architecture_fingerprint():
    stats = []
    for policy in stage3.POLICIES:
        for index in range(6):
            stats.append(
                {
                    "decoder_policy": policy,
                    "point_id": f"p{index:02d}",
                    "time_index": index,
                    "selection_region": "top20" if index < 3 else "middle",
                    "architecture_fingerprint": f"arch-{index // 2}",
                    "odd_seed_mean": 0.7 + index * 0.01,
                    "even_seed_mean": 0.7 + index * 0.01,
                }
            )
    ordering = stage3.odd_even_ordering(stats)
    all_rows = [row for row in ordering if row["selection_region"] == "all"]
    assert {row["unique_architecture_count"] for row in all_rows} == {3}
    assert all(row["estimate"] == pytest.approx(1.0) for row in ordering)


def test_vae_readiness_is_independent_of_representation_classification():
    assert stage3.compute_vae_finetune_ready(
        noise_gate_passed=True,
        independent_architecture_count=60,
        minimum_independent_architectures=60,
        group_leakage_count=0,
        test_label_used=False,
        ordering_gate_passed=True,
    ) is True


@pytest.mark.parametrize(
    "override",
    [
        {"noise_gate_passed": False},
        {"independent_architecture_count": 59},
        {"group_leakage_count": 1},
        {"test_label_used": True},
        {"ordering_gate_passed": False},
    ],
)
def test_vae_readiness_rejects_each_failed_admission_gate(override):
    arguments = {
        "noise_gate_passed": True,
        "independent_architecture_count": 60,
        "minimum_independent_architectures": 60,
        "group_leakage_count": 0,
        "test_label_used": False,
        "ordering_gate_passed": True,
    }
    arguments.update(override)
    assert stage3.compute_vae_finetune_ready(**arguments) is False


def test_policy_difference_inference_uses_point_means_not_seed_rows():
    rows = []
    for point in ("p0", "p1"):
        for seed in range(3):
            for policy, value in (
                ("historical_modal3", 0.7),
                ("deterministic_greedy", 0.8 if point == "p0" else 0.6),
            ):
                rows.append(
                    {
                        "point_id": point,
                        "decoder_policy": policy,
                        "seed": seed,
                        "val_acc": value,
                        "valid": True,
                        "terminal_status": "success",
                    }
                )
    _paired, summary = stage3.paired_policy_statistics(rows)
    assert summary["inference_unit"] == "point_mean_of_seed_paired_differences"
    assert summary["paired_point_count"] == 2
    assert summary["paired_seed_row_count"] == 6
    assert summary["mean_greedy_minus_historical"] == pytest.approx(0.0)
