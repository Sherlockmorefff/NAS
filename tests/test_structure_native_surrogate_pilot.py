from __future__ import annotations

from collections import Counter
import inspect

import numpy as np

from analyse.evaluate_structure_native_surrogate_pilot import (
    configuration_statistics,
    resolve_classification,
)
from analyse.run_structure_native_surrogate_labels import active_configs_and_seeds
from analyse.structure_native_surrogate_pilot import (
    MODEL_NAMES,
    _raw_feature_matrix,
    architecture_balanced_weights,
    build_fixed_splits,
    inner_split,
    interval_score,
    metric_bundle,
    project_unbiased_history_candidate,
    select_inner_ensemble_weights,
    select_unbiased_candidates,
    stratified_bootstrap_samples,
)


def _row(index: int, fingerprint: str | None = None) -> dict:
    fp = fingerprint or f"fp_{index:03d}"
    return {
        "config_id": f"config_{index:03d}",
        "architecture_fingerprint": fp,
        "mu_arch": [float(index)] * 12,
        "structure_vector": [float((index + bit) % 2) for bit in range(55)],
        "raw_hp_vector": [0.1, 0.2, 0.3, 0.4],
        "validation_mean": 0.7 + index * 0.001,
        "validation_ci95_low": 0.695 + index * 0.001,
        "validation_ci95_high": 0.705 + index * 0.001,
        "validation_sample_std": 0.01,
        "development_source": "synthetic",
    }


def test_group_splits_and_inner_splits_have_zero_architecture_leakage():
    rows = [_row(index) for index in range(20)]
    rows.append({**_row(100, "fp_000"), "config_id": "extra_hp"})
    split = build_fixed_splits(rows)[0]
    train = {row["architecture_fingerprint"] for row in rows if row["config_id"] in split["train_config_ids"]}
    validation = {row["architecture_fingerprint"] for row in rows if row["config_id"] in split["validation_config_ids"]}
    assert not train & validation
    train_rows = [row for row in rows if row["config_id"] in split["train_config_ids"]]
    fit, stop, audit = inner_split(train_rows, split_id=split["split_id"], training_seed=0)
    assert not {row["architecture_fingerprint"] for row in fit} & {row["architecture_fingerprint"] for row in stop}
    assert audit["architecture_leakage_count"] == 0


def test_architecture_balanced_weights_equalize_multi_hp_architectures():
    rows = [_row(0, "a"), {**_row(1, "a"), "config_id": "a2"}, _row(2, "b")]
    weights = architecture_balanced_weights(rows, uncertainty_weighted=False)
    assert np.isclose(weights[:2].sum(), weights[2])
    assert np.isclose(weights.sum(), 1.0)


def test_d_excludes_mu_and_h_contains_mu_structure_and_hp():
    row = _row(1)
    changed = {**row, "mu_arch": [99.0] * 12}
    d = _raw_feature_matrix([row], "d_structure")
    d_changed = _raw_feature_matrix([changed], "d_structure")
    h = _raw_feature_matrix([row], "h_structure")
    h_changed = _raw_feature_matrix([changed], "h_structure")
    assert d.shape[1] == 59
    assert h.shape[1] == 71
    assert np.array_equal(d, d_changed)
    assert not np.array_equal(h, h_changed)


def test_new_pilot_contains_no_latent_distance_gp_implementation():
    import analyse.structure_native_surrogate_pilot as module

    source = inspect.getsource(module)
    assert "_fixed_gp_predict" not in source
    assert "GaussianProcess" not in source
    assert "latent_distance_gp_allowed\": False" in source


def test_ensemble_weight_selection_uses_only_inner_config_ids():
    development = [_row(index) for index in range(4)]
    task_results = []
    for model, predictions in (("refit_head", [0.70, 0.71]), ("h_structure", [0.71, 0.72]), ("d_structure", [0.69, 0.70])):
        task_results.append(
            {
                "split_id": "group_000",
                "training_seed": 0,
                "weight_mode": "architecture_balanced",
                "model": model,
                "inner_validation_config_ids": ["config_000", "config_001"],
                "inner_validation_predictions": predictions,
            }
        )
    first = select_inner_ensemble_weights(task_results, development)
    development[2]["validation_mean"] = -100.0
    development[3]["validation_mean"] = 100.0
    second = select_inner_ensemble_weights(task_results, development)
    assert first == second
    assert first[0]["outer_validation_labels_used"] is False


def test_ensemble_weight_selection_aggregates_multiple_hp_by_architecture():
    development = [
        _row(0, "shared"),
        {**_row(1, "shared"), "config_id": "config_shared_2"},
        _row(2, "other"),
    ]
    config_ids = ["config_000", "config_shared_2", "config_002"]
    task_results = []
    for model, predictions in (
        ("refit_head", [0.70, 0.90, 0.72]),
        ("h_structure", [0.71, 0.69, 0.73]),
        ("d_structure", [0.70, 0.70, 0.70]),
    ):
        task_results.append(
            {
                "split_id": "group_000",
                "training_seed": 0,
                "weight_mode": "architecture_balanced",
                "model": model,
                "inner_validation_config_ids": config_ids,
                "inner_validation_predictions": predictions,
            }
        )
    selected = select_inner_ensemble_weights(task_results, development)[0]
    assert selected["inner_architecture_count"] == 2
    assert selected["inner_configuration_count"] == 3
    assert selected["inner_metric_unit"] == "architecture_fingerprint_equal_weight"


def test_candidate_projection_masks_labels_predictions_and_values(tmp_path):
    raw = {
        "operations": ["GCNConv"],
        "edges": [[0, 1], [1, 2]],
        "z_search": [0.0] * 12 + [0.1, 0.2, 0.3, 0.4],
        "step": 4,
        "val_acc": 0.99,
        "value": 9.0,
        "gp_prediction": 0.8,
        "dkl_residual": 1.0,
    }
    path = tmp_path / "search_seed2" / "history_final.json"
    path.parent.mkdir()
    projected, masked = project_unbiased_history_candidate(raw, source_path=path, source_index=0)
    assert {"val_acc", "value", "gp_prediction", "dkl_residual"} <= set(masked)
    assert not any(key in projected for key in ("val_acc", "value", "gp_prediction", "dkl_residual", "z_search"))
    assert projected["raw_hp_vector"] == [0.1, 0.2, 0.3, 0.4]


def test_unbiased_selection_is_unique_disjoint_12_12_and_gpu_complete():
    pool = []
    index = 0
    for seed in range(5):
        for quartile in range(4):
            for duplicate in range(2):
                pool.append(
                    {
                        "architecture_fingerprint": f"fp_{index:03d}",
                        "hp_fingerprint": f"hp_{index:03d}",
                        "operations": ["GCNConv"],
                        "edges": [[0, 1], [1, 2]],
                        "raw_hp_vector": [0.1, 0.2, 0.3, 0.4],
                        "source_search_seed": seed,
                        "source_step": index,
                        "source_index": index,
                        "source_path": "history.json",
                        "time_quartile": quartile,
                        "structure_vector": [float((index >> bit) & 1) for bit in range(55)],
                    }
                )
                index += 1
    selected, audit = select_unbiased_candidates(pool)
    assert len(selected) == len({row["architecture_fingerprint"] for row in selected}) == 24
    assert Counter(row["selection_stratum"] for row in selected) == Counter({"hash_uniform": 12, "structure_diverse": 12})
    assert Counter(row["assigned_gpu"] for row in selected) == Counter({0: 12, 1: 12})
    assert audit["selection_used_label_values"] is False
    assert audit["selection_used_model_predictions"] is False


def test_adaptive_seed_rule_uses_only_validation_interval():
    selection = [{"config_id": "a", "architecture_fingerprint": "fa", "selection_stratum": "hash_uniform", "assigned_gpu": 0}]
    rows = [
        {"config_id": "a", "architecture_fingerprint": "fa", "seed": seed, "val_acc": 0.7 + (0.1 if seed % 2 else 0), "valid": True, "terminal_status": "success", "retry_count": 0}
        for seed in range(15)
    ]
    stats = configuration_statistics(rows, selection)
    assert stats[0]["seed_count"] == 15
    assert stats[0]["validation_ci95_half_width"] > 0.006


def test_stratified_architecture_bootstrap_preserves_12_12():
    rows = [
        {"architecture_fingerprint": f"h{index}", "selection_stratum": "hash_uniform"}
        for index in range(12)
    ] + [
        {"architecture_fingerprint": f"d{index}", "selection_stratum": "structure_diverse"}
        for index in range(12)
    ]
    rows.append({"architecture_fingerprint": "h0", "selection_stratum": "hash_uniform"})
    sample = stratified_bootstrap_samples(rows, replicates=2, seed=7)[0]
    original = {row["architecture_fingerprint"]: row["selection_stratum"] for row in rows}
    counts = Counter(original[fingerprint] for fingerprint, _draw in sample)
    assert counts == Counter({"hash_uniform": 12, "structure_diverse": 12})
    assert len(sample) == 24


def test_regret_interval_width_and_interval_score_metrics():
    rows = []
    for index, prediction in enumerate((0.1, 0.4, 0.3, 0.2)):
        rows.append(
            {
                "config_id": f"c{index}",
                "architecture_fingerprint": f"f{index}",
                "selection_stratum": "all",
                "validation_mean": 0.1 * (index + 1),
                "validation_ci95_low": 0.1 * (index + 1) - 0.01,
                "validation_ci95_high": 0.1 * (index + 1) + 0.01,
                "prediction": prediction,
                "prediction_interval_low": prediction - 0.05,
                "prediction_interval_high": prediction + 0.05,
            }
        )
    metrics = metric_bundle(rows, interval_alpha=0.10)
    assert set(f"regret_at_{k}" for k in (1, 3, 5, 10)) <= set(metrics)
    assert np.isclose(metrics["prediction_interval_mean_width"], 0.1)
    assert metrics["prediction_interval_score"] >= metrics["prediction_interval_mean_width"]
    assert np.isclose(interval_score(0.5, 0.4, 0.6, 0.1), 0.2)


def _classification_evidence():
    summaries = {}
    for model in MODEL_NAMES:
        summaries[model] = {
            "model": model,
            "kendall": 0.2,
            "kendall_ci95_low": 0.05,
            "kendall_ci95_high": 0.4,
            "reliable_pair_count": 30,
            "reliable_architecture_count": 20,
            "regret_at_1": 0.0,
            "regret_at_3": 0.0,
            "head_top_quartile_false_positive_count": 3,
        }
    gates = {model: {"passed": False} for model in MODEL_NAMES}
    differences = {}
    for candidate in MODEL_NAMES:
        for baseline in ("refit_head", "d_structure", "h_structure"):
            differences[(candidate, baseline)] = {"estimate": 0.0, "ci95_low": -0.1, "ci95_high": 0.1}
    return summaries, gates, differences


def test_all_seven_final_classification_branches():
    summaries, gates, differences = _classification_evidence()
    gates["d_structure"]["passed"] = True
    assert resolve_classification([], summaries, gates, differences)[0] == "D_VALIDATED_FOR_OFFLINE_REPLAY"

    summaries, gates, differences = _classification_evidence()
    gates["h_structure"]["passed"] = True
    differences[("h_structure", "d_structure")] = {"estimate": 0.1, "ci95_low": 0.01, "ci95_high": 0.2}
    assert resolve_classification([], summaries, gates, differences)[0] == "H_VALIDATED_FOR_OFFLINE_REPLAY"

    summaries, gates, differences = _classification_evidence()
    gates["head_h_mean"]["passed"] = True
    differences[("head_h_mean", "d_structure")] = {"estimate": 0.1, "ci95_low": 0.01, "ci95_high": 0.2}
    assert resolve_classification([], summaries, gates, differences)[0] == "HEAD_H_ENSEMBLE_VALIDATED_FOR_OFFLINE_REPLAY"

    summaries, gates, differences = _classification_evidence()
    summaries["head_h_conservative"]["head_top_quartile_false_positive_count"] = 1
    assert resolve_classification([], summaries, gates, differences)[0] == "H_FILTER_SIGNAL_ONLY"

    summaries, gates, differences = _classification_evidence()
    summaries["refit_head"]["reliable_pair_count"] = 10
    summaries["head_h_conservative"]["head_top_quartile_false_positive_count"] = 3
    assert resolve_classification([], summaries, gates, differences)[0] == "SURROGATE_PILOT_UNDERPOWERED"

    summaries, gates, differences = _classification_evidence()
    assert resolve_classification([], summaries, gates, differences)[0] == "NO_STRUCTURE_SURROGATE_VALIDATED"
    assert resolve_classification(["bad protocol"], summaries, gates, differences)[0] == "PROTOCOL_OR_DATA_ERROR"


def test_dual_gpu_shards_are_disjoint_and_complete():
    selection = [
        {"config_id": f"c{index}", "architecture_fingerprint": f"f{index}", "assigned_gpu": index % 2}
        for index in range(24)
    ]
    plan = {"initial_seeds": list(range(15)), "adaptive_seeds": list(range(15, 30))}
    left, left_seeds = active_configs_and_seeds(plan, selection, shard=0, stage="initial", adaptive_path=None)
    right, right_seeds = active_configs_and_seeds(plan, selection, shard=1, stage="initial", adaptive_path=None)
    assert len(left) == len(right) == 12
    assert not {row["config_id"] for row in left} & {row["config_id"] for row in right}
    assert {row["config_id"] for row in left + right} == {row["config_id"] for row in selection}
    assert left_seeds == right_seeds == list(range(15))
