from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import analyse.h_vs_head_model_blind_prospective as pilot
import analyse.run_h_vs_head_prospective_labels as runner


def _config(fingerprint: str, structure: str, regime: str, hp_value: float) -> dict:
    edge_count = {
        "edges_sparse_le9": 7,
        "edges_medium_10_11": 10,
        "edges_dense_12_13": 12,
        "edges_very_dense_ge14": 14,
    }[structure]
    base_edges = [(0, 1), (1, 2), (2, 3), (5, 6)]
    possible = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    edges = list(dict.fromkeys(base_edges + possible))[:edge_count]
    raw = [hp_value] * 4
    return {
        "config_id": f"{fingerprint}-{regime}",
        "architecture_fingerprint": fingerprint,
        "hp_fingerprint": pilot.hp_fingerprint(raw),
        "operations": ["GCNConv"] * 5,
        "edges": [list(edge) for edge in edges],
        "effective_layers": 5,
        "structure_stratum": structure,
        "hp_regime": regime,
        "selection_stratum": f"{structure}__{regime}",
        "raw_hp_vector": raw,
        "lr": 0.01, "dropout": 0.2, "hidden_dim": 64, "l2": 0.0005,
        "gat_heads": 1, "sage_aggr": "mean", "gin_eps": 0.0,
        "gat_heads_by_layer": None, "sage_aggr_by_layer": ["mean"] * 5,
        "gin_eps_by_layer": None,
    }


def _synthetic_pool() -> list[dict]:
    values = {"hp_low": 0.1, "hp_mid_low": 0.35, "hp_mid_high": 0.65, "hp_high": 0.9}
    rows = []
    for structure_index, structure in enumerate(
        ("edges_sparse_le9", "edges_medium_10_11", "edges_dense_12_13", "edges_very_dense_ge14")
    ):
        for arch_index in range(20):
            fingerprint = f"s{structure_index}-a{arch_index:02d}"
            for regime, value in values.items():
                rows.append(_config(fingerprint, structure, regime, value))
    return rows


def test_actual_clean24_frozen_head_h_reproduction_is_exact():
    comparisons, audit = pilot.reproduce_clean_predictions()
    assert len(comparisons) == 48
    assert audit["passed"]
    assert audit["maximum_absolute_error"] <= 1e-8


def test_exposure_registry_includes_all_required_source_groups():
    fingerprints, audit, payload = pilot.build_exposure_registry()
    assert audit["passed"], audit["parse_failures"]
    assert set(audit["group_counts"]) == {
        "original_ceiling_hunt", "layernorm_search_space", "posthoc_truth_outputs", "legacy_histories",
    }
    assert audit["group_counts"]["original_ceiling_hunt"] >= 1199
    assert audit["group_counts"]["layernorm_search_space"] >= 1247
    assert len(fingerprints) == audit["union_count"]
    assert payload[0]["source_inventory"]


def test_model_blind_selection_produces_48_configs_and_40_architectures():
    selection, audit = pilot.select_candidates(_synthetic_pool())
    assert audit["passed"]
    assert len(selection) == 48
    assert len({row["architecture_fingerprint"] for row in selection}) == 40
    assert audit["two_hp_architecture_count"] == 8
    assert audit["maximum_hp_per_architecture"] == 2
    assert all(item["raw_hp_linf"] >= 1 / 3 for item in audit["hp_distance_audit"])
    assert all(item["raw_hp_l1"] >= 0.75 for item in audit["hp_distance_audit"])
    assert all(value == 24 for value in audit["gpu_configuration_counts"].values())


def test_insufficient_structure_stratum_uses_preregistered_refill():
    pool = _synthetic_pool()
    keep_very_dense = {
        f"s3-a{index:02d}" for index in range(5)
    }
    pool = [
        row for row in pool
        if row["structure_stratum"] != "edges_very_dense_ge14"
        or row["architecture_fingerprint"] in keep_very_dense
    ]
    selection, audit = pilot.select_candidates(pool)
    assert len(selection) == 48
    assert audit["allocation"]["edges_very_dense_ge14"]["selected_architectures"] == 5
    assert sum(item["refill_added"] for item in audit["allocation"].values()) == 5


def test_static_legality_and_structure_strata():
    config = _config("a", "edges_sparse_le9", "hp_low", 0.1)
    valid, reason = pilot.validate_static_config(config)
    assert valid and reason is None
    assert pilot.structure_stratum(config) == "edges_sparse_le9"
    invalid = {**config, "edges": [[1, 0]]}
    assert pilot.validate_static_config(invalid)[0] is False


def test_protocol_records_clean_seed_design_difference():
    clean = pilot.load_json(
        pilot.REPO_ROOT / pilot.CLEAN_RELATIVE / "phase_4/label_task_plan.json"
    )
    protocol = pilot.protocol_payload(clean)
    assert protocol["training_seeds"] == [0, 1, 2]
    assert protocol["planned_gnn_runs"] == 144
    assert protocol["clean24_protocol_difference"]["clean24_initial_seeds"] == list(range(15))
    assert protocol["track_test"] is False


def test_seed_level_bootstrap_keeps_two_hp_architecture_clustered():
    rows = []
    for architecture in ("a", "b", "c"):
        for hp_index in range(2 if architecture == "a" else 1):
            rows.append({
                "config_id": f"{architecture}-{hp_index}",
                "architecture_fingerprint": architecture,
                "hp_fingerprint": f"h{hp_index}",
                "structure_stratum": "edges_sparse_le9" if architecture != "c" else "edges_medium_10_11",
                "seed_val_accs": [0.7, 0.71, 0.72],
                "original_frozen_h": 0.8 - hp_index * 0.01,
                "original_frozen_head": 0.7 - hp_index * 0.01,
            })
    rng = pilot.np.random.default_rng(7)
    sampled = pilot._bootstrap_rows(rows, rng)
    counts = {}
    for row in sampled:
        root_id = row["config_id"].split("#bootstrap", 1)[0]
        counts[root_id] = counts.get(root_id, 0) + 1
    assert counts.get("a-0", 0) == counts.get("a-1", 0)
    assert all(len(row["seed_val_accs"]) == 3 for row in sampled)
    copied_fps = {row["architecture_fingerprint"] for row in sampled}
    assert all("#bootstrap" in fingerprint for fingerprint in copied_fps)
    stratum_counts = {}
    for fingerprint in copied_fps:
        source = fingerprint.split("#bootstrap", 1)[0]
        stratum = "edges_sparse_le9" if source != "c" else "edges_medium_10_11"
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
    assert stratum_counts == {"edges_sparse_le9": 2, "edges_medium_10_11": 1}
    bundle = pilot.metric_bundle(sampled, "original_frozen_h")
    assert bundle["candidate_count"] == len(sampled)
    intervals, paired, sensitivity = pilot.bootstrap_analysis(rows, replicates=2, seed=11)
    assert intervals and paired
    assert sensitivity["bootstrap_replicates"] == 2


def test_merge_allows_four_incomplete_configs_but_requires_every_task_key():
    rows = []
    expected = set()
    expected_architecture_strata = {
        f"a{architecture_index:02d}": "edges_sparse_le9"
        for architecture_index in range(40)
    }
    for config_index in range(48):
        config_id = f"c{config_index:02d}"
        fingerprint = f"a{config_index % 40:02d}"
        for seed in (0, 1, 2):
            expected.add((config_id, fingerprint, seed))
            valid = config_index < 44
            rows.append({
                "config_id": config_id,
                "architecture_fingerprint": fingerprint,
                "hp_fingerprint": f"h{config_index:02d}",
                "selection_stratum": "edges_sparse_le9__hp_low",
                "structure_stratum": "edges_sparse_le9",
                "seed": seed,
                "val_acc": 0.7 + seed * 0.01 if valid else None,
                "valid": valid,
                "terminal_status": "success" if valid else "failed",
            })
    statistics_rows, audit = pilot.aggregate_seed_results(
        rows, expected, expected_architecture_strata
    )
    assert audit["passed"]
    assert audit["valid_configuration_count"] == 44
    assert audit["valid_architecture_count"] == 40
    assert len(statistics_rows) == 44
    _, missing_audit = pilot.aggregate_seed_results(
        rows[:-1], expected, expected_architecture_strata
    )
    assert not missing_audit["passed"]
    assert missing_audit["missing_task_count"] == 1


def test_classification_priority_and_success_gate():
    h = {
        "kendall_tau_b": 0.3, "top_region_kendall_tau_b": 0.2, "ndcg_at_10": 0.9,
        "regret_at_10": 0.01, "false_positive_rate": 0.1,
    }
    head = {
        "kendall_tau_b": 0.2, "top_region_kendall_tau_b": 0.1, "ndcg_at_10": 0.8,
        "regret_at_10": 0.02, "false_positive_rate": 0.2,
    }
    intervals = {
        ("original_frozen_h", "kendall_tau_b"): {"ci95_low": 0.1, "ci95_high": 0.5},
        ("original_frozen_h", "top_region_kendall_tau_b"): {"ci95_low": 0.01, "ci95_high": 0.4},
    }
    paired = {"kendall_tau_b": {"difference_ci95_low": 0.02, "difference_ci95_high": 0.2}}
    classification, gates = pilot.classify(h, head, intervals, paired, 0.05)
    assert classification == "H_PROSPECTIVELY_VALIDATED_FOR_ACQUISITION_CALIBRATION"
    assert all(gates.values())


def test_runner_dry_run_does_not_load_dataset_or_train(monkeypatch, tmp_path, capsys):
    selection = []
    for gpu in (0, 1):
        for index in range(24):
            selection.append({
                **_config(f"g{gpu}-a{index}", "edges_sparse_le9", "hp_low", 0.1),
                "assigned_gpu": gpu,
            })
    selection_path = tmp_path / "selection.json"
    pilot.dump_json(selection_path, selection)
    plan = {
        "selection_path": str(selection_path), "training_seeds": [0, 1, 2],
        "shard_to_gpu_uuid": {"0": "unused", "1": "unused"},
    }
    monkeypatch.setattr(runner, "verify_plan", lambda path: plan)
    monkeypatch.setattr(runner, "load_dataset_from_request", lambda request: pytest.fail("dataset loaded"))
    monkeypatch.setattr(runner, "train_and_eval_arch", lambda **kwargs: pytest.fail("training started"))
    code = runner.main([
        "--plan", str(tmp_path / "plan.json"), "--shard", "0",
        "--output", str(tmp_path / "dry"), "--dry-run",
    ])
    assert code == 0
    payload = pilot.load_json(tmp_path / "dry/run_plan.json")
    assert payload["expected_task_count"] == 72
    assert payload["dataset_loaded"] is False
    assert payload["training_started"] is False


def test_selection_and_prepare_source_do_not_use_model_scores_or_labels():
    selection_source = inspect.getsource(pilot.select_candidates)
    assert '["prediction"]' not in selection_source
    assert '["val_acc"]' not in selection_source
    module_source = inspect.getsource(pilot)
    assert "GaussianProcessRegressor(" not in module_source
    assert ".fit(" not in module_source
    assert "train_and_eval_arch" not in module_source
    assert "import bo_phase4" not in module_source
