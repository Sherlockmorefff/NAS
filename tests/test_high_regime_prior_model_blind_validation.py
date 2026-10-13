from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from analyse import analyse_high_regime_prior_model_blind_validation as analysis
from analyse import high_regime_prior_model_blind_validation as plan
from analyse import run_high_regime_prior_model_blind_validation_labels as runner


def _fp(*parts: object) -> str:
    return hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()


def _candidate(pool: int, batch: int, slot: int, regime: str) -> dict:
    architecture = _fp("arch", pool, batch, slot)
    hp = _fp("hp", pool, batch, slot, regime)
    return {
        "pool_id": pool, "batch_id": batch, "architecture_slot": slot,
        "architecture_fingerprint": architecture, "operations": [0, 1, 2, 3],
        "edges": [[0, 1], [1, 2], [2, 3]], "effective_layers": 3,
        "config_id": f"c-{pool}-{batch}-{slot}-{regime}", "hp_regime": regime,
        "raw_hp_vector": [0.9, 0.9, 0.9, 0.9], "hp_fingerprint": hp,
        "joint_config_fingerprint": plan.joint_config_fingerprint(architecture, hp),
        "lr": 0.01, "dropout": 0.5, "hidden_dim": 64, "l2": 5e-4,
        "gat_heads": 1, "sage_aggr": "mean", "gin_eps": 0.0,
        "gat_heads_by_layer": None, "sage_aggr_by_layer": None,
        "gin_eps_by_layer": None,
    }


@pytest.fixture(scope="module")
def synthetic_freeze() -> tuple[list[dict], list[dict], list[dict], dict]:
    candidates = [
        _candidate(pool, batch, slot, regime)
        for pool in range(24) for batch in range(2) for slot in range(1, 5)
        for regime in plan.REGIME_ORDER
    ]
    selections, audit = plan.freeze_formal_strategies(candidates)
    union, tasks = plan.make_union_tasks(selections)
    return candidates, selections, union, {"tasks": tasks, "audit": audit}


def test_v2_head_hp_collapse_recomputes_expected_counts() -> None:
    result = plan.recompute_v2_hp_collapse()
    assert result["passed"]
    assert result["per_architecture_argmax_regime_counts"] == {"hp_high": 1535, "hp_mid_high": 1}
    assert result["b_slot_regime_counts"] == {"hp_high": 87, "hp_mid_high": 9}


def test_v2_input_manifest_and_classification_are_current() -> None:
    result = plan.verify_v2_evidence()
    assert result["passed"]
    assert result["classification"] == "HP_FILTERING_ONLY"
    assert result["pre_label_file_count"] == 28


def test_balanced_random_selector_uses_each_regime_once_and_is_deterministic() -> None:
    rows = [{
        "pool_id": 0, "batch_id": 0, "architecture_slot": slot,
        "architecture_fingerprint": _fp("a", slot),
        "regime_config_ids": {regime: f"{slot}-{regime}" for regime in plan.REGIME_ORDER},
    } for slot in range(1, 5)]
    first = plan.balanced_random_hp_assignment(rows, 0, 0)
    second = plan.balanced_random_hp_assignment(rows, 0, 0)
    assert first == second
    assert Counter(first.values()) == Counter(plan.REGIME_ORDER)


def test_balanced_random_narrow_schema_rejects_head_label_trajectory_and_mu() -> None:
    base = {
        "pool_id": 0, "batch_id": 0, "architecture_slot": 1,
        "architecture_fingerprint": _fp("a"),
        "regime_config_ids": {regime: regime for regime in plan.REGIME_ORDER},
    }
    for forbidden in ("head_prediction_mean", "head_prediction_std", "validation", "trajectory", "test_metric", "mu"):
        rows = [{**base, "architecture_slot": slot, "architecture_fingerprint": _fp(slot)} for slot in range(1, 5)]
        rows[0][forbidden] = 0.0
        with pytest.raises(ValueError):
            plan.balanced_random_hp_assignment(rows, 0, 0)


def test_f_and_r_freeze_exact_slots_shared_high_and_union(synthetic_freeze) -> None:
    _, selections, union, context = synthetic_freeze
    assert context["audit"]["passed"]
    assert len(selections) == 384
    for pool in range(24):
        for batch in range(2):
            rows = [row for row in selections if row["pool_id"] == pool and row["batch_id"] == batch]
            f = [row for row in rows if row["strategy"] == "HIGH_REGIME_PRIOR"]
            r = [row for row in rows if row["strategy"] == "BALANCED_RANDOM_HP"]
            assert [row["architecture_fingerprint"] for row in f] == [row["architecture_fingerprint"] for row in r]
            assert {row["hp_regime"] for row in f} == {"hp_high"}
            assert Counter(row["hp_regime"] for row in r) == Counter(plan.REGIME_ORDER)
            assert len({row["joint_config_fingerprint"] for row in rows}) == 7
            assert len({row["joint_config_fingerprint"] for row in f} & {row["joint_config_fingerprint"] for row in r}) == 1
    assert len(union) == 336


def test_exact_1008_tasks_three_seeds_and_pool_gpu_split(synthetic_freeze) -> None:
    _, selections, union, context = synthetic_freeze
    tasks = context["tasks"]
    assert len(tasks) == 1008
    assert Counter(task["assigned_gpu"] for task in tasks) == Counter({0: 504, 1: 504})
    for config in union:
        selected = [task for task in tasks if task["joint_config_fingerprint"] == config["joint_config_fingerprint"]]
        assert {task["seed"] for task in selected} == {0, 1, 2}
        assert len({task["assigned_gpu"] for task in selected}) == 1
    shared = Counter(row["joint_config_fingerprint"] for row in selections)
    for joint_fp, slot_count in shared.items():
        matching_tasks = [task for task in tasks if task["joint_config_fingerprint"] == joint_fp]
        assert len(matching_tasks) == 3
        assert slot_count in (1, 2)


def test_synthetic_architectures_are_unique_across_pool_and_batch(synthetic_freeze) -> None:
    candidates = synthetic_freeze[0]
    fingerprints = [row["architecture_fingerprint"] for row in candidates if row["hp_regime"] == "hp_high"]
    assert len(fingerprints) == 192
    assert len(set(fingerprints)) == 192


def test_trajectory_firewall_removes_null_test_and_rehashes() -> None:
    raw = {
        "trajectory": [{
            "epoch_1based": 1, "loss": 1.0, "validation_accuracy": 0.5,
            "test_accuracy": None,
        }],
        "trajectory_hash": "raw", "best_epoch": 1, "track_test": False,
    }
    cleaned = runner.sanitize_training_metadata(raw)
    assert not plan.strict_test_key_hits(cleaned)
    assert cleaned["trajectory"] == [{"epoch_1based": 1, "loss": 1.0, "validation_accuracy": 0.5}]
    assert cleaned["sanitized_trajectory_hash"] == plan.stable_hash(cleaned["trajectory"])


def test_resume_loader_rejects_duplicate_and_test_named_rows(tmp_path: Path) -> None:
    row = {
        "task_stage": "initial", "pool_id": 0, "joint_config_fingerprint": _fp("j"),
        "architecture_fingerprint": _fp("a"), "hp_fingerprint": _fp("h"), "seed": 0,
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        runner.load_jsonl(path)
    path.write_text(json.dumps({**row, "test_accuracy": None}) + "\n")
    with pytest.raises(ValueError, match="test-named"):
        runner.load_jsonl(path)


def _constant_values(f0: float, f1: float, r0: float, r1: float) -> np.ndarray:
    values = np.empty((24, 2, 2, 4, 3), dtype=np.float64)
    values[:, 0, 0, :, :] = f0
    values[:, 1, 0, :, :] = f1
    values[:, 0, 1, :, :] = r0
    values[:, 1, 1, :, :] = r1
    return values


def test_pool_metric_averages_two_batches_not_better_batch() -> None:
    values = _constant_values(0.9, 0.1, 0.4, 0.4)
    pools, _ = analysis.observed_pool_metrics(values)
    assert pools[0]["F_mean_batch_best4"] == pytest.approx(0.5)
    assert pools[0]["R_mean_batch_best4"] == pytest.approx(0.4)
    assert pools[0]["F_minus_R"] == pytest.approx(0.1)


def test_hierarchical_bootstrap_is_deterministic_and_shared_identity_is_zero() -> None:
    values = _constant_values(0.7, 0.7, 0.7, 0.7)
    first, summary1, _ = analysis.hierarchical_bootstrap(values, 128, 17, chunk_size=31)
    second, summary2, _ = analysis.hierarchical_bootstrap(values, 128, 17, chunk_size=31)
    assert np.array_equal(first, second)
    assert np.all(first == 0)
    assert summary1["combined_indices_sha256"] == summary2["combined_indices_sha256"]
    assert summary1["same_matched_batch_indices_F_R"]
    assert summary1["same_seed_indices_all_configs_and_strategies"]


def test_hierarchical_bootstrap_uses_same_batch_draw_for_f_and_r() -> None:
    # F and R are identical within a batch but the two batches are very different.
    # A shared J must therefore make every replicate difference exactly zero.
    values = _constant_values(0.9, 0.1, 0.9, 0.1)
    samples, _, _ = analysis.hierarchical_bootstrap(values, 256, 123)
    assert np.all(samples == 0)


def test_bootstrap_recomputes_best_after_seed_resampling() -> None:
    values = np.full((24, 2, 2, 4, 3), 0.4, dtype=np.float64)
    values[:, :, 0, 0, :] = [0.9, 0.1, 0.1]
    values[:, :, 0, 1, :] = [0.1, 0.9, 0.1]
    values[:, :, 1, 0, :] = [0.5, 0.5, 0.5]
    samples, _, _ = analysis.hierarchical_bootstrap(values, 256, 91)
    assert np.std(samples) > 0


def test_pool_only_bootstrap_reads_fixed_pool_values_only() -> None:
    rows = _pool_rows(0.01, 18)
    first, summary1 = analysis.pool_only_bootstrap(rows, 128, 44)
    second, summary2 = analysis.pool_only_bootstrap(rows, 128, 44)
    assert np.array_equal(first, second)
    assert summary1["indices_sha256"] == summary2["indices_sha256"]
    assert summary1["fixed_observed_three_seed_means"]
    assert summary1["fixed_two_batch_arithmetic_mean"]


def _pool_rows(delta: float, wins: int) -> list[dict]:
    return [{
        "pool_id": pool, "F_mean_batch_best4": 0.8 + (delta if pool < wins else -abs(delta)),
        "R_mean_batch_best4": 0.8,
        "F_minus_R": delta if pool < wins else -abs(delta),
    } for pool in range(24)]


def test_all_first_stage_classification_branches() -> None:
    confirmed, gate = analysis.classify(
        _pool_rows(0.01, 24), {"F_minus_R": {"ci95_lower": 0.002, "ci95_upper": 0.02}}, 192,
    )
    assert confirmed == "HIGH_REGIME_PRIOR_CONFIRMED_HEAD_REDUNDANT"
    assert gate["all_primary_gates_passed"]
    not_confirmed, _ = analysis.classify(
        _pool_rows(0.01, 0), {"F_minus_R": {"ci95_lower": -0.02, "ci95_upper": 0.004}}, 192,
    )
    assert not_confirmed == "HIGH_REGIME_PRIOR_NOT_CONFIRMED"
    inconclusive, _ = analysis.classify(
        _pool_rows(0.01, 14), {"F_minus_R": {"ci95_lower": -0.001, "ci95_upper": 0.02}}, 192,
    )
    assert inconclusive == "HIGH_REGIME_PRIOR_INCONCLUSIVE"


def test_stage_validator_uses_configuration_not_seed_as_unit() -> None:
    tasks = []
    results = []
    for seed in (0, 1, 2):
        common = {
            "task_stage": "initial", "pool_id": 0, "batch_id": 0,
            "architecture_slot": 1, "config_id": "c", "joint_config_fingerprint": _fp("j"),
            "architecture_fingerprint": _fp("a"), "hp_fingerprint": _fp("h"),
            "seed": seed, "assigned_gpu": 0, "task_key": f"k{seed}",
        }
        tasks.append(common)
        trajectory = [{"epoch_1based": 1, "loss": 1.0, "validation_accuracy": 0.5}]
        results.append({
            **common, "dataset_content_fingerprint": "d", "split_fingerprint": "s",
            "hp_mode": "global4", "normalization": "none", "eval_epochs": 300,
            "patience": 80, "actual_gpu": {"physical_index": 0},
            "trajectory": trajectory, "sanitized_trajectory_hash": plan.stable_hash(trajectory),
            "terminal_status": "success", "valid": True, "retry_count": 0,
        })
    audit = analysis.validate_stage(tasks, results, (0, 1, 2), "initial")
    assert audit["passed"]
    assert audit["configuration_count"] == 1


def test_top8_freeze_uses_initial_means_stable_tie_and_deduplicates_union(
    synthetic_freeze, tmp_path: Path,
) -> None:
    _, selections, union, _ = synthetic_freeze
    plan.dump_json(tmp_path / "selected_union_configs.json", union)
    required = [
        "phase_b/initial/gpu0/seed_results.jsonl", "phase_b/initial/gpu1/seed_results.jsonl",
        "phase_b/initial/seed_results_merged.jsonl", "analysis/initial_configuration_validation.json",
        "analysis/pool_level_primary_metrics.json", "analysis/hierarchical_bootstrap_summary.json",
        "analysis/final_gate_evidence.json", "first_stage_classification.json",
    ]
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]\n" if path.suffix in {".json", ".jsonl"} else "\n")
    config_rows = [{
        "pool_id": row["pool_id"], "batch_id": row["batch_id"],
        "architecture_slot": row["architecture_slot"],
        "joint_config_fingerprint": row["joint_config_fingerprint"],
        "architecture_fingerprint": row["architecture_fingerprint"],
        "hp_fingerprint": row["hp_fingerprint"], "hp_regime": row["hp_regime"],
        "mean_validation": 0.75,
    } for row in union]
    audit = analysis._freeze_top8(tmp_path, config_rows, selections)
    top8 = plan.load_json(tmp_path / "stabilization/frozen_top8_selections.json")
    tasks = plan.load_json(tmp_path / "stabilization/stabilization_tasks.json")
    assert audit["passed"]
    assert len(top8) == 16
    for strategy in ("HIGH_REGIME_PRIOR", "BALANCED_RANDOM_HP"):
        actual = [row["joint_config_fingerprint"] for row in top8 if row["strategy"] == strategy]
        expected = sorted({
            row["joint_config_fingerprint"] for row in selections if row["strategy"] == strategy
        })[:8]
        assert actual == expected
        assert all(row["selection_seed_ids"] == [0, 1, 2] for row in top8 if row["strategy"] == strategy)
    union_count = len({row["joint_config_fingerprint"] for row in top8})
    assert len(tasks) == union_count * 27
    assert {task["seed"] for task in tasks} == set(range(3, 30))
    assert len({task["task_key"] for task in tasks}) == len(tasks)


def test_all30_final_candidate_rule_and_holdout27_are_separate() -> None:
    top8 = []
    initial = []
    additional = []
    for strategy_index, strategy in enumerate(("HIGH_REGIME_PRIOR", "BALANCED_RANDOM_HP")):
        for rank in range(1, 9):
            fp = f"{strategy_index}-{rank:02d}"
            top8.append({
                "strategy": strategy, "initial_rank": rank,
                "joint_config_fingerprint": fp, "pool_id": rank - 1,
                "batch_id": 0, "architecture_slot": 1,
                "architecture_fingerprint": f"a-{fp}", "hp_fingerprint": f"h-{fp}",
                "hp_regime": "hp_high", "initial_three_seed_mean_validation": 0.7,
                "selection_seed_ids": [0, 1, 2], "tie_break": "joint_config_fingerprint",
            })
            # Ranks 1 and 2 tie on all-30 mean; fingerprint must break the tie.
            value = 0.82 if rank in (1, 2) else 0.80 - rank / 1000
            for seed in range(3):
                initial.append({"joint_config_fingerprint": fp, "seed": seed, "val_acc": value})
            for seed in range(3, 30):
                additional.append({"joint_config_fingerprint": fp, "seed": seed, "val_acc": value})
    rows, summary = analysis._strategy_top8_statistics(top8, initial, additional)
    for strategy in ("HIGH_REGIME_PRIOR", "BALANCED_RANDOM_HP"):
        assert summary[strategy]["final_candidate_joint_config_fingerprint"].endswith("-01")
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        assert all("holdout27_mean_validation_sensitivity" in row for row in strategy_rows)
        assert all(row["holdout27_not_used_to_rewrite_first_stage_classification"] for row in strategy_rows)


def test_source_dependency_manifest_includes_all_four_new_entry_files() -> None:
    manifest = plan.source_dependency_manifest()
    assert set(plan.EXPECTED_NEW_FILES) == set(manifest["entry_files"])
