from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analyse import original_head_confirmatory_attribution_v2 as freeze
from analyse.analyse_original_head_confirmatory_attribution_v2 import (
    classify,
    hierarchical_bootstrap,
    observed_pool_metrics,
    validate_complete_seed_results,
)
from analyse.run_original_head_confirmatory_attribution_v2_labels import (
    ProtocolFirewallViolation,
    load_jsonl,
    sanitize_training_metadata,
)


def _fp(*parts: object) -> str:
    return freeze.stable_hash(list(parts))


def candidate(pool: int, architecture: int, regime_index: int) -> dict:
    architecture_fp = _fp("architecture", pool, architecture)
    raw = [0.1 + 0.2 * regime_index, 0.12 + 0.2 * regime_index,
           0.14 + 0.2 * regime_index, 0.16 + 0.2 * regime_index]
    hp_fp = freeze.v1.hp_fingerprint(raw)
    row = {
        "pool_id": pool,
        "config_id": f"p{pool}_a{architecture}_h{regime_index}",
        "architecture_fingerprint": architecture_fp,
        "hp_fingerprint": hp_fp,
        "joint_config_fingerprint": freeze.joint_config_fingerprint(architecture_fp, hp_fp),
        "hp_regime": tuple(freeze.HP_REGIMES)[regime_index],
        "raw_hp_vector": raw,
        "operations": ["GCNConv", "GATConv"],
        "edges": [[0, 1], [1, 2], [2, 3]],
        "effective_layers": 2,
    }
    row.update(freeze.v1.realize_global4_hp(row))
    return row


@pytest.fixture(scope="module")
def frozen_synthetic() -> tuple[list[dict], dict[str, float], list[dict], list[dict], dict]:
    candidates = [
        candidate(pool, architecture, regime)
        for pool in range(freeze.POOL_COUNT)
        for architecture in range(freeze.ARCHITECTURES_PER_POOL)
        for regime in range(4)
    ]
    means = {}
    for row in candidates:
        architecture = int(row["config_id"].split("_a")[1].split("_")[0])
        regime = int(row["config_id"].rsplit("h", 1)[1])
        # All four top slots deliberately come from architecture zero.
        means[row["config_id"]] = 10.0 - architecture - regime * 0.01
    selections, mappings, audit = freeze.freeze_selections(candidates, means)
    return candidates, means, selections, mappings, audit


def test_all_24_pools_and_global_head_top4_are_frozen(frozen_synthetic) -> None:
    candidates, means, selections, _, audit = frozen_synthetic
    assert audit["passed"]
    assert len(candidates) == 24 * 64 * 4
    assert len(selections) == 24 * 20
    for pool in range(24):
        pool_candidates = [row for row in candidates if row["pool_id"] == pool]
        expected = [
            row["config_id"] for row in sorted(
                pool_candidates, key=lambda row: (-means[row["config_id"]], row["config_id"]),
            )[:4]
        ]
        actual = [
            row["config_id"] for row in sorted(
                (row for row in selections if row["pool_id"] == pool
                 and row["strategy"] == "HEAD_GLOBAL_TOP4"),
                key=lambda row: row["budget_order"],
            )
        ]
        assert actual == expected


def test_matched_batches_preserve_B_hp_repeat_pattern_and_share_C_architecture(
    frozen_synthetic,
) -> None:
    _, _, selections, mappings, audit = frozen_synthetic
    assert all(row["b_repeat_architecture_pattern"] == [4] for row in audit["pool_audits"])
    for pool in range(24):
        b_by_slot = {
            int(row["source_b_slot"]): row for row in selections
            if row["pool_id"] == pool and row["strategy"] == "HEAD_GLOBAL_TOP4"
        }
        replacement_sets = []
        for batch in (0, 1):
            m_by_slot = {
                int(row["source_b_slot"]): row for row in selections
                if row["pool_id"] == pool and row["strategy"] == "MATCHED_HP_RANDOM_ARCH"
                and row["random_batch"] == batch
            }
            c_by_slot = {
                int(row["source_b_slot"]): row for row in selections
                if row["pool_id"] == pool and row["strategy"] == "BLIND_JOINT_RANDOM"
                and row["random_batch"] == batch
            }
            assert set(m_by_slot) == set(c_by_slot) == set(b_by_slot) == {1, 2, 3, 4}
            assert len({row["architecture_fingerprint"] for row in m_by_slot.values()}) == 1
            for slot in b_by_slot:
                assert m_by_slot[slot]["raw_hp_vector"] == b_by_slot[slot]["raw_hp_vector"]
                assert m_by_slot[slot]["architecture_fingerprint"] == c_by_slot[slot]["architecture_fingerprint"]
            replacement_sets.append({row["architecture_fingerprint"] for row in m_by_slot.values()})
        assert replacement_sets[0].isdisjoint(replacement_sets[1])
    assert all(row["source_b_slots"] == [1, 2, 3, 4] for row in mappings)


def test_narrow_random_selectors_reject_head_and_label_fields() -> None:
    slots = [{"source_b_slot": 1, "b_architecture_group": "g0", "raw_hp_vector": [0.1] * 4}]
    architecture = freeze.narrow_architecture(candidate(0, 1, 0))
    with pytest.raises(ValueError, match="schema mismatch"):
        freeze.select_replacement_architectures(
            slots, [dict(architecture, head_prediction_mean=0.9)], 0, 0,
        )
    hp_rows = [{key: candidate(0, 1, regime)[key] for key in freeze.CANDIDATE_HP_SCHEMA}
               for regime in range(4)]
    replacement = {"g0": architecture}
    with pytest.raises(ValueError, match="schema mismatch"):
        freeze.select_blind_hp_configs(
            slots, replacement, [dict(row, val_acc=0.9) for row in hp_rows], 0, 0,
        )
    selected = freeze.select_blind_hp_configs(slots, replacement, hp_rows, 0, 0)
    assert selected[0][0] == 1
    assert selected[0][1]["architecture_fingerprint"] == architecture["architecture_fingerprint"]


def test_union_deduplicates_overlap_and_keeps_three_seeds_on_one_gpu(frozen_synthetic) -> None:
    _, _, selections, _, _ = frozen_synthetic
    # Force one M slot to be byte-identical to B to exercise cross-strategy sharing.
    modified = [dict(row) for row in selections]
    b = next(row for row in modified if row["pool_id"] == 0 and row["strategy"] == "HEAD_GLOBAL_TOP4")
    m = next(row for row in modified if row["pool_id"] == 0 and row["strategy"] == "MATCHED_HP_RANDOM_ARCH")
    for key in freeze.CANDIDATE_HP_SCHEMA:
        m[key] = b[key]
    union, tasks = freeze.make_union_tasks(modified)
    assert len(tasks) == len(union) * 3
    assert len({row["task_key"] for row in tasks}) == len(tasks)
    for config in union:
        matching = [row for row in tasks if row["pool_id"] == config["pool_id"]
                    and row["joint_config_fingerprint"] == config["joint_config_fingerprint"]]
        assert {row["seed"] for row in matching} == {0, 1, 2}
        assert len({row["assigned_gpu"] for row in matching}) == 1


def test_full_v1_512_pool_is_explicitly_added_to_exposure(monkeypatch, tmp_path: Path) -> None:
    v1_root = tmp_path / freeze.V1_RELATIVE
    v1_root.mkdir(parents=True)
    rows = [{"architecture_fingerprint": _fp("v1", index)} for index in range(512)]
    (v1_root / "frozen_candidate_pools.json").write_text(json.dumps(rows), encoding="utf-8")
    for name in ("exposure_registry.json", "frozen_strategy_selections.json", "selected_union_configs.json"):
        (v1_root / name).write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(freeze.v1, "exposure_sources", lambda: [])
    registry, _, metadata = freeze.build_exposure_registry(tmp_path / "new_output")
    assert len(registry) == 512
    assert metadata["audit"]["v1_candidate_fingerprints_covered"] == 512


def _raw_metadata(value: float | None = None) -> dict:
    return {
        "trajectory": [
            {"epoch_1based": 1, "loss": 1.0, "validation_accuracy": 0.5,
             "test_accuracy": value},
            {"epoch_1based": 2, "loss": 0.8, "validation_accuracy": 0.6,
             "test_accuracy": value},
        ],
        "trajectory_hash": "raw-hash-must-not-survive",
        "track_test": False,
        "epochs_ran": 2,
    }


def test_trajectory_firewall_drops_null_test_fields_and_rehashes() -> None:
    cleaned = sanitize_training_metadata(_raw_metadata())
    assert not freeze.strict_test_key_hits(cleaned)
    assert "trajectory_hash" not in cleaned
    assert cleaned["sanitized_trajectory_hash"] == freeze.stable_hash(cleaned["trajectory"])
    assert set(cleaned["trajectory"][0]) == {"epoch_1based", "loss", "validation_accuracy"}
    with pytest.raises(ProtocolFirewallViolation):
        sanitize_training_metadata(_raw_metadata(0.9))


def _complete_tasks_and_results() -> tuple[list[dict], list[dict]]:
    tasks, results = [], []
    for seed in (0, 1, 2):
        base = {
            "pool_id": 0,
            "config_id": "canonical",
            "joint_config_fingerprint": "j" * 64,
            "architecture_fingerprint": "a" * 64,
            "hp_fingerprint": "b" * 64,
            "seed": seed,
            "assigned_gpu": 0,
            "task_key": f"task-{seed}",
        }
        tasks.append(dict(base))
        trajectory = [{"epoch_1based": 1, "loss": 1.0, "validation_accuracy": 0.7}]
        results.append({
            **base,
            "dataset_content_fingerprint": "data",
            "split_fingerprint": "split",
            "hp_mode": "global4",
            "normalization": "none",
            "eval_epochs": 300,
            "patience": 80,
            "terminal_status": "success",
            "valid": True,
            "val_acc": 0.7 + seed * 0.01,
            "retry_count": 0,
            "actual_gpu": {"physical_index": 0, "uuid": "gpu0"},
            "trajectory": trajectory,
            "sanitized_trajectory_hash": freeze.stable_hash(trajectory),
            "epochs_ran": 1,
        })
    return tasks, results


def test_three_seed_completeness_uses_configuration_not_seed_as_unit() -> None:
    tasks, results = _complete_tasks_and_results()
    assert validate_complete_seed_results(tasks, results)["passed"]
    assert not validate_complete_seed_results(tasks, results[:-1])["passed"]
    contaminated = [dict(results[0], test_accuracy=0.9), *results[1:]]
    assert not validate_complete_seed_results(tasks, contaminated)["passed"]


def test_resume_loader_rejects_duplicate_and_test_named_results(tmp_path: Path) -> None:
    _, results = _complete_tasks_and_results()
    path = tmp_path / "seed_results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in results) + "\n", encoding="utf-8")
    assert len(load_jsonl(path)) == 3
    path.write_text(json.dumps(results[0]) + "\n" + json.dumps(results[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(path)
    bad = dict(results[0], test_accuracy=0.8)
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test-named"):
        load_jsonl(path)


def _constant_slot_tensor(b: float, m0: float, m1: float, c0: float, c1: float) -> np.ndarray:
    values = np.empty((24, 5, 4, 3), dtype=np.float64)
    for index, value in enumerate((b, m0, m1, c0, c1)):
        values[:, index, :, :] = value
    return values


def test_random_batches_are_averaged_not_best_selected() -> None:
    values = _constant_slot_tensor(0.7, 0.9, 0.1, 0.8, 0.0)
    pool_rows, _ = observed_pool_metrics(values)
    assert pool_rows[0]["M_mean_batch_best_of_4"] == pytest.approx(0.5)
    assert pool_rows[0]["C_mean_batch_best_of_4"] == pytest.approx(0.4)


def test_hierarchical_bootstrap_shares_batch_and_seed_indices() -> None:
    # M0-C0 and M1-C1 are both exactly .1. Shared matched-batch indices must
    # therefore make every M-C replicate exactly .1.
    values = _constant_slot_tensor(0.7, 0.9, 0.1, 0.8, 0.0)
    samples, summary, indices = hierarchical_bootstrap(values, 500, 123, chunk_size=100)
    assert np.allclose(samples[:, 2], 0.1)
    assert summary["same_matched_batch_indices_for_M_and_C"]
    assert summary["same_seed_indices_all_configs_and_strategies_within_pool_draw"]
    assert indices[0].shape == (500, 24)
    # Shared configurations represented by identical seed vectors stay exactly paired.
    values[:, 1:3, :, :] = values[:, 0:1, :, :]
    samples, _, _ = hierarchical_bootstrap(values, 300, 321, chunk_size=100)
    assert np.allclose(samples[:, 1], 0.0)


def test_bootstrap_recomputes_best_after_seed_resampling() -> None:
    values = _constant_slot_tensor(0.0, 0.0, 0.0, 0.2, 0.2)
    # Which B slot wins changes with the sampled seed IDs.
    values[:, 0, 0, :] = [0.9, 0.1, 0.1]
    values[:, 0, 1, :] = [0.1, 0.9, 0.9]
    samples, _, _ = hierarchical_bootstrap(values, 600, 999, chunk_size=100)
    assert len(np.unique(np.round(samples[:, 0], 8))) > 1


def _pool_rows(bc: float, bm: float, mc: float, positive_count: int = 24) -> list[dict]:
    rows = []
    for pool in range(24):
        sign = 1.0 if pool < positive_count else -1.0
        rows.append({
            "pool_id": pool,
            "B_minus_C": abs(bc) * sign if bc >= 0 else bc,
            "B_minus_M": abs(bm) * sign if bm >= 0 else bm,
            "M_minus_C": abs(mc) * sign if mc >= 0 else mc,
        })
    return rows


def _ci(bc95: float, bm975: float, mc975: float, bm95: float | None = None) -> dict:
    return {"comparisons": {
        "B_minus_C": {"ci95_lower": bc95, "ci97_5_lower": bc95},
        "B_minus_M": {"ci95_lower": bm975 if bm95 is None else bm95, "ci97_5_lower": bm975},
        "M_minus_C": {"ci95_lower": mc975, "ci97_5_lower": mc975},
    }}


def test_all_statistical_classification_branches() -> None:
    assert classify(_pool_rows(.01, .01, 0), _ci(.002, .001, -.001))[0] == "JOINT_AND_ARCH_CONFIRMED"
    assert classify(_pool_rows(.01, 0, .01), _ci(.002, -.001, .001))[0] == "HP_FILTERING_ONLY"
    assert classify(_pool_rows(.01, .003, .003), _ci(.002, -.001, -.001))[0] == "JOINT_CONFIRMED_ATTRIBUTION_INCONCLUSIVE"
    assert classify(_pool_rows(-.001, 0, 0), _ci(-.003, -.003, -.003))[0] == "NO_HEAD_ADVANTAGE"
    assert classify(_pool_rows(.004, .002, .002), _ci(-.001, -.001, -.001))[0] == "CONFIRMATORY_INCONCLUSIVE"
    assert set(freeze.FINAL_CLASSIFICATIONS) == {
        "JOINT_AND_ARCH_CONFIRMED", "HP_FILTERING_ONLY",
        "JOINT_CONFIRMED_ATTRIBUTION_INCONCLUSIVE", "NO_HEAD_ADVANTAGE",
        "CONFIRMATORY_INCONCLUSIVE", "INSUFFICIENT_FRESH_CANDIDATES",
        "EXECUTION_OR_PROTOCOL_FAILURE",
    }


def test_source_dependency_manifest_covers_all_four_v2_entries() -> None:
    manifest = freeze.source_dependency_manifest()
    assert set(manifest["entry_files"]) == set(freeze.EXPECTED_NEW_FILES)
    relative = {row["relative_path"] for row in manifest["files"]}
    assert set(freeze.EXPECTED_NEW_FILES).issubset(relative)
