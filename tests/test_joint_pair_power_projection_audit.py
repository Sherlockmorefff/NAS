"""Focused CPU-safe checks for the frozen pair-power projection audit.

The tests exercise only validation arithmetic, frozen result reconstruction,
reporting, and packaging.  They never call ``execute_formal`` and never invoke
any evaluator, metric encoder, regressor, GP, or acquisition implementation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
import tarfile
from typing import Any

import pytest

from analyse import joint_pair_power_projection_audit as audit


COUNT_KEYS = (
    "positive_pairs",
    "negative_pairs",
    "ambiguous_pairs",
    "anchors_with_positive_and_negative",
    "cross_architecture_positive_pairs",
    "cross_factor_positive_pairs",
    "reliable_negative_pairs",
)
ZERO_THRESHOLDS = {
    "anchors_with_positive_and_negative": 0,
    "cross_architecture_positive_pairs": 0,
    "cross_factor_positive_pairs": 0,
    "reliable_negative_pairs": 0,
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_base(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    joints = sorted({str(pair[key]) for pair in pairs for key in ("left", "right")})
    normalized = []
    for pair in pairs:
        normalized.append({
            "left": str(pair["left"]),
            "right": str(pair["right"]),
            "mean3": float(pair["mean3"]),
            "sd3": float(pair["sd3"]),
            "cross_architecture": bool(pair.get("cross_architecture", True)),
            "cross_factor": bool(pair.get("cross_factor", False)),
        })
    return {
        "train_joint_fingerprints": joints,
        "train_configuration_count": len(joints),
        "pairs": normalized,
    }


@pytest.fixture(scope="session")
def frozen_real_inputs() -> dict[str, Any]:
    """Load the two explicitly permitted frozen result roots, without a formal run."""
    config_path = audit.DEFAULT_CONFIG
    v1_root = audit.REPO_ROOT / "results/posthoc/joint-configuration-metric-gate-global4-v1"
    source_root = audit.REPO_ROOT / (
        "results/search/high-regime-prior-model-blind-validation-global4-v1"
    )
    if not config_path.is_file() or not v1_root.is_dir() or not source_root.is_dir():
        pytest.skip("frozen projection inputs are not present in this checkout")
    config = audit.load_config(config_path)
    configurations, inventory, provenance = audit.joint_v1.load_and_aggregate_source(config)
    return {
        "config": config,
        "configurations": configurations,
        "inventory": inventory,
        "provenance": provenance,
        "split": _json(v1_root / "split_manifest.json"),
        "pairs": _json(v1_root / "pair_inventory.json"),
    }


def test_n3_projection_exactly_reproduces_all_frozen_fold_pair_counts(
    frozen_real_inputs: dict[str, Any],
) -> None:
    """Requirement 1: n=3 is an exact reproduction, not a fresh interpretation."""
    config = frozen_real_inputs["config"]
    by_joint = {
        str(row["joint_fingerprint"]): row
        for row in frozen_real_inputs["configurations"]
    }
    expected_by_fold = {
        int(row["fold"]): row for row in frozen_real_inputs["pairs"]["folds"]
    }
    thresholds = audit._pair_thresholds(config)
    observed_folds = set()
    for split_fold in frozen_real_inputs["split"]["folds"]:
        fold = int(split_fold["fold"])
        observed_folds.add(fold)
        train_ids = set(map(str, split_fold["train_joint_fingerprints"]))
        train_rows = [by_joint[joint] for joint in sorted(train_ids)]
        base = audit.build_fold_pair_base(
            train_rows,
            allowed_joint_fingerprints=train_ids,
            equivalence_margin=0.006,
            far_margin=0.012,
        )
        projected = audit.inventory_at_n(base, 3, thresholds)
        expected = expected_by_fold[fold]
        assert {
            key: int(projected[key]) for key in COUNT_KEYS
        } == {
            key: int(expected["counts"][key]) for key in COUNT_KEYS
        }
        assert projected["total_pair_count"] == len(train_rows) * (len(train_rows) - 1) // 2
        assert math.isclose(
            base["hp_distance_median"],
            float(expected["hp_distance_median"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        )
        compact_pairs = [
            {
                "left": pair["left"],
                "right": pair["right"],
                "label": pair["label_n3_v1"],
                "cross_architecture": pair["cross_architecture"],
                "cross_factor": pair["cross_factor"],
            }
            for pair in base["pairs"]
        ]
        assert audit.joint_v1.stable_hash(compact_pairs) == expected["pair_labels_sha256"]
    assert observed_folds == set(range(5))


def test_frozen_split_logical_sha_and_group_isolation(
    frozen_real_inputs: dict[str, Any],
) -> None:
    """Requirement 2: rebuild the logical SHA and retain joint/architecture grouping."""
    split = frozen_real_inputs["split"]
    payload = dict(split)
    recorded = payload.pop("split_sha256")
    assert recorded == "b80b1a211a05285121cafe4dabd4c67baddea93606c1913e777fcd34dd79a2ec"
    assert audit.joint_v1.stable_hash(payload) == recorded

    by_joint = {
        str(row["joint_fingerprint"]): row
        for row in frozen_real_inputs["configurations"]
    }
    joint_owner: dict[str, int] = {}
    architecture_owners: dict[str, set[int]] = {}
    for fold_row in split["folds"]:
        fold = int(fold_row["fold"])
        train = set(map(str, fold_row["train_joint_fingerprints"]))
        heldout = set(map(str, fold_row["heldout_joint_fingerprints"]))
        assert train.isdisjoint(heldout)
        for joint in heldout:
            assert joint not in joint_owner
            joint_owner[joint] = fold
            architecture = str(by_joint[joint]["architecture_fingerprint"])
            architecture_owners.setdefault(architecture, set()).add(fold)
    assert set(joint_owner) == set(by_joint)
    assert all(len(owners) == 1 for owners in architecture_owners.values())


def test_exact_t_critical_uses_n_minus_one_degrees_of_freedom() -> None:
    """Requirement 3: each projected n uses its own exact Student-t critical value."""
    scipy_stats = pytest.importorskip("scipy.stats")
    radii = []
    for n in (3, 5, 7, 10, 30):
        expected = float(scipy_stats.t.ppf(0.95, df=n - 1))
        assert audit.t_critical_90(n) == pytest.approx(expected, rel=0.0, abs=1e-14)
        interval = audit.projected_interval(0.003, 0.02, n)
        assert interval["degrees_of_freedom"] == n - 1
        assert interval["radius"] == pytest.approx(expected * 0.02 / math.sqrt(n))
        radii.append(interval["radius"])
    assert radii == sorted(radii, reverse=True)


def test_projected_positive_and_negative_counts_are_monotone_in_n() -> None:
    """Requirements 4-5: fixed-mean/fixed-SD reliable counts never decrease."""
    base = _synthetic_base([
        {"left": "a", "right": "b", "mean3": 0.000, "sd3": 0.012,
         "cross_factor": True},
        {"left": "a", "right": "c", "mean3": 0.003, "sd3": 0.008},
        {"left": "b", "right": "c", "mean3": 0.022, "sd3": 0.018},
        {"left": "a", "right": "d", "mean3": -0.018, "sd3": 0.010},
        {"left": "c", "right": "d", "mean3": 0.008, "sd3": 0.002},
    ])
    seed_grid = [3, 5, 7, 10, 15, 20, 30, 50, 100, 200]
    inventories = [audit.inventory_at_n(base, n, ZERO_THRESHOLDS) for n in seed_grid]
    positives = [row["positive_pairs"] for row in inventories]
    negatives = [row["negative_pairs"] for row in inventories]
    assert positives == sorted(positives)
    assert negatives == sorted(negatives)
    for inventory in inventories:
        assert (
            inventory["positive_pairs"]
            + inventory["negative_pairs"]
            + inventory["ambiguous_pairs"]
        ) == inventory["total_pair_count"]


@pytest.mark.parametrize(
    ("mean", "expected"),
    [
        (-0.0120001, "negative"),
        (-0.012, "ambiguous"),
        (-0.006, "positive"),
        (0.0, "positive"),
        (0.006, "positive"),
        (0.012, "ambiguous"),
        (0.0120001, "negative"),
    ],
)
def test_point_limit_boundaries_are_frozen(mean: float, expected: str) -> None:
    """Requirement 6: inclusive equivalence and strict far boundaries are exact."""
    assert audit.classify_point_limit(mean) == expected


def test_minimum_n_marks_impossible_pairs_without_mislabeling_boundary_pairs() -> None:
    """Requirement 7: only abs(mean)>delta_eq is fixed-mean never-positive."""
    outside = audit.minimum_seed_requirements({"mean3": 0.0060001, "sd3": 0.0})
    boundary = audit.minimum_seed_requirements({"mean3": 0.006, "sd3": 0.0})
    noisy_boundary = audit.minimum_seed_requirements({"mean3": 0.006, "sd3": 0.02})
    far_boundary = audit.minimum_seed_requirements({"mean3": 0.012, "sd3": 0.0})
    far_outside = audit.minimum_seed_requirements({"mean3": 0.0120001, "sd3": 0.0})
    assert outside["never_positive_under_fixed_mean"] is True
    assert outside["minimum_n_for_positive"] is None
    assert boundary["never_positive_under_fixed_mean"] is False
    assert boundary["minimum_n_for_positive"] == 3
    assert noisy_boundary["never_positive_under_fixed_mean"] is False
    assert noisy_boundary["positive_not_reached_by_n500"] is True
    assert far_boundary["never_negative_under_fixed_mean"] is True
    assert far_outside["never_negative_under_fixed_mean"] is False
    assert far_outside["minimum_n_for_negative"] == 3


def test_heldout_validation_cannot_enter_outer_train_pair_base(
    frozen_real_inputs: dict[str, Any],
) -> None:
    """Requirement 8: pair construction accepts exactly the declared outer-train IDs."""
    split_fold = frozen_real_inputs["split"]["folds"][0]
    by_joint = {
        str(row["joint_fingerprint"]): row
        for row in frozen_real_inputs["configurations"]
    }
    train_ids = sorted(map(str, split_fold["train_joint_fingerprints"]))[:6]
    heldout_id = str(split_fold["heldout_joint_fingerprints"][0])
    heldout = dict(by_joint[heldout_id])
    heldout["validation_seed_0"] = 999.0
    heldout["validation_seed_1"] = -999.0
    heldout["validation_seed_2"] = 123.0
    base = audit.build_fold_pair_base(
        [by_joint[joint] for joint in train_ids],
        allowed_joint_fingerprints=set(train_ids),
    )
    assert base["outer_heldout_labels_used"] is False
    assert base["primary_parameter_seeds"] == [0, 1, 2]
    assert all(heldout_id not in (pair["left"], pair["right"]) for pair in base["pairs"])
    with pytest.raises(audit.ProtocolError, match="allowed outer-train"):
        audit.build_fold_pair_base(
            [by_joint[joint] for joint in train_ids] + [heldout],
            allowed_joint_fingerprints=set(train_ids),
        )


@pytest.mark.parametrize(
    "forbidden",
    ["test_accuracy", "test_mask", "test_labels", "test_predictions"],
)
def test_loader_layer_rejects_every_test_field(forbidden: str) -> None:
    """Requirement 9: forbidden test-named payloads fail before aggregation."""
    with pytest.raises(audit.InputInvalidError, match="test field rejected at loader layer"):
        audit._reject_test_fields({"nested": {forbidden: [1]}}, "synthetic-loader-record")


class _GuardedValues:
    def __init__(self, values: list[float], maximum_index: int) -> None:
        self.values = values
        self.maximum_index = maximum_index
        self.accesses: list[int] = []

    def __getitem__(self, index: int) -> float:
        if index > self.maximum_index:
            raise AssertionError(f"forbidden seed access: {index}")
        self.accesses.append(index)
        return self.values[index]


def test_calibration_projection_reads_only_seeds_zero_through_two() -> None:
    """Requirement 10: projected-prefix parameters structurally cannot read later seeds."""
    left = _GuardedValues([0.80, 0.81, 0.79] + [99.0] * 27, maximum_index=2)
    right = _GuardedValues([0.79, 0.80, 0.78] + [-99.0] * 27, maximum_index=2)
    projected = audit.project_calibration_pair_from_first_three(left, right, 30)
    assert projected["label"] in audit.LABELS
    assert left.accesses == [0, 1, 2]
    assert right.accesses == [0, 1, 2]


def test_actual_prefix_reads_exactly_the_requested_seed_prefix() -> None:
    """Requirement 11: actual-prefix n reads 0..n-1 and nothing else."""
    left = _GuardedValues([0.80 + seed * 0.001 for seed in range(30)], maximum_index=6)
    right = _GuardedValues([0.79 + seed * 0.001 for seed in range(30)], maximum_index=6)
    actual = audit.paired_prefix_statistics(left, right, 7)
    assert actual["label"] in audit.LABELS
    assert left.accesses == list(range(7))
    assert right.accesses == list(range(7))


def test_calibration_tail_cannot_change_primary_three_seed_projection() -> None:
    """Requirement 12: seeds 3..29 do not enter primary plug-in parameters."""
    first_three_left = [0.80, 0.82, 0.81]
    first_three_right = [0.79, 0.80, 0.80]
    calm_left = first_three_left + [0.81] * 27
    calm_right = first_three_right + [0.80] * 27
    wild_left = first_three_left + [1000.0 if seed % 2 else -1000.0 for seed in range(27)]
    wild_right = first_three_right + [-500.0 if seed % 2 else 500.0 for seed in range(27)]
    for n in (3, 5, 10, 20, 30):
        assert audit.project_calibration_pair_from_first_three(
            calm_left, calm_right, n,
        ) == audit.project_calibration_pair_from_first_three(wild_left, wild_right, n)


def test_calibration_replay_has_91_pairs_and_complete_transition_matrices() -> None:
    subset = {
        f"joint-{index:02d}": {
            "joint_fingerprint": f"joint-{index:02d}",
            "values": [
                0.75 + index * 0.002 + ((seed % 3) - 1) * 0.0005
                for seed in range(30)
            ],
        }
        for index in range(14)
    }
    rows, transitions, metrics = audit.run_calibration_replay(subset, [3, 5, 30])
    assert [row["pair_count"] for row in rows] == [91, 91, 91]
    for row in rows:
        assert sum(row[f"actual_{label}"] for label in audit.LABELS) == 91
        assert sum(row[f"projected_{label}"] for label in audit.LABELS) == 91
        assert isinstance(row["undefined_metrics"], list)
    for n in (3, 5, 30):
        matrix = transitions["by_n"][str(n)]
        assert sum(matrix[projected][actual] for projected in audit.LABELS
                   for actual in audit.LABELS) == 91
        assert metrics["by_n"][str(n)]["transition_matrix_orientation"] == (
            "rows=projected, columns=actual-prefix"
        )


def test_full_topup_cost_deducts_existing_common_prefix_seeds() -> None:
    """Requirement 13: existing 30-seed configurations are never charged again."""
    coverage = {
        "three": {0, 1, 2},
        "five": set(range(5)),
        "thirty": set(range(30)),
    }
    by_target = {row["target_n"]: row for row in audit.full_topup_cost(coverage, [5, 7, 30])}
    assert by_target[5] == {
        "status": "AVAILABLE",
        "target_n": 5,
        "config_count": 3,
        "configs_already_complete": 2,
        "configs_needing_topup": 1,
        "additional_gnn_runs": 2,
        "total_resulting_seed_records": 15,
        "gpu_hour_estimate": "unavailable",
    }
    assert by_target[7]["additional_gnn_runs"] == 6
    assert by_target[30]["additional_gnn_runs"] == 52
    assert audit.existing_common_prefix_seed_count({0, 1, 2, 4, 5}) == 3


def _priority_row(
    fingerprint: str, *, missing: int = 0, anchor: int = 0,
    cross_factor: int = 0, cross_architecture: int = 0, runs: int = 3,
) -> dict[str, Any]:
    return {
        "joint_fingerprint": fingerprint,
        "number_of_outer_folds_in_which_it_can_create_a_missing_cross_factor_edge": missing,
        "number_of_outer_folds_in_which_it_can_create_a_new_eligible_anchor": anchor,
        "projected_cross_factor_positive_degree": cross_factor,
        "projected_cross_architecture_positive_degree": cross_architecture,
        "additional_runs_to_target_n": runs,
    }


def test_priority_ranking_has_the_frozen_reproducible_tie_break() -> None:
    """Requirement 14: all six ordering levels are stable and deterministic."""
    expected = ["missing", "anchor", "factor", "architecture", "cheap", "alpha", "beta"]
    rows = [
        _priority_row("beta"),
        _priority_row("architecture", missing=1, anchor=1, cross_factor=1,
                      cross_architecture=2),
        _priority_row("factor", missing=1, anchor=1, cross_factor=2),
        _priority_row("missing", missing=2),
        _priority_row("cheap", missing=1, anchor=1, cross_factor=1,
                      cross_architecture=1, runs=1),
        _priority_row("anchor", missing=1, anchor=2),
        _priority_row("alpha"),
    ]
    for seed in range(10):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        assert [row["joint_fingerprint"] for row in sorted(
            shuffled, key=audit.priority_sort_key,
        )] == expected


@pytest.mark.parametrize(
    ("folds", "expected"),
    [
        ([False] * 5, "PAIR_POWER_SUPPORT_LIMITED"),
        ([True] * 5, "PAIR_POWER_PRECISION_LIMITED"),
        ([True, False, True, False, False], "PAIR_POWER_MIXED_LIMITED"),
    ],
)
def test_primary_classification_uses_only_point_limit_fold_statuses(
    folds: list[bool], expected: str,
) -> None:
    """Requirements 15-16 and 23: calibration has no vote in the main decision."""
    assert audit.classify_point_gate(folds) == expected
    assert tuple(inspect.signature(audit.classify_point_gate).parameters) == ("fold_passes",)
    contradictory_calibration = {
        "overall_label_accuracy": 0.0,
        "macro_f1": 0.0,
        "claimed_classification": "PAIR_POWER_INPUT_INVALID",
    }
    assert contradictory_calibration  # Deliberately cannot be supplied to the classifier.
    assert audit.classify_point_gate(folds) == expected


def test_source_snapshot_is_complete_and_every_hash_recomputes(tmp_path: Path) -> None:
    """Requirements 17-19: mandatory and runtime-imported local bytes are preserved."""
    output = tmp_path / "snapshot-result"
    config = audit.load_config(audit.DEFAULT_CONFIG)
    expected_runtime_paths = set()
    for module in list(sys.modules.values()):
        source = audit._source_path_for_module(module)
        if source is not None:
            expected_runtime_paths.add(source.relative_to(audit.REPO_ROOT.resolve()).as_posix())

    manifest = audit.snapshot_runtime_sources(
        output, config["source_snapshot"]["mandatory_files"],
    )
    verified = audit.verify_source_snapshot(output)
    entries = {
        row["repository_relative_path"]: row for row in manifest["entries"]
    }
    assert verified["manifest_root_sha256"] == manifest["manifest_root_sha256"]
    assert {
        "analyse/joint_configuration_metric_gate.py",
        "analyse/performance_metric_quick_gate.py",
    } <= set(entries)
    assert expected_runtime_paths <= set(entries)
    imported = _json(output / "source_snapshot/imported_local_modules.json")
    imported_paths = {
        row["repository_relative_path"] for row in imported["modules"]
    }
    assert imported_paths == expected_runtime_paths

    for relative in config["source_snapshot"]["mandatory_files"]:
        assert (output / "source_snapshot" / relative).read_bytes() == (
            audit.REPO_ROOT / relative
        ).read_bytes()
    for relative, row in entries.items():
        copied = output / row["snapshot_relative_path"]
        assert copied.stat().st_size == row["size_bytes"]
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == row["sha256"]
    assert audit.stable_hash(manifest["entries"]) == manifest["manifest_root_sha256"]


def test_final_report_has_all_twelve_sections_nonempty_tables_and_length() -> None:
    """Requirement 20: the standalone report cannot regress to an empty shell."""
    config = audit.load_config(audit.DEFAULT_CONFIG)
    report = audit.render_final_report({
        "classification": "PAIR_POWER_SUPPORT_LIMITED",
        "environment": {"branch": "synthetic", "head": "0" * 40, "dirty_tree": True},
        "integrity": {"status": "VALID"},
        "inventory": {},
        "reproduction": {},
        "point": {"folds_passing": 0},
        "projection_rows": [],
        "projection_summary": {},
        "minimum_summary": {},
        "calibration_inventory": {"calibration_status": "UNAVAILABLE"},
        "calibration_rows": [],
        "calibration_transitions": {},
        "cost_rows": [],
        "priority_rows": [],
    })
    sections = config["reporting"]["required_sections"]
    audit.validate_final_report(report, sections)
    assert len("".join(report.split())) >= 3000
    for index, title in enumerate(sections, 1):
        assert f"## {index}. {title}" in report
    for section_number in (3, 4, 5, 6, 7, 8, 9, 12):
        start = report.index(f"## {section_number}. ")
        end_token = f"## {section_number + 1}. "
        end = report.index(end_token, start) if section_number < 12 else len(report)
        assert len([line for line in report[start:end].splitlines() if line.startswith("|")]) >= 3


def test_dry_run_does_not_create_formal_result_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 21: dry-run reports intent without materializing formal output."""
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        audit,
        "execute_formal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("formal path called")),
    )
    result = audit.main([
        "--config", str(audit.DEFAULT_CONFIG),
        "--output", str(output),
        "--dry-run",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["formal_data_read"] is False
    assert payload["formal_result_root_created"] is False
    assert payload["candidate_gnn_training_started"] is False
    assert payload["test_data_accessed"] is False
    assert not output.exists()


def test_nonempty_formal_directory_is_never_overwritten(tmp_path: Path) -> None:
    """Requirement 22: the non-overwrite guard preserves all existing bytes."""
    output = tmp_path / "formal"
    output.mkdir()
    sentinel = output / "user-owned.txt"
    sentinel.write_bytes(b"preserve-me")
    with pytest.raises(audit.ProtocolError, match="exists and is non-empty"):
        audit.ensure_output_available(output)
    assert sentinel.read_bytes() == b"preserve-me"
    assert sorted(path.name for path in output.iterdir()) == ["user-owned.txt"]


def test_synthetic_smoke_reports_all_three_frozen_limit_cases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 23: the public smoke mode covers support/precision/mixed cases."""
    assert audit.synthetic_smoke() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["synthetic_support_limited"] is True
    assert payload["synthetic_precision_limited"] is True
    assert payload["synthetic_mixed_limited"] is True
    assert payload["model_training_started"] is False
    assert payload["formal_data_read"] is False


def test_failure_bundle_is_complete_snapshot_backed_and_packagable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 24: a controlled failure still produces a verifiable archive."""
    output = tmp_path / "failed-audit"
    archive = tmp_path / "failed-audit.tar.gz"
    sidecar = tmp_path / "failed-audit.tar.gz.sha256"

    def fake_environment(path: Path) -> dict[str, Any]:
        environment = {
            "branch": "synthetic", "head": "0" * 40, "dirty_tree": True,
            "cpu_only": True, "gpu_used": False, "network_access_used": False,
            "gnn_training_started": False, "metric_model_training_started": False,
            "regressor_training_started": False, "gp_fitting_started": False,
            "acquisition_used": False, "test_data_accessed": False,
        }
        audit.dump_json(path / "environment.json", environment)
        audit.dump_text(path / "git_status.txt", "synthetic dirty tree\n")
        audit.dump_text(path / "git_diff.patch", "")
        return environment

    monkeypatch.setattr(audit, "write_git_and_environment", fake_environment)
    audit.write_failure_bundle(
        output,
        "synthetic controlled failure",
        config_sha256="1" * 64,
        source_sha256="2" * 64,
        config_path=audit.DEFAULT_CONFIG,
    )
    assert (output / "FAILED").is_file()
    assert not (output / "COMPLETE").exists()
    assert _json(output / "decision.json")["classification"] == "RUN_FAILED"
    assert _json(output / "state.json")["status"] == "failed"
    assert _json(output / "error.json")["classification"] == "RUN_FAILED"
    audit.verify_source_snapshot(output)

    digest = audit.create_archive(output, archive, sidecar)
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert sidecar.read_text(encoding="utf-8").split()[0] == digest
    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    prefix = output.name
    assert f"{prefix}/FAILED" in names
    assert f"{prefix}/error.json" in names
    assert f"{prefix}/source_snapshot/imported_local_modules.json" in names
    assert f"{prefix}/source_snapshot/sha256_manifest.json" in names
    assert f"{prefix}/source_snapshot/analyse/joint_configuration_metric_gate.py" in names
    assert f"{prefix}/source_snapshot/analyse/performance_metric_quick_gate.py" in names
    assert not any("__pycache__" in name for name in names)
    assert not any(name.endswith((".pyc", ".pyo", ".pt", ".pth", ".ckpt")) for name in names)
