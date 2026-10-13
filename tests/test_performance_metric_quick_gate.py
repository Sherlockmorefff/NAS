from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyse import performance_metric_quick_gate as gate


CONFIG = REPO_ROOT / "configs/performance_metric_quick_gate_v2.yaml"


def _architecture(index: int) -> dict:
    operations = [gate.OPERATION_ORDER[index % len(gate.OPERATION_ORDER)]]
    config = {"operations": operations, "edges": [[0, 1], [1, 2]]}
    return {**config, "architecture_fingerprint": gate.architecture_fingerprint(config)}


def _configuration(index: int, hp: int = 0) -> dict:
    architecture = _architecture(index)
    # Make fingerprints unique while retaining a valid canonical DAG.
    fingerprint = f"arch-{index:03d}"
    return {
        **architecture,
        "architecture_fingerprint": fingerprint,
        "hp_fingerprint": f"hp-{hp}",
        "pool_identity": f"pool-{index // 8}",
        "validation_mean": 0.60 + index * 0.001 + hp * 0.0001,
        "validation_std": 0.01,
        "standard_error": 0.01 / (3 ** 0.5),
        "raw_hp": {
            "lr": 0.001 * (hp + 1), "dropout": 0.2 + hp * 0.01,
            "hidden_dim": 32 * (2 ** hp), "l2": 0.0001 * (hp + 1),
        },
    }


def test_same_architecture_never_crosses_outer_folds():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    configurations = [_configuration(index) for index in range(80)]
    manifest = gate.build_split_manifest(configurations)
    heldout = [
        fingerprint for fold in manifest["folds"]
        for fingerprint in fold["heldout_architecture_fingerprints"]
    ]
    assert len(heldout) == len(set(heldout)) == 80
    gate.assert_configuration_split_integrity(configurations, manifest)


def test_same_architecture_different_hp_never_crosses_folds():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    configurations = [
        _configuration(index, hp) for index in range(80) for hp in (0, 1)
    ]
    manifest = gate.build_split_manifest(configurations)
    for row in configurations:
        containing = [
            fold["fold"] for fold in manifest["folds"]
            if row["architecture_fingerprint"] in fold["heldout_architecture_fingerprints"]
        ]
        assert len(containing) == 1
    gate.assert_configuration_split_integrity(configurations, manifest)


def test_outer_heldout_label_cannot_enter_metric_pairs():
    rows = [
        {"architecture_fingerprint": "train", "target": 0.0, "standard_error": 0.1},
        {"architecture_fingerprint": "heldout", "target": 1.0, "standard_error": 0.1},
    ]
    with pytest.raises(gate.ProtocolError, match="outer-held-out label"):
        gate.construct_metric_pairs(
            rows, tau=1.0, weight_low=1.0, weight_high=10.0,
            allowed_fingerprints={"train"},
        )


def test_inner_early_stopping_split_reads_outer_train_only():
    outer_train = [
        {"architecture_fingerprint": f"train-{index}", "target": index / 10, "standard_error": 0.1}
        for index in range(20)
    ]
    fit, stop = gate._inner_architecture_split(outer_train, seed=0, train_fraction=0.8)
    assert {row["architecture_fingerprint"] for row in fit}.isdisjoint(
        {row["architecture_fingerprint"] for row in stop}
    )
    assert all(row["architecture_fingerprint"].startswith("train-") for row in stop)


def _ridge_case():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    configurations = [
        _configuration(index, hp) for index in range(20) for hp in (0, 1)
    ]
    train = {f"arch-{index:03d}" for index in range(15)}
    heldout = {f"arch-{index:03d}" for index in range(15, 20)}
    return gate.fold_local_hp_residuals(
        configurations, train, heldout, [0.01, 0.1, 1.0], 5,
    )


def test_hp_ridge_uses_outer_train_only():
    _train, _heldout, audit = _ridge_case()
    assert set(audit["ridge_fit_architecture_fingerprints"]).isdisjoint(
        audit["outer_heldout_architecture_fingerprints"]
    )
    assert audit["outer_heldout_used_for_ridge_fit"] is False


def test_train_residuals_use_cross_fitted_predictions():
    train, heldout, audit = _ridge_case()
    assert train and heldout
    assert audit["train_prediction_mode"] == "cross_fitted_group_kfold"
    assert audit["heldout_prediction_mode"] == "final_outer_train_ridge"


def test_test_field_fails_at_loader_layer(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"test_accuracy": 0.9}) + "\n", encoding="utf-8")
    with pytest.raises(gate.ProtocolError, match="test field rejected at loader layer"):
        gate._read_result_jsonl(path)


def test_preregistration_metadata_test_words_are_not_metric_fields(tmp_path: Path):
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps({
        "training": {"track_test": False},
        "source_code_hashes": {"/repo/tests/test_protocol.py": "abc"},
        "statistics": {"sign_test": "exact binomial"},
    }), encoding="utf-8")
    assert gate._read_preregistration(path)["training"]["track_test"] is False
    path.write_text(json.dumps({
        "training": {"track_test": False, "test_accuracy": 0.9},
    }), encoding="utf-8")
    with pytest.raises(gate.ProtocolError, match="training protocol fields"):
        gate._read_preregistration(path)


def test_knn_rejects_query_self():
    np = pytest.importorskip("numpy")
    with pytest.raises(gate.ProtocolError, match="query self-match"):
        gate.weighted_knn_predict(
            np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]]),
            np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]), np.asarray([[0.0]]),
            train_fingerprints=["same", "b", "c", "d", "e"],
            query_fingerprints=["same"],
        )


def test_graph_gp_and_metric_gp_share_one_protocol():
    config = gate.load_config(CONFIG)
    graph_signature = gate.gp_protocol_signature(config)
    metric_signature = gate.gp_protocol_signature(config)
    assert graph_signature == metric_signature
    assert graph_signature["likelihood"] == "fixed_se_squared_plus_learned_noise_floor"


def test_metric_seeds_are_averaged_not_selected():
    np = pytest.importorskip("numpy")
    averaged = gate.average_metric_seed_predictions({
        0: np.asarray([0.0]), 1: np.asarray([3.0]), 2: np.asarray([6.0]),
    })
    assert averaged.tolist() == [3.0]
    with pytest.raises(gate.ProtocolError):
        gate.average_metric_seed_predictions({0: np.asarray([0.0]), 2: np.asarray([6.0])})


def test_decision_tree_matches_frozen_thresholds():
    config = gate.load_config(CONFIG)
    geometry, predictions, regret = gate._synthetic_gate_inputs(True)
    assert gate.classify_quick_gate(39, None, None, None, config["decision"])["classification"] == "INSUFFICIENT_DATA"
    assert gate.classify_quick_gate(79, None, None, None, config["decision"])["classification"] == "QUICK_GATE_UNDERPOWERED"
    assert gate.classify_quick_gate(80, geometry, predictions, regret, config["decision"])["classification"] == "QUICK_GATE_GP_PASSED"
    rejected = gate.classify_quick_gate(80, *gate._synthetic_gate_inputs(False), config["decision"])
    assert rejected["classification"] == "QUICK_GATE_REJECTED"


def test_nonempty_formal_result_directory_cannot_be_overwritten(tmp_path: Path):
    output = tmp_path / "formal"
    output.mkdir()
    (output / "occupied.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(gate.ProtocolError, match="non-empty"):
        gate.ensure_output_policy(output, resume=False)


def test_dry_run_does_not_create_formal_result(tmp_path: Path):
    output = tmp_path / "formal"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "analyse/performance_metric_quick_gate.py"),
         "--config", str(CONFIG), "--output", str(output), "--dry-run"],
        cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    assert json.loads(result.stdout)["formal_data_read"] is False


def test_synthetic_pass_case_passes():
    config = gate.load_config(CONFIG)
    decision = gate.classify_quick_gate(
        80, *gate._synthetic_gate_inputs(True), config["decision"],
    )
    assert decision["classification"] == "QUICK_GATE_GP_PASSED"


def test_synthetic_leakage_and_reject_cases_fail_closed():
    config = gate.load_config(CONFIG)
    rejected = gate.classify_quick_gate(
        80, *gate._synthetic_gate_inputs(False), config["decision"],
    )
    assert rejected["classification"] == "QUICK_GATE_REJECTED"
    assert gate.synthetic_smoke(CONFIG) == 0
