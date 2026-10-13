from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analyse import cora_accuracy_stage_a as stage_a


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "legacy_artifacts/pre_20260813/results"
CHECKPOINT = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"


def _architecture(operation: str, extra_edge: bool = False) -> dict:
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    if extra_edge:
        edges.append([0, 5])
    return {
        "operations": [operation, "GATConv", "SAGEConv", "GINConv", "Identity"],
        "edges": edges,
        "z_search": [0.0] * 16,
        "val_acc": 0.8,
    }


def test_candidate_fingerprint_deduplication_is_float32_canonical() -> None:
    z64 = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
    z32 = z64.astype(np.float32)
    assert stage_a.candidate_fingerprint(z64) == stage_a.candidate_fingerprint(z32)
    assert len({stage_a.candidate_fingerprint(z64), stage_a.candidate_fingerprint(z32)}) == 1


def test_distances_are_symmetric_and_in_range() -> None:
    left = _architecture("GCNConv")
    right = _architecture("GCNII", extra_edge=True)
    right["z_search"] = [1.0] * 12 + [0.5] * 4
    forward = stage_a.pair_distances(left, right)
    backward = stage_a.pair_distances(right, left)
    for field in (
        "latent_distance", "operation_distance", "edge_distance",
        "combined_distance", "path_distance",
    ):
        assert forward[field] == pytest.approx(backward[field])
        assert 0.0 <= forward[field] <= 1.0


def test_prefix_only_rejects_future_label_and_test_data() -> None:
    valid = {
        "full_evaluation_index": 50,
        "gp_train_size_before": 50,
        "gp_train_size_after": 51,
        "evaluation_stage": "online_bo",
        "gp_update_performed": True,
    }
    stage_a.assert_prefix_only([valid], "synthetic")
    with pytest.raises(stage_a.IntegrityFailure, match="future-label"):
        stage_a.assert_prefix_only([{**valid, "gp_train_size_before": 51}], "synthetic")
    with pytest.raises(stage_a.IntegrityFailure, match="validation/test leakage"):
        stage_a.assert_prefix_only([{**valid, "test_acc": 0.9}], "synthetic")


def test_integrity_gate_population_and_lf_pairing() -> None:
    gate, full, low = stage_a.integrity_gate(
        REPO_ROOT,
        ARTIFACT_ROOT,
        CHECKPOINT,
        REPO_ROOT / "Cora",
        None,
    )
    assert gate["status"] == "PASS"
    assert len(full) == 4500
    assert len({row["candidate_fingerprint"] for row in full}) == 3582
    assert sum(row.get("gp_pred_mean") is not None for row in full) == 3750
    assert gate["population"]["direct_lf_hf_pairs"] == {"G100": 500, "G150": 750}
    assert len(stage_a._direct_pairs(full, low, "G100", 0)) == 100
    assert len(stage_a._direct_pairs(full, low, "G150", 0)) == 150


def test_fixed_seed_decoder_is_reproducible_on_cpu() -> None:
    model = stage_a.load_vae(CHECKPOINT)
    z = torch.linspace(-0.5, 0.5, stage_a.ARCH_DIM)
    seed = 71423
    with stage_a.isolated_rng(seed), torch.no_grad():
        first = model.arch_vae.decode(z.unsqueeze(0), stochastic=True)[0]
    with stage_a.isolated_rng(seed), torch.no_grad():
        second = model.arch_vae.decode(z.unsqueeze(0), stochastic=True)[0]
    first_config = stage_a._config_from_graph(model, first)
    second_config = stage_a._config_from_graph(model, second)
    assert stage_a._config_fingerprint(first_config) == stage_a._config_fingerprint(second_config)


def test_analysis_source_has_no_gnn_training_call() -> None:
    source_path = REPO_ROOT / "analyse/cora_accuracy_stage_a.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden = {"train_and_eval_arch", "eval_z_search", "eval_candidate"}
    assert not forbidden.intersection(called | attributes)
    assert "from eval_utils import" not in source
    assert "from bo_phase4 import" not in source


def test_output_manifest_is_deterministic(tmp_path: Path) -> None:
    gate = {"status": "PASS"}
    a1 = {"classification": "A1-PASS"}
    a2 = {"classification": "A2-PASS"}
    a3 = {"classification": "A3-RETAIN"}
    first = tmp_path / "first"
    second = tmp_path / "second"
    stage_a.write_outputs(first, gate, a1, a2, a3, REPO_ROOT)
    stage_a.write_outputs(second, gate, a1, a2, a3, REPO_ROOT)
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
