from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from analyse import performance_aware_vae_strict_regression as strict
from analyse import evaluate_performance_aware_vae_strict_regression as evaluate
from analyse.performance_aware_vae_pilot import contains_test_field, sha256_file


ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ROOT / "results/posthoc/vae-label-expansion-global4-v1"
OLD = ROOT / "results/posthoc/performance-aware-vae-pilot-global4-v1"
LABELS = UNIFIED / "unified_training_data_no_test.json"
BASE = ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"


@pytest.fixture(scope="module")
def labels():
    return json.loads(LABELS.read_text())


@pytest.fixture(scope="module")
def fixed_splits(labels):
    return strict.build_fixed_splits(labels)


def test_unified_labels_have_76_configurations_and_60_architectures(labels):
    assert len(labels) == 76
    assert len({row["architecture_fingerprint"] for row in labels}) == 60


def test_validation_training_export_has_no_test_field(labels):
    assert not contains_test_field(labels)


def test_fixed_split_ids_and_counts_reproduce_archive(fixed_splits):
    audit = json.loads((UNIFIED / "representation_split_audit.json").read_text())
    expected = {
        (row["protocol"], row["split_id"]): (
            row["train_observation_count"],
            row["validation_observation_count"],
            row["train_architecture_group_count"],
            row["validation_architecture_group_count"],
        )
        for row in audit
    }
    observed = {
        (row["protocol"], row["split_id"]): (
            row["train_observation_count"],
            row["validation_observation_count"],
            row["train_architecture_count"],
            row["validation_architecture_count"],
        )
        for row in fixed_splits
    }
    assert observed == expected


def test_fixed_time_prefix_reproduces(fixed_splits):
    prefixes = [row for row in fixed_splits if row["protocol"] == "time_prefix"]
    assert [row["split_id"] for row in prefixes] == [
        "prefix_0020",
        "prefix_0030",
        "prefix_0040",
        "prefix_0050",
        "prefix_0060",
        "prefix_0070",
    ]


def test_every_fixed_split_has_zero_architecture_leakage(fixed_splits):
    assert len(fixed_splits) == 56
    for split in fixed_splits:
        assert split["leakage_count"] == 0
        assert not set(split["train_architecture_fingerprints"]) & set(
            split["validation_architecture_fingerprints"]
        )


def test_same_architecture_hp_configurations_never_cross_split(labels, fixed_splits):
    lookup = {row["config_id"]: row for row in labels}
    for split in fixed_splits:
        strict.assert_same_architecture_side(split, lookup)


def test_standardization_uses_only_training_rows():
    train = [
        {"raw_hp_vector": [0.0, 1.0, 2.0, 3.0], "validation_mean": 0.2},
        {"raw_hp_vector": [2.0, 3.0, 4.0, 5.0], "validation_mean": 0.4},
    ]
    scaling = strict._scaling(train)
    assert scaling["hp_mean"] == [1.0, 2.0, 3.0, 4.0]
    assert scaling["y_mean"] == pytest.approx(0.3)


def test_old_head_and_h_memberships_are_incompatible_with_fixed_splits(fixed_splits):
    old = strict._old_pilot_audit(OLD)
    old_folds = json.loads(Path(old["old_split_file"]).read_text())
    old_members = {frozenset(row["outer_test_architectures"]) for row in old_folds}
    fixed_members = {
        frozenset(row["validation_architecture_fingerprints"]) for row in fixed_splits
    }
    assert old_members != fixed_members


def test_incompatible_predictions_are_not_marked_reusable(fixed_splits):
    old_folds = json.loads((OLD / "architecture_group_folds.json").read_text())
    old_members = {frozenset(row["outer_test_architectures"]) for row in old_folds}
    fixed_members = {
        frozenset(row["validation_architecture_fingerprints"]) for row in fixed_splits
    }
    assert old_members != fixed_members


def test_strict_task_enables_encoder_decoder_and_predictor_source_contract():
    source = (ROOT / "analyse/performance_aware_vae_strict_regression.py").read_text()
    assert "for parameter in model.arch_vae.parameters():\n            parameter.requires_grad_(True)" in source
    assert "for parameter in head.parameters():\n        parameter.requires_grad_(True)" in source


def test_decoder_is_not_frozen_in_strict_tasks():
    source = (ROOT / "analyse/performance_aware_vae_strict_regression.py").read_text()
    assert "model.arch_vae.parameters()" in source
    assert '"decoder_requires_grad": task["task_kind"] == "strict_ft"' in source


def test_forbidden_auxiliary_loss_weights_are_zero():
    source = (ROOT / "analyse/performance_aware_vae_strict_regression.py").read_text()
    for name in ("ranking_weight", "contrastive_weight", "pseudo_label_weight", "icnn_weight"):
        assert f'"{name}": 0.0' in source


@pytest.mark.parametrize("value", strict.LAMBDAS)
def test_only_preregistered_lambdas_are_accepted(value):
    assert strict.validate_lambda(value) == value


@pytest.mark.parametrize("value", [-1.0, 0.0, 0.3, 1.0, 2.0])
def test_unregistered_lambdas_are_rejected(value):
    with pytest.raises(ValueError):
        strict.validate_lambda(value)


def test_mse_has_positive_minimization_sign():
    parameter = torch.tensor(2.0, requires_grad=True)
    loss = strict.convex_combination_loss(parameter * 0, parameter**2, 0.5)
    loss.backward()
    assert parameter.grad.item() > 0


def test_convex_combination_boundaries():
    vae = torch.tensor(3.0)
    mse = torch.tensor(7.0)
    assert strict.convex_combination_loss(vae, mse, 0.0).item() == 3.0
    assert strict.convex_combination_loss(vae, mse, 1.0).item() == 7.0


def test_each_lambda_task_is_an_independent_base_checkpoint_initialization_source():
    source = (ROOT / "analyse/performance_aware_vae_strict_regression.py").read_text()
    assert "model = load_joint_vae(plan[\"inputs\"][\"base_checkpoint\"]" in source
    assert "phase_seed" in source


class _FakeArchVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(12))

    def encode(self, graphs):
        mu = self.bias.unsqueeze(0).repeat(len(graphs), 1)
        return mu, torch.zeros_like(mu)


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch_vae = _FakeArchVAE()


def test_performance_head_features_include_hp(labels):
    model = _FakeModel()
    rows = [dict(labels[0]), dict(labels[0])]
    rows[1]["raw_hp_vector"] = [value + 0.5 for value in rows[1]["raw_hp_vector"]]
    scaling = strict._scaling(rows)
    features, _ = strict._features(
        model, rows, scaling, "strict_ft_predictor", torch.device("cpu"), grad=False
    )
    assert features.shape[1] == 16
    assert not torch.equal(features[0, -4:], features[1, -4:])


def test_reconstruction_sampler_deduplicates_architectures(labels):
    model = _FakeModel()
    graph = strict.config_to_igraph(labels[0])
    # Fake config conversion requires model.op_mapping, so use a tiny compatible object.
    real_model = strict.load_joint_vae(BASE, "global4", device="cpu")
    unique = strict._unique_replay_graphs(
        real_model,
        [(graph.copy(), torch.zeros(4)), (graph.copy(), torch.ones(4))],
    )
    assert len(unique) == 1


def test_nonzero_performance_gradient_reaches_encoder():
    parameter = nn.Parameter(torch.tensor(2.0))
    value = strict.gradient_norm(parameter**2, [parameter], retain_graph=False)
    assert value == pytest.approx(4.0)


def test_nonzero_reconstruction_gradient_reaches_decoder():
    parameter = nn.Parameter(torch.tensor(3.0))
    value = strict.gradient_norm(parameter**2, [parameter], retain_graph=False)
    assert value == pytest.approx(6.0)


def test_base_checkpoint_hash_is_known_and_read_only():
    assert sha256_file(BASE) == strict.EXPECTED_BASE_CHECKPOINT_SHA256


def test_ft_gp_feature_path_uses_model_mu(labels):
    model = _FakeModel()
    x = evaluate._latent_features(labels[:2], representation="FT", model=model)
    assert x.shape == (2, 16)
    assert np.allclose(x[:, :12], 0.0)


def test_top10_regret_is_formally_computed(labels):
    rows = []
    for index, row in enumerate(labels[:12]):
        rows.append(
            {
                **row,
                "prediction": -float(row["validation_mean"]),
                "prediction_interval_low": -10.0,
                "prediction_interval_high": 10.0,
            }
        )
    metrics = evaluate.metric_bundle(rows)
    assert "top_10_regret" in metrics
    assert metrics["top_10_regret"] >= 0


def test_reliable_pair_count_is_formally_recorded(labels):
    rows = [
        {
            **row,
            "prediction": float(row["validation_mean"]),
            "prediction_interval_low": 0.0,
            "prediction_interval_high": 1.0,
        }
        for row in labels
    ]
    metrics = evaluate.metric_bundle(rows)
    assert isinstance(metrics["reliable_top_pair_count"], int)
    assert isinstance(metrics["raw_top_pair_count"], int)


def test_holm_correction_is_monotone_and_bounded():
    corrected = evaluate._holm(
        [
            {"lambda": 0.1, "raw_p_value": 0.01},
            {"lambda": 0.5, "raw_p_value": 0.04},
            {"lambda": 0.9, "raw_p_value": 0.02},
        ]
    )
    assert all(0 <= row["holm_adjusted_p_value"] <= 1 for row in corrected)
    by_raw = sorted(corrected, key=lambda row: row["raw_p_value"])
    assert [row["holm_adjusted_p_value"] for row in by_raw] == sorted(
        row["holm_adjusted_p_value"] for row in by_raw
    )


def test_high_noise_sensitivity_tasks_are_true_retraining_tasks(labels, fixed_splits):
    reduced = [row for row in labels if not row["high_noise"]]
    reduced_splits = strict.filter_split_manifest(fixed_splits, reduced)
    tasks = strict._make_tasks({"full": fixed_splits, "exclude_high_noise": reduced_splits})
    selected = [
        row
        for row in tasks
        if row["dataset_variant"] == "exclude_high_noise"
        and row["task_kind"] == "strict_ft"
    ]
    assert len(selected) == len(fixed_splits) * 3 * 3
    assert {row["lambda"] for row in selected} == set(strict.LAMBDAS)


def _minimal_plan(tmp_path, tasks):
    path = tmp_path / "task_plan.json"
    plan = {
        "format_version": strict.FORMAT_VERSION,
        "num_shards": 2,
        "tasks": tasks,
        "inputs": {"labels": "unused", "replay": "unused"},
        "fixed_split_manifest": str(tmp_path / "manifest.json"),
        "maximum_retries": 2,
        "code_hashes": {},
    }
    path.write_text(json.dumps(plan))
    return path, plan


def test_dry_run_does_not_load_models_or_start_training(monkeypatch, tmp_path):
    task = {"task_id": "x", "assigned_shard": 0}
    plan_path, _plan = _minimal_plan(tmp_path, [task])
    monkeypatch.setattr(strict, "verify_frozen_plan", lambda *_: None)
    monkeypatch.setattr(strict, "load_validation_labels", lambda *_: pytest.fail("labels loaded"))
    monkeypatch.setattr(strict, "load_replay_dataset", lambda *_: pytest.fail("replay loaded"))
    args = argparse.Namespace(
        task_plan=plan_path,
        output=tmp_path / "out",
        shard_index=0,
        num_shards=2,
        max_tasks=None,
        dry_run=True,
        device=None,
        smoke_epochs=None,
    )
    assert strict.run_shard(args) == 0


def test_resume_does_not_repeat_successful_task(monkeypatch, tmp_path):
    task = {"task_id": "x", "assigned_shard": 0}
    plan_path, _plan = _minimal_plan(tmp_path, [task])
    out = tmp_path / "out"
    result = out / strict.RESULTS_DIRNAME / "x.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {"task_id": "x", "assigned_shard": 0, "terminal_status": "success"}
        )
    )
    (tmp_path / "manifest.json").write_text("[]")
    monkeypatch.setattr(strict, "verify_frozen_plan", lambda *_: None)
    monkeypatch.setattr(strict, "load_validation_labels", lambda *_: [])
    monkeypatch.setattr(strict, "load_replay_dataset", lambda *_: [])
    monkeypatch.setattr(strict, "train_task", lambda *_args, **_kwargs: pytest.fail("retrained"))
    args = argparse.Namespace(
        task_plan=plan_path,
        output=out,
        shard_index=0,
        num_shards=2,
        max_tasks=None,
        dry_run=False,
        device="cpu",
        smoke_epochs=None,
    )
    assert strict.run_shard(args) == 0


def test_strict_code_does_not_import_gnn_training_entrypoint():
    sources = "\n".join(
        path.read_text()
        for path in (
            ROOT / "analyse/performance_aware_vae_strict_regression.py",
            ROOT / "analyse/evaluate_performance_aware_vae_strict_regression.py",
        )
    )
    assert "train_and_eval_arch" not in sources
    assert "eval_z_search" not in sources


def test_formal_bo_default_file_remains_at_prefrozen_hash():
    assert sha256_file(ROOT / "bo_phase4.py") == (
        "3fd80c3e49e7f56311930c8451419d97d5676e40dbe53ed995de81601ac0a458"
    )
