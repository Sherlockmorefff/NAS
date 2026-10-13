from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bo_phase4
import cora_layernorm_search_space_runner as runner
import final_eval
from analyse.cora_layernorm_search_space_pilot import (
    summarize_confirmation,
    summarize_history,
)
from dataset_utils import evaluation_cache_key, legacy_cora_context
from eval_utils import DynamicGNN
from layernorm_search_space import (
    NORMALIZATION_LAYER_NORM,
    NORMALIZATION_NONE,
    balanced_initialization_assignment,
    base_candidate_fingerprint,
    candidate_identity,
    compose_search_vector,
    full_candidate_fingerprint,
    matched_training_seed,
    normalization_name,
    top_normalization_statistics,
    validate_resume_normalization_space,
)
from surrogate.accuracy_gp import (
    AccuracyGPPredictor,
    denormalize_search_vector,
    normalize_search_vector,
)


def _base(value: float = 0.0) -> np.ndarray:
    row = np.full(16, value, dtype=np.float32)
    row[:12] = np.clip(row[:12], -2.5, 2.5)
    row[12:] = np.clip(row[12:], 0.0, 1.0)
    return row


def test_old_16d_path_regression() -> None:
    base = _base(0.25)
    normalized = normalize_search_vector(
        base, arch_nz=12, hp_mode="global4", z_bound=2.5
    )
    restored = denormalize_search_vector(
        normalized, arch_nz=12, hp_mode="global4", z_bound=2.5
    )
    np.testing.assert_allclose(restored, base)
    predictor = AccuracyGPPredictor(
        arch_nz=12, hp_mode="global4", z_bound=2.5
    )
    assert predictor.search_dim == 16
    assert predictor.normalized_bounds.shape == (2, 16)


def test_17d_gp_shape_bounds_and_ard_dimension() -> None:
    generator = torch.Generator().manual_seed(7)
    base = torch.rand(6, 16, generator=generator, dtype=torch.double)
    base[:, :12] = base[:, :12] * 5.0 - 2.5
    states = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    predictor = AccuracyGPPredictor.fit_offline(
        torch.cat((base, states), dim=1),
        torch.linspace(0.6, 0.8, 6),
        arch_nz=12,
        hp_mode="global4",
        z_bound=2.5,
        include_normalization=True,
        fit_steps=1,
    )
    assert predictor.search_dim == 17
    assert predictor.train_X_norm.shape == (6, 17)
    assert predictor.normalized_bounds.shape == (2, 17)
    assert predictor.model.covar_module.lengthscale.reshape(-1).numel() == 17


@pytest.mark.parametrize("bad", [0.27, 0.63, 0.81, -1, 2, True, "relu"])
def test_normalization_only_accepts_discrete_values(bad) -> None:
    with pytest.raises(ValueError, match="normalization"):
        normalization_name(bad)
    row = np.concatenate((_base(), np.asarray([0.27], dtype=np.float32)))
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        normalize_search_vector(
            row,
            arch_nz=12,
            hp_mode="global4",
            z_bound=2.5,
            include_normalization=True,
        )


def test_discrete_acquisition_enumerates_both_states(monkeypatch) -> None:
    class Predictor:
        use_conditional_kernel = False
        normalized_bounds = torch.stack((torch.zeros(17), torch.ones(17))).double()

        def make_logei(self, _best_f):
            return lambda X: X[..., -1].reshape(X.shape[0]) + 0.1

        def acquisition_inputs(self, X, *_args, **_kwargs):
            return X

        def predict_batch(self, X, **_kwargs):
            return [{"mean": float(row[-1]), "std": 0.1} for row in X]

    fixed_states = []

    def fake_optimize(_acq, *, fixed_features=None, **_kwargs):
        state = float(fixed_features[16])
        fixed_states.append(state)
        row = torch.full((1, 17), 0.5, dtype=torch.double)
        row[0, -1] = state
        return row, torch.tensor(state)

    monkeypatch.setattr(bo_phase4, "optimize_acqf", fake_optimize)
    monkeypatch.setattr(bo_phase4, "_mask_for_candidate", lambda *_args: [1.0] * 17)
    args = SimpleNamespace(
        hp_mode="global4",
        normalization_space="none_layernorm",
        num_restarts=2,
        raw_samples=4,
        n_extra=2,
        z_bound=2.5,
    )
    candidate, _score, _mask, diagnostics = bo_phase4.optimize_acq(
        Predictor(),
        0.8,
        args,
        None,
        None,
        torch.device("cpu"),
        torch.Generator().manual_seed(3),
    )
    assert fixed_states == [0.0, 1.0]
    assert float(candidate[-1]) in (0.0, 1.0)
    assert diagnostics["none_logei"] != diagnostics["layernorm_logei"]
    assert diagnostics["selected_normalization"] == NORMALIZATION_LAYER_NORM


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_initialization_shares_50_base_points_and_is_25_25(
    seed, monkeypatch
) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    monkeypatch.setattr(
        runner,
        "_decode_architecture_fingerprints",
        lambda _config, _seed, points: [f"arch-{index}" for index in range(len(points))],
    )
    original, expanded, comparison = runner.build_initialization_manifests(
        config, seed
    )
    assert comparison["shared_base_points"] == 50
    assert comparison["original_counts"] == {"none": 50, "layer_norm": 0}
    assert comparison["expanded_counts"] == {"none": 25, "layer_norm": 25}
    assert comparison["matched_training_seeds"] is True
    assert [row["base_vector"] for row in original["rows"]] == [
        row["base_vector"] for row in expanded["rows"]
    ]


def test_balanced_assignment_is_deterministic_and_label_free() -> None:
    fingerprints = [base_candidate_fingerprint(_base(index / 100)) for index in range(50)]
    first = balanced_initialization_assignment(fingerprints, search_seed=10)
    second = balanced_initialization_assignment(fingerprints, search_seed=10)
    assert first == second
    assert list(first.values()).count(NORMALIZATION_NONE) == 25
    assert list(first.values()).count(NORMALIZATION_LAYER_NORM) == 25


@pytest.mark.parametrize(
    ("method", "expected_dim"), (("original", 16), ("expanded", 17))
)
def test_frozen_pilot_command_validates(method, expected_dim) -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    command, _output, _log = runner.build_search_command(
        config, method, 10, "python"
    )
    checked = runner.validate_search_command(command)
    assert checked["search_dim"] == expected_dim
    assert checked["planned_full_evaluations"] == 150


def test_base_full_fingerprints_and_cache_are_isolated() -> None:
    base = _base(0.2)
    base_fp = base_candidate_fingerprint(base)
    none = full_candidate_fingerprint(base_fp, NORMALIZATION_NONE)
    layer = full_candidate_fingerprint(base_fp, NORMALIZATION_LAYER_NORM)
    assert none != layer
    assert candidate_identity(base, NORMALIZATION_NONE)["base_candidate_fingerprint"] == base_fp
    seed = matched_training_seed(base_fp, 0)
    context = legacy_cora_context()
    assert evaluation_cache_key(
        none, context, evaluation_fidelity="full", evaluation_seed=seed
    ) != evaluation_cache_key(
        layer, context, evaluation_fidelity="full", evaluation_seed=seed
    )


def test_training_seed_excludes_normalization() -> None:
    base_fp = base_candidate_fingerprint(_base(0.3))
    assert matched_training_seed(base_fp, 0) == matched_training_seed(base_fp, 0)
    assert full_candidate_fingerprint(base_fp, "none") != full_candidate_fingerprint(
        base_fp, "layer_norm"
    )


def test_history_serializes_base_and_full_vectors() -> None:
    base = _base(0.2)
    full = compose_search_vector(
        base,
        "layer_norm",
        arch_nz=12,
        hp_mode="global4",
        normalization_space="none_layernorm",
    )
    identity = candidate_identity(base, "layer_norm")
    seed = matched_training_seed(identity["base_candidate_fingerprint"], 0)
    args = SimpleNamespace(
        hp_mode="global4",
        normalization_space="none_layernorm",
        z_bound=2.5,
        seed=10,
        gcnii_alpha=0.1,
        gcnii_theta=0.5,
        eval_epochs=150,
        patience=40,
        initial_selection_strategy="schur",
        method_label="expanded",
        deterministic_provenance=None,
        formal_source_id=None,
        formal_config_fingerprint=None,
    )
    result = {
        "hp": {
            "lr": 0.001,
            "dropout": 0.5,
            "hidden_dim": 64,
            "weight_decay": 0.0005,
            "gat_heads": 1,
            "sage_aggr": "mean",
            "gin_eps": 0.0,
        },
        "config": {"operations": ["GCNConv"], "edges": [(0, 1)]},
        "val_acc": 0.8,
        "valid": True,
        "search_seed": 10,
        "candidate_fingerprint": identity["full_candidate_fingerprint"],
        "candidate_eval_seed": seed,
        "evaluation_seed": seed,
        "candidate_evaluation_seed_scheme": bo_phase4.LAYERNORM_TRAINING_SEED_SCHEME,
        "evaluation_fidelity": "full",
    }
    row = bo_phase4.history_record(0, "lhs_init", torch.tensor(full), result, args)
    assert len(row["z_base"]) == 16
    assert len(row["z_search"]) == 17
    assert row["normalization"] == "layer_norm"
    assert row["base_candidate_fingerprint"] == identity["base_candidate_fingerprint"]
    assert row["full_candidate_fingerprint"] == identity["full_candidate_fingerprint"]


def test_resume_rejects_normalization_space_mismatch(tmp_path: Path) -> None:
    validate_resume_normalization_space(
        {"normalization_space": "none_layernorm"}, "none_layernorm"
    )
    with pytest.raises(ValueError, match="normalization_space mismatch"):
        validate_resume_normalization_space(
            {"normalization_space": "none"}, "none_layernorm"
        )
    x = torch.zeros(4, 17, dtype=torch.double)
    x[:, -1] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    predictor = AccuracyGPPredictor.fit_offline(
        x,
        torch.tensor([0.6, 0.7, 0.65, 0.75]),
        arch_nz=12,
        hp_mode="global4",
        z_bound=2.5,
        include_normalization=True,
        fit_steps=1,
    )
    checkpoint = tmp_path / "expanded.pt"
    predictor.save(checkpoint)
    with pytest.raises(ValueError, match="metadata mismatch"):
        AccuracyGPPredictor.load(
            checkpoint, expected={"include_normalization": False}
        )


def test_final_confirmation_uses_common_seed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "seeds.json"
    seeds = [11, 22, 33, 44, 55]
    manifest.write_text(json.dumps({"seeds": seeds}), encoding="utf-8")
    args = SimpleNamespace(
        top_k=1,
        eval_epochs=300,
        patience=80,
        hp_mode="global4",
        n_replicates=5,
        n_seeds=5,
        legacy_sequential_seeds=False,
        common_seed_manifest=str(manifest),
        final_base_seed=20260816,
        method_label="method",
        search_seed=10,
        sort_by="val_acc",
        history_steps=None,
        include_baselines=False,
    )
    final_eval._validate_args(args)
    left = {"candidate_fingerprint": "a" * 64}
    right = {"candidate_fingerprint": "b" * 64}
    assert [final_eval._seed_for_replicate(args, left, i) for i in range(5)] == seeds
    assert [final_eval._seed_for_replicate(args, right, i) for i in range(5)] == seeds


def test_confirmation_plan_has_six_winners_plus_archive_best() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    plan = runner.build_confirmation_plan(config)
    assert plan["candidate_count"] == 7
    assert plan["replicates_per_candidate"] == 5
    assert plan["training_count"] == 35
    assert plan["nas_training_count"] == 900
    assert plan["total_protocol_training_count"] == 935
    assert [row["source"] for row in plan["slots"]].count("nas_winner") == 6
    assert [row["source"] for row in plan["slots"]].count("historical_archive") == 1
    assert all(
        row["confirmation_seeds"] == plan["common_confirmation_seeds"]
        for row in plan["slots"]
    )


def test_historical_best_is_validation_only_and_frozen() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG)
    candidate = runner.load_historical_best(config)
    assert candidate["selection_metric"] == "validation_accuracy"
    assert candidate["selection_test_access"] == "PROHIBITED"
    assert candidate["historical_validation_accuracy"] == 0.828
    assert candidate["normalization"] == NORMALIZATION_NONE
    assert candidate["candidate_fingerprint"] == (
        "421c269719718dc5cc95f572e054d70bee9ee0d7431dfb09854f0e7b5231c441"
    )


def test_confirmation_answers_four_final_questions() -> None:
    seeds = [11, 22, 33, 44, 55]

    def candidate(method, search_seed, candidate_id, normalization, val, test):
        return {
            "method": method,
            "search_seed": search_seed,
            "candidate_id": candidate_id,
            "normalization": normalization,
            "per_seed": [
                {
                    "seed": seed,
                    "validation_accuracy": val,
                    "test_accuracy": test,
                }
                for seed in seeds
            ],
        }

    rows = [
        candidate("original", seed, f"original-{seed}", "none", 0.81, 0.80)
        for seed in (10, 11, 12)
    ]
    rows.extend(
        [
            candidate("expanded", 10, "expanded-10", "layer_norm", 0.840, 0.830),
            candidate("expanded", 11, "expanded-11", "layer_norm", 0.837, 0.827),
            candidate("expanded", 12, "expanded-12", "none", 0.835, 0.825),
            candidate("archive_best", 0, "archive", "none", 0.825, 0.815),
        ]
    )
    summary = summarize_confirmation(rows, seeds)
    assert summary["selected_expanded_candidate_id"] == "expanded-10"
    assert summary["layernorm_effective"] is True
    assert summary["beats_original"] is True
    assert summary["beats_historical_archive_best"] is True
    assert summary["breaks_historical_ceiling"] is True
    assert summary["test_used_for_selection"] is False


def _history_record(index: int, normalization: str, value: float) -> dict:
    base_fp = f"{index + 1:064x}"
    return {
        "evaluation_fidelity": "full",
        "type": "lhs_init" if index < 50 else "bo",
        "valid": True,
        "val_acc": value,
        "normalization": normalization,
        "base_candidate_fingerprint": base_fp,
        "full_candidate_fingerprint": full_candidate_fingerprint(base_fp, normalization),
        "proposal_attempts": [],
    }


def test_result_parser_and_top10_normalization_statistics() -> None:
    rows = [
        _history_record(
            index,
            NORMALIZATION_LAYER_NORM if index % 2 else NORMALIZATION_NONE,
            0.60 + index / 1000,
        )
        for index in range(150)
    ]
    summary = summarize_history(rows)
    assert summary["evaluation_count"] == 150
    assert summary["best_found_evaluation"] == 150
    assert summary["overall_normalization"]["top_k"] == 10
    stats = top_normalization_statistics(rows, top_k=10)
    assert stats["top_layernorm_count"] == 5
    assert stats["top_layernorm_fraction"] == 0.5


def test_layernorm_semantics_match_completed_experiment() -> None:
    model = DynamicGNN(
        {
            "operations": ["GCNConv", "Identity", "SAGEConv"],
            "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
        },
        in_ch=8,
        out_ch=3,
        hidden_dim=16,
        component_variant="layer_norm",
        activation_variant="relu",
    )
    assert set(model.layer_norms.keys()) == {"0", "2"}
    assert "1" not in model.layer_norms
