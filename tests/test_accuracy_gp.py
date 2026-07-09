from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")

from surrogate.accuracy_gp import (
    AccuracyGPPredictor,
    denormalize_search_vector,
    normalize_search_vector,
)


def _synthetic_data(count: int = 12):
    generator = torch.Generator().manual_seed(7)
    X = torch.rand(count, 16, generator=generator)
    X[:, :12] = X[:, :12] * 5.0 - 2.5
    y = 0.7 + 0.08 * torch.sin(X[:, 0]) - 0.03 * X[:, 12]
    return X, y


def test_fixed_normalization_round_trip_and_validation():
    X, _ = _synthetic_data(3)
    normalized = normalize_search_vector(X, arch_nz=12, hp_mode="global4", z_bound=2.5)
    restored = denormalize_search_vector(normalized, arch_nz=12, hp_mode="global4", z_bound=2.5)
    assert normalized.device == X.device
    assert normalized.dtype == X.dtype
    assert torch.allclose(X, restored, atol=1e-6)
    with pytest.raises(ValueError, match="last dimension"):
        normalize_search_vector(torch.zeros(3, 15), arch_nz=12, hp_mode="global4", z_bound=2.5)
    bad = X.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        normalize_search_vector(bad, arch_nz=12, hp_mode="global4", z_bound=2.5)


def test_fit_save_load_append_and_warm_refit(tmp_path):
    X, y = _synthetic_data()
    predictor = AccuracyGPPredictor.fit_offline(
        X, y, arch_nz=12, hp_mode="global4", z_bound=2.5,
        metadata={"dataset": "Cora", "metric": "val_acc", "eval_epochs": 100,
                  "patience": 20, "vae_checkpoint": "vae.pt", "vae_version": "v1", "seed": 42},
        fit_steps=3,
    )
    before = predictor.predict(X[0])
    assert predictor.train_size == 12
    assert all(math.isfinite(before[key]) for key in ("mean", "std", "lower_95", "upper_95"))
    checkpoint = tmp_path / "gp.pt"
    predictor.save(checkpoint)
    loaded = AccuracyGPPredictor.load(
        checkpoint,
        expected={"arch_nz": 12, "hp_mode": "global4", "search_dim": 16, "z_bound": 2.5},
    )
    after = loaded.predict(X[0])
    assert after == pytest.approx(before, rel=1e-7, abs=1e-7)

    loaded.append_observation(X[0] * 0.95, float(y[0]))
    loaded.refit(optimize=False)
    assert loaded.train_size == 13
    loaded.append_observation(X[1] * 0.95, float(y[1]))
    loaded.refit(optimize=True, steps=2)
    assert loaded.train_size == 14
    assert math.isfinite(loaded.predict(X[2])["mean"])


def test_checkpoint_metadata_mismatch_is_fatal(tmp_path):
    X, y = _synthetic_data(6)
    predictor = AccuracyGPPredictor.fit_offline(
        X, y, arch_nz=12, hp_mode="global4", z_bound=2.5, fit_steps=2,
    )
    checkpoint = tmp_path / "gp.pt"
    predictor.save(checkpoint)
    with pytest.raises(ValueError, match="metadata mismatch"):
        AccuracyGPPredictor.load(checkpoint, expected={"z_bound": 3.0})


def test_holdout_vectors_are_not_appendable():
    X, y = _synthetic_data(10)
    predictor = AccuracyGPPredictor.fit_offline(
        X[:8], y[:8], arch_nz=12, hp_mode="global4", z_bound=2.5,
        holdout_X_raw=X[8:], holdout_Y=y[8:], fit_steps=2,
    )
    with pytest.raises(ValueError, match="fixed holdout"):
        predictor.append_observation(X[8], float(y[8]))


def test_conditional_kernel_masks_survive_online_update():
    generator = torch.Generator().manual_seed(11)
    X = torch.rand(8, 19, generator=generator)
    X[:, :12] = X[:, :12] * 5.0 - 2.5
    y = 0.65 + 0.1 * X[:, 0]
    masks = torch.ones_like(X)
    masks[:, 16:] = 0.0
    predictor = AccuracyGPPredictor.fit_offline(
        X, y, arch_nz=12, hp_mode="hybrid_cond7", z_bound=2.5,
        use_conditional_kernel=True, condition_masks=masks, fit_steps=2,
    )
    assert predictor.model.covar_module.base_kernel.ard_num_dims == 2 * predictor.search_dim
    prediction = predictor.predict(X[0], condition_mask=masks[0])
    assert math.isfinite(prediction["std"])
    predictor.append_observation(X[1] * 0.99, float(y[1]), condition_mask=masks[1])
    predictor.refit(optimize=False)
    assert predictor.train_size == 9
    assert predictor.train_condition_masks.shape == (9, 19)
