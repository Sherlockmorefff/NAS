from __future__ import annotations

import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
from gpytorch.kernels import RBFKernel, ScaleKernel

from eval_utils import isolated_rng
from surrogate.accuracy_gp import AccuracyGPPredictor, normalize_search_vector
from surrogate.dkl_accuracy_gp import (
    DKLAccuracyGPPredictor,
    DeepFeatureKernel,
    SmallFeatureExtractor,
)
from surrogate.predictor_factory import fit_accuracy_predictor, load_accuracy_predictor


def _data(count: int = 9):
    generator = torch.Generator().manual_seed(314)
    X = torch.rand(count, 16, generator=generator, dtype=torch.double)
    X[:, :12] = X[:, :12] * 5.0 - 2.5
    y = 0.72 + 0.04 * torch.sin(1.7 * X[:, 0]) - 0.02 * X[:, 12]
    return X, y


def _fit(X=None, y=None, **overrides):
    if X is None or y is None:
        X, y = _data()
    options = {
        "arch_nz": 12,
        "hp_mode": "global4",
        "z_bound": 2.5,
        "hidden_dim": 12,
        "feature_dim": 4,
        "activation": "silu",
        "lr": 0.01,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "init_steps": 6,
        "refit_steps": 3,
        "early_stopping_patience": 4,
        "min_delta": 1e-7,
        "fit_steps": 6,
        "training_seed": 12345,
    }
    options.update(overrides)
    return DKLAccuracyGPPredictor.fit_offline(X, y, **options)


@pytest.mark.parametrize("feature_dim", [4, 8])
@pytest.mark.parametrize("activation", ["silu", "relu"])
def test_small_feature_extractor_shape_dtype_and_structure(feature_dim, activation):
    extractor = SmallFeatureExtractor(16, hidden_dim=11, feature_dim=feature_dim, activation=activation)
    output = extractor(torch.ones(3, 16, dtype=torch.double))
    assert output.shape == (3, feature_dim)
    assert output.dtype == torch.double
    assert not any(isinstance(module, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm)) for module in extractor.modules())


def test_small_feature_extractor_defaults_are_16_to_32_to_8():
    extractor = SmallFeatureExtractor(16)
    linears = [module for module in extractor.modules() if isinstance(module, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (16, 32), (32, 8),
    ]
    assert any(isinstance(module, torch.nn.SiLU) for module in extractor.modules())
    assert all(parameter.dtype == torch.double for parameter in extractor.parameters())
    assert not any(isinstance(module, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm)) for module in extractor.modules())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"feature_dim": 3}, "feature_dim"), ({"activation": "gelu"}, "activation"), ({"hidden_dim": 0}, "hidden_dim")],
)
def test_small_feature_extractor_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SmallFeatureExtractor(16, **kwargs)


def test_deep_feature_kernel_is_symmetric_finite_and_backpropagates():
    kernel = DeepFeatureKernel(16, hidden_dim=10, feature_dim=4).double()
    X = torch.rand(5, 16, dtype=torch.double, requires_grad=True)
    covariance = kernel(X).to_dense()
    assert covariance.shape == (5, 5)
    assert torch.isfinite(covariance).all()
    assert torch.allclose(covariance, covariance.T, atol=1e-10)
    assert isinstance(kernel.latent_kernel, ScaleKernel)
    assert isinstance(kernel.latent_kernel.base_kernel, RBFKernel)
    assert kernel.latent_kernel.base_kernel.ard_num_dims == 4
    covariance.sum().backward()
    gradients = [parameter.grad for parameter in kernel.feature_extractor.parameters()]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_dkl_fit_predict_append_refit_logei_and_feature_update():
    X, y = _data()
    seed = 2026
    y_mean = float(y.mean())
    y_std = float(y.std(unbiased=False))
    skeleton = DKLAccuracyGPPredictor(
        arch_nz=12, hp_mode="global4", z_bound=2.5,
        y_mean=y_mean, y_std=y_std, hidden_dim=12, feature_dim=4,
        init_steps=6, refit_steps=3, early_stopping_patience=4,
    )
    skeleton.train_X_norm = normalize_search_vector(
        X, arch_nz=12, hp_mode="global4", z_bound=2.5,
    ).double()
    skeleton.train_Y_norm = ((y - y_mean) / y_std).reshape(-1, 1)
    skeleton.train_observation_counts = torch.ones(len(X), dtype=torch.long)
    with isolated_rng(seed, torch.device("cpu")):
        initial_model = skeleton._build_model()
    initial_features = {
        key: value.detach().clone()
        for key, value in initial_model.covar_module.feature_extractor.state_dict().items()
    }
    predictor = _fit(X, y, training_seed=seed)
    learned_features = predictor.model.covar_module.feature_extractor.state_dict()
    assert any(not torch.allclose(initial_features[key], learned_features[key]) for key in initial_features)
    prediction = predictor.predict(X[0])
    assert all(math.isfinite(prediction[key]) for key in ("mean", "std", "lower_95", "upper_95"))
    assert prediction["std"] >= 0.0
    before_size = predictor.train_size
    before_features = {
        key: value.detach().clone()
        for key, value in predictor.model.covar_module.feature_extractor.state_dict().items()
    }
    initial_y_stats = (predictor.y_mean, predictor.y_std)
    predictor.append_observation(X[0], float(y[0]) + 0.01)
    assert predictor.train_size == before_size
    predictor.refit(optimize=False, training_seed=99)
    assert (predictor.y_mean, predictor.y_std) == initial_y_stats
    assert all(torch.equal(value, predictor.model.covar_module.feature_extractor.state_dict()[key]) for key, value in before_features.items())
    new_x = X[0].clone()
    new_x[0] = torch.clamp(new_x[0] + 0.01, max=2.5)
    predictor.append_observation(new_x, float(y[0]) + 0.005)
    assert predictor.train_size == before_size + 1
    predictor.refit(optimize=False, training_seed=100)
    assert predictor.train_size == before_size + 1
    assert (predictor.y_mean, predictor.y_std) == initial_y_stats
    candidate = predictor.acquisition_inputs(X[1])
    score = predictor.make_logei(float(y.max()))(candidate.unsqueeze(1)).reshape(-1)
    assert torch.isfinite(score).all()
    predictor.model.zero_grad(set_to_none=True)
    candidate = candidate.detach().requires_grad_(True)
    predictor.make_logei(float(y.max()))(candidate.unsqueeze(1)).sum().backward()
    assert candidate.grad is not None and torch.isfinite(candidate.grad).all()
    feature_gradients = [
        parameter.grad
        for parameter in predictor.model.covar_module.feature_extractor.parameters()
    ]
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in feature_gradients
    )


def test_dkl_best_state_early_stopping_determinism_and_rng_isolation():
    X, y = _data(7)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    first = _fit(
        X, y, fit_steps=6, min_delta=1e9, early_stopping_patience=1,
        training_seed=77,
    )
    after = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    assert before[0] == after[0]
    assert np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])
    assert first.training_summary["early_stopped"] is True
    assert first.training_summary["best_step"] == 0
    second = _fit(
        X, y, fit_steps=6, min_delta=1e9, early_stopping_patience=1,
        training_seed=77,
    )
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key])
    assert first.predict(X[2]) == pytest.approx(second.predict(X[2]), abs=1e-12)


def test_dkl_joint_fit_calls_gradient_clipping(monkeypatch):
    X, y = _data(6)
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters, max_norm, *args, **kwargs):
        calls.append(float(max_norm))
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    _fit(X, y, fit_steps=3, grad_clip=1.25)
    assert calls == [1.25, 1.25, 1.25]


def test_dkl_nonfinite_gradient_fails_clearly(monkeypatch):
    X, y = _data(6)

    def poison_gradients(parameters, max_norm, *args, **kwargs):
        for parameter in parameters:
            parameter.grad.fill_(float("nan"))
        return torch.tensor(float("nan"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", poison_gradients)
    with pytest.raises(RuntimeError, match="non-finite gradient during DKL refit"):
        _fit(X, y, fit_steps=2)


def test_dkl_weight_decay_is_limited_to_feature_extractor_parameters():
    X, y = _data(6)
    predictor = _fit(X, y, fit_steps=2, weight_decay=1e-4)
    groups = predictor._optimizer_groups(predictor.model)
    feature_ids = {
        id(parameter)
        for parameter in predictor.model.covar_module.feature_extractor.parameters()
    }
    decayed_ids = {id(parameter) for parameter in groups[0]["params"]}
    unregularized_ids = {id(parameter) for parameter in groups[1]["params"]}
    assert groups[0]["weight_decay"] == pytest.approx(1e-4)
    assert groups[1]["weight_decay"] == pytest.approx(0.0)
    assert decayed_ids == feature_ids
    assert decayed_ids.isdisjoint(unregularized_ids)
    assert id(predictor.model.likelihood.raw_noise) in unregularized_ids
    assert id(predictor.model.covar_module.latent_kernel.base_kernel.raw_lengthscale) in unregularized_ids


def test_dkl_conditional_inputs_and_checkpoint_type_safety(tmp_path):
    generator = torch.Generator().manual_seed(9)
    X = torch.rand(7, 19, generator=generator, dtype=torch.double)
    X[:, :12] = X[:, :12] * 5.0 - 2.5
    y = 0.7 + 0.03 * X[:, 0]
    masks = torch.ones_like(X)
    masks[:, 17:] = 0.0
    predictor = _fit(
        X, y, hp_mode="hybrid_cond7", use_conditional_kernel=True,
        condition_masks=masks,
    )
    assert predictor.feature_input_dim == 38
    assert predictor.acquisition_inputs(X[0], masks[0]).shape == (1, 38)
    before = predictor.predict(X[1], condition_mask=masks[1])
    checkpoint = tmp_path / "dkl.pt"
    predictor.save(checkpoint)
    loaded = DKLAccuracyGPPredictor.load(
        checkpoint,
        expected={"hp_mode": "hybrid_cond7", "search_dim": 19, "feature_dim": 4},
    )
    assert loaded.predict(X[1], condition_mask=masks[1]) == pytest.approx(before, abs=1e-10)
    assert any(key.startswith("covar_module.feature_extractor") for key in loaded.model.state_dict())
    with pytest.raises(ValueError, match="expected 'exact_gp'.*actual 'dkl_gp'"):
        AccuracyGPPredictor.load(checkpoint)
    with pytest.raises(ValueError, match="metadata mismatch"):
        DKLAccuracyGPPredictor.load(checkpoint, expected={"activation": "relu"})
    payload = torch.load(checkpoint, weights_only=False)
    payload["feature_dim"] = 8
    mismatched = tmp_path / "dkl_mismatched_state.pt"
    torch.save(payload, mismatched)
    with pytest.raises(ValueError, match="feature architecture metadata"):
        DKLAccuracyGPPredictor.load(mismatched)
    payload = torch.load(checkpoint, weights_only=False)
    payload["activation"] = "relu"
    activation_mismatch = tmp_path / "dkl_activation_mismatch.pt"
    torch.save(payload, activation_mismatch)
    with pytest.raises(ValueError, match="feature architecture metadata"):
        DKLAccuracyGPPredictor.load(activation_mismatch)
    bad_masks = masks.clone()
    bad_masks[0, 0] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _fit(
            X, y, hp_mode="hybrid_cond7", use_conditional_kernel=True,
            condition_masks=bad_masks,
        )

    exact = AccuracyGPPredictor.fit_offline(
        X[:, :16], y, arch_nz=12, hp_mode="global4", z_bound=2.5, fit_steps=2,
    )
    exact_path = tmp_path / "exact.pt"
    exact.save(exact_path)
    with pytest.raises(ValueError, match="expected 'dkl_gp'.*actual 'exact_gp'"):
        DKLAccuracyGPPredictor.load(exact_path)
    with pytest.raises(ValueError, match="expected 'dkl_gp'.*actual 'exact_gp'"):
        load_accuracy_predictor(exact_path, surrogate_type="dkl_gp")


def test_factory_defaults_to_existing_exact_predictor():
    X, y = _data(6)
    predictor = fit_accuracy_predictor(
        X, y, arch_nz=12, hp_mode="global4", z_bound=2.5, fit_steps=2,
    )
    assert type(predictor) is AccuracyGPPredictor
    dkl = fit_accuracy_predictor(
        X, y, surrogate_type="dkl_gp", arch_nz=12, hp_mode="global4", z_bound=2.5,
        hidden_dim=8, feature_dim=4, init_steps=2, refit_steps=2,
        early_stopping_patience=2, fit_steps=2, training_seed=4,
    )
    assert isinstance(dkl, DKLAccuracyGPPredictor)


def test_legacy_exact_checkpoint_without_surrogate_type_still_loads(tmp_path):
    X, y = _data(6)
    exact = AccuracyGPPredictor.fit_offline(
        X, y, arch_nz=12, hp_mode="global4", z_bound=2.5, fit_steps=2,
    )
    payload = exact.checkpoint_payload()
    payload.pop("surrogate_type")
    checkpoint = tmp_path / "legacy_exact.pt"
    torch.save(payload, checkpoint)
    loaded = AccuracyGPPredictor.load(checkpoint)
    assert loaded.predict(X[0]) == pytest.approx(exact.predict(X[0]), abs=1e-10)
