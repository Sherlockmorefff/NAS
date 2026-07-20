from __future__ import annotations

import csv
import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")

from analyse import diagnose_dkl_uncertainty as diagnosis
from surrogate.accuracy_gp import AccuracyGPPredictor
from surrogate.dkl_accuracy_gp import DKLAccuracyGPPredictor


def _data(count: int = 7):
    generator = torch.Generator().manual_seed(8675309)
    X = torch.rand(count, 16, generator=generator, dtype=torch.double)
    X[:, :12] = X[:, :12] * 5.0 - 2.5
    y = 0.71 + 0.025 * torch.sin(X[:, 0]) + 0.01 * X[:, 12]
    return X, y


def _fit_predictors():
    X, y = _data()
    exact = AccuracyGPPredictor.fit_offline(
        X,
        y,
        arch_nz=12,
        hp_mode="global4",
        z_bound=2.5,
        fit_steps=2,
    )
    dkl = DKLAccuracyGPPredictor.fit_offline(
        X,
        y,
        arch_nz=12,
        hp_mode="global4",
        z_bound=2.5,
        hidden_dim=8,
        feature_dim=4,
        init_steps=3,
        refit_steps=2,
        early_stopping_patience=2,
        fit_steps=3,
        training_seed=123,
    )
    return X, y, exact, dkl


@pytest.mark.parametrize("surrogate", ["exact_gp", "dkl_gp"])
def test_observed_posterior_units_and_noise_are_correct(surrogate):
    X, _, exact, dkl = _fit_predictors()
    predictor = exact if surrogate == "exact_gp" else dkl
    result = diagnosis.posterior_diagnostic(predictor, X[0])
    production = predictor.predict(X[0])
    noise = float(predictor.likelihood.noise.detach().reshape(-1)[0])

    assert result["observed_variance_standardized"] + 1e-12 >= result[
        "latent_variance_standardized"
    ]
    assert (
        result["observed_variance_standardized"]
        - result["latent_variance_standardized"]
    ) == pytest.approx(noise, rel=1e-6, abs=1e-10)
    assert result["likelihood_noise_contribution_variance_standardized"] == pytest.approx(
        noise, rel=1e-6, abs=1e-10
    )
    assert result["latent_variance_raw"] == pytest.approx(
        result["latent_variance_standardized"] * predictor.y_std**2
    )
    assert result["latent_std_raw"] == pytest.approx(
        result["latent_std_standardized"] * predictor.y_std
    )
    assert result["observed_variance_raw"] == pytest.approx(
        result["observed_variance_standardized"] * predictor.y_std**2
    )
    assert result["likelihood_noise_contribution_std_raw"] == pytest.approx(
        math.sqrt(noise) * predictor.y_std, rel=1e-6, abs=1e-10
    )
    assert result["latent_mean_raw"] == pytest.approx(production["mean"], abs=1e-10)
    assert result["latent_std_raw"] == pytest.approx(production["std"], abs=1e-10)


def test_diagnostic_state_is_read_only_rng_neutral_and_acquisition_neutral():
    X, y, _, predictor = _fit_predictors()
    before_prediction = predictor.predict(X[1])
    candidate = predictor.acquisition_inputs(X[2]).unsqueeze(1)
    acquisition = predictor.make_logei(float(y.max()))
    before_logei = acquisition(candidate).detach().clone()
    before_state = {
        key: value.detach().clone() for key, value in predictor.model.state_dict().items()
    }
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    rng_before = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )

    hyperparameters, features = diagnosis.diagnostic_state(
        predictor,
        search_seed=0,
        history_step=49,
        node="initial_fit",
    )
    posterior = diagnosis.posterior_diagnostic(predictor, X[1])

    rng_after = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )
    after_prediction = predictor.predict(X[1])
    after_logei = acquisition(candidate).detach().clone()
    assert before_prediction == pytest.approx(after_prediction, abs=1e-12)
    assert torch.allclose(before_logei, after_logei, rtol=1e-12, atol=1e-12)
    assert all(
        torch.equal(value, predictor.model.state_dict()[key])
        for key, value in before_state.items()
    )
    assert rng_before[0] == rng_after[0]
    assert np.array_equal(rng_before[1][1], rng_after[1][1])
    assert rng_before[1][2:] == rng_after[1][2:]
    assert torch.equal(rng_before[2], rng_after[2])
    assert hyperparameters["training_requested_steps"] == 3
    assert hyperparameters["training_executed_steps"] >= 1
    assert math.isfinite(hyperparameters["training_gradient_norm"])
    assert math.isfinite(hyperparameters["training_feature_gradient_norm"])
    assert features["has_feature_extractor"] is True
    assert features["feature_has_nonfinite"] is False
    assert all(
        math.isfinite(float(value))
        for value in json.loads(features["feature_dim_std_json"])
    )
    assert math.isfinite(posterior["observed_std_raw"])


def test_pairwise_feature_distance_statistics_are_mechanical():
    features = torch.tensor(
        [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=torch.double
    )
    statistics = diagnosis.pairwise_distance_statistics(features)
    assert statistics["feature_has_nonfinite"] is False
    assert statistics["unique_feature_ratio"] == pytest.approx(1.0)
    assert statistics["pair_distance_q50"] == pytest.approx(5.0)
    assert statistics["nearest_distance_q01"] == pytest.approx(5.0)
    assert statistics["nearest_distance_q95"] == pytest.approx(5.0)
    assert statistics["pair_count_lt_1e_8"] == 0

    duplicates = diagnosis.pairwise_distance_statistics(
        torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.double)
    )
    assert duplicates["unique_feature_ratio"] == pytest.approx(2.0 / 3.0)
    assert duplicates["pair_count_lt_1e_8"] == 1
    assert duplicates["pair_count_lt_1e_6"] == 1
    assert duplicates["pair_count_lt_1e_4"] == 1


class _FakePredictor:
    def __init__(self, values):
        self.values = list(values)
        self.train_size = len(values)

    def append_observation(self, X_raw, y, condition_mask=None):
        self.values.append(float(y))
        self.train_size += 1

    def refit(self, *, optimize, steps=None, training_seed=None):
        return None


def _records(count: int = 7):
    return [
        {
            "step": step,
            "valid": step != 4,
            "z_search": [0.01 * (step + index) for index in range(16)],
            "condition_mask_full": [1.0] * 16,
            "val_acc": 0.70 + 0.01 * step,
        }
        for step in range(count)
    ]


def _diagnostic_args():
    return SimpleNamespace(
        n_init=3,
        gp_refit_every=2,
        gp_refit_steps=2,
        dkl_refit_steps=2,
        use_conditional_kernel=False,
    )


def test_diagnostic_prequential_replay_has_no_future_leakage(monkeypatch):
    fitted = []

    def fake_fit(initial, args, search_seed):
        values = [row["val_acc"] for row in initial]
        predictors = {
            "exact_gp": _FakePredictor(values),
            "dkl_gp": _FakePredictor(values),
        }
        fitted.append(predictors)
        return predictors, {"exact_gp": 0.0, "dkl_gp": 0.0}

    def fake_posterior(predictor, X_raw, condition_mask=None):
        mean = sum(predictor.values) / len(predictor.values)
        return {
            "latent_mean_raw": mean,
            "observed_mean_raw": mean,
            "latent_std_raw": 0.02,
            "observed_std_raw": 0.03,
            "likelihood_noise_contribution_std_raw": 0.01,
        }

    def fake_state(predictor, *, search_seed, history_step, node):
        base = {
            "search_seed": search_seed,
            "history_step": history_step,
            "node": node,
            "train_size": predictor.train_size,
        }
        return dict(base), dict(base)

    monkeypatch.setattr(diagnosis, "_fit_predictors", fake_fit)
    monkeypatch.setattr(diagnosis, "posterior_diagnostic", fake_posterior)
    monkeypatch.setattr(diagnosis, "diagnostic_state", fake_state)

    rows_a, hyper_a, _ = diagnosis.run_history_diagnostics(
        "synthetic", _records(), 11, _diagnostic_args()
    )
    changed = _records()
    changed[-1]["val_acc"] = 0.01
    rows_b, _, _ = diagnosis.run_history_diagnostics(
        "synthetic", changed, 11, _diagnostic_args()
    )
    exact_a = [row for row in rows_a if row["surrogate_type"] == "exact_gp"]
    exact_b = [row for row in rows_b if row["surrogate_type"] == "exact_gp"]
    assert [row["latent_mean_raw"] for row in exact_a] == pytest.approx(
        [row["latent_mean_raw"] for row in exact_b]
    )
    assert [row["train_size_before"] for row in exact_a] == [3, 4, 4, 5]
    assert fitted[0]["exact_gp"].values == pytest.approx([0.70, 0.71, 0.72, 0.73, 0.75, 0.76])
    assert fitted[0]["dkl_gp"].values == pytest.approx(fitted[0]["exact_gp"].values)
    assert [(row["node"], row["history_step"]) for row in hyper_a[::2]] == [
        ("initial_fit", 2),
        ("optimized_refit", 5),
        ("final", 6),
    ]


def test_diagnostic_cli_writes_all_outputs_with_small_real_models(tmp_path):
    history = tmp_path / "history.json"
    rows = []
    for step in range(6):
        rows.append(
            {
                "step": step,
                "valid": True,
                "z_search": [0.01 * (step + index) for index in range(16)],
                "val_acc": 0.70 + 0.01 * step,
                "hp_mode": "global4",
                "search_dim": 16,
                "search_seed": 0,
                "condition_mask_vector": [1.0] * 4,
                "operations": ["GCNConv"],
            }
        )
    history.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "diagnostics"
    diagnosis.main(
        [
            "--history_paths",
            str(history),
            "--output",
            str(output),
            "--n_init",
            "4",
            "--max_records",
            "6",
            "--gp_refit_every",
            "2",
            "--gp_init_steps",
            "2",
            "--gp_refit_steps",
            "2",
            "--dkl_hidden_dim",
            "8",
            "--dkl_feature_dim",
            "4",
            "--dkl_init_steps",
            "3",
            "--dkl_refit_steps",
            "2",
            "--dkl_early_stopping_patience",
            "2",
        ]
    )
    assert {path.name for path in output.iterdir()} == {
        "diagnostic_config.json",
        "hyperparameter_trace.csv",
        "feature_trace.csv",
        "uncertainty_predictions.csv",
        "uncertainty_summary.csv",
    }
    with (output / "uncertainty_predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 4
    assert all(
        float(row["observed_variance_standardized"])
        >= float(row["latent_variance_standardized"])
        for row in predictions
    )
    with (output / "uncertainty_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert {row["surrogate_type"] for row in summary} == {"exact_gp", "dkl_gp"}
    config = json.loads((output / "diagnostic_config.json").read_text(encoding="utf-8"))
    assert config["diagnostics_are_read_only"] is True
    assert "noise added exactly once" in config["field_units"]["observed posterior"]
