from __future__ import annotations

import csv
import json
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")

from analyse import compare_exact_vs_dkl_prequential as comparison


def _records(count=7):
    rows = []
    for step in range(count):
        rows.append(
            {
                "step": step,
                "valid": step != 4,
                "z_search": [0.01 * (step + index) for index in range(16)],
                "val_acc": 0.70 + 0.01 * step,
                "hp_mode": "global4",
                "search_dim": 16,
                "search_seed": 11,
                "condition_mask_vector": [1.0] * 4,
                "operations": ["GCNConv"],
            }
        )
    return rows


class _FakePredictor:
    def __init__(self, initial):
        self.values = list(initial)
        self.train_size = len(initial)

    def predict(self, z, condition_mask=None):
        mean = sum(self.values) / len(self.values)
        return {"mean": mean, "std": 0.02, "lower_95": mean - 0.0392, "upper_95": mean + 0.0392}

    def append_observation(self, z, actual, condition_mask=None):
        self.values.append(actual)
        self.train_size += 1

    def refit(self, optimize, steps, training_seed=None):
        return None


def _args():
    return SimpleNamespace(
        n_init=3, gp_refit_every=2, gp_refit_steps=2, dkl_refit_steps=2,
        use_conditional_kernel=False,
    )


def test_strict_history_loading_and_hp_only_mask_expansion(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps(_records()), encoding="utf-8")
    records, seed = comparison.load_ordered_history(
        path, n_init=3, max_records=7, arch_nz=12, hp_mode="global4",
    )
    assert seed == 11
    assert len(records[0]["condition_mask_full"]) == 16
    assert records[0]["condition_mask_full"][:12] == [1.0] * 12
    bad = _records()
    bad[1]["condition_mask_vector"] = [1.0] * 5
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="condition mask has length"):
        comparison.load_ordered_history(
            path, n_init=3, max_records=7, arch_nz=12, hp_mode="global4",
        )


def test_history_loading_strictly_applies_max_records(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps(_records(8)), encoding="utf-8")
    records, _ = comparison.load_ordered_history(
        path, n_init=3, max_records=5, arch_nz=12, hp_mode="global4",
    )
    assert [record["step"] for record in records] == [0, 1, 2, 3, 4]


def test_prequential_predictions_use_only_past_and_invalid_is_not_trained(monkeypatch):
    records = _records()

    def fake_fit(initial, args, seed):
        values = [row["val_acc"] for row in initial]
        return {"exact_gp": _FakePredictor(values), "dkl_gp": _FakePredictor(values)}, {"exact_gp": 0.0, "dkl_gp": 0.0}

    monkeypatch.setattr(comparison, "_fit_predictors", fake_fit)
    rows, _ = comparison.run_history("synthetic", records, 11, _args())
    exact = [row for row in rows if row["surrogate_type"] == "exact_gp"]
    dkl = [row for row in rows if row["surrogate_type"] == "dkl_gp"]
    assert exact[0]["train_size_before"] == 3
    assert exact[0]["pred_mean"] == pytest.approx((0.70 + 0.71 + 0.72) / 3)
    assert [row["step"] for row in exact] == [3, 4, 5, 6]
    assert [row["train_size_before"] for row in exact] == [3, 4, 4, 5]
    assert [row["train_size_after"] for row in exact] == [4, 4, 5, 6]
    assert [row["train_size_before"] for row in dkl] == [3, 4, 4, 5]
    invalid = next(row for row in exact if row["step"] == 4)
    assert invalid["update_performed"] is False
    assert invalid["train_size_after"] == invalid["train_size_before"]

    changed = _records()
    changed[-1]["val_acc"] = 0.01
    changed_rows, _ = comparison.run_history("synthetic", changed, 11, _args())
    changed_exact = [row for row in changed_rows if row["surrogate_type"] == "exact_gp"]
    assert [row["pred_mean"] for row in exact] == pytest.approx(
        [row["pred_mean"] for row in changed_exact]
    )


def test_metric_and_paired_summary_definitions_are_per_seed():
    rows = [
        {"valid": True, "actual": actual, "pred_mean": predicted, "pred_std": 0.05}
        for actual, predicted in [(0.70, 0.71), (0.72, 0.73), (0.75, 0.74), (0.80, 0.79)]
    ]
    metrics = comparison.compute_metrics(rows)
    assert metrics["n_predictions"] == 4
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["pairwise_pair_count"] == 6
    assert metrics["best_candidate_regret"] == pytest.approx(0.0)
    per_seed = []
    for seed, exact, dkl in [(0, 0.03, 0.02), (1, 0.04, 0.05)]:
        for name, mae in [("exact_gp", exact), ("dkl_gp", dkl)]:
            row = {"search_seed": seed, "surrogate_type": name}
            row.update({metric: None for metric in comparison.METRIC_DIRECTIONS})
            row["mae"] = mae
            per_seed.append(row)
    summary, _ = comparison._paired_summary(per_seed)
    mae = summary["metrics"]["mae"]
    assert mae["valid_seed_count"] == 2
    assert [row["difference"] for row in mae["per_seed_differences"]] == pytest.approx([-0.01, 0.01])
    assert (mae["wins"], mae["ties"], mae["losses"]) == (1, 0, 1)


def test_prequential_cli_writes_complete_outputs_with_real_small_gps(tmp_path):
    history = tmp_path / "history.json"
    records = _records(10)
    records[4]["valid"] = True
    records[7]["valid"] = False
    history.write_text(json.dumps(records), encoding="utf-8")

    def run(output):
        comparison.main([
            "--history_paths", str(history),
            "--output", str(output),
            "--n_init", "6",
            "--max_records", "10",
            "--gp_refit_every", "2",
            "--gp_init_steps", "2",
            "--gp_refit_steps", "2",
            "--dkl_hidden_dim", "32",
            "--dkl_feature_dim", "8",
            "--dkl_init_steps", "5",
            "--dkl_refit_steps", "3",
            "--dkl_early_stopping_patience", "3",
        ])
        return list(csv.DictReader((output / "prequential_predictions.csv").open()))

    output = tmp_path / "comparison_first"
    predictions = run(output)
    expected = {
        "config.json", "prequential_predictions.csv", "per_seed_metrics.csv",
        "paired_summary.json", "paired_summary.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert len(predictions) == 2 * (10 - 6)
    assert {row["surrogate_type"] for row in predictions} == {"exact_gp", "dkl_gp"}
    assert [int(row["train_size_before"]) for row in predictions] == [6, 6, 7, 7, 7, 7, 8, 8]
    assert all(math.isfinite(float(row["pred_mean"])) for row in predictions)
    assert all(math.isfinite(float(row["pred_std"])) for row in predictions)
    assert all(float(row["prediction_seconds"]) >= 0.0 for row in predictions)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["surrogate_types"] == ["exact_gp", "dkl_gp"]
    assert config["history_paths"] == [str(history)]
    assert config["analysis_type"] == "off-policy surrogate screening on recorded search trajectories"

    second_output = tmp_path / "comparison_second"
    second_predictions = run(second_output)
    deterministic_fields = ("step", "surrogate_type", "pred_mean", "pred_std", "train_size_before")
    assert [tuple(row[field] for field in deterministic_fields) for row in predictions] == [
        tuple(row[field] for field in deterministic_fields) for row in second_predictions
    ]
    paired = json.loads((output / "paired_summary.json").read_text(encoding="utf-8"))
    assert paired["seed_count"] == 1
    assert "cannot replace a true DKL-BO run" in paired["warning"]
