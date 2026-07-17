from __future__ import annotations

from surrogate.metrics import PREQUENTIAL_METRIC_FIELDS, GPConvergenceMonitor


def _monitor(**kwargs):
    settings = {
        "min_bo_samples": 2, "max_bo_samples": 20, "convergence_check_every": 1,
        "convergence_patience": 3, "prequential_window": 2, "mae_relative_tol": 0.05,
        "mae_absolute_tol": 0.01, "std_relative_tol": 0.05, "spearman_tol": 0.05,
        "degradation_tolerance": 0.02, "best_acc_patience": 4,
        "best_acc_min_delta": 0.001, "max_wall_time_hours": 1.0,
    }
    settings.update(kwargs)
    return GPConvergenceMonitor(**settings)


def _add(monitor, sample, mae=0.1, std=0.05, spearman=0.8):
    monitor.observe_bo_result(0.7 + sample * 0.002, {"gp_pred_mean": 0.7 + sample * 0.002})
    return monitor.add_check(
        online_samples=sample, gp_train_size=20 + sample,
        holdout_metrics={"mae": mae, "rmse": mae, "pearson": spearman,
                         "spearman": spearman, "predictive_95_coverage": 0.95},
        probe_stds=[std, std], elapsed_seconds=float(sample),
        eval_seconds=[1.0], gp_update_seconds=[0.1],
    )


def _add_scratch(monitor, sample, actual, predicted, std=0.05):
    monitor.observe_bo_result(
        actual,
        {
            "valid": True,
            "gp_pred_mean": predicted,
            "gp_pred_std": 0.02,
        },
    )
    return monitor.add_check(
        online_samples=sample,
        gp_train_size=20 + sample,
        holdout_metrics=None,
        probe_stds=[std, std],
        elapsed_seconds=float(sample),
        eval_seconds=[1.0],
        gp_update_seconds=[0.1],
    )


def test_requires_consecutive_stable_checks():
    monitor = _monitor()
    _add(monitor, 2)
    assert not _add(monitor, 3, mae=0.099, std=0.049)["converged"]
    assert not _add(monitor, 4, mae=0.098, std=0.048)["converged"]
    assert _add(monitor, 5, mae=0.097, std=0.047)["converged"]


def test_obvious_degradation_is_not_convergence():
    monitor = _monitor()
    _add(monitor, 2, mae=0.05, std=0.03)
    row = _add(monitor, 3, mae=0.2, std=0.1, spearman=0.2)
    assert not row["stable_this_check"]
    assert not row["converged"]


def test_scratch_converges_from_stable_full_prequential_window():
    monitor = _monitor(best_acc_patience=99)
    monitor.observe_bo_result(
        0.69,
        {"valid": True, "gp_pred_mean": 0.68, "gp_pred_std": 0.02},
    )

    first = _add_scratch(monitor, 2, 0.70, 0.69)
    assert first["convergence_evidence_source"] == "prequential"
    assert not first["converged"]
    assert _add_scratch(monitor, 3, 0.71, 0.70)["stable_checks"] == 1
    assert _add_scratch(monitor, 4, 0.72, 0.71)["stable_checks"] == 2
    final = _add_scratch(monitor, 5, 0.73, 0.72)

    assert final["stable_checks"] == 3
    assert final["converged"]
    assert final["prequential_spearman"] == 1.0
    assert set(PREQUENTIAL_METRIC_FIELDS).issubset(final)


def test_scratch_window_or_ranking_unavailable_is_not_converged():
    incomplete = _monitor(prequential_window=3, best_acc_patience=99)
    row = _add_scratch(incomplete, 2, 0.70, 0.69)
    assert not row["convergence_evidence_ready"]
    assert row["convergence_not_ready_reason"] == "prequential_window_incomplete:1/3"
    assert not row["converged"]

    constant = _monitor(best_acc_patience=99)
    for sample in range(2, 7):
        row = _add_scratch(constant, sample, 0.70, 0.70)
    assert row["convergence_not_ready_reason"] == "prequential_spearman_unavailable"
    assert row["valid_convergence_checks"] == 0
    assert row["stable_checks"] == 0
    assert not row["converged"]


def test_scratch_metric_degradation_resets_stable_streak():
    monitor = _monitor(best_acc_patience=99)
    monitor.observe_bo_result(
        0.69,
        {"valid": True, "gp_pred_mean": 0.68, "gp_pred_std": 0.02},
    )
    _add_scratch(monitor, 2, 0.70, 0.69)
    stable = _add_scratch(monitor, 3, 0.71, 0.70)
    assert stable["stable_checks"] == 1

    degraded = _add_scratch(monitor, 4, 0.72, 0.50, std=0.10)

    assert not degraded["stable_this_check"]
    assert degraded["stable_checks"] == 0
    assert not degraded["converged"]


def test_scratch_can_converge_with_low_but_stable_ranking_quality():
    monitor = _monitor(best_acc_patience=99)
    monitor.observe_bo_result(
        0.69,
        {"valid": True, "gp_pred_mean": 0.71, "gp_pred_std": 0.02},
    )
    baseline = _add_scratch(monitor, 2, 0.70, 0.70)
    assert baseline["prequential_spearman"] == -1.0

    first = _add_scratch(monitor, 3, 0.69, 0.71)
    second = _add_scratch(monitor, 4, 0.70, 0.70)
    final = _add_scratch(monitor, 5, 0.69, 0.71)

    assert first["stable_checks"] == 1
    assert second["stable_checks"] == 2
    assert final["prequential_spearman"] == -1.0
    assert final["stable_checks"] == 3
    assert final["converged"]


def test_budget_time_and_stagnation_stop_reasons():
    budget = _monitor(max_bo_samples=2)
    assert budget.stop_decision(2, 0.0, 0.0).stop_reason == "sample_budget_reached"
    assert not budget.stop_decision(2, 0.0, 0.0).converged

    timing = _monitor(max_wall_time_hours=0.001)
    assert timing.stop_decision(2, 3.0, 1.0).stop_reason == "time_budget_reached"

    stagnant = _monitor(best_acc_patience=2)
    stagnant.observe_bo_result(0.8, {"gp_pred_mean": 0.8})
    stagnant.observe_bo_result(0.8, {"gp_pred_mean": 0.8})
    stagnant.observe_bo_result(0.8, {"gp_pred_mean": 0.8})
    deferred = stagnant.stop_decision(3, 0.0, 0.0)
    assert deferred.stop_reason is None
    assert deferred.deferred_reason is not None


def test_stagnation_waits_for_three_valid_convergence_checks():
    monitor = _monitor(best_acc_patience=2, convergence_patience=3)
    for _ in range(3):
        monitor.observe_bo_result(0.8, {"gp_pred_mean": 0.8})

    first = _add(monitor, 2)
    assert first["valid_convergence_checks"] == 1
    assert monitor.stop_decision(3, 0.0, 0.0).stop_reason is None
    second = _add(monitor, 3)
    assert second["valid_convergence_checks"] == 2
    assert monitor.stop_decision(3, 0.0, 0.0).stop_reason is None
    third = _add(monitor, 4)
    assert third["valid_convergence_checks"] == 3
    assert monitor.stop_decision(4, 0.0, 0.0).stop_reason == "bo_best_acc_stagnated"


def test_stagnation_gate_does_not_block_budget_or_wall_time():
    budget = _monitor(max_bo_samples=2, best_acc_patience=1)
    budget.observe_bo_result(0.8, {"gp_pred_mean": 0.8})
    budget.observe_bo_result(0.8, {"gp_pred_mean": 0.8})
    decision = budget.stop_decision(2, 0.0, 0.0)
    assert decision.stop_reason == "sample_budget_reached"
    assert decision.deferred_reason is not None

    timing = _monitor(max_wall_time_hours=0.001, best_acc_patience=1)
    timing.observe_bo_result(0.8, {"gp_pred_mean": 0.8})
    timing.observe_bo_result(0.8, {"gp_pred_mean": 0.8})
    decision = timing.stop_decision(2, 3.0, 1.0)
    assert decision.stop_reason == "time_budget_reached"
    assert decision.deferred_reason is not None
