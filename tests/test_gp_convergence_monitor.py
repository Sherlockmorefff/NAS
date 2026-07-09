from __future__ import annotations

from surrogate.metrics import GPConvergenceMonitor


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


def test_missing_holdout_metrics_do_not_soft_converge():
    monitor = _monitor(best_acc_patience=99)
    for sample in range(2, 7):
        monitor.observe_bo_result(0.7, {"gp_pred_mean": 0.7})
        row = monitor.add_check(
            online_samples=sample, gp_train_size=20 + sample,
            holdout_metrics=None, probe_stds=[0.05, 0.05],
            elapsed_seconds=float(sample), eval_seconds=[1.0], gp_update_seconds=[0.1],
        )
    assert row["stable_checks"] == 0
    assert not row["converged"]
    assert monitor.stop_decision(6, 0.0, 0.0).stop_reason is None


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
    assert stagnant.stop_decision(3, 0.0, 0.0).stop_reason == "bo_best_acc_stagnated"
