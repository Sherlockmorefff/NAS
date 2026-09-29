"""Targeted Stage-2 probes. Synthetic data only; never open a label source."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from stage2_core import (Branch, Budget, BudgetExceeded, RawBranch, check_forks,
                         cpu_setup, independent_nll, numerical_settings,
                         predict_timed, restore, snapshot, train_block)


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def synthetic(n=8, seed=771):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.normal(size=(n, 16)))
    y = 0.4 * x[:, 0] - 0.3 * x[:, 4] + torch.sin(x[:, 8])
    known = torch.linspace(0.0001, 0.01, n)
    return x, y, known


def _run(model, mode, steps, budget, **kwargs):
    traces = []
    result = train_block(model, mode, steps, traces, "synthetic", "probe", "probe", budget, **kwargs)
    return result, traces


def test_raw50_nll_and_latent_variance_units(budget, directory):
    x, y, known = synthetic(50)
    model = RawBranch(x, y, known)
    nll = independent_nll(model)
    assert nll["pass"], nll
    assert not hasattr(model, "mapper") and model.gp.train_inputs[0].shape == (50, 16)
    assert float(model.gp.mean_module.constant) == 0
    assert math.isclose(float(model.gp.covar_module.base_kernel.lengthscale), 1)
    assert math.isclose(float(model.gp.covar_module.outputscale), 1)
    assert math.isclose(float(model.gp.likelihood.second_noise), .01)
    _run(model, "gp", 3, budget)
    query = synthetic(7, seed=789)[0]
    mean, variance = predict_timed(model, query, budget)
    ell = float(model.gp.covar_module.base_kernel.lengthscale.detach())
    scale = float(model.gp.covar_module.outputscale.detach())
    extra = float(model.gp.likelihood.second_noise.detach())
    const = float(model.gp.mean_module.constant.detach())

    def kernel(a, b):
        distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1) / ell
        return scale * (1 + np.sqrt(5) * distance + 5 * distance ** 2 / 3) * np.exp(-np.sqrt(5) * distance)

    covariance = kernel(x.numpy(), x.numpy()) + np.diag(known.numpy() + extra)
    cross = kernel(query.numpy(), x.numpy())
    expected_mean = const + cross @ np.linalg.solve(covariance, y.numpy() - const)
    expected_variance = scale - np.sum(cross * np.linalg.solve(covariance, cross.T).T, axis=1)
    assert np.allclose(mean.numpy(), expected_mean, atol=1e-8, rtol=1e-6)
    assert np.allclose(variance.numpy(), expected_variance, atol=1e-8, rtol=1e-6)
    center, y_scale = .8, .03
    raw_mean, raw_variance = center + y_scale * mean.numpy(), y_scale ** 2 * variance.numpy()
    assert np.allclose(raw_mean, center + y_scale * expected_mean)
    assert np.allclose(raw_variance, y_scale ** 2 * expected_variance)
    restored = restore(snapshot(model))
    assert isinstance(restored, RawBranch) and not hasattr(restored, "mapper")
    assert restored.fingerprint() == model.fingerprint()
    return {"pass": True, "selected_n": 50, "gp_input_dimensions": 16,
            "independent_nll": nll, "posterior_mean_max_abs": float(np.max(np.abs(mean.numpy() - expected_mean))),
            "latent_variance_max_abs": float(np.max(np.abs(variance.numpy() - expected_variance))),
            "accuracy_variance_factor": y_scale ** 2,
            "query_noise_used": False, "raw_restored_input_dimensions": 16}


def test_optimizer_logging_and_exact_schedule(budget, directory):
    x, y, known = synthetic(6)
    cpu_setup(17)
    initial = snapshot(Branch(x, y, known, 17))
    quiet, logged = restore(copy.deepcopy(initial)), restore(copy.deepcopy(initial))
    no_log, _ = _run(quiet, "joint", 10, budget, diagnostic_every=0, checkpoint_every=0)
    with_log, traces = _run(logged, "joint", 10, budget, diagnostic_every=1, checkpoint_every=1,
                            checkpoint_path=directory / "logged_every_step.pt")
    assert quiet.fingerprint() == logged.fingerprint()
    assert exact(no_log["state"]["optimizer"], with_log["state"]["optimizer"])
    assert [(r["lr_mapping"], r["lr_gp"]) for r in traces] == [(1e-3, 1e-2)] * 8 + [(1e-4, 1e-3)] * 2
    assert all(int(value["step"]) == 10 for value in with_log["state"]["optimizer"]["state"].values())

    # Exercise the actual scientific schedule lengths with synthetic tensors.
    regression_model = restore(copy.deepcopy(initial))
    reg_result, reg_traces = _run(regression_model, "regression", 1000, budget,
                                diagnostic_every=1, checkpoint_every=0)
    raw = RawBranch(x, y, known)
    gp_result, gp_traces = _run(raw, "gp", 300, budget, diagnostic_every=1, checkpoint_every=0)
    assert [r["lr_mapping"] for r in reg_traces] == [1e-3] * 800 + [1e-4] * 200
    assert [r["lr_gp"] for r in gp_traces] == [1e-2] * 240 + [1e-3] * 60
    assert all(int(value["step"]) == 1000 for value in reg_result["state"]["optimizer"]["state"].values())
    assert all(int(value["step"]) == 300 for value in gp_result["state"]["optimizer"]["state"].values())
    return {"pass": True, "logging_checkpoint_state_exact": True, "adam_moments_exact": True,
            "optimizer_instances_per_call": 1, "regression_actual_updates": 1000,
            "gp_actual_updates": 300, "regression_lr_counts": [800, 200], "gp_lr_counts": [240, 60],
            "joint_logger_comparison_updates": [10, 10]}


def test_restore_crosses_lr_boundary(budget, directory):
    x, y, known = synthetic(8)
    results = {}
    for name, seed_model, mode in (("MAPPED8", Branch(x, y, known, 6), "joint"),
                                   ("RAW16", RawBranch(x, y, known), "gp")):
        initial = snapshot(seed_model)
        continuous = restore(copy.deepcopy(initial))
        full, full_logs = _run(continuous, mode, 10, budget, diagnostic_every=1, checkpoint_every=0)
        interrupted = restore(copy.deepcopy(initial))
        partial, partial_logs = _run(interrupted, mode, 10, budget, stop_at=7, diagnostic_every=1,
                                    checkpoint_every=0, checkpoint_path=directory / (name + "_partial.pt"))
        torch.rand(4)
        np.random.random(4)
        random.random()
        saved = torch.load(directory / (name + "_partial.pt"), map_location="cpu", weights_only=False)
        resumed = restore(saved)
        final, final_logs = _run(resumed, mode, 10, budget, resume=saved, diagnostic_every=1, checkpoint_every=0)
        assert math.isclose(partial["seconds"] + final["call_seconds"], final["seconds"], rel_tol=0, abs_tol=1e-9)
        assert final["updates"] == 10 and final["call_updates"] == 3
        assert saved["accounting"]["completed_updates"] == 7
        assert continuous.fingerprint() == resumed.fingerprint()
        for field in ("optimizer", "scheduler", "rng"):
            assert exact(full["state"][field], final["state"][field]), field
        assert [(r["step"], r["lr_mapping"], r["lr_gp"]) for r in full_logs] == [
            (r["step"], r["lr_mapping"], r["lr_gp"]) for r in partial_logs + final_logs]
        query = synthetic(5, seed=332)[0]
        assert exact(predict_timed(continuous, query, budget), predict_timed(resumed, query, budget))
        results[name] = {"state_exact": True, "optimizer_exact": True, "rng_exact": True,
                         "schedule_exact": True, "prediction_exact": True, "interruption_step": 7}
    return {"pass": True, "modes": results}


def test_three_forks_and_snapshot_read_only(budget, directory):
    x, y, known = synthetic(7)
    common = Branch(x, y, known, 8)
    forked = check_forks(common)
    branches = forked["branches"]
    unchanged = {name: branch.fingerprint() for name, branch in branches.items()}
    dkl = branches["DKL"]
    query = synthetic(3, seed=234)[0]
    predict_timed(dkl, query, budget)
    before = (dkl.fingerprint(), dkl.mapper.training, dkl.gp.training,
              dkl.gp.likelihood.training, id(dkl.gp.prediction_strategy), dkl.cache_version,
              [p.requires_grad for p in dkl.mapper.parameters()])
    saved = snapshot(dkl)
    after = (dkl.fingerprint(), dkl.mapper.training, dkl.gp.training,
             dkl.gp.likelihood.training, id(dkl.gp.prediction_strategy), dkl.cache_version,
             [p.requires_grad for p in dkl.mapper.parameters()])
    assert before == after
    head_before = copy.deepcopy(dkl.mapper.regressor.state_dict())
    _run(dkl, "joint", 2, budget)
    assert dkl.fingerprint() != unchanged["DKL"]
    assert exact(head_before, dkl.mapper.regressor.state_dict())
    assert all(branches[name].fingerprint() == unchanged[name] for name in ("FROZEN", "REG"))
    frozen_before = copy.deepcopy(branches["FROZEN"].mapper.state_dict())
    _run(branches["FROZEN"], "gp", 2, budget)
    assert exact(frozen_before, branches["FROZEN"].mapper.state_dict())
    return {"pass": True, **forked["checks"], "snapshot_left_cache_modes_parameters_untouched": True,
            "joint_changed_only_target_branch": True, "joint_head_frozen": True, "gp_mapper_frozen": True}


def test_synthetic_query_label_poisoning(budget, directory):
    from stage2_data import query_projection
    x, y, known = synthetic(20)
    query = synthetic(9, seed=993)[0].numpy()
    rows = [{"candidate_id": "q%02d" % index, "architecture_fingerprint": "a%d" % index,
             "x_raw": vector.tolist(), "y": .5, "measurement_se": .01}
            for index, vector in enumerate(query)]
    poisoned = copy.deepcopy(rows)
    for index, row in enumerate(poisoned):
        row.update(y=1e90 * (index + 1), measurement_se=-1e80, query_rank=100 - index)
    projections = [query_projection(rows), query_projection(poisoned)]
    assert exact(projections[0], projections[1])
    center, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    y_center, y_scale = y.mean(), y.std(unbiased=False).clamp_min(1e-8)
    fingerprints, predictions, selected = [], [], []
    for projected in projections:
        cpu_setup(33)
        model = Branch((x - center) / scale, (y - y_center) / y_scale, known / y_scale ** 2, 33)
        _run(model, "regression", 7, budget)
        _run(model, "gp", 3, budget)
        _run(model, "joint", 4, budget)
        features = (torch.as_tensor(projected["x_raw"], dtype=torch.float64) - center) / scale
        predicted = predict_timed(model, features, budget)
        fingerprints.append(model.fingerprint())
        predictions.append(predicted)
        selected.append(sorted(range(len(query)), key=lambda i: (-float(predicted[0][i]), projected["ids"][i]))[:5])
    assert fingerprints[0] == fingerprints[1] and exact(predictions[0], predictions[1])
    assert selected[0] == selected[1]
    return {"pass": True, "synthetic_only": True, "query_projection_exact": True,
            "fixed_train_standardization_exact": True, "fit_state_exact": True,
            "mean_latent_variance_exact": True, "top5_ids_exact": True,
            "updates_per_fit": 14, "query_y_and_se_received_by_fit": False}


def test_budget_interrupt_preserves_state(budget, directory):
    class StopAfterTwo(Budget):
        def charge(self, seconds, kind="training", updates=0):
            super().charge(seconds, kind, updates)
            if self.updates >= 2:
                self.limit = self.seconds

    # Mirror the interrupted computation's measured cost into the real ledger.
    local = StopAfterTwo()
    x, y, known = synthetic(5)
    model = RawBranch(x, y, known)
    try:
        _run(model, "gp", 10, local, checkpoint_path=directory / "budget_partial.pt")
        raise AssertionError("Expected budget preemption")
    except BudgetExceeded as error:
        assert error.summary["completed"] == 2
        assert error.summary["status"] == "BUDGET_EXHAUSTED"
        saved = torch.load(directory / "budget_partial.pt", map_location="cpu", weights_only=False)
        assert saved["scheduler"]["completed"] == 2 and saved["optimizer"]
        assert restore(saved).fingerprint() == model.fingerprint()
    finally:
        budget.charge(local.training_seconds, updates=local.updates)
    return {"pass": True, "saved_completed_updates": 2, "optimizer_saved": True,
            "raw_state_restore_exact": True, "per_step_budget_preemption": True}


def run_checks(outdir, budget=None):
    directory = Path(outdir)
    directory.mkdir(parents=True, exist_ok=True)
    budget = budget or Budget(directory / "synthetic_resource_ledger.json")
    cpu_setup(0)
    before_seconds, before_updates = budget.seconds, budget.updates
    checks = {}
    functions = [test_raw50_nll_and_latent_variance_units, test_optimizer_logging_and_exact_schedule,
                 test_restore_crosses_lr_boundary, test_three_forks_and_snapshot_read_only,
                 test_synthetic_query_label_poisoning, test_budget_interrupt_preserves_state]
    for function in functions:
        tick, prior = time.monotonic(), budget.seconds
        try:
            checks[function.__name__] = function(budget, directory)
        except BudgetExceeded:
            raise
        except Exception as error:
            checks[function.__name__] = {"pass": False, "error": repr(error), "traceback": traceback.format_exc()}
        finally:
            # Include synthetic direct NLL, dense posterior checks, setup and
            # verification overhead, while not charging train/predict twice.
            residual = max(0., time.monotonic() - tick - (budget.seconds - prior))
            budget.charge(residual)
            budget.persist()
    result = {"pass": all(row["pass"] for row in checks.values()), "checks": checks,
              "synthetic_optimizer_updates": budget.updates - before_updates,
              "charged_seconds": budget.seconds - before_seconds,
              "real_source_data_read": False}
    (directory / "core_checks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    result = run_checks(args.outdir, Budget(args.ledger) if args.ledger else None)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["pass"] else 1)
