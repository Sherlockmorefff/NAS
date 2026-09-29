"""Small synthetic tests for the new endpoint-LR tail; never read real data."""
from __future__ import annotations

import copy
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from length_core import FIXED_LRS, TailSession, exact
from stage2_core import Branch, cpu_setup, snapshot, train_block


def make_synthetic_source(budget, updates=8):
    """An explicitly MOCK 1000-contract checkpoint with real nonzero moments.

    Only ``updates`` synthetic updates are executed, and each Adam parameter's
    true step remains ``updates``. The source scheduler is a mock 1000-endpoint
    contract, not a claim that the synthetic model trained 1000 times. This also
    checks that continuation records actual source Adam steps rather than
    blindly replacing every parameter step with 1000.
    """
    cpu_setup(470)
    generator = np.random.default_rng(991)
    x = torch.from_numpy(generator.normal(size=(7, 16)))
    y = 0.8 * x[:, 0] - 0.3 * x[:, 4] + torch.sin(x[:, 7])
    known = torch.linspace(0.001, 0.009, len(x))
    model = Branch(x, y, known, 470)
    result = train_block(model, "joint", updates, [], "synthetic", "DKL", "joint", budget,
                         diagnostic_every=0, checkpoint_every=0)
    state = copy.deepcopy(result["state"])
    state["scheduler"].update(completed=1000, planned_steps=1000, next_lr_stage="final20")
    state.pop("accounting", None)
    state["synthetic_fixture"] = {"actual_source_optimizer_updates": updates,
                                  "scheduler_is_mock_1000_endpoint_contract": True,
                                  "adam_step_not_rewritten": True}
    return state


def _run(session, steps, budget):
    records = []
    optimizer_id = id(session.optimizer)
    for _ in range(steps):
        records.append(session.step(budget))
        assert id(session.optimizer) == optimizer_id
        assert session.lrs == {"mapping": FIXED_LRS[0], "gp": FIXED_LRS[1]}
    return records


def test_restore_source_and_actual_adam_steps(source, budget, directory):
    session = TailSession.from_source(copy.deepcopy(source))
    audit = session.optimizer_audit()
    assert session.total_joint_step == 1000 and session.last_gradient is None
    assert session.source_scheduler == source["scheduler"]
    assert session.model.fingerprint() == source["fingerprint"]
    assert exact(session.optimizer.state_dict(), source["optimizer"])
    assert exact(session.snapshot()["rng"], source["rng"])
    actual = source["synthetic_fixture"]["actual_source_optimizer_updates"]
    assert all(p["step"] == actual for group in audit["groups"] for p in group["parameters"])
    assert any(p["exp_avg_l2"] > 0 for group in audit["groups"] for p in group["parameters"])
    assert audit["all_steps_match_added_updates"]
    assert not any(p.requires_grad for p in session.model.mapper.regressor.parameters())
    head = copy.deepcopy(session.model.mapper.regressor.state_dict())
    _run(session, 1, budget)
    assert exact(head, session.model.mapper.regressor.state_dict())
    assert session.model.fingerprint() != source["fingerprint"]
    assert all(p["step"] == actual + 1 for group in session.optimizer_audit()["groups"] for p in group["parameters"])
    return {"pass": True, "actual_source_adam_steps": actual, "nonzero_momentum": True,
            "full_source_adam_exact": True, "source_rng_exact": True, "head_unchanged": True,
            "after_one_step_actual_adam_steps": actual + 1, "initial_audit": audit}


def test_short_interrupted_continuation(source, budget, directory):
    continuous = TailSession.from_source(copy.deepcopy(source))
    _run(continuous, 6, budget)
    full = continuous.snapshot()
    interrupted = TailSession.from_source(copy.deepcopy(source))
    _run(interrupted, 3, budget)
    partial = interrupted.snapshot()
    path = directory / "synthetic_interrupted_after_3.pt"
    torch.save(partial, path)
    torch.rand(5)
    np.random.random(5)
    random.random()
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    resumed = TailSession.from_continuation(loaded)
    _run(resumed, 3, budget)
    final = resumed.snapshot()
    for name in ("mapper", "gp", "x", "y", "known", "optimizer", "scheduler", "rng", "ordered_parameter_names"):
        assert exact(full[name], final[name]), name
    assert full["fingerprint"] == final["fingerprint"]
    assert continuous.last_gradient == resumed.last_gradient
    assert continuous.last_loss_before_update == resumed.last_loss_before_update
    assert exact(continuous.optimizer_audit(), resumed.optimizer_audit())
    assert final["continuation"]["source_completed_steps"] == 1000
    assert final["continuation"]["additional_completed_steps"] == 6
    assert final["continuation"]["total_joint_step"] == 1006
    assert final["continuation"]["continuation_target"] == 4000
    assert final["scheduler"] == source["scheduler"]
    return {"pass": True, "continuous_updates": 6, "interrupted_updates": [3, 3],
            "parameters_exact": True, "adam_exact": True, "rng_exact": True,
            "actual_lrs_exact": True, "source_scheduler_untouched": True}


def test_save_at_2000_retains_optimizer(source, budget, directory):
    session = TailSession.from_source(copy.deepcopy(source))
    optimizer_id = id(session.optimizer)
    _run(session, 1000, budget)
    assert session.total_joint_step == 2000
    before_optimizer = copy.deepcopy(session.optimizer.state_dict())
    before_cache = (session.model.mapper.training, session.model.gp.training,
                    session.model.gp.likelihood.training, id(session.model.gp.prediction_strategy),
                    session.model.cache_version, id(session.model.gp.train_inputs[0]))
    saved = session.snapshot()
    path = directory / "synthetic_total_joint_2000.pt"
    torch.save(saved, path)
    after_cache = (session.model.mapper.training, session.model.gp.training,
                   session.model.gp.likelihood.training, id(session.model.gp.prediction_strategy),
                   session.model.cache_version, id(session.model.gp.train_inputs[0]))
    assert before_cache == after_cache
    assert id(session.optimizer) == optimizer_id and exact(before_optimizer, session.optimizer.state_dict())
    assert saved["scheduler"] == source["scheduler"] and saved["scheduler"]["planned_steps"] == 1000
    assert saved["continuation"]["total_joint_step"] == 2000
    assert saved["continuation"]["additional_completed_steps"] == 1000
    assert saved["continuation"]["schedule"] == "fixed_endpoint_lr"
    # Continue the same optimizer across 2000, then compare to a real disk restore.
    _run(session, 2, budget)
    continuous = session.snapshot()
    restored = TailSession.from_continuation(torch.load(path, map_location="cpu", weights_only=False))
    _run(restored, 2, budget)
    final = restored.snapshot()
    for key in ("mapper", "gp", "optimizer", "scheduler", "rng"):
        assert exact(continuous[key], final[key]), key
    assert continuous["fingerprint"] == final["fingerprint"]
    actual = source["synthetic_fixture"]["actual_source_optimizer_updates"] + 1002
    assert all(p["step"] == actual for group in restored.optimizer_audit()["groups"] for p in group["parameters"])
    return {"pass": True, "updates_to_2000": 1000, "updates_after_save": 2,
            "restored_comparison_updates": 2, "optimizer_object_survived_2000_save": True,
            "model_modes_inputs_and_cache_unchanged_by_save": True,
            "parameters_adam_rng_exact_after_disk_restore": True,
            "actual_end_adam_step": actual, "actual_lrs": session.lrs}


def test_reject_invalid_sources(source, budget, directory):
    cases = {}
    bad = copy.deepcopy(source)
    bad["scheduler"]["planned_steps"] = 4000
    cases["rewritten_source_schedule"] = bad
    bad = copy.deepcopy(source)
    bad["optimizer"] = None
    cases["prediction_only_endpoint"] = bad
    bad = copy.deepcopy(source)
    bad["optimizer"]["param_groups"][0]["lr"] = 1e-3
    cases["base_lr_used_as_actual_lr"] = bad
    bad = copy.deepcopy(source)
    key = bad["optimizer"]["param_groups"][0]["params"][0]
    bad["optimizer"]["state"].pop(key)
    cases["missing_moments"] = bad
    bad = copy.deepcopy(source)
    group = bad["optimizer"]["param_groups"][0]["params"]
    group[0], group[1] = group[1], group[0]
    cases["wrong_parameter_order"] = bad
    errors = {}
    for name, state in cases.items():
        try:
            TailSession.from_source(state)
        except ValueError as error:
            errors[name] = str(error)
        else:
            raise AssertionError("Invalid source was accepted: " + name)
    return {"pass": True, "rejected_cases": errors, "optimizer_updates": 0}


def run_checks(output_dir, budget):
    """Run only when invoked by the unique parent runner, using its test ledger."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    seconds_before, updates_before = budget.seconds, budget.updates
    wall_started = time.monotonic()
    source = make_synthetic_source(budget)
    checks = {}
    for function in (test_restore_source_and_actual_adam_steps, test_short_interrupted_continuation,
                     test_save_at_2000_retains_optimizer, test_reject_invalid_sources):
        try:
            checks[function.__name__] = function(source, budget, directory)
        except Exception as error:
            checks[function.__name__] = {"pass": False, "error": repr(error), "traceback": traceback.format_exc()}
    # Include recovery, finite checks, serialization and other validation work
    # conservatively, without double charging timed core updates.
    residual = max(0.0, time.monotonic() - wall_started - (budget.seconds - seconds_before))
    budget.charge(residual, kind="training")
    budget.persist()
    result = {"pass": all(check["pass"] for check in checks.values()), "checks": checks,
              "updates": budget.updates - updates_before, "charged_seconds": budget.seconds - seconds_before,
              "synthetic_only": True, "real_data_or_source_checkpoints_opened": False}
    (directory / "core_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result
