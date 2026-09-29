"""Bounded engineering probes on copies of a selected-data Branch.

No source dataset is opened here. The caller supplies only permitted tensors.
Every optimizer update and the direct gradient probe debit the supplied Budget.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from controlled_dkl import (ATOL, RTOL, Branch, compare, delta,
                            gradient_norm, independent_nll, train_block)


def _copy_state(model):
    return {name: value.detach().clone() for name, value in model.state().items()}


def _exact(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b)
                and all(_exact(a[key], b[key]) for key in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_exact(x, y) for x, y in zip(a, b)))
    return a == b


def _named_storage(model):
    values = dict(model.state())
    values.update({"gp.train_input." + str(i): value
                   for i, value in enumerate(model.gp.train_inputs)})
    values["gp.train_targets"] = model.gp.train_targets
    return {name: tensor.untyped_storage().data_ptr()
            for name, tensor in values.items() if tensor.numel()}


def _body_state(model):
    return {name: value.detach().clone() for name, value in model.mapper.state_dict().items()
            if not name.startswith("regressor.")}


def _prediction_comparison(left, right):
    mean = compare(left[0], right[0])
    variance = compare(left[1], right[1])
    return {"pass": mean["pass"] and variance["pass"], "mean": mean,
            "variance": variance, "mean_exact": torch.equal(left[0], right[0]),
            "variance_exact": torch.equal(left[1], right[1])}


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_traces(path, traces):
    fields = sorted({key for row in traces for key in row})
    if not fields:
        fields = ["run", "branch", "block", "step", "status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(traces)


def check_engineering(model, query, outdir, budget):
    """Return named checks, each with ``pass`` and numerical evidence.

    ``overall`` also contains counts and artifact names. Probe failures are
    recorded separately and do not alter the supplied Branch. Budget exhaustion
    is re-raised after preserving completed checks and traces.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    query = query.detach().clone().double()
    checks, traces = {}, []
    source_state = _copy_state(model)
    source_modes = (model.mapper.training, model.gp.training, model.gp.likelihood.training)
    source_cache = (model.cache_version, id(model.gp.prediction_strategy))
    source_grad_flags = ([p.requires_grad for p in model.mapper.parameters()],
                         [p.requires_grad for p in model.gp.parameters()])
    incoming_rng = (torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state()),
                    random.getstate())
    budget_before = budget.seconds
    run = outdir.name

    def run_check(name, function):
        try:
            checks[name] = function()
        except TimeoutError:
            checks[name] = {"pass": False, "status": "INCOMPLETE", "reason": "training budget exhausted"}
            raise
        except Exception as error:
            checks[name] = {"pass": False, "status": "FAIL", "error_type": type(error).__name__,
                            "reason": str(error)}

    def clone_and_isolation():
        clones = [model.clone() for _ in range(3)]
        originals = [model] + clones
        storage = [_named_storage(branch) for branch in originals]
        shared = []
        for i in range(len(storage)):
            for j in range(i + 1, len(storage)):
                overlap = set(storage[i].values()) & set(storage[j].values())
                if overlap:
                    shared.append({"left": i, "right": j, "count": len(overlap)})
        exact_state = [_exact(source_state, _copy_state(branch)) for branch in clones]
        history_current = []
        for branch in clones:
            with torch.no_grad():
                embedding, _ = branch.encode(branch.x)
            history_current.append(torch.equal(embedding, branch.gp.train_inputs[0]))
        kernels = [branch.kernel() for branch in clones]
        predictions = [branch.predict(query) for branch in clones]
        kernel_exact = [torch.equal(kernels[0], value) for value in kernels[1:]]
        prediction_exact = [_prediction_comparison(predictions[0], value) for value in predictions[1:]]
        result = {"pass": all(exact_state + history_current + kernel_exact)
                  and not shared and all(row["mean_exact"] and row["variance_exact"] for row in prediction_exact),
                  "state_exact": exact_state, "history_current_before_prediction": history_current,
                  "kernel_exact": kernel_exact, "predictions": prediction_exact,
                  "cross_branch_shared_storages": shared,
                  "checked_tensor_counts": [len(value) for value in storage],
                  "mapper_buffer_count": len(list(model.mapper.named_buffers())),
                  "gp_buffer_count": len(list(model.gp.named_buffers()))}
        return result

    def direct_gradient():
        probe = model.clone()
        for parameter in list(probe.mapper.parameters()) + list(probe.gp.parameters()):
            parameter.requires_grad_(True)
            parameter.grad = torch.ones_like(parameter)  # Verify that stale gradients are cleared.
        probe.mapper.zero_grad(set_to_none=True)
        probe.gp.zero_grad(set_to_none=True)
        cleared = all(parameter.grad is None for parameter in
                      list(probe.mapper.parameters()) + list(probe.gp.parameters()))
        before = probe.fingerprint()
        tick = time.perf_counter()
        budget.check()
        try:
            loss, components = probe.objective("joint")
            loss.backward()
            mapping_gradient = gradient_norm(probe.mapping_params())
            gp_gradient = gradient_norm(probe.gp.parameters())
            map_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                             for parameter in probe.mapping_params())
            gp_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                            for parameter in probe.gp.parameters())
            head_absent = all(parameter.grad is None for parameter in probe.mapper.regressor.parameters())
            unchanged = before == probe.fingerprint()
            traces.append({"run": run, "branch": "gradient_probe", "block": "gp_objective_backward_only",
                           "mode": "joint", "step": 0, "planned_steps": 0, **components,
                           "loss_before_update": float(loss.detach()), "mapping_gradient": mapping_gradient,
                           "gp_gradient": gp_gradient, "head_gradient": gradient_norm(probe.mapper.regressor.parameters()),
                           "state_before": before, "state_after": probe.fingerprint(),
                           "seconds": time.perf_counter() - tick})
            return {"pass": cleared and math.isfinite(float(loss.detach())) and map_finite and gp_finite
                    and mapping_gradient > 0 and head_absent and unchanged,
                    "old_gradients_cleared": cleared, "mapping_gradient_l2": mapping_gradient,
                    "gp_gradient_l2": gp_gradient, "mapping_gradients_finite": map_finite,
                    "gp_gradients_finite": gp_finite, "head_gradient_absent": head_absent,
                    "parameters_unchanged_without_optimizer": unchanged}
        finally:
            budget.charge(time.perf_counter() - tick)

    def update_isolation():
        clones = [model.clone() for _ in range(3)]
        states = [_copy_state(branch) for branch in clones]
        body_before = _body_state(clones[0])
        head_before = copy.deepcopy(clones[0].mapper.regressor.state_dict())
        train_block(clones[0], "joint", 1, traces, run, "isolation", "joint_once", budget)
        body_change = delta(body_before, _body_state(clones[0]))
        others_unchanged = [_exact(states[i], _copy_state(clones[i])) for i in (1, 2)]
        head_unchanged = _exact(head_before, clones[0].mapper.regressor.state_dict())
        return {"pass": math.isfinite(body_change) and body_change > 0 and all(others_unchanged) and head_unchanged,
                "mapping_body_parameter_delta_l2": body_change,
                "other_clones_exactly_unchanged": others_unchanged,
                "unused_linear_head_exactly_unchanged": head_unchanged}

    def frozen_mapping():
        probe = model.clone()
        before = copy.deepcopy(probe.mapper.state_dict())
        buffers_before = copy.deepcopy(dict(probe.mapper.named_buffers()))
        train_block(probe, "gp", 1, traces, run, "frozen", "gp_once", budget)
        exact = _exact(before, probe.mapper.state_dict())
        buffer_exact = _exact(buffers_before, dict(probe.mapper.named_buffers()))
        gradient_absent = all(parameter.grad is None for parameter in probe.mapper.parameters())
        return {"pass": exact and buffer_exact and gradient_absent,
                "all_mapper_parameters_and_buffers_exact": exact, "buffers_exact": buffer_exact,
                "registered_buffer_count": len(buffers_before), "mapping_gradient_absent": gradient_absent,
                "mapping_delta_l2": delta(before, probe.mapper.state_dict())}

    def cache_consistency():
        probe = model.clone()
        probe.predict(query)
        warm_version = probe.cache_version
        warm_prediction_strategy = probe.gp.prediction_strategy
        warm_history = probe.gp.train_inputs[0].detach().clone()
        was_warm = warm_prediction_strategy is not None
        before = probe.fingerprint()
        body_before = _body_state(probe)
        for parameter in probe.mapper.parameters():
            parameter.requires_grad_(False)
        for parameter in probe.mapping_params():
            parameter.requires_grad_(True)
        for parameter in probe.gp.parameters():
            parameter.requires_grad_(True)
        # Match train_block(steps=1): its final-20%-schedule applies at step zero.
        optimizer = torch.optim.Adam([
            {"params": probe.mapping_params(), "lr": 1e-4},
            {"params": list(probe.gp.parameters()), "lr": 1e-3}], weight_decay=0.0)
        params = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        budget.check()
        tick = time.perf_counter()
        try:
            probe.mapper.zero_grad(set_to_none=True)
            probe.gp.zero_grad(set_to_none=True)
            loss, components = probe.objective("joint")
            loss.backward()
            mapping_gradient = gradient_norm(probe.mapping_params())
            gp_gradient = gradient_norm(probe.gp.parameters())
            gradient = float(torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True))
            if not math.isfinite(float(loss.detach())):
                raise FloatingPointError("Nonfinite cache-probe loss")
            optimizer.step()
            # Deliberately omit explicit cache invalidation/snapshot: predict
            # must identify the changed state and re-encode historical inputs.
            traces.append({"run": run, "branch": "cache", "block": "joint_once_without_explicit_invalidation",
                           "mode": "joint", "step": 1, "planned_steps": 1, **components,
                           "loss_before_update": float(loss.detach()), "mapping_gradient": mapping_gradient,
                           "gp_gradient": gp_gradient, "gradient_before_clip": gradient,
                           "lr_mapping": 1e-4, "lr_gp": 1e-3,
                           "mapping_body_delta": delta(body_before, _body_state(probe)),
                           "state_before": before, "state_after": probe.fingerprint(),
                           "seconds": time.perf_counter() - tick})
        finally:
            budget.charge(time.perf_counter() - tick)
        old_version_preserved = probe.cache_version == warm_version
        stale_history_before_predict = torch.equal(probe.gp.train_inputs[0], warm_history)
        # objective() switches the GP into train mode and normally clears its
        # posterior strategy. Restore the prewarmed strategy only in this test
        # copy to simulate an external mapper update with a stale posterior.
        probe.gp.eval()
        probe.gp.likelihood.eval()
        probe.gp.prediction_strategy = warm_prediction_strategy
        stale_posterior_present = probe.gp.prediction_strategy is warm_prediction_strategy and was_warm
        normal = probe.predict(query)
        posterior_cache_rebuilt = (probe.gp.prediction_strategy is not None
                                   and probe.gp.prediction_strategy is not warm_prediction_strategy)
        with torch.no_grad():
            current_history, _ = probe.encode(probe.x)
        current_history_matches = torch.equal(probe.gp.train_inputs[0], current_history)
        rebuilt = probe.clone()
        fresh = rebuilt.predict(query)
        comparison = _prediction_comparison(normal, fresh)
        body_change = delta(body_before, _body_state(probe))
        prediction_finite = bool(torch.isfinite(normal[0]).all() and torch.isfinite(normal[1]).all())
        variance_min = float(normal[1].min())
        return {"pass": was_warm and old_version_preserved and stale_history_before_predict
                and stale_posterior_present and posterior_cache_rebuilt
                and current_history_matches and body_change > 0 and comparison["pass"]
                and probe.cache_version == probe.fingerprint() and prediction_finite and variance_min >= -ATOL,
                "cache_was_warm": was_warm, "old_cache_version_preserved_until_predict": old_version_preserved,
                "preupdate_history_still_present_until_predict": stale_history_before_predict,
                "stale_posterior_cache_present_before_predict": stale_posterior_present,
                "posterior_cache_invalidated_and_rebuilt": posterior_cache_rebuilt,
                "postupdate_history_matches_current_mapper": current_history_matches,
                "mapping_body_delta_l2": body_change, "prediction_vs_full_rebuild": comparison,
                "warm_state_id": warm_version, "new_state_id": probe.cache_version,
                "variance_min": variance_min, "variance_min_tolerance": -ATOL,
                "prediction_finite": prediction_finite}

    def disk_restore():
        probe = model.clone()
        kernel = probe.kernel()
        predictions = probe.predict(query)
        state = probe.snapshot()
        path = outdir / "save_restore_probe.pt"
        torch.save(state, path)
        restored = Branch.restore(_load(path))
        kernel_check = compare(kernel, restored.kernel())
        prediction_check = _prediction_comparison(predictions, restored.predict(query))
        exact = _exact(_copy_state(probe), _copy_state(restored))
        return {"pass": exact and kernel_check["pass"] and prediction_check["pass"],
                "state_exact": exact, "kernel": kernel_check, "predictions": prediction_check,
                "saved_state_id": state["fingerprint"], "restored_state_id": restored.fingerprint(),
                "checkpoint": path.name}

    def interrupted_resume():
        base = model.clone().snapshot()
        continuous = Branch.restore(copy.deepcopy(base))
        continuous_state = train_block(continuous, "joint", 5, traces, run, "continuous", "resume_probe", budget,
                                       checkpoint_dir=outdir / "resume_checkpoints")
        interrupted = Branch.restore(copy.deepcopy(base))
        partial = train_block(interrupted, "joint", 5, traces, run, "interrupted", "resume_probe", budget,
                              checkpoint_dir=outdir / "resume_checkpoints", stop_at=2)
        path = outdir / "interrupted_after_two.pt"
        torch.save(partial, path)
        # A perturbed ambient RNG must be replaced by the checkpoint state.
        torch.rand(3)
        np.random.random(3)
        random.random()
        loaded = _load(path)
        resumed = Branch.restore(loaded)
        resumed_state = train_block(resumed, "joint", 5, traces, run, "resumed", "resume_probe", budget,
                                    checkpoint_dir=outdir / "resume_checkpoints", resume=loaded)
        state_exact = _exact(_copy_state(continuous), _copy_state(resumed))
        optimizer_exact = _exact(continuous_state["optimizer"], resumed_state["optimizer"])
        scheduler_exact = _exact(continuous_state["scheduler"], resumed_state["scheduler"])
        rng_exact = _exact(continuous_state["rng"], resumed_state["rng"])
        kernel = compare(continuous.kernel(), resumed.kernel())
        predictions = _prediction_comparison(continuous.predict(query), resumed.predict(query))
        continuous_rows = [row for row in traces if row.get("branch") == "continuous"]
        resumed_rows = [row for row in traces if row.get("branch") in ("interrupted", "resumed")]
        lr_exact = [(row["step"], row["lr_mapping"], row["lr_gp"]) for row in continuous_rows] == [
            (row["step"], row["lr_mapping"], row["lr_gp"]) for row in resumed_rows]
        return {"pass": state_exact and optimizer_exact and scheduler_exact and rng_exact and lr_exact
                and kernel["pass"] and predictions["pass"], "state_exact": state_exact,
                "optimizer_exact": optimizer_exact, "scheduler_exact": scheduler_exact,
                "rng_exact": rng_exact, "learning_rate_sequence_exact": lr_exact,
                "continuous_updates": len(continuous_rows), "interrupted_plus_resumed_updates": len(resumed_rows),
                "interruption_after": loaded["scheduler"]["completed"],
                "completed_after_resume": resumed_state["scheduler"]["completed"],
                "kernel": kernel, "predictions": predictions,
                "continuous_state_id": continuous.fingerprint(), "resumed_state_id": resumed.fingerprint(),
                "interruption_checkpoint": path.name}

    try:
        run_check("clone_initialization_and_storage", clone_and_isolation)
        run_check("gp_objective_gradient_path", direct_gradient)
        run_check("actual_update_and_branch_isolation", update_isolation)
        run_check("frozen_mapping_parameters_and_buffers", frozen_mapping)
        run_check("independent_numpy_nll", lambda: independent_nll(model.clone()))
        run_check("mapping_version_and_prediction_cache", cache_consistency)
        run_check("disk_save_restore", disk_restore)
        run_check("optimizer_schedule_rng_interrupted_resume", interrupted_resume)
    finally:
        torch.set_rng_state(incoming_rng[0])
        np.random.set_state(incoming_rng[1])
        random.setstate(incoming_rng[2])
        source_unchanged = _exact(source_state, _copy_state(model))
        modes_unchanged = source_modes == (model.mapper.training, model.gp.training, model.gp.likelihood.training)
        cache_unchanged = source_cache == (model.cache_version, id(model.gp.prediction_strategy))
        flags_unchanged = source_grad_flags == ([p.requires_grad for p in model.mapper.parameters()],
                                              [p.requires_grad for p in model.gp.parameters()])
        checks["input_model_untouched"] = {"pass": source_unchanged and modes_unchanged and cache_unchanged and flags_unchanged,
                                           "state_exact": source_unchanged, "modes_unchanged": modes_unchanged,
                                           "cache_unchanged": cache_unchanged, "gradient_flags_unchanged": flags_unchanged}
        checks["overall"] = {"pass": len(checks) == 9 and all(row["pass"] for row in checks.values()),
                             "completed_check_count": len(checks), "expected_check_count": 9,
                             "optimizer_updates": sum(int(row.get("step", 0)) > 0 for row in traces),
                             "gradient_only_probes": sum(int(row.get("step", 0)) == 0 for row in traces),
                             "charged_training_seconds": budget.seconds - budget_before,
                             "atol": ATOL, "rtol": RTOL, "trace_file": "verification_traces.csv"}
        _write_traces(outdir / "verification_traces.csv", traces)
        checks = _json_safe(checks)
        (outdir / "engineering_checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")
    return checks
