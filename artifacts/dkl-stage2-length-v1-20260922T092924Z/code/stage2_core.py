"""Stage-2 fixed-block training around the unchanged, audited Stage-01 GP.

This module has no data-source access. Its tensors contain selected training
observations only; queries enter only predict_timed as feature tensors.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import gpytorch
import numpy as np
import torch

from stage01_core import Branch, ExactGP, cpu_setup, tensor_hash
from stage01_core import independent_nll as _independent_nll


@contextmanager
def numerical_settings():
    """Exactly the frozen precision policy; no escalating jitter retries."""
    with ExitStack() as stack:
        stack.enter_context(gpytorch.settings.cholesky_jitter(double_value=1e-8))
        stack.enter_context(gpytorch.settings.cholesky_max_tries(1))
        stack.enter_context(gpytorch.settings.max_cholesky_size(10000))
        stack.enter_context(gpytorch.settings.min_fixed_noise(double_value=0.0))
        stack.enter_context(gpytorch.settings.fast_pred_var(False))
        yield


class BudgetExceeded(TimeoutError):
    def __init__(self, message="Cumulative fitting/prediction budget exhausted", summary=None):
        super().__init__(message)
        self.summary = summary
        self.state = None if summary is None else summary["state"]


class Budget:
    """Monotonic time ledger, debited per update and persisted periodically.

    Training errors are still charged. Checkpoint disk I/O is separate from the
    experiment's fitting/prediction time; diagnostics are conservatively charged
    to training. A preempted update is retained and accounted, never discarded.
    """
    def __init__(self, path=None, limit=7200):
        self.path = None if path is None else Path(path)
        self.limit = float(limit)
        previous = json.loads(self.path.read_text()) if self.path and self.path.exists() else {}
        if previous and float(previous["limit"]) != self.limit:
            raise ValueError("Budget limit differs from existing resource ledger")
        self.training_seconds = float(previous.get("training_seconds", previous.get("seconds", 0)))
        self.prediction_seconds = float(previous.get("prediction_seconds", 0))
        self.verification_seconds = float(previous.get("verification_seconds", 0))
        self.verification_updates = int(previous.get("verification_updates", 0))
        self.updates = int(previous.get("updates", 0))
        self.predictions = int(previous.get("predictions", 0))
        self.failed_training_attempts = int(previous.get("failed_training_attempts", 0))

    @property
    def seconds(self):
        return self.training_seconds + self.prediction_seconds + self.verification_seconds

    def charge(self, seconds, kind="training", updates=0):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Elapsed time must be finite and nonnegative")
        if kind == "training":
            self.training_seconds += seconds
            self.updates += updates
        elif kind == "prediction":
            self.prediction_seconds += seconds
            self.predictions += 1
        elif kind == "verification":
            self.verification_seconds += seconds
            self.verification_updates += updates
        else:
            raise ValueError("Unknown budget cost category")

    def check(self):
        if self.seconds >= self.limit:
            raise BudgetExceeded()

    def as_dict(self):
        return {"seconds": self.seconds, "limit": self.limit,
                "training_seconds": self.training_seconds,
                "prediction_seconds": self.prediction_seconds,
                "verification_seconds": self.verification_seconds,
                "verification_updates": self.verification_updates,
                "updates": self.updates, "predictions": self.predictions,
                "failed_training_attempts": self.failed_training_attempts,
                "clock": "time.monotonic", "diagnostics_charged_as_training": True}

    def persist(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(self.as_dict(), indent=2, allow_nan=False) + "\n")
            temporary.replace(self.path)


class RawBranch:
    """No mapper: the GP consumes the selected standardized [n,16] directly."""
    def __init__(self, x, y, known, seed=None):
        self.x, self.y, self.known = (v.detach().clone().double() for v in (x, y, known))
        if self.x.shape != (len(y), 16) or self.known.shape != self.y.shape:
            raise ValueError("RAW16 requires aligned selected x16/y/known vectors")
        if not all(bool(torch.isfinite(v).all()) for v in (self.x, self.y, self.known)):
            raise ValueError("Nonfinite RAW16 data")
        if bool((self.known < 0).any()):
            raise ValueError("Known variance must be nonnegative")
        self.gp = ExactGP(self.x.clone(), self.y, self.known)
        self.cache_version = None

    def encode(self, x):
        if x.ndim != 2 or x.shape[1] != 16:
            raise ValueError("RAW16 GP inputs must retain exactly 16 dimensions")
        return x, None

    def mapping_params(self):
        return []

    def state(self):
        return {**{"gp." + k: v for k, v in self.gp.state_dict().items()},
                "data.x": self.x, "data.y": self.y, "data.known": self.known,
                "likelihood.fixed_noise": self.gp.likelihood.noise_covar.noise}

    def fingerprint(self):
        return tensor_hash(self.state())

    def synchronize_history(self):
        self.gp.set_train_data(inputs=self.x.detach().clone(), targets=self.y, strict=False)
        self.gp.prediction_strategy = None
        self.cache_version = None

    def clone(self):
        result = RawBranch(self.x, self.y, self.known)
        result.gp.load_state_dict(copy.deepcopy(self.gp.state_dict()))
        result.synchronize_history()
        return result

    def objective(self, mode="gp"):
        if mode != "gp":
            raise ValueError("RAW16 supports only GP optimization")
        self.gp.train()
        self.gp.set_train_data(inputs=self.x, targets=self.y, strict=False)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        with numerical_settings():
            loss = -mll(self.gp(self.x), self.y)
        return loss, {"data_nll": float(loss.detach())}

    def predict(self, query):
        self.encode(query)
        version = self.fingerprint()
        with torch.no_grad(), numerical_settings():
            if self.cache_version != version:
                self.synchronize_history()
                self.cache_version = version
            self.gp.eval()
            self.gp.likelihood.eval()
            posterior = self.gp(query)
            return posterior.mean.clone(), posterior.lazy_covariance_matrix.diagonal().clone()

    def kernel(self):
        with torch.no_grad():
            return self.gp.covar_module(self.x).to_dense().clone()

    def snapshot(self, optimizer=None, block=None, completed=0):
        return snapshot(self, optimizer, block, completed)

    @classmethod
    def restore(cls, state):
        if state.get("model_type") != "RAW16":
            raise ValueError("Cannot load mapped-state parameters into RAW16")
        return restore(state)


def snapshot(model, optimizer=None, block=None, completed=0, mode=None, planned_steps=None):
    """Save without touching module modes, GP caches, train inputs, or RNG."""
    with torch.no_grad():
        history, _ = model.encode(model.x)
    state = {"model_type": "RAW16" if isinstance(model, RawBranch) else "MAPPED8",
             "model_kind": "RAW16" if isinstance(model, RawBranch) else "MAPPED8",
             "gp": copy.deepcopy(model.gp.state_dict()), "x": model.x.clone(),
             "y": model.y.clone(), "known": model.known.clone(),
             "fingerprint": model.fingerprint(), "history_embedding": history.detach().clone(),
             "optimizer": None if optimizer is None else copy.deepcopy(optimizer.state_dict()),
             "scheduler": {"block": block, "completed": completed, "mode": mode,
                           "planned_steps": planned_steps,
                           "policy": "first 80 percent base LR, final 20 percent base LR / 10",
                           "next_lr_stage": None if planned_steps is None else
                               ("final20" if completed >= int(0.8 * planned_steps) else "first80")},
             "rng": {"torch": torch.get_rng_state().clone(),
                     "numpy": copy.deepcopy(np.random.get_state()), "python": random.getstate()}}
    if hasattr(model, "mapper"):
        state["mapper"] = copy.deepcopy(model.mapper.state_dict())
    return state


def restore(state):
    if state.get("model_type", "MAPPED8") == "RAW16":
        model = RawBranch(state["x"], state["y"], state["known"])
    else:
        model = Branch(state["x"], state["y"], state["known"], 0)
        model.mapper.load_state_dict(state["mapper"])
    model.gp.load_state_dict(state["gp"])
    model.synchronize_history()
    if not torch.equal(model.gp.train_inputs[0], state["history_embedding"]):
        raise RuntimeError("Checkpoint history does not match current representation")
    if model.fingerprint() != state["fingerprint"]:
        raise RuntimeError("Checkpoint state fingerprint mismatch")
    torch.set_rng_state(state["rng"]["torch"])
    np.random.set_state(state["rng"]["numpy"])
    random.setstate(state["rng"]["python"])
    return model


def _save(state, path):
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)


def _optimizer(model, mode):
    if mode not in ("regression", "gp", "joint"):
        raise ValueError("Expected regression, gp, or joint mode")
    raw = isinstance(model, RawBranch)
    if raw and mode != "gp":
        raise ValueError("RAW16 cannot train a mapper or a regression head")
    if not raw:
        for p in model.mapper.parameters():
            p.requires_grad_(mode == "regression")
        for p in model.mapping_params():
            p.requires_grad_(mode in ("regression", "joint"))
        model.mapper.train(mode in ("regression", "joint"))
        model.mapper.zero_grad(set_to_none=True)
    for p in model.gp.parameters():
        p.requires_grad_(mode != "regression")
    model.gp.zero_grad(set_to_none=True)
    model.gp.train()
    groups = []
    if mode == "regression":
        groups.append({"params": list(model.mapper.parameters()), "lr": 1e-3, "base_lr": 1e-3})
    elif mode == "joint":
        groups.append({"params": model.mapping_params(), "lr": 1e-3, "base_lr": 1e-3})
    if mode != "regression":
        groups.append({"params": list(model.gp.parameters()), "lr": 1e-2, "base_lr": 1e-2})
    return torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def _diagnostics(model, previous_kernel):
    with torch.no_grad(), numerical_settings():
        embedding, _ = model.encode(model.x)
        kernel = model.gp.covar_module(embedding).to_dense().detach()
        ell = float(model.gp.covar_module.base_kernel.lengthscale.item())
        scale = float(model.gp.covar_module.outputscale.item())
        extra = float(model.gp.likelihood.second_noise.item())
        if not bool(torch.isfinite(embedding).all() and torch.isfinite(kernel).all()):
            raise FloatingPointError("Nonfinite diagnostic embedding or kernel")
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.gp.parameters()):
            raise FloatingPointError("Nonfinite GP parameter at diagnostic/fixed endpoint")
        result = {"embedding_dimensions": int(embedding.shape[1]),
                  "embedding_rms": float(embedding.square().mean().sqrt()),
                  "embedding_row_norm_mean": float(embedding.norm(dim=1).mean()),
                  "embedding_coordinate_std_mean": float(embedding.std(dim=0, unbiased=False).mean()),
                  "mean": float(model.gp.mean_module.constant.item()), "lengthscale": ell,
                  "outputscale": scale, "extra_noise_variance_standardized": extra,
                  "lengthscale_distance_to_lower": ell - 1e-4,
                  "outputscale_distance_to_lower": scale - 1e-8,
                  "extra_noise_distance_to_lower": extra - 1e-6,
                  "training_kernel_frobenius_norm": float(kernel.norm()),
                  "training_kernel_relative_change_since_previous_log": None if previous_kernel is None else
                      float((kernel - previous_kernel).norm() / previous_kernel.norm().clamp_min(1e-30)),
                  "state_after": model.fingerprint()}
    return result, kernel.clone()


def train_block(model, mode, steps, traces, run, branch, block, budget,
                checkpoint_path=None, resume=None, stop_at=None,
                diagnostic_every=100, checkpoint_every=100):
    """One Adam per block, with an absolute 80/20 schedule even after resume.

    Returns a summary with its restorable ``state``. ``stop_at`` exists solely
    for synthetic checkpoint verification; formal runs always use ``None``.
    Budget exhaustion saves the current state then raises BudgetExceeded.
    """
    if steps <= 0 or diagnostic_every < 0 or checkpoint_every < 0:
        raise ValueError("Invalid fixed training-block configuration")
    target = steps if stop_at is None else int(stop_at)
    start = int(resume["scheduler"]["completed"]) if resume is not None else 0
    if not 0 <= start <= target <= steps:
        raise ValueError("Invalid resume/stop position")
    if resume is not None:
        schedule = resume["scheduler"]
        if (schedule["block"], schedule["mode"], schedule["planned_steps"]) != (block, mode, steps):
            raise ValueError("Resume block/mode/schedule mismatch")
        if model.fingerprint() != resume["fingerprint"]:
            raise ValueError("Resume requires the matching restored parameter state")
        torch.set_rng_state(resume["rng"]["torch"])
        np.random.set_state(resume["rng"]["numpy"])
        random.setstate(resume["rng"]["python"])
    budget_before = budget.seconds
    prior_block_seconds = float(resume.get("accounting", {}).get("block_cumulative_seconds", 0.0)) if resume is not None else 0.0
    if resume is not None and start and "accounting" not in resume:
        raise ValueError("Stage-2 block resume requires checkpoint cost accounting")
    setup_start = time.monotonic()
    optimizer = _optimizer(model, mode)
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    fixed_embedding = None
    if mode == "gp":
        with torch.no_grad():
            fixed_embedding, _ = model.encode(model.x)
            fixed_embedding = fixed_embedding.detach().clone()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.gp.likelihood, model.gp)
    budget.charge(time.monotonic() - setup_start)
    completed, previous_kernel = start, None

    def checkpoint_state():
        tick = time.monotonic()
        try:
            state = snapshot(model, optimizer, block, completed, mode, steps)
        finally:
            budget.charge(time.monotonic() - tick)
        state["accounting"] = {"block_cumulative_seconds": prior_block_seconds + budget.seconds - budget_before,
                               "completed_updates": completed}
        return state

    def finish(status):
        state = checkpoint_state()
        _save(state, checkpoint_path)
        budget.persist()
        return {"state": state, "updates": completed, "completed": completed,
                "seconds": prior_block_seconds + budget.seconds - budget_before,
                "call_updates": completed - start, "call_seconds": budget.seconds - budget_before,
                "complete": completed == steps, "status": status,
                "optimizer_instances_this_call": 1}

    try:
        with numerical_settings():
            for step in range(start, target):
                budget.check()
                tick = time.monotonic()
                updated = 0
                try:
                    for group in optimizer.param_groups:
                        group["lr"] = group["base_lr"] * (0.1 if step >= int(0.8 * steps) else 1.0)
                    optimizer.zero_grad(set_to_none=True)
                    components = {"regression": None, "embedding_norm": None, "data_nll": None}
                    if mode == "regression":
                        embedding, pred = model.encode(model.x)
                        regression = (pred - model.y).square().mean()
                        embedding_norm = embedding.square().sum(-1).mean()
                        loss = regression + 0.001 * embedding_norm
                        components.update(regression=float(regression.detach()),
                                          embedding_norm=float(embedding_norm.detach()))
                    else:
                        embedding = fixed_embedding if mode == "gp" else model.encode(model.x)[0]
                        model.gp.set_train_data(inputs=embedding, targets=model.y, strict=False)
                        loss = -mll(model.gp(embedding), model.y)
                        components["data_nll"] = float(loss.detach())
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite fixed-block training objective")
                    loss.backward()
                    gradient = float(torch.nn.utils.clip_grad_norm_(all_params, 5.0, error_if_nonfinite=True))
                    optimizer.step()
                    completed = step + 1
                    updated = 1
                    model.cache_version = None
                    if completed == target or (diagnostic_every and completed % diagnostic_every == 0):
                        diagnostic, previous_kernel = _diagnostics(model, previous_kernel)
                        traces.append({"run": run, "branch": branch, "block": block,
                                       "mode": mode, "step": completed, "planned_steps": steps,
                                       "loss_before_update": float(loss.detach()), **components,
                                       "gradient_before_clip": gradient,
                                       "lr_mapping": next((g["lr"] for g in optimizer.param_groups if g["base_lr"] == 1e-3), 0.),
                                       "lr_gp": next((g["lr"] for g in optimizer.param_groups if g["base_lr"] == 1e-2), 0.),
                                       "cumulative_seconds": budget.seconds + time.monotonic() - tick,
                                       "block_seconds": prior_block_seconds + budget.seconds - budget_before + time.monotonic() - tick,
                                       **diagnostic})
                finally:
                    budget.charge(time.monotonic() - tick, updates=updated)
                if checkpoint_every and completed % checkpoint_every == 0:
                    _save(checkpoint_state(), checkpoint_path)
                    budget.persist()
                if budget.seconds >= budget.limit:
                    raise BudgetExceeded(summary=finish("BUDGET_EXHAUSTED"))
    except BudgetExceeded as error:
        if error.summary is None:
            error.summary = finish("BUDGET_EXHAUSTED")
            error.state = error.summary["state"]
        raise
    except Exception:
        budget.failed_training_attempts += 1
        finish("FAILED")
        raise
    return finish("COMPLETE" if completed == steps else "PAUSED")


def predict_timed(model, query, budget):
    budget.check()
    tick = time.monotonic()
    try:
        query = query.detach().clone().double()
        if query.ndim != 2 or query.shape[1] != 16 or not bool(torch.isfinite(query).all()):
            raise ValueError("Prediction receives finite query features [m,16] only")
        with numerical_settings():
            mean, variance = model.predict(query)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()):
            raise FloatingPointError("Nonfinite GP prediction")
        if bool((variance < -1e-8).any()):
            raise FloatingPointError("Raw standardized latent variance below -1e-8")
        return mean, variance
    finally:
        budget.charge(time.monotonic() - tick, kind="prediction")
        budget.persist()


def independent_nll(model):
    with numerical_settings():
        return _independent_nll(model)


def check_forks(common, branches=None):
    """Actual each-fit check: identical state and no cross-branch tensor storage."""
    if branches is None:
        named = {name: common.clone() for name in ("FROZEN", "REG", "DKL")}
        return {"checks": check_forks(common, list(named.values())), "branches": named}
    expected = common.fingerprint()
    models = [common] + list(branches)
    storage = []
    for model in models:
        values = list(model.state().values()) + list(model.gp.train_inputs) + [model.gp.train_targets]
        storage.append({v.untyped_storage().data_ptr() for v in values if v.numel()})
    equal = [b.fingerprint() == expected for b in branches]
    overlaps = [{"left": i, "right": j, "shared_storage_count": len(storage[i] & storage[j])}
                for i in range(len(models)) for j in range(i + 1, len(models))
                if storage[i] & storage[j]]
    result = {"pass": all(equal) and not overlaps, "common_fingerprint": expected,
              "branch_fingerprints": [b.fingerprint() for b in branches],
              "cross_model_storage_overlaps": overlaps}
    if not result["pass"]:
        raise RuntimeError("Common-to-three-branch state/storage isolation check failed")
    return result
