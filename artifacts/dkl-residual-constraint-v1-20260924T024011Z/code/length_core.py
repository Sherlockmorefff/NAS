"""Fixed endpoint-LR continuation of a verified Stage-2 DKL checkpoint.

No source paths, query observations, scoring, selection, or scheduler rewriting
occur here. A session owns one Adam instance for joint updates 1001..4000.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time

import gpytorch
import numpy as np
import torch

from stage01_core import tensor_hash
from stage2_core import _optimizer, numerical_settings, restore, snapshot

SOURCE_STEPS = 1000
TARGET_STEPS = 4000
FIXED_LRS = (1e-4, 1e-3)


def exact(left, right):
    """Strict state equality, including tensor dtype/shape and nested metadata."""
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _parameter_groups(model):
    return [[("mapper." + name, parameter) for name, parameter in model.mapper.named_parameters()
             if not name.startswith("regressor.")],
            [("gp." + name, parameter) for name, parameter in model.gp.named_parameters()]]


def _source_schedule_check(state):
    schedule = state.get("scheduler")
    if not isinstance(schedule, dict) or (schedule.get("mode"), schedule.get("planned_steps"),
                                         schedule.get("completed")) != ("joint", SOURCE_STEPS, SOURCE_STEPS):
        raise ValueError("Source must retain joint/planned_steps=1000/completed=1000 scheduler metadata")
    if state.get("model_type", "MAPPED8") != "MAPPED8" or "mapper" not in state:
        raise ValueError("Only the Stage-2 mapped DKL checkpoint can continue")
    if not isinstance(state.get("optimizer"), dict):
        raise ValueError("A prediction-only endpoint cannot substitute for an Adam checkpoint")
    for key in ("x", "y", "known", "rng", "gp", "fingerprint", "history_embedding"):
        if key not in state:
            raise ValueError("Incomplete continuation source: missing " + key)


def _step_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Adam step must be a scalar")
        value = float(value.item())
    else:
        value = float(value)
    if not math.isfinite(value) or value < 1 or value != int(value):
        raise ValueError("Adam parameter step must be a positive finite integer")
    return int(value)


def _check_adam_state(saved, named_groups):
    groups = saved.get("param_groups", [])
    if len(groups) != 2:
        raise ValueError("Expected mapping then GP Adam groups")
    all_ids, steps = [], {}
    for index, (group, names) in enumerate(zip(groups, named_groups)):
        ids = group.get("params", [])
        if len(ids) != len(names):
            raise ValueError("Adam group size does not match frozen named parameter order")
        if float(group.get("lr", -1)) != FIXED_LRS[index]:
            raise ValueError("Saved actual endpoint LR is not mapping=1e-4 / GP=1e-3")
        if float(group.get("base_lr", -1)) != (1e-3, 1e-2)[index]:
            raise ValueError("Saved base_lr identity changed; actual LR must be preserved separately")
        if (tuple(group.get("betas", [])) != (0.9, 0.999)
                or float(group.get("eps", -1)) != 1e-8
                or float(group.get("weight_decay", -1)) != 0.0
                or bool(group.get("amsgrad", False))
                or bool(group.get("maximize", False))):
            raise ValueError("Adam mathematical settings differ from the frozen Stage-2 optimizer")
        for key, (name, parameter) in zip(ids, names):
            item = saved.get("state", {}).get(key)
            if not isinstance(item, dict) or not {"step", "exp_avg", "exp_avg_sq"} <= item.keys():
                raise ValueError("Missing complete Adam state for " + name)
            steps[name] = _step_value(item["step"])
            for moment in ("exp_avg", "exp_avg_sq"):
                value = item[moment]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                        or value.dtype != parameter.dtype or value.device.type != "cpu"
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError("Adam moment shape/dtype/finiteness mismatch for " + name)
            if bool((item["exp_avg_sq"] < 0).any()):
                raise ValueError("Negative Adam second moment for " + name)
        all_ids.extend(ids)
    if len(set(all_ids)) != len(all_ids) or set(saved.get("state", {})) != set(all_ids):
        raise ValueError("Adam parameter identities are duplicated, missing, or extraneous")
    return steps


class TailSession:
    """One immutable-source schedule and one live optimizer per trajectory."""
    def __init__(self):
        raise TypeError("Use TailSession.from_source or TailSession.from_continuation")

    @property
    def source_scheduler(self):
        return copy.deepcopy(self._source_scheduler)

    @property
    def lrs(self):
        return {"mapping": float(self.optimizer.param_groups[0]["lr"]),
                "gp": float(self.optimizer.param_groups[1]["lr"])}

    @classmethod
    def _load(cls, state, continuation):
        _source_schedule_check(state)
        obj = object.__new__(cls)
        obj.model = restore(state)
        if (obj.model.x.dtype != torch.float64 or obj.model.x.device.type != "cpu"
                or obj.model.x.shape != (len(obj.model.y), 16)):
            raise ValueError("Continuation requires CPU float64 selected x[n,16]")
        obj._named_groups = _parameter_groups(obj.model)
        obj.ordered_parameter_names = [[name for name, _ in group] for group in obj._named_groups]
        if continuation and state.get("ordered_parameter_names") != obj.ordered_parameter_names:
            raise ValueError("Continuation named Adam parameter order changed")
        source_steps = _check_adam_state(state["optimizer"], obj._named_groups)
        # Exactly the original group construction and frozen-head convention.
        # load_state_dict then restores real moments, steps and *actual* LRs.
        obj.optimizer = _optimizer(obj.model, "joint")
        obj.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        if not exact(obj.optimizer.state_dict(), state["optimizer"]):
            raise ValueError("Adam state changed while restoring the source")
        obj._source_scheduler = copy.deepcopy(state["scheduler"])
        obj._all_parameters = [parameter for group in obj.optimizer.param_groups for parameter in group["params"]]
        obj._mll = gpytorch.mlls.ExactMarginalLogLikelihood(obj.model.gp.likelihood, obj.model.gp)
        obj.total_joint_step = SOURCE_STEPS
        obj.last_gradient = None
        obj.last_loss_before_update = None
        obj.additional_training_seconds = 0.0
        obj.source_optimizer_parameter_steps = source_steps
        obj.source_state_hash = state["fingerprint"]
        if continuation:
            metadata = state.get("continuation")
            if not isinstance(metadata, dict):
                raise ValueError("Missing continuation metadata")
            total = metadata.get("total_joint_step")
            if (not isinstance(total, int) or not SOURCE_STEPS <= total <= TARGET_STEPS
                    or metadata.get("source_completed_steps") != SOURCE_STEPS
                    or metadata.get("additional_completed_steps") != total - SOURCE_STEPS
                    or metadata.get("continuation_target") != TARGET_STEPS
                    or metadata.get("schedule") != "fixed_endpoint_lr"):
                raise ValueError("Continuation step/target/fixed-LR metadata mismatch")
            if state.get("source_scheduler") != obj._source_scheduler:
                raise ValueError("Original source scheduler was altered")
            initial_steps = metadata.get("source_optimizer_parameter_steps")
            if not isinstance(initial_steps, dict) or initial_steps.keys() != source_steps.keys():
                raise ValueError("Missing per-parameter source Adam steps")
            for name, current in source_steps.items():
                if current != initial_steps[name] + total - SOURCE_STEPS:
                    raise ValueError("Adam parameter steps disagree with additional joint updates for " + name)
            obj.total_joint_step = total
            obj.last_gradient = metadata.get("last_gradient_before_clip")
            obj.last_loss_before_update = metadata.get("last_loss_before_update")
            if total == SOURCE_STEPS:
                if obj.last_gradient is not None or obj.last_loss_before_update is not None:
                    raise ValueError("The zero-additional-update baseline has no new-step gradient/loss")
            elif (obj.last_gradient is None or not math.isfinite(obj.last_gradient)
                  or obj.last_gradient < 0 or obj.last_loss_before_update is None
                  or not math.isfinite(obj.last_loss_before_update)):
                raise ValueError("Invalid last-update diagnostics in continuation")
            obj.additional_training_seconds = float(metadata.get("additional_training_seconds", 0.0))
            if not math.isfinite(obj.additional_training_seconds) or obj.additional_training_seconds < 0:
                raise ValueError("Invalid continuation training cost")
            obj.source_optimizer_parameter_steps = copy.deepcopy(initial_steps)
            obj.source_state_hash = metadata["source_state_hash"]
        elif "continuation" in state:
            raise ValueError("A continued state requires from_continuation, not from_source")
        obj._assert_training_contract()
        return obj

    @classmethod
    def from_source(cls, state):
        return cls._load(state, continuation=False)

    @classmethod
    def from_continuation(cls, state):
        return cls._load(state, continuation=True)

    def _assert_training_contract(self):
        if tuple(float(group["lr"]) for group in self.optimizer.param_groups) != FIXED_LRS:
            raise RuntimeError("Fixed endpoint learning rates were changed")
        if not (self.model.mapper.training and self.model.gp.training and self.model.gp.likelihood.training):
            raise RuntimeError("Training instance was switched to eval; use isolated copies for diagnostics")
        if any(parameter.requires_grad for parameter in self.model.mapper.regressor.parameters()):
            raise RuntimeError("The unused linear regression head must remain frozen")
        if not all(parameter.requires_grad for parameter in self._all_parameters):
            raise RuntimeError("Joint mapping/GP trainable parameter contract changed")

    def step(self, budget):
        """One joint update; no scheduler, parameter reset, or new optimizer."""
        budget.check()
        if self.total_joint_step >= TARGET_STEPS:
            raise RuntimeError("The authorized fixed endpoint 4000 is already complete")
        tick, updated = time.monotonic(), 0
        try:
            self._assert_training_contract()
            self.optimizer.zero_grad(set_to_none=True)
            with numerical_settings():
                embedding, _ = self.model.encode(self.model.x)
                self.model.gp.set_train_data(inputs=embedding, targets=self.model.y, strict=False)
                loss = -self._mll(self.model.gp(embedding), self.model.y)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite continuation pure-data NLL")
                loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(self._all_parameters, 5.0, error_if_nonfinite=True))
            self.optimizer.step()
            self.total_joint_step += 1
            updated = 1
            self.last_gradient = gradient
            self.last_loss_before_update = float(loss.detach())
            self.model.cache_version = None
        except Exception:
            budget.failed_training_attempts += 1
            raise
        finally:
            seconds = time.monotonic() - tick
            self.additional_training_seconds += seconds
            budget.charge(seconds, kind="training", updates=updated)
        # If the last completed update crossed the time bound, the caller can
        # immediately snapshot this intact state without performing another step.
        budget.check()
        return {"total_joint_step": self.total_joint_step,
                "additional_completed_steps": self.total_joint_step - SOURCE_STEPS,
                "loss_before_update": self.last_loss_before_update,
                "gradient_before_clip": self.last_gradient, "seconds": seconds,
                "lr_mapping": FIXED_LRS[0], "lr_gp": FIXED_LRS[1]}

    def snapshot(self):
        """Read-only snapshot; in particular saving at 2000 does not reset Adam."""
        state = snapshot(self.model, self.optimizer)
        state["scheduler"] = copy.deepcopy(self._source_scheduler)
        state["source_scheduler"] = copy.deepcopy(self._source_scheduler)
        state["ordered_parameter_names"] = copy.deepcopy(self.ordered_parameter_names)
        state["continuation"] = {
            "source_completed_steps": SOURCE_STEPS,
            "total_joint_step": self.total_joint_step,
            "additional_completed_steps": self.total_joint_step - SOURCE_STEPS,
            "continuation_target": TARGET_STEPS,
            "schedule": "fixed_endpoint_lr",
            "actual_lrs": self.lrs,
            "last_gradient_before_clip": self.last_gradient,
            "last_gradient_applies_to_update": None if self.total_joint_step == SOURCE_STEPS else self.total_joint_step,
            "last_loss_before_update": self.last_loss_before_update,
            "additional_training_seconds": self.additional_training_seconds,
            "source_optimizer_parameter_steps": copy.deepcopy(self.source_optimizer_parameter_steps),
            "source_state_hash": self.source_state_hash,
        }
        return state

    def optimizer_audit(self):
        """JSON evidence linking every saved moment/step to its ordered name."""
        serial = self.optimizer.state_dict()
        current_steps = _check_adam_state(serial, self._named_groups)
        groups = []
        all_tensors = {}
        for index, (group, names) in enumerate(zip(serial["param_groups"], self._named_groups)):
            parameters = []
            for key, (name, parameter) in zip(group["params"], names):
                item = serial["state"][key]
                tensors = {k: v for k, v in item.items() if isinstance(v, torch.Tensor)}
                all_tensors.update({name + "." + k: v for k, v in tensors.items()})
                parameters.append({"name": name, "saved_parameter_id": key,
                    "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                    "step": current_steps[name], "source_step": self.source_optimizer_parameter_steps[name],
                    "step_dtype": str(item["step"].dtype) if isinstance(item["step"], torch.Tensor) else type(item["step"]).__name__,
                    "state_keys": sorted(item),
                    "exp_avg_sha256": tensor_hash({"exp_avg": item["exp_avg"]}),
                    "exp_avg_sq_sha256": tensor_hash({"exp_avg_sq": item["exp_avg_sq"]}),
                    "all_tensor_state_sha256": tensor_hash(tensors),
                    "exp_avg_l2": float(item["exp_avg"].norm()),
                    "exp_avg_sq_l2": float(item["exp_avg_sq"].norm())})
            groups.append({"index": index, "name": ("mapping", "gp")[index],
                           "actual_lr": float(group["lr"]), "saved_base_lr": float(group["base_lr"]),
                           "betas": list(group["betas"]), "eps": group["eps"],
                           "weight_decay": group["weight_decay"],
                           "ordered_parameter_names": [name for name, _ in names], "parameters": parameters})
        metadata = [{k: v for k, v in group.items() if k != "parameters"} for group in groups]
        return {"total_joint_step": self.total_joint_step, "groups": groups,
                "parameter_count": sum(len(group) for group in self._named_groups),
                "parameter_order_origin": "frozen stage2_core._optimizer(mode=joint): mapping_params then gp.parameters",
                "moments_and_steps_sha256": tensor_hash(all_tensors),
                "group_metadata_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "source_scheduler_unchanged": self.source_scheduler,
                "all_steps_match_added_updates": all(current_steps[name] == initial + self.total_joint_step - SOURCE_STEPS
                                                      for name, initial in self.source_optimizer_parameter_steps.items())}
