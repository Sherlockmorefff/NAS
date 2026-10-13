"""Explicit W-HI kernel; optimizer numerics copied from frozen phase1 with RNG/work logging added.
Old modules are unchanged. Only W-HI may be optimized here.
"""
from __future__ import annotations
import copy
import math
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import GreaterThan
from linear_operator.utils.errors import NotPSDError

from analyse.structured_kernel_phase1.io import append_jsonl, read_json, write_json, now
import random
from .common import KERNEL_KIND
from analyse.structured_kernel_phase1.protocol import STARTS, LBFGS


def cpu_setup():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64, device="cpu").clone()


def positive_median(x):
    x = np.asarray(x, dtype=np.float64)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    values = d[np.triu_indices(len(x), 1)]
    positive = values[values > 0]
    return {"value": float(np.median(positive)) if len(positive) else 1.,
            "degenerate": len(positive) == 0, "positive_pairs": len(positive)}


def make_inputs(training, query, method):
    """Only T scalers and T positive distances determine all transformations."""
    t = np.asarray(training["x_standardized"], dtype=np.float64)
    s = training["scaler"]
    q = (np.asarray(query["x_raw"], dtype=np.float64) - s["x_mean"]) / s["x_scale"]
    if t.shape[1] != 16 or q.shape[1] != 16:
        raise ValueError("Expected latent12 plus HP4")
    if method == "RAW16":
        scales = {"raw16": positive_median(t)}
        tx, qx = t / scales["raw16"]["value"], q / scales["raw16"]["value"]
        arch_dim = 16
    else:
        at, aq = (t[:, :12], q[:, :12]) if method.startswith("L-") else (
            np.asarray(training["wl"], dtype=np.float64), np.asarray(query["wl"], dtype=np.float64))
        scales = {"architecture": positive_median(at), "hp": positive_median(t[:, 12:])}
        tx = np.concatenate((at / scales["architecture"]["value"], t[:, 12:] / scales["hp"]["value"]), 1)
        qx = np.concatenate((aq / scales["architecture"]["value"], q[:, 12:] / scales["hp"]["value"]), 1)
        arch_dim = at.shape[1]
    if not (np.isfinite(tx).all() and np.isfinite(qx).all()):
        raise ValueError("Nonfinite transformed features")
    return tensor(tx), tensor(qx), arch_dim, scales


class StructuredMaternKernel(gpytorch.kernels.Kernel):
    def __init__(self, arch_dim, kernel_kind=KERNEL_KIND):
        super().__init__()
        if kernel_kind != "hp_plus_interaction":raise ValueError("Unknown explicit kernel_kind")
        self.arch_dim, self.kernel_kind = int(arch_dim), kernel_kind
        self.arch = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=None,
            active_dims=tuple(range(arch_dim)), lengthscale_constraint=GreaterThan(1e-4))
        self.hp = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=None,
            active_dims=tuple(range(arch_dim, arch_dim + 4)), lengthscale_constraint=GreaterThan(1e-4))

    def forward(self, x1, x2, diag=False, **params):
        ka = self.arch(x1, x2, diag=diag, **params)
        kh = self.hp(x1, x2, diag=diag, **params)
        if self.kernel_kind != "hp_plus_interaction":raise ValueError("Invalid kernel dispatch")
        return (1./3.) * kh + (2./3.) * ka * kh


class StaticExactGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, known, method, arch_dim, initial_ell=1.):
        if method != "W-HI":
            raise ValueError("Unknown method")
        if not all(bool(torch.isfinite(v).all()) for v in (x,y,known)) or bool((known < 0).any()):
            raise ValueError("Invalid training tensors")
        with gpytorch.settings.min_fixed_noise(double_value=0.):
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=known.clone(), learn_additional_noise=True, noise_constraint=GreaterThan(1e-6))
        super().__init__(x, y, likelihood)
        self.method, self.arch_dim = method, int(arch_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        base = StructuredMaternKernel(arch_dim, kernel_kind=KERNEL_KIND)
        self.kernel_kind = KERNEL_KIND
        self.covar_module = gpytorch.kernels.ScaleKernel(base, outputscale_constraint=GreaterThan(1e-8))
        self.double()
        self.mean_module.constant = 0.
        self.covar_module.outputscale = 1.
        self.likelihood.second_noise = .01
        base.arch.lengthscale = initial_ell
        base.hp.lengthscale = initial_ell
        count = sum(p.numel() for p in self.parameters())
        if count != 5:
            raise AssertionError(f"Unexpected learnable parameter count: {count}")
        if list(self.named_priors()) or list(self.named_added_loss_terms()):
            raise AssertionError("No priors or additional losses allowed")
        if not torch.equal(self.likelihood.noise_covar.noise, known):
            raise AssertionError("Known variance was clipped")

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


@contextmanager
def settings(jitter):
    with ExitStack() as stack:
        for setting in (gpytorch.settings.cholesky_jitter(double_value=jitter),
                        gpytorch.settings.cholesky_max_tries(1),
                        gpytorch.settings.max_cholesky_size(10000),
                        gpytorch.settings.min_fixed_noise(double_value=0.),
                        gpytorch.settings.fast_pred_var(False)):
            stack.enter_context(setting)
        yield


class NumericalPolicy:
    def __init__(self, index=0, events=None):
        self.index, self.events = index, list(events or [])

    @property
    def jitter(self):
        return (1e-8, 1e-7, 1e-6)[self.index]

    def evaluate(self, function, model, where):
        while True:
            try:
                with settings(self.jitter):
                    return function()
            except (NotPSDError, torch.linalg.LinAlgError) as exc:
                self.events.append({"at": where, "jitter": self.jitter, "error": str(exc), "time": now()})
                if self.index == 2:
                    raise
                self.index += 1
                model.prediction_strategy = None


def parameter_values(model):
    base = model.covar_module.base_kernel
    result = {"mean": float(model.mean_module.constant.detach()),
              "outputscale": float(model.covar_module.outputscale.detach()),
              "extra_noise": float(model.likelihood.second_noise.detach())}
    if model.method == "RAW16":
        result["lengthscale"] = float(base.lengthscale.detach().item())
    else:
        result.update(ell_a=float(base.arch.lengthscale.detach().item()),ell_h=float(base.hp.lengthscale.detach().item()))
    return result


def _save_torch(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def independent_nll(model, x, y, known, jitter=1e-8, return_jitter=False):
    """Independent Gaussian-density formula using the exact latent covariance."""
    with torch.no_grad():
        k = model.covar_module(x).to_dense().numpy()
    p = parameter_values(model)
    covariance = k + np.diag(known.numpy() + p["extra_noise"])
    used_jitter=0.
    try:
        l = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        # Same numerical ceiling as the actual GP calculation; never inflate noise.
        l = np.linalg.cholesky(covariance+jitter*np.eye(len(y)))
        used_jitter=float(jitter)
    residual = y.numpy() - p["mean"]
    solve = np.linalg.solve(l, residual)
    value=float((.5 * np.dot(solve,solve) + np.log(np.diag(l)).sum() + .5 * len(y)*np.log(2*np.pi)) / len(y))
    return (value,used_jitter) if return_jitter else value


def restore_model(state):
    if state.get("kernel_kind") != KERNEL_KIND:raise ValueError("Missing explicit checkpoint kernel_kind")
    model = StaticExactGP(state["x"],state["y"],state["known"],state["method"],state["arch_dim"])
    model.load_state_dict(state["model"])
    model.prediction_strategy = None
    return model


def fit_start(x, y, known, method, arch_dim, start_id, checkpoint_path, identity, resume=True):
    """Optimize using T tensors only. A start never receives Q data or metrics."""
    cpu_setup()
    x,y,known = (tensor(v) for v in (x,y,known))
    path = Path(checkpoint_path)
    result_path = path.with_suffix(".json")
    log_path = path.with_suffix(".jsonl")
    started = time.perf_counter()
    old = None
    if resume and path.exists():
        old = torch.load(path,map_location="cpu",weights_only=False)
        if old["identity"] != identity or old["start_id"] != start_id:
            raise ValueError("Checkpoint identity mismatch")
        if old["phase"] in ("COMPLETE","FAILED"):
            result = read_json(result_path) if result_path.exists() else old.get("result")
            if result is None:
                raise RuntimeError("Terminal checkpoint has no recoverable summary; refusing another optimization budget")
            if not result_path.exists():
                write_json(result_path,result)
            return result
    append_jsonl(log_path,{"event":"physical_attempt","time":now(),"resumed":old is not None,"saved_phase":old["phase"] if old else None,"saved_adam":old.get("adam_completed",0) if old else 0})
    if old and "rng" in old:
        torch.set_rng_state(old["rng"]["torch"]);np.random.set_state(old["rng"]["numpy"]);random.setstate(old["rng"]["python"])
    model = restore_model(old) if old else StaticExactGP(x,y,known,method,arch_dim,STARTS[start_id])
    if old:
        append_jsonl(log_path,{"event":"resume","at":now(),"saved_phase":old["phase"],
            "saved_adam_completed":old.get("adam_completed",0),
            "lbfgs_replay_from_adam_boundary":old["phase"]=="LBFGS_PENDING",
            "note":"Interrupted work beyond last checkpoint remains in the append-only log; not a new start."})
    policy = NumericalPolicy(old.get("jitter_index",0),old.get("jitter_events",[])) if old else NumericalPolicy()
    losses = list(old.get("adam_losses",[])) if old else []
    completed = int(old.get("adam_completed",0)) if old else 0
    previous_seconds = float(old.get("elapsed_seconds",0)) if old else 0.
    optimizer = torch.optim.Adam(model.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    if old and old.get("optimizer") and old["phase"] == "ADAM":
        optimizer.load_state_dict(old["optimizer"])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)

    def snapshot(phase, opt=None):
        return {"identity":identity,"start_id":start_id,"method":method,"arch_dim":arch_dim,"kernel_kind":KERNEL_KIND,
                "rng":{"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":random.getstate()},
                "x":x,"y":y,"known":known,"model":copy.deepcopy(model.state_dict()),
                "optimizer":None if opt is None else copy.deepcopy(opt.state_dict()),
                "phase":phase,"adam_completed":completed,"adam_losses":losses,
                "jitter_index":policy.index,"jitter_events":policy.events,
                "elapsed_seconds":previous_seconds+time.perf_counter()-started}

    def objective():
        model.train()
        model.likelihood.train()
        model.prediction_strategy = None
        return -mll(model(x),y)

    def closure(opt, where):
        opt.zero_grad(set_to_none=True)
        loss = policy.evaluate(objective,model,where)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training NLL")
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite gradient")
        return loss

    if old is None:_save_torch(path,snapshot("ADAM",optimizer))
    lb_calls = 0
    lb_losses = []
    try:
        while completed < 300:
            loss = closure(optimizer,f"adam:{completed}")
            losses.append(float(loss.detach()))
            optimizer.step()
            completed += 1
            append_jsonl(log_path,{"stage":"adam","update":completed,"loss_before_update":losses[-1]})
            if completed % 25 == 0:
                _save_torch(path,snapshot("ADAM",optimizer))
        adam_final = float(policy.evaluate(objective,model,"adam_final").detach())
        adam_improvement = losses[250] - adam_final
        _save_torch(path,snapshot("LBFGS_PENDING",optimizer))
        lb = torch.optim.LBFGS(model.parameters(),**LBFGS)

        def lb_closure():
            nonlocal lb_calls
            loss = closure(lb,f"lbfgs:{lb_calls}")
            lb_calls += 1
            lb_losses.append(float(loss.detach()))
            append_jsonl(log_path,{"stage":"lbfgs","closure":lb_calls,"nll":lb_losses[-1]})
            return loss

        lb.step(lb_closure)
        loss = closure(lb,"final")
        final_nll = float(loss.detach())
        values = parameter_values(model)
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError("Nonfinite final parameters")
        gradients = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        state = lb.state[next(iter(model.parameters()))]
        iterations = int(state.get("n_iter",0))
        evaluations = int(state.get("func_evals",lb_calls))
        if iterations >= 100:
            reason = "max_iter"
        elif evaluations >= 125:
            reason = "max_eval"
        elif float(gradients.abs().max()) <= LBFGS["tolerance_grad"]:
            reason = "gradient_tolerance"
        else:
            reason = "step_or_objective_tolerance"
        independent, independent_jitter = independent_nll(model,x,y,known,policy.jitter,return_jitter=True)
        if not np.isclose(final_nll,independent,atol=1e-7,rtol=1e-7):
            raise AssertionError(f"Independent NLL mismatch {final_nll} vs {independent}")
        result = {"success":True,"start_id":start_id,"identity":identity,"initial_ell":STARTS[start_id],
                  "final_nll":final_nll,"independent_nll":independent,"independent_nll_jitter":independent_jitter,"parameters":values,
                  "parameter_count":sum(p.numel() for p in model.parameters()),"kernel_kind":KERNEL_KIND,
                  "adam_updates":completed,"adam_final50_improvement":adam_improvement,
                  "lbfgs_iterations":iterations,"lbfgs_evaluations":evaluations,"end_reason":reason,
                  "final_gradient_norm":float(gradients.norm()),
                  "final_gradient_max":float(gradients.abs().max()),
                  "lbfgs_last_objective_change":lb_losses[-2]-final_nll if len(lb_losses)>1 else None,
                  "convergence_warning":adam_improvement>1e-4 or reason in ("max_iter","max_eval"),
                  "jitter_setting":policy.jitter,"jitter_events":policy.events,
                  "jitter_semantics":"GPyTorch attempts unjittered factorization, then configured jitter on failure",
                  "elapsed_seconds":previous_seconds+time.perf_counter()-started,"completed_at":now()}
        _save_torch(path,{**snapshot("COMPLETE",lb),"result":result})
        write_json(result_path,result)
        return result
    except Exception as exc:
        result = {"success":False,"start_id":start_id,"identity":identity,"initial_ell":STARTS[start_id],
                  "error_type":type(exc).__name__,"error":str(exc),"adam_updates":completed,
                  "lbfgs_evaluations":lb_calls,"jitter_setting":policy.jitter,"jitter_events":policy.events,
                  "elapsed_seconds":previous_seconds+time.perf_counter()-started,"completed_at":now()}
        _save_torch(path,{**snapshot("FAILED",optimizer),"result":result})
        write_json(result_path,result)
        return result


def select_start(results):
    valid = [r for r in results if r.get("success") and math.isfinite(r["final_nll"])]
    if not valid:
        raise RuntimeError("No finite successful starting point")
    return min(valid,key=lambda r:(r["final_nll"],r["start_id"]))["start_id"]


def predict_frozen(checkpoint_path, query_x):
    state = torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    model = restore_model(state)
    policy = NumericalPolicy(state["jitter_index"],state["jitter_events"])
    def predict():
        model.eval()
        model.likelihood.eval()
        posterior = model(tensor(query_x))
        # Bypass Distribution.variance default minimum-variance clipping.
        return posterior.mean.detach(),posterior.lazy_covariance_matrix.diagonal().detach()
    with torch.no_grad():
        mu,var = policy.evaluate(predict,model,"prediction")
    if not bool(torch.isfinite(mu).all() and torch.isfinite(var).all()) or bool((var < -1e-8).any()):
        raise FloatingPointError("Nonfinite or invalid latent posterior")
    return {"mean_standardized":mu.tolist(),"latent_variance_standardized":var.tolist(),
            "extra_noise_variance_standardized":parameter_values(model)["extra_noise"],
            "jitter_setting":policy.jitter,"jitter_events":policy.events}
