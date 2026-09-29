"""Isolated stage-1 exact GP experiment; receives selected observations only."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import GreaterThan
from source_reference.fuzzy_metric import JointMetricModel

ATOL, RTOL = 1e-8, 1e-6


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_setup(seed):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, known):
        with gpytorch.settings.min_fixed_noise(double_value=0.0):
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=known.clone(), learn_additional_noise=True,
                noise_constraint=GreaterThan(1e-6),
            )
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=None,
                                         lengthscale_constraint=GreaterThan(1e-4)),
            outputscale_constraint=GreaterThan(1e-8),
        )
        self.double()
        self.mean_module.constant = 0.0
        self.covar_module.base_kernel.lengthscale = 1.0
        self.covar_module.outputscale = 1.0
        self.likelihood.second_noise = 0.01
        if list(self.named_priors()) or list(self.named_added_loss_terms()):
            raise RuntimeError("Pure data objective must not register priors or added losses")

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


class Branch:
    def __init__(self, x, y, known, seed):
        self.x, self.y, self.known = (v.detach().clone().double() for v in (x, y, known))
        if self.x.shape != (len(y), 16) or known.shape != y.shape:
            raise ValueError("Only aligned selected x16/y/variance vectors are accepted")
        if not all(torch.isfinite(t).all() for t in (self.x, self.y, self.known)):
            raise ValueError("Nonfinite selected data")
        if (self.known < 0).any():
            raise ValueError("Known variance must be nonnegative")
        self.mapper = JointMetricModel(z_dim=12, seed=seed).double()
        with torch.no_grad():
            e, _ = self.encode(self.x)
        self.gp = ExactGP(e.clone(), self.y, self.known)
        self.cache_version = None

    def encode(self, x):
        return self.mapper(x[:, :12], x[:, 12:])

    def state(self):
        return {**{"mapper."+k: v for k,v in self.mapper.state_dict().items()},
                **{"gp."+k: v for k,v in self.gp.state_dict().items()},
                "data.x": self.x, "data.y": self.y, "data.known": self.known,
                "likelihood.fixed_noise": self.gp.likelihood.noise_covar.noise}

    def fingerprint(self):
        return tensor_hash(self.state())

    def clone(self):
        # ExactGP training inputs may hold an autograd graph: rebuild them explicitly.
        result = Branch(self.x, self.y, self.known, 0)
        result.mapper.load_state_dict(copy.deepcopy(self.mapper.state_dict()))
        result.gp.load_state_dict(copy.deepcopy(self.gp.state_dict()))
        result.synchronize_history()
        return result

    def synchronize_history(self):
        self.mapper.eval()
        with torch.no_grad():
            history, _ = self.encode(self.x)
        self.gp.set_train_data(inputs=history.detach().clone(), targets=self.y, strict=False)
        self.gp.prediction_strategy = None
        self.cache_version = None

    def mapping_params(self):
        return [v for k,v in self.mapper.named_parameters() if not k.startswith("regressor.")]

    def objective(self, mode):
        self.gp.train()
        self.mapper.train(mode in ("regression", "joint"))
        e, pred = self.encode(self.x)
        if mode == "regression":
            regression = (pred-self.y).square().mean()
            norm = e.square().sum(-1).mean()
            return regression+0.001*norm, {"regression": float(regression.detach()),
                                         "embedding_norm": float(norm.detach()), "data_nll": None}
        if mode != "joint":
            e = e.detach()
        self.gp.set_train_data(inputs=e, targets=self.y, strict=False)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        with gpytorch.settings.cholesky_jitter(double_value=1e-8), gpytorch.settings.max_cholesky_size(10000):
            loss = -mll(self.gp(e), self.y)
        return loss, {"regression": None, "embedding_norm": None, "data_nll": float(loss.detach())}

    def predict(self, query):
        self.mapper.eval()
        version = self.fingerprint()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False), gpytorch.settings.cholesky_jitter(double_value=1e-8):
            if version != self.cache_version:
                history, _ = self.encode(self.x)
                self.gp.set_train_data(inputs=history.detach(), targets=self.y, strict=False)
                self.gp.prediction_strategy = None
                self.cache_version = version
            self.gp.eval()
            self.gp.likelihood.eval()
            features, _ = self.encode(query)
            posterior = self.gp(features)
            # Inspect the raw diagonal, avoiding variance-property clamping that
            # could hide a negative numerical posterior variance.
            return posterior.mean.clone(), posterior.lazy_covariance_matrix.diagonal().clone()

    def kernel(self):
        self.mapper.eval()
        with torch.no_grad():
            e, _ = self.encode(self.x)
            return self.gp.covar_module(e).to_dense().clone()

    def snapshot(self, optimizer=None, block=None, completed=0):
        self.synchronize_history()
        return {"mapper": copy.deepcopy(self.mapper.state_dict()),
                "gp": copy.deepcopy(self.gp.state_dict()),
                "x": self.x.clone(), "y": self.y.clone(), "known": self.known.clone(),
                "fingerprint": self.fingerprint(),
                "history_embedding": self.gp.train_inputs[0].detach().clone(),
                "optimizer": None if optimizer is None else copy.deepcopy(optimizer.state_dict()),
                "scheduler": {"block": block, "completed": completed,
                              "policy": "first 80 percent base LR, final 20 percent base LR / 10"},
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                        "python": random.getstate()}}

    @classmethod
    def restore(cls, state):
        model = cls(state["x"], state["y"], state["known"], 0)
        model.mapper.load_state_dict(state["mapper"])
        model.gp.load_state_dict(state["gp"])
        model.synchronize_history()
        if not torch.equal(model.gp.train_inputs[0], state["history_embedding"]):
            raise RuntimeError("Restored history does not match current mapping")
        if model.fingerprint() != state["fingerprint"]:
            raise RuntimeError("Checkpoint state fingerprint mismatch")
        torch.set_rng_state(state["rng"]["torch"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        return model


def delta(before, after):
    return math.sqrt(sum(float((after[k].detach()-v).square().sum()) for k,v in before.items()))


def gradient_norm(params):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in params if p.grad is not None))


class Budget:
    """Persist elapsed real training, including incomplete attempts and verification updates."""
    def __init__(self, path=None, limit=1200):
        self.path = Path(path) if path else None
        self.limit = limit
        self.seconds = json.loads(self.path.read_text())["seconds"] if self.path and self.path.exists() else 0.0

    def charge(self, seconds):
        self.seconds += seconds
        if self.path:
            self.path.write_text(json.dumps({"seconds": self.seconds, "limit": self.limit}, indent=2))
        self.check()

    def check(self):
        if self.seconds >= self.limit:
            raise TimeoutError("Cumulative real training budget exhausted")


def train_block(model, mode, steps, traces, run, branch, block, budget,
                checkpoint_dir=None, resume=None, stop_at=None):
    for p in model.mapper.parameters():
        p.requires_grad_(mode == "regression")
    for p in model.mapping_params():
        p.requires_grad_(mode in ("regression", "joint"))
    for p in model.gp.parameters():
        p.requires_grad_(mode != "regression")
    groups = []
    if mode == "regression":
        groups.append({"params": list(model.mapper.parameters()), "lr": 1e-3, "base_lr": 1e-3})
    if mode == "joint":
        groups.append({"params": model.mapping_params(), "lr": 1e-3, "base_lr": 1e-3})
    if mode != "regression":
        groups.append({"params": list(model.gp.parameters()), "lr": 1e-2, "base_lr": 1e-2})
    optimizer = torch.optim.Adam(groups, weight_decay=0.0)
    start = 0
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        start = resume["scheduler"]["completed"]
        if (resume["scheduler"]["block"] != block or resume["scheduler"]["mode"] != mode
                or resume["scheduler"]["planned_steps"] != steps or not 0 <= start <= steps):
            raise ValueError("Resume block/mode/schedule mismatch")
    all_params = [p for g in optimizer.param_groups for p in g["params"]]
    for step in range(start, steps if stop_at is None else stop_at):
        budget.check()
        tick = time.perf_counter()
        try:
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * (0.1 if step >= int(0.8*steps) else 1.0)
            model.mapper.zero_grad(set_to_none=True)
            model.gp.zero_grad(set_to_none=True)
            before = {k:v.detach().clone() for k,v in model.mapper.state_dict().items()}
            before_gp = {k:v.detach().clone() for k,v in model.gp.named_parameters()}
            before_id = model.fingerprint()
            loss, components = model.objective(mode)
            loss.backward()
            map_grad = gradient_norm(model.mapping_params())
            gp_grad = gradient_norm(model.gp.parameters())
            head_grad = gradient_norm(model.mapper.regressor.parameters())
            grad = float(torch.nn.utils.clip_grad_norm_(all_params, 5.0, error_if_nonfinite=True))
            if not math.isfinite(float(loss.detach())):
                raise FloatingPointError("Nonfinite loss")
            optimizer.step()
            model.cache_version = None
            map_delta = delta(before, model.mapper.state_dict())
            body_delta = delta({k:v for k,v in before.items() if not k.startswith("regressor.")},
                               model.mapper.state_dict())
            head_delta = delta({k:v for k,v in before.items() if k.startswith("regressor.")},
                               model.mapper.state_dict())
            gp_delta = delta(before_gp, dict(model.gp.named_parameters()))
            traces.append({"run": run, "branch": branch, "block": block, "mode": mode,
                           "step": step+1, "planned_steps": steps, "loss_before_update": float(loss.detach()),
                           **components, "gradient_before_clip": grad, "mapping_gradient": map_grad,
                           "head_gradient": head_grad, "gp_gradient": gp_grad,
                           "mapping_state_delta": map_delta, "gp_parameter_delta": gp_delta,
                           "mapping_body_delta": body_delta, "head_delta": head_delta,
                           "lr_mapping": next((g["lr"] for g in optimizer.param_groups if g["base_lr"]==1e-3), 0.0),
                           "lr_gp": next((g["lr"] for g in optimizer.param_groups if g["base_lr"]==1e-2), 0.0),
                           "lengthscale": float(model.gp.covar_module.base_kernel.lengthscale.detach().item()),
                           "extra_noise": float(model.gp.likelihood.second_noise.detach().item()),
                           "outputscale": float(model.gp.covar_module.outputscale.detach()),
                           "state_before": before_id, "state_after": model.fingerprint(),
                           "seconds": time.perf_counter()-tick})
        finally:
            budget.charge(time.perf_counter()-tick)
    completed = steps if stop_at is None else stop_at
    state = model.snapshot(optimizer, block, completed)
    state["scheduler"].update(mode=mode, planned_steps=steps)
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(state, Path(checkpoint_dir)/f"{run}_{branch}_{block}.pt")
    return state


def compare(a, b):
    return {"pass": bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL)),
            "max_abs": float((a-b).abs().max()), "atol": ATOL, "rtol": RTOL}


def independent_nll(model):
    """NumPy kernel + independent dense Cholesky, including all observation noise."""
    with torch.no_grad():
        e, _ = model.encode(model.x)
    x = e.numpy()
    ell = float(model.gp.covar_module.base_kernel.lengthscale.detach().item())
    scale = float(model.gp.covar_module.outputscale.detach())
    extra = float(model.gp.likelihood.second_noise.detach().item())
    distance = np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2, axis=-1))/ell
    kernel = scale*(1+np.sqrt(5)*distance+5*distance**2/3)*np.exp(-np.sqrt(5)*distance)
    covariance = kernel+np.diag(model.known.numpy()+extra)
    chol = np.linalg.cholesky(covariance)
    centered = model.y.numpy()-float(model.gp.mean_module.constant.detach())
    whitened = np.linalg.solve(chol, centered)
    expected = (0.5*np.dot(whitened,whitened)+np.log(np.diag(chol)).sum()+0.5*len(x)*np.log(2*np.pi))/len(x)
    actual, _ = model.objective("gp")
    return {"pass": bool(np.isclose(float(actual.detach()),expected,atol=ATOL,rtol=RTOL)),
            "gpytorch_nll_per_sample": float(actual.detach()), "numpy_cholesky_nll_per_sample": float(expected),
            "abs_error": abs(float(actual.detach())-expected), "registered_priors": [], "added_losses": []}
