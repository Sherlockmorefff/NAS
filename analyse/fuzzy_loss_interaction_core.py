"""Isolated original-fuzzy computations. Only explicitly observed labels enter."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import warnings

import gpytorch
import numpy as np
import torch

from analyse.fuzzy_metric_offline_core import _label_arrays
from analyse.fuzzy_metric_replay import ReplayGP
from surrogate.fuzzy_metric import JointMetricModel, metric_loss, neighborhood_affinity

PERF = {"disk_write_seconds": 0., "diagnostic_backward_seconds": 0.,
        "diagnostic_backward_calls": 0, "prediction_seconds": 0.}


def utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, immutable=False):
    start = time.perf_counter()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    if immutable and path.exists():
        assert path.read_text() == text, str(path)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    PERF["disk_write_seconds"] += time.perf_counter() - start


def append(path, row):
    start = time.perf_counter()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    PERF["disk_write_seconds"] += time.perf_counter() - start


def journal(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def save(path, value):
    start = time.perf_counter()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    PERF["disk_write_seconds"] += time.perf_counter() - start


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def tree_hash(value):
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(str((str(item.dtype), tuple(item.shape))).encode())
            h.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for sub in item:
                visit(sub)
        elif isinstance(item, np.ndarray):
            h.update(str((str(item.dtype), item.shape)).encode())
            h.update(item.tobytes())
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def rng_state():
    return dict(torch=torch.get_rng_state().clone(), numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def reload_model(state):
    model = JointMetricModel(12, seed=0)
    model.load_state_dict(copy.deepcopy(state))
    return model


def adam(parameters, lr):
    # Match the original torch.optim.Adam defaults, including None dispatch flags.
    return torch.optim.Adam(parameters, lr=lr, betas=(.9, .999), eps=1e-8,
                            weight_decay=0., amsgrad=False, foreach=None, fused=None)


def scaler_from_training(features, ids, labels):
    ids = sorted(ids)
    assert set(ids) == set(labels)
    positions = {c: i for i, c in enumerate(features["ids"])}
    ix = [positions[c] for c in ids]
    z = torch.tensor(features["z"], dtype=torch.float32)[ix]
    hp = torch.tensor(features["hp"], dtype=torch.float32)[ix]
    y, _, _ = _label_arrays(ids, labels)
    return dict(fit_ids=ids, z_mean=z.mean(0).tolist(), z_scale=z.std(0, correction=0).clamp_min(1e-6).tolist(),
                hp_mean=hp.mean(0).tolist(), hp_scale=hp.std(0, correction=0).clamp_min(1e-6).tolist(),
                y_mean=float(y.mean()), y_scale=float(y.std(correction=0).clamp_min(1e-8)),
                input_dtype="float32", target_dtype="float64_then_float32_for_mapping", std_correction=0)


def observed_data(features, scaler, labels, epsilon):
    ids = sorted(labels)
    pos = {c: i for i, c in enumerate(features["ids"])}
    ix = [pos[c] for c in ids]
    all_z = (torch.tensor(features["z"], dtype=torch.float32) - torch.tensor(scaler["z_mean"])) / torch.tensor(scaler["z_scale"])
    all_hp = (torch.tensor(features["hp"], dtype=torch.float32) - torch.tensor(scaler["hp_mean"])) / torch.tensor(scaler["hp_scale"])
    y, se, seeds = _label_arrays(ids, labels)
    target64 = (y - scaler["y_mean"]) / scaler["y_scale"]
    paired_se = (seeds[:, None, :] - seeds[None, :, :]).std(-1, correction=1) / math.sqrt(3)
    weights = 1 / (1 + (paired_se / epsilon).square())
    affinity = neighborhood_affinity(y, mode="N2", epsilon=epsilon)
    assert all(torch.isfinite(x).all() for x in [all_z, all_hp, target64, weights])
    return dict(ids=ids, indices=ix, all_ids=features["ids"], all_z=all_z, all_hp=all_hp,
                z=all_z[ix], hp=all_hp[ix], y=y, se=se, seeds=seeds,
                target64=target64, target32=target64.float(), affinity=affinity, weights=weights,
                paired_se=paired_se, epsilon=epsilon, scaler=scaler)


def components(model, data, lambda_f):
    e, prediction = model(data["z"], data["hp"])
    losses = metric_loss(e, prediction, data["target32"], data["affinity"],
                         lambda_f=lambda_f, lambda_e=.001, pair_weights=data["weights"])
    left, right = torch.triu_indices(len(e), len(e), 1)
    clip = max(torch.finfo(e.dtype).eps, 1e-7)
    q = (1 / (1 + (e[left] - e[right]).square().sum(-1))).clamp(clip, 1 - clip)
    p = data["affinity"].float()[left, right]
    w = data["weights"].float()[left, right]
    denom = w.sum()
    pull = -(w * p * q.log()).sum() / denom if len(w) and float(denom) > 0 else e.sum() * 0
    separate = -(w * (1 - p) * torch.log1p(-q)).sum() / denom if len(w) and float(denom) > 0 else e.sum() * 0
    assert all(torch.isfinite(v) for v in losses.values())
    raw_prediction = prediction.detach().double() * data["scaler"]["y_scale"] + data["scaler"]["y_mean"]
    numbers = {k: float(v.detach()) for k, v in losses.items()}
    numbers.update(weighted_regression=numbers["regression"], weighted_neighborhood=lambda_f*numbers["neighborhood"],
                   weighted_scale=.001*numbers["scale"], pull=float(pull.detach()), separation=float(separate.detach()),
                   weighted_pull=lambda_f*float(pull.detach()), weighted_separation=lambda_f*float(separate.detach()),
                   positive_pair_ratio=float((p > 0).double().mean()) if len(p) else 0., pairs=len(p), pair_weight_sum=float(denom),
                   prediction_mae_raw=float((raw_prediction-data["y"]).abs().mean()),
                   prediction_signed_error_raw=float((raw_prediction-data["y"]).mean()),
                   embedding_norm_mean=float(e.detach().norm(dim=1).mean()),
                   head_weight_norm=float(model.regressor.weight.detach().double().norm()))
    return losses, numbers, e, prediction


class Ledger:
    def __init__(self, folder, protocol, synthetic=False):
        self.folder, self.protocol, self.synthetic = Path(folder), protocol, synthetic
        self.folder.mkdir(parents=True, exist_ok=True)
        self.limits = {"mapping": 20 if synthetic else 13398, "gp": 20 if synthetic else 4290}
        self.counts = {k: len(journal(self.folder/(k+"_intents.jsonl"))) for k in self.limits}

    def update(self, kind, task, step, optimizer, purpose):
        assert self.counts[kind] < self.limits[kind], kind+" update budget exhausted"
        self.counts[kind] += 1
        record = dict(id=self.counts[kind], kind=kind, task=task, step=step, purpose=purpose,
                      protocol=self.protocol, time=utc())
        append(self.folder/(kind+"_intents.jsonl"), record)
        optimizer.step()
        append(self.folder/(kind+"_receipts.jsonl"), record)
        return record["id"]


def capture(model, optimizer, completed, identity, metadata, scaler):
    model_state, optimizer_state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    return dict(model=model_state, optimizer=optimizer_state, rng=rng_state(), completed_updates=completed,
                phase="post_update" if completed else "initial_pre_update", identity=identity, metadata=metadata,
                scaler=scaler, model_hash=tree_hash(model_state), optimizer_hash=tree_hash(optimizer_state))


def fit_mapping(folder, initial, data, lambda_f, ledger, metadata, steps=100, shots=(0,10,50,99,100), stop_after=None):
    folder = Path(folder)
    identity = digest(dict(metadata=metadata, ids=data["ids"], scaler=data["scaler"], lambda_f=lambda_f, steps=steps,
                           data=tree_hash({k:data[k] for k in ["z","hp","y","se","seeds","weights","affinity"]})))
    model = reload_model(initial)
    optimizer = adam(model.parameters(), .001)
    if (folder/"latest.pt").exists():
        state = load(folder/"latest.pt")
        assert state["identity"] == identity
        assert state["model_hash"] == tree_hash(state["model"])
        assert state["optimizer_hash"] == tree_hash(state["optimizer"])
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        append(folder/"resume.jsonl", dict(completed_updates=state["completed_updates"], time=utc()))
    else:
        state = capture(model, optimizer, 0, identity, metadata, data["scaler"])
        state.update(history=[], optimization_seconds=0.)
        save(folder/"latest.pt", state)
    if state["completed_updates"] in shots and not (folder/f"post{state['completed_updates']:03}.pt").exists():
        save(folder/f"post{state['completed_updates']:03}.pt", state)
    model.train()
    for before in range(state["completed_updates"], steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses, numbers, _, _ = components(model, data, lambda_f)
        losses["total"].backward()
        gradients = torch.cat([p.grad.detach().reshape(-1).double() for p in model.parameters()])
        assert torch.isfinite(gradients).all()
        gid = ledger.update("mapping", metadata["task"], before+1, optimizer, "formal_mapping")
        seconds = time.perf_counter()-start
        history = state["history"] + [dict(numbers, completed_updates=before, phase="pre_update", actual_update=before+1,
                                               gradient_norm=float(gradients.norm()), ledger_id=gid, seconds=seconds)]
        total_seconds = state["optimization_seconds"] + seconds
        state = capture(model, optimizer, before+1, identity, metadata, data["scaler"])
        state.update(history=history, optimization_seconds=total_seconds)
        save(folder/"latest.pt", state)
        if before+1 in shots:
            save(folder/f"post{before+1:03}.pt", state)
        if stop_after == before+1:
            return state
    write(folder/"training.json", dict(identity=identity, metadata=metadata, lambda_f=lambda_f,
          completed_updates=state["completed_updates"], history=state["history"], seconds=state["optimization_seconds"]))
    return state


def embedding_cache(model, data):
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        e, head = model(data["all_z"], data["all_hp"])
    assert not e.requires_grad and torch.isfinite(e).all() and torch.isfinite(head).all()
    PERF["prediction_seconds"] += time.perf_counter()-started
    return dict(ids=data["all_ids"], embeddings=e.detach().double(),
                head=head.detach().double()*data["scaler"]["y_scale"]+data["scaler"]["y_mean"],
                model_hash=tree_hash(model.state_dict()), scaler=data["scaler"])


def gp_from_state(state):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        gp = ReplayGP(state["x"].detach().double(), state["target"].detach().double(), state["se"].detach().double(), state["scale"]).double()
    gp.load_state_dict(copy.deepcopy(state["model"]))
    assert torch.equal(gp.likelihood.noise_covar.noise, state["fixed_noise_normalized"])
    gp.eval()
    gp.likelihood.eval()
    return gp


def fit_gp(folder, cache, data, ledger, metadata, steps=20, stop_after=None):
    folder = Path(folder)
    x = cache["embeddings"][data["indices"]].detach().double()
    y, se, scale = data["target64"].detach(), data["se"].detach(), data["scaler"]["y_scale"]
    identity = digest(dict(metadata=metadata, model=cache["model_hash"], inputs=tree_hash({"x":x,"y":y,"se":se}), scaler=data["scaler"], steps=steps))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        gp = ReplayGP(x,y,se,scale).double()
    optimizer = adam(gp.parameters(), .05)
    if (folder/"latest.pt").exists():
        state = load(folder/"latest.pt")
        assert state["identity"] == identity
        gp.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        append(folder/"resume.jsonl", dict(completed_updates=state["completed_updates"], time=utc()))
    else:
        state = dict(identity=identity, metadata=metadata, x=x, target=y, se=se, scale=scale, scaler=data["scaler"], train_ids=data["ids"],
                     cache=cache, model=copy.deepcopy(gp.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()), rng=rng_state(),
                     completed_updates=0, phase="initial_pre_update", history=[], optimization_seconds=0.,
                     fixed_noise_normalized=gp.likelihood.noise_covar.noise.detach().clone(), initial_parameters_hash=tree_hash(gp.state_dict()),
                     numerics=dict(dtype="float64", kernel="isotropic_Matern_5_2", gp_lr=.05, jitter=1e-6,
                                   cholesky_max_tries=3, max_cholesky_size=800, min_fixed_noise=0.,
                                   initial_additional_noise_raw_accuracy_variance=1e-8, additional_noise_lower_bound_raw_accuracy_variance=1e-16))
        append(ledger.folder/"gp_fits.jsonl", dict(task=metadata["task"], identity=identity, time=utc()))
        save(folder/"initial.pt", state)
        save(folder/"latest.pt", state)
    gp.train()
    gp.likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood,gp)
    for before in range(state["completed_updates"],steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with warnings.catch_warnings(record=True) as caught, gpytorch.settings.cholesky_jitter(double_value=1e-6), gpytorch.settings.cholesky_max_tries(3), gpytorch.settings.max_cholesky_size(800):
            warnings.simplefilter("always")
            loss = -mll(gp(gp.train_inputs[0]),gp.train_targets)
            assert torch.isfinite(loss)
            loss.backward()
        grad = torch.cat([p.grad.detach().reshape(-1) for p in gp.parameters()])
        assert torch.isfinite(grad).all() and not x.requires_grad
        gid = ledger.update("gp",metadata["task"],before+1,optimizer,"formal_gp")
        seconds = time.perf_counter()-start
        state["history"].append(dict(completed_updates=before,phase="pre_update",actual_update=before+1,loss=float(loss.detach()),
                                     gradient_norm=float(grad.norm()), ledger_id=gid, seconds=seconds, warnings=[str(w.message) for w in caught]))
        state.update(model=copy.deepcopy(gp.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()), rng=rng_state(),
                     completed_updates=before+1, phase="post_update", optimization_seconds=state["optimization_seconds"]+seconds)
        state["model_hash"],state["optimizer_hash"] = tree_hash(state["model"]),tree_hash(state["optimizer"])
        save(folder/"latest.pt",state)
        if stop_after == before+1:
            return state
    write(folder/"training.json",dict(identity=identity,metadata=metadata,completed_updates=steps,history=state["history"],seconds=state["optimization_seconds"]))
    return state


def predict(cache, train_ids, query_ids, gp_state=None):
    started = time.perf_counter()
    positions = {c:i for i,c in enumerate(cache["ids"])}
    train_ids,query_ids = sorted(train_ids),sorted(query_ids)
    assert not set(train_ids)&set(query_ids) and len(train_ids)>=5
    ti,qi = [positions[c] for c in train_ids],[positions[c] for c in query_ids]
    x = cache["embeddings"]
    means = variances = None
    if gp_state is not None:
        assert gp_state["cache"]["model_hash"] == cache["model_hash"]
        assert torch.equal(gp_state["x"],x[ti])
        gp = gp_from_state(gp_state)
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(double_value=1e-6), gpytorch.settings.cholesky_max_tries(3), gpytorch.settings.max_cholesky_size(800):
            posterior = gp(x[qi])
            means = (posterior.mean*cache["scaler"]["y_scale"]+cache["scaler"]["y_mean"]).tolist()
            variances = (posterior.variance.clamp_min(0)*cache["scaler"]["y_scale"]**2).tolist()
    distances = torch.cdist(x[qi],x[ti]).numpy()
    rows = []
    for i,c in enumerate(query_ids):
        nearest = sorted(range(len(train_ids)),key=lambda j:(float(distances[i,j]),train_ids[j]))[:5]
        row = dict(candidate_id=c,head_mean=float(cache["head"][positions[c]]),
                   neighbor_ids=[train_ids[j] for j in nearest],neighbor_distances=[float(distances[i,j]) for j in nearest])
        if means is not None:
            row.update(gp_mean=means[i],latent_variance=variances[i])
        assert all(math.isfinite(v) for k,v in row.items() if isinstance(v,float))
        rows.append(row)
    PERF["prediction_seconds"] += time.perf_counter()-started
    return rows
