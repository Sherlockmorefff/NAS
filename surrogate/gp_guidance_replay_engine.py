"""Replay computation: explicit revealed labels only, no oracle or GNN training."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import random
import re
import time
import warnings
import numpy as np
import torch
import gpytorch
from analyse.relative_metric_pilot import read, write, save, load, digest, tensor_hash, utc
from analyse.fuzzy_metric_offline_core import _label_arrays
from analyse.fuzzy_metric_replay import ReplayGP
from surrogate.fuzzy_metric import JointMetricModel
from surrogate.relative_metric import fresh_adam
from surrogate.gp_mean_guidance import architecture_groups, from_gp, posterior
from surrogate.gp_mean_guidance_sync import objective, SolveMeter, gradient_probe


def append(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def journal(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []


class Ledger:
    """Write intent before actual update; unconfirmed intents stay charged."""
    def __init__(self, root, identity, limits):
        self.root, self.identity, self.limits = Path(root), identity, limits
        self.counts = {k: len(journal(self.root / (k + '-intents.jsonl')))
                       for k in ['mapping', 'gp', 'gp_fits']}

    def intent(self, kind, task, step):
        if self.counts[kind] >= self.limits[kind]:
            raise RuntimeError(f'{kind} hard budget exhausted')
        self.counts[kind] += 1
        gid = self.counts[kind]
        append(self.root / (kind + '-intents.jsonl'), dict(id=gid, task=task,
            step=step, protocol=self.identity, time=utc()))
        return gid

    def update(self, kind, task, step, optimizer):
        gid = self.intent(kind, task, step)
        optimizer.step()
        append(self.root / (kind + '-receipts.jsonl'), dict(id=gid, task=task, step=step))
        return gid


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def reload_model(state):
    model = JointMetricModel(12, seed=0)
    model.load_state_dict(copy.deepcopy(state))
    return model


def scaler_from_initial(features, initial_ids, labels):
    ids = sorted(initial_ids)
    pos = {cid: i for i, cid in enumerate(features['ids'])}
    z = torch.tensor(features['z'], dtype=torch.float32)[[pos[c] for c in ids]]
    hp = torch.tensor(features['hp'], dtype=torch.float32)[[pos[c] for c in ids]]
    y, _, _ = _label_arrays(ids, labels)
    return dict(fit_ids=ids, z_mean=z.mean(0).tolist(), z_scale=z.std(0, correction=0).clamp_min(1e-6).tolist(),
        hp_mean=hp.mean(0).tolist(), hp_scale=hp.std(0, correction=0).clamp_min(1e-6).tolist(),
        y_mean=float(y.mean()), y_scale=float(y.std(correction=0).clamp_min(1e-8)),
        input_std_floor=1e-6, target_std_floor=1e-8, population_std=True)


def input_tensors(features, scaler, labels):
    ids = features['ids']
    z = (torch.tensor(features['z'], dtype=torch.float32) - torch.tensor(scaler['z_mean'])) / torch.tensor(scaler['z_scale'])
    hp = (torch.tensor(features['hp'], dtype=torch.float32) - torch.tensor(scaler['hp_mean'])) / torch.tensor(scaler['hp_scale'])
    train_ids = sorted(labels)
    positions = {c: i for i, c in enumerate(ids)}
    indices = [positions[c] for c in train_ids]
    y, se, _ = _label_arrays(train_ids, labels)
    target = (y - scaler['y_mean']) / scaler['y_scale']
    assert z.shape == (179, 12) and hp.shape == (179, 4)
    assert torch.isfinite(z).all() and torch.isfinite(hp).all()
    return dict(ids=train_ids, indices=indices, z=z[indices], hp=hp[indices],
        all_z=z, all_hp=hp, all_ids=ids, y=y, se=se, target64=target, target32=target.float(),
        architectures=[features['architectures'][i] for i in indices], scaler=scaler)


def pause_once(folder, name, enabled, details):
    path = Path(folder) / ('pause-' + name + '.json')
    if enabled and not path.exists():
        write(path, dict(time=utc(), **details))
        raise SystemExit(75)


def fit_gp(folder, x, y, se, scale, metadata, ledger, recovery_test=False):
    folder = Path(folder)
    ident = digest(dict(metadata=metadata, inputs=tensor_hash({'x': x, 'y': y, 'se': se}), scale=scale))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        gp = ReplayGP(x.detach().double(), y.detach().double(), se.detach().double(), scale).double()
    gp.train()
    gp.likelihood.train()
    opt = torch.optim.Adam(gp.parameters(), lr=.05)
    if (folder / 'latest.pt').exists():
        state = load(folder / 'latest.pt')
        assert state['identity'] == ident
        gp.load_state_dict(state['model'])
        opt.load_state_dict(copy.deepcopy(state['optimizer']))
        restore_rng(state['rng'])
        if 0 < state['step'] < 20:
            append(folder / 'recovery.jsonl', dict(step=state['step'], model=tensor_hash(gp.state_dict()),
                optimizer_loaded=True, rng_loaded=True, forward_finite=bool(torch.isfinite(gp(x).mean).all())))
    else:
        attempt = ledger.intent('gp_fits', metadata['task'], 0)
        state = dict(identity=ident, attempt=attempt, step=0, history=[], committed=[], seconds=0.,
            model=copy.deepcopy(gp.state_dict()), optimizer=copy.deepcopy(opt.state_dict()), rng=rng_state(),
            x=x.detach().clone(), target=y.detach().clone(), se=se.detach().clone(), scale=scale, metadata=metadata)
        save(folder / 'latest.pt', state)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    for step in range(state['step'] + 1, 21):
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        with warnings.catch_warnings(record=True) as caught, gpytorch.settings.cholesky_jitter(double_value=1e-6), gpytorch.settings.cholesky_max_tries(3), gpytorch.settings.min_fixed_noise(double_value=0., float_value=0., half_value=0.):
            warnings.simplefilter('always')
            loss = -mll(gp(gp.train_inputs[0]), gp.train_targets)
            assert torch.isfinite(loss)
            loss.backward()
        grad = torch.cat([p.grad.reshape(-1) for p in gp.parameters()])
        assert torch.isfinite(grad).all()
        gid = ledger.update('gp', metadata['task'], step, opt)
        messages = [str(w.message) for w in caught]
        jitter = max([0.] + [float(v) for m in messages for v in re.findall(r'jitter of ([\deE.+-]+)', m)])
        state['history'].append(dict(step=step, id=gid, loss=float(loss.detach()), gradient_norm=float(grad.norm()), actual_jitter=jitter, warnings=messages))
        state['committed'].append(gid)
        state.update(step=step, model=copy.deepcopy(gp.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
            rng=rng_state(), seconds=state['seconds'] + time.monotonic() - start)
        save(folder / 'latest.pt', state)
        if step in [1, 20]:
            save(folder / f'step{step:02}.pt', state)
        pause_once(folder, 'gp', recovery_test and step == 1, dict(step=step, kind='gp'))
    if (folder / 'teacher.pt').exists():
        teacher = load(folder / 'teacher.pt')
        assert teacher['metadata'] == metadata
    else:
        teacher = from_gp(gp, x, y, se, scale, metadata)
        save(folder / 'teacher.pt', teacher)
    return teacher


def fit_mapping(folder, model, data, steps, enabled, ledger, metadata, recovery_test=False):
    folder = Path(folder)
    data_hash = tensor_hash({k: data[k] for k in ['z', 'hp', 'target64', 'se']})
    ident = digest(dict(metadata=metadata, data_hash=data_hash, steps=steps, enabled=enabled, scaler=data['scaler']))
    opt = fresh_adam(model)
    groups = architecture_groups(data['ids'], data['architectures']) if enabled else None
    if (folder / 'latest.pt').exists():
        state = load(folder / 'latest.pt')
        assert state['identity'] == ident
        model.load_state_dict(state['model'])
        opt.load_state_dict(copy.deepcopy(state['optimizer']))
        restore_rng(state['rng'])
        if 0 < state['step'] < steps:
            with torch.no_grad():
                e, h = model(data['z'], data['hp'])
            append(folder / 'recovery.jsonl', dict(step=state['step'], model=tensor_hash(model.state_dict()),
                optimizer_steps=sorted({float(s['step']) for s in opt.state.values()}), forward=tensor_hash({'e': e, 'h': h}), rng_loaded=True))
    else:
        state = dict(identity=ident, step=0, history=[], committed=[], seconds=0., solve_records={},
            model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()), rng=rng_state(),
            initial_model_hash=tensor_hash(model.state_dict()), train_ids=data['ids'], groups=groups, scaler=data['scaler'], metadata=metadata, probes=[])
        save(folder / 'latest.pt', state)
        save(folder / 'step000.pt', state)
    assert state['groups'] == groups
    meter = SolveMeter(state['solve_records'])
    teachers, active_block = None, None
    while state['step'] < steps:
        step = state['step']
        block = step // 50
        if enabled and block != active_block:
            snap_path = folder / 'blocks' / f'b{block}.pt'
            if snap_path.exists():
                snap = load(snap_path)
            else:
                assert step % 50 == 0
                with torch.no_grad():
                    emb, _ = model(data['z'], data['hp'])
                snap = dict(e=emb.detach().double(), model=tensor_hash(model.state_dict()), step=step)
                save(snap_path, snap)
            teachers = []
            for group in groups:
                si = group['support_indices']
                name = f'b{block}-g{group["group"]}'
                meta = dict(task=metadata['task'] + '/' + name, train_ids=group['support_ids'], query_ids=group['query_ids'],
                    model_version=metadata['version'], snapshot=snap['model'], mapping_step=block * 50)
                teachers.append(fit_gp(folder / 'gps' / name, snap['e'][si], data['target64'][si], data['se'][si], data['scaler']['y_scale'], meta, ledger))
            active_block = block
            if step == 0 and not state['probes']:
                state['probes'].append(gradient_probe(model, data, data['target32'], data['target64'], data['se'], data['scaler']['y_scale'], groups, teachers, meter))
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        e, p = model(data['z'], data['hp'])
        losses = objective(e, p, data['target32'], data['target64'], data['se'], data['scaler']['y_scale'], groups, teachers, enabled, meter)
        assert all(torch.isfinite(v) for v in losses.values())
        losses['total'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        gid = ledger.update('mapping', metadata['task'], step + 1, opt)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        state['history'].append(dict(step=step + 1, id=gid, **{k: float(v.detach()) for k, v in losses.items()}))
        state['committed'].append(gid)
        state.update(step=step + 1, model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
            rng=rng_state(), seconds=state['seconds'] + time.monotonic() - start)
        save(folder / 'latest.pt', state)
        if state['step'] in [1, 50, 100, 150, 200]:
            save(folder / f'step{state["step"]:03}.pt', state)
        pause_once(folder, 'mapping', recovery_test and state['step'] == 1, dict(step=1, kind='mapping'))
    assert tensor_hash({k: data[k] for k in ['z', 'hp', 'target64', 'se']}) == data_hash
    return state


def score_candidates(model, data, teacher=None):
    start = time.monotonic()
    with torch.no_grad():
        e, h = model(data['all_z'], data['all_hp'])
        scale, center = data['scaler']['y_scale'], data['scaler']['y_mean']
        head = h.double() * scale + center
        if teacher is None:
            scores, lower, upper = head, None, None
        else:
            assert torch.equal(teacher['support'], e[data['indices']].double())
            mu, variance, _ = posterior(teacher, e.double())
            scores = mu * scale + center
            sd = (variance * scale ** 2 + teacher['parameters']['noise'] * scale ** 2 + data['se'].square().mean()).sqrt()
            lower, upper = scores - 1.96 * sd, scores + 1.96 * sd
        known = set(data['ids'])
        rows = [dict(candidate_id=cid, score=float(scores[i]), head=float(head[i]),
            lower=None if lower is None else float(lower[i]), upper=None if upper is None else float(upper[i]))
            for i, cid in enumerate(data['all_ids']) if cid not in known]
    assert all(np.isfinite(r['score']) for r in rows)
    rows.sort(key=lambda r: (-r['score'], r['candidate_id']))
    return rows, dict(embedding=e, head=h, ids=data['all_ids'], model_hash=tensor_hash(model.state_dict()),
        embedding_hash=tensor_hash({'e': e}), train_ids=data['ids']), time.monotonic() - start
