"""R1 fixed relative-order objective; no file access, GP, VAE or GNN objects."""
from __future__ import annotations
import copy
import itertools
import math
import numpy as np
import torch

ADAM = dict(lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False,
            foreach=False, fused=False)
FEATURE_KEYS = {'ids', 'architectures', 'z', 'hp'}
LABEL_KEYS = {'training_seeds', 'validation_seeds', 'validation_mean', 'validation_se'}


def validate_batch(train, query, labels, scaler):
    """Strict boundary: only explicitly unlabelled feature fields and train labels."""
    for part in (train, query):
        if set(part) != FEATURE_KEYS:
            raise ValueError('feature payload contains unexpected fields / label contamination')
        n = len(part['ids'])
        if len(set(part['ids'])) != n or len(part['architectures']) != n:
            raise ValueError('duplicate or mismatched feature identity')
        for name, dim in [('z', 12), ('hp', 4)]:
            t = part[name]
            if not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or t.shape != (n, dim):
                raise ValueError('feature shape/dtype differs')
            if t.requires_grad or not torch.isfinite(t).all():
                raise ValueError('cached features must be finite and detached')
    ids = train['ids']
    if set(ids) & set(query['ids']) or set(train['architectures']) & set(query['architectures']):
        raise ValueError('train/query partition overlap')
    if set(labels) != set(ids) or scaler['fit_ids'] != ids:
        raise ValueError('training label whitelist or scaler fit ID order differs')
    y, se = [], []
    for cid in ids:
        row = labels[cid]
        if set(row) != LABEL_KEYS or row['training_seeds'] != [0, 1, 2]:
            raise ValueError('label fields or training seeds differ')
        values = np.asarray(row['validation_seeds'], dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError('invalid validation measurements')
        mean, error = float(values.mean()), float(values.std(ddof=1)/math.sqrt(3))
        if abs(mean-row['validation_mean']) > 1e-15 or abs(error-row['validation_se']) > 1e-15:
            raise ValueError('validation mean / SE mismatch')
        y.append(row['validation_mean']); se.append(row['validation_se'])
    y, se = np.asarray(y, dtype=np.float64), np.asarray(se, dtype=np.float64)
    target64 = (y-scaler['y_mean'])/scaler['y_scale']
    target = torch.tensor(target64, dtype=torch.float32)
    if not np.isfinite(target64).all() or scaler['y_scale'] <= 0:
        raise ValueError('invalid saved target scale')
    return y, se, target


def legal_pairs(y, se, ids, i, references, ymean, yscale):
    """All unordered reference pairs, oriented by raw performance gaps."""
    pairs = np.asarray(list(itertools.combinations(references, 2)), dtype=np.int32).reshape(-1, 2)
    if not len(pairs):
        return pairs, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    delta = np.abs(y-y[i])
    pairs = pairs[np.abs(delta[pairs[:, 0]]-delta[pairs[:, 1]]) > 1e-12]
    flip = delta[pairs[:, 0]] > delta[pairs[:, 1]]
    pairs[flip] = pairs[flip, ::-1]
    j, k = pairs.T
    g = delta[k]-delta[j]
    yt = (y-ymean)/yscale
    margin = np.minimum(np.abs(yt[k]-yt[i])**2 - np.abs(yt[j]-yt[i])**2, 1.)
    u = 2*se[i]+se[j]+se[k]
    weights = g*g/(g*g+u*u)
    if not np.isfinite(margin).all() or not (margin > 0).all() or not np.isfinite(weights).all():
        raise ValueError('nonpositive/nonfinite legal margin or weight')
    return pairs, margin, weights


def build_plan(y, se, ids, ymean, yscale, seed, steps=200):
    """Use isolated PCG64; group padding is excluded without renormalizing centers."""
    y, se = np.asarray(y, dtype=np.float64), np.asarray(se, dtype=np.float64)
    if len(set(ids)) != len(ids) or y.shape != se.shape or y.shape != (len(ids),):
        raise ValueError('plan IDs/labels differ')
    if not np.isfinite(y).all() or not np.isfinite(se).all() or (se < 0).any():
        raise ValueError('invalid plan labels')
    n = len(ids); pools = []
    for i in range(n):
        refs = [j for j in range(n) if j != i]
        local = sorted(refs, key=lambda j: (abs(y[i]-y[j]), ids[j]))[:10]
        pools.append([legal_pairs(y,se,ids,i,r,ymean,yscale) for r in (refs,local)])
    rng = np.random.Generator(np.random.PCG64(20260911+seed))
    refs = np.full((steps,n,2,8,2), -1, dtype=np.int16)
    margins = np.zeros((steps,n,2,8), dtype=np.float64)
    weights = np.zeros_like(margins)
    counts = np.zeros((steps,n,2), dtype=np.int16)
    for s in range(steps):
        for i in range(n):
            for group, (pairs, m, w) in enumerate(pools[i]):
                count = min(8, len(pairs)); counts[s,i,group] = count
                selected = rng.choice(len(pairs), count, replace=False) if len(pairs) > 8 else np.arange(count)
                refs[s,i,group,:count] = pairs[selected]
                margins[s,i,group,:count] = m[selected]
                weights[s,i,group,:count] = w[selected]
    return dict(refs=refs, margins=margins, weights=weights, counts=counts, ids=np.asarray(ids),
                rng_state=np.asarray(str(rng.bit_generator.state)))


def tensor_plan(plan, step, dtype=torch.float32):
    refs = torch.tensor(plan['refs'][step], dtype=torch.long)
    margins = torch.tensor(plan['margins'][step], dtype=dtype)
    weights = torch.tensor(plan['weights'][step], dtype=dtype)
    valid = refs[..., 0] >= 0
    if not torch.isfinite(margins).all() or not (margins[valid] > 0).all() or not torch.isfinite(weights).all():
        raise ValueError('positive margin lost in training dtype conversion')
    return refs, margins, weights, torch.tensor(plan['counts'][step], dtype=dtype)


def order_terms(embedding, sample):
    refs, margins, weights, counts = sample
    n = len(embedding)
    if refs.shape != (n,2,8,2) or margins.shape != (n,2,8):
        raise ValueError('sample and embedding dimensions differ')
    valid = refs[...,0] >= 0
    if not torch.equal(valid.sum(-1).to(counts), counts):
        raise ValueError('sample group counts differ')
    safe = refs.clamp_min(0)
    center = embedding[:,None,None,:]
    dj = (center-embedding[safe[...,0]]).square().sum(-1)
    dk = (center-embedding[safe[...,1]]).square().sum(-1)
    hinge = (margins+dj-dk).clamp_min(0)
    group_means = (weights*hinge*valid).sum(-1)/counts.clamp_min(1)
    return group_means.mean(), hinge, valid


def objective(embedding, prediction, target, sample, lambda_r):
    if lambda_r not in (0., .01):
        raise ValueError('R1 protocol coefficient conflict')
    regression = (prediction-target.detach()).square().mean()
    size = embedding.square().sum(-1).mean()
    order = order_terms(embedding, sample)[0] if lambda_r else embedding.sum()*0.
    return dict(total=regression+lambda_r*order+.001*size,
                regression=regression, order=order, size=size)


def make_branches(warm_model):
    a, b = copy.deepcopy(warm_model), copy.deepcopy(warm_model)
    for original, pa, pb in zip(warm_model.parameters(), a.parameters(), b.parameters()):
        if not torch.equal(pa,pb) or len({original.data_ptr(),pa.data_ptr(),pb.data_ptr()}) != 3:
            raise ValueError('branch initialization/storage differs')
    return a, b


def fresh_adam(model):
    optimizer = torch.optim.Adam(model.parameters(), **ADAM)
    expected = {id(p) for name,p in model.named_parameters()
                if name.split('.')[0] in ('hp_encoder','mapping','regressor') and p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(actual) != expected or len(actual) != len(expected) or optimizer.state:
        raise ValueError('optimizer parameter set or initial momentum differs')
    return optimizer


def probe(model, train, target, sample):
    """autograd.grad only; no optimizer, accumulated gradients or RNG mutation."""
    rng = torch.get_rng_state().clone()
    existing = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    e,p = model(train['z'],train['hp'])
    reg = (p-target).square().mean(); order,hinge,valid = order_terms(e,sample)
    shared = list(model.hp_encoder.parameters())+list(model.mapping.parameters())
    head = list(model.regressor.parameters())
    gr = torch.cat([g.reshape(-1) for g in torch.autograd.grad(reg, shared, retain_graph=True)])
    go = torch.cat([g.reshape(-1) for g in torch.autograd.grad(order, shared, retain_graph=True)])
    gh = torch.cat([g.reshape(-1) for g in torch.autograd.grad(reg, head)])
    nr, no = float(gr.norm()), float(go.norm())
    count = int(valid.sum()); active = int(((hinge>0)&valid).sum())
    reason = None if no else ('no legal triplets' if count == 0 else 'constraints satisfied' if active == 0 else 'disconnected or zero local derivative; inspect')
    if not torch.equal(rng,torch.get_rng_state()):
        raise RuntimeError('probe altered RNG')
    for before, par in zip(existing,model.parameters()):
        if (before is None) != (par.grad is None) or (before is not None and not torch.equal(before,par.grad)):
            raise RuntimeError('probe altered accumulated gradients')
    w = sample[2][valid].detach().double().numpy()
    result = dict(order_loss=float(order.detach()), regression_loss=float(reg.detach()),
        size_loss=float(e.detach().square().sum(-1).mean()), embedding_norm_mean=float(e.detach().norm(dim=-1).mean()),
        embedding_norm_min=float(e.detach().norm(dim=-1).min()),embedding_norm_max=float(e.detach().norm(dim=-1).max()),
        active_fraction=active/count if count else None,triplets=count,empty_groups=int((sample[3]==0).sum()),
        w_quantiles=np.quantile(w,[0,.25,.5,.75,1]).tolist() if count else [],
        regression_shared_gradient_norm=nr,order_shared_gradient_norm=no,head_regression_gradient_norm=float(gh.norm()),
        weighted_order_to_regression_ratio=.01*no/nr if nr else None,
        gradient_cosine=float(torch.dot(gr,go)/(gr.norm()*go.norm())) if nr and no else None,
        NA_reason=reason if nr else 'zero regression gradient denominator',order_gradient_reason=reason)
    if not np.isfinite([nr,no,float(gh.norm()),result['order_loss']]).all():
        raise RuntimeError('nonfinite probe')
    return result
