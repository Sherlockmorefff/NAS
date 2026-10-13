"""Focused replay boundary checks. No optimizer updates or GNN data loads."""
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import torch
from analyse.gp_guidance_development_replay import id_order, SEEDS, guard_worker
from surrogate.gp_guidance_replay_engine import scaler_from_initial, input_tensors, score_candidates, reload_model
from surrogate.fuzzy_metric import JointMetricModel
from surrogate.gp_mean_guidance import architecture_groups, matern52
from surrogate.gp_mean_guidance_sync import synchronized_mean


def data():
    g = torch.Generator().manual_seed(17)
    ids = [f'id{i:03}' for i in range(179)]
    features = dict(ids=ids, architectures=[f'arch{i//2:03}' for i in range(179)],
        z=torch.randn(179,12,generator=g).tolist(), hp=torch.randn(179,4,generator=g).tolist())
    labels = {c:dict(validation_seeds=[.5+i*.001]*3, validation_mean=.5+i*.001,
        validation_se=0., training_seeds=[0,1,2]) for i,c in enumerate(ids)}
    return features, labels


def test_initial_and_random_orders():
    f, _ = data()
    for seed in SEEDS:
        initial = id_order(f['ids'], seed, 'init')[:50]
        expected = sorted(f['ids'], key=lambda c:(hashlib.sha256(f'gp_guidance_replay_v1:init:{seed}:{c}'.encode()).hexdigest(),c))[:50]
        assert initial == expected and len(set(initial)) == 50
        remaining = [c for c in f['ids'] if c not in initial]
        order = id_order(remaining, seed, 'random')
        assert len(order) == len(set(order)) == 129 and not set(order) & set(initial)


def test_unrevealed_perturbation_before_and_after_reveal():
    f, oracle = data()
    initial = id_order(f['ids'], SEEDS[0], 'init')[:50]
    known = {c:oracle[c] for c in initial}
    sc = scaler_from_initial(f, initial, known)
    model = JointMetricModel(12, seed=0)
    before = copy.deepcopy(model.state_dict())
    rows, cache, _ = score_candidates(model, input_tensors(f, sc, known))
    cid = rows[0]['candidate_id']
    changed = copy.deepcopy(oracle)
    changed[cid].update(validation_mean=.1, validation_seeds=[.1]*3)
    known2 = {c:changed[c] for c in initial}
    rows2, cache2, _ = score_candidates(model, input_tensors(f, sc, known2))
    assert known == known2 and rows == rows2
    assert torch.equal(cache['embedding'], cache2['embedding'])
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
    original_revealed = {**known, cid:oracle[cid]}
    changed_revealed = {**known, cid:changed[cid]}
    a, b = input_tensors(f, sc, original_revealed), input_tensors(f, sc, changed_revealed)
    assert not torch.equal(a['target64'], b['target64'])
    def grad(d):
        m = reload_model(before)
        e,p = m(d['z'],d['hp'])
        loss=(p-d['target32']).square().mean()+.001*e.square().sum(-1).mean()
        return torch.cat([v.flatten() for v in torch.autograd.grad(loss, m.parameters())])
    assert not torch.equal(grad(a), grad(b))
    assert sc['fit_ids'] == sorted(initial)


def test_model_clones_independent():
    m = JointMetricModel(12,seed=0)
    a,b = reload_model(m.state_dict()),reload_model(m.state_dict())
    for p,q in zip(a.parameters(),b.parameters()):
        assert torch.equal(p,q) and p.data_ptr()!=q.data_ptr()
    with torch.no_grad():
        next(a.parameters()).add_(1)
    assert not torch.equal(next(a.parameters()),next(b.parameters()))


def test_synchronized_formula_and_both_gradients():
    torch.manual_seed(9)
    s=torch.randn(7,8,dtype=torch.double,requires_grad=True)
    q=torch.randn(3,8,dtype=torch.double,requires_grad=True)
    y=torch.randn(7,dtype=torch.double);se=torch.full((7,),.01,dtype=torch.double)
    pars=dict(lengthscale=2.,outputscale=.7,noise=.01,mean=.2)
    actual=synchronized_mean(s,q,y,se,.2,pars)
    cov=matern52(s,s,2.,.7)+torch.diag((se/.2)**2+.01)
    expected=.2+matern52(q,s,2.,.7)@torch.linalg.solve(cov,y-.2)
    assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    gs,gq=torch.autograd.grad(actual.square().mean(),[s,q])
    assert torch.isfinite(gs).all() and torch.isfinite(gq).all() and gs.norm()>0 and gq.norm()>0


def test_architecture_groups_no_leak():
    ids=[str(i) for i in range(20)];arches=[str(i//2) for i in range(20)]
    groups=architecture_groups(ids,arches)
    assert sorted(c for g in groups for c in g['query_ids'])==sorted(ids)
    for g in groups:
        assert not set(g['support_architectures'])&set(g['query_architectures'])


def test_file_guard_in_separate_process():
    code='''
from pathlib import Path
from analyse.gp_guidance_development_replay import ROOT,DEFAULT,guard_worker
o=DEFAULT; folder=o/'trajectories'/'2026091600-C_GP'
guard_worker(o,folder,2026091600,'C_GP')
targets=[ROOT/'results/posthoc/fuzzy-metric-stage2-20260910-v1/A_validation.json',o/'trajectories/2026091600-S_GP/state.pt',ROOT/'checkpoints/joint_model_pipeline_global4_best.pth']
for target in targets:
    try: target.read_bytes()
    except PermissionError: continue
    raise AssertionError(str(target))
print('3 forbidden paths rejected')
'''
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert r.returncode==0, r.stderr
    assert '3 forbidden paths rejected' in r.stdout
