"""Fivefold preflight. Every test forbids Adam.step; no training smoke runs."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from analyse import relative_metric_fivefold as f
from surrogate.fuzzy_metric import JointMetricModel
from surrogate.relative_metric import build_plan,tensor_plan,order_terms,probe
spec=importlib.util.spec_from_file_location('prior_tests',f.ROOT/'tests/test_relative_metric_pilot.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
# Reuse independent engineering checks, preserving the original pilot entry.
test_scalar_ties_empty=prior.test_manual_formula_realizability_ties_zeroSE_and_small_groups
test_order_gradient=prior.test_nonzero_order_gradient_direction_and_weight_scaling
test_R0_gradient=prior.test_R0_matches_old_regression_loss_and_every_gradient
test_branch_adam=prior.test_branch_optimizer_state_and_serialization
test_label_boundary=prior.test_boundary_with_disabled_reader_and_deidentified_sentinels
test_rng_probe=prior.test_RNG_isolation_noise_orientation_and_probe_no_side_effects
test_self_exclusion=prior.test_self_exclusion_and_1d_is_model_prediction
test_atomic_nodes=prior.test_atomic_node_rejects_partial_existing_checkpoint
OUT=Path(os.environ.get('R1_FIVEFOLD_OUTPUT',f.DEFAULT))
@pytest.fixture(autouse=True)
def zero_updates(monkeypatch):
    def reject(*a,**k):raise AssertionError('preflight attempted optimizer update')
    monkeypatch.setattr(torch.optim.Adam,'step',reject);torch.set_num_threads(1)


def test_all15_starts_variable_sizes_full_plan_and_scalar():
    p,tasks=f.verify_protocol(OUT);checks=f.read(OUT/'warmstart_checks.json');assert len(checks)==15 and len(tasks)==30 and not p['missing']
    assert {len(t['train_ids']) for t in tasks}=={141,143,144,145}
    for t in tasks[::2]:
        d=f.load(OUT/t['input']);y,se,target=f.assert_task_data(t,d);n=len(y)
        plan=dict(np.load(OUT/t['plan']));assert plan['refs'].shape==(200,n,2,8,2)
        rebuilt=build_plan(y,se,d['train']['ids'],d['scaler']['y_mean'],d['scaler']['y_scale'],t['seed'],steps=1)
        for name in ['refs','margins','weights','counts']:assert np.array_equal(rebuilt[name][0],plan[name][0])
        i=np.arange(n)[None,:,None,None];j,k=plan['refs'][...,0],plan['refs'][...,1];v=j>=0
        gap=np.abs(y[i]-y[k])-np.abs(y[i]-y[j]);assert (gap[v]>1e-12).all() and ((j!=i)&(k!=i)&(j!=k))[v].all()
        yt=(y-d['scaler']['y_mean'])/d['scaler']['y_scale'];margin=np.minimum((yt[i]-yt[k])**2-(yt[i]-yt[j])**2,1.)
        np.testing.assert_allclose(plan['margins'][v],margin[v],atol=1e-14,rtol=0)
        noise=2*se[i]+se[j]+se[k]
        np.testing.assert_allclose(plan['weights'][v],(gap**2/(gap**2+noise**2))[v],atol=1e-14,rtol=0)
        assert np.array_equal(plan['counts'],v.sum(-1))
        # Every group contains distinct reference pairs; local refs belong to exact local10.
        for center in range(n):
            local=set(sorted([a for a in range(n) if a!=center],key=lambda a:(abs(y[center]-y[a]),d['train']['ids'][a]))[:10])
            assert set(plan['refs'][:,center,1].reshape(-1))-{-1}<=local
            for step in range(200):
                for g in range(2):
                    pairs=[tuple(x) for x in plan['refs'][step,center,g,:plan['counts'][step,center,g]]]
                    assert len(pairs)==len(set(pairs))
        m=JointMetricModel(12,seed=t['seed']);m.load_state_dict(f.load(OUT/t['warmstart'])['model'])
        e,_=m(d['train']['z'],d['train']['hp']);sample=tensor_plan(plan,0)
        loss=order_terms(e,sample)[0];scalar=prior.scalar_loss(e.detach().double().tolist(),plan,0)
        assert abs(float(loss.detach())-scalar)<2e-6
        diagnostics=f.probe_groups(m,d['train'],target,sample)
        assert abs(sum(g['half_weighted_loss'] for g in diagnostics['groups'].values())-float(loss.detach()))<2e-6
        assert diagnostics['loss_fraction_is_not_gradient_fraction']
    for r in checks:
        assert r['restored_train_embedding_maxdiff']==r['old_neighbor_distance_maxdiff']==0
        assert r['old_head_maxdiff']<=1e-12 and not r['old_head_missing_ids']
        assert r['target_source_maxdiff']<=1e-12 and r['SE2_source_maxdiff']<=1e-12
        assert r['original_Adam_steps']==100 and r['effective_objects']['optimizer_initial_state_entries']==0


def test_old112_saved_core_output_unchanged():
    for seed in range(3):
        d=f.load(f.PILOT/'inputs'/f's{seed}.pt');assert len(d['train']['ids'])==112
        plan=dict(np.load(f.PILOT/'plans'/f's{seed}.npz'));sample=tensor_plan(plan,0)
        for branch in ['R0','R1']:
            folder=f.PILOT/'trajectories'/f's{seed}-{branch}';state=f.load(folder/'add000.pt')
            m=JointMetricModel(12,seed=seed);m.load_state_dict(state['model'])
            now=f.probe_groups(m,d['train'],d['target'],sample);old=f.read(folder/'probes.json')['0']
            for key,value in old.items():assert now[key]==value


def test_fold_worker_runtime_file_boundary_and_wrong_fold():
    p,tasks=f.verify_protocol(OUT);t=tasks[0];d=f.load(OUT/t['input']);wrong=copy.deepcopy(t);wrong['fold']=1
    with pytest.raises(ValueError,match='cross-fold'):f.assert_task_data(wrong,d)
    script='''from analyse import relative_metric_fivefold as f
p,t=f.verify_protocol(f.DEFAULT,0)
f.guard_fold_files(f.DEFAULT,0,t)
f.load(f.DEFAULT/t[0]['input'])
import os
directory=f.DEFAULT/'trajectories'/t[0]['task_id'];directory.mkdir(parents=True,exist_ok=True)
fd=os.open(directory,os.O_RDONLY);os.fsync(fd);os.close(fd)
for name in ['evaluation_labels/fold0.json','inputs/fold1/s0.pt','input_originals/fuzzy-metric-stage2-20260910-v1_A_validation.json']:
    try: (f.DEFAULT/name).open('rb')
    except PermissionError: pass
    else: raise AssertionError('forbidden file opened')
print('own-fold allowed; 3 forbidden paths blocked before reading')
'''
    child=subprocess.run([sys.executable,'-c',script],cwd=f.ROOT,capture_output=True,text=True)
    assert child.returncode==0,child.stderr


def test_cost_commits_discarded_and_uncertain_metadata_only(tmp_path):
    for n in range(1,4):f.write(tmp_path/'intents'/f'{n:06d}.json',{'global_id':n})
    for n in range(1,3):f.write(tmp_path/'receipts'/f'{n:06d}.json',{'global_id':n})
    f.save(tmp_path/'trajectories/task/latest.pt',dict(step=1,plan_position=1,history=[{}],committed_global_ids=[1]))
    c=f.costs(tmp_path);assert c['effective_updates']==1 and c['confirmed_actual_updates']==2
    assert c['discarded_or_repeated_updates']==1 and c['uncertain_global_ids']==[3] and c['charged_upper_bound']==3
    assert f.costs(tmp_path)==c


def test_cli_conflicting_configuration_before_any_updates(tmp_path):
    bad=copy.deepcopy(f.CONFIG);bad['additional_steps']=201
    path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
    child=subprocess.run([sys.executable,str(Path(f.__file__)),'--config',str(path),'--output',str(tmp_path/'run'),'--run'],capture_output=True,text=True)
    assert child.returncode!=0 and 'configuration conflicts' in child.stderr
    assert not (tmp_path/'run').exists()
