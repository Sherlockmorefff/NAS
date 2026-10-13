"""Zero optimizer updates: scalar, gradient, identity and recovery checks."""
import builtins
import copy
import io
import math
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from surrogate.fuzzy_metric import JointMetricModel, metric_loss
from surrogate.relative_metric import (build_plan, legal_pairs, tensor_plan, order_terms, objective,
    validate_batch, make_branches, fresh_adam, probe)
from analyse.relative_metric_pilot import (DEFAULT,CONFIG,audit,read,load,save,write,verify_protocol,
    actual_objects,cost_status,tensor_hash,infer_neighbors,freeze_copy)

@pytest.fixture(autouse=True)
def no_steps(monkeypatch):
    def reject(*a,**kw): raise AssertionError('tests must perform zero optimizer updates')
    monkeypatch.setattr(torch.optim.Adam,'step',reject)
    torch.set_num_threads(1)


def scalar_loss(e,plan,s):
    # Independent scalar implementation: no tested loss/geometry calls.
    n=len(e);total=0.
    for i in range(n):
        for g in range(2):
            values=[]
            for t in range(8):
                j,k=map(int,plan['refs'][s,i,g,t])
                if j<0: continue
                dij=sum((float(a)-float(b))**2 for a,b in zip(e[i],e[j]))
                dik=sum((float(a)-float(b))**2 for a,b in zip(e[i],e[k]))
                values.append(float(plan['weights'][s,i,g,t])*max(0.,float(plan['margins'][s,i,g,t])+dij-dik))
            total += sum(values)/len(values)/2/n if values else 0.
    return total


def test_manual_formula_realizability_ties_zeroSE_and_small_groups():
    ids=['d','a','c','b'];y=np.array([0.,.2,.2,.7]);se=np.array([0.,.01,.02,.03])
    plan=build_plan(y,se,ids,0.,.5,0,steps=2)
    for i in range(4):
        for g in range(2):
            seen=set()
            for t in range(int(plan['counts'][0,i,g])):
                j,k=map(int,plan['refs'][0,i,g,t]);assert len({i,j,k})==3
                assert (j,k) not in seen;seen.add((j,k))
                a,b=abs(y[i]-y[j]),abs(y[i]-y[k]);gap=b-a
                assert gap>1e-12
                assert math.isclose(plan['margins'][0,i,g,t],min((b/.5)**2-(a/.5)**2,1.),abs_tol=1e-14)
                u=2*se[i]+se[j]+se[k]
                assert math.isclose(plan['weights'][0,i,g,t],gap**2/(gap**2+u**2),abs_tol=1e-14)
    e=torch.tensor([[0.,1.],[2.,0.],[1.,0.],[.3,.4]],dtype=torch.float64,requires_grad=True)
    value=order_terms(e,tensor_plan(plan,0,torch.float64))[0]
    assert math.isclose(float(value.detach()),scalar_loss(e.detach().tolist(),plan,0),abs_tol=1e-14)
    realized=torch.tensor((y/.5)[:,None],dtype=torch.float64)
    assert float(order_terms(realized,tensor_plan(plan,0,torch.float64))[0])<=1e-14
    p0=build_plan(y,np.zeros(4),ids,0.,.5,0,steps=1)
    assert np.all(p0['weights'][p0['refs'][...,0]>=0]==1)
    assert (plan['counts']<8).all()
    empty=build_plan(np.zeros(4),np.zeros(4),ids,0.,1.,0,steps=1)
    assert not empty['counts'].any()
    assert float(order_terms(e,tensor_plan(empty,0,torch.float64))[0])==0.


def test_nonzero_order_gradient_direction_and_weight_scaling():
    plan={'refs':np.full((1,3,2,8,2),-1,dtype=int),'margins':np.zeros((1,3,2,8)),
          'weights':np.zeros((1,3,2,8)),'counts':np.zeros((1,3,2))}
    plan['refs'][0,0,0,0]=[1,2];plan['margins'][0,0,0,0]=.3;plan['weights'][0,0,0,0]=.8;plan['counts'][0,0,0]=1
    e=torch.tensor([[0.],[2.],[1.]],dtype=torch.float64,requires_grad=True)
    value=order_terms(e,tensor_plan(plan,0,torch.float64))[0]
    assert math.isclose(float(value.detach()),.8*(.3+4-1)/6,abs_tol=1e-14)
    grad,=torch.autograd.grad(value,e)
    assert torch.allclose(grad,torch.tensor([[-2.],[4.],[-2.]],dtype=torch.float64)*.8/6,atol=1e-14)
    assert scalar_loss((e.detach()-1e-5*grad).tolist(),plan,0)<float(value.detach())
    lower=copy.deepcopy(plan);lower['weights']*=.1
    assert math.isclose(float(order_terms(e,tensor_plan(lower,0,torch.float64))[0]),float(value.detach())*.1,abs_tol=1e-14)


def test_R0_matches_old_regression_loss_and_every_gradient():
    model=JointMetricModel(12,seed=8)
    z=torch.arange(60,dtype=torch.float32).reshape(5,12)/30;hp=torch.ones(5,4);target=torch.linspace(-1,1,5)
    e,p=model(z,hp);zeros=torch.zeros(5,5)
    old=metric_loss(e,p,target,zeros,lambda_f=0.,lambda_e=.001,pair_weights=zeros)['total']
    newer=objective(e,p,target,None,0.)['total']
    go=torch.autograd.grad(old,list(model.parameters()),retain_graph=True)
    gn=torch.autograd.grad(newer,list(model.parameters()))
    assert torch.equal(old,newer)
    assert all(torch.equal(a,b) for a,b in zip(go,gn))
    with pytest.raises(ValueError,match='coefficient'): objective(e,p,target,None,.1)


def test_branch_optimizer_state_and_serialization():
    warm=JointMetricModel(12,seed=2);a,b=make_branches(warm)
    opt=fresh_adam(a);objects=actual_objects(a,opt)
    assert objects['optimizer_initial_state_entries']==0
    assert objects['param_groups'][0]['fused'] is False and objects['param_groups'][0]['foreach'] is False
    buf=io.BytesIO();torch.save({'model':a.state_dict(),'optimizer':opt.state_dict()},buf);buf.seek(0)
    state=torch.load(buf,weights_only=False);b.load_state_dict(state['model']);other=fresh_adam(b);other.load_state_dict(copy.deepcopy(state['optimizer']))
    assert not other.state and tensor_hash(a.state_dict())==tensor_hash(b.state_dict())
    assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters()))


def test_boundary_with_disabled_reader_and_deidentified_sentinels(monkeypatch):
    def features(ids):return {'ids':ids,'architectures':['arch'+x for x in ids],'z':torch.zeros(len(ids),12),'hp':torch.zeros(len(ids),4)}
    tr,qu=features(['a','b','c']),features(['heldout'])
    labels={i:dict(training_seeds=[0,1,2],validation_seeds=[.3,.3,.3],validation_mean=.3,validation_se=0.) for i in tr['ids']}
    sc={'fit_ids':tr['ids'],'y_mean':.3,'y_scale':1.}
    def blocked(*a,**k):raise AssertionError('core attempted file access')
    monkeypatch.setattr(builtins,'open',blocked);monkeypatch.setattr(Path,'open',blocked)
    y,se,t=validate_batch(tr,qu,labels,sc);plan=build_plan(y,se,tr['ids'],.3,1.,0,steps=1)
    contaminated=copy.deepcopy(qu);contaminated['holdout_y']=[999]
    with pytest.raises(ValueError):validate_batch(tr,contaminated,labels,sc)
    extra=copy.deepcopy(labels);extra['heldout']=extra['a']
    with pytest.raises(ValueError):validate_batch(tr,qu,extra,sc)
    nested=copy.deepcopy(labels);nested['a']['test']=999
    with pytest.raises(ValueError):validate_batch(tr,qu,nested,sc)
    wrong=copy.deepcopy(qu);wrong['architectures']=['archa']
    with pytest.raises(ValueError):validate_batch(tr,wrong,labels,sc)
    wrongids=copy.deepcopy(qu);wrongids['ids']=['a']
    with pytest.raises(ValueError):validate_batch(tr,wrongids,labels,sc)
    m=JointMetricModel(12);e,p=m(tr['z'],tr['hp']);torch.autograd.grad(objective(e,p,t,tensor_plan(plan,0),.01)['total'],list(m.parameters()))
    assert tr['z'].grad is None


def test_RNG_isolation_noise_orientation_and_probe_no_side_effects():
    ids=['a','b','c','d'];y=np.array([0.,.1,.35,.8]);se=np.array([0.,.01,.02,.04])
    torchstate=torch.get_rng_state().clone();npstate=np.random.get_state()
    a=build_plan(y,se,ids,0.,.3,1,steps=2);b=build_plan(y,se*3,ids,0.,.3,1,steps=2)
    assert np.array_equal(a['refs'],b['refs']);assert torch.equal(torchstate,torch.get_rng_state())
    assert np.array_equal(npstate[1],np.random.get_state()[1])
    m=JointMetricModel(12);tr={'z':torch.arange(48).reshape(4,12).float()/24,'hp':torch.ones(4,4)}
    for p in m.parameters():p.grad=torch.ones_like(p)
    before=[p.grad.clone() for p in m.parameters()];info=probe(m,tr,torch.tensor(y,dtype=torch.float32),tensor_plan(a,0))
    assert info['order_shared_gradient_norm']>0
    assert all(torch.equal(p.grad,g) for p,g in zip(m.parameters(),before))


def test_restore_ledger_commit_and_uncertain_intent_without_training(tmp_path):
    # Simulated metadata only, not real or synthetic optimizer updates.
    write(tmp_path/'intents/000001.json',{'task':'s0-R0'})
    state={'step':1,'history':[{'step':1}],'committed_global_ids':[1],'last_global_id':1}
    save(tmp_path/'trajectories/s0-R0/latest.pt',state)
    a=cost_status(tmp_path);assert a['effective_updates']==1 and a['uncertain_reserved_updates']==0
    write(tmp_path/'intents/000002.json',{'task':'s0-R0'})
    b=cost_status(tmp_path);assert b['charged_upper_bound']==2 and b['uncertain_global_ids']==[2]
    assert cost_status(tmp_path)==b
    state={'step':2,'history':[{'step':1},{'step':2}],'committed_global_ids':[1,2],'last_global_id':2}
    save(tmp_path/'trajectories/s0-R0/latest.pt',state)
    c=cost_status(tmp_path);assert c['confirmed_actual_updates']==2 and c['uncertain_reserved_updates']==0


def test_saved_warmstarts_plan_whitelist_and_config():
    output=Path(os.environ.get('R1_OUTPUT',DEFAULT));p=verify_protocol(output)
    assert len(p['train_ids'])==112 and len(p['query_ids'])==29
    checks=read(output/'warmstart_checks.json');assert len(checks)==3
    for s in [0,1,2]:
        d=load(output/'inputs'/f's{s}.pt');y,se,target=validate_batch(d['train'],d['query'],d['labels'],d['scaler'])
        plan=dict(np.load(output/'plans'/f's{s}.npz'))
        assert plan['refs'].shape==(200,112,2,8,2) and plan['ids'].tolist()==p['train_ids']
        # Independently verify raw direction, margins and noise weights for every sample.
        i=np.arange(112)[None,:,None,None];j=plan['refs'][...,0];k=plan['refs'][...,1];valid=j>=0
        dj=np.abs(y[i]-y[j]);dk=np.abs(y[i]-y[k]);gap=dk-dj
        assert (gap[valid]>1e-12).all()
        assert ((j!=i)&(k!=i)&(j!=k))[valid].all()
        yt=(y-d['scaler']['y_mean'])/d['scaler']['y_scale']
        expected=np.minimum((yt[i]-yt[k])**2-(yt[i]-yt[j])**2,1.)
        np.testing.assert_allclose(plan['margins'][valid],expected[valid],rtol=0,atol=1e-14)
        noise=2*se[i]+se[j]+se[k]
        np.testing.assert_allclose(plan['weights'][valid],(gap*gap/(gap*gap+noise*noise))[valid],rtol=0,atol=1e-14)
        assert checks[s]['old_optimizer_exists'] and checks[s]['head_maxdiff']<=1e-12
        warm=load(output/'warmstarts'/f's{s}-M100.pt');m=JointMetricModel(12,seed=s);m.load_state_dict(warm['model'])
        with torch.no_grad(): e,pred=m(d['train']['z'],d['train']['hp'])
        assert torch.equal(e,warm['train_embeddings'])
        for step in range(200): tensor_plan(plan,step)
    wrong=copy.deepcopy(CONFIG);wrong['lambda_r']['R1']=.1
    assert wrong!=CONFIG


def test_self_exclusion_and_1d_is_model_prediction():
    ids=['a','b','c','d','e','f'];bundle=dict(branch='R0',train_ids=ids,query_ids=['q'],
        train_embeddings=[[float(i),0.] for i in range(6)],query_embeddings=[[1.5,0.]],
        train_predictions=[.6,.5,.4,.3,.2,.1],query_predictions=[.32])
    result=infer_neighbors(bundle)
    assert result['R0-1D']['train_embeddings']==[[x] for x in bundle['train_predictions']]
    for v in result.values():
        for row in v['rows']['train']:
            assert row['candidate_id'] not in row['reference_ids'] and len(row['neighbor_ids'])==5


def test_atomic_node_rejects_partial_existing_checkpoint(tmp_path):
    source=tmp_path/'latest.pt';save(source,{'step':0,'model':{'weight':torch.ones(3)}})
    destination=tmp_path/'add000.pt';freeze_copy(source,destination)
    assert source.read_bytes()==destination.read_bytes()
    freeze_copy(source,destination)
    destination.write_bytes(b'partial-checkpoint')
    with pytest.raises(RuntimeError,match='identity'):freeze_copy(source,destination)


def test_cli_conflicting_config_rejected_before_run(tmp_path):
    import json,subprocess,sys
    from analyse.relative_metric_pilot import ROOT
    bad=copy.deepcopy(CONFIG);bad['additional_steps']=201
    config=tmp_path/'bad.json';config.write_text(json.dumps(bad))
    proc=subprocess.run([sys.executable,str(ROOT/'analyse/relative_metric_pilot.py'),'--config',str(config),'--output',str(tmp_path/'output'),'--run'],capture_output=True,text=True)
    assert proc.returncode!=0 and 'conflicts with fixed protocol' in proc.stderr
    assert not (tmp_path/'output').exists()
