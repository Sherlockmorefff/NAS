"""Targeted CPU checks; at most four mapping and four GP synthetic updates."""
import argparse
import copy
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
from analyse.fuzzy_loss_interaction_core import *
from analyse.fuzzy_loss_interaction_diagnostics import gradients,flat,loss_shares
from analyse.fuzzy_loss_interaction_run import install_guard,respond_oracle


def guard_probe(o):
    p=read(o/'protocol.json');task=read(o/'tasks.json')[0];folder=o/'synthetic/guard'
    folder.mkdir(parents=True,exist_ok=True);accesses,denied=install_guard(o,folder,task,p)
    for path in [o/'private/evaluation_A.json',o/'train_labels/f1-s0.json',ROOT/'results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2/oracle_validation.json']:
        try:path.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('hidden read allowed')
    observed=read(o/task['label_file']);assert set(observed)==set(task['train_ids'])
    write(folder/'check.json',dict(passed=True,denied=denied,allowed=sorted(accesses)))


def checks(o):
    folder=o/'synthetic';folder.mkdir(exist_ok=True)
    assert not (o/'engineering_checks.json').exists(),'existing valid checks should be reused'
    torch.manual_seed(991);np.random.seed(991)
    y=torch.tensor([0.,.0625,.125],dtype=torch.float64);p=neighborhood_affinity(y,epsilon=.125)
    assert p[0,2]==0 and p[2,0]==0 and abs(float(p[0,1])-math.exp(-1))<1e-15
    assert torch.equal(neighborhood_affinity(torch.tensor([0.,.5,1.]),epsilon=.01),torch.zeros(3,3))
    e=torch.tensor([[0.,0.],[1.,0.],[0.,2.]],dtype=torch.float64,requires_grad=True)
    pred=torch.tensor([.1,.2,.3],dtype=torch.float64,requires_grad=True);target=torch.zeros(3,dtype=torch.float64)
    w=torch.tensor([1.,2.,4.],dtype=torch.float64)
    loss=metric_loss(e,pred,target,p,.1,.001,w)
    manual=0.
    for k,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
        q=1/(1+float((e[i]-e[j]).square().sum()));aff=float(p[i,j]);manual+=float(w[k])*(-aff*math.log(q)-(1-aff)*math.log1p(-q))
    assert abs(float(loss['neighborhood'])-manual/7)<1e-14 and abs(float(loss['scale'])-5/3)<1e-14
    assert abs(float(loss['regression'])-(.01+.04+.09)/3)<1e-14
    for ee,pp,tt,aa,ww in [(e,pred,target,p,torch.zeros(3)),(e[:1],pred[:1],target[:1],p[:1,:1],None)]:
        result=metric_loss(ee,pp,tt,aa,.1,.001,ww);assert result['neighborhood']==0 and torch.isfinite(result['total'])
    loss['total'].backward();assert torch.isfinite(e.grad).all()
    ids=[f'synthetic-{i:02}' for i in range(12)];gen=np.random.default_rng(12)
    features=dict(ids=ids,z=gen.normal(size=(12,12)).tolist(),hp=gen.normal(size=(12,4)).tolist())
    labels={}
    for i,c in enumerate(ids[:8]):
        values=np.array([.65+i*.01,.651+i*.009,.648+i*.011])
        labels[c]=dict(validation_mean=float(values.mean()),validation_se=float(values.std(ddof=1)/math.sqrt(3)),validation_seeds=values.tolist(),training_seeds=[0,1,2])
    scaler=scaler_from_training(features,sorted(labels),labels);data=observed_data(features,scaler,labels,.01)
    try:observed_data(features,scaler,dict(labels,not_in_pool=labels[ids[0]]),.01)
    except (KeyError,ValueError):pass
    else:raise AssertionError('non-observed label not rejected')
    expected=np.std(np.array(labels[ids[0]]['validation_seeds'])-np.array(labels[ids[1]]['validation_seeds']),ddof=1)/math.sqrt(3)
    assert abs(float(data['paired_se'][0,1])-expected)<1e-14
    model=JointMetricModel(12,seed=0);initial=copy.deepcopy(model.state_dict());z=data['z'].clone().requires_grad_()
    emb,head=model(z,data['hp']);head.sum().backward();assert z.grad is None
    model.zero_grad(set_to_none=True);before=tree_hash(dict(model=model.state_dict(),rng=rng_state()))
    vec,numbers=gradients(model,data)
    assert before==tree_hash(dict(model=model.state_dict(),rng=rng_state())) and all(p.grad is None for p in model.parameters())
    assert flat(vec,'regression','regressor').norm()>0 and flat(vec,'neighborhood','regressor').norm()==0 and flat(vec,'scale','regressor').norm()==0
    assert all(flat(vec,k).norm()>0 for k in ['regression','neighborhood','scale'])
    ordinary=components(model,data,0)[0]['total'];old=dict(data,affinity=torch.zeros_like(data['affinity']),weights=torch.zeros_like(data['weights']))
    disabled=components(model,old,0)[0]['total'];assert torch.equal(ordinary,disabled)
    ga=torch.autograd.grad(ordinary,list(model.parameters()));gb=torch.autograd.grad(disabled,list(model.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(ga,gb))
    assert abs(sum(loss_shares(model,data).values())-1)<1e-12
    ledger=Ledger(folder/'ledger','synthetic_only',synthetic=True)
    baseline=fit_mapping(folder/'map_baseline',initial,data,.1,ledger,dict(task='synthetic-map'),steps=2,shots=(0,1,2))
    fit_mapping(folder/'map_resume',initial,data,.1,ledger,dict(task='synthetic-map'),steps=2,shots=(0,1,2),stop_after=1)
    resumed=fit_mapping(folder/'map_resume',initial,data,.1,ledger,dict(task='synthetic-map'),steps=2,shots=(0,1,2))
    assert baseline['model_hash']==resumed['model_hash'] and baseline['optimizer_hash']==resumed['optimizer_hash']
    cache=embedding_cache(reload_model(baseline['model']),data)
    ga=fit_gp(folder/'gp_baseline',cache,data,ledger,dict(task='synthetic-gp'),steps=2)
    fit_gp(folder/'gp_resume',cache,data,ledger,dict(task='synthetic-gp'),steps=2,stop_after=1)
    gb=fit_gp(folder/'gp_resume',cache,data,ledger,dict(task='synthetic-gp'),steps=2)
    assert ga['model_hash']==gb['model_hash'] and ga['optimizer_hash']==gb['optimizer_hash']
    first=predict(cache,data['ids'],ids[8:],ga);second=predict(cache,data['ids'],ids[8:],load(folder/'gp_baseline/latest.pt'))
    assert first==second
    # Repeated oracle delivery reuses the saved receipt without another logical query.
    oracle=folder/'oracle';task=dict(name='synthetic-replay',initial_ids=ids[:2]);pp=dict(identity='synthetic')
    dd=oracle/'tasks'/task['name']/'batches/b00';selected=ids[2:7]
    write(dd/'selection.json',dict(protocol='synthetic',query_count_before=0,known_ids=ids[:2],selected_ids=selected,saved_at=utc()))
    event=dict(batch=0,selection_sha=file_hash(dd/'selection.json'))
    r1=respond_oracle(oracle,task,pp,event,labels);r2=respond_oracle(oracle,task,pp,event,labels)
    assert r1==r2 and len(journal(oracle/'oracle_access.jsonl'))==1 and r1['query_count_after']==5
    subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),str(o),'--guard-probe'],check=True,cwd=ROOT)
    write(o/'engineering_checks.json',dict(passed=True,hand_calculation=True,strict_absolute_boundary=True,p_zero_separation_retained=True,
          zero_weights_and_no_pairs=True,paired_seed_SE=True,shared_gradient_path=True,head_only_regression=True,VAE_detach=True,
          diagnostics_no_grad_optimizer_rng_pollution=True,C_zero_weight_matches_original=True,loss_share_sums_one=True,
          mapping_Adam_resume_exact=True,GP_Adam_resume_exact=True,checkpoint_scores_exact=True,oracle_resume_no_double_query=True,
          hidden_label_guard=True,synthetic_counts=dict(mapping=ledger.counts['mapping'],gp=ledger.counts['gp'],logical_queries=5),time=utc()))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('output',type=Path);a.add_argument('--guard-probe',action='store_true');args=a.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    guard_probe(args.output.resolve()) if args.guard_probe else checks(args.output.resolve())
