"""Resumable head training, reusing existing IO, budget, scaler and model utilities."""
from __future__ import annotations
import copy
from pathlib import Path
import time
import torch
from analyse.relative_metric_pilot import save,load,write,digest,tensor_hash
from surrogate.relative_metric import fresh_adam
from surrogate.gp_guidance_replay_engine import rng_state,restore_rng,append,pause_once
from surrogate.budget_champion_ranking import list_indices,objective


class TimedStep:
    def __init__(self,optimizer):
        self.optimizer=optimizer
        self.seconds=0.
    def step(self):
        started=time.monotonic();self.optimizer.step();self.seconds=time.monotonic()-started


def fit(folder,model,data,steps,weight,rho,meta,ledger,interrupt=False):
    folder=Path(folder)
    input_hash=tensor_hash({k:data[k] for k in ['z','hp','target32','y']})
    identity=digest(dict(meta=meta,steps=steps,weight=weight,rho=rho,input_hash=input_hash,scaler=data['scaler']))
    opt=fresh_adam(model)
    if (folder/'latest.pt').exists():
        state=load(folder/'latest.pt');assert state['identity']==identity
        model.load_state_dict(state['model']);opt.load_state_dict(copy.deepcopy(state['optimizer']));restore_rng(state['rng'])
        if 0<state['step']<steps:
            next_lists=list_indices(data['ids'],meta['stage'],meta['task_id'],meta['update_event'],state['step'])
            with torch.no_grad():e,s=model(data['z'],data['hp'])
            append(folder/'recovery.jsonl',dict(step=state['step'],model=tensor_hash(model.state_dict()),
                optimizer_steps=sorted({float(x['step']) for x in opt.state.values()}),next_list_hash=digest(next_lists),
                forward_hash=tensor_hash({'embedding':e,'scores':s}),rng_restored=True,scaler_hash=digest(data['scaler'])))
    else:
        state=dict(identity=identity,meta=meta,step=0,steps=steps,weight=weight,rho=rho,scaler=data['scaler'],
            train_ids=data['ids'],input_hash=input_hash,history=[],committed=[],compute_seconds=0.,save_seconds=0.,
            initial_model_hash=tensor_hash(model.state_dict()),model=copy.deepcopy(model.state_dict()),
            optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state(),gradient_checks=[])
        save(folder/'latest.pt',state);save(folder/'step000.pt',state)
    model.train()
    while state['step']<steps:
        step=state['step'];started=time.monotonic()
        lists=list_indices(data['ids'],meta['stage'],meta['task_id'],meta['update_event'],step)
        opt.zero_grad(set_to_none=True);embedding,scores=model(data['z'],data['hp'])
        losses,diag=objective(embedding,scores,data['target32'],data['y'],data['ids'],lists,rho,weight)
        assert all(torch.isfinite(v) for v in losses.values())
        if step==0:
            grads=torch.autograd.grad(losses['ranking'],list(model.parameters()),retain_graph=True,allow_unused=True)
            byname={n:None if g is None else float(g.double().norm()) for (n,_),g in zip(model.named_parameters(),grads)}
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            state['gradient_checks'].append(dict(step=step,ranking_gradient_norms=byname,
                input_requires_grad=[data['z'].requires_grad,data['hp'].requires_grad],nonzero_fraction=diag['nonzero_fraction']))
        snapshot_step=step+1 in [1,50,100,200,300]
        if snapshot_step:
            diag_sample=dict(optimizer_step=step,lists=lists,ids=data['ids'],raw_y=data['y'].clone(),
                scores=scores.detach().clone(),embedding=embedding.detach().clone(),rho=rho,
                ranking_loss=float(losses['ranking'].detach()),diagnostics=diag)
        losses['total'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        before_ledger=time.monotonic();timed=TimedStep(opt)
        gid=ledger.update('mapping',meta['stage']+'/'+meta['owner']+'/'+meta['update_event'],step+1,timed)
        after_ledger=time.monotonic();compute_seconds=before_ledger-started+timed.seconds
        assert all(torch.isfinite(p).all() for p in model.parameters())
        state['committed'].append(gid)
        state['history'].append(dict(step=step+1,optimizer_step=step,id=gid,list_hash=digest(lists),
            **{k:float(v.detach()) for k,v in losses.items()},**diag,
            training_head_MAE=float((scores.detach().double()*data['scaler']['y_scale']+data['scaler']['y_mean']-data['y']).abs().mean())))
        state.update(step=step+1,model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
            rng=rng_state(),compute_seconds=state['compute_seconds']+compute_seconds)
        if snapshot_step:
            save(folder/'diagnostics'/f'step{step+1:03}.pt',diag_sample)
            save(folder/f'step{step+1:03}.pt',state)
        # Save duration of completed writes is recorded on the next durable state.
        save(folder/'latest.pt',state)
        save_seconds=time.monotonic()-after_ledger+after_ledger-before_ledger-timed.seconds
        state['save_seconds']+=save_seconds
        append(folder/'timing.jsonl',dict(step=step+1,compute_seconds=compute_seconds,save_and_state_seconds=save_seconds))
        pause_once(folder,'training',interrupt and step==0,
            dict(kind='training',step=1,next_optimizer_step=1,model_hash=tensor_hash(model.state_dict())))
    save(folder/'latest.pt',state)
    assert tensor_hash({k:data[k] for k in ['z','hp','target32','y']})==input_hash
    return state
