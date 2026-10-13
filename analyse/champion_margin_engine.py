"""Small-network training; durable logical steps, separate physical attempts."""
import copy,time,uuid
from pathlib import Path
import torch
from analyse.relative_metric_pilot import save,load,write,digest,tensor_hash,utc
from surrogate.gp_guidance_replay_engine import rng_state,restore_rng,append,journal
from surrogate.champion_margin import list_indices,objective


def fit(folder,model,data,steps,config,mode,weight,rho,meta,interrupt_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(meta=meta,steps=steps,config=config,mode=mode,weight=weight,rho=rho,
        input_hash=tensor_hash({k:data[k] for k in ['z','hp','target32','y']}),scaler=data['scaler']))
    opt=torch.optim.Adam(model.parameters(),**config['adam'])
    if (folder/'latest.pt').exists():
        state=load(folder/'latest.pt');assert state['identity']==identity
        model.load_state_dict(state['model']);opt.load_state_dict(copy.deepcopy(state['optimizer']));restore_rng(state['rng'])
        append(folder/'resumes.jsonl',dict(completed_steps=state['step'],model_hash=tensor_hash(state['model']),time=utc()))
    else:
        state=dict(identity=identity,meta=meta,step=0,steps=steps,mode=mode,weight=weight,rho=rho,config=config,
            scaler=data['scaler'],known_ids=data['ids'],model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
            rng=rng_state(),last_log=None,compute_seconds=0.)
        save(folder/'latest.pt',state);save(folder/'post_step000.pt',state)
    attempts=journal(folder/'physical_attempts.jsonl');seen={a['event_key'] for a in attempts if a['phase']=='intent'}
    model.train()
    while state['step']<steps:
        step=state['step'];started=time.monotonic();event_key=f"{meta['owner']}/{meta['update_event']}/{step+1}"
        lists=list_indices(data['ids'],'replay',meta['task_id'],meta['update_event'],step,config)
        opt.zero_grad(set_to_none=True);embedding,scores=model(data['z'],data['hp'])
        losses,diag=objective(embedding,scores,data['target32'],data['y'],data['ids'],lists,rho,data['scaler']['y_scale'],config,mode,weight)
        assert all(torch.isfinite(v) for v in losses.values())
        pre_hash=tensor_hash(model.state_dict())
        log=dict(timing='pre_update',completed_steps=step,next_step=step+1,event=meta['update_event'],model_hash=pre_hash,
            scaler_hash=digest(data['scaler']),known_ids_hash=digest(data['ids']),list_hash=digest(lists),event_key=event_key,
            **{k:float(v.detach()) for k,v in losses.items()},**diag)
        snapshot=(step+1 in config['snapshot_steps'] or step+1==steps)
        if snapshot:
            pre=dict(state);pre.update(timing='pre_update',model_hash=pre_hash,scaler_hash=digest(data['scaler']),known_ids_hash=digest(data['ids']),
                scores=scores.detach().clone(),embedding=embedding.detach().clone(),lists=lists,diagnostics=log)
            save(folder/f'pre_step{step:03}.pt',pre)
        losses['total'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        attempt=uuid.uuid4().hex
        append(folder/'physical_attempts.jsonl',dict(phase='intent',attempt=attempt,event_key=event_key,time=utc(),
            retry=event_key in seen,reason='recompute_after_uncommitted_checkpoint' if event_key in seen else 'first_attempt'))
        opt.step()
        append(folder/'physical_attempts.jsonl',dict(phase='computed',attempt=attempt,event_key=event_key,time=utc()))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        log['post_model_hash']=tensor_hash(model.state_dict());log['compute_seconds']=time.monotonic()-started
        state.update(step=step+1,model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state(),
            last_log=log,compute_seconds=state['compute_seconds']+log['compute_seconds'])
        # The durable state defines logical commit; a missing log is recoverable from last_log.
        save(folder/'latest.pt',state)
        write(folder/'step_logs'/f'{step+1:03}.json',log)
        if snapshot:
            post=dict(state);post.update(timing='post_update',model_hash=log['post_model_hash'],scaler_hash=digest(data['scaler']),known_ids_hash=digest(data['ids']))
            save(folder/f'post_step{step+1:03}.pt',post)
        if interrupt_after==step+1:return state
    if state['last_log'] is not None:write(folder/'step_logs'/f"{state['step']:03}.json",state['last_log'])
    return state
