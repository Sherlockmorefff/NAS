"""Independent member updates, durable commits, bounded physical retries."""
import copy,time,uuid
from pathlib import Path
import torch
from analyse.relative_metric_pilot import save,load,write,digest,tensor_hash,utc
from surrogate.gp_guidance_replay_engine import rng_state,restore_rng,append,journal,pause_once
from surrogate.champion_margin import list_indices,objective

def fit(folder,model,data,steps,cfg,meta,retry_counter,pause_at=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ident=digest(dict(meta=meta,steps=steps,config=cfg,input_hash=tensor_hash({k:data[k] for k in ['z','hp','y','target32']}),scaler=data['scaler']))
    opt=torch.optim.Adam(model.parameters(),**cfg['adam']);rho=(50-meta['q'])/(159-meta['q'])
    if (folder/'latest.pt').exists():
        state=load(folder/'latest.pt');assert state['identity']==ident
        model.load_state_dict(state['model']);opt.load_state_dict(copy.deepcopy(state['optimizer']));restore_rng(state['rng'])
        append(folder/'resumes.jsonl',dict(completed_steps=state['step'],model_hash=tensor_hash(state['model']),time=utc()))
    else:
        state=dict(identity=ident,meta=meta,step=0,steps=steps,model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state(),
            known_ids=data['ids'],scaler=data['scaler'],known_ids_hash=digest(data['ids']),scaler_hash=digest(data['scaler']),model_hash=tensor_hash(model.state_dict()),last_log=None)
        save(folder/'latest.pt',state);save(folder/'step000.pt',{**state,'timing':'event_start'})
    logs=journal(folder/'steps.jsonl');assert len({r['completed_steps'] for r in logs})==len(logs)
    if state['step'] and (not logs or logs[-1]['completed_steps']<state['step']):append(folder/'steps.jsonl',state['last_log'])
    seen={r['event_key'] for r in journal(folder/'physical.jsonl') if r['phase']=='intent'}
    snapshots={50,100,steps};model.train()
    if state['step'] in snapshots and not (folder/f"step{state['step']:03}.pt").exists():save(folder/f"step{state['step']:03}.pt",{**state,'timing':'post_update'})
    while state['step']<steps:
        step=state['step'];start=time.monotonic();key=f"{meta['owner']}/{meta['event']}/member{meta['member']}/{step+1}"
        if key in seen:
            if retry_counter['remaining']<=0:raise RuntimeError('512 extra physical attempts budget exhausted')
            retry_counter['remaining']-=1
        lists=list_indices(data['ids'],'replay',f"start{meta['seed']}-s0",meta['event'],step,cfg)
        opt.zero_grad(set_to_none=True);e,s=model(data['z'],data['hp'])
        losses,_=objective(e,s,data['target32'],data['y'],data['ids'],lists,rho,data['scaler']['y_scale'],cfg,'fixed',0.)
        assert all(torch.isfinite(v) for v in losses.values())
        log=dict(timing='pre_update',pre_completed_steps=step,completed_steps=step+1,event_key=key,pre_model_hash=tensor_hash(model.state_dict()),
            scaler_hash=state['scaler_hash'],known_ids_hash=state['known_ids_hash'],list_hash=digest(lists),**{k:float(v.detach()) for k,v in losses.items()})
        losses['total'].backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        attempt=uuid.uuid4().hex;append(folder/'physical.jsonl',dict(phase='intent',attempt=attempt,event_key=key,retry=key in seen,time=utc()))
        opt.step();append(folder/'physical.jsonl',dict(phase='computed',attempt=attempt,event_key=key,time=utc()))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        log.update(post_model_hash=tensor_hash(model.state_dict()),compute_seconds=time.monotonic()-start)
        state.update(step=step+1,model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state(),last_log=log,model_hash=log['post_model_hash'])
        save(folder/'latest.pt',state);append(folder/'steps.jsonl',log)
        if state['step'] in snapshots:save(folder/f"step{state['step']:03}.pt",{**state,'timing':'post_update'})
        pause_once(folder,'member',state['step']==pause_at,dict(kind='member_training',completed_steps=state['step'],model_hash=state['model_hash']))
    if state['step'] in snapshots and not (folder/f"step{state['step']:03}.pt").exists():save(folder/f"step{state['step']:03}.pt",{**state,'timing':'post_update'})
    write(folder/'completed.json',dict(steps=steps,model_hash=state['model_hash'],latest_sha=__import__('analyse.relative_metric_pilot',fromlist=['file_hash']).file_hash(folder/'latest.pt')))
    return state
