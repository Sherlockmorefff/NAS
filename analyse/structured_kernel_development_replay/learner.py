"""Learner consumes public pool + progressively released observations only."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from analyse.structured_kernel_phase1.io import read_json,write_json,digest,sha256,append_jsonl,now
from analyse.structured_kernel_phase1.data import training_projection
from analyse.structured_kernel_phase1.gp import cpu_setup,make_inputs,fit_start,select_start,restore_model,predict_frozen,_save_torch
from .protocol import METHODS,SEEDS
from .acquisition import FrozenAdapter,enumerate_scores,mc_seed,choose
from .oracle import access_guard


def seal(path,value):
    if path.exists():
        if read_json(path)!=value:
            raise ValueError('Refusing to overwrite a sealed artifact: '+str(path))
    else:
        write_json(path,value)


def build_training(public,observations,seed,method):
    index={c:i for i,c in enumerate(public['ids'])}
    ids=[o['candidate_id'] for o in observations]
    if len(ids)!=len(set(ids)) or not set(ids)<=set(index):
        raise ValueError('Invalid observed identities')
    positions=[index[c] for c in ids]
    return training_projection(ids,[public['x_raw'][i] for i in positions],
        [o['mean'] for o in observations],[o['se']**2 for o in observations],
        [public['wl'][i] for i in positions],subset_id=f'{seed}_{method}_n{len(ids)}',fold=-1,draw=seed)


def fit_current(folder,public,observations,seed,method,identity):
    t=build_training(public,observations,seed,method)
    observed=set(t['ids'])
    qi=sorted((i for i,c in enumerate(public['ids']) if c not in observed),key=lambda i:public['ids'][i])
    query={k:[public[k][i] for i in qi] for k in ('ids','x_raw','wl')}
    x,q,arch,scales=make_inputs(t,query,method)
    fitidentity=digest({'run_identity':identity,'method':method,'seed':seed,'observations':observations,
        'training':t,'distance_scales':scales})
    folder.mkdir(parents=True,exist_ok=True)
    summary_path=folder/'fit.json'
    if summary_path.exists():
        fit=read_json(summary_path)
        if fit['identity']!=fitidentity:
            raise ValueError('Fit identity mismatch')
        for rel,expected in fit['files'].items():
            if sha256(folder/rel)!=expected:
                raise ValueError('Fit artifact changed')
        return fit,q,query
    seal(folder/'training.json',t)
    results=[]
    for start in range(3):
        path=folder/f'start{start}.pt'
        result=fit_start(x,t['y_standardized'],t['known_variance_standardized'],method,arch,start,path,fitidentity,resume=True)
        results.append(result)
        if not result['success']:
            raise RuntimeError('Numerical failure, cannot discard failed start: '+str(result))
        # All three starts must have finite posteriors, even though selection is
        # based exclusively on training NLL.
        predict_frozen(path,q)
    selected=select_start(results)
    files={p.name:sha256(p) for p in sorted(folder.iterdir()) if p.suffix in ('.pt','.json','.jsonl')}
    fit=dict(identity=fitidentity,n=len(observations),method=method,seed=seed,selected_start=selected,
        selected_model=f'start{selected}.pt',selected_model_sha256=sha256(folder/f'start{selected}.pt'),
        starts=results,scaler=t['scaler'],distance_scales=scales,files=files,
        fit_seconds=sum(r['elapsed_seconds'] for r in results),completed_at=now())
    seal(summary_path,fit)
    return fit,q,query


class OracleClient:
    def __init__(self,root):
        self.process=subprocess.Popen([sys.executable,'-B','-m','analyse.structured_kernel_development_replay',
            '_oracle','--run-dir',str(root)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        response=json.loads(self.process.stdout.readline())
        if not response.get('ready'):
            raise RuntimeError('Oracle failed initialization')

    def request(self,value):
        self.process.stdin.write(json.dumps(value)+'\n'); self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:
            raise RuntimeError('Oracle process disconnected')
        result=json.loads(line)
        if not result['ok']:
            raise RuntimeError(result['error'])
        return result

    def release(self,tr,slot,cid):
        return self.request(dict(operation='release',trajectory=tr,slot=slot,candidate_id=cid))['observation']

    def close(self):
        if self.process.poll() is None:
            result=self.request({'operation':'close'})
            self.process.wait(timeout=30)
            return result['summary']


def validate_complete(folder):
    done=read_json(folder/'complete.json')
    for name,expected in done['files'].items():
        if sha256(folder/name)!=expected:
            raise ValueError('Complete trajectory artifact changed: '+name)
    observations=read_json(folder/'observations.json')
    ids=[o['candidate_id'] for o in observations]
    if len(ids)!=71 or len(set(ids))!=71 or len(list((folder/'fits').glob('n*/fit.json')))!=21:
        raise ValueError('Invalid complete budget')
    if len(list((folder/'decisions').glob('*.json')))!=21:
        raise ValueError('Invalid decision budget')
    return done


def run_worker(root,identity,max_new_decisions=None):
    cpu_setup()
    public=read_json(root/'public_features.json')
    initial=read_json(root/'initial_observation_manifest.json')
    client=OracleClient(root)
    access_guard(root,Path.cwd(),'learner')
    new_decisions=0
    started=time.perf_counter()
    try:
        for seed in SEEDS:
            for method in METHODS:
                tr=f'{seed}_{method}'
                folder=root/'trajectories'/tr
                folder.mkdir(parents=True,exist_ok=True)
                if (folder/'complete.json').exists():
                    validate_complete(folder)
                    continue
                obs_path=folder/'observations.json'
                observations=read_json(obs_path) if obs_path.exists() else []
                if [o['candidate_id'] for o in observations[:50]]!=initial[str(seed)]['ids'][:min(50,len(observations))]:
                    raise ValueError('Initial observation mismatch')
                while len(observations)<50:
                    slot=len(observations)
                    o=client.release(tr,slot,initial[str(seed)]['ids'][slot])
                    observations.append(o); write_json(obs_path,observations)
                    append_jsonl(root/'learner_access_log.jsonl',dict(time=now(),trajectory=tr,slot=slot,candidate_id=o['candidate_id'],operation='receive_initial'))
                while len(observations)<71:
                    n=len(observations); step=n-49
                    fdir=folder/'fits'/f'n{n:03d}'
                    fit,q,query=fit_current(fdir,public,observations,seed,method,identity)
                    path=folder/'decisions'/f'{step:02d}.json'
                    if path.exists():
                        decision=read_json(path)
                        if decision['model_sha256']!=fit['selected_model_sha256'] or decision['observed_ids']!=[o['candidate_id'] for o in observations]:
                            raise ValueError('Decision source changed')
                        sampler_path=folder/'decisions'/f'{step:02d}_sampler.pt'
                        if decision['kind']=='qlogei' and sha256(sampler_path)!=decision['sampler_sha256']:
                            raise ValueError('Sampler identity changed')
                    else:
                        start=time.perf_counter()
                        checkpoint=torch.load(fdir/fit['selected_model'],map_location='cpu',weights_only=False)
                        adapter=FrozenAdapter(restore_model(checkpoint),checkpoint['jitter_index'])
                        kind='greedy' if step==21 else 'qlogei'
                        sampler_hash=None
                        best_f=(max(o['mean'] for o in observations)-fit['scaler']['y_mean'])/fit['scaler']['y_scale']
                        if kind=='qlogei':
                            scores,base=enumerate_scores(adapter,q,best_f,seed,step)
                            sampler_path=folder/'decisions'/f'{step:02d}_sampler.pt'
                            _save_torch(sampler_path,{'mc_seed':mc_seed(seed,step),'base_samples':base})
                            sampler_hash=sha256(sampler_path)
                        else:
                            with torch.no_grad():
                                scores=adapter.posterior(q.unsqueeze(-2)).mean[:,0,0].numpy()*fit['scaler']['y_scale']+fit['scaler']['y_mean']
                        cid=choose(query['ids'],scores)
                        decision=dict(trajectory=tr,step=step,kind=kind,observed_n=n,
                            observed_ids=[o['candidate_id'] for o in observations],candidate_id=cid,
                            model_sha256=fit['selected_model_sha256'],fit_identity=fit['identity'],
                            candidate_ids=query['ids'],scores=scores.tolist(),best_f_standardized=best_f,
                            mc_seed=mc_seed(seed,step) if kind=='qlogei' else None,sampler_sha256=sampler_hash,
                            numerical_events=adapter.policy.events,jitter=adapter.policy.jitter,
                            acquisition_seconds=time.perf_counter()-start,sealed_at=now())
                        seal(path,decision)
                    o=client.release(tr,n,decision['candidate_id'])
                    observations.append(o); write_json(obs_path,observations)
                    append_jsonl(root/'learner_access_log.jsonl',dict(time=now(),trajectory=tr,slot=n,candidate_id=o['candidate_id'],operation='receive_'+decision['kind'],decision_sha256=sha256(path)))
                    new_decisions+=1
                    progress={'state':'RUNNING','trajectory':tr,'observed_n':len(observations),'updated_at':now(),
                        'independent_confirmation':False,'run_identity':identity,
                        'completed_trajectories':len(list((root/'trajectories').glob('*/complete.json'))),
                        'completed_fits':len(list((root/'trajectories').glob('*/fits/n*/fit.json')))}
                    progress['estimated_remaining_seconds']=((time.perf_counter()-started)/new_decisions)*(420-progress['completed_fits'])
                    write_json(root/'status.json',progress)
                    print(json.dumps(progress),flush=True)
                    if max_new_decisions and new_decisions>=max_new_decisions:
                        return 'INCOMPLETE'
                files={str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file() and not p.name.endswith('.tmp')}
                seal(folder/'complete.json',dict(trajectory=tr,run_identity=identity,files=files,observations=71,fits=21,starts=63,qlogei=20,greedy=1))
                if seed==SEEDS[0] and method==METHODS[-1]:
                    # Engineering checks only. No scorer, full-pool optimum or
                    # performance gate is involved in continuation.
                    for m in METHODS:
                        validate_complete(root/'trajectories'/f'{seed}_{m}')
                    write_json(root/'pilot_acceptance.json',dict(state='PASS',trajectories=2,fits=42,starts=126,performance_used=False,automatically_continue=True))
        return 'COMPLETE'
    finally:
        client.close()
