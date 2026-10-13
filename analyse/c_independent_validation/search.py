import time,sys,subprocess
import numpy as np
import torch
from .common import *
from .performance import SEEDS,METHODS,seal,set_stage,verify_performance,validate_public
from .oracle import Client,learner_guard
from analyse.structured_kernel_phase1.gp import cpu_setup,restore_model
from analyse.structured_kernel_development_replay.learner import fit_current
from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter,enumerate_scores,mc_seed,choose

def complete_trajectory(folder):
 done=read(folder/'complete.json');verify(folder,done['files']);obs=read(folder/'observations.json')
 if len(obs)!=71 or len({o['candidate_id'] for o in obs})!=71:raise ValueError('Invalid trajectory budget')
 return done

def worker(root,repo):
 cpu_setup();public=read(root/'public_C.json');validate_public(public);initial=read(root/'initial_observation_manifest.json');random=read(root/'random_query_manifest.json');identity=read(root/'performance_identity.json')['identity']
 client=Client(root);learner_guard(root,repo);started=time.monotonic();new=0
 try:
  for seed in SEEDS:
   for method in METHODS:
    tr=f'{seed}_{method}';folder=root/'trajectories'/tr;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():complete_trajectory(folder);continue
    op=folder/'observations.json';obs=read(op) if op.exists() else []
    if [o['candidate_id'] for o in obs[:50]]!=initial[str(seed)]['ids'][:min(len(obs),50)]:raise ValueError('Initialization drift')
    while len(obs)<50:
     obs.append(client.release(tr,len(obs),initial[str(seed)]['ids'][len(obs)]));write(op,obs)
    while len(obs)<71:
     n=len(obs);step=n-49;dp=folder/'decisions'/f'{step:02d}.json';dp.parent.mkdir(exist_ok=True)
     if method!='Random':fit,q,query=fit_current(folder/'fits'/f'n{n}',public,obs,seed,method,identity)
     if dp.exists():
      d=read(dp)
      if method!='Random':
       if d['model_sha256']!=fit['selected_model_sha256'] or d['fit_identity']!=fit['identity'] or d['candidate_ids']!=query['ids']:raise ValueError('Resumed decision model changed')
       if step<=20 and sha(dp.with_suffix('.sampler.pt'))!=d['sampler_sha256']:raise ValueError('Resumed sampler changed')
     else:
      start=time.monotonic()
      if method=='Random':
       cid=random[str(seed)]['query_ids'][step-1];d={'trajectory':tr,'step':step,'observed_n':n,'kind':'random','candidate_id':cid,'observed_ids':[o['candidate_id'] for o in obs],'sealed_at':now()}
      else:
       state=torch.load(folder/'fits'/f'n{n}'/fit['selected_model'],map_location='cpu',weights_only=False);model=restore_model(state);adapter=FrozenAdapter(model,state['jitter_index'])
       best_f=(max(o['mean'] for o in obs)-fit['scaler']['y_mean'])/fit['scaler']['y_scale'];kind='qlogei' if step<=20 else 'terminal_greedy'
       if step<=20:
        scores,sampler=enumerate_scores(adapter,q,best_f,seed,step)
        sampler_path=dp.with_suffix('.sampler.pt');torch.save(sampler,sampler_path);sampler_hash=sha(sampler_path)
       else:
        with torch.no_grad():scores=adapter.posterior(q.unsqueeze(-2)).mean[:,0,0].numpy()*fit['scaler']['y_scale']+fit['scaler']['y_mean']
        sampler_hash=None
       cid=choose(query['ids'],scores);d={'trajectory':tr,'step':step,'observed_n':n,'kind':kind,'candidate_id':cid,'observed_ids':[o['candidate_id'] for o in obs],
        'candidate_ids':query['ids'],'scores':scores.tolist(),'model_sha256':fit['selected_model_sha256'],'fit_identity':fit['identity'],
        'best_f_standardized':best_f,'mc_seed':mc_seed(seed,step) if step<=20 else None,'sampler_sha256':sampler_hash,
        'numerical_events':adapter.policy.events,'jitter':adapter.policy.jitter,'acquisition_seconds':time.monotonic()-start,'sealed_at':now()}
      seal(dp,d)
     if d['observed_ids']!=[o['candidate_id'] for o in obs] or d['observed_n']!=len(obs):raise ValueError('Decision state drift')
     obs.append(client.release(tr,n,d['candidate_id']));write(op,obs);new+=1
     write(root/'P5_progress.json',{'state':'RUNNING','trajectory':tr,'n':len(obs),'new_decisions_this_process':new,'elapsed_seconds':time.monotonic()-started,'time':now()})
     print(json.dumps({'P5_trajectory':tr,'n':len(obs)}),flush=True)
    files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='complete.json' and not p.name.endswith('.tmp')}
    seal(folder/'complete.json',{'trajectory':tr,'performance_identity':identity,'files':files,'observations':71,'fits':0 if method=='Random' else 21,'starts':0 if method=='Random' else 63})
 finally:client.close()

def run_search(root,repo):
 verify_performance(root,repo)
 if (root/'P5_complete.json').exists():verify(root,read(root/'P5_complete.json')['files']);return
 assert (root/'P4_complete.json').exists();set_stage(root,'P5','RUNNING');started=time.monotonic()
 with (root/'P5_worker.log').open('a') as f:
  result=subprocess.run([sys.executable,'-B','-m','analyse.c_independent_validation','_search_worker','--run-dir',str(root)],cwd=repo,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('Search worker failed; decisions/fits/releases retained')
 files={};fits=starts=0
 for seed in SEEDS:
  for method in METHODS:
   folder=root/'trajectories'/f'{seed}_{method}';done=complete_trajectory(folder);fits+=done['fits'];starts+=done['starts'];files[str((folder/'complete.json').relative_to(root))]=sha(folder/'complete.json')
 assert fits==420 and starts==1260
 oracle=read(root/'oracle_summary.json');assert oracle['logical_releases']==2130
 write(root/'P5_complete.json',{'files':files,'fits':fits,'starts':starts,'trajectories':30,'oracle':oracle,'wall_seconds':time.monotonic()-started,'time':now()});set_stage(root,'P5','COMPLETE')
