#!/usr/bin/env python3
"""One exclusive serial executor. No resume, labels, or adaptive endpoint choice."""
import argparse,json,hashlib,os,time,traceback,datetime,platform,subprocess
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import gpytorch
from readout_core import budget_limit
from readout_core import Budget,BudgetExceeded,ReadoutModel,Session,cpu_setup
from readout_data import verify_cached_context
from readout_diagnostics import diagnose,stability_at,DIAGNOSTIC_STEPS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
@contextmanager
def measured(budget,kind):
 start=time.monotonic()
 try:yield
 finally:budget.charge(time.monotonic()-start,kind)
def verify_files(root,m):
 for f,h in m['files'].items():
  if sha(root/f)!=h:raise RuntimeError('Frozen file changed: '+f)
def freeze(root):
 if not read(root/'verification/v2_migration_checks.json')['pass']:raise RuntimeError('V2 migration verification failed')
 if not read(root/'audit/migration_invariance.json')['pass']:raise RuntimeError('Independent migration audit failed')
 if (root/'frozen_manifest.json').exists():raise RuntimeError('Already frozen')
 verify_files(root,read(root/'source_manifest.json'));verify_files(root,read(root/'cache_manifest.json'))
 pre=read(root/'validation_parent/preflight.json');checks=read(root/'validation_parent/source_cache_checks.json')
 if not pre['pass'] or not checks['pass'] or not read(root/'validation_parent/final_identity_check/acceptance.json')['pass']:raise RuntimeError('Preflight not accepted')
 budget=Budget(root/'budget.json',limit=budget_limit(root))
 # Timings are actual cumulative successful synthetic step wall times.
 cells=pre['timing_cells'];assert len(cells)==6
 fit=sum(c['seconds_per_step']*45000 for c in cells)
 pred=pre['prediction_seconds']*1080/pre['prediction_calls']
 conservative=budget.seconds+1.5*fit+max(30.,2*pred)
 admit={'admitted':conservative<=budget.limit,'used_model_seconds':budget.seconds,'estimated_fit_seconds':fit,'estimated_diagnostic_seconds':pred,'expected_total_seconds':budget.seconds+fit+pred,'conservative_total_seconds':conservative,'safety_factor_fit':1.5,'diagnostic_safety_factor':2,'minimum_diagnostic_reserve_seconds':30,'planned_formal_updates':270000,'planned_diagnostic_calls':1080,'remaining_verification_seconds':0,'source_verification_finished':True,'formula_predeclared_in_protocol':True,'timing_cells':cells,'limit':budget.limit,'parent_timing_reused':True}
 dump(root/'admission.json',admit)
 if not admit['admitted']:
  dump(root/'status.json',{'status':'BLOCKED_BUDGET_ADMISSION','admission':admit});print(admit);return
 files={str(p.relative_to(root)):sha(p) for p in sorted((root/'code').rglob('*.py'))}
 dump(root/'code_manifest.json',{'files':files})
 frozen={f:sha(root/f) for f in ['PROMPT.md','protocol.json','worklist.json','source_manifest.json','cache_manifest.json','code_manifest.json','admission.json','validation_parent/preflight.json','validation_parent/source_cache_checks.json','protocol_amendment.json','validation_reuse.json','budget_carryover.json','migration_manifest.json','environment_frozen.json','audit/migration_invariance.json']}
 frozen.update(read(root/'cache_manifest.json')['files'])
 frozen.update(read(root/'migration_manifest.json')['files'])
 frozen.update({str(p.relative_to(root)):sha(p) for p in sorted((root/'verification').rglob('*.json'))})
 dump(root/'frozen_manifest.json',{'frozen_at_utc':now(),'files':frozen,'no_changes_after_formal_entry':True})

 dump(root/'status.json',{'status':'READY_ADMITTED','admission':admit});print(admit)

def run(root):
 start=time.monotonic();budget=Budget(root/'budget.json',limit=budget_limit(root));cpu_setup(0)
 if read(root/'status.json')['status']!='READY_ADMITTED':raise RuntimeError('Not admitted')
 admission=read(root/'admission.json')
 if not admission['admitted'] or admission['limit']!=budget.limit or admission['conservative_total_seconds']>budget.limit or admission['used_model_seconds']!=budget.seconds:raise RuntimeError('Admission identity/budget mismatch')
 if budget.formal_updates or budget.formal_optimizer_calls or (root/'scoring/release.json').exists():raise RuntimeError('Formal run must start at zero before scoring')
 for name in ['frozen_manifest','source_manifest','cache_manifest','code_manifest']:verify_files(root,read(root/(name+'.json')))
 lock=root/'formal_run_once.lock';lock.mkdir()
 dump(lock/'entry.json',{'at_utc':now(),'pid':os.getpid(),'automatic_resume':False,'unclean_exit_status':'INTERRUPTED_UNVERIFIED','protocol_sha256':sha(root/'protocol.json')})
 dump(root/'status.json',{'status':'RUNNING_UNVERIFIED_UNTIL_SEALED','hard_kill_must_be_classified':'INTERRUPTED_UNVERIFIED','started_at_utc':now()})
 work=read(root/'worklist.json');entries=[];failure=None;active=None;active_uid=None;cache=None
 costs=[];status='COMPLETE_VALID'
 try:
  for c in work['contexts']:
   cid=c['context_id'];ct=time.monotonic();budget.check()
   with measured(budget,'verification'):
    cache=torch.load(root/'caches'/f'{cid}.pt',map_location='cpu',weights_only=False)
    verify_cached_context(cache);t,q=cache['training'],cache['query'];ident=cache['identity'];sc=t['scaler'];yc,ys=sc['y_mean'],sc['y_scale']
   def base(method,state_path,statehash,fingerprint):
    return {**{k:c[k] for k in ['context_id','subset_id','fold','sample_seed','model_seed','n']},'unit_id':cid+'_'+method,'method':method,'candidate_ids':q['ids'],'architecture_fingerprints':q['architectures'],'training_ids_sha256':ident['training_ids_sha256'],'y_center':yc,'y_scale':ys,'source_fingerprint':ident['source_fingerprint'],'cached_features_sha256':ident['cache_sha256'],'endpoint_state_hash':fingerprint,'state_path':state_path,'state_file_sha256':statehash,'query_labels_or_SE_present':False,'protocol_sha256':sha(root/'protocol.json'),'source_manifest_sha256':sha(root/'source_manifest.json')}
   def finish(p,state_path):
    uid=p['unit_id'];path='predictions/'+uid+'.json';dump(root/path,p)
    entry={'unit_id':uid,'path':path,'sha256':sha(root/path),'state_path':state_path,'state_file_sha256':sha(root/state_path),'endpoint_state_hash':p['endpoint_state_hash']}
    # The completion marker is emitted only after saved state + prediction.
    dump(root/'completed'/f'{uid}.json',{'completed_at_utc':now(),**entry,'actual_updates':p['actual_updates'],'budget':budget.as_dict()})
    entries.append(entry);budget.persist()
   # HEAD uses cached s0 once; no optimizer and no fabricated variance.
   with measured(budget,'prediction'):
    headmean=q['s0'].tolist();headraw=(q['s0']*ys+yc).tolist()
   budget.check()
   sp='caches/'+cid+'.pt';p=base('HEAD',sp,sha(root/sp),ident['cache_sha256'])
   p.update(mean_standardized=headmean,mean_raw=headraw,latent_variance_standardized=None,latent_variance_raw=None,extra_noise_variance_standardized=None,actual_updates=0,semi_decomposition=None,stability=None)
   finish(p,sp)
   for method in c['group_order']:
    active_uid=cid+'_'+method;before=budget.as_dict();st=time.monotonic();budget.check()
    with measured(budget,'verification'):
     model=ReadoutModel(method,t['h0'],t['s0'],t['y_standardized'],t['known_variance_standardized'],cache['common_gp_state'],ident['source_fingerprint']);active=Session(model)
    records=[];updates=[]
    def diagnostic(step):
     budget.check();ds=time.monotonic()
     try:r=diagnose(model,q['h0'],q['s0'],q['ids'],yc,ys,step)
     finally:budget.charge(time.monotonic()-ds,'prediction')
     records.append(r)
     # Lossless covariance/kernel archive; vectors and metadata remain readable.
     matrices={k:np.asarray(r[k],dtype=np.float64) for k in ['query_latent_covariance_standardized','query_latent_covariance_raw','matern_training_kernel','full_training_kernel']}
     rel=f'diagnostics/{active_uid}/j{step}.npz';(root/rel).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(root/rel,**matrices)
     small={k:v for k,v in r.items() if k not in matrices};small['matrix_archive_path']=rel;small['matrix_archive_sha256']=sha(root/rel)
     dump(root/f'diagnostics/{active_uid}/j{step}.json',small);budget.persist();budget.check()
    diagnostic(0)
    for step in range(1,1501):
     updates.append(active.step(budget,'formal'))
     if step in DIAGNOSTIC_STEPS:diagnostic(step)
    with measured(budget,'verification'):
     verify_cached_context(cache)
     if model.frozen_hash()!=model.initial_frozen_hash:raise RuntimeError('Frozen model input changed')
     stability=stability_at(records)
    # Snapshot covers model, fresh optimizer, scheduler, RNG, frozen h/s and cost.
    with measured(budget,'verification'):state=active.snapshot()
    budget.check()
    state.update(protocol_sha256=sha(root/'protocol.json'),source_manifest_sha256=sha(root/'source_manifest.json'),cache_identity=ident,budget_at_save=budget.as_dict())
    sp='states/'+active_uid+'.pt';(root/'states').mkdir(exist_ok=True);torch.save(state,root/sp)
    last=records[-1];p=base(method,sp,sha(root/sp),state['fingerprint'])
    p.update(mean_standardized=last['query_mean_standardized'],mean_raw=last['query_mean_raw'],latent_variance_standardized=last['query_latent_variance_standardized'],latent_variance_raw=last['query_latent_variance_raw'],latent_covariance_standardized=last['query_latent_covariance_standardized'],extra_noise_variance_standardized=last['extra_noise_variance_standardized'],semi_decomposition=last['semi_decomposition'],stability=stability,actual_updates=active.completed,parameter_names=model.parameter_names(),diagnostic_steps=list(DIAGNOSTIC_STEPS),gp_parameters=last['gp_parameters'])
    dump(root/'updates'/f'{active_uid}.json',updates)
    costs.append({'unit_id':active_uid,'method':method,'n':c['n'],'actual_updates':active.completed,'training_seconds':active.training_seconds,'model_seconds_total':budget.seconds-before['seconds'],'parameter_names':model.parameter_names()})
    finish(p,sp);active=None;active_uid=None
   dump(root/'progress.json',{'context_id':cid,'completed_endpoints':len(entries),'completed_gp':sum(x['unit_id'].split('_')[-1]!='HEAD' for x in entries),'budget':budget.as_dict(),'wall_seconds':time.monotonic()-start})
   print(json.dumps({'context':cid,'endpoints':len(entries),'updates':budget.formal_updates,'seconds':round(budget.seconds,3)}),flush=True)
  budget.check()
  if len(entries)!=240 or budget.formal_updates!=270000:raise RuntimeError('Complete matrix accounting failed')
 except BudgetExceeded as e:
  status='STOPPED_BUDGET_PARTIAL';failure=str(e)
 except BaseException as e:
  status='INTERRUPTED_UNVERIFIED';failure=traceback.format_exc()
 finally:
  if active is not None:
   try:
    with measured(budget,'verification'):state=active.snapshot()
    state.update(failure=failure,budget_at_save=budget.as_dict());(root/'partial').mkdir(exist_ok=True);torch.save(state,root/'partial'/f'{active_uid}.pt')
   except Exception as e:failure=(failure or '')+'\nPartial snapshot failed: '+repr(e)
  budget.persist();dump(root/'fit_costs.json',costs)
  expected=[e['unit_id'] for e in work['endpoints']];actual={e['unit_id'] for e in entries}
  manifest={'sealed':True,'sealed_at_utc':now(),'execution_status':status,'expected_endpoint_count':240,'expected_unit_ids':expected,'files':entries,'missing_unit_ids':[u for u in expected if u not in actual],'protocol_sha256':sha(root/'protocol.json'),'source_manifest_sha256':sha(root/'source_manifest.json'),'code_manifest_sha256':sha(root/'code_manifest.json'),'worklist_sha256':sha(root/'worklist.json'),'failure':failure,'resources':budget.as_dict(),'next_stage_authorized':False}
  dump(root/'predictions_manifest.json',manifest);dump(root/'status.json',{'status':status,'completed_endpoints':len(entries),'formal_updates':budget.formal_updates,'budget':budget.as_dict(),'formal_wall_seconds':time.monotonic()-start,'failure':failure,'at_utc':now()})
  print(json.dumps({'status':status,'completed_endpoints':len(entries),'budget':budget.as_dict(),'failure':failure}),flush=True)
 if status!='COMPLETE_VALID':raise SystemExit(2)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--freeze',action='store_true');v=a.parse_args();cpu_setup(0);freeze(v.root) if v.freeze else run(v.root)
