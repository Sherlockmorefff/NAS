"""Serial, single-use OOF executor. No scoring labels are opened here."""
from __future__ import annotations
import argparse,datetime,hashlib,json,pathlib,time,traceback

def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def freeze(root):
 if (root/'code_manifest.json').exists():raise RuntimeError('Code already frozen')
 files={str(p.relative_to(root)):sha(p) for p in sorted((root/'code').rglob('*.py')) if '__pycache__' not in str(p)}
 write(root/'code_manifest.json',{'files':files,'frozen_at_utc':now()})
 refs=['protocol.json','source_manifest.json','code_manifest.json','worklist.json','inner_splits_manifest.json','admission.json','environment.json','PROMPT.md','verification/preflight.json','verification/data_checks.json','verification/calibration_checks.json','verification/scoring_checks.json']
 refs+=list(read(root/'inner_splits_manifest.json')['files'])
 write(root/'frozen_manifest.json',{'files':{rel:sha(root/rel) for rel in refs},'frozen_at_utc':now(),'no_resume':True})
 return files

def run(root):
 from oof_core import Budget,budget_limit,prepare_common,common_session,freeze_features,readout_session,common_diagnostic,head_predictions,readout_diagnostic,charged_snapshot,measured
 from oof_data import load_inner
 from readout_diagnostics import stability_at
 import torch
 from stage01_core import cpu_setup
 admission=read(root/'admission.json')
 if admission.get('status')!='ADMITTED':raise RuntimeError('Formal execution requires ADMITTED preflight')
 for name in ('protocol','worklist','source_manifest'):
  if admission[name+'_sha256']!=sha(root/(name+'.json')):raise RuntimeError('Frozen admission input changed: '+name)
 for rel,h in read(root/'frozen_manifest.json')['files'].items():
  if sha(root/rel)!=h:raise RuntimeError('Frozen input changed: '+rel)
 for rel,h in read(root/'code_manifest.json')['files'].items():
  if sha(root/rel)!=h:raise RuntimeError('Code changed after freeze: '+rel)
 for rel,h in read(root/'source_manifest.json')['files'].items():
  if sha(root/rel)!=h:raise RuntimeError('Source changed: '+rel)
 lock=root/'formal_once.lock';lock.mkdir();write(lock/'entry.json',{'created':now(),'permanent':True,'no_resume':True})
 budget=Budget(root/'budget.json',limit=budget_limit(root))
 if not admission['admitted'] or admission['conservative']>budget.limit or admission['already_used']!=budget.seconds:raise RuntimeError('Stale or invalid admission')
 if admission['preflight_sha256']!=sha(root/'verification/preflight.json'):raise RuntimeError('Preflight changed after admission')
 for name,h in admission['preflight_checks'].items():
  if sha(root/('verification/'+name+'.json'))!=h:raise RuntimeError('Preflight check changed: '+name)
 if budget.formal_updates or budget.formal_optimizer_calls:raise RuntimeError('Uncertain or nonzero formal history; no resume')
 with measured(budget,'formal_runtime_setup'):
  cpu_setup(0)
 work=read(root/'worklist.json');pipelines=work['pipelines']
 assert len(pipelines)==90
 wall=time.monotonic();done=[];files={};initials={};current=None;session=None
 def save_json(rel,obj):write(root/rel,obj);files[rel]=sha(root/rel)
 def save_state(rel,obj,metadata):
  state=charged_snapshot(obj,budget);state['oof_identity']=metadata
  path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);torch.save(state,path);files[rel]=sha(path)
  return state
 def status(value,**extra):
  budget.persist();write(root/'status.json',{'status':value,'updated_at_utc':now(),'completed_pipelines':len(done),'formal_updates':budget.formal_updates,'current_pipeline':None if current is None else current['pipeline_id'],'budget_seconds':budget.seconds,'formal_wall_seconds':time.monotonic()-wall,'outer_labels_released':False,**extra})
 try:
  status('RUNNING')
  for current in pipelines:
   session=None
   pid=current['pipeline_id'];data=load_inner(root,current)
   prep=prepare_common(data['training'],data['query'],current['model_seed'],budget)
   model,scaler=prep['model'],prep['scaler'];h=prep['initial_mapper_hash'];seed=str(current['model_seed'])
   if seed in initials and initials[seed]!=h:raise RuntimeError('Same-seed fresh initialization changed')
   initials[seed]=h
   meta={**current,'input_identity':data['identity'],'scaler':scaler,'scaler_sha256':digest(scaler),'initial_mapper_hash':h,'protocol_sha256':sha(root/'protocol.json'),'source_manifest_sha256':sha(root/'source_manifest.json'),'code_manifest_sha256':sha(root/'code_manifest.json'),'query_labels_or_SE_present':False}
   save_json(f'inputs/{pid}.json',{'identity':meta,'training':data['training'],'query':data['query']})
   save_state(f'states/{pid}_INITIAL.pt',model,meta)
   session=common_session(model,'COMMON_MSE',pid+'_COMMON_MSE',budget)
   common_states={}
   for stage,steps,ds in [('COMMON_MSE',1000,(0,800,900,1000)),('COMMON_GP',300,(0,240,300))]:
    if stage=='COMMON_GP':
     frozen_before=freeze_features(model,prep['x_query'],budget)
     session=common_session(model,stage,pid+'_'+stage,budget)
    records=[common_diagnostic(session,budget)]
    log=root/f'updates/{pid}_{stage}.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as out:
     for _ in range(steps):
      row=session.step(budget);out.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
      if session.completed in ds:records.append(common_diagnostic(session,budget))
    files[str(log.relative_to(root))]=sha(log)
    save_json(f'diagnostics/{pid}_{stage}.json',records)
    state=save_state(f'states/{pid}_{stage}.pt',session,meta)
    if stage=='COMMON_GP' and state['mapper_hash']!=head['mapper_hash']:
     raise RuntimeError('COMMON_GP changed frozen mapper/head')
    common_states[stage]={'path':f'states/{pid}_{stage}.pt','sha256':files[f'states/{pid}_{stage}.pt'],'fingerprint':state['fingerprint']}
    if stage=='COMMON_MSE':
     head=head_predictions(model,prep['x_query'],scaler,budget)
     hp={**meta,'method':'HEAD','unit_id':pid+'_HEAD','candidate_ids':data['query']['ids'],'architecture_fingerprints':data['query']['architectures'],'train_candidate_ids':data['training']['ids'],'train_architecture_fingerprints':data['training']['architectures'],'mean_raw':head['v_mean_raw'],'mean_standardized':head['v_mean_standardized'],'train_mean_raw':head['u_mean_raw'],'train_mean_standardized':head['u_mean_standardized'],'model_variance_raw':None,'y_center':float(scaler['y_mean']),'y_scale':float(scaler['y_scale']),'actual_updates':1000,'common_mse_state':common_states[stage]}
   frozen=freeze_features(model,prep['x_query'],budget)
   cache_path=root/f'caches/{pid}.pt';cache_path.parent.mkdir(parents=True,exist_ok=True)
   torch.save({'frozen':frozen,'identity':meta},cache_path);files[str(cache_path.relative_to(root))]=sha(cache_path)
   hp['common_states']=common_states
   save_json(f'predictions_inner/{pid}_HEAD.json',hp)
   for method in current['group_order']:
    if method not in ('GP_CONST','GP_TREND'):raise RuntimeError('Readout outside frozen matrix')
    session=readout_session(method,frozen,budget)
    records=[readout_diagnostic(session,frozen,data['query']['ids'],scaler,0,budget)]
    log=root/f'updates/{pid}_{method}.jsonl'
    with log.open('x') as out:
     for _ in range(1500):
      row=session.step(budget);out.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
      if session.completed in (1000,1200,1300,1400,1500):records.append(readout_diagnostic(session,frozen,data['query']['ids'],scaler,session.completed,budget))
    files[str(log.relative_to(root))]=sha(log)
    save_json(f'diagnostics/{pid}_{method}.json',records)
    state=save_state(f'states/{pid}_{method}.pt',session,meta)
    last=records[-1];scale=float(scaler['y_scale']);extra=float(last['extra_noise_variance_standardized'])
    p={**meta,'unit_id':pid+'_'+method,'method':method,'candidate_ids':data['query']['ids'],'architecture_fingerprints':data['query']['architectures'],'mean_raw':last['query_mean_raw'],'mean_standardized':last['query_mean_standardized'],'latent_variance_raw':last['query_latent_variance_raw'],'latent_variance_standardized':last['query_latent_variance_standardized'],'latent_covariance_standardized':last['query_latent_covariance_standardized'],'latent_covariance_raw':last['query_latent_covariance_raw'],'negative_roundoff_variance_count':last['negative_roundoff_variance_count'],'extra_noise_variance_standardized':extra,'model_variance_raw':[scale*scale*(max(v,0.)+extra) for v in last['query_latent_variance_standardized']],'y_center':float(scaler['y_mean']),'y_scale':scale,'actual_updates':1500,'common_states':common_states,'frozen_mapper_hash':frozen['mapper_hash'],'endpoint_state_hash':last['state_hash'],'state_path':f'states/{pid}_{method}.pt','state_file_sha256':files[f'states/{pid}_{method}.pt'],'stability':stability_at(records),'parameter_names':session.model.parameter_names(),'gp_parameters':last['gp_parameters']}
    save_json(f'predictions_inner/{pid}_{method}.json',p)
   done.append(pid);status('RUNNING');print(json.dumps({'completed_pipelines':len(done),'formal_updates':budget.formal_updates,'model_seconds':budget.seconds,'pipeline':pid}),flush=True)
  if budget.formal_updates!=387000 or budget.formal_optimizer_calls!=387000:raise RuntimeError('Incomplete formal accounting')
  preds=[rel for rel in files if rel.startswith('predictions_inner/')]
  rows=sum(len(read(root/rel)['candidate_ids']) for rel in preds)
  if len(preds)!=270 or rows!=4500:raise RuntimeError('OOF cardinality mismatch')
  save_json('initialization_identity.json',{'same_seed_across_all_pipeline_inputs':True,'hashes_by_model_seed':initials})
  budget.persist()
  seal={'sealed':True,'sealed_at_utc':now(),'execution_status':'COMPLETE_VALID','completed_pipelines':90,'completed_pipeline_ids':done,'formal_updates':387000,'oof_rows':4500,'prediction_sets':270,'files':files,'protocol_sha256':sha(root/'protocol.json'),'source_manifest_sha256':sha(root/'source_manifest.json'),'code_manifest_sha256':sha(root/'code_manifest.json'),'worklist_sha256':sha(root/'worklist.json'),'inner_splits_manifest_sha256':sha(root/'inner_splits_manifest.json'),'resources':budget.as_dict(),'outer_labels_released':False}
  write(root/'oof_manifest.json',seal);status('OOF_COMPLETE_SEALED')
 except BaseException as e:
  classification='STOPPED_BUDGET_PARTIAL' if budget.seconds>=budget.limit else 'INVALID_EXECUTION'
  if isinstance(e,(KeyboardInterrupt,SystemExit)):classification='INTERRUPTED_UNVERIFIED'
  if session is not None:
   # Pure serialization of live tensors and optimizer; no forward/replay beyond limit.
   target=root/'partial_live_state.pt'
   torch.save({'pipeline':current,'stage':session.stage,'completed':session.completed,'gp':session.model.gp.state_dict(),'mapper':session.model.mapper.state_dict() if hasattr(session.model,'mapper') else None,'optimizer':session.optimizer.state_dict(),'last_update':session.last_update,'resources':budget.as_dict(),'resume_allowed':False},target)
  status(classification,failure=repr(e),traceback=traceback.format_exc())
  write(root/'partial_manifest.json',{'sealed':False,'execution_status':classification,'completed_pipeline_ids':done,'files':files,'resources':budget.as_dict()})
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=pathlib.Path);p.add_argument('--freeze',action='store_true');a=p.parse_args();freeze(a.root) if a.freeze else run(a.root)
