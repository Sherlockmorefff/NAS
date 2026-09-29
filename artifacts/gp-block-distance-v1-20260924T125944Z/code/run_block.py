"""One-shot serial block-GP execution; immutable protocol, one ledger, no resume."""
import argparse,csv,datetime,hashlib,json,pathlib,time,traceback
from block_core import (Budget,budget_limit,cpu_setup,measured,prepare_inputs,create_model,session,
 charged_snapshot,diagnose,distance_diagnostics,as_fit_start,stability_at,digest,METHODS)
from block_data import load_inputs
import torch

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def verify(root,manifest):
 for rel,h in manifest['files'].items():
  if sha(root/rel)!=h:raise RuntimeError('Frozen file changed: '+rel)
def freeze(root):
 if (root/'formal_once.lock').exists() or (root/'frozen_manifest.json').exists():raise RuntimeError('Already frozen/executed; no resume')
 b=Budget(root/'budget.json',limit=budget_limit(root));p=read(root/'verification/preflight.json')
 assert p['pass'] and b.formal_updates==0
 assert read(root/'verification/scoring_synthetic.json')['pass']
 assert read(root/'verification/data_synthetic.json')['pass']
 assert read(root/'audit/block_data_review.json')['pass']
 files={str(p.relative_to(root)):sha(p) for p in sorted((root/'code').rglob('*.py'))}
 files.update({str(p.relative_to(root)):sha(p) for p in sorted((root/'audit').glob('*.py'))})
 write(root/'code_manifest.json',{'files':files,'frozen_at_utc':now()})
 used=b.seconds;fit=p['fit_est'];setup=p['setup_est'];diag=p['diagnostic_est'];conservative=used+1.5*(fit+setup)+max(20,2*diag)
 admitted=conservative<=600
 write(root/'admission.json',{'admitted':admitted,'already_used':used,'fit_est':fit,'setup_est':setup,'diagnostic_est':diag,'conservative':conservative,'limit':600,'synthetic_optimizer_calls':b.synthetic_optimizer_calls,'timing_cells':p['timing_cells'],'preflight_sha256':sha(root/'verification/preflight.json'),'formula':'used+1.5*(fit+setup)+max(20,2*diagnostic)','no_retiming_or_matrix_shrink':True,'created_at_utc':now()})
 if not admitted:
  write(root/'status.json',{'status':'BLOCKED_BUDGET_ADMISSION','completed_fits':0,'formal_updates':0,'resources':b.as_dict(),'outer_labels_released':False});print(json.dumps(read(root/'admission.json')));return
 frozenfiles={}
 for name in ['PROMPT.md','protocol.json','source_manifest.json','code_manifest.json','data_manifest.json','masks_manifest.json','historical_refs.json','worklist.json','environment.json','admission.json','verification/preflight.json','verification/scoring_synthetic.json','verification/data_synthetic.json','audit/block_data_review.json']:
  frozenfiles[name]=sha(root/name)
 for name in ['code_manifest.json','source_manifest.json','data_manifest.json','masks_manifest.json']:
  m=read(root/name);verify(root,m);frozenfiles.update(m['files'])
 write(root/'frozen_manifest.json',{'files':frozenfiles,'frozen_at_utc':now(),'formal_updates':0,'outer_labels_released':False,'automatic_resume':False})
 write(root/'status.json',{'status':'ADMITTED_FROZEN','completed_fits':0,'formal_updates':0,'outer_labels_released':False,'resources':b.as_dict()})
 print(json.dumps(read(root/'admission.json')))

def run(root):
 start=time.monotonic();b=Budget(root/'budget.json',limit=budget_limit(root));assert b.formal_updates==0 and b.formal_optimizer_calls==0
 admission=read(root/'admission.json');assert admission['admitted'] and admission['limit']==b.limit==600 and admission['already_used']==b.seconds and admission['conservative']<=b.limit
 verify(root,read(root/'frozen_manifest.json'))
 lock=root/'formal_once.lock';lock.mkdir();write(lock/'entry.json',{'created_at_utc':now(),'permanent':True,'automatic_resume':False})
 files={};records=[];done=[];current=None;active=None;model=None;flat=[];state_index=[];diagnostic_index=[]
 def save(rel,value):write(root/rel,value);files[rel]=sha(root/rel)
 def status(value,**extra):
  b.persist();write(root/'status.json',{'status':value,'completed_fits':len(done),'completed_fit_ids':done,'formal_updates':b.formal_updates,'resources':b.as_dict(),'formal_wall_seconds':time.monotonic()-start,'outer_labels_released':False,'updated_at_utc':now(),**extra})
 def savestate(rel,obj,meta):
  state=charged_snapshot(obj,b);state['identity']=meta;p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);torch.save(state,p);files[rel]=sha(p)
  entry={'path':rel,'sha256':files[rel],'unit_id':meta['unit_id'],'method':meta['method'],'total_step':state['total_steps'],'fingerprint':state['fingerprint'],'parameter_spec':state['parameter_spec'],'scheduler':state.get('scheduler'),'optimizer_state_entries':len(state.get('optimizer',{}).get('state',{}))};state_index.append(entry);return entry
 try:
  with measured(b,'runtime_setup'):cpu_setup(0)
  work=read(root/'worklist.json');assert len(work['fits'])==30
  for current in work['fits']:
   active=None;model=None
   verify(root,read(root/'frozen_manifest.json'))
   uid=current['unit_id'];data=load_inputs(root,current);prepared=prepare_inputs(data['training'],data['query'],data['scaler'],b)
   model=create_model(current['method'],prepared,b)
   meta={**current,'input_identity':data['identity'],'training_ids_sha256':digest(data['training']['ids']),'query_ids_sha256':digest(data['query']['ids']),'scaler':data['scaler'],'scaler_sha256':digest(data['scaler']),'protocol_sha256':sha(root/'protocol.json'),'source_manifest_sha256':sha(root/'source_manifest.json'),'code_manifest_sha256':sha(root/'code_manifest.json'),'frozen_manifest_sha256':sha(root/'frozen_manifest.json'),'query_labels_or_SE_present':False,'is_historical':False}
   initial=savestate('states/'+uid+'_INITIAL.pt',model,meta)
   distances=[distance_diagnostics(model,prepared,b)]
   active=session(model,'INIT_GP',b);diags=[diagnose(active,prepared,b)]
   stage_states={};stages=[]
   for stage,nsteps,points in [('INIT_GP',300,(240,300)),('FIT_GP',1500,(1000,1200,1300,1400,1500))]:
    if stage=='FIT_GP':
     active=session(model,'FIT_GP',b)
     if active.optimizer.state:raise RuntimeError('Second stage optimizer state not empty')
     # Same computed state, separate stage index; no additional numerical call.
     alias=as_fit_start(diags[-1]);stages.append({'stage':'FIT_GP','local_step':0,'total_step':300,'diagnostic_index':2,'alias_of':'INIT_GP:300'})
    logrel='updates/'+uid+'_'+stage+'.jsonl';log=root/logrel;log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as out:
     for _ in range(nsteps):
      row=active.step();out.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
      if active.completed in points:diags.append(diagnose(active,prepared,b))
      if active.completed%100==0:b.persist()
    files[logrel]=sha(log)
    stage_states[stage]=savestate('states/'+uid+'_'+stage+'.pt',active,meta)
   assert len(diags)==8 and model.total_steps==1800
   stages=[{'stage':d['stage'],'local_step':d['local_step'],'total_step':d['total_step'],'diagnostic_index':i} for i,d in enumerate(diags)]+stages
   stages.sort(key=lambda d:(d['total_step'],d['stage']=='FIT_GP'))
   diagnostic_index.append({'unit_id':uid,'path':'diagnostics/'+uid+'.json','distinct_states':8,'stage_indices':stages})
   save('diagnostics/'+uid+'.json',{'unit_id':uid,'records':diags,'stage_indices':stages,'alias_INIT300_to_FIT0':True})
   distances.append(distance_diagnostics(model,prepared,b));save('distances/'+uid+'.json',distances)
   # These are statistics on saved arrays; no forward/gradient/replay.
   stable=stability_at(diags+[alias]);last=diags[-1]
   pred={**meta,'candidate_ids':data['query']['ids'],'architecture_fingerprints':data['query']['architectures'],'train_candidate_ids':data['training']['ids'],'train_architecture_fingerprints':data['training']['architectures'],'y_center':data['scaler']['y_mean'],'y_scale':data['scaler']['y_scale'],'actual_updates':1800,'total_steps':1800,'negative_mll':last['nll_after_update'],'endpoint_nll_after_update':last['nll_after_update'],'mean_raw':last['query_mean_raw'],'mean_standardized':last['query_mean_standardized'],'latent_variance_raw':last['query_latent_variance_raw'],'latent_variance_standardized':last['query_latent_variance_standardized'],'latent_covariance_raw':last['query_latent_covariance_raw'],'latent_covariance_standardized':last['query_latent_covariance_standardized'],'model_variance_raw':last['model_variance_raw'],'extra_noise_variance_standardized':last['extra_noise_variance_standardized'],'negative_roundoff_variance_count':last['negative_roundoff_variance_count'],'latent_covariance_symmetry_max_abs':last['latent_covariance_symmetry_max_abs'],'latent_covariance_min_eigenvalue':last['latent_covariance_min_eigenvalue'],'gp_parameters':last['gp_parameters'],'parameter_spec':stage_states['FIT_GP']['parameter_spec'],'parameter_names':[p['name'] for p in stage_states['FIT_GP']['parameter_spec']],'endpoint_state_hash':last['state_hash'],'state_path':stage_states['FIT_GP']['path'],'state_file_sha256':stage_states['FIT_GP']['sha256'],'initial_state':initial,'stage_states':stage_states,'stability':stable,'ranking_ids':last['ranking_ids'],'top5_ids':last['top5_ids'],'query_exact_mean_tie_groups':last['query_exact_mean_tie_groups'],'query_exact_mean_tie_candidates':last['query_exact_mean_tie_candidates'],'top5_boundary_exact_tie':last['top5_boundary_exact_tie'],'top5_boundary_gap_raw':last['top5_boundary_gap_raw']}
   if pred['endpoint_state_hash']!=stage_states['FIT_GP']['fingerprint']:raise RuntimeError('Endpoint diagnostic/state differs')
   rel='predictions/'+uid+'.json';save(rel,pred)
   records.append({k:pred[k] for k in ('unit_id','method','subset_id','fold','sample_seed','model_seed','is_historical')}|{'path':rel,'sha256':files[rel]})
   for i,cid in enumerate(pred['candidate_ids']):
    flat.append({'unit_id':uid,'subset_id':current['subset_id'],'method':current['method'],'candidate_id':cid,'architecture_fingerprint':pred['architecture_fingerprints'][i],'mean_raw':pred['mean_raw'][i],'latent_variance_standardized':pred['latent_variance_standardized'][i],'extra_noise_variance_standardized':pred['extra_noise_variance_standardized'],'model_variance_raw':pred['model_variance_raw'][i]})
   done.append(uid);status('RUNNING');print(json.dumps({'completed_fits':len(done),'formal_updates':b.formal_updates,'model_seconds':b.seconds,'unit_id':uid}),flush=True)
  if b.formal_updates!=54000 or b.formal_optimizer_calls!=54000 or len(flat)!=1074:raise RuntimeError('Full matrix accounting mismatch')
  assert b.as_dict()['formal_steps_by_stage']=={'INIT_GP':9000,'FIT_GP':45000}
  assert all(v==18000 for v in b.as_dict()['formal_steps_by_method'].values())
  save('state_index.json',{'records':state_index,'states':90});save('diagnostic_index.json',{'records':diagnostic_index,'distinct_states':240,'stage_indices':270})
  rel='new_candidate_predictions.csv'
  with (root/rel).open('w',newline='') as out:
   w=csv.DictWriter(out,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
  files[rel]=sha(root/rel)
  for ref in read(root/'historical_refs.json')['records']:
   assert sha(root/ref['path'])==ref['sha256'];records.append(ref);files[ref['path']]=ref['sha256']
  masks=read(root/'masks_manifest.json');verify(root,masks);files.update(masks['files'])
  for f in ['protocol.json','source_manifest.json','worklist.json','masks_manifest.json','historical_refs.json','data_manifest.json']:
   files[f]=sha(root/f)
  verify(root,read(root/'frozen_manifest.json'));b.persist()
  seal={'sealed':True,'sealed_at_utc':now(),'execution_status':'COMPLETE_VALID','new_endpoint_count':30,'historical_reference_count':90,'raw_unit_count':120,'mask_count':30,'new_candidate_rows':1074,'formal_updates':54000,'state_count':90,'unique_diagnostic_count':240,'files':files,'records':records,'resources':b.as_dict(),'outer_labels_released':False,**{k+'_sha256':sha(root/(k+'.json')) for k in ('protocol','source_manifest','code_manifest','worklist','masks_manifest','frozen_manifest','data_manifest')}}
  write(root/'predictions_manifest.json',seal);status('COMPLETE_VALID',sealed_at_utc=seal['sealed_at_utc'])
 except BaseException as e:
  classification='STOPPED_BUDGET_PARTIAL' if b.seconds>=b.limit else 'FAILED_FORMAL'
  if isinstance(e,(KeyboardInterrupt,SystemExit)):classification='INTERRUPTED_UNVERIFIED'
  if active is not None:
   torch.save({'unit':current,'stage':active.stage,'completed':active.completed,'gp':active.model.gp.state_dict(),'optimizer':active.optimizer.state_dict(),'last_update':active.last_update,'resources':b.as_dict(),'resume_allowed':False},root/'partial_live_state.pt')
  status(classification,failure=repr(e),traceback=traceback.format_exc());write(root/'partial_manifest.json',{'sealed':False,'execution_status':classification,'completed_fit_ids':done,'files':files,'resources':b.as_dict()});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--freeze',action='store_true');a=p.parse_args();freeze(a.root) if a.freeze else run(a.root)
