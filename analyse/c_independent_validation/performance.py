"""Pre-label protocol, A-only prediction seals, and isolated validation cache."""
import math,sys,subprocess,time
from .common import *
import numpy as np

SEEDS=list(range(2026100200,2026100210))
METHODS=['W-I','RAW16','Random']
REPLAY='results/structured_kernel_development_replay_20261002T235614Z'
FOLLOWUP='results/structured_kernel_followup_20261005T152816Z'

def set_stage(root,stage,state,**extra):
 s=read(root/'status.json');s[stage]=state;s['updated_at']=now();s.update(extra);write(root/'status.json',s)

def seal(path,x):
 if path.exists():
  previous=read(path)
  if 'sealed_at' in previous and 'sealed_at' in x:x={**x,'sealed_at':previous['sealed_at']}
  if previous!=x:raise ValueError('Frozen file differs: '+str(path))
 else:write(path,x)

def validate_public(p):
 if set(p)-{'ids','x_raw','wl','architecture_ids'}:raise ValueError('Public pool contains non-feature fields')
 n=len(p['ids'])
 if n!=len(set(p['ids'])) or np.asarray(p['x_raw']).shape!=(n,16) or len(p['wl'])!=n or len(p['architecture_ids'])!=n:raise ValueError('Public feature shape/ID mismatch')
 if not all(np.isfinite(p[k]).all() for k in ('x_raw','wl')):raise ValueError('Nonfinite public feature')

def plans(ids):
 ids=sorted(ids);initial={};random={}
 if len(ids)<=71:raise ValueError('P1 budget admission requires U > 71')
 for seed in SEEDS:
  selected=np.random.default_rng(np.random.SeedSequence([seed,0])).permutation(ids).tolist()[:50]
  remaining=sorted(set(ids)-set(selected))
  queries=np.random.default_rng(np.random.SeedSequence([seed,20261005,2])).permutation(remaining).tolist()[:21]
  assert len(set(selected+queries))==71
  initial[str(seed)]={'ids':selected};random[str(seed)]={'initial_ids':selected,'query_ids':queries}
 return initial,random

def code_identity(repo):
 dirs=['analyse/c_independent_validation','analyse/structured_kernel_phase1','analyse/structured_kernel_development_replay']
 files=[p for d in dirs for p in sorted((repo/d).glob('*.py'))]
 files += [repo/p for p in ['analyse/structured_kernel_followup/validation_train.py','analyse/structured_kernel_followup/common.py','eval_utils.py','dataset_utils.py','deterministic_runtime.py','hp_modes.py','nas_space.py','surrogate/accuracy_gp.py']]
 return {str(p.relative_to(repo)):sha(p) for p in files}

def verify_performance(root,repo):
 frozen=read(root/'performance_identity.json');verify(repo,frozen['code']);verify(root,frozen['inputs']);return frozen['identity']

def freeze_performance(root,repo):
 if (root/'performance_identity.json').exists():verify_performance(root,repo);return
 from analyse.structured_kernel_development_replay.protocol import protocol as oldprotocol
 from analyse.structured_kernel_phase1.data import A_LABEL_HASH,SOURCE,A_ID_HASH
 from analyse.c_independence_minimal_audit.identity import fingerprint
 from analyse.structured_kernel_phase1.features import build_operator_graph,graph_hash
 verify(root,read(root/'P1_complete.json')['files']);qualified=read(root/'C_qualified_manifest.json');configs=qualified['configurations'];U=len(configs)
 assert U>71 and U<=176 and all(c['architecture_fingerprint']!=EXCLUDED for c in configs)
 acceptance=read(root/'performance_acceptance.json');assert acceptance['state']=='PASS' and acceptance['new_code']=={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'analyse/c_independent_validation').glob('*.py'))}
 old=read(repo/REPLAY/'public_features.json');ext=read(repo/P4/'C_preparation/extended_A_features.json');cf=read(repo/P4/'C_preparation/C_features.json');fm=read(repo/P4/'C_preparation/extended_feature_manifest.json')
 dictionary=read(repo/P4/'C_preparation/frozen_dictionary.json');assert dictionary==fm['dictionaries'] and fm['append_only']
 A={k:old[k] for k in ('ids','x_raw','architecture_ids')};A['wl']=[ext[a] for a in A['architecture_ids']]
 lifted=np.zeros((len(A['ids']),fm['psi_dimension']))
 for m in fm['old_to_new_coordinates']:lifted[:,m['new']]=np.asarray(old['wl'])[:,m['old']]
 assert np.array_equal(lifted,np.asarray(A['wl'])) and len(A['ids'])==179 and digest(sorted(A['ids']))==A_ID_HASH
 C={'ids':[c['candidate_id'] for c in configs],'x_raw':[c['latent12']+c['HP4'] for c in configs],
  'wl':[cf[c['architecture_fingerprint']] for c in configs],'architecture_ids':[c['architecture_fingerprint'] for c in configs]}
 validate_public(A);validate_public(C);assert len(C['wl'][0])==len(A['wl'][0])==1522 and not set(C['ids'])&set(A['ids'])
 for c in configs:
  assert c['hp_mode']=='global4' and c['normalization']=='none' and c['gcnii_alpha']==.1 and c['gcnii_theta']==.5
  assert fingerprint(c)==c['architecture_fingerprint'] and graph_hash(build_operator_graph(c))==c['graph_hash']
  assert c['candidate_id']==digest({'architecture_fingerprint':c['architecture_fingerprint'],'hp_fingerprint':c['hp_fingerprint']})
 initial,random=plans(C['ids']);write(root/'public_A.json',A);write(root/'public_C.json',C)
 write(root/'initial_observation_manifest.json',initial);write(root/'random_query_manifest.json',random)
 base=oldprotocol();protocol={'version':'C-performance-frozen-1','methods':METHODS,'purpose':'independent_architecture_validation',
  'pool_configurations':U,'pool_architectures':U//4,'original_pool':180,'precautionary_removed':4,'coverage_71_over_U':71/U,
  'dictionary_exposure':'A and all original 45 C structures; transductive; append-only old coordinates unchanged',
  'gp':{k:base[k] for k in ('input','noise','optimizer','kernel','execution')},'acquisition':base['acquisition'],
  'A_only_prediction':'Fit all authoritative A179 separately for W-I and RAW16; each three starts; never refit after C labels',
  'C_search':'No A labels; fit only released mean/SE; no n71 fit; latent qLogEI 20 + terminal mean 1',
  'initial':{'n':50,'rng':'default_rng(SeedSequence([seed,0])).permutation(sorted(C_ids))[:50]'},
  'random_rng':'default_rng(SeedSequence([seed,20261005,2])).permutation(sorted(C_ids-initial))[:21]',
  'search_seeds':SEEDS,'budget':{'GNN_tasks':U*3,'GNN_seeds':[0,1,2],'GP_fits':422,'GP_starts':1266,'search_GP_fits':420,'trajectories':30,'releases':2130},
  'recipe':{'dataset':'Cora','split':'public','features':'NormalizeFeatures','global4':True,'normalization':'none','batch':'full_batch','epochs':300,'patience':80,
   'optimizer':'Adam','scheduler':'CosineAnnealingLR T_max=300 eta_min=lr*.01','activation':'relu','component_variant':'baseline','seed':[0,1,2],
   'entrypoint':'analyse.structured_kernel_followup.validation_train.train_validation_only','precision':'float32','first_strict_validation_improvement_checkpoint':True,'checkpoint_every_epochs':5,
   'device_rule':'One worker, select GPU with greatest free memory after at least 2048 MiB free; lock device for run; never terminate another process'},
  'data':{'dataset_content_fingerprint':'373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7','split_fingerprint':'bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09'},
  'target':'mean three best validation accuracies; SE=std(ddof=1)/sqrt(3); known variance=SE^2',
  'prediction_metrics':['MAE_pp','RMSE_pp','signed_bias_pp','latent_coverage90','latent_width90_pp','latent_IS90_pp','prospective_coverage90','prospective_width90_pp','prospective_IS90_pp'],
  'prospective_variance':'latent_raw+extra_raw+median_A(SE^2); freeze in P3; C-SE only retrospective auxiliary',
  'prediction_metric_CI':{'repetitions':2000,'rng':2026100601,'sampling':'architectures with all HP together'},
  'search_primary':'W-I minus Random n71 best observed validation; W-I minus RAW16 reference',
  'search_bootstrap':{'repetitions':2000,'rng':2026100531,'shared_paired_indices':True,'CI':[.025,.975],'scope':'fixed C pool, initialization variation only'},
  'return_rule':'max released mean at n71, exact ties smallest full candidate_id',
  'test_rule':'Seal 30 returns and all unique returned seed0/1/2 best checkpoint hashes first; at most90 evaluations; no whole-pool test or old B/test scores',
  'performance_gate':None,'no_method_changes_after_labels':True,'cost_boundary':'Full U*3 cache cost disclosed; replay is finite-pool query efficiency, not online end-to-end acceleration',
  'A_label_path':str(SOURCE/'private/evaluation_A.json'),'A_label_sha256':A_LABEL_HASH,'frozen_at':now()}
 write(root/'performance_protocol.json',protocol)
 tasks=[{'task_id':c['candidate_id']+'_s'+str(seed),'candidate_id':c['candidate_id'],'seed':seed,'configuration':c} for c in sorted(configs,key=lambda c:c['candidate_id']) for seed in [0,1,2]]
 write(root/'training_task_manifest.json',{'tasks':tasks,'count':len(tasks),'order':'candidate_id then seed','no_extra_benchmark':True})
 accepted=repo/FOLLOWUP/'code/analyse/structured_kernel_followup/validation_train.py'
 assert accepted.is_file() and sha(accepted)==sha(repo/'analyse/structured_kernel_followup/validation_train.py'),'Accepted training implementation changed'
 write(root/'P2_interface_checks.json',{'A179':True,'C_configurations':U,'architecture_count':U//4,'A_coordinate_lift_exact':True,'WL_dimension':1522,
  'C_graph_and_config_IDs_recomputed':True,'training_code_equals_accepted_version':True,'accepted_training_code_sha256':sha(accepted),
  'A_labels_not_yet_loaded':True,'C_labels_never_read':True,'test_never_scored':True})
 inputs={p:sha(root/p) for p in ['performance_protocol.json','C_qualified_manifest.json','public_A.json','public_C.json','initial_observation_manifest.json','random_query_manifest.json','training_task_manifest.json']}
 identity={'code':code_identity(repo),'inputs':inputs,'P1_identity':sha(root/'P1_complete.json'),'time':now()};identity['identity']=digest(identity);write(root/'performance_identity.json',identity)
 set_stage(root,'P2','PROTOCOL_FROZEN');print(json.dumps({'P2':'PROTOCOL_FROZEN','U':U,'GNN_tasks':len(tasks),'coverage':71/U}),flush=True)

def seal_predictions(root,repo):
 identity=verify_performance(root,repo)
 if (root/'P3_complete.json').exists():verify(root,read(root/'P3_complete.json')['files']);return
 from analyse.structured_kernel_phase1.data import read_A_labels
 from analyse.structured_kernel_phase1.gp import cpu_setup,predict_frozen
 from analyse.structured_kernel_development_replay.learner import fit_current
 cpu_setup();set_stage(root,'P3','RUNNING');A=read(root/'public_A.json');C=read(root/'public_C.json');labels=read_A_labels(repo)
 combined={k:A[k]+C[k] for k in A};observations=[{'candidate_id':cid,**labels[cid]} for cid in A['ids']];rows=[];models={}
 for method in ['W-I','RAW16']:
  folder=root/'A_only_models'/method;fit,q,query=fit_current(folder,combined,observations,0,method,identity)
  p=predict_frozen(folder/fit['selected_model'],q);s=fit['scaler'];scale=s['y_scale'];means=s['y_mean']+scale*np.array(p['mean_standardized']);var=scale**2*np.array(p['latent_variance_standardized']);extra=scale**2*p['extra_noise_variance_standardized'];known=float(np.median([o['se']**2 for o in observations]))
  if np.any(var<0):raise ValueError('Negative frozen latent variance')
  for i,cid in enumerate(query['ids']):rows.append({'method':method,'candidate_id':cid,'mean':float(means[i]),'latent_variance':float(var[i]),'extra_noise_variance':extra,'A_median_known_variance':known,'prospective_variance':float(var[i]+extra+known)})
  models[method]={'path':str(folder/fit['selected_model']),'sha256':fit['selected_model_sha256'],'fit_identity':fit['identity'],'starts':3,'A_observations':179}
  print(json.dumps({'P3_model_complete':method,'new_formal_fit_budget':1}),flush=True)
 seal(root/'sealed_predictions.json',{'models':models,'predictions':rows,'protocol_sha256':sha(root/'performance_protocol.json'),'performance_identity':identity,'C_labels_read':False,'sealed_at':now()})
 names=['sealed_predictions.json']+[str(p.relative_to(root)) for p in (root/'A_only_models').rglob('*') if p.is_file()]
 write(root/'P3_complete.json',{'files':{n:sha(root/n) for n in names},'fits':2,'starts':6,'C_score_or_SE_access':0,'time':now()});set_stage(root,'P3','COMPLETE')

def dataset(root,repo,device):
 from dataset_utils import resolve_dataset_request,load_dataset_from_request,dataset_context_from_manifest
 bundle=load_dataset_from_request(resolve_dataset_request('cora',None,cora_root=str(repo/'Cora')))
 context=dataset_context_from_manifest(bundle.manifest);expected=read(root/'performance_protocol.json')['data']
 assert all(context[k]==v for k,v in expected.items()),'Dataset/split changed'
 return bundle,bundle.data.to(device),context

def select_device(root):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 available=[tuple(map(int,line.split(','))) for line in raw.strip().splitlines()];eligible=[(free,index) for index,free in available if free>=2048]
 if (root/'resource_lock.json').exists():
  device=read(root/'resource_lock.json')['device'];index=int(device.split(':')[1])
  return device if any(i==index and free>=2048 for i,free in available) else None
 if not eligible:
  write(root/'resource_wait.json',{'state':'WAITING_RESOURCE','minimum_free_MiB':2048,'inventory':raw,'time':now()});return None
 _,index=max(eligible);value={'device':'cuda:'+str(index),'inventory':raw,'time':now(),'single_worker':True,'other_processes_untouched':True};write(root/'resource_lock.json',value);return value['device']

def train_one(root,repo,task_id):
 identity=verify_performance(root,repo);verify(root,read(root/'P3_complete.json')['files'])
 import torch
 from analyse.structured_kernel_followup.validation_train import train_validation_only
 task=next(t for t in read(root/'training_task_manifest.json')['tasks'] if t['task_id']==task_id);device=torch.device(read(root/'resource_lock.json')['device']);folder=root/'training'/task_id
 bundle,data,context=dataset(root,repo,device)
 tid=digest({'performance_identity':identity,'task':task,'data':context,'device':str(device)})
 result=train_validation_only(task['configuration'],data,bundle.num_features,bundle.num_classes,task['seed'],device,folder,tid)
 write(folder/'task_binding.json',{'task_id':task_id,'candidate_id':task['candidate_id'],'seed':task['seed'],'identity':tid,'dataset_context':context,
  'Cora_full_label_bytes_loaded':True,'training_view_has_only_train_validation_labels':True,'test_metric_computations':0,'source_result_sha256':sha(folder/'result.json')})
 print(json.dumps({'task_id':task_id,'state':'COMPLETE','seconds':result['elapsed_seconds'],'peak_GPU_bytes':result['peak_gpu_memory_allocated_bytes']}),flush=True)

def build_cache(root,repo):
 identity=verify_performance(root,repo)
 if (root/'P4_complete.json').exists():verify(root,read(root/'P4_complete.json')['files']);return
 verify(root,read(root/'P3_complete.json')['files']);set_stage(root,'P4','RUNNING');tasks=read(root/'training_task_manifest.json')['tasks'];started=time.monotonic()
 while select_device(root) is None:
  set_stage(root,'P4','WAITING_RESOURCE');print(json.dumps({'P4':'WAITING_RESOURCE','time':now()}),flush=True);time.sleep(30)
 set_stage(root,'P4','RUNNING');completed=0;durations=[];files={}
 for task in tasks:
  folder=root/'training'/task['task_id'];result=folder/'result.json';binding=folder/'task_binding.json'
  if not (result.exists() and binding.exists()):
   while select_device(root) is None:
    set_stage(root,'P4','WAITING_RESOURCE');print(json.dumps({'P4':'WAITING_RESOURCE','task_id':task['task_id'],'time':now()}),flush=True);time.sleep(30)
   set_stage(root,'P4','RUNNING')
   log(root/'training_attempts.jsonl',{'task_id':task['task_id'],'resume_checkpoint_present':(folder/'resume.pt').exists(),'time':now()})
   cmd=[sys.executable,'-B','-m','analyse.c_independent_validation','_train_one','--run-dir',str(root),'--task-id',task['task_id']]
   p=subprocess.run(cmd,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   (root/'training_worker_logs').mkdir(exist_ok=True);(root/'training_worker_logs'/f'{task["task_id"]}.log').write_text(p.stdout)
   if p.returncode:
    set_stage(root,'P4','TECHNICALLY_INCOMPLETE',failed_task=task['task_id']);raise RuntimeError('Training task failed; exact checkpoint retained: '+task['task_id'])
  value=read(result);b=read(binding);assert value['identity']==b['identity'] and b['candidate_id']==task['candidate_id'] and b['seed']==task['seed']
  assert value['identity']==digest({'performance_identity':identity,'task':task,'data':b['dataset_context'],'device':read(root/'resource_lock.json')['device']})
  assert b['source_result_sha256']==sha(result);verify(folder,value['checkpoint_hashes']);completed+=1;durations.append(value['elapsed_seconds'])
  files[str(result.relative_to(root))]=sha(result);files[str(binding.relative_to(root))]=sha(binding)
  for name,h in value['checkpoint_hashes'].items():files[str((folder/name).relative_to(root))]=h
  write(root/'P4_progress.json',{'state':'RUNNING','completed_tasks':completed,'total_tasks':len(tasks),'last_task':task['task_id'],
   'actual_task_seconds':sum(durations),'estimated_remaining_seconds':float(np.mean(durations))*(len(tasks)-completed),'time':now()})
  print(json.dumps({'P4_completed':completed,'total':len(tasks),'ETA_seconds':int(np.mean(durations)*(len(tasks)-completed))}),flush=True)
 # Trusted cache builder sees the full validation cache. Search runs in a
 # separate process and can only request released rows from the oracle.
 byid={}
 for task in tasks:
  value=read(root/'training'/task['task_id']/'result.json');byid.setdefault(task['candidate_id'],[]).append(value['best_validation_accuracy'])
 cache={cid:{'mean':float(np.mean(v)),'se':float(np.std(v,ddof=1)/math.sqrt(3)),'seeds':[0,1,2],'validation_values':v} for cid,v in byid.items()}
 assert all(len(v)==3 for v in byid.values())
 write(root/'private/C_validation_cache.json',cache);files['private/C_validation_cache.json']=sha(root/'private/C_validation_cache.json')
 write(root/'P4_complete.json',{'files':files,'tasks':len(tasks),'configurations':len(cache),'task_seconds':sum(durations),'wall_seconds':time.monotonic()-started,
  'test_evaluations':0,'prediction_seal_sha256':sha(root/'sealed_predictions.json'),'time':now()});set_stage(root,'P4','COMPLETE')
