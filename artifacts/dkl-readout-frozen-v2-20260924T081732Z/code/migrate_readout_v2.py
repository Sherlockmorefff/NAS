#!/usr/bin/env python3
"""One-shot, byte-preserving parent migration; no models or labels decoded."""
import argparse,copy,datetime,hashlib,json,platform,shutil,time,subprocess
from pathlib import Path
PARENT=Path('/home/fangxy/D-VAE-worktrees/dkl-readout-frozen-v1-20260924T072004Z/artifacts/dkl-readout-frozen-v1-20260924T072004Z')
RUN='dkl-readout-frozen-v2-20260924T081732Z'
PARENT_PROMPT_SHA='0dd3fee1ff9aa33b5e4187bf7312ff1378a428c53c85417ada5da868c91fb063'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def import_budget(root,parent):
 if (root/'budget.json').exists() or (root/'budget_carryover.json').exists():raise RuntimeError('Budget import may occur exactly once into a new run')
 old=read(parent/'budget.json');limit=read(root/'protocol.json')['budget']['model_seconds']
 if old['limit']!=1200 or limit!=1500:raise ValueError('Only explicit V1-to-V2 resource amendment allowed')
 if old['updates']!=old['formal_updates'] or old['verification_updates']!=old['synthetic_updates']:raise ValueError('Parent aliases inconsistent')
 if old['formal_updates'] or old['formal_optimizer_calls'] or old['training_seconds'] or old['prediction_seconds']:raise ValueError('Parent contains formal model work')
 if old['verification_seconds']!=old['seconds'] or old['seconds']!=3.2575200367718935 or old['synthetic_updates']!=182 or old['synthetic_optimizer_calls']!=182:raise ValueError('Parent cost/count identity differs')
 if old['automatic_resume'] or old['synthetic_optimizer_limit']!=300:raise ValueError('Parent policy differs')
 new=copy.deepcopy(old);new['limit']=float(limit)
 write(root/'budget.json',new)
 write(root/'budget_carryover.json',{'parent_run_id':parent.name,'parent_budget_path':'parent/budget.json','parent_budget_sha256':sha(parent/'budget.json'),'inherited_model_seconds':old['seconds'],'inherited_synthetic_updates':old['synthetic_updates'],'inherited_synthetic_optimizer_calls':old['synthetic_optimizer_calls'],'parent_ledger':old,'imported_initial_ledger':new,'imported_initial_budget_sha256':sha(root/'budget.json'),'changed_fields':{'limit':{'old':1200.,'new':1500.}},'import_rule':'copy each verified parent field exactly once; only resource limit changes; add no second copy of past cost','remaining_synthetic_calls':118,'automatic_resume':False,'imported_at_utc':now()})
 return new

def main(root,parent):
 start=time.monotonic()
 if (root/'protocol.json').exists() or (root/'migration_manifest.json').exists():raise RuntimeError('Migration is one-shot and cannot overwrite evidence')
 (root/'migration_once.lock').mkdir()
 inventory={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (parent/'FILES.sha256').read_text().splitlines()}
 for f,h in inventory.items():
  if sha(parent/f)!=h:raise RuntimeError('BLOCKED_PARENT_SOURCE: '+f)
 archive=sha(parent/'evidence.tar.gz')
 if archive!=(parent/'evidence.tar.gz.sha256').read_text().split()[0]:raise RuntimeError('Parent package hash differs')
 if sha(parent/'PROMPT.md')!=PARENT_PROMPT_SHA:raise RuntimeError('Parent prompt identity differs')
 if read(parent/'status.json')['status']!='BLOCKED_BUDGET_ADMISSION':raise RuntimeError('Parent status differs')
 for f in ['formal_run_once.lock','predictions_manifest.json','predictions','states','scoring/labels_A.json','scoring/release.json','scoring/summary.json']:
  if (parent/f).exists():raise RuntimeError('Unexpected parent formal/scoring activity: '+f)
 if not all(read(parent/f)['pass'] for f in ['verification/preflight.json','verification/source_cache_checks.json','verification/final_identity_check/acceptance.json','audit/readout_accounting.json']):raise RuntimeError('Parent verification incomplete')
 oldsource=read(parent/'source_manifest.json');oldcache=read(parent/'cache_manifest.json');oldprotocol=read(parent/'protocol.json')
 if oldsource['source_package_sha256']!='39d9fc6474e0d9da6f0a456452a25f6b04f51c01442f6c4cf1823ba6ca0416ce':raise RuntimeError('Residual root identity differs')
 migrated={};origins={}
 def cp(src,dst=None):
  dst=dst or src
  if src not in inventory or sha(parent/src)!=inventory[src]:raise RuntimeError('Unverified copied parent path: '+src)
  target=root/dst;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/src,target)
  migrated[dst]=sha(target);origins[dst]=str(parent/src)
 for f in oldsource['files']:cp(f)
 for f in oldcache['files']:cp(f)
 cp('worklist.json');cp('PROMPT.md','PROMPT_parent_v1.md')
 for f in ['PROMPT.md','README.md','REPRODUCE.md','protocol.json','worklist.json','budget.json','admission.json','status.json','source_manifest.json','cache_manifest.json','code_manifest.json','blocked_manifest.json','environment_frozen.json','audit/readout_accounting.json','audit/workspace_preservation.json']:
  cp(f,'parent/'+f)
 for f in sorted(inventory):
  if f.startswith('verification/'):cp(f,'validation_parent/'+f[len('verification/'):])
  if f.startswith('code/') and f.endswith('.py'):cp(f,'parent/'+f)
 for f in ['FILES.sha256','evidence.tar.gz.sha256']:
  shutil.copyfile(parent/f,root/'parent'/f);migrated['parent/'+f]=sha(root/'parent'/f);origins['parent/'+f]=str(parent/f)
 protocol=copy.deepcopy(oldprotocol);protocol.update(run_id=RUN,version='frozen-readout-v2-25min',normative_prompt_path='PROMPT.md',normative_prompt_sha256=sha(root/'PROMPT.md'),parent_run_id=parent.name,parent_root=str(parent),parent_package_sha256=archive,parent_protocol_sha256=sha(parent/'protocol.json'),parent_prompt_sha256=PARENT_PROMPT_SHA,protocol_amendment_path='protocol_amendment.json')
 protocol['budget']['model_seconds']=1500;protocol['budget']['inherited_readout_model_seconds']=3.2575200367718935;protocol['budget']['inherited_synthetic_calls']=182
 protocol['scoring']['release']='V2: COMPLETE_VALID 180 GP + 60 HEAD + 270000 updates uniform seal only; isolated scorer; legal A only'
 write(root/'protocol.json',protocol)
 write(root/'protocol_amendment.json',{'parent_run_id':parent.name,'parent_protocol_sha256':sha(parent/'protocol.json'),'supplement_prompt_sha256':sha(root/'PROMPT.md'),'science_configuration_changes':[],'changes':[{'field':'budget.model_seconds','old':1200,'new':1500,'reason':'Explicit V2 25-minute supplement'},{'field':'run identity and artifact output paths','old':parent.name,'new':RUN,'reason':'Separate immutable lineage; V1 remains blocked'},{'field':'verification execution','old':'run full checks and precompute','new':'reuse byte-verified prior evidence/cache; targeted migration checks only','reason':'Do not spend duplicate validation budget'},{'field':'budget import','old':'new zero ledger','new':'inherit 3.2575200367718935 seconds and182 synthetic calls exactly once','reason':'Cumulative readout cost retained'},{'field':'scoring release gate','old':'full or partial sealed run','new':'full240/270000 only','reason':'V2 section7 explicit restriction'}],'no_change_to':['model equations','finite beta prior','training data/scalers/order/seeds','1500 updates and1000/500 LR schedule','diagnostic step1200','three stability windows and thresholds','scoring formulas and7flags'],'formal_resume':False})
 source=copy.deepcopy(oldsource);source.update(parent_run_id=parent.name,parent_root=str(parent),parent_package_sha256=archive,parent_source_manifest_sha256=sha(parent/'source_manifest.json'),copied_from_parent_origins={f:str(parent/f) for f in oldsource['files']},labels_copied=False)
 write(root/'source_manifest.json',source)
 cache=copy.deepcopy(oldcache);cache.update(protocol_sha256=sha(root/'protocol.json'),source_manifest_sha256=sha(root/'source_manifest.json'),worklist_sha256=sha(root/'worklist.json'),parent_cache_manifest_path='parent/cache_manifest.json',parent_cache_manifest_sha256=sha(parent/'cache_manifest.json'),parent_validation_path='validation_parent/source_cache_checks.json',parent_run_id=parent.name,reused_payload_bytes_unchanged=True,v2_bound_at_utc=now(),frozen_at_utc=oldcache['frozen_at_utc'])
 write(root/'cache_manifest.json',cache)
 reused=[]
 for path in migrated:
  if path.startswith('validation_parent/'):
   original='verification/'+path[len('validation_parent/'):]
   reused.append({'original_run_id':parent.name,'original_path':str(parent/original),'v2_path':path,'sha256':migrated[path],'coverage':'Original unchanged report; exact report contents determine checked cases','applicability':'Identical scientific model/scoring definitions, identical tensor bytes and runtime versions; only budget/identity/release-gate migration differs','rerun_in_v2':False})
 write(root/'validation_reuse.json',{'parent_run_id':parent.name,'reports':reused,'parent_code_manifest_sha256':sha(parent/'code_manifest.json'),'reports_rewritten':False,'v2_new_validation_records':['verification/v2_migration_checks.json','audit/migration_invariance.json'],'no_new_full_preflight_or_reencoding':True,'scientific_source_diff_review':'audit/migration_science_review.md'})
 import_budget(root,parent)
 import torch,gpytorch,numpy
 oldenv=read(parent/'environment_frozen.json');env={'python':platform.python_version(),'torch':torch.__version__,'gpytorch':gpytorch.__version__,'numpy':numpy.__version__,'platform':platform.platform(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent.parent,text=True).strip(),'device_used':'cpu','dtype_used':'float64','threads_used':1}
 changes={k:[oldenv[k],env[k]] for k in env if k in oldenv and oldenv[k]!=env[k]}
 if changes:raise RuntimeError('BLOCKED_ENVIRONMENT_CHANGED: '+str(changes))
 write(root/'environment_frozen.json',{**env,'parent_environment_sha256':sha(parent/'environment_frozen.json'),'parent_runtime_compatible':True,'comparison_differences':changes,'actual_worktree':str(root.parent.parent)})
 write(root/'migration_manifest.json',{'files':migrated,'origins':origins,'parent_run_id':parent.name,'parent_root':str(parent),'parent_files_manifest_sha256':sha(parent/'FILES.sha256'),'parent_archive_sha256':archive,'parent_inventory_verified_files':len(inventory),'parent_worklist_exact_bytes':sha(root/'worklist.json')==sha(parent/'worklist.json'),'migration_wall_seconds':time.monotonic()-start,'no_model_calls':True,'model_budget_imported_once':True,'labels_read':False,'new_run_id':RUN,'started_at_utc':now()})
 write(root/'status.json',{'status':'MIGRATED_AWAITING_TARGETED_VALIDATION','at_utc':now()})
 print(json.dumps({'status':'MIGRATED','parent_files_verified':len(inventory),'copied_files':len(migrated),'parent_archive_sha256':archive,'inherited_model_seconds':3.2575200367718935,'limit':1500,'wall_seconds':time.monotonic()-start}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,default=PARENT);a=p.parse_args();main(a.root,a.parent)
