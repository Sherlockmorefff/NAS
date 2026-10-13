import csv,hashlib,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
P4='results/structured_kernel_arch_main_ablation_20261006T065616Z'
ORIGIN='results/search/high-regime-prior-model-blind-validation-global4-v1'
V1='results/posthoc/original-head-deterministic-batch-pilot-global4-v1'
V2='results/posthoc/original-head-confirmatory-attribution-global4-v2'
H='results/posthoc/h-vs-head-model-blind-prospective-global4-v1'
PREP='results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2'
B='results/structured_kernel_phase2_20261002T161300Z'
REPLAY='results/structured_kernel_development_replay_20261002T235614Z'
PERF='PERFORMANCE_DEVELOPMENT';FREE='LABEL_FREE_ONLY_VERIFIED';UNKNOWN='USE_UNRESOLVED'
FULL_SOURCES={ORIGIN+'/frozen_candidate_configs.json':(192,768),V1+'/frozen_candidate_pools.json':(512,2048),V2+'/frozen_candidate_pools.json':(1536,6144),PREP+'/candidates.json':(192,336)}
C_FILES=['C_unlabeled_manifest.json','C_eligibility_manifest.json','historical_use_registry.json','historical_fingerprints.json','historical_graphs.json','extended_feature_manifest.json','frozen_dictionary.json','C_features.json','summary.json']
META_NAMES=('protocol.json','preregistration.json','input_manifest.json','source_manifest.json','source_dependency_manifest.json','source_identity.json','run_identity.json','data_use.json')
def now():return datetime.now(timezone.utc).isoformat()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def seal(path,x):
 if Path(path).exists():
  if read(path)!=x:raise ValueError('Refusing changed immutable evidence: '+str(path))
 else:write(path,x)
def csvwrite(path,rows):
 rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def log(path,row):
 with Path(path).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def guard(root,repo,allowed):
 allowed={str((repo/p).resolve()) for p in allowed};counts={'permitted_project_read_opens':0,'forbidden_read_attempts':0};root=root.resolve();repo=repo.resolve()
 def audit(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes)):return
  p=Path(args[0]).resolve();mode=args[1];flags=args[2]
  writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and (flags & (os.O_WRONLY|os.O_RDWR)))
  if p.is_relative_to(root):return
  if not writing and (p.is_relative_to(repo/'results') or p.is_relative_to(repo/'legacy_artifacts')):
   if str(p) not in allowed:
    counts['forbidden_read_attempts']+=1;raise PermissionError('Audit allowlist refuses '+str(p))
   counts['permitted_project_read_opens']+=1
  if writing and (p.is_relative_to(repo/'results') or p.is_relative_to(repo/'legacy_artifacts')):raise PermissionError('No changes to historical material')
 sys.addaudithook(audit);return counts
