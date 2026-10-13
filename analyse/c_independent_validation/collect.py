import sys,time
from collections import Counter
from .common import *
from .projection import IdentityProjector,FIELDS

def collect(root,repo,resume=False):
 started=time.monotonic();plan=read(root/'identity_projection_plan.json');identity=read(root/'run_identity.json')
 assert sha(root/'identity_projection_plan.json')==identity['projection_plan_sha256']
 assert sha(root/'precautionary_exclusion.json')==identity['precautionary_exclusion_sha256']
 code={str(p.relative_to(repo)):sha(p) for p in [repo/'analyse/c_independent_validation/projection.py',repo/'analyse/c_provenance_evidence_closure/projector.py']}
 lock=root/'projection_code_identity.json'
 if lock.exists():assert read(lock)==code,'Projector changed; new identity required'
 else:
  acceptance=read(root/'projection_acceptance.json');assert acceptance['state']=='PASS' and acceptance['code']==code
  write(lock,code)
 allowed={(repo/p).resolve() for p in plan};denials=[]
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes)):return
  path=Path(args[0]).resolve();mode=args[1];flags=args[2]
  if path.is_relative_to(root):return
  if path.is_relative_to(repo/'results') or path.is_relative_to(repo/'legacy_artifacts'):
   writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
   if writing or path not in allowed:
    denials.append(str(path));raise PermissionError('Outside exact P1 projection allowlist')
 sys.addaudithook(guard)
 totals=Counter();outputs={};new=0;reused=0
 status=read(root/'status.json');status['P1']='RUNNING';write(root/'status.json',status)
 for i,(rel,spec) in enumerate(sorted(plan.items(),key=lambda x:('cora-layernorm-search-space' not in x[0],x[0])),1):
  assert spec['allowed_fields']==sorted(FIELDS)
  key=digest(rel);target=root/'identity_projections'/f'{key}.json';mark=root/'projection_checkpoints'/f'{key}.json';p=repo/rel
  if mark.exists():
   if not resume:raise ValueError('Existing projection requires --resume')
   saved=read(mark);assert saved['code']==code and sha(target)==saved['projection_sha256']
   if saved.get('source_sha256'):assert p.is_file() and sha(p)==saved['source_sha256']
   elif saved['state']=='MISSING':assert not p.exists()
   outputs[rel]=saved;reused+=1;continue
  if not p.is_file():
   result={'source':rel,'state':'MISSING','reason':'Exact referenced server path absent at projection time'}
  else:
   before=p.stat();parser=None
   log(root/'P1_access_log.jsonl',{'event':'byte_projection_start','source':rel,'time':now()})
   try:
    with p.open('rb') as f:
     parser=IdentityProjector(f);value=parser.all(rel.endswith('.jsonl'))
    counter=parser.counters();after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Input changed during read'
    result={'source':rel,'state':'PROJECTED','identity_only':value,'source_sha256':counter['sha256'],'access':counter,
     'source_stat':{'bytes':after.st_size,'mtime_ns':after.st_mtime_ns},'performance_values_decoded':False}
    totals.update({k:v for k,v in counter.items() if type(v) is int});new+=1
   except Exception as exc:
    counter=parser.counters() if parser else {};totals.update({k:v for k,v in counter.items() if type(v) is int})
    result={'source':rel,'state':'UNSUPPORTED_SAFE_PROJECTION','error_type':type(exc).__name__,'access':counter,
     'reason':'Parser/schema failure; no raw exception payload persisted'}
  write(target,result)
  checkpoint={'path':str(target.relative_to(root)),'projection_sha256':sha(target),'source_sha256':result.get('source_sha256'),
   'state':result['state'],'code':code};write(mark,checkpoint);outputs[rel]=checkpoint
  if i%25==0:write(root/'P1_progress.json',{'projected_or_checked':i,'total_paths':len(plan),'time':now()});print(json.dumps({'P1_projection':i,'total':len(plan)}),flush=True)
 write(root/'projection_manifest.json',outputs)
 report={'state':'COMPLETE','new_projections':new,'reused_projections':reused,'total_paths':len(plan),'access':dict(totals),
  'source_states':dict(Counter(x['state'] for x in outputs.values())),'new_orbit_evaluations':0,'new_model_operations':0,
  'forbidden_read_attempts':len(denials),'elapsed_seconds':time.monotonic()-started,'time':now()}
 write(root/('projection_resume.json' if resume else 'projection_summary.json'),report)
 print(json.dumps(report),flush=True)

