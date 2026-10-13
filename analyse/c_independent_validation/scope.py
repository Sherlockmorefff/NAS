import os, shutil, subprocess
from collections import Counter
from .common import *
from .projection import FIELDS

def group(p):
 bits=p.split('/')
 return '/'.join(bits[:3]) if p.startswith(('results/posthoc/','results/search/')) else '/'.join(bits[:2])

def freeze(root,repo,prompt):
 if root.exists():raise ValueError('A new result directory is required')
 root.mkdir(parents=True);shutil.copy2(prompt,root/'PROMPT.txt')
 cut=now();old=repo/PREVIOUS;minimum=repo/MINIMAL
 verify(old,read(old/'review_complete.json')['files'])
 verify(minimum,read(minimum/'audit_complete.json')['files'])
 mpath=repo/P4/'C_preparation/C_unlabeled_manifest.json';m=read(mpath)
 matches=[a for a in m['architectures'] if a['architecture_fingerprint']==EXCLUDED]
 configs=[c for c in m['configurations'] if c['architecture_fingerprint']==EXCLUDED]
 assert len(matches)==1 and len(configs)==4 and len({x['candidate_id'] for x in configs})==4
 assert len(m['architectures'])==45 and len(m['configurations'])==180
 write(root/'precautionary_exclusion.json',{'architecture_id':EXCLUDED,'candidate_ids':[c['candidate_id'] for c in configs],
  'classification':'PRECAUTIONARY_PROVENANCE_EXCLUSION','basis':'Three coarse low-fidelity matches; complete effective attributes/use unresolved; no score-based selection',
  'confirmed_development_leakage':False,'original_manifest_sha256':sha(mpath),'excluded_before_new_projection':True,'time':cut})
 gaps=read(old/'remaining_gaps.json');queue=[r for r in gaps if r['gap']!='MATCHED_EFFECTIVE_ATTRIBUTES_AND_USE'];assert len(queue)==161
 sources={r['source']:{'reason':'Prior concrete unresolved coverage obligation','group':group(r['source']),'old_obligation':True} for r in queue}
 # Only companions of already referenced runs. Names choose schema/identity
 # carriers, but never certify their contents safe; all use byte projection.
 parents={ (repo/r['source']).parent for r in queue }
 parents.update(repo/group(r['source']) for r in queue)
 for parent in sorted(parents):
  if not parent.is_dir():continue
  for p in sorted(parent.iterdir()):
   if p.is_file() and p.suffix in ('.json','.jsonl') and any(w in p.name for w in ('manifest','config','protocol','identity','freeze','source','selected','task_plan')):
    rel=str(p.relative_to(repo));sources.setdefault(rel,{'reason':'Identity/schema companion in exact referenced run','group':group(rel),'old_obligation':False})
 for p,s in sources.items():
  path=repo/p;s.update({'format':'jsonl' if p.endswith('.jsonl') else 'json','projection':'IdentityProjector: explicit scalar allowlist, recursive JSON containers',
   'allowed_fields':sorted(FIELDS),'producer_basis':'actual source code and run companions; no version qualification assumed from current code','exists':path.is_file()})
  if path.is_file():s.update(size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
 write(root/'identity_projection_plan.json',sources)
 write(root/'coverage_worklist.json',queue)
 protocol={'version':'C-independent-validation-1','cutoff':cut,'previous_cutoff':'2026-10-06T12:29:47.968642+00:00',
  'definition':'No performance development use of architecture under any HP/seed; label-free exposure permitted and disclosed',
  'method':'W-I','excluded_architecture':EXCLUDED,'remaining_max_architectures':44,'remaining_max_configurations':176,
  'P1_gate':'U > 71 with identity and representation consistency','max_GNN_tasks':528,'training_seeds':[0,1,2],
  'max_GP_fits':422,'max_GP_starts':1266,'search_seeds':list(range(2026100200,2026100210)),
  'stages':['P0','P1','P2','P3','P4','P5','P6'],'P2_performance_protocol_state':'NOT_STARTED',
  'P1_no_performance_value_decoding':True,'bytes_and_hash_access_authorized':True,
  'new_candidates_or_encoder_updates':False,'conditional_auto_continue':True,
  'queue_policy':'Exact gap file -> related run/snapshot/Git -> known same-project backup; only newly explicit references extend queue',
  'no_automatic_dataset_or_method_change':True}
 write(root/'protocol.json',protocol)
 # Metadata-only single pass. File bytes are not opened by this inventory.
 timestamp=datetime.fromisoformat(protocol['previous_cutoff']).timestamp();changed=[]
 for base in ('results','logs','artifacts'):
  for parent,dirs,files in os.walk(repo/base):
   dirs[:]=[d for d in dirs if Path(parent)/d != root]
   for name in files:
    p=Path(parent)/name
    try:st=p.stat()
    except FileNotFoundError:continue
    if st.st_mtime>timestamp:changed.append({'path':str(p.relative_to(repo)),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
 write(root/'post_cutoff_inventory.json',{'since':protocol['previous_cutoff'],'through':cut,'files':changed,'byte_reads':0,
  'limitation':'Recorded files/metadata only; not a claim about deleted or unlogged activity'})
 (root/'process_snapshot.txt').write_text(subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid,comm,etime'],text=True))
 (root/'workspace_diff_before.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=repo))
 (root/'git_status_before.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=repo))
 safe=[old/'README_RESULTS.md',old/'NEXT_REQUIRED_STEP.md',old/'GPT_ANALYSIS_REQUEST.md',old/'replacement_C_route.md',old/'actual_commands.md',old/'review_code_identity.json',old/'remaining_gaps.csv',old/'new_evidence_manifest.json',minimum/'input_manifest.json',mpath]
 for name in ('C_features.json','frozen_dictionary.json','extended_feature_manifest.json','C_eligibility_manifest.json'):
  safe.append(repo/P4/'C_preparation'/name)
 for part in ('structured_kernel_development_replay_20261002T235614Z','structured_kernel_followup_20261005T152816Z'):
  for name in ('protocol.json','source_manifest.json','run_identity.json'):
   safe.append(repo/'results'/part/name)
 code=[p for part in ('c_independent_validation','structured_kernel_development_replay','structured_kernel_followup') for p in sorted((repo/'analyse'/part).glob('*.py'))]
 code += [repo/'analyse/c_provenance_evidence_closure/projector.py',repo/'final_eval.py',repo/'eval_utils.py',repo/'bo_phase4.py']
 identity={'time':cut,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
  'safe_inputs':{str(p.relative_to(repo)):sha(p) for p in safe},'initial_code':{str(p.relative_to(repo)):sha(p) for p in code},
  'protocol_sha256':sha(root/'protocol.json'),'projection_plan_sha256':sha(root/'identity_projection_plan.json'),
  'prompt_sha256':sha(root/'PROMPT.txt'),'precautionary_exclusion_sha256':sha(root/'precautionary_exclusion.json')}
 identity['identity']=digest(identity);write(root/'run_identity.json',identity)
 write(root/'status.json',{'P0':'COMPLETE','P1':'PENDING',**{p:'NOT_STARTED' for p in ('P2','P3','P4','P5','P6')}})
 write(root/'costs.json',{'authorized_max_GNN_tasks':528,'authorized_max_GP_fits':422,'authorized_max_GP_starts':1266,'actual':ZERO})
 print(json.dumps({'run_dir':str(root),'cutoff':cut,'exclusion':len(configs),'old_obligations':len(queue),'projection_paths':len(sources),
  'post_cutoff_files':len(changed),'changed_top_groups':dict(Counter(group(x['path']) for x in changed))}))

