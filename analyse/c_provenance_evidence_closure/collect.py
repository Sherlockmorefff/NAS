import ast,hashlib,os,re,shutil,subprocess,sys,xml.etree.ElementTree as ET
from collections import Counter,defaultdict
from .common import *
from .projector import project,objects,IDENTITY,CONTAINERS,HASH_MAPS
from analyse.c_independence_minimal_audit.identity import fingerprint

OPS={'GCNConv','GATConv','SAGEConv','GINConv','GCNII','Identity'}
def valid_structure(row):
 ops=row.get('operations');edges=row.get('edges')
 return isinstance(ops,list) and 1<=len(ops)<=5 and all(x in OPS for x in ops) and isinstance(edges,list) and bool(edges) and all(isinstance(e,list) and len(e)==2 and all(type(i) is int for i in e) and 0<=e[0]<e[1]<len(ops)+2 for e in edges) and len({tuple(e) for e in edges})==len(edges)
def summarize(value,orbit_union):
 records=[];hits=[];mapping=defaultdict(set);invalid=0;declared_ids=set();references=set();identities=[]
 top=value if isinstance(value,list) else [value];top_valid=0;top_unresolved=0;top_nonmembers=0
 for i,row in enumerate(top):
  structural=[];ids=[];bad=False
  for pointer,obj in objects(row,'$/'+str(i)):
   aid=obj.get('architecture_fingerprint')
   if isinstance(aid,str) and aid:declared_ids.add(aid);ids.append(aid)
   for k in ('config_id','candidate_id','source_config_id','configuration_id'):
    if isinstance(obj.get(k),str):references.add(obj[k])
   if 'operations' not in obj and 'edges' not in obj:continue
   if not valid_structure(obj):
    invalid+=1;bad=True
    records.append({'pointer':pointer,'status':'EMPTY_OR_UNSUPPORTED_STRUCTURE','stored_architecture_fingerprint':aid,'valid_flag':obj.get('valid')});continue
   coarse=fingerprint(obj);structural.append(coarse)
   if isinstance(aid,str) and aid:mapping[aid].add(coarse)
   records.append({'pointer':pointer,'status':'CONCRETE_RECORDED_STRUCTURE','recomputed_coarse_fingerprint':coarse,'stored_architecture_fingerprint':aid,'stored_identity_matches_recomputed':None if not aid else aid==coarse,'internal_node_count':len(obj['operations']),'effective_attribute_names':sorted(k for k in obj if k in ('gat_heads','sage_aggr','gin_eps','gcnii_alpha','gcnii_theta','activation','merge','normalization'))})
   if coarse in orbit_union:hits.append({'pointer':pointer,'coarse_fingerprint':coarse,'identity_and_structure':{k:v for k,v in obj.items() if k in IDENTITY}})
  if structural and not bad:
   top_valid+=1
   for aid in ids:mapping[aid].update(structural)
  elif isinstance(row,dict) and row.get('valid') is False and not ids:
   top_nonmembers+=1
  else:top_unresolved+=1
  identities.append({'record_index':i,'declared_architecture_ids':sorted(set(ids)),'concrete_structure_ids':sorted(set(structural)),'structure_complete_for_record':bool(structural) and not bad})
 return {'record_count':len(top),'records_with_supported_structure':top_valid,'unresolved_top_records':top_unresolved,'explicit_invalid_records_without_architecture':top_nonmembers,'structure_record_count':sum(r['status']=='CONCRETE_RECORDED_STRUCTURE' for r in records),'unsupported_or_empty_structure_records':invalid,'declared_architecture_ids':sorted(declared_ids),'configuration_references':sorted(references),'concrete_structure_ids':sorted({r['recomputed_coarse_fingerprint'] for r in records if 'recomputed_coarse_fingerprint' in r}),'stored_to_concrete':{k:sorted(v) for k,v in mapping.items()},'records':records,'record_membership':identities,'coarse_hits':hits}
def stage_code(repo):
 paths=['analyse/c_provenance_evidence_closure/'+n for n in ('common.py','projector.py','collect.py')]
 paths += [str(p.relative_to(repo)) for p in (repo/'tests/c_provenance_evidence_closure').glob('*.py')]
 return {p:sha(repo/p) for p in sorted(paths)}
def guard(root,repo,allowed):
 allowed={str((repo/p).resolve()) for p in allowed};denied=[]
 def hook(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes)):return
  p=Path(args[0]).resolve();mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
  if p.is_relative_to(root):return
  if p.is_relative_to(repo) and p.suffix in ('.pt','.pth','.ckpt','.pkl'):raise PermissionError('No weights/dataset objects allowed')
  if p.is_relative_to(repo/'results') or p.is_relative_to(repo/'legacy_artifacts'):
   if writing or str(p) not in allowed:denied.append(str(p));raise PermissionError('Outside frozen evidence allowlist: '+str(p))
 sys.addaudithook(hook);return denied

def git_binding(root,repo,safe):
 targets=defaultdict(list);commits=[]
 for p in safe:
  if p.endswith('/source_sha256.txt'):
   for line in (repo/p).read_text().splitlines():
    parts=line.split()
    if len(parts)==2 and re.fullmatch('[0-9a-f]{64}',parts[0]):targets[(parts[1].lstrip('*'),parts[0])].append(p)
  if p.endswith('/git_commit.txt'):commits.append({'source':p,'recorded_commit':(repo/p).read_text().strip(),'binding_limit':'Commit alone does not exclude uncommitted working changes.'})
 bindings=[]
 for (name,h),records in sorted(targets.items()):
  known=[p for p,v in safe.items() if p.endswith('/'+name) or p==name if v==h]
  revisions=subprocess.check_output(['git','log','--all','--format=%H','--',name],cwd=repo,text=True).splitlines();matched=[];unique={}
  for rev in revisions:
   obj=subprocess.run(['git','show',rev+':'+name],cwd=repo,capture_output=True)
   if obj.returncode:continue
   blob=hashlib.sha256(obj.stdout).hexdigest();unique[blob]=rev
   if blob==h:matched.append(rev)
  bindings.append({'recorded_code_path':name,'recorded_sha256':h,'run_source_hash_records':records,'exact_snapshot_matches':known,'exact_git_matches':matched,'distinct_git_versions_checked':len(unique),'status':'EXACT_CODE_CONTENT_FOUND' if known or matched else 'NO_EXACT_CODE_CONTENT_FOUND','limit':'Recorded source hash belongs to named pilot metadata; actual run/version linkage still assessed separately. A similar function is not a matching source version.'})
 result={'bindings':bindings,'commit_only_records':commits,'known_snapshot_files_considered':sum('/source_snapshot/' in p for p in safe),'proof_does_not_assume_clean_worktree':True}
 write(root/'historical_version_search.json',result)
 return result

def collect(root,repo,resume=False,acceptance=None):
 safe=read(root/'safe_input_manifest.json');mixed=read(root/'mixed_allowlist.json');identity=read(root/'run_identity.json')
 assert identity['mixed_allowlist_sha256']==sha(root/'mixed_allowlist.json') and identity['protocol_sha256']==sha(root/'protocol.json')
 verify(repo,safe);code=stage_code(repo)
 lock=root/'collection_code_identity.json'
 if lock.exists():assert read(lock)==code,'Collection code changed; new evidence identity required'
 else:
  if acceptance is None:raise ValueError('Projection tests must pass before opening mixed files')
  suites=list(ET.parse(acceptance/'pytest.xml').getroot().iter('testsuite'))
  assert sum(int(s.attrib.get('tests',0)) for s in suites)>=8 and not any(int(s.attrib.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
  assert read(acceptance/'code_identity.json')==code
  shutil.copytree(acceptance,root/'acceptance');write(lock,code)
 denied=guard(root,repo,set(safe)|set(mixed))
 if (root/'collection_complete.json').exists():
  assert resume;verify(root,read(root/'collection_complete.json')['files']);write(root/'collection_resume.json',{'state':'PASS','new_evidence_collections':0,'new_mixed_file_reads':0,'new_enumerations':0,'new_graph_recoveries':0,'new_model_operations':0,'core_hashes_unchanged':True,'time':now()});print(canonical(read(root/'collection_resume.json')));return
 state=read(root/'phase_status.json');state['P1']='RUNNING';write(root/'phase_status.json',state)
 om=read(repo/OLD/'orbit_manifest.json');orbit_union=set()
 for v in om['architectures'].values():
  assert sha(repo/OLD/v['path'])==v['sha256'];orbit_union.update(read(repo/OLD/v['path'])['unique_coarse_hashes'])
 versions=git_binding(root,repo,safe);outputs={};counts=Counter();errors=[];new_reads=0
 for number,(rel,spec) in enumerate(sorted(mixed.items()),1):
  key=digest(rel);target=root/'projections'/f'{key}.json';complete=root/'projection_checkpoints'/f'{key}.json'
  if complete.exists():
   saved=read(complete);assert saved['output_sha256']==sha(target) and saved['code_identity']==digest(code);outputs[rel]=saved;continue
  if not (repo/rel).is_file():
   result={'source':rel,'state':'MISSING','reason':'Exact frozen path not present; no same-name substitution'}
  else:
   before=(repo/rel).stat();log(root/'access_log.jsonl',{'event':'mixed_stream_start','source':rel,'at':now()})
   try:
    value,counter=project(repo/rel);after=(repo/rel).stat();new_reads+=1;counts.update({k:v for k,v in counter.items() if isinstance(v,int)})
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during projection')
    expected=spec.get('recorded_sha256');bound=expected is None or expected==counter['sha256']
    summary=summarize(value,orbit_union)
    # Persist raw structural fields only for a targeted coarse hit. Other
    # histories retain identity-only records, not all historical full graphs.
    result={'source':rel,'state':'COMPLETE' if bound else 'HASH_MISMATCH','profile':spec['profile'],'group':spec['group'],'actual_sha256':counter['sha256'],'expected_historical_sha256':expected,'historical_hash_matches':bound,'access':counter,'identity_summary':summary}
    if spec['profile']!='BO_history_record_list':result['identity_projection']=value
    result['new_full_graphs_recovered']=len(summary['coarse_hits'])
   except Exception as e:
    result={'source':rel,'state':'PROJECTION_FAILED','reason':type(e).__name__+': '+str(e)};errors.append({'source':rel,'reason':result['reason']})
  write(target,result);saved={'path':str(target.relative_to(root)),'output_sha256':sha(target),'code_identity':digest(code),'source_sha256':result.get('actual_sha256'),'state':result['state']};write(complete,saved);outputs[rel]=saved
  if number%25==0:write(root/'progress.json',{'stage':'P1','processed_frozen_paths':number,'total':len(mixed),'new_streams_this_invocation':new_reads,'errors':len(errors),'time':now()})
 # Aggregate from saved evidence for interrupted-run correctness.
 total=Counter();statuses=Counter();structures=0;hits=0
 for v in outputs.values():
  d=read(root/v['path']);statuses[d['state']]+=1;total.update({k:n for k,n in d.get('access',{}).items() if isinstance(n,int)});structures+=d.get('identity_summary',{}).get('structure_record_count',0);hits+=d.get('new_full_graphs_recovered',0)
 write(root/'projection_manifest.json',outputs);write(root/'projection_errors.json',errors)
 result={'sources':len(outputs),'status_counts':dict(statuses),'access_counters':dict(total),'historical_structure_records_projected':structures,'targeted_coarse_hit_records':hits,'new_candidate_orbit_evaluations':0,'reused_candidate_orbit_evaluations':5304,'full_historical_graph_reconstruction_performed':False,'forbidden_file_attempts':denied,'operations':ZERO}
 write(root/'collection_summary.json',result);write(root/'access_counters.json',{**dict(total),'model_operations':ZERO})
 names=['projection_manifest.json','projection_errors.json','collection_summary.json','access_counters.json','historical_version_search.json']+[v['path'] for v in outputs.values()]
 write(root/'collection_complete.json',{'files':{p:sha(root/p) for p in names},'time':now(),'identity':identity['identity']});state['P1']='COMPLETE';state['P2']='PENDING';write(root/'phase_status.json',state)
 print(canonical(result))
