"""Metadata-only provenance: preserve known links and explicit possible links."""
import ast
from collections import Counter,defaultdict
from .common import *
from .identity import fingerprint,isolated_functions,SCHEMA
from analyse.structured_kernel_phase1.features import build_operator_graph,graph_hash

def snippets(path,names):
 text=Path(path).read_text();lines=text.splitlines();tree=ast.parse(text)
 return [{'name':n.name,'line':n.lineno,'end_line':n.end_lineno,'source':'\n'.join(lines[n.lineno-1:n.end_lineno])} for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]

def build(root,repo):
 inputs=read(root/'input_manifest.json');scope=read(root/'history_scope.json')
 schema={'id':SCHEMA,'payload':{'operations':'str-valued ordered internal operations','edges':'sorted lists of int endpoints'},'serialization':'json.dumps(sort_keys=True,separators=(comma,colon),ensure_ascii=True,allow_nan=False).encode(utf-8)','hash':'SHA256','input_output':'0 and m+1','attributes_omitted':['activation','merge','GAT heads','GCNII layer/alpha/theta','SAGE aggregator','GIN eps'],'HP_omitted':True,'aliases':'none proven; no speculative conversion','full_permutation_superset':True,'proof':'Exact typed isomorphism implies at least one matching coarse numbering when source schema and coverage are verified.'}
 pure=isolated_functions((repo/'architecture_diagnostics.py').read_text(),['architecture_fingerprint'])['architecture_fingerprint']
 canon=isolated_functions((repo/'initialization_wgmm_ted.py').read_text(),['canonical_json_fingerprint'])['canonical_json_fingerprint']
 bo=isolated_functions((repo/'bo_phase4.py').read_text(),['_decoded_architecture_fingerprint'],{'canonical_json_fingerprint':canon})['_decoded_architecture_fingerprint']
 code_evidence={}
 for p,names in [('architecture_diagnostics.py',['architecture_fingerprint']),('initialization_wgmm_ted.py',['canonical_json_fingerprint']),('bo_phase4.py',['_decoded_architecture_fingerprint','_save_history']),('analyse/h_vs_head_model_blind_prospective.py',['_truth_row_fingerprint','_collect_truth_fingerprints','_scan_json_source','exposure_source_groups','build_exposure_registry'])]:
  code_evidence[p]={'sha256':inputs[p],'functions':snippets(repo/p,names)}
 sources={};graphs={};members={};identity_map=[];bindings={};full_counts={}
 prior=read(repo/P4/'final_checks/prior_historical_use_metadata.json')
 dev_ids=set(prior['confirmed_use'][0]['members'])
 for path,(narch,nconfig) in FULL_SOURCES.items():
  rows=read(repo/path);assert len(rows)==nconfig
  g={};member=[]
  for row in rows:
   aid=row['architecture_fingerprint'];assert fingerprint(row)==aid==pure(row)==bo(row)
   graph=build_operator_graph(row)
   if aid in g:assert graph==g[aid]
   g[aid]=graph
   cid=row.get('candidate_id',row.get('joint_config_fingerprint',row.get('joint_fingerprint',row.get('config_id'))))
   member.append({'architecture_fingerprint':aid,'configuration_identity':cid,'configuration_identity_field':next((k for k in ('candidate_id','joint_config_fingerprint','joint_fingerprint','config_id') if k in row),None),'graph_hash':graph_hash(graph)})
  assert len(g)==narch
  if path.startswith(PREP):assert {r['candidate_id'] for r in rows}==dev_ids
  graphs[path]=g;members[path]=sorted(g);identity_map.extend({'source':path,**r} for r in member)
  base=str(Path(path).parent);bound=False;binding={'actual_pool_sha256':inputs[path],'regression_rows':len(rows),'regression_current_functions_equal':True}
  immutable_path=base+'/pre_label_immutable_manifest.json'
  if immutable_path in inputs:
   im={r['relative_path']:r['sha256'] for r in read(repo/immutable_path)['files']}
   binding['prelabel_pool_match']=im.get(Path(path).name)==inputs[path]
   assert binding['prelabel_pool_match']
   dep=base+'/source_dependency_manifest.json';snap=base+'/source_snapshot/architecture_diagnostics.py'
   if dep in inputs and snap in inputs:
    dependencies={r['relative_path']:r['sha256'] for r in read(repo/dep)['files']}
    binding['prelabel_dependency_match']=im.get('source_dependency_manifest.json')==inputs[dep]
    binding['snapshot_architecture_code_match']=dependencies.get('architecture_diagnostics.py')==inputs[snap]
    historical=isolated_functions((repo/snap).read_text(),['architecture_fingerprint'])['architecture_fingerprint']
    assert all(historical(r)==r['architecture_fingerprint'] for r in rows)
    producer='analyse/high_regime_prior_model_blind_validation.py' if base==ORIGIN else 'analyse/original_head_confirmatory_attribution_v2.py'
    producer_snap=base+'/source_snapshot/'+producer
    binding['producer_snapshot_bound']=producer_snap in inputs and dependencies.get(producer)==inputs[producer_snap]
    binding['producer_snapshot_path']=producer_snap
    bound=all(binding.get(k,False) for k in ('prelabel_dependency_match','snapshot_architecture_code_match','producer_snapshot_bound'))
    binding['fingerprint_snapshot_path']=snap;binding['fingerprint_snapshot_sha256']=inputs[snap]
   else:binding['limitation']='Prelabel producer exists, but this pool freeze does not bind imported architecture fingerprint dependency version; full graphs still prove local exact disjointness.'
  bindings[path]=binding
  sources[path]={'source_id':path,'use':PERF if path.startswith((ORIGIN,PREP)) else UNKNOWN,'use_evidence':P4+'/final_checks/prior_historical_use_metadata.json' if path.startswith((ORIGIN,PREP)) else 'Generated pool contains multiple later-use subsets; no blanket label-free/performance assignment.',
   'architecture_member_count':narch,'configuration_count':nconfig,'identity_type':'architecture_fingerprint mapped from complete configurations','identity_type_verified':True,'coverage_complete':True,'coverage_scope':'all records in this frozen config file; not a claim to cover all project history','schema_verified':bound,'schema':SCHEMA if bound else None,'observed_schema_regression':SCHEMA,'producer_version_evidence':binding,'source_sha256':inputs[path],'member_ref':path,'exact_graphs_available':True,'associations':[],'gaps':[]}
  full_counts[path]={'architectures':narch,'configurations':nconfig,'regression_matches':nconfig}
 # The 192 known development architectures cover all four HPs/any seed by definition.
 assert set(graphs[ORIGIN+'/frozen_candidate_configs.json'])==set(graphs[PREP+'/candidates.json'])
 group_members=defaultdict(set)
 hrows=read(repo/H/'phase_a/exposure_registry.json')
 for row in hrows:
  for group in row['source_groups']:group_members[group].add(row['architecture_fingerprint'])
 associations=[]
 for group,ids in sorted(group_members.items()):
  key='source_group:'+group;members[key]=sorted(ids)
  for aid in sorted(ids):associations.append({'architecture_fingerprint':aid,'known_source_group':group,'original_file_association':'not retained; candidate files are those in this group, not asserted occurrences in each file'})
 for base,reg in [(H,H+'/phase_a/exposure_registry.json'),(V1,V1+'/exposure_registry.json'),(V2,V2+'/exposure_registry.json'),(ORIGIN,ORIGIN+'/exposure_registry.json')]:
  ids=sorted({r['architecture_fingerprint'] for r in read(repo/reg)});members[reg]=ids
  sources[reg]={'source_id':reg,'use':UNKNOWN,'use_evidence':'Mixed provenance aggregation; presence is not proof of performance development.','architecture_member_count':len(ids),'identity_type':'declared architecture_fingerprint; original producer/version may be missing','identity_type_verified':False,'coverage_complete':False,'observed_registry_members_complete':True,'schema_verified':False,'schema':None,'member_ref':reg,'source_sha256':inputs[reg],'exact_graphs_available':False,'associations':[],'gaps':['G1_HISTORICAL_VERSION_AND_ASSOCIATION','G2_SCOPE_MEMBER_CLOSURE'],'producer_version_evidence':'Current collector accepts stored architecture_fingerprint verbatim; original producer version not bound.'}
 original=scope['historical_original_source_records'];zero_sources=[]
 for row in original:
  path=str(Path(row['path']).relative_to(repo));group=row['source_group'];metadata=str(Path(path).parent/'history_metadata.json')
  meta=read(repo/metadata) if metadata in inputs else None
  record={'source_id':path,'use':UNKNOWN,'use_evidence':'Historical collector found truth-shaped records; cache presence does not establish downstream performance use.','architecture_member_count':row['fingerprint_count'],'identity_type':'declared architecture_fingerprint from stored-or-recomputed collector branch','identity_type_verified':False,'coverage_complete':False,'schema_verified':False,'member_ref':'source_group:'+group,'member_ref_relation':'conservative group superset; per-file member set lost','original_file_opened':False,'recorded_source_sha256':row['sha256'],'recorded_hash_rechecked':False,'source_group':group,'inventory_evidence':H+'/phase_a/exposure_source_inventory.json','metadata_evidence':metadata if meta is not None else None,'producer_version_evidence':{'formal_source_id':meta.get('formal_source_id') if meta else None,'historical_source_code_hash_bound':False},'associations':[],'gaps':['G1_HISTORICAL_VERSION_AND_ASSOCIATION','G2_SCOPE_MEMBER_CLOSURE']}
  sources[path]=record
  if not row['fingerprint_count']:zero_sources.append(path)
 recent=scope['recent_original_source_records'];duplicates=defaultdict(list)
 for row in recent['inventory']:
  path=str(Path(row['path']).relative_to(repo));h=row.get('sha256');duplicates[h].append(path)
  if path in sources:sources[path]['associations'].append({'inventory':ORIGIN+'/exposure_source_inventory.json','record':row});continue
  refs=row.get('resolved_references',[]);direct=row.get('direct_fingerprint_count',0)
  key=path if refs and direct==0 else ORIGIN+'/exposure_registry.json'
  if key==path:members[path]=sorted({r['architecture_fingerprint'] for r in refs})
  sources[path]={'source_id':path,'use':UNKNOWN,'use_evidence':'Selection/generation/cache origin recorded; exact downstream membership not established from metadata alone.','architecture_member_count':direct+row.get('resolved_reference_count',0),'identity_type':'architecture_fingerprint and/or mapped config_id','identity_type_verified':False,'coverage_complete':False,'schema_verified':False,'member_ref':key,'member_ref_relation':'preserved explicit config references' if key==path else 'conservative merged registry superset; direct per-file members not saved','original_file_opened':False,'recorded_source_sha256':h,'recorded_hash_rechecked':False,'inventory_record':row,'associations':[],'gaps':['G1_HISTORICAL_VERSION_AND_ASSOCIATION','G2_SCOPE_MEMBER_CLOSURE']}
  for ref in refs:associations.append({'architecture_fingerprint':ref['architecture_fingerprint'],'configuration_identity':ref['config_id'],'configuration_identity_type':'source-local config_id','known_source_file':path,'mapping_sources':ref['resolved_by']})
 # Copies are identified from existing recorded file hashes only, never by directory names.
 copies=[{'recorded_sha256':h,'paths':v,'meaning':'identical referenced file content according to frozen inventory; not proof entire experiment directories are copies'} for h,v in duplicates.items() if h and len(v)>1]
 cpath=P4+'/C_preparation/C_unlabeled_manifest.json'
 ccode='analyse/structured_kernel_arch_main_ablation/c_prepare.py';runid=read(repo/P4/'run_identity.json');completion=read(repo/P4/'C_completion.json')
 assert runid['code'][ccode]==inputs[ccode]
 assert completion['files']['C_preparation/C_unlabeled_manifest.json']==inputs[cpath]
 sources['event:fourth_round_C_structure_only']={'source_id':'event:fourth_round_C_structure_only','use':FREE,'label_free_event_closed':True,'coverage_complete':True,'identity_type_verified':True,'schema_verified':True,'schema':SCHEMA,'architecture_member_count':45,'configuration_count':180,'exact_graphs_available':False,'member_ref':'C','use_evidence':{'code':ccode,'code_sha256':inputs[ccode],'run_identity':P4+'/run_identity.json','completion':P4+'/C_completion.json','actual_summary':P4+'/C_preparation/summary.json','downstream_scope':'Through frozen cutoff: C generation/encoding event only; all earlier source occurrences assessed independently. No claim the architecture was never otherwise seen.'},'gaps':[]}
 sources['coverage:project_development_members']={'source_id':'coverage:project_development_members','use':UNKNOWN,'coverage_complete':False,'schema_verified':False,'identity_type_verified':False,'architecture_member_count':None,'member_ref':None,'gaps':['G2_SCOPE_MEMBER_CLOSURE'],'use_evidence':'Existing directory inventory and fingerprint inventories do not certify member closure of every related development run up to cutoff. No safe exhaustive per-run member/use ledger is present.','possible_candidate_domain':'No proven necessary-condition mismatch for any C architecture; project uses the same operator grammar.'}
 evidence={'schema':schema,'current_function_evidence':code_evidence,'current_functions_equivalent_only_on_valid_domain':True,'historical_schema_bindings':bindings,'original_file_count':len(original),'history_metadata_files_checked':sum(p.endswith('/history_metadata.json') for p in inputs),'history_metadata_bound_source_versions':0,'full_graph_sources':full_counts,'same_content_file_copies':copies,'zero_extracted_member_source_count':len(zero_sources),'zero_extracted_members_not_equivalent_to_exhaustive_experiment_coverage':True}
 result={'sources':list(sources.values()),'cutoff':scope['cutoff_utc'],'uses':dict(Counter(s['use'] for s in sources.values())),'verified_byte_schema_count':1,'schema_evidence':evidence,'identity_type_rules':{'architecture_fingerprint':'coarse ops/edges identity only after schema proof','architecture_group_id':'must map using producer; never equated to a SHA architecture fingerprint','candidate_id/joint_config_fingerprint':'configuration identities, mapped only via safe full configs','hp_fingerprint':'never architecture identity'}}
 write(root/'source_schema_and_use_registry.json',result);write(root/'source_member_sets.json',members);write(root/'source_identity_mapping.json',identity_map);write(root/'record_source_associations.json',associations);write(root/'source_full_graphs.json',graphs);write(root/'schema_evidence.json',evidence)
 return result,members,graphs
