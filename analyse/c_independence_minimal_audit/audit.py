import ast,math,shutil,sys,time,xml.etree.ElementTree as ET
from collections import Counter,defaultdict
from .common import *
from .identity import *
from .sources import build
from analyse.structured_kernel_phase1.features import build_operator_graph,graph_hash

ZERO={k:0 for k in ['GP_fits','GP_optimization','GP_predictions','GNN_training','VAE_updates','decoder_calls','new_candidates','C_performance_reads','C_acquisition_scores','B_labels_or_SE_reads','node_test_score_reads','KG']}
def verify_files(root,files):
 for p,h in files.items():
  if sha(root/p)!=h:raise ValueError('Evidence identity mismatch: '+p)
def code_files(repo):
 return {str(p.relative_to(repo)):sha(p) for base in ('analyse/c_independence_minimal_audit','tests/c_independence_minimal_audit') for p in sorted((repo/base).glob('*.py'))}
def freeze_code(root,repo,acceptance):
 code=code_files(repo)
 for rel in code:ast.parse((repo/rel).read_text())
 lock={'scope_lock':sha(root/'scope_lock.json'),'code':code,'input_manifest':sha(root/'input_manifest.json')}
 if (root/'run_identity.json').exists():
  if read(root/'run_identity.json')!=lock:raise ValueError('Changed code/input/scope; new run required')
 else:
  if acceptance is None:raise ValueError('Synthetic acceptance required before qualification')
  suites=list(ET.parse(acceptance/'pytest.xml').getroot().iter('testsuite'))
  if sum(int(s.attrib.get('tests',0)) for s in suites)<10 or any(int(s.attrib.get(k,0)) for s in suites for k in ('failures','errors','skipped')):raise ValueError('Synthetic acceptance incomplete')
  if read(acceptance/'code_identity.json')!=code:raise ValueError('Tests not bound to current code')
  shutil.copytree(acceptance,root/'acceptance');write(root/'run_identity.json',lock)
 return lock
def check_candidates(root,repo):
 m=read(repo/P4/'C_preparation/C_unlabeled_manifest.json');cs=m['configurations'];ars=m['architectures'];assert len(cs)==180 and len(ars)==45
 assert m['labels_or_scores_present'] is False
 features=read(repo/P4/'C_preparation/C_features.json');fm=read(repo/P4/'C_preparation/extended_feature_manifest.json');dictionary=read(repo/P4/'C_preparation/frozen_dictionary.json')
 assert dictionary==fm['dictionaries'] and fm['psi_dimension']==1522
 byid={a['architecture_fingerprint']:a for a in ars};assert len(byid)==45 and set(features)==set(byid)
 index=ExactIndex()
 for aid,a in byid.items():
  assert fingerprint(a)==aid and build_operator_graph(a)==a['graph'] and graph_hash(a['graph'])==a['graph_hash']
  assert not index.match(a['graph']);index.add(aid,a['graph'])
  assert len(features[aid])==fm['psi_dimension'] and all(math.isfinite(v) for v in features[aid])
 ids=set();hp=defaultdict(set);latent={}
 for c in cs:
  aid=c['architecture_fingerprint'];cid=c['candidate_id'];assert aid in byid and cid not in ids;ids.add(cid)
  assert c['joint_config_fingerprint']==cid==digest({'architecture_fingerprint':aid,'hp_fingerprint':c['hp_fingerprint']})
  assert c['hp_fingerprint']==digest([float(x) for x in c['raw_hp_vector']])
  assert fingerprint(c)==aid and graph_hash(build_operator_graph(c))==c['graph_hash']==byid[aid]['graph_hash']
  assert c['hp_mode']=='global4' and c['normalization']=='none'
  assert all(math.isfinite(x) and 0<=x<=1 for x in c['raw_hp_vector'])
  assert 1e-4<=c['lr']<=10**-1.5 and .1<=c['dropout']<=.6
  assert c['hidden_dim'] in [16,32,64,128,256,512] and c['l2'] in [1e-5,5e-5,1e-4,5e-4,1e-3]
  expected=[math.log10(c['lr']),c['dropout'],math.log2(c['hidden_dim']),math.log10(c['l2'])]
  assert all(abs(x-y)<1e-12 for x,y in zip(expected,c['HP4'])) and len(c['HP4'])==4
  assert len(c['latent12'])==12 and all(math.isfinite(x) for x in c['latent12'])
  if aid in latent:assert latent[aid]==c['latent12']
  latent[aid]=c['latent12'];hp[aid].add(tuple(c[k] for k in ('lr','dropout','hidden_dim','l2')))
 assert all(len(v)==4 for v in hp.values())
 write(root/'candidate_identity_checks.json',{'architectures':45,'configurations':180,'unique_exact_typed_architectures':45,'four_unique_realized_HP_per_architecture':True,'architecture_and_joint_and_HP_ids_recomputed':True,'HP4_and_feature_finite_checks':True,'psi_dimension':1522,'latent_dimension':12,'WL_recomputed':False,'features_reused_from':P4+'/C_preparation','dictionary_exposure':'A plus all original 45 C structures; no feature/dictionary reselection','source_feature_sha256':sha(repo/P4/'C_preparation/C_features.json'),'dictionary_sha256':sha(repo/P4/'C_preparation/frozen_dictionary.json')})
 return m
def run(root,repo,resume=False,acceptance=None):
 started=time.time();inputs=read(root/'input_manifest.json');allowed=read(root/'allowed_paths.json');counts=guard(root,repo,allowed)
 lock=read(root/'scope_lock.json')
 for file,k in [('history_scope.json','definition_and_scope_sha256'),('protocol.json','protocol_sha256'),('input_manifest.json','input_manifest_sha256'),('PROMPT.txt','prompt_sha256')]:
  assert sha(root/file)==lock[k]
 identity=freeze_code(root,repo,acceptance);verify_files(repo,inputs)
 if (root/'audit_complete.json').exists():
  if not resume:raise ValueError('Already complete; use --resume')
  complete=read(root/'audit_complete.json');verify_files(root,complete['files'])
  write(root/'resume_verification.json',{'state':'PASS','time':now(),'seconds':time.time()-started,'new_architecture_enumerations':0,'new_coarse_hash_evaluations':0,'new_graph_recoveries':0,'new_model_operations':0,'verified_result_files':len(complete['files']),'major_result_hashes_unchanged':True,'operations':ZERO,'access_guard':counts,'run_identity':digest(identity)})
  print(canonical(read(root/'resume_verification.json')));return
 state=read(root/'module_status.json');state['synthetic_acceptance']='COMPLETE';state['sources']='RUNNING';write(root/'module_status.json',state)
 log(root/'execution_log.jsonl',{'at':now(),'command':'run','resume':resume,'identity':digest(identity)})
 if (root/'sources_complete.json').exists():
  verify_files(root,read(root/'sources_complete.json'));registry=read(root/'source_schema_and_use_registry.json');members=read(root/'source_member_sets.json');graphs=read(root/'source_full_graphs.json')
 else:
  registry,members,graphs=build(root,repo)
  write(root/'sources_complete.json',{p:sha(root/p) for p in ['source_schema_and_use_registry.json','source_member_sets.json','source_identity_mapping.json','record_source_associations.json','source_full_graphs.json','schema_evidence.json']})
 state['sources']='COMPLETE';state['orbits']='RUNNING';write(root/'module_status.json',state)
 manifest=check_candidates(root,repo);arches=manifest['architectures'];configs=manifest['configurations'];sources=registry['sources']
 sourcehash=sha(root/'source_schema_and_use_registry.json');indices={}
 for source,g in graphs.items():
  idx=ExactIndex()
  for aid,graph in g.items():idx.add(aid,graph)
  indices[source]=idx
 historical_union=set(read(repo/P4/'C_preparation/historical_fingerprints.json'));oldgraphs=read(repo/P4/'C_preparation/historical_graphs.json')
 member_sets={k:set(v) for k,v in members.items()}
 physical=0;neworbits=0;orbits={};unionhits=[];matrix=[];eligibility=[];recovery=[];gap_candidates=defaultdict(list)
 for a in arches:
  aid=a['architecture_fingerprint'];op=root/'orbits'/f'{aid}.json';marker=root/'orbits'/f'{aid}.complete.json'
  if marker.exists():
   cert=read(marker)
   if cert['file_sha256']!=sha(op) or cert['graph_sha256']!=digest(a['graph']) or cert['run_identity']!=digest(identity):raise ValueError('Orbit checkpoint changed')
   o=read(op)
  else:
   log(root/'execution_log.jsonl',{'at':now(),'operation':'enumerate_start','architecture_id':aid})
   o=orbit(a['graph']);o['architecture_id']=aid;write(op,o);write(marker,{'file_sha256':sha(op),'graph_sha256':digest(a['graph']),'run_identity':digest(identity)})
   physical+=o['permutation_count'];neworbits+=1
  orbits[aid]={'path':str(op.relative_to(root)),'sha256':sha(op),'permutation_count':o['permutation_count'],'unique_hash_count':len(o['unique_coarse_hashes'])}
  orbitset=set(o['unique_coarse_hashes']);hits=sorted(orbitset&historical_union)
  if hits:
   unionhits.append({'architecture_id':aid,'coarse_hashes':hits,'meaning':'conditional on original record byte schema; no performance use or exact typed overlap inferred'})
   # Target only already-safe full graph evidence, never mixed histories.
   matched_records=[(k,v) for k,v in oldgraphs.items() if set(v['architecture_fingerprints'])&set(hits)]
   recovery.append({'architecture_id':aid,'coarse_hashes':hits,'available_safe_full_graph_records':[{'graph_key':k,'sources':v['sources'],'exact_typed_isomorphic':isomorphic_operator_graph(a['graph'],v['graph'])} for k,v in matched_records],'new_graphs_recovered':0,'remaining':'If matching record has no safe graph, require a graph-only certificate bound to original record/source/version; do not read mixed truth outputs.'})
  decisions=[];unresolved=[];dev=[];passes=Counter()
  for s in sources:
   sid=s['source_id'];ref=s.get('member_ref');matches=sorted(orbitset&member_sets.get(ref,set()));exact=[]
   if sid in indices:
    exact=indices[sid].match(a['graph']);relation='EXACT_OVERLAP' if exact else 'EXACT_DISJOINT_COMPLETE'
   elif s['use']==FREE:relation='EXACT_STRUCTURE_EXPOSURE_IN_C_EVENT'
   elif ref is None:relation='MEMBER_COVERAGE_UNRESOLVED'
   else:relation='COARSE_ORBIT_HIT' if matches else 'ORBIT_MISS'
   decision=source_decision(s,relation);decisions.append(decision);passes[decision]+=1
   gaps=s.get('gaps',[]) if decision=='PROVENANCE_UNRESOLVED' else []
   if decision=='PROVENANCE_UNRESOLVED':unresolved.append(sid)
   if decision=='DEVELOPMENT_USED':dev.append(sid)
   for gap in gaps:
    if aid not in gap_candidates[gap]:gap_candidates[gap].append(aid)
   matrix.append({'architecture_id':aid,'source_id':sid,'use':s['use'],'relation':relation,'decision':decision,'coarse_matches':matches,'exact_typed_matches':exact,'member_reference':ref,'association_precision':s.get('member_ref_relation','exact file/event membership'),'proof_registry_sha256':sourcehash,'orbit_sha256':orbits[aid]['sha256'],'gaps':gaps})
  eligibility.append({'architecture_id':aid,'graph_hash':a['graph_hash'],'classification':classify(decisions),'candidate_ids':[c['candidate_id'] for c in configs if c['architecture_fingerprint']==aid],'source_decision_counts':dict(passes),'development_sources':dev,'unresolved_source_count':len(unresolved),'unresolved_sources':unresolved,'coarse_history_hits':hits,'structure_exposure':['A/C frozen WL dictionary','fourth-round C generation and encoding event'],'reason':'See source-specific proof matrix; no missing-graph-total eligibility rule.'})
  write(root/'progress.json',{'completed_architectures':len(eligibility),'total_architectures':45,'new_permutations_this_invocation':physical,'updated_at':now()})
 total=sum(o['permutation_count'] for o in orbits.values());assert total<=5400
 amap={a['architecture_id']:a for a in eligibility};qualified=[c for c in configs if amap[c['architecture_fingerprint']]['classification']=='ELIGIBLE_WITH_EVIDENCE'];n=len(qualified)
 byconfig=[{'candidate_id':c['candidate_id'],'architecture_id':c['architecture_fingerprint'],'classification':amap[c['architecture_fingerprint']]['classification'],'graph_hash':c['graph_hash']} for c in configs]
 result={'architectures':eligibility,'configurations':byconfig,'architecture_counts':dict(Counter(a['classification'] for a in eligibility)),'configuration_counts':dict(Counter(c['classification'] for c in byconfig)),'state':'NO_QUALIFIED_C' if not n else ('QUALIFIED_C_READY_FOR_PROTOCOL' if n==180 else 'PARTIAL_QUALIFIED_C'),'scope_identity':lock['identity'],'old_eligibility_modified':False}
 write(root/'orbit_manifest.json',{'schema':SCHEMA,'architectures':orbits,'enumerated_architectures':45,'coarse_hash_evaluations':total,'maximum_budget':5400,'new_evaluations_this_invocation':physical,'conditional_history_hits':unionhits,'all_permutations_kept':True})
 write(root/'candidate_source_matrix.json',matrix);csvwrite(root/'candidate_source_matrix.csv',matrix)
 write(root/'C_eligibility_manifest.json',result)
 write(root/'C_qualified_manifest.json',{'state':result['state'],'architecture_count':n//4,'configuration_count':n,'configurations':qualified,'original_manifest':P4+'/C_preparation/C_unlabeled_manifest.json','original_manifest_sha256':inputs[P4+'/C_preparation/C_unlabeled_manifest.json'],'original_feature_sha256':inputs[P4+'/C_preparation/C_features.json'],'dictionary_sha256':inputs[P4+'/C_preparation/frozen_dictionary.json'],'representation_reselected':False,'independent_performance_evidence':False})
 write(root/'targeted_graph_evidence.json',{'coarse_hit_candidates':recovery,'new_graph_recoveries':0,'forbidden_source_opened':False})
 gaps=[{'id':'G1_HISTORICAL_VERSION_AND_ASSOCIATION','candidate_architecture_ids':gap_candidates['G1_HISTORICAL_VERSION_AND_ASSOCIATION'],'missing':'Bind original record to exact source file, original producer code/version and stored-vs-recomputed fingerprint branch. Group-only links and 85 metadata files without source code binding do not establish this.','location':[H+'/phase_a/exposure_source_inventory.json',H+'/phase_a/exposure_registry.json','adjacent history_metadata.json','original historical source snapshots or pre-existing access/member ledgers'],'minimal_sufficient_resolution':'For the relevant member collections, provide a safe identity-only certificate of producer/schema and full membership. Complete-orbit misses then need no historical graph recovery. For a coarse hit, provide only the matched typed graph and use evidence.'},
 {'id':'G2_SCOPE_MEMBER_CLOSURE','candidate_architecture_ids':gap_candidates['G2_SCOPE_MEMBER_CLOSURE'],'missing':'No complete per-development-run member/use linkage up to cutoff, including subsequent references of historical generation pools. Directory list and extraction union are not an exhaustive use/member ledger.','location':[P4+'/final_checks/prior_historical_use_metadata.json',H+'/phase_a/exposure_source_inventory.json',ORIGIN+'/exposure_source_inventory.json'],'minimal_sufficient_resolution':'Safe per-run member architecture list tied to run/version/use, or proof that missing runs only reference already-covered disjoint member sets. No scores, all-history graph reconstruction or new training required.'}]
 write(root/'minimal_missing_evidence.json',gaps)
 summary={'calculation_state':'COMPLETE','eligibility_state':result['state'],'source_count':len(sources),'original_historical_files':registry['schema_evidence']['original_file_count'],'verified_byte_schemas':1,'schema_bound_sources':sum(bool(s.get('schema_verified')) for s in sources),'complete_membership_sources':sum(bool(s.get('coverage_complete')) for s in sources),'use_counts':registry['uses'],'architecture_counts':result['architecture_counts'],'configuration_counts':result['configuration_counts'],'enumerated_architectures':45,'coarse_hash_evaluations':total,'conditional_coarse_hit_architectures':len(unionhits),'unique_conditional_historical_hashes_hit':len({h for r in unionhits for h in r['coarse_hashes']}),'exact_development_overlap_architectures':result['architecture_counts'].get('DEVELOPMENT_USED',0),'new_graph_recoveries':0,'matrix_rows':len(matrix),'decision_counts':dict(Counter(r['decision'] for r in matrix)),'remaining_material_gap_classes':sum(bool(g['candidate_architecture_ids']) for g in gaps),'feature_identity_valid':True,'qualified_configurations':n,'minimum_nonrepeating_original_search_budget':71,'missing_configurations_for_71':max(0,71-n),'future_GNN_budget_formula':'U × R only; no jobs launched','operations':ZERO,'access_guard':counts,'elapsed_seconds':time.time()-started,'scope_identity':lock['identity'],'retained_method':'W-I'}
 write(root/'summary.json',summary)
 for key in ('orbits','qualification'):state[key]='COMPLETE'
 state['report']='PENDING';write(root/'module_status.json',state)
 names=['summary.json','C_eligibility_manifest.json','C_qualified_manifest.json','candidate_source_matrix.json','candidate_source_matrix.csv','orbit_manifest.json','minimal_missing_evidence.json','targeted_graph_evidence.json','candidate_identity_checks.json','source_schema_and_use_registry.json','source_member_sets.json','source_full_graphs.json','record_source_associations.json','source_identity_mapping.json','schema_evidence.json']
 names+= [str(p.relative_to(root)) for p in sorted((root/'orbits').glob('*.json'))]
 write(root/'audit_complete.json',{'run_identity':digest(identity),'files':{p:sha(root/p) for p in names},'completed_at':now()})
 print(canonical(summary))
