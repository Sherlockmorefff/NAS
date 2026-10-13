import ast,hashlib,subprocess
from collections import Counter,defaultdict
from .common import *
from .followups import run_followups
from .backups import recover
from analyse.c_independence_minimal_audit.identity import classify

def certificate_can_exclude(cert):
 return bool(cert.get('complete_membership') and cert.get('comparison_rule_proven') and cert.get('input_binding_verified'))
def local_decision(cert,orbit):
 if not certificate_can_exclude(cert):return 'PROVENANCE_UNRESOLVED'
 return 'HISTORICAL_OVERLAP_POSSIBLE' if set(cert['structure_ids'])&orbit else 'PASS_BOUND_MEMBER_SUPERSET'
def code_evidence(repo,path,names):
 text=(repo/path).read_text();lines=text.splitlines();tree=ast.parse(text)
 return {'path':path,'sha256':sha(repo/path),'functions':[{'name':n.name,'start_line':n.lineno,'source':'\n'.join(lines[n.lineno-1:n.end_lineno])} for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]}

def review(root,repo,resume=False):
 if (root/'review_complete.json').exists():
  assert resume;verify(root,read(root/'review_complete.json')['files']);verify(repo,read(root/'safe_input_manifest.json'))
  write(root/'review_resume.json',{'state':'PASS','new_evidence_collections':0,'new_enumerations':0,'new_graph_recoveries':0,'new_model_operations':0,'qualification_and_core_hashes_unchanged':True,'time':now()});print(canonical(read(root/'review_resume.json')));return
 verify(root,read(root/'collection_complete.json')['files']);follow=run_followups(root,repo);backups=recover(root,repo)
 state=read(root/'phase_status.json');state['P2']='RUNNING';write(root/'phase_status.json',state)
 mf=read(root/'projection_manifest.json');data={p:read(root/v['path']) for p,v in mf.items()};safe=read(root/'safe_input_manifest.json');oldregistry=read(repo/OLD/'source_schema_and_use_registry.json');sources={s['source_id']:s for s in oldregistry['sources']}
 certs={};covered={};allstructures=set();source_notes=[]
 for src,d in data.items():
  s=d.get('identity_summary',{});allstructures.update(s.get('concrete_structure_ids',[]))
  if d['state']!='COMPLETE' or not s.get('records_with_supported_structure') or s['unresolved_top_records']!=0:continue
  cert={'id':'DIRECT:'+digest(src),'path':'C: complete concrete recorded structural payloads','source':src,'source_sha256':d['actual_sha256'],'identity_projection':mf[src]['path'],'identity_projection_sha256':mf[src]['output_sha256'],'complete_membership':True,'comparison_rule_proven':True,'input_binding_verified':d['historical_hash_matches'],'structure_ids':s['concrete_structure_ids'],'record_count':s['record_count'],'scope':'All identity-bearing records in this exact file, not a claim all observations of the experiment were written here. Additional run observation files assessed separately.','derivation':['Streaming entire frozen file completed without decoding metrics.','Every top record has supported concrete operations/edges, or is explicitly invalid with no architecture.','Direct ops/edge projection is a necessary consequence of typed graph isomorphism; negative match needs no old stored-fingerprint schema assumption.'],'old_stored_fingerprint_version_required':False,'use':'USE_UNRESOLVED','use_note':'Disjointness suffices; presence in a cache is not promoted to a global performance-use declaration.'}
  certs[cert['id']]=cert;covered[src]=cert['id']
  if src not in sources:sources[src]={'source_id':src,'use':'USE_UNRESOLVED','gaps':['NEWLY_INCLUDED_OBSERVATION_COLLECTION']}
 # Actual completed strict task group: plan -> code hashes -> complete inputs
 # and splits -> no-extra-task integrity audit. Never open task score files.
 planpath=STRICT+'/task_plan.json';plan=data[planpath]['identity_projection'];labelpath=str(Path(plan['inputs']['labels']).relative_to(repo));splitpath=str(Path(plan['fixed_split_manifest']).relative_to(repo));audit=follow[STRICT+'/result_audit.json']['identity_projection'];label=data[labelpath]['identity_summary']
 producer_paths=['analyse/performance_aware_vae_strict_regression.py','analyse/evaluate_performance_aware_vae_strict_regression.py']
 checks={'task_plan_sidecar':(repo/(STRICT+'/task_plan.sha256')).read_text().split()[0]==data[planpath]['actual_sha256'],'label_input_hash':plan['input_hashes'][plan['inputs']['labels']]==data[labelpath]['actual_sha256'],'split_hash':plan['fixed_split_manifest_sha256']==data[splitpath]['actual_sha256'],'code_bindings':all(plan['code_hashes'][str(repo/p)]==safe[p] for p in producer_paths),'actual_completed_task_count':audit['expected_task_count']==audit['observed_task_count']==len(plan['tasks'])==1680,'actual_no_missing_extra_or_drift':audit['passed'] and all(audit[k]==[] for k in ['duplicate_task_ids','missing_task_ids','unexpected_task_ids','failed_task_ids','configuration_drift_task_ids']),'full_concrete_members':label['record_count']==label['records_with_supported_structure']==76 and len(label['concrete_structure_ids'])==60}
 taskids={t['task_id'] for t in plan['tasks']};group_sources=[p for p in sources if p.startswith(STRICT+'/task_results/') or p.startswith(STRICT+'/evaluation_task_cache/')]
 checks['all_inventory_task_names_in_frozen_plan']=all(Path(p).stem in taskids for p in group_sources)
 poolcert={'id':'GROUP:strict_regression_frozen_60','path':'B/C: run-bound common member superset','source':STRICT,'source_sha256':data[planpath]['actual_sha256'],'complete_membership':all(checks.values()),'comparison_rule_proven':True,'input_binding_verified':all(checks.values()),'structure_ids':label['concrete_structure_ids'],'checks':checks,'covered_original_source_files':group_sources,'frozen_task_count':1680,'member_configurations':76,'member_architectures':60,'actual_audit':STRICT+'/result_audit.json','actual_audit_sha256':follow[STRICT+'/result_audit.json']['source_sha256'],'member_source':labelpath,'member_source_sha256':data[labelpath]['actual_sha256'],'split_source':splitpath,'split_sha256':data[splitpath]['actual_sha256'],'use':'PERFORMANCE_DEVELOPMENT','derivation':['verify_frozen_plan validates exact plan, code and input identities.','train_task obtains train/validation rows only by IDs in the frozen split from the 76-row input.','evaluate checks exact task IDs/count/metadata and reuses those same labels and split.','Actual integrity audit passed with 1680 tasks and no missing/extra/drift.','All 2688 referenced task/cache filenames are in that frozen plan; their performance payloads were not opened.'],'label_free_other_input':'Unlabelled reconstruction replay is distinct from performance target members; no replay dataset or checkpoint loaded.'}
 certs[poolcert['id']]=poolcert
 if certificate_can_exclude(poolcert):
  for src in group_sources:covered[src]=poolcert['id']
 codeproof=[code_evidence(repo,producer_paths[0],['verify_frozen_plan','train_task','_train_phase']),code_evidence(repo,producer_paths[1],['evaluate'])]
 write(root/'group_producer_code_evidence.json',codeproof)
 # A exact old eval_utils blob is found, but the matching historical BO caller
 # is not; do not invent its effective GCNII overrides or downstream use.
 version=read(root/'historical_version_search.json');ev=next(x for x in version['bindings'] if x['recorded_code_path']=='eval_utils.py');bo=next(x for x in version['bindings'] if x['recorded_code_path']=='bo_phase4.py')
 targeted=[]
 for src,d in follow.items():
  if d.get('profile')!='targeted_structure':continue
  for obj in d['identity_projection']:
   r=obj['record'];targeted.append({'source':src,'source_sha256':d['source_sha256'],'record_index':obj['record_index'],'coarse_fingerprint':r['architecture_fingerprint'],'stored_effective_attributes':r.get('hp',{}),'recorded_operation':{'valid':r['valid'],'evaluation_stage':r['evaluation_stage'],'evaluation_fidelity':r['evaluation_fidelity'],'initialization_strategy':r['initialization_strategy']},'exact_typed_isomorphism':'NOT_ESTABLISHED','classification':'HISTORICAL_OVERLAP_POSSIBLE','known_evaluator_source_sha256':ev['recorded_sha256'],'known_evaluator_git_matches':ev['exact_git_matches'],'missing':['Exact BO caller snapshot matching '+bo['recorded_sha256'],'Actual effective GCNII alpha/theta and complete activation/merge/normalization provenance at this call; defaults alone are not runtime argument proof.','Run-bound downstream performance-use linkage sufficient for DEVELOPMENT_USED.'],'no_default_attributes_invented':True})
 if ev['exact_git_matches']:
  text=subprocess.check_output(['git','show',ev['exact_git_matches'][0]+':eval_utils.py'],cwd=repo,text=True);assert hashlib.sha256(text.encode()).hexdigest()==ev['recorded_sha256']
  out=root/'code_evidence'/'historical_eval_utils.py';out.parent.mkdir(exist_ok=True);out.write_text(text)
 write(root/'targeted_overlap_review.json',{'records':targeted,'unique_coarse_structures':len({r['coarse_fingerprint'] for r in targeted}),'full_effective_graphs_confirmed':0,'new_full_typed_graph_recoveries':0,'collection_counter_note':'P1 field new_full_graphs_recovered names saved coarse-hit identity/structure packets. It is not a count of established full effective graphs; the verified count is zero here.'})
 state['P2']='COMPLETE';state['P3']='RUNNING';write(root/'phase_status.json',state)
 write(root/'producer_schema_certificates.json',{'certificates':list(certs.values()),'certificates_are_new_indexes_of_original_evidence_not_historical_attestations':True,'explicit_historical_version_search':'historical_version_search.json','backup_search':'backup_search.json'})
 write(root/'development_member_use_coverage.json',{'covered_sources':covered,'group_certificate':poolcert,'sources_without_new_certificate':[p for p in sources if p not in covered and not p.startswith('coverage:')],'old_passes_are_reused_separately':True,'post_cutoff':{'old_audit_archive_only':all(r['path']==OLD+'.tar.gz' for r in read(root/'post_cutoff_inventory.json')['modified_non_audit_project_files']),'recorded_new_performance_runs':0,'evidence':'post_cutoff_inventory.json','limit':'No assertion about unlogged or deleted runs.'}})
 oldmatrix=read(repo/OLD/'candidate_source_matrix.json');old_by=defaultdict(dict)
 for row in oldmatrix:old_by[row['architecture_id']][row['source_id']]=row
 original=read(repo/P4/'C_preparation/C_unlabeled_manifest.json');om=read(repo/OLD/'orbit_manifest.json');matrix=[];archresults=[];changes=[];gap_sources=defaultdict(list)
 for a in original['architectures']:
  aid=a['architecture_fingerprint'];orb=read(repo/OLD/om['architectures'][aid]['path']);orbit=set(orb['unique_coarse_hashes']);rows=[]
  for sid,source in sources.items():
   if sid.startswith('coverage:'):continue
   previous=old_by[aid].get(sid);cert=certs.get(covered.get(sid));prior=previous['decision'] if previous else 'NOT_PREVIOUSLY_INCLUDED'
   if cert:
    decision=local_decision(cert,orbit);proof=cert['id'];relation='COARSE_MATCH_REQUIRES_EFFECTIVE_ATTRIBUTES' if decision=='HISTORICAL_OVERLAP_POSSIBLE' else 'BOUND_COMPLETE_MEMBER_SUPERSET_DISJOINT'
   else:
    decision=prior if previous else 'PROVENANCE_UNRESOLVED';proof='REUSED_OLD_LOCAL_PROOF' if previous and prior.startswith('PASS_') else 'UNRESOLVED_ORIGINAL_SOURCE';relation=previous['relation'] if previous else 'NO_COMPLETE_MEMBER_PROOF'
   rec={'architecture_id':aid,'source_id':sid,'use':cert.get('use','USE_UNRESOLVED') if cert else source.get('use','USE_UNRESOLVED'),'decision':decision,'previous_decision':prior,'relation':relation,'proof':proof,'evidence_sha256':digest(cert) if cert else sha(repo/OLD/'source_schema_and_use_registry.json')}
   rows.append(rec)
   if decision!=prior:changes.append(rec)
   if decision in ('PROVENANCE_UNRESOLVED','HISTORICAL_OVERLAP_POSSIBLE'):gap_sources[sid].append(aid)
  unresolved=[r['source_id'] for r in rows if r['decision'] in ('PROVENANCE_UNRESOLVED','HISTORICAL_OVERLAP_POSSIBLE')]
  # This scope row is derived from concrete remaining source gaps; it is not
  # an unconditional global veto or a missing-history-graph-count check.
  rows.append({'architecture_id':aid,'source_id':'coverage:project_development_members','use':'USE_UNRESOLVED','decision':'PROVENANCE_UNRESOLVED' if unresolved else 'PASS_ALL_RELEVANT_SOURCE_OBLIGATIONS','previous_decision':'PROVENANCE_UNRESOLVED','relation':'DERIVED_FROM_REMAINING_RELEVANT_SOURCES','proof':'remaining_gaps.csv','evidence_sha256':digest(unresolved)})
  classification=classify(['PROVENANCE_UNRESOLVED' if r['decision']=='HISTORICAL_OVERLAP_POSSIBLE' else r['decision'] for r in rows])
  archresults.append({'architecture_id':aid,'classification':classification,'candidate_ids':[c['candidate_id'] for c in original['configurations'] if c['architecture_fingerprint']==aid],'coarse_overlap_possible':any(r['decision']=='HISTORICAL_OVERLAP_POSSIBLE' for r in rows),'unresolved_source_count':len(unresolved),'source_decision_counts':dict(Counter(r['decision'] for r in rows))});matrix.extend(rows)
 amap={r['architecture_id']:r for r in archresults};configresults=[{'candidate_id':c['candidate_id'],'architecture_id':c['architecture_fingerprint'],'classification':amap[c['architecture_fingerprint']]['classification'],'coarse_overlap_possible':amap[c['architecture_fingerprint']]['coarse_overlap_possible']} for c in original['configurations']];qualified=[c for c in original['configurations'] if amap[c['architecture_fingerprint']]['classification']=='ELIGIBLE_WITH_EVIDENCE'];n=len(qualified)
 csvwrite(root/'candidate_source_matrix.csv',matrix);write(root/'candidate_source_matrix.json',matrix);csvwrite(root/'evidence_decision_changes.csv',changes)
 eligibility={'architectures':archresults,'configurations':configresults,'architecture_counts':dict(Counter(x['classification'] for x in archresults)),'configuration_counts':dict(Counter(x['classification'] for x in configresults)),'state':'NO_QUALIFIED_C' if n==0 else 'QUALIFIED_C_READY_FOR_PROTOCOL' if n==180 else 'PARTIAL_QUALIFIED_C','old_audit_unchanged':True}
 write(root/'C_eligibility_manifest.json',eligibility);write(root/'C_qualified_manifest.json',{'architecture_count':n//4,'configuration_count':n,'configurations':qualified,'original_manifest_sha256':safe[P4+'/C_preparation/C_unlabeled_manifest.json'],'original_feature_sha256':safe[P4+'/C_preparation/C_features.json'],'frozen_dictionary_sha256':safe[P4+'/C_preparation/frozen_dictionary.json'],'representation_changed':False})
 gaps=[]
 for sid,aids in sorted(gap_sources.items()):
  overlap=any(t['source']==sid for t in targeted)
  gaps.append({'source':sid,'affected_architectures':len(aids),'architecture_ids':aids,'gap':'MATCHED_EFFECTIVE_ATTRIBUTES_AND_USE' if overlap else 'MEMBER_OR_VERSION_COVERAGE_UNRESOLVED','checked_locations':[sid,'frozen source registry','associated run metadata','631 frozen code snapshots','relevant Git code histories','four relevant known code archives'],'minimum_material':targeted[0]['missing'] if overlap else 'A run/version-bound complete identity/member pool or a proved conservative superset for this source; group-level certificates are accepted. No score values needed.','judgment_if_resolved':'Exact typed/use match can exclude one architecture; proven mismatch releases this source obligation.' if overlap else 'Complete negative member comparison releases this source obligation; remaining sources still apply.','repeating_current_run_helps':False})
 csvwrite(root/'remaining_gaps.csv',gaps);write(root/'remaining_gaps.json',gaps)
 new={'state':'SUBSTANTIVE_NEW_EVIDENCE_FOUND','direct_complete_collections':sum(c['id'].startswith('DIRECT:') for c in certs.values()),'group_bound_source_files':len(group_sources) if certificate_can_exclude(poolcert) else 0,'strict_group_checks':checks,'newly_found_coarse_overlap_architectures':len({t['coarse_fingerprint'] for t in targeted}),'overlap_records':len(targeted),'historical_structure_ids_observed':len(allstructures),'new_source_collections':sum(sid not in next(iter(old_by.values())) for sid in sources),'changed_candidate_source_decisions':len(changes),'old_schema_guessing_used':False,'old_orbit_evaluations_reused':5304,'new_orbit_evaluations':0}
 write(root/'new_evidence_manifest.json',new)
 access=Counter(read(root/'collection_summary.json')['access_counters'])
 for d in [read(root/'followup_access_counters.json'),backups]:
  for k in ['mixed_file_byte_reads','hash_reads','bytes_read','performance_field_decodes','performance_field_exposures','performance_field_writes','performance_field_uses']:access[k]+=d[k]
 summary={'calculation_state':'COMPLETE','evidence_state':'PARTIAL_EVIDENCE_CLOSURE','C_state':eligibility['state'],'qualified_architectures':n//4,'qualified_configurations':n,'architecture_counts':eligibility['architecture_counts'],'configuration_counts':eligibility['configuration_counts'],'coarse_overlap_possible_architectures':sum(x['coarse_overlap_possible'] for x in archresults),'coarse_overlap_possible_configurations':sum(x['coarse_overlap_possible'] for x in configresults),'confirmed_development_overlap_architectures':eligibility['architecture_counts'].get('DEVELOPMENT_USED',0),'source_count':len(sources),'remaining_concrete_source_gaps':len(gaps),'proof_matrix_rows':len(matrix),'decision_counts':dict(Counter(r['decision'] for r in matrix)),'new_evidence':new,'access_counters':dict(access),'operations':ZERO,'new_full_typed_graph_recoveries':0,'new_candidate_orbit_evaluations':0,'reused_candidate_orbit_evaluations':5304,'future_budget':'U × R only; U must be at least 71 for the original nonrepeating 50+20+1 schedule','next_stage_started':False,'retained_method':'W-I'}
 write(root/'summary.json',summary);state['P3']='COMPLETE';write(root/'phase_status.json',state)
 work=read(root/'evidence_worklist.json')
 for item in work:item['status']='BOUNDED_SEARCH_COMPLETE';item['result_reference']={'G1_SEARCH_RECORD_IDENTITY':'producer_schema_certificates.json','G1_SNAPSHOT_GIT_BINDING':'historical_version_search.json','G2_SHARED_DEVELOPMENT_POOLS':'development_member_use_coverage.json','G2_POST_CUTOFF':'post_cutoff_inventory.json'}[item['id']]
 write(root/'evidence_worklist.json',work)
 names=['producer_schema_certificates.json','development_member_use_coverage.json','group_producer_code_evidence.json','targeted_overlap_review.json','candidate_source_matrix.json','candidate_source_matrix.csv','evidence_decision_changes.csv','C_eligibility_manifest.json','C_qualified_manifest.json','remaining_gaps.csv','remaining_gaps.json','new_evidence_manifest.json','summary.json','evidence_worklist.json']
 write(root/'review_code_identity.json',{str(p.relative_to(repo)):sha(p) for p in (repo/'analyse/c_provenance_evidence_closure').glob('*.py')});write(root/'review_complete.json',{'files':{p:sha(root/p) for p in names},'time':now(),'run_identity':read(root/'run_identity.json')['identity']});print(canonical(summary))
