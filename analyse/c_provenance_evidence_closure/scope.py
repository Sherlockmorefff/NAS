import ast,datetime,os,shutil,subprocess
from collections import Counter
from .common import *

CODE=['architecture_diagnostics.py','bo_phase4.py','initialization_wgmm_ted.py','initialization_gmm_schur.py','eval_utils.py','hp_modes.py','models.py','nas_space.py',
 'analyse/h_vs_head_model_blind_prospective.py','analyse/structure_native_surrogate_pilot.py','analyse/performance_aware_vae_strict_regression.py','analyse/evaluate_performance_aware_vae_strict_regression.py','analyse/vae_label_expansion.py','analyse/original_head_deterministic_batch_pilot.py','analyse/original_head_confirmatory_attribution_v2.py','analyse/high_regime_prior_model_blind_validation.py','analyse/h_targeted_reliable_pair_validation.py']
def freeze(repo,prompt):
 cutoff=now();stamp=datetime.datetime.fromisoformat(cutoff).strftime('%Y%m%dT%H%M%SZ');root=repo/'results'/('C_provenance_evidence_closure_'+stamp);root.mkdir()
 shutil.copyfile(prompt,root/'PROMPT.txt');scope=read(repo/OLD/'history_scope.json');previous=read(repo/OLD/'input_manifest.json')
 safe={p:sha(repo/p) for p in previous if not p.endswith('/README_RESULTS.md') and not p.endswith('/GPT_ANALYSIS_REQUEST.md')}
 oldfiles=['README_RESULTS.md','GPT_ANALYSIS_REQUEST.md','NEXT_REQUIRED_STEP.md','minimal_missing_evidence.json','schema_evidence.json','history_scope.json','source_coverage_report.md','source_schema_and_use_registry.json','record_source_associations.json','targeted_graph_evidence.json','protocol.json','run_identity.json','input_manifest.json','orbit_manifest.json','audit_complete.json','C_eligibility_manifest.json','C_qualified_manifest.json','candidate_identity_checks.json','candidate_source_matrix.json','source_full_graphs.json','source_member_sets.json','summary.json','commands.md','directory_relocation.json','directory_coverage_details.json']
 for n in oldfiles:safe[OLD+'/'+n]=sha(repo/OLD/n)
 for p in (repo/OLD/'orbits').glob('*.json'):safe[str(p.relative_to(repo))]=sha(p)
 for p in CODE:
  if (repo/p).is_file():safe[p]=sha(repo/p)
 mixed={};excluded=[];search_parents=set();metadata=set();old_inventory=scope['historical_original_source_records']
 for row in old_inventory:
  p=Path(row['path']);rel=str(p.relative_to(repo));name=p.name
  if name.startswith('history_') and p.suffix=='.json' and row['source_group'] in ('legacy_histories','original_ceiling_hunt','layernorm_search_space'):
   mixed[rel]={'purpose':'historical performance-development member identities','profile':'BO_history_record_list','schema_evidence':'bo_phase4.py::_save,history_record,_low_fidelity_evaluate_candidate','fields':'identity + operations/edges/effective attributes only','group':row['source_group'],'recorded_sha256':row['sha256'],'recorded_fingerprint_count':row['fingerprint_count']};search_parents.add(p.parent)
  else:excluded.append({'path':rel,'reason':'Do not open model evaluation/seed/test outputs; seek a bound member pool or safe metadata instead.','group':row['source_group']})
 for parent in search_parents:
  for name in ['initialization_full_history.json','low_fidelity_history.json','online_history.json']:
   p=parent/name
   if p.is_file():mixed[str(p.relative_to(repo))]={'purpose':'other performance-development observations in the same bounded search run','profile':'BO_history_record_list','schema_evidence':'bo_phase4.py::_save and initialization/online persistence paths','fields':'identity + operations/edges/effective attributes only','group':'additional_search_observations','recorded_sha256':None}
  for ancestor in (parent,parent.parent,parent.parent.parent):
   for name in ['history_metadata.json','dataset_manifest.json','candidate_pool_metadata.json','initialization_config.json','gmm_schur_lowfid_initialization_config.json','gmm_schur_initialization_config.json','git_commit.txt','source_sha256.txt']:
    p=ancestor/name
    if p.is_file():metadata.add(str(p.relative_to(repo)))
 # Exact recorded paths only, no directory-wide permission to mixed files.
 for row in scope['recent_original_source_records']['inventory']:
  rel=str(Path(row['path']).relative_to(repo))
  if rel in safe:continue
  if Path(rel).name in ['frozen_selection_slot_labels.json']:
   excluded.append({'path':rel,'reason':'Dedicated label/model evaluation output; forbidden','group':'recent'});continue
  mixed[rel]={'purpose':'historical development member pool or configuration-reference selection','profile':'member_config_or_reference_collection','schema_evidence':'structure_native_surrogate_pilot.py; h_vs_head_model_blind_prospective.py; corresponding frozen inventory','fields':'identities, operations, edges, effective attributes and configuration references','group':'recent_member_sources','recorded_sha256':row.get('sha256')}
 # Task plans and running metadata may contain diagnostic numbers; they also
 # go through the same projection, never unrestricted JSON loading.
 additions=[STRICT+'/task_plan.json',STRICT+'/run_environment.json',STRICT+'/dry_run_report.json',H+'/input_identities.json',H+'/gnn_run_manifest.json',H+'/protocol.json',STRUCT+'/provenance/source_manifest.json',STRUCT+'/provenance/source_identity.json']
 for rel in additions:
  if (repo/rel).is_file():mixed[rel]={'purpose':'bind shared task members, producer version and actual execution','profile':'run_metadata_and_hash_maps','schema_evidence':'strict-regression verify_frozen_plan,run_shard; H protocol writer; structure-native provenance writer','fields':'run/code/hash/input references only','group':'run_binding','recorded_sha256':None}
 for rel in metadata:
  if rel.endswith('.json'):mixed[rel]={'purpose':'search run and producer binding','profile':'run_metadata_and_hash_maps','schema_evidence':'bo_phase4.py::_save and initialization provenance writers','fields':'run/code/hash/input references only','group':'run_binding','recorded_sha256':None}
  else:safe[rel]=sha(repo/rel)
 for rel in ['legacy_artifacts/pre_20260813/migration_manifest.json','legacy_artifacts/pre_20260813/deletion_manifest.json',STRICT+'/task_plan.sha256',STRICT+'/fixed_split_manifest.sha256']:
  if (repo/rel).is_file():safe[rel]=sha(repo/rel)
 # Resolve the one explicitly linked known backup mechanism: code snapshots
 # beneath experiment roots already in scope. List/hash code only, not data.
 snapshot_paths=[];archive_paths=[]
 roots=set(scope['prior_directory_inventory'])|{str(p.parent.relative_to(repo)) for p in search_parents}
 for base in sorted(roots):
  if not (repo/base).is_dir():continue
  for d in [repo/base/'source_snapshot',repo/base/'provenance/source_snapshot',repo/base/'code_snapshot']:
   if d.is_dir():
    for p in d.rglob('*.py'):
     rel=str(p.relative_to(repo));safe[rel]=sha(p);snapshot_paths.append(rel)
  for p in (repo/base).iterdir():
   if p.is_file() and p.name.endswith(('.tar.gz','.zip','.tar')):archive_paths.append(str(p.relative_to(repo)))
 # No change after the old cutoff in project result/log/artifact files,
 # excluding the completed audit itself. This is bounded recorded activity,
 # not a claim about unlogged/deleted actions.
 since=datetime.datetime.fromisoformat(scope['cutoff_utc']).timestamp();new=[]
 for base in ['results','logs','artifacts']:
  for folder,dirs,files in os.walk(repo/base):
   dirs[:]=[d for d in dirs if d not in ['.git','__pycache__','node_modules']]
   for name in files:
    p=Path(folder)/name
    if p.is_relative_to(root) or p.is_relative_to(repo/OLD):continue
    st=p.stat()
    if st.st_mtime>since:new.append({'path':str(p.relative_to(repo)),'size':st.st_size,'mtime_ns':st.st_mtime_ns})
 mixed={p:v for p,v in mixed.items() if p not in safe}
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();status=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
 gitlog=subprocess.check_output(['git','log','--format=%H %aI %s','--','architecture_diagnostics.py','bo_phase4.py','initialization_wgmm_ted.py'],cwd=repo,text=True)
 protocol={'definition':scope['definition'],'old_cutoff':scope['cutoff_utc'],'cutoff':cutoff,'retained_method':'W-I','stages':['P0','P1','P2','P3','P4'],'permitted_proofs':['actual record binding','bound group schema + complete or conservative member superset','complete actual graph/member pool exclusion'],'prohibitions':ZERO,'mixed_access':'Only exact frozen paths; streaming identity projection; never decode skipped values. Dedicated B/C/seed_results/model-evaluation/node-test outputs forbidden.','candidate_orbits':'Reuse previous 5304 checks; never re-enumerate without a newly evidenced schema.','new_references':'Metadata/code may extend worklist only through explicit new references, recorded before access. New mixed files require appended exact-path freeze and schema rationale before opening.','stop':'Finish bounded worklist; no repeated identical searches; no next performance stage.'}
 work=[{'id':'G1_SEARCH_RECORD_IDENTITY','affects_architectures':45,'facts_needed':'Complete actual config/member projection or bound historical producer rules; avoid requiring all missing graphs','priority_paths':sorted(p for p,v in mixed.items() if v['profile']=='BO_history_record_list'),'alternative':'Group safe superset from true config records; no old fingerprint version required when concrete recorded structures suffice','status':'PENDING'},
 {'id':'G1_SNAPSHOT_GIT_BINDING','affects_architectures':45,'facts_needed':'Run-bound source hashes, matching original Git blobs or stored snapshots','priority_paths':snapshot_paths+[p for p in safe if p.endswith(('git_commit.txt','source_sha256.txt'))],'alternative':'Concrete member graph path','status':'PENDING'},
 {'id':'G2_SHARED_DEVELOPMENT_POOLS','affects_architectures':45,'facts_needed':'Actual run/plan code binding proves all performance members are selected from frozen pool, other inputs accounted for','priority_paths':[STRICT+'/task_plan.json',STRICT+'/fixed_split_manifest.json',H+'/gnn_run_manifest.json',STRUCT+'/phase_4/unbiased_candidate_pool.json'],'alternative':'Exact member identity projection','status':'PENDING'},
 {'id':'G2_POST_CUTOFF','affects_architectures':45,'facts_needed':'Recorded performance activity from old cutoff until new cutoff','priority_paths':['results','logs','artifacts'],'alternative':'Record uncovered/unlogged activity as explicit scope limitation','status':'PENDING'}]
 write(root/'protocol.json',protocol);write(root/'safe_input_manifest.json',safe);write(root/'mixed_allowlist.json',mixed);write(root/'excluded_sources.json',excluded);write(root/'evidence_worklist.json',work)
 write(root/'post_cutoff_inventory.json',{'from':scope['cutoff_utc'],'to':cutoff,'modified_non_audit_project_files':new,'interpretation':'No recorded artifact changes' if not new else 'Inspect listed run metadata; no inference of absent activity from names alone','limitation':'No claim to detect unlogged or deleted actions.'})
 write(root/'source_discovery.json',{'snapshot_code_files':snapshot_paths,'archive_paths_not_opened':archive_paths,'scope_directories':sorted(roots),'missing_requested_previous_source_manifest':not (repo/OLD/'source_manifest.json').exists(),'actual_previous_input_manifest':OLD+'/input_manifest.json','previous_path_alias':read(repo/OLD/'directory_relocation.json')})
 (root/'git_history.txt').write_text(gitlog);(root/'workspace_status_before.txt').write_text(status)
 (root/'workspace_diff_before.patch').write_text(subprocess.check_output(['git','diff','--','bo_phase4.py','eval_utils.py'],cwd=repo,text=True))
 identity={'created_at':cutoff,'HEAD':head,'safe_inputs':safe,'mixed_allowlist_sha256':sha(root/'mixed_allowlist.json'),'protocol_sha256':sha(root/'protocol.json'),'prompt_sha256':sha(root/'PROMPT.txt'),'old_scope_identity':read(repo/OLD/'scope_lock.json')['identity'],'workspace_status_sha256':sha(root/'workspace_status_before.txt')};identity['identity']=digest(identity);write(root/'run_identity.json',identity)
 write(root/'phase_status.json',{'P0':'COMPLETE','P1':'PENDING','P2':'PENDING','P3':'PENDING','P4':'PENDING'})
 write(root/'access_counters.json',{'mixed_file_byte_reads':0,'hash_reads':0,'performance_field_decodes':0,'performance_field_exposures':0,'performance_field_writes':0,'performance_field_uses':0,'model_operations':ZERO})
 print(canonical({'run_dir':str(root),'safe_inputs':len(safe),'frozen_mixed_paths':len(mixed),'post_cutoff_non_audit_files':len(new),'source_snapshots':len(snapshot_paths),'known_archives':len(archive_paths),'identity':identity['identity']}))
