"""Freeze sources and rules before opening C for semantic inspection."""
import shutil,subprocess
from .common import *

PRODUCERS=['architecture_diagnostics.py','bo_phase4.py','initialization_wgmm_ted.py',
 'analyse/h_vs_head_model_blind_prospective.py','analyse/high_regime_prior_model_blind_validation.py',
 'analyse/original_head_deterministic_batch_pilot.py','analyse/original_head_confirmatory_attribution_v2.py',
 'analyse/structure_native_surrogate_pilot.py','analyse/structured_kernel_phase1/features.py',
 'analyse/structured_kernel_arch_main_ablation/c_prepare.py','analyse/structured_kernel_arch_main_ablation/c_features.py',
 'analyse/structured_kernel_arch_main_ablation/runner.py','hp_modes.py','eval_utils.py']
def freeze(root,repo,prompt):
 if root.exists() and any(root.iterdir()):raise ValueError('Freeze needs a new empty directory')
 root.mkdir(parents=True);shutil.copyfile(prompt,root/'PROMPT.txt')
 paths=set(PRODUCERS)|set(FULL_SOURCES)
 paths|={P4+'/C_preparation/'+n for n in C_FILES}
 paths|={P4+'/'+n for n in ['README_RESULTS.md','GPT_ANALYSIS_REQUEST.md','method_selection.json','protocol.json','run_identity.json','source_manifest.json','C_completion.json','module_status.json','final_checks/prior_historical_use_metadata.json','final_checks/historical_gap_metadata_summary.json']}
 paths|={B+'/status.json',PREP+'/data_use.json'}
 for base in (ORIGIN,V1,V2,H):
  paths|={base+'/'+n for n in META_NAMES}
  paths|={base+'/'+n for n in ['pre_label_immutable_manifest.json','source_snapshot_audit.json','candidate_generation_audit.json','exposure_registry.json','exposure_source_inventory.json']}
  # Source snapshots are code only; never inspect adjacent model/evaluation files.
  paths|={str(p.relative_to(repo)) for p in (repo/base/'source_snapshot').rglob('*.py')}
 paths|={H+'/phase_a/'+n for n in ['exposure_registry.json','exposure_source_inventory.json','exposure_exclusion_audit.json']}
 inv=read(repo/H/'phase_a/exposure_source_inventory.json')
 for row in inv:
  if row['source_group'] in ('legacy_histories','original_ceiling_hunt'):
   p=Path(row['path']).parent/'history_metadata.json'
   if p.exists():paths.add(str(p.relative_to(repo)))
 prior=read(repo/P4/'final_checks/prior_historical_use_metadata.json')
 directories=[]
 for parent in [repo/'results',repo/'results/posthoc',repo/'results/search']:
  directories.extend(str(p.relative_to(repo)) for p in sorted(parent.iterdir()) if p.is_dir() and p!=root)
 existing={p:sha(repo/p) for p in sorted(paths) if (repo/p).is_file()}
 missing=sorted(paths-set(existing))
 inventory=read(repo/ORIGIN/'exposure_source_inventory.json')
 scope={'cutoff_utc':now(),'repository':'/home/fangxy/D-VAE','head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
  'definition':'C architecture and every HP/seed performance result must not have participated in prior performance surrogate training, method selection/tuning, BO development or development evaluation within this project scope.',
  'label_free_exposure':'Unlabelled generation, unsupervised pretraining, encoding and dictionary exposure do not alone disqualify; subsequent uses remain separately assessed.',
  'prior_directory_inventory':prior['historical_directories_inventory'],'current_bounded_directory_inventory':directories,
  'historical_original_source_records':inv,'recent_original_source_records':inventory,
  'scope_rule':'Existing project experiment directories plus their frozen inventory references; no other users or whole-server scan. Directory presence is not member completeness.',
  'coverage_rule':'A missing member/version binding remains a specific coverage gap; no assumption that unlisted or inaccessible relevant development records were unused.',
  'conditional_recovery':'Only matched records with an independently documented label-free config source may be recovered. Never open mixed score/cache/history files. A new safe metadata certificate needs a new frozen run.',
  'missing_named_safe_inputs':missing}
 protocol={'version':'minimal-c-independence-1','method_retained':'W-I','architectures_expected':45,'configurations_expected':180,
  'allowed_fields':['unlabelled operations/edges/effective attributes','architecture/candidate/joint/HP identities with explicit types','frozen HP4 and latent12/WL vectors','source/member/use metadata','protocols and source code'],
  'forbidden_operations':['GP fit','GP optimization','GP prediction','GNN training','VAE update','decoder','new candidate generation','C performance or acquisition score','B labels/SE','node test score','KG'],
  'rules':['Full m! internal permutations; fixed input/output; carry complete node attributes including GCNII layer.',
   'One documented compact-ASCII sorted operations/edges SHA256 schema; do not try alternative serializations.',
   'A coarse orbit hit requires exact typed graph verification; a hit alone is not development use.',
   'An orbit miss passes only with verified producer/schema, identity type and complete source membership.',
   'Full-graph exact disjointness passes even if source use is unresolved.',
   'Historical missing graph total is never an eligibility condition.',
   'Unknown relevant versions, member coverage or downstream use must remain unresolved.',
   'Same architecture shares eligibility across four HPs; no new candidate or representation selection.'],
  'enumeration_budget_per_verified_schema':5400,'next_stage_automatically_authorized':False}
 write(root/'history_scope.json',scope);write(root/'protocol.json',protocol);write(root/'input_manifest.json',existing)
 write(root/'allowed_paths.json',sorted(existing))
 frozen={'created_at':now(),'definition_and_scope_sha256':sha(root/'history_scope.json'),'protocol_sha256':sha(root/'protocol.json'),'input_manifest_sha256':sha(root/'input_manifest.json'),'prompt_sha256':sha(root/'PROMPT.txt')}
 frozen['identity']=digest(frozen);write(root/'scope_lock.json',frozen)
 write(root/'module_status.json',{'freeze':'COMPLETE','synthetic_acceptance':'PENDING','sources':'PENDING','orbits':'PENDING','qualification':'PENDING','report':'PENDING'})
 write(root/'baseline.json',{'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True),'protected':{p:existing[p] for p in ['bo_phase4.py','eval_utils.py',P4+'/C_preparation/C_eligibility_manifest.json',B+'/status.json'] if p in existing}})
 print(canonical({'scope_frozen':frozen['identity'],'safe_inputs':len(existing),'historical_source_records':len(inv),'missing_safe_inputs':missing}))
