"""Explicit reference followups: integrity metadata and three matching records."""
import io,json,hashlib,subprocess
from .common import *
from .projector import Projector,objects

AUDIT_FIELDS={'passed','expected_task_count','observed_task_count','duplicate_task_ids','missing_task_ids','unexpected_task_ids','failed_task_ids','configuration_drift_task_ids','protected_files_unchanged','test_label_used','formal_bo_executed','protocol_path','protocol_sha256','selection_path','selection_sha256','exposure_registry_path','exposure_registry_sha256','exposure_inventory_path','exposure_inventory_sha256','configuration_count','architecture_count','training_seeds','task_ids','task_count','n_init','n_iter','low_fidelity_epochs','low_fidelity_patience','formal_source_id','formal_config_fingerprint','layernorm','use_layernorm','gnn_layernorm','normalization_mode','source_hash','source_file_hashes','invocation','command','argv'}
class BindingProjector(Projector):
 def field_value(self,key,parent):
  if key in AUDIT_FIELDS:return self.value('literal',key)
  if key in {'hp','args','input_identities','search_config','evaluation_config','configuration'}:return self.value('project',key)
  return super().field_value(key,parent)

def freeze_followups(root,repo):
 manifest=read(root/'projection_manifest.json');paths={};targets={}
 for src,item in manifest.items():
  d=read(root/item['path']);hits=d['identity_summary']['coarse_hits']
  if not hits:continue
  targets[src]=[int(h['pointer'].split('/')[1]) for h in hits]
  paths[src]={'purpose':'Only matching record effective GAT/GIN/GCNII attributes; no scores','profile':'targeted_structure','indices':targets[src],'expected_sha256':d['actual_sha256'],'schema_evidence':'bo_phase4.py low-fidelity row hp/config persistence; prior identity projection proves target positions'}
  for name in ('initialization_config.json','gmm_schur_lowfid_initialization_config.json','history_metadata.json','gmm_schur_initialization_config.json'):
   p=Path(src).parent/name
   if (repo/p).is_file():paths[str(p)]={'purpose':'Actual matched search run settings and effective operator attributes','profile':'metadata','schema_evidence':'bo_phase4 initialization configuration writer; linked source sibling','expected_sha256':manifest.get(str(p),{}).get('source_sha256')}
 for p in [STRICT+'/result_audit.json',H+'/gnn_run_manifest.json']:
  paths[p]={'purpose':'Actual task coverage and frozen code/member-pool binding','profile':'metadata','schema_evidence':'evaluate_performance_aware_vae_strict_regression.py::evaluate writes only integrity counts/IDs/flags; h_vs_head::task_plan writer writes frozen run identities','expected_sha256':manifest.get(p,{}).get('source_sha256')}
 definition={'paths':paths,'extra_identity_fields':sorted(AUDIT_FIELDS),'allowed_extra_containers':['hp','args','input_identities','search_config','evaluation_config','configuration'],'frozen_at':now(),'basis':'New exact references from completed P1 projection and inspected metadata writer code; no dedicated score/evaluation output admitted.'}
 if (root/'followup_allowlist.json').exists():return read(root/'followup_allowlist.json')
 write(root/'followup_allowlist.json',definition);return definition

def run_followups(root,repo):
 if (root/'followup_complete.json').exists():verify(root,read(root/'followup_complete.json')['files']);return read(root/'followup_evidence.json')
 definitions=freeze_followups(root,repo)
 # Synthetic acceptance before touching newly admitted files/fields.
 probe={'hp':{'gat_heads':1,'gin_eps':0.0,'lr':.3,'val_acc':.999},'expected_task_count':4,'observed_task_count':4,'passed':True,'predictions':{'score':.888},'test_acc':.777}
 parser=BindingProjector(io.BytesIO(json.dumps(probe).encode()));v=parser.all()
 assert v=={'hp':{'gat_heads':1,'gin_eps':0.0},'expected_task_count':4,'observed_task_count':4,'passed':True}
 write(root/'followup_synthetic_test.json',{'passed':True,'only_synthetic_records':True,'performance_fields_retained':False,'code_sha256':sha(Path(__file__))})
 result={};total={k:0 for k in ['mixed_file_byte_reads','hash_reads','bytes_read','performance_field_decodes','performance_field_exposures','performance_field_writes','performance_field_uses']}
 for rel,spec in definitions['paths'].items():
  if not (repo/rel).is_file():result[rel]={'state':'MISSING'};continue
  with (repo/rel).open('rb') as f:
   p=BindingProjector(f);value=p.all();c=p.counters()
  for k in total:total[k]+=c[k]
  if spec.get('expected_sha256') and c['sha256']!=spec['expected_sha256']:raise ValueError('Followup source changed: '+rel)
  if spec['profile']=='targeted_structure':value=[{'record_index':i,'record':value[i]} for i in spec['indices']]
  result[rel]={'state':'COMPLETE','source_sha256':c['sha256'],'identity_projection':value,'access':c,'profile':spec['profile']}
 write(root/'followup_evidence.json',result);write(root/'followup_access_counters.json',total)
 names=['followup_allowlist.json','followup_evidence.json','followup_access_counters.json','followup_synthetic_test.json'];write(root/'followup_complete.json',{'files':{p:sha(root/p) for p in names},'time':now()});return result
