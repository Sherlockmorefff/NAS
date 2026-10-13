"""Reuse the tested byte parser; traverse nested identity-bearing JSON containers.

Only explicitly named identity values are decoded. Unknown scalar values,
including performance/SE/rank/prediction values, never become Python values.
Traversal retains dictionary keys and empty shape, never raw score text.
"""
from analyse.c_provenance_evidence_closure.projector import Projector, IDENTITY, HASH_MAPS

EXTRA=set('lr dropout hidden_dim l2 raw_hp_vector component_variant activation_variant variant panel_role replicate_id final_eval_seed confirmation_seeds seeds _method _run_seed _run_name history_path history_sha256 common_seed_manifest common_seed_manifest_sha256 seed_scheme final_evaluation_seed_scheme input_file input_file_sha256 source_root source_run source_run_id source_history source_history_path source_history_sha256 candidate_manifest candidate_manifest_sha256 dataset_content_fingerprint split_fingerprint split_protocol training_mode code_hash freeze_commit evaluator_sha256 config_sha256 source_identity_sha256 protocol_sha256 candidate_ids configuration_ids source_candidate_id base_candidate_fingerprint architecture_hash graph_hash source_config_fingerprint frozen_pool_sha256 version completed status'.split())
FIELDS=(IDENTITY|EXTRA)-{'step','iteration','index','fingerprint_count'}

class IdentityProjector(Projector):
 def field_value(self,key,parent):
  if key in HASH_MAPS:
   self.decoded_fields[key]+=1;return self.value('hash_map',key)
  if key in FIELDS:
   self.decoded_fields[key]+=1;return self.value('literal',key)
  self.s.space()
  if self.s.peek() in (91,123):
   return self.value('project',key)
  self.skipped[key]+=1;self.skip();return None

def project(path):
 with open(path,'rb') as f:
  parser=IdentityProjector(f)
  value=parser.all(str(path).endswith('.jsonl'))
  return value,parser.counters()

