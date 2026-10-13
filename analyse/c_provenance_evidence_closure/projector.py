"""Streaming JSON identity projection. Unknown values are skipped, never decoded.

No json.load/read_text of mixed inputs. Keys are decoded; only explicit identity
fields have values decoded. Even strings within a skipped subtree stay bytes.
"""
import hashlib,json,re
from collections import Counter

IDENTITY=set('architecture_fingerprint architecture_group_id candidate_id config_id configuration_id joint_config_fingerprint joint_fingerprint hp_fingerprint candidate_fingerprint record_id task_id run_id protocol_id recipe_id source_id source_config_id source_file source_path source_lines source_reference source_refs input_path output_path checkpoint_sha256 formal_source_id formal_config_fingerprint git_commit commit head code_sha256 source_sha256 format_version created_at started_at finished_at timestamp time type record_type operation_type evaluation_stage evaluation_fidelity method_label initialization_strategy search_seed evaluation_seed seed step iteration index valid terminal_status dataset dataset_name dataset_variant split protocol split_id train_config_ids validation_config_ids train_architecture_fingerprints validation_architecture_fingerprints config_ids architecture_fingerprints training_seed fixed_split_manifest fixed_split_manifest_sha256 task_plan task_plan_sha256 pool_path candidate_pool frozen_candidate_pool input_sha256 file_sha256 sha256 relative_path path source_group fingerprint_count effective_layers operations edges gat_heads sage_aggr gin_eps gcnii_alpha gcnii_theta gat_heads_by_layer sage_aggr_by_layer gin_eps_by_layer activation merge normalization hp_mode role layer alpha theta shared_weights concat aggr eps train_eps heads train_ids validation_ids selected_ids member_ids members source_config_ids source_architecture_fingerprints'.split())
CONTAINERS=set('records rows history configurations candidates architectures candidate config configuration decoded_config architecture graph nodes data inputs sources files tasks splits selections selected selection development development_rows labels rows_by_id dataset_context config_by_id configurations_by_id reference_config config_references frozen_selection initial_configs online_history initialization_history source_identity provenance run_metadata metadata source_manifest candidate_configs manifest original expanded full exclude_high_noise'.split())
HASH_MAPS={'code_hashes','input_hashes','protected_hashes','source_hashes','dependency_hashes'}
PATH_FIELDS={'labels','unified_root','old_pilot_root','replay','base_checkpoint','base_configs','prepared','candidate_pool','configurations','source'}
SCORING_WORDS=re.compile(r'(^|_)(acc|accuracy|score|scores|se|stderr|variance|mean|std|rank|ranking|regret|reward|gain|prediction|predictions|acquisition|metric|metrics|loss|target|targets|y|nlpd|mae|rmse|auc)(_|$)',re.I)
HEX=re.compile(r'^[a-fA-F0-9]{64}$')
class ProjectionError(ValueError):pass
class Stream:
 def __init__(self,fp):
  self.fp=fp;self.buf=b'';self.i=0;self.bytes=0;self.hash=hashlib.sha256();self.eof=False
 def peek(self):
  if self.i==len(self.buf) and not self.eof:
   self.buf=self.fp.read(65536);self.i=0;self.bytes+=len(self.buf);self.hash.update(self.buf);self.eof=not self.buf
  return self.buf[self.i] if self.i<len(self.buf) else None
 def take(self):
  v=self.peek()
  if v is None:raise ProjectionError('Unexpected EOF')
  self.i+=1;return v
 def space(self):
  while self.peek() in (9,10,13,32):self.take()

class Projector:
 def __init__(self,fp):
  self.s=Stream(fp);self.skipped=Counter();self.decoded_fields=Counter();self.scalars_decoded=0;self.skipped_values=0
 def string(self,decode):
  if self.s.take()!=34:raise ProjectionError('String expected')
  raw=bytearray(b'"') if decode else None
  while True:
   c=self.s.take()
   if decode:raw.append(c)
   if c==92:
    c=self.s.take()
    if decode:raw.append(c)
   elif c==34:break
   elif c<32:raise ProjectionError('Unescaped control byte')
  return json.loads(raw) if decode else None
 def scalar(self,decode):
  if self.s.peek()==34:return self.string(decode)
  raw=bytearray() if decode else None;count=0
  while self.s.peek() is not None and self.s.peek() not in (9,10,13,32,44,93,125):
   c=self.s.take();count+=1
   if decode:raw.append(c)
  if not count:raise ProjectionError('Missing scalar')
  return json.loads(raw) if decode else None
 def skip(self):
  self.s.space();c=self.s.peek();self.skipped_values+=1
  if c==34:self.string(False);return
  if c not in (91,123):self.scalar(False);return
  # Lexical stack validates bracket/string boundaries without decoding keys or
  # values in this forbidden subtree. It does not interpret number tokens.
  stack=[self.s.take()]
  while stack:
   self.s.space();c=self.s.peek()
   if c==34:self.string(False)
   elif c in (91,123):stack.append(self.s.take())
   elif c in (93,125):
    self.s.take();opening=stack.pop()
    if (opening,c) not in ((91,93),(123,125)):raise ProjectionError('Mismatched skipped container')
   elif c is None:raise ProjectionError('Truncated skipped subtree')
   else:self.s.take()
 def value(self,mode='project',field=None):
  self.s.space();c=self.s.peek()
  if mode=='skip':self.skip();return None
  if c==123:
   self.s.take();out={};self.s.space()
   if self.s.peek()==125:self.s.take();return out
   while True:
    self.s.space();key=self.string(True);self.s.space()
    if self.s.take()!=58:raise ProjectionError('Missing colon')
    if mode=='hash_map':
     self.s.space()
     if self.s.peek()!=34:raise ProjectionError('Hash map requires digest strings')
     val=self.string(True)
     if not HEX.fullmatch(val):raise ProjectionError('Non-digest in version hash map')
     out[key]=val
    elif mode=='literal':
     # Structured identity payloads (e.g. members) still use field projection.
     val=self.field_value(key,field)
     if val is not None:out[key]=val
    else:
     val=self.field_value(key,field)
     if val is not None:out[key]=val
    self.s.space();sep=self.s.take()
    if sep==125:break
    if sep!=44:raise ProjectionError('Missing object comma')
   return out
  if c==91:
   self.s.take();out=[];self.s.space()
   if self.s.peek()==93:self.s.take();return out
   while True:
    val=self.value(mode,field)
    if val is not None:out.append(val)
    self.s.space();sep=self.s.take()
    if sep==93:break
    if sep!=44:raise ProjectionError('Missing array comma')
   return out
  if mode=='literal':self.scalars_decoded+=1;return self.scalar(True)
  # Unexpected non-container inside a structural section is skipped. Path
  # references under inputs are separately allowed by their explicit keys.
  self.scalar(False);return None
 def field_value(self,key,parent):
  if key in HASH_MAPS:self.decoded_fields[key]+=1;return self.value('hash_map',key)
  if key in IDENTITY:
   self.decoded_fields[key]+=1;return self.value('literal',key)
  if parent=='inputs' and key in PATH_FIELDS:
   self.decoded_fields['inputs.'+key]+=1;self.s.space()
   if self.s.peek()!=34:raise ProjectionError('Input references must be strings')
   return self.string(True)
  if key in CONTAINERS or HEX.fullmatch(key):return self.value('project',key)
  self.skipped[key]+=1;self.skip();return None
 def all(self,jsonl=False):
  result=[];self.s.space()
  while self.s.peek() is not None:
   result.append(self.value());self.s.space()
   if not jsonl and self.s.peek() is not None:raise ProjectionError('Trailing non-JSON content')
  return result if jsonl else result[0] if result else None
 def counters(self):return {'mixed_file_byte_reads':1,'hash_reads':1,'bytes_read':self.s.bytes,'sha256':self.s.hash.hexdigest(),'performance_field_decodes':0,'performance_field_exposures':0,'performance_field_writes':0,'performance_field_uses':0,'skipped_value_count':self.skipped_values,'selected_scalar_decodes':self.scalars_decoded,'decoded_field_names':sorted(self.decoded_fields),'skipped_field_names':sorted(self.skipped)}

def project(path):
 with open(path,'rb') as f:
  parser=Projector(f);value=parser.all(str(path).endswith('.jsonl'));return value,parser.counters()

def objects(value,pointer='$'):
 if isinstance(value,dict):
  yield pointer,value
  for key,v in value.items():yield from objects(v,pointer+'/'+key)
 elif isinstance(value,list):
  for i,v in enumerate(value):yield from objects(v,pointer+'/'+str(i))
