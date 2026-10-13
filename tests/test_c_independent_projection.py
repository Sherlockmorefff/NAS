import io,json
import pytest
from analyse.c_independent_validation.projection import IdentityProjector

def run(x,jsonl=False):
 p=IdentityProjector(io.BytesIO(x));v=p.all(jsonl);return v,p

def test_nested_test_results_keep_ids_and_skip_values():
 x={'test':{'evaluation':{'seed_results':[{'candidate_id':'cid','seed':2,'test_accuracy':0.987654321,'SE':0.012345678,'rank':777,'prediction':{'mean':888,'variance':999}}]}}}
 v,p=run(json.dumps(x).encode());assert v['test']['evaluation']['seed_results'][0]['candidate_id']=='cid'
 assert all(s not in json.dumps(v) for s in ('0.987654321','0.012345678','777','888','999'))
 assert p.counters()['performance_field_decodes']==0

def test_score_values_never_call_json_decoder(monkeypatch):
 real=json.loads
 def checked(value,*args,**kw):
  if isinstance(value,(bytes,bytearray)) and b'987654321' in value:raise AssertionError('Forbidden value decoded')
  return real(value,*args,**kw)
 monkeypatch.setattr(json,'loads',checked)
 v,p=run(b'{"candidate_id":"c","loss":987654321,"se":"987654321","ranks":[987654321],"test":{"score":987654321}}')
 assert v['candidate_id']=='c'

def test_architecture_payload_and_hp():
 x={'candidate_panel':[{'operations':['GCNII','GATConv'],'edges':[[0,1],[1,2],[2,3]],'gcnii_alpha':.1,'gcnii_theta':.5,'lr':.001,'hidden_dim':64,'candidate_fingerprint':'abc'}]}
 assert run(json.dumps(x).encode())[0]==x

def test_jsonl_seeds_keep_rows():
 v,p=run(b'{"candidate_id":"a","seed":0,"val_acc":0.7}\n{"candidate_id":"a","seed":1,"val_acc":0.8}\n',True)
 assert v==[{'candidate_id':'a','seed':0},{'candidate_id':'a','seed':1}]

def test_long_skipped_string_escapes_and_chunk_boundary():
 secret=('x'*70000)+'\\\"payload';x={'candidate_id':'cid','loss':secret}
 v,p=run(json.dumps(x).encode());assert v=={'candidate_id':'cid'}

def test_malformed_source_fails_closed():
 with pytest.raises(ValueError):run(b'{"candidate_id":"a","loss":[123}')

def test_version_hashes_preserved_only_as_digests():
 x={'source_hashes':{'file.py':'a'*64},'loss':345}
 v,p=run(json.dumps(x).encode());assert v=={'source_hashes':{'file.py':'a'*64}}
 with pytest.raises(ValueError):run(b'{"source_hashes":{"x":"not-a-hash"}}')
