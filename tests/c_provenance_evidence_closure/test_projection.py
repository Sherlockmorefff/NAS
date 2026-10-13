import io,json,math
from unittest.mock import patch
import pytest
from analyse.c_provenance_evidence_closure.projector import Projector,ProjectionError
from analyse.c_provenance_evidence_closure.collect import summarize,valid_structure
from analyse.c_independence_minimal_audit.identity import fingerprint,source_decision,UNKNOWN

def row():return {'architecture_fingerprint':'a'*64,'operations':['GCNConv'],'edges':[[0,1],[1,2]]}
def parse(data):
 p=Projector(io.BytesIO(data));v=p.all();return v,p

def test_forbidden_numbers_and_strings_are_never_decoded():
 sentinel='9.876543210123456';data=('['+json.dumps(row())[:-1]+',"val_acc":'+sentinel+',"test_acc":"secret_score","SE":'+sentinel+',"predictions":{"candidate_id":"forbidden","mean":'+sentinel+'}}]').encode()
 original=json.loads;decoded=[]
 def loads(x,*a,**kw):
  s=bytes(x).decode() if isinstance(x,(bytes,bytearray)) else x;decoded.append(s);assert sentinel not in s and 'secret_score' not in s and s!='"forbidden"';return original(x,*a,**kw)
 with patch('analyse.c_provenance_evidence_closure.projector.json.loads',loads):v,p=parse(data)
 assert v==[row()] and p.counters()['performance_field_decodes']==0

def test_nested_braces_escaped_quotes_and_skipped_array_boundaries():
 r=row();data=json.dumps({'rows':[r],'metrics':{'x':'braces } ] \\"','nested':[{'candidate_id':'hidden','score':.99}]}}).encode();v,p=parse(data)
 assert v=={'rows':[r]}

def test_stream_crosses_chunks_without_whole_file_read():
 r=row();b=json.dumps({'score_blob':'x'*200000,'rows':[r]}).encode();v,p=parse(b)
 assert v=={'rows':[r]} and p.s.bytes==len(b)

def test_unknown_subtree_is_skipped_conservatively():
 v,p=parse(json.dumps({'unknown':{'architecture_fingerprint':'hidden','score':.5},'rows':[row()]}).encode())
 assert v=={'rows':[row()]}

def test_hash_maps_only_allow_digest_values():
 v,p=parse(json.dumps({'code_hashes':{'/code.py':'b'*64},'inputs':{'labels':'/development/members.json'}}).encode())
 assert v['code_hashes']['/code.py']=='b'*64 and v['inputs']['labels']=='/development/members.json'
 with pytest.raises(ProjectionError):parse(b'{"code_hashes":{"x":0.123}}')

def test_truncated_forbidden_value_stops_no_partial_certificate():
 with pytest.raises(ProjectionError):parse(b'{"operations":["GCNConv"],"score":{"a":"unterminated')

def test_actual_record_structure_can_prove_miss_without_stored_schema():
 r=row();summary=summarize([r],set());assert summary['records_with_supported_structure']==1
 assert summary['stored_to_concrete']['a'*64]==[fingerprint(r)] and summary['records'][0]['stored_identity_matches_recomputed'] is False

def test_unsupported_member_prevents_complete_structure_proof():
 r=row();r['operations']=['unproven_alias'];s=summarize([row(),r],set());assert s['unresolved_top_records']==1

def test_group_negative_requires_membership_and_comparison_proof():
 s={'use':UNKNOWN,'coverage_complete':True,'schema_verified':True,'identity_type_verified':True}
 assert source_decision(s,'ORBIT_MISS')=='PASS_CERTIFIED_ORBIT_MISS'
 s['coverage_complete']=False;assert source_decision(s,'ORBIT_MISS')=='PROVENANCE_UNRESOLVED'

def test_coarse_hits_are_kept_for_targeted_review_only():
 r=row();s=summarize([r],{fingerprint(r)})
 assert len(s['coarse_hits'])==1 and 'classification' not in s

def test_metadata_numbers_do_not_admit_score_field_aliases():
 v,p=parse(b'{"seed":1,"rank":1,"score":1,"test":{"values":[.999]},"mean":0.8,"variance":0.1}')
 assert v=={'seed':1}

def test_jsonl_projection_and_no_metric_values_in_output():
 p=Projector(io.BytesIO((json.dumps({**row(),'mae':.2})+'\n'+json.dumps({**row(),'nlpd':-.3})).encode()));v=p.all(True)
 assert v==[row(),row()] and 'mae' not in json.dumps(v)
