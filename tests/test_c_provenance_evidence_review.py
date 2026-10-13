import io,json
import pytest
from analyse.c_provenance_evidence_closure.review import certificate_can_exclude,local_decision
from analyse.c_provenance_evidence_closure.followups import BindingProjector
from analyse.c_provenance_evidence_closure.common import verify,sha

def cert():return {'complete_membership':True,'comparison_rule_proven':True,'input_binding_verified':True,'structure_ids':['a'],'use':'USE_UNRESOLVED'}

@pytest.mark.parametrize('key',['complete_membership','comparison_rule_proven','input_binding_verified'])
def test_incomplete_certificate_blocks(key):
 c=cert();c[key]=False;assert not certificate_can_exclude(c) and local_decision(c,{'z'})=='PROVENANCE_UNRESOLVED'

def test_complete_group_unknown_use_is_not_a_veto():
 c=cert();assert local_decision(c,{'z'})=='PASS_BOUND_MEMBER_SUPERSET' and c['use']=='USE_UNRESOLVED'

def test_coarse_match_never_becomes_development_used():
 c=cert();c['use']='PERFORMANCE_DEVELOPMENT';assert local_decision(c,{'a'})=='HISTORICAL_OVERLAP_POSSIBLE'

def test_effective_attrs_only_not_global_hp_or_metrics():
 p=BindingProjector(io.BytesIO(json.dumps({'hp':{'gat_heads':1,'gin_eps':0.,'gcnii_alpha':.1,'lr':.2,'SE':.999},'score':.88}).encode()))
 assert p.all()=={'hp':{'gat_heads':1,'gin_eps':0.,'gcnii_alpha':.1}}

def test_integrity_metadata_keeps_ids_but_skips_predictions():
 p=BindingProjector(io.BytesIO(b'{"passed":true,"missing_task_ids":[],"expected_task_count":2,"validation_predictions":[{"config_id":"forbidden","target":0.999}]}'))
 assert p.all()=={'passed':True,'missing_task_ids':[],'expected_task_count':2}

def test_changed_input_rejects_reused_proof(tmp_path):
 p=tmp_path/'identity.json';p.write_text('{}');saved={'identity.json':sha(p)};verify(tmp_path,saved);p.write_text('{"changed":true}')
 with pytest.raises(ValueError):verify(tmp_path,saved)
