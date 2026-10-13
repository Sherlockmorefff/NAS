from analyse.c_independent_validation.review import decision,members
from collections import defaultdict

def test_bound_superset_unknown_use_negative_passes():
 assert decision({'complete_membership':True,'binding_verified':True,'structure_ids':['history'],'use':'USE_UNRESOLVED'},['C'])=='PASS_BOUND_COMPLETE_MEMBERS'

def test_unknown_unbound_source_cannot_pass_negative():
 assert decision({'complete_membership':True,'binding_verified':False,'structure_ids':['history']},['C'])=='PROVENANCE_UNRESOLVED'

def test_coarse_hit_does_not_claim_development_leakage():
 assert decision({'complete_membership':True,'binding_verified':True,'structure_ids':['C']},['C'])=='HISTORICAL_OVERLAP_POSSIBLE'

def test_unknown_identity_is_not_dropped():
 value={'config_ids':['a','unknown'],'validation_architecture_fingerprints':['graph']};lookup=defaultdict(set,{('config_id','a'):{'g'},('architecture_fingerprint','graph'):{'g'}})
 r=members(value,lookup);assert r['structure_ids']==['g'] and len(r['unmapped'])==1

def test_multiple_mapped_identities_keep_conservative_union():
 lookup=defaultdict(set,{('config_id','x'):{'g1','g2'}})
 r=members([{'config_id':'x'}],lookup);assert r['structure_ids']==['g1','g2']

def test_dataset_edge_count_is_not_an_architecture():
 lookup=defaultdict(set,{('config_id','x'):{'g'}})
 r=members({'dataset_identity':{'edges':10556,'split_protocol':'public'},'candidates':[{'config_id':'x'}]},lookup)
 assert not r['unsupported_structure_paths'] and r['structure_ids']==['g']

def test_summary_only_allowed_when_actual_parent_graph_matches():
 row={'operations':['GCNConv'],'edges':[[0,1],[1,2]],'architecture_summary':{'operations':['GCNConv']}}
 assert not members([row],defaultdict(set))['unsupported_structure_paths']
 row['architecture_summary']['operations']=['GINConv']
 assert members([row],defaultdict(set))['unsupported_structure_paths']==['$/0/architecture_summary']
