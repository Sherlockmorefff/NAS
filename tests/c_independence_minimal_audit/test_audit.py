import copy,itertools,json,math,subprocess,sys
from pathlib import Path
import pytest
from analyse.c_independence_minimal_audit.identity import *
from analyse.c_independence_minimal_audit.common import write,sha
from analyse.c_independence_minimal_audit.audit import verify_files
from analyse.structured_kernel_phase1.features import build_operator_graph

def graph(ops=('GCNII','Identity','GCNII')):
 return build_operator_graph({'operations':list(ops),'edges':[[i,i+1] for i in range(len(ops)+1)]})
def source(**kw):return dict(use=UNKNOWN,coverage_complete=True,schema_verified=True,identity_type_verified=True,**kw)

def test_all_bijections_and_attributes():
 g=graph();before=copy.deepcopy(g)
 for p in itertools.permutations(range(1,4)):
  h=relabel(g,p)
  assert isomorphic_operator_graph(g,h)
  for i,j in enumerate(p,1):assert g['nodes'][i]==h['nodes'][j]
 assert g==before
 swapped=relabel(g,(3,2,1));assert swapped['nodes'][3]['layer']==1 and swapped['nodes'][1]['layer']==2

@pytest.mark.parametrize('m',range(1,6))
def test_enumeration_budget(m):
 o=orbit(graph(tuple(['Identity']*m)))
 assert o['permutation_count']==math.factorial(m)
 assert len({tuple(p['internal_old_to_new']) for p in o['permutations']})==math.factorial(m)

def test_orbit_invariance():
 g=graph();h=relabel(g,(2,3,1))
 assert orbit(g)['unique_coarse_hashes']==orbit(h)['unique_coarse_hashes']
 assert {r['typed_graph_hash'] for r in orbit(g)['permutations']}=={r['typed_graph_hash'] for r in orbit(h)['permutations']}

def test_coarse_hit_is_not_typed_or_development_proof():
 g=graph();h=copy.deepcopy(g);h['nodes'][1]['layer']=9
 assert fingerprint(config_of(g))==fingerprint(config_of(h))
 assert not isomorphic_operator_graph(g,h)
 s=source();s['use']=PERF
 assert source_decision(s,'COARSE_ORBIT_HIT')=='PROVENANCE_UNRESOLVED'

def test_different_directed_connections_and_roles():
 g=graph();h=copy.deepcopy(g);h['edges'][-1]=[1,4]
 assert not isomorphic_operator_graph(g,h)
 h=copy.deepcopy(g);h['nodes'][0]['role']='operator'
 assert not isomorphic_operator_graph(g,h)

@pytest.mark.parametrize('key',['schema_verified','coverage_complete','identity_type_verified'])
def test_unknown_certificate_cannot_pass_negative(key):
 s=source();s[key]=False
 assert source_decision(s,'ORBIT_MISS')=='PROVENANCE_UNRESOLVED'

def test_unknown_use_does_not_block_proven_disjointness():
 assert source_decision(source(),'ORBIT_MISS')=='PASS_CERTIFIED_ORBIT_MISS'
 assert source_decision(source(),'EXACT_DISJOINT_COMPLETE')=='PASS_EXACT_DISJOINT'

def test_structure_exposure_only_local_event():
 s=source();s.update(use=FREE,label_free_event_closed=True)
 d=source_decision(s,'EXACT_OVERLAP');assert d=='PASS_DISCLOSE_STRUCTURE_EXPOSURE'
 assert classify([d,'PROVENANCE_UNRESOLVED'])=='PROVENANCE_UNRESOLVED'
 assert classify([d,'PASS_EXACT_DISJOINT'])=='ELIGIBLE_WITH_EVIDENCE'

def test_architecture_use_applies_to_new_hp_and_seed():
 s=source();s['use']=PERF
 decisions=[source_decision(s,'EXACT_OVERLAP') for _ in range(4)]
 assert all(x=='DEVELOPMENT_USED' for x in decisions)

def test_missing_graph_count_never_gates_proof():
 s=source();s['missing_history_graph_count']=11823
 assert classify([source_decision(s,'ORBIT_MISS')])=='ELIGIBLE_WITH_EVIDENCE'
 s['missing_history_graph_count']=0;s['schema_verified']=False
 assert classify([source_decision(s,'ORBIT_MISS')])=='PROVENANCE_UNRESOLVED'

def test_exact_index_and_edge_order():
 g=graph();idx=ExactIndex();idx.add('g',g);h=relabel(g,(2,1,3));h['edges'].reverse()
 assert idx.match(h)==['g']

def test_checkpoint_integrity_refuses_changed_evidence(tmp_path):
 write(tmp_path/'proof.json',{'known':1});ident={'proof.json':sha(tmp_path/'proof.json')};verify_files(tmp_path,ident)
 write(tmp_path/'proof.json',{'known':2})
 with pytest.raises(ValueError,match='identity mismatch'):verify_files(tmp_path,ident)

def test_forbidden_cache_read_is_refused_before_parsing(tmp_path):
 script='''from pathlib import Path
from analyse.c_independence_minimal_audit.common import guard
import sys
r=Path(sys.argv[1]);(r/'results').mkdir();(r/'results/poison.json').write_text('{"B_label":0.99}')
guard(r/'new_run',r,[])
try:(r/'results/poison.json').read_text()
except PermissionError:sys.exit(0)
sys.exit(1)
'''
 subprocess.run([sys.executable,'-B','-c',script,str(tmp_path)],check=True)

def test_hp_not_in_architecture_identity():
 c={'operations':['GCNConv'],'edges':[[0,1],[1,2]],'lr':.01};d={**c,'lr':.001,'hidden_dim':512,'dropout':.5,'seed':999}
 assert fingerprint(c)==fingerprint(d) and build_operator_graph(c)==build_operator_graph(d)
