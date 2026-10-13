"""One documented byte schema; complete internal permutations, attributes carried."""
import ast,copy,hashlib,itertools,json,math,typing
from collections import Counter
from .common import canonical,digest,PERF,FREE,UNKNOWN
from analyse.structured_kernel_phase1.features import isomorphic_operator_graph
SCHEMA='ordered_ops_sorted_edges_compact_ascii_sha256_v1'
def fingerprint(config):
 payload={'operations':[str(v) for v in config['operations']],'edges':sorted([list(map(int,e)) for e in config['edges']])}
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')).hexdigest()
def relabel(graph,permutation):
 m=len(graph['nodes'])-2
 if sorted(permutation)!=list(range(1,m+1)):raise ValueError('Not a complete internal bijection')
 mapping=[0,*permutation,m+1];nodes=[None]*(m+2)
 for i,j in enumerate(mapping):nodes[j]=copy.deepcopy(graph['nodes'][i])
 return {'nodes':nodes,'edges':sorted([[mapping[u],mapping[v]] for u,v in graph['edges']])}
def config_of(graph):return {'operations':[n['operation'] for n in graph['nodes'][1:-1]],'edges':graph['edges']}
def orbit(graph):
 m=len(graph['nodes'])-2
 if not 1<=m<=5 or graph['nodes'][0]['role']!='input' or graph['nodes'][-1]['role']!='output':raise ValueError('Outside frozen grammar')
 records=[]
 for p in itertools.permutations(range(1,m+1)):
  g=relabel(graph,p);records.append({'internal_old_to_new':list(p),'coarse_hash':fingerprint(config_of(g)),'typed_graph_hash':digest(g)})
 if len(records)!=math.factorial(m):raise AssertionError('Incomplete orbit')
 return {'node_count':m+2,'internal_count':m,'permutations':records,'permutation_count':len(records),'schema':SCHEMA,'unique_coarse_hashes':sorted({r['coarse_hash'] for r in records}),'attribute_semantics':'deep-copy complete node attributes; never rebuild GCNII layer after relabeling'}
def isolated_functions(text,names,extra=None):
 tree=ast.parse(text);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('Missing audited pure function')
 unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
 env={'json':json,'hashlib':hashlib,'Mapping':typing.Mapping,'Sequence':typing.Sequence,'Any':typing.Any,**(extra or {})}
 exec(compile(ast.fix_missing_locations(unit),'<audited pure fingerprint functions>','exec'),env)
 return env
class ExactIndex:
 def __init__(self):self.groups={}
 @staticmethod
 def key(g):return (len(g['edges']),tuple(sorted(Counter(canonical(n) for n in g['nodes']).items())))
 def add(self,i,g):self.groups.setdefault(self.key(g),[]).append((i,g))
 def match(self,g):return [i for i,h in self.groups.get(self.key(g),[]) if isomorphic_operator_graph(g,h)]
def source_decision(source,relation):
 """Missing graph counts never enter this decision."""
 if relation=='EXACT_OVERLAP' and source['use']==PERF:return 'DEVELOPMENT_USED'
 if source['use']==FREE and source.get('label_free_event_closed',False):return 'PASS_DISCLOSE_STRUCTURE_EXPOSURE'
 if relation=='EXACT_DISJOINT_COMPLETE' and source['coverage_complete']:return 'PASS_EXACT_DISJOINT'
 if relation=='ORBIT_MISS' and all(source.get(k,False) for k in ('schema_verified','coverage_complete','identity_type_verified')):return 'PASS_CERTIFIED_ORBIT_MISS'
 if relation=='NECESSARY_CONDITIONS_DISJOINT' and source.get('necessary_conditions_proven',False):return 'PASS_NECESSARY_CONDITIONS'
 return 'PROVENANCE_UNRESOLVED'
def classify(decisions):
 if any(d=='DEVELOPMENT_USED' for d in decisions):return 'DEVELOPMENT_USED'
 if any(d=='PROVENANCE_UNRESOLVED' for d in decisions):return 'PROVENANCE_UNRESOLVED'
 return 'ELIGIBLE_WITH_EVIDENCE'
