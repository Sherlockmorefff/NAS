"""Original loose-end graph family and complete stable directed-WL axes."""
from collections import Counter
import hashlib
import itertools
import json
import numpy as np
from .common import digest
from analyse.structured_kernel_phase1.features import build_operator_graph

OPS=['GCNConv','GATConv','SAGEConv','GINConv','GCNII','Identity']
FIXED={'normalization':'none','hp_mode':'global4','gat_heads':1,'sage_aggr':'mean','gin_eps':0.,
       'gat_heads_by_layer':None,'sage_aggr_by_layer':None,'gin_eps_by_layer':None,'gcnii_alpha':.1,'gcnii_theta':.5}
HIDDEN=[16,32,64,128,256,512];L2=[1e-5,5e-5,1e-4,5e-4,1e-3]

def rebuild(ops,internal):
    m=len(ops);edges=sorted(set(tuple(map(int,e)) for e in internal))
    if not 1<=m<=5 or any(op not in OPS for op in ops) or all(op=='Identity' for op in ops):raise ValueError('Invalid operator family')
    if any(not 0<=u<v<=m for u,v in edges):raise ValueError('Invalid internal ordered edge')
    loose=[v for v in range(m+1) if not any(u==v for u,w in edges)]
    return {'operations':list(ops),'edges':[list(e) for e in sorted(edges+[(v,m+1) for v in loose])],
            'effective_layers':sum(op!='Identity' for op in ops)}
def valid(config):
    ops=config['operations'];m=len(ops)
    internal=[e for e in config['edges'] if e[1]!=m+1]
    rebuilt=rebuild(ops,internal)
    if sorted(config['edges'])!=rebuilt['edges'] or config['effective_layers']!=rebuilt['effective_layers']:raise ValueError('Graph output edges/effective layers differ from original loose ends')
    if any(config.get(k)!=v for k,v in FIXED.items()):raise ValueError('Global4 fixed attributes changed')
    if len(config['edges'])>16:raise ValueError('Graph edge bound exceeded')
    build_operator_graph(config);return True
def config(arch,hidden,l2,v,land=False):
    v=np.clip(np.asarray(v,dtype=np.float64),0.,1.)
    if land:v=v.astype('<f4').astype(np.float64)
    c={**arch,**FIXED,'hidden_dim':int(hidden),'l2':float(l2),'lr':float(10**(-4+2.5*v[0])),'dropout':float(.1+.5*v[1])}
    valid(c);return c
def aid(c):return digest(build_operator_graph(c))
def hp(c):return [float(np.log10(c['lr'])),c['dropout'],float(np.log2(c['hidden_dim'])),float(np.log10(c['l2']))]
def semantic(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def tokens(c):
    graph=build_operator_graph(c);inc=[[] for _ in graph['nodes']];out=[[] for _ in graph['nodes']]
    for u,v in graph['edges']:inc[v].append(u);out[u].append(v)
    labels=[semantic(node) for node in graph['nodes']];cnt=Counter()
    for h in range(3):
        cnt.update((h,l) for l in labels)
        if h<2:labels=[semantic([labels[v],sorted(labels[u] for u in inc[v]),sorted(labels[w] for w in out[v])]) for v in range(len(labels))]
    return cnt,len(labels),len(graph['edges'])
def vocabulary(configs):return sorted({key for c in configs for key in tokens(c)[0]})
def features(configs,vocab,nmax=7,emax=16):
    index={tuple(t):i for i,t in enumerate(vocab)};x=np.zeros((len(configs),len(index)+2),dtype=np.float64)
    for row,c in enumerate(configs):
        cnt,n,e=tokens(c);norm=np.sqrt(sum(v*v for v in cnt.values()))
        for key,v in cnt.items():
            if key not in index:raise ValueError('Incomplete stable token union')
            x[row,index[key]]=v/norm
        x[row,-2:]=[n/nmax,e/emax]
    return x
def propose(obs,seeds):
    random=np.random.default_rng(seeds['random']);mutation=np.random.default_rng(seeds['mutation'])
    seen={o['architecture_id'] for o in obs};parents={}
    for o in obs:
        key=o['architecture_id']
        if key not in parents or o['validation_accuracy']>parents[key]['validation_accuracy']:parents[key]=o
    ranked=sorted(parents.values(),key=lambda o:(-o['validation_accuracy'],o['architecture_id']))[:5]
    chosen=[];used=set();trace=[]
    for flow,rng in [('random',random),('mutation',mutation)]:
        count=0
        for attempt in range(128):
            meta={'flow':flow,'attempt':attempt,'parent':None};reason=None
            if flow=='random':
                m=int(rng.integers(1,6));ops=[OPS[int(i)] for i in rng.integers(0,6,m)]
                internal=[(u,v) for u in range(m+1) for v in range(u+1,m+1) if rng.random()<.5]
                meta['raw']={'operations':ops,'internal_edges':internal}
            else:
                parent=ranked[int(rng.integers(len(ranked)))];pc=parent['configuration'];m=len(pc['operations']);ops=list(pc['operations'])
                internal=[tuple(e) for e in pc['edges'] if e[1]!=m+1];meta['parent']=parent['architecture_id']
                if rng.random()<.5:
                    pos=int(rng.integers(m));old=ops[pos];ops[pos]=rng.choice([o for o in OPS if o!=old]).item();meta['change']={'kind':'operator','position':pos,'old':old,'new':ops[pos]}
                else:
                    possible=[(u,v) for u in range(m+1) for v in range(u+1,m+1)];edge=possible[int(rng.integers(len(possible)))]
                    internal.remove(edge) if edge in internal else internal.append(edge);meta['change']={'kind':'toggle_edge','edge':edge}
            try:
                arch=rebuild(ops,internal);c=config(arch,64,5e-4,[.5,.5]);key=aid(c)
                if key in used:reason='round_duplicate'
                elif flow=='random' and key in seen:reason='historical_graph'
                elif flow=='mutation' and key==meta['parent']:reason='same_as_parent'
            except ValueError as exc:reason=str(exc);key=None
            trace.append({**meta,'architecture_id':key,'rejected':reason})
            if reason is None:
                chosen.append({'architecture':arch,'architecture_id':key,'source':meta});used.add(key);count+=1
            if count==8:break
    return chosen,trace,{'random':random.bit_generator.state,'mutation':mutation.bit_generator.state}
