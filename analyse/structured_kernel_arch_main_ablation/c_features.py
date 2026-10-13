"""Unlabeled append-only WL extension, never used by the formal A learner."""
import json
from collections import Counter
import numpy as np
from analyse.structured_kernel_phase1.features import _canonical,isomorphic_operator_graph,graph_hash
from .common import digest


def iso_key(graph):
    return (len(graph['edges']),tuple(sorted(Counter(_canonical(n) for n in graph['nodes']).items())))


class ExactIndex:
    def __init__(self,graphs=()):
        self.groups={}
        for identity,g in graphs:self.add(identity,g)
    def add(self,identity,graph):self.groups.setdefault(iso_key(graph),[]).append((identity,graph))
    def match(self,graph):
        return [identity for identity,old in self.groups.get(iso_key(graph),[]) if isomorphic_operator_graph(graph,old)]


def encode_frozen(graphs,dictionaries,n_max,e_max):
    labels={a:[_canonical(n) for n in g['nodes']] for a,g in graphs.items()};counts={a:[] for a in graphs};round_oov=[]
    for h in range(3):
        index={s:i for i,s in enumerate(dictionaries[h]['labels'])}
        missing=sorted({s for v in labels.values() for s in v if s not in index})
        if missing:raise ValueError(f'ENCODING_BLOCKED: h{h} has {len(missing)} unfrozen signatures')
        codes={a:[index[s] for s in ls] for a,ls in labels.items()}
        for a in graphs:
            hist=Counter(codes[a]);counts[a].extend(hist.get(i,0) for i in range(len(index)))
        if h<2:labels=refine(graphs,codes)
    features={}
    for a,g in graphs.items():
        v=np.array(counts[a],dtype=float);norm=np.linalg.norm(v)
        if not np.isfinite(norm) or norm<=0:raise ValueError('Invalid WL norm')
        features[a]=np.r_[v/norm,len(g['nodes'])/n_max,len(g['edges'])/e_max if e_max else 0.].tolist()
    return features,counts


def refine(graphs,codes):
    result={}
    for a,g in graphs.items():
        incoming=[[] for _ in g['nodes']];outgoing=[[] for _ in g['nodes']]
        for u,v in g['edges']:incoming[v].append(u);outgoing[u].append(v)
        result[a]=[_canonical([codes[a][i],sorted(codes[a][u] for u in incoming[i]),sorted(codes[a][v] for v in outgoing[i])]) for i in range(len(g['nodes']))]
    return result


def extend_dictionary(Agraphs,Cgraphs,old_manifest,old_features):
    graphs={**Agraphs,**Cgraphs}
    if set(Agraphs)&set(Cgraphs):raise ValueError('C identity overlaps A')
    labels={a:[_canonical(n) for n in g['nodes']] for a,g in graphs.items()}
    dictionaries=[];added=[];mapping=[];old_offset=new_offset=0
    for h in range(3):
        old=old_manifest['dictionaries'][h]['labels']
        fresh=sorted({s for values in labels.values() for s in values}-set(old))
        vocabulary=old+fresh;dictionaries.append({'h':h,'labels':vocabulary,'size':len(vocabulary)})
        added.append({'h':h,'old_count':len(old),'new_signature_count':len(fresh),'new_signatures':fresh,
            'C_OOV_node_occurrences':sum(s not in set(old) for a in Cgraphs for s in labels[a])})
        mapping.extend({'old':old_offset+i,'new':new_offset+i,'round':h,'signature':s} for i,s in enumerate(old))
        old_offset+=len(old);new_offset+=len(vocabulary)
        codes={a:[vocabulary.index(s) for s in values] for a,values in labels.items()}
        if h<2:labels=refine(graphs,codes)
    features,counts=encode_frozen(graphs,dictionaries,old_manifest['n_max'],old_manifest['e_max'])
    mapping.extend([{'old':old_offset,'new':new_offset,'feature':'n_nodes/n_max'},{'old':old_offset+1,'new':new_offset+1,'feature':'n_edges/e_max'}])
    aid=sorted(Agraphs);before=np.array([old_features[a] for a in aid]);after=np.array([features[a] for a in aid])
    lifted=np.zeros_like(after)
    for m in mapping:lifted[:,m['new']]=before[:,m['old']]
    if not np.array_equal(lifted,after):raise ValueError('ENCODING_REVIEW_REQUIRED: old A coordinates/norm changed')
    def distances(x):return np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=-1))
    da,db=distances(before),distances(after)
    error=float(np.max(np.abs(da-db)))
    if error>1e-12:raise ValueError('ENCODING_REVIEW_REQUIRED: old A distances changed')
    errors={}
    for ell in (.5,1.,2.):
        def matern(d):
            r=np.sqrt(5)*d/ell;return (1+r+r*r/3)*np.exp(-r)
        errors[str(ell)]=float(np.max(np.abs(matern(da)-matern(db))))
    if max(errors.values())>1e-12:raise ValueError('ENCODING_REVIEW_REQUIRED: old A kernel changed')
    groups={}
    for a,v in counts.items():groups.setdefault(tuple(v),[]).append(a)
    collisions=[]
    for group in groups.values():
        for i,a in enumerate(group):
            for b in group[i+1:]:
                if not isomorphic_operator_graph(graphs[a],graphs[b]):collisions.append([a,b])
    manifest={'H':2,'dictionaries':dictionaries,'append_only':True,'added':added,'old_to_new_coordinates':mapping,
        'n_max':old_manifest['n_max'],'e_max':old_manifest['e_max'],'phi_dimension':new_offset,'psi_dimension':new_offset+2,
        'A_distance_max_abs_error':error,'A_Matern5_2_kernel_max_abs_errors':errors,'A_coordinate_reembedding_exact':True,
        'nonisomorphic_WL_collisions':collisions,'finite':bool(np.isfinite(np.array(list(features.values()))).all()),
        'structure_exposure':'all frozen provisional/eligible C structures; transductive unlabeled dictionary preparation',
        'performance_access':False,'post_freeze_OOV':'raise ENCODING_BLOCKED','A_formal_input_changed':False}
    manifest['identity']=digest(manifest)
    return features,manifest
