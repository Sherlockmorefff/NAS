#!/usr/bin/env python3
"""Independently verify fivefold closure exports using Python's standard library only.

Reads only the supplied evidence directory; no project imports, model loading,
training libraries, optimizers, or access to original performance databases.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics

METRICS=('global_order','local10_order','De_pp','Dstar_pp','E_pp','Drandom_pp','De_minus_random_pp')


def read(path):
    return json.loads(path.read_text())


def same(actual,expected,where):
    if actual is None or expected is None:
        assert actual is expected,where
    else:
        assert abs(actual-expected)<=1e-10,(where,actual,expected)


def ordering(d,g,scale):
    c={'concordant':0,'discordant':0,'distance_ties':0,'performance_ties':0}
    for i,j in itertools.combinations(range(len(d)),2):
        if abs(g[i]-g[j])<=1e-12:
            c['performance_ties']+=1
        elif abs(d[i]-d[j])<=1e-10*scale:
            c['distance_ties']+=1
        elif (g[i]<g[j])==(d[i]<d[j]):
            c['concordant']+=1
        else:
            c['discordant']+=1
    c['total_pairs']=len(d)*(len(d)-1)//2
    c['valid_denominator']=c['total_pairs']-c['performance_ties']
    c['score']=(c['concordant']+.5*c['distance_ties'])/c['valid_denominator'] if c['valid_denominator'] else None
    return c


def values_for(rows,key,weighting):
    if weighting=='query_equal':
        return [r[key] for r in rows if r[key] is not None]
    groups={}
    for r in rows:
        if r[key] is not None:
            groups.setdefault(r['architecture_fingerprint'],[]).append(r[key])
    return [statistics.fmean(v) for v in groups.values()]


def run(root):
    protocol=read(root/'protocol.json');partition=protocol['partition'];aids=partition['A_ids'];folds=partition['folds']
    means=read(root/'allowed_validation_means.json')['means']
    identities=read(root/'candidate_identity_before_labels.json');by_id={r['candidate_id']:r for r in identities}
    assert set(means)==set(aids)==set(by_id)
    assert len(aids)==179 and len({r['architecture_fingerprint'] for r in identities})==96
    allq=[c for f in folds for c in f['query_ids']]
    assert len(allq)==len(set(allq))==len(aids) and set(allq)==set(aids)
    high={}
    for f in folds:
        ta={by_id[c]['architecture_fingerprint'] for c in f['train_ids']}
        qa={by_id[c]['architecture_fingerprint'] for c in f['query_ids']}
        assert not ta&qa and set(f['train_ids'])|set(f['query_ids'])==set(aids)
        high[f['family']]=sorted(f['query_ids'],key=lambda c:(-means[c],c))[:math.ceil(.1*len(f['query_ids']))]
    for r in read(root/'high_performance_queries.json'):
        assert r['candidate_ids']==high[r['family']]
    high_ids={c for ids in high.values() for c in ids}
    refs={(r['state_id'],r['candidate_id']):r for r in read(root/'old_prediction_reference.json')}
    by_state={};all_rows=[]
    inventory={r['state_id']:r for r in read(root/'state_inventory.json')}
    for folder in sorted((root/'states').iterdir()):
        g=read(folder/'geometry_before_labels.json');gapfile=read(folder/'performance_gap_matrix.json');rows=read(folder/'per_query.json')
        fold=next(f for f in folds if f['family']==g['family']);tid,qid=fold['train_ids'],fold['query_ids']
        assert g['train_ids']==tid and g['query_ids']==qid
        assert gapfile['train_ids']==tid and gapfile['query_ids']==qid
        assert g['setting']==protocol['selected_parameters'][fold['family']]['settings'][g['method']]
        assert g['provenance']['completion_identity']==inventory[folder.name]['completion_identity']
        assert g['provenance']['task']['stage']=='refit'
        assert g['standardization']['fit_ids']==tid
        assert len(rows)==len(qid) and len(g['euclidean_distances'])==len(qid)
        for qi,r in enumerate(rows):
            assert r['candidate_id']==qid[qi] and r['family']==fold['family']
            d=g['euclidean_distances'][qi];gaps=[abs(means[qid[qi]]-means[c]) for c in tid]
            assert len(d)==len(tid)
            for j in range(len(tid)):
                same(d[j],math.dist(g['query_embeddings'][qi],g['train_embeddings'][j]),'8D distance')
                same(100*gaps[j],gapfile['performance_gaps_pp'][qi][j],'performance gap matrix')
            ne=sorted(range(len(tid)),key=lambda j:(d[j],tid[j]))[:5]
            ny=sorted(range(len(tid)),key=lambda j:(gaps[j],tid[j]))[:5]
            local=sorted(range(len(tid)),key=lambda j:(gaps[j],tid[j]))[:10]
            assert ne==g['embedding_neighbor_indices'][qi]
            assert r['embedding_neighbor_ids']==[tid[j] for j in ne]
            assert r['oracle_neighbor_ids']==[tid[j] for j in ny]
            assert r['local10_reference_ids']==[tid[j] for j in local]
            for prefix,indices in [('global',list(range(len(tid)))),('local10',local)]:
                result=ordering([d[j] for j in indices],[gaps[j] for j in indices],max(d))
                same(r[f'{prefix}_order'],result.pop('score'),'order score')
                for key,value in result.items():assert r[f'{prefix}_{key}']==value
            de=statistics.fmean(gaps[j] for j in ne);ds=statistics.fmean(gaps[j] for j in ny);dr=statistics.fmean(gaps)
            assert de-ds>=-1e-12
            for key,value in {'De_pp':100*de,'Dstar_pp':100*ds,'E_pp':100*max(0.,de-ds),'Drandom_pp':100*dr,'De_minus_random_pp':100*(de-dr)}.items():
                same(r[key],value,key)
            for tolerance in (.006,.012):
                same(r[f'exceeds_tolerance_{tolerance}_fraction'],sum(gaps[j]>tolerance for j in ne)/5,'tolerance proportion')
            reference=refs[(folder.name,qid[qi])]
            same(r['old_gp_error_pp'],100*abs(reference['old_GP_prediction']-means[qid[qi]]),'old GP error')
            if reference['old_head_prediction'] is not None:
                same(r['old_head_error_pp'],100*abs(reference['old_head_prediction']-means[qid[qi]]),'old head error')
                same(r['old_knn_error_pp'],100*abs(statistics.fmean(means[tid[j]] for j in ne)-means[qid[qi]]),'old knn error')
        by_state[folder.name]=rows;all_rows.extend(rows)
    assert len(by_state)==30 and len(all_rows)==1074
    for method in ('regression','N2'):
        for seed in range(3):
            r=[r for r in all_rows if (r['method'],r['seed'])==(method,seed)]
            assert len(r)==len({x['candidate_id'] for x in r})==179
            assert {x['candidate_id'] for x in r}==set(aids)
    for cid in aids:
        for k in ('Dstar_pp','Drandom_pp'):
            assert len({r[k] for r in all_rows if r['candidate_id']==cid})==1
    original=read(root/'per_query.json')
    assert sorted(original,key=lambda r:(r['state_id'],r['candidate_id']))==sorted(all_rows,key=lambda r:(r['state_id'],r['candidate_id']))
    for s in read(root/'state_summaries.json'):
        r=by_state[s['state_id']]
        if s['subset']!='all_queries':r=[x for x in r if x['candidate_id'] in high_ids]
        v=values_for(r,s['metric'],s['weighting'])
        assert s['count']==len(v)
        for key,func in [('mean',statistics.fmean),('median',statistics.median),('min',min),('max',max)]:same(s[key],func(v) if v else None,'state '+key)
    recomputed={}
    for p in read(root/'paired_summary.json'):
        ids=aids if p['scope']=='overall' else next(f['query_ids'] for f in folds if f['family']==p['scope'])
        if p['subset']!='all_queries':ids=[cid for cid in ids if cid in high_ids]
        deltas=[];method_values={'regression':[],'N2':[]}
        for seed in range(3):
            a={r['candidate_id']:r for r in all_rows if r['method']=='regression' and r['seed']==seed}
            b={r['candidate_id']:r for r in all_rows if r['method']=='N2' and r['seed']==seed}
            common=[];excluded=[];key=p['metric']
            for cid in ids:
                if a[cid][key] is None or b[cid][key] is None:excluded.append(cid)
                else:common.append({**a[cid],'delta':b[cid][key]-a[cid][key],'regression':a[cid][key],'N2':b[cid][key]})
            vals=values_for(common,'delta',p['weighting'])
            delta=statistics.fmean(vals) if vals else None
            same(p[f'seed{seed}_delta'],delta,'paired seed delta')
            assert p['per_seed'][str(seed)]['valid_query_count']==len(common)
            assert p['per_seed'][str(seed)]['excluded_query_count']==len(excluded)
            if delta is not None:deltas.append(delta)
            for method in method_values:
                v=values_for(common,method,p['weighting'])
                if v:method_values[method].append(statistics.fmean(v))
        average=statistics.fmean(deltas) if deltas else None
        same(p['delta_distribution']['mean'],average,'paired seed mean')
        same(p['delta_distribution']['min'],min(deltas) if deltas else None,'paired min')
        same(p['delta_distribution']['max'],max(deltas) if deltas else None,'paired max')
        for method,v in method_values.items():same(p[method+'_mean'],statistics.fmean(v) if v else None,'paired method mean')
        if p['subset']=='all_queries' and p['weighting']=='query_equal':recomputed[(p['scope'],key)]=average
    def clean(value,tolerance):return 0. if abs(value)<=tolerance else value
    c=clean(recomputed[('overall','global_order')],1e-12);e=clean(recomputed[('overall','E_pp')],1e-10)
    joint=sum(clean(recomputed[(f['family'],'global_order')],1e-12)>0 and clean(recomputed[(f['family'],'E_pp')],1e-10)<0 for f in folds)
    decision=read(root/'closure_decision.json')
    assert decision['material_complete'] and decision['joint_improvement_fold_count']==joint
    assert decision['rule_satisfied']==(c>0 and e<0 and joint>=4)
    same(decision['raw_overall']['global_order'],recomputed[('overall','global_order')],'closure C')
    same(decision['raw_overall']['E_pp'],recomputed[('overall','E_pp')],'closure E')
    return {'status':'passed','states':30,'query_state_rows':len(all_rows),'A_unique_candidates':179,
            'global_pair_comparisons':sum(r['global_total_pairs'] for r in all_rows),'local_pair_comparisons':sum(r['local10_total_pairs'] for r in all_rows),
            'closure_rule_satisfied':decision['rule_satisfied'],'joint_improvement_fold_count':joint,
            'checks':'fold/OOF identity; Euclidean/gap matrices; global/local counts; oracle/retrieved IDs; errors; excess/random gaps; state distributions; common-query paired means/ranges; fixed closure rule',
            'runtime_dependencies':'Python standard library only; no project/model/training imports','training_calls':0,'optimization_updates':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('evidence_directory',type=Path)
    print(json.dumps(run(parser.parse_args().evidence_directory),indent=2))
