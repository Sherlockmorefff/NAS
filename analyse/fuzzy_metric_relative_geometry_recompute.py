#!/usr/bin/env python3
"""Independently verify exported geometry using Python's standard library only.

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
    means=read(root/'allowed_validation_means.json')
    protocol=read(root/'protocol.json')
    tid,qid=protocol['train_ids'],protocol['query_ids']
    assert set(means['train'])==set(tid) and set(means['query'])==set(qid)
    assert len(tid)==112 and len(qid)==29
    top=sorted(qid,key=lambda c:(-means['query'][c],c))[:3]
    assert top==[r['candidate_id'] for r in read(root/'high_performance_queries.json')['queries']]
    candidates=read(root/'candidate_identity_before_labels.json')
    train_arch={r['architecture_fingerprint'] for r in candidates['train']}
    query_arch={r['architecture_fingerprint'] for r in candidates['query']}
    assert len(query_arch)==16 and not train_arch&query_arch
    by_state={}
    for folder in sorted((root/'states').iterdir()):
        saved=read(folder/'geometry_before_labels.json')
        gaps_file=read(folder/'performance_gap_matrix.json')
        rows=read(folder/'per_query.json')
        assert saved['train_ids']==tid and saved['query_ids']==qid
        assert gaps_file['train_ids']==tid and gaps_file['query_ids']==qid
        assert len(saved['euclidean_distances'])==29 and len(rows)==29
        for qi,row in enumerate(rows):
            assert row['candidate_id']==qid[qi]
            d=saved['euclidean_distances'][qi]
            g=[abs(means['query'][qid[qi]]-means['train'][c]) for c in tid]
            assert len(d)==112
            for j in range(112):
                same(d[j],math.dist(saved['query_embeddings'][qi],saved['train_embeddings'][j]),'8D Euclidean distance')
                same(gaps_file['performance_gaps_pp'][qi][j],100*g[j],'performance gap matrix')
            ne=sorted(range(112),key=lambda j:(d[j],tid[j]))[:5]
            ny=sorted(range(112),key=lambda j:(g[j],tid[j]))[:5]
            local=sorted(range(112),key=lambda j:(g[j],tid[j]))[:10]
            assert ne==saved['embedding_neighbor_indices'][qi]
            assert row['embedding_neighbor_ids']==[tid[j] for j in ne]
            assert row['oracle_neighbor_ids']==[tid[j] for j in ny]
            assert row['local10_reference_ids']==[tid[j] for j in local]
            for prefix,which in [('global',list(range(112))),('local10',local)]:
                counters=ordering([d[j] for j in which],[g[j] for j in which],max(d))
                same(row[f'{prefix}_order'],counters.pop('score'),'ordering score')
                for key,value in counters.items():
                    assert row[f'{prefix}_{key}']==value,(folder.name,qi,prefix,key)
            de=statistics.fmean(g[j] for j in ne)
            ds=statistics.fmean(g[j] for j in ny)
            dr=statistics.fmean(g)
            assert de-ds>=-1e-12
            for key,value in {'De_pp':100*de,'Dstar_pp':100*ds,'E_pp':100*max(0.,de-ds),'Drandom_pp':100*dr,'De_minus_random_pp':100*(de-dr)}.items():
                same(row[key],value,key)
        by_state[folder.name]=rows
    assert len(by_state)==12
    expected_rows=[r for sid in sorted(by_state) for r in by_state[sid]]
    exported=read(root/'per_query_348.json')
    assert sorted(exported,key=lambda r:(r['state_id'],r['candidate_id']))==sorted(expected_rows,key=lambda r:(r['state_id'],r['candidate_id']))
    for s in read(root/'summaries.json'):
        rows=by_state[s['state_id']]
        if s['subset']=='top3_descriptive':
            rows=[r for r in rows if r['candidate_id'] in top]
        vals=values_for(rows,s['metric'],s['weighting'])
        assert s['count']==len(vals)
        same(s['mean'],statistics.fmean(vals) if vals else None,'summary mean')
        same(s['median'],statistics.median(vals) if vals else None,'summary median')
        same(s['min'],min(vals) if vals else None,'summary min')
        same(s['max'],max(vals) if vals else None,'summary max')
    for p in read(root/'paired_differences.json'):
        name=p['comparison']
        if name.endswith(': M300-M100'):
            method=name.split(':')[0];a,b=(method,100),(method,300)
        else:
            steps=int(name.split(':')[0][1:]);a,b=('regression',steps),('N2',steps)
        deltas=[]
        for seed in range(3):
            arows=by_state[f'{a[0]}-s{seed}-M{a[1]}'];brows=by_state[f'{b[0]}-s{seed}-M{b[1]}']
            common=[];excluded=[];key=p['metric']
            for x,y in zip(arows,brows):
                assert x['candidate_id']==y['candidate_id']
                if x[key] is None or y[key] is None:
                    excluded.append(x['candidate_id'])
                else:
                    common.append({**x,key:y[key]-x[key]})
            vals=values_for(common,key,p['weighting'])
            delta=statistics.fmean(vals) if vals else None
            same(p[f'seed{seed}'],delta,'paired seed delta')
            assert p['seed_results'][str(seed)]['excluded_query_count']==len(excluded)
            if delta is not None:deltas.append(delta)
        same(p['mean'],statistics.fmean(deltas) if deltas else None,'paired mean')
        same(p['min'],min(deltas) if deltas else None,'paired min')
        same(p['max'],max(deltas) if deltas else None,'paired max')
    return {'status':'passed','states':len(by_state),'query_state_rows':len(exported),
            'global_pair_comparisons':sum(r['global_total_pairs'] for r in exported),
            'local_pair_comparisons':sum(r['local10_total_pairs'] for r in exported),
            'checks':'Euclidean/gap matrices; global/local counts and scores; neighbor IDs; excess/random gaps; means/medians/ranges; same-seed paired deltas; identity boundary',
            'runtime_dependencies':'Python standard library only; no project/model/training imports',
            'training_calls':0,'optimization_updates':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_directory',type=Path)
    print(json.dumps(run(parser.parse_args().evidence_directory),indent=2))
