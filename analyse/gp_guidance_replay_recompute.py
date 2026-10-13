"""Independent standard-library reconstruction from sealed selections + A seed scores.

Does not import the runner or primary analysis; primary tables are read only
after all independent values have been constructed, for comparison.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def read(p):
    return json.loads(Path(p).read_text())


def main(o):
    labels=read(o/'evaluation_A.json')
    y={c:sum(r['validation_seeds'])/3 for c,r in labels.items()}
    assert len(y)==179
    for cid,row in labels.items():
        assert abs(row['validation_mean']-y[cid])<1e-12
        se=math.sqrt(sum((v-y[cid])**2 for v in row['validation_seeds'])/2/3)
        assert abs(se-row['validation_se'])<1e-12
    starts=read(o/'starts.json');computed={};targets={};curves={};query_total=0
    for seed in range(2026091600,2026091605):
        s=starts[str(seed)]
        ids=sorted(y,key=lambda c:(hashlib.sha256(f'gp_guidance_replay_v1:init:{seed}:{c}'.encode()).hexdigest(),c))[:50]
        assert ids==s['initial_ids']
        initial=max(y[c] for c in ids)
        for method in ['C_GP','S_GP','C_HEAD','RANDOM']:
            selected=[]
            for batch in range(10):
                p=o/'trajectories'/f'{seed}-{method}'/'batches'/f'b{batch:02}'
                r=read(p/'selection.json');receipt=read(p/'reveal.json')
                assert r['known_ids']==sorted(ids+selected)
                assert r['query_count_before']==batch*5
                assert receipt['selection_sha']==hashlib.sha256((p/'selection.json').read_bytes()).hexdigest()
                if method=='RANDOM':
                    order=sorted([c for c in y if c not in ids],key=lambda c:(hashlib.sha256(f'gp_guidance_replay_v1:random:{seed}:{c}'.encode()).hexdigest(),c))
                    assert order==s['random_order']
                    expected=order[batch*5:batch*5+5]
                else:
                    assert {x['candidate_id'] for x in r['rows']}==set(y)-set(ids+selected)
                    order=sorted(r['rows'],key=lambda x:(-x['score'],x['candidate_id']))
                    assert order==r['rows']
                    expected=[x['candidate_id'] for x in order[:5]]
                assert r['selected_ids']==expected and not set(expected)&set(ids+selected)
                selected.extend(expected)
                for c in expected:assert receipt['labels'][c]==labels[c]
            assert len(set(ids+selected))==100
            query_total+=len(selected)
            path=[max(y[c] for c in ids+selected[:k]) for k in range(51)]
            curves[seed,method]=path
            key=(seed,method)
            computed[key]=dict(initial_best=initial,final_best=path[-1],curve_mean=sum(path[1:])/50,
                new_best=max(y[c] for c in selected),gap=max(y.values())-path[-1],
                record_breaks=sum(path[k]>path[k-1]+1e-12 for k in range(1,51)))
            for name,threshold,strict in [('strict_initial_improvement',initial,True),('gain_0.5pp',initial+.005,False)]:
                qualifies=lambda c:y[c]>threshold+1e-12 if strict else y[c]>=threshold-1e-12
                hit=next((k for k,c in enumerate(selected,1) if qualifies(c)),None)
                attainable=any(qualifies(c) for c in y if c not in ids)
                targets[seed,method,name]=str(hit) if hit is not None else ('>50' if attainable else 'unattainable')
    comparisons={}
    for ref in ['C_GP','RANDOM','C_HEAD']:
        values=[100*(computed[s,'S_GP']['final_best']-computed[s,ref]['final_best']) for s in range(2026091600,2026091605)]
        comparisons[ref]=dict(five_deltas_pp=values,mean=statistics.mean(values),median=statistics.median(values),
            wins=sum(v>1e-10 for v in values),ties=sum(abs(v)<=1e-10 for v in values),losses=sum(v< -1e-10 for v in values))
    differences=[]
    for row in csv.DictReader((o/'curves.csv').open()):
        differences.append(abs(float(row['best'])-curves[int(row['seed']),row['method']][int(row['query'])]))
    for row in csv.DictReader((o/'summaries.csv').open()):
        expected=computed[int(row['seed']),row['method']]
        differences.extend(abs(float(row[k])-v) for k,v in expected.items())
    for row in csv.DictReader((o/'targets.csv').open()):
        assert row['cost']==targets[int(row['seed']),row['method'],row['target']]
    for row in csv.DictReader((o/'paired.csv').open()):
        expected=comparisons[row['reference']]['five_deltas_pp'][int(row['seed'])-2026091600]
        differences.append(abs(float(row['final_delta_pp'])-expected))
    assert max(differences)<1e-10 and query_total==1000
    out=dict(passed=True,independent_trajectories=20,queries=query_total,curve_points=1020,target_costs=40,
        numeric_checks=len(differences),maximum_difference=max(differences),comparisons=comparisons,
        optimizer_updates=0,source='selection order and original three validation seed scores; no primary summary reused')
    (o/'independent_recomputation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
