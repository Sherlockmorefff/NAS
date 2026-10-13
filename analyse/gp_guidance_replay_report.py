"""Post-completion evaluation of sealed sequential selections; never fits models."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import statistics as st
import sys


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def table(p, rows):
    with Path(p).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def report(o):
    assert read(o/'all_completed.json')['completed']==20
    labels=read(o/'evaluation_A.json')
    oracle=max(row['validation_mean'] for row in labels.values())
    starts=read(o/'starts.json');curves=[];summaries=[];targets=[];predictions=[]
    methods=['C_GP','S_GP','C_HEAD','RANDOM']
    for seed,s in starts.items():
        initial=max(labels[c]['validation_mean'] for c in s['initial_ids'])
        remaining=[labels[c]['validation_mean'] for c in labels if c not in s['initial_ids']]
        for method in methods:
            folder=o/'trajectories'/f'{seed}-{method}'
            done=read(folder/'completed.json');order=done['query_order']
            assert done['initial_ids']==s['initial_ids'] and len(order)==len(set(order))==50
            assert not set(order)&set(s['initial_ids'])
            truth=[]
            for batch in range(10):
                d=folder/'batches'/f'b{batch:02}';sel=read(d/'selection.json');r=read(d/'reveal.json')
                assert sel['selected_ids']==order[5*batch:5*batch+5]
                assert sel['selected_ids']==[x['candidate_id'] for x in sel['rows'][:5]]
                assert sel['known_ids']==sorted(s['initial_ids']+order[:5*batch])
                for rank,cid in enumerate(sel['selected_ids']):
                    assert r['labels'][cid]==labels[cid]
                    y=r['labels'][cid]['validation_mean'];truth.append(y);pred=sel['rows'][rank]
                    predictions.append(dict(seed=int(seed),method=method,query=5*batch+rank+1,candidate_id=cid,
                        truth=y,score=pred['score'],error=None if pred['score'] is None else pred['score']-y,
                        abs_error=None if pred['score'] is None else abs(pred['score']-y),
                        lower=pred['lower'],upper=pred['upper'],covered=None if pred['lower'] is None else pred['lower']<=y<=pred['upper']))
            best=initial;newbest=None;breaks=0;heights=[];magnitudes=[]
            curves.append(dict(seed=int(seed),method=method,query=0,best=best,new_best=None,gap=oracle-best,gain=0.,record_break=0,record_gain=0.))
            for q,y in enumerate(truth,1):
                gain=max(y-best,0.);isbreak=y>best+1e-12
                breaks+=isbreak
                if isbreak:magnitudes.append(gain)
                best=max(best,y);newbest=y if newbest is None else max(newbest,y);heights.append(best)
                curves.append(dict(seed=int(seed),method=method,query=q,best=best,new_best=newbest,gap=oracle-best,gain=best-initial,record_break=int(isbreak),record_gain=gain))
            summary=dict(seed=int(seed),method=method,initial_best=initial,final_best=best,new_best=newbest,
                gap=oracle-best,curve_mean=st.mean(heights),record_breaks=breaks,total_record_gain=sum(magnitudes),
                no_refresh_space=oracle<=initial+1e-12,queries=50)
            summaries.append(summary)
            for name,threshold,strict in [('strict_initial_improvement',initial,True),('gain_0.5pp',initial+.005,False)]:
                qualify=lambda v:v>threshold+1e-12 if strict else v>=threshold-1e-12
                possible=any(qualify(y) for y in remaining)
                first=next((i for i,y in enumerate(truth,1) if qualify(y)),None)
                targets.append(dict(seed=int(seed),method=method,target=name,initial_best=initial,threshold=threshold,
                    attainable=possible,cost=first if first is not None else ('>50' if possible else 'unattainable'),
                    reached=first is not None))
    pair=[];aggregate=[]
    lookup={(r['seed'],r['method']):r for r in summaries}
    for ref in ['C_GP','RANDOM','C_HEAD']:
        rows=[]
        for seed in sorted(map(int,starts)):
            a,b=lookup[seed,'S_GP'],lookup[seed,ref]
            row=dict(seed=seed,reference=ref,final_delta_pp=100*(a['final_best']-b['final_best']),
                curve_delta_pp=100*(a['curve_mean']-b['curve_mean']),new_best_delta_pp=100*(a['new_best']-b['new_best']))
            pair.append(row);rows.append(row)
        ds=[r['final_delta_pp'] for r in rows]
        aggregate.append(dict(reference=ref,mean_delta_pp=st.mean(ds),median_delta_pp=st.median(ds),
            wins=sum(v>1e-10 for v in ds),ties=sum(abs(v)<=1e-10 for v in ds),losses=sum(v< -1e-10 for v in ds),
            five_deltas_pp=ds,curve_mean_delta_pp=st.mean(r['curve_delta_pp'] for r in rows)))
    fixed=[]
    for method in methods:
        for q in [0,5,10,20,30,40,50]:
            rs=[r for r in curves if r['method']==method and r['query']==q]
            fixed.append(dict(method=method,budget=q,best_percent=100*st.mean(r['best'] for r in rs),
                new_best_percent=None if q==0 else 100*st.mean(r['new_best'] for r in rs)))
    predsum=[]
    for method in methods[:-1]:
        rs=[r for r in predictions if r['method']==method]
        predsum.append(dict(method=method,prequery_MAE_pp=100*st.mean(r['abs_error'] for r in rs),
            prequery_bias_pp=100*st.mean(r['error'] for r in rs),overestimated_fraction=st.mean(r['error']>0 for r in rs),
            coverage95=None if method=='C_HEAD' else st.mean(r['covered'] for r in rs)))
    result=dict(scope='A development, five overlapping starts; no independent confirmation',pool_best=oracle,
        no_refresh_starts=sum(max(labels[c]['validation_mean'] for c in s['initial_ids'])>=oracle-1e-12 for s in starts.values()),
        summaries=summaries,paired=pair,comparisons=aggregate,fixed_budgets=fixed,prediction_explanations=predsum)
    for name,rows in [('curves',curves),('summaries',summaries),('targets',targets),('paired',pair),('prequery_predictions',predictions),('fixed_budgets',fixed)]:
        table(o/(name+'.csv'),rows)
    write(o/'analysis.json',result)
    return result


def plot(o):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((o/'curves.csv').open()))
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    methods=['C_GP','S_GP','C_HEAD','RANDOM']
    colors=['#3274a1','#c44e52','#55a868','#888888']
    for ax,seed in zip(axes.flat,range(2026091600,2026091605)):
        for method,color in zip(methods,colors):
            rs=[r for r in rows if int(r['seed'])==seed and r['method']==method]
            ax.step([int(r['query']) for r in rs],[100*float(r['best']) for r in rs],where='post',label=method,color=color)
        ax.set_title(f'Start {seed}');ax.grid(alpha=.2)
    ax=axes.flat[-1]
    for method,color in zip(methods,colors):
        ys=[st.mean(100*float(r['best']) for r in rows if r['method']==method and int(r['query'])==q) for q in range(51)]
        ax.step(range(51),ys,where='post',label=method,color=color)
    ax.set_title('Mean of five starts');ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.supxlabel('Additional historical queries');fig.supylabel('Best validation accuracy so far (%)')
    fig.suptitle('A development replay: common initial 50, then 50 queries')
    fig.tight_layout();fig.savefig(o/'best_so_far.png',dpi=180);fig.savefig(o/'best_so_far.svg');plt.close(fig)


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve()
    r=report(root);plot(root)
    print(json.dumps(dict(no_refresh_starts=r['no_refresh_starts'],comparisons=r['comparisons']),indent=2))
