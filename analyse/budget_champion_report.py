"""Endpoint-only analysis of sealed predictions and finished A-only replay."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import spearmanr
from analyse.relative_metric_pilot import read,write,load
from surrogate.gp_guidance_replay_engine import journal

METHODS=['C_HEAD','BR_HEAD','RANDOM'];KS=[1,5,10,20];BUDGETS=[5,10,20,30,40,50]
def stats(xs):
    return dict(mean=float(np.mean(xs)),median=float(np.median(xs)),win=sum(x>1e-12 for x in xs),tie=sum(abs(x)<=1e-12 for x in xs),loss=sum(x< -1e-12 for x in xs),values=xs)
def analyse(o):
    assert read(o/'all_completed.json')['offline']==30
    labels=read(o/'evaluation_A.json');y={c:v['validation_mean'] for c,v in labels.items()}
    poolbest=max(y.values());champions={c for c,v in y.items() if abs(v-poolbest)<=1e-12}
    tasks=read(o/'tasks.json');offline=[];replay=[];traces=[];diagnostics=[];training=0.;saving=0.;scoring=0.;oracle=0.
    for task in tasks:
        folder=o/task['stage']/task['name']
        for event in sorted((folder/'events').glob('*')):
            st=load(event/'latest.pt');h=st['history'];ts=journal(event/'timing.jsonl')
            training+=sum(t['compute_seconds'] for t in ts);saving+=sum(t['save_and_state_seconds'] for t in ts)
            diagnostics.append(dict(task=task['name'],stage=task['stage'],method=task['method'],event=event.name,
                steps=st['step'],rho=st['rho'],initial_loss=h[0],final_loss=h[-1],
                mean_nonzero_fraction=float(np.mean([x['nonzero_fraction'] for x in h])),
                mean_ranking_loss=float(np.mean([x['ranking'] for x in h])),
                champion_rank_in_D_mean=float(np.mean([r for x in h for r in x['champion_ranks_in_D']])),
                gradient_checks=st['gradient_checks']))
        if task['stage']=='offline':
            p=read(folder/'predictions.json');rows=p['rows'];ids=[r['candidate_id'] for r in rows];scoring+=p['score_seconds']
            best=max(y[c] for c in ids);w={c for c in ids if abs(y[c]-best)<=1e-12};N=len(ids)
            metrics={str(k):dict(hit=int(bool(w&set(ids[:k]))),best=max(y[c] for c in ids[:k]),gap_pp=100*(best-max(y[c] for c in ids[:k])),
                random_hit_probability=1-math.comb(N-len(w),k)/math.comb(N,k)) if k<=N else None for k in KS}
            offline.append(dict(task=task['name'],fold=task['fold'],seed=task['model_seed'],method=task['method'],N=N,champions=sorted(w),
                champion_best=best,champion_rank=min(ids.index(c)+1 for c in w),K=metrics,
                head_MAE=float(np.mean([abs(r['score']-y[r['candidate_id']]) for r in rows])),
                spearman=float(spearmanr([r['score'] for r in rows],[y[r['candidate_id']] for r in rows]).statistic)))
        else:
            initial=task['initial_ids'];known=set(initial);chosen=[];base=max(y[c] for c in initial)
            curve=[base];hitcurve=[int(bool(champions&known))]
            for b in range(10):
                d=folder/'batches'/f'b{b:02}';s=read(d/'selection.json');scoring+=s['score_seconds'];oracle+=read(d/'oracle_timing.json')['seconds']
                ranks={r['candidate_id']:(i+1,r['score']) for i,r in enumerate(s['rows'])}
                traces.append(dict(task=task['name'],seed=task['search_seed'],method=task['method'],batch=b,queries_before=5*b,version=s['version'],
                    champions=[dict(candidate_id=c,already_known=c in known,rank=ranks[c][0] if c in ranks else None,
                        score=ranks[c][1] if c in ranks else None,selected=c in s['selected_ids']) for c in sorted(champions)]))
                for c in s['selected_ids']:
                    assert c not in known;known.add(c);chosen.append(c);curve.append(max(curve[-1],y[c]));hitcurve.append(int(bool(champions&known)))
            def first(condition,possible):
                if not possible:return 'pool_unattainable'
                return next((i for i in range(1,51) if condition(curve[i])), '>50')
            replay.append(dict(task=task['name'],seed=task['search_seed'],method=task['method'],initial_ids=initial,query_order=chosen,
                initial_best=base,initial_hit=hitcurve[0],final_best=curve[-1],final_hit=hitcurve[-1],
                first_champion_query=next((i for i,h in enumerate(hitcurve) if h),'>50'),
                strict_improvement_query=first(lambda v:v>base+1e-12,poolbest>base+1e-12),
                improve_half_pp_query=first(lambda v:v>=base+.005-1e-12,poolbest>=base+.005-1e-12),
                new_selected_best=max(y[c] for c in chosen),curve=curve,hit_curve=hitcurve,curve_average=float(np.mean(curve[1:])),
                checkpoints={str(b):dict(best=curve[b],gap_pp=100*(poolbest-curve[b]),hit=hitcurve[b]) for b in BUDGETS}))
    fold_summary=[]
    for fold in range(5):
        d={m:[r for r in offline if r['fold']==fold and r['method']==m] for m in METHODS[:2]}
        fold_summary.append(dict(fold=fold,methods={m:dict(hit10=sum(r['K']['10']['hit'] for r in rs),seeds=3,
            mean_gap10_pp=float(np.mean([r['K']['10']['gap_pp'] for r in rs])),mean_champion_rank=float(np.mean([r['champion_rank'] for r in rs]))) for m,rs in d.items()},
            paired_hit_BR_minus_C=[b['K']['10']['hit']-a['K']['10']['hit'] for a,b in zip(d['C_HEAD'],d['BR_HEAD'])],
            paired_best10_BR_minus_C_pp=[100*(b['K']['10']['best']-a['K']['10']['best']) for a,b in zip(d['C_HEAD'],d['BR_HEAD'])]))
    pairs={}
    for a,b in [('BR_HEAD','C_HEAD'),('BR_HEAD','RANDOM'),('C_HEAD','RANDOM')]:
        aa=sorted([r for r in replay if r['method']==a],key=lambda r:r['seed']);bb=sorted([r for r in replay if r['method']==b],key=lambda r:r['seed'])
        pairs[a+'-minus-'+b]={k:stats([(100 if k!='final_hit' else 1)*(x[k]-z[k]) for x,z in zip(aa,bb)]) for k in ['final_best','final_hit','curve_average']}
    summary=dict(offline={m:dict(hit10=sum(r['K']['10']['hit'] for r in offline if r['method']==m),trajectories=15,
            mean_gap10_pp=float(np.mean([r['K']['10']['gap_pp'] for r in offline if r['method']==m]))) for m in METHODS[:2]},
        replay={m:dict(champion_hits=sum(r['final_hit'] for r in replay if r['method']==m),trajectories=5,
            mean_final_best=float(np.mean([r['final_best'] for r in replay if r['method']==m])),mean_curve=float(np.mean([r['curve_average'] for r in replay if r['method']==m]))) for m in METHODS},
        pool_best=poolbest,champions=sorted(champions),initial_champion_starts=sorted({r['seed'] for r in replay if r['initial_hit']}))
    budget=read(o/'all_completed.json')['cost'];budget.update(effective=20000,failed=0,discarded=0,uncertain=0,
        synthetic_updates=read(o/'checks_complete.json')['synthetic_optimizer_updates'],GNN_trainings=0,VAE_trainings=0,initial_logical_labels=300,initial_shared_by_start=100)
    by_method={}
    for task in tasks:
        key=task['stage']+'/'+task['method'];v=by_method.setdefault(key,dict(training_compute_seconds=0.,training_save_state_seconds=0.,scoring_seconds=0.,oracle_io_seconds=0.))
        folder=o/task['stage']/task['name']
        for pp in (folder/'events').glob('*/timing.jsonl'):
            for x in journal(pp):v['training_compute_seconds']+=x['compute_seconds'];v['training_save_state_seconds']+=x['save_and_state_seconds']
        if task['stage']=='offline':v['scoring_seconds']+=read(folder/'predictions.json')['score_seconds']
        else:
            for pp in (folder/'batches').glob('*/selection.json'):v['scoring_seconds']+=read(pp)['score_seconds']
            for pp in (folder/'batches').glob('*/oracle_timing.json'):v['oracle_io_seconds']+=read(pp)['seconds']
    runs=journal(o/'runs.jsonl');timing=dict(by_stage_method=by_method,training_compute_seconds=training,scoring_seconds=scoring,oracle_io_seconds=oracle,
        training_save_state_seconds=saving,first_run_wall_seconds=runs[0]['seconds'],
        other_startup_selection_save_verification_seconds=runs[0]['seconds']-training-scoring-oracle-saving,
        note='Compute includes deterministic lists, loss/gradient diagnostics and optimizer; excludes ledger/checkpoint IO. Remainder includes process startup and nontraining saves; audit/report/verification runs are separate.')
    for name,obj in [('offline_results',offline),('replay_results',replay),('fold_summary',fold_summary),('paired_results',pairs),('champion_traces',traces),('training_diagnostics',diagnostics),('summary',summary),('budget',budget),('timing',timing)]:
        write(o/'analysis'/f'{name}.json',obj,immutable=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    for ax,seed in zip(axs.flat,range(2026091600,2026091605)):
        for m in METHODS:
            r=next(r for r in replay if r['seed']==seed and r['method']==m);ax.step(range(51),100*np.array(r['curve']),where='post',label=m)
        ax.axhline(100*poolbest,color='gray',linestyle=':');ax.set_title(str(seed));ax.set_xlabel('New historical queries');ax.set_ylabel('Best validation accuracy (%)');ax.grid(alpha=.2)
    axs.flat[-1].axis('off');axs.flat[0].legend();fig.tight_layout();fig.savefig(o/'analysis/best_so_far.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/posthoc/budget-champion-ranking-v1');a=p.parse_args();analyse(a.output)
