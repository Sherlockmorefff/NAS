"""Post-seal evaluation only. Never called by a selector or optimizer."""
import sys,statistics,itertools,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from analyse.fuzzy_convergence_core import *

def avg(xs):return float(statistics.mean(xs))
def med(xs):return float(statistics.median(xs))


def evaluate(o):
    started=time.perf_counter();tasks=read(o/'tasks.json');truth={c:r['validation_mean'] for c,r in read(o/'private/evaluation_A.json').items()}
    tables={k:[] for k in ['convergence_summary','offline_comparison','offline_predictions','offline_main','offline_paired','replay_summary','replay_paired','replay_queries','replay_champion_ranks']}
    # All training has ended or been explicitly failed before any outer truth is read here.
    assert (o/'execution_complete.json').exists()
    for task in tasks:
        folder=o/'tasks'/task['name']
        assert (folder/'completed.json').exists() or (folder/'failure.json').exists()
        for sp in sorted(folder.rglob('summary.json')):
            s=read(sp);last=s['decisions'][-1] if s['decisions'] else {}
            tables['convergence_summary'].append(dict(task=task['name'],stage=task['stage'],group=task['group'],kind=s['kind'],
                path=str(sp.relative_to(o)),source='new',status=s['status'],completed_updates=s['completed_updates'],
                failed_conditions=last.get('failed'),final_learning_rate=s['final_learning_rate'],stable_windows=s['scheduler']['stable_count'],
                optimization_seconds=s['optimization_seconds'],fit_seconds=s['fit_seconds'],metadata=s['metadata'],
                gradient_rms=last.get('measures',{}).get('gradient_rms_max'),distance_l2=last.get('measures',{}).get('distance_l2'),
                distance_pair=last.get('measures',{}).get('distance_pair'),prediction_range=last.get('measures',{}).get('prediction_range')))
        if task['stage']!='offline' or not (folder/'completed.json').exists():continue
        for mt,gt in itertools.product(['FIXED','ADAPTIVE'],repeat=2):
            prediction=read(folder/f'predictions-{mt}-{gt}.json');rows=prediction['rows']
            ms=load(folder/'mapping'/('post100.pt' if mt=='FIXED' else 'latest.pt'))
            gs=load(folder/f'gp-{mt}'/('post020.pt' if gt=='FIXED' else 'latest.pt'))
            common=dict(task=task['name'],fold=task['fold'],model_seed=task['model_seed'],group=task['group'],mapping_protocol=mt,gp_protocol=gt,
                mapping_status='FIXED_STEP_STOP' if mt=='FIXED' else ms['status'],gp_status='FIXED_STEP_STOP' if gt=='FIXED' else gs['status'],
                mapping_steps=ms['completed_updates'],gp_steps=gs['completed_updates'],mapping_optimization_seconds=ms['optimization_seconds'],gp_optimization_seconds=gs['optimization_seconds'],
                mapping_fit_seconds=ms['fit_seconds'],gp_fit_seconds=gs['fit_seconds'])
            for row in rows:
                c=row['candidate_id'];tables['offline_predictions'].append(dict(common,**row,true_validation=truth[c],
                    neighbor_difference=avg([abs(truth[c]-truth[n]) for n in row['neighbor_ids']])))
            best=max(truth[r['candidate_id']] for r in rows);champs={r['candidate_id'] for r in rows if best-truth[r['candidate_id']]<=1e-12}
            for score in ['head_mean','gp_mean']:
                ranked=sorted(rows,key=lambda r:(-r[score],r['candidate_id']));rank=min(i+1 for i,r in enumerate(ranked) if r['candidate_id'] in champs)
                row=dict(common,score=score,mae_raw=avg([abs(r[score]-truth[r['candidate_id']]) for r in rows]),champion_rank=rank,
                    neighbor_difference_fold_median=med([avg([abs(truth[r['candidate_id']]-truth[n]) for n in r['neighbor_ids']]) for r in rows]),query_count=len(rows))
                for k in [5,10,20]:row.update({f'top{k}_champion_hit':rank<=k,f'top{k}_best_gap_raw':best-max(truth[r['candidate_id']] for r in ranked[:k])})
                tables['offline_comparison'].append(row)
    for group,mt,gt in itertools.product(['C','F','W'],['FIXED','ADAPTIVE'],['FIXED','ADAPTIVE']):
        rows=[r for r in tables['offline_predictions'] if (r['group'],r['mapping_protocol'],r['gp_protocol'])==(group,mt,gt)]
        if len(rows)!=537:continue
        tables['offline_main'].append(dict(group=group,mapping_protocol=mt,gp_protocol=gt,query_rows=len(rows),independent_datasets=1,
            head_mae_raw=avg([abs(r['head_mean']-r['true_validation']) for r in rows]),gp_mae_raw=avg([abs(r['gp_mean']-r['true_validation']) for r in rows]),
            primary_neighbor_raw=avg([med([r['neighbor_difference'] for r in rows if r['model_seed']==s]) for s in range(3)])))
    lookup={(r['task'],r['mapping_protocol'],r['gp_protocol'],r['score']):r for r in tables['offline_comparison']}
    for t in tasks:
        if t['stage']!='offline':continue
        for score in ['head_mean','gp_mean']:
            for target in [('FIXED','ADAPTIVE'),('ADAPTIVE','FIXED'),('ADAPTIVE','ADAPTIVE')]:
                a=lookup.get((t['name'],*target,score));b=lookup.get((t['name'],'FIXED','FIXED',score))
                if a and b:tables['offline_paired'].append(dict(task=t['name'],fold=t['fold'],model_seed=t['model_seed'],group=t['group'],score=score,
                    comparison='/'.join(target)+' - FIXED/FIXED',mae_difference_raw=a['mae_raw']-b['mae_raw'],champion_rank_difference=a['champion_rank']-b['champion_rank']))
    for mt,gt,score in itertools.product(['FIXED','ADAPTIVE'],['FIXED','ADAPTIVE'],['head_mean','gp_mean']):
        for fold,seed in itertools.product(range(5),range(3)):
            for g,base in [('F','C'),('W','C'),('W','F')]:
                a=lookup.get((f'offline-f{fold}-s{seed}-{g}',mt,gt,score));b=lookup.get((f'offline-f{fold}-s{seed}-{base}',mt,gt,score))
                if a and b:tables['offline_paired'].append(dict(fold=fold,model_seed=seed,mapping_protocol=mt,gp_protocol=gt,score=score,comparison=g+'-'+base,
                    mae_difference_raw=a['mae_raw']-b['mae_raw'],champion_rank_difference=a['champion_rank']-b['champion_rank']))
    paths=[(o/'tasks'/t['name'],t,'ADAPTIVE',False) for t in tasks if t['stage']=='replay']
    paths += [(o/'imports'/r['task']['name'],r['task'],'RANDOM' if r['task']['group']=='RANDOM' else 'FIXED',True) for r in read(o/'reuse_manifest.json')['trajectories']]
    curves={};champions={c for c in truth if max(truth.values())-truth[c]<=1e-12}
    for folder,t,protocol,imported in paths:
        known=set(t['initial_ids']);best=max(truth[c] for c in known);first=0 if known&champions else None;query_order=[];curve={0:best}
        common=dict(task=t['name'],group=t['group'],protocol=protocol,search_seed=t['search_seed'],imported=imported)
        tables['replay_summary'].append(dict(common,queries=0,best_validation=best,best_gap=max(truth.values())-best,champion_hit=first is not None,first_champion_query=first if first is not None else '>50',first_champion_batch=0 if first==0 else None))
        for b in range(10):
            bd=folder/'batches'/f'b{b:02}'
            if not (bd/'reveal.json').exists():break
            s=read(bd/'selection.json');r=read(bd/'reveal.json');assert s['known_ids']==sorted(known) and r['selection_sha']==file_hash(bd/'selection.json')
            scores={row['candidate_id']:row for row in s['rows']};rankings={}
            if scores:
                for score in ['head_mean','gp_mean']:rankings[score]={c:i+1 for i,c in enumerate(sorted(scores,key=lambda c:(-scores[c][score],c)))}
                for c in sorted(champions-known):
                    tables['replay_champion_ranks'].append(dict(common,batch=b+1,queries_before=b*5,candidate_id=c,head_rank=rankings['head_mean'][c],gp_rank=rankings['gp_mean'][c],
                        head_error=scores[c]['head_mean']-truth[c],gp_error=scores[c]['gp_mean']-truth[c]))
            for i,c in enumerate(r['ids']):
                assert c not in known;known.add(c);query_order.append(c);best=max(best,truth[c]);q=len(query_order);curve[q]=best
                if c in champions and first is None:first=q
                tables['replay_queries'].append(dict(common,query=q,batch=b+1,position_in_batch=i+1,candidate_id=c,true_validation=truth[c],best_so_far=best,
                    gp_mean=scores.get(c,{}).get('gp_mean'),head_mean=scores.get(c,{}).get('head_mean'),gp_rank=rankings.get('gp_mean',{}).get(c),
                    signed_error=(scores[c]['gp_mean']-truth[c]) if scores else None,fallback=s.get('fallback',False),
                    mapping_status=s.get('mapping_status','FIXED_STEP_STOP' if scores else 'RANDOM'),gp_status=s.get('gp_status','FIXED_STEP_STOP' if scores else 'RANDOM')))
            tables['replay_summary'].append(dict(common,queries=len(query_order),best_validation=best,best_gap=max(truth.values())-best,champion_hit=first is not None,
                first_champion_query=first if first is not None else '>50',first_champion_batch=(first+4)//5 if first is not None else None,
                fallback=s.get('fallback',False)))
        curves[(t['search_seed'],protocol,t['group'])]=curve
        if imported and t['group']!='RANDOM':
            for sp in list((folder/'mapping').rglob('post100.pt'))+list((folder/'batches').rglob('gp/latest.pt')):
                st=load(sp);kind='mapping' if '/mapping/' in str(sp) else 'gp'
                tables['convergence_summary'].append(dict(task=t['name'],stage='replay',group=t['group'],kind=kind,path=str(sp.relative_to(o)),source='imported',status='FIXED_STEP_STOP',
                    completed_updates=st['completed_updates'],failed_conditions='adaptive window not executed',final_learning_rate=st['optimizer']['param_groups'][0]['lr'],optimization_seconds=st['optimization_seconds']))
    comparisons=[(('ADAPTIVE',g),('FIXED',g)) for g in ['C','F','W']]
    comparisons += [((p,a),(p,b)) for p in ['FIXED','ADAPTIVE'] for a,b in [('F','C'),('W','C'),('W','F')]]
    comparisons += [((p,g),('RANDOM','RANDOM')) for p in ['FIXED','ADAPTIVE'] for g in ['C','F','W']]
    for a,b in comparisons:
        for q in [5,10,20,30,40,50]:
            deltas=[]
            for seed in range(2026091900,2026091905):
                ca=curves[(seed,*a)];cb=curves[(seed,*b)]
                complete=q in ca and q in cb;delta=ca[q]-cb[q] if complete else None
                if complete:deltas.append(delta)
                tables['replay_paired'].append(dict(comparison='/'.join(a)+' - '+'/'.join(b),queries=q,search_seed=seed,complete=complete,difference_raw=delta))
            tables['replay_paired'].append(dict(comparison='/'.join(a)+' - '+'/'.join(b),queries=q,search_seed='all_five',complete=len(deltas)==5,
                difference_raw=avg(deltas) if len(deltas)==5 else None,wins=sum(v>1e-12 for v in deltas),ties=sum(abs(v)<=1e-12 for v in deltas),losses=sum(v<-1e-12 for v in deltas),completed_pairs=len(deltas)))
    main=[]
    for p,g in [('FIXED',g) for g in ['C','F','W']]+[('ADAPTIVE',g) for g in ['C','F','W']]+[('RANDOM','RANDOM')]:
        rows=[r for r in tables['replay_summary'] if r['protocol']==p and r['group']==g and r['queries']==50]
        main.append(dict(protocol=p,group=g,completed=len(rows),mean_best_raw=avg([r['best_validation'] for r in rows]) if len(rows)==5 else None,
            mean_gap_raw=avg([r['best_gap'] for r in rows]) if len(rows)==5 else None,champion_hits=sum(r['champion_hit'] for r in rows)))
    for n,rows in tables.items():write(o/'analysis/tables'/(n+'.json'),rows)
    counts={k:len(journal(o/'ledger'/(k+'_intents.jsonl'))) for k in ['mapping','gp']}
    for k in counts:assert counts[k]==len(journal(o/'ledger'/(k+'_receipts.jsonl'))) and counts[k]<=LIMITS[k]
    resource=dict(actual_updates=counts,GP_fits=len(journal(o/'ledger/gp_fits.jsonl')),synthetic=read(o/'engineering_checks.json')['synthetic'],
        actual_new_history_queries=len(journal(o/'oracle_access.jsonl')),actual_initial_accesses=20*sum((d/'initial_access.json').exists() for d,t,p,im in paths if not im),
        imported_queries=1000,imported_initial_accesses=400,GNN_training=0,VAE_updates=0,B_test_access=0)
    write(o/'resource_counts.json',resource)
    status={kind:dict(collections.Counter(r['status'] for r in tables['convergence_summary'] if r['kind']==kind and r['source']=='new')) for kind in ['mapping','gp']}
    write(o/'analysis/main.json',dict(replay=main,offline=tables['offline_main'],new_fitting_status=status,resources=resource,evaluation_seconds=time.perf_counter()-started,
        all_trajectories_complete=all(len(c)==51 for c in curves.values()),trajectories=len(curves)))
    plots(o,tables,tasks);print(json.dumps(dict(replay=main,status=status,resources=resource)),flush=True)


def plots(o,tables,tasks):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=o/'analysis/plots';out.mkdir(parents=True,exist_ok=True)
    # Predeclared pilot curves, all groups; complete machine-readable curves exist for every fit.
    for kind in ['mapping','gp']:
        fig,axes=plt.subplots(3,3,figsize=(15,11))
        for row,g in enumerate(['C','F','W']):
            folder=o/'tasks'/f'offline-f0-s0-{g}'/('mapping' if kind=='mapping' else 'gp-ADAPTIVE');s=read(folder/'summary.json')
            h=s['history'];axes[row,0].plot([r['step'] for r in h],[r['total'] for r in h],label='total')
            if kind=='mapping':
                for k in ['weighted_regression','weighted_neighborhood','weighted_scale']:axes[row,0].plot([r['step'] for r in h],[r[k] for r in h],label=k)
            checks=s['all_checks'];axes[row,1].semilogy([r['step'] for r in checks],[r['gradient']['rms'] for r in checks],label='gradient RMS')
            ds=[d for d in s['decisions'] if d['valid']]
            for k in (['distance_l2','distance_pair'] if kind=='mapping' else ['prediction_range','std_range']):
                axes[row,2].plot([d['step'] for d in ds],[d['measures'][k] for d in ds],label=k)
            for ax in axes[row]:
                ax.axvline(100 if kind=='mapping' else 20,color='gray',linestyle='--',label='FIXED prefix');ax.axvline(s['completed_updates'],color='black',linestyle=':',label='ADAPTIVE stop/cap')
                ax.set_xlabel('optimizer updates');ax.set_title(g+' / '+s['status']);ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(out/(kind+'_convergence.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for i,g in enumerate(['C','F','W']):
        for p in ['FIXED','ADAPTIVE','RANDOM']:
            rows=[r for r in tables['replay_summary'] if r['protocol']==p and r['group']==('RANDOM' if p=='RANDOM' else g)]
            qs=sorted(set(r['queries'] for r in rows));means=[avg([r['best_validation']*100 for r in rows if r['queries']==q]) for q in qs]
            axes[i].plot(qs,means,label=p,marker='o',markersize=3)
        axes[i].set_title(g);axes[i].set_xlabel('new historical queries');axes[i].set_ylabel('best validation (%)');axes[i].legend()
    fig.tight_layout();fig.savefig(out/'replay_best.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,9))
    for g in ['C','F','W']:
        s=read(o/'tasks'/f'offline-f0-s0-{g}'/'mapping/summary.json');checks=s['all_checks']
        for ax,key in zip(axes,['distance_median','head_weight_norm','embedding_norm_mean']):ax.plot([r['step'] for r in checks],[r['extra'][key] for r in checks],label=g);ax.set_ylabel(key);ax.axvline(100,color='gray',linestyle='--');ax.axvline(s['completed_updates'],color='black',linestyle=':');ax.legend()
    axes[-1].set_xlabel('mapping updates');fig.tight_layout();fig.savefig(out/'scale_and_head.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);evaluate(Path(sys.argv[1]).resolve())
