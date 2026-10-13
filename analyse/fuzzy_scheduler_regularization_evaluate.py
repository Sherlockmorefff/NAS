"""Post-seal summaries only; no training, no feedback into task selection."""
import sys,statistics,itertools,collections,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from analyse.fuzzy_scheduler_regularization_core import *
from analyse.fuzzy_convergence_recompute import gp_values,map_values

def avg(x):return float(statistics.mean(x))
def median(x):return float(statistics.median(x))
def resolve(o,t):
    folder=o/'tasks'/t['name'];alias=read(folder/'logical_alias.json') if (folder/'logical_alias.json').exists() else None
    return (o/'tasks'/alias['target'] if alias else folder),alias


def evaluate(o):
    begun=time.perf_counter();assert (o/'execution_complete.json').exists();tasks=read(o/'tasks.json');features=read(o/'features.json');labels=read(o/'private/evaluation_A.json');truth={c:r['validation_mean'] for c,r in labels.items()}
    tables={n:[] for n in ['scheduler_repair_cases','scheduler_timeline','inner_validation_scores','selected_regularization','convergence_summary','offline_comparison','offline_predictions','offline_main','offline_paired','replay_summary','replay_paired','replay_queries','replay_main','heldout_error_curves','parameter_geometry_curves']}
    for t in tasks:
        folder=o/'tasks'/t['name']
        if t['stage']=='engineering':tables['scheduler_repair_cases'].append(dict(task=t['name'],**read(folder/'fit/comparison.json')))
        if t['stage']=='inner_cv' and (o/'inner_scores'/(t['name']+'.json')).exists():tables['inner_validation_scores'].append(read(o/'inner_scores'/(t['name']+'.json')))
        for f in folder.rglob('summary.json'):
            st=read(f);common=dict(task=t['name'],purpose=t['purpose'],group=t['group'],kind=st['kind'],path=str(f.relative_to(o)),lambda_w=st.get('lambda_w',st['metadata'].get('lambda_w',0.)))
            tables['convergence_summary'].append(dict(common,status=st['status'],steps=st['completed_updates'],reason_flags=st.get('reason_flags'),failed=st['decisions'][-1]['failed'],lr_use_counts=st['learning_rate_use_counts'],lr_configured_next=st['final_learning_rate'],optimization_seconds=st['optimization_seconds'],fit_seconds=st['fit_seconds'],final_loss=st['history'][-1],final_decision=st['decisions'][-1],metadata=st['metadata']))
            for event in st['scheduler']['events']:tables['scheduler_timeline'].append(dict(common,**event))
            state=load(f.parent/'latest.pt')
            for shot in state['audit_snapshots']:
                if state['kind']=='mapping':
                    squared={name:float(v.double().square().sum()) for name,v in shot['model'].items()}
                    tables['parameter_geometry_curves'].append(dict(common,step=shot['step'],parameter_squared_sum_double=sum(squared.values()),head_parameter_norm=(sum(v for name,v in squared.items() if name.startswith('regressor.')))**.5,head_weight_norm=shot['extra']['head_weight_norm'],embedding_norm_mean=shot['extra']['embedding_norm_mean'],distance_median=shot['extra']['distance_median'],coordinate_std=shot['extra']['coordinate_std']))
                if t['stage']=='offline':
                    ids=t['query_ids'];idx=[features['ids'].index(c) for c in ids]
                    if state['kind']=='mapping':
                        _,predicted=map_values(shot['model'],features,state['scaler']);values=[float(predicted[i]) for i in idx];score='head_mean'
                    else:values=gp_values(dict(state,model=shot['model']),ids).tolist();score='gp_mean'
                    tables['heldout_error_curves'].append(dict(common,step=shot['step'],score=score,heldout_mae_raw=avg([abs(v-truth[c]) for c,v in zip(ids,values)]),used_for_training_or_stopping=False,query_count=len(ids)))

    for path in sorted((o/'selections').glob('*.json')):tables['selected_regularization'].append(read(path))
    # Both logical flows remain present when selected-zero results are aliases.
    offline=[(t,*resolve(o,t),'REPAIRED' if t['protocol_type']=='REPAIRED' else 'REPAIRED_SELECT_REG') for t in tasks if t['stage']=='offline']
    offline += [(r['task'],o/'imports'/r['task']['name'],None,'OLD_ADAPTIVE') for r in read(o/'reuse_manifest.json') if r['task']['stage']=='offline']
    for t,folder,alias,flow in offline:
        predfile=folder/('predictions-ADAPTIVE-ADAPTIVE.json' if flow=='OLD_ADAPTIVE' else 'predictions.json')
        if not predfile.exists():continue
        pred=read(predfile);rows=pred['rows'];ms=load(folder/'mapping/latest.pt');gs=load(folder/('gp-ADAPTIVE/latest.pt' if flow=='OLD_ADAPTIVE' else 'gp/latest.pt'))
        w=pred.get('lambda_w',0.);common=dict(task=t['name'],group=t['group'],flow=flow,fold=t['fold'],model_seed=t['model_seed'],lambda_w=w,logical_alias=alias['target'] if alias else None,mapping_status=ms['status'],gp_status=gs['status'],mapping_steps=ms['completed_updates'],gp_steps=gs['completed_updates'])
        train_rows=pred.get('train_rows')
        if train_rows is None:
            emb,head=map_values(ms['model'],features,ms['scaler']);mu=gp_values(gs,gs['train_ids']);pos={c:i for i,c in enumerate(features['ids'])}
            train_rows=[dict(candidate_id=c,head_mean=float(head[pos[c]]),gp_mean=float(v)) for c,v in zip(gs['train_ids'],mu)]
        for row in rows:tables['offline_predictions'].append(dict(common,**row,true_validation=truth[row['candidate_id']],neighbor_difference=avg([abs(truth[row['candidate_id']]-truth[n]) for n in row['neighbor_ids']])))
        for score in ['head_mean','gp_mean']:
            ranked=sorted(rows,key=lambda r:(-r[score],r['candidate_id']));best=max(truth[r['candidate_id']] for r in rows);rank=min(i+1 for i,r in enumerate(ranked) if best-truth[r['candidate_id']]<=1e-12)
            row=dict(common,score=score,train_mae_raw=avg([abs(r[score]-truth[r['candidate_id']]) for r in train_rows]),holdout_mae_raw=avg([abs(r[score]-truth[r['candidate_id']]) for r in rows]),champion_rank=rank,query_count=len(rows))
            for k in [5,10,20]:row.update({f'top{k}_best_gap_raw':best-max(truth[r['candidate_id']] for r in ranked[:k]),f'top{k}_champion_hit':rank<=k})
            tables['offline_comparison'].append(row)
    for flow,group in itertools.product(['OLD_ADAPTIVE','REPAIRED','REPAIRED_SELECT_REG'],['C','F']):
        rows=[r for r in tables['offline_predictions'] if r['flow']==flow and r['group']==group]
        if len(rows)!=179*3:continue
        comps=[r for r in tables['offline_comparison'] if r['flow']==flow and r['group']==group]
        tables['offline_main'].append(dict(flow=flow,group=group,rows=len(rows),holdout_head_mae_raw=avg([abs(r['head_mean']-r['true_validation']) for r in rows]),holdout_gp_mae_raw=avg([abs(r['gp_mean']-r['true_validation']) for r in rows]),train_head_mae_raw=avg([r['train_mae_raw'] for r in comps if r['score']=='head_mean']),train_gp_mae_raw=avg([r['train_mae_raw'] for r in comps if r['score']=='gp_mean']),primary_neighbor_raw=avg([median([r['neighbor_difference'] for r in rows if r['model_seed']==seed]) for seed in range(3)]),gp_top5_gap_raw=avg([r['top5_best_gap_raw'] for r in comps if r['score']=='gp_mean']),gp_top5_hits=sum(r['top5_champion_hit'] for r in comps if r['score']=='gp_mean')))
    comparisons=[(('REPAIRED_SELECT_REG',g),('REPAIRED',g)) for g in ['C','F']]+[(('REPAIRED',g),('OLD_ADAPTIVE',g)) for g in ['C','F']]+[((f,'F'),(f,'C')) for f in ['REPAIRED','REPAIRED_SELECT_REG']]
    for a,b in comparisons:
        for fold,seed,score in itertools.product(range(5),range(3),['head_mean','gp_mean']):
            def find(k):return next((r for r in tables['offline_comparison'] if (r['flow'],r['group'],r['fold'],r['model_seed'],r['score'])==(*k,fold,seed,score)),None)
            x,y=find(a),find(b);complete=x is not None and y is not None
            tables['offline_paired'].append(dict(comparison='/'.join(a)+' - '+'/'.join(b),fold=fold,model_seed=seed,score=score,complete=complete,mae_difference_raw=x['holdout_mae_raw']-y['holdout_mae_raw'] if complete else None,top5_gap_difference_raw=x['top5_best_gap_raw']-y['top5_best_gap_raw'] if complete else None))
    paths=[(t,*resolve(o,t),t['protocol_type']) for t in tasks if t['stage']=='replay']+[(r['task'],o/'imports'/r['task']['name'],None,r['protocol']) for r in read(o/'reuse_manifest.json') if r['task']['stage']=='replay']
    champion={c for c in truth if max(truth.values())-truth[c]<=1e-12};curves={}
    for t,folder,alias,flow in paths:
        known=set(t['initial_ids']);best=max(truth[c] for c in known);first=0 if known&champion else None;curve={0:best}
        w=read(o/'tasks'/t['name']/'assignment.json')['lambda_w'] if flow in ['REPAIRED','REPAIRED_SELECT_REG'] else 0.
        common=dict(task=t['name'],group=t['group'],flow=flow,search_seed=t['search_seed'],lambda_w=w,logical_alias=alias['target'] if alias else None,imported=flow in ['OLD_ADAPTIVE','RANDOM'])
        def summary(q):return dict(common,queries=q,best_validation=best,best_gap=max(truth.values())-best,champion_hit=first is not None,first_champion_query=first if first is not None else '>50',first_champion_batch=(first+4)//5 if first is not None else None)
        tables['replay_summary'].append(summary(0))
        for batch in range(10):
            bd=folder/'batches'/f'b{batch:02}'
            if not (bd/'reveal.json').exists():break
            sel=read(bd/'selection.json');receipt=read(bd/'reveal.json');assert receipt['selection_sha']==file_hash(bd/'selection.json') and sel['known_ids']==sorted(known)
            scores={r['candidate_id']:r for r in sel['rows']}
            for index,c in enumerate(receipt['ids']):
                assert c not in known;known.add(c);q=5*batch+index+1;best=max(best,truth[c]);curve[q]=best
                if first is None and c in champion:first=q
                tables['replay_queries'].append(dict(common,query=q,batch=batch+1,position_in_batch=index+1,candidate_id=c,true_validation=truth[c],best_so_far=best,gp_mean=scores.get(c,{}).get('gp_mean'),head_mean=scores.get(c,{}).get('head_mean'),fallback=sel.get('fallback',False),mapping_status=sel.get('mapping_status','RANDOM'),gp_status=sel.get('gp_status','RANDOM')))
            tables['replay_summary'].append(summary(q))
        curves[(t['search_seed'],flow,t['group'])]=curve
    replay_comp=comparisons+[ ((flow,g),('RANDOM','RANDOM')) for flow in ['OLD_ADAPTIVE','REPAIRED','REPAIRED_SELECT_REG'] for g in ['C','F'] ]
    for a,b in replay_comp:
        for q in [10,20,50]:
            deltas=[]
            for seed in range(2026091900,2026091905):
                x,y=curves[(seed,*a)],curves[(seed,*b)];complete=q in x and q in y;delta=x[q]-y[q] if complete else None
                if complete:deltas.append(delta)
                tables['replay_paired'].append(dict(comparison='/'.join(a)+' - '+'/'.join(b),queries=q,search_seed=seed,complete=complete,difference_raw=delta))
            tables['replay_paired'].append(dict(comparison='/'.join(a)+' - '+'/'.join(b),queries=q,search_seed='all_five',complete=len(deltas)==5,completed_pairs=len(deltas),difference_raw=avg(deltas) if len(deltas)==5 else None,wins=sum(d>1e-12 for d in deltas),ties=sum(abs(d)<=1e-12 for d in deltas),losses=sum(d<-1e-12 for d in deltas)))
    for flow,g in [(f,g) for f in ['OLD_ADAPTIVE','REPAIRED','REPAIRED_SELECT_REG'] for g in ['C','F']]+[('RANDOM','RANDOM')]:
        rows=[r for r in tables['replay_summary'] if (r['flow'],r['group'],r['queries'])==(flow,g,50)]
        tables['replay_main'].append(dict(flow=flow,group=g,completed=len(rows),mean_best_raw=avg([r['best_validation'] for r in rows]) if len(rows)==5 else None,champion_hits=sum(r['champion_hit'] for r in rows),mean_gap_raw=avg([r['best_gap'] for r in rows]) if len(rows)==5 else None))
    for name,rows in tables.items():write(o/'analysis/tables'/(name+'.json'),rows)
    budgets={}
    for kind in ['mapping','gp']:
        ints=journal(o/'ledger'/(kind+'_intents.jsonl'));receipts=journal(o/'ledger'/(kind+'_receipts.jsonl'));assert ints==receipts and len(ints)<=LIMITS[kind]
        budgets[kind]=dict(total=len(ints),by_purpose=dict(collections.Counter(Budget.category(r['task']) for r in ints)))
    timing={};unavailable=[]
    for t in tasks:
        f=o/'tasks'/t['name']/'timing.json'
        if not f.exists():continue
        r=read(f);bucket=timing.setdefault(t['purpose'],collections.defaultdict(float))
        for k,v in r.items():
            if isinstance(v,(int,float)):bucket[k]+=v
        if 'phase_counter_source' not in r:unavailable.append(t['name'])
    counts=dict(actual_updates=budgets,actual_gp_fits=len(journal(o/'ledger/gp_fits.jsonl')),actual_mapping_fits=sum(r['kind']=='mapping' for r in tables['convergence_summary']),actual_new_queries=len(journal(o/'oracle_access.jsonl')),actual_initial_accesses=20*sum((o/'tasks'/t['name']/'initial_access.json').exists() for t in tasks if t['stage']=='replay'),logical_new_queries=sum(len(curves[(t['search_seed'],t['protocol_type'],t['group'])])-1 for t in tasks if t['stage']=='replay'),logical_new_initial=400,imported_queries=750,imported_initial=300,all_logical_queries=sum(len(v)-1 for v in curves.values()),all_logical_initial=700,logical_alias_tasks=sum((o/'tasks'/t['name']/'logical_alias.json').exists() for t in tasks),synthetic=read(o/'synthetic_resume_windows.json')['total_synthetic_updates'],inner_existing_label_partition_accesses=sum(len(t['train_ids'])+len(t['query_ids']) for t in tasks if t['stage']=='inner_cv' and (o/'tasks'/t['name']/'completed.json').exists()),GNN_training=0,VAE_updates=0,new_measurements=0,B_test_reads=0)
    status={kind:dict(collections.Counter(r['status'] for r in tables['convergence_summary'] if r['kind']==kind)) for kind in ['mapping','gp']}
    bypurpose={purpose:{kind:dict(collections.Counter(r['status'] for r in tables['convergence_summary'] if r['kind']==kind and r['purpose']==purpose)) for kind in ['mapping','gp']} for purpose in ['engineering','offline_inner','offline_formal','replay_inner','replay_formal']}
    main=dict(replay=tables['replay_main'],offline=tables['offline_main'],status=status,status_by_purpose=bypurpose,resources=counts,all_35_complete=len(curves)==35 and all(len(v)==51 for v in curves.values()),evaluation_seconds=time.perf_counter()-begun)
    # Attribute measured work and logical standalone work separately; aliases are not new runs.
    cost_by_flow=[]
    for stage in ['offline','replay']:
        for flow in ['REPAIRED','REPAIRED_SELECT_REG']:
            for group in ['C','F']:
                physical=dict(mapping_updates=0,gp_updates=0,task_seconds=0.,fits=0);logical=dict(mapping_updates=0,gp_updates=0,task_seconds_proxy=0.,fits=0);alias_count=0
                own=[t for t in tasks if t['stage']==stage and t['protocol_type']==flow and t['group']==group]
                if flow=='REPAIRED_SELECT_REG':own += [t for t in tasks if t['stage']=='inner_cv' and t['purpose']==stage+'_inner' and t['group']==group]
                for t in own:
                    folder,alias=resolve(o,t);alias_count+=bool(alias)
                    if not (folder/'completed.json').exists():continue
                    seconds=read(folder/'timing.json')['task_seconds'];logical['task_seconds_proxy']+=seconds
                    if not alias:physical['task_seconds']+=seconds
                    for sf in folder.rglob('summary.json'):
                        st=read(sf);key=st['kind']+'_updates';logical[key]+=st['completed_updates'];logical['fits']+=1
                        if not alias:physical[key]+=st['completed_updates'];physical['fits']+=1
                cost_by_flow.append(dict(stage=stage,flow=flow,group=group,includes_inner_selection=flow=='REPAIRED_SELECT_REG',physical_additional_work=physical,logical_standalone_work=logical,alias_tasks=alias_count))
    write(o/'analysis/cost_attribution.json',dict(rows=cost_by_flow,logical_alias_seconds_are_source_runtime_proxy_not_independent_measurement=True,physical_rows_are_disjoint=True,engineering_excluded_and_separately_reported=True))
    write(o/'resource_counts.json',counts);write(o/'analysis/timing.json',dict(by_purpose=timing,missing_detailed_phase_counters=unavailable,controller=read(o/'execution_complete.json'),nested_times_do_not_sum=True));write(o/'analysis/main.json',main)
    plots(o,tables);print(json.dumps(main),flush=True)


def plots(o,tables):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=o/'analysis/plots';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,g in zip(axes,['C','F']):
        for flow in ['OLD_ADAPTIVE','REPAIRED','REPAIRED_SELECT_REG','RANDOM']:
            rows=[r for r in tables['replay_summary'] if r['flow']==flow and r['group']==('RANDOM' if flow=='RANDOM' else g)];qs=sorted({r['queries'] for r in rows})
            ax.plot(qs,[100*avg([r['best_validation'] for r in rows if r['queries']==q]) for q in qs],label=flow,marker='o',markersize=2)
        ax.set(title=g,xlabel='New historical queries',ylabel='Best validation (%)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'replay_best.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for i,case in enumerate(tables['scheduler_repair_cases']):
        ax=axes.flat[i];folder=o/'tasks'/case['task']/'fit';s=read(folder/'summary.json');old=read(o/'engineering_inputs'/case['task'].replace('engineering-','')/'summary.json')
        for name,st in [('old',old),('repaired',s)]:ax.step([h['step'] for h in st['history'][1:]],[h.get('lr_used_for_update',h['lr']) for h in st['history'][1:]],where='post',label=name)
        for event in s['scheduler']['lr_changes']:ax.axvspan(event['after_step'],event['cooldown_until'],alpha=.12,color='green')
        ax.set(title=case['task']+' / '+case['kind'],xlabel='Updates',ylabel='Actual learning rate',yscale='log');ax.legend()
    fig.tight_layout();fig.savefig(out/'scheduler_cases.png',dpi=150);plt.close(fig)
    for g in ['C','F']:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for flow,color in [('REPAIRED','tab:blue'),('REPAIRED_SELECT_REG','tab:orange')]:
            short='repaired' if flow=='REPAIRED' else 'selected'
            task=next(t for t in read(o/'tasks.json') if t['stage']=='offline' and t['fold']==0 and t['model_seed']==0 and t['group']==g and t['protocol_type']==flow);folder,_=resolve(o,task);s=read(folder/'mapping/summary.json');hs=s['history'];ds=[d for d in s['decisions'] if d['valid']]
            for k in ['weighted_regression','weighted_neighborhood','weighted_scale']:axes[0,0].plot([h['step'] for h in hs],[h[k] for h in hs],label=short+'/'+k.replace('weighted_',''))
            for k in ['L0','weighted_parameter','total']:axes[0,1].plot([h['step'] for h in hs],[h[k] for h in hs],label=short+'/'+k.replace('weighted_',''))
            axes[0,2].semilogy([c['step'] for c in s['all_checks']],[c['gradient']['rms'] for c in s['all_checks']],label=flow)
            for k in ['distance_l2','distance_pair']:axes[1,0].plot([d['step'] for d in ds],[d['measures'][k] for d in ds],label=short+'/'+k.replace('weighted_',''))
            for k in ['embedding_norm_mean','head_weight_norm']:axes[1,1].plot([h['step'] for h in hs],[h[k] for h in hs],label=short+'/'+k.replace('weighted_',''))
            gp=read(folder/'gp/summary.json');axes[1,2].plot([d['step'] for d in gp['decisions'] if d['valid']],[d['measures']['prediction_range'] for d in gp['decisions'] if d['valid']],label=flow)
        for ax,title in zip(axes.flat,['Original weighted losses','L0, parameter term, total','Actual gradient RMS','Relative distance changes','Embedding and head norms','GP raw mean range']):ax.set(title=title,xlabel='Updates');ax.legend(fontsize=6)
        fig.suptitle('Fold 0 / seed 0 / '+g);fig.tight_layout();fig.savefig(out/f'loss_geometry_{g}.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);evaluate(Path(sys.argv[1]).resolve())
