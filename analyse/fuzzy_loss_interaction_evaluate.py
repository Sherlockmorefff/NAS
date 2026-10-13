"""Separate evaluation of already fixed predictions, with no optimizer steps."""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
from analyse.fuzzy_loss_interaction_core import *
from analyse.fuzzy_loss_interaction_diagnostics import flat,cosine
from analyse.fuzzy_loss_interaction_run import verify_protocol,validate_done

TOL=1e-12


def mean(xs):return float(statistics.mean(xs))
def median(xs):return float(statistics.median(xs))
def distribution(xs):
    xs=list(xs)
    return dict(count=len(xs),mean=mean(xs),median=median(xs),minimum=min(xs),maximum=max(xs),
                **{f'q{q}':float(np.quantile(xs,q/100)) for q in [10,25,75,90,95]}) if xs else dict(count=0)


def metadata(task):return {k:task[k] for k in ['name','stage','group','model_seed','fold','search_seed'] if k in task}


def score_check(gp_path,rows,query_ids,model_path=None,features=None):
    gp=load(gp_path);assert gp['completed_updates']==20
    assert tree_hash(gp['model'])==gp['model_hash'] and tree_hash(gp['optimizer'])==gp['optimizer_hash']
    assert not gp['x'].requires_grad and torch.equal(gp['fixed_noise_normalized'],(gp['se']/gp['scale']).square())
    if model_path is not None:
        s=load(model_path);assert s['model_hash']==gp['cache']['model_hash']
        # Re-encode every known/unknown ID in this same saved coordinate version.
        scale=s['scaler'];z=(torch.tensor(features['z'])-torch.tensor(scale['z_mean']))/torch.tensor(scale['z_scale'])
        hp=(torch.tensor(features['hp'])-torch.tensor(scale['hp_mean']))/torch.tensor(scale['hp_scale'])
        with torch.no_grad():e,h=reload_model(s['model'])(z,hp)
        assert torch.equal(e.double(),gp['cache']['embeddings'])
        assert torch.equal(h.double()*scale['y_scale']+scale['y_mean'],gp['cache']['head'])
    rebuilt=predict(gp['cache'],gp['train_ids'],query_ids,gp)
    actual={r['candidate_id']:r for r in rows};errors=[]
    for row in rebuilt:
        old=actual[row['candidate_id']]
        errors.append(abs(row['gp_mean']-old['gp_mean']))
        assert errors[-1]<=1e-8 and abs(row['head_mean']-old['head_mean'])<=1e-12
        assert row['neighbor_ids']==old['neighbor_ids']
    return dict(path=str(gp_path),query_count=len(rows),maximum_mean_absolute_difference=max(errors,default=0.),
                gp_fit_steps=0,model_hash=gp['cache']['model_hash'],gp_hash=gp['model_hash'])


def gate(o):
    started=time.perf_counter();p=verify_protocol(o);tasks=[t for t in read(o/'tasks.json') if t['stage']=='offline'];features=read(o/'features.json')
    checks=[];paired=defaultdict(list)
    for task in tasks:
        validate_done(o,task,p);d=o/'tasks'/task['name'];pred=read(d/'predictions.json')
        assert set(pred['query_ids'])==set(task['query_ids']) and set(pred['train_ids'])==set(task['train_ids'])
        assert len(read(d/'access_boundary.json')['denied_attempts'])>=4
        initial=load(d/'mapping/post000.pt');weights=load(o/task['weight_file']);assert initial['model_hash']==weights['model_hash']
        assert initial['optimizer']['state']=={} and weights['untrained']
        for step in [0,10,50,99,100]:
            s=load(d/'mapping'/f'post{step:03}.pt');assert s['completed_updates']==step and s['model_hash']==tree_hash(s['model'])
            assert s['optimizer_hash']==tree_hash(s['optimizer']) and len(s['history'])==step
        gp=load(d/'gp/latest.pt');g0=load(d/'gp/initial.pt')
        checks.append(score_check(d/'gp/latest.pt',pred['rows'],task['query_ids'],d/'mapping/post100.pt',features))
        paired[(task['fold'],task['model_seed'])].append((initial['model_hash'],initial['optimizer_hash'],tree_hash(initial['rng']),digest(initial['scaler']),g0['initial_parameters_hash']))
    assert all(len(x)==3 and len(set(x))==1 for x in paired.values())
    result=dict(passed=True,tasks=45,score_checks=checks,matched_initial_model_optimizer_scaler_GP=True,
                score_checks_optimization_updates=0,engineering_only_gate=True,MAE_or_neighbor_gain_required=False,seconds=time.perf_counter()-started,time=utc())
    write(o/'offline_engineering_gate.json',result)
    return result


def offline(o,tasks,labels):
    predictions=[];summaries=[];gradient=[];probe=[];sensitive=[];target=[];share=[]
    for task in tasks:
        d=o/'tasks'/task['name'];meta=metadata(task);raw=read(d/'predictions.json');ys={c:labels[c]['validation_mean'] for c in task['query_ids']}
        best=max(ys.values());champ={c for c,v in ys.items() if best-v<=TOL}
        rows=[]
        for row in raw['rows']:
            c=row['candidate_id'];truth=ys[c];nbr=[abs(truth-labels[n]['validation_mean']) for n in row['neighbor_ids']]
            rows.append(dict(meta,**row,true_validation=truth,head_absolute_error=abs(row['head_mean']-truth),gp_absolute_error=abs(row['gp_mean']-truth),
                             neighbor_absolute_differences=nbr,neighbor_mean_absolute_difference=mean(nbr)))
        predictions.extend(rows)
        for score in ['gp_mean','head_mean']:
            ranked=sorted(rows,key=lambda r:(-r[score],r['candidate_id']));champ_rank=min(i+1 for i,r in enumerate(ranked) if r['candidate_id'] in champ)
            err=[abs(r[score]-r['true_validation']) for r in rows]
            for k in [5,10,20]:
                found=max(r['true_validation'] for r in ranked[:k])
                summaries.append(dict(meta,score=score,k=k,mae_raw=mean(err),mae_percentage_points=100*mean(err),top_k_best=found,
                                      holdout_best=best,gap_raw=best-found,gap_percentage_points=100*(best-found),hit_champion=champ_rank<=k,champion_rank=champ_rank,
                                      neighbor_mean_raw=mean([r['neighbor_mean_absolute_difference'] for r in rows]),
                                      neighbor_median_raw=median([r['neighbor_mean_absolute_difference'] for r in rows])))
        for step in [0,10,50,99]:
            diag=d/'diagnostics'/f'post{step:03}';g=load(diag/'gradients.pt')
            for row in read(diag/'rows.json'):
                r=flat(g['vectors'],'regression',row['block']);f=flat(g['vectors'],'neighborhood',row['block'])*g['weights']['fuzzy'];e=flat(g['vectors'],'scale',row['block'])*.001
                for key,val in [('regression_norm',float(r.norm())),('fuzzy_weighted_norm',float(f.norm())),('norm_weighted_norm',float(e.norm())),('total_norm',float((r+f+e).norm()))]:
                    assert abs(row[key]-val)<1e-12
                assert row['cosine_rf']==cosine(r,f)
                gradient.append(row)
            if task['group']!='C':continue
            pd=d/'probes'/f'post{step:03}';before=read(pd/'before_predictions.json');bq={r['candidate_id']:r for r in before['query']}
            for branch in ['C','F','W']:
                res=read(pd/branch/'result.json');after=read(pd/branch/'after_predictions.json');aq={r['candidate_id']:r for r in after['query']}
                orig=load(d/'mapping'/f'post{step:03}.pt');post=load(pd/branch/'post_one_step.pt')
                assert post['source_optimizer_hash']==orig['optimizer_hash']
                for n,delta in post['actual_parameter_delta'].items():assert torch.equal(delta,(post['model'][n]-orig['model'][n]).reshape(-1).double())
                row=dict(meta,branch=branch,completed_updates=step,lambda_f=res['lambda_f'],source_model_hash=res['source_model_hash'],
                         **{k:v for k,v in res.items() if k.endswith('_actual_delta_norm')},
                         **{'post_'+k:v for k,v in res['post_losses'].items()})
                for name,b,a in [('train',before['train'],after['train']),('query',{c:r['head_mean'] for c,r in bq.items()},{c:r['head_mean'] for c,r in aq.items()})]:
                    ids=sorted(a);be=mean([abs(b[c]-labels[c]['validation_mean']) for c in ids]);ae=mean([abs(a[c]-labels[c]['validation_mean']) for c in ids])
                    row.update({name+'_mae_before':be,name+'_mae_after':ae,name+'_mae_change':ae-be,
                                name+'_prediction_absolute_change_mean':mean([abs(a[c]-b[c]) for c in ids])})
                def nd(r):return mean([abs(labels[r['candidate_id']]['validation_mean']-labels[n]['validation_mean']) for n in r['neighbor_ids']])
                row.update(query_neighbor_difference_before=mean([nd(r) for r in bq.values()]),query_neighbor_difference_after=mean([nd(r) for r in aq.values()]),
                           query_neighbor_replacement_rate=mean([1-len(set(bq[c]['neighbor_ids'])&set(aq[c]['neighbor_ids']))/5 for c in bq]))
                probe.append(row)
        if task['group']=='C':
            sensitive.extend(read(d/'sensitivity/rows.json'));target.extend(read(d/'sensitivity/targets.json'));share.extend(read(d/'sensitivity/loss_shares.json'))
    main=[]
    for group in ['C','F','W']:
        rs=[r for r in predictions if r['group']==group]
        medians=[median([r['neighbor_mean_absolute_difference'] for r in rs if r['model_seed']==s]) for s in [0,1,2]]
        main.append(dict(group=group,head_mae_raw=mean([r['head_absolute_error'] for r in rs]),gp_mae_raw=mean([r['gp_absolute_error'] for r in rs]),
                         neighbor_seed_medians=medians,primary_neighbor_raw=mean(medians)))
    pairs=[]
    for a,b in [('F','C'),('W','C'),('W','F')]:
        for fold in range(5):
            aa=[r for r in predictions if r['group']==a and r['fold']==fold];bb=[r for r in predictions if r['group']==b and r['fold']==fold]
            pairs.append(dict(comparison=a+'-'+b,fold=fold,head_mae_difference=mean([r['head_absolute_error'] for r in aa])-mean([r['head_absolute_error'] for r in bb]),
                              gp_mae_difference=mean([r['gp_absolute_error'] for r in aa])-mean([r['gp_absolute_error'] for r in bb]),
                              neighbor_difference=mean([r['neighbor_mean_absolute_difference'] for r in aa])-mean([r['neighbor_mean_absolute_difference'] for r in bb])))
    write(o/'analysis/offline_main.json',main);write(o/'analysis/offline_paired.json',pairs)
    return dict(offline_predictions=predictions,offline_summary=summaries,gradient_diagnostics=gradient,one_step_probes=probe,
                label_sensitivity=sensitive,label_target_changes=target,candidate_loss_shares=share)


def replay(o,tasks,labels,features):
    curves=[];queries=[];predictions=[];errorgroups=[];scorechecks=[];rankrows=[];initial_counts=0;shares=[];sens=[];targets=[]
    best=max(r['validation_mean'] for r in labels.values());champ={c for c,r in labels.items() if best-r['validation_mean']<=TOL}
    values={c:r['validation_mean'] for c,r in labels.items()};all_ids=set(values)
    for task in tasks:
        d=o/'tasks'/task['name'];meta=metadata(task);known=set(task['initial_ids']);initial_counts+=20
        initial_champ=bool(known&champ);hit=0 if initial_champ else None
        incumbent=max(values[c] for c in known)
        curves.append(dict(meta,queries=0,best_validation=incumbent,gap_raw=best-incumbent,champion_hit=initial_champ,first_hit_query=hit,initial_champion=initial_champ))
        state=load(d/'state.pt');assert len(state['query_order'])==len(set(state['query_order']))==50
        for batch in range(10):
            bd=d/'batches'/f'b{batch:02}';s=read(bd/'selection.json');receipt=read(bd/'reveal.json');q=batch*5
            assert set(s['known_ids'])==known and s['known_labels_hash']==digest({c:labels[c] for c in known})
            assert receipt['selection_sha']==file_hash(bd/'selection.json') and receipt['ids']==s['selected_ids'] and receipt['labels']=={c:labels[c] for c in receipt['ids']}
            assert s['saved_at']<=receipt['revealed_at'] and len(receipt['ids'])==len(set(receipt['ids']))==5 and not known&set(receipt['ids'])
            assert receipt['query_count_after']==q+5
            unknown=all_ids-known;unknown_best=max(values[c] for c in unknown);unknown_champ={c for c in unknown if unknown_best-values[c]<=TOL}
            if task['group']!='RANDOM':
                assert set(r['candidate_id'] for r in s['rows'])==unknown
                ranked=sorted(s['rows'],key=lambda r:(-r['gp_mean'],r['candidate_id']))
                assert [r['candidate_id'] for r in ranked[:5]]==s['selected_ids']
                assert s['representation_version']==q//10
                model=d/'mapping'/f'known{(q//10)*10:02}'/'post100.pt'
                scorechecks.append(score_check(bd/'gp/latest.pt',s['rows'],sorted(unknown),model,features))
                high=set(sorted(unknown,key=lambda c:(-values[c],c))[:math.ceil(.1*len(unknown))])
                rows=[dict(meta,batch=batch+1,query_count_before=q,rank=i+1,**r,true_validation=values[r['candidate_id']],signed_error=r['gp_mean']-values[r['candidate_id']],
                           selected=r['candidate_id'] in s['selected_ids'],true_top_decile=r['candidate_id'] in high,
                           prediction_bin_lower=math.floor(r['gp_mean']/.05)*.05) for i,r in enumerate(ranked)]
                predictions.extend(rows)
                subsets={'selected':[r for r in rows if r['selected']],'all_unknown':rows,'true_top_decile':[r for r in rows if r['true_top_decile']]}
                for key,sub in subsets.items():
                    errorgroups.append(dict(meta,batch=batch+1,query_count_before=q,subgroup=key,prediction_bin_lower=None,**distribution([r['signed_error'] for r in sub])))
                for binvalue in sorted({r['prediction_bin_lower'] for r in rows}):
                    for key,sub in subsets.items():
                        xs=[r['signed_error'] for r in sub if r['prediction_bin_lower']==binvalue]
                        if xs:errorgroups.append(dict(meta,batch=batch+1,query_count_before=q,subgroup=key,prediction_bin_lower=binvalue,**distribution(xs)))
                globalranks=[i+1 for i,r in enumerate(ranked) if r['candidate_id'] in champ]
                rankrows.append(dict(meta,batch=batch+1,query_count_before=q,unknown_best=unknown_best,
                                     unknown_champion_rank=min(i+1 for i,r in enumerate(ranked) if r['candidate_id'] in unknown_champ),
                                     pool_champion_rank=min(globalranks) if globalranks else None,pool_champion_already_known=not globalranks,
                                     selected_unknown_champion=bool(unknown_champ&set(s['selected_ids'])),selected_pool_champion=bool(champ&set(s['selected_ids']))))
                byid={r['candidate_id']:r for r in ranked}
            else:
                assert s['rows']==[] and s['selected_ids']==read(o/'starts.json')[str(task['search_seed'])]['random_order'][q:q+5]
                byid={}
            batch_incumbent=incumbent
            for position,c in enumerate(receipt['ids'],1):
                pred=byid[c]['gp_mean'] if c in byid else None;error=None if pred is None else pred-values[c]
                if c in champ and hit is None:hit=q+position
                queries.append(dict(meta,batch=batch+1,query_position=q+position,candidate_id=c,preselection_gp_mean=pred,true_validation=values[c],signed_error=error,
                                    high_overestimate=None if error is None else error>=.05,does_not_improve_prebatch_best=values[c]<=batch_incumbent+TOL,
                                    incumbent_before_batch=batch_incumbent,selection_sha=file_hash(bd/'selection.json'),receipt_sha=file_hash(bd/'reveal.json')))
                known.add(c);incumbent=max(incumbent,values[c])
            curves.append(dict(meta,queries=q+5,best_validation=incumbent,gap_raw=best-incumbent,champion_hit=bool(known&champ),
                               first_hit_query=hit,first_hit_batch=None if hit is None else math.ceil(hit/5),initial_champion=initial_champ))
        assert set(state['known'])==known and state['query_order']==[r['candidate_id'] for r in queries if r['name']==task['name']]
        if task['group']=='C':
            sens.extend(read(d/'sensitivity/rows.json'));targets.extend(read(d/'sensitivity/targets.json'));shares.extend(read(d/'sensitivity/loss_shares.json'))
    pair=[];pairagg=[]
    for a,b in [('F','C'),('W','C'),('W','F'),('C','RANDOM'),('F','RANDOM'),('W','RANDOM')]:
        for q in [10,20,50]:
            vals=[]
            for seed in range(2026091900,2026091905):
                av=next(r['best_validation'] for r in curves if r['group']==a and r['search_seed']==seed and r['queries']==q)
                bv=next(r['best_validation'] for r in curves if r['group']==b and r['search_seed']==seed and r['queries']==q)
                delta=av-bv;vals.append(delta);pair.append(dict(comparison=a+'-'+b,search_seed=seed,queries=q,difference_raw=delta,difference_percentage_points=100*delta,
                                                            outcome='win' if delta>TOL else 'loss' if delta<-TOL else 'tie'))
            pairagg.append(dict(comparison=a+'-'+b,queries=q,mean_difference_raw=mean(vals),mean_difference_percentage_points=100*mean(vals),
                                wins=sum(v>TOL for v in vals),ties=sum(abs(v)<=TOL for v in vals),losses=sum(v<-TOL for v in vals)))
    speed=[]
    for group in ['F','W']:
        for seed in range(2026091900,2026091905):
            aa=[r for r in curves if r['group']==group and r['search_seed']==seed];cc=[r for r in curves if r['group']=='C' and r['search_seed']==seed]
            final=aa[-1]['best_validation'];same=abs(final-cc[-1]['best_validation'])<=TOL
            aq=min(r['queries'] for r in aa if final-r['best_validation']<=TOL);cq=min(r['queries'] for r in cc if cc[-1]['best_validation']-r['best_validation']<=TOL)
            speed.append(dict(group=group,search_seed=seed,same_final=same,first_batch_budget_to_own_final=aq,C_first_batch_budget_to_own_final=cq,
                              same_final_speed_gain=same and aq<cq))
    end={r['comparison']:r for r in pairagg if r['queries']==50}
    wc,wf,fc=end['W-C'],end['W-F'],end['F-C'];candidate=None
    if wc['mean_difference_raw']>TOL and wc['wins']>wc['losses'] and wf['mean_difference_raw']>=-TOL:candidate='W'
    elif fc['mean_difference_raw']>TOL and fc['wins']>fc['losses'] and wf['mean_difference_raw']<=TOL:candidate='F'
    local=wc['wins']>0 or fc['wins']>0 or any(r['same_final_speed_gain'] for r in speed)
    status='CANDIDATE_FOR_LARGER_DEVELOPMENT_REPLAY' if candidate else 'MIXED_DEVELOPMENT_EVIDENCE' if local else 'NO_SELECTION_GAIN_IN_THIS_PILOT'
    outcomes=[]
    for group in ['C','F','W','RANDOM']:
        rows=[r for r in curves if r['group']==group and r['queries']==50]
        outcomes.append(dict(group=group,final_mean=mean([r['best_validation'] for r in rows]),final_gap_mean=mean([r['gap_raw'] for r in rows]),
                             champion_hits=sum(r['champion_hit'] for r in rows),initial_hits=sum(r['initial_champion'] for r in rows),
                             new_discovery_opportunity_starts=sum(not r['initial_champion'] for r in rows),
                             first_hit_positions=[r['first_hit_query'] if r['first_hit_query'] is not None else '>50' for r in rows]))
    write(o/'analysis/replay_main.json',dict(status=status,candidate=candidate,groups=outcomes,paired=pairagg,speed=speed,initial_access_count=initial_counts,
                                           logical_queries=len(queries),all_A_development_only=True))
    write(o/'analysis/replay_score_recomputation.json',dict(checks=scorechecks,GP_fit_steps=0,passed=True))
    return dict(replay_summary=curves,replay_queries=queries,replay_predictions=predictions,replay_error_groups=errorgroups,replay_champion_ranks=rankrows,
                replay_paired=pair,label_sensitivity=sens,label_target_changes=targets,candidate_loss_shares=shares)


def evaluate(o):
    start=time.perf_counter();p=verify_protocol(o);assert read(o/'offline_engineering_gate.json')['passed']
    tasks=read(o/'tasks.json');features=read(o/'features.json')
    for t in tasks:validate_done(o,t,p)
    # Full truth is first loaded in this evaluation process after all predictions were persisted.
    labels=read(o/'private/evaluation_A.json');assert file_hash(o/'private/evaluation_A.json')==p['private_A_sha']
    table=offline(o,[t for t in tasks if t['stage']=='offline'],labels)
    second=replay(o,[t for t in tasks if t['stage']=='replay'],labels,features)
    for k,v in second.items():table.setdefault(k,[]).extend(v)
    assert len(table['offline_predictions'])==1611 and len(table['gradient_diagnostics'])==720 and len(table['one_step_probes'])==180
    assert len(table['label_sensitivity'])==60 and len(table['replay_queries'])==1000
    for key,rows in table.items():write(o/'analysis/tables'/f'{key}.json',rows)
    ledger={k:dict(intents=len(journal(o/'ledger'/f'{k}_intents.jsonl')),receipts=len(journal(o/'ledger'/f'{k}_receipts.jsonl'))) for k in ['mapping','gp']}
    assert ledger['mapping']==dict(intents=12180,receipts=12180) and ledger['gp']==dict(intents=3900,receipts=3900)
    assert len(journal(o/'ledger/gp_fits.jsonl'))==195 and len(journal(o/'oracle_access.jsonl'))==200
    assert sum(r['count'] for r in journal(o/'oracle_access.jsonl'))==1000
    budget=dict(mapping=ledger['mapping'],GP=ledger['gp'],GP_fits=195,logical_queries=1000,initial_accesses=400,GNN_updates=0,VAE_updates=0,B_test_reads=0,
                synthetic=read(o/'engineering_checks.json')['synthetic_counts'],attempts=len(journal(o/'attempts.jsonl')),
                failures_and_recovery=journal(o/'failures_and_recovery.jsonl'))
    write(o/'resource_counts.json',budget)
    for n,h in read(o/'protected_before.json').items():assert file_hash(ROOT/n)==h,n
    write(o/'protected_after.json',dict(passed=True,checked=read(o/'protected_before.json'),time=utc()))
    timings=[read(f) for t in tasks for f in (o/'tasks'/t['name']).glob('worker_runtime_*.json')]
    write(o/'analysis/timing.json',dict(tasks=timings,evaluation_seconds=time.perf_counter()-start,
          total_diagnostic_backward_calls=sum(t['diagnostic_backward_calls'] for t in timings),
          total_diagnostic_backward_seconds=sum(t['diagnostic_backward_seconds'] for t in timings),
          total_worker_wall_seconds=sum(r['wall_seconds'] for r in journal(o/'task_execution.jsonl'))))
    write(o/'analysis/evaluation_checks.json',dict(passed=True,table_rows={k:len(v) for k,v in table.items()},
          source_of_summary='saved predictions and oracle receipts; no fitting',checkpoint_coordinate_rebuilds=195,
          duplicate_queries=0,hidden_label_boundary_verified=True,budget_verified=True,time=utc()))
    print(read(o/'analysis/replay_main.json')['status'],flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('command',choices=['gate','all']);a.add_argument('output',type=Path);args=a.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    gate(args.output.resolve()) if args.command=='gate' else evaluate(args.output.resolve())


if __name__=='__main__':main()
