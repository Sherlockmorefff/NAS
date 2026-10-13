"""Read-only paired C/G/S evaluation after all S predictions have been sealed."""
from pathlib import Path
import sys,os,math,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyse.relative_metric_pilot import read,write,file_hash,utc
from analyse.gp_mean_guidance_report import spearman
BRANCHES=['C','G','S'];PAIRS=['S-G','S-C']
POINT_METRICS=['GP_MAE_pp','GP_RMSE_pp','head_MAE_pp','head_RMSE_pp','coverage95','width95_pp']
TASK_METRICS=[kind+'_'+metric for kind in ['GP','head'] for metric in ['Spearman','top5_true_mean_pp','top5_true_best_pp','top5_gap_pp','top5_overlap']]

def point_metric(rows,metric,weighting):
    field={'GP_MAE_pp':'GP_error_pp','GP_RMSE_pp':'GP_squared_error_pp','head_MAE_pp':'head_error_pp','head_RMSE_pp':'head_squared_error_pp','coverage95':'covered','width95_pp':'width_pp'}[metric]
    if not rows:return None
    if weighting=='candidate':value=float(np.mean([r[field] for r in rows]))
    else:
        grouped={}
        for row in rows:grouped.setdefault(row['architecture'],[]).append(row[field])
        value=float(np.mean([np.mean(values) for values in grouped.values()]))
    return math.sqrt(value) if 'RMSE' in metric else value

def export(o,name,rows):
    write(o/'evaluation'/f'{name}.json',rows)
    path=o/'evaluation'/f'{name}.csv'
    with path.open('w',newline='') as out:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(rows)

def report(o):
    import torch
    def forbidden(*a,**kw):raise AssertionError('read-only report cannot train')
    torch.optim.Adam.step=forbidden
    p=read(o/'protocol.json');seal=read(o/'all_predictions_sealed.json');assert seal['protocol']==p['identity']
    assert all(file_hash(o/n)==h for n,h in seal['artifacts'].items())
    refs=read(o/'old_reference_manifest.json');assert file_hash(o/'old_reference_manifest.json')==p['old_reference_sha'] and refs['protocol']==p['old_protocol']
    tasks=[]
    for ref in refs['references']:
        task=ref['task'];rel=f"references/{task['task_id']}/predictions_before_labels.json"
        original=next(n for n in ref['files'] if n.endswith('/predictions_before_labels.json'))
        assert file_hash(o/rel)==ref['files'][original]
        tasks.append({**task,'prediction':rel})
    tasks += [{**t,'prediction':f"trajectories/{t['task_id']}/predictions_before_labels.json"} for t in read(o/'task_matrix.json')]
    write(o/'evaluation_task_matrix.json',tasks)
    if not (o/'evaluation_label_access.json').exists():write(o/'evaluation_label_access.json',dict(time=utc(),seal_time=seal['time'],seal_sha=file_hash(o/'all_predictions_sealed.json'),pid=os.getpid()))
    labels={}
    for fold in range(5):
        qp=o/'evaluation_labels'/f'fold{fold}.json';assert file_hash(qp)==p['query_label_hashes'][str(qp.relative_to(o))]
        labels[fold]={part:{c:r['validation_mean'] for c,r in read(path).items()} for part,path in [('train',o/'inputs'/f'fold{fold}'/'train_labels.json'),('query',qp)]}
        write(o/'evaluation_inputs'/f'fold{fold}.json',labels[fold])
    records=[];taskrows=[]
    for t in tasks:
        fold,seed,branch=t['fold'],t['seed'],t['branch'];bundle=read(o/t['prediction'])
        for part in ['train','query']:
            v=bundle[part];truth=labels[fold][part];assert v['ids']==t[part+'_ids'] and set(v['ids'])==set(truth);ys=[truth[c] for c in v['ids']]
            for i,c in enumerate(v['ids']):
                ge=100*(v['GP'][i]-ys[i]);he=100*(v['head'][i]-ys[i]);lo,hi=v['lower'][i],v['upper'][i]
                records.append(dict(fold=fold,seed=seed,branch=branch,partition=part,candidate_id=c,architecture=v['architectures'][i],y=ys[i],GP=v['GP'][i],head=v['head'][i],GP_error_pp=abs(ge),GP_squared_error_pp=ge*ge,head_error_pp=abs(he),head_squared_error_pp=he*he,covered=lo<=ys[i]<=hi,width_pp=100*(hi-lo)))
            best=sorted(v['ids'],key=lambda c:(-truth[c],c))[:5];oracle=float(np.mean([truth[c] for c in best]))
            tr=dict(task=t['task_id'],fold=fold,seed=seed,branch=branch,partition=part,count=len(v['ids']),oracle_top5_ids=best,oracle_top5_mean_pp=100*oracle)
            for kind in ['GP','head']:
                selected=[v['ids'][i] for i in sorted(range(len(v['ids'])),key=lambda i:(-v[kind][i],v['ids'][i]))[:5]];assert selected==v['top5_'+kind]
                actual=float(np.mean([truth[c] for c in selected]));tr[kind+'_top5_ids']=selected
                tr[kind+'_top5_true_mean_pp']=100*actual;tr[kind+'_top5_true_best_pp']=100*max(truth[c] for c in selected)
                tr[kind+'_top5_gap_pp']=100*(oracle-actual);tr[kind+'_top5_overlap']=len(set(selected)&set(best))/5
                tr[kind+'_Spearman']=spearman(v[kind],ys);tr[kind+'_Spearman_NA']=None if tr[kind+'_Spearman'] is not None else 'constant sequence'
            taskrows.append(tr)
    assert len(records)==8055 and sum(r['partition']=='query' for r in records)==1611
    for branch in BRANCHES:
        for seed in range(3):
            ids=[r['candidate_id'] for r in records if r['branch']==branch and r['seed']==seed and r['partition']=='query'];assert len(ids)==len(set(ids))==179 and set(ids)==set(p['A_ids'])
    summaries=[];paired=[];overall=[]
    for scope in ['all']+[f'fold{fold}' for fold in range(5)]:
      for part in ['train','query']:
       for weighting in ['candidate','architecture']:
        for metric in POINT_METRICS:
            values={b:[] for b in BRANCHES+PAIRS}
            for seed in range(3):
                selection={b:{(r['fold'],r['candidate_id']):r for r in records if r['branch']==b and r['seed']==seed and r['partition']==part and (scope=='all' or scope==f"fold{r['fold']}")} for b in BRANCHES}
                common=set.intersection(*(set(v) for v in selection.values()));assert all(set(v)==common for v in selection.values())
                meta=dict(scope=scope,partition=part,weighting=weighting,metric=metric,seed=seed,common_queries=len(common))
                for b in BRANCHES:
                    value=point_metric([selection[b][i] for i in sorted(common)],metric,weighting);values[b].append(value);summaries.append(dict(**meta,branch=b,value=value))
                for pair in PAIRS:
                    base=pair[-1];delta=values['S'][-1]-values[base][-1];values[pair].append(delta);paired.append(dict(**meta,comparison=pair,delta=delta))
            for b,vs in values.items():overall.append(dict(scope=scope,partition=part,weighting=weighting,metric=metric,branch=b,value=float(np.mean(vs)),seed_values=vs))
    selection=[];selection_overall=[]
    for part in ['train','query']:
      for metric in TASK_METRICS:
        fold_values={b:[] for b in BRANCHES+PAIRS}
        for fold in range(5):
            values={b:[] for b in BRANCHES+PAIRS}
            for seed in range(3):
                match={r['branch']:r for r in taskrows if r['partition']==part and r['fold']==fold and r['seed']==seed};assert set(match)==set(BRANCHES)
                v={b:match[b][metric] for b in BRANCHES};valid=all(x is not None for x in v.values())
                for pair in PAIRS:v[pair]=v['S']-v[pair[-1]] if valid else None
                selection.append(dict(partition=part,metric=metric,fold=fold,seed=seed,paired_valid=valid,**v))
                for b in values:values[b].append(v[b] if valid else None)
            for b,vs in values.items():
                good=[v for v in vs if v is not None];mean=float(np.mean(good)) if good else None;fold_values[b].append(mean)
                selection_overall.append(dict(scope=f'fold{fold}',partition=part,metric=metric,branch=b,value=mean,valid_seeds=len(good)))
        for b,vs in fold_values.items():
            good=[v for v in vs if v is not None];selection_overall.append(dict(scope='all',partition=part,metric=metric,branch=b,value=float(np.mean(good)) if good else None,valid_folds=len(good),fold_values=vs))
    for name,rows in [('per_candidate',records),('per_task',taskrows),('summaries',summaries),('paired',paired),('overall',overall),('selection_paired',selection),('selection_overall',selection_overall)]:export(o,name,rows)
    write(o/'evaluation/coverage.json',dict(full=True,S_holdout=537,all_holdout=1611,all_training=6444,tasks=45,shared_candidates=True,primary_aggregation='179 unique OOF candidates per seed then mean3',selection_aggregation='per fold seed then mean seeds then folds',NA_selection=sum(not r['paired_valid'] for r in selection),hidden_query_SE_used=False,optimizer_updates=0))
    for r in overall+selection_overall:
        if r['scope']=='all' and r['partition']=='query' and r.get('weighting','candidate')=='candidate':print(r)
if __name__=='__main__':report(Path(sys.argv[1]).resolve())
