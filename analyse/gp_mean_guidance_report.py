"""Read-only metrics from sealed GP/head predictions, then outer A labels."""
from pathlib import Path
import sys,os,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyse.relative_metric_pilot import read,write,file_hash,utc

POINT_METRICS=['GP_MAE_pp','GP_RMSE_pp','head_MAE_pp','coverage95','width95_pp']
TASK_METRICS=['GP_Spearman','head_Spearman','GP_top5_true_mean_pp','head_top5_true_mean_pp','GP_top5_gap_pp','head_top5_gap_pp']

def ranks(values):
    values=np.asarray(values,dtype=float);order=np.argsort(values,kind='stable');result=np.empty(len(values),float);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:end+=1
        result[order[start:end]]=(start+end-1)/2+1;start=end
    return result

def spearman(a,b):
    ra,rb=ranks(a),ranks(b)
    if np.std(ra)==0 or np.std(rb)==0:return None
    return float(np.corrcoef(ra,rb)[0,1])

def point_metric(rows,key,weighting):
    if not rows:return None
    field={'GP_MAE_pp':'GP_error_pp','GP_RMSE_pp':'GP_squared_error_pp','head_MAE_pp':'head_error_pp','coverage95':'covered','width95_pp':'width_pp'}[key]
    if weighting=='candidate':value=float(np.mean([r[field] for r in rows]))
    else:
        groups={}
        for r in rows:groups.setdefault(r['architecture'],[]).append(r[field])
        value=float(np.mean([np.mean(v) for v in groups.values()]))
    return math.sqrt(value) if key=='GP_RMSE_pp' else value

def report(o):
    protocol=read(o/'protocol.json');tasks=read(o/'task_matrix.json');seal=read(o/'all_predictions_sealed.json')
    assert seal['protocol']==protocol['identity'] and all(file_hash(o/n)==h for n,h in seal['artifacts'].items())
    access=o/'evaluation_label_access.json'
    if not access.exists():write(access,dict(time=utc(),pid=os.getpid(),seal_time=seal['time'],seal_sha=file_hash(o/'all_predictions_sealed.json')))
    labels={}
    for fold in range(5):
        qp=o/'evaluation_labels'/f'fold{fold}.json';assert file_hash(qp)==protocol['query_label_hashes'][str(qp.relative_to(o))]
        labels[fold]={part:{c:r['validation_mean'] for c,r in read(path).items()} for part,path in [('train',o/'inputs'/f'fold{fold}'/'train_labels.json'),('query',qp)]}
        write(o/'evaluation_inputs'/f'fold{fold}.json',labels[fold])
    records=[];taskrows=[]
    for task in tasks:
        fold,seed,branch=task['fold'],task['seed'],task['branch'];folder=o/'trajectories'/task['task_id'];bundle=read(folder/'predictions_before_labels.json')
        for part in ['train','query']:
            data=bundle[part];truth=labels[fold][part];assert set(data['ids'])==set(truth)
            ys=[truth[c] for c in data['ids']]
            for i,c in enumerate(data['ids']):
                ge=100*(data['GP'][i]-ys[i]);he=100*(data['head'][i]-ys[i]);lo,hi=data['lower'][i],data['upper'][i]
                records.append(dict(fold=fold,seed=seed,branch=branch,partition=part,candidate_id=c,architecture=data['architectures'][i],y=ys[i],GP=data['GP'][i],head=data['head'][i],GP_error_pp=abs(ge),GP_squared_error_pp=ge*ge,head_error_pp=abs(he),covered=lo<=ys[i]<=hi,width_pp=100*(hi-lo)))
            best=sorted(data['ids'],key=lambda c:(-truth[c],c))[:5];oracle=float(np.mean([truth[c] for c in best]))
            tr=dict(task=task['task_id'],fold=fold,seed=seed,branch=branch,partition=part,query_count=len(data['ids']),oracle_top5_ids=best,oracle_top5_mean_pp=100*oracle)
            for kind in ['GP','head']:
                expected=[data['ids'][i] for i in sorted(range(len(data['ids'])),key=lambda i:(-data[kind][i],data['ids'][i]))[:5]];assert data['top5_'+kind]==expected
                true_mean=float(np.mean([truth[c] for c in expected]));tr[kind+'_top5_ids']=expected;tr[kind+'_top5_true_mean_pp']=100*true_mean;tr[kind+'_top5_gap_pp']=100*(oracle-true_mean)
                tr[kind+'_Spearman']=spearman(data[kind],ys);tr[kind+'_Spearman_NA']=None if tr[kind+'_Spearman'] is not None else 'constant sequence'
            taskrows.append(tr)
    if seal['full_fivefold']:
        assert sum(r['partition']=='query' for r in records)==1074 and sum(r['partition']=='train' for r in records)==4296
        for seed in range(3):
            for branch in ['C','G']:
                ids=[r['candidate_id'] for r in records if r['partition']=='query' and r['seed']==seed and r['branch']==branch]
                assert len(ids)==len(set(ids))==179 and set(ids)==set(protocol['A_ids'])
    summaries=[];paired=[];overall=[]
    for scope in ['all']+[f'fold{fold}' for fold in range(5)]:
      for part in ['train','query']:
       for weighting in ['candidate','architecture']:
        for metric in POINT_METRICS:
         per={'C':[],'G':[],'G-C':[]}
         for seed in range(3):
            selected={b:{(r['fold'],r['candidate_id']):r for r in records if r['seed']==seed and r['branch']==b and r['partition']==part and (scope=='all' or scope==f"fold{r['fold']}")} for b in ['C','G']}
            common=set(selected['C'])&set(selected['G']);full=set(selected['C'])|set(selected['G'])
            values={b:point_metric([selected[b][i] for i in sorted(common)],metric,weighting) for b in ['C','G']}
            for b in ['C','G']:
                summaries.append(dict(scope=scope,partition=part,weighting=weighting,metric=metric,seed=seed,branch=b,value=values[b],common_queries=len(common),excluded=[list(x) for x in sorted(full-common)]));per[b].append(values[b])
            delta=values['G']-values['C'] if common else None;per['G-C'].append(delta)
            paired.append(dict(scope=scope,partition=part,weighting=weighting,metric=metric,seed=seed,delta=delta,common_queries=len(common),excluded=[list(x) for x in sorted(full-common)]))
         for b,vs in per.items():
            valid=[v for v in vs if v is not None];overall.append(dict(scope=scope,partition=part,weighting=weighting,metric=metric,branch=b,seed_values=vs,value=float(np.mean(valid)) if valid else None,NA_seeds=3-len(valid)))
    selection=[];selection_overall=[]
    for part in ['train','query']:
      for metric in TASK_METRICS:
        fold_values={'C':[],'G':[],'G-C':[]}
        for fold in range(5):
          paired_seeds=[];branch_values={'C':[],'G':[]}
          for seed in range(3):
            match={r['branch']:r for r in taskrows if r['partition']==part and r['fold']==fold and r['seed']==seed}
            valid=set(match)=={'C','G'} and all(match[b][metric] is not None for b in ['C','G'])
            if valid:
                delta=match['G'][metric]-match['C'][metric];paired_seeds.append(delta)
                for b in ['C','G']:branch_values[b].append(match[b][metric])
            else:delta=None
            selection.append(dict(partition=part,metric=metric,fold=fold,seed=seed,C=match.get('C',{}).get(metric),G=match.get('G',{}).get(metric),delta=delta,paired_valid=valid))
          for b in ['C','G','G-C']:
            vals=paired_seeds if b=='G-C' else branch_values[b];mean=float(np.mean(vals)) if vals else None;fold_values[b].append(mean)
            selection_overall.append(dict(scope=f'fold{fold}',partition=part,metric=metric,branch=b,value=mean,valid_seeds=len(vals)))
        for b,vs in fold_values.items():
            valid=[v for v in vs if v is not None];selection_overall.append(dict(scope='all',partition=part,metric=metric,branch=b,value=float(np.mean(valid)) if valid else None,valid_folds=len(valid),fold_values=vs))
    for name,value in [('per_candidate',records),('per_task',taskrows),('summaries',summaries),('paired',paired),('overall',overall),('selection_paired',selection),('selection_overall',selection_overall)]:write(o/'evaluation'/f'{name}.json',value)
    write(o/'evaluation/coverage.json',dict(full=seal['full_fivefold'],query_rows=sum(r['partition']=='query' for r in records),train_rows=sum(r['partition']=='train' for r in records),train_queries_are_repeated=True,primary_aggregation='179 unique per seed, then mean3 seeds',selection_aggregation='each fold/seed first; seeds then folds',GP_intervals='observation: latent*sT^2+extra_noise*sT^2+mean(training_SE^2); z1.96; no clipping; no query SE',selection_NA=sum(not r['paired_valid'] for r in selection),excluded_primary=sum(len(r['excluded']) for r in paired)))
    assert all(file_hash(o/n)==h for n,h in seal['artifacts'].items())
    print('evaluation complete',len(records),'rows',flush=True)
    for r in overall:
        if r['scope']=='all' and r['partition']=='query' and r['weighting']=='candidate':print(r)
    for r in selection_overall:
        if r['scope']=='all' and r['partition']=='query':print(r)
if __name__=='__main__':report(Path(sys.argv[1]).resolve())
