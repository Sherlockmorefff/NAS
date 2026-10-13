"""Read-only evaluation after all fixed fivefold trajectories are sealed."""
from pathlib import Path
import sys,os,math,inspect
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from analyse.relative_metric_pilot import read,write,file_hash,utc,freeze_copy
from analyse.fuzzy_metric_relative_geometry import evaluate_query
METRICS=['global_order','local10_order','De_pp','Dstar_pp','E_pp','Drandom_pp','De_minus_random_pp','head_MAE_pp','knn_MAE_pp']
METHODS=['R0-8D','R0-1D','R1-8D']

def average(rows,key,weighting):
    if not rows:return None
    if weighting=='candidate':return float(np.mean([r[key] for r in rows]))
    groups={}
    for r in rows:groups.setdefault(r['architecture'],[]).append(r[key])
    return float(np.mean([np.mean(v) for v in groups.values()]))

def summarize(rows,protocol):
    summaries=[];paired=[];overall=[]
    for scope in ['all']+[f'fold{f}' for f in range(5)]:
      subset=[r for r in rows if scope=='all' or scope==f"fold{r['fold']}"]
      for part in ['train','query']:
       for key in METRICS:
        for weighting in ['candidate','architecture']:
         for seed in range(3):
          selected={m:{(r['fold'],r['candidate_id']):r for r in subset if r['partition']==part and r['seed']==seed and r['method']==m} for m in METHODS}
          for method,records in selected.items():
           valid={i for i,r in records.items() if r[key] is not None}
           summaries.append(dict(scope=scope,partition=part,metric=key,weighting=weighting,seed=seed,method=method,
             mean=average([records[i] for i in sorted(valid)],key,weighting),valid_queries=len(valid),NA_queries=len(records)-len(valid),
             excluded_ids=[list(i) for i in sorted(set(records)-valid)],NA_reason=None if valid else 'no valid queries'))
          for control in METHODS[:2]:
           a,b=selected['R1-8D'],selected[control];full=set(a)|set(b)
           common={i for i in set(a)&set(b) if a[i][key] is not None and b[i][key] is not None}
           diffs=[{'architecture':a[i]['architecture'],key:a[i][key]-b[i][key]} for i in sorted(common)]
           paired.append(dict(scope=scope,partition=part,metric=key,weighting=weighting,seed=seed,comparison='R1-8D minus '+control,
             mean_delta=average(diffs,key,weighting),valid_queries=len(common),excluded_ids=[list(i) for i in sorted(full-common)],
             missing_either_task=sum(i not in a or i not in b for i in full),NA_reason=None if common else 'no jointly valid queries'))
         for method in METHODS+['R1-8D minus '+m for m in METHODS[:2]]:
          source=paired if ' minus ' in method else summaries;field='comparison' if ' minus ' in method else 'method';value='mean_delta' if ' minus ' in method else 'mean'
          matching=[r for r in source if r['scope']==scope and r['partition']==part and r['metric']==key and r['weighting']==weighting and r[field]==method]
          values=[next(r[value] for r in matching if r['seed']==s) for s in range(3)];valid=[v for v in values if v is not None]
          overall.append(dict(scope=scope,partition=part,metric=key,weighting=weighting,method=method,seed_values=values,
             mean=float(np.mean(valid)) if valid else None,minimum=min(valid) if valid else None,maximum=max(valid) if valid else None,valid_seeds=len(valid)))
    return summaries,paired,overall

def report(o):
    p=read(o/'protocol.json');seal=read(o/'all_inference_sealed.json');tasks=read(o/'task_matrix.json')
    assert seal['protocol_identity']==p['identity'] and seal['cost']['effective_updates']==len(tasks)*200
    assert len(tasks)==seal['available_trajectories'] and seal['orchestrator_pid']!=os.getpid()
    assert all(file_hash(o/n)==h for n,h in seal['artifacts'].items())
    access=o/'evaluation_label_access.json'
    if not access.exists():write(access,dict(time=utc(),pid=os.getpid(),sealed_at=seal['sealed_at'],seal_sha256=file_hash(o/'all_inference_sealed.json'),all_trajectories_sealed=True))
    labels={}
    for f in range(5):
        path=o/'evaluation_labels'/f'fold{f}.json';assert file_hash(path)==p['query_label_hashes'][str(path.relative_to(o))]
        tr=read(o/'inputs'/f'fold{f}'/'train_labels.json');qu=read(path)
        assert set(tr)==set(p['folds'][f]['train_ids']) and set(qu)==set(p['folds'][f]['query_ids'])
        labels[f]={part:{c:r['validation_mean'] for c,r in data.items()} for part,data in [('train',tr),('query',qu)]}
        write(o/'evaluation_inputs'/f'fold{f}.json',labels[f])
    arch={r['candidate_id']:r['architecture_fingerprint'] for r in read(o/'candidate_identity.json')}
    rows=[]
    for task in tasks:
        f,seed,branch=task['fold'],task['seed'],task['branch'];folder=o/'trajectories'/task['task_id']
        done=read(folder/'completed.json');assert done['effective_steps']==200 and done['fold']==f
        raw=read(folder/'inference_add200.json');spaces=read(folder/'neighbors_before_labels.json')
        assert raw['train_ids']==task['train_ids'] and raw['query_ids']==task['query_ids'] and raw['fold']==f
        for method,space in spaces.items():
         if method=='R0-1D':
            assert space['train_embeddings']==[[v] for v in raw['train_predictions']] and space['query_embeddings']==[[v] for v in raw['query_predictions']]
         for part in ['train','query']:
          for cid,head,saved in zip(raw[part+'_ids'],raw[part+'_predictions'],space['rows'][part]):
            refs=saved['reference_ids'];assert saved['candidate_id']==cid and cid not in refs
            assert set(refs)==set(task['train_ids'])-({cid} if part=='train' else set())
            actual=labels[f][part][cid];train=labels[f]['train']
            distances=np.asarray(saved['distances']);gaps=np.asarray([abs(actual-train[c]) for c in refs]);near=[refs.index(c) for c in saved['neighbor_ids']]
            metrics=evaluate_query(distances,gaps,refs,near);knn=float(np.mean([train[c] for c in saved['neighbor_ids']]))
            rows.append(dict(state=f'fold{f}-s{seed}-{method}',fold=f,seed=seed,method=method,partition=part,candidate_id=cid,architecture=arch[cid],validation_mean=actual,
              head_prediction=head,knn_prediction=knn,head_MAE_pp=100*abs(head-actual),knn_MAE_pp=100*abs(knn-actual),**metrics))
    if seal['full_fivefold']:
        assert sum(r['partition']=='query' for r in rows)==1611 and sum(r['partition']=='train' for r in rows)==6444
        for m in METHODS:
         for s in range(3):
          r=[r for r in rows if r['partition']=='query' and r['seed']==s and r['method']==m]
          assert len(r)==179 and len({v['candidate_id'] for v in r})==179 and {v['candidate_id'] for v in r}==set(p['A_ids'])
    write(o/'evaluation/per_query.json',rows)
    summaries,paired,overall=summarize(rows,p)
    for name,data in [('summaries',summaries),('paired',paired),('overall',overall)]:write(o/'evaluation'/f'{name}.json',data)
    consistency=[]
    for w in ['candidate','architecture']:
     for control in METHODS[:2]:
      vals=[]
      for f in range(5):
        d={r['metric']:r['mean'] for r in overall if r['scope']==f'fold{f}' and r['partition']=='query' and r['weighting']==w and r['method']=='R1-8D minus '+control}
        vals.append(dict(fold=f,deltas=d,both_global_C_up_E_down=d['global_order'] is not None and d['E_pp'] is not None and d['global_order']>0 and d['E_pp']<0))
      consistency.append(dict(weighting=w,control=control,folds=vals,both_improve_count=sum(r['both_global_C_up_E_down'] for r in vals)))
    write(o/'evaluation/fold_consistency.json',consistency)
    write(o/'evaluation/coverage.json',dict(full_fivefold=seal['full_fivefold'],rows=len(rows),holdout_rows=sum(r['partition']=='query' for r in rows),train_rows=sum(r['partition']=='train' for r in rows),holdout_unique=179 if seal['full_fivefold'] else None,
        train_fold_candidate_pairs=716 if seal['full_fivefold'] else None,training_rows_are_repeated_diagnostics=True,architecture_weighting='within architecture mean over current scope fold/candidate rows, then equal architectures; average model seeds last',
        NA_global=sum(r['global_order'] is None for r in rows),NA_local=sum(r['local10_order'] is None for r in rows),negative_E_clipped=sum(r['E_raw_accuracy']<0 for r in rows),
        paired_excluded=sum(len(r['excluded_ids']) for r in paired),GP=None,search=None,GP_search_reason='未评价'))
    sources=['analyse/relative_metric_fivefold_report.py','analyse/fuzzy_metric_relative_geometry.py']
    for rel in sources:freeze_copy(ROOT/rel,o/'report_source_snapshot'/rel)
    write(o/'report_source_identity.json',dict(hashes={r:file_hash(ROOT/r) for r in sources},geometry_function_import_path=inspect.getsourcefile(evaluate_query),evaluation_pid=os.getpid(),optimization_updates=0))
    assert all(file_hash(o/n)==h for n,h in seal['artifacts'].items())
    print('evaluation complete',len(rows),'rows')
    for r in overall:
        if r['scope']=='all' and r['partition']=='query' and r['weighting']=='candidate':print(r['metric'],r['method'],r['mean'],r['seed_values'])
    print('fold consistency',[(r['control'],r['weighting'],r['both_improve_count']) for r in consistency])
if __name__=='__main__':report(Path(sys.argv[1]).resolve())
