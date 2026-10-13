"""Saved R1 pilot evaluation. No model forward or optimization; separate process."""
from pathlib import Path
import argparse
import os
import sys
import math
import inspect
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


def report(output):
    seal=read(output/'all_inference_sealed.json');protocol=read(output/'protocol.json')
    assert seal['protocol_identity']==protocol['identity'] and seal['cost']['effective_updates']==1200
    assert seal['training_process_pid']!=os.getpid()
    for path,h in seal['artifacts'].items():assert file_hash(output/path)==h
    querypath=Path(protocol['query_label_source'])
    assert file_hash(querypath)==protocol['query_label_source_sha256']
    # This is the first query-label association in the R1 workflow.
    access=output/'evaluation_label_access.json'
    if not access.exists():write(access,dict(time=utc(),pid=os.getpid(),sealed_training_pid=seal['training_process_pid'],seal_sha256=file_hash(output/'all_inference_sealed.json')))
    trainlabels=read(output/'inputs/training_labels.json');querylabels=read(querypath)
    assert set(trainlabels)==set(protocol['train_ids']) and set(querylabels)==set(protocol['query_ids'])
    assert all(isinstance(y,(int,float)) and math.isfinite(y) and 0<=y<=1 for y in querylabels.values())
    labels={cid:row['validation_mean'] for cid,row in trainlabels.items()}
    write(output/'evaluation_inputs/query_validation_means.json',querylabels)
    write(output/'evaluation_inputs/training_validation_means.json',labels)
    identity=read(output/'inputs/candidate_identity.json');architectures={c['candidate_id']:c['architecture_fingerprint'] for part in identity.values() for c in part}
    allrows=[];spaces={}
    for seed in [0,1,2]:
        for branch in ['R0','R1']:
            folder=output/'trajectories'/f's{seed}-{branch}'
            done=read(folder/'completed.json');assert done['effective_steps']==200
            bundle=read(folder/'inference_add200.json');sp=read(folder/'neighbors_before_labels.json')
            for method,space in sp.items():
                state=f's{seed}-{method}';spaces[state]=space
                if method=='R0-1D':
                    assert space['train_embeddings']==[[v] for v in bundle['train_predictions']]
                    assert space['query_embeddings']==[[v] for v in bundle['query_predictions']]
                for part in ['train','query']:
                    ids=bundle[part+'_ids'];pred=bundle[part+'_predictions'];actual=labels if part=='train' else querylabels
                    assert ids==[r['candidate_id'] for r in space['rows'][part]]
                    for cid,head,row in zip(ids,pred,space['rows'][part]):
                        refs=row['reference_ids']; assert cid not in refs
                        assert set(refs)==set(protocol['train_ids'])-({cid} if part=='train' else set())
                        d=np.array(row['distances']);g=np.array([abs(actual[cid]-labels[i]) for i in refs])
                        neighbors=[refs.index(c) for c in row['neighbor_ids']]
                        metrics=evaluate_query(d,g,refs,neighbors)
                        knn=float(np.mean([labels[c] for c in row['neighbor_ids']]))
                        allrows.append(dict(state=state,seed=seed,method=method,partition=part,candidate_id=cid,architecture=architectures[cid],
                            validation_mean=actual[cid],head_prediction=head,knn_prediction=knn,head_MAE_pp=100*abs(head-actual[cid]),
                            knn_MAE_pp=100*abs(knn-actual[cid]),**metrics))
    write(output/'evaluation/per_query.json',allrows)
    summaries=[];paired=[];overall=[]
    for part in ['train','query']:
        for key in METRICS:
            for weighting in ['candidate','architecture']:
                seedvalues={method:[] for method in METHODS}
                delta_values={method:[] for method in METHODS[:2]}
                for seed in [0,1,2]:
                    selected={m:{r['candidate_id']:r for r in allrows if r['seed']==seed and r['partition']==part and r['method']==m} for m in METHODS}
                    common=set.intersection(*[{cid for cid,r in rows.items() if r[key] is not None} for rows in selected.values()])
                    full=set(protocol['train_ids'] if part=='train' else protocol['query_ids'])
                    assert all(set(rows)==full for rows in selected.values())
                    for method,rows in selected.items():
                        mean=average([rows[c] for c in sorted(common)],key,weighting);seedvalues[method].append(mean)
                        summaries.append(dict(partition=part,metric=key,weighting=weighting,seed=seed,method=method,mean=mean,
                            valid_queries=len(common),excluded_ids=sorted(full-common),NA_reason=None if common else 'no jointly valid queries'))
                    for control in METHODS[:2]:
                        delta=[]
                        for cid in sorted(common):
                            delta.append({'architecture':selected['R1-8D'][cid]['architecture'],key:selected['R1-8D'][cid][key]-selected[control][cid][key]})
                        value=average(delta,key,weighting);delta_values[control].append(value)
                        paired.append(dict(partition=part,metric=key,weighting=weighting,seed=seed,comparison='R1-8D minus '+control,mean_delta=value,
                            valid_queries=len(common),excluded_ids=sorted(full-common)))
                for method,values in seedvalues.items():
                    valid=[v for v in values if v is not None]
                    overall.append(dict(partition=part,metric=key,weighting=weighting,method=method,seed_values=values,mean=float(np.mean(valid)) if valid else None,
                        minimum=min(valid) if valid else None,maximum=max(valid) if valid else None))
                for control,values in delta_values.items():
                    valid=[v for v in values if v is not None]
                    overall.append(dict(partition=part,metric=key,weighting=weighting,method='R1-8D minus '+control,seed_values=values,
                        mean=float(np.mean(valid)) if valid else None,minimum=min(valid) if valid else None,maximum=max(valid) if valid else None))
    write(output/'evaluation/summaries.json',summaries);write(output/'evaluation/paired.json',paired);write(output/'evaluation/overall.json',overall)
    write(output/'evaluation/coverage.json',dict(rows=len(allrows),holdout_unique=29,train_unique=112,methods=3,seeds=3,
        NA_global=sum(r['global_order'] is None for r in allrows),NA_local=sum(r['local10_order'] is None for r in allrows),
        performance_ties=sum(r['global_performance_ties'] for r in allrows),distance_ties=sum(r['global_distance_ties'] for r in allrows),
        negative_E_clipped=sum(r['E_raw_accuracy']<0 for r in allrows),global_pairs=sum(r['global_total_pairs'] for r in allrows),
        excluded_comparisons=sum(len(r['excluded_ids']) for r in paired),GP_MAE=None,GP_MAE_reason='本轮未评价',uncertainty=None,search_gain=None))
    sources=['analyse/relative_metric_report.py','analyse/fuzzy_metric_relative_geometry.py']
    for p in sources:freeze_copy(ROOT/p,output/'report_source_snapshot'/p)
    write(output/'report_source_identity.json',dict(hashes={p:file_hash(ROOT/p) for p in sources},geometry_function_import_path=inspect.getsourcefile(evaluate_query),
        execution='saved inference only; no model construction/forward, optimizer or GP fitting'))
    # Recheck the sealed inference after evaluation.
    assert all(file_hash(output/p)==h for p,h in seal['artifacts'].items())
    print('saved evaluation complete: 9 endpoint spaces,1269 rows; heldout queries29, no optimization')
    for r in overall:
        if r['partition']=='query' and r['weighting']=='candidate' and r['metric'] in ['global_order','E_pp','head_MAE_pp','knn_MAE_pp']:
            print(r['metric'],r['method'],r['mean'],r['seed_values'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);report(p.parse_args().output.resolve())
