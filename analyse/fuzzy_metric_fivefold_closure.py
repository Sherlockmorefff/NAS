#!/usr/bin/env python3
"""Zero-training closure of the original outer-fold N2 procedure.

Reuses verified geometry formulas with fold-specific reference counts. A state is
selected by completed task identity and artifact hashes, never by latest attempt.
All geometry is saved before any A validation means are decoded for evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import datetime
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from analyse.fuzzy_metric_relative_geometry import evaluate_query, select_neighbors, distribution, matrix_csv
from analyse.fuzzy_metric_diagnosis_supplement import read, write, csv_write, no_optimization
from analyse.fuzzy_metric_offline import read_selected_labels
from analyse.fuzzy_metric_replay import digest, file_hash, tensor_hash
from analyse.joint_configuration_metric_gate import hp_feature_vector
from surrogate.fuzzy_metric import JointMetricModel

OLD=ROOT/'results/posthoc/fuzzy-metric-stage2-20260910-v1'
DIAG=ROOT/'results/posthoc/fuzzy-metric-stage2-diagnosis-20260910-v1'
GEOM=ROOT/'results/posthoc/fuzzy-metric-relative-geometry-20260910-v1'
OUT=ROOT/'results/posthoc/fuzzy-metric-fivefold-closure-20260910-v1'
METHODS=('regression','N2')
GEOMETRY_METRICS=('global_order','local10_order','De_pp','Dstar_pp','E_pp','Drandom_pp','De_minus_random_pp',
                  'exceeds_tolerance_0.006_fraction','exceeds_tolerance_0.012_fraction')
PREDICTION_METRICS=('old_gp_error_pp','old_head_error_pp','old_knn_error_pp')
METRICS=GEOMETRY_METRICS+PREDICTION_METRICS


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def weighted_values(rows,key,weighting):
    rows=[r for r in rows if r[key] is not None]
    if weighting=='query_equal':return [r[key] for r in rows]
    groups=defaultdict(list)
    for r in rows:groups[r['architecture_fingerprint']].append(r[key])
    return [float(np.mean(v)) for _,v in sorted(groups.items())]


def validate_fold(fold,candidates,a_ids):
    tids,qids=fold['train_ids'],fold['query_ids']
    if len(set(tids+qids))!=len(tids)+len(qids) or set(tids+qids)!=set(a_ids):
        raise ValueError('fold must be a disjoint candidate partition of A')
    ta,qa=[{candidates[c]['architecture_fingerprint'] for c in ids} for ids in (tids,qids)]
    if ta & qa:raise ValueError('fold train/query architectures overlap')
    if min(len(tids),len(qids))<1:raise ValueError('empty frozen fold')
    return {'family':fold['family'],'train_count':len(tids),'query_count':len(qids),
            'train_architecture_count':len(ta),'query_architecture_count':len(qa),'architecture_overlap':0}


def oof_identity(rows,a_ids):
    counts=Counter(r['candidate_id'] for r in rows)
    duplicates=sorted(c for c,n in counts.items() if n!=1)
    return {'count':len(rows),'unique_count':len(counts),'duplicates':duplicates,
            'missing_ids':sorted(set(a_ids)-set(counts)),'unexpected_ids':sorted(set(counts)-set(a_ids)),
            'complete':not duplicates and set(counts)==set(a_ids)}


def completed_state(task,expected_identity):
    valid,rejected=[],[]
    for done_path in sorted((OLD/'tasks'/task['task_id']).glob('attempt*/completed.json')):
        try:
            done=read(done_path)
            if done['identity']!=expected_identity:raise ValueError('completion identity mismatch')
            for k in ('task_id','family','stage','seed','setting'):
                if done[k]!=task[k]:raise ValueError(f'completion {k} mismatch')
            if Path(done['directory']).resolve()!=done_path.parent.resolve():raise ValueError('completion directory mismatch')
            if done_path.parent.name!=f"attempt{int(done['attempt']):02d}":raise ValueError('attempt identity mismatch')
            if set(done['artifact_hashes'])!={'model.pt','predictions_before_reveal.json','evaluation.json'}:
                raise ValueError('completion artifact set mismatch')
            for name,h in done['artifact_hashes'].items():
                if file_hash(done_path.parent/name)!=h:raise ValueError(f'artifact hash mismatch: {name}')
            started=read(done_path.parent/'started.json')
            if started['identity']!=expected_identity or started['task']!=task:raise ValueError('started task differs')
            valid.append((done_path,done))
        except Exception as exc:rejected.append({'completion':str(done_path),'reason':str(exc)})
    if len(valid)!=1:raise ValueError(f'expected exactly one identity-valid completion, got {len(valid)}; rejected={rejected}')
    path,done=valid[0]
    return {'task':task,'completion_path':str(path),'completion_sha256':file_hash(path),
            'model_path':str(path.parent/'model.pt'),'artifact_hashes':done['artifact_hashes'],
            'completion_identity':expected_identity,'attempt':done['attempt'],'rejected_completions':rejected}


def restore(record,common,candidates,base):
    task=record['task'];folder=Path(record['model_path']).parent
    state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    tids,qids=task['train_ids'],task['query_ids']
    if state['train_ids']!=tids or state['query_ids']!=qids:raise ValueError('state split/order mismatch')
    if state['setting']!=task['setting'] or state['model_seed']!=task['seed']:raise ValueError('state setting/seed mismatch')
    if state['common_config']!=common or common['representation_steps']!=100:raise ValueError('saved common config is not original100')
    if {float(v['step']) for v in state['optimizer']['state'].values()}!={100.}:raise ValueError('saved mapping steps not100')
    train,query=[[candidates[c] for c in ids] for ids in (tids,qids)]
    if digest({'train':train,'query':query})!=state['candidate_identity_hash']:raise ValueError('state candidate identity mismatch')
    scaler=state['standardization']
    if scaler['fit_ids']!=tids or scaler['hp_feature_order']!=['log10_lr','dropout','log2_hidden_dim','log10_l2']:
        raise ValueError('original scaler identity mismatch')
    def encoded(rows):
        z=torch.tensor([base[c['architecture_fingerprint']] for c in rows],dtype=torch.float32)
        hp=torch.tensor([hp_feature_vector(c) for c in rows],dtype=torch.float32)
        zm,zs,hm,hs=[torch.tensor(scaler[k],dtype=torch.float32) for k in ('z_mean','z_scale','hp_mean','hp_scale')]
        return (z-zm)/zs,(hp-hm)/hs
    z,h=encoded(train);qz,qh=encoded(query)
    model=JointMetricModel(z_dim=z.shape[1],seed=task['seed'])
    if tensor_hash(model.state_dict())!=state['initialization_hash']:raise ValueError('model architecture/init differs')
    model.load_state_dict(state['model'],strict=True);model.requires_grad_(False);model.eval()
    parameter_before=tensor_hash(model.state_dict())
    e,_=model(z,h);qe,_=model(qz,qh)
    if not torch.equal(e,state['train_embeddings']) or not torch.equal(e.double(),state['gp_train_inputs']):
        raise ValueError('saved training embedding not bitwise reproduced')
    if e.shape!=(len(tids),8) or qe.shape!=(len(qids),8):raise ValueError('embedding dimension differs')
    d=torch.cdist(qe.double(),e.double()).numpy()
    if not np.isfinite(d).all() or tensor_hash(model.state_dict())!=parameter_before:raise ValueError('invalid frozen inference')
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('model not frozen/eval')
    old=read(folder/'predictions_before_reveal.json')
    for k in ('train_ids','query_ids','setting','model_seed','candidate_identity_hash','standardization','initialization_hash'):
        if old[k]!=state[k]:raise ValueError(f'old prediction/state {k} differs')
    if old['diagnostics']['mapping_steps']!=100 or old['diagnostics']['gp_steps']!=common['gp_steps']:
        raise ValueError('old recorded training steps differ')
    if [r['candidate_id'] for r in old['predictions']]!=qids:raise ValueError('old prediction order differs')
    neighbors=[]
    for i,row in enumerate(old['predictions']):
        nearest=select_neighbors(d[i],tids,5)
        if row['neighbor_ids']!=[tids[j] for j in nearest]:raise ValueError('old top5 IDs differ')
        if not np.array_equal(d[i,nearest],np.array(row['neighbor_distances'])):raise ValueError('old top5 distances differ')
        neighbors.append(nearest)
    return {'state_id':task['task_id'],'family':task['family'],'method':task['method'],'seed':task['seed'],
            'setting':task['setting'],'mapping_steps':100,'train_ids':tids,'query_ids':qids,
            'train_architecture_by_id':{c['candidate_id']:c['architecture_fingerprint'] for c in train},
            'query_architecture_by_id':{c['candidate_id']:c['architecture_fingerprint'] for c in query},
            'candidate_identity_hash':state['candidate_identity_hash'],'standardization':scaler,
            'train_embeddings':e.tolist(),'query_embeddings':qe.tolist(),'euclidean_distances':d.tolist(),
            'embedding_neighbor_indices':neighbors,'embedding_neighbor_ids':[[tids[j] for j in ns] for ns in neighbors],
            'provenance':record,'parameter_hash':parameter_before,'saved_before_label_access_utc':utc(),
            'checks':{'train_embeddings_bitwise_equal':True,'gp_train_inputs_same_coordinates':True,
                      'old_top5_IDs_equal':True,'old_top5_distance_max_diff':0.,'parameters_unchanged':True,
                      'eval_mode':True,'requires_grad':False,'device':'cpu'}}


def common_pair(a,b,ids,key,weighting):
    a,b={r['candidate_id']:r for r in a},{r['candidate_id']:r for r in b}
    paired,exclusions=[],[]
    for cid in ids:
        if cid not in a or cid not in b:
            exclusions.append({'candidate_id':cid,'reason':'state/query missing'});continue
        if a[cid][key] is None or b[cid][key] is None:
            exclusions.append({'candidate_id':cid,'reason':'no comparable performance pairs' if key.endswith('order') else 'reference metric missing'});continue
        if a[cid]['family']!=b[cid]['family'] or a[cid]['architecture_fingerprint']!=b[cid]['architecture_fingerprint']:
            raise ValueError('same-query pair uses another fold/architecture')
        paired.append({'candidate_id':cid,'architecture_fingerprint':a[cid]['architecture_fingerprint'],
                       'regression':a[cid][key],'N2':b[cid][key],'delta':b[cid][key]-a[cid][key]})
    vals={k:weighted_values(paired,k,weighting) for k in ('regression','N2','delta')}
    return {'expected_queries':len(ids),'valid_query_count':len(paired),'excluded_query_count':len(exclusions),
            'valid_architecture_count':len({r['architecture_fingerprint'] for r in paired}),'exclusions':exclusions,
            **{k:float(np.mean(v)) if v else None for k,v in vals.items()}}


def pair_summaries(rows,partition,high_ids):
    output=[]
    scopes=[('overall',partition['A_ids'])]+[(f['family'],f['query_ids']) for f in partition['folds']]
    for scope,ids in scopes:
        for subset in ('all_queries','top10_each_fold_descriptive'):
            chosen=[c for c in ids if subset=='all_queries' or c in high_ids]
            for weighting in ('query_equal','architecture_equal'):
                for key in METRICS:
                    per_seed={}
                    for seed in (0,1,2):
                        a=[r for r in rows if r['method']=='regression' and r['seed']==seed and r['candidate_id'] in chosen]
                        b=[r for r in rows if r['method']=='N2' and r['seed']==seed and r['candidate_id'] in chosen]
                        per_seed[str(seed)]=common_pair(a,b,chosen,key,weighting)
                    values=[v['delta'] for v in per_seed.values() if v['delta'] is not None]
                    method_means={m:float(np.mean([v[m] for v in per_seed.values() if v[m] is not None])) if any(v[m] is not None for v in per_seed.values()) else None for m in METHODS}
                    output.append({'scope':scope,'subset':subset,'weighting':weighting,'metric':key,
                                   'unit':'0_to_1' if key.endswith('order') else ('fraction' if key.endswith('fraction') else 'percentage_points'),
                                   'per_seed':per_seed,'regression_mean':method_means['regression'],'N2_mean':method_means['N2'],
                                   **{f'seed{s}_delta':per_seed[str(s)]['delta'] for s in (0,1,2)},'delta_distribution':distribution(values)})
    return output


def closure_decision(pairs,identity_complete,fold_names):
    def get(scope,key):
        row=next(r for r in pairs if (r['scope'],r['metric'],r['subset'],r['weighting'])==(scope,key,'all_queries','query_equal'))
        return row['delta_distribution']['mean']
    def clean(x,tol):return None if x is None else (0. if abs(x)<=tol else x)
    overall_raw={k:get('overall',k) for k in ('global_order','E_pp')}
    overall={k:clean(v,1e-12 if k=='global_order' else 1e-10) for k,v in overall_raw.items()}
    folds=[]
    for family in fold_names:
        raw={k:get(family,k) for k in ('global_order','E_pp')}
        c,e=clean(raw['global_order'],1e-12),clean(raw['E_pp'],1e-10)
        folds.append({'family':family,'raw_delta_C':raw['global_order'],'raw_delta_E_pp':raw['E_pp'],
                      'effective_delta_C':c,'effective_delta_E_pp':e,'usable':c is not None and e is not None,
                      'joint_improvement':c is not None and e is not None and c>0 and e<0})
    material_complete=identity_complete and all(f['usable'] for f in folds) and all(v is not None for v in overall.values())
    main_good=all(v is not None for v in overall.values()) and overall['global_order']>0 and overall['E_pp']<0
    fold_count=sum(f['joint_improvement'] for f in folds)
    passed=material_complete and main_good and fold_count>=4
    if not material_complete:status='现有材料不足以完成五折判断，已完成可用部分；当前N2继续暂停进入搜索。'
    elif passed:status='观察到跨折相对几何收益线索；本轮收尾完成。'
    else:status='当前N2流程额外收益证据不足，本版不进入搜索；本轮收尾完成。'
    return {'status':status,'material_complete':material_complete,'raw_overall':overall_raw,'effective_overall':overall,
            'overall_joint_improvement':main_good,'joint_improvement_fold_count':fold_count,'required_fold_count':4,
            'folds':folds,'rule_satisfied':passed,'original_stage2_passed':False,'enter_search':False,
            'rule_role':'this closure resource-decision rule only; not original Stage2 gate or significance test',
            'same_version_further_split_seed_threshold_search':False}


def state_summaries(rows,high_ids):
    result=[]
    for sid in dict.fromkeys(r['state_id'] for r in rows):
        group=[r for r in rows if r['state_id']==sid]
        for subset in ('all_queries','top10_each_fold_descriptive'):
            chosen=[r for r in group if subset=='all_queries' or r['candidate_id'] in high_ids]
            for weighting in ('query_equal','architecture_equal'):
                for key in METRICS:
                    result.append({'state_id':sid,'subset':subset,'weighting':weighting,'metric':key,
                                   'query_count':len(chosen),'excluded_query_count':sum(r[key] is None for r in chosen),
                                   **distribution(weighted_values(chosen,key,weighting))})
    return result


def run(output):
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise FileExistsError('initialized result; use a fresh --output; never overwrite evidence')
    prior,partition,matrix,boundary=[read(OLD/n) for n in ('protocol.json','partition.json','task_matrix.json','label_boundary.json')]
    if digest(partition)!=prior['partition_hash'] or digest(matrix)!=prior['task_matrix_hash']:raise ValueError('original split/matrix changed')
    for p,h in prior['source_identity'].items():
        if file_hash(ROOT/p)!=h:raise ValueError(f'original numerical source changed: {p}')
    previous=read(GEOM/'delivery_manifest.json')
    for p,h in previous['files_changed_this_turn'].items():
        if p.endswith('.py') and file_hash(ROOT/p)!=h:raise ValueError(f'previous verified geometry source changed: {p}')
    prep=ROOT/prior['config']['prepared'];candidate_rows,base=read(prep/'candidates.json'),read(prep/'base_encoding.json')
    if digest(candidate_rows)!=prior['prepared_manifest']['pool_hash'] or digest(base)!=prior['prepared_manifest']['base_encoding_hash']:
        raise ValueError('original unlabeled inputs changed')
    aids=partition['A_ids'];candidates={c['candidate_id']:c for c in candidate_rows if c['candidate_id'] in aids}
    if len(aids)!=179 or len({c['architecture_fingerprint'] for c in candidates.values()})!=96:raise ValueError('original A identity count differs')
    folds=partition['folds'];fold_info=[validate_fold(f,candidates,aids) for f in folds]
    qcounts=[r['query_count'] for r in fold_info];tcounts=[r['train_count'] for r in fold_info]
    if qcounts!=[38,36,36,34,35] or tcounts!=[141,143,143,145,144]:raise ValueError('original fold counts differ')
    if Counter(c for f in folds for c in f['query_ids'])!=Counter(aids):raise ValueError('frozen fold queries do not cover A once')
    if set(aids)&set(partition['B_ids']):raise ValueError('A/B identity overlap')
    sources=['analyse/fuzzy_metric_fivefold_closure.py','tests/test_fuzzy_metric_fivefold_closure.py',
             'analyse/fuzzy_metric_relative_geometry.py','analyse/fuzzy_metric_diagnosis_supplement.py',
             'analyse/fuzzy_metric_offline.py','analyse/fuzzy_metric_replay.py','analyse/joint_configuration_metric_gate.py','surrogate/fuzzy_metric.py']
    records,inventory,missing=[],[],[];selections={}
    for fold in folds:
        sel_path=OLD/'selection'/f"{fold['family']}.json";selection=read(sel_path)
        selections[fold['family']]={'settings':{m:selection['settings'][m] for m in METHODS},'source':str(sel_path),'sha256':file_hash(sel_path),
            'original_inner_prediction_gate_satisfied':selection['N1_N2_joint_prediction_gate_satisfied']}
        for method in METHODS:
            for seed in (0,1,2):
                sid=f"{fold['family']}-refit-{method}-s{seed}"
                try:
                    matches=[r for r in matrix if r.get('task_id')==sid]
                    if len(matches)!=1:raise ValueError('matrix task missing/duplicated')
                    task=copy.deepcopy(matches[0])
                    if task['stage']!='refit' or task['family']!=fold['family'] or task['method']!=method or task['seed']!=seed:
                        raise ValueError('not requested outer refit task')
                    if task['train_ids']!=fold['train_ids'] or task['query_ids']!=fold['query_ids'] or task['setting_source']!=f"selection/{fold['family']}.json":
                        raise ValueError('matrix split/selection source differs')
                    task['setting']=selection['settings'][method]
                    identity=digest({'task':task,'protocol':prior['protocol_hash'],'A_labels':boundary['exported_hash']})
                    rec=completed_state(task,identity);records.append(rec)
                    inventory.append({'state_id':sid,'status':'valid_completed_record',**rec})
                except Exception as exc:
                    item={'state_id':sid,'status':'missing_or_invalid','reason':str(exc)};inventory.append(item);missing.append(item)
    input_paths=[OLD/n for n in ('protocol.json','partition.json','task_matrix.json','label_boundary.json','A_validation.json')]
    input_paths += [prep/'candidates.json',prep/'base_encoding.json',DIAG/'saved_diagnostics/train_holdout_spaces.json',DIAG/'saved_diagnostics/completed.json']
    input_paths += [Path(r['completion_path']) for r in records]+[Path(r['model_path']) for r in records]
    frozen={'frozen_utc':utc(),'purpose':'closure of original inner-selected N2 procedure on historical A; not independent confirmation',
            'partition':{'A_ids':aids,'A_architectures':partition['A_architectures'],'folds':[{k:v for k,v in f.items() if k!='inner'} for f in folds]},
            'fold_info':fold_info,'state_inventory':inventory,'expected_states':30,'expected_query_state_rows':len(aids)*6,
            'selected_parameters':selections,'model_seeds':[0,1,2],'mapping_steps':100,'lambda_e':prior['config']['lambda_e'],
            'distance':'8D original Euclidean; saved float32 inputs/weights; distance float64; no normalization or scaler refit',
            'performance_tie_accuracy':1e-12,'distance_tie_relative':1e-10,'distance_scale':'maximum over current fold entire reference pool; same for local10',
            'scores':'performance ties excluded; same1, reverse0, distance ties0.5; no valid pairs NA',
            'neighbor_order':'exact raw distance/gap then candidate_id; numerical scoring ties never ID-broken',
            'embedding_k':5,'oracle_k':5,'oracle_local_k':10,'E_negative_error_accuracy':-1e-12,'gap_units':'percentage_points; raw E accuracy retained',
            'random_reference':'exact current fold mean absolute gap; expectation of uniform random5 without replacement, not MAE',
            'tolerances_accuracy':[.006,.012],'high_group':'per fold ceil(.1*query_count), validation mean descending and ID tie; all fold references',
            'main_aggregation':'per seed concatenate fold-local query metrics; each A candidate once, candidate-equal mean, then seed mean',
            'fold_aggregation':'same fold/seed/query pairs, query mean then mean of seed deltas',
            'architecture_sensitivity':'within-architecture query mean, architecture-equal mean per seed, then seed mean; does not replace main',
            'NA_policy':'per metric common valid queries on both sides; exclusions disclosed; NA alone not failure; missing state/identity or unusable primary fold means incomplete',
            'closure_rule':{'overall':'delta_C>0 AND delta_E_pp<0','folds':'at least4 of5 with seed-mean delta_C_f>0 AND delta_E_f_pp<0',
                            'numerical_zero_C':1e-12,'numerical_zero_E_pp':1e-10,'auxiliary_metrics_change_gate':False,
                            'role':'resource-decision working rule; not original gate/significance/general standard'},
            'allowed_validation_ids':aids,'B_labels_read':0,'test_labels_read':0,'A_expansion_authorized':True,
            'training_budget':{'GNN':0,'VAE':0,'mapping':0,'GP':0},'optimization_budget':0,'threads':1,'device':'cpu',
            'source_hashes':{p:file_hash(ROOT/p) for p in sources},'input_hashes':{str(p):file_hash(p) for p in input_paths}}
    write(output/'protocol.json',frozen);write(output/'state_inventory.json',inventory)
    write(output/'candidate_identity_before_labels.json',[candidates[c] for c in aids])
    for p in sources:
        dest=output/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dest)
    restored={};failures=[]
    with no_optimization() as guard:
        for rec in records:
            sid=rec['task']['task_id']
            try:
                snap=restore(rec,prior['config'],candidates,base);folder=output/'states'/sid
                write(folder/'geometry_before_labels.json',snap)
                matrix_csv(folder/'distance_matrix.csv',snap['query_ids'],snap['train_ids'],snap['euclidean_distances'])
                restored[sid]=snap
            except Exception:
                failures.append({'state_id':sid,'phase':'readonly_restore','traceback':traceback.format_exc()})
                write(output/'failures.json',failures)
        freeze={'frozen_at_utc':utc(),'states':len(restored),'files':{str(p.relative_to(output)):file_hash(p) for p in (output/'states').rglob('*') if p.is_file()}}
        write(output/'distance_freeze.json',freeze)
        # The expanded A label permission is used only after all available distance files are frozen.
        first_label_access=utc();label_records=read_selected_labels(OLD/'A_validation.json',aids)
        if digest(label_records)!=boundary['exported_hash']:raise ValueError('original A-only validation records changed')
        means={c:float(label_records[c]['validation_mean']) for c in aids};del label_records
        write(output/'allowed_validation_means.json',{'first_label_access_utc':first_label_access,'means':means,
              'meaning':'existing validation means over original GNN seeds0/1/2; A development only'})
        csv_write(output/'allowed_validation_means.csv',[{'candidate_id':c,'architecture_fingerprint':candidates[c]['architecture_fingerprint'],'validation_mean':means[c]} for c in aids])
        high=[]
        for f in folds:
            chosen=sorted(f['query_ids'],key=lambda c:(-means[c],c))[:math.ceil(.1*len(f['query_ids']))]
            high.append({'family':f['family'],'candidate_ids':chosen,'query_count':len(f['query_ids']),'reference_count':len(f['train_ids'])})
        high_ids={c for f in high for c in f['candidate_ids']};write(output/'high_performance_queries.json',high)
        # Existing diagnostics of the SAME original outer models, never the later inner M300 models.
        spaces_path=DIAG/'saved_diagnostics/train_holdout_spaces.json'
        saved_done=read(DIAG/'saved_diagnostics/completed.json')
        if file_hash(spaces_path)!=saved_done['artifacts'][spaces_path.name]:raise ValueError('old head/knn diagnostic file changed')
        spaces={r['task_id']:r for r in read(spaces_path) if r['task_id'] in restored}
        rows=[];state_checks=[];references=[]
        for sid,snap in restored.items():
            tids,qids=snap['train_ids'],snap['query_ids'];folder=output/'states'/sid
            y=np.array([means[c] for c in tids]);qy=np.array([means[c] for c in qids]);gaps=np.abs(qy[:,None]-y[None,:]);d=np.array(snap['euclidean_distances'])
            original_folder=Path(snap['provenance']['model_path']).parent
            old_eval=read(original_folder/'evaluation.json')
            old_rows={r['candidate_id']:r for r in old_eval['per_query']}
            sp=spaces.get(sid);sp_rows={r['candidate_id']:r for r in sp['unseen_architecture_query']['per_query']} if sp else {}
            if sp and (sp['family']!=snap['family'] or sp['stage']!='refit' or sp['seed']!=snap['seed'] or sp['method']!=snap['method'] or set(sp_rows)!=set(qids)):
                raise ValueError('old head/knn reference identity mismatch')
            state_rows=[]
            for i,cid in enumerate(qids):
                metric=evaluate_query(d[i],gaps[i],tids,snap['embedding_neighbor_indices'][i]);oldr=old_rows[cid]
                if oldr['true_validation_mean']!=means[cid] or oldr['neighbor_ids']!=metric['embedding_neighbor_ids']:
                    raise ValueError('old evaluation label/neighbor identity differs')
                if abs(100*oldr['neighbor_mean_abs_difference']-metric['De_pp'])>1e-12:raise ValueError('old query neighbor metric differs')
                spr=sp_rows.get(cid)
                if spr and (spr['neighbor_ids']!=metric['embedding_neighbor_ids'] or abs(spr['neighbor_mean_abs_difference']*100-metric['De_pp'])>1e-12):
                    raise ValueError('old head/knn reference used another geometry')
                if spr and (abs(spr['head_absolute_error']-abs(spr['head_prediction']-means[cid]))>1e-12 or abs(spr['knn_absolute_error']-abs(float(y[snap['embedding_neighbor_indices'][i]].mean())-means[cid]))>1e-12):
                    raise ValueError('old predictor reference labels differ')
                row={'state_id':sid,'family':snap['family'],'method':snap['method'],'seed':snap['seed'],
                     'candidate_id':cid,'architecture_fingerprint':candidates[cid]['architecture_fingerprint'],'validation_mean':means[cid],
                     **metric,'old_gp_error_pp':100*oldr['gp_absolute_error'],
                     'old_head_error_pp':100*spr['head_absolute_error'] if spr else None,'old_knn_error_pp':100*spr['knn_absolute_error'] if spr else None}
                state_rows.append(row)
                references.append({'state_id':sid,'candidate_id':cid,'family':snap['family'],'seed':snap['seed'],'method':snap['method'],
                    'GP_source':str(original_folder/'evaluation.json'),'head_knn_source':str(spaces_path) if spr else None,
                    'GP_error_pp':row['old_gp_error_pp'],'head_error_pp':row['old_head_error_pp'],'knn_error_pp':row['old_knn_error_pp'],
                    'old_GP_prediction':oldr['prediction_mean'],'old_head_prediction':spr['head_prediction'] if spr else None,
                    'record_status':'complete' if spr else 'head/knn saved reference absent; no new prediction computed'})
            med=float(np.median([r['De_pp'] for r in state_rows]));oldmed=100*old_eval['metrics']['main_neighborhood_metric']
            if abs(med-oldmed)>1e-12:raise ValueError('old median metric differs')
            for tol in ('.006','.012'):
                key=str(float(tol));new=float(np.mean([r[f'exceeds_tolerance_{key}_fraction'] for r in state_rows]))
                if abs(new-old_eval['metrics']['tolerances'][key]['wrong_neighbor_ratio'])>1e-12:raise ValueError('old tolerance ratio differs')
            write(folder/'performance_gap_matrix.json',{'query_ids':qids,'train_ids':tids,'unit':'percentage_points','performance_gaps_pp':(100*gaps).tolist()})
            matrix_csv(folder/'performance_gap_matrix_pp.csv',qids,tids,100*gaps)
            write(folder/'per_query.json',state_rows);csv_write(folder/'per_query.csv',state_rows)
            rows.extend(state_rows);state_checks.append({'state_id':sid,**snap['checks'],'query_count':len(qids),'train_count':len(tids),
                                                       'old_median_metric_difference_pp':med-oldmed,'old_prediction_reference_count':len(sp_rows)})
        oof={f'{m}-s{s}':oof_identity([r for r in rows if r['method']==m and r['seed']==s],aids) for m in METHODS for s in (0,1,2)}
        identities_complete=len(restored)==30 and not missing and not failures and all(v['complete'] for v in oof.values())
        for cid in aids:
            rr=[r for r in rows if r['candidate_id']==cid]
            for key in ('Dstar_pp','Drandom_pp'):
                if len({r[key] for r in rr})>1:raise ValueError('same-fold oracle/random reference differs by state')
        pairs=pair_summaries(rows,partition,high_ids);summaries=state_summaries(rows,high_ids)
        verdict=closure_decision(pairs,identities_complete,[f['family'] for f in folds])
        paired_query=[]
        by={(r['family'],r['seed'],r['candidate_id'],r['method']):r for r in rows}
        for r in rows:
            if r['method']!='regression':continue
            b=by.get((r['family'],r['seed'],r['candidate_id'],'N2'))
            if b:paired_query.append({k:r[k] for k in ('family','seed','candidate_id','architecture_fingerprint')} | {k:b[k]-r[k] if b[k] is not None and r[k] is not None else None for k in METRICS})
        exports={'per_query':rows,'state_summaries':summaries,'paired_summary':pairs,'paired_per_query':paired_query,'old_prediction_reference':references}
        for name,value in exports.items():write(output/f'{name}.json',value);csv_write(output/f'{name}.csv',value)
        write(output/'closure_decision.json',verdict)
        unchanged=all(file_hash(output/p)==h for p,h in freeze['files'].items())
        if not unchanged:raise ValueError('pre-label geometry files changed')
        checks={'expected_states':30,'completed_states':len(restored),'expected_rows':len(aids)*6,'query_state_rows':len(rows),
                'missing_or_invalid_states':missing,'restore_failures':failures,'OOF':oof,'folds':fold_info,'per_state':state_checks,
                'all_geometry_frozen_before_labels':freeze['frozen_at_utc']<first_label_access,'pre_label_files_unchanged':unchanged,
                'distance_freeze_utc':freeze['frozen_at_utc'],'first_label_access_utc':first_label_access,
                'Dstar_Drandom_invariant_within_fold':True,'E_negative_clamped':sum(r['E_raw_accuracy']<0 for r in rows),
                'global_pairs':sum(r['global_total_pairs'] for r in rows),'global_valid_pairs':sum(r['global_valid_denominator'] for r in rows),
                'global_performance_ties':sum(r['global_performance_ties'] for r in rows),'global_distance_ties':sum(r['global_distance_ties'] for r in rows),
                'local_pairs':sum(r['local10_total_pairs'] for r in rows),'global_NA':sum(r['global_order'] is None for r in rows),'local_NA':sum(r['local10_order'] is None for r in rows),
                'head_knn_reference_missing_rows':sum(r['old_head_error_pp'] is None for r in rows),
                'new_training_counts':{'GNN':0,'VAE':0,'mapping':0,'GP':0},'new_optimization_steps':0,'zero_optimization_guard':guard.copy(),
                'VAE_objects_constructed':0,'GP_objects_constructed':0,'B_labels_read':0,'test_labels_read':0,'unique_A_means_read':len(means),
                'device':'cpu','torch_threads':torch.get_num_threads(),'elapsed_seconds':time.monotonic()-started}
        write(output/'checks.json',checks)
    print({'output':str(output),'states':len(restored),'rows':len(rows),'status':verdict['status'],'new_training':0,'updates':0})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--readonly',action='store_true',required=True);p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);run(args.output)


if __name__=='__main__':main()
