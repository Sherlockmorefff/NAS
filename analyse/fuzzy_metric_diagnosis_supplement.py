#!/usr/bin/env python3
"""Zero-training supplement using 24 existing snapshots from one fixed inner split.

No optimizer is instantiated. The only label reader decodes a frozen ID whitelist
from A_validation.json. Each GP is a fresh eval-only object with its own cache.
All predictions and kernel summaries are saved before query-label evaluation.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import gpytorch
import numpy as np
import torch
from analyse.fuzzy_metric_offline import read_selected_labels
from analyse.fuzzy_metric_offline_core import _label_arrays
from analyse.fuzzy_metric_replay import ReplayGP,digest,file_hash,tensor_hash
from analyse.joint_configuration_metric_gate import hp_feature_vector
from surrogate.fuzzy_metric import JointMetricModel
OLD=ROOT/'results/posthoc/fuzzy-metric-stage2-diagnosis-20260910-v1'
OUT=ROOT/'results/posthoc/fuzzy-metric-stage2-readonly-supplement-20260910-v1'
METHODS=('regression','N2')
METRICS=('main_neighborhood_metric','wrong_0.006','wrong_0.012','head_mae','knn_mae','gp_mae','constant_mae',
         'coverage90','coverage95','width90','width95')


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def csv_write(path,rows):
    rows=list(rows)
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def distribution(values):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all():raise ValueError('nonfinite diagnosis values')
    return {'count':int(values.size),'mean':float(values.mean()),'std_population':float(values.std()),
        'min':float(values.min()),'p25':float(np.percentile(values,25)),'median':float(np.median(values)),
        'p75':float(np.percentile(values,75)),'max':float(values.max())}


@contextmanager
def no_optimization():
    counts={'optimizer_initializations':0,'backward_calls':0,'optimization_updates':0}
    def forbidden_optimizer(*args,**kwargs):
        counts['optimizer_initializations']+=1;raise RuntimeError('optimizer creation forbidden in readonly supplement')
    def forbidden_backward(*args,**kwargs):
        counts['backward_calls']+=1;raise RuntimeError('backward forbidden in readonly supplement')
    with patch.object(torch.optim.Optimizer,'__init__',forbidden_optimizer),patch.object(torch.Tensor,'backward',forbidden_backward),patch('torch.autograd.backward',forbidden_backward),torch.no_grad():
        yield counts


def restore_snapshot(method,seed,mstep,gstep,protocol,candidates,base,train_labels):
    mp=OLD/'tasks'/f'{method}-s{seed}-mapping';gpdir=OLD/'tasks'/f'{method}-s{seed}-M{mstep}-gp'
    paths={'mapping':mp/f'M{mstep}.pt','gp':gpdir/f'G{gstep}.pt','prediction':gpdir/f'G{gstep}_predictions_before_reveal.json'}
    for key in ('mapping','gp'):
        if not paths[key].exists():raise FileNotFoundError(str(paths[key]))
    state=load(paths['mapping']);gstate=load(paths['gp'])
    for folder in (mp,gpdir):
        completed=read(folder/'completed.json')
        for name,h in completed['artifacts'].items():
            if name.endswith('.pt') or name.endswith('_predictions_before_reveal.json'):
                if file_hash(folder/name)!=h:raise ValueError(f'saved artifact changed: {folder/name}')
    ids,qids=protocol['inner']['train_ids'],protocol['inner']['query_ids']
    if state['train_ids']!=ids or state['query_ids']!=qids:raise ValueError('checkpoint split/order differs')
    if state['setting']!=protocol['settings'][method] or state['model_seed']!=seed:raise ValueError('checkpoint setting/seed differs')
    if {float(v['step']) for v in state['optimizer']['state'].values()}!={float(mstep)}:raise ValueError('mapping checkpoint step differs')
    if gstate['step']!=gstep or {float(v['step']) for v in gstate['optimizer']['state'].values()}!={float(gstep)}:raise ValueError('GP checkpoint step differs')
    # Optimizer dictionaries are read only for provenance; none is constructed or loaded.
    cfg=state['common_config']
    if (cfg['lambda_e'],cfg['gp_seed'],cfg['gp_kernel'])!=(.001,0,'isotropic_Matern_5_2'):raise ValueError('original model assumptions changed')
    train,query=[[candidates[c] for c in which] for which in (ids,qids)]
    if digest({'train':train,'query':query})!=state['candidate_identity_hash']:raise ValueError('candidate configuration changed')
    if {c['architecture_fingerprint'] for c in train}&{c['architecture_fingerprint'] for c in query}:raise ValueError('architecture leakage')
    scaler=state['standardization']
    if scaler['fit_ids']!=ids or scaler['hp_feature_order']!=['log10_lr','dropout','log2_hidden_dim','log10_l2']:raise ValueError('scaler training identity differs')
    y,se,_=_label_arrays(ids,train_labels)
    ym,ys=torch.tensor(scaler['y_mean'],dtype=torch.double),torch.tensor(scaler['y_scale'],dtype=torch.double)
    if float(y.mean())!=float(ym) or float(y.std(correction=0).clamp_min(1e-8))!=float(ys):raise ValueError('saved target transform differs from training records')
    def encoded(rows):
        z=torch.tensor([base[c['architecture_fingerprint']] for c in rows],dtype=torch.float32)
        hp=torch.tensor([hp_feature_vector(c) for c in rows],dtype=torch.float32)
        zm,zs,hm,hs=[torch.tensor(scaler[k],dtype=torch.float32) for k in ('z_mean','z_scale','hp_mean','hp_scale')]
        return (z-zm)/zs,(hp-hm)/hs
    z,h=encoded(train);qz,qh=encoded(query)
    model=JointMetricModel(z_dim=z.shape[1],seed=seed)
    if tensor_hash(model.state_dict())!=state['initialization_hash']:raise ValueError('saved model architecture/init differs')
    model.load_state_dict(state['model'],strict=True);model.requires_grad_(False);model.eval()
    e,_=model(z,h);qe,qhead=model(qz,qh)
    if not torch.equal(e,state['train_embeddings']) or not torch.equal(e.double(),gstate['train_inputs']):raise ValueError('GP input belongs to another mapping snapshot')
    target=(y-ym)/ys
    if not torch.equal(target,gstate['train_targets']):raise ValueError('GP target transform differs')
    gp=ReplayGP(gstate['train_inputs'],gstate['train_targets'],se,float(ys)).double()
    gp.load_state_dict(gstate['gp'],strict=True);gp.requires_grad_(False);gp.eval();gp.likelihood.eval()
    fixed=(se/ys).square()
    if not torch.equal(gp.likelihood.noise_covar.noise,fixed):raise ValueError('fixed observation noise was not restored')
    if gp.training or gp.likelihood.training or model.training:raise ValueError('inference requires eval mode')
    before_model=tensor_hash(model.state_dict());before_gp=tensor_hash(gp.state_dict())
    with gpytorch.settings.cholesky_jitter(double_value=1e-6):
        posterior=gp(qe.double())
        means=(posterior.mean*ys+ym).numpy();latent=(posterior.variance.clamp_min(0)*ys.square()).numpy()
    heads=(qhead.double()*ys+ym).numpy()
    gp_mean_function=(gp.mean_module(qe.double())*ys+ym).numpy()
    x,xq=gp.train_inputs[0],qe.double()
    lengthscale=gp.covar_module.base_kernel.lengthscale.detach()
    relative=torch.cdist(xq/lengthscale,x/lengthscale).numpy()
    raw_distance=torch.cdist(qe.double(),e.double()).numpy()
    cross=gp.covar_module(xq,x).to_dense()
    qdiag=gp.covar_module(xq,diag=True);tdiag=gp.covar_module(x,diag=True)
    rho=(cross/torch.sqrt(qdiag[:,None]*tdiag[None,:])).numpy()
    if not np.isfinite(rho).all() or rho.min()<-1e-12 or rho.max()>1+1e-12:raise ValueError('invalid signal kernel correlation')
    pooled=float(se.square().mean());learned=float(gp.likelihood.second_noise)*float(ys)**2
    rows=[]
    for i,c in enumerate(query):
        nearest=sorted(range(len(ids)),key=lambda j:(float(raw_distance[i,j]),ids[j]))[:5]
        correlated=sorted(range(len(ids)),key=lambda j:(-float(rho[i,j]),ids[j]))[:5]
        relative_nearest=min(range(len(ids)),key=lambda j:(float(relative[i,j]),ids[j]))
        variance=float(latent[i])+pooled+learned
        row={'candidate_id':c['candidate_id'],'architecture_fingerprint':c['architecture_fingerprint'],
            'gp_mean':float(means[i]),'head_mean':float(heads[i]),'knn_mean':float(y[nearest].mean()),
            'constant_mean':float(y.mean()),'gp_mean_function_accuracy':float(gp_mean_function[i]),
            'gp_minus_training_mean':float(means[i]-float(y.mean())),
            'gp_minus_mean_function':float(means[i]-gp_mean_function[i]),
            'abs_gp_minus_training_mean':float(abs(means[i]-float(y.mean()))),
            'abs_gp_minus_mean_function':float(abs(means[i]-gp_mean_function[i])),
            'neighbor_ids':[ids[j] for j in nearest],'neighbor_distances':[float(raw_distance[i,j]) for j in nearest],
            'kernel_max_correlation':float(rho[i,correlated[0]]),'kernel_top5_mean_correlation':float(rho[i,correlated].mean()),
            'kernel_top5_ids':[ids[j] for j in correlated],'kernel_top5_correlations':[float(rho[i,j]) for j in correlated],
            'nearest_GP_distance_over_lengthscale':float(relative[i,relative_nearest]),'nearest_GP_id':ids[relative_nearest],
            'kernel_top5_equals_metric_neighbors':correlated==nearest,
            'latent_variance':float(latent[i]),'pooled_training_SE_squared':pooled,
            'additional_noise_accuracy_squared':learned,'observation_variance':variance}
        for level,zscore in [('90',1.6448536269514722),('95',1.959963984540054)]:
            half=zscore*math.sqrt(variance);row[f'interval{level}']=[row['gp_mean']-half,row['gp_mean']+half]
        rows.append(row)
    old_prediction=read(paths['prediction']) if paths['prediction'].exists() else None
    checks={'GP_instance':'new independent eval-only instance; fresh prediction cache','mapping_step':mstep,'GP_step':gstep,
        'model_state_unchanged':before_model==tensor_hash(model.state_dict()),'GP_state_unchanged':before_gp==tensor_hash(gp.state_dict()),
        'fixed_noise_restored_exact':True,'train_inputs_exact':True,'train_targets_exact':True,'saved_prediction_available':old_prediction is not None}
    if old_prediction:
        if old_prediction['train_ids']!=ids or old_prediction['query_ids']!=qids or old_prediction['standardization']!=scaler:raise ValueError('saved predictions use different preprocessing/IDs')
        maximum=0.
        for new,old in zip(rows,old_prediction['predictions']):
            if new['candidate_id']!=old['candidate_id'] or new['neighbor_ids']!=old['neighbor_ids']:raise ValueError('old prediction identity/neighbors differ')
            for key in ('gp_mean','head_mean','knn_mean','latent_variance','pooled_training_SE_squared','additional_noise_accuracy_squared','observation_variance'):
                maximum=max(maximum,abs(new[key]-old[key]))
            for key in ('neighbor_distances','interval90','interval95'):
                maximum=max(maximum,float(np.max(np.abs(np.array(new[key])-old[key]))))
        if maximum>1e-12:raise ValueError(f'restored predictions differ: {maximum}')
        checks['max_reproduction_abs_error']=maximum
        # Existing prediction values are primary; inference augments their kernel diagnostics.
        for new,old in zip(rows,old_prediction['predictions']):new.update(old)
    params={'lengthscale':lengthscale.tolist(),'lengthscale_scalar':float(lengthscale) if lengthscale.numel()==1 else None,
        'outputscale_normalized':float(gp.covar_module.outputscale),
        'signal_variance_accuracy_squared':float(gp.covar_module.outputscale)*float(ys)**2,
        'signal_std_accuracy':math.sqrt(float(gp.covar_module.outputscale))*float(ys),
        'GP_mean_function_accuracy':float(gp_mean_function[0]),'training_mean_accuracy':float(ym),
        'target_scale':float(ys),'additional_noise_normalized':float(gp.likelihood.second_noise),
        'additional_noise_accuracy_squared':learned,'pooled_training_SE_squared':pooled}
    return {'method':method,'seed':seed,'mapping_steps':mstep,'gp_steps':gstep,'train_ids':ids,'query_ids':qids,
        'standardization':scaler,'predictions':rows,'gp_parameters':params,'checks':checks,
        'source_paths':{k:str(p) for k,p in paths.items()},'source_hashes':{k:file_hash(p) for k,p in paths.items() if p.exists()},
        'GP_input_space':'saved joint embedding converted float32 to double; no extra preprocessing',
        'neighbor_space':'original eight-dimensional joint embedding Euclidean distance',
        'kernel_rule':'actual signal covariance divided by sqrt(diagonal products); excludes noise and jitter',
        'query_labels_used_for_inference':False,'query_SE_used_for_inference':False,'new_training_steps':0}


def evaluate(bundle,train_labels,query_labels):
    if set(query_labels)!=set(bundle['query_ids']):raise ValueError('query label boundary differs')
    rows=[]
    for p in bundle['predictions']:
        y=query_labels[p['candidate_id']]['validation_mean'];gaps=[abs(y-train_labels[c]['validation_mean']) for c in p['neighbor_ids']]
        row={**p,'true_validation_mean':y,'neighbor_abs_differences':gaps,'neighbor_mean_abs_difference':float(np.mean(gaps))}
        for m in ('head','knn','gp','constant'):row[f'{m}_absolute_error']=abs(y-row[f'{m}_mean'])
        row['mean_function_absolute_error']=abs(y-row['gp_mean_function_accuracy'])
        for m in ('head','knn','gp'):row[f'{m}_error_minus_constant']=row[f'{m}_absolute_error']-row['constant_absolute_error']
        for t in (.006,.012):row[f'wrong_{t}']=float(np.mean(np.array(gaps)>t))
        for level in ('90','95'):
            lo,hi=p[f'interval{level}'];row[f'coverage{level}']=float(lo<=y<=hi);row[f'width{level}']=hi-lo
        rows.append(row)
    metric={'main_neighborhood_metric':float(np.median([r['neighbor_mean_abs_difference'] for r in rows])),
        **{f'{m}_mae':float(np.mean([r[f'{m}_absolute_error'] for r in rows])) for m in ('head','knn','gp','constant')}}
    for k in ('wrong_0.006','wrong_0.012','coverage90','coverage95','width90','width95'):
        metric[k]=float(np.mean([r[k] for r in rows]))
    for m in ('head','knn','gp'):metric[f'{m}_mae_minus_constant']=metric[f'{m}_mae']-metric['constant_mae']
    keys=['gp_mean','gp_minus_training_mean','gp_minus_mean_function','abs_gp_minus_training_mean','abs_gp_minus_mean_function',
        'kernel_max_correlation','kernel_top5_mean_correlation','nearest_GP_distance_over_lengthscale',
        'latent_variance','pooled_training_SE_squared','additional_noise_accuracy_squared','observation_variance','gp_absolute_error']
    mechanism={k:distribution([r[k] for r in rows]) for k in keys}
    weak=sorted(rows,key=lambda r:(r['kernel_max_correlation'],r['candidate_id']))[:math.ceil(len(rows)/4)]
    weak_ids={r['candidate_id'] for r in weak};strong=[r for r in rows if r['candidate_id'] not in weak_ids]
    mechanism['weakest_correlation_quartile']={'rule':'within-snapshot bottom ceil(n/4) by max correlation, ID ties; descriptive only',
        'candidate_ids':sorted(weak_ids),'count':len(weak),'GP_MAE':float(np.mean([r['gp_absolute_error'] for r in weak])),
        'other_count':len(strong),'other_GP_MAE':float(np.mean([r['gp_absolute_error'] for r in strong]))}
    mechanism['mean_function_mae']=float(np.mean([r['mean_function_absolute_error'] for r in rows]))
    return {'metrics':metric,'mechanism':mechanism,'per_query':rows}


def mean_range(values):
    return {'mean':float(np.mean(values)),'min':float(min(values)),'max':float(max(values))}


def paired_differences(combos):
    by={(r['method'],r['seed'],r['mapping_steps'],r['gp_steps']):r for r in combos};rows=[]
    pairs=[]
    for method in METHODS:
        for g in (20,100):pairs.append((f'{method}: M300-M100 at G{g}',[(method,s,100,g) for s in (0,1,2)],[(method,s,300,g) for s in (0,1,2)]))
        for m in (100,300):pairs.append((f'{method}: G100-G20 at M{m}',[(method,s,m,20) for s in (0,1,2)],[(method,s,m,100) for s in (0,1,2)]))
    for m in (100,300):
        for g in (20,100):pairs.append((f'N2-regression at M{m}/G{g}',[('regression',s,m,g) for s in (0,1,2)],[('N2',s,m,g) for s in (0,1,2)]))
    for name,before,after in pairs:
        if any(k not in by for k in before+after):continue
        for metric in METRICS:
            vals=[by[a]['metrics'][metric]-by[b]['metrics'][metric] for a,b in zip(after,before)]
            rows.append({'comparison':name,'metric':metric,'seed0':vals[0],'seed1':vals[1],'seed2':vals[2],**mean_range(vals)})
    return rows


def safe_correlation(a,b):
    a,b=np.array(a),np.array(b)
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None


def mechanism_changes(combos):
    by={(r['method'],r['seed'],r['mapping_steps'],r['gp_steps']):r for r in combos};output=[]
    for method in METHODS:
        for seed in (0,1,2):
            for m in (100,300):
                ka,kb=(method,seed,m,20),(method,seed,m,100)
                if ka not in by or kb not in by:continue
                a,b=by[ka],by[kb];rows=[]
                for x,y in zip(a['evaluation']['per_query'],b['evaluation']['per_query']):
                    if x['candidate_id']!=y['candidate_id']:raise ValueError('paired candidate order differs')
                    keys=('kernel_max_correlation','kernel_top5_mean_correlation','nearest_GP_distance_over_lengthscale',
                        'abs_gp_minus_training_mean','abs_gp_minus_mean_function','gp_absolute_error','latent_variance',
                        'pooled_training_SE_squared','additional_noise_accuracy_squared','observation_variance','width90','width95')
                    rows.append({'candidate_id':x['candidate_id'],'architecture_fingerprint':x['architecture_fingerprint'],
                        **{f'{k}_G20':x[k] for k in keys},**{f'{k}_G100':y[k] for k in keys},
                        **{f'{k}_change':y[k]-x[k] for k in keys}})
                weak_ids=set(b['evaluation']['mechanism']['weakest_correlation_quartile']['candidate_ids'])
                weak=[r for r in rows if r['candidate_id'] in weak_ids];other=[r for r in rows if r['candidate_id'] not in weak_ids]
                positive=[max(0,r['gp_absolute_error_change']) for r in rows];den=sum(positive)
                summary={'mean_changes':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k.endswith('_change')},
                    'GP_prediction_std_G20':a['evaluation']['mechanism']['gp_mean']['std_population'],
                    'GP_prediction_std_G100':b['evaluation']['mechanism']['gp_mean']['std_population'],
                    'correlation_change_vs_error_change_Pearson':safe_correlation([r['kernel_max_correlation_change'] for r in rows],[r['gp_absolute_error_change'] for r in rows]),
                    'G100_correlation_vs_error_change_Pearson':safe_correlation([r['kernel_max_correlation_G100'] for r in rows],[r['gp_absolute_error_change'] for r in rows]),
                    'weakest_G100_quartile_mean_error_change':float(np.mean([r['gp_absolute_error_change'] for r in weak])),
                    'other_mean_error_change':float(np.mean([r['gp_absolute_error_change'] for r in other])),
                    'weakest_quartile_share_of_positive_error_increase':sum(max(0,r['gp_absolute_error_change']) for r in weak)/den if den else None,
                    'weaker_correlation_count':sum(r['kernel_max_correlation_change']<0 for r in rows),
                    'worse_error_count':sum(r['gp_absolute_error_change']>0 for r in rows),
                    'both_weaker_and_worse_count':sum(r['kernel_max_correlation_change']<0 and r['gp_absolute_error_change']>0 for r in rows)}
                output.append({'method':method,'seed':seed,'mapping_steps':m,'summary':summary,'per_query':rows})
    return output


def run(output=OUT):
    started=time.monotonic();output.mkdir(parents=True,exist_ok=True)
    protocol=read(OLD/'protocol.json');stage2=ROOT/protocol['config']['old_output'];prior=read(stage2/'protocol.json')
    for p,h in {**prior['source_identity'],**protocol['sources']}.items():
        if file_hash(ROOT/p)!=h:raise ValueError(f'original numerical source changed: {p}')
    ids,qids=protocol['inner']['train_ids'],protocol['inner']['query_ids'];allowed=set(ids+qids)
    if (len(ids),len(qids))!=(112,29) or allowed&set(protocol['outer_holdout_ids_excluded_from_new_comparisons']):raise ValueError('wrong inner split')
    prep=ROOT/prior['config']['prepared']
    candidate_rows=read(prep/'candidates.json');base=read(prep/'base_encoding.json')
    if digest(candidate_rows)!=prior['prepared_manifest']['pool_hash'] or digest(base)!=prior['prepared_manifest']['base_encoding_hash']:raise ValueError('unlabeled inputs changed')
    by_id={c['candidate_id']:c for c in candidate_rows if c['candidate_id'] in allowed}
    boundary={'train_ids':ids,'query_ids':qids,'train_count':112,'query_count':29,'model_seeds':[0,1,2],
        'B_labels_read':0,'outer_holdout_labels_read':0,'test_labels_read':0,
        'use':'already used inner development set; same29 across all model seeds; not87 independent candidates'}
    write(output/'boundary.json',boundary)
    training=read_selected_labels(stage2/'A_validation.json',ids)
    write(output/'training_labels.json',training)
    write(output/'training_access.json',{'ids':ids,'time':time.time(),'source':str(stage2/'A_validation.json')})
    sources=['analyse/fuzzy_metric_diagnosis_supplement.py','surrogate/fuzzy_metric.py','analyse/fuzzy_metric_replay.py',
        'analyse/fuzzy_metric_offline.py','analyse/fuzzy_metric_offline_core.py','analyse/joint_configuration_metric_gate.py']
    frozen={'old_diagnosis_protocol':str(OLD/'protocol.json'),'old_protocol_sha256':file_hash(OLD/'protocol.json'),
        'old_stage2_protocol_sha256':file_hash(stage2/'protocol.json'),'train_ids':ids,'query_ids':qids,
        'settings':protocol['settings'],'lambda_e':.001,'gp_seed':0,'mapping_steps':[100,300],'gp_steps':[20,100],
        'model_seeds':[0,1,2],'methods':list(METHODS),'new_training_budget':{'GNN':0,'VAE':0,'mapping':0,'GP':0},
        'new_optimizer_step_budget':0,'execution':'single thread CPU; no optimizer; no backward',
        'sources':{p:file_hash(ROOT/p) for p in sources},'unlabeled_inputs':{str(prep/p):file_hash(prep/p) for p in ['candidates.json','base_encoding.json']},
        'label_source_sha256':file_hash(stage2/'A_validation.json'),'reference':'unweighted arithmetic mean of112 training candidate validation means',
        'analysis_rule':'fixed2x2; descriptive seed ranges; bottom correlation quartile predefined ceil(n/4), no inference threshold tuning'}
    if (output/'config_and_identity.json').exists() and read(output/'config_and_identity.json')!=frozen:raise ValueError('readonly source/config changed; archive prior attempt before resuming')
    write(output/'config_and_identity.json',frozen)
    for p in sources:
        dest=output/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(ROOT/p,dest)
    missing=[];bundles=[];checks=[];reused=0
    with no_optimization() as counts:
        for method in METHODS:
            for seed in (0,1,2):
                for m in (100,300):
                    for g in (20,100):
                        name=f'{method}-s{seed}-M{m}-G{g}';folder=output/'combinations'/name;folder.mkdir(parents=True,exist_ok=True)
                        try:
                            done=folder/'inference_complete.json'
                            if done.exists():
                                record=read(done)
                                if file_hash(folder/'predictions_before_evaluation.json')!=record['prediction_sha256']:raise ValueError('stored inference changed')
                                bundle=read(folder/'predictions_before_evaluation.json');reused+=1
                            else:
                                bundle=restore_snapshot(method,seed,m,g,protocol,by_id,base,training)
                                write(folder/'predictions_before_evaluation.json',bundle)
                                write(done,{'time':time.time(),'prediction_sha256':file_hash(folder/'predictions_before_evaluation.json')})
                            bundles.append((folder,bundle));checks.append({'combination':name,**bundle['checks']})
                        except FileNotFoundError as error:
                            missing.append({'combination':name,'reason':str(error)});write(folder/'missing.json',missing[-1])
                        except Exception:
                            n=len(list(folder.glob('readonly_failure*.json')))+1
                            write(folder/f'readonly_failure{n:02d}.json',{'time':time.time(),'traceback':traceback.format_exc(),'new_optimization_steps':0});raise
        # All new inference records precede this query-label read. Old evaluations are not used to form predictions.
        access={'ids':qids,'time':time.time(),'predictions':{f.name:read(f/'inference_complete.json') for f,_ in bundles}}
        write(output/'evaluation_access.json',access)
        query=read_selected_labels(stage2/'A_validation.json',qids)
        write(output/'query_validation_means.json',{c:query[c]['validation_mean'] for c in qids})
        combos=[];flat=[]
        for folder,bundle in bundles:
            evaluation=evaluate(bundle,training,query);write(folder/'evaluation.json',evaluation)
            meta={k:bundle[k] for k in ['method','seed','mapping_steps','gp_steps']}
            combos.append({**meta,'metrics':evaluation['metrics'],'gp_parameters':bundle['gp_parameters'],'evaluation':evaluation})
            flat.extend([{**meta,**r} for r in evaluation['per_query']])
    grouped=[]
    for method in METHODS:
        for m in (100,300):
            for g in (20,100):
                rows=[r for r in combos if (r['method'],r['mapping_steps'],r['gp_steps'])==(method,m,g)]
                grouped.append({'method':method,'mapping_steps':m,'gp_steps':g,'seeds':[r['seed'] for r in rows],
                    'metrics':{metric:mean_range([r['metrics'][metric] for r in rows]) for metric in METRICS} if rows else {}})
    differences=paired_differences(combos);mechanism=mechanism_changes(combos)
    write(output/'summary_2x2.json',grouped);write(output/'paired_differences.json',differences)
    write(output/'mechanism_GP20_to100.json',mechanism)
    metrics_rows=[{k:r[k] for k in ['method','seed','mapping_steps','gp_steps']}|r['metrics'] for r in combos]
    csv_write(output/'metrics_24_combinations.csv',metrics_rows);write(output/'metrics_24_combinations.json',metrics_rows)
    csv_write(output/'paired_differences.csv',differences);csv_write(output/'per_candidate.csv',flat);write(output/'per_candidate.json',flat)
    mechanisms=[{**{k:r[k] for k in ['method','seed','mapping_steps','gp_steps']},'gp_parameters':r['gp_parameters'],**r['evaluation']['mechanism']} for r in combos]
    write(output/'kernel_and_mean_summaries.json',mechanisms)
    fixed=[]
    index={(b['method'],b['seed'],b['mapping_steps'],b['gp_steps']):b for _,b in bundles}
    for method in METHODS:
        for seed in (0,1,2):
            for m in (100,300):
                a,b=index.get((method,seed,m,20)),index.get((method,seed,m,100))
                if a is None or b is None:continue
                for x,y in zip(a['predictions'],b['predictions']):
                    for k in ['candidate_id','neighbor_ids','neighbor_distances','head_mean','knn_mean','constant_mean']:
                        if x[k]!=y[k]:raise ValueError(f'GP changed fixed mapping output: {k}')
                fixed.append(f'{method}-s{seed}-M{m}')
    constants={r['metrics']['constant_mae'] for r in combos}
    if len(constants)!=1:raise ValueError('constant reference differs between snapshots')
    # Preserve the original18 reported combinations and verify the previously omitted6 evaluations too.
    previous=read(OLD/'comparison.json');old_differences=[]
    condition={(100,20):'baseline',(100,100):'long_gp',(300,100):'long_mapping'}
    for row in combos:
        old_ev=read(OLD/'tasks'/f"{row['method']}-s{row['seed']}-M{row['mapping_steps']}-gp"/f"G{row['gp_steps']}_evaluation.json")
        delta=max(abs(row['metrics'][k]-old_ev['metrics'][k]) for k in METRICS if k!='constant_mae')
        if (row['mapping_steps'],row['gp_steps']) in condition:
            old_row=next(x for x in previous['methods'][row['method']]['per_seed'] if x['seed']==row['seed'] and x['condition']==condition[row['mapping_steps'],row['gp_steps']])
            delta=max(delta,max(abs(row['metrics'][k]-old_row[k]) for k in METRICS if k!='constant_mae'))
        if delta>1e-12:raise ValueError('old evaluation or summary not reproduced')
        old_differences.append(delta)
    verification={'expected_combinations':24,'completed_combinations':len(combos),'missing':missing,'checks':checks,
        'fixed_mapping_GP_pairs_identical':len(fixed),'constant_MAE_identical':list(constants)[0],
        'max_old_metric_difference':max(old_differences),'prediction_before_evaluation':all(read(f/'inference_complete.json')['time']<=access['time'] for f,_ in bundles),
        'zero_update_guards':counts,'new_training_counts':{'GNN':0,'VAE':0,'mapping':0,'GP':0},
        'new_optimizer_updates':0,'query_unique_candidates':29,'not_independent_87_candidates':True,
        'GP_instances_fresh_per_snapshot':True,'no_query_SE_in_prediction':True,'B_and_outer_and_test_labels_read':0,
        'reused_inference_combinations':reused,'elapsed_seconds':time.monotonic()-started}
    write(output/'checks.json',verification);print(json.dumps({k:v for k,v in verification.items() if k!='checks'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--readonly',action='store_true');args=p.parse_args()
    if not args.readonly:p.error('only --readonly is supported')
    torch.set_num_threads(1)
    run()
