#!/usr/bin/env python3
"""Fixed, serial Stage-2 diagnosis. No global oracle, GNN, or VAE execution.

Old saved evidence is written by --audit before --run may optimize. Training
functions receive train labels only; query labels enter after prediction files
are durable. Each scientific trajectory has its own step checkpoint and ledger.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import gpytorch
import numpy as np
import torch
from analyse.fuzzy_metric_offline import read_selected_labels
from analyse.fuzzy_metric_offline_core import _label_arrays, _state_cpu, _loss_numbers, _gradient_norm, NORMAL_QUANTILES
from analyse.fuzzy_metric_replay import ReplayGP, digest, file_hash, tensor_hash, validate_candidates
from analyse.joint_configuration_metric_gate import hp_feature_vector
from surrogate.fuzzy_metric import JointMetricModel, metric_loss, neighborhood_affinity


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read(path) != value:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(_state_cpu(state), temp)
    temp.replace(path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def inputs(train, query, base, labels):
    """Fit every scaler from train only, identical to the frozen Stage-2 core."""
    train, query = [sorted(rows, key=lambda c: c['candidate_id']) for rows in (train, query)]
    validate_candidates(train + query)
    ids, qids = [[c['candidate_id'] for c in rows] for rows in (train, query)]
    if len(set(ids + qids)) != len(ids + qids):
        raise ValueError('train/query IDs overlap')
    if {c['architecture_fingerprint'] for c in train} & {c['architecture_fingerprint'] for c in query}:
        raise ValueError('train/query architectures overlap')
    y, se, seed_y = _label_arrays(ids, labels)
    z = torch.tensor([base[c['architecture_fingerprint']] for c in train], dtype=torch.float32)
    hp = torch.tensor([hp_feature_vector(c) for c in train], dtype=torch.float32)
    zm, zs = z.mean(0), z.std(0, correction=0).clamp_min(1e-6)
    hm, hs = hp.mean(0), hp.std(0, correction=0).clamp_min(1e-6)
    ym, ys = y.mean(), y.std(correction=0).clamp_min(1e-8)
    qz = torch.tensor([base[c['architecture_fingerprint']] for c in query], dtype=torch.float32)
    qh = torch.tensor([hp_feature_vector(c) for c in query], dtype=torch.float32)
    standardization = {'fit_ids':ids,'hp_feature_order':['log10_lr','dropout','log2_hidden_dim','log10_l2'],
        'z_mean':zm.tolist(),'z_scale':zs.tolist(),'hp_mean':hm.tolist(),'hp_scale':hs.tolist(),
        'y_mean':float(ym),'y_scale':float(ys),'input_scale_floor':1e-6,'target_scale_floor':1e-8,
        'target_standard_deviation_correction':0,'input_standard_deviation_correction':0}
    return dict(train=train,query=query,ids=ids,qids=qids,z=(z-zm)/zs,hp=(hp-hm)/hs,
        qz=(qz-zm)/zs,qhp=(qh-hm)/hs,y=y,se=se,seed_y=seed_y,ym=ym,ys=ys,
        target=((y-ym)/ys).float(),standardization=standardization,
        candidate_identity_hash=digest({'train':train,'query':query}))


def targets(data, setting):
    n = len(data['ids'])
    if setting['method'] == 'regression':
        zeros = torch.zeros(n,n,dtype=torch.double)
        return zeros, zeros
    eps = setting['epsilon']
    p = neighborhood_affinity(data['y'],mode=setting['method'],epsilon=eps,k=10)
    pair_se = (data['seed_y'][:,None,:]-data['seed_y'][None,:,:]).std(-1,correction=1)/math.sqrt(3)
    return p, 1/(1+(pair_se/eps).square())


def losses(model, data, setting):
    e, p = model(data['z'],data['hp'])
    affinity, weights = targets(data,setting)
    return metric_loss(e,p,data['target'],affinity,lambda_f=setting['lambda_f'],lambda_e=.001,pair_weights=weights)


def restore_mapping(state, data, *, verify=True):
    model = JointMetricModel(z_dim=data['z'].shape[1], seed=state['model_seed'])
    if verify:
        if state['train_ids'] != data['ids'] or state['query_ids'] != data['qids']:
            raise ValueError('checkpoint is not the exact frozen inner split')
        if state['standardization'] != data['standardization'] or state['candidate_identity_hash'] != data['candidate_identity_hash']:
            raise ValueError('checkpoint scaler or candidate identity differs')
        if tensor_hash(model.state_dict()) != state['initialization_hash']:
            raise ValueError('initialization differs from the saved seed')
        c = state['common_config']
        if any(c[k] != v for k,v in {'learning_rate':.001,'lambda_e':.001,'gp_learning_rate':.05,'gp_seed':0,'early_stopping':False}.items()):
            raise ValueError('original optimization protocol differs')
    model.load_state_dict(state['model'])
    opt = torch.optim.Adam(model.parameters(),lr=.001)
    opt.load_state_dict(_state_cpu(state['optimizer']))
    if any(g['lr'] != .001 for g in opt.param_groups):
        raise ValueError('Adam learning rate differs')
    return model,opt


def mapping_step(model,opt,data,setting,step):
    model.train()
    opt.zero_grad(set_to_none=True)
    value = losses(model,data,setting)
    row = _loss_numbers(value)
    value['total'].backward()
    row.update(step=step,gradient_norm=_gradient_norm(model,'mapping'))
    opt.step()
    return row


def mapping_state(model,opt,data,setting,seed,old):
    model.eval()
    with torch.no_grad():
        e,_ = model(data['z'],data['hp'])
        final = _loss_numbers(losses(model,data,setting))
    return dict(model=_state_cpu(model.state_dict()),optimizer=_state_cpu(opt.state_dict()),
        model_seed=seed,setting=setting,train_ids=data['ids'],query_ids=data['qids'],
        standardization=data['standardization'],candidate_identity_hash=data['candidate_identity_hash'],
        initialization_hash=old['initialization_hash'],model_initial_state=old['model_initial_state'],
        common_config=old['common_config'],train_embeddings=e.detach(),final_losses=final,
        torch_rng=torch.get_rng_state())


def new_gp(data,e):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        gp = ReplayGP(e.detach().double(),(data['y']-data['ym'])/data['ys'],data['se'],float(data['ys'])).double()
    return gp,torch.optim.Adam(gp.parameters(),lr=.05)


def gp_parameters(gp,data):
    return {'lengthscale':float(gp.covar_module.base_kernel.lengthscale.detach()),
        'outputscale_normalized':float(gp.covar_module.outputscale.detach()),
        'outputscale_accuracy_squared':float(gp.covar_module.outputscale.detach())*float(data['ys'])**2,
        'mean_normalized':float(gp.mean_module.constant.detach()),
        'additional_noise_normalized':float(gp.likelihood.second_noise.detach()),
        'additional_noise_accuracy_squared':float(gp.likelihood.second_noise.detach())*float(data['ys'])**2,
        'pooled_training_SE_squared':float(data['se'].square().mean())}


def gp_step(gp,opt,step,data):
    gp.train(); gp.likelihood.train()
    opt.zero_grad(set_to_none=True)
    mll=gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood,gp)
    with gpytorch.settings.cholesky_jitter(double_value=1e-6):
        loss=-mll(gp(gp.train_inputs[0]),gp.train_targets)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite GP objective')
        loss.backward()
        norm=_gradient_norm(gp,'GP')
        opt.step()
    return {'step':step,'negative_marginal_log_likelihood':float(loss.detach()),
        'gradient_norm':norm,'parameters_after_update':gp_parameters(gp,data)}


def neighbor_indices(e,refs,ids,qids):
    distances=torch.cdist(e.double(),refs.double()).detach().numpy()
    return [[j for j in sorted(range(len(ids)),key=lambda j:(float(distances[i,j]),ids[j]))
             if ids[j]!=qid][:5] for i,qid in enumerate(qids)],distances


def predict(model,gp,data):
    """Three predictors and intervals; no query mean, repeated score, or SE input."""
    model.eval();gp.eval();gp.likelihood.eval()
    with torch.no_grad(),gpytorch.settings.cholesky_jitter(double_value=1e-6):
        e,_=model(data['z'],data['hp']);qe,qhead=model(data['qz'],data['qhp'])
        if not torch.equal(e.double(),gp.train_inputs[0]):
            raise ValueError('GP and mapping coordinates differ')
        posterior=gp(qe.double())
        means=(posterior.mean*data['ys']+data['ym']).numpy()
        variances=(posterior.variance.clamp_min(0)*data['ys'].square()).numpy()
        heads=(qhead.double()*data['ys']+data['ym']).numpy()
    neighbors,distances=neighbor_indices(qe,e,data['ids'],data['qids'])
    noise=gp_parameters(gp,data)
    rows=[]
    for i,c in enumerate(data['query']):
        js=neighbors[i]; variance=float(variances[i])+noise['pooled_training_SE_squared']+noise['additional_noise_accuracy_squared']
        row={'candidate_id':c['candidate_id'],'architecture_fingerprint':c['architecture_fingerprint'],
            'gp_mean':float(means[i]),'head_mean':float(heads[i]),'knn_mean':float(data['y'][js].mean()),
            'neighbor_ids':[data['ids'][j] for j in js],'neighbor_distances':[float(distances[i,j]) for j in js],
            'latent_variance':float(variances[i]),'pooled_training_SE_squared':noise['pooled_training_SE_squared'],
            'additional_noise_accuracy_squared':noise['additional_noise_accuracy_squared'],'observation_variance':variance}
        for level,z in NORMAL_QUANTILES.items():
            width=z*math.sqrt(variance)
            row[f'interval{level}']=[row['gp_mean']-width,row['gp_mean']+width]
        rows.append(row)
    bundle={'train_ids':data['ids'],'query_ids':data['qids'],'standardization':data['standardization'],
        'candidate_identity_hash':data['candidate_identity_hash'],'predictions':rows,'gp_parameters':noise,
        'query_labels_used':False,'query_SE_used':False,'test_labels_used':False,
        'target':'finite_observed_three_seed_validation_mean','variance_units':'accuracy_squared',
        'knn_rule':'uniform mean of five train labels; distance then candidate_id; exclude self',
        'query_ranking':[r['candidate_id'] for r in sorted(rows,key=lambda r:(-r['gp_mean'],r['candidate_id']))]}
    # Reject nonfinite values before persisting or reading query labels.
    json.dumps(bundle,allow_nan=False)
    return bundle


def evaluate(bundle,train_labels,query_labels):
    _label_arrays(bundle['train_ids'],train_labels);_label_arrays(bundle['query_ids'],query_labels)
    rows=[]
    for p in bundle['predictions']:
        y=query_labels[p['candidate_id']]['validation_mean']
        gaps=[abs(y-train_labels[c]['validation_mean']) for c in p['neighbor_ids']]
        row={**p,'true_validation_mean':y,'neighbor_mean_abs_difference':float(np.mean(gaps)),
            'neighbor_abs_differences':gaps,**{f'{m}_absolute_error':abs(y-p[f'{m}_mean']) for m in ('head','knn','gp')}}
        for t in (.006,.012): row[f'wrong_{t}']=float(np.mean(np.asarray(gaps)>t))
        for level in NORMAL_QUANTILES:
            lo,hi=p[f'interval{level}'];row[f'coverage{level}']=float(lo<=y<=hi);row[f'width{level}']=hi-lo
        rows.append(row)
    metrics={'main_neighborhood_metric':float(np.median([r['neighbor_mean_abs_difference'] for r in rows])),
        **{f'{m}_mae':float(np.mean([r[f'{m}_absolute_error'] for r in rows])) for m in ('head','knn','gp')}}
    for k in ('wrong_0.006','wrong_0.012','coverage90','coverage95','width90','width95'):
        metrics[k]=float(np.mean([r[k] for r in rows]))
    return {'metrics':metrics,'per_query':rows,'evaluation_only':True}


def tail(history,keys,n):
    rows=history[-n:]
    return {k:{'first':rows[0][k],'last':rows[-1][k],
        'first5_mean':float(np.mean([r[k] for r in rows[:5]])),
        'last5_mean':float(np.mean([r[k] for r in rows[-5:]])),
        'change':rows[-1][k]-rows[0][k],
        'slope_per_step':float(np.polyfit([r['step'] for r in rows],[r[k] for r in rows],1)[0])}
        for k in keys}


class Context:
    def __init__(self,config):
        self.config=config;self.output=ROOT/config['output'];self.old=ROOT/config['old_output']
        self.protocol=read(self.old/'protocol.json');self.partition=read(self.old/'partition.json')
        self.matrix=read(self.old/'task_matrix.json');self.selection=read(self.old/'selection/fold0.json')
        if digest(self.partition)!=self.protocol['partition_hash'] or digest(self.matrix)!=self.protocol['task_matrix_hash']:
            raise RuntimeError('old frozen split/matrix changed')
        for p,h in self.protocol['source_identity'].items():
            if file_hash(ROOT/p)!=h or file_hash(self.old/'source_snapshot'/p)!=h:
                raise RuntimeError(f'old numerical source changed: {p}')
        prep=ROOT/self.protocol['config']['prepared']
        all_candidates=read(prep/'candidates.json');self.base=read(prep/'base_encoding.json')
        if digest(all_candidates)!=self.protocol['prepared_manifest']['pool_hash'] or digest(self.base)!=self.protocol['prepared_manifest']['base_encoding_hash']:
            raise RuntimeError('candidate/encoding source changed')
        self.by_id={c['candidate_id']:c for c in all_candidates if c['candidate_id'] in set(self.partition['A_ids'])}
        if (len(self.by_id),len(self.partition['A_architectures']),len(self.partition['B_ids']),len(self.partition['B_architectures']))!=(179,96,157,96):
            raise RuntimeError('A/B identity counts differ')
        self.inner=self.partition['folds'][0]['inner']
        self.done={}
        for p in self.old.glob('tasks/*/attempt*/completed.json'):
            row=read(p)
            for name,h in row['artifact_hashes'].items():
                if file_hash(p.parent/name)!=h:raise RuntimeError(f'old artifact changed: {p.parent/name}')
            if row['task_id'] in self.done:raise RuntimeError('duplicate old success')
            self.done[row['task_id']]=(row,p.parent)

    def data(self,train_ids,query_ids):
        if not set(train_ids+query_ids)<=set(self.by_id):raise ValueError('outside A')
        labels=read_selected_labels(self.old/'A_validation.json',train_ids)
        return inputs([self.by_id[c] for c in train_ids],[self.by_id[c] for c in query_ids],self.base,labels),labels

    def old_task(self,method,seed):
        setting=self.selection['settings'][method]
        tid=f"fold0-inner-{setting['config_id']}-s{seed}"
        row,path=self.done[tid]
        s=load(path/'model.pt')
        if row['stage']!='inner' or row['family']!='fold0' or s['setting']!=setting:
            raise RuntimeError('wrong original inner task')
        if s['train_ids']!=self.inner['train_ids'] or s['query_ids']!=self.inner['query_ids']:
            raise RuntimeError('wrong original inner IDs')
        return tid,path,s

    def freeze(self):
        reuse=[]
        for method in ('regression','N2'):
            for seed in (0,1,2):
                tid,path,s=self.old_task(method,seed);d,_=self.data(s['train_ids'],s['query_ids'])
                model,opt=restore_mapping(s,d)
                with torch.no_grad():e,_=model(d['z'],d['hp'])
                if not torch.equal(e,s['train_embeddings']) or not torch.equal(e.double(),s['gp_train_inputs']):
                    raise RuntimeError('M100 coordinate mismatch')
                if {float(v['step']) for v in opt.state.values()}!={100.0}:
                    raise RuntimeError('M100 optimizer prefix differs')
                reuse.append({'method':method,'seed':seed,'old_task_id':tid,'checkpoint':str(path/'model.pt'),
                    'sha256':file_hash(path/'model.pt'),'scaler_hash':digest(s['standardization']),
                    'init_hash':s['initialization_hash'],'M100_embedding_bitwise_equal':True,'optimizer_step':100,
                    'resume_mode':'model_and_Adam_100_to_300','scheduler':None,'learning_rate':.001})
        source_paths=['analyse/fuzzy_metric_diagnose.py','analyse/fuzzy_metric_diagnose_saved.py','configs/fuzzy_metric_diagnosis_v1.json']
        frozen={'config':self.config,'old_protocol_hash':self.protocol['protocol_hash'],
            'old_files':{p:file_hash(self.old/p) for p in ['protocol.json','partition.json','task_matrix.json','selection/fold0.json','outer_summary.json','budget_stop.json','fit_attempts.jsonl','A_validation.json']},
            'sources':{p:file_hash(ROOT/p) for p in source_paths},'inner':self.inner,
            'outer_holdout_ids_excluded_from_new_comparisons':self.partition['folds'][0]['query_ids'],
            'settings':{m:self.selection['settings'][m] for m in ('regression','N2')},'reuse':reuse,
            'initialization_rule':'original model seed 0/1/2; fresh GP seed 0 per mapping snapshot',
            'data_use':'A development only; B labels not opened; no independent confirmation',
            'optional_trigger':'N2 M300/G100 mean training regression loss > regression AND mean holdout GP MAE >= regression',
            'optional_weight':'selected N2 lambda_f / 10; no other changes',
            'actual_step_limits':{'mapping':2700,'gp':1800},'scientific_trajectory_limits':{'mapping':9,'gp':18},
            'retry_limit_per_task':2,'execution':'serial CPU single thread; fail stops dispatch',
            'baseline_gp20_comparison_atol':1e-12,'mapping_prefix_tolerance':'bitwise equality'}
        write(self.output/'protocol.json',frozen,immutable=True)
        for p in source_paths:
            dest=self.output/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(ROOT/p,dest)
        return frozen


def journal(output,event):
    with (Path(output)/'attempts.jsonl').open('a') as s:
        s.write(json.dumps({**event,'time':time.time()},allow_nan=False)+'\n');s.flush();os.fsync(s.fileno())


def completed_task(path):
    if not (path/'completed.json').exists():return None
    row=read(path/'completed.json')
    for p,h in row['artifacts'].items():
        if file_hash(path/p)!=h:raise RuntimeError('new completed artifact changed')
    return row


def begin_task(ctx,tid,kind):
    allowed={f'{m}-s{s}-mapping' for m in ('regression','N2','N2_small') for s in (0,1,2)}
    allowed|={f'{m}-s{s}-M{n}-gp' for m in ('regression','N2','N2_small') for s in (0,1,2) for n in (100,300)}
    if tid not in allowed:raise ValueError('trajectory outside fixed diagnosis plan')
    if tid.startswith('N2_small') and not read(ctx.output/'optional_trigger.json')['execute']:
        raise RuntimeError('smaller weight condition was not triggered')
    path=ctx.output/'tasks'/tid;path.mkdir(parents=True,exist_ok=True)
    prior=sorted(path.glob('attempt*/started.json'))
    if prior:
        last=prior[-1].parent
        if (last/'failed.json').exists():
            # A human-reviewed repair record is needed before retry, never automatic fanout.
            if not (last/'repair_verified.json').exists():raise RuntimeError(f'repair proof required before retry: {tid}')
            if len(prior)>=3:raise RuntimeError(f'two failed retries exhausted: {tid}')
        elif not (last/'ended.json').exists():
            # Ordinary interruption resumes this attempt and its latest exact step.
            return path,last
    attempt=path/f'attempt{len(prior)+1:02d}';attempt.mkdir()
    write(attempt/'started.json',{'task_id':tid,'kind':kind,'time':time.time(),'pid':os.getpid()})
    journal(ctx.output,{'event':'start','task_id':tid,'kind':kind,'attempt':attempt.name})
    return path,attempt


def finish_task(path,attempt,artifacts,start):
    row={'artifacts':{str(p.relative_to(path)):file_hash(p) for p in artifacts},
        'elapsed_seconds_this_invocation':time.monotonic()-start,'time':time.time()}
    write(attempt/'ended.json',row);write(path/'completed.json',row,immutable=True)
    return row


def fail_task(ctx,path,attempt,start):
    write(attempt/'failed.json',{'traceback':traceback.format_exc(),'time':time.time(),
        'elapsed_seconds':time.monotonic()-start,'progress_steps':load(path/'progress.pt')['step'] if (path/'progress.pt').exists() else 0})
    journal(ctx.output,{'event':'failure','task_id':path.name,'attempt':attempt.name})


def mapping_trajectory(ctx,method,seed):
    tid=f'{method}-s{seed}-mapping';path=ctx.output/'tasks'/tid
    if completed_task(path):return path
    if costs(ctx)['mapping']['actual_completed_optimizer_steps']>=2700:raise RuntimeError('mapping update limit exhausted')
    path,attempt=begin_task(ctx,tid,'mapping');start=time.monotonic()
    try:
        source_method='N2' if method=='N2_small' else method
        old_tid,old_path,old=ctx.old_task(source_method,seed)
        data,labels=ctx.data(old['train_ids'],old['query_ids'])
        setting=copy.deepcopy(old['setting'])
        if method=='N2_small':setting['lambda_f']/=10
        if (path/'progress.pt').exists():
            progress=load(path/'progress.pt');model,opt=restore_mapping(progress['state'],data)
            history=progress['history'];step=progress['step'];charged=progress['new_steps'];state=progress['state']
        elif method=='N2_small':
            model=JointMetricModel(z_dim=data['z'].shape[1],seed=seed)
            opt=torch.optim.Adam(model.parameters(),lr=.001);history=[];step=0;charged=0
            if tensor_hash(model.state_dict())!=old['initialization_hash']:raise RuntimeError('small weight init differs')
        else:
            model,opt=restore_mapping(old,data);step=100;charged=0
            history=read(old_path/'predictions_before_reveal.json')['diagnostics']['loss_history']
            save(path/'M100.pt',mapping_state(model,opt,data,setting,seed,old))
        write(path/'source.json',{'old_task_id':old_tid,'old_checkpoint_hash':file_hash(old_path/'model.pt'),
            'train_ids':data['ids'],'query_ids':data['qids'],'setting':setting,'seed':seed,
            'old_mapping_steps_reused':100 if method!='N2_small' else 0,'scheduler':None},immutable=True)
        for step in range(step+1,301):
            # The intent plus committed checkpoint make an interrupted update auditable.
            write(path/'step_intent.json',{'step':step,'kind':'mapping','previous_new_steps':charged})
            history.append(mapping_step(model,opt,data,setting,step));charged+=1
            state=mapping_state(model,opt,data,setting,seed,old)
            if step in (100,300):save(path/f'M{step}.pt',state)
            save(path/'progress.pt',{'step':step,'new_steps':charged,'state':state,'history':history})
        write(path/'trajectory.json',{'history':history,'new_steps':charged,'final_losses':state['final_losses'],
            'method':method,'seed':seed,'training_labels_hash':digest(labels),'seconds':time.monotonic()-start})
        finish_task(path,attempt,[path/'M100.pt',path/'M300.pt',path/'trajectory.json',path/'source.json'],start)
        return path
    except Exception:
        fail_task(ctx,path,attempt,start);raise


def gp_trajectory(ctx,method,seed,mapping_path,mstep):
    tid=f'{method}-s{seed}-M{mstep}-gp';path=ctx.output/'tasks'/tid
    if completed_task(path):return path
    if costs(ctx)['gp']['actual_completed_optimizer_steps']>=1800:raise RuntimeError('GP update limit exhausted')
    path,attempt=begin_task(ctx,tid,'gp');start=time.monotonic()
    try:
        state=load(mapping_path/f'M{mstep}.pt');data,train_labels=ctx.data(state['train_ids'],state['query_ids'])
        model,_=restore_mapping(state,data);e=state['train_embeddings'];gp,opt=new_gp(data,e)
        initial={'parameters':gp_parameters(gp,data),'state_hash':tensor_hash(gp.state_dict()),'seed':0}
        write(path/'initial_gp.json',initial,immutable=True)
        if (path/'progress.pt').exists():
            progress=load(path/'progress.pt');gp.load_state_dict(progress['gp']);opt.load_state_dict(_state_cpu(progress['optimizer']))
            step=progress['step'];history=progress['history']
        else:step=0;history=[]
        def snapshot(step):
            save(path/f'G{step}.pt',{'gp':gp.state_dict(),'optimizer':opt.state_dict(),
                'train_inputs':e.double(),'train_targets':gp.train_targets,'step':step})
            bundle=predict(model,gp,data)
            write(path/f'G{step}_predictions_before_reveal.json',bundle,immutable=True)
            if mstep==100 and method!='N2_small' and step==20:
                _,op,old=ctx.old_task(method,seed)
                baseline=read(op/'predictions_before_reveal.json')
                exact=all(torch.equal(v,old['gp'][k]) for k,v in gp.state_dict().items())
                delta=max(abs(a['gp_mean']-b['mean']) for a,b in zip(bundle['predictions'],baseline['predictions']))
                same_neighbors=all(a['neighbor_ids']==b['neighbor_ids'] for a,b in zip(bundle['predictions'],baseline['predictions']))
                prefix_equal=all(abs(a['negative_marginal_log_likelihood']-b['negative_marginal_log_likelihood'])<=1e-12
                    for a,b in zip(history,baseline['diagnostics']['gp_loss_history']))
                proof={'gp_state_bitwise_equal':exact,'max_prediction_difference':delta,'neighbors_equal':same_neighbors,
                    'loss_prefix_equal':prefix_equal,'old_task':str(op),'old_fit_recharged':False}
                write(path/'baseline_reuse_check.json',proof,immutable=True)
                if not exact or delta>1e-12 or not same_neighbors or not prefix_equal:
                    raise RuntimeError('G20 reconstruction differs from original baseline')
        if step in (20,100):snapshot(step)
        for step in range(step+1,101):
            write(path/'step_intent.json',{'step':step,'kind':'gp','previous_new_steps':step-1})
            history.append(gp_step(gp,opt,step,data))
            if step in (20,100):snapshot(step)
            save(path/'progress.pt',{'step':step,'new_steps':step,'gp':gp.state_dict(),'optimizer':opt.state_dict(),'history':history})
        # Both GP snapshots and all three predictors exist before new query-label access.
        if not (path/'evaluation_access.json').exists():write(path/'evaluation_access.json',{'ids':data['qids'],'time':time.time(),
            'prediction_hashes':{f'G{s}':file_hash(path/f'G{s}_predictions_before_reveal.json') for s in (20,100)}},immutable=True)
        query_labels=read_selected_labels(ctx.old/'A_validation.json',data['qids'])
        for step in (20,100):
            bundle=read(path/f'G{step}_predictions_before_reveal.json')
            write(path/f'G{step}_evaluation.json',evaluate(bundle,train_labels,query_labels),immutable=True)
        write(path/'trajectory.json',{'history':history,'new_steps':100,'method':method,'seed':seed,'mapping_step':mstep,
            'mapping_hash':file_hash(mapping_path/f'M{mstep}.pt'),'final_parameters':gp_parameters(gp,data),'seconds':time.monotonic()-start})
        artifacts=[path/'trajectory.json',path/'initial_gp.json',path/'evaluation_access.json']
        artifacts += [path/f'G{s}{suffix}' for s in (20,100) for suffix in ('.pt','_predictions_before_reveal.json','_evaluation.json')]
        finish_task(path,attempt,artifacts,start);return path
    except Exception:
        fail_task(ctx,path,attempt,start);raise


def costs(ctx):
    starts=[]
    if (ctx.output/'attempts.jsonl').exists(): starts=[json.loads(l) for l in (ctx.output/'attempts.jsonl').read_text().splitlines()]
    result={'gnn_trainings':0,'old_mapping_starts':640,'old_gp_starts':557,'old_mapping_failures':83}
    for kind in ('mapping','gp'):
        events=[r for r in starts if r['event']=='start' and r['kind']==kind]
        paths=[p for p in (ctx.output/'tasks').glob('*') if p.name.endswith('-'+kind)]
        newsteps=sum(load(p/'progress.pt')['new_steps'] for p in paths if (p/'progress.pt').exists())
        result[kind]={'unique_scientific_trajectories':len({r['task_id'] for r in events}),'attempt_starts':len(events),
            'completed':sum((p/'completed.json').exists() for p in paths),'actual_completed_optimizer_steps':newsteps,
            'failures':sum(len(list(p.glob('attempt*/failed.json'))) for p in paths),
            'retries':sum(max(0,len(list(p.glob('attempt*/started.json')))-1) for p in paths),
            'seconds_successful_trajectory_sum':sum(read(p/'trajectory.json')['seconds'] for p in paths if (p/'completed.json').exists())}
    result['cumulative_mapping_attempts']=640+result['mapping']['attempt_starts']
    result['cumulative_gp_attempts']=557+result['gp']['attempt_starts']
    return result


def collect(ctx,methods):
    result={}
    for method in methods:
        rows=[]
        for seed in (0,1,2):
            mp=ctx.output/'tasks'/f'{method}-s{seed}-mapping'
            for condition,m,g in [('baseline',100,20),('long_gp',100,100),('long_mapping',300,100)]:
                gp=ctx.output/'tasks'/f'{method}-s{seed}-M{m}-gp'
                evaluation=read(gp/f'G{g}_evaluation.json')
                s=load(mp/f'M{m}.pt')
                rows.append({'seed':seed,'condition':condition,'mapping_steps':m,'gp_steps':g,
                    'train_losses':s['final_losses'],**evaluation['metrics']})
        grouped={}
        for condition in ('baseline','long_gp','long_mapping'):
            selected=[r for r in rows if r['condition']==condition]
            keys=[k for k,v in selected[0].items() if isinstance(v,(int,float)) and k not in ('seed','mapping_steps','gp_steps')]
            grouped[condition]={k:{'mean':float(np.mean([r[k] for r in selected])),
                'min':min(r[k] for r in selected),'max':max(r[k] for r in selected)} for k in keys}
            grouped[condition]['training_regression_loss']={'mean':float(np.mean([r['train_losses']['regression'] for r in selected]))}
        result[method]={'per_seed':rows,'summary':grouped}
    return result


def paired(result):
    rows=[]
    metrics=['main_neighborhood_metric','gp_mae','head_mae','knn_mae','wrong_0.006','wrong_0.012','coverage90','coverage95','width90','width95']
    for method,v in result.items():
        index={(r['seed'],r['condition']):r for r in v['per_seed']}
        for before,after in [('baseline','long_gp'),('long_gp','long_mapping')]:
            for seed in (0,1,2):
                rows.append({'comparison':f'{after} minus {before}','method':method,'seed':seed,
                    **{k:index[seed,after][k]-index[seed,before][k] for k in metrics}})
    for method in result:
        if method=='regression':continue
        for condition in ('baseline','long_gp','long_mapping'):
            for seed in (0,1,2):
                a=next(r for r in result[method]['per_seed'] if r['seed']==seed and r['condition']==condition)
                b=next(r for r in result['regression']['per_seed'] if r['seed']==seed and r['condition']==condition)
                rows.append({'comparison':f'{method} minus regression','condition':condition,'seed':seed,
                    **{k:a[k]-b[k] for k in metrics}})
    return rows


def should_reduce_weight(n2_loss,regression_loss,n2_mae,regression_mae):
    return n2_loss>regression_loss and n2_mae>=regression_mae


def run(ctx):
    ctx.freeze()
    audit=read(ctx.output/'saved_diagnostics/completed.json')
    for p,h in audit['artifacts'].items():
        if file_hash(ctx.output/'saved_diagnostics'/p)!=h:raise RuntimeError('saved diagnostics changed before fitting')
    lock=(ctx.output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=costs(ctx);started=time.monotonic()
    try:
        # First real trajectory and both associated GPs finish before another seed starts.
        for method in ('regression','N2'):
            for seed in (0,1,2):
                mp=mapping_trajectory(ctx,method,seed)
                for m in (100,300):gp_trajectory(ctx,method,seed,mp,m)
                print(json.dumps({'completed':f'{method}-s{seed}','costs':costs(ctx)}),flush=True)
        main=collect(ctx,['regression','N2'])
        a=main['N2']['summary']['long_mapping'];b=main['regression']['summary']['long_mapping']
        trigger={'N2_training_regression_loss':a['training_regression_loss']['mean'],
            'regression_training_regression_loss':b['training_regression_loss']['mean'],
            'N2_gp_mae':a['gp_mae']['mean'],'regression_gp_mae':b['gp_mae']['mean']}
        trigger['execute']=should_reduce_weight(trigger['N2_training_regression_loss'],trigger['regression_training_regression_loss'],trigger['N2_gp_mae'],trigger['regression_gp_mae'])
        trigger['rule']='strictly worse training regression loss AND GP MAE not lower; means of all three seeds'
        write(ctx.output/'optional_trigger.json',trigger,immutable=True)
        methods=['regression','N2']
        if trigger['execute']:
            methods.append('N2_small')
            for seed in (0,1,2):
                mp=mapping_trajectory(ctx,'N2_small',seed)
                for m in (100,300):gp_trajectory(ctx,'N2_small',seed,mp,m)
                print(json.dumps({'completed':f'N2_small-s{seed}','costs':costs(ctx)}),flush=True)
        results=collect(ctx,methods)
        write(ctx.output/'comparison.json',{'methods':results,'paired_per_seed':paired(results),
            'optional_trigger':trigger,'interpretation':'one fixed inner split; three model seeds; development diagnosis only'},immutable=True)
        write(ctx.output/'costs.json',costs(ctx))
        write(ctx.output/f'invocation{len(list(ctx.output.glob("invocation*.json")))+1:02d}.json',
            {'before':before,'after':costs(ctx),'elapsed_seconds':time.monotonic()-started,'success':True})
    finally:
        write(ctx.output/'costs.json',costs(ctx));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/fuzzy_metric_diagnosis_v1.json')
    parser.add_argument('--audit',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    config=read(ROOT/args.config)
    if config['gnn_training_budget']!=0 or config['workers']!=1 or config['threads']!=1:
        raise ValueError('this protocol requires GNN=0 and serial single-thread CPU')
    ctx=Context(config)
    if args.audit:
        ctx.freeze()
        from analyse.fuzzy_metric_diagnose_saved import audit_saved
        audit_saved(ctx)
    if args.run:run(ctx)
    if not (args.audit or args.run):parser.error('choose --audit or --run')


if __name__=='__main__':main()
