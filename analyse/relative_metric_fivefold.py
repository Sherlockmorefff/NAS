"""Unchanged R1 core on original five outer folds, with isolated fold workers."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from surrogate.fuzzy_metric import JointMetricModel
from surrogate.relative_metric import (validate_batch,build_plan,tensor_plan,objective,make_branches,fresh_adam,probe,order_terms)
from analyse.relative_metric_pilot import (CONFIG as PILOT_CONFIG,read,write,save,load,freeze_copy,file_hash,digest,tensor_hash,
    feature_part,actual_objects,checkpoint,inference,infer_neighbors,utc)
PILOT=ROOT/'results/posthoc/relative-metric-r1-pilot-20260911-v1'
OLD=ROOT/'results/posthoc/fuzzy-metric-stage2-20260910-v1'
CLOSURE=ROOT/'results/posthoc/fuzzy-metric-fivefold-closure-20260910-v1'
DEFAULT=ROOT/'results/posthoc/relative-metric-r1-fivefold-20260911-v2'
CONFIG={**copy.deepcopy(PILOT_CONFIG),'version':'relative-r1-fivefold-fixed-20260911-v1','folds':[0,1,2,3,4],
        'actual_update_limit':6300,'effective_update_target':6000}
SOURCES=['analyse/relative_metric_fivefold.py','analyse/relative_metric_pilot.py','surrogate/relative_metric.py','surrogate/fuzzy_metric.py']


def fold_check(fold,candidates,aids):
    ids,qids=fold['train_ids'],fold['query_ids']
    if len(set(ids+qids))!=len(ids+qids) or set(ids+qids)!=set(aids):raise ValueError('fold does not partition A')
    ta,qa=[{candidates[c]['architecture_fingerprint'] for c in seq} for seq in (ids,qids)]
    if ta&qa:raise ValueError('train/query architecture overlap')
    return dict(family=fold['family'],train_count=len(ids),query_count=len(qids),train_architectures=len(ta),query_architectures=len(qa))


def assert_task_data(task,data):
    if task['train_ids']!=data['train']['ids'] or task['query_ids']!=data['query']['ids'] or task['seed']!=data['seed'] or task['fold']!=data['fold']:
        raise ValueError('cross-fold/seed input identity mismatch')
    y,se,target=validate_batch(data['train'],data['query'],data['labels'],data['scaler'])
    if not torch.equal(target,data['target']):raise ValueError('cached target differs')
    return y,se,target


def probe_groups(model,train,target,sample):
    """Add group weight/loss diagnostics without changing the core objective."""
    result=probe(model,train,target,sample)
    with torch.no_grad():
        e,_=model(train['z'],train['hp']);total,hinge,valid=order_terms(e,sample)
        means=(sample[2]*hinge*valid).sum(-1)/sample[3].clamp_min(1)
        groups={}
        for g,name in enumerate(('global','local10')):
            mask=valid[:,g,:];w=sample[2][:,g,:][mask].double().numpy()
            loss=.5*float(means[:,g].mean())
            groups[name]=dict(w_quantiles=np.quantile(w,[0,.25,.5,.75,1]).tolist() if len(w) else [],
                w_mean=float(w.mean()) if len(w) else None,triplets=int(mask.sum()),empty_centers=int((sample[3][:,g]==0).sum()),
                half_weighted_loss=loss,loss_fraction=loss/float(total) if float(total)>0 else None,
                loss_fraction_NA_reason=None if float(total)>0 else 'zero total order loss',
                active_fraction=float(((hinge[:,g,:]>0)&mask).sum())/int(mask.sum()) if int(mask.sum()) else None)
        if not math.isclose(sum(v['half_weighted_loss'] for v in groups.values()),float(total),rel_tol=2e-6,abs_tol=1e-7):raise ValueError('group loss decomposition differs')
        result['groups']=groups;result['loss_fraction_is_not_gradient_fraction']=True
    return result


def audit(output):
    # This preparation process may access all A labels; fold workers cannot.
    from analyse.fuzzy_metric_fivefold_closure import completed_state
    old,part,matrix,boundary=[read(OLD/n) for n in ['protocol.json','partition.json','task_matrix.json','label_boundary.json']]
    previous=read(PILOT/'protocol.json')
    if previous['config']!=PILOT_CONFIG:raise ValueError('pilot config/source mismatch')
    for path,h in previous['source_hashes'].items():
        if file_hash(ROOT/path)!=h:raise ValueError('verified R1 source changed: '+path)
    if digest(part)!=old['partition_hash'] or digest(matrix)!=old['task_matrix_hash']:raise ValueError('original partition/task matrix changed')
    prep=ROOT/old['config']['prepared'];pool=read(prep/'candidates.json');base=read(prep/'base_encoding.json')
    if digest(pool)!=old['prepared_manifest']['pool_hash'] or digest(base)!=old['prepared_manifest']['base_encoding_hash']:raise ValueError('original unlabeled inputs changed')
    aids=part['A_ids'];candidates={c['candidate_id']:c for c in pool if c['candidate_id'] in aids}
    assert len(candidates)==179 and len({c['architecture_fingerprint'] for c in candidates.values()})==96
    folds=part['folds'];info=[fold_check(f,candidates,aids) for f in folds]
    assert [r['train_count'] for r in info]==[141,143,143,145,144] and [r['query_count'] for r in info]==[38,36,36,34,35]
    assert Counter(c for f in folds for c in f['query_ids'])==Counter(aids)
    labels=read(OLD/'A_validation.json');assert set(labels)==set(aids) and digest(labels)==boundary['exported_hash']
    inventory=read(CLOSURE/'state_inventory.json');headrefs=read(CLOSURE/'old_prediction_reference.json')
    inventory={r['state_id']:r for r in inventory};headrefs={(r['state_id'],r['candidate_id']):r for r in headrefs}
    input_sources={str(p):file_hash(p) for p in [PILOT/'protocol.json',OLD/'protocol.json',OLD/'partition.json',OLD/'task_matrix.json',OLD/'label_boundary.json',OLD/'A_validation.json',CLOSURE/'state_inventory.json',CLOSURE/'old_prediction_reference.json',prep/'candidates.json',prep/'base_encoding.json']}
    for src in input_sources:freeze_copy(Path(src),output/'input_originals'/(Path(src).parent.name+'_'+Path(src).name))
    write(output/'candidate_identity.json',list(candidates.values()));write(output/'folds.json',folds)
    tasks=[];checks=[];missing=[];plans=[];data_hashes={};query_hashes={}
    for fi,fold in enumerate(folds):
        ids,qids=fold['train_ids'],fold['query_ids'];train,query=[[candidates[c] for c in seq] for seq in (ids,qids)]
        folder=output/'inputs'/f'fold{fi}';folder.mkdir(parents=True,exist_ok=True)
        tl={c:labels[c] for c in ids};ql={c:labels[c] for c in qids}
        write(folder/'train_labels.json',tl);write(output/'evaluation_labels'/f'fold{fi}.json',ql)
        data_hashes[str((folder/'train_labels.json').relative_to(output))]=file_hash(folder/'train_labels.json')
        query_hashes[f'evaluation_labels/fold{fi}.json']=file_hash(output/'evaluation_labels'/f'fold{fi}.json')
        selection=read(OLD/'selection'/f'fold{fi}.json')
        input_sources[str(OLD/'selection'/f'fold{fi}.json')]=file_hash(OLD/'selection'/f'fold{fi}.json')
        for seed in [0,1,2]:
            sid=f'fold{fi}-refit-regression-s{seed}'
            try:
                matches=[t for t in matrix if t['task_id']==sid]
                if len(matches)!=1:raise ValueError('not exactly one original matrix task')
                task=copy.deepcopy(matches[0]);task['setting']=selection['settings']['regression']
                assert task['stage']=='refit' and task['family']==f'fold{fi}' and task['seed']==seed and task['method']=='regression'
                assert task['train_ids']==ids and task['query_ids']==qids and task['setting']['lambda_f']==0
                identity=digest({'task':task,'protocol':old['protocol_hash'],'A_labels':boundary['exported_hash']})
                rec=completed_state(task,identity)
                prev=inventory[sid]
                assert rec['completion_identity']==prev['completion_identity'] and rec['model_path']==prev['model_path'] and rec['artifact_hashes']==prev['artifact_hashes']
                path=Path(rec['model_path']);s=load(path)
                assert s['train_ids']==ids and s['query_ids']==qids and s['model_seed']==seed and s['setting']==task['setting']
                assert s['common_config']==old['config'] and s['common_config']['representation_steps']==100
                assert {float(v['step']) for v in s['optimizer']['state'].values()}=={100.}
                assert s['candidate_identity_hash']==digest({'train':train,'query':query})
                sc=s['standardization'];assert sc['fit_ids']==ids and sc['hp_feature_order']==['log10_lr','dropout','log2_hidden_dim','log10_l2']
                tr,qu=feature_part(train,base,sc),feature_part(query,base,sc);y,se,target=validate_batch(tr,qu,tl,sc)
                yt=torch.tensor((y-sc['y_mean'])/sc['y_scale'],dtype=torch.float64);noise=torch.tensor((se/sc['y_scale'])**2,dtype=torch.float64)
                td=float((yt-s['gp_train_targets']).abs().max());nd=float((noise-s['gp_fixed_noise_normalized']).abs().max())
                assert td<=1e-12 and nd<=1e-12
                m=JointMetricModel(12,seed=seed);assert tensor_hash(m.state_dict())==s['initialization_hash'];m.load_state_dict(s['model'],strict=True)
                with torch.no_grad():e,pr=m(tr['z'],tr['hp']);qe,qp=m(qu['z'],qu['hp'])
                assert torch.equal(e,s['train_embeddings']) and torch.equal(e.double(),s['gp_train_inputs'])
                oldpred=read(path.parent/'predictions_before_reveal.json');assert oldpred['train_ids']==ids and oldpred['query_ids']==qids
                d=torch.cdist(qe.double(),e.double()).numpy();maxd=0.;headmax=0.;headmissing=[]
                for qi,row in enumerate(oldpred['predictions']):
                    assert row['candidate_id']==qids[qi]
                    near=sorted(range(len(ids)),key=lambda j:(float(d[qi,j]),ids[j]))[:5]
                    assert [ids[j] for j in near]==row['neighbor_ids'];maxd=max(maxd,float(np.max(np.abs(d[qi,near]-np.asarray(row['neighbor_distances'])))))
                    h=headrefs.get((sid,qids[qi]),{}).get('old_head_prediction')
                    if h is None:headmissing.append(qids[qi])
                    else:headmax=max(headmax,abs(float(qp[qi].double()*sc['y_scale']+sc['y_mean'])-h))
                assert maxd==0. and headmax<=1e-12
                r0,r1=make_branches(m);objects=actual_objects(r0,fresh_adam(r0));assert not fresh_adam(r1).state
                data=dict(train=tr,query=qu,labels=tl,scaler=sc,target=target,fold=fi,seed=seed,candidate_identity_hash=s['candidate_identity_hash'])
                ip=folder/f's{seed}.pt';wp=output/'warmstarts'/f'fold{fi}-s{seed}.pt'
                if not ip.exists():save(ip,data)
                if not wp.exists():save(wp,{k:copy.deepcopy(s[k]) for k in ['model','standardization','train_ids','query_ids','model_seed','initialization_hash','candidate_identity_hash']})
                plan=build_plan(y,se,ids,sc['y_mean'],sc['y_scale'],seed)
                casts=[]
                for k in range(200):
                    sample=tensor_plan(plan,k);casts.append([float(np.max(np.abs(sample[1].double().numpy()-plan['margins'][k]))),float(np.max(np.abs(sample[2].double().numpy()-plan['weights'][k])))])
                pp=output/'plans'/f'fold{fi}-s{seed}.npz';pp.parent.mkdir(exist_ok=True)
                if not pp.exists():np.savez_compressed(pp,**plan)
                else:
                    with np.load(pp) as existing:assert all(np.array_equal(existing[k],v) for k,v in plan.items())
                probe_path=output/'fixed_probe_inputs'/f'fold{fi}-s{seed}.json'
                write(probe_path,dict(ids=ids,y=y.tolist(),se=se.tolist(),y_mean=sc['y_mean'],y_scale=sc['y_scale'],refs=plan['refs'][0].tolist(),margins=plan['margins'][0].tolist(),weights=plan['weights'][0].tolist(),counts=plan['counts'][0].tolist(),plan_sha256=file_hash(pp)))
                for src in [ip,wp,pp,probe_path]:data_hashes[str(src.relative_to(output))]=file_hash(src)
                checks.append(dict(fold=fi,seed=seed,source=rec,parameter_hash=tensor_hash(s['model']),original_Adam_present=True,original_Adam_steps=100,Adam_action='reset both branches',branches_independent=True,
                    restored_train_embedding_maxdiff=0.,old_neighbor_distance_maxdiff=maxd,old_head_maxdiff=headmax,old_head_missing_ids=headmissing,target_source_maxdiff=td,SE2_source_maxdiff=nd,standardization=sc,
                    target_cast_maxdiff=float((yt-target.double()).abs().max()),effective_objects=objects))
                plans.append(dict(fold=fi,seed=seed,random_seed=20260911+seed,path=str(pp.relative_to(output)),sha256=file_hash(pp),shape=list(plan['refs'].shape),empty_groups=int((plan['counts']==0).sum()),fewer8_groups=int((plan['counts']<8).sum()),margin_cast_maxdiff=max(x[0] for x in casts),weight_cast_maxdiff=max(x[1] for x in casts)))
                for branch in ['R0','R1']:
                    tasks.append(dict(task_id=f'fold{fi}-s{seed}-{branch}',fold=fi,seed=seed,branch=branch,train_ids=ids,query_ids=qids,
                        input=str(ip.relative_to(output)),warmstart=str(wp.relative_to(output)),plan=str(pp.relative_to(output)),probe_input=str(probe_path.relative_to(output))))
                for f in [path,path.parent/'completed.json',path.parent/'started.json',path.parent/'predictions_before_reveal.json']:
                    input_sources[str(f)]=file_hash(f)
                    if f.suffix=='.json':freeze_copy(f,output/'input_originals'/'state_records'/sid/f.name)
            except (ValueError,KeyError,AssertionError,FileNotFoundError) as exc:
                missing.append(dict(fold=fi,seed=seed,original_task=sid,error=repr(exc),traceback=traceback.format_exc()))
    write(output/'missing_before_evaluation.json',missing);write(output/'task_matrix.json',tasks);write(output/'warmstart_checks.json',checks);write(output/'plan_checks.json',plans)
    write(output/'input_sources.json',input_sources);write(output/'effective_config.json',dict(config=CONFIG,objects=[r['effective_objects'] for r in checks]))
    for rel in SOURCES:freeze_copy(ROOT/rel,output/'source_snapshot'/rel)
    protocol=dict(config=CONFIG,A_ids=aids,folds=folds,fold_info=info,task_matrix_hash=digest(tasks),expected_trajectories=30,available_trajectories=len(tasks),missing=missing,
        pilot_protocol_sha256=file_hash(PILOT/'protocol.json'),original_partition_sha256=file_hash(OLD/'partition.json'),input_hashes=data_hashes,query_label_hashes=query_hashes,
        source_hashes={rel:file_hash(ROOT/rel) for rel in SOURCES},original_sources=input_sources,
        original_prompt_sha256=file_hash(output/'input_originals/Codex_Server_R1_Fivefold_Validation_Prompt.md'),
        interpretation='development replication; report exact direction/magnitude/fold counts; no new numerical pass gate; no GP/search/other experiments',
        aggregation='holdout per seed179 unique OOF candidates then mean3 seeds; train716 fold/candidate rows per seed separately; pair common fold/candidate IDs; architecture-equal auxiliary')
    protocol['identity']=digest(protocol);write(output/'protocol.json',protocol)
    write(output/'audit_complete.json',dict(protocol_identity=protocol['identity'],warmstarts=len(checks),missing=missing,updates=0))
    print(json.dumps({'audit':'complete','warmstarts':len(checks),'available_trajectories':len(tasks),'missing':missing,'folds':info}),flush=True)


def verify_protocol(output,fold=None):
    p=read(output/'protocol.json');identity=p.pop('identity');assert digest(p)==identity;p['identity']=identity
    if p['config']!=CONFIG:raise ValueError('fivefold configuration conflict')
    for rel,h in p['source_hashes'].items():
        if file_hash(ROOT/rel)!=h:raise ValueError('frozen training source conflict: '+rel)
    tasks=read(output/'task_matrix.json');assert digest(tasks)==p['task_matrix_hash']
    chosen=tasks if fold is None else [t for t in tasks if t['fold']==fold]
    paths=set(p['input_hashes']) if fold is None else {t[k] for t in chosen for k in ['input','warmstart','plan','probe_input']}
    for rel in paths:
        if file_hash(output/rel)!=p['input_hashes'][rel]:raise ValueError('frozen input conflict: '+rel)
    return p,chosen


def costs(output):
    intentions={int(p.stem) for p in (output/'intents').glob('*.json')};receipts={int(p.stem) for p in (output/'receipts').glob('*.json')}
    committed=[];trajectories=[]
    for path in sorted((output/'trajectories').glob('*/latest.pt')):
        s=load(path);assert s['step']==s['plan_position']==len(s['history'])==len(s['committed_global_ids'])
        committed+=s['committed_global_ids'];trajectories.append(dict(task=path.parent.name,effective_steps=s['step']))
    assert len(set(committed))==len(committed) and set(committed)<=intentions and receipts<=intentions
    confirmed=set(committed)|receipts;unknown=intentions-confirmed
    return dict(effective_updates=len(committed),confirmed_actual_updates=len(confirmed),discarded_or_repeated_updates=len(receipts-set(committed)),
        uncertain_reserved_updates=len(unknown),uncertain_global_ids=sorted(unknown),charged_upper_bound=len(intentions),actual_limit=6300,
        repair_allowance_used=len(intentions)-len(committed),trajectories=trajectories,GNN_training=0,VAE_training=0,GP_training=0,synthetic_updates=0)


def guard_fold_files(output,fold,tasks):
    """Runtime read boundary: worker cannot open other fold inputs or query labels."""
    allowed={str((output/t[k]).resolve()) for t in tasks for k in ['input','warmstart','plan','probe_input']}
    allowed|={str((output/n).resolve()) for n in ['protocol.json','task_matrix.json','checks_complete.json']}
    prefixes=[str((output/'trajectories'/t['task_id']).resolve())+'/' for t in tasks]
    prefixes += [str((output/n).resolve())+'/' for n in ['intents','receipts','failures']]
    result_root=str((ROOT/'results').resolve())+'/'
    def audit_hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=str(Path(os.fsdecode(args[0])).resolve())
        # Root lock/write evidence is permitted; label/evaluation/other-fold reads are not.
        if path.startswith(result_root) and path not in allowed and not any(path==x[:-1] or path.startswith(x) for x in prefixes):
            if path not in [str(output/'worker.lock')]:raise PermissionError('fold worker forbidden result file: '+path)
    sys.addaudithook(audit_hook)


def worker(output,fold):
    protocol,tasks=verify_protocol(output,fold);check=read(output/'checks_complete.json')
    assert check['passed'] and check['protocol_identity']==protocol['identity']
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gid=len(list((output/'intents').glob('*.json')))
        guard_fold_files(output,fold,tasks)
        for seed in [0,1,2]:
            selected=[t for t in tasks if t['seed']==seed]
            if not selected:continue
            data=load(output/selected[0]['input']);y,se,target=assert_task_data(selected[0],data)
            warm=load(output/selected[0]['warmstart']);warm_hash=tensor_hash(warm['model'])
            base=JointMetricModel(12,seed=seed);base.load_state_dict(warm['model'],strict=True);branches=make_branches(base)
            planpath=output/selected[0]['plan'];plan=dict(np.load(planpath));assert plan['ids'].tolist()==data['train']['ids']
            samples=[tensor_plan(plan,k) for k in range(200)]
            ih=tensor_hash({'z':data['train']['z'],'hp':data['train']['hp'],'qz':data['query']['z'],'qhp':data['query']['hp'],'target':target})
            for task,m in zip(sorted(selected,key=lambda t:t['branch']),branches):
                branch=task['branch'];folder=output/'trajectories'/task['task_id'];folder.mkdir(parents=True,exist_ok=True)
                identity=digest(dict(protocol=protocol['identity'],task=task,warmstart=warm_hash,plan=file_hash(planpath)))
                if (folder/'completed.json').exists():
                    done=read(folder/'completed.json');assert done['identity']==identity and done['effective_steps']==200
                    for name,h in done['artifacts'].items():assert file_hash(folder/name)==h
                    continue
                opt=fresh_adam(m);write(folder/'actual_initial_objects.json',actual_objects(m,opt))
                if (folder/'latest.pt').exists():
                    state=load(folder/'latest.pt');assert state['identity']==identity and state['plan_sha256']==file_hash(planpath) and state['input_hash']==ih
                    m.load_state_dict(state['model']);opt.load_state_dict(copy.deepcopy(state['optimizer']))
                    step,history,probes,committed=state['step'],state['history'],state['probes'],state['committed_global_ids']
                    assert step==state['plan_position']==len(committed)
                    assert {float(v['step']) for v in opt.state.values()}==({float(step)} if step else set())
                    torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng'])
                else:
                    step,history,committed=0,[],[];probes={'0':probe_groups(m,data['train'],target,samples[0])}
                    state=checkpoint(m,opt,step,history,probes,committed,identity,file_hash(planpath),ih);save(folder/'latest.pt',state)
                def node(k):
                    freeze_copy(folder/'latest.pt',folder/f'add{k:03d}.pt')
                    write(folder/f'inference_add{k:03d}.json',{**inference(m,data,k,seed,branch),'fold':fold})
                if step in CONFIG['probe_steps']:node(step)
                while step<200:
                    k=step+1;sample=samples[step];before={n:p.detach().clone() for n,p in m.named_parameters()}
                    m.train();opt.zero_grad(set_to_none=True);e,pred=m(data['train']['z'],data['train']['hp'])
                    values=objective(e,pred,target,sample,CONFIG['lambda_r'][branch]);numbers={n:float(v.detach()) for n,v in values.items()}
                    assert math.isclose(numbers['total'],numbers['regression']+CONFIG['lambda_r'][branch]*numbers['order']+.001*numbers['size'],rel_tol=2e-6,abs_tol=2e-7)
                    if not all(math.isfinite(v) for v in numbers.values()):raise RuntimeError('nonfinite loss')
                    values['total'].backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise RuntimeError('missing/nonfinite gradient')
                    norm=float(torch.cat([p.grad.reshape(-1) for p in m.parameters()]).norm())
                    if gid>=6300:raise RuntimeError('actual update budget exhausted')
                    gid+=1
                    write(output/'intents'/f'{gid:06d}.json',dict(global_id=gid,task=task['task_id'],additional_step=k,previous_commit=committed[-1] if committed else None,protocol_identity=protocol['identity'],time=utc()))
                    opt.step()
                    write(output/'receipts'/f'{gid:06d}.json',dict(global_id=gid,task=task['task_id'],additional_step=k,optimizer_step_returned=True))
                    if any(not torch.isfinite(p).all() for p in m.parameters()):raise RuntimeError('nonfinite parameter')
                    changes={n:float((p.detach()-before[n]).abs().max()) for n,p in m.named_parameters()}
                    assert max(changes.values())>0 and max(changes.values())<=.05
                    assert {float(v['step']) for v in opt.state.values()}=={float(k)}
                    assert tensor_hash({'z':data['train']['z'],'hp':data['train']['hp'],'qz':data['query']['z'],'qhp':data['query']['hp'],'target':target})==ih and tensor_hash(warm['model'])==warm_hash
                    row=dict(step=k,global_id=gid,plan_position_used=step,lambda_r=CONFIG['lambda_r'][branch],lambda_e=.001,gradient_norm=norm,**numbers)
                    if k==1:
                        with torch.no_grad():ae,ap=m(data['train']['z'],data['train']['hp']);av=objective(ae,ap,target,sample,CONFIG['lambda_r'][branch])
                        row['after_same_sample']={n:float(v) for n,v in av.items()};row['parameter_max_changes']=changes;row['cached_inputs_unchanged']=True
                    history.append(row);committed.append(gid);step=k
                    if step in CONFIG['probe_steps']:probes[str(step)]=probe_groups(m,data['train'],target,samples[0])
                    state=checkpoint(m,opt,step,history,probes,committed,identity,file_hash(planpath),ih);save(folder/'latest.pt',state)
                    if step in CONFIG['probe_steps']:node(step)
                bundle=read(folder/'inference_add200.json');write(folder/'neighbors_before_labels.json',infer_neighbors(bundle))
                write(folder/'loss_history.json',history);write(folder/'probes.json',probes)
                write(folder/'completed.json',dict(identity=identity,protocol_identity=protocol['identity'],fold=fold,seed=seed,branch=branch,effective_steps=200,endpoint=300,
                    completed_at=utc(),worker_pid=os.getpid(),query_label_reads=0,cross_fold_label_reads=0,
                    artifacts={p.name:file_hash(p) for p in folder.iterdir() if p.is_file() and p.name!='completed.json'}))
                print(json.dumps({'completed':task['task_id'],'effective_steps':200,'charged_updates':gid}),flush=True)
        assert not any(k in sys.modules for k in ['gpytorch','botorch','torch_geometric'])


def run(output):
    protocol,tasks=verify_protocol(output);before=costs(output);started=time.monotonic()
    if before['repair_allowance_used']>300:raise RuntimeError('repair allowance exhausted')
    invocation=len(list((output/'invocations').glob('run*.json')))+1
    for fold in range(5):
        log=output/'logs'/f'run{invocation:02d}-fold{fold}.log'
        command=[sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--run','--worker',str(fold)]
        with log.open('w') as f:
            child=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,text=True)
        if child.returncode:raise RuntimeError(f'fold{fold} worker failed; inspect {log}')
        print(json.dumps({'fold':fold,'status':'completed available tasks','cost':costs(output)['charged_upper_bound']}),flush=True)
    after=costs(output);assert after['effective_updates']==len(tasks)*200
    assert len(list((output/'trajectories').glob('*/completed.json')))==len(tasks)
    sealed={str(p.relative_to(output)):file_hash(p) for p in (output/'trajectories').rglob('*') if p.is_file()}
    if not (output/'all_inference_sealed.json').exists():
        write(output/'all_inference_sealed.json',dict(protocol_identity=protocol['identity'],sealed_at=utc(),available_trajectories=len(tasks),full_fivefold=len(tasks)==30,
            artifacts=sealed,orchestrator_pid=os.getpid(),cost=after))
    else:assert read(output/'all_inference_sealed.json')['artifacts']==sealed
    receipt=dict(time=utc(),before=before,after=after,actual_updates_this_invocation=after['confirmed_actual_updates']-before['confirmed_actual_updates'],elapsed_seconds=time.monotonic()-started)
    write(output/'invocations'/f'run{invocation:02d}.json',receipt);print(json.dumps(receipt))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--config',type=Path)
    p.add_argument('--worker',type=int,choices=range(5),help=argparse.SUPPRESS)
    mode=p.add_mutually_exclusive_group(required=True)
    for k in ['audit','check','run','report']:mode.add_argument('--'+k,action='store_true')
    args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.config and read(args.config)!=CONFIG:raise ValueError('CLI configuration conflicts with frozen fivefold protocol')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    try:
        if args.audit:audit(out)
        elif args.check:
            protocol,_=verify_protocol(out);cmd=[sys.executable,'-m','pytest','-q','tests/test_relative_metric_fivefold.py']
            checked=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,env={**os.environ,'R1_FIVEFOLD_OUTPUT':str(out)})
            (out/'logs/checks.log').write_text(checked.stdout+checked.stderr)
            if checked.returncode:raise RuntimeError(checked.stdout+checked.stderr)
            write(out/'checks_complete.json',dict(protocol_identity=protocol['identity'],passed=True,command=cmd,log_sha256=file_hash(out/'logs/checks.log'),optimizer_updates=0));print(checked.stdout)
        elif args.run:
            if args.worker is not None:worker(out,args.worker)
            else:
                with (out/'orchestrator.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(out)
        else:subprocess.run([sys.executable,str(ROOT/'analyse/relative_metric_fivefold_report.py'),str(out)],check=True)
    except Exception:
        if args.worker is None:
            folder=out/'failures';folder.mkdir(exist_ok=True);n=len(list(folder.glob('*.json')))+1
            write(folder/f'{n:03d}.json',dict(time=utc(),error=traceback.format_exc(),cost=costs(out)))
        raise
if __name__=='__main__':main()
