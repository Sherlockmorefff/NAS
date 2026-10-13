"""Fixed R1 pilot. Separate audit/check, training and saved-artifact reporting processes."""
from __future__ import annotations
import argparse
import copy
import datetime
import difflib
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from surrogate.fuzzy_metric import JointMetricModel
from surrogate.relative_metric import (ADAM, validate_batch, build_plan, tensor_plan, objective,
    make_branches, fresh_adam, probe, legal_pairs)
OLD = ROOT/'results/posthoc/fuzzy-metric-stage2-diagnosis-20260910-v1'
SUP = ROOT/'results/posthoc/fuzzy-metric-stage2-readonly-supplement-20260910-v1'
PREP = ROOT/'results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2'
DEFAULT = ROOT/'results/posthoc/relative-metric-r1-pilot-20260911-v1'
CONFIG = dict(version='relative-r1-fixed-20260911-v1',model_seeds=[0,1,2],branches=['R0','R1'],
    additional_steps=200,warmstart_step=100,lambda_r={'R0':0.,'R1':.01},lambda_e=.001,
    optimizer={**ADAM,'betas':list(ADAM['betas'])},device='cpu',threads=1,dtype='float32',
    plan_dtype='float64',sampling_seed_base=20260911,groups=['global','local10'],samples_per_group=8,
    global_group_weight=.5,local_group_weight=.5,performance_tie_tolerance=1e-12,margin_cap=1.,
    probe_steps=[0,1,50,100,200],actual_update_limit=1260,effective_update_target=1200,
    scheduler=None,early_stopping=False,gradient_clip=None,embedding_normalization=None,
    GNN=0,VAE=0,GP=0,optimizer_reset=True)
TRAIN_SOURCES=['surrogate/fuzzy_metric.py','surrogate/relative_metric.py','analyse/relative_metric_pilot.py']


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_hash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,v,immutable=True):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n'
    if immutable and p.exists():
        if p.read_text()!=content: raise RuntimeError(f'immutable artifact differs: {p}')
        return
    temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('w') as f: f.write(content); f.flush(); os.fsync(f.fileno())
    os.replace(temp,p)
    fd=os.open(p.parent,os.O_DIRECTORY); os.fsync(fd); os.close(fd)
def save(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); temp=p.with_suffix('.tmp')
    with temp.open('wb') as f: torch.save(v,f); f.flush(); os.fsync(f.fileno())
    os.replace(temp,p)
    fd=os.open(p.parent,os.O_DIRECTORY); os.fsync(fd); os.close(fd)
def load(p): return torch.load(p,map_location='cpu',weights_only=False)
def freeze_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if file_hash(src)!=file_hash(dst): raise RuntimeError(f'copy identity differs: {dst}')
    else:
        temp=dst.with_suffix(dst.suffix+'.copytmp')
        with Path(src).open('rb') as source, temp.open('wb') as target:
            shutil.copyfileobj(source,target);target.flush();os.fsync(target.fileno())
        os.replace(temp,dst)
        fd=os.open(dst.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def hp_vector(c):
    # Exact old hp_feature_vector computation/order, checked against archived inputs.
    return [math.log10(float(c['lr'])),float(c['dropout']),math.log2(float(c['hidden_dim'])),math.log10(float(c['l2']))]

def feature_part(rows,base,scaler):
    z=torch.tensor([base[c['architecture_fingerprint']] for c in rows],dtype=torch.float32)
    hp=torch.tensor([hp_vector(c) for c in rows],dtype=torch.float32)
    zm,zs,hm,hs=[torch.tensor(scaler[k],dtype=torch.float32) for k in ('z_mean','z_scale','hp_mean','hp_scale')]
    return dict(ids=[c['candidate_id'] for c in rows],architectures=[c['architecture_fingerprint'] for c in rows],
                z=(z-zm)/zs,hp=(hp-hm)/hs)

def actual_objects(model,optimizer):
    named={id(p):n for n,p in model.named_parameters()}
    groups=[{**{k:v for k,v in g.items() if k!='params'},'params':[named[id(p)] for p in g['params']]} for g in optimizer.param_groups]
    groups=json.loads(json.dumps(groups))
    for g in groups:
        for k,v in CONFIG['optimizer'].items():
            if g[k]!=v: raise ValueError(f'actual Adam {k} conflict')
    return dict(model=str(model),parameters={n:{'shape':list(p.shape),'dtype':str(p.dtype),'requires_grad':p.requires_grad} for n,p in model.named_parameters()},
        optimizer_class=type(optimizer).__module__+'.'+type(optimizer).__name__,param_groups=groups,
        optimizer_initial_state_entries=len(optimizer.state),device=str(next(model.parameters()).device),threads=torch.get_num_threads(),
        import_paths={name:str(Path(inspect.getsourcefile(obj)).resolve()) for name,obj in [('JointMetricModel',JointMetricModel),('objective',objective),('build_plan',build_plan),('fresh_adam',fresh_adam)]})

def audit(output):
    """No optimization; cross-check original completion records before creating inputs."""
    old=read(OLD/'protocol.json'); sup=read(SUP/'config_and_identity.json')
    ids,qids=old['inner']['train_ids'],old['inner']['query_ids']
    assert len(ids)==112 and len(qids)==29 and sup['train_ids']==ids and sup['query_ids']==qids
    assert sup['old_protocol_sha256']==file_hash(OLD/'protocol.json')
    candidates={c['candidate_id']:c for c in read(PREP/'candidates.json')}; base=read(PREP/'base_encoding.json')
    train,query=[[candidates[i] for i in seq] for seq in (ids,qids)]
    assert len({c['architecture_fingerprint'] for c in train})==60
    assert len({c['architecture_fingerprint'] for c in query})==16
    assert len({c['training_protocol_identity'] for c in train+query})==1
    # Verify against prior supplement's delivered separated source identity.
    delivery=read(SUP/'delivery_manifest.json')
    assert delivery['key_results']['config_and_identity.json']==file_hash(SUP/'config_and_identity.json')
    label_path=SUP/'training_labels.json'
    labels=read(label_path)
    originals={str(p):file_hash(p) for p in (OLD/'protocol.json',SUP/'config_and_identity.json',SUP/'delivery_manifest.json',label_path,PREP/'candidates.json',PREP/'base_encoding.json')}
    warm_checks=[]; plan_info=[]; all_inputs=[]
    for seed in CONFIG['model_seeds']:
        folder=OLD/'tasks'/f'regression-s{seed}-mapping'; path=folder/'M100.pt'
        completed=read(folder/'completed.json'); source=read(folder/'source.json')
        assert file_hash(path)==completed['artifacts']['M100.pt']
        assert file_hash(folder/'source.json')==completed['artifacts']['source.json']
        reuse=next(r for r in old['reuse'] if r['method']=='regression' and r['seed']==seed)
        origin=Path(reuse['checkpoint']); original_completion=read(origin.parent/'completed.json')
        original_started=read(origin.parent/'started.json')
        original_protocol=read(origin.parents[3]/'protocol.json')
        original_boundary=read(origin.parents[3]/'label_boundary.json')
        old_task=original_started['task']
        assert old_task['train_ids']==ids and old_task['query_ids']==qids and old_task['seed']==seed
        assert old_task['setting']['method']=='regression'
        expected_original_identity=digest({'task':old_task,'protocol':original_protocol['protocol_hash'],'A_labels':original_boundary['exported_hash']})
        assert original_started['identity']==original_completion['identity']==expected_original_identity
        assert file_hash(origin)==reuse['sha256']==source['old_checkpoint_hash']==original_completion['artifact_hashes']['model.pt']
        s,orig=load(path),load(origin)
        assert tensor_hash(s['model'])==tensor_hash(orig['model']) and s['standardization']==orig['standardization']
        assert s['candidate_identity_hash']==orig['candidate_identity_hash']==digest({'train':train,'query':query})
        assert s['setting']==old['settings']['regression'] and s['model_seed']==seed
        assert s['train_ids']==ids and s['query_ids']==qids and orig['train_ids']==ids and orig['query_ids']==qids
        assert {float(v['step']) for v in s['optimizer']['state'].values()}=={100.}
        assert {float(v['step']) for v in orig['optimizer']['state'].values()}=={100.}
        scaler=s['standardization']; assert scaler['fit_ids']==ids
        assert scaler['hp_feature_order']==['log10_lr','dropout','log2_hidden_dim','log10_l2']
        features,qfeatures=feature_part(train,base,scaler),feature_part(query,base,scaler)
        y,se,target=validate_batch(features,qfeatures,labels,scaler)
        raw_target=torch.tensor((y-scaler['y_mean'])/scaler['y_scale'],dtype=torch.float64)
        raw_noise=torch.tensor((se/scaler['y_scale'])**2,dtype=torch.float64)
        target_source_diff=float((raw_target-orig['gp_train_targets']).abs().max())
        noise_source_diff=float((raw_noise-orig['gp_fixed_noise_normalized']).abs().max())
        assert target_source_diff<=1e-12 and noise_source_diff<=1e-12, 'training target/SE differs from original M100 task'
        m=JointMetricModel(12,seed=seed); assert tensor_hash(m.state_dict())==s['initialization_hash']
        m.load_state_dict(s['model'],strict=True)
        with torch.no_grad():
            e,p=m(features['z'],features['hp']); qe,qp=m(qfeatures['z'],qfeatures['hp'])
        assert torch.equal(e,s['train_embeddings']) and torch.equal(e,orig['train_embeddings'])
        old_prediction=OLD/'tasks'/f'regression-s{seed}-M100-gp'/'G20_predictions_before_reveal.json'
        pc=read(old_prediction.parent/'completed.json'); assert file_hash(old_prediction)==pc['artifacts'][old_prediction.name]
        previous=read(old_prediction)
        assert previous['train_ids']==ids and previous['query_ids']==qids and previous['standardization']==scaler
        byid={r['candidate_id']:r for r in previous['predictions']}
        pred=qp.double().numpy()*scaler['y_scale']+scaler['y_mean']
        diff=float(np.max(np.abs(pred-np.array([byid[c]['head_mean'] for c in qids]))))
        assert diff<=1e-12, f'warmstart head differs {diff}'
        distances=torch.cdist(qe.double(),e.double()).numpy()
        maxdistance=0.
        for qi,cid in enumerate(qids):
            near=sorted(range(112),key=lambda j:(float(distances[qi,j]),ids[j]))[:5]
            assert [ids[j] for j in near]==byid[cid]['neighbor_ids']
            maxdistance=max(maxdistance,float(np.max(np.abs(distances[qi,near]-np.array(byid[cid]['neighbor_distances'])))))
        assert maxdistance==0.
        r0,r1=make_branches(m); opt=fresh_adam(r0); objects=actual_objects(r0,opt)
        inputs=dict(train=features,query=qfeatures,labels=labels,scaler=scaler,target=target,
                    candidate_identity_hash=s['candidate_identity_hash'],seed=seed)
        inp_path=output/'inputs'/f's{seed}.pt'
        if not inp_path.exists(): save(inp_path,inputs)
        else:
            a=load(inp_path); assert tensor_hash({'z':a['train']['z'],'hp':a['train']['hp'],'target':a['target']})==tensor_hash({'z':features['z'],'hp':features['hp'],'target':target})
        warm_path=output/'warmstarts'/f's{seed}-M100.pt'; freeze_copy(path,warm_path)
        plan=build_plan(y,se,ids,scaler['y_mean'],scaler['y_scale'],seed)
        maxcast=max(float(np.max(np.abs(plan['margins'][k]-tensor_plan(plan,k)[1].double().numpy()))) for k in range(200))
        planpath=output/'plans'/f's{seed}.npz'; planpath.parent.mkdir(exist_ok=True)
        if not planpath.exists(): np.savez_compressed(planpath,**plan)
        else:
            with np.load(planpath) as oldplan:
                assert all(np.array_equal(oldplan[k],v) for k,v in plan.items())
        # Check all legal constraints, not only the sampled plan.
        worst=0.; count=0
        yt=(y-scaler['y_mean'])/scaler['y_scale']
        for i in range(112):
            pairs,margins,weights=legal_pairs(y,se,ids,i,[j for j in range(112) if j!=i],scaler['y_mean'],scaler['y_scale'])
            j,k=pairs.T; hinges=np.maximum(0,margins+(yt[i]-yt[j])**2-(yt[i]-yt[k])**2)
            worst=max(worst,float(hinges.max(initial=0)));count+=len(hinges)
        assert worst<=1e-12
        warm_checks.append(dict(seed=seed,source=str(path),source_sha256=file_hash(path),original_task=str(origin),
            original_task_identity=original_completion['identity'],original_started_identity=original_started.get('identity'),
            original_sha256=file_hash(origin),parameter_hash=tensor_hash(s['model']),scaler_hash=digest(scaler),
            train_embeddings_bitwise_equal=True,embedding_maxdiff=0.,head_maxdiff=diff,head_tolerance=1e-12,old_neighbor_distance_maxdiff=maxdistance,
            original_training_target_maxdiff=target_source_diff,original_training_SE2_scaled_maxdiff=noise_source_diff,training_source_tolerance=1e-12,
            old_optimizer_exists=True,old_optimizer_steps=100,new_optimizer_momentum='intentionally reset both branches',branches_independent=True,
            target_cast_maxdiff=float(np.max(np.abs(target.double().numpy()-yt))),effective_objects=objects))
        plan_info.append(dict(seed=seed,path=str(planpath.relative_to(output)),sha256=file_hash(planpath),shape=list(plan['refs'].shape),
            empty_groups=int((plan['counts']==0).sum()),fewer8_groups=int((plan['counts']<8).sum()),margin_cast_maxdiff=maxcast,
            margin_cast_nonpositive=0,realizability_all_legal_triplets=count,realizability_max_hinge=worst))
        all_inputs.extend([inp_path,warm_path,planpath])
        for item in (path,origin,origin.parent/'completed.json',origin.parent/'started.json',folder/'source.json',folder/'completed.json',old_prediction,old_prediction.parent/'completed.json'):
            originals[str(item)]=file_hash(item)
            freeze_copy(item,output/'input_originals'/'state_records'/f's{seed}'/(('original_' if item.parent==origin.parent else 'diagnosis_prediction_' if item.parent==old_prediction.parent else 'diagnosis_mapping_')+item.name))
    write(output/'warmstart_checks.json',warm_checks)
    write(output/'plan_checks.json',plan_info)
    write(output/'inputs'/'candidate_identity.json',dict(train=train,query=query))
    write(output/'inputs'/'training_labels.json',labels)
    write(output/'input_sources.json',originals)
    write(output/'effective_config.json',dict(config=CONFIG,objects=[c['effective_objects'] for c in warm_checks]))
    source_hashes={p:file_hash(ROOT/p) for p in TRAIN_SOURCES}
    for p in TRAIN_SOURCES: freeze_copy(ROOT/p,output/'source_snapshot'/p)
    patch=''.join(''.join(difflib.unified_diff([], (ROOT/p).read_text().splitlines(True),fromfile='/dev/null',tofile=p)) for p in TRAIN_SOURCES if p!='surrogate/fuzzy_metric.py')
    (output/'training_changes.patch').write_text(patch)
    protocol=dict(config=CONFIG,train_ids=ids,query_ids=qids,train_architectures=60,query_architectures=16,
        source_hashes=source_hashes,input_hashes={str(p.relative_to(output)):file_hash(p) for p in all_inputs},original_sources=originals,
        labels='only separated112 train;29 holdout accessed by report after all6 sealed; B/test/otherA excluded',
        query_label_source=str(SUP/'query_validation_means.json'),
        query_label_source_sha256=file_hash(SUP/'query_validation_means.json'),
        protocol_document_sha256=file_hash(output/'input_originals/Codex_Server_R1_Audit_Implement_Pilot_Prompt.md'),
        semantics='delta raw, standardized squared variable margin capped1, squared embedding distance; mean(w*hinge) per group; fixed half groups/all centers; no sum(w) denominator',
        evaluation='same relative geometry definitions; train self excluded; R0-8D,R0-1D(predicted raw accuracy),R1-8D; common valid queries; architecture auxiliary',
        limits='single reused development split; 87 seed/query rows are29 candidates; no GP/search conclusion')
    protocol['identity']=digest(protocol); write(output/'protocol.json',protocol)
    write(output/'audit_complete.json',dict(protocol_identity=protocol['identity'],warmstarts=3,optimizer_updates=0))
    print(json.dumps({'audit':'complete','warmstarts':3,'train':112,'query':29,'plans':plan_info}))


def verify_protocol(output):
    p=read(output/'protocol.json'); identity=p.pop('identity'); assert digest(p)==identity; p['identity']=identity
    if p['config']!=CONFIG: raise ValueError('runtime config differs from frozen protocol')
    for name,h in p['source_hashes'].items():
        if file_hash(ROOT/name)!=h: raise ValueError(f'training source conflict: {name}')
    for name,h in p['input_hashes'].items():
        if file_hash(output/name)!=h: raise ValueError(f'input conflict: {name}')
    return p


def cost_status(output):
    intents={int(p.stem):read(p) for p in (output/'intents').glob('*.json')}
    committed=[]; effective=0; trajectories=[]
    for p in (output/'trajectories').glob('*/latest.pt'):
        s=load(p); committed+=s['committed_global_ids']; effective+=s['step']
        assert s['step']==len(s['history'])==len(s['committed_global_ids'])
        trajectories.append(dict(task=p.parent.name,effective_steps=s['step'],last_global_id=s['last_global_id']))
    if len(set(committed))!=len(committed) or not set(committed)<=set(intents): raise RuntimeError('update ledger inconsistent')
    uncertain=sorted(set(intents)-set(committed))
    return dict(effective_updates=effective,confirmed_actual_updates=len(committed),discarded_or_repeated_confirmed_updates=0,
        uncertain_reserved_updates=len(uncertain),uncertain_global_ids=uncertain,charged_upper_bound=len(intents),actual_limit=1260,
        remaining_repair_allowance=60-len(uncertain),trajectories=trajectories,GNN_training=0,VAE_training=0,GP_training=0,synthetic_optimizer_updates=0)


def checkpoint(model,opt,step,history,probes,committed,identity,plan_hash,input_hash):
    return dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),step=step,history=history,
        probes=probes,committed_global_ids=committed,last_global_id=committed[-1] if committed else None,
        identity=identity,plan_sha256=plan_hash,input_hash=input_hash,plan_position=step,torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(),cost=dict(effective_updates=step,confirmed_updates=len(committed)))


def inference(model,data,step,seed,branch):
    model.eval()
    with torch.no_grad():
        e,p=model(data['train']['z'],data['train']['hp']); qe,qp=model(data['query']['z'],data['query']['hp'])
    scaler=data['scaler']
    return dict(seed=seed,branch=branch,additional_step=step,train_ids=data['train']['ids'],query_ids=data['query']['ids'],
        train_embeddings=e.tolist(),query_embeddings=qe.tolist(),
        train_predictions=(p.double()*scaler['y_scale']+scaler['y_mean']).tolist(),
        query_predictions=(qp.double()*scaler['y_scale']+scaler['y_mean']).tolist(),parameter_hash=tensor_hash(model.state_dict()))


def infer_neighbors(bundle):
    """Save all three method spaces and self-excluded train neighbors before holdout labels."""
    methods=[bundle['branch']+'-8D']+(['R0-1D'] if bundle['branch']=='R0' else [])
    out={}
    for method in methods:
        te=np.asarray(bundle['train_embeddings'],dtype=np.float64); qe=np.asarray(bundle['query_embeddings'],dtype=np.float64)
        if method=='R0-1D': te=np.asarray(bundle['train_predictions'])[:,None]; qe=np.asarray(bundle['query_predictions'])[:,None]
        parts={}
        for part,ee,ids in [('train',te,bundle['train_ids']),('query',qe,bundle['query_ids'])]:
            rows=[]
            for cid,e in zip(ids,ee):
                refs=[j for j,r in enumerate(bundle['train_ids']) if r!=cid]
                d=np.sqrt(np.sum((te[refs]-e)**2,axis=1))
                near=sorted(range(len(refs)),key=lambda k:(float(d[k]),bundle['train_ids'][refs[k]]))[:5]
                rows.append(dict(candidate_id=cid,reference_ids=[bundle['train_ids'][j] for j in refs],distances=d.tolist(),
                    neighbor_ids=[bundle['train_ids'][refs[j]] for j in near]))
            parts[part]=rows
        out[method]=dict(train_embeddings=te.tolist(),query_embeddings=qe.tolist(),rows=parts)
    return out


def run(output):
    protocol=verify_protocol(output)
    check=read(output/'checks_complete.json')
    if check['protocol_identity']!=protocol['identity'] or not check['passed']: raise RuntimeError('checks not complete')
    # This process reads no query means. Frozen input payloads contain only training labels.
    started=time.monotonic(); before=cost_status(output)
    if before['uncertain_reserved_updates']>60: raise RuntimeError('repair reserve exceeded')
    for seed in [0,1,2]:
        data=load(output/'inputs'/f's{seed}.pt'); y,se,target=validate_batch(data['train'],data['query'],data['labels'],data['scaler'])
        assert torch.equal(target,data['target'])
        warm=load(output/'warmstarts'/f's{seed}-M100.pt'); original_hash=tensor_hash(warm['model'])
        model=JointMetricModel(12,seed=seed); model.load_state_dict(warm['model'],strict=True)
        branches=make_branches(model)
        planpath=output/'plans'/f's{seed}.npz'; plan=dict(np.load(planpath)); assert plan['ids'].tolist()==data['train']['ids']
        samples=[tensor_plan(plan,k) for k in range(200)]
        input_hash=tensor_hash({'z':data['train']['z'],'hp':data['train']['hp'],'qz':data['query']['z'],'qhp':data['query']['hp'],'target':target})
        for branch,m in zip(['R0','R1'],branches):
            folder=output/'trajectories'/f's{seed}-{branch}'; folder.mkdir(parents=True,exist_ok=True)
            tid=folder.name; lr=CONFIG['lambda_r'][branch]
            identity=digest(dict(protocol=protocol['identity'],seed=seed,branch=branch,warmstart=original_hash,plan=file_hash(planpath)))
            if (folder/'completed.json').exists():
                done=read(folder/'completed.json'); assert done['identity']==identity and done['effective_steps']==200
                for f,h in done['artifacts'].items(): assert file_hash(folder/f)==h
                continue
            opt=fresh_adam(m)
            write(folder/'actual_initial_objects.json',actual_objects(m,opt))
            if (folder/'latest.pt').exists():
                state=load(folder/'latest.pt'); assert state['identity']==identity and state['plan_sha256']==file_hash(planpath) and state['input_hash']==input_hash
                m.load_state_dict(state['model'],strict=True); opt.load_state_dict(copy.deepcopy(state['optimizer']))
                assert state['step']==state['plan_position']==len(state['committed_global_ids'])
                if state['step']: assert {float(v['step']) for v in opt.state.values()}=={float(state['step'])}
                torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng'])
                step,history,probes,committed=state['step'],state['history'],state['probes'],state['committed_global_ids']
            else:
                step,history,committed=0,[],[]; probes={'0':probe(m,data['train'],target,samples[0])}
                state=checkpoint(m,opt,step,history,probes,committed,identity,file_hash(planpath),input_hash)
                save(folder/'latest.pt',state)
                write(folder/'actual_initial_objects.json',actual_objects(m,opt))
            # Regenerate a missing diagnostic export only from its exact committed checkpoint.
            def node(k,state):
                path=folder/f'add{k:03d}.pt'
                freeze_copy(folder/'latest.pt',path)
                write(folder/f'inference_add{k:03d}.json',inference(m,data,k,seed,branch))
            if step in CONFIG['probe_steps']: node(step,state)
            while step<200:
                k=step+1;sample=samples[step]
                model_before={n:p.detach().clone() for n,p in m.named_parameters()}
                m.train();opt.zero_grad(set_to_none=True)
                e,p=m(data['train']['z'],data['train']['hp']); values=objective(e,p,target,sample,lr)
                nums={key:float(v.detach()) for key,v in values.items()}
                assert math.isclose(nums['total'],nums['regression']+lr*nums['order']+.001*nums['size'],abs_tol=2e-7,rel_tol=2e-6)
                if not all(math.isfinite(v) for v in nums.values()): raise RuntimeError('nonfinite loss')
                values['total'].backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()): raise RuntimeError('nonfinite/missing gradient')
                grad_norm=float(torch.cat([p.grad.reshape(-1) for p in m.parameters()]).norm())
                cost=cost_status(output); gid=cost['charged_upper_bound']+1
                if gid>1260: raise RuntimeError('actual update upper bound exhausted')
                write(output/'intents'/f'{gid:06d}.json',dict(global_id=gid,task=tid,additional_step=k,previous_commit=committed[-1] if committed else None,
                    protocol_identity=protocol['identity'],parameter_hash_before=tensor_hash(m.state_dict()),time=utc()))
                opt.step()  # The only optimizer.step call in the implementation.
                if any(not torch.isfinite(p).all() for p in m.parameters()): raise RuntimeError('nonfinite parameter after update; intent reserved')
                changes={n:float((p.detach()-model_before[n]).abs().max()) for n,p in m.named_parameters()}
                assert max(changes.values())>0 and all(v<=.05 for v in changes.values())
                assert {float(v['step']) for v in opt.state.values()}=={float(k)}
                assert tensor_hash({'z':data['train']['z'],'hp':data['train']['hp'],'qz':data['query']['z'],'qhp':data['query']['hp'],'target':target})==input_hash
                assert tensor_hash(warm['model'])==original_hash
                row=dict(step=k,global_id=gid,plan_position_used=step,lambda_r=lr,lambda_e=.001,**nums,gradient_norm=grad_norm)
                if k==1:
                    with torch.no_grad():
                        ae,ap=m(data['train']['z'],data['train']['hp']); av=objective(ae,ap,target,sample,lr)
                    row['after_same_sample']={key:float(v) for key,v in av.items()};row['parameter_max_changes']=changes
                    row['frozen_inputs_unchanged']=True;row['original_warm_state_unchanged']=True;row['Adam_steps']=1
                history.append(row);committed.append(gid);step=k
                if step in CONFIG['probe_steps']: probes[str(step)]=probe(m,data['train'],target,samples[0])
                state=checkpoint(m,opt,step,history,probes,committed,identity,file_hash(planpath),input_hash)
                save(folder/'latest.pt',state)
                if step in CONFIG['probe_steps']: node(step,state)
            bundle=read(folder/'inference_add200.json');write(folder/'neighbors_before_labels.json',infer_neighbors(bundle))
            write(folder/'loss_history.json',history);write(folder/'probes.json',probes)
            artifacts={p.name:file_hash(p) for p in folder.iterdir() if p.is_file() and p.name!='completed.json'}
            write(folder/'completed.json',dict(identity=identity,protocol_identity=protocol['identity'],effective_steps=200,original_step=100,endpoint=300,
                confirmed_actual_updates=len(committed),artifacts=artifacts,completed_at=utc()))
            print(json.dumps({'completed':tid,'updates':200,'cost':cost_status(output)['charged_upper_bound']}),flush=True)
    cost=cost_status(output); assert cost['effective_updates']==1200 and len(list((output/'trajectories').glob('*/completed.json')))==6
    assert not any(k in sys.modules for k in ('gpytorch','botorch','torch_geometric'))
    sealed={str(p.relative_to(output)):file_hash(p) for p in (output/'trajectories').rglob('*') if p.is_file()}
    if not (output/'all_inference_sealed.json').exists():
        write(output/'all_inference_sealed.json',dict(protocol_identity=protocol['identity'],sealed_at=utc(),artifacts=sealed,
            training_process_pid=os.getpid(),query_label_reads=0,forbidden_model_packages_loaded=[],cost=cost))
    else:
        seal=read(output/'all_inference_sealed.json');assert seal['artifacts']==sealed
    idx=len(list((output/'invocations').glob('run*.json')))+1
    receipt=dict(before=before,after=cost,optimizer_updates_this_invocation=cost['confirmed_actual_updates']-before['confirmed_actual_updates'],elapsed_seconds=time.monotonic()-started,pid=os.getpid())
    write(output/'invocations'/f'run{idx:02d}.json',receipt)
    print(json.dumps(receipt))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--config',type=Path)
    modes=parser.add_mutually_exclusive_group(required=True)
    for k in ['audit','check','run','report']: modes.add_argument('--'+k,action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.config and read(args.config)!=CONFIG: raise ValueError('CLI configuration conflicts with fixed protocol')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    try:
        if args.audit: audit(out)
        elif args.check:
            protocol=verify_protocol(out)
            command=[sys.executable,'-m','pytest','-q','tests/test_relative_metric_pilot.py']
            done=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,env={**os.environ,'R1_OUTPUT':str(out)})
            (out/'logs'/'checks.log').write_text(done.stdout+done.stderr)
            if done.returncode: raise RuntimeError(done.stdout+done.stderr)
            write(out/'checks_complete.json',dict(protocol_identity=protocol['identity'],passed=True,command=command,log_sha256=file_hash(out/'logs/checks.log'),optimizer_updates=0))
            print(done.stdout)
        elif args.run:
            with (out/'run.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); run(out)
        else:
            subprocess.run([sys.executable,str(ROOT/'analyse/relative_metric_report.py'),str(out)],check=True)
    except Exception:
        folder=out/'failures';folder.mkdir(exist_ok=True)
        n=len(list(folder.glob('*.json')))+1
        write(folder/f'{n:03d}.json',dict(time=utc(),mode=next(k for k in ['audit','check','run','report'] if getattr(args,k)),error=traceback.format_exc(),
            cost=cost_status(out) if (out/'intents').exists() else {'confirmed_actual_updates':0,'uncertain_reserved_updates':0}))
        raise
if __name__=='__main__': main()
