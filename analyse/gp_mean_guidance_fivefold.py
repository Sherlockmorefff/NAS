"""Fixed M100->M300 GP-mean experiment with fold isolation and two update ledgers."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,inspect,json,math,os,re,subprocess,sys,time,traceback,warnings
from pathlib import Path
import numpy as np
import torch,gpytorch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from analyse.relative_metric_pilot import read,write,save,load,freeze_copy,file_hash,digest,tensor_hash,feature_part,actual_objects,utc
from surrogate.relative_metric import validate_batch,make_branches,fresh_adam,ADAM
from surrogate.fuzzy_metric import JointMetricModel
from analyse.fuzzy_metric_replay import ReplayGP
from surrogate.gp_mean_guidance import architecture_groups,from_gp,condition,mean,posterior,objective,gradient_probe
DEFAULT=ROOT/'results/posthoc/gp-mean-guidance-fivefold-20260913-v1'
OLD=ROOT/'results/posthoc/fuzzy-metric-stage2-20260910-v1'
R1=ROOT/'results/posthoc/relative-metric-r1-fivefold-20260911-v2'
CLOSURE=ROOT/'results/posthoc/fuzzy-metric-fivefold-closure-20260910-v1'
CONFIG=dict(version='gp_mean_guidance_fivefold_v1',folds=list(range(5)),seeds=[0,1,2],branches=['C','G'],
 input=dict(z=12,hp=4,hp_order=['log10_lr','dropout','log2_hidden_dim','log10_l2'],hp_embedding=16,mapping=[28,32,32,8],head=[8,1]),
 warmstart=100,additional_steps=200,mapping_optimizer={**ADAM,'betas':list(ADAM['betas'])},lambda_guidance=.1,lambda_size=.001,
 block_starts=[0,50,100,150],nodes=[0,1,50,100,150,200],groups=3,group_hash_prefix='gp_mean_guidance_v1:',
 gp_steps=20,gp_seed=0,gp_optimizer=dict(lr=.05,betas=[.9,.999],eps=1e-8,weight_decay=0,amsgrad=False,maximize=False,foreach=None,capturable=False,differentiable=False,fused=None),
 gp_kernel='ConstantMean+ScaleKernel(isotropic Matern5/2)',fixed_noise='(SE/outer_y_scale)^2',noise_initial_raw_sd=1e-4,noise_lower_raw_sd=1e-8,min_fixed_noise=0.,cholesky_jitter_double=1e-6,cholesky_max_tries=3,
 device='cpu',threads=1,mapping_dtype='float32',gp_dtype='float64',guidance_target_dtype='float64',interval_z=1.96,
 effective_mapping=6000,effective_gp=4200,limits={'mapping':6300,'gp':4410},GNN=0,VAE=0,selection='prediction descending, candidate_id ascending; five per outer fold',
 scheduler=None,gradient_clip=None,early_stop=False,normalization=None,neighborhood=False,pseudo_labels=False,variance_minimization=False,
 tolerances=dict(old_target=1e-12,gp_mean=1e-9,gp_variance=1e-8,finite_difference_step=1e-6,finite_difference_atol=2e-6,finite_difference_rtol=2e-5))
SOURCES=['surrogate/gp_mean_guidance.py','analyse/gp_mean_guidance_fivefold.py','surrogate/fuzzy_metric.py','surrogate/relative_metric.py','analyse/relative_metric_pilot.py','analyse/fuzzy_metric_replay.py']

def audit(o):
    from analyse.fuzzy_metric_fivefold_closure import completed_state
    old,part,matrix,boundary=[read(OLD/n) for n in ['protocol.json','partition.json','task_matrix.json','label_boundary.json']]
    rp=read(R1/'protocol.json');assert digest(part)==old['partition_hash'] and digest(matrix)==old['task_matrix_hash']
    assert rp['A_ids']==part['A_ids'] and rp['folds']==part['folds']
    for rel,h in rp['source_hashes'].items():assert file_hash(ROOT/rel)==h
    labels=read(OLD/'A_validation.json');assert set(labels)==set(part['A_ids']) and digest(labels)==boundary['exported_hash']
    prep=ROOT/old['config']['prepared'];pool=read(prep/'candidates.json');base=read(prep/'base_encoding.json')
    assert digest(pool)==old['prepared_manifest']['pool_hash'] and digest(base)==old['prepared_manifest']['base_encoding_hash']
    candidates={c['candidate_id']:c for c in pool if c['candidate_id'] in part['A_ids']}
    assert len(candidates)==179 and len({c['architecture_fingerprint'] for c in candidates.values()})==96
    from collections import Counter
    assert Counter(c for f in part['folds'] for c in f['query_ids'])==Counter(part['A_ids'])
    inventory={r['state_id']:r for r in read(CLOSURE/'state_inventory.json')}
    tasks=[];checks=[];missing=[];hashes={};orig={};group_records=[]
    originals=[OLD/n for n in ['protocol.json','partition.json','task_matrix.json','label_boundary.json','A_validation.json']]+[R1/'protocol.json',CLOSURE/'state_inventory.json',prep/'candidates.json',prep/'base_encoding.json']
    for src in originals:orig[str(src)]=file_hash(src);freeze_copy(src,o/'input_originals'/(src.parent.name+'_'+src.name))
    write(o/'candidate_identity.json',list(candidates.values()))
    for fold,split in enumerate(part['folds']):
      ids,qids=split['train_ids'],split['query_ids'];tr,qu=[[candidates[c] for c in seq] for seq in [ids,qids]]
      assert not {r['architecture_fingerprint'] for r in tr}&{r['architecture_fingerprint'] for r in qu}
      assert len(ids)==[141,143,143,145,144][fold] and len(qids)==[38,36,36,34,35][fold]
      groups=architecture_groups(ids,[r['architecture_fingerprint'] for r in tr]);group_records.append(dict(fold=fold,groups=groups))
      write(o/'inputs'/f'fold{fold}'/'groups.json',groups)
      write(o/'inputs'/f'fold{fold}'/'train_labels.json',{c:labels[c] for c in ids})
      write(o/'evaluation_labels'/f'fold{fold}.json',{c:labels[c] for c in qids})
      sel=read(OLD/'selection'/f'fold{fold}.json');orig[str(OLD/'selection'/f'fold{fold}.json')]=file_hash(OLD/'selection'/f'fold{fold}.json')
      for seed in range(3):
       sid=f'fold{fold}-refit-regression-s{seed}'
       try:
        task=copy.deepcopy(next(t for t in matrix if t['task_id']==sid));task['setting']=sel['settings']['regression']
        assert task['method']=='regression' and task['setting']['lambda_f']==0 and task['train_ids']==ids and task['query_ids']==qids
        identity=digest(dict(task=task,protocol=old['protocol_hash'],A_labels=boundary['exported_hash']));rec=completed_state(task,identity)
        assert rec['completion_identity']==inventory[sid]['completion_identity'] and rec['artifact_hashes']==inventory[sid]['artifact_hashes']
        src=Path(rec['model_path']);s=load(src)
        assert s['model_seed']==seed and s['setting']==task['setting'] and s['common_config']['representation_steps']==100
        assert {float(v['step']) for v in s['optimizer']['state'].values()}=={100.}
        assert s['train_ids']==ids and s['query_ids']==qids and s['candidate_identity_hash']==digest({'train':tr,'query':qu})
        sc=s['standardization'];assert sc['fit_ids']==ids and sc['hp_feature_order']==CONFIG['input']['hp_order']
        train,query=feature_part(tr,base,sc),feature_part(qu,base,sc);tl={c:labels[c] for c in ids};y,se,t32=validate_batch(train,query,tl,sc)
        t64=torch.tensor((y-sc['y_mean'])/sc['y_scale'],dtype=torch.double);se=torch.tensor(se,dtype=torch.double)
        assert float((t64-s['gp_train_targets']).abs().max())<=1e-12 and float(((se/sc['y_scale'])**2-s['gp_fixed_noise_normalized']).abs().max())<=1e-12
        m=JointMetricModel(12,seed=seed);assert tensor_hash(m.state_dict())==s['initialization_hash'];m.load_state_dict(s['model']);m.requires_grad_(True)
        with torch.no_grad():e,_=m(train['z'],train['hp'])
        assert torch.equal(e,s['train_embeddings']) and torch.equal(e.double(),s['gp_train_inputs'])
        rw=load(R1/'warmstarts'/f'fold{fold}-s{seed}.pt');assert tensor_hash(rw['model'])==tensor_hash(s['model'])
        a,b=make_branches(m);objects=actual_objects(a,fresh_adam(a));assert not fresh_adam(b).state and all(p.requires_grad for p in a.parameters())
        inp=o/'inputs'/f'fold{fold}'/f's{seed}.pt';warm=o/'warmstarts'/f'fold{fold}-s{seed}.pt'
        if not inp.exists():save(inp,dict(fold=fold,seed=seed,train=train,query=query,labels=tl,scaler=sc,target32=t32,target64=t64,se=se,groups=groups))
        if not warm.exists():save(warm,dict(model=copy.deepcopy(s['model']),scaler=sc,source_sha256=file_hash(src),seed=seed,train_ids=ids,query_ids=qids))
        for name in [src,src.parent/'completed.json',src.parent/'started.json']:orig[str(name)]=file_hash(name)
        for name in [src.parent/'completed.json',src.parent/'started.json']:freeze_copy(name,o/'input_originals'/'state_records'/sid/name.name)
        for path in [inp,warm]:hashes[str(path.relative_to(o))]=file_hash(path)
        checks.append(dict(fold=fold,seed=seed,source=rec,weight_sha256=tensor_hash(s['model']),original_steps=100,requires_grad=True,train_embedding_maxdiff=0.,standardization=sc,target_cast_maxdiff=float((t64-t32.double()).abs().max()),target_original_maxdiff=float((t64-s['gp_train_targets']).abs().max()),SE2_original_maxdiff=float(((se/sc['y_scale'])**2-s['gp_fixed_noise_normalized']).abs().max()),objects=objects))
        for branch in ['C','G']:tasks.append(dict(task_id=f'fold{fold}-s{seed}-{branch}',fold=fold,seed=seed,branch=branch,input=str(inp.relative_to(o)),warmstart=str(warm.relative_to(o)),train_ids=ids,query_ids=qids))
       except (FileNotFoundError,KeyError,ValueError,AssertionError,StopIteration) as exc:missing.append(dict(fold=fold,seed=seed,error=repr(exc),traceback=traceback.format_exc()))
    for path in (o/'inputs').rglob('*.json'):hashes[str(path.relative_to(o))]=file_hash(path)
    write(o/'task_matrix.json',tasks);write(o/'warmstart_checks.json',checks);write(o/'groups.json',group_records);write(o/'missing_before_evaluation.json',missing);write(o/'input_sources.json',orig)
    write(ROOT/'configs/gp_mean_guidance_fivefold_v1.json',CONFIG)
    for rel in SOURCES:freeze_copy(ROOT/rel,o/'source_snapshot'/rel)
    protocol=dict(config=CONFIG,A_ids=part['A_ids'],folds=part['folds'],task_matrix_sha=digest(tasks),tasks=len(tasks),missing=missing,input_hashes=hashes,source_hashes={r:file_hash(ROOT/r) for r in SOURCES},
      query_label_hashes={str(p.relative_to(o)):file_hash(p) for p in (o/'evaluation_labels').glob('*.json')},groups_sha=digest(group_records),prompt_sha=file_hash(o/'input_originals/Codex_Server_GP_Mean_Guidance_Prompt.md'),
      interpretation='development only; unseen architecture performance training labels, VAE may have seen architecture; not strict internal cross-fitting; no search; input comparison deferred; M100->M300')
    protocol['identity']=digest(protocol);write(o/'protocol.json',protocol)
    print(json.dumps(dict(warmstarts=len(checks),tasks=len(tasks),missing=missing,updates=0)),flush=True)

def verify_protocol(o,fold=None):
    p=read(o/'protocol.json');ident=p.pop('identity');assert digest(p)==ident;p['identity']=ident
    assert p['config']==CONFIG
    for rel,h in p['source_hashes'].items():assert file_hash(ROOT/rel)==h,rel
    tasks=read(o/'task_matrix.json');assert digest(tasks)==p['task_matrix_sha']
    selected=tasks if fold is None else [t for t in tasks if t['fold']==fold]
    paths=p['input_hashes'] if fold is None else {t[k] for t in selected for k in ['input','warmstart']}
    for rel in paths:assert file_hash(o/rel)==p['input_hashes'][rel]
    return p,selected

def costs(o):
    result={}
    for kind in ['mapping','gp']:
        intents={int(p.stem) for p in (o/'intents'/kind).glob('*.json')};receipts={int(p.stem) for p in (o/'receipts'/kind).glob('*.json')}
        paths=(o/'trajectories').glob('*/mapping/latest.pt') if kind=='mapping' else (o/'trajectories').glob('*/gps/*/latest.pt')
        committed=[];fits=0
        for path in paths:
            s=load(path);assert s['step']==len(s['history'])==len(s['committed']);committed+=s['committed'];fits+=s['step']==(200 if kind=='mapping' else 20)
        assert len(set(committed))==len(committed) and set(committed)<=intents and receipts<=intents
        result[kind]=dict(effective=len(committed),actual_confirmed=len(receipts|set(committed)),discarded=len(receipts-set(committed)),uncertain=len(intents-(receipts|set(committed))),charged=len(intents),limit=CONFIG['limits'][kind],complete_fits=fits)
    return result

class Ledger:
    def __init__(self,o,identity):
        self.o=o;self.identity=identity;self.next={k:len(list((o/'intents'/k).glob('*.json'))) for k in ['mapping','gp']}
    def update(self,kind,task,step,opt):
        if self.next[kind]>=CONFIG['limits'][kind]:raise RuntimeError(kind+' actual budget exhausted')
        self.next[kind]+=1;gid=self.next[kind]
        write(self.o/'intents'/kind/f'{gid:06d}.json',dict(global_id=gid,task=task,step=step,protocol=self.identity,time=utc()))
        opt.step()
        write(self.o/'receipts'/kind/f'{gid:06d}.json',dict(global_id=gid,task=task,step=step,optimizer_returned=True))
        return gid

def guard_files(o,tasks):
    allowed={str((o/t[k]).resolve()) for t in tasks for k in ['input','warmstart']}
    allowed|={str(o/n) for n in ['protocol.json','task_matrix.json','checks_complete.json','worker.lock']}
    prefixes=[str((o/'trajectories'/t['task_id']).resolve()) for t in tasks]+[str(o/n) for n in ['intents','receipts']]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=str(Path(os.fsdecode(args[0])).resolve())
        if p.startswith(str(ROOT/'results')+'/') and p not in allowed and not any(p==x or p.startswith(x+'/') for x in prefixes):raise PermissionError('fold worker forbidden file: '+p)
    sys.addaudithook(hook)

def make_gp(x,y,se,scale):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0);gp=ReplayGP(x.detach().double(),y.detach().double(),se.detach().double(),scale).double()
    gp.train();gp.likelihood.train();opt=torch.optim.Adam(gp.parameters(),lr=.05)
    actual={k:(list(v) if isinstance(v,tuple) else v) for k,v in opt.param_groups[0].items() if k!='params'}
    assert actual==CONFIG['gp_optimizer'],actual
    return gp,opt

def state_rng():return dict(torch_rng=torch.get_rng_state().clone(),numpy_rng=np.random.get_state())
def restore_rng(s):torch.set_rng_state(s['torch_rng']);np.random.set_state(s['numpy_rng'])

def fit_gp(folder,x,y,se,scale,metadata,ledger):
    ident=digest(dict(metadata=metadata,x=tensor_hash({'x':x}),target=tensor_hash({'y':y,'se':se}),scale=scale,protocol=ledger.identity))
    gp,opt=make_gp(x,y,se,scale);start=time.monotonic()
    if (folder/'latest.pt').exists():
        s=load(folder/'latest.pt');assert s['identity']==ident;gp.load_state_dict(s['model']);opt.load_state_dict(copy.deepcopy(s['optimizer']));restore_rng(s)
    else:
        s=dict(identity=ident,step=0,history=[],committed=[],model=copy.deepcopy(gp.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),seconds=0.,metadata=metadata,x=x.detach().clone(),target=y.detach().clone(),se=se.detach().clone(),scale=scale,**state_rng());save(folder/'latest.pt',s)
    if s['step'] and not (folder/f"step{s['step']:02d}.pt").exists():freeze_copy(folder/'latest.pt',folder/f"step{s['step']:02d}.pt")
    mll=gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood,gp)
    for k in range(s['step']+1,21):
        before=time.monotonic();opt.zero_grad(set_to_none=True)
        with warnings.catch_warnings(record=True) as captured,gpytorch.settings.cholesky_jitter(double_value=1e-6),gpytorch.settings.cholesky_max_tries(3),gpytorch.settings.min_fixed_noise(double_value=0.,float_value=0.,half_value=0.):
            warnings.simplefilter('always');loss=-mll(gp(gp.train_inputs[0]),gp.train_targets)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite GP objective')
            loss.backward()
        grad=torch.cat([p.grad.reshape(-1) for p in gp.parameters()]);assert torch.isfinite(grad).all()
        gid=ledger.update('gp',metadata['task']+'/'+metadata['name'],k,opt)
        messages=[str(w.message) for w in captured];jitter=max([0.]+[float(v) for msg in messages for v in re.findall(r'jitter of ([\deE.+-]+)',msg)])
        s['history'].append(dict(step=k,global_id=gid,loss=float(loss.detach()),gradient_norm=float(grad.norm()),warnings=messages,actual_jitter=jitter))
        s['committed'].append(gid);s.update(step=k,model=copy.deepcopy(gp.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),seconds=s['seconds']+time.monotonic()-before,**state_rng());save(folder/'latest.pt',s)
        freeze_copy(folder/'latest.pt',folder/f'step{k:02d}.pt')
    if (folder/'teacher.pt').exists():teacher=load(folder/'teacher.pt');assert teacher['metadata']==metadata
    else:
        teacher=from_gp(gp,x,y,se,scale,metadata);save(folder/'teacher.pt',teacher)
        write(folder/'conditioning.json',dict(identity=ident,metadata=metadata,parameters=teacher['parameters'],jitter=teacher['jitter'],attempts=teacher['cholesky_attempts'],support_snapshot_sha=tensor_hash({'x':x}),target_sha=tensor_hash({'y':y}),fit_seconds=s['seconds'],fit_steps=20))
    return teacher

def map_snapshot(o,folder,m,opt,state,data,step,phase):
    state.update(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),**state_rng());save(folder/'mapping/latest.pt',state)
    with torch.no_grad():e,p=m(data['train']['z'],data['train']['hp']);qe,qp=m(data['query']['z'],data['query']['hp'])
    path=folder/'nodes'/f'add{step:03d}-{phase}.pt'
    if not path.exists():save(path,{**copy.deepcopy(state), 'train_e':e.detach().clone(),'query_e':qe.detach().clone(),'train_head':p.detach().clone(),'query_head':qp.detach().clone(),'phase':phase})
    else:assert tensor_hash(load(path)['model'])==tensor_hash(m.state_dict())

def endpoint_prediction(m,data,teacher):
    sc=data['scaler'];result={};pooled=float(data['se'].square().mean());extra=teacher['parameters']['noise']*sc['y_scale']**2
    with torch.no_grad():
      for part in ['train','query']:
        e,head=m(data[part]['z'],data[part]['hp']);mu,var,raw=posterior(teacher,e.double());pred=mu*sc['y_scale']+sc['y_mean'];lv=var*sc['y_scale']**2;ov=lv+extra+pooled;sd=ov.sqrt()
        hp=head.double()*sc['y_scale']+sc['y_mean'];ids=data[part]['ids']
        result[part]=dict(ids=ids,architectures=data[part]['architectures'],embeddings=e.tolist(),GP=pred.tolist(),head=hp.tolist(),latent_variance=lv.tolist(),raw_latent_variance_standardized=raw.tolist(),observation_variance=ov.tolist(),lower=(pred-1.96*sd).tolist(),upper=(pred+1.96*sd).tolist(),
          top5_GP=[ids[i] for i in sorted(range(len(ids)),key=lambda i:(-float(pred[i]),ids[i]))[:5]],top5_head=[ids[i] for i in sorted(range(len(ids)),key=lambda i:(-float(hp[i]),ids[i]))[:5]])
    return dict(**result,scaler=sc,pooled_training_SE2=pooled,extra_noise_raw=extra,teacher_parameters=teacher['parameters'],teacher_jitter=teacher['jitter'],hidden_query_SE_used=False)

def record_probe(state,value):
    key=('step','phase','block')
    existing=[v for v in state['probes'] if all(v[k]==value[k] for k in key)]
    if existing:assert len(existing)==1 and existing[0]==value
    else:state['probes'].append(value)

def support_diagnostic(data,snapshot,teachers):
    started=time.monotonic();rows=[]
    for group,teacher in zip(data['groups'],teachers):
        full=condition(snapshot['embedding'],data['target64'],data['se'],data['scaler']['y_scale'],teacher['parameters'])
        js=group['query_indices']
        for label,t in [('excluded_group',teacher),('full_training_support',full)]:
            q=snapshot['embedding'][js].detach().clone().requires_grad_(True);mu=mean(t,q);loss=(mu-data['target64'][js]).square().mean()
            gm,=torch.autograd.grad(mu.sum(),q,retain_graph=True);gl,=torch.autograd.grad(loss,q)
            rows.append(dict(group=group['group'],conditioning=label,query_ids=group['query_ids'],support_count=len(t['support']),MSE=float(loss.detach()),mean_query_gradient_norm=float(gm.norm()),MSE_query_gradient_norm=float(gl.norm()),parameters=t['parameters'],jitter=t['jitter']))
    return dict(rows=rows,matrix_seconds=time.monotonic()-started,optimizer_updates=0,note='Same parameters; removal also reduces support count, not a pure single-point causal experiment.')

def worker(o,fold):
    protocol,tasks=verify_protocol(o,fold);check=read(o/'checks_complete.json');assert check['passed'] and check['protocol']==protocol['identity']
    with (o/'worker.lock').open('a') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger=Ledger(o,protocol['identity']);guard_files(o,tasks)
      for task in tasks:
        folder=o/'trajectories'/task['task_id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'completed.json').exists():
            done=read(folder/'completed.json');assert done['protocol']==protocol['identity']
            for n,h in done['artifacts'].items():assert file_hash(folder/n)==h
            continue
        start=time.monotonic();data=load(o/task['input']);warm=load(o/task['warmstart']);assert data['fold']==fold and data['seed']==task['seed']
        assert data['train']['ids']==task['train_ids'] and data['query']['ids']==task['query_ids'];validate_batch(data['train'],data['query'],data['labels'],data['scaler'])
        m=JointMetricModel(12,seed=task['seed']);m.load_state_dict(warm['model']);m.requires_grad_(True);m.train();opt=fresh_adam(m)
        ident=digest(dict(protocol=protocol['identity'],task=task,warm=tensor_hash(warm['model'])))
        input_hash=tensor_hash({'z':data['train']['z'],'hp':data['train']['hp'],'target64':data['target64']})
        if (folder/'mapping/latest.pt').exists():
            state=load(folder/'mapping/latest.pt');assert state['identity']==ident;m.load_state_dict(state['model']);opt.load_state_dict(copy.deepcopy(state['optimizer']));restore_rng(state)
        else:
            state=dict(identity=ident,step=0,history=[],committed=[],probes=[],mapping_seconds=0.,input_hash=input_hash,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),**state_rng())
            write(folder/'initial_objects.json',actual_objects(m,opt));map_snapshot(o,folder,m,opt,state,data,0,'before_refresh')
        assert input_hash==state['input_hash'];assert {float(v['step']) for v in opt.state.values()}==({float(state['step'])} if state['step'] else set())
        teachers=None;current_block=None
        while state['step']<200:
            step=state['step'];block=step//50;blockstart=50*block
            if task['branch']=='G' and current_block!=block:
                snap=folder/'blocks'/f'b{block}'/'snapshot.pt'
                if snap.exists():snapshot=load(snap)
                else:
                    assert step==blockstart
                    with torch.no_grad():emb,_=m(data['train']['z'],data['train']['hp'])
                    snapshot=dict(embedding=emb.detach().double(),mapping_sha=tensor_hash(m.state_dict()),step=step,fold=fold,seed=task['seed']);save(snap,snapshot)
                if step==blockstart:assert snapshot['mapping_sha']==tensor_hash(m.state_dict())
                teachers=[]
                for group in data['groups']:
                    g=group['group'];js=group['support_indices'];name=f'b{block}-g{g}'
                    meta=dict(task=task['task_id'],name=name,fold=fold,seed=task['seed'],block=block,group=g,mapping_step=blockstart,mapping_sha=snapshot['mapping_sha'],support_ids=group['support_ids'],support_architectures=group['support_architectures'],query_ids=group['query_ids'],scaler=data['scaler'])
                    teachers.append(fit_gp(folder/'gps'/name,snapshot['embedding'][js],data['target64'][js],data['se'][js],data['scaler']['y_scale'],meta,ledger))
                write(folder/'blocks'/f'b{block}'/'ready.json',dict(step=blockstart,mapping_sha=snapshot['mapping_sha'],teachers={f'b{block}-g{g}':file_hash(folder/'gps'/f'b{block}-g{g}'/'teacher.pt') for g in range(3)}))
                current_block=block
                if block==0 and not (folder/'self_support_diagnostic.json').exists():write(folder/'self_support_diagnostic.json',support_diagnostic(data,snapshot,teachers))
                if step==blockstart:
                    pr=gradient_probe(m,data['train'],data['target32'],data['target64'],data['groups'],teachers);record_probe(state,dict(step=step,phase='after_refresh',block=block,**pr));map_snapshot(o,folder,m,opt,state,data,step,'after_refresh')
            before=time.monotonic();opt.zero_grad(set_to_none=True);e,p=m(data['train']['z'],data['train']['hp'])
            losses=objective(e,p,data['target32'],data['target64'],data['groups'],teachers,task['branch']=='G');assert all(torch.isfinite(v) for v in losses.values())
            losses['total'].backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            gid=ledger.update('mapping',task['task_id'],step+1,opt);assert all(torch.isfinite(p).all() for p in m.parameters())
            state['step']=step+1;state['committed'].append(gid);state['history'].append(dict(step=step+1,global_id=gid,block=block if teachers else None,**{k:float(v.detach()) for k,v in losses.items()}));state['mapping_seconds']+=time.monotonic()-before
            state.update(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),**state_rng());save(folder/'mapping/latest.pt',state)
            if state['step'] in CONFIG['nodes']:
                if teachers:record_probe(state,dict(step=state['step'],phase='before_refresh' if state['step']%50==0 else 'within_block',block=block,**gradient_probe(m,data['train'],data['target32'],data['target64'],data['groups'],teachers)))
                map_snapshot(o,folder,m,opt,state,data,state['step'],'before_refresh' if state['step']%50==0 else 'within_block')
        assert tensor_hash({'z':data['train']['z'],'hp':data['train']['hp'],'target64':data['target64']})==input_hash
        with torch.no_grad():train_e,_=m(data['train']['z'],data['train']['hp'])
        meta=dict(task=task['task_id'],name='endpoint',fold=fold,seed=task['seed'],block=None,group=None,mapping_step=200,mapping_sha=tensor_hash(m.state_dict()),support_ids=data['train']['ids'],support_architectures=sorted(set(data['train']['architectures'])),scaler=data['scaler'])
        final=fit_gp(folder/'gps/endpoint',train_e.detach().double(),data['target64'],data['se'],data['scaler']['y_scale'],meta,ledger)
        write(folder/'predictions_before_labels.json',endpoint_prediction(m,data,final));write(folder/'history.json',state['history']);write(folder/'probes.json',state['probes'])
        gp_seconds=sum(load(p)['seconds'] for p in (folder/'gps').glob('*/latest.pt'))
        write(folder/'timing.json',dict(mapping_update_seconds=state['mapping_seconds'],gp_update_seconds=gp_seconds,wall_seconds_this_invocation=time.monotonic()-start,includes_IO=True))
        write(folder/'completed.json',dict(protocol=protocol['identity'],task=task,mapping_steps=200,GP_fits=13 if task['branch']=='G' else 1,completed_at=utc(),pid=os.getpid(),query_labels_read=False,artifacts={str(p.relative_to(folder)):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name!='completed.json'}))
        print(json.dumps(dict(completed=task['task_id'],mapping_updates=ledger.next['mapping'],gp_updates=ledger.next['gp'])),flush=True)

def run(o):
    p,tasks=verify_protocol(o);before=costs(o);n=len(list((o/'invocations').glob('*.json')))+1
    for fold in range(5):
        log=o/'logs'/f'run{n:02d}-fold{fold}.log'
        with log.open('w') as out:r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run','--worker',str(fold),'--output',str(o)],stdout=out,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'fold worker failed: {log}')
        print(json.dumps(dict(fold=fold,cost=costs(o))),flush=True)
    after=costs(o);assert after['mapping']['effective']==len(tasks)*200 and after['gp']['effective']==sum(13 if t['branch']=='G' else 1 for t in tasks)*20
    assert len(list((o/'trajectories').glob('*/completed.json')))==len(tasks)
    artifacts={str(f.relative_to(o)):file_hash(f) for f in (o/'trajectories').rglob('*') if f.is_file()}
    if not (o/'all_predictions_sealed.json').exists():write(o/'all_predictions_sealed.json',dict(protocol=p['identity'],time=utc(),full_fivefold=len(tasks)==30,tasks=len(tasks),artifacts=artifacts,cost=after))
    else:assert read(o/'all_predictions_sealed.json')['artifacts']==artifacts
    write(o/'invocations'/f'run{n:02d}.json',dict(before=before,after=after,new_updates={k:after[k]['actual_confirmed']-before[k]['actual_confirmed'] for k in after},time=utc()))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--config',type=Path);parser.add_argument('--worker',type=int,choices=range(5))
    modes=parser.add_mutually_exclusive_group(required=True)
    for name in ['audit','check','run','report']:modes.add_argument('--'+name,action='store_true')
    a=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.config and read(a.config)!=CONFIG:raise ValueError('configuration conflicts with fixed GP guidance protocol')
    o=a.output.resolve()
    try:
        if a.audit:audit(o)
        elif a.check:
            p,_=verify_protocol(o);r=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_gp_mean_guidance.py'],cwd=ROOT,capture_output=True,text=True,env={**os.environ,'GP_GUIDANCE_OUTPUT':str(o)})
            (o/'logs/check.log').write_text(r.stdout+r.stderr)
            if r.returncode:raise RuntimeError(r.stdout+r.stderr)
            write(o/'checks_complete.json',dict(passed=True,protocol=p['identity'],updates=0,test_sha=file_hash(ROOT/'tests/test_gp_mean_guidance.py')));print(r.stdout)
        elif a.run:
            if a.worker is not None:worker(o,a.worker)
            else:
                with (o/'orchestrator.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(o)
        else:subprocess.run([sys.executable,str(ROOT/'analyse/gp_mean_guidance_report.py'),str(o)],check=True)
    except Exception:
        if a.worker is None:
            path=o/'failures'/f'{time.time_ns()}.json';write(path,dict(error=traceback.format_exc(),cost=costs(o),time=utc()))
        raise
if __name__=='__main__':main()
