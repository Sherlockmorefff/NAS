"""Synchronized support/query M100->M300 guidance; standalone protocol and ledgers."""
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
from surrogate.gp_mean_guidance import architecture_groups,from_gp,posterior
from surrogate.gp_mean_guidance_sync import objective,gradient_probe,SolveMeter,guidance
DEFAULT=ROOT/'results/posthoc/gp-mean-guidance-sync-fivefold-v1'
OLD=ROOT/'results/posthoc/fuzzy-metric-stage2-20260910-v1'
R1=ROOT/'results/posthoc/relative-metric-r1-fivefold-20260911-v2'
CLOSURE=ROOT/'results/posthoc/fuzzy-metric-fivefold-closure-20260910-v1'
CONFIG=dict(version='gp_mean_guidance_sync_fivefold_v1',folds=list(range(5)),seeds=[0,1,2],branches=['S'],
 input=dict(z=12,hp=4,hp_order=['log10_lr','dropout','log2_hidden_dim','log10_l2'],hp_embedding=16,mapping=[28,32,32,8],head=[8,1]),
 warmstart=100,additional_steps=200,mapping_optimizer={**ADAM,'betas':list(ADAM['betas'])},lambda_guidance=.1,lambda_size=.001,
 block_starts=[0,50,100,150],nodes=[0,1,50,100,150,200],groups=3,group_hash_prefix='gp_mean_guidance_v1:',
 gp_steps=20,gp_seed=0,gp_optimizer=dict(lr=.05,betas=[.9,.999],eps=1e-8,weight_decay=0,amsgrad=False,maximize=False,foreach=None,capturable=False,differentiable=False,fused=None),
 gp_kernel='ConstantMean+ScaleKernel(isotropic Matern5/2)',fixed_noise='(SE/outer_y_scale)^2',noise_initial_raw_sd=1e-4,noise_lower_raw_sd=1e-8,min_fixed_noise=0.,cholesky_jitter_double=1e-6,cholesky_max_tries=3,
 device='cpu',threads=1,mapping_dtype='float32',gp_dtype='float64',guidance_target_dtype='float64',interval_z=1.96,
 effective_mapping=3000,effective_gp=3900,limits={'mapping':3150,'gp':4095},GNN=0,VAE=0,selection='prediction descending, candidate_id ascending; five per outer fold',
 scheduler=None,gradient_clip=None,early_stop=False,normalization=None,neighborhood=False,pseudo_labels=False,variance_minimization=False,
 tolerances=dict(old_target=1e-12,gp_mean=1e-9,gp_variance=1e-8,finite_difference_step=1e-6,finite_difference_atol=2e-6,finite_difference_rtol=2e-5))
PREVIOUS=ROOT/'results/posthoc/gp-mean-guidance-fivefold-20260913-v1'
PREVIOUS_ID='fc28feb8b78e72e2b687270e01a11021b41fea238c507a8f8666bb38227fb62e'
CONFIG.update(support='current; gradient through support and query, K_SS, Cholesky and alpha',formal_conditionings=9000)
SOURCES=['surrogate/gp_mean_guidance_sync.py','analyse/gp_mean_guidance_sync_fivefold.py','tests/test_gp_mean_guidance_sync.py','surrogate/gp_mean_guidance.py','surrogate/fuzzy_metric.py','surrogate/relative_metric.py','analyse/relative_metric_pilot.py','analyse/fuzzy_metric_replay.py']

def audit(o):
    previous=read(PREVIOUS/'protocol.json');ident=previous.pop('identity');assert ident==PREVIOUS_ID and digest(previous)==ident
    previous['identity']=ident;assert previous['config']==read(ROOT/'configs/gp_mean_guidance_fivefold_v1.json')
    for rel,h in previous['source_hashes'].items():assert file_hash(ROOT/rel)==h and file_hash(PREVIOUS/'source_snapshot'/rel)==h,rel
    seal=read(PREVIOUS/'all_predictions_sealed.json');assert seal['protocol']==ident
    assert all(file_hash(PREVIOUS/n)==h for n,h in seal['artifacts'].items())
    runtime=read(PREVIOUS/'runtime_validation.json');assert runtime['passed'] and runtime['cost']['mapping']['effective']==6000 and runtime['cost']['gp']['effective']==4200
    checks=read(PREVIOUS/'warmstart_checks.json');oldtasks=read(PREVIOUS/'task_matrix.json');groups=read(PREVIOUS/'groups.json')
    assert digest(oldtasks)==previous['task_matrix_sha'] and digest(groups)==previous['groups_sha']
    origins=read(PREVIOUS/'input_sources.json');assert all(file_hash(Path(n))==h for n,h in origins.items())
    tasks=[];newchecks=[];hashes={};references=[];used=dict(origins)
    for name in ['protocol.json','input_sources.json','warmstart_checks.json','runtime_validation.json','task_matrix.json','groups.json','all_predictions_sealed.json','candidate_identity.json']:
        used[str(PREVIOUS/name)]=file_hash(PREVIOUS/name)
    for row in checks:
        fold,seed=row['fold'],row['seed'];pair=[t for t in oldtasks if t['fold']==fold and t['seed']==seed];assert {t['branch'] for t in pair}=={'C','G'}
        t=pair[0];inp=load(PREVIOUS/t['input']);warm=load(PREVIOUS/t['warmstart']);src=Path(row['source']['model_path']);original=load(src)
        assert original['model_seed']==seed and original['setting']['lambda_f']==0 and original['common_config']['representation_steps']==100
        assert {float(v['step']) for v in original['optimizer']['state'].values()}=={100.}
        assert tensor_hash(original['model'])==tensor_hash(warm['model'])==row['weight_sha256']
        assert warm['source_sha256']==file_hash(src) and set(warm)=={'model','scaler','source_sha256','seed','train_ids','query_ids'}
        assert inp['scaler']==warm['scaler']==original['standardization'] and inp['scaler']['fit_ids']==t['train_ids']
        for part in ['train','query']:
            assert inp[part]['ids']==t[part+'_ids']==original[part+'_ids']
        assert not set(inp['train']['architectures'])&set(inp['query']['architectures'])
        y,se,target32=validate_batch(inp['train'],inp['query'],inp['labels'],inp['scaler'])
        assert torch.equal(target32,inp['target32']) and np.array_equal(se,inp['se'].numpy())
        assert torch.equal(inp['target64'],original['gp_train_targets'])
        assert torch.allclose((inp['se']/inp['scaler']['y_scale'])**2,original['gp_fixed_noise_normalized'],atol=1e-12,rtol=0)
        for cid,lab in inp['labels'].items():
            vals=np.asarray(lab['validation_seeds']);assert len(vals)==3
            assert abs(float(vals.mean())-lab['validation_mean'])<1e-12 and abs(float(vals.std(ddof=1)/np.sqrt(3))-lab['validation_se'])<1e-12
        expected=architecture_groups(inp['train']['ids'],inp['train']['architectures']);assert inp['groups']==expected==groups[fold]['groups']
        m=JointMetricModel(12,seed=seed);m.load_state_dict(warm['model']);m.requires_grad_(True)
        with torch.no_grad():e,h=m(inp['train']['z'],inp['train']['hp'])
        assert torch.equal(e,original['train_embeddings']) and torch.equal(e.double(),original['gp_train_inputs'])
        for rel in [t['input'],t['warmstart']]:
            assert file_hash(PREVIOUS/rel)==previous['input_hashes'][rel];freeze_copy(PREVIOUS/rel,o/rel);hashes[rel]=file_hash(o/rel);used[str(PREVIOUS/rel)]=file_hash(PREVIOUS/rel)
        for name in ['train_labels.json','groups.json']:
            rel=f'inputs/fold{fold}/{name}';freeze_copy(PREVIOUS/rel,o/rel);hashes[rel]=file_hash(o/rel)
        rel=f'evaluation_labels/fold{fold}.json';assert file_hash(PREVIOUS/rel)==previous['query_label_hashes'][rel];freeze_copy(PREVIOUS/rel,o/rel)
        for oldtask in pair:
            folder=PREVIOUS/'trajectories'/oldtask['task_id'];done=read(folder/'completed.json');assert done['protocol']==ident and done['task']==oldtask
            initial=load(folder/'nodes/add000-before_refresh.pt');assert tensor_hash(initial['model'])==row['weight_sha256']
            ref=dict(task=oldtask,start_sha=row['weight_sha256'],config_sha=file_hash(ROOT/'configs/gp_mean_guidance_fivefold_v1.json'),files={})
            for name in ['nodes/add000-before_refresh.pt','mapping/latest.pt','predictions_before_labels.json','completed.json','timing.json','gps/endpoint/teacher.pt']:
                path=folder/name;assert file_hash(path)==seal['artifacts'][str(path.relative_to(PREVIOUS))] if name!='completed.json' else True
                ref['files'][str(path)]=file_hash(path);used[str(path)]=file_hash(path)
                if name in ['predictions_before_labels.json','timing.json']:freeze_copy(path,o/'references'/oldtask['task_id']/name)
            pred=read(folder/'predictions_before_labels.json');assert pred['train']['ids']==t['train_ids'] and pred['query']['ids']==t['query_ids']
            references.append(ref)
        new=copy.deepcopy(t);new.update(task_id=f'fold{fold}-s{seed}-S',branch='S');tasks.append(new)
        newchecks.append(dict(fold=fold,seed=seed,original= str(src),original_sha=file_hash(src),weight_sha=row['weight_sha256'],M100=True,embedding_exact=True,scaler_original=True,labels_SE_verified=True,groups_original=True,counts=[len(t['train_ids']),len(t['query_ids'])]))
    assert len(tasks)==15 and len(references)==30
    for rel in SOURCES:
        if not rel.endswith('_sync.py') and not rel.endswith('_sync_fivefold.py') and not rel.startswith('tests/'):
            used[str(ROOT/rel)]=file_hash(ROOT/rel)
    write(o/'task_matrix.json',tasks);write(o/'warmstart_checks.json',newchecks);write(o/'groups.json',groups);write(o/'missing_before_evaluation.json',[])
    write(o/'old_reference_manifest.json',dict(protocol=ident,sealed_at=utc(),references=references))
    write(o/'input_sources.json',used);freeze_copy(PREVIOUS/'candidate_identity.json',o/'candidate_identity.json')
    write(ROOT/'configs/gp_mean_guidance_sync_fivefold_v1.json',CONFIG)
    write(o/'audit_identity.json',dict(config=CONFIG,A_ids=previous['A_ids'],folds=previous['folds'],task_matrix_sha=digest(tasks),tasks=len(tasks),missing=[],input_hashes=hashes,
      old_protocol=ident,old_reference_sha=file_hash(o/'old_reference_manifest.json'),query_label_hashes={str(p.relative_to(o)):file_hash(p) for p in (o/'evaluation_labels').glob('*.json')},groups_sha=digest(groups),
      prompt_sha=file_hash(o/'input_originals/Codex_Server_GP_Synchronized_Guidance_Prompt.md'),interpretation='A development; same M100, groups, scales, C/G read-only; current support AND query gradients; no search'))
    write(o/'audit_findings.json',dict(M100_starts=15,old_branches=['C','G'],old_input_source_hashes_verified=len(origins),old_sealed_artifacts_verified=len(seal['artifacts']),old_protocol=ident,new_implementation='surrogate.gp_mean_guidance_sync.objective -> guidance -> synchronized_mean -> Cholesky solve with live support and query',old_condition_not_used_for_training=True,new_finalization_fix='retain existing consistent timing; completion can be rebuilt idempotently',missing=[],updates=0))
    print(json.dumps(dict(warmstarts=len(tasks),references=len(references),missing=[],updates=0)),flush=True)

def freeze_protocol(o):
    p=read(o/'audit_identity.json');p['source_hashes']={r:file_hash(ROOT/r) for r in SOURCES}
    for rel in SOURCES:freeze_copy(ROOT/rel,o/'source_snapshot'/rel)
    p['identity']=digest(p);write(o/'protocol.json',p);return p

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

def finalize_timing(folder,timing):
    path=folder/'timing.json'
    if path.exists():
        previous=read(path)
        for key in ['mapping_update_seconds','mapping_backward_seconds','gp_update_seconds','solve_records']:assert previous[key]==timing[key]
        return previous
    write(path,timing);return timing

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
            state=dict(identity=ident,step=0,history=[],committed=[],probes=[],mapping_seconds=0.,backward_seconds=0.,solve_records={},input_hash=input_hash,model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),**state_rng())
            write(folder/'initial_objects.json',actual_objects(m,opt));map_snapshot(o,folder,m,opt,state,data,0,'before_refresh')
        assert input_hash==state['input_hash'];assert {float(v['step']) for v in opt.state.values()}==({float(state['step'])} if state['step'] else set())
        meter=SolveMeter(state['solve_records']);teachers=None;current_block=None
        while state['step']<200:
            step=state['step'];block=step//50;blockstart=50*block
            if current_block!=block:
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
                if step==blockstart:
                    pr=gradient_probe(m,data['train'],data['target32'],data['target64'],data['se'],data['scaler']['y_scale'],data['groups'],teachers,meter);record_probe(state,dict(step=step,phase='after_refresh',block=block,**pr));map_snapshot(o,folder,m,opt,state,data,step,'after_refresh')
            before=time.monotonic();opt.zero_grad(set_to_none=True);e,p=m(data['train']['z'],data['train']['hp'])
            losses=objective(e,p,data['target32'],data['target64'],data['se'],data['scaler']['y_scale'],data['groups'],teachers,True,meter);assert all(torch.isfinite(v) for v in losses.values())
            backward_start=time.monotonic();losses['total'].backward();state['backward_seconds']+=time.monotonic()-backward_start;assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            gid=ledger.update('mapping',task['task_id'],step+1,opt);assert all(torch.isfinite(p).all() for p in m.parameters())
            state['step']=step+1;state['committed'].append(gid);state['history'].append(dict(step=step+1,global_id=gid,block=block if teachers else None,**{k:float(v.detach()) for k,v in losses.items()}));state['mapping_seconds']+=time.monotonic()-before
            state.update(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),**state_rng());save(folder/'mapping/latest.pt',state)
            if state['step'] in CONFIG['nodes']:
                if teachers:record_probe(state,dict(step=state['step'],phase='before_refresh' if state['step']%50==0 else 'within_block',block=block,**gradient_probe(m,data['train'],data['target32'],data['target64'],data['se'],data['scaler']['y_scale'],data['groups'],teachers,meter)))
                map_snapshot(o,folder,m,opt,state,data,state['step'],'before_refresh' if state['step']%50==0 else 'within_block')
        assert tensor_hash({'z':data['train']['z'],'hp':data['train']['hp'],'target64':data['target64']})==input_hash
        with torch.no_grad():train_e,_=m(data['train']['z'],data['train']['hp'])
        meta=dict(task=task['task_id'],name='endpoint',fold=fold,seed=task['seed'],block=None,group=None,mapping_step=200,mapping_sha=tensor_hash(m.state_dict()),support_ids=data['train']['ids'],support_architectures=sorted(set(data['train']['architectures'])),scaler=data['scaler'])
        final=fit_gp(folder/'gps/endpoint',train_e.detach().double(),data['target64'],data['se'],data['scaler']['y_scale'],meta,ledger)
        write(folder/'predictions_before_labels.json',endpoint_prediction(m,data,final));write(folder/'history.json',state['history']);write(folder/'probes.json',state['probes'])
        gp_seconds=sum(load(p)['seconds'] for p in (folder/'gps').glob('*/latest.pt'))
        finalize_timing(folder,dict(mapping_update_seconds=state['mapping_seconds'],mapping_backward_seconds=state['backward_seconds'],gp_update_seconds=gp_seconds,wall_seconds_this_invocation=time.monotonic()-start,includes_IO=True,solve_records=state['solve_records'],timing_note='initial timing kept if finalization is re-entered; invocation overhead reported separately'))
        write(folder/'completed.json',dict(protocol=protocol['identity'],task=task,mapping_steps=200,GP_fits=13,completed_at=utc(),pid=os.getpid(),query_labels_read=False,artifacts={str(p.relative_to(folder)):file_hash(p) for p in folder.rglob('*') if p.is_file() and p.name!='completed.json'}))
        print(json.dumps(dict(completed=task['task_id'],mapping_updates=ledger.next['mapping'],gp_updates=ledger.next['gp'])),flush=True)

def run(o):
    p,tasks=verify_protocol(o);before=costs(o);n=len(list((o/'invocations').glob('*.json')))+1
    for fold in range(5):
        log=o/'logs'/f'run{n:02d}-fold{fold}.log'
        with log.open('w') as out:r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run','--worker',str(fold),'--output',str(o)],stdout=out,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'fold worker failed: {log}')
        print(json.dumps(dict(fold=fold,cost=costs(o))),flush=True)
    after=costs(o);assert after['mapping']['effective']==len(tasks)*200 and after['gp']['effective']==len(tasks)*13*20
    assert len(list((o/'trajectories').glob('*/completed.json')))==len(tasks)
    artifacts={str(f.relative_to(o)):file_hash(f) for f in (o/'trajectories').rglob('*') if f.is_file()}
    if not (o/'all_predictions_sealed.json').exists():write(o/'all_predictions_sealed.json',dict(protocol=p['identity'],time=utc(),full_fivefold=len(tasks)==15,tasks=len(tasks),artifacts=artifacts,cost=after))
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
            a_identity=read(o/'audit_identity.json');assert a_identity['config']==CONFIG
            r=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_gp_mean_guidance_sync.py'],cwd=ROOT,capture_output=True,text=True,env={**os.environ,'GP_SYNC_OUTPUT':str(o)})
            (o/'logs/check.log').write_text(r.stdout+r.stderr)
            if r.returncode:raise RuntimeError(r.stdout+r.stderr)
            p=freeze_protocol(o);write(o/'checks_complete.json',dict(passed=True,protocol=p['identity'],updates=0,test_sha=file_hash(ROOT/'tests/test_gp_mean_guidance_sync.py')));print(r.stdout)
        elif a.run:
            if a.worker is not None:worker(o,a.worker)
            else:
                with (o/'orchestrator.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(o)
        else:subprocess.run([sys.executable,str(ROOT/'analyse/gp_mean_guidance_sync_report.py'),str(o)],check=True)
    except Exception:
        if a.worker is None:
            path=o/'failures'/f'{time.time_ns()}.json';write(path,dict(error=traceback.format_exc(),cost=costs(o),time=utc()))
        raise
if __name__=='__main__':main()
