"""Conditional, bounded ensemble replay. Oracle controller and guarded learners."""
import argparse,copy,fcntl,hashlib,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import torch,numpy as np
from analyse.relative_metric_pilot import read,write,save,load,freeze_copy,digest,file_hash,tensor_hash,utc
from surrogate.fuzzy_metric import JointMetricModel
from surrogate.gp_guidance_replay_engine import rng_state,restore_rng,reload_model,scaler_from_initial,input_tensors,pause_once,append,journal
from analyse.champion_margin_run import install_guard
from surrogate.ensemble_conditional import hash_order,score_ensemble,choose
from analyse.ensemble_conditional_engine import fit

PREVIOUS=ROOT/'results/posthoc/champion-margin-exploration-v1-20260917T051522Z'
CONFIG=ROOT/'configs/ensemble_conditional_search_v1.json'
NEW=['surrogate/ensemble_conditional.py','analyse/ensemble_conditional_engine.py','analyse/ensemble_conditional_rules.py',
     'analyse/ensemble_conditional_run.py','analyse/ensemble_conditional_evaluate.py','analyse/ensemble_conditional_diagnostics.py',
     'tests/test_ensemble_conditional.py','configs/ensemble_conditional_search_v1.json']

def protocol(o):
    p=read(o/'protocol.json');assert digest({k:v for k,v in p.items() if k!='identity'})==p['identity']
    for n,h in p['sources'].items():assert file_hash(ROOT/n)==h,n
    for n,h in p['inputs'].items():assert file_hash(o/n)==h,n
    return p

def prepare(o,prompt):
    o.mkdir(parents=True,exist_ok=True)
    if (o/'draft_protocol.json').exists():return
    old=read(PREVIOUS/'protocol.json');assert old['identity']=='7c41f45ca47ecfe61d91caaeee694431c0a8b6893983877fcf9972f7bad09135'
    info=read(PREVIOUS/'bundle_info.json');assert file_hash(Path(info['path']))==info['sha256']=='5f25f50bf2bdd992eb3d5fb58dd292c2fd0b528a52b3203e3462551a25738444'
    assert read(PREVIOUS/'checks.json')['passed']
    for n,h in old['sources'].items():assert file_hash(ROOT/n)==h,n
    for n,h in old['inputs'].items():assert file_hash(PREVIOUS/n)==h,n
    baseline=read(PREVIOUS/'baseline_reuse_manifest.json');reuse={k:v for k,v in baseline['trajectories'].items() if k.endswith(('C_GREEDY','RANDOM'))}
    for r in reuse.values():
        for n,h in r['files'].items():assert file_hash(Path(n))==h,n
    protected=read(PREVIOUS/'protected_before.json')
    for f in PREVIOUS.rglob('*'):
        if f.is_file() and 'runtime_tmp' not in f.parts:protected[str(f.relative_to(ROOT))]=file_hash(f)
    for n in list(old['sources'])+['AGENTS.md','bo_phase4.py','eval_utils.py']:protected[n]=file_hash(ROOT/n)
    for n,h in protected.items():assert file_hash(ROOT/n)==h
    write(o/'protected_before.json',protected)
    write(o/'workspace_before.json',dict(hostname=platform.node(),root=str(ROOT),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest(),python=sys.version,torch=torch.__version__,numpy=np.__version__,time=utc(),device='cpu',threads=1))
    cfg=read(CONFIG);freeze_copy(CONFIG,o/'config.json');freeze_copy(prompt,o/'REQUEST.md')
    for n in ['features.json','candidate_identity.json']:freeze_copy(PREVIOUS/n,o/n)
    features=read(o/'features.json');ids=features['ids'];assert len(ids)==179 and len(set(features['architectures']))==96
    assert [c['candidate_id'] for c in read(o/'candidate_identity.json')]==ids
    s1={s:dict(initial_ids=r['initial_ids']) for s,r in read(PREVIOUS/'starts.json').items()}
    s2={str(seed):dict(initial_ids=hash_order(ids,f"{cfg['start_prefix']}:{seed}:")[:20]) for seed in cfg['stage2_seeds']}
    write(o/'starts_stage1.json',s1);write(o/'starts_stage2.json',s2)
    weights=[]
    for j in cfg['members']:
        model=JointMetricModel(12,seed=j);assert sum(p.numel() for p in model.parameters())==2337
        fp=o/'initial_weights'/f'member{j}.pt';save(fp,dict(model=model.state_dict(),untrained=True,seed=j,parameters=2337))
        weights.append(dict(member=j,path=str(fp.relative_to(o)),tensor_hash=tensor_hash(model.state_dict()),sha256=file_hash(fp)))
    assert len({r['tensor_hash'] for r in weights})==5
    for seed in cfg['stage1_seeds']:assert tensor_hash(load(PREVIOUS/f'initial_weights/replay-start{seed}-s0.pt')['model'])==weights[0]['tensor_hash']
    write(o/'initial_weight_manifest.json',weights)
    # IDs have been sealed before looking up any stage-two initial label values.
    labels=read(Path(old['A_labels_path']));assert file_hash(Path(old['A_labels_path']))==old['A_labels_sha']
    tasks=[];label_hashes={}
    for stage,starts,methods in [(1,s1,['E_MEAN','E_NOMINATE']),(2,s2,['WINNER','C_GREEDY','RANDOM'])]:
        for seed,row in starts.items():
            name=f'initial_labels/{seed}.json';write(o/name,{c:labels[c] for c in row['initial_ids']});label_hashes[name]=file_hash(o/name)
            for method in methods:tasks.append(dict(name=f's{stage}-{seed}-{method}',stage=stage,seed=int(seed),method=method,initial_ids=row['initial_ids'],label_file=name,
                members=0 if method=='RANDOM' else 1 if method=='C_GREEDY' else 5))
    write(o/'tasks.json',tasks);write(o/'baseline_reuse_manifest.json',dict(trajectories=reuse,previous=str(PREVIOUS),previous_protocol=old['identity'],previous_zip=info))
    write(o/'resource_estimate.json',dict(stage1_minutes=[15,25],stage2_additional_minutes=[35,55],old_11000_seconds=172.897381991148,CPU_threads=1,GNN=0,VAE=0,GP=0))
    write(o/'runtime_resources.json',dict(python=sys.executable,CUDA_available=torch.cuda.is_available(),device_used='cpu',loadavg=Path('/proc/loadavg').read_text().strip(),
        memory=subprocess.check_output(['free','-h'],text=True),GPU_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)))
    inputs=['config.json','REQUEST.md','features.json','candidate_identity.json','starts_stage1.json','starts_stage2.json','tasks.json','initial_weight_manifest.json']+[r['path'] for r in weights]
    p=dict(id=cfg['id'],config=cfg,inputs={n:file_hash(o/n) for n in inputs},initial_label_hashes=label_hashes,prior_sources=old['sources'],
        A_labels_path=old['A_labels_path'],A_labels_sha=old['A_labels_sha'],VAE=old['VAE'],baseline_sha=file_hash(o/'baseline_reuse_manifest.json'),
        source_names=sorted(set(NEW+list(old['sources']))),rules=cfg['rules'],prepared_at=utc())
    write(o/'draft_protocol.json',p)

def seal(o):
    if (o/'protocol.json').exists():return protocol(o)
    assert read(o/'preflight_checks.json')['passed'] and not list((o/'trajectories').glob('*/events/*/member*/latest.pt'))
    p=read(o/'draft_protocol.json');p.pop('prior_sources')
    p['sources']={n:file_hash(ROOT/n) for n in p.pop('source_names')}
    for n in p['sources']:freeze_copy(ROOT/n,o/'source_snapshot'/n)
    p.update(preflight_sha=file_hash(o/'preflight_checks.json'),sealed_at=utc(),formal_updates_before_seal=0)
    p['identity']=digest(p);write(o/'protocol.json',p);return p

def worker(o,name,method,retry_allowance):
    p=read(o/'protocol.json');cfg=p['config'];task=next(t for t in read(o/'tasks.json') if t['name']==name)
    assert method==(task['method'] if task['method']!='WINNER' else method) and method in ['E_MEAN','E_NOMINATE','C_GREEDY','RANDOM']
    d=o/'trajectories'/name;d.mkdir(parents=True,exist_ok=True);install_guard(o,d,task,p);protocol(o)
    if (d/'completed.json').exists():assert read(d/'completed.json')['state_sha']==file_hash(d/'state.pt');return
    assert file_hash(o/task['label_file'])==p['initial_label_hashes'][task['label_file']]
    initial=read(o/task['label_file']);features=read(o/'features.json');scaler=scaler_from_initial(features,task['initial_ids'],initial)
    weights=[load(o/f'initial_weights/member{j}.pt')['model'] for j in range(task['members'])];retry_counter={'remaining':retry_allowance}
    if (d/'state.pt').exists():
        st=load(d/'state.pt');assert st['protocol']==p['identity'] and st['method']==method and st['scaler']==scaler;restore_rng(st['rng'])
    else:
        st=dict(protocol=p['identity'],method=method,known=initial,query_order=[],version=-1,members=weights,scaler=scaler,rng=rng_state(),event_q=None,member_done=[False]*task['members'])
        save(d/'state.pt',st)
    recovery=task['stage']==1 and task['seed']==cfg['stage1_seeds'][0] and method=='E_MEAN'
    while len(st['query_order'])<50:
        q=len(st['query_order']);b=q//5;folder=d/'batches'/f'b{b:02}'
        if task['members'] and q%10==0 and st['version']<q//10:
            event='initial' if q==0 else f'new{q}';steps=cfg['initial_steps'] if q==0 else cfg['update_steps']
            if st['event_q']!=q:st.update(event_q=q,member_done=[False]*task['members']);save(d/'state.pt',st)
            for j in range(task['members']):
                if st['member_done'][j]:continue
                model=reload_model(st['members'][j]);state=fit(d/'events'/event/f'member{j}',model,input_tensors(features,scaler,st['known']),steps,cfg,
                    dict(owner=name,stage=task['stage'],seed=task['seed'],event=event,q=q,member=j),retry_counter,pause_at=2 if recovery and q==0 and j==1 else None)
                st['members'][j]=state['model'];st['member_done'][j]=True;st['rng']=rng_state();save(d/'state.pt',st)
                print(json.dumps(dict(event='member_complete',task=name,q=q,member=j)),flush=True)
            st['version']=q//10;save(d/'state.pt',st)
        if not (folder/'selection.json').exists():
            data=input_tensors(features,scaler,st['known'])
            rows=score_ensemble(st['members'],data) if task['members'] else [dict(candidate_id=c,member_scores=[],member_ranks=[],mean=None,sd=None,mean_rank=None) for c in features['ids'] if c not in st['known']]
            choice=choose(rows,st['known'],method,task['stage'],task['seed'],b,cfg)
            write(folder/'selection.json',dict(**choice,rows=rows,known_ids=sorted(st['known']),known_labels_hash=digest(st['known']),version=st['version'],
                model_hashes=[tensor_hash(w) for w in st['members']],scaler_hash=digest(scaler),remaining_budget=50-q,selected_at=utc(),protocol=p['identity'],method=method))
        selection=read(folder/'selection.json');assert selection['known_ids']==sorted(st['known'])
        pause_once(folder,'selected',recovery and b==0,dict(kind='selected_not_revealed',selection_sha=file_hash(folder/'selection.json'),query_count=q))
        if not (folder/'reveal.json').exists():
            print(json.dumps(dict(event='reveal',task=name,batch=b,selection_sha=file_hash(folder/'selection.json'))),flush=True)
            line=sys.stdin.readline();assert line and json.loads(line)==read(folder/'reveal.json')
        receipt=read(folder/'reveal.json');assert receipt['selection_sha']==file_hash(folder/'selection.json')
        pause_once(folder,'revealed',recovery and b==0,dict(kind='revealed_not_aggregated',selection_sha=receipt['selection_sha'],query_count=q))
        assert set(receipt['labels'])==set(selection['selected_ids']) and not set(receipt['labels'])&set(st['known'])
        st['known'].update(receipt['labels']);st['query_order'].extend(selection['selected_ids']);st['rng']=rng_state();save(d/'state.pt',st)
        write(folder/'aggregate.json',dict(query_count=len(st['query_order']),selection_sha=receipt['selection_sha']))
    assert len(set(st['query_order']))==50 and len(st['known'])==70
    for b in range(10):
        f=d/'batches'/f'b{b:02}';write(f/'aggregate.json',dict(query_count=(b+1)*5,selection_sha=file_hash(f/'selection.json')))
    write(d/'completed.json',dict(task=task,method=method,protocol=p['identity'],state_sha=file_hash(d/'state.pt'),time=utc()))

def budget(o):
    logical=0;intents=[];computed=[];counts={};presentations=0
    for t in read(o/'tasks.json'):
        d=o/'trajectories'/t['name'];steps=0
        for f in (d/'events').glob('*/member*/latest.pt'):
            s=load(f);steps+=s['step'];presentations+=s['step']*len(s['known_ids'])
            for r in journal(f.parent/'physical.jsonl'):(intents if r['phase']=='intent' else computed).append(r)
        queries=sum(read(f)['logical_queries'] for f in (d/'batches').glob('*/reveal.json'));logical+=steps
        if (d/'state.pt').exists():counts[t['name']]=dict(stage=t['stage'],steps=steps,queries=queries,initial_label_presentations=20)
    extra=len(intents)-logical;assert extra>=0 and extra<=512 and logical<=187000
    queries=sum(r['queries'] for r in counts.values());assert queries<=3500
    result=dict(formal_logical_updates=logical,formal_physical_confirmed_steps=len(computed),physical_intents=len(intents),charged_extra_attempts=extra,
        confirmed_repeated_steps=len(computed)-len({r['event_key'] for r in computed}),unconfirmed_intents=len({r['attempt'] for r in intents}-{r['attempt'] for r in computed}),
        new_history_queries=queries,initial_label_presentations=sum(r['initial_label_presentations'] for r in counts.values()),training_sample_presentations=presentations,
        synthetic_steps=sum(1 for f in (o/'validation/synthetic').glob('*/physical.jsonl') for r in journal(f) if r['phase']=='computed'),
        ridge_solves=len(list((o/'ridge').glob('*/solve_receipt.json'))),GNN=0,VAE=0,GP=0,tasks=counts)
    assert result['synthetic_steps']<=32 and result['ridge_solves']<=50
    write(o/'budget_ledger.json',result,immutable=False);return result

def run_stage(o,stage,winner=None):
    p=protocol(o);labels=read(Path(p['A_labels_path']));assert file_hash(Path(p['A_labels_path']))==p['A_labels_sha'];begun=time.monotonic()
    tasks=[t for t in read(o/'tasks.json') if t['stage']==stage]
    for task in tasks:
        method=winner if task['method']=='WINNER' else task['method'];assert method
        if (o/'trajectories'/task['name']/'completed.json').exists():continue
        pauses=0
        while True:
            cost=budget(o);log=o/'logs'/f"{task['name']}-{time.time_ns()}.log";log.parent.mkdir(exist_ok=True);start=time.monotonic()
            with log.open('w') as err:
                proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',task['name'],'--method',method,'--retry-allowance',str(512-cost['charged_extra_attempts']),'--output',str(o)],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
                for line in proc.stdout:
                    req=json.loads(line)
                    if req['event']=='member_complete':write(o/'progress.json',{**req,'time':utc()},immutable=False);continue
                    assert req['event']=='reveal' and req['task']==task['name'];f=o/'trajectories'/task['name']/'batches'/f"b{req['batch']:02}"
                    s=read(f/'selection.json');assert file_hash(f/'selection.json')==req['selection_sha'] and len(s['selected_ids'])==len(set(s['selected_ids']))==5
                    assert not set(s['selected_ids'])&set(s['known_ids'])
                    write(f/'reveal.json',dict(selection_sha=req['selection_sha'],labels={c:labels[c] for c in s['selected_ids']},logical_queries=5,revealed_at=utc()))
                    proc.stdin.write(json.dumps(read(f/'reveal.json'))+'\n');proc.stdin.flush()
                code=proc.wait()
            append(o/'invocations.jsonl',dict(task=task['name'],stage=stage,returncode=code,seconds=time.monotonic()-start,time=utc(),log=str(log)))
            if code==75:pauses+=1;assert pauses<=3;continue
            if code:raise RuntimeError(f'EXECUTION_INVALID: restore original trajectory; see {log}')
            break
        print(json.dumps(dict(completed=task['name'],counts={k:v for k,v in budget(o).items() if k!='tasks'})),flush=True)
    append(o/'stage_times.jsonl',dict(stage=stage,seconds=time.monotonic()-begun,time=utc()))

def run(o):
    from analyse.ensemble_conditional_evaluate import evaluate_stage,validate_stage,seal_stage1,seal_final,export_results
    from analyse.ensemble_conditional_diagnostics import diagnose,ridge
    p=protocol(o)
    with (o/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_stage(o,1);validate_stage(o,1);one=evaluate_stage(o,1);decision=seal_stage1(o,one)
        if decision['winner']:
            run_stage(o,2,decision['winner']);validate_stage(o,2);two=evaluate_stage(o,2);final=seal_final(o,decision,two)
        else:final=seal_final(o,decision,None)
        diagnose(o)
        if final['ridge_required']:ridge(o)
        export_results(o);cost=budget(o)
        expected=(187000,3500) if decision['winner'] else (55000,500)
        assert (cost['formal_logical_updates'],cost['new_history_queries'])==expected
        assert cost['ridge_solves']==(50 if final['ridge_required'] else 0)
        write(o/'all_completed.json',dict(protocol=p['identity'],final_status=final['status'],time=utc(),budget=cost))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--prompt',type=Path);ap.add_argument('--method');ap.add_argument('--retry-allowance',type=int,default=512)
    g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--seal',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker')
    a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);o=a.output.resolve()
    if a.prepare:prepare(o,a.prompt)
    elif a.seal:print(json.dumps({'identity':seal(o)['identity']}))
    elif a.worker:worker(o,a.worker,a.method,a.retry_allowance)
    else:
        try:run(o)
        except Exception:write(o/'failures'/f'{time.time_ns()}.json',dict(status='EXECUTION_INVALID',error=traceback.format_exc(),time=utc()));raise

if __name__=='__main__':main()
