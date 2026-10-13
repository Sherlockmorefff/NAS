"""Two-stage, A-only margin repair then conditional exploration replay."""
import argparse,copy,fcntl,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import torch
from analyse.relative_metric_pilot import read,write,save,load,freeze_copy,digest,file_hash,tensor_hash,utc
from surrogate.gp_guidance_replay_engine import rng_state,restore_rng,reload_model,scaler_from_initial,input_tensors,score_candidates,pause_once,append,journal
from analyse.fuzzy_metric_offline_core import _label_arrays
from analyse.champion_margin_engine import fit
from surrogate.champion_margin import select_batch

BASE=ROOT/'results/posthoc/budget-champion-ranking-v1'
AUDIT=ROOT/'results/posthoc/current-selection-audit-20260916T143140Z'
CONFIG=ROOT/'configs/champion_margin_exploration_v1.json'
SOURCES=['analyse/champion_margin_run.py','analyse/champion_margin_engine.py','analyse/champion_margin_analysis.py',
 'surrogate/champion_margin.py','tests/test_champion_margin.py','configs/champion_margin_exploration_v1.json']


def verify_protocol(o):
    p=read(o/'protocol.json');identity=p.pop('identity');assert digest(p)==identity;p['identity']=identity
    assert read(CONFIG)==p['config']
    for n,h in p['sources'].items():assert file_hash(ROOT/n)==h,n
    for n,h in p['inputs'].items():assert file_hash(o/n)==h,n
    return p


def prepare(o,prompt):
    if (o/'protocol.json').exists():verify_protocol(o);return
    o.mkdir(parents=True,exist_ok=True)
    facts=read(AUDIT/'runtime_facts.json');old=read(BASE/'protocol.json');oldid=old['identity']
    assert digest({k:v for k,v in old.items() if k!='identity'})==oldid
    for n,h in old['sources'].items():
        assert file_hash(ROOT/n)==file_hash(BASE/'source_snapshot'/n)==h==facts['source_versions'][n]['current'],n
    for n,h in old['inputs'].items():assert file_hash(BASE/n)==h,n
    for n,h in facts['data_fingerprints'].items():assert file_hash(BASE/n)==h,n
    assert file_hash(Path(old['VAE_manifest']['checkpoint']))==old['VAE_manifest']['checkpoint_sha256']
    assert read(AUDIT/'checks.json')['passed']
    cfg=read(CONFIG);starts=read(BASE/'starts.json');features=read(BASE/'features.json');candidates=read(BASE/'candidate_identity.json')
    assert len(features['ids'])==179 and len(set(features['architectures']))==96
    assert sorted(map(int,starts))==cfg['search_seeds']
    assert [c['candidate_id'] for c in candidates]==features['ids']
    labels=read(BASE/'evaluation_A.json');_label_arrays(features['ids'],labels)
    assert file_hash(BASE/'evaluation_A.json')==old['A_labels_sha']
    protected=read(AUDIT/'protected_before.json')
    for n,h in protected.items():assert file_hash(ROOT/n)==h,n
    for path in AUDIT.rglob('*'):
        if path.is_file():protected[str(path.relative_to(ROOT))]=file_hash(path)
    write(o/'protected_before.json',protected)
    write(o/'workspace_before.json',dict(hostname=platform.node(),repo=str(ROOT),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        tracked_diff_sha=digest(subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True)),python=sys.version,torch=torch.__version__,device='cpu',threads=1,time=utc()))
    freeze_copy(prompt,o/'REQUEST.md');freeze_copy(CONFIG,o/'config.json')
    for n in ['features.json','candidate_identity.json','starts.json']:freeze_copy(BASE/n,o/n)
    initial_hashes={};tasks=[];reuse={};input_names=['features.json','candidate_identity.json','starts.json','config.json','REQUEST.md']
    aliases={'C_HEAD':'C_GREEDY','BR_HEAD':'BR_FIXED_GREEDY','RANDOM':'RANDOM'}
    for seed in cfg['search_seeds']:
        initial=starts[str(seed)]['initial_ids'];assert len(initial)==len(set(initial))==20
        wp=f'initial_weights/replay-start{seed}-s0.pt';weight=load(BASE/wp);assert weight['untrained'] and weight['seed']==0
        freeze_copy(BASE/wp,o/wp);input_names.append(wp)
        label_file=f'initial_labels/{seed}.json';write(o/label_file,{c:labels[c] for c in initial});initial_hashes[label_file]=file_hash(o/label_file)
        for method,alias in aliases.items():
            d=BASE/'replay'/f'start{seed}-s0-{method}';done=read(d/'completed.json')
            assert done['protocol']==oldid and done['task']['initial_ids']==initial
            for n,h in done['files'].items():assert file_hash(d/n)==h
            if method!='RANDOM':assert tensor_hash(load(d/'events/initial/step000.pt')['model'])==tensor_hash(weight['model'])
            files=[d/'completed.json',d/'state.pt']+sorted((d/'batches').glob('*/selection.json'))+sorted((d/'batches').glob('*/reveal.json'))
            assert len(list((d/'batches').glob('*/selection.json')))==10
            reuse[f'{seed}-{alias}']=dict(directory=str(d),files={str(p):file_hash(p) for p in files},new_training_steps=0,new_queries=0)
        for stage in [1,2]:
            tasks.append(dict(name=f'stage{stage}-start{seed}-s0',stage=stage,search_seed=seed,model_seed=0,task_id=f'start{seed}-s0',
                initial_ids=initial,weight_file=wp,label_file=label_file,loss='gap' if stage==1 else 'selected_by_stage1',strategy=cfg['stage_strategies'][str(stage)]))
    write(o/'tasks.json',tasks);input_names.append('tasks.json')
    write(o/'baseline_reuse_manifest.json',dict(old_protocol=oldid,source_equivalence=True,files={str(BASE/n):h for n,h in old['inputs'].items()},trajectories=reuse,
        audit_files={n:file_hash(AUDIT/n) for n in ['CURRENT_SELECTION_AUDIT.md','runtime_facts.json','loss_and_selection_source.md','checks.json']}))
    source_names=sorted(set(SOURCES+list(old['sources'])))
    for n in source_names:freeze_copy(ROOT/n,o/'source_snapshot'/n)
    timing=read(BASE/'analysis/timing.json')
    write(o/'resource_estimate.json',dict(old_replay_compute_and_save_seconds=sum(v['training_compute_seconds']+v['training_save_state_seconds'] for k,v in timing['by_stage_method'].items() if k.startswith('replay/')),
        planned_updates=11000,planned_queries=500,expected_wall_minutes=[3,8],CPU_threads=1,GNN=0,VAE=0,GP=0,basis=str(BASE/'analysis/timing.json')))
    p=dict(id=cfg['id'],config=cfg,baseline_protocol=oldid,A_ids=features['ids'],A_labels_path=str(BASE/'evaluation_A.json'),A_labels_sha=file_hash(BASE/'evaluation_A.json'),
        inputs={n:file_hash(o/n) for n in input_names},initial_label_hashes=initial_hashes,sources={n:file_hash(ROOT/n) for n in source_names},
        decision_rule=cfg['decision_rule'],baseline_manifest_sha=file_hash(o/'baseline_reuse_manifest.json'),VAE=old['VAE_manifest'],sealed_at=utc(),formal_updates_before_seal=0)
    p['identity']=digest(p);write(o/'protocol.json',p)
    write(o/'data_checks.json',dict(passed=True,A=179,architectures=96,starts=cfg['search_seeds'],all_initial_untrained=True,baseline_source_hashes_match_audit=True,B_test_opened=0))


def install_guard(o,folder,task,p):
    import tempfile
    runtime_tmp=folder/'runtime_tmp';runtime_tmp.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(runtime_tmp.resolve())
    allowed={str((o/n).resolve()) for n in p['inputs']}
    allowed|={str((o/task['label_file']).resolve()),str((o/'protocol.json').resolve())}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing and not path.is_relative_to(folder):raise PermissionError('learner writes only own trajectory: '+str(path))
        protected=any(path.is_relative_to(ROOT/n) for n in ['results','checkpoints','data','Cora'])
        if protected and str(path) not in allowed and not path.is_relative_to(folder):raise PermissionError('unrevealed label/other trajectory denied: '+str(path))
    sys.addaudithook(hook)


def worker(o,name,selected_loss):
    p=read(o/'protocol.json');task=next(t for t in read(o/'tasks.json') if t['name']==name);cfg=p['config']
    folder=o/'trajectories'/name;folder.mkdir(parents=True,exist_ok=True)
    install_guard(o,folder,task,p);verify_protocol(o)
    if (folder/'completed.json').exists():
        done=read(folder/'completed.json');assert done['protocol']==p['identity'] and done['state_sha']==file_hash(folder/'state.pt');return
    if task['stage']==1:assert selected_loss=='gap'
    assert selected_loss in ['C','gap'];mode=cfg['margin_modes'][selected_loss];weight=cfg['ranking_weight'] if selected_loss=='gap' else 0.
    assert file_hash(o/task['label_file'])==p['initial_label_hashes'][task['label_file']]
    initial=read(o/task['label_file']);features=read(o/'features.json');scaler=scaler_from_initial(features,task['initial_ids'],initial)
    weights=load(o/task['weight_file']);assert weights['untrained'] and weights['seed']==task['model_seed']
    def train(model,known,q):
        event='initial' if q==0 else f'new{q}';steps=cfg['initial_steps'] if q==0 else cfg['update_steps']
        rho=(cfg['query_budget']-q)/(len(features['ids'])-cfg['initial_count']-q)
        return fit(folder/'events'/event,model,input_tensors(features,scaler,known),steps,cfg,mode,weight,rho,
            dict(owner=name,task_id=task['task_id'],update_event=event,stage='replay'))
    if (folder/'state.pt').exists():
        state=load(folder/'state.pt');assert state['protocol']==p['identity'] and state['loss']==selected_loss and state['scaler']==scaler;restore_rng(state['rng'])
    else:
        model=reload_model(weights['model']);train(model,initial,0)
        state=dict(protocol=p['identity'],loss=selected_loss,strategy=task['strategy'],known=initial,query_order=[],version=0,scaler=scaler,
            model=copy.deepcopy(model.state_dict()),rng=rng_state(),initial_ids=task['initial_ids'],initial_weight_hash=tensor_hash(weights['model']))
        save(folder/'state.pt',state)
    while len(state['query_order'])<cfg['query_budget']:
        q=len(state['query_order']);batch=q//cfg['batch_size'];d=folder/'batches'/f'b{batch:02}'
        assert len(state['known'])==cfg['initial_count']+q
        if not (d/'selection.json').exists():
            model=reload_model(state['model'])
            if q>0 and q%cfg['update_interval']==0 and state['version']<q//cfg['update_interval']:
                train(model,state['known'],q);state.update(model=copy.deepcopy(model.state_dict()),version=q//cfg['update_interval'],rng=rng_state());save(folder/'state.pt',state)
            rows,cache,seconds=score_candidates(model,input_tensors(features,scaler,state['known']))
            assert len(rows)==len(features['ids'])-len(state['known'])
            choice=select_batch(rows,state['known'],task['search_seed'],batch,cfg,task['strategy'])
            write(d/'selection.json',dict(rows=rows,**choice,known_ids=sorted(state['known']),known_labels_hash=digest(state['known']),query_count_before=q,
                remaining_budget=cfg['query_budget']-q,version=state['version'],scaler_hash=digest(scaler),model_hash=tensor_hash(state['model']),
                selected_at=utc(),protocol=p['identity'],loss=selected_loss,score_seconds=seconds))
        selection=read(d/'selection.json');assert selection['known_ids']==sorted(state['known']) and selection['model_hash']==tensor_hash(state['model'])
        pause_once(d,'selected',task['search_seed']==cfg['search_seeds'][0] and batch==0,
            dict(kind='selected_not_revealed',query_count=q,selection_sha=file_hash(d/'selection.json')))
        if not (d/'reveal.json').exists():
            print(json.dumps(dict(event='reveal',task=name,batch=batch,selection_sha=file_hash(d/'selection.json'))),flush=True)
            line=sys.stdin.readline()
            if not line:raise RuntimeError('controller unavailable; selection retained')
            assert json.loads(line)==read(d/'reveal.json')
        receipt=read(d/'reveal.json');assert receipt['selection_sha']==file_hash(d/'selection.json')
        pause_once(d,'revealed',task['stage']==2 and task['search_seed']==cfg['search_seeds'][0] and batch==0,
            dict(kind='revealed_not_aggregated',query_count=q,selection_sha=receipt['selection_sha']))
        assert set(receipt['labels'])==set(selection['selected_ids']) and not set(receipt['labels'])&set(state['known'])
        _label_arrays(sorted(receipt['labels']),receipt['labels'])
        state['known'].update(receipt['labels']);state['query_order'].extend(selection['selected_ids']);state['rng']=rng_state();save(folder/'state.pt',state)
        write(d/'aggregated.json',dict(query_count=len(state['query_order']),selection_sha=receipt['selection_sha']))
    assert len(set(state['query_order']))==cfg['query_budget'] and len(state['known'])==70
    # Repair bookkeeping after a crash between the authoritative state and a marker.
    for batch in range(cfg['query_budget']//cfg['batch_size']):
        d=folder/'batches'/f'b{batch:02}';write(d/'aggregated.json',dict(query_count=(batch+1)*cfg['batch_size'],selection_sha=file_hash(d/'selection.json')))
    write(folder/'completed.json',dict(protocol=p['identity'],task=name,loss=selected_loss,query_count=len(state['query_order']),state_sha=file_hash(folder/'state.pt'),time=utc()))


def budget(o):
    logical=0;intents=[];computed=[];queries=0;tasks={}
    for t in read(o/'tasks.json'):
        d=o/'trajectories'/t['name'];n=0
        for f in (d/'events').glob('*/latest.pt'):
            s=load(f);n+=s['step']
            for r in journal(f.parent/'physical_attempts.jsonl'):(intents if r['phase']=='intent' else computed).append(r)
        q=sum(read(f)['logical_queries'] for f in (d/'batches').glob('*/reveal.json'));logical+=n;queries+=q;tasks[t['name']]=dict(logical_updates=n,queries=q)
    unique={r['event_key'] for r in computed};attemptids={r['attempt'] for r in computed};pending=[r for r in intents if r['attempt'] not in attemptids]
    result=dict(formal_logical_updates=logical,formal_physical_confirmed_steps=len(computed),physical_intents=len(intents),unconfirmed_physical_intents=len(pending),
        repeated_physical_steps=len(computed)-len(unique),new_history_queries=queries,tasks=tasks,
        synthetic_steps=len(journal(o/'validation/synthetic_steps.jsonl')),GNN_trainings=0,VAE_trainings=0,GP_trainings=0,
        planned_formal_updates=11000,planned_queries=500,synthetic_limit=32)
    assert logical<=11000 and queries<=500 and result['synthetic_steps']<=32
    write(o/'budget_ledger.json',result,immutable=False);return result


def run(o):
    from analyse.champion_margin_analysis import decide
    p=verify_protocol(o);assert read(o/'checks.json')['preflight_passed'];labels=read(Path(p['A_labels_path']));assert file_hash(Path(p['A_labels_path']))==p['A_labels_sha']
    all_tasks=read(o/'tasks.json');cfg=p['config'];begun=time.monotonic();before=budget(o)
    with (o/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in [1,2]:
            loss='gap' if stage==1 else decide(o)['selected_loss']
            for task in [t for t in all_tasks if t['stage']==stage]:
                retries=0
                while True:
                    stamp=time.time_ns();log=o/'logs'/f"{task['name']}-{stamp}.log";log.parent.mkdir(exist_ok=True);start=time.monotonic()
                    with log.open('w') as err:
                        proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',task['name'],'--selected-loss',loss,'--output',str(o)],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
                        for line in proc.stdout:
                            req=json.loads(line);assert req['event']=='reveal' and req['task']==task['name']
                            d=o/'trajectories'/task['name']/'batches'/f"b{req['batch']:02}";s=read(d/'selection.json');assert file_hash(d/'selection.json')==req['selection_sha']
                            ids=s['selected_ids'];assert len(ids)==len(set(ids))==cfg['batch_size'] and not set(ids)&set(s['known_ids']) and set(ids)<=set(p['A_ids'])
                            if not (d/'reveal.json').exists():write(d/'reveal.json',dict(labels={c:labels[c] for c in ids},selection_sha=req['selection_sha'],revealed_at=utc(),logical_queries=len(ids),event_key=f"{task['name']}/batch{req['batch']}"))
                            proc.stdin.write(json.dumps(read(d/'reveal.json'))+'\n');proc.stdin.flush()
                        code=proc.wait()
                    append(o/'invocations.jsonl',dict(task=task['name'],stage=stage,returncode=code,seconds=time.monotonic()-start,log=str(log),time=utc()))
                    if code==75:
                        retries+=1;assert retries<=3;continue
                    if code:raise RuntimeError(f'worker stopped; recover same trajectory: {log}')
                    break
                print(json.dumps(dict(completed=task['name'],budget=budget(o))),flush=True)
            if stage==1:print(json.dumps(dict(stage1_decision=decide(o))),flush=True)
        cost=budget(o);assert cost['formal_logical_updates']==11000 and cost['new_history_queries']==500
        append(o/'runs.jsonl',dict(before=before,after=cost,seconds=time.monotonic()-begun,time=utc()))
        write(o/'all_completed.json',dict(trajectories=10,protocol=p['identity'],time=utc(),budget=cost))


def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--prompt',type=Path);a.add_argument('--selected-loss',choices=['gap','C'],default='gap')
    g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker')
    args=a.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);o=args.output.resolve()
    if args.prepare:prepare(o,args.prompt)
    elif args.worker:worker(o,args.worker,args.selected_loss)
    else:
        try:run(o)
        except Exception:
            write(o/'failures'/f'{time.time_ns()}.json',dict(error=traceback.format_exc(),time=utc()));raise


if __name__=='__main__':main()
