"""Fixed offline then sequential A-only budget-ranking experiment."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from analyse.relative_metric_pilot import read,write,save,load,freeze_copy,digest,file_hash,tensor_hash,utc
from surrogate.gp_guidance_replay_engine import Ledger,journal,append,rng_state,restore_rng,reload_model,scaler_from_initial,input_tensors,score_candidates,pause_once
from analyse.fuzzy_metric_offline_core import _label_arrays
from analyse.gp_guidance_development_replay import id_order
from analyse.budget_champion_engine import fit
from surrogate.fuzzy_metric import JointMetricModel

DEFAULT=ROOT/'results/posthoc/budget-champion-ranking-v1'
PRIOR=ROOT/'results/posthoc/gp-guidance-development-replay-20260916-v1'
OLD=ROOT/'results/posthoc/fuzzy-metric-stage2-20260910-v1'
CONFIG=ROOT/'configs/budget_champion_ranking_v1.json'
SEEDS=list(range(2026091600,2026091605))
METHODS=['C_HEAD','BR_HEAD','RANDOM']
SOURCES=['analyse/budget_champion_run.py','analyse/budget_champion_engine.py','surrogate/budget_champion_ranking.py',
 'tests/test_budget_champion_ranking.py','configs/budget_champion_ranking_v1.json','surrogate/gp_guidance_replay_engine.py',
 'analyse/relative_metric_pilot.py','surrogate/fuzzy_metric.py','surrogate/relative_metric.py','analyse/fuzzy_metric_offline_core.py',
 'analyse/gp_guidance_development_replay.py']


def check_protocol(o):
    p=read(o/'protocol.json');identity=p.pop('identity');assert digest(p)==identity;p['identity']=identity
    assert p['config']==read(CONFIG)
    for name,h in p['sources'].items():assert file_hash(ROOT/name)==h,name
    for name,h in p['inputs'].items():assert file_hash(o/name)==h,name
    return p


def labels_for_controller(p):
    path=OLD/'A_validation.json';assert file_hash(path)==p['A_labels_sha']
    labs=read(path);assert sorted(labs)==p['A_ids']
    assert digest(labs)==read(OLD/'label_boundary.json')['exported_hash']
    _label_arrays(sorted(labs),labs)
    return labs


def provision(o,p):
    labels=labels_for_controller(p)
    for task in read(o/'tasks.json'):
        write(o/task['label_file'],{cid:labels[cid] for cid in task['initial_ids']})


def prepare(o):
    if (o/'protocol.json').exists():
        p=check_protocol(o);provision(o,p);return
    old=read(OLD/'protocol.json');part=read(OLD/'partition.json');assert digest(part)==old['partition_hash']
    prior=read(PRIOR/'protocol.json');features=read(PRIOR/'features.json');candidates=read(PRIOR/'candidate_identity.json')
    assert prior['identity']=='bfdb424ceac9dd3ba57ca9092ad926aec14160cd5f5459f555dc285887042e32'
    assert file_hash(PRIOR/'features.json')==prior['features_sha'] and file_hash(PRIOR/'candidate_identity.json')==prior['candidate_sha']
    for name,h in prior['source_hashes'].items():assert file_hash(ROOT/name)==h,name
    prep=ROOT/old['config']['prepared'];manifest=read(prep/'manifest.json')
    assert manifest==old['prepared_manifest']==prior['VAE_manifest']
    assert file_hash(Path(manifest['checkpoint']))==manifest['checkpoint_sha256'] and manifest['vae_performance_supervision'] is False
    pool=read(prep/'candidates.json');base=read(prep/'base_encoding.json')
    assert digest(pool)==manifest['pool_hash'] and digest(base)==manifest['base_encoding_hash']
    ids=sorted(part['A_ids']);assert features['ids']==ids and len(ids)==179
    assert len(set(features['architectures']))==96
    assert candidates==sorted([c for c in pool if c['candidate_id'] in ids],key=lambda c:c['candidate_id'])
    write(o/'features.json',features);write(o/'candidate_identity.json',candidates)
    starts={};tasks=[];weights=[]
    for fold,split in enumerate(part['folds']):
        tr,qu=split['train_ids'],split['query_ids'];assert len(qu)>10 and not set(tr)&set(qu)
        byid=dict(zip(features['ids'],features['architectures']))
        assert not {byid[c] for c in tr}&{byid[c] for c in qu}
        for seed in [0,1,2]:
            task_id=f'fold{fold}-s{seed}';wp=f'initial_weights/offline-{task_id}.pt'
            save(o/wp,dict(model=JointMetricModel(12,seed=seed).state_dict(),seed=seed,untrained=True));weights.append(wp)
            for method in METHODS[:2]:
                tasks.append(dict(name=f'{task_id}-{method}',stage='offline',task_id=task_id,fold=fold,model_seed=seed,
                    method=method,initial_ids=tr,query_ids=qu,rho=10/len(qu),weight_file=wp,label_file=f'train_labels/offline-fold{fold}.json'))
    oldstarts=read(PRIOR/'starts.json')
    for seed in SEEDS:
        fifty=oldstarts[str(seed)]['initial_ids'];assert fifty==id_order(ids,seed,'init')[:50]
        for method in ['C_GP','S_GP','C_HEAD','RANDOM']:
            assert read(PRIOR/'trajectories'/f'{seed}-{method}'/'completed.json')['initial_ids']==fifty
        twenty=fifty[:20]
        order=sorted(set(ids)-set(twenty),key=lambda c:(hashlib.sha256(f'budget_champion_v1:random:{seed}:{c}'.encode()).hexdigest(),c))
        starts[str(seed)]=dict(initial_ids=twenty,old_initial50=fifty,random_order=order)
        task_id=f'start{seed}-s0';wp=f'initial_weights/replay-{task_id}.pt'
        save(o/wp,dict(model=JointMetricModel(12,seed=0).state_dict(),seed=0,untrained=True));weights.append(wp)
        for method in METHODS:
            tasks.append(dict(name=f'{task_id}-{method}',stage='replay',task_id=task_id,search_seed=seed,model_seed=0,method=method,
                initial_ids=twenty,weight_file=wp,label_file=f'train_labels/replay-{seed}.json'))
    write(o/'starts.json',starts);write(o/'folds.json',part['folds']);write(o/'tasks.json',tasks)
    for name in SOURCES:freeze_copy(ROOT/name,o/'source_snapshot'/name)
    inputs={name:file_hash(o/name) for name in ['features.json','candidate_identity.json','starts.json','folds.json','tasks.json']+weights}
    p=dict(id='budget-champion-ranking-v1',config=read(CONFIG),A_ids=ids,A_labels_sha=file_hash(OLD/'A_validation.json'),
        sources={name:file_hash(ROOT/name) for name in SOURCES},inputs=inputs,old_partition_sha=file_hash(OLD/'partition.json'),
        old_replay_protocol=prior['identity'],VAE_manifest=manifest,prompt_sha=file_hash(o/'input_originals/Prompt.md'),
        sealed_at=utc(),labels_given_to_learners_before_seal=0,scope='A development; offline fivefold then fixed initial20 replay')
    p['identity']=digest(p);write(o/'protocol.json',p)
    provision(o,p)
    write(o/'data_checks.json',dict(A=179,architectures=96,fold_query_counts=[len(f['query_ids']) for f in part['folds']],
        old_initial50_order_verified_all20_trajectories=True,initial20_rule='first20 of original sealed sequence',
        VAE_unmodified=True,B_parsed=0,test_parsed=0,pretraining_limit=manifest['pretraining_membership']))


def guard(o,folder,task):
    allowed={str((o/n).resolve()) for n in ['features.json','starts.json',task['label_file'],task['weight_file']]}
    prefixes=[str(folder.resolve()),str((o/'ledger').resolve())]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=str(Path(os.fsdecode(args[0])).resolve())
        protected=any(path.startswith(str(ROOT/n)+'/') for n in ['results','checkpoints','data'])
        if protected and path not in allowed and not any(path==x or path.startswith(x+'/') for x in prefixes):
            raise PermissionError('forbidden hidden labels, other trajectory, or trained checkpoint: '+path)
    sys.addaudithook(hook)


def worker(o,name):
    p=check_protocol(o);task=next(t for t in read(o/'tasks.json') if t['name']==name)
    folder=o/task['stage']/name;folder.mkdir(parents=True,exist_ok=True)
    guard(o,folder,task)
    if (folder/'completed.json').exists():
        done=read(folder/'completed.json');assert done['protocol']==p['identity']
        for f,h in done['files'].items():assert file_hash(folder/f)==h
        return
    features=read(o/'features.json');assert set(features)=={'ids','architectures','z','hp'}
    initial=read(o/task['label_file']);assert set(initial)==set(task['initial_ids'])
    scaler=scaler_from_initial(features,task['initial_ids'],initial)
    weights=load(o/task['weight_file']);assert weights['untrained'] and weights['seed']==task['model_seed']
    ledger=Ledger(o/'ledger',p['identity'],p['config']['limits'])
    method=task['method'];weight=.1 if method=='BR_HEAD' else 0.
    def train(model,known,event,steps,rho):
        data=input_tensors(features,scaler,known)
        meta=dict(stage=task['stage'],owner=name,task_id=task['task_id'],update_event=event)
        return fit(folder/'events'/event,model,data,steps,weight,rho,meta,ledger,
            interrupt=name=='fold0-s0-BR_HEAD' and event=='initial')
    if task['stage']=='offline':
        model=reload_model(weights['model']);train(model,initial,'initial',300,task['rho'])
        rows,cache,seconds=score_candidates(model,input_tensors(features,scaler,initial))
        assert {r['candidate_id'] for r in rows}==set(task['query_ids'])
        save(folder/'coordinates.pt',cache)
        write(folder/'predictions.json',dict(rows=rows,known_ids=sorted(initial),scaler=scaler,
            model_hash=tensor_hash(model.state_dict()),initial_weight_hash=tensor_hash(weights['model']),score_seconds=seconds,
            predictions_sealed_at=utc(),protocol=p['identity']))
    else:
        if (folder/'state.pt').exists():
            state=load(folder/'state.pt');assert state['protocol']==p['identity'] and state['scaler']==scaler
            restore_rng(state['rng'])
        else:
            model=None if method=='RANDOM' else reload_model(weights['model'])
            if model is not None:train(model,initial,'initial',300,50/159)
            state=dict(protocol=p['identity'],known=initial,query_order=[],version=0,scaler=scaler,rng=rng_state(),
                model={} if model is None else copy.deepcopy(model.state_dict()),initial_ids=task['initial_ids'])
            save(folder/'state.pt',state)
        for prior_batch in range(len(state['query_order'])//5):
            d=folder/'batches'/f'b{prior_batch:02}'
            if not (d/'aggregated.json').exists():
                write(d/'aggregated.json',dict(query_count=5*(prior_batch+1),selected_ids=state['query_order'][5*prior_batch:5*prior_batch+5]))
        while len(state['query_order'])<50:
            n=len(state['query_order']);batch=n//5;d=folder/'batches'/f'b{batch:02}'
            assert len(state['known'])==20+n
            if not (d/'selection.json').exists():
                started=time.monotonic()
                if method=='RANDOM':
                    order=read(o/'starts.json')[str(task['search_seed'])]['random_order']
                    rows=[dict(candidate_id=c,score=None) for c in order if c not in state['known']]
                    seconds=time.monotonic()-started
                else:
                    model=reload_model(state['model'])
                    if n in [10,20,30,40] and state['version']<n//10:
                        train(model,state['known'],f'new{n}',200,(50-n)/(159-n))
                        state.update(model=copy.deepcopy(model.state_dict()),version=n//10,rng=rng_state());save(folder/'state.pt',state)
                    rows,cache,seconds=score_candidates(model,input_tensors(features,scaler,state['known']))
                    cache['version']=state['version'];save(d/'coordinates.pt',cache)
                write(d/'selection.json',dict(rows=rows,selected_ids=[r['candidate_id'] for r in rows[:5]],
                    known_ids=sorted(state['known']),known_labels_hash=digest(state['known']),query_count_before=n,
                    remaining_budget=50-n,version=state['version'],scaler_hash=digest(scaler),score_seconds=seconds,
                    model_hash=None if method=='RANDOM' else tensor_hash(state['model']),selected_at=utc(),protocol=p['identity']))
            selection=read(d/'selection.json');assert selection['known_ids']==sorted(state['known'])
            pause_once(d,'selected',name=='start2026091600-s0-BR_HEAD' and batch==0,
                dict(kind='selected_not_revealed',query_count=n,selection_sha=file_hash(d/'selection.json')))
            if not (d/'reveal.json').exists():
                print(json.dumps(dict(event='reveal',task=name,batch=batch,selection_sha=file_hash(d/'selection.json'))),flush=True)
                response=json.loads(sys.stdin.readline());assert response==read(d/'reveal.json')
            receipt=read(d/'reveal.json');assert receipt['selection_sha']==file_hash(d/'selection.json')
            assert set(receipt['labels'])==set(selection['selected_ids']) and not set(receipt['labels'])&set(state['known'])
            _label_arrays(sorted(receipt['labels']),receipt['labels'])
            state['known'].update(receipt['labels']);state['query_order'].extend(selection['selected_ids']);state['rng']=rng_state()
            save(folder/'state.pt',state)
            write(d/'aggregated.json',dict(query_count=n+5,selected_ids=selection['selected_ids']))
        assert len(state['known'])==70 and len(set(state['query_order']))==50
    paths=['predictions.json','events/initial/latest.pt'] if task['stage']=='offline' else ['state.pt']
    write(folder/'completed.json',dict(protocol=p['identity'],task=task,files={n:file_hash(folder/n) for n in paths},completed_at=utc()))


def costs(o):
    intents=journal(o/'ledger/mapping-intents.jsonl');receipts=journal(o/'ledger/mapping-receipts.jsonl')
    return dict(attempted=len(intents),receipts=len(receipts),queries=5*len(list((o/'replay').glob('*/batches/*/reveal.json'))),
        GP_updates=len(journal(o/'ledger/gp-intents.jsonl')),GP_fits=len(journal(o/'ledger/gp_fits-intents.jsonl')))


def run(o):
    p=check_protocol(o);assert read(o/'checks_complete.json')['passed'];labels=labels_for_controller(p)
    tasks=read(o/'tasks.json');start=time.monotonic();before=costs(o)
    with (o/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for task in tasks:
            retries=0
            while True:
                invocation=time.monotonic();log=o/'logs'/f'{task["stage"]}-{task["name"]}-{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
                with log.open('w') as err:
                    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',task['name'],'--output',str(o)],
                        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,
                        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
                    for line in proc.stdout:
                        req=json.loads(line);assert req['event']=='reveal' and req['task']==task['name'] and task['stage']=='replay'
                        d=o/'replay'/task['name']/'batches'/f'b{req["batch"]:02}';selection=read(d/'selection.json')
                        assert file_hash(d/'selection.json')==req['selection_sha']
                        ids=selection['selected_ids'];assert len(ids)==len(set(ids))==5 and not set(ids)&set(selection['known_ids'])
                        assert set(ids)<=set(p['A_ids'])
                        if not (d/'reveal.json').exists():
                            t=time.monotonic();write(d/'reveal.json',dict(labels={c:labels[c] for c in ids},selection_sha=req['selection_sha'],revealed_at=utc(),logical_queries=5))
                            write(d/'oracle_timing.json',dict(seconds=time.monotonic()-t))
                        proc.stdin.write(json.dumps(read(d/'reveal.json'))+'\n');proc.stdin.flush()
                    code=proc.wait()
                append(o/'invocations.jsonl',dict(task=task['name'],stage=task['stage'],returncode=code,seconds=time.monotonic()-invocation,time=utc()))
                if code==75:
                    retries+=1;assert retries<=3;continue
                if code:raise RuntimeError(f'worker failed; state retained: {log}')
                break
            print(json.dumps(dict(completed=task['stage']+'/'+task['name'],cost=costs(o))),flush=True)
            if task==tasks[29] and not (o/'offline_completed.json').exists():
                assert len(list((o/'offline').glob('*/completed.json')))==30
                write(o/'offline_completed.json',dict(completed=30,cost=costs(o),time=utc(),protocol=p['identity']))
        after=costs(o);assert after==dict(attempted=20000,receipts=20000,queries=750,GP_updates=0,GP_fits=0)
        append(o/'runs.jsonl',dict(before=before,after=after,new={k:after[k]-before[k] for k in before},seconds=time.monotonic()-start,time=utc()))
        if not (o/'all_completed.json').exists():
            write(o/'all_completed.json',dict(offline=30,replay=15,cost=after,time=utc(),protocol=p['identity']))
            write(o/'evaluation_A.json',labels)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--worker')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);o=args.output.resolve()
    if args.prepare:prepare(o)
    elif args.worker:worker(o,args.worker)
    else:
        try:run(o)
        except Exception:
            write(o/'failures'/f'{time.time_ns()}.json',dict(error=traceback.format_exc(),cost=costs(o),time=utc()));raise


if __name__=='__main__':main()
