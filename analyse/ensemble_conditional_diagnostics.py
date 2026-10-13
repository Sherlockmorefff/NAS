"""After-stage diagnostic predictions; fixed fifty-solve fallback, never new queries."""
import argparse,json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
from analyse.relative_metric_pilot import read,write,load,digest,file_hash,tensor_hash,utc
from surrogate.gp_guidance_replay_engine import input_tensors,scaler_from_initial
from surrogate.ensemble_conditional import score_ensemble

def metrics(pred,y,champ,meta):
    ids=sorted(pred);assert ids
    order=sorted(ids,key=lambda c:(-pred[c],c));true=sorted(ids,key=lambda c:(-y[c],c));ranks={c:i+1 for i,c in enumerate(order)}
    unknownchamp=set(ids)&champ
    return dict(**meta,unknown_count=len(ids),MAE=math.fsum(abs(pred[c]-y[c]) for c in ids)/len(ids),
        unknown_pool_champion=bool(unknownchamp),unknown_pool_champion_rank=min(ranks[c] for c in unknownchamp) if unknownchamp else None,
        remaining_best_id=true[0],remaining_best_rank=ranks[true[0]],
        top5_true_best=max(y[c] for c in order[:5]),top10_true_best=max(y[c] for c in order[:10]),top20_true_best=max(y[c] for c in order[:20]))

def diagnose(o):
    if (o/'diagnostics.json').exists():return
    started=time.monotonic();p=read(o/'protocol.json');features=read(o/'features.json');predictions=[]
    for task in read(o/'tasks.json'):
        d=o/'trajectories'/task['name']
        if not (d/'completed.json').exists() or not task['members']:continue
        st=load(d/'state.pt');final109=sorted(set(features['ids'])-set(st['known']));assert len(final109)==109
        for q in range(0,50,10):
            event='initial' if q==0 else f'new{q}';end=300 if q==0 else 200;ids=sorted(set(task['initial_ids'])|set(st['query_order'][:q]));labels={c:st['known'][c] for c in ids}
            for n in [0,50,100,end]:
                f=o/'diagnostic_predictions'/task['name']/f'{event}-step{n:03}.json'
                if not f.exists():
                    snapshots=[load(d/'events'/event/f'member{j}'/f'step{n:03}.pt') for j in range(task['members'])]
                    assert all(s['step']==n and s['known_ids']==ids and s['scaler']==st['scaler'] for s in snapshots)
                    rows=score_ensemble([s['model'] for s in snapshots],input_tensors(features,st['scaler'],labels))
                    write(f,dict(task=task['name'],stage=task['stage'],seed=task['seed'],method=st['method'],event=event,q=q,completed_steps=n,
                        timing='event_start' if n==0 else 'post_update',known_ids=ids,known_ids_hash=digest(ids),scaler_hash=digest(st['scaler']),
                        model_hashes=[s['model_hash'] for s in snapshots],final109=final109,rows=rows,predictions_sealed_at=utc()))
                predictions.append(f)
    # Full A truth is opened only by this evaluator, after score artifacts are saved.
    labels=read(Path(p['A_labels_path']));assert file_hash(Path(p['A_labels_path']))==p['A_labels_sha']
    y={c:math.fsum(v['validation_seeds'])/3 for c,v in labels.items()};star=max(y.values());champ={c for c,v in y.items() if abs(v-star)<=1e-12}
    rows=[];excellent=[]
    for f in predictions:
        a=read(f);lookup={r['candidate_id']:r for r in a['rows']};base={k:a[k] for k in ['stage','seed','method','event','q','completed_steps','timing']}
        ids=sorted(lookup);top=sorted(ids,key=lambda c:(-y[c],c))[:10];count=len(a['model_hashes'])
        for scope,subset in [('event_unknown',ids),('final_unrevealed109',a['final109'])]:
            for j in range(count+1):
                name=f'member{j}' if j<count else 'ensemble_mean';pred={c:(lookup[c]['member_scores'][j] if j<count else lookup[c]['mean']) for c in subset}
                row=metrics(pred,y,champ,dict(**base,scope=scope,predictor=name,prediction_file=str(f),prediction_sha=file_hash(f)))
                chosen_top=sorted(subset,key=lambda c:(-y[c],c))[:10]
                row['excellent_mean_absolute_error']=math.fsum(abs(pred[c]-y[c]) for c in chosen_top)/10
                row['excellent_large_error_low_disagreement']=sum(abs(pred[c]-y[c])>=.05 and lookup[c]['sd'] is not None and lookup[c]['sd']<=.01 for c in chosen_top)
                rows.append(row)
        for c in top:
            r=lookup[c];excellent.append(dict(**base,candidate_id=c,truth=y[c],mean=r['mean'],sd=r['sd'],member_scores=r['member_scores'],
                member_absolute_errors=[abs(v-y[c]) for v in r['member_scores']],mean_absolute_error=abs(r['mean']-y[c]),
                all_members_underestimate=all(v<y[c] for v in r['member_scores']),
                large_mean_error_low_disagreement=abs(r['mean']-y[c])>=.05 and r['sd'] is not None and r['sd']<=.01))
    write(o/'diagnostics.json',rows);write(o/'excellent_candidate_diagnostics.json',excellent)
    write(o/'diagnostic_timing.json',dict(seconds=time.monotonic()-started,prediction_snapshots=len(predictions),rows=len(rows),training_steps=0,
        conclusion_retained='HEAD_UNCERTAINTY_REJECTED',uncertainty_intervals_or_acquisition=False))

def ridge_worker(o,name):
    from analyse.champion_margin_run import install_guard
    p=read(o/'protocol.json');d=o/'ridge'/name;task=dict(label_file=str((d/'input.json').relative_to(o)))
    install_guard(o,d,task,p)
    if (d/'predictions.json').exists():
        a=read(d/'predictions.json');write(d/'solve_receipt.json',dict(solves=1,predictions_sha=file_hash(d/'predictions.json'),sealed_at=a['sealed_at']));return
    assert not (d/'solve_intent.json').exists(),'Unconfirmed linear solve: do not silently repeat beyond 50 solves'
    inp=read(d/'input.json');features=read(o/'features.json');scaler=inp['scaler'];ids=features['ids'];known=sorted(inp['known']);pos={c:i for i,c in enumerate(ids)}
    z=(np.array(features['z'],dtype=np.float64)-scaler['z_mean'])/scaler['z_scale']
    hp=(np.array(features['hp'],dtype=np.float64)-scaler['hp_mean'])/scaler['hp_scale']
    allx=np.concatenate([z,hp],axis=1) if inp['mode']=='RIDGE16' else hp
    x=allx[[pos[c] for c in known]];target=np.array([(math.fsum(inp['known'][c]['validation_seeds'])/3-scaler['y_mean'])/scaler['y_scale'] for c in known],dtype=np.float64)
    xm=x.mean(0);ym=target.mean();xc=x-xm;yc=target-ym;n=len(known)
    write(d/'solve_intent.json',dict(solves=1,lambda_penalty=1.,known_ids=known,known_ids_hash=digest(known),scaler_hash=digest(scaler),time=utc()))
    beta=np.linalg.solve(xc.T@xc+n*np.eye(x.shape[1]),xc.T@yc);intercept=ym-xm@beta
    prediction=(intercept+allx@beta)*scaler['y_scale']+scaler['y_mean'];assert np.isfinite(prediction).all()
    write(d/'predictions.json',dict(mode=inp['mode'],seed=inp['seed'],q=inp['q'],known_ids=known,known_ids_hash=digest(known),scaler_hash=digest(scaler),
        beta=beta.tolist(),intercept=float(intercept),rows={c:float(prediction[pos[c]]) for c in ids if c not in inp['known']},sealed_at=utc(),float_dtype='float64',lambda_penalty=1.))
    a=read(d/'predictions.json');write(d/'solve_receipt.json',dict(solves=1,predictions_sha=file_hash(d/'predictions.json'),sealed_at=a['sealed_at']))

def ridge(o):
    if (o/'ridge_diagnostics.json').exists():return
    assert read(o/'final_decision.json')['ridge_required'];started=time.monotonic();p=read(o/'protocol.json');features=read(o/'features.json')
    refs=read(o/'baseline_reuse_manifest.json')['trajectories'];contexts=[]
    for seed in p['config']['stage1_seeds']:
        d=Path(refs[f'{seed}-C_GREEDY']['directory']);initial=read(o/f'initial_labels/{seed}.json');known=dict(initial);scaler=scaler_from_initial(features,sorted(initial),initial)
        for q in range(0,50,10):
            if q:
                for b in [(q-10)//5,(q-10)//5+1]:known.update(read(d/'batches'/f'b{b:02}'/'reveal.json')['labels'])
            assert len(known)==20+q
            old=read(d/'batches'/f'b{q//5:02}'/'selection.json');assert set(old['known_ids'])==set(known)
            for mode in ['RIDGE16','RIDGE_HP4']:
                name=f'{seed}-q{q}-{mode}';f=o/'ridge'/name;write(f/'input.json',dict(seed=seed,q=q,mode=mode,known=known,scaler=scaler))
                command=[sys.executable,'-B',__file__,'--output',str(o),'--ridge-worker',name]
                subprocess.run(command,check=True,capture_output=True,text=True)
                contexts.append(dict(path=str(f/'predictions.json'),old_C_scores={r['candidate_id']:r['score'] for r in old['rows']},old_C_selection_sha=file_hash(d/'batches'/f'b{q//5:02}'/'selection.json')))
    assert len(contexts)==50
    labels=read(Path(p['A_labels_path']));assert file_hash(Path(p['A_labels_path']))==p['A_labels_sha'];y={c:math.fsum(v['validation_seeds'])/3 for c,v in labels.items()}
    star=max(y.values());champ={c for c,v in y.items() if abs(v-star)<=1e-12};rows=[]
    for r in contexts:
        a=read(Path(r['path']));base=dict(seed=a['seed'],q=a['q'],known_count=20+a['q'],prediction_file=r['path'],prediction_sha=file_hash(Path(r['path'])))
        assert set(a['rows'])==set(r['old_C_scores'])
        rows.append(metrics(a['rows'],y,champ,dict(**base,predictor=a['mode'])))
        if a['mode']=='RIDGE16':rows.append(metrics(r['old_C_scores'],y,champ,dict(**base,predictor='OLD_C',prediction_file_original_hash=r['old_C_selection_sha'])))
    write(o/'ridge_minimal_inputs.json',dict(contexts=contexts,labels=labels,predictions=[read(Path(r['path'])) for r in contexts]))
    write(o/'ridge_diagnostics.json',rows);write(o/'ridge_timing.json',dict(seconds=time.monotonic()-started,solves=50,new_queries=0,training_steps=0,static_diagnostic_only=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--ridge-worker');a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.ridge_worker:ridge_worker(a.output.resolve(),a.ridge_worker)
    else:diagnose(a.output.resolve())
