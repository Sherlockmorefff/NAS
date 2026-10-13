"""Read-only verification of actual training, sealed coordinates, recovery and budgets."""
import argparse,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from analyse.relative_metric_pilot import read,write,load,digest,tensor_hash,file_hash
from analyse.budget_champion_run import check_protocol,OLD,PRIOR
from surrogate.gp_guidance_replay_engine import journal,scaler_from_initial,input_tensors,reload_model,score_candidates,restore_rng,rng_state
from surrogate.fuzzy_metric import JointMetricModel
from surrogate.budget_champion_ranking import list_indices

def verify(o):
    begin=time.monotonic();p=check_protocol(o);f=read(o/'features.json');labs=read(o/'evaluation_A.json');tasks=read(o/'tasks.json')
    assert file_hash(Path(p['VAE_manifest']['checkpoint']))==p['VAE_manifest']['checkpoint_sha256']
    assert read(o/'folds.json')==read(OLD/'partition.json')['folds']
    assert file_hash(OLD/'A_validation.json')==p['A_labels_sha'] and read(OLD/'A_validation.json')==labs
    intents=journal(o/'ledger/mapping-intents.jsonl');receipts=journal(o/'ledger/mapping-receipts.jsonl')
    assert len(intents)==len(receipts)==20000
    assert [r['id'] for r in intents]==list(range(1,20001))==[r['id'] for r in receipts]
    for a,b in zip(intents,receipts):assert all(a[k]==b[k] for k in ['id','task','step']) and a['protocol']==p['identity']
    byid={x['id']:x for x in receipts};visited=[];nsteps=0;ncheckpoints=0;ncoords=0;scalar=[];gradient_active=0
    for t in tasks:
        folder=o/t['stage']/t['name'];done=read(folder/'completed.json');assert done['protocol']==p['identity']
        for name,h in done['files'].items():assert file_hash(folder/name)==h
        initial={c:labs[c] for c in t['initial_ids']};assert read(o/t['label_file'])==initial
        scale=scaler_from_initial(f,t['initial_ids'],initial)
        start=load(o/t['weight_file']);assert start['untrained'] and tensor_hash(start['model'])==tensor_hash(JointMetricModel(12,seed=t['model_seed']).state_dict())
        known=dict(initial);versions={};expectedstart=start['model']
        events=['initial'] if t['stage']=='offline' else ['initial','new10','new20','new30','new40']
        if t['method']=='RANDOM':events=[]
        for event in events:
            n=0 if event=='initial' else int(event[3:]);target=300 if n==0 else 200
            if n:
                known=dict(initial)
                for b in range(n//5):known.update(read(folder/'batches'/f'b{b:02}'/'reveal.json')['labels'])
            data=input_tensors(f,scale,known);d=folder/'events'/event;state=load(d/'latest.pt');zero=load(d/'step000.pt')
            assert state['step']==target and state['steps']==target and len(state['history'])==target
            assert state['train_ids']==sorted(known) and state['scaler']==scale
            assert state['input_hash']==tensor_hash({k:data[k] for k in ['z','hp','target32','y']})
            assert tensor_hash(zero['model'])==tensor_hash(expectedstart)==state['initial_model_hash']
            assert not zero['optimizer']['state'] and state['weight']==(.1 if t['method']=='BR_HEAD' else 0.)
            rho=t['rho'] if t['stage']=='offline' else (50-n)/(159-n);assert state['rho']==rho
            for checkpoint in sorted(d.glob('step[0-9][0-9][0-9].pt')):
                s=load(checkpoint);ncheckpoints+=1
                assert s['scaler']==scale and s['train_ids']==sorted(known)
                assert all(torch.isfinite(v).all() for v in s['model'].values())
                for group in s['optimizer']['param_groups']:
                    assert group['lr']==.001 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0.
                for optstate in s['optimizer']['state'].values():
                    assert float(optstate['step'])==s['step'] and torch.isfinite(optstate['exp_avg']).all() and torch.isfinite(optstate['exp_avg_sq']).all()
            for h in state['history']:
                ls=list_indices(sorted(known),t['stage'],t['task_id'],event,h['optimizer_step'])
                assert digest(ls)==h['list_hash'] and h['step']==h['optimizer_step']+1
                true_ranks=[]
                for js in ls:
                    best=max(float(data['y'][i]) for i in js)
                    true_ranks.append(1+int((data['y']>best+1e-12).sum()))
                assert h['champion_ranks_in_D']==true_ranks
                assert all(np.isfinite(h[k]) for k in ['total','regression','size','ranking','training_head_MAE'])
                assert abs(h['total']-(h['regression']+.001*h['size']+state['weight']*h['ranking']))<2e-5
                assert byid[h['id']]['task']==t['stage']+'/'+t['name']+'/'+event and byid[h['id']]['step']==h['step']
                visited.append(h['id']);nsteps+=1
            norms=state['gradient_checks'][0]['ranking_gradient_norms']
            assert all(v is not None and np.isfinite(v) for v in norms.values())
            assert state['gradient_checks'][0]['input_requires_grad']==[False,False]
            gradient_active+=int(sum(v for k,v in norms.items() if k.startswith('mapping.'))>0 and norms['regressor.weight']>0)
            # Four real snapshots per stage/task representative, retaining original tensors as scalar inputs.
            if t['name'] in ['fold0-s0-C_HEAD','fold0-s0-BR_HEAD','start2026091600-s0-BR_HEAD']:
                for pp in sorted((d/'diagnostics').glob('*.pt')):
                    z=load(pp);scalar.append(dict(source=str(pp.relative_to(o)),**{k:z[k] for k in ['ids','lists','rho','ranking_loss']},scores=z['scores'].tolist(),raw_y=z['raw_y'].tolist()))
            versions[n//10]=state['model'];expectedstart=state['model']
        if t['stage']=='offline':
            saved=read(folder/'predictions.json');rows,cache,_=score_candidates(reload_model(versions[0]),input_tensors(f,scale,initial))
            assert rows==saved['rows'] and saved['scaler']==scale;oldcache=load(folder/'coordinates.pt')
            assert tensor_hash(cache)==tensor_hash(oldcache) if all(isinstance(v,torch.Tensor) for v in cache.values()) else cache['embedding_hash']==oldcache['embedding_hash']
            ncoords+=1
            assert set(saved['known_ids']).isdisjoint(t['query_ids'])
            a=dict(zip(f['ids'],f['architectures']));assert not {a[c] for c in saved['known_ids']}&{a[c] for c in t['query_ids']}
        else:
            known=dict(initial);order=[]
            for b in range(10):
                d=folder/'batches'/f'b{b:02}';s=read(d/'selection.json');rv=read(d/'reveal.json')
                assert s['known_ids']==sorted(known) and s['known_labels_hash']==digest(known) and s['scaler_hash']==digest(scale)
                assert s['query_count_before']==5*b and s['remaining_budget']==50-5*b
                assert rv['selection_sha']==file_hash(d/'selection.json') and s['selected_at']<=rv['revealed_at']
                assert s['selected_ids']==[r['candidate_id'] for r in s['rows'][:5]] and not set(s['selected_ids'])&set(known)
                if t['method']!='RANDOM':
                    model=reload_model(versions[b//2]);rows,cache,_=score_candidates(model,input_tensors(f,scale,known))
                    assert rows==s['rows'] and tensor_hash(model.state_dict())==s['model_hash'] and s['version']==b//2
                    oldcache=load(d/'coordinates.pt');assert cache['embedding_hash']==oldcache['embedding_hash']
                    ncoords+=1
                assert rv['labels']=={c:labs[c] for c in s['selected_ids']}
                known.update(rv['labels']);order.extend(s['selected_ids']);assert read(d/'aggregated.json')['query_count']==len(order)
            final=load(folder/'state.pt');assert final['known']==known and final['query_order']==order and len(set(order))==50 and len(known)==70
            assert final['scaler']==scale and final['initial_ids']==t['initial_ids']
    assert len(set(visited))==nsteps==20000
    # Actual interruption evidence: durable step 1 is the exact model and Adam resumed.
    d=o/'offline/fold0-s0-BR_HEAD/events/initial';st=load(d/'step001.pt');rec=journal(d/'recovery.jsonl');assert len(rec)==1 and rec[0]['step']==1
    assert rec[0]['model']==tensor_hash(st['model']) and rec[0]['optimizer_steps']==[1.]
    task=next(t for t in tasks if t['name']=='fold0-s0-BR_HEAD');data=input_tensors(f,st['scaler'],{c:labs[c] for c in st['train_ids']})
    with torch.no_grad():e,s=reload_model(st['model'])(data['z'],data['hp'])
    assert rec[0]['forward_hash']==tensor_hash(dict(embedding=e,scores=s))
    assert rec[0]['next_list_hash']==digest(list_indices(st['train_ids'],'offline','fold0-s0','initial',1))
    assert rec[0]['scaler_hash']==digest(st['scaler']) and rec[0]['rng_restored']
    restore_rng(st['rng']);r=rng_state();assert torch.equal(r['torch'],st['rng']['torch']) and np.array_equal(r['numpy'][1],st['rng']['numpy'][1]) and r['python']==st['rng']['python']
    d=o/'replay/start2026091600-s0-BR_HEAD/batches/b00';pause=read(d/'pause-selected.json')
    assert pause['query_count']==0 and pause['selection_sha']==file_hash(d/'selection.json')
    assert pause['time']<read(d/'reveal.json')['revealed_at']
    inv=journal(o/'invocations.jsonl');assert len([x for x in inv if x['returncode']==75])==2 and all(x['returncode'] in [0,75] for x in inv)
    reentries=journal(o/'runs.jsonl');assert len(reentries)>=2 and all(v==0 for v in reentries[-1]['new'].values())
    write(o/'verification/scalar_inputs.json',scalar,immutable=False)
    result=dict(passed=True,actual_effective_updates=nsteps,events=80,checkpoints=ncheckpoints,all_step_list_identities=nsteps,
        candidate_coordinate_batches=ncoords,predictions_exact=True,scalers_initial_only=True,untrained_parameters_verified=True,
        gradient_active_events=gradient_active,gradient_connected_events=80,VAE_hash_unchanged=True,B_test_parsed=0,
        training_interruptions=1,selected_not_revealed_interruptions=1,completed_reentry_new=reentries[-1]['new'],seconds=time.monotonic()-begin)
    write(o/'verification/runtime.json',result,immutable=False);print(result)

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/posthoc/budget-champion-ranking-v1');verify(p.parse_args().output)
