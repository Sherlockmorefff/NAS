"""Read-only verification of completed replay states, coordinates, budgets and recovery."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from analyse.gp_guidance_development_replay import verify_protocol, SEEDS, METHODS, counts
from analyse.relative_metric_pilot import read, write, load, file_hash, tensor_hash, digest
from surrogate.gp_guidance_replay_engine import journal, input_tensors, reload_model, score_candidates, scaler_from_initial


def verify(o):
    p=verify_protocol(o);seal=read(o/'all_completed.json')
    assert seal['completed']==20
    for rel,h in seal['artifacts'].items():assert file_hash(o/rel)==h,rel
    features=read(o/'features.json');labels=read(o/'evaluation_A.json');starts=read(o/'starts.json')
    mapping=[];gps=[];checkpoints=0;coordinate_count=0;dense_diffs=[];timings={};clone_checks=[]
    for seed in SEEDS:
        s=starts[str(seed)];initial={c:labels[c] for c in s['initial_ids']}
        scaler=scaler_from_initial(features,s['initial_ids'],initial)
        common=load(o/'shared'/str(seed)/'M100.pt')
        assert common['scaler']==scaler and common['steps']==100
        cinit=load(o/'trajectories'/f'{seed}-C_GP'/'initial.pt')
        hinit=load(o/'trajectories'/f'{seed}-C_HEAD'/'initial.pt')
        sinit=load(o/'trajectories'/f'{seed}-S_GP'/'initial.pt')
        assert cinit['warm_model_hash']==sinit['warm_model_hash']==tensor_hash(common['model'])
        assert tensor_hash(cinit['model'])==tensor_hash(hinit['model'])
        a,b=reload_model(cinit['model']),reload_model(hinit['model'])
        assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters()))
        clone_checks.append(dict(seed=seed,common_M100=True,C_HEAD_equals_initial_C=True,independent_objects=True))
        for method in METHODS:
            folder=o/'trajectories'/f'{seed}-{method}';done=read(folder/'completed.json');state=load(folder/'state.pt')
            assert state['query_order']==done['query_order'] and len(state['query_order'])==50
            assert len(set(state['query_order']+s['initial_ids']))==100
            assert state['known']=={c:labels[c] for c in s['initial_ids']+state['query_order']}
            assert state['scaler']==scaler
            for batch in range(10):
                bdir=folder/'batches'/f'b{batch:02}';sel=read(bdir/'selection.json');reveal=read(bdir/'reveal.json');agg=read(bdir/'aggregated.json')
                known={c:labels[c] for c in s['initial_ids']+state['query_order'][:batch*5]}
                assert sel['known_ids']==sorted(known) and sel['known_labels_hash']==digest(known)
                assert agg['query_count']==batch*5+5 and reveal['selection_sha']==file_hash(bdir/'selection.json')
                assert sel['selected_at']<=reveal['revealed_at']
                assert sel['scaler_hash']==digest(scaler)
                if method=='RANDOM':continue
                version=batch//2+1
                assert sel['version']==version
                ms=load(folder/'initial.pt') if version==1 else load(folder/'maps'/f'v{version}'/'latest.pt')
                model=reload_model(ms['model']);d=input_tensors(features,scaler,known)
                cache=load(bdir/'coordinates.pt');assert cache['version']==version
                with torch.no_grad():e,h=model(d['all_z'],d['all_hp'])
                assert torch.equal(e,cache['embedding']) and torch.equal(h,cache['head'])
                assert tensor_hash(model.state_dict())==sel['model_hash']==cache['model_hash']
                teacher=None
                if method in ['C_GP','S_GP']:
                    teacher=load(folder/'selection_gps'/f'b{batch:02}'/'teacher.pt')
                    assert torch.equal(teacher['support'],e[d['indices']].double())
                    assert teacher['metadata']['version']==version and teacher['metadata']['train_ids']==sorted(known)
                rows,_,_=score_candidates(model,d,teacher)
                assert rows==sel['rows']
                coordinate_count+=1
    for path in list((o/'shared').glob('*/mapping/latest.pt'))+list((o/'trajectories').glob('*/maps/*/latest.pt')):
        s=load(path);mapping.append(s)
        assert s['step']==len(s['history'])==len(s['committed'])
        assert s['step']==(100 if '-COMMON' in s['metadata']['task'] else 200)
        task=s['metadata']['task'];seed=int(task[:10]);initial=starts[str(seed)]['initial_ids']
        if '-COMMON' in task:expected=sorted(initial)
        else:
            owner,suffix=task.split('/')
            n=0 if suffix=='initial' else int(suffix.removeprefix('update'))
            done=read(o/'trajectories'/owner/'completed.json')
            expected=sorted(initial+done['query_order'][:n])
        assert s['train_ids']==expected and s['scaler']['fit_ids']==sorted(initial)
        assert all(torch.isfinite(v).all() for v in s['model'].values())
        assert {float(v['step']) for v in s['optimizer']['state'].values()}=={float(s['step'])}
        if s['groups'] is not None:
            assert len(s['probes'])==1 and s['probes'][0]['support_and_query_connected']
            assert s['probes'][0]['support_shared_norm']>0 and s['probes'][0]['query_shared_norm']>0
            for g in s['groups']:assert not set(g['support_architectures'])&set(g['query_architectures'])
        for node in path.parent.glob('step*.pt'):
            ns=load(node);m=reload_model(ns['model'])
            d=input_tensors(features,ns['scaler'],{c:labels[c] for c in ns['train_ids']})
            with torch.no_grad():emb,head=m(d['z'],d['hp'])
            assert torch.isfinite(emb).all() and torch.isfinite(head).all()
            assert ns['step']==len(ns['history'])
            checkpoints+=1
    for path in (o/'trajectories').rglob('latest.pt'):
        if path.parent.parent.name not in ['gps','selection_gps']:continue
        s=load(path);gps.append(s)
        assert s['step']==20==len(s['history'])==len(s['committed'])
        ids=s['metadata']['train_ids'];seed=int(s['metadata']['task'][:10])
        sc=load(o/'shared'/str(seed)/'M100.pt')['scaler']
        expected=np.array([(labels[c]['validation_mean']-sc['y_mean'])/sc['y_scale'] for c in ids])
        assert np.max(np.abs(s['target'].numpy()-expected))<1e-12
        assert np.max(np.abs(s['se'].numpy()-np.array([labels[c]['validation_se'] for c in ids])))<1e-12
        teacher=load(path.parent/'teacher.pt');pars=teacher['parameters'];x=teacher['support'].numpy()
        distance=np.linalg.norm((x[:,None]-x[None,:])/pars['lengthscale'],axis=-1)*np.sqrt(5.)
        kernel=pars['outputscale']*(1+distance+distance**2/3)*np.exp(-distance)
        covariance=kernel+np.diag((s['se'].numpy()/s['scale'])**2+pars['noise']+teacher['jitter'])
        alpha=np.linalg.solve(covariance,s['target'].numpy()-pars['mean'])
        dense_diffs.append(float(np.max(np.abs(alpha-teacher['alpha'].numpy()))))
        assert np.allclose(alpha,teacher['alpha'].numpy(),atol=2e-7,rtol=1e-8)
        for node in path.parent.glob('step*.pt'):
            ns=load(node);assert ns['step']==len(ns['history'])
            for key,value in ns['model'].items():
                if key.endswith('_constraint.upper_bound'):
                    assert torch.isposinf(value).all()  # GreaterThan intentionally has no upper bound.
                else:
                    assert torch.isfinite(value).all(), key
            checkpoints+=1
    effective={k:[gid for s in ss for gid in s['committed']] for k,ss in [('mapping',mapping),('gp',gps)]}
    budget={}
    for kind in ['mapping','gp']:
        intents=journal(o/'ledger'/(kind+'-intents.jsonl'));receipts=journal(o/'ledger'/(kind+'-receipts.jsonl'))
        ei=set(effective[kind]);ii={r['id'] for r in intents};ri={r['id'] for r in receipts}
        assert len(ei)==len(effective[kind]) and ei==ii==ri
        budget[kind]=dict(effective=len(ei),actual=len(ri),charged=len(ii),uncertain=0,discarded=0)
    assert len(mapping)==75 and len(gps)==400 and len({s['attempt'] for s in gps})==400
    assert budget['mapping']['effective']==14500 and budget['gp']['effective']==8000
    assert counts(o)==dict(mapping=14500,gp=8000,gp_fits=400,queries=1000)
    # Perturb a real, initially hidden label after completion and reconstruct
    # the first selection from the actual saved model/GP, with zero updates.
    seed=SEEDS[0];folder=o/'trajectories'/f'{seed}-C_GP'
    first=read(folder/'batches/b00/selection.json');hidden=first['selected_ids'][0]
    altered=copy.deepcopy(labels)
    altered[hidden].update(validation_mean=.1,validation_seeds=[.1,.1,.1],validation_se=0.)
    initial_ids=starts[str(seed)]['initial_ids']
    warm=load(folder/'initial.pt');model=reload_model(warm['model'])
    teacher=load(folder/'selection_gps/b00/teacher.pt')
    original_known={c:labels[c] for c in initial_ids};altered_known={c:altered[c] for c in initial_ids}
    assert original_known==altered_known
    original_rows,_,_=score_candidates(model,input_tensors(features,warm['scaler'],original_known),teacher)
    altered_rows,_,_=score_candidates(model,input_tensors(features,warm['scaler'],altered_known),teacher)
    assert original_rows==altered_rows==first['rows']
    perturbation=dict(candidate_id=hidden,first_real_selection_identical=True,all_remaining_scores_identical=True,
        model_updates=0,GP_updates=0,changed_label_not_supplied_until_reveal=True)
    recoveries=[dict(path=str(f.relative_to(o)),rows=journal(f)) for f in o.rglob('recovery.jsonl')]
    pauses=[str(f.relative_to(o)) for f in o.rglob('pause-*.json')]
    assert len(pauses)==5 and len(recoveries)==3
    runs=journal(o/'runs.jsonl');assert len(runs)>=2 and all(v==0 for v in runs[-1]['new'].values())
    for method in METHODS:
        ms=[s for s in mapping if f'-{method}/' in s['metadata']['task']]
        gs=[s for s in gps if f'-{method}/' in s['metadata']['task']]
        bs=list((o/'trajectories').glob(f'*-{method}/batches/*/selection.json'))
        ts=list((o/'trajectories').glob(f'*-{method}/batches/*/oracle_timing.json'))
        timings[method]=dict(mapping_seconds=sum(s['seconds'] for s in ms),GP_seconds=sum(s['seconds'] for s in gs),
            selection_seconds=sum(read(f)['selection_seconds'] for f in bs),oracle_IO_seconds=sum(read(f)['seconds'] for f in ts),
            worker_wall_seconds_first_run=sum(v['seconds'] for v in journal(o/'invocations.jsonl') if v['method']==method and v['time']<=runs[0]['time']),
            completed_reentry_worker_seconds=sum(v['seconds'] for v in journal(o/'invocations.jsonl') if v['method']==method and v['time']>runs[0]['time']),
            physical_mapping_steps=sum(s['step'] for s in ms),GP_fits=len(gs),logical_mapping_steps=0 if method=='RANDOM' else 5500)
    common_seconds=sum(s['seconds'] for s in mapping if '-COMMON' in s['metadata']['task'])
    c_initial_seconds=sum(s['seconds'] for s in mapping if '-C_GP/initial' in s['metadata']['task'])
    for method in METHODS[:-1]:
        timings[method]['independent_mapping_seconds_equivalent']=timings[method]['mapping_seconds']+common_seconds+(c_initial_seconds if method=='C_HEAD' else 0)
    solves={}
    for s in mapping:
        for kind,r in s['solve_records'].items():
            t=solves.setdefault(kind,dict(conditionings=0,seconds=0.,max_jitter=0.))
            t['conditionings']+=r['conditionings'];t['seconds']+=r['seconds'];t['max_jitter']=max(t['max_jitter'],r['max_jitter'])
    assert solves['formal_mapping_forward']['conditionings']==15000
    result=dict(passed=True,trajectories=20,labels_each='50 initial + 50 queried',coordinate_batches=coordinate_count,
        loaded_checkpoints=checkpoints,mapping_trainings=len(mapping),GP_fits=len(gps),budgets=budget,
        GP_dense_alpha_maxdiff=max(dense_diffs),clones=clone_checks,recoveries=recoveries,pause_paths=pauses,
        completed_reentry=runs[-1],timing=timings,shared_timing=dict(common_M100_mapping_seconds=common_seconds,
        C_initial_mapping_seconds=c_initial_seconds),total_first_run_wall_seconds=runs[0]['seconds'],
        solve_records=solves,actual_GP_max_jitter=max(h['actual_jitter'] for s in gps for h in s['history']),
        real_hidden_label_perturbation=perturbation,GNN_trainings=0,VAE_trainings=0,synthetic_mapping_updates=0,synthetic_GP_updates=0,
        labels_boundary='worker file guard; own initial + preselected batches; fixed initial scaler; all saved states verified')
    write(o/'runtime_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['timing','clones','recoveries','pause_paths']},indent=2))


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    verify(Path(sys.argv[1]).resolve())
