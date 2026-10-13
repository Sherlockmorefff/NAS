"""Zero-update restoration and independent dense GP checks for actual saved states."""
from pathlib import Path
import sys,copy,math,json,time
import numpy as np,torch,gpytorch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyse import gp_mean_guidance_fivefold as f
from surrogate import gp_mean_guidance as g

def dense(x,z,p):
    r=np.sqrt(np.sum((x[:,None,:]-z[None,:,:])**2,axis=-1))/p['lengthscale'];a=np.sqrt(5.)*r
    return p['outputscale']*(1+a+a*a/3)*np.exp(-a)

def verify(o):
    def blocked(*a,**kw):raise AssertionError('verification optimizer.step forbidden')
    torch.optim.Adam.step=blocked;torch.set_num_threads(1)
    before=f.costs(o);p,tasks=f.verify_protocol(o);nodes=[];gp_checks=[];c_matches=[];ledger=[];probes=[];jitter=[]
    for task in tasks:
        folder=o/'trajectories'/task['task_id'];d=f.load(o/task['input']);warm=f.load(o/task['warmstart']);history=f.read(folder/'history.json');assert len(history)==200
        latest=f.load(folder/'mapping/latest.pt');assert latest['step']==200 and len(latest['committed'])==200
        for path in sorted((folder/'nodes').glob('*.pt')):
            s=f.load(path);step=s['step'];assert step==len(s['history'])==len(s['committed'])
            m=f.JointMetricModel(12,seed=task['seed']);m.load_state_dict(s['model']);m.requires_grad_(True);opt=f.fresh_adam(m);opt.load_state_dict(copy.deepcopy(s['optimizer']));f.restore_rng(s)
            assert {float(v['step']) for v in opt.state.values()}==({float(step)} if step else set())
            assert all(torch.isfinite(v[k]).all() for v in opt.state.values() for k in ['exp_avg','exp_avg_sq'])
            with torch.no_grad():e,h=m(d['train']['z'],d['train']['hp']);qe,qh=m(d['query']['z'],d['query']['hp'])
            assert all(torch.equal(a,b) for a,b in [(e,s['train_e']),(h,s['train_head']),(qe,s['query_e']),(qh,s['query_head'])])
            if step==0:assert f.tensor_hash(s['model'])==f.tensor_hash(warm['model']) and not opt.state
            if task['branch']=='G' and not(step==0 and s['phase']=='before_refresh'):
                block=step//50 if s['phase']=='after_refresh' else (step-1)//50
                teachers=[f.load(folder/'gps'/f'b{block}-g{i}'/'teacher.pt') for i in range(3)]
                info=g.gradient_probe(m,d['train'],d['target32'],d['target64'],d['groups'],teachers)
                matches=[r for r in s['probes'] if r['step']==step and r['phase']==s['phase'] and r['block']==block]
                assert len(matches)==1 and all(matches[0][k]==v for k,v in info.items())
                probes.append(dict(task=task['task_id'],step=step,phase=s['phase'],**info))
            nodes.append(dict(task=task['task_id'],node=path.name,step=step,predictions_exact=True,RNG_Adam_loaded=True))
        initial=f.load(folder/'nodes/add000-before_refresh.pt');first=f.load(folder/'nodes/add001-within_block.pt')
        changes={n:float((first['model'][n]-initial['model'][n]).abs().max()) for n in initial['model']};assert max(changes.values())>0
        f.write(o/'verification'/(task['task_id']+'-parameter_changes.json'),dict(first_step=changes,initial_sha=f.tensor_hash(initial['model']),endpoint_sha=f.tensor_hash(latest['model']),inputs_unchanged=f.tensor_hash({'z':d['train']['z'],'hp':d['train']['hp'],'target64':d['target64']})==latest['input_hash']))
        for h,gid in zip(history,latest['committed']):
            intent=f.read(o/'intents/mapping'/f'{gid:06d}.json');receipt=f.read(o/'receipts/mapping'/f'{gid:06d}.json')
            assert h['global_id']==gid==intent['global_id']==receipt['global_id'] and h['step']==intent['step']==receipt['step'] and intent['task']==receipt['task']==task['task_id']
            ledger.append(dict(kind='mapping',task=task['task_id'],step=h['step'],global_id=gid))
        if task['branch']=='C':
            previous=f.load(f.R1/'trajectories'/f"fold{task['fold']}-s{task['seed']}-R0"/'add200.pt')
            assert f.tensor_hash(latest['model'])==f.tensor_hash(previous['model'])
            c_matches.append(dict(task=task['task_id'],old_R1_R0_endpoint_exact=True,sha=f.tensor_hash(latest['model'])))
        for gpdir in sorted((folder/'gps').iterdir()):
            saved=f.load(gpdir/'latest.pt');teacher=f.load(gpdir/'teacher.pt');assert saved['step']==20 and teacher['metadata']==saved['metadata']
            meta=saved['metadata'];assert meta['fold']==task['fold'] and meta['seed']==task['seed'] and meta['scaler']==d['scaler']
            if gpdir.name=='endpoint':
                assert meta['support_ids']==d['train']['ids'] and meta['mapping_sha']==f.tensor_hash(latest['model'])
            else:
                group=d['groups'][meta['group']];snapshot=f.load(folder/'blocks'/f"b{meta['block']}"/'snapshot.pt')
                assert meta['support_ids']==group['support_ids'] and not set(meta['support_architectures'])&set(group['query_architectures'])
                assert meta['mapping_sha']==snapshot['mapping_sha'] and meta['mapping_step']==snapshot['step']==meta['block']*50
                assert torch.equal(saved['x'],snapshot['embedding'][group['support_indices']])
            gp,opt=f.make_gp(saved['x'],saved['target'],saved['se'],saved['scale'])
            for k in range(1,21):
                state=f.load(gpdir/f'step{k:02d}.pt');assert state['step']==len(state['committed'])==len(state['history'])==k and state['identity']==saved['identity']
                gp.load_state_dict(state['model']);opt.load_state_dict(copy.deepcopy(state['optimizer']));f.restore_rng(state)
                assert {float(v['step']) for v in opt.state.values()}=={float(k)}
                assert all(torch.isfinite(v[z]).all() for v in opt.state.values() for z in ['exp_avg','exp_avg_sq'])
                gid=state['committed'][-1];intent=f.read(o/'intents/gp'/f'{gid:06d}.json');receipt=f.read(o/'receipts/gp'/f'{gid:06d}.json')
                assert intent['step']==receipt['step']==k and intent['task']==receipt['task']==task['task_id']+'/'+gpdir.name
                ledger.append(dict(kind='gp',task=task['task_id']+'/'+gpdir.name,step=k,global_id=gid))
                if k in [1,20]:
                    with torch.no_grad():forward=gp(saved['x']);assert torch.isfinite(forward.mean).all() and torch.isfinite(forward.covariance_matrix).all()
            rebuilt=g.from_gp(gp,saved['x'],saved['target'],saved['se'],saved['scale'],meta)
            assert rebuilt['parameters']==teacher['parameters'] and torch.equal(rebuilt['alpha'],teacher['alpha']) and torch.equal(rebuilt['cholesky'],teacher['cholesky'])
            x=saved['x'].numpy();y=saved['target'].numpy();se=saved['se'].numpy();pars=teacher['parameters'];K=dense(x,x,pars)+np.diag((se/saved['scale'])**2+pars['noise']+teacher['jitter'])
            L=np.linalg.cholesky(K);alpha=np.linalg.solve(L.T,np.linalg.solve(L,y-pars['mean']))
            query=np.asarray(f.read(folder/'predictions_before_labels.json')['query']['embeddings']) if gpdir.name=='endpoint' else snapshot['embedding'][group['query_indices']].numpy()
            cross=dense(query,x,pars);mu=pars['mean']+cross@alpha;var=pars['outputscale']-(np.linalg.solve(L,cross.T)**2).sum(0)
            a,v,_=g.posterior(teacher,torch.tensor(query,dtype=torch.double));md=float(np.max(abs(mu-a.detach().numpy())));vd=float(np.max(abs(np.maximum(var,0)-v.detach().numpy())))
            assert md<=1e-9 and vd<=1e-8,(md,vd)
            gp_checks.append(dict(task=task['task_id'],GP=gpdir.name,states_reloaded=20,mean_dense_maxdiff=md,variance_dense_maxdiff=vd,conditioning_jitter=teacher['jitter']))
            jitter.extend([r['actual_jitter'] for r in saved['history']])
        m=f.JointMetricModel(12,seed=task['seed']);m.load_state_dict(latest['model']);now=f.endpoint_prediction(m,d,f.load(folder/'gps/endpoint/teacher.pt'));assert now==f.read(folder/'predictions_before_labels.json')
    assert len(ledger)==10200 and len(c_matches)==15 and len(gp_checks)==210
    assert all(f.file_hash(Path(n))==h for n,h in f.read(o/'input_sources.json').items());assert f.costs(o)==before
    rerun=f.read(o/'invocations/run02.json');assert rerun['new_updates']=={'mapping':0,'gp':0}
    result=dict(passed=True,mapping_nodes=len(nodes),GP_states_reloaded=4200,GP_forward_states=420,GP_dense_checks=210,C_exact_matches=c_matches,
      nodes=nodes,GP_checks=gp_checks,guidance_probes_recomputed=len(probes),GP_fit_jitter_nonzero=sum(x>0 for x in jitter),GP_fit_max_jitter=max(jitter),
      matrix_max_jitter=max(r['conditioning_jitter'] for r in gp_checks),original_inputs_unchanged=True,validation_updates=0,completed_reentry_new_updates=rerun['new_updates'],cost=before)
    f.write(o/'runtime_validation.json',result);f.write(o/'update_ledger.json',ledger);f.write(o/'actual_guidance_probes.json',probes)
    print(json.dumps({k:v for k,v in result.items() if k not in ['nodes','GP_checks','C_exact_matches']}))
if __name__=='__main__':verify(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else f.DEFAULT)
