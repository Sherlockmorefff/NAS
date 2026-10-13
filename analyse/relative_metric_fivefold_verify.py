"""Reload every saved fivefold node, optimizer and RNG without updates."""
from pathlib import Path
import sys,copy,math,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyse import relative_metric_fivefold as f
from surrogate.fuzzy_metric import JointMetricModel
from surrogate.relative_metric import fresh_adam,tensor_plan,objective

def verify(o):
    def reject(*a,**k):raise AssertionError('read-only verification attempted optimizer step')
    torch.optim.Adam.step=reject;torch.set_num_threads(1);before=f.costs(o)
    protocol,tasks=f.verify_protocol(o);nodes=[];ledger=[]
    for task in tasks:
        seed,branch=task['seed'],task['branch'];folder=o/'trajectories'/task['task_id']
        d=f.load(o/task['input']);_,_,target=f.assert_task_data(task,d);warm=f.load(o/task['warmstart']);plan=dict(np.load(o/task['plan']))
        history=f.read(folder/'loss_history.json');assert len(history)==200 and [r['step'] for r in history]==list(range(1,201))
        init=f.load(folder/'add000.pt');assert f.tensor_hash(init['model'])==f.tensor_hash(warm['model']) and not init['optimizer']['state']
        for step in [0,1,50,100,200]:
            state=f.load(folder/f'add{step:03d}.pt');assert state['step']==state['plan_position']==len(state['history'])==len(state['committed_global_ids'])==step
            assert state['last_global_id']==(state['committed_global_ids'][-1] if step else None)
            m=JointMetricModel(12,seed=seed);m.load_state_dict(state['model']);opt=fresh_adam(m);opt.load_state_dict(copy.deepcopy(state['optimizer']))
            assert {float(v['step']) for v in opt.state.values()}==({float(step)} if step else set())
            for g in opt.param_groups:
                for k,v in f.CONFIG['optimizer'].items():assert (list(g[k]) if k=='betas' else g[k])==v
            assert all(torch.isfinite(p).all() for p in m.parameters())
            assert all(torch.isfinite(v[k]).all() for v in opt.state.values() for k in ['exp_avg','exp_avg_sq'])
            torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng'])
            assert torch.equal(torch.get_rng_state(),state['torch_rng']) and np.array_equal(np.random.get_state()[1],state['numpy_rng'][1])
            saved=f.read(folder/f'inference_add{step:03d}.json');now={**f.inference(m,d,step,seed,branch),'fold':task['fold']};assert saved==now
            actual_probe=f.probe_groups(m,d['train'],target,tensor_plan(plan,0));assert actual_probe==state['probes'][str(step)]
            nodes.append(dict(task=task['task_id'],step=step,restored_predictions_and_probes_exact=True,optimizer_moments_finite=True,RNG_restored=True))
        first=f.load(folder/'add001.pt');firstrow=history[0]
        for state,key in [(init,'before'),(first,'after')]:
            m=JointMetricModel(12,seed=seed);m.load_state_dict(state['model'])
            with torch.no_grad():e,p=m(d['train']['z'],d['train']['hp']);vals=objective(e,p,target,tensor_plan(plan,0),f.CONFIG['lambda_r'][branch])
            expected=firstrow if key=='before' else firstrow['after_same_sample']
            assert all(float(v)==expected[k] for k,v in vals.items())
        for name in warm['model']:assert float((first['model'][name]-init['model'][name]).abs().max())==firstrow['parameter_max_changes'][name]
        latest=f.load(folder/'latest.pt');assert latest['step']==200
        for h,gid in zip(history,latest['committed_global_ids']):
            intent=f.read(o/'intents'/f'{gid:06d}.json');receipt=f.read(o/'receipts'/f'{gid:06d}.json')
            assert h['global_id']==intent['global_id']==receipt['global_id']==gid
            assert intent['task']==receipt['task']==task['task_id'] and intent['additional_step']==receipt['additional_step']==h['step']
            assert intent['protocol_identity']==protocol['identity'] and receipt['optimizer_step_returned']
            assert h['plan_position_used']==h['step']-1 and h['lambda_r']==f.CONFIG['lambda_r'][branch] and h['lambda_e']==.001
            assert all(math.isfinite(h[k]) for k in ['total','regression','order','size','gradient_norm'])
            ledger.append(dict(task=task['task_id'],step=h['step'],global_id=gid))
    assert all(f.file_hash(Path(n))==h for n,h in f.read(o/'input_sources.json').items())
    assert f.costs(o)==before
    rerun=f.read(o/'invocations/run02.json');assert rerun['actual_updates_this_invocation']==0 and rerun['before']==rerun['after']
    result=dict(passed=True,actual_nodes_reloaded=len(nodes),checks=nodes,committed_steps_checked=len(ledger),first_step_loss_and_parameter_changes_exact=True,
      all_original_sources_unchanged=True,completed_reentry_additional_updates=0,validation_optimizer_updates=0,cost=before)
    f.write(o/'runtime_validation.json',result);f.write(o/'update_ledger.json',ledger)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
if __name__=='__main__':verify(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else f.DEFAULT)
