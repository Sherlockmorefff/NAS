"""Restore all30 actual R1 diagnostic checkpoints without optimizer updates."""
from pathlib import Path
import sys,json,copy,math
import numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from analyse.relative_metric_pilot import DEFAULT as O,CONFIG,read,load,write,file_hash,tensor_hash,cost_status,inference
from surrogate.fuzzy_metric import JointMetricModel
from surrogate.relative_metric import fresh_adam,tensor_plan,objective,validate_batch

def reject(*args,**kwargs):raise AssertionError('readonly verification cannot update')

def verify(O):
    torch.optim.Adam.step=reject;torch.set_num_threads(1)
    cost_before=cost_status(O);rows=[];casts=[]
    for seed in range(3):
     data=load(O/'inputs'/f's{seed}.pt');warm=load(O/'warmstarts'/f's{seed}-M100.pt');plan=dict(np.load(O/'plans'/f's{seed}.npz'))
     y,se,target=validate_batch(data['train'],data['query'],data['labels'],data['scaler'])
     casts.append(dict(seed=seed,weight_max_error=float(np.max(np.abs(plan['weights']-plan['weights'].astype(np.float32).astype(np.float64)))),margin_max_error=float(np.max(np.abs(plan['margins']-plan['margins'].astype(np.float32).astype(np.float64))))))
     for branch in ['R0','R1']:
      folder=O/'trajectories'/f's{seed}-{branch}';history=read(folder/'loss_history.json');probe=read(folder/'probes.json')
      assert len(history)==200 and [r['step'] for r in history]==list(range(1,201))
      init=load(folder/'add000.pt');assert tensor_hash(init['model'])==tensor_hash(warm['model']) and not init['optimizer']['state']
      for step in [0,1,50,100,200]:
       state=load(folder/f'add{step:03d}.pt');assert state['step']==state['plan_position']==len(state['history'])==len(state['committed_global_ids'])==step
       m=JointMetricModel(12,seed=seed);m.load_state_dict(state['model']);opt=fresh_adam(m);opt.load_state_dict(copy.deepcopy(state['optimizer']))
       assert {float(v['step']) for v in opt.state.values()}==({float(step)} if step else set())
       for g in opt.param_groups:
        for k,v in CONFIG['optimizer'].items():assert (list(g[k]) if k=='betas' else g[k])==v
       for p in m.parameters():assert torch.isfinite(p).all()
       for v in opt.state.values():assert torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
       saved=read(folder/f'inference_add{step:03d}.json');now=inference(m,data,step,seed,branch);assert saved==now
       rows.append(dict(seed=seed,branch=branch,step=step,restored_model_prediction_maxdiff=0.,optimizer_moment_tensors_finite=True,optimizer_steps=step,restorable=True))
      first=load(folder/'add001.pt');firstrow=history[0]
      for state,key in [(init,'before'),(first,'after')]:
       m=JointMetricModel(12,seed=seed);m.load_state_dict(state['model'])
       with torch.no_grad():
        e,p=m(data['train']['z'],data['train']['hp']);v=objective(e,p,target,tensor_plan(plan,0),CONFIG['lambda_r'][branch])
       expected=firstrow if key=='before' else firstrow['after_same_sample']
       for name,val in v.items():assert float(val)==expected[name]
      for name in warm['model']:
       actual=float((first['model'][name]-init['model'][name]).abs().max());assert actual==firstrow['parameter_max_changes'][name]
      assert all(r['plan_position_used']==r['step']-1 for r in history)
      assert all(r['lambda_r']==CONFIG['lambda_r'][branch] and r['lambda_e']==.001 for r in history)
      assert all(math.isfinite(r[k]) for r in history for k in ['total','regression','order','size','gradient_norm'])
    # Original source model, candidate, label, encoding and source records have not changed.
    original=read(O/'input_sources.json');assert all(file_hash(p)==h for p,h in original.items())
    assert cost_status(O)==cost_before
    resume=read(O/'invocations/run02.json');assert resume['optimizer_updates_this_invocation']==0 and resume['before']==resume['after']
    write(O/'runtime_validation.json',dict(passed=True,actual_checkpoints_restored=30,checks=rows,first_step_losses_recomputed_exact=True,
     first_step_parameter_changes_recomputed_exact=True,all_step_losses_finite=True,all_original_sources_unchanged=True,float64_to_float32=casts,
     completed_rerun_additional_updates=0,validation_optimizer_updates=0,cost=cost_before))
    print(json.dumps({'runtime_validation':'passed','checkpoints':30,'first_step_losses':'exact','resume_new_updates':0,'cost':cost_before}))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=O);verify(p.parse_args().output.resolve())
