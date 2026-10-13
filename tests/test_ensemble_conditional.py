"""Bounded preflight: six synthetic optimizer steps; no new historical queries."""
import argparse,copy,json,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
from analyse.relative_metric_pilot import read,write,load,tensor_hash,digest,utc
from surrogate.gp_guidance_replay_engine import reload_model,scaler_from_initial,input_tensors,rng_state,restore_rng,journal
from surrogate.champion_margin import list_indices,objective
from surrogate.budget_champion_ranking import objective as old_objective
from surrogate.ensemble_conditional import combine_scores,choose
from analyse.ensemble_conditional_engine import fit
from analyse.ensemble_conditional_rules import stage1_rule,stage2_rule,paired

def guard(o):
    from analyse.champion_margin_run import install_guard
    p=read(o/'draft_protocol.json');task=read(o/'tasks.json')[0];folder=o/'validation/guard';folder.mkdir(parents=True,exist_ok=True)
    other=o/'validation/another_trajectory/reveal.json';write(other,dict(synthetic_receipt=True))
    install_guard(o,folder,task,p);read(o/task['label_file']);opt=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]);assert not opt.state
    denied=0
    for f in [Path(p['A_labels_path']),o/f"initial_labels/{p['config']['stage1_seeds'][1]}.json",other]:
        try:read(f)
        except PermissionError:denied+=1
        else:raise AssertionError('hidden/other-trajectory labels accessible')
    assert denied==3;print(json.dumps(dict(own_initial_allowed=True,oracle_other_seed_other_trajectory_blocked=True,B_test_opened=False,Adam_initialization=True)))

def run(o):
    if (o/'preflight_checks.json').exists():assert read(o/'preflight_checks.json')['passed'];return
    p=read(o/'draft_protocol.json');cfg=p['config'];features=read(o/'features.json');initial=read(o/f"initial_labels/{cfg['stage1_seeds'][0]}.json")
    scaler=scaler_from_initial(features,sorted(initial),initial);data=input_tensors(features,scaler,initial)
    weights=[load(o/f'initial_weights/member{j}.pt')['model'] for j in range(5)];assert len({tensor_hash(w) for w in weights})==5
    m1=reload_model(weights[0]);m2=reload_model(weights[0]);assert sum(p.numel() for p in m1.parameters())==2337
    e1,s1=m1(data['z'],data['hp']);e2,s2=m2(data['z'],data['hp']);assert torch.equal(e1,e2) and torch.equal(s1,s2)
    lists=list_indices(data['ids'],'replay',f"start{cfg['stage1_seeds'][0]}-s0",'initial',0,cfg)
    old,_=old_objective(e1,s1,data['target32'],data['y'],data['ids'],lists,50/159,0.)
    new,_=objective(e2,s2,data['target32'],data['y'],data['ids'],lists,50/159,scaler['y_scale'],cfg,'fixed',0.)
    assert all(torch.equal(old[k],new[k]) for k in old)
    assert all(torch.equal(a,b) for a,b in zip(torch.autograd.grad(old['total'],tuple(m1.parameters())),torch.autograd.grad(new['total'],tuple(m2.parameters()))))
    ids=[f'c{i:03}' for i in range(179)];members=[{c:(i%11)*.01+j*.001*((i+j)%7) for i,c in enumerate(ids)} for j in range(5)]
    rows=combine_scores(members);assert all(abs(r['sd']-float(np.std(r['member_scores'],ddof=1)))<1e-14 for r in rows)
    before=rng_state();a=choose(rows,[], 'E_NOMINATE',1,17,0,cfg);b=choose(rows,[],'E_NOMINATE',1,17,0,cfg);after=rng_state()
    assert a==b and torch.equal(before['torch'],after['torch']) and before['python']==after['python'] and np.array_equal(before['numpy'][1],after['numpy'][1])
    assert a['nomination']['eligible_count']==175 and len(set(a['selected_ids']))==5
    j=a['nomination']['member'];eligible=[r for r in rows if r['candidate_id'] not in a['selected_ids'][:4]]
    assert a['selected_ids'][-1]==min(eligible,key=lambda r:(-r['member_scores'][j],r['candidate_id']))['candidate_id']
    ties=combine_scores([{c:1. for c in ids} for _ in range(5)]);assert choose(ties,[],'E_MEAN',1,17,0,cfg)['selected_ids']==ids[:5]
    for method in ['E_MEAN','E_NOMINATE']:
        known=set(ids[:20]);origins=[]
        for batch in range(10):
            current=combine_scores([{c:v for c,v in m.items() if c not in known} for m in members]);choice=choose(current,known,method,1,17,batch,cfg)
            known.update(choice['selected_ids']);origins.extend(choice['sources'])
        assert len(known)==70 and (method!='E_NOMINATE' or origins.count('nomination')==10)
    base={'C_GREEDY':dict(H=1,R=.01,AUC_regret=.02),'E_MEAN':dict(H=1,R=.01,AUC_regret=.02),'E_NOMINATE':dict(H=1,R=.01,AUC_regret=.02)}
    assert stage1_rule(base)['winner'] is None
    x=copy.deepcopy(base);x['E_NOMINATE']['H']=2;assert stage1_rule(x)['winner']=='E_NOMINATE'
    x['E_MEAN']['H']=2;assert stage1_rule(x)['winner']=='E_MEAN'
    x['E_NOMINATE']['AUC_regret']=.019;assert stage1_rule(x)['winner']=='E_NOMINATE'
    x['E_NOMINATE']['R']=.02;assert stage1_rule(x)['qualified']==['E_MEAN']
    rows2=[dict(seed=s,method=m,initial_contains_champion=False,hit=(m=='E_MEAN'),gap_accuracy=0. if m=='E_MEAN' else .01) for s in range(20) for m in ['E_MEAN','C_GREEDY','RANDOM']]
    assert stage2_rule(rows2,'E_MEAN')['status']=='A_POOL_START_ROBUSTNESS_PASS'
    for r in rows2:r['hit']=r['seed']<(2 if r['method']=='E_MEAN' else 1)
    assert stage2_rule(rows2,'E_MEAN')['status']=='A_POOL_TREND_INCONCLUSIVE'
    for r in rows2:r['hit']=False
    assert stage2_rule(rows2,'E_MEAN')['status']=='A_POOL_PRIMARY_NOT_MET'
    for r in rows2:r['initial_contains_champion']=r['seed']<11
    assert stage2_rule(rows2,'E_MEAN')['status']=='INSUFFICIENT_OPPORTUNITIES'
    assert paired([dict(hit=False)],[dict(hit=False)])['p']==1.
    probe=subprocess.run([sys.executable,'-B',__file__,'--output',str(o),'--guard'],capture_output=True,text=True,check=True)
    physical=sum(r['phase']=='computed' for f in (o/'validation/synthetic').glob('*/physical.jsonl') for r in journal(f));assert physical+6<=32
    torch.manual_seed(1917);np.random.seed(1917);random.seed(1917)
    synth=dict(ids=[f's{i:02}' for i in range(20)],z=torch.randn(20,12),hp=torch.randn(20,4),y=torch.linspace(.2,.8,20,dtype=torch.double),scaler=dict(y_mean=.5,y_scale=.2))
    synth['target32']=((synth['y']-.5)/.2).float();meta=dict(owner='synthetic',seed=1917,event='initial',q=0,member=0,stage=0);stamp=str(time.time_ns())
    full=fit(o/'validation/synthetic'/f'{stamp}-full',reload_model(weights[0]),synth,3,cfg,meta,{'remaining':0})
    folder=o/'validation/synthetic'/f'{stamp}-resume'
    try:fit(folder,reload_model(weights[0]),synth,3,cfg,meta,{'remaining':0},pause_at=2)
    except SystemExit as e:assert e.code==75
    else:raise AssertionError('pause not reached')
    saved=load(folder/'latest.pt');restore_rng(saved['rng']);expected=(torch.rand(3),np.random.rand(3),random.random());restore_rng(saved['rng'])
    actual=(torch.rand(3),np.random.rand(3),random.random());assert torch.equal(expected[0],actual[0]) and np.array_equal(expected[1],actual[1]) and expected[2]==actual[2]
    resumed=fit(folder,reload_model(weights[0]),synth,3,cfg,meta,{'remaining':0})
    assert tensor_hash(full['model'])==tensor_hash(resumed['model']) and full['last_log']['list_hash']==resumed['last_log']['list_hash']
    for k,state in full['optimizer']['state'].items():
        for n in ['step','exp_avg','exp_avg_sq']:assert torch.equal(state[n],resumed['optimizer']['state'][k][n])
    for n in p['source_names']:
        if 'ensemble_conditional' in n and n.endswith('.py'):compile((ROOT/n).read_text(),n,'exec')
    assert subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True).returncode==0
    write(o/'preflight_checks.json',dict(passed=True,time=utc(),oldC_forward_loss_gradients_exact=True,parameters=2337,independent_initial_weights=5,
        selection_exclusion_ties_full_eligibility=True,sample_sd_denominator=4,nomination_reproducible_local_rng=True,synthetic_normal_50_queries=True,
        decision_all_branches=True,information_boundary=json.loads(probe.stdout),synthetic_optimizer_steps=6,optimizer_model_RNG_resume_exact=True,
        formal_member_and_batch_boundary_checks='scheduled once in first E_MEAN trajectory',syntax_and_git_diff_check=True))
    print('preflight passed; synthetic updates=6')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--guard',action='store_true');a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    guard(a.output.resolve()) if a.guard else run(a.output.resolve())
