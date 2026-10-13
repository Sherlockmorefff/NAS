"""Required bounded checks; real-label comparisons have zero optimizer steps."""
import argparse,copy,json,math,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
from analyse.relative_metric_pilot import read,write,load,save,digest,tensor_hash,utc
from surrogate.gp_guidance_replay_engine import reload_model,scaler_from_initial,input_tensors,score_candidates,rng_state,restore_rng,journal,append
from surrogate.budget_champion_ranking import list_indices as old_lists,objective as old_objective
from surrogate.champion_margin import list_indices,objective,select_batch
from analyse.champion_margin_engine import fit


def guard_probe(o):
    from analyse.champion_margin_run import install_guard
    p=read(o/'protocol.json');task=read(o/'tasks.json')[0];folder=o/'trajectories'/task['name'];folder.mkdir(parents=True,exist_ok=True)
    install_guard(o,folder,task,p)
    optimizer=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    assert not optimizer.state
    read(o/task['label_file']);blocked=0
    # Probe only A label locations; never attempt opening B/test files.
    for path in [Path(p['A_labels_path']),o/'initial_labels'/f"{p['config']['search_seeds'][1]}.json"]:
        try:read(path)
        except PermissionError:blocked+=1
        else:raise AssertionError('unrevealed labels accessible')
    assert blocked==2;print(json.dumps(dict(allowed_own_initial=True,blocked_A_oracle_and_other_start=2,optimizer_initialization_under_guard=True)))


def run(o):
    if (o/'checks.json').exists() and read(o/'checks.json').get('preflight_passed'):print('preflight already passed, no extra synthetic steps');return
    p=read(o/'protocol.json');cfg=p['config'];v=o/'validation';v.mkdir(exist_ok=True);results={}
    base=ROOT/'results/posthoc/budget-champion-ranking-v1';features=read(o/'features.json');seed=cfg['search_seeds'][0];initial=read(o/f'initial_labels/{seed}.json')
    scaler=scaler_from_initial(features,sorted(initial),initial);data=input_tensors(features,scaler,initial)
    # Exact archived data/list compatibility at a real saved forward, without fitting.
    d=base/'replay'/f'start{seed}-s0-BR_HEAD'/'events/initial';sample=load(d/'diagnostics/step001.pt');state=load(d/'step000.pt')
    assert sample['ids']==data['ids'] and scaler==state['scaler']
    ls=list_indices(data['ids'],'replay',f'start{seed}-s0','initial',0,cfg)
    assert ls==old_lists(data['ids'],'replay',f'start{seed}-s0','initial',0)==sample['lists']
    comparisons=[]
    for weight in [0.,cfg['ranking_weight']]:
        m1=reload_model(state['model']);m2=reload_model(state['model']);e1,s1=m1(data['z'],data['hp']);e2,s2=m2(data['z'],data['hp'])
        old,_=old_objective(e1,s1,data['target32'],data['y'],data['ids'],ls,50/159,weight)
        new,_=objective(e2,s2,data['target32'],data['y'],data['ids'],ls,50/159,scaler['y_scale'],cfg,'fixed',weight)
        for k in old:assert torch.equal(old[k],new[k]),k
        g1=torch.autograd.grad(old['total'],tuple(m1.parameters()));g2=torch.autograd.grad(new['total'],tuple(m2.parameters()))
        assert all(torch.equal(a,b) for a,b in zip(g1,g2))
        comparisons.append(dict(weight=weight,losses_exact=True,all_parameter_gradients_exact=True))
    results['old_behavior_compatibility']=comparisons
    model=reload_model(load(d/'latest.pt')['model']);rows,_,_=score_candidates(model,data)
    oldsel=read(base/'replay'/f'start{seed}-s0-BR_HEAD/batches/b00/selection.json')
    assert rows==oldsel['rows'];assert select_batch(rows,initial,seed,0,cfg,'greedy')['selected_ids']==oldsel['selected_ids']
    results['saved_pure_greedy_scores_and_selection_exact']=True
    # Known real labels with small standardized separation: exact predictions need no gap penalty.
    ys=data['target64'];lossfixed,_=objective(torch.zeros(len(ys),8),ys,data['target64'],data['y'],data['ids'],ls,50/159,scaler['y_scale'],cfg,'fixed',.1)
    lossgap,_=objective(torch.zeros(len(ys),8),ys,data['target64'],data['y'],data['ids'],ls,50/159,scaler['y_scale'],cfg,'gap',.1)
    assert float(lossgap['ranking'])==0
    results['perfect_real_known_outputs']=dict(fixed=float(lossfixed['ranking']),gap=float(lossgap['ranking']))
    ids=['a','b','c'];raw=torch.tensor([.800,.799,.700],dtype=torch.double);lists=[[0,1]]*16
    perfect=(raw-.7)/.2
    fixed,_=objective(torch.zeros(3,8),perfect,perfect,raw,ids,lists,.5,.2,cfg,'fixed',.1)
    gap,_=objective(torch.zeros(3,8),perfect,perfect,raw,ids,lists,.5,.2,cfg,'gap',.1)
    assert float(fixed['ranking'])>0 and float(gap['ranking'])==0
    scores=torch.tensor([0.,1.,-2.],requires_grad=True)
    loss,diag=objective(torch.zeros(3,8),scores,perfect,raw,ids,lists,.5,.2,cfg,'gap',.1)
    grad=torch.autograd.grad(loss['ranking'],scores)[0];assert grad[0]<0 and grad[1]>0
    tied=torch.tensor([.8,.8,.7],dtype=torch.double)
    zz=torch.tensor([0.,0.,0.],requires_grad=True)
    loss,diag=objective(torch.zeros(3,8),zz,zz.detach(),tied,ids,[[0,1,2]]*16,1.,.2,cfg,'gap',.1)
    assert float(loss['ranking'])==0 and torch.equal(torch.autograd.grad(loss['ranking'],zz)[0],torch.zeros(3))
    assert set(diag['competitor_indices'])=={None}
    # k=2 must use second nonchampion; 1 active and 15 inactive lists keep denominator 16.
    ids4=['a','b','c','d'];raw4=torch.tensor([.8,.79,.7,.6],dtype=torch.double);s4=torch.tensor([0.,1.,-1.,-2.],requires_grad=True)
    a,di=objective(torch.zeros(4,8),s4,s4.detach(),raw4,ids4,[[0,1,2,3]]*16,.5,.2,cfg,'fixed',.1)
    assert di['competitor_indices']==[2]*16
    mixed_lists=[[0,1]]+[[0,2]]*15
    a,di=objective(torch.zeros(4,8),s4,s4.detach(),raw4,ids4,mixed_lists,.5,.2,cfg,'gap',.1)
    expected=(1.+min(.1,.5*(.8-.79)/.2))**2/16
    assert math.isclose(float(a['ranking']),expected,rel_tol=1e-6) and di['triggered_lists']==1
    results['formula_branches']=dict(small_gap_zero=True,wrong_order_gradient_direction=True,ties_and_insufficient_competitor_zero=True,k_index=True,mean_over_all16=True)
    # Local RNG independence, full eligible set, stable reservations and two greedy batches.
    before=rng_state();choice=select_batch(rows,initial,seed,0,cfg,'mixed');after=rng_state()
    assert torch.equal(before['torch'],after['torch']) and before['python']==after['python'] and np.array_equal(before['numpy'][1],after['numpy'][1])
    assert choice==select_batch(rows,initial,seed,0,cfg,'mixed') and len(set(choice['selected_ids']))==5
    assert choice['exploration']['eligible_count']==len(rows)-4 and choice['sources']==['greedy']*4+['exploration']
    first=select_batch(rows,initial,seed,0,cfg,'greedy')['selected_ids'];rest=[r for r in rows if r['candidate_id'] not in first]
    assert first+select_batch(rest,set(initial)|set(first),seed,1,cfg,'greedy')['selected_ids']==[r['candidate_id'] for r in rows[:10]]
    # All ten mixed batches, no model training, must consume 40+10 unique IDs.
    known=set(initial);sources=[]
    for b in range(10):
        current=[r for r in rows if r['candidate_id'] not in known];pick=select_batch(current,known,seed,b,cfg,'mixed');known.update(pick['selected_ids']);sources.extend(pick['sources'])
    assert len(known)==70 and sources.count('greedy')==40 and sources.count('exploration')==10
    results['selection_and_rng']=dict(local_rng_unchanged=True,full_eligible_population=True,deterministic_restore=True,pure_two5_equal_top10=True,mixed_unique_queries=50,greedy_slots=40,exploration_slots=10)
    probe=subprocess.run([sys.executable,'-B',__file__,'--output',str(o),'--guard-probe'],text=True,capture_output=True,check=True)
    results['information_boundary']=json.loads(probe.stdout)
    # Exactly 6 synthetic optimizer steps: uninterrupted3 versus interrupted2+1.
    done_physical=sum(len([x for x in journal(path) if x['phase']=='computed']) for path in (v/'synthetic').glob('*/physical_attempts.jsonl'))
    assert done_physical+6<=cfg['synthetic_update_limit']
    attempt=str(time.time_ns());torch.manual_seed(1927);np.random.seed(1927);random.seed(1927)
    synth=dict(ids=[f's{i:02}' for i in range(20)],z=torch.randn(20,12),hp=torch.randn(20,4),y=torch.linspace(.2,.8,20,dtype=torch.double),scaler=dict(y_scale=.2,y_mean=.5))
    synth['target32']=((synth['y']-.5)/.2).float();weights=load(o/f'initial_weights/replay-start{seed}-s0.pt')['model'];m=reload_model(weights)
    meta=dict(owner='synthetic',task_id='start1927-s0',update_event='initial',stage='replay')
    full=fit(v/'synthetic'/f'{attempt}-full',m,synth,3,cfg,'gap',.1,50/159,meta)
    m=reload_model(weights);part=fit(v/'synthetic'/f'{attempt}-resumed',m,synth,3,cfg,'gap',.1,50/159,meta,interrupt_after=2)
    restore_rng(part['rng']);expected_rng=(torch.rand(3),np.random.rand(3),random.random());restore_rng(part['rng']);actual_rng=(torch.rand(3),np.random.rand(3),random.random())
    assert torch.equal(expected_rng[0],actual_rng[0]) and np.array_equal(expected_rng[1],actual_rng[1]) and expected_rng[2]==actual_rng[2]
    resumed=fit(v/'synthetic'/f'{attempt}-resumed',reload_model(weights),synth,3,cfg,'gap',.1,50/159,meta)
    assert tensor_hash(full['model'])==tensor_hash(resumed['model'])
    for i,a in full['optimizer']['state'].items():
        for k in ['step','exp_avg','exp_avg_sq']:assert torch.equal(a[k],resumed['optimizer']['state'][i][k])
    for folder in [v/'synthetic'/f'{attempt}-full',v/'synthetic'/f'{attempt}-resumed']:
        for r in journal(folder/'physical_attempts.jsonl'):
            if r['phase']=='computed':append(v/'synthetic_steps.jsonl',dict(source=str(folder),attempt=r['attempt'],event_key=r['event_key']))
    results['synthetic_resume']=dict(optimizer_steps=6,model_exact=True,Adam_moments_and_steps_exact=True,next_list_hash_exact=full['last_log']['list_hash']==resumed['last_log']['list_hash'],RNG_restored=True)
    for name in p['sources']:
        if name.startswith(('analyse/champion_margin','surrogate/champion_margin','tests/test_champion_margin')):compile((ROOT/name).read_text(),name,'exec')
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,text=True,capture_output=True);assert diff.returncode==0,diff.stdout+diff.stderr
    results['git_diff_check']=True
    write(o/'checks.json',dict(preflight_passed=True,checks=results,formal_boundary_resume='pending first sealed selection; verified after formal run',time=utc()))
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--guard-probe',action='store_true');args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    guard_probe(args.output) if args.guard_probe else run(args.output)
