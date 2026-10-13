"""Protocol-specific synthetic checks; actual optimization <=60 per kind."""
import sys,copy,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from analyse.fuzzy_scheduler_regularization_core import *
from analyse.fuzzy_scheduler_regularization_run import split_inner,choose_strength,GRID


def scheduler_checks():
    evidence=[]
    for kind,t in [('gp',200),('mapping',400)]:
        rule=RULES[kind];interval=rule['interval'];guard=GUARD[kind]
        m=torch.nn.Linear(1,1);op=adam(m.parameters(),rule['lr']);s=base_state(kind,'test',{})['scheduler'];s.update(best=-1.,stagnation=2)
        no=dict(valid=False,passed=False);yes=dict(valid=True,passed=True)
        schedule(kind,s,no,-1.,t,op);assert s['lr_epoch_start']==t+1 and s['cooldown_until']==t+guard
        for step in range(t+interval,t+guard+1,interval):
            decision=yes if step>=t+rule['width']+interval else no
            schedule(kind,s,decision,-1.,step,op)
            assert len(s['lr_changes'])==1 and s['stagnation']==0
        assert s['stable_count']==3
        assert [r['step'] for r in s['events'] if r['action'] in ['passing_window_pause','stable_stop']]==[t+guard-2*interval,t+guard-interval,t+guard]
        # Interrupted scheduler/Adam/RNG state inside protection preserves the remaining window.
        a=base_state(kind,'restore',{})['scheduler'];a.update(best=-1.,stagnation=2);op2=adam(m.parameters(),rule['lr']);schedule(kind,a,no,-1.,t,op2)
        saved=copy.deepcopy(a);ops=copy.deepcopy(op2.state_dict());rng=rng_state()
        for step in range(t+interval,t+guard+4*interval,interval):schedule(kind,a,no,-1.,step,op2,cap=t+4*guard)
        b=copy.deepcopy(saved);op3=adam(m.parameters(),rule['lr']);op3.load_state_dict(ops);restore_rng(rng)
        for step in range(t+interval,t+guard+4*interval,interval):schedule(kind,b,no,-1.,step,op3,cap=t+4*guard)
        assert tree_hash(a)==tree_hash(b) and tree_hash(op2.state_dict())==tree_hash(op3.state_dict())
        assert a['lr_changes'][1]['after_step']==t+guard+3*interval
        for remaining,expected in [(guard,'lr_reduced'),(guard-1,'lr_reduction_blocked_insufficient_budget'),(0,'at_cap_no_next_update')]:
            c=base_state(kind,'cap',{})['scheduler'];c.update(best=1.,stagnation=2);opt=adam(m.parameters(),rule['lr']);schedule(kind,c,no,1.,t,opt,cap=t+remaining);assert c['events'][-1]['action']==expected
        c=base_state(kind,'minimum',{})['scheduler'];c.update(best=1.,stagnation=2,lr_epoch_start=17);opt=adam(m.parameters(),rule['min_lr']);schedule(kind,c,no,1.,t,opt);assert c['lr_epoch_start']==17 and not c['lr_changes']
        c.update(stagnation=2);schedule(kind,c,yes,1.,t+interval,opt);assert c['stable_count']==1 and not c['lr_changes']
        evidence.append(dict(kind=kind,changes=s['lr_changes'],events=s['events'],resume_exact=True,cap_boundaries_passed=True))
    return evidence


def window_checks():
    h=[dict(step=t,total=1001.,L0=1.+.01*(t/200),weighted_regression=1.+.01*(t/200),weighted_neighborhood=0.,weighted_scale=0.,weighted_parameter=1000.) for t in range(201)]
    ss=[dict(step=t,gradient=dict(rms=0.,max=0.),distance=torch.tensor([1.,2.,3.]),neighbors=[['a','b','c','d','e']]) for t in [100,120,140,160,180,200]]
    state=dict(lr_epoch_start=0,g0=dict(rms=1.,max=1.),reference=dict(norm=float(torch.tensor([1.,2.,3.]).norm()),median=2.))
    d=window_decision('mapping',h,ss,200,state,0.,.001)
    assert d['checks']['loss'] and not d['checks']['L0'] and not d['checks']['weighted_regression']
    for kind,t in [('gp',200),('mapping',400)]:
        r=RULES[kind];st=dict(state,lr_epoch_start=t+1)
        history=[dict(step=i,total=-1.,L0=-1.,weighted_regression=-1.,weighted_neighborhood=0.,weighted_scale=0.,weighted_parameter=0.) for i in range(t+GUARD[kind]+1)]
        shots=[dict(step=i,gradient=dict(rms=0.,max=0.),distance=torch.tensor([1.,2.,3.]),neighbors=[['a','b','c','d','e']],prediction=torch.zeros(3),std=torch.ones(3)) for i in range(0,t+GUARD[kind]+1,r['interval'])]
        assert not window_decision(kind,history,shots,t+r['width'],st)['valid']
        assert all(window_decision(kind,history,shots,i,st)['passed'] for i in [t+GUARD[kind]-2*r['interval'],t+GUARD[kind]-r['interval'],t+GUARD[kind]])
    return dict(large_regularizer_cannot_hide_L0=True,full_window_timing=True)


def tuning_checks():
    features=dict(ids=[f'c{i:02}' for i in range(24)],architectures=[f'a{i//2}' for i in range(24)])
    spl=split_inner(features,features['ids'],123);assert spl['sufficient'];arch=dict(zip(features['ids'],features['architectures']))
    for row in spl['folds']:assert not {arch[c] for c in row['train_ids']}&{arch[c] for c in row['validation_ids']}
    assert spl==split_inner(features,features['ids'],123)
    rows=[dict(lambda_w=w,inner_fold=i,valid=True,r2=.1,r1=.2,mae=.3) for w in GRID for i in range(3)]
    assert choose_strength(rows)['lambda_w']==0.
    for metric in ['r2','r1','mae']:
        rr=copy.deepcopy(rows)
        for row in rr:
            if row['lambda_w']==1e-4:row[metric]-=.01
        chosen=choose_strength(rr);assert chosen['lambda_w']==1e-4 and chosen['deciding_metric']==metric
    rows[0]['valid']=False;assert choose_strength(rows)['lambda_w']==1e-5
    for row in rows:row['valid']=False
    assert choose_strength(rows)['status']=='TUNING_FAILED_FALLBACK_ZERO'
    assert choose_strength([],False)['status']=='INNER_VALIDATION_INSUFFICIENT'
    return dict(grouped_split=True,selection_order=True,zero_and_ties=True,failures_and_insufficient=True)


def check(o):
    o.mkdir(parents=True,exist_ok=True);assert not (o/'synthetic_checks.json').exists(),'synthetic ledger must not be silently reset'
    schedule_result=scheduler_checks();window_result=window_checks();tuning=tuning_checks()
    features=dict(ids=[f'x{i:02}' for i in range(12)],architectures=[f'a{i}' for i in range(12)],z=np.random.default_rng(4).normal(size=(12,12)).tolist(),hp=np.random.default_rng(5).normal(size=(12,4)).tolist())
    labels={c:dict(training_seeds=[0,1,2],validation_mean=.5+i*.001,validation_seeds=[.499+i*.001,.5+i*.001,.501+i*.001],validation_se=.001/(3**.5)) for i,c in enumerate(features['ids'][:9])}
    # _label_arrays requires exactly the archived three-seed label schema.
    scaler=scaler_from_training(features,sorted(labels),labels);data=observed_data(features,scaler,labels,.01);initial=JointMetricModel(12,seed=0).state_dict();m=reload_model(initial)
    old,_,_,_=original_components(m,data,.1);old_grad=torch.autograd.grad(old['total'],tuple(m.parameters()))
    new,nums,_,_=components(m,data,.1,0.);new_grad=torch.autograd.grad(new['total'],tuple(m.parameters()))
    assert torch.equal(old['total'],new['total']) and all(torch.equal(a,b) for a,b in zip(old_grad,new_grad))
    reg,nums,_,_=components(m,data,.1,.001);reggrad=torch.autograd.grad(reg['total'],tuple(m.parameters()))
    expected=sum(float(p.detach().square().sum()) for p in m.parameters());assert abs(nums['parameter_squared_sum']-expected)<1e-4
    assert sum(p.numel() for p in m.parameters())==2337
    for g,b,p in zip(reggrad,old_grad,m.parameters()):assert torch.allclose(g,b+.002*p,atol=1e-7,rtol=1e-5)
    ledger=Budget(o/'synthetic/ledger','synthetic',synthetic=True);meta=dict(stage='inner_cv',task='synthetic',epsilon=.01)
    rng=rng_state();a=fit_mapping(o/'synthetic/map-cont',initial,data,.1,ledger,meta,cap_override=10,lambda_w=.001)
    restore_rng(rng);b=fit_mapping(o/'synthetic/map-resume',initial,data,.1,ledger,meta,cap_override=10,lambda_w=.001,stop_after=4);b=fit_mapping(o/'synthetic/map-resume',initial,data,.1,ledger,meta,cap_override=10,lambda_w=.001)
    assert tree_hash(a['model'])==tree_hash(b['model']) and tree_hash(a['optimizer'])==tree_hash(b['optimizer']) and tree_hash(a['rng'])==tree_hash(b['rng']) and a['scheduler']==b['scheduler']
    cache=embedding_cache(reload_model(a['model']),data);rng=rng_state();g=fit_gp(o/'synthetic/gp-cont',cache,data,ledger,meta,cap_override=10)
    restore_rng(rng);j=fit_gp(o/'synthetic/gp-resume',cache,data,ledger,meta,cap_override=10,stop_after=4);j=fit_gp(o/'synthetic/gp-resume',cache,data,ledger,meta,cap_override=10)
    assert tree_hash(g['model'])==tree_hash(j['model']) and tree_hash(g['optimizer'])==tree_hash(j['optimizer']) and tree_hash(g['rng'])==tree_hash(j['rng']) and g['scheduler']==j['scheduler']
    for st in [a,b,g,j]:
        assert st['stop_ids' if st['kind']=='mapping' else 'probe_ids']==sorted(labels)
        assert len(st['history'][1:])==10 and all(h['lr_used_for_update']==h['lr_configured_next'] for h in st['history'][1:])
    result=dict(passed=True,scheduler=schedule_result,window=window_result,tuning=tuning,zero_loss_gradient_exact=True,explicit_sum_all_2337_including_bias=True,regularizer_gradient_matches_2lambda_parameter=True,map_and_gp_restore_exact=True,inner_stop_ids_training_only=True,synthetic_updates=ledger.counts,time=utc())
    write(o/'synthetic_checks.json',result);print(json.dumps(dict(passed=True,synthetic_updates=ledger.counts)))
if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);check(Path(sys.argv[1]).resolve())


def protected_window_resume(o):
    import analyse.fuzzy_scheduler_regularization_core as core
    assert not (o/'synthetic_resume_windows.json').exists()
    # Toy intervals exercise an actual active cooldown with only 40 extra updates per kind.
    original_rules=copy.deepcopy(core.RULES);original_guard=copy.deepcopy(core.GUARD);original_schedule=core.schedule
    for kind in core.RULES:core.RULES[kind].update(prefix=2,interval=2,width=6,cap=20);core.GUARD[kind]=12
    def plateau(kind,state,decision,loss,step,optimizer,cap=None):
        return original_schedule(kind,state,dict(decision,passed=False),1.,step,optimizer,cap)
    core.schedule=plateau
    prior=load(o/'synthetic/map-cont/latest.pt');gp_prior=load(o/'synthetic/gp-cont/latest.pt')
    features=dict(ids=[f'x{i:02}' for i in range(12)],architectures=[f'a{i}' for i in range(12)],z=np.random.default_rng(4).normal(size=(12,12)).tolist(),hp=np.random.default_rng(5).normal(size=(12,4)).tolist())
    labels={c:dict(training_seeds=[0,1,2],validation_mean=.5+i*.001,validation_seeds=[.499+i*.001,.5+i*.001,.501+i*.001],validation_se=.001/(3**.5)) for i,c in enumerate(features['ids'][:9])}
    data=observed_data(features,prior['scaler'],labels,.01);ledger=Budget(o/'synthetic/ledger','synthetic',synthetic=True);meta=dict(stage='inner_cv',task='synthetic-window',epsilon=.01);states=[]
    for kind in ['mapping','gp']:
        def fit(dest,stop=None):
            if kind=='mapping':return core.fit_mapping(dest,JointMetricModel(12,seed=0).state_dict(),data,.1,ledger,meta,lambda_w=.001,stop_after=stop)
            return core.fit_gp(dest,gp_prior['cache'],data,ledger,meta,stop_after=stop)
        rng=rng_state();a=fit(o/'synthetic'/f'{kind}-window-cont');restore_rng(rng);b=fit(o/'synthetic'/f'{kind}-window-resume',12)
        assert b['scheduler']['lr_epoch_start']==9 and b['scheduler']['cooldown_until']==20 and b['completed_updates']==12
        save(o/'synthetic'/f'{kind}-window-interrupted.pt',b)
        b=fit(o/'synthetic'/f'{kind}-window-resume')
        for key in ['model','optimizer','rng','scheduler','history','decisions']:
            assert tree_hash(a[key])==tree_hash(b[key]),(kind,key)
        assert a['history'][8]['lr_used_for_update']!=a['history'][8]['lr_configured_next']
        assert a['history'][9]['lr_used_for_update']==a['history'][8]['lr_configured_next']
        assert len(a['scheduler']['lr_changes'])==1
        states.append(dict(kind=kind,interrupted_step=12,active_cooldown_until=20,all_saved_state_exact=True,lr_use_counts=summary(a)['learning_rate_use_counts']))
    core.RULES.clear();core.RULES.update(original_rules);core.GUARD.clear();core.GUARD.update(original_guard);core.schedule=original_schedule
    write(o/'synthetic_resume_windows.json',dict(passed=True,toy_rules=True,not_real_data=True,states=states,total_synthetic_updates=ledger.counts,time=utc()))
    print(json.dumps(dict(passed=True,total_synthetic_updates=ledger.counts)))
