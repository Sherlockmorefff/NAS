"""Isolated scheduler repair and explicit parameter regularization."""
from __future__ import annotations
from contextlib import contextmanager
from analyse.fuzzy_loss_interaction_core import *
from analyse.fuzzy_loss_interaction_core import components as original_components

LIMITS = dict(mapping=403000, gp=150900)
RULES = {
    'mapping': dict(prefix=100, interval=20, width=60, cap=1000, lr=.001, min_lr=1e-5,
                    loss_tol=.002, grad_rms=1e-5, grad_max=1e-4),
    'gp': dict(prefix=20, interval=10, width=30, cap=300, lr=.05, min_lr=1e-4,
               loss_tol=.001, grad_rms=1e-6, grad_max=1e-5)}
TIMES = dict(optimization_seconds=0., loss_forward_seconds=0., gradient_seconds=0.,
             geometry_seconds=0., gp_prediction_seconds=0., checkpoint_seconds=0.)


GUARD = {'mapping':120,'gp':60}

def components(model,data,lambda_f,lambda_w=0.):
    losses,numbers,e,prediction=original_components(model,data,lambda_f)
    l0=losses['total']
    theta=sum(p.square().sum() for p in model.parameters())
    # Zero uses exactly the original graph: no zero-multiplied branch.
    if lambda_w:losses['total']=l0+lambda_w*theta
    numbers.update(L0=float(l0.detach()),parameter_squared_sum=float(theta.detach()),
        weighted_parameter=lambda_w*float(theta.detach()),total=float(losses['total'].detach()),lambda_w=lambda_w)
    return losses,numbers,e,prediction


@contextmanager
def gp_numerics():
    with gpytorch.settings.cholesky_jitter(double_value=1e-6), gpytorch.settings.cholesky_max_tries(3), gpytorch.settings.max_cholesky_size(800):
        yield


class Budget:
    def __init__(self, folder, identity, synthetic=False):
        self.folder, self.identity = Path(folder), identity
        self.limits = dict(mapping=60, gp=60) if synthetic else LIMITS
        self.synthetic=synthetic
        records={k:journal(self.folder/(k+'_intents.jsonl')) for k in self.limits}
        self.counts={k:len(v) for k,v in records.items()}
        self.category_counts={(c,k):sum(self.category(r['task'])==c for r in records[k]) for c in ['engineering','offline_inner','offline_formal','replay_inner','replay_formal'] for k in self.limits}

    @staticmethod
    def category(task):
        if task.startswith('engineering-'):return 'engineering'
        if task.startswith('inner-offline'):return 'offline_inner'
        if task.startswith('inner-replay'):return 'replay_inner'
        if task.startswith('offline'):return 'offline_formal'
        return 'replay_formal'

    def step(self, kind, task, step, optimizer):
        assert self.counts[kind] < self.limits[kind], 'actual update cap: '+kind
        category=self.category(task)
        if not self.synthetic:
            limits={'engineering':(3000,900),'offline_inner':(120000,36000),'offline_formal':(60000,18000),'replay_inner':(120000,36000),'replay_formal':(100000,60000)}
            assert self.category_counts[category,kind]<limits[category][0 if kind=='mapping' else 1],'category update cap: '+category
            self.category_counts[category,kind]+=1
        self.counts[kind] += 1
        row = dict(id=self.counts[kind], task=task, step=step, protocol=self.identity, time=utc())
        append(self.folder/(kind+'_intents.jsonl'), row)
        optimizer.step()
        append(self.folder/(kind+'_receipts.jsonl'), row)

    def fit(self, metadata, identity):
        assert len(journal(self.folder/'gp_fits.jsonl')) < 503
        append(self.folder/'gp_fits.jsonl',dict(metadata=metadata,identity=identity,time=utc()))


def finite_tree(x):
    if isinstance(x,torch.Tensor): return bool(torch.isfinite(x).all())
    if isinstance(x,dict): return all(finite_tree(v) for v in x.values())
    if isinstance(x,(list,tuple)): return all(finite_tree(v) for v in x)
    if isinstance(x,float): return math.isfinite(x)
    return True


def gradient_check(loss, model):
    start=time.perf_counter();named=list(model.named_parameters())
    before=[None if p.grad is None else p.grad.clone() for _,p in named];rng=tree_hash(rng_state())
    gs=torch.autograd.grad(loss,[p for _,p in named],allow_unused=False)
    vec=torch.cat([g.detach().reshape(-1).double() for g in gs])
    assert torch.isfinite(vec).all(),'nonfinite diagnostic gradient'
    for (_,p),g in zip(named,before): assert (p.grad is None and g is None) or (g is not None and torch.equal(p.grad,g))
    assert tree_hash(rng_state())==rng
    def stats(v):return dict(rms=float(v.square().mean().sqrt()),max=float(v.abs().max()))
    result=dict(stats(vec),vector=vec,blocks={})
    for block,flag in [('shared',False),('head',True)]:
        vs=[g.detach().reshape(-1).double() for (n,_),g in zip(named,gs) if n.startswith('regressor.')==flag]
        if vs:result['blocks'][block]=stats(torch.cat(vs))
    TIMES['gradient_seconds']+=time.perf_counter()-start
    return result


def map_snapshot(model,data,stop_ids,step,losses,numbers):
    grad=gradient_check(losses['total'],model);start=time.perf_counter()
    pos={c:i for i,c in enumerate(data['all_ids'])};ix=[pos[c] for c in stop_ids]
    with torch.no_grad():
        e,head=model(data['all_z'][ix],data['all_hp'][ix]);e=e.double()
        d=torch.pdist(e,p=2)
        mu=head.double()*data['scaler']['y_scale']+data['scaler']['y_mean']
        known=[stop_ids.index(c) for c in data['ids']]
        distances=torch.cdist(e,e[known])
        neighbors=[];gaps=[]
        for i,c in enumerate(stop_ids):
            js=sorted((j for j,k in enumerate(data['ids']) if c!=k),key=lambda j:(float(distances[i,j]),data['ids'][j]))
            neighbors.append([data['ids'][j] for j in js[:5]])
            gaps.append(float(distances[i,js[5]]-distances[i,js[4]]) if len(js)>5 else 0.)
    assert finite_tree([d,e,mu])
    extra=dict(distance_q10=float(torch.quantile(d,.1)),distance_median=float(d.median()),distance_q90=float(torch.quantile(d,.9)),
               coordinate_std=e.std(0,correction=0).tolist(),embedding_norm_mean=float(e.norm(dim=1).mean()),
               head_weight_norm=numbers['head_weight_norm'],neighbor_boundary_gap_min=min(gaps))
    TIMES['geometry_seconds']+=time.perf_counter()-start
    return dict(step=step,gradient=grad,distance=d,prediction=mu,neighbors=neighbors,extra=extra)


def window_decision(kind,history,snapshots,step,state,lambda_f=0.,lambda_w=0.):
    r=RULES[kind];start=step-r['width']
    result=dict(step=step,start=start,valid=False,passed=False,failed=['incomplete_window'])
    required=list(range(start,step+1,r['interval']))
    ss={s['step']:s for s in snapshots}
    if start < r['prefix'] or start < state['lr_epoch_start'] or not all(t in ss for t in required): return result
    hs=[h for h in history if start<=h['step']<=step]
    assert len(hs)==r['width']+1 and [h['step'] for h in hs]==list(range(start,step+1))
    shots=[ss[t] for t in required];v=np.array([h['total'] for h in hs]);g0=state['g0']
    scale=max(float(np.abs(v).mean()),1e-8) if kind=='mapping' else max(1.,abs(float(v.mean())))
    threshold=r['loss_tol']*scale
    checks={'loss':float(np.ptp(v))<=threshold}
    measures=dict(loss_range=float(np.ptp(v)),loss_scale=scale,loss_threshold=threshold)
    if kind=='mapping':
        scale0=max(float(np.mean([abs(h['L0']) for h in hs])),1e-8)
        measures.update(L0_scale=scale0,L0_threshold=.002*scale0)
        for key in ['weighted_regression','weighted_neighborhood','weighted_scale','L0']:
            measures[key+'_range']=float(np.ptp([h[key] for h in hs]))
            if key!='weighted_neighborhood' or lambda_f!=0:checks[key]=measures[key+'_range']<=.002*scale0
        measures['weighted_parameter_range']=float(np.ptp([h['weighted_parameter'] for h in hs]))
        if lambda_w:checks['weighted_parameter']=measures['weighted_parameter_range']<=threshold
    gr=max(r['grad_rms'],.01*g0['rms']);gm=max(r['grad_max'],.01*g0['max'])
    measures.update(gradient_rms_threshold=gr,gradient_max_threshold=gm,
                    gradient_rms_max=max(s['gradient']['rms'] for s in shots),gradient_max_max=max(s['gradient']['max'] for s in shots))
    checks['gradient_rms']=measures['gradient_rms_max']<=gr;checks['gradient_max']=measures['gradient_max_max']<=gm
    if kind=='mapping':
        ds=torch.stack([s['distance'] for s in shots]);ref=state['reference']
        l2=max(float((a-b).norm())/ref['norm'] for a in ds for b in ds)
        pair=float((ds.max(0).values-ds.min(0).values).max())/ref['median']
        checks.update(distance_l2=l2<=.01,distance_pair=pair<=.02)
        measures.update(distance_l2=l2,distance_pair=pair,
            scale_removed_shape_change=max(float((a/a.norm().clamp_min(1e-8)-b/b.norm().clamp_min(1e-8)).norm()) for a in ds for b in ds),
            neighbor_retention_min=min(sum(len(set(a)&set(b))/5 for a,b in zip(shots[i]['neighbors'],shots[j]['neighbors']))/len(shots[i]['neighbors']) for i in range(4) for j in range(4)))
    else:
        mus=torch.stack([s['prediction'] for s in shots]);stds=torch.stack([s['std'] for s in shots])
        drift=float((mus.max(0).values-mus.min(0).values).max())
        checks['prediction']=drift<=.0005
        measures.update(prediction_range=drift,prediction_threshold=.0005,std_range=float((stds.max(0).values-stds.min(0).values).max()))
    failed=[k for k,v in checks.items() if not v]
    return dict(step=step,start=start,valid=True,passed=not failed,failed=failed,checks=checks,measures=measures)


def schedule(kind,state,decision,loss,step,optimizer,cap=None):
    r=RULES[kind];cap=r['cap'] if cap is None else cap
    state.setdefault('cooldown_until',0);state.setdefault('events',[])
    before=copy.deepcopy({k:v for k,v in state.items() if k not in ['events','reference','g0','lr_changes']})
    state['stable_count']=state['stable_count']+1 if decision['valid'] and decision['passed'] else 0
    action='prefix'
    if decision['valid'] and decision['passed']:
        action='stable_stop' if state['stable_count']>=3 else 'passing_window_pause'
    elif step==r['prefix']:state['best']=loss
    elif step>r['prefix']:
        best=state['best'];threshold=max(1e-8,.002*abs(best)) if kind=='mapping' else .001*max(1.,abs(best))
        improved=best-loss>threshold
        if improved:state['best']=loss
        if step<=state['cooldown_until']:
            state['stagnation']=0;action='cooldown'
        elif improved:state['stagnation']=0;action='improvement'
        else:
            state['stagnation']+=1;action='stagnation'
            if state['stagnation']>=3:
                old=optimizer.param_groups[0]['lr'];new=max(r['min_lr'],old*.5);state['stagnation']=0
                if step>=cap:action='at_cap_no_next_update'
                elif cap-step<GUARD[kind]:action='lr_reduction_blocked_insufficient_budget'
                elif new==old:action='minimum_learning_rate_no_reset'
                else:
                    for group in optimizer.param_groups:group['lr']=new
                    state.update(stable_count=0,lr_epoch_start=step+1,cooldown_until=step+GUARD[kind])
                    state['lr_changes'].append(dict(after_step=step,old=old,new=new,first_new_lr_post_step=step+1,cooldown_until=state['cooldown_until']))
                    action='lr_reduced'
    state['events'].append(dict(step=step,action=action,loss=loss,before=before,
        after={k:v for k,v in state.items() if k not in ['events','reference','g0','lr_changes']},lr_configured_next=optimizer.param_groups[0]['lr'],remaining_budget=cap-step))


def terminal_flags(kind,state,decisions,snapshot,data=None):
    last=decisions[-3:];sufficient=len(last)==3 and all(d['valid'] for d in last)
    return dict(stable=state['stable_count']>=3,insufficient_observation=not sufficient,
        degenerate=bool(kind=='mapping' and float(data['y'].max()-data['y'].min())>0 and float(snapshot['distance'].max())<=1e-8),
        final_failed_conditions=sorted(set(k for d in last for k in d['failed'])),
        plateau_large_gradient=sufficient and all(d['failed'] and all(k.startswith('gradient_') for k in d['failed']) for d in last))


def terminal_status(kind,state,decisions,snapshot,data=None):
    f=terminal_flags(kind,state,decisions,snapshot,data)
    if f['degenerate']:return 'DEGENERATE_GEOMETRY'
    if f['stable']:return 'STABLE_WITHIN_TOLERANCE'
    if f['insufficient_observation']:return 'INSUFFICIENT_OBSERVATION'
    if f['plateau_large_gradient']:return 'PLATEAU_WITH_LARGE_GRADIENT'
    return 'CAP_REACHED_NOT_STABLE'


def base_state(kind,identity,metadata):
    return dict(kind=kind,identity=identity,metadata=metadata,completed_updates=0,history=[],decisions=[],snapshots=[],check_history=[],audit_snapshots=[],
                scheduler=dict(best=None,stagnation=0,stable_count=0,lr_epoch_start=0,lr_changes=[],cooldown_until=0,events=[]),
                status=None,optimization_seconds=0.,fit_seconds=0.,phase='initial_pre_update')


def checkpoint(folder,state,model,optimizer,shot=None):
    start=time.perf_counter()
    state.update(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),rng=rng_state(),
                 model_hash=tree_hash(model.state_dict()),optimizer_hash=tree_hash(optimizer.state_dict()))
    if state['snapshots']:
        state['snapshots'][-1].update(model=state['model'],optimizer=state['optimizer'],rng=state['rng'],scheduler=copy.deepcopy(state['scheduler']))
    if state['snapshots']:
        record=copy.deepcopy(state['snapshots'][-1]);record.pop('optimizer',None);record.pop('rng',None);record.pop('scheduler',None)
        if not state['audit_snapshots'] or state['audit_snapshots'][-1]['step']!=record['step']:state['audit_snapshots'].append(record)
        else:state['audit_snapshots'][-1]=record
    save(folder/'latest.pt',state)
    if shot is not None:save(folder/f'post{shot:03}.pt',state)
    TIMES['checkpoint_seconds']+=time.perf_counter()-start


def resume_or_start(folder,state,model,optimizer):
    if (folder/'latest.pt').exists():
        old=load(folder/'latest.pt');assert old['identity']==state['identity']
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);restore_rng(old['rng'])
        return old
    return state


def fit_mapping(folder,initial,data,lambda_f,ledger,metadata,adaptive=True,stop_after=None,cap_override=None,lambda_w=0.):
    folder=Path(folder);r=RULES['mapping'];cap=(r['cap'] if adaptive else r['prefix']) if cap_override is None else cap_override
    assert cap_override is None or ledger.limits['mapping']==60
    stop_ids=sorted(data['ids'] if metadata['stage'] in ['offline','inner_cv'] else data['all_ids'])
    identity=digest(dict(metadata=metadata,scaler=data['scaler'],labels=tree_hash([data['y'],data['se'],data['seeds']]),
                         initial=tree_hash(initial),lambda_f=lambda_f,lambda_w=lambda_w,adaptive=adaptive,cap=cap,stop_ids=stop_ids))
    model=reload_model(initial);model.train();optimizer=adam(model.parameters(),r['lr'])
    state=base_state('mapping',identity,metadata);state.update(scaler=data['scaler'],train_ids=data['ids'],stop_ids=stop_ids,
        known_labels_hash=tree_hash([data['ids'],data['y'],data['se'],data['seeds']]),lambda_f=lambda_f,lambda_w=lambda_w,epsilon=data['epsilon'],initial_model_hash=tree_hash(initial))
    state=resume_or_start(folder,state,model,optimizer)
    if state['status'] is not None:return state
    start=time.perf_counter()
    try:
        if not state['history']:
            losses,numbers,_,_=components(model,data,lambda_f,lambda_w);s=map_snapshot(model,data,stop_ids,0,losses,numbers)
            state['scheduler']['g0']={k:s['gradient'][k] for k in ['rms','max']}
            state['history']=[dict(numbers,step=0,lr=r['lr'],lr_used_for_update=None,lr_configured_next=r['lr'])];state['snapshots']=[s]
            checkpoint(folder,state,model,optimizer,0)
        for t in range(state['completed_updates']+1,cap+1):
            ts=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            losses,_,_,_=components(model,data,lambda_f,lambda_w);losses['total'].backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),'nonfinite mapping gradient'
            ledger.step('mapping',metadata['task'],t,optimizer)
            assert finite_tree(model.state_dict()),'nonfinite model';state['optimization_seconds']+=time.perf_counter()-ts
            ts=time.perf_counter();losses,numbers,_,_=components(model,data,lambda_f,lambda_w)
            TIMES['loss_forward_seconds']+=time.perf_counter()-ts
            state['history'].append(dict(numbers,step=t,lr=optimizer.param_groups[0]['lr'],lr_used_for_update=optimizer.param_groups[0]['lr'],lr_configured_next=optimizer.param_groups[0]['lr']))
            state.update(completed_updates=t,phase='post_update')
            if t%r['interval']==0 or t==cap or t==stop_after:
                shot=map_snapshot(model,data,stop_ids,t,losses,numbers)
                state['snapshots'].append(shot);state['snapshots']=state['snapshots'][-6:];state['check_history'].append(scalar_shot(shot))
                if t==r['prefix']:
                    state['scheduler']['reference']=dict(norm=max(float(shot['distance'].norm()),1e-8),median=max(float(shot['distance'].median()),1e-8))
                dec=window_decision('mapping',state['history'],state['snapshots'],t,state['scheduler'],lambda_f,lambda_w)
                state['decisions'].append(dec)
                if adaptive and t%r['interval']==0:schedule('mapping',state['scheduler'],dec,numbers['total'],t,optimizer,cap)
                state['history'][-1]['lr_configured_next']=optimizer.param_groups[0]['lr']
                if not adaptive and t==cap:state['status']='FIXED_STEP_STOP'
                elif t==cap or state['scheduler']['stable_count']>=3:state['status']=terminal_status('mapping',state['scheduler'],state['decisions'],shot,data);state['reason_flags']=terminal_flags('mapping',state['scheduler'],state['decisions'],shot,data)
                state['fit_seconds']+=time.perf_counter()-start;start=time.perf_counter()
                checkpoint(folder,state,model,optimizer,t if t==r['prefix'] else None)
                if state['status'] or t==stop_after:break
        state['fit_seconds']+=time.perf_counter()-start
        write(folder/'summary.json',summary(state))
    except (AssertionError,RuntimeError,FloatingPointError) as exc:
        if isinstance(exc,AssertionError) and not any(x in str(exc).lower() for x in ['nonfinite','finite']):raise
        if 'budget' in str(exc) or 'cap:' in str(exc):raise
        state.update(status='NUMERICAL_FAILURE',error=repr(exc));checkpoint(folder,state,model,optimizer);write(folder/'summary.json',summary(state));raise
    return state


def gp_prediction(gp_state,probe_ids):
    ts=time.perf_counter();g=gp_from_state(gp_state);pos={c:i for i,c in enumerate(gp_state['cache']['ids'])}
    x=gp_state['cache']['embeddings'][[pos[c] for c in probe_ids]]
    with torch.no_grad(),gp_numerics():
        p=g(x);mu=p.mean*gp_state['scale']+gp_state['scaler']['y_mean'];std=p.variance.clamp_min(0).sqrt()*gp_state['scale']
    TIMES['gp_prediction_seconds']+=time.perf_counter()-ts
    assert finite_tree([mu,std]);return mu,std


def gp_snapshot(gp,state,loss,t):
    grad=gradient_check(loss,gp);temporary=dict(state,model=copy.deepcopy(gp.state_dict()))
    mu,std=gp_prediction(temporary,state['probe_ids'])
    rank=sorted(range(len(mu)),key=lambda i:(-float(mu[i]),state['probe_ids'][i]))
    lower=float(gp.likelihood.second_noise_covar.raw_noise_constraint.lower_bound)
    pars=dict(lengthscale=float(gp.covar_module.base_kernel.lengthscale),signal_variance=float(gp.covar_module.outputscale),
              mean=float(gp.mean_module.constant),additional_noise=float(gp.likelihood.second_noise),additional_noise_lower=lower,
              noise_distance_to_lower=float(gp.likelihood.second_noise)-lower,
              noise_lower_ratio=float(gp.likelihood.second_noise)/lower,
              lengthscale_distance_to_lower=float(gp.covar_module.base_kernel.lengthscale),
              signal_variance_distance_to_lower=float(gp.covar_module.outputscale),
              raw_parameters={n:p.detach().reshape(-1).tolist() for n,p in gp.named_parameters()})
    return dict(step=t,gradient=grad,prediction=mu,std=std,parameters=pars,
                top5=[state['probe_ids'][i] for i in rank[:5]],boundary_gap=float(mu[rank[4]]-mu[rank[5]]) if len(rank)>5 else None)


def fit_gp(folder,cache,data,ledger,metadata,adaptive=True,stop_after=None,cap_override=None):
    folder=Path(folder);r=RULES['gp'];cap=(r['cap'] if adaptive else r['prefix']) if cap_override is None else cap_override
    assert cap_override is None or ledger.limits['gp']==60
    x=cache['embeddings'][data['indices']].detach().double();y=data['target64'].detach();se=data['se'].detach();scale=data['scaler']['y_scale']
    probe_ids=sorted(data['ids'] if metadata['stage'] in ['offline','inner_cv'] else set(data['all_ids'])-set(data['ids']))
    identity=digest(dict(metadata=metadata,model=cache['model_hash'],inputs=tree_hash([x,y,se]),scaler=data['scaler'],adaptive=adaptive,cap=cap,probe_ids=probe_ids))
    with torch.random.fork_rng(devices=[]):torch.manual_seed(0);gp=ReplayGP(x,y,se,scale).double()
    optimizer=adam(gp.parameters(),r['lr']);state=base_state('gp',identity,metadata)
    state.update(x=x,target=y,se=se,scale=scale,scaler=data['scaler'],train_ids=data['ids'],cache=cache,probe_ids=probe_ids,
        fixed_noise_normalized=gp.likelihood.noise_covar.noise.detach().clone(),mapping_hash_before=cache['model_hash'],scaler_hash=digest(data['scaler']),
        candidate_order_hash=digest(cache['ids']),known_labels_hash=tree_hash([data['ids'],data['y'],data['se'],data['seeds']]),
        mll_normalization='sum log_probability / number_of_training_points',jitter=1e-6,cholesky_max_tries=3,max_cholesky_size=800,
        initial_parameters_hash=tree_hash(gp.state_dict()))
    resumed=(folder/'latest.pt').exists();state=resume_or_start(folder,state,gp,optimizer)
    if state['status'] is not None:return state
    if not resumed:ledger.fit(metadata,identity)
    gp.train();gp.likelihood.train();mll=gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood,gp)
    def loss_now():
        with gp_numerics():v=-mll(gp(x),y)
        assert torch.isfinite(v),'nonfinite NLL';return v
    start=time.perf_counter()
    try:
        if not state['history']:
            loss=loss_now();s=gp_snapshot(gp,state,loss,0)
            state['scheduler']['g0']={k:s['gradient'][k] for k in ['rms','max']}
            state['history']=[dict(total=float(loss.detach()),step=0,lr=r['lr'],lr_used_for_update=None,lr_configured_next=r['lr'])];state['snapshots']=[s]
            checkpoint(folder,state,gp,optimizer,0)
        for t in range(state['completed_updates']+1,cap+1):
            ts=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss=loss_now();loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gp.parameters()),'nonfinite GP gradient'
            ledger.step('gp',metadata['task'],t,optimizer);assert finite_tree(dict(gp.named_parameters())),'nonfinite GP'
            state['optimization_seconds']+=time.perf_counter()-ts
            ts=time.perf_counter();loss=loss_now();TIMES['loss_forward_seconds']+=time.perf_counter()-ts
            state['history'].append(dict(total=float(loss.detach()),step=t,lr=optimizer.param_groups[0]['lr'],lr_used_for_update=optimizer.param_groups[0]['lr'],lr_configured_next=optimizer.param_groups[0]['lr']))
            state.update(completed_updates=t,phase='post_update')
            if t%r['interval']==0 or t==cap or t==stop_after:
                shot=gp_snapshot(gp,state,loss,t);state['snapshots'].append(shot);state['snapshots']=state['snapshots'][-6:];state['check_history'].append(scalar_shot(shot))
                dec=window_decision('gp',state['history'],state['snapshots'],t,state['scheduler']);state['decisions'].append(dec)
                if adaptive and t%r['interval']==0:schedule('gp',state['scheduler'],dec,float(loss.detach()),t,optimizer,cap)
                state['history'][-1]['lr_configured_next']=optimizer.param_groups[0]['lr']
                if not adaptive and t==cap:state['status']='FIXED_STEP_STOP'
                elif t==cap or state['scheduler']['stable_count']>=3:state['status']=terminal_status('gp',state['scheduler'],state['decisions'],shot);state['reason_flags']=terminal_flags('gp',state['scheduler'],state['decisions'],shot)
                state['mapping_hash_after']=cache['model_hash'];assert state['mapping_hash_before']==state['mapping_hash_after'] and state['scaler_hash']==digest(cache['scaler']) and not x.requires_grad
                state['fit_seconds']+=time.perf_counter()-start;start=time.perf_counter()
                checkpoint(folder,state,gp,optimizer,t if t==r['prefix'] else None)
                if state['status'] or t==stop_after:break
        state['fit_seconds']+=time.perf_counter()-start;write(folder/'summary.json',summary(state))
    except (AssertionError,RuntimeError,FloatingPointError) as exc:
        if isinstance(exc,AssertionError) and not any(x in str(exc).lower() for x in ['nonfinite','finite']):raise
        if 'budget' in str(exc) or 'cap:' in str(exc):raise
        state.update(status='NUMERICAL_FAILURE',error=repr(exc));checkpoint(folder,state,gp,optimizer);write(folder/'summary.json',summary(state));raise
    return state


def summary(state):
    return {k:state.get(k) for k in ['identity','kind','metadata','lambda_w','reason_flags','completed_updates','status','error','history','decisions','scheduler',
            'optimization_seconds','fit_seconds','model_hash','optimizer_hash','scaler_hash','mapping_hash_before','mapping_hash_after','known_labels_hash']} | {
            'all_checks':state['check_history'],
            'final_learning_rate':state['optimizer']['param_groups'][0]['lr'],
            'learning_rate_use_counts':{str(lr):sum(h.get('lr_used_for_update')==lr for h in state['history'][1:]) for lr in sorted({h['lr_used_for_update'] for h in state['history'][1:]})},
            'checks':[{k:v for k,v in s.items() if k in ['step','extra','parameters','top5','boundary_gap']} | {'gradient':{k:v for k,v in s['gradient'].items() if k!='vector'}} for s in state['snapshots']]}


def scalar_shot(s):
    return {k:v for k,v in s.items() if k in ['step','extra','parameters','top5','boundary_gap']} | {'gradient':{k:v for k,v in s['gradient'].items() if k!='vector'}}
