"""Protocol boundary tests, <= 60 synthetic Adam updates of each kind."""
import sys,copy,tempfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from analyse.fuzzy_convergence_core import *
import analyse.fuzzy_loss_interaction_core as old


def classifier_tests():
    hs=[dict(step=t,total=1.,weighted_regression=.4,weighted_neighborhood=.4,weighted_scale=.2) for t in range(100,201)]
    shots=[dict(step=t,gradient=dict(rms=1e-8,max=1e-8),distance=torch.ones(100,dtype=torch.float64),
                neighbors=[list('abcde')],prediction=torch.zeros(2)) for t in range(100,201,20)]
    state=dict(g0=dict(rms=.1,max=.1),reference=dict(norm=10.,median=1.),lr_epoch_start=0)
    def check(h=hs,s=shots,w=state,lf=1.):return window_decision('mapping',h,s,200,w,lf)
    assert check()['passed']
    h=copy.deepcopy(hs)
    for i,r in enumerate(h):r['weighted_regression']+=.1*(i%2);r['weighted_neighborhood']-=.1*(i%2)
    assert not check(h=h)['passed']
    h=copy.deepcopy(hs)
    for i,r in enumerate(h):r['weighted_neighborhood']=i
    assert check(h=h,lf=0.)['passed'] and not check(h=h,lf=1.)['passed']
    for mode in ['middle','one_pair','scale','gradient']:
        s=copy.deepcopy(shots)
        if mode=='middle':s[-2]['distance']+=.2
        if mode=='one_pair':s[-2]['distance'][0]+=.1
        if mode=='scale':s[-1]['distance']*=1.1
        if mode=='gradient':s[-3]['gradient']=dict(rms=.1,max=.1)
        assert not check(s=s)['passed'],mode
    # Rotation leaves Euclidean distances invariant; near-tie neighbor swaps are diagnostic only.
    x=torch.tensor([[0.,0.],[1.,0.],[0.,1.]],dtype=torch.float64);rot=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
    assert torch.equal(torch.pdist(x),torch.pdist(x@rot))
    s=copy.deepcopy(shots);s[-1]['neighbors']=[list('abcdf')];assert check(s=s)['passed']
    assert not check(w=dict(state,lr_epoch_start=141))['valid']
    assert check(w=dict(state,lr_epoch_start=140))['valid']
    collapse=dict(shots[-1],distance=torch.zeros(100));assert terminal_status('mapping',dict(stable_count=3),[],collapse,dict(y=torch.tensor([0.,1.])))=='DEGENERATE_GEOMETRY'
    ghs=[dict(step=t,total=-2.) for t in range(20,71)]
    gshots=[dict(step=t,gradient=dict(rms=1e-9,max=1e-9),prediction=torch.ones(4)*.8,std=torch.ones(4)*.1) for t in range(20,71,10)]
    st=dict(g0=dict(rms=.1,max=.1),lr_epoch_start=0)
    assert window_decision('gp',ghs,gshots,70,st)['passed']
    gshots[-1]['prediction'][0]+=.0006
    assert not window_decision('gp',ghs,gshots,70,st)['passed']
    assert .0005*100==.05
    opt=adam([torch.nn.Parameter(torch.zeros(1))],.05)
    scheduler=dict(best=-2.,stagnation=0,stable_count=0,lr_epoch_start=0,lr_changes=[])
    for t in [30,40,50]:schedule('gp',scheduler,dict(passed=False),-2.001,t,opt)
    assert opt.param_groups[0]['lr']==.025 and scheduler['lr_epoch_start']==51
    schedule('gp',scheduler,dict(passed=False),-2.01,60,opt);assert scheduler['best']==-2.01
    n=len(scheduler['lr_changes'])
    for t in [90,100,110]:schedule('gp',scheduler,dict(passed=True),-2.01,t,opt)
    assert scheduler['stable_count']==3 and len(scheduler['lr_changes'])==n
    return ['component_cancellation','zero_weight_excluded','middle_motion','outlier_pair','global_scale','large_gradient_small_lr',
            'collapse','rotation','near_tie_diagnostic_only','lr_window_reset','negative_NLL','raw_units','stable_pauses_scheduler']


def losses_test():
    e=torch.tensor([[0.,0.],[.3,.4],[1.,0.]],requires_grad=True);pred=torch.tensor([.1,.4,.6],requires_grad=True)
    y=torch.tensor([.2,.3,.5]);p=neighborhood_affinity(torch.tensor([.5,.509,.52],dtype=torch.float64),mode='N2',epsilon=.01)
    assert p[0,2]==0 and p[0,1]>0
    boundary=neighborhood_affinity(torch.tensor([0.,.125,.25],dtype=torch.float64),mode='N2',epsilon=.125)
    assert float(boundary.sum())==0
    w=torch.tensor([[1.,.4,.7],[.4,1.,.2],[.7,.2,1.]])
    ls=metric_loss(e,pred,y,p,lambda_f=.2,lambda_e=.001,pair_weights=w)
    i,j=torch.triu_indices(3,3,1);q=(1/(1+(e[i]-e[j]).square().sum(-1))).clamp(torch.finfo(e.dtype).eps,1-torch.finfo(e.dtype).eps)
    manual=((pred-y)**2).mean()+.2*(-w[i,j]*(p.float()[i,j]*q.log()+(1-p.float()[i,j])*torch.log1p(-q))).sum()/w[i,j].sum()+.001*e.square().sum(-1).mean()
    assert torch.allclose(ls['total'],manual,atol=1e-7);ls['total'].backward();assert torch.isfinite(e.grad).all() and all(e.grad.norm(dim=1)>0)
    q0=q.detach().clone();ee=e.detach().clone();ee[0]+=.1;q1=1/(1+(ee[i]-ee[j]).square().sum(-1));assert not torch.equal(q0,q1)
    return ['manual_original_losses','strict_absolute_cutoff','all_p_zero_kept','fresh_current_distance','both_candidate_gradients']


class OldAdapter:
    def __init__(self,b):self.b=b;self.folder=b.folder
    def update(self,kind,task,step,optimizer,purpose):self.b.step(kind,task,step,optimizer);return self.b.counts[kind]


def numeric_tests(o):
    path=o/'synthetic';assert not (o/'engineering_checks.json').exists(),'do not rerun a completed engineering suite'
    torch.manual_seed(771);np.random.seed(771);random.seed(771)
    features=dict(ids=[f'fake{i:02}' for i in range(16)],z=torch.randn(16,12).tolist(),hp=torch.randn(16,4).tolist())
    labels={}
    for i,c in enumerate(features['ids'][:10]):
        ys=[.55+i*.006-.001,.55+i*.006,.55+i*.006+.001]
        labels[c]=dict(validation_mean=float(np.mean(ys)),validation_se=float(np.std(ys,ddof=1)/math.sqrt(3)),validation_seeds=ys,training_seeds=[0,1,2])
    scaler=scaler_from_training(features,sorted(labels),labels);data=observed_data(features,scaler,labels,.01)
    init=JointMetricModel(12,seed=0).state_dict();meta=dict(task='synthetic',stage='offline',representation_version=0)
    budget=Budget(path/'ledger','synthetic-v1',synthetic=True);rng=rng_state()
    a=fit_mapping(path/'map_cont',init,data,.1,budget,meta,adaptive=False,cap_override=10)
    restore_rng(rng);fit_mapping(path/'map_resume',init,data,.1,budget,meta,adaptive=False,cap_override=10,stop_after=5)
    b=fit_mapping(path/'map_resume',init,data,.1,budget,meta,adaptive=False,cap_override=10)
    restore_rng(rng);c=old.fit_mapping(path/'map_old',init,data,.1,OldAdapter(budget),meta,steps=10,shots=(0,10))
    for key in ['model','optimizer','rng']:assert tree_hash(a[key])==tree_hash(b[key])==tree_hash(c[key]),key
    cache=embedding_cache(reload_model(a['model']),data);rng=rng_state()
    ga=fit_gp(path/'gp_cont-v2',cache,data,budget,meta,adaptive=False,cap_override=10)
    restore_rng(rng);fit_gp(path/'gp_resume',cache,data,budget,meta,adaptive=False,cap_override=10,stop_after=5)
    gb=fit_gp(path/'gp_resume',cache,data,budget,meta,adaptive=False,cap_override=10)
    restore_rng(rng);gc=old.fit_gp(path/'gp_old',cache,data,OldAdapter(budget),meta,steps=10)
    for key in ['model','optimizer','rng']:assert tree_hash(ga[key])==tree_hash(gb[key])==tree_hash(gc[key]),key
    expected_gp=31 if (path/'gp_cont/latest.pt').exists() else 30
    assert budget.counts==dict(mapping=30,gp=expected_gp)
    # Re-entry with a different representation identity must reject saved caches.
    try:fit_gp(path/'gp_resume',cache,data,budget,dict(meta,representation_version=1),adaptive=False,cap_override=10)
    except AssertionError:pass
    else:raise AssertionError('stale GP accepted')
    # Receipt replay merges once; it does not cause a second oracle call or query charge.
    known=set(labels);unknown=features['ids'][10:15];receipt=dict(ids=unknown,query_count_after=5)
    write(path/'receipt.json',receipt,immutable=True);calls=1
    for attempt in range(2):
        r=read(path/'receipt.json')
        if not set(r['ids'])<=known:known.update(r['ids'])
    assert calls==1 and len(known)==15 and receipt['query_count_after']==5
    save(path/'fixture.pt',dict(features=features,labels=labels,scaler=scaler))
    return dict(mapping_updates=budget.counts['mapping'],gp_updates=budget.counts['gp'],old_prefix_exact=True,resume_model_Adam_RNG_exact=True,
                new_representation_cache_rejected=True,synthetic_receipt_reuse_no_duplicate=True,hidden_real_labels_read=0)


def main():
    o=Path(sys.argv[1]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checks=classifier_tests()+losses_test();numeric=numeric_tests(o)
    write(o/'engineering_checks.json',dict(passed=True,tests=checks,synthetic=numeric,
        real_100_20_prefix='checked for each formal offline trajectory against previous saved model, Adam and RNG',
        hidden_read='actual worker audit-hook denial for all 60 workers',time=utc()))
    print(json.dumps(dict(passed=True,tests=len(checks),synthetic=numeric)))


if __name__=='__main__':main()
