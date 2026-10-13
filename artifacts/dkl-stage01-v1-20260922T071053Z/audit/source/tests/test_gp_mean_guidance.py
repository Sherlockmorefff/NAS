"""Zero-update tests of GP mean, input gradients, identities and runtime boundary."""
from pathlib import Path
import copy,os,subprocess,sys,math
import numpy as np
import torch,gpytorch,pytest
from analyse import gp_mean_guidance_fivefold as f
from surrogate import gp_mean_guidance as g
from surrogate.relative_metric import objective as old_objective
O=Path(os.environ.get('GP_GUIDANCE_OUTPUT',f.DEFAULT))
@pytest.fixture(autouse=True)
def no_updates(monkeypatch):
    def reject(*a,**k):raise AssertionError('preflight optimizer.step forbidden')
    monkeypatch.setattr(torch.optim.Adam,'step',reject);torch.set_num_threads(1)

def numpy_kernel(x,z,p):
    r=np.sqrt(((x[:,None,:]-z[None,:,:])**2).sum(2))/p['lengthscale'];q=np.sqrt(5.)*r
    return p['outputscale']*(1+q+q*q/3)*np.exp(-q)

def test_dense_mean_variance_and_finite_difference():
    x=torch.tensor([[0.,.2],[.8,-.1],[-.3,1.1],[1.3,.7]],dtype=torch.double)
    y=torch.tensor([-.7,.9,-.1,.4],dtype=torch.double);se=torch.tensor([.01,.02,.015,.03],dtype=torch.double);scale=.2
    gp,opt=f.make_gp(x,y,se,scale);assert not opt.state
    teacher=g.from_gp(gp,x,y,se,scale);pars=teacher['parameters'];assert teacher['jitter']==0
    q=torch.tensor([[.1,.7],[.65,.4]],dtype=torch.double,requires_grad=True)
    K=numpy_kernel(x.numpy(),x.numpy(),pars)+np.diag((se.numpy()/scale)**2+pars['noise']);L=np.linalg.cholesky(K);alpha=np.linalg.solve(L.T,np.linalg.solve(L,y.numpy()-pars['mean']))
    def independent(v):return pars['mean']+numpy_kernel(v,x.numpy(),pars)@alpha
    pred,var,_=g.posterior(teacher,q);cross=numpy_kernel(q.detach().numpy(),x.numpy(),pars)
    np.testing.assert_allclose(pred.detach().numpy(),independent(q.detach().numpy()),rtol=0,atol=1e-9)
    np.testing.assert_allclose(var.detach().numpy(),pars['outputscale']-np.linalg.solve(L,cross.T).__pow__(2).sum(0),rtol=0,atol=1e-9)
    with torch.no_grad(),gpytorch.settings.cholesky_jitter(double_value=1e-6):library=gp(q.detach())
    torch.testing.assert_close(pred.detach(),library.mean,rtol=0,atol=1e-9);torch.testing.assert_close(var.detach(),library.variance,rtol=0,atol=1e-8)
    target=np.array([.3,-.4]);loss=(pred-torch.tensor(target)).square().mean();grad,=torch.autograd.grad(loss,q)
    finite=np.zeros((2,2));eps=1e-6
    for i in range(2):
      for j in range(2):
        a=q.detach().numpy().copy();b=a.copy();a[i,j]+=eps;b[i,j]-=eps
        finite[i,j]=(((independent(a)-target)**2).mean()-((independent(b)-target)**2).mean())/(2*eps)
    np.testing.assert_allclose(grad.numpy(),finite,rtol=2e-5,atol=2e-6)
    assert not any(p.requires_grad for p in gp.parameters()) and teacher['support'].grad is None and not teacher['alpha'].requires_grad
    f.write(O/'numeric_preflight.json',dict(mean_maxdiff=float(np.max(abs(pred.detach().numpy()-independent(q.detach().numpy())))),finite_difference_maxdiff=float(np.max(abs(grad.numpy()-finite))),tolerances=f.CONFIG['tolerances'],actual_jitter=teacher['jitter'],updates=0))

def test_fifteen_startpoints_C_gradient_and_groups():
    p,tasks=f.verify_protocol(O);checks=f.read(O/'warmstart_checks.json');assert len(checks)==15 and len(tasks)==30 and not p['missing']
    for t in tasks[::2]:
        d=f.load(O/t['input']);w=f.load(O/t['warmstart']);m=f.JointMetricModel(12,seed=t['seed']);m.load_state_dict(w['model']);a,b=f.make_branches(m)
        assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters()))
        opt=f.fresh_adam(a);assert not opt.state and all(x.requires_grad for x in a.parameters())
        groups=g.architecture_groups(d['train']['ids'],d['train']['architectures']);assert groups==d['groups']
        assert sorted(i for group in groups for i in group['query_indices'])==list(range(len(d['target32'])))
        for group in groups:assert not set(group['query_architectures'])&set(group['support_architectures'])
        e,pred=a(d['train']['z'],d['train']['hp']);old=old_objective(e,pred,d['target32'],None,0.)['total'];new=g.objective(e,pred,d['target32'],d['target64'])['total']
        assert torch.equal(old,new)
        go=torch.autograd.grad(old,list(a.parameters()),retain_graph=True);gn=torch.autograd.grad(new,list(a.parameters()));assert all(torch.equal(x,y) for x,y in zip(go,gn))
        assert w['scaler']['fit_ids']==t['train_ids'] and d['target64'].dtype==torch.float64

def test_real_guidance_only_reaches_live_mapping_HP():
    task=f.read(O/'task_matrix.json')[0];d=f.load(O/task['input']);m=f.JointMetricModel(12);m.load_state_dict(f.load(O/task['warmstart'])['model']);m.requires_grad_(True)
    z=d['train']['z'].clone().requires_grad_(True);train={**d['train'],'z':z}
    with torch.no_grad():snapshot,_=m(train['z'],train['hp'])
    teachers=[]
    for group in d['groups']:
        js=group['support_indices'];gp,opt=f.make_gp(snapshot[js].double(),d['target64'][js],d['se'][js],d['scaler']['y_scale'])
        teachers.append(g.from_gp(gp,snapshot[js].double(),d['target64'][js],d['se'][js],d['scaler']['y_scale']))
    before=f.tensor_hash(m.state_dict());p=g.gradient_probe(m,train,d['target32'],d['target64'],d['groups'],teachers)
    assert p['guidance_shared_norm']>0 and p['guidance_query_gradient_norm']>0 and p['frozen_support_gradients_none']
    assert all(v is not None and math.isfinite(v) for n,v in p['parameter_gradient_norms'].items() if not n.startswith('regressor.'))
    e,pred=m(z,train['hp']);guide=g.guidance(e,d['target64'],d['groups'],teachers);zg,=torch.autograd.grad(guide,[z],allow_unused=True);assert zg is None
    assert f.tensor_hash(m.state_dict())==before
    # Explicitly check candidate-weighted reduction despite unequal groups.
    point=[]
    for group,t in zip(d['groups'],teachers):
        js=group['query_indices'];point.extend(((g.mean(t,e[js])-d['target64'][js])**2).detach().tolist())
    assert abs(float(guide.detach())-sum(point)/len(point))<1e-12
    f.write(O/'guidance_preflight.json',dict(**p,mapping_weights_unchanged=True,VAE_input_gradient_is_None=True,updates=0,teachers='initialized but not optimized; mechanism check only'))

def test_size_and_regression_reductions():
    e=torch.ones(3,8,requires_grad=True);pred=torch.tensor([1.,-1.,0.],requires_grad=True);target=torch.zeros(3)
    v=g.objective(e,pred,target,target.double());assert float(v['size'])==8 and float(v['regression'])==pytest.approx(2/3)

def test_boundary_and_config_rejection(tmp_path):
    script='''from analyse import gp_mean_guidance_fivefold as f
p,t=f.verify_protocol(f.DEFAULT,0)
f.guard_files(f.DEFAULT,t)
f.load(f.DEFAULT/t[0]['input'])
for name in ['evaluation_labels/fold0.json','inputs/fold1/s0.pt','input_originals/fuzzy-metric-stage2-20260910-v1_A_validation.json']:
    try:(f.DEFAULT/name).open('rb')
    except PermissionError:continue
    raise AssertionError('forbidden labels accessible')
'''
    r=subprocess.run([sys.executable,'-c',script],cwd=f.ROOT,capture_output=True,text=True);assert r.returncode==0,r.stderr
    bad=copy.deepcopy(f.CONFIG);bad['lambda_guidance']=1.;f.write(tmp_path/'bad.json',bad)
    r=subprocess.run([sys.executable,str(Path(f.__file__)),'--config',str(tmp_path/'bad.json'),'--run'],capture_output=True,text=True)
    assert r.returncode!=0 and 'configuration conflicts' in r.stderr

def test_two_ledgers_recovery_metadata_no_steps(tmp_path):
    for kind in ['mapping','gp']:
      for n in [1,2,3]:f.write(tmp_path/'intents'/kind/f'{n:06d}.json',dict(global_id=n))
      for n in [1,2]:f.write(tmp_path/'receipts'/kind/f'{n:06d}.json',dict(global_id=n))
      path=tmp_path/'trajectories/task'/('mapping' if kind=='mapping' else 'gps/test')/'latest.pt'
      f.save(path,dict(step=1,history=[{}],committed=[1]))
    a=f.costs(tmp_path)
    assert all(v['effective']==1 and v['actual_confirmed']==2 and v['discarded']==1 and v['uncertain']==1 and v['charged']==3 for v in a.values())
    assert a==f.costs(tmp_path)

def test_jitter_only_on_failure_and_rng_isolated():
    L,jitter,attempts=g.solve_cholesky(torch.eye(4,dtype=torch.double));assert jitter==0 and len(attempts)==1
    L,jitter,attempts=g.solve_cholesky(torch.ones(4,4,dtype=torch.double));assert jitter==1e-6 and len(attempts)==2
    before=torch.get_rng_state().clone();gp,opt=f.make_gp(torch.eye(3,dtype=torch.double),torch.ones(3,dtype=torch.double),torch.ones(3,dtype=torch.double)*.01,.2)
    assert torch.equal(before,torch.get_rng_state()) and not opt.state
