"""Zero optimizer updates: full synchronized gradient and protocol checks."""
import os,copy,math,tempfile,subprocess,sys,json
from pathlib import Path
import pytest,numpy as np,torch
from analyse import gp_mean_guidance_sync_fivefold as f
from surrogate import gp_mean_guidance_sync as s
from surrogate import gp_mean_guidance as old

def dense(x,z,p):
    r=np.sqrt(np.sum((x[:,None]-z[None])**2,axis=-1))/p['lengthscale'];r=np.sqrt(5)*r
    return p['outputscale']*(1+r+r*r/3)*np.exp(-r)

@pytest.fixture(autouse=True)
def no_updates(monkeypatch):
    def blocked(*a,**k):raise AssertionError('zero-update preflight cannot call optimizer.step')
    monkeypatch.setattr(torch.optim.Adam,'step',blocked);torch.set_num_threads(1)

def fixture():
    rng=np.random.default_rng(321);e=torch.tensor(rng.normal(size=(9,4)),dtype=torch.double,requires_grad=True)
    y=torch.tensor(rng.normal(size=9),dtype=torch.double);se=torch.linspace(.001,.003,9,dtype=torch.double);scale=.15
    groups=old.architecture_groups([str(i) for i in range(9)],[str(i//2) for i in range(9)])
    pars=dict(mean=.17,lengthscale=.83,outputscale=1.2,noise=(1e-4/scale)**2)
    teachers=[dict(parameters=copy.deepcopy(pars)) for g in groups]
    return e,y,se,scale,groups,teachers

def test_numpy_forward_and_full_finite_difference():
    e,y,se,scale,groups,teachers=fixture();loss=s.guidance(e,y,se,scale,groups,teachers);grad,=torch.autograd.grad(loss,e)
    a=e.detach().numpy();yy=y.numpy();sn=se.numpy();total=0;max_mu=0
    for group,t in zip(groups,teachers):
        si,qi=group['support_indices'],group['query_indices'];p=t['parameters'];x=a[si];q=a[qi]
        K=dense(x,x,p)+np.diag((sn[si]/scale)**2+p['noise']);L=np.linalg.cholesky(K);alpha=np.linalg.solve(L.T,np.linalg.solve(L,yy[si]-p['mean']))
        mu=p['mean']+dense(q,x,p)@alpha
        v=s.synchronized_mean(e[si],e[qi],y[si],se[si],scale,p)
        max_mu=max(max_mu,float(np.max(abs(mu-v.detach().numpy()))));total+=np.sum((mu-yy[qi])**2)
    assert max_mu<1e-9 and abs(total/len(e)-float(loss.detach()))<1e-9
    eps=1e-6;finite=torch.empty_like(e)
    for i in range(e.shape[0]):
        for j in range(e.shape[1]):
            a=e.detach().clone();b=a.clone();a[i,j]+=eps;b[i,j]-=eps
            finite[i,j]=(s.guidance(a,y,se,scale,groups,teachers)-s.guidance(b,y,se,scale,groups,teachers))/(2*eps)
    assert torch.allclose(grad,finite,atol=2e-6,rtol=2e-5)
    out=Path(os.environ['GP_SYNC_OUTPUT']);f.write(out/'numeric_preflight.json',dict(dense_mean_maxdiff=max_mu,full_gradient_finite_difference_maxdiff=float((grad-finite).abs().max()),atol=2e-6,rtol=2e-5,epsilon=eps,optimizer_updates=0))

def test_both_paths_translation_rotation_directional_derivative():
    e,y,se,scale,groups,teachers=fixture();value=s.guidance(e,y,se,scale,groups,teachers);full,=torch.autograd.grad(value,e)
    q=e.detach().clone().requires_grad_(True);support=e.detach().clone().requires_grad_(True)
    value2=s.guidance(e,y,se,scale,groups,teachers,query_embedding=q,support_embedding=support);gq,gs=torch.autograd.grad(value2,[q,support])
    assert gq.norm()>0 and gs.norm()>0 and torch.allclose(full,gq+gs,atol=1e-10,rtol=1e-9)
    shift=torch.tensor([.3,-.7,1.2,.4],dtype=torch.double);rot,_=torch.linalg.qr(torch.tensor(np.random.default_rng(142).normal(size=(4,4)),dtype=torch.double))
    shifted=s.guidance(e+shift,y,se,scale,groups,teachers);rotated=s.guidance(e@rot,y,se,scale,groups,teachers)
    assert abs(float(shifted.detach()-value.detach()))<1e-10 and abs(float(rotated.detach()-value.detach()))<1e-10
    assert full.sum(0).norm()<1e-9
    eps=1e-6;fd=(s.guidance(e.detach()+eps*shift,y,se,scale,groups,teachers)-s.guidance(e.detach()-eps*shift,y,se,scale,groups,teachers))/(2*eps)
    direction=(full*shift).sum();assert abs(float(fd-direction))<2e-6
    skew=torch.tensor([[0,.1,.2,.3],[-.1,0,.4,.5],[-.2,-.4,0,.6],[-.3,-.5,-.6,0]],dtype=torch.double)
    rotation_direction=e.detach()@skew;assert abs(float((full*rotation_direction).sum()))<1e-9
    f.write(Path(os.environ['GP_SYNC_OUTPUT'])/'invariance_preflight.json',dict(query_norm=float(gq.norm()),support_norm=float(gs.norm()),sum_gradient_norm=float(full.sum(0).norm()),decomposition_maxdiff=float((full-gq-gs).abs().max()),translation_directional_derivative=float(direction),translation_finite_difference=float(fd),rotation_directional_derivative=float((full*rotation_direction).sum()),updates=0))

def test_zero_duplicate_near_coordinates_with_original_noise():
    e,y,se,scale,groups,teachers=fixture()
    for delta in [0.,1e-14,1e-8]:
        a=e.detach().clone();a[1]=a[0]+delta;a.requires_grad_(True)
        val=s.guidance(a,y,se,scale,groups,teachers);g,=torch.autograd.grad(val,a)
        assert torch.isfinite(val) and torch.isfinite(g).all()
    a=torch.zeros_like(e,requires_grad=True);val=s.guidance(a,y,se,scale,groups,teachers);g,=torch.autograd.grad(val,a)
    assert torch.isfinite(val) and torch.isfinite(g).all() and torch.equal(g,torch.zeros_like(g))

def test_all_original_starts_and_old_G_same_forward_and_C_gradients():
    o=Path(os.environ['GP_SYNC_OUTPUT']);rows=[]
    for task in f.read(o/'task_matrix.json'):
        d=f.load(o/task['input']);warm=f.load(o/task['warmstart']);m=f.JointMetricModel(12,seed=task['seed']);m.load_state_dict(warm['model']);m.requires_grad_(True)
        a,b=f.make_branches(m);oa,ob=f.fresh_adam(a),f.fresh_adam(b);assert not oa.state and not ob.state
        assert all(x.data_ptr()!=z.data_ptr() and torch.equal(x,z) for x,z in zip(a.parameters(),b.parameters()))
        folder=f.PREVIOUS/'trajectories'/f"fold{task['fold']}-s{task['seed']}-G"
        teachers=[f.load(folder/'gps'/f'b0-g{g}'/'teacher.pt') for g in range(3)]
        e,p=m(d['train']['z'],d['train']['hp']);sy=s.guidance(e,d['target64'],d['se'],d['scaler']['y_scale'],d['groups'],teachers)
        oldval=old.guidance(e,d['target64'],d['groups'],teachers);difference=float(abs(sy.detach()-oldval.detach()));assert difference<1e-9
        probe=s.gradient_probe(m,d['train'],d['target32'],d['target64'],d['se'],d['scaler']['y_scale'],d['groups'],teachers)
        new=s.objective(e,p,d['target32'],d['target64'],d['se'],d['scaler']['y_scale'],enabled=False)
        original=old.objective(e,p,d['target32'],d['target64'],enabled=False)
        assert all(torch.equal(new[k],original[k]) for k in new)
        ng=torch.autograd.grad(new['total'],list(m.parameters()),retain_graph=True);og=torch.autograd.grad(original['total'],list(m.parameters()))
        assert all(torch.equal(x,y) for x,y in zip(ng,og))
        assert not d['train']['z'].requires_grad and not d['target64'].requires_grad and all(not t['support'].requires_grad for t in teachers)
        rows.append(dict(task=task['task_id'],M100_sha=f.tensor_hash(warm['model']),old_G_forward_maxdiff=difference,C_loss_and_gradients_exact=True,independent_parameters_Adam=True,**probe))
    f.write(o/'real_preflight.json',dict(rows=rows,updates=0))

def test_candidate_weighting_not_equal_group_means():
    e,y,se,scale,groups,teachers=fixture();losses=[]
    for group,t in zip(groups,teachers):
        si,qi=group['support_indices'],group['query_indices'];v=s.synchronized_mean(e[si],e[qi],y[si],se[si],scale,t['parameters']);losses.append((v-y[qi]).square())
    assert len({len(v) for v in losses})>1
    assert torch.equal(s.guidance(e,y,se,scale,groups,teachers),torch.cat(losses).mean())
    assert abs(float(torch.cat(losses).mean()-torch.stack([v.mean() for v in losses]).mean()))>1e-6

def test_finalization_after_timing_before_completion_is_idempotent(tmp_path):
    timing=dict(mapping_update_seconds=1.,mapping_backward_seconds=.5,gp_update_seconds=2.,solve_records={'formal_mapping_forward':dict(conditionings=600)},wall_seconds_this_invocation=3.)
    a=f.finalize_timing(tmp_path,timing);b=f.finalize_timing(tmp_path,{**timing,'wall_seconds_this_invocation':99.})
    assert a==b and f.read(tmp_path/'timing.json')==timing

def test_fold_guard_rejects_other_labels():
    o=Path(os.environ['GP_SYNC_OUTPUT'])
    script='''from pathlib import Path
from analyse import gp_mean_guidance_sync_fivefold as f
o=Path(%r);tasks=[t for t in f.read(o/'task_matrix.json') if t['fold']==0]
f.guard_files(o,tasks)
for p in [o/'evaluation_labels/fold0.json',o/'inputs/fold1/train_labels.json',f.PREVIOUS/'inputs/fold0/train_labels.json',f.OLD/'A_validation.json']:
 try: p.read_bytes()
 except PermissionError: continue
 raise AssertionError(str(p))
print('blocked four forbidden label paths')
'''%str(o)
    r=subprocess.run([sys.executable,'-c',script],cwd=f.ROOT,capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
    assert 'blocked four' in r.stdout
