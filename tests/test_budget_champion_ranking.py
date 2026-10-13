import copy
import hashlib
import subprocess
import sys
import torch
from surrogate.budget_champion_ranking import list_indices,budget_loss,objective
from surrogate.fuzzy_metric import JointMetricModel
from surrogate.relative_metric import fresh_adam
from surrogate.gp_guidance_replay_engine import scaler_from_initial,input_tensors,score_candidates,reload_model


def loss(s,y,rho):
    return budget_loss(s,torch.tensor(y,dtype=torch.double),['a','b','c','d'],[list(range(4))]*16,rho)


def test_direction():
    s=torch.tensor([0.,3.,2.,1.],requires_grad=True);y=[.9,.8,.7,.6]
    a,_=loss(s,y,.5);assert torch.allclose(a,torch.tensor(4.41))
    b,_=loss(s+torch.tensor([1.,0.,0.,0.]),y,.5);assert b<a
    z,_=loss(s+torch.tensor([2.2,0.,0.,0.]),y,.5);assert z==0
    a.backward();assert s.grad[0]<0 and s.grad[2]>0 and torch.isfinite(s.grad).all()


def test_k_and_insufficient_nonchampions():
    s=torch.tensor([0.,3.,2.,1.],requires_grad=True)
    a,_=loss(s,[.9,.8,.7,.6],.25);assert torch.allclose(a,torch.tensor(9.61))
    b,_=loss(s,[.9,.8,.7,.6],1.);assert b==0
    c,_=loss(s,[.9,.9,.7,.6],.75);assert c==0


def test_all_tied_and_multiple_champions():
    s=torch.tensor([-1.,4.,3.,2.],requires_grad=True)
    a,_=loss(s,[.9]*4,.25);a.backward();assert a==0 and torch.equal(s.grad,torch.zeros_like(s))
    b,_=loss(s,[.9,.9,.8,.7],.25);assert b==0  # The second champion suffices.
    c,d=loss(s,[.9,.9-5e-13,.8,.7],.25);assert c==0 and d['winner_counts']==[2]*16


def test_score_ties_choose_candidate_id():
    s=torch.tensor([0.,2.,2.,1.],requires_grad=True)
    a,d=loss(s,[.9,.8,.7,.6],.25);a.backward()
    assert d['competitor_indices']==[1]*16 and s.grad[1]>0 and s.grad[2]==0


def test_list_identity_and_rng():
    ids=[f'id{i}' for i in range(30)];before=torch.get_rng_state().clone()
    lists=list_indices(ids,'offline','fold0-s0','initial',0)
    for li,js in enumerate(lists):
        expected=sorted(range(len(ids)),key=lambda i:(hashlib.sha256(f'budget_champion_v1:list:offline:fold0-s0:initial:0:{li}:{ids[i]}'.encode()).hexdigest(),ids[i]))[:16]
        assert js==expected and len(set(js))==16
    assert torch.equal(before,torch.get_rng_state())


def fixture():
    g=torch.Generator().manual_seed(4);ids=[f'c{i:03}' for i in range(179)]
    f=dict(ids=ids,architectures=[f'a{i//2}' for i in range(179)],z=torch.randn(179,12,generator=g).tolist(),hp=torch.randn(179,4,generator=g).tolist())
    y={c:dict(training_seeds=[0,1,2],validation_seeds=[.4+i*.002]*3,validation_mean=.4+i*.002,validation_se=0.) for i,c in enumerate(ids)}
    return f,y


def test_network_and_gradient_connections():
    f,y=fixture();known={c:y[c] for c in f['ids'][:20]};sc=scaler_from_initial(f,list(known),known);d=input_tensors(f,sc,known)
    m=JointMetricModel(12,seed=0);layers=[(x.in_features,x.out_features) for x in m.modules() if isinstance(x,torch.nn.Linear)]
    assert layers==[(4,16),(28,32),(32,32),(32,8),(8,1)]
    assert len([x for x in m.modules() if isinstance(x,torch.nn.SiLU)])==3
    e,s=m(d['z'],d['hp']);lists=list_indices(d['ids'],'replay','start2026091600-s0','initial',0)
    losses,_=objective(e,s,d['target32'],d['y'],d['ids'],lists,50/159,.1)
    grads=torch.autograd.grad(losses['ranking'],list(m.parameters()))
    norms={n:g.norm() for (n,_),g in zip(m.named_parameters(),grads)}
    assert norms['regressor.weight']>0 and sum(v for n,v in norms.items() if n.startswith('mapping.'))>0
    assert all(torch.isfinite(g).all() for g in grads) and not d['z'].requires_grad and not d['hp'].requires_grad


def test_zero_weight_matches_C_update_two_synthetic_steps():
    f,y=fixture();known={c:y[c] for c in f['ids'][:20]};d=input_tensors(f,scaler_from_initial(f,list(known),known),known)
    a=JointMetricModel(12,seed=0);b=reload_model(a.state_dict());oa,ob=fresh_adam(a),fresh_adam(b)
    assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(a.parameters(),b.parameters()))
    e,s=a(d['z'],d['hp']);base=(s-d['target32']).square().mean()+.001*e.square().sum(-1).mean();base.backward();oa.step()
    e,s=b(d['z'],d['hp']);ls,_=objective(e,s,d['target32'],d['y'],d['ids'],list_indices(d['ids'],'offline','fold0-s0','initial',0),.3,0.)
    ls['total'].backward();ob.step()
    assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))


def test_hidden_label_perturbation_offline_and_replay():
    f,vault=fixture()
    for stage,n in [('offline',141),('replay',20)]:
        known_ids=f['ids'][:n];altered=copy.deepcopy(vault);hidden=f['ids'][n]
        altered[hidden].update(validation_mean=.1,validation_seeds=[.1]*3)
        def inputs(source):
            labels={c:source[c] for c in known_ids};sc=scaler_from_initial(f,known_ids,labels)
            return input_tensors(f,sc,labels),sc
        a,sc=inputs(vault);b,sc2=inputs(altered);assert sc==sc2
        model=JointMetricModel(12,seed=0)
        def output(d):
            e,s=model(d['z'],d['hp']);lists=list_indices(d['ids'],stage,'paired-s0','initial',0)
            losses,_=objective(e,s,d['target32'],d['y'],d['ids'],lists,.3,.1)
            g=torch.cat([x.flatten() for x in torch.autograd.grad(losses['total'],model.parameters())])
            rows,_,_=score_candidates(model,d)
            return lists,g,rows
        la,ga,ra=output(a);lb,gb,rb=output(b)
        assert la==lb and torch.equal(ga,gb) and ra==rb
        da=input_tensors(f,sc,{c:vault[c] for c in known_ids+[hidden]})
        db=input_tensors(f,sc,{c:altered[c] for c in known_ids+[hidden]})
        assert not torch.equal(output(da)[1],output(db)[1])


def test_worker_guard():
    code='''
from analyse.budget_champion_run import ROOT,DEFAULT,guard
task=dict(label_file='train_labels/offline-fold0.json',weight_file='initial_weights/offline-fold0-s0.pt')
guard(DEFAULT,DEFAULT/'offline/fold0-s0-C_HEAD',task)
for p in [ROOT/'results/posthoc/fuzzy-metric-stage2-20260910-v1/A_validation.json',ROOT/'results/posthoc/gp-guidance-development-replay-20260916-v1/shared/2026091600/M100.pt',DEFAULT/'offline/fold0-s0-BR_HEAD/events/initial/latest.pt']:
    try:p.read_bytes()
    except PermissionError:continue
    raise AssertionError(str(p))
'''
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
