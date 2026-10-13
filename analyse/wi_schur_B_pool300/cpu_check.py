"""Required CPU-only engineering checks; artificial labels never enter formal B."""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from .common import *

def run():
    import torch
    import gpytorch
    import bo_phase4 as original
    from analyse.structured_kernel_phase1.features import build_operator_graph,directed_wl
    from analyse.structured_kernel_followup.validation_train import training_view
    from types import SimpleNamespace
    from . import gp
    from .search import decision,training_projection
    started=time.perf_counter();gp.cpu_setup();rng=np.random.default_rng(14)
    rows=[{'candidate_id':f'{i:03d}','z_search':rng.normal(size=16).astype(np.float32).tolist()} for i in range(24)]
    a=rbf_schur(rows,5);b=rbf_schur(rows,5);assert a==b
    x=np.asarray([r['z_search'] for r in rows]);xn=(x-x.min(0))/np.maximum(x.max(0)-x.min(0),1e-8)
    kernel=original._rbf_kernel_np(xn,lengthscale=4);selected=[]
    for s in a['selection']:
        i=int(s['candidate_id'])
        v=1. if not selected else 1-kernel[i,selected]@np.linalg.solve(kernel[np.ix_(selected,selected)]+np.eye(len(selected))*1e-6,kernel[selected,i])
        assert np.isclose(v,s['conditional_variance'],atol=1e-12);selected.append(i)
    old=original.schur_greedy_select([torch.tensor(r['z_search'],dtype=torch.float32) for r in rows],5)
    assert [int(np.argmin(np.linalg.norm(x-v.numpy(),axis=1))) for v in old]==selected
    xx=torch.tensor(rng.normal(size=(8,10)),dtype=torch.float64);yy=torch.tensor(np.linspace(-1,1,8))
    model=gp.StaticExactGP(xx,yy,'W-I',6)
    assert isinstance(model.likelihood,gpytorch.likelihoods.GaussianLikelihood)
    assert not isinstance(model.likelihood,gpytorch.likelihoods.FixedNoiseGaussianLikelihood)
    k=model.covar_module(xx).to_dense().detach().numpy()
    from scipy.spatial.distance import cdist
    def matern(distance):
        t=np.sqrt(5)*distance;return (1+t+t*t/3)*np.exp(-t)
    ka=matern(cdist(xx[:,:6].numpy(),xx[:,:6].numpy()));kh=matern(cdist(xx[:,6:].numpy(),xx[:,6:].numpy()))
    assert np.allclose(k,.25*(ka+kh)+.5*ka*kh,atol=1e-12)
    assert sum(p.numel() for p in model.parameters())==5
    from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]),seed=seed(0,'mc',250))
    with torch.no_grad():
        score250=qLogExpectedImprovement(FrozenAdapter(model),best_f=0.,sampler=sampler)(xx[:2].unsqueeze(-2))
    assert score250.shape==(2,) and bool(torch.isfinite(score250).all())
    view=training_view(SimpleNamespace(x=torch.zeros(5,3),edge_index=torch.tensor([[0],[1]]),y=torch.arange(5),
        train_mask=torch.tensor([1,0,0,0,0],dtype=torch.bool),val_mask=torch.tensor([0,1,0,0,0],dtype=torch.bool),test_mask=torch.tensor([0,0,1,1,1],dtype=torch.bool)))
    assert not hasattr(view,'test_mask') and view.y.tolist()==[0,1,-100,-100,-100]
    allseeds=[seed(g,stage,i) for g in range(5) for stage,n in [('search',1),('validation',30),('final',30)] for i in range(n)]
    assert len(allseeds)==len(set(allseeds));assert seed(0,'mc',250)!=seed(0,'mc',1)
    with tempfile.TemporaryDirectory(prefix='wi_B_cpu_') as tmp:
        root=Path(tmp);folder=root/'groups'/'0';folder.mkdir(parents=True)
        write(root/'freeze.json',{'identity':'CPU_fake_only'})
        ids=[r['candidate_id'] for r in rows[:8]];pub={'ids':ids,'x_raw':x[:8].tolist(),'wl':xx[:8,:6].tolist()}
        write(folder/'public_features.json',pub)
        obs=[{'candidate_id':ids[i],'validation_accuracy':.5+i*.03,'task_id':f'fake{i}'} for i in range(3)]
        for i in range(2):
            d=decision(root,0,obs,initial_count=3)
            assert d['step']==i+1 and len(d['qlogei'])==8-len(obs) and d['observed_ids']==[o['candidate_id'] for o in obs]
            before=sha(folder/d['fit_checkpoint']);again=decision(root,0,obs,initial_count=3)
            assert d==again and before==sha(folder/d['fit_checkpoint'])
            obs.append({'candidate_id':d['candidate_id'],'validation_accuracy':.61+i*.02,'task_id':f'fakeOnline{i}'})
        assert len({o['candidate_id'] for o in obs})==5
        # Unobserved arbitrary metadata cannot affect observed-only scaling or labels.
        poisoned=dict(pub,private_unrequested_performance=[float('nan')]*8)
        assert training_projection(pub,obs)==training_projection(poisoned,obs)
        init_db(root)
        config={'operations':['GCNConv'],'edges':[[0,1],[1,2]],'normalization':'none','lr':.001}
        write(root/'protocol.json',{'dataset_context':{},'search_recipe':{'max_epochs':150,'patience':40},'evaluation_code_identity':'fake'})
        row={'candidate_id':digest(config),'configuration':config};key=enqueue(root,0,'search',0,row,123)
        assert enqueue(root,0,'search',0,row,123)==key
        with connect(root) as db:db.execute("UPDATE tasks SET status='succeeded' WHERE id=?",(key,))
        write(root/'training'/key/'result.json',{'identity':key,'test_metric_computations':0,'best_validation_accuracy':.7})
        assert result(root,key)['best_validation_accuracy']==.7
        assert enqueue(root,0,'search',0,row,123)==key and counts(root)['actual']=={'succeeded':1}
        fit_seconds=[read(p)['fit_seconds'] for p in folder.glob('fits/n*/fit.json')]
    result_report={'state':'PASS','checks':['RBF formula and original sequence','deterministic configuration keys and nonoverlapping seeds',
      'exact W-I formula, GaussianLikelihood and five parameters','fake3 initialization +2 qLogEI online','q=1 shape and MC step250',
      'observed-only isolation','saved decisions resume without refit','exact task cache resume without retrain','test_mask redacted'],
      'fake_GP_fit_seconds':fit_seconds,'elapsed_seconds':time.perf_counter()-started,'GPU_training_tasks':0,'fake_labels_used_only_in_temporary_directory':True}
    print(json.dumps(result_report),flush=True)
    return result_report

if __name__=='__main__':run()
