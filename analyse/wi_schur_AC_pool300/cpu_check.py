import json
import sqlite3
import tempfile
import time
from pathlib import Path
import numpy as np
from .common import *

def run():
    import torch
    import gpytorch
    from scipy.spatial.distance import cdist
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.acquisition.logei import qLogExpectedImprovement
    from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter
    from . import gp
    from .initialization import signal_kernel,select
    from .search import decision,training_projection
    from .scoring import construct_model
    start=time.perf_counter();gp.cpu_setup();rng=np.random.default_rng(142)
    pub={'ids':[f'{i:03d}' for i in range(10)],'x_raw':rng.normal(size=(10,16)).tolist(),'wl':rng.normal(size=(10,8)).tolist()}
    matrix,transform=signal_kernel(pub)
    hp=(np.asarray(pub['x_raw'])[:,-4:]-transform['HP_pool_mean'])/transform['HP_pool_std_ddof0']
    def matern(d):
        z=np.sqrt(5)*d;return (1+z+z*z/3)*np.exp(-z)
    kw=matern(cdist(np.asarray(pub['wl'])/transform['dW']['value'],np.asarray(pub['wl'])/transform['dW']['value']))
    kh=matern(cdist(hp/transform['dH']['value'],hp/transform['dH']['value']))
    assert np.allclose(matrix,.25*kw+.25*kh+.5*kw*kh,atol=1e-12)
    chosen=select(pub,'003',5);assert chosen==select(dict(pub,unrequested_performance=[float('nan')]*10),'003',5)
    assert chosen['first_point']=='003' and len(set(chosen['ids']))==5
    indexes=[]
    for s in chosen['selection']:
        i=pub['ids'].index(s['candidate_id'])
        variance=matrix[i,i] if not indexes else matrix[i,i]-matrix[i,indexes]@np.linalg.solve(matrix[np.ix_(indexes,indexes)]+np.eye(len(indexes))*1e-6,matrix[indexes,i])
        assert np.isclose(variance,s['conditional_variance'],atol=1e-12);indexes.append(i)
    for method,dim,count in [('RAW16',16,4),('W-I',12,5)]:
        xx=torch.tensor(rng.normal(size=(8,dim)),dtype=torch.float64);y=torch.linspace(-1,1,8)
        model=gp.StaticExactGP(xx,y,method,16 if method=='RAW16' else 8)
        assert sum(p.numel() for p in model.parameters())==count
        assert isinstance(model.likelihood,gpytorch.likelihoods.GaussianLikelihood) and not isinstance(model.likelihood,gpytorch.likelihoods.FixedNoiseGaussianLikelihood)
        if method=='RAW16':expected=matern(cdist(xx.numpy(),xx.numpy()))
        else:
            ka=matern(cdist(xx[:,:8].numpy(),xx[:,:8].numpy()));kh=matern(cdist(xx[:,8:].numpy(),xx[:,8:].numpy()));expected=.25*(ka+kh)+.5*ka*kh
        assert np.allclose(model.covar_module(xx).to_dense().detach().numpy(),expected,atol=1e-12)
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]),seed=seed(0,'mc',250))
        with torch.no_grad():score=qLogExpectedImprovement(FrozenAdapter(model),best_f=0.,sampler=sampler)(xx[:2].unsqueeze(-2))
        assert score.shape==(2,) and bool(torch.isfinite(score).all())
    config={'operations':['GCNConv'],'edges':[[0,1],[1,2]],'normalization':'none','lr':.001,'dropout':.1,'hidden_dim':8,'l2':.0005,
       'gcnii_alpha':.1,'gcnii_theta':.5,'gat_heads':1,'sage_aggr':'mean','gin_eps':0.}
    torch.set_default_dtype(torch.float64);scorer=construct_model(config,3,2,17,torch.device('cpu'))
    assert torch.get_default_dtype()==torch.float32 and all(p.dtype==torch.float32 for p in scorer.parameters())
    scorer.eval()
    with torch.no_grad():assert scorer(torch.randn(5,3),torch.tensor([[0,1,2,3],[1,2,3,4]])).dtype==torch.float32
    seedset=[seed(g,phase,i) for g in range(5) for phase,n in [('search',1),('validation',30),('final',30)] for i in range(n)]
    assert len(seedset)==len(set(seedset)) and seed(0,'mc',250)!=seed(0,'mc',1)
    with tempfile.TemporaryDirectory(prefix='AC_cpu_') as tmp:
        root=Path(tmp)/'AC';b=Path(tmp)/'B';root.mkdir();b.mkdir()
        write(root/'freeze.json',{'identity':'CPU_only_fake_labels'})
        write(inputs(root,0)/'public_features.json',pub)
        timings={}
        for method in METHODS:
            f=folder(root,method,0);f.mkdir(parents=True)
            obs=[{'candidate_id':pub['ids'][i],'validation_accuracy':.5+i*.03,'task_id':f'fake{i}'} for i in range(3)]
            for i in range(2):
                d=decision(root,method,0,obs,initial_count=3)
                assert d['method']==method and d['step']==i+1 and len(d['qlogei'])==10-len(obs)
                assert decision(root,method,0,obs,initial_count=3)==d
                obs.append({'candidate_id':d['candidate_id'],'validation_accuracy':.61+i*.02,'task_id':f'fakeOnline{i}'})
            assert training_projection(pub,obs)==training_projection(dict(pub,unknown_labels=[float('nan')]*10),obs)
            timings[method]=[read(p)['fit_seconds'] for p in f.glob('fits/*/fit.json')]
        recipe={'max_epochs':150,'patience':40};protocol={'B_root':str(b),'dataset_context':{},'search_recipe':recipe,'final_recipe':{'max_epochs':300,'patience':80},'evaluation_code_identity':'fake'}
        write(root/'AC_PROTOCOL.json',protocol);init_db(root)
        with sqlite3.connect(b/'tasks.sqlite') as db:
            db.execute('CREATE TABLE tasks(id TEXT,status TEXT,config TEXT,seed INTEGER,epochs INTEGER,patience INTEGER,elapsed REAL,finished REAL)')
            key=task_key(protocol,config,11,'search');db.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?)',(key,'succeeded',json.dumps(config),11,150,40,2.,123.))
        write(b/'training'/key/'result.json',{'identity':key,'seed':11,'test_metric_computations':0,'best_validation_accuracy':.8})
        row={'candidate_id':digest(config),'configuration':config}
        assert enqueue(root,'A',0,'search',0,row,11)==key and enqueue(root,'C',0,'search',0,row,11)==key
        assert result(root,key)['best_validation_accuracy']==.8
        new=enqueue(root,'A',0,'search',1,row,12);assert enqueue(root,'C',0,'search',1,row,12)==new and new!=key
        with connect(root) as db:
            assert db.execute('SELECT count(*) FROM logical').fetchone()[0]==4
            assert db.execute('SELECT count(*) FROM tasks').fetchone()[0]==2
            db.execute('UPDATE tasks SET status=? WHERE id=?',('succeeded',new))
        write(root/'training'/new/'result.json',{'identity':new,'seed':12,'test_metric_computations':0,'best_validation_accuracy':.7})
        assert enqueue(root,'A',0,'search',1,row,12)==new and result(root,new)['best_validation_accuracy']==.7 and task(root,new)['attempts']==0
        assert task_key(protocol,config,12,'final')!=new
    report={'state':'PASS','checks':['RAW16 and WI formula/4 vs5 parameters/GaussianLikelihood','C label-free fixed Schur reference/shared first/unique order',
       'A and C fake3 initialization+2 qLogEI each','MC250 seed and512 sampler shape','method/group fit path isolation',
       'exact B/AC shared task key and separate logic','completed cache/resume does not enqueue training','seen-only transforms',
       'float64 GP to isolated float32 scoring construction/CPU forward','stage and group seeds nonoverlapping'],
       'CPU_fake_GP_seconds':timings,'elapsed_seconds':time.perf_counter()-start,'GPU_training_tasks':0,'Cora_downloads':0}
    print(json.dumps(report),flush=True);return report

if __name__=='__main__':run()
