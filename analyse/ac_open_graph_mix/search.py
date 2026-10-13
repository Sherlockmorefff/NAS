"""Seen-only full-history fitting and differentiable frozen acquisition."""
import math
import time
from pathlib import Path
import numpy as np
import torch
import gpytorch
from scipy.spatial.distance import pdist
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.logei import qLogExpectedImprovement
from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter
from .common import *
from . import gp,graphs

def positive(x):
    d=pdist(x);p=d[d>0]
    return {'value':float(np.median(p)) if len(p) else 1.,'degenerate':not len(p),'positive_pairs':len(p)}
def projection(obs,method,nmax=7,emax=16):
    ids=[o['candidate_id'] for o in obs];y=np.array([o['validation_accuracy'] for o in obs]);ym=float(y.mean());ys=float(y.std())
    if len(ids)!=len(set(ids)) or not np.isfinite(y).all() or ys<=0:raise ValueError('Invalid/constant seen-only observations')
    hp=np.array([graphs.hp(o['configuration']) for o in obs]);hs=hp.std(0);hs[np.ptp(hp,axis=0)==0]=1.
    if method=='A':
        raw=np.array([o['latent']+graphs.hp(o['configuration']) for o in obs]);xm=raw.mean(0);xs=raw.std(0);xs[np.ptp(raw,axis=0)==0]=1.
        z=(raw-xm)/xs;scales={'raw16':positive(z)};x=z/scales['raw16']['value'];ad=16
        scaler={'x_mean':xm.tolist(),'x_scale':xs.tolist(),'y_mean':ym,'y_scale':ys,'ddof':0}
        vocab=[]
    else:
        configs=[o['configuration'] for o in obs];vocab=graphs.vocabulary(configs);psi=graphs.features(configs,vocab,nmax,emax)
        z=(hp-hp.mean(0))/hs;scales={'architecture':positive(psi),'hp':positive(z)}
        x=np.c_[psi/scales['architecture']['value'],z/scales['hp']['value']];ad=psi.shape[1]
        scaler={'hp_mean':hp.mean(0).tolist(),'hp_scale':hs.tolist(),'y_mean':ym,'y_scale':ys,'ddof':0}
    return {'ids':ids,'scaler':scaler,'distance_scales':scales,'tokens':vocab,'x':x.tolist(),'y':((y-ym)/ys).tolist(),
            'arch_dim':ad,'method':method,'n_max':nmax,'e_max':emax,'full_history_token_axes':method=='C'}

class GraphPosterior(Model):
    def __init__(self,state,model,training,observations):
        super().__init__();self.p=gp.parameter_values(model);self.training=training;self.obs=observations
        x=state['x'];ad=state['arch_dim'];self.h=x[:,ad:];self.graph_ids={};self.ka=[]
        with torch.no_grad(),gp.settings((1e-8,1e-7,1e-6)[state['jitter_index']]):K=model.covar_module(x).to_dense()
        covariance=K+torch.eye(len(x),dtype=torch.float64)*self.p['extra_noise'];self.jitter_used=0.
        try:self.L=torch.linalg.cholesky(covariance)
        except torch.linalg.LinAlgError:
            self.jitter_used=(1e-8,1e-7,1e-6)[state['jitter_index']]
            self.L=torch.linalg.cholesky(covariance+torch.eye(len(x),dtype=torch.float64)*self.jitter_used)
        self.alpha=torch.cholesky_solve((state['y']-self.p['mean'])[:,None],self.L)[:,0]
    def add_graph(self,cfg):
        key=graphs.aid(cfg)
        if key in self.graph_ids:return self.graph_ids[key]
        # Explicit complete T-union-Q axes; both historical and query new symbols participate.
        configs=[o['configuration'] for o in self.obs]+[cfg];vocab=graphs.vocabulary(configs)
        psi=graphs.features(configs,vocab,self.training['n_max'],self.training['e_max'])
        da=np.linalg.norm(psi[:-1]-psi[-1],axis=1)/self.training['distance_scales']['architecture']['value']
        t=math.sqrt(5)*torch.tensor(da,dtype=torch.float64)/self.p['ell_a']
        ka=(1+t+t*t/3)*torch.exp(-t);index=len(self.ka);self.ka.append(ka);self.graph_ids[key]=index
        return index
    @property
    def num_outputs(self):return 1
    @property
    def batch_shape(self):return torch.Size([])
    def posterior(self,X,observation_noise=False,**kw):
        if observation_noise is not False or X.shape[-2:]!=(1,5):raise ValueError('Expected independent q1 graph-index/HP query')
        q=X.squeeze(-2);h=q[...,:4];indices=q[...,-1].long();ka=torch.stack(self.ka)[indices]
        dh=torch.linalg.vector_norm(h.unsqueeze(-2)-self.h,dim=-1);t=math.sqrt(5)*dh/self.p['ell_h']
        kh=(1+t+t*t/3)*torch.exp(-t);cross=self.p['outputscale']*(.25*(ka+kh)+.5*ka*kh)
        mu=self.p['mean']+cross@self.alpha
        solve=torch.linalg.solve_triangular(self.L,cross.reshape(-1,len(self.h)).T,upper=False)
        var=self.p['outputscale']-(solve*solve).sum(0).reshape(mu.shape)
        if not torch.isfinite(mu).all() or not torch.isfinite(var).all() or (var<0).any():raise FloatingPointError('Invalid differentiable latent posterior')
        return GPyTorchPosterior(gpytorch.distributions.MultivariateNormal(mu[...,None],var[...,None,None]))

class Acquisition:
    def __init__(self,state,training,obs,method,mc):
        self.method=method;self.training=training;self.obs=obs;self.model=gp.restore_model(state).eval();self.model.likelihood.eval()
        self.parameter_state={k:v.detach().clone() for k,v in self.model.named_parameters()}
        for p in self.model.parameters():p.requires_grad_(False)
        self.adapter=FrozenAdapter(self.model,state['jitter_index']) if method=='A' else GraphPosterior(state,self.model,training,obs)
        self.sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]),seed=mc)
        self.acq=qLogExpectedImprovement(model=self.adapter,best_f=float(state['y'].max()),sampler=self.sampler,fat=True,tau_max=.01,tau_relu=1e-6,eta=.001)
        self.posterior_points=0;self.gradients=0;self.forward_seconds=0.;self.backward_seconds=0.;self.WL_seconds=0.
    def graph_index(self,cfg):
        started=time.perf_counter();i=self.adapter.add_graph(cfg);self.WL_seconds+=time.perf_counter()-started;return i
    def raw_transform(self,raw):
        s=self.training['scaler'];return (raw-torch.tensor(s['x_mean']))/torch.tensor(s['x_scale'])/self.training['distance_scales']['raw16']['value']
    def a_input(self,u):
        return self.raw_transform(torch.cat([-2.5+5*u[...,:12],(-4+2.5*u[...,12])[...,None],(.1+.5*u[...,13])[...,None],(4+5*u[...,14])[...,None],(-5+2*u[...,15])[...,None]],-1))
    def c_input(self,v,index,hidden,l2):
        raw=torch.cat([(-4+2.5*v[...,0])[...,None],(.1+.5*v[...,1])[...,None],torch.full_like(v[...,:1],math.log2(hidden)),torch.full_like(v[...,:1],math.log10(l2))],-1)
        s=self.training['scaler'];h=(raw-torch.tensor(s['hp_mean']))/torch.tensor(s['hp_scale'])/self.training['distance_scales']['hp']['value']
        return torch.cat([h,torch.full_like(h[...,:1],float(index))],-1)
    def score(self,x,gradient=False):
        started=time.perf_counter();value=self.acq(x.unsqueeze(-2));self.forward_seconds+=time.perf_counter()-started;self.posterior_points+=x.reshape(-1,x.shape[-1]).shape[0]
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite fixed-MC qLogEI')
        return value
    def evaluate(self,u,gradient=False,index=None,hidden=None,l2=None):
        u=torch.tensor(np.asarray(u),dtype=torch.float64,requires_grad=gradient)
        x=self.a_input(u) if self.method=='A' else self.c_input(u,index,hidden,l2)
        value=self.score(x.reshape(1,-1)).reshape(());derivative=None
        if gradient:
            start=time.perf_counter();derivative=torch.autograd.grad(value,u)[0];self.backward_seconds+=time.perf_counter()-start;self.gradients+=1
            if not torch.isfinite(derivative).all():raise FloatingPointError('Nonfinite exact acquisition gradient')
            derivative=derivative.detach().numpy()
        return float(value.detach()),derivative
    def actual_A(self,cfg,latent):
        raw=torch.tensor(latent+graphs.hp(cfg),dtype=torch.float64);return float(self.score(self.raw_transform(raw).reshape(1,16)).detach())
    def unchanged(self):
        if any(not torch.equal(v,self.parameter_state[k]) for k,v in self.model.named_parameters()):raise ValueError('GP changed during acquisition')

def fit(root,method,g,obs):
    gp.cpu_setup();f=folder(root,method,g);n=len(obs);nmax,emax=read(inputs(root,g)/'wl_vocabulary.json')['n_max'],read(inputs(root,g)/'wl_vocabulary.json')['e_max']
    start=time.perf_counter();training=projection(obs,method,nmax,emax);transform_seconds=time.perf_counter()-start
    identity=digest({'freeze':read(root/'freeze.json')['identity'],'method':method,'group':g,'observations':obs,'transform':training})
    fd=f/'fits'/f'n{n:03d}';fd.mkdir(parents=True,exist_ok=True)
    seal(fd/'training.json',training)
    if (fd/'fit.json').exists():
        meta=read(fd/'fit.json')
        if meta['identity']!=identity:raise ValueError('Open fit identity changed')
    else:
        starts=[gp.fit_start(training['x'],training['y'],KERNELS[method],training['arch_dim'],i,fd/f'start{i}.pt',identity) for i in range(3)]
        if not any(x['success'] and np.isfinite(x['final_nll']) for x in starts):raise RuntimeError('All open GP starts failed: see saved starts')
        meta={'identity':identity,'selected_start':gp.select_start(starts),'starts':starts,'fit_seconds':sum(x['elapsed_seconds'] for x in starts),'transform_seconds':transform_seconds,'n':n,'sealed_at':now()};seal(fd/'fit.json',meta)
    checkpoint=fd/f"start{meta['selected_start']}.pt";state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    return state,training,meta,checkpoint

def decision(root,method,g,obs,post_only=False):
    f=folder(root,method,g);path=f/'decisions'/f'{len(obs)-49:03d}.json'
    if path.exists():
        d=read(path)
        if d['observations_identity']!=digest(obs):raise ValueError('Open decision observations mismatch')
        return d
    state,training,meta,checkpoint=fit(root,method,g,obs)
    from .optimize import Round
    r=Round(root,method,g,obs,Acquisition(state,training,obs,method,read(inputs(root,g)/'seeds.json')['qlogei_mc'][len(obs)-50]))
    selected=r.execute(post_only)
    saved={'method':method,'group':g,'step':len(obs)-49,'observed_n':len(obs),'observed_ids':[o['candidate_id'] for o in obs],
      'observations_identity':digest(obs),'candidate_id':selected['candidate_id'],'selected_row':selected,'fit_identity':meta['identity'],
      'fit_checkpoint':str(checkpoint.relative_to(f)),'model_sha256':sha(checkpoint),'fit_seconds':meta['fit_seconds'],
      'transform_seconds':meta['transform_seconds'],'acquisition_seconds':r.elapsed(),'requests':r.state['requests'],
      'post_requests':r.state['post_requests'],'fallback_requests':r.state['fallback_requests'],'diagnostics':r.diagnostics(),
      'mc_seed':r.mc,'parameters':gp.parameter_values(r.p.model),'sealed_at':now()}
    seal(path,saved);r.save()
    if r.elapsed()>300:raise TimeoutError('Decision recording passed acquisition deadline')
    return saved
