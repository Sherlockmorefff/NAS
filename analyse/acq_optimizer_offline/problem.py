"""Restore actual selected checkpoints; freeze parameters and original MC."""
import importlib
import math
import time
import numpy as np
import torch
import gpytorch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.logei import qLogExpectedImprovement
from .common import B,AC,read,sha,tensor_hash
from .geometry import WL

def matern(d,ell):
    t=math.sqrt(5.)*d/ell
    return (1.+t+t*t/3.)*torch.exp(-t)

class FixedSobol(SobolQMCNormalSampler):
    def __init__(self,base,seed):
        super().__init__(sample_shape=torch.Size([512]),seed=seed)
        self.saved=base.detach().clone().double()
        self.register_buffer('base_samples',self.saved.clone())
    def _construct_base_samples(self,posterior):
        shape=self._get_collapsed_shape(posterior)
        if shape.numel()!=512:raise ValueError('Fixed MC supports q1 single-output only')
        self.base_samples=self.saved.reshape(shape).to(device=posterior.device,dtype=posterior.dtype)

class ExtendedWI(Model):
    def __init__(self,state,parameters,gram):
        super().__init__();self.x=state['x'].detach().clone();self.y=state['y'].detach().clone()
        self.arch_dim=state['arch_dim'];self.p=parameters;self.jitter_setting=(1e-8,1e-7,1e-6)[state['jitter_index']]
        cov=gram+torch.eye(len(self.y),dtype=torch.float64)*parameters['extra_noise']
        self.used_jitter=0.
        try:self.L=torch.linalg.cholesky(cov)
        except torch.linalg.LinAlgError:
            self.L=torch.linalg.cholesky(cov+torch.eye(len(self.y),dtype=torch.float64)*self.jitter_setting);self.used_jitter=self.jitter_setting
        self.alpha=torch.cholesky_solve((self.y-parameters['mean'])[:,None],self.L)[:,0]
        self.norm_a=(self.x[:,:self.arch_dim]**2).sum(1)
        self.norm_h=(self.x[:,self.arch_dim:]**2).sum(1)
    @property
    def num_outputs(self):return 1
    @property
    def batch_shape(self):return torch.Size([])
    def posterior(self,X,observation_noise=False,**kwargs):
        if observation_noise is not False or X.shape[-2]!=1 or X.shape[-1]!=self.arch_dim+5:raise ValueError('Invalid extended latent query layout')
        q=X.squeeze(-2);a=q[...,:self.arch_dim];h=q[...,self.arch_dim:self.arch_dim+4];unknown=q[...,-1]
        da2=(a*a).sum(-1)[...,None]+unknown[...,None]+self.norm_a-2*a@self.x[:,:self.arch_dim].T
        dh2=(h*h).sum(-1)[...,None]+self.norm_h-2*h@self.x[:,self.arch_dim:].T
        if bool((da2<-1e-10).any() or (dh2<-1e-10).any()):raise FloatingPointError('Negative squared feature distance')
        # Roundoff in Euclidean dot-product identities only; never clip variances.
        ka=matern(da2.clamp_min(0.).sqrt(),self.p['ell_a']);kh=matern(dh2.clamp_min(0.).sqrt(),self.p['ell_h'])
        cross=self.p['outputscale']*(.25*(ka+kh)+.5*ka*kh)
        mean=self.p['mean']+cross@self.alpha
        solve=torch.linalg.solve_triangular(self.L,cross.reshape(-1,len(self.y)).T,upper=False)
        var=self.p['outputscale']-(solve*solve).sum(0).reshape(mean.shape)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(var).all()) or bool((var<0).any()):raise FloatingPointError('Invalid exact latent posterior')
        distribution=gpytorch.distributions.MultivariateNormal(mean.unsqueeze(-1),var[...,None,None])
        return GPyTorchPosterior(distribution)

class Problem:
    def __init__(self,family,n,wl=None):
        self.family,self.n=family,n
        self.source=(AC/'trajectories/A/0') if family=='A' else (B/'groups/0')
        self.fitdir=self.source/'fits'/f'n{n:03d}';self.fit=read(self.fitdir/'fit.json')
        self.cp=self.fitdir/f"start{self.fit['selected_start']}.pt"
        self.state=torch.load(self.cp,map_location='cpu',weights_only=False)
        if self.state['phase']!='COMPLETE' or self.state['identity']!=self.fit['identity']:raise ValueError('Frozen model identity mismatch')
        self.gp=importlib.import_module('analyse.wi_schur_AC_pool300.gp' if family=='A' else 'analyse.wi_schur_B_pool300.gp')
        self.model=self.gp.restore_model(self.state).eval();self.model.likelihood.eval()
        self.learned_before=tensor_hash({k:v for k,v in self.model.named_parameters()})
        expected=tensor_hash({k:self.state['model'][k] for k,v in self.model.named_parameters()})
        if expected!=self.learned_before:raise ValueError('Learnable checkpoint state changed on restore')
        for p in self.model.parameters():p.requires_grad_(False)
        self.parameters=self.gp.parameter_values(self.model)
        self.scaler=self.fit['scaler'];self.scales=self.fit['distance_scales']
        self.seen=set(self.scaler['fit_ids']);self.training=read(self.fitdir/'training.json')
        if self.training['ids']!=self.scaler['fit_ids'] or len(self.seen)!=n:raise ValueError('Frozen training order changed')
        self.step=n-49;self.decision=read(self.source/'decisions'/f'{self.step:03d}.json')
        if self.decision['model_sha256']!=sha(self.cp):raise ValueError('Decision checkpoint changed')
        self.sampler_path=self.source/'decisions'/f'{self.step:03d}_sampler.pt'
        sampler=torch.load(self.sampler_path,map_location='cpu',weights_only=False)
        if self.decision['sampler_sha256']!=sha(self.sampler_path):raise ValueError('Original MC state changed')
        self.base_samples=sampler['base_samples'];self.mc_seed=sampler['mc_seed'];self.best_f=float(self.state['y'].max())
        from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter
        self.original=FrozenAdapter(self.model,self.state['jitter_index'])
        with torch.no_grad(),self.gp.settings((1e-8,1e-7,1e-6)[self.state['jitter_index']]):
            self.gram=self.model.covar_module(self.state['x']).to_dense().detach()
        self.wl=wl or WL(read(B/'groups/0/wl_vocabulary.json'))
        self.adapter=self.original if family=='A' else ExtendedWI(self.state,self.parameters,self.gram)
        self.acq=self.make_acq(self.adapter);self.original_acq=self.make_acq(self.original)
        self.posterior_calls=0
        self.xmean=torch.tensor(self.scaler['x_mean'],dtype=torch.float64);self.xscale=torch.tensor(self.scaler['x_scale'],dtype=torch.float64)
    def make_acq(self,adapter):
        return qLogExpectedImprovement(model=adapter,best_f=self.best_f,sampler=FixedSobol(self.base_samples,self.mc_seed),
            fat=True,tau_max=.01,tau_relu=1e-6,eta=.001)
    def assert_unchanged(self):
        if tensor_hash({k:v for k,v in self.model.named_parameters()})!=self.learned_before:raise ValueError('GP parameter update detected')
        if self.original.policy.index!=self.state['jitter_index']:raise ValueError('Jitter strategy escalated from checkpoint')
    def relaxed_tensor(self,u):
        raw=torch.cat((-2.5+5*u[...,:12],(-4+2.5*u[...,12])[...,None],(.1+.5*u[...,13])[...,None],
                       (4+5*u[...,14])[...,None],(-5+2*u[...,15])[...,None]),dim=-1)
        return ((raw-self.xmean)/self.xscale)/self.scales['raw16']['value']
    def relaxed(self,u,gradient=False):
        x=torch.as_tensor(np.asarray(u),dtype=torch.float64).clone().requires_grad_(gradient)
        started=time.perf_counter();value=self.acq(self.relaxed_tensor(x).reshape(1,1,16)).reshape(())
        forward=time.perf_counter()-started;self.posterior_calls+=1
        if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite relaxed qLogEI')
        backward=0.;grad=None
        if gradient:
            started=time.perf_counter();grad=torch.autograd.grad(value,x)[0];backward=time.perf_counter()-started
            if not bool(torch.isfinite(grad).all()):raise FloatingPointError('Nonfinite relaxed gradient')
            grad=grad.detach().numpy()
        return float(value.detach()),grad,forward,backward
    def realized(self,cfg,z_float32):
        hp=np.array([np.log10(cfg['lr']),cfg['dropout'],np.log2(cfg['hidden_dim']),np.log10(cfg['l2'])])
        if self.family=='A':
            raw=np.r_[np.asarray(z_float32[:12],dtype='<f4').astype(np.float64),hp]
            x=(raw-np.asarray(self.scaler['x_mean']))/np.asarray(self.scaler['x_scale'])/self.scales['raw16']['value'];unknown=[]
        else:
            a,unknown2,unknown=self.wl.feature(cfg);d=self.scales['architecture']['value']
            h=(hp-np.asarray(self.scaler['x_mean'])[12:])/np.asarray(self.scaler['x_scale'])[12:]/self.scales['hp']['value']
            x=np.r_[a/d,h,unknown2/(d*d)]
        started=time.perf_counter()
        with torch.no_grad():score=self.acq(torch.tensor(x,dtype=torch.float64).reshape(1,1,-1)).item()
        self.posterior_calls+=1
        if not math.isfinite(score):raise FloatingPointError('Nonfinite realized qLogEI')
        return score,time.perf_counter()-started,len(unknown)
    def manifest(self):
        return {'family':self.family,'n':self.n,'source':str(self.source),'checkpoint':str(self.cp),'checkpoint_sha256':sha(self.cp),
          'selected_start':self.fit['selected_start'],'fit_identity':self.fit['identity'],'learnable_state_hash':self.learned_before,
          'training_order':self.training['ids'],'training_tensor_hash':tensor_hash({'x':self.state['x'],'y':self.state['y']}),
          'fit_json_sha256':sha(self.fitdir/'fit.json'),'training_json_sha256':sha(self.fitdir/'training.json'),
          'scaler':self.scaler,'distance_scales':self.scales,'parameters':self.parameters,'jitter_index':self.state['jitter_index'],
          'jitter_setting':(1e-8,1e-7,1e-6)[self.state['jitter_index']],'MC_seed':self.mc_seed,'MC_path':str(self.sampler_path),
          'MC_sha256':sha(self.sampler_path),'MC_base_tensor_hash':tensor_hash({'base':self.base_samples}),'best_f_standardized':self.best_f,
          'y_units':'standardized frozen validation; realized improvement multiplies y_scale once'}
