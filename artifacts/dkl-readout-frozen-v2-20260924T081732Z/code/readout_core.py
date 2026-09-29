"""Fixed-feature exact GP readouts. No labels/files outside supplied training tensors.

The Matérn receives h[:, :16] only. The fixed trend and finite-rank covariance
receive s alone. No mapper, head, coefficient optimizer, or resume entry exists.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
import gpytorch
import numpy as np
import torch
from torch import nn
from gpytorch.constraints import GreaterThan

METHODS = ('GP_CONST', 'GP_TREND', 'GP_SEMI')
COEFFICIENT_VARIANCES = (.0625, .25)
GP_NAMES = ['likelihood.second_noise_covar.raw_noise', 'covar_module.raw_outputscale',
            'covar_module.base_kernel.raw_lengthscale']


def cpu_setup(seed=0):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


@contextmanager
def numerical_settings():
    with ExitStack() as stack:
        for context in (gpytorch.settings.cholesky_jitter(double_value=1e-8),
                        gpytorch.settings.cholesky_max_tries(1),
                        gpytorch.settings.max_cholesky_size(10000),
                        gpytorch.settings.fast_pred_var(False),
                        gpytorch.settings.min_fixed_noise(double_value=0.0)):
            stack.enter_context(context)
        yield


def rng_state():
    return {'torch':torch.get_rng_state().clone(),'numpy':copy.deepcopy(np.random.get_state()),'python':random.getstate()}


def set_rng(value):
    torch.set_rng_state(value['torch']); np.random.set_state(value['numpy']); random.setstate(value['python'])


def tensor_hash(values):
    h=hashlib.sha256()
    for name,value in sorted(values.items()):
        value=value.detach().cpu().contiguous()
        h.update(json.dumps([name,list(value.shape),str(value.dtype)]).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class BudgetExceeded(TimeoutError): pass
class SyntheticStepLimit(RuntimeError): pass


def budget_limit(root):
    value=float(json.loads((Path(root)/'protocol.json').read_text())['budget']['model_seconds'])
    if not math.isfinite(value) or value<=0:raise ValueError('Invalid frozen protocol model budget')
    return value


class Budget:
    """Single persistent run ledger. Synthetic optimizer calls are capped at 300.

    Successful and failed optimizer invocations are counted separately. A failed
    validation can mark its already-counted successful updates discarded without
    removing them from synthetic/formal totals. No budget reset/resume is offered.
    """
    def __init__(self,path=None,limit=None):
        if limit is None:raise ValueError('Explicit frozen protocol budget limit required')
        self.path=None if path is None else Path(path); self.limit=float(limit)
        old=json.loads(self.path.read_text()) if self.path and self.path.exists() else {}
        if old and old['limit']!=self.limit:raise ValueError('Existing budget limit changed')
        if old and (old.get('updates',old['formal_updates'])!=old['formal_updates'] or old.get('verification_updates',old['synthetic_updates'])!=old['synthetic_updates']):raise ValueError('Existing budget aliases inconsistent')
        for name in ('training_seconds','prediction_seconds','verification_seconds'):
            setattr(self,name,float(old.get(name,0.)))
        for name in ('formal_updates','synthetic_updates','formal_optimizer_calls','synthetic_optimizer_calls',
                     'failed_optimizer_calls','failed_updates','discarded_updates','predictions','failed_training_attempts'):
            setattr(self,name,int(old.get(name,0)))
        self.synthetic_limit=300

    @property
    def updates(self): return self.formal_updates
    @property
    def verification_updates(self): return self.synthetic_updates
    @property
    def seconds(self): return self.training_seconds+self.prediction_seconds+self.verification_seconds

    def check(self):
        if self.seconds>=self.limit:raise BudgetExceeded(f'Cumulative {self.limit:g}-second model budget exhausted')

    def check_step(self,kind):
        self.check()
        if kind not in ('formal','synthetic'):raise ValueError('Explicit formal/synthetic update classification required')
        if kind=='synthetic' and self.synthetic_optimizer_calls>=self.synthetic_limit:
            raise SyntheticStepLimit('Cumulative synthetic optimizer.step limit of 300 reached')

    def before_optimizer_call(self,kind):
        self.check_step(kind)
        name=kind+'_optimizer_calls';setattr(self,name,getattr(self,name)+1)
        # Persist reservations for the scarce synthetic-call allowance; failures
        # and process exits must not create free extra verification calls.
        if kind=='synthetic':self.persist()

    def charge(self,seconds,kind='prediction',updates=0):
        if not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid elapsed cost')
        key={'training':'training_seconds','prediction':'prediction_seconds','verification':'verification_seconds'}.get(kind)
        if key is None:raise ValueError('Unknown timing category')
        setattr(self,key,getattr(self,key)+seconds)
        if kind=='prediction':self.predictions+=1
        if updates:
            if kind=='training':self.formal_updates+=updates
            elif kind=='verification':self.synthetic_updates+=updates
            else:raise ValueError('Prediction cannot consume optimizer updates')

    def mark_discarded(self,updates):
        if updates<0:raise ValueError('Invalid discarded update count')
        self.discarded_updates+=updates;self.persist()

    def as_dict(self):
        names=('training_seconds','prediction_seconds','verification_seconds','formal_updates','synthetic_updates',
               'formal_optimizer_calls','synthetic_optimizer_calls','failed_optimizer_calls','failed_updates',
               'discarded_updates','predictions','failed_training_attempts')
        return {**{name:getattr(self,name) for name in names},'seconds':self.seconds,'limit':self.limit,
                'updates':self.formal_updates,'verification_updates':self.synthetic_updates,'synthetic_optimizer_limit':300,
                'clock':'time.monotonic','automatic_resume':False}

    def persist(self):
        if self.path:
            self.path.parent.mkdir(parents=True,exist_ok=True);temp=self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.as_dict(),indent=2,allow_nan=False)+'\n');temp.replace(self.path)


def _isolate_transforms(module):
    for parent in list(module.modules()):
        for name,child in list(parent._modules.items()):
            if isinstance(child,nn.Softplus):setattr(parent,name,copy.deepcopy(child))


class FixedTrendMean(gpytorch.means.Mean):
    def forward(self,x): return x[...,16]


class _ReadoutGP(gpytorch.models.ExactGP):
    def __init__(self,x,y,known,common,method,coefficient_variances):
        with numerical_settings():
            likelihood=gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=known.clone(),learn_additional_noise=True,
                noise_constraint=GreaterThan(1e-6))
        super().__init__(x,y,likelihood)
        self.method=method
        self.mean_module=gpytorch.means.ConstantMean() if method=='GP_CONST' else FixedTrendMean()
        self.covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=None,
            lengthscale_constraint=GreaterThan(1e-4)),outputscale_constraint=GreaterThan(1e-8))
        self.register_buffer('coefficient_variances',torch.tensor(coefficient_variances,dtype=torch.float64))
        self.double();_isolate_transforms(self)
        def component(prefix):return {k[len(prefix):]:v.detach().clone() for k,v in common.items() if k.startswith(prefix)}
        self.likelihood.load_state_dict(component('likelihood.'),strict=True)
        self.covar_module.load_state_dict(component('covar_module.'),strict=True)
        if method=='GP_CONST':self.mean_module.load_state_dict(component('mean_module.'),strict=True)
        if list(self.named_priors()) or list(self.named_added_loss_terms()):raise RuntimeError('Prior-free data MLL required')

    def fixed_kernel(self,x,z):
        if self.method!='GP_SEMI':return torch.zeros(x.shape[-2],z.shape[-2],dtype=x.dtype,device=x.device)
        phi=torch.stack((torch.ones_like(x[...,16]),x[...,16]),dim=-1)
        psi=torch.stack((torch.ones_like(z[...,16]),z[...,16]),dim=-1)
        return (phi*self.coefficient_variances)@psi.transpose(-1,-2)

    def forward(self,x):
        covariance=self.covar_module(x[...,:16])
        if self.method=='GP_SEMI':covariance=covariance+self.fixed_kernel(x,x)
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x),covariance)


class ReadoutModel:
    def __init__(self,method,h_train,s_train,y_std,known_std,common_gp_state,source_fingerprint=None,
                 *,coefficient_variances=COEFFICIENT_VARIANCES,synthetic_fixture=False):
        if method not in METHODS:raise ValueError('Unknown GP readout method')
        if tuple(coefficient_variances)!=COEFFICIENT_VARIANCES and not synthetic_fixture:
            raise ValueError('Formal coefficient prior is fixed; override is synthetic-only')
        self.method=method;self.source_fingerprint=source_fingerprint;self.synthetic_fixture=bool(synthetic_fixture)
        self.h_train,self.s_train,self.y,self.known=[v.detach().clone().to(dtype=torch.float64,device='cpu') for v in (h_train,s_train,y_std,known_std)]
        n=len(self.y)
        if self.h_train.shape!=(n,16) or any(v.shape!=(n,) for v in (self.s_train,self.y,self.known)):
            raise ValueError('Readout requires aligned frozen h16/s/y/known variance')
        if not all(bool(torch.isfinite(v).all()) for v in (self.h_train,self.s_train,self.y,self.known)) or bool((self.known<0).any()):
            raise ValueError('Nonfinite readout training data or negative known variance')
        self._common_state=copy.deepcopy(common_gp_state)
        self.x=torch.cat((self.h_train,self.s_train[:,None]),dim=-1)
        self.gp=_ReadoutGP(self.x,self.y,self.known,self._common_state,method,coefficient_variances)
        self._mll=gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood,self.gp)
        self.initial_frozen_hash=self.frozen_hash()

    def frozen_hash(self):return tensor_hash({'h0':self.h_train,'s0':self.s_train,'y':self.y,'known':self.known,'x17':self.x})
    def parameter_names(self):return [name for name,p in self.gp.named_parameters() if p.requires_grad]
    def hyperparameters(self):
        return {'lengthscale':float(self.gp.covar_module.base_kernel.lengthscale.detach().item()),
                'outputscale':float(self.gp.covar_module.outputscale.detach().item()),
                'extra_noise_variance_standardized':float(self.gp.likelihood.second_noise.detach().item()),
                'constant_mean':float(self.gp.mean_module.constant.detach().item()) if self.method=='GP_CONST' else None}
    def configuration(self):
        return {'method':self.method,'input':'h0_16 plus fixed_s0','matern_dimensions':16,'mean':'constant' if self.method=='GP_CONST' else 'fixed_s0',
                'coefficient_prior_mean':[0.,1.] if self.method=='GP_SEMI' else None,
                'coefficient_prior_variances':self.gp.coefficient_variances.tolist() if self.method=='GP_SEMI' else None,
                'outputscale_scope':'Matern_only','objective':'negative_ExactMLL_per_training_point','synthetic_fixture':self.synthetic_fixture}
    def fingerprint(self):
        parameters=tensor_hash({**{'gp.'+k:v for k,v in self.gp.state_dict().items()},
                               'h0':self.h_train,'s0':self.s_train,'y':self.y,'known':self.known})
        # TREND and SEMI can share every learned tensor at initialization while
        # defining different covariances. Their mathematical identities must
        # therefore include the fixed model/prior configuration as well.
        return hashlib.sha256((parameters+'|'+digest(self.configuration())).encode()).hexdigest()

    def nll(self):
        self.gp.train();self.gp.likelihood.train()
        with numerical_settings():return -self._mll(self.gp(self.x),self.y)

    def _query(self,h,s):
        if h.ndim!=2 or h.shape[1]!=16 or s.shape!=(len(h),) or h.dtype!=torch.float64 or s.dtype!=torch.float64:
            raise ValueError('Query requires frozen float64 h16 and s0')
        if h.requires_grad or s.requires_grad or not bool(torch.isfinite(h).all() and torch.isfinite(s).all()):
            raise ValueError('Query features must be detached and finite')
        return torch.cat((h,s[:,None]),dim=-1)

    def predict(self,h,s):
        query=self._query(h,s)
        with torch.no_grad(),numerical_settings():
            self.gp.eval();self.gp.likelihood.eval()
            posterior=self.gp(query)
            mean=posterior.mean.detach().clone();covariance=posterior.lazy_covariance_matrix.to_dense().detach().clone()
        if not bool(torch.isfinite(mean).all() and torch.isfinite(covariance).all()) or bool((covariance.diagonal() < -1e-8).any()):
            raise FloatingPointError('Invalid complete latent posterior')
        return mean,covariance

    def kernel_blocks(self,h,s,h_other=None,s_other=None):
        first=self._query(h,s);second=first if h_other is None else self._query(h_other,s_other)
        matern=self.gp.covar_module(first[:,:16],second[:,:16]).to_dense()
        return {'matern':matern,'full':matern+self.gp.fixed_kernel(first,second)}

    def kernels(self,h,s):return self.kernel_blocks(h,s)

    def semi_posterior(self,h,s):
        if self.method!='GP_SEMI':raise ValueError('Coefficient posterior belongs to GP_SEMI')
        with torch.no_grad(),numerical_settings():
            phi=torch.stack((torch.ones_like(self.s_train),self.s_train),dim=-1)
            phiq=torch.stack((torch.ones_like(s),s),dim=-1)
            B=torch.diag(self.gp.coefficient_variances); b0=torch.tensor([0.,1.],dtype=torch.float64)
            c=self.kernels(self.h_train,self.s_train)['full']+torch.diag(self.known+self.gp.likelihood.second_noise)
            chol,info=torch.linalg.cholesky_ex(c);jitter=0.
            if bool(info.any()):
                jitter=1e-8;chol=torch.linalg.cholesky(c+jitter*torch.eye(len(c),dtype=c.dtype))
            alpha=torch.cholesky_solve((self.y-self.s_train)[:,None],chol).squeeze(-1)
            beta=b0+B@phi.T@alpha
            beta_cov=B-B@phi.T@torch.cholesky_solve(phi@B,chol)
            trend=phiq@beta
            residual=self.kernel_blocks(h,s,self.h_train,self.s_train)['matern']@alpha
        return {'beta_mean':beta,'beta_cov':beta_cov,'trend_mean':trend,'residual_mean':residual,'solver_jitter':jitter}

    def snapshot(self):
        return {'method':self.method,'configuration':self.configuration(),'configuration_hash':digest(self.configuration()),
                'source_fingerprint':self.source_fingerprint,'gp':copy.deepcopy(self.gp.state_dict()),
                'common_gp_state':copy.deepcopy(self._common_state),'h_train':self.h_train.clone(),'s_train':self.s_train.clone(),
                'y_std':self.y.clone(),'known_std':self.known.clone(),'frozen_hash':self.frozen_hash(),
                'fingerprint':self.fingerprint(),'parameter_names':self.parameter_names(),'rng':rng_state()}

    @classmethod
    def from_snapshot(cls,state):
        config=state['configuration']
        model=cls(state['method'],state['h_train'],state['s_train'],state['y_std'],state['known_std'],state['common_gp_state'],
                  state['source_fingerprint'],coefficient_variances=state['gp']['coefficient_variances'].tolist(),synthetic_fixture=config['synthetic_fixture'])
        model.gp.load_state_dict(state['gp'],strict=True)
        if model.fingerprint()!=state['fingerprint'] or model.frozen_hash()!=state['frozen_hash'] or digest(model.configuration())!=state['configuration_hash']:
            raise ValueError('Restored readout state/configuration changed')
        set_rng(state['rng']);return model

    def clone(self):
        incoming=rng_state()
        try:return self.from_snapshot(self.snapshot())
        finally:set_rng(incoming)


def learning_rate_for_update(update):
    if not 1<=update<=1500:raise ValueError('Readout update number outside fixed 1..1500')
    return .01 if update<=1000 else .001


class Session:
    def __init__(self,model,planned_steps=1500):
        if planned_steps!=1500:raise ValueError('All readout trajectories retain the fixed 1500-step schedule')
        self.model=model;self.planned_steps=1500;self.completed=0;self.training_seconds=0.;self.last_update=None
        self.names=model.parameter_names();self.params=[p for p in model.gp.parameters() if p.requires_grad]
        expected=set(GP_NAMES+(['mean_module.raw_constant'] if model.method=='GP_CONST' else []))
        if set(self.names)!=expected or len(self.names)!=len(expected):raise ValueError('Unauthorized/missing trainable readout parameters: '+str(self.names))
        self.optimizer=torch.optim.Adam([{'params':self.params,'lr':.01,'name':'gp'}],betas=(.9,.999),eps=1e-8,weight_decay=0.)

    @property
    def lrs(self):return {'gp':float(self.optimizer.param_groups[0]['lr'])}

    def step(self,budget,kind='formal'):
        budget.check_step(kind)
        if self.model.synthetic_fixture and kind!='synthetic':raise ValueError('Synthetic fixture cannot enter formal fit')
        if self.completed>=1500:raise RuntimeError('Fixed readout endpoint already reached')
        start=time.monotonic();called=False;updated=0
        try:
            self.optimizer.param_groups[0]['lr']=learning_rate_for_update(self.completed+1)
            self.optimizer.zero_grad(set_to_none=True);loss=self.model.nll()
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite readout NLL')
            loss.backward()
            gradient=float(torch.nn.utils.clip_grad_norm_(self.params,5.,error_if_nonfinite=True))
            if any(p.grad is None for p in self.params):raise RuntimeError('Trainable GP parameter has no gradient')
            budget.before_optimizer_call(kind);called=True;self.optimizer.step();updated=1;self.completed+=1
            self.last_update={'method':self.model.method,'step':self.completed,'planned_steps':1500,
                'nll_before_update':float(loss.detach()),'gradient_before_clip':gradient,'learning_rates':self.lrs,'kind':kind}
        except Exception:
            budget.failed_training_attempts+=1
            if called:budget.failed_optimizer_calls+=1;budget.failed_updates+=1
            raise
        finally:
            elapsed=time.monotonic()-start;self.training_seconds+=elapsed
            budget.charge(elapsed,'training' if kind=='formal' else 'verification',updates=updated)
        self.last_update.update(step_seconds=elapsed,block_training_seconds=self.training_seconds,
                                formal_updates=budget.formal_updates,synthetic_updates=budget.synthetic_updates)
        budget.check();return self.last_update

    def snapshot(self):
        state=self.model.snapshot()
        state.update(optimizer=copy.deepcopy(self.optimizer.state_dict()),ordered_parameter_names=list(self.names),
            scheduler={'completed':self.completed,'planned_steps':1500,'policy':'updates1to1000_0.01_then1001to1500_0.001'},
            accounting={'completed_updates':self.completed,'block_training_seconds':self.training_seconds},last_update=copy.deepcopy(self.last_update))
        return state


def make_synthetic_fixture(n=20,seed=123):
    """Deterministic data and non-default shared GP state; no optimizer updates."""
    from stage01_core import ExactGP
    incoming=rng_state()
    try:
        gen=torch.Generator(device='cpu').manual_seed(seed)
        h=.3*torch.randn(n,16,generator=gen,dtype=torch.float64)
        hq=.3*torch.randn(12,16,generator=gen,dtype=torch.float64)
        s=.25*h.sum(1)+torch.linspace(-1.3,1.3,n,dtype=torch.float64)
        sq=.25*hq.sum(1)+torch.linspace(-1.7,1.7,len(hq),dtype=torch.float64)
        y=.2+.85*s+.07*torch.randn(n,generator=gen,dtype=torch.float64)
        known=(torch.linspace(.004,.012,n,dtype=torch.float64)/.2).square()
        gp=ExactGP(h,y,known);_isolate_transforms(gp)
        gp.mean_module.constant=.2;gp.covar_module.base_kernel.lengthscale=1.7
        gp.covar_module.outputscale=.8;gp.likelihood.second_noise=.03
        return {'h_train':h,'s_train':s,'y_std':y,'known_std':known,'common_gp_state':copy.deepcopy(gp.state_dict()),
            'h_query':hq,'s_query':sq,'training_ids':['train_%03d'%i for i in range(n)],
            'query_ids':['query_%03d'%i for i in range(len(hq))],'y_center':.6,'y_scale':.2}
    finally:set_rng(incoming)
