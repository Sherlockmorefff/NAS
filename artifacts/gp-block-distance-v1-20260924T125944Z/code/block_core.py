"""Three raw-input exact GPs, fixed schedules and a fresh classified budget.

No mapper, head, historical model restore, query observation interface or resume.
The block kernel owns exactly two scalar length parameters. Its internal
Matérn has one fixed unit length, excluded from optimization.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch import nn
import gpytorch
from gpytorch.constraints import GreaterThan
import readout_core as inherited
from readout_core import numerical_settings,cpu_setup,rng_state,set_rng,tensor_hash,digest,BudgetExceeded,SyntheticStepLimit

METHODS=('RAW16_ISO','HP4_ISO','RAW16_BLOCK2')
STAGES=('INIT_GP','FIT_GP')
STEPS={'INIT_GP':300,'FIT_GP':1500}
TRAIN_KEYS={'ids','architectures','x_standardized','y_standardized','known_variance_standardized'}
QUERY_KEYS={'ids','architectures','x_standardized'}
COUNTERS=('optimizer_calls','successful_optimizer_calls','updates','failed_optimizer_calls','failed_updates','failed_training_attempts')


def budget_limit(root):
    limit=inherited.budget_limit(root)
    if limit!=600:raise ValueError('Block-distance model budget must be exactly600')
    return limit


class Budget(inherited.Budget):
    def __init__(self,path=None,limit=None):
        if limit!=600:raise ValueError('Explicit frozen600-second budget required')
        old=json.loads(Path(path).read_text()) if path and Path(path).exists() else {}
        if old and old.get('accounting_version')!='block_distance_v1':raise ValueError('Cannot inherit previous experiment cost/count ledger')
        super().__init__(path,limit)
        self.method_stage_counters=copy.deepcopy(old.get('method_stage_counters',{
            m:{s:{k:{c:0 for c in COUNTERS} for k in ('formal','synthetic')} for s in STAGES} for m in METHODS}))
        self.operation_stats=copy.deepcopy(old.get('operation_stats',{}))
        for kind in ('formal','synthetic'):
            for field,suffix in [('optimizer_calls','optimizer_calls'),('updates','updates')]:
                if sum(self.method_stage_counters[m][s][kind][field] for m in METHODS for s in STAGES)!=getattr(self,kind+'_'+suffix):
                    raise ValueError('Classified/global budget mismatch')

    def before_optimizer_call(self,kind,method=None,stage=None):
        if method not in METHODS or stage not in STAGES:raise ValueError('Classify every optimizer invocation')
        self.check_step(kind)
        setattr(self,kind+'_optimizer_calls',getattr(self,kind+'_optimizer_calls')+1)
        self.method_stage_counters[method][stage][kind]['optimizer_calls']+=1
        if kind=='synthetic':self.persist()

    def charge(self,seconds,kind='prediction',updates=0):
        if updates:raise ValueError('Updates require the classified session path')
        super().charge(seconds,kind,0)

    def step_charge(self,seconds,kind,method,stage,updated):
        super().charge(seconds,'training' if kind=='formal' else 'verification',updated)
        self.method_stage_counters[method][stage][kind]['updates']+=updated

    def as_dict(self):
        result=super().as_dict()
        result.update(accounting_version='block_distance_v1',method_stage_counters=copy.deepcopy(self.method_stage_counters),
            formal_steps_by_method={m:sum(self.method_stage_counters[m][s]['formal']['updates'] for s in STAGES) for m in METHODS},
            formal_steps_by_stage={s:sum(self.method_stage_counters[m][s]['formal']['updates'] for m in METHODS) for s in STAGES},
            operation_stats=copy.deepcopy(self.operation_stats),historical_cost_imported=False)
        return result


@contextmanager
def measured(budget,operation,kind='formal',category='verification'):
    if kind not in ('formal','synthetic'):raise ValueError('Explicit computation kind required')
    budget.check();start=time.monotonic();before=budget.seconds
    try:yield
    finally:
        elapsed=time.monotonic()-start
        budget.charge(max(0.,elapsed-(budget.seconds-before)),'verification' if kind=='synthetic' else category)
        cell=budget.operation_stats.setdefault(operation,{}).setdefault(kind,{'calls':0,'seconds':0.})
        cell['calls']+=1;cell['seconds']+=elapsed
    budget.check()


class BlockMaternKernel(gpytorch.kernels.Kernel):
    """Matérn of combined squared block distances, never sum/product of kernels."""
    has_lengthscale=False
    def __init__(self):
        super().__init__()
        self.register_parameter('raw_architecture_lengthscale',nn.Parameter(torch.zeros(1,1,dtype=torch.float64)))
        self.register_parameter('raw_hp_lengthscale',nn.Parameter(torch.zeros(1,1,dtype=torch.float64)))
        self.register_constraint('raw_architecture_lengthscale',GreaterThan(1e-4))
        self.register_constraint('raw_hp_lengthscale',GreaterThan(1e-4))
        self.internal=gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=None,lengthscale_constraint=GreaterThan(1e-4)).double()
        self.internal.lengthscale=1.
        self.internal.raw_lengthscale.requires_grad_(False)
        self.architecture_lengthscale=1.;self.hp_lengthscale=1.

    @property
    def architecture_lengthscale(self):return self.raw_architecture_lengthscale_constraint.transform(self.raw_architecture_lengthscale)
    @architecture_lengthscale.setter
    def architecture_lengthscale(self,value):
        value=torch.as_tensor(value,dtype=torch.float64,device=self.raw_architecture_lengthscale.device)
        self.initialize(raw_architecture_lengthscale=self.raw_architecture_lengthscale_constraint.inverse_transform(value))
    @property
    def hp_lengthscale(self):return self.raw_hp_lengthscale_constraint.transform(self.raw_hp_lengthscale)
    @hp_lengthscale.setter
    def hp_lengthscale(self,value):
        value=torch.as_tensor(value,dtype=torch.float64,device=self.raw_hp_lengthscale.device)
        self.initialize(raw_hp_lengthscale=self.raw_hp_lengthscale_constraint.inverse_transform(value))

    def forward(self,x1,x2,diag=False,**params):
        if x1.shape[-1]!=16 or x2.shape[-1]!=16:raise ValueError('Block kernel requires z12+HP4')
        scale=torch.cat((self.architecture_lengthscale.reshape(1).expand(12),self.hp_lengthscale.reshape(1).expand(4)))
        return self.internal(x1/scale,x2/scale,diag=diag,**params)


class ExactBlockGP(gpytorch.models.ExactGP):
    def __init__(self,x,y,known,method):
        with numerical_settings():
            likelihood=gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=known.clone(),learn_additional_noise=True,
                noise_constraint=GreaterThan(1e-6))
        super().__init__(x,y,likelihood)
        self.mean_module=gpytorch.means.ConstantMean()
        base=BlockMaternKernel() if method=='RAW16_BLOCK2' else gpytorch.kernels.MaternKernel(
            nu=2.5,ard_num_dims=None,lengthscale_constraint=GreaterThan(1e-4))
        self.covar_module=gpytorch.kernels.ScaleKernel(base,outputscale_constraint=GreaterThan(1e-8))
        self.double();self.mean_module.constant=0.;self.covar_module.outputscale=1.;self.likelihood.second_noise=.01
        if method!='RAW16_BLOCK2':self.covar_module.base_kernel.lengthscale=1.
        inherited._isolate_transforms(self)
        if list(self.named_priors()) or list(self.named_added_loss_terms()):raise ValueError('Only pure data ExactMLL allowed')
    def forward(self,x):return gpytorch.distributions.MultivariateNormal(self.mean_module(x),self.covar_module(x))


class BlockModel:
    def __init__(self,method,x16,y,known):
        if method not in METHODS:raise ValueError('Unknown block experiment method')
        self.method=method
        self.raw_x=x16.detach().clone().to(device='cpu',dtype=torch.float64)
        self.y=y.detach().clone().to(device='cpu',dtype=torch.float64)
        self.known=known.detach().clone().to(device='cpu',dtype=torch.float64)
        if self.raw_x.shape!=(len(self.y),16) or self.y.ndim!=1 or self.known.shape!=self.y.shape:
            raise ValueError('Aligned raw standardized x16/y/known tensors required')
        if not all(bool(torch.isfinite(v).all()) for v in (self.raw_x,self.y,self.known)) or bool((self.known<0).any()):
            raise ValueError('Invalid finite training data/variance')
        self.x=self.raw_x[:,-4:].clone() if method=='HP4_ISO' else self.raw_x.clone()
        self.gp=ExactBlockGP(self.x,self.y,self.known,method)
        self.mll=gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood,self.gp)
        self.total_steps=0
        self.initial_input_hash=self.input_hash()
        expected=5 if method=='RAW16_BLOCK2' else 4
        if sum(p.numel() for p in self.gp.parameters() if p.requires_grad)!=expected:raise ValueError('Unexpected learnable scalar count')
        if any(p.numel()!=1 for p in self.gp.parameters() if p.requires_grad):raise ValueError('Only actual scalar GP parameters authorized')

    def configuration(self):
        return {'method':self.method,'input_dimension':4 if self.method=='HP4_ISO' else 16,'mean':'trainable_constant',
            'kernel':'Scale_Matern52_of_combined_distance','architecture_dimensions':12,'hp_dimensions':4,
            'block_dimension_normalization':False,'lengthscale_parameters':2 if self.method=='RAW16_BLOCK2' else 1,
            'internal_matern_lengthscale':1. if self.method=='RAW16_BLOCK2' else None,'neural_parameters':0}
    def input_hash(self):return tensor_hash({'x16':self.raw_x,'gp_input':self.x,'y':self.y,'known':self.known})
    def fingerprint(self):return hashlib.sha256((tensor_hash({'gp.'+k:v for k,v in self.gp.state_dict().items()})+'|'+self.input_hash()+'|'+digest(self.configuration())).encode()).hexdigest()
    def parameter_names(self):return [name for name,p in self.gp.named_parameters() if p.requires_grad]
    def parameter_spec(self):return [{'name':name,'shape':list(p.shape),'numel':p.numel()} for name,p in self.gp.named_parameters() if p.requires_grad]
    def hyperparameters(self):
        base=self.gp.covar_module.base_kernel
        if self.method=='RAW16_BLOCK2':la,lh=float(base.architecture_lengthscale.detach()),float(base.hp_lengthscale.detach())
        elif self.method=='RAW16_ISO':la=lh=float(base.lengthscale.detach())
        else:la=None;lh=float(base.lengthscale.detach())
        return {'constant_mean':float(self.gp.mean_module.constant.detach()),'lengthscale_architecture':la,'lengthscale_hp':lh,
            'lengthscale':lh if self.method!='RAW16_BLOCK2' else None,'outputscale':float(self.gp.covar_module.outputscale.detach()),
            'extra_noise_variance_standardized':float(self.gp.likelihood.second_noise.detach()),
            'lengthscale_ratio_A_over_H':None if la is None else la/lh,
            'near_lower_bounds':{'architecture_lengthscale':None if la is None else la<=1.01e-4,'hp_lengthscale':lh<=1.01e-4,
                'outputscale':float(self.gp.covar_module.outputscale.detach())<=1.01e-8,
                'extra_noise':float(self.gp.likelihood.second_noise.detach())<=1.01e-6}}
    def query_input(self,q16):
        if q16.ndim!=2 or q16.shape[1]!=16 or q16.dtype!=torch.float64 or q16.requires_grad or not bool(torch.isfinite(q16).all()):
            raise ValueError('Only finite detached standardized raw16 query tensors allowed')
        return q16[:,-4:] if self.method=='HP4_ISO' else q16
    def nll(self):
        self.gp.train();self.gp.likelihood.train()
        with numerical_settings():return -self.mll(self.gp(self.x),self.y)
    def kernel(self,left=None,right=None):
        a=self.x if left is None else self.query_input(left)
        b=a if right is None else self.query_input(right)
        with numerical_settings():return self.gp.covar_module(a,b).to_dense()
    def predict(self,q16):
        q=self.query_input(q16)
        with torch.no_grad(),numerical_settings():
            self.gp.eval();self.gp.likelihood.eval();posterior=self.gp(q)
            mu=posterior.mean.detach().clone();cov=posterior.lazy_covariance_matrix.to_dense().detach().clone()
        return mu,cov
    def snapshot(self):
        return {'method':self.method,'configuration':self.configuration(),'gp':copy.deepcopy(self.gp.state_dict()),
            'x_raw_standardized':self.raw_x.clone(),'gp_input':self.x.clone(),'y_standardized':self.y.clone(),
            'known_variance_standardized':self.known.clone(),'input_hash':self.input_hash(),'fingerprint':self.fingerprint(),
            'total_steps':self.total_steps,'parameter_spec':self.parameter_spec(),'parameter_names':self.parameter_names(),'rng':rng_state()}
    def clone(self):
        incoming=rng_state()
        try:
            other=BlockModel(self.method,self.raw_x,self.y,self.known)
            other.gp.load_state_dict(copy.deepcopy(self.gp.state_dict()),strict=True);other.total_steps=self.total_steps
            if other.fingerprint()!=self.fingerprint():raise RuntimeError('Diagnostic clone state differs')
            return other
        finally:set_rng(incoming)


def prepare_inputs(training,query,scaler,budget,kind='formal'):
    with measured(budget,'prepare_inputs',kind):
        if set(training)!=TRAIN_KEYS or set(query)!=QUERY_KEYS:raise ValueError('Strict training/query whitelist violation')
        ids,qids=list(training['ids']),list(query['ids'])
        if len(ids)!=50 or len(qids) not in (34,35,36,38) or len(set(ids))!=50 or len(set(qids))!=len(qids):raise ValueError('Expected T50 and original Q sizes')
        if set(ids)&set(qids) or set(training['architectures'])&set(query['architectures']):raise ValueError('T/Q identity leakage')
        if len(training['architectures'])!=50 or len(query['architectures'])!=len(qids):raise ValueError('Architecture identity shape')
        if scaler['fit_ids']!=ids or scaler['std_correction']!=0 or scaler['frozen'] is not True or scaler['dtype']!='float64':raise ValueError('Training-only frozen scaler required')
        x=torch.tensor(training['x_standardized'],dtype=torch.float64);q=torch.tensor(query['x_standardized'],dtype=torch.float64)
        y=torch.tensor(training['y_standardized'],dtype=torch.float64);known=torch.tensor(training['known_variance_standardized'],dtype=torch.float64)
        if x.shape!=(50,16) or q.shape!=(len(qids),16) or y.shape!=(50,) or known.shape!=(50,):raise ValueError('Canonical16 projected input shape')
        if not all(bool(torch.isfinite(v).all()) for v in (x,q,y,known)) or bool((known<0).any()):raise ValueError('Finite input/noise domain')
        if not math.isfinite(scaler['y_mean']) or not math.isfinite(scaler['y_scale']) or scaler['y_scale']<1e-8:raise ValueError('Invalid y scaler')
        return {'x_train16':x,'x_query16':q,'y_std':y,'known_std':known,'scaler':copy.deepcopy(scaler),
            'ids':ids,'query_ids':qids,'architectures':list(training['architectures']),'query_architectures':list(query['architectures'])}


def create_model(method,prepared,budget,kind='formal'):
    with measured(budget,'model_'+method,kind):return BlockModel(method,prepared['x_train16'],prepared['y_std'],prepared['known_std'])


def learning_rate(stage,update):
    if stage not in STAGES or not 1<=update<=STEPS[stage]:raise ValueError('Invalid prescribed local update')
    return .01 if update<=(240 if stage=='INIT_GP' else 1000) else .001


class Session:
    def __init__(self,model,stage,budget,kind):
        if stage not in STAGES or kind not in ('formal','synthetic'):raise ValueError('Invalid stage/kind')
        if kind=='formal' and model.total_steps!=(0 if stage=='INIT_GP' else 300):raise ValueError('Formal phase must start at pure0 or own300 steps')
        self.model,self.stage,self.budget,self.kind=model,stage,budget,kind
        self.completed=0;self.planned_steps=STEPS[stage];self.start_total_steps=model.total_steps
        self.training_seconds=0.;self.last_update=None
        self.params=[p for p in model.gp.parameters() if p.requires_grad];self.names=model.parameter_names()
        self.optimizer=torch.optim.Adam([{'params':self.params,'lr':.01,'name':'gp'}],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    @property
    def lrs(self):return {'gp':float(self.optimizer.param_groups[0]['lr'])}
    def step(self,budget=None):
        if budget is not None and budget is not self.budget:raise ValueError('Training budget changed')
        b=self.budget;b.check_step(self.kind)
        if self.completed>=self.planned_steps:raise RuntimeError('Fixed block endpoint reached')
        start=time.monotonic();called=False;returned=False;updated=0
        cell=b.method_stage_counters[self.model.method][self.stage][self.kind]
        try:
            self.optimizer.param_groups[0]['lr']=learning_rate(self.stage,self.completed+1)
            self.optimizer.zero_grad(set_to_none=True);loss=self.model.nll()
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training NLL')
            loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(self.params,5.,error_if_nonfinite=True))
            if any(p.grad is None for p in self.params):raise RuntimeError('Trainable GP parameter lacks gradient')
            b.before_optimizer_call(self.kind,self.model.method,self.stage);called=True
            self.optimizer.step();returned=True;cell['successful_optimizer_calls']+=1
            self.completed+=1;self.model.total_steps+=1;updated=1
            self.last_update={'method':self.model.method,'stage':self.stage,'step':self.completed,'local_step':self.completed,
                'total_step':self.model.total_steps,'planned_steps':self.planned_steps,'nll_before_update':float(loss.detach()),
                'gradient_before_clip':norm,'learning_rates':self.lrs,'accounting_kind':self.kind}
        except Exception:
            b.failed_training_attempts+=1;cell['failed_training_attempts']+=1
            if called and not returned:
                b.failed_optimizer_calls+=1;b.failed_updates+=1;cell['failed_optimizer_calls']+=1;cell['failed_updates']+=1
            raise
        finally:
            elapsed=time.monotonic()-start;self.training_seconds+=elapsed
            b.step_charge(elapsed,self.kind,self.model.method,self.stage,updated)
        self.last_update.update(step_seconds=elapsed,block_training_seconds=self.training_seconds,
            formal_updates=b.formal_updates,synthetic_updates=b.synthetic_updates)
        b.check();return self.last_update
    def snapshot(self):
        result=self.model.snapshot()
        result.update(optimizer=copy.deepcopy(self.optimizer.state_dict()),ordered_parameter_names=self.names,
            scheduler={'stage':self.stage,'planned_steps':self.planned_steps,'completed':self.completed,
                'start_total_steps':self.start_total_steps,'total_steps':self.model.total_steps,
                'policy':'INIT240base60tenth' if self.stage=='INIT_GP' else 'FIT1000base500tenth'},
            accounting={'kind':self.kind,'block_training_seconds':self.training_seconds,'completed_updates':self.completed},
            last_update=copy.deepcopy(self.last_update),automatic_resume=False)
        return result


def session(model,stage,budget,kind='formal'):
    with measured(budget,'session_'+model.method+'_'+stage,kind):return Session(model,stage,budget,kind)


def charged_snapshot(obj,budget,kind='formal'):
    model=obj.model if isinstance(obj,Session) else obj;stage=obj.stage if isinstance(obj,Session) else 'INITIAL'
    with measured(budget,'snapshot_'+model.method+'_'+stage,kind):
        if model.input_hash()!=model.initial_input_hash:raise RuntimeError('Frozen GP training input changed')
        return obj.snapshot()


def _signature(model):
    return {'fingerprint':model.fingerprint(),'modes':[(n,m.training) for n,m in model.gp.named_modules()],
        'requires_grad':[p.requires_grad for p in model.gp.parameters()],
        'gradients':[None if p.grad is None else p.grad.detach().clone() for p in model.gp.parameters()],
        'inputs':[v.untyped_storage().data_ptr() for v in model.gp.train_inputs],
        'targets':model.gp.train_targets.untyped_storage().data_ptr(),'cache':id(model.gp.prediction_strategy)}


def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def diagnose(active,prepared,budget,kind='formal'):
    model=active.model
    with measured(budget,'diagnostic_'+model.method,kind,'prediction'):
        incoming=rng_state();signature=_signature(model)
        try:
            isolated=model.clone()
            with torch.no_grad():
                nll=isolated.nll();mu,cov=isolated.predict(prepared['x_query16']);kernel=isolated.kernel()
                if not all(bool(torch.isfinite(v).all()) for v in (nll,mu,cov,kernel)):raise FloatingPointError('Nonfinite diagnostic')
                var=cov.diagonal().clone();asym=float((cov-cov.T).abs().max());mineig=float(torch.linalg.eigvalsh((cov+cov.T)/2).min())
                if asym>1e-10 or mineig < -1e-8 or bool((var < -1e-8).any()):raise FloatingPointError('Invalid raw latent covariance')
                sc=prepared['scaler'];yc,ys=sc['y_mean'],sc['y_scale'];raw=yc+ys*mu;sd=ys*var.clamp_min(0).sqrt()
                ids=prepared['query_ids'];order=sorted(range(len(ids)),key=lambda i:(-float(raw[i]),ids[i]));hp=isolated.hyperparameters()
                counts={float(v):int((raw==v).sum()) for v in raw}
                result={'method':model.method,'stage':active.stage,'step':active.completed,'local_step':active.completed,
                    'total_step':model.total_steps,'state_hash':signature['fingerprint'],'nll_after_update':float(nll),
                    'nll_units':'negative exact marginal log likelihood per training observation, standardized y',
                    'query_ids':ids,'query_mean_standardized':mu.tolist(),'query_mean_raw':raw.tolist(),
                    'query_latent_covariance_standardized':cov.tolist(),'query_latent_covariance_raw':(cov*ys**2).tolist(),
                    'query_latent_variance_standardized':var.tolist(),'query_latent_variance_raw':(var*ys**2).tolist(),
                    'query_latent_sd_raw':sd.tolist(),'training_kernel':kernel.tolist(),'gp_parameters':hp,
                    'extra_noise_variance_standardized':hp['extra_noise_variance_standardized'],
                    'model_variance_raw':((var.clamp_min(0)+hp['extra_noise_variance_standardized'])*ys**2).tolist(),
                    'negative_roundoff_variance_count':int((var<0).sum()),'latent_covariance_symmetry_max_abs':asym,
                    'latent_covariance_min_eigenvalue':mineig,'y_center':yc,'y_scale':ys,'learning_rates':active.lrs,
                    'top5_ids':[ids[i] for i in order[:5]],'ranking_ids':[ids[i] for i in order],
                    'query_exact_mean_tie_groups':sum(c>1 for c in counts.values()),
                    'query_exact_mean_tie_candidates':sum(c for c in counts.values() if c>1),
                    'top5_boundary_exact_tie':bool(raw[order[4]]==raw[order[5]]) if len(ids)>5 else False,
                    'top5_boundary_gap_raw':float(raw[order[4]]-raw[order[5]]) if len(ids)>5 else None,
                    'query_labels_or_SE_present':False,'diagnostic_isolated':True,'latent_variance_clamp_for_sqrt_only':True}
            if not exact(signature,_signature(model)):raise RuntimeError('Diagnostic mutated live modes, gradients, parameters or cache')
            return result
        finally:set_rng(incoming)


def as_fit_start(record):
    if (record['stage'],record['local_step'],record['total_step'])!=('INIT_GP',300,300):raise ValueError('Only own INIT300 can be reused as FIT0')
    result=copy.deepcopy(record);result.update(stage='FIT_GP',step=0,local_step=0,learning_rates={'gp':.01},
        reused_state_diagnostic={'stage':'INIT_GP','local_step':300,'total_step':300,'no_new_model_call':True})
    return result


def stability_at(records):
    lookup={r['local_step']:r for r in records if r['stage']=='FIT_GP'}
    if any(s not in lookup for s in (1200,1300,1400,1500)):raise ValueError('All fixed FIT stability windows required')
    windows=[]
    for a,b in ((1200,1300),(1300,1400),(1400,1500)):
        x,y=lookup[a],lookup[b]
        if x['query_ids']!=y['query_ids'] or x['method']!=y['method']:raise ValueError('Stability identity mismatch')
        ka,kb=np.asarray(x['training_kernel']),np.asarray(y['training_kernel'])
        w={'from_step':a,'to_step':b,
            'mll_relative_change':abs(y['nll_after_update']-x['nll_after_update'])/max(1.,abs(x['nll_after_update'])),
            'query_mean_max_abs_change_raw':max(abs(v-u) for u,v in zip(x['query_mean_raw'],y['query_mean_raw'])),
            'query_latent_sd_max_abs_change_raw':max(abs(v-u) for u,v in zip(x['query_latent_sd_raw'],y['query_latent_sd_raw'])),
            'training_kernel_relative_change':float(np.linalg.norm(kb-ka)/max(float(np.linalg.norm(ka)),1e-12)),
            'top5_stable':set(x['top5_ids'])==set(y['top5_ids']),'top5_order_stable':x['top5_ids']==y['top5_ids']}
        w['criteria']={'mll_stable':w['mll_relative_change']<=1e-4,'mean_stable':w['query_mean_max_abs_change_raw']<=.001,
            'latent_sd_stable':w['query_latent_sd_max_abs_change_raw']<=.001,'kernel_stable':w['training_kernel_relative_change']<=.01}
        w['numeric_predictive_stable']=all(w['criteria'].values());windows.append(w)
    return {'numeric_predictive_stable':all(w['numeric_predictive_stable'] for w in windows),
        'top5_stable':all(w['top5_stable'] for w in windows),'top5_order_stable':all(w['top5_order_stable'] for w in windows),
        'criteria':{k:all(w['criteria'][k] for w in windows) for k in windows[0]['criteria']},'windows':windows}


def distance_diagnostics(model,prepared,budget,kind='formal'):
    with measured(budget,'distance_'+model.method,kind,'prediction'):
        x=model.raw_x;hp=model.hyperparameters();n=len(x)
        ix=torch.triu_indices(n,n,offset=1);za=x[ix[0],:12]-x[ix[1],:12];pa=x[ix[0],12:]-x[ix[1],12:]
        if model.method=='HP4_ISO':
            def duplicates(data,architectures):
                groups={}
                for i,row in enumerate(data.tolist()):groups.setdefault(tuple(row),[]).append(i)
                repeated=[v for v in groups.values() if len(v)>1]
                return {'duplicate_vector_groups':len(repeated),'duplicate_candidates':sum(map(len,repeated)),
                    'equal_HP_pairs':sum(len(v)*(len(v)-1)//2 for v in repeated),
                    'different_architecture_equal_HP_pairs':sum(architectures[i]!=architectures[j] for v in repeated for k,i in enumerate(v) for j in v[k+1:])}
            return {'method':model.method,'total_step':model.total_steps,'training':duplicates(x[:,12:],prepared['architectures']),
                'query':duplicates(prepared['x_query16'][:,12:],prepared['query_architectures']),
                'gp_parameters':hp,'raw_input_space_only':True}
        da=za.square().sum(1)/hp['lengthscale_architecture']**2;dh=pa.square().sum(1)/hp['lengthscale_hp']**2
        total=da+dh;nonzero=total>0
        def describe(v):
            if not v.numel():return None
            return {'mean':float(v.mean()),'median':float(torch.quantile(v,.5)),
                'q10':float(torch.quantile(v,.1)),'q90':float(torch.quantile(v,.9))}
        shares=da[nonzero]/total[nonzero]
        return {'method':model.method,'total_step':model.total_steps,'pair_count':len(total),'zero_distance_pairs':int((~nonzero).sum()),
            'architecture_squared_distance':describe(da),'hp_squared_distance':describe(dh),
            'share_architecture':describe(shares),'share_hp':describe(1-shares),'gp_parameters':hp,
            'block_dimension_normalization':False,'raw_input_space_only':True,'causal_importance_claim':False}
