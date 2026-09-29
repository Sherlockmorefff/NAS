"""OOF orchestration adapters; unchanged frozen residual/readout mathematics.

Only U labels enter this module. V accepts an exact public-only whitelist.
All numerical helpers are charged; optimizer invocation and success are separate.
There is no production restore/resume entry point or historical model loading.
"""
from __future__ import annotations

import copy
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import gpytorch

import residual_core as residual
import readout_core as readout
from readout_core import BudgetExceeded, SyntheticStepLimit, cpu_setup, tensor_hash
from readout_diagnostics import diagnose, stability_at, DIAGNOSTIC_STEPS

CLASSES=('COMMON_MSE','COMMON_GP','GP_CONST','GP_TREND')
PLANNED_STEPS={'COMMON_MSE':1000,'COMMON_GP':300,'GP_CONST':1500,'GP_TREND':1500}
TRAIN_KEYS={'ids','architectures','x_raw','y','measurement_se'}
QUERY_KEYS={'ids','architectures','x_raw'}
COUNTER_KEYS=('optimizer_calls','successful_optimizer_calls','updates','failed_optimizer_calls',
              'failed_updates','failed_training_attempts')


def budget_limit(root):
    value=readout.budget_limit(root)
    if value!=1500:raise ValueError('OOF protocol requires exactly 1500 model seconds')
    return value


class Budget(readout.Budget):
    """New run starts at zero; four stage counters accompany every actual call."""
    def __init__(self,path=None,limit=None):
        if limit!=1500:raise ValueError('Explicit OOF frozen budget of 1500 required')
        old=json.loads(Path(path).read_text()) if path and Path(path).exists() else {}
        if old and old.get('accounting_version')!='oof_four_stage_v1':
            raise ValueError('Cannot import a historical ledger into the new OOF budget')
        super().__init__(path,limit)
        self.class_counters=copy.deepcopy(old.get('class_counters',{
            stage:{kind:{name:0 for name in COUNTER_KEYS} for kind in ('formal','synthetic')} for stage in CLASSES}))
        self.operation_stats=copy.deepcopy(old.get('operation_stats',{}))
        self._validate_counts()

    def _validate_counts(self):
        if set(self.class_counters)!=set(CLASSES):raise ValueError('OOF class ledger differs')
        for kind in ('formal','synthetic'):
            for key,attribute in [('optimizer_calls',kind+'_optimizer_calls'),('updates',kind+'_updates')]:
                if sum(self.class_counters[s][kind][key] for s in CLASSES)!=getattr(self,attribute):
                    raise ValueError('OOF class/global ledger mismatch: '+attribute)

    def before_optimizer_call(self,kind,stage=None):
        if stage not in CLASSES:raise ValueError('Every optimizer call requires its OOF stage')
        self.check_step(kind)
        setattr(self,kind+'_optimizer_calls',getattr(self,kind+'_optimizer_calls')+1)
        self.class_counters[stage][kind]['optimizer_calls']+=1
        if kind=='synthetic':self.persist()

    def charge(self,seconds,kind='prediction',updates=0):
        if updates:raise ValueError('Optimizer updates must use the classified training adapter')
        super().charge(seconds,kind,updates=0)

    def _step_charge(self,seconds,kind,stage,updates):
        super().charge(seconds,'training' if kind=='formal' else 'verification',updates=updates)
        self.class_counters[stage][kind]['updates']+=int(updates)

    def _failure(self,name,delta,kind,stage):
        if delta<0:raise ValueError('Failure counters cannot decrease')
        setattr(self,name,getattr(self,name)+delta)
        self.class_counters[stage][kind][name]+=delta

    @property
    def formal_steps_by_class(self):return {s:self.class_counters[s]['formal']['updates'] for s in CLASSES}

    def as_dict(self):
        value=super().as_dict()
        value.update(accounting_version='oof_four_stage_v1',class_counters=copy.deepcopy(self.class_counters),
                     formal_steps_by_class=self.formal_steps_by_class,operation_stats=copy.deepcopy(self.operation_stats),
                     historical_cost_imported=False)
        return value


@contextmanager
def measured(budget,operation,kind='formal',category='verification'):
    """Charge a helper exactly once, including failures; nested charges excluded."""
    if kind not in ('formal','synthetic'):raise ValueError('Explicit work kind required')
    budget.check();start=time.monotonic();before=budget.seconds
    try:yield
    finally:
        elapsed=time.monotonic()-start
        budget.charge(max(0.,elapsed-(budget.seconds-before)), 'verification' if kind=='synthetic' else category)
        cell=budget.operation_stats.setdefault(operation,{}).setdefault(kind,{'calls':0,'seconds':0.})
        cell['calls']+=1;cell['seconds']+=elapsed
    budget.check()


class _StepBudget:
    """Bridge the two frozen training APIs without changing their mathematics."""
    def __init__(self,budget,stage,kind):self.budget,self.stage,self.kind=budget,stage,kind
    @property
    def seconds(self):return self.budget.seconds
    @property
    def updates(self):return self.budget.formal_updates
    @property
    def formal_updates(self):return self.budget.formal_updates
    @property
    def synthetic_updates(self):return self.budget.synthetic_updates
    def check(self):self.budget.check()
    def check_step(self,kind):
        if kind!=self.kind:raise ValueError('Readout accounting kind changed')
        self.budget.check_step(kind)
    def before_optimizer_call(self,kind):
        if kind!=self.kind:raise ValueError('Optimizer accounting kind changed')
        self.budget.before_optimizer_call(kind,self.stage)
    def charge(self,seconds,kind='training',updates=0):
        if kind not in ('training','verification'):raise ValueError('Unexpected step timing category')
        self.budget._step_charge(seconds,self.kind,self.stage,updates)


def _failure_property(name):
    return property(lambda self:getattr(self.budget,name),
        lambda self,value:self.budget._failure(name,int(value)-getattr(self.budget,name),self.kind,self.stage))


for _name in ('failed_training_attempts','failed_optimizer_calls','failed_updates'):
    setattr(_StepBudget,_name,_failure_property(_name))


class TrainingSession:
    """One new optimizer for one prescribed block; no automatic restore."""
    def __init__(self,inner,stage,budget,kind):
        if stage not in CLASSES or kind not in ('formal','synthetic'):raise ValueError('Invalid training identity')
        self.inner,self.stage,self.budget,self.kind=inner,stage,budget,kind
        self._view=_StepBudget(budget,stage,kind)
        self.training_seconds=0.
        original=inner.optimizer.step
        def counted_step(*args,**kwargs):
            if stage.startswith('COMMON_'):self._view.before_optimizer_call(kind)
            try:result=original(*args,**kwargs)
            except Exception:
                # Readout Session already accounts a failed call in its except.
                if stage.startswith('COMMON_'):
                    self._view.failed_optimizer_calls+=1;self._view.failed_updates+=1
                raise
            budget.class_counters[stage][kind]['successful_optimizer_calls']+=1
            return result
        inner.optimizer.step=counted_step

    @property
    def model(self):return self.inner.model
    @property
    def completed(self):return self.inner.completed
    @property
    def planned_steps(self):return self.inner.planned_steps
    @property
    def optimizer(self):return self.inner.optimizer
    @property
    def lrs(self):return self.inner.lrs
    @property
    def last_update(self):return self.inner.last_update

    def step(self,budget=None):
        if budget is not None and budget is not self.budget:raise ValueError('Training ledger changed')
        self.budget.check_step(self.kind)
        before=self.budget.seconds;start=time.monotonic()
        try:
            result=self.inner.step(self._view) if self.stage.startswith('COMMON_') else self.inner.step(self._view,self.kind)
        finally:
            elapsed=time.monotonic()-start
            self.training_seconds+=elapsed
            self.budget.charge(max(0.,elapsed-(self.budget.seconds-before)),
                               'training' if self.kind=='formal' else 'verification')
        result.update(oof_stage=self.stage,accounting_kind=self.kind,
                      adapter_step_seconds=elapsed,adapter_block_seconds=self.training_seconds)
        self.budget.check()
        return result

    def snapshot(self):
        state=self.inner.snapshot()
        state['oof_accounting']={'stage':self.stage,'kind':self.kind,'completed_updates':self.completed,
            'optimizer_calls':self.budget.class_counters[self.stage][self.kind]['optimizer_calls'],
            'adapter_block_seconds':self.training_seconds,'class_counters_at_save':copy.deepcopy(self.budget.class_counters),
            'automatic_resume':False}
        return state


def _public_validate(training,query):
    if set(training)!=TRAIN_KEYS or set(query)!=QUERY_KEYS:raise ValueError('U/V exact whitelist violated')
    n,q=len(training['ids']),len(query['ids'])
    if n not in (33,34) or q not in (16,17):raise ValueError('OOF expects U33/34 and V16/17')
    if len(set(training['ids']))!=n or len(set(query['ids']))!=q or set(training['ids'])&set(query['ids']):
        raise ValueError('OOF duplicate/overlapping candidate IDs')
    if len(training['architectures'])!=n or len(query['architectures'])!=q or set(training['architectures'])&set(query['architectures']):
        raise ValueError('OOF architectures overlap or are misaligned')
    if not all(isinstance(s,str) and s for s in training['ids']+query['ids']+training['architectures']+query['architectures']):
        raise ValueError('Invalid public identity')


def prepare_common(training,query,seed,budget,kind='formal'):
    with measured(budget,'prepare_common',kind):
        _public_validate(training,query)
        x=torch.tensor(training['x_raw'],dtype=torch.float64);q=torch.tensor(query['x_raw'],dtype=torch.float64)
        y=torch.tensor(training['y'],dtype=torch.float64);se=torch.tensor(training['measurement_se'],dtype=torch.float64)
        if x.shape!=(len(training['ids']),16) or q.shape!=(len(query['ids']),16) or y.shape!=(len(training['ids']),) or y.shape!=se.shape:
            raise ValueError('OOF tensor shapes')
        if not all(bool(torch.isfinite(v).all()) for v in (x,q,y,se)) or bool((se<0).any()):raise ValueError('OOF finite/noise domain')
        xm=x.mean(0);xs=x.std(0,unbiased=False).clamp_min(1e-6)
        yc=y.mean();ys=y.std(unbiased=False).clamp_min(1e-8)
        xx=(x-xm)/xs;qq=(q-xm)/xs;yy=(y-yc)/ys;known=se.square()/ys.square()
        scaler={'x_mean':xm.tolist(),'x_scale':xs.tolist(),'y_mean':float(yc),'y_scale':float(ys),
                'fit_ids':list(training['ids']),'std_correction':0,'frozen':True,'dtype':'float64'}
        model=residual.ResidualBranch(xx,yy,known,int(seed),'bounded')
        return {'model':model,'scaler':scaler,'x_train':xx,'x_query':qq,'y_std':yy,'known_std':known,
                'ids':list(training['ids']),'query_ids':list(query['ids']),
                'initial_mapper_hash':tensor_hash(model.mapper.state_dict()),'model_seed':int(seed)}


def common_session(model,stage,block,budget,kind='formal'):
    if stage not in ('COMMON_MSE','COMMON_GP'):raise ValueError('Expected one prescribed common stage')
    with measured(budget,'session_'+stage,kind):
        if stage=='COMMON_GP':
            model.synchronize_history()
            # MSE never optimizes GP parameters. No inherited fitted GP is legal.
            initial=[float(model.gp.mean_module.constant),float(model.gp.covar_module.base_kernel.lengthscale),
                     float(model.gp.covar_module.outputscale),float(model.gp.likelihood.second_noise)]
            if any(abs(a-b)>1e-12 for a,b in zip(initial,(0.,1.,1.,.01))):raise ValueError('COMMON_GP must start from fixed fresh GP parameters')
        inner=residual.BlockSession(model,'regression' if stage=='COMMON_MSE' else 'gp',PLANNED_STEPS[stage],block)
        return TrainingSession(inner,stage,budget,kind)


def freeze_features(model,x_query,budget,kind='formal'):
    with measured(budget,'freeze_features',kind):
        for p in model.mapper.parameters():p.requires_grad_(False);p.grad=None
        model.mapper.eval();model.synchronize_history()
        with torch.no_grad():ht,st=model.encode(model.x);hq,sq=model.encode(x_query)
        if not torch.equal(ht,model.gp.train_inputs[0]):raise RuntimeError('MSE GP training input synchronization failed')
        return {'h_train':ht.detach().clone(),'s_train':st.detach().clone(),
                'h_query':hq.detach().clone(),'s_query':sq.detach().clone(),'y_std':model.y.detach().clone(),
                'known_std':model.known.detach().clone(),'common_gp_state':copy.deepcopy(model.gp.state_dict()),
                'source_fingerprint':model.fingerprint(),'mapper_hash':tensor_hash(model.mapper.state_dict())}


def readout_session(method,frozen,budget,kind='formal'):
    if method not in ('GP_CONST','GP_TREND'):raise ValueError('OOF has exactly CONST/TREND readouts')
    with measured(budget,'session_'+method,kind):
        model=readout.ReadoutModel(method,frozen['h_train'],frozen['s_train'],frozen['y_std'],frozen['known_std'],
                                  frozen['common_gp_state'],frozen['source_fingerprint'])
        return TrainingSession(readout.Session(model),method,budget,kind)


def _residual_surface(model):
    return {'hash':model.fingerprint(),'modes':[(name,m.training) for root in (model.mapper,model.gp) for name,m in root.named_modules()],
            'requires_grad':[p.requires_grad for root in (model.mapper,model.gp) for p in root.parameters()],
            'gradients':[None if p.grad is None else p.grad.detach().clone() for root in (model.mapper,model.gp) for p in root.parameters()],
            'inputs':[v.untyped_storage().data_ptr() for v in model.gp.train_inputs],
            'cache':id(model.gp.prediction_strategy)}


def common_diagnostic(session,budget,kind='formal'):
    stage=session.stage
    if stage not in ('COMMON_MSE','COMMON_GP'):raise ValueError('Common diagnostic stage')
    with measured(budget,'diagnostic_'+stage,kind,'prediction'):
        model=session.model;incoming=residual.rng_state();surface=_residual_surface(model)
        try:
            isolated=model.clone()
            current_gradient=None
            if stage=='COMMON_MSE':
                isolated.mapper.zero_grad(set_to_none=True)
                loss,components=isolated.objective('regression')
                loss.backward()
                current_gradient=residual._grad_norm(isolated.mapper.parameters())
                if not math.isfinite(current_gradient):raise FloatingPointError('Nonfinite isolated MSE diagnostic gradient')
            else:
                with torch.no_grad():loss,components=isolated.objective('gp')
            with torch.no_grad():
                spectral=residual.spectral_stats(isolated)
                params={'constant_mean':float(isolated.gp.mean_module.constant),
                        'lengthscale':float(isolated.gp.covar_module.base_kernel.lengthscale),
                        'outputscale':float(isolated.gp.covar_module.outputscale),
                        'extra_noise_variance_standardized':float(isolated.gp.likelihood.second_noise)}
            if not math.isfinite(float(loss)):raise FloatingPointError('Nonfinite common diagnostic objective')
            if not residual.exact(surface,_residual_surface(model)):raise RuntimeError('Common diagnostic changed live surface')
            return {'stage':stage,'step':session.completed,'planned_steps':session.planned_steps,'loss_after_update':float(loss),
                    **components,'spectral':spectral,'gp_parameters':params,'learning_rates':session.lrs,
                    'gradient_after_update_before_clip':current_gradient,
                    'gradient_scope':'all mapper and head parameters on isolated clone' if stage=='COMMON_MSE' else None,
                    'last_update':copy.deepcopy(session.last_update),'state_hash':surface['hash'],
                    'diagnostic_model_isolated':True,'query_labels_or_SE_present':False}
        finally:residual.set_rng(incoming)


def head_predictions(model,x_query,scaler,budget,kind='formal'):
    with measured(budget,'head_predictions',kind,'prediction'):
        before=tensor_hash(model.mapper.state_dict())
        with torch.no_grad():_,u=model.encode(model.x);_,v=model.encode(x_query)
        yc,ys=scaler['y_mean'],scaler['y_scale']
        if before!=tensor_hash(model.mapper.state_dict()):raise RuntimeError('HEAD prediction mutated mapping')
        return {'u_mean_standardized':u.tolist(),'v_mean_standardized':v.tolist(),
                'u_mean_raw':(yc+ys*u).tolist(),'v_mean_raw':(yc+ys*v).tolist(),
                'mapper_hash':before,'query_labels_or_SE_present':False}


def readout_diagnostic(session,frozen,query_ids,scaler,step,budget,kind='formal'):
    with measured(budget,'diagnostic_'+session.stage,kind,'prediction'):
        return diagnose(session.model,frozen['h_query'],frozen['s_query'],query_ids,
                        scaler['y_mean'],scaler['y_scale'],step)


def charged_snapshot(obj,budget,kind='formal'):
    stage=getattr(obj,'stage','INITIAL')
    with measured(budget,'snapshot_'+stage,kind):
        model=getattr(obj,'model',obj)
        if isinstance(model,readout.ReadoutModel) and model.frozen_hash()!=model.initial_frozen_hash:
            raise RuntimeError('Readout changed frozen h/s/training labels')
        result=obj.snapshot()
        if hasattr(model,'mapper'):
            result['mapper_hash']=tensor_hash(result['mapper'])
            result['spectral']=residual.spectral_stats(model)
        return result


def assert_readout_isolation(sessions,budget,kind='formal'):
    with measured(budget,'readout_isolation',kind):
        if {s.stage for s in sessions}!={'GP_CONST','GP_TREND'}:raise ValueError('Expected two readouts')
        a,b=[s.model for s in sessions]
        if a.source_fingerprint!=b.source_fingerprint:raise RuntimeError('Readout source differs')
        for key in a._common_state:
            if not torch.equal(a._common_state[key],b._common_state[key]):raise RuntimeError('Readout common state differs')
        if {id(m) for m in a.gp.modules()}&{id(m) for m in b.gp.modules()}:raise RuntimeError('Shared mutable GP modules')
        def ptrs(m):return {v.untyped_storage().data_ptr() for v in list(m.gp.state_dict().values())+[m.h_train,m.s_train,m.y,m.known] if v.numel()}
        if ptrs(a)&ptrs(b):raise RuntimeError('Readout tensor storage overlaps')
        return {'pass':True,'shared_initial_common_state':True,'mutable_storage_overlap':False,
                'source_fingerprint':a.source_fingerprint,'learned_parameter_names':{s.stage:s.model.parameter_names() for s in sessions}}
