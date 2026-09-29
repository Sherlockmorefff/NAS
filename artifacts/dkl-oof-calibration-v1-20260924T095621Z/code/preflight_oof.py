"""Single fixed synthetic preflight: 260 timing calls + 16 isolation calls.

Root alone executes this file. No historical model is restored and no real
training/query source is opened. Original mathematical tests are reused only
through unchanged-source identity recorded by the root's source audit.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

import residual_core as residual
import readout_core as readout
from oof_core import (Budget,budget_limit,cpu_setup,CLASSES,PLANNED_STEPS,prepare_common,
    common_session,freeze_features,readout_session,common_diagnostic,head_predictions,
    readout_diagnostic,charged_snapshot,assert_readout_isolation,measured,tensor_hash)

FORMAL_UPDATES={'COMMON_MSE':90000,'COMMON_GP':27000,'GP_CONST':135000,'GP_TREND':135000}
SETUP_COUNTS={'formal_runtime_setup':1,'prepare_common':90,'session_COMMON_MSE':90,'session_COMMON_GP':90,
    'freeze_features':180,'session_GP_CONST':90,'session_GP_TREND':90,
    'snapshot_INITIAL':90,'snapshot_COMMON_MSE':90,'snapshot_COMMON_GP':90,
    'snapshot_GP_CONST':90,'snapshot_GP_TREND':90}
PREDICTION_COUNTS={'diagnostic_COMMON_MSE':360,'diagnostic_COMMON_GP':270,
    'head_predictions':90,'diagnostic_GP_CONST':540,'diagnostic_GP_TREND':540}


def synthetic_fixture():
    generator=torch.Generator(device='cpu').manual_seed(592024)
    x=torch.randn((34,16),generator=generator,dtype=torch.float64)
    q=torch.randn((17,16),generator=generator,dtype=torch.float64)
    y=.7+.035*x[:,0]-.015*x[:,2]+.008*torch.sin(x[:,4])
    se=.001+.002*torch.rand((34,),generator=generator,dtype=torch.float64)
    train={'ids':['synthetic_U_%02d'%i for i in range(34)],'architectures':['U_arch_%02d'%i for i in range(34)],
           'x_raw':x.tolist(),'y':y.tolist(),'measurement_se':se.tolist()}
    query={'ids':['synthetic_V_%02d'%i for i in range(17)],'architectures':['V_arch_%02d'%i for i in range(17)],'x_raw':q.tolist()}
    return train,query


def _assert(ok,message):
    if not ok:raise AssertionError(message)


def manual_mapper_state(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        w1=nn.Linear(16,8,bias=True,device='cpu',dtype=torch.float64)
        w2=nn.Linear(8,16,bias=False,device='cpu',dtype=torch.float64)
        head=nn.Linear(16,1,bias=True,device='cpu',dtype=torch.float64)
    with torch.no_grad():
        w1.bias.zero_();w2.weight.zero_()
        u,s,vh=torch.linalg.svd(w1.weight,full_matrices=False)
        if float(s.max())>math.sqrt(.5):w1.weight.copy_((u*s.clamp(max=math.sqrt(.5))[None,:])@vh)
    return {**{'W1.'+k:v for k,v in w1.state_dict().items()},**{'W2.'+k:v for k,v in w2.state_dict().items()},
            **{'head.'+k:v for k,v in head.state_dict().items()}}


def _session_signature(session):
    model=session.model
    roots=(model.mapper,model.gp) if hasattr(model,'mapper') else (model.gp,)
    return {'fingerprint':model.fingerprint(),'optimizer':copy.deepcopy(session.optimizer.state_dict()),
        'rng':residual.rng_state(),'modes':[(name,m.training) for root in roots for name,m in root.named_modules()],
        'requires_grad':[p.requires_grad for root in roots for p in root.parameters()],
        'gradients':[None if p.grad is None else p.grad.detach().clone() for root in roots for p in root.parameters()],
        'inputs':[v.untyped_storage().data_ptr() for v in model.gp.train_inputs],
        'cache':id(model.gp.prediction_strategy)}


def run_checks(root,budget):
    root=Path(root);directory=root/'verification';directory.mkdir(parents=True,exist_ok=True)
    lock=directory/'oof_preflight_once.lock';lock.mkdir()
    (lock/'entry.json').write_text(json.dumps({'permanent':True,'automatic_repeat':False,
        'fixed_timing_calls':260,'planned_extra_isolation_calls':16,'maximum_all_attempts':300})+'\n')
    if budget.formal_updates or budget.formal_optimizer_calls:raise RuntimeError('Preflight cannot follow formal fitting')
    before=budget.as_dict();started=time.monotonic()
    result={'pass':False,'checks':[],'timing_cells':[],'planned_timing_optimizer_calls':260,
            'planned_isolation_optimizer_calls':16,'actual_optimizer_calls':0,'setup_counts':SETUP_COUNTS,
            'prediction_counts':PREDICTION_COUNTS,'n_fit':34,'n_holdout':17,
            'fixture_only':True,'timing_repetitions':1,'source_label_files_read':False}
    observations={}

    def observe(name,fn):
        start=time.monotonic();value=fn();elapsed=time.monotonic()-start
        observations.setdefault(name,[]).append(elapsed)
        return value

    def timing(session):
        _assert(session.planned_steps==PLANNED_STEPS[session.stage],'Timing changed prescribed planned_steps')
        call_before=budget.synthetic_optimizer_calls
        for _ in range(5):session.step(budget)
        windows=[]
        for j in range(3):
            tick=time.monotonic()
            for _ in range(20):session.step(budget)
            elapsed=time.monotonic()-tick
            windows.append({'segment':j,'steps':20,'seconds':elapsed,'seconds_per_step':elapsed/20})
        _assert(session.completed==65 and budget.synthetic_optimizer_calls-call_before==65,'Fixed 5+3x20 timing count')
        cell={'stage':session.stage,'n_fit':34,'n_holdout':17,'planned_steps':session.planned_steps,
              'warmup_steps':5,'segments':windows,'seconds_per_step':max(x['seconds_per_step'] for x in windows),
              'actual_optimizer_calls':65,'timing_statistic':'max of three contiguous 20-step means',
              'last_learning_rates':session.lrs}
        result['timing_cells'].append(cell)

    try:
        observe('formal_runtime_setup',lambda:_runtime_setup(budget))
        with measured(budget,'synthetic_fixture_and_checks','synthetic'):
            training,query=synthetic_fixture()
        prepared=observe('prepare_common',lambda:prepare_common(training,query,0,budget,'synthetic'))
        model=prepared['model'];scaler=prepared['scaler'];xq=prepared['x_query']
        observe('snapshot_INITIAL',lambda:charged_snapshot(model,budget,'synthetic'))
        with measured(budget,'initialization_scaler_checks','synthetic'):
            _assert(tensor_hash(manual_mapper_state(0))==prepared['initial_mapper_hash'],'Initialization creation order/projection differs')
            changed=copy.deepcopy(training);changed['y']=[-v+10 for v in training['y']];changed['measurement_se']=[v*2 for v in training['measurement_se']]
            other=prepare_common(changed,query,0,budget,'synthetic')
            _assert(other['initial_mapper_hash']==prepared['initial_mapper_hash'],'Mapper initialization depends on U labels/SE')
            xv=np.asarray(training['x_raw']);yv=np.asarray(training['y']);sev=np.asarray(training['measurement_se'])
            _assert(np.allclose(scaler['x_mean'],xv.mean(0),rtol=0,atol=1e-14),'Scaler not U-only x mean')
            _assert(np.allclose(scaler['x_scale'],np.maximum(xv.std(0),1e-6),rtol=0,atol=1e-14),'Scaler ddof/floor')
            _assert(abs(scaler['y_mean']-float(yv.mean()))<1e-14 and abs(scaler['y_scale']-float(yv.std()))<1e-14,'U-only y scaling')
            _assert(np.allclose(prepared['known_std'].numpy(),sev**2/scaler['y_scale']**2,rtol=1e-14,atol=1e-14),'Known SE counted more than once')
            for key in ('y','measurement_se','query_y','query_SE','labels'):
                poison={**query,key:object()}
                try:prepare_common(training,poison,0,budget,'synthetic')
                except ValueError:pass
                else:raise AssertionError('V label/SE crossed whitelist')
        result['checks'].append('manual_initialization_U_only_scaler_known_SE_once_strict_V_whitelist')

        mse=observe('session_COMMON_MSE',lambda:common_session(model,'COMMON_MSE','synthetic_COMMON_MSE',budget,'synthetic'))
        timing(mse)
        observe('diagnostic_COMMON_MSE',lambda:common_diagnostic(mse,budget,'synthetic'))
        observe('snapshot_COMMON_MSE',lambda:charged_snapshot(mse,budget,'synthetic'))
        observe('head_predictions',lambda:head_predictions(model,xq,scaler,budget,'synthetic'))
        frozen=observe('freeze_features',lambda:freeze_features(model,xq,budget,'synthetic'))
        common=observe('session_COMMON_GP',lambda:common_session(model,'COMMON_GP','synthetic_COMMON_GP',budget,'synthetic'))
        timing(common)
        observe('diagnostic_COMMON_GP',lambda:common_diagnostic(common,budget,'synthetic'))
        observe('snapshot_COMMON_GP',lambda:charged_snapshot(common,budget,'synthetic'))
        finalfrozen=observe('freeze_features',lambda:freeze_features(model,xq,budget,'synthetic'))
        _assert(frozen['mapper_hash']==finalfrozen['mapper_hash'],'COMMON_GP changed mapper/head')
        sessions={method:observe('session_'+method,lambda method=method:readout_session(method,finalfrozen,budget,'synthetic'))
                  for method in ('GP_CONST','GP_TREND')}
        assert_readout_isolation(list(sessions.values()),budget,'synthetic')
        for method,session in sessions.items():
            timing(session)
            observe('diagnostic_'+method,lambda session=session:readout_diagnostic(session,finalfrozen,query['ids'],scaler,0,budget,'synthetic'))
            observe('snapshot_'+method,lambda session=session:charged_snapshot(session,budget,'synthetic'))
        _assert(budget.synthetic_optimizer_calls-before['synthetic_optimizer_calls']==260,'Timing did not consume exactly 260 actual calls')
        result['checks'].append('fixed_260_timing_calls_four_classes_planned_schedules_retained_shared_readout_initialization')

        # Four bounded pairs, two actual steps per branch: 16 additional calls.
        # No changed planned_steps and no shifted synthetic scheduler counter.
        for stage in CLASSES:
            with measured(budget,'isolation_fixture_setup','synthetic'):
                if stage.startswith('COMMON_'):
                    aa=prepare_common(training,query,2,budget,'synthetic')
                    bb=prepare_common(training,query,2,budget,'synthetic')
                    pair=[common_session(z['model'],stage,'isolation_'+stage,budget,'synthetic') for z in (aa,bb)]
                    diag=lambda s:common_diagnostic(s,budget,'synthetic')
                else:
                    pair=[readout_session(stage,finalfrozen,budget,'synthetic') for _ in range(2)]
                    diag=lambda s:readout_diagnostic(s,finalfrozen,query['ids'],scaler,0,budget,'synthetic')
                first_rng=residual.rng_state()
            with measured(budget,'isolation_validation','synthetic'):
                for branch_index,session in enumerate(pair):
                    residual.set_rng(first_rng)
                    for _ in range(2):
                        session.step(budget)
                        if branch_index:
                            signature=_session_signature(session);diag(session)
                            _assert(residual.exact(signature,_session_signature(session)),'Diagnostic changed live optimizer/RNG/gradients/cache: '+stage)
                    if branch_index==0:
                        ref={'fingerprint':session.model.fingerprint(),'optimizer':copy.deepcopy(session.optimizer.state_dict()),'rng':residual.rng_state()}
                    else:
                        actual={'fingerprint':session.model.fingerprint(),'optimizer':copy.deepcopy(session.optimizer.state_dict()),'rng':residual.rng_state()}
                        _assert(residual.exact(ref,actual),'Diagnostic changed training trajectory: '+stage)
                    if stage=='COMMON_MSE':
                        _assert(residual.spectral_stats(session.model)['within_bounded_limits'],'Exact SVD projection did not preserve cap')
                    _assert(session.last_update['gradient_before_clip']>=0,'Missing/nonfinite gradient record')
                    _assert(all(float(v['step'])==2 for v in session.optimizer.state_dict()['state'].values()),'Per-parameter Adam update count')
            result['checks'].append(stage+':two_step_diagnostic_toggle_exact_optimizer_RNG_gradient_cache_isolation')

        with measured(budget,'final_counter_checks','synthetic'):
            _assert(readout.learning_rate_for_update(1000)==.01 and readout.learning_rate_for_update(1001)==.001,'Readout LR boundary')
            _assert(int(.8*PLANNED_STEPS['COMMON_MSE'])==800 and int(.8*PLANNED_STEPS['COMMON_GP'])==240,'Common LR boundary')
            for stage in CLASSES:
                cell=budget.class_counters[stage]['synthetic'];old=before['class_counters'][stage]['synthetic']
                _assert(cell['optimizer_calls']-old['optimizer_calls']==69,'Per-stage actual call count')
                _assert(cell['updates']-old['updates']==69 and cell['successful_optimizer_calls']-old['successful_optimizer_calls']==69,'Per-stage success count')
                _assert(cell['failed_optimizer_calls']==old['failed_optimizer_calls'],'Unexpected failed optimizer invocation')
            _assert(budget.formal_updates==budget.formal_optimizer_calls==0,'Synthetic work contaminated formal counts')
        result['checks'].append('276_total_synthetic_successes_four_classes_69_each_zero_formal')
        # Forecast samples are only the predeclared first synthetic pipeline.
        # Extra isolation work cannot replace a measured timing cell.
        per_operation={k:max(v) for k,v in observations.items()}
        _assert(set(SETUP_COUNTS)|set(PREDICTION_COUNTS)<=set(per_operation),'Admission omitted a numerical operation')
        fit=sum(cell['seconds_per_step']*FORMAL_UPDATES[cell['stage']] for cell in result['timing_cells'])
        setup=sum(per_operation[k]*v for k,v in SETUP_COUNTS.items())
        pred=sum(per_operation[k]*v for k,v in PREDICTION_COUNTS.items())
        result.update(pass_=True,operation_measurements=observations,operation_seconds_for_admission=per_operation,
            fit_est=fit,setup_est=setup,pred_est=pred,setup_seconds_per_pipeline=setup/90,
            gp_diagnostic_calls=1080,formal_updates_by_class=FORMAL_UPDATES,
            admission_formula='used + 1.5*(fit_est+setup_est) + max(30,2*pred_est)',
            mathematical_preflight_reuse={'residual_core.py':hashlib.sha256((root/'code/residual_core.py').read_bytes()).hexdigest(),
                'readout_core.py':hashlib.sha256((root/'code/readout_core.py').read_bytes()).hexdigest(),
                'readout_diagnostics.py':hashlib.sha256((root/'code/readout_diagnostics.py').read_bytes()).hexdigest(),
                'source':'unchanged audited parent math; no claim of a newly run independent GP mathematical suite'})
        result['pass']=result.pop('pass_')
    except BaseException:
        result['failure']=traceback.format_exc()
        if budget.synthetic_updates>before['synthetic_updates']:
            budget.mark_discarded(budget.synthetic_updates-before['synthetic_updates'])
        raise
    finally:
        # Cover numerical assertions, fixture generation and adapter/test setup
        # not individually timed, without counting nested helper costs twice.
        elapsed=time.monotonic()-started
        budget.charge(max(0.,elapsed-(budget.seconds-before['seconds'])),'verification')
        budget.persist()
        result.update(actual_optimizer_calls=budget.synthetic_optimizer_calls-before['synthetic_optimizer_calls'],
            actual_successful_updates=budget.synthetic_updates-before['synthetic_updates'],
            charged_seconds=budget.seconds-before['seconds'],wall_seconds=elapsed,budget_before=before,budget_after=budget.as_dict())
        (directory/'preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    budget.check()
    return result


def _runtime_setup(budget):
    with measured(budget,'formal_runtime_setup','synthetic'):cpu_setup(123)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);args=parser.parse_args()
    result=run_checks(args.root,Budget(args.root/'budget.json',limit=budget_limit(args.root)))
    print(json.dumps({'pass':result['pass'],'calls':result['actual_optimizer_calls'],'seconds':result['charged_seconds'],
        'fit_est':result.get('fit_est'),'setup_est':result.get('setup_est'),'pred_est':result.get('pred_est')}))
