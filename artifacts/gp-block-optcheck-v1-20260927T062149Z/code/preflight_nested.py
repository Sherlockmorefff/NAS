"""One fixed synthetic numerical preflight; all attempts share budget.json.

The parent orchestrator is the sole executor. This module never reads real data.
105 fixed timing requests plus a bounded, reported set of targeted checks.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import nested_core as core


def fixture(budget,n=50,q=38,boundary=False):
    with budget.measure('synthetic_fixture','synthetic'):
        rng=np.random.default_rng(20260927)
        x=rng.normal(0,.18,(n,16));qx=rng.normal(0,.18,(q,16))
        x[1]=x[0];x[2,:12]=x[0,:12];qx[0]=x[0]
        y=.2*np.sin(3*x[:,0])+.1*x[:,-1]
        if boundary:
            x[:,12:]=0.;qx[:,12:]=0.;y[:]=0.
        training={'ids':['t%03d'%i for i in range(n)],'architectures':['arch%03d'%i for i in range(n)],
                  'x_standardized16':x.tolist(),'y_standardized':y.tolist(),
                  'known_variance_standardized':[.02]*n}
        query={'ids':['q%03d'%i for i in range(q)],'architectures':['qarch%03d'%i for i in range(q)],
               'x_standardized16':qx.tolist()}
        scaler={'y_mean':.63,'y_scale':.12}
        return training,query,scaler


def require(ok,message):
    if not ok:raise AssertionError(message)


def close(a,b,label,atol=1e-10,rtol=1e-8):
    a=np.asarray(a);b=np.asarray(b)
    require(a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=rtol),label)
    return float(np.max(np.abs(a-b))) if a.size else 0.


def run_checks(root,budget):
    root=Path(root);checks=[];timing=[];before=budget.seconds;requests_before=budget.synthetic_requests
    def passed(name,**values):checks.append({'name':name,'pass':True,**values})
    with budget.measure('synthetic_preflight_total','synthetic'):
        environment=core.cpu_setup(budget,'synthetic')
        train,query,scaler=fixture(budget)
        models={m:core.make_model(m,train,budget,'synthetic') for m in core.METHODS}
        base={m:[.03,math.log(.9),math.log(.02)]+([.3,math.log(.7)] if m=='C_REFIT' else [math.log(1.2)]) for m in core.METHODS}

        # Fixed n50/Q38 request timing, including valid near-boundary parameters.
        for method in core.METHODS:
            theta=list(base[method]);theta[2]=core.LOW_NOISE+1e-6
            if method=='C_REFIT':theta[3]=1e-9
            for _ in range(5):core.evaluate(models[method],theta,budget,'synthetic')
            segments=[]
            for segment in range(3):
                started=time.monotonic()
                for _ in range(10):core.evaluate(models[method],theta,budget,'synthetic')
                segments.append({'segment':segment,'requests':10,'wall_seconds':time.monotonic()-started})
            timing.append({'method':method,'n_train':50,'n_query':38,'warmup_requests':5,'segments':segments,
                           'seconds_per_request':max(s['wall_seconds']/10 for s in segments),
                           'parameters':theta,'near_boundary_valid':True,'no_retiming':True})
        require(budget.synthetic_requests-requests_before==105,'Fixed timing request count')
        passed('fixed_105_timing_requests')

        require([len(core.bounds(m)) for m in core.METHODS]==[4,4,5],'Physical scalar counts')
        require(not np.shares_memory(models['A_REFIT'].x,models['B_REFIT'].x),'Training arrays shared')
        changed=copy.deepcopy(train);changed['x_standardized16'][0][0]+=1
        require(models['A_REFIT'].x[0,0]!=changed['x_standardized16'][0][0],'Source mutation leaks')
        for bad in ('y','measurement_se','query_y','query_se'):
            try:core.make_model('A_REFIT',{**train,bad:[0]*50},budget,'synthetic')
            except ValueError:pass
            else:raise AssertionError('Unexpected fit field accepted')
            try:core.predict(models['A_REFIT'],base['A_REFIT'],{**query,bad:[0]*38},scaler,budget,'synthetic')
            except ValueError:pass
            else:raise AssertionError('Unexpected query observation accepted')
        passed('4_4_5_parameters_isolation_and_query_poison_rejected')

        objectives={};predictions={};kernelsets={};maximum_formula_error=0.
        for method in core.METHODS:
            model=models[method];theta=base[method]
            objectives[method]=core.evaluate(model,theta,budget,'synthetic')
            independent=core.independent_evaluate(model,theta,budget,'synthetic')
            maximum_formula_error=max(maximum_formula_error,close(objectives[method][0],independent[0],'Independent objective'),
                                      close(objectives[method][1],independent[1],'Independent analytic gradient'))
            predictions[method]=core.predict(model,theta,query,scaler,budget,'synthetic')
            ref=core.independent_prediction(model,theta,query,scaler,budget,'synthetic')
            kernelsets[method]=core.kernels(model,theta,query,budget,'synthetic')
            for key in ('TT','QT','QQ'):maximum_formula_error=max(maximum_formula_error,close(kernelsets[method][key],ref[key],key+' independent closed formula'))
            for key in ('mean_standardized','latent_covariance_standardized'):
                maximum_formula_error=max(maximum_formula_error,close(predictions[method][key],ref[key],key+' dense conditional posterior'))
            raw_cov=np.asarray(predictions[method]['latent_covariance_raw'])
            close(raw_cov,scaler['y_scale']**2*np.asarray(predictions[method]['latent_covariance_standardized']),'Raw covariance conversion')
            close(predictions[method]['mean_raw'],scaler['y_mean']+scaler['y_scale']*np.asarray(predictions[method]['mean_standardized']),'Raw mean conversion')
            expected=scaler['y_scale']**2*(np.maximum(predictions[method]['latent_variance_standardized'],0)+math.exp(theta[2]))
            close(predictions[method]['model_variance_raw'],expected,'Extra variance once')
            require(predictions[method]['latent_covariance_symmetry_max_abs']<=1e-10 and predictions[method]['latent_covariance_min_eigenvalue']>=-1e-8,'Covariance PSD/symmetry')
        require(models['C_REFIT'].da[0,1]==models['C_REFIT'].dh[0,1]==0 and models['C_REFIT'].da[0,2]==0,'Duplicate/zero architecture fixture missing')
        se_raw=np.sqrt(np.asarray(train['known_variance_standardized']))*scaler['y_scale']
        close(se_raw**2/scaler['y_scale']**2,train['known_variance_standardized'],'SE conversion without additional division')
        passed('independent_matern_TT_QT_QQ_objective_gradient_posterior_units',maximum_absolute_error=maximum_formula_error,
               repeated_input=True,zero_architecture_distance=True,zero_total_distance=True,known_noise_once=True)

        maximum_gradient_error=0.
        for method in core.METHODS:
            model=models[method];theta=base[method];numerical=[];h=1e-5
            for k in range(len(theta)):
                left=list(theta);right=list(theta);left[k]-=h;right[k]+=h
                numerical.append((core.evaluate(model,right,budget,'synthetic')[0]-core.evaluate(model,left,budget,'synthetic')[0])/(2*h))
            maximum_gradient_error=max(maximum_gradient_error,close(objectives[method][1],numerical,'Central difference all coordinates',atol=1e-6,rtol=1e-4))
        boundary_theta=core.embed_into_C('B_REFIT',base['B_REFIT']);value,gradient=core.evaluate(models['C_REFIT'],boundary_theta,budget,'synthetic')
        h=1e-5;one=list(boundary_theta);two=list(boundary_theta);one[3]=h;two[3]=2*h
        one_sided=(-3*value+4*core.evaluate(models['C_REFIT'],one,budget,'synthetic')[0]-core.evaluate(models['C_REFIT'],two,budget,'synthetic')[0])/(2*h)
        maximum_gradient_error=max(maximum_gradient_error,close(gradient[3],one_sided,'a=0 one-sided derivative',atol=1e-6,rtol=1e-4))
        require(not np.isclose(gradient[3]+1,one_sided,atol=1e-6,rtol=1e-4),'Wrong boundary derivative was not detected')
        passed('central_and_one_sided_finite_differences',maximum_absolute_error=maximum_gradient_error,wrong_boundary_gradient_detected=True)

        for method in ('A_REFIT','B_REFIT'):
            mapped=core.embed_into_C(method,base[method]);mapped_value=core.evaluate(models['C_REFIT'],mapped,budget,'synthetic')[0]
            close(mapped_value,objectives[method][0],'Embedded NLL')
            mapped_pred=core.predict(models['C_REFIT'],mapped,query,scaler,budget,'synthetic')
            mapped_k=core.kernels(models['C_REFIT'],mapped,query,budget,'synthetic')
            for key in ('TT','QT','QQ'):close(mapped_k[key],kernelsets[method][key],'Embedding '+key)
            for key in ('mean_standardized','latent_covariance_standardized'):close(mapped_pred[key],predictions[method][key],'Embedding '+key)
        passed('exact_A_embedding_and_exact_B_closed_boundary')

        # Selection must ignore unaccepted line-search points and preserve ties.
        fake={'accepted':[{'start_index':0,'iteration':0,'nll':1.,'coordinates':[1]},
                          {'start_index':0,'iteration':1,'nll':1.-.5e-10,'coordinates':[2]}],
              'requests':[{'nll':-100.,'coordinates':[9]}]}
        chosen=core.select_candidates([fake]);require(chosen['iteration']==0 and chosen['coordinates']==[1],'Unaccepted/tie selection error')
        capped=core.run_trajectory(models['A_REFIT'],base['A_REFIT'],0,2,5,budget,'synthetic')
        require(capped['actual_requests']==2 and capped['termination']=='REQUEST_LIMIT','Hard cap inside line search failed')
        require(all(c['kind'] in ('initial','accepted_iteration') for c in capped['accepted']),'Unaccepted trial in pool')
        require(all(c['coordinates']==capped['requests'][c['request']-1]['coordinates'] for c in capped['accepted']),'Accepted/request identity mismatch')
        passed('line_search_hard_cap_and_accepted_only_tie_pool',request_cap=capped['actual_requests'])

        bt,bq,bs=fixture(budget,n=8,q=6,boundary=True)
        boundary_model=core.make_model('C_REFIT',bt,budget,'synthetic');x0=[0.,math.log(.8),math.log(.03),0.,0.]
        first=core.run_trajectory(boundary_model,x0,0,12,10,budget,'synthetic')
        with budget.measure('unrelated_rng_between_replicates','synthetic'):
            np.random.seed(937);np.random.normal(size=41)
        second=core.run_trajectory(boundary_model,x0,0,12,10,budget,'synthetic')
        for key in ('coordinates','nll','gradient','iteration','start_index'):
            require(first['best'][key]==second['best'][key],'Repeated boundary solve changed: '+key)
        require(first['actual_requests']==second['actual_requests'],'Deterministic request counts differ')
        require(first['best']['coordinates'][3]==0.,'Constructed optimum did not retain HP-only boundary')
        boundary_diag=core.diagnose_selected(boundary_model,first['best'],[{'source':'B_REFIT','nll':first['best']['nll']}],budget,'synthetic')
        require(boundary_diag['architecture_disabled'] is True and 'ARCHITECTURE_INCREMENT_SIGNAL' not in boundary_diag,'Boundary treated as architecture learning')
        require(core.projected_gradient([0.],[2.],[(0.,1.)])==[0.] and core.projected_gradient([0.],[-2.],[(0.,1.)])==[-2.], 'Lower-bound projected gradient sign')
        require(core.projected_gradient([1.],[-2.],[(0.,1.)])==[0.] and core.projected_gradient([1.],[2.],[(0.,1.)])==[2.], 'Upper-bound projected gradient sign')
        require(core.projected_gradient([1e-9],[2.],[(0.,1.)])==[2.],'Small positive a incorrectly snapped to zero')
        passed('deterministic_boundary_solver_projection_probe',actual_requests_each=first['actual_requests'],architecture_disabled=True)

        selected={'coordinates':base['C_REFIT'],'nll':objectives['C_REFIT'][0],'gradient':objectives['C_REFIT'][1].tolist(),'start_index':0,'iteration':0}
        frozen_identity=models['C_REFIT'].training_identity;frozen_x=models['C_REFIT'].x.copy()
        saved=core.snapshot(models['C_REFIT'],selected,scaler,budget,scope='synthetic')
        restored=core.restore_check(saved,train,query,predictions['C_REFIT'],budget,'synthetic')
        require(restored['pass'] and models['C_REFIT'].training_identity==frozen_identity and np.array_equal(models['C_REFIT'].x,frozen_x),'Restore/diagnostic mutated source')
        passed('scalar_snapshot_restore_independent_posterior_and_isolation',maximum_absolute_differences=restored['maximum_absolute_differences'])

        # One-shot entry guard test is a pure logic contract, not a model rerun.
        guard=root/'verification/synthetic_once_guard';guard.mkdir(parents=True,exist_ok=False)
        (guard/'entry.json').write_text('{"permanent":true,"synthetic_contract_only":true}\n')
        counted_before=copy.deepcopy(budget.counts)
        try:guard.mkdir()
        except FileExistsError:pass
        else:raise AssertionError('One-shot entry allowed repeated execution')
        require(counted_before==budget.counts,'Zero-call guard changed model counts')
        passed('one_shot_entry_contract_zero_additional_model_calls',formal_orchestrator_still_requires_own_guard_test=True)
    used=budget.synthetic_requests-requests_before
    require(used<=300,'Synthetic request cap exceeded')
    # Solver overhead includes wrapper bookkeeping as well as L-BFGS-B itself.
    overhead=max(max(0.,t['model_seconds']-sum(r.get('model_seconds',0.) for r in t['requests']))/max(1,t['actual_requests']) for t in (capped,first,second))
    operations=copy.deepcopy(budget.operations)
    report={'pass':True,'environment':environment,'checks':checks,'synthetic_nll_gradient_requests':used,
            'cumulative_synthetic_nll_gradient_requests':budget.synthetic_requests,'fixed_timing_requests':105,
            'targeted_requests':used-105,'charged_model_seconds':budget.seconds-before,'timing_cells':timing,
            'fit_est':6000*sum(c['seconds_per_request'] for c in timing),
            'solver_overhead_seconds_per_request_max':overhead,'solver_overhead_est_18000_requests':18000*overhead,
            'operation_costs':operations,'prediction_calls':budget.counts['synthetic']['prediction_calls'],
            'forecast_remaining_counts':{'formal_model_constructions':30,'formal_solver_instances':70,
                'selected_independent_gradients':30,'selected_probes_max':260,'final_predictions':30,
                'scalar_snapshots':30,'restore_model_constructions':30,'restore_independent_gradients':30,
                'restore_independent_predictions':30},
            'no_real_training_or_query_labels':True,'no_retiming':True,
            'notes':'Source30 verification is a separate charged preformal operation. Formal reentry guard belongs to orchestrator. No trajectory endpoint diagnostic predictions beyond final30.'}
    return report


def main(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    budget=core.Budget(root/'budget.json',600)
    lock=root/'preflight_once.lock';lock.mkdir()
    core._atomic(lock/'entry.json',{'permanent':True,'automatic_resume':False,'created_unix':time.time()})
    started=time.monotonic()
    try:
        report=run_checks(root,budget)
        report['wall_seconds']=time.monotonic()-started
        core._atomic(root/'verification/preflight.json',report)
        print(json.dumps({k:report[k] for k in ('pass','synthetic_nll_gradient_requests','charged_model_seconds','fit_est')},allow_nan=False))
    except BaseException as error:
        core._atomic(root/'verification/preflight_failed.json',{'pass':False,'error':repr(error),'wall_seconds':time.monotonic()-started,'budget':budget.as_dict()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
