"""Bounded synthetic checks. Run only under the main executor's shared ledger.

Default planned calls: six n/method timing cells x30 + two LR-boundary calls =182.
All attempts consume the same 300-call allowance; no formal data is loaded here.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from readout_core import budget_limit
from readout_core import (Budget, ReadoutModel, Session, METHODS, make_synthetic_fixture,
    cpu_setup, numerical_settings, tensor_hash, learning_rate_for_update)


def require(value,message):
    if not value:raise AssertionError(message)


def close(actual,expected,message):
    a=np.asarray(actual.detach().numpy() if isinstance(actual,torch.Tensor) else actual,dtype=np.float64)
    b=np.asarray(expected,dtype=np.float64)
    require(a.shape==b.shape and np.allclose(a,b,atol=1e-7,rtol=1e-7),message)
    return float(np.max(np.abs(a-b))) if a.size else 0.


def model_from(f,method,**extra):
    return ReadoutModel(method,f['h_train'],f['s_train'],f['y_std'],f['known_std'],f['common_gp_state'],**extra)


def independent_reference(model,f):
    """NumPy Matérn/Cholesky; no use of the tested model's kernel or predictor."""
    p=model.hyperparameters();h=f['h_train'].numpy();q=f['h_query'].numpy()
    s=f['s_train'].numpy();sq=f['s_query'].numpy();y=f['y_std'].numpy()
    h_all=np.concatenate((h,q));s_all=np.concatenate((s,sq));n=len(h)
    distance=np.sqrt(np.maximum(np.sum((h_all[:,None,:]-h_all[None,:,:])**2,axis=-1),0))
    r=math.sqrt(5)*distance/p['lengthscale']
    km=p['outputscale']*(1+r+r*r/3)*np.exp(-r)
    B=np.diag(model.gp.coefficient_variances.detach().numpy())
    phi=np.column_stack((np.ones(len(s_all)),s_all))
    kernel=km+(phi@B@phi.T if model.method=='GP_SEMI' else 0)
    mean=np.full(len(s_all),p['constant_mean']) if model.method=='GP_CONST' else s_all
    noise=f['known_std'].numpy()+p['extra_noise_variance_standardized']
    covariance=kernel[:n,:n]+np.diag(noise);jitter=0.
    try:chol=np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        jitter=1e-8;covariance=covariance+jitter*np.eye(n);chol=np.linalg.cholesky(covariance)
    residual=y-mean[:n];alpha=np.linalg.solve(chol.T,np.linalg.solve(chol,residual))
    nll=(.5*residual@alpha+np.log(np.diag(chol)).sum()+.5*n*math.log(2*math.pi))/n
    mu=mean[n:]+kernel[n:,:n]@alpha
    solved=np.linalg.solve(chol,kernel[:n,n:]);cov=kernel[n:,n:]-solved.T@solved
    answer={'nll':nll,'mean':mu,'covariance':cov,'jitter':jitter,'matern':km,'full':kernel}
    if model.method=='GP_SEMI':
        # A distinct explicit joint generative construction: independent beta
        # and g, followed by noisy linear observations. Condition the entire
        # beta/g Gaussian, then project it onto query f. Its covariance contains
        # beta-g cross terms, not a sum of independent marginal variances.
        total=len(h_all);prior=np.zeros((2+total,2+total));prior[:2,:2]=B;prior[2:,2:]=km
        location=np.zeros(2+total);location[1]=1
        observation=np.zeros((n,2+total));observation[:,:2]=phi[:n];observation[:,2:2+n]=np.eye(n)
        query=np.zeros((total-n,2+total));query[:,:2]=phi[n:];query[:,2+n:]=np.eye(total-n)
        obs_cov=observation@prior@observation.T+np.diag(noise)+jitter*np.eye(n)
        cross=prior@observation.T
        post_mean=location+cross@np.linalg.solve(obs_cov,y-observation@location)
        post_cov=prior-cross@np.linalg.solve(obs_cov,cross.T)
        marginal_sum=phi[n:]@post_cov[:2,:2]@phi[n:].T+post_cov[2+n:,2+n:]
        answer.update(joint_mean=query@post_mean,joint_covariance=query@post_cov@query.T,
            beta_mean=post_mean[:2],beta_cov=post_cov[:2,:2],
            omitted_cross_term_max=float(np.max(np.abs(query@post_cov@query.T-marginal_sum))))
    return answer


def math_checks(output,budget):
    results=[]
    for n in (20,50):
        f=make_synthetic_fixture(n,seed=100+n)
        for method in METHODS:
            model=model_from(f,method);reference=independent_reference(model,f)
            with torch.no_grad():nll=float(model.nll());mu,cov=model.predict(f['h_query'],f['s_query'])
            error={'nll':close(nll,reference['nll'],'Independent normalized training NLL'),
                'mean':close(mu,reference['mean'],'Independent query mean'),
                'full_covariance':close(cov,reference['covariance'],'Independent complete query covariance')}
            symmetry=float(torch.max(torch.abs(cov-cov.T)))
            smallest=float(np.linalg.eigvalsh(cov.numpy()).min())
            require(symmetry<=1e-10 and smallest>=-1e-8,'Posterior symmetry/PSD')
            require(np.any(np.abs(cov.numpy()-np.diag(np.diag(cov.numpy())))>1e-7),'Nontrivial offdiagonal covariance checked')
            scale=f['y_scale'];center=f['y_center'];raw_mu=center+scale*mu;raw_cov=scale*scale*cov
            close((raw_mu-center)/scale,mu.numpy(),'Mean unit roundtrip')
            close(raw_cov/scale**2,cov.numpy(),'Complete covariance unit roundtrip')
            extra=model.hyperparameters()['extra_noise_variance_standardized']
            se_q=np.linspace(.002,.008,len(mu));v_obs=raw_cov.diagonal().numpy()+scale**2*extra+se_q**2
            close(v_obs,scale**2*(cov.diagonal().numpy()+extra)+se_q**2,'Extra/query SE each once, no predictive jitter')
            close(f['known_std'],(np.linspace(.004,.012,n)/scale)**2,'Training mean SE squared and standardized once')
            state=model.snapshot();clone=ReadoutModel.from_snapshot(state)
            repeat=clone.predict(f['h_query'],f['s_query'])
            require(torch.equal(mu,repeat[0]) and torch.equal(cov,repeat[1]),'Repeated saved-state prediction exact')
            require(model.frozen_hash()==model.initial_frozen_hash==clone.frozen_hash(),'Frozen tensors unchanged')
            if method=='GP_SEMI':
                error['explicit_joint_mean']=close(mu,reference['joint_mean'],'Joint generative conditional mean')
                error['explicit_joint_covariance']=close(cov,reference['joint_covariance'],'Joint generative conditional full covariance')
                beta=model.semi_posterior(f['h_query'],f['s_query'])
                error['beta_mean']=close(beta['beta_mean'],reference['beta_mean'],'Joint beta posterior mean')
                error['beta_cov']=close(beta['beta_cov'],reference['beta_cov'],'Joint beta posterior covariance')
                close(beta['trend_mean']+beta['residual_mean'],mu.numpy(),'Posterior mean decomposition')
                require(reference['omitted_cross_term_max']>1e-6,'Fixture must detect omitted beta-GP posterior correlation')
            if method=='GP_CONST':
                from stage01_core import ExactGP
                old=ExactGP(f['h_train'],f['y_std'],f['known_std']);old.load_state_dict(f['common_gp_state'])
                old.eval();old.likelihood.eval()
                with torch.no_grad(),numerical_settings():
                    original=old(f['h_query']);old_mu=original.mean.clone();old_cov=original.lazy_covariance_matrix.to_dense().clone()
                close(mu,old_mu.numpy(),'GP_CONST same-state old native prediction mean')
                close(cov,old_cov.numpy(),'GP_CONST same-state old native full covariance')
            results.append({'n':n,'method':method,'maximum_absolute_errors':error,'covariance_symmetry_error':symmetry,
                'covariance_minimum_eigenvalue':smallest,'solver_jitter':reference['jitter']})
        trend=model_from(f,'GP_TREND');zero=model_from(f,'GP_SEMI',coefficient_variances=(0.,0.),synthetic_fixture=True)
        with torch.no_grad():
            close(zero.nll(),trend.nll().numpy(),'Synthetic B=0 restores trend NLL')
            mz,cz=zero.predict(f['h_query'],f['s_query']);mt,ct=trend.predict(f['h_query'],f['s_query'])
        close(mz,mt.numpy(),'B=0 trend mean');close(cz,ct.numpy(),'B=0 trend covariance')
        semi=model_from(f,'GP_SEMI');s=f['s_train'];h=f['h_train']
        with torch.no_grad():
            first=semi.kernels(h,s);expected=.0625+.25*s[:,None]*s[None,:]
            close(first['full']-first['matern'],expected.numpy(),'Fixed finite-rank additive term')
            require(bool((expected<0).any()),'Negative s cross covariance remains negative')
            semi.gp.covar_module.outputscale=semi.gp.covar_module.outputscale.detach()*3
            second=semi.kernels(h,s)
            close(second['full']-second['matern'],expected.numpy(),'Outputscale affects only Matern')
            empty=semi.kernels(h,torch.zeros_like(s))
            close(empty['full']-empty['matern'],np.full((n,n),.0625),'Zero trend leaves constant rank-one term')
            require(float(np.linalg.eigvalsh(first['full'].numpy()).min())>=-1e-8,'Negative-cross full kernel remains PSD')
    return {'pass':True,'checks':results,'optimizer_updates':0,
        'independent_routes':['NumPy direct integrated kernel Cholesky','joint beta/g Gaussian plus linear-observation conditioning']}


def timing_checks(output,budget):
    from readout_diagnostics import diagnose_isolated
    results=[];prediction_seconds=0.;prediction_calls=0
    for n in (20,50):
        f=make_synthetic_fixture(n,seed=200+n)
        models={method:model_from(f,method) for method in METHODS}
        modules=[{id(v) for v in model.gp.modules()} for model in models.values()]
        stores=[{v.untyped_storage().data_ptr() for v in list(model.gp.state_dict().values())+
                  [model.h_train,model.s_train,model.x,model.y,model.known] if v.numel()} for model in models.values()]
        for i in range(3):
            for j in range(i):require(not modules[i]&modules[j] and not stores[i]&stores[j],'Readout branches share mutable state')
        for method in METHODS:
            model=models[method];session=Session(model);peers={k:v.fingerprint() for k,v in models.items() if k!=method}
            rows=[];before=budget.synthetic_updates
            for step in range(30):
                rows.append(session.step(budget,kind='synthetic'))
                if step==0:
                    require(all(models[k].fingerprint()==value for k,value in peers.items()),'Single-group update changed another readout')
            require(budget.synthetic_updates-before==30,'Exact synthetic timing-cell update count')
            require(model.frozen_hash()==model.initial_frozen_hash,'Optimization changed frozen h/s/targets')
            require(set(model.parameter_names())==set(session.names),'Trainable parameter inventory drift')
            for value in session.optimizer.state.values():require(float(value['step'])==30,'Adam per-parameter step count')
            # The data/diagnostics agent supplies the actual formal diagnostic
            # routine, so the budget forecast includes cloning/full-cov/beta.
            start=time.monotonic()
            record=diagnose_isolated(model,f['h_query'],f['s_query'],f['query_ids'],f['y_center'],f['y_scale'],
                                    0)
            elapsed=time.monotonic()-start;prediction_seconds+=elapsed;prediction_calls+=1
            row={'n':n,'method':method,'updates':30,'training_seconds':session.training_seconds,
                 'seconds_per_update':session.training_seconds/30,'diagnostic_seconds':elapsed,
                 'gradient_finite':all(math.isfinite(r['gradient_before_clip']) for r in rows),
                 'parameter_names':session.names,'frozen_feature_hash':model.frozen_hash()}
            results.append(row)
            (output/f'timing_n{n}_{method}.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
    return {'pass':True,'cells':results,'optimizer_updates':180,'diagnostic_prediction_seconds':prediction_seconds,
            'diagnostic_prediction_calls':prediction_calls,'forecast_training_seconds':sum(r['seconds_per_update']*45000 for r in results)}


def schedule_checks(output,budget):
    require(learning_rate_for_update(1)==learning_rate_for_update(1000)==.01 and
            learning_rate_for_update(1001)==learning_rate_for_update(1500)==.001,'Exact 1000/500 absolute schedule')
    f=make_synthetic_fixture(20,seed=777);session=Session(model_from(f,'GP_SEMI',synthetic_fixture=True))
    # Synthetic cursor fixture only: two Adam calls cross the fixed boundary;
    # it is explicitly not a claim to have performed 1001 training updates.
    session.completed=999
    first=session.step(budget,'synthetic');second=session.step(budget,'synthetic')
    require(first['step']==1000 and first['learning_rates']['gp']==.01 and second['step']==1001 and second['learning_rates']['gp']==.001,'Actual optimizer LR across boundary')
    require(all(float(v['step'])==2 for v in session.optimizer.state.values()),'Schedule fixture must consume exactly two Adam updates')
    return {'pass':True,'optimizer_updates':2,'synthetic_cursor_fixture_start':999,'actual_Adam_parameter_steps':2,
            'observed_rates':[first['learning_rates'],second['learning_rates']]}


def run_checks(output_dir,budget,only='all'):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    steps0=budget.synthetic_updates;calls0=budget.synthetic_optimizer_calls;cost0=budget.seconds;start=time.monotonic()
    result={'pass':False,'section':only,'checks':{},'failure':None}
    sections={'math':math_checks,'timing':timing_checks,'schedule':schedule_checks}
    try:
        for name,function in sections.items():
            if only not in ('all',name):continue
            budget.check();result['checks'][name]=function(output,budget);budget.check()
        result['pass']=True
    except Exception as exc:
        result['failure']={'type':type(exc).__name__,'message':str(exc)}
        budget.mark_discarded(budget.synthetic_updates-steps0)
    finally:
        # Covers fixtures, independent NumPy verification and diagnostics, while
        # charging Session.step's measured time only once.
        elapsed=time.monotonic()-start;already=budget.seconds-cost0
        budget.charge(max(0.,elapsed-already),kind='verification');budget.persist()
    result.update(synthetic_updates=budget.synthetic_updates-steps0,synthetic_optimizer_calls=budget.synthetic_optimizer_calls-calls0,
                  charged_seconds=budget.seconds-cost0,resources=budget.as_dict(),model_data='synthetic_only')
    (output/'core_checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--only',choices=('all','math','timing','schedule'),default='all');args=parser.parse_args()
    cpu_setup(123);budget=Budget(args.root/'budget.json',limit=budget_limit(args.root))
    directory=args.root/'verification';directory.mkdir(exist_ok=True)
    attempt=1+len(list(directory.glob('core_attempt_*')))
    result=run_checks(directory/f'core_attempt_{attempt}',budget,args.only)
    if args.only=='all' and result['pass']:
        from readout_diagnostics import run_checks as diagnostic_checks
        try:
            result['checks']['diagnostics']=diagnostic_checks(directory/f'core_attempt_{attempt}',budget)
            result['pass']=bool(result['checks']['diagnostics']['pass'])
        except Exception as exc:
            result['pass']=False;result['failure']={'type':type(exc).__name__,'message':str(exc),'section':'diagnostics'}
    result['resources']=budget.as_dict()
    (directory/f'core_summary_attempt_{attempt}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if args.only=='all':
        timing=result['checks'].get('timing',{})
        aggregate={'pass':result['pass'],'attempt':attempt,'core_summary_path':f'verification/core_summary_attempt_{attempt}.json',
            'timing_cells':[{**row,'seconds_per_step':row['seconds_per_update']} for row in timing.get('cells',[])],
            'prediction_seconds':timing.get('diagnostic_prediction_seconds',0.),
            'prediction_calls':timing.get('diagnostic_prediction_calls',0),
            'synthetic_optimizer_updates':budget.synthetic_updates,'synthetic_optimizer_calls':budget.synthetic_optimizer_calls,
            'verification_seconds':budget.verification_seconds,'resources':budget.as_dict(),'failure':result['failure']}
        (directory/'preflight.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'pass':result['pass'],'synthetic_updates':result['synthetic_updates'],'charged_seconds':result['charged_seconds'],'failure':result['failure']}))
    raise SystemExit(0 if result['pass'] else 1)
