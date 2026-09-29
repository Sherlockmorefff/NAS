"""One fixed synthetic timing suite plus targeted new-kernel verification.

Root alone executes it. 195 timing optimizer calls + 21 short verification
calls =216. All model operations, gradient checks and independent Cholesky
computations are charged. No source training or query truth is opened.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from block_core import (METHODS,Budget,budget_limit,BlockModel,prepare_inputs,create_model,session,diagnose,
    charged_snapshot,distance_diagnostics,measured,learning_rate,exact,cpu_setup,rng_state,set_rng,_signature)

ATOL,RTOL=1e-10,1e-8
SETUP_COUNTS={'runtime_setup':1,'prepare_inputs':30,**{'model_'+m:10 for m in METHODS},
    **{'session_'+m+'_'+s:10 for m in METHODS for s in ('INIT_GP','FIT_GP')}}
DIAGNOSTIC_COUNTS={**{'diagnostic_'+m:80 for m in METHODS},**{'distance_'+m:20 for m in METHODS},
    **{'snapshot_'+m+'_'+s:10 for m in METHODS for s in ('INITIAL','INIT_GP','FIT_GP')}}


def require(ok,message):
    if not ok:raise AssertionError(message)
def close(a,b,name,atol=ATOL,rtol=RTOL):
    a,b=torch.as_tensor(a,dtype=torch.float64),torch.as_tensor(b,dtype=torch.float64)
    if a.shape!=b.shape or not torch.allclose(a,b,atol=atol,rtol=rtol):
        raise AssertionError(name+': '+str(None if a.shape!=b.shape else float((a-b).abs().max())))


def fixture(n=50,q=38,seed=1977):
    g=torch.Generator(device='cpu').manual_seed(seed)
    x=torch.randn((n,16),generator=g,dtype=torch.float64)*.7
    z=torch.randn((q,16),generator=g,dtype=torch.float64)*.7
    # Both exact duplicate rows and exact train/query coincidence are intentional.
    x[1]=x[0];z[0]=x[0];z[1]=z[0]
    y=.25*x[:,0]-.35*x[:,13]+.1*torch.cos(x[:,4]);known=torch.full((n,),.0002,dtype=torch.float64)
    ids=['synthetic_T%02d'%i for i in range(n)];qids=['synthetic_Q%02d'%i for i in range(q)]
    train={'ids':ids,'architectures':['T_arch%02d'%i for i in range(n)],'x_standardized':x.tolist(),
           'y_standardized':y.tolist(),'known_variance_standardized':known.tolist()}
    query={'ids':qids,'architectures':['Q_arch%02d'%i for i in range(q)],'x_standardized':z.tolist()}
    scaler={'x_mean':[0.]*16,'x_scale':[1.]*16,'y_mean':.73,'y_scale':.047,'fit_ids':ids,'std_correction':0,'frozen':True,'dtype':'float64'}
    return train,query,scaler,x,z,y,known


def set_lengths(model,architecture,hp=None):
    base=model.gp.covar_module.base_kernel
    if model.method=='RAW16_BLOCK2':base.architecture_lengthscale=architecture;base.hp_lengthscale=architecture if hp is None else hp
    else:base.lengthscale=architecture


def closed_kernel(left,right,method,architecture,hp,scale):
    """Independent explicit polynomial/exponential Matérn, no model kernel call."""
    difference=left[:,None,:]-right[None,:,:]
    if method=='HP4_ISO':r2=difference[...,12:].square().sum(-1)/hp**2
    else:r2=difference[...,:12].square().sum(-1)/architecture**2+difference[...,12:].square().sum(-1)/hp**2
    r=r2.clamp_min(0).sqrt();t=math.sqrt(5.)*r
    return scale*(1+t+5*r2/3)*torch.exp(-t)


def dense_reference(model,q):
    p=model.hyperparameters();a=p['lengthscale_architecture'];h=p['lengthscale_hp'];sf=p['outputscale'];m=p['constant_mean']
    kernel=lambda x,z:closed_kernel(x,z,model.method,a,h,sf)
    k=kernel(model.raw_x,model.raw_x);kqt=kernel(q,model.raw_x);kqq=kernel(q,q)
    covariance=k+torch.diag(model.known+p['extra_noise_variance_standardized'])
    chol,info=torch.linalg.cholesky_ex(covariance);jitter=0.
    if bool(info.any()):
        jitter=1e-8;chol=torch.linalg.cholesky(covariance+jitter*torch.eye(len(k),dtype=torch.float64))
    error=(model.y-m)[:,None];alpha=torch.cholesky_solve(error,chol)
    nll=(.5*(error.T@alpha).squeeze()+torch.log(torch.diagonal(chol)).sum()+.5*len(k)*math.log(2*math.pi))/len(k)
    return {'nll':nll,'mean':m+(kqt@alpha).squeeze(-1),'covariance':kqq-kqt@torch.cholesky_solve(kqt.T,chol),
            'K_TT':k,'K_QT':kqt,'K_QQ':kqq,'actual_independent_jitter':jitter}


def mathematical_checks(budget):
    checks=[];details={}
    with measured(budget,'new_kernel_mathematical_checks','synthetic'):
        _,_,_,x,q,y,known=fixture(13,9)
        models={m:BlockModel(m,x,y,known) for m in METHODS}
        allmodules=[{id(v) for v in m.gp.modules()} for m in models.values()]
        require(not any(allmodules[i]&allmodules[j] for i in range(3) for j in range(i+1,3)),'Models share mutable GP/constraint modules')
        for method,model in models.items():
            spec=model.parameter_spec();expected=5 if method=='RAW16_BLOCK2' else 4
            require(sum(r['numel'] for r in spec)==len(spec)==expected,'Actual scalar parameter count '+method)
            require(not model.known.requires_grad,'Known noise must not be learned')
            if method=='RAW16_BLOCK2':
                base=model.gp.covar_module.base_kernel
                require(not base.internal.raw_lengthscale.requires_grad,'Forbidden third learnable lengthscale')
                close(base.internal.lengthscale,torch.ones_like(base.internal.lengthscale),'Frozen internal unit length')
                set_lengths(model,.83,1.61)
            elif method=='HP4_ISO':
                require(torch.equal(model.x,x[:,-4:]),'HP input is not exact last4 projection')
                set_lengths(model,1.21)
            else:set_lengths(model,.79)
            model.gp.covar_module.outputscale=.91;model.gp.likelihood.second_noise=.017;model.gp.mean_module.constant=-.13
            p=model.hyperparameters()
            reference=dense_reference(model,q)
            close(model.kernel(),reference['K_TT'],method+' independent duplicate/diagonal train kernel')
            close(model.kernel(q,x),reference['K_QT'],method+' independent cross kernel')
            close(model.kernel(q,q),reference['K_QQ'],method+' independent query kernel')
            close(model.nll().detach(),reference['nll'],method+' independent once-normalized NLL')
            mu,cov=model.predict(q)
            close(mu,reference['mean'],method+' independent GP posterior mean')
            close(cov,reference['covariance'],method+' independent full posterior covariance')
            close(model.gp.likelihood.noise_covar.noise,known,'Known noise preserved once')
            require(float((cov-cov.T).abs().max())<=1e-10 and float(torch.linalg.eigvalsh((cov+cov.T)/2).min())>=-1e-8,'Full covariance symmetry/PSD')
            details[method]={'parameter_spec':spec,'dense_cholesky_jitter':reference['actual_independent_jitter'],
                'max_mean_absolute_difference':float((mu-reference['mean']).abs().max()),
                'max_covariance_absolute_difference':float((cov-reference['covariance']).abs().max())}
        checks+=['4_4_5_actual_scalar_parameters_no_neural_no_third_length','three_models_no_shared_mutable_modules',
                 'independent_closed_Matern_TT_QT_QQ_duplicates_diagonals_unequal_blocks',
                 'independent_dense_NLL_full_posterior_known_extra_noise_once_actual_jitter_recorded']

        a=BlockModel('RAW16_ISO',x,y,known);c=BlockModel('RAW16_BLOCK2',x,y,known)
        for ell in (1.,.7,2.):
            set_lengths(a,ell);set_lengths(c,ell,ell)
            for name,xx,zz in [('TT',x,x),('QT',q,x),('QQ',q,q)]:close(a.kernel(xx,zz),c.kernel(xx,zz),'Nested A=C '+name+' ell='+str(ell))
            close(a.nll(),c.nll(),'Nested NLL ell='+str(ell))
            am,ac=a.predict(q);cm,cc=c.predict(q);close(am,cm,'Nested posterior mean');close(ac,cc,'Nested full covariance')
        xc=x.clone();qc=q.clone();xc[:,:12]=.35;qc[:,:12]=.35
        b=BlockModel('HP4_ISO',xc,y,known);cconstant=BlockModel('RAW16_BLOCK2',xc,y,known)
        set_lengths(b,1.37);set_lengths(cconstant,.44,1.37)
        close(b.kernel(qc,xc),cconstant.kernel(qc,xc),'Constant architecture C reduces to HP-only')
        hp_pair=torch.stack((x[2],x[3]));hp_pair[1,:12]=hp_pair[0,:12]
        z_pair=torch.stack((x[2],x[3]));z_pair[1,12:]=z_pair[0,12:]
        set_lengths(c,.6,1.2);hp_before=c.kernel(hp_pair,hp_pair).detach().clone()
        set_lengths(c,2.3,1.2);close(hp_before,c.kernel(hp_pair,hp_pair),'Pure HP pair changed with architecture length')
        z_before=c.kernel(z_pair,z_pair).detach().clone()
        set_lengths(c,2.3,.47);close(z_before,c.kernel(z_pair,z_pair),'Pure architecture pair changed with HP length')
        checks.append('A_C_equal_1_0p7_2_and_block_specific_effects_and_constant_architecture_reduction')

        c=BlockModel('RAW16_BLOCK2',x,y,known);set_lengths(c,.87,1.43)
        c.gp.zero_grad(set_to_none=True);c.nll().backward();base=c.gp.covar_module.base_kernel
        gradient_rows=[]
        for name in ('raw_architecture_lengthscale','raw_hp_lengthscale'):
            parameter=getattr(base,name);autograd=float(parameter.grad);require(math.isfinite(autograd) and abs(autograd)>1e-10,'Ineffective block gradient '+name)
            original=parameter.detach().clone();eps=1e-5
            with torch.no_grad():parameter.copy_(original+eps)
            plus=float(c.nll().detach())
            with torch.no_grad():parameter.copy_(original-eps)
            minus=float(c.nll().detach())
            with torch.no_grad():parameter.copy_(original)
            finite_difference=(plus-minus)/(2*eps)
            close(autograd,finite_difference,'Finite difference '+name,1e-6,1e-4)
            gradient_rows.append({'parameter':name,'autograd':autograd,'finite_difference':finite_difference})
        a=BlockModel('RAW16_ISO',x,y,known);c=BlockModel('RAW16_BLOCK2',x,y,known)
        for model in (a,c):model.gp.zero_grad(set_to_none=True);model.nll().backward()
        ga=a.gp.covar_module.base_kernel.raw_lengthscale.grad
        gc=c.gp.covar_module.base_kernel.raw_architecture_lengthscale.grad+c.gp.covar_module.base_kernel.raw_hp_lengthscale.grad
        close(ga,gc,'Common raw gradient equals sum of two constrained block raw gradients',1e-6,1e-4)
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for model in (a,c) for p in model.gp.parameters() if p.requires_grad),'Duplicate/r0 gradient nonfinite')
        details['finite_difference']=gradient_rows
        checks.append('finite_effective_two_raw_gradients_FD_and_common_gradient_sum_with_duplicate_r0')
    return checks,details


def run_checks(root,budget):
    root=Path(root);directory=root/'verification';directory.mkdir(parents=True,exist_ok=True)
    lock=directory/'block_preflight_once.lock';lock.mkdir()
    (lock/'entry.json').write_text(json.dumps({'permanent':True,'fixed_timing_calls':195,'short_validation_calls':21,'maximum_all_attempts':300})+'\n')
    require(budget.formal_optimizer_calls==0,'Preflight cannot follow formal execution')
    start=time.monotonic();before=budget.as_dict();observations={}
    result={'pass':False,'checks':[],'timing_cells':[],'setup_counts':SETUP_COUNTS,'diagnostic_counts':DIAGNOSTIC_COUNTS,
        'n_train':50,'n_query':38,'planned_timing_calls':195,'planned_short_validation_calls':21,
        'source_or_query_labels_read':False,'timing_repetitions':1}
    def observe(name,fn):
        tick=time.monotonic();value=fn();observations.setdefault(name,[]).append(time.monotonic()-tick);return value
    def runtime():
        with measured(budget,'runtime_setup','synthetic'):cpu_setup(0)
    try:
        observe('runtime_setup',runtime)
        with measured(budget,'synthetic_fixture','synthetic'):train,query,scaler,*_=fixture()
        prepared=observe('prepare_inputs',lambda:prepare_inputs(train,query,scaler,budget,'synthetic'))
        checks,details=mathematical_checks(budget);result['checks']+=checks;result['mathematical_details']=details
        with measured(budget,'query_whitelist_and_projection_checks','synthetic'):
            for key in ('y','measurement_se','labels','query_y','query_SE'):
                try:prepare_inputs(train,{**query,key:object()},scaler,budget,'synthetic')
                except ValueError:pass
                else:raise AssertionError('Query truth crossed whitelist')
            again=prepare_inputs(train,query,scaler,budget,'synthetic')
            for key in ('x_train16','x_query16','y_std','known_std'):require(torch.equal(prepared[key],again[key]),'Query poison changed projected inputs')
            check_model=create_model('RAW16_ISO',prepared,budget,'synthetic')
            first=check_model.predict(prepared['x_query16']);second=check_model.predict(again['x_query16'])
            close(first[0],second[0],'Rejected Q poison changed mean');close(first[1],second[1],'Rejected Q poison changed covariance')
            all_zero=BlockModel('RAW16_ISO',torch.zeros_like(prepared['x_train16']),prepared['y_std'],prepared['known_std'])
            zero_distance=distance_diagnostics(all_zero,prepared,budget,'synthetic')
            require(zero_distance['pair_count']==zero_distance['zero_distance_pairs']==1225 and
                zero_distance['share_architecture'] is None and zero_distance['share_hp'] is None,'All-zero distance pairs must yield N/A shares')
        result['checks'].append('strict_public_query_whitelist_and_identical_projected_inputs_after_poison')

        for method in METHODS:
            model=observe('model_'+method,lambda method=method:create_model(method,prepared,budget,'synthetic'))
            observe('snapshot_'+method+'_INITIAL',lambda:charged_snapshot(model,budget,'synthetic'))
            # Timing remains a1500-step FIT schedule with exactly65 real calls.
            active=observe('session_'+method+'_FIT_GP',lambda:session(model,'FIT_GP',budget,'synthetic'))
            for _ in range(5):active.step(budget)
            segments=[]
            for index in range(3):
                tick=time.monotonic()
                for _ in range(20):active.step(budget)
                elapsed=time.monotonic()-tick;segments.append({'segment':index,'steps':20,'seconds':elapsed,'seconds_per_step':elapsed/20})
            require(active.planned_steps==1500 and active.completed==65,'Timing altered prescribed planned steps')
            result['timing_cells'].append({'method':method,'planned_steps':1500,'actual_optimizer_calls':65,
                'warmup_steps':5,'segments':segments,'seconds_per_step':max(v['seconds_per_step'] for v in segments),
                'covers':'INIT and FIT steps have identical NLL/backward/clipping/accounting computation; phase changes only prescribed LR/new Adam',
                'last_learning_rates':active.lrs})
            record=observe('diagnostic_'+method,lambda:diagnose(active,prepared,budget,'synthetic'))
            observe('snapshot_'+method+'_FIT_GP',lambda:charged_snapshot(active,budget,'synthetic'))
            observe('distance_'+method,lambda:distance_diagnostics(model,prepared,budget,'synthetic'))
            with measured(budget,'unit_checks_'+method,'synthetic'):
                v=torch.tensor(record['query_latent_variance_standardized']);cov=torch.tensor(record['query_latent_covariance_standardized'])
                ys=scaler['y_scale'];extra=record['extra_noise_variance_standardized']
                close(v,cov.diagonal(),'Unclamped latent diagonal')
                close(torch.tensor(record['query_latent_covariance_raw']),cov*ys**2,'Raw covariance units',1e-14,1e-12)
                close(torch.tensor(record['model_variance_raw']),(v.clamp_min(0)+extra)*ys**2,'Extra GP variance exactly once, no SE/jitter',1e-14,1e-12)
                close(torch.tensor(record['query_mean_raw']),scaler['y_mean']+ys*torch.tensor(record['query_mean_standardized']),'Raw mean units',1e-14,1e-12)
        require(budget.synthetic_optimizer_calls-before['synthetic_optimizer_calls']==195,'Fixed timing count differs')
        result['checks'].append('fixed195_timing_calls_5warm_3x20_max_segment_n50_q38_same1500schedule')

        for method in METHODS:
            # 2 INIT +1 FIT actual steps demonstrate own-state retention/new Adam.
            with measured(budget,'stage_fixture_'+method,'synthetic'):
                model=create_model(method,prepared,budget,'synthetic');initial=model.fingerprint()
            init=observe('session_'+method+'_INIT_GP',lambda:session(model,'INIT_GP',budget,'synthetic'))
            for _ in range(2):init.step(budget)
            observe('snapshot_'+method+'_INIT_GP',lambda:charged_snapshot(init,budget,'synthetic'))
            with measured(budget,'stage_switch_validation_'+method,'synthetic'):
                own=model.fingerprint();old_optimizer=copy.deepcopy(init.optimizer.state_dict())
                require(own!=initial and bool(old_optimizer['state']),'Short first phase did not change parameters/moments')
                second=session(model,'FIT_GP',budget,'synthetic')
                require(model.fingerprint()==own and not second.optimizer.state and second.completed==0 and second.planned_steps==1500,'Second phase reset parameters or retained Adam moments')
                second.step(budget)
                require(all(float(v['step'])==1 for v in second.optimizer.state_dict()['state'].values()),'New Adam step is not1')
                require(model.total_steps==3 and second.last_update['local_step']==1 and second.last_update['total_step']==3,'Phase/cumulative step identity')
            # Two equal trajectories, unrelated RNG seeds, diagnostic on one.
            with measured(budget,'determinism_and_diagnostic_isolation_'+method,'synthetic'):
                outputs=[]
                for i,seed in enumerate((71,991)):
                    cpu_setup(seed);model=create_model(method,prepared,budget,'synthetic');active=session(model,'FIT_GP',budget,'synthetic')
                    for _ in range(2):
                        active.step(budget)
                        if i:
                            signature=_signature(model);optimizer=copy.deepcopy(active.optimizer.state_dict());incoming=rng_state()
                            diagnose(active,prepared,budget,'synthetic')
                            require(exact(signature,_signature(model)) and exact(optimizer,active.optimizer.state_dict()) and exact(incoming,rng_state()),'Diagnostic changed live model/Adam/RNG/gradients/cache')
                    outputs.append({'fingerprint':model.fingerprint(),'optimizer':copy.deepcopy(active.optimizer.state_dict())})
                require(exact(outputs[0],outputs[1]),'Unrelated RNG seed changed deterministic GP trajectory')
            result['checks'].append(method+':own_phase_state_new_Adam_determinism_and_diagnostic_toggle')
        with measured(budget,'final_schedule_accounting_checks','synthetic'):
            require(learning_rate('INIT_GP',240)==.01 and learning_rate('INIT_GP',241)==.001 and
                learning_rate('FIT_GP',1000)==.01 and learning_rate('FIT_GP',1001)==.001,'Prescribed LR boundary differs')
            require(budget.synthetic_optimizer_calls-before['synthetic_optimizer_calls']==216 and budget.synthetic_updates-before['synthetic_updates']==216,'Expected216 actual synthetic calls/successes')
            require(budget.formal_updates==budget.formal_optimizer_calls==0,'Synthetic work contaminated formal totals')
            require(all(budget.method_stage_counters[m][s]['synthetic']['failed_optimizer_calls']==0 for m in METHODS for s in ('INIT_GP','FIT_GP')),'Unexpected failed call')
        estimates={name:max(values) for name,values in observations.items()}
        require(set(SETUP_COUNTS)|set(DIAGNOSTIC_COUNTS)<=set(estimates),'Missing forecast operation')
        fit=18000*sum(c['seconds_per_step'] for c in result['timing_cells'])
        setup=sum(estimates[k]*v for k,v in SETUP_COUNTS.items())
        diagnostics=sum(estimates[k]*v for k,v in DIAGNOSTIC_COUNTS.items())
        result.update(fit_est=fit,setup_est=setup,diagnostic_est=diagnostics,operation_measurements=observations,
            operation_seconds_for_admission=estimates,distinct_formal_diagnostic_states=240,formal_optimizer_updates=54000,
            actual_optimizer_calls=216,checks_complete_before_admission=True,
            input_and_scoring_checks='Additional pure-data/scoring checks are required by root before final admission; not claimed here')
        result['pass']=True
    except BaseException:
        result['failure']=traceback.format_exc()
        if budget.synthetic_updates>before['synthetic_updates']:budget.mark_discarded(budget.synthetic_updates-before['synthetic_updates'])
        raise
    finally:
        elapsed=time.monotonic()-start
        budget.charge(max(0.,elapsed-(budget.seconds-before['seconds'])),'verification');budget.persist()
        result.update(actual_optimizer_calls=budget.synthetic_optimizer_calls-before['synthetic_optimizer_calls'],
            actual_successful_updates=budget.synthetic_updates-before['synthetic_updates'],charged_seconds=budget.seconds-before['seconds'],
            wall_seconds=elapsed,budget_before=before,budget_after=budget.as_dict())
        (directory/'preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    budget.check();return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);args=parser.parse_args()
    result=run_checks(args.root,Budget(args.root/'budget.json',limit=budget_limit(args.root)))
    print(json.dumps({k:result[k] for k in ('pass','actual_optimizer_calls','charged_seconds','fit_est','setup_est','diagnostic_est')}))
