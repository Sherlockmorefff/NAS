"""Isolated fixed-readout diagnostics and pure three-window recomputation.

No query observation/SE input exists. The caller owns timing and budget checks.
Kernels and full covariance may be extracted losslessly to NPZ by the runner.
"""
from __future__ import annotations
import copy
import math
import random

DIAGNOSTIC_STEPS=(0,1000,1200,1300,1400,1500)
CRITERIA=('mll_stable','mean_stable','latent_sd_stable','kernels_stable')


def _array(value,name):
    import numpy as np
    a=np.asarray(value,dtype=np.float64)
    if not a.size or not np.isfinite(a).all():raise ValueError('Invalid finite diagnostic array: '+name)
    return a


def _detach_shared(live,isolated):
    live_ids={id(m) for m in live.gp.modules()};detached=[]
    def walk(parent,path):
        for name,child in list(parent._modules.items()):
            if child is None:continue
            namepath=path+'.'+name
            if id(child) in live_ids:
                child=copy.deepcopy(child);setattr(parent,name,child);detached.append(namepath)
            walk(child,namepath)
    if id(isolated.gp) in live_ids:raise RuntimeError('Readout diagnostic shares GP root')
    walk(isolated.gp,'gp')
    if live_ids & {id(m) for m in isolated.gp.modules()}:raise RuntimeError('Readout diagnostic retained shared mutable modules')
    return detached


def _live_signature(model):
    from stage01_core import tensor_hash
    inputs=model.gp.train_inputs or ()
    return {'fingerprint':model.fingerprint(),'module_modes':{k:v.training for k,v in model.gp.named_modules()},
        'requires_grad':{k:p.requires_grad for k,p in model.gp.named_parameters()},
        'input_pointers':[v.untyped_storage().data_ptr() for v in inputs],
        'input_values':tensor_hash({str(i):v for i,v in enumerate(inputs)}),
        'target_pointer':model.gp.train_targets.untyped_storage().data_ptr(),
        'target_values':tensor_hash({'targets':model.gp.train_targets}),
        'prediction_strategy_identity':id(model.gp.prediction_strategy)}


def diagnose(model,hq,sq,query_ids,y_center,y_scale,step):
    """Current NLL/posterior on a private copy; never changes training/cache/RNG."""
    import numpy as np
    import torch
    from stage01_core import tensor_hash
    query_ids=list(query_ids)
    if int(step) not in DIAGNOSTIC_STEPS or len(query_ids)<5 or len(set(query_ids))!=len(query_ids):
        raise ValueError('Fixed diagnostic step/query identity')
    if hq.shape!=(len(query_ids),16) or sq.shape!=(len(query_ids),):raise ValueError('Frozen query h0/s0 shape')
    if not all(v.dtype==torch.float64 and not v.requires_grad and bool(torch.isfinite(v).all()) for v in (hq,sq)):
        raise ValueError('Diagnostics accept only frozen finite float64 h0/s0')
    if not all(math.isfinite(v) for v in (y_center,y_scale)) or y_scale<1e-8:raise ValueError('Invalid training y scaler')
    rng=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
    live=_live_signature(model)
    try:
        isolated=model.clone();detached=_detach_shared(model,isolated)
        if isolated.fingerprint()!=live['fingerprint']:raise RuntimeError('Readout diagnostic clone identity')
        with torch.no_grad():
            nll=isolated.nll()
            mean,cov=isolated.predict(hq.detach().clone(),sq.detach().clone())
            kernels=isolated.kernels(isolated.h_train,isolated.s_train)
            params=isolated.hyperparameters()
            tensors=[nll,mean,cov,kernels['matern'],kernels['full']]
            if not all(bool(torch.isfinite(t).all()) for t in tensors):raise FloatingPointError('Nonfinite readout diagnostic')
            if mean.shape!=(len(query_ids),) or cov.shape!=(len(query_ids),len(query_ids)):
                raise ValueError('Readout full covariance shape')
            asym=float((cov-cov.T).abs().max())
            mineig=float(torch.linalg.eigvalsh((cov+cov.T)*.5).min())
            if asym>1e-10 or mineig < -1e-8:raise FloatingPointError('Readout posterior covariance symmetry/PSD failure')
            var=cov.diagonal().clone()
            if bool((var < -1e-8).any()):raise FloatingPointError('Raw standardized latent variance below -1e-8')
            raw=mean*y_scale+y_center;sd=var.clamp_min(0).sqrt()*y_scale
            order=sorted(range(len(query_ids)),key=lambda i:(-float(raw[i]),query_ids[i]))
            gap=float(raw[order[4]]-raw[order[5]]) if len(order)>5 else None
            result={'step':int(step),'method':isolated.method,'state_hash':live['fingerprint'],
                'query_ids':query_ids,'nll_after_update':float(nll),
                'nll_unit':'pure negative exact marginal log likelihood per selected observation, standardized labels',
                'query_mean_standardized':mean.tolist(),'query_mean_raw':raw.tolist(),
                'query_latent_covariance_standardized':cov.tolist(),'query_latent_covariance_raw':(cov*y_scale*y_scale).tolist(),
                'query_latent_variance_standardized':var.tolist(),'query_latent_variance_raw':(var*y_scale*y_scale).tolist(),
                'query_latent_sd_raw':sd.tolist(),
                'matern_training_kernel':kernels['matern'].tolist(),'full_training_kernel':kernels['full'].tolist(),
                'kernel_definition':'noiseless training latent kernels; full includes integrated coefficient kernel exactly once',
                'gp_parameters':params,'extra_noise_variance_standardized':params['extra_noise_variance_standardized'],
                'y_center':float(y_center),'y_scale':float(y_scale),
                'top5_ids':[query_ids[i] for i in order[:5]],'top1_id':query_ids[order[0]],
                'ranking_ids':[query_ids[i] for i in order],'top5_boundary_gap_raw':gap,
                'top5_boundary_gap_pp':None if gap is None else 100*gap,
                'latent_covariance_symmetry_max_abs':asym,'latent_covariance_min_eigenvalue':mineig,
                'negative_roundoff_variance_count':int((var<0).sum()),
                'frozen_input_sha256':tensor_hash({'train_h0':isolated.h_train,'train_s0':isolated.s_train,
                    'query_h0':hq,'query_s0':sq}),
                'query_labels_or_SE_present':False,'diagnostic_model_isolated':True,
                'diagnostic_shared_module_paths_detached':detached,'semi_decomposition':None}
            if isolated.method=='GP_SEMI':
                semi=isolated.semi_posterior(hq,sq)
                beta,cov_beta=semi['beta_mean'],semi['beta_cov']
                trend,residual=semi['trend_mean'],semi['residual_mean']
                if not all(bool(torch.isfinite(t).all()) for t in (beta,cov_beta,trend,residual)):
                    raise FloatingPointError('Nonfinite coefficient posterior')
                if not torch.allclose(trend+residual,mean,atol=1e-7,rtol=1e-7):
                    raise RuntimeError('Coefficient trend plus Matern residual does not reconstruct unified posterior mean')
                if float(cov_beta.diagonal().min()) < -1e-8:raise FloatingPointError('Negative coefficient posterior variance')
                result['semi_decomposition']={
                    'beta_posterior_mean':beta.tolist(),'beta_posterior_covariance':cov_beta.tolist(),
                    'beta_posterior_sd':cov_beta.diagonal().clamp_min(0).sqrt().tolist(),
                    'trend_mean_standardized':trend.tolist(),'matern_residual_mean_standardized':residual.tolist(),
                    'trend_mean_raw':(trend*y_scale+y_center).tolist(),'matern_residual_mean_raw':(residual*y_scale).tolist(),
                    'solver_jitter':float(semi['solver_jitter']),
                    'variance_definition':'full unified latent covariance retains beta/residual posterior cross-covariance; do not add marginal variances'}
        if _live_signature(model)!=live:raise RuntimeError('Readout diagnostics changed the live model surface')
        return result
    finally:
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])


# Name retained for a caller that wants to make isolation explicit.
diagnose_isolated=diagnose


def window(previous,current):
    import numpy as np
    a,b=int(previous['step']),int(current['step'])
    if b-a!=100 or (a,b) not in ((1200,1300),(1300,1400),(1400,1500)):
        raise ValueError('Only the three prescribed low-LR windows define stability')
    if previous['query_ids']!=current['query_ids'] or previous['method']!=current['method']:
        raise ValueError('Diagnostic identity/order changed')
    la,lb=float(previous['nll_after_update']),float(current['nll_after_update'])
    if not all(math.isfinite(x) for x in (la,lb)):raise ValueError('Nonfinite NLL')
    def maximum(key):
        x,y=_array(previous[key],key),_array(current[key],key)
        if x.shape!=y.shape:raise ValueError('Diagnostic shape changed')
        return float(np.max(np.abs(y-x)))
    def relative(key):
        x,y=_array(previous[key],key),_array(current[key],key)
        if x.shape!=y.shape or x.ndim!=2 or x.shape[0]!=x.shape[1]:raise ValueError('Training kernel shape changed')
        return float(np.linalg.norm(y-x)/max(float(np.linalg.norm(x)),1e-12))
    metrics={'mll_absolute_change':abs(lb-la),'mll_relative_change':abs(lb-la)/max(1.,abs(la)),
        'query_mean_max_abs_change_raw':maximum('query_mean_raw'),
        'query_latent_sd_max_abs_change_raw':maximum('query_latent_sd_raw'),
        'matern_training_kernel_relative_change':relative('matern_training_kernel'),
        'full_training_kernel_relative_change':relative('full_training_kernel')}
    criteria={'mll_stable':metrics['mll_relative_change']<=1e-4,
        'mean_stable':metrics['query_mean_max_abs_change_raw']<=.001,
        'latent_sd_stable':metrics['query_latent_sd_max_abs_change_raw']<=.001,
        'kernels_stable':max(metrics['matern_training_kernel_relative_change'],metrics['full_training_kernel_relative_change'])<=.01}
    return {'from_step':a,'to_step':b,**metrics,'criteria':criteria,
        'numeric_predictive_stable':all(criteria.values()),
        'top5_stable':set(previous['top5_ids'])==set(current['top5_ids']),
        'top5_order_stable':previous['top5_ids']==current['top5_ids'],
        'top5_boundary_gap_raw':current['top5_boundary_gap_raw'],'top5_boundary_gap_pp':current['top5_boundary_gap_pp'],
        'gp_parameters_previous':copy.deepcopy(previous['gp_parameters']),
        'gp_parameters_current':copy.deepcopy(current['gp_parameters'])}


def stability_at(records,step=1500):
    if int(step)!=1500:raise ValueError('Only fixed endpoint1500 has three-window classification')
    lookup={}
    for record in records:
        t=int(record['step'])
        if t in lookup:raise ValueError('Duplicate readout diagnostic step')
        lookup[t]=record
    required=(1200,1300,1400,1500);missing=[i for i in required if i not in lookup]
    if missing:
        return {'step':1500,'numeric_predictive_stable':None,'NUMERIC_PREDICTIVE_STABLE':None,
            'top5_stable':None,'criteria':dict.fromkeys(CRITERIA,None),'windows':[],
            'last_window':None,'missing_steps':missing,'reason':'three prescribed low-LR windows incomplete'}
    windows=[window(lookup[a],lookup[b]) for a,b in zip(required[:-1],required[1:])]
    criteria={k:all(w['criteria'][k] for w in windows) for k in CRITERIA}
    stable=all(criteria.values())
    return {'step':1500,'numeric_predictive_stable':stable,'NUMERIC_PREDICTIVE_STABLE':stable,
        'top5_stable':all(w['top5_stable'] for w in windows),'criteria':criteria,
        'windows':windows,'last_window':windows[-1],'missing_steps':[],'reason':None,
        'top5_boundary_gap_raw':lookup[1500]['top5_boundary_gap_raw'],
        'top5_boundary_gap_pp':lookup[1500]['top5_boundary_gap_pp']}


def run_checks(output_dir,budget):
    """Zero optimizer updates; the root invokes and accounts this one preflight."""
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from readout_core import ReadoutModel,make_synthetic_fixture
    from readout_data import query_projection
    start=time.perf_counter();charged_before=budget.seconds
    result={'pass':False,'synthetic_updates':0,'checks':[]}
    try:
        budget.check();fixture=make_synthetic_fixture(n=20,seed=734)
        public={'ids':fixture['query_ids'],'architectures':['synthetic_arch_'+str(i) for i in range(len(fixture['query_ids']))],
            'x_raw':np.zeros((len(fixture['query_ids']),16)).tolist()}
        projection=query_projection(public)
        for key in ('query_y','query_SE','labels','validation_seeds'):
            poisoned={**public,key:object()}
            try:query_projection(poisoned)
            except ValueError:pass
            else:raise AssertionError('Query observation field crossed the strict boundary: '+key)
        if query_projection(public)!=projection:raise AssertionError('Public projection changed after poisoning attempts')
        result['checks'].append('query_y_SE_other_fields_rejected_without_reading_values')
        for method in ('GP_CONST','GP_TREND','GP_SEMI'):
            model=ReadoutModel(method,fixture['h_train'],fixture['s_train'],fixture['y_std'],fixture['known_std'],fixture['common_gp_state'])
            signature=_live_signature(model)
            rng=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
            a=diagnose(model,fixture['h_query'],fixture['s_query'],fixture['query_ids'],fixture['y_center'],fixture['y_scale'],0)
            b=diagnose(model,fixture['h_query'],fixture['s_query'],fixture['query_ids'],fixture['y_center'],fixture['y_scale'],0)
            if a!=b or _live_signature(model)!=signature:raise AssertionError('Repeated diagnostic/live surface differs')
            if random.getstate()!=rng[0] or not np.array_equal(np.random.get_state()[1],rng[1][1]) or not torch.equal(torch.get_rng_state(),rng[2]):
                raise AssertionError('Diagnostic changed RNG')
            cov=np.asarray(a['query_latent_covariance_standardized']);var=np.asarray(a['query_latent_variance_standardized'])
            if not np.array_equal(cov.diagonal(),var) or not np.allclose(a['query_latent_covariance_raw'],cov*fixture['y_scale']**2,atol=1e-15,rtol=1e-15):
                raise AssertionError('Full covariance/diagonal/raw unit conversion')
            if not np.allclose(a['query_latent_sd_raw'],np.sqrt(np.maximum(var,0))*fixture['y_scale'],atol=1e-15,rtol=1e-15):
                raise AssertionError('Latent SD definition')
            rows=[dict(copy.deepcopy(a),step=t) for t in (1200,1300,1400,1500)]
            if not stability_at(rows)['numeric_predictive_stable']:raise AssertionError('Identical four-point stability should pass')
            rows[-1]['top5_ids']=list(reversed(rows[-1]['top5_ids']))
            if not stability_at(rows)['numeric_predictive_stable'] or not stability_at(rows)['top5_stable']:
                raise AssertionError('Order-only top5 change affected numerical stability')
            rows[-1]['top5_ids'][-1]=fixture['query_ids'][-1]
            if not stability_at(rows)['numeric_predictive_stable']:raise AssertionError('Top5 set change affected numerical stability')
            rows[-1]['query_mean_raw'][0]+=.0011
            if stability_at(rows)['numeric_predictive_stable']:raise AssertionError('One failed window was ignored')
            result['checks'].append(method+':isolation_repeated_prediction_units_three_windows')
        result['pass']=True
        return result
    finally:
        elapsed=time.perf_counter()-start
        residual=max(0.,elapsed-(budget.seconds-charged_before))
        if residual:budget.charge(residual,kind='verification')
        budget.persist();result['charged_seconds']=budget.seconds-charged_before
        Path(output_dir).mkdir(parents=True,exist_ok=True)
        (Path(output_dir)/'readout_diagnostics_checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        budget.check()
