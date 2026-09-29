"""Unlabelled geometry and strictly isolated residual/RAW diagnostics.

No filesystem or scoring inputs are accepted. Pure saved-record calculations
import only NumPy; PyTorch/model imports occur solely inside model diagnostics.
The caller owns all timing/charging, including failed diagnostic calls.
"""
from __future__ import annotations
import copy
import math
import random

CRITERIA = ('objective_small', 'similarity_stable', 'mean_prediction_stable', 'uncertainty_stable')
TOLERANCES = {'nll_relative': .001, 'kernel_relative': .01,
              'mean_max_abs_raw': .001, 'sd_max_abs_raw': .001}


def query_projection(rows):
    """Copy the public whitelist without inspecting query y/SE, even if present."""
    ids = [r['candidate_id'] for r in rows]
    x = [[float(v) for v in r['x_standardized']] for r in rows]
    if len(set(ids)) != len(ids) or not ids or any(len(r) != 16 for r in x):
        raise ValueError('Query identity/features shape')
    if any(not math.isfinite(v) for r in x for v in r):
        raise ValueError('Nonfinite query features')
    return {'query_ids': ids, 'query_x_standardized': x}


def _array(value, name, ndim=None):
    import numpy as np
    a = np.asarray(value, dtype=np.float64)
    if not a.size or not np.isfinite(a).all() or (ndim is not None and a.ndim != ndim):
        raise ValueError('Invalid finite array: ' + name)
    return a


def _stats(value):
    import numpy as np
    a = _array(value, 'distance').ravel()
    return {'count': int(a.size), 'median': float(np.median(a)),
            'p10': float(np.quantile(a, .1)), 'p90': float(np.quantile(a, .9)),
            'zero_fraction': float(np.mean(a == 0)), 'min': float(a.min()), 'max': float(a.max())}


def _relative(a, b, floor=1e-30):
    import numpy as np
    a, b = _array(a, 'previous'), _array(b, 'current')
    if a.shape != b.shape:
        raise ValueError('Pair identity/shape changed')
    return float(np.linalg.norm(b-a) / max(float(np.linalg.norm(a)), floor))


def _matern52(distance, lengthscale):
    import numpy as np
    r = np.sqrt(5.) * distance / lengthscale
    return (1. + r + r*r/3.) * np.exp(-r)


def gp_parameters(model):
    p = {'mean': float(model.gp.mean_module.constant.detach().item()),
         'lengthscale': float(model.gp.covar_module.base_kernel.lengthscale.detach().item()),
         'outputscale': float(model.gp.covar_module.outputscale.detach().item()),
         'extra_noise_variance_standardized': float(model.gp.likelihood.second_noise.detach().item())}
    if not all(math.isfinite(v) for v in p.values()):
        raise FloatingPointError('Nonfinite GP parameters')
    for name, lower in [('lengthscale',1e-4),('outputscale',1e-8),('extra_noise_variance_standardized',1e-6)]:
        if p[name] < lower:
            raise FloatingPointError('GP parameter below frozen bound: ' + name)
        p[name + '_distance_to_lower'] = p[name] - lower
    return p


def geometry_from_embeddings(train_h, query_h, train_ids, query_ids, gp_parameters=None):
    """Exact pair identities: TT upper triangle, QT query-major/train-minor.

    Positive-distance scale has no epsilon replacement. Ties use candidate ID,
    including constant embeddings. Kernels here are analytic isotropic Matérn5/2
    noiseless K and R; the light GP diagnostic also saves actual model kernels.
    """
    import numpy as np
    th, qh = _array(train_h,'train h',2), _array(query_h,'query h',2)
    train_ids, query_ids = list(train_ids), list(query_ids)
    if (len(train_ids) != len(th) or len(query_ids) != len(qh) or th.shape[1] != qh.shape[1]
            or len(th) < 5 or len(set(train_ids)) != len(th) or len(set(query_ids)) != len(qh)
            or set(train_ids) & set(query_ids)):
        raise ValueError('Training/query identity or representation mismatch')
    tt = np.linalg.norm(th[:,None,:]-th[None,:,:], axis=-1)
    qt = np.linalg.norm(qh[:,None,:]-th[None,:,:], axis=-1)
    pairs = np.triu_indices(len(th), k=1)
    dtt = tt[pairs]
    positive = dtt[dtt > 0]
    scale = float(np.median(positive)) if positive.size else None
    order = [sorted(range(len(th)), key=lambda j:(float(qt[i,j]),train_ids[j])) for i in range(len(qh))]
    nearest = [train_ids[v[0]] for v in order]
    five = [[train_ids[j] for j in v[:5]] for v in order]
    geometry = {'dimensions': int(th.shape[1]), 's_h': scale,
        'constant_representation': scale is None,
        'scale_undefined_reason': 'no strictly positive training pair distance' if scale is None else None,
        'pair_order': {'TT':'upper triangle i<j in training_ids order', 'QT':'query_ids rows, training_ids columns'},
        'd_TT': dtt.tolist(), 'd_QT': qt.tolist(),
        'distance_stats': {'TT':_stats(dtt),'QT':_stats(qt)},
        'normalized_distance_stats': None if scale is None else {'TT':_stats(dtt/scale),'QT':_stats(qt/scale)},
        'd_TT_over_s_h': None if scale is None else (dtt/scale).tolist(),
        'd_QT_over_s_h': None if scale is None else (qt/scale).tolist(),
        'gp_parameters': copy.deepcopy(gp_parameters), 'ell_over_s_h': None,
        'effective_distance_stats': None, 'training_correlation':None,'query_training_correlation':None,
        'training_kernel':None,'query_training_kernel':None}
    rho = most_ids = most_rho = None
    if gp_parameters is not None:
        ell, out = float(gp_parameters['lengthscale']), float(gp_parameters['outputscale'])
        if ell <= 0 or out <= 0 or not all(math.isfinite(v) for v in (ell,out)):
            raise ValueError('Invalid fitted GP scale')
        rtt, rqt = _matern52(tt,ell), _matern52(qt,ell)
        np.fill_diagonal(rtt,1.)
        most = [sorted(range(len(th)),key=lambda j:(-float(rqt[i,j]),train_ids[j]))[0] for i in range(len(qh))]
        rho = [float(rqt[i,v[0]]) for i,v in enumerate(order)]
        most_ids, most_rho = [train_ids[j] for j in most], [float(rqt[i,j]) for i,j in enumerate(most)]
        geometry.update(ell_over_s_h=None if scale is None else ell/scale,
            effective_distance_stats={'TT':_stats(dtt/ell),'QT':_stats(qt/ell)},
            training_correlation=rtt.tolist(),query_training_correlation=rqt.tolist(),
            training_kernel=(out*rtt).tolist(),query_training_kernel=(out*rqt).tolist())
    return {'training_ids':train_ids,'query_ids':query_ids,'nearest_train_ids':nearest,
            'five_nearest_train_ids':five,'nearest_correlation':rho,
            'most_correlated_train_ids':most_ids,'most_correlated_rho':most_rho,
            'geometry':geometry,'query_labels_or_SE_present':False}


def geometry_comparison(previous, current):
    import numpy as np
    if previous['training_ids'] != current['training_ids'] or previous['query_ids'] != current['query_ids']:
        raise ValueError('Geometry comparison requires identical candidate pair order')
    a,b = previous['geometry'],current['geometry']
    shape = {}
    for key in ('TT','QT'):
        da,db = a['d_'+key+'_over_s_h'],b['d_'+key+'_over_s_h']
        shape[key] = None if da is None or db is None else _relative(da,db,1e-12)
    result = {'shape_change':shape,
        'shape_undefined_reason':'constant representation has undefined positive-distance scale' if None in shape.values() else None,
        'nearest_changed_fraction':float(np.mean([x!=y for x,y in zip(previous['nearest_train_ids'],current['nearest_train_ids'])])),
        'five_neighbor_set_changed_fraction':float(np.mean([set(x)!=set(y) for x,y in zip(previous['five_nearest_train_ids'],current['five_nearest_train_ids'])])),
        'five_neighbor_order_changed_fraction':float(np.mean([x!=y for x,y in zip(previous['five_nearest_train_ids'],current['five_nearest_train_ids'])])),
        'K_relative_change':None,'R_relative_change':None,'outputscale_relative_change':None,
        'outputscale_signed_change':None}
    if a['gp_parameters'] is not None and b['gp_parameters'] is not None:
        result['K_relative_change'] = {'TT':_relative(a['training_kernel'],b['training_kernel']),
                                      'QT':_relative(a['query_training_kernel'],b['query_training_kernel'])}
        result['R_relative_change'] = {'TT':_relative(a['training_correlation'],b['training_correlation']),
                                      'QT':_relative(a['query_training_correlation'],b['query_training_correlation'])}
        oa,ob=a['gp_parameters']['outputscale'],b['gp_parameters']['outputscale']
        result['outputscale_relative_change']=abs(ob-oa)/max(abs(oa),1e-30)
        result['outputscale_signed_change']=ob-oa
    return result


def _module_roots(model):
    return [(name,getattr(model,name)) for name in ('mapper','gp') if hasattr(model,name)]


def _detach_shared_modules(live, isolated):
    """Replace shared GPyTorch Softplus singleton before any train/eval call."""
    live_ids={id(m) for _,r in _module_roots(live) for m in r.modules()}
    detached=[]
    def visit(parent,path):
        for name,child in list(parent._modules.items()):
            if child is None: continue
            where=path+'.'+name
            if id(child) in live_ids:
                child=copy.deepcopy(child);setattr(parent,name,child);detached.append(where)
            visit(child,where)
    for name,root in _module_roots(isolated):
        if id(root) in live_ids:raise RuntimeError('Shared diagnostic root module')
        visit(root,name)
    if live_ids & {id(m) for _,r in _module_roots(isolated) for m in r.modules()}:
        raise RuntimeError('Diagnostic copy retains shared modules')
    return detached


def diagnose_isolated(model, query_x, query_ids, y_center, y_scale, block_step,
                      mode='gp', with_geometry=False, training_ids=None,
                      last_gradient=None, lrs=None):
    """Current post-update state, entirely on a clone and with RNG restored.

    Regression mode never asks the GP for a prediction. GP/joint modes save actual
    noiseless K, R, raw posterior variance and both prescribed SD definitions.
    """
    import numpy as np
    import torch
    import gpytorch
    from stage2_core import numerical_settings
    from residual_core import spectral_stats
    if mode not in ('regression','gp','joint') or int(block_step)<0:
        raise ValueError('Diagnostic block/mode')
    query_ids=list(query_ids)
    if len(query_ids)<5 or len(set(query_ids))!=len(query_ids) or query_x.shape!=(len(query_ids),16):
        raise ValueError('Diagnostic query shape/identity')
    if query_x.dtype!=torch.float64 or not bool(torch.isfinite(query_x).all()):
        raise ValueError('Diagnostic query must be finite float64')
    if not all(math.isfinite(v) for v in (y_center,y_scale)) or y_scale<=0:
        raise ValueError('Invalid training scaler')
    if with_geometry and training_ids is None:
        raise ValueError('Endpoint geometry requires real selected training IDs')
    if training_ids is not None and len(training_ids)!=len(model.y):
        raise ValueError('Training identity length')
    rng=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
    fingerprint=model.fingerprint()
    try:
        isolated=model.clone()
        detached=_detach_shared_modules(model,isolated)
        if isolated.fingerprint()!=fingerprint:raise RuntimeError('Diagnostic clone changed mathematical state')
        with torch.no_grad(), numerical_settings():
            th,head=isolated.encode(isolated.x)
            qh,_=isolated.encode(query_x.detach().clone())
            if not all(bool(torch.isfinite(t).all()) for t in (th,qh)):
                raise FloatingPointError('Nonfinite representation')
            # Even light diagnostics retain scalar distance/scale information.
            geom=geometry_from_embeddings(th.numpy(),qh.numpy(),
                training_ids if training_ids is not None else ['__train_%06d'%i for i in range(len(th))],
                query_ids, None if mode=='regression' else gp_parameters(isolated))
            result={'block_step':int(block_step),'mode':mode,'state_hash':fingerprint,
                'query_ids':query_ids,'training_ids':None if training_ids is None else list(training_ids),
                'gradient_before_last_update':last_gradient,'learning_rates':copy.deepcopy(lrs),
                'query_labels_or_SE_present':False,'diagnostic_model_isolated':True,
                'diagnostic_shared_module_paths_detached':detached,
                'geometry_summary':{k:geom['geometry'][k] for k in ('dimensions','s_h','constant_representation',
                    'scale_undefined_reason','distance_stats','normalized_distance_stats','ell_over_s_h','effective_distance_stats')},
                'gp_parameters':gp_parameters(isolated),
                'spectral':spectral_stats(isolated,enforce=True)}
            if with_geometry:
                raw_geometry=geometry_from_embeddings(isolated.x.numpy(),query_x.detach().numpy(),
                    training_ids,query_ids)
                geom['comparisons_to_raw']=geometry_comparison(raw_geometry,geom)
                result['geometry_row']=geom
            if mode=='regression':
                objective,components=isolated.objective('regression')
                if head is None:raise ValueError('Regression diagnostics need a regression head')
                mse=(head-isolated.y).square().mean()
                if not torch.equal(objective,mse):raise RuntimeError('Regression objective is not pure MSE')
                error=(head-isolated.y)*y_scale
                result.update(objective_after_update=float(objective),mse_after_update=float(mse),
                    nll_after_update=None,train_mae_raw=float(error.abs().mean()),
                    train_rmse_raw=float(error.square().mean().sqrt()),
                    train_head_mean_standardized=head.tolist(),
                    train_head_mean_raw=(head*y_scale+y_center).tolist(),
                    prediction_status='not_computed_for_intermediate_regression')
            else:
                isolated.gp.train();isolated.gp.likelihood.train()
                isolated.gp.set_train_data(inputs=th,targets=isolated.y,strict=False)
                mll=gpytorch.mlls.ExactMarginalLogLikelihood(isolated.gp.likelihood,isolated.gp)
                nll=-mll(isolated.gp(th),isolated.y)
                combined_h=torch.cat((th,qh),dim=0)
                tk=isolated.gp.covar_module(th).to_dense()
                ck=isolated.gp.covar_module(combined_h).to_dense()
                mean,var=isolated.predict(torch.cat((isolated.x,query_x.detach().clone()),dim=0))
                if not all(bool(torch.isfinite(t).all()) for t in (nll,tk,ck,mean,var)):
                    raise FloatingPointError('Nonfinite GP diagnostic')
                if bool((var < -1e-8).any()):raise FloatingPointError('Raw latent variance below -1e-8')
                nt=len(th);tm,qm=mean[:nt],mean[nt:];tv,qv=var[:nt],var[nt:]
                p=result['gp_parameters'];extra=p['extra_noise_variance_standardized'];out=p['outputscale']
                raw_mean=qm*y_scale+y_center
                order=sorted(range(len(query_ids)),key=lambda i:(-float(raw_mean[i]),query_ids[i]))
                error=(tm-isolated.y)*y_scale
                result.update(objective_after_update=float(nll),nll_after_update=float(nll),
                    nll_unit='pure data negative log marginal likelihood per selected observation, standardized labels',
                    train_mean_standardized=tm.tolist(),train_mean_raw=(tm*y_scale+y_center).tolist(),
                    query_mean_standardized=qm.tolist(),query_mean_raw=raw_mean.tolist(),
                    train_latent_variance_standardized=tv.tolist(),query_latent_variance_standardized=qv.tolist(),
                    train_latent_variance_raw=(tv*y_scale*y_scale).tolist(),query_latent_variance_raw=(qv*y_scale*y_scale).tolist(),
                    query_latent_sd_raw=(qv.clamp_min(0).sqrt()*y_scale).tolist(),
                    query_model_sd_raw=((qv.clamp_min(0)+extra).sqrt()*y_scale).tolist(),
                    extra_noise_variance_standardized=extra,y_center=float(y_center),y_scale=float(y_scale),
                    training_kernel=tk.tolist(),combined_kernel=ck.tolist(),
                    training_correlation=(tk/out).tolist(),combined_correlation=(ck/out).tolist(),
                    kernel_definition='noiseless K; combined order training_ids then query_ids; R=K/outputscale',
                    train_mae_raw=float(error.abs().mean()),train_rmse_raw=float(error.square().mean().sqrt()),
                    top5_ids=[query_ids[i] for i in order[:5]],top1_id=query_ids[order[0]],
                    negative_roundoff_variance_count_train=int((tv<0).sum()),negative_roundoff_variance_count_query=int((qv<0).sum()))
            # Check every numeric output, while preserving deliberate None fields.
            def finite_tree(v):
                if isinstance(v,dict):return all(finite_tree(x) for x in v.values())
                if isinstance(v,(list,tuple)):return all(finite_tree(x) for x in v)
                return not isinstance(v,float) or math.isfinite(v)
            if not finite_tree(result):raise FloatingPointError('Nonfinite serialized diagnostic')
        if model.fingerprint()!=fingerprint:raise RuntimeError('Diagnostic changed live mathematical state')
        return result
    finally:
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])


def window(previous,current):
    """One block-local 100-update window; outputscale/R drift is descriptive."""
    import numpy as np
    a,b=int(previous['block_step']),int(current['block_step'])
    if b-a!=100:raise ValueError('Stability requires a 100-update interval')
    if previous['query_ids']!=current['query_ids'] or previous.get('training_ids')!=current.get('training_ids'):
        raise ValueError('Diagnostic candidate order changed')
    nlla,nllb=float(previous['nll_after_update']),float(current['nll_after_update'])
    if not all(math.isfinite(v) for v in (nlla,nllb)):raise ValueError('Nonfinite NLL')
    def delta(key):
        x,y=_array(previous[key],key),_array(current[key],key)
        if x.shape!=y.shape:raise ValueError('Diagnostic vector shape changed')
        return float(np.max(np.abs(y-x)))
    outa,outb=previous['gp_parameters']['outputscale'],current['gp_parameters']['outputscale']
    metrics={'nll_relative_change':abs(nllb-nlla)/max(1.,abs(nlla)),'nll_signed_change':nllb-nlla,
        'training_kernel_relative_change':_relative(previous['training_kernel'],current['training_kernel']),
        'combined_kernel_relative_change':_relative(previous['combined_kernel'],current['combined_kernel']),
        'training_correlation_relative_change':_relative(np.asarray(previous['training_kernel'])/outa,np.asarray(current['training_kernel'])/outb),
        'combined_correlation_relative_change':_relative(np.asarray(previous['combined_kernel'])/outa,np.asarray(current['combined_kernel'])/outb),
        'outputscale_relative_change':abs(outb-outa)/max(abs(outa),1e-30),'outputscale_signed_change':outb-outa,
        'train_mean_max_abs_change_raw':delta('train_mean_raw'),'query_mean_max_abs_change_raw':delta('query_mean_raw'),
        'query_latent_sd_max_abs_change_raw':delta('query_latent_sd_raw'),'query_model_sd_max_abs_change_raw':delta('query_model_sd_raw')}
    criteria={'objective_small':metrics['nll_relative_change']<=.001,
        'similarity_stable':max(metrics['training_kernel_relative_change'],metrics['combined_kernel_relative_change'])<=.01,
        'mean_prediction_stable':max(metrics['train_mean_max_abs_change_raw'],metrics['query_mean_max_abs_change_raw'])<=.001,
        'uncertainty_stable':max(metrics['query_latent_sd_max_abs_change_raw'],metrics['query_model_sd_max_abs_change_raw'])<=.001}
    old,new=previous['top5_ids'],current['top5_ids']
    return {'from_step':a,'to_step':b,**metrics,'criteria':criteria,'practical_stability_window':all(criteria.values()),
        'top5_selection_stable':set(old)==set(new),'top5_order_stable':old==new,
        'top1_selection_stable':previous['top1_id']==current['top1_id'],'tolerances':dict(TOLERANCES)}


def stability_at(records,step=1000):
    lookup={}
    for r in records:
        t=int(r['block_step'])
        if t in lookup:raise ValueError('Duplicate block diagnostic step')
        lookup[t]=r
    step=int(step);required=[step-300,step-200,step-100,step]
    missing=[v for v in required if v not in lookup]
    if min(required)<0 or missing:
        return {'block_step':step,'practical_stability':None,'criteria':dict.fromkeys(CRITERIA,None),
            'windows_required':3,'windows_available':0,'missing_diagnostic_steps':missing,
            'reason':'three consecutive block-local 100-update windows unavailable','windows':[],
            'top5_selection_stable':None,'top5_order_stable':None,'top1_selection_stable':None}
    windows=[window(lookup[a],lookup[b]) for a,b in zip(required[:-1],required[1:])]
    criteria={k:all(w['criteria'][k] for w in windows) for k in CRITERIA}
    return {'block_step':step,'practical_stability':all(criteria.values()),'criteria':criteria,
        'windows_required':3,'windows_available':3,'missing_diagnostic_steps':[],'reason':None,'windows':windows,
        **{k:all(w[k] for w in windows) for k in ('top5_selection_stable','top5_order_stable','top1_selection_stable')}}
