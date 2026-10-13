"""Seen-only W-I fitting and real finite-pool qLogEI decisions."""
import time
import warnings
from pathlib import Path
import numpy as np
from .common import read, write, seal, sha, digest, seed, now, folder as trajectory_folder, inputs, KERNELS

def training_projection(public, observations):
    positions={cid:i for i,cid in enumerate(public['ids'])}; ids=[o['candidate_id'] for o in observations]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate observed configuration')
    indexes=[positions[cid] for cid in ids]
    x=np.asarray(public['x_raw'],dtype=np.float64)[indexes];wl=np.asarray(public['wl'],dtype=np.float64)[indexes]
    y=np.asarray([o['validation_accuracy'] for o in observations],dtype=np.float64)
    xm=x.mean(0);xs=x.std(0);constant=np.ptp(x,axis=0)==0;xs[constant]=1.
    ym=float(y.mean());ys=float(y.std())
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(ys) or ys<=0 or np.ptp(y)==0:
        raise ValueError('Nonfinite inputs or zero seen-only target scale')
    return {'ids':ids,'x_raw':x.tolist(),'x_standardized':((x-xm)/xs).tolist(),'wl':wl.tolist(),
       'y_standardized':((y-ym)/ys).tolist(),
       'scaler':{'x_mean':xm.tolist(),'x_scale':xs.tolist(),'y_mean':ym,'y_scale':ys,'fit_ids':ids,'ddof':0,
           'constant_columns':np.flatnonzero(constant).tolist()},
       'known_SE_provided':False,'observation_semantics':'one fixed training seed per configuration'}

def decision(root, method, group, observations, initial_count=50):
    import torch
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.acquisition.logei import qLogExpectedImprovement
    from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter
    from . import gp
    folder=trajectory_folder(root,method,group); n=len(observations);step=n-initial_count+1
    if not 1<=step<=250:raise ValueError('A/C online step must be 1..250')
    path=folder/'decisions'/f'{step:03d}.json';public=read(inputs(root,group)/'public_features.json')
    observed_ids=[o['candidate_id'] for o in observations]
    if path.exists():
        saved=read(path)
        if saved['observed_ids']!=observed_ids or saved['observations_identity']!=digest(observations):raise ValueError('Decision observations changed')
        if sha(folder/saved['fit_checkpoint'])!=saved['model_sha256']:raise ValueError('Decision model changed')
        return saved
    gp.cpu_setup();training=training_projection(public,observations);remaining=sorted(set(public['ids'])-set(observed_ids))
    positions={cid:i for i,cid in enumerate(public['ids'])};qi=[positions[cid] for cid in remaining]
    query={k:[public[k][i] for i in qi] for k in ('ids','x_raw','wl')}
    x,q,arch,scales=gp.make_inputs(training,query,KERNELS[method])
    identity=digest({'freeze':read(root/'freeze.json')['identity'],'method':method,'kernel':KERNELS[method],'group':group,'observations':observations,'scales':scales,'scaler':training['scaler']})
    fdir=folder/'fits'/f'n{n:03d}';fdir.mkdir(parents=True,exist_ok=True)
    seal(fdir/'training.json',training)
    started=time.perf_counter()
    if (fdir/'fit.json').exists():
        fit=read(fdir/'fit.json')
        if fit['identity']!=identity:raise ValueError('Fit identity changed')
    else:
        results=[]
        for start in range(3):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                r=gp.fit_start(x,training['y_standardized'],KERNELS[method],arch,start,fdir/f'start{start}.pt',identity,resume=True)
            r['python_warnings']=r.get('python_warnings',[])+[{'category':w.category.__name__,'message':str(w.message)} for w in caught]
            write(fdir/f'start{start}.json',r);results.append(r)
        if not all(r['success'] and np.isfinite(r['final_nll']) for r in results):
            write(fdir/'failed_fit.json',{'identity':identity,'starts':results});raise RuntimeError('A/C GP start failure; all starts preserved')
        selected=gp.select_start(results)
        fit={'identity':identity,'n':n,'selected_start':selected,'scaler':training['scaler'],'distance_scales':scales,
            'starts':results,'fit_seconds':sum(r['elapsed_seconds'] for r in results),'ended_at':now()}
        seal(fdir/'fit.json',fit)
    checkpoint=fdir/f"start{fit['selected_start']}.pt"
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    adapter=FrozenAdapter(gp.restore_model(state),state['jitter_index'])
    scaler=training['scaler'];best_f=(max(o['validation_accuracy'] for o in observations)-scaler['y_mean'])/scaler['y_scale']
    mc=seed(group,'mc',step);sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]),seed=mc)
    acquisition=qLogExpectedImprovement(model=adapter,best_f=float(best_f),sampler=sampler,fat=True,tau_max=.01,tau_relu=1e-6,eta=.001)
    scores=[];means=[];stds=[];acquisition_started=time.perf_counter()
    with torch.no_grad():
        for chunk in q.split(64):
            batched=chunk.unsqueeze(-2);posterior=adapter.posterior(batched)
            mean=posterior.mean[:,0,0];variance=posterior.distribution.lazy_covariance_matrix.diagonal(dim1=-2,dim2=-1)[:,0]
            if bool((variance<0).any()):raise FloatingPointError('Negative latent posterior variance')
            means.extend((mean*scaler['y_scale']+scaler['y_mean']).tolist())
            stds.extend((variance.sqrt()*scaler['y_scale']).tolist());scores.extend(acquisition(batched).tolist())
    if not np.isfinite([scores,means,stds]).all() or len(scores)!=len(remaining):raise FloatingPointError('Invalid finite-pool acquisition')
    pick=min(range(len(remaining)),key=lambda i:(-scores[i],remaining[i]))
    sampler_path=folder/'decisions'/f'{step:03d}_sampler.pt';gp._save_torch(sampler_path,{'mc_seed':mc,'base_samples':sampler.base_samples.detach().clone()})
    saved={'method':method,'kernel':KERNELS[method],'group':group,'step':step,'kind':'qlogei','observed_n':n,'observed_ids':observed_ids,
        'observations_identity':digest(observations),'fit_identity':identity,'fit_checkpoint':str(checkpoint.relative_to(folder)),
        'model_sha256':sha(checkpoint),'remaining_ids':remaining,'posterior_mean_accuracy':means,
        'posterior_latent_std_accuracy':stds,'qlogei':scores,'qlogei_units':'log expected improvement; not accuracy',
        'candidate_id':remaining[pick],'best_observed_validation':max(o['validation_accuracy'] for o in observations),
        'parameters':fit['starts'][fit['selected_start']]['parameters'],'mc_seed':mc,'sampler_sha256':sha(sampler_path),
        'fit_seconds':fit['fit_seconds'],'fit_wall_seconds_this_call':acquisition_started-started,
        'acquisition_seconds':time.perf_counter()-acquisition_started,'numerical_events':adapter.policy.events,'sealed_at':now()}
    seal(path,saved);return saved
