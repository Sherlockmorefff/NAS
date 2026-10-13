"""No optimizer imports or calls: restore 86 frozen states and decompose means."""
import math
import numpy as np
import torch
import gpytorch
from .common import *
from analyse.structured_kernel_phase1.gp import cpu_setup,make_inputs,restore_model,NumericalPolicy,parameter_values,tensor


def rank(ids,values,cid):
    if cid not in ids:return None
    values=np.asarray(values);target=values[ids.index(cid)]
    return 1+sum(v>target+1e-12 or (abs(v-target)<=1e-12 and c<cid) for c,v in zip(ids,values))


def mean_decomposition(model,x,q,y,known,scale,center,jitter):
    with torch.no_grad():
        base=model.covar_module.base_kernel;sf=float(model.covar_module.outputscale)
        ka=base.arch(q,x).to_dense();kh=base.hp(q,x).to_dense()
        parts={'architecture':.25*sf*ka,'hp':.25*sf*kh,'interaction':.5*sf*ka*kh}
        cov=model.covar_module(x).to_dense()+torch.diag(known+model.likelihood.second_noise.reshape(()))
        used=0.
        try:chol=torch.linalg.cholesky(cov)
        except torch.linalg.LinAlgError:
            used=jitter;chol=torch.linalg.cholesky(cov+torch.eye(len(x))*jitter)
        m=model.mean_module.constant.reshape(());residual=y-m
        alpha=torch.cholesky_solve(residual[:,None],chol)
        component={};points={}
        for name,k in parts.items():
            component[name]=float((k@alpha).squeeze())*100*scale
            beta=torch.cholesky_solve(k.T,chol).squeeze()
            points[name]={'contributions':(100*scale*beta*residual).tolist(),'weights':beta.tolist(),'kernel':k.squeeze().tolist()}
        baseline=100*(center+scale*float(m))
        return {'baseline_pp':baseline,'components_pp':component,'mean_pp':baseline+sum(component.values()),
            'used_jitter':used,'ka':ka.squeeze().tolist(),'kh':kh.squeeze().tolist(),'points':points}


def run(root,repo):
    if (root/'M1_summary.json').exists() and read_json(root/'M1_summary.json')['state']=='COMPLETE':return read_json(root/'M1_summary.json')
    cpu_setup();set_status(root,'M1','RUNNING')
    public=read_json(repo/REPLAY/'public_features.json');idx={c:i for i,c in enumerate(public['ids'])}
    truth=read_json(repo/REPLAY/'summary.json')['pool_optimum_validation_mean']
    traces=[];components=[];points=[];validations=[];failures=[]
    for seed,method,n in models():
        key=f'{seed}_{method}_n{n}'
        try:
            folder=repo/REPLAY/'trajectories'/f'{seed}_{method}'
            fdir=folder/'fits'/f'n{n:03d}';fit=read_json(fdir/'fit.json');training=read_json(fdir/'training.json')
            obs=read_json(folder/'observations.json');decision=read_json(folder/'decisions'/f'{n-49:02d}.json')
            state=torch.load(fdir/fit['selected_model'],map_location='cpu',weights_only=False)
            x,q,arch,scales=make_inputs(training,public,method)
            assert torch.equal(x,state['x']) and torch.equal(tensor(training['y_standardized']),state['y'])
            assert torch.equal(tensor(training['known_variance_standardized']),state['known']) and scales==fit['distance_scales']
            model=restore_model(state);policy=NumericalPolicy(state['jitter_index'])
            model.train();model.likelihood.train()
            mll=gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)
            with torch.no_grad():nll=float(policy.evaluate(lambda:-mll(model(x),state['y']),model,'reconstruction_nll'))
            nllerr=abs(nll-fit['starts'][fit['selected_start']]['final_nll'])
            if nllerr>1e-8:raise ValueError(f'NLL reconstruction mismatch {nllerr}')
            model.eval();model.likelihood.eval();model.prediction_strategy=None
            def predict():
                p=model(q);return p.mean.detach(),p.lazy_covariance_matrix.diagonal().detach()
            with torch.no_grad():mu,var=policy.evaluate(predict,model,'frozen_posterior')
            if not torch.isfinite(mu).all() or not torch.isfinite(var).all() or torch.any(var<0):raise ValueError('Invalid posterior')
            scale=training['scaler']['y_scale'];center=training['scaler']['y_mean']
            raw=center+scale*mu.numpy();rawvar=scale**2*var.numpy();extra=float(model.likelihood.second_noise)
            prospective=rawvar+scale**2*(extra+float(np.median(training['known_variance_standardized'])))
            remaining=decision['candidate_ids'];rmu=[raw[idx[c]] for c in remaining]
            greedy_error=None
            if n==70:
                greedy_error=float(np.max(np.abs(np.asarray(rmu)-decision['scores'])))
                if greedy_error>1e-8:raise ValueError(f'Greedy posterior mismatch {greedy_error}')
            champ=idx[CHAMPION];selected=decision['candidate_id'];observed=CHAMPION in training['ids']
            saved_score=decision['scores'][remaining.index(CHAMPION)] if CHAMPION in remaining else None
            chosen_score=decision['scores'][remaining.index(selected)]
            tr={'seed':seed,'method':method,'n':n,'decision':n-49,'decision_kind':decision['kind'],
                'champion_observed':observed,'champion_true_percent':100*truth,'champion_mu_percent':100*raw[champ],
                'latent_sd_pp':100*math.sqrt(rawvar[champ]),'prospective_sd_pp':100*math.sqrt(prospective[champ]),
                'bias_pp':100*(raw[champ]-truth),'remaining_mean_rank':rank(remaining,rmu,CHAMPION),
                'sealed_qlogei_score':saved_score if n<70 else None,'sealed_qlogei_rank':rank(remaining,decision['scores'],CHAMPION) if n<70 else None,
                'sealed_greedy_raw_mean':saved_score if n==70 else None,'sealed_greedy_mean_rank':rank(remaining,decision['scores'],CHAMPION) if n==70 else None,
                'selected_minus_champion_score':chosen_score-saved_score if saved_score is not None else None,
                'incumbent_percent':100*max(o['mean'] for o in obs[:n]),'selected_id':selected,
                'selected_mu_percent':100*raw[idx[selected]],'selected_revealed_percent':100*obs[n]['mean'],
                'jitter_setting':policy.jitter,'jitter_events':policy.events,'parameters':parameter_values(model)}
            traces.append(tr)
            validations.append({'model':key,'state':'PASS','NLL_abs_error':nllerr,'n70_mean_max_abs_error':greedy_error,
                'training_tensors_exact':True,'checkpoint_sha256':sha256(fdir/fit['selected_model'])})
            if method=='W-I' and n in (50,60,70):
                d=mean_decomposition(model,x,q[champ:champ+1],state['y'],state['known'],scale,center,policy.jitter)
                error=abs(d['mean_pp']/100-raw[champ])
                if error>1e-8:raise ValueError('Mean decomposition mismatch')
                components.append({'seed':seed,'n':n,'baseline_percent':d['baseline_pp'],
                    **{k+'_contribution_pp':v for k,v in d['components_pp'].items()},'sum_percent':d['mean_pp'],
                    'full_posterior_percent':100*raw[champ],'raw_abs_error':error,'actual_diagonal_jitter':d['used_jitter']})
                for name,p in d['points'].items():
                    top=sorted(range(n),key=lambda i:(-p['contributions'][i],training['ids'][i]))
                    low=sorted(range(n),key=lambda i:(p['contributions'][i],training['ids'][i]))
                    for i,cid in enumerate(training['ids']):
                        points.append({'seed':seed,'n':n,'component':name,'candidate_id':cid,
                            'observed_validation_percent':100*obs[i]['mean'],'contribution_pp':p['contributions'][i],
                            'weight_beta':p['weights'][i],'centered_y_standardized':float(state['y'][i]-model.mean_module.constant.detach()),
                            'architecture_similarity':d['ka'][i],'hp_similarity':d['kh'][i],'component_cross_kernel':p['kernel'][i],
                            'top5_positive_rank':top.index(i)+1 if i in top[:5] else None,
                            'top5_negative_rank':low.index(i)+1 if i in low[:5] else None})
        except Exception as exc:
            failures.append({'model':key,'type':type(exc).__name__,'error':str(exc)})
    write_csv(root/'champion_trace.csv',traces);write_csv(root/'kernel_mean_contributions.csv',components)
    write_csv(root/'training_point_contributions.csv',points);write_json(root/'posterior_reconstruction_validation.json',{'models':validations,'failures':failures})
    state='COMPLETE' if len(validations)==86 and not failures and len(components)==12 else 'INCOMPLETE'
    summary={'state':state,'verified_models':len(validations),'expected_models':86,'decompositions':len(components),'new_GP_fits':0,'failures':failures}
    write_json(root/'M1_summary.json',summary);set_status(root,'M1',state,**{k:v for k,v in summary.items() if k!='state'})
    plot(root,traces)
    return summary


def plot(root,traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for seed in M1_SEEDS:
        rows=[r for r in traces if r['seed']==seed and r['method']=='W-I']
        if not rows:continue
        x=[r['n'] for r in rows];mu=np.array([r['champion_mu_percent'] for r in rows]);sd=np.array([r['latent_sd_pp'] for r in rows])
        line,=axes[0].plot(x,mu,label=str(seed))
        axes[0].fill_between(x,mu-1.645*sd,mu+1.645*sd,alpha=.12,color=line.get_color())
        q=[r for r in rows if r['sealed_qlogei_rank'] is not None]
        axes[1].plot([r['n'] for r in q],[r['sealed_qlogei_rank'] for r in q],label=str(seed))
    axes[0].axhline(81.53333333333332,color='black',ls='--',label='Cached champion')
    axes[0].set(xlabel='Observed n',ylabel='Champion posterior mean (%)',title='Frozen GP mean and 90% latent interval')
    axes[1].invert_yaxis();axes[1].set(xlabel='Observed n (n70 is greedy, excluded)',ylabel='Sealed qLogEI rank',title='Remaining-pool ranking; observed champion = NA')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'M1_champion_diagnosis.png',dpi=160);plt.close(fig)
