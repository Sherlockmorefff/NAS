"""Post-seal A-only scoring. This module is never imported by the learner."""
import csv
import math
from pathlib import Path
import numpy as np
import torch
import gpytorch
from .common import *
from . import gp
from .learner import validate_complete
from analyse.structured_kernel_phase1 import gp as old_gp
from analyse.structured_kernel_phase1.score import score_prediction,_interval_metrics
from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter,enumerate_scores,choose


def rank(ids,scores,cid):
    if cid not in ids:return None
    x=float(scores[ids.index(cid)])
    return 1+sum(float(v)>x+1e-12 or (abs(float(v)-x)<=1e-12 and c<cid) for c,v in zip(ids,scores))


def fixed_parameter_model(old_state):
    """Replace the complete kernel; load the same five physical parameters."""
    m=gp.StaticExactGP(old_state['x'],old_state['y'],old_state['known'],'W-HI',old_state['arch_dim'])
    m.load_state_dict(old_state['model']);m.prediction_strategy=None
    before=old_gp.parameter_values(old_gp.restore_model(old_state));after=gp.parameter_values(m)
    if before!=after:raise ValueError('Fixed physical parameters changed')
    return m


def posterior(model,q,training,jitter_index=0):
    model.eval();model.likelihood.eval();model.prediction_strategy=None
    policy=gp.NumericalPolicy(jitter_index)
    def calculate():
        p=model(q);return p.mean.detach(),p.lazy_covariance_matrix.diagonal().detach()
    with torch.no_grad():mu,var=policy.evaluate(calculate,model,'post-seal prediction')
    if not torch.isfinite(mu).all() or not torch.isfinite(var).all() or (var < -1e-8).any():raise ValueError('Invalid latent posterior')
    s=training['scaler'];p=gp.parameter_values(model)
    return {'mean_raw':(s['y_mean']+s['y_scale']*mu.numpy()).tolist(),'mean_standardized':mu.tolist(),
        'latent_variance_standardized':var.tolist(),'extra_noise_variance_standardized':p['extra_noise'],
        'median_known_var_standardized':float(np.median(training['known_variance_standardized'])),
        'y_center':s['y_mean'],'y_scale':s['y_scale'],'parameters':p,'jitter_setting':policy.jitter,'jitter_events':policy.events}


def restore(folder,public,method):
    fit=read_json(folder/'fit.json');training=read_json(folder/'training.json');path=folder/fit['selected_model']
    if sha256(path)!=fit['selected_model_sha256']:raise ValueError('Checkpoint hash changed')
    state=torch.load(path,map_location='cpu',weights_only=False)
    x,q,arch,scales=old_gp.make_inputs(training,public,method)
    if not torch.equal(x,state['x']) or not torch.equal(gp.tensor(training['y_standardized']),state['y']) or not torch.equal(gp.tensor(training['known_variance_standardized']),state['known']) or scales!=fit['distance_scales']:raise ValueError('Training/scaler/distance mismatch')
    model=gp.restore_model(state) if method=='W-HI' else old_gp.restore_model(state)
    model.train();model.likelihood.train();policy=gp.NumericalPolicy(state['jitter_index'])
    mll=gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)
    with torch.no_grad():nll=float(policy.evaluate(lambda:-mll(model(x),state['y']),model,'NLL reconstruction'))
    err=abs(nll-fit['starts'][fit['selected_start']]['final_nll'])
    if err>1e-8:raise ValueError('NLL reconstruction error '+str(err))
    return model,state,fit,training,q,posterior(model,q,training,state['jitter_index']),err


def subset_unit(pred,public,ids,seed,method,model_hash):
    pos={c:i for i,c in enumerate(public['ids'])}
    result={**pred,'setting_id':f'{seed}_{method}_Q0','method':method,'n':50,'fold':-1,'draw':seed,'seed':seed,'ids':ids,'model_hash':model_hash}
    for name in ('mean_raw','mean_standardized','latent_variance_standardized'):result[name]=[pred[name][pos[c]] for c in ids]
    return result


def prediction_metrics(unit,labels,top10):
    score=score_prediction(unit,labels);score['legacy_q_top10percent_bias_pp']=score.pop('q_top10_bias_pp')
    score['seed']=unit['seed'];ids=unit['ids'];indices=[ids.index(c) for c in top10]
    mu=np.array(unit['mean_raw']);y=np.array([labels[c]['mean'] for c in ids]);scale=unit['y_scale']
    lv=np.array(unit['latent_variance_standardized'])*scale**2
    pv=lv+(unit['extra_noise_variance_standardized']+unit['median_known_var_standardized'])*scale**2
    error=mu[indices]-y[indices];pred5=sorted(range(len(ids)),key=lambda i:(-mu[i],ids[i]))[:5]
    score.update(rmse_pp=float(np.sqrt(np.mean((mu-y)**2))*100),top10_count=10,top10_signed_bias_pp=float(error.mean()*100),
        top10_mean_underestimate_pp=float(np.maximum(-error,0).mean()*100),top10_MAE_pp=float(np.abs(error).mean()*100),
        top10_mean_latent_sd_pp=float(np.sqrt(lv[indices]).mean()*100),top10_mean_prospective_sd_pp=float(np.sqrt(pv[indices]).mean()*100),
        predicted_top5_true_mean_percent=float(y[pred5].mean()*100),predicted_top5_positive_overestimate_pp=float(np.maximum(mu[pred5]-y[pred5],0).mean()*100),
        top10_ids=top10,predicted_top5_ids=[ids[i] for i in pred5])
    for name,variance in [('latent',lv),('prospective',pv)]:
        score.update({'top10_'+name+'_'+k:v for k,v in _interval_metrics(y[indices],mu[indices],variance[indices],density=name=='prospective').items()})
    return score


def run(root,repo):
    folder=root/'analysis';folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists() and all((folder/name).exists() for name in ('search_curves.png','unknown_champion_means.png')):return read_json(folder/'summary.json')
    for seed in SEEDS:validate_complete(root/'trajectories'/f'{seed}_W-HI')
    status(root,'analysis','RUNNING');gp.cpu_setup()
    public=read_json(root/'public_features.json');initial=read_json(root/'initial_observation_manifest.json');index={c:i for i,c in enumerate(public['ids'])}
    frozen_hashes={str(p.relative_to(root)):sha256(p) for p in (root/'trajectories').rglob('complete.json')}
    units=[];raw_traces=[];reconstruction=[];fixed_records=[];errors=[]
    for seed in SEEDS:
        q0=sorted(set(public['ids'])-set(initial[str(seed)]['ids']));assert len(q0)==129
        for method,base in [('W-I',repo/REPLAY),('W-HI',root)]:
            trajectory=base/'trajectories'/f'{seed}_{method}';observations=read_json(trajectory/'observations.json')
            hit=next((max(0,i-49) for i,r in enumerate(observations) if r['candidate_id']==CHAMPION),None)
            for n in range(50,71):
                try:
                    model,state,fit,training,q,pred,nllerr=restore(trajectory/'fits'/f'n{n:03d}',public,method)
                    decision=read_json(trajectory/'decisions'/f'{n-49:02d}.json');remaining=decision['candidate_ids'];mus=[pred['mean_raw'][index[c]] for c in remaining]
                    record={'seed':seed,'method':method,'n':n,'NLL_abs_error':nllerr,'checkpoint_sha256':fit['selected_model_sha256'],'status':'PASS'}
                    if n==70:
                        err=float(np.max(np.abs(np.array(mus)-decision['scores'])));record['n70_sealed_mean_max_abs_error']=err
                        if err>1e-8:raise ValueError('Saved terminal posterior mean differs')
                    if n==50:
                        units.append(subset_unit(pred,public,q0,seed,method,fit['selected_model_sha256']))
                        if method=='W-I':
                            # Old n50 full posterior means were not separately
                            # archived. Reproduce its sealed latent qLogEI output
                            # as a forward check, never to change any decision.
                            qidx=[index[c] for c in remaining]
                            scores,_=enumerate_scores(FrozenAdapter(model,state['jitter_index']),q[qidx],decision['best_f_standardized'],seed,1)
                            err=float(np.max(np.abs(scores-np.array(decision['scores']))));record['n50_sealed_qlogei_max_abs_error']=err
                            record['n50_posterior_evidence']='selected checkpoint + exact training NLL + sealed first qLogEI forward reproduction; full n50 posterior vector not separately stored historically'
                            if err>1e-8:raise ValueError('Old n50 sealed qLogEI reproduction failed')
                            fixed=fixed_parameter_model(state);fpre=posterior(fixed,q,training,state['jitter_index'])
                            identity=digest({'old_checkpoint':fit['selected_model_sha256'],'kernel_kind':KERNEL_KIND,'parameters':fpre['parameters']})
                            units.append(subset_unit(fpre,public,q0,seed,'W-HI_FIXED_WI_PARAMS',identity))
                            fixed_records.append({'seed':seed,'old_checkpoint_sha256':fit['selected_model_sha256'],'fixed_prediction_identity':identity,
                                'parameters':fpre['parameters'],'own_C_inverse_recomputed':True,'optimizer_calls':0,'NLL_abs_error':nllerr,
                                'sealed_qlogei_abs_error':err,'scaler':training['scaler'],'distance_scales':fit['distance_scales']})
                    ci=index[CHAMPION];observed=CHAMPION in training['ids'];var=pred['latent_variance_standardized'][ci]*pred['y_scale']**2
                    raw_traces.append({'seed':seed,'method':method,'n':n,'initial_contains_champion':CHAMPION in initial[str(seed)]['ids'],
                        'champion_observed':observed,'unknown_champion_diagnostic':not observed,'first_hit_query':hit,
                        'mu_raw':pred['mean_raw'][ci],'latent_sd_pp':100*math.sqrt(var),
                        'prospective_sd_pp':100*math.sqrt(var+(pred['extra_noise_variance_standardized']+pred['median_known_var_standardized'])*pred['y_scale']**2),
                        'remaining_mean_rank':rank(remaining,mus,CHAMPION),'actual_qlogei_rank':rank(remaining,decision['scores'],CHAMPION) if n<70 else None,
                        'decision_kind':decision['kind'],'selected_id':decision['candidate_id'],'actual_score':decision['scores'][remaining.index(CHAMPION)] if CHAMPION in remaining else None})
                    reconstruction.append(record)
                except Exception as exc:errors.append({'seed':seed,'method':method,'n':n,'error':repr(exc)})
    write_json(folder/'frozen_prediction_units.json',units);write_csv(folder/'champion_prediction_before_scoring.csv',raw_traces)
    write_json(folder/'fixed_parameter_diagnostics.json',fixed_records);write_json(folder/'posterior_reconstruction.json',{'checks':reconstruction,'errors':errors})
    seal_hash=sha256(folder/'frozen_prediction_units.json')
    write_json(folder/'prediction_seal.json',{'prediction_sha256':seal_hash,'all10_formal_trajectories_sealed':frozen_hashes,'sealed_at':now()})
    # Trusted independent scorer: A only; all formal selections are sealed.
    from analyse.structured_kernel_phase1.data import read_A_labels
    from .oracle import access_guard
    counts={'physical_A_label_file_opens':0}
    # Broad historical reads above have ended. The scorer cannot open any
    # historical file except the A whitelist; baseline rows are loaded below
    # from immutable copies prepared before this guard.
    baseline_curves=list(csv.DictReader((repo/FOLLOWUP/'random_curves.csv').open()))
    oldpair=list(csv.DictReader((repo/REPLAY/'paired_comparisons.csv').open()))
    randomrows=list(csv.DictReader((repo/FOLLOWUP/'random_trajectories.csv').open()))
    historical_observations={(seed,method):read_json(repo/REPLAY/'trajectories'/f'{seed}_{method}'/'observations.json') for seed in SEEDS for method in ('W-I','RAW16')}
    historical_observations.update({(seed,'Random'):read_json(repo/FOLLOWUP/'random_observations'/f'{seed}.json') for seed in SEEDS})
    access_guard(root,repo,'scorer',counts);labels=read_A_labels(repo)
    optimum=max(v['mean'] for v in labels.values());optids=sorted(c for c,v in labels.items() if v['mean']==optimum)
    top10={str(s):sorted(set(public['ids'])-set(initial[str(s)]['ids']),key=lambda c:(-labels[c]['mean'],c))[:10] for s in SEEDS}
    write_json(folder/'Q0_true_top10.json',{'constructed_after_trajectory_seal':True,'top10':top10,'count_per_seed':10,'labels_scope':'A-only'})
    metrics=[];prediction_rows=[]
    for u in units:
        metrics.append(prediction_metrics(u,labels,top10[str(u['seed'])]))
        for i,c in enumerate(u['ids']):prediction_rows.append({'seed':u['seed'],'method':u['method'],'candidate_id':c,'true_mean':labels[c]['mean'],'retrospective_SE':labels[c]['se'],
            'mu_raw':u['mean_raw'][i],'latent_variance_raw':u['latent_variance_standardized'][i]*u['y_scale']**2,
            'prospective_variance_raw':(u['latent_variance_standardized'][i]+u['extra_noise_variance_standardized']+u['median_known_var_standardized'])*u['y_scale']**2,
            'is_true_top10':c in top10[str(u['seed'])]})
    traces=[{**r,'champion_true_raw':labels[CHAMPION]['mean'],'mu_percent':100*r['mu_raw'],'bias_pp':100*(r['mu_raw']-labels[CHAMPION]['mean'])} for r in raw_traces]
    write_csv(folder/'champion_trace.csv',traces);write_csv(folder/'unknown_champion_trace.csv',[r for r in traces if r['unknown_champion_diagnostic']])
    write_csv(folder/'Q0_prediction_metrics.csv',metrics);write_csv(folder/'Q0_predictions.csv',prediction_rows)
    agg=[]
    numeric=[k for k,v in metrics[0].items() if isinstance(v,(float,int)) and k not in ('n','fold','draw','seed','query_count','top10_count','latent_negative_variance_count')] if metrics else []
    for method in ('W-I','W-HI_FIXED_WI_PARAMS','W-HI'):
        part=[r for r in metrics if r['method']==method]
        if len(part)==10:agg.append({'method':method,'contexts':10,**{k:float(np.mean([r[k] for r in part])) if all(r[k] is not None for r in part) else None for k in numeric}})
    write_csv(folder/'Q0_metrics_summary.csv',agg)
    curves=[{**r,'seed':int(r['seed']),'new_queries':int(r['new_queries']),'best_validation_mean':float(r['best_validation_mean']),
        'best_pp':float(r['best_pp']),'cached_simple_regret_pp':float(r['cached_simple_regret_pp'])} for r in baseline_curves]
    trajectories=[];paired=[];returns={}
    for seed,old,rand in zip(SEEDS,oldpair,randomrows):
        assert int(old['seed'])==int(rand['seed'])==seed
        obs=read_json(root/'trajectories'/f'{seed}_W-HI'/'observations.json')
        if any({k:o[k] for k in ('mean','se','seeds')}!=labels[o['candidate_id']] for o in obs):raise ValueError('Released labels differ from pinned A')
        best=[max(o['mean'] for o in obs[:50+q]) for q in range(22)]
        for q,b in enumerate(best):curves.append({'seed':seed,'method':'W-HI','new_queries':q,'kind':'initial' if q==0 else ('greedy' if q==21 else 'qlogei'),'best_validation_mean':b,'best_pp':100*b,'cached_simple_regret_pp':100*(optimum-b)})
        winner=min(obs,key=lambda o:(-o['mean'],o['candidate_id']))['candidate_id'];returns[str(seed)]={'candidate_id':winner,'mean':labels[winner]['mean'],'terminal_queried_candidate_id':obs[-1]['candidate_id'],'returned_from_all_observed71':True}
        hit=next((max(0,i-49) for i,o in enumerate(obs) if o['candidate_id']==CHAMPION),None)
        t={'seed':seed,'initial_best_pp':100*best[0],'initial_contains_champion':CHAMPION in initial[str(seed)]['ids'],
            'n70_best_pp':100*best[20],'n71_best_pp':100*best[21],'greedy_increment_pp':100*(best[21]-best[20]),
            'regret_integral_q0_20':float(np.trapezoid(100*(optimum-np.array(best[:21])))/20),'champion_hit':hit is not None,'first_hit_query':hit,'returned_id':winner}
        trajectories.append(t);p={'seed':seed,'initial_best_pp':t['initial_best_pp'],'initial_contains_champion':t['initial_contains_champion'],
            'W_HI_regret_integral':t['regret_integral_q0_20'],'W_I_regret_integral':float(old['WI_regret_integral']),'RAW16_regret_integral':float(old['RAW16_regret_integral']),'Random_regret_integral':float(rand['regret_integral_q0_20'])}
        for n in (70,71):
            p[f'W_HI_n{n}_pp']=100*best[n-50]
            for method,value in [('W_I',float(old[f'n{n}_WI_pp'])),('RAW16',float(old[f'n{n}_RAW16_pp'])),('Random',float(rand[f'n{n}_best_pp']))]:
                p[f'{method}_n{n}_pp']=value;p[f'W_HI_minus_{method}_n{n}_pp']=p[f'W_HI_n{n}_pp']-value
        p['W_HI_greedy_increment_pp']=t['greedy_increment_pp'];p['W_I_greedy_increment_pp']=float(old['WI_greedy_increment_pp']);p['RAW16_greedy_increment_pp']=float(old['RAW16_greedy_increment_pp']);p['Random_slot21_increment_pp']=float(rand['slot21_increment_pp'])
        paired.append(p)
    hits=[]
    for seed in SEEDS:
        for method in ('W-HI','W-I','RAW16','Random'):
            observed=read_json(root/'trajectories'/f'{seed}_W-HI'/'observations.json') if method=='W-HI' else historical_observations[(seed,method)]
            hit=next((max(0,i-49) for i,o in enumerate(observed) if o['candidate_id']==CHAMPION),None)
            hits.append({'seed':seed,'method':method,'champion_hit_n71':hit is not None,'first_hit_query':hit,'initial_contains_champion':hit==0})
    write_csv(folder/'champion_hits.csv',hits)
    indices=np.random.default_rng(2026100631).integers(0,10,size=(2000,10));comparisons={}
    for method in ('W_I','RAW16','Random'):
        for n in (70,71):
            vals=np.array([r[f'W_HI_minus_{method}_n{n}_pp'] for r in paired]);tie=np.abs(vals/100)<=1e-12
            comparisons[f'W_HI_minus_{method}_n{n}']={'mean_pp':float(vals.mean()),'CI95_pp':np.quantile(vals[indices].mean(1),[.025,.975]).tolist(),
                'wins':int(np.sum((vals>0)&~tie)),'ties':int(tie.sum()),'losses':int(np.sum((vals<0)&~tie)),'differences_pp':vals.tolist()}
    auc={m:float(np.mean([r[m+'_regret_integral'] for r in paired])) for m in ('W_HI','W_I','RAW16','Random')}
    mean=comparisons['W_HI_minus_W_I_n71']['mean_pp'];recommend='W-HI' if mean>1e-9 and auc['W_HI']<=auc['W_I']+1e-9 else 'W-I'
    optimization=[];physical={'attempts':0,'resumed_attempts':0,'adam_updates':0,'lbfgs_closures':0}
    for path in sorted((root/'trajectories').glob('*/fits/n*/start*.json')):
        r=read_json(path);optimization.append({'trajectory':path.parents[2].name,'n':int(path.parent.name[1:]),**r})
        for line in path.with_suffix('.jsonl').read_text().splitlines():
            e=json.loads(line)
            if e.get('event')=='physical_attempt':physical['attempts']+=1;physical['resumed_attempts']+=bool(e['resumed'])
            if e.get('stage')=='adam':physical['adam_updates']+=1
            if e.get('stage')=='lbfgs':physical['lbfgs_closures']+=1
    valid=len(optimization)==630 and all(r['success'] for r in optimization)
    selection={'method_selection':'RECOMMENDATION' if valid else 'INCONCLUSIVE','recommended_method':recommend if valid else 'W-I',
        'n71_W_HI_minus_W_I_mean_pp':mean,'CI95_pp':comparisons['W_HI_minus_W_I_n71']['CI95_pp'],'mean_regret_integral':auc,
        'rule_mean_positive':mean>1e-9,'rule_auc_not_worse':auc['W_HI']<=auc['W_I']+1e-9,'requires_independent_confirmation':True,
        'scientific_PASS':False,'evidence_boundary':'A development selection; bootstrap conditional on fixed cache and initialization only; not unseen architecture/GNN seed/test confirmation',
        'remaining_questions':['Prediction and calibration tradeoffs','Historical C graph provenance gaps','Independent confirmation still required']}
    if selection['CI95_pp'][0]<=0<=selection['CI95_pp'][1]:selection['interval_note']='CI includes zero: weak evidence; recommendation is not confirmation'
    stats={'starts':len(optimization),'failed_starts':sum(not r['success'] for r in optimization),
        'adam_warnings':sum(r.get('adam_final50_improvement',0)>1e-4 for r in optimization),'lbfgs_budget_warnings':sum(r.get('end_reason') in ('max_iter','max_eval') for r in optimization),
        'jitter_events':sum(len(r.get('jitter_events',[])) for r in optimization),'physical_work':physical,
        'logical_adam_updates':sum(r.get('adam_updates',0) for r in optimization),'physical_repeated_adam_updates':physical['adam_updates']-sum(r.get('adam_updates',0) for r in optimization),
        'logical_lbfgs_evaluations':sum(r.get('lbfgs_evaluations',0) for r in optimization),
        'fit_seconds':sum(r['elapsed_seconds'] for r in optimization)}
    write_json(root/'method_selection.json',selection);write_json(folder/'optimization_summary.json',stats);write_csv(folder/'optimization_by_start.csv',optimization)
    write_csv(folder/'trajectories.csv',trajectories);write_csv(folder/'paired_comparisons.csv',paired);write_csv(folder/'curves.csv',curves);write_json(folder/'returned_candidates.json',returns)
    write_json(folder/'search_comparisons.json',comparisons);write_json(folder/'bootstrap_identity.json',{'seed':2026100631,'replicates':2000,'shared_indices_sha256':digest(indices.tolist())})
    if sha256(folder/'frozen_prediction_units.json')!=seal_hash:raise ValueError('Scorer changed frozen predictions')
    for p,h in frozen_hashes.items():
        if sha256(root/p)!=h:raise ValueError('Scorer changed trajectory completion seal')
    under={r['method']:r for r in agg};under_diagnosis={}
    if 'W-I' in under and 'W-HI' in under:
        under_diagnosis={'top10_underestimate_reduced':under['W-HI']['top10_mean_underestimate_pp']<under['W-I']['top10_mean_underestimate_pp'],
            'top10_MAE_reduced':under['W-HI']['top10_MAE_pp']<under['W-I']['top10_MAE_pp'],
            'top10_underestimate_delta_pp':under['W-HI']['top10_mean_underestimate_pp']-under['W-I']['top10_mean_underestimate_pp'],
            'top10_MAE_delta_pp':under['W-HI']['top10_MAE_pp']-under['W-I']['top10_MAE_pp']}
    write_json(folder/'metric_units.json',{'accuracy':'raw0..1','errors_width_IS':'percentage points','NLPD':'nat density on raw cached three-seed accuracy mean','true_top10':'exactly10, not legacy top10percent13','legacy_q_top10percent_bias_pp':'original scorer field, ceil(.1*129)=13, auxiliary only','bootstrap':'fixed A and initialized search conditional, not independent training repetitions'})
    result={'state':'COMPLETE' if valid and len(units)==30 and len(fixed_records)==10 and not errors else 'INCOMPLETE',
        'formal_trajectories':10,'formal_fits':210,'formal_starts':len(optimization),'fixed_parameter_checks':len(fixed_records),
        'fixed_parameter_optimizer_calls':0,'old_GP_refits':0,'prediction_units':len(units),'posterior_errors':errors,
        'comparisons':comparisons,'regret_integral_means':auc,'method_selection':selection,'underestimation':under_diagnosis,
        'champion_hits':{m:sum(r['champion_hit_n71'] for r in hits if r['method']==m) for m in ('W-HI','W-I','RAW16','Random')},
        'physical_scoring_A_label_file_opens':counts['physical_A_label_file_opens'],'scorer_predictions_unchanged':True,
        'B_performance_SE_reads':0,'C_performance_predictions_or_scores':0,'GNN_training':0,'node_test_scores':0,'KG':0}
    plots(folder,curves,traces,agg)
    write_json(folder/'summary.json',result);status(root,'analysis',result['state'],fixed_parameter_checks=len(fixed_records),errors=errors)
    return result


def plots(folder,curves,traces,agg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for method in ('W-HI','W-I','RAW16','Random'):
        vals=[[r['cached_simple_regret_pp'] for r in curves if r['method']==method and r['seed']==s] for s in SEEDS]
        ax.plot(range(22),np.mean(vals,axis=0),marker='.',label=method)
    ax.axvspan(20,21,color='gray',alpha=.15);ax.set(xlabel='New query (slot21: GP greedy / Random random)',ylabel='Mean cached simple regret (pp)',title='All10 paired A development trajectories');ax.legend();fig.tight_layout();fig.savefig(folder/'search_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(15,6),sharey=True)
    for ax,s in zip(axes.flat,SEEDS):
        for method in ('W-I','W-HI'):
            rs=[r for r in traces if r['seed']==s and r['method']==method and not r['champion_observed']]
            if rs:ax.plot([r['n'] for r in rs],[r['mu_percent'] for r in rs],marker='.',label=method)
        ax.axhline(81.53333333333332,color='black',ls=':',lw=1);ax.set_title(str(s));ax.set_xlabel('Observed n')
        if s in (2026100206,2026100208):ax.text(.5,.5,'Champion in initial50',ha='center',transform=ax.transAxes,fontsize=8)
    axes[0,0].set_ylabel('Unknown champion mean (%)');axes[1,0].set_ylabel('Unknown champion mean (%)');axes[0,0].legend();fig.tight_layout();fig.savefig(folder/'unknown_champion_means.png',dpi=160);plt.close(fig)
