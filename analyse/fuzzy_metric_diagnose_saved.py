"""Read-only numerical diagnosis of saved Stage-2 models, before any new fit."""
from __future__ import annotations
from collections import defaultdict
import math
from pathlib import Path
import time
import numpy as np
import torch
from analyse.fuzzy_metric_diagnose import (read,write,load,restore_mapping,targets,losses,
    new_gp,gp_parameters,neighbor_indices,tail,file_hash,tensor_hash)
from surrogate.fuzzy_metric import neighborhood_affinity,metric_loss


def space_diagnostics(model,data,query_labels):
    """Train excludes self; same-architecture other HPs remain allowed explicitly."""
    with torch.no_grad():
        e,p=model(data['z'],data['hp']);qe,qp=model(data['qz'],data['qhp'])
    result={}
    for name,ids,rows,emb,head,labels in [
        ('train',data['ids'],data['train'],e,p,dict(zip(data['ids'],data['y'].tolist()))),
        ('unseen_architecture_query',data['qids'],data['query'],qe,qp,
         {c:r['validation_mean'] for c,r in query_labels.items()})]:
        js,dist=neighbor_indices(emb,e,data['ids'],ids)
        records=[]
        head=(head.detach().double()*data['ys']+data['ym']).tolist()
        for i,c in enumerate(rows):
            truth=labels[c['candidate_id']];neighbors=js[i]
            gaps=[abs(truth-float(data['y'][j])) for j in neighbors]
            same=[data['train'][j]['architecture_fingerprint']==c['architecture_fingerprint'] for j in neighbors]
            records.append({'candidate_id':c['candidate_id'],'architecture_fingerprint':c['architecture_fingerprint'],
                'head_absolute_error':abs(head[i]-truth),'head_prediction':head[i],
                'knn_absolute_error':abs(float(data['y'][neighbors].mean())-truth),
                'neighbor_ids':[data['ids'][j] for j in neighbors],
                'neighbor_mean_abs_difference':float(np.mean(gaps)),
                'same_architecture_neighbor_fraction':float(np.mean(same)),
                'wrong_0.006':float(np.mean(np.array(gaps)>.006)),
                'wrong_0.012':float(np.mean(np.array(gaps)>.012))})
        result[name]={'count':len(ids),'main_neighborhood_metric':float(np.median([r['neighbor_mean_abs_difference'] for r in records])),
            **{k:float(np.mean([r[k] for r in records])) for k in ['head_absolute_error','knn_absolute_error','same_architecture_neighbor_fraction','wrong_0.006','wrong_0.012']},
            'per_query':records,'self_excluded':True,'same_architecture_other_HP_allowed':name=='train'}
    return result


def gradient_diagnostics(model,data,setting):
    """No optimizer update. Compare actual weighted terms on shared map/HP params."""
    before=tensor_hash(model.state_dict());model.zero_grad(set_to_none=True)
    values=losses(model,data,setting)
    shared=[p for name,p in model.named_parameters() if not name.startswith('regressor.')]
    terms={'regression':values['regression'],'weighted_neighborhood':setting['lambda_f']*values['neighborhood'],
        'weighted_size':.001*values['scale']}
    vectors={}
    for name,value in terms.items():
        grads=torch.autograd.grad(value,shared,retain_graph=True,allow_unused=True)
        vectors[name]=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().double().flatten() for g,p in zip(grads,shared)])
    norms={k:float(v.norm()) for k,v in vectors.items()}
    cos={}
    for a,b in [('regression','weighted_neighborhood'),('regression','weighted_size'),('weighted_neighborhood','weighted_size')]:
        denominator=norms[a]*norms[b]
        cos[a+'__'+b]=float(torch.dot(vectors[a],vectors[b]))/denominator if denominator else None
    if tensor_hash(model.state_dict())!=before:raise RuntimeError('gradient-only check changed model')
    return {'norms':norms,'cosines':cos,'weighted_loss_values':{k:float(v.detach()) for k,v in terms.items()},
        'shared_parameters':'hp_encoder and mapping; regressor excluded','optimizer_updates':0}


def target_comparison(model,data,setting):
    p1=neighborhood_affinity(data['y'],mode='N1',epsilon=setting['epsilon'])
    p2=neighborhood_affinity(data['y'],mode='N2',epsilon=setting['epsilon'])
    _,w=targets(data,setting);i,j=torch.triu_indices(len(data['ids']),len(data['ids']),1)
    delta=p1[i,j]-p2[i,j];weighted=(w[i,j]*delta).sum();mass=(w[i,j]*p1[i,j]).sum()
    with torch.no_grad():
        e,p=model(data['z'],data['hp'])
        l1=metric_loss(e,p,data['target'],p1,lambda_f=setting['lambda_f'],lambda_e=.001,pair_weights=w)
        l2=metric_loss(e,p,data['target'],p2,lambda_f=setting['lambda_f'],lambda_e=.001,pair_weights=w)
    return {'epsilon':setting['epsilon'],'lambda_f':setting['lambda_f'],
        'max_target_difference':float(delta.max()),'mean_absolute_target_difference':float(delta.mean()),
        'weighted_N1_target_mass':float(mass),'truncated_weighted_target_mass':float(weighted),
        'truncated_fraction_of_weighted_target_mass':float(weighted/mass),
        'mean_weighted_target_difference':float(weighted/w[i,j].sum()),
        'same_embedding_N1_neighborhood_loss':float(l1['neighborhood']),
        'same_embedding_N2_neighborhood_loss':float(l2['neighborhood']),
        'same_embedding_neighborhood_loss_difference':float(l2['neighborhood']-l1['neighborhood']),
        'weighted_total_loss_difference':float(setting['lambda_f']*(l2['neighborhood']-l1['neighborhood'])),
        'embedding_hash':tensor_hash({'e':e}),'same_training_labels_noise_weights_and_embedding':True}


def improvement_concentration(ctx):
    by_method={m:{} for m in ['regression','N0','N1','N2']}
    for tid,(done,path) in ctx.done.items():
        if done['stage']=='refit' and done['family']!='final_A':
            m=done['setting']['method'];seed=done['seed']
            by_method[m].setdefault(seed,[]).extend(read(path/'evaluation.json')['per_query'])
    out={}
    for method in ['N0','N1','N2']:
        seedrows=[]
        for seed in (0,1,2):
            base={r['candidate_id']:r for r in by_method['regression'][seed]}
            rows=[]
            topids={r['candidate_id'] for r in sorted(base.values(),key=lambda r:(-r['true_validation_mean'],r['candidate_id']))[:math.ceil(.1*len(base))]}
            for r in by_method[method][seed]:
                b=base[r['candidate_id']]
                rows.append({'candidate_id':r['candidate_id'],'architecture_fingerprint':r['architecture_fingerprint'],
                    'top10':r['candidate_id'] in topids,
                    'neighbor_improvement':b['neighbor_mean_abs_difference']-r['neighbor_mean_abs_difference'],
                    'gp_error_improvement':b['gp_absolute_error']-r['gp_absolute_error']})
            groups={}
            for group,sub in [('all',rows),('top10',[r for r in rows if r['top10']]),('remaining90',[r for r in rows if not r['top10']])]:
                arches=defaultdict(list)
                for r in sub:arches[r['architecture_fingerprint']].append(r['neighbor_improvement'])
                positive=sorted([max(0,r['neighbor_improvement']) for r in sub],reverse=True)
                groups[group]={'query_count':len(sub),'architecture_count':len(arches),
                    'improved_queries':sum(r['neighbor_improvement']>0 for r in sub),
                    'worse_queries':sum(r['neighbor_improvement']<0 for r in sub),
                    'mean_neighbor_improvement':float(np.mean([r['neighbor_improvement'] for r in sub])),
                    'mean_gp_error_improvement':float(np.mean([r['gp_error_improvement'] for r in sub])),
                    'top5_query_share_of_positive_improvement':sum(positive[:5])/sum(positive) if sum(positive)>0 else None,
                    'architecture_mean_improvements':{a:float(np.mean(v)) for a,v in arches.items()},
                    'improved_architectures':int(sum(np.mean(v)>0 for v in arches.values()))}
            seedrows.append({'seed':seed,'groups':groups,'per_query':rows})
        out[method]=seedrows
    return out


def audit_saved(ctx):
    start=time.monotonic();out=ctx.output/'saved_diagnostics';out.mkdir(exist_ok=True)
    if (out/'completed.json').exists():
        for p,h in read(out/'completed.json')['artifacts'].items():
            if file_hash(out/p)!=h:raise RuntimeError('old diagnosis artifact changed')
        print('Saved diagnosis already complete; no refit or recharge.');return
    trajectories=[];grid=[];spaces=[];params=[];targetrows=[];grads=[]
    gradient_seconds=0.0
    cached_names=['training_tails_and_curves.json','train_holdout_spaces.json','gp_parameters_at20.json',
        'inner_grid_fixed_fold_epsilon.json','N1_N2_same_embedding.json','weighted_shared_gradients.json']
    reuse_saved=all((out/p).exists() for p in cached_names)
    for tid,(done,path) in ([] if reuse_saved else sorted(ctx.done.items())):
        bundle=read(path/'predictions_before_reveal.json');d=bundle['diagnostics']
        trajectories.append({'task_id':tid,'family':done['family'],'stage':done['stage'],'seed':done['seed'],'setting':done['setting'],
            'mapping_last20':tail(d['loss_history'],['regression','neighborhood','scale','total'],20),
            'GP_last10':tail(d['gp_loss_history'],['negative_marginal_log_likelihood'],10),
            'mapping_history':d['loss_history'],'gp_history':d['gp_loss_history'],'final_losses':d['final_losses']})
        if done['stage']=='inner' and done['family']!='final_A':
            grid.append({'task_id':tid,'fold':done['family'],'seed':done['seed'],**done['setting'],
                'final_regression_loss':d['final_losses']['regression'],'final_neighborhood_loss':d['final_losses']['neighborhood'],
                'main_neighborhood_metric':done['metrics']['main_neighborhood_metric'],'gp_mae':done['metrics']['gp_mae']})
        selected_outer=done['stage']=='refit' and done['family']!='final_A'
        selected_inner=done['family']=='fold0' and done['stage']=='inner' and done['setting']['config_id'] in {
            ctx.selection['settings'][m]['config_id'] for m in ('regression','N1','N2')}
        if selected_outer or selected_inner:
            state=load(path/'model.pt');data,labels=ctx.data(state['train_ids'],state['query_ids'])
            model,opt=restore_mapping(state,data)
            from analyse.fuzzy_metric_offline import read_selected_labels
            qlabels=read_selected_labels(ctx.old/'A_validation.json',data['qids'])
            spaces.append({'task_id':tid,'family':done['family'],'stage':done['stage'],'seed':done['seed'],
                'method':done['setting']['method'],**space_diagnostics(model,data,qlabels)})
            gp,_=new_gp(data,state['train_embeddings']);gp.load_state_dict(state['gp'])
            params.append({'task_id':tid,'step':20,**gp_parameters(gp,data)})
            if done['setting']['method'] in ('N1','N2'):
                targetrows.append({'task_id':tid,'embedding_method':done['setting']['method'],**target_comparison(model,data,done['setting'])})
                gstart=time.monotonic();grads.append({'task_id':tid,**gradient_diagnostics(model,data,done['setting'])});gradient_seconds+=time.monotonic()-gstart
    # Reuse already durable read-only diagnostics after a report serialization failure.
    if reuse_saved:
        trajectories,spaces,params,cached_weights,targetrows,grads=[read(out/p) for p in cached_names]
    # Every comparison remains inside a fixed fold, method, epsilon and seed.
    weight_groups=[]
    for fold in ([] if reuse_saved else [f'fold{i}' for i in range(5)]):
        for method in ('N0','N1','N2'):
            for eps in (.005,.01,.02):
                selected=[r for r in grid if r['fold']==fold and r['method']==method and r['epsilon']==eps]
                values={}
                for w in (.01,.1,1.):
                    rs=sorted([r for r in selected if r['lambda_f']==w],key=lambda r:r['seed'])
                    if [r['seed'] for r in rs]!=[0,1,2]:raise RuntimeError('missing old fivefold grid seed')
                    values[str(w)]={'per_seed':rs,**{k:{'mean':float(np.mean([r[k] for r in rs])),
                        'min':min(r[k] for r in rs),'max':max(r[k] for r in rs)} for k in
                        ['final_regression_loss','final_neighborhood_loss','main_neighborhood_metric','gp_mae']}}
                changes=[]
                for before,after in [(.01,.1),(.1,1.)]:
                    for seed in (0,1,2):
                        a=values[str(after)]['per_seed'][seed];b=values[str(before)]['per_seed'][seed]
                        changes.append({'from_weight':before,'to_weight':after,'seed':seed,
                            **{k:a[k]-b[k] for k in ['final_regression_loss','final_neighborhood_loss','main_neighborhood_metric','gp_mae']}})
                weight_groups.append({'fold':fold,'method':method,'epsilon':eps,'by_weight':values,'paired_seed_changes':changes})
    if reuse_saved:weight_groups=cached_weights
    write(out/'training_tails_and_curves.json',trajectories)
    write(out/'train_holdout_spaces.json',spaces)
    write(out/'gp_parameters_at20.json',params)
    write(out/'inner_grid_fixed_fold_epsilon.json',weight_groups)
    write(out/'N1_N2_same_embedding.json',targetrows)
    write(out/'weighted_shared_gradients.json',grads)
    write(out/'improvement_concentration.json',improvement_concentration(ctx))
    write(out/'limitations.json',{'old_mapping_checkpoints':'M100 only; final20 pre-update loss samples plus final post-update loss saved',
        'old_GP_checkpoints':'G20 only; twenty objective/gradient samples; intermediate kernel/noise parameters NOT saved',
        'cannot_reconstruct_old_parameter_trajectory_without_refit':True,'new_parameter_trajectories':'fixed authorized G0..G100 runs only',
        'GP_tail_definition':'last10 of20, with first5/last5 means and fitted slope',
        'mapping_tail_definition':'last20 pre-update samples, with first5/last5 means and fitted slope',
        'common_scale_note':'a shared positive rescaling cannot change nearest-neighbor order; norm alone is not a cause',
        'gradient_note':'weighted shared gradients give local clues, not causal proof',
        'original_fivefold_gate':'negative unchanged; top10 posthoc only; model seeds not independent searches'})
    # Scientific plots from actual saved histories, all selected outer seeds overlaid.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    colors={'regression':'#666666','N0':'#009988','N1':'#ee7733','N2':'#0077bb'}
    for row in trajectories:
        if row['stage']!='refit' or row['family']=='final_A':continue
        m=row['setting']['method'];color=colors[m]
        for ax,k in zip(axes.flat[:4],['regression','neighborhood','scale','total']):
            hs=row['mapping_history'];ax.plot([r['step'] for r in hs],[r[k] for r in hs],color=color,alpha=.28,lw=.8);ax.set_title(k);ax.set_xlabel('mapping step')
        hs=row['gp_history'];axes.flat[4].plot([r['step'] for r in hs],[r['negative_marginal_log_likelihood'] for r in hs],color=color,alpha=.28,lw=.8)
    axes.flat[4].set_title('GP negative marginal log likelihood');axes.flat[4].set_xlabel('GP step')
    axes.flat[5].axis('off')
    for m,c in colors.items():axes.flat[5].plot([],[],label=m,color=c)
    axes.flat[5].legend(loc='center');fig.suptitle('Saved selected outer refits: five folds x three model seeds (no new fit)')
    fig.tight_layout();fig.savefig(out/'old_trajectories.png',dpi=150);fig.savefig(out/'old_trajectories.pdf');plt.close(fig)
    costs={'saved_successful_tasks_read':len(trajectories),'selected_models_reconstructed_without_update':len(spaces),
        'fixed_fold_epsilon_weight_groups':len(weight_groups),'same_embedding_comparisons':len(targetrows),
        'gradient_only_evaluations':len(grads),'gradient_seconds_this_invocation':gradient_seconds,'reused_diagnostics_after_serialization_failure':reuse_saved,
        'prior_gradient_seconds':None if reuse_saved else gradient_seconds,
        'total_read_only_analysis_seconds':time.monotonic()-start,'new_mapping_steps':0,'new_gp_steps':0,'gnn_trainings':0}
    write(out/'analysis_costs.json',costs)
    files=[p for p in out.iterdir() if p.is_file()]
    write(out/'completed.json',{'time':time.time(),'artifacts':{p.name:file_hash(p) for p in files},'costs':costs},immutable=True)
    print('Old diagnosis saved before new fitting:',costs,flush=True)
