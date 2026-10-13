"""Analysis only after all trajectories; returned checkpoints sealed before test."""
import csv,math,time
import numpy as np
from .common import *
from .performance import SEEDS,METHODS,set_stage,verify_performance,seal,dataset

Z90=1.6448536269514722
def csvwrite(path,rows):
 rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def metric_rows(y,mu,latent,prospective):
 err=mu-y;d={'MAE_pp':100*np.abs(err),'RMSE_squared_pp':10000*err**2,'signed_bias_pp':100*err}
 for name,var in [('latent',latent),('prospective',prospective)]:
  width=2*Z90*np.sqrt(var);lo=mu-width/2;hi=mu+width/2
  d[name+'_coverage90']=((y>=lo)&(y<=hi)).astype(float)
  d[name+'_width90_pp']=100*width
  d[name+'_IS90_pp']=100*(width+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0))
 return d

def aggregate(d,index=None):
 out={k:float(np.mean(v if index is None else v[index])) for k,v in d.items()}
 out['RMSE_pp']=math.sqrt(out.pop('RMSE_squared_pp'));return out

def paired(values,indices):
 v=np.array(values,dtype=float)
 return {'differences_pp':v.tolist(),'mean_pp':float(v.mean()),'CI95_pp':np.quantile(v[indices].mean(1),[.025,.975]).tolist(),
  'wins':int((v>0).sum()),'ties':int((v==0).sum()),'losses':int((v<0).sum())}

def score(root,repo):
 verify_performance(root,repo)
 if (root/'analysis_complete.json').exists():verify(root,read(root/'analysis_complete.json')['files']);return
 verify(root,read(root/'P5_complete.json')['files'])
 if len(list((root/'trajectories').glob('*/complete.json')))!=30:raise ValueError('Full-cache analysis forbidden before 30 formal trajectories')
 labels=read(root/'private/C_validation_cache.json');public=read(root/'public_C.json');pred=read(root/'sealed_predictions.json');ids=public['ids'];idx={c:i for i,c in enumerate(ids)}
 arches=sorted(set(public['architecture_ids']));groups={a:[i for i,v in enumerate(public['architecture_ids']) if v==a] for a in arches};rng=np.random.default_rng(2026100601)
 draws=[np.concatenate([groups[arches[i]] for i in draw]) for draw in rng.integers(0,len(arches),size=(2000,len(arches)))];metrics={};diagnostic=[]
 for method in ['W-I','RAW16']:
  rows={r['candidate_id']:r for r in pred['predictions'] if r['method']==method};y=np.array([labels[c]['mean'] for c in ids]);mu=np.array([rows[c]['mean'] for c in ids]);latent=np.array([rows[c]['latent_variance'] for c in ids]);pros=np.array([rows[c]['prospective_variance'] for c in ids])
  per=metric_rows(y,mu,latent,pros);main=aggregate(per);boot=[aggregate(per,d) for d in draws]
  interval={k:np.quantile([b[k] for b in boot],[.025,.975]).tolist() for k in main}
  top=sorted(range(len(ids)),key=lambda i:(-y[i],ids[i]))[:int(math.ceil(.1*len(ids)))];predtop=sorted(range(len(ids)),key=lambda i:(-mu[i],ids[i]))[:5]
  from scipy.stats import spearmanr
  retro=latent+np.array([rows[c]['extra_noise_variance']+labels[c]['se']**2 for c in ids]);retro_rows=metric_rows(y,mu,latent,retro)
  metrics[method]={'metrics':main,'architecture_bootstrap_CI95':interval,'true_top10pct_bias_pp':float(100*np.mean(mu[top]-y[top])),
   'predicted_top5_bias_pp':float(100*np.mean(mu[predtop]-y[predtop])),'spearman':float(spearmanr(y,mu).statistic),
   'retrospective_Q_SE_metrics':{k:v for k,v in aggregate(retro_rows).items() if k.startswith('prospective_')},'retrospective_label':'C SE used after labels, never for sealed prospective intervals'}
  diagnostic.extend({'method':method,'candidate_id':c,'architecture_id':public['architecture_ids'][idx[c]],'validation_mean':labels[c]['mean'],'SE':labels[c]['se'],**{k:rows[c][k] for k in ('mean','latent_variance','prospective_variance')}} for c in ids)
 curves=[];returns=[];trajectory_metrics={};champ=max(v['mean'] for v in labels.values())
 for seed in SEEDS:
  for method in METHODS:
   tr=f'{seed}_{method}';obs=read(root/'trajectories'/tr/'observations.json');assert len(obs)==71
   best=[]
   for q in range(22):
    value=max(o['mean'] for o in obs[:50+q]);best.append(value);curves.append({'seed':seed,'method':method,'n':50+q,'new_queries':q,'best_observed_pp':100*value,'regret_pp':100*(champ-value)})
   winner=min(obs,key=lambda o:(-o['mean'],o['candidate_id']));returns.append({'trajectory':tr,'seed':seed,'method':method,'candidate_id':winner['candidate_id'],'validation_mean':winner['mean'],'SE':winner['se']})
   trajectory_metrics[tr]={'n50_best_pp':100*best[0],'n70_best_pp':100*best[20],'n71_best_pp':100*best[21],'terminal_increment_pp':100*(best[21]-best[20]),'champion_hit':best[21]==champ}
 indices=np.random.default_rng(2026100531).integers(0,10,size=(2000,10));comparisons={}
 for other in ['Random','RAW16']:
  diff=[trajectory_metrics[f'{s}_W-I']['n71_best_pp']-trajectory_metrics[f'{s}_{other}']['n71_best_pp'] for s in SEEDS];comparisons['W-I_minus_'+other]=paired(diff,indices)
 write(root/'prediction_metrics.json',metrics);csvwrite(root/'prediction_diagnostics.csv',diagnostic);write(root/'search_metrics.json',{'comparisons':comparisons,'trajectories':trajectory_metrics,'bootstrap_indices_sha256':digest(indices.tolist()),'fixed_pool_initialization_uncertainty_only':True});csvwrite(root/'search_curves.csv',curves);csvwrite(root/'returned_validation_results.csv',returns)
 checkpoints={};taskmap={t['task_id']:t for t in read(root/'training_task_manifest.json')['tasks']}
 for cid in sorted({r['candidate_id'] for r in returns}):
  for seed in [0,1,2]:
   tid=f'{cid}_s{seed}';folder=root/'training'/tid;result=read(folder/'result.json');path=folder/'best_model.pt';assert sha(path)==result['checkpoint_hashes']['best_model.pt']
   checkpoints[tid]={'candidate_id':cid,'seed':seed,'path':str(path),'sha256':sha(path),'training_identity':result['identity'],'best_epoch':result['best_epoch'],'validation_accuracy':result['best_validation_accuracy']}
 assert len(checkpoints)<=90
 seal(root/'returned_configurations.json',{'returns':returns,'checkpoints':checkpoints,'selection_rule':read(root/'performance_protocol.json')['return_rule'],
  'P5_complete_sha256':sha(root/'P5_complete.json'),'test_metrics_accessed':False,'sealed_at':now()})
 names=['prediction_metrics.json','prediction_diagnostics.csv','search_metrics.json','search_curves.csv','returned_validation_results.csv','returned_configurations.json']
 write(root/'analysis_complete.json',{'files':{n:sha(root/n) for n in names},'time':now(),'formal_models_added':0})

def final_test(root,repo):
 verify_performance(root,repo)
 if (root/'P6_complete.json').exists():verify(root,read(root/'P6_complete.json')['files']);return
 verify(root,read(root/'analysis_complete.json')['files']);returned=read(root/'returned_configurations.json');assert returned['test_metrics_accessed'] is False
 set_stage(root,'P6','RUNNING');import torch
 from analyse.structured_kernel_followup.validation_train import original
 torch.set_default_dtype(torch.float32);device=torch.device(read(root/'resource_lock.json')['device']);bundle,data,context=dataset(root,repo,device)
 configs={c['candidate_id']:c for c in read(root/'C_qualified_manifest.json')['configurations']};rows=[];new=0
 for tid,record in returned['checkpoints'].items():
  path=Path(record['path']);out=root/'test_evaluations'/f'{tid}.json'
  if not path.is_file() or sha(path)!=record['sha256']:
   set_stage(root,'P6','TEST_BLOCKED',test_blocked_task=tid);raise RuntimeError('Returned checkpoint missing/mismatched: '+tid)
  if out.exists():
   value=read(out);assert value['checkpoint_sha256']==record['sha256'] and value['return_seal_sha256']==sha(root/'returned_configurations.json');rows.append(value);continue
  c=configs[record['candidate_id']];started=time.monotonic();state=torch.load(path,map_location='cpu',weights_only=False);assert state['identity']==record['training_identity']
  with original.isolated_rng(record['seed'],device):
   model=original.DynamicGNN(c,bundle.num_features,bundle.num_classes,dropout=float(c['dropout']),hidden_dim=int(c['hidden_dim']),gcnii_alpha=float(c['gcnii_alpha']),gcnii_theta=float(c['gcnii_theta']),gat_heads=int(c['gat_heads']),sage_aggr=c['sage_aggr'],gin_eps=float(c['gin_eps']),gat_heads_by_layer=c.get('gat_heads_by_layer'),sage_aggr_by_layer=c.get('sage_aggr_by_layer'),gin_eps_by_layer=c.get('gin_eps_by_layer'),component_variant='baseline',activation_variant='relu').to(device)
   model.load_state_dict(state['model']);model.eval()
   with torch.no_grad():
    pred=model(data.x,data.edge_index).argmax(dim=1);val=float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()/int(data.val_mask.sum()));assert val==record['validation_accuracy']
    test=float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()/int(data.test_mask.sum()))
  value={'task_id':tid,'candidate_id':record['candidate_id'],'seed':record['seed'],'checkpoint_sha256':record['sha256'],'return_seal_sha256':sha(root/'returned_configurations.json'),'best_epoch':record['best_epoch'],'validation_accuracy':val,'test_accuracy':test,'dataset_context':context,'test_evaluations':1,'new_training':0,'elapsed_seconds':time.monotonic()-started,'time':now()};write(out,value);rows.append(value);new+=1
  del model,state;torch.cuda.empty_cache()
 byid={cid:[r['test_accuracy'] for r in rows if r['candidate_id']==cid] for cid in sorted(configs) if any(r['candidate_id']==cid for r in rows)}
 stats={cid:{'mean':float(np.mean(v)),'std_ddof1':float(np.std(v,ddof=1)),'SE':float(np.std(v,ddof=1)/math.sqrt(3)),'values':v} for cid,v in byid.items()};assert all(len(v)==3 for v in byid.values())
 tr={r['trajectory']:{**r,'test_mean':stats[r['candidate_id']]['mean'],'test_std':stats[r['candidate_id']]['std_ddof1']} for r in returned['returns']};indices=np.random.default_rng(2026100531).integers(0,10,size=(2000,10));comparisons={}
 for other in ['Random','RAW16']:comparisons['W-I_minus_'+other]=paired([100*(tr[f'{s}_W-I']['test_mean']-tr[f'{s}_{other}']['test_mean']) for s in SEEDS],indices)
 write(root/'test_results.json',{'configurations':stats,'trajectories':tr,'comparisons':comparisons,'unique_checkpoint_evaluations':len(rows),'new_evaluations_this_invocation':new,'additional_training':0,'return_seal_sha256':sha(root/'returned_configurations.json')});csvwrite(root/'test_seed_results.csv',rows)
 names=['test_results.json','test_seed_results.csv']+[str(p.relative_to(root)) for p in (root/'test_evaluations').glob('*.json')]
 write(root/'P6_complete.json',{'files':{n:sha(root/n) for n in names},'test_evaluations':len(rows),'new_GNN_tasks':0,'time':now()});set_stage(root,'P6','COMPLETE')
