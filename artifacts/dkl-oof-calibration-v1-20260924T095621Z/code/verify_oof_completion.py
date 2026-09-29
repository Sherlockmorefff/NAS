"""Read-only structural audit: no tensor runtime, model calls, or outer truth."""
import argparse,hashlib,json,math,pathlib,time,collections

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
 start=time.monotonic();work=read(root/'worklist.json');seal=read(root/'oof_manifest.json');budget=read(root/'budget.json');checks=0
 def check(ok,name):
  nonlocal checks
  if not ok:raise AssertionError(name)
  checks+=1
 check(seal['sealed'] is True and seal['execution_status']=='COMPLETE_VALID','uniform complete seal')
 check(len(work['pipelines'])==len(seal['completed_pipeline_ids'])==seal['completed_pipelines']==90,'90 pipelines')
 check(seal['completed_pipeline_ids']==[p['pipeline_id'] for p in work['pipelines']] and seal['prediction_sets']==270,'pipeline ordering and270 prediction sets')
 for doc in ['source_manifest.json','code_manifest.json','inner_splits_manifest.json']:
  for rel,h in read(root/doc)['files'].items():check(sha(root/rel)==h,doc+':'+rel)
 for key,rel in [('protocol_sha256','protocol.json'),('source_manifest_sha256','source_manifest.json'),('code_manifest_sha256','code_manifest.json'),('worklist_sha256','worklist.json'),('inner_splits_manifest_sha256','inner_splits_manifest.json')]:check(seal[key]==sha(root/rel),'seal:'+key)
 for rel,h in seal['files'].items():check(sha(root/rel)==h,'sealed:'+rel)
 expected={'COMMON_MSE':90000,'COMMON_GP':27000,'GP_CONST':135000,'GP_TREND':135000};counts=collections.Counter();predrows=0;diags=collections.Counter();initials={};conststats={m:[] for m in ('GP_CONST','GP_TREND')}
 for p in work['pipelines']:
  pid=p['pipeline_id'];split=read(root/p['split_path'])['folds'][p['inner_fold']]
  for stage,n,steps,cut,base in [('COMMON_MSE',1000,[0,800,900,1000],800,.001),('COMMON_GP',300,[0,240,300],240,.01),('GP_CONST',1500,[0,1000,1200,1300,1400,1500],1000,.01),('GP_TREND',1500,[0,1000,1200,1300,1400,1500],1000,.01)]:
   records=read(root/f'diagnostics/{pid}_{stage}.json');check([d['step'] for d in records]==steps,pid+stage+':diagnostic schedule');diags[stage]+=len(records)
   i=0
   with (root/f'updates/{pid}_{stage}.jsonl').open() as f:
    for i,line in enumerate(f,1):
     r=json.loads(line);lr=r.get('learning_rates',r.get('learning_rate'))
     values=list(lr.values()) if isinstance(lr,dict) else [lr]
     check(r['step']==i and r['planned_steps']==n and r['oof_stage']==stage and r['accounting_kind']=='formal',pid+stage+':update identity '+str(i))
     target=base if i<=cut else base*.1
     check(all(math.isclose(float(v),target,rel_tol=1e-14,abs_tol=0) for v in values),pid+stage+':lr '+str(i))
   check(i==n,pid+stage+':updates');counts[stage]+=i
   check(f'states/{pid}_{stage}.pt' in seal['files'],pid+stage+':saved state')
  for method in ['HEAD','GP_CONST','GP_TREND']:
   r=read(root/f'predictions_inner/{pid}_{method}.json');predrows+=len(r['candidate_ids'])
   check(r['candidate_ids']==split['V_ids'] and r['architecture_fingerprints']==split['V_architectures'],pid+method+':V identity')
   check(r['scaler']['fit_ids']==split['U_ids'] and r['scaler']['std_correction']==0,pid+method+':U scaler')
   check(r['query_labels_or_SE_present'] is False,pid+method+':no truth')
   seed=str(p['model_seed']);h=r['initial_mapper_hash'];check(seed not in initials or initials[seed]==h,pid+':same initialization');initials[seed]=h
   check(all(math.isfinite(x) for x in r['mean_raw']),pid+method+':finite means')
   if method=='HEAD':check(r['train_candidate_ids']==split['U_ids'] and len(r['train_mean_raw'])==p['n_fit'],pid+':HEAD U predictions')
   else:
    v=r['latent_variance_standardized'];cov=r['latent_covariance_standardized'];scale=r['y_scale'];extra=r['extra_noise_variance_standardized']
    check(len(cov)==len(v) and all(len(row)==len(v) for row in cov),pid+method+':full covariance')
    check(all(a>=-1e-8 and a==cov[i][i] for i,a in enumerate(v)),pid+method+':unclamped diagonal')
    check(all(math.isclose(b,scale**2*(max(a,0)+extra),rel_tol=1e-12,abs_tol=1e-15) for a,b in zip(v,r['model_variance_raw'])),pid+method+':model variance excludes SE')
    records={d['step']:d for d in read(root/f'diagnostics/{pid}_{method}.json')}
    stable=True;top5=True
    for lo,hi in [(1200,1300),(1300,1400),(1400,1500)]:
     a,b=records[lo],records[hi]
     ok=abs(b['nll_after_update']-a['nll_after_update'])<=1e-4*max(1,abs(a['nll_after_update']))
     for key in ['query_mean_raw','query_latent_sd_raw']:ok=ok and max(abs(x-y) for x,y in zip(a[key],b[key]))<=.001
     for key in ['matern_training_kernel','full_training_kernel']:
      delta=math.sqrt(math.fsum((x-y)**2 for ar,br in zip(a[key],b[key]) for x,y in zip(ar,br)));norm=math.sqrt(math.fsum(x*x for ar in a[key] for x in ar))
      ok=ok and delta/max(norm,1e-12)<=.01
     stable=stable and ok;top5=top5 and set(a['top5_ids'])==set(b['top5_ids'])
    check(stable==r['stability']['numeric_predictive_stable'] and top5==r['stability']['top5_stable'],pid+method+':independent three-window stability')
    check(r['actual_updates']==1500,pid+method+':endpoint');conststats[method].append({'pipeline_id':pid,'context_id':p['context_id'],'fold':p['fold'],'stable':r['stability']['numeric_predictive_stable'],'top5_stable':r['stability']['top5_stable']})
 check(dict(counts)==expected,'four class formal update totals')
 check(sum(counts.values())==budget['formal_updates']==budget['formal_optimizer_calls']==seal['formal_updates']==387000,'formal accounting exact')
 check(predrows==seal['oof_rows']==4500,'4500 OOF rows')
 check(diags['GP_CONST']==diags['GP_TREND']==540 and diags['COMMON_MSE']==360 and diags['COMMON_GP']==270,'full diagnostics cardinality')
 check(budget['seconds']<1500 and budget['limit']==1500 and budget['synthetic_optimizer_calls']<=300,'resource limits')
 check(all(budget[k]==0 for k in ['failed_optimizer_calls','failed_updates','discarded_updates','failed_training_attempts']),'no failed or discarded work')
 for stage,n in expected.items():
  c=budget['class_counters'][stage]['formal'];check(c['optimizer_calls']==c['updates']==c['successful_optimizer_calls']==n,stage+':call/update/success counts')
 stats={}
 for m,rows in conststats.items():
  stats[m]={'trajectories':90,'stable':sum(r['stable'] for r in rows),'top5_stable':sum(r['top5_stable'] for r in rows),'by_fold':{str(f):sum(r['stable'] for r in rows if r['fold']==f) for f in range(5)},'by_context':{cid:sum(r['stable'] for r in rows if r['context_id']==cid) for cid in sorted({r['context_id'] for r in rows})},'rows':rows}
 return {'status':'PASS','checks_passed':checks,'completed_pipelines':90,'formal_updates_by_class':dict(counts),'formal_updates':387000,'oof_rows':4500,'diagnostic_calls':dict(diags),'initialization_hashes':initials,'stability':stats,'outer_truth_read':False,'model_calls':0,'wall_seconds':time.monotonic()-start}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args();out=verify(a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='stability'}))
