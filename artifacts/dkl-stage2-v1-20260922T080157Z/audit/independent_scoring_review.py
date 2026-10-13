import csv, datetime, hashlib, itertools, json, math, statistics, sys
from pathlib import Path

root=Path(sys.argv[1] if len(sys.argv)>1 else '/home/fangxy/D-VAE/artifacts/dkl-stage2-v1-20260922T080157Z')
def read(p):return json.loads((root/p).read_text())
def sha(p):return hashlib.sha256((root/p).read_bytes()).hexdigest()
def avg(v):return math.fsum(v)/len(v)
def close(a,b,atol=1e-9):
    if a is None or b is None:
        assert a is None and b is None,(a,b)
    else:assert math.isclose(a,b,rel_tol=1e-10,abs_tol=atol),(a,b)
manifest=read('predictions_manifest.json')
assert manifest['sealed'] is True and manifest['phase2_status']=='COMPLETE_VALID'
assert manifest['phase2_fit_status']=='COMPLETE' and len(manifest['files'])==200
assert manifest['resources']['formal_optimizer_updates']==344000
units=[]
for e in manifest['files']:
    assert sha(e['path'])==e['sha256']
    assert sha(e['state_path'])==e['state_sha256']
    p=read(e['path']);assert p['state_file_sha256']==e['state_sha256']
    assert p['unit_id']==e['unit_id'];units.append(p)
# Only open the A-only scoring labels after all hashes have been checked.
lp=read('scoring/labels_A.json');labels=lp['labels']
assert len(labels)==179
assert hashlib.sha256(json.dumps(sorted(labels),sort_keys=True,separators=(',',':')).encode()).hexdigest()=='d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2'
assert lp['metadata']['prediction_seal_sha256']==sha('predictions_manifest.json')
assert datetime.datetime.fromisoformat(lp['metadata']['released_at_utc'])>datetime.datetime.fromisoformat(manifest['sealed_at_utc'])
for x in labels.values():
    assert set(x)=={'validation_mean','validation_se','validation_seeds','training_seeds'}
    assert x['training_seeds']==[0,1,2]
    close(x['validation_mean'],avg(x['validation_seeds']))
    close(x['validation_se'],statistics.stdev(x['validation_seeds'])/math.sqrt(3))
with (root/'scoring/unit_metrics.csv').open() as s:official={r['unit_id']:r for r in csv.DictReader(s)}
summary=read('scoring/summary.json');random_official=read('scoring/random_reference.json')
computed={};max_errors={};selected_checks=0
norm=statistics.NormalDist()
for p in units:
    ids=p['candidate_ids'];truth=[labels[c]['validation_mean'] for c in ids]
    pred=p['mean_raw'];order=sorted(range(len(ids)),key=lambda i:(-pred[i],ids[i]));selected=order[:5]
    best=min(selected,key=lambda i:(-truth[i],ids[i]));top=max(truth)
    row=official[p['unit_id']]
    assert json.loads(row['selected_top5_ids'])==[ids[i] for i in selected]
    assert row['selected_best_id']==ids[best]
    assert row['selected_top1_id']==ids[order[0]]
    selected_checks+=1
    errors=[a-b for a,b in zip(pred,truth)]
    stdlatent=p['latent_variance_standardized']
    assert min(stdlatent)>=-1e-8
    v=[p['y_scale']**2*(latent+p['extra_noise_variance_standardized'])+labels[c]['validation_se']**2 for latent,c in zip(stdlatent,ids)]
    assert min(v)>0
    def corr(x,y):
        if len(set(x))==1 or len(set(y))==1:return None
        def ranks(x):return [1+sum(t<a for t in x)+(sum(t==a for t in x)-1)/2 for a in x]
        rx,ry=ranks(x),ranks(y);mx,my=avg(rx),avg(ry)
        return sum((a-mx)*(b-my) for a,b in zip(rx,ry))/math.sqrt(sum((a-mx)**2 for a in rx)*sum((a-my)**2 for a in ry))
    m={'regret5_pp':100*(top-max(truth[i] for i in selected)),
       'top1_regret_pp':100*(top-truth[order[0]]),
       'mae_pp':100*avg([abs(x) for x in errors]),
       'rmse_pp':100*math.sqrt(avg([x*x for x in errors])),
       'spearman':corr(pred,truth),
       'nlpd_accuracy_units':avg([math.log(math.sqrt(2*math.pi*variance))+(error/math.sqrt(variance))**2/2 for error,variance in zip(errors,v)]),
       'mean_latent_sd_pp':100*p['y_scale']*avg([math.sqrt(max(0,x)) for x in stdlatent]),
       'mean_observed_sd_pp':100*avg([math.sqrt(x) for x in v])}
    for level in (90,95):
        z=norm.inv_cdf((1+level/100)/2)
        widths=[2*z*math.sqrt(x) for x in v]
        m[f'interval{level}_width_pp']=100*avg(widths)
        m[f'coverage{level}_fraction']=sum(abs(e)<=w/2 for e,w in zip(errors,widths))/len(errors)
    for k,value in m.items():
        ref=float(row[k]) if row[k] else None;close(value,ref)
        if value is not None:max_errors[k]=max(max_errors.get(k,0),abs(value-ref))
    computed[(p['fold'],p['sample_seed'],p['n'],p['model_seed'],p['method'])]=m
    c=p['cost'];close(c['standalone_fit_predict_seconds'],c['common_training_seconds']+c['branch_training_seconds']+c['prediction_seconds'])

refs=[]
for fold in range(5):
    p=next(p for p in units if p['fold']==fold)
    y=sorted(labels[c]['validation_mean'] for c in p['candidate_ids']);m=len(y)
    # Independent equivalent formula: integrate the CDF of the maximum across
    # adjacent sorted score gaps, then verify by exhaustive 5-subset enumeration.
    gap_regret=sum((y[i]-y[i-1])*math.comb(i,5)/math.comb(m,5) for i in range(5,m))
    total=0.0;combinations=0
    for group in itertools.combinations(y,5):
        total+=group[-1];combinations+=1
    enumeration_regret=y[-1]-total/combinations
    close(enumeration_regret,gap_regret,atol=1e-11)
    original=next(r for r in random_official['references_by_fold'] if r['fold']==fold)
    close(100*gap_regret,original['expected_regret5_pp'])
    refs.append({'fold':fold,'query_size':m,'exact_gap_formula_regret5_pp':100*gap_regret,
                 'all_combinations_enumerated':combinations,'enumeration_regret5_pp':100*enumeration_regret})

methods=['FROZEN','REG','DKL','RAW16'];ns={}
for n in (20,50):
    method_results={};fold_values={}
    for method in methods:
        seeds=[None] if method=='RAW16' else [0,1,2]
        fold_values[method]=[]
        aggregated={}
        for metric in next(iter(computed.values())):
            fvals=[]
            for fold in range(5):
                svals=[]
                for sample in (20260922,20260923):
                    vals=[computed[(fold,sample,n,seed,method)][metric] for seed in seeds]
                    vals=[x for x in vals if x is not None]
                    svals.append(avg(vals) if vals else None)
                svals=[x for x in svals if x is not None]
                fvals.append(avg(svals) if svals else None)
            vals=[x for x in fvals if x is not None]
            aggregated[metric]=avg(vals) if vals else None
            close(aggregated[metric],summary['per_n'][str(n)]['methods'][method]['metrics'][metric]['mean'])
            if metric=='regret5_pp':fold_values[method]=fvals
        method_results[method]=aggregated
    pairs={}
    for comparator in ('REG','FROZEN','RAW16'):
        # Pair every DKL model seed against the matched comparator endpoint first,
        # then independently perform the two aggregation levels.
        fold_improvements=[]
        for fold in range(5):
            sample_means=[]
            for sample in (20260922,20260923):
                differences=[computed[(fold,sample,n,None if comparator=='RAW16' else seed,comparator)]['regret5_pp']-computed[(fold,sample,n,seed,'DKL')]['regret5_pp'] for seed in (0,1,2)]
                sample_means.append(avg(differences))
            fold_improvements.append(avg(sample_means))
        paired={'fold_improvements_pp':fold_improvements,'mean_pp':avg(fold_improvements),'median_pp':statistics.median(fold_improvements),
                'improved_folds':sum(x>1e-9 for x in fold_improvements),'tied_folds':sum(abs(x)<=1e-9 for x in fold_improvements),'worse_folds':sum(x < -1e-9 for x in fold_improvements)}
        target=summary['per_n'][str(n)]['paired_regret5_improvement_pp'][comparator]
        close(paired['mean_pp'],target['mean']);close(paired['median_pp'],target['median'])
        for k in ('improved_folds','tied_folds','worse_folds'):assert paired[k]==target[k]
        pairs[comparator]=paired
    dkl=method_results['DKL']['regret5_pp'];raw=method_results['RAW16']['regret5_pp'];random=avg([r['exact_gap_formula_regret5_pp'] for r in refs])
    reg=pairs['REG']
    if reg['mean_pp']>1e-9 and reg['improved_folds']>=4 and raw-dkl>1e-9 and random-dkl>1e-9:signal='较一致的离线选优信号'
    elif reg['improved_folds']>0:signal='局部或混合信号'
    else:signal='未见相对继续回归的收益'
    assert signal==summary['scientific_signal_n'+str(n)]
    ns[str(n)]={'classification':signal,'method_metrics':method_results,'pairs':pairs,
                'random5_regret5_pp':random,'random5_minus_DKL_pp':random-dkl}

commons={}
for p in units:
    if p['method']!='RAW16':
        k=(p['fold'],p['sample_seed'],p['n'],p['model_seed'])
        if k in commons:close(commons[k],p['cost']['common_training_seconds'])
        commons[k]=p['cost']['common_training_seconds']
actual_endpoints=math.fsum(commons.values())+math.fsum(p['cost']['branch_training_seconds']+p['cost']['prediction_seconds'] for p in units)+math.fsum(p['cost']['common_training_seconds'] for p in units if p['method']=='RAW16')
cost_summary=read('scoring/cost_summary.json')
close(actual_endpoints,cost_summary['completed_endpoint_fit_predict_seconds_with_shared_common_cost_once'])
ledger=read('resource_ledger.json')
formal_fit_predict=ledger['training_seconds']+ledger['prediction_seconds']
close(formal_fit_predict,manifest['resources']['fit_predict_seconds'])
close(formal_fit_predict+ledger['verification_seconds'],ledger['seconds'])
standalone_means={str(n):{method:avg([p['cost']['standalone_fit_predict_seconds'] for p in units if p['n']==n and p['method']==method]) for method in methods} for n in (20,50)}
for n in ('20','50'):
    for method in methods:close(standalone_means[n][method],cost_summary['per_n_method_standalone_cost'][n][method]['mean'])
result={'status':'PASS','reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviewer':'independent scoring subagent; independent standard-library formulas; no scorer import',
        'prediction_seal_sha256':sha('predictions_manifest.json'),'labels_A_sha256':sha('scoring/labels_A.json'),
        'endpoints':len(units),'selected_id_sets_checked':selected_checks,'raw_endpoint_count':sum(p['method']=='RAW16' for p in units),
        'all_prediction_and_state_hashes_verified':True,'legal_A_label_count':179,'labels_released_after_seal':True,
        'per_n':ns,'random_reference_independent_verification':refs,'maximum_unit_metric_absolute_differences':max_errors,
        'costs':{'unit':'seconds','standalone_cost_identity_checked_for_all_200':True,'common_cost_once_per_60_contexts':True,
                 'mean_standalone_fit_predict_seconds':standalone_means,
                 'completed_endpoints_shared_actual_cost':actual_endpoints,'formal_fit_predict_seconds':formal_fit_predict,
                 'additional_60_common_fork_checks_seconds_charged_as_prediction':formal_fit_predict-actual_endpoints,
                 'synthetic_verification_seconds':ledger['verification_seconds'],'synthetic_verification_updates':ledger['verification_updates'],
                 'total_budget_used_seconds':ledger['seconds'],'formal_optimizer_updates':ledger['updates'],
                 'prediction_ledger_count_includes_200_endpoints_plus_60_fork_checks':ledger['predictions']},
        'limitations':['A is repeatedly used development data, not independent confirmation','Five folds overlap in training; no significance claim','Offline top-five choice is not real sequential search','n20 and n50 remain separately classified'],
        'training_performed_by_this_review':False,'B_or_test_read':False,'frozen_files_modified':False,'next_stage_authorized':False,'findings':[]}
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
