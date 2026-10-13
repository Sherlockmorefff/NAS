#!/usr/bin/env python3
"""Independent post-output A gate/B scoring audit; no production scorer imports."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TOL=1e-9
METHODS=('R_FROZEN','R_REG','R_DKL_C','R_DKL_U','RAW16')
GATE_METHODS=('COMMON_REG','OLD_DKL_1000')
def need(ok,message):
    if not ok:raise ValueError(message)
def mean(values):
    values=[v for v in values if v is not None]
    return math.fsum(values)/len(values) if values else None
def close(a,b,label):
    if a is None or b is None:need(a is None and b is None,label+': undefined mismatch')
    else:need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=1e-9,rel_tol=1e-10),label+f': {a} != {b}')
def read(root,relative):return json.loads((root/relative).read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def csvrows(root,relative):
    with (root/relative).open() as f:return list(csv.DictReader(f))
def sealed_rows(root,filename):
    seal=read(root,filename);need(seal['sealed'] is True,'Unsealed inputs')
    need(seal['execution_status'] in ('COMPLETE_VALID','PARTIAL_BUDGET','PARTIAL'),'Invalid or blocked inputs')
    rows=[]
    for e in seal['files']:
        need(sha(root/e['path'])==e['sha256'],'Prediction/geometry hash mismatch')
        if 'state_path' in e:need(sha(root/e['state_path'])==e.get('state_file_sha256',e.get('state_sha256')),'State file hash mismatch')
        r=read(root,e['path']);need(r['unit_id']==e['unit_id'],'Unit identity differs');rows.append(r)
    return seal,rows
def label_map(root,phase):
    source=read(root,'source_manifest.json');path=root/('scoring_'+phase)/'labels_A.json'
    need(sha(path)==source['scoring_labels_expected_sha256'],'Scoring label hash mismatch')
    labels=json.loads(path.read_text())['labels']
    need(len(labels)==179,'Wrong legal A size')
    need(hashlib.sha256(json.dumps(sorted(labels),sort_keys=True,separators=(',',':')).encode()).hexdigest()=='d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2','Wrong legal A IDs')
    return labels
def nn_scores(row,labels):
    train,query=row['training_ids'],row['query_ids'];near=row['nearest_train_ids'];five=row['five_nearest_train_ids']
    need(not set(train)&set(query),'Training/query overlap')
    # Independently reconstruct fixed distance-based rankings where distances
    # are present; nearest tie-breaking uses candidate IDs, never labels.
    distances=row.get('geometry',{}).get('d_QT')
    if distances is not None:
        need(len(distances)==len(query),'Wrong distance matrix query count')
        for i,values in enumerate(distances):
            order=sorted(range(len(train)),key=lambda j:(values[j],train[j]))
            need([train[j] for j in order[:5]]==five[i] and train[order[0]]==near[i],'Nearest-neighbor IDs disagree with distances')
    need(all(c in train for c in near) and all(len(v)==5 and len(set(v))==5 and all(c in train for c in v) for v in five),'Invalid selected neighbors')
    e1=100*mean([abs(labels[q]['validation_mean']-labels[t]['validation_mean']) for q,t in zip(query,near)])
    e5=100*mean([mean([abs(labels[q]['validation_mean']-labels[t]['validation_mean']) for t in neighbors]) for q,neighbors in zip(query,five)])
    return e1,e5
def fold_values(rows,metric):
    output={}
    for f in range(5):
        values=[]
        for sample in (20260922,20260923):
            subset=[r[metric] for r in rows if r['fold']==f and r['sample_seed']==sample]
            if subset:values.append(mean(subset))
        if values:output[f]=mean(values)
    return output
def audit_A(root):
    seal,rows=sealed_rows(root,'historical_manifest.json');gate=read(root,'diagnostic_gate.json')
    if len(rows)!=280:
        need(gate['diagnostic_route']=='BLOCKED' and not gate['diagnostic_complete'],'Incomplete A must not admit B')
        return {'status':'PASS_INCOMPLETE_NOT_ADMITTED','diagnostic_route':'BLOCKED','historical_units':len(rows)}
    labels=label_map(root,'A');official={r['unit_id']:r for r in csvrows(root,'scoring_A/unit_metrics.csv')}
    computed=[]
    for row in rows:
        e1,e5=nn_scores(row,labels);reference=official[row['unit_id']]
        close(e1,float(reference['E1_pp']),'Historical E1');close(e5,float(reference['E5_pp']),'Historical E5')
        computed.append({**{k:row[k] for k in ('fold','sample_seed','n','model_seed','representation','unit_id')},'E1':e1,'E5':e5})
    raw={(r['fold'],r['sample_seed'],r['n']):r for r in computed if r['representation']=='RAW_X'}
    comparisons={};admit=False
    for n in (20,50):
        for method in GATE_METHODS:
            part=[]
            for r in computed:
                if r['n']==n and r['representation']==method:
                    part.append({**r,'delta':r['E1']-raw[(r['fold'],r['sample_seed'],n)]['E1']})
            values=fold_values(part,'delta');need(len(values)==5 and len(part)==30,'Incomplete gate comparison')
            avg=mean(list(values.values()));positive=sum(v>TOL for v in values.values());support=avg>TOL and positive>=4
            target=gate['comparisons'][f'n{n}_{method}']
            close(avg,target['mean_delta_E1_pp'],'Gate fold mean')
            for f,v in values.items():close(v,target['fold_delta_E1_pp'][str(f)],'Gate fold delta')
            need(positive==target['positive_folds'] and support==target['mapping_support'],'Gate support differs')
            comparisons[f'n{n}_{method}']={'mean_delta_E1_pp':avg,'positive_folds':positive,'fold_values':values,'mapping_support':support}
            admit=admit or support
    route='RUN_RESIDUAL_ABLATION' if admit else 'STOP_DIAGNOSTIC_ONLY'
    need(route==gate['diagnostic_route'],'A route differs')
    admission=read(root,'admission.json')
    need(set(admission)=={'route','diagnostic_gate_sha256','protocol_sha256','code_manifest_sha256','diagnostic_complete'},'Admission leaks diagnostic detail')
    need(admission['route']==route and admission['diagnostic_gate_sha256']==sha(root/'diagnostic_gate.json'),'Admission token differs')
    return {'status':'PASS','diagnostic_route':route,'historical_units':280,'comparisons':comparisons,
            'nearest_lists_reconstructed_from_stored_distances':True,'COMMON_GP_not_counted_as_second_gate_evidence':True}
def random_reference(y,incumbent):
    # Alternate exact CDF gap identity; does not reuse order-statistic weights.
    ys=sorted(y);m=len(y);den=math.comb(m,5)
    regret=math.fsum((ys[i]-ys[i-1])*math.comb(i,5)/den for i in range(5,m))
    gains=sorted(max(0,v-incumbent) for v in y)
    gain=gains[-1]-math.fsum((gains[i]-gains[i-1])*math.comb(i,5)/den for i in range(5,m))
    nonbreaking=sum(v-incumbent<=1e-11 for v in y)
    chance=1-(math.comb(nonbreaking,5)/den if nonbreaking>=5 else 0)
    return 100*regret,100*gain,chance
def formal_metrics(p,labels):
    nn=p['geometry_row'];ids=p['candidate_ids'];y=[labels[c]['validation_mean'] for c in ids]
    trainbest=max(labels[c]['validation_mean'] for c in nn['training_ids'])
    order=sorted(range(len(ids)),key=lambda i:(-p['mean_raw'][i],ids[i]));best=max(y[i] for i in order[:5])
    random_regret,random_gain,random_break=random_reference(y,trainbest);e1,e5=nn_scores(nn,labels)
    errors=[mu-v for mu,v in zip(p['mean_raw'],y)]
    var=[p['y_scale']**2*(max(0,v)+p['extra_noise_variance_standardized'])+labels[c]['validation_se']**2 for v,c in zip(p['latent_variance_standardized'],ids)]
    need(min(p['latent_variance_standardized'])>=-1e-8 and min(var)>0,'Variance validity')
    result={'regret5_pp':100*(max(y)-best),'gain5_pp':100*max(0,best-trainbest),'headroom_pp':100*max(0,max(y)-trainbest),
        'break_best_rate':float(best-trainbest>1e-11),'has_headroom_fraction':float(max(y)-trainbest>1e-11),
        'random_regret5_pp':random_regret,'random_gain5_pp':random_gain,'random_break_best_probability':random_break,
        'E1_pp':e1,'E5_pp':e5,'mae_pp':100*mean([abs(e) for e in errors]),'rmse_pp':100*math.sqrt(mean([e*e for e in errors])),
        'top1_regret_pp':100*(max(y)-y[order[0]]),'nlpd_accuracy_units':mean([math.log(math.sqrt(2*math.pi*v))+.5*(e/math.sqrt(v))**2 for e,v in zip(errors,var)])}
    for level in (90,95):
        z=statistics.NormalDist().inv_cdf((1+level/100)/2);half=[z*math.sqrt(v) for v in var]
        result[f'coverage{level}_fraction']=mean([float(abs(e)<=h) for e,h in zip(errors,half)])
        result[f'interval{level}_width_pp']=200*mean(half)
        if level==90:
            score=[]
            for mu,truth,h in zip(p['mean_raw'],y,half):
                lo,hi=mu-h,mu+h
                score.append(hi-lo+20*(lo-truth if truth<lo else truth-hi if truth>hi else 0))
            result['IS90_pp']=100*mean(score)
    return result,[ids[i] for i in order[:5]]
def audit_B(root):
    seal,predictions=sealed_rows(root,'predictions_manifest.json');labels=label_map(root,'B')
    official={r['unit_id']:r for r in csvrows(root,'scoring_B/unit_metrics.csv')};summary=read(root,'scoring_B/summary.json')
    complete=seal['execution_status']=='COMPLETE_VALID';need(not complete or len(predictions)==260,'Complete B size differs')
    rows=[];maxdiff=0
    for p in predictions:
        values,ids=formal_metrics(p,labels);target=official[p['unit_id']]
        need(ids==json.loads(target['selected_top5_ids']),'Formal selected IDs differ')
        for metric,value in values.items():close(value,float(target[metric]),'B unit '+metric);maxdiff=max(maxdiff,abs(value-float(target[metric])))
        rows.append({**{k:p[k] for k in ('unit_id','fold','sample_seed','n','model_seed','method')},**values})
    if not rows:
        need(not complete,'Complete B cannot have zero endpoints')
        need(all(summary['per_n'][str(n)]['selection_signal'] is None for n in (20,50)),'Empty partial B cannot be classified')
        return {'status':'PASS_PARTIAL_NO_SCIENTIFIC_ENDPOINTS','endpoint_count':0,'training_performed':False}
    lookup={(r['fold'],r['sample_seed'],r['n'],r['model_seed'],r['method']):r for r in rows};result={}
    for n in (20,50):
        means={}
        for method in METHODS:
            group=[r for r in rows if r['n']==n and r['method']==method]
            means[method]={metric:mean(list(fold_values(group,metric).values())) for metric in rows[0] if metric.endswith('_pp') or metric in ('break_best_rate','random_break_best_probability')}
            for metric,value in means[method].items():close(value,summary['per_n'][str(n)]['method_means'][method][metric],'B hierarchical mean')
        paired={};regpairs=[]
        for comparator in ('R_REG','R_DKL_U','R_FROZEN','RAW16'):
            differences=[]
            for c in (r for r in rows if r['n']==n and r['method']=='R_DKL_C'):
                ref=lookup.get((c['fold'],c['sample_seed'],n,None if comparator=='RAW16' else c['model_seed'],comparator))
                if ref is not None:differences.append({**c,'delta':ref['regret5_pp']-c['regret5_pp'],'gain_delta':c['gain5_pp']-ref['gain5_pp']})
            folds=fold_values(differences,'delta');gains=fold_values(differences,'gain_delta')
            target=summary['per_n'][str(n)]['paired_comparisons'][comparator]
            close(mean(list(folds.values())),target['metrics']['regret5_improvement_pp']['mean'],'B paired mean')
            close(mean(list(gains.values())),target['metrics']['gain5_improvement_pp']['mean'],'B gain paired mean')
            for f,v in folds.items():close(v,target['fold_regret5_improvement_pp'][str(f)],'B paired fold')
            paired[comparator]=folds
            if comparator=='R_REG':regpairs=differences
        if complete:
            c,reg,raw=means['R_DKL_C'],means['R_REG'],means['RAW16'];delta=list(paired['R_REG'].values());udelta=list(paired['R_DKL_U'].values())
            selection=mean(delta)>TOL and sum(v>TOL for v in delta)>=4 and raw['regret5_pp']-c['regret5_pp']>TOL and c['random_regret5_pp']-c['regret5_pp']>TOL
            mechanism=mean(udelta)>TOL and sum(v>TOL for v in udelta)>=4
            seed_sample={str(s):mean(list(fold_values([r for r in regpairs if r['sample_seed']==s],'delta').values())) for s in (20260922,20260923)}
            seed_model={str(m):mean(list(fold_values([r for r in regpairs if r['model_seed']==m],'delta').values())) for m in (0,1,2)}
            gain_ok=all(c['gain5_pp']>=v-TOL for v in (reg['gain5_pp'],raw['gain5_pp'],c['random_gain5_pp']))
            break_ok=all(c['break_best_rate']>=v-1e-12 for v in (reg['break_best_rate'],raw['break_best_rate'],c['random_break_best_probability']))
            follow=selection and gain_ok and break_ok and all(v>0 for v in seed_sample.values()) and sum(v>0 for v in seed_model.values())>=2
            target=summary['per_n'][str(n)]
            need(selection==target['selection_signal'] and mechanism==target['constraint_signal_pass'] and follow==target['followup_candidate'],'Scientific two-axis/followup rule differs')
            result[str(n)]={'selection_signal':selection,'constraint_signal_pass':mechanism,'followup_candidate':follow,'method_means':means,'paired_fold_delta':paired}
        else:need(summary['per_n'][str(n)]['selection_signal'] is None,'Partial B cannot receive complete classification')
    ledger=read(root,'resource_ledger.json');blocks=[json.loads(p.read_text()) for p in (root/'block_summaries').glob('*.json')]
    updates=sum(r['completed'] for r in blocks);need(updates<=ledger['updates'],'Completed block updates exceed ledger')
    if complete:need(updates==ledger['updates']==404000,'Complete formal update count differs')
    close(ledger['seconds'],ledger['training_seconds']+ledger['prediction_seconds']+ledger['verification_seconds'],'Resource additive accounting')
    return {'status':'PASS','endpoint_count':len(predictions),'RAW16_actual_endpoints':sum(p['method']=='RAW16' for p in predictions),
        'max_absolute_unit_metric_difference':maxdiff,'per_n':result,'completed_block_optimizer_updates':updates,'formal_updates_ledger':ledger['updates'],
        'fit_predict_verify_seconds':ledger['seconds'],'budget_limit_seconds':ledger['limit'],'training_performed':False}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('historical','formal','all'));ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();result={'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'review_script_sha256':sha(Path(__file__)),
        'independent_statistics_no_scoring_or_model_imports':True,'B_or_test_labels_accessed':False,'next_stage_authorized':False}
    if args.mode in ('historical','all'):result['A']=audit_A(root)
    if args.mode in ('formal','all') and (root/'scoring_B/summary.json').exists():result['B']=audit_B(root)
    elif args.mode=='formal':raise ValueError('B formal scoring output does not exist')
    result['status']='PASS';out=root/'audit'/('independent_'+args.mode+'_scoring.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'PASS','output':str(out)}))
if __name__=='__main__':main()
