"""Portable independent metrics: standard library only, saved predictions plus A labels."""
from pathlib import Path
import json,math,hashlib,sys

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def avg(xs):return math.fsum(xs)/len(xs) if xs else None

def rank(values):
    groups={}
    for i,v in enumerate(values):groups.setdefault(v,[]).append(i)
    result=[0.]*len(values);pos=1
    for v,indices in sorted(groups.items()):
        value=pos+(len(indices)-1)/2
        for i in indices:result[i]=value
        pos+=len(indices)
    return result

def corr(a,b):
    a,b=rank(a),rank(b);ma,mb=avg(a),avg(b);aa=[v-ma for v in a];bb=[v-mb for v in b]
    va,vb=math.fsum(v*v for v in aa),math.fsum(v*v for v in bb)
    return math.fsum(x*y for x,y in zip(aa,bb))/math.sqrt(va*vb) if va and vb else None

def recompute(o):
    seal=read(o/'all_predictions_sealed.json');protocol=read(o/'protocol.json');assert seal['protocol']==protocol['identity']
    assert all(sha(o/n)==h for n,h in seal['artifacts'].items())
    tolerance=1e-10;differences={}
    def close(key,a,b):
        if a is None or b is None:assert a is b,(key,a,b);return
        diff=abs(float(a)-float(b));differences[key]=max(differences.get(key,0.),diff)
        assert diff<=tolerance,(key,a,b,diff)
    original=read(o/'evaluation/per_candidate.json');lookup={(r['fold'],r['seed'],r['branch'],r['partition'],r['candidate_id']):r for r in original}
    original_tasks={(r['fold'],r['seed'],r['branch'],r['partition']):r for r in read(o/'evaluation/per_task.json')}
    records=[];taskrecords=[]
    for t in read(o/'task_matrix.json'):
        fold,seed,branch=t['fold'],t['seed'],t['branch'];bundle=read(o/'trajectories'/t['task_id']/'predictions_before_labels.json');labels=read(o/'evaluation_inputs'/f'fold{fold}.json')
        train_raw=read(o/'inputs'/f'fold{fold}'/'train_labels.json');ses=[]
        for cid,row in train_raw.items():
            values=row['validation_seeds'];m=avg(values);se=math.sqrt(math.fsum((v-m)**2 for v in values)/2)/math.sqrt(3)
            close('source_mean',m,row['validation_mean']);close('source_SE',se,row['validation_se']);ses.append(se*se)
        pooled=avg(ses);close('pooled_training_SE2',pooled,bundle['pooled_training_SE2']);extra=bundle['teacher_parameters']['noise']*bundle['scaler']['y_scale']**2
        close('extra_noise_units',extra,bundle['extra_noise_raw']);assert not bundle['hidden_query_SE_used']
        for part in ['train','query']:
            v=bundle[part];ids=v['ids'];truth=labels[part];assert ids==t[part+'_ids'] and set(ids)==set(truth)
            for i,cid in enumerate(ids):
                y=truth[cid];ge=100*(v['GP'][i]-y);he=100*(v['head'][i]-y);ov=v['latent_variance'][i]+pooled+extra
                close('observation_variance',ov,v['observation_variance'][i]);lower=v['GP'][i]-1.96*math.sqrt(ov);upper=v['GP'][i]+1.96*math.sqrt(ov)
                close('interval_lower',lower,v['lower'][i]);close('interval_upper',upper,v['upper'][i])
                row=dict(fold=fold,seed=seed,branch=branch,partition=part,candidate_id=cid,architecture=v['architectures'][i],y=y,GP=v['GP'][i],head=v['head'][i],GP_error_pp=abs(ge),GP_squared_error_pp=ge*ge,head_error_pp=abs(he),covered=lower<=y<=upper,width_pp=100*(upper-lower))
                old=lookup[(fold,seed,branch,part,cid)]
                for k in ['y','GP','head','GP_error_pp','GP_squared_error_pp','head_error_pp','covered','width_pp']:close(k,row[k],old[k])
                records.append(row)
            oracle=sorted(ids,key=lambda cid:(-truth[cid],cid))[:5];best=avg([truth[c] for c in oracle]);tr=dict(fold=fold,seed=seed,branch=branch,partition=part)
            ref=original_tasks[(fold,seed,branch,part)];assert oracle==ref['oracle_top5_ids'];close('oracle_top5_mean_pp',100*best,ref['oracle_top5_mean_pp'])
            for kind in ['GP','head']:
                selected=sorted(range(len(ids)),key=lambda i:(-v[kind][i],ids[i]))[:5];chosen=[ids[i] for i in selected];assert chosen==v['top5_'+kind]==ref[kind+'_top5_ids']
                selected_mean=avg([truth[c] for c in chosen]);tr[kind+'_top5_true_mean_pp']=100*selected_mean;tr[kind+'_top5_gap_pp']=100*(best-selected_mean);tr[kind+'_Spearman']=corr(v[kind],[truth[c] for c in ids])
                for k in [kind+'_top5_true_mean_pp',kind+'_top5_gap_pp',kind+'_Spearman']:close(k,tr[k],ref[k])
            taskrecords.append(tr)
    assert len(records)==len(original)
    if seal['full_fivefold']:
        assert sum(r['partition']=='query' for r in records)==1074 and sum(r['partition']=='train' for r in records)==4296
        for s in range(3):
            for b in ['C','G']:
                ids=[r['candidate_id'] for r in records if r['seed']==s and r['branch']==b and r['partition']=='query'];assert len(ids)==len(set(ids))==179
    def selected(meta,b):
        return {(r['fold'],r['candidate_id']):r for r in records if r['branch']==b and r['seed']==meta['seed'] and r['partition']==meta['partition'] and (meta['scope']=='all' or meta['scope']=='fold'+str(r['fold']))}
    def metric(rows,key,weighting):
        field={'GP_MAE_pp':'GP_error_pp','GP_RMSE_pp':'GP_squared_error_pp','head_MAE_pp':'head_error_pp','coverage95':'covered','width95_pp':'width_pp'}[key]
        if not rows:return None
        values=[r[field] for r in rows]
        if weighting=='architecture':
            groups={}
            for r in rows:groups.setdefault(r['architecture'],[]).append(r[field])
            values=[avg(v) for v in groups.values()]
        value=avg(values);return math.sqrt(value) if key=='GP_RMSE_pp' else value
    stored={}
    for summary in read(o/'evaluation/summaries.json'):
        a,b=selected(summary,'C'),selected(summary,'G');common=set(a)&set(b);chosen=a if summary['branch']=='C' else b
        value=metric([chosen[i] for i in sorted(common)],summary['metric'],summary['weighting']);close('summary_'+summary['metric'],value,summary['value'])
        assert len(common)==summary['common_queries'] and [list(i) for i in sorted((set(a)|set(b))-common)]==summary['excluded']
        stored[tuple(summary[k] for k in ['scope','partition','weighting','metric','seed','branch'])]=value
    for pair in read(o/'evaluation/paired.json'):
        prefix=tuple(pair[k] for k in ['scope','partition','weighting','metric','seed']);a,b=stored[prefix+('C',)],stored[prefix+('G',)];delta=b-a if a is not None and b is not None else None
        close('paired_'+pair['metric'],delta,pair['delta']);stored[prefix+('G-C',)]=delta
    for summary in read(o/'evaluation/overall.json'):
        prefix=tuple(summary[k] for k in ['scope','partition','weighting','metric']);values=[stored[prefix+(s,summary['branch'])] for s in range(3)]
        for a,b in zip(values,summary['seed_values']):close('overall_seed_'+summary['metric'],a,b)
        close('overall_'+summary['metric'],avg([v for v in values if v is not None]),summary['value'])
    selection_values={}
    for pair in read(o/'evaluation/selection_paired.json'):
        matches={r['branch']:r for r in taskrecords if all(r[k]==pair[k] for k in ['partition','fold','seed'])};key=pair['metric'];a=matches.get('C',{}).get(key);b=matches.get('G',{}).get(key)
        close('selection_C_'+key,a,pair['C']);close('selection_G_'+key,b,pair['G']);delta=b-a if a is not None and b is not None else None;close('selection_paired_'+key,delta,pair['delta'])
        for label,v in [('C',a),('G',b),('G-C',delta)]:selection_values[(pair['partition'],key,pair['fold'],pair['seed'],label)]=v if delta is not None else None
    for summary in read(o/'evaluation/selection_overall.json'):
        per_fold=[]
        for fold in range(5):
            vals=[selection_values[(summary['partition'],summary['metric'],fold,s,summary['branch'])] for s in range(3)];per_fold.append(avg([v for v in vals if v is not None]))
        value=avg([v for v in per_fold if v is not None]) if summary['scope']=='all' else per_fold[int(summary['scope'][4:])]
        close('selection_overall_'+summary['metric'],value,summary['value'])
    result=dict(passed=True,source_sha256=sha(Path(__file__)),implementation='Python standard library; no project or primary report imports; all metrics recomputed from sealed predictions and A-only evaluation labels',max_absolute_differences=differences,tolerance=tolerance,
      candidate_rows=len(records),holdout_rows=sum(r['partition']=='query' for r in records),training_rows=sum(r['partition']=='train' for r in records),fold_seed_method_part_records=len(taskrecords),
      all_prediction_top5_IDs_exact=True,SE_and_interval_units_independently_checked=True,optimizer_updates=0)
    (o/'independent_recompute.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(json.dumps(result))
if __name__=='__main__':recompute(Path(sys.argv[1]).resolve())
