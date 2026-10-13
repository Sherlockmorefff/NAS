"""Portable independent R1 recomputation; Python standard library only, no project imports."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def order(distances,gaps,scale):
    counts={'concordant':0,'discordant':0,'distance_ties':0,'performance_ties':0,'valid_denominator':0,'total_pairs':0}
    for i,j in itertools.combinations(range(len(gaps)),2):
        counts['total_pairs']+=1;a=gaps[i]-gaps[j];b=distances[i]-distances[j]
        if abs(a)<=1e-12:counts['performance_ties']+=1;continue
        counts['valid_denominator']+=1
        if abs(b)<=scale*1e-10:counts['distance_ties']+=1
        elif (a>0)==(b>0):counts['concordant']+=1
        else:counts['discordant']+=1
    counts['score']=(counts['concordant']+.5*counts['distance_ties'])/counts['valid_denominator'] if counts['valid_denominator'] else None
    return counts


def recompute(root):
    protocol=read(root/'protocol.json');seal=read(root/'all_inference_sealed.json')
    for name,h in seal['artifacts'].items():
        if sha(root/name)!=h:raise ValueError('sealed artifact changed: '+name)
    checks=read(root/'pretraining_validation.json')
    for name,h in checks['frozen_additional_evidence'].items():
        if sha(root/name)!=h:raise ValueError('pretraining evidence changed: '+name)
    rows=read(root/'evaluation/per_query.json')
    lookup={(r['seed'],r['method'],r['partition'],r['candidate_id']):r for r in rows}
    train=read(root/'evaluation_inputs/training_validation_means.json');query=read(root/'evaluation_inputs/query_validation_means.json')
    maxdiff={};totals={'rows':0,'global_comparisons':0,'performance_ties':0,'distance_ties':0,'NA':0,'negative_E_clipped':0,'probe_states':0,'probe_triplets':0}
    def close(key,a,b,tol=1e-10):
        if a is None or b is None:
            if a is not b:raise ValueError('NA mismatch '+key)
            return
        diff=abs(a-b);maxdiff[key]=max(maxdiff.get(key,0.),diff)
        if diff>tol:raise ValueError(f'{key} mismatch {a} vs {b}: {diff} > {tol}')
    for seed in [0,1,2]:
        for branch in ['R0','R1']:
            folder=root/'trajectories'/f's{seed}-{branch}'
            raw=read(folder/'inference_add200.json');spaces=read(folder/'neighbors_before_labels.json')
            tids=raw['train_ids']
            for method,space in spaces.items():
                te=space['train_embeddings'];qe=space['query_embeddings']
                if method=='R0-1D':
                    assert te==[[x] for x in raw['train_predictions']] and qe==[[x] for x in raw['query_predictions']]
                else:
                    assert te==raw['train_embeddings'] and qe==raw['query_embeddings']
                for part,embeddings,labels in [('train',te,train),('query',qe,query)]:
                    for cid,e,pred,saved in zip(raw[part+'_ids'],embeddings,raw[part+'_predictions'],space['rows'][part]):
                        js=[j for j,r in enumerate(tids) if r!=cid];refs=[tids[j] for j in js]
                        assert refs==saved['reference_ids'] and cid not in refs
                        d=[math.sqrt(sum((float(x)-float(y))**2 for x,y in zip(e,te[j]))) for j in js]
                        for a,b in zip(d,saved['distances']):close('distance',a,b)
                        g=[abs(labels[cid]-train[r]) for r in refs]
                        near=sorted(range(len(refs)),key=lambda j:(d[j],refs[j]))[:5]
                        oracle=sorted(range(len(refs)),key=lambda j:(g[j],refs[j]))[:5]
                        local=sorted(range(len(refs)),key=lambda j:(g[j],refs[j]))[:10]
                        assert [refs[j] for j in near]==saved['neighbor_ids']
                        r=lookup[(seed,method,part,cid)];assert [refs[j] for j in oracle]==r['oracle_neighbor_ids']
                        full=order(d,g,max(d));loc=order([d[j] for j in local],[g[j] for j in local],max(d))
                        for prefix,values in [('global',full),('local10',loc)]:
                            close(prefix+'_order',values['score'],r[prefix+'_order'])
                            for key in values:
                                if key!='score':assert values[key]==r[prefix+'_'+key],(prefix,key,cid)
                        de=sum(g[j] for j in near)/5;ds=sum(g[j] for j in oracle)/5;dr=sum(g)/len(g);eraw=de-ds
                        assert eraw>=-1e-12
                        expected={'De_pp':100*de,'Dstar_pp':100*ds,'E_pp':100*max(0.,eraw),'Drandom_pp':100*dr,
                            'De_minus_random_pp':100*(de-dr),'head_MAE_pp':100*abs(pred-labels[cid]),
                            'knn_MAE_pp':100*abs(sum(train[refs[j]] for j in near)/5-labels[cid])}
                        for key,v in expected.items():close(key,v,r[key])
                        totals['rows']+=1;totals['global_comparisons']+=full['total_pairs'];totals['performance_ties']+=full['performance_ties'];totals['distance_ties']+=full['distance_ties'];totals['NA']+=full['score'] is None;totals['negative_E_clipped']+=eraw<0
            # Recompute fixed first-plan probe from raw y/SE, never using saved margins/weights as the formula.
            data=read(root/'fixed_probe_inputs'/f's{seed}.json');ids=data['ids'];y=data['y'];se=data['se'];mean=data['y_mean'];scale=data['y_scale'];yt=[(v-mean)/scale for v in y]
            for step in [0,1,50,100,200]:
                embedded=read(folder/f'inference_add{step:03d}.json')['train_embeddings']
                reference=read(folder/'probes.json')[str(step)];total=0.;active=0;count=0
                for i in range(len(ids)):
                    for group in range(2):
                        terms=[]
                        for slot,(j,k) in enumerate(data['refs'][i][group]):
                            if j<0:continue
                            assert len({i,j,k})==3
                            a,b=abs(y[i]-y[j]),abs(y[i]-y[k]);gap=b-a;assert gap>1e-12
                            margin=min((yt[i]-yt[k])**2-(yt[i]-yt[j])**2,1.);assert margin>0
                            noise=2*se[i]+se[j]+se[k];w=gap*gap/(gap*gap+noise*noise)
                            close('raw_margin',margin,data['margins'][i][group][slot],1e-12)
                            close('raw_weight',w,data['weights'][i][group][slot],1e-12)
                            dij=sum((x-z)**2 for x,z in zip(embedded[i],embedded[j]));dik=sum((x-z)**2 for x,z in zip(embedded[i],embedded[k]))
                            hinge=max(0.,margin+dij-dik);terms.append(w*hinge);active+=hinge>0;count+=1
                        assert len(terms)==data['counts'][i][group]
                        if terms:total+=sum(terms)/len(terms)/2/len(ids)
                close('fixed_probe_order_loss_float64_vs_training_float32',total,reference['order_loss'],1e-6)
                close('fixed_probe_active_fraction',active/count if count else None,reference['active_fraction'],1e-10)
                totals['probe_states']+=1;totals['probe_triplets']+=count
    # Independently verify common-query and architecture-weighted summaries and paired seed means.
    for summary in read(root/'evaluation/summaries.json'):
        key=summary['metric'];selected=[r for r in rows if r['partition']==summary['partition'] and r['seed']==summary['seed'] and r['method']==summary['method']]
        other=[[r for r in rows if r['partition']==summary['partition'] and r['seed']==summary['seed'] and r['method']==m] for m in ['R0-8D','R0-1D','R1-8D']]
        common=set.intersection(*[{r['candidate_id'] for r in rs if r[key] is not None} for rs in other])
        selected=[r for r in selected if r['candidate_id'] in common];values=[r[key] for r in selected]
        assert len(common)==summary['valid_queries']
        if summary['weighting']=='architecture':
            architectures={r['architecture'] for r in selected};values=[sum(r[key] for r in selected if r['architecture']==a)/sum(r['architecture']==a for r in selected) for a in architectures]
        mean=sum(values)/len(values) if values else None;close('summary_'+key,mean,summary['mean'])
    summaries=read(root/'evaluation/summaries.json')
    for p in read(root/'evaluation/paired.json'):
        selected={r['method']:r['mean'] for r in summaries if all(r[k]==p[k] for k in ['partition','metric','weighting','seed'])}
        control=p['comparison'].split(' minus ')[1]
        expected=selected['R1-8D']-selected[control] if selected['R1-8D'] is not None and selected[control] is not None else None
        close('paired_'+p['metric'],expected,p['mean_delta'])
    paired_rows=read(root/'evaluation/paired.json')
    for r in read(root/'evaluation/overall.json'):
        if ' minus ' in r['method']:
            values=[p['mean_delta'] for seed in [0,1,2] for p in paired_rows if p['seed']==seed and p['comparison']==r['method'] and all(p[k]==r[k] for k in ['partition','metric','weighting'])]
        else:
            values=[p['mean'] for seed in [0,1,2] for p in summaries if p['seed']==seed and p['method']==r['method'] and all(p[k]==r[k] for k in ['partition','metric','weighting'])]
        assert len(values)==3
        for a,b in zip(values,r['seed_values']):close('overall_seed_'+r['metric'],a,b)
        valid=[v for v in values if v is not None]
        for key,value in [('mean',sum(valid)/len(valid) if valid else None),('minimum',min(valid) if valid else None),('maximum',max(valid) if valid else None)]:
            close('overall_'+key+'_'+r['metric'],value,r[key])
    result=dict(passed=True,independent_implementation='Python standard library only; no imports of project/training/evaluation functions',
        source_sha256=sha(Path(__file__)),max_absolute_differences=maxdiff,tolerances={'geometry_and_MAE_pp':1e-10,'counts_and_IDs':'exact','raw_triplet_parameters':1e-12,'probe_loss_float64_vs_float32':1e-6},counts=totals,
        optimizer_updates=0,GP_VAE_GNN_objects=0)
    destination=root/'independent_recompute.json'
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);recompute(p.parse_args().output.resolve())
