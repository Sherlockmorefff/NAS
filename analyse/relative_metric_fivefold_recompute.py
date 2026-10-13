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
    lookup={(r['fold'],r['seed'],r['method'],r['partition'],r['candidate_id']):r for r in rows}
    independently_computed=[]
    maxdiff={};totals={'rows':0,'global_comparisons':0,'performance_ties':0,'distance_ties':0,'NA':0,'negative_E_clipped':0,'probe_states':0,'probe_triplets':0}
    def close(key,a,b,tol=1e-10):
        if a is None or b is None:
            if a is not b:raise ValueError('NA mismatch '+key)
            return
        diff=abs(a-b);maxdiff[key]=max(maxdiff.get(key,0.),diff)
        if diff>tol:raise ValueError(f'{key} mismatch {a} vs {b}: {diff} > {tol}')
    for task in read(root/'task_matrix.json'):
        fold,seed,branch=task['fold'],task['seed'],task['branch']
        label_input=read(root/'evaluation_inputs'/f'fold{fold}.json');train=label_input['train'];query=label_input['query']
        assert set(train)==set(task['train_ids']) and set(query)==set(task['query_ids'])
        folder=root/'trajectories'/task['task_id']
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
                    r=lookup[(fold,seed,method,part,cid)];assert [refs[j] for j in oracle]==r['oracle_neighbor_ids']
                    assert [refs[j] for j in local]==r['local10_reference_ids'] and [refs[j] for j in near]==r['embedding_neighbor_ids']
                    full=order(d,g,max(d));loc=order([d[j] for j in local],[g[j] for j in local],max(d))
                    for prefix,values in [('global',full),('local10',loc)]:
                        close(prefix+'_order',values['score'],r[prefix+'_order'])
                        for key in values:
                            if key!='score':assert values[key]==r[prefix+'_'+key],(prefix,key,cid)
                    de=sum(g[j] for j in near)/5;ds=sum(g[j] for j in oracle)/5;dr=sum(g)/len(g);eraw=de-ds
                    assert eraw>=-1e-12
                    expected={'De_pp':100*de,'Dstar_pp':100*ds,'E_pp':100*max(0.,eraw),'E_raw_accuracy':eraw,'Drandom_pp':100*dr,
                        'De_minus_random_pp':100*(de-dr),'head_MAE_pp':100*abs(pred-labels[cid]),
                        'knn_MAE_pp':100*abs(sum(train[refs[j]] for j in near)/5-labels[cid])}
                    for key,v in expected.items():close(key,v,r[key])
                    independently_computed.append({**r,**expected,'global_order':full['score'],'local10_order':loc['score']})
                    totals['rows']+=1;totals['global_comparisons']+=full['total_pairs'];totals['performance_ties']+=full['performance_ties'];totals['distance_ties']+=full['distance_ties'];totals['NA']+=full['score'] is None;totals['negative_E_clipped']+=eraw<0
        # Recompute fixed first-plan probe from raw y/SE, never using saved margins/weights as the formula.
        data=read(root/'fixed_probe_inputs'/f'fold{fold}-s{seed}.json');ids=data['ids'];y=data['y'];se=data['se'];mean=data['y_mean'];scale=data['y_scale'];yt=[(v-mean)/scale for v in y]
        for step in [0,1,50,100,200]:
            embedded=read(folder/f'inference_add{step:03d}.json')['train_embeddings']
            reference=read(folder/'probes.json')[str(step)];total=0.;active=0;count=0;group_losses=[0.,0.];group_weights=[[],[]];group_active=[0,0]
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
                        group_weights[group].append(w);group_active[group]+=hinge>0
                    assert len(terms)==data['counts'][i][group]
                    if terms:
                        contribution=sum(terms)/len(terms)/2/len(ids);total+=contribution;group_losses[group]+=contribution
            close('fixed_probe_order_loss_float64_vs_training_float32',total,reference['order_loss'],1e-6)
            close('fixed_probe_active_fraction',active/count if count else None,reference['active_fraction'],1e-10)
            for group,name in enumerate(['global','local10']):
                saved=reference['groups'][name];weights=sorted(group_weights[group]);assert len(weights)==saved['triplets']
                close('group_loss',group_losses[group],saved['half_weighted_loss'],1e-6)
                close('group_loss_fraction',group_losses[group]/total if total else None,saved['loss_fraction'],2e-6)
                close('group_weight_mean',sum(weights)/len(weights) if weights else None,saved['w_mean'],1e-7)
                for q,v in zip([0,.25,.5,.75,1],saved['w_quantiles']):
                    pos=(len(weights)-1)*q;lo=int(pos);hi=math.ceil(pos)
                    close('group_weight_quantile',weights[lo]+(weights[hi]-weights[lo])*(pos-lo),v,1e-7)
                close('group_active_fraction',group_active[group]/len(weights) if weights else None,saved['active_fraction'],1e-10)
            totals['probe_states']+=1;totals['probe_triplets']+=count
    # Independent aggregation uses the scalar recomputed metrics above.
    rows=independently_computed
    assert len({(r['fold'],r['seed'],r['method'],r['partition'],r['candidate_id']) for r in rows})==len(rows)
    if seal['full_fivefold']:
        assert sum(r['partition']=='query' for r in rows)==1611 and sum(r['partition']=='train' for r in rows)==6444
        for seed in range(3):
            for method in ['R0-8D','R0-1D','R1-8D']:
                chosen=[r['candidate_id'] for r in rows if r['partition']=='query' and r['seed']==seed and r['method']==method]
                assert len(chosen)==len(set(chosen))==179 and set(chosen)==set(protocol['A_ids'])
    def select(meta,method):
        return {(r['fold'],r['candidate_id']):r for r in rows if r['method']==method and r['seed']==meta['seed'] and r['partition']==meta['partition'] and (meta['scope']=='all' or meta['scope']=="fold"+str(r['fold']))}
    def avg(records,key,weighting):
        if not records:return None
        if weighting=='candidate':return sum(r[key] for r in records)/len(records)
        groups={}
        for r in records:groups.setdefault(r['architecture'],[]).append(r[key])
        return sum(sum(v)/len(v) for v in groups.values())/len(groups)
    summaries=read(root/'evaluation/summaries.json');paired=read(root/'evaluation/paired.json');aggregate={}
    for r in summaries:
        key=r['metric'];selected=select(r,r['method']);valid={i for i,x in selected.items() if x[key] is not None}
        expected=avg([selected[i] for i in sorted(valid)],key,r['weighting']);close('summary_'+key,expected,r['mean'])
        assert r['valid_queries']==len(valid) and r['NA_queries']==len(selected)-len(valid)
        assert r['excluded_ids']==[list(i) for i in sorted(set(selected)-valid)]
        aggregate[(r['scope'],r['partition'],key,r['weighting'],r['method'],r['seed'])]=expected
    for r in paired:
        key=r['metric'];control=r['comparison'].split(' minus ')[1];a,b=select(r,'R1-8D'),select(r,control)
        common={i for i in set(a)&set(b) if a[i][key] is not None and b[i][key] is not None}
        diff=[{'architecture':a[i]['architecture'],key:a[i][key]-b[i][key]} for i in sorted(common)]
        expected=avg(diff,key,r['weighting']);close('paired_'+key,expected,r['mean_delta'])
        assert r['valid_queries']==len(common) and r['excluded_ids']==[list(i) for i in sorted((set(a)|set(b))-common)]
        aggregate[(r['scope'],r['partition'],key,r['weighting'],r['comparison'],r['seed'])]=expected
    overall=read(root/'evaluation/overall.json')
    for r in overall:
        values=[aggregate[(r['scope'],r['partition'],r['metric'],r['weighting'],r['method'],seed)] for seed in range(3)]
        for a,b in zip(values,r['seed_values']):close('overall_seed_'+r['metric'],a,b)
        valid=[v for v in values if v is not None]
        for key,value in [('mean',sum(valid)/len(valid) if valid else None),('minimum',min(valid) if valid else None),('maximum',max(valid) if valid else None)]:
            close('overall_'+key+'_'+r['metric'],value,r[key])
    for r in read(root/'evaluation/fold_consistency.json'):
        count=0
        for fold in r['folds']:
            d={metric:sum(aggregate[('fold'+str(fold['fold']),'query',metric,r['weighting'],'R1-8D minus '+r['control'],s)] for s in range(3))/3 for metric in ['global_order','E_pp']}
            assert (d['global_order']>0 and d['E_pp']<0)==fold['both_global_C_up_E_down']
            count+=d['global_order']>0 and d['E_pp']<0
        assert count==r['both_improve_count']
    result=dict(passed=True,independent_implementation='Python standard library only; saved coordinates/IDs/A-only labels; independent pair loops, triplet formula and aggregation; no project imports',
        source_sha256=sha(Path(__file__)),max_absolute_differences=maxdiff,tolerances={'geometry_and_MAE_pp':1e-10,'counts_and_IDs':'exact','raw_triplet_parameters':1e-12,'probe_loss_float64_vs_float32':1e-6,'group_loss_fraction':2e-6,'weight_float64_vs_float32':1e-7},counts=totals,
        optimizer_updates=0,GP_VAE_GNN_objects=0)
    (root/'independent_recompute.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);recompute(p.parse_args().output.resolve())
