"""Independent standard-library recomputation; no learner or primary analysis imports."""
import argparse,hashlib,json,math,statistics,time
from pathlib import Path
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def run(o):
    begun=time.monotonic();raw=read(o/'evaluation_A.json')
    y={c:math.fsum(v['validation_seeds'])/3 for c,v in raw.items()}
    for c,v in raw.items():assert abs(y[c]-v['validation_mean'])<1e-12 and v['training_seeds']==[0,1,2]
    top=max(y.values());champ={c for c in y if abs(y[c]-top)<=1e-12}
    off=[];rep=[];traces=[]
    for t in read(o/'tasks.json'):
        d=o/t['stage']/t['name']
        if t['stage']=='offline':
            rows=read(d/'predictions.json')['rows'];ordered=sorted(rows,key=lambda r:(-r['score'],r['candidate_id']))
            assert ordered==rows;ids=[r['candidate_id'] for r in rows];highest=max(map(y.get,ids));w={c for c in ids if abs(y[c]-highest)<=1e-12};N=len(ids)
            metrics={}
            for k in [1,5,10,20]:
                metrics[str(k)]=None if k>N else dict(hit=int(any(c in w for c in ids[:k])),best=max(map(y.get,ids[:k])),
                    gap_pp=100*(highest-max(map(y.get,ids[:k]))),random_hit_probability=1-math.comb(N-len(w),k)/math.comb(N,k))
            off.append(dict(task=t['name'],fold=t['fold'],seed=t['model_seed'],method=t['method'],champion_rank=next(i+1 for i,c in enumerate(ids) if c in w),K=metrics,
                head_MAE=statistics.mean(abs(r['score']-y[r['candidate_id']]) for r in rows)))
        else:
            known=set(t['initial_ids']);base=max(map(y.get,known));curve=[base];hc=[int(any(c in champ for c in known))];order=[]
            for b in range(10):
                f=d/'batches'/f'b{b:02}';s=read(f/'selection.json');v=read(f/'reveal.json');ids=[r['candidate_id'] for r in s['rows']]
                assert set(ids)==set(y)-known and ids[:5]==s['selected_ids'] and len(ids)==len(set(ids))
                assert v['selection_sha']==hashlib.sha256((f/'selection.json').read_bytes()).hexdigest()
                assert set(v['labels'])==set(ids[:5]) and set(s['known_ids'])==known
                assert s['query_count_before']==5*b and s['remaining_budget']==50-5*b
                if t['method']!='RANDOM':assert s['rows']==sorted(s['rows'],key=lambda r:(-r['score'],r['candidate_id']))
                else:
                    rnd=sorted(set(y)-set(t['initial_ids']),key=lambda c:(hashlib.sha256(f'budget_champion_v1:random:{t["search_seed"]}:{c}'.encode()).hexdigest(),c))
                    assert ids==[c for c in rnd if c not in known]
                ranks={r['candidate_id']:(i+1,r['score']) for i,r in enumerate(s['rows'])}
                traces.append(dict(task=t['name'],batch=b,champions=[dict(candidate_id=c,already_known=c in known,
                    rank=ranks[c][0] if c in ranks else None,score=ranks[c][1] if c in ranks else None,selected=c in ids[:5]) for c in sorted(champ)]))
                for c in ids[:5]:
                    assert v['labels'][c]==raw[c];known.add(c);order.append(c);curve.append(max(curve[-1],y[c]));hc.append(int(bool(known&champ)))
            def cost(threshold,strict=False):
                pred=(lambda v:v>threshold+1e-12) if strict else (lambda v:v>=threshold-1e-12)
                return 'pool_unattainable' if not pred(top) else next((i for i in range(1,51) if pred(curve[i])), '>50')
            rep.append(dict(task=t['name'],seed=t['search_seed'],method=t['method'],query_order=order,initial_best=base,initial_hit=hc[0],
                final_best=curve[-1],final_hit=hc[-1],first_champion_query=next((i for i,v in enumerate(hc) if v),'>50'),
                strict_improvement_query=cost(base,True),improve_half_pp_query=cost(base+.005),new_selected_best=max(map(y.get,order)),
                curve=curve,hit_curve=hc,curve_average=statistics.mean(curve[1:]),
                checkpoints={str(b):dict(best=curve[b],gap_pp=100*(top-curve[b]),hit=hc[b]) for b in [5,10,20,30,40,50]}))
    def stats(vals):return dict(mean=statistics.mean(vals),median=statistics.median(vals),win=sum(v>1e-12 for v in vals),tie=sum(abs(v)<=1e-12 for v in vals),loss=sum(v< -1e-12 for v in vals),values=vals)
    pairs={}
    for a,b in [('BR_HEAD','C_HEAD'),('BR_HEAD','RANDOM'),('C_HEAD','RANDOM')]:
        aa=sorted([r for r in rep if r['method']==a],key=lambda r:r['seed']);bb=sorted([r for r in rep if r['method']==b],key=lambda r:r['seed'])
        pairs[a+'-minus-'+b]={k:stats([(1 if k=='final_hit' else 100)*(x[k]-z[k]) for x,z in zip(aa,bb)]) for k in ['final_best','final_hit','curve_average']}
    summary=dict(offline={m:dict(hit10=sum(r['K']['10']['hit'] for r in off if r['method']==m),trajectories=15,
        mean_gap10_pp=statistics.mean(r['K']['10']['gap_pp'] for r in off if r['method']==m)) for m in ['C_HEAD','BR_HEAD']},
        replay={m:dict(champion_hits=sum(r['final_hit'] for r in rep if r['method']==m),trajectories=5,
            mean_final_best=statistics.mean(r['final_best'] for r in rep if r['method']==m),mean_curve=statistics.mean(r['curve_average'] for r in rep if r['method']==m)) for m in ['C_HEAD','BR_HEAD','RANDOM']},
        pool_best=top,champions=sorted(champ),initial_champion_starts=sorted({r['seed'] for r in rep if r['initial_hit']}))
    # All metrics above were derived before opening any primary result files.
    checks=0;max_error=0.
    def compare(a,b,path=''):
        nonlocal checks,max_error
        if isinstance(a,dict):
            for k,v in a.items():compare(v,b[k],path+'/'+k)
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,z) in enumerate(zip(a,b)):compare(x,z,path+'/'+str(i))
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            err=abs(a-b);max_error=max(max_error,err);assert err<1e-10,(path,a,b);checks+=1
        else:assert a==b,(path,a,b);checks+=1
    for name,data in [('offline_results',off),('replay_results',rep),('paired_results',pairs),('champion_traces',traces),('summary',summary)]:compare(data,read(o/'analysis'/f'{name}.json'),name)
    scalar_errors=[]
    def scalar(scores,truth,ids,lists,rho):
        terms=[]
        for js in lists:
            best=max(truth[j] for j in js);w=[j for j in js if abs(truth[j]-best)<=1e-12];los=[j for j in js if j not in w]
            k=min(len(js),max(1,math.ceil(len(js)*rho)))
            plus=max(scores[j] for j in w)
            neg=sorted(los,key=lambda j:(-scores[j],ids[j]))
            terms.append(0. if len(los)<k else max(0.,.1+scores[neg[k-1]]-plus)**2)
        return statistics.mean(terms)
    for v in read(o/'verification/scalar_inputs.json'):
        actual=scalar(v['scores'],v['raw_y'],v['ids'],v['lists'],v['rho']);err=abs(actual-v['ranking_loss']);scalar_errors.append(err)
        assert err<2e-6,(v['source'],actual,v['ranking_loss'])
    synthetic=[([0.,3.,2.,1.],[.9,.8,.7,.6],.5,4.41),([0.,3.,2.,1.],[.9,.8,.7,.6],.25,9.61),
        ([0.,3.,2.,1.],[.9]*4,.25,0.),([-1.,4.,3.,2.],[.9,.9,.8,.7],.25,0.),([0.,2.,2.,1.],[.9,.8,.7,.6],.25,4.41),
        ([0.,3.,2.,1.],[.9,.9,.7,.6],.75,0.),([0.,3.,2.,1.],[.9,.8,.7,.6],1.,0.)]
    for s,y0,rho,expected in synthetic:assert abs(scalar(s,y0,list('abcd'),[list(range(4))]*16,rho)-expected)<1e-12
    intents=lines(o/'ledger/mapping-intents.jsonl');receipts=lines(o/'ledger/mapping-receipts.jsonl')
    assert len(intents)==len(receipts)==20000 and len(rep)*50==750
    assert not lines(o/'ledger/gp-intents.jsonl') and not lines(o/'ledger/gp_fits-intents.jsonl')
    result=dict(passed=True,independent='Python standard library; recompute original three-seed means before reading primary aggregates',
        offline_trajectories=len(off),replay_trajectories=len(rep),curve_points=sum(len(r['curve']) for r in rep),checks=checks,
        max_metric_error=max_error,scalar_real_lists=16*len(scalar_errors),scalar_synthetic_cases=len(synthetic),max_scalar_error=max(scalar_errors),
        effective_updates=len(receipts),new_queries=750,seconds=time.monotonic()-begun)
    (o/'verification/independent.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
