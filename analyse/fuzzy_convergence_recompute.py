"""Read-only independent terminal-window, analytic posterior and receipt audit."""
import sys,json,hashlib,math,statistics,itertools,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
from analyse.fuzzy_metric_replay import ReplayGP
from surrogate.fuzzy_metric import JointMetricModel

def read(p):return json.loads(p.read_text())
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-10):assert abs(float(a)-float(b))<=tol,(a,b)
def mean(v):return float(statistics.mean(v))


def inputs(features,scaler):
    z=(torch.tensor(features['z'],dtype=torch.float32)-torch.tensor(scaler['z_mean']))/torch.tensor(scaler['z_scale'])
    hp=(torch.tensor(features['hp'],dtype=torch.float32)-torch.tensor(scaler['hp_mean']))/torch.tensor(scaler['hp_scale'])
    return z,hp


def map_values(model_state,features,scaler):
    m=JointMetricModel(12,seed=0);m.load_state_dict(model_state);m.eval();z,hp=inputs(features,scaler)
    with torch.no_grad():e,h=m(z,hp)
    return e.double(),h.double()*scaler['y_scale']+scaler['y_mean']


def gp_values(st,ids):
    gp=ReplayGP(st['x'],st['target'],st['se'],st['scale']).double();gp.load_state_dict(st['model']);gp.eval()
    pos={c:i for i,c in enumerate(st['cache']['ids'])};xx=st['cache']['embeddings'][[pos[c] for c in ids]];x=st['x']
    with torch.no_grad():
        cov=gp.covar_module(x).to_dense()+torch.diag(st['fixed_noise_normalized']+gp.likelihood.second_noise)
        L,info=torch.linalg.cholesky_ex(cov)
        if info.item():
            for k in range(3):
                L,info=torch.linalg.cholesky_ex(cov+torch.eye(len(x),dtype=torch.float64)*1e-6*10**k)
                if not info.item():break
        assert not info.item()
        alpha=torch.cholesky_solve((st['target']-gp.mean_module(x))[:,None],L)
        mu=(gp.mean_module(xx)+gp.covar_module(xx,x).to_dense().matmul(alpha).squeeze(1))*st['scale']+st['scaler']['y_mean']
    return mu


def gradient_at(st,snapshot,labels,features):
    if st['kind']=='mapping':
        model=JointMetricModel(12,seed=0);model.load_state_dict(snapshot['model']);z,hp=inputs(features,st['scaler']);pos={c:i for i,c in enumerate(features['ids'])};ix=[pos[c] for c in st['train_ids']]
        e,pred=model(z[ix],hp[ix]);sy=torch.tensor([labels[c]['validation_seeds'] for c in st['train_ids']],dtype=torch.float64)
        y=torch.tensor([labels[c]['validation_mean'] for c in st['train_ids']],dtype=torch.float64);target=((y-st['scaler']['y_mean'])/st['scaler']['y_scale']).float()
        eps=st['metadata'].get('epsilon')
        if eps is None:eps=st.get('epsilon')
        assert eps is not None
        delta=(y[:,None]-y[None,:]).abs();p=(torch.exp(-4*(delta/eps)**2)*(delta<eps)).float()
        weights=(1/(1+(((sy[:,None,:]-sy[None,:,:]).std(-1,correction=1)/math.sqrt(3))/eps)**2)).float()
        pred=pred.reshape(-1);regression=(pred-target).square().mean();norm=e.square().sum(-1).mean()
        i,j=torch.triu_indices(len(y),len(y),1);q=(1/(1+(e[i]-e[j]).square().sum(-1))).clamp(max(torch.finfo(e.dtype).eps,1e-7),1-max(torch.finfo(e.dtype).eps,1e-7))
        fuzzy=-(weights[i,j]*(p[i,j]*q.log()+(1-p[i,j])*torch.log1p(-q))).sum()/weights[i,j].sum()
        loss=regression+st['lambda_f']*fuzzy+.001*norm
    else:
        import gpytorch
        model=ReplayGP(st['x'],st['target'],st['se'],st['scale']).double();model.load_state_dict(snapshot['model']);model.train();model.likelihood.train()
        with gpytorch.settings.cholesky_jitter(double_value=1e-6),gpytorch.settings.cholesky_max_tries(3),gpytorch.settings.max_cholesky_size(800):
            loss=-gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)(model(st['x']),st['target'])
    gs=torch.autograd.grad(loss,list(model.parameters()));vec=torch.cat([g.detach().reshape(-1).double() for g in gs])
    assert torch.allclose(vec,snapshot['gradient']['vector'],atol=1e-9,rtol=1e-7)
    close(float(loss.detach()),st['history'][snapshot['step']]['total'],1e-7 if st['kind']=='mapping' else 1e-10)
    return dict(rms=float(vec.square().mean().sqrt()),max=float(vec.abs().max()))


def status_from_checkpoints(st,prefix,features,labels):
    kind=st['kind'];interval,width,first=(20,60,100) if kind=='mapping' else (10,30,20)
    snapshots=st['snapshots'];ss={s['step']:s for s in snapshots};g0=st['scheduler']['g0']
    pos={c:i for i,c in enumerate(features['ids'])};evidence=[]
    for s in snapshots:
        actual=gradient_at(st,s,labels,features);close(actual['rms'],s['gradient']['rms']);close(actual['max'],s['gradient']['max'])
        if kind=='mapping':
            emb,_=map_values(s['model'],features,st['scaler']);distance=torch.pdist(emb[[pos[c] for c in st['stop_ids']]])
            assert torch.equal(distance,s['distance'])
        else:
            mu=gp_values(dict(st,model=s['model']),st['probe_ids']);assert torch.allclose(mu,s['prediction'],atol=1e-8,rtol=0)
    if kind=='mapping':
        ep,_=map_values(prefix['model'],features,st['scaler']);ref=torch.pdist(ep[[pos[c] for c in st['stop_ids']]])
        norm=max(float(ref.norm()),1e-8);median=max(float(ref.median()),1e-8)
        close(norm,st['scheduler']['reference']['norm']);close(median,st['scheduler']['reference']['median'])
    end=st['completed_updates']
    for t in range(end-2*interval,end+1,interval):
        start=t-width;steps=list(range(start,t+1,interval));hs=[h for h in st['history'] if start<=h['step']<=t]
        valid=start>=first and len(hs)==width+1 and len({h['lr'] for h in hs})==1 and all(x in ss for x in steps)
        if not valid:evidence.append(dict(step=t,valid=False,passed=False,failed=['incomplete_window']));continue
        shots=[ss[x] for x in steps];vals=[h['total'] for h in hs];scale=max(mean([abs(x) for x in vals]),1e-8) if kind=='mapping' else max(1.,abs(mean(vals)))
        threshold=(.002 if kind=='mapping' else .001)*scale;checks=dict(loss=max(vals)-min(vals)<=threshold)
        if kind=='mapping':
            for k in ['weighted_regression','weighted_neighborhood','weighted_scale']:
                if k!='weighted_neighborhood' or st['lambda_f']!=0:checks[k]=max(h[k] for h in hs)-min(h[k] for h in hs)<=threshold
        gr=max(1e-5 if kind=='mapping' else 1e-6,.01*g0['rms']);gm=max(1e-4 if kind=='mapping' else 1e-5,.01*g0['max'])
        checks.update(gradient_rms=all(s['gradient']['rms']<=gr for s in shots),gradient_max=all(s['gradient']['max']<=gm for s in shots))
        if kind=='mapping':
            ds=torch.stack([s['distance'] for s in shots]);l2=max(float((a-b).norm()) for a in ds for b in ds)/norm
            pair=float((ds.max(0).values-ds.min(0).values).max())/median;checks.update(distance_l2=l2<=.01,distance_pair=pair<=.02)
        else:
            mus=torch.stack([s['prediction'] for s in shots]);checks['prediction']=float((mus.max(0).values-mus.min(0).values).max())<=.0005
        failed=[k for k,v in checks.items() if not v];evidence.append(dict(step=t,valid=True,passed=not failed,failed=failed,checks=checks))
        recorded=next(d for d in st['decisions'] if d['step']==t);assert recorded['valid'] and recorded['checks']==checks,(st['metadata'],t,recorded,checks)
    if kind=='mapping' and max(labels[c]['validation_mean'] for c in st['train_ids'])>min(labels[c]['validation_mean'] for c in st['train_ids']) and float(snapshots[-1]['distance'].max())<=1e-8:status='DEGENERATE_GEOMETRY'
    elif all(d['passed'] for d in evidence):status='STABLE_WITHIN_TOLERANCE'
    elif all(d['valid'] and d['failed'] and all(k.startswith('gradient_') for k in d['failed']) for d in evidence):status='PLATEAU_WITH_LARGE_GRADIENT'
    else:status='CAP_REACHED_NOT_STABLE'
    assert status==st['status'],(st['metadata'],status,st['status'])
    return dict(status=status,final_windows=evidence,gradient_and_snapshot_checks=len(snapshots))


def recompute(o):
    started=time.perf_counter();features=read(o/'features.json');labels=read(o/'private/evaluation_A.json');truth={c:r['validation_mean'] for c,r in labels.items()}
    tasks=read(o/'tasks.json');main=read(o/'analysis/main.json');tables=o/'analysis/tables';statuses=[];gp_count=0;maxerror=0.;holdout=[]
    for t in tasks:
        folder=o/'tasks'/t['name']
        for sp in folder.rglob('summary.json'):
            st=load(sp.parent/'latest.pt')
            if st['status']=='NUMERICAL_FAILURE':statuses.append(dict(path=str(sp.relative_to(o)),status='NUMERICAL_FAILURE',validated_failure_record=True));continue
            prefix=load(sp.parent/('post100.pt' if st['kind']=='mapping' else 'post020.pt'))
            st['epsilon']=t['epsilon'];audit=status_from_checkpoints(st,prefix,features,labels);statuses.append(dict(path=str(sp.relative_to(o)),**audit))
            initial=load(sp.parent/'post000.pt');initial['epsilon']=t['epsilon'];initial['history']=initial['history']
            g0=gradient_at(initial,initial['snapshots'][0],labels,features)
            close(g0['rms'],st['scheduler']['g0']['rms']);close(g0['max'],st['scheduler']['g0']['max'])
            if st['kind']=='mapping' and t['stage']=='offline':
                pos={c:i for i,c in enumerate(features['ids'])};ix=[pos[c] for c in t['query_ids']];dist=[];pred=[]
                for s in st['snapshots'][-4:]:
                    e,h=map_values(s['model'],features,st['scaler']);dist.append(torch.pdist(e[ix]));pred.append(h[ix])
                e0,_=map_values(prefix['model'],features,st['scaler']);d0=torch.pdist(e0[ix]);ds=torch.stack(dist);ps=torch.stack(pred)
                holdout.append(dict(task=t['name'],used_for_stopping=False,kind='mapping',final_snapshot_steps=[s['step'] for s in st['snapshots'][-4:]],
                    distance_l2=max(float((a-b).norm()) for a in ds for b in ds)/max(float(d0.norm()),1e-8),
                    distance_pair=float((ds.max(0).values-ds.min(0).values).max())/max(float(d0.median()),1e-8),prediction_range=float((ps.max(0).values-ps.min(0).values).max())))
            if st['kind']=='gp':
                assert st['mapping_hash_before']==st['mapping_hash_after']==st['cache']['model_hash']
                mapping=folder/'mapping'/('post100.pt' if st['metadata'].get('mapping_protocol')=='FIXED' else 'latest.pt') if t['stage']=='offline' else folder/'mapping'/f"known{(st['metadata']['new_queries']//10)*10:02}"/'latest.pt'
                ms=load(mapping);emb,head=map_values(ms['model'],features,ms['scaler']);assert torch.equal(emb,st['cache']['embeddings'])
                pos={c:i for i,c in enumerate(features['ids'])};assert torch.equal(st['x'],emb[[pos[c] for c in st['train_ids']]])
                assert st['scaler']==ms['scaler']
                if t['stage']=='offline':
                    mus=torch.stack([gp_values(dict(st,model=s['model']),t['query_ids']) for s in st['snapshots'][-4:]])
                    holdout.append(dict(task=t['name'],kind='gp',mapping_protocol=st['metadata']['mapping_protocol'],used_for_stopping=False,prediction_range=float((mus.max(0).values-mus.min(0).values).max())))
        if t['stage']=='offline' and (folder/'completed.json').exists():
            for mt,gt in itertools.product(['FIXED','ADAPTIVE'],repeat=2):
                rows=read(folder/f'predictions-{mt}-{gt}.json')['rows'];st=load(folder/f'gp-{mt}'/('post020.pt' if gt=='FIXED' else 'latest.pt'))
                mu=gp_values(st,[r['candidate_id'] for r in rows]);error=max(abs(float(v)-r['gp_mean']) for v,r in zip(mu,rows));assert error<=1e-8;maxerror=max(maxerror,error);gp_count+=1
                expected=[r for r in read(tables/'offline_comparison.json') if r['task']==t['name'] and r['mapping_protocol']==mt and r['gp_protocol']==gt]
                for score in ['head_mean','gp_mean']:
                    ex=next(r for r in expected if r['score']==score);close(ex['mae_raw'],mean([abs(r[score]-truth[r['candidate_id']]) for r in rows]),1e-14)
                    ranked=sorted(rows,key=lambda r:(-r[score],r['candidate_id']));best=max(truth[r['candidate_id']] for r in rows)
                    rank=min(i+1 for i,r in enumerate(ranked) if best-truth[r['candidate_id']]<=1e-12);assert rank==ex['champion_rank']
                    for k in [5,10,20]:assert ex[f'top{k}_champion_hit']==(rank<=k);close(ex[f'top{k}_best_gap_raw'],best-max(truth[r['candidate_id']] for r in ranked[:k]),1e-14)
    curves={};queries=initials=0;new=0;firsts={};query_table=read(tables/'replay_queries.json');summary=read(tables/'replay_summary.json')
    paths=[(o/'tasks'/t['name'],t,'ADAPTIVE',False) for t in tasks if t['stage']=='replay']+[(o/'imports'/r['task']['name'],r['task'],'RANDOM' if r['task']['group']=='RANDOM' else 'FIXED',True) for r in read(o/'reuse_manifest.json')['trajectories']]
    champion={c for c in truth if max(truth.values())-truth[c]<=1e-12}
    for folder,t,p,imported in paths:
        assert t['initial_ids']==read(o/'starts.json')[str(t['search_seed'])]['initial_ids']
        known=set(t['initial_ids']);initials+=len(known);best=max(truth[c] for c in known);curve={0:best};first=0 if known&champion else None
        for b in range(10):
            bd=folder/'batches'/f'b{b:02}'
            if not (bd/'reveal.json').exists():break
            s=read(bd/'selection.json');r=read(bd/'reveal.json');assert s['known_ids']==sorted(known) and r['selection_sha']==sha(bd/'selection.json') and s['saved_at']<=r['revealed_at']
            ids=s['selected_ids'];assert ids==r['ids'] and len(set(ids))==5 and not set(ids)&known
            if t['group']!='RANDOM':
                st=load(bd/'gp/latest.pt');rows=s['rows'];order=[x['candidate_id'] for x in rows];mu=gp_values(st,order)
                error=max(abs(float(v)-x['gp_mean']) for v,x in zip(mu,rows));assert error<=1e-8;maxerror=max(maxerror,error);gp_count+=1
                selected=[order[i] for i in sorted(range(len(order)),key=lambda i:(-float(mu[i]),order[i]))[:5]];assert selected==ids
            for i,c in enumerate(ids):
                known.add(c);queries+=1;new+=not imported;best=max(best,truth[c]);q=5*b+i+1;curve[q]=best
                if c in champion and first is None:first=q
                ex=next(x for x in query_table if x['task']==t['name'] and x['protocol']==p and x['query']==q)
                assert ex['candidate_id']==c;close(ex['best_so_far'],best,1e-14);close(ex['true_validation'],r['labels'][c]['validation_mean'],1e-14)
        for ex in (x for x in summary if x['task']==t['name'] and x['protocol']==p):close(ex['best_validation'],curve[ex['queries']],1e-14)
        ex=next(x for x in summary if x['task']==t['name'] and x['protocol']==p and x['queries']==max(curve))
        assert ex['first_champion_query']==(first if first is not None else '>50')
        curves[(t['search_seed'],p,t['group'])]=curve;firsts[str((t['search_seed'],p,t['group']))]=first if first is not None else '>50'
    for row in read(tables/'replay_paired.json'):
        a,b=row['comparison'].split(' - ');a=tuple(a.split('/'));b=tuple(b.split('/'));q=row['queries'];seeds=range(2026091900,2026091905) if row['search_seed']=='all_five' else [row['search_seed']]
        ds=[curves[(s,*a)][q]-curves[(s,*b)][q] for s in seeds if q in curves[(s,*a)] and q in curves[(s,*b)]]
        assert row['complete']==(len(ds)==len(seeds))
        if row['complete']:close(row['difference_raw'],mean(ds),1e-14)
    for row in read(tables/'offline_main.json'):
        rs=[r for r in read(tables/'offline_predictions.json') if (r['group'],r['mapping_protocol'],r['gp_protocol'])==(row['group'],row['mapping_protocol'],row['gp_protocol'])]
        close(row['primary_neighbor_raw'],mean([statistics.median([mean([abs(truth[r['candidate_id']]-truth[n]) for n in r['neighbor_ids']]) for r in rs if r['model_seed']==s]) for s in range(3)]),1e-14)
    result=dict(passed=True,terminal_fits=len(statuses),terminal_statuses=statuses,analytic_GP_scoring_cases=gp_count,max_analytic_GP_error=maxerror,
        logical_queries=queries,actual_new_queries=new,initial_accesses=initials,trajectory_count=len(curves),all_50_complete=all(len(v)==51 for v in curves.values()),first_champion_queries=firsts,
        offline_2x2_recomputed=True,all_paired_recomputed=True,optimization_updates=0,seconds=time.perf_counter()-started,holdout_drift=holdout)
    dest=o/'analysis/independent_recompute.json'
    if dest.exists():dest=o/'verification_outputs'/f'recompute_{time.time_ns()}.json'
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['terminal_statuses','holdout_drift','first_champion_queries']}))


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);recompute(Path(sys.argv[1]).resolve())
