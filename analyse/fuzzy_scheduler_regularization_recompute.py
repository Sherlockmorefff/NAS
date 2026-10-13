"""Independent raw-checkpoint reconstruction; performs zero optimizer updates."""
import sys,math,statistics,hashlib,json,time,collections,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import torch,numpy as np
from surrogate.fuzzy_metric import JointMetricModel
from analyse.fuzzy_metric_replay import ReplayGP
from analyse.fuzzy_convergence_recompute import gp_values,map_values,inputs

def read(p):return json.loads(p.read_text())
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(a,b,tol=1e-10):assert abs(float(a)-float(b))<=tol,(a,b)
def mean(x):return float(statistics.mean(x))
def records(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def gradient(st,shot,features,labels):
    if st['kind']=='mapping':
        m=JointMetricModel(12,seed=0);m.load_state_dict(shot['model']);z,hp=inputs(features,st['scaler']);pos={c:i for i,c in enumerate(features['ids'])};idx=[pos[c] for c in st['train_ids']]
        emb,pred=m(z[idx],hp[idx]);y=torch.tensor([labels[c]['validation_mean'] for c in st['train_ids']],dtype=torch.float64);seeds=torch.tensor([labels[c]['validation_seeds'] for c in st['train_ids']],dtype=torch.float64)
        target=((y-st['scaler']['y_mean'])/st['scaler']['y_scale']).float();eps=st['epsilon'];delta=(y[:,None]-y[None,:]).abs()
        p=(torch.exp(-4*(delta/eps).square())*(delta<eps)).float();weights=(1/(1+(((seeds[:,None,:]-seeds[None,:,:]).std(-1,correction=1)/math.sqrt(3))/eps).square())).float()
        regression=(pred-target).square().mean();norm=emb.square().sum(-1).mean();i,j=torch.triu_indices(len(y),len(y),1);clip=max(torch.finfo(emb.dtype).eps,1e-7)
        q=(1/(1+(emb[i]-emb[j]).square().sum(-1))).clamp(clip,1-clip);fuzzy=-(weights[i,j]*(p[i,j]*q.log()+(1-p[i,j])*torch.log1p(-q))).sum()/weights[i,j].sum()
        l0=regression+st['lambda_f']*fuzzy+.001*norm;theta=sum(p.square().sum() for p in m.parameters());loss=l0+st['lambda_w']*theta if st['lambda_w'] else l0
        h=st['history'][shot['step']]
        for k,v in dict(L0=l0,regression=regression,neighborhood=fuzzy,scale=norm,parameter_squared_sum=theta,total=loss).items():close(h[k],v.detach(),1e-7)
    else:
        import gpytorch
        m=ReplayGP(st['x'],st['target'],st['se'],st['scale']).double();m.load_state_dict(shot['model']);m.train();m.likelihood.train()
        with gpytorch.settings.cholesky_jitter(double_value=1e-6),gpytorch.settings.cholesky_max_tries(3),gpytorch.settings.max_cholesky_size(800):loss=-gpytorch.mlls.ExactMarginalLogLikelihood(m.likelihood,m)(m(st['x']),st['target'])
        close(loss.detach(),st['history'][shot['step']]['total'],1e-10)
    vec=torch.cat([v.detach().reshape(-1).double() for v in torch.autograd.grad(loss,list(m.parameters()))]);assert torch.isfinite(vec).all()
    delta=float((vec-shot['gradient']['vector']).abs().max());assert torch.allclose(vec,shot['gradient']['vector'],atol=1e-9,rtol=1e-7),(st['metadata'],shot['step'],delta)
    close(vec.square().mean().sqrt(),shot['gradient']['rms']);close(vec.abs().max(),shot['gradient']['max'])
    return delta


def decision(st,t,epoch,g0,reference):
    mapping=st['kind']=='mapping';width,interval,prefix=(60,20,100) if mapping else (30,10,20);start=t-width;shots={s['step']:s for s in st['audit_snapshots']};required=list(range(start,t+1,interval))
    if start<prefix or start<epoch or not all(k in shots for k in required):return dict(valid=False,passed=False,failed=['incomplete_window'])
    hs=st['history'][start:t+1];assert len(hs)==width+1 and [h['step'] for h in hs]==list(range(start,t+1));ss=[shots[k] for k in required];v=np.array([h['total'] for h in hs]);scale=max(float(np.abs(v).mean()),1e-8) if mapping else max(1.,abs(float(v.mean())))
    checks=dict(loss=float(np.ptp(v))<=(.002 if mapping else .001)*scale)
    if mapping:
        s0=max(float(np.mean([abs(h['L0']) for h in hs])),1e-8)
        for key in ['weighted_regression','weighted_neighborhood','weighted_scale','L0']:
            if key!='weighted_neighborhood' or st['lambda_f']!=0:checks[key]=float(np.ptp([h[key] for h in hs]))<=.002*s0
        if st['lambda_w']:checks['weighted_parameter']=float(np.ptp([h['weighted_parameter'] for h in hs]))<=.002*scale
    rms=max(1e-5 if mapping else 1e-6,.01*g0['rms']);maximum=max(1e-4 if mapping else 1e-5,.01*g0['max'])
    checks['gradient_rms']=max(s['gradient']['rms'] for s in ss)<=rms;checks['gradient_max']=max(s['gradient']['max'] for s in ss)<=maximum
    if mapping:
        ds=torch.stack([s['distance'] for s in ss]);checks['distance_l2']=max(float((a-b).norm())/reference['norm'] for a in ds for b in ds)<=.01
        checks['distance_pair']=float((ds.max(0).values-ds.min(0).values).max())/reference['median']<=.02
    else:
        mus=torch.stack([s['prediction'] for s in ss]);checks['prediction']=float((mus.max(0).values-mus.min(0).values).max())<=.0005
    failed=[k for k,v in checks.items() if not v];return dict(valid=True,passed=not failed,failed=failed,checks=checks)


def audit_fit(st,features,labels):
    mapping=st['kind']=='mapping';interval,prefix,cap,guard,initial_lr,min_lr=(20,100,1000,120,.001,1e-5) if mapping else (10,20,300,60,.05,1e-4)
    assert st['completed_updates']<=cap and len(st['history'])==st['completed_updates']+1
    pos={c:i for i,c in enumerate(features['ids'])};max_grad=0.;max_mu=0.
    for shot in st['audit_snapshots']:
        max_grad=max(max_grad,gradient(st,shot,features,labels))
        if mapping:
            emb,_=map_values(shot['model'],features,st['scaler']);dist=torch.pdist(emb[[pos[c] for c in st['stop_ids']]])
            assert torch.equal(dist,shot['distance'])
        else:
            mu=gp_values(dict(st,model=shot['model']),st['probe_ids']);error=float((mu-shot['prediction']).abs().max());assert error<=1e-8;max_mu=max(max_mu,error)
    g0={k:st['audit_snapshots'][0]['gradient'][k] for k in ['rms','max']};assert g0==st['scheduler']['g0'];reference=None
    if mapping:
        prefixshot=next(s for s in st['audit_snapshots'] if s['step']==100);ds=prefixshot['distance'];reference=dict(norm=max(float(ds.norm()),1e-8),median=max(float(ds.median()),1e-8));assert reference==st['scheduler']['reference']
    best=None;stagnation=stable=epoch=cooldown=0;lr=initial_lr;events=[];decisions=[];changes=[]
    for t in range(1,st['completed_updates']+1):
        h=st['history'][t];assert h['lr_used_for_update']==lr and h['lr']==lr,(st['metadata'],t,'lr used')
        if t%interval==0:
            d=decision(st,t,epoch,g0,reference);recorded=next(d for d in st['decisions'] if d['step']==t)
            for k,v in d.items():assert recorded[k]==v,(st['metadata'],t,k,recorded[k],v)
            decisions.append(d);stable=stable+1 if d['valid'] and d['passed'] else 0;action='prefix'
            if d['valid'] and d['passed']:action='stable_stop' if stable>=3 else 'passing_window_pause'
            elif t==prefix:best=h['total']
            elif t>prefix:
                threshold=max(1e-8,.002*abs(best)) if mapping else .001*max(1.,abs(best));improved=best-h['total']>threshold
                if improved:best=h['total']
                if t<=cooldown:stagnation=0;action='cooldown'
                elif improved:stagnation=0;action='improvement'
                else:
                    stagnation+=1;action='stagnation'
                    if stagnation>=3:
                        stagnation=0;new=max(min_lr,lr*.5)
                        if t>=cap:action='at_cap_no_next_update'
                        elif cap-t<guard:action='lr_reduction_blocked_insufficient_budget'
                        elif new==lr:action='minimum_learning_rate_no_reset'
                        else:
                            changes.append(dict(after_step=t,old=lr,new=new,first_new_lr_post_step=t+1,cooldown_until=t+guard));lr=new;epoch=t+1;cooldown=t+guard;stable=0;action='lr_reduced'
            event=next(e for e in st['scheduler']['events'] if e['step']==t);assert event['action']==action
            expected=dict(best=best,stagnation=stagnation,stable_count=stable,lr_epoch_start=epoch,cooldown_until=cooldown)
            assert event['after']==expected,(st['metadata'],t,event['after'],expected)
            assert event['remaining_budget']==cap-t and event['lr_configured_next']==lr
            if stable>=3:assert t==st['completed_updates']
        assert h['lr_configured_next']==lr
    assert changes==st['scheduler']['lr_changes']
    sufficient=len(decisions)>=3 and all(d['valid'] for d in decisions[-3:]);degenerate=mapping and max(labels[c]['validation_mean'] for c in st['train_ids'])>min(labels[c]['validation_mean'] for c in st['train_ids']) and float(st['audit_snapshots'][-1]['distance'].max())<=1e-8
    plateau=sufficient and all(d['failed'] and all(k.startswith('gradient_') for k in d['failed']) for d in decisions[-3:])
    status='DEGENERATE_GEOMETRY' if degenerate else 'STABLE_WITHIN_TOLERANCE' if stable>=3 else 'INSUFFICIENT_OBSERVATION' if not sufficient else 'PLATEAU_WITH_LARGE_GRADIENT' if plateau else 'CAP_REACHED_NOT_STABLE'
    assert status==st['status'];assert all(e['after_step']<st['completed_updates'] and cap-e['after_step']>=guard for e in changes)
    return dict(status=status,checks=len(decisions),max_gradient_difference=max_grad,max_snapshot_GP_difference=max_mu,final_windows_complete=sufficient,lr_reductions=len(changes),mapping_distance_stable=all(d.get('checks',{}).get('distance_l2',False) and d.get('checks',{}).get('distance_pair',False) for d in decisions[-3:]) if mapping else None)


def independent_split(features,ids,seed):
    groups=collections.defaultdict(list);arch=dict(zip(features['ids'],features['architectures']))
    for c in sorted(ids):groups[arch[c]].append(c)
    order=np.random.Generator(np.random.PCG64(seed)).permutation(sorted(groups)).tolist();bins=[[] for _ in range(3)]
    for a in order:
        sizes=[len(b) for b in bins];bins[sizes.index(min(sizes))].extend(groups[a])
    return order,[dict(inner_fold=i,validation_ids=sorted(v),train_ids=sorted(set(ids)-set(v))) for i,v in enumerate(bins)]


def independent_choice(rows):
    aggregate=[]
    for w in [0.,1e-5,1e-4,1e-3]:
        r=[r for r in rows if r['lambda_w']==w];valid=len(r)==3 and all(x['valid'] for x in r)
        if valid:aggregate.append(dict(lambda_w=w,**{k:mean([v[k] for v in r]) for k in ['r2','r1','mae']}))
    if not aggregate:return 0.,'predeclared_fallback'
    for k in ['r2','r1','mae']:
        minimum=min(x[k] for x in aggregate);aggregate=[x for x in aggregate if x[k]<=minimum+1e-12]
        if len(aggregate)==1:return aggregate[0]['lambda_w'],k
    return min(x['lambda_w'] for x in aggregate),'smaller_lambda_w'


def run(o,partial=False):
    started=time.perf_counter();tasks=read(o/'tasks.json');features=read(o/'features.json');labels=read(o/'private/evaluation_A.json');truth={c:r['validation_mean'] for c,r in labels.items()};pos={c:i for i,c in enumerate(features['ids'])}
    if not partial:assert (o/'execution_complete.json').exists()
    for reference in read(o/'prefix_reference_manifest.json')['files']:assert sha(o/reference['destination'])==reference['sha256']
    assert read(o/'import_verification.json')['passed']
    audits=[];prefix_checks=[];boundaries=[];gp_cases=0;maxerr=0.;inner=[];curves={};aliases=0
    splits=read(o/'inner_splits.json');contexts={c['context']:c for c in read(o/'contexts.json')};arch=dict(zip(features['ids'],features['architectures']))
    for name,sp in splits.items():
        order,folds=independent_split(features,contexts[name]['train_ids'],sp['seed']);assert order==sp['architecture_order'] and folds==sp['folds']
        assert digest({k:v for k,v in sp.items() if k!='identity'})==sp['identity']
        for fold in folds:assert not {arch[c] for c in fold['train_ids']}&{arch[c] for c in fold['validation_ids']}
    for t in tasks:
        folder=o/'tasks'/t['name']
        if (folder/'logical_alias.json').exists():
            alias=read(folder/'logical_alias.json');target=o/'tasks'/alias['target'];assert 'SELECT_REG' in t['name'] and read(folder/'assignment.json')['lambda_w']==0
            assert alias['source_completion_sha']==sha(target/('completed.json' if (target/'completed.json').exists() else 'failure.json'));assert alias['additional_updates']==alias['additional_oracle_calls']==0;aliases+=1;continue
        if not (folder/'completed.json').exists():
            if partial:continue
            assert (folder/'failure.json').exists() or (t['stage']=='inner_cv' and not t['sufficient']);continue
        completed=read(folder/'completed.json')
        for n,h in completed['files'].items():assert sha(folder/n)==h,(t['name'],n)
        access=read(folder/'access_boundary.json');assert len(access['denied'])>=6 and access['B_test_reads']==0 and not access['full_A_received']
        legal=read(o/t['label_file']);scaler_ids=sorted(legal)
        state_paths=sorted(folder.rglob('summary.json'))
        for f in state_paths:
            st=load(f.parent/'latest.pt');assert st['metadata']['task']==t['name']
            own=dict(legal)
            if t['stage']=='replay':
                q=st['metadata']['new_queries']
                for b in range(q//5):own.update(read(folder/'batches'/f'b{b:02}'/'reveal.json')['labels'])
            assert st['train_ids']==sorted(own)
            for c in own:assert own[c]==labels[c]
            sc=st['scaler'];fitids=sc['fit_ids']
            if t['stage']=='engineering':assert sc==load(o/t['input_directory']/'post000.pt')['scaler']
            else:assert fitids==scaler_ids
            ix=[pos[c] for c in fitids];z=torch.tensor(features['z'],dtype=torch.float32)[ix];hp=torch.tensor(features['hp'],dtype=torch.float32)[ix];yy=torch.tensor([truth[c] for c in fitids],dtype=torch.float64)
            assert sc['z_mean']==z.mean(0).tolist() and sc['z_scale']==z.std(0,correction=0).clamp_min(1e-6).tolist();assert sc['hp_mean']==hp.mean(0).tolist() and sc['hp_scale']==hp.std(0,correction=0).clamp_min(1e-6).tolist();close(sc['y_mean'],yy.mean(),0.);close(sc['y_scale'],yy.std(correction=0).clamp_min(1e-8),0.)
            if st['kind']=='mapping':
                expected=sorted(own) if st['metadata']['stage'] in ['offline','inner_cv'] else sorted(features['ids']);assert st['stop_ids']==expected
            else:
                expected=sorted(own) if st['metadata']['stage'] in ['offline','inner_cv'] else sorted(set(features['ids'])-set(own));assert st['probe_ids']==expected
                assert st['mapping_hash_before']==st['mapping_hash_after']==st['cache']['model_hash'];assert st['scaler']==st['cache']['scaler'];assert st['scaler_hash']==digest(sc)
                y=torch.tensor([truth[c] for c in sorted(own)],dtype=torch.float64);se=torch.tensor([labels[c]['validation_se'] for c in sorted(own)],dtype=torch.float64)
                assert torch.equal(st['target'],(y-sc['y_mean'])/sc['y_scale']) and torch.equal(se,st['se'])
                assert torch.equal(st['fixed_noise_normalized'],(se/sc['y_scale']).square())
                assert torch.equal(st['x'],st['cache']['embeddings'][[pos[c] for c in sorted(own)]])
                if t['stage']!='engineering':
                    mpath=folder/'mapping/latest.pt' if t['stage'] in ['offline','inner_cv'] else folder/'mapping'/f"known{st['metadata']['new_queries']//10*10:02}"/'latest.pt';ms=load(mpath);emb,head=map_values(ms['model'],features,ms['scaler']);assert torch.equal(emb,st['cache']['embeddings']) and torch.equal(head,st['cache']['head']);assert ms['model_hash']==st['mapping_hash_before']
            audit=audit_fit(st,features,labels);audits.append(dict(task=t['name'],path=str(f.relative_to(o)),kind=st['kind'],purpose=t['purpose'],steps=st['completed_updates'],**audit));boundaries.append(dict(path=str(f.relative_to(o)),train=len(own),scaler_ids=len(fitids),own_labels_only=True))
        if t['stage'] in ['offline','replay'] and read(folder/'assignment.json')['lambda_w']==0:
            prefix=folder/('mapping/post100.pt' if t['stage']=='offline' else 'mapping/known00/post100.pt');oldname=f"offline-f{t['fold']}-s{t['model_seed']}-{t['group']}" if t['stage']=='offline' else f"replay-start{t['search_seed']}-{t['group']}"
            oldfile=o/'prefix_reference'/oldname/'mapping_post100.pt'
            a,b=load(prefix),load(oldfile)
            from analyse.fuzzy_loss_interaction_core import tree_hash
            assert all(tree_hash(a[k])==tree_hash(b[k]) for k in ['model','optimizer','rng']);prefix_checks.append(dict(task=t['name'],mapping_fixed_prefix_exact=True,old_sha=sha(oldfile)))
            newgp=load(folder/('gp/post020.pt' if t['stage']=='offline' else 'batches/b00/gp/post020.pt'));oldgpfile=o/'prefix_reference'/oldname/'gp_post020.pt';oldgp=load(oldgpfile)
            same_inputs=tree_hash([newgp['x'],newgp['target'],newgp['se'],newgp['scaler']])==tree_hash([oldgp['x'],oldgp['target'],oldgp['se'],oldgp['scaler']])
            if same_inputs:
                assert all(tree_hash(newgp[k])==tree_hash(oldgp[k]) for k in ['model','optimizer','rng'])
            prefix_checks[-1].update(GP_same_inputs=same_inputs,GP_prefix_exact=True if same_inputs else None,old_GP_sha=sha(oldgpfile))

        if t['stage'] in ['offline','inner_cv']:
            pred=read(folder/'predictions.json');gp=load(folder/'gp/latest.pt');ids=[r['candidate_id'] for r in pred['rows']];mu=gp_values(gp,ids);err=max(abs(float(v)-r['gp_mean']) for v,r in zip(mu,pred['rows']));assert err<=1e-8;gp_cases+=1;maxerr=max(maxerr,err)
            if t['stage']=='inner_cv':
                sealed=read(o/'inner_scores'/(t['name']+'.json'));assert sealed['sealed_sha']==sha(folder/'completed.json') and sealed['prediction_sealed_at']<=sealed['evaluated_at'];assert set(ids)==set(t['query_ids']);rank=sorted(range(len(ids)),key=lambda i:(-float(mu[i]),ids[i]));best=max(truth[c] for c in ids)
                vals=dict(r2=best-max(truth[ids[i]] for i in rank[:2]),r1=best-truth[ids[rank[0]]],mae=mean([abs(float(v)-truth[c]) for c,v in zip(ids,mu)]))
                for k,v in vals.items():close(sealed[k],v,1e-14)
                inner.append(dict(context=t['context'],group=t['group'],lambda_w=t['lambda_w'],inner_fold=t['inner_fold'],valid=True,**vals))
    for path in (o/'selections').glob('*.json'):
        chosen=read(path);rows=[r for r in inner if r['context']==chosen['context'] and r['group']==chosen['group']]
        if len(rows)!=12:
            if partial:continue
            assert chosen['status'] in ['TUNING_FAILED_FALLBACK_ZERO','INNER_VALIDATION_INSUFFICIENT']
        else:
            w,key=independent_choice(rows);assert chosen['lambda_w']==w and chosen['deciding_metric']==key
            for aggregate in chosen['aggregates']:
                matching=[r for r in rows if r['lambda_w']==aggregate['lambda_w']]
                for k in ['r2','r1','mae']:close(aggregate[k],mean([r[k] for r in matching]),1e-14)
    if partial:
        result=dict(passed=True,partial=True,completed_physical_fits=len(audits),inner_scores=len(inner),gp_scoring_cases=gp_cases,max_gp_error=maxerr,seconds=time.perf_counter()-started)
        dest=o/'verification_outputs'/f'partial_{time.time_ns()}.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
    # Independent reconstruction of new, aliased, and imported historical receipts.
    paths=[]
    for t in tasks:
        if t['stage']!='replay':continue
        folder=o/'tasks'/t['name'];alias=read(folder/'logical_alias.json') if (folder/'logical_alias.json').exists() else None
        paths.append((t,o/'tasks'/alias['target'] if alias else folder,t['protocol_type'],False,bool(alias)))
    for imported in read(o/'reuse_manifest.json'):
        folder=o/'imports'/imported['task']['name']
        for n,h in imported['files'].items():assert sha(folder/n)==h
        t=imported['task']
        if t['stage']=='replay':paths.append((t,folder,imported['protocol'],True,False))
        else:
            pred=read(folder/'predictions-ADAPTIVE-ADAPTIVE.json');gp=load(folder/'gp-ADAPTIVE/latest.pt');mu=gp_values(gp,[r['candidate_id'] for r in pred['rows']]);err=max(abs(float(v)-r['gp_mean']) for v,r in zip(mu,pred['rows']));assert err<=1e-8;gp_cases+=1;maxerr=max(maxerr,err)
    allqueries=[];champion={c for c in truth if max(truth.values())-truth[c]<=1e-12};initials=0;newactual=0;firsts={};table=read(o/'analysis/tables/replay_queries.json');summaries=read(o/'analysis/tables/replay_summary.json')
    for t,folder,flow,imported,alias in paths:
        initial=read(o/'starts.json')[str(t['search_seed'])]['initial_ids'];assert initial==t['initial_ids'];known=set(initial);initials+=len(known);best=max(truth[c] for c in known);curve={0:best};first=0 if known&champion else None
        for batch in range(10):
            bd=folder/'batches'/f'b{batch:02}'
            if not (bd/'reveal.json').exists():break
            sel=read(bd/'selection.json');rec=read(bd/'reveal.json');assert sel['known_ids']==sorted(known) and sel['selected_ids']==rec['ids'];assert sha(bd/'selection.json')==rec['selection_sha'] and sel['saved_at']<=rec['revealed_at']
            ids=rec['ids'];assert len(set(ids))==5 and not set(ids)&known
            if t['group']!='RANDOM':
                gp=load(bd/'gp/latest.pt');assert gp['train_ids']==sorted(known);rows=sel['rows'];order=[r['candidate_id'] for r in rows];assert set(order)==set(features['ids'])-known
                mu=gp_values(gp,order);err=max(abs(float(v)-r['gp_mean']) for v,r in zip(mu,rows));assert err<=1e-8;maxerr=max(maxerr,err);gp_cases+=1
                assert [order[i] for i in sorted(range(len(order)),key=lambda i:(-float(mu[i]),order[i]))[:5]]==ids
            else:assert ids==read(o/'starts.json')[str(t['search_seed'])]['random_order'][5*batch:5*batch+5]
            for i,c in enumerate(ids):
                assert rec['labels'][c]==labels[c];known.add(c);best=max(best,truth[c]);q=5*batch+i+1;curve[q]=best
                if first is None and c in champion:first=q
                matches=[r for r in table if r['task']==t['name'] and r['flow']==flow and r['query']==q];assert len(matches)==1;assert matches[0]['candidate_id']==c;close(matches[0]['best_so_far'],best,1e-14)
                allqueries.append((t['name'],flow,q,c));newactual+=not imported and not alias
            assert rec['query_count_after']==5*(batch+1)
        for r in summaries:
            if r['task']==t['name'] and r['flow']==flow:close(r['best_validation'],curve[r['queries']],1e-14)
        end=next(r for r in summaries if r['task']==t['name'] and r['flow']==flow and r['queries']==max(curve));assert end['first_champion_query']==(first if first is not None else '>50')
        curves[(t['search_seed'],flow,t['group'])]=curve;firsts[str((t['search_seed'],flow,t['group']))]=first if first is not None else '>50'
    for row in read(o/'analysis/tables/replay_paired.json'):
        a,b=[tuple(x.split('/')) for x in row['comparison'].split(' - ')];seeds=range(2026091900,2026091905) if row['search_seed']=='all_five' else [row['search_seed']];q=row['queries'];ds=[curves[(s,*a)][q]-curves[(s,*b)][q] for s in seeds if q in curves[(s,*a)] and q in curves[(s,*b)]]
        assert row['complete']==(len(ds)==len(seeds))
        if row['complete']:close(row['difference_raw'],mean(ds),1e-14)
    # Recompute every outer metric from raw predictions rather than trust summary tables.
    for row in read(o/'analysis/tables/offline_comparison.json'):
        task=next(t for t in tasks if t['name']==row['task']) if row['flow']!='OLD_ADAPTIVE' else None
        if task:
            folder=o/'tasks'/task['name'];folder=o/'tasks'/read(folder/'logical_alias.json')['target'] if (folder/'logical_alias.json').exists() else folder;pred=read(folder/'predictions.json')
        else:pred=read(o/'imports'/row['task']/'predictions-ADAPTIVE-ADAPTIVE.json')
        rs=pred['rows'];score=row['score'];ranked=sorted(rs,key=lambda r:(-r[score],r['candidate_id']));best=max(truth[r['candidate_id']] for r in rs);rank=min(i+1 for i,r in enumerate(ranked) if best-truth[r['candidate_id']]<=1e-12)
        assert row['champion_rank']==rank;close(row['holdout_mae_raw'],mean([abs(r[score]-truth[r['candidate_id']]) for r in rs]),1e-14)
        for k in [5,10,20]:assert row[f'top{k}_champion_hit']==(rank<=k);close(row[f'top{k}_best_gap_raw'],best-max(truth[r['candidate_id']] for r in ranked[:k]),1e-14)
    # Heldout aggregates preserve the original per-query/per-seed neighbor definition.
    raw_predictions=read(o/'analysis/tables/offline_predictions.json')
    for row in read(o/'analysis/tables/offline_main.json'):
        rs=[r for r in raw_predictions if r['flow']==row['flow'] and r['group']==row['group']]
        neighbor=[]
        for seed in range(3):
            subset=[r for r in rs if r['model_seed']==seed];assert len(subset)==179
            neighbor.append(statistics.median([mean([abs(truth[r['candidate_id']]-truth[c]) for c in r['neighbor_ids']]) for r in subset]))
        close(row['primary_neighbor_raw'],mean(neighbor),1e-14)
        for score,key in [('head_mean','holdout_head_mae_raw'),('gp_mean','holdout_gp_mae_raw')]:close(row[key],mean([abs(r[score]-truth[r['candidate_id']]) for r in rs]),1e-14)
    outer=read(o/'analysis/tables/offline_comparison.json')
    for row in read(o/'analysis/tables/offline_paired.json'):
        a,b=[tuple(x.split('/')) for x in row['comparison'].split(' - ')]
        def find(key):return next((r for r in outer if (r['flow'],r['group'],r['fold'],r['model_seed'],r['score'])==(*key,row['fold'],row['model_seed'],row['score'])),None)
        a,b=find(a),find(b);assert row['complete']==(a is not None and b is not None)
        if row['complete']:
            close(row['mae_difference_raw'],a['holdout_mae_raw']-b['holdout_mae_raw'],1e-14);close(row['top5_gap_difference_raw'],a['top5_best_gap_raw']-b['top5_best_gap_raw'],1e-14)
    resources=read(o/'resource_counts.json');cost={}
    for kind,cap in [('mapping',403000),('gp',150900)]:
        intents=records(o/'ledger'/(kind+'_intents.jsonl'));receipts=records(o/'ledger'/(kind+'_receipts.jsonl'));assert intents==receipts and [r['id'] for r in intents]==list(range(1,len(intents)+1)) and len(intents)<=cap;cost[kind]=len(intents);assert resources['actual_updates'][kind]['total']==len(intents)
        complete_steps=sum(r['steps'] for r in audits if r['kind']==kind);assert len(intents)>=complete_steps
    oracle=records(o/'oracle_access.jsonl');assert len(oracle)==newactual==resources['actual_new_queries'];assert len({(r['task'],r['query']) for r in oracle})==len(oracle)
    assert len(allqueries)==resources['all_logical_queries'] and initials==700 and len(curves)==35
    result=dict(passed=True,physical_fit_count=len(audits),fit_checks=audits,label_boundaries=boundaries,prefix_checks=prefix_checks,independent_inner_scores=len(inner),independent_selections=len(list((o/'selections').glob('*.json'))),GP_scoring_cases=gp_cases,max_GP_score_error=maxerr,logical_trajectories=len(curves),all_50_complete=all(len(v)==51 for v in curves.values()),logical_queries=len(allqueries),logical_initial=initials,actual_oracle_queries=newactual,logical_alias_tasks=aliases,actual_updates=cost,first_champion_query=firsts,optimization_updates_during_recompute=0,seconds=time.perf_counter()-started)
    dest=o/'analysis/independent_recompute.json'
    if dest.exists():dest=o/'verification_outputs'/f'recompute_{time.time_ns()}.json'
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['fit_checks','label_boundaries','prefix_checks','first_champion_query']}),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);run(Path(sys.argv[1]).resolve(),'--partial' in sys.argv)
