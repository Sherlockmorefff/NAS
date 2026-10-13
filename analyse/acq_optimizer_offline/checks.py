"""Checks only for the frozen protocol's specific mathematical risks."""
import ast
import math
import time
import importlib.metadata
from pathlib import Path
import numpy as np
import torch
from .common import B,AC,read,write,sha,now,ZERO,run_specs
from .geometry import Decoder,WL,digest
from .problem import Problem,matern

def compare(a,b,label):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError(label+' malformed/nonfinite')
    absolute=float(np.max(np.abs(a-b)))
    relative=float(np.max(np.abs(a-b)/np.maximum(np.abs(b),1e-15)))
    if not np.allclose(a,b,atol=1e-7,rtol=1e-6):raise AssertionError(f'{label}: abs={absolute}, relative={relative}')
    return {'max_absolute':absolute,'max_relative':relative,'PASS':True}

def prepare(root,prompt):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'PROTOCOL.json').exists():
        verify(root);return
    if (root/'runs.sqlite').exists():raise RuntimeError('Unsealed existing run database: inspect before proceeding')
    from analyse.wi_schur_B_pool300.prepare import verify as verify_b
    from analyse.wi_schur_AC_pool300.prepare import verify as verify_ac
    verify_b(B);verify_ac(AC)
    source={str(p):sha(p) for p in sorted(Path('analyse/acq_optimizer_offline').glob('*')) if p.suffix in ['.py','.sh']}
    if prompt:write(root/'prompt_reference.json',{'path':str(prompt),'sha256':sha(prompt)});(root/'PROMPT.txt').write_bytes(Path(prompt).read_bytes())
    else:raise ValueError('Preparation requires the explicit execution prompt')
    wl=WL(read(B/'groups/0/wl_vocabulary.json'));models=[Problem(f,n,wl) for n in [50,150,250] for f in ['A','WI']]
    manifest=[p.manifest() for p in models];write(root/'model_manifest.json',manifest)
    refs={}
    for p in models:
        for path in [p.cp,p.fitdir/'fit.json',p.fitdir/'training.json',p.source/'decisions'/f'{p.step:03d}.json',p.sampler_path]:refs[str(path)]=sha(path)
    for n in ['candidates.json','public_features.json','wl_vocabulary.json','seeds.json']:refs[str(B/'groups/0'/n)]=sha(B/'groups/0'/n)
    bp=read(B/'protocol.json');refs[bp['vae']['path']]=bp['vae']['sha256']
    protocol={'title':'Frozen offline acquisition optimizer comparison v1','created_at':now(),'base_seed':2026101207,
      'group':0,'stages':[50,150,250],'repeats':[0,1,2],'runs':36,'pairs':18,'families':{'A':['GRADIENT','DE'],'WI':['RANDOM','DE']},
      'models':manifest,'order':run_specs(),'domain':'u=[0,1]^16; z_arch=-2.5+5u[:12]',
      'A_relaxation':['z_arch12','-4+2.5u12','.1+.5u13','4+5u14','-5+2u15'],
      'actual_HP':'original global4 float32 equal-bin hidden[16,32,64,128,256,512],l2[1e-5,5e-5,1e-4,5e-4,1e-3]',
      'decoder':{'CPU':'float32','trials':5,'policy':'historical_modal3','salt':'acq_optimizer_offline_v1_decoder_2026101207\\0',
        'seed':'uint32 little endian first4(SHA256(salt + clipped latent12 little-endian float32 bytes))','seed_excludes':['HP','method','n','r','visit_order'],
        'old768':'unlabelled frozen architecture priority; conflicts block','per_run_new_caches_empty':True},
      'WL':{'H':2,'normalization':'all H0/H1/H2 symbol counts included in L2 norm; recursive stable semantics',
            'n_max':7,'e_max':16,'unknown_symbols':'orthogonal new axes, included in norm/distance; no dropped symbols','layout':'old phi, n/7,e/16,HP4,unknown squared norm'},
      'MC':{'q':1,'samples':512,'source':'original step 1/101/201 saved base samples','fat':True,'tau_max':.01,'tau_relu':1e-6,'eta':.001,'posterior':'latent without extra likelihood noise'},
      'budget':{'inner_vectors':512,'initial_LHS':[32,16],'postprocessing_vectors':32,'hard_seconds':300,'deadline':'external independent supervisor from first request through postprocessing I/O',
                'count_illegal_duplicate_cached':True,'fun_jac_joint_one_request':True},
      'DE':{'population':32,'generations':15,'strategy':'rand/1/bin','F':.5,'CR':.7,'updating':'deferred','bounds':'clip','polish':False,'ties':'retain parent; feasibility before score'},
      'GRADIENT':{'restarts':8,'requests_per_start':60,'method':'L-BFGS-B','maxiter':60,'maxls':20,'ftol':1e-9,'gtol':1e-6,'early_convergence_allowed':True},
      'RANDOM':{'initial':32,'independent_uniform':480,'no_replacement_resampling':True},
      'postprocessing':'own top32 distinct float64 normalized vectors; ties vector bytes; decode, quantize, exclude seen; final score then CID/bytes',
      'failure_policy':'no seed changes/no replacement models/no rerun terminal; interrupted without exact solver state -> FAILED INTERRUPTED, budget retained',
      'decision':{'positive_pairs_at_least':6,'pairs':9,'median_improvement_ratio_min':1.10,'median_time_ratio_max':2.,'uninformative_pairs_min':5,
                 'eps':'max(1e-12,1e-8*y_scale)','improvement':'y_scale*exp(realized_qLogEI), approximate smoothed MC improvement, not measured accuracy'},
      'prohibitions':ZERO,'source_identity':source,'readonly_inputs':refs,'B_freeze':read(B/'freeze.json')['identity'],'AC_freeze':read(AC/'freeze.json')['identity']}
    write(root/'PROTOCOL.json',protocol)
    from .common import db
    with db(root) as c:
        c.execute('CREATE TABLE runs(id TEXT PRIMARY KEY,ord INTEGER UNIQUE,spec TEXT,state TEXT,worker INTEGER,pid INTEGER,start REAL,end REAL,error TEXT)')
        for spec in run_specs():c.execute('INSERT INTO runs(id,ord,spec,state) VALUES(?,?,?,?)',(spec['id'],spec['order'],__import__('json').dumps(spec),'queued'))
    write(root/'status.json',{'state':'PREPARED','at':now(),**ZERO})

def verify(root):
    p=read(Path(root)/'PROTOCOL.json')
    for path,h in {**p['readonly_inputs'],**p['source_identity']}.items():
        if sha(path)!=h:raise ValueError('Frozen source/model/input changed: '+path)
    from analyse.wi_schur_B_pool300.prepare import verify as verify_b
    from analyse.wi_schur_AC_pool300.prepare import verify as verify_ac
    verify_b(B);verify_ac(AC)

def sanity(root):
    started=time.perf_counter();verify(root)
    public=read(B/'groups/0/public_features.json');pool=read(B/'groups/0/candidates.json');index={cid:i for i,cid in enumerate(public['ids'])}
    wl=WL(read(B/'groups/0/wl_vocabulary.json'));decoder=Decoder();report={'at':now(),**ZERO,'models':[]}
    old_features=[]
    for row in pool:
        a,unknown,_=wl.feature(row['configuration'])
        if unknown!=0:raise AssertionError('Frozen graph has unknown symbols')
        old_features.append(a)
        cfg,meta=decoder.decode_z(row['z_search'])
        if digest(cfg)!=row['candidate_id']:raise AssertionError('Frozen768 float32/HP/config mapping mismatch')
    report['old768_WL']=compare(old_features,public['wl'],'migrated WL old768')
    report['old768_decoder_mapping']='PASS';report['same_float32_latent_conflict']='none'
    raw=np.asarray(public['x_raw']);all_wl=np.asarray(public['wl'])
    for n in [50,150,250]:
        for family in ['A','WI']:
            p=Problem(family,n,wl);ids=p.decision['remaining_ids'];positions=[index[c] for c in ids]
            training=p.training
            sx,query,ad,scales=p.gp.make_inputs(training,{'x_raw':raw[positions].tolist(),'wl':all_wl[positions].tolist()},p.state['method'])
            record={'family':family,'n':n,'restored_learnable_state_hash':p.learned_before}
            record['training_x']=compare(sx.numpy(),p.state['x'].numpy(),'training tensor')
            record['training_y']=compare(training['y_standardized'],p.state['y'].numpy(),'training target order')
            if family=='WI':
                a=sx[:,:ad];h=sx[:,ad:]
                ka=matern(torch.cdist(a,a),p.parameters['ell_a']);kh=matern(torch.cdist(h,h),p.parameters['ell_h'])
                extended_gram=p.parameters['outputscale']*(.25*(ka+kh)+.5*ka*kh)
                record['training_Gram']=compare(extended_gram.numpy(),p.gram.numpy(),'same-kernel extended training Gram')
                extended_query=torch.cat([query,torch.zeros(len(query),1,dtype=torch.float64)],dim=1)
            else:extended_query=query;record['training_Gram']={'max_absolute':0.,'PASS':True,'same_original_adapter':True}
            original_mu=[];original_var=[];new_mu=[];new_var=[];original_score=[];new_score=[]
            with torch.no_grad():
                for i in range(0,len(query),64):
                    x=query[i:i+64].unsqueeze(-2);new=extended_query[i:i+64].unsqueeze(-2)
                    op=p.original.posterior(x);ep=p.adapter.posterior(new)
                    original_mu.extend(op.mean[:,0,0].tolist());new_mu.extend(ep.mean[:,0,0].tolist())
                    original_var.extend(op.distribution.lazy_covariance_matrix.diagonal(dim1=-2,dim2=-1)[:,0].tolist())
                    new_var.extend(ep.distribution.lazy_covariance_matrix.diagonal(dim1=-2,dim2=-1)[:,0].tolist())
                    original_score.extend(p.original_acq(x).tolist());new_score.extend(p.acq(new).tolist())
                point_scores=[p.acq(extended_query[i:i+1].unsqueeze(-2)).item() for i in range(4)]
            record['posterior_mean']=compare(new_mu,original_mu,'latent posterior mean')
            record['posterior_variance']=compare(new_var,original_var,'latent posterior variance')
            record['MC_qLogEI']=compare(new_score,original_score,'fixed MC acquisition extension')
            record['saved_decision_qLogEI']=compare(original_score,p.decision['qlogei'],'original archived acquisition')
            record['saved_decision_mean']=compare(np.asarray(original_mu)*p.scaler['y_scale']+p.scaler['y_mean'],p.decision['posterior_mean_accuracy'],'archived mean')
            record['point_batch_MC']=compare(point_scores,new_score[:4],'point/batch fixed MC')
            if family=='A':
                u=np.full(16,.43);v,g,_,_=p.relaxed(u,True);fd=[]
                for k in range(16):
                    lo=u.copy();hi=u.copy();lo[k]-=1e-6;hi[k]+=1e-6
                    fd.append((p.relaxed(hi)[0]-p.relaxed(lo)[0])/(2e-6))
                record['autograd_central_difference']=compare(g,fd,'A differentiable relaxation')
                record['gradient_DE_same_u']=compare([p.relaxed(u)[0]],[v],'A common objective')
            p.assert_unchanged();record['learnable_parameter_hash_after']=p.learned_before;report['models'].append(record)
    u=np.linspace(.21,.78,16);cfg1,meta1=decoder.decode(u);decoder.decode(1-u);cfg2,meta2=decoder.decode(u)
    decoder.reset();cfg3,meta3=decoder.decode(u)
    if cfg1 is None or digest(cfg1)!=digest(cfg2) or digest(cfg1)!=digest(cfg3):raise AssertionError('New decoder point/order/run determinism failed')
    report['new_decode_determinism']={'PASS':True,'seed':meta1['decoder_seed'],'candidate_id':digest(cfg1)}
    unknown_cfg=dict(cfg1,operations=['GCNConv']*5,edges=[[i,j] for i in range(7) for j in range(i+1,7)],effective_layers=5)
    known,unknown,symbols=wl.feature(unknown_cfg)
    if unknown<=0 or known[-1]!=21/16:raise AssertionError('New WL symbols/full norm/frozen edge scale lost')
    counts,_,_=wl.counts(unknown_cfg)
    if abs(float((known[:-2]**2).sum()+unknown)-1)>1e-12:raise AssertionError('Unknown-symbol norm incomplete')
    # Compare extended sparse distance with an explicit union-coordinate calculation.
    old_counts,_,_=wl.counts(pool[0]['configuration']);symbols_all=sorted(set(counts)|set(old_counts))
    left=np.asarray([counts[k] for k in symbols_all],dtype=float);left/=np.linalg.norm(left)
    right=np.asarray([old_counts[k] for k in symbols_all],dtype=float);right/=np.linalg.norm(right)
    dense_d=float(((left-right)**2).sum());sparse_d=float(((known[:-2]-np.asarray(old_features)[0,:-2])**2).sum()+unknown)
    report['new_symbols_exact_distance']=compare([sparse_d],[dense_d],'unknown symbols exact orthogonal extension')
    report['new_symbols_exact_distance'].update(unknown_symbols=len(symbols),unknown_norm2=unknown,edge_ratio=known[-1])
    p=Problem('A',50,wl);a=p.realized(cfg1,meta1['z_float32'])[0];b=p.realized(cfg3,meta3['z_float32'])[0]
    report['A_common_postprocess']=compare([a],[b],'same u actual landing')
    report['budget_checks']=budget_checks(root)
    report['environment']={n:importlib.metadata.version(n) for n in ['torch','gpytorch','botorch','scipy','numpy']}
    report['state']='PASS';report['elapsed_seconds']=time.perf_counter()-started;write(root/'sanity_report.json',report)
    return report

def budget_checks(root):
    from .algorithms import Run,BudgetStop,de_trials,better
    from .common import tensor_hash
    class FakeProblem:
        seen=set();family='A';posterior_calls=0;learned_before='fake';scaler={'y_scale':1.};mc_seed=1
        def relaxed(self,u,gradient=False):
            self.posterior_calls+=1;v=-float(np.sum((u-.4)**2));return v,-2*(u-.4) if gradient else None,0.,0.
        def realized(self,cfg,z):self.posterior_calls+=1;return -float(np.sum((np.asarray(z)-.4)**2)),0.,0
        def assert_unchanged(self):return True
        def manifest(self):return {'MC_base_tensor_hash':'fake'}
    class FakeDecoder:
        pool_ids=set();decode_calls=map_hits=cache_hits=0
        def reset(self):self.decode_calls=0
        def decode(self,u):
            self.decode_calls+=1
            if u[0]<.1:return None,{'reason':'fake_illegal'}
            cfg={'fake':np.asarray(u).tolist()};return cfg,{'z_float32':np.asarray(u).tolist(),'candidate_id':digest(cfg)}
    reports=[]
    for family,method in [('A','DE'),('WI','DE'),('WI','RANDOM'),('A','GRADIENT')]:
        p=FakeProblem();p.family=family
        spec={'id':'fake_'+family+'_'+method,'family':family,'method':method,'n':50,'replicate':0,'initial_seed':1,'optimizer_seed':2}
        f=root/'sanity_budget'/'runs'/spec['id']
        if f.exists():
            import shutil
            shutil.rmtree(f)
        run=Run(root/'sanity_budget',spec,p,FakeDecoder());out=run.execute()
        if method in ['DE','RANDOM'] and out['requests']!=512:raise AssertionError('Explicit vector count mismatch')
        if out['requests']>512 or out['post_requests']>32:raise AssertionError('Budget exceeded')
        if method=='GRADIENT' and (out['gradient_calls']!=sum(x['requests'] for x in out['gradient_starts']) or any(x['requests']>60 for x in out['gradient_starts'])):raise AssertionError('Joint fun+jac count mismatch')
        run.requests=512
        try:run.request(np.full(16,.5));raise AssertionError('Budget not enforced')
        except BudgetStop:enforced=True
        reports.append({'family':family,'method':method,'requests':out['requests'],'post_requests':out['post_requests'],'gradient_calls':out['gradient_calls'],'cap_enforced':enforced})
    if better({'valid':False,'score':None},{'valid':True,'score':-1e7}):raise AssertionError('Penalty beat feasible extreme score')
    if better({'valid':True,'score':1.},{'valid':True,'score':1.}):raise AssertionError('DE replaced on tie')
    import subprocess,sys
    probe=subprocess.Popen([sys.executable,'-c','import time;time.sleep(2)']);started=time.monotonic()
    try:probe.wait(timeout=.05);raise AssertionError('Timeout probe unexpectedly exited')
    except subprocess.TimeoutExpired:probe.kill();probe.wait()
    return {'PASS':True,'algorithms':reports,'external_timeout_probe_seconds':time.monotonic()-started,'timeout_independent_of_algorithm':True}
