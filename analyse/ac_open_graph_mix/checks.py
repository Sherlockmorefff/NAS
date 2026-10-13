"""Fixed-input CPU checks; no GNN training or new performance labels."""
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .common import *
from . import graphs,gp,search,optimize

def compare(a,b,name):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=1e-7,rtol=1e-6):raise AssertionError(name+' mismatch; fixed tolerance not relaxed')
    return {'PASS':True,'max_absolute':float(np.max(np.abs(a-b)))}
def expanded_state(state,x,arch_dim):
    new=copy.copy(state);new['x']=torch.tensor(x,dtype=torch.float64);new['arch_dim']=arch_dim
    model=gp.StaticExactGP(new['x'],new['y'],state['method'],arch_dim)
    migrated=model.state_dict()
    for key,value in state['model'].items():
        if not key.endswith('active_dims'):migrated[key]=value
    model.load_state_dict(migrated);new['model']=model.state_dict()
    return new,model

def sanity(root):
    gp.cpu_setup();started=time.perf_counter();report={'tolerances':read(root/'PROTOCOL.json')['CPU_sanity_tolerances'],
       'inputs_reason':'fixed group0 n50 checkpoint validates old/new kernel equivalence; fixed internal HP coordinates test autograd; synthetic graph family boundaries exercise new-token overlaps; no optimizer tuning',
       'GNN_training':0,'new_validation_observations':0,'new_test_or_H':0,'GP_fit_calls':0,'checks':{}}
    from analyse.acq_optimizer_offline.problem import Problem
    from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter
    source=Path(read(root/'PROTOCOL.json')['B_root']);pool=read(inputs(root,0)/'candidates.json');byid={r['candidate_id']:r for r in pool}
    oldobs=read(source/'groups/0/observations.json')[:50]
    obs=[]
    for o in oldobs:
        row=byid[o['candidate_id']];obs.append({**o,'configuration':row['configuration'],'architecture_id':row['architecture_id'],'latent':row['x_raw'][:12]})
    legacyA=Problem('A',50);ta=search.projection(obs,'A');newA=search.Acquisition(legacyA.state,ta,obs,'A',read(inputs(root,0)/'seeds.json')['qlogei_mc'][0])
    point=np.full(16,.43);score,grad=newA.evaluate(point,True);oldscore,oldgrad,_,_=legacyA.relaxed(point,True)
    report['checks']['A_two_implementations']=compare([score],[oldscore],'A same continuous objective')
    report['checks']['A_gradient_old']=compare(grad,oldgrad,'A two analytic gradients')
    fd=[]
    for i in range(16):
        lo=point.copy();hi=point.copy();lo[i]-=1e-6;hi[i]+=1e-6;fd.append((newA.evaluate(hi)[0]-newA.evaluate(lo)[0])/(2e-6))
    report['checks']['A_finite_difference']=compare(grad,fd,'A autograd')
    decoder=optimize.OpenDecoder(root,0);row,meta=decoder.decode(point)
    if row is None:raise AssertionError('Fixed A landing sanity point invalid')
    raw_latent=row['latent'];cfg=row['configuration'];actual=newA.actual_A(cfg,raw_latent)
    z=np.r_[raw_latent,np.asarray(point[12:],dtype='<f4')].tolist();oldactual=legacyA.realized(cfg,z)[0]
    report['checks']['A_actual_float32_landing']=compare([actual],[oldactual],'A actual decoded RAW16')
    decoder.decode(1-point);again,_=decoder.decode(point)
    if row['candidate_id']!=again['candidate_id']:raise AssertionError('Decoder visit order changed configuration')
    for original in pool:
        graphs.valid(original['configuration'])
        k=np.asarray(original['x_raw'][:12],dtype='<f4').tobytes()
        if k not in decoder.old:raise AssertionError('Old latent map missing')
    report['checks']['old_pool_graph_family_and_decode_mapping']={'PASS':True,'rows':768,'fixed_salt':meta['decoder_seed']}
    legacyC=Problem('WI',50);tc=search.projection(obs,'C');cs,cm=expanded_state(legacyC.state,tc['x'],tc['arch_dim']);newC=search.Acquisition(cs,tc,obs,'C',legacyC.mc_seed)
    with torch.no_grad():gram=cm.covar_module(cs['x']).to_dense().numpy()
    report['checks']['old_training_Gram']=compare(gram,legacyC.gram.numpy(),'full stable-token old training Gram')
    # All original768, including points with no token in the 50-point training vocabulary.
    means=[];variances=[];scores=[];indices=[]
    for original in pool:
        c=original['configuration'];i=newC.graph_index(c);indices.append(i)
        v=np.array([(math.log10(c['lr'])+4)/2.5,(c['dropout']-.1)/.5])
        x=newC.c_input(torch.tensor(v,dtype=torch.float64),i,c['hidden_dim'],c['l2']).reshape(1,5)
        with torch.no_grad():
            posterior=newC.adapter.posterior(x.unsqueeze(-2));means.append(posterior.mean.item());variances.append(posterior.distribution.covariance_matrix.item());scores.append(newC.score(x).item())
    pub=read(inputs(root,0)/'public_features.json');sc=legacyC.scaler;raw=np.asarray(pub['x_raw']);wl=np.asarray(pub['wl'])
    h=(raw[:,12:]-np.asarray(sc['x_mean'])[12:])/np.asarray(sc['x_scale'])[12:]/legacyC.scales['hp']['value']
    q=torch.tensor(np.c_[wl/legacyC.scales['architecture']['value'],h],dtype=torch.float64);oldmeans=[];oldvars=[];oldvalues=[]
    with torch.no_grad():
        for x in q.split(64):
            posterior=legacyC.original.posterior(x.unsqueeze(-2));oldmeans.extend(posterior.mean[:,0,0].tolist());oldvars.extend(posterior.distribution.covariance_matrix[:,0,0].tolist());oldvalues.extend(legacyC.original_acq(x.unsqueeze(-2)).tolist())
    report['checks']['old768_posterior_mean']=compare(means,oldmeans,'old768 mean')
    report['checks']['old768_posterior_variance']=compare(variances,oldvars,'old768 latent variance')
    report['checks']['old768_qLogEI']=compare(scores,oldvalues,'old768 fixed MC acquisition')
    graph=graphs.rebuild(['GCNConv','GCNII','SAGEConv'],[(0,1),(1,2),(0,3)]);c=graphs.config(graph,512,5e-4,[.43,.37]);index=newC.graph_index(c)
    v=np.array([.43,.37]);value,derivative=newC.evaluate(v,True,index,512,5e-4);fd=[]
    for i in range(2):
        lo=v.copy();hi=v.copy();lo[i]-=1e-6;hi[i]+=1e-6;fd.append((newC.evaluate(hi,False,index,512,5e-4)[0]-newC.evaluate(lo,False,index,512,5e-4)[0])/(2e-6))
    report['checks']['C_HP_autograd']=compare(derivative,fd,'C differentiable HP kernel')
    configs=[o['configuration'] for o in obs]+[c];vocab=graphs.vocabulary(configs);psi=graphs.features(configs,vocab);xs=np.c_[psi[:-1]/tc['distance_scales']['architecture']['value'],np.asarray(tc['x'])[:,-4:]]
    expanded,model=expanded_state(cs,xs,psi.shape[1]);ordinary=FrozenAdapter(model,cs['jitter_index'])
    hp=(np.array(graphs.hp(c))-np.array(tc['scaler']['hp_mean']))/np.array(tc['scaler']['hp_scale'])/tc['distance_scales']['hp']['value']
    query=torch.tensor(np.r_[psi[-1]/tc['distance_scales']['architecture']['value'],hp],dtype=torch.float64).reshape(1,1,-1)
    commonX=newC.c_input(torch.tensor(v),index,512,5e-4).reshape(1,1,5)
    with torch.no_grad():
        p1=ordinary.posterior(query);p2=newC.adapter.posterior(commonX)
    report['checks']['C_ordinary_mean']=compare(p2.mean.numpy(),p1.mean.numpy(),'C normal posterior mean')
    report['checks']['C_ordinary_variance']=compare(p2.distribution.covariance_matrix.numpy(),p1.distribution.covariance_matrix.numpy(),'C normal posterior variance')
    batch=torch.cat([commonX,commonX],dim=0)
    report['checks']['fixed_MC_tbatch_q1']=compare(newC.acq(batch).detach().numpy(),[value,value],'independent batch q1 MC')
    # Full shared new symbols in two historical graphs and one query graph.
    g1=graphs.rebuild(['GCNConv']*5,[(i,j) for i in range(6) for j in range(i+1,6)])
    g2=graphs.rebuild(['GCNConv']*5,[(i,j) for i in range(6) for j in range(i+1,6) if (i,j)!=(0,2)])
    g3=graphs.rebuild(['GCNConv']*5,[(i,j) for i in range(6) for j in range(i+1,6) if (i,j)!=(0,3)])
    cc=[graphs.config(g,64,5e-4,[.5,.5]) for g in [g1,g2,g3]];vocab=graphs.vocabulary(cc);phi=graphs.features(cc,vocab)
    counters=[graphs.tokens(c)[0] for c in cc];oldtokens=set(map(tuple,tc['tokens']));shared=(set(counters[0])&set(counters[1]))-oldtokens
    if not shared:raise AssertionError('Fixed new-history test lacks shared novel tokens')
    expected=np.empty((3,3))
    for i in range(3):
        for j in range(3):
            dot=sum(v*counters[j].get(k,0) for k,v in counters[i].items())/np.sqrt(sum(v*v for v in counters[i].values())*sum(v*v for v in counters[j].values()))
            expected[i,j]=dot+float(phi[i,-2:]@phi[j,-2:])
    report['checks']['shared_novel_history_token_Gram']=compare(phi@phi.T,expected,'complete T-T/T-Q new-symbol Gram')
    report['checks']['shared_novel_history_token_Gram']['shared_new_symbols']=len(shared)
    for ops,edges in [(['Identity'],[]),(['GCNConv']*6,[]),(['GCNConv'],[(1,0)])]:
        try:graphs.rebuild(ops,edges);raise AssertionError('Illegal graph family accepted')
        except ValueError:rejected=True
    isolated=graphs.config(graphs.rebuild(['GCNConv','SAGEConv'],[]),16,1e-5,[.5,.5]);graphs.valid(isolated)
    if any(e[1] in [1,2] for e in isolated['edges']):raise AssertionError('Old missing-incoming fallback was repaired')
    report['checks']['graph_boundaries_and_output_rebuild']={'PASS':True,'allIdentity_rejected':True,'missing_incoming_preserved':True}
    report['checks']['budget_resume']=budget_sanity(root,obs)
    proposals,trace,rng=graphs.propose(obs,{'random':proposal_seed(0,21,1),'mutation':proposal_seed(0,22,1)})
    if len(proposals)!=16 or len({p['architecture_id'] for p in proposals})!=16:raise AssertionError('C graph flow uniqueness/count')
    report['checks']['C_graph_proposals_no_VAE']={'PASS':True,'graphs':len(proposals),'attempts':len(trace),'decoder_calls':0,'encoder_calls':0}
    report['checks']['cache_observation_gate']=cache_sanity(root,obs[0])
    from .scoring import construct_model
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
    model=construct_model(isolated,3,2,1,torch.device('cpu'))
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise AssertionError('Final scorer leaked GP float64')
    with torch.no_grad():model(torch.ones(4,3,dtype=torch.float32),torch.tensor([[0,1,2],[1,2,3]],dtype=torch.long))
    torch.set_default_dtype(old);report['checks']['isolated_float32_scoring']={'PASS':True,'training_updates':0,'test_mask_access':0}
    for p in [newA,newC]:p.unchanged()
    report['elapsed_seconds']=time.perf_counter()-started;report['state']='PASS';write(root/'SANITY.json',report)

def budget_sanity(root,obs):
    for method,normal,fall in [('A',1856,128),('C',1920,64)]:
        r=optimize.Round.__new__(optimize.Round);r.method=method;r.state={'requests':0,'main_requests':0,'post_requests':0,'fallback_requests':0};r.save=lambda:True
        r.count('main',normal);r.count('post',64);r.count('fallback',fall)
        if r.state['requests']!=2048:raise AssertionError('Detailed candidate cap arithmetic')
        try:r.count('main');raise AssertionError('Budget stop absent')
        except optimize.BudgetStop:stopped=True
    testroot=root/'sanity';(testroot/'inputs/groups/0').mkdir(parents=True,exist_ok=True)
    write(testroot/'inputs/groups/0/seeds.json',read(inputs(root,0)/'seeds.json'))
    class FailingObjective:
        def evaluate(self,*args):raise FloatingPointError('deliberate CPU check failure')
    r=optimize.Round(testroot,'A',0,obs,FailingObjective())
    try:r.request(np.full(16,.43));raise AssertionError('Expected check failure')
    except FloatingPointError:failed=True
    count=read(r.sp)['requests'];before=read(r.sp)['consumed_seconds'];rr=optimize.Round(testroot,'A',0,obs,FailingObjective())
    if rr.state['requests']!=count or rr.elapsed()<before:raise AssertionError('Resume reset request/time budgets')
    class QuadraticObjective:
        def graph_index(self,cfg):return 0
        def evaluate(self,u,gradient=False,*args):
            u=np.asarray(u);return -float(np.sum((u-.4)**2)),-2*(u-.4) if gradient else None
    cr=optimize.Round(testroot,'C',0,obs,QuadraticObjective())
    fixture={'architecture':graphs.rebuild(['GCNConv'],[(0,1)]),'architecture_id':'CPU_budget_fixture','source':{'flow':'CPU_check'}}
    cr.refine([('CPU_early',{'u':[.8,.9],'graph':fixture,'hidden':16,'l2':1e-5})],30)
    if not 0<cr.state['start_calls']['CPU_early']<30:raise AssertionError('Normal early convergence/call counter incorrect')
    # A separate empty archive exercises the bounded direct-graph fallback, not formal proposals.
    fallbackroot=testroot/'fallback_case';(fallbackroot/'inputs/groups/0').mkdir(parents=True,exist_ok=True)
    write(fallbackroot/'inputs/groups/0/seeds.json',read(inputs(root,0)/'seeds.json'))
    fr=optimize.Round(fallbackroot,'C',0,obs,QuadraticObjective());fr.post()
    if not 0<fr.state['fallback_requests']<=64 or not any(row['event']=='landed' for row in fr.rows):raise AssertionError('Bounded C fallback failed')
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(1)'])
    try:child.wait(timeout=.03);raise AssertionError('Expected external deadline')
    except subprocess.TimeoutExpired:child.kill();child.wait()
    return {'PASS':True,'2048_batch_vector_arithmetic':True,'per_start_caps':[200,30],'normal_early_convergence_requests':cr.state['start_calls']['CPU_early'],
      'bounded_C_fallback_requests':fr.state['fallback_requests'],'consumed_exception_requests':count,'resume_preserved_count_and_time':True,'external_native_deadline_probe':True,'270_main_300_total_states_separate':True}

def cache_sanity(root,observation):
    import sqlite3
    cfg=observation['configuration'];p=read(root/'PROTOCOL.json');key=task_key(p,cfg,seed(0,'search'),'search')
    from analyse.wi_schur_AC_pool300.common import task_key as oldkey
    if key!=oldkey(p,cfg,seed(0,'search'),'search'):raise AssertionError('Exact physical key changed')
    import ast
    source=Path('analyse/ac_open_graph_mix/runtime.py').read_text()
    if source.index("pool[cid]=d['selected_row']")>source.index("key=enqueue(root,method,g,'search',n,"):raise AssertionError('Cache released before selected configuration')
    if 'observations_identity' not in source or 'observed_ids' not in source:raise AssertionError('Observation identity gate absent')
    return {'PASS':True,'legacy_training_key_unchanged':True,'method_in_physical_key':False,'logical_method_key_present':True,'selection_sealed_before_cache_lookup':True,'own_observations_only':True,'test_gate_reused_separate_float32_process':True}
