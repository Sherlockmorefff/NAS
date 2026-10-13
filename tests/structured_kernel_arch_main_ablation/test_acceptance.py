"""Synthetic-only acceptance: no historical labels, decoder calls, or GNN training."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import gpytorch
from analyse.structured_kernel_arch_main_ablation import gp, learner
from analyse.structured_kernel_arch_main_ablation.analysis import fixed_parameter_model, prediction_metrics
from analyse.structured_kernel_arch_main_ablation.c_features import ExactIndex, extend_dictionary, encode_frozen
from analyse.structured_kernel_arch_main_ablation.common import write_json, read_json, sha256, digest
from analyse.structured_kernel_arch_main_ablation.oracle import Ledger
from analyse.structured_kernel_phase1 import gp as old_gp
from analyse.structured_kernel_phase1.features import build_operator_graph, directed_wl, isomorphic_operator_graph
from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter, enumerate_scores, choose


@pytest.fixture(autouse=True)
def deterministic():
    gp.cpu_setup()


def tensors():
    rng=np.random.default_rng(37)
    return gp.tensor(rng.normal(size=(7,6))),gp.tensor([-.8,.2,.4,.1,1.2,-.9,.3]),gp.tensor([.001,.003,.002,.0002,.004,.002,.003])


def synthetic_public(n=16):
    rng=np.random.default_rng(91)
    ids=[f'{i:064x}' for i in range(n)]
    p={'ids':ids,'x_raw':rng.normal(size=(n,16)).tolist(),'wl':rng.normal(size=(n,2)).tolist()}
    observations=[{'candidate_id':c,'mean':.65+.12*np.sin(i),'se':.004,'seeds':[.646+.12*np.sin(i),.65+.12*np.sin(i),.654+.12*np.sin(i)]} for i,c in enumerate(ids)]
    return p,observations


def test_explicit_kernel_weights_diagonal_psd_and_five_parameters():
    x,y,k=tensors();m=gp.StaticExactGP(x,y,k,'W-HI',2)
    m.covar_module.outputscale=2.7
    base=m.covar_module.base_kernel
    ka=base.arch(x).to_dense();kh=base.hp(x).to_dense()
    expected=2.7*(kh/3+2*ka*kh/3)
    actual=m.covar_module(x).to_dense()
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(actual.diag(),torch.full((7,),2.7))
    assert torch.linalg.eigvalsh(actual).min()>-1e-12
    assert sum(p.numel() for p in m.parameters())==5
    with pytest.raises(ValueError):gp.StructuredMaternKernel(2,kernel_kind='interaction')
    with pytest.raises(ValueError):gp.StaticExactGP(x,y,k,'W-I',2)
    old=old_gp.StaticExactGP(x,y,k,'W-I',2)
    torch.testing.assert_close(old.covar_module(x).to_dense(),.25*ka+.25*kh+.5*ka*kh)
    assert not torch.allclose(actual/2.7,old.covar_module(x).to_dense())


def test_complete_posterior_and_fixed_parameter_replacement_no_optimizer(monkeypatch):
    x,y,k=tensors();old=old_gp.StaticExactGP(x,y,k,'W-I',2)
    old.mean_module.constant=.12;old.covar_module.outputscale=1.7
    old.covar_module.base_kernel.arch.lengthscale=.6
    old.covar_module.base_kernel.hp.lengthscale=1.4
    old.likelihood.second_noise=.02
    state={'x':x,'y':y,'known':k,'method':'W-I','arch_dim':2,'model':old.state_dict()}
    def forbidden(*args,**kwargs):raise AssertionError('No optimizer in fixed parameter diagnosis')
    monkeypatch.setattr(torch.optim,'Adam',forbidden);monkeypatch.setattr(torch.optim,'LBFGS',forbidden)
    new=fixed_parameter_model(state);q=x[:3]+.1
    params=gp.parameter_values(new);assert params==old_gp.parameter_values(old)
    with torch.no_grad():
        cov=new.covar_module(x).to_dense()+torch.diag(k+params['extra_noise'])
        cross=new.covar_module(q,x).to_dense()
        expected_mu=params['mean']+cross@torch.linalg.solve(cov,y-params['mean'])
        expected_cov=new.covar_module(q).to_dense()-cross@torch.linalg.solve(cov,cross.T)
        new.eval();new.likelihood.eval()
        with gp.settings(1e-8):post=new(q)
        torch.testing.assert_close(post.mean,expected_mu,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(post.covariance_matrix,expected_cov,atol=1e-10,rtol=1e-10)
        old.eval();old.likelihood.eval()
        assert not torch.allclose(old(q).mean,new(q).mean)


def test_noise_scaling_is_once_and_future_labels_not_in_training():
    public,obs=synthetic_public();t=learner.build_training(public,obs[:7],2026100200,'W-HI')
    sy=np.std([r['mean'] for r in obs[:7]],ddof=0)
    np.testing.assert_allclose(t['known_variance_standardized'],.004**2/sy**2,rtol=1e-14)
    poisoned=copy.deepcopy(public);poisoned['future_means']=[-1e9]*16;poisoned['future_SE']=[1e9]*16
    assert t==learner.build_training(poisoned,obs[:7],2026100200,'W-HI')
    x,q,arch,scales=gp.make_inputs(t,public,'W-HI')
    m=gp.StaticExactGP(x,gp.tensor(t['y_standardized']),gp.tensor(t['known_variance_standardized']),'W-HI',arch)
    assert gp.parameter_values(m)['extra_noise']==pytest.approx(.01)
    torch.testing.assert_close(m.likelihood.noise_covar.noise,gp.tensor(t['known_variance_standardized']))


def test_fit_and_sealed_query_invariant_to_future_label_se_poisoning(tmp_path):
    public,obs=synthetic_public();poisoned=copy.deepcopy(public)
    poisoned['unreleased_means']=[1e8+i for i in range(16)];poisoned['unreleased_SE']=[1e6]*16
    outputs=[]
    for i,p in enumerate((public,poisoned)):
        f,q,query=learner.fit_current(tmp_path/str(i),p,obs[:7],2026100200,'W-HI','synthetic-only')
        state=torch.load(tmp_path/str(i)/f['selected_model'],weights_only=False)
        scores,_=enumerate_scores(FrozenAdapter(gp.restore_model(state)),q,(max(o['mean'] for o in obs[:7])-f['scaler']['y_mean'])/f['scaler']['y_scale'],2026100200,1)
        sealed={'selected_start':f['selected_start'],'train_nll':[r['final_nll'] for r in f['starts']],
                'parameters':f['starts'][f['selected_start']]['parameters'],'selected_candidate':choose(query['ids'],scores),'scores':scores.tolist()}
        write_json(tmp_path/f'decision{i}.json',sealed);outputs.append(sealed)
    assert outputs[0]==outputs[1]
    assert sha256(tmp_path/'decision0.json')==sha256(tmp_path/'decision1.json')


def test_interrupted_adam_restores_exactly_completed_start_is_not_reoptimized(tmp_path,monkeypatch):
    x,y,k=tensors();baseline=gp.fit_start(x,y,k,'W-HI',2,0,tmp_path/'baseline.pt','synth')
    assert baseline['success']
    save=gp._save_torch
    def interrupt(path,state):
        save(path,state)
        if state['phase']=='ADAM' and state['adam_completed']==25:raise KeyboardInterrupt('synthetic interruption after persisted update25')
    monkeypatch.setattr(gp,'_save_torch',interrupt)
    with pytest.raises(KeyboardInterrupt):gp.fit_start(x,y,k,'W-HI',2,0,tmp_path/'resume.pt','synth')
    checkpoint=torch.load(tmp_path/'resume.pt',weights_only=False)
    assert set(checkpoint['rng'])=={'torch','numpy','python'} and checkpoint['optimizer']
    monkeypatch.setattr(gp,'_save_torch',save)
    restored=gp.fit_start(x,y,k,'W-HI',2,0,tmp_path/'resume.pt','synth')
    assert restored['success'] and restored['final_nll']==baseline['final_nll'] and restored['parameters']==baseline['parameters']
    log=(tmp_path/'resume.jsonl').read_text();assert sum(json.loads(l).get('stage')=='adam' for l in log.splitlines())==300
    hashes=[sha256(tmp_path/p) for p in ('resume.pt','resume.json','resume.jsonl')]
    def forbidden(*a,**kw):raise AssertionError('Completed start must return before constructing optimizer')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    assert gp.fit_start(x,y,k,'W-HI',2,0,tmp_path/'resume.pt','synth')==restored
    assert hashes==[sha256(tmp_path/p) for p in ('resume.pt','resume.json','resume.jsonl')]
    with pytest.raises(ValueError):gp.fit_start(x,y,k,'W-HI',2,0,tmp_path/'resume.pt','changed')


def test_top10_is_exactly_ten_and_prospective_does_not_use_Q_SE():
    ids=[f'{i:064x}' for i in range(129)]
    labels={c:{'mean':.4+i/300,'se':.003,'seeds':[]} for i,c in enumerate(ids)}
    u={'ids':ids,'mean_raw':[.65]*129,'latent_variance_standardized':[.2]*129,'extra_noise_variance_standardized':.01,'median_known_var_standardized':.02,
       'y_scale':.1,'setting_id':'synthetic','method':'W-HI','n':50,'fold':-1,'draw':1,'seed':1,'model_hash':'synthetic'}
    top=ids[-10:][::-1];first=prediction_metrics(u,labels,top)
    poisoned={c:{**r,'se':.8} for c,r in labels.items()};second=prediction_metrics(u,poisoned,top)
    assert first['top10_count']==10 and first['top10_ids']==top
    assert first['nlpd']==second['nlpd'] and first['top10_prospective_is90_pp']==second['top10_prospective_is90_pp']
    assert first['retrospective_nlpd']!=second['retrospective_nlpd']


def test_exact_typed_directed_isomorphism_and_append_only_extension():
    g=build_operator_graph({'operations':['GCNII','GCNII'],'edges':[[0,1],[1,2],[2,3]]})
    perm=[3,1,0,2];h={'nodes':[None]*4,'edges':[[perm[u],perm[v]] for u,v in g['edges']]}
    for i,p in enumerate(perm):h['nodes'][p]=g['nodes'][i]
    assert isomorphic_operator_graph(g,h) and ExactIndex([('a',g)]).match(h)==['a']
    wrong=copy.deepcopy(h);wrong['nodes'][perm[2]]['layer']=1
    assert not isomorphic_operator_graph(g,wrong)
    a={'a':g,'b':build_operator_graph({'operations':['GCNConv'],'edges':[[0,1],[1,2]]})}
    c={'c':build_operator_graph({'operations':['SAGEConv','GATConv'],'edges':[[0,1],[0,2],[1,3],[2,3]]})}
    old,manifest=directed_wl(a);extended,m=extend_dictionary(a,c,manifest,old)
    assert m['A_coordinate_reembedding_exact'] and m['A_distance_max_abs_error']<=1e-12
    for left,right in zip(manifest['dictionaries'],m['dictionaries']):assert right['labels'][:len(left['labels'])]==left['labels']
    new=copy.deepcopy(c);new['c']['nodes'][1]['aggr']='max'
    with pytest.raises(ValueError,match='ENCODING_BLOCKED'):encode_frozen(new,m['dictionaries'],m['n_max'],m['e_max'])
    hp_changed=build_operator_graph({'operations':['GCNConv'],'edges':[[0,1],[1,2]],'hidden_dim':512,'dropout':.6,'lr':.01,'l2':1e-3})
    assert hp_changed==a['b']


def test_oracle_sealed_decisions_budget_and_idempotence(tmp_path):
    public,obs=synthetic_public(80);seed='2026100200';tr=seed+'_W-HI'
    labels={r['candidate_id']:{k:r[k] for k in ('mean','se','seeds')} for r in obs}
    ledger=Ledger(tmp_path,labels,{seed:{'ids':public['ids'][:50]}})
    observed=[]
    for slot,cid in enumerate(public['ids'][:71]):
        if slot>=50:
            request={'trajectory':tr,'slot':slot,'candidate_id':cid}
            with pytest.raises(FileNotFoundError):ledger.release(request)
            write_json(tmp_path/'trajectories'/tr/'decisions'/f'{slot-49:02d}.json',{'candidate_id':cid,'observed_n':slot,'observed_ids':observed,'kind':'greedy' if slot==70 else 'qlogei'})
        request={'trajectory':tr,'slot':slot,'candidate_id':cid}
        first=ledger.release(request);assert ledger.release(request)==first;observed.append(cid)
    result=ledger.export();assert result['logical_releases']==71 and result['oracle_release_calls']==142 and result['idempotent_replays']==71
    with pytest.raises(ValueError):ledger.release({'trajectory':tr,'slot':71,'candidate_id':public['ids'][71]})
    ledger.db.close()


def test_n70_terminal_to_n71_has_one_greedy_and_no_n71_fit(tmp_path,monkeypatch):
    public,obs=synthetic_public(80);seed=2026100200;tr=f'{seed}_W-HI';folder=tmp_path/'trajectories'/tr
    write_json(tmp_path/'public_features.json',public);write_json(tmp_path/'initial_observation_manifest.json',{str(seed):{'ids':public['ids'][:50]}})
    write_json(folder/'observations.json',obs[:70]);fit_calls=[];released=[]
    def fakefit(path,p,o,s,method,identity):
        fit_calls.append(len(o));t=learner.build_training(p,o,s,method);ids=p['ids'][70:]
        query={k:p[k][70:] for k in ('ids','x_raw','wl')};x,q,arch,scales=gp.make_inputs(t,query,method)
        m=gp.StaticExactGP(x,gp.tensor(t['y_standardized']),gp.tensor(t['known_variance_standardized']),method,arch)
        state={'x':x,'y':gp.tensor(t['y_standardized']),'known':gp.tensor(t['known_variance_standardized']),'method':method,'arch_dim':arch,'kernel_kind':'hp_plus_interaction','model':m.state_dict(),'jitter_index':0}
        gp._save_torch(path/'synthetic.pt',state)
        return {'selected_model':'synthetic.pt','selected_model_sha256':sha256(path/'synthetic.pt'),'scaler':t['scaler'],'identity':'synthetic'},q,query
    class FakeOracle:
        def __init__(self,root):pass
        def release(self,trajectory,slot,cid):
            decision=read_json(folder/'decisions/21.json');assert decision['candidate_id']==cid and decision['kind']=='greedy'
            released.append((slot,cid));return obs[public['ids'].index(cid)]
        def close(self):pass
    monkeypatch.setattr(learner,'fit_current',fakefit);monkeypatch.setattr(learner,'OracleClient',FakeOracle);monkeypatch.setattr(learner,'access_guard',lambda *a:None)
    monkeypatch.setattr(learner,'enumerate_scores',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No qLogEI step21')))
    learner.run_worker(tmp_path,'synthetic',max_new_decisions=1,only_seed=seed)
    assert fit_calls==[70] and len(released)==1 and released[0][0]==70
    assert len(read_json(folder/'observations.json'))==71 and not (folder/'fits/n071').exists()


def test_learner_and_C_file_boundaries_in_independent_process(tmp_path):
    repo=tmp_path/'repo';root=repo/'results/new';root.mkdir(parents=True)
    script='''
from pathlib import Path
from analyse.structured_kernel_arch_main_ablation.oracle import access_guard
from analyse.structured_kernel_arch_main_ablation.c_prepare import guard
import sys
root,repo=map(Path,sys.argv[1:3]);role=sys.argv[3]
if role=='learner':access_guard(root,repo,role)
else:guard(root,repo)
blocked=[repo/'results/old/mixed.json',root/'analysis/future.json',root/'oracle_summary.json']
if role=='learner':blocked.append(root/'C_preparation/future.json')
for p in blocked:
    try:p.read_text()
    except PermissionError:continue
    raise AssertionError('Expected guard refusal: '+str(p))
'''
    for role in ('learner','C'):
        result=subprocess.run([sys.executable,'-B','-c',script,str(root),str(repo),role],capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_start_selection_is_training_nll_only_and_exact_tie_index():
    results=[{'success':True,'final_nll':1.,'start_id':2,'query_mae':0.}, {'success':True,'final_nll':1.,'start_id':0,'query_mae':999.}, {'success':True,'final_nll':2.,'start_id':1,'query_mae':-999.}]
    assert gp.select_start(results)==0
    for r in results:r['query_mae']=-r['query_mae']
    assert gp.select_start(results)==0
    assert choose(['b','a'],[.1,.1+1e-13])=='a'
