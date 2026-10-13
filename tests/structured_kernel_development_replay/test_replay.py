import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from analyse.structured_kernel_phase1 import gp
from analyse.structured_kernel_phase1.io import read_json,write_json,digest,sha256
from analyse.structured_kernel_development_replay.protocol import eligibility,SEEDS,METHODS,protocol
from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter,enumerate_scores,choose,mc_seed
from analyse.structured_kernel_development_replay.data import initial_ids
from analyse.structured_kernel_development_replay.learner import build_training,seal
from analyse.structured_kernel_development_replay.oracle import Ledger
from analyse.structured_kernel_development_replay.report import gate,bootstrap


@pytest.fixture(autouse=True)
def cpu():
    gp.cpu_setup()


def toy():
    rng=np.random.default_rng(713)
    public={'ids':[f'candidate{i:03d}' for i in range(90)],'x_raw':rng.normal(size=(90,16)).tolist(),
        'wl':rng.normal(size=(90,7)).tolist()}
    labels={cid:{'mean':float(.7+.05*np.sin(i)), 'se':.002+i*.000001,'seeds':[.69,.7,.71]} for i,cid in enumerate(public['ids'])}
    return public,labels


def test_purpose_is_not_an_independence_bypass():
    assert eligibility('independent_confirmation','B')=='BLOCKED'
    assert eligibility('development_replay','A179')=='ELIGIBLE'
    for kw in ({'b_access':True},{'test_access':True}):
        with pytest.raises(ValueError):eligibility('development_replay','A179',**kw)
    with pytest.raises(ValueError):eligibility('development_replay','B')


@pytest.mark.parametrize('method',METHODS)
def test_adapter_direct_covariance_noise_dtype_autograd_and_chunking(method):
    rng=np.random.default_rng(43)
    x=gp.tensor(rng.normal(size=(8,16)));y=gp.tensor(rng.normal(size=8));known=gp.tensor(np.linspace(0,1e-7,8))
    model=gp.StaticExactGP(x,y,known,method,12)
    adapter=FrozenAdapter(model)
    q=gp.tensor(rng.normal(size=(69,16))).requires_grad_()
    p=adapter.posterior(q[:,None,:])
    with gp.settings(1e-8):
        direct=model(q[:,None,:])
        covariance=direct.lazy_covariance_matrix.to_dense()
    torch.testing.assert_close(p.mean[...,0],direct.mean,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(p.distribution.covariance_matrix,covariance,rtol=1e-12,atol=1e-12)
    assert p.mean.shape==(69,1,1) and p.mean.dtype==torch.float64
    assert torch.equal(model.likelihood.noise_covar.noise,known)
    p.mean.sum().backward();assert bool(torch.isfinite(q.grad).all())
    with pytest.raises(ValueError):adapter.posterior(q[:,None,:],observation_noise=True)
    with pytest.raises(ValueError):adapter.posterior(q[:,None,:].float())
    with pytest.raises(ValueError):adapter.posterior(q)
    full,base=enumerate_scores(adapter,q.detach(),float(y.max()),SEEDS[0],1,chunk_size=100)
    ids=[f'id{i:03d}' for i in range(69)]
    for chunk in (1,17,64):
        partial,other=enumerate_scores(adapter,q.detach(),float(y.max()),SEEDS[0],1,chunk_size=chunk)
        np.testing.assert_allclose(full,partial,atol=1e-10,rtol=1e-12)
        assert torch.equal(base,other) and choose(ids,full)==choose(ids,partial)


def test_seed_and_tie_contract():
    assert mc_seed(SEEDS[0],1)==int(np.random.SeedSequence([SEEDS[0],1,1]).generate_state(1,dtype=np.uint32)[0])
    with pytest.raises(ValueError):mc_seed(SEEDS[0],0)
    assert choose(['b','a','c'],[1.,1.-5e-13,1.-2e-12])=='a'
    assert choose(['b','a'],[1.,1.-2e-12])=='b'


def test_future_label_se_poisoning_cannot_change_fits_scores_or_choice(tmp_path):
    public,labels=toy();future=copy.deepcopy(labels)
    ids=public['ids'][:8]
    for cid in public['ids'][8:]:future[cid].update(mean=-1e9,se=1e12)
    outputs=[]
    for copy_id,oracle in enumerate((labels,future)):
        observed=[dict(candidate_id=cid,**oracle[cid]) for cid in ids]
        training=build_training(public,observed,SEEDS[0],'RAW16')
        x,q,arch,_=gp.make_inputs(training,public,'RAW16')
        results=[]
        for start in range(3):
            results.append(gp.fit_start(x,training['y_standardized'],training['known_variance_standardized'],
                'RAW16',arch,start,tmp_path/f'{copy_id}_{start}.pt','synthetic',resume=True))
        assert all(r['success'] for r in results)
        selected=gp.select_start(results)
        state=torch.load(tmp_path/f'{copy_id}_{selected}.pt',weights_only=False)
        scores,base=enumerate_scores(FrozenAdapter(gp.restore_model(state)),q[8:],max(training['y_standardized']),SEEDS[0],1)
        outputs.append((training,[r['final_nll'] for r in results],selected,scores,choose(public['ids'][8:],scores)))
    assert outputs[0][0]==outputs[1][0] and outputs[0][1:3]==outputs[1][1:3]
    assert np.array_equal(outputs[0][3],outputs[1][3]) and outputs[0][4]==outputs[1][4]


def test_optimizer_interruption_restores_same_cold_start(tmp_path,monkeypatch):
    rng=np.random.default_rng(291)
    x=gp.tensor(rng.normal(size=(8,16)));y=gp.tensor(rng.normal(size=8));known=gp.tensor(np.ones(8)*.0001)
    continuous=gp.fit_start(x,y,known,'RAW16',16,0,tmp_path/'continuous.pt','toy')
    original=gp._save_torch
    def interrupt(path,state):
        original(path,state)
        if state['phase']=='ADAM' and state['adam_completed']==25:
            raise KeyboardInterrupt('synthetic crash after checkpoint')
    with monkeypatch.context() as patch:
        patch.setattr(gp,'_save_torch',interrupt)
        with pytest.raises(KeyboardInterrupt):gp.fit_start(x,y,known,'RAW16',16,0,tmp_path/'resumed.pt','toy')
    resumed=gp.fit_start(x,y,known,'RAW16',16,0,tmp_path/'resumed.pt','toy')
    assert continuous['success'] and resumed['success']
    assert continuous['final_nll']==resumed['final_nll']
    assert continuous['parameters']==resumed['parameters']
    before=sha256(tmp_path/'resumed.pt')
    again=gp.fit_start(x,y,known,'RAW16',16,0,tmp_path/'resumed.pt','toy')
    assert again==resumed and sha256(tmp_path/'resumed.pt')==before
    with pytest.raises(ValueError):gp.fit_start(x,y,known,'RAW16',16,0,tmp_path/'resumed.pt','changed')


def test_oracle_lost_reply_idempotency_and_full_paired_budget(tmp_path):
    public,labels=toy();seed=SEEDS[0]
    initial={str(seed):{'ids':initial_ids(public['ids'],seed)}}
    ledger=Ledger(tmp_path,labels,initial)
    histories=[]
    for method in METHODS:
        tr=f'{seed}_{method}';ids=[]
        for slot in range(71):
            cid=initial[str(seed)]['ids'][slot] if slot<50 else min(set(public['ids'])-set(ids))
            if slot>=50:
                seal(tmp_path/'trajectories'/tr/'decisions'/f'{slot-49:02d}.json',dict(candidate_id=cid,observed_n=slot,
                    observed_ids=ids.copy(),kind='greedy' if slot==70 else 'qlogei'))
            request=dict(trajectory=tr,slot=slot,candidate_id=cid)
            value=ledger.release(request)
            if method==METHODS[1] and slot in (0,49,50,69,70):
                # Simulated lost response: reopen persisted ledger, retry exact
                # token before the learner can append the returned observation.
                ledger.db.close();ledger=Ledger(tmp_path,labels,initial)
                assert ledger.release(request)==value
            ids.append(cid)
        histories.append(ids)
    assert histories[0]==histories[1] and len(set(histories[0]))==71
    summary=ledger.export()
    assert summary['logical_releases']==142 and summary['oracle_release_calls']==147 and summary['idempotent_replays']==5
    with pytest.raises(ValueError):ledger.release(dict(trajectory=tr,slot=71,candidate_id=public['ids'][0]))
    with pytest.raises(ValueError):ledger.release(dict(trajectory=tr,slot=50,candidate_id='B_candidate'))
    ledger.db.close()
    p=protocol()['budget'];assert p['fit_n']==list(range(50,71)) and p['terminal_n']==71


def test_access_guard_rejects_B_and_learner_A_source(tmp_path):
    for role,path in [('oracle','results/private/evaluation_B.json'),('learner','results/private/evaluation_A.json'),('scorer','results/node_test.json')]:
        code='from pathlib import Path\nfrom analyse.structured_kernel_development_replay.oracle import access_guard\nr=Path.cwd()\naccess_guard(r/"results/synthetic",r,'+repr(role)+')\nopen(r/'+repr(path)+')'
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        assert result.returncode!=0 and 'PermissionError' in result.stderr


def test_main_gate_not_rescued_by_other_metrics():
    assert gate([.4]*10,20)[0]=='DEVELOPMENT_REPLAY_GAIN_SUPPORTED'
    assert gate([.29]*10,20)[0]=='DEVELOPMENT_REPLAY_NO_STABLE_GAIN'
    assert not gate([.5]*10,19)[1]['20_complete_trajectories']
    assert bootstrap([.2]*10)==pytest.approx([.2,.2])


def test_immutable_seal_refuses_overwrite(tmp_path):
    path=tmp_path/'decision.json';seal(path,{'a':1});before=sha256(path)
    seal(path,{'a':1});assert sha256(path)==before
    with pytest.raises(ValueError):seal(path,{'a':2})
