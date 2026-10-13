from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from analyse.structured_kernel_followup.common import CONFIG_IDS,SEEDS,sha256,read_json
from analyse.structured_kernel_followup.preflight import random_manifest,selected_mapping
from analyse.structured_kernel_followup.runner import paired_results
from analyse.structured_kernel_followup import validation_train as vt


def test_frozen_gp_components_and_candidate_weights(monkeypatch):
    from analyse.structured_kernel_phase1.gp import StaticExactGP,tensor,settings,cpu_setup
    from analyse.structured_kernel_followup.diagnose import mean_decomposition
    cpu_setup()
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('M1 must not optimize'))
    rng=np.random.default_rng(5);x=tensor(rng.normal(size=(8,6)));q=tensor(rng.normal(size=(1,6)))
    y=tensor(rng.normal(size=8));known=tensor(np.linspace(.0001,.02,8));m=StaticExactGP(x,y,known,'W-I',2)
    m.mean_module.constant=.25;m.eval();m.likelihood.eval()
    with torch.no_grad(),settings(1e-8):expected=float(m(q).mean)*.12+.65
    d=mean_decomposition(m,x,q,y,known,.12,.65,1e-8)
    assert d['mean_pp']/100==pytest.approx(expected,abs=1e-10)
    for key,row in d['points'].items():
        assert sum(row['contributions'])==pytest.approx(d['components_pp'][key],abs=1e-10)
        assert np.allclose(row['contributions'],100*.12*np.array(row['weights'])*(y.numpy()-.25))


def synthetic_manifest():
    ids=[f'candidate{i:03d}' for i in range(179)]
    initial={str(s):{'ids':ids[:50]} for s in SEEDS}
    return ids,initial,random_manifest(ids,initial)


def test_random_freeze_and_future_poison():
    ids,initial,first=synthetic_manifest()
    future={c:float(i) for i,c in enumerate(ids)}
    future={c:-1e12*v for c,v in future.items()}
    assert first==random_manifest(ids,initial)
    for s in SEEDS:
        r=first[str(s)];assert len(set(r['query_ids']))==21 and not set(r['query_ids'])&set(r['initial_ids'])
        expected=np.random.default_rng(np.random.SeedSequence([s,20261005,2])).permutation(sorted(set(ids)-set(initial[str(s)]['ids']))).tolist()[:21]
        assert r['query_ids']==expected


def synthetic_returns():
    result={}
    for i,s in enumerate(SEEDS):
        a=CONFIG_IDS[i%5];b=CONFIG_IDS[(i+1)%5]
        if i in (3,6,8):b=a
        for method,c in [('W-I',a),('RAW16',b)]:result[f'{s}_{method}']={'candidate_id':c,'configuration':{'candidate_id':c}}
    return result


def test_mapping_shares_measurements_and_keeps_all_pairs():
    mapping=selected_mapping(synthetic_returns());assert mapping['unique_configurations']==5
    values={(c,s):.5+.02*i+.001*s for i,c in enumerate(CONFIG_IDS) for s in (3,4,5)}
    pairs=paired_results(mapping['mapping'],values);assert len(pairs)==30 and len(values)==15
    for row in pairs:
        if row['search_seed'] in (2026100203,2026100206,2026100208):assert row['shared_measurement'] and row['WI_minus_RAW16_pp']==0
    bad=synthetic_returns();bad['2026100203_RAW16']['candidate_id']='wrong'
    with pytest.raises(ValueError):selected_mapping(bad)


def test_random_oracle_resume_and_budget(tmp_path):
    from analyse.structured_kernel_followup.random_reference import RandomOracle
    ids,initial,manifest=synthetic_manifest();labels={c:{'mean':i/200,'se':.001,'seeds':[i/200]*3} for i,c in enumerate(ids)}
    oracle=RandomOracle(tmp_path,labels,manifest);s=SEEDS[0];order=manifest[str(s)]['initial_ids']+manifest[str(s)]['query_ids']
    for i,c in enumerate(order):oracle.release(s,i,c)
    payload=oracle.release(s,70,order[70]);assert payload['candidate_id']==order[70]
    assert oracle.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==71
    assert oracle.db.execute('SELECT SUM(replay) FROM calls').fetchone()[0]==1
    with pytest.raises(ValueError):oracle.release(s,71,ids[-1])
    with pytest.raises(ValueError):oracle.release(s,1,ids[-1])
    oracle.db.close()


class TestForbiddenGraph:
    __test__=False
    @property
    def test_mask(self):raise AssertionError('Test scoring was accessed')


def synthetic_graph():
    torch.set_default_dtype(torch.float32);torch.set_num_threads(1)
    torch.manual_seed(33);d=TestForbiddenGraph();d.x=torch.rand(18,5)
    d.edge_index=torch.tensor([[i for i in range(18)]+[(i+1)%18 for i in range(18)],[(i+1)%18 for i in range(18)]+[i for i in range(18)]])
    d.y=torch.arange(18)%2;d.train_mask=torch.arange(18)<8;d.val_mask=(torch.arange(18)>=8)&(torch.arange(18)<14)
    return d


def config():
    return {'operations':['GCNConv','SAGEConv'],'effective_layers':2,'edges':[(0,1),(1,2),(2,3)],
        'lr':.01,'dropout':.2,'hidden_dim':8,'l2':.0005,'gat_heads':2,'sage_aggr':'mean','gin_eps':0.}


def test_original_training_numerics_and_validation_only(tmp_path):
    d=synthetic_graph();c=config();device=torch.device('cpu')
    best,valid,metadata=vt.original.train_and_eval_arch(c,d,5,2,lr=c['lr'],dropout=c['dropout'],hidden_dim=c['hidden_dim'],
        weight_decay=c['l2'],gat_heads=c['gat_heads'],sage_aggr=c['sage_aggr'],gin_eps=c['gin_eps'],device=device,
        max_epochs=12,patience=80,seed=3,track_test=False,return_metadata=True,return_reproducibility_metadata=True,return_trajectory=True)
    assert valid
    actual=vt.train_validation_only(c,d,5,2,3,device,tmp_path,'synthetic',max_epochs=12,patience=80)
    assert actual['best_validation_accuracy']==best
    assert actual['model_final_before_restore_sha256']==metadata['model_final_state_hash']
    assert actual['trajectory']==[{k:v for k,v in r.items() if k!='test_accuracy'} for r in metadata['trajectory']]
    view=vt.training_view(d);assert not hasattr(view,'test_mask') and torch.all(view.y[~(d.train_mask|d.val_mask)]==-100)
    assert actual['test_metric_computations']==actual['test_score_reads']==0


def test_training_interruption_exact_resume_and_completed_skip(tmp_path,monkeypatch):
    d=synthetic_graph();c=config();device=torch.device('cpu')
    continuous=vt.train_validation_only(c,d,5,2,4,device,tmp_path/'continuous','synth',max_epochs=12,patience=80)
    original_save=vt.save_torch
    def interrupt(path,value):
        original_save(path,value)
        if Path(path).name=='resume.pt' and value['epoch']==5 and value['result'] is None:raise KeyboardInterrupt('synthetic interruption')
    monkeypatch.setattr(vt,'save_torch',interrupt)
    with pytest.raises(KeyboardInterrupt):vt.train_validation_only(c,d,5,2,4,device,tmp_path/'resumed','synth',max_epochs=12,patience=80)
    monkeypatch.setattr(vt,'save_torch',original_save)
    resumed=vt.train_validation_only(c,d,5,2,4,device,tmp_path/'resumed','synth',max_epochs=12,patience=80)
    for key in ('trajectory','model_final_before_restore_sha256','best_validation_accuracy','best_epoch','stopped_epoch'):
        assert resumed[key]==continuous[key]
    before=sha256(tmp_path/'resumed'/'resume.pt')
    monkeypatch.setattr(vt,'epoch_core',lambda *a:pytest.fail('Completed task must not retrain'))
    assert vt.train_validation_only(c,d,5,2,4,device,tmp_path/'resumed','synth',max_epochs=12,patience=80)==resumed
    assert sha256(tmp_path/'resumed'/'resume.pt')==before
    with pytest.raises(ValueError,match='identity'):vt.train_validation_only(c,d,5,2,4,device,tmp_path/'resumed','wrong',max_epochs=12,patience=80)
