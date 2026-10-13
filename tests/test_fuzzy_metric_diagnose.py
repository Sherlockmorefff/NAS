"""CPU synthetic engineering checks; no historical data or scientific fit ledger.

Fixture: one M2/G2 fit. Resume test: two M2 continuations (four updates).
GP test: two G4 trajectories plus one G2 continuation (ten updates).
Total synthetic optimizer updates: mapping=6, GP=12. No other test optimizes.
"""
import copy
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from analyse import fuzzy_metric_diagnose as d
from analyse.fuzzy_metric_offline_core import fit_predict
from analyse.fuzzy_metric_diagnose_saved import target_comparison,gradient_diagnostics


def label(mean,offset=.003):
    values=[mean-offset,mean,mean+offset]
    return {'validation_mean':float(np.mean(values)),'validation_se':float(np.std(values,ddof=1)/math.sqrt(3)),
        'validation_seeds':values,'training_seeds':[0,1,2]}


@pytest.fixture(scope='module')
def case():
    torch.set_num_threads(1)
    def c(i):return {'candidate_id':f'c{i:02d}','joint_fingerprint':f'c{i:02d}','architecture_fingerprint':f'a{i:02d}',
        'raw_hp':{'lr':.001*(i+1),'dropout':.02*i,'hidden_dim':32 if i%2 else 64,'l2':.0001*(i+1)}}
    rows=[c(i) for i in range(10)];base={r['architecture_fingerprint']:np.random.default_rng(i).normal(size=12).tolist() for i,r in enumerate(rows)}
    train,query=rows[:8],rows[8:];labels={r['candidate_id']:label(.5+.025*i) for i,r in enumerate(train)}
    qlabels={r['candidate_id']:label(.7+.01*i) for i,r in enumerate(query)}
    setting={'method':'N2','epsilon':.02,'lambda_f':.01}
    common={'representation_steps':2,'gp_steps':2,'learning_rate':.001,'gp_learning_rate':.05,'gp_seed':0,'lambda_e':.001,'early_stopping':False}
    bundle,state=fit_predict(train,query,base,labels,setting,0,common)
    data=d.inputs(train,query,base,labels)
    return data,labels,qlabels,state,setting


def test_training_scalers_ids_and_frozen_coordinates(case):
    data,labels,qlabels,state,setting=case
    model,opt=d.restore_mapping(state,data)
    with torch.no_grad():e,_=model(data['z'],data['hp'])
    assert torch.equal(e,state['train_embeddings'])
    assert state['standardization']==data['standardization']
    bad=copy.deepcopy(state);bad['query_ids']=state['train_ids']
    with pytest.raises(ValueError,match='inner split'):d.restore_mapping(bad,data)
    bad=copy.deepcopy(state);bad['standardization']['y_scale']*=2
    with pytest.raises(ValueError,match='scaler'):d.restore_mapping(bad,data)


def test_adam_mapping_resume_exact_and_old_snapshot_unchanged(case,tmp_path):
    data,_,_,state,setting=case
    old_hash=d.tensor_hash(state['model'])
    m,o=d.restore_mapping(state,data)
    d.mapping_step(m,o,data,setting,3)
    mid=d.mapping_state(m,o,data,setting,0,state)
    d.save(tmp_path/'mid.pt',mid)
    d.mapping_step(m,o,data,setting,4)
    expected=d.tensor_hash(m.state_dict())
    m2,o2=d.restore_mapping(d.load(tmp_path/'mid.pt'),data)
    d.mapping_step(m2,o2,data,setting,4)
    assert d.tensor_hash(m2.state_dict())==expected
    assert d.tensor_hash(state['model'])==old_hash
    # A second restore must see the original Adam state, not moments mutated by the first continuation.
    m3,o3=d.restore_mapping(state,data);d.mapping_step(m3,o3,data,setting,3)
    assert d.tensor_hash(m3.state_dict())==d.tensor_hash(mid['model'])


def test_GP_snapshot_prediction_does_not_change_prefix_or_resume(case,tmp_path):
    data,_,_,state,_=case;m,_=d.restore_mapping(state,data)
    g,o=d.new_gp(data,state['train_embeddings']);initial=d.tensor_hash(g.state_dict())
    for step in (1,2):d.gp_step(g,o,step,data)
    d.save(tmp_path/'g2.pt',{'gp':g.state_dict(),'optimizer':o.state_dict()})
    old_model=d.tensor_hash(m.state_dict());d.predict(m,g,data)
    for step in (3,4):d.gp_step(g,o,step,data)
    expected=d.tensor_hash(g.state_dict())
    g2,o2=d.new_gp(data,state['train_embeddings'])
    assert d.tensor_hash(g2.state_dict())==initial
    for step in range(1,5):d.gp_step(g2,o2,step,data)
    assert d.tensor_hash(g2.state_dict())==expected
    g3,o3=d.new_gp(data,state['train_embeddings']);mid=d.load(tmp_path/'g2.pt')
    g3.load_state_dict(mid['gp']);o3.load_state_dict(mid['optimizer'])
    for step in (3,4):d.gp_step(g3,o3,step,data)
    assert d.tensor_hash(g3.state_dict())==expected
    assert d.tensor_hash(m.state_dict())==old_model


def test_query_labels_SE_excluded_and_three_predictors_noise_exact(case,tmp_path):
    data,train,query,state,_=case;m,_=d.restore_mapping(state,data)
    g,_=d.new_gp(data,state['train_embeddings']);g.load_state_dict(state['gp'])
    before=d.predict(m,g,data);d.write(tmp_path/'prediction.json',before)
    ev=d.evaluate(before,train,query)
    changed={cid:label(.95,.02) for cid in query}
    changed_ev=d.evaluate(d.read(tmp_path/'prediction.json'),train,changed)
    assert d.predict(m,g,data)==before
    assert ev['metrics']['gp_mae']!=changed_ev['metrics']['gp_mae']
    for row in before['predictions']:
        assert row['observation_variance']==row['latent_variance']+row['pooled_training_SE_squared']+row['additional_noise_accuracy_squared']
        assert row['knn_mean']==pytest.approx(float(np.mean([train[c]['validation_mean'] for c in row['neighbor_ids']])),abs=1e-15,rel=0)
        assert len(row['neighbor_ids'])==5


def test_self_exclusion_stable_ID_ties_and_common_scale():
    e=torch.zeros(7,8);ids=[f'c{i}' for i in range(7)]
    a,_=d.neighbor_indices(e,e,ids,ids);b,_=d.neighbor_indices(3*e,3*e,ids,ids)
    assert a==b
    assert a[0]==[1,2,3,4,5]
    assert all(i not in js for i,js in enumerate(a))


def test_same_embedding_target_mass_weighted_gradients_no_updates(case):
    data,_,_,state,setting=case;m,_=d.restore_mapping(state,data)
    h=d.tensor_hash(m.state_dict());row=target_comparison(m,data,setting);grad=gradient_diagnostics(m,data,setting)
    assert row['truncated_weighted_target_mass']>=0
    assert 0<=row['truncated_fraction_of_weighted_target_mass']<=1
    assert all(math.isfinite(v) for v in grad['norms'].values())
    assert d.tensor_hash(m.state_dict())==h


def test_trigger_requires_both_conditions_and_uses_strict_training_loss():
    assert d.should_reduce_weight(.2,.1,.1,.1)
    assert not d.should_reduce_weight(.1,.1,.2,.1)
    assert not d.should_reduce_weight(.2,.1,.09,.1)


def test_completed_task_reuse_is_free_and_costs_read_only(tmp_path):
    ctx=SimpleNamespace(output=tmp_path)
    path=tmp_path/'tasks'/'regression-s0-mapping';path.mkdir(parents=True)
    d.write(path/'trajectory.json',{'seconds':1})
    d.save(path/'progress.pt',{'new_steps':200})
    d.write(path/'completed.json',{'artifacts':{'trajectory.json':d.file_hash(path/'trajectory.json')}})
    d.journal(tmp_path,{'event':'start','task_id':path.name,'kind':'mapping','attempt':'attempt01'})
    before=d.costs(ctx)
    assert d.mapping_trajectory(ctx,'regression',0)==path
    assert d.costs(ctx)==before
    assert before['mapping']['actual_completed_optimizer_steps']==200
    assert before['mapping']['attempt_starts']==1


def test_ordinary_interruption_keeps_attempt_and_failure_blocks_next_retry(tmp_path):
    ctx=SimpleNamespace(output=tmp_path)
    p,a=d.begin_task(ctx,'regression-s0-mapping','mapping')
    assert d.begin_task(ctx,'regression-s0-mapping','mapping')==(p,a)
    d.write(a/'failed.json',{'reason':'synthetic no-fit fault'})
    with pytest.raises(RuntimeError,match='repair proof'):d.begin_task(ctx,'regression-s0-mapping','mapping')
    d.write(a/'repair_verified.json',{'test':'synthetic repair verified; no fits'})
    p,b=d.begin_task(ctx,'regression-s0-mapping','mapping')
    assert b.name=='attempt02'
    with pytest.raises(ValueError,match='outside'):d.begin_task(ctx,'final_A-refit-N2-s0','mapping')
