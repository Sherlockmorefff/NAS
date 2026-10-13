"""Manual arrays, metadata files, and existing frozen exports; zero model training."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import analyse.fuzzy_metric_fivefold_closure as c
from analyse.fuzzy_metric_relative_geometry import evaluate_query,select_neighbors
from analyse.fuzzy_metric_diagnosis_supplement import no_optimization
from analyse.fuzzy_metric_replay import file_hash


def minimal_pairs(overall=(.01,-.1),folds=None):
    folds=folds or [(.01,-.1)]*5
    out=[]
    for scope,values in [('overall',overall)]+[(f'fold{i}',v) for i,v in enumerate(folds)]:
        for metric,value in zip(('global_order','E_pp'),values):
            out.append({'scope':scope,'metric':metric,'subset':'all_queries','weighting':'query_equal',
                        'delta_distribution':{'mean':value}})
    return out


def test_variable_reference_counts_keep_fixed_math():
    for n in (10,141,143,144,145):
        ids=[f'{i:03d}' for i in range(n)];d=np.arange(n,dtype=float);g=np.arange(n,dtype=float)/n
        r=evaluate_query(d,g,ids,select_neighbors(d,ids,5))
        assert r['global_total_pairs']==n*(n-1)//2
        assert r['local10_total_pairs']==45
        assert r['global_order']==1 and r['local10_order']==1 and r['E_pp']==0
        assert r['Drandom_pp']==pytest.approx(100*g.mean())


def test_fold_label_and_architecture_boundary():
    candidates={'a':{'architecture_fingerprint':'A'},'b':{'architecture_fingerprint':'B'},'c':{'architecture_fingerprint':'C'}}
    fold={'family':'fold0','train_ids':['a','b'],'query_ids':['c']}
    assert c.validate_fold(fold,candidates,list(candidates))['architecture_overlap']==0
    with pytest.raises(ValueError,match='partition'):
        c.validate_fold({**fold,'train_ids':['a','c']},candidates,list(candidates))
    candidates['c']['architecture_fingerprint']='A'
    with pytest.raises(ValueError,match='architectures overlap'):
        c.validate_fold(fold,candidates,list(candidates))


def test_oof_candidate_once_and_missing_states():
    rows=[{'candidate_id':v} for v in ('a','b','c')]
    assert c.oof_identity(rows,['a','b','c'])['complete']
    assert not c.oof_identity(rows+rows[:1],['a','b','c'])['complete']
    assert c.oof_identity(rows[:2],['a','b','c'])['missing_ids']==['c']
    r=c.closure_decision(minimal_pairs(),False,[f'fold{i}' for i in range(5)])
    assert not r['material_complete'] and not r['rule_satisfied']


def test_common_na_pairing_uses_same_queries_and_reason():
    a=[{'candidate_id':str(i),'architecture_fingerprint':arch,'family':'fold0','global_order':v}
       for i,(arch,v) in enumerate([('A',.1),('A',None),('B',.2)])]
    b=[{**r,'global_order':v} for r,v in zip(a,[.3,.4,None])]
    r=c.common_pair(a,b,['0','1','2'],'global_order','query_equal')
    assert r['valid_query_count']==1 and r['excluded_query_count']==2
    assert r['delta']==pytest.approx(.2) and all('no comparable' in x['reason'] for x in r['exclusions'])
    b[0]['family']='fold1'
    with pytest.raises(ValueError,match='another fold'):c.common_pair(a,b,['0'],'global_order','query_equal')


def test_oof_primary_weights_candidates_not_folds():
    rows=[{'candidate_id':'a','architecture_fingerprint':'A','value':0},
          {'candidate_id':'b','architecture_fingerprint':'B','value':10},
          {'candidate_id':'c','architecture_fingerprint':'B','value':10}]
    assert np.mean(c.weighted_values(rows,'value','query_equal'))==pytest.approx(20/3)
    assert np.mean(c.weighted_values(rows,'value','architecture_equal'))==5


def test_closure_needs_both_overall_signs_and_four_joint_folds():
    folds=[(.01,-.1)]*4+[(-.01,.1)]
    r=c.closure_decision(minimal_pairs(folds=folds),True,[f'fold{i}' for i in range(5)])
    assert r['rule_satisfied'] and r['joint_improvement_fold_count']==4
    assert not r['enter_search'] and not r['original_stage2_passed']
    r=c.closure_decision(minimal_pairs(folds=[(.01,-.1)]*3+[(-.01,-.1),(.01,.1)]),True,[f'fold{i}' for i in range(5)])
    assert not r['rule_satisfied'] and r['joint_improvement_fold_count']==3
    for overall in ((-.01,-.1),(.01,.1)):
        assert not c.closure_decision(minimal_pairs(overall=overall),True,[f'fold{i}' for i in range(5)])['rule_satisfied']


def test_closure_numerical_zero_and_unusable_primary_fold():
    for overall in ((1e-12,-.1),(.01,-1e-10)):
        assert not c.closure_decision(minimal_pairs(overall=overall),True,[f'fold{i}' for i in range(5)])['rule_satisfied']
    assert c.closure_decision(minimal_pairs(overall=(1.01e-12,-1.01e-10)),True,[f'fold{i}' for i in range(5)])['rule_satisfied']
    r=c.closure_decision(minimal_pairs(folds=[(None,-.1)]+[(.01,-.1)]*4),True,[f'fold{i}' for i in range(5)])
    assert not r['material_complete']


def test_completed_state_uses_identity_hashes_not_latest(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OLD',tmp_path)
    task={'task_id':'fold0-refit-regression-s0','stage':'refit','family':'fold0','seed':0,'setting':{'method':'regression'}}
    parent=tmp_path/'tasks'/task['task_id'];folder=parent/'attempt02';folder.mkdir(parents=True)
    names=('model.pt','predictions_before_reveal.json','evaluation.json')
    for name in names:(folder/name).write_bytes(b'manual artifact identity fixture; no model instantiated')
    hashes={name:file_hash(folder/name) for name in names}
    done={**task,'identity':'expected','attempt':2,'directory':str(folder),'artifact_hashes':hashes}
    (folder/'completed.json').write_text(json.dumps(done));(folder/'started.json').write_text(json.dumps({'task':task,'identity':'expected'}))
    later=parent/'attempt99';later.mkdir();(later/'completed.json').write_text(json.dumps({**done,'identity':'wrong'}))
    result=c.completed_state(task,'expected')
    assert result['attempt']==2 and len(result['rejected_completions'])==1
    (folder/'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='got 0'):c.completed_state(task,'expected')


def test_no_training_manual_checks():
    with no_optimization() as guard:
        c.closure_decision(minimal_pairs(),True,[f'fold{i}' for i in range(5)])
    assert all(v==0 for v in guard.values())


def test_saved_real_30_states_and_oof():
    if not (c.OUT/'checks.json').exists():pytest.skip('real frozen output not yet computed')
    checks=json.loads((c.OUT/'checks.json').read_text())
    assert checks['completed_states']==30 and checks['query_state_rows']==1074
    assert not checks['missing_or_invalid_states'] and not checks['restore_failures']
    assert all(r['complete'] and r['count']==179 for r in checks['OOF'].values())
    assert checks['all_geometry_frozen_before_labels'] and checks['pre_label_files_unchanged']
    assert checks['Dstar_Drandom_invariant_within_fold']
    assert all(v==0 for v in checks['new_training_counts'].values())
    assert all(v==0 for v in checks['zero_optimization_guard'].values())
    assert all(r['train_embeddings_bitwise_equal'] and r['old_top5_IDs_equal'] for r in checks['per_state'])
    inventory=json.loads((c.OUT/'state_inventory.json').read_text())
    assert len(inventory)==30 and all(r['status']=='valid_completed_record' for r in inventory)
    assert all(r['task']['stage']=='refit' and r['task']['family']!='final_A' for r in inventory)
