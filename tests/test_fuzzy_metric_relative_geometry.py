"""Manual arrays and existing frozen artifacts only; no synthetic model training."""
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
from analyse.fuzzy_metric_relative_geometry import (
    OUT, order_score, select_neighbors, attainable_gap, evaluate_query, pair_common,
)
from analyse.fuzzy_metric_diagnosis_supplement import no_optimization


def test_same_and_reverse_order():
    assert order_score([1,2,3],[.01,.02,.03])['score']==1.
    assert order_score([3,2,1],[.01,.02,.03])['score']==0.


def test_all_distance_ties_are_half_including_zero_scale():
    for distances in ([3,3,3],[0,0,0]):
        r=order_score(distances,[.01,.02,.03])
        assert r['score']==.5 and r['distance_ties']==3 and r['valid_denominator']==3


def test_performance_ties_excluded_and_na():
    r=order_score([1,2,3],[.1,.1,.3])
    assert r['performance_ties']==1 and r['valid_denominator']==2 and r['score']==1.
    r=order_score([1,2,3],[.1,.1+1e-13,.1])
    assert r['score'] is None and r['valid_denominator']==0 and r['NA_reason']


def test_distance_numerical_ties_half_and_scale_invariance():
    for scale in (1e-8,1.,1e8):
        r=order_score(np.array([1,1+5e-11,3])*scale,[.01,.02,.03])
        assert r['score']==pytest.approx(5/6) and r['distance_ties']==1
    # Local scoring uses the same full-row maximum, not its subset maximum.
    assert order_score([1e-9,2e-9,3e-9],[.01,.02,.03],scale=100)['score']==.5


def test_retrieval_exact_ties_use_ids_but_scoring_does_not():
    ids=['c','a','b']
    assert select_neighbors([1,1,1],ids,2)==[1,2]
    assert order_score([1,1,1],[3,1,2])['score']==.5
    assert select_neighbors([1+1e-12,1,1+2e-12],ids,3)==[1,0,2]


def test_best_attainable_gap_and_clamp():
    ids=[str(j) for j in range(10)]
    d=np.arange(10,dtype=float);g=np.arange(10,dtype=float)/100
    assert evaluate_query(d,g,ids,select_neighbors(d,ids,5))['E_pp']==0.
    r=evaluate_query(d,g[::-1],ids,select_neighbors(d,ids,5))
    assert r['E_pp']==pytest.approx(5.)
    assert attainable_gap(0.,5e-13)==(0.,-5e-13)
    with pytest.raises(ValueError,match='negative excess'):
        attainable_gap(0.,2e-12)


def test_random_expected_mean_gap_is_not_prediction_mae():
    differences=np.array([.02,.04,.06,.08,.10,.12])
    all_five=[float(differences[list(c)].mean()) for c in itertools.combinations(range(6),5)]
    assert np.mean(all_five)==pytest.approx(differences.mean(),abs=1e-15)
    query=.5;neighbor_means=np.array([.3,.4,.5,.6,.7])
    assert np.abs(neighbor_means-query).mean()>abs(neighbor_means.mean()-query)


def test_common_valid_pairing_and_architecture_equal():
    a=[{'candidate_id':str(i),'architecture_fingerprint':arch,'value':v}
       for i,(arch,v) in enumerate([('a',0.),('a',None),('b',2.),('a',0.)])]
    b=[{**r,'value':v} for r,v in zip(a,[1.,3.,None,3.])]
    r=pair_common(a,b,'value','query_equal')
    assert r['delta']==2. and r['excluded_query_count']==2 and r['valid_query_count']==2
    r=pair_common(a,b,'value','architecture_equal')
    assert r['delta']==2. and r['valid_architecture_count']==1


def test_readonly_manual_checks_create_no_optimizer_or_backward():
    with no_optimization() as counts:
        order_score([1,2,3],[1,2,3])
    assert counts=={'optimizer_initializations':0,'backward_calls':0,'optimization_updates':0}


def test_saved_states_and_independent_matrix_recalculation():
    if not (OUT/'checks.json').exists():
        pytest.skip('existing real-state analysis has not yet run')
    checks=json.loads((OUT/'checks.json').read_text())
    assert checks['completed_states']==12 and checks['query_state_rows']==348
    assert checks['global_pair_comparisons']==2163168 and checks['local_pair_comparisons']==15660
    assert not checks['restore_failures']
    assert checks['same_reference_pool_all_states'] and checks['same_Dstar_Drandom_per_query']
    assert all(v==0 for v in checks['zero_optimization_guard'].values())
    assert all(v==0 for v in checks['new_training_counts'].values())
    assert checks['distance_freeze_at']<checks['first_label_access_at']
    labels=json.loads((OUT/'allowed_validation_means.json').read_text())
    pool_ids=None
    for folder in sorted((OUT/'states').iterdir()):
        g=json.loads((folder/'geometry_before_labels.json').read_text())
        rows=json.loads((folder/'per_query.json').read_text())
        pool_ids=g['train_ids'] if pool_ids is None else pool_ids
        assert pool_ids==g['train_ids']
        e,qe=np.array(g['train_embeddings']),np.array(g['query_embeddings'])
        d=np.array(g['euclidean_distances'])
        assert d.shape==(29,112)
        direct=np.sqrt(((qe[:,None,:]-e[None,:,:])**2).sum(axis=-1))
        np.testing.assert_allclose(d,direct,rtol=0,atol=1e-12)
        yy=np.array([labels['train'][c] for c in pool_ids])
        for qi,r in enumerate(rows):
            gaps=np.abs(yy-labels['query'][r['candidate_id']])
            # Independent scalar reference, not the vectorized scoring function.
            same=opposite=ties=perf_ties=0
            tolerance=1e-10*d[qi].max()
            for i,j in itertools.combinations(range(112),2):
                if abs(gaps[i]-gaps[j])<=1e-12:
                    perf_ties+=1
                elif abs(d[qi,i]-d[qi,j])<=tolerance:
                    ties+=1
                elif (gaps[i]<gaps[j])==(d[qi,i]<d[qi,j]):
                    same+=1
                else:
                    opposite+=1
            assert [same,opposite,ties,perf_ties]==[r['global_concordant'],r['global_discordant'],r['global_distance_ties'],r['global_performance_ties']]
            assert r['global_order']==(same+.5*ties)/(same+opposite+ties)
            assert r['De_pp']>=r['Dstar_pp']-1e-10
