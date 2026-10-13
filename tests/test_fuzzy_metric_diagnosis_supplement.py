"""Read-only CPU artifact checks. No synthetic models, optimizers, or training."""
import ast
import csv
import json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/posthoc/fuzzy-metric-stage2-readonly-supplement-20260910-v1'
read=lambda p:json.loads(p.read_text())


@pytest.fixture(scope='module')
def evidence():
    assert (OUT/'checks.json').exists(),'Run the readonly supplement first; tests never fit missing models.'
    return read(OUT/'checks.json'),read(OUT/'metrics_24_combinations.json'),read(OUT/'per_candidate.json')


def test_exact_24_combinations_same29_not87(evidence):
    checks,metrics,rows=evidence
    assert checks['completed_combinations']==24 and checks['missing']==[]
    assert {(r['method'],r['seed'],r['mapping_steps'],r['gp_steps']) for r in metrics}=={
        (method,seed,m,g) for method in ['regression','N2'] for seed in range(3) for m in [100,300] for g in [20,100]}
    boundary=read(OUT/'boundary.json');assert len(boundary['train_ids'])==112
    assert len({r['candidate_id'] for r in rows})==29 and len(rows)==696
    assert {r['candidate_id'] for r in rows}==set(boundary['query_ids'])


def test_no_optimizer_or_backward_was_called(evidence):
    checks,_,_=evidence
    assert checks['zero_update_guards']=={'optimizer_initializations':0,'backward_calls':0,'optimization_updates':0}
    assert checks['new_optimizer_updates']==0
    assert set(checks['new_training_counts'].values())=={0}
    tree=ast.parse((ROOT/'analyse/fuzzy_metric_diagnosis_supplement.py').read_text())
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls&{'step','backward','fit','minimize'}
    for check in checks['checks']:assert check['model_state_unchanged'] and check['GP_state_unchanged']


def test_corresponding_snapshots_and_exact_old_prediction_reproduction(evidence):
    checks,_,_=evidence
    assert checks['max_old_metric_difference']==0
    assert all(c['max_reproduction_abs_error']==0 for c in checks['checks'])
    for folder in (OUT/'combinations').iterdir():
        b=read(folder/'predictions_before_evaluation.json');m,g=b['mapping_steps'],b['gp_steps']
        assert b['source_paths']['mapping'].endswith(f'/M{m}.pt')
        assert b['source_paths']['gp'].endswith(f'-M{m}-gp/G{g}.pt')
        assert b['checks']['fixed_noise_restored_exact'] and b['checks']['train_inputs_exact'] and b['checks']['train_targets_exact']
        assert b['standardization']['fit_ids']==b['train_ids']


def test_constant_uses_training_candidate_mean_and_error_identical(evidence):
    _,metrics,rows=evidence
    train=read(OUT/'training_labels.json');query=read(OUT/'query_validation_means.json')
    mean=float(np.mean([train[c]['validation_mean'] for c in sorted(train)]))
    mae=float(np.mean([abs(y-mean) for y in query.values()]))
    assert len(train)==112 and len(query)==29
    assert all(abs(r['constant_mean']-mean)<1e-15 for r in rows)
    assert all(abs(r['constant_mae']-mae)<1e-15 for r in metrics)
    assert len({r['constant_mae'] for r in metrics})==1


def test_metrics_independently_recomputed_from_exported_records(evidence):
    _,metrics,rows=evidence
    train=read(OUT/'training_labels.json')
    for metric in metrics:
        subset=[r for r in rows if all(r[k]==metric[k] for k in ['method','seed','mapping_steps','gp_steps'])]
        gaps=[[abs(r['true_validation_mean']-train[c]['validation_mean']) for c in r['neighbor_ids']] for r in subset]
        assert metric['main_neighborhood_metric']==pytest.approx(float(np.median(np.mean(gaps,axis=1))),abs=1e-15)
        for model in ['head','knn','gp','constant']:
            error=np.mean([abs(r['true_validation_mean']-r[f'{model}_mean']) for r in subset])
            assert metric[f'{model}_mae']==pytest.approx(float(error),abs=1e-15)
        for t in [.006,.012]:assert metric[f'wrong_{t}']==pytest.approx(float((np.array(gaps)>t).mean()),abs=1e-15)


def test_GP_step_preserves_representation_outputs_but_not_GP_predictions(evidence):
    _,_,rows=evidence
    for method in ['regression','N2']:
        for seed in range(3):
            for m in [100,300]:
                groups={g:[r for r in rows if (r['method'],r['seed'],r['mapping_steps'],r['gp_steps'])==(method,seed,m,g)] for g in [20,100]}
                for x,y in zip(groups[20],groups[100]):
                    assert all(x[k]==y[k] for k in ['candidate_id','head_mean','knn_mean','neighbor_ids','neighbor_distances'])
                assert max(abs(x['gp_mean']-y['gp_mean']) for x,y in zip(groups[20],groups[100]))>1e-6


def test_signal_correlation_noise_and_units(evidence):
    _,_,rows=evidence
    for r in rows:
        assert 0<=r['kernel_max_correlation']<=1
        assert len(r['kernel_top5_ids'])==len(set(r['kernel_top5_ids']))==5
        assert r['kernel_max_correlation']==max(r['kernel_top5_correlations'])
        assert r['kernel_top5_mean_correlation']==pytest.approx(np.mean(r['kernel_top5_correlations']),abs=1e-15)
        assert r['nearest_GP_distance_over_lengthscale']>=0
        assert r['observation_variance']==r['latent_variance']+r['pooled_training_SE_squared']+r['additional_noise_accuracy_squared']
        for level,z in [('90',1.6448536269514722),('95',1.959963984540054)]:
            lo,hi=r[f'interval{level}']
            assert (hi-lo)==pytest.approx(2*z*np.sqrt(r['observation_variance']),abs=1e-15)


def test_label_timing_scope_and_complete_paired_differences(evidence):
    checks,_,_=evidence
    assert checks['prediction_before_evaluation'] and checks['no_query_SE_in_prediction']
    assert checks['B_and_outer_and_test_labels_read']==0
    access=read(OUT/'evaluation_access.json');boundary=read(OUT/'boundary.json')
    assert access['ids']==boundary['query_ids']
    for value in access['predictions'].values():assert value['time']<=access['time']
    pairs=read(OUT/'paired_differences.json');assert len(pairs)==12*11
    for r in pairs:
        values=[r[f'seed{s}'] for s in range(3)]
        assert r['mean']==np.mean(values) and r['min']==min(values) and r['max']==max(values)
    with (OUT/'per_candidate.csv').open() as f:assert len(list(csv.DictReader(f)))==696
