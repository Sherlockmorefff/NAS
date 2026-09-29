"""Bounded synthetic-only model preflight; parent owns all actual execution."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from residual_core import (Budget, BudgetExceeded, BlockSession, CAP, ResidualBranch, RawBranch,
                           check_forks, cpu_setup, exact, project_mapping, project_weight,
                           restore, rng_state, snapshot, spectral_stats)
from residual_diagnostics import (diagnose_isolated, query_projection, window, stability_at,
                                  geometry_from_embeddings, geometry_comparison)
from stage01_core import independent_nll


def synthetic(n=20, seed=202):
    rng = np.random.default_rng(seed)
    x_raw = torch.from_numpy(rng.normal(size=(n, 16)))
    y_raw = .4 * x_raw[:, 0] - .2 * x_raw[:, 4] + torch.sin(x_raw[:, 7])
    x_center, x_scale = x_raw.mean(0), x_raw.std(0, unbiased=False).clamp_min(1e-6)
    y_center, y_scale = y_raw.mean(), y_raw.std(unbiased=False).clamp_min(1e-8)
    se = torch.linspace(.01, .03, n)
    return (x_raw - x_center) / x_scale, (y_raw - y_center) / y_scale, se.square() / y_scale.square(), {
        'x_center': x_center, 'x_scale': x_scale, 'y_center': y_center, 'y_scale': y_scale,
        'measurement_se': se, 'known_variance_formula': 'SE squared / y_scale squared, no extra division by3'}


def run_steps(model, mode, steps, budget, stop_at=None):
    session = BlockSession(model, mode, steps, 'synthetic_' + mode)
    logs = []
    for _ in range(steps if stop_at is None else stop_at):
        logs.append(copy.deepcopy(session.step(budget)))
    return session, logs


def diagnostic(session, query, step=None):
    return diagnose_isolated(session.model, query, ['q%02d' % i for i in range(len(query))], .7, .12,
        session.completed if step is None else step, mode=session.mode, with_geometry=True,
        training_ids=['t%02d' % i for i in range(len(session.model.y))],
        last_gradient=None if session.last_update is None else session.last_update['gradient_before_clip'],
        lrs=session.lrs)


def test_initialization_projection_distance(budget, directory):
    x, y, known, scaler = synthetic()
    model = ResidualBranch(x, y, known, 5)
    h, _ = model.encode(x)
    assert torch.equal(x, h)
    assert sum(p.numel() for p in model.mapping_params()) == 264
    assert sum(p.numel() for p in model.mapper.head.parameters()) == 17
    assert torch.count_nonzero(model.mapper.W1.weight) > 0
    assert torch.count_nonzero(model.mapper.W1.bias) == 0 and torch.count_nonzero(model.mapper.W2.weight) == 0
    initial = spectral_stats(model)
    loss, _ = model.objective('regression')
    loss.backward()
    assert torch.count_nonzero(model.mapper.W1.weight.grad) == 0 and torch.count_nonzero(model.mapper.W1.bias.grad) == 0
    assert torch.count_nonzero(model.mapper.W2.weight.grad) > 0
    # Distinct singular values prove clipping, rather than a uniform rescale.
    matrix = torch.zeros((16, 8), dtype=torch.float64)
    singular = torch.tensor([2*CAP, .8*CAP, .5*CAP, .3*CAP, .2*CAP, .1*CAP, .05*CAP, .01*CAP])
    matrix[:8] = torch.diag(singular)
    projected = project_weight(matrix)
    assert projected['projected'] and torch.allclose(torch.linalg.svdvals(matrix), singular.clamp(max=CAP), atol=1e-12, rtol=1e-12)
    before = matrix.clone()
    project_weight(matrix)
    assert torch.equal(before, matrix), 'Within-cap matrices must not be reconstructed'
    with torch.no_grad():
        generator = torch.Generator().manual_seed(77)
        model.mapper.W1.weight.copy_(torch.randn(model.mapper.W1.weight.shape, generator=generator) * 3)
        model.mapper.W2.weight.copy_(torch.randn(model.mapper.W2.weight.shape, generator=generator) * 3)
        model.mapper.W1.bias.fill_(1.3)
    project_mapping(model)
    bounded = spectral_stats(model)
    with torch.no_grad():
        h, _ = model.encode(x)
        dx, dh = torch.pdist(x), torch.pdist(h)
        assert torch.all(dx > 0)
        assert bool((dh >= .5*dx-1e-10).all()) and bool((dh <= 1.5*dx+1e-10).all())
    assert torch.equal(known, scaler['measurement_se'].square()/scaler['y_scale'].square())
    return {'pass': True, 'mapping_parameters': 264, 'head_parameters': 17,
            'initial_h_equals_x_exact': True, 'initial_w1_gradient_zero_expected': True,
            'svd_is_singular_value_clipping_not_global_scaling': True,
            'within_cap_projection_bitwise_noop': True, 'actual_initial_spectral': initial,
            'actual_projected_spectral': bounded,
            'sampled_distance_ratio_min': float((dh/dx).min()), 'sampled_distance_ratio_max': float((dh/dx).max()),
            'known_noise_uses_SE_without_second_division_by3': True}


def test_objective_forks_and_constraint_modes(budget, directory):
    x, y, known, _ = synthetic()
    model = ResidualBranch(x, y, known, 4)
    direct = (model.encode(x)[1]-y).square().mean()
    objective, _ = model.objective('regression')
    assert torch.equal(direct, objective)
    common_reg, logs = run_steps(model, 'regression', 6, budget)
    assert logs[0]['loss_before_update'] == float(direct.detach())
    run_steps(model, 'gp', 3, budget)
    fork = check_forks(model)
    branches = fork['branches']
    assert branches['R_DKL_C'].fingerprint() == branches['R_DKL_U'].fingerprint()
    assert branches['R_DKL_C'].configuration_hash() != branches['R_DKL_U'].configuration_hash()
    query = synthetic(8, 404)[0]
    predictions = [branch.predict(query) for branch in branches.values()]
    assert all(exact(predictions[0], p) for p in predictions[1:])
    # Toggle every nested module in one branch; no other mutable module changes.
    others = [branches[n] for n in ['R_REG', 'R_DKL_C', 'R_DKL_U']]
    mode_before = [[m.training for root in [m.mapper, m.gp] for m in root.modules()] for m in others]
    branches['R_FROZEN'].mapper.train(); branches['R_FROZEN'].gp.train()
    mode_after = [[m.training for root in [m.mapper, m.gp] for m in root.modules()] for m in others]
    assert mode_before == mode_after
    c = branches['R_DKL_C']; head = copy.deepcopy(c.mapper.head.state_dict())
    untouched = branches['R_REG'].fingerprint()
    c_session, c_logs = run_steps(c, 'joint', 2, budget)
    assert c_logs[0]['mapping_gradient_before_clip'] > 0 and c_logs[0]['gp_gradient_before_clip'] > 0
    assert exact(head, c.mapper.head.state_dict()) and branches['R_REG'].fingerprint() == untouched
    spectral_stats(c)
    u = branches['R_DKL_U']
    with torch.no_grad():u.mapper.W2.weight.mul_(1000)
    assert spectral_stats(u)['w2_spectral_norm'] > CAP
    _, u_logs = run_steps(u, 'joint', 2, budget)
    assert all(not log['projection']['projection_evaluated'] for log in u_logs)
    assert spectral_stats(u)['w2_spectral_norm'] > CAP
    state = snapshot(u); restored = restore(state)
    assert restored.constraint_mode == 'unbounded' and restored.fingerprint() == u.fingerprint()
    frozen = branches['R_FROZEN']; fixed = copy.deepcopy(frozen.mapper.state_dict())
    _, frozen_logs = run_steps(frozen, 'gp', 2, budget)
    assert exact(fixed, frozen.mapper.state_dict()) and all(not r['projection']['projection_evaluated'] for r in frozen_logs)
    return {'pass': True, 'pure_mse_actual_update_verified': True, 'four_forks': fork['checks'],
            'four_initial_predictions_exact': True, 'module_mode_isolation': True,
            'C_mapping_gradient': c_logs[0]['mapping_gradient_before_clip'], 'C_gp_gradient': c_logs[0]['gp_gradient_before_clip'],
            'C_head_frozen': True, 'C_post_projection_legal': True, 'U_kept_excess_singular_value': True,
            'U_restore_no_projection': True, 'frozen_mapping_and_no_reprojection': True}


def test_independent_nll_variance_and_raw50(budget, directory):
    checks = []
    for kind, n in [('RESIDUAL16', 20), ('RAW16', 50)]:
        x, y, known, _ = synthetic(n)
        model = ResidualBranch(x, y, known, 7) if kind == 'RESIDUAL16' else RawBranch(x, y, known)
        if kind == 'RESIDUAL16':run_steps(model, 'regression', 4, budget)
        run_steps(model, 'gp', 3, budget)
        # Independent NumPy dense Matern covariance + Cholesky, normalized once.
        with torch.no_grad():result = independent_nll(model)
        assert result['pass'], result
        query = synthetic(7, 771)[0]
        mean, variance = model.predict(query)
        with torch.no_grad():
            train_h = model.encode(model.x)[0].numpy(); query_h = model.encode(query)[0].numpy()
        ell = float(model.gp.covar_module.base_kernel.lengthscale.detach())
        scale = float(model.gp.covar_module.outputscale.detach())
        extra = float(model.gp.likelihood.second_noise.detach())
        constant = float(model.gp.mean_module.constant.detach())
        def kernel(a,b):
            d=np.linalg.norm(a[:,None]-b[None,:],axis=-1)/ell
            return scale*(1+np.sqrt(5)*d+5*d*d/3)*np.exp(-np.sqrt(5)*d)
        covariance = kernel(train_h,train_h)+np.diag(known.numpy()+extra)
        cross = kernel(query_h,train_h)
        expected_mean = constant+cross@np.linalg.solve(covariance,y.numpy()-constant)
        expected_variance = scale-np.sum(cross*np.linalg.solve(covariance,cross.T).T,axis=1)
        assert np.allclose(mean.numpy(),expected_mean,atol=1e-8,rtol=1e-6)
        assert np.allclose(variance.numpy(),expected_variance,atol=1e-8,rtol=1e-6)
        assert np.allclose(.12**2*variance.numpy(), .12**2*expected_variance)
        assert model.gp.train_inputs[0].shape == (n,16)
        if kind == 'RAW16':assert not hasattr(model,'mapper')
        checks.append({'model_type':kind,'n':n,'gp_dimension':16,'independent_nll':result,
                       'posterior_mean_max_abs_error':float(np.max(np.abs(mean.numpy()-expected_mean))),
                       'posterior_latent_variance_max_abs_error':float(np.max(np.abs(variance.numpy()-expected_variance))),
                       'raw_accuracy_variance_factor':.12**2})
    return {'pass':True,'models':checks}


def test_block_resume_across_lr_and_accounting(budget, directory):
    x,y,known,_=synthetic(8)
    results=[]
    for kind, mode, model in [('bounded_joint','joint',ResidualBranch(x,y,known,3)),
                              ('unbounded_joint','joint',ResidualBranch(x,y,known,3,'unbounded')),
                              ('bounded_regression','regression',ResidualBranch(x,y,known,3)),
                              ('raw_gp','gp',RawBranch(x,y,known))]:
        seed=snapshot(model)
        full,logs=run_steps(restore(copy.deepcopy(seed)),mode,10,budget)
        uninterrupted=full.snapshot()
        partial,logs_a=run_steps(restore(copy.deepcopy(seed)),mode,10,budget,stop_at=7)
        saved=partial.snapshot();path=directory/(kind+'_after7.pt');torch.save(saved,path)
        torch.rand(4);np.random.random(4);random.random()
        continued=BlockSession.from_snapshot(torch.load(path,map_location='cpu',weights_only=False))
        logs_b=[copy.deepcopy(continued.step(budget)) for _ in range(3)]
        final=continued.snapshot()
        assert full.model.fingerprint()==continued.model.fingerprint()
        for field in ['optimizer','scheduler','rng','ordered_parameter_names','configuration']:
            assert exact(uninterrupted[field],final[field]),field
        assert [row['learning_rates'] for row in logs]==[row['learning_rates'] for row in logs_a+logs_b]
        assert all(row['learning_rates']==logs[0]['learning_rates'] for row in logs[:8])
        assert all(all(math.isclose(v,.1*logs[0]['learning_rates'][k]) for k,v in row['learning_rates'].items()) for row in logs[8:])
        assert final['accounting']['completed_updates']==10
        assert final['accounting']['block_training_seconds'] >= saved['accounting']['block_training_seconds']
        assert final['accounting']['budget_updates_after_last_step']==budget.updates
        results.append({'kind':kind,'continuous_updates':10,'split_updates':[7,3],
            'parameters_adam_rng_exact':True,'lr_8plus2_exact':True,'cost_and_update_accounting_preserved':True})
    return {'pass':True,'cases':results}


def _mode_cache_signature(model):
    roots=([model.mapper] if hasattr(model,'mapper') else [])+[model.gp]
    return ([m.training for root in roots for m in root.modules()],id(model.gp.prediction_strategy),
            tuple(id(v) for v in model.gp.train_inputs),model.cache_version)


def test_diagnostic_invariance_query_poison_and_stability(budget,directory):
    x,y,known,_=synthetic(12)
    query=synthetic(7,681)[0]
    seed=snapshot(ResidualBranch(x,y,known,8))
    off=BlockSession(restore(copy.deepcopy(seed)),'joint',10,'diag_compare')
    for _ in range(4):off.step(budget)
    state_off=off.snapshot()
    on=BlockSession(restore(copy.deepcopy(seed)),'joint',10,'diag_compare')
    for _ in range(4):
        on.step(budget)
        before=_mode_cache_signature(on.model);rng=rng_state();opt=copy.deepcopy(on.optimizer.state_dict())
        diagnostic(on,query)
        assert before==_mode_cache_signature(on.model) and exact(rng,rng_state()) and exact(opt,on.optimizer.state_dict())
    state_on=on.snapshot()
    for field in ['mapper','gp','optimizer','scheduler','rng']:
        assert exact(state_off[field],state_on[field]),field
    # Warm the primary cache, then ensure even its eval/mode/cache state survives
    # diagnostics; no optimizer step is taken on this deliberately eval model.
    on.model.predict(query);signature=_mode_cache_signature(on.model);rng=rng_state()
    diagnostic(on,query)
    assert signature==_mode_cache_signature(on.model) and exact(rng,rng_state())
    rows=[{'candidate_id':'q%02d'%i,'x_standardized':row.tolist(),'y':.5,'SE':.01} for i,row in enumerate(query)]
    poisoned=copy.deepcopy(rows)
    for row in poisoned:row.update(y=object(),SE=object(),rank=object(),champion=object())
    public=[query_projection(rows),query_projection(poisoned)]
    assert exact(public[0],public[1])
    outcomes=[]
    for projected in public:
        session=BlockSession(restore(copy.deepcopy(seed)),'joint',10,'poison')
        session.step(budget)
        record=diagnose_isolated(session.model,torch.tensor(projected['query_x_standardized']),projected['query_ids'],.7,.12,
            1000,mode='joint',with_geometry=True,training_ids=['t%02d'%i for i in range(len(x))])
        repeated=[]
        for step in [700,800,900,1000]:
            copyrow=copy.deepcopy(record);copyrow['block_step']=step;repeated.append(copyrow)
        stable=stability_at(repeated,1000)
        assert stable['practical_stability'] is True
        outcomes.append((session.model.fingerprint(),record,stable))
    assert exact(outcomes[0],outcomes[1])
    regression=BlockSession(restore(copy.deepcopy(seed)),'regression',10,'regression_diagnostic')
    row=diagnostic(regression,query)
    assert row['nll_after_update'] is None and 'query_mean_raw' not in row
    assert row['objective_after_update']==row['mse_after_update']
    return {'pass':True,'diagnostic_on_off_parameters_adam_rng_exact':True,'nested_module_mode_cache_inputs_exact':True,
            'poison_query_y_SE_ignored':True,'prediction_geometry_top5_stability_exact':True,
            'block1000_stability_uses700_800_900_1000':True,'intermediate_regression_has_only_pure_mse_no_GP_prediction':True}


def test_geometry_constant_and_scale_invariance(budget,directory):
    train_ids=['t5','t1','t3','t0','t2','t4'];query_ids=['q0','q1']
    constant=geometry_from_embeddings(np.zeros((6,3)),np.zeros((2,3)),train_ids,query_ids)
    assert constant['geometry']['s_h'] is None and constant['geometry']['d_TT_over_s_h'] is None
    assert constant['nearest_train_ids']==['t0','t0']
    assert constant['five_nearest_train_ids']==[sorted(train_ids)[:5]]*2
    train=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.],[1.,2.,3.],[-1.,-2.,0.]])
    query=np.array([[.2,.1,.4],[1.2,1.7,2.5]])
    params={'lengthscale':1.7,'outputscale':2.3,'extra_noise_variance_standardized':.01,'mean':0.}
    first=geometry_from_embeddings(train,query,train_ids,query_ids,params)
    changed_params={**params,'lengthscale':params['lengthscale']*3.5}
    second=geometry_from_embeddings(train*3.5+8,query*3.5+8,train_ids,query_ids,changed_params)
    difference=geometry_comparison(first,second)
    assert max(difference['shape_change'].values())<1e-12
    assert difference['nearest_changed_fraction']==0
    assert max(difference['R_relative_change'].values())<1e-12
    assert geometry_comparison(constant,constant)['shape_change']=={'TT':None,'QT':None}
    return {'pass':True,'constant_scale_is_null_without_epsilon':True,'constant_neighbors_use_ID_tie_break':True,
            'joint_h_ell_scaling_plus_translation_preserves_neighbors_R_shape':True,'differences':difference}


def run_checks(output_dir,budget):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    cpu_setup(0)
    start_seconds,start_updates=budget.seconds,budget.updates
    checks={}
    for function in [test_initialization_projection_distance,test_objective_forks_and_constraint_modes,
                     test_independent_nll_variance_and_raw50,test_block_resume_across_lr_and_accounting,
                     test_diagnostic_invariance_query_poison_and_stability,test_geometry_constant_and_scale_invariance]:
        tick,prior=time.monotonic(),budget.seconds
        try:checks[function.__name__]=function(budget,directory)
        except BudgetExceeded:raise
        except Exception as error:checks[function.__name__]={'pass':False,'error':repr(error),'traceback':traceback.format_exc()}
        finally:
            # Include all untimed synthetic model computations, recovery and
            # checks without charging BlockSession updates a second time.
            budget.charge(max(0.,time.monotonic()-tick-(budget.seconds-prior)),kind='training')
            budget.persist()
    result={'pass':all(v['pass'] for v in checks.values()),'checks':checks,
            'updates':budget.updates-start_updates,'charged_seconds':budget.seconds-start_seconds,
            'only_synthetic_data':True,'real_A_scores_or_diagnostics_opened':False}
    (directory/'core_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def main(root):
    root=Path(root);directory=root/'verification';directory.mkdir(exist_ok=True)
    budget=Budget(directory/'synthetic_ledger.json',limit=1200)
    attempt=1+len(list(directory.glob('model_preflight_attempt_*.json')))
    result=run_checks(directory/('model_attempt_%d'%attempt),budget)
    summary={'pass':result['pass'],'attempt':attempt,'verification_seconds':budget.seconds,
             'synthetic_optimizer_updates':budget.updates,'all_attempts_included_in_budget':True,
             'model_checks':result,'statistics_tests':'root runs and combines separately',
             'formal_model_updates':0,'real_A_scores_or_diagnostics_opened':False}
    text=json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    (directory/('model_preflight_attempt_%d.json'%attempt)).write_text(text)
    (directory/'model_preflight_summary.json').write_text(text)
    print(json.dumps({k:summary[k] for k in ['pass','attempt','verification_seconds','synthetic_optimizer_updates']},ensure_ascii=False))
    if not result['pass']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    main(args.root)
