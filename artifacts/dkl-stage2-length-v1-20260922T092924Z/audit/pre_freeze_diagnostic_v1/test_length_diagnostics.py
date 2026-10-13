"""Bounded synthetic checks for diagnostic isolation and stability units.

No data-source access. Root invokes run_checks once with the shared verification
budget. The fixture has actual nonzero Adam history and clearly labels its
synthetic source scheduler; it is not a formal scientific trajectory.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from length_diagnostics import diagnose_isolated, query_projection, stability_at, window


def require(value, message):
    if not bool(value):
        raise AssertionError(message)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def rng_state():
    return {'python': random.getstate(), 'numpy': copy.deepcopy(np.random.get_state()), 'torch': torch.get_rng_state().clone()}


def surface(session):
    model = session.model
    return {'fingerprint': model.fingerprint(), 'optimizer': copy.deepcopy(session.optimizer.state_dict()),
            'module_modes': [(name, module.training) for name, module in model.mapper.named_modules()] +
                            [('gp.' + name, module.training) for name, module in model.gp.named_modules()],
            'requires_grad': [(name, value.requires_grad) for name, value in model.mapper.named_parameters()] +
                             [('gp.' + name, value.requires_grad) for name, value in model.gp.named_parameters()],
            'train_input_pointers': [value.data_ptr() for value in model.gp.train_inputs],
            'train_input_values': [value.detach().clone() for value in model.gp.train_inputs],
            'train_target_pointer': model.gp.train_targets.data_ptr(),
            'train_target_values': model.gp.train_targets.detach().clone(),
            'prediction_strategy_id': id(model.gp.prediction_strategy), 'cache_version': model.cache_version,
            'rng': rng_state()}


def synthetic_record(step=1000):
    return {'total_joint_step': step, 'nll_after_update': -2.0,
            'query_ids': ['q' + str(i) for i in range(6)],
            'training_kernel': [[2., 0.], [0., 2.]],
            'combined_kernel': [[2. if i == j else 0. for j in range(8)] for i in range(8)],
            'train_mean_raw': [.7, .8], 'query_mean_raw': [.8, .7, .6, .5, .4, .3],
            'query_latent_sd_raw': [.01] * 6, 'query_model_sd_raw': [.02] * 6,
            'top5_ids': ['q0', 'q1', 'q2', 'q3', 'q4'], 'top1_id': 'q0'}


def formula_checks():
    base, current = synthetic_record(), synthetic_record(1100)
    current['nll_after_update'] = -2.001
    current['training_kernel'] = [[v * 1.005 for v in row] for row in base['training_kernel']]
    current['combined_kernel'] = [[v * 1.005 for v in row] for row in base['combined_kernel']]
    current['query_mean_raw'] = [x + 0.0009 for x in base['query_mean_raw']]
    current['train_mean_raw'] = [x - 0.0009 for x in base['train_mean_raw']]
    current['query_latent_sd_raw'] = [x + 0.0009 for x in base['query_latent_sd_raw']]
    current['query_model_sd_raw'] = [x + 0.0009 for x in base['query_model_sd_raw']]
    value = window(base, current)
    require(math.isclose(value['nll_relative_change'], 0.0005, abs_tol=1e-14), 'NLL denominator must use max(1,abs(previous))')
    require(math.isclose(value['training_kernel_relative_change'], .005, abs_tol=1e-14), 'Frobenius relative change formula')
    require(value['practical_stability_window'], '0.0009 accuracy change should pass 0.1 percentage point threshold')
    wrong_units = copy.deepcopy(current)
    wrong_units['query_mean_raw'][0] = base['query_mean_raw'][0] + .0011
    require(not window(base, wrong_units)['criteria']['mean_prediction_stable'], '0.0011 raw accuracy must fail mean stability')
    wrong_sd = copy.deepcopy(current)
    wrong_sd['query_model_sd_raw'][0] = base['query_model_sd_raw'][0] + .0011
    require(not window(base, wrong_sd)['criteria']['uncertainty_stable'], 'Both latent and model SD are required')
    wrong_kernel = copy.deepcopy(current)
    wrong_kernel['combined_kernel'] = [[v * 1.011 for v in row] for row in base['combined_kernel']]
    require(not window(base, wrong_kernel)['criteria']['similarity_stable'], 'Both training and combined kernels are required')
    zero = synthetic_record()
    zero['training_kernel'] = [[0., 0.], [0., 0.]]
    zero_b = copy.deepcopy(zero)
    zero_b['total_joint_step'] = 1100
    require(window(zero, zero_b)['training_kernel_relative_change'] == 0., 'Kernel denominator floor at zero')
    records = [synthetic_record(step) for step in (1000, 1100, 1200, 1300)]
    stable = stability_at(records, 1300)
    require(stable['practical_stability'] and stable['windows_available'] == 3, 'Exactly three consecutive windows required')
    require(stability_at(records, 1000)['practical_stability'] is None, '1000 historical stability unavailable')
    require(stability_at(records[:3], 1200)['practical_stability'] is None, 'Two windows insufficient')
    contaminated = copy.deepcopy(records)
    contaminated[0]['query_mean_raw'][0] += .01
    require(not stability_at(contaminated, 1300)['practical_stability'], 'Earliest of final three windows cannot be ignored')
    reordered = copy.deepcopy(records)
    reordered[-1]['top5_ids'] = ['q1', 'q0', 'q2', 'q3', 'q4']
    reordered[-1]['top1_id'] = 'q1'
    result = stability_at(reordered, 1300)
    require(result['top5_selection_stable'] and not result['top5_order_stable'] and not result['top1_selection_stable'],
            'Top5 set, order and top1 must be reported separately')
    changed = copy.deepcopy(records)
    changed[-1]['top5_ids'][-1] = 'q5'
    require(not stability_at(changed, 1300)['top5_selection_stable'], 'Changed fifth candidate must be visible')
    recent = [synthetic_record(step) for step in (1600, 1700, 1800, 1900, 2000)]
    recent[0]['nll_after_update'] = 100.
    require(stability_at(recent, 2000)['practical_stability'], '2000 must use only 1700→1800→1900→2000')
    for record in records:
        record['query_y'] = 'POISON_LABEL'
        record['query_se'] = object()
    require(stability_at(records, 1300) == stable, 'Stability functions must ignore query label/SE poison')
    return {'pass': True, 'criteria_and_units': True, 'three_consecutive_windows': True,
            'baseline_history_unavailable': True, 'top5_set_vs_order_vs_top1': True,
            'poison_stability_invariance': True}


def _run_checks(output_dir, budget):
    from length_core import TailSession
    from test_length_core import make_synthetic_source

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before_seconds, before_updates = budget.seconds, budget.updates
    start = time.monotonic()
    results = {}
    source = make_synthetic_source(budget)
    query_x = torch.linspace(-1.0, 1.0, 6 * 16, dtype=torch.float64).reshape(6, 16)
    query_ids = ['q' + str(i) for i in range(6)]
    center, scale = 0.70, 0.05

    def diagnose(session, x=query_x, ids=query_ids):
        budget.check()
        tick = time.monotonic()
        try:
            return diagnose_isolated(session.model, x, ids, center, scale, session.total_joint_step,
                                     last_gradient=session.last_gradient, lrs=session.lrs)
        finally:
            budget.charge(time.monotonic() - tick, kind='prediction')

    # Warm the real session cache deliberately; diagnostics must preserve its
    # eval flags, cache object identity, train-input pointers and RNG exactly.
    session = TailSession.from_source(copy.deepcopy(source))
    tick = time.monotonic()
    try:
        session.model.predict(query_x)
    finally:
        budget.charge(time.monotonic() - tick, kind='prediction')
    prior = surface(session)
    baseline = diagnose(session)
    require(equal(prior, surface(session)), 'Diagnostic changed live model modes/cache/inputs/optimizer/RNG')
    require(baseline['gradient_before_last_update'] is None, 'Source baseline gradient must be unavailable')
    require(baseline['state_hash'] == session.model.fingerprint(), 'Updated diagnostic state hash differs')
    results['main_surface_exactly_unchanged'] = True

    # One clean diagnostic path, one no-diagnostic path, and the same public
    # query features accompanied by unreadable y/SE poison before projection.
    rows = [{'candidate_id': cid, 'x_standardized': query_x[i].tolist(), 'y': 0.5, 'SE': 0.01}
            for i, cid in enumerate(query_ids)]
    clean_projection = query_projection(rows)
    for row in rows:
        row['y'], row['SE'] = object(), object()
    poisoned_projection = query_projection(rows)
    require(clean_projection == poisoned_projection, 'Query poison changed public projection')
    paths = {}
    for name in ('off', 'on', 'poison'):
        current = TailSession.from_source(copy.deepcopy(source))
        diagnostics = []
        for _ in range(4):
            current.step(budget)
            if name != 'off':
                projected = poisoned_projection if name == 'poison' else clean_projection
                prior = surface(current)
                diagnostics.append(diagnose(current, torch.tensor(projected['query_x_standardized'], dtype=torch.float64), projected['query_ids']))
                require(equal(prior, surface(current)), 'Per-step diagnostic changed live surface')
        paths[name] = {'fingerprint': current.model.fingerprint(), 'optimizer': copy.deepcopy(current.optimizer.state_dict()),
                       'rng': rng_state(), 'diagnostics': diagnostics, 'scaler': {'center': center, 'scale': scale}}
    for name in ('on', 'poison'):
        for key in ('fingerprint', 'optimizer', 'rng', 'scaler'):
            require(equal(paths['off'][key], paths[name][key]), 'Diagnostic switch/poison changed training path or scaler: ' + key)
    require(paths['on']['diagnostics'] == paths['poison']['diagnostics'], 'Poison changed diagnostic predictions/selected IDs')
    results['diagnostics_on_off_exact_training_path'] = True
    results['query_poison_training_scaler_predictions_selection_invariant'] = True

    qvar = np.asarray(baseline['query_latent_variance_standardized'])
    qsd = np.asarray(baseline['query_latent_sd_raw'])
    msd = np.asarray(baseline['query_model_sd_raw'])
    extra = baseline['extra_noise_variance_standardized']
    require(np.allclose(qsd, scale * np.sqrt(np.maximum(qvar, 0)), rtol=0, atol=1e-15), 'Latent SD variance units wrong')
    require(np.allclose(msd, scale * np.sqrt(np.maximum(qvar, 0) + extra), rtol=0, atol=1e-15), 'Model SD must add extra variance once, without query SE')
    require(np.all(msd >= qsd), 'Model SD cannot be below latent SD')
    rank = sorted(range(len(query_ids)), key=lambda i: (-baseline['query_mean_raw'][i], query_ids[i]))
    require(baseline['top5_ids'] == [query_ids[i] for i in rank[:5]] and baseline['top1_id'] == query_ids[rank[0]], 'Prediction rank/tie-break mismatch')
    results['variance_units_and_selection'] = True
    results['stability_formula_checks'] = formula_checks()

    elapsed = time.monotonic() - start
    uncharged = elapsed - (budget.seconds - before_seconds)
    if uncharged > 0:
        budget.charge(uncharged, kind='training')
    result = {'pass': True, 'checks': results, 'charged_seconds': budget.seconds - before_seconds,
              'synthetic_optimizer_updates': budget.updates - before_updates,
              'updates': budget.updates - before_updates,
              'real_label_rows_read': 0, 'query_labels_passed_to_model': False,
              'diagnostic_baseline': {k: baseline[k] for k in ('total_joint_step', 'state_hash', 'nll_after_update', 'top5_ids')},
              'fixture': '8 actual nonzero-Adam updates; mock scheduler 1000; no scientific trajectory',
              'accounting': 'per-step training and diagnostics charged; remaining validation wall time conservatively charged once'}
    (output_dir / 'diagnostics_checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def run_checks(output_dir, budget):
    """Return a preserved result and charge failed verification attempts too."""
    started = time.monotonic()
    seconds_before, updates_before = budget.seconds, budget.updates
    try:
        result = _run_checks(output_dir, budget)
    except Exception as error:
        result = {'pass': False, 'checks': {}, 'error': repr(error),
                  'traceback': traceback.format_exc(), 'real_label_rows_read': 0}
    residual = max(0., time.monotonic() - started - (budget.seconds - seconds_before))
    if residual:
        budget.charge(residual, kind='training')
    budget.persist()
    result.update(charged_seconds=budget.seconds - seconds_before,
                  updates=budget.updates - updates_before,
                  synthetic_optimizer_updates=budget.updates - updates_before)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'diagnostics_checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
