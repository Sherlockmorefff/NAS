"""Isolated diagnostics and label-free, preregistered stability calculations.

Importing this module requires only the Python standard library. In particular,
window/stability_at can be recomputed without importing PyTorch or training code.
The diagnostic function imports the frozen mathematical core lazily and works
only on a separate model, restoring Python/NumPy/Torch RNG in a finally clause.
"""
from __future__ import annotations

import copy
import math
import random


CRITERIA = ('objective_small', 'similarity_stable', 'mean_prediction_stable', 'uncertainty_stable')
TOLERANCES = {'nll_relative': 1e-3, 'kernel_relative': 0.01,
              'mean_max_abs_raw': 0.001, 'sd_max_abs_raw': 0.001}


def query_projection(rows):
    """Whitelist public query fields; y/SE values are deliberately never read."""
    ids = [row['candidate_id'] for row in rows]
    x = [[float(value) for value in row['x_standardized']] for row in rows]
    if len(ids) != len(set(ids)) or any(len(row) != 16 for row in x):
        raise ValueError('Query IDs must be unique and inputs must have 16 columns')
    if any(not math.isfinite(value) for row in x for value in row):
        raise ValueError('Query inputs must be finite')
    return {'query_ids': ids, 'query_x_standardized': x}


def _vector(value, name):
    result = [float(x) for x in value]
    if not result or any(not math.isfinite(x) for x in result):
        raise ValueError('Invalid finite vector: ' + name)
    return result


def _max_change(previous, current, name):
    a, b = _vector(previous, name), _vector(current, name)
    if len(a) != len(b):
        raise ValueError('Vector shape changed: ' + name)
    return max(abs(y - x) for x, y in zip(a, b))


def _kernel_change(previous, current, name):
    if len(previous) != len(current) or not previous:
        raise ValueError('Kernel size changed: ' + name)
    n = len(previous)
    if any(len(row) != n for row in previous) or any(len(row) != n for row in current):
        raise ValueError('Kernel must remain square: ' + name)
    a = _vector((x for row in previous for x in row), name)
    b = _vector((x for row in current for x in row), name)
    return math.sqrt(math.fsum((y - x) ** 2 for x, y in zip(a, b))) / max(math.sqrt(math.fsum(x * x for x in a)), 1e-30)


def window(previous, current):
    """Compute one consecutive 100-update window using only saved diagnostics."""
    a, b = int(previous['total_joint_step']), int(current['total_joint_step'])
    if b - a != 100:
        raise ValueError('Stability windows must span exactly 100 joint updates')
    if previous['query_ids'] != current['query_ids']:
        raise ValueError('Query identity/order changed across diagnostic window')
    nll_a, nll_b = float(previous['nll_after_update']), float(current['nll_after_update'])
    if not math.isfinite(nll_a) or not math.isfinite(nll_b):
        raise ValueError('Nonfinite diagnostic NLL')
    metrics = {
        'nll_relative_change': abs(nll_b - nll_a) / max(1.0, abs(nll_a)),
        'nll_signed_change': nll_b - nll_a,
        'training_kernel_relative_change': _kernel_change(previous['training_kernel'], current['training_kernel'], 'training'),
        'combined_kernel_relative_change': _kernel_change(previous['combined_kernel'], current['combined_kernel'], 'combined'),
        'train_mean_max_abs_change_raw': _max_change(previous['train_mean_raw'], current['train_mean_raw'], 'train mean'),
        'query_mean_max_abs_change_raw': _max_change(previous['query_mean_raw'], current['query_mean_raw'], 'query mean'),
        'query_latent_sd_max_abs_change_raw': _max_change(previous['query_latent_sd_raw'], current['query_latent_sd_raw'], 'query latent SD'),
        'query_model_sd_max_abs_change_raw': _max_change(previous['query_model_sd_raw'], current['query_model_sd_raw'], 'query model SD'),
    }
    criteria = {
        'objective_small': metrics['nll_relative_change'] <= TOLERANCES['nll_relative'],
        'similarity_stable': max(metrics['training_kernel_relative_change'], metrics['combined_kernel_relative_change']) <= TOLERANCES['kernel_relative'],
        'mean_prediction_stable': max(metrics['train_mean_max_abs_change_raw'], metrics['query_mean_max_abs_change_raw']) <= TOLERANCES['mean_max_abs_raw'],
        'uncertainty_stable': max(metrics['query_latent_sd_max_abs_change_raw'], metrics['query_model_sd_max_abs_change_raw']) <= TOLERANCES['sd_max_abs_raw'],
    }
    previous_top, current_top = list(previous['top5_ids']), list(current['top5_ids'])
    if len(previous_top) != min(5, len(previous['query_ids'])) or len(current_top) != min(5, len(current['query_ids'])):
        raise ValueError('Invalid top-five diagnostic size')
    if len(set(previous_top)) != len(previous_top) or len(set(current_top)) != len(current_top):
        raise ValueError('Duplicate top-five candidate ID')
    old, new = set(previous_top), set(current_top)
    return {'from_step': a, 'to_step': b, **metrics, 'criteria': criteria,
            'practical_stability_window': all(criteria.values()),
            'top5_selection_stable': old == new,
            'top5_order_stable': previous_top == current_top,
            'top1_selection_stable': previous['top1_id'] == current['top1_id'],
            'top5_intersection_count': len(old & new),
            'top5_added_ids': sorted(new - old), 'top5_removed_ids': sorted(old - new),
            'top5_previous_ids': previous_top, 'top5_current_ids': current_top,
            'units': {'mean_sd_changes': 'accuracy fraction; 0.001 equals 0.1 percentage points',
                      'kernel_relative_changes': 'unitless fraction; 0.01 equals 1 percent',
                      'nll': 'standardized-label per-sample NLL'},
            'tolerances': dict(TOLERANCES)}


def stability_at(records, step):
    """Require all four criteria in each of the last three consecutive windows."""
    step = int(step)
    lookup = {}
    for record in records:
        t = int(record['total_joint_step'])
        if t in lookup:
            raise ValueError('Duplicate diagnostic step: ' + str(t))
        lookup[t] = record
    if step not in lookup:
        raise ValueError('Requested diagnostic checkpoint is absent')
    required = [step - 300, step - 200, step - 100, step]
    missing = [t for t in required if t not in lookup]
    if step == 1000 or missing or min(required) < 1000:
        return {'total_joint_step': step, 'practical_stability': None,
                'criteria': dict.fromkeys(CRITERIA, None),
                'top5_selection_stable': None, 'top1_selection_stable': None,
                'top5_order_stable': None, 'windows_required': 3, 'windows_available': 0,
                'missing_diagnostic_steps': missing,
                'reason': '该项历史信息不足' if step == 1000 else '连续三个100步诊断窗口不完整',
                'windows': []}
    windows = [window(lookup[a], lookup[b]) for a, b in zip(required[:-1], required[1:])]
    criteria = {key: all(value['criteria'][key] for value in windows) for key in CRITERIA}
    return {'total_joint_step': step, 'practical_stability': all(criteria.values()),
            'criteria': criteria, 'top5_selection_stable': all(value['top5_selection_stable'] for value in windows),
            'top1_selection_stable': all(value['top1_selection_stable'] for value in windows),
            'top5_order_stable': all(value['top5_order_stable'] for value in windows),
            'windows_required': 3, 'windows_available': 3, 'missing_diagnostic_steps': [],
            'reason': None, 'windows': windows}


def diagnose_isolated(model, query_x, query_ids, y_center, y_scale, total_joint_step,
                      last_gradient=None, lrs=None):
    """Read current state on an isolated copy; never predict on the live model.

    query_x has only standardized features, and the caller supplies no query
    labels/SE. Training metrics use only model.y (the current selected subset).
    Timing and budget charging are owned by the caller so no nested double count
    occurs. total_joint_step is the post-update state represented by this record.
    """
    import numpy as np
    import torch
    import gpytorch
    from stage2_core import numerical_settings

    query_ids = list(query_ids)
    if len(query_ids) != len(set(query_ids)) or query_x.shape != (len(query_ids), 16) or len(query_ids) < 5:
        raise ValueError('Diagnostics require unique query IDs and [m,16] query features, m>=5')
    if query_x.dtype != torch.float64 or not bool(torch.isfinite(query_x).all()):
        raise ValueError('Query diagnostic features must be finite float64')
    if not math.isfinite(y_center) or not math.isfinite(y_scale) or y_scale <= 0:
        raise ValueError('Invalid fixed training y scaler')
    if int(total_joint_step) < 1000:
        raise ValueError('Length diagnostics cannot precede the fixed 1000-step source')
    if int(total_joint_step) == 1000 and last_gradient is not None:
        raise ValueError('1000-step baseline has no newly observed update gradient')
    if last_gradient is not None and (not math.isfinite(last_gradient) or last_gradient < 0):
        raise ValueError('Invalid last-update preclip gradient norm')
    rng = {'python': random.getstate(), 'numpy': copy.deepcopy(np.random.get_state()), 'torch': torch.get_rng_state().clone()}
    fingerprint = model.fingerprint()
    try:
        isolated = model.clone()
        if isolated.fingerprint() != fingerprint:
            raise RuntimeError('Diagnostic clone differs from current mathematical state')
        with torch.no_grad(), numerical_settings():
            train_embedding, _ = isolated.encode(isolated.x)
            combined_x = torch.cat((isolated.x, query_x.detach().clone()), dim=0)
            combined_embedding, _ = isolated.encode(combined_x)
            isolated.gp.train()
            isolated.gp.likelihood.train()
            isolated.gp.set_train_data(inputs=train_embedding, targets=isolated.y, strict=False)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(isolated.gp.likelihood, isolated.gp)
            nll = -mll(isolated.gp(train_embedding), isolated.y)
            train_kernel = isolated.gp.covar_module(train_embedding).to_dense()
            combined_kernel = isolated.gp.covar_module(combined_embedding).to_dense()
            # This path refreshes history/cache on the isolated model only.
            means, variances = isolated.predict(combined_x)
            if not all(bool(torch.isfinite(value).all()) for value in (nll, means, variances, train_kernel, combined_kernel, train_embedding)):
                raise FloatingPointError('Nonfinite diagnostic objective/kernel/prediction')
            if bool((variances < -1e-8).any()):
                raise FloatingPointError('Raw standardized latent variance below -1e-8')
            count = len(isolated.y)
            train_mean, query_mean = means[:count], means[count:]
            train_var, query_var = variances[:count], variances[count:]
            extra = float(isolated.gp.likelihood.second_noise.item())
            lengthscale = float(isolated.gp.covar_module.base_kernel.lengthscale.item())
            outputscale = float(isolated.gp.covar_module.outputscale.item())
            mean_parameter = float(isolated.gp.mean_module.constant.item())
            if (not all(math.isfinite(value) for value in (extra, lengthscale, outputscale, mean_parameter))
                    or extra < 1e-6 or lengthscale < 1e-4 or outputscale < 1e-8):
                raise FloatingPointError('Invalid diagnostic GP parameters')
            train_mean_raw = train_mean * y_scale + y_center
            query_mean_raw = query_mean * y_scale + y_center
            train_y_raw = isolated.y * y_scale + y_center
            delta = train_mean_raw - train_y_raw
            latent_sd = y_scale * query_var.clamp_min(0).sqrt()
            model_sd = y_scale * (query_var.clamp_min(0) + extra).sqrt()
            if not all(bool(torch.isfinite(value).all()) for value in (train_mean_raw, query_mean_raw, delta, latent_sd, model_sd)):
                raise FloatingPointError('Nonfinite raw-unit diagnostic mean/standard deviation')
            order = sorted(range(len(query_ids)), key=lambda i: (-float(query_mean_raw[i]), query_ids[i]))
            gp = {'mean': mean_parameter, 'lengthscale': lengthscale, 'outputscale': outputscale,
                  'extra_noise_variance_standardized': extra,
                  'lengthscale_distance_to_lower': lengthscale - 1e-4,
                  'outputscale_distance_to_lower': outputscale - 1e-8,
                  'extra_noise_distance_to_lower': extra - 1e-6}
            result = {
                'total_joint_step': int(total_joint_step), 'additional_completed_steps': int(total_joint_step) - 1000,
                'state_hash': fingerprint, 'nll_after_update': float(nll),
                'nll_unit': 'pure data NLL per selected observation; standardized labels; updated state',
                'gradient_before_last_update': last_gradient,
                'gradient_timing': 'before the latest update' if last_gradient is not None else 'no added update at restored baseline',
                'learning_rates': copy.deepcopy(lrs), 'query_ids': query_ids,
                'train_mean_standardized': train_mean.tolist(), 'train_mean_raw': train_mean_raw.tolist(),
                'query_mean_standardized': query_mean.tolist(), 'query_mean_raw': query_mean_raw.tolist(),
                'train_latent_variance_standardized': train_var.tolist(), 'train_latent_variance_raw': (train_var * y_scale ** 2).tolist(),
                'query_latent_variance_standardized': query_var.tolist(), 'query_latent_variance_raw': (query_var * y_scale ** 2).tolist(),
                'query_latent_sd_raw': latent_sd.tolist(), 'query_model_sd_raw': model_sd.tolist(),
                'extra_noise_variance_standardized': extra, 'y_center': float(y_center), 'y_scale': float(y_scale),
                'train_mae_raw': float(delta.abs().mean()), 'train_rmse_raw': float(delta.square().mean().sqrt()),
                'training_kernel': train_kernel.tolist(), 'combined_kernel': combined_kernel.tolist(),
                'kernel_definition': 'GP noiseless covariance; combined order selected training then query input order',
                'gp_parameters': gp,
                'representation': {'dimensions': int(train_embedding.shape[1]),
                                   'training_rms': float(train_embedding.square().mean().sqrt()),
                                   'training_row_norm_mean': float(train_embedding.norm(dim=1).mean()),
                                   'training_coordinate_std_mean': float(train_embedding.std(dim=0, unbiased=False).mean())},
                'top5_ids': [query_ids[i] for i in order[:5]], 'top1_id': query_ids[order[0]],
                'negative_roundoff_variance_count_train': int((train_var < 0).sum()),
                'negative_roundoff_variance_count_query': int((query_var < 0).sum()),
                'units': {'mean_raw': 'accuracy fraction', 'latent_variance_raw': 'accuracy fraction squared',
                          'sd_raw': 'accuracy fraction', 'extra_noise': 'standardized variance, not standard deviation'},
                'query_y_or_SE_used': False, 'diagnostic_model_isolated': True,
            }
        if model.fingerprint() != fingerprint:
            raise RuntimeError('Diagnostics changed the live mathematical state')
        return result
    finally:
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
