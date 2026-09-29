"""Recompute sealed, label-free length stability without importing PyTorch.

Reads compact diagnostic JSON and lossless float64 kernels. It does not load
model states, selected labels, scoring labels, a trainer, or a prediction model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from length_diagnostics import CRITERIA, TOLERANCES, stability_at, window

STEPS = tuple(range(1000, 4001, 100))
CHECKPOINTS = (1000, 2000, 4000)
WINDOW_METRICS = (
    'nll_relative_change', 'nll_signed_change', 'training_kernel_relative_change',
    'combined_kernel_relative_change', 'train_mean_max_abs_change_raw',
    'query_mean_max_abs_change_raw', 'query_latent_sd_max_abs_change_raw',
    'query_model_sd_max_abs_change_raw',
)


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_path(root, relative, expected_hash):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'Sealed path escapes artifact root')
    require(sha(path) == expected_hash, 'Sealed artifact hash differs: ' + str(relative))
    return path


def equivalent(a, b):
    if isinstance(a, bool) or a is None or isinstance(a, str):
        return type(a) is type(b) and a == b
    if isinstance(a, (float, int)):
        return isinstance(b, (float, int)) and not isinstance(b, bool) and math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equivalent(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return False


def spread(values):
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {'count': 0, 'min': None, 'q25': None, 'median': None, 'q75': None,
                'q90': None, 'q95': None, 'max': None, 'mean': None}
    require(values.ndim == 1 and np.isfinite(values).all(), 'Nonfinite/invalid drift summary values')
    q = np.quantile(values, [0., .25, .5, .75, .9, .95, 1.], method='linear')
    return {'count': len(values), **dict(zip(('min', 'q25', 'median', 'q75', 'q90', 'q95', 'max'), map(float, q))),
            'mean': float(values.mean())}


def counts(rows, field):
    return {'stable': sum(row[field] is True for row in rows),
            'unstable': sum(row[field] is False for row in rows),
            'unavailable': sum(row[field] is None for row in rows)}


def missing_checkpoint(context, step):
    return {'context_id': context['context_id'], 'n': context['n'], 'fold': context['fold'],
            'sample_seed': context['sample_seed'], 'model_seed': context['model_seed'],
            'total_joint_step': step, 'practical_stability': None,
            'criteria': dict.fromkeys(CRITERIA, None),
            'top5_selection_stable': None, 'top5_order_stable': None, 'top1_selection_stable': None,
            'windows_required': 3, 'windows_available': 0, 'windows': [],
            'reason': '该项历史信息不足' if step == 1000 else '预定检查点或连续三个窗口未完成',
            'checkpoint_record_available': False}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows, fallback_fields):
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(fallback_fields)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def point_summary(row):
    keys = ('context_id', 'subset_id', 'n', 'fold', 'sample_seed', 'model_seed', 'total_joint_step',
            'additional_completed_steps', 'state_hash', 'nll_after_update', 'gradient_before_last_update',
            'learning_rates', 'train_mae_raw', 'train_rmse_raw', 'y_center', 'y_scale',
            'extra_noise_variance_standardized', 'top5_ids', 'top1_id',
            'negative_roundoff_variance_count_train', 'negative_roundoff_variance_count_query',
            'kernel_path', 'kernel_sha256')
    result = {key: row[key] for key in keys}
    result.update({'gp_' + key: value for key, value in row['gp_parameters'].items()})
    result.update({'representation_' + key: value for key, value in row['representation'].items()})
    result.update({'cumulative_' + key: row['resource_totals'][key]
                   for key in ('training_seconds', 'prediction_seconds', 'verification_seconds', 'seconds', 'updates')})
    return result


def check_point(row, context):
    require(row['query_y_or_SE_used'] is False and row['diagnostic_model_isolated'] is True, 'Diagnostic query-label/copy contract failed')
    for key in ('context_id', 'subset_id', 'n', 'fold', 'sample_seed', 'model_seed'):
        require(row[key] == context[key], 'Diagnostic context differs: ' + key)
    step = row['total_joint_step']
    require(step in STEPS and row['additional_completed_steps'] == step - 1000, 'Diagnostic step schedule differs')
    require(math.isfinite(row['nll_after_update']), 'Nonfinite updated diagnostic NLL')
    require(row['gradient_before_last_update'] is None if step == 1000 else
            row['gradient_before_last_update'] is not None and math.isfinite(row['gradient_before_last_update']) and row['gradient_before_last_update'] >= 0,
            'Baseline/new-update gradient timing differs')
    ids = row['query_ids']
    require(len(ids) == len(set(ids)) and len(ids) >= 5, 'Invalid diagnostic query candidate IDs')
    n, m = context['n'], len(ids)
    require(np.asarray(row['training_kernel']).shape == (n, n), 'Training kernel size differs from selected n')
    require(np.asarray(row['combined_kernel']).shape == (n + m, n + m), 'Combined kernel order/size differs')
    require(np.isfinite(row['training_kernel']).all() and np.isfinite(row['combined_kernel']).all(), 'Nonfinite saved kernel')
    require(np.allclose(np.asarray(row['combined_kernel'])[:n, :n], row['training_kernel'], rtol=0, atol=1e-12),
            'Combined kernel training-prefix does not match training kernel')
    for key, count in (('train_mean_raw', n), ('query_mean_raw', m), ('train_mean_standardized', n),
                       ('query_mean_standardized', m), ('query_latent_sd_raw', m), ('query_model_sd_raw', m),
                       ('train_latent_variance_standardized', n), ('query_latent_variance_standardized', m)):
        require(np.asarray(row[key]).shape == (count,) and np.isfinite(row[key]).all(), 'Invalid saved diagnostic vector: ' + key)
    scale, center = row['y_scale'], row['y_center']
    require(math.isfinite(scale) and scale > 0 and math.isfinite(center), 'Invalid fixed diagnostic scaler')
    extra = row['extra_noise_variance_standardized']
    require(math.isfinite(extra) and extra >= 1e-6, 'Invalid extra variance')
    for prefix in ('train', 'query'):
        variance = np.asarray(row[prefix + '_latent_variance_standardized'])
        require(np.all(variance >= -1e-8), 'Standardized latent variance below failure tolerance')
        require(np.allclose(row[prefix + '_mean_raw'], np.asarray(row[prefix + '_mean_standardized']) * scale + center,
                            rtol=0, atol=1e-12), 'Mean units changed')
        require(np.allclose(row[prefix + '_latent_variance_raw'], variance * scale ** 2, rtol=0, atol=1e-12), 'Variance units changed')
        require(row['negative_roundoff_variance_count_' + prefix] == int(np.sum(variance < 0)), 'Negative-roundoff count differs')
    variance = np.asarray(row['query_latent_variance_standardized'])
    require(np.allclose(row['query_latent_sd_raw'], scale * np.sqrt(np.maximum(variance, 0)), rtol=0, atol=1e-12), 'Latent SD units/formula changed')
    require(np.allclose(row['query_model_sd_raw'], scale * np.sqrt(np.maximum(variance, 0) + extra), rtol=0, atol=1e-12), 'Model SD extra variance composition changed')
    order = sorted(range(m), key=lambda i: (-row['query_mean_raw'][i], ids[i]))
    require(row['top5_ids'] == [ids[i] for i in order[:5]] and row['top1_id'] == ids[order[0]], 'Saved selection disagrees with GP means')
    rates = row['learning_rates']
    require(rates == [1e-4, 1e-3] or rates == {'mapping': 1e-4, 'gp': 1e-3}, 'Diagnostic recorded LR differs from fixed endpoint LR')


def run(root, output_dir=None):
    root = Path(root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / 'stability_summary'
    require(output not in {root, root / 'code', root / 'source', root / 'diagnostics', root / 'scoring'}, 'Output would replace an input artifact directory')
    seal_path = root / 'predictions_manifest.json'
    seal = read(seal_path)
    require(seal.get('sealed') is True, 'Predictions/diagnostics are not uniformly sealed')
    require(seal.get('execution_status') in ('COMPLETE_VALID', 'PARTIAL'), 'Only COMPLETE_VALID/PARTIAL sealed runs can be summarized')
    for name in ('protocol', 'code_manifest', 'source_manifest', 'diagnostics_manifest'):
        verified_path(root, name + '.json', seal[name + '_sha256'])
    protocol = read(root / 'protocol.json')
    require(protocol['stability_thresholds'] == {'nll_relative': 1e-3, 'training_and_combined_kernel_relative': .01,
            'train_and_query_mean_max_raw': .001, 'query_latent_and_model_sd_max_raw': .001, 'consecutive_windows': 3},
            'Frozen practical stability thresholds differ')
    code_manifest = read(root / 'code_manifest.json')
    for name, value in code_manifest['files'].items():
        verified_path(root, name, value)
    sources = read(root / 'source_manifest.json')
    context_list = sources['contexts']
    contexts = {row['context_id']: row for row in context_list}
    expected_contexts = {(fold, ss, n, ms) for fold in range(5) for ss in (20260922, 20260923) for n in (20, 50) for ms in (0, 1, 2)}
    require(len(context_list) == len(contexts) == 60 and
            {(c['fold'], c['sample_seed'], c['n'], c['model_seed']) for c in context_list} == expected_contexts,
            'Source matrix is not all original 60 contexts')
    manifest = read(root / 'diagnostics_manifest.json')
    entries = manifest['files']
    require(manifest['record_count'] == len(entries), 'Diagnostic record_count differs from file list')
    require(manifest['kernel_values_saved_losslessly_float64'] is True and manifest['queries_contain_no_y_or_SE'] is True,
            'Diagnostic manifest precision/query-label contract differs')
    lookup = {}
    for entry in entries:
        key = (entry['context_id'], entry['total_joint_step'])
        require(key[0] in contexts and key[1] in STEPS and key not in lookup, 'Unknown or duplicate diagnostic context/step')
        lookup[key] = entry
    expected_keys = {(cid, step) for cid in contexts for step in STEPS}
    missing = sorted(expected_keys - set(lookup))
    complete = seal['execution_status'] == 'COMPLETE_VALID'
    require(not complete or (len(entries) == 1860 and not missing), 'COMPLETE_VALID requires all 1860 diagnostic times')
    require(len(entries) <= 1860, 'Extra diagnostic times outside fixed protocol')

    points, windows, stability_rows = [], [], []
    recorded_windows_verified = 0
    recorded_checkpoint_stability_verified = 0
    for context in context_list:
        cid = context['context_id']
        query_path = context['query_path']
        query = read(verified_path(root, query_path, sources['files'][query_path]))
        records = []
        by_step = {}
        for step in STEPS:
            entry = lookup.get((cid, step))
            if entry is None:
                continue
            row = read(verified_path(root, entry['path'], entry['sha256']))
            require(row['context_id'] == cid and row['total_joint_step'] == step, 'Diagnostic file/manifest identity differs')
            require(row['kernel_path'] == entry['kernel_path'] and row['kernel_sha256'] == entry['kernel_sha256'], 'Kernel reference differs from sealed diagnostic manifest')
            kernel_path = verified_path(root, entry['kernel_path'], entry['kernel_sha256'])
            with np.load(kernel_path, allow_pickle=False) as kernels:
                require(set(kernels.files) == {'training_kernel', 'combined_kernel'}, 'Unexpected kernel archive contents')
                for name in ('training_kernel', 'combined_kernel'):
                    require(kernels[name].dtype == np.dtype('float64'), 'Kernel archive is not lossless float64')
                    row[name] = kernels[name].tolist()
            check_point(row, context)
            require(row['query_ids'] == query['ids'], 'Diagnostic query order differs from original frozen context')
            if step == 1000:
                require(row['state_hash'] == context['model_state_hash'], '1000-step baseline diagnostic state differs from original source')
            by_step[step] = row
            records.append(row)
            points.append(point_summary(row))
            if step == 1000:
                require(row['window_from_previous'] is None, 'Baseline cannot contain a fabricated historical prediction window')
            if step - 100 in by_step:
                drift = window(by_step[step - 100], row)
                require(equivalent(drift, row['window_from_previous']), 'Saved window differs from independently recomputed kernel/prediction drift')
                recorded_windows_verified += 1
                windows.append({'context_id': cid, 'n': context['n'], 'fold': context['fold'],
                                'sample_seed': context['sample_seed'], 'model_seed': context['model_seed'],
                                'from_step': drift['from_step'], 'to_step': drift['to_step'],
                                **{key: drift[key] for key in WINDOW_METRICS},
                                **drift['criteria'],
                                'practical_stability_window': drift['practical_stability_window'],
                                **{key: drift[key] for key in ('top5_selection_stable', 'top5_order_stable', 'top1_selection_stable',
                                                              'top5_intersection_count', 'top5_added_ids', 'top5_removed_ids')}})
        for step in CHECKPOINTS:
            if step in by_step:
                value = {'context_id': cid, 'n': context['n'], 'fold': context['fold'],
                         'sample_seed': context['sample_seed'], 'model_seed': context['model_seed'],
                         **stability_at(records, step), 'checkpoint_record_available': True}
                saved_path = root / f'stability/{cid}_j{step}.json'
                if saved_path.exists():
                    recorded = read(saved_path)
                    expected = {'context_id': cid, 'n': context['n'], **stability_at(records, step)}
                    require(equivalent(recorded, expected), 'Recorded checkpoint stability differs from recomputation')
                    recorded_checkpoint_stability_verified += 1
            else:
                value = missing_checkpoint(context, step)
            stability_rows.append(value)
    require(not complete or len(windows) == 1800, 'COMPLETE_VALID requires all 1800 adjacent windows')

    summary = {
        'execution_status': seal['execution_status'], 'complete_diagnostic_matrix': not missing,
        'diagnostic_record_count': len(points), 'diagnostic_record_count_expected': 1860,
        'adjacent_window_count': len(windows), 'adjacent_window_count_expected': 1800,
        'missing_diagnostic_pairs': [{'context_id': cid, 'total_joint_step': step} for cid, step in missing],
        'recorded_windows_verified': recorded_windows_verified,
        'recorded_checkpoint_stability_verified': recorded_checkpoint_stability_verified,
        'criteria_tolerances': dict(TOLERANCES), 'consecutive_windows_required': 3,
        'units': {'prediction_mean_and_sd_drift': 'accuracy fraction; 0.001 is 0.1 percentage points',
                  'kernel_relative_drift': 'fraction; 0.01 is one percent', 'NLL': 'standardized-label per-observation NLL'},
        'baseline1000': '该项历史信息不足；只作后续变化基线，不回填历史稳定状态',
        'quantile_method': 'linear interpolation of ordered values at (count-1)*q; descriptive only',
        'window_observations_are_not_independent_experiments': True,
        'missing_results_are_not_imputed': True, 'next_stage_authorized': False,
        'per_n': {},
    }
    for n in (20, 50):
        all_windows = [row for row in windows if row['n'] == n]
        checkpoints = {}
        for step in CHECKPOINTS:
            selected = [row for row in stability_rows if row['n'] == n and row['total_joint_step'] == step]
            require(len(selected) == 30, 'Stability denominator must remain 30 per n/checkpoint')
            status_counts = counts(selected, 'practical_stability')
            terminal_windows = [drift for row in selected for drift in row['windows']]
            maxima = {metric: [max(w[metric] for w in row['windows']) for row in selected if len(row['windows']) == 3]
                      for metric in WINDOW_METRICS}
            checkpoints[str(step)] = {
                'expected_trajectory_count': 30,
                'checkpoint_record_count': sum(row['checkpoint_record_available'] for row in selected),
                'evaluated_trajectory_count': status_counts['stable'] + status_counts['unstable'],
                'practical_stability_counts': status_counts,
                'stable_trajectory_ids': [row['context_id'] for row in selected if row['practical_stability'] is True],
                'unstable_trajectory_ids': [row['context_id'] for row in selected if row['practical_stability'] is False],
                'unavailable_trajectory_ids': [row['context_id'] for row in selected if row['practical_stability'] is None],
                'all_30_trajectories_practically_stable': status_counts['stable'] == 30,
                'criteria_counts': {key: {'stable': sum(row['criteria'][key] is True for row in selected),
                                          'unstable': sum(row['criteria'][key] is False for row in selected),
                                          'unavailable': sum(row['criteria'][key] is None for row in selected)} for key in CRITERIA},
                'top5_selection_counts': counts(selected, 'top5_selection_stable'),
                'top5_order_counts': counts(selected, 'top5_order_stable'),
                'top1_selection_counts': counts(selected, 'top1_selection_stable'),
                'last_three_window_metric_spread': {metric: spread(w[metric] for w in terminal_windows) for metric in WINDOW_METRICS},
                'trajectory_worst_of_last_three_metric_spread': {metric: spread(values) for metric, values in maxima.items()},
                'reason_if_unavailable': '该项历史信息不足' if step == 1000 else '检查点或连续三个窗口缺失时不推断稳定',
            }
        summary['per_n'][str(n)] = {'checkpoints': checkpoints, 'all_adjacent_windows': len(all_windows),
                                    'all_window_metric_spread': {metric: spread(w[metric] for w in all_windows) for metric in WINDOW_METRICS}}

    require('torch' not in sys.modules, 'Stability-only recomputation unexpectedly imported PyTorch')
    provenance = {'recomputed_at_utc': datetime.now(timezone.utc).isoformat(),
                  'predictions_manifest_sha256': sha(seal_path),
                  'diagnostics_manifest_sha256': sha(root / 'diagnostics_manifest.json'),
                  'source_manifest_sha256': sha(root / 'source_manifest.json'),
                  'stability_module_sha256': sha(Path(__file__).parent / 'length_diagnostics.py'),
                  'recompute_source_sha256': sha(__file__), 'torch_imported': False,
                  'training_triggered': False, 'model_prediction_triggered': False,
                  'scoring_labels_read': False, 'training_labels_read': False,
                  'all_diagnostic_and_kernel_hashes_verified': True}
    summary['provenance'] = provenance
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'diagnostic_points.csv', points, ('context_id', 'total_joint_step', 'state_hash'))
    write_csv(output / 'window_metrics.csv', windows, ('context_id', 'from_step', 'to_step'))
    write_json(output / 'trajectory_stability.json', stability_rows)
    flat = [{**{key: value for key, value in row.items() if key not in ('criteria', 'windows')},
             **row['criteria']} for row in stability_rows]
    write_csv(output / 'trajectory_stability.csv', flat, ('context_id', 'total_joint_step', 'practical_stability'))
    write_json(output / 'summary.json', summary)
    write_json(output / 'provenance.json', provenance)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    result = run(args.root, args.output_dir)
    compact = {'execution_status': result['execution_status'], 'diagnostic_records': result['diagnostic_record_count'],
               'adjacent_windows': result['adjacent_window_count'],
               'stability_counts': {n: {step: result['per_n'][n]['checkpoints'][step]['practical_stability_counts']
                                         for step in ('2000', '4000')} for n in ('20', '50')},
               'training_triggered': False, 'torch_imported': False}
    print(json.dumps(compact, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
