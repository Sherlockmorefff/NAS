"""Read-only audit of a completed DKL tail using saved tensors and journals.

Never constructs a GP/model/optimizer, predicts, fits, or opens scoring labels.
The only writes are the requested audit JSON result. Requires torch/numpy solely
for decoding the already-saved state and kernel files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def semantic(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def saved_fingerprint(state):
    values = {'mapper.' + name: value for name, value in state['mapper'].items()}
    values.update({'gp.' + name: value for name, value in state['gp'].items()})
    values.update({'data.x': state['x'], 'data.y': state['y'], 'data.known': state['known'],
                   'likelihood.fixed_noise': state['known']})
    return tensor_hash(values)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def names_from_frozen_state(source):
    mapping = ['mapper.' + name for name in source['mapper'] if not name.startswith('regressor.')]
    gp = ['gp.' + name for name in source['gp']
          if name.rsplit('.', 1)[-1] in {'raw_noise', 'raw_constant', 'raw_outputscale', 'raw_lengthscale'}]
    need(len(mapping) == 8 and len(gp) == 4, 'Frozen C8/ExactGP named parameter structure differs')
    # State dictionary insertion order is the frozen module registration order;
    # this is the order used by named_parameters in the original joint Adam.
    return [mapping, gp]


def parameter_tensor(state, name):
    category, key = name.split('.', 1)
    return state[category][key]


def adam_steps(state, names):
    optimizer = state['optimizer']
    need(isinstance(optimizer, dict) and len(optimizer['param_groups']) == 2, 'Adam state/group missing')
    counts, identities = {}, []
    for index, (group, ordered_names) in enumerate(zip(optimizer['param_groups'], names)):
        need(group['lr'] == (1e-4, 1e-3)[index], 'Actual endpoint LR changed')
        need(group['base_lr'] == (1e-3, 1e-2)[index], 'Source base LR changed')
        need(tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0,
             'Adam betas/eps/weight_decay changed')
        need(len(group['params']) == len(ordered_names), 'Adam parameter group length changed')
        for key, name in zip(group['params'], ordered_names):
            item = optimizer['state'][key]
            need({'step', 'exp_avg', 'exp_avg_sq'} <= item.keys(), 'Incomplete moments: ' + name)
            value = float(item['step'])
            need(math.isfinite(value) and value >= 1 and int(value) == value, 'Invalid actual Adam step: ' + name)
            counts[name] = int(value)
            parameter = parameter_tensor(state, name)
            need(parameter.dtype == torch.float64 and bool(torch.isfinite(parameter).all()), 'Nonfinite/non-double parameter')
            for moment in ['exp_avg', 'exp_avg_sq']:
                array = item[moment]
                need(array.dtype == parameter.dtype and array.shape == parameter.shape and bool(torch.isfinite(array).all()),
                     'Moment shape/dtype/finiteness differs: ' + name)
            need(not bool((item['exp_avg_sq'] < 0).any()), 'Negative Adam second moment')
        identities.extend(group['params'])
    need(len(identities) == len(set(identities)) and set(identities) == set(optimizer['state']), 'Adam IDs incomplete/duplicated')
    return counts


def checked_file(root, relative, expected):
    path = root / relative
    need(path.is_file(), 'Missing evidence file: ' + relative)
    need(sha(path) == expected, 'File hash mismatch: ' + relative)
    return path


def source_input_check(state, source, train, context):
    need(state['x'].shape == (context['n'], 16) and state['known'].shape == state['y'].shape == (context['n'],),
         'Selected training dimensions differ')
    for key, field in [('x', 'x_standardized'), ('y', 'y_standardized'), ('known', 'known_variance_standardized')]:
        need(exact(state[key], source[key]), 'Saved selected tensors differ from source: ' + key)
        need(torch.equal(state[key], torch.tensor(train[field], dtype=torch.float64)), 'Selected training snapshot differs: ' + key)
        need(bool(torch.isfinite(state[key]).all()), 'Nonfinite selected tensors')
    need(not bool((state['known'] < 0).any()), 'Negative known variance')
    need(train['scaler'] == context['scaler'] and semantic(train['scaler']) == context['scaler_sha256'], 'Scaler changed')
    need(train['ids'] == train['scaler']['fit_ids'] and len(train['ids']) == context['n'], 'Selected IDs/scaler scope mismatch')
    need(semantic(train['ids']) == context['training_ids_sha256'], 'Selected training order changed')
    need(tensor_hash({'x': state['x'], 'y': state['y'], 'known': state['known']}) == context['training_tensor_sha256'],
         'Training tensor semantic hash mismatch')
    for name, parameter in source['mapper'].items():
        if name.startswith('regressor.'):
            need(exact(state['mapper'][name], parameter), 'Frozen regression head changed')
    for name, value in source['gp'].items():
        if '_constraint.' in name:
            need(exact(state['gp'][name], value), 'GP constraint buffer changed')


def endpoint_check(state, source, train, context, total, names, origin_steps):
    need(state['model_type'] == 'MAPPED8', 'Endpoint is not the mapped DKL model')
    need(state['scheduler'] == source['scheduler'] == state['source_scheduler'], 'Original source scheduler changed')
    need(state['ordered_parameter_names'] == names, 'Ordered Adam names changed')
    metadata = state['continuation']
    need(metadata['source_completed_steps'] == 1000 and metadata['total_joint_step'] == total
         and metadata['additional_completed_steps'] == total - 1000 and metadata['continuation_target'] == 4000
         and metadata['schedule'] == 'fixed_endpoint_lr', 'Continuation metadata mismatch')
    need(metadata['actual_lrs'] == {'mapping': 1e-4, 'gp': 1e-3}, 'Continuation LR metadata mismatch')
    need(metadata['source_state_hash'] == source['fingerprint'], 'Continuation source math hash changed')
    need(metadata['source_optimizer_parameter_steps'] == origin_steps, 'Original per-parameter steps changed')
    need(state['optimizer']['param_groups'] == source['optimizer']['param_groups'], 'Adam group metadata changed')
    steps = adam_steps(state, names)
    need(all(steps[name] == initial + total - 1000 for name, initial in origin_steps.items()), 'Adam step increment mismatch')
    need(saved_fingerprint(state) == state['fingerprint'], 'Recomputed saved math fingerprint differs')
    source_input_check(state, source, train, context)
    return steps


def audit(root):
    started = time.monotonic()
    seal = read(root / 'predictions_manifest.json')
    need(seal['sealed'] is True and seal['execution_status'] == 'COMPLETE_VALID', 'Audit requires COMPLETE_VALID sealed run')
    need(seal['new_formal_updates'] == 180000 and seal['new_endpoint_count'] == 120
         and seal['completed_trajectories'] == 60, 'Formal matrix seal counts mismatch')
    for key, filename in [('protocol_sha256', 'protocol.json'), ('code_manifest_sha256', 'code_manifest.json'),
                          ('source_manifest_sha256', 'source_manifest.json'), ('diagnostics_manifest_sha256', 'diagnostics_manifest.json')]:
        checked_file(root, filename, seal[key])
    checked_hashes = 0
    for filename in ['frozen_manifest.json', 'code_manifest.json', 'source_manifest.json']:
        for relative, expected in read(root / filename)['files'].items():
            need(not relative.startswith('scoring/'), 'Unexpected scoring data in training-source manifest')
            checked_file(root, relative, expected)
            checked_hashes += 1
    contexts = read(root / 'source_manifest.json')['contexts']
    need(len(contexts) == 60 and len({c['context_id'] for c in contexts}) == 60, 'Context count/identity mismatch')
    files = seal['files']
    need(len(files) == 180 and len({(p['context_id'], p['total_joint_step']) for p in files}) == 180,
         'Scientific endpoint count/uniqueness mismatch')
    endpoints = {(p['context_id'], p['total_joint_step']): p for p in files}
    journal = read(root / 'diagnostics_manifest.json')
    need(journal['record_count'] == len(journal['files']) == 1860, 'Diagnostic journal must contain1860 records')
    records = {(r['context_id'], r['total_joint_step']): r for r in journal['files']}
    need(len(records) == 1860, 'Duplicate diagnostic point')
    window_count, baseline_count, endpoint_count = 0, 0, 0
    trajectory_cost = {key: 0.0 for key in ['training_seconds', 'prediction_seconds', 'verification_seconds', 'updates']}
    maximum_baseline_error = {'mean': 0.0, 'variance': 0.0}
    origin_step_values, final_step_values = set(), set()
    new_prediction_points = 0
    minimum_latent_variance = math.inf
    examples = []
    for context in contexts:
        cid = context['context_id']
        source_path = checked_file(root, context['checkpoint_path'], context['checkpoint_sha256'])
        source = load(source_path)
        schedule = source['scheduler']
        need((schedule['mode'], schedule['planned_steps'], schedule['completed']) == ('joint', 1000, 1000), 'Invalid source scheduler')
        need(saved_fingerprint(source) == source['fingerprint'] == context['model_state_hash'], 'Source math hash mismatch')
        train = read(checked_file(root, context['training_path'], context['training_sha256']))
        original_prediction = read(checked_file(root, context['prediction_path'], context['prediction_sha256']))
        need(original_prediction['endpoint_state_hash'] == source['fingerprint'], 'Source prediction math identity mismatch')
        source_input_check(source, source, train, context)
        names = names_from_frozen_state(source)
        origin_steps = adam_steps(source, names)
        origin_step_values.update(origin_steps.values())
        baseline = read(root / 'verification/baseline' / (cid + '.json'))
        need(baseline['pass'] is True and baseline['new_updates'] == 0, 'Baseline verification did not pass')
        need(baseline['source_checkpoint_sha256'] == context['checkpoint_sha256']
             and baseline['model_state_hash'] == source['fingerprint']
             and baseline['scaler_sha256'] == context['scaler_sha256'], 'Baseline verification identity mismatch')
        need(baseline['optimizer']['all_steps_match_added_updates'] is True, 'Baseline optimizer step audit failed')
        baseline_count += 1
        previous = None
        for total in range(1000, 4001, 100):
            descriptor = records[(cid, total)]
            row = read(checked_file(root, descriptor['path'], descriptor['sha256']))
            need(row['context_id'] == cid and row['total_joint_step'] == total
                 and row['additional_completed_steps'] == total - 1000, 'Diagnostic context/step mismatch')
            kernel_path = checked_file(root, row['kernel_path'], row['kernel_sha256'])
            need(descriptor['kernel_path'] == row['kernel_path'] and descriptor['kernel_sha256'] == row['kernel_sha256'],
                 'Journal kernel linkage differs')
            with np.load(kernel_path, allow_pickle=False) as kernel:
                for key, n in [('training_kernel', context['n']),
                               ('combined_kernel', context['n'] + len(original_prediction['candidate_ids']))]:
                    matrix = kernel[key]
                    need(matrix.dtype == np.float64 and matrix.shape == (n, n) and np.isfinite(matrix).all(), 'Invalid saved diagnostic kernel')
                    need(np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-8), 'Kernel is not symmetric')
                    need(np.allclose(np.diag(matrix), row['gp_parameters']['outputscale'], atol=1e-10, rtol=1e-8),
                         'Kernel diagonal is not noiseless outputscale')
            need(row['query_ids'] == original_prediction['candidate_ids'], 'Diagnostic query ID/order mismatch')
            need(row['query_y_or_SE_used'] is False and row['diagnostic_model_isolated'] is True, 'Diagnostic isolation metadata differs')
            need(row['learning_rates'] in ([1e-4, 1e-3], {'mapping': 1e-4, 'gp': 1e-3}), 'Logged actual LR differs')
            need(math.isfinite(row['nll_after_update']), 'Nonfinite post-update NLL')
            for key in ['query_mean_standardized', 'query_mean_raw', 'query_latent_variance_standardized',
                        'query_latent_sd_raw', 'query_model_sd_raw', 'train_mean_raw']:
                need(np.isfinite(np.asarray(row[key])).all(), 'Nonfinite diagnostic vector: ' + key)
            variance = np.asarray(row['query_latent_variance_standardized'])
            need(np.all(variance >= -1e-8), 'Negative standardized latent variance failure')
            minimum_latent_variance = min(minimum_latent_variance, float(variance.min()))
            if total == 1000:
                need(row['window_from_previous'] is None and row['gradient_before_last_update'] is None,
                     'Baseline fabricated preceding window or added-step gradient')
                for observed, original, label in [('query_mean_standardized', 'mean_standardized', 'mean'),
                                                  ('query_latent_variance_standardized', 'latent_variance_standardized', 'variance')]:
                    new, old = np.asarray(row[observed]), np.asarray(original_prediction[original])
                    need(new.shape == old.shape and np.allclose(new, old, atol=1e-8, rtol=1e-6), 'Baseline prediction does not match')
                    maximum_baseline_error[label] = max(maximum_baseline_error[label], float(np.max(np.abs(new - old))))
            else:
                window = row['window_from_previous']
                need(window['from_step'] == total - 100 and window['to_step'] == total, 'Diagnostic window has gap/wrong timing')
                need(previous['total_joint_step'] == total - 100, 'Previous diagnostic absent')
                expected_change = abs(row['nll_after_update'] - previous['nll_after_update']) / max(1., abs(previous['nll_after_update']))
                need(math.isclose(window['nll_relative_change'], expected_change, rel_tol=1e-12, abs_tol=1e-12),
                     'Recorded relative objective change differs')
                window_count += 1
            previous = row
            if total in [1000, 2000, 4000]:
                descriptor = endpoints[(cid, total)]
                prediction = read(checked_file(root, descriptor['path'], descriptor['sha256']))
                state_path = checked_file(root, descriptor['state_path'], descriptor['state_file_sha256'])
                if total == 1000:
                    need(descriptor['path'] == context['prediction_path'] and descriptor['state_path'] == context['checkpoint_path'],
                         'Baseline must cite original prediction/checkpoint')
                    continue
                state = load(state_path)
                current_steps = endpoint_check(state, source, train, context, total, names, origin_steps)
                final_step_values.update(current_steps.values())
                need(state['fingerprint'] == descriptor['endpoint_state_hash'] == prediction['endpoint_state_hash'] == row['state_hash'],
                     'Checkpoint/prediction/diagnostic state hashes disagree')
                need(prediction['state_path'] == descriptor['state_path'] and prediction['state_file_sha256'] == descriptor['state_file_sha256'],
                     'Prediction endpoint file linkage mismatch')
                need(prediction['candidate_ids'] == row['query_ids'] and prediction['query_labels_or_SE_present'] is False,
                     'Prediction query identity/boundary differs')
                for key, diagnostic in [('mean_standardized', 'query_mean_standardized'), ('mean_raw', 'query_mean_raw'),
                                        ('latent_variance_standardized', 'query_latent_variance_standardized'),
                                        ('latent_variance_raw', 'query_latent_variance_raw')]:
                    need(prediction[key] == row[diagnostic], 'Endpoint prediction differs from isolated diagnostic')
                need(prediction['y_center'] == train['scaler']['y_mean'] and prediction['y_scale'] == train['scaler']['y_scale'],
                     'Endpoint scaler changed')
                endpoint_count += 1
                new_prediction_points += len(prediction['candidate_ids'])
                if total == 4000:
                    latest = load(root / 'latest' / (cid + '.pt'))
                    endpoint_check(latest, source, train, context, 4000, names, origin_steps)
                    for key in ['mapper', 'gp', 'optimizer', 'rng', 'scheduler', 'ordered_parameter_names']:
                        need(exact(latest[key], state[key]), 'Latest state differs from fixed4000 endpoint: ' + key)
        trajectory = read(root / 'trajectories' / (cid + '.json'))
        need(trajectory['complete'] is True and trajectory['total_joint_step'] == 4000 and trajectory['cost']['updates'] == 3000,
             'Trajectory completion/update ledger mismatch')
        need(len(trajectory['endpoint_files']) == 2
             and {p['total_joint_step'] for p in trajectory['endpoint_files']} == {2000, 4000}
             and len(trajectory['diagnostic_files']) == 31,
             'Trajectory endpoint/journal count mismatch')
        for entry in trajectory['endpoint_files']:
            need(entry == endpoints[(cid, entry['total_joint_step'])], 'Trajectory endpoint differs from seal')
        for entry in trajectory['diagnostic_files']:
            need(entry == records[(cid, entry['total_joint_step'])], 'Trajectory diagnostic differs from journal')
        for key in trajectory_cost:
            trajectory_cost[key] += trajectory['cost'][key]
        examples.append({'context_id': cid, 'n': context['n'], 'source_actual_adam_steps': sorted(set(origin_steps.values())),
                         'source_state_hash': source['fingerprint'], 'new_formal_updates': 3000})
    need(baseline_count == 60 and endpoint_count == 120 and window_count == 1800, 'Final audited counts differ')
    ledger = read(root / 'resource_ledger.json')
    preflight = read(root / 'verification/preflight_summary.json')
    need(preflight['pass'] is True, 'Preflight did not pass')
    need(ledger['updates'] == trajectory_cost['updates'] == 180000, 'Formal updates differ from trajectory costs')
    need(ledger['verification_updates'] == preflight['synthetic_optimizer_updates'], 'Verification/formal updates mixed')
    need(ledger['failed_training_attempts'] == 0, 'This COMPLETE audit does not hide failed/replayed optimization attempts')
    need(ledger['seconds'] <= ledger['limit'] == 1200, 'Resource budget exceeded')
    need(math.isclose(ledger['seconds'], sum(ledger[k] for k in ['training_seconds', 'prediction_seconds', 'verification_seconds']),
                      rel_tol=1e-12, abs_tol=1e-9), 'Budget category total differs')
    for key in ['training_seconds', 'prediction_seconds', 'verification_seconds']:
        expected = trajectory_cost[key] + (preflight['verification_seconds'] if key == 'verification_seconds' else 0.)
        need(math.isclose(ledger[key], expected, rel_tol=1e-12, abs_tol=1e-8), 'Trajectory/global cost mismatch: ' + key)
        need(ledger[key] == seal['resources'][key], 'Sealed/final resource ledger differs: ' + key)
    return {'pass': True, 'scope': 'saved-state/journal/resource audit only; no model construction, fitting, prediction, or scoring labels',
            'verified_frozen_source_code_hashes': checked_hashes, 'baseline_pass_count': baseline_count,
            'new_checkpoint_count': endpoint_count, 'scientific_endpoint_count': len(files),
            'diagnostic_record_count': len(records), 'consecutive100_step_window_count': window_count,
            'new_formal_optimizer_updates': 180000, 'verification_optimizer_updates': ledger['verification_updates'],
            'source_actual_adam_step_values': sorted(origin_step_values),
            'new_endpoint_actual_adam_step_values': sorted(final_step_values),
            'all_actual_lrs_preserved': True, 'all_source_schedulers_unchanged': True,
            'all_ordered_parameter_names_and_moments_valid': True, 'all_step_increments_match': True,
            'all_checkpoint_math_hashes_recomputed': True, 'all_selected_tensors_scalers_heads_unchanged': True,
            'all_latest_states_equal4000_endpoints': True, 'all_endpoint_and_journal_hashes_valid': True,
            'baseline_prediction_max_abs_errors': maximum_baseline_error,
            'new_endpoint_prediction_count': new_prediction_points,
            'minimum_diagnostic_raw_standardized_latent_variance': minimum_latent_variance,
            'resources': ledger, 'trajectory_cost_sums': trajectory_cost,
            'no_scoring_labels_opened': True, 'no_training_or_predictions_performed': True,
            'contexts': examples, 'audit_wall_seconds': time.monotonic() - started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        result = audit(root)
    except Exception as error:
        result = {'pass': False, 'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                  'no_scoring_labels_opened': True, 'no_training_or_predictions_performed': True}
    output = root / 'audit/final_matrix_review.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'contexts'}, ensure_ascii=False))
    raise SystemExit(0 if result['pass'] else 1)
