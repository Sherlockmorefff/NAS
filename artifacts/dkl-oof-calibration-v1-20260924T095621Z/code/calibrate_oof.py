"""Sealed full-pipeline OOF moment calibration; no models or outer Q labels.

The only observations opened here belong to the current authorized T50 snapshot,
and only after all 90 pipelines and 4500 held-out predictions have been sealed.
"""
import argparse
import copy
import json
import math
import tempfile
import time
from pathlib import Path
from oof_data import read, write, sha, digest, now, require, child, checked_source

METHODS = ('HEAD', 'GP_CONST', 'GP_TREND')
OUTER_METHODS = ('HEAD', 'CONST_RAW', 'CONST_CAL', 'TREND_RAW', 'TREND_CAL')


def finite(values, name):
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in values), 'Nonfinite ' + name)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)


def verify_files(root, files):
    for path, value in files.items():
        require(sha(child(root, path)) == value, 'Sealed file changed: ' + path)


def seal_gate(root):
    """This function does not open any training labels or scoring directory."""
    root = Path(root)
    seal = read(root / 'oof_manifest.json')
    require(seal.get('sealed') is True and seal.get('execution_status') == 'COMPLETE_VALID', 'Complete uniform OOF seal required before calibration')
    require(seal['completed_pipelines'] == 90 and seal['formal_updates'] == 387000 and seal['oof_rows'] == 4500,
            'OOF matrix/count is incomplete')
    for name in ('protocol', 'worklist', 'source_manifest', 'code_manifest'):
        require(seal[name + '_sha256'] == sha(root / (name + '.json')), 'OOF seal binding differs: ' + name)
    for name in ('source_manifest', 'code_manifest', 'frozen_manifest'):
        verify_files(root, read(root / (name + '.json'))['files'])
    verify_files(root, seal['files'])
    work = read(root / 'worklist.json')
    require(len(work['pipelines']) == 90 and len(work['outer_contexts']) == 30, 'Frozen worklist count differs')
    expected = {'predictions_inner/' + p['pipeline_id'] + '_' + m + '.json' for p in work['pipelines'] for m in METHODS}
    actual = {path for path in seal['files'] if path.startswith('predictions_inner/') and path.endswith('.json')}
    require(actual == expected, 'OOF seal must contain exactly the 270 prescribed prediction sets')
    budget = read(root / 'budget.json')
    require(budget['formal_updates'] == budget['formal_optimizer_calls'] == 387000, 'Successful and called formal updates must both be387000')
    require(budget['limit'] == read(root / 'protocol.json')['budget']['model_seconds'] == 1500
            and budget['seconds'] <= 1500 and budget['synthetic_optimizer_calls'] <= 300, 'OOF budget invalid')
    require(all(budget[k] == 0 for k in ('failed_optimizer_calls', 'failed_updates', 'discarded_updates')), 'Failed/discarded formal execution is not a complete matrix')
    predictions, counts = {}, {}
    for pipeline in work['pipelines']:
        split = read(root / pipeline['split_path'])
        require(sha(root / pipeline['split_path']) == pipeline['split_sha256'], 'Split changed')
        fold = split['folds'][pipeline['inner_fold']]
        for method in METHODS:
            path = 'predictions_inner/' + pipeline['pipeline_id'] + '_' + method + '.json'
            p = read(root / path)
            require(p['method'] == method and p['pipeline_id'] == pipeline['pipeline_id'], 'Inner prediction identity differs')
            require(all(p[k] == pipeline[k] for k in ('context_id', 'subset_id', 'fold', 'sample_seed', 'model_seed',
                    'inner_fold', 'n_outer', 'n_fit', 'n_holdout')), 'Inner context metadata differs')
            require(p['candidate_ids'] == fold['V_ids'] and p['architecture_fingerprints'] == fold['V_architectures'], 'OOF V order/architecture differs')
            require(p['scaler']['fit_ids'] == fold['U_ids'], 'Prediction scaler was not fitted on U')
            require(len(p['mean_raw']) == fold['n_holdout'], 'OOF mean shape differs')
            finite(p['mean_raw'], 'OOF means')
            require(not any(k in p for k in ('y', 'query_y', 'query_se', 'measurement_se', 'truth_raw', 'validation_mean', 'validation_se')), 'OOF predictions contain held-out truth')
            if method == 'HEAD':
                require(p['train_candidate_ids'] == fold['U_ids'] and len(p['train_mean_raw']) == fold['n_fit'], 'Same-model HEAD U predictions missing')
                finite(p['train_mean_raw'], 'HEAD U means')
                require(p.get('model_variance_raw') is None, 'HEAD must not have fabricated uncertainty')
            else:
                validate_model_variance(p)
            key = (pipeline['context_id'], method)
            counts.setdefault(key, []).extend(p['candidate_ids'])
            predictions[(pipeline['pipeline_id'], method)] = p
    require(len(counts) == 90 and sum(len(v) for v in counts.values()) == 4500,
            'Merged OOF row total differs')
    for c in work['outer_contexts']:
        split = read(root / 'inner_splits' / (c['subset_id'] + '.json'))
        for method in METHODS:
            ids = counts[(c['context_id'], method)]
            require(len(ids) == len(set(ids)) == 50 and set(ids) == set(split['training_ids']), 'Candidate must be held out exactly once per method/context')
    return seal, work, predictions


def validate_model_variance(p):
    latent = p['latent_variance_standardized']
    variance = p['model_variance_raw']
    extra, scale = p['extra_noise_variance_standardized'], p['y_scale']
    finite(list(latent) + list(variance) + [extra, scale], 'GP variance/scaler')
    require(len(latent) == len(variance) == len(p['candidate_ids']) and scale >= 1e-8 and extra >= 1e-6,
            'GP variance shape/lower bound')
    require(all(v >= -1e-8 for v in latent), 'Negative latent variance exceeds original tolerance')
    expected = [scale * scale * (max(v, 0.) + extra) for v in latent]
    require(all(v >= 0 and close(v, e) for v, e in zip(variance, expected)), 'Extra noise must be included exactly once; heldout SE must be absent')
    if 'latent_variance_raw' in p:
        require(all(close(v, scale * scale * w) for v, w in zip(p['latent_variance_raw'], latent)), 'Unclamped latent raw unit differs')
    cov = p['latent_covariance_standardized']
    require(len(cov) == len(latent) and all(len(row) == len(latent) for row in cov), 'Full latent covariance missing')
    finite([v for row in cov for v in row], 'latent covariance')
    require(all(close(cov[i][i], latent[i]) for i in range(len(latent))), 'Latent diagonal differs from full covariance')
    return variance


def estimate_tau(rows):
    """Pure fixed moment estimator. Real use has exactly50 equally weighted rows."""
    require(bool(rows) and len({r['candidate_id'] for r in rows}) == len(rows), 'Tau input must have distinct candidates')
    n = len(rows)
    contributions = []
    for row in rows:
        y, mu, variance, se = (row[k] for k in ('y', 'mean_raw', 'model_variance_raw', 'measurement_se'))
        finite([y, mu, variance, se], 'tau input')
        require(variance >= 0 and se >= 0, 'Negative noise/variance in tau input')
        error = y - mu
        error_squared, measurement_variance = error * error, se * se
        value = error_squared - variance - measurement_variance
        contributions.append({'candidate_id': row['candidate_id'], 'architecture_fingerprint': row['architecture_fingerprint'],
            'truth_raw': y, 'mean_raw': mu, 'error_raw': error, 'squared_error_raw': error_squared,
            'squared_error_pp2': 10000 * error_squared, 'model_variance_raw': variance,
            'measurement_se_raw': se, 'measurement_variance_raw': measurement_variance,
            'unscaled_moment_raw': value, 'contribution_to_unclipped_tau_squared_raw': value / n})
    raw = math.fsum(r['unscaled_moment_raw'] for r in contributions) / n
    finite([raw], 'unclipped tau')
    tau2 = max(0., raw)
    error_sum = math.fsum(r['squared_error_raw'] for r in contributions)
    grouped = {}
    for row in contributions:
        grouped.setdefault(row['architecture_fingerprint'], []).append(row)
        row['error_squared_share'] = row['squared_error_raw'] / error_sum if error_sum > 0 else 0.
    sensitivity = []
    for arch in sorted(grouped):
        selected = grouped[arch]
        retained = [r for r in contributions if r['architecture_fingerprint'] != arch]
        leave_raw = math.fsum(r['unscaled_moment_raw'] for r in retained) / len(retained) if retained else None
        arch_error = math.fsum(r['squared_error_raw'] for r in selected)
        sensitivity.append({'architecture_fingerprint': arch, 'removed_candidate_ids': [r['candidate_id'] for r in selected],
            'removed_count': len(selected), 'remaining_count': len(retained), 'squared_error_sum_raw': arch_error,
            'squared_error_share': arch_error / error_sum if error_sum > 0 else 0.,
            'leave_one_architecture_out_unclipped_tau_squared_raw': leave_raw,
            'leave_one_architecture_out_tau_squared_raw': None if leave_raw is None else max(0., leave_raw),
            'descriptive_no_refitting': True, 'used_to_change_estimator': False})
    largest_candidate = min(contributions, key=lambda r: (-r['squared_error_raw'], r['candidate_id']))
    largest_arch = min(sensitivity, key=lambda r: (-r['squared_error_sum_raw'], r['architecture_fingerprint']))
    result = {'n_oof': n, 'tau_squared_raw': tau2, 'unclipped_tau_squared_raw': raw,
        'tau_sd_pp': 100 * math.sqrt(tau2), 'clipped_to_zero': raw < 0, 'candidate_weight': 1 / n,
        'mean_squared_error_raw': error_sum / n,
        'mean_model_variance_raw': math.fsum(r['model_variance_raw'] for r in contributions) / n,
        'mean_measurement_variance_raw': math.fsum(r['measurement_variance_raw'] for r in contributions) / n,
        'maximum_single_candidate_error_contributor': {'candidate_id': largest_candidate['candidate_id'],
            'squared_error_raw': largest_candidate['squared_error_raw'], 'share': largest_candidate['error_squared_share']},
        'maximum_single_architecture_error_contributor': {'architecture_fingerprint': largest_arch['architecture_fingerprint'],
            'squared_error_sum_raw': largest_arch['squared_error_sum_raw'], 'share': largest_arch['squared_error_share']},
        'estimator': 'max(0,mean((y-mu_oof)^2-model_variance_raw-SE^2)); no tuning or refitting',
        'variance_unit': 'raw proportion squared', 'sample_size_transfer_adjustment': None}
    return result, contributions, sensitivity


def parent_variance(p):
    if p['method'] == 'HEAD':
        return None
    latent, extra, scale = p['latent_variance_standardized'], p['extra_noise_variance_standardized'], p['y_scale']
    finite(list(latent) + [extra, scale], 'parent variance')
    require(extra >= 1e-6 and scale >= 1e-8 and all(v >= -1e-8 for v in latent), 'Parent variance lower bound')
    # Parent stores latent and extra separately; extra is a variance, not SD.
    return [scale * scale * (max(v, 0.) + extra) for v in latent]


def outer_wrapper(parent, parent_path, parent_hash, method, tau2, protocol_hash, source_hash):
    require(parent['n'] == 50 and parent['method'] in METHODS, 'Only prescribed n50 parent methods allowed')
    expected_parent = {'HEAD': 'HEAD', 'CONST_RAW': 'GP_CONST', 'CONST_CAL': 'GP_CONST',
                       'TREND_RAW': 'GP_TREND', 'TREND_CAL': 'GP_TREND'}[method]
    require(parent['method'] == expected_parent, 'Outer method mapped to wrong parent')
    require(not any(k in parent for k in ('truth_raw', 'y', 'query_y', 'measurement_se', 'validation_mean', 'validation_se')),
            'Parent prediction must be unlabelled')
    finite(parent['mean_raw'], 'parent means')
    require(len(parent['mean_raw']) == len(parent['candidate_ids']), 'Parent mean shape')
    raw = parent_variance(parent)
    require(tau2 is None if method == 'HEAD' else isinstance(tau2, (int, float)) and math.isfinite(tau2) and tau2 >= 0,
            'Invalid outer tau')
    p = copy.deepcopy(parent)
    p.update(method=method, unit_id=parent['context_id'] + '_' + method, n_outer=50,
        output_kind='variance_derivative' if method.endswith('_CAL') else 'parent_reference', new_model_fit=False,
        parent_method=parent['method'], parent_prediction_path=parent_path, parent_prediction_sha256=parent_hash,
        parent_protocol_sha256=parent['protocol_sha256'], parent_source_manifest_sha256=parent['source_manifest_sha256'],
        protocol_sha256=protocol_hash, source_manifest_sha256=source_hash,
        model_variance_raw=None if raw is None else [v + tau2 for v in raw],
        uncalibrated_model_variance_raw=raw, tau_squared_raw=tau2,
        mean_array_sha256=digest(parent['mean_raw']), parent_mean_array_sha256=digest(parent['mean_raw']),
        model_variance_definition='latent+GP_extra+tau_if_CAL; excludes outer query SE and jitter',
        latent_covariance_unchanged=True, joint_new_error_covariance_assumed=False,
        ranking_ids=[parent['candidate_ids'][i] for i in sorted(range(len(parent['candidate_ids'])),
                     key=lambda i: (-parent['mean_raw'][i], parent['candidate_ids'][i]))])
    require(p['mean_raw'] == parent['mean_raw'] and digest(p['mean_raw']) == p['mean_array_sha256'], 'Calibrator changed parent means')
    if 'latent_covariance_standardized' in parent:
        require(p['latent_covariance_standardized'] == parent['latent_covariance_standardized'], 'Calibrator changed latent covariance')
    return p


def calibrate(root):
    root = Path(root); start = time.monotonic()
    require(not (root / 'outer_predictions_manifest.json').exists() and not (root / 'calibration/tau.json').exists(), 'Calibration outputs already exist; no automatic overwrite')
    seal, work, predictions = seal_gate(root)  # ALL verification before observation access.
    (root / 'calibration_once.lock').mkdir()
    write(root / 'calibration_once.lock/entry.json', {'created_at_utc': now(), 'permanent': True,
        'automatic_resume': False, 'oof_manifest_sha256': sha(root / 'oof_manifest.json'),
        'protocol_sha256': sha(root / 'protocol.json'), 'model_calls': 0,
        'purpose': 'single post-OOF calibration and outer prediction seal; no outer Q observations'})
    write(root / 'calibration/inner_release.json', {'released_at_utc': now(), 'oof_manifest_sha256': sha(root / 'oof_manifest.json'),
        'completed_pipelines': 90, 'formal_updates': 387000, 'oof_rows': 4500,
        'access_scope': 'one current T50 snapshot at a time; U/V only; no outer Q labels or SE',
        'all_oof_hashes_verified_before_label_access': True, 'training_may_not_resume': True})
    taus, contributions, sensitivity = [], [], []
    for c in work['outer_contexts']:
        # Never aggregate labels from other contexts into a reusable lookup.
        t = read(checked_source(root, 'training/' + c['subset_id'] + '.json'))
        require(len(t['ids']) == len(set(t['ids'])) == 50 and t['metadata']['subset_id'] == c['subset_id'], 'Current T50 identity')
        current = {cid: {'y': t['y'][i], 'measurement_se': t['measurement_se'][i],
                         'architecture_fingerprint': t['metadata']['candidate_metadata'][i]['architecture_fingerprint']}
                   for i, cid in enumerate(t['ids'])}
        for method in ('GP_CONST', 'GP_TREND'):
            rows = []
            for fold in range(3):
                p = predictions[(c['context_id'] + '_i' + str(fold), method)]
                for i, cid in enumerate(p['candidate_ids']):
                    require(current[cid]['architecture_fingerprint'] == p['architecture_fingerprints'][i], 'OOF architecture changed')
                    rows.append({'candidate_id': cid, **current[cid], 'mean_raw': p['mean_raw'][i],
                                 'model_variance_raw': p['model_variance_raw'][i], 'inner_fold': fold})
            require(len(rows) == 50 and {r['candidate_id'] for r in rows} == set(t['ids']), 'Tau needs exact50 current-context OOF rows')
            index = {r['candidate_id']: r for r in rows}
            rows = [index[cid] for cid in t['ids']]
            result, cs, ss = estimate_tau(rows)
            metadata = {k: c[k] for k in ('context_id', 'subset_id', 'fold', 'sample_seed', 'model_seed')}
            metadata.update(method=method, n_outer=50)
            taus.append({**metadata, **result, 'oof_candidate_ids_sha256': digest(t['ids'])})
            contributions.extend({**metadata, 'inner_fold': index[row['candidate_id']]['inner_fold'], **row} for row in cs)
            sensitivity.extend({**metadata, **row} for row in ss)
        del current, t
    require(len(taus) == 60 and len(contributions) == 3000, 'Tau/contribution totals differ')
    write(root / 'calibration/tau.json', {'rows': taus, 'tau_count': 60, 'oof_manifest_sha256': sha(root / 'oof_manifest.json'),
        'protocol_sha256': sha(root / 'protocol.json'), 'source_manifest_sha256': sha(root / 'source_manifest.json'),
        'outer_query_labels_or_SE_accessed': False, 'estimation_finished_at_utc': now()})
    with (root / 'calibration/contributions.jsonl').open('x') as f:
        for row in contributions:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    write(root / 'calibration/architecture_sensitivity.json', {'rows': sensitivity,
        'interpretation': 'Descriptive deletion from fixed OOF table; no model refit, no estimator change; not strict leave-architecture validation'})
    tau_map = {(r['context_id'], r['method']): r['tau_squared_raw'] for r in taus}
    files, records = {}, []
    for c in work['outer_contexts']:
        refs = {m: 'source/parent_predictions/' + c['context_id'] + '_' + m + '.json' for m in METHODS}
        parents = {m: read(checked_source(root, rel)) for m, rel in refs.items()}
        q = read(checked_source(root, 'queries/fold' + str(c['fold']) + '.json'))
        for method in OUTER_METHODS:
            original = 'HEAD' if method == 'HEAD' else 'GP_CONST' if method.startswith('CONST') else 'GP_TREND'
            parent = parents[original]
            require(all(parent[k] == c[k] for k in ('context_id', 'subset_id', 'fold', 'sample_seed', 'model_seed')), 'Parent context differs')
            require(parent['candidate_ids'] == q['ids'] and parent['architecture_fingerprints'] == q['architectures'], 'Outer Q identity/order changed')
            tau2 = None if method == 'HEAD' else tau_map[(c['context_id'], original)] if method.endswith('_CAL') else 0.
            p = outer_wrapper(parent, refs[original], sha(root / refs[original]), method, tau2,
                              sha(root / 'protocol.json'), sha(root / 'source_manifest.json'))
            p['tau_file_sha256'] = sha(root / 'calibration/tau.json')
            rel = 'outer_predictions/' + p['unit_id'] + '.json'
            write(root / rel, p); files[rel] = sha(root / rel)
            records.append({'path': rel, 'sha256': files[rel], 'unit_id': p['unit_id'], 'context_id': c['context_id'],
                'method': method, 'output_kind': p['output_kind'], 'parent_prediction_path': refs[original],
                'parent_prediction_sha256': p['parent_prediction_sha256'], 'mean_array_sha256': p['mean_array_sha256']})
    require(len(files) == len(records) == 150, 'Outer endpoint count differs')
    support = {rel: sha(root / rel) for rel in ('calibration/tau.json', 'calibration/contributions.jsonl',
                'calibration/architecture_sensitivity.json', 'calibration/inner_release.json')}
    manifest = {'sealed': True, 'sealed_at_utc': now(), 'execution_status': 'COMPLETE_VALID',
        'expected_endpoint_count': 150, 'completed_endpoints': 150, 'tau_count': 60,
        'parent_reference_count': 90, 'variance_derivative_count': 60, 'new_outer_model_fits': 0,
        'completed_pipelines': 90, 'formal_updates': 387000, 'oof_rows': 4500,
        'files': files, 'records': records, 'supporting_files': support,
        'tau_sha256': support['calibration/tau.json'], 'oof_manifest_sha256': sha(root / 'oof_manifest.json'),
        **{name + '_sha256': sha(root / (name + '.json')) for name in ('protocol', 'source_manifest', 'code_manifest', 'worklist')},
        'outer_query_labels_or_SE_accessed': False, 'mean_and_ranking_arrays_unchanged': True,
        'model_calls': 0, 'optimizer_calls': 0, 'pure_statistics_wall_seconds': time.monotonic() - start,
        'next_stage_authorized': False}
    write(root / 'outer_predictions_manifest.json', manifest)
    return {'status': 'CALIBRATION_AND_OUTER_SEAL_COMPLETE', 'tau_count': 60, 'outer_endpoints': 150,
            'model_calls': 0, 'wall_seconds': manifest['pure_statistics_wall_seconds']}


def run_checks():
    def rows(y, mu, var, se):
        return [{'candidate_id': str(i), 'architecture_fingerprint': 'a' + str(i), 'y': a,
                 'mean_raw': b, 'model_variance_raw': v, 'measurement_se': s}
                for i, (a, b, v, s) in enumerate(zip(y, mu, var, se))]
    cases = {}
    positive = rows([.4, .2], [.2, .1], [.01, .002], [.1, .02])
    r, c, s = estimate_tau(positive)
    expected = ((.2 ** 2 - .01 - .1 ** 2) + (.1 ** 2 - .002 - .02 ** 2)) / 2
    cases['manual_positive_and_SE_once'] = close(r['tau_squared_raw'], expected)
    cases['contribution_sum_exact_estimator'] = close(math.fsum(x['contribution_to_unclipped_tau_squared_raw'] for x in c), expected)
    zero, _, _ = estimate_tau(rows([.1, .2], [.1, .2], [.01, .01], [.1, .1]))
    cases['negative_estimate_clipped_but_preserved'] = zero['tau_squared_raw'] == 0 and zero['unclipped_tau_squared_raw'] < 0
    boundary, _, _ = estimate_tau(rows([1.], [0.], [.75], [.5]))
    epsilon, _, _ = estimate_tau(rows([1.], [0.], [.75 - 1e-10], [.5]))
    cases['zero_and_strictly_positive_boundary_no_extra_threshold'] = boundary['tau_squared_raw'] == 0 and 0 < epsilon['tau_squared_raw'] < 2e-10
    scaled, _, _ = estimate_tau(rows([40., 20.], [20., 10.], [100., 20.], [10., 2.]))
    cases['raw_variance_to_pp2_scaling'] = close(scaled['tau_squared_raw'], 10000 * expected)
    parent = {'n': 50, 'method': 'GP_CONST', 'context_id': 'synthetic', 'candidate_ids': ['b', 'a'],
        'mean_raw': [.2, .2], 'latent_variance_standardized': [.1, .2], 'latent_covariance_standardized': [[.1, 0.], [0., .2]],
        'extra_noise_variance_standardized': .01, 'y_scale': .2, 'protocol_sha256': 'old', 'source_manifest_sha256': 'oldsrc'}
    raw = outer_wrapper(parent, 'source/p.json', 'parenthash', 'CONST_RAW', 0., 'protocol', 'source')
    cal0 = outer_wrapper(parent, 'source/p.json', 'parenthash', 'CONST_CAL', 0., 'protocol', 'source')
    cal = outer_wrapper(parent, 'source/p.json', 'parenthash', 'CONST_CAL', expected, 'protocol', 'source')
    cases['tau0_recovers_raw_variance'] = raw['model_variance_raw'] == cal0['model_variance_raw']
    cases['mean_exact_tie_order_invariant'] = raw['mean_raw'] == cal['mean_raw'] and raw['ranking_ids'] == cal['ranking_ids'] == ['a', 'b']
    cases['covariance_unchanged_only_marginal_added'] = cal['latent_covariance_standardized'] == parent['latent_covariance_standardized'] and all(close(a, b + expected) for a, b in zip(cal['model_variance_raw'], raw['model_variance_raw']))
    try:
        estimate_tau(rows([float('nan')], [0.], [0.], [0.]))
        rejected = False
    except ValueError:
        rejected = True
    cases['nonfinite_rejected'] = rejected
    for name, payload in [('unsealed_rejected', {'sealed': False}),
                          ('partial_seal_rejected', {'sealed': True, 'execution_status': 'STOPPED_BUDGET_PARTIAL'})]:
        with tempfile.TemporaryDirectory(prefix='oof_pure_gate_') as tmp:
            write(Path(tmp) / 'oof_manifest.json', payload)
            try:
                seal_gate(Path(tmp))
                rejected = False
            except ValueError:
                rejected = True
            cases[name] = rejected
    require(all(cases.values()), 'Pure tau/unit/isolation checks failed')
    return {'pass': True, 'checks': cases, 'model_calls': 0, 'optimizer_calls': 0, 'real_labels_accessed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path); p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    require(args.check_only or args.root is not None, '--root required')
    print(json.dumps(run_checks() if args.check_only else calibrate(args.root), ensure_ascii=False))
