"""Frozen raw-input data boundary for three GP distance models. Stdlib only.

Real numerical preparation must be charged by a provided budget or an explicit
caller timing scope. Every fit receives only its selected T observations and
public Q inputs. No historical checkpoint or scoring observation is loaded.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import time
from pathlib import Path

METHODS = ('RAW16_ISO', 'HP4_ISO', 'RAW16_BLOCK2')
RAW_QUERY_KEYS = {'ids', 'architectures', 'x_raw'}
QUERY_KEYS = {'ids', 'architectures', 'x_standardized'}
TRAIN_KEYS = QUERY_KEYS | {'y_standardized', 'known_variance_standardized'}
SCALER_KEYS = {'x_mean', 'x_scale', 'y_mean', 'y_scale', 'fit_ids', 'std_correction', 'frozen', 'dtype'}
QUERY_COUNTS = (38, 36, 36, 34, 35)
ATOL, RTOL = 1e-12, 1e-10


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def child(root, relative):
    p = (Path(root) / relative).resolve()
    require(p.is_relative_to(Path(root).resolve()), 'Path escapes run root')
    return p


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= ATOL + RTOL * abs(b)


def vector_close(a, b):
    return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))


def checked_source(root, relative, source=None):
    source = source or read(Path(root) / 'source_manifest.json')
    require(relative in source['files'], 'Unpinned source path: ' + relative)
    path = child(root, relative)
    require(sha(path) == source['files'][relative], 'BLOCKED_SOURCE_MISMATCH: ' + relative)
    return path


def _public(doc, count, x_key):
    require(set(doc) == {'ids', 'architectures', x_key}, 'Q exact whitelist violation')
    ids, arches, values = doc['ids'], doc['architectures'], doc[x_key]
    require(len(ids) == count and len(set(ids)) == count and len(arches) == len(values) == count, 'Public ID/shape mismatch')
    require(all(isinstance(v, str) and v for v in ids + arches), 'Invalid ID/architecture')
    require(all(len(row) == 16 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in row)
                for row in values), 'Public inputs must be finite canonical16')
    return copy.deepcopy(doc)


def query_projection(query):
    require(len(query.get('ids', [])) in QUERY_COUNTS, 'Unexpected outer query size')
    return _public(query, len(query['ids']), 'x_standardized')


def training_projection(training):
    require(set(training) == TRAIN_KEYS, 'T exact whitelist violation')
    result = _public({key: training[key] for key in ('ids', 'architectures', 'x_standardized')}, 50, 'x_standardized')
    for key in ('y_standardized', 'known_variance_standardized'):
        values = training[key]
        require(len(values) == 50 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values), 'Invalid selected ' + key)
        if key == 'known_variance_standardized':
            require(all(v >= 0 for v in values), 'Negative selected known variance')
        result[key] = list(values)
    return result


def _mean_scale(values, floor):
    mean = math.fsum(values) / len(values)
    scale = max(floor, math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values)))
    return mean, scale


def standardized_projection(training_snapshot, raw_query, public, subset):
    """Only current T50 fits a scaler; Q contains no observations or statistics."""
    t = training_snapshot
    ids = list(t['ids']); meta = t['metadata']; sid = subset['subset_id']; fold = subset['fold']
    require(ids == subset['ids'] and len(ids) == len(set(ids)) == 50, 'Current T50 membership/order mismatch')
    require(meta['subset_id'] == sid and meta['fold'] == fold and meta['sample_seed'] == subset['sample_seed']
            and meta['n'] == 50 and meta['query_labels_released'] == meta['query_SE_released'] == 0, 'Source selected context mismatch')
    candidates = meta['candidate_metadata']
    require([c['candidate_id'] for c in candidates] == ids, 'Training metadata row order mismatch')
    arches = [c['architecture_fingerprint'] for c in candidates]
    rawt = _public({'ids': ids, 'architectures': arches, 'x_raw': t['x_raw']}, 50, 'x_raw')
    rawq = _public(raw_query, QUERY_COUNTS[fold], 'x_raw')
    require(not set(ids) & set(rawq['ids']) and not set(arches) & set(rawq['architectures']), 'T/Q candidate or architecture overlap')
    require(public['label_values_decoded'] == public['SE_values_decoded'] == 0, 'Public source unexpectedly contains observations')
    pubmap = {cid: i for i, cid in enumerate(public['ids'])}
    require(len(pubmap) == len(public['ids']) == 179, 'Public source identity size')
    for raw in (rawt, rawq):
        for cid, arch, row in zip(raw['ids'], raw['architectures'], raw['x_raw']):
            i = pubmap[cid]; pm = public['candidate_metadata'][cid]
            require(pm['candidate_id'] == cid and pm['architecture_fingerprint'] == arch == public['architectures'][i], 'Candidate/architecture metadata mismatch')
            require(len(public['z'][i]) == 12 and len(public['hp'][i]) == 4 and row == public['z'][i] + public['hp'][i], 'x_raw must be original z12 plus transformed real HP4')
            hp = pm['raw_hp']
            require(hp['lr'] > 0 and hp['hidden_dim'] > 0 and hp['l2'] > 0, 'Invalid original HP transform domain')
            expected = [math.log10(hp['lr']), hp['dropout'], math.log2(hp['hidden_dim']), math.log10(hp['l2'])]
            require(vector_close(row[12:], expected), 'Saved HP columns are not the declared once-transformed raw HP')
    require(all(c == public['candidate_metadata'][cid] for c, cid in zip(candidates, ids)), 'Selected metadata differs from public identity map')
    require(len(t['y']) == len(t['measurement_se']) == 50 and all(math.isfinite(v) for v in t['y'])
            and all(math.isfinite(v) and v >= 0 for v in t['measurement_se']), 'Selected label/SE alignment or finite domain')
    xms = [_mean_scale([row[j] for row in rawt['x_raw']], 1e-6) for j in range(16)]
    xm, xs = [v[0] for v in xms], [v[1] for v in xms]
    ym, ys = _mean_scale(t['y'], 1e-8)
    scaler = {'x_mean': xm, 'x_scale': xs, 'y_mean': ym, 'y_scale': ys, 'fit_ids': ids,
              'std_correction': 0, 'frozen': True, 'dtype': 'float64'}
    old = t['scaler']
    require(set(old) == SCALER_KEYS and old['fit_ids'] == ids and old['std_correction'] == 0
            and old['frozen'] is True and old['dtype'] == 'float64', 'Parent scaler fit scope changed')
    require(vector_close(xm, old['x_mean']) and vector_close(xs, old['x_scale'])
            and close(ym, old['y_mean']) and close(ys, old['y_scale']), 'T-only recomputed scaler disagrees with parent tolerance')
    tx = [[(v - xm[j]) / xs[j] for j, v in enumerate(row)] for row in rawt['x_raw']]
    qx = [[(v - xm[j]) / xs[j] for j, v in enumerate(row)] for row in rawq['x_raw']]
    y = [(v - ym) / ys for v in t['y']]
    known = [se * se / (ys * ys) for se in t['measurement_se']]
    require(all(vector_close(a, b) for a, b in zip(tx, t['x_standardized'])) and vector_close(y, t['y_standardized'])
            and vector_close(known, t['known_variance_standardized'])
            and vector_close([se * se for se in t['measurement_se']], t['known_variance']), 'Stored selected standardization/SE variance differs')
    training = training_projection({'ids': ids, 'architectures': arches, 'x_standardized': tx,
                                    'y_standardized': y, 'known_variance_standardized': known})
    query = query_projection({'ids': rawq['ids'], 'architectures': rawq['architectures'], 'x_standardized': qx})
    hp4 = {'input_columns': [12, 13, 14, 15], 'training_x_standardized': [row[12:] for row in tx],
           'query_x_standardized': [row[12:] for row in qx], 'additional_transform': None}
    identity = {'subset_id': sid, 'training_ids_sha256': digest(ids), 'training_architectures_sha256': digest(arches),
        'query_ids_sha256': digest(query['ids']), 'query_architectures_sha256': digest(query['architectures']),
        'training_raw16_sha256': digest(rawt['x_raw']), 'query_raw16_sha256': digest(rawq['x_raw']),
        'training_standardized16_sha256': digest(tx), 'query_standardized16_sha256': digest(qx),
        'training_hp4_sha256': digest(hp4['training_x_standardized']), 'query_hp4_sha256': digest(hp4['query_x_standardized']),
        'y_standardized_sha256': digest(y), 'known_variance_standardized_sha256': digest(known),
        'scaler_sha256': digest(scaler), 'parent_scaler_sha256': digest(old), 'query_labels_or_SE_present': False,
        'hp_transform_applied_to_raw_inputs': False, 'canonical_input_columns': 16, 'hp_columns': [12, 13, 14, 15]}
    return {'subset_id': sid, 'sample_seed': subset['sample_seed'], 'fold': fold, 'n_train': 50, 'n_query': len(query['ids']),
        'training': training, 'query': query, 'scaler': scaler, 'identity': identity,
        'raw_training': {**rawt, 'y': list(t['y']), 'measurement_se': list(t['measurement_se'])},
        'raw_query': rawq, 'hp4_projection': hp4,
        'scaler_parent_comparison': {'pass': True, 'atol': ATOL, 'rtol': RTOL, 'ddof': 0,
            'x_mean_max_abs': max(abs(a - b) for a, b in zip(xm, old['x_mean'])),
            'x_scale_max_abs': max(abs(a - b) for a, b in zip(xs, old['x_scale'])),
            'y_mean_abs': abs(ym - old['y_mean']), 'y_scale_abs': abs(ys - old['y_scale'])}}


def head_mask(prediction, context, source_path, source_hash):
    ids, means = prediction['candidate_ids'], prediction['mean_raw']
    require(prediction['method'] == 'HEAD' and prediction['n'] == 50 and len(ids) == len(means)
            and len(set(ids)) == len(ids) and all(math.isfinite(v) for v in means), 'Invalid unlabelled parent HEAD')
    ranking = sorted(range(len(ids)), key=lambda i: (-means[i], ids[i]))
    count = math.ceil(.2 * len(ids))
    return {**{k: context[k] for k in ('context_id', 'subset_id', 'model_seed', 'fold', 'sample_seed')},
        'candidate_ids': [ids[i] for i in ranking[:count]], 'query_ids': list(ids),
        'ranked_query_ids': [ids[i] for i in ranking], 'mask_size': count, 'fraction': .2,
        'ranking': 'exact mean descending then candidate_id ascending; no seed averaging',
        'source_prediction_path': source_path, 'source_prediction_sha256': source_hash,
        'mean_array_sha256': digest(means), 'query_labels_or_SE_used': False}


def _prepare(root):
    root = Path(root)
    require(not (root / 'data_manifest.json').exists() and not (root / 'worklist.json').exists(), 'Data preparation is one-shot')
    (root / 'data_prepare_once.lock').mkdir()
    write(root / 'data_prepare_once.lock/entry.json', {'created_at_utc': now(), 'permanent': True, 'automatic_resume': False})
    source = read(root / 'source_manifest.json')
    parent_work = read(checked_source(root, 'source/parent_worklist.json', source))
    contexts = parent_work['outer_contexts']
    require(len(contexts) == 30 and {(c['sample_seed'], c['fold'], c['model_seed']) for c in contexts}
            == {(s, f, m) for s in (20260922, 20260923) for f in range(5) for m in (0, 1, 2)}, 'Historical context matrix differs')
    subset_source = read(checked_source(root, 'subsets.json', source))['subsets']
    public = read(checked_source(root, 'public_inputs.json', source))
    all_files, mask_files, mask_records, refs, subset_rows, fits = {}, {}, [], [], [], []
    for index, (sample_seed, fold) in enumerate((s, f) for s in (20260922, 20260923) for f in range(5)):
        sid = 's' + str(sample_seed) + '_f' + str(fold) + '_n50'
        selected = subset_source[sid]
        tpath, qpath = 'training/' + sid + '.json', 'queries/fold' + str(fold) + '.json'
        snapshot = read(checked_source(root, tpath, source)); query = read(checked_source(root, qpath, source))
        data = standardized_projection(snapshot, query, public, selected)
        data['identity'].update(source_training_path=tpath, source_training_sha256=source['files'][tpath],
            source_query_path=qpath, source_query_sha256=source['files'][qpath],
            public_inputs_sha256=source['files']['public_inputs.json'], protocol_sha256=sha(root / 'protocol.json'),
            source_manifest_sha256=sha(root / 'source_manifest.json'))
        relative = 'data/' + sid + '.json'; write(root / relative, data); all_files[relative] = sha(root / relative)
        order = list(METHODS[index % 3:] + METHODS[:index % 3])
        row = {'subset_id': sid, 'sample_seed': sample_seed, 'fold': fold, 'n': 50, 'n_query': len(query['ids']),
               'subset_index': index, 'group_order': order, 'data_path': relative, 'data_sha256': all_files[relative]}
        subset_rows.append(row)
        for method in order:
            fits.append({**row, 'unit_id': sid + '_' + method, 'method': method, 'model_seed': None, 'rng_seed': 0,
                         'input_dimension': 4 if method == 'HP4_ISO' else 16})
        for c in sorted([v for v in contexts if v['subset_id'] == sid], key=lambda c: c['model_seed']):
            for parent_method, method in (('HEAD', 'HEAD'), ('GP_CONST', 'H_CONST'), ('GP_TREND', 'H_TREND')):
                path = 'source/parent_predictions/' + c['context_id'] + '_' + parent_method + '.json'
                p = read(checked_source(root, path, source))
                require(p['method'] == parent_method and p['n'] == 50 and p['candidate_ids'] == query['ids']
                    and p['architecture_fingerprints'] == query['architectures'] and p['training_ids_sha256'] == digest(snapshot['ids'])
                    and p['query_labels_or_SE_present'] is False, 'Historical T/Q identity mismatch')
                require(all(p[k] == c[k] for k in ('context_id', 'subset_id', 'fold', 'sample_seed', 'model_seed'))
                    and close(p['y_center'], data['scaler']['y_mean']) and close(p['y_scale'], data['scaler']['y_scale']), 'Historical context/scaler differs')
                require(isinstance(p['endpoint_state_hash'], str) and len(p['endpoint_state_hash']) == 64
                    and isinstance(p['state_file_sha256'], str) and len(p['state_file_sha256']) == 64, 'Historical state identity absent')
                refs.append({**{k: c[k] for k in ('context_id', 'subset_id', 'fold', 'sample_seed', 'model_seed')},
                    'unit_id': c['context_id'] + '_' + method, 'method': method, 'parent_method': parent_method,
                    'path': path, 'sha256': source['files'][path], 'is_historical': True, 'new_model_fit': False,
                    'training_ids_sha256': p['training_ids_sha256'], 'query_ids_sha256': digest(p['candidate_ids']),
                    'parent_endpoint_state_hash': p['endpoint_state_hash'], 'parent_state_file_sha256': p['state_file_sha256'],
                    'parent_protocol_sha256': p['protocol_sha256'], 'parent_y_center': p['y_center'], 'parent_y_scale': p['y_scale']})
                if parent_method == 'HEAD':
                    mask = head_mask(p, c, path, source['files'][path])
                    mpath = 'head_masks/' + c['context_id'] + '.json'; write(root / mpath, mask)
                    mask_files[mpath] = sha(root / mpath); all_files[mpath] = mask_files[mpath]
                    mask_records.append({'path': mpath, 'sha256': mask_files[mpath], **mask})
    require(len(fits) == 30 and len(refs) == 90 and len(mask_records) == 30 and len(subset_rows) == 10, 'Prepared matrix differs')
    write(root / 'historical_refs.json', {'records': refs, 'historical_reference_count': 90, 'new_model_fits': 0,
        'source_manifest_sha256': sha(root / 'source_manifest.json'), 'protocol_sha256': sha(root / 'protocol.json')})
    write(root / 'masks_manifest.json', {'files': mask_files, 'records': mask_records, 'mask_count': 30,
        'protocol_sha256': sha(root / 'protocol.json'), 'source_manifest_sha256': sha(root / 'source_manifest.json'),
        'frozen_before_new_model_training': True, 'query_labels_or_SE_used': False})
    work = {'subsets': subset_rows, 'fits': fits, 'subset_count': 10, 'new_fit_count': 30, 'new_prediction_rows': 1074,
        'historical_reference_count': 90, 'head_mask_count': 30, 'formal_updates': 54000,
        'formal_updates_by_stage': {'INIT_GP': 9000, 'FIT_GP': 45000},
        'methods': list(METHODS), 'subset_order': ['sample_seed', 'fold'],
        'method_order': '[A,B,C] rotated left by subset_index modulo3', 'new_model_seed': None, 'execution_rng_seed': 0}
    write(root / 'worklist.json', work)
    for path in ('historical_refs.json', 'masks_manifest.json', 'worklist.json'):
        all_files[path] = sha(root / path)
    write(root / 'data_manifest.json', {'files': all_files, 'data_projection_count': 10, 'mask_count': 30,
        'historical_reference_count': 90, 'new_fit_count': 30, 'frozen_at_utc': now(),
        **{k + '_sha256': sha(root / (k + '.json')) for k in ('protocol', 'source_manifest', 'worklist', 'masks_manifest', 'historical_refs')},
        'query_labels_or_SE_present': False, 'standardization': 'selected T50 only; ddof0; no HP retransform',
        'numeric_preparation_requires_budget_charge': True})
    return {'pass': True, 'data_projections': 10, 'masks': 30, 'historical_refs': 90, 'new_fits': 30, 'model_calls': 0}


def prepare(root, budget=None, *, charged_by_caller=False):
    require(budget is not None or charged_by_caller, 'Real numerical data preparation needs a budget or explicit caller timing')
    if budget is not None:
        budget.check()
    start = time.monotonic()
    try:
        result = _prepare(Path(root))
    finally:
        elapsed = time.monotonic() - start
        if budget is not None:
            # Conservative: includes accompanying hash/I/O as well as numerics.
            budget.charge(elapsed, 'verification'); budget.persist()
    if budget is not None:
        budget.check()
    result['preparation_wall_seconds'] = elapsed
    result['charge_scope'] = 'provided budget, conservative all elapsed' if budget is not None else 'external caller timing required'
    return result


def load_inputs(root, item):
    """Canonical16 for every method; core takes exact last4 for HP4_ISO."""
    root = Path(root); work = read(root / 'worklist.json')
    matches = [f for f in work['fits'] if f['unit_id'] == item['unit_id']]
    require(len(matches) == 1 and matches[0] == item, 'Unrecognized frozen fit')
    require(item['method'] in METHODS and item['model_seed'] is None and item['rng_seed'] == 0, 'Unprescribed model identity')
    manifest = read(root / 'data_manifest.json')
    require(manifest['protocol_sha256'] == sha(root / 'protocol.json')
            and manifest['worklist_sha256'] == sha(root / 'worklist.json'), 'Data protocol/worklist changed')
    path = child(root, item['data_path'])
    require(sha(path) == item['data_sha256'] == manifest['files'][item['data_path']], 'Frozen data projection changed')
    data = read(path)
    training, query = training_projection(data['training']), query_projection(data['query'])
    require(data['subset_id'] == item['subset_id'] and data['scaler']['fit_ids'] == training['ids']
        and not set(training['ids']) & set(query['ids']) and not set(training['architectures']) & set(query['architectures']), 'Fit/Q context or scaler leakage')
    require(digest(data['scaler']) == data['identity']['scaler_sha256'], 'Frozen scaler identity differs')
    return {'training': training, 'query': query, 'scaler': copy.deepcopy(data['scaler']), 'identity': copy.deepcopy(data['identity'])}


def run_checks():
    training = {'ids': ['t' + str(i) for i in range(50)], 'architectures': ['a' + str(i) for i in range(50)],
        'x_standardized': [[float(i + j) for j in range(16)] for i in range(50)],
        'y_standardized': [i / 50. for i in range(50)], 'known_variance_standardized': [.01] * 50}
    query = {'ids': ['q' + str(i) for i in range(38)], 'architectures': ['qa' + str(i) for i in range(38)],
             'x_standardized': [[float(i + j) for j in range(16)] for i in range(38)]}
    a, q = training_projection(training), query_projection(query)
    checks = {'canonical16_retained': a['x_standardized'] == training['x_standardized'],
              'HP4_exact_slice_no_transform': [r[12:] for r in a['x_standardized']][0] == [12., 13., 14., 15.]}
    for field in ('y', 'measurement_se', 'query_se', 'truth_raw'):
        try:
            query_projection({**query, field: [float('nan')] * 38})
            rejected = False
        except ValueError:
            rejected = True
        checks['reject_query_' + field] = rejected
    mean, scale = _mean_scale([1., 1., 1.], 1e-6)
    checks['ddof0_and_floor'] = mean == 1. and scale == 1e-6 and _mean_scale([0., 2.], 1e-8) == (1., 1.)
    fake = {'method': 'HEAD', 'n': 50, 'candidate_ids': ['b', 'a', 'c', 'd', 'e'], 'mean_raw': [1., 1., 0., 0., 0.]}
    context = {'context_id': 'c', 'subset_id': 's', 'model_seed': 2, 'fold': 0, 'sample_seed': 20260922}
    mask = head_mask(fake, context, 'source/head.json', 'sha')
    checks['mask_exact_tie_and_ceiling'] = mask['candidate_ids'] == ['a'] and mask['model_seed'] == 2
    require(all(checks.values()), 'Pure data interface checks failed')
    return {'pass': True, 'checks': checks, 'model_calls': 0, 'optimizer_calls': 0, 'real_query_labels_accessed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path); parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        result = run_checks()
    else:
        require(args.root is not None, '--root is required')
        from block_core import Budget
        limit = read(args.root / 'protocol.json')['budget']['model_seconds']
        require(limit == 600, 'Block experiment requires600-second budget')
        result = prepare(args.root, Budget(args.root / 'budget.json', limit=limit))
    print(json.dumps(result, ensure_ascii=False))
