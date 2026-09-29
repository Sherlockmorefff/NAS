"""Immutable parent numeric snapshots and narrow per-subset GP interfaces.

No global label object, scoring table, case report or checkpoint is loaded.
All real numeric preparation is charged to the new model budget by the caller.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS = ('A_REFIT', 'B_REFIT', 'C_REFIT')
OLD = {'A_REFIT': 'RAW16_ISO', 'B_REFIT': 'HP4_ISO', 'C_REFIT': 'RAW16_BLOCK2'}
TRAIN_KEYS = {'ids', 'architectures', 'x_standardized16', 'y_standardized', 'known_variance_standardized'}
QUERY_KEYS = {'ids', 'architectures', 'x_standardized16'}
OLD_TRAIN_KEYS = {'ids', 'architectures', 'x_standardized', 'y_standardized', 'known_variance_standardized'}
OLD_QUERY_KEYS = {'ids', 'architectures', 'x_standardized'}


def require(ok, message):
    if not ok: raise ValueError(message)
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
def close(a, b): return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-12 + 1e-10 * abs(b)
def same_numeric(a, b, name):
    if isinstance(b, list):
        require(isinstance(a, list) and len(a) == len(b), name + ' shape differs')
        for x, y in zip(a, b): same_numeric(x, y, name)
    else: require(close(a, b), name + ' differs')
def checked(root, rel):
    p = (Path(root) / rel).resolve(); require(p.is_relative_to(Path(root).resolve()), 'Source path escapes run')
    m = read(Path(root) / 'source_manifest.json')
    require(rel in m['files'] and sha(p) == m['files'][rel], 'BLOCKED_SOURCE_MISMATCH: ' + rel)
    return p


def project_training(value):
    require(set(value) == OLD_TRAIN_KEYS, 'Training snapshot contains unapproved fields')
    result = {k: copy.deepcopy(v) for k, v in value.items() if k != 'x_standardized'}
    result['x_standardized16'] = copy.deepcopy(value['x_standardized'])
    require(set(result) == TRAIN_KEYS and len(result['ids']) == 50 and len(set(result['ids'])) == 50, 'Current T50 required')
    require(len(result['architectures']) == len(result['y_standardized']) == len(result['known_variance_standardized']) == 50, 'Training row alignment differs')
    require(len(result['x_standardized16']) == 50 and all(len(row) == 16 and all(math.isfinite(v) for v in row) for row in result['x_standardized16']), 'Canonical finite T16 required')
    require(all(math.isfinite(v) for v in result['y_standardized']) and all(math.isfinite(v) and v >= 0 for v in result['known_variance_standardized']), 'Invalid training observations')
    return result


def project_query(value):
    require(set(value) == OLD_QUERY_KEYS, 'Query labels/SE or unknown field rejected')
    result = {'ids': copy.deepcopy(value['ids']), 'architectures': copy.deepcopy(value['architectures']),
              'x_standardized16': copy.deepcopy(value['x_standardized'])}
    n = len(result['ids']); require(n in (34, 35, 36, 38) and len(set(result['ids'])) == n, 'Legal public Q required')
    require(len(result['architectures']) == len(result['x_standardized16']) == n, 'Query row alignment differs')
    require(all(len(row) == 16 and all(math.isfinite(v) for v in row) for row in result['x_standardized16']), 'Canonical finite Q16 required')
    return result


def _data(root, subset_id):
    require(subset_id in {f's{s}_f{f}_n50' for s in (20260922, 20260923) for f in range(5)}, 'Unprescribed T subset')
    d = read(checked(root, 'data/' + subset_id + '.json'))
    require(d['subset_id'] == subset_id and d['identity']['query_labels_or_SE_present'] is False, 'Wrong frozen subset identity')
    return d


def load_fit(root, subset_id): return project_training(_data(root, subset_id)['training'])
def load_query(root, subset_id): return project_query(_data(root, subset_id)['query'])
def load_scaler(root, subset_id): return copy.deepcopy(_data(root, subset_id)['scaler'])
def load_identity(root, subset_id):
    d = _data(root, subset_id)
    return {'subset_id': subset_id, 'sample_seed': d['sample_seed'], 'fold': d['fold'],
            'data_path': 'data/' + subset_id + '.json', 'data_sha256': sha(Path(root) / ('data/' + subset_id + '.json')),
            'parent_input_identity': copy.deepcopy(d['identity']), 'scaler_sha256': digest(d['scaler']),
            'training_ids_sha256': digest(d['training']['ids']), 'query_ids_sha256': digest(d['query']['ids']),
            'query_labels_or_SE_present': False}


def load_parent_start(root, subset_id, method):
    require(method in METHODS, 'Unsupported model method')
    d = _data(root, subset_id); rel = 'source/parent_predictions/' + subset_id + '_' + OLD[method] + '.json'
    p = read(checked(root, rel))
    require(p['subset_id'] == subset_id and p['method'] == OLD[method] and p['total_steps'] == 1800 and p['model_seed'] is None, 'Wrong parent endpoint')
    require(p['train_candidate_ids'] == d['training']['ids'] and p['candidate_ids'] == d['query']['ids'] and p['scaler'] == d['scaler'], 'Parent start crossed T/Q/scaler boundary')
    require(p['training_ids_sha256'] == digest(d['training']['ids']) and p['scaler_sha256'] == digest(d['scaler']), 'Parent start identity differs')
    keys = ('constant_mean', 'outputscale', 'extra_noise_variance_standardized', 'lengthscale_architecture', 'lengthscale_hp')
    physical = {k: p['gp_parameters'][k] for k in keys}
    require(all(v is None or math.isfinite(v) for v in physical.values()), 'Nonfinite physical start')
    require(physical['outputscale'] >= 1e-8 and physical['extra_noise_variance_standardized'] >= 1e-6 and physical['lengthscale_hp'] >= 1e-4, 'Parent start violates frozen bounds')
    require(physical['lengthscale_architecture'] is None if method == 'B_REFIT' else physical['lengthscale_architecture'] >= 1e-4, 'Parent architecture length mismatch')
    return {'physical': physical, 'source': {'path': rel, 'sha256': sha(Path(root) / rel), 'parent_method': OLD[method],
            'parent_endpoint_state_hash': p['endpoint_state_hash'], 'parent_negative_mll': p['negative_mll'],
            'subset_id': subset_id, 'training_ids_sha256': digest(d['training']['ids']), 'scaler_sha256': digest(d['scaler'])}}


def validate_subset(root, sid, public):
    d = _data(root, sid); t = read(checked(root, 'training/' + sid + '.json'))
    rawq = read(checked(root, 'queries/fold' + str(d['fold']) + '.json'))
    require(set(rawq) == {'ids', 'architectures', 'x_raw'}, 'Raw Q whitelist violation')
    train, query = project_training(d['training']), project_query(d['query'])
    require(train['ids'] == t['ids'] == d['scaler']['fit_ids'] and query['ids'] == rawq['ids'], 'Parent ID order changed')
    require(query['architectures'] == rawq['architectures'] and len(query['ids']) == (38, 36, 36, 34, 35)[d['fold']], 'Query architecture/count differs')
    require(not (set(train['ids']) & set(query['ids'])) and not (set(train['architectures']) & set(query['architectures'])), 'T/Q ID or architecture leakage')
    require(d['raw_training']['x_raw'] == t['x_raw'] and d['raw_query'] == rawq, 'Frozen raw snapshot changed')
    require(d['scaler']['std_correction'] == 0 and d['scaler']['frozen'] is True and d['scaler']['dtype'] == 'float64', 'Frozen ddof0 scaler required')
    positions = {cid: i for i, cid in enumerate(public['ids'])}
    for ids, architectures, raw in ((train['ids'], train['architectures'], t['x_raw']), (query['ids'], query['architectures'], rawq['x_raw'])):
        for cid, arch, x in zip(ids, architectures, raw):
            ix = positions[cid]; metadata = public['candidate_metadata'][cid]
            require(arch == public['architectures'][ix] == metadata['architecture_fingerprint'], 'Architecture identity mismatch')
            require(x == public['z'][ix] + public['hp'][ix], 'Raw z12+HP4 identity differs')
            hp = metadata['raw_hp']; expected = [math.log10(hp['lr']), hp['dropout'], math.log2(hp['hidden_dim']), math.log10(hp['l2'])]
            same_numeric(x[12:], expected, 'Already-transformed HP metadata')
            require(metadata['training_protocol_identity'] == public['source_identity']['recipe_sha256'], 'Validation recipe identity differs')
    xm = [statistics.fmean(col) for col in zip(*t['x_raw'])]
    xs = [max(statistics.pstdev(col), 1e-6) for col in zip(*t['x_raw'])]
    ym = statistics.fmean(t['y']); ys = max(statistics.pstdev(t['y']), 1e-8)
    for sc in (d['scaler'], t['scaler']):
        same_numeric(sc['x_mean'], xm, 'T-only x_mean'); same_numeric(sc['x_scale'], xs, 'T-only x_scale')
        same_numeric(sc['y_mean'], ym, 'T-only y_mean'); same_numeric(sc['y_scale'], ys, 'T-only y_scale')
    sc = d['scaler']
    for raw, std in ((t['x_raw'], train['x_standardized16']), (rawq['x_raw'], query['x_standardized16'])):
        expected = [[(v - sc['x_mean'][j]) / sc['x_scale'][j] for j, v in enumerate(row)] for row in raw]
        same_numeric(std, expected, 'Frozen standardized input')
    same_numeric(train['y_standardized'], [(v-sc['y_mean'])/sc['y_scale'] for v in t['y']], 'T-only standardized y')
    same_numeric(train['known_variance_standardized'], [(v/sc['y_scale'])**2 for v in t['measurement_se']], 'SE variance exactly once')
    require(d['hp4_projection']['training_x_standardized'] == [r[12:] for r in d['training']['x_standardized']], 'B projection not exact last4')
    require(d['hp4_projection']['query_x_standardized'] == [r[12:] for r in d['query']['x_standardized']], 'B query projection not exact last4')
    for method in METHODS: load_parent_start(root, sid, method)
    return {'subset_id': sid, 'sample_seed': d['sample_seed'], 'fold': d['fold'], 'n': 50, 'n_query': len(query['ids']),
            'data_path': 'data/' + sid + '.json', 'data_sha256': sha(Path(root) / ('data/' + sid + '.json')),
            'training_ids_sha256': digest(train['ids']), 'query_ids_sha256': digest(query['ids']), 'scaler_sha256': digest(sc),
            'method_order': list(METHODS)}


def _prepare(root):
    root = Path(root); require(not (root / 'data_manifest.json').exists(), 'Numeric preparation is one-shot')
    lock = root / 'data_prepare_once.lock'; lock.mkdir(); write(lock / 'entry.json', {'permanent': True, 'model_calls': 0})
    public = read(checked(root, 'public_inputs.json'))
    require(len(public['ids']) == len(set(public['ids'])) == 179 and public['label_values_decoded'] == 0 and public['SE_values_decoded'] == 0, 'Public-only legal A179 input required')
    subsets, groups, trajectories = [], [], []
    for seed in (20260922, 20260923):
        for fold in range(5):
            sid = f's{seed}_f{fold}_n50'; sub = validate_subset(root, sid, public); sub['subset_index'] = len(subsets); subsets.append(sub)
            for method in METHODS:
                uid = sid + '_' + method; kinds = ['PARENT', 'COMMON_INITIAL'] if method != 'C_REFIT' else ['PARENT', 'A_SELECTED_EMBEDDING', 'B_SELECTED_BOUNDARY']
                group = {**sub, 'unit_id': uid, 'method': method, 'model_seed': None, 'rng_seed': 0,
                         'group_index': len(groups), 'trajectory_ids': [uid + '_start' + str(i) for i in range(len(kinds))],
                         'max_optimization_fg_calls': 600, 'parent_prediction_path': 'source/parent_predictions/' + sid + '_' + OLD[method] + '.json'}
                group['parent_prediction_sha256'] = sha(root / group['parent_prediction_path']); groups.append(group)
                for i, kind in enumerate(kinds):
                    trajectories.append({**{k: group[k] for k in ('unit_id', 'subset_id', 'sample_seed', 'fold', 'method', 'model_seed', 'rng_seed', 'data_path', 'data_sha256', 'training_ids_sha256', 'scaler_sha256')},
                        'trajectory_id': uid + '_start' + str(i), 'trajectory_index': len(trajectories), 'start_index': i,
                        'start_kind': kind, 'max_fg_calls': 200 if method == 'C_REFIT' else 300,
                        'max_accepted_iterations': 150 if method == 'C_REFIT' else 250,
                        'start_parent_prediction_path': group['parent_prediction_path'] if kind == 'PARENT' else None,
                        'start_selected_unit_id': sid + ('_A_REFIT' if kind == 'A_SELECTED_EMBEDDING' else '_B_REFIT') if kind in ('A_SELECTED_EMBEDDING', 'B_SELECTED_BOUNDARY') else None})
    work = {'subsets': subsets, 'groups': groups, 'trajectories': trajectories, 'subset_count': 10, 'group_count': 30,
            'trajectory_count': 70, 'new_prediction_rows': 1074, 'scoring_units': 150, 'scoring_candidate_rows': 5370,
            'optimization_fg_limit': 18000, 'method_order': list(METHODS), 'subset_order': 'sample_seed then fold ascending',
            'model_seed': None, 'execution_rng_seed': 0, 'trajectory_endpoint_predictions': False}
    require(len(groups) == 30 and len(trajectories) == 70 and sum(t['max_fg_calls'] for t in trajectories) == 18000, 'Fixed work matrix differs')
    write(root / 'worklist.json', work)
    original_refs = read(root / 'source/parent/historical_refs.json')['records']; refs = []
    for r in original_refs:
        rel = 'source/historical_predictions/' + Path(r['path']).name
        require(sha(checked(root, rel)) == r['sha256'], 'Historical prediction bytes changed')
        refs.append({**r, 'path': rel, 'parent_source_path': r['path'], 'is_historical': True, 'reference_kind': 'historical_readout'})
    for sub in subsets:
        for method, mapped in zip(METHODS, ('A_OLD', 'B_OLD', 'C_OLD')):
            rel = 'source/parent_predictions/' + sub['subset_id'] + '_' + OLD[method] + '.json'; p = read(checked(root, rel))
            refs.append({'unit_id': sub['subset_id'] + '_' + mapped, 'subset_id': sub['subset_id'], 'fold': sub['fold'],
                'sample_seed': sub['sample_seed'], 'model_seed': None, 'method': mapped, 'parent_method': OLD[method],
                'path': rel, 'sha256': sha(root / rel), 'is_historical': True, 'reference_kind': 'parent_block_GP',
                'parent_unit_id': p['unit_id'], 'training_ids_sha256': p['training_ids_sha256'], 'scaler_sha256': p['scaler_sha256']})
    write(root / 'historical_refs.json', {'records': refs, 'reference_count': 120, 'parent_block_models': 30, 'historical_readout_models': 90,
                                        'new_model_fits': 0, 'source_manifest_sha256': sha(root / 'source_manifest.json')})
    original_masks = read(root / 'source/parent/masks_manifest.json'); masks = []
    for r in original_masks['records']:
        require(sha(checked(root, r['path'])) == r['sha256'], 'Frozen HEAD mask changed')
        masks.append({**r, 'resolved_source_prediction_path': 'source/historical_predictions/' + Path(r['source_prediction_path']).name})
    require(len(masks) == 30, '30 individual HEAD masks required')
    write(root / 'masks_manifest.json', {'files': original_masks['files'], 'records': masks, 'mask_count': 30,
        'parent_masks_manifest_sha256': sha(root / 'source/parent/masks_manifest.json'), 'original_mask_bytes_unchanged': True})
    files = {s['data_path']: s['data_sha256'] for s in subsets}
    files.update({r['path']: r['sha256'] for r in masks})
    for name in ('worklist.json', 'historical_refs.json', 'masks_manifest.json'): files[name] = sha(root / name)
    result = {'files': files, 'subsets': subsets, 'data_projection_count': 10, 'trajectory_count': 70, 'selected_model_count': 30,
        'historical_reference_count': 120, 'mask_count': 30, 'source_manifest_sha256': sha(root / 'source_manifest.json'),
        'protocol_sha256': sha(root / 'protocol.json'), 'worklist_sha256': sha(root / 'worklist.json'),
        'query_label_values_decoded': 0, 'parent_numeric_snapshot_bytes_unchanged': True,
        'training_interface': sorted(TRAIN_KEYS), 'query_interface': sorted(QUERY_KEYS),
        'interface_key_mapping': 'parent x_standardized renamed to x_standardized16 only; numeric values unchanged',
        'numeric_preparation_charged': True, 'scaler_scope': 'each own T50; no pooled labels/statistics', 'scaler_comparison_atol': 1e-12, 'scaler_comparison_rtol': 1e-10}
    write(root / 'data_manifest.json', result)
    return {'pass': True, 'subsets': 10, 'groups': 30, 'trajectories': 70, 'references': 120, 'masks': 30,
            'GP_model_calls': 0, 'query_label_values_decoded': 0}


def prepare(root, budget):
    require(budget is not None, 'Numeric data preparation must use the experiment budget')
    with budget.measure('frozen_parent_data_validation', 'setup'):
        result = _prepare(root)
    budget.persist(); return result


def run_checks():
    ids = [str(i) for i in range(50)]; arches = ['arch' + str(i) for i in range(50)]
    train = {'ids': ids, 'architectures': arches, 'x_standardized': [[float(i+j) for j in range(16)] for i in range(50)],
             'y_standardized': [0.] * 50, 'known_variance_standardized': [.1] * 50}
    query = {k: copy.deepcopy(train[k][:38]) for k in ('ids', 'architectures', 'x_standardized')}
    projected = project_query(query)
    require(project_training(train)['x_standardized16'] == train['x_standardized'], 'Training projection changed values')
    for key in ('y', 'measurement_se', 'labels', 'query_y', 'query_SE', 'case_audit'):
        try: project_query({**query, key: object()})
        except ValueError: pass
        else: raise AssertionError('Query poison accepted')
    require(project_query(query) == projected, 'Poison affected subsequent projection')
    projected['x_standardized16'][0][0] = 999.
    require(query['x_standardized'][0][0] != 999., 'Projection shares mutable values')
    return {'pass': True, 'checks': ['current_T_exact_whitelist', 'public_Q_exact_whitelist', 'six_poison_fields_rejected', 'same_numeric16_no_transform', 'no_shared_mutable_rows'], 'model_calls': 0, 'query_labels_read': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path); parser.add_argument('--synthetic-checks', action='store_true')
    args = parser.parse_args()
    if args.synthetic_checks: result = run_checks()
    else:
        from nested_core import Budget
        require(args.root is not None, '--root required'); result = prepare(args.root, Budget(args.root / 'budget.json'))
    print(json.dumps(result, ensure_ascii=False))
