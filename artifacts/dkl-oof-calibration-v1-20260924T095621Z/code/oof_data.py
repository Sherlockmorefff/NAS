"""Architecture OOF splits and narrow U/V boundary. Standard library only.

No global observation table is built. Each load opens one authorized T50
snapshot, returns only U observations, and drops the full snapshot at return.
The scaler is intentionally absent: the numerical core must fit it on U.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path

ALGORITHM = 'oof-arch3-v1'
QUERY_KEYS = {'ids', 'architectures', 'x_raw'}
TRAIN_KEYS = QUERY_KEYS | {'y', 'measurement_se'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def child(root, rel):
    path = (Path(root) / rel).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Path escapes task artifact root')
    return path


def checked_source(root, rel):
    source = read(Path(root) / 'source_manifest.json')
    require(rel in source['files'], 'Input is not a pinned source: ' + rel)
    path = child(root, rel)
    require(sha(path) == source['files'][rel], 'Pinned input hash differs: ' + rel)
    return path


def _public(doc, counts, label):
    require(set(doc) == QUERY_KEYS, label + ' violates exact public whitelist')
    ids, arches, x = doc['ids'], doc['architectures'], doc['x_raw']
    require(len(ids) in counts and len(set(ids)) == len(ids), label + ' candidate count/identity')
    require(len(arches) == len(x) == len(ids), label + ' shape mismatch')
    require(all(isinstance(v, str) and v for v in ids + arches), label + ' invalid ID/architecture')
    require(all(len(row) == 16 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in row)
                for row in x), label + ' invalid raw features')
    return copy.deepcopy(doc)


def query_projection(doc):
    """V is 16/17 public rows; any observation or extra key is rejected."""
    return _public(doc, (16, 17), 'Inner V')


def training_projection(doc):
    require(set(doc) == TRAIN_KEYS, 'Inner U violates exact training whitelist')
    out = _public({k: doc[k] for k in QUERY_KEYS}, (33, 34), 'Inner U')
    require(len(doc['y']) == len(doc['measurement_se']) == len(out['ids']), 'U observation shape')
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in doc['y']), 'Nonfinite U labels')
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in doc['measurement_se']), 'Invalid U SE')
    out.update(y=list(doc['y']), measurement_se=list(doc['measurement_se']))
    return out


def snapshot_public(snapshot):
    ids = list(snapshot['ids'])
    require(len(ids) == len(set(ids)) == 50, 'Only current T50 snapshot is permitted')
    meta = snapshot['metadata']
    require(meta['n'] == 50 and meta['query_labels_released'] == 0 and meta['query_SE_released'] == 0,
            'Source snapshot includes nontraining observations')
    candidates = meta['candidate_metadata']
    require([row['candidate_id'] for row in candidates] == ids, 'Source candidate metadata order differs')
    return _public({'ids': ids, 'architectures': [row['architecture_fingerprint'] for row in candidates],
                    'x_raw': snapshot['x_raw']}, (50,), 'Current T50 public projection')


def architecture_split(subset_id, public, outer_query):
    public = _public(public, (50,), 'Split T50')
    require(set(outer_query) == QUERY_KEYS, 'Outer Q public whitelist differs')
    require(not set(public['ids']) & set(outer_query['ids']), 'T and outer Q share candidates')
    require(not set(public['architectures']) & set(outer_query['architectures']), 'T and outer Q share architectures')
    groups = {}
    for i, arch in enumerate(public['architectures']):
        groups.setdefault(arch, []).append(i)
    require(39 <= len(groups) <= 44 and all(len(indices) in (1, 2) for indices in groups.values()),
            'BLOCKED_SOURCE_OR_SPLIT_MISMATCH: expected 39-44 architectures of size1/2')
    ordered = sorted(groups, key=lambda a: (-len(groups[a]), hashlib.sha256(
        (ALGORITHM + '|' + subset_id + '|' + a).encode('utf-8')).hexdigest(), a))
    allocation, counts = {}, [0, 0, 0]
    for arch in ordered:
        fold = min(range(3), key=lambda f: (counts[f], f))
        allocation[arch] = fold
        counts[fold] += len(groups[arch])
    require(sorted(counts) == [16, 17, 17], 'BLOCKED_SOURCE_OR_SPLIT_MISMATCH: V counts differ')
    folds = []
    for fold in range(3):
        vi = [i for i, a in enumerate(public['architectures']) if allocation[a] == fold]
        ui = [i for i, a in enumerate(public['architectures']) if allocation[a] != fold]
        record = {'inner_fold': fold, 'n_outer': 50, 'n_fit': len(ui), 'n_holdout': len(vi),
                  'U_indices': ui, 'V_indices': vi,
                  'U_ids': [public['ids'][i] for i in ui], 'V_ids': [public['ids'][i] for i in vi],
                  'U_architectures': [public['architectures'][i] for i in ui],
                  'V_architectures': [public['architectures'][i] for i in vi]}
        for key in ('U_ids', 'V_ids', 'U_architectures', 'V_architectures'):
            record[key + '_sha256'] = digest(record[key])
        require(not set(record['U_architectures']) & set(record['V_architectures']), 'U/V architecture overlap')
        folds.append(record)
    seen = [cid for fold in folds for cid in fold['V_ids']]
    require(len(seen) == len(set(seen)) == 50 and set(seen) == set(public['ids']), 'OOF partition is not exact')
    return {'algorithm': ALGORITHM, 'subset_id': subset_id, 'n_outer': 50,
            'architecture_count': len(groups), 'architecture_group_sizes': {a: len(groups[a]) for a in sorted(groups)},
            'group_assignment_order': ordered, 'group_to_fold': allocation,
            'training_ids': public['ids'], 'training_architectures': public['architectures'],
            'training_ids_sha256': digest(public['ids']), 'outer_query_ids_sha256': digest(outer_query['ids']),
            'outer_query_architectures_sha256': digest(outer_query['architectures']),
            'folds': folds, 'uses_label_values': False, 'uses_SE_values': False}


def split_training_snapshot(snapshot, split, inner_fold):
    """Reads only selected U values; V labels/SE can be arbitrary poisoned objects."""
    public = snapshot_public(snapshot)
    require(public['ids'] == split['training_ids'] and public['architectures'] == split['training_architectures'],
            'Current T50 differs from frozen split')
    fold = split['folds'][int(inner_fold)]
    require(fold['inner_fold'] == int(inner_fold), 'Inner fold identity differs')
    ui, vi = fold['U_indices'], fold['V_indices']
    training = {k: [public[k][i] for i in ui] for k in QUERY_KEYS}
    training.update(y=[snapshot['y'][i] for i in ui], measurement_se=[snapshot['measurement_se'][i] for i in ui])
    query = {k: [public[k][i] for i in vi] for k in QUERY_KEYS}
    training, query = training_projection(training), query_projection(query)
    require(training['ids'] == fold['U_ids'] and query['ids'] == fold['V_ids'], 'U/V order differs from frozen split')
    require(not set(training['architectures']) & set(query['architectures']), 'U/V architecture leakage')
    return {'training': training, 'query': query,
            'identity': {key: fold[key] for key in ('inner_fold', 'n_outer', 'n_fit', 'n_holdout',
                         'U_ids_sha256', 'V_ids_sha256', 'U_architectures_sha256', 'V_architectures_sha256')}}


def load_inner(root, pipeline):
    root = Path(root)
    work = read(root / 'worklist.json')
    matches = [p for p in work['pipelines'] if p['pipeline_id'] == pipeline['pipeline_id']]
    require(len(matches) == 1 and matches[0] == pipeline, 'Pipeline is not the frozen work item')
    sm = read(root / 'inner_splits_manifest.json')
    require(sm['protocol_sha256'] == sha(root / 'protocol.json'), 'Split protocol changed')
    rel = pipeline['split_path']
    require(sm['files'][rel] == sha(root / rel) == pipeline['split_sha256'], 'Frozen split changed')
    snapshot = read(checked_source(root, 'training/' + pipeline['subset_id'] + '.json'))
    require(snapshot['metadata']['subset_id'] == pipeline['subset_id'], 'Wrong current T snapshot')
    result = split_training_snapshot(snapshot, read(root / rel), pipeline['inner_fold'])
    result['identity'].update(pipeline_id=pipeline['pipeline_id'], context_id=pipeline['context_id'],
                              subset_id=pipeline['subset_id'], source_training_sha256=sha(root / 'training' / (pipeline['subset_id'] + '.json')),
                              split_sha256=pipeline['split_sha256'], query_labels_or_SE_present=False)
    return result


def prepare(root):
    root = Path(root)
    require(not (root / 'worklist.json').exists() and not (root / 'inner_splits_manifest.json').exists(), 'OOF data preparation is one-shot')
    parent = read(checked_source(root, 'source/parent_worklist.json'))
    pub = read(checked_source(root, 'public_inputs.json'))
    require(pub['label_values_decoded'] == pub['SE_values_decoded'] == 0, 'Public source has observations')
    mapping = {cid: (arch, z + hp) for cid, arch, z, hp in zip(pub['ids'], pub['architectures'], pub['z'], pub['hp'])}
    contexts = sorted([c for c in parent['contexts'] if c['n'] == 50], key=lambda c: (c['sample_seed'], c['model_seed'], c['fold']))
    require(len(contexts) == 30 and len({c['context_id'] for c in contexts}) == 30, 'Parent n50 contexts differ')
    splits, files, pipelines = {}, {}, []
    for c in contexts:
        sid = c['subset_id']
        if sid not in splits:
            snapshot = read(checked_source(root, 'training/' + sid + '.json'))
            public = snapshot_public(snapshot)
            q = read(checked_source(root, 'queries/fold' + str(c['fold']) + '.json'))
            require(set(q) == QUERY_KEYS, 'Outer Q must be public only')
            require(snapshot['metadata']['fold'] == c['fold'] and snapshot['metadata']['sample_seed'] == c['sample_seed'], 'T context metadata differs')
            require(digest(public['ids']) == c['selected_ids_sha256'], 'Parent T order changed')
            for view in (public, q):
                require(all(mapping[cid] == (arch, x) for cid, arch, x in zip(view['ids'], view['architectures'], view['x_raw'])), 'Architecture/raw feature source mismatch')
            split = architecture_split(sid, public, q)
            rel = 'inner_splits/' + sid + '.json'
            write(root / rel, split)
            files[rel] = sha(root / rel)
            splits[sid] = split
        for fold in splits[sid]['folds']:
            index = len(pipelines)
            record = {k: c[k] for k in ('context_id', 'subset_id', 'fold', 'sample_seed', 'model_seed')}
            record.update(pipeline_id=c['context_id'] + '_i' + str(fold['inner_fold']),
                inner_fold=fold['inner_fold'], n_outer=50, n_fit=fold['n_fit'], n_holdout=fold['n_holdout'],
                split_path='inner_splits/' + sid + '.json', split_sha256=files['inner_splits/' + sid + '.json'],
                group_order=['GP_CONST', 'GP_TREND'] if index % 2 == 0 else ['GP_TREND', 'GP_CONST'])
            pipelines.append(record)
    require(len(splits) == 10 and len(pipelines) == 90, 'Fixed inner matrix differs')
    work = {'outer_contexts': contexts, 'pipelines': pipelines, 'subset_count': 10, 'outer_context_count': 30,
            'pipeline_count': 90, 'inner_prediction_count': 270, 'oof_rows': 4500, 'tau_count': 60,
            'outer_endpoint_count': 150, 'formal_optimizer_updates': 387000,
            'formal_updates_by_stage': {'COMMON_MSE': 90000, 'COMMON_GP': 27000, 'GP_CONST': 135000, 'GP_TREND': 135000},
            'execution_order': ['sample_seed', 'model_seed', 'fold', 'inner_fold'], 'readout_order': 'pipeline index even CONST,TREND; odd TREND,CONST'}
    write(root / 'worklist.json', work)
    write(root / 'inner_splits_manifest.json', {'algorithm': ALGORITHM, 'files': files, 'subset_count': 10,
        'pipeline_count': 90, 'protocol_sha256': sha(root / 'protocol.json'), 'source_manifest_sha256': sha(root / 'source_manifest.json'),
        'worklist_sha256': sha(root / 'worklist.json'), 'frozen_at_utc': now(), 'label_or_SE_values_used_for_split': False})
    return {'pass': True, 'subsets': 10, 'contexts': 30, 'pipelines': 90,
            'holdout_counts': {sid: [f['n_holdout'] for f in split['folds']] for sid, split in splits.items()}, 'model_calls': 0}


def run_checks():
    ids = ['synthetic_' + str(i) for i in range(50)]
    arches = ['arch_' + str(i % 40) for i in range(50)]
    public = {'ids': ids, 'architectures': arches, 'x_raw': [[i / 50.] * 16 for i in range(50)]}
    outer = {'ids': ['outside'], 'architectures': ['outside_arch'], 'x_raw': [[0.] * 16]}
    split = architecture_split('synthetic_subset', public, outer)
    snapshot = {**public, 'y': [i / 100. for i in range(50)], 'measurement_se': [.01] * 50,
                'metadata': {'n': 50, 'query_labels_released': 0, 'query_SE_released': 0,
                    'candidate_metadata': [{'candidate_id': cid, 'architecture_fingerprint': a} for cid, a in zip(ids, arches)]}}
    checks = {}
    checks['deterministic_architecture_partition'] = split == architecture_split('synthetic_subset', public, outer)
    for j in range(3):
        a = split_training_snapshot(snapshot, split, j)
        poisoned = copy.deepcopy(snapshot)
        for i in split['folds'][j]['V_indices']:
            poisoned['y'][i] = {'poison': 'V label must not be inspected'}
            poisoned['measurement_se'][i] = float('nan')
        b = split_training_snapshot(poisoned, split, j)
        checks['V_observation_poison_no_effect_fold' + str(j)] = digest(a) == digest(b)
        bad = dict(a['query'], y=['poison'] * a['identity']['n_holdout'])
        try:
            query_projection(bad)
            rejected = False
        except ValueError:
            rejected = True
        checks['query_observation_rejected_fold' + str(j)] = rejected
    require(all(checks.values()), 'Pure data isolation check failed')
    return {'pass': True, 'checks': checks, 'model_calls': 0, 'optimizer_calls': 0, 'real_labels_accessed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path); p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.check_only:
        result = run_checks()
    else:
        require(args.root is not None, '--root required')
        result = prepare(args.root)
    print(json.dumps(result, ensure_ascii=False))
