"""Stage-2 A-only metadata preparation and per-fit selective label release.

The training runner receives only its selected 20/50 labels. Query projection
contains public inputs only. Full A validation observations are decoded only by
the separate scoring command after verification of the unified prediction seal.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

from stage01_data import (
    A_ID_HASH, RECIPE_HASH, EXPECTED_PUBLIC_HASHES, digest, file_hash, now,
    public_sources, read_json, selected_objects, validate_selected_labels,
    write_new, mock_checks as stage01_mock_checks,
)

SOURCE_DIR = Path('/home/fangxy/D-VAE/results/posthoc/fuzzy-scheduler-regularization-v1-20260921T064826Z')
PARTITION_PATH = Path('/home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/partition.json')
SAMPLE_SEEDS = (20260922, 20260923)
MODEL_SEEDS = (0, 1, 2)
EXPECTED_QUERY_COUNTS = [38, 36, 36, 34, 35]


def _identity(value):
    value['identity'] = digest(value)
    return value


def _check_identity(value):
    if value.get('identity') != digest({k: v for k, v in value.items() if k != 'identity'}):
        raise ValueError('Manifest semantic identity mismatch')


def _verified_path(root, relative, expected_sha):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or file_hash(path) != expected_sha:
        raise ValueError('Frozen file identity mismatch: ' + str(relative))
    return path


def query_projection(rows):
    """Explicit whitelist: even deliberately poisoned y/SE values are not read."""
    ids = [row['candidate_id'] for row in rows]
    architectures = [row['architecture_fingerprint'] for row in rows]
    x = np.asarray([row['x_raw'] for row in rows], dtype=np.float64)
    if len(set(ids)) != len(ids) or x.shape != (len(ids), 16) or not np.isfinite(x).all():
        raise ValueError('Invalid public query inputs')
    return {'ids': ids, 'architectures': architectures, 'x_raw': x.tolist()}


def build_folds(features, partition_path):
    ids = set(features['ids'])
    arch = dict(zip(features['ids'], features['architectures']))
    partition_path = Path(partition_path)
    if partition_path.exists():
        original = read_json(partition_path)
        # Historical partition also contains unrelated identity metadata. Only
        # these A identity/group fields are retained; no B file is opened.
        if set(original['A_ids']) != ids or len(original['A_ids']) != 179 or original['A_hash'] != A_ID_HASH:
            raise ValueError('Existing partition legal A identity mismatch')
        by_name = {row['family']: row for row in original['folds']}
        names = ['fold' + str(fold) for fold in range(5)]
        if len(original['folds']) != 5 or set(by_name) != set(names):
            raise ValueError('Existing partition fold names changed')
        rows = []
        for fold, name in enumerate(names):
            old = by_name[name]
            rows.append({'fold': fold, 'original_fold_id': name,
                         'query_ids': sorted(old['query_ids']),
                         'training_pool_ids': sorted(old['train_ids'])})
        origin = {'path': str(partition_path), 'sha256': file_hash(partition_path),
                  'reused_original': True, 'historical_fold_pairing_valid': True}
    else:
        ordered_arch = sorted(set(arch.values()), key=lambda a: (hashlib.sha256(a.encode('utf8')).hexdigest(), a))
        assigned = {a: i % 5 for i, a in enumerate(ordered_arch)}
        rows = []
        for fold in range(5):
            query = sorted(cid for cid in ids if assigned[arch[cid]] == fold)
            rows.append({'fold': fold, 'original_fold_id': None, 'query_ids': query,
                         'training_pool_ids': sorted(ids - set(query))})
        origin = {'path': str(partition_path), 'sha256': None, 'reused_original': False,
                  'historical_fold_pairing_valid': False,
                  'note': '新分组，不能与历史逐折数值直接配对',
                  'fallback': 'sort architecture by SHA256 UTF8 then fingerprint; index mod5'}
    seen = []
    for row in rows:
        q, t = row['query_ids'], row['training_pool_ids']
        if len(q) != len(set(q)) or len(t) != len(set(t)) or not set(q) <= ids:
            raise ValueError('Duplicate or non-A fold IDs')
        if set(t) != ids - set(q):
            raise ValueError('Training pool is not exact query complement')
        qa, ta = {arch[c] for c in q}, {arch[c] for c in t}
        if not qa.isdisjoint(ta):
            raise ValueError('Architecture overlaps query and training pool')
        row.update(query_count=len(q), training_pool_count=len(t), query_architecture_count=len(qa),
                   training_pool_architecture_count=len(ta))
        seen.extend(q)
    if len(seen) != 179 or set(seen) != ids:
        raise ValueError('A IDs do not appear exactly once as outer queries')
    if origin['reused_original'] and [len(row['query_ids']) for row in rows] != EXPECTED_QUERY_COUNTS:
        raise ValueError('Existing partition fold candidate counts changed')
    return _identity({'folds': rows, 'source': origin,
                      'original_to_current': {row['original_fold_id']: row['fold'] for row in rows if row['original_fold_id'] is not None},
                      'legal_A_ids_sha256': A_ID_HASH,
                      'checks': {'A_query_coverage_exact_once': True, 'architecture_disjoint': True,
                                 'train_exact_query_complement': True}})


def build_subsets(features, folds):
    arch = dict(zip(features['ids'], features['architectures']))
    subsets = {}
    for sample_seed in SAMPLE_SEEDS:
        for foldrow in folds['folds']:
            fold = foldrow['fold']
            rng = np.random.default_rng(np.random.SeedSequence([sample_seed, fold]))
            ordered = rng.permutation(sorted(foldrow['training_pool_ids'])).tolist()
            for n in (20, 50):
                key = f's{sample_seed}_f{fold}_n{n}'
                chosen = ordered[:n]
                subsets[key] = {'subset_id': key, 'fold': fold, 'sample_seed': sample_seed, 'n': n,
                                'ids': chosen, 'selected_ids_sha256': digest(chosen),
                                'architecture_count': len({arch[c] for c in chosen})}
            if subsets[f's{sample_seed}_f{fold}_n20']['ids'] != subsets[f's{sample_seed}_f{fold}_n50']['ids'][:20]:
                raise AssertionError('20 must be the ordered prefix of 50')
    return _identity({'subsets': subsets, 'sample_seeds': list(SAMPLE_SEEDS),
                      'algorithm': 'default_rng(SeedSequence([sample_seed,fold])).permutation(sorted(training_pool_ids))[:n]',
                      'model_rng_consumed': False, 'label_values_decoded_before_selection': 0})


def build_worklist(subsets):
    contexts = []
    endpoints = []
    for ss in SAMPLE_SEEDS:
        for ms in MODEL_SEEDS:
            for fold in range(5):
                for n in (20, 50):
                    sid = f's{ss}_f{fold}_n{n}'
                    cid = f's{ss}_m{ms}_f{fold}_n{n}'
                    raw_id = sid + '_RAW16'
                    context = {'context_id': cid, 'subset_id': sid, 'fold': fold, 'sample_seed': ss,
                               'n': n, 'model_seed': ms, 'run_raw': ms == 0, 'raw_unit_id': raw_id,
                               'selected_ids_sha256': subsets['subsets'][sid]['selected_ids_sha256'],
                               'branch_order': ['FROZEN', 'REG', 'DKL']}
                    contexts.append(context)
                    if ms == 0:
                        endpoints.append({'unit_id': raw_id, 'context_id': cid, 'subset_id': sid, 'method': 'RAW16',
                                          'fold': fold, 'sample_seed': ss, 'n': n, 'model_seed': None})
                    for method in context['branch_order']:
                        endpoints.append({'unit_id': cid + '_' + method, 'context_id': cid, 'subset_id': sid,
                                          'method': method, 'fold': fold, 'sample_seed': ss, 'n': n, 'model_seed': ms})
    if len(contexts) != 60 or len(endpoints) != 200 or len({x['unit_id'] for x in endpoints}) != 200:
        raise AssertionError('Fixed matrix counts changed')
    return _identity({'contexts': contexts, 'endpoints': endpoints, 'context_count': 60,
                      'mapped_endpoint_count': 180, 'raw_endpoint_count': 20, 'total_endpoint_count': 200,
                      'execution_order': 'sample_seed,model_seed,fold,n; raw first when model_seed0; FROZEN,REG,DKL',
                      'formal_optimizer_updates': 344000})


def prepare_public(source_dir, partition_path, output_dir):
    """No label object is deserialized in this operation (hash bytes only)."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    features, metadata, recipe, checks, historical = public_sources(source_dir)
    label_path = source_dir / 'private/evaluation_A.json'
    expected_label_hash = historical['private_inputs']['private/evaluation_A.json']
    if file_hash(label_path) != expected_label_hash:
        raise ValueError('Source A-only validation label file changed')
    public = _identity({**features, 'candidate_metadata': metadata,
                       'source_identity': {'source_dir': str(source_dir), 'public_sha256': EXPECTED_PUBLIC_HASHES,
                                           'source_protocol_sha256': file_hash(source_dir / 'protocol.json'),
                                           'source_protocol_identity': historical['identity'],
                                           'label_relative_path': 'private/evaluation_A.json',
                                           'source_A_label_sha256': expected_label_hash,
                                           'recipe_sha256': RECIPE_HASH, 'recipe': recipe['training_protocol_identity'],
                                           'recipe_preregistration': recipe['actual_preregistration_training'],
                                           'legal_A_ids_sha256': A_ID_HASH},
                       'created_at_utc': now(), 'label_values_decoded': 0, 'SE_values_decoded': 0})
    folds = build_folds(features, partition_path)
    subsets = build_subsets(features, folds)
    worklist = build_worklist(subsets)
    for filename, value in [('public_inputs.json', public), ('folds.json', folds),
                            ('subsets.json', subsets), ('worklist.json', worklist)]:
        write_new(output_dir / filename, value)
    audit = {'status': 'PASS', 'checked_at_utc': now(), 'labels_decoded': 0, 'B_label_files_opened': 0,
             'test_metric_files_opened': 0, 'query_counts': [f['query_count'] for f in folds['folds']],
             'training_pool_counts': [f['training_pool_count'] for f in folds['folds']],
             'source_A_label_sha256': expected_label_hash, 'partition': folds['source'],
             'subsets_count': 20, 'nested20_in50': True, 'contexts': 60, 'endpoints': 200,
             'query_projection_fields': ['ids', 'architectures', 'x_raw'],
             'architecture_counts': {sid: row['architecture_count'] for sid, row in subsets['subsets'].items()}}
    write_new(output_dir / 'audit/data_preparation.json', audit)
    return audit


class Stage2Adapter:
    def __init__(self, output_dir, frozen_manifest_path=None):
        self.root = Path(output_dir)
        self.freeze_path = Path(frozen_manifest_path) if frozen_manifest_path else self.root / 'frozen_manifest.json'
        self.freeze = read_json(self.freeze_path)
        required = {'protocol.json', 'public_inputs.json', 'folds.json', 'subsets.json', 'worklist.json', 'code_manifest.json'}
        if not required <= set(self.freeze['files']) or self.freeze['labels_released'] is not False:
            raise ValueError('Missing pre-label frozen manifest')
        for name, expected in self.freeze['files'].items():
            _verified_path(self.root, name, expected)
        codes = read_json(self.root / 'code_manifest.json')
        for name, expected in codes['files'].items():
            _verified_path(self.root, name, expected)
        if self.freeze['protocol_sha256'] != file_hash(self.root / 'protocol.json'):
            raise ValueError('Protocol hash mismatch')
        if self.freeze['code_manifest_sha256'] != file_hash(self.root / 'code_manifest.json'):
            raise ValueError('Code manifest hash mismatch')
        self.protocol = read_json(self.root / 'protocol.json')
        if self.protocol['next_stage_authorized'] is not False:
            raise ValueError('Stage scope changed')
        self.public = read_json(self.root / 'public_inputs.json')
        self.folds = read_json(self.root / 'folds.json')
        self.subsets = read_json(self.root / 'subsets.json')['subsets']
        self.worklist = read_json(self.root / 'worklist.json')
        _check_identity(self.public)
        _check_identity(self.folds)
        _check_identity(self.worklist)
        if digest(sorted(self.public['ids'])) != A_ID_HASH:
            raise ValueError('Frozen A universe mismatch')
        self.positions = {cid: i for i, cid in enumerate(self.public['ids'])}

    def query_inputs(self, fold):
        ids = self.folds['folds'][fold]['query_ids']
        if self.folds['folds'][fold]['fold'] != fold:
            raise ValueError('Fold indexing changed')
        rows = []
        for cid in ids:
            i = self.positions[cid]
            rows.append({'candidate_id': cid, 'architecture_fingerprint': self.public['architectures'][i],
                         'x_raw': self.public['z'][i] + self.public['hp'][i]})
        return query_projection(rows)

    def _source_labels(self, ids):
        source = self.public['source_identity']
        path = Path(source['source_dir']) / source['label_relative_path']
        if file_hash(path) != source['source_A_label_sha256']:
            raise ValueError('A validation label source changed after freeze')
        with path.open() as stream:
            labels, access = selected_objects(stream, ids)
        return labels, access

    def release_training(self, selected_ids):
        ids = list(selected_ids)
        matches = [row for row in self.subsets.values() if row['ids'] == ids]
        if len(matches) != 1 or len(ids) not in (20, 50):
            raise ValueError('Training IDs and order must match one frozen 20/50 subset')
        subset = matches[0]
        if set(ids) & set(self.folds['folds'][subset['fold']]['query_ids']):
            raise ValueError('Current query candidate included in training release')
        opened_at = now()
        labels, access = self._source_labels(ids)
        y, se, repeats = validate_selected_labels(ids, labels)
        ix = [self.positions[cid] for cid in ids]
        z = np.asarray(self.public['z'], dtype=np.float64)[ix]
        hp = np.asarray(self.public['hp'], dtype=np.float64)[ix]
        x = np.concatenate([z, hp], axis=1)
        xm = x.mean(axis=0)
        xs = np.maximum(x.std(axis=0, ddof=0), 1e-6)
        ym = float(y.mean())
        ys = float(max(y.std(ddof=0), 1e-8))
        data = _identity({'ids': ids, 'z': z.tolist(), 'hp': hp.tolist(), 'x_raw': x.tolist(), 'y': y.tolist(),
                          'measurement_se': se.tolist(), 'known_variance': (se ** 2).tolist(), 'validation_seeds': repeats,
                          'scaler': {'fit_ids': ids, 'x_mean': xm.tolist(), 'x_scale': xs.tolist(),
                                     'y_mean': ym, 'y_scale': ys, 'std_correction': 0, 'dtype': 'float64', 'frozen': True},
                          'x_standardized': ((x - xm) / xs).tolist(), 'y_standardized': ((y - ym) / ys).tolist(),
                          'known_variance_standardized': ((se / ys) ** 2).tolist(),
                          'metadata': {'subset_id': subset['subset_id'], 'fold': subset['fold'],
                                       'sample_seed': subset['sample_seed'], 'n': len(ids),
                                       'protocol_sha256': self.freeze['protocol_sha256'],
                                       'frozen_manifest_sha256': file_hash(self.freeze_path),
                                       'source_label_file_sha256': self.public['source_identity']['source_A_label_sha256'],
                                       'candidate_metadata': [self.public['candidate_metadata'][cid] for cid in ids],
                                       'labels_opened_at_utc': opened_at, 'frozen_at_utc': self.freeze['frozen_at_utc'],
                                       'access': access, 'query_labels_released': 0, 'query_SE_released': 0}})
        # A per-subset snapshot has only the labels permitted for that subset.
        # Repeated model seeds receive a fresh narrow dict, never a full-A map.
        snapshot = self.root / 'training' / (subset['subset_id'] + '.json')
        if not snapshot.exists():
            write_new(snapshot, data)
        return data

    def release_scoring(self, seal_path):
        """Call only after all executable fixed endpoints are uniformly sealed."""
        seal_path = Path(seal_path)
        seal = read_json(seal_path)
        if seal.get('sealed') is not True or seal.get('phase2_fit_status') not in ('COMPLETE', 'PARTIAL'):
            raise ValueError('Scoring requires a complete or explicitly partial prediction seal')
        if seal['protocol_sha256'] != self.freeze['protocol_sha256'] or seal['code_manifest_sha256'] != self.freeze['code_manifest_sha256']:
            raise ValueError('Prediction seal protocol/code identity mismatch')
        files = seal['files']
        if not isinstance(files, list) or seal['expected_endpoint_count'] != 200:
            raise ValueError('Prediction seal matrix schema mismatch')
        actual = [row['unit_id'] for row in files]
        expected = [row['unit_id'] for row in self.worklist['endpoints']]
        if len(actual) != len(set(actual)) or not set(actual) <= set(expected):
            raise ValueError('Duplicate or unknown prediction endpoint')
        if set(seal['expected_unit_ids']) != set(expected):
            raise ValueError('Prediction seal expected matrix differs from frozen worklist')
        if seal['phase2_fit_status'] == 'COMPLETE' and set(actual) != set(expected):
            raise ValueError('Cannot seal incomplete matrix as complete')
        prohibited = {'y', 'se', 'measurement_se', 'validation_mean', 'validation_se', 'validation_seeds', 'query_y', 'query_se', 'true_y'}

        def check_no_truth(value):
            if isinstance(value, dict):
                if prohibited & set(value):
                    raise ValueError('Prediction file contains query observation fields')
                for nested in value.values():
                    check_no_truth(nested)
            elif isinstance(value, list):
                for nested in value:
                    check_no_truth(nested)

        for row in files:
            path = _verified_path(self.root, row['path'], row['sha256'])
            check_no_truth(read_json(path))
        ids = self.public['ids']
        labels, access = self._source_labels(ids)
        validate_selected_labels(ids, labels)
        result = {'labels': labels, 'metadata': {'released_at_utc': now(),
                  'prediction_seal_path': str(seal_path), 'prediction_seal_sha256': file_hash(seal_path),
                  'source_A_label_sha256': self.public['source_identity']['source_A_label_sha256'],
                  'protocol_sha256': self.freeze['protocol_sha256'], 'purpose': 'post-seal independent scoring only',
                  'accuracy_unit': '0-1', 'SE_rule': 'sample std(ddof=1) of three GNN validation seeds divided by sqrt(3)',
                  'B_label_files_opened': 0, 'test_metric_files_opened': 0, **access}}
        write_new(self.root / 'scoring/labels_A.json', result)
        return result


def mock_checks():
    checks = stage01_mock_checks()
    rows = [{'candidate_id': 'q' + str(i), 'architecture_fingerprint': 'a' + str(i),
             'x_raw': [float(i + j) for j in range(16)], 'y': 'POISON_Y', 'se': 'POISON_SE'} for i in range(6)]
    q1 = query_projection(rows)
    for row in rows:
        row['y'], row['se'] = object(), object()
    q2 = query_projection(rows)
    if q1 != q2 or set(q1) != {'ids', 'architectures', 'x_raw'}:
        raise AssertionError('Query truth changed public projection')
    checks.update(query_poison_projection_invariant=True, query_fields_whitelisted=True,
                  standardization_scope='release_training computes statistics from selected x/y only',
                  actual_labels_read=0)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'mock', 'release-scoring'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR)
    parser.add_argument('--partition', type=Path, default=PARTITION_PATH)
    parser.add_argument('--seal', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        result = prepare_public(args.source_dir, args.partition, args.root)
    elif args.mode == 'mock':
        result = mock_checks()
    else:
        result = Stage2Adapter(args.root).release_scoring(args.seal)['metadata']
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
