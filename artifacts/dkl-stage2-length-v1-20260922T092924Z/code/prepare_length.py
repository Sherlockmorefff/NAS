"""A-only source snapshot and freeze; scoring labels are copied only after seal."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

SOURCE = Path('/home/fangxy/D-VAE/artifacts/dkl-stage2-v1-20260922T080157Z')
OLD_PROTOCOL = 'cea701e4aaa64974b8d80053ddfb3cd8af421ff86dda38014284b02d0c67aee0'
OLD_PACKAGE = '3898e1189ae17f9d056a2478e94ab4a7f1b6b7539b6aae5fd26153dd5b642717'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def prepare(root, source):
    import torch
    from stage01_core import tensor_hash
    if sha(source / 'protocol.json') != OLD_PROTOCOL or sha(source / 'evidence.tar.gz') != OLD_PACKAGE:
        raise ValueError('Original protocol/package identity differs')
    old_seal = read(source / 'predictions_manifest.json')
    if old_seal['phase2_status'] != 'COMPLETE_VALID' or len(old_seal['files']) != 200:
        raise ValueError('Original stage2 is not COMPLETE_VALID with 200 endpoints')
    old_hashes = dict(line.split('  ', 1)[::-1] for line in (source / 'FILES.sha256').read_text().splitlines())
    source_files = {}

    def copy(original, target):
        src, dst = source / original, root / target
        expected = old_hashes[original]
        if sha(src) != expected:
            raise ValueError('Original source file identity differs: ' + original)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(dst)
        shutil.copy2(src, dst)
        source_files[target] = expected

    for name in ['protocol.json', 'worklist.json', 'folds.json', 'subsets.json', 'public_inputs.json',
                 'predictions_manifest.json', 'code_manifest.json', 'environment_frozen.json']:
        copy(name, 'source/' + name)
    old_codes = read(source / 'code_manifest.json')['files']
    for name in ['stage01_core.py', 'stage2_core.py', 'score_stage2.py', 'source_reference/fuzzy_metric.py']:
        if sha(root / 'code' / name) != old_codes['code/' + name]:
            raise ValueError('Copied frozen math/scoring source changed: ' + name)
    worklist = read(root / 'source/worklist.json')
    contexts = worklist['contexts']
    if len(contexts) != 60:
        raise ValueError('Expected 60 original contexts')
    public = read(root / 'source/public_inputs.json')
    by_id = {cid: {'id': cid, 'architecture': arch, 'x': z + hp}
             for cid, arch, z, hp in zip(public['ids'], public['architectures'], public['z'], public['hp'])}
    old_predictions = {p['unit_id']: p for p in old_seal['files']}
    details = []
    for ctx in contexts:
        cid, sid = ctx['context_id'], ctx['subset_id']
        checkpoint = f'checkpoints/{cid}_DKL_joint.pt'
        predname = f'predictions/{cid}_DKL.json'
        training = f'training/{sid}.json'
        copy(checkpoint, 'source/' + checkpoint)
        copy(predname, 'source/' + predname)
        if 'source/' + training not in source_files:
            copy(training, 'source/' + training)
        pred, train = read(root / 'source' / predname), read(root / 'source' / training)
        if sha(root / 'source' / predname) != old_predictions[cid + '_DKL']['sha256']:
            raise ValueError('Original sealed prediction changed')
        state = torch.load(root / 'source' / checkpoint, map_location='cpu', weights_only=False)
        schedule = state['scheduler']
        if (schedule['mode'], schedule['planned_steps'], schedule['completed']) != ('joint', 1000, 1000):
            raise ValueError('Source checkpoint is not the 1000 joint endpoint')
        if state['fingerprint'] != pred['endpoint_state_hash'] or state['optimizer'] is None:
            raise ValueError('Checkpoint math state/optimizer differs from endpoint')
        for key, field in [('x', 'x_standardized'), ('y', 'y_standardized'), ('known', 'known_variance_standardized')]:
            if not torch.equal(state[key], torch.tensor(train[field], dtype=torch.double)):
                raise ValueError('Source training tensor mismatch: ' + key)
        if train['ids'] != train['scaler']['fit_ids'] or len(train['ids']) != ctx['n']:
            raise ValueError('Source scaler/ID scope differs')
        if digest(train['ids']) != ctx['selected_ids_sha256'] or digest(train['ids']) != pred['training_ids_sha256']:
            raise ValueError('Source subset identity differs')
        tensor_digest = tensor_hash({'x': state['x'], 'y': state['y'], 'known': state['known']})
        if tensor_digest != pred['training_tensor_sha256']:
            raise ValueError('Source prediction training tensor digest differs')
        query_name = f'source/queries/fold{ctx["fold"]}.json'
        query = {'ids': pred['candidate_ids'], 'architectures': pred['architecture_fingerprints'],
                 'x_raw': [by_id[c]['x'] for c in pred['candidate_ids']]}
        if (root / query_name).exists():
            if read(root / query_name) != query:
                raise ValueError('Query order differs within fold')
        else:
            write(root / query_name, query)
            source_files[query_name] = sha(root / query_name)
        details.append({**ctx, 'checkpoint_path': 'source/' + checkpoint,
                        'checkpoint_sha256': source_files['source/' + checkpoint],
                        'model_state_hash': state['fingerprint'], 'source_scheduler': schedule,
                        'prediction_path': 'source/' + predname, 'prediction_sha256': sha(root / 'source' / predname),
                        'training_path': 'source/' + training, 'training_sha256': sha(root / 'source' / training),
                        'training_ids_sha256': digest(train['ids']), 'training_tensor_sha256': tensor_digest,
                        'scaler': train['scaler'], 'scaler_sha256': digest(train['scaler']),
                        'query_path': query_name, 'source_actual_lrs': [g['lr'] for g in state['optimizer']['param_groups']]})
    for row in old_seal['files']:
        if row['unit_id'].endswith('_DKL'):
            continue
        copy(row['path'], 'source/controls/' + row['path'])
    manifest = {'source_root': str(source), 'source_protocol_sha256': OLD_PROTOCOL,
                'source_package_sha256': OLD_PACKAGE, 'files': source_files, 'contexts': details,
                'scoring_labels_expected_sha256': old_hashes['scoring/labels_A.json'],
                'scoring_labels_copied': False, 'new_updates_performed': 0,
                'only_A_selected_training_and_unlabeled_queries': True}
    write(root / 'source_manifest.json', manifest)
    write(root / 'worklist.json', {'contexts': contexts, 'context_count': 60, 'order': 'unchanged original',
                                  'scientific_steps': [1000, 2000, 4000], 'new_endpoints': 120})
    print(json.dumps({'prepared': True, 'contexts': len(details), 'source_files': len(source_files)}))


def freeze(root):
    if not read(root / 'verification/preflight_summary.json')['pass']:
        raise ValueError('Preflight must pass before freeze')
    protocol = {'task_id': root.name, 'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'scope': 'developmental length sensitivity after A results exposure; not independent confirmation',
                'source_protocol_sha256': OLD_PROTOCOL, 'source_package_sha256': OLD_PACKAGE,
                'source_completed_steps': 1000, 'continuation_target': 4000,
                'new_formal_updates_expected': 180000, 'new_endpoint_count_expected': 120,
                'scientific_endpoint_count_expected': 180, 'contexts': 60, 'scientific_steps': [1000, 2000, 4000],
                'diagnostic_steps': list(range(1000, 4001, 100)), 'diagnostic_rows_expected': 1860,
                'windows_expected': 1800, 'controls': 'original REG/FROZEN/RAW16 fixed budget, no additional training',
                'schedule': 'fixed_endpoint_lr', 'mapping_lr': 1e-4, 'gp_lr': 1e-3,
                'optimizer': 'restore full source Adam including ordered parameter mapping, per-parameter steps and moments',
                'scheduler': 'original planned1000 metadata preserved; new continuation metadata separate',
                'checkpoint2000': 'save only, same optimizer continues through4000',
                'objective': 'pure-data -ExactMarginalLogLikelihood; normalized by n once',
                'unchanged_math': 'frozen stage2 C8/ExactGP Matern5/2 isotropic; known noise plus extra; frozen head',
                'precision': {'dtype': 'float64', 'device': 'cpu', 'threads': 1, 'deterministic': True,
                              'jitter': 1e-8, 'cholesky_max_tries': 1, 'clip_global_norm': 5},
                'diagnostics': 'independent rebuilt model copy with caller RNG restored; all quantities post-update except gradient',
                'kernel': 'noise-free covariance; selected training then original query order',
                'prediction_sd': {'latent': 'y_scale*sqrt(max(latent_variance_std,0))',
                                  'model': 'y_scale*sqrt(max(latent_variance_std,0)+extra_variance_std)',
                                  'query_SE': 'not available to training/diagnostics'},
                'stability_thresholds': {'nll_relative': 1e-3, 'training_and_combined_kernel_relative': .01,
                                         'train_and_query_mean_max_raw': .001,
                                         'query_latent_and_model_sd_max_raw': .001, 'consecutive_windows': 3},
                'stability_baseline1000': 'historical prediction windows unavailable; null, not backfilled',
                'top5_stability': 'set unchanged all last3 windows; top1 separately',
                'selection': 'GP mean descending/candidate_id lexicographic; 5 candidates, no diversity',
                'negative_variance_std': {'failure_below': -1e-8, 'roundoff': 'retain raw, clamp only for sqrt'},
                'score_after_uniform_seal': True, 'score_steps': [1000, 2000, 4000],
                'regret_improvement': 'earlier regret5 minus later', 'gain_improvement': 'later gain5 minus earlier',
                'gain5': 'max(0,max(selected5_y)-max(selected_training_y))',
                'query_gain_upper_bound': 'max(0,max(current_query_y)-max(selected_training_y))',
                'random5': 'exact no-replacement same-query expected regret5 and gain5',
                'aggregation': 'model seeds within sample/fold; sample seeds within fold; equal5folds; n separate',
                'tie_tolerance_pp': 1e-9, 'statistics': 'descriptive only; no independent or significance claim',
                'budget_seconds': 1200, 'budget_includes': ['new training', 'diagnostics and baseline prediction',
                    'synthetic validation', 'restore verification', 'failures/retries'],
                'budget_excludes': ['old experiment', 'read-only audit', 'disk IO/packaging'],
                'stop_policy': 'per-step budget check, preserve and seal PARTIAL; INVALID on identity/leakage/numerical error',
                'no_early_stop_or_best_checkpoint_selection': True, 'next_stage_authorized': False}
    write(root / 'protocol.json', protocol)
    codes = {str(p.relative_to(root)): sha(p) for p in sorted((root / 'code').rglob('*.py'))}
    write(root / 'code_manifest.json', {'files': codes, 'source_head': '64a44c14316f11debaf01cf8583426acfe1410e6'})
    write(root / 'environment_frozen.json', {'python': sys.version, 'executable': sys.executable,
          'platform': platform.platform(), 'packages': {p: importlib.metadata.version(p) for p in ['torch', 'gpytorch', 'numpy']},
          'threads': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}})
    names = ['protocol.json', 'code_manifest.json', 'environment_frozen.json', 'source_manifest.json', 'worklist.json',
             'verification/preflight_summary.json']
    write(root / 'frozen_manifest.json', {'files': {name: sha(root / name) for name in names},
          'status': 'FROZEN', 'frozen_at_utc': protocol['frozen_at_utc'], 'new_training_started': False,
          'new_scoring_exposed': False})
    print(json.dumps({'frozen': True, 'protocol_sha256': sha(root / 'protocol.json'), 'source_manifest_sha256': sha(root / 'source_manifest.json')}))


def release(root, labels_from=None):
    seal = read(root / 'predictions_manifest.json')
    if not seal.get('sealed') or seal['execution_status'] not in ['COMPLETE_VALID', 'PARTIAL']:
        raise ValueError('Scoring labels require a valid COMPLETE/PARTIAL seal')
    for field, name in [('protocol_sha256', 'protocol.json'), ('source_manifest_sha256', 'source_manifest.json'),
                        ('code_manifest_sha256', 'code_manifest.json')]:
        if sha(root / name) != seal[field]:
            raise ValueError('Seal identity changed: ' + name)
    for row in seal['files']:
        for pathkey, hashkey in [('path', 'sha256'), ('state_path', 'state_file_sha256')]:
            if sha(root / row[pathkey]) != row[hashkey]:
                raise ValueError('Sealed file changed: ' + row[pathkey])
    source = read(root / 'source_manifest.json')
    labels = Path(labels_from) if labels_from is not None else Path(source['source_root']) / 'scoring/labels_A.json'
    if sha(labels) != source['scoring_labels_expected_sha256']:
        raise ValueError('Original A scoring labels changed')
    target = root / 'scoring/labels_A.json'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(labels, target)
    write(root / 'scoring/release.json', {'released_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'prediction_seal_sha256': sha(root / 'predictions_manifest.json'), 'labels_A_sha256': sha(target),
          'only_A': True, 'new_updates_after_release': False})
    print(json.dumps({'labels_released_after_new_seal': True}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'freeze', 'release-scoring'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--labels-from', type=Path)
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.root, a.source)
    elif a.mode == 'freeze':
        freeze(a.root)
    else:
        release(a.root, a.labels_from)
