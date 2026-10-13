"""Independent, read-only Stage-2 matrix audit; never fits or scores a model.

Run after prediction sealing. Reads only this task's public metadata, selected
training snapshots, states, logs, and predictions. Writes one audit JSON. It
does not import the training implementation or open scoring/historical labels.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

EXPECTED_PROTOCOL = 'cea701e4aaa64974b8d80053ddfb3cd8af421ff86dda38014284b02d0c67aee0'
A_IDS_SHA = 'd48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2'
RECIPE_SHA = '40bb99cbc3e33dcc417869a97d969b8e8e1c4ca7c6a5fad66c29ed85edea5978'
PUBLIC_SHA = {'features.json': '4aedd1482cdc36c055c4016a25fe4a8158c305321673e16e868b994ab830c797',
              'candidate_identity.json': 'c66c2aa70c4cf2d5fe7d6462bd822a35c4fdd64e950b326cfce8fcf013cab46e'}
A_LABEL_SHA = 'd6823a6dbe62507f4fdc53de01a69e33cdd3ec8de4fc0aae0cca89563766fa58'


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def semantic(value):
    require(value['identity'] == digest({k: v for k, v in value.items() if k != 'identity'}), 'Semantic manifest identity differs')


def close(actual, expected, name, atol=1e-12):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name + ' shape/nonfinite')
    require(np.allclose(a, b, rtol=0, atol=atol), name + ' differs')


def path_inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'Artifact path escaped task root')
    return path


def tensor_hash(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        result.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def state_hash(state):
    tensors = {'gp.' + key: value for key, value in state['gp'].items()}
    tensors.update({'data.x': state['x'], 'data.y': state['y'], 'data.known': state['known'],
                    'likelihood.fixed_noise': state['known']})
    if state['model_type'] == 'MAPPED8':
        tensors.update({'mapper.' + key: value for key, value in state['mapper'].items()})
    return tensor_hash(tensors)


def load_state(path, training, model_type):
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(state['model_type'] == state['model_kind'] == model_type, 'Wrong state input representation')
    require(state_hash(state) == state['fingerprint'], 'Saved mathematical state fingerprint differs')
    for name, source in [('x', 'x_standardized'), ('y', 'y_standardized'), ('known', 'known_variance_standardized')]:
        tensor = state[name]
        require(tensor.dtype == torch.float64 and tensor.device.type == 'cpu', 'Saved training tensors are not CPU float64')
        require(torch.equal(tensor, torch.tensor(training[source], dtype=torch.float64)), 'State training subset/scaler differs')
    n = len(training['ids'])
    width = 16 if model_type == 'RAW16' else 8
    require(tuple(state['history_embedding'].shape) == (n, width), 'Saved training history dimensions differ')
    require(torch.isfinite(state['history_embedding']).all(), 'Nonfinite saved history')
    if model_type == 'RAW16':
        require('mapper' not in state and torch.equal(state['history_embedding'], state['x']), 'RAW16 does not bypass mapping')
    else:
        require(all(torch.isfinite(value).all() for value in state['mapper'].values()), 'Nonfinite mapper state')
    for name, value in state['gp'].items():
        if name.endswith('upper_bound'):
            require(not torch.isnan(value).any(), 'NaN GP upper constraint')
        else:
            require(torch.isfinite(value).all(), 'Nonfinite GP state: ' + name)
    require(set(state['rng']) == {'torch', 'numpy', 'python'}, 'Missing RNG recovery state')
    return state


def audit(root, expected_protocol):
    results = {}
    protocol = read(root / 'protocol.json')
    frozen = read(root / 'frozen_manifest.json')
    require(sha(root / 'protocol.json') == expected_protocol == frozen['protocol_sha256'], 'Frozen protocol SHA mismatch')
    require(frozen['labels_released'] is False and protocol['next_stage_authorized'] is False, 'Frozen scope/gate differs')
    for name, value in frozen['files'].items():
        require(sha(path_inside(root, name)) == value, 'Frozen artifact changed: ' + name)
    codes = read(root / 'code_manifest.json')
    require(sha(root / 'code_manifest.json') == frozen['code_manifest_sha256'], 'Code manifest SHA differs')
    for name, value in codes['files'].items():
        require(sha(path_inside(root, name)) == value, 'Frozen source changed: ' + name)
    results['frozen_code_and_protocol'] = {'pass': True, 'source_file_count': len(codes['files'])}

    public = read(root / 'public_inputs.json')
    folds_doc = read(root / 'folds.json')
    subsets_doc = read(root / 'subsets.json')
    work = read(root / 'worklist.json')
    for value in (public, folds_doc, subsets_doc, work):
        semantic(value)
    ids, architectures = public['ids'], public['architectures']
    require(len(ids) == len(set(ids)) == 179 and len(set(architectures)) == 96, 'Legal A pool counts differ')
    require(digest(sorted(ids)) == A_IDS_SHA, 'A candidate identity changed')
    source = public['source_identity']
    for name, value in PUBLIC_SHA.items():
        require(source['public_sha256'][name] == value, 'Public source identity changed')
    require(source['source_A_label_sha256'] == A_LABEL_SHA, 'Frozen A label-source identity changed')
    require(source['recipe_sha256'] == digest(source['recipe']) == RECIPE_SHA, 'Training recipe changed')
    positions = {cid: i for i, cid in enumerate(ids)}
    arch = dict(zip(ids, architectures))
    folds = folds_doc['folds']
    require([f['fold'] for f in folds] == list(range(5)), 'Fold numbering differs')
    require([len(f['query_ids']) for f in folds] == [38, 36, 36, 34, 35], 'Query fold counts differ')
    require([len(f['training_pool_ids']) for f in folds] == [141, 143, 143, 145, 144], 'Training fold counts differ')
    query_counts = Counter()
    for f in folds:
        q, t = f['query_ids'], f['training_pool_ids']
        require(q == sorted(set(q)) and t == sorted(set(t)), 'Fold IDs duplicate or unsorted')
        require(set(t) == set(ids) - set(q), 'Training pool is not query complement')
        require({arch[c] for c in q}.isdisjoint(arch[c] for c in t), 'Architecture crosses query/train boundary')
        query_counts.update(q)
    require(dict(query_counts) == dict.fromkeys(ids, 1), 'Every A ID must be queried exactly once')
    require(folds_doc['source']['reused_original'] is True, 'Unexpected fallback fold grouping')
    require(folds_doc['source']['sha256'] == 'dba3ef021d1d13417b4eedbf83730a8862f4bc7de8a8b831ccb58cd115ab5643', 'Original partition hash changed')
    results['source_and_architecture_folds'] = {'pass': True, 'A_candidates': 179, 'A_architectures': 96,
                                              'query_counts': [38, 36, 36, 34, 35]}

    subsets = subsets_doc['subsets']
    require(len(subsets) == 20, 'Wrong selected subset count')
    contexts, expected_endpoints, expected_blocks = [], [], []
    for ss in (20260922, 20260923):
        for fold in range(5):
            ordered = np.random.default_rng(np.random.SeedSequence([ss, fold])).permutation(sorted(folds[fold]['training_pool_ids'])).tolist()
            for n in (20, 50):
                sid = f's{ss}_f{fold}_n{n}'
                selected = subsets[sid]
                require(selected['ids'] == ordered[:n] and selected['selected_ids_sha256'] == digest(ordered[:n]), 'Frozen sampling rule/order changed')
                require(selected['architecture_count'] == len({arch[c] for c in ordered[:n]}), 'Subset architecture count differs')
            require(subsets[f's{ss}_f{fold}_n20']['ids'] == subsets[f's{ss}_f{fold}_n50']['ids'][:20], 'n20 is not prefix of n50')
        for ms in (0, 1, 2):
            for fold in range(5):
                for n in (20, 50):
                    cid, sid = f's{ss}_m{ms}_f{fold}_n{n}', f's{ss}_f{fold}_n{n}'
                    context = {'context_id': cid, 'subset_id': sid, 'sample_seed': ss, 'model_seed': ms,
                               'fold': fold, 'n': n, 'run_raw': ms == 0}
                    contexts.append(context)
                    if ms == 0:
                        expected_endpoints.append(sid + '_RAW16')
                        expected_blocks.extend([(cid, 'RAW16', 'raw_initial', 'gp', 300), (cid, 'RAW16', 'raw_continue', 'gp', 1000)])
                    expected_blocks.extend([(cid, 'COMMON', 'regression', 'regression', 1000), (cid, 'COMMON', 'gp', 'gp', 300),
                                            (cid, 'FROZEN', 'gp', 'gp', 1000), (cid, 'REG', 'regression', 'regression', 1000),
                                            (cid, 'REG', 'gp', 'gp', 1000), (cid, 'DKL', 'joint', 'joint', 1000)])
                    expected_endpoints.extend(cid + '_' + method for method in ('FROZEN', 'REG', 'DKL'))
    require(len(work['contexts']) == 60 and len(work['endpoints']) == 200, 'Worklist matrix dimensions changed')
    for actual, expected in zip(work['contexts'], contexts):
        require(all(actual[key] == value for key, value in expected.items()), 'Frozen context order differs')
        require(actual['branch_order'] == ['FROZEN', 'REG', 'DKL'], 'Branch order changed')
        require(actual['selected_ids_sha256'] == subsets[actual['subset_id']]['selected_ids_sha256'], 'Methods do not share selected IDs')
    require([e['unit_id'] for e in work['endpoints']] == expected_endpoints, 'Endpoint ordering differs')
    results['subsets_and_run_order'] = {'pass': True, 'contexts': 60, 'subsets': 20, 'RAW16_independent_runs': 20}

    training = {}
    require(len(list((root / 'training').glob('*.json'))) == 20, 'Unexpected training snapshot count')
    for sid, subset in subsets.items():
        data = read(root / 'training' / (sid + '.json'))
        semantic(data)
        selected = subset['ids']
        n = subset['n']
        require(data['ids'] == selected and data['scaler']['fit_ids'] == selected, 'Scaler fit IDs include other candidates')
        ix = [positions[c] for c in selected]
        x = np.concatenate((np.asarray(public['z'])[ix], np.asarray(public['hp'])[ix]), axis=1)
        y, se = np.asarray(data['y']), np.asarray(data['measurement_se'])
        repeats = np.asarray(data['validation_seeds'])
        require(x.shape == (n, 16) and y.shape == se.shape == (n,) and repeats.shape == (n, 3), 'Selected training dimensions differ')
        require(np.all((repeats >= 0) & (repeats <= 1)) and np.all(se >= 0), 'Invalid selected validation observations')
        close(y, repeats.mean(axis=1), 'Training mean vs three-seed validation')
        close(se, repeats.std(axis=1, ddof=1) / math.sqrt(3), 'Training SE vs three-seed validation')
        xm, xs = x.mean(axis=0), np.maximum(x.std(axis=0, ddof=0), 1e-6)
        ym, ys = float(y.mean()), float(max(y.std(ddof=0), 1e-8))
        scaler = data['scaler']
        close(data['x_raw'], x, 'Selected public x')
        close(scaler['x_mean'], xm, 'Selected-only x center')
        close(scaler['x_scale'], xs, 'Selected-only x scale')
        close(scaler['y_mean'], ym, 'Selected-only y center')
        close(scaler['y_scale'], ys, 'Selected-only y scale')
        close(data['x_standardized'], (x - xm) / xs, 'Standardized x')
        close(data['y_standardized'], (y - ym) / ys, 'Standardized y')
        close(data['known_variance'], se ** 2, 'Known variance without extra divide-by-three')
        close(data['known_variance_standardized'], se ** 2 / ys ** 2, 'Standardized known variance')
        require(scaler['std_correction'] == 0 and scaler['frozen'] is True, 'Scaler ddof/frozen changed')
        metadata = data['metadata']
        require(metadata['source_label_file_sha256'] == A_LABEL_SHA and metadata['protocol_sha256'] == expected_protocol, 'Training source/protocol mismatch')
        require(metadata['access']['selected_label_rows_decoded'] == n and metadata['access']['hidden_label_rows_decoded'] == 0,
                'Training release decoded unselected label rows')
        require(metadata['query_labels_released'] == metadata['query_SE_released'] == 0, 'Query observation released to training')
        require(metadata['frozen_at_utc'] <= metadata['labels_opened_at_utc'], 'Labels released before freeze')
        training[sid] = data
    results['selected_training_and_scalers'] = {'pass': True, 'snapshots': 20, 'scope': 'selected20or50 only',
                                              'training_label_values_not_copied_into_audit': True}

    traces = read(root / 'training_traces.json')
    actual_trace_keys = [(r['run'], r['branch'], r['block'], r['step']) for r in traces]
    expected_trace_keys = [(cid, method, name, step) for cid, method, name, mode, steps in expected_blocks for step in range(100, steps + 1, 100)]
    require(actual_trace_keys == expected_trace_keys and len(traces) == 3440, 'Missing/reordered/duplicate block diagnostics')
    trace_map = {(r['run'], r['branch'], r['block'], r['step']): r for r in traces}
    require(len(list((root / 'diagnostics').glob('*_block.json'))) == len(expected_blocks) == 400, 'Wrong training block count')
    require({p.name for p in (root / 'checkpoints').glob('*.pt')} ==
            {f'{cid}_{method}_{name}.pt' for cid, method, name, mode, steps in expected_blocks},
            'Missing or extra recovery checkpoints')
    context_by_id = {c['context_id']: c for c in contexts}
    block_rows = {}
    checkpoint_states = {}
    total_steps = 0
    for cid, method, name, mode, steps in expected_blocks:
        key = (cid, method, name)
        row = read(root / 'diagnostics' / f'{cid}_{method}_{name}_block.json')
        require(row['complete'] is True and row['status'] == 'COMPLETE', 'Incomplete block reported in completed matrix')
        require(row['completed'] == row['updates'] == row['planned_steps'] == steps, 'Block endpoint/update count differs')
        require(row['context_id'] == cid and row['method'] == method and row['block'] == name and row['mode'] == mode, 'Block identity differs')
        require(row['optimizer_instances_this_call'] == 1 and row['call_updates'] == steps, 'Unexpected repeated/resumed optimizer block')
        require(math.isfinite(row['seconds']) and row['seconds'] >= 0, 'Invalid block time')
        for step in range(100, steps + 1, 100):
            trace = trace_map[(cid, method, name, step)]
            require(trace['planned_steps'] == steps and trace['mode'] == mode, 'Trace schedule context differs')
            multiplier = 0.1 if step - 1 >= int(0.8 * steps) else 1.0
            close(trace['lr_mapping'], (0.001 * multiplier if mode in ('regression', 'joint') else 0), 'Mapping learning-rate stage', 1e-15)
            close(trace['lr_gp'], (0.01 * multiplier if mode in ('gp', 'joint') else 0), 'GP learning-rate stage', 1e-15)
            for field in ('loss_before_update', 'gradient_before_clip', 'embedding_rms', 'mean', 'lengthscale', 'outputscale',
                          'extra_noise_variance_standardized', 'cumulative_seconds', 'block_seconds'):
                require(math.isfinite(trace[field]), 'Nonfinite training diagnostic: ' + field)
            require(trace['gradient_before_clip'] >= 0, 'Negative gradient norm')
            require(trace['embedding_dimensions'] == (16 if method == 'RAW16' else 8), 'Diagnostic representation width differs')
            require(trace['lengthscale_distance_to_lower'] >= -1e-14 and trace['outputscale_distance_to_lower'] >= -1e-14
                    and trace['extra_noise_distance_to_lower'] >= -1e-14, 'GP parameter below frozen lower bound')
        ctx = context_by_id[cid]
        state = load_state(path_inside(root, row['checkpoint']), training[ctx['subset_id']], 'RAW16' if method == 'RAW16' else 'MAPPED8')
        require(state['fingerprint'] == row['state_hash'] == trace_map[(cid, method, name, steps)]['state_after'], 'Checkpoint does not match fixed endpoint diagnostic')
        schedule = state['scheduler']
        require(schedule['completed'] == schedule['planned_steps'] == steps and schedule['block'] == name and schedule['mode'] == mode,
                'Checkpoint recovery schedule differs')
        require(schedule['next_lr_stage'] == 'final20', 'Checkpoint final learning-rate stage differs')
        optimizer = state['optimizer']
        require(optimizer and optimizer['state'], 'Checkpoint omits Adam state')
        require(all(float(value['step']) == steps for value in optimizer['state'].values()), 'Adam step count reveals block restart/missing update')
        bases = [0.001] if mode == 'regression' else ([0.01] if mode == 'gp' else [0.001, 0.01])
        require([group['base_lr'] for group in optimizer['param_groups']] == bases, 'Optimizer groups/base rates differ')
        for group in optimizer['param_groups']:
            close(group['lr'], group['base_lr'] * 0.1, 'Final Adam LR', 1e-15)
            require(tuple(group['betas']) == (0.9, 0.999) and group['eps'] == 1e-8 and group['weight_decay'] == 0, 'Adam recipe differs')
        total_steps += steps
        block_rows[key] = row
        checkpoint_states[key] = state
    require(total_steps == 344000, 'Formal update matrix total differs')
    results['fixed_blocks_logs_and_optimizer_schedule'] = {'pass': True, 'blocks': 400, 'formal_updates': 344000,
                                                          'diagnostic_rows': 3440, 'all_Adam_steps_match_full_blocks': True}

    common_states = {}
    require(len(list((root / 'states').glob('*_COMMON.pt'))) == 60, 'Wrong common-state count')
    require(len(list((root / 'diagnostics').glob('*_fork.json'))) == 60, 'Wrong fork-audit count')
    for ctx in contexts:
        cid, sid = ctx['context_id'], ctx['subset_id']
        state = load_state(root / 'states' / (cid + '_COMMON.pt'), training[sid], 'MAPPED8')
        fork = read(root / 'diagnostics' / (cid + '_fork.json'))
        require(fork['pass'] is True and fork['cross_model_storage_overlaps'] == [], 'Fork storage isolation check failed')
        require(fork['branch_fingerprints'] == [state['fingerprint']] * 3, 'Forks did not start from exact common state')
        require(fork['common_state_hash'] == fork['common_fingerprint'] == state['fingerprint'], 'Fork common fingerprint differs')
        require(state['fingerprint'] == checkpoint_states[(cid, 'COMMON', 'gp')]['fingerprint'], 'Common state not saved at fixed common GP endpoint')
        require(tensor_hash(state['mapper']) == tensor_hash(checkpoint_states[(cid, 'COMMON', 'regression')]['mapper']),
                'Common GP-only block changed the regression representation')
        common_states[cid] = state
    results['common_state_and_forks'] = {'pass': True, 'common_states': 60, 'exact_three_branch_start_checks': 60}

    seal = read(root / 'predictions_manifest.json')
    require(seal['sealed'] is True and seal['phase2_fit_status'] == 'COMPLETE' and seal['phase2_status'] == 'COMPLETE_VALID', 'Matrix is not sealed complete/valid')
    require(seal['failure'] is None and seal['missing_unit_ids'] == [], 'Sealed matrix contains failures or missing units')
    require(seal['expected_unit_ids'] == expected_endpoints and seal['expected_endpoint_count'] == 200, 'Seal expected matrix changed')
    require(seal['protocol_sha256'] == expected_protocol and seal['code_manifest_sha256'] == frozen['code_manifest_sha256'], 'Seal frozen-source identity differs')
    require(seal['worklist_sha256'] == sha(root / 'worklist.json'), 'Seal worklist hash differs')
    files = seal['files']
    require(len(files) == 200 and set(f['unit_id'] for f in files) == set(expected_endpoints), 'Sealed prediction endpoint set differs')
    require(len(list((root / 'predictions').glob('*.json'))) == 200, 'Unsealed extra prediction files')
    require({p.name for p in (root / 'states').glob('*.pt')} ==
            {unit + '.pt' for unit in expected_endpoints} | {ctx['context_id'] + '_COMMON.pt' for ctx in contexts},
            'Missing or extra common/endpoint state files')
    allowed = {'unit_id', 'context_id', 'subset_id', 'fold', 'sample_seed', 'n', 'model_seed', 'method',
               'candidate_ids', 'architecture_fingerprints', 'mean_standardized', 'mean_raw',
               'latent_variance_standardized', 'latent_variance_raw', 'y_center', 'y_scale',
               'extra_noise_variance_standardized', 'endpoint_state_hash', 'state_path', 'state_file_sha256',
               'training_ids_sha256', 'training_tensor_sha256', 'query_labels_or_SE_present',
               'negative_roundoff_variance_count', 'input_dimension', 'units', 'cost', 'finished_at_utc'}
    predictions = {}
    method_counts = Counter()
    negative_count = 0
    forbidden_truth = {'y', 'se', 'measurement_se', 'validation_mean', 'validation_se', 'validation_seeds',
                       'query_y', 'query_se', 'query_SE', 'true_y', 'true_mean', 'query_rank', 'champion'}

    def no_query_truth(value):
        if isinstance(value, dict):
            require(not (set(value) & forbidden_truth), 'Prediction contains a query truth/ranking field')
            for nested in value.values():
                no_query_truth(nested)
        elif isinstance(value, list):
            for nested in value:
                no_query_truth(nested)

    for entry in files:
        path = path_inside(root, entry['path'])
        require(sha(path) == entry['sha256'], 'Sealed prediction hash mismatch')
        p = read(path)
        require(set(p) == allowed and p['query_labels_or_SE_present'] is False, 'Unexpected prediction fields/query truth present')
        no_query_truth(p)
        require(all(p[key] == entry[key] for key in ('unit_id', 'context_id', 'subset_id', 'fold', 'sample_seed', 'n', 'model_seed', 'method')),
                'Prediction/manifest unit identity differs')
        ctx, subset = context_by_id[p['context_id']], subsets[p['subset_id']]
        require(p['fold'] == ctx['fold'] == subset['fold'] and p['n'] == ctx['n'] == subset['n'], 'Prediction uses wrong context/subset')
        require(p['sample_seed'] == ctx['sample_seed'] == subset['sample_seed'], 'Prediction sampling seed differs')
        require(p['candidate_ids'] == folds[p['fold']]['query_ids'], 'Prediction query pool differs')
        require(p['architecture_fingerprints'] == [arch[c] for c in p['candidate_ids']], 'Prediction query architecture mismatch')
        data = training[p['subset_id']]
        require(p['training_ids_sha256'] == digest(data['ids']) == subset['selected_ids_sha256'], 'Methods trained on different subsets')
        close(p['y_center'], data['scaler']['y_mean'], 'Prediction train-only y center')
        close(p['y_scale'], data['scaler']['y_scale'], 'Prediction train-only y scale')
        qn = len(p['candidate_ids'])
        for key in ('mean_standardized', 'mean_raw', 'latent_variance_standardized', 'latent_variance_raw'):
            require(np.asarray(p[key]).shape == (qn,) and np.isfinite(p[key]).all(), 'Nonfinite/misaligned prediction vector')
        variance = np.asarray(p['latent_variance_standardized'])
        require(np.all(variance >= -1e-8), 'Raw standardized latent variance violates numeric tolerance')
        close(p['mean_raw'], np.asarray(p['mean_standardized']) * p['y_scale'] + p['y_center'], 'Prediction mean units')
        close(p['latent_variance_raw'], variance * p['y_scale'] ** 2, 'Prediction variance units')
        require(p['negative_roundoff_variance_count'] == int((variance < 0).sum()), 'Negative variance roundoff count differs')
        negative_count += int((variance < 0).sum())
        require(math.isfinite(p['extra_noise_variance_standardized']) and p['extra_noise_variance_standardized'] >= 1e-6, 'Invalid extra noise variance')
        raw = p['method'] == 'RAW16'
        require(p['input_dimension'] == (16 if raw else 8), 'GP input dimensionality differs')
        require(p['model_seed'] is None if raw else p['model_seed'] == ctx['model_seed'], 'Wrong model seed or duplicated raw seed')
        require(ctx['model_seed'] == 0 if raw else True, 'Raw GP was rerun under a later model seed')
        spath = path_inside(root, p['state_path'])
        require(entry['state_path'] == p['state_path'], 'Prediction/seal endpoint state path differs')
        require(sha(spath) == p['state_file_sha256'] == entry['state_sha256'], 'Endpoint state file hash mismatch')
        state = load_state(spath, data, 'RAW16' if raw else 'MAPPED8')
        require(state['fingerprint'] == p['endpoint_state_hash'] == entry['endpoint_state_hash'], 'Prediction endpoint mathematical state differs')
        require(p['training_tensor_sha256'] == tensor_hash({'x': state['x'], 'y': state['y'], 'known': state['known']}), 'Prediction training tensor identity differs')
        final_block = 'raw_continue' if raw else ('joint' if p['method'] == 'DKL' else 'gp')
        require(state['fingerprint'] == checkpoint_states[(ctx['context_id'], p['method'], final_block)]['fingerprint'], 'Prediction not from final fixed checkpoint')
        if p['method'] == 'FROZEN':
            require(tensor_hash(state['mapper']) == tensor_hash(common_states[ctx['context_id']]['mapper']), 'FROZEN representation changed')
        if p['method'] == 'DKL':
            head = lambda s: {k: v for k, v in s['mapper'].items() if k.startswith('regressor.')}
            require(tensor_hash(head(state)) == tensor_hash(head(common_states[ctx['context_id']])), 'DKL regression head changed')
        if p['method'] == 'REG':
            require(tensor_hash(state['mapper']) == tensor_hash(checkpoint_states[(ctx['context_id'], 'REG', 'regression')]['mapper']), 'REG mapper changed during GP-only block')
        common_seconds = 0 if raw else sum(block_rows[(ctx['context_id'], 'COMMON', block)]['seconds'] for block in ('regression', 'gp'))
        branch_seconds = sum(row['seconds'] for (cid, method, block), row in block_rows.items() if cid == ctx['context_id'] and method == p['method'])
        cost = p['cost']
        close(cost['common_training_seconds'], common_seconds, 'Standalone common training cost', 1e-9)
        close(cost['branch_training_seconds'], branch_seconds, 'Branch training cost', 1e-9)
        require(math.isfinite(cost['prediction_seconds']) and cost['prediction_seconds'] >= 0, 'Invalid prediction cost')
        close(cost['standalone_fit_predict_seconds'], common_seconds + branch_seconds + cost['prediction_seconds'], 'Standalone total cost', 1e-9)
        predictions[p['unit_id']] = p
        method_counts[p['method']] += 1
    require(dict(method_counts) == {'RAW16': 20, 'FROZEN': 60, 'REG': 60, 'DKL': 60}, 'Raw/mapped endpoint counts differ')
    for ctx in contexts:
        cid = ctx['context_id']
        mapped = [predictions[cid + '_' + method] for method in ('FROZEN', 'REG', 'DKL')]
        require(len({p['cost']['common_training_seconds'] for p in mapped}) == 1, 'Shared costs were charged asymmetrically')
        require(len({p['training_tensor_sha256'] for p in mapped}) == 1, 'Methods received different training tensors')
        require(predictions[ctx['subset_id'] + '_RAW16']['training_tensor_sha256'] == mapped[0]['training_tensor_sha256'], 'Raw16 used different training tensors')
    results['sealed_predictions_and_endpoint_states'] = {'pass': True, 'endpoints': 200,
                                                        'method_counts': dict(method_counts),
                                                        'negative_roundoff_variances': negative_count,
                                                        'query_truth_fields': 0, 'shared_cost_symmetry': True}

    ledger = read(root / 'resource_ledger.json')
    preflight = read(root / 'verification/preflight_summary.json')
    resources = seal['resources']
    require(ledger['updates'] == resources['formal_optimizer_updates'] == total_steps == 344000, 'Resource updates differ')
    require(ledger['failed_training_attempts'] == 0 and ledger['predictions'] == 260, 'Unexpected failed attempts/prediction count')
    close(ledger['training_seconds'], sum(r['seconds'] for r in block_rows.values()), 'Training block cost sum vs ledger', 1e-6)
    close(ledger['seconds'], ledger['training_seconds'] + ledger['prediction_seconds'] + ledger['verification_seconds'], 'Total budget sum', 1e-6)
    close(ledger['verification_seconds'], preflight['verification_seconds'], 'Verification budget inclusion', 1e-9)
    require(ledger['verification_updates'] == preflight['synthetic_optimizer_updates'], 'Verification update count differs')
    for source_key, seal_key in [('training_seconds', 'training_seconds'), ('prediction_seconds', 'prediction_seconds'),
                                 ('verification_seconds', 'verification_seconds'), ('seconds', 'budget_used_seconds')]:
        close(ledger[source_key], resources[seal_key], 'Seal resource ledger consistency', 1e-6)
    require(ledger['limit'] == resources['budget_seconds'] == 7200 and ledger['seconds'] < 7200, 'Complete matrix exceeded resource limit')
    require(ledger['prediction_seconds'] >= sum(p['cost']['prediction_seconds'] for p in predictions.values()), 'Prediction accounting omits endpoint costs')
    for name in ('failure.json', 'matrix_failure.json'):
        require(not (root / name).exists(), 'Unexpected preserved failure requires explicit review')
    results['resources_and_costs'] = {'pass': True, 'formal_optimizer_updates': 344000,
                                    'verification_optimizer_updates': ledger['verification_updates'],
                                    'training_seconds': ledger['training_seconds'], 'prediction_seconds': ledger['prediction_seconds'],
                                    'verification_seconds': ledger['verification_seconds'], 'budget_used_seconds': ledger['seconds'],
                                    'experiment_wall_seconds': resources['experiment_wall_seconds'],
                                    'standalone_common_cost_included_for_all_mapped_methods': True,
                                    'failed_training_attempts': 0}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha256', default=EXPECTED_PROTOCOL)
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.monotonic()
    torch.set_num_threads(1)
    result = {'pass': False, 'audit': 'independent saved matrix integrity, no fitting/prediction/scoring',
              'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'expected_protocol_sha256': args.expected_protocol_sha256, 'root': str(root),
              'scoring_labels_opened': 0, 'historical_label_files_opened': 0,
              'B_label_files_opened': 0, 'test_metric_files_opened': 0,
              'formal_training_started': False, 'model_predictions_recomputed': False}
    try:
        result['checks'] = audit(root, args.expected_protocol_sha256)
        result['pass'] = True
    except Exception as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    result['audit_wall_seconds'] = time.monotonic() - started
    result['audit_source_sha256'] = sha(__file__)
    output = root / 'audit/matrix_integrity.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'pass': result['pass'], 'checks': list(result.get('checks', {})),
                      'error': result.get('error', {}).get('message'), 'output': str(output),
                      'audit_wall_seconds': result['audit_wall_seconds']}, ensure_ascii=False))
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
