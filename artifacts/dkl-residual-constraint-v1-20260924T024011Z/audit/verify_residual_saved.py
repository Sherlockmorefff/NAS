#!/usr/bin/env python3
"""Independent saved-evidence audit. No model imports, initialization or execution.

Run only after COMPLETE_VALID sealing. Reads trusted local tensor checkpoints,
public/selected-training inputs and unlabelled predictions; never opens scoring
labels or scored A/B tables. Tensor equality/hash and saved-array statistics only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

GP_NAMES = ['gp.likelihood.second_noise_covar.raw_noise', 'gp.mean_module.raw_constant',
            'gp.covar_module.raw_outputscale', 'gp.covar_module.base_kernel.raw_lengthscale']
MAP_NAMES = ['mapper.W1.weight', 'mapper.W1.bias', 'mapper.W2.weight']
HEAD_NAMES = ['mapper.head.weight', 'mapper.head.bias']
METHODS = ['R_FROZEN', 'R_REG', 'R_DKL_C', 'R_DKL_U']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def near(a, b, message, atol=1e-9, rtol=1e-9):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol), message)


def tensor_hash(values):
    out = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        out.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        out.update(value.numpy().tobytes())
    return out.hexdigest()


def exact_tensors(a, b, message):
    require(a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]) for k in a), message)


def fingerprint(state):
    tensors = {'gp.' + k: v for k, v in state['gp'].items()}
    tensors.update({'mapper.' + k: v for k, v in state.get('mapper', {}).items()})
    tensors.update({'data.x': state['x'], 'data.y': state['y'], 'data.known': state['known'],
                    'likelihood.fixed_noise': state['known']})
    return tensor_hash(tensors)


def finite_tensor(value):
    return bool(torch.isfinite(value).all())


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(b-a) / max(float(np.linalg.norm(a)), 1e-30))


def stable_window(a, b):
    """Independent formula implementation on saved values; no GP operations."""
    require(b['block_step'] - a['block_step'] == 100, 'Window spacing')
    require(a['query_ids'] == b['query_ids'] and a['training_ids'] == b['training_ids'], 'Window identities')
    da, db = float(a['nll_after_update']), float(b['nll_after_update'])
    oa, ob = a['gp_parameters']['outputscale'], b['gp_parameters']['outputscale']
    def maximum(key):
        return float(np.max(np.abs(np.asarray(b[key])-np.asarray(a[key]))))
    metrics = {'nll_relative_change': abs(db-da)/max(1., abs(da)), 'nll_signed_change': db-da,
        'training_kernel_relative_change': relative(a['training_kernel'], b['training_kernel']),
        'combined_kernel_relative_change': relative(a['combined_kernel'], b['combined_kernel']),
        'training_correlation_relative_change': relative(np.asarray(a['training_kernel'])/oa, np.asarray(b['training_kernel'])/ob),
        'combined_correlation_relative_change': relative(np.asarray(a['combined_kernel'])/oa, np.asarray(b['combined_kernel'])/ob),
        'outputscale_relative_change': abs(ob-oa)/max(abs(oa), 1e-30), 'outputscale_signed_change': ob-oa,
        'train_mean_max_abs_change_raw': maximum('train_mean_raw'),
        'query_mean_max_abs_change_raw': maximum('query_mean_raw'),
        'query_latent_sd_max_abs_change_raw': maximum('query_latent_sd_raw'),
        'query_model_sd_max_abs_change_raw': maximum('query_model_sd_raw')}
    criteria = {'objective_small': metrics['nll_relative_change'] <= .001,
        'similarity_stable': max(metrics['training_kernel_relative_change'], metrics['combined_kernel_relative_change']) <= .01,
        'mean_prediction_stable': max(metrics['train_mean_max_abs_change_raw'], metrics['query_mean_max_abs_change_raw']) <= .001,
        'uncertainty_stable': max(metrics['query_latent_sd_max_abs_change_raw'], metrics['query_model_sd_max_abs_change_raw']) <= .001}
    return {**metrics, 'criteria': criteria, 'practical_stability_window': all(criteria.values()),
            'top5_selection_stable': set(a['top5_ids']) == set(b['top5_ids']),
            'top5_order_stable': a['top5_ids'] == b['top5_ids'],
            'top1_selection_stable': a['top1_id'] == b['top1_id']}


class Audit:
    def __init__(self, root):
        self.root = root
        self.failures, self.passed, self.states, self.summaries = [], [], {}, {}
        self.diagnostic_count = self.trace_rows = self.window_count = self.hash_files = 0

    def check(self, name, function):
        try:
            result = function()
            self.passed.append(name)
            return result
        except Exception as exc:
            self.failures.append({'check': name, 'type': type(exc).__name__, 'message': str(exc)})
            return None

    def child(self, relative):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root.resolve()), 'Path escapes artifact: ' + str(relative))
        return path

    def checked_hash(self, relative, expected):
        path = self.child(relative)
        require(sha(path) == expected, 'Hash mismatch: ' + str(relative))
        self.hash_files += 1
        return path

    def load_state(self, key):
        if key not in self.states:
            self.states[key] = torch.load(self.root/'checkpoints'/f'{key}.pt', map_location='cpu', weights_only=False)
        return self.states[key]

    def validate_state(self, state, training):
        require(state['model_type'] in ('RESIDUAL16', 'RAW16'), 'Model type')
        require(state['model_kind'] == state['model_type'], 'Model kind')
        config = state['configuration']
        require(digest(config) == state['configuration_hash'], 'Configuration hash')
        require(config['model_type'] == state['model_type'] and config['constraint_mode'] == state['constraint_mode'], 'Configuration identity')
        for k, source in [('x', 'x_standardized'), ('y', 'y_standardized'), ('known', 'known_variance_standardized')]:
            expected = torch.tensor(training[source], dtype=torch.float64)
            require(state[k].dtype == torch.float64 and torch.equal(state[k], expected), 'Selected training tensor changed: ' + k)
        require(tuple(state['x'].shape) == (len(training['ids']), 16), 'Training tensor shape')
        require(tuple(state['history_embedding'].shape) == tuple(state['x'].shape) and
                finite_tensor(state['history_embedding']), 'Saved current embedding shape/finite')
        require(all(finite_tensor(v) for v in state.get('mapper', {}).values()), 'Nonfinite mapping weights')
        # GP lower-bound buffers may legitimately include +inf upper bounds.
        require(all(finite_tensor(v) for k, v in state['gp'].items() if k.endswith(('raw_noise', 'raw_constant', 'raw_outputscale', 'raw_lengthscale'))), 'Nonfinite GP parameters')
        require(fingerprint(state) == state['fingerprint'], 'Mathematical state fingerprint')
        if state['model_type'] == 'RAW16':
            require('mapper' not in state and torch.equal(state['history_embedding'], state['x']), 'RAW must be direct x16')
        else:
            require(config['regression_objective'] == 'standardized_y_pure_mse_v1', 'MSE objective identity')
            near(config['spectral_cap'], math.sqrt(.5), 'Spectral cap')
            require(sum(v.numel() for k, v in state['mapper'].items() if not k.startswith('head.')) == 264, 'Mapping parameter count')
            require(sum(v.numel() for k, v in state['mapper'].items() if k.startswith('head.')) == 17, 'Head parameter count')

    def diagnostic(self, key, step, training, query, matrices=False):
        record = read(self.root/'diagnostics'/key/f'j{step}.json')
        require(record['block_key'] == key and record['block_step'] == step, 'Diagnostic block/step')
        require(record['query_ids'] == query['ids'] and record['training_ids'] == training['ids'], 'Diagnostic IDs')
        require(record['query_labels_or_SE_present'] is False and record['diagnostic_model_isolated'] is True, 'Diagnostic isolation/truth declaration')
        require(finite_tree(record), 'Nonfinite serialized diagnostic')
        spectral = record['spectral']
        if spectral['constraint_mode'] == 'bounded':
            require(max(spectral['w1_spectral_norm'], spectral['w2_spectral_norm']) <= math.sqrt(.5)+1e-10 and
                    spectral['residual_lipschitz_upper_bound'] <= .5+1e-10 and spectral['within_bounded_limits'], 'Saved measured bounded norms')
        if step:
            row = record['last_update']
            require(row['step'] == step and row['block'] == key and row['learning_rates'] == record['lrs'], 'Update/diagnostic temporal attribution')
        if record['mode'] != 'regression':
            variance = np.asarray(record['query_latent_variance_standardized'])
            require(len(variance) == len(query['ids']) and np.isfinite(variance).all() and variance.min() >= -1e-8, 'Raw latent variance')
            if matrices:
                path = self.checked_hash(record['kernel_arrays_path'], record['kernel_arrays_sha256'])
                with np.load(path, allow_pickle=False) as z:
                    require(set(z.files) == {'training_kernel', 'combined_kernel', 'training_correlation', 'combined_correlation'}, 'K/R array keys')
                    for name in z.files:
                        values = np.array(z[name], copy=True)
                        n = len(training['ids']) + (len(query['ids']) if name.startswith('combined') else 0)
                        require(values.shape == (n, n) and np.isfinite(values).all(), 'K/R dimensions/finite')
                        record[name] = values
                out = record['gp_parameters']['outputscale']
                for part in ('training', 'combined'):
                    require(np.allclose(record[part+'_kernel']/out, record[part+'_correlation'], rtol=1e-12, atol=1e-12), 'K/R separation')
        return record

    def block(self, key, mode, steps, training, query, constraint, final=False):
        summary = read(self.root/'block_summaries'/f'{key}.json')
        self.checked_hash(summary['checkpoint'], summary['checkpoint_sha256'])
        require(summary['block_key'] == key and summary['mode'] == mode and summary['planned_steps'] == summary['completed'] == steps, 'Summary endpoint')
        state = self.load_state(key)
        self.validate_state(state, training)
        require(state['constraint_mode'] == constraint, 'Block constraint mode')
        schedule = state['scheduler']
        require(schedule['block'] == key and schedule['mode'] == mode and schedule['completed'] == schedule['planned_steps'] == steps, 'Checkpoint block schedule')
        require(schedule['policy'] == 'first80base_last20base_div10' and schedule['next_lr_stage'] == 'last20', 'Fixed LR policy')
        account = state['accounting']
        require(account['completed_updates'] == steps, 'Block accounting update count')
        near(account['block_training_seconds'], summary['training_seconds'], 'Block training cost')
        near(state['runner_block_model_seconds'], summary['model_seconds'], 'Block model cost')
        require(summary['model_seconds'] >= summary['training_seconds'] >= 0, 'Nonnegative model/training cost')
        require(summary['math_state_hash'] == state['fingerprint'], 'Summary mathematical identity')
        require(account['budget_updates_after_last_step'] == state['runner_budget_at_save']['updates'], 'Snapshot global update stamp')
        expected_names = ([MAP_NAMES+HEAD_NAMES] if mode == 'regression' else
                          [MAP_NAMES, GP_NAMES] if mode == 'joint' else [GP_NAMES])
        require(state['ordered_parameter_names'] == expected_names, 'Ordered Adam parameter names')
        optimizer = state['optimizer']
        require(len(optimizer['param_groups']) == len(expected_names), 'Adam group count')
        parameters = {'mapper.'+k: v for k, v in state.get('mapper', {}).items()}
        parameters.update({'gp.'+k: v for k, v in state['gp'].items()})
        seen = []
        for group, names in zip(optimizer['param_groups'], expected_names):
            is_gp = names == GP_NAMES
            base = .01 if is_gp else .001
            require(group['name'] == ('gp' if is_gp else 'mapping_head' if mode == 'regression' else 'mapping'), 'Adam group name')
            near(group['base_lr'], base, 'Base LR'); near(group['lr'], base/10, 'Final actual LR')
            require(tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0, 'Adam hyperparameters')
            require(len(group['params']) == len(names), 'Adam parameter/name alignment')
            for index, name in zip(group['params'], names):
                seen.append(index)
                entry = optimizer['state'][index]
                require(float(entry['step']) == steps, 'Per-parameter Adam step: ' + name)
                for moment in ('exp_avg', 'exp_avg_sq'):
                    require(entry[moment].shape == parameters[name].shape and finite_tensor(entry[moment]), 'Adam moment shape/finite: ' + name)
                require(bool((entry['exp_avg_sq'] >= 0).all()), 'Negative second moment')
        require(len(set(seen)) == len(seen) and set(seen) == set(optimizer['state']), 'Unique complete Adam ownership')
        rows = [json.loads(line) for line in (self.root/'diagnostics'/key/'updates.jsonl').read_text().splitlines() if line]
        require([r['step'] for r in rows] == list(range(100, steps+1, 100)), 'Every-100 update trace completeness')
        require(rows[-1] == state['last_update'], 'Final update/checkpoint identity')
        for row in rows:
            require(row['block'] == key and row['mode'] == mode and finite_tree(row), 'Trace metadata/finite')
            fraction = .1 if row['step'] > int(.8*steps) else 1.
            for name, lr in row['learning_rates'].items():
                near(lr, (.01 if name == 'gp' else .001)*fraction, 'Observed absolute step LR')
            evaluated = constraint == 'bounded' and mode in ('regression', 'joint')
            require(row['projection']['projection_evaluated'] == evaluated, 'Projection policy')
        self.trace_rows += len(rows)
        records = {}
        for step in range(0, steps+1, 100):
            records[step] = self.diagnostic(key, step, training, query, matrices=final and step >= 700)
            self.diagnostic_count += 1
        require(records[steps]['state_hash'] == state['fingerprint'], 'Final diagnostic/checkpoint identity')
        if final:
            saved = summary['stability']
            require(saved['windows_available'] == saved['windows_required'] == 3 and saved['missing_diagnostic_steps'] == [], 'Three final windows')
            require([(r['from_step'], r['to_step']) for r in saved['windows']] == [(700,800),(800,900),(900,1000)], 'Final window positions')
            calculated = []
            for index, (a, b) in enumerate(zip((700,800,900), (800,900,1000))):
                value = stable_window(records[a], records[b]); calculated.append(value)
                original = saved['windows'][index]
                for name, number in value.items():
                    if isinstance(number, float): near(number, original[name], 'Saved stability metric: '+name)
                    else: require(number == original[name], 'Saved stability criterion: '+name)
                self.window_count += 1
            criteria = {k: all(w['criteria'][k] for w in calculated) for k in calculated[0]['criteria']}
            require(saved['criteria'] == criteria and saved['practical_stability'] == all(criteria.values()), 'Endpoint stability aggregate')
            for name in ('top5_selection_stable', 'top5_order_stable', 'top1_selection_stable'):
                require(saved[name] == all(w[name] for w in calculated), 'Endpoint selection stability')
        self.summaries[key] = summary
        return state

    def context(self, ctx):
        cid, sid = ctx['context_id'], ctx['subset_id']
        training = read(self.root/'training'/f'{sid}.json')
        query = read(self.root/'queries'/f'fold{ctx["fold"]}.json')
        require(set(query) == {'ids','architectures','x_raw'}, 'Public query schema')
        require(not set(training['ids']) & set(query['ids']), 'Train/query identity overlap')
        initial = torch.load(self.root/'states'/f'{cid}_INITIAL.pt', map_location='cpu', weights_only=False)
        self.validate_state(initial, training)
        require(torch.count_nonzero(initial['mapper']['W2.weight']) == 0 and torch.count_nonzero(initial['mapper']['W1.bias']) == 0, 'Identity residual initialization')
        require(torch.equal(initial['history_embedding'], initial['x']), 'Initial h=x saved identity')
        reg = self.block(cid+'_COMMON_regression', 'regression',1000,training,query,'bounded')
        common = self.block(cid+'_COMMON_gp','gp',300,training,query,'bounded')
        exact_tensors(initial['gp'], reg['gp'], 'Common MSE changed GP')
        exact_tensors(reg['mapper'],common['mapper'],'Common GP changed mapping/head')
        fork = read(self.root/'diagnostics'/f'{cid}_fork.json')
        require(fork['pass'] and fork['same_mathematical_state'] and fork['cross_model_overlaps'] == [] and
                fork['common_state_hash'] == common['fingerprint'], 'Four-clone isolation evidence')
        for method in METHODS:
            mode = 'joint' if method.startswith('R_DKL') else 'gp'
            constraint = 'unbounded' if method == 'R_DKL_U' else 'bounded'
            source = common
            if method == 'R_REG':
                source = self.block(cid+'_'+method+'_regression','regression',1000,training,query,'bounded')
                exact_tensors(common['gp'],source['gp'],'R_REG MSE changed common GP')
                zero = read(self.root/'diagnostics'/f'{cid}_{method}_regression/j0.json')
                require(zero['state_hash'] == common['fingerprint'], 'R_REG common start')
            key = cid+'_'+method+'_'+mode
            final = self.block(key,mode,1000,training,query,constraint,True)
            zero = read(self.root/'diagnostics'/key/'j0.json')
            require(zero['state_hash'] == source['fingerprint'], 'Branch start state')
            require(zero['spectral']['constraint_mode'] == constraint, 'Branch start constraint')
            require(fork['branch_configuration_hashes'][method] == final['configuration_hash'], 'Branch configuration identity')
            if mode == 'gp': exact_tensors(source['mapper'], final['mapper'], 'Frozen GP block changed mapper/head')
            else:
                exact_tensors({k:v for k,v in common['mapper'].items() if k.startswith('head.')},
                              {k:v for k,v in final['mapper'].items() if k.startswith('head.')}, 'Joint branch changed head')
        require(fork['branch_configuration_hashes']['R_DKL_C'] != fork['branch_configuration_hashes']['R_DKL_U'], 'C/U configuration distinction')
        c = self.states[cid+'_R_DKL_C_joint']['configuration'].copy()
        u = self.states[cid+'_R_DKL_U_joint']['configuration'].copy()
        c.pop('constraint_mode'); u.pop('constraint_mode')
        require(c == u, 'C/U differ beyond constraint mode')
        if ctx['run_raw']:
            self.block(cid+'_RAW16_initial','gp',300,training,query,'none')
            raw = self.block(cid+'_RAW16_final','gp',1000,training,query,'none',True)
            zero = read(self.root/'diagnostics'/f'{cid}_RAW16_final/j0.json')
            require(zero['state_hash'] == self.states[cid+'_RAW16_initial']['fingerprint'], 'RAW phase continuity')
            check = read(self.root/'diagnostics'/f'{sid}_RAW16_historical_consistency.json')
            old_path = self.checked_hash(f'source/stage2/predictions/{sid}_RAW16.json', check['source_sha256'])
            require(check['pass'] and check['new_fit_not_substitution'], 'RAW historical consistency declaration')
            old, new = read(old_path), read(self.root/'predictions'/f'{sid}_RAW16.json')
            for name in ('mean_standardized','latent_variance_standardized'):
                require(np.allclose(old[name],new[name],rtol=1e-6,atol=1e-8), 'RAW historical numeric equality')
                near(max(abs(a-b) for a,b in zip(old[name],new[name])), check['maximum_absolute_difference'][name], 'RAW difference report')

    def endpoints(self, seal, work):
        expected = {r['unit_id']:r for r in work['endpoints']}
        require(len(expected) == len(work['endpoints']) == 260, 'Expected endpoint cardinality')
        require(len(seal['files']) == 260 and set(seal['expected_unit_ids']) == set(expected), 'Sealed endpoint cardinality')
        require({p.stem for p in (self.root/'predictions').glob('*.json')} == set(expected), 'Prediction directory exact identities')
        seen = set()
        for item in seal['files']:
            uid = item['unit_id']; require(uid in expected and uid not in seen, 'Duplicate/extra sealed endpoint'); seen.add(uid)
            p = read(self.checked_hash(item['path'], item['sha256']))
            self.checked_hash(item['state_path'], item['state_sha256'])
            require(p['state_path'] == item['state_path'] and p['state_file_sha256'] == item['state_sha256'], 'Endpoint checkpoint identity')
            for name in ('unit_id','subset_id','method','fold','sample_seed','model_seed','n'):
                require(p[name] == item[name] == expected[uid][name], 'Endpoint matrix coordinate: '+name)
            key = Path(p['state_path']).stem; state = self.load_state(key)
            require(p['endpoint_state_hash'] == item['endpoint_state_hash'] == state['fingerprint'], 'Endpoint model fingerprint')
            training = read(self.root/'training'/f'{p["subset_id"]}.json'); query = read(self.root/'queries'/f'fold{p["fold"]}.json')
            require(p['candidate_ids'] == query['ids'] and p['architecture_fingerprints'] == query['architectures'], 'Endpoint public query identity')
            require(p['query_labels_or_SE_present'] is False and p['input_dimension'] == 16, 'Endpoint query truth/dimension declaration')
            require(p['training_ids_sha256'] == digest(training['ids']) and p['scaler_sha256'] == digest(training['scaler']), 'Endpoint selected IDs/scaler')
            require(p['training_tensor_sha256'] == tensor_hash({k:state[k] for k in ('x','y','known')}), 'Endpoint selected tensor hash')
            require(p['completed_steps'] == (1300 if p['method']=='RAW16' else 3300 if p['method']=='R_REG' else 2300), 'Endpoint cumulative steps')
            d = read(self.root/'diagnostics'/key/'j1000.json')
            require(p['mean_raw'] == d['query_mean_raw'] and p['latent_variance_standardized'] == d['query_latent_variance_standardized'] and p['geometry_row'] == d['geometry_row'], 'Endpoint reuses exact final diagnostic')
            require(finite_tree(p), 'Finite endpoint data')
            variance = np.asarray(p['latent_variance_standardized']); require(variance.min() >= -1e-8,'Endpoint negative variance')
            require(p['negative_roundoff_variance_count'] == int((variance<0).sum()),'Negative roundoff count')
            scale = training['scaler']; near(p['y_scale'],scale['y_scale'],'Target scale'); near(p['y_center'],scale['y_mean'],'Target center')
            require(np.allclose(p['latent_variance_raw'],variance*scale['y_scale']**2,rtol=1e-12,atol=1e-14),'Variance units')
            require(np.allclose(p['mean_standardized'],(np.asarray(p['mean_raw'])-scale['y_mean'])/scale['y_scale'],rtol=1e-12,atol=1e-12),'Mean units')
            near(p['extra_noise_variance_standardized'],d['gp_parameters']['extra_noise_variance_standardized'],'Extra variance units')
            cost=p['cost']; near(cost['standalone_fit_predict_seconds'],cost['common_model_seconds']+cost['branch_model_seconds'],'Endpoint model cost sum')
            cid=p['context_id']; common_keys=[] if p['method']=='RAW16' else [cid+'_COMMON_regression',cid+'_COMMON_gp']
            branch_keys=[key]
            if p['method']=='RAW16':branch_keys.insert(0,cid+'_RAW16_initial')
            if p['method']=='R_REG':branch_keys.insert(0,cid+'_R_REG_regression')
            near(cost['common_training_seconds'],sum(self.summaries[k]['training_seconds'] for k in common_keys),'Endpoint common training cost')
            near(cost['branch_training_seconds'],sum(self.summaries[k]['training_seconds'] for k in branch_keys),'Endpoint branch training cost')
            require(cost['common_model_seconds']+1e-9 >= sum(self.summaries[k]['model_seconds'] for k in common_keys),'Endpoint shared model cost coverage')
            require(cost['branch_model_seconds']+1e-9 >= sum(self.summaries[k]['model_seconds'] for k in branch_keys),'Endpoint branch model cost coverage')

    def run(self):
        started=time.perf_counter()
        seal=read(self.root/'predictions_manifest.json')
        # Refuse incomplete/in-progress input without writing a spurious FAIL audit.
        require(seal.get('sealed') is True and seal.get('execution_status') == 'COMPLETE_VALID', 'Audit requires completed valid B seal')
        work=read(self.root/'worklist.json'); ledger=read(self.root/'resource_ledger.json')
        self.check('exclusive_fresh_entry',lambda:require((self.root/'formal_run_once.lock').is_dir(),'Permanent exclusive run-once lock missing'))
        def manifests():
            for name in ('frozen_manifest.json','code_manifest.json','source_manifest.json'):
                for relative, expected in read(self.root/name)['files'].items():self.checked_hash(relative,expected)
            for name in ('protocol','source_manifest','code_manifest','worklist'):
                self.checked_hash(name+'.json',seal[name+'_sha256'])
            require(seal['missing_unit_ids'] == [] and seal['failure'] is None,'Complete seal has missing/failure')
        self.check('frozen_source_code_and_seal_hashes',manifests)
        contexts=work['contexts']
        self.check('matrix_contexts',lambda:require(len(contexts)==60 and len({r['context_id'] for r in contexts})==60 and
                   sum(bool(r['run_raw']) for r in contexts)==20,'Expected 60 unique contexts / 20 RAW fits'))
        for ctx in contexts:self.check('context:'+ctx['context_id'],lambda ctx=ctx:self.context(ctx))
        self.check('sealed_endpoints',lambda:self.endpoints(seal,work))
        def costs():
            require(len(self.summaries)==460,'Expected 460 completed blocks')
            require(len(list((self.root/'checkpoints').glob('*.pt')))==460,'Checkpoint matrix count')
            require(len(list((self.root/'states').glob('*_INITIAL.pt')))==60,'Initial common state count')
            require(len(list((self.root/'block_summaries').glob('*.json')))==460,'Summary matrix count')
            require(sum(r['completed'] for r in self.summaries.values())==ledger['updates']==404000,'Formal update accounting')
            global_updates=0
            for ctx in contexts:
                cid=ctx['context_id']
                ordered=([cid+'_RAW16_initial',cid+'_RAW16_final'] if ctx['run_raw'] else [])
                ordered += [cid+'_COMMON_regression',cid+'_COMMON_gp',cid+'_R_FROZEN_gp',
                            cid+'_R_REG_regression',cid+'_R_REG_gp',cid+'_R_DKL_C_joint',cid+'_R_DKL_U_joint']
                for key in ordered:
                    global_updates+=self.summaries[key]['completed']
                    state=self.load_state(key)
                    require(state['accounting']['budget_updates_after_last_step']==global_updates and
                            state['runner_budget_at_save']['updates']==global_updates and
                            state['last_update']['budget_formal_updates']==global_updates,'Serial block/global update order: '+key)
            require(self.trace_rows==4040 and self.diagnostic_count==4500 and self.window_count==780,'Trace/diagnostic/window matrix counts')
            near(sum(r['training_seconds'] for r in self.summaries.values()),ledger['training_seconds'],'Formal block seconds sum',atol=1e-7)
            near(ledger['seconds'],sum(ledger[k] for k in ('training_seconds','prediction_seconds','verification_seconds')),'Ledger categories')
            require(ledger['limit']==1200 and 0<=ledger['seconds']<1200 and ledger['failed_training_attempts']==0,'Completed budget/failure status')
            for key in ('seconds','training_seconds','prediction_seconds','verification_seconds','updates','verification_updates','failed_training_attempts'):
                near(ledger[key],seal['resources'][key],'Ledger/seal resource identity: '+key)
            synthetic=read(self.root/'verification/synthetic_ledger.json')
            require(ledger['verification_updates']==synthetic['updates'],'Synthetic verification update accounting')
            require(ledger['verification_seconds']+1e-9>=synthetic['seconds'],'Synthetic time included in cumulative budget')
            preflight=read(self.root/'verification/preflight_summary.json')
            require(preflight['pass'] and preflight['synthetic_optimizer_updates']==synthetic['updates']==115,'Unified verification update count')
            require(ledger['verification_seconds']+1e-9>=preflight['verification_seconds']>=synthetic['seconds'],'Synthetic/statistical checks included')
            # Historical manifest contains only endpoint descriptors/accounting.
            # Do not open the linked A geometry, labels, scored gate or tables.
            historical=read(self.root/'historical_manifest.json')
            require(historical['execution_status']=='COMPLETE_VALID' and historical['formal_updates_actual']==0 and
                    historical['resources']['updates']==0,'Historical diagnostics performed no fitting')
            near(historical['resources']['prediction_seconds'],historical['A_new_model_seconds'],'A diagnostics category')
            near(historical['resources']['verification_seconds'],preflight['verification_seconds'],'A preserves verification ledger')
            require(ledger['prediction_seconds']>=historical['resources']['prediction_seconds'],'A prediction cost included in final ledger')
            require(not list((self.root/'recovery').glob('*_STOPPED.pt')) and not (self.root/'failure.json').exists(),'Unexpected stopped/failed formal attempt')
            status=read(self.root/'status.json')
            require(status['formal_contexts_actual']==60 and status['formal_endpoints_actual']==260 and status['formal_updates_actual']==404000,'Status matrix count')
            near(status['resource_seconds'],ledger['seconds'],'Status resource seconds')
        self.check('formal_and_verification_accounting',costs)
        result={'pass':not self.failures,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'checks_passed':len(self.passed),'failed_checks':self.failures,'passed_checks':self.passed,
            'blocks_loaded':len(self.summaries),'checkpoint_states_loaded':len(self.states),
            'diagnostic_rows_checked':self.diagnostic_count,'update_rows_checked':self.trace_rows,
            'final_saved_windows_recomputed':self.window_count,'hash_checks':self.hash_files,
            'resource_ledger':ledger,'predictions_manifest_sha256':sha(self.root/'predictions_manifest.json'),
            'auditor_sha256':sha(Path(__file__)),'audit_wall_seconds':time.perf_counter()-started,
            'model_initializations':0,'forward_predictions':0,'optimizer_updates':0,'NLL_evaluations':0,'SVD_evaluations':0,
            'scoring_labels_or_A_scores_read':False,'accounting':'saved-artifact statistics and IO only; excluded from fit/predict budget',
            'limits':['Bounded norms validated against saved isolated diagnostic measurements; no new SVD performed.',
                      'Permanent lock existence checked; shell launch history must be retained separately.',
                      'Complete mathematical state hashes verified from saved tensors; history embeddings not recomputed.']}
        out=self.root/'audit/final_matrix_review.json';out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('pass','checks_passed','failed_checks','blocks_loaded','diagnostic_rows_checked','update_rows_checked','final_saved_windows_recomputed','audit_wall_seconds')},ensure_ascii=False))
        return 0 if result['pass'] else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(Audit(args.root).run())
