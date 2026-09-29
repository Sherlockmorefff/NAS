"""One-shot orchestration; all model numerics use the same persistent ledger."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import time
import traceback
import nested_core as core
import optcheck_data as data


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value): core._atomic(p, value)
def require(ok, message):
    if not ok: raise RuntimeError(message)
def verify(root, manifest):
    for rel, expected in manifest['files'].items():
        require(sha(root / rel) == expected, 'Frozen file changed: ' + rel)


def verify_sources(root):
    lock = root / 'source_numeric_once.lock'; lock.mkdir()
    write(lock / 'entry.json', {'created_at_utc': now(), 'permanent': True})
    budget = core.Budget(root / 'budget.json'); before = budget.seconds
    records, neighbors = [], []
    try:
        core.cpu_setup(budget)
        for group in read(root / 'worklist.json')['groups']:
            sid, method = group['subset_id'], group['method']
            with budget.measure('source_input_projection', 'source_validation'):
                training, query, scaler = data.load_fit(root, sid), data.load_query(root, sid), data.load_scaler(root, sid)
                parent = data.load_parent_start(root, sid, method)
                theta = core.parent_theta(method, parent['physical'])
            model = core.make_model(method, training, budget, 'source_validation')
            saved = read(root / parent['source']['path'])
            old = data.OLD[method]
            diag = read(root / ('source/parent_diagnostics/' + sid + '_' + old + '.json'))
            result = core.parent_verify(model, theta, saved, query, scaler, budget, diag['records'][-1]['training_kernel'])
            result.update(unit_id=group['unit_id'], subset_id=sid, parent_prediction_sha256=parent['source']['sha256'])
            rel = 'verification/parent_numerical/' + group['unit_id'] + '.json'
            write(root / rel, result)
            records.append({'unit_id': group['unit_id'], 'path': rel, 'sha256': sha(root / rel), 'pass': result['pass']})
            neighbors.extend({'subset_id': sid, 'method': method, 'parent_method': old,
                              'parent_prediction_sha256': parent['source']['sha256'], **v} for v in result['neighbors'])
        write(root / 'verification/parent_neighbors.json', {'records': neighbors, 'query_labels_read': False})
        report = {'pass': True, 'records': records, 'verified_parent_count': 30,
                  'source_model_seconds': budget.seconds - before, 'query_labels_read': False,
                  'files': {r['path']: r['sha256'] for r in records}}
        report['files']['verification/parent_neighbors.json'] = sha(root / 'verification/parent_neighbors.json')
        write(root / 'verification/parent_numerical.json', report)
        print(json.dumps({k: report[k] for k in ('pass', 'verified_parent_count', 'source_model_seconds')}), flush=True)
    except BaseException as error:
        failure = 'STOPPED_BUDGET_PARTIAL' if isinstance(error, core.BudgetExceeded) else 'INTERRUPTED_UNVERIFIED' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'BLOCKED_SOURCE_MISMATCH'
        write(root / 'status.json', {'status': failure, 'error': repr(error), 'resources': budget.as_dict()})
        raise


def freeze(root):
    require(not (root / 'frozen_manifest.json').exists(), 'Already frozen')
    require(not (root / 'formal_once.lock').exists(), 'Formal entry consumed')
    budget = core.Budget(root / 'budget.json')
    require(budget.optimization_requests == 0, 'Formal requests already consumed')
    pre = read(root / 'verification/preflight.json')
    required = ['verification/preflight.json', 'verification/parent_numerical.json',
                'verification/data_synthetic.json', 'verification/data_preparation.json',
                'verification/scoring_synthetic.json']
    for rel in required: require(read(root / rel)['pass'], 'Failed prerequisite: ' + rel)
    # Forecast numerical work from measured preflight operation maxima. An
    # additional solver-overhead allowance covers persistence and Python/SciPy.
    operations = pre['operation_costs']
    def measured_cost(prefix):
        values = [v['max_wall_seconds'] for k, scopes in operations.items()
                  if k.startswith(prefix) for v in scopes.values() if v['calls']]
        require(bool(values), 'Missing timing operation: ' + prefix)
        return max(values)
    parts = {
        'model_constructions_60': 60 * measured_cost('model_'),
        'independent_gradients_60': 60 * measured_cost('independent_nll_gradient_'),
        'probes_260': 260 * max(c['seconds_per_request'] for c in pre['timing_cells']),
        'predictions_30': 30 * measured_cost('predict_'),
        'independent_restoration_predictions_30': 30 * measured_cost('independent_predict_'),
        'snapshots_30': 30 * measured_cost('snapshot_'),
        'solver_overhead_18000': pre['solver_overhead_est_18000_requests'],
        'projection_and_other_reserve': 5.,
        'remaining_source_validation': 0.,
    }
    total = budget.seconds + 1.5 * (pre['fit_est'] + sum(parts.values())) + 20
    admission = {'admitted': total <= 600, 'already_used': budget.seconds, 'limit': 600,
                 'fit_est': pre['fit_est'], 'nonsolver_estimates': parts, 'conservative': total,
                 'formula': 'used + 1.5 * (6000 * (tA+tB+tC) + remaining numerical and solver overhead estimates) + 20',
                 'timing_cells': pre['timing_cells'], 'no_retiming_or_matrix_shrink': True, 'created_at_utc': now()}
    write(root / 'admission.json', admission)
    if not admission['admitted']:
        write(root / 'status.json', {'status': 'BLOCKED_BUDGET_ADMISSION', 'formal_started': False, 'resources': budget.as_dict()})
        print(json.dumps(admission)); return
    code = {str(p.relative_to(root)): sha(p) for folder in ('code', 'audit') for p in sorted((root / folder).glob('*.py'))}
    write(root / 'code_manifest.json', {'files': code, 'frozen_at_utc': now()})
    files = {}
    for name in ['PROMPT.md', 'protocol.json', 'source_manifest.json', 'code_manifest.json', 'data_manifest.json',
                 'masks_manifest.json', 'historical_refs.json', 'worklist.json', 'environment.json', 'admission.json'] + required:
        files[name] = sha(root / name)
    for name in ('source_manifest', 'code_manifest', 'data_manifest', 'masks_manifest'):
        manifest = read(root / (name + '.json')); verify(root, manifest); files.update(manifest['files'])
    parent_check = read(root / 'verification/parent_numerical.json'); verify(root, parent_check); files.update(parent_check['files'])
    for path in sorted((root / 'verification/prefreeze_code').glob('*')):
        if path.is_file(): files[str(path.relative_to(root))] = sha(path)
    write(root / 'frozen_manifest.json', {'files': files, 'frozen_at_utc': now(), 'formal_requests': 0,
                                        'outer_labels_released': False, 'automatic_resume': False})
    write(root / 'status.json', {'status': 'ADMITTED_FROZEN', 'formal_started': False, 'resources': budget.as_dict()})
    print(json.dumps(admission), flush=True)


def run(root):
    if (root / 'seal.json').exists():
        seal = read(root / 'seal.json'); manifest = read(root / 'predictions_manifest.json')
        require(seal['sealed'] and seal['execution_status'] == 'COMPLETE_VALID' and manifest['sealed'], 'Invalid existing seal')
        require(sha(root / 'predictions_manifest.json') == seal['predictions_manifest_sha256'], 'Existing seal hash differs')
        verify(root, manifest)
        print(json.dumps({'existing_sealed_result': True, 'additional_numerical_calls': 0,
                          'predictions_manifest_sha256': seal['predictions_manifest_sha256']}), flush=True)
        return
    budget = core.Budget(root / 'budget.json'); started = time.monotonic()
    require(budget.optimization_requests == 0, 'No resume or duplicate formal execution')
    require(read(root / 'admission.json')['admitted'], 'Not admitted')
    verify(root, read(root / 'frozen_manifest.json'))
    case = read(root / 'case_audit/summary.json')
    require(case['confirmed_data_inconsistency'] is False, 'BLOCKED_DATA_INCONSISTENCY')
    require(case['case_manifest_sha256'] == sha(root / 'case_audit_manifest.json'), 'Case audit manifest differs')
    lock = root / 'formal_once.lock'; lock.mkdir()
    write(lock / 'entry.json', {'created_at_utc': now(), 'permanent': True, 'automatic_resume': False})
    files, records, done, selected_by_uid, flat = {}, [], [], {}, []
    states, diagnostics, traces, restores = [], [], [], []
    def save(rel, value):
        write(root / rel, value); files[rel] = sha(root / rel); return {'path': rel, 'sha256': files[rel]}
    def status(value, **extra):
        write(root / 'status.json', {'status': value, 'completed_fits': len(done), 'completed_fit_ids': done,
              'completed_trajectories': len(traces), 'resources': budget.as_dict(),
              'formal_wall_seconds': time.monotonic() - started, 'outer_labels_released': False, **extra})
    try:
        core.cpu_setup(budget)
        work = read(root / 'worklist.json')
        for group in work['groups']:
            uid, sid, method = group['unit_id'], group['subset_id'], group['method']
            with budget.measure('formal_input_projection', 'setup'):
                training, scaler = data.load_fit(root, sid), data.load_scaler(root, sid)
                parent = data.load_parent_start(root, sid, method)
                starts = [core.parent_theta(method, parent['physical'])]
                if method == 'C_REFIT':
                    starts.extend(core.embed_into_C(m, selected_by_uid[sid + '_' + m]['coordinates']) for m in ('A_REFIT', 'B_REFIT'))
                else: starts.append(core.common_theta(method))
            model = core.make_model(method, training, budget, 'setup')
            trajectories = []
            cells = [t for t in work['trajectories'] if t['unit_id'] == uid]
            for cell, theta in zip(cells, starts):
                tid = cell['trajectory_id']; logrel = 'optimization/' + tid + '.jsonl'
                trace = core.run_trajectory(model, theta, cell['start_index'], cell['max_fg_calls'],
                    cell['max_accepted_iterations'], budget, request_log=root / logrel, source=cell)
                files[logrel] = sha(root / logrel)
                item = save('optimization/' + tid + '.json', {**cell, **trace})
                traces.append({**item, 'trajectory_id': tid, 'unit_id': uid,
                               'actual_requests': trace['actual_requests'], 'accepted_iterations': trace['accepted_iterations']})
                trajectories.append(trace)
            with budget.measure('accepted_state_selection', 'optimization'):
                selected = core.select_candidates(trajectories)
            selected_by_uid[uid] = selected
            known = [{'source': 'parent_C' if method == 'C_REFIT' else 'parent', 'nll': trajectories[0]['accepted'][0]['nll']}]
            if method == 'C_REFIT':
                for i, m in enumerate(('A_REFIT', 'B_REFIT'), 1):
                    nll = trajectories[i]['accepted'][0]['nll']; original = selected_by_uid[sid + '_' + m]['nll']
                    require(abs(nll - original) <= 1e-10 + 1e-8 * abs(original), 'Embedded selected NLL differs')
                    known.append({'source': m, 'nll': nll, 'source_selected_nll': original})
            identity = {**group, **data.load_identity(root, sid), 'is_historical': False, 'execution_rng_seed': 0,
                        **{name + '_sha256': sha(root / (name + '.json')) for name in ('protocol', 'code_manifest', 'source_manifest', 'frozen_manifest')}}
            d = core.diagnose_selected(model, selected, known, budget)
            d.update(unit_id=uid, subset_id=sid, selected_start_index=selected['start_index'],
                     selected_iteration=selected['iteration'], selected_minus_numerical_minimum=selected['selected_minus_numerical_minimum'])
            di = save('diagnostics/' + uid + '.json', d); diagnostics.append({'unit_id': uid, **di})
            state = core.snapshot(model, selected, scaler, budget, identity=identity)
            si = save('states/' + uid + '.json', state); states.append({'unit_id': uid, **si})
            for index, trajectory in enumerate(trajectories):
                save('states/trajectory_states/' + uid + '_start' + str(index) + '.json', {
                    'method': method, 'unit_id': uid, 'start_index': index,
                    'formula_version': core.FORMULA_VERSION, 'training_identity': state['training_identity'],
                    'scaler': scaler, 'identity': identity, 'source': trajectory['source'],
                    'initial': trajectory['accepted'][0], 'best_accepted': trajectory['best'],
                    'last_accepted': trajectory['accepted'][-1], 'resume_allowed': False,
                    'trajectory_prediction_computed': False})
            with budget.measure('query_projection', 'diagnostic'): query = data.load_query(root, sid)
            pred = {**identity, **core.predict(model, selected['coordinates'], query, scaler, budget),
                    'selected': selected, 'negative_mll': selected['nll'], 'endpoint_nll_after_update': selected['nll'],
                    'gp_parameters': state['physical_parameters'], 'state_path': si['path'], 'state_file_sha256': si['sha256'],
                    'diagnostics_path': di['path'], 'diagnostics_sha256': di['sha256'],
                    'train_candidate_ids': training['ids'], 'train_architecture_fingerprints': training['architectures'],
                    'scaler': scaler, 'query_labels_or_SE_present': False}
            restored = core.restore_check(read(root / si['path']), training, query, pred, budget)
            restores.append({'unit_id': uid, **save('verification/restoration/' + uid + '.json', restored)})
            pi = save('predictions/' + uid + '.json', pred)
            records.append({k: pred[k] for k in ('unit_id', 'subset_id', 'fold', 'sample_seed', 'model_seed', 'method', 'is_historical')} | pi)
            for i, cid in enumerate(pred['candidate_ids']):
                flat.append({'unit_id': uid, 'subset_id': sid, 'method': method, 'candidate_id': cid,
                             'mean_raw': pred['mean_raw'][i], 'model_variance_raw': pred['model_variance_raw'][i]})
            done.append(uid); status('RUNNING')
            print(json.dumps({'completed_fits': len(done), 'completed_trajectories': len(traces),
                              'optimization_requests': budget.optimization_requests, 'model_seconds': budget.seconds, 'unit_id': uid}), flush=True)
        require(len(done) == 30 and len(traces) == 70 and len(flat) == 1074, 'Incomplete matrix')
        require(sum(t['actual_requests'] for t in traces) == budget.optimization_requests <= 18000, 'Request ledger mismatch')
        for name, items in [('state_index', states), ('diagnostic_index', diagnostics), ('optimization_index', traces), ('restoration_index', restores)]:
            save(name + '.json', {'records': items, 'count': len(items)})
        with (root / 'new_candidate_predictions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        files['new_candidate_predictions.csv'] = sha(root / 'new_candidate_predictions.csv')
        for ref in read(root / 'historical_refs.json')['records']:
            require(sha(root / ref['path']) == ref['sha256'], 'Reference changed'); records.append(ref); files[ref['path']] = ref['sha256']
        files.update(read(root / 'masks_manifest.json')['files'])
        for path in sorted((root / 'case_audit').rglob('*')):
            if path.is_file(): files[str(path.relative_to(root))] = sha(path)
        verify(root, read(root / 'frozen_manifest.json')); budget.check(); budget.persist()
        seal = {'sealed': True, 'sealed_at_utc': now(), 'execution_status': 'COMPLETE_VALID', 'files': files,
                'records': records, 'new_endpoint_count': 30, 'parent_reference_count': 30, 'historical_reference_count': 90,
                'raw_unit_count': 150, 'mask_count': 30, 'completed_trajectories': 70, 'new_candidate_rows': 1074,
                'all_candidate_rows': 5370, 'independent_restoration_count': 30, 'local_diagnostic_count': 30,
                'resources': budget.as_dict(), 'outer_labels_released': False}
        for name in ('protocol', 'source_manifest', 'code_manifest', 'worklist', 'masks_manifest', 'frozen_manifest', 'data_manifest', 'case_audit_manifest'):
            seal[name + '_sha256'] = sha(root / (name + '.json'))
        write(root / 'predictions_manifest.json', seal)
        write(root / 'seal.json', {'sealed': True, 'execution_status': 'COMPLETE_VALID', 'sealed_at_utc': now(),
                                 'predictions_manifest_sha256': sha(root / 'predictions_manifest.json')})
        status('COMPLETE_VALID')
    except BaseException as error:
        failure = 'STOPPED_BUDGET_PARTIAL' if isinstance(error, core.BudgetExceeded) else 'INTERRUPTED_UNVERIFIED' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'FAILED_FORMAL_NUMERICS' if isinstance(error, core.NumericalFailure) else 'FAILED_FORMAL'
        status(failure, error=repr(error), traceback=traceback.format_exc())
        write(root / 'partial_manifest.json', {'sealed': False, 'files': files, 'records': records, 'resources': budget.as_dict()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('source', 'freeze', 'run'), required=True)
    args = parser.parse_args(); {'source': verify_sources, 'freeze': freeze, 'run': run}[args.phase](args.root)
