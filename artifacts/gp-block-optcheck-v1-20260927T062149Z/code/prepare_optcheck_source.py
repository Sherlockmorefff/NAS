"""Copy an explicit, hash-verified parent whitelist into a fresh optcheck run.

No checkpoint deserialization, model calls, test-field decoding or label parsing.
The parent's A scoring labels are recorded by byte identity but not copied here.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

PARENT_RUN = 'gp-block-distance-v1-20260924T125944Z'
PARENT_SHA = 'e86d367e5c36a234d9b8b04d990b0aaae70f755f3aa4f58ef94c61c94d0e25ff'
PROMPT_SHA = '8644766c36f2da5d5c574d675d7984f628d74dfb24203143ad777cccbc0787a0'
PARENT_METHODS = ('RAW16_ISO', 'HP4_ISO', 'RAW16_BLOCK2')
METHODS = ('A_REFIT', 'B_REFIT', 'C_REFIT')


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def require(ok, message):
    if not ok: raise ValueError(message)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
def safe(root, relative):
    p = root / relative
    require(not Path(relative).is_absolute() and p.resolve().is_relative_to(root.resolve()), 'Unsafe relative source path')
    return p


def inventory(root):
    entries = {}
    for line in (root / 'FILES.sha256').read_text().splitlines():
        expected, relative = line.split(None, 1); relative = relative.lstrip('* ')
        require(len(expected) == 64 and relative not in entries, 'Invalid parent file inventory')
        require(sha(safe(root, relative)) == expected, 'BLOCKED_SOURCE_MISMATCH: ' + relative)
        entries[relative] = expected
    require(entries, 'Empty parent inventory')
    return entries


def protocol(run_id):
    selection = {'regret_improvement_min_pp': .20, 'positive_folds_min': 4,
                 'both_samples_positive': True, 'gain_change_min_pp': -.02, 'tol_pp': 1e-9}
    probability = {'nlpd_increase_max_nat': .10, 'IS90_increase_max_pp': 1.,
                   'total_90cov_fall_max': .05, 'champion_90cov_fall_max': .05}
    return {
        'run_id': run_id, 'version': 'gp-block-optcheck-v1', 'prompt_sha256': PROMPT_SHA,
        'parent_run_id': PARENT_RUN, 'parent_package_sha256': PARENT_SHA,
        'scientific_question': 'Training-only nested GP optimization adequacy and repeatable architecture increment in Top5 selection; developmental A evidence',
        'methods': list(METHODS), 'parent_methods': list(PARENT_METHODS),
        'matrix': {'subsets': 10, 'sample_seeds': [20260922, 20260923], 'folds': list(range(5)), 'n_train': 50,
                   'query_counts': [38, 36, 36, 34, 35], 'trajectories': 70, 'selected_models': 30,
                   'parent_models': 30, 'historical_models': 90, 'scoring_units': 150,
                   'new_prediction_rows': 1074, 'scoring_candidate_rows': 5370,
                   'head_masks': 30, 'model_seed': None, 'execution_rng_seed': 0,
                   'order': 'sample_seed ascending, fold ascending, A_REFIT then B_REFIT then C_REFIT'},
        'input': {'canonical_columns': 16, 'architecture_columns': list(range(12)), 'hp_columns': list(range(12, 16)),
                  'hp_transform': ['log10(lr)', 'dropout', 'log2(hidden_dim)', 'log10(l2)'],
                  'additional_transform': None, 'block_dimension_normalization': False,
                  'training_allowed_keys': ['ids', 'architectures', 'x_standardized16', 'y_standardized', 'known_variance_standardized'],
                  'query_allowed_keys': ['ids', 'architectures', 'x_standardized16'],
                  'scaler': {'scope': 'current T only; reuse verified parent snapshot', 'ddof': 0, 'x_floor': 1e-6, 'y_floor': 1e-8, 'comparison_atol': 1e-12, 'comparison_rtol': 1e-10},
                  'known_variance': 'SE_T**2/y_scale_T**2; SE already includes three-seed mean uncertainty'},
        'model': {'formula_version': 'dense_matern52_nested_closed_boundary_v1',
                  'kernel': 'signal_variance*(1+sqrt(5*s)+5*s/3)*exp(-sqrt(5*s))',
                  'distance': {'A_REFIT': 'exp(-2*u)*(D_A+D_H)', 'B_REFIT': 'exp(-2*u)*D_H', 'C_REFIT': 'expm1(a)*D_A+exp(-2*u_H)*D_H'},
                  'trainable_scalars': {'A_REFIT': 4, 'B_REFIT': 4, 'C_REFIT': 5},
                  'coordinates': {'A_REFIT': ['m', 'log_signal_variance', 'log_extra_noise_variance', 'log_lengthscale'],
                                  'B_REFIT': ['m', 'log_signal_variance', 'log_extra_noise_variance', 'log_lengthscale'],
                                  'C_REFIT': ['m', 'log_signal_variance', 'log_extra_noise_variance', 'a', 'log_lengthscale_H']},
                  'bounds': {'m': [None, None], 'log_signal_variance': [math.log(1e-8), None],
                             'log_extra_noise_variance': [math.log(1e-6), None], 'log_lengthscale': [math.log(1e-4), None],
                             'log_lengthscale_H': [math.log(1e-4), None], 'a': [0., math.log1p(1e8)]},
                  'nested_maps': {'A_to_C': 'copy m,signal,extra; u_H=u_A; a=log1p(exp(-2*u_A))',
                                  'B_to_C': 'copy m,signal,extra; u_H=u_B; a=0 exactly',
                                  'parent_C': 'copy m,signal,extra; a=log1p(1/ell_A**2); u_H=log(ell_H)'},
                  'boundary': 'a=0 is exact HP-only; lengthscale_A JSON null with architecture_disabled=true',
                  'objective': '(0.5*v.T@solve(K_y,v)+0.5*logdet(K_y)+0.5*n*log(2*pi))/n; n=50 exactly once',
                  'K_y': 'K_TT+diag(known_variance+extra_noise_variance)',
                  'gradient': 'analytic; Cholesky solve against identity allowed; no generic inverse',
                  'dk_ds': '-(5/6)*signal_variance*(1+sqrt(5*s))*exp(-sqrt(5*s)); same finite value at s=0',
                  'neural_parameters': 0, 'priors_or_penalties': None, 'third_lengthscale': False},
        'numerics': {'dtype': 'float64', 'device': 'cpu', 'threads': 1, 'deterministic': True, 'jitter': 0.,
                     'cholesky_failure': 'stop; no adaptive jitter', 'equivalence_atol': 1e-10, 'equivalence_rtol': 1e-8,
                     'gradient_atol': 1e-6, 'gradient_rtol': 1e-4, 'latent_roundoff_lower': -1e-8},
        'optimizer': {'algorithm': 'scipy.optimize.minimize L-BFGS-B', 'jac': True, 'ftol': 1e-12, 'gtol': 1e-6,
                      'maxcor': 10, 'maxls': 20, 'gradient_clipping': None, 'workers': None,
                      'fresh_Hessian_history_per_start': True, 'hard_wrapper_before_every_fg_request': True,
                      'A_REFIT': {'starts': ['PARENT', 'COMMON_INITIAL'], 'max_fg_calls_each': 300, 'max_accepted_iterations_each': 250},
                      'B_REFIT': {'starts': ['PARENT', 'COMMON_INITIAL'], 'max_fg_calls_each': 300, 'max_accepted_iterations_each': 250},
                      'C_REFIT': {'starts': ['PARENT', 'A_SELECTED_EMBEDDING', 'B_SELECTED_BOUNDARY'], 'max_fg_calls_each': 200, 'max_accepted_iterations_each': 150},
                      'common_initial': {'constant_mean': 0., 'outputscale': 1., 'extra_noise_variance_standardized': .01, 'lengthscale': 1.},
                      'unused_allowance_transfer': False, 'merge_identical_starts': False,
                      'finite_solver_stop': ['normal convergence', 'iteration limit', 'request hard limit', 'finite line-search abnormal termination'],
                      'failure': 'nonfinite, incorrect gradient, covariance anomaly, untrustworthy counts: stop and retain all cost'},
        'selection': {'pool': 'initial feasible plus optimizer accepted iterates only', 'criterion': 'current T negative MLL only',
                      'tie_tolerance_nat_per_observation': 1e-10, 'tie_order': 'fixed start order then smaller accepted iteration',
                      'retain_known_feasible_starts': True, 'record_numerical_minimum_gap': True,
                      'line_search_trials_excluded': True, 'probes_excluded': True, 'query_inputs_or_labels_used': False},
        'budget': {'model_seconds': 600., 'carryover_seconds': 0., 'formal_optimization_fg_limit': 18000,
                   'formal_optimization_fg_limit_per_method': 6000, 'synthetic_fg_limit_all_attempts': 300,
                   'local_probe_limit': 260, 'automatic_resume': False,
                   'timing': {'n_train': 50, 'n_query': 38, 'warmup_fg_each_method': 5, 'segments': 3, 'fg_per_segment': 10,
                              'total_timing_fg': 105, 'estimator': 'maximum segment mean per method'},
                   'admission_formula': 'already_used + 1.5*(6000*(t_A+t_B+t_C)+remaining_non_solver_numeric_seconds) + 20',
                   'counting': 'every joint f/g request incl failed/repeated/trial; accepted iterations separate; factorization count separate',
                   'model_scopes': ['numeric preparation', 'kernels', 'NLL/gradient', 'solver wall including callback', 'prediction', 'independent gradient/probes', 'restoration verification'],
                   'nested_timing': 'no double charging', 'excluded_scopes': ['byte hashing/copy', 'SSH wait', 'code writing', 'raw validation log tracing', 'statistics on sealed arrays']},
        'diagnostics': {'parent_readonly_reproduction': 30, 'selected_state_independent_restore': 30,
                        'stationarity': {'projected_gradient_inf_max': 1e-5, 'boundary_tolerance': '1e-12*max(1,abs(bound))', 'lower_bound': 'only negative gradient violates', 'upper_bound': 'only positive gradient violates'},
                        'probes': {'coordinate_delta': .001, 'directions': [-1, 1], 'clip_to_finite_bounds': True, 'skip_zero_displacement': True, 'max_improvement_nat': 1e-6, 'max_calls': 260, 'may_enter_selection': False},
                        'dominance_tolerance_nat': 1e-6, 'adequacy': 'all10 known-feasible dominance; each method at least8/10 both stationarity and probes',
                        'trajectory_endpoint_predictions': False, 'final_model_predictions': 30,
                        'start_comparison': 'training NLL and parameter difference only; no claim of uncomputed prediction differences',
                        'old_Adam_stability': 'historical comparison only; no stopping use'},
        'case_audit': {'anchor_candidate_id': '93ddd76173ade24158c0d0aeee42abc9bff3a0c79f263d3e7656f359e9bffe02',
                       'rule': 'anchor union top5 unique candidate IDs by mean parent C candidate NLPD across both sample seeds; ties IDlex',
                       'after_protocol_freeze': True, 'max_log_search_wall_seconds': 900, 'validation_fields_only': True,
                       'block_only_confirmed_source_error': True, 'may_change_design': False, 'all_cases_remain_in_scoring': True},
        'prediction': {'full_latent_covariance': True, 'raw_mean': 'y_center+y_scale*mean_std', 'raw_covariance': 'y_scale**2*cov_std',
                       'model_variance_raw': 'y_scale**2*(max(latent_diagonal_std,0)+extra_variance_std)',
                       'observed_variance_raw': 'model_variance_raw+SE_Q**2 exactly once in postseal scorer'},
        'scoring': {'aggregation': 'historical seed metrics within subset, two samples within fold, five folds equal', 'tol_pp': 1e-9,
                    'primary_comparison': 'C_REFIT vs B_REFIT', 'required_comparisons': ['C_REFIT vs A_REFIT', 'each REFIT vs own OLD', 'C_REFIT vs HEAD', 'C_REFIT vs H_TREND', 'C_REFIT vs H_CONST', 'C_REFIT vs exact random5'],
                    'regions': ['all_Q', 'original_HEAD_pred20_each_seed', 'true_top20', 'true_bottom20', 'champions'],
                    'top5_rule': 'descending mean then candidate_id', 'probability_units': 'original accuracy proportion', 'pp_multiplier': 100,
                    'flags': {'ARCHITECTURE_INCREMENT_SIGNAL': {'comparison': 'C_REFIT vs B_REFIT', **selection},
                              'BLOCK_SCALE_SIGNAL': {'comparison': 'C_REFIT vs A_REFIT', **selection},
                              'HISTORICAL_SELECTION_RETAINED': {'baselines': ['HEAD', 'H_TREND'], 'regret_increase_max_pp': .10, 'gain_fall_max_pp': .02},
                              'PROBABILITY_NO_LARGE_DEGRADATION': {'baselines_separately': ['A_REFIT', 'A_OLD'], 'both_required': True, **probability},
                              'DEVELOPMENT_CANDIDATE_FOR_CONFIRMATION_DESIGN': 'complete valid matrix AND no confirmed source error AND OPTIMIZATION_ADEQUATE AND ARCHITECTURE_INCREMENT_SIGNAL AND BLOCK_SCALE_SIGNAL AND HISTORICAL_SELECTION_RETAINED AND both probability flags; unresolved logs remain explicit limitation'}},
        'permissions': {'history_A_queries_previously_used': True, 'case_audit_historical_A_labels_after_freeze': True,
                        'fit_query_labels_SE_history_metrics_or_case_report': False, 'postseal_scoring_A': True,
                        'algorithmic_whitelist_not_OS_access_control': True},
        'execution_order': ['source/config', 'synthetic/preflight/admission', 'freeze', 'readonly case audit', 'one formal run', 'full seal', 'A scoring', 'independent audit', 'package'],
        'failure_states': ['COMPLETE_VALID', 'BLOCKED_SOURCE_MISMATCH', 'BLOCKED_DATA_INCONSISTENCY', 'BLOCKED_DEPENDENCY', 'BLOCKED_PREFLIGHT', 'BLOCKED_BUDGET_ADMISSION', 'STOPPED_BUDGET_PARTIAL', 'FAILED_FORMAL_NUMERICS', 'FAILED_AUDIT', 'INTERRUPTED_UNVERIFIED'],
        'not_authorized': ['B pool/test', 'GNN/VAE calls', 'new candidates', 'neural mapping/head', 'joint DKL', 'new TREND/SEMI/OOF', 'tau', 'priors/regularizers', 'ARD', 'sum/product kernels', 'ranking losses', 'ensembles', 'EI/KG/BO', 'mean/interval clipping', 'case deletion', 'parameter scans', 'extra restarts', 'automatic next stage', 'push/merge/mainline edits'],
        'interpretation': 'A-reused developmental experiment; no independent confirmation, global optimum, causal architecture importance or BO authorization'}


def prepare(root, parent, archive=None):
    started = time.monotonic(); root = Path(root).resolve(); parent = Path(parent).resolve()
    require(root != parent and not root.is_relative_to(parent), 'New root must be independent of immutable parent')
    require(sha(root / 'PROMPT.md') == PROMPT_SHA, 'Wrong authorized prompt')
    require(not (root / 'source_manifest.json').exists() and not (root / 'prepare_once.lock').exists(), 'Source preparation is one-shot')
    lock = root / 'prepare_once.lock'; lock.mkdir(); write(lock / 'entry.json', {'created_at_utc': now(), 'permanent': True})
    inv = inventory(parent)
    archive = Path(archive) if archive else parent / 'evidence.tar.gz'
    archive_checked = archive.is_file()
    if archive_checked: require(sha(archive) == PARENT_SHA, 'BLOCKED_SOURCE_MISMATCH: parent evidence archive')
    else:
        expected = (parent / 'evidence.tar.gz.sha256').read_text().split()[0]
        require(expected == PARENT_SHA, 'Unpacked identity lacks expected parent archive provenance')
    parent_protocol = read(parent / 'protocol.json'); parent_seal = read(parent / 'predictions_manifest.json')
    require(parent_protocol['run_id'] == PARENT_RUN and parent_seal['sealed'] and parent_seal['execution_status'] == 'COMPLETE_VALID', 'Invalid parent sealed identity')
    for manifest in ('source_manifest.json', 'code_manifest.json', 'data_manifest.json', 'frozen_manifest.json', 'predictions_manifest.json'):
        for rel, expected in read(parent / manifest)['files'].items():
            require(sha(safe(parent, rel)) == expected, 'Parent nested manifest mismatch: ' + rel)
    files, origins = {}, {}
    def copy(relative, destination):
        require(relative in inv or relative == 'FILES.sha256', 'Uninventoried parent source: ' + relative)
        src = safe(parent, relative); target = safe(root, destination)
        require(not target.exists(), 'Do not overwrite existing source: ' + destination)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, target)
        files[destination] = sha(target); origins[destination] = str(src)
        require(files[destination] == (inv[relative] if relative in inv else sha(src)), 'Source copy mismatch')
    for rel in ('public_inputs.json', 'subsets.json', 'folds.json'):
        copy(rel, rel)
    for folder, pattern in (('data', '*.json'), ('training', '*.json'), ('queries', '*.json'), ('head_masks', '*.json')):
        for p in sorted((parent / folder).glob(pattern)): copy(str(p.relative_to(parent)), str(p.relative_to(parent)))
    for rel in ('PROMPT.md', 'protocol.json', 'README.md', 'REPRODUCE.md', 'FILES.sha256', 'source_manifest.json',
                'code_manifest.json', 'data_manifest.json', 'frozen_manifest.json', 'predictions_manifest.json',
                'worklist.json', 'masks_manifest.json', 'historical_refs.json', 'state_index.json', 'diagnostic_index.json',
                'budget.json', 'cost_summary.json', 'status.json', 'environment.json'):
        copy(rel, 'source/parent/' + rel)
    for p in sorted((parent / 'code').glob('*.py')): copy(str(p.relative_to(parent)), 'source/parent/' + str(p.relative_to(parent)))
    for rel in ('audit/block_data_review.json', 'audit/block_completion_review.json', 'audit/independent_block_scoring.json', 'audit/release_order.json'):
        copy(rel, 'source/parent/' + rel)
    parent_new = [r for r in parent_seal['records'] if not r['is_historical']]
    require(len(parent_new) == 30, 'Parent must have30 original fits')
    for r in parent_new:
        copy(r['path'], 'source/parent_predictions/' + Path(r['path']).name)
        copy('diagnostics/' + r['unit_id'] + '.json', 'source/parent_diagnostics/' + r['unit_id'] + '.json')
    historical = read(parent / 'historical_refs.json')['records']; require(len(historical) == 90, '90 historical references required')
    for r in historical: copy(r['path'], 'source/historical_predictions/' + Path(r['path']).name)
    labels = parent / 'scoring/labels_A.json'; require('scoring/labels_A.json' in inv, 'Parent A labels absent from inventory')
    source = {'run_id': root.name, 'parent_run_id': PARENT_RUN, 'parent_root': str(parent), 'parent_package_sha256': PARENT_SHA,
              'parent_archive_path': str(archive), 'parent_archive_hash_verified': archive_checked,
              'parent_identity_verification': 'archive plus full FILES' if archive_checked else 'full FILES plus recorded sealed identity; original archive not re-created',
              'parent_FILES_sha256': sha(parent / 'FILES.sha256'), 'parent_inventory': inv,
              'parent_seal_sha256': sha(parent / 'predictions_manifest.json'), 'files': files, 'origins': origins,
              'legal_A_scoring_truth_path': str(labels), 'legal_A_scoring_truth_sha256': inv['scoring/labels_A.json'],
              'source_label_path': str(labels), 'expected_labels_sha256': inv['scoring/labels_A.json'],
              'query_labels_copied': False, 'query_label_values_decoded': 0, 'test_fields_decoded': 0,
              'historical_model_seconds': read(parent / 'budget.json')['seconds'], 'new_budget_carryover': 0,
              'copy_whitelist': 'frozen raw numeric data, parent JSON predictions/diagnostics, historical predictions/masks, identity/code/audit; no checkpoint tensors or metrics tables',
              'created_at_utc': now()}
    write(root / 'source_manifest.json', source); write(root / 'protocol.json', protocol(root.name))
    versions = {}
    for name in ('numpy', 'scipy'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    write(root / 'environment.json', {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                                      'packages': versions, 'runtime_not_imported_for_metadata': True,
                                      'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}, 'created_at_utc': now()})
    write(root / 'status.json', {'status': 'SOURCE_PREPARED', 'formal_started': False, 'new_model_calls': 0, 'source_files': len(files), 'source_wall_seconds': time.monotonic() - started})
    return {'pass': True, 'source_files': len(files), 'parent_inventory_files': len(inv), 'parent_archive_hash_verified': archive_checked,
            'model_calls': 0, 'query_label_values_decoded': 0, 'source_wall_seconds': time.monotonic() - started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True); parser.add_argument('--parent-archive', type=Path)
    args = parser.parse_args(); print(json.dumps(prepare(args.root, args.parent_root, args.parent_archive), ensure_ascii=False))
