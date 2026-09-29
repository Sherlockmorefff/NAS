"""One bounded, synthetic-only preflight with persistent cumulative accounting."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from stage01_core import cpu_setup
from stage2_core import Budget
from test_length_core import run_checks as core_checks
from test_length_diagnostics import run_checks as diagnostic_checks


def main(root, diagnostics_only=False):
    cpu_setup(20260922)
    directory = root / 'verification'
    directory.mkdir(exist_ok=True)
    budget = Budget(directory / 'synthetic_ledger.json', limit=1200)
    attempt_number = len(list(directory.glob('preflight_attempt_*.json'))) + 1
    attempt = directory / f'attempt_{attempt_number}'
    attempt.mkdir()
    previous = json.loads((directory / 'preflight_summary.json').read_text()) if diagnostics_only else None
    if previous and not (previous['core']['pass'] and previous['scoring']['pass']):
        raise ValueError('Targeted diagnostic recheck requires existing passing core/scoring checks')
    core = previous['core'] if previous else core_checks(attempt / 'core', budget)
    diagnostics = diagnostic_checks(attempt / 'diagnostics', budget)
    if previous:
        scoring_result = previous['scoring']
    else:
        tick = time.monotonic()
        scoring = subprocess.run([sys.executable, str(root / 'code/test_length_score.py')], capture_output=True, text=True)
        budget.charge(time.monotonic() - tick)
        (attempt / 'scoring_tests.txt').write_text(scoring.stdout + scoring.stderr)
        scoring_result = {'pass': scoring.returncode == 0, 'returncode': scoring.returncode,
                          'output': str((attempt/'scoring_tests.txt').relative_to(root))}
    budget.persist()
    files = sorted((root / 'code').rglob('*.py'))
    compile_check = subprocess.run([sys.executable, '-m', 'py_compile', *[str(p) for p in files]], capture_output=True, text=True)
    diff_check = subprocess.run(['git', 'diff', '--check'], cwd=root.parents[1], capture_output=True, text=True)
    whitespace = []
    for path in files:
        result = subprocess.run(['git', 'diff', '--no-index', '--check', '/dev/null', str(path)], capture_output=True, text=True)
        whitespace.append({'path': str(path.relative_to(root)), 'pass': result.returncode in [0, 1] and not result.stdout and not result.stderr,
                           'stdout': result.stdout, 'stderr': result.stderr})
    result = {'pass': core['pass'] and diagnostics['pass'] and scoring_result['pass'] and compile_check.returncode == 0
                       and diff_check.returncode == 0 and all(row['pass'] for row in whitespace),
              'attempt': attempt_number, 'verification_seconds': budget.seconds, 'synthetic_optimizer_updates': budget.updates,
              'all_attempts_included_in_budget': True, 'core': core, 'diagnostics': diagnostics,
              'scoring': scoring_result, 'unchanged_core_scoring_pass_reused': diagnostics_only,
              'syntax': {'pass': compile_check.returncode == 0, 'error': compile_check.stderr},
              'git_diff_check': {'pass': diff_check.returncode == 0, 'stdout': diff_check.stdout, 'stderr': diff_check.stderr},
              'whitespace': whitespace, 'real_training_updates': 0, 'real_query_labels_read': 0}
    data = json.dumps(result, indent=2, allow_nan=False) + '\n'
    (directory / f'preflight_attempt_{attempt_number}.json').write_text(data)
    (directory / 'preflight_summary.json').write_text(data)
    print(json.dumps({k: result[k] for k in ['pass', 'attempt', 'verification_seconds', 'synthetic_optimizer_updates']}, ensure_ascii=False))
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--diagnostics-only', action='store_true')
    args = p.parse_args()
    main(args.root.resolve(), args.diagnostics_only)
