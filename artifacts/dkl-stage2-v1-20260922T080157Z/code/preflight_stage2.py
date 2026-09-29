"""Bounded metadata and synthetic checks; no actual labels are decoded."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from stage2_core import Budget, cpu_setup
from stage2_data import mock_checks
from test_stage2_core import run_checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    directory=root/'verification';directory.mkdir(exist_ok=True)
    cpu_setup(0)
    ledger=Budget(directory/'synthetic_resource_ledger.json')
    core=run_checks(directory/'core',ledger)
    tick=time.perf_counter();data=mock_checks();data_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    score=subprocess.run([sys.executable,str(root/'code/test_scoring.py')],capture_output=True,text=True)
    scoring_seconds=time.perf_counter()-tick
    (directory/'scoring_tests.txt').write_text(score.stdout+score.stderr)
    (directory/'data_mock.json').write_text(json.dumps(data,indent=2)+'\n')
    codefiles=sorted((root/'code').rglob('*.py'))
    compile_result=subprocess.run([sys.executable,'-m','py_compile',*[str(p) for p in codefiles]],capture_output=True,text=True)
    repository=root.parents[1]
    diff=subprocess.run(['git','diff','--check'],cwd=repository,capture_output=True,text=True)
    whitespace=[]
    for path in codefiles:
        p=subprocess.run(['git','diff','--no-index','--check','/dev/null',str(path)],capture_output=True,text=True)
        whitespace.append({'path':str(path.relative_to(root)),'pass':p.returncode in [0,1] and not p.stdout and not p.stderr,
                           'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    result={'pass':core['pass'] and data['pass'] and score.returncode==0 and compile_result.returncode==0 and
                  diff.returncode==0 and all(p['pass'] for p in whitespace),
            'verification_seconds':core['charged_seconds']+data_seconds+scoring_seconds,
            'synthetic_optimizer_updates':core['synthetic_optimizer_updates'],
            'core':{'pass':core['pass'],'checks':list(core['checks']),'charged_seconds':core['charged_seconds']},
            'data_mock':data,'scoring_tests':{'pass':score.returncode==0,'returncode':score.returncode,'seconds':scoring_seconds,
                                            'output_file':'verification/scoring_tests.txt'},
            'syntax_compile':{'pass':compile_result.returncode==0,'stdout':compile_result.stdout,'stderr':compile_result.stderr},
            'repository_diff_check':{'pass':diff.returncode==0,'stdout':diff.stdout,'stderr':diff.stderr},
            'new_source_whitespace':whitespace,'real_labels_decoded':0}
    (directory/'preflight_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['pass','verification_seconds','synthetic_optimizer_updates','real_labels_decoded']}))
    if not result['pass']:raise SystemExit(1)


if __name__=='__main__':main()
