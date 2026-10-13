import argparse
import fcntl
import json
import shlex
import subprocess
import sys
from pathlib import Path
from analyse.structured_kernel_phase1.io import read_json,write_json,append_jsonl,sha256,now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['preflight','run','score','report','package','_oracle','_worker'])
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--prompt',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--max-new-decisions',type=int)
    parser.add_argument('--identity')
    args=parser.parse_args();root=args.run_dir.resolve();repo=Path.cwd().resolve()
    if args.workers!=1:
        raise ValueError('Single CPU worker required')
    if args.command=='_oracle':
        from .oracle import serve
        serve(root,repo);return
    if args.command=='_worker':
        from .learner import run_worker
        result=run_worker(root,args.identity,args.max_new_decisions)
        write_json(root/'worker_exit.json',{'state':result,'time':now()});return
    from .data import preflight,verify
    if args.command=='preflight':
        result=preflight(root,repo,args.prompt)
        print(json.dumps({'preflight':'PASS','run_identity':result['run_identity']}),flush=True)
        return
    frozen=verify(root,repo)
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        append_jsonl(root/'commands.jsonl',{'time':now(),'cwd':str(repo),'argv':sys.argv,'executable':sys.executable})
        if args.command=='run':
            from .learner import validate_complete
            from .protocol import SEEDS,METHODS
            complete=all((root/'trajectories'/f'{s}_{m}'/'complete.json').exists() for s in SEEDS for m in METHODS)
            if complete:
                before={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file() and
                    ('trajectories' in p.parts or p.name in ('summary.json','curves.csv','paired_comparisons.csv'))}
                for s in SEEDS:
                    for m in METHODS:
                        validate_complete(root/'trajectories'/f'{s}_{m}')
                if any(sha256(root/p)!=h for p,h in before.items()):
                    raise ValueError('Resume modified completed results')
                write_json(root/'resume_verification.json',dict(state='PASS',additional_optimization_steps=0,additional_queries=0,
                    additional_logical_releases=0,additional_label_file_opens=0,unchanged_files=len(before),hashes=before,time=now()))
                print('COMPLETE resume: zero optimization / query / label access; artifacts unchanged',flush=True)
                return
            if (root/'trajectories').exists() and not args.resume:
                raise ValueError('Existing progress requires --resume')
            command=[sys.executable,'-B','-m','analyse.structured_kernel_development_replay','_worker','--run-dir',str(root),'--identity',frozen['run_identity']]
            if args.max_new_decisions:
                command+=['--max-new-decisions',str(args.max_new_decisions)]
            result=subprocess.run(command)
            state=read_json(root/'worker_exit.json')['state'] if result.returncode==0 else 'INCOMPLETE'
            write_json(root/'status.json',{'state':state,'run_identity':frozen['run_identity'],'independent_confirmation':False,
                'returncode':result.returncode,'updated_at':now()})
            if result.returncode:
                raise RuntimeError('Worker interrupted/failed; saved progress retained, inspect logs before resume')
        elif args.command=='score':
            from .report import score
            s=score(root,repo);print(json.dumps(s),flush=True)
        elif args.command=='report':
            from .report import write_reports
            write_reports(root,repo)
        elif args.command=='package':
            from .report import package
            print(package(root))


if __name__=='__main__':
    main()
