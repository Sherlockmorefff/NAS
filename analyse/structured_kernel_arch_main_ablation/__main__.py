import argparse
import os
import sys
from pathlib import Path
from .common import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['preflight','run','_learner','_oracle','analyze','prepare-C','report','package'])
    p.add_argument('--run-dir',required=True);p.add_argument('--prompt');p.add_argument('--acceptance-dir');p.add_argument('--seed',type=int)
    p.add_argument('--resume',action='store_true');args=p.parse_args();repo=Path.cwd();root=Path(args.run_dir).resolve()
    if args.command=='preflight':
        from .preflight import preflight
        preflight(root,repo,Path(args.prompt),Path(args.acceptance_dir))
    elif args.command not in ('_oracle','_learner'):
        verify(root,repo,'A' if args.command=='analyze' else 'C' if args.command=='prepare-C' else 'shared')
    if args.command!='_oracle':append_jsonl(root/'commands.jsonl',{'time':now(),'cwd':str(repo),'argv':[sys.executable,'-B','-m','analyse.structured_kernel_arch_main_ablation',*sys.argv[1:]],'env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}})
    if args.command=='run':
        from .runner import run
        run(root,repo,args.resume)
    elif args.command=='_learner':
        from .learner import run_worker
        run_worker(root,read_json(root/'run_identity.json')['identity'],only_seed=args.seed)
    elif args.command=='_oracle':
        from .oracle import serve
        serve(root,repo)
    elif args.command=='analyze':
        from .analysis import run
        run(root,repo)
    elif args.command=='prepare-C':
        from .c_prepare import run
        run(root,repo)
    elif args.command=='report':
        from .report import report
        report(root,repo)
    elif args.command=='package':
        from .report import package
        package(root,repo)

if __name__=='__main__':main()
