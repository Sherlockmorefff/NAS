import argparse
import os
import shlex
import sys
import traceback
from pathlib import Path
from .common import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['preflight','run','module','train-task','report','package'])
    p.add_argument('--run-dir',required=True);p.add_argument('--prompt');p.add_argument('--resume',action='store_true')
    p.add_argument('--acceptance-dir');p.add_argument('--name',choices=['M1','M2','M3','M4']);p.add_argument('--task-id')
    args=p.parse_args();repo=Path.cwd();root=(repo/args.run_dir).resolve()
    if args.command=='preflight':
        from .preflight import preflight
        preflight(root,repo,Path(args.prompt),Path(args.acceptance_dir))
    else:
        verify(root,repo)
    append_jsonl(root/'commands.jsonl',{'time':now(),'cwd':str(repo),'argv':[sys.executable,'-B','-m','analyse.structured_kernel_followup',*sys.argv[1:]],
        'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG')}})
    if args.command=='run':
        from .runner import run
        run(root,repo,args.resume)
    elif args.command=='module':
        if args.name=='M1':from .diagnose import run
        elif args.name=='M2':from .random_reference import run
        elif args.name=='M4':from .c_audit import run
        else:from .runner import train_all as run
        run(root,repo)
    elif args.command=='train-task':
        from .validation_train import run_task
        run_task(root,repo,args.task_id)
    elif args.command=='report':
        from .report import report
        report(root,repo)
    elif args.command=='package':
        from .report import package
        package(root,repo)

if __name__=='__main__':main()
