import argparse
import os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='Matched A/C only; frozen B is read-only')
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('prepare','run','worker','gp_worker','score','accept','check','eta'):
        p=sub.add_parser(name);p.add_argument('--run-dir',type=Path,required=True)
        if name=='prepare':p.add_argument('--prompt',type=Path,required=True)
        if name=='run':p.add_argument('--recover',action='store_true')
        if name in ('worker','gp_worker'):p.add_argument('--index',type=int,choices=(0,1),required=True)
    args=parser.parse_args();root=args.run_dir.resolve();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.action=='prepare':
        from .prepare import prepare
        prepare(root,args.prompt)
    elif args.action=='run':
        from .runtime import controller
        controller(root,args.recover)
    elif args.action=='worker':
        from .runtime import training_worker
        training_worker(root,args.index)
    elif args.action=='gp_worker':
        from .runtime import gp_worker
        gp_worker(root,args.index)
    elif args.action=='score':
        from .scoring import score
        score(root)
    elif args.action=='accept':
        from .report import timed_acceptance
        timed_acceptance(root)
    elif args.action=='check':
        from .report import accept_complete
        print(accept_complete(root))
    elif args.action=='eta':
        from .report import update_eta
        print(update_eta(root))

if __name__=='__main__':main()
