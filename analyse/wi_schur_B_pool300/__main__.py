import argparse
import os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='B only: original RBF-Schur50 + single-seed W-I qLogEI250')
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('prepare','run','worker','accept','check','eta'):
        p=sub.add_parser(name);p.add_argument('--run-dir',type=Path,required=True)
        if name=='prepare':p.add_argument('--prompt',type=Path,action='append',required=True)
        if name=='run':p.add_argument('--recover',action='store_true')
        if name=='worker':p.add_argument('--index',type=int,choices=(0,1),required=True)
    args=parser.parse_args();root=args.run_dir.resolve()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.action=='prepare':
        from .prepare import prepare
        prepare(root,args.prompt)
    elif args.action=='run':
        from .runtime import controller
        controller(root,args.recover)
    elif args.action=='worker':
        from .runtime import worker
        worker(root,args.index)
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
