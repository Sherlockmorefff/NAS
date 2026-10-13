import argparse
import os
from pathlib import Path
from .common import ROOT

def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','worker','score','timer','status'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--prompt',type=Path)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--recover',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':
        from .prepare import prepare
        print(prepare(args.root,args.prompt),flush=True)
    elif args.action=='run':
        from .runtime import controller
        controller(args.root,args.recover)
    elif args.action=='worker':
        from .runtime import worker
        worker(args.root,args.index)
    elif args.action=='score':
        from .scoring import score
        score(args.root)
    elif args.action=='timer':
        from .runtime import timer
        timer(args.root)
    else:
        import json
        from .common import read_json
        print(json.dumps({n:read_json(args.root/n) for n in ['STATUS.json','ETA.json','acceptance_schedule.json','acceptance_status.json','ACCEPTANCE.json','COMPLETE.json'] if (args.root/n).exists()},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
