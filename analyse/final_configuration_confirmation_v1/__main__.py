import argparse
import os
from pathlib import Path
import sys
import traceback

class Tee:
    def __init__(self, first, second):
        self.first,self.second=first,second
    def write(self,text):
        self.first.write(text);self.second.write(text);return len(text)
    def flush(self):
        self.first.flush();self.second.flush()

def main():
    from deterministic_runtime import prepare_deterministic_environment
    prepare_deterministic_environment()
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['preflight','start','stop','controller','worker','train-task','oracle','search','verify','score-H'])
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--worker',type=int,choices=[1,2])
    parser.add_argument('--assignment')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--pause',action='store_true')
    parser.add_argument('--log-file')
    args=parser.parse_args();root=Path(args.run_dir).resolve();repo=Path(__file__).resolve().parents[2]
    if args.log_file:
        stream=Path(args.log_file).open('a',buffering=1)
        sys.stdout=Tee(sys.stdout,stream);sys.stderr=Tee(sys.stderr,stream)
    if args.command=='preflight':
        from .protocol import preflight
        preflight(root,repo)
    elif args.command=='start':
        from .launch import start
        start(root,repo,args.resume)
    elif args.command=='stop':
        from .launch import stop
        stop(root)
    elif args.command=='controller':
        from .controller import run
        run(root,repo,args.resume)
    elif args.command=='worker':
        from .training import worker
        if args.worker is None:
            parser.error('--worker required')
        worker(root,repo,args.worker)
    elif args.command=='train-task':
        from .training import train_task
        if args.assignment is None:
            parser.error('--assignment required')
        return train_task(root,repo,Path(args.assignment),args.pause)
    elif args.command=='oracle':
        from .search import serve
        serve(root)
    elif args.command=='search':
        from .search import run
        run(root,repo)
    elif args.command=='verify':
        from .analysis import verify_complete
        print(verify_complete(root,repo))
    elif args.command=='score-H':
        raise PermissionError('Prompt 2 forbids H scoring; H is not certified and Prompt 3 is not authorized')
    return 0

if __name__=='__main__':
    sys.exit(main())
