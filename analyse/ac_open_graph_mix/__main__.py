import argparse
import json
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','status','worker','gp_worker','job','score','timer'])
    p.add_argument('--run-dir',type=Path,default=Path('/home/fangxy/D-VAE/results/ac_open_graph_mix_20261012v1'))
    p.add_argument('--prompt',type=Path);p.add_argument('--recover',action='store_true');p.add_argument('--index',type=int,default=0)
    p.add_argument('--method',choices=['A','C']);p.add_argument('--group',type=int);p.add_argument('--n',type=int);p.add_argument('--post-only',action='store_true');a=p.parse_args()
    if a.action=='status':
        from .common import read
        print(json.dumps({n:read(a.run_dir/n) for n in ['STATUS.json','ETA.json','acceptance_status.json','ACCEPTANCE.json','SANITY.json'] if (a.run_dir/n).exists()},ensure_ascii=False,indent=2));return
    if a.action in ['run','prepare']:
        from .prepare import prepare
        try:prepare(a.run_dir,a.prompt)
        except BaseException:
            import traceback,os
            from .common import write,now
            write(a.run_dir/'STATUS.json',{'state':'FAILED','pid':os.getpid(),'phase':'preparation/CPU_checks','updated_at':now(),'error':traceback.format_exc()})
            raise
        if a.action=='prepare':return
        from .runtime import controller
        controller(a.run_dir,a.recover);return
    if a.action=='worker':
        from .runtime import training_worker
        training_worker(a.run_dir,a.index);return
    if a.action=='gp_worker':
        from .runtime import gp_worker
        gp_worker(a.run_dir,a.index);return
    if a.action=='job':
        from .runtime import job
        job(a.run_dir,a.method,a.group,a.n,a.post_only);return
    if a.action=='score':
        from .scoring import score
        score(a.run_dir);return
    while not (a.run_dir/'STATUS.json').exists():time.sleep(2)
    from .report import timer
    timer(a.run_dir)

if __name__=='__main__':main()
