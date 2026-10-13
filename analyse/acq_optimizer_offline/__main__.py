import argparse
import json
from pathlib import Path
from .common import ROOT,setup

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','status','resume','supervisor','compute','acceptance'])
    p.add_argument('--run-dir',type=Path,default=ROOT);p.add_argument('--prompt',type=Path);p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    if a.action=='status':
        from .common import read
        print(json.dumps({n:read(a.run_dir/n) for n in ['status.json','ETA.json','acceptance_status.json','decision.json'] if (a.run_dir/n).exists()},ensure_ascii=False,indent=2));return 0
    if a.action=='prepare':
        setup();from .checks import prepare,sanity
        prepare(a.run_dir,a.prompt);sanity(a.run_dir);return 0
    from .runner import controller,compute,supervisor
    if a.action in ['run','resume']:return controller(a.run_dir,a.prompt,a.action=='resume')
    if a.action=='compute':compute(a.run_dir,a.worker);return 0
    if a.action=='supervisor':return supervisor(a.run_dir,a.worker)
    from .report import acceptance_loop
    return acceptance_loop(a.run_dir)

if __name__=='__main__':raise SystemExit(main())
