"""Prepare a NEW, protocol-identical run from this bundle; never touch its files."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

def main(bundle,out):
    bundle=bundle.resolve();out=out.resolve()
    if out.exists():raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    for folder in ['code','source','training','queries']:
        shutil.copytree(bundle/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__','._*'))
    for name in ['PROMPT.md','protocol.json','source_manifest.json','worklist.json','folds.json','subsets.json','public_inputs.json']:
        shutil.copyfile(bundle/name,out/name)
    protocol=json.loads((out/'protocol.json').read_text())
    protocol['task_id']=out.name;protocol['frozen_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    protocol['reproduced_from_bundle']=str(bundle)
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,str(out/'code/preflight_residual.py'),'--root',str(out)],env=env,check=True)
    start=time.monotonic()
    result=subprocess.run([sys.executable,'-m','unittest','-v','test_residual_scoring.py'],cwd=out/'code',env=env,capture_output=True,text=True)
    elapsed=time.monotonic()-start
    v=out/'verification';(v/'scoring_preflight.log').write_text(result.stdout+result.stderr)
    (v/'scoring_preflight.json').write_text(json.dumps({'pass':result.returncode==0,'verification_seconds':elapsed,'synthetic_updates':0},indent=2)+'\n')
    result.check_returncode()
    sys.path.insert(0,str(out/'code'))
    from residual_core import Budget
    from prepare_residual import freeze,write
    m=json.loads((v/'model_preflight_summary.json').read_text())
    summary={'pass':m['pass'],'verification_seconds':m['verification_seconds']+elapsed,'synthetic_optimizer_updates':m['synthetic_optimizer_updates'],
       'only_synthetic_data':True,'formal_updates':0}
    write(v/'preflight_summary.json',summary)
    b=Budget(out/'resource_ledger.json',1200);b.charge(summary['verification_seconds'],'verification',summary['synthetic_optimizer_updates']);b.persist()
    freeze(out)
    print('Prepared fresh run: '+str(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.bundle,args.output)
