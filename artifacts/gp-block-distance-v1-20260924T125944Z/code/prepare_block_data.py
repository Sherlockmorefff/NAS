"""One-shot data preparation and independent numerical scaler check, all charged."""
import argparse,importlib.util,json,pathlib,time
from block_core import Budget,budget_limit
from block_data import prepare,run_checks

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main(root):
 b=Budget(root/'budget.json',limit=budget_limit(root));result=prepare(root,budget=b);write(root/'verification/data_preparation.json',result)
 check=run_checks();write(root/'verification/data_synthetic.json',check)
 spec=importlib.util.spec_from_file_location('independent_data_audit',root/'audit/verify_block_data.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 started=time.monotonic()
 try:report=module.audit(root,None)
 finally:b.charge(time.monotonic()-started,'verification');b.persist()
 write(root/'audit/block_data_review.json',report)
 if not report['pass']:raise RuntimeError(str(report['failures']))
 print(json.dumps({'preparation':result,'independent_data_checks':len(report['checks']),'independent_pass':report['pass'],'model_seconds':b.seconds}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);main(p.parse_args().root)
