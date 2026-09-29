"""Zero-update recheck after state-identity-only preflight fix; shared budget."""
import argparse,json
from pathlib import Path
from readout_core import budget_limit
from readout_core import Budget,cpu_setup
from preflight_readout import run_checks
from readout_diagnostics import run_checks as diagnostics
p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();cpu_setup(123)
b=Budget(a.root/'budget.json',limit=budget_limit(a.root));out=a.root/'verification/final_identity_check';out.mkdir()
r=run_checks(out,b,'math')
if r['pass']:r['diagnostics']=diagnostics(out,b);r['pass']=r['diagnostics']['pass']
r['change']='Fingerprint includes fixed model configuration; numerical equations, optimizer and timing implementation unchanged'
r['resources']=b.as_dict();(out/'acceptance.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'pass':r['pass'],'synthetic_total':b.synthetic_updates,'seconds':b.seconds}));raise SystemExit(0 if r['pass'] else 1)
