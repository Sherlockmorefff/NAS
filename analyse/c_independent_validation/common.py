import hashlib, json, os
from pathlib import Path
from datetime import datetime, timezone

PREVIOUS='results/C_provenance_evidence_closure_20261006T122947Z'
MINIMAL='results/C_independence_minimal_audit_20261006T113512Z'
P4='results/structured_kernel_arch_main_ablation_20261006T065616Z'
EXCLUDED='5c3ed607fdb7f89cbbdf4126e6492829cbe9d794714682cdfec414593ebbfabc'
ZERO={k:0 for k in ('GP_fits','GP_starts','GNN_tasks','VAE_updates','decoder_calls','C_performance_reads','B_score_decodes','historical_test_score_decodes','test_evaluations','KG')}
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def log(p,x):
 with Path(p).open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
def verify(base,files):
 for p,h in files.items():
  if sha(base/p)!=h:raise ValueError('Identity mismatch: '+p)

