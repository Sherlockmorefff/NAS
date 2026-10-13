import csv
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timezone

BASE = 2026101207
ROOT = Path('/home/fangxy/D-VAE/results/acq_optimizer_offline_20261012v1')
B = Path('/home/fangxy/D-VAE/results/wi_schur_B_pool300_20261011v1')
AC = Path('/home/fangxy/D-VAE/results/wi_schur_AC_pool300_20261012v1')
TERMINAL = {'complete','complete_early','timeout','no_valid_return','failed'}
ZERO = dict(train_GNN_count=0, GP_parameter_updates=0, new_validation_observations=0,
            new_test_computations=0, H_computations=0)

def now(): return datetime.now(timezone.utc).isoformat()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,p)
def append(p,x):
    with Path(p).open('a') as f: f.write(canonical(x)+'\n');f.flush()
def csv_write(p,rows):
    keys=sorted({k for x in rows for k in x})
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for x in rows:w.writerow({k:canonical(v) if isinstance(v,(list,dict)) else v for k,v in x.items()})
def db(root):
    c=sqlite3.connect(Path(root)/'runs.sqlite',timeout=30,isolation_level=None)
    c.row_factory=sqlite3.Row;c.execute('PRAGMA journal_mode=WAL');return c
def alive(pid):
    if not pid:return False
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False
def setup():
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64);torch.use_deterministic_algorithms(True)
    # This entry is inference-only. Fail immediately if any optimizer is invoked.
    def prohibited(*args,**kwargs):raise RuntimeError('OFFLINE_ONLY: optimizer update prohibited')
    for cls in [torch.optim.Adam,torch.optim.LBFGS,torch.optim.SGD,torch.optim.AdamW]:cls.step=prohibited
    import surrogate.checkpoint_io
    import eval_utils
    for name in ['train_dynamic_gnn','train_and_eval_arch','eval_z_search','eval_arch']:
        if hasattr(eval_utils,name):setattr(eval_utils,name,prohibited)
def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def run_specs():
    import numpy as np
    rows=[]
    for n in [50,150,250]:
        for r in range(3):
            for family,code,methods in [('A',0,['GRADIENT','DE']),('WI',1,['RANDOM','DE'])]:
                order=methods if r%2==0 else methods[::-1]
                for method in order:
                    rows.append({'id':f'{family}_n{n:03d}_r{r}_{method}','family':family,'n':n,'replicate':r,'method':method,
                        'initial_seed':int(np.random.SeedSequence([BASE,code,n,r,1]).generate_state(1)[0]),
                        'optimizer_seed':int(np.random.SeedSequence([BASE,code,n,r,2]).generate_state(1)[0]),'order':len(rows)})
    return rows
