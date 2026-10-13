"""Atomic records, method-independent seeds, RBF-Schur and exact task keys."""
import csv
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

BASE = 2026101101
STAGES = {'lhs': 1, 'decoder': 2, 'search': 3, 'mc': 4, 'validation': 5, 'final': 6}

def seed(group, stage, index=0):
    return int(np.random.SeedSequence([BASE, group, STAGES[stage], index]).generate_state(1, dtype=np.uint32)[0])

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def seal(path, value):
    if Path(path).exists():
        if read(path)!=value: raise ValueError('Sealed artifact differs: '+str(path))
    else: write(path,value)

def append(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o644)
    try: os.write(fd,data);os.fsync(fd)
    finally: os.close(fd)

def csv_write(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        keys=sorted({k for r in rows for k in r}); w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows: w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def rbf_schur(rows, k=50):
    x=np.asarray([r['z_search'] for r in rows],dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=16 or not np.isfinite(x).all(): raise ValueError('Expected finite 16D pool')
    lo=x.min(0);span=np.maximum(x.max(0)-lo,1e-8); xn=(x-lo)/span
    d=xn[:,None,:]-xn[None,:,:]; kernel=np.exp(-np.sum(d*d,axis=2)/(2*16))
    if not np.isfinite(kernel).all(): raise FloatingPointError('Nonfinite RBF')
    remaining=list(range(len(rows)));selected=[];steps=[]
    while len(selected)<k:
        if not selected:
            score=np.sum((xn[remaining]-xn.mean(0))**2,axis=1); criterion='farthest_from_pool_mean'
        else:
            ks=kernel[np.ix_(selected,selected)]+np.eye(len(selected))*1e-6
            kr=kernel[np.ix_(remaining,selected)]
            chol=np.linalg.cholesky(ks);sol=np.linalg.solve(chol,kr.T)
            score=1-np.sum(sol*sol,axis=0);criterion='conditional_variance'
            if not np.isfinite(score).all() or np.min(score)<-1e-10: raise FloatingPointError('Invalid Schur variance')
        best=min(range(len(remaining)),key=lambda j:(-float(score[j]),rows[remaining[j]]['candidate_id']))
        picked=remaining.pop(best);steps.append({'step':len(selected)+1,'candidate_id':rows[picked]['candidate_id'],
            'score':float(score[best]),'criterion':criterion,'conditional_variance':1. if not selected else float(score[best])})
        selected.append(picked)
    return {'normalization_min':lo.tolist(),'normalization_range':span.tolist(),'ell':4.,'signal':1.,'epsilon':1e-6,
        'dtype':'float64','first_point':steps[0]['candidate_id'],'selection':steps,'ids':[rows[i]['candidate_id'] for i in selected],
        'labels_used':False,'kernel':'original_global_RBF_Schur'}

def connect(root):
    db=sqlite3.connect(str(Path(root)/'tasks.sqlite'),timeout=60)
    db.row_factory=sqlite3.Row;db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    return db

def init_db(root):
    with connect(root) as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, cid TEXT, config TEXT, seed INTEGER, epochs INTEGER,
          patience INTEGER, stage TEXT, status TEXT, worker INTEGER, pid INTEGER, started REAL, finished REAL,
          elapsed REAL, error TEXT, attempts INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS logical(g INTEGER, phase TEXT, slot TEXT, task TEXT,
          PRIMARY KEY(g,phase,slot));
        CREATE TABLE IF NOT EXISTS attempts(task TEXT, attempt INTEGER, worker INTEGER, pid INTEGER,
          started REAL, finished REAL, exit_code INTEGER, error TEXT, PRIMARY KEY(task,attempt));
        ''')

def enqueue(root, group, phase, slot, row, train_seed):
    protocol=read(Path(root)/'protocol.json'); recipe=protocol['search_recipe' if phase=='search' else 'final_recipe']
    payload={'configuration':row['configuration'],'dataset':protocol['dataset_context'],'recipe':recipe,
             'seed':train_seed,'evaluation_code_identity':protocol['evaluation_code_identity']}
    key=digest(payload)
    with connect(root) as db:
        db.execute('INSERT OR IGNORE INTO tasks(id,cid,config,seed,epochs,patience,stage,status) VALUES(?,?,?,?,?,?,?,?)',
           (key,row['candidate_id'],json.dumps(row['configuration'],sort_keys=True),train_seed,recipe['max_epochs'],recipe['patience'],phase,'queued'))
        old=db.execute('SELECT task FROM logical WHERE g=? AND phase=? AND slot=?',(group,phase,str(slot))).fetchone()
        if old and old['task']!=key: raise ValueError('Logical task changed')
        db.execute('INSERT OR IGNORE INTO logical VALUES(?,?,?,?)',(group,phase,str(slot),key))
    return key

def result(root, key):
    with connect(root) as db:
        row=db.execute('SELECT * FROM tasks WHERE id=?',(key,)).fetchone()
    if row['status']=='failed': raise RuntimeError('Training failed '+key+': '+str(row['error']))
    if row['status']!='succeeded': return None
    value=read(Path(root)/'training'/key/'result.json')
    if value['identity']!=key or value['test_metric_computations']!=0: raise ValueError('Invalid committed result')
    return value

def counts(root):
    with connect(root) as db:
        actual={r['status']:r['n'] for r in db.execute('SELECT status,count(*) n FROM tasks GROUP BY status')}
        logical={r['phase']:r['n'] for r in db.execute('SELECT phase,count(*) n FROM logical GROUP BY phase')}
    return {'actual':actual,'logical_enqueued':logical,'logical_planned':{'search':1500,'validation':1500,'final':150}}

def pid_alive(pid):
    try: os.kill(int(pid),0);return True
    except (OSError,TypeError): return False
