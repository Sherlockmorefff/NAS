import json
import os
import sqlite3
import time
from pathlib import Path
from analyse.wi_schur_B_pool300.common import seed,now,digest,sha,read,write,seal,append,csv_write,pid_alive

METHODS=('A','C')
KERNELS={'A':'RAW16','C':'W-I'}
TRAJECTORIES=[(m,g) for g in range(5) for m in METHODS]

def folder(root,method,g):return Path(root)/'trajectories'/method/str(g)
def inputs(root,g):return Path(root)/'inputs'/'groups'/str(g)
def b_root(root):return Path(read(Path(root)/'PROTOCOL.json')['B_root'])
def legacy_roots(root):return [Path(x) for x in read(Path(root)/'PROTOCOL.json')['cache_sources']]
def proposal_seed(g,phase,index):
    import numpy as np
    return int(np.random.SeedSequence([2026101223,g,phase,index]).generate_state(1)[0])

def connect(root):
    db=sqlite3.connect(str(Path(root)/'tasks.sqlite'),timeout=60);db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');return db

def init_db(root):
    with connect(root) as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,cid TEXT,config TEXT,seed INTEGER,epochs INTEGER,patience INTEGER,
          stage TEXT,status TEXT,source TEXT,result_root TEXT,creator_method TEXT,creator_group INTEGER,
          worker INTEGER,pid INTEGER,started REAL,finished REAL,elapsed REAL,error TEXT,attempts INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS logical(method TEXT,g INTEGER,phase TEXT,slot TEXT,task TEXT,cache_source TEXT,
          requested REAL,PRIMARY KEY(method,g,phase,slot));
        CREATE TABLE IF NOT EXISTS attempts(task TEXT,attempt INTEGER,worker INTEGER,pid INTEGER,started REAL,
          finished REAL,exit_code INTEGER,error TEXT,PRIMARY KEY(task,attempt));
        ''')

def task_key(protocol,config,train_seed,phase):
    recipe=protocol['search_recipe' if phase=='search' else 'final_recipe']
    return digest({'configuration':config,'dataset':protocol['dataset_context'],'recipe':recipe,'seed':int(train_seed),
                   'evaluation_code_identity':protocol['evaluation_code_identity']})

def enqueue(root,method,g,phase,slot,row,train_seed):
    protocol=read(Path(root)/'PROTOCOL.json');recipe=protocol['search_recipe' if phase=='search' else 'final_recipe']
    key=task_key(protocol,row['configuration'],train_seed,phase)
    with connect(root) as db:
        old=db.execute('SELECT * FROM logical WHERE method=? AND g=? AND phase=? AND slot=?',(method,g,phase,str(slot))).fetchone()
        if old:
            if old['task']!=key:raise ValueError('Logical method/group slot changed')
            return key
        physical=db.execute('SELECT * FROM tasks WHERE id=?',(key,)).fetchone()
        if physical:
            cache=physical['source'] if physical['source']!='OPEN-new' else 'OPEN-shared'
        else:
            # Only query the one already decided/requested key. No B performance scan.
            hit=None;origin=None
            cache_started=time.perf_counter()
            for legacy in legacy_roots(root):
                with sqlite3.connect((legacy/'tasks.sqlite').as_uri()+'?mode=ro',uri=True) as old_db:
                    old_db.row_factory=sqlite3.Row
                    found=old_db.execute('SELECT * FROM tasks WHERE id=? AND status=?',(key,'succeeded')).fetchone()
                if found:hit=dict(found);origin=legacy;break
            source=('legacy-B' if origin==b_root(root) else 'legacy-AC') if hit else 'OPEN-new';cache=source
            result_root=Path(hit['result_root']) if hit and hit.get('result_root') else origin/'training' if hit else Path(root)/'training'
            if hit:
                source='legacy-B' if result_root.is_relative_to(b_root(root)) else 'legacy-AC';cache=source
            append(Path(root)/'cache_lookup_times.jsonl',{'task':key,'method':method,'group':g,'phase':phase,'hit':bool(hit),'seconds':time.perf_counter()-cache_started,'at':now()})
            if hit and (json.loads(hit['config'])!=row['configuration'] or hit['seed']!=train_seed or hit['epochs']!=recipe['max_epochs'] or hit['patience']!=recipe['patience']):
                raise ValueError('Exact B cache metadata mismatch')
            db.execute('INSERT INTO tasks(id,cid,config,seed,epochs,patience,stage,status,source,result_root,creator_method,creator_group,elapsed,finished) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (key,row['candidate_id'],json.dumps(row['configuration'],sort_keys=True),int(train_seed),recipe['max_epochs'],recipe['patience'],phase,
                 'succeeded' if hit else 'queued',source,str(result_root),method,g,hit['elapsed'] if hit else None,hit['finished'] if hit else None))
        db.execute('INSERT INTO logical VALUES(?,?,?,?,?,?,?)',(method,g,phase,str(slot),key,cache,time.time()))
    return key

def task(root,key):
    with connect(root) as db:r=db.execute('SELECT * FROM tasks WHERE id=?',(key,)).fetchone()
    if r is None:raise ValueError('Unrequested physical task')
    return dict(r)

def result(root,key):
    r=task(root,key)
    if r['status']=='failed':raise RuntimeError('Training failed '+key+': '+str(r['error']))
    if r['status']!='succeeded':return None
    value=read(Path(r['result_root'])/key/'result.json')
    if value['identity']!=key or value['test_metric_computations']!=0 or value['seed']!=r['seed']:raise ValueError('Invalid committed/cached observation')
    return value

def model_record(root,key):
    r=task(root,key);p=Path(r['result_root'])/key/'best_model.pt';v=result(root,key)
    if v is None:raise ValueError('Final model not committed')
    h=sha(p)
    if h!=v['checkpoint_hashes']['best_model.pt']:raise ValueError('Final checkpoint identity changed')
    return {'task_id':key,'seed':r['seed'],'checkpoint':str(p),'checkpoint_sha256':h,'cache_source':r['source'],
       'prior_test_exposure':any((legacy/sub/(h+'.json')).exists() for legacy in legacy_roots(root) for sub in ['test_scores','new_test_scores']),
       'validation_accuracy':v['best_validation_accuracy']}

def counts(root):
    with connect(root) as db:
        physical=[dict(r) for r in db.execute('SELECT source,status,count(*) n FROM tasks GROUP BY source,status')]
        logical=[dict(r) for r in db.execute('SELECT l.method,l.phase,t.status,l.cache_source,count(*) n FROM logical l JOIN tasks t ON t.id=l.task GROUP BY l.method,l.phase,t.status,l.cache_source')]
        attempts=db.execute('SELECT count(*),sum(CASE WHEN exit_code!=0 THEN 1 ELSE 0 END) FROM attempts').fetchone()
    return {'physical':physical,'logical':logical,'training_attempts':attempts[0],'failed_attempts':attempts[1] or 0,
            'planned_logical_AC':6300,'planned_per_method':3150,'B_new_trajectories':0}
