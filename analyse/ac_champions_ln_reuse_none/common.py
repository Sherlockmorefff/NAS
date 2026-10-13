import csv
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'ac_champions_ln_reuse_none_20261013v2'
SOURCE = Path('/home/fangxy/D-VAE/results/ac_open_graph_mix_20261012v1')
ROOT = Path('/home/fangxy/D-VAE/results/ac_champions_ln_reuse_none_20261013v1')
MODULE = 'analyse.ac_champions_ln_reuse_none'
TRAJECTORIES = [(m, g) for g in range(5) for m in ('A', 'C')]
EXPECTED_CIDS = {
 ('A',0):'f2f5f48d90c350a075e81f07fff627255293f5bfa715bd425014cf75933c7609',
 ('A',1):'c3ab008fdc94c4a547fb39d8f43314e76e3526d4668316b0e55527ee12c25965',
 ('A',2):'dfe5293a15df019759d5165d8cc7210e29078359d530047462a64a49e0bb8d1c',
 ('A',3):'5e89d6aa7258430b57cf1824261d0d5acbd992bf1ad8a9ec49553f7ef6dd2ce8',
 ('A',4):'4b8e5bd8c79330387c28c1346834a5ce210272f0723da7a32a005b9b0edd5b70',
 ('C',0):'c084dc51c23abfe95e26bae5a9e7d535c14690a51a0e00104eae68788c67a599',
 ('C',1):'4d5e139523b1bed3c6e2232993dd4857aa0f62249b2b464117b88e9f907220f2',
 ('C',2):'8a92e4edadc38c9a0da5124d1adbb463d1c8d01e4d814a062929d7bd390633a7',
 ('C',3):'e4161a04babc9309963ac24cd43fb28d9cc1b257643507abe430ff6d4fd068d1',
 ('C',4):'97efff9d1795084e4b8cb19752e8cf61d41da649e70cef54fa176d4291736134'}

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)

def seal(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError('Frozen artifact differs: ' + str(path))
    else:
        write_json(path, value)

def append_jsonl(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()

def csv_write(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def csv_read(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def connect(root):
    db = sqlite3.connect(str(root/'tasks.sqlite'), timeout=60)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    return db

def tasks(root):
    return read_json(root/'LN_TASK_MANIFEST.json')['tasks']

def result(root, task):
    value = read_json(root/'training'/task['task_id']/'result.json')
    if value['identity'] != task['task_id'] or value['seed'] != task['seed'] or value['test_metric_computations'] != 0 or value['test_score_reads'] != 0:
        raise ValueError('Invalid LN result identity/test boundary')
    if value['component_variant'] != 'layer_norm' or not value['valid']:
        raise ValueError('Invalid LN condition/result')
    if sha256(root/'training'/task['task_id']/'best_model.pt') != value['checkpoint_hashes']['best_model.pt']:
        raise ValueError('LN checkpoint changed')
    return value

def verify(root, baseline_weights=False):
    frozen = read_json(root/'freeze.json')
    if frozen['identity'] != digest({k:v for k,v in frozen.items() if k != 'identity'}):
        raise ValueError('Freeze identity differs')
    for p,h in frozen['source'].items():
        if sha256(p) != h:
            raise ValueError('Source changed: '+p)
    for p,h in frozen['inputs'].items():
        if sha256(root/p) != h:
            raise ValueError('Frozen input changed: '+p)
    for p,h in read_json(root/'BASELINE_SOURCE_FILES.json').items():
        if sha256(p) != h:
            raise ValueError('Historical source changed: '+p)
    if baseline_weights:
        for b in read_json(root/'BASELINE_MANIFEST.json')['rows']:
            if not b['weights_verified_from_source_manifest_only'] and sha256(b['checkpoint']) != b['checkpoint_sha256']:
                raise ValueError('Historical None checkpoint changed')
    return frozen
