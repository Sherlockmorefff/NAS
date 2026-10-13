import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

OLD = 'results/C_independent_validation_20261006T133947Z'
DATA_MANIFEST = 'results/cora-layernorm-search-space-pilot-v1/expanded/search_seed10/dataset_manifest.json'
FROZEN = ['diagnostic_protocol.json', 'candidate_manifest.json', 'seed_manifest.json',
          'split_manifest.json', 'comparison_manifest.json', 'training_recipe.json',
          'task_manifest.json', 'dataset_manifest.json', 'development_data.pt']

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def log(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

def jsonlines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []

def verify(base, files):
    for name, h in files.items():
        if sha(Path(base) / name) != h:
            raise ValueError('Frozen identity mismatch: ' + str(name))

def csvwrite(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def verify_run(root, repo):
    value = read(root / 'freeze.json')
    verify(root, value['inputs'])
    verify(repo, value['code'])
    return value['identity']

def task_identity(root, task):
    frozen = read(root / 'freeze.json')
    return digest({'run': frozen['identity'], 'task': task,
                   'dataset': frozen['inputs']['dataset_manifest.json'],
                   'masks': read(root / 'split_manifest.json')['splits'][task['split']]['identity'],
                   'recipe': frozen['inputs']['training_recipe.json']})
