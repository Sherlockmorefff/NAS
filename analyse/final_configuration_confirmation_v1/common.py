import csv
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from fractions import Fraction
from pathlib import Path
from functools import lru_cache

MODULE = 'analyse.final_configuration_confirmation_v1'
OLD = 'results/C_independent_validation_20261006T133947Z'
DIAG = 'results/rank_reversal_diagnostic_20261007T075829Z'
RULES = ('U', 'S', 'M')

def read(path):
    return json.loads(Path(path).read_text())

@lru_cache(maxsize=None)
def frozen_read(path):
    return read(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def times(timestamp):
    dt = datetime.fromtimestamp(timestamp, timezone.utc)
    return {'UTC': dt.isoformat(), 'Beijing': dt.astimezone(timezone(timedelta(hours=8))).isoformat()}

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)

def seal(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Sealed artifact differs: ' + str(path))
    else:
        write(path, value)

def log(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False, ensure_ascii=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())

def csvwrite(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)

def verify(base, manifest):
    for name, expected in manifest.items():
        if sha(Path(base) / name) != expected:
            raise ValueError('Identity mismatch: ' + str(name))

def verify_run(root, repo):
    frozen = read(root / 'FREEZE.json')
    verify(root, frozen['inputs'])
    verify(repo, read(root / 'CODE_MANIFEST.json')['files'])
    return frozen['identity']

def exclusive(path):
    stream = Path(path).open('a+')
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream

def effective_config(c):
    keys = ['operations', 'edges', 'effective_layers', 'lr', 'dropout', 'hidden_dim',
        'l2', 'gcnii_alpha', 'gcnii_theta', 'gat_heads', 'sage_aggr', 'gin_eps',
        'gat_heads_by_layer', 'sage_aggr_by_layer', 'gin_eps_by_layer', 'hp_mode', 'normalization']
    result = {k: c[k] for k in keys}
    result.update(activation_variant='relu', component_variant='baseline')
    return result

def exact_mean(rows):
    if not rows:
        raise ValueError('Empty exact mean')
    return sum((Fraction(r['correct'], r['denominator']) for r in rows), Fraction()) / len(rows)

def ordered_observations(observations, configs):
    ids = [o['candidate_id'] for o in observations]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate observed configuration')
    return sorted(observations, key=lambda o: (
        -Fraction(o['query_correct_sum'], o['query_denominator']),
        configs[o['candidate_id']]['normalized_config_id']))

def physical_task(root, group, role, cid, seed, split):
    config = frozen_read(root / 'CONFIGURATIONS.json')['by_id'][cid]
    masks = frozen_read(root / 'SPLITS.json')['splits'][split]
    key = {'run': frozen_read(root / 'FREEZE.json')['identity'], 'group': group,
        'role': role, 'configuration_hash': config['configuration_hash'],
        'data': frozen_read(root / 'FREEZE.json')['inputs']['development_data.pt'],
        'masks': masks['identity'], 'seed': seed,
        'recipe': frozen_read(root / 'FREEZE.json')['inputs']['TRAINING_RECIPE.json'],
        'code': frozen_read(root / 'FREEZE.json')['inputs']['CODE_MANIFEST.json']}
    identity = digest(key)
    return {'task_id': identity, 'scientific_identity': key, 'group': group, 'role': role,
        'candidate_id': cid, 'normalized_config_id': config['normalized_config_id'],
        'configuration_hash': config['configuration_hash'], 'seed': seed, 'split': split,
        'parameter_count': config['parameter_count']}

def execution_identity(root, task, gpu):
    return digest({'task': task, 'execution': {'GPU_UUID': gpu['uuid'],
        'physical_index': gpu['index'], 'logical_device': 'cuda:0',
        'precision': 'float32', 'strict_determinism': True}})

def access_guard(root, repo, role):
    root, repo = root.resolve(), repo.resolve()
    allowed_names = {'FREEZE.json', 'PROTOCOL.json', 'CODE_MANIFEST.json', 'PUBLIC_C.json',
        'INITIALIZATIONS.json', 'SEEDS.json', 'GPUS.json', 'CONFIGURATIONS.json',
        'SPLITS.json', 'TRAINING_RECIPE.json', 'development_data.pt'}
    def gate(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(repo / 'Cora') or path.is_relative_to(repo / 'data') or path.is_relative_to(repo / 'raw'):
            raise PermissionError('Original label/data storage blocked')
        if path.is_relative_to(root):
            rel = path.relative_to(root)
            if role == 'search' and not (rel.parts[0] in {'trajectories', 'search_ready'} or str(rel) in allowed_names
                or rel.name.startswith(('SEARCH_COMPLETE.json', 'SEARCH_PROGRESS.json'))):
                raise PermissionError('Search cannot access unreleased labels or training results')
            if role == 'train' and rel.parts[0] in {'private', 'trajectories', 'returns', 'validation_matrix'}:
                raise PermissionError('Training cannot access selection or historical scores')
            return
        if any(path.is_relative_to(repo / p) for p in ['results', 'legacy_artifacts', 'artifacts']):
            raise PermissionError('Other experiment results blocked')
    sys.addaudithook(gate)

def require_seal(root, group, rule, task):
    path = root / 'returns' / f'g{group:02d}_{rule}.json'
    sealed = read(path)
    if sealed['candidate_id'] != task['candidate_id'] or task['seed'] not in sealed['evaluation_seeds']:
        raise ValueError('Evaluation must match an immutable returned configuration')
    return sha(path)
