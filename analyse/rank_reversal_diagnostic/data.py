"""Reconstruct exact public tensors without opening ty or processed full labels."""
from types import SimpleNamespace
import numpy as np
import torch
from .common import *

def mask(indices, n):
    out = torch.zeros(n, dtype=torch.bool)
    out[indices] = True
    return out

def reconstruct_development(repo, manifest):
    from torch_geometric.io.planetoid import read_file, edge_index_from_dict
    from torch_geometric.data import Data
    from torch_geometric.transforms import NormalizeFeatures
    from dataset_utils import _tensor_fingerprint
    storage = Path(manifest['storage_root'])
    # Byte hashes bind even unopened label archives; no label decoding here.
    for f in manifest['source_files']:
        assert sha(storage / f['relative_path']) == f['sha256'], f['relative_path']
    raw = storage / 'raw'
    allx, tx, graph, test_index, ally, y_train = [
        read_file(str(raw), 'cora', name)
        for name in ('allx', 'tx', 'graph', 'test.index', 'ally', 'y')]
    n = manifest['num_nodes']
    train_index = list(range(y_train.shape[0]))
    val_index = list(range(y_train.shape[0], y_train.shape[0] + 500))
    assert not set(train_index + val_index) & set(test_index.tolist())
    x = torch.cat([allx, tx], 0)
    x[test_index] = x[test_index.sort()[0]]
    edges = edge_index_from_dict(graph, num_nodes=n)
    x = NormalizeFeatures()(Data(x=x)).x
    labels = torch.full((n,), -100, dtype=torch.long)
    dev = train_index + val_index
    labels[dev] = ally[dev].argmax(1)
    assert torch.equal(labels[train_index], y_train.argmax(1))
    original = {'train': train_index, 'selection': val_index,
                'early_stop': val_index, 'test': sorted(test_index.tolist())}
    tensors = {'x': x, 'edge_index': edges, 'train_mask': mask(train_index, n),
               'val_mask': mask(val_index, n), 'test_mask': mask(original['test'], n)}
    for key, value in tensors.items():
        assert _tensor_fingerprint(value) == manifest['tensor_fingerprints'][key], key
    return {'x': x, 'edge_index': edges, 'development_y': labels,
            'num_features': manifest['num_features'], 'num_classes': manifest['num_classes']}, original

def describe_split(roles, labels, split_seed=None):
    test = set(roles['test'])
    assert not test & (set(roles['train']) | set(roles['selection']) | set(roles['early_stop']))
    assert not set(roles['train']) & set(roles['selection'])
    assert roles['early_stop'] == roles['selection']
    counts = {r: {str(int(c)): int((labels[roles[r]] == c).sum())
                  for c in range(7)} for r in ('train', 'early_stop', 'selection')}
    return {'indices': roles, 'counts': {r: len(v) for r, v in roles.items()},
            'class_counts': counts, 'index_hashes': {r: digest(v) for r, v in roles.items()},
            'identity': digest(roles), 'split_seed': split_seed,
            'test_class_distribution': 'NOT_READ'}

def make_splits(original, labels):
    splits = {'original': describe_split(original, labels)}
    pool = sorted(set(original['train'] + original['selection']))
    quotas = splits['original']['class_counts']['train']
    seen = {splits['original']['identity']}
    next_seed = 2026100701
    rejected = []
    for i in range(5):
        while True:
            seed = next_seed
            next_seed += 1
            rng = np.random.default_rng(seed)
            train = []
            val = []
            for c in range(7):
                members = sorted(n for n in pool if int(labels[n]) == c)
                selected = rng.permutation(members).tolist()
                q = quotas[str(c)]
                assert len(selected) >= q
                train.extend(selected[:q])
                val.extend(selected[q:])
            roles = {'train': sorted(train), 'early_stop': sorted(val),
                     'selection': sorted(val), 'test': original['test']}
            split = describe_split(roles, labels, seed)
            if split['identity'] not in seen:
                break
            rejected.append(seed)
        assert split['class_counts'] == splits['original']['class_counts']
        seen.add(split['identity'])
        splits[f'split{i + 1}'] = split
    overlaps = []
    for a, av in splits.items():
        for b, bv in splits.items():
            for role in ('train', 'selection'):
                aa, bb = set(av['indices'][role]), set(bv['indices'][role])
                overlaps.append({'a': a, 'b': b, 'role': role,
                                 'intersection': len(aa & bb), 'Jaccard': len(aa & bb) / len(aa | bb)})
    return {'splits': splits, 'development_pool': pool, 'development_pool_hash': digest(pool),
            'rejected_duplicate_seeds': rejected, 'overlaps': overlaps,
            'algorithm': 'NumPy default_rng(PCG64), ascending classes; permutation of sorted class members; original train quotas; remainder shared early-stop/selection',
            'test_labels_used': False}

def load_training_view(root, task, device):
    from dataset_utils import _tensor_fingerprint
    saved = torch.load(root / 'development_data.pt', map_location='cpu', weights_only=False)
    split = read(root / 'split_manifest.json')['splits'][task['split']]
    roles = split['indices']
    n = len(saved['development_y'])
    assert digest(roles) == split['identity']
    assert torch.all(saved['development_y'][roles['test']] == -100)
    view = SimpleNamespace(x=saved['x'].to(device), edge_index=saved['edge_index'].to(device),
                           y=saved['development_y'].to(device),
                           train_mask=mask(roles['train'], n).to(device),
                           val_mask=mask(roles['selection'], n).to(device))
    for key, role in [('train_mask', 'train'), ('val_mask', 'selection')]:
        actual = getattr(view, key).nonzero().flatten().cpu().tolist()
        assert digest(actual) == split['index_hashes'][role]
    assert not hasattr(view, 'test_mask')
    return view, saved['num_features'], saved['num_classes'], split

def block_original_data(repo):
    """Process-local audit gate after the development view is available."""
    import sys
    banned = (str((repo / 'Cora').resolve()),)
    def gate(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = str(Path(os.fsdecode(args[0])).resolve())
            if any(p == b or p.startswith(b + os.sep) for b in banned):
                raise PermissionError('Training process cannot open original dataset: ' + p)
    sys.addaudithook(gate)
