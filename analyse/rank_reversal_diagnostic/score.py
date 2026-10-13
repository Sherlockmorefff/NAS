"""Only this entry point decodes original test labels, after D1 seal."""
import time
import numpy as np
import torch
from .common import *
from .audit import make_model
from .data import load_training_view

def validate_test_tasks(tasks):
    assert all(t['phase'] == 'D1' and t['split'] == 'original' for t in tasks), 'D2 test forbidden'

def score(root, repo):
    verify_run(root, repo)
    sealed = read(root / 'D1_checkpoint_seal.json')
    assert sealed['training_terminal'] and sealed['test_evaluations_at_seal'] == 0
    tasks = sealed['successful_tasks']
    validate_test_tasks(tasks)
    verify(root, sealed['files'])
    if (root / 'D4_complete.json').exists():
        verify(root, read(root / 'D4_complete.json')['files'])
        return
    # Decode ty only here. Original Cora test.index permutation matches PyG.
    from torch_geometric.io.planetoid import read_file
    manifest = read(root / 'dataset_manifest.json')
    raw = Path(manifest['storage_root']) / 'raw'
    expected = {f['relative_path']: f['sha256'] for f in manifest['source_files']}
    assert sha(raw / 'ind.cora.ty') == expected['raw/ind.cora.ty']
    assert sha(raw / 'ind.cora.test.index') == expected['raw/ind.cora.test.index']
    ty = read_file(str(raw), 'cora', 'ty').argmax(1)
    test_index = read_file(str(raw), 'cora', 'test.index')
    # PyG: y[unsorted_test_index] = cat(ally,ty)[sorted_test_index].
    assert torch.equal(test_index.sort()[0], torch.arange(1708, 2708))
    ytest = torch.full((manifest['num_nodes'],), -100, dtype=torch.long)
    ytest[test_index] = ty
    idx = test_index.sort()[0]
    assert idx.tolist() == read(root / 'split_manifest.json')['splits']['original']['indices']['test']
    log(root / 'test_access.jsonl', {'time': now(), 'phase': 'D4', 'label_source': str(raw / 'ind.cora.ty'),
        'D1_seal_sha256': sha(root / 'D1_checkpoint_seal.json'), 'D2_authorized': False,
        'test_label_decode': 1, 'training_updates': 0})
    candidates = {c['config_id']: c for c in read(root / 'candidate_manifest.json')['candidates']}
    device = torch.device(read(root / 'resource_lock.json')['device'])
    torch.set_num_threads(1)
    files = {}
    for task in tasks:
        outpath = root / 'test_evaluations' / (task['task_id'] + '.json')
        folder = root / 'training' / task['task_id']
        ckpt = folder / 'best_model.pt'
        h = sha(ckpt)
        if outpath.exists():
            previous = read(outpath)
            assert previous['checkpoint_sha256'] == h and previous['seal_sha256'] == sha(root / 'D1_checkpoint_seal.json')
        else:
            started = time.perf_counter()
            data, in_ch, out_ch, _ = load_training_view(root, task, device)
            model = make_model(candidates[task['config_id']]['configuration'], in_ch, out_ch, device)
            saved = torch.load(ckpt, map_location=device, weights_only=False)
            assert saved['identity'] == task_identity(root, task)
            model.load_state_dict(saved['model'])
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                pred = logits.argmax(1).cpu()
            # Reuse same forward for restoration cross-check and test accuracy.
            val_idx = data.val_mask.nonzero().flatten().cpu()
            val = float(pred[val_idx].eq(data.y.cpu()[val_idx]).sum().item() / len(val_idx))
            assert val == saved['selection']
            acc = float(pred[idx].eq(ytest[idx]).sum().item() / len(idx))
            assert sha(ckpt) == h
            write(outpath, {'task_id': task['task_id'], 'config_id': task['config_id'], 'seed': task['seed'],
                'phase': 'D4_old_task_repeated_diagnosis', 'test_accuracy': acc, 'selection_recheck': val,
                'checkpoint_sha256': h, 'seal_sha256': sha(root / 'D1_checkpoint_seal.json'),
                'forward_count': 1, 'training_updates': 0, 'elapsed_seconds': time.perf_counter() - started, 'time': now()})
            np.savez_compressed(outpath.with_suffix('.npz'), indices=idx.numpy(), correct=pred[idx].eq(ytest[idx]).numpy())
        for p in [outpath, outpath.with_suffix('.npz')]:
            files[str(p.relative_to(root))] = sha(p)
    verify(root, sealed['files'])
    write(root / 'D4_complete.json', {'files': files, 'D1_test_forwards': len(tasks), 'D2_test_forwards': 0, 'time': now()})
