"""Exact original epoch and early-stop semantics, complete latest-state resume."""
import time
import numpy as np
import torch
import eval_utils as original
from analyse.structured_kernel_followup.validation_train import epoch_core, capture_rng, restore_rng, save_torch
from .audit import make_model
from .common import *
from .data import load_training_view, block_original_data

class PlannedPause(Exception):
    """A saved epoch boundary, never a failed/randomly restarted task."""

def train(config, data, in_ch, out_ch, seed, device, folder, identity,
          *, max_epochs=300, patience=80, pause_after=None, on_initialization=None):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        value = read(folder / 'result.json')
        assert value['identity'] == identity
        verify(folder, value['files'])
        return value
    original.assert_strict_torch_determinism(torch)
    assert not hasattr(data, 'test_mask')
    assert torch.all(data.y[~(data.train_mask | data.val_mask)] == -100)
    torch.set_default_dtype(torch.float32)
    checkpoint = folder / 'latest.pt'
    old = torch.load(checkpoint, map_location=device, weights_only=False) if checkpoint.exists() else None
    if old:
        assert old['identity'] == identity
        if old.get('completed_result'):
            value = old['completed_result']
            value['files']['latest.pt'] = sha(checkpoint)
            write(folder / 'result.json', value)
            return value
    start = time.perf_counter()
    previous = old['elapsed_seconds'] if old else 0.
    with original.isolated_rng(seed, device):
        if old is None and on_initialization:
            on_initialization()
        model = make_model(config, in_ch, out_ch, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=float(config['l2']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=float(config['lr']) * .01)
        best_val, best_epoch, no_improve, completed, best_state = 0., None, 0, 0, None
        trajectory = []
        gpu_seconds = 0.
        if old:
            model.load_state_dict(old['model'])
            optimizer.load_state_dict(old['optimizer'])
            scheduler.load_state_dict(old['scheduler'])
            best_val, best_epoch, no_improve = old['best_val'], old['best_epoch'], old['no_improve']
            best_state, completed, trajectory = old['best_state'], old['epoch'], old['trajectory']
            gpu_seconds = old['gpu_seconds']
            restore_rng(old['rng'])
        log(folder / 'attempts.jsonl', {'time': now(), 'identity': identity, 'resume_epoch': completed,
            'random_initialization': old is None, 'strict_resume': old is not None,
            'restored_fields': ['model', 'optimizer', 'scheduler', 'rng', 'epoch', 'best_state', 'best_epoch', 'no_improve'] if old else []})
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        def snapshot(completed_result=None):
            return {'identity': identity, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng': capture_rng(), 'epoch': completed,
                'best_state': best_state, 'best_val': best_val, 'best_epoch': best_epoch, 'no_improve': no_improve,
                'trajectory': trajectory, 'elapsed_seconds': previous + time.perf_counter() - start,
                'gpu_seconds': gpu_seconds, 'completed_result': completed_result}
        if old is None:
            save_torch(checkpoint, snapshot())
        for _ in range(completed, max_epochs):
            if no_improve >= patience:
                break
            if device.type == 'cuda':
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            loss, val = epoch_core(model, data, optimizer, scheduler)
            if device.type == 'cuda':
                end.record()
                end.synchronize()
                gpu_seconds += begin.elapsed_time(end) / 1000
            completed += 1
            trajectory.append({'epoch_1based': completed, 'loss': loss, 'selection': val})
            if val > best_val:
                best_val, best_epoch, no_improve = val, completed, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            save_torch(checkpoint, snapshot())
            log(folder / 'epoch_events.jsonl', {'epoch': completed, 'loss': loss, 'selection': val,
                'best_epoch': best_epoch, 'next_lr': optimizer.param_groups[0]['lr'], 'time': now()})
            if pause_after == completed:
                write(folder / 'planned_pause.json', {'time': now(), 'epoch': completed,
                    'latest_sha256': sha(checkpoint), 'trajectory_hash': digest(trajectory),
                    'saved_complete_state': True, 'random_initializations': 1})
                raise PlannedPause('Saved exact epoch ' + str(completed))
        if best_state is None:
            raise ValueError('No strict positive validation improvement; original recipe cannot select checkpoint')
        final_hash = original._model_state_hash(model)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            restored = float(out.argmax(1)[data.val_mask].eq(data.y[data.val_mask]).sum().item() / int(data.val_mask.sum()))
        assert restored == best_val, 'Best checkpoint restore changed selection'
        idx = data.val_mask.nonzero().flatten()
        np.savez_compressed(folder / 'selection.npz', indices=idx.cpu().numpy(),
            logits=out[idx].detach().cpu().numpy(), correct=out.argmax(1)[idx].eq(data.y[idx]).cpu().numpy())
        save_torch(folder / 'best_model.pt', {'identity': identity, 'model': best_state, 'best_epoch': best_epoch,
                                             'selection': best_val})
        value = {'identity': identity, 'state': 'COMPLETE', 'seed': seed,
            'selection': restored, 'early_stop': best_val, 'best_epoch': best_epoch, 'stopped_epoch': completed,
            'stop_reason': 'patience' if no_improve >= patience else 'max_epochs', 'best_restore_verified': True,
            'trajectory': trajectory, 'model_final_before_restore_sha256': final_hash,
            'worker_seconds': previous + time.perf_counter() - start, 'gpu_event_seconds': gpu_seconds,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'test_forwards': 0, 'dtype': 'float32', 'device': str(device), 'completed_at': now(),
            'files': {n: sha(folder / n) for n in ['best_model.pt', 'selection.npz']}}
        # Final result has restored model; no updates remain. Completed-result
        # recovery writes metadata only, never performs another forward pass.
        save_torch(checkpoint, snapshot(value))
        value['files']['latest.pt'] = sha(checkpoint)
        write(folder / 'result.json', value)
        return value

def worker(root, repo, task_id, pause_after=None):
    verify_run(root, repo)
    task = next(t for t in read(root / 'task_manifest.json')['tasks'] if t['task_id'] == task_id)
    c = next(c for c in read(root / 'candidate_manifest.json')['candidates'] if c['config_id'] == task['config_id'])
    assert digest(c['configuration']) == task['configuration_hash']
    folder = root / 'training' / task_id
    identity = task_identity(root, task)
    device = torch.device(read(root / 'resource_lock.json')['device'])
    torch.set_num_threads(1)
    data, in_ch, out_ch, split = load_training_view(root, task, device)
    block_original_data(repo)
    def initialize():
        entries = jsonlines(root / 'initializations.jsonl')
        if len(entries) >= 125:
            raise RuntimeError('125 initialization budget exhausted')
        if any(x['task_id'] == task_id for x in entries):
            raise RuntimeError('Task previously initialized without complete latest state; no automatic restart')
        log(root / 'initializations.jsonl', {'task_id': task_id, 'identity': identity, 'time': now(),
                                           'ordinal': len(entries) + 1})
    result = train(c['configuration'], data, in_ch, out_ch, task['seed'], device, folder, identity,
                   pause_after=pause_after, on_initialization=initialize)
    write(folder / 'binding.json', {'task': task, 'identity': identity, 'masks_hash': split['identity'],
        'actual_train_hash': digest(data.train_mask.nonzero().flatten().cpu().tolist()),
        'actual_selection_hash': digest(data.val_mask.nonzero().flatten().cpu().tolist()),
        'original_data_access_blocked': True, 'test_labels_in_view': False, 'test_forwards': 0,
        'result_sha256': sha(folder / 'result.json')})
    return result
