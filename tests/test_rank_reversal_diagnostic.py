from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
import eval_utils as original
from analyse.rank_reversal_diagnostic.common import digest, read, sha, write, task_identity
from analyse.rank_reversal_diagnostic.data import make_splits, mask
from analyse.rank_reversal_diagnostic.train import train, PlannedPause
from analyse.rank_reversal_diagnostic.score import validate_test_tasks
from analyse.rank_reversal_diagnostic.stats import paired

class Tiny(torch.nn.Module):
    def __init__(self, config, in_ch, out_ch, **kwargs):
        super().__init__()
        self.fc = torch.nn.Linear(in_ch, out_ch)
    def forward(self, x, edges):
        return self.fc(torch.nn.functional.dropout(x, .2, self.training))

def test_exact_training_checkpoint_and_interruption(tmp_path, monkeypatch):
    from analyse.structured_kernel_followup.validation_train import train_validation_only
    monkeypatch.setattr(original, 'DynamicGNN', Tiny)
    config = {'lr': .02, 'l2': .001, 'dropout': .2, 'hidden_dim': 4, 'gat_heads': 1, 'sage_aggr': 'mean', 'gin_eps': 0}
    x = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 36
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, -100, -100, -100, -100])
    data = SimpleNamespace(x=x, edge_index=torch.empty((2, 0), dtype=torch.long), y=y,
                           train_mask=mask([0, 1, 2, 3], 12), val_mask=mask([4, 5, 6, 7], 12))
    dev = torch.device('cpu')
    ref = train_validation_only(config, data, 3, 2, 100, dev, tmp_path / 'ref', 'ref', max_epochs=12, patience=20)
    with pytest.raises(PlannedPause):
        train(config, data, 3, 2, 100, dev, tmp_path / 'new', 'new', max_epochs=12, patience=20, pause_after=5)
    new = train(config, data, 3, 2, 100, dev, tmp_path / 'new', 'new', max_epochs=12, patience=20)
    assert new['selection'] == ref['best_validation_accuracy'] and new['best_epoch'] == ref['best_epoch']
    assert [(r['loss'], r['selection']) for r in new['trajectory']] == [(r['loss'], r['validation_accuracy']) for r in ref['trajectory']]
    assert new['model_final_before_restore_sha256'] == ref['model_final_before_restore_sha256']
    before = sha(tmp_path / 'new/result.json')
    train(config, data, 3, 2, 100, dev, tmp_path / 'new', 'new', max_epochs=12, patience=20)
    assert sha(tmp_path / 'new/result.json') == before
    with pytest.raises(AssertionError):
        train(config, data, 3, 2, 100, dev, tmp_path / 'new', 'mismatch', max_epochs=12, patience=20)

def test_mask_stratification_and_test_poison():
    original_split = {'train': list(range(140)), 'selection': list(range(140, 640)),
                      'early_stop': list(range(140, 640)), 'test': list(range(1000, 2000))}
    y = torch.full((2708,), -100)
    y[:640] = torch.arange(640) % 7
    first = make_splits(original_split, y)
    y[1000:2000] = 98765
    second = make_splits(original_split, y)
    assert first == second
    assert len({v['identity'] for v in first['splits'].values()}) == 6
    assert all(v['counts']['train'] == 140 and v['counts']['selection'] == 500 for v in first['splits'].values())

def test_task_identity_separates_seed_config_and_mask(tmp_path):
    write(tmp_path / 'freeze.json', {'identity': 'f', 'inputs': {'dataset_manifest.json': 'd', 'training_recipe.json': 'r'}})
    write(tmp_path / 'split_manifest.json', {'splits': {'a': {'identity': 'a'}, 'b': {'identity': 'b'}}})
    task = {'split': 'a', 'seed': 100, 'config_id': 'C0'}
    values = {task_identity(tmp_path, t) for t in [task, {**task, 'split': 'b'}, {**task, 'seed': 101}, {**task, 'config_id': 'C1'}]}
    assert len(values) == 4

def test_d2_test_forbidden():
    validate_test_tasks([{'phase': 'D1', 'split': 'original'}])
    with pytest.raises(AssertionError):
        validate_test_tasks([{'phase': 'D2', 'split': 'split1'}])

def test_original_dataset_access_blocked(tmp_path):
    (tmp_path / 'Cora').mkdir()
    (tmp_path / 'Cora/test').write_text('labels')
    program = 'from pathlib import Path\nfrom analyse.rank_reversal_diagnostic.data import block_original_data\nr=Path(' + repr(str(tmp_path)) + ')\nblock_original_data(r)\ntry:\n (r/"Cora/test").read_text()\nexcept PermissionError:\n print("DENIED")\nelse:\n raise AssertionError("test data leak")\n'
    p = subprocess.run([sys.executable, '-B', '-c', program], text=True, capture_output=True)
    assert p.returncode == 0 and 'DENIED' in p.stdout

def test_paired_interval_keeps_actual_pair_count():
    v = paired([1., -1., 0.])
    assert v['n'] == 3 and v['mean_pp'] == 0 and (v['wins'], v['ties'], v['losses']) == (1, 1, 1)
    assert v == paired([1., -1., 0.])
    assert paired([])['CI95_pp'] is None
