"""Synthetic engineering checks only; none of these records is real replay data.

The fake cached codes/checkpoint marker and fake validation oracle intentionally
avoid VAE/GNN training. The production historical replay is run separately.
"""

from collections import Counter
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from analyse.fuzzy_metric_replay import (
    HistoryOracle,
    Replay,
    digest,
    file_hash,
    json_write,
    p2_select,
)


class SyntheticOracle:
    def __init__(self, values):
        self.values = copy.deepcopy(values)
        self.calls = []

    def reveal(self, ids):
        self.calls.append(list(ids))
        return {cid: copy.deepcopy(self.values[cid]) for cid in ids}


@pytest.fixture(autouse=True)
def single_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


@pytest.fixture
def synthetic_pool(tmp_path):
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/fuzzy_metric_stage1_v1.json').read_text())
    config.update(initial_size=8, query_budget=20, representation_steps=3, gp_steps=3, acquisition_samples=16)
    prepared = tmp_path / 'synthetic_prepared'
    prepared.mkdir()
    generator = np.random.default_rng(8701)
    candidates, labels, encoding = [], {}, {}
    for i in range(48):
        cid, arch = f'joint_{i:03d}', f'arch_{i // 3:03d}'
        raw_hp = {'lr': 10.0 ** (-4 + (i % 3) * 0.6), 'dropout': 0.1 * (i % 4),
                  'hidden_dim': 32 * (1 + i % 3), 'l2': 10.0 ** (-6 + i % 3)}
        candidates.append({'candidate_id': cid, 'joint_fingerprint': cid,
                           'architecture_fingerprint': arch, 'hp_fingerprint': f'hp_{i % 3}',
                           'raw_hp': raw_hp, **raw_hp})
        if arch not in encoding:
            encoding[arch] = generator.normal(size=12).tolist()
        mean = 0.60 + 0.003 * i
        labels[cid] = {'validation_mean': mean, 'validation_se': 0.004 / math.sqrt(3),
                       'validation_seeds': [mean - 0.004, mean, mean + 0.004], 'training_seeds': [0, 1, 2]}
    checkpoint = prepared / 'synthetic_checkpoint_marker.bin'
    checkpoint.write_bytes(b'engineering marker only; not a VAE checkpoint')
    json_write(prepared / 'candidates.json', candidates)
    json_write(prepared / 'base_encoding.json', encoding)
    json_write(prepared / 'oracle_validation.json', labels)
    manifest = {'pool_hash': digest(candidates), 'base_encoding_hash': digest(encoding),
                'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint),
                'oracle_sha256': file_hash(prepared / 'oracle_validation.json')}
    json_write(prepared / 'manifest.json', manifest)
    return config, prepared, labels


def candidate(cid, arch, joint=None):
    raw_hp = {'lr': 0.001, 'dropout': 0.2, 'hidden_dim': 32, 'l2': 1e-5}
    return {'candidate_id': cid, 'joint_fingerprint': joint or cid,
            'architecture_fingerprint': arch, 'raw_hp': raw_hp, **raw_hp}


def test_p2_original_quota_and_fingerprint_tie_rules():
    candidates = [candidate('id_a', 'A', 'z'), candidate('id_b', 'A', 'a'),
                  candidate('id_c', 'C', 'c'), candidate('id_d', 'D', 'd'),
                  candidate('id_e', 'E', 'e')]
    scores = {c['candidate_id']: 1.0 for c in candidates}
    counts = {'C': 2, 'D': 1}
    assert p2_select(candidates, scores, counts, batch_size=3) == ['id_b', 'id_d', 'id_e']
    assert counts == {'C': 2, 'D': 1}  # Selection alone never charges the budget.
    with pytest.raises(RuntimeError, match='cannot fill'):
        p2_select(candidates, scores, counts, batch_size=4)


@pytest.mark.parametrize('label_key', ['validation_mean', 'validation_se', 'test_accuracy', 'val_acc', 'score'])
def test_p2_candidate_whitelist_rejects_label_fields(label_key):
    row = candidate('a', 'A')
    row[label_key] = 0.95
    with pytest.raises(ValueError, match='non-candidate metadata'):
        p2_select([row], {'a': 1.0}, {}, batch_size=1)


def test_initial_records_do_not_charge_architecture_cap(synthetic_pool, tmp_path):
    config, prepared, labels = synthetic_pool
    oracle = SyntheticOracle(labels)
    replay = Replay(config, prepared, tmp_path / 'initial', oracle=oracle)
    assert len(replay.state['known']) == config['initial_size']
    assert replay.state['query_order'] == []
    assert replay.state['architecture_counts'] == {}
    assert oracle.calls == [replay.state['initial_ids']]
    available = [c for c in replay.candidates if c['candidate_id'] not in replay.state['known']]
    selected = p2_select(available, {c['candidate_id']: 1.0 for c in available}, replay.state['architecture_counts'])
    assert len({replay.by_id[cid]['architecture_fingerprint'] for cid in selected}) == 5


def test_hidden_label_pollution_cannot_change_next_predictions_or_selection(synthetic_pool, tmp_path):
    config, prepared, labels = synthetic_pool
    original, polluted = SyntheticOracle(labels), SyntheticOracle(labels)
    first = Replay(config, prepared, tmp_path / 'clean', oracle=original)
    second = Replay(config, prepared, tmp_path / 'polluted', oracle=polluted)
    assert first.state['initial_ids'] == second.state['initial_ids']
    for cid, row in polluted.values.items():
        if cid not in second.state['known']:
            row['validation_mean'] = 1.0 - row['validation_mean']
            row['validation_seeds'] = [1.0 - value for value in row['validation_seeds']]
            row['validation_se'] *= 30
            row['test_accuracy'] = 1.0
    first.refresh()
    second.refresh()
    first.select()
    second.select()
    assert first.state['pending'] == second.state['pending']
    assert torch.equal(first.state['embeddings'], second.state['embeddings'])
    assert len(original.calls) == len(polluted.calls) == 1
    assert first.state['pending']['labels_revealed'] is False
    assert set(first.state['pending']['selected_ids']).isdisjoint(first.state['known'])


def test_resume_before_reveal_and_after_queries_keeps_exact_counts_and_order(synthetic_pool, tmp_path):
    config, prepared, labels = synthetic_pool
    reference_oracle = SyntheticOracle(labels)
    reference = Replay(config, prepared, tmp_path / 'uninterrupted', oracle=reference_oracle)
    reference_summary = reference.run()
    resumed_oracle = SyntheticOracle(labels)
    output = tmp_path / 'resumed'
    paused = Replay(config, prepared, output, oracle=resumed_oracle)
    paused_summary = paused.run(pause_before_reveal_batch=1)
    assert paused_summary['new_history_queries'] == 0
    assert len(paused_summary['pending_selected_ids']) == 5
    assert len(resumed_oracle.calls) == 1
    saved_predictions = (output / 'batch_01_before_reveal.json').read_bytes()
    selected = list(paused_summary['pending_selected_ids'])
    pending_resume = Replay(config, prepared, output, resume=True, oracle=resumed_oracle)
    assert pending_resume.state['pending']['selected_ids'] == selected
    pending_resume.reveal_pending()
    assert len(pending_resume.state['query_order']) == 5
    assert pending_resume.state['refresh_needed'] is True
    assert pending_resume.state['pending'] is None
    # Crash after revealing a saved batch, before fitting the now-expanded GP.
    intermediate = Replay(config, prepared, output, resume=True, oracle=resumed_oracle)
    assert len(intermediate.state['query_order']) == 5
    assert intermediate.state['refresh_needed'] is True
    assert intermediate.gp is None
    partial = intermediate.run(stop_after_queries=10)
    assert partial['new_history_queries'] == 10
    assert partial['logical_records_seen'] == 18
    assert partial['query_order'][:5] == selected
    assert (output / 'batch_01_before_reveal.json').read_bytes() == saved_predictions
    completed = Replay(config, prepared, output, resume=True, oracle=resumed_oracle)
    final = completed.run()
    assert final == reference_summary
    assert final['new_history_queries'] == 20 and final['logical_records_seen'] == 28
    assert final['representation_update_query_counts'] == [0, 10, 20]
    assert final['gp_refit_query_counts'] == [0, 5, 10, 15, 20]
    assert final['representation_updates_after_initial'] == 2
    assert final['actual_new_gnn_trainings'] == 0
    assert len(resumed_oracle.calls) == 5
    assert len({cid for call in resumed_oracle.calls for cid in call}) == 28
    counts = Counter(completed.by_id[cid]['architecture_fingerprint'] for cid in final['query_order'])
    assert max(counts.values()) <= 2
    assert dict(counts) == completed.state['architecture_counts']
    for offset in range(0, 20, 5):
        assert len({completed.by_id[cid]['architecture_fingerprint'] for cid in final['query_order'][offset:offset + 5]}) == 5
    before = len(resumed_oracle.calls)
    assert completed.run() == final
    assert len(resumed_oracle.calls) == before
    for batch in range(1, 5):
        assert json.loads((output / f'batch_{batch:02d}_before_reveal.json').read_text()) == json.loads((reference.output / f'batch_{batch:02d}_before_reveal.json').read_text())


def test_new_and_old_coordinates_and_gp_training_coordinates_are_enforced(synthetic_pool, tmp_path):
    config, prepared, labels = synthetic_pool
    replay = Replay(config, prepared, tmp_path / 'coordinates', oracle=SyntheticOracle(labels))
    replay.refresh()
    original = replay.state['embeddings'].clone()
    replay.state['embeddings'][0, 0] += 1
    with pytest.raises(RuntimeError, match='old/new candidate coordinates'):
        replay.select()
    replay.state['embeddings'] = original
    replay.gp.train_inputs[0][0, 0] += 1
    with pytest.raises(RuntimeError, match='stale coordinates'):
        replay.select()


def test_candidate_identity_change_is_rejected(synthetic_pool, tmp_path):
    config, prepared, labels = synthetic_pool
    replay = Replay(config, prepared, tmp_path / 'identity', oracle=SyntheticOracle(labels))
    replay.run(stop_after_queries=0)
    replay.candidates[0]['raw_hp']['lr'] *= 2
    with pytest.raises(RuntimeError, match='configuration mutated'):
        replay.summary()
    candidates = json.loads((prepared / 'candidates.json').read_text())
    candidates[0]['raw_hp']['lr'] *= 2
    candidates[0]['lr'] *= 2
    json_write(prepared / 'candidates.json', candidates)
    with pytest.raises(RuntimeError, match='identity changed'):
        Replay(config, prepared, replay.output, resume=True, oracle=SyntheticOracle(labels))


def test_real_oracle_schema_rejects_test_fields_only_when_selected(synthetic_pool):
    _, prepared, _ = synthetic_pool
    path = prepared / 'oracle_validation.json'
    records = json.loads(path.read_text())
    records['joint_001']['test_accuracy'] = 0.9
    json_write(path, records)
    oracle = HistoryOracle(path)
    assert set(oracle.reveal(['joint_000'])) == {'joint_000'}
    with pytest.raises(RuntimeError, match='unexpected fields'):
        oracle.reveal(['joint_001'])


@pytest.mark.parametrize('location', ['top_level', 'nested_hp'])
def test_candidate_loading_rejects_labels_before_initial_reveal(synthetic_pool, tmp_path, location):
    config, prepared, labels = synthetic_pool
    candidates = json.loads((prepared / 'candidates.json').read_text())
    target = candidates[0] if location == 'top_level' else candidates[0]['raw_hp']
    target['validation_mean'] = 0.99
    json_write(prepared / 'candidates.json', candidates)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    manifest['pool_hash'] = digest(candidates)
    json_write(prepared / 'manifest.json', manifest)
    oracle = SyntheticOracle(labels)
    with pytest.raises(ValueError, match='candidate'):
        Replay(config, prepared, tmp_path / ('forbidden_' + location), oracle=oracle)
    assert oracle.calls == []


def test_disagreeing_hp_copies_are_rejected():
    row = candidate('a', 'A')
    row['lr'] *= 2
    with pytest.raises(ValueError, match='representations disagree'):
        p2_select([row], {'a': 1.0}, {}, batch_size=1)
