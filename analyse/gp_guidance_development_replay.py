"""Fixed A-only sequential replay. Oracle controller and learners are separate processes."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from analyse.relative_metric_pilot import read, write, save, load, freeze_copy, file_hash, digest, tensor_hash, utc
from surrogate.gp_guidance_replay_engine import (Ledger, journal, append, rng_state, restore_rng, reload_model,
    scaler_from_initial, input_tensors, fit_mapping, fit_gp, score_candidates, pause_once)
from surrogate.fuzzy_metric import JointMetricModel
from analyse.fuzzy_metric_offline_core import _label_arrays
from analyse.fuzzy_metric_replay import validate_candidates
from analyse.joint_configuration_metric_gate import hp_feature_vector

DEFAULT = ROOT / 'results/posthoc/gp-guidance-development-replay-20260916-v1'
OLD = ROOT / 'results/posthoc/fuzzy-metric-stage2-20260910-v1'
PREP = ROOT / 'results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2'
CONFIG_PATH = ROOT / 'configs/gp_guidance_development_replay_v1.json'
METHODS = ['C_GP', 'S_GP', 'C_HEAD', 'RANDOM']
SEEDS = list(range(2026091600, 2026091605))
SOURCES = ['analyse/gp_guidance_development_replay.py', 'surrogate/gp_guidance_replay_engine.py',
    'tests/test_gp_guidance_development_replay.py', 'configs/gp_guidance_development_replay_v1.json',
    'surrogate/gp_mean_guidance_sync.py', 'surrogate/gp_mean_guidance.py', 'surrogate/fuzzy_metric.py',
    'surrogate/relative_metric.py', 'analyse/fuzzy_metric_replay.py', 'analyse/fuzzy_metric_offline_core.py',
    'analyse/relative_metric_pilot.py', 'analyse/joint_configuration_metric_gate.py', 'architecture_diagnostics.py']


def id_order(ids, seed, kind):
    prefix = 'gp_guidance_replay_v1:' + kind + ':' + str(seed) + ':'
    return sorted(ids, key=lambda c: (hashlib.sha256((prefix + c).encode('utf-8')).hexdigest(), c))


def verify_protocol(o):
    p = read(o / 'protocol.json')
    ident = p.pop('identity')
    assert digest(p) == ident
    p['identity'] = ident
    assert p['config'] == read(CONFIG_PATH)
    for rel, h in p['source_hashes'].items():
        assert file_hash(ROOT / rel) == h, rel
    assert file_hash(o / 'features.json') == p['features_sha']
    assert digest(read(o / 'starts.json')) == p['starts_hash']
    return p


def prepare(o):
    if (o / 'protocol.json').exists():
        p = verify_protocol(o)
        provision_initial(o, p)
        return p
    old, part, manifest = read(OLD / 'protocol.json'), read(OLD / 'partition.json'), read(PREP / 'manifest.json')
    pool, base = read(PREP / 'candidates.json'), read(PREP / 'base_encoding.json')
    assert digest(part) == old['partition_hash']
    assert manifest == old['prepared_manifest']
    assert digest(pool) == manifest['pool_hash'] and digest(base) == manifest['base_encoding_hash']
    assert file_hash(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']
    assert manifest['vae_performance_supervision'] is False and manifest['vae_requires_grad_count'] == 0
    assert manifest['vae_tensor_hash_before'] == manifest['vae_tensor_hash_after']
    ids = sorted(part['A_ids'])
    candidates = sorted([c for c in pool if c['candidate_id'] in set(ids)], key=lambda c: c['candidate_id'])
    validate_candidates(candidates)
    from analyse.joint_configuration_metric_gate import architecture_fingerprint, hp_fingerprint, stable_hash
    for c in candidates:
        assert architecture_fingerprint(c) == c['architecture_fingerprint']
        assert hp_fingerprint(c['raw_hp_vector']) == c['hp_fingerprint']
        assert stable_hash({k:c[k] for k in ['architecture_fingerprint', 'hp_fingerprint']}) == c['candidate_id'] == c['joint_fingerprint']
    assert len(candidates) == 179 and len({c['architecture_fingerprint'] for c in candidates}) == 96
    assert len({c['training_protocol_identity'] for c in candidates}) == 1
    starts = {str(seed): dict(initial_ids=id_order(ids, seed, 'init')[:50]) for seed in SEEDS}
    for seed in SEEDS:
        s = starts[str(seed)]
        s['random_order'] = id_order([c for c in ids if c not in s['initial_ids']], seed, 'random')
    features = dict(ids=ids, architectures=[c['architecture_fingerprint'] for c in candidates],
        z=[base[c['architecture_fingerprint']] for c in candidates], hp=[hp_feature_vector(c) for c in candidates])
    write(o / 'features.json', features)
    write(o / 'candidate_identity.json', candidates)
    write(o / 'starts.json', starts)
    matrix = [dict(seed=seed, method=method, initial_size=50, query_budget=50, batches=10) for seed in SEEDS for method in METHODS]
    write(o / 'task_matrix.json', matrix)
    paths = [OLD / n for n in ['protocol.json', 'partition.json', 'label_boundary.json', 'A_validation.json']]
    paths += [PREP / n for n in ['manifest.json', 'candidates.json', 'base_encoding.json', 'provenance.json']]
    paths += [Path(manifest['checkpoint']), ROOT / 'docs/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md']
    sources = {r: file_hash(ROOT / r) for r in SOURCES}
    for rel in SOURCES:
        freeze_copy(ROOT / rel, o / 'source_snapshot' / rel)
    p = dict(config=read(CONFIG_PATH), A_ids=ids, features_sha=file_hash(o / 'features.json'),
        candidate_sha=file_hash(o / 'candidate_identity.json'), starts_hash=digest(starts), matrix_hash=digest(matrix),
        source_hashes=sources, input_sources={str(p): file_hash(p) for p in paths},
        VAE_manifest=manifest, prompt_sha=file_hash(o / 'input_originals/Prompt.md'), sealed_at=utc(),
        labels_provided_to_learners_before_seal=0, scope='A development replay, not old stage3/B or independent confirmation')
    p['identity'] = digest(p)
    write(o / 'protocol.json', p)
    # Only after all starts/settings/source identities are sealed, open A labels.
    provision_initial(o, p)
    write(o / 'data_checks.json', dict(A_candidates=179, A_architectures=96, B_labels=0, test_labels=0,
        initial_label_files_after_seal=True, three_seed_statistics_verified=179,
        recipe_id=candidates[0]['training_protocol_identity'], VAE_checkpoint_unchanged=True))
    return p


def provision_initial(o, p):
    labels = oracle_labels(p)
    starts = read(o / 'starts.json')
    for seed in SEEDS:
        initial = starts[str(seed)]['initial_ids']
        row = {c: labels[c] for c in initial}
        _label_arrays(sorted(initial), row)
        write(o / 'initial_labels' / f'{seed}.json', row)


def oracle_labels(protocol):
    path = OLD / 'A_validation.json'
    assert file_hash(path) == protocol['input_sources'][str(path)]
    labels = read(path)
    assert set(labels) == set(protocol['A_ids'])
    assert digest(labels) == read(OLD / 'label_boundary.json')['exported_hash']
    _label_arrays(sorted(labels), labels)
    return labels


def guard_worker(o, folder, seed, method):
    exact = {str((o / n).resolve()) for n in ['protocol.json', 'features.json', 'starts.json']}
    exact.add(str((o / 'initial_labels' / f'{seed}.json').resolve()))
    if method != 'COMMON':
        exact.add(str((o / 'shared' / str(seed) / 'M100.pt').resolve()))
    if method == 'C_HEAD':
        exact.add(str((o / 'shared' / str(seed) / 'C_init.pt').resolve()))
    prefixes = [str(folder.resolve()), str((o / 'ledger').resolve())]
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = str(Path(os.fsdecode(args[0])).resolve())
        if p.startswith(str(ROOT / 'results') + '/') or p.startswith(str(ROOT / 'checkpoints') + '/') or p.startswith(str(ROOT / 'data') + '/'):
            if p not in exact and not any(p == x or p.startswith(x + '/') for x in prefixes):
                raise PermissionError('learner may not read another history, oracle, or old model: ' + p)
    sys.addaudithook(hook)
    return hook


def worker(o, seed, method):
    p = verify_protocol(o)
    folder = o / ('shared' if method == 'COMMON' else 'trajectories') / (str(seed) if method == 'COMMON' else f'{seed}-{method}')
    folder.mkdir(parents=True, exist_ok=True)
    guard_worker(o, folder, seed, method)
    features = read(o / 'features.json')
    assert set(features) == {'ids', 'architectures', 'z', 'hp'}
    initial = read(o / 'initial_labels' / f'{seed}.json')
    start = read(o / 'starts.json')[str(seed)]
    assert set(initial) == set(start['initial_ids'])
    ledger = Ledger(o / 'ledger', p['identity'], p['config']['limits'])
    if method == 'COMMON':
        if (folder / 'M100.pt').exists():
            return
        scaler = scaler_from_initial(features, start['initial_ids'], initial)
        data = input_tensors(features, scaler, initial)
        model = JointMetricModel(12, seed=0)
        state = fit_mapping(folder / 'mapping', model, data, 100, False, ledger,
            dict(task=f'{seed}-COMMON', version=0), recovery_test=seed == SEEDS[0])
        save(folder / 'M100.pt', dict(model=state['model'], scaler=scaler, initial_ids=start['initial_ids'],
            source='fresh seed0 network on initial50 only', steps=100, protocol=p['identity']))
        return
    if (folder / 'completed.json').exists():
        done = read(folder / 'completed.json')
        assert done['protocol'] == p['identity'] and done['queries'] == 50
        return
    if (folder / 'state.pt').exists():
        state = load(folder / 'state.pt')
        assert state['protocol'] == p['identity']
        restore_rng(state['rng'])
    else:
        warm = load(o / 'shared' / str(seed) / ('C_init.pt' if method == 'C_HEAD' else 'M100.pt'))
        assert warm['initial_ids'] == start['initial_ids']
        model = None if method == 'RANDOM' else reload_model(warm['model'])
        scaler = warm['scaler']
        data = input_tensors(features, scaler, initial)
        if method in ['C_GP', 'S_GP']:
            fit_mapping(folder / 'maps' / 'v1', model, data, 200, method == 'S_GP', ledger,
                dict(task=f'{seed}-{method}/initial', version=1), recovery_test=seed == SEEDS[0] and method == 'S_GP')
        initial_state = dict(model={} if model is None else copy.deepcopy(model.state_dict()), scaler=scaler, initial_ids=start['initial_ids'],
            warm_model_hash=None if method == 'RANDOM' else tensor_hash(warm['model']), protocol=p['identity'])
        save(folder / 'initial.pt', initial_state)
        state = dict(protocol=p['identity'], method=method, seed=seed, known=initial, query_order=[], version=1,
            model=initial_state['model'], scaler=scaler, rng=rng_state(), initial_ids=start['initial_ids'])
        save(folder / 'state.pt', state)
    # A crash after the atomic state update may precede its derived aggregation receipt.
    for old_batch in range(len(state['query_order']) // 5):
        old_dir = folder / 'batches' / f'b{old_batch:02}'
        if not (old_dir / 'aggregated.json').exists():
            old_selection = read(old_dir / 'selection.json')
            assert old_selection['selected_ids'] == state['query_order'][old_batch*5:old_batch*5+5]
            ids = sorted(state['initial_ids'] + state['query_order'][:(old_batch+1)*5])
            write(old_dir / 'aggregated.json', dict(query_count=(old_batch+1)*5, known_ids=ids,
                selection_sha=file_hash(old_dir / 'selection.json'), receipt_sha=file_hash(old_dir / 'reveal.json')))
    while len(state['query_order']) < 50:
        n = len(state['query_order'])
        assert n % 5 == 0 and len(state['known']) == 50 + n
        batch = n // 5
        bdir = folder / 'batches' / f'b{batch:02}'
        if not (bdir / 'selection.json').exists():
            data = input_tensors(features, state['scaler'], state['known'])
            if method == 'RANDOM':
                t = time.monotonic()
                remaining = [c for c in start['random_order'] if c not in state['known']]
                rows = [dict(candidate_id=c, score=None, head=None, lower=None, upper=None) for c in remaining]
                timing = time.monotonic() - t
                embedding_hash = None
            else:
                model = reload_model(state['model'])
                expected_version = n // 10 + 1
                if n in [10, 20, 30, 40] and state['version'] < expected_version:
                    fit_mapping(folder / 'maps' / f'v{expected_version}', model, data, 200, method == 'S_GP', ledger,
                        dict(task=f'{seed}-{method}/update{n}', version=expected_version))
                    state.update(version=expected_version, model=copy.deepcopy(model.state_dict()), rng=rng_state())
                    save(folder / 'state.pt', state)
                # Both known references and every query are gathered from this same forward.
                with torch.no_grad():
                    all_e, _ = model(data['all_z'], data['all_hp'])
                teacher = None
                if method in ['C_GP', 'S_GP']:
                    teacher = fit_gp(folder / 'selection_gps' / f'b{batch:02}', all_e[data['indices']].detach().double(),
                        data['target64'], data['se'], state['scaler']['y_scale'],
                        dict(task=f'{seed}-{method}/select{n}', train_ids=data['ids'], version=state['version'],
                             model_hash=tensor_hash(model.state_dict())), ledger,
                        recovery_test=seed == SEEDS[0] and method == 'C_GP' and batch == 0)
                rows, cache, timing = score_candidates(model, data, teacher)
                cache['version'] = state['version']
                save(bdir / 'coordinates.pt', cache)
                embedding_hash = cache['embedding_hash']
            selected = [r['candidate_id'] for r in rows[:5]]
            selection = dict(seed=seed, method=method, batch=batch, known_ids=sorted(state['known']),
                known_labels_hash=digest(state['known']), query_count_before=n, version=state['version'],
                model_hash=tensor_hash(state['model']) if method != 'RANDOM' else None,
                scaler_hash=digest(state['scaler']), embedding_hash=embedding_hash, rows=rows, selected_ids=selected,
                selection_seconds=timing, selected_at=utc(), protocol=p['identity'])
            write(bdir / 'selection.json', selection)
        selection = read(bdir / 'selection.json')
        assert selection['known_ids'] == sorted(state['known']) and selection['known_labels_hash'] == digest(state['known'])
        pause_once(bdir, 'selected', seed == SEEDS[0] and method == 'C_GP' and batch == 0,
            dict(kind='selected_not_revealed', selection_sha=file_hash(bdir / 'selection.json'), query_count=n))
        receipt_path = bdir / 'reveal.json'
        if not receipt_path.exists():
            print(json.dumps(dict(event='reveal', seed=seed, method=method, batch=batch,
                selection_sha=file_hash(bdir / 'selection.json'))), flush=True)
            response = json.loads(sys.stdin.readline())
            assert response == read(receipt_path)
        receipt = read(receipt_path)
        assert receipt['selection_sha'] == file_hash(bdir / 'selection.json')
        assert set(receipt['labels']) == set(selection['selected_ids'])
        assert not set(receipt['labels']) & set(state['known'])
        _label_arrays(sorted(receipt['labels']), receipt['labels'])
        pause_once(bdir, 'revealed', seed == SEEDS[0] and method == 'C_GP' and batch == 0,
            dict(kind='revealed_not_aggregated', query_count=n, receipt_sha=file_hash(receipt_path)))
        state['known'].update(receipt['labels'])
        state['query_order'].extend(selection['selected_ids'])
        state['rng'] = rng_state()
        save(folder / 'state.pt', state)
        write(bdir / 'aggregated.json', dict(query_count=len(state['query_order']), known_ids=sorted(state['known']),
            selection_sha=file_hash(bdir / 'selection.json'), receipt_sha=file_hash(receipt_path)))
    assert len(state['query_order']) == len(set(state['query_order'])) == 50 and len(state['known']) == 100
    write(folder / 'completed.json', dict(protocol=p['identity'], seed=seed, method=method, queries=50,
        query_order=state['query_order'], initial_ids=state['initial_ids'], final_version=state['version'], completed_at=utc(),
        state_sha=file_hash(folder / 'state.pt')))


def run_worker(o, seed, method, labels, p):
    attempts = 0
    while True:
        attempts += 1
        start = time.monotonic()
        errpath = o / 'logs' / f'{seed}-{method}-{time.time_ns()}.stderr'
        errpath.parent.mkdir(exist_ok=True)
        with errpath.open('w') as err:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', method,
                '--seed', str(seed), '--output', str(o)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=err, text=True, bufsize=1, env={**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
                    'OPENBLAS_NUM_THREADS':'1', 'CUDA_VISIBLE_DEVICES':''})
            for line in proc.stdout:
                request = json.loads(line)
                assert request['event'] == 'reveal' and request['seed'] == seed and request['method'] == method
                bdir = o / 'trajectories' / f'{seed}-{method}' / 'batches' / f'b{request["batch"]:02}'
                selection = read(bdir / 'selection.json')
                assert file_hash(bdir / 'selection.json') == request['selection_sha']
                assert len(selection['selected_ids']) == 5 and len(set(selection['selected_ids'])) == 5
                assert not set(selection['selected_ids']) & set(selection['known_ids'])
                assert set(selection['selected_ids']) <= set(p['A_ids'])
                path = bdir / 'reveal.json'
                if not path.exists():
                    before = time.monotonic()
                    receipt = dict(selection_sha=request['selection_sha'], selected_ids=selection['selected_ids'],
                        labels={c: labels[c] for c in selection['selected_ids']}, logical_queries=5,
                        protocol=p['identity'], revealed_at=utc())
                    write(path, receipt)
                    write(bdir / 'oracle_timing.json', dict(seconds=time.monotonic()-before))
                proc.stdin.write(json.dumps(read(path)) + '\n')
                proc.stdin.flush()
            code = proc.wait()
        append(o / 'invocations.jsonl', dict(seed=seed, method=method, returncode=code,
            seconds=time.monotonic()-start, time=utc(), stderr=str(errpath.relative_to(o))))
        if code == 75:
            assert attempts <= 8
            continue
        if code:
            raise RuntimeError(f'{method} failed; retained state and budgets: {errpath}')
        break


def counts(o):
    return {k: len(journal(o / 'ledger' / (k + '-intents.jsonl'))) for k in ['mapping', 'gp', 'gp_fits']} | {
        'queries': 5 * len(list((o / 'trajectories').glob('*/batches/*/reveal.json')))}


def run(o):
    p = verify_protocol(o)
    assert read(o / 'checks_complete.json')['passed']
    labels = oracle_labels(p)
    with (o / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = counts(o)
        started = time.monotonic()
        for seed in SEEDS:
            run_worker(o, seed, 'COMMON', labels, p)
            for method in METHODS:
                run_worker(o, seed, method, labels, p)
                if method == 'C_GP':
                    src = o / 'trajectories' / f'{seed}-C_GP' / 'initial.pt'
                    freeze_copy(src, o / 'shared' / str(seed) / 'C_init.pt')
                print(json.dumps(dict(completed=f'{seed}-{method}', cost=counts(o))), flush=True)
        after = counts(o)
        assert after['queries'] == 1000
        assert len(list((o / 'trajectories').glob('*/completed.json'))) == 20
        append(o / 'runs.jsonl', dict(before=before, after=after,
            new={k: after[k]-before[k] for k in before}, seconds=time.monotonic()-started, time=utc()))
        if not (o / 'all_completed.json').exists():
            write(o / 'all_completed.json', dict(protocol=p['identity'], completed=20, cost=after, time=utc(),
                artifacts={str(f.relative_to(o)):file_hash(f) for f in (o / 'trajectories').rglob('*') if f.is_file()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--worker', choices=METHODS + ['COMMON'])
    parser.add_argument('--seed', type=int, choices=SEEDS)
    a = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    o = a.output.resolve()
    if a.prepare:
        prepare(o)
    elif a.worker:
        worker(o, a.seed, a.worker)
    else:
        try:
            run(o)
        except Exception:
            write(o / 'failures' / f'{time.time_ns()}.json', dict(error=traceback.format_exc(), time=utc(), cost=counts(o)))
            raise


if __name__ == '__main__':
    main()
