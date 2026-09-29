"""Single serial continuation runner; receives selected snapshots, never scoring labels."""
import argparse
import datetime
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from stage01_core import cpu_setup, tensor_hash
from stage2_core import Budget, BudgetExceeded
from length_core import TailSession
from length_diagnostics import diagnose_isolated, window, stability_at
from prepare_length import sha, read, digest


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temp.replace(path)


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def verified_freeze(root):
    for manifest in ['frozen_manifest.json', 'code_manifest.json', 'source_manifest.json']:
        for name, expected in read(root / manifest)['files'].items():
            if sha(root / name) != expected:
                raise ValueError('Frozen identity mismatch: ' + name)


def load_diagnostic(root, path):
    row = read(root / path)
    if sha(root / row['kernel_path']) != row['kernel_sha256']:
        raise ValueError('Stored kernel identity differs')
    with np.load(root / row['kernel_path'], allow_pickle=False) as kernels:
        for key in ['training_kernel', 'combined_kernel']:
            row[key] = kernels[key].tolist()
    return row


class Runner:
    def __init__(self, root, resume=False):
        self.root, self.resume = root, resume
        verified_freeze(root)
        if (root / 'predictions_manifest.json').exists():
            raise ValueError('Already sealed; no continuation after scoring exposure')
        self.source = read(root / 'source_manifest.json')
        self.contexts = self.source['contexts']
        self.budget = Budget(root / 'resource_ledger.json', limit=1200)
        if not resume:
            if self.budget.updates or self.budget.verification_updates:
                raise ValueError('Existing work requires explicit resume')
            check = read(root / 'verification/preflight_summary.json')
            self.budget.charge(check['verification_seconds'], kind='verification', updates=check['synthetic_optimizer_updates'])
            self.budget.persist()
        self.started = time.monotonic()
        self.records, self.costs = [], []
        self.endpoint_files = [{'unit_id': c['context_id'] + '_j1000', 'context_id': c['context_id'],
             'total_joint_step': 1000, 'path': c['prediction_path'], 'sha256': c['prediction_sha256'],
             'state_path': c['checkpoint_path'], 'state_file_sha256': c['checkpoint_sha256'],
             'endpoint_state_hash': c['model_state_hash'], 'baseline_is_original_prediction': True}
             for c in self.contexts]
        self.forecast = {}
        self.session, self.active_ctx = None, None
        self.prior_wall = read(root / 'progress.json').get('experiment_wall_seconds', 0) if resume and (root / 'progress.json').exists() else 0
        if resume:
            saved_updates = 0
            for c in self.contexts:
                done = root / f'trajectories/{c["context_id"]}.json'
                latest = root / f'latest/{c["context_id"]}.pt'
                if done.exists():
                    saved_updates += 3000
                elif latest.exists():
                    state = torch.load(latest, map_location='cpu', weights_only=False)
                    saved_updates += state['continuation']['additional_completed_steps']
            if saved_updates != self.budget.updates:
                raise ValueError('Resume ledger/checkpoint update mismatch; preserve attempt and reconcile before new updates')

    def progress(self, ctx=None):
        write(self.root / 'progress.json', {'updated_at_utc': now(), 'active_context': None if ctx is None else ctx['context_id'],
              'total_joint_step': None if self.session is None else self.session.total_joint_step,
              'new_formal_updates': self.budget.updates, 'completed_trajectories': len(self.costs),
              'budget_used_seconds': self.budget.seconds, 'experiment_wall_seconds': self.prior_wall + time.monotonic() - self.started})

    def snapshot(self, session, path):
        tick = time.monotonic()
        try:
            state = session.snapshot()
        finally:
            self.budget.charge(time.monotonic() - tick)
        save_state(path, state)
        self.budget.persist()
        return state

    def record_diagnostic(self, ctx, session, query_x, query, train, previous):
        self.budget.check()
        tick = time.monotonic()
        try:
            row = diagnose_isolated(session.model, query_x, query['ids'], train['scaler']['y_mean'],
                    train['scaler']['y_scale'], session.total_joint_step, last_gradient=session.last_gradient,
                    lrs=[g['lr'] for g in session.optimizer.param_groups])
            row.update(context_id=ctx['context_id'], subset_id=ctx['subset_id'], fold=ctx['fold'],
                       sample_seed=ctx['sample_seed'], model_seed=ctx['model_seed'], n=ctx['n'])
            row['window_from_previous'] = None if previous is None else window(previous, row)
        finally:
            self.budget.charge(time.monotonic() - tick, kind='prediction')
        kernel_path = f'diagnostics/{ctx["context_id"]}/kernels_j{session.total_joint_step}.npz'
        path = self.root / kernel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, training_kernel=np.asarray(row['training_kernel'], dtype=np.float64),
                            combined_kernel=np.asarray(row['combined_kernel'], dtype=np.float64))
        row.update(kernel_path=kernel_path, kernel_sha256=sha(path), resource_totals=self.budget.as_dict())
        output = {k: v for k, v in row.items() if k not in ['training_kernel', 'combined_kernel']}
        jsonpath = f'diagnostics/{ctx["context_id"]}/j{session.total_joint_step}.json'
        write(self.root / jsonpath, output)
        self.records.append({'context_id': ctx['context_id'], 'total_joint_step': session.total_joint_step,
                             'path': jsonpath, 'sha256': sha(self.root / jsonpath),
                             'kernel_path': kernel_path, 'kernel_sha256': row['kernel_sha256']})
        self.budget.persist()
        return row

    def baseline(self, ctx, session, row, train):
        tick = time.monotonic()
        try:
            pred = read(self.root / ctx['prediction_path'])
            if session.model.fingerprint() != ctx['model_state_hash'] or row['state_hash'] != pred['endpoint_state_hash']:
                raise ValueError('Restored baseline mathematical state differs')
            for key, field in [('x', 'x_standardized'), ('y', 'y_standardized'), ('known', 'known_variance_standardized')]:
                if not torch.equal(getattr(session.model, key), torch.tensor(train[field], dtype=torch.double)):
                    raise ValueError('Restored baseline training tensor differs: ' + key)
            errors = {}
            for actual, expected in [('query_mean_standardized', 'mean_standardized'),
                                     ('query_latent_variance_standardized', 'latent_variance_standardized')]:
                x, y = np.asarray(row[actual]), np.asarray(pred[expected])
                if not np.allclose(x, y, atol=1e-8, rtol=1e-6):
                    raise ValueError('Restored baseline prediction differs: ' + actual)
                errors[actual] = float(np.abs(x - y).max())
            check = {'pass': True, 'context_id': ctx['context_id'], 'new_updates': 0,
                     'source_checkpoint_sha256': ctx['checkpoint_sha256'], 'model_state_hash': row['state_hash'],
                     'scaler_sha256': digest(train['scaler']), 'prediction_max_absolute_errors': errors,
                     'atol': 1e-8, 'rtol': 1e-6, 'optimizer': session.optimizer_audit()}
        finally:
            self.budget.charge(time.monotonic() - tick, kind='verification')
        write(self.root / f'verification/baseline/{ctx["context_id"]}.json', check)

    def endpoint(self, ctx, session, row, train, query, cost_start):
        step = session.total_joint_step
        state_rel = f'checkpoints/{ctx["context_id"]}_j{step}.pt'
        self.snapshot(session, self.root / state_rel)
        current = self.budget.as_dict()
        training = current['training_seconds'] - cost_start['training_seconds']
        prediction = current['prediction_seconds'] - cost_start['prediction_seconds']
        verification = current['verification_seconds'] - cost_start['verification_seconds']
        original = read(self.root / ctx['prediction_path'])
        pred = {**original, 'unit_id': f'{ctx["context_id"]}_j{step}', 'total_joint_step': step,
                'mean_standardized': row['query_mean_standardized'], 'mean_raw': row['query_mean_raw'],
                'latent_variance_standardized': row['query_latent_variance_standardized'],
                'latent_variance_raw': row['query_latent_variance_raw'],
                'extra_noise_variance_standardized': row['extra_noise_variance_standardized'],
                'endpoint_state_hash': row['state_hash'], 'state_path': state_rel,
                'state_file_sha256': sha(self.root / state_rel), 'finished_at_utc': now(),
                'negative_roundoff_variance_count': sum(v < 0 for v in row['query_latent_variance_standardized']),
                'cost': {'additional_training_seconds': training, 'diagnostic_prediction_seconds': prediction,
                         'verification_seconds': verification, 'new_fit_predict_seconds': training + prediction,
                         'new_budget_seconds': training + prediction + verification,
                         'inherited_stage2_fit_predict_seconds': original['cost']['standalone_fit_predict_seconds']}}
        rel = f'predictions/{pred["unit_id"]}.json'
        write(self.root / rel, pred)
        self.endpoint_files.append({'unit_id': pred['unit_id'], 'context_id': ctx['context_id'], 'total_joint_step': step,
             'path': rel, 'sha256': sha(self.root / rel), 'state_path': state_rel,
             'state_file_sha256': pred['state_file_sha256'], 'endpoint_state_hash': pred['endpoint_state_hash']})

    def trajectory(self, ctx):
        self.active_ctx = ctx
        self.session = None
        root, cid = self.root, ctx['context_id']
        done = root / f'trajectories/{cid}.json'
        latest = root / f'latest/{cid}.pt'
        if done.exists():
            if not self.resume:
                raise FileExistsError(done)
            saved = read(done)
            self.costs.append(saved)
            self.endpoint_files.extend(saved['endpoint_files'])
            self.records.extend(saved['diagnostic_files'])
            return
        self.budget.check()
        cost_path = root / f'latest/{cid}_cost_start.json'
        cost_start = read(cost_path) if latest.exists() else self.budget.as_dict()
        write(cost_path, cost_start)
        train, query = read(root / ctx['training_path']), read(root / ctx['query_path'])
        x = torch.tensor(query['x_raw'], dtype=torch.double)
        x = (x - torch.tensor(train['scaler']['x_mean'])) / torch.tensor(train['scaler']['x_scale'])
        tick = time.monotonic()
        try:
            if latest.exists():
                if not self.resume:
                    raise FileExistsError(latest)
                session = TailSession.from_continuation(torch.load(latest, map_location='cpu', weights_only=False))
            else:
                session = TailSession.from_source(torch.load(root / ctx['checkpoint_path'], map_location='cpu', weights_only=False))
        finally:
            self.budget.charge(time.monotonic() - tick, kind='verification')
        self.session = session
        start_records, start_endpoints = len(self.records), len(self.endpoint_files)
        records = []
        if session.total_joint_step > 1000:
            for point in range(1000, session.total_joint_step + 1, 100):
                rel = f'diagnostics/{cid}/j{point}.json'
                if not (root / rel).exists():
                    continue
                row = load_diagnostic(root, rel)
                records.append(row)
                self.records.append({'context_id': cid, 'total_joint_step': point, 'path': rel,
                    'sha256': sha(root / rel), 'kernel_path': row['kernel_path'], 'kernel_sha256': row['kernel_sha256']})
            for point in [2000, 4000]:
                rel = f'predictions/{cid}_j{point}.json'
                if (root / rel).exists():
                    pred = read(root / rel)
                    self.endpoint_files.append({'unit_id': pred['unit_id'], 'context_id': cid, 'total_joint_step': point,
                         'path': rel, 'sha256': sha(root / rel), 'state_path': pred['state_path'],
                         'state_file_sha256': pred['state_file_sha256'], 'endpoint_state_hash': pred['endpoint_state_hash']})
        else:
            row = self.record_diagnostic(ctx, session, x, query, train, None)
            records.append(row)
            self.baseline(ctx, session, row, train)
            self.snapshot(session, latest)
            self.budget.check()
        while session.total_joint_step < 4000:
            session.step(self.budget)
            if session.total_joint_step % 100 == 0:
                row = self.record_diagnostic(ctx, session, x, query, train, records[-1])
                records.append(row)
                if session.total_joint_step in [2000, 4000]:
                    self.endpoint(ctx, session, row, train, query, cost_start)
                    write(root / f'stability/{cid}_j{session.total_joint_step}.json',
                          {'context_id': cid, 'n': ctx['n'], **stability_at(records, session.total_joint_step)})
                self.snapshot(session, latest)
                self.progress(ctx)
                self.budget.check()
        cost = {k: self.budget.as_dict()[k] - cost_start[k]
                for k in ['training_seconds', 'prediction_seconds', 'verification_seconds', 'updates']}
        summary = {'context_id': cid, 'n': ctx['n'], 'complete': True, 'cost': cost,
                   'total_joint_step': 4000, 'source_checkpoint_sha256': ctx['checkpoint_sha256'],
                   'final_optimizer': session.optimizer_audit(), 'endpoint_files': self.endpoint_files[start_endpoints:],
                   'diagnostic_files': self.records[start_records:]}
        write(done, summary)
        self.costs.append(summary)
        if ctx['n'] not in self.forecast:
            self.forecast[ctx['n']] = sum(cost[k] for k in ['training_seconds', 'prediction_seconds', 'verification_seconds'])
            write(root / 'forecast.json', {'first_formal_context_seconds_by_n': self.forecast,
                  'estimated_remaining_seconds': sum(self.forecast.values()) / len(self.forecast) * (60 - len(self.costs)),
                  'retained_in_formal_matrix': True, 'not_an_extra_pilot': True})
        print(json.dumps({'context_complete': cid, 'completed_trajectories': len(self.costs),
                          'new_updates': self.budget.updates, 'budget_seconds': self.budget.seconds}), flush=True)

    def run(self):
        status, failure = 'COMPLETE_VALID', None
        try:
            for ctx in self.contexts:
                self.trajectory(ctx)
            self.budget.check()
            if self.budget.updates != 180000 or len(self.endpoint_files) != 180 or len(self.records) != 1860:
                raise ValueError('Complete matrix accounting mismatch')
        except Exception as error:
            status = 'PARTIAL' if isinstance(error, BudgetExceeded) else 'INVALID'
            failure = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(),
                       'context_id': None if self.active_ctx is None else self.active_ctx['context_id'], 'at_utc': now()}
            write(self.root / 'failure.json', failure)
            if self.session is not None and self.active_ctx is not None:
                self.snapshot(self.session, self.root / f'latest/{self.active_ctx["context_id"]}.pt')
        self.budget.persist()
        self.progress(self.active_ctx)
        write(self.root / 'diagnostics_manifest.json', {'files': self.records, 'record_count': len(self.records),
              'kernel_values_saved_losslessly_float64': True, 'queries_contain_no_y_or_SE': True})
        manifest = {'sealed': True, 'sealed_at_utc': now(), 'execution_status': status, 'files': self.endpoint_files,
          'expected_endpoint_count': 180, 'new_endpoint_count_expected': 120,
          'new_endpoint_count': sum(p['total_joint_step'] > 1000 for p in self.endpoint_files),
          'new_formal_updates_expected': 180000, 'new_formal_updates': self.budget.updates,
          'completed_trajectories': len(self.costs), 'protocol_sha256': sha(self.root / 'protocol.json'),
          'code_manifest_sha256': sha(self.root / 'code_manifest.json'), 'source_manifest_sha256': sha(self.root / 'source_manifest.json'),
          'diagnostics_manifest_sha256': sha(self.root / 'diagnostics_manifest.json'),
          'resources': {**self.budget.as_dict(), 'experiment_wall_seconds': self.prior_wall + time.monotonic() - self.started},
          'failure': failure, 'next_stage_authorized': False}
        write(self.root / 'predictions_manifest.json', manifest)
        print(json.dumps({k: manifest[k] for k in ['execution_status', 'completed_trajectories', 'new_endpoint_count',
                          'new_formal_updates', 'resources', 'failure']}), flush=True)
        if status == 'INVALID':
            raise RuntimeError('Invalid run preserved; do not score')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    torch.set_num_interop_threads(1)
    cpu_setup(0)
    Runner(a.root.resolve(), a.resume).run()
