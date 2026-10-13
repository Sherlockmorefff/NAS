import argparse
import fcntl
import shlex
import subprocess
import sys
import time
from .common import *

def select_device(root):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    values = [tuple(map(int, x.split(','))) for x in raw.strip().splitlines()]
    if (root / 'resource_lock.json').exists():
        device = read(root / 'resource_lock.json')['device']
        idx = int(device.split(':')[1])
        return device if any(i == idx and free >= 2048 for i, free, util in values) else None
    eligible = [(free, -util, i) for i, free, util in values if free >= 2048]
    if not eligible:
        return None
    _, _, idx = max(eligible)
    device = 'cuda:' + str(idx)
    write(root / 'resource_lock.json', {'device': device, 'time': now(), 'single_worker': True, 'inventory': raw})
    return device

def task_status(root, task):
    folder = root / 'training' / task['task_id']
    if (folder / 'result.json').exists() and (folder / 'binding.json').exists():
        result = read(folder / 'result.json')
        binding = read(folder / 'binding.json')
        assert result['identity'] == binding['identity'] == task_identity(root, task)
        assert sha(folder / 'result.json') == binding['result_sha256']
        verify(folder, result['files'])
        return 'COMPLETE'
    if (folder / 'failure.json').exists():
        return 'FAILED'
    return 'PENDING'

def core_artifacts(root, tasks):
    files = {}
    for t in tasks:
        folder = root / 'training' / t['task_id']
        if task_status(root, t) == 'COMPLETE':
            for name in ['result.json', 'binding.json', 'best_model.pt', 'latest.pt', 'selection.npz', 'epoch_events.jsonl']:
                p = folder / name
                files[str(p.relative_to(root))] = sha(p)
    return files

def seal_d1(root, tasks):
    path = root / 'D1_checkpoint_seal.json'
    if path.exists():
        verify(root, read(path)['files'])
        return
    d1 = [t for t in tasks if t['phase'] == 'D1']
    assert all(task_status(root, t) in ['COMPLETE', 'FAILED'] for t in d1)
    success = [t for t in d1 if task_status(root, t) == 'COMPLETE']
    write(path, {'successful_tasks': success, 'failed_tasks': [t for t in d1 if t not in success],
        'files': core_artifacts(root, d1), 'training_terminal': True, 'test_evaluations_at_seal': 0,
        'frozen_protocol_sha256': sha(root / 'diagnostic_protocol.json'), 'time': now()})

def update_progress(root, tasks, current=None):
    states = {t['task_id']: task_status(root, t) for t in tasks}
    completed = [t for t in tasks if states[t['task_id']] == 'COMPLETE']
    total_seconds = [read(root / 'training' / t['task_id'] / 'result.json')['worker_seconds'] for t in completed]
    remaining = states.values()
    value = {'states': states, 'completed': len(completed), 'failed': sum(s == 'FAILED' for s in states.values()),
        'pending': sum(s == 'PENDING' for s in states.values()), 'current': current,
        'initializations': len(jsonlines(root / 'initializations.jsonl')), 'worker_seconds': sum(total_seconds),
        'ETA_seconds': sum(total_seconds) / len(total_seconds) * sum(s == 'PENDING' for s in states.values()) if total_seconds else None,
        'time': now()}
    write(root / 'progress.json', value)
    status = read(root / 'status.json')
    for phase in ['D1', 'D2']:
        ss = [states[t['task_id']] for t in tasks if t['phase'] == phase]
        status[phase] = 'COMPLETE' if all(s == 'COMPLETE' for s in ss) else ('PARTIAL' if all(s != 'PENDING' for s in ss) else 'RUNNING')
    status['updated_at'] = now()
    write(root / 'status.json', status)
    return value

def run(root, repo, stop_after=None, exercise=False):
    verify_run(root, repo)
    tasks = read(root / 'task_manifest.json')['tasks']
    if (root / 'controller_pause.json').exists():
        verify(root, read(root / 'controller_pause.json')['files'])
    started = time.perf_counter()
    log(root / 'controller_sessions.jsonl', {'time': now(), 'event': 'start', 'stop_after': stop_after})
    for task in tasks:
        state = task_status(root, task)
        if state in ['COMPLETE', 'FAILED']:
            continue
        while select_device(root) is None:
            write(root / 'resource_wait.json', {'state': 'WAITING_RESOURCE', 'time': now()})
            time.sleep(15)
        folder = root / 'training' / task['task_id']
        folder.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-B', '-m', 'analyse.rank_reversal_diagnostic', '_worker',
               '--run-dir', str(root), '--task-id', task['task_id']]
        if exercise and task == tasks[0] and not (folder / 'latest.pt').exists():
            cmd += ['--pause-after', '5']
        while True:
            log(root / 'worker_commands.jsonl', {'time': now(), 'command': shlex.join(cmd), 'task_id': task['task_id']})
            update_progress(root, tasks, task['task_id'])
            with (folder / 'worker.log').open('a') as f:
                proc = subprocess.run(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT)
            if proc.returncode == 75:
                assert (folder / 'planned_pause.json').exists()
                cmd = [x for x in cmd]  # next process resumes exact complete state
                cmd = cmd[:cmd.index('--pause-after')] if '--pause-after' in cmd else cmd
                log(root / 'controller_sessions.jsonl', {'event': 'worker_interrupted_at_saved_epoch', 'time': now(), 'task_id': task['task_id']})
                continue
            if proc.returncode:
                write(folder / 'failure.json', {'time': now(), 'returncode': proc.returncode,
                    'random_initializations': sum(x['task_id'] == task['task_id'] for x in jsonlines(root / 'initializations.jsonl')),
                    'retry': 'none; budget and failed sample preserved', 'worker_log': str(folder / 'worker.log')})
            break
        p = update_progress(root, tasks)
        print(json.dumps({k: p[k] for k in ['completed', 'failed', 'pending', 'initializations', 'ETA_seconds']}), flush=True)
        if all(task_status(root, t) != 'PENDING' for t in tasks if t['phase'] == 'D1'):
            seal_d1(root, tasks)
        if stop_after is not None and p['completed'] >= stop_after:
            write(root / 'controller_pause.json', {'time': now(), 'files': core_artifacts(root, tasks),
                'completed': p['completed'], 'initializations': p['initializations'], 'reason': 'real controller resume acceptance'})
            log(root / 'controller_sessions.jsonl', {'event': 'planned_controller_stop', 'time': now(), 'wall_seconds': time.perf_counter() - started})
            return
    seal_d1(root, tasks)
    p = update_progress(root, tasks)
    log(root / 'controller_sessions.jsonl', {'event': 'complete', 'time': now(), 'wall_seconds': time.perf_counter() - started})
    write(root / 'training_complete.json', {'completed': p['completed'], 'failed': p['failed'],
        'not_executed': p['pending'], 'initializations': p['initializations'], 'time': now()})

def check_resume(root):
    tasks = read(root / 'task_manifest.json')['tasks']
    first = root / 'training' / tasks[0]['task_id']
    pause = read(first / 'planned_pause.json')
    result = read(first / 'result.json')
    assert digest(result['trajectory'][:pause['epoch']]) == pause['trajectory_hash']
    events = jsonlines(first / 'epoch_events.jsonl')
    assert [e['epoch'] for e in events] == list(range(1, result['stopped_epoch'] + 1))
    attempts = jsonlines(first / 'attempts.jsonl')
    assert len(attempts) == 2 and attempts[1]['resume_epoch'] == pause['epoch']
    assert sum(a['random_initialization'] for a in attempts) == 1
    cp = read(root / 'controller_pause.json')
    verify(root, cp['files'])
    value = {'state': 'PASS', 'time': now(), 'real_worker_process_interruption_epoch': pause['epoch'],
        'all_state_restored': attempts[1]['restored_fields'], 'prefix_trajectory_unchanged': True,
        'repeated_completed_epochs': 0, 'first_task_initializations': 1,
        'real_controller_stop_after_completed': cp['completed'], 'all_completed_artifact_hashes_unchanged': True,
        'no_completed_task_retraining': all(sum(x['random_initialization'] for x in jsonlines(root / 'training' / t['task_id'] / 'attempts.jsonl')) == 1
                                           for t in tasks if task_status(root, t) == 'COMPLETE')}
    write(root / 'resume_verification.json', value)
    return value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['preflight', 'run', '_worker', 'score', 'analyze', 'verify-resume'])
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--prompt')
    ap.add_argument('--task-id')
    ap.add_argument('--pause-after', type=int)
    ap.add_argument('--stop-after', type=int)
    ap.add_argument('--exercise-interruption', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    root = Path(args.run_dir).resolve()
    repo = Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    log(root / 'actual_commands.jsonl', {'time': now(), 'cwd': str(repo),
        'command': shlex.join([sys.executable, '-B', '-m', 'analyse.rank_reversal_diagnostic', *sys.argv[1:]])})
    if args.command == '_worker':
        from .train import worker, PlannedPause
        try:
            worker(root, repo, args.task_id, args.pause_after)
        except PlannedPause:
            sys.exit(75)
        return
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'preflight':
            from .audit import preflight
            preflight(root, repo, args.prompt)
        elif args.command == 'run':
            run(root, repo, args.stop_after, args.exercise_interruption)
        elif args.command == 'score':
            from .score import score
            score(root, repo)
            s = read(root / 'status.json'); s['D4'] = 'COMPLETE'; write(root / 'status.json', s)
        elif args.command == 'analyze':
            verify_run(root, repo)
            from .stats import analyze
            analyze(root)
        elif args.command == 'verify-resume':
            verify_run(root, repo)
            print(json.dumps(check_resume(root)), flush=True)

